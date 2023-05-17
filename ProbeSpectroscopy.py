import numpy as np
import time
from numbers import Number
from scipy import linalg
from scipy.signal import invres,unique_roots
from common.log import Logger
from common import plotting
from common import numerical_recipes as numrec
from common.baseclasses import AWA
import ProbeCavityEigenfields as PCE

#--- Utilities

def get_superset_indices(subset,superset):

    subset=np.array(subset); superset=np.array(superset)
    assert len(set(subset))==len(subset), 'All elements of `subset` must be unique!'
    assert len(set(superset))==len(superset), 'All elements of `superset` must be unique!'
    assert len(set(subset).intersection(superset))==len(subset), '`subset` must be entirely within `superset`!'

    #-- It's still possible that superset contains an element of subset more than once..
    superset_inds,subset_inds = np.argwhere( superset[:, np.newaxis] \
                                             == subset[np.newaxis, :] ).T

    # This makes sure indices are ascending with the ordering of subset
    subset_inds,superset_inds = zip(*sorted(zip(subset_inds,superset_inds)))

    return np.array(superset_inds)

#--- Probe Spectroscopy

class ProbeSpectroscopy(object):

    def __init__(self,P):

        # Store details of the Probe to restore later
        assert isinstance(P,PCE.Probe)
        self._P=P

        self._recorded_eigenrhos = {}
        self._recorded_eigencharges = {}
        self._recorded_self_impedances = {}
        self._recorded_mirror_impedances = {}

        # lists to be populated during sorting
        self.eigenrhos = None
        self.eigencharges = None

    def record(self, x):
        """Record probe self impedance and eigenset at coordinate `x`,
        as well as the Probe itself - this is all that's necessary to
        restore response functions at `x`."""

        self._recorded_eigenrhos[x] = self._P.get_eigenrhos()
        self._recorded_eigencharges[x] = self._P.get_eigencharges()

        Zself = self._P.get_self_impedance(recompute=False)
        self._recorded_self_impedances[x] = Zself

        Zmirror = self._P.get_mirror_impedance(recompute=False)
        self._recorded_mirror_impedances[x] = Zmirror

    def get_probe(self): return self._P

    def get_coords(self):
        return np.array(sorted(self._recorded_eigenrhos.keys()))

    def get_probe_at_coord(self, coord,
                           update_charges=True, update_Zself=True,
                           verbose=True):

        coords = self.get_coords()
        best_coord = coords[np.argmin(np.abs(coord - coords))]
        if verbose: Logger.write('Updating eigenrhos to coordinate %1.2E...' % best_coord)
        rhos = self._recorded_eigenrhos[best_coord]
        self._P._eigenrhos = rhos
        if update_charges:
            if verbose: Logger.write('\tUpdating eigencharges...')
            charges = self._recorded_eigencharges[best_coord]
            self._P._eigencharges = charges
        if update_Zself:
            if verbose: Logger.write('\tUpdating self impedance...')
            ZSelf = self._recorded_self_impedances[best_coord]
            self._P._ZSelf = ZSelf

        return self.get_probe()

    def distance(self, xm, xn, m, n, by_eigenrho=True):
        """Get 'distance' between eigencharge at coordinate `xm`, index `m`,
        and that at coordinate `xn`, index `n`."""

        if by_eigenrho:
            rho_m = self._recorded_eigenrhos[xm][m]
            rho_n = self._recorded_eigenrhos[xn][n]
            return np.abs(rho_m-rho_n)

        Qm = self._recorded_eigencharges[xm][m]
        Qn = self._recorded_eigencharges[xn][n]

        Qm = np.matrix(Qm).T  # column vector
        Qn = np.matrix(Qn).T  # column vector

        #--- Get overlap operator at "midpoint"
        # `O = (Om + On)/2`
        # `Om = -PhiM = PhiS / rho[m]` whenever operating on `Qm`
        # because `(rho[m] Zm + Zs) Qm = 0`, so `Qm * Zs/rho[m] * Qm` is maximized at `~1`
        Om = self._recorded_self_impedances[xm] / self._recorded_eigenrhos[xm][m]
        On = self._recorded_self_impedances[xn] / self._recorded_eigenrhos[xn][n]
        #Om = self._recorded_mirror_impedances[xm]
        #On = self._recorded_mirror_impedances[xn]
        Oavg = (Om + On)/2
        #WPhiRWm = self.overlap_operators[xm]
        #WPhiRWn = self.overlap_operators[xn]

        overlap = Qn.T @ Oavg @ Qm

        # -- This is the standard fall-back for distance measure
        distance = 1 / np.abs(np.complex(overlap))

        # -- This one's a no-brainer, and workable
        # distance=np.min( (np.sum(np.abs(v1-v2)),\
        #                  np.sum(np.abs(v1+v2))) )

        # -- This one seems to work comparable to the standard, maybe better?
        # rho1=self.eigenrhos[xm][m]
        # rho2=self.eigenrhos[xn][n]
        # distr2 = (np.log(rho1.real)-np.log(rho2.real))**2
        # disti2 = (np.log(-rho1.imag)-np.log(-rho2.imag))**2
        # distance=np.sqrt(distr2+disti2)

        return distance

    def classify_eigensets(self, Nmodes=10, reversed=False, **kwargs):
        # This is the robust way of labeling eigensets by an eigenindex
        # in a way that is uniform across the internal coordinates
        # `reversed` is untested

        if reversed: print('Classifying eigensets by eigenindex, with reversal...')
        else: print('Classifying eigensets by eigenindex...')
        coords = sorted(list(self._recorded_eigenrhos.keys()))
        if reversed: coords = coords[::-1]

        # --- Assign labels to modes
        connectivity = []
        for coordind in range(len(coords)):
            print('Measuring distances at coordinate index %i of %i...'%(coordind,len(coords)))

            coord = coords[coordind]
            N = np.min((Nmodes, len(self._recorded_eigenrhos[coord])))

            if coordind == 0:
                connections = list(range(N))

            else:
                coord_prev = coords[coordind - 1]
                M = np.min((Nmodes, len(self._recorded_eigenrhos[coord_prev])))

                connections = []
                for n in range(N):
                    print('Measuring distances to eigenset %i of %i...'%(n,N))
                    ds = [self.distance(coord_prev, coord, m, n,
                                        **kwargs) for m in range(M)]
                    m = np.argmin(ds)
                    connections.append(connectivity[coordind - 1][m])

            connectivity.append(connections)

        self.connectivity = connectivity

        # --- Sort modes based on their labels
        all_rhos = []
        all_charges = []
        connection_index = 0
        while True:
            rhos = [];
            charges = [];
            assoc_coords = []
            for coordind, coord in enumerate(coords):
                connections = connectivity[coordind]
                try:
                    m = connections.index(connection_index)
                    rhos.append(self._recorded_eigenrhos[coord][m])
                    charges.append(self._recorded_eigencharges[coord][m])
                    assoc_coords.append(coord)
                except ValueError:
                    continue
            if not rhos: break  # Nothing existed with this label, move on
            all_rhos.append(AWA(rhos, axes=[assoc_coords], axis_names=['coord']))
            all_charges.append(AWA(charges, axes=[assoc_coords,\
                                                  charges[0].axes[0]],\
                                   axis_names=['coord', 'z']))
            connection_index += 1

        self.sorted_eigenrhos = all_rhos
        self.sorted_eigencharges = all_charges

    def __call__(self,*args,**kwargs): return self.classify_eigensets(*args,**kwargs)

    def get_eigenrhos_AWA(self,Nmodes=10,recompute=False,verbose=True):

        try:
            precomputed = self._eigenrhos_AWA
            if Nmodes != len(precomputed) or recompute: #If we need more modes than we have, also recompute
                raise AttributeError
            return precomputed
        except AttributeError: pass #proceed to evaluate

        if self.sorted_eigenrhos is None:
            raise RuntimeError('Call the spectroscopy instance first (i.e. `Spectroscopy()` ) to sort eigensets!')

        #--- Find coordinates that are common to all eigensets
        #  since some eigensets may have been dropped during the sorting process
        coords = [rhos.axes[0] for rhos in self.sorted_eigenrhos[:Nmodes]]
        Nmodes=len(coords)
        coords_common = sorted(list(set(coords[0]).intersection(*coords[1:])))
        if verbose:
            Logger.write('For Nmodes=%i, there were %i identifiable mutual coordinates.'%(Nmodes,len(coords_common)))

        ind_sets = [ get_superset_indices(subset=coords_common,\
                                         superset=coord_set) \
                    for coord_set in coords ]

        eigenrhos_grid = [rhos[ind_set] for rhos,ind_set \
                          in zip(self.sorted_eigenrhos[:Nmodes],\
                                 ind_sets)]

        self._eigenrhos_AWA = AWA(eigenrhos_grid,axes=[None,coords_common],\
                                  axis_names=['eigenindex','coordinate'])

        return self._eigenrhos_AWA

    def align_eigencharge_signs(self,eigencharges, coords, eigenindex):
        """Align a set of `eigencharges` all at `eigenindex` but varying across `coords`
        so that they do not flip in sign by -1."""

        aligned_eigencharges = [ eigencharges[0] ]
        for i in range(len(coords)-1):

            xm = coords[i]
            xn = coords[i+1]
            Qm = np.matrix(aligned_eigencharges[-1]).T
            Qn = np.matrix(eigencharges[i+1]).T
            charge_to_add = eigencharges[i+1]
            # --- Get overlap operator at "midpoint" between `i,i+1`
            # `O = (Om + On)/2`
            # `Om = -PhiM = PhiS / rho[m]` whenever operating on `Qm`
            # because `(rho[m] Zm + Zs) Qm = 0`, so `Qm * Zs/rho[m] * Qm` is maximized at `~1`
            Om = self._recorded_self_impedances[xm] / self._recorded_eigenrhos[xm][eigenindex]
            On = self._recorded_self_impedances[xn] / self._recorded_eigenrhos[xn][eigenindex]
            Oavg = (Om + On) / 2

            # If overlap is negative, the added charge needs to be flipped in sign
            overlap = complex( Qm.T @ Oavg @ Qn )
            if overlap.real<0: charge_to_add = -charge_to_add

            aligned_eigencharges.append(charge_to_add)

        return aligned_eigencharges

    def get_eigencharges_AWA(self,Nmodes=10,recompute=False,
                             verbose=True):

        try:
            precomputed = self._eigencharges_AWA
            if Nmodes != len(precomputed) or recompute: #If we need more modes than we have, also recompute
                raise AttributeError
            return precomputed
        except AttributeError: pass #proceed to evaluate

        if self.sorted_eigenrhos is None:
            raise RuntimeError('Call the spectroscopy instance first (i.e. `Spectroscopy()` ) to sort eigensets!')

        #--- Find coordinates that are common to all eigensets
        #  since some eigensets may have been dropped during the sorting process
        coords = [ rhos.axes[0] for rhos in self.sorted_eigenrhos[:Nmodes] ]
        Nmodes = len(coords)
        coords_common = sorted(list(set(coords[0]).intersection(*coords[1:])))
        if verbose:
            Logger.write('Obtaining eigencharges for Nmodes=%i, across  %i mutual spectroscopy coordinates.'%(Nmodes,len(coords_common)))

        ind_sets = [ get_superset_indices(subset=coords_common,\
                                         superset=coord_set) \
                    for coord_set in coords ]

        eigencharges_grid = [ charges[ind_set] for charges, ind_set \
                              in zip(self.sorted_eigencharges[:Nmodes], \
                                     ind_sets) ]

        if verbose: Logger.write('\tAligning eigencharge signage...')
        eigencharges_grid = [ self.align_eigencharge_signs(eigencharges,\
                                                           coords=coords_common,\
                                                           eigenindex=eigenindex) \
                              for eigenindex,eigencharges in enumerate(eigencharges_grid) ]

        self._eigencharges_AWA = AWA(eigencharges_grid,axes=[None,coords_common,\
                                                            eigencharges_grid[0][0].axes[0]],\
                                     axis_names=['eigenindex','coordinate',\
                                                 eigencharges_grid[0][0].axis_names[0]])

        return self._eigencharges_AWA #Axes are eigenindex, coordinate, z-on-probe

    def get_brightnesses_AWA(self,Nmodes=None,recompute=False,
                             recompute_eigencharges=False,
                             eigencharges_AWA=None,
                             angles=np.linspace(10,80,20),
                             verbose=True,**kwargs):

        try:
            precomputed = self._brightnesses_AWA
            if (Nmodes and Nmodes != len(precomputed)) or recompute: #If we need more modes than we have, also recompute
                raise AttributeError
            return precomputed
        except AttributeError:
            import traceback
            pass #proceed to evaluate

        if eigencharges_AWA is None:
            eigencharges_AWA = self.get_eigencharges_AWA(Nmodes,
                                                         recompute=recompute_eigencharges,
                                                         verbose=verbose)
        coords=eigencharges_AWA.axes[1]
        if Nmodes:
            eigencharges_AWA = eigencharges_AWA[:Nmodes]

        if verbose: Logger.write('Computing brightnesses across %i spectroscopy coordinates...'%len(coords))

        all_brightnesses=[]
        for coord in coords:
            eigencharges = eigencharges_AWA.cslice[:,coord]
            P = self.get_probe_at_coord(coord,verbose=False)
            brightnesses = [P.get_brightness(charge,angles=angles,**kwargs) \
                            for charge in eigencharges]
            all_brightnesses.append(brightnesses)

        self._brightnesses_AWA = AWA(all_brightnesses,
                                          axes=[coords,None],
                                          axis_names=['coordinate','eigenindex']).T #transpose to conform with previous convention

        return self._brightnesses_AWA


    def plot_eigenrhos_scatter(self,Nmodes=10,versus_coord=False,**kwargs):

        from matplotlib import pyplot as plt
        from matplotlib import cm,colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        coords=self.get_coords()
        cs = plotting.bluered_colors(len(coords))
        for coord in coords:
            rhos = np.array(self._recorded_eigenrhos[coord][:Nmodes])
            c = next(cs)
            if versus_coord:
                plt.plot([coord]*len(rhos),rhos.real,color=c,marker='o',ls='',**kwargs)
                plt.plot([coord]*len(rhos),rhos.imag,color=c,marker='s',ls='',**kwargs)
            else: plt.plot(rhos.real, rhos.imag, color=c, marker='o', ls='',**kwargs)

        plt.axhline(0,ls='--',color='k',alpha=.5)

        plt.gca().set_yscale('symlog')
        plt.gca().set_xscale('log')
        plt.xlabel(r'Re $\rho_i$')
        plt.ylabel(r'Im $\rho_i$')

        cmap = plt.get_cmap('BlueRed')
        norm = colors.Normalize(vmin=coords.min(), vmax=coords.max())

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%',pad=.1)
        plt.gcf().colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cax, orientation='vertical',label='coordinate')

    def plot_eigenrhos(self,Nmodes=10):

        from matplotlib import pyplot as plt
        from matplotlib import cm,colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        eigenrhos_AWA = self.get_eigenrhos_AWA(Nmodes=Nmodes)

        cs = plotting.bluered_colors(len(eigenrhos_AWA))
        for rhos in eigenrhos_AWA:
            c = next(cs)
            rhos.real.plot(color=c)
            rhos.imag.plot(color=c,ls='--')

        plt.axhline(0,ls='--',color='k',alpha=.5)

        plt.gca().set_yscale('symlog')
        plt.gca().set_xscale('log')
        plt.xlabel('coordinate')
        plt.ylabel(r'Im $\rho_i$      Re $\rho_i$')
        yl=np.max(np.abs(eigenrhos_AWA))
        plt.ylim(-yl,yl)

        cmap = plt.get_cmap('BlueRed')
        norm = colors.Normalize(vmin=1, vmax=len(eigenrhos_AWA) )

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%',pad=.1)
        plt.gcf().colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cax, orientation='vertical')
        cax.set_title(label=r'$i$')

#--- Parallel spectroscopy

def compute_eigenset_at_freq(P, freq, PhiM=False, **kwargs):

        print('Starting freq=%1.2g' % freq)
        t0 = time.time()
        P.set_freq(freq)
        Zself = P.get_self_impedance(k=P.get_k(), recompute=True, display=False)

        if PhiM:
            # Zmirror = P.get_mirror_impedance(k=P.get_k(),recompute=True,\
            #                               sommerfeld=True,rp=None,Nkappas=244*4);
            # Zmirror = P.get_mirror_impedance(k=P.get_k(),nonsingular=False,recompute=True,display=False)
            Zmirror = P.get_mirror_impedance(k=P.get_k(), farfield=False, \
                                             recompute=True, sommerfeld=True, Nkappas=244 * 8)

        rhos, Qs = P.solve_eigenmodes(condition_ZM=True, condition_ZS=False, ZMthresh=0, recompute_impedance=False)
        print('Finished freq=%1.2g, elapsed time: %1.2g s' % (freq, time.time() - t0))

        return rhos, Qs, Zself

def compute_eigenset_at_gap(P, gap=1, k=0, sommerfeld=True, **kwargs):

        print('Starting gap=%1.2g' % gap)
        t0 = time.time()
        # Zself = P.get_self_impedance(k=P.get_k(),recompute=True,display=False)

        P.set_gap(gap)
        if k is None: k=P.get_k()

        # Use sommerfeld calculation because it's faster; q-grid is adaptive
        if sommerfeld and 'kappa_max' not in kwargs:
            kwargs['kappa_max'] = 10*np.max( (1/gap, 1/P.get_a()) )
        Zmirror = P.get_mirror_impedance(k=k, recompute=True, sommerfeld=sommerfeld, **kwargs)

        rhos, Qs = P.solve_eigenmodes(condition_ZM=True, condition_ZS=False, ZMthresh=0, recompute_impedance=False)
        print('Finished gap=%1.2g, elapsed time: %1.2g s' % (gap, time.time() - t0))

        Zself = P.get_self_impedance(recompute=False, display=False)

        return rhos, Qs, Zself

class ProbeSpectroscopyParallel(ProbeSpectroscopy):

    class serializableProbe(object):
        """A context manager for adding / removing attributes of a Probe
        that are unnecessary to serialize in parallel computation."""

        def __init__(self, P):
            self.removed_attrs = {}
            self.attrs = ['_gapSpectroscopy']
            self.P = P

        def __enter__(self):
            # --- Remove (temporarily) attributes that slow serialization of the Probe
            for attr in self.attrs:
                if hasattr(self.P, attr):
                    self.removed_attrs[attr] = getattr(self.P, attr)
                    delattr(self.P, attr)

        def __exit__(self,type, value, traceback): #last arguments are obligator
            # --- Restore any removed Probe attributes
            for attr in self.removed_attrs:
                setattr(self.P, attr,
                        self.removed_attrs[attr])

    def __init__(self,P, coords, eigenset_calculator=compute_eigenset_at_gap,\
                 ncpus=8, backend='multiprocessing', Nmodes=20, reversed=False, **kwargs):
        from joblib import Parallel, delayed, parallel_backend

        super().__init__(P)

        #--- Make sure probe impedances have already been computed
        # We don't know which ones will be recomputed across parallel iterations,
        # but we can make sure the complement is available
        ZSelf = P.get_self_impedance(k=P.get_k(), recompute=False)
        ZMirror = P.get_mirror_impedance(k=0, recompute=False)

        t0 = time.time()
        eigensets = []
        with self.serializableProbe(P) as context:
            with Parallel(n_jobs=ncpus,backend=backend) as parallel:
                while len(eigensets) < len(coords):
                    #--- Delegate group of coords to a single parallel job
                    nstart = len(eigensets);
                    nstop = len(eigensets) + ncpus
                    coords_sub = coords[nstart:nstop]
                    new_eigensets = parallel(delayed(eigenset_calculator)(P, coord, **kwargs) \
                                                                          for coord in coords_sub)
                    print('got results!')
                    eigensets = eigensets + new_eigensets
                print('Time elapsed: ', time.time() - t0)

        for coord, eigenset in zip(coords, eigensets):
            rhos, charges, ZSelf = eigenset
            P._eigenrhos = rhos
            P._eigencharges = charges
            P._ZSelf = ZSelf
            self.record(coord)

        # Run classification by eigenindex
        self.classify_eigensets(Nmodes=Nmodes,reversed=reversed)

def get_cheb_zs_weights(Ncheb=8, A=2, gapmin=.1, demod_order=5):

    from numpy.polynomial import chebyshev

    xs_ev = np.cos(
        np.pi * (np.arange(Ncheb) + 1 / 2) / Ncheb)  # Gauss-Chebyshev zeros of order `Ncheb` polynomial
    zs_ev = A * (1 + xs_ev) + gapmin

    weights = {}
    for harmonic in range(demod_order):
        c = [0] * (harmonic + 1);
        c[-1] = 1
        # (-1) bit is to make polynomial positive the most negative point (in-contact)
        weights[harmonic] = 2 / Ncheb * chebyshev.Chebyshev(c)(xs_ev) * (-1) ** harmonic

    return zs_ev, weights

class ProbeFrequencySpectroscopyParallel(ProbeSpectroscopyParallel):

    def __init__(self,P, freqs,
                 ncpus=8, backend='multiprocessing',
                 Nmodes=20, reversed=False,
                 **kwargs):

        #--- Make sure mirror impedance has been calculated
        Zmirror = P.get_mirror_impedance(k=0, recompute=True, sommerfeld=False)

        #--- Hard-code the frequency calculator
        super().__init__(P, coords=freqs, eigenset_calculator=compute_eigenset_at_freq,
                         ncpus=ncpus, backend=backend, Nmodes=Nmodes, reversed=reversed,
                         PhiM=False, **kwargs)

class ProbeGapSpectroscopyParallel(ProbeSpectroscopyParallel):

    def __init__(self,P, gaps=np.logspace(-1.5,1,100),\
                 ncpus=8, backend='multiprocessing',\
                 Nmodes=20, sommerfeld=True, reversed=False,
                 **kwargs):

        #--- Make sure self impedance has been calculated

        #--- Hard-code the gap calculator
        kwargs['k'] = 0 #we insist on a quasistatic calculation, otherwise some features of this class become meaningless
        super().__init__(P, coords=gaps, eigenset_calculator=compute_eigenset_at_gap,
                         ncpus=ncpus, backend=backend, Nmodes=Nmodes, reversed=reversed,
                         sommerfeld=sommerfeld, **kwargs)

class EncodedEigenfields(object):
    """This function condensed a `ProbeSpectroscopy` object
    down to an encoded version that can duck-type in some functionalities
    as a `Probe` object.

    Particularly useful to condense a `ProbeGapSpectroscopy` so that
    demodulated near-field scattering signals can be computed easily.

    Wrapping other probe spectroscopy types is not yet supported (or useful?)."""

    def __init__(self, Spec,gap0=1, Nmodes=10,
                kappa_min=None, kappa_max=np.inf,
                Nkappas=244*8, qquadrature=PCE.numrec.GL,
                **brightnesskwargs):
        # This will be the one and only time we have to explicitly calculate eigenfields
        # for any coordinate besides the reference coordinate; ideally all forthcoming
        # computations of reflectance etc. will need only be done for the reference coordinate!

        #assert isinstance(Spec,ProbeSpectroscopy)

        #--- Get reference probe and reference brightnesses
        eigencharges_vs_gap = Spec.get_eigencharges_AWA(Nmodes=Nmodes,recompute=False)
        gaps = eigencharges_vs_gap.axes[1] # axes are eigenindex, coordinate, zprobe
        Logger.write('Encoding eigenfields to gap=%1.2g across %i gap values from gap=%1.2g to %1.2g...' \
                     % ( gap0,len(gaps),gaps.min(),gaps.max() ) )
        ind0 = np.argmin( (gaps-gap0)**2 )
        gap0 = gaps[ind0]
        P0 = Spec.get_probe_at_coord(gap0)

        # we take k=0 propagators to conform with the quasistatic context for the eigenfields
        dP = P0.getFourPotentialPropagators(farfield=False, k=0,
                                           kappa_min=kappa_min, kappa_max=kappa_max, Nkappas=Nkappas,
                                           recompute=True, qquadrature=qquadrature)
        kappas = dP['kappas']
        dkappas = dP['dkappas']; dKappas = np.matrix(np.diag(dkappas))

        #--- Compute eigenfields and their brightnesses at all gaps
        PhisVsGap=[] # axes will be `gap,eigenindex,q`
        for gap in gaps:
            Pgap = np.matrix(np.diag(np.exp(-gap*kappas)))
            ChargeMat = np.matrix( eigencharges_vs_gap.cslice[:,gap,:] ).T #column vectors of eigencharges
            PhiMat = Pgap @ dP['Phi'] @ ChargeMat #This is already the Fourier transform multiplied by q
            PhisVsGap.append( np.array(PhiMat).T ) #append row vectors, axes are `eigenindex,q`
        Phis0 = PhisVsGap[ind0]

        #--- Evaluate encoding matrix at each gap
        Phi0Vecs = np.matrix(Phis0).T
        PsiMats = [] # list will run over gaps
        for i,gap in enumerate(gaps):
            PhiMat = np.matrix(PhisVsGap[i]).T #eigenfields at `gap` as column vectors
            # matrix elements for `PsiMat` are `(phi_n(z0)|q|phi_m(z))`
            # (where bra-ket already has `dq q^2 = `dq q` measure times `q` orthogonality operand)
            PsiMat = Phi0Vecs.T @ dKappas @ PhiMat
            PsiMats.append(PsiMat)

        #--- Package results
        Brightnesses = Spec.get_brightnesses_AWA(Nmodes=Nmodes, recompute=False,
                                                 eigencharges_AWA = eigencharges_vs_gap,
                                                 **brightnesskwargs)[:Nmodes].T #Put coordinate (gaps) axis first
        Poles = Spec.get_eigenrhos_AWA(Nmodes=Nmodes, verbose=False).T #Put coordinate (gaps) axis first
        PsiMats = AWA(PsiMats, axes=[gaps, None,None],
                      axis_names=['gap', 'eigenindex n','eigenindex m'])

        self.kappas = kappas
        self.dkappas = dkappas
        self.Phis0 = Phis0
        self.Phi0Vecs = np.matrix(Phis0).T #column vectors of scalar potential vs. kappa
        self.Brightnesses = Brightnesses
        self.Poles = Poles
        self.PsiMats_ = PsiMats #underscore because method will share same name
        self._do_cache=False

        #--- Attach probe with the correctly signed eigencharges
        self.Probe=P0
        self.Probe._eigencharges = eigencharges_vs_gap.cslice[:,gap0,:]

    def cache(self,do_cache=True): self._do_cache = do_cache

    def BVecs(self, at_coords, extrapolate=True):

        try:
            if not self._do_cache: raise AttributeError
            return self._BVecs
        except AttributeError: pass # proceed to compute

        brightnesses_at_coords = self.Brightnesses.interpolate_axis(at_coords, axis=0,
                                                                  extrapolate=extrapolate,
                                                                  bounds_error=~extrapolate,
                                                                  kind='quadratic')
        self._BVecs = [np.matrix(brightnesses_at_coord).T for
                       brightnesses_at_coord in brightnesses_at_coords] #column vectors

        return self._BVecs

    def PsiMats(self, at_coords, extrapolate=True):

        try:
            if not self._do_cache: raise AttributeError
            return self._PsiMats
        except AttributeError: pass # proceed to compute

        PsiMats_at_coords = self.PsiMats_.interpolate_axis(at_coords, axis=0,
                                                          extrapolate=extrapolate,
                                                          bounds_error=~extrapolate,
                                                          kind='quadratic')
        self._PsiMats = [np.matrix(PsiMat_at_coord) for
                         PsiMat_at_coord in PsiMats_at_coords] #column vectors

        return self._PsiMats

    def PoleMats(self, at_coords, extrapolate=True):

        try:
            if not self._do_cache: raise AttributeError
            return self._PoleMats
        except AttributeError: pass # proceed to compute

        Poles_at_coords = self.Poles.interpolate_axis(at_coords, axis=0,
                                                          extrapolate=extrapolate,
                                                          bounds_error=~extrapolate,
                                                          kind='quadratic')
        self._PoleMats = [np.matrix(np.diag(Poles_at_coord)) for
                         Poles_at_coord in Poles_at_coords] #column vectors

        return self._PoleMats

    def EradVsGap(self, at_gaps, freq, RMat0=None, rp=None, **kwargs):

        kappas = self.kappas
        dkappas = self.dkappas
        Phi0Vecs = self.Phi0Vecs #column vectors of scalar field vs kappa
        BVecs = self.BVecs(at_gaps)
        PsiMats = self.PsiMats(at_gaps)
        PoleMats = self.PoleMats(at_gaps)

        #--- Get RMat for fields at reference gap
        if RMat0 is None:
            k = 2 * np.pi * freq
            qs = np.sqrt(k ** 2 + kappas ** 2).real #This will keep us out of the light cone
            if isinstance(rp,Number): rp_vs_q=rp
            else:
                rp_vs_q = rp(freq, qs, **kwargs) #rp will have to anticipate dimensionless arguments
                rp_vs_q[np.isnan(rp_vs_q)+np.isinf(rp_vs_q)] = 0  # This is just probably trying to evaluate at too large a momentum
            RMat0 = Phi0Vecs.T @ np.diag(dkappas * rp_vs_q) @ Phi0Vecs
        else:
            assert RMat0.shape == PsiMats[0].shape,\
                '`RMat` must be of shape `Nmodes x Nmodes`!'

        #--- Execute the loop over gaps values
        Erads = []
        for igap in range(len(at_gaps)):
            BVec=BVecs[igap]
            PsiMat = PsiMats[igap]
            PoleMat=PoleMats[igap]

            RMat = PsiMat.T @ RMat0 @ PsiMat
            """Erad = BVec.T @ \
                      ( (RMat - PoleMat).I + (PoleMat).I ) \
                        @ BVec  # scattering/summing etc. is implicit"""
            Erad = BVec.T @ \
                   ( linalg.solve(RMat - PoleMat, BVec)
                    + linalg.solve(PoleMat, BVec) ) # Same as matrix inverse; should boost performance in longer runs
            # scattering/summing etc. is implicit

            Erads.append( complex(Erad) )

        return AWA(Erads, axes=[at_gaps], axis_names=['gap'])

    def EradSpectrumDemodulated(self, freqs, rp=None,
                                gapmin=.1, amplitude=2,
                                Ngaps=24, demod_order=5,
                                **kwargs):
        T = PCE.Timer()

        #gaps, weights = get_cheb_zs_weights(Ncheb=Ngaps, A=amplitude, gapmin=gapmin, demod_order=demod_order)
        at_gaps,dgaps=numrec.GetQuadrature(xmin=gapmin,xmax=gapmin+2*amplitude,
                                         N=Ngaps,quadrature=numrec.CC)

        #--- Turn on caching for the duration of all frequencies, since gaps will be the same
        original_caching = self._do_cache
        self.cache(False)

        #--- Execute the loop over frequency values
        EradsVsFreq = []
        if not hasattr(freqs, '__len__'): freqs = [freqs]
        for freq in freqs:
            print('\tComputing at freq=%1.3E...'%freq)
            dErad = self.EradVsGap(at_gaps, freq, rp=rp, **kwargs)
            EradsVsFreq.append(dErad)

            # --- Turn on caching for the duration of all frequencies, since gaps will be the same
            self.cache(True)

        gaps = EradsVsFreq[0].axes[0]
        EradsVsFreq = AWA(EradsVsFreq, axes=[freqs, gaps],
                          axis_names=['Frequency', r'$z_\mathrm{tip}$']).T
        result = dict(Erad=EradsVsFreq)

        # --- Demodulate with chebyshev polynomials
        if demod_order:
            Logger.write('Demodulating...')
            sns = PCE.demodulate(EradsVsFreq, demod_order=demod_order,Nts=Ngaps)
            result['Sn'] = sns

        """sns=[]
        for harmonic in range(demod_order):
            weights = np.cos(harmonic * 2*np.pi * np.linspace(0,.5,len(at_gaps)))
            Sn = np.sum(weights[:,np.newaxis] * EradsVsFreq, axis=0)  # sum on z-axis
            sns.append(Sn)
        result['Sn'] = AWA(sns,axes=[None,freqs],\
                           axis_names=['harmonic','Frequency'])"""

        Logger.write('\t' + T())

        #--- Restore original caching (perhaps disabling it)
        self.cache(original_caching)

        return result

    def getNormalizedSignal(self,freqs_wn,rp,
                            a_nm=30,amplitude_nm=50,demod_order=5,
                            Ngaps=24*4,gapmin=.15,
                            rp_norm = None,
                            freqs_wn_norm = None):

        # Wrap the provided rp functions so they can expand out dimensionless frequencies and wavevectors
        # The supplied reflection function should take `frequency (wavenumbers), q (wavenumbers)`
        rp_wrapped = lambda freq,q: rp( freq / (a_nm * 1e-7),
                                        q / (a_nm * 1e-7))
        if rp_norm is None:
            from NearFieldOptics import Materials as M
            rp_norm = M.Au.reflection_p
        rp_norm_wrapped = lambda freq,q: rp_norm( freq / (a_nm * 1e-7),
                                                 q / (a_nm * 1e-7))
        amplitude = amplitude_nm / a_nm
        freqs = freqs_wn * (a_nm * 1e-7)
        if freqs_wn_norm is None: freqs_wn_norm = np.mean(freqs_wn)
        freqs_norm = freqs_wn_norm * (a_nm * 1e-7)

        signals = self.EradSpectrumDemodulated(freqs, rp=rp_wrapped,
                                               gapmin=gapmin, amplitude=amplitude,
                                               Ngaps=Ngaps, demod_order=demod_order)
        signals_ref = self.EradSpectrumDemodulated(freqs=freqs_norm, rp=rp_norm_wrapped,
                                                   gapmin=gapmin, amplitude=amplitude,
                                                   Ngaps=Ngaps, demod_order=demod_order)
        signals['Sn'] /= signals_ref['Sn']
        signals['Sn'].set_axes([None,freqs_wn],
                               axis_names=[None,'Frequency (cm$^{-1}$)'])
        signals['Erad'].set_axes([None,freqs_wn],
                                 axis_names=[None,'Frequency (cm$^{-1}$)'])

        return signals

    def getBasisFourPotentialAtZ(self, *args,**kwargs):

        return self.Probe.getFourPotentialAtZ(*args,**kwargs)

#------- Tools for inversion

def companion_matrix(p):
    """Assemble the companion matrix associated with a polynomial
    whose coefficients are given by `poly`, in order of decreasing
    degree.

    Currently unused, but might find a role in a custom polynomial
    root-finder.

    References:
    1) http://en.wikipedia.org/wiki/Companion_matrix
    """

    A = np.diag(np.ones((len(p) - 2,), np.complex128), -1)
    A[0, :] = -p[1:] / p[0]

    return np.matrix(A)

def find_roots(p, scaling=10):
    """Find roots of a polynomial with coefficients `p` by
    computing eigenvalues of the associated companion matrix.

    Overflow can be avoided during the eigenvalue calculation
    by preconditioning the companion matrix with a similarity
    transformation defined by scaling factor `scaling` > 1.
    This transformation will render the eigenvectors in a new
    basis, but the eigenvalues will remain unchanged.

    Empirically, `scaling=10` appears to enable successful
    root-finding on polynomials up to degree 192."""

    scaling = np.float64(scaling)
    M = companion_matrix(p)
    N = len(M)
    D = np.matrix(np.diag(scaling ** np.arange(N)))
    Dinv = np.matrix(np.diag((1 / scaling) ** np.arange(N)))
    Mstar = Dinv * M * D  # Similarity transform will leave the eigenvalues unchanged

    return linalg.eigvals(Mstar)


class InvertibleEigenfields(object):
    verbose = True

    def __init__(self, GapSpec, Nmodes=20,
                 interpolation='cubic',**kwargs):
        """For some reason using Nqs>=244, getting higher q-resolution,
        only makes more terms relevant, requiring twice as many terms for
        stability and smoothness in approach curves...
        (although overall """

        Brightnesses = GapSpec.get_brightnesses_AWA(Nmodes=Nmodes, recompute=False).T  # Put coordinate (gaps) axis first
        self.Rs = Brightnesses**2 #@TODO: Might want to double check this
        self.Ps = GapSpec.get_eigenrhos_AWA(Nmodes=Nmodes, verbose=False,recompute=False).T  # Put coordinate (gaps) axis first

        self.Nmodes = self.Rs.shape[1]
        self.zs = self.Rs.axes[0]

        self.build_poles_residues_interpolator(interpolation=interpolation,**kwargs)

    def polyfit_poles_residues(self, deg=6, zmax=None):
        #This is just in case we want to export later a better parametrization of the poles / residues vs. distance
        #But should change to Padé approximant

        if not zmax: zmax=np.max(self.zs)
        Nterms = self.Ps.shape[1]
        Rs = self.Rs.cslice[:zmax]
        Ps = self.Ps.cslice[:zmax]
        zs = Rs.axes[0]

        if self.verbose:
            Logger.write('Finding complex polynomial approximations of degree %i ' % deg + \
                         'to the first %i poles and residues, up to a value z/a=%s...' % (Nterms, zmax))

        self.Ppolys = []
        for i in range(Nterms):
            Ppoly = np.polyfit(zs, Ps[:, i], deg=deg)
            self.Ppolys.append(Ppoly)

        self.Rpolys = []
        for i in range(Nterms):
            Rpoly = np.polyfit(zs, Rs[:, i], deg=deg)
            self.Rpolys.append(Rpoly)

    def build_poles_residues_interpolator(self,interpolation='cubic',**kwargs):

        from scipy.interpolate import interp1d
        self.Ps_interp = [interp1d(self.zs,P,kind=interpolation,**kwargs) \
                          for P in self.Ps.T]
        self.Rs_interp = [interp1d(self.zs,R,kind=interpolation,**kwargs) \
                          for R in self.Rs.T]

    def evaluate_poles(self, zs=None, Nmodes=None):

        if zs is None: zs = self.zs

        # Default to all terms...
        if not Nmodes:
            Nmodes = self.Nmodes

        #Ps = self.Ps[:, :Nmodes].interpolate_axis(zs, axis=0, kind=interpolation,
        #                                          bounds_error=False, extrapolate=True)
        Ps = [interp(zs) for interp in self.Ps_interp[:Nmodes]]

        return np.array(Ps).T #keep z-axis first

    def evaluate_residues(self, zs=None, Nmodes=None):

        if zs is None: zs = self.zs

        # Default to all terms...
        if not Nmodes:
            Nmodes = self.Nmodes

        #Rs = self.Rs[:, :Nmodes].interpolate_axis(zs, axis=0, kind=interpolation,
        #                                          bounds_error=False, extrapolate=True)
        Rs = [interp(zs) for interp in self.Rs_interp[:Nmodes]]

        return np.array(Rs).T #keep z-axis first

    def evaluate_poles_poly(self, zs=None, Nmodes=None):

        if zs is None: zs = self.zs

        # Default to all terms...
        if not Nmodes:
            Nmodes = self.Nmodes

        Ps = np.array([np.polyval(Ppoly, zs) for Ppoly in self.Ppolys[:Nmodes]])

        return Ps.T

    def evaluate_residues_poly(self, zs=None, Nmodes=None, *args):

        if zs is None: zs = self.zs

        # Default to all terms...
        if not Nmodes:
            Nmodes = self.Nmodes

        Rs = np.array([np.polyval(Rpoly, zs) for Rpoly in self.Rpolys[:Nmodes]])

        return Rs.T

    # evaluate_poles=evaluate_poles_poly
    # evaluate_residues=evaluate_residues_poly

    def get_demodulation_nodes(self, zmin=.01, amplitude=2, quadrature=numrec.GL,
                               harmonic=3, Nts=None):
        # GL quadrature is the best, can do even up to harmonic 3 with 6 points on e.g. SiO2
        # TS requires twice as many points
        # quadrature `None` needs replacing, this linear quadrature is terrible

        # max harmonic resolvable will by 1/dt=Nts
        if not Nts: Nts = 8 * (harmonic+1)
        if isinstance(quadrature, str) or hasattr(quadrature, 'calc_nodes'):
            ts, wts = numrec.GetQuadrature(N=Nts, xmin=-.5, xmax=0, quadrature=quadrature)

        else:
            ts = np.linspace(-1 + 1 / np.float(Nts),
                                0 - 1 / np.float(Nts), Nts) * .5
            wts = np.ones((Nts,)) / np.float(Nts) * .5

        # This is what's necessary for fourier element
        # cos harmonic kernel, *2 for full period integration, *2 for coefficient
        wts *= 4 * np.cos(2 * np.pi * harmonic * ts)
        zs = zmin + amplitude * (1 + np.cos(2 * np.pi * ts))

        return list(zip(zs, wts))

    def get_Erad_from_nodes(self, beta, nodes, Nmodes=None):

        if not hasattr(beta, '__len__'): beta=[beta]
        if not isinstance(beta,np.ndarray): beta=np.array(beta)
        assert beta.ndim == 1

        # `Frequency` axis will be first
        if isinstance(beta, np.ndarray): beta = beta.reshape((len(beta), 1, 1))

        # Weights apply across z-values
        zs, ws = list(zip(*nodes))
        ws_grid = np.array(ws).reshape((1, len(ws), 1))
        zs = np.array(zs)

        # Evaluate at all nodal points
        Rs = self.evaluate_residues(zs, Nmodes)
        Ps = self.evaluate_poles(zs, Nmodes)

        # Should broadcast over freqs if beta has an additional first axis
        # The offset term is absolutely critical, offsets false z-dependence arising from first terms
        signals = np.sum(Rs * (1 / (beta - Ps) + 1 / Ps) * ws_grid, axis=-1)  # +Rs/Ps,axis=-1)

        signals = signals.squeeze()
        if isinstance(beta,AWA):
            axes=[beta.axes[0],zs]
            axis_names = [beta.axis_names[0], 'z/a']
            signals = AWA(signals,axes=axes,axis_names=axis_names)

        if not signals.ndim: signals = signals.tolist()

        return signals

    def get_Erad(self, beta, zs=None, Nmodes=None):

        if zs is None: zs=self.zs

        nodes = [(z,1) for z in zs]

        return self.get_Erad_from_nodes( beta, nodes, Nmodes=Nmodes)

    def get_Erad_demodulated(self,beta, zmin=.01,amplitude=2,
                             max_harmonic=5,Nts=None, Nmodes=None):

        harmonics = np.arange(max_harmonic+1)

        signals = []
        for harmonic in harmonics:

            nodes = self.get_demodulation_nodes( zmin=zmin, amplitude=amplitude, quadrature=numrec.GL,
                                                harmonic=harmonic, Nts=Nts)
            signal_at_nodes = self.get_Erad_from_nodes(beta, nodes, Nmodes=Nmodes)
            signal = np.sum(signal_at_nodes,axis=-1) #last axis will be across nodes
            signals.append(signal)

        axes=[harmonics]
        axis_names = ['Harmonic']
        if isinstance(signal,AWA):
            axes += signal.axes
            axis_names += signal.axis_names

        return AWA(signals,axes=axes,axis_names=axis_names)

    def __call__(self, *args, **kwargs):
        return self.get_Erad(*args, **kwargs)

    @staticmethod
    def demodulate(signals, zmin=.01, amplitude=2, max_harmonic=5, Nts=None,
                   quadrature=numrec.GL):
        """This only exists as a "check" on `get_Erad_demodulated`; they indeed provide the same answer"""

        global ts, wts, weights, signals_vs_time

        harmonics=np.arange(max_harmonic+1)

        # max harmonic resolvable will be frequency = 1/dt = Nts
        if not Nts: Nts = 4 * (np.max(harmonics)+1)
        if isinstance(quadrature, str) or hasattr(quadrature, 'calc_nodes'):
            ts, wts = numrec.GetQuadrature(N=Nts, xmin=-.5, xmax=0, quadrature=quadrature)

        else:
            ts, wts = np.linspace(-.5, 0, Nts), None

        zs = amplitude * (1 + np.cos(2 * np.pi * ts)) + zmin

        harmonics = np.array(harmonics).reshape((len(harmonics), 1))
        weights = np.cos(2 * np.pi * harmonics * ts)
        if wts is not None: weights *= wts
        weights_grid = weights.reshape(weights.shape + (1,) * (signals.ndim - 1))

        signals_vs_time = signals.interpolate_axis(zs, axis=0, bounds_error=False, extrapolate=True)
        signals_vs_time.set_axes([ts], axis_names=['t'])
        integrand = signals_vs_time * weights_grid

        if wts is not None:
            demodulated = 2 * 2 * np.sum(integrand, axis=1)  # perform quadrature
        else:
            demodulated = 2 * 2 * quadrature(integrand, x=ts, axis=1)

        axes = [harmonics]
        axis_names = ['harmonic']
        if isinstance(signals, AWA):
            axes += signals.axes[1:]
            axis_names += signals.axis_names[1:]
        demodulated = AWA(demodulated, axes=axes, axis_names=axis_names)

        return demodulated

    def invert_signal(self, signals, nodes, Nmodes=10,
                      select_by='continuity',
                      target=0,
                      scaling=10):
        """The inversion is not unique, consequently the selected solution
        will probably be wrong if signal values correspond with 
        "beta" values that are too large (`|beta|~>min{|Poles|}`).
        This can be expected to break at around `|beta|>2`."""
        # Default is to invert signal in contact
        # ~10 terms seem required to converge on e.g. SiO2 spectrum,
        # especially on the Re(beta)<0 (low signal) side of phonons

        global roots, poly, root_scaling

        # global betas,all_roots,pmin,rs,ps,As,Bs,roots,to_minimize
        if self.verbose:
            Logger.write('Inverting `signals` based on the provided `nodes` to obtain consistent beta values...')

        if not hasattr(signals, '__len__'): signals = [signals]
        if not isinstance(signals, AWA): signals = AWA(signals)

        zs, ws = list(zip(*nodes))
        ws_grid = np.array(ws).reshape((len(ws), 1))  # last dimension is to broadcast over all `Nterms` equally
        zs = np.array(zs)

        Rs = self.evaluate_residues(zs,Nmodes)
        Ps = self.evaluate_poles(zs,Nmodes)

        # `rs` and `ps` can safely remain as arrays for `invres`
        rs = (Rs * ws_grid).flatten()
        ps = Ps.flatten()

        k0 = np.sum(rs / ps).tolist()

        # Rescale units so their order of magnitude centers around 1
        rscaling = np.exp(-(np.log(np.abs(rs).max()) +
                               np.log(np.abs(rs).min())) / 2.)
        pscaling = np.exp(-(np.log(np.abs(ps).max()) +
                               np.log(np.abs(ps).min())) / 2.)
        root_scaling = 1 / pscaling
        # rscaling=1
        # pscaling=1
        if self.verbose:
            Logger.write('\tScaling residues by a factor %1.2e to reduce floating point overflow...' % rscaling)
            Logger.write('\tScaling poles by a factor %1.2e to reduce floating point overflow...' % pscaling)
        rs *= rscaling
        ps *= pscaling
        k0 *= rscaling / pscaling
        signals = signals * rscaling / pscaling

        # highest order first in `Ps` and `Qs`
        # VERY SLOW - about 100ms on practical inversions (~60 terms)
        As, Bs = invres(rs, ps, k=[k0], tol=1e-16,
                        rtype='avg')  # tol=1e-16 is the smallest allowable to `unique_roots`..

        dtype = np.complex128  # Double precision offers noticeable protection against overflow
        As = np.array(As, dtype=dtype)
        Bs = np.array(Bs, dtype=dtype)
        signals = signals.astype(dtype)

        # import time

        betas = []
        self.closest_roots=[]
        for i, signal in enumerate(signals):
            # t1=time.time()

            # Root finding `roots` seems to give noisy results when `Bs` has degree >84, with dynamic range ~1e+/-30 in coefficients...
            # Pretty fast - 5-9 ms on practical inversions with rank ~60 companion matrices, <1 ms with ~36 terms
            # @TODO: Root finding chokes on `Nterms=9` (number of eigenfields) and `Nts=12` (number of nodes),
            #       necessary for truly converged S3 on resonant phonons, probably due to
            #       floating point overflow - leading term increases exponentially with
            #       number of terms, leading to huge dynamic range.
            #       Perhaps limited by the double precision of DGEEV.
            #       So, replace with faster / more reliable root finder?
            #       We need 1) speed, 2) ALL roots (or at least the first ~10 smallest)
            poly = As - signal * Bs
            roots = find_roots(poly, scaling=scaling)
            #roots = np.roots(poly)
            where_valid = roots.imag > 0
            if not where_valid.any():
                raise ValueError('No physical roots found, the model is invalid!  Please change input parameters.')
            roots = roots[where_valid]
            roots *= root_scaling  # since all beta units scaled by `pscaling`, undo that here

            # print time.time()-t1

            # How should we select the most likely beta among the multiple solutions?
            # 1. Avoids large changes in value of beta
            if select_by == 'difference' and i >= 1:
                if i == 1 and self.verbose:
                    Logger.write('\tSelecting remaining roots by minimizing differences with prior...')
                to_minimize = np.abs(roots - betas[i - 1])

            # 2. Avoids large changes in slope of beta (best for spectroscopy)
            # Nearly guarantees good beta spectrum, with exception of very loosely sampled SiC spectrum
            # Loosely samples SiO2-magnitude phonons still perfectly fine
            elif select_by == 'continuity' and i >= 2:
                if i == 2 and self.verbose:
                    Logger.write('\tSelecting remaining roots by ensuring continuity with prior...')
                earlier_diff = betas[i - 1] - betas[i - 2]
                current_diffs = roots - betas[i - 1]
                to_minimize = np.abs(current_diffs - earlier_diff)

            # 3. Select specifically which pole we want |beta| to be closest to
            elif select_by=='target':
                if hasattr(target, '__len__'): target = target[i]
                else: target = target

                if self.verbose and i==0:
                    Logger.write('\tSeeding inversion closest to pole %s...' % target)
                to_minimize = np.abs(target - roots)

            else:
                raise ValueError('`select_by=%s` not recognized!  Choose "difference", "continuity", or "target".'%select_by)

            self.closest_roots.append( roots[np.argsort(to_minimize)][:5] )

            beta = roots[to_minimize == to_minimize.min()].squeeze()
            betas.append(beta)
            if not i % 5 and self.verbose:
                Logger.write('\tProgress: %1.2f%%  -  Inverted %i signals of %i.' % \
                             (((i + 1) / np.float(len(signals)) * 100),
                              (i + 1), len(signals)))

        betas = AWA(betas)
        betas.adopt_axes(signals)
        betas = betas.squeeze()
        if not betas.ndim: betas = betas.tolist()

        return betas

