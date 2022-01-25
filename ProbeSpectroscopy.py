import numpy as np
import time
from numbers import Number
from common.log import Logger
from common import plotting
from common import numerical_recipes as numrec
from common.baseclasses import AWA
import ProbeCavityEigenfields as PCE

#--- Utilities

def get_superset_indices(subset,superset):

    subset=np.array(subset); superset=np.array(superset)
    assert len(set(subset))==len(subset), 'All elements of `subset` must be unique!'
    assert len(set(superset))==len(superset), 'All elements of `subset` must be unique!'
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

    def distance(self, xm, xn, m, n):
        """Get 'distance' between eigencharge at coordinate `xm`, index `m`,
        and that at coordinate `xn`, index `n`."""

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

    def classify_eigensets(self, Nmodes=10, reversed=False):
        # This is the robust way of labeling eigensets by an eigenindex
        # in a way that is uniform across the internal coordinates
        # `reversed` is untested

        print('Classifying eigensets by eigenindex...')
        coords = sorted(list(self._recorded_eigenrhos.keys()))
        if reversed: coords = coords[::-1]

        # --- Assign labels to modes
        connectivity = []
        for coordind in range(len(coords)):

            coord = coords[coordind]
            N = np.min((Nmodes, len(self._recorded_eigenrhos[coord])))

            if coordind == 0:
                connections = list(range(N))

            else:
                coord_prev = coords[coordind - 1]
                M = np.min((Nmodes, len(self._recorded_eigenrhos[coord_prev])))

                connections = []
                for n in range(N):
                    ds = [self.distance(coord_prev, coord, m, n) for m in range(M)]
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

    def get_eigenrhos_AWA(self,Nmodes=10,verbose=True):

        if self.sorted_eigenrhos is None:
            raise RuntimeError('Call the spectroscopy instance first (i.e. `Spectroscopy()` ) to sort eigensets!')

        #--- Find coordinates that are common to all eigensets
        #  since some eigensets may have been dropped during the sorting process
        coords = [rhos.axes[0] for rhos in self.sorted_eigenrhos[:Nmodes]]
        coords_common = sorted(list(set(coords[0]).intersection(*coords[1:])))
        if verbose:
            Logger.write('For Nmodes=%i, there were %i identifiable mutual coordinates.'%(Nmodes,len(coords_common)))

        ind_sets = [ get_superset_indices(subset=coords_common,\
                                         superset=coord_set) \
                    for coord_set in coords ]

        eigenrhos_grid = [rhos[ind_set] for rhos,ind_set \
                          in zip(self.sorted_eigenrhos[:Nmodes],\
                                 ind_sets)]

        eigenrhos_grid = AWA(eigenrhos_grid,axes=[None,coords_common],\
                             axis_names=['eigenindex','coordinate'])

        return eigenrhos_grid

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
            overlap = np.complex( Qm.T @ Oavg @ Qn )
            if overlap.real<0: charge_to_add = -charge_to_add

            aligned_eigencharges.append(charge_to_add)

        return aligned_eigencharges

    def get_eigencharges_AWA(self,Nmodes=10,verbose=True):

        if self.sorted_eigenrhos is None:
            raise RuntimeError('Call the spectroscopy instance first (i.e. `Spectroscopy()` ) to sort eigensets!')

        #--- Find coordinates that are common to all eigensets
        #  since some eigensets may have been dropped during the sorting process
        coords = [ rhos.axes[0] for rhos in self.sorted_eigenrhos[:Nmodes] ]
        coords_common = sorted(list(set(coords[0]).intersection(*coords[1:])))
        if verbose:
            Logger.write('For Nmodes=%i, there were %i mutual spectroscopy coordinates.'%(Nmodes,len(coords_common)))

        ind_sets = [ get_superset_indices(subset=coords_common,\
                                         superset=coord_set) \
                    for coord_set in coords ]

        eigencharges_grid = [ charges[ind_set] for charges, ind_set \
                              in zip(self.sorted_eigencharges[:Nmodes], \
                                     ind_sets) ]
        len(eigencharges_grid)

        if verbose: Logger.write('\tAligning eigencharge signage...')
        eigencharges_grid = [ self.align_eigencharge_signs(eigencharges,\
                                                           coords=coords_common,\
                                                           eigenindex=eigenindex) \
                              for eigenindex,eigencharges in enumerate(eigencharges_grid) ]

        eigencharges_AWA = AWA(eigencharges_grid,axes=[None,coords_common,\
                                                        eigencharges_grid[0][0].axes[0]],\
                             axis_names=['eigenindex','coordinate',\
                                         eigencharges_grid[0][0].axis_names[0]])

        return eigencharges_AWA #Axes are eigenindex, coordinate, z-on-probe

    def get_brightnesses_AWA(self,Nmodes=None,recompute=False,\
                             angles=np.linspace(10,80,20),**kwargs):

        try:
            if recompute: raise AttributeError
            return self._brightesses_AWA
        except AttributeError: pass #proceed to evaluate

        eigencharges_AWA = self.get_eigencharges_AWA(Nmodes,verbose=False)
        coords=eigencharges_AWA.axes[1]
        Nmodes=len(eigencharges_AWA)

        Logger.write('Computing brightnesses across %i spectroscopy coordinates...'%len(coords))

        all_brightnesses=[]
        for coord in coords:
            eigencharges = eigencharges_AWA.cslice[:,coord]
            P = self.get_probe_at_coord(coord,verbose=False)
            brightnesses = [P.get_brightness(charge,angles=angles,**kwargs) \
                            for charge in eigencharges]
            all_brightnesses.append(brightnesses)

        self._brightnesses_AWA = AWA(all_brightnesses,\
                                          axes=[coords,None],\
                                          axis_names=['coordinate','eigenindex']).T #transpose to conform with previous convention

        return self._brightnesses_AWA


    def plot_eigenrhos_scatter(self):

        from matplotlib import pyplot as plt
        from matplotlib import cm,colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        coords=self.get_coords()
        cs = plotting.bluered_colors(len(coords))
        for coord in coords:
            rhos = np.array(self._recorded_eigenrhos[coord])
            c = next(cs)
            plt.plot(rhos.real, rhos.imag, color=c, marker='o', ls='')

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

    def plot_eigenrhos(self):

        from matplotlib import pyplot as plt
        from matplotlib import cm,colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        eigenrhos_AWA = self.get_eigenrhos_AWA()

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

        print('Starting freq=%1.2f' % freq)
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
        print('Finished freq=%1.2f THz, elapsed time: %1.2g' % (freq, time.time() - t0))

        return rhos, Qs, Zself

def compute_eigenset_at_gap(P, gap=1, k=0, **kwargs):

        print('Starting gap=%1.2f' % gap)
        t0 = time.time()
        # Zself = P.get_self_impedance(k=P.get_k(),recompute=True,display=False)

        P.set_gap(gap)
        if k is None: k=P.get_k()
        # kappa_max=20/gap
        kappa_max = np.inf
        # Zmirror = P.get_mirror_impedance(k=k,farfield=False,\
        #                                 recompute=True,sommerfeld=True,Nkappas=244*8,kappa_max=kappa_max)

        # Use sommerfeld calculation because it's faster
        Zmirror = P.get_mirror_impedance(k=k, recompute=True, sommerfeld=True, **kwargs)

        rhos, Qs = P.solve_eigenmodes(condition_ZM=True, condition_ZS=False, ZMthresh=0, recompute_impedance=False)
        print('Finished gap=%1.2f, elapsed time: %1.2g' % (gap, time.time() - t0))

        Zself = P.get_self_impedance(recompute=False, display=False)

        return rhos, Qs, Zself

class ProbeSpectroscopyParallel(ProbeSpectroscopy):

    def __init__(self,P, coords, eigenset_calculator=compute_eigenset_at_gap,\
                 ncpus=8, backend='multiprocessing', Nmodes=20, **kwargs):
        from joblib import Parallel, delayed, parallel_backend

        super().__init__(P)

        #--- Make sure probe has self impedance already computed, so at least it doesn't need to be recomputed across parallel iterations
        ZSelf = P.get_self_impedance(k=P.get_k(), recompute=False)

        t0 = time.time()
        eigensets = []
        while len(eigensets) < len(coords):
            #--- Delegate group of coords to a single parallel job
            nstart = len(eigensets);
            nstop = len(eigensets) + ncpus
            coords_sub = coords[nstart:nstop]
            new_eigensets = Parallel(backend=backend, n_jobs=ncpus)(delayed(eigenset_calculator)(P, coord, **kwargs) \
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
        self.classify_eigensets(Nmodes=Nmodes)

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

class ProbeGapSpectroscopyParallel(ProbeSpectroscopyParallel):

    def __init__(self,P, gaps=np.logspace(-1.5,1,100),\
                 ncpus=8, backend='multiprocessing',\
                 Nmodes=20, **kwargs):

        #--- Hard-code the gap calculator
        kwargs['k'] = 0 #we insist on a quasistatic calculation, otherwise some features of this class become meaningless
        super().__init__(P, coords=gaps, eigenset_calculator=compute_eigenset_at_gap,\
                         ncpus=ncpus, backend=backend, Nmodes=Nmodes, **kwargs)

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
        eigencharges_vs_gap = Spec.get_eigencharges_AWA(Nmodes=Nmodes)
        gaps = eigencharges_vs_gap.axes[1] # axes are eigenindex, coordinate, zprobe
        Logger.write('Encoding eigenfields to gap=%1.2f across %i gap values from %s to %s...' \
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
        Brightnesses = Spec.get_brightnesses_AWA(Nmodes=None, recompute=False,
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
                rp_vs_q[np.isnan(rp_vs_q)] = 0  # This is just probably trying to evaluate at too large a momentum
                rp_vs_q[np.isinf(rp_vs_q)] = 0  # This is just probably trying to evaluate at too large a momentum
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
            Erad = BVec.T @ \
                      ( (RMat - PoleMat).I + (PoleMat).I ) \
                        @ BVec  # scattering/summing etc. is implicit

            Erads.append( np.complex(Erad) )

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

    def getBasisFourPotentialAtZ(self, *args,**kwargs):

        return self.Probe.getFourPotentialAtZ(*args,**kwargs)



