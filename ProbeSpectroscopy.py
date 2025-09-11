import numpy as np
import os
import time
import copy
import pickle
import functools
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

    def __init__(self, Probe):

        # Store details of the Probe to restore later
        assert isinstance(Probe, PCE.Probe)
        self.Probe=Probe

        self._recorded_eigenrhos = {}
        self._recorded_eigencharges = {}
        self._recorded_self_impedances = {}
        self._recorded_mirror_impedances = {}

        # lists to be populated during sorting
        self.eigenrhos = None
        self.eigencharges = None

    def __setstate__(self, state):

        # The primary purpose here is to rename legacy attributes upon unpickling
        self.__dict__.update(state)

        newattrs = {'_P': 'Probe'}
        for dest, src in newattrs.items():
            if not hasattr(self, dest):
                if isinstance(src, str):  # interpret src as a key to write from
                    if src in state:  setattr(self, dest, state[src])
                else:
                    setattr(self, dest, src)

    def get_probe(self): return self.Probe

    def check(self):

        charges=self._recorded_eigencharges
        if not charges: return

        # Test that we can associate this spectroscopy with the attached probe.
        ctest = list(charges.values())[0][0] #First coord, first eigenindex
        try: self.Probe._toAWA(ctest)
        except:
            raise ValueError(
                'The attached probe does not seem to associated with this `ProbeSpectroscopy`!  The latter should be recomputed.')

    filename_template = '(%s)_ProbeSpectroscopy.pickle'

    def save(self, overwrite=False):  return PCE.save(self, overwrite=overwrite)

    def record(self, x):
        """Record probe self impedance and eigenset at coordinate `x`,
        as well as the Probe itself - this is all that's necessary to
        restore response functions at `x`."""

        P = self.get_probe()

        self._recorded_eigenrhos[x] = P.get_eigenrhos()
        self._recorded_eigencharges[x] = P.get_eigencharges()

        Zself =P.get_self_impedance(recompute=False)
        self._recorded_self_impedances[x] = Zself

        Zmirror = P.get_mirror_impedance(recompute=False)
        self._recorded_mirror_impedances[x] = Zmirror

    def get_coords(self):
        return np.array(sorted(self._recorded_eigenrhos.keys()))

    def get_probe_at_coord(self, coord,
                           update_charges=True, update_Zself=True,update_Zmirror=True,
                           verbose=True):

        self.check()

        coords = self.get_coords()
        self._coord = coords[np.argmin(np.abs(coord - coords))]

        if verbose: Logger.write('Updating eigenrhos to coordinate %1.2E...' % self._coord)
        rhos = self._recorded_eigenrhos[self._coord]
        self.Probe._eigenrhos = rhos

        if update_charges:
            if verbose: Logger.write('\tUpdating eigencharges...')
            charges = self._recorded_eigencharges[self._coord]
            self.Probe._eigencharges = charges
        if update_Zself:
            if verbose: Logger.write('\tUpdating self impedance...')
            ZSelf = self._recorded_self_impedances[self._coord]
            self.Probe._ZSelf = ZSelf
        if update_Zmirror:
            if verbose: Logger.write('\tUpdating mirror impedance...')
            ZMirror = self._recorded_mirror_impedances[self._coord]
            self.Probe._ZMirror = ZMirror

        return self.get_probe()

    def set_eigenset(self,probe,coord,
                     update_charges=True,update_Zself=True,update_Zmirror=True):

        best_probe = self.get_probe_at_coord(coord,update_charges=update_charges,
                                             update_Zself=update_Zself,update_Zmirror=update_Zmirror)
        probe._eigenrhos = best_probe._eigenrhos

        if update_charges:
            probe._eigencharges = best_probe._eigencharges
        if update_Zself:
            probe._ZSelf = best_probe._ZSelf
        if update_Zmirror:
            probe._ZMirror = best_probe._ZMirror

    # 2023.05.31 - Deprecated in favor of more parallelized and transparent `eigensets_overlap_matrix`
    """def distance(self, xm, xn, m, n, by_eigenrho=True):
        "Get 'distance' between eigencharge at coordinate `xm`, index `m`,
        and that at coordinate `xn`, index `n`."

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
        distance = 1 / np.abs(complex(overlap))

        # -- This one's a no-brainer, and workable
        # distance=np.min( (np.sum(np.abs(v1-v2)),\
        #                  np.sum(np.abs(v1+v2))) )

        # -- This one seems to work comparable to the standard, maybe better?
        # rho1=self.eigenrhos[xm][m]
        # rho2=self.eigenrhos[xn][n]
        # distr2 = (np.log(rho1.real)-np.log(rho2.real))**2
        # disti2 = (np.log(-rho1.imag)-np.log(-rho2.imag))**2
        # distance=np.sqrt(distr2+disti2)

        return distance"""

    def eigenrhos_distance(self,xm, xn, Nmodes=10):

        rhosn = self._recorded_eigenrhos[xn][:Nmodes][:,np.newaxis]
        rhosm = self._recorded_eigenrhos[xm][:Nmodes][np.newaxis,:]

        distances = rhosn-rhosm #entry `n,m` will correspond to distance from n to m
        #distances /= (rhosn+rhosm)/2 # Divide by average rho value

        return distances

    def eigencharges_overlap_matrix(self, xm, xn,
                                    Qn=None,Qm=None, Nmodes=10):
        """If `Qm` or `Qn` are not provided, they will be taken from `self._recorded_eigencharges` (unsorted)."""

        if Qm is None or Qn is None:
            Qm = self._recorded_eigencharges[xm][:Nmodes]
            Qn = self._recorded_eigencharges[xn][:Nmodes]
        else:
            assert len(Qn)==len(Qm)
            Nmodes = len(Qn)

        Qm = np.matrix(Qm).T  # column vectors
        Qn = np.matrix(Qn).T  # column vectors

        #--- Get overlap operator at "midpoint"
        # `O = (Om + On)/2`
        # `Om = -PhiM = PhiS / rho[m]` whenever operating on `Qm`
        # because `(rho[m] Zm + Zs) Qm = 0`, so `Qm * Zs/rho[m] * Qm` is maximized at `~1`

        Om = self._recorded_self_impedances[xm]
        On = self._recorded_self_impedances[xn]

        #Qm has to be divided by rhos[m] before impedance matrix, so make a diagonal matrix that does it
        InvRhom = np.matrix(np.diag(1/self._recorded_eigenrhos[xm][:Nmodes]))
        InvRhon = np.matrix(np.diag(1/self._recorded_eigenrhos[xn][:Nmodes]))

        overlap1 = np.array((Qn).T @ Om @ (Qm @ InvRhom))
        overlap2 = np.array((Qn @ InvRhon).T @ On @ (Qm))
        #overlap1 = np.array((Qn).T @ Om @ Qm @ InvRhom))
        #overlap2=1

        return np.array(overlap1+overlap2)/2 #entry [n,m] is overlap between eigencharge n and m

    def classify_eigensets(self, Nmodes=10, reversed=False, by_rho=True, debug=False,
                           coordmin=None,coordmax=None,**kwargs):
        # This is the robust way of labeling eigensets by an eigenindex
        # in a way that is uniform across the internal coordinates

        from scipy.optimize import linear_sum_assignment
        import copy,sys

        if reversed: print('Classifying eigensets by eigenindex, with reversal...')
        else: print('Classifying eigensets by eigenindex...')
        coords = sorted(list(self._recorded_eigenrhos.keys()))
        coords = np.array(coords)
        if reversed: coords = coords[::-1]
        if coordmin: coords = coords[coords>=coordmin]
        if coordmax:  coords = coords[coords<=coordmax]

        # Make sure we don't look for more modes than we have available at all coords
        Nmodes_original=Nmodes
        for coord in coords:
            Nmodes_here = len(self._recorded_eigenrhos[coord])
            if Nmodes_here < Nmodes: Nmodes = Nmodes_here
        if Nmodes< Nmodes_original:
            print('Looking for Nmodes=%i (maximum uniformly available in coordinate range).'%Nmodes)

        if by_rho: cost_func = self.eigenrhos_distance
        else: cost_func = self.eigencharges_overlap_matrix

        # --- Assign labels to modes
        all_nu_values=[np.arange(Nmodes)]
        all_rhos = [self._recorded_eigenrhos[coords[0]][:Nmodes]]
        all_charges = [self._recorded_eigencharges[coords[0]][:Nmodes]]
        charge_zs = all_charges[0][0].axes[0]

        charge_shape = all_charges[0][0].shape
        default_charges_list = [np.zeros(charge_shape) for i in range(Nmodes)]
        default_rhos_list = [np.nan for i in range(Nmodes)]

        for coordind,coord in enumerate(coords):
            if coordind==0: continue # We've already populated the zeroth entries
            coord_prev = coords[coordind-1]

            progress=coordind / len(coords)*100
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write('\tProgress: %1.2f%%'%progress)
            sys.stdout.flush()
            if debug: print('\n coordind=%1.2f'%coordind)

            cost_matrix = cost_func(xm=coord_prev, xn=coord, Nmodes=Nmodes)
            row_ind, col_ind = linear_sum_assignment(np.abs(cost_matrix))

            #We have answered which present indices match the previous ones.
            # we will re-map that answer to "which previous nu values.."
            prev_nu_values = all_nu_values[-1][row_ind]
            local_nu_values = prev_nu_values[col_ind]

            these_charges = copy.copy(default_charges_list)
            these_rhos = copy.copy(default_rhos_list)
            for index,nu in enumerate(local_nu_values):
                these_charges[nu]=self._recorded_eigencharges[coord][index]
                these_rhos[nu]=self._recorded_eigenrhos[coord][index]

            all_nu_values.append(local_nu_values)
            all_rhos.append(these_rhos)
            all_charges.append(these_charges)

        self._eigenrhos_AWA = AWA(all_rhos, axes=[coords,None],axis_names=['coord',r'$\nu$'])
        self._eigenrhos_AWA = np.transpose(self._eigenrhos_AWA, (1,0)) #put `nu` index first
        self._eigenrhos_AWA = self._eigenrhos_AWA.sort_by_axes()
        self._eigencharges_AWA = AWA(all_charges, axes=[coords,None,charge_zs],axis_names=['coord',r'$\nu$','z'])
        self._eigencharges_AWA = np.transpose(self._eigencharges_AWA, (1,0,2)) #put `nu` index first, then coord, then charge z-axis
        self._eigencharges_AWA = self._eigencharges_AWA.sort_by_axes()
        self._eigencharges_AWA = self.align_eigencharge_signs(self._eigencharges_AWA)

        #self.sorted_eigenrhos = all_rhos
        #self.sorted_eigencharges = all_charges

    def __call__(self,*args,**kwargs): return self.classify_eigensets(*args,**kwargs)

    def get_eigenrhos_AWA(self,Nmodes=None):

        result = self._eigenrhos_AWA
        if Nmodes is not None: result = result[:Nmodes]

        return result

        try:
            precomputed = self._eigenrhos_AWA
            if (Nmodes and Nmodes != len(precomputed)) or recompute: #If we need more modes than we have, also recompute
                raise AttributeError
            return precomputed[:Nmodes]
        except AttributeError: pass #proceed to evaluate

        if Nmodes is None: Nmodes=10

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

    def align_eigencharge_signs(self,eigencharges):
        """Assume `eigencharges` has dimensions:
        eigenindex, spectroscopy coord, zs along probe
        """

        Logger.write('Aligning signage of %i eigenmode charges over %s coordinates...'%eigencharges.shape[:2])
        wzs = self.get_probe().get_weights()
        eigencharges = np.asanyarray(eigencharges)
        Nmodes,Ncoords = eigencharges.shape[:2]

        eigencharges_aligned = eigencharges.copy()

        for mode_index in range(Nmodes):
            for coord_index in np.arange(Ncoords-1)+1:
                Qm = eigencharges_aligned[mode_index,coord_index-1] # already aligned
                Qn = eigencharges_aligned[mode_index,coord_index] # not aligned

                sum = np.sum(wzs*np.abs(Qm+Qn))
                diff = np.sum(wzs*np.abs(Qm-Qn))
                if diff<sum: Qn_best = Qn
                else: Qn_best = -Qn

                eigencharges_aligned[mode_index,coord_index] = Qn_best

        return eigencharges_aligned

    def get_eigencharges_AWA(self,Nmodes=None):

        result = self._eigencharges_AWA
        if Nmodes is not None: result = result[:Nmodes]

        return result

    def get_eigenbrightness_AWA(self, Nmodes=None, recompute=False,
                                eigencharges_AWA=None,
                                angles=None,
                                verbose=True, **kwargs):

        try:
            precomputed = self._brightnesses_AWA
            if (Nmodes and Nmodes != len(precomputed)) or recompute: #If we need more modes than we have, also recompute
                raise AttributeError
            return precomputed[:Nmodes]
        except AttributeError:
            import traceback
            pass #proceed to evaluate

        if Nmodes is None: Nmodes=10

        if eigencharges_AWA is None:
            eigencharges_AWA = self.get_eigencharges_AWA(Nmodes)
        coords=eigencharges_AWA.axes[1]
        if Nmodes:
            eigencharges_AWA = eigencharges_AWA[:Nmodes]

        if verbose: Logger.write('Computing brightnesses across %i spectroscopy coordinates...'%len(coords))

        all_brightnesses=[]
        for coord in coords:
            eigencharges = np.array(eigencharges_AWA.cslice[:,coord])
            P = self.get_probe_at_coord(coord,verbose=False)
            brightnesses = P.get_brightness(eigencharges, illum_angles=angles, **kwargs) # vectorized over charge vectors
            #Average over angles (first axis)
            brightnesses=np.array(brightnesses)
            brightnesses = np.mean(brightnesses,axis=0)
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
                plt.xlabel('Coordinate')
                plt.ylabel(r'Re/Im $\rho_i$')
                plt.gca().set_yscale('symlog')
                plt.gca().set_xscale('linear')
            else:
                plt.plot(rhos.real, rhos.imag, color=c, marker='o', ls='',**kwargs)
                plt.xlabel(r'Re $\rho_i$')
                plt.ylabel(r'Im $\rho_i$')
                plt.gca().set_yscale('symlog')
                plt.gca().set_xscale('log')

        plt.axhline(0,ls='--',color='k',alpha=.5)

        cmap = plt.get_cmap('BlueRed')
        norm = colors.Normalize(vmin=coords.min(), vmax=coords.max())

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%',pad=.1)
        plt.gcf().colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cax, orientation='vertical',label='coordinate')

    def plot_eigenrhos(self,Nmodes=10,**kwargs):

        from matplotlib import pyplot as plt
        from matplotlib import cm,colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        eigenrhos_AWA = self.get_eigenrhos_AWA(Nmodes=Nmodes)

        cs = plotting.bluered_colors(len(eigenrhos_AWA))
        for rhos in eigenrhos_AWA:
            c = next(cs)
            rhos.real.plot(color=c,**kwargs)
            rhos.imag.plot(color=c,ls='--',**kwargs)

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
                                             recompute=True, sommerfeld=True, **kwargs)
        else: Zmirror = P.get_mirror_impedance(recompute=False)

        rhos, Qs = P.solve_eigenmodes(condition_ZM=True, condition_ZS=False, ZMthresh=0, recompute_impedance=False)
        print('Finished freq=%1.2g, elapsed time: %1.2g s' % (freq, time.time() - t0))

        return rhos, Qs, Zself, Zmirror

def compute_eigenset_at_gap(P, gap=1, k=0, sommerfeld=True,
                            basis=None,**kwargs):

        print('Starting gap=%1.2g' % gap)
        t0 = time.time()

        #--- Set new gap and recompute mirror impedance
        P.set_gap(gap)
        if k is None: k=P.get_k()

        # Use sommerfeld calculation because it's faster
        # Make the q-grid adaptive; as gap gets smaller, there is sampling to higher q-values
        if sommerfeld and 'kappa_max' not in kwargs:
            kwargs['kappa_max'] = 10*np.max( (1/gap, 1/P.get_a()) )
        Zmirror = P.get_mirror_impedance(k=k, recompute=True, sommerfeld=sommerfeld, **kwargs)
        Zself = P.get_self_impedance(recompute=False, display=False)

        rhos, Qs = P.solve_eigenmodes(condition_ZM=True, condition_ZS=False, ZMthresh=0,basis=basis,
                                      recompute_impedance=False,plot=False)
        print('Finished gap=%1.2g, elapsed time: %1.2g s' % (gap, time.time() - t0))

        return rhos, Qs, Zself, Zmirror

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

        def __exit__(self,type, value, traceback): #last arguments are obligatory
            # --- Restore any removed Probe attributes
            for attr in self.removed_attrs:
                setattr(self.P, attr,
                        self.removed_attrs[attr])

    def __init__(self, Probe, coords, eigenset_calculator=compute_eigenset_at_gap, \
                 ncpus=8, backend='multiprocessing', Nmodes=20, reversed=False, **kwargs):

        from joblib import Parallel, delayed, parallel_backend

        super().__init__(Probe)

        #--- Make sure probe impedances have already been computed
        # We don't know which ones will be recomputed across parallel iterations,
        # but we can make sure the complement is available
        ZSelf = Probe.get_self_impedance(k=Probe.get_k(), recompute=False)
        ZMirror = Probe.get_mirror_impedance(k=0, recompute=False)

        t0 = time.time()
        eigensets = []
        with self.serializableProbe(Probe) as context:
            # Reference to the probe will have its needless attributes torn down on `__enter__`
            with Parallel(n_jobs=ncpus,backend=backend) as parallel:
                while len(eigensets) < len(coords):
                    #--- Delegate group of coords to a single parallel job
                    nstart = len(eigensets)
                    nstop = len(eigensets) + ncpus
                    coords_sub = coords[nstart:nstop]
                    new_eigensets = parallel(delayed(eigenset_calculator)(Probe, coord, **kwargs) \
                                                                          for coord in coords_sub)
                    Logger.write('\tParallel calculation retrieved %i results!'%ncpus)
                    eigensets = eigensets + new_eigensets
                Logger.write('\tTime elapsed: ', time.time() - t0)
            #Probe attributes are restored on `__exit__`

        for coord, eigenset in zip(coords, eigensets):
            rhos, charges, ZSelf, ZMirror = eigenset
            Probe._eigenrhos = rhos
            Probe._eigencharges = charges
            Probe._ZSelf = ZSelf
            Probe._ZMirror = ZMirror
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

# Probe spectroscopies
class ProbeFrequencySpectroscopyParallel(ProbeSpectroscopyParallel):

    filename_template = '(%s)_ProbeFrequencySpectroscopy.pickle'

    def __init__(self, Probe, freqs,
                 ncpus=8, backend='multiprocessing',
                 Nmodes=20, reversed=False,
                 **kwargs):

        #--- Hard-code the frequency calculator
        super().__init__(Probe, coords=freqs, eigenset_calculator=compute_eigenset_at_freq,
                         ncpus=ncpus, backend=backend, Nmodes=Nmodes, reversed=reversed,
                         PhiM=False, **kwargs)

    def get_probe_at_coord(self, freq, *args,**kwargs):

        result = super().get_probe_at_coord(freq, *args,**kwargs)

        # Don't call `set_freq` which would trigger `reset_eigenproperties` unnessecarily
        self.Probe._freq = self._coord

        return result

    def set_eigenset(self,Probe, freq, *args,**kwargs):

        result = super().set_eigenset(Probe,freq,*args,**kwargs)

        # Don't call `set_freq` which would trigger `reset_eigenproperties` unnessecarily
        Probe._freq = self._coord

        return result

    get_probe_at_freq = get_probe_at_coord

class ProbeGapSpectroscopyParallel(ProbeSpectroscopyParallel):

    filename_template = '(%s)_ProbeGapSpectroscopy.pickle'

    def __init__(self, Probe, gaps=np.logspace(-1.5, 1, 100), \
                 ncpus=8, backend='multiprocessing', \
                 Nmodes=20, sommerfeld=True,
                 basis_gap=None,
                 reversed=False,
                 **kwargs):

        assert isinstance(Nmodes, int) and Nmodes > 0

        #--- Make sure self impedance has been calculated
        ZSelf = Probe.get_self_impedance(k=Probe.get_k(), recompute=False)

        if basis_gap:
            print('We are computing an eigenset at gap=%1.2f, and using its first %i modes as a basis for forthcoming calculations!'%(basis_gap,Nmodes))
            #--- Make sure we make the zeroth eigensolution available for the dispatched worker
            compute_eigenset_at_gap(Probe, gap=basis_gap, k=0, sommerfeld=sommerfeld,
                                    basis=None, **kwargs) #This will attach desired eigenset to `P`
            # We will only need `Nmodes` of these!
            kwargs['basis']= Probe.get_eigencharges()[:Nmodes] #Pass basis on to kwargs where it will be fed down to parallel jobs
            print('Basis will have length %i'%kwargs['basis'].shape[0])

        #--- Hard-code the gap calculator
        kwargs['k'] = 0 #we insist on a quasistatic calculation, otherwise some features of this class become meaningless
        super().__init__(Probe, coords=gaps, eigenset_calculator=compute_eigenset_at_gap,
                         ncpus=ncpus, backend=backend, Nmodes=Nmodes, reversed=reversed,
                         sommerfeld=sommerfeld, **kwargs)

    def get_probe_at_coord(self, gap,*args,**kwargs):

        result = super().get_probe_at_coord(gap,*args,**kwargs)

        # Don't call `set_gap` which would trigger `reset_eigenproperties` unnessecarily
        self.Probe._gap = self._coord

        return result

    get_probe_at_gap = get_probe_at_coord

    def set_eigenset(self,Probe, gap, *args,**kwargs):

        result = super().set_eigenset(Probe,gap,*args,**kwargs)

        # Don't call `set_gap` which would trigger `reset_eigenproperties` unnessecarily
        Probe._gap = self._coord

        return result

    def Encode(self,*args,recompute=False,reload=False,**kwargs):

        # Try to load from file rather than compute anew
        try:
            if recompute: raise ValueError #Bail to the calculation
            if not hasattr(self,'_encodedEigenfields') \
                or self._encodedEigenfields is None \
                or reload:
                self._encodedEigenfields = PCE.load(self.get_probe(),
                                                    EncodedEigenfields)
                self._encodedEigenfields.EncodedEigenfields = self.get_probe() # ensure that we re-attach our real Probe
                # There is nothing now to check, `EncodedEigenfields` holds no explicit connection to discretization of probe..`

        except (OSError,ValueError):
            self._encodedEigenfields = EncodedEigenfields(self,*args,**kwargs)

        return self._encodedEigenfields

class EncodedEigenfields(object):
    """This function condensed a `ProbeSpectroscopy` object
    down to an encoded version that can duck-type in some functionalities
    as a `Probe` object.

    Particularly useful to condense a `ProbeGapSpectroscopy` so that
    demodulated near-field scattering signals can be computed easily.

    Wrapping other probe spectroscopy types is not yet supported (or useful?)."""

    verbose = True

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
        if self.verbose: Logger.write('Encoding %i eigenfields to gap=%1.2g across %i gap values from gap=(%1.2g to %1.2g)...' \
                         % (len(eigencharges_vs_gap),
                            gap0,len(gaps),gaps.min(),gaps.max() ) )
        ind0 = np.argmin( (gaps-gap0)**2 )
        self._gap0 = gaps[ind0]
        P0 = Spec.get_probe_at_coord(self._gap0)

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
            PhiMat = Pgap @ dP['Phi'] @ ChargeMat #This is already the Fourier transform of the potential multiplied by q
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
        Brightnesses = Spec.get_eigenbrightness_AWA(Nmodes=Nmodes, recompute=False,
                                                    eigencharges_AWA = eigencharges_vs_gap,
                                                    **brightnesskwargs)[:Nmodes].T #Put coordinate (gaps) axis first
        Poles = Spec.get_eigenrhos_AWA(Nmodes=Nmodes).T #Put coordinate (gaps) axis first
        PsiMats = AWA(PsiMats, axes=[gaps, None,None],
                      axis_names=['gap', 'eigenindex n','eigenindex m'])

        self.kappas = kappas
        self.dkappas = dkappas
        self.Phis0 = Phis0
        self.Phi0Vecs = np.matrix(Phis0).T #column vectors of scalar potential vs. kappa
        self.Brightnesses = Brightnesses
        self.Poles = Poles
        self.PsiMats_ = PsiMats #underscore because method will share same name

        #--- Attach probe with the correctly signed eigencharges
        self.Probe=P0
        self.Probe._eigencharges = eigencharges_vs_gap.cslice[:,gap0,:]

    def __getstate__(self):
        # The primary purpose here is to trim away unnecessary or redundant attributes before pickling

        self.Probe._gapSpectroscopy = None #Spectroscopy data is too enormous to keep, hope it was saved independently

        return self.__dict__

    def get_probe(self): return self.Probe

    filename_template = '(%s)_EncodedEigenfields.pickle'

    def save(self, overwrite=False):  return PCE.save(self, overwrite=overwrite)

    @functools.lru_cache(maxsize=3) #Likely to be called often with the same arguments
    def BVecs(self, *at_coords, extrapolate=True):

        brightnesses_at_coords = self.Brightnesses.interpolate_axis(at_coords, axis=0,
                                                                  extrapolate=extrapolate,
                                                                  bounds_error=(not extrapolate),
                                                                  kind='quadratic')
        self._BVecs = [np.matrix(brightnesses_at_coord).T for
                       brightnesses_at_coord in brightnesses_at_coords] #column vectors
        self._BVecs = np.array(self._BVecs)

        return self._BVecs

    @functools.lru_cache(maxsize=3) #Likely to be called often with the same arguments
    def PsiMats(self, *at_coords, extrapolate=True):

        PsiMats_at_coords = self.PsiMats_.interpolate_axis(at_coords, axis=0,
                                                          extrapolate=extrapolate,
                                                          bounds_error=(not extrapolate),
                                                          kind='quadratic')
        self._PsiMats = [np.matrix(PsiMat_at_coord) for
                         PsiMat_at_coord in PsiMats_at_coords] #column vectors
        self._PsiMats = np.array(self._PsiMats)

        return self._PsiMats

    @functools.lru_cache(maxsize=3) #Likely to be called often with the same arguments
    def PoleMats(self, *at_coords, extrapolate=True):

        Poles_at_coords = self.Poles.interpolate_axis(at_coords, axis=0,
                                                          extrapolate=extrapolate,
                                                          bounds_error=(not extrapolate),
                                                          kind='quadratic')
        self._PoleMats = [np.matrix(np.diag(Poles_at_coord)) for
                         Poles_at_coord in Poles_at_coords] #column vectors
        self._PoleMats = np.array(self._PoleMats)

        return self._PoleMats

    def EradVsGap(self, at_gaps, freq, RMat0=None, Nmodes=None,rp=None,
                  record_rp_vals=False, as_AWA=True, **kwargs):

        #--- Get ingredients and restrict to a narrower range of `Nmodes` if requested
        kappas = self.kappas
        dkappas = self.dkappas
        Phi0Vecs = self.Phi0Vecs #column vectors of scalar field vs kappa

        BVecs = self.BVecs(*at_gaps)
        PsiMats = self.PsiMats(*at_gaps)
        PoleMats = self.PoleMats(*at_gaps)

        if RMat0 is not None: Nmodes = RMat0.shape[0] #If a reflection matrix is provided, it has the number of modes desired
        if Nmodes:
            Phi0Vecs = Phi0Vecs[:Nmodes]
            BVecs = BVecs[:,:Nmodes]
            PsiMats = PsiMats[:,:Nmodes,:Nmodes]
            PoleMats = PoleMats[:,:Nmodes,:Nmodes]

        #--- Get RMat for fields at reference gap
        if RMat0 is None:
            #sometimes `rp` function "leak" into the light cone, this lets us buffer it (we are anticipating imprecision in `rp`!)
            k = 2 * np.pi * freq
            light_cone_buffer = self.Probe.light_cone_buffer # a class attribute
            qs = np.sqrt(k ** 2 * light_cone_buffer + kappas ** 2).real #This will keep us out of the light cone
            if isinstance(rp,Number): rp_vs_q=rp
            else:
                rp_vs_q = rp(freq, qs, **kwargs) #rp will have to anticipate dimensionless arguments
                rp_vs_q[np.isnan(rp_vs_q)+np.isinf(rp_vs_q)] = 0  # This is just probably trying to evaluate at too large a momentum
            if record_rp_vals:
                try: self.rp_vals_vs_q.append(rp_vs_q)
                except AttributeError: pass
            # Here we can see `Phi0Vecs` has column vectors that are the Hankel transform of the potential multiplied by q
            # or, equivalently, they are the Hankel transform of the Ez-eigenfield
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

        if as_AWA: return AWA(Erads, axes=[at_gaps], axis_names=['gap'])
        else: return Erads

    def EradSpectrumDemodulated(self, freqs, rp=None,
                                gapmin=.1, amplitude=2.,
                                Ngaps=24, demod_order=5,
                                record_rp_vals = False,
                                **kwargs):
        T = PCE.Timer()

        #gaps, weights = get_cheb_zs_weights(Ncheb=Ngaps, A=amplitude, gapmin=gapmin, demod_order=demod_order)
        at_gaps,dgaps=numrec.GetQuadrature(xmin=gapmin,xmax=gapmin+2*amplitude,
                                           N=Ngaps,quadrature=numrec.CC)

        if record_rp_vals: self.rp_vals_vs_q=[]

        #--- Execute the loop over frequency values
        EradsVsFreq = []
        if not hasattr(freqs, '__len__'): freqs = [freqs]
        for freq in freqs:
            if self.verbose: Logger.write('\tComputing at freq=%1.3E...'%freq)
            dErad = self.EradVsGap(at_gaps, freq, rp=rp, record_rp_vals=record_rp_vals, **kwargs)
            EradsVsFreq.append(dErad)

        gaps = EradsVsFreq[0].axes[0]
        EradsVsFreq = AWA(EradsVsFreq, axes=[freqs, gaps],
                          axis_names=['Frequency', r'$z_\mathrm{tip}$']).T
        result = dict(Erad=EradsVsFreq)

        # --- Demodulate with chebyshev polynomials
        if demod_order:
            if self.verbose: Logger.write('Demodulating...')
            sns = PCE.demodulate(EradsVsFreq, demod_order=demod_order)
            result['Sn'] = sns

        """sns=[]
        for harmonic in range(demod_order):
            weights = np.cos(harmonic * 2*np.pi * np.linspace(0,.5,len(at_gaps)))
            Sn = np.sum(weights[:,np.newaxis] * EradsVsFreq, axis=0)  # sum on z-axis
            sns.append(Sn)
        result['Sn'] = AWA(sns,axes=[None,freqs],\
                           axis_names=['harmonic','Frequency'])"""

        if self.verbose:  Logger.write('\t' + T())

        return result

    def getNormalizedSignal(self,freqs_wn,rp,
                            a_nm=30,amplitude_nm=50,demod_order=5,
                            Ngaps=24*4,gapmin_nm=.15,
                            L_cm=24e-4,
                            rp_norm = None,
                            norm_single_Freq = True,
                            **kwargs):

        # Adapt dimensional arguments to the same units as the probe discretization
        to_nm = a_nm / self.Probe.get_a(); from_nm = 1/to_nm
        amplitude = amplitude_nm * from_nm
        gapmin = gapmin_nm * from_nm
        q_to_wn = from_nm * 1e7 # converting q values to wavenumbers will be also be done using the known tip dimensionful radius

        # Convert frequency in wavenumbers to internal units
        # Properly wrapped `rp` will just undo this conversion identically, back to wavenumbers
        # But the dimensionless frequencies will be sized according to the indicated probe length `L_cm`
        L = np.ptp(self.Probe.get_zs())  # Height of probe in internal units
        freq_to_wn = L/L_cm
        freqs = freqs_wn / freq_to_wn  # conversion factor from wavenumbers to internal frequency units

        # Wrap the provided rp functions so they can expand out dimensionless frequencies and wavevectors
        # The supplied reflection function should take `frequency (wavenumbers), q (wavenumbers)`
        wrapped_rp = PCE.wrap_rp(rp,freq_to_wn,q_to_wn)

        signals = self.EradSpectrumDemodulated(freqs, rp=wrapped_rp,
                                               gapmin=gapmin, amplitude=amplitude,
                                               Ngaps=Ngaps, demod_order=demod_order,
                                               **kwargs)

        signals['Sn'].set_axes([None,freqs_wn],
                               axis_names=[None,'Frequency (cm$^{-1}$)'])
        signals['Erad'].set_axes([None,freqs_wn],
                                 axis_names=[None,'Frequency (cm$^{-1}$)'])

        # Normalize only if normalization is requested
        if rp_norm is not None:
            if norm_single_Freq: freqs_wn_norm = np.mean(freqs_wn) # a single frequency
            else: freqs_wn_norm = freqs_wn
            freqs_norm = freqs_wn_norm / freq_to_wn

            wrapped_rp_norm = PCE.wrap_rp(rp_norm, freq_to_wn,q_to_wn)
            signals_ref = self.EradSpectrumDemodulated(freqs_norm, rp=wrapped_rp_norm,
                                                       gapmin=gapmin, amplitude=amplitude,
                                                       Ngaps=Ngaps, demod_order=demod_order,
                                                       **kwargs)
            signals['Sn_norm'] = signals['Sn'] / signals_ref['Sn']

            signals['Sn_norm'].set_axes([None,freqs_wn],
                                   axis_names=[None,'Frequency (cm$^{-1}$)'])

        return signals

    def getBasisFourPotentialAtZ(self, *args,**kwargs):

        P = self.Spectroscopy.get_probe_at_coord(self._gap0)
        return P.getFourPotentialAtZ(*args,**kwargs)

    # Tools for quick library-based calculation of signal from 2D material
    def build_Rmat2D_library(self,qps=np.logspace(-3, 2, 100)):

        kappas = self.kappas
        dkappas = self.dkappas
        Phi0Vecs = self.Phi0Vecs  # column vectors of scalar field vs kappa

        #freq = self..Probe.get_freq()
        # k = 2 * np.pi * freq
        # light_cone_buffer = Enc.Probe.light_cone_buffer # a class attribute
        # qs = np.sqrt(k ** 2 + kappas ** 2).real #This will keep us out of the light cone
        qs = kappas # We reasoned that this is the correct operand for 2D material, at least when substrate is Air

        def get_rps2D(q):

            # Substrate part, then 2D part
            return (1) / (1 - q), \
                   (-q) / (1 - q)

        def getRmats(qp):

            nonlocal qs, Phi0Vecs, dkappas
            rp1_vs_q, rp2_vs_q = get_rps2D(qs / qp)
            R1 = Phi0Vecs.T @ np.diag(dkappas * rp1_vs_q) @ Phi0Vecs
            R2 = Phi0Vecs.T @ np.diag(dkappas * rp2_vs_q) @ Phi0Vecs
            return R1, R2

        Nmodes = len(Phi0Vecs.T) # Row vectors of eigenfields
        qps_amp = qps; Npts = len(qps_amp)
        if self.verbose:  Logger.write('Computing library of 2D material reflectance matrices ' \
                                         + 'from %i eigenfields to %i x %i qp values...' % (Nmodes, Npts, Npts))
        qps_phase = np.linspace(0, 2 * np.pi, 2 * Npts + 2)[1:2 * Npts + 1]

        # Holders for numerically computed data
        Rmats2D_subs = np.zeros([len(qps_amp),
                           len(qps_phase),
                           Nmodes, Nmodes],
                          dtype=complex)
        Rmats2D_sigma = np.zeros([len(qps_amp),
                           len(qps_phase),
                           Nmodes, Nmodes],
                          dtype=complex)

        # Compute Rmat for substrate and sigma parts, at each qp
        for i, qp_amp in enumerate(qps_amp):
            for j, qp_phase in enumerate(qps_phase):
                progress = (i*len(qps_phase)+j)/(len(qps_amp)*len(qps_phase))*100
                if self.verbose:  print('\tProgress: %1.2f%%' % progress,
                                     end='\r', flush=True)
                qp = qp_amp * np.exp(1j * qp_phase)
                R1, R2 = getRmats(qp)
                Rmats2D_subs[i, j] = R1
                Rmats2D_sigma[i, j] = R2

        axes = [qps_amp, qps_phase, None, None]
        axis_names = ['Abs(qp)', 'Arg(qp)', None, None]
        Rmats2D_subs = AWA(Rmats2D_subs, axes=axes, axis_names=axis_names)
        self.Rmats2D_subs = Rmats2D_subs.sort_by_axes()
        Rmats2D_sigma = AWA(Rmats2D_sigma, axes=axes, axis_names=axis_names)
        self.Rmats2D_sigma = Rmats2D_sigma.sort_by_axes()

        # Build an interpolator (or, NxN/2 interpolators) based on the numerically computed Rmat values
        from scipy.interpolate import RectBivariateSpline as RBS

        self.Rmat2D_interp_subs_r = {}
        self.Rmat2D_interp_subs_i = {}
        self.Rmat2D_interp_sigma_r = {}
        self.Rmat2D_interp_sigma_i = {}
        pts = (qps_amp,qps_phase)
        interp_kwargs = dict(s=0,kx=3,ky=3) # smoothing factor `s=1` is for interpolation, `kx` and `ky` are degrees of spline
        for i in range(self.Rmats2D_subs.shape[2]):
            for j in range(self.Rmats2D_subs.shape[2]):
                inds = (i, j)
                if j > i: continue
                self.Rmat2D_interp_subs_r[inds] = RBS(*pts, self.Rmats2D_subs[:, :, i, j].real, **interp_kwargs)
                self.Rmat2D_interp_subs_i[inds] = RBS(*pts, self.Rmats2D_subs[:, :, i, j].imag, **interp_kwargs)
                self.Rmat2D_interp_sigma_r[inds] = RBS(*pts, self.Rmats2D_sigma[:, :, i, j].real, **interp_kwargs)
                self.Rmat2D_interp_sigma_i[inds] = RBS(*pts, self.Rmats2D_sigma[:, :, i, j].imag, **interp_kwargs)

    def interpolate_Rmat2D_from_library(self, qp, eps):

        kappa = (eps + 1) / 2
        beta = (eps - 1) / (eps + 1)
        qp_eff = kappa * qp

        q1 = np.abs(qp_eff)
        q2 = np.angle(qp_eff) % (2*np.pi) # We want interpolator "look up" to refer to positive phase values from 0 to 2*pi, as they were recorded in the lookup table

        assert hasattr(self,'Rmats2D_subs') and hasattr(self,'Rmat2D_interp_subs_r') and len(self.Rmat2D_interp_subs_r),\
            'First compute a library of 2D material reflectance matrices at `qps` for this encoding, via `build_Rmat2D_library(qps)`.'

        RMat = np.zeros(self.Rmats2D_subs.shape[2:],
                        dtype=complex)
        for i in range(RMat.shape[0]):
            for j in range(RMat.shape[1]):
                inds = (i, j)
                if j > i: continue

                # Substrate part
                Rsubs = self.Rmat2D_interp_subs_r[inds](q1, q2, grid=False) \
                     + 1j * self.Rmat2D_interp_subs_i[inds](q1, q2, grid=False)
                Rsubs *= beta #This may be zero, if substrate is Air!

                # 2D part
                Rsigma = self.Rmat2D_interp_sigma_r[inds](q1, q2, grid=False) \
                     + 1j * self.Rmat2D_interp_sigma_i[inds](q1, q2, grid=False)

                RMat[i, j] = RMat[j, i] = Rsubs + Rsigma

        return RMat

    def EradDemodulated2DFromLibrary(self,qp,eps,**kwargs):

        RMat0 = self.interpolate_Rmat2D_from_library(qp, eps)
        result = self.EradSpectrumDemodulated(freqs=self.Probe.get_freq(),
                                                RMat0=RMat0, **kwargs)

        return result # Result will have whatever format the `Erad` function uses

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

def find_roots_custom(p, scaling=10):
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

        Brightnesses = GapSpec.get_eigenbrightness_AWA(Nmodes=Nmodes, recompute=False).T  # Put coordinate (gaps) axis first
        self.Rs = Brightnesses**2 #@TODO: Might want to double check this
        self.Ps = GapSpec.get_eigenrhos_AWA(Nmodes=Nmodes).T  # Put coordinate (gaps) axis first

        self.Nmodes = self.Rs.shape[1]
        self.zs = self.Rs.axes[0]

        self.Probe = GapSpec.Probe

        self.build_poles_residues_interpolator(interpolation=interpolation,**kwargs)

    def get_probe(self): return self.Probe

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

    @functools.lru_cache(maxsize=3) #Likely to be called often with the same arguments
    def evaluate_poles(self, *zs, Nmodes=None):

        if not len(zs): zs = self.zs

        # Default to all terms...
        if not Nmodes:
            Nmodes = self.Nmodes

        #Ps = self.Ps[:, :Nmodes].interpolate_axis(zs, axis=0, kind=interpolation,
        #                                          bounds_error=False, extrapolate=True)
        Ps = [interp(zs) for interp in self.Ps_interp[:Nmodes]]

        return np.array(Ps).T #keep z-axis first

    @functools.lru_cache(maxsize=3) #Likely to be called often with the same arguments
    def evaluate_residues(self, *zs, Nmodes=None):

        if not len(zs): zs = self.zs

        # Default to all terms...
        if not Nmodes:
            Nmodes = self.Nmodes

        #Rs = self.Rs[:, :Nmodes].interpolate_axis(zs, axis=0, kind=interpolation,
        #                                          bounds_error=False, extrapolate=True)
        Rs = [interp(zs) for interp in self.Rs_interp[:Nmodes]]

        return np.array(Rs).T #keep z-axis first

    @functools.lru_cache(maxsize=3) #Likely to be called often with the same arguments
    def evaluate_poles_poly(self, *zs, Nmodes=None):

        if not len(zs): zs = self.zs

        # Default to all terms...
        if not Nmodes:
            Nmodes = self.Nmodes

        Ps = np.array([np.polyval(Ppoly, zs) for Ppoly in self.Ppolys[:Nmodes]])

        return Ps.T

    @functools.lru_cache(maxsize=3) #Likely to be called often with the same arguments
    def evaluate_residues_poly(self, *zs, Nmodes=None):

        if not len(zs): zs = self.zs

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
        #Nts=48
        print('Nts=',Nts)
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

        self.zs = zs
        self.ts=ts

        return list(zip(zs, wts))

    def get_Erad_from_nodes(self, beta, nodes, Nmodes=None):

        if not hasattr(beta, '__len__'): beta=[beta]
        if not isinstance(beta,np.ndarray): beta=np.array(beta)
        assert beta.ndim == 1

        # `Frequency` axis will be first
        if isinstance(beta, np.ndarray):
            beta = beta.reshape((len(beta), 1, 1))

        # Weights apply across z-values, second axis will be zs
        zs, ws = list(zip(*nodes))
        ws_grid = np.array(ws).reshape((1, len(ws), 1))
        zs = np.array(zs)

        # Last axis will be eigenmodes, until summed at end

        # Evaluate at all nodal points
        # Add first axis as frequencies
        Rs = self.evaluate_residues(*zs, Nmodes=Nmodes)[np.newaxis,:]
        Ps = self.evaluate_poles(*zs, Nmodes=Nmodes)[np.newaxis,:]

        # Should broadcast over freqs if beta has an additional first axis
        # The offset term is absolutely critical, offsets false z-dependence arising from first terms
        signals = np.sum(Rs * (1 / (beta - Ps) + 1 / Ps) * ws_grid, axis=-1)  # +Rs/Ps,axis=-1)
        # Now the last axis is nodes (z-positions)

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

    def EradSpectrumDemodulated(self,beta, zmin=.01,amplitude=2,
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

    def getNormalizedSignal(self,beta,
                            a_nm=30,amplitude_nm=50,demod_order=5,
                            Ngaps=24*4,gapmin_nm=.15,
                            beta_norm = None,
                            max_harmonic=5,
                            Nmodes=None,
                            **kwargs):

        # Adapt dimensional arguments to the same units as the probe discretization
        to_nm = a_nm / self.Probe.get_a(); from_nm = 1/to_nm
        amplitude = amplitude_nm * from_nm
        gapmin = gapmin_nm * from_nm
        print('amplitude=',amplitude)
        print('gapmin=',gapmin)

        signals={}
        signals['Sn'] = self.EradSpectrumDemodulated(beta, zmin=gapmin,amplitude=amplitude,
                                            max_harmonic=max_harmonic,Nts=Ngaps, Nmodes=Nmodes)

        # Normalize only if normalization is requested
        if beta_norm is not None:

            signals_ref={}
            signals_ref['Sn'] = self.EradSpectrumDemodulated(beta_norm, zmin=gapmin,amplitude=amplitude,
                                                             max_harmonic=max_harmonic,Nts=Ngaps, Nmodes=Nmodes)
            signals['Sn_norm'] = signals['Sn'] / signals_ref['Sn']

        return signals

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

        Rs = self.evaluate_residues(*zs,Nmodes=Nmodes)
        Ps = self.evaluate_poles(*zs,Nmodes=Nmodes)

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
            roots = find_roots_custom(poly, scaling=scaling)

            #from numpy.polynomial import polynomial
            #roots = polynomial.polyroots(poly)

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
                if i<=2: to_minimize = np.abs(target - roots)
                else:
                    raise ValueError('`select_by=%s` not recognized!  Choose "difference", "continuity", or "target".'%select_by)

            self.closest_roots.append( roots[np.argsort(to_minimize)][:5] )

            beta = roots[to_minimize == to_minimize.min()].squeeze()
            betas.append(beta)
            if not i % 5 and self.verbose:
                Logger.write('\tProgress: %1.2f%%  -  Inverted %i signals of %i.' % \
                             (((i + 1) / float(len(signals)) * 100),
                              (i + 1), len(signals)))

        betas = AWA(betas)
        betas.adopt_axes(signals)
        betas = betas.squeeze()
        if not betas.ndim: betas = betas.tolist()

        return betas