# -*- coding: utf-8 -*-

import os
import copy
import pickle
import time
import numpy as np
from collections import UserDict
from numbers import Number
from common.log import Logger
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from scipy.linalg import eig,eigh,solve
from common import numerics
num=numerics
from common.baseclasses import ArrayWithAxes as AWA

from . import RotationalMoM as RotMoM
from .RotationalMoM import *

class Timer(object):

    def __init__(self):

        self.t0=time.time()

    def __call__(self,reset=True):

        t=time.time()
        msg='\tTime elapsed: %s'%(t-self.t0)

        if reset: self.t0=t

        return msg

#2023.05.25 -- Efields in this module are deprecated and moved to `RotationalMoM` module
"""#--- Fields

def EBesselBeamFF(angle=60, k = 2*np.pi*.003, z0=0):
    
    anglerad=np.deg2rad(angle)

    q=k*np.sin(anglerad)
    kz=-k*np.cos(anglerad)
    
    
    def Ez(r,z):
        Exp=np.exp(1j*kz*(z-z0))
        J0=j0(q*r)
        
        return q*J0*Exp/k #normalization by k is to give field unity amplitude

    def Er(r,z):
        
        Exp=np.exp(1j*kz*(z-z0))
        J1=j1(q*r)
        
        return kz*J1*Exp/k
    
    return Er,Ez

def EBesselBeamNF(q,k=2*np.pi*.003):
    
    def Ez(r,z):

        kz=np.sqrt(k**2-q**2)
        
        Exp=np.exp(1j*kz*z)
        J0=j0(q*r)
        
        return q*J0*Exp

    def Er(r,z):

        kz=np.sqrt(k**2-q**2)
        
        Exp=np.exp(1j*kz*z)
        J1=j1(q*r)
        
        return kz*J1*Exp
    
    return Er,Ez"""

#--- Probe class helpder functions

def mirror_double_image(im, axis=0):

    new_axes = im.axes
    ax = new_axes[axis]
    new_axes[axis] = np.append(-ax[::-1], ax)

    slices = [slice(None) for d in range(im.ndim)]
    slices[axis] = slice(None, None, -1)
    new_im = np.concatenate((im[tuple(slices)], im), axis=axis)

    return AWA(new_im, axes=new_axes, axis_names=im.axis_names)

class Efield_generator(object):
    
    def __init__(self,
                 Er_Mat,Ez_Mat,
                 Er_Mat_m,Ez_Mat_m,
                 rs,zs):
        
        self.Er_Mat=np.matrix(Er_Mat)
        self.Ez_Mat=np.matrix(Ez_Mat)
        self.Er_Mat_m=np.matrix(Er_Mat_m)
        self.Ez_Mat_m=np.matrix(Ez_Mat_m)
        self.rs=rs; self.zs=zs
        self.shape0=(len(rs),len(zs))
        
    def __call__(self,charge,rho=0):
        
        #Turn charge into column vector
        if not isinstance(charge,np.matrix): charge = np.matrix(charge).T
        
        Er_flat = np.array(self.Er_Mat @ charge)
        Ez_flat = np.array(self.Ez_Mat @ charge)
        
        Er_flat_m = rho * np.array(self.Er_Mat_m @ charge)
        Ez_flat_m = rho * np.array(self.Ez_Mat_m @ charge)
            
        Er=Er_flat.reshape(self.shape0)
        Ez=Ez_flat.reshape(self.shape0)
        
        Er_m=Er_flat_m.reshape(self.shape0)
        Ez_m=Ez_flat_m.reshape(self.shape0)
        
        Er=AWA(Er+Er_m,axes=[self.rs,self.zs],axis_names=['r','z'])
        Ez=AWA(Ez+Ez_m,axes=[self.rs,self.zs],axis_names=['r','z'])
    
        return Er,Ez
    
def demodulate(signal_AWA,demod_order=5,quadrature=numrec.GL,Nts=None,verbose=False):
    "Demodulate `signal_AWA` along its first axis."

    global signal_vs_z
    
    #--- Define domain `x \in [-1,1]` and fit to Chebyshev polynomials
    zs=signal_AWA.axes[0] #Assume we want to demodulate over entire range
    A=(zs.max()-zs.min())/2
    if Nts is None: Nts=len(zs)*4
    if verbose: Logger.write('Demodulating across A=%1.2G with Nt=%i...'%(A,Nts))
    ts,dts=numrec.GetQuadrature(N=Nts,xmin=-.5,xmax=0,quadrature=quadrature)
    zs_sweep = zs.min() + A*(1+np.cos(2*np.pi*ts)) # positions low to high; if zs were CC quadrature, then these are the same z-values but reordered
    signal_vs_t=signal_AWA.interpolate_axis(zs_sweep,bounds_error=False,extrapolate=True,axis=0,kind='quadratic')

    Sns=[]
    harmonics=np.arange(demod_order+1)
    kernel_shape= (len(ts),)+(1,)*(signal_AWA.ndim - 1)
    for harmonic in harmonics:
        kernel = (np.cos(2*np.pi*harmonic*ts) * dts).reshape( kernel_shape )
        if harmonic: kernel -= np.mean(kernel) # this is crucial because any finite mean in kernel might improperly couple in a large DC into the integrand
        Sn = 4 * np.sum( kernel * signal_vs_t, axis=0)
        Sns.append(Sn)

    #Sns = chebfit(xs,signal_AWA,deg = demod_order)
    
    #--- Prepare axes
    axes=[harmonics] #Demod order will be first axis
    axis_names=['$n$']
    if signal_AWA.ndim>1:
        axes+=signal_AWA[0].axes
        axis_names+=signal_AWA[0].axis_names
    Sns = AWA(Sns,axes=axes,\
             axis_names=axis_names)
    
    return Sns

class _ProbesCollection(UserDict):

    def generate_name(self):

        probe_names = list(self.data.keys())
        k = len(probe_names)+1

        probe_name=None
        while probe_name is None:
            test='probe_%i'%k
            if test in probe_names: k+=1
            else: probe_name=test

        return probe_name

    _overwrite = False

    def overwrite(self, enable=None):

        if enable is not None:
            assert isinstance(enable,bool)
            self._overwrite=enable

        return self._overwrite

    def __setitem__(self, probe_name, probe):

        assert isinstance(probe_name,str)

        # remove probe entry first if we are adding it again with a different name
        keys_to_remove=[]
        for key,val in self.items():
            if val is probe: keys_to_remove.append(key)
        for key in keys_to_remove: self.pop(key)

        # Or Raise error if we already have such key
        if probe_name in self.data:
            if not self.overwrite():
                raise ValueError('A probe named "%s" is already instantiated!  If you wish to overwrite this probe, '%probe_name+\
                                 'remove it from the collection of probes by using the `del` command!')
            else: Logger.write('Overwriting registered probe "%s"...'%probe_name)
        else: Logger.write('Registering probe "%s"...'%probe_name)

        self.data[probe_name] = probe

ProbesCollection=_ProbesCollection()

PC = ProbesCollection # A short alias
##############################
# --- Save / load facilities.
# Any object can utilize these, who defines attribute `filename_template` and method `get_probe()`
#############################

probe_models_dir = os.path.join(os.path.dirname(__file__),'Probe models')
if not os.path.exists(probe_models_dir): os.mkdir(probe_models_dir)

def get_filepath(probe, cls): #Get the filepath for object of `cls` associated with `probe`

    if isinstance(probe,Probe): probe_name = probe.get_name()
    else:
        assert isinstance(probe,str)
        probe_name = probe
    filename = cls.filename_template % probe_name
    filepath = os.path.join(probe_models_dir, filename)

    return filepath

def save(obj, overwrite=False):

    if hasattr(obj,'get_probe'): probe = obj.get_probe()
    else:
        assert isinstance(obj,Probe)
        probe=obj

    filepath = get_filepath(probe, obj)

    if os.path.exists(filepath) and not overwrite:
        raise OSError('File "%s" exists!  Set `overwrite=True` to do so.' % filepath)

    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
        print('Successfully saved to file "%s"!' % filepath)

def load(probe, cls, overwrite_probe=False): #Load the object of `cls` associated with `probe`

    filepath = get_filepath(probe,cls)
    print('Attemping to load from file "%s"...'%filepath)

    prev_val = ProbesCollection._overwrite
    if overwrite_probe: #(temporarily) set to overwrite if commanded
        ProbesCollection.overwrite(True)

    try_path = os.path.join(probe_models_dir,filepath)
    if not os.path.exists(filepath) and os.path.exists(try_path):
        filepath = try_path

    try:
        with open(filepath,'rb') as f:
            obj = pickle.load(f)
            print('Successfully loaded from file "%s"!'%filepath)
        ProbesCollection.overwrite(prev_val)
    except ValueError as e:
        ProbesCollection.overwrite(prev_val) # Whether error or not, make sure previous overwrite status is restored
        msg = e.args[0]
        msg += '  You may avert this error by passing keyword `overwrite=True` to `load`!'
        raise ValueError(msg)

    return obj

def wrap_rp(rp,freq_to_wn,q_to_wn):
    """Wrap `rp` so that it accepts dimensionless arguments, and converts them to wavenumbers,
    before feeding to the underlying `rp` function."""

    def wrapped_rp(freq,q):

        freq_wn = freq * freq_to_wn
        q_wn = q * q_to_wn
        #print('min k (wn): %1.3f'%np.min(2*np.pi*freq_wn)) #Debug messages to ensure we are outside the light cone
        #print('min q (wn): %1.3f'%np.min(q_wn))

        return rp(freq_wn,q_wn)

    return wrapped_rp
    
#--- Probe classes

class Probe(object):

    defaults = {'freq': 30e-7 / 10e-4,
                     'a': 1,
                     'gap': 1,
                     'L': 19e-4 / 30e-7,
                     'illum_angles': np.linspace(45, 90, 20),
                     'excitation': EPlaneWaveFF}

    # The data type to use for the serious calculations
    # Helps to NOT use np.complex128, because that is more expensive
    dtype = np.complex64
    
    def __init__(self,zs=None,rs=None,
                 Nnodes=244,L=None,quadrature=numrec.GL,
                 a=None,taper_angle=20,geometry='hyperboloid',Rtop=0,\
                 freq=None,gap=None,Nsubnodes=2,closed=False,
                 name=None,**kwargs):

        self.set_name(name)
        Logger.write('Generating probe "%s"...'%self.get_name())

        ## Set up defaults
        self.defaults = {'freq': 30e-7 / 10e-4,
                            'a': 1,
                            'gap': 1,
                            'L': 19e-4 / 30e-7,
                            'illum_angles': np.linspace(45, 90, 20),
                             'excitation':EPlaneWaveFF}
        
        if a is None: a=self.defaults['a']
        if L is None: L=self.defaults['L']
        self._a=a
        if zs is None:
            zs,wzs=numrec.GetQuadrature(Nnodes,xmin=0,xmax=L,quadrature=quadrature)
            zs -= zs.min()
            #zs += wzs[0]/2 #Bring the "zero" coordinate where it belongs before evaluating geometry

        if rs is not None: assert len(rs)==len(zs)
        else:
            if isinstance(geometry,str):
                rs=get_BoR_radii(zs, L=L, z0=0, a=a, \
                                 taper_angle=taper_angle, \
                                 geometry=geometry, Rtop=Rtop)
            else:
                assert hasattr(geometry,'__call__')
                rs=geometry(zs,L=L,a=a,**kwargs)

        self.MethodOfMoments = RotMoM.BodyOfRevolution(rs, zs, closed=closed, \
                                                       Nsubnodes=Nsubnodes, display=True)
        self.MoM=self.MethodOfMoments

        self._eigenrhos=None
        self._eigencharges=None
        self._eigenexcitations=None
        self._kappa_min_factor=.1 #This will already put qs very close to the light line:  `qs/k ~ 1 + (factor)**2/2`
        self._kappa_max_factor=20 # This will make maximum kappa value in quadratures at least 20 inverse probe radii
        
        self._ZSelf=None
        self._ZMirror=None
        self._gapSpectroscopy=None

        self.set_freq(freq)
        self.set_gap(gap)

        self.all_rp_vals=[]

    def get_name(self):
        try: return copy.copy(self._name)
        except: return None

    def set_name(self,name=None,overwrite=False):

        # If no name is provided, keep existing one (which may itself be `None`)
        if name is None: name = self.get_name()
        # If still none, generate one
        if name is None: name = ProbesCollection.generate_name()

        # Re-register probe if overwriting, or if not existent in collection
        if name not in ProbesCollection: ProbesCollection[name] = self
        elif overwrite: # Overwrite if called for (the default when using `__setstate__`)
            overwrite_prev_set = ProbesCollection.overwrite()
            ProbesCollection.overwrite(True)
            ProbesCollection[name] = self
            ProbesCollection.overwrite(overwrite_prev_set)

        self._name = name

    def __setstate__(self, state):
        # The primary purpose here is to rename legacy attributes upon unpickling
        self.__dict__.update(state)

        newattrs = {'MethodOfMoments':'Discretization',
                    'MoM':'Discretization',
                    '_name':None}
        for dest,src in newattrs.items():
            if not hasattr(self, dest):
                if isinstance(src,str): #interpret src as a key to write from
                    if src in state:  setattr(self,dest, state[src])
                else:
                    setattr(self,dest,src)

        # Register in Probes collection
        self.set_name(overwrite=False) # Don't overwrite unless `ProbesCollections` was otherwise set to `overwrite=True`

    filename_template = '(%s)_Probe.pickle'

    def save(self,*args,**kwargs): return save(self,*args,**kwargs)
        
    def get_a(self): return self._a

    def reset_eigenproperties(self):

        if np.any(self._eigenrhos): Logger.write('Resetting eigenproperties...')

        self._eigenrhos=None
        self._eigencharges=None
        self._eigenexcitations=None
        self._eigenbrightness=None
    
    def set_freq(self,freq=None):

        if freq is None: freq=self.defaults['freq']
        try:
            if freq == self.get_freq(): return # Do nothing
        except AttributeError: pass
                 
        self._freq=freq
        self._ZSelf=None #Re-set the self interaction
        self.reset_eigenproperties()

    def set_gap(self,gap=None):
        
        if gap is None: gap=self.defaults['gap']
        try:
            if gap == self.get_gap(): return # Do nothing
        except AttributeError: pass
                 
        self._gap=gap
        self._ZMirror=None #Re-set the mirror interaction
        self.reset_eigenproperties()
        
    def get_freq(self): return copy.copy(self._freq)
    
    def get_k(self): return 2*np.pi*self.get_freq()
    
    def get_kappa_min(self):

        kappa_L = 1/np.ptp(self.get_zs()) * self._kappa_min_factor

        return kappa_L
        # Disable k-dependence for now, it creates problems when comparing absolute scales of integrated results at different frequencies
        # especially for rp values that are large for small values of kappa (near the light cone)
        #kappa_k = 0 #k * self._kappa_min_factor

        # return whichever of the two is larger (not to undersample the probe length!)
        #return np.max( (kappa_L, kappa_k) )

    def get_kappa_max(self):

        kappa_a = 1/self.get_a()

        return kappa_a * self._kappa_max_factor
    
    def get_gap(self): return copy.copy(self._gap)
        
    def get_zs(self): return copy.copy(self.MethodOfMoments.node_zs)

    def _toAWA(self,charge,with_gap=False):
        """Works even on input column vectors."""

        charge=np.asarray(charge).squeeze()
        zs=self.get_zs()
        if with_gap: zs+=self.get_gap()

        return AWA(charge,axes=[zs],axis_names=['z'])

    def get_weights(self): return copy.copy(self.MethodOfMoments.node_dts)

    def get_weights_matrix(self): return np.matrix(np.diag(self.get_weights()))
        
    def get_radii(self):

        Rs = copy.copy(self.MethodOfMoments.node_Rs)
        return self._toAWA(Rs)

    def get_dRdz(self):

        dR = np.gradient(self.get_radii())
        dz = np.gradient(self.get_zs())

        return dR/dz

    def plot_geometry(self,**kwargs):

        zsprobe = self.get_zs()
        rsprobe = self.get_radii()

        # -- Plot probe shape

        plt.figure()
        # plt.plot(-radii,zs,radii,zs,color='b')
        xsprobe = np.append(-rsprobe[::-1], rsprobe)
        ysprobe = np.append(zsprobe[::-1], zsprobe)
        plt.fill_between(x=xsprobe,
                         y1=ysprobe, y2=ysprobe.max(),
                         color='gray', edgecolor='k', lw=2)
        plt.plot(xsprobe, ysprobe, marker='o', ls='', color='k',**kwargs)
        plt.gca().set_aspect('equal')

        plt.xlabel('Radial coordinate')
        plt.ylabel('Z')
        plt.title(self.get_name())

    #--- Momentum space propagators & reflectance

    def getFourPotentialPropagators(self,k=None,farfield=False,
                                 kappa_min=None,kappa_max=None,Nkappas=244,
                                 qquadrature=numrec.GL,recompute=False):
        """
        Outputs a propagator for the scalar potential and for
        r- and z-components of the vector potential down to the plane at the tip apex.
        """

        if not recompute:
            try: return self._fourpotentialpropagators
            except AttributeError: pass

        # Infer what the user wants if no k-value is provided
        if k is None:
            if farfield: k = self.get_k() #We clearly need to choose a finite k-value
            else: k=0 #Completely quasi-static treatment of the fields (otherwise, it doesn't make much sense to have finite-k and to leave out far-field)

        Logger.write('Computing field propagators at k=%1.2G...'%k)

        #--- Get geometric parameters
        zs=self.get_zs(); Zs=zs[np.newaxis,:]
        rs=self.get_radii(); Rs=rs[np.newaxis,:]
        #dRdt = np.gradient(rs) / self.D.node_dts
        #Sin = dRdt[np.newaxis,:]
        #Cos = np.sqrt(1-Sin**2)
        Sin= self.MoM.node_sin[np.newaxis, :]
        Cos= self.MoM.node_cos[np.newaxis, :]
        Dts= self.MoM.node_dts[np.newaxis, :]

        #--- Prepare Quadrature Grid
        if kappa_min is None: kappa_min=self.get_kappa_min()
        if kappa_max is None: kappa_max=self.get_kappa_max()
        Logger.write('\tIncluding evanescent (near-field) waves..')
        self.kappas,self.dkappas=numrec.GetQuadrature(xmin=kappa_min,
                                            xmax=kappa_max,
                                            N=Nkappas,
                                            quadrature=qquadrature)
        if farfield and k!=0:
            Logger.write('\tIncluding propogating (far-field) waves..')
            qs,dqs=numrec.GetQuadrature(xmin=0,xmax=k,
                                            N=Nkappas,
                                            quadrature=qquadrature)
            kzs=np.sqrt(k**2-qs**2)
            self.kappasFF=-1j*kzs
            self.dkappasFF=1j*qs/kzs*dqs
            self.kappas=np.append(self.kappasFF,self.kappas) #Prepend our far-field bit to kappas
            self.dkappas=np.append(self.dkappasFF,self.dkappas) #Prepend the measure of our far-field bit

        qs=np.sqrt(self.kappas**2+k**2).real.astype(float) #Will always be real
        Kappas=self.kappas[:,np.newaxis]
        Qs=qs[:,np.newaxis]

        #--- Handle the exponential z-dependence
        Exp=np.exp(-Kappas*Zs)

        J0=j0(Qs*Rs)
        J1=j1(Qs*Rs)

        #--- Coulomb propagator
        PCoul = (Cos*Kappas*J0 + Sin*Qs*J1) * Exp * Dts #post-factor would be `J0(q*rho)`
        #PCoul = (2*np.pi * 1j/k) * np.matrix( -(Cos*Kappas - Sin*Qs) * Exp * Dts) #post-factor would be `J0(q*rho)`
        #PCoul = (2 * np.pi * 1j / k) * np.matrix(
        #    -(Sin * Kappas + Cos * Qs) * Exp * Dts)  # post-factor would be `J0(q*rho)`

        #--- Faraday propagator - include only if we want to invoke propagation of far-fields
        if k==0 or not farfield:
            PAz = PAr = np.zeros(PCoul.shape)
        else:
            #--- Az propagator
            PAz = 1j*k * ( +Cos*J0 * Exp * Dts) #post-factor would be `J0(q*rho)`

            #--- Ar propagator
            PAr = 1j*k * ( +Sin*J1 * Exp * Dts) #post-factor would be `J1(q*rho)`

        self._fourpotentialpropagators = dict(Phi=PCoul,
                                                Az=PAz,
                                                Ar=PAr,
                                                kappas=self.kappas,
                                                dkappas=self.dkappas)


        #These should be integrated w.r.t. `dkappas`, and with
        # mirror signatures (Phi,Ar,Az) -> (-,-,+)
        return self._fourpotentialpropagators

    def getRp(self,freq,qs,recompute=True,rp=None,\
              remove_nan=True,**kwargs):
        """Compute rp at `freq,qs` from a provided or stored `rp` function or AWA."""

        if not recompute:
            try: return self.rpvals
            except AttributeError: pass

        if rp is None: rp = lambda freq,q: 1+0j

        if hasattr(rp,'__call__'): rpvals=rp(freq,qs,**kwargs)
        elif isinstance(rp,AWA):
            if rp.ndim==2: rpvals=rp.cslice[freq]
            else: rpvals=rp
            assert len(rpvals)==len(qs)
        elif isinstance(rp,Number): rpvals=np.array([rp]*len(qs))
        else: rpvals = rp

        if remove_nan: rpvals = np.where(np.isfinite(rpvals),rpvals,0)

        self.rpfreq = freq
        self.rpqs = qs
        self.rpvals=AWA(rpvals,axes=[qs],axis_names=['q'])
        self.all_rp_vals.append(self.rpvals)

        return rpvals

    # sometimes `rp` output "leaks" beyond light cone, this lets us buffer it (we are anticipating imprecision in `rp`!)
    light_cone_buffer = 2

    def getRpGapPropagator(self,rp_freq,kappas,dkappas,
                           gap,rp=None,recompute_rp=True,**kwargs):

        #deduce qs from kappa and freq - inexpensive, and also helps us avoid the light-line!
        k = 2*np.pi*rp_freq
        self.rpk = k
        self.rpkappas = kappas
        qs=np.sqrt(k**2 * self.light_cone_buffer + kappas**2).real.astype(float) #This will ensure we are always evaluating outside the light cone if `kappas>0`
        rpvals=self.getRp(rp_freq,qs,rp=rp,recompute=recompute_rp,**kwargs)

        # if `kappas`, then `kzs=1j*kappas`
        return rpvals*np.exp(-2*kappas*gap)*dkappas

    def getSommerfeldMirrorImpedance(self,k=None,gap=None,
                                       recompute_rp=True,rp=None,rp_freq=None,
                                       recompute_propagators=False,farfield=False,
                                       qquadrature=numrec.GL,Nkappas=244,
                                       kappa_min=None,kappa_max=None,
                                       **kwargs):

        #--- Ingredients for sommerfeld integral
        dP=self.getFourPotentialPropagators(k,recompute=recompute_propagators,
                                             farfield=farfield,
                                             qquadrature=qquadrature,Nkappas=Nkappas,
                                              kappa_min=kappa_min,kappa_max=kappa_max,
                                            **kwargs)
        if gap is None: gap=self.get_gap()
        if rp_freq is None: rp_freq = self.get_freq()
        rpP = self.getRpGapPropagator(rp_freq,kappas=dP['kappas'],dkappas=dP['dkappas'],\
                                       gap=gap,rp=rp,recompute_rp=recompute_rp).astype(self.dtype)
        rpP = rpP[:,np.newaxis]

        MPhi = dP['Phi'].astype(self.dtype)
        MAr = dP['Ar'].astype(self.dtype)
        MAz = dP['Az'].astype(self.dtype)

        # Mirror sign applies to Coulomb and Ar fields
        # other -1 signs follow from definition of generalized reflectance
        mirror_sign = self.dtype(-1)
        Zmirror =  MPhi.T @ (mirror_sign * rpP * MPhi)
        #if k!=0:
        if farfield:
            Zmirror = Zmirror + \
                     -MAr.T @ (mirror_sign * rpP * MAr) \
                        - MAz.T @ (rpP * MAz)

        return Zmirror
        
    def get_self_impedance(self,k=None,recompute=False,**kwargs):
        
        if recompute or self._ZSelf is None:

            if k is None: k=self.get_k()
            self._ZSelf = self.MethodOfMoments.get_self_impedance(k, **kwargs)
            
        return copy.copy(self._ZSelf)

    get_self_interaction = get_self_impedance

    def get_mirror_impedance(self,k=None,recompute=False,\
                             sommerfeld=False,rp=None,**kwargs):
        
        if recompute or self._ZMirror is None:

            if k is None: k=0 #Let's default to a positive definite mirror interaction #k=self.get_k()
            gap=self.get_gap()

            if sommerfeld:
                self._ZMirror = self.getSommerfeldMirrorImpedance(k, gap=gap, \
                                                                   recompute_rp=True, rp=rp, \
                                                                   recompute_propagators=True, \
                                                                   **kwargs)
            else:
                self._ZMirror = self.MethodOfMoments.get_mirror_impedance(k, gap=gap, **kwargs)
            
        return copy.copy(self._ZMirror)

    get_mirror_interaction = get_mirror_impedance
    
    #--- Direct solution for induced charge
    
    def get_impedance_matrix(self,rho=0,**kwargs):
        
        Z = self.get_self_impedance(**kwargs)
        
        if rho:
            ZM = self.get_mirror_impedance(**kwargs)
            Z = Z + rho*ZM
            
        return Z

    get_scattering_matrix = get_impedance_matrix

    #--- Charge characterization tools
    
    def get_charge_overlap(self, Q1, Q2):
        
        if not isinstance(Q1, np.matrix): Q1 = np.matrix(Q1).T
        if not isinstance(Q2, np.matrix): Q2 = np.matrix(Q2).T

        ZM = self.get_mirror_impedance()
        
        #If ZM is a definite matrix, this inner product should be nonzero only for `ec2=ec1`
        N2=np.array(Q1.T @ -ZM @ Q2).squeeze()
        
        return complex(N2)
    
    def get_field_overlap(self, Q, excitation, Vt=None, **kwargs):
        """If Q is more than one charge vector, provide them as row vectors."""

        if Vt is None: Vt,Vphi = self.MethodOfMoments.get_excitation(excitation,**kwargs)
        # otherwise assume V is a vector

        if not isinstance(Q, np.matrix): Q=np.matrix(Q).T
        
        overlap = Q.T @ Vt #V carries the annular weights
        
        #try to collapse to single number
        try: return complex(overlap)
        except TypeError: return np.array(overlap).squeeze() #squeeze down vectors
    
    def get_charge_density(self,Q):

        Rs = self.get_radii()
        dts = self.MethodOfMoments.node_dts
        sigma = np.gradient(Q)/dts / (2*np.pi*Rs) #This formula uses Q as the accumulated charge at lozation z

        return self._toAWA(sigma)

    def get_current_density(self,Q): #this will be 1/c times current density (units of Q/m^2)

        k = self.get_k()
        Rs = self.get_radii()
        j = 1j*k*Q / (2*np.pi*Rs)

        return self._toAWA(j)
        
    #--- Charge post-processing tools
        
    def normalize_charge(self,Q,as_vector=False):
        
        N2 = self.get_charge_overlap(Q,Q)
        
        # normalization factor will in general be complex, but determined only up to +/- 1- how to choose?
        Q_norm = Q/np.sqrt(N2)
        
        if as_vector: return Q_norm
        
        return self._toAWA(Q_norm)

    # Experimental, should work but somehow there's a subtle instability (eigenmodes get missed!)
    def solve_eigenmodes(self,recompute_impedance=False,
                            condition_ZS=False,\
                            condition_ZM=True,\
                            ZMthresh=0,\
                            ZSthresh=0,\
                            basis = None,
                           plot=True):

        self.reset_eigenproperties()

        ZS = self.get_self_impedance(recompute=recompute_impedance)
        ZM = self.get_mirror_impedance(recompute=recompute_impedance)

        # --- Conditioning of ZS
        if condition_ZS:
            ea, va = eigh(-ZS.imag) # This operator should be positive definite
            if plot:
                plt.figure()
                plt.plot(ea, marker='o')
                plt.gca().set_yscale('symlog', linthresh=1e-20)
                plt.axhline(ZSthresh, ls='--')
                plt.title('Pos. def. spectrum: `-Im(self impedance)`')
            where_good = ea/ea.real.max() > ZSthresh
            ea = ea[where_good]
            va = va[:, where_good]
            ZS = ZS.real + 1j * va @ np.diag(-ea) @ va.T

        #--- Default is a projective calculation, which will default to full-rank
        V = np.eye(len(ZM)); a=None; b=None

        # --- Conditioning of ZM
        if condition_ZM:
            #ZM = -numrec.nearestPD(-ZM)
            eb, Vb = eigh( -ZM.real ) #This operator should be positive definite

            if plot:
                plt.figure()
                plt.plot(eb.real / eb.real.max(), marker='o', color='r')
                plt.axhline(ZMthresh, ls='--')
                plt.title('Pos. def. spectrum: `-Re(mirror impedance)`')
                plt.gca().set_yscale('symlog',linthresh=1e-14)

            where_ok = eb / eb.real.max() > ZMthresh
            #eb[~where_ok] = eb[where_ok].min()
            #vb = np.eye(len(ZM))
            eb = eb[where_ok]
            Vb = Vb[:, where_ok]

            b = np.diag(eb)
            ZM = Vb @ -b @ Vb.T #Calculate the explicit mirror impedance because we'll at the very least store it away
            self._ZMirror = ZM # We have modified, so now store away; Don't modify self impedance, we need to keep one of them full rank

            # so now, PhiM_cs = vb @ -eb @ vb.T; we use P=vb @ vb.T projector
            # and projected we have: PhiS_cs = vb @ vb.T @ PhiS_cs @ vb @ vb.T
            # Hit (vb.T .. vb) to both and we have:
            # PhiM = -eb
            # PhiS = vb.T @ PhiS_cs @ vb
            # These are now in the basis of (csb = cs @ vb)
            # plt.matshow((csb.T @ W @ csb).real)
            V = Vb # This will be our orthogonal basis for generalized eigenvalue problem

        if np.any(basis):
            basis = np.array(basis)
            assert basis.ndim == 2
            # Verify basis is orthogonal, or else we'll be in trouble
            from numpy.linalg import qr
            Q, R = qr(basis.T) # take column vectors of basis as target for orthogonalization
            tol = 1e-9
            indep = np.abs(np.diag(R)) > tol
            V = Q[:, indep]  # Should now be (a portion of) orthonormal matrix
            print('Basis has length %i'%V.shape[1])
            b = None # assert we have to recompute `b` now

        #--- Prepare matrices for generalized eigenvalue problem
        if a is None:  a = V.T @ ZS @ V
        if b is None:  b = V.T @ -ZM @ V

        rhos, c = eig(a=a, b=b)  # `c` will have column vectors of length `Nmodes`
        Qvecs = V @ c

        #-- Filtering
        rhothresh = 1e-2
        where_ok = (np.abs(rhos) > rhothresh) * np.isfinite(rhos)
        rhos = rhos[where_ok]
        Qvecs = Qvecs[:,where_ok] #project back from restricted basis

        #--- Normalization
        # Make sure to normalize w.r.t. conditioned operator, NOT original one
        #Qs = [self.normalize_charge(Q) for Q in Qs]
        N2 = np.diag(Qvecs.T @ -ZM @ Qvecs)
        Qvecs = Qvecs @ np.diag(1/np.sqrt(N2))
        Qs = np.array(Qvecs).T

        #-- Sort
        sorter = lambda a: np.abs(a)
        tup_sorter = lambda tup: sorter(tup[0])
        rhos, Qs = zip(*sorted(zip(rhos, Qs),
                               key=tup_sorter))
        rhos = np.array(rhos)
        Qs = np.array(Qs)
        rhos = AWA(rhos, axes=[None], axis_names=['eigenindex'])
        Qs = AWA(Qs, axes=[None, self.get_zs()], axis_names=['eigenindex', 'z'])

        self._eigenrhos = rhos
        self._eigencharges = Qs

        return rhos, Qs

    def get_Nmodes(self): return len(self.get_eigenrhos())
        
    def get_eigenrhos(self): return copy.copy(self._eigenrhos)
            
    def get_eigenrho(self,eigenindex):
        
        return self.get_eigenrhos()[eigenindex]
            
    def plot_eigenrhos(self,xmax=20,ymax=1e6,f=None):
        """Use this to plot the computed eigenreflectances and
        check for any unjustifiably "physical" values."""
        
        rhos=self._eigenrhos
        assert rhos is not None,'Run `solve_eigenmodes` or solve_eigenmodes0` first!'
        
        from matplotlib import pyplot as plt
         
        if f is None: plt.figure()
        np.abs(rhos).plot(marker='o',color='k',label=r'Abs($\rho$)')
        np.real(rhos).plot(marker='o',color='r',label=r'Re($\rho$)')
        np.imag(rhos).plot(marker='o',color='b',label=r'Im($\rho$)')
        plt.xlim(-.5,xmax)
        plt.ylim(-ymax,ymax)
        plt.gca().set_yscale('symlog',linthresh=1e-1)
        plt.legend()
        plt.axhline(0)
        plt.title(self.get_name())
        plt.xlabel('Min. eigenvalue: %1.2E+%1.2Ej'%(rhos[0].real,rhos[0].imag))
    
    def get_eigencharge(self,eigenindex,as_vector=False):
        """Return an eigencurrent associated with `eigenindex`, optionally
        in a post-processed form.  (Already normalized.)"""
        
        Qs=self._eigencharges
        assert Qs is not None,'Run `solve_eigenmodes` first!'
        
        Q = copy.copy(Qs[eigenindex])
        
        #If we want column vectors
        if as_vector: Q=np.matrix(Q).T
        
        return Q

    def get_eigencharges(self,as_vector=False):

        Qs = copy.copy(self._eigencharges)

        #If we want column vectors
        if as_vector: Qs=np.matrix(Qs).T

        return Qs.astype(self.dtype)

    def get_eigencharge_density(self,eigenindex,as_vector=False):

        Q = self.get_eigencharge(eigenindex)
        sigma = self.get_charge_density(Q)

        if as_vector: sigma = np.matrix(sigma).T

        return sigma

    def get_eigencharge_densities(self,as_vector=False):

        Qs = np.array(self.get_eigencharges())
        sigmas = [self.get_charge_density(Q) for Q in Qs]

        if as_vector: return np.matrix(sigmas).T

        return AWA(sigmas, adopt_axes_from=Qs)

    def get_eigencurrent_density(self,eigenindex,as_vector=False):

        Q = self.get_eigencharge(eigenindex)
        j = self.get_current_density(Q)

        if as_vector: j = np.matrix(j).T

        return j

    def get_eigencurrent_densities(self,as_vector=False):

        Qs = np.array(self.get_eigencharges())
        js = [self.get_current_density(Q) for Q in Qs]

        if as_vector: return np.matrix(js).T

        return AWA(js, adopt_axes_from=Qs)
    
    def get_eigenexcitations(self,Exc=None,Vt=None,\
                             recompute=False):
        
        if recompute or self._eigenexcitations is None:
            
            Vns=[self.get_field_overlap(Q, Exc, Vt=Vt)
                 for Q in np.array(self.get_eigencharges())]
                
            self._eigenexcitations=np.array(Vns)
            
        return self._eigenexcitations

    def get_brightness(self, Q, k=None, illum_angles=None,
                       illum_angle_weights=None,**kwargs):

        if k is None: k =self.get_k()
        if illum_angles is None: illum_angles = self.defaults['illum_angles']

        Rns = []
        if not hasattr(illum_angles, '__len__'): illum_angles = [illum_angles]
        if illum_angle_weights is None: illum_angle_weights = np.ones((len(illum_angles),))
        elif not hasattr(illum_angle_weights,'__len__'):
            illum_angle_weights=[illum_angle_weights]*len(illum_angles)
        for angle,w in zip(illum_angles,illum_angle_weights):
            Exc = self.defaults['excitation'](angle=angle, k=k)
            Rn = self.get_field_overlap(Q, Exc, **kwargs)
            Rns.append(Rn*w)

        axes=[illum_angles]
        axis_names=['Angle (deg.)']
        if hasattr(Rn,'__len__'): #In case we received column vectorS for charge
            axes+=[None]
            axis_names += [None]

        Rns = AWA(Rns, axes=axes,
                  axis_names=axis_names)

        #2023.05.23 - Remove averaging altogether; it only adds complexity- we will average at runtime
        """if average:
            # 2023.05.21 - Sin factor was removed, because focusing optics should equally collimate entire collection range
            #if Rns.ndim==2: angles=angles[:,np.newaxis] #just so we broadcast over a "hidden" axis
            #Rns = np.mean(np.sin(np.deg2rad(angles)) * Rns, axis=0)
            Rns = np.mean(Rns, axis=0)"""

        return Rns
    
    def get_eigenbrightness(self, k=None, illum_angles=None,
                            illum_angle_weights=None,
                            Nmodes=20,recompute=False,
                            **kwargs):
        """This is the same as `get_eigenexcitations` except we build the plane wave at each of the `illum_angles` automatically."""

        if k is None: k =self.get_k()
        if illum_angles is None: illum_angles = self.defaults['illum_angles']
        
        if not recompute:
            try:
                Bs=self._eigenbrightness
                if Bs.shape[1]<Nmodes: raise AttributeError # We have reason to compute more modes
                return Bs
            except AttributeError: pass
        Logger.write('Computing eigenbrightness at k=%1.2f...'%k)

        Qs = np.array(self.get_eigencharges())[:Nmodes]
        Bs=[]
        if not hasattr(illum_angles, '__len__'): illum_angles=[illum_angles]
        if illum_angle_weights is None: illum_angle_weights = np.ones((len(illum_angles),))
        elif not hasattr(illum_angle_weights,'__len__'):
            illum_angle_weights=[illum_angle_weights]*len(illum_angles)
        for angle,w in zip(illum_angles,illum_angle_weights):
            
            Exc=self.defaults['excitation'](angle=angle, k=k)
            Rn=self.get_field_overlap(Qs,Exc,**kwargs) #Vectorized over all charge vectors
            Bs.append(Rn*w)
            
        Bs=AWA(Bs, axes=[illum_angles, None],
                axis_names=['Angle (deg.)','eigenindex'])

        #2023.05.23 - Remove averaging altogether; it only adds complexity- we will average at runtime
        """if average:
            # 2023.05.21 - Sin factor was removed, because focusing optics should equally collimate entire collection range
            #Rns=np.mean(np.sin(np.deg2rad(angles[:,np.newaxis])) * Rns, axis=0)
            Rns=np.mean(Rns, axis=0)"""
            
        self._eigenbrightness=Bs
        
        return self._eigenbrightness

    get_eigenreceptivity = get_eigenbrightness #a legacy alias
    
    def get_eigenamplitudes(self,Exc=None,Vt=None,rho=0):
        
        Vns=self.get_eigenexcitations(Exc=Exc,Vt=Vt,\
                                      recompute=True)
        
        amps=[]
        for n in range(self.get_Nmodes()):
            eigenrho=self.get_eigenrho(n)
            amps.append(Vns[n]/(rho-eigenrho))
            
        return AWA(amps,axes=[None],axis_names=['Eigenindex'])

    #--- Charge solutions

    def solve_induced_charge_direct(self,excitation=None,Vt=None,\
                                     rho=0,\
                                     as_vector=False,\
                                     **kwargs):

        if Vt is None: Vt,Vphi = self.MethodOfMoments.get_excitation(excitation)

        Z = self.get_impedance_matrix(rho=rho,**kwargs)

        #--- Solve linear system
        Q=solve(Z, -Vt)
        Q=np.matrix(Q) #somehow matrix type is not default output of `solve`..

        if as_vector: return Q

        Q = self._toAWA(Q)

        return Q
    
    def solve_induced_charge_eigen(self,excitation=None,\
                                     Vt=None,rho=0,\
                                     Nmodes=20,Veff=True):

        if Veff:
            Q0 = self.solve_induced_charge_direct(excitation, Vt=Vt, rho=0)
            Zmirror = self.get_mirror_impedance()
            Vt = np.array(Zmirror @ np.matrix(Q0).T).squeeze()
        else:
            if Vt is None: Vt,Vphi = self.MethodOfMoments.get_excitation(excitation)
            Q0=0

        self.eigenamplitudes=self.get_eigenamplitudes(Vt=Vt,rho=rho)[:Nmodes]
            
        Qs=[self.get_eigencharge(n) \
                 for n in range(len(self.eigenamplitudes))]
        
        dQ=np.sum([amp*Q for amp,Q \
                       in zip(self.eigenamplitudes,Qs)],axis=0)
        if Veff: dQ*=rho
        
        Q=self._toAWA(dQ+Q0)
        
        return Q

    #--- Real-space field calculators
    
    def getFourPotentialAtZ(self,rs=np.logspace(-1,2,100),z=0,\
                            farfield=False,k=None,\
                            kappa_min=None,kappa_max=None,Nkappas=244,
                             qquadrature=numrec.GL):
        
        #--- Obtain propagators
        gap=self.get_gap()
        assert z<gap
        if k is None: k = self.get_k()
        dP = self.getFourPotentialPropagators(farfield=farfield,k=k,
                                         kappa_min=kappa_min,kappa_max=kappa_max,Nkappas=Nkappas,
                                         qquadrature=qquadrature,recompute=True)
        kappas = dP['kappas']; dkappas = dP['dkappas']
        qs=np.sqrt(k**2+kappas**2).real
        Pgap = np.exp((z-gap)*kappas)
        
        #--- Obtain matrix representations of charges and real-space factors
        Cs = self.get_eigencharges(as_vector=True)
        dKappas = dkappas[:,np.newaxis]
        Kappas = kappas[:,np.newaxis]
        
        Rs = rs[:,np.newaxis]
        Qs = qs[np.newaxis,:]
        J0s = j0(Rs*Qs)
        J1s = j1(Rs*Qs)
        
        MPhi = Pgap[:,np.newaxis] * (dP['Phi'] @ Cs) #This is alread the Fourier transform multiplied by q
        MAr = Pgap[:,np.newaxis] * (dP['Ar'] @ Cs)
        MAz = Pgap[:,np.newaxis] * (dP['Az'] @ Cs)
        
        Phi = +J0s @ (dKappas * MPhi)
        Ar = +J0s @ (dKappas * MAr)
        Az = +J1s @ (dKappas * MAz)
        Ez = +J0s @ (Kappas * dKappas * MPhi)
        
        Phi = AWA(Phi.T, axes=[None,rs], axis_names=['eigenindex',r'$r/a$'])
        Ar = AWA(Ar.T, adopt_axes_from=Phi)
        Az = AWA(Az.T, adopt_axes_from=Phi)
        Ez = AWA(Ez.T, axes=[None,rs], axis_names=['eigenindex',r'$r/a$'])
        
        return Phi,Ar,Az,Ez
    
    def getRsampleMatrix(self,rp_freq,gap,Nmodes=20,
                           recompute_rp=True,rp=None,recompute_propagators=False,\
                            farfield=False,
                           k=None,kappa_min=None,kappa_max=None,
                          qquadrature=numrec.GL,Nkappas=244,\
                           **kwargs):

        Nmodes=np.min((self.get_Nmodes(),
                       Nmodes))

        Zmirror = self.getSommerfeldMirrorImpedance(gap=gap, rp_freq=rp_freq,
                                                    recompute_rp=recompute_rp, rp=rp,
                                                    recompute_propagators=recompute_propagators,
                                                    farfield=farfield,
                                                    qquadrature=qquadrature, Nkappas=Nkappas,
                                                    k=k,kappa_min=kappa_min, kappa_max=kappa_max,
                                                    **kwargs)
        
        Cs = self.get_eigencharges(as_vector=True)[:,:Nmodes]

        # The first minus sign emulates `-ZM` as the positive definite operator;
        RSampMat = Cs.T @ -Zmirror @ Cs
            
        return RSampMat
    
    def EradVsGap(self, freq, gapmin=.1, gapmax=4,
                  Ngaps=20, zquadrature=numrec.CC,
                  Nmodes=20, illum_angles=None,
                  illum_angle_weights=None,
                  farfield=False,
                  rp=None, recompute_rp=True,
                  recompute_propagators=True,
                  recompute_brightness=True,
                  subtract_background=True,
                  **kwargs):
        
        #--- Get initial mode amplitudes
        Nmodes=np.min((self.get_Nmodes(),\
                       Nmodes))
        RhoMat=np.diag(self.get_eigenrhos()[:Nmodes])
        
        #--- Retrieve receptivity of eigenmodes
        # Recompute only if instructed; this whole bit could be removed
        if recompute_brightness:
            k= 2 * np.pi * freq #If we're recomputing, it's because we want brightness at `freq` (in this case, `ZSelf` should also be recomputed in principle..)
        else: k=self.get_k()
        Vn=self.get_eigenbrightness(k=k, illum_angles=illum_angles,
                                    illum_angle_weights=illum_angle_weights,
                                    recompute=recompute_brightness,Nmodes=Nmodes)
        #If we have an angle axis, sum it like an integral (axis=0)
        if Vn.ndim==2: Vn = np.sum(Vn,axis=0)
        Vn = Vn[:Nmodes]
        Vn=np.matrix(Vn).T
            
        gaps,dgaps=numrec.GetQuadrature(xmin=gapmin,
                                        xmax=gapmax, \
                                        N=Ngaps, \
                                        quadrature=zquadrature)
        
        Erads=[]
        eigenamplitudes=[]
        ScatMats=[]
        Logger.write('Computing response for %i gaps at freq=%s...' % (len(gaps), freq))
        for gap in gaps:
            
            self.RSampMat=self.getRsampleMatrix(freq, gap, Nmodes=Nmodes, \
                                                recompute_rp=recompute_rp, \
                                                recompute_propagators=recompute_propagators, \
                                                farfield=farfield,
                                                rp=rp, **kwargs)
            self.ScatMat = (self.RSampMat-RhoMat).getI()
            if subtract_background: self.ScatMat += np.matrix(RhoMat).getI()
            ScatMats.append(self.ScatMat)
            # Store the radiated field, as well as the population of eigenmodes
            Erad = complex(Vn.T @ self.ScatMat @ Vn)
            Erads.append(Erad)
            eigenamplitudes.append(np.array( self.ScatMat @ Vn).squeeze())
            
            recompute_rp=False
            recompute_propagators=False
        
        #Store the radiated
        Erads=AWA(Erads,axes=[gaps],axis_names=['$z_{\mathrm{tip}}$'])
        eigenamplitudes=AWA(eigenamplitudes,axes=[gaps,None],\
                            axis_names=['$z_{\mathrm{tip}}$','eigenindex'])
        ScatMats=AWA(ScatMats,axes=[gaps,None,None],\
                            axis_names=['$z_{\mathrm{tip}}$','eigenindex 1','eigenindex 2'])
        result=dict(Erad=Erads,eigenamplitude=eigenamplitudes,scattering_matrices=ScatMats)
        
        return result
    
    def EradSpectrumDemodulated(self, freqs, gapmin=.1, amplitude=2,
                                Ngaps=16, zquadrature=numrec.CC,
                                Nmodes=20, illum_angles=np.linspace(10,80,20),
                                illum_angle_weights=None,
                                rp=None, demod_order=4,
                                farfield=False,
                                update_propagators=True,
                                update_brightness=False,
                                probe_spectroscopy=None,
                                probe_spectroscopy_kwargs={},
                                **kwargs):

        self.all_rp_vals=[]

        T=Timer()
        gapmax = gapmin + 2 * amplitude
        
        #--- First pass will compute propagators and eigenexcitations
        recompute_propagators=True
        
        EradsVsFreq=[]
        eigenamplitudesVsFreq=[]
        if not hasattr(freqs, '__len__'): freqs=[freqs]
        for freq_wn in freqs:
            if probe_spectroscopy is not None:
                probe_spectroscopy.set_eigenset(self,freq_wn,
                                                **probe_spectroscopy_kwargs)
                # `set_eigenset` will be completely entrusted with reconfiguring the probe
            dErad=self.EradVsGap(freq_wn, gapmin=gapmin, gapmax=gapmax,
                                 Ngaps=Ngaps, zquadrature=zquadrature,
                                 Nmodes=Nmodes, illum_angles=illum_angles,
                                 illum_angle_weights=illum_angle_weights,
                                 rp=rp, recompute_rp=True,
                                 farfield=farfield,
                                 recompute_propagators=recompute_propagators,
                                 recompute_brightness=update_brightness,
                                 **kwargs)
            EradsVsFreq.append(dErad['Erad'])
            eigenamplitudesVsFreq.append(dErad['eigenamplitude'])
            
            # These don't necessarily have to update throughout the calculation, they are approximately fixed
            if not update_propagators: recompute_propagators=False
            #if not update_brightness: recompute_brightness=False
        
        gaps=EradsVsFreq[0].axes[0]
        EradsVsFreq=AWA(EradsVsFreq, axes=[freqs, gaps],
                        axis_names=['Frequency',r'$z_\mathrm{tip}$']).T
        eigenamplitudesVsFreq=AWA(eigenamplitudesVsFreq, axes=[freqs, gaps, None],
                                  axis_names=['Frequency',r'$z_\mathrm{tip}$','eigenindex']).T
        result=dict(Erad=EradsVsFreq,eigenamplitude=eigenamplitudesVsFreq)
        
        #--- Demodulate with chebyshev polynomials
        if demod_order:
            Sn=demodulate(EradsVsFreq,demod_order=demod_order)
            result['Sn']=Sn
        
        Logger.write('\t'+T())
        
        return result

    def getNormalizedSignal(self, freqs_wn, rp,
                            a_nm=30, amplitude_nm=50, demod_order=5,
                            Ngaps=24*4, gapmin_nm=.15,
                            L_cm=24e-4,
                            rp_norm = None,
                            norm_single_freq = True,
                            **kwargs):

        # Adapt dimensional arguments to the same units as the probe discretization
        to_nm = a_nm / self.get_a(); from_nm = 1/to_nm
        amplitude = amplitude_nm * from_nm
        gapmin = gapmin_nm * from_nm
        print('amplitude=',amplitude)
        print('gapmin=',gapmin)
        q_to_wn = from_nm * 1e7 # converting q values to wavenumbers will be also be done using the known tip dimensionful radius

        # Convert frequency in wavenumbers to internal units
        # Properly wrapped `rp` will just undo this conversion identically, back to wavenumbers
        # But the dimensionless frequencies will be sized according to the indicated probe length `L_cm`
        L = np.ptp(self.get_zs())  # Height of probe in internal units
        freq_to_wn = L/L_cm
        freqs = freqs_wn / freq_to_wn  # conversion factor from wavenumbers to internal frequency units

        # Wrap the provided rp functions so they can expand out dimensionless frequencies and wavevectors
        # The supplied reflection function should take `frequency (wavenumbers), q (wavenumbers)`
        if hasattr(rp,'__call__'): wrapped_rp = wrap_rp(rp,freq_to_wn,q_to_wn)
        else: wrapped_rp = rp # just numbers

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
            if norm_single_freq: freqs_wn_norm = np.mean(freqs_wn) # a single frequency
            else: freqs_wn_norm = freqs_wn
            freqs_norm = freqs_wn_norm / freq_to_wn

            if hasattr(rp_norm,'__call__'): wrapped_rp_norm = wrap_rp(rp_norm, freq_to_wn,q_to_wn)
            else: wrapped_rp_norm = rp_norm

            signals_ref = self.EradSpectrumDemodulated(freqs_norm, rp=wrapped_rp_norm,
                                                       gapmin=gapmin, amplitude=amplitude,
                                                       Ngaps=Ngaps, demod_order=demod_order,
                                                       **kwargs)
            signals['Sn_norm'] = signals['Sn'] / signals_ref['Sn']

            signals['Sn_norm'].set_axes([None,freqs_wn],
                                   axis_names=[None,'Frequency (cm$^{-1}$)'])

        return signals
    
    def EradApproachCurveDemodulated(self, freq, gaps=np.logspace(-1, 1, 50), amplitude=2,
                                     Nzs_demod=12, zquadrature=numrec.CC,
                                     Nmodes=20, illum_angles=None,
                                     rp=None, recompute_rp=True, demod_order=4,
                                     recompute_propagators=True,
                                     recompute_brightness=True,
                                     **kwargs):
        
        Logger.write('Building approach curve...')
        T=Timer()
        
        Erads=[]
        for gap0 in gaps:
            Logger.write('\tWorking on z=%1.2f...'%gap0)
            dErad=self.EradVsGap(freq, gapmin=gap0, gapmax=gap0+2*amplitude,
                                 Ngaps=Nzs_demod, zquadrature=zquadrature,
                                 Nmodes=Nmodes, illum_angles=illum_angles,
                                 rp=rp, recompute_rp=recompute_rp,
                                 recompute_propagators=recompute_propagators,
                                 recompute_brightness=recompute_brightness,
                                 **kwargs)
            Erad=dErad['Erad']
            gaps_sub = Erad.axes[0] - gap0
            
            #These definitely won't change for the remainder of the calculation
            recompute_propagators=False
            recompute_brightness=False
            recompute_rp=False
        
            Erads.append(Erad) #append the trace of signals from [z0,z0+2*amplitude]
        
        #--- Demodulate traces for all z0 at once
        Erads=AWA(Erads, axes=[gaps, Erads[0].axes[0]],
                  axis_names=['z0','gap']).T #For demodulation, we need `gap` as first axis
        Sns_all = demodulate(Erads,demod_order=demod_order) # demodulation order will be first axis, gap0's second
        Erad = Erads.cslice[0] #This evaluated Erad at `gap=0`, for all z0s

        result = dict(Erad=Erad,Sn=Sns_all)
        
        Logger.write('\t'+T())
        
        return result

    #--- Quasi-static field display

    def computePhiImage(self, Q, rs_out, zs_out, mirror=False):
        """ Real-space map of scalar potential in quasistatic limit."""

        def PhiRing(R1,R2,Z1,Z2):

            from scipy.special import ellipk
            num = 4 * R1 * R2
            den = (R1 + R2) ** 2 + (Z1 - Z2) ** 2

            return 2 * ellipk(num / den) / (np.pi * np.sqrt(den))

        Qrings=np.gradient(Q)
        Rrings=self.get_radii()
        Zrings=self.get_zs() + self.get_gap()

        Zs_out = zs_out[np.newaxis,:]
        Rs_out = rs_out[:,np.newaxis]

        mirror_sign = -1 if mirror else +1

        Phi=0

        Nrings=len(Qrings)
        for i,Qring,Rring,Zring in zip(range(Nrings),\
                                       Qrings,Rrings,Zrings):
            Phi += PhiRing(Rring,Rs_out,\
                           mirror_sign*Zring,Zs_out) * Qring

        return AWA(mirror_sign * Phi,\
                   axes=[rs_out,zs_out],\
                   axis_names=['r','z'])

    def computePhiImageSommerfeld(self, Q, rs_out, zs_out, \
                                  rp, freq=None, Nqs_factor=4,display=True):

        Qrings=np.gradient(Q)
        Rrings=self.get_radii()
        Zrings=self.get_zs() + self.get_gap()

        Zs_out = zs_out[np.newaxis,:,np.newaxis]
        Rs_out = rs_out[:,np.newaxis,np.newaxis]

        NRs = len(rs_out); Nqs = Nqs_factor * NRs
        Dr = np.ptp(rs_out); dr = Dr / NRs
        qs = np.linspace(0, np.pi / dr, Nqs);
        dQ = np.diff(qs)[0]

        Qs = qs[np.newaxis,np.newaxis,:]

        if freq is None: freq=self.get_freq()
        Rp = rp(freq, qs)
        if np.size(Rp) > 1: Rp = Rp[np.newaxis,np.newaxis,:]

        integrand1 = j0(Qs * Rs_out) * -Rp
        integrand2 = 0

        if display: Logger.write('Computing Sommerfeld summation of reflected quasi-static scalar potential at %i q-points...'%Nqs)
        Nrings=len(Qrings)
        for i,Qring,Rring,Zring in zip(range(Nrings),\
                                       Qrings,Rrings,Zrings):
            if display:
                progress = i / Nrings * 100
                sys.stdout.write('\r')
                sys.stdout.write('\tProgress: %1.2f%%' % progress)
                sys.stdout.flush()

            Exp = np.exp(-Qs * np.abs(Zs_out + Zring)) #Here we are placing `Zring` below at `z<0`
            integrand2 += j0(Qs * Rring) * Exp * Qring

        PhiMirror = np.sum(integrand1 * integrand2 * dQ, axis=-1)

        return AWA(PhiMirror,axes=[rs_out,zs_out],\
                   axis_names=['r','z'])

    def computeEfieldImages(self, Q, rs_out, zs_out, rho=1,\
                            rp=None, freq=None, Nqs_factor=4,
                            display=True,mirror_double_images=True):
        """ Real-space map of electric field in quasistatic limit."""

        if mirror_double_images: rs_out=rs_out[rs_out>0]

        PhiS = self.computePhiImage(Q, rs_out, zs_out, mirror=False)
        if rp is not None:
            PhiM = self.computePhiImageSommerfeld(Q, rs_out, zs_out, \
                                                 rp, freq=freq, Nqs_factor=Nqs_factor,display=display)
        else: PhiM = rho * self.computePhiImage(Q, rs_out, zs_out, mirror=True)

        PhiTot = PhiS + PhiM

        Er, Ez = np.gradient(-PhiTot)
        Er *= 1 / np.gradient(rs_out)[:, np.newaxis]
        Ez *= 1 / np.gradient(zs_out)[np.newaxis, :]

        Ez = AWA(Ez, adopt_axes_from=PhiS)
        Er = AWA(Er, adopt_axes_from=PhiS)

        if mirror_double_images:
            Er = mirror_double_image(Er, axis=0)
            Ez = mirror_double_image(Ez, axis=0)

        return Er, Ez

    # This function relies heavily on concepts and entities defined in module `ProbeSpectroscopy`
    def gapSpectroscopy(self, gaps = np.logspace(-1.5, 1, 100),
                        ncpus = 8, backend = 'multiprocessing',
                        Nmodes = 20, recompute=False, reload=False,
                        ** kwargs):

        from . import ProbeSpectroscopy as PS

        try:
            if recompute: raise OSError
            if not hasattr(self,'_gapSpectroscopy') \
                or self._gapSpectroscopy is None \
                or reload: # not instructed to recompute, so try first to load from file

                self._gapSpectroscopy = load(self, PS.ProbeGapSpectroscopyParallel)
                self._gapSpectroscopy.EncodedEigenfields = self # ensure that we re-attach self
                self._gapSpectroscopy.check() # Check that it made sense to attach self

        except (OSError,ValueError):
            self._gapSpectroscopy = PS.ProbeGapSpectroscopyParallel(self, gaps=gaps,
                                                                    ncpus=ncpus, backend=backend,
                                                                    Nmodes=Nmodes, **kwargs)

        return self._gapSpectroscopy
