# -*- coding: utf-8 -*-

import copy
import time
import numpy as np
from numbers import Number
from common.log import Logger
from matplotlib import pyplot as plt

import warnings
import sys
warnings.filterwarnings('ignore')

from scipy.special import j0,j1,legendre
from scipy.linalg import eig,eigh,solve
from numba import njit
from common import numerics
num=numerics
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA

from . import RotationalMoM as RotMoM

class Timer(object):

    def __init__(self):

        self.t0=time.time()

    def __call__(self,reset=True):

        t=time.time()
        msg='\tTime elapsed: %s'%(t-self.t0)

        if reset: self.t0=t

        return msg

def get_probe_radii(zs,L=1000,z0=0,a=1,taper_angle=20,geometry='cone',Rtop=0):

    #Establish location of tip and make tip coordinates
    global Rs
    Where_Tip=(zs>=z0)*(zs<=(z0+L))
    zs_tip=zs-z0

    Logger.write('Getting geometry for selection "%s"...'%geometry)
    if geometry=='PtSi':

        from NearFieldOptics.PolarizationModels import PtSiTipProfile
        Rs=a*PtSiTipProfile((zs-z0)/float(a),L)

    elif geometry=='sphere':
        L=2*a
        Rs=np.zeros(zs.shape)+\
           a*np.sqrt(1-(zs_tip-L/2.)**2/(L/2.)**2)

    elif geometry=='ellipsoid':
        b=L/2.
        R=np.sqrt(b*np.float(a)) #Maintains curvature of 1/a at tip
        Rs=np.zeros(zs.shape)+\
           R*np.sqrt(1-(zs_tip-b)**2/b**2)

    elif geometry in ['cone','hyperboloid']:

        if geometry=='hyperboloid':
            assert taper_angle>0
            tan=np.tan(np.deg2rad(taper_angle))
            R=a/tan**2
            Rs=tan*np.sqrt((zs_tip+R)**2-R**2)

        if geometry=='cone':
            ZShft_Bottom=a*(1-np.sin(np.deg2rad(taper_angle)))
            RShft_Bottom=a*np.cos(np.deg2rad(taper_angle))

            alpha=np.tan(np.deg2rad(taper_angle))

            Rs=RShft_Bottom+(zs_tip-ZShft_Bottom)*alpha

            Where_SphBottom=(zs_tip<=ZShft_Bottom)
            Rs[Where_SphBottom]=np.sqrt(a**2-(a-zs_tip[Where_SphBottom])**2)
            #Rs[np.isnan(Rs)+np.isinf(Rs)]=0

        #Add rounded sphere profile to top of cone/hyperboloid
        ZShft_Top=L-Rtop*(1+np.sin(np.deg2rad(taper_angle)))
        Where_SphTop=(ZShft_Top<zs_tip)*(zs_tip<=L)
        if Where_SphTop.any():

            RShft_Top=Rs[Where_SphTop][0]
            RSph_Top=RShft_Top-Rtop*np.cos(np.deg2rad(taper_angle))
            ZSph_Top=L-Rtop
            Rs[Where_SphTop]=RSph_Top+np.sqrt(Rtop**2-(ZSph_Top-zs_tip[Where_SphTop])**2)

    else: Logger.raiseException('"%s" is an invalid geometry type!'%geometry,exception=ValueError)

    #Make finite only where tip is present
    Rs*=Where_Tip
    Rs[np.isnan(Rs)+np.isinf(Rs)]=0
    minR=1e-40
    Rs[Rs<=minR]=minR #token small value

    return Rs

#--- Root-finding tools

def companion_matrix(p):
    """Assemble the companion matrix associated with a polynomial
    whose coefficients are given by `poly`, in order of decreasing
    degree.
    
    Currently unused, but might find a role in a custom polynomial
    root-finder.
    
    References:
    1) http://en.wikipedia.org/wiki/Companion_matrix
    """
    
    A = np.diag(np.ones((len(p)-2,), np.complex128), -1)
    A[0, :] = -p[1:] / p[0]

    return np.matrix(A)

def find_roots(p,scaling=10):
    """Find roots of a polynomial with coefficients `p` by
    computing eigenvalues of the associated companion matrix.
    
    Overflow can be avoided during the eigenvalue calculation
    by preconditioning the companion matrix with a similarity
    transformation defined by scaling factor `scaling` > 1.
    This transformation will render the eigenvectors in a new
    basis, but the eigenvalues will remain unchanged.
    
    Empirically, `scaling=10` appears to enable successful 
    root-finding on polynomials up to degree 192."""
    
    from scipy.linalg import eigvals
    
    global M
    scaling=np.float64(scaling)
    M=companion_matrix(p); N=len(M)
    D=np.matrix(np.diag(scaling**np.arange(N)))
    Dinv=np.matrix(np.diag((1/scaling)**np.arange(N)))
    Mstar=Dinv*M*D # Similarity transform will leave the eigenvalues unchanged
    
    print(np.abs(Mstar).max())
    
    return eigvals(Mstar)

#--- Fields
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
    
    return Er,Ez

#--- Probe class helpder functions

def qs_envelopeNF(freq,q):
    
    k=2*np.pi*freq; dq=.5*k
    q0=2*k
    env=np.tanh((q-q0)/dq)
    env-=np.tanh((-q0)/dq)
    env[env<0]=0
    env/=env.max() #bring level up to 1
    
    return env

def mirror_double_image(im, axis=0):

    new_axes = im.axes
    ax = new_axes[axis]
    new_axes[axis] = np.append(-ax[::-1], ax)

    slices = [slice(None) for d in range(im.ndim)]
    slices[axis] = slice(None, None, -1)
    new_im = np.concatenate((im[slices], im), axis=axis)

    return AWA(new_im, axes=new_axes, axis_names=im.axis_names)

class Efield_generator(object):
    
    def __init__(self,\
                 Er_Mat,Ez_Mat,\
                 Er_Mat_m,Ez_Mat_m,\
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
    
def demodulate(signal_AWA,demod_order=5,quadrature=numrec.GL,Nts=244):
    "Demodulate `signal_AWA` along its first axis."
    
    from numpy.polynomial.chebyshev import chebfit
    global signal_vs_z
    
    #--- Define domain `x \in [-1,1]` and fit to Chebyshev polynomials
    zs=signal_AWA.axes[0] #Assume we want to demodulate over entire range
    A=(zs.max()-zs.min())/2
    if Nts is None: Nts=len(zs)
    ts,dts=numrec.GetQuadrature(N=Nts,xmin=-.5,xmax=0,quadrature=quadrature)
    zs_sweep = zs.min() + A*(1+np.cos(2*np.pi*ts)) # high to low; if zs were CC quadrature, then these are the same z-values but reordered
    signal_vs_t=signal_AWA.interpolate_axis(zs_sweep,bounds_error=False,extrapolate=True,axis=0,kind='quadratic')

    Sns=[]
    harmonics=np.arange(demod_order+1)
    kernel_shape= (len(ts),)+(1,)*(signal_AWA.ndim-1)
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
    
#--- Probe class

class Probe(object):
    
    defaults={'freq':30e-7/10e-4,\
               'a':1,\
              'gap':1,\
              'L':19e-4/30e-7,\
              'skin_depth':.05}
    
    def __init__(self,zs=None,rs=None,
                 Nnodes=244,L=None,quadrature=numrec.TS,\
                 a=None,taper_angle=20,geometry='hyperboloid',Rtop=0,\
                 freq=None,gap=None,Nsubnodes=2,closed=False,**kwargs):
        
        if a is None: a=self.defaults['a']
        if L is None: L=self.defaults['L']
        self._a=a
        if zs is None:
            zs,wzs=numrec.GetQuadrature(Nnodes,xmin=0,xmax=L,quadrature=quadrature)
            zs -= zs.min()
            zs += wzs[0]/2 #Bring the "zero" coordinate where it belongs before evaluating geometry

        if rs is not None: assert len(rs)==len(zs)
        else:
            if isinstance(geometry,str):
                rs=get_probe_radii(zs,L=L,z0=0,a=a,\
                                      taper_angle=taper_angle,\
                                      geometry=geometry,Rtop=Rtop)
            else:
                assert hasattr(geometry,'__call__')
                rs=geometry(zs,L=L,a=a,**kwargs)

        self.Discretization = RotMoM.Discretization(rs,zs,closed=closed,\
                                                    Nsubnodes=Nsubnodes,display=True)
        self.D=self.Discretization

        self._eigenrhos=None
        self._eigencharges=None
        self._eigenexcitations=None
        self._kappa_min_factor=.1 #This will already put qs very close to the light line:  `qs/k ~ 1 + (factor)**2/2`
        
        self._ZSelf=None
        self._ZMirror=None
        self._gapSpectroscopy=None

        self.set_freq(freq)
        self.set_gap(gap)
        
    def get_a(self): return self._a
    
    def set_freq(self,freq=None):

        if freq is None: freq=self.defaults['freq']
        try:
            if freq == self.get_freq(): return # Do nothing
        except AttributeError: pass
                 
        self._freq=freq
        self._ZSelf=None #Re-set the self interaction
    
    def set_gap(self,gap=None):
        
        if gap is None: gap=self.defaults['gap']
                 
        self._gap=gap
        self._ZMirror=None #Re-set the mirror interaction
        self._eigenrhos=None
        self._eigencharges=None
        self._eigenexcitations=None
        
    def get_freq(self): return copy.copy(self._freq)
    
    def get_k(self): return 2*np.pi*self.get_freq()
    
    def get_kappa_min(self,k):

        kappa_L = 1/np.max(self.get_zs()) * self._kappa_min_factor
        kappa_k = k * self._kappa_min_factor

        # return whichever of the two is larger (not to undersample the probe length!)
        return np.max( (kappa_L, kappa_k) )
    
    def get_gap(self): return copy.copy(self._gap)
        
    def get_zs(self): return copy.copy(self.Discretization.node_zs)

    def _toAWA(self,charge,with_gap=False):
        """Works even on input column vectors."""

        charge=np.asarray(charge).squeeze()
        zs=self.get_zs()
        if with_gap: zs+=self.get_gap()

        return AWA(charge,axes=[zs],axis_names=['z'])

    def get_weights(self): return copy.copy(self.Discretization.node_dts)

    def get_weights_matrix(self): return np.matrix(np.diag(self.get_weights()))
        
    def get_radii(self):

        Rs = copy.copy(self.Discretization.node_Rs)
        return self._toAWA(Rs)

    def get_dRdz(self):

        dR = np.gradient(self.get_radii())
        dz = np.gradient(self.get_zs())

        return dR/dz

    #--- Momentum space propagators & reflectance

    def getFourPotentialPropagators(self,k,farfield=False,\
                                 kappa_min=None,kappa_max=np.inf,Nkappas=244,\
                                 qquadrature=numrec.GL,recompute=False):
        """
        Outputs a propagator for the scalar potential and for
        r- and z-components of the vector potential down to the plane at the tip apex.
        """

        if not recompute:
            try: return self._fourpotentialpropagators
            except AttributeError: pass

        Logger.write('Computing field propagators...')

        #--- Get geometric parameters
        zs=self.get_zs(); Zs=zs[np.newaxis,:]
        rs=self.get_radii(); Rs=rs[np.newaxis,:]
        #dRdt = np.gradient(rs) / self.D.node_dts
        #Sin = dRdt[np.newaxis,:]
        #Cos = np.sqrt(1-Sin**2)
        Sin=self.D.node_sin[np.newaxis,:]
        Cos=self.D.node_cos[np.newaxis,:]
        Dts=self.D.node_dts[np.newaxis,:]

        #--- Prepare Quadrature Grid
        if kappa_min is None: kappa_min=self.get_kappa_min(k)
        Logger.write('\tIncluding evanescent (near-field) waves..')
        self.kappas,self.dkappas=numrec.GetQuadrature(xmin=kappa_min,
                                            xmax=kappa_max,\
                                            N=Nkappas,\
                                            quadrature=qquadrature)
        if farfield and k!=0:
            Logger.write('\tIncluding propogating (far-field) waves..')
            qs,dqs=numrec.GetQuadrature(xmin=0,xmax=k,\
                                            N=Nkappas,\
                                            quadrature=numrec.GL)
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
        PCoul = np.matrix( (Cos*Kappas*J0 + Sin*Qs*J1) * Exp * Dts) #post-factor would be `J0(q*rho)`
        #PCoul = (2*np.pi * 1j/k) * np.matrix( -(Cos*Kappas - Sin*Qs) * Exp * Dts) #post-factor would be `J0(q*rho)`
        #PCoul = (2 * np.pi * 1j / k) * np.matrix(
        #    -(Sin * Kappas + Cos * Qs) * Exp * Dts)  # post-factor would be `J0(q*rho)`

        if k==0: PAz = PAr = np.zeros(PCoul.shape)
        else:
            #--- Az propagator
            PAz = 1j*k * np.matrix( +Cos*J0 * Exp * Dts) #post-factor would be `J0(q*rho)`

            #--- Ar propagator
            PAr = 1j*k * np.matrix( +Sin*J1 * Exp * Dts) #post-factor would be `J1(q*rho)`

        self._fourpotentialpropagators = dict(Phi=PCoul,
                                                Az=PAz,
                                                Ar=PAr,\
                                                kappas=self.kappas,\
                                                dkappas=self.dkappas)


        #These should be integrated w.r.t. `dkappas`, and with
        # mirror signatures (Phi,Ar,Az) -> (-,-,+)
        return self._fourpotentialpropagators

    def getRp(self,freq,qs,recompute=True,rp=None,remove_nan=True,**kwargs):
        """Compute rp at `freq,qs` from a provided or stored `rp` function or AWA."""

        if not recompute:
            try: return self.rpvals
            except AttributeError: pass

        if rp is None:
            if hasattr(self,'rp'): rp=self.rp
            else: rp = lambda freq,q: 1+0j

        else: self.rp=rp

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

        return rpvals

    def getRpGapPropagator(self,rp_freq,kappas,dkappas,
                           gap,rp=None,recompute_rp=True,**kwargs):

        #deduce qs from kappa and freq - inexpensive, and also helps us avoid the light-line!
        k = 2*np.pi*rp_freq
        self.rpk = k
        self.rpkappas = kappas
        qs=np.sqrt(k**2+kappas**2).real.astype(float) #This will ensure we are always evaluating outside the light cone if `kappas>0`
        rpvals=self.getRp(rp_freq,qs,rp=rp,recompute=recompute_rp,**kwargs)

        # if `kappas`, then `kzs=1j*kappas`
        rpmat=np.diag(rpvals*np.exp(-2*kappas*gap)*dkappas)

        return np.matrix(rpmat)

    def getSommerfeldMirrorImpedance(self,k,gap=None,\
                                       recompute_rp=True,rp=None,rp_freq=None,
                                       recompute_propagators=False,farfield=False,
                                       qquadrature=numrec.GL,Nkappas=244,\
                                       kappa_min=None,kappa_max=np.inf,\
                                       **kwargs):

        #--- Ingredients for sommerfeld integral
        dP=self.getFourPotentialPropagators(k,recompute=recompute_propagators,\
                                             farfield=farfield,
                                             qquadrature=qquadrature,Nkappas=Nkappas,\
                                              kappa_min=kappa_min,kappa_max=kappa_max,**kwargs)
        if gap is None: gap=self.get_gap()
        if rp_freq is None: rp_freq = self.get_freq()
        rpMat=self.getRpGapPropagator(rp_freq,kappas=dP['kappas'],dkappas=dP['dkappas'],\
                                       gap=gap,rp=rp,recompute_rp=recompute_rp)

        MPhi = dP['Phi']
        MAr = dP['Ar']
        MAz = dP['Az']

        # Mirror sign applies to Coulomb and Ar fields
        # other -1 signs follow from definition of generalized reflectance
        mirror_sign = -1
        Zmirror =  MPhi.T @ (mirror_sign * rpMat) @ MPhi
        if k!=0:
            Zmirror = Zmirror + \
                     -MAr.T @ (mirror_sign * rpMat) @ MAr \
                        - MAz.T @ rpMat @ MAz

        return Zmirror
        
    def get_self_impedance(self,k=None,recompute=False,**kwargs):
        
        if recompute or self._ZSelf is None:

            if k is None: k=self.get_k()
            self._ZSelf = self.Discretization.get_self_impedance(k,**kwargs)
            
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
                self._ZMirror = self.Discretization.get_mirror_impedance(k,gap=gap,**kwargs)
            
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
        
        return np.complex(N2)
    
    def get_field_overlap(self, Q, Er=None, Ez=None, V=None):

        if V is None: V = self.Discretization.get_excitation(Er,Ez)
        # otherwise assume V is a vector

        if not isinstance(Q, np.matrix): Q=np.matrix(Q).T
        
        overlap = Q.T @ V #V carries the annular weights
        
        #try to collapse to single number
        try: return np.complex(overlap)
        except TypeError: return np.array(overlap).squeeze() #squeeze down vectors
    
    def get_charge_density(self,Q):

        Rs = self.get_radii()
        dts = self.Discretization.node_dts
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
                           plot=True):

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

        # --- Conditioning of ZM
        if condition_ZM:
            #ZM = -numrec.nearestPD(-ZM)
            eb, vb = eigh( -ZM.real ) #This operator should be positive definite

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
            vb = vb[:, where_ok]

            # so now, PhiM_cs = vb @ -eb @ vb.T; we use P=vb @ vb.T projector
            # and projected we have: PhiS_cs = vb @ vb.T @ PhiS_cs @ vb @ vb.T
            # Hit (vb.T .. vb) to both and we have:
            # PhiM = -eb
            # PhiS = vb.T @ PhiS_cs @ vb
            # These are now in the basis of (csb = cs @ vb)
            # plt.matshow((csb.T @ W @ csb).real)
            b = np.diag(eb)

        else:
            vb = np.eye(len(ZM))
            b = -ZM

        a = vb.T @ ZS @ vb

        #-- Project and store modified operators
        # Notably, we are introducing a null space, so it may profit to use operators that need no conditioning
        cs = np.eye(len(ZS))
        csb = cs @ vb
        Pb = csb @ csb.T
        ZM = Pb.T @ ZM @ Pb
        self._ZMirror = ZM  #Don't modify self impedance, we need to keep one of them full rank

        rhos, Qvecs = eig(a=a, b=b)

        #-- Filtering
        rhothresh = 1e-2
        where_ok = (np.abs(rhos) > rhothresh) * np.isfinite(rhos)
        rhos = rhos[where_ok]
        Qvecs = (csb @ Qvecs)[:,where_ok] #project back from restricted basis

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
        plt.title('Min. eigenvalue: %1.2E+%1.2Ej'%(rhos[0].real,rhos[0].imag))
    
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

        return Qs

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
    
    def get_eigenexcitations(self,Er=None,Ez=None,V=None,\
                             recompute=False):
        
        if recompute or self._eigenexcitations is None:
            
            Vns=[self.get_field_overlap(Q,Er=Er,Ez=Ez,V=V) \
                 for Q in np.array(self.get_eigencharges())]
                
            self._eigenexcitations=np.array(Vns)
            
        return self._eigenexcitations

    def get_brightness(self,Q,k=None,angles=np.linspace(45,90,20),\
                        illumination=EBesselBeamFF,average=True):

        if k is None: k =self.get_k()

        Rns = []
        if not hasattr(angles, '__len__'): angles = [angles]
        for angle in angles:
            Er, Ez = illumination(angle=angle, k=k)
            Rn = self.get_field_overlap(Q, Er=Er, Ez=Ez)
            Rns.append(Rn)

        Rns = AWA(Rns, axes=[angles], \
                  axis_names=['Angle (deg.)'])

        if average:
            if Rns.ndim==2: angles=angles[:,np.newaxis] #just so we broadcast over a "hidden" axis
            Rns = np.mean(np.sin(np.deg2rad(angles)) * Rns, axis=0)

        return Rns
    
    def get_eigenbrightness(self, k=None, angles=np.linspace(45, 90, 20), \
                            illumination=EBesselBeamFF, average=True, \
                            recompute=False):
        """This is the same as `get_eigenexcitations` except brightness is averaged over illumination angles."""

        if k is None: k =self.get_k()
        
        if not recompute:
            try: return self._eigenbrightness
            except AttributeError: pass
        
        Rns=[]
        if not hasattr(angles,'__len__'): angles=[angles]
        for angle in angles:
            
            Er,Ez=illumination(angle=angle,k=k)
            Rn=[self.get_field_overlap(Q,Er=Er,Ez=Ez) \
                 for Q in np.array(self.get_eigencharges())]
            Rns.append(Rn)
            
        Rns=AWA(Rns,axes=[angles,None],\
                axis_names=['Angle (deg.)','eigenindex'])
        
        if average:
            Rns=np.mean(np.sin(np.deg2rad(angles[:,np.newaxis])) * Rns, axis=0)
            
        self._eigenbrightness=Rns
        
        return self._eigenbrightness

    get_eigenreceptivity = get_eigenbrightness #a legacy alias
    
    def get_eigenamplitudes(self,Er=None,Ez=None,V=None,rho=0):
        
        Vns=self.get_eigenexcitations(Er=Er,Ez=Ez,V=V,\
                                      recompute=True)
        
        amps=[]
        for n in range(self.get_Nmodes()):
            eigenrho=self.get_eigenrho(n)
            amps.append(Vns[n]/(rho-eigenrho))
            
        return AWA(amps,axes=[None],axis_names=['Eigenindex'])

    #--- Charge solutions

    def solve_induced_charge_direct(self,Er=None,Ez=None,V=None,\
                                     rho=0,\
                                     as_vector=False,\
                                     **kwargs):

        if V is None: V = self.Discretization.get_excitation(Er=Er,Ez=Ez)

        Z = self.get_impedance_matrix(rho=rho,**kwargs)

        #--- Solve linear system
        Q=solve(Z, -V)
        Q=np.matrix(Q) #somehow matrix type is not default output of `solve`..

        if as_vector: return Q

        Q = self._toAWA(Q)

        return Q
    
    def solve_induced_charge_eigen(self,Er=None,Ez=None,\
                                     V=None,rho=0,\
                                     Nmodes=20,Veff=True):

        if Veff:
            Q0 = self.solve_induced_charge_direct(Er=Er, Ez=Ez, V=V, rho=0)
            Zmirror = self.get_mirror_impedance()
            V = np.array(Zmirror @ np.matrix(Q0).T).squeeze()
        else:
            if V is None: V=self.Discretization.get_excitation(Er,Ez)
            Q0=0

        self.eigenamplitudes=self.get_eigenamplitudes(V=V,rho=rho)[:Nmodes]
            
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
                            kappa_min=None,kappa_max=np.inf,Nkappas=244,
                             qquadrature=numrec.GL):
        
        #--- Obtain propagators
        gap=self.get_gap()
        assert z<gap
        dP = self.getFourPotentialPropagators(farfield=farfield,k=k,
                                         kappa_min=kappa_min,kappa_max=kappa_max,Nkappas=Nkappas,
                                         qquadrature=qquadrature,recompute=True)
        kappas = dP['kappas']; dkappas = dP['dkappas']
        if k is None: k = self.get_k()
        qs=np.sqrt(k**2+kappas**2).real
        Pgap = np.matrix(np.diag(np.exp((z-gap)*kappas)))
        
        #--- Obtain matrix representations of charges and real-space factors
        Cs = self.get_eigencharges(as_vector=True)
        dKappas = np.matrix(np.diag(dkappas))
        Kappas = np.matrix(np.diag(kappas))
        
        Rs = rs[:,np.newaxis]
        Qs = qs[np.newaxis,:]
        J0s = np.matrix(j0(Rs*Qs))
        J1s = np.matrix(j1(Rs*Qs))
        
        MPhi = Pgap @ dP['Phi'] @ Cs #This is alread the Fourier transform multiplied by q
        MAr = Pgap @ dP['Ar'] @ Cs
        MAz = Pgap @ dP['Az'] @ Cs
        
        Phi = +J0s @ dKappas @ MPhi
        Ar = +J0s @ dKappas @ MAr
        Az = +J1s @ dKappas @ MAz
        Ez = +J0s @ Kappas @ dKappas @ MPhi 
        
        Phi = AWA(Phi.T, axes=[None,rs], axis_names=['eigenindex',r'$r/a$'])
        Ar = AWA(Ar.T, adopt_axes_from=Phi)
        Az = AWA(Az.T, adopt_axes_from=Phi)
        Ez = AWA(Ez.T, axes=[None,rs], axis_names=['eigenindex',r'$r/a$'])
        
        return Phi,Ar,Az,Ez
    
    def getRsampleMatrix(self,freq,gap,Nmodes=20,\
                   recompute_rp=True,rp=None,recompute_propagators=False,\
                    farfield=False,
                   k=None,kappa_min=None,kappa_max=np.inf,qquadrature=numrec.GL,Nkappas=244,\
                   **kwargs):

        # If a k-value for field evaluation is not specified, inherit from frequency and set the probe to that state
        if k is None:
            self.set_freq(freq)
            k=2*np.pi*freq
        Nmodes=np.min((self.get_Nmodes(),\
                       Nmodes))

        Zmirror = self.getSommerfeldMirrorImpedance(gap=gap, rp_freq=freq, \
                                                    recompute_rp=recompute_rp, rp=rp,\
                                                    recompute_propagators=recompute_propagators, \
                                                    farfield=farfield,
                                                    qquadrature=qquadrature, Nkappas=Nkappas, \
                                                    k=k,kappa_min=kappa_min, kappa_max=kappa_max, \
                                                    **kwargs)
        
        Cs = self.get_eigencharges(as_vector=True)[:,:Nmodes]

        # The first minus sign emulates `-ZM` as the positive definite operator;
        RSampMat = Cs.T @ -Zmirror @ Cs
            
        return RSampMat
    
    def EradVsGap(self, freq, zmin=.1, zmax=4, \
                  Nzs=20, zquadrature=numrec.CC, \
                  Nmodes=20, illum_angles=np.linspace(10,80,20), \
                  rp=None, \
                  recompute_rp=True, \
                  recompute_propagators=True, \
                  recompute_brightness=True, \
                  subtract_background=True, \
                  illumination=EBesselBeamFF, \
                  **kwargs):
        
        #--- Get initial mode amplitudes
        Nmodes=np.min((self.get_Nmodes(),\
                       Nmodes))
        RhoMat=np.matrix(np.diag(self.get_eigenrhos()[:Nmodes]))
        
        #--- Retrieve receptivity of eigenmodes
        # Recompute only if instructed
        Vn=self.get_eigenbrightness(k=2*np.pi*freq,angles=illum_angles,average=True,\
                                     illumination=illumination,\
                                     recompute=recompute_brightness)[:Nmodes]
        Vn=np.matrix(Vn).T
            
        gaps,dgaps=numrec.GetQuadrature(xmin=zmin,
                                          xmax=zmax,\
                                           N=Nzs,\
                                           quadrature=zquadrature)
        
        Erads=[]
        eigenamplitudes=[]
        ScatMats=[]
        Logger.write('Computing response for gaps at freq=%s...'%freq)
        for gap in gaps:
            
            self.RSampMat=self.getRsampleMatrix(freq,gap,Nmodes=Nmodes,\
                                           recompute_rp=recompute_rp,\
                                           recompute_propagators=recompute_propagators,\
                                           rp=rp,**kwargs)
            self.ScatMat = (self.RSampMat-RhoMat).getI()
            if subtract_background: self.ScatMat += RhoMat.getI()
            ScatMats.append(self.ScatMat)
            # Store the radiated field, as well as the population of eigenmodes
            Erad = np.complex(Vn.T @ (self.ScatMat) @ Vn)
            Erads.append(Erad)
            eigenamplitudes.append(np.array( (self.ScatMat) @ Vn).squeeze())
            
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
    
    def EradSpectrumDemodulated(self,freqs,zmin=.1,amplitude=2,\
                        Nzs=16,zquadrature=numrec.CC,\
                        Nmodes=20,illum_angles=np.linspace(10,80,20),\
                        rp=None,demod_order=4,\
                            update_propagators=True,
                            update_brightness=False,
                            probe_spectroscopy=None,
                            update_charges=True,
                            **kwargs):
        T=Timer()
        zmax = zmin+2*amplitude
        
        #--- First pass will compute propagators and eigenexcitations
        recompute_propagators=True
        recompute_brightness=True
        
        EradsVsFreq=[]
        eigenamplitudesVsFreq=[]
        if not hasattr(freqs,'__len__'): freqs=[freqs]
        for freq in freqs:
            if probe_spectroscopy is not None:
                probe_spectroscopy.set_eigenset(self,freq,update_charges=update_charges)
            dErad=self.EradVsGap(freq, zmin=zmin, zmax=zmax,
                                 Nzs=Nzs, zquadrature=zquadrature,
                                 Nmodes=Nmodes, illum_angles=illum_angles,
                                 rp=rp, recompute_rp=True,
                                 recompute_propagators=recompute_propagators,
                                 recompute_brightness=recompute_brightness,
                                 **kwargs)
            EradsVsFreq.append(dErad['Erad'])
            eigenamplitudesVsFreq.append(dErad['eigenamplitude'])
            
            # These don't necessarily have to update throughout the calculation, they are approximately fixed
            if not update_propagators: recompute_propagators=False
            if not update_brightness: recompute_brightness=False
        
        gaps=EradsVsFreq[0].axes[0]
        EradsVsFreq=AWA(EradsVsFreq,axes=[freqs,gaps],\
                        axis_names=['Frequency',r'$z_\mathrm{tip}$']).T
        eigenamplitudesVsFreq=AWA(eigenamplitudesVsFreq,axes=[freqs,gaps,None],\
                                  axis_names=['Frequency',r'$z_\mathrm{tip}$','eigenindex']).T
        result=dict(Erad=EradsVsFreq,eigenamplitude=eigenamplitudesVsFreq)
        
        #--- Demodulate with chebyshev polynomials
        if demod_order:
            Logger.write('Demodulating...')
            Sn=demodulate(EradsVsFreq,demod_order=demod_order)
            result['Sn']=Sn
        
        Logger.write('\t'+T())
        
        return result
    
    def EradApproachCurveDemodulated(self,freq,zs=np.logspace(-1,1,50),amplitude=2,\
                                    Nzs_demod=12,zquadrature=numrec.CC,\
                                    Nmodes=20,illum_angles=np.linspace(10,80,20),\
                                    rp=None,recompute_rp=True,demod_order=4,\
                                    recompute_propagators=True,
                                    recompute_brightness=True,
                                    **kwargs):
        
        Logger.write('Building approach curve...')
        T=Timer()
        
        Erads=[]
        
        for z0 in zs:
            Logger.write('\tWorking on z=%1.2f...'%z0)
            zmin=z0
            zmax=z0+2*amplitude
            dErad=self.EradVsGap(freq, zmin=zmin, zmax=zmax, \
                                 Nzs=Nzs_demod, zquadrature=zquadrature, \
                                 Nmodes=Nmodes, illum_angles=illum_angles, \
                                 rp=rp, recompute_rp=recompute_rp, \
                                 recompute_propagators=recompute_propagators, \
                                 recompute_brightness=recompute_brightness, \
                                 **kwargs)
            Erad=dErad['Erad']
            
            #These definitely won't change for the remainder of the calculation
            recompute_propagators=False
            recompute_brightness=False
            recompute_rp=False
        
            Erads.append(Erad) #append the trace of signals from [z0,z0+2*amplitude]
        
        #--- Demodulate traces for all z0 at once
        Erads=AWA(Erads,axes=[zs,Erads[0].axes[0]],\
                  axis_names=['z0','gap']).T #For demodulation, we need `gap` as first axis
        Sns=demodulate(Erads,demod_order=demod_order)
        
        result = dict(Erad=Erads[0],Sn=Sns)
        
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
                            rp=None, freq=None, Nqs_factor=4,display=True,mirror_double_images=True):

        if mirror_double_images: rs_out=rs_out[rs_out>0]

        PhiS = self.computePhiImage(Q, rs_out, zs_out, mirror=False)
        if rp is not None:
            PhiM = self.computePhiImageSommerfeld(Q, rs_out, zs_out, \
                                                 rp, freq=freq, Nqs_factor=Nqs_factor,display=display)
        else: PhiM = self.computePhiImage(Q, rs_out, zs_out, mirror=True)

        PhiTot = PhiS + rho * PhiM

        Er, Ez = np.gradient(-PhiTot)
        Er *= 1 / np.gradient(rs_out)[:, np.newaxis]
        Ez *= 1 / np.gradient(zs_out)[np.newaxis, :]

        Ez = AWA(Ez, adopt_axes_from=PhiS)
        Er = AWA(Er, adopt_axes_from=PhiS)

        if mirror_double_images:
            Er = mirror_double_image(Er, axis=0)
            Ez = mirror_double_image(Ez, axis=0)

        return Er, Ez

    def gapSpectroscopy(self, gaps = np.logspace(-1.5, 1, 100), \
                        ncpus = 8, backend = 'multiprocessing', \
                        Nmodes = 20, recompute=False, ** kwargs):

        from . import ProbeSpectroscopy as PS

        if recompute or self._gapSpectroscopy is None:
            self._gapSpectroscopy = PS.ProbeGapSpectroscopyParallel(self, gaps=gaps, \
                                                                    ncpus=ncpus, backend=backend,\
                                                                    Nmodes=Nmodes, **kwargs)

        return self._gapSpectroscopy
