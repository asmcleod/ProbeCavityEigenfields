import sys
import numpy as np
from scipy.special import j0,j1,ellipk
from scipy.integrate import quad,nquad
from numba import njit
from common.log import Logger
from common import numerical_recipes as numrec
from common.baseclasses import AWA
import time

quadlimit=100
MultiSamples=(-1,+1) #Simpson's rule

def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
    """Adapted from `scipy.integrad.dblquad`, simply added `limit` to kwargs."""

    def temp_ranges(*args):
        return [gfun(args[0]) if callable(gfun) else gfun,
                hfun(args[0]) if callable(hfun) else hfun]

    return nquad(func, [temp_ranges, [a, b]], args=args,
            opts={"epsabs": epsabs, "epsrel": epsrel, "limit": limit})

"""
#--- Spectral domain Green's functions

#- Coulomb

@njit
def Spectral_Coul_SingDiag_integrand(kappa, R, wz, k):
    q = np.sqrt(k ** 2 + kappa ** 2)

    kw=kappa*wz
    pulse_integral = 2/kw * (1 + (np.exp(-kw) - 1) / kw)

    return +j0(R * q)**2 * pulse_integral

def Spectral_Coul_SingDiag_integral(R, wz, k):

    cr = quad(Spectral_Coul_SingDiag_integrand, 0, np.inf, args=(R, wz, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    return cr[0]

@njit
def Spectral_Coul_SingOffdiag_integrand(kappa, R1, R2, dz, wz1, wz2, k):
    q = np.sqrt(k ** 2 + kappa ** 2)

    wsum=(wz1+wz2)/2
    wdiff=(wz1-wz2)/2

    dist_part=np.exp(-kappa*(dz+wsum)) \
             + np.exp(-kappa*(dz-wsum)) \
             - np.exp(-kappa*(dz+wdiff)) \
             - np.exp(-kappa*(dz-wdiff))
    dist_part*=1/((kappa*wz1/2)*(kappa*wz2/2)) #This is application of pulses, somehow doesn't work, check
    dist_part=np.exp(-kappa*dz)

    return +j0(R1 * q) * j0(R2 * q) * dist_part

def Spectral_Coul_SingOffdiag_integral(R, wz, k):

    cr = quad(Spectral_Coul_SingOffdiag_integrand, 0, np.inf, args=(R, wz, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    return cr[0]

@njit
def Spectral_Coul_Nonsing_integrand(kz, R1, R2, dz, k):
    q = np.sqrt(k ** 2 - kz ** 2)

    return +j0(R1 * q) * j0(R2 * q) * np.cos(kz * dz) #nonsingular result should be multiplied by 1j


def Spectral_Coul_Nonsing_integral(R, wz, k):
    ci = quad(Spectral_Coul_Nonsing_integrand, 0, np.inf, args=(R, wz, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    return ci[0]*1j

#-- Faraday

@njit
def Spectral_Ar_SingDiag_integrand(kappa, R, wz, k):
    q = np.sqrt(k ** 2 + kappa ** 2)

    kw=kappa*wz
    pulse_integral = 2/kw * (1 + (np.exp(-kw) - 1) / kw)

    return +j1(R * q)**2 * pulse_integral

def Spectral_Ar_SingDiag_integral(R, wz, k):
    ar = quad(Spectral_Ar_SingDiag_integrand, 0, np.inf, args=(R, wz, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    return ar[0]

@njit
def Spectral_Ar_SingOffdiag_integrand(kappa, R1, R2, dz, wz1, wz2, k):
    q = np.sqrt(k ** 2 + kappa ** 2)

    wsum=(wz1+wz2)/2
    wdiff=(wz1-wz2)/2

    dist_part=np.exp(-kappa*(dz+wsum)) \
             + np.exp(-kappa*(dz-wsum)) \
             - np.exp(-kappa*(dz+wdiff)) \
             - np.exp(-kappa*(dz-wdiff))
    dist_part*=1/((kappa*wz1/2)*(kappa*wz2/2)) #This is application of pulses
    dist_part=np.exp(-kappa*dz)

    return +j1(R1 * q) * j1(R2 * q) * dist_part

def Spectral_Ar_SingOffdiag_integral(R, wz, k):
    ar = quad(Spectral_Ar_SingOffdiag_integrand, 0, np.inf, args=(R, wz, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    return ar[0]

@njit
def Spectral_Ar_Nonsing_integrand(kz, R1, R2, dz, k):
    q = np.sqrt(k ** 2 - kz ** 2)

    return +j1(R1 * q) * j1(R2 * q) * np.cos(kz * dz) #nonsingular result should be multiplied by 1j

def Spectral_Ar_Nonsing_integral(R, wz, k):
    ai = quad(Spectral_Ar_SingOffdiag_integrand, 0, np.inf, args=(R, wz, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    return ai[0]*1j
"""

#--- Spatial domain Green's functions

#TODO: Implement diagonal-adjacent integrals for singular part
#       Do this by:
#            1. symmetric (avg over both sites) single-Quadrature on regularized integrand +
#            2. Quadrature on Ellipk + logarithm (omit this!)
#            3. symmetric (avg over both sites) integrated Logarithm to O(w^2)

#-- Coulomb, singular

@njit
def Coul_Sing_integrand_regularized(phi, X2, Y2, k):
    dz = 0
    Del = np.sqrt(X2 + dz ** 2 - Y2 * np.cos(phi))

    return (np.cos(k * Del) - 1) / Del

def Coul_SingDiag_integral(R, drdz, wz, k):

    X2 = Y2 = 2 * R ** 2
    angint = quad(Coul_Sing_integrand_regularized, 0, np.pi, args=(X2, Y2, k),
                  epsabs=1.49e-08,
                  epsrel=1.49e-08, limit=quadlimit)

    # Second and third rows are the correction proportional to `(w*drdz)**2`
    #logpart = 2/R * (1 + np.log(16 * R / wz)) \
    #            + wz**2 * drdz**2 / (36 * R**3) \
    #                * (2 * (1 + 3 * np.log(16 * R / wz)) -9 )

    logpart = 2 / R * (1 + np.log(16 * R / wz))
    # A correction term at second order in `wz`
    logpart += wz**2 * (2 - 3 * np.log(16 * R / wz)) / (288 * R**3)
    # A correction term at second order in `wz*dRdz`
    logpart += -(wz*drdz)**2 * (4 + 3 * np.log(16 * R / wz)) / (36 * R**3)

    return angint[0] / np.pi + logpart / (2 * np.pi)

@njit
def Coul_Sing_integrandDbl_regularized(dz, phi, R, drdz, k, alpha):
    #Last argument will not be used, just to uniformize call signature

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return (np.cos(k * Del) - 1) / Del

@njit
def Coul_Sing_integrandDbl_regularized_alpha(dz, phi, R, drdz, k, alpha):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return (np.cos(k * Del) * np.cos(alpha * phi) - 1) / Del

def Coul_SingDiag_integralDbl(R, drdz, wz, k, alpha=0):

    if not alpha: integrand = Coul_Sing_integrandDbl_regularized
    else: integrand = Coul_Sing_integrandDbl_regularized_alpha

    angint = dblquad(integrand, \
                     0, np.pi, lambda phi: -wz / 2, lambda phi: +wz / 2, args=(R, drdz, k, alpha),
                      epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=quadlimit)

    # Second and third rows are the correction proportional to `(w*drdz)**2`
    #logpart = 2/R * (1 + np.log(16 * R / wz)) \
    #            + wz**2 * drdz**2 / (36 * R**3) \
    #                * (2 * (1 + 3 * np.log(16 * R / wz)) -9 )

    logpart = 2 / R * (1 + np.log(16 * R / wz))
    # A correction term at second order in `wz`
    logpart += wz**2 * (2 - 3 * np.log(16 * R / wz)) / (288 * R**3)
    # A correction term at second order in `wz*dRdz`
    logpart += -(wz*drdz)**2 * (4 + 3 * np.log(16 * R / wz)) / (36 * R**3)

    return 1 / wz * angint[0] / np.pi + logpart / (2 * np.pi)

#TODO: diagonal-adjacent integrals

@njit
def Coul_SingOffdiag_integrand(phi, X2,Y2,k):

    Del = np.sqrt(X2 - Y2 * np.cos(phi))

    return np.cos(k*Del) / Del

def Coul_SingOffdiag_integral(R1, R2, dz, k):

    X2 = R1**2 + R2**2 + dz**2
    Y2 = 2*R1*R2
    cr = quad(Coul_SingOffdiag_integrand, 0, np.pi, args=(X2, Y2, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)
    return 2 * cr[0] / (2 * np.pi)  # factor of 2 emulates 0 to 2*pi, divisor enforces average

#NEW: a trapezoidal version of - spatial coul singular off-diag

@njit
def Coul_SingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.cos(k*Del) / Del

    return result/len(X2Y2s)

def Coul_SingOffdiag_integralMulti(r1, r2, dz,\
                                           drdz1,drdz2,\
                                          wz1,wz2, k):

    # Sample multiple times inside each annulus, evaluating in the middle of the other
    X2Y2s=[]
    for s1 in MultiSamples:
        R1 = (r1 + s1*drdz1*wz1/2)
        dZ = (dz + s1*wz1/2)
        X2 = R1 ** 2 + r2 ** 2 + dZ ** 2
        Y2 = 2 * R1 * r2
        X2Y2s.append((X2,Y2))
    for s2 in MultiSamples:
        R2 = (r2 + s2*drdz2*wz2/2)
        dZ = (dz + s2*wz2/2)
        X2 = r1 ** 2 + R2 ** 2 + dZ ** 2
        Y2 = 2 * r1 * R2
        X2Y2s.append((X2,Y2))

    cr = quad(Coul_SingOffdiag_integrandMulti, 0, np.pi, args=(k,)+tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average, N is trapezoid rule
    return 2 * cr[0] / (2 * np.pi)

#-- Coulomb, nonsingular

@njit
def Coul_Nonsing_integrand(phi, X2, Y2, k):

    Del = np.sqrt(X2 + - Y2 * np.cos(phi))

    return np.sin(k * Del) / Del

def Coul_Nonsing_integral(R1, R2, dz, k):

    X2 = R1 ** 2 + R2 ** 2 + dz ** 2
    Y2 = 2 * R1 * R2

    angint = quad(Coul_Nonsing_integrand, 0, np.pi, args=(X2, Y2, k),
                  epsabs=1.49e-08,
                  epsrel=1.49e-08, limit=quadlimit)

    return angint[0] / np.pi * 1j

def Coul_Nonsing_integralDiag(R, drdz, wz, k):

    return Coul_Nonsing_integral(R1=R, R2=R, dz=0, k=k)

@njit
def Coul_NonsingDiag_integrandDbl(dz, phi, R, drdz, k):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return np.sin(k * Del) / Del

def Coul_NonsingDiag_integralDbl(R, drdz, wz, k):

    angint = dblquad(Coul_NonsingDiag_integrandDbl, \
                     0, np.pi, lambda phi: -wz / 2, lambda phi: +wz / 2, args=(R, drdz, k),
                      epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=quadlimit)

    return 1 / wz * angint[0] / np.pi * 1j

@njit
def Coul_NonsingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.sin(k*Del) / Del

    return result/len(X2Y2s)

def Coul_NonsingOffdiag_integralMulti(r1, r2, dz,\
                                      drdz1,drdz2,\
                                       wz1,wz2, k):

    # Sample multiple times inside each annulus, evaluating in the middle of the other
    X2Y2s=[]
    for s1 in MultiSamples:
        R1 = (r1 + s1*drdz1*wz1/2)
        dZ = (dz + s1*wz1/2)
        X2 = R1 ** 2 + r2 ** 2 + dZ ** 2
        Y2 = 2 * R1 * r2
        X2Y2s.append((X2,Y2))
    for s2 in MultiSamples:
        R2 = (r2 + s2*drdz2*wz2/2)
        dZ = (dz + s2*wz2/2)
        X2 = r1 ** 2 + R2 ** 2 + dZ ** 2
        Y2 = 2 * r1 * R2
        X2Y2s.append((X2,Y2))

    ci = quad(Coul_NonsingOffdiag_integrandMulti, 0, np.pi, args=(k,)+tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average, N is trapezoid rule
    return 2 * ci[0] / (2 * np.pi) * 1j

#-- Faraday, singular

@njit
def Ar_Sing_integrand_regularized(phi, X2, Y2, k):
    dz = 0
    Del = np.sqrt(X2 + dz ** 2 - Y2 * np.cos(phi))

    return (np.cos(phi)*np.cos(k * Del) - 1) / Del

def Ar_SingDiag_integral(R, drdz, wz, k):
    X2 = Y2 = 2 * R ** 2
    angint = quad(Ar_Sing_integrand_regularized, 0, np.pi, args=(X2, Y2, k),
                  epsabs=1.49e-08,
                  epsrel=1.49e-08, limit=quadlimit)

    # Second and third rows are the correction proportional to `(w*drdz)**2`
    logpart = 2 / R * (1 + np.log(16 * R / wz)) \
                + wz ** 2 * drdz ** 2 / (36 * R ** 3) \
                   * (2 * (1 + 3 * np.log(16 * R / wz)) - 9)

    return angint[0] / np.pi + logpart / (2 * np.pi)

@njit
def Ar_Sing_integrandDbl_regularized(dz, phi, R, drdz, k):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return (np.cos(phi)*np.cos(k * Del) - 1) / Del

def Ar_SingDiag_integralDbl(R, drdz, wz, k):

    angint = dblquad(Ar_Sing_integrandDbl_regularized, \
                     0, np.pi, -wz / 2, +wz / 2, args=(R, drdz, k),
                      epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=quadlimit)

    # Second and third rows are the correction proportional to `(w*drdz)**2`
    logpart = 2 / R * (1 + np.log(16 * R / wz)) \
                + wz ** 2 * drdz ** 2 / (36 * R ** 3) \
                   * (2 * (1 + 3 * np.log(16 * R / wz)) - 9)

    return 1 / wz * angint[0] / np.pi + logpart / (2 * np.pi)

#TODO: diagonal-adjacent integrals

@njit
def Ar_SingOffdiag_integrand(phi, X2,Y2,k):

    Del = np.sqrt(X2 - Y2 * np.cos(phi))

    return np.cos(phi)*np.cos(k*Del) / Del

def Ar_SingOffdiag_integral(R1, R2, dz, k):

    X2 = R1**2 + R2**2 + dz**2
    Y2 = 2*R1*R2
    ar = quad(Ar_SingOffdiag_integrand, 0, np.pi, args=(X2, Y2, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)
    return 2 * ar[0] / (2 * np.pi)  # factor of 2 emulates 0 to 2*pi, divisor enforces average

@njit
def Ar_SingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.cos(phi) * np.cos(k*Del) / Del

    return result/len(X2Y2s)

def Ar_SingOffdiag_integralMulti(r1, r2, dz,\
                                           drdz1,drdz2,\
                                          wz1,wz2, k):

    # Sample multiple times inside each annulus, evaluating in the middle of the other
    X2Y2s=[]
    for s1 in MultiSamples:
        R1 = (r1 + s1*drdz1*wz1/2)
        dZ = (dz + s1*wz1/2)
        X2 = R1 ** 2 + r2 ** 2 + dZ ** 2
        Y2 = 2 * R1 * r2
        X2Y2s.append((X2,Y2))
    for s2 in MultiSamples:
        R2 = (r2 + s2*drdz2*wz2/2)
        dZ = (dz + s2*wz2/2)
        X2 = r1 ** 2 + R2 ** 2 + dZ ** 2
        Y2 = 2 * r1 * R2
        X2Y2s.append((X2,Y2))

    ar = quad(Ar_SingOffdiag_integrandMulti, 0, np.pi, args=(k,)+tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average
    return 2 * ar[0] / (2 * np.pi)

#-- Faraday, nonsingular

@njit
def Ar_Nonsing_integrand(phi, X2,Y2,k):

    Del = np.sqrt(X2 - Y2 * np.cos(phi))

    return np.cos(phi)*np.sin(k*Del) / Del

def Ar_Nonsing_integral(R1, R2, dz, k):

    X2 = R1**2 + R2**2 + dz**2
    Y2 = 2*R1*R2
    ai = quad(Ar_Nonsing_integrand, 0, np.pi, args=(X2, Y2, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)
    return 2 * ai[0]*1j / (2 * np.pi)  # factor of 2 emulates 0 to 2*pi, divisor enforces average

def Ar_Nonsing_integralDiag(R, drdz, wz, k):

    return Ar_Nonsing_integral(R1=R, R2=R, dz=0, k=k)

@njit
def Ar_NonsingDiag_integrandDbl(dz, phi, R, drdz, k):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return np.cos(phi) * np.sin(k * Del) / Del

def Ar_NonsingDiag_integralDbl(R, drdz, wz, k):

    angint = dblquad(Ar_NonsingDiag_integrandDbl, \
                     0, np.pi, lambda phi: -wz / 2, lambda phi: +wz / 2, args=(R, drdz, k),
                      epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=quadlimit)

    return 1 / wz * angint[0] / np.pi * 1j

@njit
def Ar_NonsingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.cos(phi) * np.sin(k*Del) / Del

    return result/len(X2Y2s)

def Ar_NonsingOffdiag_integralMulti(r1, r2, dz,\
                                           drdz1,drdz2,\
                                          wz1,wz2, k):

    # Sample multiple times inside each annulus, evaluating in the middle of the other
    X2Y2s=[]
    for s1 in MultiSamples:
        R1 = (r1 + s1*drdz1*wz1/2)
        dZ = (dz + s1*wz1/2)
        X2 = R1 ** 2 + r2 ** 2 + dZ ** 2
        Y2 = 2 * R1 * r2
        X2Y2s.append((X2,Y2))
    for s2 in MultiSamples:
        R2 = (r2 + s2*drdz2*wz2/2)
        dZ = (dz + s2*wz2/2)
        X2 = r1 ** 2 + R2 ** 2 + dZ ** 2
        Y2 = 2 * r1 * R2
        X2Y2s.append((X2,Y2))

    ai = quad(Ar_NonsingOffdiag_integrandMulti, 0, np.pi, args=(k,) + tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average
    return 2 * ai[0] / (2 * np.pi) * 1j

#--- Excitation profiles

class EFieldExcitation(object): #For inheritance only

    _kmin = 1e-4

class ConstantEField(EFieldExcitation):

    def __init__(self,ex=0,ey=0,ez=1): #Basic constant excitation

        self.ex = ex
        self.ey = ey
        self.ez = ez

    def __call__(self,x,y,z):

        ones = np.ones(np.broadcast(x,y,z).shape) # a space-filling array
        Ex = self.ex * ones
        Ey = self.ey * ones
        Ez = self.ez * ones

        return Ex,Ey,Ez

class EBesselBeamFF(EFieldExcitation):

    def __init__(self,angle=60,k=2 * np.pi * .003, z0=0):

        if k==0: k=self._kmin

        anglerad = np.deg2rad(angle)
        self.k = k
        self.q = k * np.sin(anglerad)
        self.kz = -k * np.cos(anglerad)
        self.z0=z0

    def __call__(self,x,y,z):

        r = np.sqrt(x ** 2 + y ** 2)
        CosPhi = x/r; SinPhi = y/r #Sine and cosine of azimuthal angle `phi`
        Exp = np.exp(1j * self.kz * (z - self.z0))
        J0 = j0(self.q * r)
        J1 = j1(self.q*r)

        Ez = self.q * J0 * Exp / self.k
        Er = self.kz*J1*Exp/self.k

        # `Ex/Er =  \hat{r} \cdot \hat{x} = \cos{\phi}
        # `Ey/Er =  \hat{r} \cdot \hat{y} = \sin{\phi}
        Ex = CosPhi * Er
        Ey = SinPhi * Er

        return Ex,Ey,Ez

class EPlaneWaveFF(EFieldExcitation):

    def __init__(self, angle=60, k=2 * np.pi * .003, e0 = -1):

        if k==0: k=self._kmin

        anglerad = np.deg2rad(angle)
        self.k = k
        self.q = -k * np.sin(anglerad)
        self.kz = -k * np.cos(anglerad) #Wave vector is actually antiparallel to `\hat{\theta}}`-direction
        self.e0 = e0

    def __call__(self, x, y, z):

        PW = np.exp( 1j*(self.q * x
                         + self.kz*z )) #plane wave profile

        CosPhi = -self.kz / self.k
        Ex = self.e0 * CosPhi * PW
        Ey = Ex*0
        SinPhi = -self.q / self.k
        Ez = self.e0 * SinPhi * PW #it's clear that `Ex^2 + Ez^2 = e0^2`

        return Ex,Ey,Ez

#--- Discretization of probe

def get_BoR_radii(zs, L=1000, z0=0, a=1, taper_angle=20, geometry='cone', Rtop=0):

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
        R=np.sqrt(b*float(a)) #Maintains curvature of 1/a at tip
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

class BodyOfRevolution(object):

    def __init__(self,Rs,zs,Nsubnodes=6,closed=False,display=True,
                 remesh=True,quadrature='GL',interpolation='quadratic'):

        # --- Checks
        assert len(Rs) == len(zs)
        assert np.min(Rs)>0 and np.min(zs)>=0
        assert not Nsubnodes % 2, 'Number of subnode divisions must be even!'
        self.Nsubnodes = Nsubnodes

        #--- Enforce open or closed geometry
        # radii of (origin + nodes + terminus) will have length = Nnodes+2
        if closed:
            Rmin = np.min(Rs)*1e-2
            if Rs[0]!=0:
                dzdR = np.diff(zs)[0]/np.diff(Rs)[0]
                dR = Rmin - Rs[0]
                z0 = zs[0] + dzdR * dR
                zs = np.append([z0], zs)
                Rs = np.append([Rmin], Rs)
                zs -= z0 #In case our preprended z0 went negative
            if Rs[-1]!=0:
                dzdR = np.diff(zs)[-1]/np.diff(Rs)[-1]
                dR = Rmin - Rs[-1]
                zT = zs[-1] + dzdR * dR
                zs = np.append(zs, [zT])
                Rs = np.append(Rs, [Rmin])

        #--- Re-mesh to quadrature-evaluatd values of annualar coordinate "t", if requested
        if remesh:
            if display: Logger.write('Re-meshing provided geometry to quadrature points along annular coordinate `t`...')
            # Compute the annular coordinates for each supplied radius
            drs = np.gradient(Rs)
            dzs = np.gradient(zs)
            dts = np.sqrt(drs**2+dzs**2)
            ts = np.cumsum(dts)
            self.ts0=ts
            self.zs0=zs
            self.Rs0=Rs

            # Build interpolators and determine target interpolation points
            zs = AWA(zs,axes=[ts]); Rs = AWA(Rs,axes=[ts])
            ts,_ = numrec.GetQuadrature(len(ts),xmin=ts.min(),xmax=ts.max(),
                                        quadrature=quadrature)

            # Interpolate, replacing earlier values of `zs`, `Rs`
            zs = zs.interpolate_axis(ts,axis=0,kind=interpolation)
            Rs = Rs.interpolate_axis(ts,axis=0,kind=interpolation)

        # --- Assign node coordinates and internal intervals
        self.node_terminal_Rs = Rs #Use whatever terminal values were supplied
        self.node_terminal_zs = zs  # Use whatever terminal values were supplied
        self.node_Rs = np.array(Rs[1:-1]) #Internal points comprise the nodes
        self.node_zs = np.array(zs[1:-1])
        self.Nnodes = len(self.node_zs)

        if display: Logger.write('Discretizing body of revolution over %i annular nodes and %i subnodes...'\
                                 %(self.Nnodes,self.Nsubnodes))

        self.node_terminal_zs = zs
        self.node_interval_dzs = np.diff(self.node_terminal_zs)  # spacing between nodes, of length `Nnodes+1`
        self.node_interval_dRs = np.diff(self.node_terminal_Rs)  # spacing between nodes, of length `Nnodes+1`

        # --- Evaluate all subnode coordinates and internal intervals
        self.pvals, self.pweights = numrec.GetQuadrature(xmin=-1, xmax=+1, quadrature='linear', N=self.Nsubnodes)
        self.Nsubnodes = len(self.pvals)

        subnode_shape = (self.Nnodes, self.Nsubnodes)
        self.subnode_zs = np.zeros(subnode_shape)
        self.subnode_dzs = np.zeros(subnode_shape)
        self.subnode_Rs = np.zeros(subnode_shape)
        self.subnode_dRs = np.zeros(subnode_shape)
        self.subnode_dts = np.zeros(subnode_shape)
        self.subnode_sin = np.zeros(subnode_shape)
        self.subnode_cos = np.zeros(subnode_shape)
        for i in range(self.Nnodes):
            for p in range(self.Nsubnodes):
                pval = self.pvals[p]
                dp = self.pweights[p]

                if pval < 0:
                    dz = self.node_interval_dzs[i]
                    dR = self.node_interval_dRs[i]
                else:
                    dz = self.node_interval_dzs[i + 1]
                    dR = self.node_interval_dRs[i + 1]
                self.subnode_dzs[i, p] = dzp = dp * dz
                self.subnode_zs[i, p] = self.node_zs[i] + pval * dz
                self.subnode_dRs[i, p] = dRp = dp * dR
                self.subnode_Rs[i, p] = self.node_Rs[i] + pval * dR

                self.subnode_dts[i, p] = dtp = np.sqrt(dRp ** 2 + dzp ** 2)

                self.subnode_sin[i, p] = dRp / dtp  # opp/hyp
                self.subnode_cos[i, p] = dzp / dtp  # adj/hyp

        # --- Populate triangle basis functions
        # Integrate subnode dts between nodes (pval<0), length will be `Nnodes + 1`
        node_interval_dts = np.sum(self.subnode_dts[:, self.pvals < 0], axis=-1)
        self.node_interval_dts = np.append(node_interval_dts,
                                              [np.sum(self.subnode_dts[-1, self.pvals > 0],
                                                      axis=-1)])  # add the interval between final node and terminus

        self.subnode_Ts = np.zeros(subnode_shape)
        self.subnode_dTdts = np.zeros(subnode_shape)
        for p in range(self.Nsubnodes):
            pval = self.pvals[p]
            for i in range(self.Nnodes):

                self.subnode_Ts[i,p] = 1 - np.abs(pval)  # Equals 1/2 at pval=+/-1/2

                if pval < 0: dt = self.node_interval_dts[i]
                else:  dt = self.node_interval_dts[i + 1]
                # Equals +/- 1/dt; this way integrates to 1 over `t \in [ -dt[i],dt[i+1] ]` (pval=-/+1)
                self.subnode_dTdts[i,p] = -np.sign(pval) / dt

        # --- Nodal averages w.r.t. Triangle functions
        self.node_dts = np.sum(1/2 * self.subnode_dts,axis=-1) #The triangle function equivalently gives half of integrated subnodes
        self.node_cos = np.sum(self.subnode_Ts * self.subnode_cos * self.subnode_dts,axis=-1) / self.node_dts #similarly, average w.r.t. the triangle function
        self.node_sin = np.sum(self.subnode_Ts * self.subnode_sin * self.subnode_dts,axis=-1) / self.node_dts

    def get_self_impedance(self,k,**kwargs):

        return ImpedanceMatrix(self,k,mirror=False,nonsingular=True,**kwargs)

    def get_mirror_impedance(self,k=0,gap=1,nonsingular=False,sommerfeld_rp=None,**kwargs):

        if sommerfeld_rp: raise NotImplementedError
        else: return ImpedanceMatrix(self,k,gap=gap,mirror=True,nonsingular=nonsingular,**kwargs)

    def get_excitation_t_phi(self,efield_excitation,alpha=0,Nphis=24): #`alpha` is the azimuthal harmonic

        assert isinstance(efield_excitation,EFieldExcitation),\
            'Excitation source must be an instance of `EFieldExcitation`!'

        Z = self.subnode_zs[np.newaxis,:,:] # shape is (1, Nnodes,Nsubnodes)
        R = self.subnode_Rs[np.newaxis,:,:]
        dPhi = 2*np.pi/Nphis #If source has cylindrical symmetry, no need for `Nphis>1` (and excitations for `alpha!=0` would be zero)
        Phi = np.arange(0,2*np.pi,dPhi)[:,np.newaxis,np.newaxis]
        ExpAlpha = np.exp(-1j*alpha*Phi)
        CosTheta = self.subnode_cos[np.newaxis, :, :] # Note: `\theta` was also elsewhere called `\gamma`
        SinTheta = self.subnode_sin[np.newaxis, :, :]

        CosPhi = np.cos(Phi)
        SinPhi = np.sin(Phi)
        X = R*CosPhi
        Y = R*SinPhi

        Ex,Ey,Ez = efield_excitation(X,Y,Z)

        # Now we have to convert into `et`, `ephi`
        # Use `er` as an intermediary, since `\hat{r} = \hat{x} * \cos(\phi) + \hat{y} * \sin(\phi)
        Er = Ex * CosPhi + Ey * SinPhi

        # Compute Et and Vt
        #`\hat{t}=\hat{z] \cos\theta + \hat{r} \sin\theta`, where `theta=0` is pointing vertically
        Et = Ez * CosTheta + Er * SinTheta
        EtPhiAvg = np.mean(Et * ExpAlpha, axis=0) #Take alpha-average across `Phi` axis
        Vt = np.sum( EtPhiAvg * self.subnode_Ts * self.subnode_dts, axis=-1) #for each node, integrate over all subnodes

        # Compute Ephi and Vphi
        # Since `\hat{x}` is along CosPhi and `\hat{y}` along SinPhi, `\hat{phi} = \hat{y} CosPhi - \hat{x} SinPhi`
        Ephi = -Ex * SinPhi + Ey * CosPhi
        EphiPhiAvg = np.mean(Ephi * ExpAlpha, axis=0) #Take alpha-average across `Phi` axis
        Vphi = np.sum( EphiPhiAvg * self.subnode_Ts * self.subnode_dts, axis=-1) #for each node, integrate over all subnodes

        # Return the "generalized potential" across each node
        # We want two column vectors (which can later be stacked)
        # Remember that we will want `j = solve(Z, -excitation)`
        return np.matrix( Vt ).T,np.matrix( Vphi ).T

    #Legacy definition
    get_excitation = get_excitation_t_phi

    #2023.05.25 --- This function is retired
    def get_excitation_old(self,Er,Ez):

        assert hasattr(Er,'__call__') and hasattr(Ez,'__call__'),\
            '`Er` and `Ez` must be vectorized functions of coordinates (r,z)!'

        zs_flat = self.subnode_zs.flatten()
        Rs_flat = self.subnode_Rs.flatten()
        Ezs = Ez(Rs_flat,zs_flat).reshape( (self.Nnodes,self.Nsubnodes) )
        Ers = Er(Rs_flat,zs_flat).reshape( (self.Nnodes,self.Nsubnodes) )
        #`\hat{t}=\hat{z] \cos\theta + \hat{r} \sin\theta`, where `theta=0` is pointing vertically
        Ets = Ezs * self.subnode_cos + Ers * self.subnode_sin
        Vts = np.sum( Ets * self.subnode_Ts * self.subnode_dts, axis=-1) #for each node, integrate all subnodes

        # Return the "generalized potential" across each node
        return np.matrix( Vts ).T #Remember that we will want `j = solve(Z, -excitation)`

#Legacy name
Discretization = BodyOfRevolution


def ImpedanceMatrix(D,k,gap=1, mirror=False,\
                    nonsingular=True, display=True, \
                  coul_kernels=(Coul_SingDiag_integralDbl,\
                                Coul_SingOffdiag_integral,\
                                Coul_NonsingDiag_integralDbl,\
                                Coul_Nonsing_integral), \
                  faraday_kernels=(Ar_SingDiag_integralDbl,\
                                Ar_SingOffdiag_integral,\
                                Ar_NonsingDiag_integralDbl,\
                                Ar_Nonsing_integral)):

    # Here are the standard integral calculators
    # Coulomb integrals:
    # CoulSingDiag = Coul_SingDiag_integralDbl,
    # CoulSingOffdiag = Coul_SingOffdiag_integral
    # CoulNonsingDiag = Coul_NonsingDiag_integralDbl
    # CoulNonsingOffdiag = Coul_Nonsing_integral

    # Faraday integrals:
    # ArSingDiag = Ar_SingDiag_integralDbl
    # ArSingOffdiag = Ar_SingOffdiag_integral
    # ArNonsingDiag = Ar_NonsingDiag_integralDbl
    # ArNonsingOffdiag = Ar_Nonsing_integral

    if k==0:
        type1 = 'quasistatic'
        nonsingular = False #would be zero anyway
    else:
        type1 = 'dynamic'

    if mirror:
        mirror_sign=-1
        type2='mirror'
    else:
        mirror_sign=+1
        type2='self'

    if display: Logger.write('Preparing %s %s impedance matrix at k=%1.2G...'%(type1,type2,k))

    CoulSingDiag,CoulSingOffdiag,\
        CoulNonsingDiag, CoulNonsingOffdiag = coul_kernels
    ArSingDiag,ArSingOffdiag,\
        ArNonsingDiag, ArNonsingOffdiag = faraday_kernels

    #-- Diagnostic stuff
    global diag_G_couls, diag_G_farads,used_Rs,used_zs, NGcomputed
    diag_G_couls =[]
    diag_G_farads=[]
    used_Rs=[]
    used_zs=[]
    NGcomputed=0
    G_coul_cache={}
    G_farad_cache={}

    #-- The loop
    Zmat = np.zeros( (D.Nnodes,)*2, dtype=complex)
    t0 = time.time()
    for i in range(D.Nnodes):

        if display:
            progress=i / D.Nnodes*100
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write('\tProgress: %1.2f%%'%progress)
            sys.stdout.flush()

        for j in range(D.Nnodes):
            if j < i: continue  # only do the upper triangle

            # Average over all segments of both triangle basis functions at i,j
            for p in range(D.Nsubnodes):

                zp = D.subnode_zs[i,p] + gap
                dzp = D.subnode_dzs[i,p]
                dtp = D.subnode_dts[i,p]
                Rp = D.subnode_Rs[i,p]

                Tp = D.subnode_Ts[i,p]
                dTdtp = D.subnode_dTdts[i,p]
                Sp = D.subnode_sin[i,p]
                Cp = D.subnode_cos[i,p]

                drp = D.subnode_dRs[i,p]
                drdzp = drp / dzp

                used_zs.append(zp)
                used_Rs.append(Rp)

                for q in range(D.Nsubnodes):
                    zq = D.subnode_zs[j,q] + gap
                    dzq = D.subnode_dzs[j,q]
                    dtq = D.subnode_dts[j,q]
                    Rq = D.subnode_Rs[j,q]

                    Tq = D.subnode_Ts[j,q]
                    dTdtq = D.subnode_dTdts[j,q]
                    Sq = D.subnode_sin[j,q]
                    Cq = D.subnode_cos[j,q]

                    dz = np.abs(zp - mirror_sign * zq)

                    #-- Evaluate kernels
                    Gkey = tuple(sorted((Rp,Rq))) #This key is the only unique identifier for a Green's kernel (Rp,Rq can be interchanged)
                    if Gkey in G_coul_cache: #Try to load kernel from cache
                        G_coul = G_coul_cache[Gkey]
                        G_farad = G_farad_cache[Gkey]

                    else: #Compute anew with quadrature
                        NGcomputed+=1

                        if Rp<0 or Rq<0:
                            G_coul=G_farad=0

                        elif dz < 1e-6 * Rp: #this implies p=q, dp=dq, and Rp=Rq, wz1=wz2 etc.
                            G_coul = CoulSingDiag(Rp, drdzp, np.abs(dzp), k)
                            diag_G_couls.append(G_coul)
                            if nonsingular: G_coul += CoulNonsingDiag(Rp, drdzp, np.abs(dzp), k)

                            G_farad = ArSingDiag(Rp, drdzp, np.abs(dzp), k)
                            diag_G_farads.append(G_farad)
                            if nonsingular: G_farad += ArNonsingDiag(Rp, drdzp, np.abs(dzp), k)

                        else:
                            G_coul = CoulSingOffdiag(Rp, Rq, dz, k)
                            if nonsingular: G_coul += CoulNonsingOffdiag(Rp, Rq, dz, k)

                            G_farad = ArSingOffdiag(Rp, Rq, dz, k)
                            if nonsingular: G_farad += ArNonsingOffdiag(Rp, Rq, dz, k)

                        # store in cache
                        G_coul_cache[Gkey] = G_coul
                        G_farad_cache[Gkey] = G_farad

                    val = ( k**2 * Tp*Tq * (Sp * Sq * G_farad * mirror_sign
                                                     + Cp * Cq * G_coul)
                             - dTdtp * dTdtq * G_coul * mirror_sign)

                    if np.isnan(val): raise ValueError

                    Zmat[i, j] += val * dtp*dtq # #This is multiplying by "length of segment" in each annulus

    dt = time.time() - t0
    if display:
        Logger.write('\tTotal quadrature time: %1.2fs, time per quadrature evaluation: %1.2Es' % (dt, dt / D.Nnodes ** 2))

    triu_inds = np.triu_indices(D.Nnodes, k=1)
    tril_inds = (triu_inds[1], triu_inds[0])
    Zmat[tril_inds] = Zmat[triu_inds]

    #alpha = (4*np.pi) #/k_alpha
    alpha = -1 #This is pursuant to formulation in terms of accumlated charge `Q`

    return alpha * Zmat #As formulated, this impedance operates on a vector of "accumulated charge" on BOR up to index n