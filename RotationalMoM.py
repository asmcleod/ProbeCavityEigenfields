import sys
import numpy as np
from scipy.special import j0,j1,ellipk
from scipy.integrate import quad,nquad
from numba import njit
from common.log import Logger
from common import numerical_recipes as numrec
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
def Spatial_Coul_Sing_integrand_regularized(phi, X2, Y2, k):
    dz = 0
    Del = np.sqrt(X2 + dz ** 2 - Y2 * np.cos(phi))

    return (np.cos(k * Del) - 1) / Del

def Spatial_Coul_SingDiag_integral(R, drdz, wz, k):

    X2 = Y2 = 2 * R ** 2
    angint = quad(Spatial_Coul_Sing_integrand_regularized, 0, np.pi, args=(X2, Y2, k),
                  epsabs=1.49e-08,
                  epsrel=1.49e-08, limit=quadlimit)

    # Second and third rows are the correction proportional to `(w*drdz)**2`
    logpart = 2/R * (1 + np.log(16 * R / wz)) \
                + wz**2 * drdz**2 / (36 * R**3) \
                    * (2 * (1 + 3 * np.log(16 * R / wz)) -9 )

    return angint[0] / np.pi + logpart / (2 * np.pi)

@njit
def Spatial_Coul_Sing_integrandDbl_regularized(dz, phi, R, drdz, k):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return (np.cos(k * Del) - 1) / Del

def Spatial_Coul_SingDiag_integralDbl(R, drdz, wz, k):

    angint = dblquad(Spatial_Coul_Sing_integrandDbl_regularized, \
                     0, np.pi, lambda phi: -wz / 2, lambda phi: +wz / 2, args=(R, drdz, k),
                      epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=quadlimit)

    # Second and third rows are the correction proportional to `(w*drdz)**2`
    logpart = 2 / R * (1 + np.log(16 * R / wz)) \
                + wz ** 2 * drdz ** 2 / (36 * R ** 3) \
                   * (2 * (1 + 3 * np.log(16 * R / wz)) - 9)

    return 1 / wz * angint[0] / np.pi + logpart / (2 * np.pi)

#TODO: diagonal-adjacent integrals

@njit
def Spatial_Coul_SingOffdiag_integrand(phi, X2,Y2,k):

    Del = np.sqrt(X2 - Y2 * np.cos(phi))

    return np.cos(k*Del) / Del

def Spatial_Coul_SingOffdiag_integral(R1, R2, dz, k):

    X2 = R1**2 + R2**2 + dz**2
    Y2 = 2*R1*R2
    cr = quad(Spatial_Coul_SingOffdiag_integrand, 0, np.pi, args=(X2, Y2, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)
    return 2 * cr[0] / (2 * np.pi)  # factor of 2 emulates 0 to 2*pi, divisor enforces average

#NEW: a trapezoidal version of - spatial coul singular off-diag

@njit
def Spatial_Coul_SingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.cos(k*Del) / Del

    return result/len(X2Y2s)

def Spatial_Coul_SingOffdiag_integralMulti(r1, r2, dz,\
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

    cr = quad(Spatial_Coul_SingOffdiag_integrandMulti, 0, np.pi, args=(k,)+tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average, N is trapezoid rule
    return 2 * cr[0] / (2 * np.pi)

#-- Coulomb, nonsingular

@njit
def Spatial_Coul_Nonsing_integrand(phi, X2, Y2, k):

    Del = np.sqrt(X2 + - Y2 * np.cos(phi))

    return np.sin(k * Del) / Del

def Spatial_Coul_Nonsing_integral(R1, R2, dz, k):

    X2 = R1 ** 2 + R2 ** 2 + dz ** 2
    Y2 = 2 * R1 * R2

    angint = quad(Spatial_Coul_Nonsing_integrand, 0, np.pi, args=(X2, Y2, k),
                  epsabs=1.49e-08,
                  epsrel=1.49e-08, limit=quadlimit)

    return angint[0] / np.pi * 1j

def Spatial_Coul_Nonsing_integralDiag(R, drdz, wz, k):

    return Spatial_Coul_Nonsing_integral(R1=R, R2=R, dz=0, k=k)

@njit
def Spatial_Coul_NonsingDiag_integrandDbl(dz, phi, R, drdz, k):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return np.sin(k * Del) / Del

def Spatial_Coul_NonsingDiag_integralDbl(R, drdz, wz, k):

    angint = dblquad(Spatial_Coul_NonsingDiag_integrandDbl, \
                     0, np.pi, lambda phi: -wz / 2, lambda phi: +wz / 2, args=(R, drdz, k),
                      epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=quadlimit)

    return 1 / wz * angint[0] / np.pi * 1j

@njit
def Spatial_Coul_NonsingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.sin(k*Del) / Del

    return result/len(X2Y2s)

def Spatial_Coul_NonsingOffdiag_integralMulti(r1, r2, dz,\
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

    ci = quad(Spatial_Coul_NonsingOffdiag_integrandMulti, 0, np.pi, args=(k,)+tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average, N is trapezoid rule
    return 2 * ci[0] / (2 * np.pi) * 1j

#-- Faraday, singular

@njit
def Spatial_Ar_Sing_integrand_regularized(phi, X2, Y2, k):
    dz = 0
    Del = np.sqrt(X2 + dz ** 2 - Y2 * np.cos(phi))

    return (np.cos(phi)*np.cos(k * Del) - 1) / Del

def Spatial_Ar_SingDiag_integral(R, drdz, wz, k):
    X2 = Y2 = 2 * R ** 2
    angint = quad(Spatial_Ar_Sing_integrand_regularized, 0, np.pi, args=(X2, Y2, k),
                  epsabs=1.49e-08,
                  epsrel=1.49e-08, limit=quadlimit)

    # Second and third rows are the correction proportional to `(w*drdz)**2`
    logpart = 2 / R * (1 + np.log(16 * R / wz)) \
                + wz ** 2 * drdz ** 2 / (36 * R ** 3) \
                   * (2 * (1 + 3 * np.log(16 * R / wz)) - 9)

    return angint[0] / np.pi + logpart / (2 * np.pi)

@njit
def Spatial_Ar_Sing_integrandDbl_regularized(dz, phi, R, drdz, k):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return (np.cos(phi)*np.cos(k * Del) - 1) / Del

def Spatial_Ar_SingDiag_integralDbl(R, drdz, wz, k):

    angint = dblquad(Spatial_Ar_Sing_integrandDbl_regularized, \
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
def Spatial_Ar_SingOffdiag_integrand(phi, X2,Y2,k):

    Del = np.sqrt(X2 - Y2 * np.cos(phi))

    return np.cos(phi)*np.cos(k*Del) / Del

def Spatial_Ar_SingOffdiag_integral(R1, R2, dz, k):

    X2 = R1**2 + R2**2 + dz**2
    Y2 = 2*R1*R2
    ar = quad(Spatial_Ar_SingOffdiag_integrand, 0, np.pi, args=(X2, Y2, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)
    return 2 * ar[0] / (2 * np.pi)  # factor of 2 emulates 0 to 2*pi, divisor enforces average

@njit
def Spatial_Ar_SingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.cos(phi) * np.cos(k*Del) / Del

    return result/len(X2Y2s)

def Spatial_Ar_SingOffdiag_integralMulti(r1, r2, dz,\
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

    ar = quad(Spatial_Ar_SingOffdiag_integrandMulti, 0, np.pi, args=(k,)+tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average
    return 2 * ar[0] / (2 * np.pi)

#-- Faraday, nonsingular

@njit
def Spatial_Ar_Nonsing_integrand(phi, X2,Y2,k):

    Del = np.sqrt(X2 - Y2 * np.cos(phi))

    return np.cos(phi)*np.sin(k*Del) / Del

def Spatial_Ar_Nonsing_integral(R1, R2, dz, k):

    X2 = R1**2 + R2**2 + dz**2
    Y2 = 2*R1*R2
    ai = quad(Spatial_Ar_Nonsing_integrand, 0, np.pi, args=(X2, Y2, k),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)
    return 2 * ai[0]*1j / (2 * np.pi)  # factor of 2 emulates 0 to 2*pi, divisor enforces average

def Spatial_Ar_Nonsing_integralDiag(R, drdz, wz, k):

    return Spatial_Ar_Nonsing_integral(R1=R, R2=R, dz=0, k=k)

@njit
def Spatial_Ar_NonsingDiag_integrandDbl(dz, phi, R, drdz, k):

    Rz = R + drdz * dz
    Del = np.sqrt(dz ** 2 + 2*Rz**2 * (1-np.cos(phi)))

    return np.cos(phi) * np.sin(k * Del) / Del

def Spatial_Ar_NonsingDiag_integralDbl(R, drdz, wz, k):

    angint = dblquad(Spatial_Ar_NonsingDiag_integrandDbl, \
                     0, np.pi, lambda phi: -wz / 2, lambda phi: +wz / 2, args=(R, drdz, k),
                      epsabs=1.49e-08,
                      epsrel=1.49e-08, limit=quadlimit)

    return 1 / wz * angint[0] / np.pi * 1j

@njit
def Spatial_Ar_NonsingOffdiag_integrandMulti(phi,k,*X2Y2s):

    result=0
    for X2,Y2 in X2Y2s:

        Del = np.sqrt(X2 - Y2 * np.cos(phi))
        result += np.cos(phi) * np.sin(k*Del) / Del

    return result/len(X2Y2s)

def Spatial_Ar_NonsingOffdiag_integralMulti(r1, r2, dz,\
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

    ai = quad(Spatial_Ar_NonsingOffdiag_integrandMulti, 0, np.pi, args=(k,) + tuple(X2Y2s),
              epsabs=1.49e-08,
              epsrel=1.49e-08, limit=quadlimit)

    # factor of 2 emulates 0 to 2*pi, divisor enforces average
    return 2 * ai[0] / (2 * np.pi) * 1j

#--- Matrix builders

def get_WUpperTri_matrix(wzs):
    "This is basically an indefinite integral calculator in z-axis."

    N = len(wzs)
    WUpperTri = np.matrix([wzs] * N).T  # weights change along row index
    WUpperTri[np.tril_indices(N, -1)] = 0  # make lower triangle zero below diagonal
    np.fill_diagonal(WUpperTri, .5 * wzs)  # half-weights on diagonal

    return WUpperTri

def MoM_matrix(zs,wzs,Rs,k,\
               gap=1,mirror=False,faraday=True,nonsingular=True,\
               coul_kernels=(Spatial_Coul_SingDiag_integralDbl,\
                             Spatial_Coul_SingOffdiag_integralMulti,\
                                Spatial_Coul_NonsingDiag_integralDbl,\
                                Spatial_Coul_NonsingOffdiag_integralMulti),\
               faraday_kernels=(Spatial_Ar_SingDiag_integralDbl,\
                                Spatial_Ar_SingOffdiag_integralMulti,\
                                Spatial_Ar_NonsingDiag_integralDbl,\
                                Spatial_Ar_NonsingOffdiag_integralMulti)):

    if mirror:
        mirror_sign=-1
        inttype='mirror'
    else:
        mirror_sign=+1
        inttype='self'

    CoulSingDiag,CoulSingOffdiag,\
        CoulNonsingDiag, CoulNonsingOffdiag = coul_kernels
    ArSingDiag,ArSingOffdiag,\
        ArNonsingDiag, ArNonsingOffdiag= faraday_kernels

    Logger.write('Computing singular part of %s interaction by spatial quadrature...'%inttype)

    Zs = zs + gap
    Nzs = len(Zs)
    drs = np.gradient(Rs) / wzs

    dRs1 = drs[np.newaxis, :]
    dRs2 = drs[:, np.newaxis]
    WUpperTri = get_WUpperTri_matrix(wzs)
    WLowerTri = WUpperTri.T

    global Coul, Ar
    Coul = np.zeros((Nzs,) * 2, dtype=np.complex)
    Ar = np.zeros((Nzs,) * 2, dtype=np.complex)

    t0 = time.time()
    for i in range(Nzs):
        for j in range(Nzs):
            if j < i: continue  # only do the upper triangle

            wz1,wz2 = wzs[i], wzs[j]
            R1, R2 = Rs[i], Rs[j]
            dRdz1, dRdz2 = drs[i], drs[j]
            z1, z2 = Zs[i], Zs[j]
            dz = np.abs(z2 - mirror_sign*z1)

            #-- Singular part
            if i==j and not mirror:
                Coul[i, i] += CoulSingDiag(R1, dRdz1, wz1, k)

                if faraday:
                    Ar[i,i] += ArSingDiag(R1, dRdz1, wz1, k)

                if not nonsingular: continue

                #-- Nonsingular part
                Coul[i,j] += CoulNonsingDiag(R1, dRdz1, wz1, k)

                if faraday:
                    Ar[i,j] += ArNonsingDiag(R1, dRdz1, wz1, k)

            else:
                Coul[i, j] += CoulSingOffdiag(R1, R2, dz, \
                                              dRdz1, dRdz2, \
                                              wz1, wz2, k)
                if faraday:
                    Ar[i,j] += ArSingOffdiag(R1, R2, dz, \
                                              dRdz1, dRdz2, \
                                              wz1, wz2, k)

                if not nonsingular: continue

                #-- Nonsingular part
                Coul[i,j] += CoulNonsingOffdiag(R1, R2, dz, \
                                              dRdz1, dRdz2, \
                                              wz1, wz2, k)
                if faraday:
                    Ar[i,j] += ArNonsingOffdiag(R1, R2, dz, \
                                              dRdz1, dRdz2, \
                                              wz1, wz2, k)

    dt = time.time() - t0
    Logger.write('\tTotal quadrature time: %1.2fs, time per quadrature evaluation: %1.2Es' % (dt, dt / Nzs ** 2))

    triu_inds = np.triu_indices(Nzs, k=1)
    tril_inds = [triu_inds[1], triu_inds[0]]
    Coul[tril_inds] = Coul[triu_inds]
    Ar[tril_inds] = Ar[triu_inds]

    FaradKernel = Coul - mirror_sign * dRs1 * dRs2 * Ar

    # Integrate both variables to obtain radiative kernel
    RadiativeKernel = -k**2 * (WLowerTri @ FaradKernel @ WUpperTri)

    # +P.build_interaction(mirror=False,FF=False,faraday=True,NF=True,\
    #                                          kappa_max=np.inf,kappa_min=kappa_min,Nkappas=Nkappas)

    return (mirror_sign*Coul + RadiativeKernel)

#--- Discretization of probe

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

class Discretization(object):

    def __init__(self,Rs,zs,Nsubnodes=6,closed=False,display=True):

        # --- Checks
        assert len(Rs) == len(zs)
        assert np.min(Rs)>0 and np.min(zs)>=0
        assert not Nsubnodes % 2, 'Number of subnode divisions must be even!'
        self.Nsubnodes = Nsubnodes

        #--- Enforce open or closed geometry
        # radii of origin + nodes + terminus, of length Nnodes+2
        if closed:
            Rmin = np.min(Rs)*1e-2
            if Rs[0]!=0:
                dzdR = np.diff(zs)[0]/np.diff(Rs)[0]
                z0 = -dzdR * Rs[0]
                zs = np.append([z0], zs)
                Rs = np.append([Rmin], Rs)
                zs -= z0 #In case our preprended z0 went negative
            if Rs[-1]!=0:
                dzdR = np.diff(zs)[-1]/np.diff(Rs)[-1]
                zT = +dzdR * Rs[-1]
                zs = np.append(zs, [zT])
                Rs = np.append(Rs, [Rmin])

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

    def get_excitation(self,Er,Ez):

        assert hasattr(Er,'__call__') and hasattr(Ez,'__call__'),\
            '`Er` and `Ez` must be vectorized functions of coordinates (r,z)!'

        zs_flat = self.subnode_zs.flatten()
        Rs_flat = self.subnode_Rs.flatten()
        Ezs = Ez(Rs_flat,zs_flat).reshape( (self.Nnodes,self.Nsubnodes) )
        Ers = Er(Rs_flat,zs_flat).reshape( (self.Nnodes,self.Nsubnodes) )
        Ets = Ezs * self.subnode_cos + Ers * self.subnode_sin
        Vts = np.sum( Ets * self.subnode_Ts * self.subnode_dts, axis=-1) #integrate along tangent direction

        return np.matrix( Vts ).T #Remember that we will want `j = solve(Z, -excitation)`

def ImpedanceMatrix(D,k,gap=1, mirror=False,\
                    nonsingular=True, display=True, \
                  coul_kernels=(Spatial_Coul_SingDiag_integralDbl,\
                                Spatial_Coul_SingOffdiag_integral,\
                                Spatial_Coul_NonsingDiag_integralDbl,\
                                Spatial_Coul_Nonsing_integral), \
                  faraday_kernels=(Spatial_Ar_SingDiag_integralDbl,\
                                Spatial_Ar_SingOffdiag_integral,\
                                Spatial_Ar_NonsingDiag_integralDbl,\
                                Spatial_Ar_Nonsing_integral)):

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

    if display: Logger.write('Preparing %s %s impedance matrix...'%(type1,type2))

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
    Zmat = np.zeros( (D.Nnodes,)*2, dtype=np.complex)
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

                    val = ( k**2 * Tp*Tq * (Sp * Sq * G_farad * mirror_sign \
                                                     + Cp * Cq * G_coul) \
                             - dTdtp * dTdtq * G_coul * mirror_sign)

                    if np.isnan(val): raise ValueError

                    Zmat[i, j] += val * dtp*dtq # #This is multiplying by "length of segment" in each annulus

    dt = time.time() - t0
    if display:
        Logger.write('\tTotal quadrature time: %1.2fs, time per quadrature evaluation: %1.2Es' % (dt, dt / D.Nnodes ** 2))

    triu_inds = np.triu_indices(D.Nnodes, k=1)
    tril_inds = [triu_inds[1], triu_inds[0]]
    Zmat[tril_inds] = Zmat[triu_inds]

    #alpha = (4*np.pi) #/k_alpha
    alpha = -1 #This is pursuant to formulation in terms of accumlated charge `Q`

    return alpha * Zmat #As formulated, this impedance operates on a vector of "accumulated charge" on BOR up to index n