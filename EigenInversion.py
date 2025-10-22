import numpy as np
import os
import time
import copy
import pickle
import functools
from numbers import Number
from scipy import linalg
from scipy.optimize import leastsq
from matplotlib import pyplot as plt
from common.log import Logger
from common import misc,plotting
from common import numerical_recipes as numrec
from common.baseclasses import AWA
import ProbeCavityEigenfields as PCE
from ProbeCavityEigenfields import ProbeSpectroscopy as PS


class EncodedEigenfieldsPredictor(object):

    """Simple predictor callable as `Predictor(qp,eps)` that will delivery a normalized signal
    as quickly as possible using the information from a `PS.EncodedEigenfields` instance."""

    def __init__(self, EncodedEigenfields,
                 zmin=.1, A=2, Nts=24,
                 harmonic=2,
                 qp_ref=1e8,
                 eps_ref=1e8,
                 Nmodes=15):

        assert isinstance(EncodedEigenfields,PS.EncodedEigenfields)

        # Demodulation quadrature nodes (and weights) in time
        self.ts, self.dts = numrec.GetQuadrature(N=Nts, xmin=-.5, xmax=0,
                                                 quadrature=numrec.GL)
        self.kernel = np.cos(2 * np.pi * harmonic * self.ts) * self.dts
        self.kernel -= np.mean(self.kernel)

        # Corresponding gap heights
        self.at_gaps = zmin + A * (1 + np.cos(2 * np.pi * self.ts))
        self.freq = EncodedEigenfields.get_probe().get_freq()
        self.EncodedEigenfields = EncodedEigenfields
        self.Nmodes = Nmodes
        self.Sref = self.set_Sref(qp_ref, eps_ref)

        self.is_vectorized = False # The `compute_signal` attribute is NOT vectorized for this Predictor

    def massage_arg(self, arg):  # Assume argument must have positive imaginary

        return arg.real + 1j * np.abs(arg.imag)

    def compute_signal(self, qp, eps):  # This simulates a 2D material

        # Inhibit negative imaginary for qp or eps
        qp = self.massage_arg(qp)
        eps = self.massage_arg(eps)

        # "Lookup" eigenfield scattering matrix
        R0 = self.EncodedEigenfields.interpolate_Rmat2D_from_library(qp, eps)

        # Compute scattered field at gap heights; output raw array
        erad_vs_gap = self.EncodedEigenfields.EradVsGap(self.at_gaps, freq=self.freq,
                                                        RMat0=R0, as_AWA=False, Nmodes=self.Nmodes)

        # here is the demodulation! at quadrature nodes
        erad_vs_gap = np.array(erad_vs_gap)
        Sn = 4 * np.sum(self.kernel * erad_vs_gap, axis=0)

        return complex(Sn)

    def set_Sref(self, qp_ref, eps_ref):

        S = self.compute_signal(qp_ref, eps_ref)
        self.S_ref = S  # Signal value to which we pretend data were referenced

    def compute_norm_signal(self, qp, eps):

        S = self.compute_signal(qp, eps)

        return S / self.S_ref

    def __call__(self, *args, **kwargs):

        return self.compute_norm_signal(*args, **kwargs)

class InvertibleEigenfieldsPredictor(object):
    """Simple predictor callable as `Predictor(qp,eps)` that will delivery a normalized signal
    as quickly as possible using the information from a `PS.EncodedEigenfields` instance."""

    def __init__(self, InvertibleEigenfields,
                 zmin=.1, A=2, Nts=24,
                 harmonic=2,
                 eps_ref=1e8,
                 Nmodes=20):
        assert isinstance(InvertibleEigenfields, PS.InvertibleEigenfields)

        # Demodulation quadrature nodes (and weights) in time
        self.ts, self.dts = numrec.GetQuadrature(N=Nts, xmin=-.5, xmax=0,
                                                 quadrature=numrec.GL)
        self.kernel = np.cos(2 * np.pi * harmonic * self.ts) * self.dts
        self.kernel -= np.mean(self.kernel)

        # Corresponding gap heights
        self.nodes = InvertibleEigenfields.get_demodulation_nodes(zmin=zmin, amplitude=A, quadrature=numrec.GL,
                                                                  harmonic=harmonic, Nts=Nts)
        self.freq = InvertibleEigenfields.get_probe().get_freq()
        self.InvertibleEigenfields = InvertibleEigenfields
        self.Nmodes=Nmodes
        self.Sref = self.set_Sref(eps_ref)

        self.is_vectorized = True # The `compute_signal` attribute is indeed vectorized for this Predictor

    def massage_arg(self, arg):  # Assume argument must have positive imaginary

        return arg.real + 1j * np.abs(arg.imag)

    def compute_signal(self, eps, **kwargs):  # This simulates a bulk material

        # Inhibit negative imaginary for qp or eps
        eps = self.massage_arg(eps)

        beta = (eps-1)/(eps+1)

        # Compute scattered field at gap heights; output raw array
        Sns = self.InvertibleEigenfields.get_Erad_from_nodes(beta=beta, nodes=self.nodes,
                                                                 Nmodes=self.Nmodes)
        Sn = np.sum(Sns,axis=-1) # Last axis will be summing across nodes

        return Sn

    def set_Sref(self, eps_ref ,**kwargs):
        S = self.compute_signal(eps_ref, **kwargs)
        self.S_ref = S  # Signal value to which we pretend data were referenced

    def compute_norm_signal(self, eps, **kwargs):

        S = self.compute_signal(eps, **kwargs)

        return S / self.S_ref

    def __call__(self, *args, **kwargs):
        return self.compute_norm_signal(*args, **kwargs)

class VariationalMaterial2D(object):

    def __init__(self, Predictor, freqs,
                 Nosc_eps=3, amp_eps=1,
                 Nosc_qp=3, amp_qp=0,
                 eps_vals=None,
                 qp_vals=None,
                 additional_normalization=None):

        self.Predictor = Predictor
        self.freqs = freqs
        self.R = None

        if eps_vals and not hasattr(eps_vals, '__len__'):
            eps_vals = [eps_vals] * len(freqs)
        self.eps_vals = eps_vals
        if qp_vals and not hasattr(qp_vals, '__len__'):
            qp_vals = [qp_vals] * len(freqs)
        self.qp_vals = qp_vals

        self.set_coarse_params_eps(Nosc_eps, amp=amp_eps)
        self.set_coarse_params_qp(Nosc_qp, amp=amp_qp)
        self.set_fine_params_eps(Nosc_eps * 3, amp=amp_eps * 0.1)
        self.set_fine_params_qp(Nosc_eps * 3, amp=amp_qp * 0.1)

        self.signal_mult = 1  # For qp fitting only

        self.optimize_target = 'eps_coarse'
        self.optimize_target_options = ['eps_coarse',
                                        'eps_fine',
                                        'qp_coarse',
                                        'qp_fine']
        self.additional_normalization = additional_normalization

        self.qpmax=1e12 # qp will never be allowed to exceed this value

        self.interrupt_error = False # Make this True to enable raising error upon keyboard interrupt during optimization

    def set_coarse_params_qp(self, Nosc, amp=0, eps0=0):

        df = np.max(self.freqs) - np.min(self.freqs)
        gamma = df / Nosc  # Let's have an oscillator every `gamma`
        amps = [amp] * Nosc
        gammas = [gamma] * Nosc
        f0s = np.linspace(np.min(self.freqs) + gamma/2,
                          np.max(self.freqs) - gamma/2,
                          Nosc)

        # First term will be offset
        params = [np.real(eps0), np.abs(np.imag(eps0))] \
                 + list(zip(amps, f0s, gammas))  # list of triplets
        self.qp_coarse_params = list(misc.flatten(params))  # flatten

    def set_coarse_params_eps(self, Nosc, amp=0, eps0=1):

        df = np.max(self.freqs) - np.min(self.freqs)
        gamma = df / Nosc  # Let's have an oscillator every `gamma`
        amps = [amp] * Nosc
        gammas = [gamma] * Nosc
        f0s = np.linspace(np.min(self.freqs) + gamma/2,
                          np.max(self.freqs) - gamma/2,
                          Nosc)
        print(f0s)

        # First terms will be offset
        params = [np.real(eps0), np.abs(np.imag(eps0))] \
                 + list(zip(amps, f0s, gammas))  # list of triplets
        self.eps_coarse_params = list(misc.flatten(params))  # flatten

    def faddeeva_oscillator(self, params):  # This is a gaussian oscillator that will be used for local (fine) fitting

        from NearFieldOptics.Materials.faddeeva import faddeeva

        assert len(params)==3
        amp, f0, gamma = params

        f = self.freqs
        z = (f - f0) / (np.sqrt(2) * gamma)
        osc = amp * 1j * faddeeva(z)

        return osc

    def set_fine_params_qp(self, Nosc, amp=0):

        self.qp_fine_amps = list(amp / (1 + np.arange(Nosc)))  # First will be offset
        df = np.max(self.freqs) - np.min(self.freqs)
        gamma = df/Nosc
        gamma_boost=1.5
        self.qp_fine_gammas = [gamma*gamma_boost] * Nosc  # Let's have an oscillator every `gamma`
        self.qp_fine_freqs = np.linspace(np.min(self.freqs) - gamma/2,
                                         np.max(self.freqs) + gamma/2,
                                         Nosc)
        self.qp_faddeeva_basis = [self.faddeeva_oscillator((1,f0,gamma)) \
                                    for f0,gamma in zip(self.qp_fine_freqs,
                                                   self.qp_fine_gammas)]

    def set_fine_params_eps(self, Nosc, amp=0):

        self.eps_fine_amps = list(amp / (1 + np.arange(Nosc)))  # First will be offset
        df = np.max(self.freqs) - np.min(self.freqs)
        gamma = df/Nosc
        gamma_boost=1.5
        self.eps_fine_gammas = [gamma*gamma_boost] * Nosc  # Let's have an oscillator every `gamma`
        self.eps_fine_freqs = np.linspace(np.min(self.freqs) - gamma/2,
                                          np.max(self.freqs) + gamma/2,
                                          Nosc)
        self.eps_faddeeva_basis = [self.faddeeva_oscillator((1,f0,gamma)) \
                                    for f0,gamma in zip(self.eps_fine_freqs,
                                                   self.eps_fine_gammas)]

    param_names = ['eps_coarse_params',
                   'eps_fine_amps',
                    'eps_faddeeva_basis',
                   'qp_coarse_params',
                   'qp_fine_amps',
                   'qp_faddeeva_basis']

    def get_params(self):

        return dict([(param_name,getattr(self,param_name)) \
                     for param_name in self.param_names])

    def set_params(self,d):

        for param_name in d:
            setattr(self,param_name,d[param_name])

    def oscillators(self, params):  # This is a lorentzian oscillator that will be used for broad (coarse) fitting

        Nosc = len(params) / 3
        assert Nosc == int(Nosc)

        f = self.freqs
        all_osc = (0 + 0j) * f
        for i in range(int(Nosc)):
            amp, f0, gamma = params[3 * i:3 * (i + 1)]
            amp = np.abs(amp)
            gamma = np.abs(gamma) * np.sqrt(2)
            osc = amp * f0 * gamma / (f0 ** 2 - f ** 2 - 1j * f * gamma)  # Non-gainful oscillators
            all_osc += osc

        return all_osc

    def get_qps(self):

        if self.qp_vals is not None: return self.qp_vals

        # Take offset from first of coarse parameters
        params_coarse = copy.copy(list(self.qp_coarse_params))
        eps2Dr = params_coarse.pop(0)
        eps2Di = params_coarse.pop(0)
        eps2D = eps2Dr + 1j * np.abs(eps2Di)

        # Coarse oscillators
        eps2D += self.oscillators(params_coarse)

        # Fine oscillators (all fixed freq and gamma)
        eps2D_faddeeva = np.sum([amp*fad for amp,fad \
                               in zip(self.qp_fine_amps,self.qp_faddeeva_basis)],
                              axis=0)
        eps2D += eps2D_faddeeva
        eps2D = eps2D.real + 1j * np.abs(eps2D.imag)

        qps = 1 / -eps2D # If eps2D==0, then qp will diverge (no polariton)
        
        # Replace any infinities with maximum value
        qps = np.where(np.isinf(qps),\
                       self.qpmax,qps)
        
        return qps

    def get_epss(self):

        if self.eps_vals is not None: return self.eps_vals

        # Take offset from first of coarse parameters
        params_coarse = copy.copy(list(self.eps_coarse_params))
        epsr = params_coarse.pop(0)
        epsi = params_coarse.pop(0)
        eps = epsr + 1j * np.abs(epsi)

        # Coarse oscillators
        eps += self.oscillators(params_coarse)

        # Fine oscillators (all fixed freq and gamma)
        eps_faddeeva = np.sum([amp*fad for amp,fad \
                               in zip(self.eps_fine_amps,self.eps_faddeeva_basis)],
                              axis=0)

        eps += eps_faddeeva
        eps = eps.real + 1j * np.abs(eps.imag)

        return eps

    def fix_qps(self):
        self.qp_vals = self.get_qps()

    def fix_epss(self):
        self.eps_vals = self.get_epss()

    def unfix_qps(self):
        self.qp_vals = None

    def unfix_epss(self):
        self.eps_vals = None

    def get_signal_mult(self, signal_mult_min=0.75):

        # map the interval [-infty,infty] to [min,infty]
        # while mapping 1 to 1
        x = self.signal_mult
        signal_mult_actual = signal_mult_min + (1 - signal_mult_min) * np.exp(0.1 * (x - 1))

        return signal_mult_actual

    def predict(self, eps_offset=None, qp_offset=None, qps_enabled=True):

        epss = self.get_epss()  # Whether hard-coded or targeted for fit, now get the values for evaluation
        if qps_enabled: qps = self.get_qps()  # Whether hard-coded or targeted for fit, now get the values for evaluation
        else: qps = [self.qpmax for eps in epss]

        if eps_offset: epss *= (1 + eps_offset)
        if qp_offset: qps *= (1 + qp_offset)

        if self.Predictor.is_vectorized: S_preds = self.Predictor(qp=qps, eps=epss)
        else: S_preds = [self.Predictor(qp=qp, eps=eps) for qp, eps in zip(qps, epss)]
        S_preds = np.array(S_preds)

        result = S_preds * self.get_signal_mult()

        if self.additional_normalization is not None:
            result /= self.additional_normalization

        return result

    def residual(self, params, S_targets_r, S_targets_i, S_targets_std, exp=0.5):
        "Infer by self context whether we are fitting eps, qs, and with how many oscillators."

        assert len(S_targets_r) == len(self.freqs)
        assert self.eps_vals is None or self.qp_vals is None, 'Nothing to optimize!'

        self.attach_params(params)
        S_preds = self.predict()
        S_targets = S_targets_r + 1j * S_targets_i

        Rs = (np.abs(S_preds - S_targets) / S_targets_std) ** exp
        self.R = np.sum(Rs)  # Store a metric for us to inspect later

        return Rs

    def attach_params(self, params):

        if self.optimize_target == 'eps_coarse':
            self.eps_coarse_params = params

        elif self.optimize_target == 'eps_fine':
            self.eps_fine_amps = params

        elif self.optimize_target == 'qp_coarse':
            if self.optimize_signal_mult:
                self.signal_mult = params[0]
                self.qp_coarse_params = params[1:]
            else: self.qp_coarse_params = params

        elif self.optimize_target == 'qp_fine':
            if self.optimize_signal_mult:
                self.signal_mult = params[0]
                self.qp_fine_amps = params[1:]
            else: self.qp_coarse_params = params

        else:
            raise ValueError('optimize target must be one of %s!' \
                             % self.optimize_target_options)

    def detach_params(self):

        if self.optimize_target == 'eps_coarse':
            params = self.eps_coarse_params

        elif self.optimize_target == 'eps_fine':
            params = self.eps_fine_amps

        elif self.optimize_target == 'qp_coarse':
            if self.optimize_signal_mult: params = [self.signal_mult]
            else: params=[]
            params += list(self.qp_coarse_params)

        elif self.optimize_target == 'qp_fine':
            if self.optimize_signal_mult: params = [self.signal_mult]
            else: params=[]
            params += list(self.qp_fine_amps)

        else:
            raise ValueError('optimize target must be one of %s!' \
                             % self.optimize_target_options)
        return params

    def optimize(self, S_targets, S_targets_std=None,
                 exp_start=0.5, exp_stop=2,
                 Nsubcycles=1, Ncycles=1,
                 factor=1, full_output=False, xtol=1e-3, ftol=1e-3,
                 randomization=1e-3, exit_criterion=1e-3,
                 plot_residuals=False,
                 optimize_signal_mult=False,
                 **kwargs):

        self.optimize_signal_mult = optimize_signal_mult

        S_targets = np.array(S_targets) # Make array type, so there is no AWA nonsense to slow us down
        S_targets_r = S_targets.real
        S_targets_i = S_targets.imag

        if S_targets_std is None:
            S_targets_std = 1
        else:
            S_targets_std = np.abs(S_targets_std)
        if hasattr(S_targets_std,'__len__'):
            S_targets_std = np.array(S_targets_std) # Make array type, so there is no AWA nonsense to slow us down

        params0 = self.detach_params()
        _ = self.residual(params0, S_targets_r, S_targets_i,
                          S_targets_std, exp=exp_start)  # Just to compute `self.R`
        all_params = [params0]
        all_R_vals = [self.R]

        # Exponents for residual to loop through in subcycles
        exps = np.linspace(exp_start, exp_stop, Nsubcycles)

        tstart = time.time()
        try:
            for n in range(Ncycles):
                print('Optimizing "%s" in cycle #%i...' % (self.optimize_target, n + 1))
                for m in range(Nsubcycles):
                    exp = exps[m]
                    print('Optimizing "%s" in subcycle #%i, exp=%1.2f...' % (self.optimize_target, m + 1, exp))

                    t0 = time.time()
                    params0 = self.detach_params()  # Whatever is stored will be our initial guess
                    params0 *= (1 + randomization * np.random.randn(len(params0)))
                    params = leastsq(self.residual,
                                     params0,
                                     args=(S_targets_r, S_targets_i, S_targets_std),
                                     factor=factor, full_output=full_output, xtol=xtol, ftol=ftol,
                                     **kwargs)[0]
                    self.attach_params(params)

                    if self.optimize_signal_mult:
                        print('Optimized signal_mult =', self.signal_mult)

                    # Examine the result
                    print('R=%1.2f; time elapsed (s) in cycle: %1.2f' % (self.R, time.time() - t0))
                    all_R_vals.append(self.R)
                    all_params.append(params)

                    if np.abs(self.R - all_R_vals[-2]) / self.R <= exit_criterion:
                        print('Exit criterion %1.2G satisfied...' % exit_criterion)
                        raise StopIteration

        except KeyboardInterrupt:
            print('Aborting optimization (with forward progress)...')
            if self.interrupt_error: raise

        except StopIteration: pass

        print('Time elapsed (s) for optimization:', time.time() - tstart)

        best_ind = np.argmin(all_R_vals)
        self.attach_params(all_params[best_ind])

        if plot_residuals:
            plt.figure()
            plt.plot(all_R_vals, marker='o')
            plt.title('Residual value vs iteration')
            plt.ylabel('Residual')
            plt.xlabel('Iteration')

    def optimize_eps_coarse(self, *args, **kwargs):

        self.fix_qps()
        self.optimize_target = 'eps_coarse'
        result = self.optimize(*args, **kwargs)
        self.unfix_qps()

        return result

    def optimize_qp_coarse(self, *args, **kwargs):

        self.fix_epss()
        self.optimize_target = 'qp_coarse'
        result = self.optimize(*args, **kwargs)
        self.unfix_epss()

        return result

    def optimize_eps_fine(self, *args, **kwargs):

        self.fix_qps()
        self.optimize_target = 'eps_fine'
        result = self.optimize(*args, **kwargs)
        self.unfix_qps()

        return result

    def optimize_qp_fine(self, *args, **kwargs):

        self.fix_epss()
        self.optimize_target = 'qp_fine'
        result = self.optimize(*args, **kwargs)
        self.unfix_epss()

        return result

    def estimate_error(self, S_targets_std, offset=.01):

        pred0 = self.predict()

        pred_eps_offset = self.predict(eps_offset=offset)
        dS_pred = (pred_eps_offset - pred0)
        deps_pred = self.get_epss() * offset
        deps = np.abs(S_targets_std * deps_pred / dS_pred)

        pred_qp_offset = self.predict(qp_offset=offset)
        dS_pred = pred_qp_offset - pred0
        dqp_pred = self.get_qps() * offset
        dqp = np.abs(S_targets_std * dqp_pred / dS_pred)

        # Now use the fact that `d(1/qp) = -dq/qp**2` to get a `deps` for 2D

        return deps, dqp