
import numpy as np

import tokovi_drugic
import helpers
import tnsmodule

U = 2.5 # eV
V = 0.785 # eV
a = 3.51 # A
b = 15.79 # A
b2 = 1.927 # A
mu = 2.84 # eV
eps0 = 0.1 # eV

dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 2000, 10, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters1 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]

dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 500, 10, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters2 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]

omegas = np.linspace(-0.2,0.2,501)

beta0 = 150
scale = 1.01
betas = beta0/scale**np.arange(1,25)
ends = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
Ts = 1/betas

Nks = [50, 100, 150, 200, 250, 300]
Gammas = [0.01, 0.025, 0.05, 0.075]

for i, Nk in enumerate(Nks):
    for j, Gamma in enumerate(Gammas):
        Ny, Nx = Nk, Nk
        s = tnsmodule.TNS(a, b, b2, Ny, Nx, U, V, mu, parameters1, parameters2, eps0)
        

s.run2(betas, ends, Gamma, omegas)