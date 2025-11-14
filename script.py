import numpy as np
import scipy.linalg as LA
import os

import numpy as np
#import os, scipy, mpmath
from numba import njit, prange
import warnings
from numba.core.errors import NumbaPerformanceWarning

# Suppress NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#os.chdir('/Users/ana/Desktop/ta2nise5/parameters')

#os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")
#os.chdir('/Users/ana/Desktop/ta2nise5/parameters')

def Rho0(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    for i in [4, 5]: rho0[i, i, :, :] = 1
    return rho0

def Rhoinfty(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    for i in range(6): rho0[i, i, :, :] = 2/6
    return rho0

def H_hopping(Kymesh, Kxmesh, a, b, file='parametri-kinetic.txt'):
    Ny, Nx = Kymesh.shape
    hop = np.zeros((6, 6, Ny, Nx), dtype='complex')
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)
            ad = t * np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
            hop[orb1 - 1, orb2 - 1] += ad
            if orb1 != orb2: hop[orb2 - 1, orb1 - 1] += ad.conjugate()
    return hop

def H_perturb(Kymesh, Kxmesh, a, b, file='perturbacija.txt'):
    return H_hopping(Kymesh, Kxmesh, a, b, file=file)

def Delta(Kxmesh, Nk, rho, i, j, x): 
    if type(x) == np.ndarray:
        return np.array([np.sum(rho[i, j] * np.exp(1j * Kxmesh * x1)) for x1 in x]) / Nk
    else: return np.sum(rho[i, j] * np.exp(1j * Kxmesh * x)) / Nk

def Phi(Kxmesh, Nk, rho, a):
    pos12 = np.array([0, a])
    pos34 = np.array([0, -a])
    phi = np.zeros(4, dtype='complex')
    phi[0] = np.sum(Delta(Kxmesh, Nk, rho, 0, 4, pos12))
    phi[1] = np.sum(Delta(Kxmesh, Nk, rho, 1, 4, pos12))
    phi[2] = np.sum(Delta(Kxmesh, Nk, rho, 2, 5, pos34))
    phi[3] = np.sum(Delta(Kxmesh, Nk, rho, 3, 5, pos34))
    return phi

def H_hartree(rho, Nk, U, V, file='parametri-interaction.txt'):
    hartree_k = np.zeros((6,6), dtype='complex')
    with open(file) as f:
        for line in f:
            [orb1, orb2] = list(map(float, line.split()))[-2:]
            orb1, orb2 = int(orb1-1), int(orb2-1)
            if orb1 == orb2: 
                hartree_k[orb1, orb1] += U * np.sum(rho[orb1,orb1])
            else:
                hartree_k[orb1, orb1] += 2 * V * np.sum(rho[orb2, orb2])
                hartree_k[orb2, orb2] += 2 * V * np.sum(rho[orb1, orb1])
    return hartree_k / Nk
    
def H_fock(Kxmesh, Nk, rho, a, V):
    fock = np.zeros(rho.shape, dtype='complex')
    phi = Phi(Kxmesh, Nk, rho, a)
    fock[0,4] = - V * Delta(Kxmesh, Nk, rho, 0, 4, 0) * (1 - np.exp(-1j * Kxmesh * a)) - \
                    V * phi[0] * np.exp(-1j * Kxmesh * a)
    fock[4,0] = fock[0,4].conjugate()
    fock[1,4] = - V * Delta(Kxmesh, Nk, rho, 1, 4, 0) * (1 - np.exp(-1j * Kxmesh * a)) - \
                    V * phi[1] * np.exp(-1j * Kxmesh * a)
    fock[4,1] = fock[1,4].conjugate()
    fock[2,5] = - V * Delta(Kxmesh, Nk, rho, 2, 5, 0) * (1 - np.exp(1j * Kxmesh * a)) - \
                    V * phi[2] * np.exp(1j * Kxmesh * a)
    fock[5,2] = fock[2,5].conjugate()
    fock[3,5] = - V * Delta(Kxmesh, Nk, rho, 3, 5, 0) * (1 - np.exp(1j * Kxmesh * a)) - \
                    V * phi[3] * np.exp(1j * Kxmesh * a)
    fock[5,3] = fock[3,5].conjugate()
    return fock

def H_diagonalize(Ny, Nx, hop, perturb, hartree, fock, T, mu, eps):
    H = hop + fock
    if eps != 0: H += perturb * eps
    energije, vecs = np.zeros((6, Ny, Nx)), np.zeros((6, 6, Ny, Nx), dtype='complex')
    fs = np.zeros((6, 6, Ny, Nx))
    for m in range(Ny):
        for n in range(Nx):
            en, v = LA.eigh(H[:, :, m, n] + hartree)
            energije[:, m, n] = en
            vecs[:, :, m, n] = v
            if T == 0: np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 0, 0, 0, 0]))
            elif T == 'infty': np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 1, 1, 1, 1])/3)
            else:
                np.fill_diagonal(fs[:, :, m, n], 1/(1 + np.exp((en - mu)/T)))
    return energije, vecs, fs

def H_diagonalize2(Ny, Nx, hop, perturb, hartree, fock, T, mu, eps):
    H = hop + fock
    if eps != 0: H += perturb * eps
    energije, vecs = np.zeros((6, Ny, Nx)), np.zeros((6, 6, Ny, Nx), dtype='complex')
    fs = np.zeros((6, 6, Ny, Nx))

    for n in range(Nx):
        en, v = LA.eigh(H[:, :, 0, n] + hartree) # diag(en) = v @ h @ v.conj().T AND NOT diag(en) = v.conj().T @ h @ v !!!
        energije[:, 0, n] = en
        vecs[:, :, 0, n] = v
        if T == 0: np.fill_diagonal(fs[:, :, 0, n], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, 0, n], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, 0, n], 1/(1 + np.exp((en - mu)/T)))

        en, v = LA.eigh(H[:, :, Ny//2, n] + hartree)
        energije[:, Ny//2, n] = en
        vecs[:, :, Ny//2, n] = v
        if T == 0: np.fill_diagonal(fs[:, :, Ny//2, n], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, Ny//2, n], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, Ny//2, n], 1/(1 + np.exp((en - mu)/T)))

    for m in range(Ny):
        en, v = LA.eigh(H[:, :, m, 0] + hartree)
        energije[:, m, 0] = en
        vecs[:, :, m, 0] = v
        if T == 0: np.fill_diagonal(fs[:, :, m, 0], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, m, 0], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, m, 0], 1/(1 + np.exp((en - mu)/T)))

        en, v = LA.eigh(H[:, :, m, Nx//2] + hartree)
        energije[:, m, Nx//2] = en
        vecs[:, :, m, Nx//2] = v
        if T == 0: np.fill_diagonal(fs[:, :, m, Nx//2], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, m, Nx//2], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, m, Nx//2], 1/(1 + np.exp((en - mu)/T)))

    for m in range(1,Ny//2):
        for n in range(1,Nx//2):
            en, v = LA.eigh(H[:, :, m, n] + hartree)
            energije[:, m, n] = en
            energije[:, -m, -n] = en
            vecs[:, :, m, n] = v
            vecs[:, :, -m, -n] = v.conj()
            if T == 0:
                np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 0, 0, 0, 0]))
                np.fill_diagonal(fs[:, :, -m, -n], np.array([1, 1, 0, 0, 0, 0]))
            elif T == 'infty':
                np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 1, 1, 1, 1])/3)
                np.fill_diagonal(fs[:, :, -m, -n], np.array([1, 1, 1, 1, 1, 1])/3)
            else:
                np.fill_diagonal(fs[:, :, m, n], 1/(1 + np.exp((en - mu)/T)))
                np.fill_diagonal(fs[:, :, -m, -n], 1/(1 + np.exp((en - mu)/T)))

            en, v = LA.eigh(H[:, :, m, n + Nx//2] + hartree)
            energije[:, m, n + Nx//2] = en
            energije[:, -m, Nx//2 - n] = en
            vecs[:, :, m, n + Nx//2] = v
            vecs[:, :, -m, Nx//2 - n] = v.conj()
            if T == 0:
                np.fill_diagonal(fs[:, :, m, n + Nx//2], np.array([1, 1, 0, 0, 0, 0]))
                np.fill_diagonal(fs[:, :, -m, Nx//2 - n], np.array([1, 1, 0, 0, 0, 0]))
            elif T == 'infty':
                np.fill_diagonal(fs[:, :, m, n + Nx//2], np.array([1, 1, 1, 1, 1, 1])/3)
                np.fill_diagonal(fs[:, :, -m, Nx//2 - n], np.array([1, 1, 1, 1, 1, 1])/3)
            else:
                np.fill_diagonal(fs[:, :, m, n + Nx//2], 1/(1 + np.exp((en - mu)/T)))
                np.fill_diagonal(fs[:, :, -m, Nx//2 - n], 1/(1 + np.exp((en - mu)/T)))

    return energije, vecs, fs

def F(Ny, Nx, rho, hop, perturb, hartree, fock, T, mu, eps=0, colors='no', occupation='no', vectors='no'):
    energije, vecs, fs = H_diagonalize2(Ny, Nx, hop, perturb, hartree, fock, T, mu, eps)
    rho_new = np.einsum('ijkl,jmkl,mnkl-> inkl', vecs, fs, np.swapaxes(vecs.conj(), 0, 1))
    if colors == 'yes':
        barve = np.einsum('ijkl->jkl', np.abs(vecs[:4, :, :, :])**2)
        return rho_new, energije, np.max(np.abs(rho - rho_new)), barve
    if occupation == 'no': return rho_new, energije, np.max(np.abs(rho - rho_new))
    elif occupation == 'yes' and vectors == 'yes':
        return rho_new, energije, fs, vecs, np.max(np.abs(rho - rho_new))
    
def Occupation(rho):
    return (np.sum(np.diag(np.einsum('ijkl->ij', rho)))/(np.prod(rho.shape[-2:]))).real
    
def GS(Kxmesh, rho, hop, perturb, hartree, fock, mu, eps0, a, U, V, epsilon, maxiter=1000, N_epsilon=5):
    Ny, Nx = Kxmesh.shape
    err, N_iters = 1, 0
    while err > epsilon and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps = 0
        rho, _, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, 0, mu, eps=eps)
        fock = H_fock(Kxmesh, Nx*Ny, rho, a, V)
        hartree = H_hartree(rho, Nx*Ny, U, V)
        N_iters += 1
    rho, energije, fs, vecs, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, 0, mu, eps=eps, occupation='yes', vectors='yes')
    #print(f'Found ground state with error={err}, occupation error={np.abs(Occupation(rho)-2)}')
    return rho, energije, fs, vecs, err, Occupation(rho), fock, hartree

''''''''''''''''''''''''''

def Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, maxiter, mix, epsilon, eps0=0.0, N_epsilon=5):
    Ny, Nx = Kxmesh.shape
    err, N_iters = 1, 0
    while err > epsilon and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps0 = 0
        rho_new, _, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, T, mu, eps=eps)
        rho = rho_new * mix + rho * (1 - mix)
        fock = H_fock(Kxmesh, Nx*Ny, rho, a, V)
        hartree = H_hartree(rho, Nx*Ny, U, V)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, energije, fs, vecs, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, T, mu, occupation='yes', vectors='yes')
    return rho, energije, fs, vecs, fock, hartree, err, Occupation(rho)

def NewMu(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, faktor1=0.001):
    rho_a, energije_a, fs_a, vecs_a, fock_a, hartree_a, err_a, n_a = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, maxiter, mix, eps_last)
    #if np.abs(n_a - 2) < n_pass and err_a < eps_last:
    #    return rho_a, energije_a, fs_a, vecs_a, fock_a, hartree_a, err_a, n_a, mu
    n_b = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu + dmu, maxiter, mix, eps_last)[-1]
    chi = (n_b - n_a)/dmu
    if chi != 0: mu = mu - mix2 * (n_a - 2)/np.abs(chi)

    pogoj = False
    koraki = 0
    if np.abs(chi) > 0: faktor = (n_a - 2)/chi * mix3
    else: faktor = faktor1
    if chi >= 0:
        if n_a >= 2:
            sign = -1
        elif n_a < 2: sign = +1
    elif chi < 0:
        if n_a >= 2: sign = +1
        elif n_a < 2: sign = -1
        
    sgns = np.ones(2) * np.sign(n_a - 2)
    ns = np.array([0, n_a])
    mus = [0, mu]
    enough = False
    while sgns[0] == sgns[1]:
        if np.abs(n_a - 2) < n_pass and err_a < eps_last:
            enough = True
            break
        rho_b, energije_b, fs_b, vecs_b, fock_b, hartree_b, err_b, n_b = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu + faktor*koraki*sign, maxiter, mix, eps_last)
        if np.abs(n_b - 2) < n_pass and err_b < eps_last: return rho_b, energije_b, fs_b, vecs_b, fock_b, hartree_b, err_b, n_b,  mu + faktor*koraki*sign
        ns[0] = n_b
        mus[0] = mu + faktor*koraki*sign
        sgns[1] = np.sign(n_b - 2)
        if sgns[0] != sgns[1]: break
        if n_b < 2 and n_b < ns[1]: sign *= -1
        if n_b > 2 and n_b > ns[1]: sign *= -1
        ns = np.roll(ns, 1)
        mus = np.roll(mus, 1)
        sgns[1] = np.sign(n_b - 2)
        koraki +=1
        if np.abs(n_b - 2) < n_pass and err_b < eps_last:
            enough = True
            mu_mid = mu + faktor*koraki*sign
            break
        
    mus = np.sort(np.array([mu + faktor*koraki*sign, mu + faktor*(koraki-1)*sign]))
    ns = np.sort(np.array(ns))

    trials = 0
    while pogoj == False:
        mu_mid = (mus[0] + mus[1])/2
        if enough == True:
            break   
        n_mid = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu_mid, maxiter, mix, eps_last)[-1]
        if n_mid > 2: mus[1] = mu_mid
        elif n_mid < 2: mus[0] = mu_mid
        if np.abs(n_mid - 2) < n_pass: break
        trials += 1 
        if trials > max_trials: break
    rho, energije, fs, vecs, fock, hartree, err, n = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu_mid, maxiter_last, mix, eps_last)
    return rho, energije, fs, vecs, fock, hartree, err, n, mu_mid

def rotate_basis(mat, vecs):
    return np.einsum('jixy, jlxy, lkxy -> ikxy', vecs.conj(), mat, vecs)

def kinetic(file='parametri-kinetic.txt', extend=None):
    seznam = []
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            seznam.append([x, y, orb1, orb2, t])
            if extend == True:
                if orb1 != orb2:
                    seznam.append([-x, -y, orb2, orb1, t])
    return np.array(seznam)

def interaction(file='parametri-interaction.txt'):
    seznam = []
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2] = list(map(float, line.split()))
            seznam.append([x, y, orb1, orb2])
            if orb1 != orb2:
                seznam.append([-x, -y, orb2, orb1])
    return np.array(seznam)

def positions(a, b, b2):
    return np.array([[0.,0.],
                     [-a/4, b/2 - b2],
                    [-a/4,b2],
                    [a/4,-b2],
                    [a/4, -b/2 + b2],
                    [a/4,b/4],
                    [-a/4, -b/4]])
   # return np.array([[0.,0.],
   #                   [1.752,13.86],
   #                   [1.752,9.807],
   #                   [3.5035,5.982],
   #                   [3.5035,1.927],
   #                   [3.5035,11.92],
   #                   [1.752,3.94]])
#xcor_[0]=1.752/a_;xcor_[1]=1.752/a_;xcor_[2]=3.5035/a_;xcor_[3]=3.5035/a_;xcor_[4]=3.5035/a_;xcor_[5]=1.752/a_
#ycor_[0]=13.86/c_;ycor_[1]=9.807/c_;ycor_[2]=5.982/c_;ycor_[3]=1.927/c_;ycor_[4]=11.92/c_;ycor_[5]=3.94/c_

''' matrix for number density operator '''
def j_tok(Kymesh, Kxmesh, a, b, b2, file):
    pos = positions(a, b, b2)
    Ny, Nx = Kymesh.shape
    jx = np.zeros((6, 6, Ny, Nx), dtype='complex')
    jy = np.copy(jx)

    for line in file:
        x, y, orb1, orb2, t = line
        x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
        if orb1 == orb2 and (x,y) == (0,0): pass # this is onsite energy, does not contribute to j
        else:
            osnova = 1j * t * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b))
            lega = pos[orb2] - pos[orb1] - np.array([x*a, y*b])
            ad_x = osnova * lega[0]
            ad_y = osnova * lega[1]

            jx[orb1 - 1, orb2 - 1] += ad_x
            if orb1 != orb2:
                jx[orb2 - 1, orb1 - 1] += ad_x.conjugate() 
            jy[orb1 - 1, orb2 - 1] += ad_y
            if orb1 != orb2:
                jy[orb2 - 1, orb1 - 1] += ad_y.conjugate()
    jmatrix = np.zeros((2,6,6,Ny,Nx), dtype='complex')
    jmatrix[0] = jx
    jmatrix[1] = jy
    return jmatrix

#def Delta(Kxmesh, Kymesh, Nk, rho, i, j, x): 
#    if type(x) == np.ndarray:
#        return np.array([np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x1[0] - 1j * Kymesh * x1[1])) for x1 in x]) / Nk
#    else: return np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x[0] - 1j * Kymesh * x[1])) / Nk

def group_velocity(Kymesh, Kxmesh, energije):
    dKy, dKx = np.diff(Kymesh[:,0])[0], np.diff(Kxmesh[0])[0]
    velocity_x = np.zeros(energije.shape)
    velocity_y = np.zeros(energije.shape)
    for i in range(6):
        gr = np.gradient(energije[i])
        velocity_x[i, :, :] = gr[1]/dKx
        velocity_y[i, :, :] = gr[0]/dKy
    return velocity_x, velocity_y

@njit(parallel=False, cache=True)
def phi_boltzmann(Kymesh, Kxmesh, velocity_x, velocity_y, energije, omegas, mu, faktor=1.):
    Ny, Nx = Kymesh.shape
    dKy, dKx = Kymesh[:,0][1] - Kymesh[:,0][0], Kxmesh[0][1] - Kxmesh[0][0]

    domega = omegas[1] - omegas[0]
    transportna_x = np.zeros(omegas.shape[0])
    transportna_y = np.zeros(omegas.shape[0])
    v_max = np.array([np.max(np.abs(velocity_x)), np.max(np.abs(velocity_y))])
    sigma = np.array([np.sqrt(v_max[0] * domega * dKx) * faktor, np.sqrt(v_max[1] * domega * dKy) * faktor])

    for m in [0, Ny//2]:
        for n in range(Nx):
            for orb in range(6):
                transportna_x += 1/np.sqrt(2*np.pi*sigma[0]**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma[0]**2)) * velocity_x[orb,m,n]**2
                transportna_y += 1/np.sqrt(2*np.pi*sigma[1]**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma[1]**2)) * velocity_y[orb,m,n]**2
    for n in [0, Nx//2]:
        for m in range(Ny):
            for orb in range(6):
                transportna_x += 1/np.sqrt(2*np.pi*sigma[0]**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma[0]**2)) * velocity_x[orb,m,n]**2
                transportna_y += 1/np.sqrt(2*np.pi*sigma[1]**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma[1]**2)) * velocity_y[orb,m,n]**2
    
    for m in range(Ny):
        for n in prange(1,Nx//2):
            if m not in [0, Ny//2]:
                for orb in range(6):
                    transportna_x += 2. * 1/np.sqrt(2*np.pi*sigma[0]**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma[0]**2)) * velocity_x[orb,m,n]**2
                    transportna_y += 2. * 1/np.sqrt(2*np.pi*sigma[1]**2) * np.exp(-(omegas - (energije[orb,m,n] - mu))**2/(2*sigma[1]**2)) * velocity_y[orb,m,n]**2
    
    return transportna_x * 2 / (Ny * Nx), transportna_y * 2 / (Ny * Nx) # factor 2 for spin

@njit(parallel=False, cache=True)
def K0_boltzmann(Kymesh, Kxmesh, velocity_x, velocity_y, energije, mu, T):
    Ny, Nx = Kymesh.shape

    K0_x, K0_y = 0., 0.
    for m in [0, Ny//2]:
        for n in prange(Nx):
            for orb in range(6):
                K0_x += velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K0_y += velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
    for n in [0, Nx//2]:
        for m in prange(Ny):
            for orb in range(6):
                K0_x += velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K0_y += velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))

    for m in prange(Ny):
        for n in prange(1,Nx//2):
            if m not in [0, Ny//2]:
                for orb in range(6):
                    K0_x += 2 * velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                    K0_y += 2 * velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
    return K0_x * 2 / (Ny * Nx), K0_y * 2 / (Ny * Nx)

@njit
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2

@njit
def spektralna_k(omega, mu, energije_k, Gamma):
    N_orb = len(energije_k)
    A = np.zeros((N_orb, N_orb))
    for orb in range(N_orb):
        A[orb, orb] = -1/np.pi * Gamma / ( (omega - (energije_k[orb] - mu))**2 + Gamma**2 )
    return A

@njit
def helper_phi(omega, j1, j2, energije, mu, Gamma):
    transportna = 0
    A = spektralna_k(omega, mu, energije, Gamma)
    transportna += np.trace(j1 @ A @ j2 @ A)
    return transportna

# caution: if parallel=True, you accumulate weird numerical errors because, apparently, parallel summation is nondeterministic
@njit(parallel=False, cache=True)
def phi_kubo(Kymesh, Kxmesh, tok_x, tok_y, energije, omegas, mu, Gamma):
    Ny, Nx = Kxmesh.shape
    transportna_x = np.zeros(len(omegas), dtype=np.complex128)
    transportna_y = np.zeros(len(omegas), dtype=np.complex128)
    
    for m in [0, Ny//2]:
        for n in prange(Nx):
            for i, omega in enumerate(omegas):
                A = spektralna_k(omega, mu, energije[:, m, n], Gamma)
                tok_kx = tok_x[:,:,m,n]
                tok_ky = tok_y[:,:,m,n]
                for a in range(6):
                    for b in range(6):
                        transportna_x[i] += tok_kx[a, b] * A[b, b] * tok_kx[b, a] * A[a, a]
                        transportna_y[i] += tok_ky[a, b] * A[b, b] * tok_ky[b, a] * A[a, a]

    for n in [0, Nx//2]:
        for m in prange(Ny):
            for i, omega in enumerate(omegas):
                A = spektralna_k(omega, mu, energije[:, m, n], Gamma)
                tok_kx = tok_x[:,:,m,n]
                tok_ky = tok_y[:,:,m,n]
                for a in range(6):
                    for b in range(6):
                        transportna_x[i] += tok_kx[a, b] * A[b, b] * tok_kx[b, a] * A[a, a]
                        transportna_y[i] += tok_ky[a, b] * A[b, b] * tok_ky[b, a] * A[a, a]

    for m in prange(Ny):
        for n in prange(1,Nx//2):
            if m not in [0, Ny//2]:
                for i, omega in enumerate(omegas):
                    A = spektralna_k(omega, mu, energije[:, m, n], Gamma)
                    tok_kx = tok_x[:,:,m,n]
                    tok_ky = tok_y[:,:,m,n]
                    for a in range(6):
                        for b in range(6):
                            transportna_x[i] += 2 *  (tok_kx[a, b] * A[b, b] * tok_kx[b, a] * A[a, a]).real
                            transportna_y[i] += 2 * (tok_ky[a, b] * A[b, b] * tok_ky[b, a] * A[a, a]).real
    return transportna_x.real * 2 / (Ny * Nx), transportna_y.real * 2 / (Ny * Nx)

@njit(parallel=False, cache=True)
def phiQ(Kymesh, Kxmesh, tok_x, tok_y, mat_x, mat_y, energije, omegas, mu, Gamma):
    Ny, Nx = Kxmesh.shape
    transportna_x = np.zeros(len(omegas), dtype=np.complex128)
    transportna_y = np.zeros(len(omegas), dtype=np.complex128)

    for m in (0, Ny//2):
        for n in prange(Nx):
            for i, omega in enumerate(omegas):
                A = spektralna_k(omega, mu, energije[:, m, n], Gamma)
                tok_kx = tok_x[:,:,m,n]
                tok_ky = tok_y[:,:,m,n]
                mat_kx = mat_x[:,:,m,n]
                mat_ky = mat_y[:,:,m,n]
                for a in range(6):
                    for b in range(6):
                        transportna_x[i] += mat_kx[a, b] * A[b, b] * tok_kx[b, a] * A[a, a]
                        transportna_y[i] += mat_ky[a, b] * A[b, b] * tok_ky[b, a] * A[a, a]

    for n in (0, Nx//2):
        for m in prange(Ny):
            for i, omega in enumerate(omegas):
                A = spektralna_k(omega, mu, energije[:, m, n], Gamma)
                tok_kx = tok_x[:,:,m,n]
                tok_ky = tok_y[:,:,m,n]
                mat_kx = mat_x[:,:,m,n]
                mat_ky = mat_y[:,:,m,n]
                for a in range(6):
                    for b in range(6):
                        transportna_x[i] += mat_kx[a, b] * A[b, b] * tok_kx[b, a] * A[a, a]
                        transportna_y[i] += mat_ky[a, b] * A[b, b] * tok_ky[b, a] * A[a, a]

    for m in prange(Ny):
        for n in prange(1,Nx//2):
            if m != 0 and m != Ny // 2:
                for i, omega in enumerate(omegas):
                    A = spektralna_k(omega, mu, energije[:, m, n], Gamma)
                    tok_kx = tok_x[:,:,m,n]
                    tok_ky = tok_y[:,:,m,n]
                    mat_kx = mat_x[:,:,m,n]
                    mat_ky = mat_y[:,:,m,n]
                    for a in range(6):
                        for b in range(6):
                            transportna_x[i] += 2 *  (mat_kx[a, b] * A[b, b] * tok_kx[b, a] * A[a, a]).real
                            transportna_y[i] += 2 * (mat_ky[a, b] * A[b, b] * tok_ky[b, a] * A[a, a]).real
    return transportna_x.real * 2 / (Ny * Nx), transportna_y.real * 2 / (Ny * Nx)

def mf_matrix1(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = len(Kymesh[:,0]), len(Kymesh[0])
    Nk = Ny * Nx
    matrix = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: V_ = U
                        else: V_ = 2 * V # factor 2 for spin multiplicity..
                        if orb2 == orb2_:
                            suma_n = np.sum(rho[orb1_ - 1, orb1_ - 1, :, :])
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1 -1, orb2 - 1] += -1j * t * V_ * lega[nu] * np.exp(-1j*Kxmesh*x*a -1j*Kymesh*y*b) / Nk * suma_n

                        if orb1 == orb2_:
                            suma_n = np.sum(rho[orb1_ - 1, orb1_ - 1, :, :])
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1 - 1, orb2 - 1] += 1j * t * V_ * lega[nu] * np.exp(-1j*Kxmesh*x*a -1j*Kymesh*y*b) / Nk * suma_n
    return matrix * 0.5

def mf_matrix2(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = len(Kymesh[:,0]), len(Kymesh[0])
    Nk = Ny * Nx
    matrix = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: V_ = U
                        else: V_ = 2 * V
                        if orb2 == orb2_:
                            suma_n = np.sum(rho[orb1 - 1, orb2 - 1, :, :] * np.exp(-1j*Kxmesh*x*a - 1j*Kymesh*y*b))
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1_ -1, orb1_ - 1] += -1j * t * V_ * lega[nu] / Nk * suma_n

                        if orb1 == orb2_:
                            suma_n = np.sum(rho[orb1 - 1, orb2 - 1, :, :] * np.exp(-1j*Kxmesh*x*a - 1j*Kymesh*y*b))
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                matrix[nu, orb1_ - 1, orb1_ - 1] += 1j * t * V_ * lega[nu]  / Nk * suma_n
    return matrix * 0.5 

# I verified that convolution via FFT yields the same as by direct sum; this is M^6 in my notes
def mf_matrix3(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = len(Kymesh[:,0]), len(Kymesh[0])
    Nk = Ny * Nx
    matrix = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    f_k = t * np.exp(-1j*Kxmesh*x*a - 1j*Kymesh*y*b) / Nk

                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: pass 
                        if orb2 == orb2_:
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = -1j * V * lega[nu] * np.exp(1j*Kxmesh*x*a + 1j*Kymesh*y*b) * np.exp(-1j*Kxmesh*x_*a - 1j*Kymesh*y_*b)
                                h = rho[orb1 - 1, orb1_ - 1, :, :]

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))

                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)

                                matrix[nu, orb1_ -1, orb2 -1] +=  f_k * gh

                        if orb1 == orb2_:
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = 1j * V * lega[nu] * np.exp(-1j*Kxmesh*x_*a - 1j*Kymesh*y_*b)
                                h = rho[orb1 - 1, orb1_ - 1, :, :]

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))

                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)

                                matrix[nu, orb1_ -1, orb2 - 1] += f_k * gh
    return -matrix * 0.5 

# I verified that convolution via FFT yields the same as by direct sum; this is M^6 in my notes
def mf_matrix4(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = len(Kymesh[:,0]), len(Kymesh[0])
    Nk = Ny * Nx
    matrix = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)

    for alpha in range(1,7):
        for beta in range(1,7):
            for line in kinetic:
                x, y, orb1, orb2, t = line
                x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
                if orb1 == alpha and orb2 == beta:
                    if orb1 == orb2 and x == 0: pass
                    for line_ in interaction:
                        x_, y_, orb1_, orb2_ = line_
                        x_, y_, orb1_, orb2_ = float(x_), float(y_), int(orb1_), int(orb2_)
                        if orb1_ == orb2_: pass 
                        if orb2 == orb2_:
                            lega = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = -1j * V * lega[nu] * np.exp(-1j*Kxmesh*x*a - 1j*Kymesh*y*b) * np.exp(1j*Kxmesh*x_*a + 1j*Kymesh*y_*b)
                                h = t * np.exp(-1j*Kxmesh*x*a - 1j*Kymesh*y*b) * rho[orb1_ - 1, orb2 - 1, :, :] / Nk

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))

                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)

                                matrix[nu, orb1 -1, orb1_ -1] +=  gh
                        if orb1 == orb2_:
                            lega = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                            for nu in range(2):
                                g = 1j * V * lega[nu] * np.exp(1j*Kxmesh*x_*a +  1j*Kymesh*y_*b)
                                h = t * np.exp(-1j*Kxmesh*x*a - 1j*Kymesh*y*b) * rho[orb1_ - 1, orb2 - 1, :, :] / Nk

                                g_fft = np.fft.fft2(np.fft.ifftshift(g))
                                h_fft = np.fft.fft2(np.fft.ifftshift(h))
                                gh = np.fft.ifft2(g_fft * h_fft)
                                gh = np.fft.fftshift(gh)
                                matrix[nu, orb1 -1, orb1_ - 1] += gh
    return -matrix * 0.5

def mf_matrices(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction):
    Ny, Nx = len(Kymesh[:,0]), len(Kymesh[0])
    matrix = np.zeros((4, 2, 6, 6, Ny, Nx), dtype=np.complex128)
    matrix[0] = mf_matrix1(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction)
    matrix[1] = mf_matrix2(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction)
    matrix[2] = mf_matrix3(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction)
    matrix[3] = mf_matrix4(Kymesh, Kxmesh, rho, a, b, U, V, pos, kinetic, interaction)
    return matrix

@njit(cache=True)
def create_jI(pos, a, b, kineticf, interactionf, Nk, kx, ky, qx, qy, U, V):
    j_I = np.zeros((2,6,6,6), dtype=np.complex128)
    for line in kineticf:
        x, y, orb1, orb2, t = line
        x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
        if orb1 == orb2 and x==0: pass
        else:
            for line_ in interactionf:
                x_, y_, orb1_, orb2_ = line_
                x_, y_, orb1_, orb2_  = float(x_), float(y_), int(orb1_), int(orb2_)
                if orb1_ == orb2_ and (x_, y_) == (0,0): V_ = U
                else: V_ = V
                if orb2 == orb2_:
                    osnova = -1j * t * V_ * np.exp(-1j*(kx*x*a + ky*y*b - qx*x_*a - qy*y_*b)) * np.exp(-1j*(qx*x*a + qy*y*b))
                    lega = (pos[orb2 - 1] - pos[orb1_ - 1] - np.array([x_*a, y_*b]))
                    for nu in range(2):
                        j_I[nu, orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * lega[nu] / Nk
                
                if orb1 == orb2_:
                    osnova = 1j * t * V_ * np.exp(-1j*(kx*x*a + ky*y*b - qx*x_*a - qy*y_*b))
                    lega = (pos[orb1 - 1] - pos[orb1_ - 1] - np.array([x_*a, y_*b]))
                    for nu in range(2):
                        j_I[nu, orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * lega[nu] / Nk
    return j_I


@njit(parallel=True, cache=True)
def phi_ii3(Ky, Kx, rho, vecs, energije, pos, kinetic, interaction, tok, mu, omega, Gamma, a, b, b2, U, V):
    Ny, Nx = len(Ky), len(Kx)
    Nk = Ny * Nx
    suma_x = 0.0 + 0.0j
    suma_y = 0.0 + 0.0j
    matrix = np.zeros((2, 6, 6, Ny, Nx), dtype=np.complex128)
    for m in prange(Ny):
        for n in range(Nx//2 + 1):
            ky, kx = Ky[m], Kx[n]
            j_I = create_jI(pos, a, b, kinetic, interaction, Nk, kx, ky, 0, 0, U, V)
            vec = vecs[:,:,m,n]
            tok_kx = np.ascontiguousarray(vec) @ np.ascontiguousarray(tok[0,:,:,m,n]) @ np.ascontiguousarray(vec).conj().T
            tok_ky = np.ascontiguousarray(vec) @ np.ascontiguousarray(tok[1,:,:,m,n]) @ np.ascontiguousarray(vec).conj().T

            tmp = np.zeros((2,6,6), dtype=np.complex128)
            for u in prange(Ny):
                for v in range(Nx):
                    for ii in range(6):
                        for jj in range(6):
                            g = np.array([rho[0,0,u,v], rho[1,1,u,v], rho[2,2,u,v], rho[3,3,u,v], rho[4,4,u,v], rho[5,5,u,v]])
                            tmp[0,ii,jj] += np.dot(j_I[0,ii,:,jj], g)
                            tmp[1,ii,jj] += np.dot(j_I[1,ii,:,jj], g)
            matrix[:,:,:,]
            M_3x = vec @ tmp[0] @ vec.conj().T
            M_3y = vec @ tmp[1] @ vec.conj().T
            A = spektralna_k(omega, mu, energije[:,m,n], Gamma)
            mat_x = M_3x @ A @ tok_kx @ A
            mat_y = M_3y @ A @ tok_ky @ A

            if m in [0, Ny//2] or m in [0, Nx//2]: multiply = 1
            else: multiply = 2
            for orb in range(6):
                suma_x += mat_x[orb,orb] * multiply
                suma_y += mat_y[orb,orb] * multiply
    return suma_x / Nk, suma_y / Nk



import numpy as np
import time

#os.chdir('/Users/ana/Desktop/ta2nise5')


#os.chdir('/Users/ana/Desktop/ta2nise5/parameters')

U = 2.5 # eV
V = 0.785 # eV
a = 3.51 # A
b = 15.79 # A
b2 = 1.927 # A
mu0 = 2.84


''' create TNS class '''
class TNS:
    def __init__(self, a, b, b2, Ny, Nx, U, V, mu0, parameters1, parameters2, eps0):
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.Nx, self.Ny = Nx, Ny
        self.Nk = Ny * Nx
        Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2) / Ny
        Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2) / Nx
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh
        self.hop = H_hopping(self.kymesh, self.kxmesh, a, b)
        self.perturb = H_perturb(self.kymesh, self.kxmesh, a, b)
        self.rho = Rho0(self.Ny, self.Nx)
        self.mu = mu0

        self.fock = H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        self.hartree = H_hartree(self.rho, self.Nk, U, V)

        self.tok = j_tok(self.kymesh, self.kxmesh, a, b, b2, kinetic())
        
        self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, eps0, a, U, V, 1e-10, maxiter=1000, N_epsilon=5)
        self.rho0 = self.rho
        self.fock0 = self.fock
        self.hartree0 = self.hartree
        self.phi = Phi(self.kxmesh, self.Nk, self.rho, a)[0].real

        self.Phi_x = []
        self.Phi_y = []
        
        self.phis = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.times_rho = []

        self.Ts = []
        self.betas = []

        self.RhoB_x = []
        self.RhoB_y = []
        self.RhoK_x = []
        self.RhoK_y = []

    def next_T(self, T, i) -> None:
        start = time.time()
        if i == 1: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters1
        elif i ==2: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters2
        rho, energije, fs, vecs, fock, hartree, err, n, mu = NewMu(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                a, U, V, T, self.mu,
                                                                dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials)
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.fock = fock
        self.hartree = hartree
        self.mu = mu
        self.err = err
        self.n = n
        self.times_rho.append(time.time() - start)
        #print(1/T, err, n, helpers.Phi(self.kxmesh, self.Nk, rho, a)[0].real)

    def run2(self, betas, stops, Gamma, eps, Nomega):
        #hbar = 6.582119569 * 1e-16 # eV s
        #e0 = 1.602176634 * 1e-19 # A s
        #kb = 8.6173303 * 1e-5 # eV/K
        a = 3.51 # A
        b = 15.79 # A
        c = 13.42 # A

        predfaktor = 2 * np.pi / (a * b * c * 1e-10) / (25.8 * 1e3)  # 1/(Ohm * m^3)

        for i, beta in enumerate(betas):
            T = 1/beta
            if i not in stops:
                if (i+1) in stops:
                    rho_save = self.rho
                    energije_save = self.energije
                    fs_save = self.fs
                    vecs_save = self.vecs
                    fock_save = self.fock
                    hartree_save = self.hartree
                    mu_save = self.mu
                    err_save = self.err
                    n_save = self.n
                self.next_T(T, 2)
            else:
                print(f'started evaluating at beta={beta}')
                self.next_T(T, 1)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(Phi(self.kxmesh, self.Nk, self.rho, a)[0].real)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)

                omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
                omegas = np.linspace(-omega_max, omega_max, Nomega)

                velocity_x, velocity_y = group_velocity(self.kymesh, self.kxmesh, self.energije)
                sigmaB_x, sigmaB_y = K0_boltzmann(self.kymesh, self.kxmesh, velocity_x, velocity_y, self.energije, self.mu, T)
                sigmaB_x = sigmaB_x  / (2*Gamma) * predfaktor # 1/(Ohm m)
                sigmaB_y = sigmaB_y / (2*Gamma) * predfaktor
                #print(np.allclose(sigmaB_y, 0))
                #phiBx, phiBy = phi_boltzmann(self.kymesh, self.kxmesh, self.energije, omegas, self.mu, faktor=1.) / (2*Gamma)
                tok_x = np.einsum('jixy, jlxy, lkxy->ikxy', self.vecs.conj(), self.tok[0], self.vecs)
                tok_y = np.einsum('jixy, jlxy, lkxy->ikxy', self.vecs.conj(), self.tok[1], self.vecs)
                phiKx, phiKy = phi_kubo(self.kymesh, self.kxmesh, tok_x, tok_y, self.energije, omegas, self.mu, Gamma)
                phiKx, phiKy = phiKx * np.pi, phiKy * np.pi

                sigmaK_x = np.sum(phiKx.real * (-fd_1(omegas, T))) * (omegas[1] - omegas[0]) * predfaktor # 1/(Ohm m)
                sigmaK_y = np.sum(phiKy.real * (-fd_1(omegas, T))) * (omegas[1] - omegas[0]) * predfaktor # 1/(Ohm m)

                rhoB_x = 1/sigmaB_x
                if sigmaB_y != 0.:
                    rhoB_y = 1/sigmaB_y # Ohm m
                else:
                    rhoB_y = 0.
                rhoK_x = 1/sigmaK_x
                if sigmaK_y != 0:
                    rhoK_y = 1/sigmaK_y # Ohm m
                else:
                    rhoK_y = 0

                self.RhoB_x.append(rhoB_x * 1e2) # Ohm cm
                self.RhoB_y.append(rhoB_y * 1e2)
                self.RhoK_x.append(rhoK_x * 1e2)
                self.RhoK_y.append(rhoK_y * 1e2)
                print(rhoB_x * 1e2, rhoK_x * 1e2)

                if i > 0:
                    self.rho = rho_save
                    self.energije = energije_save
                    self.fs = fs_save
                    self.vecs = vecs_save
                    self.fock = fock_save
                    self.hartree = hartree_save
                    self.mu = mu_save
                    self.err = err_save
                    self.n = n_save

    def reset(self, mu0):
        self.rho = self.rho0
        self.hartree = self.hartree0
        self.fock = self.fock0
        self.mu = mu0

    def reset_infty(self):
        self.rho = Rhoinfty(self.Ny, self.Nx)
        self.hartree = H_hartree(self.rho, self.Nk, U, V)
        self.fock = H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        
        _, energije, fs, vecs, _, _, _, _ = Rho_next(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, a, U, V, 0, self.mu, 50, 0.5, 1e-10, eps0=0.0, N_epsilon=5)
        self.energije = energije
        self.fs = fs
        self.vecs = vecs


dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 2000, 2000, 1e-7, 0.5, 0.001, 1.5, 1e-4, 30
parameters1 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]

dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 500, 500, 1e-7, 0.5, 0.001, 1.5, 1e-4, 30
parameters2 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]


Ny, Nx = 100, 100
mu = 2.84
eps0 = 0.1
print('evaluating')
s = TNS(a, b, b2, Ny, Nx, U, V, mu, parameters1, parameters2, eps0)
print('found Gs')

beta0 = 350
scale = 1.005
betas = beta0/scale**np.arange(1,200,3)
stops = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
Ts = 1/betas

eps = 1e-5
Nomega = 701

Gamma = 0.0075

s.run2(betas, stops, Gamma, eps, Nomega)


np.save(f'RhoB_x_{Nx}.npy', s.RhoB_x)
np.save(f'RhoK_x_{Nx}.npy', s.RhoK_x)
np.save(f'RhoB_y_{Nx}.npy', s.RhoB_y)
np.save(f'RhoK_y_{Nx}.npy', s.RhoK_y)
np.save(f'Ts_{Nx}.npy', s.Ts)

import matplotlib
import matplotlib.pyplot as plt
kb = 8.6 * 1e-5
plt.plot(np.array(s.Ts)/kb, s.RhoB_x, '.-')
plt.plot(np.array(s.Ts)/kb, s.RhoK_x, '.-')

plt.yscale('log')
plt.ylim(1e-4, 1e6)
plt.show()