import numpy as np
#import os, scipy, mpmath
from numba import njit, prange
import warnings
from numba.core.errors import NumbaPerformanceWarning

# Suppress NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#os.chdir('/Users/ana/Desktop/ta2nise5/parameters')

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

def Delta(Kxmesh, Kymesh, Nk, rho, i, j, x): 
    if type(x) == np.ndarray:
        return np.array([np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x1[0] - 1j * Kymesh * x1[1])) for x1 in x]) / Nk
    else: return np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x[0] - 1j * Kymesh * x[1])) / Nk

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
    K1_x, K1_y = 0., 0.
    for m in [0, Ny//2]:
        for n in prange(Nx):
            for orb in range(6):
                K0_x += velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K0_y += velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K1_x += (energije[orb,m,n] - mu) * velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K1_y += (energije[orb,m,n] - mu) * velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
    for n in [0, Nx//2]:
        for m in prange(Ny):
            for orb in range(6):
                K0_x += velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K0_y += velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K1_x += (energije[orb,m,n] - mu) * velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                K1_y += (energije[orb,m,n] - mu) * velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
    for m in prange(Ny):
        for n in prange(1,Nx//2):
            if m not in [0, Ny//2]:
                for orb in range(6):
                    K0_x += 2 * velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                    K0_y += 2 * velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                    K1_x += 2 * (energije[orb,m,n] - mu) * velocity_x[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
                    K1_y += 2 * (energije[orb,m,n] - mu) * velocity_y[orb,m,n]**2 * (-fd_1(energije[orb,m,n] - mu, T))
    return K0_x * 2 / (Ny * Nx), K0_y * 2 / (Ny * Nx), K1_x * 2 / (Ny * Nx), K1_y * 2 / (Ny * Nx)

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
