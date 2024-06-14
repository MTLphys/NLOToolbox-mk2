import numpy as np 
import torch 
import scipy.constants as c 


def Efield(t,P,wl,pw,t0):
    E= np.zeros(len(t),complex) #initialize efield 
    r = 50.0e-6 #set focal spot size
    RR= 79.8e6  #set rep rate   
    for i,Pi in enumerate(P):
        Psi = Pi/(np.pi*r**2)/RR #energetic flux per pulse joules/m^2  
        E0 = np.sqrt(Psi*c.c*c.mu_0)#calculated envelope field in V/m  
        om = np.pi*2*c.c/wl[i]
        sig = pw/(2*np.sqrt(2*np.log(2)))
        print("exciting frequency :",om)
        E+= E0/(sig**2*np.pi*1)**(1/4)*np.exp(-1/2*((t-t0[i])/sig)**2)*np.exp(1.0j*om*t)
    return E 

def makeH(H,E,nstep,n,muij,En):
    for i in range(nstep*n):
        for j in range(nstep*n):
            if((np.floor(i/n)==np.floor(j/n))):
                H[i,j]=(i==j)*En[i%n]-muij[int(i%n),int(j%n)]*E[int(np.floor(i/n))] 
def makeHf(H,E,nstep,n,muij,En):
    for i in range(nstep*n):
        for j in range(nstep*n):
            if((np.floor(i/n)==np.floor(j/n))):
                H[i,j]=(i==j)*En[i%n]-muij[int(i%n),int(j%n)]*E[int(np.floor(i/n))] 

def getEfield(Rho,nstep,n):
    ef= torch.zeros(nstep)
    for i in range(n-1):
        for j in range(n-i-1):
            ef=ef+torch.diagonal(Rho,i+1)[j::n]
    return torch.imag(ef)

def getOccupation(Rho,nstep,n):
    oc= torch.zeros(nstep)
    for i in range(n-1):
        oc =oc+torch.diagonal(Rho,0)[i+1::n]
    return torch.real(oc)

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as c 
from tqdm import tqdm
#parameters of excting field
ts =  -3e-12  #s time window start of measurement
te =  3e-12  #s time window end of measurment 
dt = .1e-15 #time resolution  
P = [.002] #mW power
wl = [830.0e-9] #m wavelength of exciting pulse
pw =  80e-15 #fs pulse width of excitation (fwhm)  
t0 = [0] # time delay for each pulse 


#create time space and efield
t = np.linspace(ts,te,int(np.ceil((te-ts)/dt)+1)) #times in s
E = Efield(t,P,wl,pw,t0)# efield in V/m

#create excited state parameters 
ax = 13e-9#nm exciton bohr radius
muii = 0.0 #occupation coupling 
Nx =1.0e32 #available excitons 
mu13 = Nx*ax*c.elementary_charge #exciton dipole moment 
print('couping strength',mu13*np.max(E))
mu12= Nx*ax*c.elementary_charge #exciton dipole moment 
mu23 = Nx*ax*c.elementary_charge #exciton dipole moment 
muij= np.array([[muii , mu12, mu13],
                [mu12 , muii, mu23],
                [mu13 , mu23, muii]],dtype=np.complex128)#,

Tii=100.0e-12#lifetime decay time 
T12=600.0e-15#dephasing decay time 
T13=700.0e-15#dephasing decay time
T23=700.0e-15 #interstate dephasing decay time
Tij = np.array([[Tii , T12, T13],
                [T12 , Tii, T23],
                [T13 , T23, Tii]],dtype=np.complex128)#

n = 3 #number of states available
nstep = 1
Egap = np.pi*2.0*366.0e12
Ebound = np.pi*2.0*3.0e12
En =[0,Egap,Egap-Ebound]

fig,ax = plt.subplots(4)
ax[0].plot(t,np.abs(E))
ax[1].plot(t,np.real(E))
ax[1].set_xlim(-pw*.01,pw*.01)
print("system frequency",Egap)
E = np.asarray([Efield(t,P,wl,pw,t0)])

H = torch.zeros((n,n),dtype=torch.complex128)
HR = torch.tensor([[1/(Tij[int(i%n),int(j%n)])*(np.floor(i/n)==np.floor(j/n)) for i in range(nstep*n)] for j in range(nstep*n)],dtype=torch.complex128)
rho0 = torch.tensor([[1.0*(i==j)*(i%n==0) for i in range(int(nstep*n))] for j in range(int(nstep*n))],dtype=torch.complex128)
P = torch.zeros(len(t))
P1 = torch.zeros(len(t))
P2 = torch.zeros(len(t))
P3 = torch.zeros(len(t))
Oc = torch.zeros(len(t))
II = torch.eye(nstep*n,dtype=torch.complex128)
gEf = getEfield
getOcc = getOccupation
rho = torch.tensor([[1.0*(i==j)*(i%n==0) for i in range(int(nstep*n))] for j in range(int(nstep*n))],dtype=torch.complex128)
print(rho0)

import time as ti 
a = np.zeros(13)
sa= ti.time()
for i in tqdm(range(len(t))):
    s= ti.time()
    makeH(H,np.real(E[:,i]),nstep,n,muij,En)
    e= ti.time()
    a[0]+=e-s
    
    s= ti.time()
    M = (II-0.5j*dt*H)
    e= ti.time()
    a[1]+=e-s
    
    s= ti.time()
    N = (II+0.5j*dt*H).inverse()
    e= ti.time()
    a[2]+=e-s
    
    
    s= ti.time()
    U = torch.matmul(N,M)
    e= ti.time()
    a[3]+=e-s
    
    s= ti.time()
    Ud = torch.conj(U)
    e= ti.time()
    a[4]+=e-s
    
    s= ti.time()    
    R = HR*(rho-rho0) 
    e= ti.time()
    a[5]+=e-s
    
    s= ti.time()    
    rho = torch.matmul(rho,Ud)
    e= ti.time()
    a[6]+=e-s
    
    s= ti.time()    
    rho = torch.matmul(U,rho) - dt*R
    e= ti.time()
    a[7]+=e-s
    
    s= ti.time()
    P[i]  = np.real(rho[0,1]+rho[0,2]+rho[1,2])
    e= ti.time()
    a[8]+=e-s
    
    s= ti.time()
    P1[i]  = np.real(rho[0,1])
    e= ti.time()
    a[9]+=e-s
    
    s= ti.time()
    P2[i]  = np.real(rho[0,2])
    e= ti.time()
    a[10]+=e-s
    
    s= ti.time()
    P3[i]  = np.real(rho[1,2])
    e= ti.time()
    a[11]+=e-s
    
    s= ti.time()
    Oc[i] = rho[0,0]-rho[1,1]-rho[2,2]
    e= ti.time()
    a[12]+=e-s
ea = ti.time()    
print(a)
print(np.sum(a))
print(ea-sa)
ax[2].plot(t,P.real,label='Total Polarization')
ax[2].plot(t,P1.real,label='X1 Polarization')
ax[2].plot(t,P2.real,label='X2 Polarization')
ax[2].plot(t,P3.real,label='X2-X1 Polarization')
ax[2].legend()

#ax[2].plot(t,E[0]*1e-42)
ax[3].plot(t,1.0-Oc)
#ax[3].set_yscale('log')
plt.show()
