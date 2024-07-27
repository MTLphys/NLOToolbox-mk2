import numpy as np 
import torch 
import scipy.constants as c 



def Efield(t,P,wl,pw,t0,phi=0):
    """Generate a multiply pulsed Efield 

    Args:
        t (N,double): time [s] N is the number time steps 
        P (n,double): power[W] n is the number of pulse 
        wl (n,double): wavlength [m]
        pw (n,double): pulse width[s]
        t0 (n,double): time of pulse arrival[s]
        phi (n,double): relative phase of each pulse[rad] 

    Returns:
        (N,double): returns efield with given pulse dimensions
    """
    if (phi ==0 ):
        phi = []
        for i in range(len(P)):
            phi.append(0)
            
    E= np.zeros(len(t),complex) #initialize efield 
    r = 50.0e-6 #set focal spot size
    RR= 79.8e6  #set rep rate   
    for i,Pi in enumerate(P):
        Psi = Pi/(np.pi*r**2)/RR #energetic flux per pulse joules/m^2  
        E0 = np.sqrt(Psi*c.c*c.mu_0/pw)#calculated envelope field in V/m  
        om = np.pi*2*c.c/wl[i] #calculate the frequency
        sig = pw/(2*np.sqrt(2*np.log(2)))#envelop scaling factor
        print("exciting frequency :",1e-15*om/np.pi/2)  
        E+= E0/(sig**2*np.pi*1)**(1/4)*np.exp(-1/2*((t-t0[i])/sig)**2)*np.exp(1.0j*(om*t+phi[i]))# generate first train element of gaussian pulses
    return E 

def makeH(H,E,nstep,n,muij,En):
    """Generate the hamiltonian of an N state system for n phase steps

    Args:
        H (N*n,N*n, complex): Hamiltonain to be modified
        E (n,complex): Electric field for each phase step
        nstep (int): number of phase steps 
        n (int): number of states
        muij (N*n,N*n,double): dipole moment of each state
        En (N,): energy levels 
    """
    for i in range(nstep*n):#iterate transversly
        for j in range(nstep*n):#iterate longitudinally 
            if((np.floor(i/n)==np.floor(j/n))):#only occupy within 
                H[i,j]=(i==j)*En[i%n]-muij[int(i%n),int(j%n)]*E[int(np.floor(i/n))]#set values
def makeHf(H,E,nstep,n,muij,En,a,b):
    """Generate the hamiltonian of an N state system for n phase steps

    Args:
        H (N*n,N*n, complex): Hamiltonain to be modified
        E (n,complex): Electric field for each phase step
        nstep (int): number of phase steps 
        n (int): number of states
        muij (N*n,N*n,double): dipole moment of each state
        En (N,): energy levels 
    """
    H[:,:]=torch.tensor(list(map( lambda i: 
                                list(map(lambda j:(np.floor(i/n)==np.floor(j/n))*
                                    ((i==j)*En[i%n]-
                                        muij[int(i%n),int(j%n)]*E[int(np.floor(i/n))]),a)),b)),dtype=torch.complex128) 

def getEfield(Rho,nstep,n):
    """calculate the polarization of the system

    Args:
        Rho (_type_): _description_
        nstep (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    ef= torch.zeros(nstep)
    for i in range(n-1):
        for j in range(n-i-1):
            ef=ef+torch.diagonal(Rho,i+1)[j::n]
    return torch.imag(ef)

def getOccupation(Rho,nstep,n):
    """Calculation the excited state incoherent occupation for the sysem

    Args:
        Rho (_type_): _description_
        nstep (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
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
P = [.002,.002] #W power
wl = [830.0e-9,830.0e-9] #m wavelength of exciting pulse
pw =  80e-15 #fs pulse width of excitation (fwhm)  
t0 = [-0.01e-12,0] # time delay for each pulse 


#create time space and efield
t = np.linspace(ts,te,int(np.ceil((te-ts)/dt)+1)) #times in s
E = Efield(t,P,wl,pw,t0)# efield in V/m

#create excited state parameters 
ax = 13e-9#nm exciton bohr radius
muii = 0.0 #occupation coupling 
Nx =1.0e16*1e6 #available excitons excitons/m^3 
mu13 = Nx*ax*c.elementary_charge #exciton dipole moment 
print('couping strength',mu13*np.max(E))
mu12= Nx*ax*c.elementary_charge #exciton dipole moment 
mu23 = Nx*ax*c.elementary_charge #exciton dipole moment 
muij= np.array([[muii , mu12, mu13],
                [mu12 , muii, mu23],
                [mu13 , mu23, muii]],dtype=np.complex128)#,

Tii=300.0e-12#lifetime decay time 
T12=450.0e-15#dephasing decay time 
T13=450.0e-15#dephasing decay time
T23=10000.0e-15 #interstate dephasing decay time
Tij = np.array([[Tii , T12, T13],
                [T12 , Tii, T23],
                [T13 , T23, Tii]],dtype=np.complex128)#decay array

n = 3 #number of states available
nstep = 1#number of batch steps for 
Egap = np.pi*2.0*361.0e12  # Energy of exciton
Esplit = -np.pi*2.0*1.0e12 # Seperations of split state
En =[0,Egap,Egap-Esplit]#energy of each level 

fig,ax = plt.subplots(4)#draw graph
ax[0].plot(t,np.abs(E))#Show plot of efield envelope
ax[1].plot(t,np.real(E))#show plot of efield 

ax[1].plot(t,np.real(E))#check resolution limit 
ax[1].set_xlim(-pw*.01,pw*.01)#zoom in 

print("system frequency",1e-16*Egap/np.pi/2,'Thz')
E = np.asarray([Efield(t,P,wl,pw,t0)])#pack up layers of efield 

H = torch.zeros((n*nstep,n*nstep),dtype=torch.complex128)# set up the hamiltonian 
HR = torch.tensor([[1/(Tij[int(i%n),int(j%n)])*(np.floor(i/n)==np.floor(j/n)) for i in range(nstep*n)] for j in range(nstep*n)],dtype=torch.complex128)#set up decay hamiltonian
rho0 = torch.tensor([[1.0*(i==j)*(i%n==0) for i in range(int(nstep*n))] for j in range(int(nstep*n))],dtype=torch.complex128)#set up 
P = torch.zeros(len(t))#net polarization of the system
P1 = torch.zeros(len(t))#polarization of first state 
P2 = torch.zeros(len(t))#polarization of second state
P3 = torch.zeros(len(t))#polarization of second and first mixed state
Oc = torch.zeros(len(t))#occupation level of the net system

#N = torch.zeros(nstep*n,dtype=torch.complex128)
#M = torch.zeros(nstep*n,dtype=torch.complex128)
II = torch.eye(nstep*n,dtype=torch.complex128)#.to_sparse()#create an identity matrix for the system 
gEf = getEfield#initialize function for polarization 
getOcc = getOccupation#initialize function for occupation 

#set up the density matrix 
rho = torch.tensor([[1.0*(i==j)*(i%n==0) for i in range(int(nstep*n))] for j in range(int(nstep*n))],dtype=torch.complex128)

print(rho0)

#set up iterable for the mapping of items into the matrix form
it1 = np.arange(n*nstep,dtype=int)
it2 = np.arange(n*nstep,dtype=int)



import time as ti 
a = np.zeros(13)
sa= ti.time()
for i in tqdm(range(len(t))):
    s= ti.time()
    makeHf(H,np.real(E[:,i]),nstep,n,muij,En,it1,it2)
    if(i==0):
        print(H)
    e= ti.time()
    a[0]+=e-s
    
    s= ti.time()
    M = (II-0.5j*dt*H)
    e= ti.time()
    a[1]+=e-s
    
    s= ti.time()
    N = (II.subtract(0.5j*dt*H).add((0.5j*dt*H).matmul(0.5j*dt*H)))
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
    R = HR.mul(rho.subtract(rho0)) 
    e= ti.time()
    a[5]+=e-s
    
    s= ti.time()    
    rho = torch.matmul(rho,Ud)
    e= ti.time()
    a[6]+=e-s
    
    s= ti.time()    
    rho = torch.matmul(U,rho).subtract(dt*R)
    e= ti.time()
    a[7]+=e-s
    
    s= ti.time()
    P[i]  = torch.real(rho[0,1]+rho[0,2]+rho[1,2])
    e= ti.time()
    a[8]+=e-s
    
    s= ti.time()
    P1[i]  = torch.real(rho[0,1])
    e= ti.time()
    a[9]+=e-s
    
    s= ti.time()
    P2[i]  = torch.real(rho[0,2])
    e= ti.time()
    a[10]+=e-s
    
    s= ti.time()
    P3[i]  = torch.real(rho[1,2])
    e= ti.time()
    a[11]+=e-s
    
    s= ti.time()
    Oc[i] = rho[1,1]+rho[2,2]
    e= ti.time()
    a[12]+=e-s
ea = ti.time()    
print(a)
print(np.sum(a))
print(ea-sa)
ax[2].plot(t,P.real,label='Total Polarization')
ax[2].plot(t,P1.real,label='X1 Polarization')
ax[2].plot(t,P2.real,label='X2 Polarization')
ax[2].plot(t,P3.real,label='X21 Polarization')
ax[2].legend()

#ax[2].plot(t,E[0]*1e-42)
ax[3].plot(t,Oc)
#ax[3].set_yscale('log')
plt.show()
