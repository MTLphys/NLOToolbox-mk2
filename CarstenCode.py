# This program considers two coupled Rabi oscillators. 
# Let 1 (rho) be the density matrix of the plasmon 
# and 2 (sigma) be the density matrix of the exciton.
#------------------------------------------------------------------------
import numpy as np
import math
import matplotlib.pyplot as plt
#
E11 = 0.             # first energy level of the plasmon
E21 = 0.5            # second energy level of the plasmon
#
E12 = 0.             # first energy level of the exciton
E22 = 0.55           # second energy level of the exciton
#
MU1 = 1.             # external field dipole coupling constant of the plasmon 
MU2 = 1.             # external field dipole coupling constant of the exciton 
#
XI1 = 0.0          # induced field dipole coupling constant of the plasmon     
XI2 = 0.0         # induced field dipole coupling constant of the exciton
#
T11 = 200.           # plasmon relaxation time
T21 = 50.            # plasmon decoherence time
#
T12 = 2000.          # exciton relaxation time
T22 = 500.           # exciton decoherence time
#--------------------------------------------------------------------------
dt = 0.05            # time step    
steps=20000          # we propagate for a total time T = (steps*dt)
#--------------------------------------------------------------------------
#    Now define the laser parameters:
#
E0 = 0.01           # field amplitude
OMEGA = 0.6         # frequency
NCYCLE = 5          # number of cycles in the pulse. N>0
#--------------------------------------------------------------------------
#   Output options:
#   which = 1: plot the upper-level population of the plasmon and the exciton
#               (rho_22 and sigma_22)
#   which = 2: plot the real parts of the polarizations
#               (rho_12 and sigma_12)
#
#   analytic = 0 or 1: plot the analytic solution for rho_22 and sigma_22
#                      (only in combination with which = 1)
#
which = 2           
analytic = 0
#-------------------------user input ends here-----------------------------
#
tend = NCYCLE*2.*math.pi/OMEGA
T2R = T21*T22/(T21+T22)
DO12 = (E21-E11) - (E22-E12)
print("resonance frequencies:")
OMEGAPLUS = (E21-E11 + E22-E12)/2. + 0.5*math.sqrt(DO12**2 + 4.*MU1*MU2*XI1*XI2)
OMEGAMINUS = (E21-E11 + E22-E12)/2. - 0.5*math.sqrt(DO12**2 + 4.*MU1*MU2*XI1*XI2)
print(OMEGAPLUS,OMEGAMINUS)
#
TIME = np.zeros(steps)
DELTA1 = np.zeros(steps)
DELTA2 = np.zeros(steps)
DELTA1A = np.zeros(steps)
DELTA2A = np.zeros(steps)
#
R12 = np.zeros(steps)
S12 = np.zeros(steps)
#
ONE = 1. + 0.j
IONE = 0. + 1.j
#
# Initialize the density matrix in the ground state
#
RHO = np.zeros((2,2),dtype=np.complex_)
RHO0 = np.zeros((2,2),dtype=np.complex_)
R = np.zeros((2,2),dtype=np.complex_)
M = np.zeros((2,2),dtype=np.complex_)
#
SIGMA = np.zeros((2,2),dtype=np.complex_)
SIGMA0 = np.zeros((2,2),dtype=np.complex_)
#
RHO0[0,0] = ONE
SIGMA0[0,0] = ONE
#
U = np.zeros((2,2),dtype=np.complex_)
UD = np.zeros((2,2),dtype=np.complex_)
for i in range(steps):
    t = i*dt
    TIME[i] = t
    
    ELASER = 0.
    if t<tend:
       ENV = (math.sin(0.5*OMEGA*(t-dt/2.)/NCYCLE))**2
       ELASER = E0*math.sin(OMEGA*(t-dt/2.))*ENV
       
       D10 = (RHO0[0,0]).real - (RHO0[1,1]).real
       D20 = (SIGMA0[0,0]).real - (SIGMA0[1,1]).real
       U0 = (RHO0[0,1]).real
       V0 = (RHO0[0,1]).imag
       R0 = (SIGMA0[0,1]).real
       S0 = (SIGMA0[0,1]).imag
    
    E1 = MU1*(ELASER + XI1*(SIGMA0[0,1]+SIGMA0[1,0]))
#
    NN = (ONE + 0.5*IONE*dt*E11)*(ONE + 0.5*IONE*dt*E21) + 0.25*dt**2*E1**2  
#    
    U[0,0] = ((ONE - 0.5*IONE*dt*E11)*(ONE + 0.5*IONE*dt*E21) - 0.25*dt**2*E1**2)/NN
    U[0,1] = IONE*dt*E1/NN
    U[1,0] = IONE*dt*E1/NN
    U[1,1] = ((ONE + 0.5*IONE*dt*E11)*(ONE - 0.5*IONE*dt*E21) - 0.25*dt**2*E1**2)/NN

    UD = np.conjugate(U)
    
    R[0,0] = (RHO0[0,0]-ONE)/T11
    R[0,1] = RHO0[0,1]/T21
    R[1,0] = RHO0[1,0]/T21
    R[1,1] = RHO0[1,1]/T11
    
    M = np.matmul(RHO0,UD)
    
    RHO = np.matmul(U,M) - dt*R
    
    RHO0 = RHO
    
    DELTA1[i] = (RHO[0,0] - RHO[1,1]).real
#
    R12[i] = (RHO0[0,1]).real
#---------------------------------------------------------------------------------
    E2 = MU2*(ELASER + XI2*(RHO0[0,1]+RHO0[1,0]))
#
    NN = (ONE + 0.5*IONE*dt*E12)*(ONE + 0.5*IONE*dt*E22) + 0.25*dt**2*E2**2  
#    
    U[0,0] = ((ONE - 0.5*IONE*dt*E12)*(ONE + 0.5*IONE*dt*E22) - 0.25*dt**2*E2**2)/NN
    U[0,1] = IONE*dt*E2/NN
    U[1,0] = IONE*dt*E2/NN
    U[1,1] = ((ONE + 0.5*IONE*dt*E12)*(ONE - 0.5*IONE*dt*E22) - 0.25*dt**2*E2**2)/NN

    UD = np.conjugate(U)
    
    R[0,0] = (SIGMA0[0,0]-ONE)/T12
    R[0,1] = SIGMA0[0,1]/T22
    R[1,0] = SIGMA0[1,0]/T22
    R[1,1] = SIGMA0[1,1]/T12
    
    M = np.matmul(SIGMA0,UD)
    
    SIGMA = np.matmul(U,M) - dt*R
    
    SIGMA0 = SIGMA
    
    DELTA2[i] = (SIGMA[0,0] - SIGMA[1,1]).real  
#    
    S12[i] = (SIGMA0[0,1]).real
#------------------------------------------------------------------------------
#   End of numerical solution. Now let us prepare the analytic solution:   
#------------------------------------------------------------------------------ 
#    
UU = R0*U0 + S0*V0
VV = U0*S0 - V0*R0
#
RA = -4.*MU1*XI1*T11*T2R*(VV*(T2R-T11)+UU*DO12*T11*T2R)/((T2R-T11)**2+(DO12*T11*T2R)**2)
IA = -4.*MU1*XI1*T11*T2R*(UU*(T2R-T11)-VV*DO12*T11*T2R)/((T2R-T11)**2+(DO12*T11*T2R)**2)
#
RC = 4.*MU2*XI2*T12*T2R*(VV*(T2R-T12)+UU*DO12*T12*T2R)/((T2R-T12)**2+(DO12*T12*T2R)**2)
IC = 4.*MU2*XI2*T12*T2R*(UU*(T2R-T12)-VV*DO12*T12*T2R)/((T2R-T12)**2+(DO12*T12*T2R)**2)
#    
for i in range(steps):
    t = i*dt  
    
    DELTA1A[i] = 1.
    DELTA2A[i] = 1.
    
    if t>tend:
        
        t = t - tend
    
        FA = RA*math.cos(DO12*t) - IA*math.sin(DO12*t)
        FC = RC*math.cos(DO12*t) - IC*math.sin(DO12*t)
    
        DELTA1A[i] = 1. + (D10-1.-RA)*math.exp(-t/T11) + FA*math.exp(-t/T2R)
        DELTA2A[i] = 1. + (D20-1.-RC)*math.exp(-t/T12) + FC*math.exp(-t/T2R)   
    

 
if which==1:
    plt.plot(TIME,(1.-DELTA1)/2.,label="plasmon")
    plt.plot(TIME,(1.-DELTA2)/2.,label="exciton")
    plt.plot(TIME,(1.-DELTA1)/2.,label="Excitation")
    
    
    if analytic==1:
        plt.plot(TIME,(1.-DELTA1A)/2.,label="rho_analytic")
        plt.plot(TIME,(1.-DELTA2A)/2.,label="sigma_analytic")
    plt.ylabel("upper level population")
    plt.yscale('log')
    plt.ylim(0.00001,0.03)
#
if which==2:
    plt.plot(TIME,R12,label="plasmon",linewidth=1)
    plt.plot(TIME,S12,label="exciton",linewidth=1)
    plt.plot(TIME,(1.-DELTA1)/2.,label="Excitation")
    plt.ylabel("polarization")
    
plt.xlabel("time")
plt.legend()
plt.show()
#-------------------------------------------------------------------------------
plt.savefig('Rabi.png',bbox_inches='tight')