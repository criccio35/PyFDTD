import numpy as np
import math
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.fromnumeric import size
from numpy import cumsum
from numpy.lib.npyio import savetxt

# ==== Preamble ===============================================================
c0   = scipy.constants.speed_of_light
mu0  = scipy.constants.mu_0
eps0 = scipy.constants.epsilon_0
imp0 = math.sqrt(mu0 / eps0)

def gaussianFunction(t, t0, spread):
    return math.exp(- math.pow(t-t0, 2) / (2.0 * math.pow(spread, 2)) )

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------

L  = 20.0
l1=(1./3.)*L    # grid change 1
l2=(2./3.)*L    # grid change 2

# -- CASO 1: Homogeneous variable grid -- #
# dx0 = 0.05
# dx_f = 4*dx0
# dx1 = dx0*np.ones(int(l1/dx0))
# dx2 = dx_f*np.ones(int((l2-l1)/dx_f))
# dx3 = dx0*np.ones(int((L-l2)/dx0))
# n2=dx2.size

# -- CASO 2: variable grid with a function -- #
dx0 = 0.05
dx1 = dx0*np.ones(int(l1/dx0))      #homogeneous grid in [0,l1]
dx3 = dx0*np.ones(int((L-l2)/dx0))  #homogeneous grid in [l2,L]
  
n2=100
dmin = dx0     # dmin > 0
dmax = 0.1      # dmax < l2-l1

# x = np.linspace(0,1,num=n2, endpoint=True)
# dx2 = eval('-4*(dmax-dmin)*x**2 + 4*(dmax-dmin)*x + dmax')
# dx2 = ((l2-l1)/sum(dx2))*dx2
x = np.linspace(l1,l2,num=n2,endpoint=True)
A = dmax / (dmin + ((l2-l1)**2)/2 )
dx2 =-A*(x-l1)*(x-l2)+dmin  
 
plt.figure()
plt.plot(x,dx2)
plt.xlim(l1-dx0,l2+dx0)
ax=plt.axes()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks([dmin, dmax])
ax.yaxis.set_ticklabels(['$dx_{min}$', '$dx_{max}$'])
ax.xaxis.set_ticks([l1, l2])
ax.xaxis.set_ticklabels(['l1', 'l2'])
ax.tick_params(labelsize=14)
plt.plot(np.linspace(l1,l1+(l2-l1)/2.,10),np.ones(10)*dmax,'r--')
plt.plot(np.linspace(l1,l2,10),np.ones(10)*dmin,'r--')
plt.title('$y=-A(x-l_1)(x-l_2)+dx_{min} $')

# ------------------------ ------------------------#
n1=dx1.size     #number of nodes in [0,l1]
n3=dx3.size     #number of nodes in [l2,L]
N=n1+n2+n3      #Total number of nodes in [0,L]

dx = np.zeros(N)
dx[0:n1]=dx1
dx[n1:n1+n2]=dx2
dx[n1+n2:N]=dx3

#L = n1*dx0 + n2*dx_f + n3*dx0
L = n1*dx0 + sum(dx2) + n3*dx0

finalTime = L/c0*4
cfl       = .99

# Electric field grid
gridE = np.zeros(N)
gridE[0:n1]     = np.linspace(0,      l1,        num=n1, endpoint=True)
gridE[n1:n1+n2] = l1*np.ones(n2) + cumsum(dx2)
gridE[n1+n2:N]  = np.linspace(l2,      L,        num=n3, endpoint=True)


# Magnetic field grid
gridH2 = np.zeros(n2+1)
gridH2[0:n2] = (l1-(dx0/2))*np.ones(n2) + cumsum(dx2)
gridH2[n2] = gridH2[n2-1] + dx2[-1] # se anade uno mas para completar la malla

gridH = np.zeros(N-1)
gridH[0:n1-1]    = np.linspace(   dx0/2.0, l1-dx0/2.0, num=n1-1,   endpoint=True)
gridH[n1-1:n1+n2]= gridH2
gridH[n1+n2:N-1] = np.linspace(l2+dx0/2.0,  L-dx0/2.0, num=n3-1,   endpoint=True)


plt.figure()
 
plt.plot(np.linspace(0,L,dx.size),dx)
plt.xlabel('L')
plt.ylabel('$dx$')
plt.title('$dx$ size')


#gridE = np.linspace(0,      L,        num=L/dx+1, endpoint=True)
#gridH = np.linspace(dx0/2.0, L-dx0/2.0, num=L/dx0,   endpoint=True)


# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
#spread = 1/math.sqrt(2.0)
# initialE = gaussianFunction(gridE, L/2, spread)

# Plane wave illumination
totalFieldBox = (L*1./8.,L*7./8.)
delay  = 8e-9
spread = 2e-9
 
# ---- Output requests --------------------------------------------------------
samplingPeriod = 0.0
 
# ==== Processing =============================================================
# ---- Solver initialization --------------------------------------------------
dt = cfl * dx / c0
numberOfTimeSteps = N

#if samplingPeriod == 0.0:
#    samplingPeriod = dt 
#nSamples  = int( math.floor(finalTime/samplingPeriod) )
nSamples  = numberOfTimeSteps
probeE    = np.zeros((gridE.size, nSamples))
probeH    = np.zeros((gridH.size, nSamples))
probeTime = np.zeros(nSamples) 

eOld = np.zeros(gridE.size)
eNew = np.zeros(gridE.size)
hOld = np.zeros(gridH.size)
hNew = np.zeros(gridH.size)
if 'initialE' in locals():
    eOld = initialE

totalFieldIndices = np.searchsorted(gridE, totalFieldBox)
#shift = (gridE[totalFieldIndices[1]] - gridE[totalFieldIndices[0]]) / c0 

shift=0

# Determines recursion coefficients
cE = cfl / eps0 / c0
cH = cfl / mu0  / c0

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time()

w = 2*math.pi * 100e6
k = c0 / w
beta = w*np.sqrt(mu0*eps0)

t = 0.0
E_Exacta = np.zeros((1,nSamples))
timeSamples = np.zeros((1,nSamples))

for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    for i in range(1, gridE.size-1):
        eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
        
    # E field boundary conditions
    # Sources   
    eNew[totalFieldIndices[0]] = eNew[totalFieldIndices[0]] + gaussianFunction(t, delay, spread)

    # PEC
#    eNew[ 0] = 0.0;
#    eNew[-1] = 0.0;
    
    # PMC
#    eNew[ 0] = eOld[ 0] - 2.0 * cE * hOld[ 0]
#    eNew[-1] = eOld[-1] + 2.0 * cE * hOld[-1]
    
    # Mur ABC
    eNew[ 0] = eOld[ 1] + (cfl-1)/(cfl+1) * (eNew[ 1] - eOld[ 0])         
    eNew[-1] = eOld[-2] + (cfl-1)/(cfl+1) * (eNew[-2] - eOld[-1]) 

    # Periodic
#    eNew[ 0] = eOld[ 0] + cE * (hOld[ -1] - hOld[ 0])         
#    eNew[ -1] = eOld[ -1] + cE * (hOld[ -2] - hOld[ -1])

    # --- Updates H field ---
    for i in range(gridH.size):
        hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
    
    # H field boundary conditions
    # Sources
    hNew[totalFieldIndices[0]-1] = hNew[totalFieldIndices[0]-1] + gaussianFunction(t, delay, spread) / imp0
#     hNew[totalFieldIndices[1]-1] = hNew[totalFieldIndices[1]-1] - gaussianFunction(t, delay+shift, spread) / imp0
          
    # --- Updates output requests ---
    probeE[:,n] = eNew[:]
    probeH[:,n] = hNew[:]
    probeTime[n] = t
    
    # --- Updates fields and time 
    eOld[:] = eNew[:]
    hOld[:] = hNew[:]
    t += dt[n] #dt(indice)

# --- guardar en ficheros
#np.savetxt("PMC_E%d.txt" %(TAM), ErrorE)
#np.savetxt("PMC_H%d.txt" %(TAM), ErrorH)

tictoc = time.time() - tic;
print('--- Processing finished ---')
print("CPU Time: %f [s]" % tictoc)

# ==== Post-processing ========================================================

# --- Creates animation ---
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1, 2, 1)
ax1 = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax1.grid(color='gray', linestyle='--', linewidth=.2)
ax1.set_xlabel('X coordinate [m]')
ax1.set_ylabel('Field')
line1,    = ax1.plot([], [], 'o', markersize=1)
timeText1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

ax2 = fig.add_subplot(2, 2, 2)
ax2 = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax2.grid(color='gray', linestyle='--', linewidth=.2)
# ax2.set_xlabel('X coordinate [m]')
# ax2.set_ylabel('Magnetic field [T]')
line2,    = ax2.plot([], [], 'o', markersize=1)
timeText2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)

def init():
    line1.set_data([], [])
    timeText1.set_text('')
    line2.set_data([], [])
    timeText2.set_text('')
    return line1, timeText1, line2, timeText2

def animate(i):
    line1.set_data(gridE, probeE[:,i])
    timeText1.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    line2.set_data(gridH, probeH[:,i]*100)
    timeText2.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    return line1, timeText1, line2, timeText2

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nSamples, interval=50, blit=True)

plt.grid(True)
plt.plot(l1*np.ones(2*36),range(-36,36),'--',color='b')
plt.plot(l2*np.ones(2*36),range(-36,36),'--',color='b')
#plt.show()

plt.figure()
posicion=15
probeEIndex = np.searchsorted(gridE, posicion)
Esim=probeE[probeEIndex,:]
plt.plot(probeTime[:]*1e9, probeE[probeEIndex,:])   #Simulada
# 
Eexacta = np.zeros((1,nSamples))
Hexacta = np.zeros((1,nSamples))
for n in range(nSamples):
    Eexacta[0,n] = gaussianFunction(probeTime[n], delay + posicion/c0 - L/8.0/c0, spread)
#    Hexacta[0,n] = gaussianFunction(probeTime[n], delay + posicion/c0 - L/8.0/c0, spread)
     
plt.plot(probeTime[:]*1e9, Eexacta[0,:]) #exacta
plt.title('Exacta vs Simulada')
 
plt.figure()

Error2=(Esim-Eexacta[0,:])**2
Error=(Esim-Eexacta[0,:])
savetxt("Errorm_%d.txt"%n2, (probeTime[:]*1e9,Error2))

plt.plot(probeTime[:]*1e9, Error)
plt.plot(probeTime[:]*1e9, Error2)
plt.title('Error')


plt.show()

print('=== Program finished ===')