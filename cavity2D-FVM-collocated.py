
import numpy as np
import matplotlib.pyplot as plt

S =  1;  # Edge Length, Square
ns = 51; # Divisions

ds = S/(ns-1); # Grid size
wallU = 1;     # Lid velocity
nu = 0.01;     # Kinematic viscosity

alphaP = 0.5;  #  
alphaU = 0.7;  # Relaxation 
alphaV = 0.7;  # Factors 

# Face area vectors [R,T,L,B]
ax = ds*np.array([1,0,-1,0]);
ay = ds*np.array([0,1,0,-1]);

# Allocate memory
u = np.zeros((ns,ns));
uPrime = np.zeros((ns,ns));
v = np.zeros((ns,ns));
vPrime = np.zeros((ns,ns));
u0 = np.zeros((ns,ns));
v0 = np.zeros((ns,ns));
p = np.zeros((ns,ns));
pPrime = np.zeros((ns,ns));


error = 1;
tIter = 1;

#Functions 
def coeffsU(u,v):
    an = np.zeros((ns,ns,4)); # ab
    ap = np.zeros((ns,ns));   # ac
    # Inner face velocities for v
    an[0:ns-1,0:ns,1] = nu + 0.5*np.maximum(-(v[0:ns-1,0:ns]+v[1:ns,0:ns])*ay[1],0) # Top Neighbours
    an[1:ns,0:ns,3] = nu + 0.5*np.maximum(-(v[1:ns,0:ns]+v[0:ns-1,0:ns])*ay[3],0) # Bottom Neighbours
    # Inner face velocities for u
    an[0:ns,0:ns-1,0] = nu + 0.5*np.maximum(-(u[0:ns,0:ns-1]+u[0:ns,1:ns])*ax[0],0) # Right Neighbours
    an[0:ns,1:ns,2] = nu + 0.5*np.maximum(-(u[0:ns,1:ns]+u[0:ns,0:ns-1])*ax[2],0) # Left Neighbours
    # Boundary face velocities for v (No inertial terms)
    an[ns-1,0:ns,1] += nu;  # Top Neighbours  
    an[0,0:ns,3] += nu;     # Bottom Neighbours
    # Boundary face velocities for u (No inertial terms)
    an[0:ns,ns-1,0] += nu;  # Right Neighbours
    an[0:ns,0,2] += nu;     # Left Neighbours 
    # ap = sigma(an) + (u|v,phi)
    ap[0:ns,0:ns] = an[0:ns,0:ns,0]+an[0:ns,0:ns,1]+an[0:ns,0:ns,2]+an[0:ns,0:ns,3];
    # v velocity
    ap[0:ns-1,0:ns] += 0.5*(v[0:ns-1,0:ns]+v[1:ns,0:ns])*ay[1];
    ap[1:ns,0:ns] += 0.5*(v[1:ns,0:ns]+v[0:ns-1,0:ns])*ay[3]; 
    # u velocity
    ap[0:ns,0:ns-1] += 0.5*(u[0:ns,0:ns-1]+u[0:ns,1:ns])*ax[0]; 
    ap[0:ns,1:ns] += 0.5*(u[0:ns,1:ns]+u[0:ns,0:ns-1])*ax[2]; 
    return an,ap/alphaV;

def coeffsP(ap):
    ab = np.zeros((ns,ns,4)); # an
    ac = np.zeros((ns,ns));   # ap 
    # Inner faces
    ab[0:ns-1,0:ns,1] = ay[1]**2*0.5*(1/ap[0:ns-1,0:ns] + 1/ap[1:ns,0:ns]);  # Top Neighbours
    ab[1:ns,0:ns,3] = ay[3]**2*0.5*(1/ap[1:ns,0:ns] + 1/ap[0:ns-1,0:ns]);    # Bottom Neighbours
    ab[0:ns,0:ns-1,0] = ax[0]**2*0.5*(1/ap[0:ns,0:ns-1] + 1/ap[0:ns,1:ns]);  # Right Neighbours
    ab[0:ns,1:ns,2] = ax[2]**2*0.5*(1/ap[0:ns,1:ns] + 1/ap[0:ns,0:ns-1]);    # Left Neighbours
    # Boundary faces 
    ab[ns-1,0:ns,1] += ay[1]**2*(1/ap[ns-1,0:ns]); # Top face
    ab[0,0:ns,3] += ay[3]**2*(1/ap[0,0:ns]);       # Bottom face
    ab[0:ns,ns-1,0] += ax[0]**2*(1/ap[0:ns,ns-1]); # Right face
    ab[0:ns,0,2] += ax[2]**2*0.5*(1/ap[0:ns,0]);   # Left face
    # ac = sigma(ab)
    ac[0:ns,0:ns] = ab[0:ns,0:ns,0]+ab[0:ns,0:ns,1]+ab[0:ns,0:ns,2]+ab[0:ns,0:ns,3];
    return ab,ac
    
def pfA(p,a):
    pf = np.zeros((ns,ns));
    # Internal faces
    pf[0:ns-1,0:ns] += 0.5*(p[0:ns-1,0:ns]+p[1:ns,0:ns])*a[1]; # +=? 
    pf[1:ns,0:ns] += 0.5*(p[1:ns,0:ns]+p[0:ns-1,0:ns])*a[3]; 
    pf[0:ns,0:ns-1] += 0.5*(p[0:ns,0:ns-1]+p[0:ns,1:ns])*a[0]; 
    pf[0:ns,1:ns] += 0.5*(p[0:ns,1:ns]+p[0:ns,0:ns-1])*a[2];
    # Boundary faces
    pf[ns-1,0:ns] += p[ns-1,0:ns]*a[1]; # Top face
    pf[0,0:ns] += p[0,0:ns]*a[3];       # Bottom face
    pf[0:ns,ns-1] += p[0:ns,ns-1]*a[0]; # Right face
    pf[0:ns,0] += p[0:ns,0]*a[2];       # Left face 
    return pf;

def interiorProduct(an,u):
    un = np.zeros((ns,ns));
    un[0:ns-1,0:ns] += an[0:ns-1,0:ns,1]*u[1:ns,0:ns];  # Top face +=?
    un[1:ns,0:ns] += an[1:ns,0:ns,3]*u[0:ns-1,0:ns];   # Bottom face
    un[0:ns,0:ns-1] += an[0:ns,0:ns-1,0]*u[0:ns,1:ns]; # Right face
    un[0:ns,1:ns] += an[0:ns,1:ns,2]*u[0:ns,0:ns-1];   # Left face
    return un;

def boundaryProductU(an,u):
    un = np.zeros((ns,ns));
    un[ns-1,0:ns] += an[ns-1,0:ns,1]*(2*wallU - u[ns-1,0:ns]); # Top face 
    un[0,0:ns]  += -an[0,0:ns,3]*u[0,0:ns];                     # Bottom face +=?
    un[0:ns,0] += -an[0:ns,0,2]*u[0:ns,0];                     # Left face
    un[0:ns,ns-1] += -an[0:ns,ns-1,0]*u[0:ns,ns-1];            # Right face 
    return un;

def boundaryProductV(an,v):
    un = np.zeros((ns,ns));
    un[ns-1,0:ns] += -an[ns-1,0:ns,1]*v[ns-1,0:ns];  # Top face 
    un[0,0:ns] += -an[0,0:ns,3]*v[0,0:ns];           # Bottom face
    un[0:ns,0] += -an[0:ns,0,2]*v[0:ns,0];           # Left face
    un[0:ns,ns-1] += -an[0:ns,ns-1,0]*v[0:ns,ns-1];  # Right face 
    return un;

def intExtProductP(pPrime,ab):
    pf = np.zeros((ns,ns));
    # Internal faces
    pf[0:ns-1,0:ns] += ab[0:ns-1,0:ns,1]*pPrime[1:ns,0:ns];
    pf[1:ns,0:ns] += ab[1:ns,0:ns,3]*pPrime[0:ns-1,0:ns];
    pf[0:ns,0:ns-1] += ab[0:ns,0:ns-1,0]*pPrime[0:ns,1:ns];
    pf[0:ns,1:ns] += ab[0:ns,1:ns,2]*pPrime[0:ns,0:ns-1];
    # Boundary faces
    pf[ns-1,0:ns] += ab[ns-1,0:ns,1]*pPrime[ns-1,0:ns];
    pf[0,0:ns] += ab[0,0:ns,3]*pPrime[0,0:ns];
    pf[0:ns,ns-1] += ab[0:ns,ns-1,0]*pPrime[0:ns,ns-1];
    pf[0:ns,0] += ab[0:ns,0,2]*pPrime[0:ns,0];
    return pf;
       
def uEqn(u,v,p,an,pn):
    err = 1;
    uIter = 1;
    
    while(err>1e-4 and uIter<20):
        un = ((interiorProduct(an,u) + boundaryProductU(an,u)) - pfA(p,ax))/ap + (1-alphaU)*u0;
        vn = ((interiorProduct(an,v) + boundaryProductV(an,v)) - pfA(p,ay))/ap + (1-alphaV)*v0;
        err = (np.linalg.norm(un-u) + np.linalg.norm(vn-v))/(ns*ns);
        u = un.copy(); v = vn.copy();
        uIter += 1;
    return u,v

# Rhie-Chow momentum interpolation technique    
def rcInterp(p,u,v,ap):
    pf = np.zeros((ns,ns));
    # Inner cells with 2 neighbours
    pf[1:ns-2,0:ns] += 0.25*((p[2:ns-1,0:ns]-p[0:ns-3,0:ns])/ap[1:ns-2,0:ns] + (p[3:ns,0:ns]-p[1:ns-2,0:ns])/ap[2:ns-1,0:ns])*ay[1]**2; # Top Face +=? 
    pf[2:ns-1,0:ns] += 0.25*((p[2:ns-1,0:ns]-p[0:ns-3,0:ns])/ap[1:ns-2,0:ns] + (p[3:ns,0:ns]-p[1:ns-2,0:ns])/ap[2:ns-1,0:ns])*ay[3]**2; # Bottom Face
    pf[0:ns,1:ns-2] += 0.25*((p[0:ns,2:ns-1]-p[0:ns,0:ns-3])/ap[0:ns,1:ns-2] + (p[0:ns,3:ns]-p[0:ns,1:ns-2])/ap[0:ns,2:ns-1])*ax[0]**2; # Right Face
    pf[0:ns,2:ns-1] += 0.25*((p[0:ns,2:ns-1]-p[0:ns,0:ns-3])/ap[0:ns,1:ns-2] + (p[0:ns,3:ns]-p[0:ns,1:ns-2])/ap[0:ns,2:ns-1])*ax[2]**2; # Left Face           
    # Boundary cells with 1 neighbour
    pf[1,0:ns] += 0.5*(0.5*(p[2,0:ns]-p[0,0:ns])/ap[1,0:ns] + (p[1,0:ns]-p[0,0:ns])/ap[0,0:ns])*ay[3]**2; # Bottom Face [1]
    pf[0,0:ns] += 0.5*(0.5*(p[2,0:ns]-p[0,0:ns])/ap[1,0:ns] + (p[1,0:ns]-p[0,0:ns])/ap[0,0:ns])*ay[1]**2; # Top Face [0]
    pf[0,0:ns] += 0.5*((p[1,0:ns]-p[0,0:ns])/ap[0,0:ns])*ay[3]**2; # Bottom Face [0]
    pf[ns-2,0:ns] += 0.5*(0.5*(p[ns-1,0:ns]-p[ns-3,0:ns])/ap[ns-2,0:ns] + (p[ns-1,0:ns]-p[ns-2,0:ns])/ap[ns-1,0:ns])*ay[1]**2; # Top Face [ns-2]
    pf[ns-1,0:ns] += 0.5*(0.5*(p[ns-1,0:ns]-p[ns-3,0:ns])/ap[ns-2,0:ns] + (p[ns-1,0:ns]-p[ns-2,0:ns])/ap[ns-1,0:ns])*ay[3]**2; # Bottom Face [ns-1]
    pf[ns-1,0:ns] += 0.5*((p[ns-1,0:ns]-p[ns-2,0:ns])/ap[ns-1,0:ns])*ay[1]**2; # Top Face [ns-1]
    pf[0:ns,1] += 0.5*(0.5*(p[0:ns,2]-p[0:ns,0])/ap[0:ns,1] + (p[0:ns,1]-p[0:ns,0])/ap[0:ns,0])*ax[2]**2; # Left Face [1]
    pf[0:ns,0] += 0.5*(0.5*(p[0:ns,2]-p[0:ns,0])/ap[0:ns,1] + (p[0:ns,1]-p[0:ns,0])/ap[0:ns,0])*ax[0]**2; # Right Face [0]
    pf[0:ns,0] += 0.5*((p[0:ns,1]-p[0:ns,0])/ap[0:ns,0])*ax[2]**2; # Left Face [0]
    pf[0:ns,ns-2] += 0.5*(0.5*(p[0:ns,ns-1]-p[0:ns,ns-3])/ap[0:ns,ns-2] + (p[0:ns,ns-1]-p[0:ns,ns-2])/ap[0:ns,ns-1])*ax[0]**2; # Right Face [ns-2]
    pf[0:ns,ns-1] += 0.5*(0.5*(p[0:ns,ns-1]-p[0:ns,ns-3])/ap[0:ns,ns-2] + (p[0:ns,ns-1]-p[0:ns,ns-2])/ap[0:ns,ns-1])*ax[2]**2; # Left Face [ns-1]
    pf[0:ns,ns-1] += 0.5*((p[0:ns,ns-1]-p[0:ns,ns-2])/ap[0:ns,ns-1])*ax[0]**2; # Right Face [ns-1]
    # pf - grad(p)
    pf[0:ns-1,0:ns] -= (p[1:ns,0:ns]-p[0:ns-1,0:ns])*ay[1]**2*0.5*(1/ap[1:ns,0:ns] + 1/ap[0:ns-1,0:ns]); 
    pf[1:ns,0:ns] -= (p[1:ns,0:ns]-p[0:ns-1,0:ns])*ay[3]**2*0.5*(1/ap[1:ns,0:ns] + 1/ap[0:ns-1,0:ns]);
    pf[0:ns,0:ns-1] -= (p[0:ns,1:ns]-p[0:ns,0:ns-1])*ax[0]**2*0.5*(1/ap[0:ns,1:ns] + 1/ap[0:ns,0:ns-1]);
    pf[0:ns,1:ns] -= (p[0:ns,1:ns]-p[0:ns,0:ns-1])*ax[2]**2*0.5*(1/ap[0:ns,1:ns] + 1/ap[0:ns,0:ns-1]);
    # momentum interpolation
    un = np.zeros((ns,ns));
    un[0:ns-1,0:ns] += 0.5*(v[0:ns-1,0:ns]+v[1:ns,0:ns])*ay[1] + (1-alphaV)*0.5*(v0[0:ns-1,0:ns]+v0[1:ns,0:ns])*ay[1];
    un[1:ns,0:ns] += 0.5*(v[1:ns,0:ns]+v[0:ns-1,0:ns])*ay[3] + (1-alphaV)*0.5*(v0[1:ns,0:ns]+v0[0:ns-1,0:ns])*ay[3];
    un[0:ns,0:ns-1] += 0.5*(u[0:ns,0:ns-1]+u[0:ns,1:ns])*ax[0] + (1-alphaU)*0.5*(u0[0:ns,0:ns-1]+u0[0:ns,1:ns])*ax[0];
    un[0:ns,1:ns] += 0.5*(u[0:ns,1:ns]+u[0:ns,0:ns-1])*ax[2] + (1-alphaU)*0.5*(u0[0:ns,1:ns]+u0[0:ns,0:ns-1])*ax[2];
    return un+pf;

def nonRcInterp(u,v):    
    un = np.zeros((ns,ns));
    un[0:ns-1,0:ns] += 0.5*(v[0:ns-1,0:ns]+v[1:ns,0:ns])*ay[1]; 
    un[1:ns,0:ns] += 0.5*(v[1:ns,0:ns]+v[0:ns-1,0:ns])*ay[3];
    un[0:ns,0:ns-1] += 0.5*(u[0:ns,0:ns-1]+u[0:ns,1:ns])*ax[0];
    un[0:ns,1:ns] += 0.5*(u[0:ns,1:ns]+u[0:ns,0:ns-1])*ax[2];
    return un;
    
def pEqn(pPrime,u,v,p,ap):
    err = 1;
    pIter = 1;
    # Coefficient estimation function
    ab,ac = coeffsP(ap);
    # Rhie-chow interpolation
    mStar = rcInterp(p,u,v,ap);
    # Non Rhie-chow formulation
    # mStar = nonRcInterp(u,v);
    pPrime[:,:] = 0;
    while(err>1e-4 and pIter<300):
        pn = (intExtProductP(pPrime,ab)-mStar)/ac;
        err = np.linalg.norm(pn-pPrime)/(ns*ns);
        pPrime = pn.copy();
        pIter += 1;
    return pPrime;
      
# SIMPLE Algorithm    
while(error>1e-7 and tIter<1000):
    u0 = u.copy();
    v0 = v.copy();
    # Coefficient estimation function
    an,ap = coeffsU(u,v);
    # Momentum equation
    u,v = uEqn(u,v,p,an,ap)
    # Pressure correction
    pPrime = pEqn(pPrime,u,v,p,ap)
    # Update velocity
    uPrime = u - pfA(pPrime,ax)/ap;
    vPrime = v - pfA(pPrime,ay)/ap;
    # Relax pressure
    pPrime = p + pPrime*alphaP;
    error  = (np.linalg.norm(pPrime-p)+np.linalg.norm(uPrime-u)
             +np.linalg.norm(vPrime-v))/(ns*ns);
    print(error);
    p = pPrime.copy(); 
    u = uPrime.copy(); 
    v = vPrime.copy();
    tIter += 1;

plt.figure(1)
cfu = plt.contourf(np.linspace(0,S,ns),np.linspace(0,S,ns),u,cmap='jet')
cb = plt.colorbar(cfu)
plt.title('U-Velocity Distributions')
plt.figure(2)
cfu = plt.contourf(np.linspace(0,S,ns),np.linspace(0,S,ns),v,cmap='jet')
cb = plt.colorbar(cfu)
plt.title('V-Velocity Distributions')
plt.figure(3)
cfu = plt.contourf(np.linspace(0,S,ns),np.linspace(0,S,ns),p,cmap='jet')
cb = plt.colorbar(cfu)
plt.title('Pressure Distributions')


















