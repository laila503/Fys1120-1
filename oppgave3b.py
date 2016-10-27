import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn


#De 2 funksjonene under er for oppgave 2
"""
def findPeriode(r,t):

    startTheta = np.arctan2(r[0,1],r[0,1])
    iChecked = 0
    times = [0.0]

    for i in range(1,r[:,0].size):
        theta = np.arctan2(r[i,1],r[i,0])

        if theta > 0 and np.arctan2(r[i-1,1],r[i-1,0]) < 0:
            times.append(t[i])
            iChecked = i


    print "Periode found: ", times[1] - times[0]


def exactSolution(time,r,omega,v):
    y = r-r*np.cos(omega*time)
    x = r*(np.sin(omega*time))
    z = time*v[2]

    return x,y,z
"""


E0 = 25e3/(90e-6)
B = np.array([0,0,2])

d = 90e-6
rd = 50e-3

c = 3e9

mp = 1.67e-27 #mass of a proton
q = 1.6e-19 #charge of a proton

omega = np.linalg.norm(B)*q/mp

def a(r,v,i):
    F_b = q*np.cross(v,B)

    if np.linalg.norm(r) < rd:
        if abs(r[0]) < d/2.:
            F = F_e[i]*np.array([1,0,0]) + F_b
            return F/mp

        else:
            F = F_b
            return F/mp
    else:
        return 0




timeToSim = 300e-9
dt = 1000e-15

n = int(timeToSim/dt)


v = np.zeros((n,3))
r = np.zeros((n,3))
time = np.linspace(0,timeToSim,n)
F_e = E0*np.cos(omega*time)*q

v[0,:] = np.array([0,0,0])
r[0,:] = np.array([0,0,0])


#main Euler-Cromer loop

for i in range(1,int(n)):
    acc = a(r[i-1,:],v[i-1,:],i-1)
    v[i,:] = v[i-1,:] + acc*dt
    r[i,:] = r[i-1,:] + v[i,:]*dt

    print (float(i)/int(n))*100, "%            \r",
print ""


print "The proton has a speed of: ", np.linalg.norm(v[-1,:])

plt.title("Position Components of Proton")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.plot(time,r[:,0])
plt.plot(time,r[:,1])
plt.plot(time,r[:,2])
plt.legend(["x(t)","y(t)","z(t)"])
plt.show()

plt.title("Velocity Components of Proton")
plt.xlabel("$v_x$")
plt.ylabel("$v_y$")
plt.plot(time,v[:,0])
plt.plot(time,v[:,1])
plt.plot(time,v[:,2])
plt.legend(["v_x(t)","v_y(t)","v_z(t)"])
plt.show()

plt.title("Position of Proton")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.plot(r[:,0],r[:,1])
plt.axis("equal")
plt.show()

plt.title("Velocity of Proton")
plt.xlabel("$v_x$")
plt.ylabel("$v_y$")
plt.plot(v[:,0],v[:,1])
plt.show()

plt.title("Speed of Proton over Time")
plt.xlabel("$time$")
plt.ylabel(r'$|\vec{v}|$')
plt.plot(time,np.linalg.norm(v,axis = 1)/3e9)
plt.show()
