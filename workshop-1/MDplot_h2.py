import numpy as np
import matplotlib.pyplot as plt
import os
from ase import Atoms, units
from ase.constraints import FixCartesian
from ase.md.langevin import Langevin
from pyscf_calc import PySCFCalculator_B3LYP

initial_bond_length = 0.74  # in Angstrom
temperature_k = 300         # in K
time_step = 0.1             # in fs
total_steps = 1000

atoms = Atoms('H2', positions=[[-initial_bond_length/2, 0, 0], [initial_bond_length/2, 0, 0]])
constraint = FixCartesian(a=[0, 1], mask=[0, 1, 1])
atoms.set_constraint(constraint)
atoms.calc = PySCFCalculator_B3LYP('cc-pVDZ')

dyn = Langevin(atoms, 
           timestep=time_step * units.fs, 
           temperature_K=temperature_k, 
           friction=0.01 / units.fs)

if not os.path.exists('PES_DFTb3lyp.txt'):
    raise FileNotFoundError("Run PES calculation first !")
with open('PES_DFTb3lyp.txt','r') as fo:
    lines = fo.readlines()
PES_x = [float(line.split()[0]) for line in lines]
PES_y = [float(line.split()[1]) for line in lines]

plt.ion() 
plt.figure(figsize=(5,10))

istep = 0
def update_plot():
    global istep
    istep += 1
    # Get current physical state
    d = atoms.get_distance(0, 1)
    e = atoms.get_potential_energy()
    v = -atoms.get_velocities()[0][0]
    a = -(atoms.get_forces() / atoms.get_masses()[:, None])[0][0]
    print(f"Distance {d}, step {istep}/{total_steps} ({time_step} fs)")
    if istep % 5 == 0: 
        plt.subplot(2,1,1)
        plt.cla()
        # plot position
        plt.scatter([-d/2,d/2],[0,0],s=100,c='white',edgecolor='black')
        plt.plot([-d/2,d/2],[0,0],ls='--',c='gray')
        if d < 3:
            plt.xlim(-1.5, 1.5)
        else:
            plt.xlim(-5, 5)
        # plot velocity and acceleration
        plt.plot([-d/2-v, -d/2], [-0.002,-0.002], c='blue')
        plt.plot([d/2,v+d/2], [-0.002,-0.002], c='blue', label='velocity')
        plt.plot([-d/2-a*time_step, -d/2], [0.002,0.002], c='red')
        plt.plot([d/2,a*time_step+d/2], [0.002,0.002], c='red', label='acceleration')
        plt.xlabel('x-coordinate (Angstrom)')
        plt.ylim(-0.1, 0.1)
        plt.yticks([])
        plt.legend()

        # potential energy surface
        plt.subplot(2,1,2)
        plt.cla()
        plt.plot(PES_x, PES_y, marker='.', c='green')
        plt.xlabel('H-H distance (Angstrom)')
        plt.ylabel('DFT-B3LYP Energy (eV)')
        y0 = np.interp(d, PES_x, PES_y)
        plt.scatter(d, y0, c='red', s=100, edgecolor='black')

        plt.pause(0.001)

dyn.attach(update_plot, interval=1)

print("Starting MD Simulation...")
dyn.run(total_steps)
plt.ioff()
print("Simulation finished.")
plt.show()


