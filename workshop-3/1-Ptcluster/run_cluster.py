from ase import units
from ase.md.langevin import Langevin
from ase.io import write, read
from ase.build import bulk
from mace.calculators import mace_mp
import numpy as np
import matplotlib.pyplot as plt

element = 'Pt'
calculator = mace_mp(model="../mace-mh-1.model", default_dtype="float32", device="cuda", head="oc20")
atoms = bulk(element, 'fcc', a=3.92, cubic=True) * (2, 2, 2)
atoms.center(vacuum=5.0)
atoms.set_calculator(calculator)

dyn = Langevin(atoms, 2.0*units.fs, temperature_K=800, friction=5e-3)
istep = 0
nstep = 1000
def write_frame():
    global istep
    print(f"Now in step {istep}/{nstep}")
    if istep % 10 == 0:
        dyn.atoms.write('md_traj.xyz', append=True)
    istep += 1
dyn.attach(write_frame, interval=1)
dyn.run(nstep)

