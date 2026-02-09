import numpy as np
from ase.build import bulk
from ase.md.langevin import Langevin
from ase import units, Atoms
from ase.io import write
from gpaw import GPAW, PW
from gpaw.mpi import world

element = 'Pt'
atoms = bulk(element, 'fcc', a=3.92, cubic=True) * (2, 1, 1)  # should be larger supercell for real calculations
atoms.calc = GPAW(mode=PW(300),
            xc='PBE',
            kpts=(1, 2, 2), # should be denser K-points for real calculations
            parallel={'kpt':4, 'band':8},
            symmetry='off',
            occupations={'name':'methfessel-paxton', 'width':0.2, 'order':1},
            txt='GPAW.log')

time_step = 2.0 * units.fs
dyn = Langevin(atoms, time_step, 
               temperature_K=2000, 
               friction=0.01 / units.fs)

train_dataset = []
test_dataset = []
istep = 0
nstep = 150
def collect_data():
    global istep
    record_atoms = Atoms(symbols=atoms.get_chemical_symbols(),
                        positions=atoms.get_positions(),
                        cell=atoms.get_cell(),
                        pbc=atoms.get_pbc())
    record_atoms.info['REF_energy'] = atoms.get_potential_energy()
    record_atoms.arrays['REF_forces'] = atoms.get_forces()
    if world.rank == 0:
        print(f"Step {istep}/{nstep}. Energy: {atoms.get_potential_energy():.4f} eV")
        if 30 < istep <= 130: train_dataset.append(record_atoms)
        elif istep > 130: test_dataset.append(record_atoms)
    istep += 1

dyn.attach(collect_data, interval=1)
dyn.run(steps=nstep)
if world.rank == 0:
    write('train_Pt.xyz', train_dataset, format='extxyz')
    write('test_Pt.xyz', test_dataset, format='extxyz')

