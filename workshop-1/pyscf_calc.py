from pyscf import gto, dft
from ase import units
from ase.calculators.calculator import Calculator, all_changes
import numpy as np 


class PySCFCalculator_B3LYP(Calculator):
    """
    A custom ASE calculator that manually passes data to PySCF.
    This avoids using the built-in 'pyscf.ase' interface.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, basis, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.basis = basis

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # calculator
        mol = gto.Mole()
        dis = atoms.get_distance(0, 1)
        mol.atom = f'H {-dis/2} 0 0; H {dis/2} 0 0'
        mol.basis = self.basis
        mol.unit = 'Angstrom'
        mol.verbose = 0
        mol.build()

        # unrestricted DFT-B3LYP calculation
        dm_guess = np.zeros((2, mol.nao, mol.nao))
        dm_guess[0, 0, 0] = 1.0
        dm_guess[1, 0, 0] = -1.0
        dm_guess[0, 1, 1] = -1.0
        dm_guess[1, 1, 1] = 1.0
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.level = 1  # should be at least 3 in real application
        mf.kernel(dm0=dm_guess)
        self.results['energy'] = mf.e_tot * units.Hartree
        g = mf.nuc_grad_method().kernel()
        self.results['forces'] = -g * (units.Hartree / units.Bohr)
