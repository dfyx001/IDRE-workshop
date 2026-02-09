import numpy as np
from ase.units import Hartree
from pyscf import gto, scf, dft, fci
import matplotlib.pyplot as plt
import time

bond_lengths = np.linspace(0.3, 3.3, 51)
all_frames = []

def build_mol(d):
    mol = gto.Mole()
    mol.atom = f'H 0 0 0; H 0 0 {d}'
    mol.basis = 'cc-pVDZ'
    mol.unit = 'Angstrom'
    mol.verbose = 0
    mol.symmetry = False
    mol.build()
    return mol

HF_energies = []
DFTb3lyp_energies = []
DFTpbe_energies = []
FCI_energies = []
for d in bond_lengths:

    # Hartree-Fock (unrestricted)
    mol = build_mol(d)
    dm_guess = np.zeros((2, mol.nao, mol.nao))
    dm_guess[0, 0, 0] = 1.0
    dm_guess[1, 0, 0] = -1.0
    dm_guess[0, 1, 1] = -1.0
    dm_guess[1, 1, 1] = 1.0
    mf = scf.UHF(mol)
    mf.kernel(dm0=dm_guess)
    HF_energies.append(mf.e_tot * Hartree)

    # Density-Functional-Theory, B3LYP (unrestricted)
    mol = build_mol(d)
    mf = dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.grids.level = 1  # should be at least 3 in real application
    mf.kernel(dm0=dm_guess)
    DFTb3lyp_energies.append(mf.e_tot * Hartree)

    # Density-Functional-Theory, PBE (unrestricted)
    mol = build_mol(d)
    mf = dft.UKS(mol)
    mf.xc = 'pbe,pbe'
    mf.grids.level = 1  # should be at least 3 in real application
    mf.kernel(dm0=dm_guess)
    DFTpbe_energies.append(mf.e_tot * Hartree)

    # Full-CI
    mol = build_mol(d)
    mf = scf.RHF(mol).run()
    cisolver = fci.FCI(mf)
    energy = cisolver.kernel()[0]
    FCI_energies.append(energy * Hartree)

    print(f"H-H distance {round(d,2)}, E(HF): {round(HF_energies[-1],6)}, \
E(DFTpbe) {round(DFTpbe_energies[-1],6)}, \
E(DFTb3lyp) {round(DFTb3lyp_energies[-1],6)}, \
E(Full-CI) {round(FCI_energies[-1],6)}")

HF_lines = [str(bond_lengths[i])+' '+str(HF_energies[i])+'\n' for i in range(len(HF_energies))]
DFTb3lyp_lines = [str(bond_lengths[i])+' '+str(DFTb3lyp_energies[i])+'\n' for i in range(len(DFTb3lyp_energies))]
DFTpbe_lines = [str(bond_lengths[i])+' '+str(DFTpbe_energies[i])+'\n' for i in range(len(DFTpbe_energies))]
FCI_lines = [str(bond_lengths[i])+' '+str(FCI_energies[i])+'\n' for i in range(len(FCI_energies))]
with open("PES_HF.txt", 'w') as fo:
    fo.writelines(HF_lines)
with open("PES_DFTb3lyp.txt", 'w') as fo:
    fo.writelines(DFTb3lyp_lines)
with open("PES_DFTpbe.txt", 'w') as fo:
    fo.writelines(DFTpbe_lines)
with open("PES_FCI.txt", 'w') as fo:
    fo.writelines(FCI_lines)

plt.plot(bond_lengths, HF_energies, marker='.', c='blue', label='Hartree-Fock')
plt.plot(bond_lengths, DFTpbe_energies, c='gray', label='DFT (PBE)', ls='--')
plt.plot(bond_lengths, DFTb3lyp_energies, marker='.', c='green', label='DFT (B3LYP)')
plt.plot(bond_lengths, FCI_energies, marker='.', c='red', label='Full-CI')
plt.xlabel('H-H distance (Angstrom)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()

