## do this if you have conda
conda create -p $(pwd)/idre-env python=3.10 -y
conda activate $(pwd)/idre-env
###
module load gcc/11.3.0
pip install ase
pip install --prefer-binary pyscf
module load intel/2020.4 intel/mpi
pip install gpaw
module load cuda/11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install mace-torch

