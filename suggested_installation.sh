conda create -p $(pwd)/idre-env python=3.10 -y
conda init bash
source ~/.bashrc
conda activate $(pwd)/idre-env
conda install -c conda-forge ase pyscf -y
module load gcc/11.3.0 intel/2020.4 intel/mpi
pip install gpaw
module load cuda/11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install mace-torch

