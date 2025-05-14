# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate

conda create -n mtl python=3.9.7
conda activate mtl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install requirements
cd ~/PSMG
python -m pip install -e .


# Navigate to experiments folder
cd ~/PSMG/experiments/nyuv2

# Create dataset
mkdir -p dataset
mkdir -p dataset/nyuv2
wget -O dataset/nyuv2.zip "https://www.dropbox.com/scl/fo/p7n54hqfpfyc6fe6n62qk/AKVb28ZmgDiGdRMNkX5WJvo?rlkey=hcf31bdrezqjih36oi8usjait&st=gp1xv3fi&dl=0"
sudo apt-get install unzip -y
unzip dataset/nyuv2.zip -d dataset/nyuv2
rm dataset/nyuv2.zip


# # modify before run
# nano run.sh

# nano trainer.py
# torch.set_num_threads(12)  # Use 12 cores per experiment (4 experiments × 12 = 48 cores)
# torch.set_float32_matmul_precision('high')  # Faster matrix operations
# os.environ['OMP_NUM_THREADS'] = '12'  # OpenMP threads
# os.environ['MKL_NUM_THREADS'] = '12'  # Intel MKL threads

# DataLoader(
#     num_workers=8,        # Optimal for 48-core machines
#     persistent_workers=True  # Maintains workers between epochs
# )

# # check log file
# tail -f "trainlogs/log_pmgd-gamma0.001-42-batch2-agent5-sparsity0.6.log"