# Parallelizing GNNs using CUDA, mpi4py, and Multiprocessing

## Description
This project demonstrates parallelization techniques for Graph Neural Networks (GNNs) using:
- CUDA for GPU acceleration
- MPI (mpi4py) for distributed computing
- Python Multiprocessing for parallel processing  
The implementation uses the PubMed dataset from PyTorch Geometric and a 2-layer GCN model.

---------------------------------------------------------------------------

## Requirements
- Python 3.7+
- PyTorch
- CUDA Toolkit (for GPU support)
- MPI (e.g., OpenMPI)
- Required Python packages:
  - Install with pip:
    pip install torch torchvision torchaudio
    pip install torch-geometric
    pip install mpi4py

---------------------------------------------------------------------------

## Files Overview
1. CUDA_based.py: Trains the GCN on GPU (CUDA) with full dataset.
2. sequential.py: Baseline CPU-based sequential training.
3. mpi_based.py: Distributed training using MPI (4 processes).
4. multiprocessor_based.py: Parallel training using Python's multiprocessing.
5. combined.py: Unified script supporting all modes (sequential, mpi, multiprocessing).

---------------------------------------------------------------------------

## Usage

### 1. Sequential Training (CPU)
Run:
python sequential.py

### 2. CUDA-Based Training (GPU)
Run:
python CUDA_based.py

### 3. MPI-Based Training
Run:
mpiexec -n 4 python mpi_based.py  # Use 4 MPI processes

### 4. Multiprocessing-Based Training
Run:
python multiprocessor_based.py

### 5. Combined Script (Switch Modes)
Run:
python combined.py --mode [sequential|mpi|multiprocessing]
Example:
python combined.py --mode mpi

---------------------------------------------------------------------------

## Expected Outputs
- Training Loss: Printed every 20 epochs.
- Accuracy Metrics: Train/Validation/Test accuracy after training.
- Execution Time: Total runtime for each method.
- GPU Memory Stats (CUDA): Allocated/reserved memory for GPU runs.

---------------------------------------------------------------------------

## Notes
1. Dataset: Automatically downloaded to /tmp/PubMed on first run.
2. MPI Setup: Ensure OpenMPI is installed. Adjust world_size in code for different process counts.
3. GPU Requirements: CUDA-based scripts require compatible NVIDIA GPUs.
4. Debugging: Check process status logs for MPI/multiprocessing failures.

---------------------------------------------------------------------------

## References
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- mpi4py: https://mpi4py.readthedocs.io/
- Planetoid Dataset: https://arxiv.org/abs/1603.08861
