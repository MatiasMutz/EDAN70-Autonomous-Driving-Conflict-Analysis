# EDAN70-Autonomous-Driving-Conflict-Analysis

## Environment Setup

This project uses Conda for environment management. Follow these steps to set up your development environment:

### Prerequisites
- Anaconda or Miniconda installed on your system

### Installation Steps

1. Create the Conda environment from the provided environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate conflict-resolution
   ```

3. Follow the instructions in the jupyter notebook to run the code.

### Updating the Environment

If new dependencies are added, update your environment with:
```bash
conda env update -f environment.yml --prune
```