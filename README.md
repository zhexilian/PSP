# PSP-Physical-informed-Sparse-learning-for-interaction-aware-vehicle-trajectory-Prediction
Submitted to 2025 ITSC 
# Getting start
## Installation
Clone this repo firstly:    
```Python
git clone https://github.com/zhexilian/PSP-Physical-informed-Sparse-learning-for-interaction-aware-vehicle-trajectory-Prediction.git
```
Create the virtual conda environment using the `environment.yaml`:    
```Python
cd PSP-Physical-informed-Sparse-learning-for-interaction-aware-vehicle-trajectory-Prediction/
cd src/
conda env create -f environment.yaml
conda activate 25ITSC
```
## Data process
This is a NGSIM data process method for trajectory prediction and planning. The original NGSIM data is transformed to numpy frames (.npz). Each frame contains the ego vehicles' history trajectory, the nearby N agents' history trajectories, the map features and the ground truth of future trajectories. To run data process:
```Python
cd cd PSP-Physical-informed-Sparse-learning-for-interaction-aware-vehicle-trajectory-Prediction/src/
python utils/data_process.py
```
