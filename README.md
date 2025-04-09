# PSP-Physical-informed-Sparse-learning-for-interaction-aware-vehicle-trajectory-Prediction
Submitted to 2025 ITSC. **The whole code will be born before 5.1**  
# Highlight of our repo  
+ **Complete and detailed code comments**
+ **Clear code architecture**
+ **Easy to run locally (3.4GB GPU memory need at a batchsize of 128)**
# Paper contributions  
+ **We introduce a novel physical-informed modeling paradigm to enhance physical plausibility of predictions.** Based on the state-space representation of modern control theory, this paradigm integrates two main components: i) the vehicle kinematic component that reflects vehicles’ inherent physical dynamics, and ii) the data-driven component that captures prediction-centric features from data.
+ **We explicitly model interactions between vehicles for enhanced vehicle trajectory prediction explainability.** The aforementioned data-driven component is designed to be interaction-aware and explicitly modeled, enabling it to capture and explain complex inter-vehicle interactions. To learn the data-driven component, we design a neural network which fully accounts for interactions between vehicles.
+ **We propose a sparse learning framework to learn the prediction model efficiently.** The learning sparsity in PSP has two practices: i) feature tensors in hidden layers are sparsely sampled during forward propagation; ii) a sparsity regularization term is incorporated into the loss function. These designs enable the model to focus on critical features while reducing training complexity. 

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
## Data processing
This is a NGSIM data processing method for trajectory prediction and planning. The original NGSIM data is transformed to numpy frames (.npz). Each frame contains the ego vehicles' history trajectory, the nearby N agents' history trajectories, the map features and the ground truth of future trajectories. Firstly, you need to decompress pre-processed data in folder "original_data/". There is a csv and it is collected from https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing.  
To run data processing:
```Python
cd PSP-Physical-informed-Sparse-learning-for-interaction-aware-vehicle-trajectory-Prediction/src/
python utils/data_process.py
```
The data processing log will be printed through your terminal.  
<img src="https://github.com/user-attachments/assets/185c6e0d-fc45-4ad2-9191-a0e0a9794e71" width="60%">
## Training
Chang settings as you wish in `src/config/config.yaml`  
Create new folder：  
```python
cd ..
mkdir log/
mkdir checkpoints/
```
Train the model:  
```Python
cd src/
python train.py
```
This script will train the model for **five times** with five different random seeds.  
The **training log** will be saved in `log/`.  
The model **checkpoints** will be saved in `checkpoints/`.  
## Testing  
Create new folder：  
```Python
cd ..
mkdir test_results/
```
Testing:  
```python
python test.py
```
The **testing results** will be saved in `test_results/`
