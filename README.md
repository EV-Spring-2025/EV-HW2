# EV HW2: Dynamic 3D Gaussians

## Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
conda env create -f environment.yml
conda activate ev_hw2

git clone git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .
cd ../
```

## Data Preparation
To download the dataset, run the command:
```
bash download_data.sh
```

## Training
To train the model, run the command:
```bash
python train.py
```

## Visualization
To visualize the dynamic scenes, run the command:
```bash
python visualize.py
```
