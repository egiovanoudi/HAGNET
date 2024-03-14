# HAGNET
This is the implementation of "Hybrid Attentive Graph Neural Networks for Time Series Gene Expression Clustering".

## Local Environment Setup
conda create --n `name` python=3.12.2 \
conda activate `name`

## Installing project requirements
- pandas
- scikit-learn
- torch == 2.2.1
- torch-geometric == 2.5.1

## Arguments
data_path: Path to dataset \
output_path: Path for writing results \
n_layers_g: Number of GCN layers \
n_layers_d: Number of TA-GAT layers \
n_layers_ta: Number of convolutional with batch normalization layers inside the TA mechanism \
threshold_g: Correlation threshold for the global graph \
threshold_d: Distance threshold for the dynamic graphs \
loss_t_weight: Regularization parameter for the temporal consistency loss \
lr: Learning rate \
epochs: Number of epochs

## Training
To train the model, please run `python main.py`

## Results
After the script completes, a file is created containing the final clusters.
