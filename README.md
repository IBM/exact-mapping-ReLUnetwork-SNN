# An Exact Mapping From ReLU Networks to Spiking Neural Networks
[![DOI](https://zenodo.org/badge/680052801.svg)](https://zenodo.org/badge/latestdoi/680052801)

This repository contains code material for the publication: Stanojevic, A., Woźniak, S., Bellec, G., Cherubini, G., Pantazi, A., &amp; Gerstner, W. (2022). An Exact Mapping From ReLU Networks to Spiking Neural Networks. arXiv preprint arXiv:2212.12522. https://arxiv.org/abs/2212.12522 

Deep spiking neural networks (SNNs) offer the promise of low-power artificial intelligence. However, training deep SNNs from scratch or converting deep artificial neural networks to SNNs without loss of performance has been a challenge. An exact mapping enables to convert a pretrained high-performance network with Rectified Linear Units (ReLUs) to an energy-efficient SNN with zero percent drop in accuracy. The exact mapping is coded in Python and Tensorflow.

## Usage
This repository contains code which maps different ReLU networks to spiking neural networks for different benchmark datasets. To use this repository, please download pretrained 'models' folder from the link https://ibm.box.com/v/models-exact-mapping and place it inside the root folder.  
  

The provided Jupyter notebook illustrates how the mapping works. 

To use the code please create an anaconda environment using the configuration file (on Linux platform):
```console
foo@bar:~$ conda env create -f environment.yml
```
The execution of the algorithm involves two phases. The calling arguments and hyperparameters for both phases below are found in *flags.txt*.

**Phase1: Preprocessing phase**

First, run the preprocessing step which takes pretrained original ReLU model as input and preprocesses it. The outputs of the preprocessing phase are: 
1. Prediction labels of the original ReLU network (to be compared with the SNN for the Agreement metric)
2. Parameters of the scaled (preprocessed) ReLU network (without BN layers, etc.)
3. Maximum layer outputs of the scaled (preprocessed) ReLU network (for the calculation of [t_min^(n), t_max^(n)] intervals)
4. Accuracy of the original as well as the scaled (preprocessed) ReLU network

The original ReLU model is prestored at "logging_dir" path (except for PLACES365 dataset) and the outputs of the first phase will also be stored there. 
For example, for MNIST dataset and fully-connected (FC) model the following command is executed:
```console
foo@bar:~$ python ./src/Phase1_ReLU_model_preprocessing.py --logging_dir="./models/MNIST/FC/" --data_name="MNIST" --model_name="mnist_fc_relu_original" --CNN=False --layers=2 
```
**Phase2: Conversion phase**

Secondly, run the conversion step which takes parameters of the scaled ReLU model and maximum layer outputs as input and creates an SNN model. The outputs of the conversion phase are Accuracy of the SNN model and Agreement metric. 

For example, for MNIST dataset and fully-connected (FC) model the following command is executed:
```console
foo@bar:~$ python ./src/Phase2_conversion_to_SNN.py --logging_dir="./models/MNIST/FC/" --data_name="MNIST" --model_name="mnist_fc_relu_scaled" --CNN=False --layers=2 
```

### Large datasets/models

**PLACES365**

1. *Download pretrained caffe VGG model*: places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel and place it inside ./src/places365 folder.

2. *Extract sample data*: Download training and validation data to a certain *path*. Execute take_train_sample.py to retrive a sample of training images; Specify the *path* to the training data as well as the *store_path*, where the training sample will be stored;

3. *Create a docker environment*: Inside PLACES365_pretrained_model_preprocessed_data.py specify paths to original validation and training sample images (e.g. *store_path*), as well as VAL_PART and BATCH_SIZE to potentially preprocess validation data in batches. Multiple outputs are saved: (i) model parameters in a dictionary (ii) validation data (to which specific caffe preprocessing was applied) (iii) training data sample (to which specific caffe preprocessing was applied). 

Execute: 
```console
foo@bar:~$ docker build -t places365_container . 
foo@bar:~$ docker run  -v /root/caffe/models_places places365_container python PLACES365_pretrained_model_preprocessed_data.py 
```
4. *Prepare everything for running the two phases:* Place the original ReLU model (saved model parameters) inside './models/PLACES365/VGG' folder (similarly as done for other datasets) and set the path to the preprocessed data (--places_data_path). Everything is now ready to run the two previously described phases.


**PASS**

1. Since the pretrained model is large to store, in Phase1 the pretrained VGG network is directly loaded from the Tensorflow repository (no action is required here).

2. Make sure to set the path to the data (--pass_data_path).

## Feedback
If you have feedback or want to contribute to the code base, please feel free to open Issues or Pull Requests via GIT directly.
