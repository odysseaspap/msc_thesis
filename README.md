### RadNet++: *Geometric supervision model for rotational radar-camera calibration in an autonomous vehicle setup.*
Created by Odysseas Papanikolaou from Technical University of Munich. Contact: odysseaspapan@gmail.com

![network_architecture](https://github.com/odysseaspap/msc_thesis/blob/master/Figures/RadNet++.png)



### Introduction
This work was created for my MSc thesis, conducted at the Chair of Robotics, Artificial Intelligence and Embedded Systems 
of the Technical University of Munich. 
RadNet++ is a follow-up project that builds on and extends the model presented in the work "Targetless Rotational Auto-Calibration of Radar and Camera for Intelligent Transportation Systems". 
You can find arXiv version of the paper <a href="https://arxiv.org/abs/1904.08743">here</a>. 

RadNet (the v1 model) was successfully applied for the rotational calibration of traffic radar and camera sensors installed on the gantry bridges of the German highway A9. This sensor setup was created for the 
smart infrastructure project <a href="https://www.fortiss.org/en/research/projects/detail/providentia">Providentia</a>. 
While RadNet used direct quaternion supervision for the learning process, in RadNet++ we extend it and apply geometric supervision via *spatial transformer layers*. 
For the development of the model we used the <a href="https://www.nuscenes.org/">nuScenes</a> dataset.
In this repository we release code and data for our RadNet++ calibration networks as well as a few utility scripts for creating a calibration dataset from the nuScenes data. 
The latter is performed by applying random rotational de-calibrations on the ground-truth nuScenes calibration. 

### Abstract
The extrinsic calibration of automotive sensors is nowadays mostly dependent on time consuming manual or target-based approaches.
Especially for calibration between radar and camera sensors, there is an inadequacy of automatic calibration methods due 
to differing sensor modalities and lack of descriptive features in sparse radar detections. Moreover, the automatic 
methods available are merely focused on the calibration of the pan angle of the radars and are limited on specific 
driving conditions and calibration sites. To overcome these limitations, this thesis introduces a new target-less, deep 
learning model for the rotational calibration between radar and camera sensors. The proposed model extends the current 
state of the art radar-camera calibration neural network by using spatial transformer layers which allow it to utilize 
the geometric computer vision nature of the extrinsic calibration problem. The first evaluation was performed with data 
recorded from camera and radar sensors installed in the gantry bridges of the German A9 highway, an infrastructure 
created for the research project Providentia. The results show calibration accuracy under 1 degree for all rotational
axes, which surpasses the current state of the art model. In addition, it is demonstrated that the proposed
network was able to successfully calibrate the tilt and roll angles on the nuScenes dataset, where it achieved a mean 
error of 0.29 degrees in tilt and 0.82 degress in roll angle. However, in this dataset, which is collected from 
autonomous vehicle sensors, the pan error is more challenging to eliminate. This is due to the fact that most radar 
points lie on the same height when projected to the camera coordinate frame and the pan error does not affect their 
height. Nevertheless, unlike the standard automatic approaches, the proposed network is not limited to the correction 
of the pan angle nor does it require specific driving scenarios and calibration sites. 
Therefore, it can be used as an effective and robust model for the initial calibration between radar and camera sensors 
on all three rotational axes, before one of the standard approaches is used for the final pan error correction.



### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a> and <a href="https://keras.io/">Keras</a>. The code is tested under TF1.13 GPU version, Python 3.7 and Keras 2.1 on Ubuntu 16.04. 
There are also some dependencies for a few Python libraries for data processing and visualizations like `numpy`, `pandas`, `seaborn`, `json` etc. 
It's highly recommended that you have access to GPUs with more than 8GB of RAM memory.

Install the <a href="https://github.com/nutonomy/nuscenes-devkit">nuScenes-dev-kit</a> using the instructions of the authors.

#### Compile Customized TF Operators
Certain TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.13.0 and cuda 10.00. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

### Usage

#### Calibration dataset creation


Download the <a href="https://www.nuscenes.org/download">nuScenes</a> dataset. To create the dataset used the script dataset_generation/decalibrate_and_store.py. 
The code contains all the necessary comments regarding the random decalibration process and the creation of the dataset. Upon making the necessary changes in the script 
for the desired part of the nuScenes dataset, run the decalibration script via:

        python decalibrate_and_store.py

Please note this script requires the nuScenes libraries to work, which can be installed as mentioned above. 


#### Model training

To train a model and see its calibration accuracy on nuScenes, first define the training configuration parameters (batch_size, learning_rate etc) in src/run_config.py. Subsequently, run the model training: 

        python run_training.py

