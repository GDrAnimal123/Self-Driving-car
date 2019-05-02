## Overview

I use supervised learning to train an autonomous car. The model use CNN that takes in camera images (from centre, left and right of the car) to predict steering angle.

My code is an updated version for who wants to train their model on lastest Keras and Tensorflow version without error.

## Quick Start

### Install required python libraries:

You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

```python
# Use TensorFlow without GPU
conda env create -f environment.yml

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

Or you can manually install the required libraries (see the contents of the environemnt*.yml files) using pip.

### Run the pretrained model

```python
python drive.py model/model-007-0.0069.h5
```

### To train the model

You need to download [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) to generate training data

Checkout [Siraj](https://www.youtube.com/watch?v=EaY5QiZwSP4&feature=youtu.be) on how to get started with the environment.

After following his video, you should get a ** data ** folder that store all the images and driving logs

Then, you can run the snippet below
```python
python model.py -d data -p model
```


## Credits
- Credits go to [naokishibuya](https://github.com/naokishibuya/car-behavioral-cloning).
- Checkout [Siraj channel](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) to get started with Machine Learning and AI.
