==========
GI-Tract-Image-Segmentation
==========

**Content**

- `Introduction`_
- `Dataset`_
- `UNet Model`_
- `Project Structure`_
- `Instructions to run our code`_


Introduction
------------
In this Kaggle competition, we are to create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. The objective is to develop algorithms using a dataset of these scans and to come up with creative deep learning solutions that will help cancer patients get better care.

We will walk you through the data of this competition, the loss function of choice, building and training our model and finally running inference on a couple of samples.

Dataset
--------

In this competition we are segmenting organs cells in images. The training annotations are provided as RLE-encoded masks, and the images are in 16-bit grayscale PNG format.

Each case in this competition is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal of this competition is to be able to generalize to both partially and wholly unseen cases.  

.. code:: none

    /train
     |---case1
         |---case1_day1
             |---scans
                 |---slice0001.png
                 |---slice0002.png
                 ...
         |---case1_day2
             |---scans
                 |---slice0001.png
                 |---slice0002.png
                 ...
         ...
     |---case2
         |---case2_day1
             |---scans
                 |---slice0001.png
                 |---slice0002.png
                 ...
     ...
  


.. figure:: https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/images/imgs_and_masks.png
    :align: center
    
    Figure1: sample scans and their corresponding masks from the UW-Madison GI Tract Image Segmentation dataset




UNet Model
----------

UNet consists of an 'encoding' and a 'decoding' part. The encoder is an alternating series of convolution-pooling layers, that extract features from the input, very much like an ordinary classifier. The decoder produces a segmentation map, based on the features derived in the encoder, by alternating transposed convolution layers (or upsampling) and convolution layers. UNet introduces skip-connections between encoder and decoder, at levels where the feature maps have the same lateral extent (number of channels). This enables the decoder to access information from the encoder, such as the general features (edges...) in the original images.
The UNet network depicted in this `paper <https://arxiv.org/pdf/2110.03352.pdf>`__ is the one we used in our project. The source code for this network implemented using MONAI is provided `here <https://docs.monai.io/en/stable/_modules/monai/networks/nets/dynunet.html>`__ . I have also implemented UNet from scratch using plain pytorch (provide below). The MONAI implementation outperformed the the later. Therefore, I decied to use the MONAI UNet. The U-Net that we are using comprises 5 levels. At each stage two convolution operations are applied, each followed by an `Instance normalization <https://paperswithcode.com/method/instance-normalization>`__  and the  `leaky ReLU <https://paperswithcode.com/method/leaky-relu>`__ activation. 

We are using the U-Net model because:

* It is a very simple architecture, which means it is easy to implement and to debug.
* Compared to other architectures, its simplicity makes it faster (less trainable parameters). This is advantageous, as we want to apply the model to a relatively large dataset within a reasonable amount of time to get a first intuition about the data. 


Project structure
------------------

In this project you will find:

* `requirements.txt <https://github.com/Rim-chan/SpaceNet7-Buildings-Detection/blob/main/requirements.txt>`__ contains all the necessary libraries;
* `args.py <https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/args.py>`__ contains all the arguments used in this project; 
* `dataloader.py <https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/dataloader.py>`__ contains the dataset and dataloader classes;
* `losses.py <https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/losses.py>`__ contains the loss function which is a combination of Dice loss and Focal loss;
* `metrics.py <https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/metrics.py>`__ contains the Dice and Hausdorff scores;
* `model.py <https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/model.py>`__ contains the UNet model;
* `main.py <https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/main.py>`__ To run training and inference.


Instructions to run our code
----------------------------

**Prepare environment**

.. code:: python

  # install MONAI 
  pip install monai  


.. code:: python

  # import the necessary libraries
  import matplotlib.pyplot as plt
  import numpy as np

.. code:: python

  # git clone source
  !git clone https://github.com/Rim-chan/GI-Tract-Image-Segmentation.git


**Train segmentation model**

.. code:: python

  !python ./GI-Tract-Image-Segmentation/main.py --base_dir ../input/uw-madison-gi-tract-image-segmentation/train --csv_path ../input/uw-madison-gi-tract-image-segmentation/train.csv

**Test segmentation model**

.. code:: python
  !mkdir predictions
  
.. code:: python
  !python ./GI-Tract-Image-Segmentation/main.py --base_dir ../input/uw-madison-gi-tract-image-segmentation/train --csv_path ../input/uw-madison-gi-tract-image-segmentation/train.csv --exec_mode 'test' --ckpt_path ./last.ckpt --save_path ./predictions/



**Load and display some samples**

.. code:: python

  preds = np.load('./predictions.npy')   #(5, 3, 224, 224)
  lbls = np.load('./labels.npy')         #(5, 3, 224, 224)

  # plot some examples
  fig, ax = plt.subplots(1,2, figsize = (10,10)) 
  ax[0].imshow(preds[0,2], cmap='gray') 
  ax[0].axis('off')
  ax[1].imshow(lbls[0,2], cmap='gray') 
  ax[1].axis('off')
  
  
  
.. figure:: https://github.com/Rim-chan/GI-Tract-Image-Segmentation/blob/main/images/prediction.png
    :align: center

    Figure2: UNet predictions and its corresponding ground truth masks 
