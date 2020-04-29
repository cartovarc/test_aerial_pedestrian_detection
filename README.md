# Testing aerial_pedestrian_detection
In this repo we will test aerial pedestrian detection pretrained model from this repo https://github.com/priya-dwivedi/aerial_pedestrian_detection

## Clone original repo
```sh
git clone https://github.com/priya-dwivedi/aerial_pedestrian_detection.git
```

## Install dependencies on new env
```sh
conda create -n aerial_test pip python=3.6.9
conda activate aerial_test

conda install tensorflow=1.15
conda install -c conda-forge matplotlib

pip install . # Setup aerial_pedestrian_detection # inside aerial_pedestrian_detection folder

```

## Copy pretrained model 
Download *resnet50_csv_12_inference.h5* from *https://drive.google.com/drive/u/1/folders/1QpE_iRDq1hUzYNBXSBSnmfe6SgTYE3J4* and put in new folder called *models* inside *aerial_pedestrian_detection*


## Create results folder
inside *aerial_pedestrian_detection* folder, create a folder called *results".

## Alternative to create folders
```sh
mkdir results && mkdir models
```

## Copy *test.py* 
Copy *test.py* script to *aerial_pedestrian_detection*

## Test
```sh
python test.py
```

## Check results
after run test.py, you will get in *results* folder all processed images from *examples* folder.
