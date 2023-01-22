# open_science_mnist
Code for the winter 2022/2023 seminar Open Science in Statistics &amp; Machine Learning LMU Munich  

## The Goals of this project are as follows:  

* Build an image classification neural network for MNIST using PyTorch  
* Experiment with an image augmentation where we first flip the images horizontally and then rotate them by 90 degrees; with each augmentation happening with a probability of 50%  
* Manage dependencies with pipenv & prepare a two versions of the code with at least one outdated functionality  
* Provide a Github repo that can be cloned to reproduce the code  

Analysis should be provided as a Jupyter Notebook that must be executable in Google Colab  

The goal is not a good classification performance, but experiment sharing.  
The code should be able to be executed with:  
* Minimal Effort
* Without any information beyond this simple readme
* With the exact same result achieved here

## How to use this repository?
This repo is self contained, to replicate the experiments simply download the Jupyter Notebook from here:  
https://github.com/benearnthof/open_science_mnist/blob/main/open_science_mnist.ipynb  
Upload it to Colab:  
https://colab.research.google.com  
And start executing the cells.  
One of the first Cells will pull this repo to colab so you can edit the code on there and change the experiments and code to suit your needs.  
Most of the code and cells have added comments, the pipenv setup is further specified in pipenv_readme.txt  

## How to compare results?  
Screenshots of the sampling of data, and the training and performance of the convnet are included in the `images` folder.  
If everything goes well they should match their respective cell outputs. 
