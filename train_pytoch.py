# set GPU ID
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #multi gpu: "0,1"
print("cml26: GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

'''
MIT License
Copyright (c) 2018 Wentao Yuan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

#import tensorflow as tf
#from tf_util import *

# import python package
import argparse
import datetime
import importlib
import time

# import install package
import torch
import torch.optim as optim
from tqdm import tqdm

# import from file
import models ###need to implement PCN
from data_util import lmdb_dataflow, get_queued_data  ###need to implement Pytorch
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views

# unused package 
#import tensorflow as tf


# setting device GPU/CPU


# main function
if __name__ == '__main__':
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/tmp3/ruby2332ruby/PCN/PCN_dataset/shapenet')
    ##########need to implement more
    args = parser.parse_args()
    train(args)