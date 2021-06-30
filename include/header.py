import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.data import DataLoader
from pyedflib import highlevel
from einops import rearrange, repeat
from torch.autograd import Function

import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
import time
import sys
import warnings
import datetime
import shutil
import csv

from scipy import signal
from scipy.signal import butter,filtfilt, lfilter
import mne
from tqdm import tnrange, tqdm

import multiprocessing
from multiprocessing import Process, Manager, Pool, Lock

from functools import partial

from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
import csv
from PIL import Image
import PIL.ImageOps 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device3 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")