# -*- coding: utf-8 -*-
import os
import re
import sys
import cPickle
import numpy as np
from skimage.io import imread
from datetime import datetime
from skimage.transform import resize
from matplotlib import pyplot as plt

class CifarKaggle():
	"""
	Lớp này để xử lý các ảnh không có văn bản
	Một thể hiện của lớp này được tạo ra bên trong lớp 
