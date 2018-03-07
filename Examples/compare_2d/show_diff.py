# coding: utf-8
import numpy as np
d1 = np.load('orig_crank.npy')
d2 = np.load('new.npy')
sq_sum_diff = np.sum((d1 - d2)**2)
print("square sum diff bewteen old and new crank is %e" % sq_sum_diff)
