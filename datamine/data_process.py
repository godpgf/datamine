#coding=utf-8
#author=godpgf

import numpy as np
import random

class DataProcessTool(object):

    @classmethod
    def replace_data(cls, x, dic):
         for i in range(len(x)):
            if x[i] in dic:
                x[i] = dic[x[i]]

    @classmethod
    def fast_replace_missing_data(cls, x, missing_values):
        can_use = [False if x[i] in missing_values else True for i in range(len(x))]
        for i in range(len(x)):
            if x[i] in missing_values:
                ri = random.randint(0,len(x)-1)
                while x[i] in missing_values or can_use[ri] == False:
                    ri = random.randint(0,len(x)-1)
                    x[i] = x[ri]
