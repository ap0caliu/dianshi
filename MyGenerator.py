#!/usr/bin/python3.6
#-*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from random import shuffle
import time

class myGenerator(object):
    def __init__(self, ratio, patch_size, batch_size, classnames, data_path):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.classnames = classnames
        self.data_path = data_path
        csv = self.data_path+'/datas/labels.csv'
        with open(csv,'r') as f:
            self.lines = f.readlines()
        data_num = len(self.lines)
        self.train_set = self.lines[:int(data_num*ratio)]
        self.valid_set = self.lines[int(data_num*ratio):]
        self.train_batch = len(self.train_set) // batch_size
        self.valid_batch = len(self.valid_set) // batch_size
        # self.train_batch = 1
        # self.valid_batch = 1

    def oneHot(self, filelabel):
        label_matrix = np.eye(len(self.classnames))
        for i in range(len(self.classnames)):
            if self.classnames[i] == filelabel:
                return label_matrix[i]


    def generator(self, trainflag):
        while True:
            shuffle(self.lines)
            if trainflag == 1:
                for i in range(self.train_batch):
                    tmp_batch = []
                    tmp_label = []
                    for j in range(self.batch_size):
                        tmp_filename = self.train_set[i*self.batch_size+j].split(',')
                        tmp_filepath = self.data_path+'/datas/trainset/'+tmp_filename[0]
                        tmp_filelabel = tmp_filename[1].strip('\n')
                        tmp_pic = cv2.imread(tmp_filepath)
                        tmp_label.append(self.oneHot(tmp_filelabel))
                        tmp_repic = cv2.resize(tmp_pic, self.patch_size)
                        tmp_repic = tmp_repic*1.0/255
                        tmp_batch.append(tmp_repic)
                    yield (np.array(tmp_batch), np.array(tmp_label))
            else:
                for i in range(self.valid_batch):
                    tmp_batch = []
                    tmp_label = []
                    for j in range(self.batch_size):
                        tmp_filename = self.valid_set[i*self.batch_size+j].split(',')
                        tmp_filepath = './datas/trainset/'+tmp_filename[0]
                        tmp_filelabel = tmp_filename[1].strip('\n')
                        tmp_pic = cv2.imread(tmp_filepath)
                        tmp_label.append(self.oneHot(tmp_filelabel))
                        tmp_repic = cv2.resize(tmp_pic, self.patch_size)
                        tmp_repic = tmp_repic*1.0/255
                        tmp_batch.append(tmp_repic)
                    yield (np.array(tmp_batch), np.array(tmp_label))

