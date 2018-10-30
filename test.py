#!/usr/bin/python3.6
#-*- coding:utf-8 -*-

from train import Network
import numpy as np
import argparse
import cv2
import time
import os

workspace = os.getcwd()
batch_size = 1
patch_size = (256, 256)
classnames = ['OCEAN', 'MOUNTAIN', 'LAKE', 'FARMLAND', 'DESERT', 'CITY']


def accessfile(filename):
    tmp_batch = []
    tmp_pic = cv2.imread(workspace+'/datas/testset/'+filename)
    tmp_repic = cv2.resize(tmp_pic, patch_size)
    tmp_repic = tmp_repic*1.0/255
    tmp_batch.append(tmp_repic)
    result = np.array(tmp_batch)
    return result


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-m", required=True)
    args = parse.parse_args()
    start_t = time.time()
    network = Network()
    network.load_weights(args.m)
    tmp_filepath = os.path.join(workspace, 'datas/testset')
    tmp_filenames = sorted(os.listdir(tmp_filepath))
    with open('result.csv', 'w') as res:
        for tmp_filename in tmp_filenames:
            result = accessfile(tmp_filename)
            preds = network.predict(result, batch_size=batch_size, verbose=1)
            print(preds)
            results = np.argmax(preds, axis=1)
            res.write(tmp_filename)
            res.write(',')
            res.write(classnames[results[0]])
            res.write('\n')
    end_t = time.time()
    run_t = end_t - start_t
    print('runtime is : ', run_t)
