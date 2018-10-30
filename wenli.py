#-*- coding:utf-8 -*-
#!/usr/bin/python3

import cv2
import os
from skimage.feature import local_binary_pattern

r = 8
n = 16
pp = './r'+str(r)+'n'+str(n)
isExist = os.path.exists(pp)
if not isExist:
    os.mkdir(pp)
imgs_path = os.listdir('./datas/trainset/')
imgs_path.sort()
for i in imgs_path:
    tmp_img = cv2.imread('./datas/trainset/'+i)
    print('./datas/trainset'+i)
    tmp_img = cv2.resize(tmp_img, (256, 256))
    grey = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)                 

    n_points = n * r
    lbp = local_binary_pattern(grey, n_points, r)

   # cv2.imshow("image", tmp_img) 
   # cv2.imshow('grey', grey)
   # cv2.imshow('wenli',lbp)

   # k = cv2.waitKey(0) 
   # if k == 27:
   #     cv2.destroyAllWindows()
   # elif k == ord('s'): 
   #     cv2.imwrite('./lbps/'+imgs_path[0],lbp)
   # cv2.destroyAllWindows()

    cv2.imwrite(pp+'/'+i, lbp)
