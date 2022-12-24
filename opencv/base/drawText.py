#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/12/24 23:26
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : drawText.py
# @Software: PyCharm
# description:
import cv2
import numpy as np
from PIL import ImageFont,ImageDraw,Image
import os


#img = np.zeros((480,640,3),np.uint8)
#img = np.full((480,640,3),fill_value=0,dtype=np.uint8)

path = os.path.abspath('../images')
img = cv2.imread(os.path.join(path,'./cat.jpeg'))

font = ImageFont.truetype(os.path.join(path,'./HGZY_CNKI.TTF'),28)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((200,150),'你好，我好，大家好',font=font,fill=(0,105,100,100))
img = np.array(img_pil)

#cv2.putText(img,'',(150,250),cv2.FONT_HERSHEY_SIMPLEX,2,[255,155,0])
cv2.imshow('drwa',img)
cv2.waitKey(0)
cv2.destroyAllWindows()