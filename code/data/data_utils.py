# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:57:41 2019

@author: Aline Sindel
"""



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF"])

def is_text_file(filename):
    return any(filename.endswith(extension) for extension in [".csv", ".txt"])