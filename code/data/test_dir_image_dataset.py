"""
Created on Wed Dec 15 18:23:24 2021
Code based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@author: Aline Sindel
"""
import os.path
from glob import glob
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset, IMG_EXTENSIONS
from PIL import Image
import PIL.ImageOps 
from data.data_utils import is_image_file
Image.MAX_IMAGE_PIXELS = None

class TestDirImageDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase, opt.nameA)
        self.dir_B = os.path.join(opt.dataroot, opt.phase, opt.nameB)
        
        self.invert = False
        if self.opt.direction == 'BtoA':
             self.img_dir = os.path.normpath(self.dir_B)
             if self.opt.invertB:
                 self.invert = True
        else:
             self.img_dir = os.path.normpath(self.dir_A)
             if self.opt.invertA:
                 self.invert = True
                 
        self.grayscale = False
        if self.opt.grayscaleA:
            self.grayscale =  True
        elif self.opt.grayscaleB:
            self.grayscale =  True           
                 
        self.list = []
        self.list.extend(os.path.normpath(os.path.join(self.img_dir, x))
                                         for x in sorted(os.listdir(self.img_dir)) if is_image_file(x))

    def __getitem__(self, index):
        file_path = self.list[index]
        img = Image.open(file_path)
        if img.mode == 'RGBA' or img.mode == 'L' or img.mode == 'CMYK':
            img = img.convert('RGB')

        if self.grayscale: #grayscale image as 3 channels
            img = img.convert('L').convert('RGB')
			
        if self.invert:
            img = PIL.ImageOps.invert(img)  			
			
        w, h = img.size
        #resize to load size
        load_size = self.opt.loadSize
        if w>h:
            f=load_size/w
        else:
            f=load_size/h
        w2 = int(w*f)
        h2 = int(h*f)
        img = img.resize((w2, h2), Image.BICUBIC)
                
        img = transforms.ToTensor()(img)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        B = A.clone()

        return {'A': A, 'B': B, 'A_paths': file_path, 'B_paths': file_path, 'w': w, 'h': h}

    def __len__(self):
        return len(self.list)

    def name(self):
        return 'TestDirImageDataset'

