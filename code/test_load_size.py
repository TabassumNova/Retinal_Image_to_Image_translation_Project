"""
Created on Wed Dec 15 18:23:24 2021
Code based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@author: Aline Sindel
"""
import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import time

def main():
    opt = TestOptions().parse()
    opt.nThreads = 0   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    if not os.path.isdir(opt.results_dir):
        os.makedirs(opt.results_dir)
        
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    
    # test
    t = time.process_time()
    N=len(dataset)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()                                                                                            
    for i, data in enumerate(dataset):
        model.set_input(data)
        img_path = model.get_image_paths()
        print('Processing %04d (%s)' % (i+1, img_path[0]))
        model.test()
        if opt.direction == 'BtoA':
            out_name = opt.nameA
        else:
            out_name = opt.nameB
        model.write_image_load_size(opt.results_dir, out_name)
    
    elapsed_time = time.process_time() - t
    print(N)
    timePerPatch = elapsed_time/N
    print('Execution time per patch:', timePerPatch)

if __name__ == '__main__':
    main()