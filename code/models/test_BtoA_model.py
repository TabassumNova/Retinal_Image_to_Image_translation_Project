from .base_model import BaseModel
from . import networks
from .cycle_gan_model import CycleGANModel
import util.util as util
from PIL import Image
import numpy as np
import os

class TestBtoAModel(BaseModel):
    def name(self):
        return 'TestBtoAModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_B', 'fake_A']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_B']
        #self.model_names = ['G' + opt.model_suffix]

        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        #setattr(self, 'netG' + opt.model_suffix, self.netG)
        

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['B_paths']
        #self.image_paths = input['A_paths']
        if 'w' in input:
            self.input_w = input['w']
        if 'h' in input:
            self.input_h = input['h']
            
    def forward(self):
        self.fake_A = self.netG_B(self.real_B)
        
    def write_image(self, out_dir):
        image_numpy = util.tensor2im(self.fake_A.detach())
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
        image_pil = image_pil.resize((self.input_w[0], self.input_h[0]), Image.BICUBIC)
        name, _ = os.path.splitext(os.path.basename(self.image_paths[0]))
        out_path = os.path.join(out_dir, name + self.opt.suffix + '.png')
        image_pil.save(out_path)
    def write_image_load_size(self, out_dir, nameA):
        image_numpy = util.tensor2im(self.fake_A.detach())
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))        
        name, _ = os.path.splitext(os.path.basename(self.image_paths[0]))
        out_path = os.path.join(out_dir, name + "_" + nameA + '.png')
        image_pil.save(out_path) 														  
