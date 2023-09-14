print('start')
from inference import SegRetino

# Initializing the SegRetino Inference
seg = SegRetino(img_path="results/input/ABNORMAL_0002_VIS.jpg")

# Running inference
seg.inference(set_weight_dir = 'unet_new3.pth', path = 'out3_ABNORMAL_0002_VIS.jpg', blend_path = 'blend_out3_ABNORMAL_0002_VIS.jpg')
