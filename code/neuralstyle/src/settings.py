import torch

DEVICE = torch.device('cuda')
SIZE = 512
EPOCHS = 300
STYLE_PATH = '../input/raphael.jpg'
STYLE_WEIGHT = 1000000
CONTENT_WEIGHT = 1
OUTPUT_PATH = '../output/ra_0000025.png'