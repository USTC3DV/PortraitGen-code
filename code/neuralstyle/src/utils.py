from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from settings import DEVICE, SIZE

# Image Transforms
loader = T.Compose([
  T.Resize(SIZE),
  T.CenterCrop(SIZE),
  T.ToTensor()
])

unloader = T.ToPILImage()

def load_image(path):
	image = loader(Image.open(path)).unsqueeze(0)
	return image.to(DEVICE, torch.float)

def save_image(tensor, path):
	image = unloader(tensor.cpu().clone().squeeze(0))
	image.save(path)