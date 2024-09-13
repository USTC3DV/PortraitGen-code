from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData
import gdl
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test

class Emoca_Recog():
    def __init__(self, device, model_name='EMOCA_v2_lr_mse_20', mode='detail'):
        self.path_to_models = str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
        self.model_name =model_name
        self.mode = mode
        self.device = device
        self.model, _ = load_model(self.path_to_models, self.model_name, self.mode)
        self.model.to(self.device)
        self.model.train()
    
    def recog(self, img):
        vals = self.model.encode(img, batch={}, training=False)
        vals = self.model.decode(vals, training=False)

        return vals["expcode"]

