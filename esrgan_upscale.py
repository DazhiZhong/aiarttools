import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from rudalle.realesrgan.model import RealESRGAN
from huggingface_hub import hf_hub_url, cached_download

MODELS = {
    2: dict(
        scale=2,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        scale=4,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        scale=8,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x8.pth',
    ),
}

class ESRGANUpscale():
    def __init__(self,esrganscale:int, **kwargs):
        if kwargs.get('cache_dir', None) is not None:
            self.cache_dir = kwargs.get('cache_dir')
        else:
            self.cache_dir = "models"
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"using {device} for esrganupscale")
        esrganscale = int(esrganscale)
        modeldir = Path(self.cache_dir) / MODELS[esrganscale]['filename']
        if not os.path.exists(modeldir):
            self.download(esrganscale)
        self.model = RealESRGAN(device, esrganscale)
        self.model.load_weights(f"models/RealESRGAN_x{esrganscale}.pth")
        print("Model loaded!")
    
    def download(self,scale: int):
        config = MODELS[scale]
        config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
        cached_download(config_file_url, cache_dir=self.cache_dir, force_filename=config['filename'])
    
    def gan_upscale(self,imgpath:str,outpath:str,return_image:bool=False):
        input_image = Image.open(str(imgpath))     
        input_image = input_image.convert('RGB')
        with torch.no_grad():
            sr_image = self.model.predict(np.array(input_image))
        sr_image.save(outpath)
        if return_image:
            return sr_image
        else:
            return None