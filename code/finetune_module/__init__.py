# from .ip2p_editor import InstructPix2Pix
# from .ip2p_editor import InstructPix2Pix
# from .ip2p_editor import InstructPix2Pix
from .get_prompt import getprompt
import os
finetune_module_dir = os.path.dirname(os.path.realpath(__file__))
diffusion_cache_dir = os.path.join(finetune_module_dir, 'cache_dir')