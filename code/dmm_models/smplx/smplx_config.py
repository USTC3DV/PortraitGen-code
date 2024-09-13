#--------------------------------------
#功能: 存储默认参数
#--------------------------------------

import os
from yacs.config import CfgNode as CN

smlx_config = CN()


file_dir_path = os.path.dirname(os.path.realpath(__file__))
smlx_config.model_path = os.path.join(file_dir_path, 'SMPLX2020')
