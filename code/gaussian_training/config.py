

import os
from yacs.config import CfgNode as CN


finetune_module_dir = os.path.dirname(os.path.realpath(__file__))
diffusion_cache_dir = os.path.join(finetune_module_dir, 'cache_dir')

file_dir_path = os.path.dirname(os.path.realpath(__file__))

model_config = CN()

pipeline_config = CN()
pipeline_config.convert_SHs_python = False
pipeline_config.compute_cov3D_python = False
pipeline_config.debug = False

optim_config = CN()
optim_config.iterations = 900_000
optim_config.position_lr_init = 0.000016
optim_config.neural_feature_lr = 0.001
# optim_config.neural_renderer_lr = 0.0005
optim_config.neural_renderer_lr = 0.003
optim_config.pose2rott_lr = 0.000001
optim_config.opacity_lr = 0.05
optim_config.scaling_lr = 0.005
optim_config.rotation_lr = 0.001
optim_config.lambda_dssim = 0.5
optim_config.use_mask = True


diffu_config = CN()
diffu_config.ft_step = 4
diffu_config.image_guide = 2
diffu_config.text_guide = 12
diffu_config.diffuse_step = 10

