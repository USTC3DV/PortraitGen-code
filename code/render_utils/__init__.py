### rendering utils for visualizetion, differentiable rendering, ...
from .render_nvdiff import MeshRenderer

def enlarge_human_masks(masks, enlarge_size = 30):
    ### masks: (b, h, w, ...)
    ### enlarge mask (<- ^ ->) for mask loss robust to occlusion

    masks_left = masks.clone()
    masks_left[:, :, :-enlarge_size] = masks[:, :, enlarge_size:]

    masks_right = masks.clone()
    masks_right[:, :, enlarge_size:] = masks[:, :, :-enlarge_size]

    masks_up = masks.clone()
    masks_up[:, :-enlarge_size] = masks[:, enlarge_size:]

    return ((masks + masks_left + masks_right + masks_up) > .5).float()

def enlarge_upper_masks(masks, enlarge_size = 100):
    ### masks: (b, h, w, ...)
    ### enlarge mask (<- ^ ->) for mask loss robust to occlusion

    masks_up = masks.clone()
    masks_up[:, :-enlarge_size] = masks[:, enlarge_size:]

    return ((masks + masks_up) > .5).float()