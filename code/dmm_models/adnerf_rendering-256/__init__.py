import os

part = 'wholebody'
file_dir_path = os.path.dirname(os.path.realpath(__file__))
adnerf_rendering_part_ids_path = os.path.join(file_dir_path, part + '_sel.txt')
adnerf_teeth_mesh_file = os.path.join(file_dir_path, part + '_tris.obj')
adnerf_gaussian_info_file = os.path.join(file_dir_path, part + '_tris_info.pt')
