import os
import sys
sys.path.append('..')
from configuration.config import *
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import smplx
import trimesh
import pickle
import numpy as np
import torch
from tqdm import tqdm

from utils.viz_util import render_generation_multview
from evaluation.load_results import synthesis_results_dict

scene_meshes = {}
for scene_name in scene_names:
    mesh_path = Path.joinpath(scene_folder, scene_name + '.ply').__str__()
    scene_meshes[scene_name] = trimesh.load_mesh(mesh_path)

body_model_dict = {
        'male': smplx.create(smplx_model_folder, model_type='smplx',
                             gender='male', ext='npz',
                             num_pca_comps=num_pca_comps),
        'female': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='female', ext='npz',
                               num_pca_comps=num_pca_comps),
        'neutral': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=num_pca_comps)
    }

default_colors = np.ones((10475, 3)) * np.array([0.80, 0.80, 0.80])

"""
Input:
    results: frames
    render_dir: directory to save rendering
    max_render_num: maximum number of samples per semantic to be rendered
    num_view: number of rendering views for each sample 
"""
def render_results(results, render_dir, max_render_num=10, num_view=2):
    for inter_idx, generation in enumerate(tqdm(results)):
        generation_dir = Path.joinpath(render_dir, generation)
        generation_dir.mkdir(parents=True, exist_ok=True)
        step_size = max(1, len(results[generation]) // max_render_num)
        for idx in range(0, len(results[generation]), step_size):
            generation_params = results[generation][idx]
            scene_mesh = scene_meshes[generation_params['scene']]
            if 'gender' in generation_params:
                body_model = body_model_dict[generation_params['gender']]
            else:
                body_model = body_model_dict['neutral']
            for key in smplx_param_names:
                if key in generation_params:
                    generation_params[key] = torch.tensor(generation_params[key], dtype=torch.float32).cpu()
            generation_params['left_hand_pose'] = generation_params['left_hand_pose'][:, :num_pca_comps]
            generation_params['right_hand_pose'] = generation_params['right_hand_pose'][:, :num_pca_comps]
            # print(generation_params)
            vertices = body_model(**generation_params).vertices.detach().cpu().numpy()
            body = trimesh.Trimesh(vertices[0], body_model.faces, vertex_colors=default_colors,
                                   process=False)

            img_collage = render_generation_multview(body, scene_mesh, clothed_body=None,
                                        body_center=True,
                                        num_view=num_view,
                                        collage_mode='grid' if num_view == 4 else 'vertical')
            export_path = Path.joinpath(generation_dir, generation + '_' +generation_params['scene'] + '_' + str(idx // step_size) + '.png')
            print(export_path)
            img_collage.save(export_path)

if __name__ == '__main__':
    """render using multiple views or generation results from different sources"""
    for method in synthesis_results_dict:
        print('render for ', method)
        render_results(synthesis_results_dict[method], Path.joinpath(render_folder, method + '_2view'),
                       max_render_num=16, num_view=2)
