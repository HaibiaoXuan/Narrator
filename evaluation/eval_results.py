import sys

import numpy

sys.path.append('..')

import torch
import json
import numpy as np
import smplx
import trimesh
import scipy.cluster
from scipy.stats import entropy
from scipy.spatial import KDTree
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from copy import copy

from configuration.config import *
from data.scene import scenes, to_trimesh
from load_results import synthesis_results_dict
from models.mesh import Mesh
from utils.metric_utils import load_scene_data, read_sdf, eval_physical_metric
from utils.chamfer_distance import chamfer_dists

# DEBUG = True
DEBUG = False

"""Get bodu vertices in scene coordinates. """
def get_vertices(generation_param, scene_coords='PROX'):
    if 'gender' in generation_param:
        body_model = body_model_dict[generation_param['gender']]
    else:
        body_model = body_model_dict['neutral']
    scene_name = generation_param['scene']
    for key in smplx_param_names:
        if key in generation_param:
            generation_param[key] = torch.tensor(generation_param[key], device=device)
    vertices = body_model(**generation_param).vertices.detach()

    return vertices

def calc_diversity(body_params, cls_num=20):
    # print(body_params.shape)
    if body_params.shape[0] < cls_num:  #deals with very few samples
        cls_num = max(1, body_params.shape[0] // 10)
    ## k-means
    codes, dist = scipy.cluster.vq.kmeans(body_params, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(body_params, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = np.histogram(vecs, np.arange(len(codes) + 1))  # count occurrences  count: [20]
    ee = entropy(counts)
    return {'entropy': float(ee),
            'mean_dist': float(np.mean(dist))}

def evaluate_diversity(results, method='default'):
    diversity_metric = {'all': []}
    colors = []
    # diversity_metric = {}
    for generation in results:
        smplx_params = []
        combination_name = generation[generation.find('_') + 1:]
        atomics = combination_name.split('+')
        verbs = [atomic.split('-')[0] for atomic in atomics]
        verb_ids = [action_names_train.index(verb) for verb in verbs]
        color_id = np.prod(np.array(verb_ids)) % 233
        for generation_param in results[generation]:
            smplx_param = []
            for param in used_smplx_param_names:
                if param in ['transl', 'global_orient', 'betas']:
                    continue
                smplx_param.append(np.asarray(generation_param[param].detach().cpu()) if torch.is_tensor(generation_param[param]) else generation_param[param])
            smplx_param = np.concatenate(smplx_param, axis=1)
            if np.isnan(smplx_param).any():
                print(generation_param)
                continue
                smplx_param = numpy.zeros_like(smplx_param)
            smplx_params.append(smplx_param)

        diversity_metric[generation] = np.concatenate(smplx_params, axis=0)
        diversity_metric['all'].append(diversity_metric[generation])
        colors += [color_id] * diversity_metric[generation].shape[0]
    diversity_metric['all'] = np.concatenate(diversity_metric['all'], axis=0)
    if len(diversity_metric['all']) == 0:
        return {}

    # """visualization"""
    # # Transform the data
    # data = diversity_metric['all']
    # codes, _ = scipy.cluster.vq.kmeans(data, 50)  # codes: [20, 72], dist: scalar
    # vecs, dist = scipy.cluster.vq.vq(data, codes)  # assign codes, vecs/dist: [1200]
    # # t-sne
    # img_file = results_folder / 'tsne_new' /(method + '_.png')
    # img_file.parent.mkdir(exist_ok=True)
    # tsne = TSNE(n_components=2, verbose=0)
    # z = tsne.fit_transform(data)
    # plt.scatter(z[:, 0], z[:, 1], s=5, c=vecs, cmap='hsv')
    # plt.axis('off')
    # plt.savefig(str(img_file))
    # plt.clf()
    #
    # # pca
    # img_file = results_folder / 'pca' / (method + '_.png')
    # img_file.parent.mkdir(exist_ok=True)
    # z = PCA(2).fit_transform(data)
    # plt.scatter(z[:, 0], z[:, 1], s=5, c=vecs, cmap='hsv')
    # plt.axis('off')
    # plt.savefig(str(img_file))
    # plt.clf()

    # diversity using different number of clusters
    for key in diversity_metric:
        diversity_metric[key] = {
            1: calc_diversity(diversity_metric[key], cls_num=1),
            20: calc_diversity(diversity_metric[key], cls_num=20),
            50: calc_diversity(diversity_metric[key], cls_num=50),
            150: calc_diversity(diversity_metric[key], cls_num=150),
                                 }

    # for key in diversity_metric:
    #     diversity_metric[key] = calc_diversity(diversity_metric[key], cls_num=3)
    # per_generation_metrics = []
    # for key in diversity_metric:
    #     per_generation_metrics.append(diversity_metric[key])
    # diversity_metric['all'] = {
    #     'entropy': np.asarray([metric['entropy'] for metric in per_generation_metrics]).mean(),
    #     'mean_dist': np.asarray([metric['mean_dist'] for metric in per_generation_metrics]).mean(),
    # }

    return diversity_metric

def calc_physical_metric(vertices, scene_data):
    nv = float(vertices.shape[1])
    x = read_sdf(vertices, scene_data['sdf'],
                            scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                            mode='bilinear').squeeze()

    contact_thresh = 0
    if x.le(contact_thresh).sum().item() < 1:  # if the number of negative sdf entries is less than one
        contact_score = torch.tensor(0.0)
    else:
        contact_score = torch.tensor(1.0)
    non_collision_score = (x >= 0).sum().float() / nv

    return float(non_collision_score.detach().cpu().squeeze()), float(contact_score.detach().cpu().squeeze())

def evaluate_physical_plausibility(results):
    contact_metric = {'all': []}
    non_collision_metric = {'all': []}
    for generation in tqdm(results):
        non_collision_scores = []
        contact_scores = []
        for generation_param in results[generation]:
            vertices = get_vertices(generation_param, scene_coords='Narrator')
            non_collision_score, contact_score = calc_physical_metric(vertices, scene_data_dict[generation_param['scene']])
            non_collision_scores.append(non_collision_score)
            contact_scores.append(contact_score)

        contact_metric[generation] = contact_scores
        contact_metric['all'] += contact_scores
        non_collision_metric[generation] = non_collision_scores
        non_collision_metric['all'] += non_collision_scores

    for key in contact_metric.keys():
        contact_metric[key] = float(np.array(contact_metric[key]).mean())
        non_collision_metric[key] = float(np.array(non_collision_metric[key]).mean())

    return contact_metric, non_collision_metric

# semantic sdf is very inaccurate
def calc_semantic_accuracy(vertices, scene_data):
    x_semantics = read_sdf(vertices, scene_data['semantics'],
                           scene_data['grid_dim'], scene_data['grid_min'],
                           scene_data['grid_max'], mode="bilinear").squeeze()

    print(np.unique(scene_data['semantics'].cpu().numpy()))
    print(scene_data['scene_semantics'].shape)
    if DEBUG:
        x_semantics = x_semantics.type(torch.int).cpu().numpy()  #(10475, 0)
        print(np.unique(x_semantics))
        colors = category_dict['color'][x_semantics].to_numpy()
        print(colors)
        colors = np.asarray([np.asarray(color) for color in colors])
        print(colors)
        body = trimesh.Trimesh(vertices=vertices[0].detach().cpu().numpy(), faces=body_model_dict['neutral'].faces,
                               vertex_colors=colors)
        body.show()
        scene_name = generation_param['scene']
        scene = trimesh.load_mesh(Path.joinpath(proxe_base_folder, 'Narrator_dir/scenes', scene_name + '.ply'))
        (body + scene).show()

    return

def calc_semantic_accuracy(vertices, scene_name, generation):
    vertices = vertices.squeeze().detach().cpu().numpy()
    atomic_generations = generation.split('+')
    scores = []
    for atomic in atomic_generations:
        verb, noun = atomic.split('-')
        if noun not in object_proximity_dict[scene_name]:
            objs = [obj for obj in scenes[scene_name].object_nodes if obj.category_name == noun]
            if len(objs) == 0:
                print(noun, scene_name)
                scores.append(0)
                continue
            points = np.concatenate([np.asarray(obj.mesh.vertices) for obj in objs], axis=0)
            # print(points.shape)
            object_proximity_dict[scene_name][noun] = KDTree(points)

        body_parts = action_body_part_mapping[verb]
        vertices_of_interest = []
        for body_part in body_parts:
            vertices_of_interest += body_part_vertices[body_part]
        vertices_of_interest = np.asarray(vertices_of_interest)
        dists, idx = object_proximity_dict[scene_name][noun].query(vertices[vertices_of_interest])
        contact_thresh_dict = {
            'sit on': 0.1,
            'stand on': 0.05,
            'lie on': 0.1,
            'touch': 0.1,
        }
        contact_thresh = contact_thresh_dict[verb]
        dist = dists.min() if verb == 'touch' else dists.mean()
        score = 1.0 if dist < contact_thresh else 0.0
        scores.append(score)

        if DEBUG:
            contact_vertices = vertices_of_interest[dists < contact_thresh]
            colors = np.ones((10475, 3)) * np.array([1.00, 0.75, 0.80])
            colors[contact_vertices, :] = np.array([1.00, 0.0, 0.0])
            body = trimesh.Trimesh(vertices=vertices, faces=body_model_dict['neutral'].faces,
                                   vertex_colors=colors)
            body.show()
            scene = trimesh.load_mesh(Path.joinpath(scene_folder, scene_name + '.ply'))
            (body + scene).show()


    return np.asarray(scores).mean()

def calc_semantic_contact(vertices, scene_name, generation, object_combination):
    vertices = body_mesh.downsample(vertices)
    atomic_generations = object_combination.split('+')
    verbs = '+'.join([atomic.split('-')[0] for atomic in atomic_generations])
    instance_idx = [int(atomic.split('-')[-1]) for atomic in atomic_generations]
    scene = scenes[scene_name]
    # print(instance_idx)
    object_points = [np.asarray(scene.get_mesh_with_accessory(node_idx).vertices) for node_idx in instance_idx]
    object_points = torch.from_numpy(np.concatenate(object_points)).unsqueeze(0).float().to(vertices.device)
    body_obj_dists = torch.sqrt(chamfer_dists(vertices, object_points)).squeeze().detach().cpu().numpy()

    contact_probability, contact_score_thresh = contact_statistics['probability'][verbs], contact_statistics['score'][verbs]
    contact_thresh = 0.05
    contact_score = np.sum((body_obj_dists < contact_thresh) * contact_probability)
    contact_score = (contact_score >= contact_score_thresh * 0.8)
    # contact_score = min(contact_score / contact_score_thresh, 1) if contact_score_thresh > 0 else 1
    return contact_score

def evaluate_semantic(results):
    semantic_metric = {'all': []}

    for generation in tqdm(results):
        semantic_scores = []
        for generation_param in results[generation]:
            vertices = get_vertices(generation_param, scene_coords='PROX')
            # semantic_score = calc_semantic_accuracy(vertices, generation_param['scene'], generation_param['generation'])
            semantic_score = calc_semantic_contact(vertices, generation_param['scene'],
                                                    generation_param['generation'], generation_param['object_combination'])
            semantic_scores.append(semantic_score)

        semantic_metric[generation] = semantic_scores
        semantic_metric['all'] += semantic_scores

    for key in semantic_metric.keys():
        semantic_metric[key] = float(np.asarray(semantic_metric[key]).mean())

    return semantic_metric

def evaluate_results(results, method='default'):
    metrics = {}
    metrics['diversity'] = evaluate_diversity(results, method=method)
    metrics['contact'], metrics['non_collision'] = evaluate_physical_plausibility(results)
    metrics['semantic'] = evaluate_semantic(results)
    return metrics

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_mesh = Mesh(num_downsampling=2)
    # load contact statistics
    with open(project_folder / 'data' / 'contact_statistics.json', 'r') as f:
        contact_statistics = json.load(f)
    body_model_dict = {
        'male': smplx.create(smplx_model_folder, model_type='smplx',
                             gender='male', ext='npz',
                             num_pca_comps=num_pca_comps).to(device),
        'female': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='female', ext='npz',
                               num_pca_comps=num_pca_comps).to(device),
        'neutral': smplx.create(smplx_model_folder, model_type='smplx',
                                gender='neutral', ext='npz',
                                num_pca_comps=num_pca_comps).to(device)
    }

    # load scenes using util function
    scene_data_dict = {}
    for scene_name in scene_names:
        scene_data_dict[scene_name] = load_scene_data(name=scene_name,
                                                      sdf_dir=Path.joinpath(sdf_folder, 'sdf').__str__(),
                                                      use_semantics=True,
                                                      no_obj_classes=42,
                                                      device=device
                                                      )
    # trimesh proximity of objects of specified category
    object_proximity_dict = {}
    for scene_name in test_scenes:
        object_proximity_dict[scene_name] = {}

    if DEBUG:
        generation_param = synthesis_results_dict['prox']['sit on-chair'][0]
        vertices = get_vertices(generation_param)
        calc_semantic_accuracy(vertices, generation_param['scene'], 'sit on-chair')

    # evaluate metrics for each interation semantics
    metrics = {}
    for method in tqdm(synthesis_results_dict):
        print('evaluate metrics for:', method)
        metrics[method] = evaluate_results(synthesis_results_dict[method], method)
    print(metrics)
    with open(Path.joinpath(results_folder, 'metrics.json'), 'w') as file:
        json.dump(metrics, file)

    # evaluate overall metrics for all interation frames
    metrics_overview = copy(metrics)
    for method in metrics_overview:
        for metric in metrics_overview[method]:
            metrics_overview[method][metric] = metrics_overview[method][metric]['all']
    with open(Path.joinpath(results_folder, 'metrics_overview.json'), 'w') as file:
        json.dump(metrics_overview, file)
