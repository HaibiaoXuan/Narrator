import sys
sys.path.append('..')

import glob
import pickle
from collections import defaultdict

from configuration.config import *
from data.hand_pca_transform import pose_to_pca
from data.load_generation import get_generation_segments

"""Load results generated by pigraph"""
def load_pigraph(result_dir):
    results = defaultdict(list)

    for generation_dir in result_dir.iterdir():
        if generation_dir.is_dir():
            generation = generation_dir.name
            for scene_dir in generation_dir.iterdir():
                scene_name = scene_dir.name
                if scene_dir.is_dir():
                    for result_file in scene_dir.iterdir():
                        if result_file.name[-4:] == '.pkl':
                            combination_name = result_file.name[:-4]
                            with result_file.open('rb') as pkl_file:
                                smplx_params = pickle.load(pkl_file)
                                T = len(smplx_params['transl'])
                                for idx in range(T):
                                    generation_param = {'scene': scene_name, 'generation': generation,
                                                         'gender': 'neutral', 'object_combination': combination_name}
                                    for key in smplx_params:
                                        if key in smplx_param_names:
                                            generation_param[key] = smplx_params[key][[idx]].cpu()
                                    generation_param['left_hand_pose'], generation_param['right_hand_pose'] = \
                                        pose_to_pca(generation_param['left_hand_pose'], generation_param['right_hand_pose'], gender=generation_param['gender'])

                                    results[scene_name + '_' + combination_name].append(generation_param)
    # print(results.keys())
    return results

"""Load results generated using POSA"""
def load_posa(result_dir):
    results = defaultdict(list)

    for scene_dir in result_dir.iterdir():
        if scene_dir.is_dir():
            scene_name = scene_dir.name
            for result_file in scene_dir.iterdir():
                if result_file.name[-4:] == '.pkl':
                    generation = result_file.name.split('.')[0]
                    with result_file.open('rb') as pkl_file:
                        smplx_params = pickle.load(pkl_file)
                        T = len(smplx_params)
                        for idx in range(T):
                            generation_param = {'scene': scene_name}
                            for key in smplx_params[idx]:
                                if key in smplx_param_names:
                                    generation_param[key] = smplx_params[idx][key]
                            results[generation].append(generation_param)
    print(results.keys())
    return results

""" Load pseudo ground truth PROX generation data from test set"""
def load_prox():
    with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
        test_data = pickle.load(data_file)
    results = defaultdict(list)
    for generation in generation_names:
        generation_data = get_generation_segments(generation.split('+'), test_data, mode='verb-noun')
        for record in generation_data:
            # scene_name, sequence, frame_idx, smplx_param, generation_labels, generation_obj_idx = record
            scene_name = record['scene_name']
            atomics = generation.split('+')
            # verbs = [atomic.split('-')[0] for atomic in atomics]
            # nouns = [atomic.split('-')[1] for atomic in atomics]
            obj_ids = [record['generation_obj_idx'][record['generation_labels'].index(atomic)] for atomic in atomics]
            combination_name = '+'.join([atomics[atomic_idx] + '-' + str(obj_ids[atomic_idx]) for atomic_idx in range(len(atomics))])
            wrong_combination = ['MPH1Library_sit down-chair-5', 'MPH1Library_step up-chair-6', 'MPH1Library_stand up-chair-6', 'MPH1Library_stand up-chair-5', 'MPH1Library_step down-chair-8']
            if (scene_name + '_' + combination_name) in wrong_combination:  # filter wrong records
                continue
            generation_param = {'scene': scene_name, 'generation': generation,
                                 'object_combination': combination_name}
            generation_param.update(record['smplx_param'])
            if not 'gender' in generation_param:
                generation_param['gender'] = 'neutral'
            generation_param['left_hand_pose'] = generation_param['left_hand_pose'][:, :num_pca_comps]
            generation_param['right_hand_pose'] = generation_param['right_hand_pose'][:, :num_pca_comps]
            results[scene_name + '_' + combination_name].append(generation_param)

    print(results.keys())
    return results


def load_coins():
    with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
        test_data = pickle.load(data_file)
    results = defaultdict(list)
    for generation in generation_names:
        generation_data = get_generation_segments(generation.split('+'), test_data, mode='verb-noun')
        for record in generation_data:
            scene_name = record['scene_name']
            atomics = generation.split('+')
            obj_ids = [record['generation_obj_idx'][record['generation_labels'].index(atomic)] for atomic in atomics]
            combination_name = '+'.join([atomics[atomic_idx] + '-' + str(obj_ids[atomic_idx]) for atomic_idx in range(len(atomics))])
            wrong_combination = ['MPH1Library_sit down-chair-5', 'MPH1Library_step up-chair-6', 'MPH1Library_stand up-chair-6', 'MPH1Library_stand up-chair-5', 'MPH1Library_step down-chair-8']
            if (scene_name + '_' + combination_name) in wrong_combination:  # filter wrong records
                continue
            generation_param = {'scene': scene_name, 'generation': generation,
                                 'object_combination': combination_name}
            generation_param.update(record['smplx_param'])
            if not 'gender' in generation_param:
                generation_param['gender'] = 'neutral'
            generation_param['left_hand_pose'] = generation_param['left_hand_pose'][:, :num_pca_comps]
            generation_param['right_hand_pose'] = generation_param['right_hand_pose'][:, :num_pca_comps]
            results[scene_name + '_' + combination_name].append(generation_param)

    print(results.keys())
    return results

"""Load results generated by our method."""
def load_results(result_dir):
    results = defaultdict(list)

    for generation_dir in result_dir.iterdir():
        if generation_dir.is_dir():
            generation = generation_dir.name
            for scene_dir in generation_dir.iterdir():
                scene_name = scene_dir.name
                if scene_dir.is_dir():
                    for result_file in scene_dir.iterdir():
                        if result_file.name[-4:] == '.pkl':
                            combination_name = result_file.name[:-4]
                            with result_file.open('rb') as pkl_file:
                                smplx_params = pickle.load(pkl_file)
                                T = len(smplx_params)
                                for idx in range(T):
                                    generation_param = {'scene': scene_name, 'generation': generation,
                                                         'gender': 'neutral', 'object_combination': combination_name}
                                    for key in smplx_params[idx]:
                                        if key in smplx_param_names:
                                            generation_param[key] = smplx_params[idx][key]
                                    results[scene_name + '_' + combination_name].append(generation_param)
    # print(results.keys())
    return results

# dict of generation results from different sources. Used in render_results.py and eval_results.py
synthesis_results_dict = {
    'prox': load_prox(),
    'pigraph_no_penetration': load_pigraph(Path('./scene_graph/results') / 'pigraph_normal'),
    'POSA_best1': load_posa(Path('./scene_graph/results') /'POSA_IPoser_best1'),
    'COINS': load_coins(Path('/home/kaizhao/projects/scene_graph/results') / 'two_stage' / 'floor_eval_try1_pene20_noseed_lr0.01' / 'optimization_after_get_body'),
	'Narrator': load_results(Path('./scene_graph/results') /'best_results')
}
