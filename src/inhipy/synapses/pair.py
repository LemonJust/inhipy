# from inhipy.synapses.classify import run_classification
from pathlib import Path
import pandas as pd
from coolname import generate_slug
import yaml
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from inhipy.utils.config import initialize_step_run, get_basedir


# from synspy> analyze > pair :
def nearest_pairs(v1, kdt1, v2, radius, out1, out2):
    """Find nearest k-dimensional point pairs between v1 and v2 and return via output arrays.

       Inputs:
         v1: array with first pointcloud with shape (m, k)
         kdt1: must be cKDTree(v1) for correct function
         v2: array with second pointcloud with shape (m, k)
         radius: maximum euclidean distance between points in a pair
         out1: output adjacency matrix of shape (n,)
         out2: output adjacency matrix of shape (m,)

       Use greedy algorithm to assign nearest neighbors without
       duplication of any point in more than one pair.

       Outputs:
         out1: for each point in kdt1, gives index of paired point from v2 or -1
         out2: for each point in v2, gives index of paired point from v1 or -1
    """
    # set depth to be the minimum of the number of points or 100 ( arbitrary)
    depth = min(max(out1.shape[0], out2.shape[0]), 100)
    out1[:] = -1
    out2[:] = -1

    dx, pairs_nn = kdt1.query(v2, depth, distance_upper_bound=radius)

    # arrays need to be 2D :
    if pairs_nn.shape == (1,):
        pairs_nn = pairs_nn[:, None]
    if dx.shape == (1,):
        dx = dx[:, None]
    if out1.shape == (1,):
        out1 = out1[:, None]
    if out2.shape == (1,):
        out2 = out2[:, None]

    # search for nearest neighbors in both directions :
    for d in range(depth):
        for idx2 in np.argsort(dx[:, d]):
            if dx[idx2, d] < radius:
                if out2[idx2] == -1 and out1[pairs_nn[idx2, d]] == -1:
                    out2[idx2] = pairs_nn[idx2, d]
                    out1[pairs_nn[idx2, d]] = idx2


def pair_centroids(centroids1, centroids2, radius_seq):
    """
    Find nearest k-dimensional point pairs between v1 and v2 and return via output arrays.

    Args:
        centroids1: array with first pointcloud with shape (m, k)
        centroids2: array with second pointcloud with shape (m, k)
        radius_seq: maximum euclidean distance between points in a pair, can be int or a list of values

    Returns:
        v1_to_v2: for each point in centroids1, gives index of paired point from centroids2 or -1
        v2_to_v1: for each point in centroids2, gives index of paired point from centroids1 or -1
    """
    ve1 = centroids1  # tp1
    ve2 = centroids2  # tp2

    kdt1 = cKDTree(ve1)
    v1_to_v2 = np.zeros((len(radius_seq), ve1.shape[0]), dtype=np.int32)
    v2_to_v1 = np.zeros((len(radius_seq), ve2.shape[0]), dtype=np.int32)
    # print(f"v1_to_v2: {v1_to_v2.shape}")
    # print(f"v2_to_v1: {v2_to_v1.shape}")

    for r_idx in range(len(radius_seq)):
        nearest_pairs(ve1, kdt1, ve2, radius_seq[r_idx], v1_to_v2[r_idx, :], v2_to_v1[r_idx, :])

    return v1_to_v2, v2_to_v1


# # plotting functions
# def pair_summary(v1_to_v2, title_intro):
#     paired = np.zeros((len(radius_seq),), dtype=np.int32)
#     n_total = reg[study_id]['tp1']['xyz'].shape[0]
#     for r_idx in range(len(radius_seq)):
#         paired[r_idx] = np.sum(v1_to_v2[r_idx, :] != -1)
#     unpaired = n_total - paired

#     paired = paired * 100 / n_total
#     unpaired = unpaired * 100 / n_total

#     print(f'radius :{radius_seq} \npaired %:\n{paired} \nunpaired %:\n{unpaired}')

#     fig = plt.figure(dpi=160)
#     plt.plot(radius_seq, paired)
#     plt.vlines(4, 0, 100, colors='r')
#     plt.grid()
#     plt.title(f'{title_intro}\nNumber of paired synapses')
#     plt.xlabel('Deforestation radius, um')
#     plt.ylabel('Synapses paired, %')

def get_synapse_centroids(centroids_config):
    """
    """
    for tp in ['tp1', 'tp2']:
        # get files with synapse centroids
        basedir = get_basedir(centroids_config[tp]['basedir'])
        file = basedir / centroids_config[tp]['file']

        # read tp1 and tp2 csvs into dataframes
        if tp == 'tp1':
            tp1_df = pd.read_csv(file)
        elif tp == 'tp2':
            tp2_df = pd.read_csv(file)

    # get centroids of synapses only
    tp1_df = tp1_df[tp1_df['class'] == 1]
    tp2_df = tp2_df[tp2_df['class'] == 1]

    # create a copy and convert to physical space
    centroids1 = tp1_df[['z','y','x']].copy()
    centroids2 = tp2_df[['z','y','x']].copy()

    for tp in ['tp1', 'tp2']:
        if 'spacing' in centroids_config[tp]:
            spacing = {key:value for key, value in zip('xyz',centroids_config[tp]['spacing'])}
            if tp == 'tp1':
                centroids1[['z','y','x']] = centroids1[['z','y','x']].multiply(spacing)
            elif tp == 'tp2':
                centroids2[['z','y','x']] = centroids2[['z','y','x']].multiply(spacing)
    return centroids1, centroids2, tp1_df, tp2_df

def run_euclidean_distance(pairing_config, verbose=True):
    """
    """
    # load input data
    centroids1, centroids2, tp1_df, tp2_df = get_synapse_centroids(pairing_config['input']['synapse_centroids'])

    # process for each distance threshold
    distances = pairing_config['parameters']['distance_thr']['d']
    if isinstance(distances, int):
        distances = [distances]

    for d in distances:
        # TODO : use dustances in pair and loop on v1_to_v2 later
        # get the pairing
        v1_to_v2, v2_to_v1 = pair_centroids(centroids1.values, centroids2.values, [d])

        # create a dataframe with paired synapses
        paired = pd.DataFrame()
        is_paired = v1_to_v2[0] != -1
        paired['id1'] = tp1_df['id'][is_paired]
        idx2 = v1_to_v2.flatten()[is_paired]
        paired['id2'] = tp2_df['id'][idx2].values
        paired['x1'] = tp1_df['x'][is_paired].values
        paired['y1'] = tp1_df['y'][is_paired].values
        paired['z1'] = tp1_df['z'][is_paired].values
        paired['x2'] = tp2_df['x'][idx2].values
        paired['y2'] = tp2_df['y'][idx2].values
        paired['z2'] = tp2_df['z'][idx2].values
        paired['prob1'] = tp1_df['prob'][is_paired].values
        paired['prob2'] = tp2_df['prob'][idx2].values
        # TODO: add distance (will need to modify pair_centroids to return it)
        # paired['distance'] = ...
        
        # create dataframe with lost synapses
        lost = pd.DataFrame()
        is_lost = v1_to_v2[0] == -1
        lost['id1'] = tp1_df['id'][is_lost]
        lost['x'] = tp1_df['x'][is_lost].values
        lost['y'] = tp1_df['y'][is_lost].values
        lost['z'] = tp1_df['z'][is_lost].values
        lost['prob'] = tp1_df['prob'][is_lost].values

        # create dataframe with gained synapses
        gained = pd.DataFrame()
        is_gained = v2_to_v1[0] == -1
        gained['id2'] = tp2_df['id'][is_gained]
        gained['x'] = tp2_df['x'][is_gained].values
        gained['y'] = tp2_df['y'][is_gained].values
        gained['z'] = tp2_df['z'][is_gained].values
        gained['prob'] = tp2_df['prob'][is_gained].values

        # save the dataframes
        output_folder = Path(pairing_config['output_folder'])
        paired_file = pairing_config['output']['files']['paired'].format(EDT=d)
        paired.to_csv(output_folder / paired_file, index=False)

        lost_file = pairing_config['output']['files']['lost'].format(EDT=d)
        lost.to_csv(output_folder / lost_file, index=False)

        gained_file = pairing_config['output']['files']['gained'].format(EDT=d)
        gained.to_csv(output_folder / gained_file, index=False)

        if verbose:
            print(f"Saved pairing for to \n{output_folder}:\n{paired_file}\n{lost_file}\n{gained_file}")
            print("First 5 rows:")
            print('paired')
            print(paired.head())
            print('lost')
            print(lost.head())
            print('gained')
            print(gained.head())


def run_pairing(config, verbose = True):
    step = 'synapse_pairing'
    config = initialize_step_run(config, step, verbose = verbose)

    # run the classification
    method = config['method']
    if method == 'euclidean_distance':
        run_method = run_euclidean_distance
    else:
        raise ValueError('Unknown synapse pairing method: {}'.format(method))
    
    run_method(config, verbose=verbose)


