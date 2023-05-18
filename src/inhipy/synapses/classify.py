# from inhipy.synapses.classify import run_classification
from pathlib import Path
import pandas as pd
from coolname import generate_slug
import yaml


def run_thresholding(config, verbose=True):
    """
    Classifies synapses based on a threshold on the probability.
    Returns a dictionary with the dataframe for each time point. 
    The dataframe contains the following columns:
        - x: x coordinate of the synapse
        - y: y coordinate of the synapse
        - z: z coordinate of the synapse
        - prob: probability of the synapse
        - class: 0 if the synapse is below the threshold, 1 if it is above the threshold
    """
    # prepare the csv files
    info_files_dir = (
        Path(config['synapse_candidates']['basedir']) / 
            config['synapse_candidates']['step'] / 
            config['synapse_candidates']['name'] )
    
    #load and process each file
    labels = {}
    for tp in config['synapse_candidates']['info_files']:
        if verbose:
            print(f"Processing {tp}")

        #load file 
        tp_df = pd.read_csv(info_files_dir / config['synapse_candidates']['info_files'][tp])

        # threshold to get the labels and add them to the dataframe
        tp_df['class'] = (tp_df['prob'] > config['parameters']['prob_thr'][tp]).astype(int)

        # save labels
        labels[tp] = tp_df

    return labels

def run_classification(config, verbose = True):

    # prepare the output directory
    step = config.get('step', 'synapse_candidates')
    name = config.get('name', generate_slug(2)) # generate random name if not provided
    output_folder = Path(config['output']['basedir']) / step / name
    output_folder.mkdir(parents=True, exist_ok=True)

    # save config to the folder 
    with open(output_folder/"config.yaml", 'w') as f:
        yaml.dump(config, f)

    # run the classification
    if config['parameters']['method'] == 'thresholding':
        run_method = run_thresholding
    else:
        raise ValueError('Unknown synapse classification method: {}'.format(config['parameters']['method']))
    
    labels = run_method(config, verbose=verbose)

    # save the labels
    for tp in labels:
        output_file = output_folder / config['output']['info_files'][tp]
        labels[tp].to_csv(output_file, index=False)

        if verbose:
            print(f"Saved labelsfor {tp} to {output_file}")
            print("First 5 rows:")
            print(labels[tp].head())

