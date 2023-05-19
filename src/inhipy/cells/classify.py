import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
from pathlib import Path
from tqdm import tqdm
from inhipy.utils.config import generate_output_folder

def get_cell_statistics(seg, gfp, hucd, dapi):
    # Get unique cell labels
    labels = np.unique(seg)

    avg_intensity_gfp = []
    avg_intensity_hucd = []
    avg_intensity_dapi = []
    cell_sizes = []

    for label in tqdm(labels):
        cell_pixels = seg == label

        cell_sizes.append(np.sum(cell_pixels))

        gfp_intensity = np.mean(gfp[cell_pixels])
        hucd_intensity = np.mean(hucd[cell_pixels])
        dapi_intensity = np.mean(dapi[cell_pixels])
        
        avg_intensity_gfp.append(gfp_intensity)
        avg_intensity_hucd.append(hucd_intensity)
        avg_intensity_dapi.append(dapi_intensity)

    df = pd.DataFrame({'cell_id': labels, 'cell_size': cell_sizes, 'avg_intensity_gfp': avg_intensity_gfp, 'avg_intensity_hucd': avg_intensity_hucd, 'avg_intensity_dapi': avg_intensity_dapi})
    return df
    


def run_get_cell_stats(config):
    seg_cells = config['input']['segmentation image']
    all_cells = config['input']['fixed multichannel image']

    gfp_channel = config['input']['channels']['gfp']
    hucd_channel = config['input']['channels']['hucd']
    dapi_channel = config['input']['channels']['dapi']

    # Load 3D images
    seg = tif.imread(seg_cells)
    _all = tif.imread(all_cells)
    gfp = np.squeeze(_all[:,gfp_channel,:,:])
    hucd = np.squeeze(_all[:,hucd_channel,:,:])
    dapi = np.squeeze(_all[:,dapi_channel,:,:])

    # Get cell statistics
    df = get_cell_statistics(seg, gfp, hucd, dapi)


    # create the output folder
    config = generate_output_folder(config, 'nuclei_classification')
    df.to_csv(Path(config['output_folder'])/ config['output']['file'], index = False)