import ants 
import yaml
import pandas as pd 
import numpy as np
from pathlib import Path
import tifffile as tif
import matplotlib.pyplot as plt

from inhipy.registration.utils import read_image, save_image, save_image_as_uint16

def load_transform_list(config, verbose = False):
    transformlist = []
    for t_file in config['files']:
        transform_file = (Path(config['basedir']) / 'registration' / 
                                            config['name'] / t_file)
        transformlist.append(transform_file.as_posix())

    if verbose:
        print('Transform List')
        print(transformlist)

    return transformlist

def load_reference_image(config, verbose = False):
    ref_file = Path(config['basedir']) / config['file']
    ref_image = tif.imread(ref_file).astype(np.float32) # ZCYX format
    if len(ref_image.shape) ==4: 
        channel = config['channel']
        ref_image = ref_image[:, channel, :, :]
    ref_image = ants.from_numpy(np.transpose(ref_image, axes=[2,1,0]), spacing=config['spacing'])

    if verbose:
        print('Reference Image')
        print(ref_image)

    return ref_image

def transform_images(transformlist, ref_image, config, output_dir, verbose = False):
    # load the reference image
    ref_image = load_reference_image(ref_image, verbose = verbose)

    for img in config:

        moving_file = Path(img['basedir']) / img['file']
        moving_image = tif.imread(moving_file).astype(np.float32) # ZCYX format

        if len(moving_image.shape) ==4: 
            channel = img['channel']
            moving_image = moving_image[:, channel, :, :]

        moving_image = ants.from_numpy(np.transpose(moving_image, axes=[2,1,0]), spacing=img['spacing'])

        # apply the transform
        transformed_image = ants.apply_transforms(
                                                    fixed=ref_image,
                                                    moving=moving_image,
                                                    transformlist=transformlist,
                                                    interpolator=img['interpolation'],
                                                    verbose=verbose,
                                                )
        # save the transformed image
        output_file = (output_dir / img['output file'])
        save_image_as_uint16(transformed_image.numpy().transpose((2, 1, 0)), output_file)
        if verbose:
            print(f'Saved transformed {img} image to: \n{output_file}')

def transform_points(transformlist, config, output_dir, verbose = False):

    for points_config in config:

        # load the points
        points_file = Path(points_config['basedir']) / points_config['point_file']
        points = pd.read_csv(points_file)

        # convert to physical space
        if 'spacing' in points_config:
            spacing = {key:value for key, value in zip('xyz',points_config['spacing'])}
            points[['z','y','x']] = points[['z','y','x']].multiply(spacing)

        # apply the transform
        n_dim = 3
        transformed_points = ants.apply_transforms_to_points(n_dim, points, transformlist,
                                                             whichtoinvert =[1,0], verbose=verbose)
        
        # convert back to voxel space
        if 'spacing' in points_config:
            transformed_points[['z','y','x']] = transformed_points[['z','y','x']].divide(spacing)

        # save the transformed points
        output_file = (output_dir / points_config['output file'])
        transformed_points.to_csv(output_file, index=False)
        if verbose:
            print(f'Saved transformed points to: \n{output_file}')


def run_transformation(config, verbose = False):

    # prepare the output directory
    output_dir = (Path(config['output']['basedir']) / config['step'] / config['name'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # save config to the folder
    with open(output_dir/"config.yaml", 'w') as f:
        yaml.dump(config, f)

    # generate the transform list
    transformlist = load_transform_list(config['registration'], verbose = verbose)
    if verbose:
        print(f'Applying transfrms: {transformlist}')

    # apply the transform to the moving images in a loop
    if 'images to transform' in config:
        transform_images(transformlist, config['reference image'], 
                         config['images to transform'], 
                         output_dir, verbose = verbose)
        
    # apply the transform to the points in a loop
    if 'points to transform' in config:
        transform_points(transformlist, config['points to transform'], 
                         output_dir, verbose = verbose)
    if verbose:
        print('Finished transformation')
        print('----------------------')
    


