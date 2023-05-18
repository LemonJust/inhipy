import ants 
import yaml
import numpy as np
from pathlib import Path
import tifffile as tif
import matplotlib.pyplot as plt
import logging
# from mpl_toolkits.mplot3d import Axes3D
from inhipy.registration.utils import read_image, save_image, save_image_as_uint16

def ants_registration(fixed_image, moving_image, registration_type='rigid'):
    if registration_type not in ['rigid', 'affine', 'non_rigid']:
        raise ValueError(f"Invalid registration type: {registration_type}")

    if registration_type == 'non_rigid':
        transform_type = 'SyN'
    else:
        transform_type = registration_type

    registration_results = ants.registration(fixed=fixed_image,
                                             moving=moving_image,
                                             type_of_transform=transform_type)
    moving_image_registered = registration_results['warpedmovout']

    return moving_image_registered

def compare_registration_3d(image_before, image_after):
    """
    A function to compare the result of 3D registration of two volumetric images.

    Parameters:
    -----------
    image_before: numpy array
        The 3D array representing the image before registration.
    image_after: numpy array
        The 3D array representing the image after registration.

    Returns:
    --------
    None
    """

    # Create a figure with three subplots
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Set the axes labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Plot the image before registration
    X, Y, Z = np.meshgrid(np.arange(image_before.shape[0]), np.arange(image_before.shape[1]), np.arange(image_before.shape[2]), indexing='ij')
    ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=image_before.flatten(), cmap='gray')

    # Plot the image after registration
    X, Y, Z = np.meshgrid(np.arange(image_after.shape[0]), np.arange(image_after.shape[1]), np.arange(image_after.shape[2]), indexing='ij')
    ax2.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=image_after.flatten(), cmap='gray')

    # Calculate and plot the difference between the two images
    image_diff = image_after - image_before
    X, Y, Z = np.meshgrid(np.arange(image_diff.shape[0]), np.arange(image_diff.shape[1]), np.arange(image_diff.shape[2]), indexing='ij')
    ax3.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=image_diff.flatten(), cmap='gray')

    # Set the titles of the subplots
    ax1.set_title('Before Registration')
    ax2.set_title('After Registration')
    ax3.set_title('Difference')

    # Show the plot
    plt.show()

def compare_registration_slices(fixed_image, moving_image_before, moving_image_after, 
                                titles = ('Moving Image Before Registration','Fixed Image','Moving Image After Registration')):
    """
    A function to compare the result of 3D registration of two volumetric images.

    Parameters:
    -----------
    fixed_image: numpy array
        The 3D array representing the fixed image.
    moving_image_before: numpy array
        The 3D array representing the moving image before registration.
    moving_image_after: numpy array
        The 3D array representing the moving image after registration.

    Returns:
    --------
    None
    """

    # Select some random slices to compare
    slice_indices = np.random.choice(fixed_image.shape[2], size=3, replace=False)

    # Create a figure with three subplots
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

    # Set the titles of the subplots
    axs[0, 0].set_title(titles[0])
    axs[0, 1].set_title(titles[1])
    axs[0, 2].set_title(titles[2])

    # Plot the selected slices
    for i, index in enumerate(slice_indices):
        # Plot the fixed image
        axs[i, 0].imshow(moving_image_before[:, :, index], cmap='gray')
        axs[i, 0].axis('off')

        # Plot the moving image before registration
        axs[i, 1].imshow(fixed_image[:, :, index], cmap='gray')
        axs[i, 1].axis('off')

        # Plot the moving image after registration
        axs[i, 2].imshow(moving_image_after[:, :, index], cmap='gray')
        axs[i, 2].axis('off')

    # Show the plot
    plt.show()

def run_registration(config, verbose = True):

    fixed_file = Path(config['data']['basedir']) / config['data']['images']['fixed']['image_file']
    moving_file = Path(config['data']['basedir']) / config['data']['images']['moving']['image_file']

    fixed_image = tif.imread(fixed_file).astype(np.float32) # ZCYX format
    moving_image = tif.imread(moving_file).astype(np.float32) 

    # select the channel if the image is 4D
    if len(fixed_image.shape) ==4: 
        channel = config['data']['images']['fixed']['channel']
        fixed_image = fixed_image[:, channel, :, :]
    if len(moving_image.shape) ==4:
        channel = config['data']['images']['moving']['channel']
        moving_image = moving_image[:, channel, :, :]
    
    fixed_image = ants.from_numpy(np.transpose(fixed_image, axes=[2,1,0]), 
                                spacing=config['data']['images']['fixed']['spacing'])
    moving_image = ants.from_numpy(np.transpose(moving_image, axes=[2,1,0]), 
                                spacing=config['data']['images']['moving']['spacing'])

    if verbose:
        print("Fixed image:")
        print(fixed_image)
        print("Moving image:")
        print(moving_image)

    # prepare the output directory
    output_dir = Path(config['output']['basedir']) / config['step'] / config['name']
    output_dir.mkdir(parents=True, exist_ok=True)

    # save config to the folder 
    with open(output_dir/"config.yaml", 'w') as f:
        yaml.dump(config, f)

    outprefix = output_dir / config['output']['prefix']

    # run the registration and save verbowe to file
    rr = ants.registration(
                            fixed=fixed_image,
                            moving=moving_image,
                            type_of_transform=config['parameters']['type_of_transform'], 
                            verbose  = verbose,
                            outprefix = outprefix.as_posix(),
                        )
    registered_image = rr['warpedmovout']

    print('\nfwdtransforms')
    print(rr['fwdtransforms'])

    print('\ninvtransforms')
    print(rr['invtransforms'])

    save_image(registered_image, outprefix.as_posix() + "_warpedmovout.nii.gz")
