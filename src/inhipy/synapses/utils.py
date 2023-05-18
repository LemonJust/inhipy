import json
import os
import warnings
from pathlib import Path
import numpy as np
import tifffile as tif
from patchify import patchify, unpatchify
from inhipy.synspy.analyze.util import load_segment_status_from_csv


def _centroids_from_npz(npz_filename):
    """
    Loads centroids from npz and shifts it by slice origin to place it into SPIM voxel coords
    """
    parts = np.load(npz_filename)
    centroids = parts['centroids'].astype(np.int32)

    props = json.loads(parts['properties'].tostring().decode('utf8'))
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)

    # convert cropped centroids back to full SPIM voxel coords
    return centroids + slice_origin


def _centroids_from_csv():
    """
    Loads centroids from csv
    """
    pass


def _labels_from_npz_and_csv(npz_filename, csv_filename):
    """
    Loads labels from npz and csv
    """
    parts = np.load(npz_filename)
    centroids = parts['centroids'].astype(np.int32)
    props = json.loads(parts['properties'].tostring().decode('utf8'))
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)
    statuses, _ = load_segment_status_from_csv(centroids, slice_origin, csv_filename)

    # interpret status flag values
    is_synapse = (statuses == 7).astype(bool)

    return is_synapse


def load_centroids(npz_filename):
    """
    Loads centroids from npz or csv
    """
    return _centroids_from_npz(npz_filename)


def load_labels(npz_filename, csv_filename):
    """
    Loads labels from npz and csv
    """
    return _labels_from_npz_and_csv(npz_filename, csv_filename)


def drop_unsegmented(centroids, labels, x_max=500, z_max=90):
    """
    Gets rid of centroids and labels in the unsegmented area.
    Keeping only stuff above Z = 90 and on the left side of X = 500.
    Currently that's what we have unsegmented for Gad1b::GFP sometimes.

    Parameters:
    x_max,z_max (int) : max x and z values (in pixels) to keep
    """
    z_coord = centroids[:, 0]
    x_coord = centroids[:, 2]

    in_z = z_coord <= z_max
    in_x = x_coord <= x_max
    in_roi = np.logical_and(in_z, in_x)

    centroids = centroids[in_roi, :]
    labels = labels[in_roi]

    return centroids, labels

def drop_unsegmented_image(img, centroids, padz = 2, padxy = 0):
    """ Crops image to fit the area with segmented synapses"""
    # 1. find min and max among segmented synapses
    zmin, ymin, xmin = np.min(centroids, axis = 0)
    zmax, ymax, xmax = np.max(centroids, axis = 0)
    print(f"Segmented synapses from {zmin, ymin, xmin} to {zmax, ymax, xmax}")
    # 2. crop image accordingly
    img = img[zmin-padz:zmax+padz,
          ymin-padxy:ymax+padxy,
          xmin-padxy:xmax+padxy]
    return img


def load_image(img_file):
    """
    Just a wrapper for tif.imread
    """
    img = tif.imread(img_file)
    return img


def get_filenames(data_dir, roi_id):

    data_folder = Path(f"{data_dir}/{roi_id}")
    img_files = [file for file in data_folder.glob("*.tiff")]
    npz_files = [file for file in data_folder.glob("*.npz")]
    csv_files = [file for file in data_folder.glob("*.csv")]

    assert len(img_files) == 1, f"Should be exactly 1 tiff file in {data_folder}, but found {len(img_files)}"
    assert len(npz_files) == 1, f"Should be exactly 1 npz file in {data_folder}, but found {len(npz_files)}"
    assert len(csv_files) == 1, f"Should be exactly 1 csv file in {data_folder}, but found {len(csv_files)}"

    return img_files[0], npz_files[0], csv_files[0]

def load_data(data_dir, roi_id):
    """
    Loads the centroids, labels and images from which we will crop
    """

    img_filename, npz_filename, csv_filename = get_filenames(data_dir, roi_id)

    centroids = load_centroids(npz_filename)
    labels = load_labels(npz_filename, csv_filename)

    warnings.warn("Assuming area with x>500 and z>90 is unsegmented. "
                  "Dropping all the centroids with x_max=500 z_max=90. ")
    centroids, labels = drop_unsegmented(centroids, labels, x_max=500, z_max=90)
    img = load_image(img_filename)

    return centroids, labels, img

def draw_balls(img, centroids, labels):
    """Draws balls around every synapse on the segmentation image ( the same size as img)
    centroids must be in pixels in the img. """
    # draw ball around centroid
    ball = np.array([[[0,0,0,0,0],
                      [0,1,1,1,0],
                      [0,1,1,1,0],
                      [0,1,1,1,0],
                      [0,0,0,0,0]], # slice -1
                     [[0,1,1,1,0],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [0,1,1,1,0]], # slice 0
                     [[0,0,0,0,0],
                      [0,1,1,1,0],
                      [0,1,1,1,0],
                      [0,1,1,1,0],
                      [0,0,0,0,0]]] # slice +1
                    )

    segmentation = np.zeros_like(img)

    for i_syn, synapse in enumerate(centroids[labels]):
        z, y, x = synapse
        segmentation[z-1:z+2,y-2:y+3, x-2:x+3] = ball*i_syn

    return segmentation


def save_patches(patches,save_folder,file_tag = 'ptch', min_masked = 39*3, save = None):
    """
    Save patches to the folder. The patches are saved in the format of
    ``{tag}_{ii}_{jj}_{kk}.tif``.

    Args:
        patches : Patches to save. Shape: (nz,ny,nx, *patch_size).
        save_folder : Folder to save the patches. If the folder does not exist,
            it will be created.
        file_tag : Tag for the filenames.
            If a list is given, the tag will be joined by '_'.
        min_masked: will save only such patches where sum of masked pixes is above the threshold
            Default is 3 balls.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # add patch shape to the tag
    if isinstance(file_tag, str):
        file_tag = [file_tag]
    file_tag.append(f'patch_shape_{patches.shape[0:3]}')
    tag = '_'.join(file_tag)

    def write_to_disk():
        file_name = Path(save_folder,'_'.join(file_tag) + f'_{ii}_{jj}_{kk}')
        tif.imwrite(file_name.with_suffix('.tif'),patches[ii, jj, kk])

    # save the patches
    saved = np.zeros((patches.shape[0:3]))
    print(f"saved : {saved.shape}")
    for ii in range(patches.shape[0]):
        for jj in range(patches.shape[1]):
            for kk in range(patches.shape[2]):
                if save is None:
                    if np.sum(patches[ii, jj, kk]>0)>min_masked:
                        write_to_disk()
                        saved[ii, jj, kk] = 1
                else:
                    assert save.shape == patches.shape[0:3], \
                        f"Save shape {save.shape} doesn't match the number of patches {patches.shape[0:3]}"
                    if save[ii, jj, kk] == 1:
                        write_to_disk()
                        saved[ii, jj, kk] = 1

    print(f"saved : {saved.shape}")
    return saved


def patchify_and_save(img,patch_size,save_folder,file_tag = 'ptch',step = None, save = None):
    """
    Patchify the image and save to folder.
    Automatically adds image shape to the file_tag.

    Args:
        img : Image to patchify as numpy array.
        patch_size : Size of the patches in 3D, int.
        step : Step size of the patches, int.
        file_tag : Tag for the filenames, str.
            If a list is given, the tag will be joined by '_'.
        save_folder : Folder to save the patches.
            If the folder does not exist, it will be created.
    """
    if step is None:
        step = patch_size

    # add image_shape to the file_tag
    if isinstance(file_tag, str):
        file_tag = [file_tag]
    file_tag.append(f'img_shape_{img.shape}')

    patches = patchify(img, (patch_size, patch_size, patch_size), step=step)
    saved = save_patches(patches, save_folder, file_tag, save = save)
    return saved
