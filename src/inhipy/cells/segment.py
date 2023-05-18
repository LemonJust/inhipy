"""
This module contains the function and classes to segment the synapse from the confocal image.
A step by step description of the segmentation process is given in the 
02_Synapse_Stardist_Segment_Candidates.ipynb notebook.
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from coolname import generate_slug
import yaml

from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import StarDist3D

np.random.seed(6)
lbl_cmap = random_label_cmap()

import pandas as pd

from inhipy.utils.plotter import plot_slice_with_insert
from inhipy.segmentation.preprocess import plot_segmentation_example



def load_image(img_file, channel = None, plot=True):
    """
    Just a wrapper for tif.imread
    """
    img = imread(img_file)
    if plot:
        plot_slice_with_insert(img)

    if channel is not None:
        img = np.squeeze(img[:, channel, : , :])

    return img

def load_model_from_file(model_dir):
    """
    Loads the model from the model_dir
    """
    model = StarDist3D(None, name=model_dir.name, basedir=str(model_dir.parent))
    return model

def preprocess(img):
    """
    Preprocesses the image for segmentation. 
    Simple normalization of the image to 1-99.8 percentile range.
    """  
    # TODO: don't I normilize per patch when training? If so, I should do the same here.

    img = normalize(img, 1,99.8, axis=(0,1,2))
    return img

def segment(model, img, n_tiles=(1,4,4), prob_thresh = 0.1, return_predict=True, plot=True):
    """
    Segments the image with the given model.
    Probability threshold is set low by default to avoid false negatives. False positives will be filtered out later.

    Args:
        model: StarDist3D model
        img: 3D image
        n_tiles: number of tiles to use for prediction
        prob_thresh: probability threshold for the prediction
        return_predict: if True, returns the prediction details, set to True to get the probability image. 
        plot: if True, plots the segmentation result

    Returns:
        labels: a tuple 
            (3D numpy array of labeled image, 
            dict of details with the probability and center of each label among other things)
        details: a tuple (3D numpy array of probability image, 3Dxn_rays numpy array for distribution image). 
            None if return_predict is False.
    """
    if return_predict:
        labels, details = model.predict_instances(img, n_tiles=n_tiles, prob_thresh=prob_thresh, return_predict=True)
    else:
        labels = model.predict_instances(img, n_tiles=n_tiles, prob_thresh=prob_thresh)
        details = None

    if plot:
        plot_segmentation_example(img, labels, lbl_cmap)

    return labels, details

def save_image_as_uint16(img, out_file, scale = None):
    """
    Save the image as uint16 tiff file after scaling it by scale. 
    Scale is used to scale float values, such as probability, to the range of uint16.
    """
    if scale is not None:
        img = img * scale

    #assert all values can fint into the uint range
    assert np.max(img) < 65535, "some labels image values are too large for uint16"
    assert np.min(img) >= 0, "some labels image values are too small for uint16"

    img = img.astype(np.uint16)
    imwrite(out_file, img, imagej=True)

def save_info_as_csv(info, out_file):
    """
    Save the label info as csv file.
    """
    # crete candidate dataframe with ID (row id) , zyx coord, prob
    df = pd.DataFrame(info['points'], columns = ['z', 'y', 'x'])
    df['prob'] = info['prob']
    df['id'] = df.index + 1 # starts at 1, because 0 is background
    #reorder columns
    df = df[['id', 'prob', 'z', 'y', 'x']]
    df.to_csv(out_file, index=False)

class SegmentationTask:
    """
    A class to perform segmentation of the image with the given model.
    """

    def __init__(self, 
                 image_files, channels,
                 model, 
                 parameters, 
                 output_files, 
                 output_folder='synapse_candidates'):
        """
        Args:
            image_files: a dict of image files to segment
            channels: a dict of channel numbers to use for segmentation
            model: a StarDist3D model to use for segmentation
            parameters: a dict of parameters to use for segmentation: n_tiles, prob_thr, scale. 
                n_tiles: number of tiles to use for prediction
                prob_thr: probability threshold for the prediction
                scale: scale to use for saving the probability image
            output_files: a dict of output parameters: basedir, label_images, probability_images, candidate_details.
                label_files: a list of tif label image file names to save the segmentation results 
                probability_files: a list of tif probability image file names to save the segmentation results 
                info_files: a list of csv file names to save the segmentation results
            output_folder: a folder to save the segmentation results
        """
        self.image_files = image_files
        self.channels = channels
        self.model = model
        self.parameters = parameters
        self.output_files = output_files
        self.output_folder = output_folder

    @classmethod
    def from_config_dict(cls, d):
        """
        Creates a SegmentationTask from a dict.
        """
        if 'basedir' in d['model']:
            model_file = Path(d['model']['basedir']) / d['model']['name']
            model = load_model_from_file(model_file)
        elif 'pre-trained' in d['model'] and d['model']['pre-trained'] == True:
            # load default model
            model = StarDist3D.from_pretrained(d['model']['name'])
        else:
            raise ValueError('model must have either basedir or pre-trained set to True')

        parameters = d['parameters']

        # create full path to image files:
        image_files = {}
        channel = {}
        basedir = Path(d['data']['basedir'])
        for file in d['data']['image_files']:
            image_files[file] = basedir / d['data']['image_files'][file]['file']
            channel[file] = d['data']['image_files'][file]['channel']
        
        # create output folder
        output_files = d['output']
        step = d.get('step', 'nuclei_segmentation')
        name = d.get('name', generate_slug(2)) # generate random name if not provided
        output_folder = Path(output_files['basedir']) / step / name

        return cls(image_files, channel, model, parameters, output_files, output_folder)
    
    @classmethod
    def from_config_file(cls, config_file):
        """
        Creates a SegmentationTask from a yaml config file.
        """
        with open(config_file) as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_config_dict(d)
    
    def create_output_folder(self):
        """
        Creates the output folder for the current run.
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def save_labels(self, labels, tp_name, verbose=True):

        out_file = self.output_folder / self.output_files['label_files'][tp_name]

        if verbose:
            print(f"Saving labels to {out_file}")
        save_image_as_uint16(labels, out_file)

    def save_probability(self, prob, tp_name, verbose=True):

        out_file = self.output_folder / self.output_files['probability_files'][tp_name]

        if verbose:
            print(f"Saving probability to {out_file}")

        save_image_as_uint16(prob, out_file, scale = self.parameters['scale'])

    def save_info(self, info, tp_name, verbose=True):

        out_file = self.output_folder / self.output_files['info_files'][tp_name]

        if verbose:
            print(f"Saving the details to {out_file}")
        save_info_as_csv(info, out_file)
    
    def save_config(self, config, verbose=True):
        """
        Saves the config to the output directory of the task.
        """
        out_file = self.output_folder / "config.yaml"

        if verbose:
            print(f"Saving the config to {out_file}")

        with open(out_file, 'w') as f:
            yaml.dump(config, f)

    def segment_image(self, img_file, channel, tp_name, verbose=True):
        """
        Segments the image and saves the results to the output directory.
        """
        # load the image
        img = load_image(img_file, channel, plot=False)

        # preprocess the image
        img = preprocess(img)

        prob_thresh = self.parameters.get('prob_thr', None)

        # segment the image
        labels, details = segment(self.model,
                                  img, 
                                  n_tiles = self.parameters['n_tiles'], 
                                  prob_thresh = prob_thresh, 
                                  return_predict=True, plot=False)
        return labels, details

    def run(self, verbose=True):
        """
        Runs the segmentation task.
        """
        if verbose:
            print(f"Running segmentation task {self.output_folder.name}")
            print(f"Segmenting {len(self.image_files)} images")

        for tp_name, img_file in self.image_files.items():

            channel = self.channels.get(tp_name)

            if verbose:
                print(f"Segmenting {tp_name} image {img_file}")

            labels, details = self.segment_image(img_file, channel,tp_name, verbose=verbose)

            # save results
            self.save_labels(labels[0], tp_name, verbose=verbose)
            # save_tiff_imagej_compatible(
            #     self.output_folder / f"{tp_name}_labels_save_tiff_imagej_compatible.tif",labels[0], axes='ZYX'
            #     )
            self.save_probability(details[0], tp_name, verbose=verbose)
            # save_tiff_imagej_compatible(
            #     self.output_folder / f"{tp_name}_prob_save_tiff_imagej_compatible.tif",details[0], axes='ZYX'
            #     )
            self.save_info(labels[1], tp_name, verbose=verbose)


def run_segmentation(config, verbose=True):
    """
    Runs the segmentation task defined in the config file or dictionary.
    """
    if isinstance(config, str) or isinstance(config, Path):
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    if isinstance(config, dict):
        task = SegmentationTask.from_config_dict(config)
    else:
        raise ValueError("config must be a path to a config file or a config dict")

    task.create_output_folder()
    task.save_config(config, verbose=verbose)
    task.run(verbose=verbose)



