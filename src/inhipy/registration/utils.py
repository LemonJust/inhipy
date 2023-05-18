import ants 
import yaml
import numpy as np
from pathlib import Path
import tifffile as tif
import matplotlib.pyplot as plt
import warnings

def read_image(image_path):
    return ants.image_read(image_path)


def save_image(image, file_path):
    ants.image_write(image, file_path)

def save_image_as_uint16(img, out_file, scale = None):
    """
    Save the image as uint16 tiff file after scaling it by scale. 
    Scale is used to scale float values, such as probability, to the range of uint16.
    """
    if scale is not None:
        img = img * scale

    #assert all values can fint into the uint range
    # assert np.max(img) < 65535, "some labels image values are too large for uint16"
    # assert np.min(img) >= 0, "some labels image values are too small for uint16"
    # throw a warning if values are outside the uint range
    if np.max(img) > 65535:
        warnings.warn("some labels image values are too large for uint16. Image will be clipped.")
    if np.min(img) < 0:
        warnings.warn("some labels image values are too small for uint16. Image will be clipped.")
    img = np.clip(img, 0, 65535)
        



    img = img.astype(np.uint16)
    tif.imwrite(out_file, img, imagej=True)