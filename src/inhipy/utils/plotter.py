"""
Plotting utils.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_slice_with_insert(img,
                           slice = 80, zoom_in_region= (200,500, 400, 400),
                           vmin = 100, vmax = 2000):
    """
    plot the full size image and an insert of the center in one figure next to each other
    draw a box around the area that is shown in the insert.

    Args:
        img: 3D numpy array, zyx
        slice: z slice in the 3D image to plot
        zoom_in_region: tuple of (x, y, width, height) of the insert
        vmin, vmax: int, min and max values for the color scale
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img[slice,:,:], cmap='gray', vmin=vmin, vmax=vmax)
    ax2.imshow(img[slice,
               zoom_in_region[0]:zoom_in_region[0] + zoom_in_region[2],
               zoom_in_region[1]:zoom_in_region[1] + zoom_in_region[3]],
               cmap='gray', vmin=vmin, vmax=vmax)
    ax1.axis('off')
    ax2.axis('off')
    ax1.add_patch(plt.Rectangle( (zoom_in_region[1], zoom_in_region[0]), # y, x
                                 zoom_in_region[3], zoom_in_region[2], #  height, width
                                 linewidth=1, edgecolor='w', facecolor='none'))