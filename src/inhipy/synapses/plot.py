import matplotlib.pyplot as plt

from stardist import random_label_cmap
lbl_cmap = random_label_cmap()

def plot_segmentation_example(img, segmentation, vmin = 100, vmax = 2000, show_slices = None):
    n_slices = img.shape[0]
    width, height = img.shape[1:]
    if show_slices is None:
        show_slices = [3*n_slices//8, 5*n_slices//8]
    fig, ax = plt.subplots(2,2, figsize=(16,16*(int(height/width))))
    for a, i_s, plot_seg in zip(ax.flat,[show_slices[0],show_slices[0], show_slices[1],show_slices[1]],
                                [False,True,False,True]):
        a.imshow(img[i_s], cmap='gray', vmin=vmin, vmax=vmax)
        if plot_seg:
            a.imshow(segmentation[i_s], cmap=lbl_cmap, alpha=0.5)
        a.set_title(i_s)
    [a.axis('off') for a in ax.flat]
    plt.tight_layout()