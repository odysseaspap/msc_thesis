import os
import pickle
import numpy as np

import matplotlib as mpl # Disable x-server requirement.
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_history(history, output_folder):
    for key in history.keys():
        if 'lr' in key:
            continue
        if 'val' in key:
            continue
        plt.figure()
        plt.plot(history[key])
        plt.plot(history['val_' + key])
        plt.title('model ' + key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        # if 'angle' in key:
            # plt.ylim(ymax=4.)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(output_folder + '{}.png'.format(key))
        plt.close()

    with open(output_folder + 'train_history.json', 'wb') as hist_file:
        pickle.dump(history, hist_file)


def draw_projection_circles(image_ax , proj, color):
    proj_nonzeros = np.nonzero(proj)
    for coord in zip(proj_nonzeros[1], proj_nonzeros[0]):
        circle = Circle(coord, radius=2, color=color)
        image_ax.add_patch(circle)

def draw_image_with_projections(rgb_image, proj_gt, proj):
    plt.imshow(rgb_image)
    img_ax = plt.gca()
    img_ax.get_xaxis().set_visible(False)
    img_ax.get_yaxis().set_visible(False)
    # Draw projections.
    draw_projection_circles(img_ax, proj_gt, (0.,0.,1.))
    draw_projection_circles(img_ax, proj, (1.,1.,0.))
    # plt.show()

def visualize_corrected_projection(rgb_images, projection_gt, projection, out_folder_path):
    # Create output folder.
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    # Draw projection image.
    for i in range(rgb_images.shape[0]):
        draw_image_with_projections(rgb_images[i], projection_gt[i], projection[i])
        plt.savefig(out_folder_path + "/{}.png".format(i))
        plt.close()
