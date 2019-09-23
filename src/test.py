#import keras
#import glob
#import os
#from run_config import RunConfig
#
#run_config = RunConfig()
#
#sample_file_names = [os.path.basename(f) for f in glob.glob(run_config.dataset_path + "samples/*.npz")]
#
#print(sample_file_names)
#print(len(sample_file_nafr
# import tensorflow as tf
# import keras

# from keras.applications import mobilenet
# from keras.layers import Input
# from keras.models import Model

# input_tensor = Input([150,240,3])
# mobilenet_model = mobilenet.MobileNet(input_tensor=input_tensor, weights='imagenet')
# mobilenet_model.summary()
# print(mobilenet_model.get_weights()[0])

# layers = mobilenet_model.layers
# print(len(layers))

# result_model = Model(inputs=mobilenet_model.input, outputs=mobilenet_model.get_layer(index=21).output)
# result_model.summary()
# print(result_model.get_weights()[0])


# import os
# import pandas as pd 
# import seaborn as sns 
# import numpy as np 
# import matplotlib.pyplot as plt

# tilt_errors = [0.1, 0.2, 0.2, -0.2]
# pan_errors = [0.3, -0.5, 0.6, -0.1]
# roll_errors = [1.5, 0.4, -10.0, -0.1]
# errors = np.array([tilt_errors, pan_errors, roll_errors])
# print(errors)
# print(errors[0,:])
# data = pd.DataFrame({'Tilt':errors[0,:], 'Pan':errors[1,:], 'Roll':errors[2,:]})
# tilt_data = pd.DataFrame({'Tilt':errors[0,:]})
# print(data)

# plt.figure()
# plt.figure(figsize=(7,7))
# sns.set(font_scale=2)
# sns.set_style('whitegrid')
# ax = sns.boxplot(data=tilt_data, palette="Blues", showfliers=False, width=0.6)# color='b', saturation=0.7)
# # ax.grid(False)
# ax.set(xlabel='', ylabel='Error in Degree')
# # Add transparency to colors
# for patch in ax.artists:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, .5))

# # medians = [data['Tilt'].median(), data['Pan'].median(), data['Roll'].median()]
# # median_labels = [str(np.round(s, 2)) for s in medians]

# # pos = range(len(medians))
# # for tick,label in zip(pos,ax.get_xticklabels()):
# #     ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 
# #             horizontalalignment='center', size='x-small', color='b', weight='semibold')

# fig = ax.get_figure()
# fig.savefig("testplot.png")
# plt.close(fig)

# import os, glob
# from run_config import RunConfig

# run_config = RunConfig()
# sample_files_list = []
# print("Loading datasets:")
# for dataset_path in run_config.dataset_paths:
#     print(dataset_path)
#     temp = [f for f in glob.glob(dataset_path + "*.npz")] # get list of all sample file names in dataset folder
#     sample_files_list.extend(temp)
    
# sample_files_list.sort()
# print("List of data samples loaded: {} samples found.".format(str(len(sample_files_list))))

import numpy as np

data = [1, -1, 1]

print(np.mean(data))
print(np.median(data))