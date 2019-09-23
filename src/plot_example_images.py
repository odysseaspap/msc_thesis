import numpy as np
import util.plotting as plotting
import util.dataloading as dl
import matplotlib as mpl # Disable x-server requirement.
mpl.use('Agg')
import matplotlib.pyplot as plt

sample_file_path = "/home/schnettler/providentia/data/calibration/mp10_near_bag20-21/samples/mp10_cam_near_1512137196663852233.npz"
out_folder_path = "/home/schnettler/providentia/data/calibration/mp10_near_bag20-21"

rgb_image, projections_decalib, projections_gt, radar_detections, decalib, K, H_gt, rgb_img_orig_dim = dl.load_complete_sample(sample_file_path)

plt.imshow(rgb_image)
img_ax = plt.gca()
img_ax.get_xaxis().set_visible(False)
img_ax.get_yaxis().set_visible(False)
# Draw projections.
plotting.draw_projection_circles(img_ax, projections_gt, (1.,1.,0.))
plt.savefig(out_folder_path + "/cal_gt.png")
plt.close()


plt.imshow(rgb_image)
img_ax = plt.gca()
img_ax.get_xaxis().set_visible(False)
img_ax.get_yaxis().set_visible(False)
# Draw projections.
plotting.draw_projection_circles(img_ax, projections_decalib, (1.,1.,0.))
plt.savefig(out_folder_path + "/cal_proj.png")
plt.close()