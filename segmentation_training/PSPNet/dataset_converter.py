from scipy import misc
import os
from glob import glob
import numpy as np
from PIL import Image

base_path = "../data_road/training/"

color_paths = glob(base_path + "gt_image_2/*.png")
color_paths.sort()

id_dir = base_path + "ids"

if not os.path.exists(id_dir):
    os.mkdir(id_dir)

id_paths = []
for color_p in color_paths:
    id_paths.append(color_p.replace("gt_image_2", "ids"))



for img_path, output_path in zip(color_paths, id_paths):
    img = misc.imread(img_path)
    img_shape = img.shape

    id_img = np.zeros((img_shape[0], img_shape[1]))

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i][j][0] == 255 and img[i][j][1] == 0 and img[i][j][2] == 255:
                id_img[i][j] = 0
            else:
                id_img[i][j] = 1

    misc.toimage(id_img, high=1).save(output_path)
