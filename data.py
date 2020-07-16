import glob
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from utils import rgb2onehot


FFD_TRAIN_RGB = "./data/freiburg_forest_annotated/train/rgb/*"
FFD_TRAIN_MASKS = "./data/freiburg_forest_annotated/train/GT_color/*"
FFD_TEST_RGB = "./data/freiburg_forest_annotated/test/rgb/*"
FFD_TEST_MASKS = "./data/freiburg_forest_annotated/test/GT_color/*"

KITTI_RGB = "./data/kitti/training/image_2/*"
KITTI_MASKS = "./data/kitti/training/semantic_rgb/*"

IMG_WIDTH = 728
IMG_HEIGHT = 448


def load_data(subset="ffd_train", should_resize=False):
    if subset == "ffd_train":
        x_paths = glob.glob(FFD_TRAIN_RGB)
        y_paths = glob.glob(FFD_TRAIN_MASKS)
    elif subset == "ffd_test":
        x_paths = glob.glob(FFD_TEST_RGB)
        y_paths = glob.glob(FFD_TEST_MASKS)
    elif subset == "dataset1":
        x_paths = glob.glob("./data/dataset_1/*")
        y_paths = []
    elif subset == "dataset2":
        x_paths = glob.glob("./data/dataset_2/*")
        y_paths = []
    elif subset == "dataset3":
        x_paths = glob.glob("./data/dataset_3/*")
        y_paths = []
    else:
        return None, None

    x_paths.sort()
    y_paths.sort()

    x_imgs = []
    y_imgs = []

    for i in range(len(x_paths)):
        # The network requires fixed input size but the input images have varying shape.
        # Images are cropped not resized to avoid introducing any interpolation artifacts.
        xi = imread(x_paths[i])

        if not should_resize:
            xi = xi[:IMG_HEIGHT, :IMG_WIDTH, :]
            if xi.shape[0] < IMG_HEIGHT or xi.shape[1] < IMG_WIDTH:
                continue
        else:
            xi = resize(xi, (IMG_HEIGHT, IMG_WIDTH, 3), anti_aliasing=False)

        x_imgs.append(xi)

    for i in range(len(y_paths)):
        # The network requires fixed input size but the input images have varying shape.
        # Images are cropped not resized to avoid introducing any interpolation artifacts.
        yi = imread(y_paths[i])

        if not should_resize:
            yi = yi[:IMG_HEIGHT, :IMG_WIDTH, :]
            if yi.shape[0] < IMG_HEIGHT or yi.shape[1] < IMG_WIDTH:
                continue
        else:
            yi = resize(yi, (IMG_HEIGHT, IMG_WIDTH, 3), anti_aliasing=False)

        y_imgs.append(yi)

    x_unprocc = np.array(x_imgs)
    y_unprocc = np.array(y_imgs)

    # Normalize x from [0, 255] value range to [0, 1] value range
    if should_resize:
        x = x_unprocc
    else:
        x = np.float32(x_unprocc / 255.)

    y = None
    if len(y_unprocc) > 0:
        y = rgb2onehot(y_unprocc, IMG_WIDTH, IMG_HEIGHT)

    return x, y
