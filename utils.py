import numpy as np


def rgb2onehot(y_unprocc, width, height):
    """
    Mapping from RGB color to class ID
    Class        R      G       B       ID
    Void         -      -       -       0
    Road         170    170     170     1
    Grass        0      255     0       2
    Vegetation   102    102     51      3
    Tree         0      60      0       3
    Sky          0      120     255     4
    Obstacle     0      0       0       5
    """
    ROAD = np.array([170, 170, 170], dtype=np.uint8)
    GRASS = np.array([0, 255, 0], dtype=np.uint8)
    VEGETATION = np.array([102, 102, 51], dtype=np.uint8)
    TREE = np.array([0, 60, 0], dtype=np.uint8)
    SKY = np.array([0, 120, 255], dtype=np.uint8)
    OBSTACLE = np.array([0, 0, 0], dtype=np.uint8)

    # Vegetation and trees have different colors but have the same class ID?
    # Convert output images from RGB to one-hot vectors
    y = np.zeros((len(y_unprocc), height, width, 5), dtype=np.float32)

    road_indices = np.where(np.all(y_unprocc == ROAD, axis=-1))
    grass_indices = np.where(np.all(y_unprocc == GRASS, axis=-1))
    vegetation_indices = np.where(np.all(y_unprocc == VEGETATION, axis=-1))
    tree_indices = np.where(np.all(y_unprocc == TREE, axis=-1))
    sky_indices = np.where(np.all(y_unprocc == SKY, axis=-1))
    obstacle_indices = np.where(np.all(y_unprocc == OBSTACLE, axis=-1))

    y[road_indices[0], road_indices[1], road_indices[2], 0] = 1.0
    y[grass_indices[0], grass_indices[1], grass_indices[2], 1] = 1.0
    y[vegetation_indices[0], vegetation_indices[1], vegetation_indices[2], 2] = 1.0
    y[tree_indices[0], tree_indices[1], tree_indices[2], 2] = 1.0
    y[sky_indices[0], sky_indices[1], sky_indices[2], 3] = 1.0
    y[obstacle_indices[0], obstacle_indices[1], obstacle_indices[2], 4] = 1.0

    return y


def process_output(y_unprocc, width, height):
    """
    Mapping from RGB color to class ID
    Class        R      G       B       ID
    Void         -      -       -       0
    Road         170    170     170     1
    Grass        0      255     0       2
    Vegetation   102    102     51      3
    Tree         0      60      0       3
    Sky          0      120     255     4
    Obstacle     0      0       0       5
    """
    ROAD_RGB = np.array([170, 170, 170], dtype=np.uint8)
    GRASS_RGB = np.array([0, 255, 0], dtype=np.uint8)
    VEGETATION_RGB = np.array([102, 102, 51], dtype=np.uint8)
    TREE_RGB = np.array([0, 60, 0], dtype=np.uint8)
    SKY_RGB = np.array([0, 120, 255], dtype=np.uint8)
    OBSTACLE_RGB = np.array([0, 0, 0], dtype=np.uint8)

    y_unprocc = np.expand_dims(np.argmax(y_unprocc, axis=-1), axis=-1)

    # Vegetation and trees have different colors but have the same class ID?
    # Convert output images from one-hot vectors to RGB
    y = np.zeros((len(y_unprocc), height, width, 3), dtype=np.uint8)

    road_indices = np.where(np.all(y_unprocc == 0, axis=-1))
    grass_indices = np.where(np.all(y_unprocc == 1, axis=-1))
    vegetation_indices = np.where(np.all(y_unprocc == 2, axis=-1))
    # tree_indices = np.where(np.all(y_unprocc == TREE_VEC, axis=-1))
    sky_indices = np.where(np.all(y_unprocc == 3, axis=-1))
    obstacle_indices = np.where(np.all(y_unprocc == 4, axis=-1))

    y[road_indices[0], road_indices[1], road_indices[2]] = ROAD_RGB
    y[grass_indices[0], grass_indices[1], grass_indices[2]] = GRASS_RGB
    y[vegetation_indices[0], vegetation_indices[1], vegetation_indices[2]] = VEGETATION_RGB
    # y[tree_indices[0], tree_indices[1], tree_indices[2]] = TREE_RGB
    y[sky_indices[0], sky_indices[1], sky_indices[2]] = SKY_RGB
    y[obstacle_indices[0], obstacle_indices[1], obstacle_indices[2]] = OBSTACLE_RGB

    return y

