import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import random
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

imaPath = r"original_image"
output = r"corrupt_image"

imaList = os.listdir(imaPath)
for files in imaList:
    path_ima = os.path.join(imaPath, files)
    path_processed = os.path.join(output, files)
    print(path_ima)
    image = np.asarray(Image.open(path_ima))
    i = random.randint(0,14)
    s = random.randint(1,5)
    corrupted = corrupt(image, corruption_number=i, severity=s)
    Image.fromarray(corrupted).save(path_processed)
    # pic.save(path_processed)