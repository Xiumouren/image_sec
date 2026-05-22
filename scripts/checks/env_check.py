import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
import imagehash
import exifread

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PIL ok")
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("imagehash ok")
print("exifread ok")

try:
    import clip
    print("clip ok")
except Exception as e:
    print("clip fail:", e)

try:
    from pytorch_grad_cam import GradCAM
    print("grad-cam ok")
except Exception as e:
    print("grad-cam fail:", e)

try:
    import faiss
    print("faiss ok")
except Exception as e:
    print("faiss fail:", e)