"""
Define a pil loader myself to open 4 channels image
"""

from PIL import Image


def my_pil_loader(path):
    """Just open a image file, do not convert to 'RGB' mode
    reference: torchvision.datasets.folder.py"""

    return Image.open(path)
