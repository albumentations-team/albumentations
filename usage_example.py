from .transforms import *
from .composition import OneOf
from .example_transform import augment_flips_color
from scipy.misc import imread
import os
from random import shuffle
import cv2

root = r'D:\tmp\bowl\train_imgs\images_all5'
masks = r'D:\tmp\bowl\train_imgs\masks_all6'
# augs = Compose([
#     CLAHE(clipLimit=5, p=1),
#     InvertImg(p=1),
#     Remap(p=1),
#     RandomRotate90(p=1),
#     Transpose(p=1),
#     Blur(blur_limit=7, p=1),
#     ElasticTransform(p=1),
#     Distort1(p=1),
#     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.30, rotate_limit=45, p=1),
#     HueSaturationValue(p=1),
#     ChannelShuffle(p=1)
# ])
# augs = Distort2(p=1)
# augs = aug_oneof()
# augs = GaussNoise(1.)
augs = augment_flips_color(p=0.5)
data = os.listdir(root)
shuffle(data)
for fn in data:
    im = imread(os.path.join(root, fn), mode='RGB')
    mask = imread(os.path.join(masks, fn), mode='RGB')
    cv2.imshow('before', im)
    dat = augs(image=im, mask=mask)
    print(dat['image'].shape)
    # cv2.imshow('after', cv2.cvtColor(dat['image'][...,:3], cv2.COLOR_RGB2BGR))
    cv2.imshow('after', dat['image'][..., :3])
    cv2.imshow('mask', dat['mask'])
    cv2.waitKey()
