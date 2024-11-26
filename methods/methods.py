import os
import cv2
import PIL 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageQt
from skimage.segmentation import clear_border
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from PyQt6.QtGui import QPixmap


def img_enhancing(im):
    enhancer = ImageEnhance.Contrast(im)
    factor = 1.0
    im_output = enhancer.enhance(factor)
    im_output.show()
    # im_output.save('more-contrast-image.png')
    # img = cv2.imread('more-contrast-image.png')

def img_segment_colors_rgb(img_path): #, r_bound, g_bound, b_bound):
    img = imread(img_path)[:, :, :3]
    img_gs_1c = rgb2gray(img)

    # Grayscale image with 3 channels (the value is triplicated)
    img_gs = ((np.stack([img_gs_1c] * 3, axis=-1) * 255)
            .astype('int').clip(0, 255))

    # Red mask
    red_mask = ((img[:, :, 0] > 150) &
                (img[:, :, 1] < 100) &
                (img[:, :, 2] < 200))
    img_red = img_gs.copy()
    img_red[red_mask] = img[red_mask]

    # Green mask
    green_mask = ((img[:, :, 0] > 150) &
                (img[:, :, 1] > 190) &
                (img[:, :, 2] > 50))
    img_green = img_gs.copy()
    img_green[green_mask] = img[green_mask]

    # Blue mask
    blue_mask = ((img[:, :, 0] < 80) &
                (img[:, :, 1] < 85) &
                (img[:, :, 2] > 50))
    img_blue = img_gs.copy()
    img_blue[blue_mask] = img[blue_mask]

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].set_title("Red Segment")
    ax[0].imshow(img_red)
    ax[0].set_axis_off()
    ax[1].set_title("Green Segment")
    ax[1].imshow(img_green)
    ax[1].set_axis_off()
    ax[2].set_title("Blue Segment")
    ax[2].imshow(img_blue)
    ax[2].set_axis_off()
    plt.show()

def img_segment_colors_hsv(img_path):
    # Convert to HSV
    img = imread(img_path)[:, :, :3]
    img_hsv = rgb2hsv(img)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].set_title("Hue Channel")
    ax[0].imshow(img_hsv[:, :, 0], cmap='gray')
    ax[0].set_axis_off()
    ax[1].set_title("Saturation Channel")
    ax[1].imshow(img_hsv[:, :, 1], cmap='gray')
    ax[1].set_axis_off()
    ax[2].set_title("Value Channel")
    ax[2].imshow(img_hsv[:, :, 2], cmap='gray')
    ax[2].set_axis_off()
    plt.show()

    # Plot Hue Channel with Colorbar
    plt.imshow(img_hsv[:, :, 0], cmap='hsv')
    plt.title('Hue Channel with Colorbar')
    plt.colorbar()
    plt.show()

def pil_to_qpixmap(img):
    qim = ImageQt.ImageQt(img)
    pm = QPixmap.fromImage(qim)
    return pm

def get_gray_img(img_path):
    img = imread(img_path)
    img_gs = ((np.stack([rgb2gray(img)] * 3, axis=-1) * 255)
          .astype('int').clip(0, 255))
    # pil_img_gs = Image.fromarray(img_gs.astype('uint8'))
    # pil_img = img
    return img, img_gs