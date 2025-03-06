import os
import cv2
import PIL 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageQt
from skimage.segmentation import clear_border
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
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

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00']) if M['m00'] else 0
    cy = int(M['m01']/M['m00']) if M['m00'] else 0

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def run_clustering(parent, num_clusters=2):
    tmp_cellpix = np.copy(parent.cellpix[0])
    # image = cv2.imread(parent.filename, cv2.IMREAD_UNCHANGED)
    image = parent.stack[0].copy()[:2]  # bug here
    df_mean_color = pd.DataFrame()

    for idx in range(parent.cellpix[0].max()):
        tmp_mask = np.copy(parent.cellpix[0])
        masked = np.zeros_like(image[:, :, 0]) ## ??

        tmp_mask[idx + 1 != parent.cellpix[0]] = 0
        tmp_mask[idx + 1 == parent.cellpix[0]] = 255
        
        contours, _ = cv2.findContours(tmp_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = [scale_contour(contour, 0.7) for contour in contours]   # We want to avoid white pixels on outskirt regions of the mask
        
        # area = int(cv2.contourArea(contours[0]))
        # if area < 700:
        #     print("TOO SMALL IDX: ", idx)

        cv2.drawContours(masked, [contours[0]], 0, 255, -1)
        B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)  # Average
        
        df = pd.DataFrame({'red': R_mean, 'green': G_mean, 'blue': B_mean}, index=[idx])
        df_mean_color = pd.concat([df_mean_color, df])

    km = KMeans(n_clusters=num_clusters)
    df_mean_color['label'] = km.fit_predict(df_mean_color)
    parent.update_cellcolors(df_mean_color['label'])
    # print("Look here: \n", df_mean_color.head())

