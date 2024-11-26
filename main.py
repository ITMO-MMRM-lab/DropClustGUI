import PIL 
from PIL import Image
import cv2
from methods.methods import img_enhancing, img_segment_colors_rgb, img_segment_colors_hsv
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border

img_path = '/home/mellamoarroz/Documents/drop_clus/frames/frame97.png'
# img_path = '/run/media/mellamoarroz/a86278da-9c9c-4825-9b11-4e1ddfc74d03/DATASETS/ITMO/Svetlana Images/АО+PI/Необычные варианты/Светлое поле/Project_Image046_ch00.jpg'
im = Image.open(img_path)
# img_enhancing(im)
# img_segment_colors_rgb(img_path)
img_segment_colors_hsv(img_path)







#### Some segmentation bs
# im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# Z = im_cv.reshape((-1, 3))
# Z = np.float32(Z)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# attempts = 10
# ret, label, center = cv2.kmeans(Z,2,None,criteria,attempts,
#                             cv2.KMEANS_RANDOM_CENTERS)
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((im_cv.shape))
# grayimg = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
# # grayimg_small = cv2.resize(grayimg, (800, 800))


# blur = 55
# clipLimit = 55
# tileGridSize = 10
# arealimit = 4000

# clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize,tileGridSize))  
# cl1 = clahe.apply(grayimg)
# blur = cv2.GaussianBlur(cl1,(blur,blur),0)
# ret3, th_blur = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# kernel = np.ones((5, 5), np.uint8)

# # cv2.imshow('th_blur', cv2.resize(th_blur, (800, 800)))
# # cv2.waitKey(0)

# kmeans = cv2.dilate(th_blur, kernel, iterations=1)
# kmeans = clear_border(kmeans) #Remove edge touching grains

# cv2.imshow('kmeans', cv2.resize(kmeans, (800, 800)))
# cv2.waitKey(0)

# # contours, hierarchy = cv2.findContours(image = kmeans , mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_SIMPLE)
# # mask = np.zeros(im_cv.shape[:2],dtype=np.uint8)
# # for i,cnt in enumerate(contours):
# #     if hierarchy[0][i][2] == -1 :
# #         if  cv2.contourArea(cnt) > arealimit:
# #             cv2.drawContours(mask,[cnt], 0, (255), -1)

# # kmeans_results =mask
# # kernel = np.ones((5, 5), np.uint8)
# # kmeans_results = cv2.erode(kmeans_results, kernel,iterations=1)

# # fig = plt.figure(figsize=(40, 40))
# # fig.add_subplot(1, 4, 1)
# # plt.axis('on')
# # plt.title("Image")
# # plt.imshow(im_cv, cmap="gray")

# # fig.add_subplot(1, 4, 2)
# # plt.axis('off')
# # plt.title("Output")
# # plt.imshow(kmeans_results, cmap="gray")
# # plt.show()

# # cv2.imshow('grayimg', grayimg_small)
# # cv2.waitKey(0)