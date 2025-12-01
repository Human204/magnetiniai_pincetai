import pandas as pd
import numpy as np
import cv2
import os
import statistics
from numpy.fft import fft2, fftshift, ifft2, ifftshift


def crop_image(img, box_top_left, box_width, box_height):
        left = int(box_top_left[0])
        top = int(box_top_left[1])
        right = int(left + box_width)
        bottom = int(top + box_height)

        return img[top:bottom, left:right]

def prepare_image(video_path,filename,crop_top_left,crop_size):
        file_path = os.path.join(video_path, filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if crop_size!=0:
                image=crop_image(image,crop_top_left,crop_size,crop_size)
        
        return image

def image_background_brightness(image):
        top_left = image[0:5, 0:5]
        top_right = image[0:5, -5:]
        bottom_left = image[-5:, 0:5]
        bottom_right = image[-5:, -5:]
        brightness = np.concatenate((top_left.flatten(), top_right.flatten(), 
                                  bottom_left.flatten(), bottom_right.flatten()))
        # brightness=top_left+top_right+bottom_left+bottom_right

        return [np.mean(brightness),np.median(brightness),statistics.mode(brightness)]

def filtered_image(ftimage,crow,ccol,r_in,r_out):
    rows, cols = ftimage.shape
    mask=np.zeros((rows,cols),dtype=np.uint8)

    x,y=np.ogrid[:rows,:cols]

    mask_area = np.logical_and(((x - crow)**2 + (y - ccol)**2 >= r_in**2),
                           ((x - crow)**2 + (y - ccol)**2 <= r_out**2))
    

    mask[mask_area] = 1

    m_app_ftimage = ftimage * mask
    i_ftimage = ifftshift(m_app_ftimage)
    result_img = ifft2(i_ftimage)
    tmp = np.log(np.abs(result_img) + 1)
    
    return tmp , result_img