import gc

import cv2

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema

import pandas as pd
from scipy.interpolate import interp1d
import traceback


from utils.prepare_image import image_background_brightness, filtered_image



def ring_image(image,r_in_center,r_out_center,ring_index,output_path,filename,prev_radius,brightness,clipLimit,titleGridSize,dp,param1,param2,resolution=0.01):
    try:
    
        bright=image_background_brightness(image)
        beta = brightness - bright[1]
        
        imageBright = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

        ftimage = fft2(imageBright)
        ftimage = fftshift(ftimage)

        rows, cols = ftimage.shape
        crow, ccol = rows // 2, cols // 2


        # applying filter for center
        tmp_center,result_img = filtered_image(ftimage,crow,ccol,r_in_center,r_out_center)

        # tmp_radius,unused = filtered_image(ftimage,crow,ccol,r_in_radius,r_out_radius)
        tmp_radius,unused = filtered_image(ftimage,crow,ccol,3,8)

        idx = tmp_center.argmax()

        crow, ccol = np.unravel_index(idx, tmp_center.shape)
        
        
        # intensity profile for circle center
        central_line_y_center = tmp_center[crow, :]
        central_line_x_center = tmp_center[:, ccol]

        peaks_y_center = get_subpixel_peak_com(argrelextrema(central_line_y_center, np.greater)[0], central_line_y_center,resolution)
        peaks_x_center = get_subpixel_peak_com(argrelextrema(central_line_x_center, np.greater)[0], central_line_x_center,resolution)

        # intensity profile for circle radius
        central_line_y_radius = tmp_radius[crow, :]
        central_line_x_radius = tmp_radius[:, ccol]
    
        peaks_y_radius = get_subpixel_peak_com(argrelextrema(central_line_y_radius, np.greater)[0], central_line_y_radius,resolution)
        peaks_x_radius = get_subpixel_peak_com(argrelextrema(central_line_x_radius, np.greater)[0], central_line_x_radius,resolution)

        minimums_y_radius = get_subpixel_minimum_com(argrelextrema(central_line_y_radius, np.less)[0],central_line_y_radius)
        minimums_x_radius = get_subpixel_minimum_com(argrelextrema(central_line_x_radius, np.less)[0],central_line_x_radius)

        if len(peaks_x_center) > 1 and len(peaks_y_center) > 1:
            prev_ccol,prev_crow=ccol,crow
            ccol, x_distance = get_center_from_peaks(peaks_y_center, ccol)
            crow, y_distance = get_center_from_peaks(peaks_x_center, crow)


            refined_peaks_x_radius=peaks_by_minimums(minimums_x_radius,peaks_x_radius)
            refined_peaks_y_radius=peaks_by_minimums(minimums_y_radius,peaks_y_radius)

            circle_radius = circle_radius_fun(refined_peaks_x_radius,refined_peaks_y_radius,ring_index,ccol,crow)

            if circle_radius==-1:
                circle_radius=prev_radius
            
            ccol_hough,crow_hough,circle_radius = houghCircle(ccol,crow,circle_radius,image,prev_radius,clipLimit,titleGridSize,dp,param1,param2)

            if circle_radius==-1:
                circle_radius=prev_radius

            ccol_hough = round(ccol_hough, 5)
            crow_hough = round(crow_hough, 5)
            circle_radius = round(circle_radius, 5)

            plt.clf()
            plt.cla()
            plt.close('all')
            del image
            del ftimage, result_img, tmp_center
            gc.collect()


            return circle_radius, pd.DataFrame({
            'x': [ccol_hough],
            'y': [crow_hough]
            })
    except Exception as e:
        print(f'fail at ring_image: {e}')
        traceback.print_exc()
        return prev_radius,pd.DataFrame({
            'x':[-1],
            'y':[-1]
        })


def get_subpixel_peak_com(peaks, intensity_profile, resolution=0.01):
    refined_peaks = []
    for peak in peaks:
        if 0 < peak < len(intensity_profile) - 1:
            # range around the peak for interpolation
            x_range = np.linspace(peak - 1, peak + 1, int(2 / resolution) + 1)
            
            # Interpolate the intensity profile
            interp_func = interp1d(np.arange(len(intensity_profile)), intensity_profile, kind='cubic')
            y_range = interp_func(x_range)
            
            max_index = np.argmax(y_range)
            refined_peak = x_range[max_index]
            refined_peaks.append(refined_peak)
        else:
            refined_peaks.append(peak)
    return refined_peaks

def get_subpixel_minimum_com(minimums, intensity_profile):
    refined_peaks = []
    for peak in minimums:
        if 0 < peak < len(intensity_profile) - 1:
            x = np.array([peak - 1, peak, peak + 1], dtype=float)
            y = intensity_profile[x.astype(int)]
            refined_peak = np.sum(x * y) / np.sum(y)
            refined_peaks.append(refined_peak)
        else:
            refined_peaks.append(peak)
    return refined_peaks

def get_center_from_peaks(peaks, current_center):
    # print(f'peaks: {peaks}\ncurrentCenter: {current_center}')
    valid_peaks = [peak for peak in peaks if np.abs(peak - current_center) >= 10]
    left_peaks = np.array([peak for peak in valid_peaks if peak < current_center])
    right_peaks = np.array([peak for peak in valid_peaks if peak > current_center])

    if len(left_peaks) == 0 or len(right_peaks) == 0:
        return current_center, 0

    left_peak = left_peaks[-1] 
    right_peak = right_peaks[0] 

    refined_center = (left_peak + right_peak) / 2

    distance_between_peaks = np.abs(right_peak - left_peak)

    return refined_center, distance_between_peaks

def peaks_by_minimums(minimums,peaks):
    refined_peaks=[]
    peaks_and_minimums=zip(peaks,minimums)
    # print(peaks_and_minimums)
    for peak,minimum in peaks_and_minimums:
        if 0<peak<len(peaks)-1:
            refined_peak=(minimum-1+minimum+1)/2
        else:
            refined_peaks.append(peak)
    # print(refined_peaks)
    return refined_peaks

def circle_radius_fun(peaks_x,peaks_y,ring_index,ccol,crow):
    try:
        closest_peak_x = min(peaks_x, key=lambda x: abs(x-ccol))
        closest_peak_y = min(peaks_y, key=lambda y: abs(y-crow))

        peak_index_x=peaks_x.index(closest_peak_x)
        peak_index_y=peaks_y.index(closest_peak_y)


        if peak_index_x > 0:
            if len(peaks_x)==peak_index_x+1:
                peak_index_x-=1
            rl_x = ccol - peaks_x[peak_index_x - ring_index]
            rr_x = peaks_x[peak_index_x + ring_index] - ccol

        if peak_index_y > 0:
            if len(peaks_y)==peak_index_y+1:
                peak_index_y-=1
            ru_y = crow - peaks_y[peak_index_y - ring_index]
            rl_y = peaks_y[peak_index_y + ring_index] - crow

        result=round((rl_x + rr_x + ru_y + rl_y) / 4., 5)

        return result
    except Exception as e:
        print(f"circle fun {e}")
        traceback.print_exc()
        return -1
    

def houghCircle(ccol,crow,radius,image,prev_radius,clipLimit,titleGridSize,dp,param1,param2):
    global centers
    try:
        # ccol -x, crow - y
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=titleGridSize)
        cl1 = clahe.apply(image)
    
        # cl1=cv2.equalizeHist(image)
        
        blur = cv2.GaussianBlur(cl1,(7,7),0)
        
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT_ALT, dp=dp, minDist=0.00001,
                                param1=param1, param2=param2, minRadius=0, maxRadius=0)
    
        results =[]
        frame_no=0
        if circles is not None:
            circles = np.squeeze(circles, axis=0) 
            closest_circle = None
            min_distance = float('inf')
    
            for circle in circles:
                detected_x, detected_y, detected_radius = circle
    
                center_distance = np.sqrt((detected_x - ccol)**2 + (detected_y - crow)**2)
    
                radius_difference = abs(detected_radius - radius)
                if(prev_radius==None):
                    change_difference=0
                else:
                    change_difference = abs(detected_radius - prev_radius)
    
                # More weight is given to the existing center (is more precise than radius)
                total_metric = 2*center_distance + 1.5*change_difference+radius_difference
    
                if total_metric < min_distance:
                    min_distance = total_metric
                    closest_circle = circle
    
            frame_no+=1
    
            if closest_circle is not None:
                closest_x, closest_y, closest_radius = closest_circle
    
            ftimage = fft2(image)
            ftimage = fftshift(ftimage)
    
            tmp,result_img = filtered_image(ftimage=ftimage,crow=int(closest_y),ccol=int(closest_x),r_in=6,r_out=12)
    
    
            if(np.sqrt((closest_x - ccol)**2 + (closest_y - crow)**2)<=10):
                return ccol,crow,closest_radius
            else:
                ccol,crow = closest_x,closest_y
                ccol,crow = get_subpixel_center_hough(ccol,crow,tmp)
                radius=-1
                return ccol,crow,radius
    
        else:
            print("No circles detected.")
            return ccol,crow,radius
    except Exception as e:
        print(f"fail hough: {e}")
        return ccol,crow,20

def get_subpixel_center_hough(ccol,crow,tmp):


    horizontal_profile = tmp[int(crow),:]
    peaks_horizontal = argrelextrema(horizontal_profile, np.greater)[0]
    distances = np.abs(peaks_horizontal - crow)
    closest_peak_index = peaks_horizontal[np.argmin(distances)]
    ccol = get_subpixel_peak_com([closest_peak_index],horizontal_profile)[0]


    vertical_profile = tmp[:,int(ccol)]
    peaks_vertical = argrelextrema(vertical_profile, np.greater)[0]
    distances = np.abs(peaks_vertical - crow)
    closest_peak_index = peaks_vertical[np.argmin(distances)]
    crow = get_subpixel_peak_com([closest_peak_index],vertical_profile)[0]

    return crow,ccol