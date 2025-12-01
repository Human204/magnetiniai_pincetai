import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema, find_peaks_cwt, argrelmin
import pandas as pd
from scipy.interpolate import interp1d


import functools
print = functools.partial(print, flush=True)

from scipy.signal import savgol_filter
import math

from scipy.interpolate import splrep, splev,BSpline 

from radius.trends import get_current_trend

from utils.visualizations import visualize_radius_track

def radial_profile(image, center=None, nbins=500):
    ny, nx = image.shape
    if center is None:
        center = (ny//2, nx//2)
    
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    r_flat = r.flatten()
    img_flat = image.flatten()

    r_max = r_flat.max()
    bin_edges = np.linspace(0, r_max, nbins+1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    profile = np.zeros(nbins)

    for i in range(nbins):
        mask = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i+1])
        if np.any(mask):
            profile[i] = img_flat[mask].mean()
    
    return bin_centers, profile

def radius_interpolated(image, ccol, crow, output_name, frame, prev_radius, trend_segments=None,prev_peak=None):
    
    scale=100000
    
    _, prof = radial_profile(image, (ccol, crow), 
                           int(math.sqrt(math.pow(image.shape[0]/2,2)+math.pow(image.shape[1]/2,2))))
    
    prof = savgol_filter(prof, window_length=10, polyorder=3)

    current_trend = get_current_trend(frame, trend_segments) if trend_segments else "stable"
    
    roi = None
    closest_point = None
    current_peak = None
    
    if prev_radius is None:
        roi, closest_point, current_peak = get_roi_with_sudden_change(prof, current_trend, None, None)
    else:
        roi, closest_point, current_peak = get_roi_with_sudden_change(prof, current_trend, prev_radius,prev_peak)

    prof_roi = prof[roi[0]:roi[1]]

    idx=find_idx(prof_roi,roi,closest_point,scale)

    # if frame:  #nah
    #     visualize_radius_track(image,crow,ccol,idx,prof,roi,prof_roi,tck,x_local,I_fit_local,local_start,local_end,output_name,frame)

    return idx, prof, current_peak 


def get_roi_with_sudden_change(radial_prof, current_trend, prev_peak, prev_chosen=None):
    """
    Enhanced ROI selection that returns fallback from previous frame on failure.
    """
    try:
        # ---- 1) Peak Detection ----
        peaks = find_peaks_cwt(radial_prof, widths=np.arange(0.5, 2))

        if len(peaks) == 0:
            raise ValueError("No peaks found")

        peak_vals = [radial_prof[p] for p in peaks]
        peak_dict = dict(zip(peaks, peak_vals))

        max_idx = max(peak_dict, key=peak_dict.get)
        threshold = peak_dict[max_idx] * 0.9

        # ---- 2) Track radius based on previous frame ----
        chosen_idx = find_tracked_radius(current_trend, prev_chosen, max_idx)

        # ---- 3) Optional next peaks ----
        next_peaks = peaks[peaks > chosen_idx]
        next_idx = next_peaks[0] if len(next_peaks) else None

        # ---- 4) ROI ----
        roi = (
            (chosen_idx - 15, next_idx + 15)
            if next_idx is not None
            else (chosen_idx - 15, chosen_idx + 15)
        )

        # ---- 5) Minima detection ----
        minima = argrelmin(radial_prof, order=2)[0]
        eligible_minima = [m for m in minima if m > chosen_idx]

        if len(eligible_minima) == 0:
            raise ValueError("No valid minima found")

        idx_closest = eligible_minima[0]

        return roi, idx_closest, chosen_idx

    except Exception as e:
        print(f"[Fallback] get_roi_with_sudden_change failed: {e}")
        
        # ---- FALLBACK TO PREVIOUS FRAME ----
        if prev_chosen is not None:
            return (prev_chosen - 20, prev_chosen + 20), prev_chosen, prev_chosen
        
        # If even prev_chosen is missing, fallback to a generic safe output
        print("[Fallback] prev_chosen is None → returning generic default ROI")
        return (0, 40), 20, 20


def find_tracked_radius(current_trend,prev_chosen,max_idx):
    if current_trend == "shrinking" and prev_chosen is not None:
        print(f"restricting to peaks <= {prev_chosen}")
        eligible = {idx: val for idx, val in eligible.items() if idx <= prev_chosen}
        shrinking_peaks = [idx for idx in eligible.keys() if idx <= prev_chosen]

        chosen_idx = max(shrinking_peaks) if shrinking_peaks else max_idx
        # print(eligible,prev_peak)
        
    elif current_trend == "expanding" and prev_chosen is not None:
        print(f"restricting to peaks >= {prev_chosen}")
        eligible = {idx: val for idx, val in eligible.items() if idx >= prev_chosen}
        expanding_peaks = [idx for idx in eligible.keys() if idx >= prev_chosen]
        chosen_idx = min(expanding_peaks) if expanding_peaks else max_idx
        
    elif current_trend == "stable" and prev_chosen is not None:
        print(eligible)
        chosen_idx = min(eligible.keys(), key=lambda x: abs(x - prev_chosen))
        # Create a new eligible dict with only the closest peak
        eligible = {chosen_idx: eligible[chosen_idx]}
    else:
        chosen_idx=max_idx
    return chosen_idx


def find_idx(prof_roi,roi,closest_point,scale=100000):
    x=np.arange(0,len(prof_roi), 0.01)

    tck = splrep(range(roi[1]-roi[0]), prof_roi, k=4) 

    # print(closest_point)
    window_size = 3  # pixels around closest_point
    local_start = max(0, closest_point - roi[0] - window_size)
    local_end = min(len(prof_roi), closest_point - roi[0] + window_size + 1)

    x_local = np.arange(local_start, local_end, 1/scale)


    I_fit_local = splev(x_local, tck)
    
    I_fit = splev(x, tck)

    find_min,_=find_peaks(-I_fit_local,prominence=0.05,distance=1)

    find_min = list(find_min)
    for i in range(len(find_min)):
        find_min[i] /= scale  
        find_min[i] += local_start + roi[0] 

    if len(find_min) > 0:
        idx = min(find_min, key=lambda x: abs(x-closest_point))
    else:
        # print(f"Warning: No minima found for frame {frame}, using closest_point")
        find_min=np.argmin(-I_fit_local)
        
        find_min/=scale
        find_min+=local_start+roi[0]
        idx=find_min

    return idx