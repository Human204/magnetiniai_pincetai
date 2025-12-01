import os

import cv2

import numpy as np

from scipy.signal import find_peaks, argrelextrema, find_peaks_cwt, argrelmin

import pandas as pd
import traceback


from scipy.signal import savgol_filter

import math

import ruptures as rpt

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



def compute_video_radial_profiles(video_path, center_data, start_frame=0, end_frame=None, ignored_frames=None):
    """
    Compute radial profiles for all frames in a video
    """
    if ignored_frames is None:
        ignored_frames = []
        
    radial_profiles = []
    valid_frames = []
    
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".png")])
    if end_frame is None:
        end_frame = len(frame_files)
    
    frame_files = frame_files[start_frame:end_frame]
    
    for i, filename in enumerate(frame_files):
        frame_idx = start_frame + i
        
        # Skip ignored frames
        if should_ignore_frame(frame_idx, ignored_frames):
            print(f"Skipping ignored frame {frame_idx}")
            continue
            
        try:
            file_path = os.path.join(video_path, filename)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Get center for this frame - use i instead of frame_idx for center_data indexing
            # since center_data is already sliced to [start_frames:end_frames]
            ccol = center_data['y'].iloc[i]  # Use i for the sliced dataframe
            crow = center_data['x'].iloc[i]
            
            # Compute radial profile
            _, prof = radial_profile(image, (crow, ccol), 
                                   int(math.sqrt(math.pow(image.shape[0]/2,2)+math.pow(image.shape[1]/2,2))))
            prof = savgol_filter(prof, window_length=10, polyorder=3)
            radial_profiles.append(prof)
            valid_frames.append(frame_idx)
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            traceback.print_exc()
            # Append a placeholder or skip? Let's skip for now
            continue
    
    return radial_profiles, valid_frames

def detect_radius_trends(radial_profiles, min_segment_length=2):
    """
    Detect trends in radii across frames using changepoint detection
    Returns: trend_segments - list of (start_frame, end_frame, trend_type)
    """
    # Extract dominant peak positions for each frame
    peak_positions = []
    peak_values = []
    
    for prof in radial_profiles:
        try:
            peaks = find_peaks_cwt(prof, widths=np.arange(0.5, 2))
            if len(peaks) > 0:
                peak_vals = [prof[peak] for peak in peaks]
                dominant_peak = peaks[np.argmax(peak_vals)]
                peak_positions.append(dominant_peak)
                peak_values.append(prof[dominant_peak])
            else:
                peak_positions.append(np.nan)
                peak_values.append(np.nan)
        except:
            peak_positions.append(np.nan)
            peak_values.append(np.nan)
    
    # Check if we have enough valid data
    valid_peaks = [p for p in peak_positions if not np.isnan(p)]
    if len(valid_peaks) < min_segment_length * 2:
        print(f"Warning: Only {len(valid_peaks)} valid peaks found, using fallback trend detection")
        return fallback_trend_detection(peak_positions, min_segment_length), peak_positions
    
    # Interpolate missing values
    peak_positions = pd.Series(peak_positions).interpolate().values
    
    try:
        # Use simpler changepoint detection that's more robust
        algo = rpt.Binseg(model="l2").fit(peak_positions.reshape(-1, 1))
        change_points = algo.predict(n_bkps=min(5, len(peak_positions)//10))  # Limit number of breakpoints
        
        # Convert to segments and classify trends
        segments = []
        change_points = [0] + change_points + [len(peak_positions)]
        
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            
            if end - start < min_segment_length:
                continue
                
            segment_peaks = peak_positions[start:end]
            if len(segment_peaks) < 2:
                trend = "stable"
            else:
                # Linear regression to determine trend
                x = np.arange(len(segment_peaks))
                slope = np.polyfit(x, segment_peaks, 1)[0]
                
                # Use relative threshold based on data range
                data_range = np.max(segment_peaks) - np.min(segment_peaks)
                threshold = data_range / len(segment_peaks) * 0.5
                
                if slope > threshold:
                    trend = "expanding"
                elif slope < -threshold:
                    trend = "shrinking"
                else:
                    trend = "stable"
            
            segments.append((start, end, trend))
        
        return segments, peak_positions
        
    except Exception as e:
        print(f"Changepoint detection failed: {e}, using fallback")
        return fallback_trend_detection(peak_positions, min_segment_length), peak_positions

def fallback_trend_detection(peak_positions, min_segment_length):
    """Fallback trend detection when changepoint analysis fails"""
    segments = []
    window_size = min_segment_length * 2
    
    for i in range(0, len(peak_positions) - window_size, window_size//2):
        segment = peak_positions[i:i+window_size]
        if len(segment) < 2 or np.all(np.isnan(segment)):
            continue
            
        # Remove NaNs for calculation
        segment_clean = segment[~np.isnan(segment)]
        if len(segment_clean) < 2:
            continue
            
        x_clean = np.arange(len(segment_clean))
        slope = np.polyfit(x_clean, segment_clean, 1)[0]
        
        # Use relative threshold
        data_range = np.max(segment_clean) - np.min(segment_clean)
        threshold = data_range / len(segment_clean) * 0.5
        
        if slope > threshold:
            trend = "expanding"
        elif slope < -threshold:
            trend = "shriking"
        else:
            trend = "stable"
        
        segments.append((i, i+window_size, trend))
    
    # If no segments found, mark entire sequence as stable
    if not segments:
        segments.append((0, len(peak_positions), "stable"))
    
    return segments

def should_ignore_frame(frame_idx, ignored_frames_list):
    """Check if a frame should be ignored"""
    return frame_idx in ignored_frames_list

def get_current_trend(frame_idx, trend_segments):
    """
    Get the current trend for a given frame
    """
    for start, end, trend in trend_segments:
        if start <= frame_idx < end:
            return trend
    return "stable"