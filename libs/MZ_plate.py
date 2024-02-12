# -*- coding: utf-8 -*-
"""
Many_Zebrafish: Plate Class

@author: kampff
"""
# Import libraries
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import profile_line

# Import local modules
from MZ_fish import Fish

# Plate Class
class Plate:
    def __init__(self, _name):
        self.name = _name
        self.intensity = []
        self.wells = []
        for i in range(96):
            self.wells.append(Fish())
        return

    def clear(self):
        self.intensity = []
        for fish in self.wells:
            fish.x = []
            fish.y = []
            fish.heading = []
            fish.area = []
            fish.motion = []

    def save(self, output_folder, start_frame=0, end_frame=-1):
        plate_folder = output_folder + '/plate'
        if not os.path.exists(plate_folder):
            os.makedirs(plate_folder)
        if end_frame == -1:
            end_frame = start_frame + len(self.intensity) - 1
        num_frames = end_frame - start_frame + 1

        # Prepare numpy arrays
        intensity = np.array(self.intensity)
        xs = np.empty(96, object)
        ys = np.empty(96, object)
        headings = np.empty(96, object)
        areas = np.empty(96, object)
        motions = np.empty(96, object)
        threshold_backgrounds = np.empty(96, object)
        threshold_motions = np.empty(96, object)
        backgrounds = np.empty(96, object)
        stacks = np.empty(96, object)
        stack_sizes = np.empty(96, object)
        stack_counts = np.empty(96, object)
        since_stack_updates = np.empty(96, object)
        previouses = np.empty(96, object)
        previous_motions = np.empty(96, object)
        for i,fish in enumerate(self.wells):
            xs[i] = np.array(fish.x, dtype=np.float32)
            ys[i] = np.array(fish.y, dtype=np.float32)
            headings[i] = np.array(fish.heading, dtype=np.float32)
            areas[i] = np.array(fish.area, dtype=np.float32)
            motions[i] = np.array(fish.motion, dtype=np.float32)
            threshold_backgrounds[i] = fish.threshold_background
            threshold_motions[i] = fish.threshold_motion
            backgrounds[i] = fish.background
            stacks[i] = fish.stack
            stack_sizes[i] = fish.stack_size
            stack_counts[i] = fish.stack_count
            since_stack_updates[i] = fish.since_stack_update     
            previouses[i] = fish.previous
            previous_motions[i] = fish.previous_motion
        
        # Save plate
        plate_path = plate_folder + f'/{self.name}_{start_frame}_{end_frame}.npz'
        np.savez(plate_path,
                 intensity=intensity,
                 xs=xs, 
                 ys=ys, 
                 headings=headings, 
                 areas=areas, 
                 motions=motions, 
                 threshold_backgrounds=threshold_backgrounds,
                 threshold_motions=threshold_motions,
                 backgrounds=backgrounds,
                 stacks=stacks, 
                 stack_sizes=stack_sizes, 
                 stack_counts=stack_counts, 
                 since_stack_updates=since_stack_updates, 
                 previouses=previouses, 
                 previous_motions=previous_motions,
                 )
        return

    def load(self, output_folder, start_frame=0, end_frame=-1):
        # Load ROIs
        roi_path = output_folder + '/roi.csv'
        self.load_rois(roi_path)

        # Prepare plate path
        plate_folder = output_folder + '/plate'
        plate_paths = sorted(glob.glob(plate_folder + '/*.npz'))
        if end_frame == -1:
            if len(plate_paths) == 1:
                plate_path = plate_paths[0]
                frame_range = plate_path[:-4].split('_')[-2:]
                start_frame = int(frame_range[0])
                end_frame = int(frame_range[1])
            else:
                print("Incompatible frame range for available plate files.")
                exit(-1)
        else:
            plate_path = plate_folder + f'/{self.name}_{start_frame}_{end_frame}.npz'
        num_frames = end_frame - start_frame + 1

        # Load plate
        plate_file = np.load(plate_path, allow_pickle=True)
        self.intensity = plate_file['intensity']
        for i, fish in enumerate(self.wells):
            fish.x = plate_file['xs'][i]
            fish.y = plate_file['ys'][i]
            fish.heading = plate_file['headings'][i]
            fish.area = plate_file['areas'][i]
            fish.motion = plate_file['motions'][i]
            fish.threshold_background = plate_file['threshold_backgrounds'][i]
            fish.threshold_motion = plate_file['threshold_motions'][i]
            fish.background = plate_file['backgrounds'][i]
            fish.stack = plate_file['stacks'][i]
            fish.stack_size = plate_file['stack_sizes'][i]
            fish.stack_count = plate_file['stack_counts'][i]
            fish.since_stack_update = plate_file['since_stack_updates'][i]
            fish.previous = plate_file['previouses'][i]
            fish.previous_motion = plate_file['previous_motions'][i]
            #print(f'Loaded fish {i} into plate ({start_frame} to {end_frame})')

        return

    def init_backgrounds(self, background):
        for fish in self.wells:
            fish.init_background(background)
        return

    def find_rois(self, image_path, blur, output_folder):
        # Load image, convert to grayscale, and blur
        image  = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(blur,blur),blur // 2, blur // 2)

        # Find extremes
        start_x, end_x, start_y, end_y = find_extremes(gauss)
        arena_width = end_x - start_x
        arena_height = end_y - start_y
        roi_width = arena_width / 12.0
        roi_height = arena_height / 8.0

        # Create display image and draw arena
        display = np.copy(image)
        display = cv2.rectangle(display, (start_x, start_y), (end_x, end_y), (0,255,255), 3)

        # Find best internal dividing vertical lines
        vert_dividers = [((start_x, start_y),(start_x, end_y))]
        for col in range(1,12):
            internal_x = int(start_x + (col * roi_width))
            x1_range = np.arange(internal_x-10, internal_x+10)
            y1_range = np.array([start_y])
            x2_range = np.arange(internal_x-10, internal_x+10)
            y2_range = np.array([end_y])
            start, end = minimal_line(gauss, x1_range, y1_range, x2_range, y2_range)
            vert_dividers.append((start,end))
            print(f'Verts: {col}: {start},{end}')
            display = cv2.line(display, (start), (end), (0,0,255), 1)
        vert_dividers.append(((end_x, start_y),(end_x, end_y)))

        # Find best internal dividing horizontal lines
        horz_dividers = [((start_x, start_y),(end_x, start_y))]
        for row in range(1,8):
            internal_y = int(start_y + (row * roi_height))
            x1_range = np.array([start_x])
            y1_range = np.arange(internal_y-10, internal_y+10)
            x2_range = np.array([end_x])
            y2_range = np.arange(internal_y-10, internal_y+10)
            start, end = minimal_line(gauss, x1_range, y1_range, x2_range, y2_range)
            horz_dividers.append((start,end))
            print(f'Horzs: {row}: {start},{end}')
            display = cv2.line(display, (start), (end), (0,0,255), 1)
        horz_dividers.append(((start_x, end_y),(end_x, end_y)))

        # Find intersection points and create ROIs
        count = 0
        for row in range(8):
            for col in range(12):
                top = horz_dividers[row]
                bottom = horz_dividers[row+1]
                left = vert_dividers[col]
                right = vert_dividers[col+1]
                ul = find_intersection(left, top)
                lr = find_intersection(right, bottom)
                self.wells[count].set_roi(ul, lr)
                count = count + 1

        # Draw ROI intersections
        for fish in self.wells:
            display = cv2.circle(display, fish.ul, 3, (0,255,0), 1)
            display = cv2.circle(display, fish.lr, 3, (0,255,0), 1)

        # Store ROI image
        ret = cv2.imwrite(output_folder + r'/roi.png', display)
        ret = cv2.imwrite(output_folder + r'/blurred.png', gauss)

        return

    def load_rois(self, roi_path):
        roi_data = np.genfromtxt(roi_path, delimiter=',')
        for i, fish in enumerate(self.wells):
            roi_line = roi_data[i]
            ul = (roi_line[0],roi_line[1])
            lr = (roi_line[2],roi_line[3])
            fish.set_roi(ul, lr)
        return

    def save_rois(self, roi_path):
        roi_file = open(roi_path, 'w')
        for fish in self.wells:
            ret = roi_file.write(f'{fish.ul[0]},{fish.ul[1]},{fish.lr[0]},{fish.lr[1]}\n')
        return

# Utilities for working with 96-well plate experiments

# Find extremes
def find_extremes(image):
    threshold = 3
    width = image.shape[1]
    height = image.shape[0]
    vert_projection = np.mean(image, axis=0)
    horz_projection = np.mean(image, axis=1)
    
    # Find extremes
    count = 0
    for i, p in enumerate(vert_projection):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                start_x = i-3
                break
        else:
            count = 0
    count = 0
    for i, p in enumerate(reversed(vert_projection)):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                end_x = width-i+3
                break
        else:
            count = 0
    count = 0
    for i, p in enumerate(horz_projection):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                start_y = i-3
                break
        else:
            count = 0
    count = 0
    for i, p in enumerate(reversed(horz_projection)):
        if p >= threshold:
            count = count + 1
            if count >= 3:
                end_y = height-i+3
                break
        else:
            count = 0
    return start_x, end_x, start_y, end_y


# Find minimal line (brute force)
def minimal_line(image, x1_range, y1_range, x2_range, y2_range):
    minima = 9999999999
    count = 0
    for x1 in x1_range:
        for y1 in y1_range:
            for x2 in x2_range:
                for y2 in y2_range:
                    start = (x1,y1)
                    end = (x2,y2)
                    intensity = line_intensity(image, start, end)
                    count = count + 1
                    #print(f'{y1}, {y2}: {intensity} - {count}')
                    if intensity < minima:
                        best_start = start
                        best_end = end
                        minima = intensity
    return best_start, best_end

# Compute intensity along a line
def line_intensity(image, start_xy, end_xy):
    start_rc = start_xy[::-1]
    end_rc = end_xy[::-1]
    pixels = profile_line(image, start_rc, end_rc)
    return np.mean(pixels)

# Find intersection of two lines
def find_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

# FIN
