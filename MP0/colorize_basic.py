
"""
Basic implementation for colorizing Prokudin-Gorskii images
Using exhaustively search to perform channel alignment
"""

import cv2
import numpy as np
import math
import itertools


# Supporting functions

def pre_process(path):
    # Read the image as numpy array
    img = cv2.imread(path, -1)
    # Split channel
    height = int(len(img) / 3)
    B = img[0:height]
    G = img[height:(height*2)]
    R = img[(height*2):(height*3)]
    return B, G, R


def find_disp(window, image, base, metric):
    '''
    Funtion to find best displacements
    Inputs: image, the image to be shifted
            window, range of pixels to shift
            base, base/reference image which is unchanged
            metric: pass 'SSD' or 'NCC'
    Return: offset, list of [x_disp, y_disp]
    '''

    # Compute scores based on SSD or NCC
    ssd = lambda mat1, mat2: ((mat1 - mat2) ** 2).sum()
    ncc = lambda mat1, mat2: cv2.matchTemplate(np.float32(mat1-mat1.mean()), 
                                               np.float32(mat2-mat2.mean()), 
                                               cv2.TM_CCORR_NORMED)[0, 0]  # var must be float32
    metrics = {'SSD': ssd, 'NCC': ncc}  

    # Loop every pixel in the window to find the best disp
    best_ssd = math.inf
    best_ncc = -math.inf
    for x, y in itertools.product(range(-window, window+1), repeat=2):
        image_roll = np.roll(image, (x, y), axis=(1, 0))  
        score = metrics[metric](base, image_roll)
        
        if metric == 'SSD':
            if score < best_ssd:  # Find min
                best_ssd = score
                offset =[x, y]
        elif metric == 'NCC':
            if score > best_ncc:  # Find max
                best_ncc = score
                offset = [x, y]

    return offset


def merge(align_1, align_2, base, base_name):
    '''
    Merge BGR channels in correct order
    Inputs: align_1 & align_2, aligned images to be merged, following BGR order
            base, base/reference image
            base_name, pass 'B', 'G' or 'R'
    Return: img_merge, merged color image
    '''
    
    if base_name == 'B':
        img_merge = cv2.merge((base, align_1, align_2))
    elif base_name == 'G':
        img_merge = cv2.merge((align_1, base, align_2))
    elif base_name == 'R':
        img_merge = cv2.merge((align_1, align_2, base))
    return img_merge


def run_colorize(window, image_1, image_2, base, base_name, metric):
    # Find disp
    offset_1 = find_disp(window, image_1, base, metric)
    offset_2 = find_disp(window, image_2, base, metric)
    # Align
    align_1 = np.roll(image_1, offset_1, axis=[1, 0])
    align_2 = np.roll(image_2, offset_2, axis=[1, 0])
    # Merge
    img_final = merge(align_1, align_2, base, base_name)
    return offset_1, offset_2, img_final


# Main test function to run 

def test(photo_name, window, B, G, R, metric):
    '''
    Run test on different base/reference images
    Run test on SSD and NCC, passed to the metric field as string
    Run test on different photo (photo_name passed as string)
    '''
    
    # Base B
    offset_G, offset_R, img_B = run_colorize(window, G, R, B, 'B', metric)
    write = cv2.imwrite('{}/{}-{}-B.png'.format(photo_name, photo_name, metric), img_B)
    print('B as reference:')
    print('disp of G: ', offset_G, '; ', 
          'disp of R: ', offset_R)
    # Base G
    offset_B, offset_R, img_G = run_colorize(window, B, R, G, 'G', metric)
    write = cv2.imwrite('{}/{}-{}-G.png'.format(photo_name, photo_name, metric), img_G)
    print('G as reference: ')
    print('disp of B: ', offset_B, '; ', 
          'disp of R: ', offset_R)
    # Base R
    offset_B, offset_G, img_R = run_colorize(window, B, G, R, 'R', metric)
    write = cv2.imwrite('{}/{}-{}-R.png'.format(photo_name, photo_name, metric), img_R)
    print('R as reference: ')
    print('disp of B: ', offset_B, '; ', 
          'disp of G: ', offset_R)


# Run all tests

# Test on photo `00125v.jpg`
photo_name = '00125v'
# Load photo & split channels
B, G, R = pre_process('data/{}.jpg'.format(photo_name))
# Write un-aligned color image
write = cv2.imwrite('{}/original.png'.format(photo_name), cv2.merge((B, G, R)))

# Test
window = 10
metric = 'SSD'
print('======= Test on SSD: =======')
test(photo_name, window, B, G, R, metric)

metric = 'NCC'
print('======= Test on NCC: =======')
test(photo_name, window, B, G, R, metric)


# Test on photo `00149v.jpg`
photo_name = '00149v'
B, G, R = pre_process('data/{}.jpg'.format(photo_name))
write = cv2.imwrite('{}/original.png'.format(photo_name), cv2.merge((B, G, R)))

# Test
window = 10
metric = 'SSD'
print('======= Test on SSD: =======')
test(photo_name, window, B, G, R, metric)

metric = 'NCC'
print('======= Test on NCC: =======')
test(photo_name, window, B, G, R, metric)


# Test on photo `00153v.jpg`
photo_name = '00153v'
B, G, R = pre_process('data/{}.jpg'.format(photo_name))
write = cv2.imwrite('{}/original.png'.format(photo_name), cv2.merge((B, G, R)))

# Test
window = 5
metric = 'SSD'
print('======= Test on SSD: =======')
test(photo_name, window, B, G, R, metric)

window = 7
metric = 'NCC'
print('======= Test on NCC: =======')
test(photo_name, window, B, G, R, metric)


# Test on photo `00351v.jpg`
photo_name = '00351v'
B, G, R = pre_process('data/{}.jpg'.format(photo_name))
write = cv2.imwrite('{}/original.png'.format(photo_name), cv2.merge((B, G, R)))

# Test
window = 11
metric = 'SSD'
print('======= Test on SSD: =======')
test(photo_name, window, B, G, R, metric)

metric = 'NCC'
print('======= Test on NCC: =======')
test(photo_name, window, B, G, R, metric)


# Test on photo `00398v.jpg`
photo_name = '00398v'
B, G, R = pre_process('data/{}.jpg'.format(photo_name))
write = cv2.imwrite('{}/original.png'.format(photo_name), cv2.merge((B, G, R)))

# Test
window = 15
metric = 'SSD'
print('======= Test on SSD: =======')
test(photo_name, window, B, G, R, metric)

metric = 'NCC'
print('======= Test on NCC: =======')
test(photo_name, window, B, G, R, metric)


# Test on photo `01112v.jpg`
photo_name = '01112v'
B, G, R = pre_process('data/{}.jpg'.format(photo_name))
write = cv2.imwrite('{}/original.png'.format(photo_name), cv2.merge((B, G, R)))

# Test
window = 10
metric = 'SSD'
print('======= Test on SSD: =======')
test(photo_name, window, B, G, R, metric)

metric = 'NCC'
print('======= Test on NCC: =======')
test(photo_name, window, B, G, R, metric)
