#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time as time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial as spatial
import skimage
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from pylab import gca
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import geometric_transform
from skimage import measure, morphology
from skimage.segmentation import flood_fill
from skimage.transform import resize
from sklearn import cluster
from tqdm import tqdm

from analyze_pore_samples.plotting_utils import (animate_data, frame_tick,
                                                 set_axes_equal)
from analyze_pore_samples.polar_cartesian_convert import (linear_polar,
                                                          map_pixel,
                                                          polar_linear)
from utils import *

rootdir =  './sample_dataset'
subfolders = [ f.path for f in os.scandir(rootdir) if f.is_dir() ]


voxelsizes = np.loadtxt('./sample_dataset/VoxelSize.txt', skiprows = 1, dtype= 'str')
voxelsizes = dict(voxelsizes)





'''
legend:
default location : upper left
default fontsize: 8
Frame is always off
'''


def load_data(folder_index, num=100, start=0, shape = None, verbose = 0):
    '''
    Folder index: [int] Index corresponding to the folder of .tiff images to be analyzed
    num: [int] Number of frames to analyze
    start: [int] Starting frame of analysis. Frames "start" until "start + num" will be analyzed.
    shape: [int] default None -- desired shape of cross-section image in pixels, will be cropped to this shape
    verbose: [int] 
    '''
    subfolder = subfolders[folder_index]
    angle_dict= {}
    for key in ['B05-04', 'B05-05', 'B05-06']:
        angle_dict[key] = 0.662
    for key in ['F26-01', 'F26-02', 'F26-03', 'F26-04', 'F26-05']:
        angle_dict[key] = 164.188
    for key in ['F26-06', 'F26-07', 'F26-08', 'F26-09', 'F26-10']:
        angle_dict[key] = -83.302
    if verbose > 0:
        print(" Currently reading: " + str(subfolder))
    pictures = os.listdir(subfolder)
    pictures.sort()
    
    # Get the dimensions of the data from folder name
    picname = pictures[0]
    voxel_name = [s for s in picname.split('_')[0].split('H') if s]
    im_test = Image.open(os.path.join(subfolder, picname))
    image_test = np.array(im_test)
    imstack = np.zeros((image_test.shape[0], image_test.shape[1], num), 'uint8')
    diameterx = []
    diametery = []
    xmaxs = []
    xmins = []
    ymaxs = []
    ymins = []
    xmin_test = np.min(np.where(image_test == 255)[0])
    xmax_test = np.max(np.where(image_test == 255)[0])
    ymin_test = np.min(np.where(image_test == 255)[1])
    ymax_test = np.max(np.where(image_test == 255)[1])
    voxelsize = float((voxelsizes)['H-'+voxel_name[0]])
    if verbose > 1:
        print(subfolder, 'SUBFOLDER', voxelsize, folder_index, voxelsize)
    for count, picture in tqdm(enumerate(pictures)):
        if count < start:
            continue
        try:
            im = Image.open(os.path.join(subfolder, picture))
            if verbose> 1:
                print(os.path.join(subfolder, picture))
            image = np.array(im, dtype = 'uint8')
            new_image = rotate(image, -angle_dict[voxel_name[0]], reshape = False, order = 0)
            image = np.array(new_image, dtype = 'uint8')

            imstack[:, :, count - start] = new_image  # [:,:]
        except:
            breakpoint()
        xmin = np.min(np.where(image == 255)[0])
        xmax = np.max(np.where(image == 255)[0])
        ymin = np.min(np.where(image == 255)[1])
        ymax = np.max(np.where(image == 255)[1])

        xcenter = image.shape[0]//2
        ycenter = image.shape[1]//2
   

        ymaxs.append(ymax)
        ymins.append(ymin)
        xmins.append(xmin)
        xmaxs.append(xmax)

        diameterx.append(xmax-xmin)
        diametery.append(ymax-ymin)
     
        if count >= num + start - 1:
            break

    if shape is None:
        return voxelsize, imstack[np.min(xmins):np.max(xmaxs), np.min(ymins):np.max(ymaxs), :], voxel_name, angle_dict[voxel_name[0]]
    else:

        if shape > imstack.shape[0] or shape > imstack.shape[1]:
            return voxelsize, imstack, voxel_name, angle_dict[voxel_name[0]]
        else:
            return voxelsize, imstack[xcenter-shape//2:xcenter+shape//2, ycenter-shape//2:ycenter+shape//2, :], voxel_name, angle_dict[voxel_name[0]]





def get_boundary(imstack):
    '''
    Return boundary given segmented image.

    Parameters
    ----------
    imstack : NumPy array, with boundary listed as pixels with intensity 255.

    Returns
    -------
    boundary_imstack: Binary NumPy array indicating boundary pixels with 1, and other pixels with 0.
    '''
   
    boundary_imstack = np.copy(imstack)
    boundary_imstack[imstack != 255] = 0
    boundary_imstack[imstack == 255] = 1
    if np.sum(boundary_imstack) == 0:
        breakpoint()
    return(boundary_imstack)
def replace_boundary(imstack):


    '''
    Return rescaled segmented image with pore and boundary phases set to 1.

    Parameters
    ----------
    imstack : NumPy array, with boundary listed as pixels with intensity 255 and pores listed with intensity 159.

    Returns
    -------
    newstack: Binary NumPy array indicating pore pixels with 1, and all other pixels with 0.
    
    boundary_imstack: Binary NumPy array indicating boundary pixels with 1, and all other pixels with 0.
    
    '''
   
    boundary_imstack = np.copy(imstack)
    boundary_imstack[imstack != 255] = 0
    boundary_imstack[imstack == 255] = 1
    newstack = np.copy(imstack)
    newstack[imstack  == 255] = 0
    newstack[imstack == 159] = 1
    newstack = np.array(newstack, dtype = 'uint8')
    return newstack, boundary_imstack  
def extract_pores(imstack):
    '''
    Return list of RegionProps objects representing the pores present in the input array.

    Parameters
    ----------
    imstack : NumPy array, with boundary listed as pixels with intensity 255 and pores listed with intensity 159.

    Returns
    -------
    props: List of RegionProps objects representing the pores present in imstack.
    
    im: Labeled binary image, with each pore represented by a group of pixels with a unique intensity.
    
    boundary_imstack: Binary NumPy array indicating boundary pixels with 1, and all other pixels with 0.
    
    '''

    boundary_imstack = np.copy(imstack)
    boundary_imstack[imstack != 255] = 0
    boundary_imstack[imstack == 255] = 1
    newstack = np.copy(imstack)
    newstack[imstack  == 255] = 0
    newstack[imstack == 159] = 1
    newstack = np.array(newstack, dtype = 'uint8')                   
    im = measure.label(newstack[:,:,:])
    props = skimage.measure.regionprops(im[:,:,:])
    return props, im, boundary_imstack
def save_data(data, folder, name, count  = 0):
    '''
    Return RegionProps objects representing the pores present in the input array.

    Parameters
    ----------
    imstack : NumPy array, with boundary listed as pixels with intensity 255 and pores listed with intensity 159.

    Returns
    -------
    props: Binary NumPy array indicating pore pixels with 1, and all other pixels with 0.
    
    boundary_imstack: Binary NumPy array indicating boundary pixels with 1, and all other pixels with 0.
    
    '''
    name = 'threechannel'
    f = h5py.File(folder+'/'+str(name)+'_'+str(count)+'.hdf5', 'w')
    f.create_dataset('data', data=data, dtype='i8', compression='gzip')
    f.close()
def create_data_channels(im, props, k = 1, pore_list_attr = [], save = False):
    '''
    Returns list of pore attributes, saves information about each pore.

    Parameters
    ----------
    im: Labeled binary image, with each pore represented by a group of pixels with a unique intensity.
    props:  List of RegionProps objects representing the pores present in im.
    k: The current folder index that is being processed.
    pore_list_attr: The previous list of pore_list_attributes that the attributes extracted here should be appended to. (Default: [])
    save: Whether to save the pore attributes extracted here. (Default: False)


    Returns
    -------
    pore_list_attr: A nested list of pore volumes, start indices, anisotropies, orientations, phis
    '''
    im_channels = im
    previous_count = len(pore_list_attr)
    pores_cross = np.unique(im_channels[:,:,0]) + 1

    output_dir = './analyze_pore_samples/results/individual_pore_samples/partsample' + str(k) + '/'
    os.makedirs(output_dir, exist_ok = True)
    
    
    size = 32
    count = 0
    name = 'pore_original'
    for p_idx in np.arange(len(props)):
        inertia_eigval =  props[p_idx].inertia_tensor_eigvals
        inertia =  props[p_idx].inertia_tensor
        maxeig = np.argmax(inertia_eigval)
        eigvec = np.linalg.eig(props[p_idx].inertia_tensor)[1]
        eigvals = np.linalg.eig(props[p_idx].inertia_tensor)[0]
        anis = 1 - np.min(eigvals)/np.max(eigvals)
        max_0 = np.max([prop.area for prop in props])
        z_0 = np.max([prop.centroid[2] for prop in props])

        maxvector = eigvec[:, maxeig]
        orientation = angle_between(maxvector, np.array([0,0,1]))
        phi = angle_between(maxvector, np.array([0,1,0]))

        pore_3d = np.zeros((size*2, size*2, size*2))
        xdist = props[p_idx].slice[0].stop - props[p_idx].slice[0].start
        ydist = props[p_idx].slice[1].stop - props[p_idx].slice[1].start
        zdist = props[p_idx].slice[2].stop - props[p_idx].slice[2].start
        if save == True:
            try:

                pore_3d[size - xdist//2: size+(xdist-xdist//2), size-ydist//2:size+(ydist- ydist//2), size-zdist//2:size+(zdist-zdist//2)]  = np.array(props[p_idx].image, dtype = 'float')
                f = h5py.File(str(output_dir)+'/'+str(name)+'_'+str(count+previous_count)+'.hdf5', 'w')
                f.create_dataset('data', data=pore_3d, dtype='i8', compression='gzip')
                f.close()
                
                count = count + 1
                if count % 100 ==0:
                    print('Saving pores ', count)
                pore_list_attr.append([props[p_idx].area, props[p_idx].slice[2].start, anis, orientation, phi])
            except Exception as e:
                print(props[p_idx].area, e)
                continue


        
        if p_idx % 50 == 0:
            print("Pore processed: " + str(p_idx) + " pores out of " + str(len(props)) + " pores...")
      

    pore_matrix =np.array(pore_list_attr)
    pore_list_dict = {'Volume': pore_matrix[:,0], 'z_start': pore_matrix[:,1], 'anisotropy': pore_matrix[:,2], 'orientation': pore_matrix[:,3], 'phi': pore_matrix[:,4]}
    df = pd.DataFrame.from_dict(pore_list_dict)
    if save:
        np.savetxt(output_dir+'pore_matrix', pore_matrix)
    return pore_list_attr

def analyze_boundaries(k, limit, index = 0, data_info = None):
    '''
    Returns 2-D projection of the surface, following conversion to polar coordinates.

    Parameters
    ----------
   
    k: The current folder index that is being processed.
    limit: The ending frame index to be loaded for processing.
    index: The starting frame to be loaded for processing. limit - index frames will be processed in this program
    data_info: Pre-loaded CT data to use for processing. Structure: data_info[0] is the voxel conversion to microns, data_info[1] is the 3D array defining the part segment.


    Returns
    -------
    profile_2d: The 2-D projection of the surface. 
    '''
    print("Processing boundaries ...")
    os.makedirs('analyze_pore_samples/results', exist_ok = True)
    window = limit
    profile_2d = []
    if data_info is None:
        voxelsize, imstack_test, _, _  = load_data(k, num = limit, start = index)
    elif limit == data_info[1].shape[2]:
        voxelsize, imstack_test = data_info
    else: 
        voxelsize, imstack_test, _, _  = load_data(k, num = limit, start = index)
    # while index < limit:
        # imstack = imstack_test[:,:, index:window+index]
        # index += window

    before_time = time.time()
    boundary_imstack = get_boundary(imstack_test)
    if np.sum(boundary_imstack) == 0:
        print("Boundary imstack is empty")
        breakpoint()
    oldtime = time.time()
    for sample in range(imstack_test.shape[2]):
        if sample % 500 == 0:
            print("processed {} samples out of {}".format(sample, imstack_test.shape[2]))
        center = boundary_imstack[:,:, sample].shape[1]//2
        polar= linear_polar(boundary_imstack[:,:, sample]) 
        line = np.argmax(polar, axis = 0)

        if 0 in line:
            # breakpoint()
            dilated = morphology.dilation(polar)    
            skeleton = morphology.skeletonize(dilated)
            polar = skeleton
            line = np.argmax(polar, axis = 0)
        if 0 in line:
            idxs = np.where(line == 0)[0]
            # breakpoint()
            for case in idxs:
                indices = np.where(line)[0]
                if len(np.where(line)[0]) == 0:
                    print("Empty line")
                    # breakpoint()
                nearest = indices[np.argmin(np.abs(np.where(line)[0] - case))]
                min_edge = np.max([0, nearest-5])
                max_edge = np.min([len(line), nearest+5])
                neighborhood = line[min_edge:max_edge]
                line[case] = int(np.mean(neighborhood[neighborhood>0]))
        
        angs = np.array([i*2*np.pi/polar.shape[1] for i in range(polar.shape[1])])
        
        profile_2d.append(line)
    after_time = time.time()
    print(after_time-before_time, "time in seconds")
    profile_2d = np.array(profile_2d)

    profile_2d[profile_2d== 0 ] = np.mean(profile_2d[profile_2d > 0 ])
    plt.clf()
    plt.yticks(np.arange(3000)[::500], np.round(np.arange(3000)*voxelsize)[::500])
    plt.xticks(np.arange(profile_2d.shape[1])[::500], np.round(angs[::500]/np.pi,1))
    plt.xlabel(r'$\theta$ [$\pi$ rad]')
    plt.ylabel(r'z [$\mu$ m]')
    ax = plt.gca()
    img = plt.imshow(profile_2d*voxelsize - np.mean(profile_2d*voxelsize), cmap = 'binary' )

    np.save('./make_surface/original_profilometry_' + str(k) + '.npy', profile_2d)
    np.save('./analyze_pore_samples/results/profilometry_' + str(k) + '.npy', profile_2d)

    plt.title('Part sample: ' + str(k) + ' Average  elevation: {:.1f}'.format(np.mean(np.abs(profile_2d - np.mean(profile_2d)))*voxelsize) + r'$\mu$m Maximum =  {:.1f}'.format(np.max(np.abs(profile_2d - np.mean(profile_2d)))*voxelsize) +r'$\mu$m', fontsize = 8)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label(r'elevation [$\mu$m]', labelpad = -40, y = 1.4, rotation = 0)
    plt.tight_layout()
    plt.savefig('./analyze_pore_samples/results/profilometry_' + str(k) + '.png')
    plt.clf()
    plt.close('all')

    return profile_2d
def pore_extract(pore_dataset):
    '''
    Returns: None

    Computes pore metrics, saves 3-D binary representation of each pore as an HDF5 file, given a list of RegionProps objects representing pores. 

    Parameters
    ----------

    pore_dataset: List of RegionProps objects representing pores. 


    Returns
    -------
    None.
    '''
    count = 0
    size = 32
    output_dir = str(size*2) + '_full_labeled_largepore_threephase_3D'
    name = 'tiff_threephase'
    os.makedirs(output_dir, exist_ok=True)
    stats = compute_statistics(pore_dataset, voxelsize = 1)
    labels_inst = np.array([stats['anisotropies'], np.array(stats['vols'])**(1/3), stats['orientations'], np.arange(len(pore_dataset))]).T
    np.savetxt(str(output_dir)+'/labels'+str(name)+'_' + str(id) + '.txt', labels_inst)

    for index, pore_sample in enumerate(pore_dataset):

        pore_3d = np.zeros((size*2, size*2, size*2))
        xdist = pore_sample.slice[0].stop - pore_sample.slice[0].start
        ydist = pore_sample.slice[1].stop - pore_sample.slice[1].start
        zdist = pore_sample.slice[2].stop - pore_sample.slice[2].start

        try:

            pore_3d[size - xdist//2: size+(xdist-xdist//2), size-ydist//2:size+(ydist- ydist//2), size-zdist//2:size+(zdist-zdist//2)]  = np.array(pore_sample.image, dtype = 'float')
   
            if len(np.where(pore_3d)[0]) == 0:
                print("Pore not found")
                breakpoint()
            f = h5py.File(str(output_dir)+'/'+str(name)+'_'+str(pore_extract.counter)+'.hdf5', 'w')
            f.create_dataset('data', data=pore_3d, dtype='i8', compression='gzip')
            f.create_dataset('target', data=labels_inst[index], compression='gzip')
            f.close()
            
            count = count + 1
            pore_extract.counter   = pore_extract.counter   + 1

        except Exception as e:
            print(pore_sample.area, e)
            
            continue
        

pore_extract.counter = 0

def find_intersection(boundary_images, centroid, polar_images = None, polar_image = None, boundary =None, real_image = None, prop = None, ts_list = None, rs_list = None, plot = False):
    '''
    Returns: For the point on the boundary nearest a given pore, returns the boundary distance, standard deviation of boundary distance, 
    angle to boundary and normalized distance from the pore to the center of the part.


    Parameters
    ----------

    boundary_images: list of boundary images, converted to polar co-ordinates.
    centroid: Co-ordinates of the center of the pore used for analysis
    polar_images: list of segmented images, containing both pore and boundary material, converted to polar co-ordinates
    boundary: list of boundary images, kept in cartesian coordinate frame
    real_image: list of segmented images, containing both pore and boundary material, in original cartesian co-ordinates
    ts_list: Angle discretization of polar co-ordinate conversion
    rs_list: Radial discretization of polar co-ordinate conversion.



    Returns
    -------
    boundary_radius: Distance from center, to closest point on the surface from the given pore.
    std_dev: Standard deviation of the area on the surface nearest the pore, represents the local roughness
    angle: Angle from center to closest point from the given pore to the surface.
    pore_distance: Distance from center to given pore.
    '''
    z_idx = int(centroid[2])
    polar_boundary = boundary_images[z_idx]
    ts = ts_list[0]
    rs = rs_list[0]
    polar_images = polar_images[z_idx]
    center_x = real_image.shape[0]//2
    center_y = real_image.shape[1]//2
    slope = (centroid[1] - center_y)/(centroid[0]  - center_x + 1e-6)
    x_trial = np.linspace(0, real_image.shape[0]-1, int(real_image.shape[0]*1.5))
    y = slope*x_trial -slope*center_x + center_y

    y_int = np.array(y, dtype = 'int')
    y_int[y_int > real_image.shape[1] - 1 ] = real_image.shape[1] - 1
    y_int[y_int < 0 ] =0
    x_int = np.array(x_trial, dtype = 'int')
    collision = np.where(boundary[x_int,y_int])[0]
    if len(collision) == 0:
        x_trial = np.linspace(0, real_image.shape[0]-1, int(real_image.shape[0]*15))
        y = slope*x_trial -slope*center_x + center_y
        y_int = np.array(y, dtype = 'int')
        y_int[y_int > real_image.shape[1] - 1 ] = real_image.shape[1] - 1
        y_int[y_int < 0 ] =0
        x_int = np.array(x_trial, dtype = 'int')
        collision = np.where(boundary[x_int,y_int])[0]
    pixel = map_pixel(int(centroid[0]), int(centroid[1]), real_image)
    angle = (2*np.pi+ np.arctan2(centroid[0]-center_x, centroid[1] - center_y))%(2*np.pi)
    idx_angle = np.argmin(np.abs(ts - angle))
    angle_pore = ts[idx_angle]
    radii = []
    radii_center = []
    try:
        nearest_idx_angle  = np.where(polar_boundary)[1][np.argmin(np.abs(np.where(polar_boundary)[1] - idx_angle))]
        radius_range = np.where(polar_boundary[:,nearest_idx_angle])[0]
    except:
        breakpoint()
    if len(radius_range) == 0:
        radius_range = np.where(polar_boundary[:, np.max([0, idx_angle - 15]):idx_angle+15])[0]
    elif len(radius_range) == 0:
        radius_range = [np.mean(np.where(polar_boundary[:, np.max([0, idx_angle - 50]):idx_angle+50])[0])]

    radii.extend((pixel[0]- radius_range)/(radius_range))
    
    test = np.array(np.copy(polar_boundary) > 0, dtype = 'int')
    test[:,idx_angle-50:idx_angle+50] = test[:,idx_angle-50:idx_angle+50]*500


    left_bound = np.max([0, idx_angle-25])
    right_bound = np.min([len(ts) - 1, idx_angle+25])
    extent_axial = int(np.round((ts[right_bound] - ts[left_bound])*np.mean(radius_range)))
    left_bound_z = np.max([0,z_idx - extent_axial//2])
    right_bound_z = np.min([z_idx + extent_axial//2, len(boundary_images)-1])

    patch =np.argmax(np.array(boundary_images)[left_bound_z:right_bound_z,:, left_bound:right_bound],axis = 1)#[:, :,idx_angle-50:idx_angle+50], axis = 0))
    patch = patch[patch > 0]
    patch = patch[patch < boundary_images[0].shape[0]-2]
    std_dev = np.std(patch)
    boundary_radius = np.min(radii)
    pore_distance = np.min(pixel[0]/radius_range)
    return boundary_radius, std_dev, angle, pore_distance
    # breakpoint()

def process_images(k, dict_properties = None, save = True, save_pores = False,data_info = None, frame_window = 200, shift = 100):
    '''
    Returns: Updated dictionary of pore properties, list of pores found in dataset.

    Parameters
    ----------

    k: Index of original folder used for analysis.
    dict_properties: Pore properties stored in a dictionary, from a previous iteration
    save: whether to save the probability matrices calculated during analysis
    save_pores: whether to save the pores extracted during analysis
    data_info: Allows for the specification of a specific part to be used
    frame_window: Specifies the length of the processing window used for analysis. 
    shift: Specifies the shift of the processing window between iterations of analysis.




    Returns
    -------
    dict_properties: Pore properties stored in a dictionary,
    pores_total: A list of regionprops objeccccts, representing every pore found in the segment used for analysis. 
    '''

    os.makedirs('analyze_pore_samples/results', exist_ok = True)

   
    index = 0
    if dict_properties == None:
        total_anisotropies = []
        total_x = []
        total_y = []
        total_orientations = []
        total_z = []
        total_surf_dist  = []
        total_maj = []
        total_min = []
        total_vols = []
        anisotropy_zs = []
        orientations_zs = []
        vols_zs = []
        total_phis = []
        surf_dist = []
        surf_angles = []
        rough  = []
        x_locs_zs = []
        y_locs_zs = []
        total_surf_angles = []

    
    else:
        total_anisotropies = dict_properties['anisotropies']
        total_x =  dict_properties['x_locs']
        total_y =  dict_properties['y_locs']
        total_orientations =  dict_properties['orientations']
        total_z =  dict_properties['z_locs']
        total_maj =  dict_properties['maj_axis_l']
        total_vols =  dict_properties['vols']
        total_phis = dict_properties['phis']
        total_surf_dist = dict_properties['surf_dist']
        total_surf_angles = dict_properties['surf_angles']
        total_surf_angles = dict_properties['surf_angles']

    all_stats = [total_x, total_y,total_z,total_anisotropies, total_phis, total_orientations, total_vols, total_surf_dist, total_surf_angles, surf_dist]
    strings = ['x centroid', 'y centroid', 'z centroid', 'Anisotropy', 'Orientation', 'Volume', 'surf_dist', 'surf_angles', 'phis']
    pore_count = 0

    maj_axis_l_zs = []
    min_axis_l_zs = []
    if data_info is None:
        limit = len(os.listdir(subfolders[k]))
        voxelsize, imstack_all, _, _  = load_data(k, num = limit, start = index)
    else:
        voxelsize, imstack_all = data_info
        limit =  data_info[1].shape[2]

    pores_total =[]
    while index < limit:
        print("Processing pores, index = " + str(index) + " out of " + str(limit) + " ..." )
        # print(index)
        num = np.min([frame_window, limit - index])

        imstack = np.copy(imstack_all[:,:, index:index + frame_window])

        if imstack.shape[2] > 0:
            props, im, im_orig = extract_pores(imstack)
        else:
            print("finished with part sample")
            return dict_properties
        polar_images_boundary = []
        polar_images = []
        polar_ts = []
        polar_rs = []
        # for sample in range()
        
        for i in range(index, np.min([limit, index+frame_window])):
            polar_boundary, rs, ts, o, r, out_h, out_w = linear_polar(np.array(im_orig[:,:,i-index] > 0)*1000, verbose = 1)
            polar_images_boundary.append(np.array(polar_boundary, 'uint8'))
            if len(polar_ts) > 0:
                if np.sum(ts-polar_ts[0])!=0:
                    breakpoint()
            polar_ts = [ts]
            polar_rs = [rs]

            polar_image = linear_polar(np.array(im[:,:,i-index]> 0)*1000)
            polar_images.append(np.array(polar_image, dtype = 'uint8'))
        endframe = frame_window -1 # right boundary of section
        startframe = 0 # left boundary of section 
        pores_keep = []
        windowboundary = frame_window - shift - 1
        windowboundarypores = [prop for prop in props if windowboundary in range(prop.slice[2].start, prop.slice[2].stop)] # identify pores that were cut off before
        alreadycountedpores = [prop for prop in props if windowboundary not in range(prop.slice[2].start, prop.slice[2].stop) and prop.centroid[2] < windowboundary] # remove pores that were counted before, and were not cut off
        pores = [prop for prop in props if endframe not in range(prop.slice[2].start, prop.slice[2].stop)] # identify pores that are not cut off
        boundaries = [prop for prop in props if endframe  in range(prop.slice[2].start, prop.slice[2].stop)] # identify pores that are cut off


        if index > 0: # some of the pores have been already counted

            for prop_idx in range(len(pores)):

                alreadycount_bool  = False
                boundary_bool = False
                alreadycount_bool =  windowboundary not in range(pores[prop_idx].slice[2].start, pores[prop_idx].slice[2].stop) and (pores[prop_idx].centroid[2] < windowboundary)
                boundary_bool = endframe  in range(pores[prop_idx].slice[2].start, pores[prop_idx].slice[2].stop)
                
                if not alreadycount_bool and not boundary_bool:
                    zlength = pores[prop_idx].slice[2].stop - pores[prop_idx].slice[2].start
                    xlength = pores[prop_idx].slice[1].stop - pores[prop_idx].slice[1].start
                    ylength = pores[prop_idx].slice[0].stop - pores[prop_idx].slice[0].start
                    
                    if xlength > 1 and ylength > 1 and zlength > 1:
                        pores_keep.append(pores[prop_idx])
                        # radius, rness  = find_intersection(boundary_image = im_orig[:,:,int(pores[prop_idx].centroid[2])], centroid = pores[prop_idx].centroid,  real_image = im[:,:,  int(pores[prop_idx].centroid[2])], prop = pores[prop_idx])
                        radius, rness, angle, min_rad  = find_intersection(boundary_images = polar_images_boundary, boundary = im_orig[:,:,  int(pores[prop_idx].centroid[2])], polar_images = polar_images, centroid = pores[prop_idx].centroid,  real_image = im[:,:,  int(pores[prop_idx].centroid[2])], prop = pores[prop_idx], ts_list = polar_ts, rs_list = polar_rs)
                        
                        surf_dist.append(min_rad)
                        rough.append(rness)
                        surf_angles.append(angle)
        elif index == 0:
            for prop_idx in range(len(pores)):
               
                start_boundary_bool = False
                start_boundary_bool = 0 in range(pores[prop_idx].slice[2].start, pores[prop_idx].slice[2].stop)
                if not start_boundary_bool:
                    zlength = pores[prop_idx].slice[2].stop - pores[prop_idx].slice[2].start
                    xlength = pores[prop_idx].slice[1].stop - pores[prop_idx].slice[1].start
                    ylength = pores[prop_idx].slice[0].stop - pores[prop_idx].slice[0].start

                    if xlength > 1 and ylength > 1 and zlength > 1:
                        pores_keep.append(pores[prop_idx])
                        
                        radius, rness, angle, min_rad  = find_intersection(boundary_images = polar_images_boundary,  boundary = im_orig[:,:,  int(pores[prop_idx].centroid[2])], polar_images = polar_images, centroid = pores[prop_idx].centroid,  real_image = im[:,:,  int(pores[prop_idx].centroid[2])], prop = pores[prop_idx], ts_list = polar_ts, rs_list = polar_rs)

                        rough.append(rness)
                        surf_dist.append(min_rad)
                        surf_angles.append(angle)
        if save_pores:
            if index == 0:
                pore_list_attr = []
            pore_list_attr = create_data_channels(im, pores_keep, k =k, pore_list_attr=pore_list_attr, save = True )
        pore_count = pore_count + 1
        pores_total.extend(pores_keep)
        print("Added " + str(len(pores_keep)) + " pores, total is now " + str(len(pores_total)) + " pores" )

        stats = compute_statistics(pores_keep, voxelsize, imstack)
        anisotropy_zs.append(np.mean(stats['anisotropies']))
        orientations_zs.append(np.mean(stats['orientations']))
        vols_zs.append(np.mean(stats['vols']))

        x_locs_zs.append(np.mean(stats['x_locs']))
        y_locs_zs.append(np.mean(stats['y_locs']))
        total_x.extend(stats['x_locs'])
        total_y.extend(stats['y_locs'])
        total_surf_dist.extend(surf_dist)
        total_surf_angles.extend(surf_angles)

        total_maj.extend(stats['maj_axis_l'])
        total_vols.extend(stats['vols'])
        total_phis.extend(stats['phis'])
        total_anisotropies.extend(stats['anisotropies'])
        total_orientations.extend(stats['orientations'])
        total_z.extend(np.array(stats['z_locs'])+voxelsize*index)
        radii, angles = calc_polar(boundary = im_orig, xs = stats['x_locs'], ys = stats['y_locs'], zs = stats['z_locs'], voxelsize =voxelsize, imstack=im)
        n_bins = 30
        dict_voxel_bins_vols = {}
        dict_polar_bins_vols = {}

        dict_voxel_bins_anis = {}
        dict_polar_bins_anis = {}
        dict_voxel_bins_phis = {}
        dict_polar_bins_phis = {}
        dict_voxel_bins_theta = {}
        dict_polar_bins_theta = {}

        dict_voxel_bins_num = {}
        dict_polar_bins_num = {}
        dict_voxel_bins_orientations = {}
        dict_polar_bins_orientations = {}

        dict_voxel_bins_vols_anis = {}
        x_y_bins = np.zeros((n_bins, n_bins))

        for idx1 in range(n_bins+1):
            dict_voxel_bins_vols_anis[str(idx1)] = []
            for idx2 in range(n_bins+1):
                dict_voxel_bins_vols[str(idx1) + '_' + str(idx2)] = []
                dict_polar_bins_vols[str(idx1) + '_' + str(idx2)] = []
                dict_voxel_bins_anis[str(idx1) + '_' + str(idx2)] = []
                dict_polar_bins_anis[str(idx1) + '_' + str(idx2)] = []

                dict_voxel_bins_orientations[str(idx1) + '_' + str(idx2)] = []
                dict_polar_bins_orientations[str(idx1) + '_' + str(idx2)] = []
                dict_polar_bins_phis[str(idx1) + '_' + str(idx2)] = []

                dict_voxel_bins_phis[str(idx1) + '_' + str(idx2)] = []

                
        ybins = np.linspace(0,im.shape[1],num=n_bins,dtype= 'int')
        xbins = np.linspace(0,im.shape[0],num=n_bins,dtype= 'int')
        surf_dist_bins = np.linspace(0,1.1,num=n_bins,dtype= 'float')
        surf_angles_bins = np.linspace(0,2*np.pi,num=n_bins,dtype= 'float')

       
        # Volume
        prob_matrix_volume = np.zeros((n_bins+1, n_bins+1, n_bins))
        prob_matrix_volume_polar = np.zeros((n_bins+1, n_bins+1, n_bins))

        for dict_idx, vol in enumerate((total_vols)):
            x_coord = total_x[dict_idx]
            y_coord = total_y[dict_idx]
            z_coord = total_z[dict_idx]
            

            x_bin = np.digitize(x_coord/voxelsize, xbins)
            y_bin = np.digitize(y_coord/voxelsize, ybins)
            dict_voxel_bins_vols[str(x_bin) + '_' + str(y_bin)].append(vol)

            radius_coord = total_surf_dist[dict_idx]
            angle_coord = total_surf_angles[dict_idx]
            rad_bin = np.digitize(radius_coord, surf_dist_bins)
            angles_bin = np.digitize(angle_coord, surf_angles_bins)
            dict_polar_bins_vols[str(rad_bin) + '_' + str(angles_bin)].append(vol)

        bin_edges_vols = np.linspace((np.min(total_vols)**(1/3))/voxelsize, (np.max(total_vols)**(1/3))/voxelsize, n_bins+1)
        polar_edges_vols = np.linspace((np.min(total_vols)**(1/3))/voxelsize, (np.max(total_vols)**(1/3))/voxelsize, n_bins+1)
        for item in dict_voxel_bins_vols.keys():
            histogram,edges = np.histogram((np.array(dict_voxel_bins_vols[item])**(1/3))/voxelsize, bins = bin_edges_vols)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_volume[first_idx,second_idx,:] = histogram
        for item in dict_polar_bins_vols.keys():
            histogram,edges = np.histogram((np.array(dict_polar_bins_vols[item])**(1/3))/voxelsize, bins = polar_edges_vols)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_volume_polar[first_idx,second_idx,:] = histogram

        prob_matrix_phi = np.zeros((n_bins+1, n_bins+1, n_bins))
        prob_matrix_phi_polar = np.zeros((n_bins+1, n_bins+1, n_bins))

        for dict_idx, phi in enumerate((total_phis)):
            x_coord = total_x[dict_idx]
            y_coord = total_y[dict_idx]
            z_coord = total_z[dict_idx]
            x_bin = np.digitize(x_coord/voxelsize, xbins)
            y_bin = np.digitize(y_coord/voxelsize, ybins)
            dict_voxel_bins_phis[str(x_bin) + '_' + str(y_bin)].append(phi)

            radius_coord = total_surf_dist[dict_idx]
            angle_coord = total_surf_angles[dict_idx]
            rad_bin = np.digitize(radius_coord, surf_dist_bins)
            angles_bin = np.digitize(angle_coord, surf_angles_bins)
            dict_polar_bins_phis[str(rad_bin) + '_' + str(angles_bin)].append(phi)


        bin_edges_phis = np.linspace(0,np.pi, n_bins+1)
        bin_edges_phis_polar = np.linspace(0,np.pi, n_bins+1)
        

        for item in dict_voxel_bins_phis.keys():
            histogram,edges = np.histogram((dict_voxel_bins_phis[item]), bins = bin_edges_phis)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_phi[first_idx,second_idx,:] = histogram
        for item in dict_polar_bins_phis.keys():
            histogram,edges = np.histogram(dict_polar_bins_phis[item], bins = bin_edges_phis_polar)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_phi_polar[first_idx,second_idx,:] = histogram

        # Anisotropy
        prob_matrix_anis = np.zeros((n_bins+1, n_bins+1, n_bins))
        prob_matrix_anis_polar = np.zeros((n_bins+1, n_bins+1, n_bins))

        for dict_idx, anis in enumerate((total_anisotropies)):
            x_coord = total_x[dict_idx]
            y_coord = total_y[dict_idx]
            z_coord = total_z[dict_idx]
            x_bin = np.digitize(x_coord/voxelsize, xbins)
            y_bin = np.digitize(y_coord/voxelsize, ybins)
            dict_voxel_bins_anis[str(x_bin) + '_' + str(y_bin)].append(anis)


            radius_coord = total_surf_dist[dict_idx]
            angle_coord = total_surf_angles[dict_idx]
            rad_bin = np.digitize(radius_coord, surf_dist_bins)
            angles_bin = np.digitize(angle_coord, surf_angles_bins)
            dict_polar_bins_anis[str(rad_bin) + '_' + str(angles_bin)].append(anis)


            
        bin_edges_anis = np.linspace(0,1, n_bins+1)
        bin_edges_anis_polar = np.linspace(0,1, n_bins+1)

        for item in dict_voxel_bins_orientations.keys():
            histogram,edges = np.histogram((dict_voxel_bins_anis[item]), bins = bin_edges_anis)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_anis[first_idx,second_idx,:] = histogram
        for item in dict_polar_bins_orientations.keys():
            histogram,edges = np.histogram((dict_polar_bins_anis[item]), bins = bin_edges_anis_polar)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_anis_polar[first_idx,second_idx,:] = histogram

        prob_matrix_orientation = np.zeros((n_bins+1, n_bins+1, n_bins))
        prob_matrix_orientation_polar = np.zeros((n_bins+1, n_bins+1, n_bins))

        for dict_idx, angles in enumerate((total_orientations)):
            x_coord = total_x[dict_idx]
            y_coord = total_y[dict_idx]
            z_coord = total_z[dict_idx]
            x_bin = np.digitize(x_coord/voxelsize, xbins)
            y_bin = np.digitize(y_coord/voxelsize, ybins)
            dict_voxel_bins_orientations[str(x_bin) + '_' + str(y_bin)].append(angles)

            radius_coord = total_surf_dist[dict_idx]
            angle_coord = total_surf_angles[dict_idx]
            rad_bin = np.digitize(radius_coord, surf_dist_bins)
            angles_bin = np.digitize(angle_coord, surf_angles_bins)
            dict_polar_bins_orientations[str(rad_bin) + '_' + str(angles_bin)].append(angles)




        bin_edges_orientations = np.linspace(0,np.pi, n_bins+1)
        bin_edges_orientations_polar = np.linspace(0,np.pi, n_bins+1)

        for item in dict_voxel_bins_orientations.keys():
            histogram,edges = np.histogram((dict_voxel_bins_orientations[item]), bins = bin_edges_orientations)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_orientation[first_idx,second_idx,:] = histogram

        for item in dict_polar_bins_orientations.keys():
            histogram,edges = np.histogram((dict_polar_bins_orientations[item]), bins = bin_edges_orientations_polar)
            first_idx = int(item.split('_')[0])
            second_idx = int(item.split('_')[1])
            prob_matrix_orientation_polar[first_idx,second_idx,:] = histogram
        # Number of pores

        prob_matrix_num = np.zeros((n_bins, n_bins))
        prob_matrix_num_polar = np.zeros((n_bins, n_bins))

        for dict_idx, anis in enumerate((total_x)):
            x_coord = total_x[dict_idx]
            y_coord = total_y[dict_idx]
            z_coord = total_z[dict_idx]
            x_bin = np.digitize(x_coord/voxelsize, xbins)
            y_bin = np.digitize(y_coord/voxelsize, ybins)
            prob_matrix_num[x_bin, y_bin] += 1
            radius_coord = total_surf_dist[dict_idx]
            angle_coord = total_surf_angles[dict_idx]
            rad_bin = np.digitize(radius_coord, surf_dist_bins)
            angles_bin = np.digitize(angle_coord, surf_angles_bins)
            prob_matrix_num_polar[rad_bin, angles_bin] +=1
        # breakpoint()
        volbin = np.zeros((n_bins+1, n_bins))
        volbin_anis = np.linspace((np.min(total_vols)**(1/3))/voxelsize, (np.max(total_vols)**(1/3))/voxelsize, n_bins)
        for dict_idx, anis in enumerate((total_anisotropies)):
            x_coord = total_x[dict_idx]
            y_coord = total_y[dict_idx]
            z_coord = total_z[dict_idx]
            vol = (total_vols[dict_idx])**(1/3)/voxelsize
            vol_bin_idx = np.digitize(vol, volbin_anis)
            y_bin = np.digitize(y_coord/voxelsize, ybins)
            dict_voxel_bins_vols_anis[str(vol_bin_idx)].append(vol)
        bin_edges_anis_vols = np.linspace((np.min(total_vols)**(1/3))/voxelsize, (np.max(total_vols)**(1/3))/voxelsize, n_bins+1)
        for item in dict_voxel_bins_vols_anis.keys():
            histogram,edges = np.histogram((dict_voxel_bins_vols_anis[item]), bins = bin_edges_anis_vols)
            first_idx = int(item)
            
            volbin[first_idx, :] = histogram
            
        # breakpoint()

        if index+shift >= limit:
            case = 'full'
        else:
            case = ''
        index = index + shift
        # index < limit
        saving_dir ='./analyze_pore_samples/results/pore_properties' 
        os.makedirs(saving_dir, exist_ok=True)

        assert np.sum(prob_matrix_num) == len(total_x)



        if save:
            prob_matrix_dir = saving_dir + '/probability_matrices/'
            # if not os.path.isdir(saving_dir + '/probability_matrices/'):
            os.makedirs(prob_matrix_dir, exist_ok = True)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allprob_matrix_volume.npy', prob_matrix_volume)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allbin_edges_vols.npy', bin_edges_vols)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allprob_matrix_num.npy', prob_matrix_num/((index+shift)/(shift)))
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allbin_edges_anis.npy', bin_edges_anis)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allprob_matrix_anis.npy', prob_matrix_anis)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allbin_edges_orientations.npy', bin_edges_orientations)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allprob_matrix_orientations.npy', prob_matrix_orientation)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allbin_edges_phis.npy', bin_edges_phis)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'allprob_matrix_phis.npy', prob_matrix_phi)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarprob_matrix_volume.npy', prob_matrix_volume_polar)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarbin_edges_vols.npy', polar_edges_vols)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarprob_matrix_num.npy', prob_matrix_num_polar/((index+shift)/(shift)))
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarbin_edges_anis.npy', bin_edges_anis_polar)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarprob_matrix_anis.npy', prob_matrix_anis_polar)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarbin_edges_orientations.npy', bin_edges_orientations_polar)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarprob_matrix_orientations.npy', prob_matrix_orientation_polar)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarbin_edges_phis.npy', bin_edges_phis_polar)
            np.save(prob_matrix_dir + str(n_bins) + '_'+ str(k) + case + 'polarprob_matrix_phis.npy', prob_matrix_phi_polar)


        dict_properties = {'x_locs': total_x, 'y_locs': total_y, 'maj_axis_l': total_maj, 'phis': total_phis, 'vols': total_vols, 'anisotropies':total_anisotropies,'orientations': total_orientations, 'z_locs': total_z, 'surf_dist': surf_dist, 'rough':rough, 'surf_angles': surf_angles }
        locations = [pore.centroid for pore in pores_keep]
        df_properties = pd.DataFrame(dict_properties)
        df_properties.to_csv(saving_dir+'/dict_properties_partial' + str(k)+ '.csv')
    del polar_images
    return dict_properties, pores_total

def compute_statistics(pore_dataset, voxelsize, imstack = None):
    anisotropies = []
    orientations = []
    vols = []
    sphericity = []
    x_locs = []
    y_locs = []
    z_locs = []
    locations = [pore.centroid for pore in pore_dataset]
    maj_axis_l = []
    min_axis_l = []
    phis = []
    for i in range(len(pore_dataset)):
        pore = pore_dataset[i]
        vols.append(pore_dataset[i]['area']*voxelsize*voxelsize*voxelsize)
        y_locs.append(pore_dataset[i]['centroid'][1]*voxelsize)
        x_locs.append(pore_dataset[i]['centroid'][0]*voxelsize)
        z_locs.append(pore_dataset[i]['centroid'][2]*voxelsize)
        maj_axis_l.append((pore_dataset[i].major_axis_length)*voxelsize)
        min_axis_l.append((pore_dataset[i].minor_axis_length)*voxelsize)
        thresh = 0.3
        
        inertia_eigval = pore.inertia_tensor_eigvals
        inertia = pore.inertia_tensor
        maxeig = np.argmax(inertia_eigval)
        eigvec = np.linalg.eig(pore.inertia_tensor)[1]
        eigvals = np.linalg.eig(pore.inertia_tensor)[0]
        anis = 1 - np.min(eigvals)/np.max(eigvals)
        anisotropies.append(anis)
        maxvector = eigvec[:, maxeig]
        orientation = angle_between(maxvector, np.array([0,0,1]))
        orientations.append(orientation)
        phi = angle_between(maxvector, np.array([0,1,0]))
        phis.append(phi)


    stats = {
        'anisotropies': anisotropies,
        'orientations' : orientations,
        'vols' : vols,
        'x_locs' : x_locs,
        'y_locs' : y_locs,
        'z_locs' : z_locs,
        'locations' : locations,
        'maj_axis_l' : maj_axis_l,
        'min_axis_l' : min_axis_l,
        'phis' : phis
    }
    return stats

def plot_pore_fn(idxs,imstack,global_idx = 0, save = False, name = None, pore_list = None, voxelsize = 3.49, quantity = 0):

        figdir= './NN_analysis/'
        if pore_list is None:
            pore_list = pores_total
        idxs = np.squeeze(np.array([idxs]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        maxlimits= []
        minlimits = []

        if idxs.size == 1:
            idx = idxs[0]
            start = pore_list[idx].slice[2].start
            stop = pore_list[idx].slice[2].stop


            meshlist = [np.arange(pore_list[idx].image.shape[0]+1)*voxelsize, np.arange(pore_list[idx].image.shape[1]+1)*voxelsize,  np.arange(pore_list[idx].image.shape[2]+1)*voxelsize]            
            meshgrid = np.meshgrid(meshlist[0], meshlist[1], meshlist[2], indexing = 'ij')           
            ax.voxels(meshgrid[0]+pore_list[idx].centroid[0]*voxelsize, meshgrid[1]+pore_list[idx].centroid[1]*voxelsize, meshgrid[2]+pore_list[idx].centroid[2]*voxelsize, pore_list[idx].image, edgecolor='k')
            ax.set_xlabel(r'x $[\mu m]$')
            ax.set_ylabel(r'y $[\mu m]$')
            ax.set_zlabel(r'z $[\mu m]$')
            plt.title('Pore volume: {:.3f}'.format(voxelsize*voxelsize*voxelsize*pore_list[idx].area) + r' [$\mu m^3$]' + " Voxel count: " + str(pore_list[idx].area))
            plt.title('Aspect Ratio: {:3.f}'.format(pore_dataset[i].minor_axis_length/pore_dataset[i].major_axis_length))
            xbound = (pore_list[idx].image.shape[0]+1)*voxelsize
            ybound = (pore_list[idx].image.shape[1]+1)*voxelsize
            zbound =(pore_list[idx].image.shape[2]+1)*voxelsize
            
            
            maxlimit = np.max(([xbound, ybound, zbound]))
            ax.set_xlim(np.min(minlimits), np.max(maxlimits))
            ax.set_ylim(np.min(minlimits), np.max(maxlimits))
            ax.set_zlim(np.min(minlimits), np.max(maxlimits))
            plt.show()
          
        else:

            print("HERE")
       
            xlimits = []
            ylimits = []
            zlimits = []
            title = ''
            
            start_all = []
            stop_all = []
            for index in idxs:
                start_all.append(pore_list[index].slice[2].start)
                stop_all.append(pore_list[index].slice[2].stop)
            start = np.min(start_all)
            stop = np.max(stop_all)

            padded_start = np.max((0, start-3))
            padded_stop  = np.min((stop+3, imstack.shape[2]))
            for index in range(padded_start, padded_stop):
                pore_found = 0
                fig_2 = plt.figure()
                ax_2 = fig_2.gca()
                for idx in idxs:
                    
                    ax_2.imshow(imstack[:, :, index])

                    if index in range(pore_list[idx].slice[2].start, pore_list[idx].slice[2].stop):
                        pore_found += 1
                        ax_2.set_title('Slice: ' + str(index-start) +
                                    " Pore: " + str(idx) + ' Frame:' + str(global_idx+index))
                        if pore_found % 2 == 0:
                            ax_2.annotate(str(idx), (pore_list[idx].centroid[1], pore_list[idx].centroid[0]), (
                                pore_list[idx].centroid[1] - 30, pore_list[idx].centroid[0] - 30), arrowprops=dict(edgecolor='r', arrowstyle='-'), color='r')

                        else:
                            ax_2.annotate(str(idx), (pore_list[idx].centroid[1], pore_list[idx].centroid[0]), (
                                pore_list[idx].centroid[1] + 30, pore_list[idx].centroid[0] + 30), arrowprops=dict(edgecolor='r', arrowstyle='-'), color='r')


                    else:
                        print(index, idx)
                
                if not pore_found: 
                    ax_2.set_title('Slice: ' + str(index-start) + " Pore: none" + ' Frame:' + str(global_idx+index))
                print(pore_found)
                fig_2.savefig(name + '/frame' + str(global_idx+index)+ 'slice' + str(index-start) + '.png')
                pore_found = 0
            
            for idx in idxs:

                meshlist = [np.arange(pore_list[idx].image.shape[0]+1)*voxelsize, np.arange(
                    pore_list[idx].image.shape[1]+1)*voxelsize,  np.arange(global_idx,global_idx+ pore_list[idx].image.shape[2]+1)*voxelsize]
                meshgrid = np.meshgrid(
                    meshlist[0], meshlist[1], meshlist[2], indexing='ij')
                ax.voxels(meshgrid[0]+pore_list[idx].centroid[0]*voxelsize, meshgrid[1]+pore_list[idx].centroid[1]
                            * voxelsize, meshgrid[2]+pore_list[idx].centroid[2]*voxelsize, pore_list[idx].image, edgecolor='k', label = 'Pore id: ' + str(idx))
                ax.set_xlabel(r'x $[\mu m]$')
                ax.set_ylabel(r'y $[\mu m]$')
                ax.set_zlabel(r'z $[\mu m]$')

                title = 'Anisotropy: {:.3f}'.format(quantity)
                xbound = (pore_list[idx].image.shape[0]+1)*voxelsize
                ybound = (pore_list[idx].image.shape[1]+1)*voxelsize
                zbound = (global_idx+ pore_list[idx].image.shape[2]+1)*voxelsize
                xlimits.append(pore_list[idx].centroid[0]*voxelsize)
                xlimits.append(
                    xbound + pore_list[idx].centroid[0]*voxelsize)
                ylimits.append(pore_list[idx].centroid[1]*voxelsize)
                ylimits.append(
                    ybound + pore_list[idx].centroid[1]*voxelsize)
                zlimits.append(global_idx*voxelsize + pore_list[idx].centroid[2]*voxelsize)
                zlimits.append(
                    zbound + pore_list[idx].centroid[2]*voxelsize)

            ax.set_title(title, fontsize = 6)
            ax.set_xlim(np.min(xlimits), np.max(xlimits))
            ax.set_ylim(np.min(ylimits), np.max(ylimits))
            ax.set_zlim(np.min(zlimits), np.max(zlimits))
            set_axes_equal(ax)
            if save:
                fig.savefig(name + '/3dpore.png')
                plt.clf()
            else:
                breakpoint()
                plt.show()
                plt.clf()

def plot_statistics(pore_dataset, voxelsize, k = 0):


    thresh = 1
    
    maxvects = []
    for thresh in [1.0, 1.5, 2.0, 3.0]:
        orientations = []
        anisotropies = []
        for pore in pore_dataset:
            
            try:
                if pore.major_axis_length/pore.minor_axis_length > thresh:
                    print(thresh, 'THRESH')
                    inertia_eigval = pore.inertia_tensor_eigvals
                    inertia = pore.inertia_tensor
                    maxeig = np.argmax(inertia_eigval)
                    eigvec = np.linalg.eig(pore.inertia_tensor)[1]
                    eigvals = np.linalg.eig(pore.inertia_tensor)[0]
                    anis = 1 - np.min(eigvals)/np.max(eigvals)
                    anisotropies.append(anis)
                    maxvector = eigvec[:, maxeig]
                    orientation = angle_between(maxvector, np.array([0,0,1]))
                    orientations.append(orientation)
               
            except Exception as e:
                print(e)
                breakpoint()

        histogram = plt.hist((np.array(orientations)/np.pi)*180, density = True, bins=30, edgecolor = 'k')                
        plt.title("Orientation, Pore Sample: " + str(k + 1) + " , threshold = " + str(thresh))
        plt.xlabel(r"Angle [Degrees]")
        plt.ylabel("Probability")
        plt.ylim(0, 1.01*np.max(histogram[0]))
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig("orientation" + str(k+ 100) + "_" + str(thresh) + ".png")
        plt.clf()
        print('done')
 

    try:
        locations = [pore.centroid for pore in pore_dataset]
        biglocations = [pore.centroid for pore in pore_dataset if pore.area > 1000]
        loc_pores = np.array(locations)
        kdtree = spatial.KDTree(locations)
        dd, ii = kdtree.query(locations, len(locations))
        nearest_neighbor = dd[:,1:]*voxelsize
        neighbors = dd[:,1]*voxelsize
        edges, hist = np.histogram(dd[:,1:]*voxelsize, bins = np.arange(0,800,25)*voxelsize)
        plt.plot(hist[:-1],edges, linewidth = 2.0)
        plt.title("Nearest Neighbor distance, Pore sample: " + str(k + 1))

        plt.xlabel(r"Distance [$\mu m$]")
        plt.ylabel("Number of Pores")
        plt.tight_layout()
        plt.savefig("rdf" + str(k+ 100) + ".png")
    except:
        breakpoint()


    vols = []
    sphericity = []
    x_locs = []
    y_locs = []
    z_locs = []
    locations = [pore.centroid for pore in pore_dataset]
    maj_axis_l = []
    min_axis_l = []
    for i in range(len(pore_dataset)):
        vols.append(pore_dataset[i]['area']*voxelsize*voxelsize*voxelsize)

        y_locs.append(pore_dataset[i]['centroid'][1]*voxelsize)
        x_locs.append(pore_dataset[i]['centroid'][0]*voxelsize)
        z_locs.append(pore_dataset[i]['centroid'][2]*voxelsize)
        maj_axis_l.append((pore_dataset[i].major_axis_length)*voxelsize)
        min_axis_l.append((pore_dataset[i].minor_axis_length)*voxelsize)

    plt.clf()
    plt.plot(np.array(vols), np.array(maj_axis_l)/np.array(min_axis_l), '.')
    plt.title('Volume correlations with shape')
    plt.xlabel('Volume')
    plt.xscale('log')  
    plt.ylabel(r'ratio: Major Axis Length/Minor Axis Length')
    plt.tight_layout()
    plt.savefig('vols_correlation' + str(k+100)+".png")
    plt.clf()
    
    histogram = plt.hist(neighbors, density = True, bins=100, edgecolor = 'k')
    plt.title("Nearest Neighbor distance, Pore sample: " + str(k + 1))
    
    plt.xlabel(r"Distance [$\mu m$]")
    plt.text(300, 0.01, 'Min distance : {:.2f}'.format(np.min(neighbors)) + r'$\mu m$')
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig("nearest" + str(k+ 100) + ".png")
    plt.clf()
    plt.clf()

    histogram = plt.hist(np.array(maj_axis_l)/np.array(min_axis_l), density = True, bins=100, edgecolor = 'k')
    plt.title("Eccentricity, Pore sample: " + str(k + 1))

    plt.xlabel(r"Ratio")
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig("ratiominmax" + str(k+ 100) + ".png")
    plt.clf()
    plt.clf()

    histogram = plt.hist(min_axis_l, density = True, bins=np.logspace(np.log10(10e0),np.log10(10e4)), edgecolor = 'k')
    plt.title("Minor Axis Length, Pore sample: " + str(k + 1))
    plt.xscale('log')  
    plt.xlabel(r"Length [$\mu m$]")
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig("minor" + str(k+ 100) + ".png")
    plt.clf()
    plt.clf()


    # # prepare some coordinates
    
    histogram = plt.hist(vols, density = True, alpha = 0.5,bins=np.logspace(np.log10(10e1),np.log10(10e5)), edgecolor = 'k')
    plt.title("Volume, Pore sample: " + str(k + 1))
    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_64 = np.ones(50)*64*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_64, redline_y, 'r', label = '64 voxel count threshold')

    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_32 = np.ones(50)*32*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_32, redline_y, 'r--', label = '32 voxel count threshold')

    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_16 = np.ones(50)*10*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_16, redline_y, 'r:', label = '10 voxel count threshold')


    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_16 = np.ones(50)*4*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_16, redline_y, 'g', label = '4 voxel count threshold')
    plt.xscale('log')    
    plt.xlabel(r"Volume [$\mu m^3$]")
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig("vols_new" + str(k+ 100) + ".png")
    plt.clf()





#@profile
def process_folder(k):
    print(k)
    frame_window = 200 # how many frames to analyze
    rgp_window = 200 # size of regionprops window
    shift = 200 # overlap of regionprops window (should be larger than all pores)
    pores_total = []
    problem_snow = 0
    neighbors = []
    pore_zs = []
    def plot_pore(idxs,imstack,global_idx = 0, save = False, name = None, pore_list = None):

        figdir= './NN_analysis/'
        if pore_list is None:
            pore_list = pores_total
        idxs = np.squeeze(np.array([idxs]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        maxlimits= []
        minlimits = []

        if idxs.size == 1:
            idx = idxs[0]
            start = pore_list[idx].slice[2].start
            stop = pore_list[idx].slice[2].stop


            meshlist = [np.arange(pore_list[idx].image.shape[0]+1)*voxelsize, np.arange(pore_list[idx].image.shape[1]+1)*voxelsize,  np.arange(pore_list[idx].image.shape[2]+1)*voxelsize]            
            meshgrid = np.meshgrid(meshlist[0], meshlist[1], meshlist[2], indexing = 'ij')           
            ax.voxels(meshgrid[0]+pore_list[idx].centroid[0]*voxelsize, meshgrid[1]+pore_list[idx].centroid[1]*voxelsize, meshgrid[2]+pore_list[idx].centroid[2]*voxelsize, pore_list[idx].image, edgecolor='k')
            ax.set_xlabel(r'x $[\mu m]$')
            ax.set_ylabel(r'y $[\mu m]$')
            ax.set_zlabel(r'z $[\mu m]$')

            plt.title('Pore volume: {:.3f}'.format(voxelsize*voxelsize*voxelsize*pore_list[idx].area) + r' [$\mu m^3$]' + " Voxel count: " + str(pore_list[idx].area))

            xbound = (pore_list[idx].image.shape[0]+1)*voxelsize
            ybound = (pore_list[idx].image.shape[1]+1)*voxelsize
            zbound =(pore_list[idx].image.shape[2]+1)*voxelsize
            
            
            maxlimit = np.max(([xbound, ybound, zbound]))
            ax.set_xlim(np.min(minlimits), np.max(maxlimits))
            ax.set_ylim(np.min(minlimits), np.max(maxlimits))
            ax.set_zlim(np.min(minlimits), np.max(maxlimits))
            plt.show()

        else:

            print("HERE")

            xlimits = []
            ylimits = []
            zlimits = []
            title = ''
     
            start_all = []
            stop_all = []
            for index in idxs:
                start_all.append(pore_list[index].slice[2].start)
                stop_all.append(pore_list[index].slice[2].stop)
            start = np.min(start_all)
            stop = np.max(stop_all)

            padded_start = np.max((0, start-3))
            padded_stop  = np.min((stop+3, imstack.shape[2]))
            for index in range(padded_start, padded_stop):
                pore_found = 0
                #slice_idx = index - start_all
                fig_2 = plt.figure()
                ax_2 = fig_2.gca()
                for idx in idxs:
                    
                    ax_2.imshow(imstack[:, :, index])
                    # ax_2.set_title('Slice: ' + str(index) + " Pore: none" + ' Frame:' + str(index))
                    if index in range(pore_list[idx].slice[2].start, pore_list[idx].slice[2].stop):
                        # breakpoint()
                        pore_found += 1
                        ax_2.set_title('Slice: ' + str(index-start) +
                                    " Pore: " + str(idx) + ' Frame:' + str(global_idx+index))
                        if pore_found % 2 == 0:
                            ax_2.annotate(str(idx), (pore_list[idx].centroid[1], pore_list[idx].centroid[0]), (
                                pore_list[idx].centroid[1] - 30, pore_list[idx].centroid[0] - 30), arrowprops=dict(edgecolor='r', arrowstyle='-'), color='r')

                        else:
                            ax_2.annotate(str(idx), (pore_list[idx].centroid[1], pore_list[idx].centroid[0]), (
                                pore_list[idx].centroid[1] + 30, pore_list[idx].centroid[0] + 30), arrowprops=dict(edgecolor='r', arrowstyle='-'), color='r')



                    else:
                        print(index, idx)
                        # breakpoint()
                
                if not pore_found: 
                    ax_2.set_title('Slice: ' + str(index-start) + " Pore: none" + ' Frame:' + str(global_idx+index))
                print(pore_found)
                fig_2.savefig(name + '/frame' + str(global_idx+index)+ 'slice' + str(index-start) + '.png')
                pore_found = 0
            
            for idx in idxs:

                meshlist = [np.arange(pore_list[idx].image.shape[0]+1)*voxelsize, np.arange(
                    pore_list[idx].image.shape[1]+1)*voxelsize,  np.arange(global_idx,global_idx+ pore_list[idx].image.shape[2]+1)*voxelsize]
                meshgrid = np.meshgrid(
                    meshlist[0], meshlist[1], meshlist[2], indexing='ij')
                ax.voxels(meshgrid[0]+pore_list[idx].centroid[0]*voxelsize, meshgrid[1]+pore_list[idx].centroid[1]
                            * voxelsize, meshgrid[2]+pore_list[idx].centroid[2]*voxelsize, pore_list[idx].image, edgecolor='k', label = 'Pore id: ' + str(idx))
                ax.set_xlabel(r'x $[\mu m]$')
                ax.set_ylabel(r'y $[\mu m]$')
                ax.set_zlabel(r'z $[\mu m]$')
            #  breakpoint()
                title += 'Pore volume: {:.3f}'.format(
                    voxelsize*voxelsize*voxelsize*pore_list[idx].area) + r' [$\mu m^3$]' + " Voxel count: " + str(pore_list[idx].area) + ', '

            #    breakpoint()
                xbound = (pore_list[idx].image.shape[0]+1)*voxelsize
                ybound = (pore_list[idx].image.shape[1]+1)*voxelsize
                zbound = (global_idx+ pore_list[idx].image.shape[2]+1)*voxelsize
                #breakpoint()
                xlimits.append(pore_list[idx].centroid[0]*voxelsize)
                xlimits.append(
                    xbound + pore_list[idx].centroid[0]*voxelsize)
                ylimits.append(pore_list[idx].centroid[1]*voxelsize)
                ylimits.append(
                    ybound + pore_list[idx].centroid[1]*voxelsize)
                zlimits.append(global_idx*voxelsize + pore_list[idx].centroid[2]*voxelsize)
                zlimits.append(
                    zbound + pore_list[idx].centroid[2]*voxelsize)

            ax.set_aspect('equal', adjustable='box')

            ax.set_title(title, fontsize = 6)
            ax.set_xlim(np.min(xlimits), np.max(xlimits))
            ax.set_ylim(np.min(ylimits), np.max(ylimits))
            ax.set_zlim(np.min(zlimits), np.max(zlimits))
            set_axes_equal(ax)
            if save:
                fig.savefig(name + '/3dpore.png')
              
                plt.clf()
            else:
                breakpoint()
                plt.show()
                plt.clf()
    totlistnum = np.zeros(len(np.arange(0,2000,50)))



    for j in range(0, len(os.listdir(subfolders[k])), frame_window ):
        voxelsize, imstack,_,_ = load_data(k, num = frame_window, start = j)
        imstack[imstack  == 255] = 0
        imstack[imstack == 159] = 1
        imstack = np.array(imstack, dtype = 'int32')
                    
        try:
            im = measure.label(imstack[:,:,:])

        except Exception as e:
            problem_snow += 1
            print(e)
            breakpoint()
    
        boundaries = []
        for i in range(0, frame_window, shift ):
            if i+ shift> frame_window:
                
                break
            print(im[:,:,i:i+shift].shape)
            print("REGIONPROPS")
            props = skimage.measure.regionprops(im[:,:,i:i+rgp_window])
            im_channels = np.copy(im)
            pores_cross = np.unique(im_channels[:,:,0]) + 1
            test = np.zeros((im_channels.shape[:2]))
            test2 = np.zeros((im_channels.shape[:2]))
            test3 = np.zeros((im_channels.shape[:2]))
            for p_idx in np.arange(len(props)):
                test[im_channels[:,:,0] == props[p_idx].label] = props[p_idx].area
                test2[im_channels[:,:,0] == props[p_idx].label] = props[p_idx].centroid[2]

                inertia_eigval =  props[p_idx].inertia_tensor_eigvals
                inertia =  props[p_idx].inertia_tensor
                maxeig = np.argmax(inertia_eigval)
                eigvec = np.linalg.eig(props[p_idx].inertia_tensor)[1]
                eigvals = np.linalg.eig(props[p_idx].inertia_tensor)[0]
                anis = 1 - np.min(eigvals)/np.max(eigvals)
    
                maxvector = eigvec[:, maxeig]
                test3[im_channels[:,:,0] == props[p_idx].label] = anis
               
            print("REGIONPROPS DONE")
        
            endframe = i+shift - 1
            pores_keep = []
            windowboundary = rgp_window - shift - 1
            windowboundarypores = [prop for prop in props if windowboundary in range(prop.slice[2].start, prop.slice[2].stop)] # identify pores that were cut off before
            alreadycountedpores = [prop for prop in props if windowboundary not in range(prop.slice[2].start, prop.slice[2].stop) and prop.centroid[2] < windowboundary] # remove pores that were counted before, and were not cut off
            pores = [prop for prop in props if endframe not in range(prop.slice[2].start, prop.slice[2].stop)] # identify pores that are not cut off
            boundaries = [prop for prop in props if endframe  in range(prop.slice[2].start, prop.slice[2].stop)] # identify pores that are cut off

    
            if i > 0:
                pores_keep = []
                for prop_idx in range(len(props)):
                    alreadycount_bool  = False
                    boundary_bool = False
                    alreadycount_bool =  windowboundary not in range(pores[prop_idx].slice[2].start, pores[prop_idx].slice[2].stop) and (pores[prop_idx].centroid[2] < windowboundary)
                    boundary_bool = endframe  in range(pores[prop_idx].slice[2].start, pores[prop_idx].slice[2].stop)
                    
                    if not alreadycount_bool and not boundary_bool:
                      
                        zlength = pores[prop_idx].slice[2].stop - pores[prop_idx].slice[2].start
                        xlength = pores[prop_idx].slice[1].stop - pores[prop_idx].slice[1].start
                        ylength = pores[prop_idx].slice[0].stop - pores[prop_idx].slice[0].start
                        
                        if xlength > 3 and ylength > 3 and zlength > 3:
                            pores_keep.append(pores[ prop_idx])

            elif i == 0:
                for prop_idx in range(len(pores)):
                    zlength = pores[prop_idx].slice[2].stop - pores[prop_idx].slice[2].start
                    xlength = pores[prop_idx].slice[1].stop - pores[prop_idx].slice[1].start
                    ylength = pores[prop_idx].slice[0].stop - pores[prop_idx].slice[0].start
                    
                    if xlength > 3 and ylength > 3 and zlength > 3:
                       
                        pores_keep.append(pores[prop_idx])
              
            curr_vols = [pore.area for pore in pores_keep]
            smallest  = np.argmin(curr_vols)
            
            parentdir = 'test_smallest_radius'
            if not os.path.isdir(parentdir):
                os.mkdir(parentdir)

            figdir = parentdir + '/smallest' + str(smallest)
            if not os.path.isdir(figdir):
                os.mkdir(figdir)

            curr_zs = np.array([pore.centroid[2] for pore in pores_keep]) + j
            pore_zs.extend(curr_zs)
            pores_total.extend(pores_keep)
            print(len(pores_total), "length of pores_total")

            print(i, i + rgp_window)

        print("saving")

        pore_dataset = pores_keep

        thresh = 1
        
        maxvects = []
        for thresh in [1.0, 1.5, 2.0, 3.0]:
            orientations = []
            anisotropies = []
            for pore in pore_dataset:
                
                try:
                    if pore.major_axis_length/pore.minor_axis_length > thresh:
                        print(thresh, 'THRESH')
                        inertia_eigval = pore.inertia_tensor_eigvals
                        inertia = pore.inertia_tensor
                        maxeig = np.argmax(inertia_eigval)
                        eigvec = np.linalg.eig(pore.inertia_tensor)[1]
                        eigvals = np.linalg.eig(pore.inertia_tensor)[0]
                        anis = 1 - np.min(eigvals)/np.max(eigvals)
                        anisotropies.append(anis)
                        maxvector = eigvec[:, maxeig]
                        orientation = angle_between(maxvector, np.array([0,0,1]))
                        orientations.append(orientation)

                except Exception as e:
                    print(e)
                    breakpoint()
            #breakpoint()
            histogram = plt.hist((np.array(orientations)/np.pi)*180, density = True, bins=30, edgecolor = 'k')                
            plt.title("Orientation, Pore Sample: " + str(k + 1) + " , threshold = " + str(thresh) + ", z = " + str(j*voxelsize))
            plt.xlabel(r"Angle [Degrees]")
            plt.ylabel("Probability")
            plt.ylim(0, 1.01*np.max(histogram[0]))
            plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
            plt.tight_layout()
            plt.savefig(str(j) + "orientation" + str(k+ 100) + "_" + str(thresh) + ".png")
            plt.clf()
            print('done')
        #breakpoint()

        try:
            locations = [pore.centroid for pore in pore_dataset]
            biglocations = [pore.centroid for pore in pore_dataset if pore.area > 1000]
            loc_pores = np.array(locations)
            kdtree = spatial.KDTree(locations)
            dd, ii = kdtree.query(locations, len(locations))
            nearest_neighbor = dd[:,1:]*voxelsize
            neighbors = dd[:,1]*voxelsize
            edges, hist = np.histogram(dd[:,1:]*voxelsize, bins = np.arange(0,800,25)*voxelsize)
            plt.plot(hist[:-1],edges, linewidth = 2.0)
            plt.title("Nearest Neighbor distance, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))

            plt.xlabel(r"Distance [$\mu m$]")
            plt.ylabel("Number of Pores")
            plt.tight_layout()
            plt.savefig(str(j) +  "rdf" + str(k+ 100) + ".png")
        except:
            breakpoint()


        vols = []
        sphericity = []
        x_locs = []
        y_locs = []
        z_locs = []
        locations = [pore.centroid for pore in pore_dataset]
        maj_axis_l = []
        min_axis_l = []
        for i in range(len(pore_dataset)):
            vols.append(pore_dataset[i]['area']*voxelsize*voxelsize*voxelsize)
            y_locs.append(pore_dataset[i]['centroid'][1]*voxelsize)
            x_locs.append(pore_dataset[i]['centroid'][0]*voxelsize)
            z_locs.append(pore_dataset[i]['centroid'][2]*voxelsize)
            maj_axis_l.append((pore_dataset[i].major_axis_length)*voxelsize)
            min_axis_l.append((pore_dataset[i].minor_axis_length)*voxelsize)

        plt.clf()
        plt.plot(np.array(vols), np.array(maj_axis_l)/np.array(min_axis_l), '.')
        plt.title('Volume correlations with shape'+ ", z = " + str(j*voxelsize))
        plt.xlabel('Volume')
        plt.xscale('log')  
    
        plt.ylabel(r'ratio: Major Axis Length/Minor Axis Length')
        plt.tight_layout()
        plt.savefig(str(j) + 'vols_correlation' + str(k+100)+".png")
        plt.clf()
        
        histogram = plt.hist(neighbors, density = True, bins=100, edgecolor = 'k')
        plt.title("Nearest Neighbor distance, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
        
        plt.xlabel(r"Distance [$\mu m$]")
        plt.text(300, 0.01, 'Min distance : {:.2f}'.format(np.min(neighbors)) + r'$\mu m$')
        plt.ylabel("Probability")
        plt.ylim(0, 1.01*np.max(histogram[0]))
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(str(j) + "nearest" + str(k+ 100) + ".png")
        plt.clf()
        plt.clf()
   
        histogram = plt.hist(np.array(maj_axis_l)/np.array(min_axis_l), density = True, bins=100, edgecolor = 'k')
        plt.title("Eccentricity, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
        plt.xlabel(r"Ratio")
        plt.ylabel("Probability")
        plt.ylim(0, 1.01*np.max(histogram[0]))
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(str(j) + "ratiominmax" + str(k+ 100) + ".png")
        plt.clf()
        plt.clf()

        histogram = plt.hist(min_axis_l, density = True, bins=np.logspace(np.log10(10e0),np.log10(10e4)), edgecolor = 'k')
        plt.title("Minor Axis Length, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
        plt.xscale('log')  
        plt.xlabel(r"Length [$\mu m$]")
        plt.ylabel("Probability")
        plt.ylim(0, 1.01*np.max(histogram[0]))
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(str(j) + "minor" + str(k+ 100) + ".png")
        plt.clf()
        plt.clf()


        # # prepare some coordinates
        

        histogram = plt.hist(vols, density = True, alpha = 0.5,bins=np.logspace(np.log10(10e1),np.log10(10e5)), edgecolor = 'k')
        plt.title("Volume, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
        redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
        redline_x_64 = np.ones(50)*64*voxelsize*voxelsize*voxelsize 
        plt.plot(redline_x_64, redline_y, 'r', label = '64 voxel count threshold')

        redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
        redline_x_32 = np.ones(50)*32*voxelsize*voxelsize*voxelsize 
        plt.plot(redline_x_32, redline_y, 'r--', label = '32 voxel count threshold')

        redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
        redline_x_16 = np.ones(50)*10*voxelsize*voxelsize*voxelsize 
        plt.plot(redline_x_16, redline_y, 'r:', label = '10 voxel count threshold')


        redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
        redline_x_16 = np.ones(50)*4*voxelsize*voxelsize*voxelsize 
        plt.plot(redline_x_16, redline_y, 'g', label = '4 voxel count threshold')
        plt.xscale('log')    
        plt.xlabel(r"Volume [$\mu m^3$]")
        plt.ylabel("Probability")
        plt.ylim(0, 1.01*np.max(histogram[0]))
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(str(j) + "vols_new" + str(k+ 100) + ".png")
        plt.clf()


    
    pore_dataset = pores_total
    thresh = 1
    
    maxvects = []
    for thresh in [1.0, 1.5, 2.0, 3.0]:
        orientations = []
        anisotropies = []
        for pore in pore_dataset:
            
            try:
                if pore.major_axis_length/pore.minor_axis_length > thresh:
                    print(thresh, 'THRESH')
                    inertia_eigval = pore.inertia_tensor_eigvals
                    inertia = pore.inertia_tensor
                    maxeig = np.argmax(inertia_eigval)
                    eigvec = np.linalg.eig(pore.inertia_tensor)[1]
                    eigvals = np.linalg.eig(pore.inertia_tensor)[0]
                    anis = 1 - np.min(eigvals)/np.max(eigvals)
                    anisotropies.append(anis)
                    maxvector = eigvec[:, maxeig]
                    orientation = angle_between(maxvector, np.array([0,0,1]))
                    orientations.append(orientation)

            except Exception as e:
                print(e)
                breakpoint()
        #breakpoint()
        histogram = plt.hist((np.array(orientations)/np.pi)*180, density = True, bins=30, edgecolor = 'k')                
        plt.title("Orientation, Pore Sample: " + str(k + 1) + " , threshold = " + str(thresh) + ", z = " + str(j*voxelsize))
        plt.xlabel(r"Angle [Degrees]")
        plt.ylabel("Probability")
        plt.ylim(0, 1.01*np.max(histogram[0]))
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(str(j) + "orientation" + str(k+ 100) + "_" + str(thresh) + ".png")
        plt.clf()
        print('done')
    #breakpoint()

    try:
        locations = [pore.centroid for pore in pore_dataset]
        biglocations = [pore.centroid for pore in pore_dataset if pore.area > 1000]
        loc_pores = np.array(locations)
        kdtree = spatial.KDTree(locations)
        dd, ii = kdtree.query(locations, len(locations))
        nearest_neighbor = dd[:,1:]*voxelsize
        neighbors = dd[:,1]*voxelsize
        edges, hist = np.histogram(dd[:,1:]*voxelsize, bins = np.arange(0,800,25)*voxelsize)
        plt.plot(hist[:-1],edges, linewidth = 2.0)
        plt.title("Nearest Neighbor distance, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))

        plt.xlabel(r"Distance [$\mu m$]")
        plt.ylabel("Number of Pores")
        plt.tight_layout()
        plt.savefig(str(j) +  "rdf" + str(k+ 100) + ".png")
    except:
        breakpoint()


    vols = []
    sphericity = []
    x_locs = []
    y_locs = []
    z_locs = []
    locations = [pore.centroid for pore in pore_dataset]
    maj_axis_l = []
    min_axis_l = []
    for i in range(len(pore_dataset)):
        vols.append(pore_dataset[i]['area']*voxelsize*voxelsize*voxelsize)
    #  sphericity.append(props[i]['sphericity'])
        y_locs.append(pore_dataset[i]['centroid'][1]*voxelsize)
        x_locs.append(pore_dataset[i]['centroid'][0]*voxelsize)
        z_locs.append(pore_dataset[i]['centroid'][2]*voxelsize)
        maj_axis_l.append((pore_dataset[i].major_axis_length)*voxelsize)
        min_axis_l.append((pore_dataset[i].minor_axis_length)*voxelsize)

    plt.clf()
    plt.plot(np.array(vols), np.array(maj_axis_l)/np.array(min_axis_l), '.')
    plt.title('Volume correlations with shape'+ ", z = " + str(j*voxelsize))
    plt.xlabel('Volume')
    plt.xscale('log')  
# breakpoint()
    plt.ylabel(r'ratio: Major Axis Length/Minor Axis Length')
    plt.tight_layout()
    plt.savefig(str(j) + 'vols_correlation' + str(k+100)+".png")
    plt.clf()
    
    histogram = plt.hist(neighbors, density = True, bins=100, edgecolor = 'k')
    plt.title("Nearest Neighbor distance, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
    
    plt.xlabel(r"Distance [$\mu m$]")
    plt.text(300, 0.01, 'Min distance : {:.2f}'.format(np.min(neighbors)) + r'$\mu m$')
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(str(j) + "nearest" + str(k+ 100) + ".png")
    plt.clf()
    plt.clf()
#    breakpoint()



    histogram = plt.hist(np.array(maj_axis_l)/np.array(min_axis_l), density = True, bins=100, edgecolor = 'k')
    plt.title("Eccentricity, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
    plt.xlabel(r"Ratio")
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(str(j) + "ratiominmax" + str(k+ 100) + ".png")
    plt.clf()
    plt.clf()




    histogram = plt.hist(min_axis_l, density = True, bins=np.logspace(np.log10(10e0),np.log10(10e4)), edgecolor = 'k')
    plt.title("Minor Axis Length, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
    plt.xscale('log')  
    plt.xlabel(r"Length [$\mu m$]")
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(str(j) + "minor" + str(k+ 100) + ".png")
    plt.clf()
    plt.clf()


    histogram = plt.hist(vols, density = True, alpha = 0.5,bins=np.logspace(np.log10(10e1),np.log10(10e5)), edgecolor = 'k')
    plt.title("Volume, Pore sample: " + str(k + 1)+ ", z = " + str(j*voxelsize))
    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_64 = np.ones(50)*64*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_64, redline_y, 'r', label = '64 voxel count threshold')

    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_32 = np.ones(50)*32*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_32, redline_y, 'r--', label = '32 voxel count threshold')

    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_16 = np.ones(50)*10*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_16, redline_y, 'r:', label = '10 voxel count threshold')


    redline_y =  np.linspace(0,2*np.max(histogram[0]), 50)
    redline_x_16 = np.ones(50)*4*voxelsize*voxelsize*voxelsize 
    plt.plot(redline_x_16, redline_y, 'g', label = '4 voxel count threshold')
    plt.xscale('log')    
    plt.xlabel(r"Volume [$\mu m^3$]")
    plt.ylabel("Probability")
    plt.ylim(0, 1.01*np.max(histogram[0]))
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(str(j) + "vols_new" + str(k+ 100) + ".png")
    plt.clf()

def save_parts(k):
    os.makedirs('./analyze_pore_samples/results/pore_examples', exist_ok= True)
    limit = 2000
    index = 0
    voxelsize, imstack_test, _, _  = load_data(k, num = limit, start = index)
    os.makedirs('./analyze_pore_samples/results/pore_examples/part{}'.format(k), exist_ok = True)
    for case in range(100):
        plt.imsave('./analyze_pore_samples/results/pore_examples/part{}'.format(k) + '/frame{}.png'.format(1000+case), imstack_test[:,:, case])
        plt.clf()
    os.system('convert ./analyze_pore_samples/results/pore_examples/part{}/*png ./analyze_pore_samples/results/pore_examples/part{}.gif'.format(k,k))
    os.system('rm ./analyze_pore_samples/results/pore_examples/part{}/*png'.format(k))
    
    return

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-folder", "--folder", dest = 'folder', default = "1", help="which folder to load")
    args = parser.parse_args()
    k = int(args.folder)
    if 'analyze_porosity_clean.py' in os.listdir():
        os.chdir('..')
    for k in range(2,12):
        analyze_boundaries(k, 2000)
        process_images(k = 0,  frame_window = 200)



if __name__ == "__main__":
    main()
