import numpy  as np 
from matplotlib import pyplot as plt
import pickle
import pandas
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import measure
import imageio
import skimage
from scipy import spatial
import time
import h5py
import os
import numpy as np 
import sys
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure, transform, segmentation, morphology
from pylab import gca
from scipy.ndimage.interpolation import geometric_transform
import numpy as np 
import torch
import os
from  reconstruction.polar_cartesian_convert import linear_polar, polar_linear, map_pixel, calc_theta, unmap_pixel
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
import numpy as np
import scipy.ndimage as ndimage


import pandas as pd

import numpy as np 
import matplotlib.pyplot as plt 
import os

import numpy as np
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def read_file(fname):
    f = h5py.File(fname, 'r') 
    return f['data']
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])




'''
Make frame thicker, make tick pointing inside, make tick thicker
default frame width is 2, default tick width is 1.5
'''
def frame_tick(frame_width = 2, tick_width = 1.5):
    ax = gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction = 'in', 
                    width = tick_width)

'''
legend:
default location : upper left
default fontsize: 8
Frame is always off
'''
def legend(location = 'upper left', fontsize = 8):
    plt.legend(loc = location, fontsize = fontsize, frameon = False)
    
'''
savefig:
bbox_inches is always tight
'''
def savefig(filename):
    plt.savefig(filename, bbox_inches = 'tight')


def topolar(img, order=1):
    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order. 
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal 
    # from a corner to the mid-point of img.
    max_radius = 0.5*np.linalg.norm( img.shape )

    def transform(coords):
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)

        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]
        return i,j

    polar = geometric_transform(img, transform, order=order)

    rads = max_radius * np.linspace(0,1,img.shape[0])
    angs = np.linspace(0, 2*np.pi, img.shape[1])

    return polar, (rads, angs)
def improve_pairplot(g, replacements):
    g.fig.set_dpi(300) 
    for idx,i in enumerate(g.axes[0]):
        for idx_j,j in enumerate(g.axes):
            g.axes[idx_j][idx].spines['left'].set_linewidth(2)
            g.axes[idx_j][idx].spines['bottom'].set_linewidth(2)
            g.axes[idx_j][idx].tick_params(direction = 'in', width = 1.5)
            xlabel = g.axes[idx_j][idx].get_xlabel()
            ylabel = g.axes[idx_j][idx].get_ylabel()
            if xlabel in replacements.keys():
                g.axes[idx_j][idx].set_xlabel(replacements[xlabel], fontsize = 18)
            if ylabel in replacements.keys():
                g.axes[idx_j][idx].set_ylabel(replacements[ylabel], fontsize = 18)
    return g
def analyze_pore(im):
    voxelsize = 1

    realvols = []
    realorientations = []
    realanisotropies = []
    realmin_axis_l = []
    realmaj_axis_l = []
    im = np.array(im, dtype = 'int')
    pores = measure.regionprops(im)
    if len(pores) == 0:
        return np.array([0,0,0])
    pore_idx = np.argmax([pore.area for pore in pores])
    if pores[pore_idx].area > 1:
        realvols.append(pores[pore_idx].area)
    realmaj_axis_l.append((pores[pore_idx].major_axis_length)*voxelsize)
    realmin_axis_l.append((pores[pore_idx].minor_axis_length)*voxelsize)


    pore = pores[pore_idx]

    inertia_eigval = pore.inertia_tensor_eigvals
    inertia = pore.inertia_tensor
    maxeig = np.argmax(inertia_eigval)
    eigvec = np.linalg.eig(pore.inertia_tensor)[1]
    eigvals = np.linalg.eig(pore.inertia_tensor)[0]
    anis = 0
    realanisotropies.append(anis)
    maxvector = eigvec[:, maxeig]
    orientation = angle_between(maxvector, np.array([0,0,1]))
    realorientations.append(orientation)
    return np.array([anis, realvols[0]**(1/3), orientation/np.pi])

def load_image(fname, dim = 4):
    im = np.loadtxt(fname).reshape(566,571,dim)
    img = np.copy(im)
    boundary = np.zeros((im.shape))
    boundary[img == [1, 1, 1]] = 1
    boundary = boundary[:,:, 0]
    img[im == [1, 1, 1]] = 0
    return boundary, img

def load_shell(fname,num_frames = 200, dim = 4, start = 0):
    total = num_frames
    im_stack = np.zeros((566,571, total, dim))
    shells = np.zeros((566,571, total))
    for i in range(start, start+total):
        im = np.loadtxt(fname+'tiff_threephase_id_'+str(i)).reshape(566,571,dim)
        print(np.unique(im[:,:,2]))
        print(fname+'tiff_threephase_id_'+str(i))

        img = np.copy(im)
        boundary = np.zeros((im.shape))
        boundary[img == [1, 1, 1]] = 1

        img[im == [1, 1, 1]] = 0
        boundary = boundary[:,:, 0]
        shells[:,:,i-start] = boundary
        im_stack[:,:,i-start] = img

        print(i)
    
    return shells ,im_stack



def compute_statistics(pore_dataset, voxelsize, imstack = None):
    anisotropies = []
    orientations = []
    vols = []
    sphericity = []
    x_locs = []
    y_locs = []
    z_locs = []
    maj_axis_l = []
    min_axis_l = []
    phis = []
    for i in range(len(pore_dataset)):
        pore = pore_dataset[i]

        if pore_dataset[i]['area'] == 1:
            continue
        vols.append(pore_dataset[i]['area']*voxelsize*voxelsize*voxelsize)
        y_locs.append(pore_dataset[i]['centroid'][1]*voxelsize)
        x_locs.append(pore_dataset[i]['centroid'][0]*voxelsize)
        z_locs.append(pore_dataset[i]['centroid'][2]*voxelsize)
        maj_axis_l.append((pore_dataset[i].major_axis_length)*voxelsize)
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
        'maj_axis_l' : maj_axis_l,
        'z_locs' : z_locs, 
        'phis': phis
    }
    if not len(np.unique([len(stat) for stat in stats.values()])) == 1:
        print([len(stat) for stat in stats.values()])
        breakpoint()
    
    return stats#, examples 
def extract_pores(imstack):
    boundary_imstack = np.copy(imstack)
    boundary_imstack[imstack != 255] = 0
    boundary_imstack[imstack == 255] = 1
    
    imstack[imstack  == 255] = 0
    imstack[imstack == 159] = 1
    imstack = np.array(imstack, dtype = 'uint8')                   
    im = measure.label(imstack[:,:,:])
    props = skimage.measure.regionprops(im[:,:,:])
    return props, im, boundary_imstack


def draw_centroid(label_img, frame_id = 0):
    plt.imshow(label_img, cmap = 'binary')
    
    props =  measure.regionprops(label_img)
    prop_labels = [prop.label for prop in props]
    for idx in np.unique(label_img):
        if idx != 0:

            prop_idx = np.where(prop_labels == idx)[0][0] 
            centroid = props[prop_idx].centroid
            plt.text(centroid[1], centroid[0], idx)
    plt.title('Frame ' + str(frame_id))
    plt.savefig('centroidgen'+ str(frame_id) + '.png')
    plt.clf()

def plot_image(images,shell, title, global_idx = 0):
    os.makedirs(title, exist_ok = True)
    print("SAVING TO", title)
    for p in range(images.shape[2]):
        plt.imsave(title + str(global_idx+p) + '.png',np.array((images[:,:, p]*(255//2) + shell[:,:,p]*255), dtype = int), cmap = 'gist_gray')
        plt.clf()


def load_existing(start = 0, num_frames = 200):
    total = num_frames
    seconds = time.time()
    shell, im_stack = load_shell('/media/cmu/DATA/francis/pore_gan/8approxstyleganthreechannelboundaryfulltest_threephase_2D/', dim = 4, start = start, num_frames = num_frames)
    pore_part = np.zeros((566,571, total)) #shell.shape
    sum_tot = np.sum(im_stack[:,:,:], axis = 3)
    sum_tot[sum_tot > 0] = 1
    return pore_part, shell, im_stack

def read_from_file(filename, iteration = None, restart = None):
    tensors = torch.load(filename)
    im_opt = np.squeeze(tensors['tensor_opt'])
    plt.imshow(im_opt, cmap = 'binary')
    plt.colorbar()
    plt.title('MST generated surface roughness profiles, iteration: ' + str(iteration) + ' instance: ' + str(restart))
    plt.savefig('modelCfigs' + str(restart) + '_' + str(iteration) +'profiletest' + '.png')

    plt.clf()

def load_boundary(num_frames = 500, start = 0, pore_part_shape = (566, 571), return_profile = False, im_opt = None, resultsdir = None, folder_index = 0):
    i = folder_index

    if resultsdir is None:
        resultsdir = 'make_surface/results/sample_number_0original_folder_{}'.format(i)
    # i = 0
    xmean = np.loadtxt(resultsdir+ '/xmean{}'.format(i))
    ymean = np.loadtxt(resultsdir+ '/ymean{}'.format(i))
    minmax = pd.read_csv(resultsdir+ '/minmax_values{}.csv'.format(i,i))
    print(minmax['max'])
    print(minmax['min'])
    maxim = np.array(minmax['max'])
    minim = np.array(minmax['min'])
    ratio = minim/maxim


    if im_opt is None:
        fname = resultsdir + '/modelC_krec' + str(0) + '_start' + str(1) + '.pt' 
        tensors = torch.load(fname) 
        im_opt = np.array(np.squeeze(tensors['tensor_opt']) + xmean+ymean)   
    print(i)

    polar_index_r = linear_polar(np.zeros(pore_part_shape)).shape[0]
    polar_index_theta = linear_polar(np.zeros(pore_part_shape)).shape[1]
    new_im = resize(im_opt, (2000, polar_index_theta), order = 3)
    line = new_im[0]

    line_int = np.array(new_im[0], dtype = 'int')
    polar_image = np.zeros((polar_index_r, polar_index_theta))
    idxs = (line_int, np.arange(polar_index_theta, dtype = int))
    polar_image[idxs] = 1
    shells = np.zeros((pore_part_shape[0], pore_part_shape[1], num_frames), dtype = 'uint8')




    xs = np.linspace(0, pore_part_shape[0], pore_part_shape[0])
    ys = np.linspace(0, pore_part_shape[1], pore_part_shape[1])

    # full coordinate arrays

    xx, yy = np.meshgrid(xs, ys)
    zz = np.sqrt((xx - pore_part_shape[0]//2)**2 + (yy - pore_part_shape[1]//2)**2)
    toprow = zz[0]
    bottomrow = zz[-1]
    left = zz[:,0]


    right = zz[:,-1]
    continuous = np.hstack((toprow, bottomrow, left, right))
    # Calculate the radius at which it would
    edge_radii = np.array(continuous)
    max_valid_radius = np.min(edge_radii)
    min_valid_radius = ratio*max_valid_radius

    for k in range(start, num_frames+start):
        line = new_im[k]*(max_valid_radius - min_valid_radius) + min_valid_radius
        line[-1] = line[0]
        line = savgol_filter(line, 27 , 3)
        line_int = np.array(line, dtype = 'int')
        polar_image = np.zeros((polar_index_r, polar_index_theta))
        idxs = (line_int, np.arange(polar_index_theta, dtype = int))
        polar_image[idxs] = 1
        shell_img = polar_linear(polar_image, output = (pore_part_shape[0], pore_part_shape[1]))
        shells[:,:, k-start] = np.array(shell_img > 0,dtype='uint8')

    if return_profile:
        return shells, new_im*(max_valid_radius - min_valid_radius) + min_valid_radius
    return shells

def analyze_results(original, pore_reconstruct, fname = None, lists_all_original = None, lists_all_new = None, voxelsize = 3.49):
    print('FINISHED RECONSTRUCTION')
    title = fname
    frame_tick()
    os.makedirs(title, exist_ok = True)
    if lists_all_original == None:
        props, _, _ = extract_pores(pore_reconstruct)
        props_orig, _, _ = extract_pores(original)
        stats = compute_statistics(props, voxelsize=voxelsize)
        # stats_orig = compute_statistics(props_orig, voxelsize=3.49)


        total_anisotropies = []
        total_x = []
        total_y = []
        total_orientations = []
        total_z = []
        total_maj = []
        total_vols = []

        total_phis = []
        gt_total_phis =[]
        total_x.extend(stats['x_locs'])
        total_y.extend(stats['y_locs'])
        total_maj.extend(stats['maj_axis_l'])
        total_vols.extend(stats['vols'])
        total_anisotropies.extend(stats['anisotropies'])
        total_orientations.extend(stats['orientations'])
        total_phis.extend(stats['phis'])
        total_z.extend(stats['z_locs'])
        stats = compute_statistics(props_orig, voxelsize=voxelsize)


        gt_total_anisotropies = []
        gt_total_x = []
        gt_total_y = []
        gt_total_orientations = []
        gt_total_z = []
        gt_total_maj = []

        gt_total_vols = []

        gt_total_x.extend(stats['x_locs'])
        gt_total_y.extend(stats['y_locs'])
        gt_total_maj.extend(stats['maj_axis_l'])
        gt_total_vols.extend(stats['vols'])
        gt_total_anisotropies.extend(stats['anisotropies'])
        gt_total_orientations.extend(stats['orientations'])
        gt_total_phis.extend(stats['phis'])
        gt_total_z.extend(stats['z_locs'])


        lists_all_original = {'x_locs': gt_total_x, 'y_locs': gt_total_y, 'maj_axis_l': gt_total_maj, 'vols': gt_total_vols, 'anisotropies':gt_total_anisotropies, 'orientations': gt_total_orientations, 'z_locs': gt_total_z, 'phis' : gt_total_phis }
        lists_all_new  = {'x_locs': total_x, 'y_locs': total_y, 'maj_axis_l': total_maj, 'vols': total_vols, 'anisotropies':total_anisotropies, 'orientations': total_orientations, 'z_locs': total_z, 'phis': total_phis }


    density = True
    fig = plt.figure(figsize=[4,3], dpi = 300)             
    histogram = plt.hist((np.array(lists_all_new['orientations'])/np.pi)*180, density = density, bins=30, edgecolor = 'k', label = 'reconstructed', alpha  = 0.7)   
    histogram2 = plt.hist((np.array(lists_all_original['orientations'])/np.pi)*180, density = density, bins=30, edgecolor = 'k', label = 'original', alpha = 0.7) 
    plt.title("Orientation")

    np.savetxt(title+ 'num_pores', np.array([len(lists_all_original['orientations']), len(lists_all_new['orientations'])]))
    plt.xlabel(r"Angle [Degrees]") 
    plt.ylabel("Probability")
    frame_tick()
    legend()
    plt.tight_layout()
    plt.savefig(title + "orientation" + ".png")
    plt.show()
    plt.clf()




    fig = plt.figure(figsize=[4,3], dpi = 300)
    histogram = plt.hist((np.array(lists_all_new['phis'])/np.pi)*180, density = density, bins=30, edgecolor = 'k', label = 'reconstructed', alpha  = 0.7)   
    histogram2 = plt.hist((np.array(lists_all_original['phis'])/np.pi)*180, density = density, bins=30, edgecolor = 'k', label = 'original', alpha = 0.7)              
    plt.title("Phi")
    np.savetxt(title+ 'num_pores', np.array([len(lists_all_original['phis']), len(lists_all_new['phis'])]))
    plt.xlabel(r"Angle [Degrees]") 
    plt.ylabel("Probability")
    frame_tick()
    legend()
    plt.tight_layout()
    plt.savefig(title + "phi" + ".png")
    plt.show()
    plt.clf()


    fig = plt.figure(figsize=[4,3], dpi = 300)
    histogram = plt.hist((np.array(lists_all_new['anisotropies'])), density = density, bins=30, edgecolor = 'k', label = 'reconstructed', alpha  = 0.7)    
    histogram2 = plt.hist(np.array(lists_all_original['anisotropies']), density = density, bins=30, edgecolor = 'k', label = 'original', alpha = 0.7)               
    plt.title("Anisotropy")
    plt.xlabel(r"Anisotropy")
    plt.ylabel("Probability")
    frame_tick()
    legend()
    plt.tight_layout()
    plt.savefig(title+"anisotropy" + ".png")
    plt.show()
    plt.clf()
    fig = plt.figure(figsize=[4,3], dpi = 300)
    histogram = plt.hist((np.array(lists_all_new['y_locs'])), density = density, bins=30, edgecolor = 'k', label = 'reconstructed', alpha  = 0.7)
    histogram2 = plt.hist((np.array(lists_all_original['y_locs'])), density = density, bins=30, edgecolor = 'k', label = 'original', alpha = 0.7)                   
    plt.title("Y location")
    plt.xlabel(r"Y location [micrometers]")
    plt.ylabel("Probability")
    frame_tick()
    legend()
    plt.tight_layout()
    plt.savefig(title+"yloc" + ".png")
    plt.show()

    plt.clf()
    fig = plt.figure(figsize=[4,3], dpi = 300)

    histogram = plt.hist((np.array(lists_all_new['x_locs'])), density = density, bins=30, edgecolor = 'k', label = 'reconstructed', alpha  = 0.7)
    histogram2 = plt.hist((np.array(lists_all_original['x_locs'])), density = density, bins=30, edgecolor = 'k', label = 'original', alpha = 0.7)                   
    plt.title("X location")
    plt.xlabel(r"X location [micrometers]")
    plt.ylabel("Probability")
    frame_tick()
    legend()
    plt.tight_layout()
    plt.savefig(title+"xloc" + ".png")
    plt.show()

    plt.clf()
    fig = plt.figure(figsize=[4,3], dpi = 300)

    histogram = plt.hist(lists_all_new['vols'], density = density, alpha = 0.7,bins=np.logspace(np.log10(10e1),np.log10(10e5)), edgecolor = 'k', label = 'reconstructed')
    histogram2 = plt.hist(lists_all_original['vols'], density = density, alpha = 0.7,bins=np.logspace(np.log10(10e1),np.log10(10e5)), edgecolor = 'k', label = 'original')

    plt.title("Volume")

    plt.xscale('log')    
    plt.xlabel(r"Volume [$\mu m^3$]")
    plt.ylabel("Probability")
    frame_tick()
    legend()
    plt.tight_layout()
    plt.savefig(title + "vols" + ".png")
    plt.show()
    plt.clf()

    print("Pores in the original sample: " + str(len(lists_all_original['vols'])))
    print("Pores in the new sample: " + str(len(lists_all_new['vols'])))
def replace_sampling(pore_part,  generated_boundary, n_bins = 30, window_size = 100, properties_folder = './analyze_pore_samples/results/pore_properties/probability_matrices/', use_generated = True, use_gt = True):
    import scipy

    prob_matrix_volume=np.load(properties_folder + str(n_bins) + '_{}allprob_matrix_volume.npy'.format(0))
    prob_matrix_num =  np.load(properties_folder + str(n_bins) + '_{}allprob_matrix_num.npy'.format(0))*(window_size/100)#/2
    bin_edges_vols =   np.load(properties_folder + str(n_bins) + '_{}allbin_edges_vols.npy'.format(0))
    bin_edges_anis =   np.load(properties_folder + str(n_bins) + '_{}allbin_edges_anis.npy'.format(0))
    bin_edges_phis = np.load(properties_folder + str(n_bins) + '_{}allbin_edges_phis.npy'.format(0))
    prob_matrix_phis = np.load(properties_folder + str(n_bins) + '_{}allprob_matrix_phis.npy'.format(0))

    prob_matrix_anis =       np.load(properties_folder + str(n_bins) + '_{}allprob_matrix_anis.npy'.format(0))
    bin_edges_orientations = np.load(properties_folder + str(n_bins) + '_{}allbin_edges_orientations.npy'.format(0))
    prob_matrix_orientation= np.load(properties_folder + str(n_bins) + '_{}allprob_matrix_orientations.npy'.format(0))

    target_list_size = []
    target_anis = []
    target_vols = []
    actual_vols = []
    generated_pore_matrix = np.loadtxt('./reconstruction/gan/figures/pore_matrix_updated.csv')
    gt_pore_matrix = np.loadtxt('./analyze_pore_samples/results/individual_pore_samples/partsample0/pore_matrix')
    x_extent = np.linspace(0,pore_part.shape[0], n_bins)
    y_extent = np.linspace(0, pore_part.shape[1], n_bins)
    z_extent = np.arange(0, pore_part.shape[2],window_size)
    gt_pores_used = []
    gen_pores_used = []
    gen_losses = []
    gt_losses = []
    for idx_x,x_sample in enumerate(x_extent):
        for idx_y,y_sample  in enumerate(y_extent):
            for idx_z, z_sample in enumerate(z_extent):
                
                vol_hist = prob_matrix_volume[idx_x, idx_y, :]

                vol_hist_dist = scipy.stats.rv_histogram((vol_hist, bin_edges_vols))
                num = prob_matrix_num[idx_x, idx_y]
                if num == 0:
                    continue
                if num < 1:
                    unif_sample = np.random.uniform()
                    if num > unif_sample:
                        num = 1
                    else:
                        num = 0
                        continue
                elif num > 1:
                    num = int(np.around(num))

                volumes = vol_hist_dist.rvs(size=int(num))
                ani_hist = prob_matrix_anis[idx_x, idx_y]
                ani_hist_dist = scipy.stats.rv_histogram((ani_hist, bin_edges_anis))
                anisotropies = ani_hist_dist.rvs(size=int(num))


                angle_hist = prob_matrix_orientation[idx_x, idx_y]
                angle_hist_dist = scipy.stats.rv_histogram((angle_hist, bin_edges_orientations))
                angles = angle_hist_dist.rvs(size=int(num))
                phi_hist = prob_matrix_phis[idx_x, idx_y]
                phi_hist_dist = scipy.stats.rv_histogram((phi_hist, bin_edges_phis))
                phis = phi_hist_dist.rvs(size = int(num))
                target_anis.extend(anisotropies)
                target_vols.extend(np.array(volumes)**3)
                for idx_pore,gen_pore in enumerate(volumes):
                    
                    gen_pore = np.max([gen_pore, 2])

                    tmp_x = np.random.randint(0,int(pore_part.shape[0]/n_bins))
                    tmp_y = np.random.randint(0,int(pore_part.shape[1]/n_bins))

                    curr_x = int(x_sample-tmp_x)
                    curr_y = int(y_sample - tmp_y)

                    curr_z = z_sample+np.random.randint(0,window_size)

                    target_pore= gen_pore**3#/25
                    target_list_size.append(target_pore)
                    anis = anisotropies[idx_pore]
                    angle = angles[idx_pore]
                    phi = phis[idx_pore]

                    pore_matrix = generated_pore_matrix
                    pore_matrix = generated_pore_matrix
                    pore_sizes = (pore_matrix[:,0])
                   
                    gen_pore_identity = np.argmin(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  generated_pore_matrix[:,2])/np.mean(generated_pore_matrix[:,2]) + np.abs(angle - generated_pore_matrix[:,3])/np.mean(generated_pore_matrix[:,3])+  np.abs(phi - generated_pore_matrix[:,4])/np.mean(generated_pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))
                    gen_loss = np.min(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  pore_matrix[:,2])/np.mean(pore_matrix[:,2]) + np.abs(angle - pore_matrix[:,3])/np.mean(pore_matrix[:,3])+  np.abs(phi - generated_pore_matrix[:,4])/np.mean(generated_pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))
                    
                    pore_matrix = gt_pore_matrix
                    gt_pore_identity = np.argmin(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  pore_matrix[:,2])/np.mean(pore_matrix[:,2]) + np.abs(angle - pore_matrix[:,3])/np.mean(pore_matrix[:,3])+  np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))
                    gt_loss = np.min(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  pore_matrix[:,2])/np.mean(pore_matrix[:,2]) + np.abs(angle - pore_matrix[:,3])/np.mean(pore_matrix[:,3])+  np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))
                    
                    gen_losses.append(gen_loss)
                    gt_losses.append(gt_loss)

                    if use_generated and use_gt: # Use both generated and ground truth pores
                        if gen_loss > gt_loss:
                            generated = False
                            pore_identity = gt_pore_identity
                            gt_pores_used.append(pore_identity)
                        else:
                            generated = True 
                            pore_identity = gen_pore_identity
                            gen_pores_used.append(pore_identity)
                    elif use_generated: # only use generated
                        generated = True 
                        pore_identity = gen_pore_identity
                        gen_pores_used.append(pore_identity)
                    else: # only use ground truth

                        generated = False
                        pore_identity = gt_pore_identity
                        gt_pores_used.append(pore_identity)

                    if generated:
                        filename = './reconstruction/gan/saved_generated_pores/generator_' + str(pore_identity) +'.hdf5'
                    else:
                        filename   = './analyze_pore_samples/results/individual_pore_samples/partsample0/pore_original_' + str(pore_identity) + '.hdf5'

                    data = read_file(filename)

                    if len(np.where(data)[0]) == 0:
                        print('continue 1 activated')
                        continue
                    center = 32
                    size = 32
                    if generated:
                        data = np.array(data)#/255
                    else:
                        data = np.array(data)
                    xmin = np.min(np.where(data)[0])
                    ymin = np.min(np.where(data)[1])
                    zmin = np.min(np.where(data)[2])

                    xmax = np.max(np.where(data)[0])+1
                    ymax = np.max(np.where(data)[1])+1
                    zmax = np.max(np.where(data)[2])+1


                    xmin_slice = np.min(np.where(data[:,:, zmin])[0])
                    ymin_slice = np.min(np.where(data[:,:,zmin])[1])
                    lowerlim_x =  0 
                    lowerlim_y =  0 
                    data_ylower = 0
                    data_xlower = 0


                    data_yupper = 64-np.abs(np.min((0, pore_part.shape[1] - (lowerlim_y + 64)))) # in case of negative indices
                    data_xupper = 64-np.abs(np.min((0, pore_part.shape[0] - (lowerlim_x + 64))))
                    target_z = curr_z
                    if int(target_z) + (zmax-zmin) > pore_part.shape[2]:
                        print('continue 2 activated: pore on back surface')
                        continue
                    if int(target_z) < 0:
                        print('continue 3 activated: pore on front surface')
                        continue
                    try:
                        test_window = pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]  + data[xmin:xmax , ymin:ymax, zmin:zmax]
                    except Exception as e:
                        print(e)
                        print('continue 4 activated: indexing exception')
                        continue
                    if 0 in test_window.shape:
                        print('continue 5 activated: unresolved pore')
                        continue
                    
                    elif np.max(test_window) > 1:

                        print('continue 6 activated: Collision')
                        collision_z = np.where(test_window >1)[2][0]
                        continue

                    try:
                        xdatastart = np.min(np.where(data)[0])
                        labelpore = measure.label(data[xmin:xmax , ymin:ymax, zmin:zmax])

                        oldpore =  pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]

                        newpore = data[xmin:xmax , ymin:ymax, zmin:zmax] +pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]
                        if len(measure.regionprops(measure.label(newpore))) != len(measure.regionprops(measure.label(oldpore))) + 1:
                            print(len(measure.regionprops(measure.label(newpore))),len(measure.regionprops(measure.label(oldpore))), 'cmerged')
                            print('continue 7 activated: Collision')
                            continue

                        if generated:
                            pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)] = 2*data[xmin:xmax , ymin:ymax, zmin:zmax]+ pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]
                        else:
                            pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)] = data[xmin:xmax , ymin:ymax, zmin:zmax]+ pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]

                    except Exception as f:
                        print(f)
                        print('continue 8 activated: Insertion process failed')

                        continue

                    actual_vols.append(measure.regionprops(measure.label(np.array(data)))[0].area)
    # breakpoint()
    return pore_part, target_list_size


def replace_sampling_polar(pore_part,  generated_boundary, profile_2d, n_bins = 30, window_size = 100, properties_folder = './analyze_pore_samples/results/pore_properties/probability_matrices/', folder_index = 0):
    import scipy#.stats
    prob_matrix_dir = properties_folder
    np.load(properties_folder + str(n_bins) + '_{}allprob_matrix_volume.npy'.format(0))
    prob_matrix_volume=np.load(prob_matrix_dir + str(n_bins) + '_{}polarprob_matrix_volume.npy'.format(folder_index))
    prob_matrix_num =  np.load(prob_matrix_dir + str(n_bins) + '_{}polarprob_matrix_num.npy'.format(folder_index))*(window_size/100)#/2
    bin_edges_vols =   np.load(prob_matrix_dir + str(n_bins) + '_{}polarbin_edges_vols.npy'.format(folder_index))
    bin_edges_anis =   np.load(prob_matrix_dir + str(n_bins) + '_{}polarbin_edges_anis.npy'.format(folder_index))
    bin_edges_phis = np.load(prob_matrix_dir + str(n_bins) + '_{}polarbin_edges_phis.npy'.format(folder_index))
    prob_matrix_phis = np.load(prob_matrix_dir + str(n_bins) + '_{}polarprob_matrix_phis.npy'.format(folder_index))
    prob_matrix_anis =       np.load(prob_matrix_dir + str(n_bins) + '_{}polarprob_matrix_anis.npy'.format(folder_index))
    bin_edges_orientations = np.load(prob_matrix_dir + str(n_bins) + '_{}polarbin_edges_orientations.npy'.format(folder_index))
    prob_matrix_orientation= np.load(prob_matrix_dir + str(n_bins) + '_{}polarprob_matrix_orientations.npy'.format(folder_index))
    target_list_size = []
    target_anis = []
    target_vols = []
    actual_vols = []
    generated_pore_matrix = np.loadtxt('./reconstruction/gan/figures/pore_matrix_updated.csv')
    gt_pore_matrix = np.loadtxt('./analyze_pore_samples/results/individual_pore_samples/partsample0/pore_matrix')
    r_extent = np.linspace(0,1.1, n_bins)
    theta_extent = np.linspace(0, np.pi*2, n_bins)
    z_extent = np.arange(0, pore_part.shape[2],window_size)
    gt_pores_used = []
    gen_pores_used = []
    gen_losses = []
    gt_losses = []
    angles = np.linspace(0, np.pi*2, profile_2d.shape[1])
    radii_totals = []
    for idx_x,x_sample in enumerate(r_extent):
        for idx_y,y_sample  in enumerate(theta_extent):
            for idx_z, z_sample in enumerate(z_extent):
                
                vol_hist = prob_matrix_volume[idx_x, idx_y, :]

                vol_hist_dist = scipy.stats.rv_histogram((vol_hist, bin_edges_vols))
                num = prob_matrix_num[idx_x, idx_y]
                if num == 0:
                    continue
                if num < 1:
                    unif_sample = np.random.uniform()
                    if num > unif_sample:
                        num = 1
                    else:
                        num = 0
                        continue
                elif num > 1:
                    num = int(np.around(num))

                volumes = vol_hist_dist.rvs(size=int(num))
                ani_hist = prob_matrix_anis[idx_x, idx_y]
                ani_hist_dist = scipy.stats.rv_histogram((ani_hist, bin_edges_anis))
                anisotropies = ani_hist_dist.rvs(size=int(num))


                angle_hist = prob_matrix_orientation[idx_x, idx_y]
                angle_hist_dist = scipy.stats.rv_histogram((angle_hist, bin_edges_orientations))
                angles = angle_hist_dist.rvs(size=int(num))
                phi_hist = prob_matrix_phis[idx_x, idx_y]
                phi_hist_dist = scipy.stats.rv_histogram((phi_hist, bin_edges_phis))
                phis = phi_hist_dist.rvs(size = int(num))

                target_anis.extend(anisotropies)
                target_vols.extend(np.array(volumes)**3)
                print(x_sample, "RADIUS SAMPLE")
                for idx_pore,gen_pore in enumerate(volumes):
                    
                    gen_pore = np.max([gen_pore, 2])
                    tmp_x = np.random.uniform(0,r_extent[1])
                    tmp_y = np.random.uniform(0,theta_extent[1])
                    curr_radius = (x_sample-tmp_x)
                    curr_angle = (y_sample - tmp_y)
                    idx_angle = np.argmin(np.abs(curr_angle-np.linspace(0,2*np.pi, profile_2d.shape[1])))

                    curr_z = z_sample+np.random.randint(0,window_size)
                    radius_total = profile_2d[curr_z, idx_angle]
                    full_radii = curr_radius*radius_total
                    if x_sample == r_extent[-1]:
                        print(curr_radius, "Current Radius, final evolution")
                    curr_x, curr_y = unmap_pixel(full_radii, theta_idx = idx_angle, theta = curr_angle,output= pore_part[:,:,0].shape)
                    radii_totals.append(curr_radius)

                    target_pore= gen_pore**3
                    target_list_size.append(target_pore)
                    anis = anisotropies[idx_pore]
                    angle = angles[idx_pore]
                    phi = phis[idx_pore]

                    pore_matrix = generated_pore_matrix
                    pore_matrix = generated_pore_matrix
                    pore_sizes = (pore_matrix[:,0])
                    
                    gen_pore_identity = np.argmin(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  generated_pore_matrix[:,2])/np.mean(generated_pore_matrix[:,2]) + np.abs(angle - generated_pore_matrix[:,3])/np.mean(generated_pore_matrix[:,3])+  np.abs(phi - generated_pore_matrix[:,4])/np.mean(generated_pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))
                    gen_loss = np.min(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  pore_matrix[:,2])/np.mean(pore_matrix[:,2]) + np.abs(angle - pore_matrix[:,3])/np.mean(pore_matrix[:,3])+  np.abs(phi - generated_pore_matrix[:,4])/np.mean(generated_pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))

                    pore_matrix = gt_pore_matrix
                    gt_pore_identity = np.argmin(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  pore_matrix[:,2])/np.mean(pore_matrix[:,2]) + np.abs(angle - pore_matrix[:,3])/np.mean(pore_matrix[:,3])+  np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))
                    gt_loss = np.min(np.abs(target_pore - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  pore_matrix[:,2])/np.mean(pore_matrix[:,2]) + np.abs(angle - pore_matrix[:,3])/np.mean(pore_matrix[:,3])+  np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))# + np.abs(phi - pore_matrix[:,4])/np.mean(pore_matrix[:,4]))
                    gen_losses.append(gen_loss)
                    gt_losses.append(gt_loss)
                   
                    if gen_loss > gt_loss:
                        generated = False
                        pore_identity = gt_pore_identity
                        gt_pores_used.append(pore_identity)
                    else:

                        generated = True 
                        pore_identity = gen_pore_identity
                        gen_pores_used.append(pore_identity)

                    if generated:
                        filename = './reconstruction/gan/saved_generated_pores/generator_' + str(pore_identity) +'.hdf5'
                    else:
                        filename   = './analyze_pore_samples/results/individual_pore_samples/partsample0/pore_original_' + str(pore_identity) + '.hdf5'
                    data = read_file(filename)
                    if len(np.where(data)[0]) == 0:
                        print('Empty pore, skipped')
                        continue

                    center = 32
                    size = 32
                    if generated:
                        data = np.array(data)#/255
                    else:
                        data = np.array(data)
                
                    xmin = np.min(np.where(data)[0])
                    ymin = np.min(np.where(data)[1])
                    zmin = np.min(np.where(data)[2])

                    xmax = np.max(np.where(data)[0])+1
                    ymax = np.max(np.where(data)[1])+1
                    zmax = np.max(np.where(data)[2])+1


                    xmin_slice = np.min(np.where(data[:,:, zmin])[0])
                    ymin_slice = np.min(np.where(data[:,:,zmin])[1])


                    lowerlim_x =  0 
                    lowerlim_y =  0
                    data_ylower = 0
                    data_xlower = 0


                    data_yupper = 64-np.abs(np.min((0, pore_part.shape[1] - (lowerlim_y + 64)))) # in case of negative indices
                    data_xupper = 64-np.abs(np.min((0, pore_part.shape[0] - (lowerlim_x + 64))))
                    target_z = curr_z
                    if int(target_z) + (zmax-zmin) > pore_part.shape[2]:
                        print('continue 2 activated')
                        continue
                    if int(target_z) < 0:
                        print('continue 3 activated')
                        continue
                    try:
                        test_window = pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]  + data[xmin:xmax , ymin:ymax, zmin:zmax]
                    except Exception as e:
                        print(e)
                        print('continue 4 activated')
                        continue
                    if 0 in test_window.shape:
                        print('continue 5 activated')
                        continue
                    
                    elif np.max(test_window) > 1:
                        print('continue 6 activated')
                        collision_z = np.where(test_window >1)[2][0]
                        continue
                    try:
                        xdatastart = np.min(np.where(data)[0])
                        labelpore = measure.label(data[xmin:xmax , ymin:ymax, zmin:zmax])

                        oldpore =  pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]
                        newpore = data[xmin:xmax , ymin:ymax, zmin:zmax] +pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]
                        if len(measure.regionprops(measure.label(newpore))) != len(measure.regionprops(measure.label(oldpore))) + 1:
                            print(len(measure.regionprops(measure.label(newpore))),len(measure.regionprops(measure.label(oldpore))), 'cmerged')
                            print('continue 7 activated')
                            continue
                        if generated:
                            pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)] = data[xmin:xmax , ymin:ymax, zmin:zmax]+ pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]
                        else:
                            pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)] = data[xmin:xmax , ymin:ymax, zmin:zmax]+ pore_part[curr_x: curr_x+ int(xmax-xmin), curr_y:curr_y + int(ymax-ymin), int(target_z):int(target_z)+(zmax-zmin)]
                    except Exception as f:
                        print(f)
                        print('continue 8 activated')
                        continue
                    actual_vols.append(measure.regionprops(measure.label(np.array(data)))[0].area)
    breakpoint()
    return pore_part, target_list_size
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
def trim_boundary(shell, pore_part):
    for sample in range(pore_part.shape[2]):
     
        oldtime = time.time()
        coords = np.array(np.where(shell[:,:,sample]))
        index_anglesort = np.argsort(calc_theta(coords, shell[:,:,sample]))
        coords_pores = np.array(np.where(pore_part[:,:,sample]))
        outside_bounds = measure.points_in_poly(coords_pores.T, np.array(coords)[:, index_anglesort].T)
        test = np.copy(pore_part[:,:,sample])

        test[coords_pores[0], coords_pores[1]] = np.array(outside_bounds, dtype = 'int')
        pore_part[:,:,sample] = test
    
    return pore_part
def threshold_pore_size(profile_2d, pore_part, boundary = None, name = ''):
    im_label  = measure.label(pore_part)
    props = measure.regionprops(im_label)
    small_props= [prop for prop in props if prop.area < 8]
    centroids = [prop.centroid for prop in props]
    polar, rs, ts,o, r, out_h, out_w  = linear_polar(pore_part[:,:, 0], verbose = 1) 
    radii = []
    angles = []
    for centroid in centroids:
        
        x = centroid[0]
        y = centroid[1]
        z = int(centroid[2])
        r_index, theta_index, theta = map_pixel(int(x),int(y),pore_part[:,:,z], o = o, r =r, out_h = out_h, out_w = out_w, debug= False )
        radius = profile_2d[z, theta_index] - r_index
        oldtime = time.time()
        r_index, theta_index, theta = map_pixel(int(x),int(y),pore_part[:,:,z], o = o, r =r, out_h = out_h, out_w = out_w, debug= False )
        newtime = time.time()
        print(newtime-oldtime)
        radii.append(radius)
        angles.append(theta)



        plt.imshow(pore_part[:,:,z]+boundary[:,:,z])
        plt.scatter(int(y), int(x))
        plt.title(str(radius))
        plt.savefig('./failures/polar' + name + 'frame'+str(z))
        plt.clf()
        plt.imshow(linear_polar(boundary[:,:, z]))
        plt.scatter(theta_index, r_index)
        plt.title(str(profile_2d[z, theta_index]) + ' radius: ' + str(r_index) +' calc diff: '+ str(radius))
        plt.savefig('./failures/polar' + name + 'polarframe'+str(z))
        plt.clf()

def plt_scatter_matrices(prior_realizations, post_realizations, variable_list, plt_prior=True, plt_post=True, plt_corr=True, print_corr=True):
    '''
    Plot the prior and posterior scatter matrices (on top of each other) for a *variable_list*.
    This is from a dictionary of *prior_realizations* and *post_realizations*.
    '''
    from pandas import DataFrame
    from pandas.plotting import scatter_matrix
    import seaborn as sns

    def plt_corr_sns(corr):
        
        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax, vmin = -1.0, vmax = 1.0)

    shape_prior = np.array(prior_realizations[variable_list[0]]).shape
    shape_post = np.array(post_realizations[variable_list[0]]).shape
    prior_data = prior_realizations.copy()
    prior_data['Case'] = ['Ground Truth']*shape_prior[0]#np.zeros(shape_prior)
    post_data = post_realizations.copy()
    post_data['Case'] = ['Reconstructed']*shape_post[0]#np.ones(shape_post)
    plt.figure(figsize = [8,6], dpi = 150)
    
    if plt_prior and plt_post:
        for key in prior_data:
            prior_data[key] = np.append(prior_data[key],post_data[key])
        data = DataFrame(prior_data)
    elif plt_post:
        data = DataFrame(post_data)
    else:
        data = DataFrame(prior_data)
    data.to_csv('Generated_data.csv')
    return
def fill_boundary(shell, pore_part):
    oldtime = time.time()
    test = np.zeros(np.shape(pore_part[:,:,:]))
    for sample in range(pore_part.shape[2]):
        center = (pore_part.shape[0]//2, pore_part.shape[1]//2)
        filled  = np.array(segmentation.flood(shell[:,:,sample], (310,310), connectivity = 0), dtype = int)
        pores = np.array(pore_part[:,:,sample]>0, dtype = int)
        test[:,:,sample] = np.array((filled - pores) > 0, dtype= int)
        quadrant_check = len(np.where(test[0,0,:])[0])  + len(np.where(test[-1,0,:])[0])+ len(np.where(test[0,-1,:])[0])+ len(np.where(test[-1,-1,:])[0])
        
        if quadrant_check > 0:
            print("Segmentation didn't work, trying dilation ", sample)

            dilated = morphology.dilation(shell[:,:,sample])
            skeleton = morphology.skeletonize(dilated)
            filled = morphology.flood(np.array(skeleton,dtype = 'uint8'), (310,310), connectivity = 0)
            pores = np.array(pore_part[:,:,sample]>0, dtype = int)
            test[:,:,sample] = np.array((filled - pores) > 0, dtype= int)
            quadrant_check = len(np.where(test[0,0,:])[0])  + len(np.where(test[-1,0,:])[0])+ len(np.where(test[0,-1,:])[0])+ len(np.where(test[-1,-1,:])[0])

            if quadrant_check >0:
                print("Segmentation didn't work, trying angle based method ", sample)
                coords = np.array(np.where(shell[:,:,sample]))
                index_anglesort = np.argsort(calc_theta(coords, shell[:,:,sample]))
                coords_pores = np.array(np.where(pore_part[:,:,sample]))

                solid = measure.grid_points_in_poly(pore_part[:,:,sample].shape, np.array(coords)[:, index_anglesort].T)
                test[:,:,sample] = np.array((np.array(solid, dtype = int) - np.array(pore_part[:,:,sample]>0, dtype = int)) > 0, dtype = int)

    return test

def n_bins_study(profile_2d, pore_part, generated_boundary, gt_pores):
    threshold_pore_size(profile_2d, pore_part, generated_boundary)
    list_means = []
    for n_bins in [5, 10, 20, 30, 50,100]:
        pore_part_test, _ = replace_sampling(np.zeros((pore_part.shape[0], pore_part.shape[1], 200)), n_bins = n_bins)
        list_means.append(np.mean(pore_part_test, axis = 2))
        plt.clf()
        plt.figure(figsize = [4,3], dpi = 150)
        pore_dataset = measure.regionprops(measure.label(pore_part_test))
        
        pore_gt_dataset,_,_ = extract_pores(gt_pores[:,:,:200])#measure.regionprops(measure.label())
        locations = [pore.centroid for pore in pore_dataset]

        loc_pores = np.array(locations)
        kdtree = spatial.KDTree(locations)
        dd, ii = kdtree.query(locations, len(locations))
        voxelsize = 3.49
        nearest_neighbor = dd[:,1:]*voxelsize
        neighbors = dd[:,1]*voxelsize
        edges, hist = np.histogram(dd[:,1:]*voxelsize, bins = np.arange(0,800,25)*voxelsize)
        plt.plot(hist[:-1],edges, linewidth = 2.0,label= "Original Distribution")


        gtlocations = [pore.centroid for pore in pore_gt_dataset]
        gtloc_pores = np.array(gtlocations)
        
        gtkdtree = spatial.KDTree(gtlocations)
        gtdd, gtii = kdtree.query(gtlocations, len(gtlocations))
        gtnearest_neighbor = gtdd[:,1:]*voxelsize
        gtneighbors = gtdd[:,1]*voxelsize
        gtedges, gthist = np.histogram(gtdd[:,1:]*voxelsize, bins = np.arange(0,800,25)*voxelsize)
        plt.plot(gthist[:-1],gtedges, linewidth = 2.0, label = 'Generated Distribution')


        plt.xlabel(r"Distance [$\mu m$]")
        plt.ylabel("Number of Pores")
        plt.tight_layout()
        legend(location='best')
        frame_tick()
        plt.savefig('matriximproved_boundaryaddednbins' +"/rdf" + str(n_bins) + ".png")
        plt.clf()
        plt.imsave('matriximproved_boundaryaddednbins' +'/truedensity'+str(n_bins)+ 'example.png', np.sum(pore_part_test+generated_boundary, axis = 2), cmap= 'binary')
        plt.imsave('matriximproved_boundaryaddednbins' +'/truedensity'+str(n_bins)+ 'gt.png', np.sum(gt_pores[:,:,:500], axis = 2), cmap= 'binary')
        plt.colorbar()
        plt.savefig()
        plt.clf()

    
def save_binary_segment(segment, fname, voxelsize, index, vname = None):
    voxel_conversions=[vname]
    if segment.shape[0] > segment.shape[1]:
        l_pad = (segment.shape[0] - segment.shape[1])//2
        r_pad = (segment.shape[0] - segment.shape[1]) - l_pad 
        padstack = np.pad(segment,  ((0,0),(l_pad,r_pad), (0,0)))

    elif segment.shape[0] < segment.shape[1]:
        l_pad = (segment.shape[1] - segment.shape[0])//2
        r_pad = (segment.shape[1] - segment.shape[0]) - l_pad 
        padstack = np.pad(segment,  ((l_pad,r_pad),(0,0), (0,0)))
    else:
        padstack = segment
    for size in [64, 128, 256, 512, padstack.shape[0]]:
        
        ratio = size/padstack.shape[0]
        z_size = np.max([int(padstack[:,:,:100].shape[2]*ratio),1])
        resized_imstack = resize(np.array(padstack[:,:,:100]), (size,size,z_size), anti_aliasing=False, order =0 )
        print(ratio,padstack.shape[2],resized_imstack.shape, int(padstack.shape[0]*ratio), (padstack.shape[0]*ratio), size*voxelsize/ratio, "diameter of bounding box")
        resized_arr = np.array(resized_imstack>0,dtype='uint8')
        if size == padstack.shape[0]:
            np.save(fname + str(index) + '_' + "fullres" , resized_arr*255)
        else: 
            np.save(fname + str(index) + '_' +str(size) , resized_arr*255)
        voxel_conversions.append(voxelsize/ratio)
    return padstack.shape, voxel_conversions


def make_caps(segments, capsize = 100):
    props = measure.regionprops(measure.label(segments[:,:,0]))
    prop_max = props[np.argmax([prop.area for prop in props])]
    centroid = prop_max.centroid
    plt.plot(centroid[1], centroid[0], 'r.')
    plt.plot(centroid[1]+ prop_max.major_axis_length//2, centroid[0] , 'r.')
    from skimage.draw import circle

    rr, cc= circle(centroid[0], centroid[1], (prop_max.major_axis_length*1.03)//2, shape =segments[:,:,0].shape )
    circ_image = np.zeros(segments[:,:,0].shape)
    circ_image[rr,cc] = 1
    start = np.repeat(circ_image[:,:,np.newaxis], capsize, axis = 2)

    
    
    props = measure.regionprops(measure.label(segments[:,:,-1]))
    prop_max = props[np.argmax([prop.area for prop in props])]
    centroid = prop_max.centroid
    plt.plot(centroid[1], centroid[0], 'r.')
    plt.plot(centroid[1]+ prop_max.major_axis_length//2, centroid[0] , 'r.')
    from skimage.draw import circle
    rr, cc= circle(centroid[0], centroid[1], (prop_max.major_axis_length*1.03)//2,shape =segments[:,:,0].shape)
    circ_image = np.zeros(segments[:,:,-1].shape)
    circ_image[rr,cc] = 1
    end = np.repeat(circ_image[:,:,np.newaxis], capsize, axis = 2)

    return np.array(start, dtype = 'uint8'), np.array(end, dtype = 'uint8')


def combine_segments(fname, num, actual_size,  caps = True, folder_index = 0):
    capsize = 100
    for resolution in [64,  128, 256, 512,"fullres"]:

        pname_test = fname+'{}_'.format(0) + str(resolution) + '.npy'
        segment_part_test =  np.array(np.load(pname_test, allow_pickle = True), dtype = 'uint8')
        segments = np.zeros((segment_part_test.shape[0], segment_part_test.shape[1], 0), dtype = 'uint8')
        for part in range(0, num, 100):
            pname = fname+'{}_'.format(part) + str(resolution) + '.npy'
            segment_part =  np.array(np.load(pname, allow_pickle = True), dtype = 'uint8')
            segments = np.dstack((segments, segment_part))
        if caps:
            start,end = make_caps(segments, capsize = capsize)
            full_segments = np.dstack((start, segments, end))
        if caps:
            save_name = fname.split('partial')[0] +"padded"
            if resolution  == "fullres":
                assert full_segments.shape == (actual_size[0], actual_size[1], actual_size[2] + capsize*2)
        else:
            save_name = fname.split('partial')[0] 
            full_segments = segments
            if resolution == 'fullres':
                assert full_segments.shape == actual_size
        np.save(save_name+'{}.npy'.format(resolution),np.array(full_segments, dtype = 'uint8'))
        if resolution == 64:
            os.makedirs(save_name + '/64datasamples', exist_ok=True)
            np.save(save_name+'/64datasamples/{}.npy'.format(folder_index),np.array(full_segments, dtype = 'uint8'))

    
def clean_segments(fname):
    command =  'rm ' + fname + '*npy' 
    print(command)
    os.system(command)

def save_sample(folder_index, generated_dir, imstack_all = None, voxelsize = None, generated = True, boundary_stack_all = None):
    frame_window = 100
    shift = 100
    num = None
    num = imstack_all.shape[2]
    shell_fragment =  'segment_{}_plane_removed_partial_'
    fragment =  'segment_{}_partial__'
    index = 0
    folder_name = generated_dir + '/Part{}/'.format(folder_index)

    os.makedirs(folder_name, exist_ok=True)
    while index < num:
        pores_total =[]
        print("Processing pores, index = " + str(index) + " out of " + str(num) + " ..." )
        im = np.copy(imstack_all[:,:, index:index + frame_window])
        boundary_stack = np.copy(boundary_stack_all[:,:, index:index+frame_window])
        fractured = np.array(fill_boundary(boundary_stack, im), 'uint8')
        pores_removed = np.array(fill_boundary(boundary_stack, np.zeros(im.shape)), 'uint8')

        square_shape, voxel_conversions  = save_binary_segment(fractured,folder_name + fragment.format(folder_index), voxelsize = voxelsize, index = index )
        square_shape, _  = save_binary_segment(pores_removed,folder_name + shell_fragment.format(folder_index), voxelsize = voxelsize, index = index)
        index = index + shift

    combine_segments(os.path.join(folder_name, fragment.format(folder_index)), num = num, actual_size = (square_shape[0], square_shape[1], imstack_all.shape[2]),  caps = True, folder_index = folder_index)
    combine_segments(os.path.join(folder_name, fragment.format(folder_index)), num = num,actual_size = (square_shape[0], square_shape[1], imstack_all.shape[2]),  caps = False, folder_index = folder_index)

    clean_segments(os.path.join(folder_name, fragment.format(folder_index)))

    combine_segments(os.path.join(folder_name, shell_fragment.format(folder_index)), num = num,actual_size = (square_shape[0], square_shape[1], imstack_all.shape[2]),  caps = True, folder_index = folder_index)

    combine_segments(os.path.join(folder_name, shell_fragment.format(folder_index)), num = num, actual_size = (square_shape[0], square_shape[1],imstack_all.shape[2]), caps = False, folder_index = folder_index)
    clean_segments(os.path.join(folder_name, shell_fragment.format(folder_index)))
    return voxel_conversions
    
def test_sample(sample, generated_dir):
    pdir  = generated_dir + '/Part{}/'.format(sample)
    npyfiles = [os.path.join(pdir,file).split('.npy')[0] for file in os.listdir(pdir) if file.endswith('npy')]
    for npyfile in npyfiles:
        sample_dir = './reconstruction/full/testing_binarization/part{}/'.format(sample) + npyfile.split('/')[-1] 
        os.makedirs(sample_dir, exist_ok = True)
        npy = np.load(npyfile+'.npy', allow_pickle = "True")
        plt.clf()
        plt.title(npyfile.split('/')[-1] + "beginning")
        plt.imshow(npy[:,:,0])
        plt.savefig(sample_dir +'/' + npyfile.split('/')[-1] + 'begin.png')
        plt.clf()
        plt.title(npyfile.split('/')[-1] + "middle")
        plt.imshow(npy[:,:,npy.shape[2]//2])
        plt.savefig(sample_dir +'/' + npyfile.split('/')[-1] + 'middle.png')
        plt.clf()
        print(npyfile.split('/')[-1])
        print("done case " + str(sample))
    os.makedirs('./reconstruction/full/testing_binarization/part{}/tiff_stack'.format(sample), exist_ok= True)
    npy = np.load(generated_dir + '/Part{}/segment_{}_fullres.npy'.format(sample, sample), allow_pickle = True)
    for i in range(npy.shape[2]):
        plt.imsave('./reconstruction/full/testing_binarization/part{}/tiff_stack'.format(sample) + "/frame{:05}.png".format(i), npy[:,:, i], cmap = 'gist_gray')
        plt.clf()
    os.system('./Fiji.app/ImageJ-linux64 -macro ./reconstruction/full/testing_binarization/Video.ijm '+ str(sample))
    os.system('rm ' +'./reconstruction/full/testing_binarization/part{}/tiff_stack'.format(sample) + "/frame*png")

def main():
    tot_frames= 2000
    num_frames = 2000
    interval = 1
    generated_dir  = './reconstruction/results/GeneratedPartSamples'
   
    groundtruth_stack_shape = (566, 571) # example shape from dataset
    
    n_bins = 30
    import time
    oldtime = time.time()
    start = 0#int(sys.argv[1])
    for iter in range(start, start+ 100):
        

       for total_frame in range(0, tot_frames, num_frames):
            pore_part = np.zeros((groundtruth_stack_shape[0],groundtruth_stack_shape[1], num_frames), dtype  = 'uint8')

            generated_boundary, profile_2d = load_boundary(num_frames = num_frames, start = total_frame, pore_part_shape = (pore_part.shape[0], pore_part.shape[1]), return_profile=True)

            
            before_polar_time = time.time()
            pore_part,_ = replace_sampling(pore_part, generated_boundary = generated_boundary,n_bins = 30)
            after_polar_time = time.time()
            print(after_polar_time - before_polar_time, "Time taken polar")



            before_cartesian_time = time.time()
            pore_part,_ = replace_sampling(pore_part, generated_boundary = generated_boundary,n_bins = 30)
            after_cartesian_time = time.time()
            print(after_cartesian_time - before_cartesian_time, "Time taken cartesian")
            trim_boundary(generated_boundary, pore_part)
        
            plt.close('all')
            voxel_to_micron = {}
            voxel_to_micron['Resolution'] = ["Name", 64, 128, 256, 512, "Full"]
            num_samples = 12
                
            conversion = save_sample(iter, voxelsize = 4.87, boundary_stack_all=generated_boundary, imstack_all = pore_part, generated_dir = generated_dir)
            
            os.system("7z a "+ generated_dir +  "/Part{}.zip ".format(iter) +  generated_dir + "/Part{}/".format(iter, iter))

            if os.path.exists(generated_dir + '/Part{}.zip'.format(iter)):
                os.system("rm " + generated_dir + "/Part{}/*npy".format(iter))
            print(time.time() - oldtime, "TIME ELAPSED PER ITERATION")
            oldtime = time.time()


if __name__ == "__main__":
    main()
