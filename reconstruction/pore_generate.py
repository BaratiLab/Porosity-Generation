import torch
import torch.nn as nn
import tifffile
import torch.nn.parallel
import torch.utils.data
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
#import torch.distributions as td
from reconstruction.dcgan_test import Generator
import torch.backends.cudnn as cudnn
from skimage import measure
import h5py
from pylab import gca
from analyze_pore_samples.plotting_utils import improve_pairplot

def read_file(fname):
    # fname = os.path.join(dir, filename)
    f = h5py.File(fname, 'r') 
    return f['data']
def plt_scatter_matrices(prior_realizations, post_realizations, variable_list, plt_prior=True, plt_post=True, show = False):
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
    # breakpoint()
    if plt_prior and plt_post:
        for key in prior_data:
            prior_data[key] = np.append(prior_data[key],post_data[key])
        data = DataFrame(prior_data)
    elif plt_post:
        data = DataFrame(post_data)
    else:
        data = DataFrame(prior_data)

    colors = ['blue','red']
    # breakpoint()
    plt.figure(figsize = [8,6], dpi = 150)
    pplot = sns.pairplot(data, hue = 'Case', kind = 'kde', dropna = True, diag_kws=dict(common_norm= 'False'))
    # breakpoint()
    replacements = {'Volume': r'$log_{10} Volume$', 'Anisotropy': 'Anisotropy',  'Orientation': r'Orientation, $\theta$ [rad]'}
    improve_pairplot(pplot, replacements = replacements)
    plt.savefig('./reconstruction/gan/figures/pairplot_gan.png')
    if show:
        plt.show()

def save_pore(pore_identity, epoch = None, real = False, fname = '', pore_matrix = None, value  =0 ):
    if real:
        filename = './analyze_pore_samples/results/individual_pore_samples/partsample0/pore_original_{}.hdf5'.format(pore_identity)
    else:
        filename   =   './reconstruction/gan/saved_generated_pores/generator_' + str(pore_identity) + '.hdf5'

        # breakpoint()

    data = np.array(read_file(filename))
    labeled = measure.label(data)
    pores = measure.regionprops(labeled)
    pore_idx = np.argmax([pore.area for pore in pores])
    data[labeled != pores[pore_idx].label] = 0
    zmin  = np.min(np.where(data)[2]) - 1
    zmax = np.max(np.where(data)[2])+ 1
    xmin  = np.min(np.where(data)[0]) - 1
    xmax = np.max(np.where(data)[0])+ 1
    ymin  = np.min(np.where(data)[1]) - 1
    ymax = np.max(np.where(data)[1])+ 1
    bounds_min =  np.min([xmin, ymin, zmin])
    bounds_max = np.max([zmax, ymax, xmax])

    if not real:
        directory_saved = './reconstruction/saved_generated_pores/pore_{}'.format(pore_identity) + fname
    else:
        directory_saved = './reconstruction/saved_example_pores/pore_{}'.format(pore_identity) + fname

    os.makedirs(directory_saved, exist_ok=True)
    plt.figure().add_subplot(projection='3d').voxels(data[bounds_min:bounds_max, bounds_min:bounds_max, bounds_min:bounds_max], edgecolor = 'k')
    plt.title( '{0:.2f}'.format(value) + ' ' + str(pore_matrix[pore_identity]))
    plt.tight_layout()
    plt.savefig(directory_saved+ '/pore.png')
    plt.clf()

def query_pore(volume, anis, angle, pore_matrix):
    return np.argmin(np.abs(volume - pore_matrix[:,0])/np.mean(pore_matrix[:,0]) + np.abs(anis -  pore_matrix[:,2])/np.mean(pore_matrix[:,2]) + np.abs(angle - pore_matrix[:,3])/np.mean(pore_matrix[:,3]))

    
params = {
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 2,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector
    'ngf' : 64,# Size of feature maps in the generator. The filtes will be multiples of this.
    'ndf' : 16, # Size of features maps in the discriminator. The filters will be multiples of this.
    'ngpu': 1, # Number of GPUs to be used
    'nepochs' : 15,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'alpha' : 1,# Size of z space
    'stride' : 16,# Stride on image to crop
    'num_samples' : 179}# Save step.

def legend(location = 'best', fontsize = 8):
    plt.legend(loc = location, fontsize = fontsize, frameon = False)
def unit_vector(vector):
    # """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
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
def extract_pore(im):
    size = 32
    im = np.array(im.cpu().detach().numpy(), dtype = 'int')
    props = measure.regionprops(measure.label(im))
    p_idx = np.argmax([pore.area for pore in props])
    pore_3d = np.zeros((size*2, size*2, size*2))
    xdist = props[p_idx].slice[0].stop - props[p_idx].slice[0].start
    ydist = props[p_idx].slice[1].stop - props[p_idx].slice[1].start
    zdist = props[p_idx].slice[2].stop - props[p_idx].slice[2].start
    
    pore_3d[size - xdist//2: size+(xdist-xdist//2), size-ydist//2:size+(ydist- ydist//2), size-zdist//2:size+(zdist-zdist//2)]  = np.array(props[p_idx].image, dtype = 'float')
    # breakpoint()
    return pore_3d
def analyze_pore(im):
    voxelsize = 3.49

    realvols = []
    realorientations = []
    realanisotropies = []
    realmin_axis_l = []
    realmaj_axis_l = []
    pores = measure.regionprops(measure.label(im))
    if len(pores) > 1:
        breakpoint()
    # breakpoint()
    if len(pores) == 0:
        # breakpoint()
        return [-1,0,0,0,0]
    # try1
    pore_idx = np.argmax([pore.area for pore in pores])
    realvols.append(pores[pore_idx].area)
    if pores[pore_idx].area< 3:
        return [-1,0,0,0,0]
    realmaj_axis_l.append((pores[pore_idx].major_axis_length)*voxelsize)
    realmin_axis_l.append((pores[pore_idx].minor_axis_length)*voxelsize)


    pore = pores[pore_idx]

    inertia_eigval = pore.inertia_tensor_eigvals
    inertia = pore.inertia_tensor
    maxeig = np.argmax(inertia_eigval)
    eigvec = np.linalg.eig(pore.inertia_tensor)[1]
    eigvals = np.linalg.eig(pore.inertia_tensor)[0]
    if np.sum(eigvals) == 0:
        breakpoint()
    try:
        anis = 1 - np.min(eigvals)/np.max(eigvals)
    except:
        anis = 0

    if np.max(eigvals) == 0:
        anis = 0
    realanisotropies.append(anis)
    maxvector = eigvec[:, maxeig]
    orientation = angle_between(maxvector, np.array([0,0,1]))
    phi = angle_between(maxvector, np.array([0,1,0]))
    realorientations.append(orientation)

    return [realvols[0], 1, anis, orientation, phi]
def save_hdf5(ndarr, filename):

    # tensor = tensor.cpu()
    # ndarr = tensor.mul(255).byte().numpy()
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=ndarr, dtype="i8", compression="gzip")    
def test_generator(epoch  = 38, folder_index = 0, num_samples = 100, show = False):
    pore_list_attr = [] 
    print("Epoch, ", epoch)
    cudnn.benchmark = True

    # Use GPU is available else use CPU.
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")
    # out_dir ='./reconstruction/generated_pore_samples' + str(epoch) +'/'
    out_dir = './reconstruction/gan/saved_generated_pores/'
    os.makedirs(str(out_dir), exist_ok=True)

    checkpoint = torch.load('./reconstruction/gan/netG_epoch_62.pth')##torch.load('./reconstruction/gan/saved_model.pth')
    def frame_tick(frame_width = 2, tick_width = 1.5):
        ax = gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(frame_width)
        plt.tick_params(direction = 'in', 
                        width = tick_width)


    if('cuda' in str(device)):
        # Create the generator.
        netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu'], size =64).to(device)
        netG.load_state_dict(checkpoint)
        netG = nn.DataParallel(netG)     
        
    else:
        # Create the generator.
        netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
        netG.load_state_dict(checkpoint)
        netG = nn.DataParallel(netG)
        
        noise = torch.FloatTensor(1, params['nz'], params['alpha'], params['alpha'], params['alpha']).normal_(0, 1)
        noise = noise.to(device)

    i =0
    # Clean generated pores
    print("Removing previously generated pores")
    file_list = glob.glob(os.path.join(out_dir,'generator*hdf5'))
    for fname in file_list:
        os.remove(fname)
    
    while i < num_samples:#params['num_samples']:
        noise = torch.FloatTensor(1, params['nz'], params['alpha'], params['alpha'], params['alpha']).normal_(0, 1)
        noise = noise.to(device)      
        fake_data = netG(noise)
        fake_argmax = fake_data.argmax(dim=1)
        im = 1-fake_argmax[0]
        if torch.sum(im) ==0:
            continue
        pore = extract_pore(1-fake_argmax[0])
        fake_stats = analyze_pore(pore)
        if fake_stats[0] < 0:
            print('An unresolved pore filtered, continuing generation process')
        else:

            pore_list_attr.append(fake_stats)
            if i%100 == 0:
                print('========================')
                print(str(i) + ' pores processed')
                print('========================')
            save_hdf5(pore, str(out_dir)+'/generator_{0}.hdf5'.format(i))
            i+= 1
    print('{} pores generated, saved in '.format(i), out_dir)
    pore_matrix = np.loadtxt('analyze_pore_samples/results/individual_pore_samples/partsample{}/pore_matrix'.format(folder_index))  
    pore_matrix_generated = np.squeeze(np.array(pore_list_attr))

    np.savetxt('generated_pore_matrix.csv',pore_matrix_generated )
    pore_identity = 100
    filename   = str(out_dir)+'generator_{0}.hdf5'.format(pore_identity)
    data = read_file(filename)
    pore_sizes = pore_matrix[:,0]

    vol_target = np.sort(pore_matrix[:,0])[int(9*len(pore_matrix)//10)]
    anis =np.sort(pore_matrix[:,2])[len(pore_matrix)//2]
    angle = np.sort(pore_matrix[:,3])[len(pore_matrix)//2]
    pore_identity_real = query_pore(vol_target, anis, angle, pore_matrix)
    pore_identity_generated = query_pore(vol_target, anis, angle, pore_matrix_generated)

    save_pore(pore_identity_real, epoch = epoch, real  = True, fname = 'baseline', pore_matrix=pore_matrix, value = vol_target)
    save_pore(pore_identity_generated, epoch = epoch, fname = 'baseline', pore_matrix = pore_matrix_generated, value = vol_target)



    vol_target = np.sort(pore_matrix[:,0])[int(9.92*len(pore_matrix)//10)]
    anis =np.sort(pore_matrix[:,2])[len(pore_matrix)//2]
    angle = np.sort(pore_matrix[:,3])[len(pore_matrix)//2]
    pore_identity_real = query_pore(vol_target, anis, angle, pore_matrix)
    pore_identity_generated = query_pore(vol_target, anis, angle, pore_matrix_generated)

    save_pore(pore_identity_real, epoch = epoch, real  = True, fname = 'bigvol', pore_matrix=pore_matrix, value = vol_target)
    save_pore(pore_identity_generated, epoch = epoch, fname = 'bigvol', pore_matrix = pore_matrix_generated, value = vol_target)


    vol_target = np.sort(pore_matrix[:,0])[int(9*len(pore_matrix)//10)]
    anis =np.sort(pore_matrix[:,2])[int(9.9*len(pore_matrix)//10)]
    angle = np.sort(pore_matrix[:,3])[len(pore_matrix)//2]
    pore_identity_real = query_pore(vol_target, anis, angle, pore_matrix)
    pore_identity_generated = query_pore(vol_target, anis, angle, pore_matrix_generated)

    save_pore(pore_identity_real, epoch = epoch, real  = True, fname = 'biganis', pore_matrix=pore_matrix, value = anis)
    save_pore(pore_identity_generated, epoch = epoch, fname = 'biganis', pore_matrix = pore_matrix_generated, value = anis)


    vol_target = np.sort(pore_matrix[:,0])[int(9*len(pore_matrix)//10)]
    anis =np.sort(pore_matrix[:,2])[len(pore_matrix)//2]
    angle = np.sort(pore_matrix[:,3])[int(9.9*len(pore_matrix)//10)]
    pore_identity_real = query_pore(vol_target, anis, angle, pore_matrix)
    pore_identity_generated = query_pore(vol_target, anis, angle, pore_matrix_generated)

    save_pore(pore_identity_real, epoch = epoch, real  = True, fname = 'bigangle', pore_matrix=pore_matrix, value = angle)
    save_pore(pore_identity_generated, epoch = epoch, fname = 'bigangle', pore_matrix = pore_matrix_generated, value = angle)
    os.makedirs('./reconstruction/gan/figures',exist_ok = True)
    np.savetxt('./reconstruction/gan/figures/pore_matrix_updated.csv',pore_matrix_generated )
    plt.figure(figsize = [4,3], dpi = 150 )

    plt.hist(pore_matrix[:,0]*(3.49)**3, density = True, alpha = 0.7,bins=np.logspace(np.log10(10e1),np.log10(10e5)), edgecolor = 'k',  label = "Ground Truth")
    plt.hist(pore_matrix_generated[:,0]*(3.49)**3, density = True, alpha = 0.7,bins=np.logspace(np.log10(10e1),np.log10(10e5)), edgecolor = 'k',  label = "Generated")
    plt.xscale('log')
    plt.yscale('log')

    legend()
    plt.xlabel(r"Volume [$\mu m^3$]")
    plt.ylabel("Probability")

    frame_tick()
    plt.tight_layout()
    plt.savefig('./reconstruction/gan/figures/volume' +str(epoch))
    if show:
        plt.show()
    plt.clf()#savefig("vols_new" + str(k+ 100) + ".png")

    plt.figure(figsize = [4,3], dpi = 150 )
    frame_tick()
    plt.hist((np.array(pore_matrix[:,3])/np.pi)*180, density = True, bins=30, edgecolor = 'k', alpha  =0.7, label = "Ground Truth")  
    plt.hist((np.array(pore_matrix_generated[:,3])/np.pi)*180,density = True, bins=30,alpha = 0.7, edgecolor = 'k', label = "Generated")  

    legend()
    plt.xlabel(r"Angle [Degrees]")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig('./reconstruction/gan/figures/angle' +str(epoch))
    if show:
        plt.show()
    plt.clf()

    plt.figure(figsize = [4,3], dpi = 150 )
    frame_tick()

    plt.hist(pore_matrix[:,2], density = True, bins=30, edgecolor = 'k', label = "Ground Truth", alpha  = 0.7) 
    plt.hist(pore_matrix_generated[:,2], density = True, bins=30, edgecolor = 'k',  label = "Generated", alpha  = 0.7)  

    legend()
    plt.xlabel(r"Anisotropy")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig('./reconstruction/gan/figures/anisotropy' +str(epoch))
    if show:
        plt.show()
    plt.clf()

    # plt.clf()
    dict_one = {'Volume': np.log10(pore_matrix[:,0]), 'Anisotropy':pore_matrix[:,2], 'Orientation':pore_matrix[:,3]}#,'phis': gt_total_phi }
    dict_two = {'Volume': np.log10(pore_matrix_generated[:,0]), 'Anisotropy':pore_matrix_generated[:,2], 'Orientation':pore_matrix_generated[:,3]}
    plt_scatter_matrices(dict_one, dict_two, list(dict_one.keys()), show = show)

    plt.clf()   
if __name__ == "__main__":
    test_generator()