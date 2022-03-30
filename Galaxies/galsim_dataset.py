from ast import Raise
from multiprocessing.sharedctypes import Value
import galsim
import astropy.table
import astropy.io.fits
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch.utils.data import TensorDataset

class Galaxy_Profile:

    def __init__(self, params, clean_image, noisy_image, path_to_image=''):
        self.params = params
        self.clean_image = clean_image
        self.noisy_image = noisy_image
        self.path_to_image = path_to_image


class Single_Component_Profile(Galaxy_Profile):

    def __init__(self, n, hlr, clean_image, noisy_image, path_to_image=''):
        super().__init__(
            {'sersic_index': n, 'half_light_radius': hlr},
            clean_image,
            noisy_image,
            path_to_image=path_to_image
        )

class Two_Component_Profile(Galaxy_Profile):

    def __init__(self, n_disk, n_bulge, h_disk, h_bulge, bdr, clean_image, noisy_image, path_to_image=''):
        super().__init__(
            {
                'sersic_index_disk': n_disk,
                'sersic_index_bulge': n_bulge,
                'half_light_radius_disk': h_disk,
                'half_light_radius_bulge': h_bulge,
                'bulge_disk_ratio': bdr
            },
            clean_image,
            noisy_image,
            path_to_image=path_to_image
        )

class NumpyEncoder(json.JSONEncoder):
    '''
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

    To restore ndarray from json:
    json_load = json.loads(json_dump)
    a_restored = np.asarray(json_load["a"])
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Some parameters to characterize the exposure
# Mimicking LSST

zenith_fwhm = 0.7 # seeing in r-band as per Ivezic et al. 2019
X = 1.2 # median airmass
central_wavelength = 6199.52 # Central wavelength for LSST r-band in Angstroms
mirror_diameter = 8.36 # Mirror diameter in meters
effective_area = 32.4 # Collecting area in square meters
px_scale = 0.2 # LSSTcam pixel scale in arcseconds
#px_scale = 1.0 # since our images are already in the 0.2 arcsec scale
sky_brightness = 21.2 # r-band sky brightness at Cerro Pachón (Ivezic et al. 2019)

# PSF generation
atmospheric_psf_fwhm = zenith_fwhm*X**0.6 # Expected FWHM for the atmospheric PSF given the airmass
atmospheric_psf_model = galsim.Kolmogorov(fwhm = atmospheric_psf_fwhm) # Atmospheric part

# Set up the optical part
lambda_over_diameter = 3600*math.degrees(1e-10*central_wavelength/mirror_diameter) # This is tells you the diffraction limit
area_ratio = effective_area/(math.pi*(0.5*mirror_diameter)**2) 
obscuration_fraction = math.sqrt(1 - area_ratio) # Part of the mirror is obscured due to the presence of the secondary (that blocks light)
optical_psf_model = galsim.Airy(lam_over_diam = lambda_over_diameter, obscuration = obscuration_fraction) # Optics part of the PSF
psf_model = galsim.Convolve(atmospheric_psf_model,optical_psf_model) # total PSF

exposure_time=5520
mag = 2 #magnitude_obs -> should be extracted from the image but here I just put some value...
zp = 2.5*np.log10(43.7*10**(-0.4*(0-24))) # Magnitude zeropoint using the flux zeropoint from https://github.com/LSSTDESC/WeakLensingDeblending

#Currently this outputs clean+PSF and clean+PSF+noise but we can modify to get what ever we want

def draw_profile_stamp(profile, psf_model, exptime, nx=495, ny=495, px_scale=px_scale, show=True, save_fig=False, save_file=''):
    """
    Routine that returns a 2D numpy array with the noisy and
    noiseless postage stamps for a given profile

    Inputs:
    -------
    profile: GSObject describing the pre-convolved profile.
    psf_model: GSObject describing the PSF model.
    exptime: float, exposure time in seconds.
    nx: int, number of pixels in the horizontal axis for the postage stamp
    ny: int, number of pixels in the vertical axis for the postage stamp
    show: bool, if True it will show the noisy and noiseless images (default: True)

    Outputs:
    --------
    pristine: ndarray (nx, ny), noiseless image.
    noisy: ndarray (nx, ny), noisy image
    """
    noise = galsim.PoissonNoise(sky_level=get_flux(sky_brightness, zp, exposure_time))
    conv_prof = galsim.Convolve(profile, psf_model)
    conv_img = conv_prof.drawImage(nx=nx, ny=ny, scale=px_scale)
    pristine = conv_img.array.copy() # we need a copy since it will be modified later
    noise.applyTo(conv_img)
    noisy = conv_img.array
    if show:
        f, ax = plt.subplots(ncols=2, nrows=1, figsize=(9, 4))
        im = ax[0].imshow(pristine)#, origin='lower')
        plt.colorbar(im, ax=ax[0], label='Flux')
        im = ax[1].imshow(noisy)#, origin='lower')
        plt.colorbar(im, ax=ax[1], label='Flux')
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_file)
    return pristine, noisy

def draw_profile_stamp_no_psf(profile, exptime, nx=495, ny=495, px_scale=px_scale, show=True, save_fig=False, save_file=''):
    """
    Routine that returns a 2D numpy array with the noisy and
    noiseless postage stamps for a given profile

    Inputs:
    -------
    profile: GSObject describing the pre-convolved profile.
    exptime: float, exposure time in seconds.
    nx: int, number of pixels in the horizontal axis for the postage stamp
    ny: int, number of pixels in the vertical axis for the postage stamp
    show: bool, if True it will show the noisy and noiseless images (default: True)

    Outputs:
    --------
    pristine: ndarray (nx, ny), noiseless image.
    noisy: ndarray (nx, ny), noisy image
    """
    noise = galsim.PoissonNoise(sky_level=get_flux(sky_brightness, zp, exposure_time))
    pristine_img = profile.drawImage(nx=nx, ny=ny, scale=px_scale)
    pristine = pristine_img.array.copy()
    noise.applyTo(pristine_img)
    noisy = pristine_img.array
    if show:
        f, ax = plt.subplots(ncols=2, nrows=1, figsize=(9, 4))
        f.suptitle('Single Sersic Profile with n = {:.2f}, half light radius = {} arcsec'.format(profile.n, profile.half_light_radius))
        ax[0].set_title('Without Noise')
        ax[1].set_title('With Noise')
        im = ax[0].imshow(pristine)#, origin='lower')
        plt.colorbar(im, ax=ax[0], label='Flux')
        im = ax[1].imshow(noisy)#, origin='lower')
        plt.colorbar(im, ax=ax[1], label='Flux')
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_file)
    return pristine, noisy


def get_flux(mag, zp, exposure_time, gain=1.0):
    #Given a magnitude and a zeropoint with exposure time, convert to flux counts
    #the gain can also be specified in GalSim when drawing the profiles
    return 10**(-0.4*(mag-zp))*exposure_time*gain


def single_component_profile_stamp(sersic_index=2, half_light_radius=20, save_fig=False, save_file=''):
    '''
    Create a Sersic object with the given sersic index and half light radius,
    calls draw_profile_stamp_no_psf and saves the result to save_file (a path)
    '''
    sersic_obj1 = galsim.Sersic(n=sersic_index, half_light_radius=half_light_radius, flux=2)
    pr_galaxy, noisy_galaxy = draw_profile_stamp_no_psf(sersic_obj1, exposure_time, save_fig=save_fig, save_file=save_file)
    return pr_galaxy, noisy_galaxy


def many_single_component_profiles(sersic_indices=[], half_light_radii=[], save_figs=False, save_dir=''):
    '''
    Create one Sersic object with each given sersic index and half light radius,
    calls draw_profile_stamp_no_psf and saves the result to a generated path
    Returns a list of Galaxy Profile objects
    '''

    profiles = []

    if not (len(sersic_indices) == len(half_light_radii)):
        raise ValueError("Given lists do not have the same length")
    
    for i in range(len(sersic_indices)):
        print("Making Single Component Profile {} of {}.".format(i+1, len(sersic_indices)))
        n = sersic_indices[i]
        hlr = half_light_radii[i]
        save_file = save_dir+'/'+'single_component_n_{:.2f}_hlr_{}.png'.format(n, hlr)
        clean, noisy = single_component_profile_stamp(n, hlr, save_fig=save_figs, save_file=save_file)
        profile = Single_Component_Profile(n, hlr, clean, noisy, save_file)
        profiles.append(profile)

    return profiles

def single_component_profiles_as_arrays(sersic_indices=[], half_light_radii=[], save_figs=False, save_dir=''):
    '''
    Create one Sersic object with each given sersic index and half light radius,
    calls draw_profile_stamp_no_psf and saves the result to a generated path
    Returns a tuple (x_clean, x_noisy, y) of data
    '''

    x_clean = []
    x_noisy = []
    y = []

    if not (len(sersic_indices) == len(half_light_radii)):
        raise ValueError("Given lists do not have the same length")
    
    for i in range(len(sersic_indices)):
        print("Making Single Component Profile {} of {}.".format(i+1, len(sersic_indices)))
        n = sersic_indices[i]
        hlr = half_light_radii[i]
        save_file = save_dir+'/'+'single_component_n_{:.2f}_hlr_{}.png'.format(n, hlr)
        clean, noisy = single_component_profile_stamp(n, hlr, save_fig=save_figs, save_file=save_file)
        # these arrays need to be added into the dataset as one element lists
        # because the first dimension is for the channel
        # X: sample # > channel # > row > column, Y: sample # > integer
        x_clean.append([clean])
        x_noisy.append([noisy])
        #y.append((n, hlr))
        y.append(n)

    x_clean = np.array(x_clean, dtype=np.float32)
    x_noisy = np.array(x_noisy, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    #print("Got arrays with shapes: ", x_clean.shape, x_noisy.shape, y.shape)
    #print("type of array[0]: ", type(x_clean[0]), type(x_noisy[0]), type(y[0]))
    #print("type of x[0][0] for x_clean and x_noisy: ", type(x_clean[0][0]), type(x_noisy[0][0]))
    #print("type of a single pixel (x[0][0][0]): ", type(x_clean[0][0][0]))

    return (x_clean, x_noisy, y)

def two_component_profile_stamp(n_disk=2, n_bulge=0.5, disk_hlr=15, bulge_hlr=5, bulge_disk_ratio=0.2, save_fig=False, save_file=''):
    '''
    Creates a composite of two Sersic objects with the given Sersic indices and half light radii
    before calling draw_profile_stamp and saving the result to save_file

    ! this function relies on global variables including flux and exposure time
    '''

    flux = get_flux(mag, zp=zp, exposure_time=exposure_time)
    disk = galsim.Sersic(n=n_disk, half_light_radius=disk_hlr, flux=(1-bulge_disk_ratio)*flux)
    bulge = galsim.Sersic(n=n_bulge, half_light_radius=bulge_hlr, flux=bulge_disk_ratio*flux)
    galaxy = disk+bulge
    pr_galaxy, noisy_galaxy = draw_profile_stamp_no_psf(galaxy, exposure_time, save_fig=save_fig, save_file=save_file)

    return


def many_two_component_profiles(sersic_indices_disk=[], sersic_indices_bulge=[], hl_radii_disk=[], hl_radii_bulge=[], bulge_disk_ratios=[], save_figs=False, save_dir=''):
    '''
    Creates a two-component Sersic object for each set of Sersic indices and half light radii provided
    The bluge_disk_ratio can also be provided, although a default of 0.2 is implemented
    The resulting images are saved to a generated path
    Returns a list of Galaxy Profile objects
    '''

    profiles = []

    L = len(sersic_indices_disk)
    if bulge_disk_ratios == []:
        bulge_disk_ratios = [0.2 for i in range(L)]
    flag_1 = (len(sersic_indices_bulge) == L)
    flag_2 = (len(hl_radii_disk) == L)
    flag_3 = (len(hl_radii_bulge) == L)
    flag_4 = (len(bulge_disk_ratios) == L)

    if not(flag_1 and flag_2 and flag_3 and flag_4):
        raise ValueError("Given lists do not all have the same length")

    for i in range(L):
        print("Making Two Component Profile {} of {}.".format(i+1, L))
        n1 = sersic_indices_disk[i]
        n2 = sersic_indices_bulge[i]
        h1 = hl_radii_disk[i]
        h2 = hl_radii_bulge[i]
        bdr = bulge_disk_ratios[i]
        save_file = save_dir+'/'+'two_component_n_disk_{}_n_bulge_{}_hlr_disk_{}_hlr_bulge_{}_bdr_{}.png'.format(n1, n2, h1, h2, bdr)
        clean, noisy = two_component_profile_stamp(n_disk=n1, n_bulge=n2, disk_hlr=h1, bulge_hlr=h2, bulge_disk_ratio=bdr, save_fig=save_figs, save_file=save_file)
        profile = Two_Component_Profile(n1, n2, h1, h2, bdr, clean, noisy, save_file)
        profiles.append(profile)

    return profiles




if __name__ == '__main__':

    '''
    Misc warnings / notes:
        - the Sersic index takes values 0.5 < n < 5.5
    
    '''


    # Making a dataset of single component galaxies
    # def many_single_component_profiles(sersic_indices=[], half_light_radii=[], save_dir=''):
    save_figs = False
    save_dir = 'images/'
    n_min = 0.5
    n_max = 4.5
    N = 1000
    #sersic_indices = [1]
    #half_light_radii = [15]
    sersic_indices = np.linspace(n_min, n_max, N)
    half_light_radii = [15 for i in range(N)]
    
    '''
    !!!
    Next todo: update this to use new single_component_profiles_as_arrays method
    single_component_profiles_as_arrays(sersic_indices=[], half_light_radii=[], save_figs=False, save_dir='')
    '''

    clean_sample_label = 'clean_single_component_galaxies_{}_values_of_n.pt'.format(N)
    noisy_sample_label = 'noisy_single_component_galaxies_{}_values_of_n.pt'.format(N)
    x_clean, x_noisy, y = single_component_profiles_as_arrays(sersic_indices, half_light_radii, save_figs=False)

    tensor_x_clean = torch.from_numpy(x_clean)
    tensor_x_noisy = torch.from_numpy(x_noisy)
    tensor_y = torch.from_numpy(y)

    #print("shapes as tensors: ", tensor_x_clean.shape, tensor_x_noisy.shape, tensor_y.shape)
    #print("types: ", tensor_x_clean.type(), tensor_x_noisy.type(), tensor_y.type())

    clean_dataset = TensorDataset(tensor_x_clean, tensor_y)
    noisy_dataset = TensorDataset(tensor_x_noisy, tensor_y)

    torch.save(clean_dataset, clean_sample_label)
    torch.save(noisy_dataset, noisy_sample_label)



        


        
