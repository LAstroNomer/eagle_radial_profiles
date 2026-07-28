# Background without fitting
from astropy.convolution import convolve
from photutils.background import BkgIDWInterpolator
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources, detect_threshold
import numpy as np
from photutils.background import Background2D, MedianBackground
from astropy.convolution import Tophat2DKernel
import warnings

def dilate_mask(mask, tophat_size):
    """ Take a mask and make the masked regions bigger."""
    area = np.pi * tophat_size ** 2.
    kernel = Tophat2DKernel(tophat_size)
    dilated_mask = convolve(mask, kernel) >= 1. / area
    return dilated_mask


class SourceMask:
    def __init__(self, img, nsigma=3., npixels=3):
        """ Helper for making & dilating a source mask.
             See Photutils docs for make_source_mask."""
        self.img = img
        self.nsigma = nsigma
        self.npixels = npixels

    def single(self, filter_fwhm=3., tophat_size=5., mask=None):
        """Mask on a single scale"""
        if mask is None:
            image = self.img
        else:
            image = self.img * (1 - mask)
        mask = make_source_mask(image, nsigma=self.nsigma,
                                npixels=self.npixels,
                                filter_fwhm=filter_fwhm)
        return dilate_mask(mask, tophat_size)

    def multiple(self, filter_fwhm=[3.], tophat_size=[3.], mask=None):
        """Mask repeatedly on different scales"""
        if mask is None:
            self.mask = np.zeros(self.img.shape, dtype=bool)
        for fwhm, tophat in zip(filter_fwhm, tophat_size):
            smask = self.single(filter_fwhm=fwhm, tophat_size=tophat)
            self.mask = self.mask | smask  # Or the masks at each iteration
        return self.mask


def make_source_mask(image, nsigma, npixels, filter_fwhm):
    threshold = detect_threshold(image, nsigma)
    kernel = make_2dgaussian_kernel(filter_fwhm, size=5)
    convolved_data = convolve(image, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=npixels)
    try:
        mask = segment_map.make_source_mask()
    except:
        return np.zeros(image.shape)
    return mask


def main_bkg(data):
    bkg_estimator = MedianBackground()
    mask = np.zeros(data.shape)
    mask = mask > 0
    # interpolator = BkgIDWInterpolator(n_neighbors=20, power=0, reg=30)
    with warnings.catch_warnings(action="ignore"):
        for i in range(5):
            print(i)
            bkg = Background2D(data, (6, 6), filter_size=(3, 3),
                               bkg_estimator=bkg_estimator, mask=mask)
            data_ = data - bkg.background  # subtract the background

            sm = SourceMask(data_, nsigma=2)
            mask = sm.multiple(filter_fwhm=[1, 3, 5], tophat_size=[4, 2, 1])

    # bkg = Background2D(data, (6, 6), filter_size=(3, 3),
    #                       bkg_estimator=bkg_estimator, mask=mask, interpolator=interpolator)
    return bkg, mask