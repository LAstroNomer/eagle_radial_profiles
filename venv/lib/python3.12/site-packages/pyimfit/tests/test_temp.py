# Unit tests for fitting.py module of pyimfit
# Execute via
#    $ pytest test_fitting.py

import os
import math
import copy
import pytest
from collections import OrderedDict
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits

from ..fitting import FitError, FitResult, Imfit
from ..descriptions import FunctionSetDescription, ModelDescription, ParameterDescription
from ..pyimfit_lib import make_imfit_function



#baseDir = "/Users/erwin/coding/pyimfit/pyimfit/tests/"
#testDataDir = baseDir + "../data/"
testDataDir = "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"
imageFile2 = testDataDir + "n3073rss_small.fits"
maskFile2 = testDataDir + "n3073rss_small_mask.fits"
configFile2 = testDataDir + "config_n3073.dat"

image_ic3478 = fits.getdata(imageFile)
image_n3073 = fits.getdata(imageFile2)
mask_n3073 = fits.getdata(maskFile2)

# ModelDescription object for fitting Exponential function to image of IC 3478
model_desc = ModelDescription.load(configFile)
# ModelDescription object for fitting Sersic + Exponential function to image of NGC 3073
model_desc2 = ModelDescription.load(configFile2)

# Simple FlatSky (constant-value) image
flatSkyFunc = make_imfit_function("FlatSky")
funcSet = FunctionSetDescription("sky", ParameterDescription('X0', 1.0), ParameterDescription('Y0', 1.0), [flatSkyFunc])
model_desc_flatsky = ModelDescription([funcSet])


class TestImfit(object):

    def setup_method( self ):
        self.modelDesc = model_desc

    def test_Imfit_optionsDict_updates( self ):
        imfit_fitter2 = Imfit(self.modelDesc)
        imfit_fitter2.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)

        optionsDict_correct1 = {'GAIN': 4.725, 'READNOISE': 4.3, 'ORIGINAL_SKY': 130.14}
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct1

        # update with empty dict should not change anything
        imfit_fitter2._updateModelDescription({})
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct1

        # now test actually updating things
        keywords_new = {'gain': 10.0, 'read_noise': 0.5}
        optionsDict_correct2 = {'GAIN': 10.0, 'READNOISE': 0.5, 'ORIGINAL_SKY': 130.14}
        imfit_fitter2._updateModelDescription(keywords_new)
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct2

        # now test adding an entry
        keywords_new = {'n_combined': 5}
        optionsDict_correct3 = {'GAIN': 10.0, 'READNOISE': 0.5, 'ORIGINAL_SKY': 130.14, 'NCOMBINED': 5}
        print("\none: ", self.modelDesc.getModelAsDict())
        imfit_fitter2._updateModelDescription(keywords_new)
        print("two: ", self.modelDesc.getModelAsDict())
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct3

    def test_Imfit_getModelDict(self):
        # Exponential model for fitting IC 3478
        p = {'PA': [18.0, 0.0, 90.0], 'ell': [0.2, 0.0, 1.0], 'I_0': [100.0, 0.0, 500.0],
             'h': [25.0, 0.0, 100.0]}
        fDict = {'name': "Exponential", 'label': '', 'parameters': p}
        fsetDict = {'X0': [129.0, 125.0, 135.0], 'Y0': [129.0, 125.0, 135.0], 'function_list': [fDict]}
        options_dict = OrderedDict()
        options_dict.update( {"GAIN": 4.725, "READNOISE": 4.3, "ORIGINAL_SKY": 130.14} )
        model_dict_correct = {"function_sets": [fsetDict], "options": options_dict}

        print(self.modelDesc.getModelAsDict())
        imfit_fitter = Imfit(copy.copy(self.modelDesc))
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        model_dict = imfit_fitter.getModelAsDict()
        assert model_dict == model_dict_correct
