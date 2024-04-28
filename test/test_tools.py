import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from analysis_tool import functions as f
# import image_csv_cutting as icc

class Test(object):
    def test_crater_location(self):
        """ Test the crater_location function"""
        data = f.crater_location([0.05, 0.85], [0.10, 0.81], 30, 20, -20, 10)
        result = np.array([[-3.5, 20.5], [-12., -26.2]])
        assert np.allclose(data, result, rtol=1.0e-6)

    def test_crater_size(self):
        """ Test the crater_location function"""
        data = f.crater_size([0.5, 0.8], [0.2, 0.5], [0.05, 0.85],
                             [0.10, 0.81], 30, 20, -20, 10, 2309)
        result = np.array([27.39195991, 596.01306045])
        assert np.allclose(data, result, rtol=1.0e-6)

    def test_iou(self):
        """ Test the iou function"""
        data = f.iou([0.01, 0.25, 0.06, 0.10], [0.04, 0.18, 0.22, 0.008])
        result = np.array(0)
        assert np.allclose(data, result, rtol=1.0e-6)

    def test_pair_choosing(self):
        """ Test the pair_choosing function"""
        data = f.pair_choosing([[0.01, 0.15, 0.03, 0.004],
                                [0.12, 0.03, 0.04, 0.01]],
                               [[0.02, 0.13, 0.03, 0.003],
                                [0.14, 0.03, 0.02, 0.01]])
        result = np.array([[0.], [0.2]])
        assert np.allclose(data, result, rtol=1.0e-6)

    def test_recall(self):
        """ Test the recall function"""
        data = f.recall([[0.02], [0.54]], [[1.9], [60]],
                        [[2.6], [1000], [0.09]], 2, 0.5)
        result = np.array([0.5, 1, 1])
        assert np.allclose(data, result, rtol=1.0e-6)

    def test_precision(self):
        """ Test the precision function"""
        data = f.precision([[0.04], [0.54]], [[23], [1000]], 2, 0.5)
        result = np.array([0.5, 1])
        assert np.allclose(data, result, rtol=1.0e-6)

    def test_f1(self):
        """ Test the f1 function"""
        data = f.F1(0.5, 0.5)
        result = np.array(0.5)
        assert np.isclose(data, result, rtol=1.0e-6)
    
        
#     def test_crop_to_pixel(self):
#         """ Test the f1 function"""
#         data = icc.crop_to_pixel('./', 416)
#         result = (549, 976)
#         assert np.isclose(data[0], result[0]) and np.isclose(data[1], result[1])


#     def test_lat_long_to_pixels(self):
#         """ Test the f1 function"""
#         data = icc.lat_long_to_pixels(1, 1)
#         result = (303.23350424151477, 303.18732031007437)
#         assert np.isclose(data[0], result[0]) and np.isclose(data[1], result[1])
