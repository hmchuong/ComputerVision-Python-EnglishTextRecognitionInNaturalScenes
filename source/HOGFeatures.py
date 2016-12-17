# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator
from skimage.feature import hog

class HOGFeatures(BaseEstimator):
    """
    L?p k? th?a BaseEstimator dùng cho Scikit-learn Pipeline
    L?p này cài ??t HOG, k? thu?t th??ng ???c dùng ?? trích xu?t 
    ??c tr?ng t? ?nh và ??a vào trong b? phân l?p
    """
    def __init__(self, 
                 size,
                 orientations =8,
                 pixels_per_cell=(10,10),
                 cells_per_block=(1,1)):
        super(HOGFeatures, self).__init__()        
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reshape((X.shape[0], self.size[0], self.size[1]))
        result = []            
        for image in X:
            features = hog(
                image,
                orientations = self.orientations,
                pixels_per_cell = self.pixels_per_cell,
                cells_per_block = self.cells_per_block,
                )
            result.append(features)

        if np.array(result).shape[1] == 0:
            self.orientations = 8 #10
            self.pixels_per_cell = (10,10) #(5,5)
            self.cells_per_block = (1,1) #(2,2) 
            result = []
            for image in X:
                features = hog(
                    image,
                    orientations = self.orientations,
                    pixels_per_cell = self.pixels_per_cell,
                    cells_per_block = self.cells_per_block,
                    )
                result.append(features)
        return np.array(result)
   