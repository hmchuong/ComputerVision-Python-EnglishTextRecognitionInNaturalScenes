import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import cPickle
from skimage.morphology import square, closing
from skimage.io import imshow
from skimage import restoration
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.measure import regionprops
from skimage import measure
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from skimage.transform import resize

class ImageData():
    # Constructor
    def __init__ (self,image_file):
        self.image = imread(image_file,as_grey = True)
        self.preprocess_image()
    ####################################################
    # Denoise and increases contrast
    def preprocess_image(self):
        # Total-variation denoising
        image = restoration.denoise_tv_chambolle(self.image,weight=0.1)
        # Return threshold value based on Otsu's method
        thresh = threshold_otsu(image)
        # Increases contrast
        self.bw = closing(image <= thresh,square(1))
        self.cleared = self.bw.copy()
        return self.cleared
    # Plot preprocessed image
    def plot_preprocessed_image(self):
        # Preprocessed image
        image = restoration.denoise_tv_chambolle(self.image,weight = 0.1)
        thresh = threshold_otsu(image)
        bw = closing(image <= thresh,square(1))
        cleared = bw.copy()

        label_image = measure.label(cleared)
        borders = np.logical_xor(bw,cleared)
        label_image[borders] = -1
        image_label_overlay  = label2rgb(label_image,image=image)
        fig, ax = plt.subplots(ncols = 1,nrows = 1, figsize=(12,12))
        ax.imshow(image_label_overlay)

        for region in regionprops(label_image):
            if region.area<10:
               continue
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc,minr),maxc-minc,maxr - minr,fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        plt.show()

    def get_text_candidates(self):
        """
        identifies objects in the image. Gets contours, draws rectangles around them
        and saves the rectangles as individual images.
        """
        label_image = measure.label(self.cleared)
        borders = np.logical_xor(self.bw, self.cleared)
        label_image[borders] = -1


        coordinates = []
        i=0

        for region in regionprops(label_image):
            if region.area > 10:
                minr, minc, maxr, maxc = region.bbox
                margin = 3
                minr, minc, maxr, maxc = minr-margin, minc-margin, maxr+margin, maxc+margin
                roi = self.image[minr:maxr, minc:maxc]
                if roi.shape[0]*roi.shape[1] == 0:
                    continue
                else:
                    if i==0:
                        samples = resize(roi, (20,20))
                        coordinates.append(region.bbox)
                        i+=1
                    elif i==1:
                        roismall = resize(roi, (20,20))
                        samples = np.concatenate((samples[None,:,:], roismall[None,:,:]), axis=0)
                        coordinates.append(region.bbox)
                        i+=1
                    else:
                        roismall = resize(roi, (20,20))
                        samples = np.concatenate((samples[:,:,:], roismall[None,:,:]), axis=0)
                        coordinates.append(region.bbox)

        self.candidates = {
                    'fullscale': samples,
                    'flattened': samples.reshape((samples.shape[0], -1)),
                    'coordinates': np.array(coordinates)
                    }

        print 'Images After Contour Detection'
        print 'Fullscale: ', self.candidates['fullscale'].shape
        print 'Flattened: ', self.candidates['flattened'].shape
        print 'Contour Coordinates: ', self.candidates['coordinates'].shape
        print '============================================================'

        return self.candidates
    def plot_to_check(self, what_to_plot, title):
        """
        plots images at several steps of the whole pipeline, just to check output.
        what_to_plot is the name of the dictionary to be plotted
        """
        n_images = what_to_plot['fullscale'].shape[0]

        fig = plt.figure(figsize=(12, 12))

        if n_images <=100:
            if n_images < 100:
                total = range(n_images)
            elif n_images == 100:
                total = range(100)

            for i in total:
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][i], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][i]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show()
        else:
            total = list(np.random.choice(n_images, 100))
            for i, j in enumerate(total):
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][j], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][j]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show()
    def show(self):
        imshow(self.image)
        plt.show()
    ####################################################