# -*- coding: utf-8 -*-
import os
import cPickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from glob import glob
import string
from matplotlib import pyplot as plt
import sys
from random import seed, sample
from pprint import pprint
from datetime import datetime
from skimage import color
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from nolearn.decaf import ConvNetFeatures
from sklearn.metrics import accuracy_score
from cifarKaggle import CifarKaggle
from HOGFeatures import HOGFeatures
from sklearn.cross_validation import KFold

class OcrData():
    """
    Lớp dùng cho việc tạo và xử lý các đối tượng văn bản
    """
    def __init__(self, config):
        """
        Nạp dữ liệu cho lớp từ cấu trúc json của file config.py
        Ngoài ra, nó sẽ tự động nạp hình cũng như chia dữ liệu thành tập train và test
        """
        self.config = self._load_config(config)
        self.folder_labels = self.config['folder_labels']
        self.folder_data = self.config['folder_data']
        self.verbose = self.config['verbose']
        self.img_size = self.config['img_size']
        self.limit = self.config['limit']
        self.pickle_data = self.config['pickle_data']
        self.from_pickle = self.config['from_pickle']
        self.automatic_split = self.config['automatic_split']
        self.plot_evaluation = self.config['plot_evaluation']
        self.split = self.config['percentage_of_test_set']
        self.cross_val_models = self.set_models()
        self.load()
        if self.automatic_split:
            self.split_train_test()


    def _load_config(self, filename):
        """
        Đọc file config.py và trả về python dictionary từ cấu trúc json
        """
        return eval(open(filename).read())

    def getRelativePath(self):
        """
        Lấy tất cả đường dẫn của ảnh từ tập Chars74K 
        """
        mFiles = [mFile for mFile in glob(os.path.join(self.folder_labels, '*.m'))]

        self.images = []

        for mFile in mFiles:
            m = open(mFile, "r")
            lines = m.readlines()
            # Xử lý tên file đầu và cuối
            for index, line in enumerate(lines):
                if line.startswith('list.ALLnames'):
                    start_index = index
                    start_image = line[18:].strip()[:-1]
                    if 'Img' in mFile:
                        self.images.append(os.path.join(*(['English','Img'] + start_image.split('/'))))
                    elif 'Hnd' in mFile:
                        self.images.append(os.path.join(*(['English','Hnd'] + start_image.split('/'))))
                    elif 'Fnt' in mFile:
                        self.images.append(os.path.join(*(['English','Fnt'] + start_image.split('/'))))
                elif line.startswith('list.classlabels'):
                    end_index = index - 1 
            if 'Img' in mFile:
                self.images += [os.path.join(*(['English','Img'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            elif 'Fnt' in mFile:
                self.images += [os.path.join(*(['English','Fnt'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            elif 'Hnd' in mFile:
                self.images += [os.path.join(*(['English','Hnd'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            m.close()

        if self.verbose:
            print 'Found {} images.'.format(len(self.images))

        return self.images

    def getLabels(self):
        """
        Trả về danh sách các nhãn của tất cả các ảnh
        Các nhãn sẽ lần lượt tương ứng với thứ tự các ảnh
        Có 62 lớp tương ứng với các kí tự tiếng anh:
        - [0-9] --> 10 lớp
        - [A-Z] --> 26 lớp
        - [a-z] --> 26 lớp
        Để đơn giản, ta coi các kí tự hoa thường là 1 thì ta có tổng cộng 36 lớp
        """
        mFiles = [ mFile for mFile in glob(os.path.join(self.folder_labels, '*.m'))]

        self.labels = []

        for mFile in mFiles:
            m = open(mFile, 'r')
            lines = m.readlines()
            #Xử lý nhãn đầu và nhãn cuối
            for index, line in enumerate(lines):
                if line.startswith('list.ALLlabels'):
                    start_index = index
                    start_label = line[18:].strip()[:-1]
                    self.labels.append(start_label)
                elif line.startswith('list.ALLnames'):
                    end_index = index - 1
            self.labels += [line.strip()[:-1] for line in lines[start_index + 1: end_index]]
            m.close()

        keys = range(1,63) #62 class nhưng sẽ gộp chữ hoa và chữ thường thành 1
        values = map(str,range(10)) + list(string.ascii_lowercase) + list(string.ascii_lowercase)

        classes = dict(zip(keys, values))
        self.labels = map(lambda x: classes[int(x)], self.labels)

        if self.verbose:
            print 'Found {} labels.'.format(len(self.labels))
        return self.labels

    def set_models(self):
        """
        Chuẩn bị các tham số cho việc kiểm chứng chéo (cross validation)
        Hàm trả về 1 dicionary là input cho GridSearchCV (kiểm chứng chéo sử dụng grid search)
        """

        models = {
            'linearsvc': (
                LinearSVC(),
                {'C':  list(np.arange(0.01,1.5,0.01))}, 
                ),
            'linearsvc-hog': (
                Pipeline([
                    ('hog', HOGFeatures(
                        orientations=2,
                        pixels_per_cell=(2, 2),
                        cells_per_block=(2, 2),
                        size = self.img_size
                        )), ('clf', LinearSVC(C=1.0))]),

                  {
                    'hog__orientations': [2, 4, 5, 10],
                    'hog__pixels_per_cell': [(2,2), (4,4), (5,5)],
                    'hog__cells_per_block': [(5,5), ],
                    'clf__C': [2, 5, 10 ],
                    },
                ),
           
            }

        return models

    def load(self):
        """
        Nếu from_pickle == False thì phương thức sẽ lấy đường dẫn ảnh và nhãn,
        zip chúng lại, nạp ảnh lên dạng greyscale, thay đổi kích thước theo img_size, flatten chúng, 
        trộn data lại một cách ngẫu nhiên và trả về dictionary:
        - images --> mảng các ảnh, mỗi ảnh có kích thước (M x N)
        - data --> ma trận ảnh đã flatten (n_images x (M x N)
        - target --> nhãn của mỗi ảnh
        """

        if self.from_pickle:
            try:
                with open(os.path.join(self.folder_data, self.pickle_data), 'rb') as fin:
                    self.ocr = cPickle.load(fin)
                    if self.limit == 0:
                        pass
                    else:
                        self.ocr = {
                            'images': self.ocr['images'][:self.limit],
                            'data': self.ocr['data'][:self.limit],
                            'target': self.ocr['target'][:self.limit]
                            }
                    if self.verbose:
                        print 'Loaded {} images each {} pixels'.format(self.ocr['images'].shape[0], self.img_size)
                    return self.ocr
            except IOError:
                print 'You have to provide a (*.pickle) file to load data from!'
                sys.exit(0)
        else:
            image_paths = self.getRelativePath()
            image_labels = self.getLabels()

            if self.limit == 0:
                complete = zip(image_paths, image_labels)
            else:
                complete = zip(image_paths[:self.limit], image_labels[:self.limit])
            n_images = len(complete)
            im = np.zeros((n_images,) + self.img_size)
            labels = []
            i = 0

            for couple in complete:
                image = imread(os.path.join(self.folder_data, couple[0] + '.png'), as_grey = True)
                sh = image.shape 
                if ((sh[0]*sh[1]) >= (self.img_size[0]*self.img_size[1])):
                    im[i] = resize(image, self.img_size)
                    i += 1
                    labels.append(couple[1])

            im = im[:len(labels)]

            seed(10)
            k = sample(range(len(im)), len(im))
            im_shuf = im[k]
            labels_shuf = np.array(labels)[k]

            if self.verbose:
                print 'Loaded {} images each {} pixels.'.format(len(labels), self.img_size)

            self.ocr = {
                'images': im_shuf,
                'data': im_shuf.reshape((im_shuf.shape[0], -1)),
                'target': labels_shuf
                }

            now = str(datetime.now()).replace(':','-')
            fname_out = 'images-{}-{}-{}.pickle'.format(len(labels), self.img_size, now)
            full_name = os.path.join(self.folder_data, fname_out)
            with open(full_name, 'wb') as fout:
                cPickle.dump(self.ocr, fout, -1)

            return self.ocr

    def split_train_test(self):
        """
        Chia tập dữ liệu thành tập train và set dựa vào trường split trong file config
        - train set chiếm (1-self.split) % trong tập dữ liệu
        - test set chiếm self.split % trong tập dữ liệu
        """

        seed(10)
        total = len(self.ocr['target'])
        population = range(total)
        if self.split == 0:
            self.images_train = self.ocr['images']
            self.data_train = self.ocr['data']
            self.labels_train = self.ocr['target']

            return self.images_train, self.data_train, self.labels_train
        else:
            k = int(np.floor(total * self.split))
            test = sample(population, k)
            train = [i for i in population if i not in test]

            self.images_train = self.ocr['images'][train]
            self.data_train = self.ocr['data'][train]
            self.labels_train = self.ocr['target'][train]

            self.images_test = self.ocr['images'][test]
            self.data_test = self.ocr['data'][test]
            self.labels_test = self.ocr['target'][test]

            return self.images_train, self.data_train, self.labels_train, self.images_test, self.data_test, self.labels_test

    def plot_some(self):
        """
        Biểu diễn 100 hình ảnh cùng nhãn một cách ngẫu nhiên từ tập dữ liệu
        """
        n_images = self.ocr['images'].shape[0]

        fig = plt.figure(figsize = (12, 12))
        fig.subplots_adjust(
            left=0, right=0, bottom=0, StopIteration=1, hspace=0.05, wspace=0.05)

        for i, j in enumerate(np.random.choice(n_images, 100)):
            ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(self.ocr['images'][j], cmap='Greys_r')
            ax.text(2, 7, str(self.ocr['target'][j]), fontsize=25, color='red')
            
        plt.show()
        
    def perform_grid_search_cv(self, model_name):
        """
        Từ tập dữ liệu đã dán nhãn (X_train, y_train) và model_name tạo 
        ra từ hàm set_models, trả về mô hình tốt nhất của tất cả các tham số
        """
        if not self.automatic_split:
            print 'Before performing any Machine Learning algorithm, you should split your data!'
            print 'Change the \'automatic_split\' to True in the config file'
            sys.exit(0)

        model, param_grid = self.cross_val_models[model_name]

        print 'Model: ', model_name
        print 'Parameters: ', param_grid
        print 'Train set shape: ', self.data_train.shape
        print 'Target shape: ', self.labels_train.shape

        gs = GridSearchCV(model, param_grid, n_jobs=1, cv=3, verbose=4)
        gs.fit(self.data_train, self.labels_train)

        pprint(sorted(gs.grid_scores_, key = lambda x: -x.mean_validation_score))
        now = str(datetime.now()).replace(':','-')
        fname_out = '{}-{}.pickle'.format(model_name, now)
        full_name = os.path.join(self.folder_data, fname_out)

        with open(full_name, 'wb') as fout:
            cPickle.dump(gs, fout, -1)

        print 'Saved model to {}'.format(full_name)

    def generate_best_hog_model(self):
        """
        Với các tham số của mô hình tốt nhất mà grid search trả về, bằng
        cách sử dụng Pipeline(hog + linearsvc) ta train lại lần nữa trên toàn bộ tập train
        """
        clf = Pipeline([('hog',HOGFeatures(orientations=5, pixels_per_cell=(2,2), cells_per_block=(4,4), size=self.img_size)),('clf',LinearSVC(C=2.0))])

        clf.fit(self.data_train, self.labels_train)
        y_pred = clf.predict(self.data_train)

        print 'Accuracy on train set: ', accuracy_score(self.labels_train, y_pred)

        now = str(datetime.now()).replace(':','-')
        fname_out = 'linearsvc-hog-fulltrain-{}.pickle'.format(now)
        full_name = os.path.join(self.folder_data, fname_out)

        with open(full_name, 'wb') as fout:
            cPickle.dump(clf, fout, -1)

        print 'Saved model to {}'.format(full_name)

    def evaluate(self, model_filename):
        """
        Đánh giá mô hình tốt nhất (dùng cross validation) trên tập test
        """
        if not self.automatic_split:
            print 'Before performing any Machine Learning algorithm, you should split your data!'
            print 'Change the \'automatic_split\' to True in the config file'
            sys.exit(0)

        if self.split==0:
            print 'The percentage_of_test_set in the config.py is set to 0'
            print 'So, you don\'t have a test set to evaluate your model'
            sys.exit(0)

        with open(model_filename, 'rb') as fin:
            model = cPickle.load(fin)

        y_pred = model.predict(self.data_test)
        print 'Test set shape: ', self.data_test.shape
        print 'Target shape: ', self.labels_test.shape
        print 'Accuracy on train set: ', accuracy_score(self.labels_train, model.predict(self.data_train))
        print 'Accuracy on test set:', accuracy_score(self.labels_test, y_pred)

        if self.plot_evaluation:
            target_names = sorted(np.unique(self.labels_test))
            n_images = self.data_test.shape[0]
            fig = plt.figure(figsize = (6,6))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

            for i, j in enumerate(np.random.choice(n_images, 64)):
                ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
                ax.imshow(self.images_test[j], cmap='Greys_r')
                predicted = model.predict(np.array([self.data_test[j]]))[0]
                if predicted == self.labels_test[j]:
                    color = 'black'
                else:
                    color = 'red'
                ax.text(2, 7, predicted, fontsize=25, color = color)
                plt.show()

                cm = confusion_matrix(self.labels_test, y_pred)
                plt.matshow(cm)
                plt.colorbar()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.xticks(range(len(target_names)), target_names, rotation='vertical')
                plt.yticks(range(len(target_names)), target_names)

    def merge_with_cifar(self):
        """
        Trộn tập Chars74K với tập CIFAR và đánh nhãn lại cho việc phân lớp nhị phân
        Kết quả ta được 1 tập gồm 100000 ảnh (50% có text, 50% không có text)
        """
        cifar = CifarKaggle('cifar-config.py')

        text = OcrData('text-config.py')

        text.ocr['target'][:] = 1

        total = 100000
        seed(10)
        k = sample(range(total), total)

        cifar_plus_text = {
            'images': np.concatenate((cifar.cif['images'], text.ocr['images'][:50000]))[k],
            'data': np.concatenate((cifar.cif['data'], text.ocr['data'][:50000]))[k],
            'target': np.concatenate((cifar.cif['target'], text.ocr['target'][:50000]))[k]
            }

        now = str(datetime.now()).replace(':','-')
        fname_out = 'images-{}-{}-{}.pickle'.format(cifar_plus_text['target'].shape[0], self.img_size, now)
        full_name = os.path.join(self.folder_data, fname_out)
        with open(full_name, 'wb') as fout:
            cPickle.dump(cifar_plus_text, fout, -1)

        return cifar_plus_text






