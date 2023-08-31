import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
from utils import *

class Dataset:
    def __init__(
        self,
        data_name,
        logging_dir,
        CNN,
        ReLU_preproc,
        part=0,
        noise=0,
        plot_dir=None,
        model_name="",
        data_paths={'PLACES365':"", 'PASS':""}
    ):
        self.name = data_name
        self.part=part
        self.CNN=CNN
        self.ReLU_preproc=ReLU_preproc
        self.noise=noise
        self.logging_dir=logging_dir
        self.model_name=model_name
        self.data_paths=data_paths
        self.plot_dir=plot_dir
        # Load original data.
        self.get_features_vectors()
        # In case of SNN, convert input data with TTFS coding.
        if not ReLU_preproc: self.convert_ttfs()       
        
    def get_features_vectors(self):
        """
        Load image datasets and transform into features. 
        """
        if 'MNIST' in self.name:
            # MNIST or Fashion-MNIST dataset.
            self.input_shape, self.num_of_classes=(28, 28, 1), 10
            self.q, self.p = 1.0, 0.0
            if self.name=='MNIST':
                # MNIST dataset
                (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
            else:
                # Fashion-MNIST dataset.
                (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
            # Scale to [0, 1] range.
            self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0
            # Reshape.
            if not self.CNN:
                self.x_train, self.x_test = self.x_train.reshape((len(self.x_train), -1)), self.x_test.reshape((len(self.x_test), -1)) 
            else:
                self.x_train, self.x_test = self.x_train.reshape(-1, 28, 28, 1), self.x_test.reshape(-1, 28, 28, 1) 
        elif 'CIFAR' in self.name:
            # CIFAR10 or CIFAR100 dataset.
            self.input_shape=(32, 32, 3)
            self.q, self.p = 3.0, -3.0
            if self.name=='CIFAR10':
                # CIFAR10 dataset.
                self.num_of_classes = 10
                (self.x_train,self.y_train), (self.x_test,self.y_test)=tf.keras.datasets.cifar10.load_data()
                if self.plot_dir is not None:
                    # Save image for Fig. 4.
                    im = Image.fromarray(self.x_test[0], mode="RGB")
                    im.save(self.plot_dir + "/input_image.jpeg")
                # Mean and std to scale input.
                self.mean_test, self.std_test=120.707, 64.15
            else:
                # CIFAR100 dataset.
                self.num_of_classes = 100
                (self.x_train,self.y_train), (self.x_test,self.y_test)=tf.keras.datasets.cifar100.load_data()
                # Mean and std to scale input.
                self.mean_test, self.std_test=121.936, 68.389
            # Scale to [-3, 3] range.
            self.x_test, self.x_train=(self.x_test-self.mean_test)/(self.std_test+1e-7), (self.x_train-self.mean_test)/(self.std_test+1e-7)
        elif self.name=='PLACES365':
            # PLACES365 dataset.
            PART_SIZE=3000
            self.input_shape, self.num_of_classes=(224, 224, 3), 365
            self.q, self.p = 200, -200
            # Read labels for validation set.
            with open(f'{self.data_paths["PLACES365"]}/places365_val.txt') as f: lines = f.readlines()
            if self.ReLU_preproc:
                # During ReLU preprocessing complete validation dataset is loaded. 
                self.x_test=np.empty((0, 3, 224, 224))
                for part in range(13):
                    x_test_part = np.load(f'{self.data_paths["PLACES365"]}/part{part}.npz')['arr_0']
                    self.x_test = np.concatenate([self.x_test, x_test_part])
            else:
                # During SNN inference dataset is loaded in parts to parallelize execution.
                self.x_test = np.load(f'{self.data_paths["PLACES365"]}/part{self.part}.npz')['arr_0']
                lines = lines[self.part*PART_SIZE:(self.part+1)*PART_SIZE]
            self.y_test = np.array(list(map(int, list(map(lambda x: x.split()[-1], lines)))))
            self.x_test = np.transpose(self.x_test, axes=[0, 2, 3, 1])
            # A sample of training data is loaded to calculate X_n. 
            self.x_train = np.load(f'{self.data_paths["PLACES365"]}/part0_train.npz')['arr_0']
            self.x_train, self.y_train = np.transpose(self.x_train, axes=[0, 2, 3, 1]), np.zeros(np.shape(self.x_train)[0])
        elif self.name=='PASS':
            # PASS dataset.
            PART_SIZE=10000
            self.input_shape, self.num_of_classes=(224, 224, 3), 1000
            self.q, self.p = 200, -200
            self.x_test=np.empty((0, 224, 224, 3))
            if self.ReLU_preproc:
                random_image_paths = shuffle_pass_data(self.logging_dir, self.data_paths['PASS'])
                #During ReLU preprocessing all 100000 test images are loaded. 
                for part in range(10):
                    print (part, '\n\n\n')
                    x_test_part = preprocess_pass_images(random_image_paths, self.data_paths['PASS'], PART_SIZE, start=50000*(part))
                    self.x_test = np.concatenate([self.x_test, x_test_part])
            else:
                # During SNN inference dataset is loaded in parts to parallelize execution.
                random_image_paths=pkl.load(open(self.logging_dir + '/random_image_paths.pk', 'rb'))
                self.x_test = preprocess_pass_images(random_image_paths, self.data_paths['PASS'], PART_SIZE, start=50000*(self.part))
            # Apply preprocessing which is done to imagenet (scaling). 
            self.x_test = tf.keras.applications.vgg16.preprocess_input(np.array(self.x_test))
            # A sample of training data is loaded to calculate X_n. 
            self.x_train = preprocess_pass_images(random_image_paths, self.data_paths['PASS'], size=5000, start=50000*(10))
            self.x_train = tf.keras.applications.vgg16.preprocess_input(np.array(self.x_train))
            self.y_train, self.y_test = np.zeros(np.shape(self.x_train)[0]), np.zeros(np.shape(self.x_test)[0])
        # Processing which is the same for all datasets.
        self.x_train, self.x_test = self.x_train.astype('float64'), self.x_test.astype('float64')
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_of_classes)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_of_classes)
        if not self.ReLU_preproc:
            # Load prediction labels for the agreement metric;
            self.y_test_agreement = pkl.load(open(self.logging_dir + '/' + self.model_name.replace('scaled', 'original_labels.pkl'), 'rb'))
            if self.name =='PLACES365' or self.name =='PASS':
                self.y_test_agreement = self.y_test_agreement[self.part*PART_SIZE:(self.part+1)*PART_SIZE]
                # Check that we have diverse labels.
                if self.name =='PASS': tf.print (np.argmax(self.y_test_agreement[:500], axis=1), summarize=500)
        print (np.shape(self.x_train), np.shape(self.y_train), '\n\n\n')
        print (np.shape(self.x_test), np.shape(self.y_test), '\n\n\n')
        
    def convert_ttfs(self):
        """
        Convert input values into time-to-first-spike spiking times.
        """
        self.x_test, self.x_train = (self.x_test - self.p)/(self.q-self.p), (self.x_train - self.p)/(self.q-self.p)
        self.x_train, self.x_test=1 - np.array(self.x_train), 1 - np.array(self.x_test)
        self.x_test=np.maximum(0, self.x_test + tf.random.normal((self.x_test).shape, stddev=self.noise, dtype=tf.dtypes.float64)) 
  

    



            
            
            
            
            
            
            
