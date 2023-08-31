import numpy as np
import sys
import caffe
import pickle
import os
from numpy import savez_compressed

data_path_train=''
data_path_val=''
VAL_PART=0
BATCH_SIZE=3000

def classify_scene(net, im):
    """
    Get preprocessed PLACES365 data.
    """
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) 
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,3,224,224)
    # Preprocess input data.
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    # Output model label for that input. 
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    return top_k[0], net.blobs['data'].data[...]


if __name__ == '__main__':

    # fetch pretrained models
    fpath_design = 'models_places/deploy_vgg16_places365.prototxt'
    fpath_weights = 'models_places/vgg16_places365.caffemodel'
    x_test, x_train, params=[], [], {}
    # initialize net
    net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)
    for k in net.params:
        params[k] = (net.params[k][0].data[...], net.params[k][1].data[...])
    # Store pretrained model parameters.
    savez_compressed('models_places/places_vgg_relu_original.npz', **params)
    # Load validation images.
    files_num = len (os.listdir(data_path_val)) 
    begin, end = VAL_PART*BATCH_SIZE+ 1, min((VAL_PART+1)*BATCH_SIZE + 1, files_num+1)
    for i in range(begin, end):
        file_path = data_path_val + '/Places365_val_{:08d}.jpg'.format(i)
        im = caffe.io.load_image(file_path)
        # Obtain prediction label and preprocessed image for the current image.
        label, x = classify_scene(net, im)
        x_test.append(np.array(np.squeeze(x)))
    # Save preprocessed validation images.
    savez_compressed('models_places/part{}.npz'.format(VAL_PART), np.array(x_test))
    # Load training images.
    for file in os.listdir(data_path_train):
        try:
            im = caffe.io.load_image(data_path_train + file)
            # Obtain preprocessed image for the current image.
            _, x = classify_scene(net, im)
            x_train.append(np.array(np.squeeze(x)))
        except IOError:
            print ('passing*******')
            pass
    print (len(x_train))
    # Save preprocessed training images which will be used to calculate X_n.
    savez_compressed('models_places/part0_train.npz', np.array(x_train))

