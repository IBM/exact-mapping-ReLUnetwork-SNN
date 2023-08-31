import os
from utils import *
from model import *
from plot_utils import *
from Dataset import Dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.keras.backend.set_floatx('float64')

FLAGS = tf.compat.v1.app.flags.FLAGS
# Basic flags
tf.compat.v1.flags.DEFINE_string("data_name", "MNIST", '["MNIST", "fMNIST", "cifar10", "cifar100", "pass", "places"]')
tf.compat.v1.flags.DEFINE_string("places_data_path", "", "path to the places dataset files (obtained by calling PLACES365_pretrained_model_preprocessed_data.py)")
tf.compat.v1.flags.DEFINE_string("pass_data_path", "", "path to the PASS dataset stored in 20 subfolders")
tf.compat.v1.flags.DEFINE_string("model_name", "", "name of the scaled ReLU model (needs to end with _scaled)")
tf.compat.v1.flags.DEFINE_bool("CNN", True, "CNN architecture yes/no")
tf.compat.v1.flags.DEFINE_string("logging_dir", '', "dir for logging")
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.compat.v1.flags.DEFINE_integer("layers", 16, "number of layers")
tf.compat.v1.flags.DEFINE_integer("part", 0, "part of the dataset")
# Flags for testing the robustness of the network to noise.
tf.compat.v1.flags.DEFINE_float("noise", 0, "standard deviation of the noise whcih is added to spiking times")
tf.compat.v1.flags.DEFINE_float("alpha_noise", 0, "standard deviation of noise which is added to alpha")
tf.compat.v1.flags.DEFINE_float("zeta", 0.5, "provides safety margin when calculating size of [t_min, t_max] range")
tf.compat.v1.flags.DEFINE_integer("trial", 0, "trial number in case of noise testing")
robustness_params={'noise':FLAGS.noise, 'alpha_noise': FLAGS.alpha_noise, 'zeta':FLAGS.zeta}

set_up_logging(FLAGS.logging_dir, 'phase2:' + FLAGS.model_name.replace('relu_scaled', f'snn{str(FLAGS.part)}'), FLAGS.zeta, FLAGS.noise, FLAGS.alpha_noise, FLAGS.trial)
plot_dir = set_up_plotting(FLAGS.logging_dir,  'phase2:' + FLAGS.model_name.replace('relu_scaled', f'snn{str(FLAGS.part)}'))

# Create data object
data = Dataset(
    FLAGS.data_name,
    FLAGS.logging_dir,
    FLAGS.CNN,
    ReLU_preproc=False,
    part=FLAGS.part,
    noise=FLAGS.noise,
    plot_dir=plot_dir,
    model_name=FLAGS.model_name,
    data_paths={'PLACES365':FLAGS.places_data_path, 'PASS':FLAGS.pass_data_path}
)
# Create model and load weights;
if 'MNIST' in FLAGS.data_name:
    # For MNIST we consider 3 architectures, 1. 2-layer fully-connected network, 2. 5-layer LeNet and 3. 16-layer VGG-like network.
    # For Fashion-MNIST we consider one architecture, a 16-layer VGG-like network.
    if not FLAGS.CNN:
        # if not CNN model, then create a shallow 2-layer fully connected network.
        model = create_fc_model_SNN(FLAGS.logging_dir, FLAGS.model_name, robustness_params=robustness_params) 
    else:
        if FLAGS.layers==5:
            # 5-layer CNN model, LeNet architecture. 
            model=create_lenet_model_SNN(FLAGS.logging_dir, FLAGS.model_name, robustness_params=robustness_params)
        elif FLAGS.layers==16:
            layers2D = [64, 64, 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
            kernel_size=(3,3)
            layers1D=[512,512]
            # 16-layer CNN model, VGG-like architecture. 
            model=create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, FLAGS.logging_dir, FLAGS.model_name, robustness_params=robustness_params)
if 'CIFAR' in FLAGS.data_name:
    # For CIFAR10 and CIFAR100 we consider one architecture, a 15-layer VGG-like network.
    layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
    kernel_size=(3,3)
    layers1D=[512]
    model=create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, FLAGS.logging_dir, FLAGS.model_name, robustness_params=robustness_params)
if 'PLACES365' in FLAGS.data_name or 'PASS' in FLAGS.data_name:
    # For PLACES 365, we consider one architecture, a VGG16 network.
    layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
    kernel_size=(3,3)
    layers1D=[4096, 4096]
    model=create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, FLAGS.logging_dir, FLAGS.model_name, robustness_params=robustness_params)
# Load weights from scaled ReLU network.
model.load_weights(FLAGS.logging_dir + '/' + FLAGS.model_name + '.h5', by_name=True) 

# Set parameters of SNN network: t_min_prev, t_min, t_max, J_ij, vartheta_i (threshold). Alpha is fixed at 1. 
t_min, t_max= 0, 1    
for layer in model.layers:
    if 'conv' in layer.name or 'dense' in layer.name: 
        t_min, t_max = layer.set_params(t_min, t_max)

logging.info ('Classification Latency:', model.layers[-1].t_min)

# Obtain accuracy of the SNN model (unless we work with PASS dataset) and prediciton agreement with the original ReLU network.   
logging.info("Testing.")
val_acc, agreement=[], []
predictions = model.predict(data.x_test, batch_size = FLAGS.batch_size, verbose=1)
predictions=np.argmax(predictions, axis=1)
if FLAGS.data_name!='PASS':
    test_acc =np.mean((predictions==np.argmax(data.y_test, axis=1)).astype(int))
    logging.info("Testing accuracy (SNN) is {}.".format(test_acc))
agreement = np.mean((predictions==np.argmax(data.y_test_agreement, axis=1)).astype(int))
logging.info("Agreement (SNN-ReLU) is {}.".format(agreement))

# In case of cifar10 dataset plot all plots in Fig. 4. 
if FLAGS.data_name=='CIFAR10':
    i=0
    for layer in model.layers[:-1]:
        if 'input' in layer.name:
            neuron_ids = plot_raster(data.x_test, layer, model, i, plot_dir, FLAGS.logging_dir)
        elif 'conv' in layer.name or 'dense' in layer.name:
            neuron_ids = plot_raster(data.x_test, layer, model, i, plot_dir, FLAGS.logging_dir)
            if layer.name=='conv2d_1':
                plot_spiking_distribution(data.x_test, layer, model, FLAGS.batch_size, plot_dir)
                plot_potential_and_spikes(data.x_test, model.get_layer('conv2d'), layer, model, neuron_ids, plot_dir, FLAGS.logging_dir)
            i+=1
    plot_output_potential(data.x_test[:1], model.get_layer('dense'), model.get_layer('dense_1'), model, plot_dir, FLAGS.logging_dir)



     
    

    
    
    