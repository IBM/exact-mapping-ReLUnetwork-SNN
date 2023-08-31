import matplotlib 
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib import colors, cm
import numpy as np
from scipy.stats import norm
import tensorflow as tf
cm = 1/2.54
    
def plot_raster(tj, layer, model, i, plot_dir, logging_dir):
    """
    Plot raster plots of input and hidden layers for Fig4.  In case of conv2d_1, the picked indices are taken from the same channel and saved to match the spiking plot. 
    """
    fig, ax = plt.subplots(figsize=(15*cm, 8*cm)) 
    plt.yticks([0, 2, 4, 6, 8], visible=True)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(24)
    ax.xaxis.label.set_fontsize(24)
    if 'input' in layer.name:
        
        print (np.shape(tj), '\n\n\n')
        
        tj = np.reshape(tj[:1], -1)
        max_neuron = np.argmax(1-tj)
        others = list(range(len(tj)))
        others.remove(max_neuron)
        neuron_inds = np.concatenate([[max_neuron], np.random.choice(others, 9)])
        spike_times = tj[neuron_inds]
        # Color is the value of ReLU output.
        color = 1-spike_times
        # Plotting margin for the markers to fit.
        plt.xlim([0, 1.04]) 
    else:
        extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
        ti = (extractor(tj[:1], training=False).numpy())[0]
        x_relu = pkl.load(open(logging_dir + '/plot_outputs.pkl', 'rb'))[i][0]
        #
        if layer.name=='conv2d_1':
            _, image_size, channels = np.shape(ti) 
            ti = tf.transpose(ti, [2, 0, 1]).numpy()
            x_relu = tf.transpose(x_relu, [2, 0, 1]).numpy()
            ti, x_relu = np.reshape(ti, -1), np.reshape(x_relu, -1)
            max_neuron = np.argmax(layer.t_max-ti)
            max_neuron_channel = max_neuron // (image_size**2)
            others = list(range(image_size**2))
            others.remove(max_neuron%(image_size**2))
            offset = max_neuron_channel*(image_size**2)
            neuron_inds = np.concatenate([[max_neuron], offset+np.random.choice(others, 9)])
        else:
            ti, x_relu = np.reshape(ti, -1), np.reshape(x_relu, -1)
            max_neuron = np.argmax(layer.t_max-ti)
            others = list(range(len(ti)))
            others.remove(max_neuron)
            neuron_inds = np.concatenate([[max_neuron], np.random.choice(others, 9)])
        # Color is the value of ReLU output.
        spike_times, color= ti[neuron_inds], x_relu[neuron_inds]
        # Plotting margin for the markers to fit.
        plt.xlim([layer.t_min, layer.t_max + 0.04*(layer.t_max-layer.t_min)])
    # Plot colorbar.
    points = ax.scatter(spike_times, list(range(10)), c=color, s=100, cmap="Blues", edgecolors='black')
    cbar = fig.colorbar(points)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    for t in cbar.ax.get_yticklabels(): t.set_fontsize(24)
    plt.savefig(f'{plot_dir}/raster_plot_{layer.name}.pdf')
    return neuron_inds
    
        
def plot_output_potential(tj, layer_prev, layer_softmax, model, plot_dir, logging_dir):
    """
    Plot membrane potential of output layer for Fig. 4. 
    """
    fig, ax = plt.subplots(figsize=(15*cm, 8*cm)) 
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(24)
    ax.xaxis.label.set_fontsize(24)
    # Get spiking times of previous layer.
    extractor = tf.keras.Model(inputs=model.inputs, outputs=layer_prev.output)
    tj = (extractor(tj[:1], training=False)).numpy()
    # Sort spiking times of previous layer and their corresponding weights.
    tj_arg_sorted = tf.argsort(tj, -1)
    tj_sorted = tf.transpose(tf.sort(tj))     
    # Integration starts at t_min_prev and ends at t_min.
    tj_sorted = tf.concat([tj_sorted, tf.fill((1, 1), tf.cast(layer_softmax.t_min, dtype=tf.float64))], axis=0)
    tj_sorted = tf.concat([tf.fill((1, 1), tf.cast(layer_softmax.t_min_prev, dtype=tf.float64)), tj_sorted], axis=0)
    J_sorted = tf.squeeze(tf.gather(layer_softmax.kernel, tj_arg_sorted, axis=0))  
    J_sorted = tf.concat([tf.expand_dims(tf.cast(layer_softmax.alpha, dtype=tf.float64), axis=0), J_sorted], axis=0)
    J_sum = tf.math.cumsum(J_sorted, axis=0) 
    tj_sorted_diff = np.diff(tj_sorted.numpy(), axis=0)                     
    W_mult_a = tf.math.multiply(J_sum, tj_sorted_diff)
    V = tf.cumsum(W_mult_a, axis=0)
    # Transparency is determined by the value of softmax output in ReLU network. Smaller values correspond to paler colors. 
    x_softmax = pkl.load(open(logging_dir + '/plot_outputs.pkl', 'rb'))[-1][0]
    line_alpha = (x_softmax-np.min(x_softmax))/(np.max(x_softmax) - np.min(x_softmax))
    # Pink color for output.
    cdict={'red':   [[0.0,  1, 1],
                   [1.0,   0.9137, 0.9137]],
         'green': [[0.0,  1, 1],
                   [1.0,  0.4901, 0.4901]],
         'blue':  [[0.0,  1, 1],
                   [1.0,  0.6313, 0.6313]]}
    cmap = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict)
    # 10 lines for 10 output neurons.
    for line in range(10):
        ax.plot(tj_sorted, tf.concat([tf.zeros((1), tf.float64), V[:, line]], axis=0), c='#E97DA1', linewidth=5, alpha=line_alpha[line])  
    # Plot colorbar.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=np.min(x_softmax),vmax=np.max(x_softmax)))
    sm.set_array([])
    cbar=plt.colorbar(sm, ticks=None, boundaries=None, ax=[ax])
    for t in cbar.ax.get_yticklabels(): t.set_fontsize(24)
    plt.xlim([layer_softmax.t_min_prev, layer_softmax.t_min])
    plt.savefig(f'{plot_dir}/potential_plot_{layer_softmax.name}.pdf')
    

def plot_spiking_distribution(tj, layer, model, batch_size, plot_dir):
    """
    Plot distribution of spikes in conv2d_1 layer accross test inputs for Fig. 4. 
    """
    fig, ax = plt.subplots(figsize=(15*cm, 8*cm)) 
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(24)
    ax.xaxis.label.set_fontsize(24)
    # Get spiking times for all inputs tj.
    extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    output = extractor.predict(tj, batch_size = batch_size, verbose=1)
    output = (tf.reshape(output, -1)).numpy()
    # Most of the spikes happen at t_max. Get the percentage of neurons that spike before t_max.
    before_tmax = tf.math.count_nonzero(layer.t_max-output)
    print ('Percentage of neurons which spike before t_max:', before_tmax/len(output))
    # Plot distributions of all spiking times in conv2d_1 layer. 
    ax.hist(output, bins=50, range=[layer.t_min, layer.t_max], color='red')
    ax.set_yscale('log')
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9),numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.xlim([layer.t_min, layer.t_max + 0.04*(layer.t_max-layer.t_min)])
    plt.savefig(f'{plot_dir}/spiking_distribution_plot_{layer.name}.pdf')

def plot_potential_and_spikes(tj, layer_prev, layer, model, neuron_ids, plot_dir, logging_dir):
    """
    Plot potential and spikes of few neurons in conv2d_1 layer for Fig. 4. 
    """
    fig, ax = plt.subplots(figsize=(20*cm, 12*cm)) 
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(24)
    # Get spiking times of previous layer.
    extractor = tf.keras.Model(inputs=model.inputs, outputs=layer_prev.output)
    tj = (extractor(tj[:1], training=False)).numpy()
    # Pad the input with t_min.
    tj=tf.pad(tj, tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]]), constant_values=layer.t_min)
    # Extract image patches. call_spiking function is called for different patches in parallel. 
    tj = tf.image.extract_patches(tj, sizes=[1, layer.kernel_size[0], layer.kernel_size[1], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    def plot_potential(neuron_ind, tj, layer, y_lim, x_relu, color):
        """
        Plot potential of one neuron in conv2d_1 layer for Fig. 4. 
        """
        image_size=32
        # Get neuron position from neuron_ind.
        channel, pixel_pos_x, pixel_pos_y = neuron_ind//(image_size**2), (neuron_ind//image_size)%image_size, (neuron_ind)%image_size
        # We reshape input and weights in order to utilize the same function as for the fully-connected layer.
        J = tf.reshape(layer.kernel, (-1, layer.filters))
        # Extract inputs for that particular neuron.
        tj = tf.reshape(tj[:, pixel_pos_x, pixel_pos_y, :], (-1, tf.shape(J)[0])) 
        # Sort spiking times of previous layer and their corresponding weights.
        tj_arg_sorted = tf.argsort(tj, -1)
        tj_sorted = tf.transpose(tf.sort(tj))
        tj_sorted = tf.concat([tj_sorted, tf.fill((1, 1), tf.cast(layer.t_max, dtype=tf.float64))], axis=0)
        tj_sorted = tf.concat([tf.fill((1, 1), tf.cast(layer.t_min_prev, tf.float64)), tj_sorted], axis=0)
        J_sorted = tf.squeeze(tf.gather(J, tj_arg_sorted, axis=0))
        # If spike hasn't happened before t_max, force it at t_max by having a large input at t_max+0.001
        if x_relu==0:
            tj_sorted = tf.concat([tj_sorted, tf.fill((1, 1), tf.cast(layer.t_max+0.01, dtype=tf.float64))], axis=0)
            J_sorted = tf.concat([J_sorted, tf.fill((1, layer.filters), tf.cast(1000.0, dtype=tf.float64))], axis=0)
        J_sorted = tf.concat([tf.expand_dims(tf.cast(layer.alpha, dtype=tf.float64), axis=0), J_sorted], axis=0)
        J_sum = tf.math.cumsum(J_sorted, axis=0) 
        tj_sorted_diff = np.diff(tj_sorted.numpy(), axis=0)
        J_mult_tj_diff = tf.math.multiply(J_sum, tj_sorted_diff)
        V = tf.cumsum(J_mult_tj_diff, axis=0)
        ax.plot(tj_sorted, tf.concat([tf.zeros((1), tf.float64), V[:, channel]], axis=0), c=color, linewidth=2) 
        plt.xlim([layer.t_min_prev, layer.t_max + 0.04*(layer.t_max-layer.t_min_prev)])
        plt.ylim([0, max(layer.threshold[0][channel]*1.2, y_lim)])
        y_lim = max(layer.threshold[0][channel]*1.2, y_lim)
        # Plot horizontal line to denote threshold (neuron is taken from the inner partition). 
        plt.hlines(layer.threshold[0][channel], layer.t_min_prev, layer.t_max + 0.04*(layer.t_max-layer.t_min_prev), color='black', linestyles='dashed')
        return y_lim
    # Load outputs of the layers in the ReLU network.
    x_relu = pkl.load(open(logging_dir + '/plot_outputs.pkl', 'rb'))[1][0]
    x_relu = np.reshape(tf.transpose(x_relu, [2, 0, 1]).numpy(), -1)
    x_relu_arg_sorted = np.argsort(x_relu[neuron_ids])
    neuron_ids, y_lim = np.array(neuron_ids), 0
    # Plot potentials of 3 chosen neurons in the same channel.
    y_lim = plot_potential(neuron_ids[x_relu_arg_sorted[-1]], tj, layer, y_lim, x_relu[neuron_ids[x_relu_arg_sorted[-1]]], 'red')
    y_lim = plot_potential(neuron_ids[x_relu_arg_sorted[-2]], tj, layer, y_lim, x_relu[neuron_ids[x_relu_arg_sorted[-2]]], 'magenta')
    y_lim = plot_potential(neuron_ids[x_relu_arg_sorted[0]], tj, layer, y_lim, x_relu[neuron_ids[x_relu_arg_sorted[0]]], 'blue')
    # Print values of the corresponding neurons in the ReLU network.
    print('maximum value in red:', x_relu[neuron_ids[x_relu_arg_sorted[-1]]], '\n\n\n****')
    print('second maximum value in magneta:', x_relu[neuron_ids[x_relu_arg_sorted[-2]]], '\n\n\n****')
    print('minimum value in blue:', x_relu[neuron_ids[x_relu_arg_sorted[2]]], '\n\n\n****')
    # Plot vertical lines for t_min_prev, t_min and t_max.
    plt.vlines(layer.t_min_prev, 0, y_lim, color='black', linestyles='dashed')
    plt.vlines(layer.t_min, 0, y_lim, color='black', linestyles='dashed')
    plt.vlines(layer.t_max,  0, y_lim, color='black', linestyles='dashed')
    plt.savefig(f'{plot_dir}/spikes_plot_{layer.name}.pdf')
    

    
    
   