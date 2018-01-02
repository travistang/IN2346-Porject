from keras.layers import *
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback

class Visualizer:
    def __init__(self,model):
        assert(self.model.__class__.__name__ == "Model", "The model type has to be Model instead of Sequential or anything else")
        self.model = model
        # build Keras functions for evaluating gradient given input
        self.gradients = K.gradients(self.model.output,self.model.trainable_weights)
        self.input_placeholder = self.model.input
        layer_outputs = list(map(lambda l: l.output),self.model.layers)
        # TODO: what if there are layers with multiple outputs?
        self.outputs = K.function(self.input_placeholder,layer_outputs)
        
    def visualize_weights(self,key = None):
        return NotImplemented
         
    def visualize_activations(self,inp,key):
        return NotImplemented
    
    def visualize_gradients(self,inp,key):
        return NotImplemented
    
    # helper functions
    '''
        Return an image containing the RGB images of the convolutional kernels with given pad
    '''
    def visualize_kernels(self,kernels,pad = 1):
        # info of the kernel
        H,W,_,__ = kernels.shape
        # normalize the filters
        mi,ma = kernels.min(),kernels.max()
        kernels = (kernels - mi) / (ma - mi)
        # expand the first dimension into a multiple of 3
        num_pad = kernels.shape[0] % 3
        if num_pad > 0 :
            pad_time = 3 - num_pad
            kernels = np.pad(kernels,((0,pad_time),(0,0),(0,0),(0,0)),mode = 'edge')

        # reshape the kernel so that the first channel is 3 (r,g,b)
        kernels = kernels.reshape(3,H,W,-1)

        # sanity check on the dimension of the resultant kernel 
        # evaluate the size of kernels
        kern_height,H,W,num_kernels = kernels.shape
        lo_c = int(np.sqrt(num_kernels))
        hi_c = lo_c + 1

        output_height = (hi_c) * H + pad * (hi_c + 1)
        output_width = lo_c * W + pad * (lo_c + 1)

        output = np.zeros((output_height,output_width,3)) # TODO: flatten this!
        # iterate on the coordinates of output
        kern_id = 0
        for i in range(pad,output_height,H + pad):
            for j in range(pad,output_width,W + pad):
                patch = kernels[:,:,:,kern_id].transpose(1,2,0)
                output[i:(i + H),j:(j + W),:] = kernels[:,:,:,kern_id].transpose(1,2,0)
                kern_id += 1
                if kern_id >= kernels.shape[-1]: break
            if kern_id >= kernels.shape[-1]: break

        return output

class MatplotlibVisualizer(Visualizer):
    def __init__(self,model):
        super(Visualizer).__init__(self)

    def visualize_weights(self, key = None):
        if type(key) == int:
            weights = self.model.layers[key]
            cls_name = self.model.layers[key].__class__.__name__
            name = self.model.name[key]
            if name == 'Conv2D':
                # a convolutional 2D layer
                kernels, bias = weights
                kernel_img = self.visualize_kernels(kernels)
                # TODO: here? really?
                plt.imshow(kernel_img)
                
                # visualize bias
                plt.bar(np.arange(0,bias.shape[0]),bias)
            if name == 'Dense':
                # a fully connected layer
                kernels, bias = weights
                kernel_img = plt.imshow(kernels)
            else: return
        elif type(key) == 'str':
            # try to find layers by name
            layer = filter(lambda i,l: l.name == key,enumerate(self.model.layers))
            if len(layer) == 0: return 
            return self.visualize_weights(layer[0][0])
        elif key is None:
            for i in range(len(self.model.layers)):
                self.visualize_weights(i)
    def visualize_activations(self,inp,key):
        # TODO: does this work?
        outputs = self.outputs(inp)
        
class VisualizeCallback(Callback):
    def __init__(self,log_batch_interval,sample_generator, visualize_function_dict,**args):
        super().__init__(**args)
        self.interval = log_batch_interval
        self.sample_generator = sample_generator
        # accept the dictionary of visualization functions only when it matches the requirements
        self._construct_visualize_function_dict(visualize_function_dict)
        self.cur_batch = 0

    def _construct_visualize_function_dict(self,func_dict):
        if 'output' not in func_dict:
            func_dict['output'] = K.function(self.model.input,self.model.output)
    def on_batch_end(self,logs = {}):
        self.cur_batch += 1
        if self.cur_batch == self.interval:
            self.cur_batch = 0
            '''
                #TODO:
                1. get a sample from the generator
                2. evaluate the function given the input,
                3. evaluate th
            '''
    def on_epoch_end(self,logs = {}):
        self.cur_batch = 0

