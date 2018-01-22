from keras.engine.topology import Layer
from keras.models import Model,load_model
from keras.layers import *
import keras.backend as K
'''
    File containing the necessary weights for getting weight vector given k,b,g,s,t from head
    M: (batch,N,M)
    k: (batch,M)
    b: (batch,)
    g: (batch,)
    s: (batch,num_shift)
'''
DEBUG = False
def _content_addressing(M,k,b):
    if DEBUG:
        assert K.ndim(M) == 3
        assert K.ndim(k) == 2
        assert K.ndim(b) == 1

    # cosine similarity
    nM = K.l2_normalize(M,axis = 1)     # (batch,N,M)
    nk = K.l2_normalize(k,axis = -1)    # (batch,M)
    nkx = K.expand_dims(nk,axis = -1)   # (batch,M,1)
    sim = K.batch_dot(nM,nkx)           # (batch,N,1)

    # multiply by b
    bx = K.expand_dims(b,axis = -1)     # (batch,1)

    mul = K.batch_dot(sim,bx)           # (batch,N)

    return K.softmax(mul)               # (batch,N)


# eq. 7 of the paper
def _interpolate(wc,w,g):
    if DEBUG:
        assert K.ndim(wc) == 2 == K.ndim(w)
        assert K.ndim(g) == 1
    wcx = K.expand_dims(wc,axis = -1)       # (batch,N,1)
    gx = K.expand_dims(g,axis = -1)         # (batch,1)
    wx = K.expand_dims(w,axis = -1)         # (batch,N,1)
    if DEBUG:
        print('wcx:',wcx)
        print('gx:',gx)
        print('wx:',wx)
    prod = K.batch_dot(wcx,gx)              # (batch,N)
    if DEBUG:
        print('prod',prod)
    prod = prod + K.batch_dot(wx, 1 - gx)   # (batch,N)

    return prod

# eq. 8 of the paper
'''
    wi: weight from interpolation (eq. 7), shape (?,N)
    s:  shifting kernel, shape (?,num_shift)
    n:  N (n_slots)
    num_shift: length of shifting kernel
'''
def _shift(wi,s):
    if DEBUG:
        assert type(n) == type(num_shift) == int
        assert K.ndim(wi) == 2 == K.ndim(s)
    num_shift = s.shape[-1].value

    bound = num_shift // 2
    head,tail = wi[:,:bound],wi[:,-bound:]
    wi = K.concatenate([tail,wi,head])
    wix = K.expand_dims(wi,1)              # (batch,1,N + num_shift)

    st = K.permute_dimensions(s,(1,0))     # (num_shift,batch)
    sx = K.expand_dims(st,1)               # (num_shift,1,batch)
    wix = K.permute_dimensions(wix,(0,2,1)) # (batch,N + num_shift,1)
    res = K.conv1d(wix,sx,data_format = 'channels_last')                 # (batch,N,batch)
    return K.stack([res[_,:,_] for _ in range(res.shape[0].value)])

# eq. 9 of the paper
def _sharpen(ws,g):
    if DEBUG:
        assert K.ndim(ws) == 2
        assert K.ndim(g) == 1

    gx = K.expand_dims(g,-1)                           # (batch,1)
    gx = K.repeat_elements(gx,ws.shape[-1],axis = 1)   # (batch,N)
    pow = K.pow(ws,gx)
    return pow / (K.sum(pow,axis = -1,keepdims = True) + 1e-12) # try not to divide by 0

# functions for getting weights from head
def _get_weight(M,w,k,b,g,s,t):
    wc = _content_addressing(M,k,b)
    wi = _interpolate(wc,w,g)
    ws = _shift(wi,s)
    w_fin  = _sharpen(ws,t)
    return w_fin

if __name__ == '__main__':
    import numpy as np
    # unit testing playground
    # batch = 100
    # n = 10
    # m = 20
    # num_shift = 3

    # M = K.ones((batch,n,m))
    # k = K.ones((batch,m))
    # b = K.ones((batch,))
    # g = K.ones((batch,))
    # w = K.ones((batch,n))
    # s = K.ones((batch,num_shift)) * 0.5
    # t = K.zeros((batch,))
    # ca = _content_addressing(M,k,b)
    # wg = _interpolate(ca,w,g)
    # ws = _shift(wg,s)
    # w_fin = _sharpen(ws,t)
    # with K.get_session().as_default():
    #     print(w_fin.eval())
    a = K.stack([K.variable(np.arange(_,10 + _,1)) for _ in range(5)])
    s = K.stack([K.variable([1. + _,0,0]) for _ in range(5)])
    with K.get_session().as_default():
        print(_shift(a,s).eval())

class NTM(Layer):
    def __init__(self,
        controller,                     # custom controller, should output a vector
        n_slots,mem_length,             # Memory config
        num_shift,                      # shifting
        num_read,num_write,             # controller-head config
        batch_size = 16,
        controller_instr_output_dim = None,
        is_controller_recurrent = False,
        initial_memory = None,
        **args):                        # others

        self.N = n_slots
        self.M = mem_length

        self.num_shift = num_shift
        self.num_read = num_read
        self.num_write = num_write

        # deal with the given controller
        self.controller = controller

        self.batch_size = batch_size
        self.controller_instr_output_dim = controller_instr_output_dim
        
        self.is_controller_recurrent = is_controller_recurrent 
        self.initial_memory = initial_memory
        if 'return_sequences' in args:
            self.return_sequences = args['return_sequences']
            del args['return_sequences']
        else:
            self.return_sequences = False
        super().__init__(**args)

    # weight construction
    def _construct_head_weights(self):
        # evaluate the output for all heads
        # b,g,t scalar, key strength (M,), shift (num_shift,)
        num_outputs = self.num_read * (3 + self.M + self.num_shift) + self.num_write * (3 + self.M + self.num_shift + self.M + self.M) # besides the weight vector,erase and add
        # create variables for the fully connected layers

        controller_out_dim = self.controller_instr_output_dim
        if not controller_out_dim:
            controller_out_dim = self.controller.output[-1].shape[-1].value

        kernel = self.add_weight(name = 'heads_kernel',
                                shape = (controller_out_dim,num_outputs),
                                initializer = 'glorot_uniform', # TODO: I should give you a choice...
                                trainable = True)

        bias = self.add_weight(name = 'heads_bias',
                                shape = (num_outputs,),
                                initializer = 'zero', # TODO: I should give you a choice...
                                trainable = True)
        return kernel,bias

    def _head_instruction_output_dim(self):
        return self.num_read * (3 + self.M + self.num_shift) + self.num_write * (3 + self.M + self.num_shift + self.M + self.M)

    def main_step_func(self,inputs,states):
        # should return output,new_states
        # retrieve the old states

        # get the rest of the inputs
        '''
            states:
            [
                [r1,r2,r3,...],
                [rw1,rw2,rw3,...],
                [ww1,ww2,ww3,...],
                M
            ]
        '''
        old_read_vectors,old_read_weights,old_write_weights,M = states
        # TODO: will this work?
        # chain the i/o of the controller to this graph
        # controller_out, (batch,...),head_instrs: (batch,?)
        controller_out, head_instrs = self.controller([inputs,old_read_vectors])

        # evaluate the instructions according to internal weights
        # head_instrs: (batch, nr * (3 + ns + m) + nw * (3 + ns + m + 2m))
        head_instrs = K.bias_add(K.dot(head_instrs,self.kernel), self.bias)
        # split the instructions
        read_heads,write_heads = self._split_head_instrutions(head_instrs)
        # read_heads should be of the form [w1,w2,w3,....], each of size (batch_size,N)
        read_heads = self._split_read_heads(M,old_read_weights,read_heads)
        # write_heads shuld be of th form [[w1,e1,a1],[w2,e2,a2],...], each w_i of size (batch_size,N), others of size(batch_size,k)
        write_heads = self._split_write_heads(M,old_write_weights,write_heads)

        # erase
        for w,e,a in write_heads:
            M = self._erase_memory(M,w,e)
        # add
        for w,e,a in write_heads:
            M = self._add_memory(M,w,a)
        # read
        next_read_vectors = []
        for weight in read_heads:
            rv = self._read_memory(M,weight)
            next_read_vectors.append(rv)

        # get only the weights of the write heads out
        for i,(w,e,a) in enumerate(write_heads):
            write_heads[i] = w

        # return everything
        next_read_vectors = K.stack(next_read_vectors,axis = 1)
        read_heads = K.stack(read_heads,axis = 1)
        write_heads = K.stack(write_heads,axis = 1)
        return controller_out,[next_read_vectors,read_heads,write_heads,M]

    # head instructions interpretations
    def _split_head_instrutions(self,head_instrs):
        # this assumes the head_instrs tensor is a vector. if it is not then errors will arise
        num_readhead_outs = self.num_read * (3 + self.M + self.num_shift)
        return head_instrs[:,:num_readhead_outs],head_instrs[:,num_readhead_outs:]

    # small loop for extracting params given a vector with shape (batch_size,3 + self.M + self.num_shift)
    def _get_weight_vector(self,M,w,head):
        cur = 0
        # split everything out
        k = head[:,cur:self.M]
        cur += self.M

        b = head[:,cur]
        cur += 1

        g = head[:,cur]
        cur += 1

        s = head[:,cur: cur + self.num_shift]
        cur += self.num_shift

        t = head[:,cur]

        # do the activations of the head
        # ref: https://blog.wtf.sg/2014/11/09/neural-turing-machines-implementation-hell/
        b = K.exp(b)
        g = K.sigmoid(g)
        s = K.softmax(s)
        t = K.softplus(t) + 1

        # DEBUG-ing purpose:
        # for _ in ['M','w','k','b','g','s','t']:
        #     print(_,eval(_))

        weight = _get_weight(M,w,k,b,g,s,t)
        return weight

    '''
        M: (batch,N,M)
        ws:(batch,num_read,N)
        read_heads_instrs: (batch,num_read * (3 + M + num_shift) )
    '''
    def _split_read_heads(self,M,ws,read_heads_instrs):
        res = []
        head_output_len = 3 + self.M + self.num_shift
        for i in range(self.num_read):
            head = read_heads_instrs[:,i * head_output_len:(i + 1) * head_output_len]

            weight = self._get_weight_vector(M,ws[:,i,:],head)
            res.append(weight)

        return res
    '''
        M: (batch,N,M)
        ws:(batch,num_read,M)
        read_heads_instrs: (batch,num_read * (3 + M + num_shift) )
    '''
    def _split_write_heads(self,M,ws,write_heads_instrs):
        res = []
        head_output_len = 3 + self.M + self.num_shift + self.M + self.M
        weight_len = 3 + self.M + self.num_shift
        for i in range(self.num_write):
            head = write_heads_instrs[:,i * head_output_len : (i + 1) * head_output_len]
            weight_head = head[:,:weight_len]

            # get the writing weights specific to the particular old head
            weight = self._get_weight_vector(M,ws[:,i,:],weight_head)

            erase_vec = head[:,weight_len:weight_len + self.M]
            add_vec = head[:,weight_len + self.M : weight_len + self.M + self.M]

            res.append([weight,erase_vec,add_vec])

        return res

    def _read_memory(self,M,w):
        Ms = K.permute_dimensions(M,(0,2,1))
        res = K.batch_dot(Ms,w)
        return res
    '''
        Add memory according to eq. 3 in (3.2)
        M : (None,M,N)
        w: (None, N)
        e: (None, M)
    '''
    def _erase_memory(self,M,w,e):
        ex = K.expand_dims(e,1) # (None,1,M)
        wx = K.expand_dims(w,-1) # (None, N,1)
        prod = K.batch_dot(wx,ex) # (None,N,M)
        prod = 1 - prod # same shape
        res = M * prod # (None,M,N)
        return res

    '''
        Add memory according to eq. 4 in (3.2)
        M : (None,M,N)
        w: (None, N)
        a: (None, M)
    '''
    def _add_memory(self,M,w,a):
        ax = K.expand_dims(a,1) # (None,1,M)
        wx = K.expand_dims(w,-1) # (None,N, 1)
        prod = K.batch_dot(wx,ax) # (None, N,M)
        return M + prod

    # overriding Recurrent layer functions
    def build(self,input_shape):
        k,b = self._construct_head_weights()
        self.kernel = k
        self.bias = b
        
        self.built = True

    def compute_output_shape(self, input_shape):
        seq_len = input_shape[1]
        single_step_input_shape = [s for i,s in enumerate(input_shape) if i != 1]
        
        read_vector_input_shape = (input_shape[0],self.num_read,self.M)
        controller_output_shape, _ = self.controller.compute_output_shape([single_step_input_shape,read_vector_input_shape])
        '''
        if not self.return_sequences:
            controller_output_shape = list(controller_output_shape)
            del controller_output_shape[1]
            controller_output_shape = tuple(controller_output_shape)
        '''
        if self.return_sequences:
            controller_output_shape = list(controller_output_shape)
            controller_output_shape.insert(1,seq_len)
            controller_output_shape = tuple(controller_output_shape)
        
        return controller_output_shape

    def call(self,x):
        # add the weights here
        self.trainable_weights += self.controller.trainable_weights
        last_output,list_outputs,states = K.rnn(self.main_step_func,x,self.get_initial_states(x))
        # plot the states
        self.save_states(states)
        return last_output if not self.return_sequences else list_outputs

    def save_states(self,states):
        read_vectors,read_weights,write_weights,M = states
        # TODO: what then?
        
    def get_initial_states(self,x):
        batch_size = self.batch_size
        # old_read_vectors,old_read_weights,old_write_weights,old_M
        read_vectors = [K.zeros((batch_size,self.M)) for i in range(self.num_read)]
        # read_weights = [K.zeros((batch_size,self.N)) for i in range(self.num_read)]
        read_weights = K.zeros((batch_size,self.num_read,self.N))
        # write_weights = [K.zeros((batch_size,self.N)) for i in range(self.num_write)]
        write_weights = K.zeros((batch_size,self.num_write,self.N))
        
        if None is self.initial_memory:
            M = np.zeros((batch_size,self.N,self.M))
            M[:,self.N // 2,:] = np.ones((batch_size,self.M))
            M = K.variable(M)
        else:
            M = K.variable(self.initial_memory)
            
        read_vectors = K.stack(read_vectors,axis = 1)

        return [read_vectors,read_weights,write_weights,M]

