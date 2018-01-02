from keras.layers.recurrent import Recurrent
from keras.engine.topology import Layer

class NTM(Layer):
    def __init__(self,
        output_dim,
        controller,                     # custom controller, should output a vector
        n_slots,mem_length,             # Memory config
        num_shift = 1,                  # shifting
        num_read,num_write,             # controller-head config
        **args):                        # others

        self.N = n_slots
        self.M = mem_length

        self.num_shift = num_shift
        self.num_read = num_read
        self.num_write = num_write

        # deal with the given controller
        self.controller = controller
        read_phs,instr_outs = self._check_controller()
        self.read_vector_input = read_phs
        self.head_instruction_outputs = instr_outs

        self.C = self._circulant(self.n_slots, self.num_shft)

        super().__init__(**args)

    def _check_controller(self):

        if not self.controller:
            raise ValueError("controller is empty")
        try:
            read_ph = self.controller.get_layer('read').input
            instr_outs = self.controller.get_layer('head').output
            return read_ph,instr_outs
        except:
            raise ValueError("The given controller should have two layers with name 'read' and 'head' respsectively")

    # thanks for https://github.com/flomlo/ntm_keras/blob/fdfe3ff0e3f5d3e4bc1a2fe51afdf67dae3aa5d8/ntm.py#L26
    def _circulant(self,leng, n_shifts):
        # This is more or less the only code still left from the original author,
        # EderSantan @ Github.
        # My implementation would probably just be worse.
        # Below his original comment:

        """
        I confess, I'm actually proud of this hack. I hope you enjoy!
        This will generate a tensor with `n_shifts` of rotated versions the
        identity matrix. When this tensor is multiplied by a vector
        the result are `n_shifts` shifted versions of that vector. Since
        everything is done with inner products, everything is differentiable.
        Paramters:
        ----------
        leng: int > 0, number of memory locations
        n_shifts: int > 0, number of allowed shifts (if 1, no shift)
        Returns:
        --------
        shift operation, a tensor with dimensions (n_shifts, leng, leng)
        """
        eye = np.eye(leng)
        shifts = range(n_shifts//2, -n_shifts//2, -1)
        C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
        return K.variable(C.astype(K.floatx()))

    '''
        aux. functions for getting weights from head
        The dimension of tensors are (supposed to be...):
        M: (N,M)
        k: (M,)
        b: ()
        g: ()
        s: (num_shift,)
        all weights: (N,)

    '''
    def _content_addressing(self,M,k,b):
        sim = self._cos_sim(M,k)
        mul = b * sim
        return K.softmax(mul)

    def _cos_sim(self,a,b):
        na = K.l2_normalize(a)
        nb = K.l2_normalize(b)
        return K.dot(na,nb)

    # eq. 7 of the paper
    def _interpolate(self,wc,w,g):
        return g * wc + (1 - g) * w

    # eq. 8 of the paper
    def _shift(self,wi,s):
        # TODO: what now?!
        pass
    # eq. 9 of the paper
    def _sharpen(self,ws,g):
        pow = K.pow(ws,g)
        return pow / (K.sum(pow) + 1e-12) # try not to divide by 0

    # functions for getting weights from head
    def _get_weight(self,M,w,k,b,g,s,t):
        wc = self._content_addressing(M,k,b)
        wi = self._interpolate(wc,w,g)
        ws = self._shift(wi,s)
        w_fin  = self._sharpen(ws,t)
        return w_fin

    # weight construction
    def _construct_head_weights(self):
        # evaluate the output for all heads
        num_ouputs = self.num_read * (3 + self.M + self.num_shift) # b,g,t scalar, key strength (M,), shift (num_shift,)
            + self.num_write * (3 + self.M + self.num_shift + self.M + self.M) # besides the weight vector,erase and add
        # create variables for the fully connected layers
        controller_out_dim = self.controller.output.shape[-1]

        kernel = self.add_weight(name = 'heads_kernel',
                                shape = (controller_out_dim,num_outputs),
                                initializer = 'glorot_uniform', # TODO: I should give you a choice...
                                trainable = True)

        bias = self.add_weight(name = 'heads_bias',
                                shape = (num_outputs,),
                                initializer = 'zero', # TODO: I should give you a choice...
                                trainable = True)
        return kernel,bias

    def main_step_func(self,inputs,states):
        # should return output,new_states
        # retrieve the old states

        # get the rest of the inputs
        old_read_vectors, old_M = states
        # TODO: i'm not sure how to work on multiple controller i/o tensors
        # for inp in self.controller.inputs:
        #     if inp.input
        # controller_inputs = [inp for inp in self.controller.inputs if inp is not self.read_vector_input][0]
        # controller_outputs = [oup for oup in self.controller.outputs if oup is not self.head_instruction_outputs][0]
        # TODO: will this work?
        controler_out, head_instrs = self.controller([inputs,old_read_vectors])

        # evaluate the instructions according to internal weights
        head_instrs = K.dot(head_instrs,self.kernel) + self.bias
        # split the instructions
        read_heads,write_heads = self._split_head_instrutions(head_instrs)
        read_heads = self._split_read_heads(read_heads)
        write_heads = self._split_write_heads(write_heads)

    # head instructions interpretations
    def _split_head_instrutions(self,head_instrs):
        # this assumes the head_instrs tensor is a vector. if it is not then errors will arise
        num_readhead_outs = self.num_read * (3 + self.M + self.num_shift)
        return head_instrs[:,:num_readhead_outs],head_instrs[:,num_readhead_outs:]

    # small loop for extracting params given a vector with shape (batch_size,3 + self.M + self.num_shift)
    def _get_weight_vector(self,head):
        cur = 0
        head = read_heads_instrs[i:i + head_output_len]
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
        return [k,b,g,s,t]

    def _split_read_heads(self,read_heads_instrs):
        res = []
        head_output_len = 3 + self.M + self.num_shift
        for i in range(0,self.num_read,head_output_len):
            cur = 0 # ref to the index of the consumed outputs
            head = read_heads_instrs[i:i + head_output_len]

            head_params = self._get_weight_vector(head)
            res.append(head_params)

        return res
    def _split_write_heads(self,write_heads_instrs):
        res = []
        head_output_len = 3 + self.M + self.num_shift + self.M + self.M

    # overriding Recurrent layer functions
    def build(self,input_shape):
        k,b = self._construct_head_weights()
        self.kernel = k
        self.bias = b

    def call(self,x):
        return K.rnn(self.main_step_func,x,self.get_initial_states)

    def get_initial_states(self):
        pass
