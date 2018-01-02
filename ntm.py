from keras.layers.recurrent import Recurrent
from keras.engine.topology import Layer

from weight import _get_weight

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
        '''
            states:
            [
                [r1,r2,r3,...],
                [rw1,rw2,rw3,...],
                [ww1,ww2,ww3,...],
                M
            ]
        '''
        old_read_vectors,old_read_weights,old_write_weights,old_M = states
        # TODO: will this work?
        # chain the i/o of the controller to this graph
        controller_out, head_instrs = self.controller([inputs,old_read_vectors])

        # evaluate the instructions according to internal weights
        head_instrs = K.dot(head_instrs,self.kernel) + self.bias
        # split the instructions
        read_heads,write_heads = self._split_head_instrutions(head_instrs)
        # read_heads should be of the form [w1,w2,w3,....], each of size (batch_size,N)
        read_heads = self._split_read_heads(old_M,old_read_vectors,read_heads)
        # write_heads shuld be of th form [[w1,e1,a1],[w2,e2,a2],...], each w_i of size (batch_size,N), others of size(batch_size,k)
        write_heads = self._split_write_heads(old_M,old_read_vectors,write_heads)

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
        return controller_out,[next_read_vectors,read_heads,write_heads]

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

        weight = _get_weight(M,w,k,b,g,s,t)
        return weight

    def _split_read_heads(self,M,ws,read_heads_instrs):
        res = []
        head_output_len = 3 + self.M + self.num_shift
        for i in range(0,self.num_read,head_output_len):
            head = read_heads_instrs[:,i:i + head_output_len]

            weight = self._get_weight_vector(M,ws[i],head)
            res.append(weight)

        return res
    def _split_write_heads(self,M,ws,write_heads_instrs):
        res = []
        head_output_len = 3 + self.M + self.num_shift + self.M + self.M
        weight_len = 3 + self.M + self.num_shift
        for i in range(0,self.num_write,head_output_len):
            head = write_heads_instrs[:,i: i + head_output_len]
            weight_head = head[:,:weight_len]

            # get the writing weights specific to the particular old head
            head_params = self._get_weight_vector(M,ws[i],weight_head)

            erase_vec = head[:,weight_len:weight_len + self.M]
            add_vec = head[:,weight_len + self.M : weight_len + self.M + self.M]

            res.append([*head_params,erase_vec,add_vec])

        return res

    def _get_read_vector(self,M,w):
        pass
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

    def call(self,x):
        return K.rnn(self.main_step_func,x,self.get_initial_states())

    def get_initial_states(self):
        pass
