from keras.engine.topology import Layer
from keras.models import Model
from weight import _get_weight
from keras.layers import *

class NTM(Layer):
    def __init__(self,
        controller,                     # custom controller, should output a vector
        n_slots,mem_length,             # Memory config
        num_shift,                      # shifting
        num_read,num_write,             # controller-head config
        batch_size = 16,
        controller_instr_output_dim = None,
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

    def compute_output_shape(self, input_shape):
        seq_len = input_shape[1]
        read_vector_input_shape = (input_shape[0],self.num_read,self.M)
        controller_output_shape, _ = self.controller.compute_output_shape([input_shape,read_vector_input_shape])

        if not self.return_sequences:
            controller_output_shape = list(controller_output_shape)
            del controller_output_shape[1]
            controller_output_shape = tuple(controller_output_shape)

        return controller_output_shape

    def call(self,x):
        last_output,list_outputs,states = K.rnn(self.main_step_func,x,self.get_initial_states(x))
        print('last_output',last_output)
        print('list_outputs',list_outputs)
        return last_output if not self.return_sequences else list_outputs

    def get_initial_states(self,x):
        batch_size = self.batch_size
        # old_read_vectors,old_read_weights,old_write_weights,old_M
        read_vectors = [K.zeros((batch_size,self.M)) for i in range(self.num_read)]
        # read_weights = [K.zeros((batch_size,self.N)) for i in range(self.num_read)]
        read_weights = K.zeros((batch_size,self.num_read,self.N))
        # write_weights = [K.zeros((batch_size,self.N)) for i in range(self.num_write)]
        write_weights = K.zeros((batch_size,self.num_write,self.N))
        # # TODO: memory initializations?
        M = K.zeros((batch_size,self.N,self.M))

        read_vectors = K.stack(read_vectors,axis = 1)

        return [read_vectors,read_weights,write_weights,M]

from keras.models import Model
if __name__ == '__main__':
    seq_len = 41
    num_bits = 9

    num_read = 2
    num_write = 3

    num_slots = 128
    mem_length = num_bits
    batch_size = 16

    controller_instr_output_dim = num_bits + num_read * mem_length
    # controller
    i = Input((num_bits,))
    read_input = Input((num_read,mem_length))
    read_input_flatten = Flatten()(read_input)
    h = Dense(num_bits,activation = 'relu')(i)
    h2 = Concatenate()([h,read_input_flatten])
    controller_out = Dense(num_bits,activation = 'sigmoid')(h2)
    controller = Model([i,read_input],[controller_out,h2])
    controller.summary()
    # NTM
    i = Input((seq_len,num_bits))
    ntm_cell = NTM(
            controller,                     # custom controller, should output a vector
            num_slots,mem_length,           # Memory config
            num_shift = 3,                  # shifting
            batch_size = batch_size,
            controller_instr_output_dim = controller_instr_output_dim,
            return_sequences = True,
            num_read = num_read,num_write = num_write)(i)
    ntm = Model(i,ntm_cell)

    print("****************** Start Training *****************")
    def copy_task_generator(batch_size = 16,min_len = 1,max_len = 20,num_bits = 8):
        while True:
            res = np.zeros((batch_size,max_len * 2 + 1,num_bits + 1))
            output = np.zeros((batch_size,max_len * 2 + 1,num_bits + 1))
            mask = np.zeros((batch_size,max_len * 2 + 1))
            for bs in range(batch_size):
                l = np.random.randint(min_len,max_len + 1)
                for i in range(l):
                    for j in range(num_bits):
                        res[bs,i,j] = np.random.randint(2)
                # delim flag
                res[bs,l,num_bits] = 1.
                # output = np.zeros((max_len * 2 + 1, num_bits + 1))
                output[bs,l + 1:2 * l + 1,:] = res[bs,:l,:]
                output[bs,l,-1] = 1.
                mask[bs,l + 1: 2 * l + 1] = 1.
            yield (res,output,mask)
    from keras.optimizers import RMSprop
    import sys
    ntm.compile(loss = 'binary_crossentropy',
        metrics = ['binary_accuracy'],
        optimizer = RMSprop(1e-4),
        sample_weight_mode = 'temporal')
    epochs = 100
    steps = 500
    data_gen = copy_task_generator()
    for epoch in range(epochs):
        print()
        print('Epoch {}/{}:'.format(epoch,epochs))
        inp,target,mask = data_gen.__next__()
        for step in range(steps):
            loss,acc = ntm.train_on_batch(inp,target,sample_weight = mask)
            sys.stdout.write('\rstep %d/%d,loss:%.6g,acc:%.6g'%(step,steps,loss,acc))
            sys.stdout.flush()
        print()
