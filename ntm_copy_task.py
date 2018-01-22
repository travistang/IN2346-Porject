from keras.models import *
from keras.layers import *
from keras.optimizers import *
from ntm import *
import csv
if __name__ == '__main__':
    min_len = 90
    max_len = 90
    seq_len = 2 * max_len + 1
    num_bits = 8

    num_read = 1
    num_write = 1

    num_slots = 128
    mem_length = num_bits
    batch_size = 4

    controller_instr_output_dim = num_bits + num_read * mem_length
    # controller
    i = Input((num_bits,))
    read_input = Input((num_read,mem_length))
    read_input_flatten = Flatten()(read_input)
    h = Concatenate()([i,read_input_flatten])
    h2 = Dense(100,activation = 'tanh')(h)
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
            #controller_instr_output_dim = controller_instr_output_dim,
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
    from keras.optimizers import RMSprop,Adam
    import sys
    ntm.compile(loss = 'binary_crossentropy',
        metrics = ['binary_accuracy'],
        optimizer = RMSprop(1e-3,clipnorm = 1.),
        sample_weight_mode = 'temporal')
    epochs = 200
    steps = 500
    data_gen = copy_task_generator(
        num_bits = num_bits - 1,
        min_len = min_len,
        max_len = max_len,
        batch_size = batch_size)
    # for visualizations
    #grad = K.gradients(ntm.outputs[0],controller.trainable_weights)
 
    finetune = True
    # load the weights
    if finetune:
        print('loading models')
        ntm.load_weights('ntm_weights.h5')
        controller = load_model('controller.h5')
    try:
        with open('log.csv','w') as csvf:
            csv_writer = csv.writer(csvf)
            for epoch in range(epochs):
                total_loss = 0.
                total_acc = 0.
                print()
                print('Epoch {}/{}:'.format(epoch,epochs))
                for step in range(steps):
                    inp,target,mask = data_gen.__next__()
                    if finetune:
                        loss,acc = ntm.evaluate(inp,target,sample_weight = mask)
                    else:
                        loss,acc = ntm.train_on_batch(inp,target,sample_weight = mask)
                    total_loss += loss
                    total_acc += acc
                    loss = total_loss / (1 + step)
                    acc = total_acc / (1 + step)
                    sys.stdout.write('\rstep %d/%d,loss:%.4g,acc:%.4g,lr: %f'%(step,steps,loss,acc,K.get_value(ntm.optimizer.lr)))
                    sys.stdout.flush()
                    
                    # write csv
                    csv_writer.writerow([epoch,step,loss,acc])
                print()
                test_inp,test_target,test_mask = data_gen.__next__()
                output = ntm.predict(test_inp,batch_size = batch_size)
                # get the first batch
                test_inp = test_inp[0]
                output = output[0]
                test_target = test_target[0]
                test_mask = test_mask[0]
                
                # save the tensor
                prefix = '' if not finetune else 'evaluate-'
                np.save('{}epoch_{}-input'.format(prefix,epoch),test_inp)
                np.save('{}epoch_{}-output'.format(prefix,epoch),output)
                np.save('{}epoch_{}-target'.format(prefix,epoch),test_target)
                np.save('{}epoch_{}-mask'.format(prefix,epoch),test_mask)
    except KeyboardInterrupt:
        pass
    finally:
        if not finetune:
            ntm.save_weights('ntm_weights.h5')
            controller.save('controller.h5')    