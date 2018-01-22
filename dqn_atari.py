from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
from ntm import NTM
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
batch_size = 1


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train','finetune','test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--model',type = str,default = None)
#TODO: this is not right! correct me!
parser.add_argument('--transfer-encoding', dest = 'transfer_encoding', action = 'store_false')
parser.add_argument('--agent-type',choices = ['conv','rnn','drnn','ntm'],default = 'ntm')
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

use_rnn = args.agent_type != 'conv'

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE if not use_rnn else (WINDOW_LENGTH,1) + INPUT_SHAPE
class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    # batch should be in shape (batch_size, channel,width,height)
    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        if use_rnn:
            processed_batch = np.expand_dims(processed_batch,axis = 2) # so that it becomes (batch_size, channel, 1, width,height) and for the channel it becomes a time axis. (1,width,height) is the image to the conv layers
        assert input_shape == processed_batch.shape[1:]
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)
    

model = Sequential()
permute_layer = None
if not use_rnn:
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
else:
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((1, 3, 4, 2), input_shape=input_shape))
        permute_layer = Permute((1, 3, 4, 2))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3,4), input_shape=input_shape))
        permute_layer = Permute((1,2,3,4))
        
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    
def get_conv_model(model):
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    print(model.layers)
    return model

def get_rnn_model(model):
    model.add(TimeDistributed(Conv2D(32,(8,8),activation = 'relu',subsample = (4,4))))
    model.add(TimeDistributed(Conv2D(64,(4,4),activation = 'relu',subsample = (2,2))))
    model.add(TimeDistributed(Conv2D(64,(3,3),activation = 'relu',subsample = (1,1))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(nb_actions,activation = 'linear'))
    print(model.summary())
    print(model.layers)
    return model
def get_double_rnn_model(model):
    model.add(TimeDistributed(Conv2D(32,(8,8),activation = 'relu',subsample = (4,4))))
    model.add(TimeDistributed(Conv2D(64,(4,4),activation = 'relu',subsample = (2,2))))
    model.add(TimeDistributed(Conv2D(64,(3,3),activation = 'relu',subsample = (1,1))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128,activation = 'relu')))
    model.add(LSTM(nb_actions,activation = 'linear'))
    print(model.summary())
    print(model.layers)
    return model
# decide your model here
def get_ntm_model():
    from keras.models import Model
    import keras.backend as K

    assert permute_layer is not None 
    num_read = 5
    num_write = 5
    mem_length = 128
    n_slots = 128
    
    model_input = Input(
        (WINDOW_LENGTH,1) + INPUT_SHAPE,
        #batch_shape = (batch_size,) + (WINDOW_LENGTH,1) + INPUT_SHAPE
    )
    
    per = permute_layer(model_input)
    
    x = TimeDistributed(Conv2D(32,(8,8),name='conv1',activation = 'relu',subsample = (4,4)))(per)
    x = TimeDistributed(Conv2D(64,(4,4),name='conv2',activation = 'relu',subsample = (2,2)))(x)
    x = TimeDistributed(Conv2D(64,(3,3),name='conv3',activation = 'relu',subsample = (1,1)))(x)
    x = TimeDistributed(Flatten(name = "Flatten1"))(x) # (batch_size,WINDOW_LENGTH,3176)
    x_shape = K.int_shape(x)
    print('x has shape:',x_shape)
    # controller construction
    controller_inp = Input((x_shape[-1],), name="controller_input") # (batch_size,3176)
    read_inp = Input((num_read,mem_length),name="read_inp") # (batch_size,n_read,n_write)
    read_inp_flatten = Flatten(name="read_inp_flatten")(read_inp) #(batch_size,n_read * n_write)
    print('controller_inp shape:',controller_inp.shape)
    #print('read_inp_flatten shape:',K.int_shape(read_inp_flatten))
    #print('read_inp_flatten_repeat shape:',K.int_shape(read_inp_flatten_repeat))
    
    concat = Concatenate(name="ctrl_inp_read_inp_concat")([controller_inp,read_inp_flatten]) # (batch_size, 3176 + num_read * mem_length)
    hidden = Dense(512,activation = 'relu')(concat)
    controller_output = Dense(nb_actions,activation = 'linear')(hidden)
    controller = Model([controller_inp,read_inp],[controller_output,hidden])
    controller.summary()
    # ntm constuction
    #TODO: reset the state for on_batch_end!!
    ntm_cell = NTM(
            controller,                     # custom controller, should output a vector
            n_slots,mem_length,           # Memory config
            num_shift = 3,                  # shifting
            batch_size = batch_size,
            #controller_instr_output_dim = controller_instr_output_dim,
            return_sequences = False,
            is_controller_recurrent = True,
            num_read = num_read,num_write = num_write)(x) # (batch_size,512)
    ntm_cell_output_shape = K.int_shape(ntm_cell)
    print('ntm_cell output:', ntm_cell_output_shape)
    ntm_cell_output_shape = ntm_cell_output_shape[1:]
    
    #model_output = Dense(nb_actions,activation = 'linear')(ntm_cell)
    
    model = Model(model_input,ntm_cell)
    model.summary()
    return model

if args.agent_type == 'conv':
    model = get_conv_model(model)
elif args.agent_type == 'rnn':
    model = get_rnn_model(model)
elif args.agent_type == 'drnn':
    model = get_double_rnn_model(model)
elif args.agent_type == 'ntm':
    model = get_ntm_model()

else:
    raise ValueError('unknown model type: {}'.format(args.agent_type))

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.02,
                              nb_steps=100000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
#policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!
custom_objects = {}
if args.agent_type == 'ntm':
    custom_objects = {'NTM': NTM}
from ntm_dqn_agent import NTMAgent
dqn = None
if not args.agent_type == 'ntm':
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=10000, gamma=.99, target_model_update=1000,
                   enable_dueling_network = False,
                   batch_size = batch_size,
                   train_interval=4, delta_clip=1.)
else:
    dqn = NTMAgent(create_model_func = get_ntm_model,
                   model = model, nb_actions = nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=10000, gamma=.99, target_model_update=1000,
                   enable_dueling_network = False,
                   batch_size = batch_size,
                   train_interval=4, delta_clip=1.)

import tensorflow as tf
def get_optimizer():
    if use_rnn:
        return Adam(1e-5,clipnorm = 1.)
    else:
        return Adam(1e-4)
dqn.compile(get_optimizer(), metrics=['mae'])
'''
if args.transfer_encoding:
    print("Transferring weights")
    if not args.weights:
        raise ValueError("If --transfer_encoding is used, weight file must be provided")
    model_file = args.model
    from keras.models import load_model
    # TODO: correct?
    old_model = load_model(weight_file)
    old_conv_layers = filter(lambda l: l.__class__.__name__ == 'Conv2D',old_model.layers)
    new_conv_layers = filter(lambda l:l.__class__.__name__ == 'Conv2D',model.layers)
    for old_l,new_l in zip(old_conv_layers,new_conv_layers):
        new_l.set_weights(old_l.get_weights())
        new_l.trainable = False # freeze the new layer
'''
if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    # this takes around 1 day
    dqn.fit(env, callbacks=callbacks, nb_steps=15000000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    print('saving weights...')
    dqn.save_weights(weights_filename, overwrite=True)
    print('saving model...')
    model.save('dqn_{}_model.h5f'.format(args.env_name))
    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
             
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=5, visualize=False)

elif args.mode == 'finetune':
    
    print("finetuning...")
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.02, value_min=.02, value_test=.02,
                              nb_steps=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=1000, gamma=.99, target_model_update=1000,
                   enable_dueling_network = False,
                   train_interval=4, delta_clip=1.)
    weight_file_name = args.weights
    model.load_weights(weight_file_name) 
    
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    
    dqn.compile(Adam(1e-5), metrics=['mae'])
    dqn.fit(env,callbacks = callbacks,nb_steps = 10000000, log_interval = 10000)
    dqn.save_weights(weights_filename,overwrite = True)
    model.save('dqn_{}_model.h5f'.format(args.env_name))
    dqn.test(env, nb_episodes=5, visualize=False)