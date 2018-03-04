#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, os

class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env_name):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        self.env = gym.make(env_name)
        self.dstate = len(self.env.reset())
        self.nact = self.env.action_space.n
        self.input, self.model, self.output = self.create_model()
        self.train, self.loss, self.labels = self.create_optimizer(self.output)
        self.opt_cnt = 0
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        keras.backend.tensorflow_backend.set_session(self.sess)
        self.m_summ = tf.summary.merge_all()
        self.w_summ = tf.summary.FileWriter('./logs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def create_model(self):
        state_ph = tf.placeholder(tf.float32, shape=[None, self.dstate])
        with tf.variable_scope("layer1"): layer1 = tf.layers.dense(state_ph, self.nact)

        return state_ph, layer1, layer1

    def create_optimizer(self, output):
        label_ph = tf.placeholder(tf.int32, shape=[None])
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.l2_loss(output))
            tf.summary.scalar('loss', loss)

        with tf.variable_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            train_op = optimizer.minimize(loss=loss)

        return train_op, loss, label_ph

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        pass

    def get_qvals(self, state):
        return self.sess.run(self.model, feed_dict={self.input: [state.flatten()]})[0]

    def update_net(self, states, q_lbls, verb=True):
        _, loss, summ = self.sess.run([self.train, self.loss, self.m_summ], feed_dict={
            self.input: states,
            self.labels: q_lbls
        })
        if verb: print("Loss: {}".format(loss))
        self.opt_cnt += 1
        self.w_summ.add_summary(summ, self.opt_cnt)


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000, batch_size=32):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = [None for i in xrange(memory_size)]
        self.next = 0
        self.size = 0
        self.batch_size = batch_size


    def sample_batch(self):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        indices = np.random.choice(self.size, self.batch_size)
        result = []
        for ind in indices:
            result += [self.memory[ind]]
        return result

    def append(self, transition):
        # Appends transition to the memory.
        self.memory[self.next] = transition
        self.next += 1
        if self.size <= self.memory:
            self.size += 1
        if self.next >= self.memory:
            self.next = 0

class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env_name, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.max_epochs = 1000
        self.max_episode_len = 1000
        self.c_eps = 0.3
        self.gamma = 0.9
        self.q_net = QNetwork(env_name)
        self.rep_mem = Replay_Memory()
        self.render = render

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() < self.c_eps: return np.random.random_integers(0, self.q_net.nact-1)
        else: return self.greedy_policy(q_values)

    def random_policy(self):
        return np.random.random_integers(0, self.q_net.nact-1)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        self.burn_in_memory()
        nstate = self.q_net.env.reset()

        for ep in xrange(self.max_epochs):
            for step in xrange(self.max_episode_len):
                if self.render: self.q_net.env.render()

                nstate, reward, term, info = self.q_net.env.step(self.epsilon_greedy_policy(self.q_net.get_qvals(nstate)))
                self.rep_mem.append((nstate, reward, term, info))

                state_batch = []
                q_batch = []
                for state, reward, term, info in self.rep_mem.sample_batch():
                    state_batch.append(state)
                    q_batch.append(reward if term else reward+np.max(self.q_net.get_qvals(nstate)))

                self.q_net.update_net(state_batch, q_batch)
            self.c_eps = max(0.05, self.c_eps*0.9)
            print('[LOG] max_episode_len reached. new epsilon value:%s' % self.c_eps)

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        pass

    def burn_in_memory(self, verb=True):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        if verb: print('[LOG] starting burn_in_memory')
        self.q_net.env.reset()
        for _ in xrange(self.rep_mem.burn_in):
            for _ in xrange(self.max_episode_len):
                nstate, reward, term, info = self.q_net.env.step(self.random_policy())
                self.rep_mem.append((nstate, reward, term, info))
                if term: break
        if verb: print('[LOG] burn_in_memory ended')

    def burn_in_expert_memory(self, bc_data_dir):
        states, actions = [], []
        shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
        for shard in shards:
            shard_path = os.path.join(bc_data_dir, shard)
            with open(shard_path, 'rb') as f:
                data = np.load(f, encoding='bytes')
                shard_states, unprocessed_actions = zip(*data)
                shard_states = [x.flatten() for x in shard_states]
                states.extend(shard_states)
                actions.extend(unprocessed_actions)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)/2
        return states, actions



def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    env_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    dqn_agent = DQN_Agent('MountainCar-v0', render=True)
    dqn_agent.train()

if __name__ == '__main__':
    main(sys.argv)

