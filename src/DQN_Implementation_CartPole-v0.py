#!/usr/bin/env python

# import matplotlib.pyplot as plt
import keras, tensorflow as tf
import numpy             as np
import collections       as col

import gym, sys, copy, argparse
import os, time

class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env_name):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        # self.goal_state = [0.5, 0.028]
        self.env_name = env_name
        env = gym.make(self.env_name)
        self.dstate = len(env.reset())
        self.nact = env.action_space.n
        env.close()
        self.lrate = 1e-3
        self.dlrate = 1e-4
        self.opt_cnt = 0
        self.input, self.model, self.output = self.create_model()
        self.train, self.loss, self.labels = self.create_optimizer(self.output)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        keras.backend.tensorflow_backend.set_session(self.sess)
        self.m_summ = tf.summary.merge_all()
        self.w_summ = tf.summary.FileWriter('./logs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def create_model(self):
        state_ph = tf.placeholder(tf.float32, shape=[None, self.dstate])
        with tf.variable_scope("layer1"): layer1 = tf.layers.dense(state_ph, self.nact)

        return state_ph, layer1, layer1

    def create_optimizer(self, output):
        label_ph = tf.placeholder(tf.float32, shape=[None])
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.l2_loss(output))
            tf.summary.scalar('loss', loss)

        with tf.variable_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
            train_op = optimizer.minimize(loss=loss)

        return train_op, loss, label_ph

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        self.saver.save(self.sess, '%s%s%s%s' % ('./model/', self.env_name, suffix, '.ckpt'))

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        pass

    def get_qvals(self, state):
        return self.sess.run(self.model, feed_dict={self.input: [state.flatten()]})[0]

    def update_net(self, states, q_lbls, verb=0):
        _, loss, summ = self.sess.run([self.train, self.loss, self.m_summ], feed_dict={
            self.input: states,
            self.labels: q_lbls
        })
        if verb > 0: print("Loss: {}".format(loss))
        self.opt_cnt += 1
        self.w_summ.add_summary(summ, self.opt_cnt)

        return loss


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
        for ind in indices: result += [self.memory[ind]]
        return result

    def append(self, transition):
        # Appends transition to the memory.
        self.memory[self.next] = transition
        self.next = (self.next+1) % self.memory_size
        if self.size < self.memory_size: self.size += 1

class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env_name):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.deps = 9e-4
        self.meps = 0.05
        self.q_net = QNetwork(env_name)
        self.rep_mem = Replay_Memory(burn_in=5000)
        self.env_name = env_name

    def epsilon_greedy_policy(self, q_values, eps):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() < eps: return np.random.random_integers(0, self.q_net.nact-1)
        else: return self.greedy_policy(q_values)

    def random_policy(self):
        return np.random.random_integers(0, self.q_net.nact-1)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def train(self, render=False, verb=0):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        if verb > 0: print('DQN_Agent::train::start')
        self.burn_in_memory(render=False, verb=1)

        max_ep = 10
        max_epi_len = 100000
        rec_len = 200

        rec_int = 1
        test_rend_int = 1

        test_rend = False
        rec_stop = 0
        record = False
        save_int = 1
        eps_upd_int = 200
        gamma = 1
        eps = 0.5
        nq_upd = 0
        ars = []
        env = gym.make(self.env_name)
        # env = gym.wrappers.Monitor(env, './recordings/', force=True)

        for ep in xrange(max_ep):
            nstate = env.reset()

            # if (ars and ars[-1] >= -120) or (ep % rec_int == 0):
            #     record = True
            #     record_stop = nq_upd+rec_len

            if (ars and ars[-1] >= -120) or (ep % rec_int == 0):
                record = False
                record_stop = nq_upd+rec_len

            # if ep % test_rend_int == 0: test_rend = True
            # else: test_rend = False

            if ep % test_rend_int == 0: test_rend = False
            else: test_rend = False

            for _ in xrange(max_epi_len):
                if nq_upd == record_stop: record = False
                if render or record: env.render()

                nstate, rew, term, info = env.step(self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), eps))
                if verb > 1: print('DQN_Agent::train::ntrans::(%s, %s, %s, %s)' % (nstate, rew, term, info))
                self.rep_mem.append((nstate, rew, term, info))
                if term:
                    if verb > 1: print('DQN_Agent::train::term_state(%s)' % nstate)
                    nstate = env.reset()

                state_batch, q_batch = [], []
                for lstate, lrew, lterm, _ in self.rep_mem.sample_batch():
                    state_batch.append(lstate)
                    q_batch.append(lrew if lterm else lrew+gamma*np.max(self.q_net.get_qvals(lstate)))

                loss = self.q_net.update_net(state_batch, q_batch)
                nq_upd += 1
                if nq_upd % eps_upd_int == 0:
                    eps = max(self.meps, eps-self.deps)
                    if verb > 0: print('DQN_Agent::train::eps_upd(%s)' % eps)

            if ep % save_int == 0:
                self.q_net.save_model_weights(ep)
                if verb > 0: print('DQN_Agent::train::save_model_weights(%s)' % ep)
            if verb > 0: print('DQN_Agent::train::eps(%s),last_loss(%s),nq_upd(%s)' % (eps, loss, nq_upd))

            ars.append(self.test(render=test_rend, verb=1))
            if ars[-1] <= 200:
                ds = [ars[i+1]-ars[i] for i in xrange(len(ars)-1)]
                if len(ds) > 0 and all(d != 0 and abs(d) < 30 for d in ds):
                    self.q_net.lrate -= self.q_net.dlrate
                    if verb > 0: print('DQN_Agent::train::lrate(%s)' % self.q_net.lrate)
            else: break

        env.render(close=True)
        env.close()

        with open(os.path.join('./recordings/', 'ars.txt'), 'wb') as f:
            np.save(f, ars)

        # plt.plot(range(len(ars)), ars)
        # plt.xlabel('epochs')
        # plt.ylabel('loss (l2 norm)')
        # plt.title('MountainCar-v0::test::avg loss')
        # plt.savefig('./recordings/MountainCar-v0_test_avg loss.png')

        if verb > 0: print('DQN_Agent::train::return')


    def test(self, model_file=None, render=False, verb=0):
        # Evaluate the performance of your agent over (100?) episodes, by calculating cummulative rewards for the (100?) episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.

        if verb > 0: print('DQN_Agent::test::start')

        max_ep = 20
        max_epi_len = 200
        eps = 0.5
        avg_rew = 0
        env = gym.make(self.env_name)

        for _ in xrange(max_ep):
            nstate = env.reset()

            for _ in xrange(max_epi_len):
                if render: env.render()

                nstate, reward, term, info = env.step(self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), self.meps))
                if verb > 1: print('DQN_Agent::test::ntrans::(%s, %s, %s, %s)' % (nstate, reward, term, info))
                avg_rew += reward
                if term:
                    if verb > 0: print('DQN_Agent::test::term_state(%s)' % nstate)
                    break

        avg_rew /= max_ep
        if render: env.render(close=True)
        env.close()
        if verb > 0: print('DQN_Agent::test::return::avg_rew:%s' % avg_rew)
        return avg_rew


    def burn_in_memory(self, render=False, verb=0):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        if verb > 0: print('DQN_Agent::burn_in_memory::start')

        env = gym.make(self.env_name)
        nstate = env.reset()

        for _ in xrange(self.rep_mem.burn_in):
            if render: env.render()
            nstate, reward, term, info = env.step(self.random_policy())
            if verb > 1: print('DQN_Agent::burn_in_memory::ntrans::(%s, %s, %s, %s)' % (nstate, reward, term, info))
            self.rep_mem.append((nstate, reward, term, info))
            if term:
                env.reset()
                if verb > 1: print('DQN_Agent::burn_in_memory::term_state(%s)' % nstate)

        if render: env.render(close=True)
        env.close()
        if verb > 0: print('DQN_Agent::burn_in_memory::return')

    def burn_in_expert(self, data_dir, verb=0):
        if verb > 0: print('DQN_Agent::burn_in_expert::start')

        transitions = []
        shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
        if verb > 0: print("Processing shards: {}".format(shards))
        for shard in shards:
            shard_path = os.path.join(bc_data_dir, shard)
            with open(shard_path, 'rb') as f:
                data = np.load(f)
                for d in data:
                    self.rep_mem.append(d)
                    if verb > 1: print('DQN_Agent::burn_in_expert::ntrans::%s' % d)

        if verb > 0: print('DQN_Agent::burn_in_expert::return')

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
    dqn_agent = DQN_Agent('CartPole-v0')
    dqn_agent.train(verb=1)

if __name__ == '__main__':
    main(sys.argv)
