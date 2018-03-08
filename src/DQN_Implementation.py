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

    def __init__(self, env_name, qmodel_type):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        # self.goal_state = [0.5, 0.028]
        self.env_name  = env_name
        env            = gym.make(self.env_name)
        self.state_dim = len(env.reset())
        self.num_act   = env.action_space.n
        env.close()

        self.lrate   = 1e-3
        self.dlrate  = 1e-4
        self.upd_cnt = 0

        self.qmodel_type = qmodel_type
        self.input, self.output            = self.create_model()
        self.train, self.loss, self.labels = self.create_optimizer(self.output)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        keras.backend.tensorflow_backend.set_session(self.sess)
        self.m_summ = tf.summary.merge_all()
        self.w_summ = tf.summary.FileWriter('./%s_%s/logs' % (self.env_name, self.qmodel_type), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def create_model(self):
        if self.qmodel_type == 'dqn':
            state_ph = tf.placeholder(tf.float32, shape=[None, self.state_dim])
            with tf.variable_scope("layer1"): hidden = tf.layers.dense(state_ph, 3, activation=tf.nn.tanh)
            with tf.variable_scope("layer2"): output = tf.layers.dense(hidden, self.num_act, activation=tf.nn.tanh)
            return state_ph, output

    def create_optimizer(self, output):
        label_ph = tf.placeholder(tf.float32, shape=[None, self.num_act])
        with tf.variable_scope("loss"):
            loss = tf.reduce_sum(tf.square(label_ph - output))
            tf.summary.scalar('loss', loss)

        with tf.variable_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
            train_op = optimizer.minimize(loss=loss)

        return train_op, loss, label_ph

    def save_model_weights(self):
        # Helper function to save your model / weights.
        self.saver.save(self.sess, './%s_%s/model/weights.ckpt' % (self.env_name, self.qmodel_type))

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        pass

    def get_qvals(self, state):
        return self.sess.run(self.output, feed_dict={self.input: [state.flatten()]})[0]

    def update_net(self, states, q_lbls, verb=0):
        _, loss, summ = self.sess.run([self.train, self.loss, self.m_summ], feed_dict={
            self.input: states,
            self.labels: q_lbls
        })
        if verb > 0: print("Loss: {}".format(loss))
        self.upd_cnt += 1
        self.w_summ.add_summary(summ, self.upd_cnt)

        return loss


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000, batch_size=32):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.memory_size = memory_size
        self.burn_in     = burn_in
        self.memory      = [None for i in xrange(memory_size)]
        self.next        = 0
        self.size        = 0
        self.batch_size  = batch_size


    def sample_batch(self):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        indices = np.random.choice(self.size, self.batch_size)
        result  = []
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

    def __init__(self, env_name, qmodel_type, render):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.env_name    = env_name
        self.render      = render
        self.qmodel_type = qmodel_type

        self.max_burn_in = 10000
        self.dec_eps     = 0.0000045
        self.min_eps     = 0.05

        self.q_net   = QNetwork(env_name, self.qmodel_type)
        self.rep_mem = Replay_Memory(burn_in=self.max_burn_in)


    def epsilon_greedy_policy(self, q_values, eps):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() < eps: return np.random.random_integers(0, self.q_net.num_act-1)
        else: return self.greedy_policy(q_values)

    def random_policy(self):
        return np.random.random_integers(0, self.q_net.num_act-1)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def train(self, render, verb):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        if verb > 0: print('DQN_Agent::train::start')

        self.burn_in_memory(render, verb)

        tick        = 0
        max_ep      = 3000
        max_epi_len = 10000

        if self.env_name == 'MountainCar-v0': rec_rew_thresh = -120
        if self.env_name == 'CartPole-v0':    rec_rew_thresh = 150

        rec_rec = render > 0
        rec_len = 200
        rec_int = 1000

        test_rec     = render > 0
        test_rec_int = 10000

        if self.env_name == 'MountainCar-v0': train_rew_thresh = -110
        if self.env_name == 'CartPole-v0':    train_rew_thresh = 200

        train_rec = render > 1

        back_up_int = 32
        save_int    = 5
        eps_upd_int = 200

        gamma = 1
        eps   = 0.5

        avg_rew = []

        env = gym.make(self.env_name).env
        env = gym.wrappers.Monitor(env, './%s_%s/recordings/' % (self.env_name, self.qmodel_type), force=True).env

        for ep in xrange(max_ep):
            if verb > 0: print('DQN_Agent::train::new_episode_reset')
            nstate = env.reset()

            if rec_rec and ((avg_rew and avg_rew[-1] >= rec_rew_thresh) or (ep % rec_int == 0)):
                rend_tr, record_stop = True, tick+rec_len
                if verb > 0: print('DQN_Agent::train::will_record_episode')

            else:                                                              rend_tr = False

            for _ in xrange(max_epi_len):
                if tick == record_stop:
                    rend_tr = False
                    if verb > 0: print('DQN_Agent::train::stop_recording')

                if tick % test_rec_int == 0:
                    run_test = True
                    if verb > 0: print('DQN_Agent::train::will_do_test')

                if rend_tr or train_rec: env.render()

                pstate = nstate
                action = self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), eps)
                nstate, rew, term, info = env.step(action)

                if verb > 1: print('DQN_Agent::train::ntrans::(%s, %s, %s, %s)' % (nstate, rew, term, info))

                self.rep_mem.append((pstate, action, rew, nstate, term))
                if term:
                    # if verb > 0: print('DQN_Agent::train::term_state(%s)' % nstate)
                    if self.env_name == 'MountainCar-v0': break
                    if self.env_name == 'CartPole-v0':    nstate = env.reset()

                state_batch, q_batch = [], []
                for pstate, action, rew, nstate, term in self.rep_mem.sample_batch():
                    y_n = self.q_net.get_qvals(pstate)
                    y_n[action] = rew if term else rew+gamma*np.max(self.q_net.get_qvals(nstate))
                    q_batch.append(y_n)
                    state_batch.append(pstate)

                loss = self.q_net.update_net(state_batch, q_batch)

                eps = max(self.min_eps, eps-self.dec_eps)

                tick += 1

            if verb > 0: print('DQN_Agent::train::episode_end::eps(%s)' % eps)

            if run_test:
                avg_rew.append(self.test(render=test_rec, verb=verb))
                run_test = False

            if avg_rew[-1] >= train_rew_thresh: break
            else:
                diff_rew = [avg_rew[i+1]-avg_rew[i] for i in xrange(len(avg_rew)-1)]
                if len(diff_rew) > 0 and all(d != 0 and abs(d) < 30 for d in diff_rew):
                    self.q_net.lrate -= self.q_net.dlrate
                    if verb > 0: print('DQN_Agent::train::lrate(%s)' % self.q_net.lrate)

        self.test(render=test_rec, verb=1)
        self.q_net.save_model_weights(ep)

        if verb > 0: print('DQN_Agent::train::save_model_weights(%s)' % ep)

        env.render(close=True)
        env.close()

        with open(os.path.join('./%s_%s/recordings/', 'avg_rew.np'), 'wb') as f: np.save(f, avg_rew)

        # plt.plot(range(len(avg_rew)), avg_rew)
        # plt.xlabel('epochs')
        # plt.ylabel('loss (l2 norm)')
        # plt.title('MountainCar-v0::test::avg loss')
        # plt.savefig('./recordings/MountainCar-v0_test_avg loss.png')

        if verb > 0: print('DQN_Agent::train::return')


    def test(self, render, verb):
        # Evaluate the performance of your agent over (100?) episodes, by calculating cummulative rewards for the (100?) episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.

        if verb > 0: print('DQN_Agent::test::start')

        max_ep      = 20
        max_epi_len = 10000
        avg_rew     = 0
        env         = gym.make(self.env_name)

        for _ in xrange(max_ep):
            nstate = env.reset()

            for step in xrange(max_epi_len):
                if render: env.render()

                nstate, reward, term, info = env.step(self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), self.min_eps))

                if verb > 1: print('DQN_Agent::test::ntrans::(%s, %s, %s, %s)' % (nstate, reward, term, info))

                avg_rew += reward
                if term:
                    if verb > 0: print('DQN_Agent::test::term_state(%s)step(%s)' % (nstate, step))
                    break

        avg_rew /= max_ep
        if render: env.render(close=True)
        env.close()

        if verb > 0: print('DQN_Agent::test::return::avg_rew:%s' % avg_rew)
        return avg_rew


    def burn_in_memory(self, render, verb):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        if verb > 0: print('DQN_Agent::burn_in_memory::start')
        if verb > 0: print('DQN_Agent::burn_in_memory::size(%s)' % self.rep_mem.burn_in)

        env    = gym.make(self.env_name).env
        nstate = env.reset()

        for _ in xrange(self.rep_mem.burn_in):
            if render: env.render()
            pstate = nstate
            action = self.random_policy()
            nstate, reward, term, info = env.step(action)

            if verb > 1: print('DQN_Agent::burn_in_memory::ntrans::(%s, %s, %s, %s)' % (pstate, reward, action, nstate))

            self.rep_mem.append((pstate, action, reward, nstate, term))
            if term:
                env.reset()
                # if verb > 0: print('DQN_Agent::burn_in_memory::term_state(%s)' % nstate)

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
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--model',dest='model',type=str)
    parser.add_argument('--verb',dest='verb',type=int,default=1)
    return parser.parse_args()

def main(args):

    args      = parse_arguments()
    dqn_agent = DQN_Agent(args.env, args.model, render=args.render)
    dqn_agent.train(render=args.render, verb=args.verb)

if __name__ == '__main__':
    main(sys.argv)


