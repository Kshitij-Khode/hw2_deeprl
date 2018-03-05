#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, os, time

class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env_name):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        self.goal_state = [0.5, 0.028]
        self.env = gym.make(env_name)
        self.dstate = len(self.env.reset())
        self.nact = self.env.action_space.n

        self.lrate = 1e-3;
        self.opt_cnt = 0
        self.input, self.model, self.output = self.create_model()
        self.train, self.loss, self.labels = self.create_optimizer(self.output)
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
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
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

    def update_net(self, states, q_lbls, verb=0):
        _, loss, summ = self.sess.run([self.train, self.loss, self.m_summ], feed_dict={
            self.input: states,
            self.labels: q_lbls
        })
        if verb: print("Loss: {}".format(loss))
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

    def epsilon_greedy_policy(self, q_values, eps):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() < eps: return np.random.random_integers(0, self.q_net.nact-1)
        else: return self.greedy_policy(q_values)

    def random_policy(self):
        return np.random.random_integers(0, self.q_net.nact-1)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def train(self, render=False):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        print('[LOG] DQN_Agent::train::start')

        max_ep = 3000
        max_epi_len = 10000
        eps_upd_int = 200
        gamma = 1
        eps = 0.5
        nq_upd = 0

        self.burn_in_memory(verb=1)
        nstate = self.q_net.env.reset()

        for _ in xrange(max_ep):
            nstate = self.q_net.env.reset()

            for _ in xrange(max_epi_len):
                if render: self.q_net.env.render()

                nstate, reward, term, info = self.q_net.env.step(self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), eps))
                self.rep_mem.append((nstate, reward, term, info))

                state_batch, q_batch = [], []
                for state, reward, term, info in self.rep_mem.sample_batch():
                    state_batch.append(state)
                    q_batch.append(reward if term else reward+gamma*np.max(self.q_net.get_qvals(nstate)))

                loss = self.q_net.update_net(state_batch, q_batch)
                nq_upd += 1
                if nq_upd % eps_upd_int == 0: eps = max(self.meps, eps-self.deps)
                if nstate[0] > self.q_net.goal_state[0] and nstate[1] < self.q_net.goal_state[1]:
                    print('[LOG] DQN_Agent::train::term_state(%s)' % nstate)
                    break

            print('[LOG] DQN_Agent::train::eps(%s),loss(%s),nq_upd(%s)' % (eps, loss, nq_upd))

            self.test(render=True)

        print('[LOG] DQN_Agent::train::return')


    def test(self, model_file=None, render=False):
        # Evaluate the performance of your agent over (100?) episodes, by calculating cummulative rewards for the (100?) episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.

        max_ep = 20
        max_epi_len = 200
        eps = 0.5
        avg_rew = 0

        for _ in xrange(max_ep):
            nstate = self.q_net.env.reset()

            for _ in xrange(max_epi_len):
                if render: self.q_net.env.render()

                nstate, reward, term, info = self.q_net.env.step(self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), self.meps))
                avg_rew += reward
                if nstate[0] > self.q_net.goal_state[0] and nstate[1] < self.q_net.goal_state[1]:
                    print('[LOG] DQN_Agent::test::term_state(%s)' % nstate)
                    break

        avg_rew /= max_ep
        print('[LOG] DQN_Agent::test::avg_rew:%s' % avg_rew)


    def burn_in_memory(self, render=False, verb=0):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        if verb > 0: print('[LOG] DQN_Agent::burn_in_memory::start')

        for _ in xrange(self.rep_mem.burn_in):
            if render: self.q_net.env.render()

            nstate, reward, term, info = self.q_net.env.step(self.random_policy())
            self.rep_mem.append((nstate, reward, term, info))
            if nstate[0] > self.q_net.goal_state[0] and nstate[1] < self.q_net.goal_state[1]:
                self.q_net.env.reset()
                print('[LOG] DQN_Agent::burn_in_memory::term_state(%s)' % nstate)

        if verb > 0: print('[LOG] DQN_Agent::burn_in_memory::return')

    def burn_in_expert(self, data_dir, verb=0):
        if verb > 0: print('[LOG] DQN_Agent::burn_in_expert::start')

        transitions = []
        shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
        print("Processing shards: {}".format(shards))
        for shard in shards:
            shard_path = os.path.join(bc_data_dir, shard)
            with open(shard_path, 'rb') as f:
                data = np.load(f)
                for d in data:
                    self.rep_mem.append(d)

        if verb > 0: print('[LOG] DQN_Agent::burn_in_expert::return')

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
    dqn_agent = DQN_Agent('MountainCar-v0')
    dqn_agent.train()

if __name__ == '__main__':
    main(sys.argv)

