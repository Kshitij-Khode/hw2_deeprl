#!/usr/bin/env python
from keras.models       import Sequential
from keras.layers.core  import Dense, Dropout, Activation
from keras              import optimizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization        import BatchNormalization

import tensorflow as tf
import numpy      as np

import gym, sys, copy, argparse, collections, time, random. os

class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.


    def __init__(self, env_name):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.env_name      = env_name
        self.state_dim     = 2
        self.num_act       = 3
        self.min_position  = -1.2
        self.max_position  = 0.6-0.1
        self.max_speed     = 0.07
        self.goal_position = 0.5
        self.pos_int       = 5
        self.vel_int       = 2

        self.lrate    = 1e-4
        self.dlrate   = 1e-5

        self.model = Sequential()
        self.model.add(Dense(30, input_shape=(self.state_dim,), init='lecun_uniform'))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(Dense(30, init='lecun_uniform'))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(Dense(30, init='lecun_uniform'))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(Dense(self.num_act))

        self.model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.lrate))
        self.model.summary()

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        self.model.save_weights('./model/%s_weights.ckpt' % suffix, overwrite=True)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights.
        pass

    def get_qvals(self, state):
        return self.model.predict(np.array([[state[0], state[1]]]))[0]

    def update_net(self, states, q_lbls):
        loss = self.model.train_on_batch(np.array(states), np.array(q_lbls))
        return loss

    def print_qmap(self):
        print('--- Q Map Start ---')
        for p in np.linspace(self.min_position, self.max_position, self.pos_int):
            for v in np.linspace(0, self.max_speed, self.vel_int):
                print('Pos: %.5s, Vel: %.5s, Qs: %s' % (p, v, self.get_qvals([p,v])))
        print('--- Q Map End ---')


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.term_mem_ln = 10
        self.memory_size = memory_size
        self.burn_in     = burn_in
        self.memory      = collections.deque(maxlen=self.memory_size)
        self.term_memory = collections.deque(maxlen=self.term_mem_ln)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        if np.random.uniform() < 0.1:
            ret_val = [self.memory[i] for i in np.random.choice(len(self.memory), batch_size-1)] + \
                      [self.term_memory[i] for i in np.random.choice(len(self.term_memory), 1)]
            random.shuffle(ret_val)
        else: ret_val = [self.memory[i] for i in np.random.choice(len(self.memory), batch_size)]
        return ret_val

    def append(self, transition):
        # Appends transition to the memory.
        if transition[4] == True:
            self.term_memory.append(transition)
        self.memory.append(transition)

    def clear(self):
        self.memory.clear()

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

        self.env_name = env_name
        self.dec_eps  = 0.0000045
        self.min_eps  = 0.05
        self.q_net1   = QNetwork(env_name)
        self.q_net2   = QNetwork(env_name)
        self.rep_mem  = Replay_Memory(burn_in=10000)

    def random_policy(self):
        return np.random.random_integers(0, self.q_net1.num_act-1)

    def epsilon_greedy_policy(self, q_values, eps):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() < eps: return np.random.random_integers(0, self.q_net1.num_act-1)
        else:                         return self.greedy_policy(q_values)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        print('Train::Start')

        tick  = 0
        eps   = 0.5
        gamma = 1

        max_brn_at = 20
        max_ep_len = 200
        max_ep     = 3000

        test_int = 10000
        save_int = 1000

        rew_rec_thresh = -160
        rew_brk_thresh = -140
        rew_max        = -200
        lr_plat_r      = 50
        avg_rew        = []
        env            = gym.make(self.env_name).env

        for _ in xrange(max_brn_at):
            if not self.burn_in_memory():
                print('Train::Bad_Burn_In::No_Term_Reached')
                self.rep_mem.clear()
            else: break

        for ep in xrange(max_ep):
            nstate = env.reset()

            self.q_net1.print_qmap()
            print('Train::Episode:(%s), LRate:%s, Eps:%s, Last Loss:%s' % (ep, self.q_net1.lrate, eps, loss if 'loss' in locals() else None))
            print('Train::Episode:(%s), LRate:%s, Eps:%s, Last Loss:%s' % (ep, self.q_net2.lrate, eps, loss if 'loss' in locals() else None))

            for _ in xrange(max_ep_len):
                pstate = nstate
                action = self.epsilon_greedy_policy(self.q_net1.get_qvals(nstate)+self.q_net2.get_qvals(nstate), eps)
                nstate, rew, term, info = env.step(action)
                self.rep_mem.append((pstate, action, rew, nstate, term))

                qchoice = np.random.uniform()
                state_batch, q_batch = [], []
                for lpst, lact, lrew, lnst, lterm in self.rep_mem.sample_batch():
                    if qchoice < 0.5:
                        y_n       = self.q_net1.get_qvals(lpst)
                        y_n[lact] = lrew if lterm else lrew+gamma*np.max(self.q_net1.get_qvals(lnst))
                    else:
                        y_n       = self.q_net2.get_qvals(lpst)
                        y_n[lact] = lrew if lterm else lrew+gamma*np.max(self.q_net2.get_qvals(lnst))
                    q_batch.append(y_n)
                    state_batch.append(lpst)

                if qchoice < 0.5:
                    loss = self.q_net1.update_net(state_batch, q_batch)
                else:
                    loss = self.q_net2.update_net(state_batch, q_batch)

                tick += 1
                eps   = max(self.min_eps, eps-self.dec_eps)

                if term:
                    print('Train::Term_state(%s)' % nstate)
                    break

            avg_rew.append(self.test())

            if (avg_rew and avg_rew[-1] >= rew_rec_thresh) or (ep % save_int == 0):
                self.q_net1.save_model_weights('%s_%s' % (ep, 1))
                self.q_net2.save_model_weights('%s_%s' % (ep, 2))

            if avg_rew[-1] >= rew_brk_thresh: break
            elif avg_rew[-1] > rew_max:
                diff_rew = [avg_rew[i+1]-avg_rew[i] for i in xrange(len(avg_rew)-1)]
                if len(diff_rew) > 0 and all(d != 0 and abs(d) < lr_plat_r for d in diff_rew):
                    self.q_net1.lrate = max(self.q_net1.dlrate, self.q_net1.lrate-self.q_net1.dlrate)
                    self.q_net2.lrate = max(self.q_net2.dlrate, self.q_net2.lrate-self.q_net2.dlrate)
                    print('Train::LRate(%s, %s)' % (self.q_net1.lrate, self.q_net2.lrate))

        self.q_net1.save_model_weights(ep)
        self.q_net2.save_model_weights(ep)
        with open(os.path.join('./model/', 'avg_rew.dat'), 'wb') as f:
            np.save(f, avg_rew)

        env.close()

        print('Train::Return')


    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        print('Test::Start')

        eps   = 0.5

        max_ep_len = 200
        max_ep     = 20

        avg_rew = 0
        env     = gym.make(self.env_name).env

        for _ in xrange(max_ep):
            nstate = env.reset()

            for _ in xrange(max_ep_len):
                action = self.epsilon_greedy_policy(self.q_net1.get_qvals(nstate)+self.q_net2.get_qvals(nstate), self.min_eps)
                nstate, rew, term, _ = env.step(action)
                avg_rew += rew

                if term: break

        avg_rew /= max_ep

        print('Test::Avg_Rew:%s' % avg_rew)

        env.close()
        return avg_rew

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of transitions.
        print('Burn_in_memory::Start')

        sterm  = False
        env    = gym.make(self.env_name).env
        nstate = env.reset()

        for _ in xrange(self.rep_mem.burn_in):
            pstate               = nstate
            action               = self.random_policy()
            nstate, rew, term, _ = env.step(action)

            self.rep_mem.append((pstate, action, rew, nstate, term))
            if term:
                sterm = True
                print('Burn_in_memory::Term_State(%s, %s, %s, %s, %s)' % (pstate, action, rew, nstate, term))
                nstate = env.reset()

        env.close()

        print('Burn_in_memory::Return')

        return sterm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    return parser.parse_args()

def main(args):

    args      = parse_arguments()
    dqn_agent = DQN_Agent('MountainCar-v0')

    dqn_agent.train()


if __name__ == '__main__':
    main(sys.argv)

