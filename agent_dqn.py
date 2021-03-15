from gym_torcs import TorcsEnv
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 超参数
N_STATES = 29
N_ACTIONS = 15
BATCH_SIZE = 32
MEMORY_CAPACITY = 100000
GAMMA_REWARD = 0.99  # reward discount
TAU = 0.001
LR = 0.01  # learning rate

MAX_EPISODE = 200000
MAX_STEPS = 100000
EPISODE_COUNT = 0
GLOBAL_STEP = 0

vision = False
done = False
restor = True
is_train = False
MODEL_PATH = ''
RESTORE_MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/model_dqn_3_1389_150000.pth'
np.random.seed(1337)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 300)
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化
        self.fc2 = nn.Linear(300, 600)
        self.fc2.weight.data.normal_(0, 0.1)  # 初始化
        self.out = nn.Linear(600, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # 初始化

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net = self.eval_net.cuda()
        self.target_net = self.target_net.cuda()
        # 记录学习到多少步
        self.learn_step_counter = 0  # for target update
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))
        self.memory_counter = 0  # for storing memory
        self.epsilon = 0.0  # greedy policy
        self.epsilon_increment = 50000
        # 初始化memory

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.cost_his = []

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        x = x.cuda()
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(x)
            action_value = action_value.cpu()
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:  # random
            if np.random.uniform() < 0.2:
                action = np.random.randint(0, 5) + 10
            else:
                action = np.random.randint(0, 10)
        return action

    # s:当前状态， a:动作, r:reward奖励, s_:下一步状态
    def store_transaction(self, s, a, r, s_, done):
        transaction = np.hstack((s, [a, r], s_, [done]))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transaction
        self.memory_counter += 1

    def learn(self):
        # target net update
        tmp_weight = self.eval_net.state_dict()
        for k1, k2 in zip(self.eval_net.state_dict(), self.target_net.state_dict()):
            tmp_weight[k1].data = TAU * self.eval_net.state_dict()[k1].data + (1 - TAU) * self.target_net.state_dict()[k2].data
        self.target_net.load_state_dict(tmp_weight)

        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, size=BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES: N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1: N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, N_STATES + 2: 2*N_STATES + 2]))
        b_done = Variable(torch.FloatTensor(b_memory[:, -1:]))

        b_s = b_s.cuda()
        b_a = b_a.cuda()
        b_r = b_r.cuda()
        b_s_ = b_s_.cuda()
        b_done = b_done.cuda()

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = GAMMA_REWARD * q_next.max(1)[0].view(BATCH_SIZE, 1)
        for k in range(BATCH_SIZE):
            if b_done[k]:
                q_target[k] = b_r[k]
            else:
                q_target[k] = b_r[k] + q_target[k]

        loss = self.loss_func(q_eval, q_target)
        self.cost_his.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 逐渐增加 epsilon, 降低行为的随机性
        if self.epsilon < 1.0:
            self.epsilon = self.epsilon + 1.0 / self.epsilon_increment
        self.learn_step_counter += 1



def handle_ob(ob):
    ob_net = np.zeros(1)
    for ob_tmp in ob:
        if len(ob_tmp.shape) == 0:
            tmp = np.zeros(1)
            tmp[0] = ob_tmp
            ob_net = np.append(ob_net, tmp, axis=0)
        else:
            ob_net = np.append(ob_net, ob_tmp, axis=0)
    ob_net = np.delete(ob_net, 0, axis=0)
    return ob_net


if __name__ == "__main__":



    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=False)
    agent = DQN()


    if restor:
        state = torch.load(RESTORE_MODEL_PATH)
        # EPISODE_COUNT = state['EPISODE_COUNT']
        # GLOBAL_STEP = state['GLOBAL_STEP']
        agent.eval_net.load_state_dict(state['state_dict_eval'])
        agent.target_net.load_state_dict(state['state_dict_target'])
        agent.optimizer.load_state_dict(state['optimizer'])
        # agent.learn_step_counter = state['learn_step_counter']
        # agent.memory_counter = state['memory_counter']
        # agent.epsilon = state['epsilon']
        # agent.epsilon_increment = state['epsilon_increment']
        # agent.memory = state['memory']
        # agent.cost_his = state['cost_his']


    if is_train:
        print("TORCS Experiment Start.")
        for i in range(MAX_EPISODE):
            print("Episode : " + str(EPISODE_COUNT))

            if np.mod(i, 100) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()


            ob_net = handle_ob(ob)

            total_reward = 0.
            for j in range(MAX_STEPS):

                action = agent.choose_action(ob_net)

                ob_, reward, done, _ = env.step(action)

                ob_net_ = handle_ob(ob_)

                # DQN 存储记忆
                agent.store_transaction(ob_net, action, reward, ob_net_, done)

                # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
                if GLOBAL_STEP > 500 :
                    agent.learn()

                # 将下一个 state_ 变为 下次循环的 state
                ob_net = ob_net_

                total_reward += reward

                GLOBAL_STEP += 1

                if GLOBAL_STEP >= 50000 and GLOBAL_STEP % 5000 == 0:
                    MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/model_dqn_3_' + str(EPISODE_COUNT) + '_' + str(GLOBAL_STEP) + '.pth'
                    state = {
                        'EPISODE_COUNT': EPISODE_COUNT,
                        'GLOBAL_STEP': GLOBAL_STEP,
                        'learn_step_counter': agent.learn_step_counter,
                        'memory_counter': agent.memory_counter,
                        'epsilon': agent.epsilon,
                        'epsilon_increment': agent.epsilon_increment,
                        'memory': agent.memory,
                        'state_dict_eval': agent.eval_net.state_dict(),
                        'state_dict_target': agent.target_net.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'cost_his': agent.cost_his,
                    }
                    torch.save(state, MODEL_PATH)


                if done:
                    break

            EPISODE_COUNT = EPISODE_COUNT + 1
            print(str(EPISODE_COUNT) +" -th Episode  ---------------  " + "TOTAL REWARD : " + str(total_reward))
            if len(agent.cost_his) > 0:
                print("loss: " + str(agent.cost_his[len(agent.cost_his)-1]))
            print("epsilon: ", agent.epsilon)
            print("Total Step: " + str(GLOBAL_STEP))
            print("")

    else:

        agent.epsilon = 1.0
        print(EPISODE_COUNT, GLOBAL_STEP)
        print("TORCS Experiment Start.")
        for i in range(MAX_EPISODE):

            speedx = []
            speedy = []
            re = []

            print("Episode : " + str(EPISODE_COUNT))

            if np.mod(i, 100) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            ob_net = handle_ob(ob)

            for j in range(MAX_STEPS):


                action = agent.choose_action(ob_net)

                ob_, reward, done, _ = env.step(action)

                ob_net_ = handle_ob(ob_)

                speedx.append(ob_net_[2] * 300.0)
                speedy.append(ob_net_[3] * 300.0)
                re.append(reward)

                # 将下一个 state_ 变为 下次循环的 state
                ob_net = ob_net_

                GLOBAL_STEP += 1

                if done:
                    break

            MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/test_dqn_track-5.pth'
            state = {
                'speedx': speedx,
                'speedy': speedy,
                're': re,
            }
            torch.save(state, MODEL_PATH)
            print("模型已保存!")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


#85000_7