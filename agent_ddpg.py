from gym_torcs import TorcsEnv
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



#####################  hyper parameters  ####################
N_STATES = 29
N_ACTIONS = 3
BATCH_SIZE = 32
MEMORY_CAPACITY = 100000
GAMMA_REWARD = 0.99     # reward discount
TAU = 0.001      # soft replacement
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic

MAX_EPISODE = 200000
MAX_STEPS = 20000
EPISODE_COUNT = 0
GLOBAL_STEP = 0

vision = False
done = False
restor = True
is_train = False
MODEL_PATH = ''
RESTORE_MODEL_PATH = '/home/azha/Torcs/azha_torcs/ckpt/model_ddpg_2_1318_185000.pth'
np.random.seed(1337)
###############################  DDPG  ####################################


class OU(object):

    def function(self, x, theta, mu, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)[0]#randn函数返回一个或一组样本，具有标准正态分布。


class ANet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 300)
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化
        self.fc2 = nn.Linear(300, 600)
        self.fc2.weight.data.normal_(0, 0.1)  # 初始化
        self.Steering = nn.Linear(600, 1)
        self.Steering.weight.data.normal_(0, 0.1)  # 初始化
        self.Acceleration = nn.Linear(600, 1)
        self.Acceleration.weight.data.normal_(0, 0.1)  # 初始化
        self.Brake = nn.Linear(600, 1)
        self.Brake.weight.data.normal_(0, 0.1)  # 初始化

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        steer = self.Steering(x)
        steer = torch.tanh(steer)
        accel = self.Acceleration(x)
        accel = torch.sigmoid(accel)
        brake = self.Brake(x)
        brake = torch.sigmoid(brake)

        actions_value = torch.cat([steer, accel, brake], dim=-1)
        return actions_value

class CNet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(CNet,self).__init__()
        self.fcs1 = nn.Linear(N_STATES,300)
        self.fcs1.weight.data.normal_(0,0.1) # initialization
        self.fcs2 = nn.Linear(300,600)
        self.fcs2.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(N_ACTIONS,600)
        self.fca.weight.data.normal_(0,0.1) # initialization

        self.h1 = nn.Linear(600,600)
        self.h1.weight.data.normal_(0,0.1) # initialization

        self.out = nn.Linear(600,N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization


    def forward(self,s,a):
        x = self.fcs1(s)
        x = F.relu(x)
        x = self.fcs2(x)
        y = self.fca(a)

        net = x + y
        net = self.h1(net)
        net = F.relu(net)

        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self):

        self.Actor_eval = ANet()
        self.Actor_target = ANet()
        self.Critic_eval = CNet()
        self.Critic_target = CNet()
        self.Actor_eval = self.Actor_eval.cuda()
        self.Actor_target = self.Actor_target.cuda()
        self.Critic_eval = self.Critic_eval.cuda()
        self.Critic_target = self.Critic_target.cuda()

        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS + 2), dtype=np.float32)
        self.memory_counter = 0  # for storing memory
        #随机
        self.epsilon = 1.0  # greedy policy
        self.epsilon_increment = 50000.0
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.cost_his_a = []
        self.cost_his_c = []
        self.ou = OU()

    def choose_action(self, x):

        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        x = x.cuda()

        action_value = self.Actor_eval.forward(x)
        action_value = action_value.cpu()

        action_value = action_value.data.numpy()
        # action_value = np.array(action_value)
        # print("action_value", action_value.shape)#[1,3]

        action = np.zeros([1, N_ACTIONS], dtype=np.float32)
        noise_t = np.zeros([1, N_ACTIONS], dtype=np.float32)

        if np.random.uniform() < 0.5:
            noise_t[0][0] = max(self.epsilon, 0) * self.ou.function(action_value[0][0], 0.60, 0.0, 0.30)
            noise_t[0][1] = max(self.epsilon, 0) * self.ou.function(action_value[0][1], 1.00, 0.5, 0.10)
            noise_t[0][2] = max(self.epsilon, 0) * self.ou.function(action_value[0][2], 1.00, 0.1, 0.05)

        action[0][0] = action_value[0][0] + noise_t[0][0]
        action[0][1] = action_value[0][1] + noise_t[0][1]
        action[0][2] = action_value[0][2] + noise_t[0][2]

        if np.random.uniform() < 0.1 and self.epsilon > 0.0:
            action[0][1] = 0.0
            action[0][2] = 0.1
        elif self.epsilon > 0.0:
            action[0][2] = 0.0

        return action

    def store_transaction(self, s, a, r, s_, done):
        transaction = np.hstack((s, a, [r], s_, [done]))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transaction
        self.memory_counter += 1

    def learn(self):

        # for x in self.Actor_target.state_dict().keys():
        #     eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        # for x in self.Critic_target.state_dict().keys():
        #     eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        tmp_weight = self.Actor_eval.state_dict()
        for k1, k2 in zip(self.Actor_eval.state_dict(), self.Actor_target.state_dict()):
            tmp_weight[k1].data = TAU * self.Actor_eval.state_dict()[k1].data + (1 - TAU) * self.Actor_target.state_dict()[k2].data
        self.Actor_target.load_state_dict(tmp_weight)

        tmp_weight = self.Critic_eval.state_dict()
        for k1, k2 in zip(self.Critic_eval.state_dict(), self.Critic_target.state_dict()):
            tmp_weight[k1].data = TAU * self.Critic_eval.state_dict()[k1].data + (1 - TAU) * self.Critic_target.state_dict()[k2].data
        self.Critic_target.load_state_dict(tmp_weight)

        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, size=BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.FloatTensor(b_memory[:, N_STATES: N_STATES + N_ACTIONS]))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS: N_STATES + N_ACTIONS + 1]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS + 1: 2*N_STATES + N_ACTIONS +1]))
        b_done = Variable(torch.FloatTensor(b_memory[:, -1:]))

        b_s = b_s.cuda()
        b_a = b_a.cuda()
        b_r = b_r.cuda()
        b_s_ = b_s_.cuda()
        b_done = b_done.cuda()


        a = self.Actor_eval(b_s)
        q = self.Critic_eval(b_s, a)
        loss_a = -torch.mean(q)
        self.cost_his_a.append(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        q_v = self.Critic_eval(b_s, b_a)
        a_ = self.Actor_target(b_s_)
        q_target = self.Critic_target(b_s_, a_).detach()
        for k in range(BATCH_SIZE):
            if b_done[k]:
                q_target[k] = b_r[k]
            else:
                q_target[k] = b_r[k] + GAMMA_REWARD * q_target[k]

        loss_c = self.loss_td(q_v, q_target)
        self.cost_his_c.append(loss_c)
        self.ctrain.zero_grad()
        loss_c.backward()
        self.ctrain.step()


        # 逐渐增加 epsilon, 降低行为的随机性
        if self.epsilon > 0:
            self.epsilon = self.epsilon - 1.0 / self.epsilon_increment


def handle_ob(ob):
    # print(ob)
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
    agent = DDPG()


    if restor:
        state = torch.load(RESTORE_MODEL_PATH)
        # EPISODE_COUNT = state['EPISODE_COUNT']
        # GLOBAL_STEP = state['GLOBAL_STEP']
        agent.Actor_eval.load_state_dict(state['state_actor_eval'])
        agent.Actor_target.load_state_dict(state['state_actor_target'])
        agent.Critic_eval.load_state_dict(state['state_critic_eval'])
        agent.Critic_target.load_state_dict(state['state_critic_target'])
        agent.atrain.load_state_dict(state['optimizer_atrain'])
        agent.ctrain.load_state_dict(state['optimizer_ctrain'])
        # agent.memory = state['memory']
        # agent.memory_counter = state['memory_counter']
        # agent.epsilon = state['epsilon']
        # agent.cost_his_a = state['cost_his_a']
        # agent.cost_his_c = state['cost_his_c']


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

                action = np.squeeze(action)

                # print("action: ", action.shape)#[3,]

                ob_, reward, done, _ = env.step(action)

                ob_net_ = handle_ob(ob_)

                # DDPG 存储记忆
                agent.store_transaction(ob_net, action, reward, ob_net_, done)

                # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
                if GLOBAL_STEP > 500 :
                    agent.learn()

                # 将下一个 state_ 变为 下次循环的 state
                ob_net = ob_net_

                total_reward += reward

                GLOBAL_STEP += 1

                if GLOBAL_STEP > 50000 and GLOBAL_STEP % 5000 == 0:
                    MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/model_ddpg_2_' + str(EPISODE_COUNT) + '_' + str(GLOBAL_STEP) + '.pth'
                    state = {
                        'EPISODE_COUNT': EPISODE_COUNT,
                        'GLOBAL_STEP': GLOBAL_STEP,
                        'state_actor_eval': agent.Actor_eval.state_dict(),
                        'state_actor_target': agent.Actor_target.state_dict(),
                        'state_critic_eval': agent.Critic_eval.state_dict(),
                        'state_critic_target': agent.Critic_target.state_dict(),
                        'optimizer_atrain': agent.atrain.state_dict(),
                        'optimizer_ctrain': agent.ctrain.state_dict(),
                        'memory': agent.memory,
                        'memory_counter': agent.memory_counter,
                        'epsilon': agent.epsilon,
                        'cost_his_a': agent.cost_his_c,
                        'cost_his_c': agent.cost_his_c,
                    }
                    torch.save(state, MODEL_PATH)


                if done:
                    break

            EPISODE_COUNT = EPISODE_COUNT + 1
            print(str(EPISODE_COUNT) +" -th Episode  ---------------  " + "TOTAL REWARD : " + str(total_reward))
            if len(agent.cost_his_a) > 0:
                print("loss_a: " + str(agent.cost_his_a[len(agent.cost_his_a)-1]))
                print("loss_c: " + str(agent.cost_his_c[len(agent.cost_his_c) - 1]))
            print("epsilon: ", agent.epsilon)
            print("Total Step: " + str(GLOBAL_STEP))
            print("")

    else:

        agent.epsilon = 0.0
        print("TORCS Experiment Start.")
        for i in range(MAX_EPISODE):


            # speedx = []
            # speedy = []
            # re = []


            print("Episode : " + str(EPISODE_COUNT))

            if np.mod(i, 100) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            ob_net = handle_ob(ob)

            for j in range(MAX_STEPS):

                action = agent.choose_action(ob_net)

                action = np.squeeze(action)

                ob_, reward, done, _ = env.step(action)

                ob_net_ = handle_ob(ob_)

                # speedx.append(ob_net_[2] * 300.0)
                # speedy.append(ob_net_[3] * 300.0)
                # re.append(reward)

                # 将下一个 state_ 变为 下次循环的 state
                ob_net = ob_net_

                GLOBAL_STEP += 1

                if done:
                    break

            # MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/test_ddpg_road.pth'
            # state = {
            #     'speedx': speedx,
            #     'speedy': speedy,
            #     're': re,
            # }
            # torch.save(state, MODEL_PATH)
            # print("模型已保存!")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

#调节OU
#调节奖励函数
#增加随机刹车
#Episode:1168
#Total Step: 175019

# 1112 -th Episode  ---------------  TOTAL REWARD : 231635.11365194156
# loss_a: tensor(-9474.9053, device='cuda:0', grad_fn=<NegBackward>)
# loss_c: tensor(155367.8906, device='cuda:0', grad_fn=<MseLossBackward>)
# epsilon:  -9.999998083753952e-06
# Total Step: 167161

# Total Step: 336713
#
# Total Step: 332308
#
# Total Step: 314820
#
# Total Step: 313084
#
# Total Step: 311841
#
# Total Step: 310855
#
# Total Step: 309879
#
# Total Step: 304156
#
# Total Step: 296640

#180000