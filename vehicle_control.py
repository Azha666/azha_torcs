from gym_torcs import TorcsEnv
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import socket, traceback, subprocess
import threading
import time

#####################  hyper parameters 超参数  ####################
N_STATES = 50  # state的维度
N_ACTIONS = 3  # action的维度
BATCH_SIZE = 64  # 每次从经验池中随机抽取N=64条Transition
MEMORY_CAPACITY = 200000  # 经验池 (si,ai,ri,si+1)
GAMMA_REWARD = 0.99  # reward discount 奖励衰减系数 值越大越考虑长期奖励
TAU = 0.001  # soft replacement 软更新系数
LR_A = 0.0001  # learning rate for actor Actor网络的学习率 值越大新Q值越重要
LR_C = 0.001  # learning rate for critic Critic网络的学习率 值越大新Q值越重要

MAX_EPISODE = 2000000  # 最大回合数 200w
MAX_STEPS = 200000  # 每回合最大步数 20w
EPISODE_COUNT = 0  # 回合计数
GLOBAL_STEP = 0  # 全局步数

vision = False  # 返回图像标识 False不返回图像
done = False  # 完成标识
restor = True  # 恢复模型标识
is_train = False  # 训练标识
MODEL_PATH = '/home/azha/Torcs/azha_torcs/ckpt/plain_torcs_ddpg_'  # 模型存储路径

RESTORE_MODEL_PATH = '/home/azha/Torcs/azha_torcs/model_train/plain_torcs_ddpg_6474_1350000.pth'  # 模型加载路径
np.random.seed(1337)  # 随机数生成
###############################  DDPG  ####################################


######################### 条件指令 ########################################

lateral_offset = 0.0  # 侧偏移量

offset = [0, -0.5, 0, 0.5]  # 偏移量取值


# host = ''
# port = 51888
#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#
# s.bind((host, port))
# s.listen(10)


# def getData():
#
#     try:
#         client_socket, client_addr = s.accept()
#     except Exception as e:
#         traceback.print_exc()
#
#     try:
#         command = client_socket.recv(4096)
#         global lateral_offset
#         lateral_offset = int(command) / 2.0
#     except Exception as e:
#         traceback.print_exc()
#
#     try:
#         client_socket.close()
#     except Exception as e:
#         traceback.print_exc()


######################### 条件指令 ########################################

# 奥恩斯坦-乌伦贝克过程
class OU(object):

    def function(self, x, theta, mu, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)[0]  # randn函数返回一个或一组样本，具有标准正态分布。

    def function1(self, x, theta, mu, sigma):
        return theta * (mu - x) + sigma * np.random.normal(loc=0, scale=0.8, size=None)


# Actor网络
class ANet(nn.Module):  # ae(s)=a
    # 初始化
    def __init__(self):
        # 调用父类的初始化函数
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 300)  # 线性全连接层1 输入维度N_STATES 输出维度300
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化
        self.fc2 = nn.Linear(300, 600)  # 线性全连接层2 输入维度300 输出维度600
        self.fc2.weight.data.normal_(0, 0.1)  # 初始化
        self.Steering = nn.Linear(600, 1)  # 转角层 输入维度600 输出维度1
        self.Steering.weight.data.normal_(0, 0.1)  # 初始化
        self.Acceleration = nn.Linear(600, 1)  # 加速层 输入维度600 输出维度1
        self.Acceleration.weight.data.normal_(0, 0.1)  # 初始化
        self.Brake = nn.Linear(600, 1)  # 刹车层 输入维度600 输出维度1
        self.Brake.weight.data.normal_(0, 0.1)  # 初始化

    # 过一遍Actor网络
    def forward(self, x):
        x = self.fc1(x)  # 初始输入经过第一层全连接层
        x = F.relu(x)  # 使用relu激活函数
        x = self.fc2(x)  # 经过第二层全连接层
        x = F.relu(x)  # 使用relu激活函数

        steer = self.Steering(x)  # 得到转角输出
        steer = torch.tanh(steer)  # 使用tanh激活函数处理转角输出
        accel = self.Acceleration(x)  # 得到加速输出
        accel = torch.sigmoid(accel)  # 使用tanh激活函数处理加速输出
        brake = self.Brake(x)  # 得到刹车输出
        brake = torch.sigmoid(brake)  # 使用tanh激活函数处理刹车输出

        actions_value = torch.cat([steer, accel, brake], dim=-1)  # 拼接三个动作，得到action的值
        return actions_value


# Critic网络
class CNet(nn.Module):  # ce(s,a)=q
    # 初始化
    def __init__(self):
        # 调用父类的初始化函数
        super(CNet, self).__init__()
        self.fcs1 = nn.Linear(N_STATES, 300)  # 线性全连接层1 输入维度N_STATE 输出维度300
        self.fcs1.weight.data.normal_(0, 0.1)  # 初始化
        self.fcs2 = nn.Linear(300, 600)  # 线性全连接层2 输入维度300E 输出维度600
        self.fcs2.weight.data.normal_(0, 0.1)  # 初始化
        self.fca = nn.Linear(N_ACTIONS, 600)  # 线性全连接层 输入维度N_ACTIONS 输出维度300
        self.fca.weight.data.normal_(0, 0.1)  # 初始化

        self.h1 = nn.Linear(600, 600)  # 线性隐藏层 输入维度600 输出维度600
        self.h1.weight.data.normal_(0, 0.1)  # 初始化

        self.out = nn.Linear(600, N_ACTIONS)  # 线性输出层 输入维度600 输出为度N_ACTIONS
        self.out.weight.data.normal_(0, 0.1)  # 初始化

    # 过一遍Critic网络
    def forward(self, s, a):
        x = self.fcs1(s)  # 初始输入经过第一层全连接层
        x = F.relu(x)  # 使用relu激活函数
        x = self.fcs2(x)  # 经过第二层全连接层
        y = self.fca(a)  # actions经过全连接层

        net = x + y  # 拼接处理后的输入以及actions
        net = self.h1(net)  # 经过隐藏层
        net = F.relu(net)  # 使用relu激活函数

        actions_value = self.out(net)  # 输出评估结果
        return actions_value


# agetn的DDPG算法
class DDPG(object):
    # 初始化
    def __init__(self):

        # 初始化两个Actor网络
        self.Actor_eval = ANet()
        self.Actor_target = ANet()
        # 初始化两个Critic网络
        self.Critic_eval = CNet()
        self.Critic_target = CNet()
        # 将模型网络参数转换为cuda tensor
        self.Actor_eval = self.Actor_eval.cuda()
        self.Actor_target = self.Actor_target.cuda()
        self.Critic_eval = self.Critic_eval.cuda()
        self.Critic_target = self.Critic_target.cuda()

        # 初始化经验回放池
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS + 2), dtype=np.float32)
        self.memory_counter = 0  # for storing memory 经验回放池计数
        # 探索策略
        self.epsilon = 1.0  # greedy policy ϵ-贪心探索策略
        self.epsilon_increment = 1
        # 为网络配置优化器
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)  # 学习率为0.001
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)  # 学习率为0.0001
        self.loss_td = nn.MSELoss()  # 损失采用均方误差
        self.cost_his_a = []  # 记录损失值
        self.cost_his_c = []  # 记录损失值
        self.ou = OU()  # 噪声添加选取OU噪声

    # 动作选择
    def choose_action(self, x):

        # 将输入处理为tensor
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        x = x.cuda()

        # 输入Actor网络，得到action
        action_value = self.Actor_eval.forward(x)
        action_value = action_value.cpu()

        action_value = action_value.data.numpy()
        # action_value = np.array(action_value)
        # print("action_value", action_value.shape)#[1,3]

        # 给action添加噪声
        action = np.zeros([1, N_ACTIONS], dtype=np.float32)
        noise_t = np.zeros([1, N_ACTIONS], dtype=np.float32)

        # if np.random.uniform() < 0.5:
        noise_t[0][0] = max(self.epsilon, 0) * self.ou.function1(action_value[0][0], 1.00, 0.0, 0.30)
        noise_t[0][1] = max(self.epsilon, 0) * self.ou.function(action_value[0][1], 1.00, 0.3, 0.10)
        noise_t[0][2] = max(self.epsilon, 0) * self.ou.function(action_value[0][2], 1.00, 0.15, 0.05)

        action[0][0] = action_value[0][0] + noise_t[0][0]
        action[0][1] = max(action_value[0][1] + noise_t[0][1], 0.0)
        action[0][2] = max(action_value[0][2] + noise_t[0][2], 0.0)

        # 根据ϵ 随机探索
        if np.random.uniform() < 0.1 and self.epsilon > 0.05:
            # 仅刹车
            action[0][1] = 0.0
            action[0][2] = 0.15
        elif self.epsilon > 0.5:
            # 仅加速
            action[0][2] = 0.0

        return action

    # 将一回合经理存储到经验回放池
    def store_transaction(self, s, a, r, s_, done):
        transaction = np.hstack((s, a, [r], s_, [done]))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transaction
        self.memory_counter += 1

    # 网络参数更新
    def learn(self):

        # for x in self.Actor_target.state_dict().keys():
        #     eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        # for x in self.Critic_target.state_dict().keys():
        #     eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # 软更新网络 (可考虑设置更新频率)
        tmp_weight = self.Actor_eval.state_dict()
        for k1, k2 in zip(self.Actor_eval.state_dict(), self.Actor_target.state_dict()):
            tmp_weight[k1].data = TAU * self.Actor_eval.state_dict()[k1].data + (1 - TAU) * \
                                  self.Actor_target.state_dict()[k2].data
        self.Actor_target.load_state_dict(tmp_weight)

        tmp_weight = self.Critic_eval.state_dict()
        for k1, k2 in zip(self.Critic_eval.state_dict(), self.Critic_target.state_dict()):
            tmp_weight[k1].data = TAU * self.Critic_eval.state_dict()[k1].data + (1 - TAU) * \
                                  self.Critic_target.state_dict()[k2].data
        self.Critic_target.load_state_dict(tmp_weight)

        # 从经验池取出一批次的回放片段
        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, size=BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.FloatTensor(b_memory[:, N_STATES: N_STATES + N_ACTIONS]))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS: N_STATES + N_ACTIONS + 1]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS + 1: 2 * N_STATES + N_ACTIONS + 1]))
        b_done = Variable(torch.FloatTensor(b_memory[:, -1:]))

        b_s = b_s.cuda()  # 真实state
        b_a = b_a.cuda()  # 真实action
        b_r = b_r.cuda()  # 真实奖励值
        b_s_ = b_s_.cuda()  # 真实下一个state
        b_done = b_done.cuda()  # 结束标识

        a = self.Actor_eval(b_s)  # 根据真实state预测的action
        q = self.Critic_eval(b_s, a)  # 根据真实state和预测action预测的Q值
        loss_a = -torch.mean(q)  # 计算actor网络的loss
        # self.cost_his_a.append(loss_a)
        self.atrain.zero_grad()  # 清空过往梯度
        loss_a.backward()  # 反向传播 计算当前梯度
        self.atrain.step()  # 根据梯度更新网络参数

        q_v = self.Critic_eval(b_s, b_a)    # 根据真实state和真实action预测的Q值
        a_ = self.Actor_target(b_s_)    # 根据真实的下一个state预测的action
        q_target = self.Critic_target(b_s_, a_).detach()    # 切断q_target的反向传播 让其梯度对主网络的梯度造成影响
        for k in range(BATCH_SIZE):
            if b_done[k]:
                q_target[k] = b_r[k]
            else:
                q_target[k] = b_r[k] + GAMMA_REWARD * q_target[k]   # y_i = r_t + γ * Q'(s_t+1, π'(s+t+1|θ') | θ_Q')
        loss_c = self.loss_td(q_v, q_target)
        # self.cost_his_c.append(loss_c)
        self.ctrain.zero_grad() # 清空过往梯度
        loss_c.backward()   # 反向传播，计算当前梯度
        self.ctrain.step()  # 根据梯度更新网络参数

        if GLOBAL_STEP > 20100:
            self.epsilon_increment = self.epsilon_increment + 1
            # 逐渐降低行为的随机性
            if self.epsilon_increment % 5 == 0:
                self.epsilon = (99997 / 100000) ** (self.epsilon_increment / 5)


# 'angle','trackPos','speedX', 'speedY', 'speedZ','track' 处理观察内容，得到网络作为网络输入的观察内容
def handle_ob(ob):
    i = 0
    ob_net = np.full(12, 0.5)
    for ob_tmp in ob:
        if i == 5:
            ob_net = np.append(ob_net, ob_tmp, axis=0)
        else:
            if i == 4:
                tmp = np.zeros(1)
                tmp[0] = ob_tmp
                ob_net = np.append(ob_net, tmp, axis=0)
            elif i == 3:
                tmp = np.full(3, ob_tmp)
                ob_net = np.append(ob_net, tmp, axis=0)
            else:
                if i == 1:
                    tmp = np.full(5, ob_tmp + lateral_offset)
                    ob_net = np.append(ob_net, tmp, axis=0)
                else:
                    tmp = np.full(5, ob_tmp)
                    ob_net = np.append(ob_net, tmp, axis=0)
        i = i + 1
    return ob_net  # ob_net最终维度为50


if __name__ == "__main__":

    # Generate a Torcs environment 创建一个Torcs环境
    env = TorcsEnv(vision=vision, throttle=True)
    agent = DDPG()  # 创建一个采用DDPG算法的agent

    # 加载已有模型
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

        agent.epsilon = 0.0
        agent.epsilon_increment = 220000

    # 训练模型
    if is_train:

        print("TORCS Experiment Start.")
        for i in range(MAX_EPISODE):
            print("Episode : " + str(EPISODE_COUNT))

            # 重置环境
            if np.mod(i, 100) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error 定数重启TORCS避免内存泄漏
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            # 处理观察内容
            ob_net = handle_ob(ob)

            total_reward = 0.0  # 定义初始奖励
            for j in range(MAX_STEPS):

                if GLOBAL_STEP % 2000 == 0:
                    print('GLOBAL_STEP: ', GLOBAL_STEP)

                action = agent.choose_action(ob_net)  # agent选择动作

                action = np.squeeze(action)

                # print("action: ", action.shape)#[3,]

                ob_, reward, done, _ = env.step(action, lateral_offset)

                ob_net_ = handle_ob(ob_)

                # 将该回合存入经验回放池
                agent.store_transaction(ob_net, action, reward, ob_net_, done)

                # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
                if GLOBAL_STEP > 10000:
                    agent.learn()

                flag = 0
                if GLOBAL_STEP >= 20000:
                    if GLOBAL_STEP % 10000 == 0:
                        lateral_offset = offset[int(((GLOBAL_STEP - 10000) / 10000) % 4)]  # ????????????????
                        flag = 1
                else:
                    flag = 0

                # 将下一个 state_ 变为 下次循环的 state
                if flag == 0:
                    ob_net = ob_net_
                else:
                    ob_net = handle_ob(ob_)

                total_reward += reward

                GLOBAL_STEP += 1

                # 存储模型
                if GLOBAL_STEP >= 600000 and GLOBAL_STEP % 30000 == 0:
                    t_MODEL_PATH = MODEL_PATH + str(EPISODE_COUNT) + '_' + str(GLOBAL_STEP) + '.pth'
                    state = {
                        'EPISODE_COUNT': EPISODE_COUNT,
                        'GLOBAL_STEP': GLOBAL_STEP,
                        'state_actor_eval': agent.Actor_eval.state_dict(),
                        'state_actor_target': agent.Actor_target.state_dict(),
                        'state_critic_eval': agent.Critic_eval.state_dict(),
                        'state_critic_target': agent.Critic_target.state_dict(),
                        'optimizer_atrain': agent.atrain.state_dict(),
                        'optimizer_ctrain': agent.ctrain.state_dict(),
                        # 'memory': agent.memory,
                        # 'memory_counter': agent.memory_counter,
                        # 'epsilon': agent.epsilon,
                        # 'cost_his_a': agent.cost_his_c,
                        # 'cost_his_c': agent.cost_his_c,
                    }
                    torch.save(state, t_MODEL_PATH)

                if done:
                    break

            EPISODE_COUNT = EPISODE_COUNT + 1
            print(str(EPISODE_COUNT) + " -th Episode  ---------------  " + "TOTAL REWARD : " + str(total_reward))
            # if len(agent.cost_his_a) > 0:
            #     print("loss_a: " + str(agent.cost_his_a[len(agent.cost_his_a)-1]))
            #     print("loss_c: " + str(agent.cost_his_c[len(agent.cost_his_c) - 1]))
            print("epsilon: ", agent.epsilon, "        ", "Total Step: " + str(GLOBAL_STEP))
            print("")

    else:

        agent.epsilon = 0.0
        print("TORCS Experiment Start.")
        for i in range(MAX_EPISODE):

            print("Episode : " + str(EPISODE_COUNT))

            if np.mod(i, 100) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error 重启torcs避免内存泄露
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            ob_net = handle_ob(ob)

            for j in range(MAX_STEPS):

                ######################### 条件指令 ########################################

                ######################### 条件指令 ########################################

                action = agent.choose_action(ob_net)

                action = np.squeeze(action)

                ob_, reward, done, _ = env.step(action, lateral_offset)

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

# 调节OU
# 增加随机刹车
# 调节奖励函数
# Episode:1168
# Total Step: 175019

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

# 180000
