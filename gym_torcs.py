import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time


# action_tmp = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
# action = np.zeros((15, 3))
# for i in range(2):
#     for j in range(5):
#         action[i * 5 + j, 0] = action_tmp[j]
#         action[i * 5 + j, 1] = 0.3 if i == 0 else 0.7
#         action[i * 5 + j, 2] = 0.0
# for i in range(5):
#     action[10 + i, 0] = action_tmp[i]
#     action[10 + i, 1] = 0
#     action[10 + i, 2] = 0.1

class TorcsEnv:
    terminal_judge_start = 200  # Speed limit is applied after this step
    termination_limit_progress = 8  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 80

    initial_reset = True

    # 初始化Torcs环境，定义action空间和observation空间
    def __init__(self, vision=False, throttle=False, gear_change=False):
        # print("Init")
        # 初始化Torcs环境
        self.vision = vision  # 图像
        self.throttle = throttle  # 油门
        self.gear_change = gear_change  # 换档

        self.initial_run = True  # 初始化运行

        # print("launch torcs")
        os.system('pkill torcs')  # 终端执行关闭torcs进程的命令
        time.sleep(0.5)  # 休眠0.5秒
        if self.vision is True:  # 如果图像标识为True
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')  # 终端执行运行torcs终端的命令，并且包含图像
        else:
            os.system('torcs  -nofuel -nodamage -nolaptime &')  # 终端执行运行torcs终端的命令，并且不包含图像
        time.sleep(0.5)  # 休眠0.5秒
        os.system('sh autostart.sh')  # 命令行执行脚本 按键
        time.sleep(0.5)  # 休眠0.5秒

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        # 定义action空间
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))  # action空间是一维数组，包含两个元素

        # 定义observation空间
        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    # 动作执行函数
    def step(self, u, lateral_offset):
        # print("Step")
        # convert thisAction to the actual torcs actionstr 转换为torcs输入命令
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action 创建一个客户端命令接受对象
        action_torcs = client.R.d

        # Save the privious full-obs from torcs for the reward calculation 存储之前的观察内容
        obs_pre = copy.deepcopy(client.S.d)

        # 将动作输入客户端命令接受对象
        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        # if obs_pre['speedX'] < 100.0:
        #     action_torcs['accel'] = this_action['accel']
        # else:
        #     action_torcs['accel'] = 0.0

        action_torcs['accel'] = this_action['accel']
        action_torcs['brake'] = this_action['brake']

        # #  Simple Autnmatic Throttle Control by Snakeoil
        # if self.throttle is False:
        #     target_speed = self.default_speed
        #     if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
        #         client.R.d['accel'] += .01
        #     else:
        #         client.R.d['accel'] -= .01
        #
        #     if client.R.d['accel'] > 0.2:
        #         client.R.d['accel'] = 0.2
        #
        #     if client.S.d['speedX'] < 10:
        #         client.R.d['accel'] += 1/(client.S.d['speedX']+.1)
        #
        #     # Traction Control System
        #     if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
        #        (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
        #         action_torcs['accel'] -= .2
        # else:
        #     action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil 自动换档

        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs 将action传给torcs客户端
        client.respond_to_server()
        # Get the response of TORCS 得到torcs客户端的反馈（下一个状态输入）
        client.get_servers_input()

        # Get the current full-observation from torcs 得到当前观察内容
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS 转换
        self.observation = self.make_observaton(obs)

        # ---------------------------------------------------------------reward-----------------------------------------------
        # Reward setting Here  奖励函数设置 #######################################
        # direction-dependent positive reward

        pos = np.array(obs['trackPos'])  # 偏离轨道中心百分比 -1右  +1左
        track = np.array(obs['track'])  # 距离轨道边缘距离
        sp = np.array(obs['speedX'])    # 纵向速度

        # 纵向速度控制：超出目标速度后，奖励值同样降低

        if sp < 63.0:
            sp = sp
        elif sp <= 80.0:
            sp = 1.22 ** (sp / 3.0)
        elif sp <= 97.0:
            sp = 1.22 ** ((80.0 - (sp - 80.0)) / 3.0)
        else:
            sp = 63.0 - ((sp - 34.0) - 63.0)

        lateral_offset = 0 - lateral_offset
        # 车辆速度高，车头转角小， 与目标位置更贴近；速度应该被限制，
        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle'])) - 4 * sp * np.abs(pos - lateral_offset)
        reward = progress

        # 横向偏离后，学习调整
        if (pos - lateral_offset) * (obs_pre['trackPos'] - lateral_offset) >= 0.0 and (
                obs_pre['trackPos'] - lateral_offset) != 0:
            if pos > lateral_offset and obs_pre['trackPos'] > pos:
                reward = reward + sp * ((obs_pre['trackPos'] - pos) / (obs_pre['trackPos'] - lateral_offset))
            elif pos < lateral_offset and obs_pre['trackPos'] < pos:
                reward = reward + sp * ((pos - obs_pre['trackPos']) / (lateral_offset - obs_pre['trackPos']))

        if sp < 50.0 and this_action['brake'] > 0.0:
            reward = -100
        # if obs['trackPos'] > 0.5 and this_action['steer'] >= 0.0:
        #     reward = -100
        # if obs['trackPos'] < -0.5 and this_action['steer'] <= 0.0:
        #     reward = -100
        # ---------------------------------------------------------------reward-----------------------------------------------

        # collision detection
        # if obs['damage'] - obs_pre['damage'] > 0:
        #     reward = -1

        # Termination judgement #########################
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = -200
            client.R.d['meta'] = True

        if self.time_step > self.terminal_judge_start:  # Episode terminates if the progress of agent is small
            if sp * np.cos(obs['angle']) < self.termination_limit_progress:
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            reward = -200
            client.R.d['meta'] = True

        if client.R.d['meta'] is True:  # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    # 重置环境
    def reset(self, relaunch=False):
        # print("Reset")

        # 重置时间
        self.time_step = 0

        # 如果不是初始重置
        if self.initial_reset is not True:

            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug! 重启TORCS避免内存泄漏
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment 连接torcs客户端
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs 启动客户端
        self.client.MAX_STEPS = np.inf  # 设置最大步长为无限大

        client = self.client  # 客户端赋值
        client.get_servers_input()  # Get the initial input from torcs 从torcs得到输入

        obs = client.S.d  # Get the current full-observation from torcs 从客户端获得当前观察
        self.observation = self.make_observaton(obs)  # 进行观察

        self.last_u = None

        # 设置初始重置表示为False
        self.initial_reset = False
        # 返回观察内容
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        # print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    # 将网络输出的action转换为torcs的输入
    def agent_to_torcs(self, u):

        # torcs_action = {'steer': action[u, 0]}
        # torcs_action.update({'accel': action[u, 1]})
        # torcs_action.update({'brake': action[u, 2]})

        # 拼接action
        torcs_action = {'steer': u[0]}
        torcs_action.update({'accel': u[1]})
        torcs_action.update({'brake': u[2]})

        # if self.throttle is True:  # throttle action is enabled
        #     torcs_action.update({'accel': u[1]})
        #
        # if self.gear_change is True: # gear change action is enabled
        #     torcs_action.update({'gear': u[2]})

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0, 12286, 3):
            temp.append(image_vec[i])
            temp.append(image_vec[i + 1])
            temp.append(image_vec[i + 2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    # 进行观察
    def make_observaton(self, raw_obs):
        # 不获取图像 返回观察内容为角度、偏离车道中心百分比、三个方向的速度、与轨道边缘距离（每10度一个，共19个）
        if self.vision is False:
            names = ['angle',
                     'trackPos',
                     'speedX', 'speedY', 'speedZ',
                     'track']
            Observation = col.namedtuple('Observaion', names)
            return Observation(angle=(np.array(raw_obs['angle'], dtype=np.float32) / 3.1416 + 1.0) / 2.0,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32),
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32) / (self.default_speed + 30.0),
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32) / (self.default_speed + 30.0),
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / (self.default_speed + 30.0),
                               track=np.array(raw_obs['track'], dtype=np.float32) / 200.0)
        # 获取图像 返回观察内容为焦点、三个方向的速度、车道、opponents、转速、车道、轮胎角速度、rgb图像
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
