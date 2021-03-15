import  numpy as np
# action = np.zeros((36, 3))
#
#
# for i in range(9):
#     action[i, 0] = -1 + 0.25 * i
#     action[i, 1] = 0
#     action[i, 2] = 0
#
# for i in range(2):
#     for j in range(9):
#         action[9 + i * 9 + j, 0] = -1 + 0.25 * j
#         action[9 + i * 9 + j, 1] = 0.3 if i == 0 else 0.7
#         action[9 + i * 9 + j, 2] = 0
#
#
# for i in range(9):
#     action[27 + i, 0] = -1 + 0.25 * i
#     action[27 + i, 1] = 0
#     action[27 + i, 2] = 0.25
#
#
#
# print(action)
# print(action.shape)
#
# # steer 0.3 0.7
# # brake 0.25

# action_tmp = np.array([-0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75])
# action = np.zeros((36, 3))
# for i in range(9):
#     action[i, 0] = action_tmp[i]
#     action[i, 1] = 0
#     action[i, 2] = 0
# for i in range(2):
#     for j in range(9):
#         action[9 + i * 9 + j, 0] = action_tmp[j]
#         action[9 + i * 9 + j, 1] = 0.3 if i == 0 else 0.7
#         action[9 + i * 9 + j, 2] = 0
# for i in range(9):
#     action[27 + i, 0] = action_tmp[i]
#     action[27 + i, 1] = 0
#     action[27 + i, 2] = 0.25
# print(action)





# import torch
# import matplotlib.pyplot as plt
#
# font1 = {'family': 'Microsoft YaHei',
#          'weight': 'normal',
#          'size': 15,
#          }
#
# RESTORE_MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/test_dqn_track-5.pth'
# state = torch.load(RESTORE_MODEL_PATH)
# # EPISODE_COUNT = state['EPISODE_COUNT']
# # GLOBAL_STEP = state['GLOBAL_STEP']
# # print(EPISODE_COUNT)
# # print(GLOBAL_STEP)
# speedx = state['speedx']
# speedy = state['speedy']
# re = state['re']
# speedx = speedx[0:1000]
# speedy = speedy[0:1000]
# re = re[0:1000]
# step = np.arange(0, 1000)
#
# new_speedx = []
# # new_speedy = []
# # new_re = []
# new_step = []
# for i in range(1000):
#     if i % 2 ==0:
#         new_speedx.append(speedx[i])
#         # new_speedy.append(speedy[i])
#         # new_re.append(re[i])
#         new_step.append(step[i])
#
#
# circle = []
# y_circle = []
# triangle = []
# y_triangle = []
# for i in range(1000):
#     if i % 100 == 0 and i!=0:
#         triangle.append(speedx[i])
#         y_triangle.append(i)
#
# for i in range(1000):
#     if i % 100 == 0 and i!=0:
#         circle.append(speedy[i])
#         y_circle.append(i)
#
# # 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.family'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
#
# # 4 个 plot 函数画出 4 条线，线形为折线，每条线对应各自的标签 label
# plt.plot(new_step, new_speedx, 'r:', label='speedx')
# plt.plot(step, speedy, 'b:', label='speedy')
# plt.plot(step, re, 'g-', label='reward')
#
# plt.plot(y_triangle, triangle, 'ro')
# plt.plot(y_circle, circle, 'b^')
#
# s = [0,200,400,600,800]
#
# plt.ylim([-10, 135]) # 设置纵坐标轴范围为 -2 到 2
# plt.xticks(s)  # 设置横坐标刻度为给定的年份
# plt.ylabel('奖励/速度',font1) # 横坐标轴的标题
# plt.xlabel('迭代步数',font1) # 纵坐标轴的标题
# plt.legend() # 显示图例，即每条线对应 label 中的内容
#
# # plt.savefig('/home/qzw/PycharmProjects/my_torcs/ckpt/dqn_track-5.jpg')
# plt.show() # 显示图形


import torch
import matplotlib.pyplot as plt

font1 = {'family': 'Microsoft YaHei',
         'weight': 'normal',
         'size': 15,
         }

RESTORE_MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/test_ddpg_road.pth'
state = torch.load(RESTORE_MODEL_PATH)
# EPISODE_COUNT = state['EPISODE_COUNT']
# GLOBAL_STEP = state['GLOBAL_STEP']
# print(EPISODE_COUNT)
# print(GLOBAL_STEP)
speedx = state['speedx']
speedy = state['speedy']
re = state['re']
speedx = speedx[0:4200]
speedy = speedy[0:4200]
re = re[0:4200]
step = np.arange(0, 4200)

new_speedx = []
new_speedy = []
new_re = []
new_step = []
for i in range(4200):
    if i % 8 ==0:
        new_speedx.append(speedx[i])
        new_speedy.append(speedy[i])
        new_re.append(re[i])
        new_step.append(step[i])


circle = []
y_circle = []
triangle = []
y_triangle = []
for i in range(4200):
    if i % 200 == 0 and i != 0:
        triangle.append(speedx[i])
        y_triangle.append(i)

for i in range(4200):
    if i % 400 == 0 and i != 0:
        circle.append(speedy[i])
        y_circle.append(i)

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



# 4 个 plot 函数画出 4 条线，线形为折线，每条线对应各自的标签 label
plt.plot(new_step, new_speedx, 'r:', label='speedx')
plt.plot(new_step, new_speedy, 'b:', label='speedy')
plt.plot(new_step, new_re, 'g-', label='reward')

plt.plot(y_triangle, triangle, 'ro')
plt.plot(y_circle, circle, 'b^')

s = [0,500,1000,1500,2000,2500,3000,3500]

plt.ylim([-30, 140]) # 设置纵坐标轴范围为 -2 到 2
plt.xticks(s)  # 设置横坐标刻度为给定的年份
plt.ylabel('奖励/速度',font1) # 横坐标轴的标题
plt.xlabel('迭代步数',font1) # 纵坐标轴的标题
plt.legend() # 显示图例，即每条线对应 label 中的内容

# plt.savefig('/home/qzw/PycharmProjects/my_torcs/ckpt/dqn_track-5.jpg')
plt.show() # 显示图形

# import torch
# import matplotlib.pyplot as plt
#
# font1 = {'family': 'Microsoft YaHei',
#          'weight': 'normal',
#          'size': 15,
#          }
#
# RESTORE_MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/test_ddpg_track-5.pth'
# state = torch.load(RESTORE_MODEL_PATH)
# # EPISODE_COUNT = state['EPISODE_COUNT']
# # GLOBAL_STEP = state['GLOBAL_STEP']
# # print(EPISODE_COUNT)
# # print(GLOBAL_STEP)
# speedx = state['speedx']
# speedy = state['speedy']
# re = state['re']
# speedx = speedx[0:1200]
# speedy = speedy[0:1200]
# re = re[0:1200]
# step = np.arange(0, 1200)
#
# new_speedx = []
# new_speedy = []
# new_re = []
# new_step = []
# for i in range(1200):
#     if i % 1 ==0:
#         new_speedx.append(speedx[i])
#         new_speedy.append(speedy[i])
#         new_re.append(re[i])
#         new_step.append(step[i])
#
#
# circle = []
# y_circle = []
# triangle = []
# y_triangle = []
# for i in range(1200):
#     if i % 120 == 0 and i != 0:
#         triangle.append(speedx[i])
#         y_triangle.append(i)
#
# for i in range(1200):
#     if i % 120 == 0 and i != 0:
#         circle.append(speedy[i])
#         y_circle.append(i)
#
# # 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.family'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
#
# # 4 个 plot 函数画出 4 条线，线形为折线，每条线对应各自的标签 label
# plt.plot(new_step, new_speedx, 'r:', label='speedx')
# plt.plot(new_step, new_speedy, 'b:', label='speedy')
# plt.plot(new_step, new_re, 'g-', label='reward')
#
# plt.plot(y_triangle, triangle, 'ro')
# plt.plot(y_circle, circle, 'b^')
#
# s = [0,200,400,600,800,1000]
#
# plt.ylim([-10, 120]) # 设置纵坐标轴范围为 -2 到 2
# plt.xticks(s)  # 设置横坐标刻度为给定的年份
# plt.ylabel('奖励/速度',font1) # 横坐标轴的标题
# plt.xlabel('迭代步数',font1) # 纵坐标轴的标题
# plt.legend() # 显示图例，即每条线对应 label 中的内容
#
# # plt.savefig('/home/qzw/PycharmProjects/my_torcs/ckpt/dqn_track-5.jpg')
# plt.show() # 显示图形