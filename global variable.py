behavior_instruction =1 # K,L,R,U,D
meta_instruction = 1 # 跟车、左右变道、超车

target_offset = 0.0
target_speedX = 80

target_state = [target_offset, target_speedX] # [水平偏移量，目标速度]



# 增加一个纵向速度扩展分量，0的时候目标速度就是80,若分量+10、速度就+10
obs_var = {
 'track': [15.2628, 98.3377, 79.6485, 67.0274, 59.9558, 56.5983, 54.8613, 53.3737, 52.33, 51.3024, 50.2912, 49.2967,
           47.9328, 46.4153, 43.6907, 38.7218, 31.8358, 24.7667, 13.0994],
 'angle': -0.0700954, 'trackPos': -0.0963396, 'speedX': 81.0043, 'speedY': -0.593423, 'speedZ': -0.435851 }

# 初始数组维度12 track(19维) × 1, angle * 5, trackPos * 5, speedX * 5, speedY * 3, speedZ * 1
