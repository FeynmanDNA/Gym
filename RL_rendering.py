import gym
import math


env = gym.make('CartPole-v0')

# reset environment
env.reset()

# 离散空间允许固定范围的非负数，因此在这种情况下，有效的动作是0或1. 
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)

print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])

################
# 空间（Spaces）
################
from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8

#  for _ in range(50):
    #  env.render()
    #  env.step(env.action_space.sample()) # take a randome action

#####################
# Env.render画图
#####################
# 首先，导入库文件（包括gym模块和gym中的渲染模块）
import gym
from gym.envs.classic_control import rendering

# 我们生成一个类，该类继承 gym.Env. 同时，可以添加元数据，改变渲染环境时的参数
class TwoLines(gym.Env):
    # 如果你不想改参数，下面可以不用写
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    # 我们在初始函数中定义一个 viewer ，即画板
    def __init__(self):
        self.viewer = rendering.Viewer(600, 400)   # 600x400 是画板的长和框
    # 继承Env render函数
    def render(self, mode='human', close=False):
        # 下面就可以定义你要绘画的元素了
        line1 = rendering.Line((100, 300), (500, 300))
        line2 = rendering.Line((100, 200), (500, 200))
        # 给元素添加颜色
        line1.set_color(0, 0, 0)
        line2.set_color(0, 0, 0)
        # 把图形元素添加到画板中
        self.viewer.add_geom(line1)
        self.viewer.add_geom(line2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

class NormalCircle(TwoLines):
    def render(self, mode='human', close=False):
        # 画一个直径为 30 的园
        circle = rendering.make_circle(30)
        self.viewer.add_geom(circle)
        # 默认情况下圆心在坐标原点, 原点在左下角
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

class TransformCircle(NormalCircle):
    def render(self, mode='human', close=False):
        # 画一个直径为 30 的园
        circle = rendering.make_circle(30)
        # 添加一个平移操作
        circle_transform = rendering.Transform(translation=(100, 200))
        # 让圆添加平移这个属性
        circle.add_attr(circle_transform)
        self.viewer.add_geom(circle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

class Polyline(TwoLines):
    def render(self, mode='human', close=False):
        circle = rendering.make_polyline([(50, 200-50*math.sqrt(3)),
                                  (100, 200), (200, 200),
                                  (250, 200 - 50 * math.sqrt(3)),
                                  (200, 200 - 100*math.sqrt(3)),
                                  (100, 200 - 100*math.sqrt(3)),
                                  (50, 200 - 50 * math.sqrt(3))])
        # 添加一个平移操作
        circle_transform = rendering.Transform(translation=(100, 200))
        circle.add_attr(circle_transform)
        self.viewer.add_geom(circle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


twoLines = TwoLines()
for _ in range(20):
    twoLines.render()

normalCircle = NormalCircle()
for _ in range(20):
    normalCircle.render()

transformCircle = TransformCircle()
for _ in range(20):
    transformCircle.render()

polyLine = Polyline()
for _ in range(20):
    polyLine.render()

