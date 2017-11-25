"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
    Jeremy:
        This is a modified environment for 3D mountain car.
        Currently the rendering only shows the x and z axises 
        (i.e. The 3D graphics is projected onto the (0,1,0) plane) 

        More to come..
'''

class ThreeDMountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.name = 'ThreeDMountainCar-v0'

        self.min_position_x = -1.2
        self.max_position_x = math.pi/6 - 0.01
        self.max_speed_x = 0.07

        # jm: The second coordinate bounds
        self.min_position_y = -1.2
        self.max_position_y = math.pi/6 - 0.01
        self.max_speed_y = 0.07

        self.goal_position = 0.5

        self.low = np.array([self.min_position_x, self.min_position_y, -self.max_speed_x, -self.max_speed_y]) # jm: x,y,x_dot,y_dot
        self.high = np.array([self.max_position_x, self.max_position_y, self.max_speed_x, self.max_speed_y]) # jm: x,y,x_dot,y_dot

        self.viewer_x = None
        self.viewer_y = None
        self.viewer_orthographic = None

        self.action_space = spaces.Discrete(5) # jm: {Neutral, West, East, South, North}
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self.reset()

    @property
    def action_size(self):
        return self.action_space.n

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position_x, position_y, velocity_x, velocity_y = self.state
        if action == 0: # neutral
            velocity_x += math.cos(3*position_x)*(-0.0025)
            velocity_y += math.cos(3*position_y)*(-0.0025)
        elif action == 1: # left x, west
            velocity_x += math.cos(3*position_x)*(-0.0025) - 0.001
            velocity_y += math.cos(3*position_y)*(-0.0025)
        elif action == 2: # right x, east
            velocity_x += math.cos(3*position_x)*(-0.0025) + 0.001
            velocity_y += math.cos(3*position_y)*(-0.0025)
        elif action == 3: # left y, south
            velocity_x += math.cos(3*position_x)*(-0.0025)
            velocity_y += math.cos(3*position_y)*(-0.0025) - 0.001
        elif action == 4:
            velocity_x += math.cos(3*position_x)*(-0.0025)
            velocity_y += math.cos(3*position_y)*(-0.0025) + 0.001


        velocity_x = np.clip(velocity_x, -self.max_speed_x, self.max_speed_x)
        velocity_y = np.clip(velocity_y, -self.max_speed_y, self.max_speed_y)

        position_x += velocity_x
        position_y += velocity_y

        position_x = np.clip(position_x, self.min_position_x, self.max_position_x)
        position_y = np.clip(position_y, self.min_position_y, self.max_position_y)

        if (position_x == self.min_position_x and velocity_x<0):
            velocity_x = 0

        if (position_y == self.min_position_y and velocity_y<0):
            velocity_y = 0

        done = bool(position_x >= self.goal_position and position_y >= self.goal_position)
        # done = bool(position_x >= self.goal_position)

        reward = -1.0

        self.state = (position_x, position_y, velocity_x, velocity_y)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), self.np_random.uniform(low=-0.6, high=-0.4), 0, 0])
        return np.array(self.state)

    # def _height_for_car(self, xs, ys):
    #     pos = np.sqrt(np.square(xs) + np.square(ys))
    #     return np.sin(3 * pos)*.45+.55

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer_x is not None:
                self.viewer_x.close()
                self.viewer_x = None
            if self.viewer_y is not None:
                self.viewer_y.close()
                self.viewer_y = None
            if self.viewer_orthographic is not None:
                self.viewer_orthographic.close()
                self.viewer_orthographic = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position_x - self.min_position_x # same for y
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        if self.viewer_x is None:
            from gym.envs.classic_control import rendering
            self.viewer_x = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position_x, self.max_position_x, 100)
            # ys = np.linspace(self.min_position_y, self.max_position_y, 100)
            # ys = np.zeros(100)
            zs = self._height(xs)
            xyzs = list(zip((xs-self.min_position_x)*scale, zs*scale))

            self.track = rendering.make_polyline(xyzs)
            self.track.set_linewidth(4)
            self.viewer_x.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans_x = rendering.Transform()
            car.add_attr(self.cartrans_x)
            self.viewer_x.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans_x)
            self.viewer_x.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans_x)
            backwheel.set_color(.5, .5, .5)
            self.viewer_x.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position_x)*scale
            flagy1 = self._height(self.goal_position)*scale #jm: need to change this
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer_x.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer_x.add_geom(flag)

        pos = self.state[0]
        self.cartrans_x.set_translation((pos-self.min_position_x)*scale, self._height(pos)*scale) #jm: need to change this
        self.cartrans_x.set_rotation(math.cos(3 * pos))

        # pos_x = self.state[0]
        # pos_y = self.state[1]
        # self.cartrans_x.set_translation((pos_x-self.min_position_x)*scale, self._height_for_car(pos_x, pos_y)*scale)
        # self.cartrans_x.set_rotation(math.cos(3 * pos_x))

        return self.viewer_x.render(return_rgb_array = mode=='rgb_array')


    def render_y(self, mode='human', close=False):
        if close:
            if self.viewer_x is not None:
                self.viewer_x.close()
                self.viewer_x = None
            if self.viewer_y is not None:
                self.viewer_y.close()
                self.viewer_y = None
            if self.viewer_orthographic is not None:
                self.viewer_orthographic.close()
                self.viewer_orthographic = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position_y - self.min_position_y # same for y
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        if self.viewer_y is None:
            from gym.envs.classic_control import rendering
            self.viewer_y = rendering.Viewer(screen_width, screen_height)
            # xs = np.linspace(self.min_position_x, self.max_position_x, 100)
            ys = np.linspace(self.min_position_y, self.max_position_y, 100)
            # ys = np.zeros(100)
            zs = self._height(ys)
            xyzs = list(zip((ys-self.min_position_y)*scale, zs*scale))

            self.track = rendering.make_polyline(xyzs)
            self.track.set_linewidth(4)
            self.viewer_y.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans_y = rendering.Transform()
            car.add_attr(self.cartrans_y)
            self.viewer_y.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans_y)
            self.viewer_y.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans_y)
            backwheel.set_color(.5, .5, .5)
            self.viewer_y.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position_x)*scale
            flagy1 = self._height(self.goal_position)*scale #jm: need to change this
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer_y.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(0,.8,.8)
            self.viewer_y.add_geom(flag)

        pos = self.state[1]
        self.cartrans_y.set_translation((pos-self.min_position_y)*scale, self._height(pos)*scale) #jm: need to change this
        self.cartrans_y.set_rotation(math.cos(3 * pos))

        # pos_x = self.state[0]
        # pos_y = self.state[1]
        # self.cartrans_y.set_translation((pos_y-self.min_position_y)*scale, self._height_for_car(pos_x, pos_y)*scale)
        # self.cartrans_y.set_rotation(math.cos(3 * pos_y))

        return self.viewer_y.render(return_rgb_array = mode=='rgb_array')

    def render_orthographic(self, mode='human', close=False):

        if close:
            if self.viewer_x is not None:
                self.viewer_x.close()
                self.viewer_x = None
            if self.viewer_y is not None:
                self.viewer_y.close()
                self.viewer_y = None
            if self.viewer_orthographic is not None:
                self.viewer_orthographic.close()
                self.viewer_orthographic = None
            return

        screen_width = 600
        screen_height = 600

        world_height = self.max_position_y - self.min_position_y # same for y
        scale = (screen_height-120)/world_height
        carwidth=20
        carheight=10

        if self.viewer_orthographic is None:
            from gym.envs.classic_control import rendering
            self.viewer_orthographic = rendering.Viewer(screen_width, screen_height)

            # ys = np.linspace(self.min_position_y, self.max_position_y, 100)

            # zs = self._height(ys)
            # xyzs = list(zip((ys-self.min_position_y)*scale, zs*scale))

            # self.track = rendering.make_polyline(xyzs)
            # self.track.set_linewidth(4)
            # self.viewer_orthographic.add_geom(self.track)

            min_x = -math.pi/6
            min_y = -math.pi/6

            origin_res = 50
            origin_radius = 2

            origin_circle = rendering.make_circle(radius=origin_radius, res= origin_res, filled = True)
            origin_circle.set_color(0,0,0)
            origin_circle.add_attr(rendering.Transform(translation=((min_x - self.min_position_x) * scale,(min_y - self.min_position_y) * scale)))
            self.viewer_orthographic.add_geom(origin_circle)


            radius_unscaled = math.sqrt((self.goal_position-min_x)**2 + (self.goal_position-min_y)**2)
            equilline_radius = radius_unscaled * scale
            #
            # points_x = []
            # points_y = []
            # offset_x, offset_y = (min_x - self.min_position_x) * scale, (min_y - self.min_position_y)*scale
            # for i in range(res):
            #     ang = 2 * math.pi * i / res
            #     points_x.append(offset_x + math.cos(ang) * radius)
            #     points_y.append(offset_y + math.sin(ang) * radius)
            #
            # equiline = list(zip(points_x, points_y))
            # self.track = rendering.make_polyline(equiline)
            # self.track.set_linewidth(4)
            equiline = rendering.make_circle(radius=equilline_radius, res= 200, filled = False)
            equiline.set_color(0,0,0)
            equiline.add_attr(rendering.Transform(translation=((min_x - self.min_position_x) * scale,(min_y - self.min_position_y) * scale)))
            equiline.add_attr(rendering.LineWidth(10)) # not sure why doesn't work
            self.viewer_orthographic.add_geom(equiline)

            clearance = 5
            clearance_wheels = 0

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight/2, -carheight/2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans_orth = rendering.Transform()
            car.add_attr(self.cartrans_orth)
            self.viewer_orthographic.add_geom(car)

            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance_wheels)))
            frontwheel.add_attr(self.cartrans_orth)
            self.viewer_orthographic.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance_wheels)))
            backwheel.add_attr(self.cartrans_orth)
            backwheel.set_color(.5, .5, .5)
            self.viewer_orthographic.add_geom(backwheel)


            flagx = (self.goal_position-self.min_position_x)*scale
            flagy1 = (self.goal_position-self.min_position_y)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer_orthographic.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(1,0,0)
            self.viewer_orthographic.add_geom(flag)

        pos_x = self.state[0]
        pos_y = self.state[1]
        self.cartrans_orth.set_translation((pos_x-self.min_position_x)*scale, (pos_y-self.min_position_y)*scale)
        # self.cartrans_orth.set_rotation(math.cos(3 * pos))


        return self.viewer_orthographic.render(return_rgb_array = mode=='rgb_array')


    def close_gui(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return