"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class ThreeDCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.y_threshold = 2.4 #jm

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2, #x
            np.finfo(np.float32).max, #x_dot
            self.theta_threshold_radians * 2, #x_theta
            np.finfo(np.float32).max, # x_theta_dot
            self.y_threshold * 2,  # y
            np.finfo(np.float32).max,  # y_dot
            self.theta_threshold_radians * 2,  # y_theta
            np.finfo(np.float32).max]  # y_theta_dot
        )

        self.action_space = spaces.Discrete(4) #jm
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, x_theta, x_theta_dot, y, y_dot, y_theta, y_theta_dot = state

        force_x = 0
        force_y = 0

        if action==0:
            force_x = -self.force_mag
        elif action==1:
            force_x = self.force_mag
        elif action==2:
            force_y = -self.force_mag
        elif action==3:
            force_y = self.force_mag

        # update x direction
        x_costheta = math.cos(x_theta)
        x_sintheta = math.sin(x_theta)
        temp = (force_x + self.polemass_length * x_theta_dot * x_theta_dot * x_sintheta) / self.total_mass
        thetaacc = (self.gravity * x_sintheta - x_costheta* temp) / (self.length * (4.0/3.0 - self.masspole * x_costheta * x_costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * x_costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        x_theta = x_theta + self.tau * x_theta_dot
        x_theta_dot = x_theta_dot + self.tau * thetaacc

        # update y direction
        y_costheta = math.cos(x_theta)
        y_sintheta = math.sin(x_theta)
        temp = (force_y + self.polemass_length * y_theta_dot * y_theta_dot * y_sintheta) / self.total_mass
        thetaacc = (self.gravity * y_sintheta - y_costheta* temp) / (self.length * (4.0/3.0 - self.masspole * y_costheta * y_costheta / self.total_mass))
        yacc  = temp - self.polemass_length * thetaacc * y_costheta / self.total_mass
        y  = y + self.tau * y_dot
        y_dot = y_dot + self.tau * yacc
        y_theta = y_theta + self.tau * y_theta_dot
        y_theta_dot = y_theta_dot + self.tau * thetaacc



        self.state = (x,x_dot,x_theta,x_theta_dot, y, y_dot, y_theta, y_theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or x_theta < -self.theta_threshold_radians \
                or x_theta > self.theta_threshold_radians \
                or y < -self.y_threshold \
                or y > self.y_threshold \
                or y_theta < -self.theta_threshold_radians\
                or y_theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(8,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def render_orthographic(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        # carty = 50 # TOP OF CART
        polewidth = 5
        polelen = scale * 0.5
        cartwidth = 15
        cartheight = 15

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)

            origin_res = 50
            origin_radius = 2

            track_res = 100
            track_radius = self.x_threshold

            origin_circle = rendering.make_circle(radius=origin_radius, res=origin_res, filled=True)
            origin_circle.set_color(0, 0, 0)
            origin_circle.add_attr(rendering.Transform(
                translation=(screen_width/2.0, screen_height/2.0)))
            self.viewer.add_geom(origin_circle)

            origin_circle = rendering.make_circle(radius=track_radius * scale, res=track_res, filled=False)
            origin_circle.set_color(0, 0, 0)
            origin_circle.add_attr(rendering.Transform(
                translation=(screen_width/2.0, screen_height/2.0)))
            self.viewer.add_geom(origin_circle)



            # self.track = rendering.Line((0,carty), (screen_width,carty))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        carty = x[4]*scale+screen_height/2.0

        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


