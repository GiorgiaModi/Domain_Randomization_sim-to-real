"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, randomization: str = 'none', randomize_thigh=False, randomize_leg=False, randomize_foot=False):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0
            # self.sim.model.body_mass[1] -= 2.0
            
        self.randomization = randomization # Randomization type: none, uniform, normal
        self.randomize_thigh = randomize_thigh
        self.randomize_leg = randomize_leg
        self.randomize_foot = randomize_foot
            
    def set_random_parameters(self):
        """Apply domain randomization to the hopper masses"""
        masses = self.sample_parameters()
        self.set_parameters(masses) 
        
    def sample_parameters(self):
        """
        Sample masses according to a domain randomization distribution.
        Modify the real element's true mass by summing/removing a value between -1 and 1
        """
        
        masses = np.copy(self.original_masses)  # Copia le masse originali
        
        # No randomization
        if self.randomization == 'none':
            return masses
        
        # List of indices to randomize
        indices_to_randomize = []

        # Complete randomization (no parameters specified)
        # If no parameters are specified, randomize all masses except the torso
        if not (self.randomize_thigh or self.randomize_leg or self.randomize_foot):
            indices_to_randomize = list(range(1, len(masses)))  # Randomize all except the torso
        else:
            # Selective randomization
            if self.randomize_thigh:
                indices_to_randomize.append(1)
            if self.randomize_leg:
                indices_to_randomize.append(2)
            if self.randomize_foot:
                indices_to_randomize.append(3)

        # Apply randomization
        for i in indices_to_randomize:
            if self.randomization == 'uniform':
                masses[i] = np.random.uniform(self.original_masses[i] - 1, self.original_masses[i] + 1)
            elif self.randomization == 'normal':
                masses[i] = truncnorm.rvs(-1, 1, loc=self.original_masses[i], scale=1)

        # Keep the torso mass as the original value
        masses[0] = self.original_masses[0]

        return masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """
        Reset the environment and randomize selected masses.
        """
        self.set_parameters(self.sample_parameters())
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

