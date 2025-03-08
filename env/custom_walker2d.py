import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm


class CustomWalker2D(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, randomization: str = 'none', randomize_thigh=False, randomize_leg=False, randomize_foot=False):
        MujocoEnv.__init__(self, 4, env_name="Walker2d")
        utils.EzPickle.__init__(self)
        
        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        
        if domain == 'source':  # Introduce reality gap in the source environment
            self.sim.model.body_mass[1] -= 1.0  # Adjust torso mass
            
            # Remove comments to add more weight on the feet
            #self.sim.model.body_mass[4] *= 3.0  # Put more weight on foot_left
            #self.sim.model.body_mass[7] *= 3.0  # Put more weight on foot_right

        self.randomization = randomization
        self.randomize_thigh = randomize_thigh
        self.randomize_leg = randomize_leg
        self.randomize_foot = randomize_foot

    def set_random_parameters(self):
        """Apply domain randomization to the walker masses."""
        masses = self.sample_parameters()
        self.set_parameters(masses)

    def sample_parameters(self):
        """
            Sample masses according to a domain randomization distribution.
            Modify the real element's true mass by summing/removing a value between -1 and 1.
        """
        """
            torso: masses[0]
            
            thigh_right: masses[1]
            thigh_left: masses[4]
    
            leg_right: masses[2]
            leg_left: masses[5]
        
            foot_right: masses[3]
            foot_left: masses[6]
        """
        
        masses = np.copy(self.original_masses)  # Copy the original masses
        
        # No randomization
        if self.randomization == 'none':
            return masses
        
        # List of indices to randomize
        indices_to_randomize = []
        
        # Complete randomization (no parameters specified)
        # If no parameters are specified, randomize all masses except the torso
        if not (self.randomize_thigh or self.randomize_leg or self.randomize_foot):
            indices_to_randomize = list(range(1, len(masses))) # Randomize all except the torso

            # Apply randomization to all masses
            for i in indices_to_randomize:
                if self.randomization == 'uniform':
                    masses[i] = np.random.uniform(self.original_masses[i] - 1, self.original_masses[i] + 1)
                elif self.randomization == 'normal':
                    masses[i] = truncnorm.rvs(-1, 1, loc=self.original_masses[i], scale=1)

        else: # Selective randomization
            if self.randomize_thigh:
                indices_to_randomize.append((1, 4)) # Index of thigh_left and thigh_right
            if self.randomize_leg:
                indices_to_randomize.append((2, 5)) # Index of leg_left and leg_right
            if self.randomize_foot:
                indices_to_randomize.append((3, 6)) # Index of foot_left and foot_right
        
            # Apply same randomization on couples of masses
            for (right_index, left_index) in indices_to_randomize:
                if self.randomization == 'uniform':
                    masses[right_index] = np.random.uniform(self.original_masses[right_index] - 1, self.original_masses[right_index] + 1)
                    masses[left_index] = masses[right_index]
                elif self.randomization == 'normal':
                    masses[right_index] = truncnorm.rvs(-1, 1, loc=self.original_masses[right_index], scale=1)
                    masses[left_index] = masses[right_index]
        
        # Keep the torso mass as the original value
        masses[0] = self.original_masses[0]
        
        return masses

    def get_parameters(self):
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, masses):
        self.sim.model.body_mass[1:] = masses

    def step(self, action):
        pos_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        pos_after, height, ang = self.sim.data.qpos[0:3]
        reward = (pos_after - pos_before) / self.dt
        reward += 1.0  # Alive bonus
        reward -= 1e-3 * np.square(action).sum()
        done = not (np.isfinite(self.state_vector()).all() and 
                    (height > 0.8) and 
                    (height < 2.0) and 
                    (abs(ang) < 1.0))
        return self._get_obs(), reward, done, {}
  
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()


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
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


"""
    Registered environments
"""

gym.envs.register(
    id="CustomWalker2D-v0",
    entry_point="__main__:CustomWalker2D",
    max_episode_steps=1000,
)

gym.envs.register(
    id="CustomWalker2D-source-v0",
    entry_point="__main__:CustomWalker2D",
    max_episode_steps=1000,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomWalker2D-target-v0",
    entry_point="__main__:CustomWalker2D",
    max_episode_steps=1000,
    kwargs={"domain": "target"}
)
