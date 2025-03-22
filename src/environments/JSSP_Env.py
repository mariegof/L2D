import gym
import numpy as np
from gym.utils import EzPickle
from src.utils.updateEntTimeLB import calEndTimeLB
from src.utils.permissibleLS import permissibleLeftShift
from src.utils.updateAdjMat import getActionNbghs
from src.features.feature_extractors import build_features
from src.rewards.reward_functions import REWARD_FUNCTIONS
import src.rewards.reward_functions as reward_functions
from Params import configs

class SJSSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m,
                 feature_set=None,
                 reward_strategy='default'):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        
        # Store reward strategy
        if reward_strategy not in REWARD_FUNCTIONS:
            print(f"Warning: Unknown reward strategy '{reward_strategy}'. Using default instead.")
            self.reward_strategy = 'default'
        else:
            self.reward_strategy = reward_strategy
        
        # Define available features for state representation
        self.available_features = {
            'LBs': False,                      # Lower bounds on operation end times
            'weighted_LBs': False,             # Lower bounds multiplied by job weights
            'finished_mark': False,            # Binary indicators of completed operations
            'weighted_priorities': False,      # WSPT ratio (weight/processing time)
            'normalized_weights': False,       # Direct job weights (normalized)
            'remaining_weighted_work': False,  # Weight × sum of remaining processing times
            'time_elapsed': False,             # Current timestep in scheduling
            'machine_contention': False,       # Number of operations needing each machine
            'weighted_priorities_normalized': False,  # Normalized WSPT ratio
            'last_op_weighted': False,         # Feature for normalized weights applied only to last operations
            'critical_path_contribution': False # Contribution to critical paths of weighted jobs
        }
        
        # Set default feature set if none specified
        if feature_set is None:
            feature_set = ['LBs', 'finished_mark', 'normalized_weights']
        
        # Store the feature set
        self.feature_set = feature_set
        
        # Enable selected features
        for feature in feature_set:
            if feature in self.available_features:
                self.available_features[feature] = True

    def done(self):
        """Check if all tasks are scheduled."""
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    def step(self, action):
        """Execute action and return new state, reward, done flag, and eligible operations."""
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:
            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1
                # This operation completes a job - update completion time record
                job_idx = action // self.number_of_machines
                # Get completion time directly from LBs (which is already updated correctly)
                job_completion_time = self.LBs[job_idx, self.number_of_machines-1]
                self.weighted_sum += self.weights[job_idx] * job_completion_time

            self.temp1[row, col] = startTime_a + dur_a
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0
                
        # Build features based on enabled feature set
        fea = build_features(self, self.feature_set)
        
        # Calculate reward using the selected reward function
        reward_method = getattr(reward_functions, REWARD_FUNCTIONS[self.reward_strategy])
        reward = reward_method(self, action)
            
        self.max_endTime = self.LBs.max()
        self.previous_weighted_sum = self._calculate_weighted_sum_estimate()  # Update for next step

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    def reset(self, data):
        """Reset environment with given problem instance data."""
        self.step_count = 0
        self.m = data[1]  # Machine matrix
        self.dur = data[0].astype(np.single)  # Duration matrix
        self.dur_cp = np.copy(self.dur)
        
        # Handle job weights (default to uniform weights if not provided)
        self.weights = data[2].astype(np.single) if len(data) > 2 else np.ones(self.number_of_jobs, dtype=np.single)
        
        # Initialize weighted sum tracking
        self.weighted_sum = 0
        
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # Initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # First column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # Last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # Initialize features
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single) 
        # Set initQuality to weighted sum rather than makespan
        self.initQuality = self._calculate_weighted_sum_estimate() if not configs.init_quality_flag else 0
        self.max_endTime = self.LBs.max()  # Keep this for compatibility
        self.previous_weighted_sum = self.initQuality
        if hasattr(self, 'current_potential'):
            delattr(self, 'current_potential')  # Remove to recalculate in potential-based reward
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)
        
        # Calculate WSPT priority for each operation 
        self.weighted_priorities = np.zeros((self.number_of_jobs, self.number_of_machines), dtype=np.single)
        for j in range(self.number_of_jobs):
            for m in range(self.number_of_machines):
                self.weighted_priorities[j, m] = self.weights[j] / self.dur[j, m]
        
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # Initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # Start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)
        
        # Build features based on enabled feature set
        fea = build_features(self, self.feature_set)

        return self.adj, fea, self.omega, self.mask
    
    def _calculate_weighted_sum_estimate(self):
        """
        Calculate current estimate of weighted sum objective.
        Uses lower bounds from LBs for all jobs, which automatically
        reflects actual completion times for finished jobs.
        """
        weighted_sum = 0
        for j in range(self.number_of_jobs):
            # Use LBs which automatically has the right value
            # for both completed and uncompleted jobs
            weighted_sum += self.weights[j] * self.LBs[j, self.number_of_machines-1]
        return weighted_sum