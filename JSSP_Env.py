import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs


class SJSSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m,
                 feature_set=None
                 ):
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
        
        # Define available features for state representation
        self.available_features = {
            'LBs': True,                      # Lower bounds on operation end times
            'finished_mark': True,            # Binary indicators of completed operations
            'weighted_priorities': True,      # WSPT ratio (weight/processing time)
            'normalized_weights': False,      # Direct job weights (normalized)
            'remaining_weighted_work': False, # Weight × sum of remaining processing times
            'time_elapsed': False,            # Current timestep in scheduling
            'machine_contention': False       # Number of operations needing each machine
        }
        
        # Set default feature set for weighted sum if none specified
        if feature_set is None:
            feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights']
        
        # Enable selected features
        for feature in feature_set:
            if feature in self.available_features:
                self.available_features[feature] = True

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        # Calculate current weighted sum estimate for reward
        current_weighted_sum = self._calculate_weighted_sum_estimate()
        #print(f"Before action {action}: weighted_sum = {current_weighted_sum}")
        
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
                job_completion_time = startTime_a + dur_a  # End time of last operation
                self.job_completion_times[job_idx] = job_completion_time
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

        # Calculate new weighted sum estimate
        new_weighted_sum = self._calculate_weighted_sum_estimate()
        #print(f"After action {action}: weighted_sum = {new_weighted_sum}")
        
        # Reward is the improvement in quality measure
        reward = current_weighted_sum - new_weighted_sum
        #print(f"Reward: {reward} (current - new = {current_weighted_sum} - {new_weighted_sum})")
        
        # Add small positive reward when needed
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
            #print(f"Reward after rewardscale: {reward} (current - new = {current_weighted_sum} - {new_weighted_sum})")
            
        self.max_endTime = self.LBs.max()

        # Build features based on enabled feature set
        fea = self._build_features()

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):
        self.step_count = 0
        self.m = data[1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        
        # Handle job weights (default to uniform weights if not provided)
        self.weights = data[2].astype(np.single) if len(data) > 2 else np.ones(self.number_of_jobs, dtype=np.single)
        
        # Initialize job completion times and weighted sum tracking
        self.job_completion_times = np.zeros(self.number_of_jobs, dtype=np.single)
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
        
        # Calculate initial weighted sum estimate for initQuality
        self.temp1 = np.zeros_like(self.dur, dtype=np.single)
        init_weighted_sum = 0
        for j in range(self.number_of_jobs):
            init_weighted_sum += self.weights[j] * self.LBs[j, self.number_of_machines - 1]
            
        # Set initQuality to weighted sum rather than makespan
        self.initQuality = init_weighted_sum if not configs.init_quality_flag else 0
        self.max_endTime = self.LBs.max()  # Keep this for compatibility
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)
        
        # Calculate WSPT priority for each operation
        self.weighted_priorities = np.zeros((self.number_of_jobs, self.number_of_machines), dtype=np.single)
        for j in range(self.number_of_jobs):
            for m in range(self.number_of_machines):
                self.weighted_priorities[j, m] = self.weights[j] / self.dur[j, m]

        # Build features based on enabled feature set
        fea = self._build_features()
        
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # Initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # Start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        return self.adj, fea, self.omega, self.mask
    
    def _calculate_weighted_sum_estimate(self):
        """
        Calculate current estimate of weighted sum objective.
        Uses lower bounds for uncompleted jobs and actual completion times for completed jobs.
        """
        weighted_sum = 0
        for j in range(self.number_of_jobs):
            if self.job_completion_times[j] > 0:
                # Use actual completion time for finished jobs
                weighted_sum += self.weights[j] * self.job_completion_times[j]
            else:
                # Use lower bound for unfinished jobs
                weighted_sum += self.weights[j] * self.LBs[j, self.number_of_machines-1]
        return weighted_sum
    
    def _calculate_remaining_work(self):
        """
        Calculate remaining processing time for each job weighted by job importance.
        Higher values indicate more critical jobs to complete.
        """
        remaining_work = np.zeros(self.number_of_jobs, dtype=np.single)
        for j in range(self.number_of_jobs):
            total_remaining = 0
            for m in range(self.number_of_machines):
                if self.finished_mark[j, m] == 0:  # If operation not completed
                    total_remaining += self.dur[j, m]
            remaining_work[j] = self.weights[j] * total_remaining
        return remaining_work
    
    def _calculate_machine_contention(self):
        """
        Calculate how many unscheduled operations need each machine.
        Higher values indicate more contested machines.
        """
        machine_contention = np.zeros(self.number_of_machines, dtype=np.single)
        for j in range(self.number_of_jobs):
            for m in range(self.number_of_machines):
                if self.finished_mark[j, m] == 0:  # If operation not completed
                    machine_id = self.m[j, m] - 1  # Machine required for this operation
                    machine_contention[machine_id] += 1
        return machine_contention
    
    def _build_features(self):
        """
        Build state features based on enabled feature set.
        Returns a concatenated feature vector for all operations.
        """
        features = []
        
        # Lower bounds on operation end times
        if self.available_features['LBs']:
            features.append(self.LBs.reshape(-1, 1)/configs.et_normalize_coef)
        
        # Binary indicators of completed operations
        if self.available_features['finished_mark']:
            features.append(self.finished_mark.reshape(-1, 1))
        
        # WSPT ratio (weight/processing time)
        if self.available_features['weighted_priorities']:
            features.append(self.weighted_priorities.reshape(-1, 1))
        
        # Direct job weights (normalized)
        if self.available_features['normalized_weights']:
            norm_weights = self.weights / np.max(self.weights)
            features.append(np.repeat(norm_weights, self.number_of_machines).reshape(-1, 1))
        
        # Weight × sum of remaining processing times
        if self.available_features['remaining_weighted_work']:
            remaining_work = self._calculate_remaining_work()
            # Normalize by maximum remaining work
            if np.max(remaining_work) > 0:
                remaining_work = remaining_work / np.max(remaining_work)
            features.append(np.repeat(remaining_work, self.number_of_machines).reshape(-1, 1))
        
        # Current timestep in scheduling
        if self.available_features['time_elapsed']:
            # Use current maximum end time as a proxy for elapsed time
            curr_time = np.max(self.temp1) if np.max(self.temp1) > 0 else 0
            time_feature = np.ones((self.number_of_tasks, 1)) * (curr_time / configs.high)
            features.append(time_feature)
        
        # Number of operations needing each machine
        if self.available_features['machine_contention']:
            machine_contention = self._calculate_machine_contention()
            # Create a feature that maps each operation to its machine's contention
            machine_feature = np.zeros((self.number_of_jobs, self.number_of_machines), dtype=np.single)
            for j in range(self.number_of_jobs):
                for m in range(self.number_of_machines):
                    machine_id = self.m[j, m] - 1
                    machine_feature[j, m] = machine_contention[machine_id]
            # Normalize by maximum contention
            if np.max(machine_feature) > 0:
                machine_feature = machine_feature / np.max(machine_feature)
            features.append(machine_feature.reshape(-1, 1))
        
        return np.concatenate(features, axis=1)