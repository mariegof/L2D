import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs

# Registry of available reward functions
REWARD_FUNCTIONS = {
    'default': 'weighted_sum_difference',
    'wspt_guided': 'wspt_guided_reward',
    'potential_based': 'potential_based_reward',
    'critical_path': 'critical_path_reward',
    'wspt': 'wspt_reward',
    'weighted_delay': 'weighted_delay_reward',
    'direct_weighted_sum': 'direct_weighted_sum_reward',
    'hurl': 'hurl_reward',
    'look_ahead': 'look_ahead_reward'
}
reward_strategy = 'critical_path'  # Try: 'default', 'wspt_guided', 'potential_based', 'critical_path'

class SJSSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m,
                 feature_set=None):
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
            'weighted_LBs': False,            # Lower bounds multiplied by job weights (NEW)
            'finished_mark': False,            # Binary indicators of completed operations
            'weighted_priorities': False,      # WSPT ratio (weight/processing time)
            'normalized_weights': False,      # Direct job weights (normalized)
            'remaining_weighted_work': False, # Weight × sum of remaining processing times
            'time_elapsed': False,            # Current timestep in scheduling
            'machine_contention': False,       # Number of operations needing each machine
            'weighted_priorities_normalized': False,  # Normalized WSPT ratio
            'last_op_weighted': False,          # Feature for normalized weights applied only to last operations
            'critical_path_contribution': False  # Contribution to critical paths of weighted jobs
        }
        
        # Set default feature set for weighted sum if none specified
        if feature_set is None:
            feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights']
        
        # Enable selected features
        for feature in feature_set:
            if feature in self.available_features:
                self.available_features[feature] = True
                
        # HuRL specific parameters
        self.current_iteration = 0  # Track current training iteration
        self.max_iterations = 10000  # Default max iterations for normalization
        self.hurl_lambda = 0.2  # Initial lambda value
        self.hurl_lambda_final = 1.0  # Final lambda value
        self.original_discount = configs.gamma  # Store original discount factor

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
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
        fea = self._build_features()
        
        # Calculate reward using the selected reward function
        reward_method = getattr(self, REWARD_FUNCTIONS[self.reward_strategy])
        reward = reward_method(action)
            
        self.max_endTime = self.LBs.max()
        self.previous_weighted_sum = self._calculate_weighted_sum_estimate()  # Update for next step

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):
        self.step_count = 0
        self.m = data[1]
        self.dur = data[0].astype(np.single)
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
        fea = self._build_features()

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
        
        # Get maximum weight for normalization
        max_weight = np.max(self.weights)
        
        # Lower bounds on operation end times
        if self.available_features['LBs']:
            features.append(self.LBs.reshape(-1, 1)/configs.et_normalize_coef)
            
        # Binary indicators of completed operations
        if self.available_features['finished_mark']:
            features.append(self.finished_mark.reshape(-1, 1))
            
        # Weighted lower bounds (LBs multiplied by job weights)
        if self.available_features['weighted_LBs']:
            # Create weighted LBs matrix
            weighted_LBs = np.zeros_like(self.LBs, dtype=np.single)
            for j in range(self.number_of_jobs):
                # Apply job's weight to all of its operations' LBs
                weighted_LBs[j, :] = self.weights[j] * self.LBs[j, :]
            
            # Add normalized weighted LBs as feature
            # Use higher normalization coefficient since values are larger
            norm_factor = configs.et_normalize_coef * np.max(self.weights) if np.max(self.weights) > 1 else configs.et_normalize_coef
            features.append(weighted_LBs.reshape(-1, 1)/norm_factor)
            
        if self.available_features['weighted_priorities_normalized']:
            # Normalize WSPT ratio by maximum value
            norm_wspt = self.weighted_priorities.copy()
            if np.max(norm_wspt) > 0:
                norm_wspt = norm_wspt / np.max(norm_wspt)
            features.append(norm_wspt.reshape(-1, 1))
        
        # WSPT ratio (weight/processing time)
        if self.available_features['weighted_priorities']:
            features.append(self.weighted_priorities.reshape(-1, 1))
        
        # Direct job weights (normalized)
        if self.available_features['normalized_weights']:
            norm_weights = self.weights / np.max(self.weights)
            features.append(np.repeat(norm_weights, self.number_of_machines).reshape(-1, 1))
            
        # Feature for normalized weights applied only to last operations
        if self.available_features['last_op_weighted']:
            last_op_weighted = np.zeros_like(self.LBs, dtype=np.single)
            for j in range(self.number_of_jobs):
                # Only apply weight to the last operation of each job
                last_op_weighted[j, self.number_of_machines-1] = self.weights[j] / np.max(self.weights)
            features.append(last_op_weighted.reshape(-1, 1))
        
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
            time_feature = np.ones((self.number_of_tasks, 1), dtype=np.float32) * (curr_time / configs.high)
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
        
        # Calculate each operation's contribution to critical paths of weighted jobs
        if self.available_features['critical_path_contribution']:
            contributions = np.zeros((self.number_of_jobs, self.number_of_machines), dtype=np.single)
    
            for j in range(self.number_of_jobs):
                # Skip completed jobs
                if np.all(self.finished_mark[j, :] == 1):
                    continue
                    
                # Find remaining operations for this job
                remaining_ops = np.where(self.finished_mark[j, :] == 0)[0]
                if len(remaining_ops) == 0:
                    continue
                    
                # Weight the contribution by job weight
                weight_factor = self.weights[j] / np.max(self.weights)
                
                # Mark all remaining operations with their weighted contribution
                for op in remaining_ops:
                    # Operations earlier in the job's sequence have higher contribution
                    position_factor = 1.0 - (op / self.number_of_machines)
                    contributions[j, op] = weight_factor * position_factor
        
            features.append(contributions.reshape(-1, 1))
            
        return np.concatenate(features, axis=1)
    
    def weighted_sum_difference(self, action):
        """Default reward function: difference in weighted sum estimate."""
        current_weighted_sum = self._calculate_weighted_sum_estimate()
        reward = -(current_weighted_sum - self.previous_weighted_sum)
        
        # Add small positive reward when needed
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        
        return reward

    def wspt_guided_reward(self, action):
        """WSPT-guided reward function."""
        # Store previous weighted sum for improvement calculation
        previous_weighted_sum = self.previous_weighted_sum
        current_weighted_sum = self._calculate_weighted_sum_estimate()
        
        # Basic improvement reward
        improvement = previous_weighted_sum - current_weighted_sum
        
        # Get job index and operation index
        job_idx = action // self.number_of_machines
        op_idx = action % self.number_of_machines
        
        # Calculate WSPT ratio for the selected operation
        wspt_ratio = self.weights[job_idx] / self.dur[job_idx, op_idx]
        
        # Calculate WSPT ratios for all eligible operations
        eligible_ops = np.where(~self.mask)[0]
        wspt_ratios = []
        
        for op in eligible_ops:
            j_idx = op // self.number_of_machines
            m_idx = op % self.number_of_machines
            wspt_ratios.append(self.weights[j_idx] / self.dur[j_idx, m_idx])
        
        # WSPT reward component (normalized between -1 and 1)
        if len(wspt_ratios) > 0:
            max_wspt = max(wspt_ratios)
            min_wspt = min(wspt_ratios)
            wspt_range = max_wspt - min_wspt if max_wspt != min_wspt else 1.0
            normalized_wspt = (wspt_ratio - min_wspt) / wspt_range if wspt_range != 0 else 0.5
            wspt_reward = 2 * normalized_wspt - 1  # Scale to [-1, 1]
        else:
            wspt_reward = 0
        
        # Combined reward (adjust alpha to balance short vs long-term signals)
        alpha = 0.7  # Emphasis on WSPT principle vs actual weighted sum improvement
        reward = alpha * wspt_reward + (1-alpha) * improvement
        
        # Add small positive reward when needed
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
            
        return reward

    def potential_based_reward(self, action):
        """Job completion potential-based reward."""
        # Store state before taking action
        previous_weighted_sum = self.previous_weighted_sum
        current_weighted_sum = self._calculate_weighted_sum_estimate()
        
        # Get job info
        job_idx = action // self.number_of_machines
        
        # Calculate immediate weighted sum improvement
        improvement = previous_weighted_sum - current_weighted_sum
        
        # Calculate the "potential" of the current state
        potential = 0
        for j in range(self.number_of_jobs):
            remaining_ops = 0
            remaining_time = 0
            for m in range(self.number_of_machines):
                if self.finished_mark[j, m] == 0:  # If operation not completed
                    remaining_ops += 1
                    remaining_time += self.dur[j, m]
            
            # Jobs with higher weights and less remaining work have higher potential
            if remaining_ops > 0:
                potential -= (self.weights[j] * remaining_time / remaining_ops)
        
        # Store current potential for next step
        if not hasattr(self, 'current_potential'):
            self.current_potential = potential
        
        # Calculate potential change
        potential_reward = potential - self.current_potential
        self.current_potential = potential  # Update for next step
        
        # Final reward combines immediate improvement and potential change
        reward = improvement + 0.5 * potential_reward
        
        # Add small positive reward when needed
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        
        return reward

    def critical_path_reward(self, action):
        """Critical path weight reward."""
        # Store previous weighted sum
        previous_weighted_sum = self.previous_weighted_sum
        current_weighted_sum = self._calculate_weighted_sum_estimate()
        
        # Get job info
        job_idx = action // self.number_of_machines
        op_idx = action % self.number_of_machines
        weight = self.weights[job_idx]
        
        # Calculate immediate improvement
        improvement = previous_weighted_sum - current_weighted_sum
        
        # Check if this completes a job
        job_completion_bonus = 0
        if action in self.last_col:  # If this is the last operation of a job
            # Reward for completing high-weight jobs earlier
            completion_time = self.max_endTime + self.dur[job_idx, op_idx]
            job_completion_bonus = weight / max(1, completion_time)  # Avoid division by zero
        
        # Calculate weighted slack for operations, slack is the weighted critical path
        slack = 0
        for j in range(self.number_of_jobs):
            # Skip completed jobs
            if np.all(self.finished_mark[j, :] == 1):
                continue
                
            # Calculate operations remaining for this job
            remaining_ops = np.sum(self.finished_mark[j, :] == 0)
                            
            # If job has pending operations, calculate its critical path contribution
            if remaining_ops > 0:
                job_weight = self.weights[j]
                slack += job_weight * (remaining_ops / self.number_of_machines)
        
        # Critical path reward - prioritize operations that reduce weighted critical path
        critical_path_reward = weight / max(1, slack) if slack > 0 else 0
        
        # Combined reward
        reward = improvement + 0.3 * job_completion_bonus + 0.3 * critical_path_reward
        
        # Add small positive reward when needed
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        
        return reward
    
    def wspt_reward(self, action):
        """Simple WSPT-based reward: job_weight/job_processing_time"""
        job_idx = action // self.number_of_machines
        op_idx = action % self.number_of_machines
        
        # Calculate the WSPT ratio
        weight = self.weights[job_idx]
        processing_time = self.dur[job_idx, op_idx]
        
        # WSPT ratio as reward
        wspt_ratio = weight / max(1, processing_time)  # Avoid division by zero
        
        # Scale the reward to be comparable to other rewards
        reward = wspt_ratio * 10  # Adjust scaling factor as needed
        
        return reward

    def weighted_delay_reward(self, action):
        """Reward based on -sum(job_weight*(operation_completion_time - operation_LB))"""
        # Calculate the weighted delay for this operation
        job_idx = action // self.number_of_machines
        op_idx = action % self.number_of_machines
        weight = self.weights[job_idx]
        
        # Get the current completion time and lower bound
        actual_completion_time = self.temp1[job_idx, op_idx]
        lower_bound = self.LBs[job_idx, op_idx] - self.dur[job_idx, op_idx]  # LB start time
        
        # Calculate weighted delay
        weighted_delay = weight * (actual_completion_time - lower_bound)
        
        # Return negative delay as reward (less delay = higher reward)
        reward = -weighted_delay
        
        return reward

    def direct_weighted_sum_reward(self, action):
        """Direct reward based on -sum(job_weight*operation_completion_time)"""
        # Calculate the contribution to weighted sum for this operation
        job_idx = action // self.number_of_machines
        op_idx = action % self.number_of_machines
        weight = self.weights[job_idx]
        
        # Get the completion time
        completion_time = self.temp1[job_idx, op_idx]
        
        # Calculate weighted completion time
        weighted_completion = weight * completion_time
        
        # Return negative weighted completion as reward (earlier = higher reward)
        reward = -weighted_completion
        
        # For the last operation in each job, give an additional reward
        if action in self.last_col:
            # This directly affects the objective function
            job_completion_time = completion_time
            reward *= 2  # Double the reward for completing jobs
        
        return reward
    
    def hurl_reward(self, action):
        """
        Heuristic-Guided Reinforcement Learning reward function.
        Implements reward reshaping according to the HuRL paper.
        """
        # Get basic information about the action
        job_idx = action // self.number_of_machines
        op_idx = action % self.number_of_machines
        weight = self.weights[job_idx]
        
        # Completion time of the operation after scheduling
        completion_time = self.temp1[job_idx, op_idx]
        
        # Original reward component (weighted completion time)
        # Negated because we want to minimize this value
        original_reward = -weight * completion_time
        
        # The heuristic function f(s')
        # Using WSPT (weight/processing_time) which is a good heuristic for weighted sum problems
        processing_time = self.dur[job_idx, op_idx]
        wspt_heuristic = weight / max(1, processing_time)
        
        # Scale the heuristic appropriately (WSPT values and completion times have different scales)
        scaled_heuristic = wspt_heuristic * 100
        
        # Calculate the reshaped reward: r̃(s,a) = r(s,a) + (1-λ)γE[f(s')]
        # Note: The original discount factor γ is used in the reshaping
        heuristic_term = (1 - self.hurl_lambda) * self.original_discount * scaled_heuristic
        reshaped_reward = original_reward + heuristic_term
        
        return reshaped_reward
        
    def look_ahead_reward(self, action):
        """Reward function that encourages strategic deviation from WSPT when beneficial"""
        # Basic weighted completion time component
        job_idx = action // self.number_of_machines
        op_idx = action % self.number_of_machines
        
        # Calculate immediate WSPT ratio
        wspt_ratio = self.weights[job_idx] / self.dur[job_idx, op_idx]
        
        # Calculate WSPT ratios for all eligible operations
        eligible_ops = np.where(~self.mask)[0]
        best_wspt_op = None
        best_wspt_ratio = 0
        
        for op in eligible_ops:
            j = op // self.number_of_machines
            m = op % self.number_of_machines
            ratio = self.weights[j] / self.dur[j, m]
            if ratio > best_wspt_ratio:
                best_wspt_ratio = ratio
                best_wspt_op = op
        
        # Base reward on weighted completion time
        reward = -self.weights[job_idx] * self.temp1[job_idx, op_idx]
        
        # If we're choosing an operation other than the WSPT choice
        if action != best_wspt_op and best_wspt_op is not None:
            # Examine if this leads to better machine utilization
            machine_id = self.m[job_idx, op_idx] - 1
            
            # Calculate how this choice affects future operations on this machine
            future_benefit = 0
            for j in range(self.number_of_jobs):
                # If job has future operations on this machine
                future_ops = [(j, m) for m in range(self.number_of_machines) 
                            if self.m[j, m] - 1 == machine_id and 
                            self.finished_mark[j, m] == 0]
                
                for future_j, future_m in future_ops:
                    # Higher weight jobs get priority in the future benefit calculation
                    future_benefit += self.weights[future_j] * 10
            
            # Add look-ahead bonus if this non-WSPT choice benefits future high-weight operations
            if future_benefit > self.weights[job_idx] * 5:
                reward += 200  # Significant bonus for strategic deviation from WSPT
        
        return reward
    
    def update_hurl_parameters(self, iteration, max_iterations):
        """
        Update HuRL parameters based on current training iteration.
        
        Args:
            iteration: Current training iteration
            max_iterations: Maximum number of training iterations
        """
        self.current_iteration = iteration
        self.max_iterations = max_iterations
        
        # Calculate lambda using a schedule that increases over time
        # This smoothly transitions from λ=0.2 to λ=1.0
        progress = min(iteration / max_iterations, 1.0)
        self.hurl_lambda = 0.2 + (self.hurl_lambda_final - 0.2) * progress
        
        # The effective discount factor for HuRL: γ̃ = λγ
        self.effective_discount = self.hurl_lambda * self.original_discount
        
        return self.hurl_lambda, self.effective_discount