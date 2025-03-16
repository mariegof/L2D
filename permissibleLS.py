from Params import configs
import numpy as np


def permissibleLeftShiftWeighted(a, durMat, mchMat, mchsStartTimes, opIDsOnMchs, weights=None):
    """
    Determines the permissible left shift for a given operation in a scheduling problem.
    Parameters:
    a (int): The index of the operation to be scheduled.
    durMat (np.ndarray): A matrix containing the durations of all operations.
    mchMat (np.ndarray): A matrix containing the machine assignments for all operations.
    mchsStartTimes (list of lists): A list where each sublist contains the start times of operations on a specific machine.
    opIDsOnMchs (list of lists): A list where each sublist contains the operation IDs on a specific machine.
    Returns:
    tuple: A tuple containing:
        - startTime_a (int): The start time of the operation `a`.
        - flag (bool): A flag indicating whether the operation `a` was successfully left-shifted.
    """
    # Calculate times and find positions
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs)
    dur_a = np.take(durMat, a)
    mch_a = np.take(mchMat, a) - 1
    startTimesForMchOfa = mchsStartTimes[mch_a]
    opsIDsForMchOfa = opIDsOnMchs[mch_a]
    flag = False

    # Find positions where job is ready before operations start
    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        else:
            # NEW: Weight-aware position selection
            #print("In the weight-aware case")
            job_a = a // len(weights)  # Get job index for current operation
            weight_a = weights[job_a]  # Get weight of current job
            
            # Evaluate each legal position considering weights
            position_scores = []
            for pos_idx, pos in enumerate(legalPos):
                # Get operations that would be delayed
                delayed_ops = [op for op in opsIDsForMchOfa[pos:] if op >= 0]
                
                # Calculate weighted penalty for this position
                delay_penalty = 0
                for op in delayed_ops:
                    delayed_job = op // len(weights)
                    # Higher penalty for delaying higher-weight jobs
                    delay_penalty += weights[delayed_job] * dur_a
                    
                # Calculate position score (higher is better)
                # Balance between:
                # 1. Starting current operation early (weight_a * endTimesForPossiblePos[pos_idx])
                # 2. Minimizing impact on other jobs (delay_penalty)
                position_score = weight_a * (startTimesForMchOfa[-1] - endTimesForPossiblePos[pos_idx]) - delay_penalty
                position_scores.append(position_score)
                
                #print(f"Scheduling op {a} (Job {job_a}, Weight {weight_a})")
                #print(f"  Legal positions: {legalPos}")
                #print(f"  End times: {endTimesForPossiblePos}")
                
                #print(f"  Position {pos}: Start={endTimesForPossiblePos[pos_idx]}")
                #print(f"    Delayed ops: {delayed_ops}")
                #print(f"    Score: {position_score}")
            
            if len(position_scores) > 0 and max(position_scores) > 0:
                # Use position with best score if positive
                best_pos_idx = np.argmax(position_scores)
                flag = True
                startTime_a = putInBetween(a, idxLegalPos[best_pos_idx:best_pos_idx+1], 
                                         legalPos[best_pos_idx:best_pos_idx+1], 
                                         endTimesForPossiblePos[best_pos_idx:best_pos_idx+1], 
                                         startTimesForMchOfa, opsIDsForMchOfa)
            else:
                # If all positions have negative scores, put at end
                startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
            
    return startTime_a, flag

def permissibleLeftShift(a, durMat, mchMat, mchsStartTimes, opIDsOnMchs):
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs)
    dur_a = np.take(durMat, a)
    mch_a = np.take(mchMat, a) - 1
    startTimesForMchOfa = mchsStartTimes[mch_a]
    opsIDsForMchOfa = opIDsOnMchs[mch_a]
    flag = False

    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa)
    return startTime_a, flag

def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa):
    """
    Schedules a job at the earliest possible start time and updates the machine's schedule.

    This function finds the first available slot in the machine's schedule (indicated by a placeholder value of -configs.high),
    calculates the earliest start time for the job based on its ready time and the machine's ready time, and updates the 
    machine's schedule with the job's start time and operation ID.

    Args:
        a (int): The operation ID of the job to be scheduled.
        jobRdyTime_a (int): The ready time of the job.
        mchRdyTime_a (int): The ready time of the machine.
        startTimesForMchOfa (np.ndarray): An array representing the start times for the machine's operations.
        opsIDsForMchOfa (np.ndarray): An array representing the operation IDs for the machine's operations.

    Returns:
        int: The start time of the scheduled job.
    """
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = a
    return startTime_a


def calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    """
    Calculate legal positions for a job operation in a scheduling problem.

    Parameters:
    dur_a (int): Duration of the current job operation.
    jobRdyTime_a (int): Ready time of the current job.
    durMat (numpy.ndarray): Matrix containing durations of all operations.
    possiblePos (numpy.ndarray): Array of possible positions for the current job operation.
    startTimesForMchOfa (numpy.ndarray): Array of start times for all operations on the machine.
    opsIDsForMchOfa (numpy.ndarray): Array of operation IDs for all operations on the machine.

    Returns:
    tuple: A tuple containing:
        - idxLegalPos (numpy.ndarray): Indices of legal positions.
        - legalPos (numpy.ndarray): Legal positions for the current job operation.
        - endTimesForPossiblePos (numpy.ndarray): End times for the possible positions.
    """
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
    durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
    startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + np.take(durMat, [opsIDsForMchOfa[possiblePos[0]-1]]))
    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    """
    Inserts the operation 'a' into the earliest legal position in the machine's schedule.

    Parameters:
    a (int): The operation ID to be inserted.
    idxLegalPos (list of int): Indices of legal positions where the operation can be inserted.
    legalPos (list of int): Legal positions where the operation can be inserted.
    endTimesForPossiblePos (list of float): End times for the possible positions.
    startTimesForMchOfa (numpy array): Start times for the machine's operations.
    opsIDsForMchOfa (numpy array): Operation IDs for the machine's operations.

    Returns:
    float: The start time of the inserted operation.
    """
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs):
    """
    Calculate the ready times for a job and its corresponding machine.
    Parameters:
    a (int): The index of the current job operation.
    mchMat (numpy.ndarray): A matrix where each element represents the machine assigned to a job operation.
    durMat (numpy.ndarray): A matrix where each element represents the duration of a job operation.
    mchsStartTimes (numpy.ndarray): A matrix where each row represents the start times of operations on a machine.
    opIDsOnMchs (numpy.ndarray): A matrix where each row represents the operation IDs assigned to a machine.
    Returns:
    tuple: A tuple containing:
        - jobRdyTime_a (int): The ready time for the job.
        - mchRdyTime_a (int): The ready time for the machine.
    """
    mch_a = np.take(mchMat, a) - 1
    # cal jobRdyTime_a
    jobPredecessor = a - 1 if a % mchMat.shape[1] != 0 else None
    if jobPredecessor is not None:
        durJobPredecessor = np.take(durMat, jobPredecessor)
        mchJobPredecessor = np.take(mchMat, jobPredecessor) - 1
        jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()
    else:
        jobRdyTime_a = 0
    # cal mchRdyTime_a
    mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None
    if mchPredecessor is not None:
        durMchPredecessor = np.take(durMat, mchPredecessor)
        mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()
    else:
        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a


if __name__ == "__main__":
    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen
    import time

    n_j = 3
    n_m = 3
    low = 1
    high = 99
    SEED = 10
    np.random.seed(SEED)
    
    # Generate instance
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
    dur = data[0]
    mch = data[1]
    
    # Define weights - high contrast to see differences
    weights = np.array([10, 3, 1], dtype=np.single)
    
    print('Dur')
    print(dur)
    print('Mach')
    print(mch)
    print('Weights')
    print(weights)
    print()

    # Create environment to determine eligible operations
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # === TEST ORIGINAL PLS ===
    print("\n=== TESTING ORIGINAL PLS ===")
    # Initialize scheduling structures
    mchsStartTimes_std = -configs.high * np.ones_like(dur.transpose(), dtype=np.int32)
    opIDsOnMchs_std = -n_j * np.ones_like(dur.transpose(), dtype=np.int32)
    
    # Get initial eligible operations
    _, _, omega, mask = env.reset(data)
    actions_taken = []
    job_completions_std = np.zeros(n_j, dtype=np.single)
    
    # Run through all operations using eligibility from environment
    while not env.done():
        # Choose an eligible action (use the first one for determinism)
        eligible_ops = omega[np.where(mask == 0)]
        action = eligible_ops[0]  # Select first eligible operation
        actions_taken.append(action)
        
        job_idx = action // n_m
        op_idx = action % n_m
        
        print(f"Scheduling op {action} (Job {job_idx})")
        
        # Apply standard PLS
        startTime_a, flag = permissibleLeftShift(
            a=action, 
            durMat=dur, 
            mchMat=mch, 
            mchsStartTimes=mchsStartTimes_std, 
            opIDsOnMchs=opIDsOnMchs_std
        )
        
        # Update completion time if last operation of job
        if op_idx == n_m - 1:
            job_completions_std[job_idx] = startTime_a + dur[job_idx, op_idx]
        
        print(f"  Start time: {startTime_a}")
        
        # Update environment to get next eligible operations
        adj, _, reward, done, omega, mask = env.step(action)
    
    # Calculate weighted sum
    weighted_sum_std = np.sum(weights * job_completions_std)
    print(f"\nJob completion times: {job_completions_std}")
    print(f"Weighted sum: {weighted_sum_std}")
    
    # === TEST WEIGHTED PLS ===
    print("\n=== TESTING WEIGHTED PLS ===")
    # Reset environment
    env = SJSSP(n_j=n_j, n_m=n_m)
    _, _, omega, mask = env.reset(data)
    
    # Initialize scheduling structures
    mchsStartTimes_w = -configs.high * np.ones_like(dur.transpose(), dtype=np.int32)
    opIDsOnMchs_w = -n_j * np.ones_like(dur.transpose(), dtype=np.int32)
    
    job_completions_w = np.zeros(n_j, dtype=np.single)
    action_index = 0
    
    # Use the same sequence of operations from the first run
    while not env.done():
        action = actions_taken[action_index]
        action_index += 1
        
        job_idx = action // n_m
        op_idx = action % n_m
        
        print(f"Scheduling op {action} (Job {job_idx}, Weight {weights[job_idx]})")
        
        # Apply weighted PLS
        startTime_a, flag = permissibleLeftShiftWeighted(
            a=action, 
            durMat=dur, 
            mchMat=mch, 
            mchsStartTimes=mchsStartTimes_w, 
            opIDsOnMchs=opIDsOnMchs_w,
            weights=weights
        )
        
        # Update completion time if last operation of job
        if op_idx == n_m - 1:
            job_completions_w[job_idx] = startTime_a + dur[job_idx, op_idx]
        
        print(f"  Start time: {startTime_a}")
        
        # Update environment to maintain consistent eligible operations
        adj, _, reward, done, omega, mask = env.step(action)
    
    # Calculate weighted sum
    weighted_sum_w = np.sum(weights * job_completions_w)
    print(f"\nJob completion times: {job_completions_w}")
    print(f"Weighted sum: {weighted_sum_w}")
    
    # Compare results
    print("\n=== RESULTS COMPARISON ===")
    print(f"Original PLS weighted sum: {weighted_sum_std}")
    print(f"Weighted PLS weighted sum: {weighted_sum_w}")
    
    if weighted_sum_std > weighted_sum_w:
        improvement = (weighted_sum_std - weighted_sum_w) / weighted_sum_std * 100
        print(f"Improvement with weighted PLS: {improvement:.2f}%")
    else:
        print("No improvement with weighted PLS on this instance.")
    
    # Visualize final schedules
    print("\n=== SCHEDULE VISUALIZATION ===")
    print("Original PLS Schedule:")
    durAlongMchs_std = np.take(dur, opIDsOnMchs_std)
    mchsEndTimes_std = mchsStartTimes_std + durAlongMchs_std
    
    for m in range(n_m):
        print(f"Machine {m+1}:", end=" ")
        for i in range(n_j):
            if opIDsOnMchs_std[m][i] >= 0:
                op = opIDsOnMchs_std[m][i]
                job_idx = op // n_m
                start = mchsStartTimes_std[m][i]
                end = mchsEndTimes_std[m][i]
                print(f"[J{job_idx}(w={weights[job_idx]}): {start}-{end}]", end=" ")
        print()
    
    print("\nWeighted PLS Schedule:")
    durAlongMchs_w = np.take(dur, opIDsOnMchs_w)
    mchsEndTimes_w = mchsStartTimes_w + durAlongMchs_w
    
    for m in range(n_m):
        print(f"Machine {m+1}:", end=" ")
        for i in range(n_j):
            if opIDsOnMchs_w[m][i] >= 0:
                op = opIDsOnMchs_w[m][i]
                job_idx = op // n_m
                start = mchsStartTimes_w[m][i]
                end = mchsEndTimes_w[m][i]
                print(f"[J{job_idx}(w={weights[job_idx]}): {start}-{end}]", end=" ")
        print()