import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

def create_colormap():
    """Create a custom colormap with distinct colors for jobs"""
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939',
        '#8c6d31', '#843c39', '#5254a3', '#6b4c9a', '#8ca252', '#bd9e39'
    ]
    return mcolors.ListedColormap(colors)

def plot_schedule(times, machines, operations_order, job_weights=None, title=None):
    """
    Plot a Gantt chart for a job shop schedule
    
    Args:
        times: Matrix of processing times
        machines: Matrix of machine assignments
        operations_order: List of operations in the order they were scheduled
        job_weights: Optional weights for each job (for display purposes)
        title: Optional title for the chart
    """
    n_jobs, n_machines = times.shape
    
    # Extract job completion data from operations_order
    machine_schedules = {}
    job_completion_times = np.zeros(n_jobs)
    
    current_time = np.zeros(n_machines)
    
    # Process each operation in the scheduled order
    for op in operations_order:
        job_idx = op // n_machines
        machine_idx = machines[job_idx, op % n_machines] - 1  # Machine indices start from 1
        proc_time = times[job_idx, op % n_machines]
        
        # Find the earliest start time (considering both job precedence and machine availability)
        if op % n_machines == 0:  # First operation of the job
            earliest_start = 0
        else:
            # Previous operation in the same job
            prev_op = op - 1
            prev_machine = machines[job_idx, prev_op % n_machines] - 1
            earliest_start = machine_schedules.get((prev_machine, prev_op), (0, 0, 0))[2]  # End time of previous op
        
        # Consider machine availability
        start_time = max(earliest_start, current_time[machine_idx])
        end_time = start_time + proc_time
        
        # Update current time for this machine
        current_time[machine_idx] = end_time
        
        # Store operation schedule: (start_time, duration, end_time, job_id)
        machine_schedules[(machine_idx, op)] = (start_time, proc_time, end_time, job_idx)
        
        # Update job completion time if this is the last operation
        if op % n_machines == n_machines - 1:
            job_completion_times[job_idx] = end_time
    
    # Calculate weighted sum objective
    weighted_sum = 0
    if job_weights is not None:
        weighted_sum = np.sum(job_weights * job_completion_times)
    
    # Create the Gantt chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colormap = create_colormap()
    
    # Plot each operation
    for (machine_idx, op), (start, duration, end, job_idx) in machine_schedules.items():
        color_idx = job_idx % colormap.N
        ax.barh(
            machine_idx,
            duration,
            left=start,
            height=0.8,
            color=colormap(color_idx),
            edgecolor='black',
            alpha=0.8
        )
        
        # Add operation label
        ax.text(
            start + duration/2,
            machine_idx,
            f"J{job_idx+1}-{op%n_machines+1}",
            ha='center',
            va='center',
            color='black',
            fontsize=8
        )
    
    # Add job completion markers
    for job_idx, completion_time in enumerate(job_completion_times):
        ax.axvline(x=completion_time, color=colormap(job_idx % colormap.N), linestyle='--', alpha=0.5)
    
    # Set chart properties
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(n_machines)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    
    # Set chart title
    if title:
        chart_title = title
    else:
        chart_title = f'Job Shop Schedule - Weighted Sum: {weighted_sum:.2f}'
    ax.set_title(chart_title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add weight information if available
    if job_weights is not None:
        weight_text = "Job Weights: " + ", ".join([f"J{i+1}={w}" for i, w in enumerate(job_weights)])
        plt.figtext(0.5, 0.01, weight_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig, ax, weighted_sum

def compare_schedules(instance, methods_results, method_names=None, weighted_sums=None):
    """
    Compare schedules from different methods side by side
    
    Args:
        instance: Tuple of (times, machines, weights)
        methods_results: List of operation sequences from different methods
        method_names: Optional list of method names
        weighted_sums: Optional list of pre-calculated weighted sums
    """
    times, machines, weights = instance
    n_methods = len(methods_results)
    
    if method_names is None:
        method_names = [f"Method {i+1}" for i in range(n_methods)]
    
    if weighted_sums is None:
        weighted_sums = [0] * n_methods  # Default to zeros if not provided
    
    # Create subplots
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 5*n_methods))
    if n_methods == 1:
        axes = [axes]
    
    result_weighted_sums = []
    
    # Plot each method's schedule
    for i, (method_name, operations, ws) in enumerate(zip(method_names, methods_results, weighted_sums)):
        # Use the provided weighted sum instead of calculating it
        _, _, calculated_ws = plot_schedule(
            times, machines, operations, weights, 
            title=f"{method_name} - Weighted Sum: {ws:.2f}"
        )
        result_weighted_sums.append(calculated_ws)
        plt.close()  # Close the individual figure
        
        # Create a new figure for the subplot
        colormap = create_colormap()
        n_jobs, n_machines = times.shape
        
        # Extract schedule data (simplified for subplots)
        machine_schedules = {}
        job_completion_times = np.zeros(n_jobs)
        current_time = np.zeros(n_machines)
        
        # Process operations
        for op in operations:
            job_idx = op // n_machines
            machine_idx = machines[job_idx, op % n_machines] - 1
            proc_time = times[job_idx, op % n_machines]
            
            # Find earliest start time
            if op % n_machines == 0:
                earliest_start = 0
            else:
                prev_op = op - 1
                prev_machine = machines[job_idx, prev_op % n_machines] - 1
                earliest_start = machine_schedules.get((prev_machine, prev_op), (0, 0, 0))[2]
            
            start_time = max(earliest_start, current_time[machine_idx])
            end_time = start_time + proc_time
            current_time[machine_idx] = end_time
            machine_schedules[(machine_idx, op)] = (start_time, proc_time, end_time, job_idx)
            
            if op % n_machines == n_machines - 1:
                job_completion_times[job_idx] = end_time
        
        # Plot on the corresponding subplot
        for (machine_idx, op), (start, duration, end, job_idx) in machine_schedules.items():
            color_idx = job_idx % colormap.N
            axes[i].barh(
                machine_idx,
                duration,
                left=start,
                height=0.8,
                color=colormap(color_idx),
                edgecolor='black',
                alpha=0.8
            )
            
            # Add operation label
            axes[i].text(
                start + duration/2,
                machine_idx,
                f"J{job_idx+1}-{op%n_machines+1}",
                ha='center',
                va='center',
                color='black',
                fontsize=8
            )
        
        # Add job completion markers
        for job_idx, completion_time in enumerate(job_completion_times):
            axes[i].axvline(x=completion_time, color=colormap(job_idx % colormap.N), linestyle='--', alpha=0.5)
        
        # Set subplot properties
        axes[i].set_yticks(range(n_machines))
        axes[i].set_yticklabels([f'Machine {j+1}' for j in range(n_machines)])
        axes[i].set_ylabel('Machine')
        axes[i].set_title(f"{method_name}")
        axes[i].grid(True, alpha=0.3)
        
        # Only add x-label for the bottom subplot
        if i == n_methods - 1:
            axes[i].set_xlabel('Time')
    
    # Add weight information
    weight_text = "Job Weights: " + ", ".join([f"J{i+1}={w}" for i, w in enumerate(weights)])
    plt.figtext(0.5, 0.01, weight_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig, result_weighted_sums

def create_precedence_graph(times, machines, operations_order=None):
    """
    Create a precedence graph data structure from JSSP instance data.
    
    Args:
        times: Matrix of processing times
        machines: Matrix of machine assignments
        operations_order: Optional list of operations in the scheduled order
    
    Returns:
        G: NetworkX DiGraph representing precedence relations
        job_indices: Mapping of operations to their job indices
    """
    n_jobs, n_machines = times.shape
    G = nx.DiGraph()
    job_indices = {}
    
    # Create nodes for all operations
    for j in range(n_jobs):
        for m in range(n_machines):
            op_id = j * n_machines + m
            G.add_node(op_id)
            job_indices[op_id] = j
            
            # Add edge to next operation in same job (precedence constraint)
            if m < n_machines - 1:
                next_op = j * n_machines + (m + 1)
                G.add_edge(op_id, next_op)
    
    # If operations_order is provided, add machine constraints
    if operations_order:
        machine_ops = {}
        for op in operations_order:
            job_idx = op // n_machines
            m_idx = op % n_machines
            machine = machines[job_idx, m_idx] - 1  # Convert to 0-based
            
            # Add this operation to its machine's sequence
            if machine not in machine_ops:
                machine_ops[machine] = []
            machine_ops[machine].append(op)
        
        # Add machine constraint edges
        for machine, ops in machine_ops.items():
            for i in range(len(ops) - 1):
                G.add_edge(ops[i], ops[i+1])
    
    return G, job_indices

def draw_disjunctive_graph(times, machines, operations_order=None, title=None):
    """
    Draw a disjunctive graph representing the JSSP instance as in the paper.
    
    Args:
        times: Matrix of processing times
        machines: Matrix of machine assignments
        operations_order: Optional list of operations in the scheduled order
        title: Optional title for the graph
    
    Returns:
        fig: Matplotlib figure
    """    
    n_jobs, n_machines = times.shape
    colormap = create_colormap()
    
    # Create graph
    G = nx.DiGraph()
    
    # Adjust node positions for better spacing
    node_spacing_x = 2.0
    node_spacing_y = 1.5
    
    # Add source (S) and sink (T) nodes
    G.add_node('S', pos=(-1 * node_spacing_x, n_jobs/2 * node_spacing_y))
    G.add_node('T', pos=(n_machines * node_spacing_x, n_jobs/2 * node_spacing_y))
    
    # Create nodes for all operations
    for j in range(n_jobs):
        for m in range(n_machines):
            # Node name: Ojm (job j, machine index m)
            node_name = f"O{j+1},{m+1}"
            G.add_node(node_name, pos=(m * node_spacing_x, j * node_spacing_y), 
                      job=j, mach_idx=m, machine=machines[j, m])
            
            # Connect first operations to source
            if m == 0:
                G.add_edge('S', node_name, color='black', linestyle='solid', weight=2)
                
            # Connect last operations to sink
            if m == n_machines - 1:
                G.add_edge(node_name, 'T', color='black', linestyle='solid', weight=2)
            
            # Add conjunctive arcs (job precedence)
            if m < n_machines - 1:
                next_node = f"O{j+1},{m+2}"
                G.add_edge(node_name, next_node, color='black', linestyle='solid', weight=2)
    
    # Add disjunctive arcs (machine constraints)
    # Group operations by machine
    machine_ops = {}
    for j in range(n_jobs):
        for m in range(n_machines):
            machine = machines[j, m]
            if machine not in machine_ops:
                machine_ops[machine] = []
            machine_ops[machine].append((j, m, f"O{j+1},{m+1}"))
    
    # Add disjunctive arcs as dashed lines between operations on same machine
    for machine, ops in machine_ops.items():
        # Use a distinct color for each machine's disjunctive arcs
        machine_color = colormap(machine % colormap.N)
        
        for i, (j1, m1, node1) in enumerate(ops):
            for j2, m2, node2 in ops[i+1:]:
                # If we have operations_order, draw directed arcs based on solution
                if operations_order is not None:
                    # Check if we can determine the order
                    op1_id = j1 * n_machines + m1
                    op2_id = j2 * n_machines + m2
                    
                    if op1_id in operations_order and op2_id in operations_order:
                        if operations_order.index(op1_id) < operations_order.index(op2_id):
                            G.add_edge(node1, node2, color='red', linestyle='dashed', weight=1)
                        else:
                            G.add_edge(node2, node1, color='red', linestyle='dashed', weight=1)
                else:
                    # For the unsolved graph, add undirected edges (represented as two directed edges)
                    G.add_edge(node1, node2, color=machine_color, linestyle='dashed', weight=0.5)
                    G.add_edge(node2, node1, color=machine_color, linestyle='dashed', weight=0.5)
    
    # Create figure with increased size
    plt.figure(figsize=(max(14, n_machines * 2.5), max(10, n_jobs * 2)))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw regular nodes (operations)
    node_colors = []
    regular_nodes = [n for n in G.nodes() if n not in ['S', 'T']]
    for node in regular_nodes:
        job = int(node.split(',')[0][1:]) - 1
        node_colors.append(colormap(job % colormap.N))
    
    # Draw operation nodes with increased size
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=regular_nodes,
        node_color=node_colors,
        node_size=1000,  # Increased from 700
        edgecolors='black',
        linewidths=1.5
    )
    
    # Draw source and sink nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=['S', 'T'],
        node_color='lightgray',
        node_size=800,  # Increased from 500
        node_shape='s',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Draw edges by type
    edge_styles = {
        'solid': [],
        'dashed': []
    }
    edge_colors = {
        'solid': [],
        'dashed': []
    }
    
    for u, v, data in G.edges(data=True):
        edge_styles[data['linestyle']].append((u, v))
        edge_colors[data['linestyle']].append(data['color'])
    
    # Draw conjunctive arcs (solid)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edge_styles['solid'],
        edge_color=edge_colors['solid'],
        width=2,
        arrowsize=15
    )
    
    # Draw disjunctive arcs (dashed)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edge_styles['dashed'],
        edge_color=edge_colors['dashed'],
        width=1,
        arrowsize=10,
        style='dashed'
    )
    
    # Draw node labels with increased font size
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,  # Increased from 10
        font_weight='bold',
        font_family='sans-serif',
        font_color='black',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3)  # Add background to labels
    )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='solid', lw=2, label='Conjunctive (job precedence)'),
        Line2D([0], [0], color='gray', linestyle='dashed', lw=1, label='Disjunctive (machine constraint)'),
    ]
    
    if operations_order is not None:
        legend_elements.append(Line2D([0], [0], color='red', linestyle='dashed', lw=1, 
                                    label='Disjunctive (selected direction)'))
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=3, fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    
    # Add title
    if title:
        plt.title(title, fontsize=16, pad=20)
    else:
        plt.title("Disjunctive Graph Representation", fontsize=16, pad=20)
    
    # Add machine and job labels
    plt.text(-1.5 * node_spacing_x, -0.8 * node_spacing_y, 
            "S: Source   T: Sink   Ojm: Operation of job j on position m", 
            fontsize=12, ha='left')
    
    # Add information about processing times
    times_text = "Processing times:\n"
    for j in range(min(n_jobs, 6)):  # Show for at most 6 jobs to avoid clutter
        times_text += f"Job {j+1}: {times[j]}\n"
    if n_jobs > 6:
        times_text += "...\n"
    
    plt.figtext(0.02, 0.02, times_text, fontsize=10, ha='left', va='bottom')
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()