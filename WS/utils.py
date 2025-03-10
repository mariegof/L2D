import os

def setup_directories(config):
    """Create necessary directories for output."""
    # Main output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Models directory
    models_dir = os.path.join(config["output_dir"], "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Figures directory
    figures_dir = os.path.join(config["output_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Results directory
    results_dir = os.path.join(config["output_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)
    
    return models_dir, figures_dir, results_dir