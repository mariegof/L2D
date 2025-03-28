"""
HTML report generation module with consistent path handling and styling.
"""
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import traceback

class HTMLReportGenerator:
    """Generate HTML reports for sweep results with consistent styling and organization."""
    
    def __init__(self, output_dir: str):
        """
        Initialize HTML generator with output directory.
        
        Args:
            output_dir: Base directory for HTML reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_index_html(self, 
                         sweep_id: str, 
                         project: str, 
                         timestamp: str,
                         sweep_type: str,
                         best_runs: Optional[Dict[str, str]] = None,
                         plot_files: Optional[List[Dict[str, str]]] = None,
                         metrics_summary: Optional[Dict[str, Dict[str, float]]] = None,
                         best_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an HTML index page for navigating sweep results.
        
        Args:
            sweep_id: WandB sweep ID
            project: WandB project name
            timestamp: Timestamp for the report
            sweep_type: Type of sweep ('environment', 'feature', 'reward', 'model')
            best_runs: Optional dictionary of best runs by metric
            plot_files: Optional list of plot file information
            metrics_summary: Optional dictionary of metric summary statistics
            best_config: Optional dictionary of best configuration
            
        Returns:
            Path to the generated HTML file
        """
        try:
            # Standardize sweep_type formatting
            sweep_type_display = sweep_type.capitalize()
            if sweep_type.lower() in ['env', 'environment']:
                sweep_type_display = "Environment"
            elif sweep_type.lower() in ['feature', 'features']:
                sweep_type_display = "Feature"
            elif sweep_type.lower() in ['reward', 'rewards']:
                sweep_type_display = "Reward"
            elif sweep_type.lower() in ['model', 'models']:
                sweep_type_display = "Model"
            
            # Start HTML content with modern styling
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{sweep_type_display} Sweep Results - {sweep_id}</title>
    <style>
        :root {{
            --primary-color: #4A6FDE;
            --secondary-color: #6B7280;
            --accent-color: #10B981;
            --background-color: #F9FAFB;
            --card-background: #FFFFFF;
            --text-color: #1F2937;
            --border-color: #E5E7EB;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background-color: var(--primary-color);
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h1, h2, h3 {{
            margin-top: 0;
            color: var(--text-color);
        }}
        
        header h1 {{
            color: white;
            margin-bottom: 10px;
        }}
        
        .meta-info {{
            margin-top: 10px;
            color: rgba(255, 255, 255, 0.9);
        }}
        
        .card {{
            background-color: var(--card-background);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: #F3F4F6;
            font-weight: 600;
        }}
        
        tr:hover {{
            background-color: #F9FAFB;
        }}
        
        .plot-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .plot {{
            flex: 1 1 calc(50% - 20px);
            min-width: 300px;
            margin-bottom: 20px;
        }}
        
        .plot img {{
            max-width: 100%;
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }}
        
        .file-list {{
            margin-top: 10px;
        }}
        
        .file-list a {{
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 8px 12px;
            background-color: #F3F4F6;
            color: var(--text-color);
            text-decoration: none;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            font-size: 14px;
        }}
        
        .file-list a:hover {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: var(--secondary-color);
            border-top: 1px solid var(--border-color);
            font-size: 14px;
        }}
        
        .metric-value {{
            font-weight: 600;
        }}
        
        .best-value {{
            color: var(--accent-color);
            font-weight: 700;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{sweep_type_display} Sweep Results</h1>
            <div class="meta-info">
                <div>Sweep ID: <strong>{sweep_id}</strong></div>
                <div>Project: <strong>{project}</strong></div>
                <div>Generated on: <strong>{timestamp}</strong></div>
            </div>
        </header>
    """
            
            # Add best runs section if provided
            if best_runs:
                html_content += """
        <section class="card">
            <h2>Best Performing Runs</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Best Run</th>
                </tr>
    """
                
                # Format metric names for display
                metric_names = {
                    'validation_weighted_sum': 'Weighted Sum',
                    'validation_win_vs_wspt': 'Win vs WSPT (%)',
                    'validation_win_vs_spt': 'Win vs SPT (%)',
                    'validation_win_rate': 'Overall Win Rate (%)',
                    'validation_improvement_over_wspt': 'Improvement over WSPT (%)',
                    'validation_improvement_over_spt': 'Improvement over SPT (%)'
                }
                
                for metric, run_name in best_runs.items():
                    display_metric = metric_names.get(metric, metric.replace('validation_', '').replace('_', ' ').title())
                    html_content += f"""
                <tr>
                    <td>{display_metric}</td>
                    <td class="best-value">{run_name}</td>
                </tr>"""
                    
                html_content += """
            </table>
        </section>
    """
            
            # Add metrics summary if provided
            if metrics_summary:
                html_content += """
        <section class="card">
            <h2>Performance Metrics Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                </tr>
    """
                
                # Format metric names for display
                metric_names = {
                    'validation_weighted_sum': 'Weighted Sum',
                    'validation_win_vs_wspt': 'Win vs WSPT (%)',
                    'validation_win_vs_spt': 'Win vs SPT (%)',
                    'validation_win_rate': 'Overall Win Rate (%)',
                    'validation_improvement_over_wspt': 'Improvement over WSPT (%)',
                    'validation_improvement_over_spt': 'Improvement over SPT (%)'
                }
                
                for metric, stats in metrics_summary.items():
                    display_metric = metric_names.get(metric, metric.replace('validation_', '').replace('_', ' ').title())
                    html_content += f"""
                <tr>
                    <td>{display_metric}</td>
                    <td class="metric-value">{stats['min']:.2f}</td>
                    <td class="metric-value">{stats['max']:.2f}</td>
                    <td class="metric-value">{stats['mean']:.2f}</td>
                    <td class="metric-value">{stats['median']:.2f}</td>
                    <td class="metric-value">{stats['std']:.2f}</td>
                </tr>"""
                    
                html_content += """
            </table>
        </section>
    """
            
            # Add best configuration if provided
            if best_config:
                html_content += """
        <section class="card">
            <h2>Best Configuration</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
    """
                
                # Filter out metadata keys
                for key, value in best_config.items():
                    if not key.startswith('__'):
                        html_content += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td class="metric-value">{value}</td>
                </tr>"""
                    
                html_content += """
            </table>
        </section>
    """
            
            # Add visualizations section if plot files provided
            if plot_files:
                html_content += """
        <section class="card">
            <h2>Visualizations</h2>
            <div class="plot-container">
    """
                
                for plot_info in plot_files:
                    filename = plot_info.get('filename', '')
                    title = plot_info.get('title', os.path.basename(filename))
                    
                    # Make path relative to HTML file
                    rel_path = os.path.relpath(filename, self.output_dir) if os.path.isabs(filename) else filename
                    
                    html_content += f"""
                <div class="plot">
                    <h3>{title}</h3>
                    <img src="{rel_path}" alt="{title}" loading="lazy">
                </div>"""
                    
                html_content += """
            </div>
        </section>
    """
            
            # Add generated files section
            html_content += """
        <section class="card">
            <h2>Generated Files</h2>
            <div class="file-list">
    """
            
            # Group files by type
            file_types = {
                'Data Files': ['csv', 'json', 'yaml', 'yml'],
                'LaTeX Tables': ['tex'],
                'Reports': ['html', 'md', 'txt'],
                'Visualizations': ['png', 'jpg', 'jpeg', 'pdf', 'svg']
            }
            
            for category, extensions in file_types.items():
                # Get all files in the output directory matching these extensions
                matching_files = []
                for ext in extensions:
                    for root, _, files in os.walk(self.output_dir):
                        for file in files:
                            if file.endswith(f".{ext}"):
                                # Skip the current file
                                if file == "index.html":
                                    continue
                                    
                                full_path = os.path.join(root, file)
                                rel_path = os.path.relpath(full_path, self.output_dir)
                                matching_files.append((file, rel_path))
                
                if matching_files:
                    html_content += f"""
                <h3>{category}</h3>
    """
                    for filename, rel_path in matching_files:
                        html_content += f"""
                <a href="{rel_path}">{filename}</a>"""
                    
            html_content += """
            </div>
        </section>
        
        <footer>
            <p>Generated by the L2D Weighted Job Shop Scheduling project.</p>
            <p>View sweep in <a href="https://wandb.ai/{project}/sweeps/{sweep_id}" target="_blank">Weights & Biases</a></p>
        </footer>
    </div>
</body>
</html>
    """
            
            # Save the HTML file
            html_file = os.path.join(self.output_dir, "index.html")
            os.makedirs(os.path.dirname(html_file), exist_ok=True)
            
            with open(html_file, "w") as f:
                f.write(html_content)
                
            print(f"Generated HTML report at {html_file}")
            return html_file
            
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            traceback.print_exc()
            # Create a minimal error report
            try:
                error_html = f"""<!DOCTYPE html>
<html>
<head><title>Error Report</title></head>
<body>
    <h1>Error Generating Report</h1>
    <p>An error occurred while generating the HTML report:</p>
    <pre>{str(e)}</pre>
    <p>Sweep ID: {sweep_id}</p>
    <p>Project: {project}</p>
    <p>Generated on: {timestamp}</p>
</body>
</html>
                """
                html_file = os.path.join(self.output_dir, "error_report.html")
                with open(html_file, "w") as f:
                    f.write(error_html)
                return html_file
            except:
                return ""
    
    def create_run_detail_html(self, run: Dict[str, Any], output_file: str) -> str:
        """
        Generate a detailed HTML page for a single run.
        
        Args:
            run: Run dictionary with config, summary, and history
            output_file: Path to save the HTML file
            
        Returns:
            Path to the generated HTML file
        """
        try:
            # Extract run information
            run_name = run.get('name', 'Unknown')
            run_id = run.get('id', 'Unknown')
            config = run.get('config', {})
            summary = run.get('summary', {})
            
            # Start HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Run Details: {run_name}</title>
    <style>
        :root {{
            --primary-color: #4A6FDE;
            --secondary-color: #6B7280;
            --accent-color: #10B981;
            --background-color: #F9FAFB;
            --card-background: #FFFFFF;
            --text-color: #1F2937;
            --border-color: #E5E7EB;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background-color: var(--primary-color);
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h1, h2, h3 {{
            margin-top: 0;
            color: var(--text-color);
        }}
        
        header h1 {{
            color: white;
            margin-bottom: 10px;
        }}
        
        .meta-info {{
            margin-top: 10px;
            color: rgba(255, 255, 255, 0.9);
        }}
        
        .card {{
            background-color: var(--card-background);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: #F3F4F6;
            font-weight: 600;
        }}
        
        tr:hover {{
            background-color: #F9FAFB;
        }}
        
        footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: var(--secondary-color);
            border-top: 1px solid var(--border-color);
            font-size: 14px;
        }}
        
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            color: var(--primary-color);
            text-decoration: none;
        }}
        
        .back-link:hover {{
            text-decoration: underline;
        }}
        
        .metric-value {{
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">← Back to Overview</a>
        
        <header>
            <h1>Run Details: {run_name}</h1>
            <div class="meta-info">
                <div>Run ID: <strong>{run_id}</strong></div>
            </div>
        </header>
        
        <section class="card">
            <h2>Configuration</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
    """
            
            # Add configuration parameters
            for key, value in config.items():
                html_content += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td class="metric-value">{value}</td>
                </tr>"""
                
            html_content += """
            </table>
        </section>
        
        <section class="card">
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
    """
            
            # Format metric names for display
            metric_names = {
                'validation_weighted_sum': 'Weighted Sum',
                'validation_win_vs_wspt': 'Win vs WSPT (%)',
                'validation_win_vs_spt': 'Win vs SPT (%)',
                'validation_win_rate': 'Overall Win Rate (%)',
                'validation_improvement_over_wspt': 'Improvement over WSPT (%)',
                'validation_improvement_over_spt': 'Improvement over SPT (%)'
            }
            
            # Add metrics
            for key, value in summary.items():
                # Skip internal metrics and non-performance metrics
                if key.startswith('_') or not any(metric in key for metric in ['validation', 'win', 'rate', 'weighted']):
                    continue
                    
                display_name = metric_names.get(key, key.replace('_', ' ').title())
                html_content += f"""
                <tr>
                    <td>{display_name}</td>
                    <td class="metric-value">{value}</td>
                </tr>"""
                
            html_content += """
            </table>
        </section>
        
        <footer>
            <p>Generated by the L2D Weighted Job Shop Scheduling project.</p>
        </footer>
    </div>
</body>
</html>
    """
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w") as f:
                f.write(html_content)
                
            print(f"Generated run detail HTML at {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error generating run detail HTML: {e}")
            traceback.print_exc()
            return ""