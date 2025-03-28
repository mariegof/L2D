"""
LaTeX formatting module using PyLaTeX for creating publication-quality tables.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import traceback
from pylatex import Document, Table, Tabular, Center, Label, MultiRow
from pylatex.utils import bold, NoEscape

class PyLatexFormatter:
    """Format data as LaTeX tables using PyLaTeX with improved error handling."""
    
    def __init__(self):
        """Initialize the LaTeX formatter."""
        # Add LaTeX preamble commands that should be included in all tables
        self.preamble = [
            r'\usepackage{booktabs}',   # For high-quality tables
            r'\usepackage{xcolor}',      # For colored text/cells
            r'\usepackage{array}'        # For advanced table formatting
        ]
    
    def format_dataframe_to_latex(self, 
                                 df: pd.DataFrame, 
                                 caption: str, 
                                 label: str, 
                                 column_format: Optional[str] = None,
                                 highlight_best: bool = True) -> str:
        """
        Convert a pandas DataFrame to a LaTeX table using PyLaTeX.
        
        Args:
            df: DataFrame to convert
            caption: Table caption
            label: Table label
            column_format: Optional column format string (default: 'l' for first column, 'r' for others)
            highlight_best: Whether to highlight the best values in metrics columns
            
        Returns:
            LaTeX code as a string
        """
        if df.empty:
            return f"% Empty table for {label}"
        
        try:
            # Create a document fragment
            doc = Document(documentclass='article', document_options=['preview'])
            
            # Add preamble
            for cmd in self.preamble:
                doc.preamble.append(NoEscape(cmd))
            
            # Create the table
            with doc.create(Table(position='htbp')) as table:
                # Center the content inside the table
                table.append(NoEscape(r'\centering'))
                
                # Set default column format if not provided
                if column_format is None:
                    column_format = 'l' + 'r' * (len(df.columns) - 1)
                
                # Create the tabular environment
                with doc.create(Tabular(column_format)) as tabular:
                    # Add toprule (requires booktabs package)
                    tabular.append(NoEscape(r'\toprule'))
                    
                    # Add header row
                    headers = [bold(col) for col in df.columns]
                    tabular.add_row(headers)
                    # Add midrule
                    tabular.append(NoEscape(r'\midrule'))
                    
                    # Identify best values in each metric column if highlight_best is True
                    best_indices = {}
                    if highlight_best:
                        for col_idx, col_name in enumerate(df.columns):
                            col_values = df[col_name]
                            # Skip non-numeric columns
                            if not pd.api.types.is_numeric_dtype(col_values):
                                continue
                                
                            # Determine if higher or lower is better for this metric
                            lower_is_better = ('weighted_sum' in col_name.lower() or 
                                            'error' in col_name.lower())
                            
                            if lower_is_better:
                                best_idx = col_values.idxmin() if not col_values.empty else None
                            else:
                                best_idx = col_values.idxmax() if not col_values.empty else None
                                
                            if best_idx is not None:
                                best_indices[(best_idx, col_idx)] = True
                    
                    # Add data rows
                    for row_idx, (_, row) in enumerate(df.iterrows()):
                        formatted_row = []
                        
                        # Format each column
                        for col_idx, value in enumerate(row):
                            col_name = df.columns[col_idx]
                            
                            # Format numeric columns with appropriate precision
                            if isinstance(value, (int, float)):
                                # Format percentages with 1 decimal place
                                if 'Win' in col_name or 'Rate' in col_name or 'Impr' in col_name or '%' in col_name:
                                    value_str = f"{value:.1f}"
                                elif 'Sum' in col_name:  # Weighted Sum should be integer
                                    value_str = f"{int(value)}"
                                else:
                                    # Format other numbers with 2 decimal places
                                    value_str = f"{value:.2f}"
                                
                                # Highlight best value if needed
                                if highlight_best and (row_idx, col_idx) in best_indices:
                                    value_str = rf"\textbf{{{value_str}}}"
                            else:
                                value_str = str(value)
                                
                            formatted_row.append(NoEscape(value_str))
                        
                        tabular.add_row(formatted_row)
                    
                    # Add bottomrule
                    tabular.append(NoEscape(r'\bottomrule'))
                
                # Add caption and label
                table.add_caption(caption)
                table.append(Label(label))
            
            # Extract the LaTeX code
            latex_code = doc.dumps()
            
            # Extract just the table environment
            table_start = latex_code.find('\\begin{table}')
            table_end = latex_code.find('\\end{table}') + len('\\end{table}')
            
            if table_start != -1 and table_end != -1:
                return latex_code[table_start:table_end]
            else:
                return latex_code  # Return full code if table markers not found
                
        except Exception as e:
            print(f"Error formatting DataFrame to LaTeX: {e}")
            traceback.print_exc()
            return f"% Error generating LaTeX table for {label}: {str(e)}"
    
    def format_summary_table(self, metrics_data: Dict[str, Dict[str, float]], caption: str, label: str) -> str:
        """
        Create a summary statistics table for metrics.
        
        Args:
            metrics_data: Dictionary of metrics with their statistics
            caption: Table caption
            label: Table label
            
        Returns:
            LaTeX code as a string
        """
        if not metrics_data:
            return f"% Empty metrics data for {label}"
        
        try:
            # Create a document fragment
            doc = Document(documentclass='article', document_options=['preview'])
            
            # Add preamble
            for cmd in self.preamble:
                doc.preamble.append(NoEscape(cmd))
            
            # Create the table
            with doc.create(Table(position='htbp')) as table:
                table.append(NoEscape(r'\centering'))
                
                # Create tabular with appropriate columns
                column_format = 'l' + 'r' * 5  # Metric + 5 statistics columns
                with doc.create(Tabular(column_format)) as tabular:
                    tabular.append(NoEscape(r'\toprule'))
                    
                    # Add header row
                    headers = ['Metric', 'Min', 'Max', 'Mean', 'Median', 'Std Dev']
                    tabular.add_row([bold(h) for h in headers])
                    tabular.append(NoEscape(r'\midrule'))
                    
                    # Add formatted metric names
                    metric_names = {
                        'validation_weighted_sum': 'Weighted Sum',
                        'validation_win_vs_wspt': 'Win vs WSPT (\%)',
                        'validation_win_vs_spt': 'Win vs SPT (\%)',
                        'validation_win_vs_srpt': 'Win vs SRPT (\%)',
                        'validation_win_rate': 'Overall Win Rate (\%)',
                        'validation_improvement_over_wspt': 'Improv. WSPT (\%)',
                        'validation_improvement_over_spt': 'Improv. SPT (\%)',
                        'validation_improvement_over_srpt': 'Improv. SRPT (\%)'
                    }
                    
                    # Add rows for each metric
                    for metric, stats in metrics_data.items():
                        display_name = metric_names.get(metric, metric.replace('validation_', '').replace('_', ' ').title())
                        
                        # Format values appropriately
                        if 'weighted_sum' in metric.lower():
                            # For weighted sum, show integers
                            row = [
                                NoEscape(display_name),
                                NoEscape(f"{int(stats['min'])}"),
                                NoEscape(f"{int(stats['max'])}"),
                                NoEscape(f"{int(stats['mean'])}"),
                                NoEscape(f"{int(stats['median'])}"),
                                NoEscape(f"{stats['std']:.1f}")
                            ]
                        else:
                            # For percentages, show 1 decimal place
                            row = [
                                NoEscape(display_name),
                                NoEscape(f"{stats['min']:.1f}"),
                                NoEscape(f"{stats['max']:.1f}"),
                                NoEscape(f"{stats['mean']:.1f}"),
                                NoEscape(f"{stats['median']:.1f}"),
                                NoEscape(f"{stats['std']:.1f}")
                            ]
                        
                        tabular.add_row(row)
                    
                    tabular.append(NoEscape(r'\bottomrule'))
                
                # Add caption and label using the correct method
                table.add_caption(caption)
                table.append(Label(label))
            
            # Extract the LaTeX code
            latex_code = doc.dumps()
            
            # Extract just the table environment
            table_start = latex_code.find('\\begin{table}')
            table_end = latex_code.find('\\end{table}') + len('\\end{table}')
            
            if table_start != -1 and table_end != -1:
                return latex_code[table_start:table_end]
            else:
                return latex_code
                
        except Exception as e:
            print(f"Error formatting summary table to LaTeX: {e}")
            traceback.print_exc()
            return f"% Error generating summary LaTeX table for {label}: {str(e)}"
    
    def format_best_runs_table(self, 
                              best_runs: Dict[str, str], 
                              run_metrics: Dict[str, Dict[str, float]], 
                              caption: str, 
                              label: str) -> str:
        """
        Create a table showing the best runs for each metric.
        
        Args:
            best_runs: Dictionary mapping metrics to best run names
            run_metrics: Dictionary mapping run names to metric values
            caption: Table caption
            label: Table label
            
        Returns:
            LaTeX code as a string
        """
        if not best_runs:
            return f"% No best runs data for {label}"
        
        try:
            # Create document
            doc = Document(documentclass='article', document_options=['preview'])
            
            # Add preamble
            for cmd in self.preamble:
                doc.preamble.append(NoEscape(cmd))
            
            # Create table
            with doc.create(Table(position='htbp')) as table:
                table.append(NoEscape(r'\centering'))
                
                # Create tabular
                with doc.create(Tabular('lll')) as tabular:
                    tabular.append(NoEscape(r'\toprule'))
                    
                    # Add header row
                    headers = ['Metric', 'Best Run', 'Value']
                    tabular.add_row([bold(h) for h in headers])
                    tabular.append(NoEscape(r'\midrule'))
                    
                    # Format metric names
                    metric_names = {
                        'validation_weighted_sum': 'Weighted Sum',
                        'validation_win_vs_wspt': 'Win vs WSPT (\%)',
                        'validation_win_vs_spt': 'Win vs SPT (\%)',
                        'validation_win_vs_srpt': 'Win vs SRPT (\%)',
                        'validation_win_rate': 'Overall Win Rate (\%)',
                        'validation_improvement_over_wspt': 'Improv. over WSPT (\%)',
                        'validation_improvement_over_spt': 'Improv. over SPT (\%)',
                        'validation_improvement_over_srpt': 'Improv. over SRPT (\%)'
                    }
                    
                    # Add rows for each metric
                    for metric, run_name in best_runs.items():
                        # Get the metric value for this run
                        if run_name in run_metrics:
                            value = run_metrics[run_name].get(metric, 'N/A')
                        else:
                            # Try to find the metric value in any matching run name
                            value = 'N/A'
                            for name, metrics in run_metrics.items():
                                if name.startswith(run_name) or run_name.startswith(name):
                                    value = metrics.get(metric, 'N/A')
                                    break
                        
                        # Format value properly
                        if isinstance(value, (int, float)):
                            if 'weighted_sum' in metric.lower():
                                # For weighted sum, show integers
                                value_str = f"{int(value)}"
                            elif 'win' in metric.lower() or 'rate' in metric.lower() or 'improv' in metric.lower():
                                # For percentages, show 1 decimal place
                                value_str = f"{value:.1f}"
                            else:
                                # For other values, show 2 decimal places
                                value_str = f"{value:.2f}"
                        else:
                            value_str = str(value)
                            
                        display_metric = metric_names.get(metric, metric.replace('validation_', '').replace('_', ' ').title())
                        
                        tabular.add_row([
                            NoEscape(display_metric),
                            NoEscape(run_name),
                            NoEscape(value_str)
                        ])
                    
                    tabular.append(NoEscape(r'\bottomrule'))
                
                # Add caption and label
                table.add_caption(caption)
                table.append(Label(label))
            
            # Extract the LaTeX code
            latex_code = doc.dumps()
            
            # Extract just the table environment
            table_start = latex_code.find('\\begin{table}')
            table_end = latex_code.find('\\end{table}') + len('\\end{table}')
            
            if table_start != -1 and table_end != -1:
                return latex_code[table_start:table_end]
            else:
                return latex_code
                
        except Exception as e:
            print(f"Error formatting best runs table to LaTeX: {e}")
            traceback.print_exc()
            return f"% Error generating best runs LaTeX table for {label}: {str(e)}"
    
    def format_comparison_table(self, 
                               df: pd.DataFrame, 
                               primary_param: str,
                               caption: str, 
                               label: str) -> str:
        """
        Create a comparison table highlighting differences between parameter values.
        
        Args:
            df: DataFrame with comparison data
            primary_param: Primary parameter used for comparison
            caption: Table caption
            label: Table label
            
        Returns:
            LaTeX code as a string
        """
        if df.empty:
            return f"% Empty comparison data for {label}"
        
        try:
            # Create a document fragment
            doc = Document(documentclass='article', document_options=['preview'])
            
            # Add preamble
            for cmd in self.preamble:
                doc.preamble.append(NoEscape(cmd))
                
            # Add extra commands for colored cells
            doc.preamble.append(NoEscape(r'\newcommand{\bestcell}[1]{\cellcolor{green!15}\textbf{#1}}'))
            doc.preamble.append(NoEscape(r'\newcommand{\secondcell}[1]{\cellcolor{blue!10}{#1}}'))
            
            # Create the table
            with doc.create(Table(position='htbp')) as table:
                table.append(NoEscape(r'\centering'))
                
                # Set column format - all metrics are right-aligned
                column_format = 'l' + 'r' * (len(df.columns) - 1)
                
                # Create the tabular environment
                with doc.create(Tabular(column_format)) as tabular:
                    tabular.append(NoEscape(r'\toprule'))
                    
                    # Add header row
                    headers = [bold(col) for col in df.columns]
                    tabular.add_row(headers)
                    # Add midrule
                    tabular.append(NoEscape(r'\midrule'))
                    
                    # Find metric columns
                    metric_cols = []
                    for col in df.columns:
                        if col != primary_param and col != 'Runs' and pd.api.types.is_numeric_dtype(df[col]):
                            metric_cols.append(col)
                    
                    # Find best and second best values for each metric
                    best_indices = {}
                    second_indices = {}
                    
                    for col in metric_cols:
                        col_idx = df.columns.get_loc(col)
                        values = df[col].values
                        
                        # Skip columns with non-numeric values
                        if not all(isinstance(v, (int, float)) for v in values):
                            continue
                            
                        # Determine if lower is better
                        lower_is_better = 'weighted_sum' in col.lower() or 'error' in col.lower()
                        
                        if lower_is_better:
                            # Sort values and get indices of best (lowest) and second best
                            sorted_indices = np.argsort(values)
                            if len(sorted_indices) > 0:
                                best_indices[(sorted_indices[0], col_idx)] = True
                            if len(sorted_indices) > 1:
                                second_indices[(sorted_indices[1], col_idx)] = True
                        else:
                            # Sort values and get indices of best (highest) and second best
                            sorted_indices = np.argsort(values)[::-1]  # Reverse for descending order
                            if len(sorted_indices) > 0:
                                best_indices[(sorted_indices[0], col_idx)] = True
                            if len(sorted_indices) > 1:
                                second_indices[(sorted_indices[1], col_idx)] = True
                    
                    # Add data rows
                    for row_idx, (_, row) in enumerate(df.iterrows()):
                        formatted_row = []
                        
                        # Format each column
                        for col_idx, value in enumerate(row):
                            col_name = df.columns[col_idx]
                            
                            # Format numeric columns with appropriate precision
                            if isinstance(value, (int, float)):
                                # Format percentages with 1 decimal place
                                if 'Win' in col_name or 'Rate' in col_name or 'Impr' in col_name or '%' in col_name:
                                    value_str = f"{value:.1f}"
                                elif 'Sum' in col_name:  # Weighted Sum should be integer
                                    value_str = f"{int(value)}"
                                else:
                                    # Format other numbers with 2 decimal places
                                    value_str = f"{value:.2f}"
                                
                                # Highlight best and second best values
                                if (row_idx, col_idx) in best_indices:
                                    value_str = rf"\bestcell{{{value_str}}}"
                                elif (row_idx, col_idx) in second_indices:
                                    value_str = rf"\secondcell{{{value_str}}}"
                            else:
                                value_str = str(value)
                                
                            formatted_row.append(NoEscape(value_str))
                        
                        tabular.add_row(formatted_row)
                    
                    # Add bottomrule
                    tabular.append(NoEscape(r'\bottomrule'))
                
                # Add caption and label
                table.add_caption(caption)
                table.append(Label(label))
            
            # Extract the LaTeX code
            latex_code = doc.dumps()
            
            # Extract just the table environment
            table_start = latex_code.find('\\begin{table}')
            table_end = latex_code.find('\\end{table}') + len('\\end{table}')
            
            if table_start != -1 and table_end != -1:
                return latex_code[table_start:table_end]
            else:
                return latex_code
                
        except Exception as e:
            print(f"Error formatting comparison table to LaTeX: {e}")
            traceback.print_exc()
            return f"% Error generating comparison LaTeX table for {label}: {str(e)}"