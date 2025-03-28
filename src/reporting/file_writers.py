"""
File I/O module for writing results to disk with clean, centralized path handling.
"""
import os
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional

class ReportWriter:
    """Write report files to disk with centralized path management."""
    
    # Define the standard directory structure as a class variable
    # so it can be imported and used by other modules
    DIRECTORY_STRUCTURE = {
        'root': '',  # Will be set to output_dir
        'artifacts': 'artifacts',
        'latex': 'artifacts/latex',
        'plots': 'artifacts/plots',
        'best_run': 'best_run',
        'best_run_media': 'best_run/media/images',
        'best_run_model': 'best_run/model'
    }
    
    def __init__(self, output_dir: str, create_all: bool = True):
        """
        Initialize the report writer and create standard directory structure.
        
        Args:
            output_dir: Base directory for all output files
            create_all: Whether to create all directories (True) or only artifacts (False)
        """
        self.output_dir = output_dir
        
        # Build full paths from structure
        self.directories = {}
        for key, subpath in self.DIRECTORY_STRUCTURE.items():
            if key == 'root':
                self.directories[key] = output_dir
            else:
                self.directories[key] = os.path.join(output_dir, subpath)
        
        # Create only the requested directories
        if create_all:
            # Create all directories
            for path in self.directories.values():
                os.makedirs(path, exist_ok=True)
            print(f"Created complete directory structure in {output_dir}")
        else:
            # Create only artifact directories for report generation
            for key in ['artifacts', 'latex', 'plots']:
                os.makedirs(self.directories[key], exist_ok=True)
            print(f"Created artifacts directory structure in {output_dir}")
    
    @classmethod
    def get_path_template(cls, dir_type: str) -> str:
        """Get path template for the specified directory type."""
        if dir_type in cls.DIRECTORY_STRUCTURE:
            return cls.DIRECTORY_STRUCTURE[dir_type]
        return ''
    
    def get_directory(self, directory_type: str = 'root') -> str:
        """
        Get the path to a specific directory type.
        
        Args:
            directory_type: Type of directory ('root', 'artifacts', 'latex', 'plots', etc.)
            
        Returns:
            Path to the requested directory
        """
        if directory_type in self.directories:
            return self.directories[directory_type]
        print(f"Warning: Unknown directory type '{directory_type}', using root")
        return self.output_dir
    
    def get_path(self, filename: str, directory_type: str = 'root') -> str:
        """
        Get the full path for a file in the specified directory.
        
        Args:
            filename: Name of the file (with or without extension)
            directory_type: Type of directory to place the file in
            
        Returns:
            Full path to the file
        """
        return os.path.join(self.get_directory(directory_type), filename)
    
    def write_latex_table(self, table_content: str, filename: str) -> str:
        """
        Write a LaTeX table to file.
        
        Args:
            table_content: LaTeX table content as string
            filename: Filename (without extension)
            
        Returns:
            Path to the written file
        """
        # Strip any path components, we'll handle the path ourselves
        filename = os.path.basename(filename)
        
        # Make sure filename has correct extension
        if not filename.endswith('.tex'):
            filename = f"{filename}.tex"
            
        # Get path in the latex directory
        filepath = self.get_path(filename, 'latex')
        
        try:
            with open(filepath, "w") as f:
                f.write(table_content)
            
            print(f"LaTeX table written to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error writing LaTeX table: {e}")
            return ""
    
    def write_metadata(self, metadata: Dict[str, Any], filename: str, directory_type: str = 'root') -> str:
        """
        Write metadata to YAML file.
        
        Args:
            metadata: Dictionary of metadata
            filename: Filename (without extension)
            directory_type: Directory type to write to
            
        Returns:
            Path to the written file
        """
        # Make sure filename has correct extension
        if not filename.endswith('.yaml') and not filename.endswith('.yml'):
            filename = f"{filename}.yaml"
            
        # Get path in the specified directory
        filepath = self.get_path(filename, directory_type)
        
        try:
            with open(filepath, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            print(f"Metadata written to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error writing metadata to {filepath}: {e}")
            return ""
    
    def write_plot(self, fig, filename: str, dpi: int = 300) -> str:
        """
        Save a matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Filename (without extension)
            dpi: Resolution for the saved image
            
        Returns:
            Path to the written file
        """
        # Make sure filename has correct extension
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
            
        # Get path in the plots directory
        filepath = self.get_path(filename, 'plots')
        
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving plot to {filepath}: {e}")
            return ""
    
    def get_files_by_extension(self, extension: str, directory_type: str = 'root') -> List[str]:
        """
        List all files with a specific extension in the specified directory.
        
        Args:
            extension: File extension to filter by (e.g., '.tex', '.csv')
            directory_type: Directory to search in
            
        Returns:
            List of matching filenames
        """
        # Get the directory to search
        search_dir = self.get_directory(directory_type)
        
        try:
            # Ensure extension has leading dot
            if not extension.startswith('.'):
                extension = f".{extension}"
                
            return [f for f in os.listdir(search_dir) if f.endswith(extension)]
        except Exception as e:
            print(f"Error listing files with extension {extension}: {e}")
            return []