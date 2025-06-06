import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as lines
from typing import Optional, Tuple
import pandas as pd

class Leonardo:
    """
    A class for creating specialized visualization plots.
    
    Static Methods:
        binary_ratio_plot: Creates a horizontal bar chart showing binary target distribution
        insights_box: Adds an insights section to an existing matplotlib figure
    """
    
    @staticmethod
    def binary_ratio_plot(data: pd.DataFrame,
                         column_name: str,
                         target_zero_name: str,
                         target_one_name: str,
                         font_color: str = 'white',
                         figsize: Tuple[float, float] = (6.5, 2),
                         plot_title: Optional[str] = None) -> None:
        """
        Creates a horizontal bar chart showing the distribution of a binary target variable.
        
        Args:
            data: Input DataFrame containing the binary column
            column_name: Name of the binary column to plot
            target_zero_name: Label for the '0' class
            target_one_name: Label for the '1' class
            font_color: Color of the text annotations
            figsize: Width and height of the figure in inches
            plot_title: Optional title for the plot
            
        Returns:
            None
        """
        # Calculate target ratios
        target_ratios = data[column_name].value_counts(normalize=True) * 100
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
        
        # Create stacked horizontal bars
        ax.barh(column_name, target_ratios[0], alpha=0.9)
        ax.barh(column_name, target_ratios[1], left=target_ratios[0])
        
        # Configure plot appearance
        ax.set_xlim([0, 100])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add annotations
        # Target "0"
        ax.annotate(f'{target_ratios[0]:.2f} %',
                   xy=(10, column_name),
                   va='center',
                   ha='center',
                   color=font_color)
        ax.annotate(target_zero_name,
                   xy=(10,-0.1),
                   va='center',
                   ha='center',
                   fontsize=10,
                   color=font_color)
        
        # Target "1"
        ax.annotate(f'{target_ratios[1]:.2f} %',
                   xy=(90, column_name),
                   va='center',
                   ha='center',
                   color=font_color)
        ax.annotate(target_one_name,
                   xy=(90, -0.1),
                   va='center',
                   ha='center',
                   fontsize=10,
                   color=font_color)
        
        if plot_title:
            plt.title(plot_title)
            
        plt.show()
    
    @staticmethod
    def insights_box(obj_figure: plt.Figure,
                    text: str,
                    text_fontsize: int = 14,
                    text_color: str = 'black',
                    text_x: float = 1.03,
                    text_y: float = 0.7,
                    font_weight: str = 'normal') -> None:
        """
        Adds an insights section with text to the right side of a matplotlib figure.
        
        Args:
            obj_figure: Matplotlib figure object to add insights to
            text: Text content for the insights section
            text_fontsize: Font size for the text
            text_color: Color of the text and separator line
            text_x: X-coordinate for text placement
            text_y: Y-coordinate for text placement
            font_weight: Font weight for the text
            
        Returns:
            None
        """
        # Add vertical separator line
        separator = lines.Line2D([1.01, 1.01], 
                               [0.1, 0.9], 
                               transform=obj_figure.transFigure,
                               figure=obj_figure,
                               color=text_color,
                               lw=0.5)
        
        obj_figure.lines.extend([separator])
        
        # Add text
        obj_figure.text(text_x,
                       text_y,
                       text,
                       fontsize=text_fontsize,
                       color=text_color,
                       fontweight=font_weight)