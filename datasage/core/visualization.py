import matplotlib.pyplot as plt
import matplotlib.lines as lines
from typing import Optional, Tuple
import pandas as pd

class Leonardo:
    """
    A class for creating specialized visualization plots.
    
    Static Methods:
        binary_ratio_plot: Creates a horizontal bar chart showing binary target distribution
        insights_box: Adds an insights section to an existing matplotlib figure
        stacked_bars: Creates a stacked bar chart showing distribution across categories
    """
    
    @staticmethod
    def binary_ratio_plot(data: pd.DataFrame,
                         column_name: str,
                         target_zero_name: str,
                         target_one_name: str,
                         ax: Optional[plt.Axes] = None,
                         font_color: str = 'white',
                         figsize: Tuple[float, float] = (6.5, 2),
                         plot_title: Optional[str] = None) -> plt.Axes:
        """
        Creates a horizontal bar chart showing the distribution of a binary target variable.
        
        Args:
            data: Input DataFrame containing the binary column
            column_name: Name of the binary column to plot
            target_zero_name: Label for the '0' class
            target_one_name: Label for the '1' class
            ax: Optional matplotlib axes object to plot on
            font_color: Color of the text annotations
            figsize: Width and height of the figure in inches
            plot_title: Optional title for the plot
            
        Returns:
            The matplotlib axes object with the plot
        """
        # Calculate target ratios
        target_ratios = data[column_name].value_counts(normalize=True) * 100
        
        # Create plot if needed
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        # Create bars and configure appearance
        ax.barh(column_name, target_ratios[0], alpha=0.9)
        ax.barh(column_name, target_ratios[1], left=target_ratios[0])
        ax.set_xlim([0, 100])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add annotations for both targets
        for idx, (pos, label) in enumerate(zip([10, 90], [target_zero_name, target_one_name])):
            ax.annotate(f'{target_ratios[idx]:.2f} %',
                       xy=(pos, column_name),
                       va='center', ha='center', color=font_color)
            ax.annotate(label,
                       xy=(pos, -0.1),
                       va='center', ha='center', fontsize=10, color=font_color)
        
        if plot_title:
            ax.set_title(plot_title)
            
        return ax
    
    @staticmethod
    def insights_box(fig: plt.Figure,
                    text: str,
                    fontsize: int = 14,
                    color: str = 'black',
                    x: float = 1.03,
                    y: float = 0.7,
                    weight: str = 'normal') -> None:
        """
        Adds an insights section with text to the right side of a matplotlib figure.
        
        Args:
            fig: Matplotlib figure object to add insights to
            text: Text content for the insights section
            fontsize: Font size for the text
            color: Color of the text and separator line
            x: X-coordinate for text placement
            y: Y-coordinate for text placement
            weight: Font weight for the text
            
        Returns:
            None
        """
        # Add separator line and text in one block
        fig.lines.extend([lines.Line2D([1.01, 1.01], [0.1, 0.9], 
                                     transform=fig.transFigure,
                                     figure=fig, color=color, lw=0.5)])
        fig.text(x, y, text, fontsize=fontsize, color=color, fontweight=weight)
        
    @staticmethod
    def stacked_bars(data: pd.DataFrame,
                    x: str,
                    y: str,
                    ax: plt.Axes,
                    title: str,
                    normalize: str = 'index',
                    fmt: str = '%.1f%%',
                    label_color: str = 'white',
                    width: float = 0.9) -> plt.Axes:
        """
        Create a stacked bar chart showing the distribution of y across x categories.
        
        Args:
            data: DataFrame containing the data
            x: Column name for categories (y-axis in horizontal bars)
            y: Column name for values to count/compare
            ax: Matplotlib axes object to plot on
            title: Title for the plot
            normalize: How to normalize the data ('index', 'columns', or None)
            fmt: Format string for bar labels
            label_color: Color of the bar labels
            width: Width of the bars
            
        Returns:
            The matplotlib axes object with the plot
        """
        # Create plot with chained methods
        pd.crosstab(data[x], data[y], normalize=normalize).mul(100).plot(
            kind='barh', stacked=True, ax=ax, legend=False, width=width)
        ax.set_xlabel('Percentage (%)')
        ax.set_ylabel('')
        ax.set_title(title)
        for c in ax.containers:
            ax.bar_label(c, fmt=fmt, label_type='center', color=label_color)
        
        return ax