import matplotlib.pyplot as plt
import matplotlib.lines as lines
from typing import Optional, Tuple
import pandas as pd
import seaborn as sns

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
                    position: str = 'right',
                    fontsize: int = 14,
                    color: str = 'black',
                    x: float = None,
                    y: float = None,
                    weight: str = 'normal') -> None:
        """
        Adds an insights section with text to any side of a matplotlib figure.
        
        Args:
            fig: Matplotlib figure object to add insights to
            text: Text content for the insights section
            position: Position of the insights box ('right', 'left', 'top', 'bottom')
            fontsize: Font size for the text
            color: Color of the text and separator line
            x: X-coordinate for text placement (overrides default position)
            y: Y-coordinate for text placement (overrides default position)
            weight: Font weight for the text
            
        Returns:
            None
        """
        # Set default coordinates and line positions based on position
        if position == 'right':
            default_x, default_y = 1.03, 0.7
            line_coords = ([1.01, 1.01], [0.1, 0.9])
        elif position == 'left':
            default_x, default_y = -0.03, 0.7
            line_coords = ([-0.01, -0.01], [0.1, 0.9])
        elif position == 'top':
            default_x, default_y = 0.5, 1.03
            line_coords = ([0.1, 0.9], [1.01, 1.01])
        elif position == 'bottom':
            default_x, default_y = 0.5, -0.03
            line_coords = ([0.1, 0.9], [-0.01, -0.01])
        else:  # Default to right if invalid position
            default_x, default_y = 1.03, 0.7
            line_coords = ([1.01, 1.01], [0.1, 0.9])
        
        # Use provided coordinates or defaults
        x = x if x is not None else default_x
        y = y if y is not None else default_y
        
        # Add separator line and text
        fig.lines.extend([lines.Line2D(line_coords[0], line_coords[1], 
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
    
    @staticmethod
    def ridge_plot(data: pd.DataFrame, 
                x_var: str, 
                group_var: str, 
                cmap=None, 
                colors=None, 
                height: float = 1, 
                aspect: float = 15, 
                title: str = None) -> sns.FacetGrid:
        """
        Create a ridge plot using seaborn's FacetGrid and kdeplot.
        
        Parameters:
        -----------
        data : DataFrame
            The input pandas DataFrame
        x_var : str
            The column name for the x-axis variable (continuous)
        group_var : str
            The column name for grouping (categorical)
        cmap : matplotlib colormap or str, optional
            Colormap to use for the gradient (e.g., 'viridis', 'plasma')
        colors : list, optional
            Custom list of colors to use instead of a colormap
        height : float, optional
            Height of each facet (default: 1)
        aspect : float, optional
            Aspect ratio of each facet
        title : str, optional
            Plot title
        
        Returns:
        --------
        g : FacetGrid
            The seaborn FacetGrid object with the ridge plot
        """
        # Determine palette
        n_cats = len(data[group_var].unique())
        
        if colors is not None:
            palette = colors
        elif cmap is not None:
            if isinstance(cmap, str):
                palette = sns.color_palette(cmap, n_cats)
            else:
                palette = [cmap(i/(n_cats-1)) for i in range(n_cats)]
        else:
            palette = sns.color_palette("viridis", n_cats)
        
        # Create the FacetGrid
        g = sns.FacetGrid(data, row=group_var, hue=group_var, aspect=aspect, height=height, palette=palette)
        
        # Draw the densities
        g.map(sns.kdeplot, x_var, bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, x_var, clip_on=False, color="w", lw=2, bw_adjust=.5)
        
        # Add a horizontal line at y=0
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        
        # Add labels
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .4, label, color=color, ha="left", va="center", transform=ax.transAxes, fontsize=12)
        
        g.map(label, x_var)
        
        # Adjust layout and formatting
        g.figure.subplots_adjust(hspace=0.1)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        
        # Format axes
        for ax in g.axes.flat:
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.xaxis.set_visible(True)
            ax.tick_params(axis='x', colors='black')
        
        # Add title if provided
        if title:
            plt.suptitle(title, y=1.02)
            
        return g

    @staticmethod
    def dumbbell_plot(df: pd.DataFrame, 
                            group_col: str, 
                            category_col: str, 
                            value_col: str, 
                            ax: Optional[plt.Axes] = None,
                            title: Optional[str] = None, 
                            subtitle: Optional[str] = None, 
                            figsize: Tuple[float, float] = (10, 4)) -> plt.Figure:
        """
        Create a dumbbell plot comparing two groups across categories.
        
        Args:
            df: DataFrame with the raw data
            group_col: Column name containing the two groups to compare
            category_col: Column name containing the categories for y-axis
            value_col: Column name containing the values to plot
            ax: Optional matplotlib axis to plot on
            title: Optional title for the plot
            subtitle: Optional subtitle for the plot
            figsize: Width and height of the figure in inches
            
        Returns:
            The matplotlib Figure object with the plot
        """
        # Pivot data to get groups as columns and categories as rows
        pivot_data = df.pivot_table(index=category_col, columns=group_col, values=value_col)
        groups = pivot_data.columns.tolist()
        
        # Create figure if ax not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Remove spines
        for s in ["right", "top", "bottom", "left"]:
            ax.spines[s].set_visible(False)
        
        # Define range for y-axis and plot elements
        my_range = range(1, len(pivot_data) + 1)
        ax.hlines(y=my_range, xmin=pivot_data[groups[0]], xmax=pivot_data[groups[1]], 
                color='gray', alpha=0.4)
        ax.scatter(pivot_data[groups[0]], my_range, s=100, label=groups[0])
        ax.scatter(pivot_data[groups[1]], my_range, s=100, label=groups[1])
        
        # Add mean lines
        ax.axvline(pivot_data[groups[0]].mean(), color='gray', linewidth=0.4, linestyle='dashdot')
        ax.axvline(pivot_data[groups[1]].mean(), color='gray', linewidth=0.4, linestyle='dashdot')
        
        # Set y-tick labels and remove tick marks
        ax.set_yticks(my_range)
        ax.set_yticklabels(pivot_data.index)
        ax.tick_params(axis='both', which='both', length=0)
        
        # Add title and subtitle
        if title:
            ax.text(0, len(pivot_data) + 1.2, title, fontsize=14, fontweight='bold')
        if subtitle:
            ax.text(0, len(pivot_data) + 0.6, subtitle, fontsize=10)
        
        return fig
    
    @staticmethod
    def lollipop_plot(df: pd.DataFrame, 
                    value_col: str, 
                    label_col: str, 
                    ax: Optional[plt.Axes] = None, 
                    n: int = 15, 
                    markersize: int = 10) -> Tuple[Optional[plt.Figure], plt.Axes]:
        """
        Create a lollipop plot showing values with markers connected to a baseline.
        
        Args:
            df: DataFrame containing the data
            value_col: Column name for the values (x-axis)
            label_col: Column name for the labels (y-axis)
            ax: Optional matplotlib axes object to plot on
            n: Number of items to display (default: 15)
            markersize: Size of the marker points (default: 10)
            
        Returns:
            Tuple containing the figure (if created) and axes objects
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        
        temp = df[:n]
        my_range = range(1, len(temp[label_col])+1)
        
        # Plot lines
        for i, val in enumerate(temp[value_col]):
            ax.plot([0, val], [my_range[i], my_range[i]], color='gray', alpha=0.8)
            ax.plot(val, my_range[i], 'o', color='#0085a1', markersize=markersize)
        
        # Remove spines
        for s in ['top', 'right', 'bottom', 'left']:
            ax.spines[s].set_visible(False)
        
        ax.set_yticks(my_range)
        ax.set_yticklabels(temp[label_col])
        
        return fig, ax