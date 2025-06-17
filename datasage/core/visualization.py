import matplotlib.pyplot as plt
import matplotlib.lines as lines
from typing import Optional, Tuple, Union, Dict, List, Any
import pandas as pd
import seaborn as sns
import requests
import matplotlib.font_manager as fm
import os
from matplotlib.colors import Colormap

class Leonardo:
    
    """
    A collection of specialized data visualization tools.
    """
    
    # Class-level cache for downloaded fonts
    _font_cache: Dict[str, str] = {}

    @staticmethod
    def _clean_axes(ax: plt.Axes, spines_to_remove: List[str] = ['top', 'right', 'bottom', 'left']) -> None:
        """
        Helper method to remove spines and clean up axes.
        """
        for spine in spines_to_remove:
            ax.spines[spine].set_visible(False)

    @staticmethod
    def create_layout(layout_spec: List[Tuple[int, int, int, int]], 
                    figsize: Tuple[float, float] = (10, 6)) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create a flexible plot layout based on a specification.
        
        Parameters:
        -----------
        layout_spec : list of tuples
            List of (row, col, rowspan, colspan) for each axis
            Example: [(0,0,1,1), (0,1,1,1), (1,0,1,2)] creates a layout with
            two plots on top row and one spanning the bottom row
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : Figure object
        axes : list of Axes objects
        
        Common Layout Specifications:
        ----------------------------
        Single row layouts:
        - 1 plot: [(0,0,1,1)]
        - 2 plots in a row: [(0,0,1,1), (0,1,1,1)]
        - 3 plots in a row: [(0,0,1,1), (0,1,1,1), (0,2,1,1)]
        
        Single column layouts:
        - 1 plot: [(0,0,1,1)]
        - 2 plots in a column: [(0,0,1,1), (1,0,1,1)]
        - 3 plots in a column: [(0,0,1,1), (1,0,1,1), (2,0,1,1)]
        
        Grid layouts:
        - 2x2 grid: [(0,0,1,1), (0,1,1,1), (1,0,1,1), (1,1,1,1)]
        - 3x3 grid: [(0,0,1,1), (0,1,1,1), (0,2,1,1), 
                    (1,0,1,1), (1,1,1,1), (1,2,1,1), 
                    (2,0,1,1), (2,1,1,1), (2,2,1,1)]
        
        Mixed layouts:
        - 2 plots on top, 1 spanning bottom: [(0,0,1,1), (0,1,1,1), (1,0,1,2)]
        - 1 snapping top, 2 plots on bottom: [(0,0,1,2), (1,0,1,1), (1,1,1,1)]
        - 1 large plot with 2 smaller ones to right: [(0,0,2,1), (0,1,1,1), (1,1,1,1)]
        - 1 large plot with 2 smaller ones to left: [(0,1,2,1), (0,0,1,1), (1,0,1,1)]
        - Sidebar with 3 plots on the left and one on the right: [(0,0,1,2), (1,0,1,2), (2,0,1,2), (0,2,3,1)]
        - Sidebar with 3 plots on the right and one on the left: [(0,0,3,1), (0,1,1,2), (1,1,1,2), (2,1,1,2)]
        - 1 spanning top, 3 plots on bottom: [(0,0,1,3), (1,0,1,1), (1,1,1,1), (1,2,1,1)]
        - 1 spanning bottom, 3 plots on top: [(0,0,1,1), (0,1,1,1), (0,2,1,1), (1,0,1,3)]
        - 3x3 grid with center empty: [(0,0,1,1), (0,1,1,1), (0,2,1,1), (1,0,1,1), (1,2,1,1), (2,0,1,1), (2,1,1,1), (2,2,1,1)]
        - Uneven grid: [(0,0,2,1), (0,1,1,1), (1,1,1,1), (2,0,1,2)]
        """
        # Calculate grid dimensions
        max_row = max(spec[0] + spec[2] for spec in layout_spec)
        max_col = max(spec[1] + spec[3] for spec in layout_spec)
        
        # Create figure and gridspec
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(max_row, max_col)
        
        # Create axes
        axes = []
        for row, col, rowspan, colspan in layout_spec:
            ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
            axes.append(ax)
        
        return fig, axes

    @classmethod
    def setup_google_font(cls, 
                          repo_name: str = 'notoserif', 
                          font_name: str = 'NotoSerif[wdth,wght]', 
                          save_locally: bool = True, 
                          force_download: bool = False) -> str:
        """
        Download a font from Google Fonts GitHub repository with caching.
        """
        # Check cache first
        cache_key = f"{repo_name}_{font_name}"
        if not force_download and cache_key in cls._font_cache:
            return cls._font_cache[cache_key]
            
        # Local file path
        local_path = f"{font_name}.ttf"
        
        # Only download if file doesn't exist or force_download is True
        if force_download or not os.path.exists(local_path):
            # GitHub raw content URL
            url = f'https://github.com/google/fonts/raw/main/ofl/{repo_name}/{font_name}.ttf'
            
            try:
                # Download font
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Save to local file
                if save_locally:
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
            except (requests.RequestException, IOError) as e:
                raise ValueError(f"Failed to download or save font: {e}")
        
        # Add the font
        fm.fontManager.addfont(local_path)
        fm._load_fontmanager(try_read_cache=False)
        font_prop = fm.FontProperties(fname=local_path)
        font_family = font_prop.get_name()
        
        # Cache the result
        cls._font_cache[cache_key] = font_family
        return font_family
    
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
            text: Text content for the insights section (supports multi-line text)
            position: Position of the insights box ('right', 'left', 'top', 'bottom')
            fontsize: Font size for the text
            color: Color of the text and separator line
            x: X-coordinate for text placement (overrides default position)
            y: Y-coordinate for text placement (overrides default position)
            weight: Font weight for the text
        """
        # Position lookup dictionary for cleaner code
        positions = {
            'right': ((1.03, 0.7), ([1.01, 1.01], [0.1, 0.9])),
            'left': ((-0.03, 0.7), ([-0.01, -0.01], [0.1, 0.9])),
            'top': ((0.5, 1.03), ([0.1, 0.9], [1.01, 1.01])),
            'bottom': ((0.5, -0.03), ([0.1, 0.9], [-0.01, -0.01]))
        }
        
        # Get position or default to right
        (default_x, default_y), line_coords = positions.get(position, positions['right'])
        
        # Use provided coordinates or defaults
        x = x if x is not None else default_x
        y = y if y is not None else default_y
        
        # Add separator line
        fig.lines.append(lines.Line2D(line_coords[0], line_coords[1], 
                                    transform=fig.transFigure,
                                    figure=fig, color=color, lw=0.5))
        
        # Add text with support for multi-line text
        fig.text(x, y, text, fontsize=fontsize, color=color, 
                fontweight=weight, verticalalignment='top')

    @staticmethod
    def binary_ratio_plot(data: pd.DataFrame,
                          column_name: str,
                          target_zero_name: str,
                          target_one_name: str,
                          ax: Optional[plt.Axes] = None,
                          font_color: str = 'white',
                          figsize: Tuple[float, float] = (6.5, 2)) -> plt.Axes:
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
        Leonardo._clean_axes(ax)
        
        # Calculate annotation positions dynamically
        pos0 = min(max(target_ratios[0] / 2, 10), 40)
        pos1 = max(min(target_ratios[0] + target_ratios[1] / 2, 90), 60)
        
        # Add annotations for both targets
        for idx, (pos, label) in enumerate(zip([pos0, pos1], [target_zero_name, target_one_name])):
            ax.annotate(f'{target_ratios[idx]:.2f} %',
                       xy=(pos, column_name),
                       va='center', ha='center', color=font_color)
            ax.annotate(label,
                       xy=(pos, -0.1),
                       va='center', ha='center', fontsize=10, color=font_color)
            
        return ax
    
    @staticmethod
    def stacked_bars_plot(data: pd.DataFrame,
                          x: str,
                          y: str,
                          ax: plt.Axes,
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
            normalize: How to normalize the data ('index', 'columns', or None)
            fmt: Format string for bar labels
            label_color: Color of the bar labels
            width: Width of the bars
            
        Returns:
            The matplotlib axes object with the plot
        """
        # Calculate crosstab once and cache it
        crosstab = pd.crosstab(data[x], data[y], normalize=normalize).mul(100)
        
        # Plot the data
        crosstab.plot(kind='barh', stacked=True, ax=ax, legend=False, width=width)
        
        # Configure axes
        ax.set_xlabel('Percentage (%)')
        ax.set_ylabel('')
        
        # Add labels to all containers at once
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, label_type='center', color=label_color)
        
        return ax
    
    @staticmethod
    def ridge_plot(data: pd.DataFrame, 
                   x_var: str, 
                   group_var: str, 
                   cmap=None, 
                   colors=None,
                   fontsize_facets: int = 12, 
                   height: float = 1, 
                   aspect: float = 15) -> sns.FacetGrid:
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
            ax.text(0, .4, label, color=color, ha="left", va="center", transform=ax.transAxes, fontsize=fontsize_facets)
        
        g.map(label, x_var)
        
        # Adjust layout and formatting
        g.figure.subplots_adjust(hspace=0.1)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        
        # Format axes using _clean_axes method, but keep bottom spine visible
        for ax in g.axes.flat:
            Leonardo._clean_axes(ax, spines_to_remove=['left', 'right', 'top'])
            ax.xaxis.set_visible(True)
            ax.tick_params(axis='x', colors='black')
            
        return g

    @staticmethod
    def dumbbell_plot(df: pd.DataFrame, 
                      group_col: str, 
                      category_col: str, 
                      value_col: str, 
                      ax: Optional[plt.Axes] = None,
                      figsize: Tuple[float, float] = (10, 4),
                      labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a dumbbell plot comparing two groups across categories.
        
        Args:
            df: DataFrame with the raw data
            group_col: Column name containing the two groups to compare
            category_col: Column name containing the categories for y-axis
            value_col: Column name containing the values to plot
            ax: Optional matplotlib axis to plot on
            figsize: Width and height of the figure in inches
            labels: Optional custom labels for the legend boxes (defaults to group values)
            
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
        Leonardo._clean_axes(ax)
        
        # Define range for y-axis and plot elements
        my_range = range(1, len(pivot_data) + 1)
        ax.hlines(y=my_range, xmin=pivot_data[groups[0]], xmax=pivot_data[groups[1]], color='gray', alpha=0.4)
        ax.scatter(pivot_data[groups[0]], my_range, s=100, label=groups[0])
        ax.scatter(pivot_data[groups[1]], my_range, s=100, label=groups[1])
        
        # Add mean lines
        ax.axvline(pivot_data[groups[0]].mean(), color='gray', linewidth=0.4, linestyle='dashdot')
        ax.axvline(pivot_data[groups[1]].mean(), color='gray', linewidth=0.4, linestyle='dashdot')
        
        # Set y-tick labels and remove tick marks
        ax.set_yticks(my_range)
        ax.set_yticklabels(pivot_data.index)
        ax.tick_params(axis='both', which='both', length=0)

        # Legend boxes
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9)
        legend_labels = labels if labels else groups
        
        # Position labels with fixed spacing
        label_spacing = 0.09  # Fixed space between labels
        pos2 = 0.98 - len(str(legend_labels[1])) * 0.01  # Second label position
        pos1 = pos2 - label_spacing - len(str(legend_labels[0])) * 0.01  # First label position

        # adding text boxes
        ax.text(pos1, 1.02, legend_labels[0], color='#0085a1', transform=ax.transAxes, fontsize=8, bbox=bbox_props)
        ax.text(pos2, 1.02, legend_labels[1], color='#242728', transform=ax.transAxes, fontsize=8, bbox=bbox_props)
        
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
        
        # Get top n items
        temp = df[:n]
        values = temp[value_col].values
        my_range = range(1, len(values)+1)
        
        # Plot lines and points in a vectorized way
        ax.hlines(y=my_range, xmin=0, xmax=values, color='gray', alpha=0.8)
        ax.scatter(values, my_range, s=markersize*10, color='#0085a1')
        
        # Remove spines
        Leonardo._clean_axes(ax)
        
        # Set y-tick labels
        ax.set_yticks(my_range)
        ax.set_yticklabels(temp[label_col])
        
        return fig, ax
