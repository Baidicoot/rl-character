"""
Aesthetic Matplotlib Configuration Module
=========================================
Import this module to automatically apply beautiful styling to all your plots.
Just add: from aesthetic_config import *
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import rcParams
import seaborn as sns

# Color palettes - Professional research colors
COLORS = {
    # Matplotlib Tab10 - Great for categorical data, up to 10 categories
    'tab10': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    
    # Seaborn Deep - Muted version of tab10, more professional
    'deep': ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
             '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd'],
    
    # Seaborn Muted - Even more muted, great for academic papers
    'muted': ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4',
              '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c4e5'],
    
    # Set2 - Pastel qualitative, good for grouped data
    'set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
             '#ffd92f', '#e5c494', '#b3b3b3'],
    
    # Custom Academic - Professional colors for academic presentations
    'academic': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83',
                 '#1B5F33', '#B07C0A', '#6C5B7B', '#355C7D', '#2A363B'],
    
    # Earthy - Natural tones, good for environmental data
    'earthy': ['#8B4513', '#A0522D', '#D2691E', '#DAA520', '#228B22',
               '#6B8E23', '#556B2F', '#8FBC8F', '#2E8B57', '#3CB371'],
    
    # High contrast for accessibility
    'contrast': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                 '#0072B2', '#D55E00', '#CC79A7'],
    
    # Monochromatic variations
    'blues': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],
    'reds': ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090'],
    'greens': ['#00441b', '#238b45', '#41ae76', '#66c2a4', '#99d8c4']
}

# Configure matplotlib with aesthetic defaults
def setup_aesthetic_style():
    """Apply aesthetic styling to matplotlib"""
    
    # Use a clean, modern style as base
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Custom rcParams for beautiful plots
    rcParams.update({
        # Figure settings
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'figure.facecolor': '#FAFAFA',
        'figure.edgecolor': 'none',
        
        # Axes settings
        'axes.facecolor': '#FAFAFA',
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelsize': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.titlepad': 20,
        'axes.labelweight': 'medium',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': mpl.cycler(color=COLORS['deep']),
        
        # Grid settings
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # Line settings
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1.5,
        'lines.markeredgecolor': 'white',
        
        # Patch settings
        'patch.edgecolor': 'white',
        'patch.linewidth': 1.5,
        
        # Font settings
        'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 11,
        
        # Legend settings
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC',
        'legend.framealpha': 0.9,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        
        # Tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': '#666666',
        'ytick.color': '#666666',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'savefig.facecolor': '#FAFAFA',
        'savefig.edgecolor': 'none'
    })

# Apply the aesthetic style when module is imported
setup_aesthetic_style()

# Utility functions for beautiful plots
def set_plot_style(ax, xlabel='', ylabel='', title='', remove_spines=True):
    """Apply consistent styling to a plot"""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='medium', labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='medium', labelpad=10)
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

def grouped_bar_chart(data, xlabel='', ylabel='', title='', legend_title='', legend_loc='best', bar_width=0.8, 
                      group_spacing=1.0, figsize=None, colors=None, color_scale=None):
    """
    Create a beautiful grouped bar chart
    
    Parameters:
    -----------
    data : dict
        Dictionary where keys are group names and values are dicts of {label: value}
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Chart title
    bar_width : float
        Width of individual bars
    group_spacing : float
        Spacing between groups
    figsize : tuple
        Figure size (width, height)
    colors : list
        Custom color palette
    color_scale : str
        Matplotlib colormap name (e.g., 'viridis', 'plasma', 'tab10')
        If provided, overrides colors parameter
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if figsize is None:
        figsize = (12, 6)
    
    # Handle color selection
    if color_scale is not None:
        # Use matplotlib colormap
        cmap = plt.cm.get_cmap(color_scale)
        labels = list(data[list(data.keys())[0]].keys())
        n_colors = len(labels)
        colors = [cmap(i / (n_colors - 1)) if n_colors > 1 else cmap(0.5) 
                  for i in range(n_colors)]
    elif colors is None:
        colors = COLORS['deep']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions
    groups = list(data.keys())
    labels = list(data[groups[0]].keys())
    n_groups = len(groups)
    n_bars = len(labels)
    
    # Calculate bar positions
    group_width = bar_width * n_bars
    indices = np.arange(n_groups) * (group_width + group_spacing)
    
    # Plot bars
    for i, label in enumerate(labels):
        values = [data[group][label] for group in groups]
        positions = indices + i * bar_width
        bars = ax.bar(positions, values, bar_width, label=label, 
                      color=colors[i % len(colors)], alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}' if height < 100 else f'{int(height)}',
                   ha='center', va='bottom', fontsize=9, fontweight='medium')
    
    # Customize
    ax.set_xticks(indices + group_width/2 - bar_width/2)
    ax.set_xticklabels(groups)
    set_plot_style(ax, xlabel, ylabel, title)
    ax.legend(loc=legend_loc, framealpha=0.9, fontsize=10, title=legend_title)
    
    plt.tight_layout()
    return fig, ax

def line_plot(x, y_data, labels=None, xlabel='', ylabel='', title='', 
              figsize=None, colors=None, color_scale=None, markers=True):
    """
    Create a beautiful line plot
    
    Parameters:
    -----------
    x : array-like
        X-axis data
    y_data : array-like or list of arrays
        Y-axis data (single array or list for multiple lines)
    labels : list
        Labels for each line
    xlabel, ylabel, title : str
        Axis labels and title
    figsize : tuple
        Figure size
    colors : list
        Custom color palette
    color_scale : str
        Matplotlib colormap name (e.g., 'viridis', 'plasma', 'tab10')
    markers : bool
        Whether to show markers
    """
    if figsize is None:
        figsize = (10, 6)
    
    # Handle single line vs multiple lines
    if not isinstance(y_data[0], (list, np.ndarray)):
        y_data = [y_data]
    
    # Handle color selection
    if color_scale is not None:
        cmap = plt.cm.get_cmap(color_scale)
        n_lines = len(y_data)
        colors = [cmap(i / (n_lines - 1)) if n_lines > 1 else cmap(0.5) 
                  for i in range(n_lines)]
    elif colors is None:
        colors = COLORS['deep']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(y_data))]
    
    # Plot lines
    for i, (y, label) in enumerate(zip(y_data, labels)):
        marker = 'o' if markers else None
        ax.plot(x, y, label=label, color=colors[i % len(colors)], 
                marker=marker, markersize=6, linewidth=2.5, alpha=0.85)
    
    set_plot_style(ax, xlabel, ylabel, title)
    if len(y_data) > 1:
        ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    return fig, ax

def scatter_plot(x, y, c=None, s=None, xlabel='', ylabel='', title='', 
                 figsize=None, cmap='viridis', alpha=0.7):
    """
    Create a beautiful scatter plot
    
    Parameters:
    -----------
    x, y : array-like
        Data points
    c : array-like, optional
        Colors for points
    s : array-like or float, optional
        Sizes for points
    xlabel, ylabel, title : str
        Axis labels and title
    figsize : tuple
        Figure size
    cmap : str
        Colormap for colored points
    alpha : float
        Point transparency
    """
    if figsize is None:
        figsize = (10, 6)
    
    if s is None:
        s = 60
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(x, y, c=c, s=s, alpha=alpha, cmap=cmap, 
                        edgecolors='white', linewidth=1)
    
    if c is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Value', fontsize=11)
    
    set_plot_style(ax, xlabel, ylabel, title)
    plt.tight_layout()
    return fig, ax

def histogram(data, bins=20, xlabel='', ylabel='Frequency', title='', 
              figsize=None, color=None, alpha=0.75, kde=True):
    """
    Create a beautiful histogram with optional KDE
    
    Parameters:
    -----------
    data : array-like
        Data to plot
    bins : int
        Number of bins
    xlabel, ylabel, title : str
        Axis labels and title
    figsize : tuple
        Figure size
    color : str
        Bar color
    alpha : float
        Bar transparency
    kde : bool
        Whether to overlay KDE
    """
    if figsize is None:
        figsize = (10, 6)
    
    if color is None:
        color = COLORS['deep'][0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins, patches = ax.hist(data, bins=bins, color=color, alpha=alpha, 
                              edgecolor='white', linewidth=1.2)
    
    # Add KDE if requested
    if kde:
        ax2 = ax.twinx()
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax2.plot(x_range, kde(x_range), color=COLORS['deep'][1], 
                linewidth=2.5, label='KDE')
        ax2.set_ylabel('Density', fontsize=12, fontweight='medium')
        ax2.tick_params(axis='y', labelsize=10)
        ax2.grid(False)
    
    set_plot_style(ax, xlabel, ylabel, title)
    plt.tight_layout()
    return fig, ax

def pie_chart(data, labels, title='', figsize=None, colors=None, 
              explode=None, autopct='%1.1f%%'):
    """
    Create a beautiful pie chart
    
    Parameters:
    -----------
    data : array-like
        Values for each slice
    labels : list
        Labels for each slice
    title : str
        Chart title
    figsize : tuple
        Figure size
    colors : list
        Custom colors
    explode : array-like
        Explosion values for each slice
    autopct : str
        Format for percentage labels
    """
    if figsize is None:
        figsize = (8, 8)
    
    if colors is None:
        colors = COLORS['set2']  # Pastel colors work well for pie charts
    
    fig, ax = plt.subplots(figsize=figsize)
    
    wedges, texts, autotexts = ax.pie(data, labels=labels, colors=colors, 
                                      explode=explode, autopct=autopct,
                                      startangle=90, pctdistance=0.85,
                                      wedgeprops={'edgecolor': 'white', 
                                                 'linewidth': 2})
    
    # Beautify text
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('medium')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax

def heatmap(data, xlabels=None, ylabels=None, xlabel='', ylabel='', 
            title='', figsize=None, cmap='RdBu_r', fmt='.2f', 
            annot=True, cbar_label=''):
    """
    Create a beautiful heatmap
    
    Parameters:
    -----------
    data : 2D array-like
        Data to plot
    xlabels, ylabels : list
        Tick labels
    xlabel, ylabel, title : str
        Axis labels and title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    fmt : str
        Format for annotations
    annot : bool
        Whether to annotate cells
    cbar_label : str
        Colorbar label
    """
    if figsize is None:
        figsize = (10, 8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Set ticks
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=11)
    
    # Add annotations
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, format(data[i, j], fmt),
                             ha="center", va="center", color="black", fontsize=9)
    
    set_plot_style(ax, xlabel, ylabel, title, remove_spines=False)
    ax.grid(False)
    
    plt.tight_layout()
    return fig, ax

# Example usage and color palette display
def show_color_palettes():
    """Display all available color palettes"""
    fig, axes = plt.subplots(len(COLORS), 1, figsize=(10, len(COLORS) * 1.5))
    
    for idx, (name, colors) in enumerate(COLORS.items()):
        ax = axes[idx] if len(COLORS) > 1 else axes
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
            ax.text(i + 0.5, 0.5, color, ha='center', va='center', 
                   fontsize=8, color='white' if i < len(colors)/2 else 'black')
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_title(f'{name.title()} Palette', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

# Set default color palette
def set_color_palette(palette_name='deep'):
    """
    Change the default color palette
    
    Available palettes:
    - 'tab10': Matplotlib default, great for categorical data
    - 'deep': Seaborn deep, muted professional colors
    - 'muted': Seaborn muted, even more subtle for papers
    - 'set2': Pastel qualitative colors
    - 'academic': Custom professional academic colors
    - 'earthy': Natural earth tones
    - 'contrast': High contrast for accessibility
    - 'blues', 'reds', 'greens': Monochromatic variations
    """
    if palette_name in COLORS:
        rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS[palette_name])
    else:
        print(f"Palette '{palette_name}' not found. Available: {list(COLORS.keys())}")

# Convenience function to save with high quality
def save_figure(fig, filename, dpi=300, transparent=False):
    """Save figure with high quality settings"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), 
                edgecolor='none', transparent=transparent)




# Example of how to use this module:
if __name__ == "__main__":
    # Example grouped bar chart with viridis colormap
    data = {
        'Q1': {'Product A': 45, 'Product B': 38, 'Product C': 52},
        'Q2': {'Product A': 50, 'Product B': 42, 'Product C': 58},
        'Q3': {'Product A': 48, 'Product B': 45, 'Product C': 61},
        'Q4': {'Product A': 52, 'Product B': 48, 'Product C': 65}
    }
    
    # Using custom color palette
    fig1, ax1 = grouped_bar_chart(data, xlabel='Quarter', ylabel='Sales (thousands)', 
                                  title='Quarterly Sales by Product - Deep Palette')
    plt.show()
    
    # Using matplotlib colormap
    fig2, ax2 = grouped_bar_chart(data, xlabel='Quarter', ylabel='Sales (thousands)', 
                                  title='Quarterly Sales by Product - Viridis Scale',
                                  color_scale='viridis')
    plt.show()