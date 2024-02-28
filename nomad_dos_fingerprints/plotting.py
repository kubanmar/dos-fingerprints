from typing import Union, Iterable

import matplotlib.pyplot as plt

from nomad_dos_fingerprints import DOSFingerprint, Grid

def _apply_grid_offset(grid_array: list, offset: float):
    for idx in range(len(grid_array)):
        grid_array[idx][0] += offset
    return grid_array

def plot_fingerprint_in_grid(fingerprint: DOSFingerprint, 
                             show: bool = True, 
                             label: str = '', 
                             axes: Union[plt.Axes, None] = None, **kwargs) -> None:
    """
    Plot a fingerprint in its grid representation.

    **Arguments:**

    fingerprint: `nomad_dos_fingerprints.DOSFingerprint`
        Fingerprint to be plotted

    **Keyword arguments:**

    show: `bool`
        Show the plot.

        default = `True`
    
    label: `str`
        Label for the plot. Labels are required for legends in plots.

        default: `''`

    axes: `matplotlib.pyplot.Axes` or `None`
        Axes to draw fingerprint on. If `None`, create new axes.

        default: `None`
    
    Additional keyword arguments are passed to matplotlib.pyplot.bar(*, **kwargs).
    """
    x, y, all_width = [], [], []
    bin_fp=fingerprint.get_bitarray()
    grid_indices=fingerprint.indices
    grid = Grid.create(grid_id=fingerprint.grid_id)
    plotgrid=_apply_grid_offset(grid.grid(), grid.e_ref)
    plotgrid=plotgrid[grid_indices[0]:grid_indices[1]]

    bit_position=0
    for index,item in enumerate(plotgrid):
        if index<len(plotgrid)-1:
            width=plotgrid[index+1][0]-item[0]
        else:
            width=abs(item[0]-plotgrid[index-1][0])
        for idx, dos_value in enumerate(item[1]):
            if bin_fp[bit_position]==1 and (idx == len(item[1])-1 or bin_fp[bit_position+1] == 0 ):
                x.append(item[0])
                y.append(dos_value)
                all_width.append(width)
            bit_position+=1
    if axes is None:
        plt.bar(x,y,width=all_width,align='edge', label = label, **kwargs)
    else:
        axes.bar(x,y,width=all_width,align='edge', label = label, **kwargs)
    if show:
        plt.show()

def plot_horizontal_lines(grid: Grid, 
                          plot_style: dict = {'c':'k'}, 
                          axes: Union[plt.Axes, None] = None, 
                          only_top: bool = False) -> None:
    """
    Plot horizontal lines of a grid.

    **Arguments:**

    grid: `nomad_dos_fingerprints.Grid`
        Grid to plot.

    **Keyword arguments:**
    
    plot_style: `dict`
        Style of grid lines

        default: `{'c' : 'k'}` 

    axes: `matplotlib.pyplot.Axes` or `None`
        Axes to draw on. If `None`, create new axes.

        default: `None`

    only_top: `bool`
        Show only the highest line

        default: `False`
    """
    grid_array = _apply_grid_offset(grid.grid(), grid.e_ref)
    if axes is None:
        axes = plt.gca()
    e_, r_ = grid_array[0]
    for idx, entry in enumerate(grid_array[1:]):
        e, rhos = entry
        if list(rhos) == list(r_):
            continue
        elif rhos[-1] > r_[-1]: # increasing bin height
            if only_top:
                axes.plot([e_, e], [max(r_), max(r_)], **plot_style)
            else:
                for rho in r_:
                    axes.plot([e_, e], [rho, rho], **plot_style)
            e_, r_ = entry
        elif rhos[-1] < r_[-1]: # decreasing bin height
            if only_top:
                axes.plot([e_, grid_array[idx][0]], [max(r_), max(r_)], **plot_style)
            else:
                for rho in r_:
                    axes.plot([e_, grid_array[idx][0]], [rho, rho], **plot_style)
            e_, r_ = grid_array[idx][0], rhos # energy of the *last* bin
    if only_top:
        axes.plot([e_, e], [max(r_), max(r_)], **plot_style)
    else:
        for rho in r_:
            axes.plot([e_, e], [rho, rho], **plot_style)


def plot_vertical_lines(grid: Grid, 
                        plot_style: dict = {'c':'k'}, 
                        axes: Union[plt.Axes, None] = None) -> None:
    """
    Plot horizontal lines of a grid.

    **Arguments:**

    grid: `nomad_dos_fingerprints.Grid`
        Grid to plot.

    **Keyword arguments:**
    
    plot_style: `dict`
        Style of grid lines

        default: `{'c' : 'k'}` 

    axes: `matplotlib.pyplot.Axes` or `None`
        Axes to draw on. If `None`, create new axes.

        default: `None`
    """
    grid_array = _apply_grid_offset(grid.grid(), offset=grid.e_ref)
    if axes is None:
        axes = plt.gca()
    for e, rhos in grid_array:
        axes.plot([e,e], [0,max(rhos)], **plot_style)

def plot_grid(grid: Grid, 
              vertical: bool = True, 
              horizontal: bool = True, 
              horizontal_only_top: bool = False, 
              plot_style: dict = {'c':'k', 'linewidth' : 0.5}, 
              figure: bool = True, 
              figsize: tuple = (15,10), 
              limits: Union[Iterable, None] = None, 
              show: bool = True, 
              axes:  Union[plt.Axes, None] = None) -> None:
    """
    Plot a Grid.

    **Arguments:**

    grid: `nomad_dos_fingerprints.Grid`
        Grid to plot

    **Keyword arguments:**

    vertical: `bool`
        Plot vertical grid lines

        default: `True`

    horizontal: `bool`
        Plot horizontal grid lines

        default: `True`
    
    horizontal_only_top: `bool`
        Plot only uppermost horizontal grid lines

        default: `False`

    plot_style: `dict`
        Style of grid lines

        default: `{'c' : 'k'}` 

    figure: `bool`
        Create figure

        default: `True`

    figsize: `tuple`
        Size of create figure,
        ignored if `figure` is `False`.

        default: `(15,10)`
    
    limits: `list` of `float` or `None`
        if not `None`, contains the axis limits:
        `[x_min, x_max, y_min, y_max]`

        default: `None`

    show: `bool`
        Show the plot.

        default = `True`

    axes: `matplotlib.pyplot.Axes` or `None`
        Axes to draw on. If `None`, create new axes.

        default: `None`

    """
    if figure:
        plt.figure(figsize=figsize)
    if axes is None:
        axes = plt.gca()
    if vertical:
        plot_vertical_lines(grid, plot_style=plot_style, axes = axes)
    if horizontal:
        plot_horizontal_lines(grid, plot_style=plot_style, axes = axes, only_top=horizontal_only_top)
    if limits is not None:
        axes.set_xlim(*limits[:2])
        axes.set_ylim(*limits[2:])
    if show:
        plt.show()
