import matplotlib.pyplot as plt
from bitarray import bitarray

def plot_FP_in_grid(byte_fingerprint, grid, show = True, label = '', axes = None, **kwargs):
    x=[]
    y=[]
    all_width=[]
    bin_fp=bitarray()
    bin_fp.frombytes(bytes.fromhex(byte_fingerprint.bins))
    grid_indices=byte_fingerprint.indices
    plotgrid=grid.grid()
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
    if axes == None:
        plt.bar(x,y,width=all_width,align='edge', label = label, **kwargs)
    else:
        axes.bar(x,y,width=all_width,align='edge', label = label, **kwargs)
    if show:
        plt.show()

def plot_horizontal_lines(grid_array: list, plot_style = {'c':'k'}, axes = None, only_top = False) -> None:
    if axes == None:
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


def plot_vertical_lines(grid_array: list, plot_style = {'c':'k'}, axes = None) -> None:
    if axes == None:
        axes = plt.gca()
    for e, rhos in grid_array:
        axes.plot([e,e], [0,max(rhos)], **plot_style)

def plot_grid(grid_array: list, horizontal = True, horizontal_only_top = False, vertical = True, plot_style = {'c':'k', 'linewidth' : 0.5}, figure = True, figsize = (15,10), limits = [-3.2, 3.2, 0, 2], show =True, axes = None) -> None:
    if figure:
        plt.figure(figsize=figsize)
    if axes == None:
        axes = plt.gca()
    if vertical:
        plot_vertical_lines(grid_array, plot_style=plot_style, axes = axes)
    if horizontal:
        plot_horizontal_lines(grid_array, plot_style=plot_style, axes = axes, only_top=horizontal_only_top)
    axes.set_xlim(*limits[:2])
    axes.set_ylim(*limits[2:])
    if show:
        plt.show()