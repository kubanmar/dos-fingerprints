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
