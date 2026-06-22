def normality_plot(s):
    '''
    this function is to generate the normality plot for residuals to check the normality of the original data.
    "s" is the pandas serie of the residuals
    It returns the nromality plot
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy import stats

    fig, ax = plt.subplots()
    ax.scatter(s.sort_values(), -stats.norm.isf(np.linspace(0,1,len(s)+1)[:-1]+1/2/len(s))) # sort the values and plot the scatter
    ax.set_yticks(-stats.norm.isf(np.linspace(0.1,0.9,9)), np.linspace(0.1,0.9,9).round(2))

    return fig, ax

