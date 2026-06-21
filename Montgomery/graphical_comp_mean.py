def graphical_comp_mean(df, treatment_col, result_col, alpha):
    '''
    This function is to create a graphical comparison of means following an analysis of variance (one way)
    The dataframe df has the first column for treatments, and the second for results
    '''
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt

    # compute mean square of residuals, an estimate of the variance of the dataset
    df['ave'] = df[result_col].mean()
    df['t_ave'] = df.groupby(treatment_col)[result_col].transform('mean')
    df['residual'] = df[result_col] - df['t_ave']
    SSE = (df.residual**2).sum()
    num_total = df.shape[0]
    num_treatments = len(df[treatment_col].unique())
    # print(num_total, num_treatments)
    sample_size = df.shape[0]/num_treatments
    MSE = SSE/(num_total-num_treatments)

    # generate x and y to plot the t distribution
    loc = df[result_col].mean()
    scale = np.sqrt(MSE/sample_size)
    t = stats.t.isf(alpha/2, df=num_total-num_treatments) # use it to generate range of the t distribution to plot

    x = np.linspace(-t,t,1001) * scale + loc
    # print(x.min(),x.max())
    y = stats.t.pdf(x, loc=loc, scale=scale, df=num_total-num_treatments)

    fig,ax = plt.subplots()

    # plot the distribution
    ax.plot(x,y)
    # plot the length of scale
    ax.plot([loc-scale/2, loc+scale/2], [0.03,0.03])
    ax.text(loc, 0.04, 'standard error', ha='center')

    # plot the scatters for the means
    treatment_means = df.groupby(treatment_col)[result_col].mean()
    ax.scatter(treatment_means.values, [0]*num_treatments)
    for i in range(num_treatments):
        ax.text(treatment_means.values[i], 0.01, treatment_means.index[i], ha='center')
    
    return fig,ax

    