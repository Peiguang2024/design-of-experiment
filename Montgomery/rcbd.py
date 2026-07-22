def rcbd(df, treatment, block, result):
    import matplotlib.pyplot as plt 
    import numpy as np 
    import pandas as pd 
    from scipy import stats
    '''
    Anava for the randomized complete block design (RCBD)
    df is a pandas dataframe includng one treatment column, one block column, and one result column
    It returns the f statistic and p value for the treatment
    '''
    df['ave'] = df[result].mean() # grand average
    df['residual'] = df[result] - df['ave'] # residual
    df['t_ave'] = df.groupby(treatment)[result].transform('mean') # treatment average
    df['t_dev'] = df['t_ave'] - df['ave'] # treatment deviations from grand average
    df['b_ave'] = df.groupby(block)[result].transform('mean') # block average
    df['b_dev'] = df['b_ave'] - df['ave'] # block deviations from grand average

    ss_total = (df['residual']**2).sum() # sum of squares of residuals

    sst = (df['t_dev']**2).sum() # sum of squares of treatment deviations
    dft = df[treatment].nunique() - 1 # degrees of freedom of treatments
    mst = sst/dft # mean square of treatment deviations

    ssb = (df['b_dev']**2).sum() # sum of squares of block deviations
    dfb = df[block].nunique()-1 # degrees of freedom of blocks
    msb = ssb/dft # mean square of block deviations

    sse = ss_total - sst - ssb # sum of squares of errors
    dfe = dft * dfb # degrees of freedom of errors
    mse = sse/dfe # mean square of erros

    ft = mst / mse # f statistic for treatment
    pt = stats.f.sf(ft, dfn=dft, dfd=dfe) # p value for treatment

    fb = msb / mse # f statistic for block
    pb = stats.f.sf(fb, dfn=dfb, dfd=dfe) # p value for block

    return {'f_statistic': ft, 'p_value': pt, 'mean_square_error': mse, 'degrees_of_freedom_error': dfe, 'n_blocks': df[block].nunique()}, df[[treatment, 't_ave']].drop_duplicates().reset_index(drop=True)


def t_scatter_plot():

    