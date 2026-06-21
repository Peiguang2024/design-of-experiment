def tukey(df, cat_col, result_col, alpha):
    '''
    The dataframe df should be in the following format. The first column is the treatment of each result in the second column.
    '''
    import numpy as np
    import pandas as pd
    from scipy import stats
    cat = df[cat_col].unique()
    # results = df.groupby(cat_col)[result_col].apply(list).values
    # generate treatment pairs
    from itertools import combinations
    cat_pair = []
    cat_pair.extend(combinations(cat,2))

    # compute the q statistic for each of the pairs
    df['ave'] = df[result_col].mean()
    df['t_ave'] = df.groupby(cat_col)[result_col].transform('mean')
    df['residual'] = df[result_col] - df['t_ave']
    SSE = (df.residual**2).sum()
    MSE = SSE/(df.shape[0]-len(cat))
    
    qs = []
    for pair in cat_pair:
        ave_0 = df[df[cat_col]==pair[0]][result_col].mean()
        n_0 = df[df[cat_col]==pair[0]][result_col].shape[0]
        ave_1 = df[df[cat_col]==pair[1]][result_col].mean()
        n_1 = df[df[cat_col]==pair[1]].shape[0]
        q = np.abs(ave_0 - ave_1) / (np.sqrt(MSE*(1/n_0)+1/n_1)/np.sqrt(2))
        qs.append(q)

    return pd.DataFrame({
        'pair': cat_pair+['critical_q'],
        'q': qs+[stats.studentized_range.isf(alpha, k=len(cat), df=df.shape[0]-len(cat))]
    })