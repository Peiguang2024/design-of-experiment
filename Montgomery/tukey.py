def tukey(df, treatment_col, result_col, alpha):
    '''
    The dataframe df should be in the following format. The first column is the treatment of each result in the second column.
    '''
    import numpy as np
    import pandas as pd
    from scipy import stats
    treatment = df[treatment_col].unique()
    # results = df.groupby(treatment_col)[result_col].apply(list).values
    # generate treatment pairs
    from itertools import combinations
    treatment_pair = []
    treatment_pair.extend(combinations(treatment,2))

    # compute the q statistic for each of the pairs
    df['ave'] = df[result_col].mean()
    df['t_ave'] = df.groupby(treatment_col)[result_col].transform('mean')
    df['residual'] = df[result_col] - df['t_ave']
    SSE = (df.residual**2).sum()
    MSE = SSE/(df.shape[0]-len(treatment))
    
    qs = []
    for pair in treatment_pair:
        ave_0 = df[df[treatment_col]==pair[0]][result_col].mean()
        n_0 = df[df[treatment_col]==pair[0]][result_col].shape[0]
        ave_1 = df[df[treatment_col]==pair[1]][result_col].mean()
        n_1 = df[df[treatment_col]==pair[1]].shape[0]
        q = np.abs(ave_0 - ave_1) / (np.sqrt(MSE*(1/n_0)+1/n_1)/np.sqrt(2))
        qs.append(q)

    return pd.DataFrame({
        'pair': treatment_pair+['critical_q'],
        'q': qs+[stats.studentized_range.isf(alpha, k=len(treatment), df=df.shape[0]-len(treatment))]
    })