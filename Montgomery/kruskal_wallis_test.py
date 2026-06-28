def kruskal_wallis_test(df, treatment_col, result_col):
    '''
    This function is for the nonparametric test, the Kruskal-Wallis test. 
    It takes a dataframe (df) with one column (treatment_col) for treatment and a second column for the test results (result_col), 
    and returns the h statistic and the p value.
    Then null hypothesis is that all variances in the treatments are equal.
    '''
    import numpy as np
    import pandas as pd
    from scipy import stats

    N = df.shape[0]
    a = df.groupby(treatment_col).ngroups
    df.sort_values(result_col, inplace=True)
    df['rk'] = np.arange(1,N+1)
    df['rk'] = df.groupby(result_col)['rk'].transform('mean')

    # mse = (((df['rk']**2).sum() - N*(N+1)**2/4)) / (N-1) # based on equation from the book
    mse = df['rk'].var(ddof=1) # within treatment deviations
    # sst = (df.groupby(treatment_col)['rk'].sum()**2 / df.groupby(treatment_col).size).sum() - 25 * 26**2 / 4 # based on equation above
    sst = ((df['rk'].mean() - df.groupby(treatment_col)['rk'].transform('mean'))**2).sum() # sum of squares of treatment deviations
    # print(N, a, mse, sst)
    q = sst/mse
    p = stats.chi2.sf(q, df=a-1)
    return q, p