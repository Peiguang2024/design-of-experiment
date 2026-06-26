def bartlett_test(df, treatment_col, result_col):
    '''
    This function takes a dataframe (df) with one column (treatment_col) for treatment and a second column for the test results (result_col), 
    and returns the statistical test results for equality of variance among the different treatments (chi2) and the p value.

    Then null hypothesis is that all variances in the treatments are equal.

    Bartlett's test
    chi-square = 2.3026 * q / c (see blow)
    '''

    import numpy as np
    import pandas as pd
    from scipy import stats
    

    N = df.shape[0] # total samples
    a = df.groupby(treatment_col).ngroups # total treatments
    sizes = df.groupby(treatment_col).size().to_numpy() # sample size of each group

    si2 = df.groupby(treatment_col)[result_col].var(ddof=1).to_numpy() # variance of each treatment
    sp2 = si2.dot(sizes.reshape(-1,1) - 1) / (N - a) # variance of the the whole sample

    q = (N - a) * np.log10(sp2) - ((sizes - 1).dot(np.log10(si2.reshape(-1,1)))).sum()
    c = 1 + 1/(3*(a-1))*((1/(sizes-1)).sum() - 1/(N - a))

    chi2 = (2.3026 * q / c)[0]
    p = stats.chi2.sf(chi2,df=a-1)

    return chi2, p



