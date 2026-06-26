def levene_test(df, treatment_col, result_col):
    '''
    This function takes a dataframe (df) with one column (treatment_col) for treatment and a second column for the test results (result_col), 
    and returns the statistical test results for equality of variance among the different treatments and the p value.
    Then null hypothesis is that all variances in the treatments are equal.
    '''
    import numpy as np
    import pandas as pd
    from scipy import stats

    df['med'] = df.groupby(treatment_col)[result_col].transform('median')
    df['dev'] = np.abs(df[result_col] - df['med'])

    return stats.f_oneway(*df.groupby(treatment_col)['dev'].apply(list).values)