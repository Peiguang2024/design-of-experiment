def mse_gen(df, treatment_col, result_col):
  '''
    This function takes a dataframe (df) with one column (treatment_col) for treatment and a second column for the test results (result_col), 
    and returns the mean square of the within group deviation (an estimate of the standard deviation of the sample) and corresponding degrees of freedom.
    '''


    a = df.groupby(treatment_col).ngroups
    df['ave']= df.weeks.mean()
    df['t_ave'] = df.groupby(treatment_col)[result_col].transform('mean')
    df['residual'] = df['weeks'] - df['t_ave']
    sst = ((df['ave'] - df['t_ave'])**2).sum() # sum of squares of btw group deviation
    sse = (df['residual']**2).sum() # sum of squares of within group deviation
    mse = sse / (df.shape[0] - a) # an estimate of variance
    return mse, (df.shape[0] - a)