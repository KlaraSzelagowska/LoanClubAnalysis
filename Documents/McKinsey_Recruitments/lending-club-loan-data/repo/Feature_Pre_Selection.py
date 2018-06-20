import pandas as pd

# Importing the data
df = pd.read_csv('encoded_data_out.csv',
                 index_col=0,
                 low_memory=False)

df.info()
df.corr().to_csv('data_correlation.csv')

# Adding Default_Binary column

df['Default_Binary'] = df.loan_status.isin([
    'Default',
    'Charged Off',
    'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off'
])
df['Default_Binary'] = df.Default_Binary.astype(int)

# Columns that not be used in model
pre_sele_feature_out = df.drop(['id',
                                'member_id',
                                'emp_title',
                                'loan_status',
                                'issue_d',
                                'url',
                                'title',
                                'earliest_cr_line',
                                'last_pymnt_d',
                                'next_pymnt_d',
                                'last_credit_pull_d',
                                'policy_code'],
                               axis=1)


pre_sele_feature_out.to_csv('pre_sele_feature_out.csv', encoding='utf-8')
pre_sele_feature_out.info()
