import pandas as pd

# Importing the data
df = pd.read_csv('pre_sele_feature_out.csv',
                 index_col=0,
                 low_memory=False)

# Data selection based on XGBoost Feature Selection and column for default client

top10_XGB_df = pd.DataFrame(data=df, columns=['out_prncp',
                                              'total_rec_prncp',
                                              'total_rec_late_fee',
                                              'recoveries',
                                              'last_pymnt_amnt',
                                              'installment',
                                              'int_rate',
                                              'funded_amnt',
                                              'total_rec_int',
                                              'zip_code',
                                              'Default_Binary'])

top10_XGB_df.to_csv('top10_XGB.csv', encoding='utf-8')

top7_XGB_df = pd.DataFrame(data=df, columns=['out_prncp',
                                             'total_rec_prncp',
                                             'total_rec_late_fee',
                                             'recoveries',
                                             'last_pymnt_amnt',
                                             'installment',
                                             'int_rate',
                                             'Default_Binary'])

top7_XGB_df.to_csv('top7_XGB.csv', encoding='utf-8')

top5_XGB_df = pd.DataFrame(data=df, columns=['out_prncp',
                                             'total_rec_prncp',
                                             'total_rec_late_fee',
                                             'recoveries',
                                             'last_pymnt_amnt',
                                             'Default_Binary'])

top5_XGB_df.to_csv('top5_XGB.csv', encoding='utf-8')
