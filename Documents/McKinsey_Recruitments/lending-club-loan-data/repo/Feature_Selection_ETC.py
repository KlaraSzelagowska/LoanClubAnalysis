import pandas as pd

# Importing the data
df = pd.read_csv('pre_sele_feature_out.csv',
                 index_col=0,
                 low_memory=False)

# Data selection based on ExtraTreesClassifier Feature Selection and column for default client
top10_ETC_df = pd.DataFrame(data=df, columns=['out_prncp',
                                              'out_prncp_inv',
                                              'total_rec_prncp',
                                              'recoveries',
                                              'funded_amnt',
                                              'funded_amnt_inv',
                                              'total_pymnt',
                                              'loan_amnt',
                                              'total_pymnt_inv',
                                              'last_pymnt_amnt',
                                              'Default_Binary'])

top10_ETC_df.to_csv('top10_ETC.csv', encoding='utf-8')

top7_ETC_df = pd.DataFrame(data=df, columns=['out_prncp',
                                             'out_prncp_inv',
                                             'total_rec_prncp',
                                             'recoveries',
                                             'funded_amnt',
                                             'funded_amnt_inv',
                                             'total_pymnt',
                                             'Default_Binary'])

top7_ETC_df.to_csv('top7_ETC.csv', encoding='utf-8')

top5_ETC_df = pd.DataFrame(data=df, columns=['out_prncp',
                                             'out_prncp_inv',
                                             'total_rec_prncp',
                                             'recoveries',
                                             'funded_amnt',
                                             'Default_Binary'])

top5_ETC_df.to_csv('top5_ETC.csv', encoding='utf-8')
