import pandas as pd

# Importing the data
df = pd.read_csv('pre_sele_feature_out.csv',
                 index_col=0,
                 low_memory=False)

# Data selection based on Recursive Feature Elimination and column for default client
top10_RFE_df = pd.DataFrame(data=df, columns=['collection_recovery_fee',
                                              'funded_amnt',
                                              'funded_amnt_inv',
                                              'last_pymnt_amnt',
                                              'out_prncp',
                                              'out_prncp_inv',
                                              'recoveries',
                                              'total_pymnt',
                                              'total_pymnt_inv',
                                              'total_rec_prncp',
                                              'Default_Binary'])

top10_RFE_df.to_csv('top10_RFE.csv', encoding='utf-8')

top7_RFE_df = pd.DataFrame(data=df, columns=['funded_amnt',
                                             'last_pymnt_amnt',
                                             'out_prncp',
                                             'recoveries',
                                             'total_rec_prncp',
                                             'out_prncp_inv',
                                             'collection_recovery_fee',
                                             'Default_Binary'])

top7_RFE_df.to_csv('top7_RFE.csv', encoding='utf-8')

top5_RFE_df = pd.DataFrame(data=df, columns=['funded_amnt',
                                             'last_pymnt_amnt',
                                             'out_prncp',
                                             'recoveries',
                                             'total_rec_prncp',
                                             'Default_Binary'])

top5_RFE_df.to_csv('top5_RFE.csv', encoding='utf-8')
