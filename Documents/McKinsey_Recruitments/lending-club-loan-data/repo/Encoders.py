import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from mlencoders.weight_of_evidence_encoder import WeightOfEvidenceEncoder

# Importing the data
df = pd.read_csv('missing_data_out.csv',
                 index_col=0,
                 low_memory=False)

print("Categorical columns:", list(df.loc[:, df.dtypes == np.object]))
print("Unique values per column:", df.nunique())

# Encoding categorical variables:
df_encoder = pd.DataFrame(data=df,
                          columns=['term',
                                   'grade',
                                   'home_ownership',
                                   'verification_status',
                                   'pymnt_plan',
                                   'purpose',
                                   'initial_list_status',
                                   'application_type'])


d = defaultdict(LabelEncoder)

fit = df_encoder.apply(lambda x: d[x.name].fit_transform(x))
fit.apply(lambda x: d[x.name].inverse_transform(x))
encoded_df_1 = df_encoder.apply(lambda x: d[x.name].transform(x))

encoded_df_2 = pd.get_dummies(encoded_df_1,
                              columns=encoded_df_1.columns.tolist(),
                              drop_first=True)

# Encoding high cardinality variables by Weight of Evidence encoder
df_woe = pd.DataFrame(data=df,
                      columns=['sub_grade',
                               'zip_code',
                               'addr_state'])

dumy_df = pd.DataFrame()
dumy_df['Default_Binary'] = df.loan_status.isin([
    'Default',
    'Charged Off',
    'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off'
])
dumy_df['Default_Binary'] = dumy_df.Default_Binary.astype(int)
y = pd.Series(dumy_df.Default_Binary)

encoder = WeightOfEvidenceEncoder(cols=['sub_grade',
                                        'zip_code',
                                        'addr_state'])
df_woe_1 = encoder.fit_transform(df_woe, y)

# Creating encoded data dataset
frames = [encoded_df_2,
          df_woe_1]
encoded_data = pd.concat(objs=frames,
                         axis=1,
                         join='outer',
                         copy=False,
                         sort=False)

# Extracting used columns names
frames = [df_encoder,
          df_woe]
encoded_col = pd.concat(objs=frames,
                        axis=1,
                        join='outer',
                        copy=False,
                        sort=False)

col_to_use = df.columns.difference(encoded_col.columns)

rest_of_data = pd.DataFrame(data=df,
                            columns=col_to_use)

# Creating output for encoded data preprocessing
frames = [encoded_data,
          rest_of_data]

encoded_data_out = pd.concat(objs=frames,
                             axis=1,
                             join='outer',
                             copy=False,
                             sort=False)
encoded_data_out.to_csv('encoded_data_out.csv')

encoded_data_out.info()
