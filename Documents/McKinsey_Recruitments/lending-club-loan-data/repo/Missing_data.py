import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

# Importing the data
df = pd.read_csv('loan.csv',
                 low_memory=False)

# Removing columns with more than 50% of NaN
todrop = pd.DataFrame(np.transpose(np.array([list(df),
                                             df.isnull().sum(),
                                             df.isnull().sum() > len(df) * 0.5])))
todrop.columns = ['columns_name',
                  'null_counts',
                  'unusable']
temp = todrop.columns_name.where(todrop.unusable == 'True')
print("Columns with more than 50 percent of NaN:", list(temp.dropna()))
df = df.drop(list(temp.dropna()), axis=1)


def num_missing(x):
    return sum(x.isnull())


print("Missing values per column:", df.apply(num_missing, axis=0))
print("Categorical columns:", list(df.loc[:, df.dtypes == np.object]))
print("Unique values per column:", df.nunique())

# Preparing emp_lenght column to imputer
emp_len_dict = {'10+ years': 10,
                '< 1 year': 0,
                '1 year': 1,
                '3 years': 3,
                '8 years': 8,
                '9 years': 9,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '2 years': 2,
                '7 years': 7}

df = df.replace({'emp_length': emp_len_dict})


# Subset for Imputer strateqy: mean
df_imp_mean = pd.DataFrame(data=df,
                           columns=['annual_inc',
                                    'emp_length',
                                    'delinq_2yrs',
                                    'inq_last_6mths',
                                    'open_acc',
                                    'pub_rec',
                                    'revol_util',
                                    'total_acc',
                                    'tot_coll_amt',
                                    'tot_cur_bal',
                                    'total_rev_hi_lim'])

imputer = Imputer(missing_values=np.nan,
                  strategy='mean',
                  axis=0)
imputed_df_1 = pd.DataFrame(imputer.fit_transform(df_imp_mean))
imputed_df_1.columns = df_imp_mean.columns
imputed_df_1.index = df_imp_mean.index

# Subset for Imputer strateqy: most_frequent
df_imp_freq = pd.DataFrame(data=df,
                           columns=['collections_12_mths_ex_med',
                                    'acc_now_delinq'])

imputer = Imputer(missing_values=np.nan,
                  strategy='most_frequent',
                  axis=0)
imputed_df_2 = pd.DataFrame(imputer.fit_transform(df_imp_freq))
imputed_df_2.columns = df_imp_freq.columns
imputed_df_2.index = df_imp_freq.index

# Creating output for missing data preprocessing
frames = [imputed_df_1,
          imputed_df_2]
missing_data = pd.concat(objs=frames,
                         axis=1,
                         join='outer',
                         copy=False,
                         sort=False)

cols_to_use = df.columns.difference(missing_data.columns)
rest_of_data = pd.DataFrame(data=df,
                            columns=cols_to_use)
frames = [missing_data,
          rest_of_data]

missing_data_out = pd.concat(objs=frames,
                             axis=1,
                             join='outer',
                             copy=False,
                             sort=False)
missing_data_out.to_csv('missing_data_out.csv', encoding='utf-8')

# Check
missing_data_out.info()
