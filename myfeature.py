from IPython.display import clear_output 
import pandas as pd
import calendar
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from IPython.display import clear_output as cclear
from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
import lightgbm as lgb
from math import sqrt
from itertools import zip_longest
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randFloat
import joblib
import xgboost as xgb

'''
This function does feature engineering on sales_train_ev or sales_train_val
There is another feature engineering function for adding columns to dataframe containing rows of only onw store
'''
def feature_engineer(df):
    day_columns = list(df.columns[6:])
    other_var = list(df.columns[:6])
    
    print('Melting out...')
    df = pd.melt(df, id_vars = other_var, value_vars = day_columns)
    df = df.rename(columns = {"variable": "d", "value": "unit_sale"})
    # print(df.shape)
    
    print('Adding Feature \'date\'...')
    cal_dict = dict(zip(calender.d,calender.date))
    df["date"] = df["d"].map(cal_dict)
    # df.head()
    
    print('Adding Feature \'day_of_week\'...')
    day_of_week_dict = dict(zip(calender.d,calender.wday))
    df['day_of_week'] = df["d"].map(day_of_week_dict)

    print('Adding Feature \'month_no\'...')
    month_no_dict = dict(zip(calender.d,calender.month))
    df['month_no'] = df["d"].map(month_no_dict)
    
    print('Adding Feature \'day_of_month\'...')
    l = [i[-2:] for i in list(calender.date)]
    calender['day_of_month'] = l
    
    day_of_month_dict = dict(zip(calender.d,calender.day_of_month))
    df['day_of_month'] = df["d"].map(day_of_month_dict)
    
    print('Done.')
    return df

def add_moving_avg_col(new_df,N, col_name):
    l = np.convolve(list(new_df[col_name]), np.ones((N,))/N, mode='valid')
    nl = [0]*(N-1)
    nl.extend(list(l))
    new_df[str(N)+'_d_mavg_'+str(col_name)] = nl
    return new_df


def event1_check(index):
    '''
     Input is 0 to 1968, it is the index of Calender.CSV
     Output is True if there is event 1, False if no event 1
    '''
    return not(calender.event_name_1[index] != calender.event_name_1[index])

def event2_check(index):
    '''
    Input is 0 to 1968, it is the index of Calender.CSV
    Output is True if there is event 2, False if no event 2
    '''
    return not(calender.event_name_2[index] != calender.event_name_2[index])

def one_feature_engineering_fun(df):
    snap_dict = dict(zip(calender.d, calender['snap_'+df.state_id.iloc[0]]))   # Add snap or not column
    df['snap_or_not'] = df["d"].map(snap_dict)
    print('snap added')
    
    df['event_or_not'] = df["d"].map(event_dict)    # Adding event or not column
    print('events added')
    
    df['item_d_col'] = df['item_id'] + df['d']      # Adding sale_price column
    df['sale_price'] = df['item_d_col'].map(sale_price_dict)
    df['sale_price'] = df['sale_price'].fillna(0)
    print('sale prices added')
    
    df = df.drop('item_d_col', 1)                   # Undoing the columns we had to add
    
    df['Total_sale'] = df.unit_sale * df.sale_price  # Adding total sale column
    
    df = add_moving_avg_col(df,7, 'sale_price')     # Adding moving averages for sale price
    df = add_moving_avg_col(df,14, 'sale_price')
    df = add_moving_avg_col(df,30, 'sale_price')
    df = add_moving_avg_col(df,60, 'sale_price')
    df = add_moving_avg_col(df,180, 'sale_price')
                 
    df['day_of_month'] = df['day_of_month'].fillna(0)
    df = df.astype({'day_of_month': 'int32'})      # Making day_of_month column as int
    
    df['date'] = df['date'].astype(str)
    
    df = add_moving_avg_col(df,7, 'unit_sale')     # Adding moving average columns
    df = add_moving_avg_col(df,14, 'unit_sale')
    df = add_moving_avg_col(df,30, 'unit_sale')
    df = add_moving_avg_col(df,60, 'unit_sale')
    df = add_moving_avg_col(df,180, 'unit_sale')
    print('Total sale and Unit sale moving averages added')
    
    l1 = df.day_of_week == 1                       # we are adding an weekend or not column
    l2 = df.day_of_week == 2

    l1 = np.logical_or(l1,l2)
    l1 = [elem*1 for elem in l1]
    df['weekend'] = l1
    print('Weekends added')
    
    return df


def encode_cat_cols(new_df):
    le = [0]*len(non_numeric_col_list)             # Encoding Categorical Columns
    for i in range(len(non_numeric_col_list)):
        print("Encoding col: ", non_numeric_col_list[i])
        le[i] = LabelEncoder()
        new_df[non_numeric_col_list[i]] = le[i].fit_transform( new_df[non_numeric_col_list[i]] )
    return le, new_df

'''
This function takes sales_train_ev or sales_train_val after feature_engineer() function has been called on them
and returns X, y and labelencoder after adding few more columns and encoding categorical features.
'''
def get_X_and_y(df, store_name):               
    print('Store Name:', store_name)
    new_df = df[df.store_id == store_name]        # Selecting rows for the selected store
    
    print('Store rows picked now working on adding columns...')
    new_df = one_feature_engineering_fun(new_df)     # working on adding more columns and changing datatype of columns
    
    y = new_df.unit_sale                          # getting the label
    new_df = new_df.drop('unit_sale', axis=1)
    
    print('Encoding categorical features...')
    le, new_df = encode_cat_cols(new_df)          # Encoding Categorical Columns

    X = new_df
    
    return X, y, le

''' This function gets X and y but doesnt add any more features than already added in feature_engineer() '''
def get_X_and_y_withou_adding_more_features(df, store_name):               
    print('Store Name:', store_name)
    new_df = df[df.store_id == store_name]        # Selecting rows for the selected store
    
    new_df['day_of_month'] = new_df['day_of_month'].fillna(0)
    new_df = new_df.astype({'day_of_month': 'int32'})      # Making day_of_month column as int
    new_df['date'] = new_df['date'].astype(str)
    
    y = new_df.unit_sale                          # getting the label
    new_df = new_df.drop('unit_sale', axis=1)
    
    print('Encoding categorical features...')
    le, new_df = encode_cat_cols(new_df)          # Encoding Categorical Columns

    X = new_df
    
    return X, y, le


''' 
Functions in below this are used to change to data back to previuos form undoing long-form'
This function can be called to get prediction and mse for one single store
'''
def mse_and_out_df(df, store_name):  # df is either sales_train_ev or sales_train_val. Store name in one of 10 stores.
    le, X_test, y_test, train_out = fit_fun_2(df, store_name)
    train_rmse = sqrt(mse(y_test, train_out))
    out_df = reverse_long_form(le, X_test, train_out)
    
    return train_rmse, out_df


def training_function(df):
    main_out_df = pd.DataFrame()
    
    for i in list(set(df.store_id)):
        _, interm_df = mse_and_out_df(df, i)
        main_out_df = pd.concat([main_out_df, interm_df], ignore_index=False)
    
    l = []      # In this part we rename the columns to F_1, F_2 ....
    for i in range(1,29):
        l.append('F'+str(i))
    l = ['id']+l

    main_out_df.columns = l
    return main_out_df


# Function for reversing the long form
def reverse_long_form(le, X_test, train_out):
    for i in range(len(non_numeric_col_list)):
        X_test[non_numeric_col_list[i]] = le[i].inverse_transform(X_test[non_numeric_col_list[i]])

    X_test['unit_sale'] = train_out
    kk = X_test.pivot(index='id', columns='d')['unit_sale']
    kk['id'] = kk.index
    kk.reset_index(drop=True, inplace=True)

    cols = list(kk)
    cols = [cols[-1]] + cols[:-1]
    kk = kk[cols]

    return kk

def reorder_data(df, csv_name):
    df['sp_index'] = (df.index)
    index_dict = dict(zip(df.id, df.sp_index))
    df = df.drop('sp_index', axis=1)

    kk = pd.read_csv(str(csv_name)+'.csv')
    kk = kk.drop(kk.columns[0], axis=1)

    kk['sp_index'] = kk["id"].map(index_dict)
    kk = kk.sort_values(by='sp_index', axis=0)
    kk = kk.drop('sp_index', axis=1)
    kk.to_csv(str(csv_name)+'.csv')
    
'''
This function takes output of fit_fun_2(), the output named out_df and plots prediction vs true label for 
given item id. However, for sales_train_ev we need to change fit_fun_2() to have prediction on 1913 to 1969 
to see prediction and true label on 1913 to 1941 bcs there is no true label for 1942 to 1969.
'''

def plot_prediction(out_df, item_id, multiply_prediction_by):
    ll = list(out_df[out_df.id == item_id].iloc[0].iloc[1:])
    ll = [elem*multiply_prediction_by for elem in ll]
    plt.figure(figsize=(20,5))

    plt.subplot(1,2,1)
    plt.plot(ll)
    plt.plot(sales_train_ev[sales_train_ev.id == item_id].iloc[0].iloc[1919:])
    plt.legend(['predict', 'original'])
    plt.title(str(item_id)+'prediction, prediction * '+ str(multiply_prediction_by))
    plt.xlabel('Days from 1914 to 1941')
    plt.ylabel('No of units sold')

    plt.subplot(1,2,2)
    ll = list(out_df[out_df.id == item_id].iloc[0].iloc[29:])
    ll = [elem*multiply_prediction_by for elem in ll]
    plt.plot(ll)
    plt.title(str(item_id)+'prediction, prediction * '+str(multiply_prediction_by))
    plt.xlabel('Days from 1942 to 1969')
    plt.ylabel('No of units sold')