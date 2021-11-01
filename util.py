""" Functions for event prediction """

import os
import shutil
import pandas as pd
from sklearn import preprocessing


def prepare_workspace(output_dir, replace=False):
    """ Make output directory """

    if replace == True:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


def encode_timestamp(df, datetime_col='date', freq='D'):
    """ Insert "date_id" column to a given DataFrame """

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    if freq == 'D':
        date_index = (df[datetime_col] - df[datetime_col].min()).dt.days.values

    df['date_id'] = date_index
    df = df.sort_values('date_id')
    df = df.reset_index()
    del df['index']
    return df


def encode_attribute(df, col, prefix):

    le = preprocessing.LabelEncoder()
    le.fit(df[col])
    df[prefix + '_id'] = le.transform(df[col])
    return df


def sample_events(df, n):

    if df.shape[0] <= n:
        return df
    else:
        return df.sample(n).reset_index(drop=True)
