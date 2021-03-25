import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from tensorflow import keras
from keras.layers import Dense, Dropout

# adults = pd.read_csv('data/adult.data', header=None)

s = requests.get(
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
).content
names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]
usecols = [
    'age', 'workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'hours_per_week', 'native_country', 'income'
]
df_census = pd.read_csv(
    io.StringIO(s.decode('utf-8')),
    sep=',', skipinitialspace=True, names=names, header=None, usecols=usecols
)

#  drop data with missing values
df_census = df_census.replace('?', np.nan).dropna()

#  generalize education column
edu_mapping = {
    'Preschool': 'Elementary',
    '1st-4th': 'Elementary',
    '5th-6th': 'Elementary',
    '7th-8th': 'Elementary',
    '9th': 'Middle',
    '10th': 'Middle',
    '11th': 'Middle',
    '12th': 'Middle',
    'Some-college': 'Undergraduate',
    'Bachelors': 'Undergraduate',
    'Assoc-acdm': 'Undergraduate',
    'Assoc-voc': 'Undergraduate',
    'Prof-school': 'Graduate',
    'Masters': 'Graduate',
    'Doctorate': 'Graduate'
}
for from_level, to_level in edu_mapping.items():
    df_census.education.replace(from_level, to_level, inplace=True)

#  convert categorical data to one-hot vectors
context_cols = [c for c in usecols if c != 'education']
df_data = pd.concat(
    [pd.get_dummies(df_census[context_cols]), df_census['education']], axis=1
)


#  simulating ad clicks


def get_ad_inventory():
    ad_inventory_probas = {
        'Elementary': 0.9, 'Middle': 0.7, 'HS-grad': 0.7, 'Undergraduate': 0.9, 'Graduate': 0.8
    }
    ad_inventory = []
    for level, proba in ad_inventory_probas.items():
        if np.random.uniform(0, 1) < proba:  # click
            ad_inventory.append(level)
    if not ad_inventory:
        ad_inventory = get_ad_inventory()
    return ad_inventory


def get_ad_click_probas():
    base_proba = 0.8
    delta = 0.3
    ed_levels = {
        'Elementary': 1, 'Middle': 2, 'HS-grad': 3, 'Undergraduate': 4, 'Graduate': 5
    }
    ad_click_probas = {
        l1: {
            l2: max(
                0, base_proba - delta * np.abs(ed_levels[l1] - ed_levels[l2])
            ) for l2 in ed_levels
        } for l1 in ed_levels
    }
    return ad_click_probas


def display_ad(ad_click_probas, user, ad):
    proba = ad_click_probas[ad][user['education']]
    click = 1 if np.random.uniform(0, 1) < proba else 0
    return click

#  when an ad is shown to a user, if the ad's target matches the user's education
#  level, there will be 0.8 chance of a click. This probability decreases by 0.3 for
#  each level of mismatch


#  FUNCTION APPROXIMATION USING A NN









































































































































































































































































































