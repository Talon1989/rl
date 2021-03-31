import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
# from tensorflow import keras
import keras

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
df_data_ = pd.concat(
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


def get_model(n_input, dropout=0.1):

    inputs = keras.Input(shape=(n_input,))

    X_1 = keras.layers.Dense(256, activation='relu')(inputs)
    if dropout > 0:
        X_1 = keras.layers.Dropout(dropout)(X_1, training=True)

    X_2 = keras.layers.Dense(256, activation='relu')(X_1)
    if dropout > 0:
        X_2 = keras.layers.Dropout(dropout)(X_2, training=True)

    phat = keras.layers.Dense(1, activation='sigmoid')(X_2)

    model = keras.Model(inputs, phat)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

    # from keras.layers import Dense
    # from keras.layers import Dropout
    # inputs = keras.Input(shape=(n_input,))
    # x = Dense(256, activation='relu')(inputs)
    # if dropout > 0:
    #     x = Dropout(dropout)(x, training=True)
    # x = Dense(256, activation='relu')(x)
    # if dropout > 0:
    #     x = Dropout(dropout)(x, training=True)
    # phat = Dense(1, activation='sigmoid')(x)
    # model = keras.Model(inputs, phat)
    # model.compile(loss=keras.losses.BinaryCrossentropy(),
    #               optimizer=keras.optimizers.Adam(),
    #               metrics=[keras.metrics.binary_accuracy])
    # return model


def update_model(model: keras.Model, X, y):
    X = np.array(X)
    X = X.reshape([X.shape[0], X.shape[2]])
    y = np.array(y)
    y = y.reshape(-1)
    model.fit(X, y, epochs=10)
    return model


def ad_onehot(ad):
    ed_levels = ['Elementary', 'Middle', 'HS-grad', 'Undergraduate', 'Graduate']
    ad_input = np.zeros(len(ed_levels))
    if ad in ed_levels:
        ad_input[ed_levels.index(ad)] = 1
    return list(map(int, ad_input))  # cast np.array to vanilla list of integers
    # ad_input = [0] * len(ed_levels)
    # if ad in ed_levels:
    #     ad_input[ed_levels.index(ad)] = 1
    # return ad_input


def select_ad(model: keras.Model, context, ad_inventory):  # Thompson sampling
    selected_ad, selected_x, max_action_val = None, None, 0
    for ad in ad_inventory:
        ad_x = ad_onehot(ad)
        x = np.array(context + ad_x).reshape([1, -1])
        action_val_pred = model.predict(x)[0][0]
        if action_val_pred >= max_action_val:
            selected_ad, selected_x, max_action_val = ad, x, action_val_pred
    return selected_ad, selected_x


def generate_user(df_data: pd.DataFrame):
    user = df_data.sample(1)
    context = user.iloc[:, :-1].values.tolist()[0]
    return user.to_dict(orient='records')[0], context


def calculate_regret(user, ad_inventory, ad_click_probas, ad_selected):
    this_p, max_p = 0, 0
    for ad in ad_inventory:
        p = ad_click_probas[ad][user['education']]
        if ad == ad_selected:
            this_p = p
        if p > max_p:
            max_p = p
    regret = max_p - this_p
    return regret


ad_click_probas = get_ad_click_probas()
df_cbandits = pd.DataFrame()
dropout_levels = [0, 0.01, 0.05, 0.1, 0.2, 0.4]
n_iter = 5000
dropout_regret = []
for d in dropout_levels:
    print('Dropout: %.2f' % d)
    np.random.seed(0)
    n_context = df_data_.shape[1] - 1  # n features
    n_ad_input = df_data_.education.nunique()  # n unique education values
    model_ = get_model(n_input=n_context+n_ad_input, dropout=d)
    X_, y_ = [], []
    regret_vec, total_regret = [], 0
    for i in range(n_iter):
        if i % 20 == 0:
            print("# of impressions: %d" % i)
        user_, context_ = generate_user(df_data_)
        ad_inventory_ = get_ad_inventory()
        ad_, x_ = select_ad(model_, context_, ad_inventory_)
        click = display_ad(ad_click_probas, user_, ad_)
        regret_ = calculate_regret(user_, ad_inventory_, ad_click_probas, ad_)
        total_regret += regret_
        regret_vec.append(total_regret)
        X_.append(x_)
        y_.append(click)
        if (i + 1) % 500 == 0:
            print('Updating the model at iteration: %d' % (i + 1))
            model_ = update_model(model_, X_, y_)
            X_, y_ = [], []
    # dropout_regret[dropout_levels.index(d)].append()
    dropout_regret.append(regret_vec)
    df_cbandits['dropout: '+str(d)] = regret_vec


cols = ['b', 'g', 'c', 'm', 'k', 'y']
for i in range(len(dropout_levels)):
    plt.plot(
        dropout_regret[i],
        c=cols[i],
        label='dropout: %.2f' % dropout_levels[i],
        linewidth=1
    )
plt.grid(alpha=0.5)
plt.legend(loc='best')
plt.xlabel('# iterations')
plt.ylabel('cumulative regret')
plt.show()
plt.clf()







































































































































































































































































































