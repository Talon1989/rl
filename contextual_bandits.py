import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
import plotly


# def visualize_bandits(ug: UserGenerator):  # visualization with plotly
#     import plotly.offline
#     from plotly.subplots import make_subplots
#     import plotly.io as pio
#     pio.renderers.default = "browser"
#     import plotly.graph_objects as go
#     import cufflinks as cf
#     cf.go_offline()
#     cf.set_config_file(world_readable=True, theme='white')
#
#     def get_scatter(x, y, name, showlegend):
#         dashmap = {
#             'A': 'solid',
#             'B': 'dot',
#             'C': 'dash',
#             'D': 'dashdot',
#             'E': 'longdash'
#         }
#         s = go.Scatter(
#             x=x, y=y, legendgroup=name, showlegend=showlegend,
#             name=name, line=dict(color='blue', dash=dashmap[name])
#         )
#         return s
#
#     ages = np.linspace(10, 70)
#     fig = make_subplots(
#         rows=2, cols=2, subplot_titles=(
#             'Desktop, International', 'Desktop, US', 'Mobile, International', 'Mobile, US'
#         )
#     )
#     for device in [0, 1]:
#         for loc in [0, 1]:
#             showlegend = device == 0 and loc == 0
#             for a in possible_actions:
#                 probas = [ug.logistic(ug.beta[a], [1, device, loc, age]) for age in ages]
#                 fig.add_trace(
#                     get_scatter(ages, probas, a, showlegend),
#                     row=device + 1, col=loc + 1
#                 )
#     fig.update_layout(template='presentation')
#     fig.show()


np.random.seed(0)
n_features = 3


class UserGenerator:

    def __init__(self):

        self.beta = {  # weights of features for each actions (A, B, C, D, E)
            'A': np.array([-4, -0.1, -3, 0.1]),
            'B': np.array([-6, -0.1, 1, 0.1]),
            'C': np.array([2, 0.1, 1, -0.1]),
            'D': np.array([4, 0.1, -3, -0.2]),
            'E': np.array([-0.1, 0, 0.5, -0.01]),
        }
        # self.beta = {}
        # for a in actions:
        #     self.beta[a] = np.random.normal(0, 0.1, size=n_features+1)
        self.context = None

    @staticmethod
    def logistic(beta, context):
        f = np.dot(beta, context)
        p = 1 / (1 + np.exp(-np.clip(f, a_min=-250, a_max=250)))
        return p

    def display_action(self, a, actions):
        if a in actions:
            p = self.logistic(self.beta[a], self.context)
            reward = np.random.binomial(n=1, p=p)
            return reward
        else:
            raise Exception('Unknown action')

    def generate_user_with_context(self):
        location = np.random.binomial(n=1, p=0.6)  # 0: International, 1: U.S.
        device = np.random.binomial(n=1, p=0.8)  # 0: Desktop, 1: Mobile
        age = 10 + int(np.random.beta(2, 3) * 60)  # User age changes between 10 and 70
        self.context = [1, device, location, age]
        return self.context


def visualize_bandits(ug: UserGenerator):  # visualization with pyplot
    ages = np.linspace(10, 70)
    fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(np.cos(test))
    # axs[0, 1].plot(np.sin(test))
    # axs[1, 0].plot(np.tan(test))
    # axs[1, 1].plot(test ** 2)

    styles = ['solid', 'dotted', 'dashed', 'dashdot', '-']
    devices, locations = ['Desktop', 'Mobile'], ['International', 'US']

    for device in [0, 1]:
        for loc in [0, 1]:
            for a in possible_actions:
                probas = [ug.logistic(ug.beta[a], [1, device, loc, age]) for age in ages]
                axs[device, loc].plot(
                    ages,
                    probas,
                    linestyle=styles[possible_actions.index(a)],
                    label=a
                )
            axs[device, loc].set_title('%s , %s' % (devices[device], locations[loc]))
            axs[device, loc].legend(loc='best', prop={'size': 6})

    plt.show()
    plt.clf()


possible_actions = ['A', 'B', 'C', 'D', 'E']
ug_ = UserGenerator()
# visualize_bandits(ug_)


class RegularizedLR:
    
    def __init__(self, name, alpha, rlambda, n_dim):
        self.name = name
        self.alpha = alpha  # exploration / exploitation trade-off
        self.rlambda = rlambda  # regularization term
        self.n_dim = n_dim  # dimension of beta parameter vector
        self.m = np.zeros(n_dim)
        self.q = np.ones(n_dim) * self.rlambda
        self.w = self.get_sampled_weights()

    def get_sampled_weights(self):
        return np.random.normal(loc=self.m, scale=self.alpha * self.q ** (-1/2))

    def fit(self, X, y):  # y = 1 = click ; y = 0 = no click
        def loss(w, *args):
            X, y = args
            regularizer = 0.5 * np.dot(self.q, (w - self.m) ** 2)
            pred_loss = sum(
                [np.log(1 + np.exp(np.dot(w, X[j]))) - y[j] * np.dot(w, X[j]) for j in range(len(y))]
            )
            return regularizer + pred_loss
        if y:
            X, y = np.array(X), np.array(y)
            minimization = optimize.minimize(
                loss, self.w, args=(X, y), method='L-BFGS-B',
                bounds=[(-10, 10)] * 3 + [(-1, 1)], options={'maxiter': 50}
            )
            # self.w = minimization.x
            self.m = self.w = minimization.x
            # p = (1 + np.exp(-np.matmul(self.w, X.T))) ** (-1)
            p = 1 / (1 + np.exp(-np.matmul(self.w, X.T)))
            self.q = self.q + np.matmul(p * (1-p), X**2)

    @staticmethod
    def calc_sigmoid(w, context):
        return 1 / (1 + np.exp(-np.dot(w, context)))

    def get_ucb(self, context):
        pred = self.calc_sigmoid(self.m, context)
        confidence = self.alpha * np.sqrt(
            np.sum(np.divide(np.array(context) ** 2, self.q))
        )
        ucb = pred + confidence
        return ucb

    def get_prediction(self, context):
        return self.calc_sigmoid(self.m, context)

    def sample_prediction(self, context):
        w = self.get_sampled_weights()
        return self.calc_sigmoid(w, context)


def calculate_regret(ug: UserGenerator, context, ad_options, ad):
    action_values = {
        a: ug.logistic(ug.beta[a], context) for a in ad_options
    }
    best_action = max(action_values, key=action_values.get)
    regret = action_values[best_action] - action_values[ad]
    return regret, best_action


#  EXPLORATION STRATEGIES


def select_ad_eps_greedy(ad_models, context, eps):
    if np.random.uniform() < eps:
        return np.random.choice(list(ad_models.keys()))
    else:
        predictions = {
            ad: ad_models[ad].get_prediction(context) for ad in ad_models
        }
        max_value = max(predictions.values())
        return np.random.choice(
            [key for key, value in predictions.items() if value == max_value]
        )


def select_ad_ucb(ad_models, context):
    ucbs = {
        ad: ad_models[ad].get_ucb(context) for ad in ad_models
    }
    max_value = max(ucbs.values())
    return np.random.choice(
        [key for key, value in ucbs.items() if value == max_value]
    )


def select_ad_thompson(ad_models, context):
    samples = {
        ad: ad_models[ad].sample_prediction(context) for ad in ad_models
    }
    max_value = max(samples.values())
    return np.random.choice(
        [key for key, value in samples.items() if value == max_value]
    )


#  RUNNING


def apply_contextual_bandits():

    ad_options = ['A', 'B', 'C', 'D', 'E']
    exploration_strategies = ['eps-greedy', 'ucb', 'Thompson']
    style = ['solid', 'dotted', 'dashed']

    for strategy in exploration_strategies:

        print('Using %s' % strategy)
        np.random.seed(0)

        #  Create the LR models for each ad
        alpha, rlambda, n_dim = 0.5, 0.5, n_features+1
        ad_models = {
            ad: RegularizedLR(ad, alpha, rlambda, n_dim) for ad in ad_options
        }

        #  Initialize data structures
        X = {ad: [] for ad in ad_options}
        y = {ad: [] for ad in ad_options}
        results = []
        total_regret = 0

        for i in range(10_000):
            context = ug_.generate_user_with_context()
            ad = None
            if strategy == 'eps-greedy':
                ad = select_ad_eps_greedy(ad_models, context, eps=0.1)
            elif strategy == 'ucb':
                ad = select_ad_ucb(ad_models, context)
            elif strategy == 'Thompson':
                ad = select_ad_thompson(ad_models, context)
            click = ug_.display_action(ad, ad_options)
            X[ad].append(context)
            y[ad].append(click)
            regret, best_action = calculate_regret(ug_, context, ad_options, ad)
            total_regret += regret
            results.append(
                [context, ad, click, best_action, regret, total_regret]
            )
            #  Update the models with the latest batch of data
            if (i + 1) % 500 == 0:
                print('Updating the model at iteration %d' % (i+1))
                for ad in ad_options:
                    ad_models[ad].fit(X[ad], y[ad])
                X = {ad: [] for ad in ad_options}
                y = {ad: [] for ad in ad_options}

        # plot
        plt.plot(
            np.array(results)[:, -1],
            linestyle=style[exploration_strategies.index(strategy)],
            label=strategy
        )

    plt.xlabel('Impressions')
    plt.ylabel('Total Regret')
    plt.legend(loc='best')
    plt.show()
    plt.clf()































































































































































































































































































































































































