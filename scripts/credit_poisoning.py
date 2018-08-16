import sys
import pickle
import argparse

import numpy as np

from sklearn.svm import SVC
from tqdm import tqdm, trange

import src.credit_utils as cred


_cached_transformations = {}


class ExpContext:
    """Experimental context: datasets, target model.

    :param X: Full dataset
    :param y: Full dataset (labels)
    :param X_train: Train examples
    :param y_train: Train labels
    :param X_test: Test examples
    :param y_test: Test labels
    :param clf: Target classifier --- a trained SVM instance
    :param svm_params: SVM parameters
    """

    def __init__(self):
        """Prepare data, train the target model."""
        self.df, self.df_X, self.df_y = cred.load_dataframes(
                'data/german_credit_data.csv')
        datasets = cred.to_numpy_data(self.df_X, self.df_y)
        self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test = datasets
        self.clf, self.svm_params = cred.train_model(
                self.X_train, self.y_train, self.X_test, self.y_test)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def score_group(exp_context, target_group_indices, custom_clf=None):
    if custom_clf is None:
        clf = exp_context.clf
    else:
        clf = custom_clf
    return np.mean(clf.predict(
        exp_context.X[target_group_indices])) * 100


def find_poisoning_group(exp_context, target_group_indices,
                         seed=1, max_group_size=30):
    """Find a poisoning group."""

    clf = exp_context.clf
    svm_params = exp_context.svm_params
    X, y = exp_context.X, exp_context.y
    X_train, y_train = exp_context.X_train, exp_context.y_train

    allowed_indices = [i for i in range(len(X))
                       if i not in target_group_indices and y[i]]
    X_train_curr = np.array(X_train)
    y_train_curr = np.array(y_train)

    cum_group_datasets = []
    cum_group_scores = []

    score_baseline = score_group(exp_context, target_group_indices)
    allowed_indices = list(range(len(X)))

    # Sort examples by their average similarity to the target group.
    sim_fn = lambda i: -np.mean(
            np.linalg.norm(X[i] - X[target_group_indices], ord=1, axis=1))
    sims = np.array(list(map(sim_fn, allowed_indices)))
    weights = softmax(sims)

    np.random.seed(seed)
    sampled_indices = np.random.choice(allowed_indices, size=len(X),
                                       replace=False, p=weights)
    progbar = tqdm(total=max_group_size)
    group_counter = 0

    for i in sampled_indices:
        x = X[i]
        if i in _cached_transformations:
            transformations = _cached_transformations[i]
        else:
            transformations = cred.generate_all_transformations(
                x, exp_context.df_X,
                transformation_kwargs=dict(
                    decrease_amount=True,
                    decrease_duration=True))
            _cached_transformations[i] = transformations

        best_score = 0
        for t in tqdm(transformations):
            # Check if this transformation would be accepted.
            if clf.predict([t]) == [1]:
                X_train_adv = np.vstack([X_train_curr, t])
                y_train_adv = np.concatenate([y_train_curr, [1]])
                clf_adv = SVC(probability=True, **svm_params).fit(X_train_adv, y_train_adv)
                score = score_group(exp_context, target_group_indices, clf_adv)
                if score > best_score:
                    best_score = score
                    best_datasets = X_train_adv, y_train_adv

        # If a score is improved, update data
        if best_score > score_baseline:
            X_train_curr, y_train_curr = best_datasets
            cum_group_datasets.append(best_datasets)
            cum_group_scores.append(best_score)
            score_baseline = best_score
            group_counter += 1
            progbar.set_description('Accepted / Score: %2.6f' % best_score)
            progbar.update()
        else:
            progbar.set_description('Rejected / Score: %2.6f' % score_baseline)

        # If group is assembled, stop.
        if group_counter >= max_group_size:
            break

    return cum_group_scores, cum_group_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sim_seed', default=1, type=int,
                        help='Simulations seed')
    parser.add_argument('--num_simulations', default=10, type=int,
                        help='Number of simulations')
    parser.add_argument('--max_group_size', default=20, type=int,
                        help='Max poisoning group size')

    args = parser.parse_args()

    exp_context = ExpContext()

    # Pick a target group to benefit from poisoning.
    df = exp_context.df
    target_group_sel = (df['Checking account_little'] == 1) \
                     & (df['Saving accounts_little'] == 1) \
                     & (df['Risk_good'] == 0)

    print('Target group size:', sum(target_group_sel))
    target_group = exp_context.X[target_group_sel]
    target_group_indices = np.where(target_group_sel)[0]

    print('Acceptance rate for the target group: %2.6f' % \
          score_group(exp_context, target_group_indices))

    for i in trange(args.num_simulations):
        cum_group_scores, cum_group_datasets = find_poisoning_group(
                exp_context, target_group_indices,
                seed=args.sim_seed + i, max_group_size=args.max_group_size)
        out_path = 'out/group_poisoning_seed_%d_sim_%d' % (args.sim_seed, i)
        with open(out_path, 'wb') as f:
            pickle.dump((cum_group_scores, cum_group_datasets), f)
