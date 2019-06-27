import sys
import pickle
import argparse
import functools

import pandas as pd
import numpy as np
import xxhash

from sklearn.svm import SVC
from tqdm import tqdm, trange

import src.credit_utils as cred


_cached_transformations = {}


class ExpContext:
    """Experimental context: datasets, target model.

    :param df: Full dataset as DataFrame
    :param df_train: Full training dataset as DataFrame
    :param df_test: Full test dataset as DataFrame
    :param df_X: Full dataset as DataFrame without labels
    :param df_y: Dataset labels as a Series
    :param X: Full dataset as numpy
    :param y: Dataset labels as numpy
    :param X_train: Train examples as numpy
    :param y_train: Train labels as numpy
    :param X_test: Test examples as numpy
    :param y_test: Test labels as numpy
    :param clf: Target classifier, a trained SVM instance
    :param model_params: SVM parameters
    """

    def __init__(self):
        """Prepare data, train the target model."""
        self.df, self.df_X, self.df_y = cred.load_dataframes(
                'data/german_credit_data.csv')
        datasets = cred.to_numpy_data(self.df_X, self.df_y)
        self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test = datasets

        self.df_train = pd.DataFrame(
            np.hstack([self.X_train, np.expand_dims(self.y_train, 1)]),
            columns=list(self.df_X.columns) + ["Risk_good"]
        )
        self.df_test = pd.DataFrame(
            np.hstack([self.X_test, np.expand_dims(self.y_test, 1)]),
            columns=list(self.df_X.columns) + ["Risk_good"]
        )

        self.clf, self.model_params = cred.train_model(
                self.X_train, self.y_train, self.X_test, self.y_test)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def example_similarity(exp_context, target_group_indices, static_cols, i):
    return -np.mean(np.linalg.norm(
        exp_context.df[static_cols].iloc[i] - \
                exp_context.df[static_cols].iloc[target_group_indices], ord=2, axis=1))


def score_group(exp_context, target_group_indices, custom_clf=None):
    """Get the average accuracy of the model on the target group"""
    if custom_clf is None:
        clf = exp_context.clf
    else:
        clf = custom_clf
    return np.sum(clf.predict(exp_context.X[target_group_indices]) == 1.)


grad_cache = {}
def logreg_grad(exp_context, x, y=1):
    """Gradient of a sklearn logistic regression."""
    x_hashed = fast_hash(np.concatenate([x, [y]]))
    if x_hashed in grad_cache:
        return grad_cache[x_hashed]

    sign = -1 if y == 1 else +1
    exp_y = np.exp(sign * exp_context.clf.decision_function([x]))
    res = sign * x / (exp_y + 1)

    grad_cache[x_hashed] = res
    return res


_hash_state = xxhash.xxh64()


def fast_hash(obj):
    """
    Fast hashing for numpy arrays.
    https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array#16592241
    """
    _hash_state.update(obj)
    result = _hash_state.intdigest()
    _hash_state.reset()
    return result

hessian_cache = {}
def hessian(exp_context, x, y=1):
    """Hessian matrix of a sklearn logistic regression."""
    x_hashed = fast_hash(np.concatenate([x, [y]]))
    if x_hashed in hessian_cache:
        return hessian_cache[x_hashed]

    n = len(x)
    sign = -1 if y == 1 else +1
    res = np.zeros((n, n))
    exp_y = np.exp(sign * exp_context.clf.decision_function([x]))
    exp_y_squared = exp_y ** 2
    normalizer = (exp_y + 1)
    normalizer_squared = normalizer ** 2
    for i in range(n):
        for j in range(n):
            prod = x[i] * x[j]
            res[i, j] = prod * exp_y / normalizer - \
                    prod * exp_y_squared / normalizer_squared

    hessian_cache[x_hashed] = res
    return res


avg_hess_inv_cache = {}
def influence(exp_context, target_group_indices, x):
    """Compute an influence function of a given example."""
    eps = 10e-10
    grad = logreg_grad(exp_context, x)
    avg_hess = sum([hessian(exp_context, x, y)
        for x, y in zip(exp_context.X_train, exp_context.y_train)])
    avg_hess /= len(exp_context.X_train)

    # Compute the standard influence function.
    avg_hess_hash = fast_hash(avg_hess)
    if avg_hess_hash in avg_hess_inv_cache:
        avg_hess_inv = avg_hess_inv_cache[avg_hess_hash]
    else:
        avg_hess_inv = avg_hess_inv_cache[avg_hess_hash] = np.linalg.pinv(avg_hess)

    influence = np.dot(avg_hess_inv, grad)

    # Task-specific influence.
    task_grad = -sum(logreg_grad(exp_context, t, 0)
            for t in exp_context.X[target_group_indices])
    res = np.dot(task_grad, influence)

    # debug_vals = sorted(zip(exp_context.df_X.columns, influence, task_grad,
    #     exp_context.clf.coef_[0], x), key=lambda t: t[1])
    # debug_data = pd.DataFrame(debug_vals, columns=["col", "inf", "task", "coef", "x"])
    # print(debug_data)

    return res

def find_poisoning_group(exp_context, target_group_indices,
                         score_diff_threshold=3,
                         seed=1, max_group_size=30):
    """Find a poisoning group."""

    clf = exp_context.clf
    X, y = exp_context.X, exp_context.y
    X_train, y_train = exp_context.X_train, exp_context.y_train

    allowed_indices = [i for i in range(len(X))
                       if i not in target_group_indices and y[i]]
    X_train_curr = np.array(X_train)
    y_train_curr = np.array(y_train)

    cum_group_datasets = []
    cum_group_scores = []

    score_baseline = score_group(exp_context, target_group_indices)

    # Weight examples by their average similarity to the target group.
    transformation_wrapper = cred.make_transformation_wrapper(exp_context.df_X.columns)
    static_cols = exp_context.df_X.columns[:transformation_wrapper.amount_start_idx]

    # sims = [example_similarity(exp_context, target_group_indices, static_cols, i)
    #         for i in allowed_indices]
    # weights = softmax(sims)
    # sims = [influence(exp_context, target_group_indices, X[i]) for i in allowed_indices]
    # weights = softmax(sims / np.sum(sims))

    # np.random.seed(seed)
    # sampled_indices = np.random.choice(
    #         allowed_indices, size=len(allowed_indices),
    #         replace=False, p=weights)
    sampled_indices = sorted(allowed_indices, key=lambda i: -example_similarity(
            exp_context, target_group_indices, static_cols, i))
    # sampled_indices = sorted(allowed_indices, key=lambda i: -influence(
    #         exp_context, target_group_indices, X[i]))
    progbar = tqdm(total=max_group_size)
    group_counter = 0

    for i in sampled_indices:
        x = X[i]
        if i in _cached_transformations:
            transformations = _cached_transformations[i]
        else:
            transformations = cred.generate_all_transformations(
                x, exp_context.df_X.columns,
                transformation_kwargs=dict(
                    decrease_amount=True,
                    decrease_duration=True))
            _cached_transformations[i] = list(transformations)

        # Pick the transformation with the highest influence.
        best_inf_val = 0
        best_datasets = None
        for t in transformations:
            # Check if this transformation would be accepted.
            if clf.predict([t]) != [1]:
                break

            inf_val = influence(exp_context, target_group_indices, t)
            if inf_val > best_inf_val:
                best_inf_val = inf_val
                X_train_adv = np.vstack([X_train_curr, t])
                y_train_adv = np.concatenate([y_train_curr, [1]])
                best_datasets = X_train_adv, y_train_adv

            # X_train_adv = np.vstack([X_train_curr, t])
            # y_train_adv = np.concatenate([y_train_curr, [1]])
            # new_clf, _ = cred.train_model(X_train_adv, y_train_adv,
            #         exp_context.X_test, exp_context.y_test, verbose=False)
            # inf_val = score_group(exp_context, target_group_indices, custom_clf=new_clf)
            # if inf_val > best_inf_val:
            #     best_inf_val = inf_val
            #     best_datasets = X_train_adv, y_train_adv

        # Retrain a classifier with the poisoned example.
        score = -1
        if best_datasets is not None:
            X_train_curr, y_train_curr = best_datasets
            new_clf, _ = cred.train_model(X_train_curr, y_train_curr,
                    exp_context.X_test, exp_context.y_test, verbose=False)
            score = score_group(exp_context, target_group_indices, custom_clf=new_clf)

        # If a score is improved, update data
        if score - score_baseline >= score_diff_threshold:
            cum_group_datasets.append(best_datasets)
            cum_group_scores.append(score)
            score_baseline = score
            group_counter += 1

            progbar.set_description(
                    'Accepted / Score: %i (%i)' % (score, score_baseline))
            progbar.update()

            # If group is assembled, stop.
            if group_counter >= max_group_size:
                break

        else:
            progbar.set_description(
                    'Rejected / Score: %i (%i)' % (score, score_baseline))

    print("Final score: %i" % score_baseline)
    print("Poisoned size: %i" % group_counter)
    print("Iterations: %i" % i)
    progbar.close()

    return cum_group_scores, cum_group_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a poisoning attack on loan approval.')
    parser.add_argument('--sim_seed', default=1, type=int,
                        help='Simulations seed')
    parser.add_argument('--num_simulations', default=1, type=int,
                        help='Number of simulations')
    parser.add_argument('--max_group_size', default=5, type=int,
                        help='Max poisoning group size')

    args = parser.parse_args()

    exp_context = ExpContext()

    # Pick a target group to benefit from poisoning.
    df = exp_context.df
    target_group_sel = (df['Checking account_little'] == 1) \
                     & (df['Saving accounts_little'] == 1) \
                     & (df['Risk_good'] == 1) \
                     & (exp_context.clf.predict(
                         exp_context.X) == 0).astype(bool)

    print('Target group size:', sum(target_group_sel))
    target_group = exp_context.X[target_group_sel]
    target_group_indices = np.where(target_group_sel)[0]
    print('Acceptance rate for the target group: %2.6f' % \
          score_group(exp_context, target_group_indices))

    for i in range(args.num_simulations):
        cum_group_scores, cum_group_datasets = find_poisoning_group(
                exp_context, target_group_indices,
                seed=args.sim_seed + i, max_group_size=args.max_group_size)
        # out_path = 'out/group_poisoning_influence_seed_%d_sim_%d' % (args.sim_seed, i)
        # with open(out_path, 'wb') as f:
        #     pickle.dump((cum_group_scores, cum_group_datasets), f)

