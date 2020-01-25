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

from src.influence import influence_func


_cached_transformations = {}


class ExpContext:
    """Experimental context: datasets, target model.

    Args:
        df: Full dataset as DataFrame
        df_train: Full training dataset as DataFrame
        df_test: Full test dataset as DataFrame
        df_X: Full dataset as DataFrame without labels
        df_y: Dataset labels as a Series
        raw_datasets: Datasets as numpy, see :py:`credit_utils.Datasets`
        clf: Target classifier
        model_params: Model parameters
    """

    def __init__(self, seed=1):
        """Prepare data, train the target model."""
        self.df, self.df_X, self.df_y = cred.load_dataframes(
            "data/german_credit_data.csv"
        )
        self.raw_datasets = cred.to_numpy_data(self.df_X, self.df_y, seed=seed)

        self.df_train = pd.DataFrame(
            np.hstack(
                [
                    self.raw_datasets.X_train,
                    np.expand_dims(self.raw_datasets.y_train, 1),
                ]
            ),
            columns=list(self.df_X.columns) + ["Risk_good"],
        )

        self.df_test = pd.DataFrame(
            np.hstack(
                [self.raw_datasets.X_test, np.expand_dims(self.raw_datasets.y_test, 1)]
            ),
            columns=list(self.df_X.columns) + ["Risk_good"],
        )

        self.clf, self.model_params = cred.train_model(
            self.raw_datasets.X_train,
            self.raw_datasets.y_train,
            self.raw_datasets.X_test,
            self.raw_datasets.y_test,
        )


def select_candidates(
    exp_context,
    target_group_idxs,
    num_best_samples=10,
    seed=1,
    use_influence_func=False,
):
    clf = exp_context.clf
    X, y = exp_context.raw_datasets.X, exp_context.raw_datasets.y
    X_train = exp_context.raw_datasets.X_train
    y_train = exp_context.raw_datasets.y_train

    candidate_idxs = [
        i for i in range(len(X)) if i not in target_group_idxs and y[i]
    ]
    X_train_curr = np.array(X_train)
    y_train_curr = np.array(y_train)

    # Weight examples by their average similarity to the target group.
    transformation_wrapper = cred.make_transformation_wrapper(exp_context.df_X.columns)
    static_cols = exp_context.df_X.columns[: transformation_wrapper.amount_start_idx]

    # sims = [example_similarity(exp_context, target_group_idxs, static_cols, i)
    #         for i in candidate_idxs]
    # weights = softmax(sims)
    # sims = [influence(exp_context, target_group_idxs, X[i]) for i in candidate_idxs]
    # weights = softmax(sims / np.sum(sims))

    # np.random.seed(seed)
    # sampled_idxs = np.random.choice(
    #         candidate_idxs, size=len(candidate_idxs),
    #         replace=False, p=weights)
    sampled_idxs = sorted(
        candidate_idxs,
        key=lambda i: -cred.example_similarity(
            exp_context, target_group_idxs, static_cols, i
        ),
    )
    # sampled_idxs = sorted(candidate_idxs, key=lambda i: -influence(
    #         exp_context, target_group_idxs, X[i]))

    if num_best_samples is not None:
        sampled_idxs = sampled_idxs[:num_best_samples]

    scored_examples = []
    influence_data = pd.DataFrame(columns=["influence", "score", "acc"])

    index_progbar = tqdm(sampled_idxs)
    for i in index_progbar:
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
        best_example = None

        for t in transformations:
            # Check if this transformation would be accepted.
            if clf.predict([t]) != [1]:
                break

            if use_influence_func:
                inf_val = influence_func(exp_context, target_group_idxs, t)
                if inf_val > best_inf_val:
                    best_inf_val = inf_val
                    best_example = t
            else:
                X_train_adv = np.vstack([X_train_curr, t])
                y_train_adv = np.concatenate([y_train_curr, [1]])
                new_clf, _ = cred.train_model(X_train_adv, y_train_adv,
                        exp_context.raw_datasets.X_test,
                        exp_context.raw_datasets.y_test, verbose=False)
                inf_val = cred.score_group(exp_context, target_group_idxs, custom_clf=new_clf)
                if inf_val > best_inf_val:
                    best_inf_val = inf_val
                    best_example = t

        # Retrain a classifier with the poisoned example-candidate.
        score = -1
        if best_example is not None:
            X_train_adv = np.vstack([X_train_curr, best_example])
            y_train_adv = np.concatenate([y_train_curr, [1]])
            new_clf, _ = cred.train_model(X_train_adv, y_train_adv,
                    exp_context.raw_datasets.X_test, exp_context.raw_datasets.y_test, verbose=False)
            score = cred.score_group(exp_context, target_group_idxs, custom_clf=new_clf)
            scored_examples.append((score, best_example))

            acc = new_clf.score(
                    exp_context.raw_datasets.X[candidate_idxs],
                    exp_context.raw_datasets.y[candidate_idxs]
            )
            influence_data = influence_data.append(
                dict(influence=best_inf_val, score=score, acc=acc),
                ignore_index=True,
            )
            index_progbar.set_description("Score: %2.4f, Infl: %3.2f" % (
                score, best_inf_val))

    influence_data.to_csv("influence_data_retrained.csv")
    with open("scored_examples.pkl", "wb") as f:
        pickle.dump(scored_examples, f)

    return scored_examples


def find_poisoning_group(
    exp_context,
    target_group_idxs,
    num_best_samples=10,
    score_goal=None,
    noise_set_size=50,
    seed=1,
    max_group_size=10,
    load_candidates_from_cache=True,
):
    """Find a poisoning group G_pot."""

    if score_goal is None:
        score_goal = np.inf

    clf = exp_context.clf
    X, y = exp_context.raw_datasets.X, exp_context.raw_datasets.y
    X_train = exp_context.raw_datasets.X_train
    y_train = exp_context.raw_datasets.y_train
    X_train_curr = np.array(X_train)
    y_train_curr = np.array(y_train)

    # Add random people to the training data.
    np.random.seed(seed)
    candidate_idxs = [
        i for i in np.arange(len(X)) if i not in target_group_idxs and y[i]
    ]
    noise_idxs = np.random.choice(
        [i for i in exp_context.raw_datasets.test_ind if i not in candidate_idxs],
        noise_set_size,
        replace=False,
    )

    X_noise = exp_context.raw_datasets.X[noise_idxs]
    y_noise = exp_context.raw_datasets.y[noise_idxs]
    X_train_noisy = np.vstack([X_train_curr, X_noise])
    y_train_noisy = np.concatenate([y_train_curr, y_noise])
    X_train_noisy_adv = X_train_noisy
    y_train_noisy_adv = y_train_noisy

    print("Size of possible noise additions: %i" % len([
        i for i in exp_context.raw_datasets.test_ind if i in candidate_idxs]))
    print("Size of the noise additions: %i" % len(noise_idxs))

    if load_candidates_from_cache:
        with open("scored_examples.pkl", "rb") as f:
            scored_examples = pickle.load(f)
    else:
        scored_examples = select_candidates(
            exp_context,
            target_group_idxs,
            num_best_samples=num_best_samples,
            seed=seed,
        )

    score = 0
    group_counter = 0
    score_baseline = 0

    # Compute initial score and datasets.
    group_datasets = [(X_train_noisy, y_train_noisy)]
    sorted_examples = sorted(scored_examples, key=lambda t: t[0])
    with tqdm(total=max_group_size) as progbar:
        while sorted_examples and score < score_goal and group_counter < max_group_size:
            score, best_example = sorted_examples.pop()

            # Add the example to the clean poisoned dataset.
            X_train_adv = np.vstack([X_train_curr, best_example])
            y_train_adv = np.concatenate([y_train_curr, [1]])
            X_train_curr, y_train_curr = X_train_adv, y_train_adv

            # Add the example to the noisy poisoned dataset.
            X_train_noisy_adv = np.vstack([X_train_noisy_adv, best_example])
            y_train_noisy_adv = np.concatenate([y_train_noisy_adv, [1]])

            new_clf, _ = cred.train_model(
                X_train_adv,
                y_train_adv,
                exp_context.raw_datasets.X_test,
                exp_context.raw_datasets.y_test,
                verbose=False,
            )
            score = cred.score_group(exp_context, target_group_idxs, custom_clf=new_clf)

            new_clf, _ = cred.train_model(
                X_train_noisy_adv,
                y_train_noisy_adv,
                exp_context.raw_datasets.X_test,
                exp_context.raw_datasets.y_test,
                verbose=False,
            )
            noisy_score = cred.score_group(
                exp_context, target_group_idxs, custom_clf=new_clf
            )

            if score > score_baseline:
                score_baseline = score
                group_datasets.append((X_train_noisy_adv, y_train_noisy_adv))
                group_counter += 1

                progbar.set_description(
                    "Accepted. Score: %2.4f. Group size: %i" % (score, group_counter)
                )
            else:
                progbar.set_description(
                    "Rejected. Score: %2.4f (%2.4f). Group size: %i"
                    % (score, score_baseline, group_counter)
                )
            progbar.update()

    print("Final score: %2.4f" % noisy_score)
    print("Poisoned size: %i" % group_counter)

    return group_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a poisoning attack on credit scoring."
    )
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument(
        "--num_best_samples",
        default=None,
        type=int,
        help="Number of samples to consider",
    )
    parser.add_argument(
        "--noise_set_size", default=0, type=int, help="Number of noise samples to add"
    )
    parser.add_argument(
        "--num_simulations", default=1, type=int, help="Number of simulations"
    )
    parser.add_argument(
        "--max_group_size", default=10, type=int, help="Max poisoning group size"
    )
    parser.add_argument(
        "--load_scores_from_cache", default=False, type=bool, help="Load candidate scores from cache"
    )

    args = parser.parse_args()

    exp_context = ExpContext(seed=args.seed)

    # Pick a target group to benefit from poisoning.
    df = exp_context.df
    target_group_sel = (
        (df["Checking account_little"] == 1)
        & (df["Saving accounts_little"] == 1)
        & (df["Risk_good"] == 1)
        & (exp_context.clf.predict(exp_context.raw_datasets.X) == 0).astype(bool)
    )

    print("Target group size:", sum(target_group_sel))
    target_group = exp_context.raw_datasets.X[target_group_sel]
    target_group_idxs = np.where(target_group_sel)[0]
    print(
        "Acceptance rate for the target group: %2.6f"
        % cred.score_group(exp_context, target_group_idxs, trade_off=0.)
    )

    for i in range(args.num_simulations):
        group_datasets = find_poisoning_group(
            exp_context,
            target_group_idxs,
            seed=args.seed + i,
            max_group_size=args.max_group_size,
            noise_set_size=args.noise_set_size,
            num_best_samples=args.num_best_samples,
            load_candidates_from_cache=args.load_scores_from_cache,
        )
        out_path = "out/group_poisoning_influence_seed_%d_noise_%d_sim_%d" % (args.seed, args.noise_set_size, i)
        with open(out_path, "wb") as f:
            pickle.dump(group_datasets, f)
