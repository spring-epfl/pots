import attr
import numpy as np
import pandas as pd
import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC


SEED = 1


@attr.s
class Datasets:
    X = attr.ib()
    y = attr.ib()
    X_train = attr.ib()
    y_train = attr.ib()
    X_test = attr.ib()
    y_test = attr.ib()
    train_ind = attr.ib()
    test_ind = attr.ib()


def load_dataframes(data_path):
    # Load the file
    df = pd.read_csv(data_path)
    df = df.drop(df.columns[0], axis=1) # remove the index column

    # Quantize credit amount, duration and age into 5 bins
    amount_series = df.loc[:, 'Credit amount']
    df.loc[:, 'Credit amount'] = pd.qcut(amount_series, 5)

    duration_series = df.loc[:, 'Duration']
    df.loc[:, 'Duration'] = pd.qcut(duration_series, 5)

    duration_series = df.loc[:, 'Age']
    df.loc[:, 'Age'] = pd.qcut(duration_series, 5)

    # Set Job type to object for one-hot encoding
    df.loc[:, 'Job'] = df.loc[:, 'Job'].astype(object)

    # Perform one-hot encoding
    df = pd.get_dummies(df)
    # Drop binary features
    df = df.drop(columns=['Sex_male', 'Risk_bad'])

    # Separate features from targets
    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    return df, df_X, df_y


def to_numpy_data(df_X, df_y, seed=SEED):
    """Convert dataframes to numpy."""

    # Convert to numpy
    X = df_X.values.astype('int8')
    y = df_y.values.astype('int8')
    print('Shape of X: {}. Shape of y: {}.'.format(X.shape, y.shape))

    # Split into training and test sets
    train_ind, test_ind = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=seed)
    return Datasets(
            X=X, y=y,
            X_train=X[train_ind],
            y_train=y[train_ind],
            X_test=X[test_ind],
            y_test=y[test_ind],
            train_ind=train_ind,
            test_ind=test_ind
    )


def train_model(X_train, y_train, X_test, y_test, verbose=True):
    """Train a model."""
    clf = LogisticRegression(C=10000000000000, solver="lbfgs")
    clf.fit(X_train, y_train)

    if verbose:
        print('Baseline accuracy is:', y_train.mean())
        print('Test score is: {:.2f}%.'.format(
                clf.score(X_test, y_test)*100))

    model_params = 0
    return clf, model_params


def make_transformation_wrapper(features):

    class TransformationWrapper:
        """Generate different loan application feature vectors."""

        amount_start_idx = features.get_loc("Credit amount_(249.999, 1262.0]")
        duration_start_idx = features.get_loc("Duration_(3.999, 12.0]")
        purpose_start_idx = features.get_loc("Purpose_business")

        def __init__(self, example, decrease_amount=False, decrease_duration=True):
            self.initial_example = example
            self.decrease_amount = decrease_amount
            self.decrease_duration = decrease_duration

            # Slices in the vector for each features
            self.static = self.initial_example[
                    :TransformationWrapper.amount_start_idx]
            self.amount = self.initial_example[
                    TransformationWrapper.amount_start_idx:TransformationWrapper.duration_start_idx]
            self.duration = self.initial_example[
                    TransformationWrapper.duration_start_idx:TransformationWrapper.purpose_start_idx]
            self.purpose = self.initial_example[
                    TransformationWrapper.purpose_start_idx:]

        def _get_neighbour(self, x, direction='pos'):
            """Get the neighbouring value in a quantized one-hot feature vector."""
            idx = np.argmax(x)
            if direction == 'pos' and idx != len(x) - 1:
                return np.roll(x, 1).tolist()
            elif direction == 'neg' and idx != 0:
                return np.roll(x, -1).tolist()
            return []

        def _expand_neighbours(self, field, directions=None):
            """Expand neighbouring values of a quantized feature."""
            if directions is None:
                directions = ['pos', 'neg']

            child_fields = []
            for direction in directions:
                child_fields.append(self._get_neighbour(field, direction=direction))

            child_fields = [x for x in child_fields if len(x) > 0]
            return np.array(child_fields, dtype='uint8')

        def _expand_all(self, field):
            """Expand all values of a categorical feature."""
            child_fields = []
            for i in range(1, len(field)):
                child_fields.append(np.roll(field, i))
            return child_fields

        def expand(self):
            """Generate new transformation."""

            children = []

            # Expand "credit amount".
            for c in self._expand_neighbours(
                    self.amount,
                    directions=['pos', 'neg'] if self.decrease_amount else ['pos']):
                child = np.concatenate((self.static, c, self.duration, self.purpose))
                children.append(child)

            # Expand "duration".
            for c in self._expand_neighbours(
                    self.duration,
                    directions=['pos', 'neg'] if self.decrease_duration else ['pos']):
                child = np.concatenate((self.static, self.amount, c, self.purpose))
                children.append(child)

            # Expand "purpose".
            for c in self._expand_all(self.purpose):
                child = np.concatenate((self.static, self.amount, self.duration, c))
                children.append(child)
            return children

        def __repr__(self):
            return 'TransformationWrapper({})'.format(self.initial_example)

    return TransformationWrapper


def _hash_fn(example):
    """Feature vector hash function."""
    return hash(str(example))


def generate_all_transformations(initial_example, features, transformation_kwargs=None):
    """Generate all transformations of a given example."""
    if transformation_kwargs is None:
        transformation_kwargs = {}
    TransformationWrapper = make_transformation_wrapper(features)

    result = []
    closed = set()

    queue = [initial_example]
    while queue:
        current_example = queue.pop()
        current_example_wrapped = TransformationWrapper(
            current_example, **transformation_kwargs)
        for next_example in current_example_wrapped.expand():
            h = _hash_fn(next_example)
            if h not in closed:
                closed.add(h)
                queue.insert(0, next_example)
                result.append(next_example)

    return result


def example_similarity(exp_context, target_group_idxs, static_cols, i):
    return -np.mean(
        np.linalg.norm(
            exp_context.df[static_cols].iloc[i]
            - exp_context.df[static_cols].iloc[target_group_idxs],
            ord=2,
            axis=1,
        )
    )


def score_group(exp_context, target_group_idxs, trade_off=0.5, custom_clf=None):
    """Get the average accuracy of the model on the target group."""
    if custom_clf is None:
        clf = exp_context.clf
    else:
        clf = custom_clf

    group_avg_score = np.mean(clf.predict(exp_context.raw_datasets.X[target_group_idxs]) == 1)

    # We want to also account for externalities on the other individuals.
    others_idxs = [i for i in exp_context.raw_datasets.train_ind if i not in target_group_idxs]
    others_avg_score = np.mean(
            clf.predict(exp_context.raw_datasets.X[others_idxs]) == exp_context.raw_datasets.y[others_idxs])
    return group_avg_score + trade_off * others_avg_score
