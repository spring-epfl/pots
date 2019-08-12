import numpy as np

from .hash import fast_hash


eps = 10e-10

grad_cache = {}
hessian_cache = {}
avg_hess_inv_cache = {}


def lr_grad(exp_context, x, y=1):
    """Gradient of a sklearn logistic regression."""
    x_hashed = fast_hash(np.concatenate([x, [y]]))
    if x_hashed in grad_cache:
        return grad_cache[x_hashed]

    sign = -1 if y == 1 else +1
    exp_y = np.exp(sign * exp_context.clf.decision_function([x]))
    res = sign * x / (exp_y + 1)

    grad_cache[x_hashed] = res
    return res


def lr_hessian(exp_context, x, y=1):
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


def influence_func(exp_context, target_group_indices, x):
    """Compute an influence function of a given example."""

    # Compute the average Hessian.
    avg_hess = sum([lr_hessian(exp_context, x, y)
        for x, y in zip(exp_context.raw_datasets.X_train, exp_context.raw_datasets.y_train)])
    avg_hess /= len(exp_context.raw_datasets.X_train)

    avg_hess_hash = fast_hash(avg_hess)
    if avg_hess_hash in avg_hess_inv_cache:
        avg_hess_inv = avg_hess_inv_cache[avg_hess_hash]
    else:
        avg_hess_inv = avg_hess_inv_cache[avg_hess_hash] = np.linalg.pinv(avg_hess)

    # Compute the standard influence function.
    grad = lr_grad(exp_context, x)
    influence = np.dot(avg_hess_inv, grad)

    # Task-specific influence.
    task_grad = -sum(lr_grad(exp_context, t, 0)
            for t in exp_context.raw_datasets.X[target_group_indices])
    res = np.dot(task_grad, influence)

    # debug_vals = sorted(zip(exp_context.df_X.columns, influence, task_grad,
    #     exp_context.clf.coef_[0], x), key=lambda t: t[1])
    # debug_data = pd.DataFrame(debug_vals, columns=["col", "inf", "task", "coef", "x"])
    # print(debug_data)

    return res


def clear_grad_cache():
    """Clear cache of computed gradients and Hessians."""
    grad_cache.clear()
    hessian_cache.clear()

