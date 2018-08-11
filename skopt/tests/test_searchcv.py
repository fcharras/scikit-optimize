"""Test scikit-optimize based implementation of hyperparameter
search with interface similar to those of GridSearchCV
"""

import pytest

from sklearn.utils.testing import assert_greater, assert_equal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV


@pytest.mark.parametrize("surrogate", ['gp', None])
@pytest.mark.parametrize("n_jobs", [1, -1])  # test sequential and parallel
def test_searchcv_runs(surrogate, n_jobs):
    """
    Test whether the cross validation search wrapper around sklearn
    models runs properly with available surrogates and with single
    or multiple workers.

    Parameters
    ----------

    * `surrogate` [str or None]:
        A class of the scikit-optimize surrogate used. None means
        to use default surrogate.

    * `n_jobs` [int]:
        Number of parallel processes to use for computations.

    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # check if empty search space is raising errors
    with pytest.raises(ValueError):
        BayesSearchCV(SVC(), {})

    # check if invalid dimensions are raising errors
    with pytest.raises(ValueError):
        BayesSearchCV(SVC(), {'C': '1 ... 100.0'})

    with pytest.raises(TypeError):
        BayesSearchCV(SVC(), ['C', (1.0, 1)])

    # create an instance of a surrogate if it is not a string
    if surrogate is not None:
        optimizer_kwargs = {'base_estimator': surrogate}
    else:
        optimizer_kwargs = None

    opt = BayesSearchCV(
        SVC(),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_jobs=n_jobs, n_iter=24,
        n_initial_points=12,
        n_points_per_iter=2,
        pre_dispatch=100,  # triggers automatic setting
        optimizer_kwargs=optimizer_kwargs
    )

    opt.fit(X_train, y_train)

    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert_greater(opt.score(X_test, y_test), 0.9)

