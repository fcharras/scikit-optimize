from collections import defaultdict, deque, Hashable
from functools import partial

import numpy as np
from scipy.stats import rankdata

from operator import itemgetter

from sklearn.base import is_classifier, clone
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import check_cv
from sklearn.model_selection._validation import _fit_and_score

from . import Optimizer
from .utils import point_asdict, dimensions_aslist
from .space import check_dimension


def _call_and_get_cfg(f, cfg, args, kwargs):
    return [*f(*args, **kwargs), *cfg]


def _make_hashable(p):
    return tuple(tuple(e) if not isinstance(e, Hashable) and not
                 isinstance(e, dict) else e for e in p)


def _return_input(x):
    return x


class BayesSearchCV(BaseSearchCV):
    """Bayesian optimization over hyper parameters.

    BayesSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    Parameters are presented as a list of skopt.space.Dimension objects.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    deterministic : bool
        if True it is considered that the estimator is deterministic,
        else uncertainty on the evaluations is taken into account and
        a same point can be evaluated several times.

    search_spaces : dict, list of dict or list of tuple containing
        (dict, dict).
        One of these cases:
        1. dictionary, where keys are parameter names (strings)
        and values are skopt.space.Dimension instances (Real, Integer
        or Categorical) or any other valid value that defines skopt
        dimension (see skopt.Optimizer docs). Represents search space
        over parameters of the provided estimator.
        2. list of dictionaries: a list of dictionaries, where every
        dictionary fits the description given in case 1 above.
        If a list of dictionary objects is given, then the search is
        performed sequentially for every parameter space with maximum
        number of evaluations set to self.n_iter and number of initial
        points set to self.n_initial_points.
        3. list of (dict, dict): an extension of case 2 above,
        where first element of every tuple is a dictionary representing
        some search subspace, similarly as in case 2, and second element
        can override global parameters for specific subspace. Currently
        support overriding `n_iter`, `n_initial_points` and
        `n_points_per_iter`.

    n_iter : int, default=128
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    n_points_per_iter: int, default = 1
        Number of points computed at each iteration.
        ???: isn't 1 always the best value ?

    n_initial_points : int, default = 1
        Must be >= n_points_per_iter.

    optimizer_kwargs : dict, optional
        Dict of arguments passed to :class:`Optimizer`.  For example,
        ``{'base_estimator': 'RF'}`` would use a Random Forest surrogate
        instead of the default Gaussian Process.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, default = "2 * n_jobs"
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    batch_size: int, default=1
        Controls the number of tasks that get dispatched at the same time.

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

    Example
    -------

    from skopt import BayesSearchCV
    # parameter ranges are specified by one of below
    from skopt.space import Real, Categorical, Integer

    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                        random_state=0)

    # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(
        SVC(),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1,8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=32
    )

    # executes bayesian optimization
    opt.fit(X_train, y_train)

    # model can be saved, used for predictions or scoring
    print(opt.score(X_test, y_test))

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    """

    def __init__(self, estimator, search_spaces, deterministic=True,
                 optimizer_kwargs=None,
                 n_iter=50, scoring=None, n_initial_points=1,
                 n_points_per_iter=1, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', batch_size=1,
                 random_state=None, error_score='raise',
                 return_train_score=False):

        self.search_spaces = search_spaces
        self.deterministic = deterministic
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points
        self.n_points_per_iter = n_points_per_iter
        self.pre_dispatch = pre_dispatch
        self.batch_size = batch_size
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)

        super(BayesSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=None,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _check_search_space(self, search_space):
        """Checks whether the search space argument is correct"""

        if len(search_space) == 0:
            raise ValueError(
                "The search_spaces parameter should contain at least one"
                "non-empty search space, got %s" % search_space
            )

        # check if space is a single dict, convert to list if so
        if isinstance(search_space, dict):
            search_space = [search_space]

        # check if the structure of the space is proper
        if isinstance(search_space, list):
            # convert to just a list of dicts
            dicts_only = []

            # 1. check the case when a tuple of space, n_iter is provided
            for elem in search_space:
                if isinstance(elem, tuple):
                    if len(elem) != 2:
                        raise ValueError(
                            "All tuples in list of search spaces should have"
                            "length 2, and contain (dict, dict), got %s" % elem
                        )
                    subspace, param_override = elem

                    if (not isinstance(param_override, dict)):
                        raise ValueError(
                            "Parameter overriding  in search space should be"
                            "given as a dict, got %s in tuple %s " %
                            (param_override, elem)
                        )

                    # save subspaces here for further checking
                    dicts_only.append(subspace)
                elif isinstance(elem, dict):
                    dicts_only.append(elem)
                else:
                    raise TypeError(
                        "A search space should be provided as a dict or"
                        "tuple (dict, dict), got %s" % elem)

            # 2. check all the dicts for correctness of contents
            for subspace in dicts_only:
                for k, v in subspace.items():
                    check_dimension(v)
        else:
            raise TypeError(
                "Search space should be provided as a dict or list of dict,"
                "got %s" % search_space)

    def _init_search_spaces(self):
        # check if space is a single dict, convert to list if so
        search_spaces = self.search_spaces
        if isinstance(search_spaces, dict):
            search_spaces = [search_spaces]

        n_iters, n_initial_points, n_points_per_iter = [], [], []
        for sp in search_spaces:
            params = sp[1] if isinstance(sp, tuple) else {}
            n_iter = params.get('n_iter', self.n_iter)
            n_initial = params.get('n_initial', self.n_initial_points)
            points_per_iter = params.get('n_points_per_iter',
                                         self.n_points_per_iter)
            if n_initial < points_per_iter:
                raise ValueError("Number of initial points must be at least "
                                 "equal to n_points_per_iter")
            n_initial_points.append(n_initial)
            n_iters.append(n_iter + n_initial)
            n_points_per_iter.append(points_per_iter)

        return search_spaces, n_initial_points, n_iters, n_points_per_iter

    # copied for compatibility with 0.19 sklearn from 0.18 BaseSearchCV
    @property
    def best_score_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['mean_test_score'][self.best_index_]

    # copied for compatibility with 0.19 sklearn from 0.18 BaseSearchCV
    @property
    def best_params_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['params'][self.best_index_]

    def _make_optimizers(self, search_spaces):
        """Instantiate skopt Optimizer class.

        Parameters
        ----------
        search_spaces : dict
            Represents parameter search spaces. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.
        """

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs_ = {}
        else:
            self.optimizer_kwargs_ = dict(self.optimizer_kwargs)
        random_state = check_random_state(self.random_state)
        if random_state is not None:
            self.optimizer_kwargs_['random_state'] = random_state

        # Instantiate optimizers for all the search spaces.
        optimizers = []
        for search_space in search_spaces:
            if isinstance(search_space, tuple):
                search_space = search_space[0]

            print(search_space)

            # TODO: different optimizer_kwargs per search space
            kwargs = self.optimizer_kwargs_.copy()
            kwargs['dimensions'] = dimensions_aslist(search_space)
            optimizers.append(Optimizer(**kwargs))

        self.optimizers_ = optimizers  # will save the states of the optimizers

    def _queue_tasks(self, cand_i, sp_i, sp, n_points, X, y, base_estimator,
                     n_splits, fit_params, cand_logs, cv_iter, task_queue):
        '''
        Generate new points from the optimizer and add them to the queue of
        points to evaluate.
        '''
        params = self.optimizers_[sp_i].ask(n_points=n_points)

        for cand_i, p in enumerate(params, cand_i + 1):
            if _make_hashable(p) in cand_logs and self.deterministic:
                for res in cand_logs[_make_hashable(p)]:
                    res[-3] = cand_i
                    task_queue.append(delayed(_return_input)(res))
            else:
                cand_logs[_make_hashable(p)] = [None] * n_splits
                p_dict = point_asdict(sp, p)
                for split_i, (train, test) in enumerate(cv_iter):

                    args = [base_estimator,
                            X, y, self.scorer_,
                            train, test, self.verbose, p_dict]

                    kwargs = dict(
                        fit_params=fit_params,
                        return_train_score=self.return_train_score,
                        return_n_test_samples=True,
                        return_times=True, return_parameters=True,
                        error_score=self.error_score)

                    cfg = (sp_i, cand_i, split_i, p)

                    task_queue.append(delayed(_call_and_get_cfg)(
                        _fit_and_score, cfg, args, kwargs))

        return cand_i

    def _gen_steps(self, pool, pre_dispatch, base_estimator, fit_params,
                   n_initial_points, n_iters, n_points_per_iter, X, y,
                   n_splits, search_spaces, cv_iter):
        """Generate points asynchronously and log the results.
        """

        optimizers = self.optimizers_
        cand_logs = {}  # save the scores for each candidate and each split
        cand_i = -1  # index of candidates

        task_queue = deque()

        # initialize initial points for each search space
        for sp_i, sp in enumerate(search_spaces):
            cand_i = self._queue_tasks(
                cand_i, sp_i, sp, n_initial_points[sp_i], X, y,
                base_estimator, n_splits, fit_params, cand_logs,
                cv_iter, task_queue,)

        # now get asynchronously the result and create new tasks from them
        # save the params and the mean scores for each search space
        sp_partial_batch_results = [([], []) for sp in search_spaces]
        n_iters_queued = n_initial_points
        while n_iters_queued < n_iters or len(task_queue) > 0:
            try:
                batch = pool.get_last_async_result().result()
            except IndexError:
                batch = []  # happens when no job have completed yet.
            for res in batch:
                (sp_i, cand_i, split_i, p) = res[-4:]
                cand_log = cand_logs[_make_hashable(p)]
                cand_log[split_i] = res
                # if all scores for all splits for all points of a batch
                # are available we can compute the means, tell the
                # optimizer and ask for a new batch.
                try:
                    scores = [res[-9] for res in cand_log]
                    mean_score = np.mean(scores)
                    params, scores = sp_partial_batch_results[sp_i]
                    params.append(p)
                    scores.append(-mean_score)
                    remaining_iter = n_iters[sp_i] - n_iters_queued[sp_i]
                    if len(scores) == n_points_per_iter[sp_i] \
                            and remaining_iter > 0:
                        sp_partial_batch_results[sp_i] = ([], [])
                        optimizers[sp_i].tell(params, scores)
                        n_points = min(n_points_per_iter[sp_i], remaining_iter)
                        cand_i = self._queue_tasks(
                            cand_i, sp_i, search_spaces[sp_i], n_points, X,
                            y, base_estimator, n_splits, fit_params,
                            cand_logs, cv_iter, task_queue,)
                        n_iters_queued[sp_i] += n_points
                except TypeError:
                    pass

            for i in range(min(self.batch_size, len(task_queue))):
                yield task_queue.popleft()

    @property
    def total_iterations(self):
        """
        Count total iterations that will be taken to explore
        all subspaces with `fit` method.

        Returns
        -------
        max_iter: int, total number of iterations to explore
        """
        total_iter = 0
        for elem in self.search_spaces:
            n_iter = self.n_iter
            if isinstance(elem, tuple):
                n_iter = elem[1]['n_iter']
            total_iter += n_iter
        return total_iter

    def fit(self, X, y=None, groups=None, callbacks=None, **fit_params):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """

        search_spaces, n_initial_points, n_iters, n_points_per_iter = \
            self._init_search_spaces()

        self._make_optimizers(search_spaces)

        estimator = self.estimator
        cv = check_cv(self.cv, y,
                      classifier=is_classifier(estimator))
        self.multimetric_ = False
        self.scorer_ = check_scoring(
            self.estimator, scoring=self.scoring)
        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        n_candidates = sum(n_iters)
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))
        cv_iter = list(cv.split(X, y, groups))
        n_jobs = self.n_jobs
        # account for case n_jobs < 0
        if n_jobs < 0:
            n_jobs = max(1, cpu_count() + n_jobs + 1)

        base_estimator = clone(self.estimator)

        max_dispatch = np.inf
        # optimal condition to limit max_dispatch to avoid process starvation
        for ip, ipt in zip(n_initial_points, n_points_per_iter):
            max_dispatch = min(int((ip - ipt) /
                                   (2 + self.batch_size / n_splits)),
                               max_dispatch)

        if hasattr(self.pre_dispatch, 'endswith'):
            pre_dispatch = int(eval(self.pre_dispatch))
        else:
            pre_dispatch = int(self.pre_dispatch)
        if pre_dispatch > max_dispatch:
            pre_dispatch = max_dispatch
            print('Setting pre_dispatch to %r tasks to prevent '
                  'process starvation' % pre_dispatch)

        pool = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                        pre_dispatch=pre_dispatch, batch_size=self.batch_size
                        )

        out = pool(self._gen_steps(
            pool, pre_dispatch, base_estimator, fit_params,
            n_initial_points, n_iters, n_points_per_iter, X, y,
            n_splits, search_spaces, cv_iter))

        # sort by candidate and then by split.
        out = map(itemgetter(slice(None, -4)),
                  sorted(out, key=itemgetter(-3, -2)))

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        assert(n_candidates == len(candidate_params))
        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(
            MaskedArray,
            np.empty(n_candidates,),
            mask=True,
            dtype=object))

        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **fit_params)
            else:
                best_estimator.fit(X, **fit_params)
            self.best_estimator_ = best_estimator

        return self
