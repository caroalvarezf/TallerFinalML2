from sklearn.metrics import make_scorer
import numpy as np


def bike_number_error(y, y_pred, understock_cost=0.6, overstock_cost=0.4):
    error = np.zeros(y.shape[0])
    error = y - y_pred
    error [error>=0] = error*understock_cost
    error [error<0]=error*overstock_cost*(-1)
    score = error.mean()

    return score


def get_metric_name_mapping():
    return {_mae(): bike_number_error}


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _mae(): make_scorer(bike_number_error, greater_is_better=False, **params)
    }
    return mapping[name]


def _mae():
    return "bike number error"