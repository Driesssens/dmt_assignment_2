from collections import namedtuple
from rankpy.models import LambdaMART

Configuration = namedtuple('Configuration', "epochs estopping max_leaf_nodes shrinkage min_samples_leaf max_depth max_features min_samples_split use_newton_method use_logit_boost use_ada_boost")


def standard_configuration():
    return make_configuration()


def make_configuration(epochs=1000,
                       estopping=50,
                       max_leaf_nodes=7,
                       shrinkage=0.1,
                       min_samples_leaf=50,
                       max_depth=None,
                       max_features=None,
                       min_samples_split=2,
                       use_newton_method=True,
                       use_logit_boost=False,
                       use_ada_boost=False):
    return Configuration(
        epochs=epochs,
        estopping=estopping,
        max_leaf_nodes=max_leaf_nodes,
        shrinkage=shrinkage,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        use_newton_method=use_newton_method,
        use_logit_boost=use_logit_boost,
        use_ada_boost=use_ada_boost)


def make_model(configuration):
    return LambdaMART(metric='nDCG',
                      n_jobs=-1,
                      random_state=42,
                      n_estimators=configuration.epochs,
                      estopping=configuration.estopping,
                      max_leaf_nodes=configuration.max_leaf_nodes,
                      shrinkage=configuration.shrinkage,
                      min_samples_leaf=configuration.min_samples_leaf,
                      max_depth=configuration.max_depth,
                      max_features=configuration.max_features,
                      min_samples_split=configuration.min_samples_split,
                      use_newton_method=configuration.use_newton_method,
                      use_logit_boost=configuration.use_logit_boost,
                      use_ada_boost=configuration.use_ada_boost)
