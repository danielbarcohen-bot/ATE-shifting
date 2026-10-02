import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

from data_loader import LalondeDataLoader, WalmartDataLoader, ACSDataLoader
from experiments import largest_data_transformations, whole_df_ops, LEGAL_OPS_BY_TYPE, LEGAL_FILL_BY_TYPE
from utils import calculate_ate_linear_regression_lstsq, apply_data_preparations_seq, get_fill_combinations


class DataPrepTransformer(BaseEstimator, TransformerMixin):
    """Applies a full (op, col) sequence via the same function the other searchers use."""

    def __init__(self, sequence=(), transformations_dict=None, features=None, treatment="treatment"):
        self.sequence = sequence
        self.transformations_dict = transformations_dict
        self.features = features
        self.treatment = treatment

    def fit(self, X, y=None):
        return self

    def apply(self, X):  # X must be the FULL df (incl. outcome)
        return apply_data_preparations_seq(X.copy(), self.sequence, self.transformations_dict)

    def transform(self, X):
        return self.apply(X)[list(self.features) + [self.treatment]].values


class LegalSequenceSampler:
    """Sampler for RandomizedSearchCV (needs only an rvs method). Emits only legal sequences."""

    def __init__(self, common_causes, col_types, legal_ops_by_type, transformations_dict,
                 whole_df_ops, fill_methods, p_whole=0.5):
        self.whole_df_ops = whole_df_ops or []
        self.fill_methods = [tuple(s) for s in fill_methods]
        self.p_whole = p_whole
        base_ops = [op for op in transformations_dict
                    if op not in self.whole_df_ops and not op.startswith("fill_")]
        self.col_ops = {}
        for col in common_causes:
            col_type = col_types.get(col) if col_types else None
            if col_type is None:
                self.col_ops[col] = list(base_ops)
            else:
                self.col_ops[col] = [op for op in base_ops if op in legal_ops_by_type.get(col_type, [])]

    def _one(self, rng):
        seq = list(self.fill_methods[rng.randint(len(self.fill_methods))]) if self.fill_methods else []
        for col, ops in self.col_ops.items():
            i = rng.randint(len(ops) + 1)  # last index = no op on this column
            if i < len(ops):
                seq.append((ops[i], col))  # one op per column => prefix rule holds automatically
        for op in self.whole_df_ops:  # each whole-df op at most once
            if rng.rand() < self.p_whole:
                seq.append((op, "TABLE"))
        return tuple(seq)

    def rvs(self, size=None, random_state=None):
        rng = check_random_state(random_state)
        if size is None:
            return self._one(rng)
        return [self._one(rng) for _ in range(size)]


def make_ate_scorer(epsilon, target_ate, treatment_col, outcome_col, common_causes):
    def scorer(estimator, X, y):
        df_t = estimator.named_steps["prep"].apply(X)
        ate = calculate_ate_linear_regression_lstsq(df_t, treatment_col, outcome_col, common_causes)
        return -max(0.0, abs(ate - target_ate) - epsilon)  # 0 == inside the target range

    return scorer


def print_sklearn_data_prep(df, treatment, outcome, common_causes, data_transformations,
                            whole_df_ops, legal_ops_by_type, fill_by_type,
                            scorer=None, n_iter=100):
    col_types = df.attrs.get('col_types', None)
    fill_methods = get_fill_combinations(df, col_types, fill_by_type) if df.isnull().values.any() else []

    sampler = LegalSequenceSampler(common_causes, col_types, legal_ops_by_type,
                                   data_transformations, whole_df_ops, fill_methods)
    pipeline = Pipeline([
        ("prep", DataPrepTransformer(transformations_dict=data_transformations,
                                     features=common_causes, treatment=treatment)),
        ("model", LinearRegression()),
    ])

    if scorer is None:
        scoring, cv = "r2", 5
    else:
        scoring = scorer
        idx = np.arange(len(df))
        cv = [(idx, idx)]  # score the ATE on the WHOLE df, same as the other searchers

    automl = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={"prep__sequence": sampler},
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=0,
        n_jobs=-1,
        error_score=-np.inf,
    )
    automl.fit(df, df[outcome])  # full df in: ops may need the outcome column

    chosen_seq = automl.best_params_["prep__sequence"]
    print("chosen sequence:", chosen_seq)
    transformed_df = apply_data_preparations_seq(df.copy(), chosen_seq, data_transformations)
    print(f"\nNEW ATE IS: {calculate_ate_linear_regression_lstsq(transformed_df, treatment, outcome, common_causes)}\n")


def run_experiment(df, target_ATE, epsilon, data_transformations, whole_df_ops, legal_ops_by_type, fill_by_type):
    common_causes = df.columns.difference(['treatment', 'outcome']).tolist()
    scorer = make_ate_scorer(epsilon=epsilon, target_ate=target_ATE, treatment_col="treatment",
                             outcome_col="outcome", common_causes=common_causes)
    args = (df, 'treatment', 'outcome', common_causes, data_transformations,
            whole_df_ops, legal_ops_by_type, fill_by_type)

    start = time.time()
    print(f"RUNNING EXPERIMENT with R2. target ATE: {target_ATE}, epsilon: {epsilon}")
    print_sklearn_data_prep(*args)
    print("took: ", time.time() - start)

    start = time.time()
    print(f"\nRUNNING EXPERIMENT with ATE scorer.")
    print_sklearn_data_prep(*args, scorer=scorer)
    print("took: ", time.time() - start)
    print("~" * 150)


if __name__ == "__main__":
    data_transformations = largest_data_transformations

    # run_experiment(LalondeDataLoader().load_data(), 0, 50, data_transformations,whole_df_ops, LEGAL_OPS_BY_TYPE, LEGAL_FILL_BY_TYPE)
    # run_experiment(LalondeDataLoader().load_data(), -500, 500, data_transformations,whole_df_ops, LEGAL_OPS_BY_TYPE, LEGAL_FILL_BY_TYPE)
    # run_experiment(WalmartDataLoader().load_data(), 10133.08, 2533, data_transformations,whole_df_ops, LEGAL_OPS_BY_TYPE, LEGAL_FILL_BY_TYPE)
    run_experiment(ACSDataLoader().load_data(), 18774, 1000, data_transformations,whole_df_ops, LEGAL_OPS_BY_TYPE, LEGAL_FILL_BY_TYPE)
