import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data_loader import LalondeDataLoader, TwinsDataLoader, ACSDataLoader, IHDPDataLoader
from utils import bin_equal_frequency_2, bin_equal_frequency_5, bin_equal_frequency_10, bin_equal_width_2, \
    bin_equal_width_5, bin_equal_width_10, min_max_norm, log_norm, zscore_clip_3, winsorize, \
    calculate_ate_linear_regression_lstsq, apply_data_preparations_seq

large_data_transformations = {
    "bin_equal_frequency_2": bin_equal_frequency_2,
    "bin_equal_frequency_5": bin_equal_frequency_5,
    "bin_equal_frequency_10": bin_equal_frequency_10,
    "bin_equal_width_2": bin_equal_width_2,
    "bin_equal_width_5": bin_equal_width_5,
    "bin_equal_width_10": bin_equal_width_10,
    "norm_min_max": min_max_norm,
    "norm_log": log_norm,
    "zscore_clip_3": zscore_clip_3,
    "winsorize": winsorize
}


def make_sklearn_transformer(series_func, name="unknown"):
    def wrapper(X, name=None):  # name as kwarg, ignored in transform
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_transformed = pd.DataFrame(index=X.index)
        for col in X.columns:
            X_transformed[col] = series_func(X[col])
        return X_transformed.values

    return FunctionTransformer(wrapper, validate=False, kw_args={"name": name})


def ate_epsilon_hinge_score(estimator, X, y, *, treatment_col, outcome_col, common_causes, target_ate, epsilon):
    prep = estimator.named_steps["prep"]

    X_trans = prep.transform(X)

    df_trans = pd.DataFrame(
        X_trans[:, :len(common_causes)],
        columns=common_causes,
        index=X.index
    )
    df_trans[treatment_col] = X_trans[:, len(common_causes)]  # treatment is last
    df_trans[outcome_col] = y.values

    ate_hat = calculate_ate_linear_regression_lstsq(
        df_trans, treatment_col, outcome_col, common_causes
    )

    excess = abs(ate_hat - target_ate) - epsilon
    return -max(0.0, excess)


def print_sklearn_data_prep(df: pd.DataFrame, treatment: str, outcome: str, common_causes: list[str], scorer=None):
    # ----------------------------
    # 1. Dataset
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(outcome, axis=1), df[outcome], test_size=0.25,
        random_state=42
    )

    # -------------------------
    # 2. Column pipelines
    # -------------------------
    preprocessor = ColumnTransformer([
        ("covariates", FunctionTransformer(lambda X: X, validate=False), common_causes),
        (treatment, "passthrough", [treatment])
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression())
    ])

    # -------------------------
    # 3. Build AutoML search space
    # -------------------------

    transform_candidates = [
        make_sklearn_transformer(fn, name=name)
        for name, fn in large_data_transformations.items()
    ]
    # Add identity (no-op)
    transform_candidates.append(FunctionTransformer(lambda X: X, validate=False))

    param_space = {
        "prep__covariates": transform_candidates
    }

    # -------------------------
    # 4. AutoML search
    # -------------------------
    if scorer is None:
        scorer = "r2"
    automl = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_space,
        n_iter=len(transform_candidates),
        cv=5,
        scoring=scorer,
        random_state=0,
        n_jobs=-1
    )
    automl.fit(X_train, y_train)

    # -------------------------
    # 5. Inspect chosen transformations per column
    # -------------------------
    best_pipeline = automl.best_estimator_
    chosen_transformer = best_pipeline.named_steps["prep"].named_transformers_["covariates"]


    if chosen_transformer.kw_args is None:
        print("AutoML chose identity (no preprocessing)")
    else:
        print(chosen_transformer.kw_args)
        chosen_seq = tuple((chosen_transformer.kw_args["name"], item) for item in common_causes)
        print(chosen_seq)
        transformed_df = apply_data_preparations_seq(df.copy(), chosen_seq, large_data_transformations)
        print(f"\nNEW ATE IS: {calculate_ate_linear_regression_lstsq(transformed_df, treatment, outcome, common_causes)}\n")

    print(f"Test {"r2" if scorer == "r2" else "ATE scorer"}:", round(automl.score(X_test, y_test), 4))


def make_ate_scorer(epsilon, target_ate, treatment_col, outcome_col, common_causes):
    def scorer(estimator, X, y):
        return ate_epsilon_hinge_score(
            estimator,
            X,
            y,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            common_causes=common_causes,
            target_ate=target_ate,
            epsilon=epsilon,
        )

    return scorer


def run_experiment(df, target_ATE, epsilon):
    common_causes = df.columns.difference(['treatment', 'outcome']).tolist()
    scorer = make_ate_scorer(epsilon=epsilon, target_ate=target_ATE, treatment_col="treatment", outcome_col="outcome",
                             common_causes=common_causes)
    start = time.time()
    print(f"RUNNING EXPERIMENT with R2. target ATE: {target_ATE}, epsilon: {epsilon}")
    print_sklearn_data_prep(df, 'treatment', 'outcome', common_causes)
    print("took: ", time.time() - start)

    start = time.time()
    print(f"\nRUNNING EXPERIMENT with ATE scorer.")
    print_sklearn_data_prep(df, 'treatment', 'outcome', common_causes, scorer)
    print("took: ", time.time() - start)
    print("~" * 150)


if __name__ == "__main__":
    run_experiment(TwinsDataLoader().load_data().dropna(), 0.0019, 0.000001)
    run_experiment(LalondeDataLoader().load_data().dropna(), 1871, 10)
    run_experiment(ACSDataLoader().load_data().dropna(), 16500, 100)
    run_experiment(IHDPDataLoader().load_data().dropna(), 4.5, 0.5)
