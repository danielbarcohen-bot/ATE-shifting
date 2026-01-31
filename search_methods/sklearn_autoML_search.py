# # # import numpy as np
# # # import pandas as pd
# # # from tpot import TPOTRegressor
# # # from sklearn.metrics import make_scorer
# # # from sklearn.linear_model import Ridge
# # # from sklearn.model_selection import train_test_split
# # #
# # # # --- 1. Define 'Legal' Transformations ---
# # # # We restrict TPOT to simple, interpretable operations (no complex stacking/ensembling)
# # # # to keep the sequence "short" and "data prep" focused.
# # # tpot_config = {
# # #     'sklearn.preprocessing.StandardScaler': {},
# # #     'sklearn.preprocessing.MinMaxScaler': {},
# # #     'sklearn.preprocessing.RobustScaler': {},
# # #     'sklearn.preprocessing.MaxAbsScaler': {},
# # #     'sklearn.decomposition.PCA': {
# # #         'n_components': range(1, 10),
# # #         'svd_solver': ['arpack', 'randomized']
# # #     },
# # #     'sklearn.feature_selection.SelectPercentile': {
# # #         'percentile': range(1, 100)
# # #     },
# # #     # We force the final estimator to be a simple Ridge regression (S-Learner base)
# # #     # This ensures TPOT focuses on finding the best *preprocessing*, not the best *model*.
# # #     'sklearn.linear_model.Ridge': {
# # #         'alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0]
# # #     }
# # # }
# # #
# # #
# # # # --- 2. The Custom Causal Scorer ---
# # # def causal_ate_scorer(estimator, X, y_true):
# # #     """
# # #     Custom Scorer for TPOT.
# # #     Instead of checking how well we predict Y, we check how close
# # #     the implied ATE is to the TARGET_ATE.
# # #     """
# # #     # 1. Identify Treatment Column (Assumed to be the last column for this generic scorer)
# # #     # In a real scenario, you might handle this by column name if X is a DataFrame
# # #     # TPOT converts X to numpy array internally, so we use indices.
# # #
# # #     # We assume Treatment is the LAST column in the feature matrix X
# # #     # (We will ensure this setup in data preparation)
# # #     X_matrix = X if isinstance(X, np.ndarray) else X.values
# # #     t_idx = -1
# # #
# # #     # 2. Create Counterfactuals (S-Learner Approach)
# # #     X_1 = X_matrix.copy()
# # #     X_0 = X_matrix.copy()
# # #
# # #     X_1[:, t_idx] = 1  # Force Treatment = 1
# # #     X_0[:, t_idx] = 0  # Force Treatment = 0
# # #
# # #     # 3. Predict Outcomes
# # #     # Note: 'estimator' here is the entire pipeline TPOT is testing at that moment
# # #     y_1 = estimator.predict(X_1)
# # #     y_0 = estimator.predict(X_0)
# # #
# # #     # 4. Calculate Estimated ATE
# # #     ate_est = np.mean(y_1 - y_0)
# # #
# # #     # 5. Calculate Error (Negative absolute difference because TPOT maximizes score)
# # #     # We use a global TARGET_ATE variable for the target
# # #     error = -abs(ate_est - TARGET_ATE)
# # #
# # #     return error
# # #
# # #
# # # # --- 3. The Main Algorithm ---
# # # def find_best_causal_sequence(data, treatment_col, outcome_col, target_ate, epsilon):
# # #     """
# # #     Uses TPOT to find the best data prep sequence to hit the target ATE.
# # #     """
# # #     global TARGET_ATE
# # #     TARGET_ATE = target_ate
# # #
# # #     # Separate Features and Target
# # #     # We MUST include Treatment in X so the model can learn the effect
# # #     features = [c for c in data.columns if c != outcome_col]
# # #
# # #     # Reorder columns so Treatment is LAST (crucial for our numpy-based scorer)
# # #     features = [c for c in features if c != treatment_col] + [treatment_col]
# # #
# # #     X = data[features]
# # #     y = data[outcome_col]
# # #
# # #     # Split Data
# # #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # #
# # #     print(f"Searching for pipeline to match ATE: {target_ate} (epsilon: {epsilon})...")
# # #
# # #     # Initialize TPOT
# # #     # periodic_checkpoint_folder helps save progress
# # #     model = TPOTRegressor(
# # #         generations=5,  # Increase for better results (e.g., 50)
# # #         population_size=20,  # Increase for better diversity (e.g., 50)
# # #         verbosity=2,
# # #         config_dict=tpot_config,  # Use our legal set
# # #         scoring=causal_ate_scorer,  # Use our custom CAUSAL objective
# # #         early_stop=5,  # Stop if we don't improve
# # #         n_jobs=-1  # Use all CPUs
# # #     )
# # #
# # #     model.fit(X_train, y_train)
# # #
# # #     # Evaluate on Test Set
# # #     final_score = causal_ate_scorer(model.fitted_pipeline_, X_test, y_test)
# # #     final_ate_error = abs(final_score)
# # #
# # #     print("\n" + "=" * 30)
# # #     print(f"Final ATE Error: {final_ate_error:.5f}")
# # #     print(f"Success? {final_ate_error < epsilon}")
# # #     print("=" * 30)
# # #
# # #     # Export the steps
# # #     print("\nBest Data Prep Sequence:")
# # #     for step_name, step_obj in model.fitted_pipeline_.steps[:-1]:  # Exclude final estimator
# # #         print(f" -> {step_name}: {step_obj}")
# # #
# # #     return model.fitted_pipeline_
# # #
# # #
# # # # --- 4. Mock Data for Demonstration ---
# # # if __name__ == "__main__":
# # #     # Generate Synthetic Data with a known confounder
# # #     np.random.seed(42)
# # #     N = 1000
# # #     # Confounder Z
# # #     Z = np.random.normal(0, 1, N)
# # #     # Treatment T depends on Z (Propensity)
# # #     T = np.random.binomial(1, 1 / (1 + np.exp(-Z)))
# # #     # Outcome Y depends on Z and T (True ATE = 2.0)
# # #     Y = 2.0 * T + 3.0 * Z + np.random.normal(0, 0.5, N)
# # #
# # #     df = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})
# # #
# # #     # Goal: Recover True ATE (2.0)
# # #     # Note: If we just did Mean(Y|T=1) - Mean(Y|T=0), we'd get biased result due to Z
# # #     pipeline = find_best_causal_sequence(
# # #         data=df,
# # #         treatment_col='T',
# # #         outcome_col='Y',
# # #         target_ate=2.0,
# # #         epsilon=0.1
# # #     )
# #
# # import numpy as np
# # import pandas as pd
# #
# # from sklearn.linear_model import LinearRegression
# # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # from sklearn.impute import SimpleImputer
# # from sklearn.decomposition import PCA
# # from sklearn.pipeline import Pipeline
# # from sklearn.metrics import make_scorer
# # from sklearn.base import clone
# #
# # from tpot import TPOTRegressor
# #
# # from data_loader import LalondeDataLoader
# #
# #
# # def estimate_ate_regression_adjustment(X, T, Y):
# #     model = LinearRegression()
# #     model.fit(np.column_stack([T, X]), Y)
# #
# #     X1 = np.column_stack([np.ones_like(T), X])
# #     X0 = np.column_stack([np.zeros_like(T), X])
# #
# #     y1 = model.predict(X1)
# #     y0 = model.predict(X0)
# #
# #     return np.mean(y1 - y0)
# #
# # def make_causal_scorer(
# #     target_ate,
# #     epsilon,
# #     treatment_index=0,
# #     lambda_complexity=0.01,
# # ):
# #     def causal_scorer(estimator, X, y):
# #         """
# #         estimator: sklearn Pipeline from TPOT
# #         X: np.ndarray (treatment + covariates)
# #         y: outcome
# #         """
# #
# #         # Split treatment / covariates
# #         T = X[:, treatment_index]
# #         X_cov = np.delete(X, treatment_index, axis=1)
# #
# #         # Apply preprocessing only (ignore final regressor)
# #         if isinstance(estimator, Pipeline):
# #             preprocess = estimator[:-1]
# #             X_t = preprocess.transform(X)
# #         else:
# #             X_t = X
# #
# #         T_t = X_t[:, treatment_index]
# #         X_cov_t = np.delete(X_t, treatment_index, axis=1)
# #
# #         ate_hat = estimate_ate_regression_adjustment(X_cov_t, T_t, y)
# #
# #         error = abs(ate_hat - target_ate)
# #
# #         # epsilon-insensitive loss
# #         if error <= epsilon:
# #             score = 0.0
# #         else:
# #             score = -error
# #
# #         # complexity penalty
# #         if isinstance(estimator, Pipeline):
# #             score -= lambda_complexity * len(estimator.steps)
# #
# #         return score
# #
# #     return make_scorer(causal_scorer, greater_is_better=True)
# #
# #
# # def get_allowed_preprocessors():
# #     return {
# #         "sklearn.preprocessing.StandardScaler": {},
# #         "sklearn.preprocessing.MinMaxScaler": {},
# #         "sklearn.impute.SimpleImputer": {
# #             "strategy": ["mean"]
# #         }
# #     }
# #
# # config_dict = {
# #     # standard scalers
# #     "sklearn.preprocessing.StandardScaler": {},
# #     "sklearn.preprocessing.MinMaxScaler": {},
# #
# #     # missing values
# #     "sklearn.impute.SimpleImputer": {"strategy": ["mean", "median"]},
# #
# #     # optional custom transformer
# #     "my_transforms.LogTransform": {"eps": [1e-6, 1e-3]},
# #
# #     # final estimator
# #     "sklearn.linear_model.LinearRegression": {}
# # }
# #
# #
# # def tpot_causal_dataprep_baseline(
# #     df,
# #     treatment_col,
# #     outcome_col,
# #     target_ate,
# #     epsilon,
# #     generations=40,
# #     population_size=80,
# # ):
# #     covariates = [c for c in df.columns if c not in [treatment_col, outcome_col]]
# #
# #     X = df[[treatment_col] + covariates].values
# #     y = df[outcome_col].values
# #
# #     scorer = make_causal_scorer(
# #         target_ate=target_ate,
# #         epsilon=epsilon,
# #         treatment_index=0,
# #     )
# #
# #     tpot = TPOTRegressor(
# #         generations=generations,
# #         population_size=population_size,
# #         scoring=scorer,
# #         cv=2,#1,  # VERY IMPORTANT: no CV for causal scoring
# #         verbosity=2,
# #         config_dict=config_dict,
# #         disable_update_check=True,
# #         random_state=0,
# #     )
# #
# #     tpot.fit(X, y)
# #
# #     pipeline = tpot.fitted_pipeline_
# #
# #     # Final evaluation
# #     X_t = pipeline[:-1].transform(X)
# #     T_t = X_t[:, 0]
# #     X_cov_t = X_t[:, 1:]
# #
# #     final_ate = estimate_ate_regression_adjustment(X_cov_t, T_t, y)
# #
# #     return {
# #         "pipeline": pipeline,
# #         "final_ate": final_ate,
# #         "error": abs(final_ate - target_ate),
# #         "pipeline_length": len(pipeline.steps),
# #     }
# #
# #
# # if __name__=="__main__":
# #
# #     df_lalonde_no_missing_values = LalondeDataLoader().load_data().dropna()
# #
# #
# #     result = tpot_causal_dataprep_baseline(
# #         df=df_lalonde_no_missing_values,
# #         treatment_col="treatment",
# #         outcome_col="outcome",
# #         target_ate=1871,
# #         epsilon=2000,#1,
# #     )
# #
# #     print("Final ATE:", result["final_ate"])
# #     print("Error:", result["error"])
# #     print("Pipeline length:", result["pipeline_length"])
# #     print("Pipeline:", result["pipeline"])
#
# import numpy as np
# import pandas as pd
# from tpot import TPOTRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import Ridge
# from sklearn.metrics import make_scorer
#
#
# # --- 1. The Scorer (No changes needed here, it's already solid) ---
# def estimate_ate_safe(X_cov_transformed, T, y):
#     try:
#         T = T.reshape(-1, 1)
#         features = np.hstack([T, X_cov_transformed])
#         model = Ridge(alpha=1.0)
#         model.fit(features, y)
#         X1 = np.hstack([np.ones_like(T), X_cov_transformed])
#         X0 = np.hstack([np.zeros_like(T), X_cov_transformed])
#         return np.mean(model.predict(X1) - model.predict(X0))
#     except:
#         return np.nan
#
#
# def make_causal_scorer(target_ate, epsilon, lambda_complexity=0.01):
#     def score_func(estimator, X, y):
#         try:
#             T = X[:, 0]
#             X_cov = X[:, 1:]
#             if hasattr(estimator, 'steps') and len(estimator.steps) > 1:
#                 prep = Pipeline(estimator.steps[:-1])
#                 X_cov_transformed = prep.transform(X_cov)
#             else:
#                 X_cov_transformed = X_cov
#             ate_hat = estimate_ate_safe(X_cov_transformed, T, y)
#             if np.isnan(ate_hat): return -10 ** 10
#             error = abs(ate_hat - target_ate)
#             score = 0 if error <= epsilon else -error
#             return score - (lambda_complexity * len(getattr(estimator, 'steps', [])))
#         except:
#             return -10 ** 10
#
#     return make_scorer(score_func, greater_is_better=True)
#
#
# # --- 2. The Corrected Configuration Strategy ---
# def find_ate_sequence(df, treatment_col, outcome_col, target_ate, epsilon):
#     covs = [c for c in df.columns if c not in [treatment_col, outcome_col]]
#     X = df[[treatment_col] + covs].values
#     y = df[outcome_col].values
#
#     # TPOT MUST HAVE STRINGS AS KEYS.
#     # To fix the 'ImportError', we use the built-in strings TPOT recognizes.
#     causal_config = {
#         'sklearn.preprocessing.StandardScaler': {},
#         'sklearn.preprocessing.RobustScaler': {},
#         'sklearn.preprocessing.MinMaxScaler': {},
#         'sklearn.decomposition.PCA': {
#             'n_components': [1, 2, 3],
#             'svd_solver': ['randomized']
#         },
#         'sklearn.linear_model.Ridge': {
#             'alpha': [1.0]
#         }
#     }
#
#     tpot = TPOTRegressor(
#         generations=5,
#         population_size=20,
#         scoring=make_causal_scorer(target_ate, epsilon),
#         cv=2,
#         verbosity=3,  # Set to 3 to see the internal "Imports" TPOT is doing
#         config_dict=causal_config,
#         random_state=42
#     )
#
#     tpot.fit(X, y)
#     return tpot
#
#
# if __name__ == "__main__":
#     # Ensure a small, clean dataset for testing
#     data = pd.DataFrame(np.random.randn(100, 4), columns=['T', 'x1', 'x2', 'Y'])
#     data['T'] = (data['T'] > 0).astype(float)
#
#     # Execute
#     try:
#         tpot_obj = find_ate_sequence(data, 'T', 'Y', target_ate=0.8, epsilon=0.05)
#         print("\nPipeline Found:", tpot_obj.fitted_pipeline_)
#     except Exception as e:
#         print(f"\nFailed with error: {e}")
#         print(
#             "Tip: Ensure you are running this in a standard Python environment (not certain restricted Jupyter kernels).")


"""
Sklearn AutoML baseline with column-specific preprocessing.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer


from data_loader import LalondeDataLoader, TwinsDataLoader, ACSDataLoader, IHDPDataLoader
from utils import bin_equal_frequency_2, bin_equal_frequency_5, bin_equal_frequency_10, bin_equal_width_2, \
    bin_equal_width_5, bin_equal_width_10, min_max_norm, log_norm, zscore_clip_3, winsorize, \
    calculate_ate_linear_regression_lstsq

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
def make_sklearn_transformer(series_func):
    """
    Wrap a function that expects a Series so it works with sklearn.

    - Input: X (DataFrame or 2D array)
    - Output: 2D array (n_samples, n_features)
    """
    def wrapper(X):
        # convert to DataFrame if numpy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_transformed = pd.DataFrame(index=X.index)

        # apply your function to each column
        for col in X.columns:
            X_transformed[col] = series_func(X[col])
        return X_transformed.values  # return 2D array for sklearn
    return FunctionTransformer(wrapper, validate=False)

def ate_epsilon_hinge_score(
    estimator,
    X,
    y,
    *,
    treatment_col,
    outcome_col,
    common_causes,
    target_ate,
    epsilon,
):
    # prep = estimator.named_steps["prep"]
    # # X_trans = prep.transform(X)
    # # feature_names = prep.get_feature_names_out()
    # #
    # # df_trans = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
    # # df_trans[treatment_col] = X[treatment_col].values
    # # df_trans[outcome_col] = y.values
    #
    # X_trans = prep.transform(X)
    # df_trans = pd.DataFrame(X_trans, index=X.index)  # no columns
    # df_trans[treatment_col] = X[treatment_col].values
    # df_trans[outcome_col] = y.values
    #
    # ate_hat = calculate_ate_linear_regression_lstsq(
    #     df=df_trans,
    #     treatment=treatment_col,
    #     outcome=outcome_col,
    #     common_causes=common_causes,
    # )
    df_for_ate = X.copy()
    df_for_ate[outcome_col] = y.values

    ate_hat = calculate_ate_linear_regression_lstsq(
        df_for_ate,
        treatment_col,
        outcome_col,
        common_causes
    )
    # Hinge loss
    excess = abs(ate_hat - target_ate) - epsilon
    return -max(0.0, excess)

def print_sklearn_data_prep(df: pd.DataFrame, treatment: str, outcome: str, common_causes: list[str], scorer = None):
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
        make_sklearn_transformer(fn)
        for fn in large_data_transformations.values()
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
        scoring=scorer,  # "r2",
        random_state=0,
        n_jobs=-1
    )
    automl.fit(X_train, y_train)

    # -------------------------
    # 5. Inspect chosen transformations per column
    # -------------------------
    best_pipeline = automl.best_estimator_
    chosen_transformer = best_pipeline.named_steps["prep"].named_transformers_["covariates"]

    for name, fn in large_data_transformations.items():
        if getattr(chosen_transformer, "func", None) is fn:
            print("AutoML chose data prep:", name)
            break
    else:
        print("AutoML chose identity (no preprocessing)")

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

def run_expirement(df, target_ATE, epsilon):
    common_causes = df.columns.difference(['treatment', 'outcome']).tolist()
    scorer = make_ate_scorer(epsilon=epsilon,target_ate=target_ATE,treatment_col="treatment",outcome_col="outcome",common_causes=common_causes)
    print(f"RUNNING EXPERIMENT with R2. target ATE: {target_ATE}, epsilon: {epsilon}")
    print_sklearn_data_prep(df, 'treatment', 'outcome', common_causes)
    print(f"\nRUNNING EXPERIMENT with ATE scorer.")
    print_sklearn_data_prep(df, 'treatment', 'outcome', common_causes, scorer)
    print("~" * 150)

if __name__ == "__main__":
    # run_expirement(TwinsDataLoader().load_data().dropna(), 0.0019, 0.000001)
    run_expirement(LalondeDataLoader().load_data().dropna(), 1871, 1)