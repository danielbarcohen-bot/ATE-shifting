from dowhy import CausalModel


class FastDoWhyATE:
    def __init__(self, df, treatment_col, outcome_col, common_causes):
        """
        Initialize the wrapper.
        Will work ONLY with same columns. (inside numbers wont matter)
        """
        self.df_orig = df.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.common_causes = list(common_causes)

        # Initialize model and identified estimand
        self._identified = self._get_identified_effect()  # cache identification

    def _get_identified_effect(self):
        df_ = self.df_orig.copy()
        df_ = df_.fillna(df_.mean())
        model = CausalModel(
            data=df_,
            treatment=self.treatment_col,
            outcome=self.outcome_col,
            common_causes=self.common_causes
        )
        return model.identify_effect()

    def calculate_ate(self, model):
        """
        Compute ATE using the current data and cached identification.
        changed_df - same structure, only confounders may change the value
        """
        estimate = model.estimate_effect(self._identified, method_name="backdoor.linear_regression")
        return estimate.value
