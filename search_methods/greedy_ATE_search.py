import time
from typing import Callable, List

import pandas as pd

from search_methods.ATE_search import ATESearch
from search_methods.probe_ATE_search import ProbManager
from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, get_base_line, \
    calculate_ate_with_uncertainty


class GreedyATESearch(ATESearch):
    def __init__(self, op_probs, max_seq_length: int):
        self.op_probs = op_probs
        self.max_seq_length = max_seq_length

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               transformations_dict: dict[str, Callable], time_out_sec: int = 14400, whole_df_ops: List[str] = None):
        df_ = df.copy()
        base_line_ate = get_base_line(common_causes, df_)
        prob_manager = ProbManager([func_name for func_name, func in transformations_dict.items()], common_causes,
                                   self.op_probs, whole_df_ops=whole_df_ops)

        start_time = time.time()
        sequence = (())

        while len(sequence) <= self.max_seq_length:
            curr_df = apply_data_preparations_seq(df_, sequence, transformations_dict)
            new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
                                                            common_causes)
            current_error = abs(new_ate - target_ate)

            if current_error < epsilon or len(sequence) == self.max_seq_length:
                if current_error < epsilon:
                    print("FOUND SOLUTION")
                else:
                    print("DIDNT FIND SOLUTION")

                print(
                    f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {sequence}\n***""",
                    flush=True)
                print(f"Execution time: {time.time() - start_time:.3f} sec")
                print(f"probability of this sequence is: {prob_manager.get_sequence_probability(sequence)}")
                try:
                    print(
                        f"uncertainty:\n{calculate_ate_with_uncertainty(curr_df.copy(), 'treatment', 'outcome', common_causes)}")
                except Exception as e:
                    print(f"Failed to calculate uncertainty:\n{e}")
                print(f"sequence is: {sequence}")
                exit()

            highest_prob = -1
            selected_func = None
            selected_col = None
            for col in common_causes:
                for func_name in transformations_dict.keys():
                    if func_name=='drop_duplicates':
                        print("here")
                    rule_name = f"{func_name}#{col}"
                    if not whole_df_ops is None and func_name in whole_df_ops:
                        rule_name = f"{func_name}#TABLE"
                    if func_name in whole_df_ops:
                        if any(f_n == func_name for f_n, c in sequence):
                            continue
                    if any(f_n.split("_")[0] == func_name.split("_")[0] for f_n, c in sequence if c == col):
                        continue
                    curr_prob = prob_manager.probs[rule_name]
                    if curr_prob > highest_prob:
                        highest_prob = curr_prob
                        selected_func = func_name
                        selected_col = col if func_name not in whole_df_ops else "TABLE"
            sequence = sequence + ((selected_func, selected_col),)
