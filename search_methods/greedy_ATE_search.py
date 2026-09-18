import time
from typing import Callable, List

import pandas as pd

from search_methods.probe_ATE_search import ProbManager
from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
    calculate_ate_with_uncertainty, get_fill_combinations


class GreedyATESearch:
    def __init__(self, op_probs, max_seq_length: int):
        self.op_probs = op_probs
        self.max_seq_length = max_seq_length

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               transformations_dict: dict[str, Callable], time_out_sec: int = 14400, whole_df_ops: List[str] = None,
               legal_ops_by_type = None, legal_fill_by_type = None):

        col_types = df.attrs.get('col_types', None)
        df_ = df.copy()
        needs_fill = df.isnull().values.any()
        prob_manager = ProbManager(
                            list(transformations_dict.keys()),
                            common_causes,
                            self.op_probs,
                            None,
                            whole_df_ops,legal_ops_by_type,legal_fill_by_type,col_types,
                            [col for col in df.columns if df[col].isna().any()])
        fill_seqs = [(seq,prob_manager.get_sequence_probability(seq)) for
                     seq in get_fill_combinations(df_, col_types, legal_fill_by_type)] if needs_fill else []

        operations = [(op, col) for op in transformations_dict.keys() for col in common_causes
                      if not op.startswith('fill_') and op in legal_ops_by_type[col_types[col]]]
        operations += [(op, 'TABLE') for op in whole_df_ops if not op.startswith('fill_')]
        operations.sort(key=lambda op_col: prob_manager.probs[op_col], reverse=True)
        op_index = 0

        start_time = time.time()
        sequence = (())

        if needs_fill:
            sequence = max(fill_seqs,key=lambda x: x[1])[0]
            curr_df = apply_data_preparations_seq(df_, sequence, transformations_dict)
            base_line_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
                                                            common_causes)
        else:
            base_line_ate = calculate_ate_linear_regression_lstsq(df_, 'treatment', 'outcome',
                                                            common_causes)
        fill_length = len(sequence)

        finished = False
        while len(sequence) - fill_length <= self.max_seq_length and not finished:
            curr_df = apply_data_preparations_seq(df_, sequence, transformations_dict)
            new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
                                                            common_causes)
            current_error = abs(new_ate - target_ate)

            if current_error < epsilon or len(sequence) - fill_length == self.max_seq_length:
                break

            # walk down the probability-sorted operations and take the first one that
            # is still allowed to enter the sequence
            selected = None
            while op_index < len(operations):
                func_name, col = operations[op_index]
                op_index += 1
                if col == 'TABLE':
                    if any(f_n == func_name for f_n, c in sequence):
                        continue  # whole-df op already used
                elif any(f_n.split("_")[0] == func_name.split("_")[0] for f_n, c in sequence if c == col):
                    continue  # same op family already applied to this column
                selected = (func_name, col)
                break

            if selected is None:
                #exhausted the families
                finished = True
                continue
            sequence = sequence + (selected,)

        #Print solution:
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

