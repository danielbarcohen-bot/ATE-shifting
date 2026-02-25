import random

import pandas as pd

from data_loader import TwinsDataLoader, LalondeDataLoader, ACSDataLoader, IHDPDataLoader
from utils import bin_equal_frequency_2, fill_median, fill_min, zscore_clip_3, bin_equal_frequency_10, \
    bin_equal_frequency_5, bin_equal_width_5, bin_equal_width_2, bin_equal_width_10, min_max_norm, log_norm, winsorize
df_twins = TwinsDataLoader().load_data()
df_lalonde = LalondeDataLoader().load_data()
df_acs = ACSDataLoader().load_data()
df_IHDP = IHDPDataLoader().load_data()

df_twins_no_missing_values = df_twins.dropna()
print(f"\n\n{'#'*100}\n             DELETE THIS{'#'*100}\n\n\n\n")
# df_twins_no_missing_values = df_twins_no_missing_values[['wt', 'gestat10'] + [c for c in df_twins_no_missing_values.columns if c not in ['wt', 'gestat10']]]
df_lalonde_no_missing_values = df_lalonde.dropna()
df_acs_no_missing_values = df_acs.dropna()
df_IHDP_no_missing_values = df_IHDP.dropna()

small_data_transformations_no_fill = {
    "zscore_clip_3": zscore_clip_3,
    "bin_2": bin_equal_frequency_2
}

small_data_transformations = {
    "fill_median": fill_median,
    "fill_min": fill_min,
    "zscore_clip_3": zscore_clip_3,
    "bin_2": bin_equal_frequency_2
}
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

EXPERIMENTS = {

    ######### df can have missing values
    "EXP1": {  # twins
        "df": df_twins,
        "transformations_dict": small_data_transformations,
        "common_causes": ['wt', 'hydra', 'nprevistq', 'gestat10'],
        "target_ate": 0.0014,
        "epsilon": 0.00005,
        "max_length": 5
        # best sequence:
        # start ATE :
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes:  sec | popped
    },
    "EXP2": {  # run algorithms on the whole dataset, different target
        "df": df_twins,
        "transformations_dict": small_data_transformations,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.00,
        "epsilon": 0.0033,
        "max_length": 10
        # best sequence: [('bin_2', 'wt'), ('zscore_clip_3', 'gestat10')]
        # start ATE:  0.0613
        # brute takes: 944 sec | popped 3965964 (optimal solution) | ATE is 0.000894
        # prune takes: 12075 sec | popped from Q 6588 | pruned  989297(optimal solution)
        # probe takes: 727 sec | popped 256 (not optimal len 4) | ATE is 0.0032208
    },
    "EXP3": {  # the result need to be with 3 operations
        "df": df_twins,
        "transformations_dict": small_data_transformations,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.0,
        "epsilon": 0.0005,
        "max_length": 10
        # best sequence:
        # start ATE: 0.06132
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes:  817 sec | popped 310 (len 3) | ATE 0.00026
    },
    "EXP4": {  # poc example - shift ATE LALONDE
        "df": df_lalonde,
        "transformations_dict": small_data_transformations,
        "common_causes": df_lalonde.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 1577,
        "epsilon": 1,
        "max_length": 10
        # best sequence:  [('zscore_clip_3', 'age'), ('bin_2', 'black'), ('zscore_clip_3', 'education'), ('bin_2', 'education'), ('bin_2', 'hispanic')]
        # start ATE : 1671.130
        # brute takes: 3377 sec | popped 14158032 (optimal solution)
        # prune takes: 2.5 sec | popped from Q 201 | pruned 4539 (optimal solution)
        # probe takes: < 1 sec | popped 48 (optimal solution)
    },
    "EXP5": {  # poc example - shift ATE ACS
        "df": df_acs,
        "transformations_dict": small_data_transformations,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 20376,
        "epsilon": 1,
        "max_length": 5
        # best sequence:  [('bin_2', 'Public health coverage'), ('bin_2', 'gender'), ('bin_2', 'insurance through employer'), ('bin_2', 'medicare for people 65 and older'), ('bin_2', 'private health coverage')]
        # start ATE : 8774.433205040823
        # brute takes:  sec | popped  (optimal solution)
        # prune takes: 200 sec | popped from Q 175 | pruned 3772 (optimal solution)
        # probe takes: 54 sec | popped 27 (optimal solution)
    },
    "EXP6": {  # poc example - shift ATE IHDP
        "df": df_IHDP,
        "transformations_dict": small_data_transformations,
        "common_causes": ["x" + str(i) for i in range(1, 26)],
        "target_ate": 3.994,
        "epsilon": 0.0005,
        "max_length": 5
        # best sequence:   (('bin_2', 'x6'), ('bin_2', 'x11'), ('bin_2', 'x22'))
        # start ATE : 3.92
        # brute takes: 3126 sec | popped 23379800 (optimal solution)
        # prune takes: 280 sec | popped from Q 3205 | pruned 292435(optimal solution)
        # probe takes: 4 sec | popped 40 (NOT optimal solution - len 5)
    },
    ########################################################################################################################
    ######### df Does not have missing values |adding new probe mechanics

    "EXP8": {  # run algorithms on the whole dataset, different target
        "df": df_twins_no_missing_values,
        "transformations_dict": small_data_transformations_no_fill,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.0019,  # 0,
        "epsilon": 0.000001,  # 0.0019,
        "max_length": 5,
        "sequence_length": 3
        # best sequence:
        # start ATE : (('zscore_clip_3', 'adequacy'), ('bin_2', 'lung'), ('bin_2', 'wt'))
        # brute takes: 215 sec | popped 1663008 (optimal solution)
        # prune takes: 1535 sec | popped from Q 6020 | pruned 449137 (optimal solution)
        # probe takes: 28 sec | popped 61| restarts 2 (optimal solution)
        # probe (lin reg heu) takes: 9 sec | popped 15| restarts 2 (optimal solution)
    },
    "EXP9": {  # the result need to be with 3 operations
        "df": df_twins_no_missing_values,
        "transformations_dict": small_data_transformations_no_fill,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": -0.06,
        "epsilon": 0.06,
        "max_length": 5
        # best sequence:  (('zscore_clip_3', 'gestat10'), ('bin_2', 'wt'))
        # start ATE : 0.06
        # brute takes: 63 sec | popped 494598 (optimal solution)
        # prune takes: 628 sec | popped from Q 2444 | pruned 172225 (optimal solution)
        # probe takes: 61 sec | popped 117 | restarts 1 (optimal solution)
        # probe (lin reg heu) takes: 34 sec | popped 67| restarts 1 (optimal solution)
    },
    "EXP10": {  # poc example - shift ATE LALONDE
        "df": df_lalonde_no_missing_values,
        "transformations_dict": small_data_transformations_no_fill,
        "common_causes": df_lalonde.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 1577,
        "epsilon": 1,
        "max_length": 10
        # best sequence: (('zscore_clip_3', 'age'), ('bin_2', 'black'), ('zscore_clip_3', 'education'), ('bin_2', 'education'), ('bin_2', 'hispanic'))
        # start ATE : 1671
        # brute takes: 85 sec | popped 179376 (optimal solution)
        # prune takes: 3 sec | popped from Q 201 | pruned 2139 (optimal solution)
        # probe takes: 1 sec | popped 48| restarts 3 (optimal solution)
        # probe (lin reg heu) takes: 1 sec | popped 56| restarts 3 (optimal solution)
    },
    "EXP11": {  # poc example - shift ATE ACS
        "df": df_acs_no_missing_values,
        "transformations_dict": small_data_transformations_no_fill,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 20376,
        "epsilon": 1,
        "max_length": 5
        # best sequence: (('bin_2', 'Public health coverage'), ('bin_2', 'gender'), ('bin_2', 'insurance through employer'), ('bin_2', 'medicare for people 65 and older'), ('bin_2', 'private health coverage'))
        # start ATE :8774
        # brute takes:  sec | popped  (optimal solution)
        # prune takes: 190 sec | popped from Q 175 | pruned 1798 (optimal solution)
        # probe takes: 48 sec | popped 29| restarts 4 (optimal)
        # probe (lin reg heu) takes: 43 sec | popped 24| restarts 3 (optimal)
    },
    "EXP12": {  # poc example - shift ATE IHDP
        "df": df_IHDP_no_missing_values,
        "transformations_dict": small_data_transformations_no_fill,
        "common_causes": ["x" + str(i) for i in range(1, 26)],
        "target_ate": 3.994,
        "epsilon": 0.0005,
        "max_length": 5
        # best sequence: (('bin_2', 'x6'), ('bin_2', 'x11'), ('bin_2', 'x22'))
        # start ATE : 3.92
        # brute takes: 158 sec | popped 1468700 (optimal solution)
        # prune takes: 252 sec | popped from Q 3205 | pruned  132235(optimal solution)
        # probe takes: 5 sec | popped 41| restarts 4 (NOT optimal solution len 5)
        # probe (lin reg heu) takes: 5 sec | popped 54| restart 5 (NOT optimal solution len 5)
    },
    ##################################################################################
    ###################SAME EXP WITH LARGE TRANSFORM DICT############################
    "EXP8_large": {  # run algorithms on the whole dataset, different target
        "df": df_twins_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.0019,  # 0,
        "epsilon": 0.000001,  # 0.0019,
        "max_length": 5
        # best sequence: (('zscore_clip_3', 'adequacy'), ('bin_2', 'lung'), ('bin_2', 'wt'))
        # start ATE : 0.06
        # brute takes: 16085 sec | popped 150936030 (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes: 130 sec | popped 61| restarts 2(optimal solution)
        # probe (lin reg heu) takes: 79 sec | popped 35| restarts 2(optimal solution)
    },
    "EXP9_large": {
        "df": df_twins_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_twins_no_missing_values.columns.difference(["treatment", "outcome"], sort=False).tolist(),
        "target_ate": -0.06,
        "epsilon": 0.06,
        "max_length": 5,
        "sequence_length": 2
        # prune sequence:  (('norm_log', 'gestat10'), ('bin_equal_frequency_2', 'wt')) | ATE: -0.0053
        # probe sequence:  (('norm_log', 'gestat10'), ('bin_equal_frequency_2', 'wt')) | ATE: -0.0053
        # random sequence: (('bin_equal_width_5', 'nprevistq'), ('bin_equal_width_2', 'othermr'))| ATE: 0.06
        # llm zero shot sequence: [{"column": "wt", "operation": "bin_equal_frequency_5"},{"column": "wt", "operation": "norm_log"},{"column": "gestat10", "operation": "bin_equal_width_5"},{"column": "gestat10", "operation": "winsorize"},{"column": "nprevistq", "operation": "bin_equal_frequency_5"},{"column": "nprevistq", "operation": "zscore_clip_3"}, {"column": "hydra", "operation": "bin_equal_width_2"},  {"column": "incervix", "operation": "bin_equal_width_2"},  {"column": "csex", "operation": "bin_equal_width_2"},{"column": "tobacco", "operation": "bin_equal_width_2"}]| ATE: 0.003
        # llm few shot sequence: [{"column": "wt","operation": "bin_equal_frequency_2"},{"column": "gestat10","operation": "bin_equal_frequency_5"},{"column": "nprevistq","operation": "bin_equal_width_2"},{"column": "hydra","operation": "bin_equal_frequency_2"},{"column": "incervix","operation": "bin_equal_frequency_2"}]| ATE:0.0126
        # llm cot sequence: [{"column": "wt", "operation": "bin_equal_frequency_5"},{"column": "gestat10", "operation": "bin_equal_frequency_5"},{"column": "nprevistq", "operation": "bin_equal_width_5"},{"column": "hydra", "operation": "norm_log"},{"column": "incervix", "operation": "norm_log"}]| ATE:0.009

    },
    "EXP10_large": {  # poc example - shift ATE LALONDE
        "df": df_lalonde_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_lalonde.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 1577,
        "epsilon": 1,
        "max_length": 10
        # best sequence: (('bin_equal_width_5', 'age'), ('bin_equal_frequency_2', 'black'), ('bin_equal_frequency_2', 'education'), ('bin_equal_frequency_2', 'hispanic'))
        # start ATE : 1671
        # brute takes: 2544 sec | popped 41179500 (optimal solution)
        # prune takes: 317 sec | popped from Q 8495 | pruned 460032 (optimal solution)
        # probe takes: 25 sec | popped 589| restarts 9 (len 6)
        # probe (lin reg heu) takes:  sec 36| popped 709| restarts 5(len 6)
    },
    "EXP11_large": {  # poc example - shift ATE ACS
        "df": df_acs_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 20376,
        "epsilon": 1,
        "max_length": 5
        # best sequence: (('bin_2', 'Public health coverage'), ('bin_2', 'gender'), ('bin_2', 'insurance through employer'), ('bin_2', 'medicare for people 65 and older'), ('bin_2', 'private health coverage'))
        # start ATE :
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes: 183 sec | popped 29| restarts 4 (optimal - 5)
        # probe (lin reg heu) takes:  sec | popped | restarts
    },
    "EXP12_large": {  # poc example - shift ATE IHDP
        "df": df_IHDP_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": ["x" + str(i) for i in range(1, 26)],
        "target_ate": 3.994,
        "epsilon": 0.0005,
        "max_length": 5
        # best sequence: (('bin_equal_width_10', 'x1'), ('norm_log', 'x6'), ('bin_equal_width_2', 'x6'))
        # start ATE : 3.92
        # brute takes: 3686 sec | popped  91647500(optimal solution)
        # prune takes: 4729 sec | popped from Q 22087| pruned 4801245(optimal solution)
        # probe takes: 15 sec | popped 40| restarts 4 (len 5)
        # probe (lin reg heu) takes: 30 sec | popped 132| restarts 6 (len 5)
    },

    ######################################################################
    ##################### large transformations ##########################
    "EXP13": {  # LALONDE, larger transforamtions | HELPER TO CHECK!
        "df": df_lalonde_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_lalonde.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0,  # 1900,
        "epsilon": 1300,  # 15,
        "max_length": 10
        # best sequence:   (('bin_equal_frequency_2', 'black'), ('bin_equal_width_2', 'education'), ('bin_equal_frequency_2', 'nodegree'))
        # start ATE : 1671.130
        # brute takes: 82 sec | popped 2013300 (optimal solution)
        # prune takes: 164 sec | popped from Q 4773 | pruned 239479(optimal solution)
        # probe takes: 3 sec | popped 107 | restarts 7(not optimal solution - 6)
        # probe (lin reg heu) takes:  sec | popped | restarts
    },
    "EXP13.5": {  # LALONDE, larger transforamtions
        "df": df_lalonde_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_lalonde.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 1871,  # wanted - 1883.3,
        "epsilon": 10,  # 1,
        "max_length": 10,
        'sequence_length': 4

        # prune sequence:  (('bin_equal_width_2', 'age'), ('bin_equal_frequency_2', 'black'), ('bin_equal_width_2', 'education'), ('bin_equal_frequency_2', 'nodegree')) | ATE:1870
        # probe sequence:  (('bin_equal_width_2', 'age'), ('bin_equal_frequency_2', 'black'), ('bin_equal_width_2', 'education'), ('bin_equal_frequency_2', 'nodegree'))| ATE:1870
        # random sequence: (('bin_equal_width_5', 'age'), ('norm_min_max', 'married'), ('norm_log', 'age'), ('zscore_clip_3', 'hispanic'))| ATE:1624
        # llm zero shot sequence: [{"column": "nodegree", "operation": "bin_equal_frequency_5"},{"column": "education", "operation": "bin_equal_frequency_5"},{"column": "age", "operation": "norm_log"},{"column": "black", "operation": "bin_equal_width_2"}]| ATE:  1623.59
        # llm few shot sequence:[{"column": "nodegree","operation": "norm_log"},{"column": "black","operation": "norm_log"},{"column": "education","operation": "norm_log"},{"column": "age","operation": "norm_min_max"}] | ATE: 1676.146
        # llm cot sequence: [{"column": "nodegree","operation": "bin_equal_frequency_2"},{"column": "hispanic","operation": "bin_equal_frequency_2"},{"column": "education","operation": "norm_min_max"},{"column": "age","operation": "bin_equal_width_5"}]| ATE:1667.79



    },
    "EXP14": {  # poc example - shift ATE ACS | HELPER TO CHECK!
        "df": df_acs_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0,
        "epsilon": 5000,
        "max_length": 10
        # best sequence:
        # start ATE : 8774
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes:  sec | popped | restarts  ()
        # probe (lin reg heu) takes:  sec | popped | restarts  ()
    },
    "EXP14.5": {  # poc example - shift ATE ACS
        "df": df_acs_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 16500,
        "epsilon": 100,
        "max_length": 10,
        "sequence_length": 4#6

        # prune sequence:    (('bin_equal_frequency_2', 'Age'), ('bin_equal_frequency_2', 'Public health coverage'), ('bin_equal_width_2', 'education'), ('bin_equal_frequency_2', 'medicare for people 65 and older'))| ATE: 16421
        # probe sequence:  (('bin_equal_width_2', 'education'), ('bin_equal_frequency_2', 'medicare for people 65 and older'), ('bin_equal_frequency_2', 'Public health coverage'), ('bin_equal_frequency_2', 'Age'))  | ATE:  16421
        # random sequence: (('bin_equal_width_2', 'gender'), ('bin_equal_frequency_2', 'medicare for people 65 and older'), ('bin_equal_frequency_5', 'Public health coverage'), ('bin_equal_width_10', 'insurance through employer'), ('bin_equal_width_5', 'education'), ('winsorize', 'Public health coverage'))| ATE:6540.8
        # llm zero shot sequence: [{"column": "Public health coverage", "operation": "bin_equal_frequency_10"},{"column": "medicare for people 65 and older", "operation": "bin_equal_frequency_10"},{"column": "Age", "operation": "bin_equal_frequency_10"},{"column": "insurance through employer", "operation": "bin_equal_frequency_5"},{"column": "private health coverage", "operation": "bin_equal_frequency_5"},{"column": "education", "operation": "bin_equal_frequency_10"}]| ATE: 6540.81
        # llm few shot sequence: [{"column": "Age","operation": "bin_equal_frequency_5"},{"column": "Public health coverage","operation": "bin_equal_frequency_2"},{"column": "medicare for people 65 and older","operation": "bin_equal_frequency_2"},{"column": "education","operation": "bin_equal_frequency_5"}]| ATE: 13535.11
        # llm cot sequence: [{"column": "Age", "operation": "bin_equal_frequency_5"},{"column": "Public health coverage", "operation": "bin_equal_frequency_2"},{"column": "medicare for people 65 and older", "operation": "bin_equal_frequency_2"},{"column": "insurance through employer", "operation": "bin_equal_frequency_2"},{"column": "education", "operation": "norm_min_max"}]| ATE:15692.30

    },
    "EXP14.5.1": {  # poc example - shift ATE ACS
        "df": df_acs_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 25000,
        "epsilon": 500,
        "max_length": 10
        # best sequence:
        # start ATE : 8774
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes: 482 sec | popped 118| restarts 4 (  len - 6)
        # probe (lin reg heu) takes: 12696 sec | popped 13843| restarts 4 (len 10)
    },
    "EXP15": {  # poc example - shift ATE IHDP | HELPER TO CHECK!
        "df": df_IHDP_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": ["x" + str(i) for i in range(1, 26)],
        "target_ate": -5,
        "epsilon": 8,
        "max_length": 10
        # best sequence:
        # start ATE : 3.92
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes:  sec | popped | restarts
        # probe (lin reg heu) takes:  sec | popped | restarts
    },
    "EXP15.5": {  # poc example - shift ATE IHDP
        "df": df_IHDP_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": ["x" + str(i) for i in range(1, 26)],
        "target_ate": 4.5,
        "epsilon": 0.5,
        "max_length": 10,
        "sequence_length": 3
        # prune sequence: (('norm_log', 'x6'), ('bin_equal_width_2', 'x6'), ('bin_equal_frequency_2', 'x11')) | ATE:4.001
        # probe sequence:   (('bin_equal_frequency_2', 'x9'), ('bin_equal_frequency_2', 'x2'), ('bin_equal_frequency_2', 'x1'), ('bin_equal_frequency_2', 'x11'), ('bin_equal_frequency_2', 'x5'), ('bin_equal_frequency_2', 'x6')) | ATE:4.008
        # random sequence: (('norm_min_max', 'x15'), ('norm_min_max', 'x21'), ('norm_log', 'x3'))| ATE: 3.92
        # llm zero shot sequence: [{"column": "x6", "operation": "bin_equal_frequency_5"},{"column": "x9", "operation": "bin_equal_width_5"},{"column": "x25", "operation": "norm_min_max"},{"column": "x15", "operation": "bin_equal_frequency_5"},{"column": "x17", "operation": "bin_equal_width_5"}]| ATE: 3.97
        # llm few shot sequence: [{"column": "x6","operation": "bin_equal_frequency_5"},{"column": "x9","operation": "bin_equal_frequency_5"},{"column": "x25","operation": "bin_equal_frequency_5"},{"column": "x17","operation": "bin_equal_frequency_5"}]| ATE:3.97
        # llm cot sequence: [{"column": "x6", "operation": "norm_min_max"},{"column": "x10", "operation": "bin_equal_frequency_2"},{"column": "x23", "operation": "bin_equal_frequency_2"},{"column": "x24", "operation": "bin_equal_frequency_2"},{"column": "x1", "operation": "norm_min_max"}]| ATE:3.9

    },
    "EXP16": {  # the result need to be with 3 operations | HELPER TO CHECK!
        "df": df_twins_no_missing_values,
        "transformations_dict": large_data_transformations,
        "common_causes": df_twins_no_missing_values.columns.difference(["treatment", "outcome"], sort=False).tolist(),
        "target_ate": 0.0022,  # -1,
        "epsilon": 0.0002,  # 1,
        "max_length": 10
        # best sequence:
        # start ATE : 0.06
        # brute takes:  sec | popped (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes:  sec | popped | restarts  ( solution)
        # probe (lin reg heu) takes:  sec | popped | restarts
    },

    ##########################################################################################
    ################################### SCALABILITY EVALUATIONS ##############################
    ##########################################################################################
    **{f"EXP17.{k}": {  # TWINS CHECK - k% of the data
        # RUN WITH NO SMALL\LARGE ATE PRINT!!!!
        "df": df_twins_no_missing_values.sample(frac=0.1 * k),
        "transformations_dict": large_data_transformations,
        "common_causes": df_twins_no_missing_values.columns.difference(["treatment", "outcome"], sort=False).tolist(),
        "target_ate": -0.06,
        "epsilon": 0.06,
        "max_length": 5
        # best sequence:
        # start ATE : 0.06
        # probe 10% takes: 42 sec | popped 51| restarts 0 ( len 1)
        # probe 20% takes: 48 sec | popped 52| restarts 1 ( len 2)
        # probe 30% takes: 191 sec | popped 195| restarts 1 ( len 2)
        # probe 40% takes: 60 sec | popped 51| restarts 0 ( len 1)
        # probe 50% takes: 65 sec | popped 51| restarts 0 ( len 1)
        # probe 60% takes: 247 sec | popped 197| restarts  1( len 2)
        # probe 70% takes: 271 sec | popped 200| restarts 1 ( len 2)
        # probe 80% takes: 535 sec | popped 413| restarts 2 ( len 3)
        # probe 90% takes: 295 sec | popped 200 | restarts  1( len 2)
    }
        for k in range(1, 10)
    },

    **{f"EXP18.{k}": {  # ACS CHECK - k% of the data
        # RUN WITH NO SMALL\LARGE ATE PRINT!!!!
        "df": df_acs_no_missing_values.sample(frac=0.1 * k),
        "transformations_dict": large_data_transformations,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 16500,
        "epsilon": 100,
        "max_length": 10
        # best sequence:
        # start ATE : 8774
        # probe 10% takes: 24 sec | popped 27| restarts 3 ( len 4)
        # probe 20% takes: 520 sec | popped 665| restarts 7 ( len 8)
        # probe 30% takes: 246 sec | popped 201| restarts 5 ( len 6)
        # probe 40% takes: 570 sec | popped 382| restarts 4 ( len 6)
        # probe 50% takes: 719 sec | popped 382| restarts 4 ( len 6)
        # probe 60% takes: 889 sec | popped 382| restarts 4 ( len 6)
        # probe 70% takes: 1015 sec | popped 382| restarts 4 ( len 6)
        # probe 80% takes: 1172 sec | popped 382| restarts 4 ( len 6)
        # probe 90% takes: 1478 sec | popped 382| restarts 4 ( len 6)
    }
        for k in range(1, 10)
    },

    **{f"EXP19.{k}": {  # TWINS CHECK - k random confunder
        # RUN WITH NO SMALL\LARGE ATE PRINT!!!!
        "transformations_dict": large_data_transformations,
        "common_causes": (cols := random.sample(df_twins_no_missing_values.columns.difference(["treatment", "outcome"], sort=False).tolist(), k=k)),
        "df": df_twins_no_missing_values[cols + ["treatment", "outcome"]],
        "target_ate": -0.06,
        "epsilon": 0.06,
        "max_length": 5
        # best sequence:
        # start ATE : 0.06
        # probe 3 confunders takes:  sec | popped | restarts  ( solution)
        # probe 6 confunders takes:  sec | popped | restarts  ( solution)
        # probe 9 confunders takes:  sec | popped | restarts  ( solution)
        # probe 12 confunders takes:  sec | popped | restarts  ( solution)
        # probe 15 confunders takes:  sec | popped | restarts  ( solution)
        # probe 18 confunders takes:  sec | popped | restarts  ( solution)
        # probe 21 confunders takes:  sec | popped | restarts  ( solution)
        # probe 24 confunders takes:  sec | popped | restarts  ( solution)
        # probe 27 confunders takes:  sec | popped | restarts  ( solution)
        # probe 30 confunders takes:  sec | popped | restarts  ( solution)
        # probe 33 confunders takes:  sec | popped | restarts  ( solution)
        # probe 36 confunders takes:  sec | popped | restarts  ( solution)
        # probe 39 confunders takes:  sec | popped | restarts  ( solution)
        # probe 42 confunders takes:  sec | popped | restarts  ( solution)
        # probe 45 confunders takes:  sec | popped | restarts  ( solution)
        # probe 48 confunders takes:  sec | popped | restarts  ( solution)
    } for k in range(3, len(df_twins.columns.difference(["treatment", "outcome"]).tolist()), 3)
    },
**{f"EXP20.{k}": {  # ACS CHECK - k random confunder
        # RUN WITH NO SMALL\LARGE ATE PRINT!!!!
        "transformations_dict": large_data_transformations,
        "common_causes": (cols := random.sample(df_acs.columns.difference(["treatment", "outcome"]).tolist(), k=k)),
        "df": df_acs_no_missing_values[cols + ["treatment", "outcome"]],
        "target_ate": 16500,
        "epsilon": 100,
        "max_length": 10
        # best sequence:
        # start ATE : 8774
        # probe 3 confunders takes: 199.24 sec | popped 216| restarts 3 ( DIDNT FIND SOLUTION)
        # probe 6 confunders takes: 694 sec | popped 252| restarts 3 ( len 5)
    } for k in range(3, len(df_acs.columns.difference(["treatment", "outcome"]).tolist()), 3)
    },
    **{f"EXP21.{k}": {  # TWINS CHECK - duplication of df
        # RUN WITH NO SMALL\LARGE ATE PRINT!!!!
        "df": pd.concat([df_twins_no_missing_values] * k, ignore_index=True),
        "transformations_dict": large_data_transformations,
        "common_causes": df_twins_no_missing_values.columns.difference(["treatment", "outcome"], sort=False).tolist(),
        "target_ate": -0.06,
        "epsilon": 0.06,
        "max_length": 5

    } for k in range(1,6)
    },
**{f"EXP22.{k}": {  # ACS CHECK - duplication of df
        # RUN WITH NO SMALL\LARGE ATE PRINT!!!!
        "df": pd.concat([df_acs_no_missing_values] * k, ignore_index=True),
        "transformations_dict": large_data_transformations,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 16500,
        "epsilon": 100,
        "max_length": 10

    } for k in range(1,6)
    },
}
