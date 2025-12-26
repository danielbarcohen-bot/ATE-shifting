from data_loader import TwinsDataLoader, LalondeDataLoader, ACSDataLoader

df_twins = TwinsDataLoader().load_data()
df_lalonde = LalondeDataLoader().load_data()
df_acs = ACSDataLoader().load_data()

EXPERIMENTS = {
    "EXP1": {  # poc example - shift ATE
        "df": df_twins,
        "common_causes": ['wt', 'hydra', 'nprevistq', 'gestat10'],
        "target_ate": 0.005,
        "epsilon": 0.0001,
        "max_length": 5
        # best sequence: [('bin_2', 'wt'), ('bin_2', 'hydra'), ('bin_2', 'nprevistq'), ('zscore_clip_3', 'gestat10')]
        # start ATE : 0.064
        # brute takes: 407 sec | popped 298224 (optimal solution)
        # prune takes: < 5 sec | popped from Q 127 | pruned 1775 (optimal solution)
        # probe (pre prune & rearange & norm) takes: 81 sec | popped 2986 (optimal solution)
        # probe takes: 2 sec | popped 46 (optimal solution)
    },
    "EXP1_new": {  # twins - small common cause, no double bin\zscore AND df.DROPNA for check ATE
        "df": df_twins,
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
    "EXP2": {  # run algorithms on the whole dataset
        "df": df_twins,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.005,
        "epsilon": 0.0001,
        "max_length": 10
    },
    "EXP3": {  # run algorithms on the whole dataset, different target
        "df": df_twins,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.003,
        "epsilon": 0.0033,#0.501,
        "max_length": 10
        # best sequence: [('bin_2', 'wt'), ('zscore_clip_3', 'gestat10')]
        # start ATE: 0.0634
        # brute takes: 1594 sec | popped 3995136 (optimal solution)
        # prune takes: 1734 sec | popped from Q 3027 | pruned 505323 (optimal solution)
        # probe (pre prune & rearange & norm) takes: 85 sec | popped 1583 (optimal solution)
        # probe takes: 189 sec | popped 212 (optimal solution)
    },
    "EXP3_new": {  # run algorithms on the whole dataset, different target
        "df": df_twins,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.00,
        "epsilon": 0.0033,
        "max_length": 10
        # best sequence: [('bin_2', 'wt'), ('zscore_clip_3', 'gestat10')]
        # start ATE:  0.0613
        # brute takes: 542 sec | popped  3966264(optimal solution) | ATE is 0.000894
        # prune takes: 1774 sec | popped from Q 3007 | pruned  503792(optimal solution)
        # probe takes: 177 sec | popped 178 (not optimal len 3) | ATE is 0.00025
    },
    "EXP4": {  # get to less than zero result
        "df": df_twins,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": -0.5,
        "epsilon": 0.44,
        "max_length": 10
    },
    "EXP5": {  # the result need to be with 3 operations
        "df": df_twins,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.064756,
        "epsilon": 0.0000005,
        "max_length": 10
        # best sequence: [('bin_2', 'adequacy'), ('zscore_clip_3', 'cigar6'), ('bin_2', 'hydra')], ATE:  0.06475623
        # start ATE: 0.0634
        # brute takes: 15632 sec | popped 35770176 (optimal solution)
        # prune takes: 12058 sec | popped from Q 9728 | pruned 1672246 (optimal solution)
        # probe takes: 302 sec | popped 209 (NOT optimal solution - len 9)
    },
    "EXP5_new": {  # the result need to be with 3 operations
        "df": df_twins,
        "common_causes": df_twins.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.0,
        "epsilon": 0.0005,
        "max_length": 10
        # best sequence:
        # start ATE: 0.06132
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned  (optimal solution)
        # probe takes: 258 sec | popped 178 (NOT optimal solution - len 3) | ATE 0.0002523
    },
    "EXP7": {  # poc example - shift ATE LALONDE
        "df": df_lalonde,
        "common_causes": df_lalonde.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.0,
        "epsilon": 1000,
        "max_length": 10
        # best sequence:
        # start ATE :
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned 1775 ()
        # probe takes:  sec | popped  ()
    },
    "EXP7_new": {  # poc example - shift ATE LALONDE
        "df": df_lalonde,
        "common_causes": df_lalonde.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 1577,
        "epsilon": 1,
        "max_length": 10
        # best sequence:  [('zscore_clip_3', 'age'), ('bin_2', 'black'), ('zscore_clip_3', 'education'), ('bin_2', 'education'), ('bin_2', 'hispanic')]
        # start ATE : 1671.130
        # brute takes: 5132 sec | popped 21238805 (optimal solution)
        # prune takes: 3 sec | popped from Q 201 | pruned 4539 (optimal solution)
        # probe takes: < 1 sec | popped 48 (optimal solution)
    },
    "EXP8": {  # poc example - shift ATE ACS
        "df": df_acs,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.0,
        "epsilon": 1000,
        "max_length": 5
        # best sequence:
        # start ATE :
        # brute takes:  sec | popped  (optimal solution)
        # prune takes:  sec | popped from Q  | pruned ()
        # probe takes:  sec | popped  ()
    },
    "EXP8_new": {  # poc example - shift ATE ACS
        "df": df_acs,
        "common_causes": df_acs.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 20376,
        "epsilon": 1,
        "max_length": 5
        # best sequence:  [('bin_2', 'Public health coverage'), ('bin_2', 'gender'), ('bin_2', 'insurance through employer'), ('bin_2', 'medicare for people 65 and older'), ('bin_2', 'private health coverage')]
        # start ATE : 8774.433205040823
        # brute takes:  sec | popped  (optimal solution)
        # prune takes: 422 sec | popped from Q 175 | pruned 3772 (optimal solution)
        # probe takes: 83.48 sec | popped 27 (optimal solution)
    }

}
