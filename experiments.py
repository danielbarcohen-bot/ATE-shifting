from data_loader import TwinsDataLoader, LalondeDataLoader, ACSDataLoader, IHDPDataLoader

df_twins = TwinsDataLoader().load_data()
df_lalonde = LalondeDataLoader().load_data()
df_acs = ACSDataLoader().load_data()
df_IHDP = IHDPDataLoader().load_data()

EXPERIMENTS = {
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
    "EXP3_new": {  # run algorithms on the whole dataset, different target
        "df": df_twins,
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
        # probe takes:  817 sec | popped 310 (len 3) | ATE 0.00026
    },
    "EXP7_new": {  # poc example - shift ATE LALONDE
        "df": df_lalonde,
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
    "EXP8_new": {  # poc example - shift ATE ACS
        "df": df_acs,
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
    "EXP9_new": {  # poc example - shift ATE IHDP
        "df": df_IHDP,
        "common_causes": ["x"+str(i) for  i in range(1,26)],
        "target_ate": 3.994,
        "epsilon": 0.0005,
        "max_length": 5
        # best sequence:   (('bin_2', 'x6'), ('bin_2', 'x11'), ('bin_2', 'x22'))
        # start ATE : 3.92
        # brute takes: 3126 sec | popped 23379800 (optimal solution)
        # prune takes: 280 sec | popped from Q 3205 | pruned 292435(optimal solution)
        # probe takes: 4 sec | popped 40 (NOT optimal solution - len 5)
    }

}
