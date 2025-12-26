import pandas as pd
import os
import dowhy.datasets


class TwinsDataLoader:
    def __init__(self):
        self.CACHE_FILE = "twins_data.pkl"

    def load_data(self) -> pd.DataFrame:
        # load from disk if exists
        if os.path.exists(self.CACHE_FILE):
            print("loaded cached data")
            return pd.read_pickle(self.CACHE_FILE)
        print("bring data")
        # The covariates data has 46 features
        x = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv")

        # The outcome data contains mortality of the lighter and heavier twin
        y = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv")

        # The treatment data contains weight in grams of both the twins
        t = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv")

        # _0 denotes features specific to the lighter twin and _1 denotes features specific to the heavier twin
        lighter_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                           'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                           'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                           'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                           'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                           'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                           'data_year', 'nprevistq', 'dfageq', 'feduc6', 'infant_id_0',
                           'dlivord_min', 'dtotord_min', 'bord_0',
                           'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']
        heavier_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                           'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                           'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                           'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                           'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                           'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                           'data_year', 'nprevistq', 'dfageq', 'feduc6',
                           'infant_id_1', 'dlivord_min', 'dtotord_min', 'bord_1',
                           'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']

        # Since data has pair property,processing the data to get separate row for each twin so that each child can be treated as an instance
        data = []

        for i in range(len(t.values)):

            # select only if both <=2kg
            if t.iloc[i].values[1] >= 2000 or t.iloc[i].values[2] >= 2000:
                continue

            this_instance_lighter = list(x.iloc[i][lighter_columns].values)
            this_instance_heavier = list(x.iloc[i][heavier_columns].values)

            # adding weight
            this_instance_lighter.append(t.iloc[i].values[1])
            this_instance_heavier.append(t.iloc[i].values[2])

            # adding treatment, is_heavier
            this_instance_lighter.append(0)
            this_instance_heavier.append(1)

            # adding the outcome
            this_instance_lighter.append(y.iloc[i].values[1])
            this_instance_heavier.append(y.iloc[i].values[2])
            data.append(this_instance_lighter)
            data.append(this_instance_heavier)

        cols = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                'data_year', 'nprevistq', 'dfageq', 'feduc6',
                'infant_id', 'dlivord_min', 'dtotord_min', 'bord',
                'brstate_reg', 'stoccfipb_reg', 'mplbir_reg', 'wt', 'treatment', 'outcome']

        df = pd.DataFrame(columns=cols, data=data)

        df.to_pickle(self.CACHE_FILE)
        return df


class LalondeDataLoader:
    def __init__(self):
        self.CACHE_FILE = "lalonde_data.pkl"

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_FILE):
            print("loaded cached data")
            return pd.read_pickle(self.CACHE_FILE)
        print("bring data")
        df = pd.read_stata("http://www.nber.org/~rdehejia/data/nsw_dw.dta")
        df = df.rename(columns={'treat': 'treatment', 're78': 'outcome'})
        df = df[['nodegree','black', 'hispanic', 'age', 'education', 'married','treatment', 'outcome']]
        df.to_pickle(self.CACHE_FILE)
        return df


class ACSDataLoader:
    def __init__(self):
        self.CACHE_FILE = "ACS_data.pkl"

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_FILE):
            print("loaded cached data")
            return pd.read_pickle(self.CACHE_FILE)
        print("bring data")
        df = pd.read_csv("acs.csv")
        df = df.rename(columns={'Educational attainment': 'education', 'Private health insurance coverage': 'private health coverage', 'Medicare, for people 65 and older, or people with certain disabilities': 'medicare for people 65 and older', 'Insurance through a current or former employer or union': 'insurance through employer', 'Sex': 'gender', 'With a disability': 'treatment', 'Wages or salary income past 12 months': 'outcome'})
        df = df[['education', 'Public health coverage', 'private health coverage', 'medicare for people 65 and older', 'insurance through employer', 'gender', 'Age', 'treatment', 'outcome']]
        df.to_pickle(self.CACHE_FILE)
        return df

