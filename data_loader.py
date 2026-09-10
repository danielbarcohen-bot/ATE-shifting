import os
from pathlib import Path

import pandas as pd


class TwinsDataLoader:
    def __init__(self):
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        self.CACHE_FILE = os.path.join(loader_dir, "twins_data.pkl")
        self.categorical_cols = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mplbir', 'ormoth', 'mrace', 'orfath', 'frace', 'crace', 'birmon', 'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']
        self.col_types = {
            "csex": "Binary",
            "dmar": "Binary",
            "anemia": "Binary",
            "cardiac": "Binary",
            "lung": "Binary",
            "diabetes": "Binary",
            "herpes": "Binary",
            "hydra": "Binary",
            "hemo": "Binary",
            "chyper": "Binary",
            "phyper": "Binary",
            "eclamp": "Binary",
            "incervix": "Binary",
            "pre4000": "Binary",
            "preterm": "Binary",
            "renal": "Binary",
            "rh": "Binary",
            "uterine": "Binary",
            "othermr": "Binary",
            "tobacco": "Binary",
            "alcohol": "Binary",
            "data_year": "Numerical",
            "nprevistq": "Numerical",
            "dfageq": "Numerical",
            "dlivord_min": "Numerical",
            "dtotord_min": "Numerical",
            "bord": "Numerical",
            "mager8": "Ordinal",
            "meduc6": "Ordinal",
            "mpre5": "Ordinal",
            "adequacy": "Ordinal",
            "gestat10": "Ordinal",
            "cigar6": "Ordinal",
            "drink5": "Ordinal",
            "feduc6": "Ordinal",
            "pldel": "Categorical",
            "birattnd": "Categorical",
            "brstate": "Categorical",
            "stoccfipb": "Categorical",
            "mplbir": "Categorical",
            "ormoth": "Categorical",
            "mrace": "Categorical",
            "orfath": "Categorical",
            "frace": "Categorical",
            "crace": "Categorical",
            "birmon": "Categorical",
            "brstate_reg": "Categorical",
            "stoccfipb_reg": "Categorical",
            "mplbir_reg": "Categorical"
        }

    def load_data(self) -> pd.DataFrame:
        # load from disk if exists
        if os.path.exists(self.CACHE_FILE):
            print("loaded cached data")
            df = pd.read_pickle(self.CACHE_FILE)
            df.attrs['categorical_causes'] = self.categorical_cols
            df.attrs['col_types'] = self.col_types
            return df
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
            # if t.iloc[i].values[1] >= 2000 or t.iloc[i].values[2] >= 2000:
            #     continue

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
        df = df.drop(columns=['wt', 'infant_id'])
        # df.fillna(value=df.mean(), inplace=True)  # filling the missing values
        # df.fillna(value=df.mode().loc[0], inplace=True)
        df.to_pickle(self.CACHE_FILE)
        df.attrs['categorical_causes'] = self.categorical_cols
        df.attrs['col_types'] = self.col_types
        return df


class LalondeDataLoader:
    def __init__(self):
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        self.CACHE_FILE = os.path.join(loader_dir, "lalonde_data.pkl")
        self.col_types =  {
            "nodegree": "Binary",
            "black": "Binary",
            "hispanic": "Binary",
            "married": "Binary",
            "age": "Numerical",
            "education": "Numerical"
        }

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_FILE):
            print("loaded cached data")
            df = pd.read_pickle(self.CACHE_FILE)
            df.attrs['col_types'] = self.col_types
            return df
        print("bring data")
        df = pd.read_stata("http://www.nber.org/~rdehejia/data/nsw_dw.dta")
        df = df.rename(columns={'treat': 'treatment', 're78': 'outcome'})
        df = df[['nodegree', 'black', 'hispanic', 'age', 'education', 'married', 'treatment', 'outcome']]
        df.to_pickle(self.CACHE_FILE)
        df.attrs['col_types'] = self.col_types
        return df


class ACSDataLoader:
    def __init__(self):
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        self.CACHE_FILE = os.path.join(loader_dir, "ACS_data.pkl")
        self.col_types = {
            "Public health coverage": "Binary",
            "private health coverage": "Binary",
            "medicare for people 65 and older": "Binary",
            "insurance through employer": "Binary",
            "gender": "Binary",
            "Age": "Numerical",
            "education": "Numerical"
        }

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_FILE):
            print("loaded cached data")
            df = pd.read_pickle(self.CACHE_FILE)
            df.attrs['col_types'] = self.col_types
            return df
        print("bring data")
        df = pd.read_csv("acs.csv")
        df = df.rename(columns={'Educational attainment': 'education',
                                'Private health insurance coverage': 'private health coverage',
                                'Medicare, for people 65 and older, or people with certain disabilities': 'medicare for people 65 and older',
                                'Insurance through a current or former employer or union': 'insurance through employer',
                                'Sex': 'gender', 'With a disability': 'treatment',
                                'Wages or salary income past 12 months': 'outcome'})
        df = df[['education', 'Public health coverage', 'private health coverage', 'medicare for people 65 and older',
                 'insurance through employer', 'gender', 'Age', 'treatment', 'outcome']]
        df.to_pickle(self.CACHE_FILE)
        df.attrs['col_types'] = self.col_types
        return df


class IHDPDataLoader:
    def __init__(self):
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        self.CACHE_FILE = os.path.join(loader_dir, "IHDP_data.pkl")
        self.col_types = {
            "x1": "Numerical",
            "x2": "Numerical",
            "x3": "Numerical",
            "x4": "Numerical",
            "x5": "Numerical",
            "x6": "Numerical",
            "x7": "Binary",
            "x8": "Binary",
            "x9": "Binary",
            "x10": "Binary",
            "x11": "Binary",
            "x12": "Binary",
            "x13": "Binary",
            "x14": "Binary",
            "x15": "Binary",
            "x16": "Binary",
            "x17": "Binary",
            "x18": "Binary",
            "x19": "Binary",
            "x20": "Binary",
            "x21": "Binary",
            "x22": "Binary",
            "x23": "Binary",
            "x24": "Binary",
            "x25": "Binary",
            "x26": "Binary"
        }

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_FILE):
            print("loaded cached data")
            df = pd.read_pickle(self.CACHE_FILE)
            df.attrs['col_types'] = self.col_types
            return df
        print("bring data")
        df = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
            header=None)
        col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ]
        for i in range(1, 26):
            col.append("x" + str(i))
        df.columns = col
        df = df.rename(columns={'y_factual': 'outcome'})
        df = df[["x" + str(i) for i in range(1, 26)] + ["treatment", "outcome"]]
        df.to_pickle(self.CACHE_FILE)
        df.attrs['col_types'] = self.col_types
        return df


class WalmartDataLoader:
    def load_data(self) -> pd.DataFrame:
        col_types = {
            "waterfront": "Binary",
            "bedrooms": "Numerical",
            "bathrooms": "Numerical",
            "sqft_living": "Numerical",
            "sqft_lot": "Numerical",
            "floors": "Numerical",
            "sqft_above": "Numerical",
            "sqft_basement": "Numerical",
            "yr_built": "Numerical",
            "yr_renovated": "Numerical",
            "sqft_living15": "Numerical",
            "sqft_lot15": "Numerical",
            "view": "Ordinal",
            "condition": "Ordinal",
            "grade": "Ordinal"
        }

        BASE_DIR = Path(__file__).resolve().parent
        FILE_PATH = BASE_DIR / "house_price_vs_walmart_distances.csv"
        df = pd.read_csv(FILE_PATH)
        df.attrs['col_types'] = col_types
        return df
