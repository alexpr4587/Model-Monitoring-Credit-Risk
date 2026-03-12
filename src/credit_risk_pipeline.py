import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X.columns = X.columns.str.strip().str.lower().str.replace(' ', '_')
        return X

class CategoricalCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].str.lower().str.replace(' ', '_').str.replace('-', '_').str.lstrip('_')
        return X

class EmpLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mode'):
        self.strategy = strategy
        self.fill_value = None
    def fit(self, X, y=None):
        if self.strategy == 'mode':
            temp_col = pd.Series(X).str.extract('(\\d+)').astype(float)
            self.fill_value = temp_col.mode()[0] if not temp_col.mode().empty else 0
        else: self.fill_value = 0
        return self
    def transform(self, X):
        s = pd.Series(X).copy()
        s = s.replace('< 1 year', '0').str.extract('(\\d+)', expand=False).astype(float)
        return s.fillna(self.fill_value).values.reshape(-1, 1)

class TermTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        s = pd.Series(X.iloc[:, 0]) if isinstance(X, pd.DataFrame) else pd.Series(X)
        return s.str.extract('(\\d+)', expand=False).astype(float).values.reshape(-1, 1)

class LoanStatusCleaner(BaseEstimator, TransformerMixin):
    def __init__(self): self.target_map = {'fully_paid': 0, 'charged_off': 1}
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy[X_copy['loan_status'].isin(self.target_map.keys())]
        X_copy['is_default'] = X_copy['loan_status'].map(self.target_map)
        return X_copy

class GradeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        grades = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        self.mapping = {f"{g}{i}": (count - 1) * 5 + i for count, g in enumerate(grades, 1) for i in range(1, 6)}
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'sub_grade' in X.columns:
            X['sub_grade_num'] = X['sub_grade'].str.lower().map(self.mapping).fillna(0)
            X = X.drop(columns=[c for c in ['grade', 'sub_grade'] if c in X.columns])
        return X

class IssueDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='issue_d', create_year=True, create_quarter=True, drop_original=False):
        self.column, self.create_year, self.create_quarter, self.drop_original = column, create_year, create_quarter, drop_original
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X[self.column] = pd.to_datetime(X[self.column], format='%b_%Y', errors='coerce')
        if self.create_year: X[f'{self.column}_year'] = X[self.column].dt.year
        if self.create_quarter: X[f'{self.column}_quarter'] = X[self.column].dt.to_period('Q')
        if self.drop_original: X = X.drop(columns=[self.column])
        return X

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper
        self.lower_bounds_, self.upper_bounds_ = {}, {}
    def fit(self, X, y=None):
        X_ = pd.DataFrame(X)
        for col in X_.columns:
            self.lower_bounds_[col] = X_[col].quantile(self.lower)
            self.upper_bounds_[col] = X_[col].quantile(self.upper)
        return self
    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        for col in X_.columns:
            X_[col] = X_[col].clip(lower=self.lower_bounds_.get(col), upper=self.upper_bounds_.get(col))
        return X_.values

class WoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=0.5):
        self.smoothing = smoothing
        self.woe_maps_ = {}
    def fit(self, X, y):
        X_, y_ = pd.DataFrame(X), pd.Series(y).values
        total_events, total_non_events = y_.sum(), len(y_) - y_.sum()
        for col in X_.columns:
            stats = pd.DataFrame({'cat': X_[col].values, 'target': y_}).groupby('cat')['target'].agg(['sum', 'count'])
            stats['non_events'] = stats['count'] - stats['sum']
            dist_e = (stats['sum'] + self.smoothing) / (total_events + self.smoothing)
            dist_ne = (stats['non_events'] + self.smoothing) / (total_non_events + self.smoothing)
            self.woe_maps_[col] = np.log(dist_ne / (dist_e + 1e-6)).to_dict()
        return self
    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        for col in X_.columns: X_[col] = X_[col].map(self.woe_maps_[col]).fillna(0)
        return X_.values

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop): self.columns_to_drop = columns_to_drop
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=[c for c in self.columns_to_drop if c in X.columns])

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, create_log_income=True): self.create_log_income = create_log_income
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['loan_to_income'] = X['funded_amnt'] / (X['annual_inc'] + 1)
        X['payment_burden'] = (X['installment'] * 12) / (X['annual_inc'] + 1)
        if self.create_log_income: X['log_annual_inc'] = np.log1p(X['annual_inc'])
        return X


class PDPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_preprocessor, woe_encoder, numeric_features, categorical_features):
        self.numeric_preprocessor = numeric_preprocessor
        self.woe_encoder = woe_encoder
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_num = self.numeric_preprocessor.transform(X)
        X_cat = self.woe_encoder.transform(X[self.categorical_features])
        return np.hstack([X_num, X_cat])

class PDModel:
    def __init__(self, model, threshold):
        self.model, self.threshold = model, threshold
    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]
    def predict(self, X): return (self.predict_proba(X) >= self.threshold).astype(int)



def calculate_el(pd_score: np.ndarray, lgd_hat: np.ndarray, ead_hat: np.ndarray) -> np.ndarray:

    return pd_score * lgd_hat * ead_hat

def calculate_el_ratio(el: np.ndarray, funded_amnt: np.ndarray) -> np.ndarray:

    return el / (funded_amnt + 1e-9)

def calculate_income(funded_amnt: np.ndarray, int_rate: np.ndarray, term: np.ndarray) -> np.ndarray:

    return (int_rate / 100) * funded_amnt * (term / 12)

def calculate_ep(pd_score: np.ndarray, lgd_hat: np.ndarray, ead_hat: np.ndarray, 
                 funded_amnt: np.ndarray, int_rate: np.ndarray, term: np.ndarray) -> np.ndarray:

    income = calculate_income(funded_amnt, int_rate, term)
    el = calculate_el(pd_score, lgd_hat, ead_hat)
    return (1 - pd_score) * income - el

def calculate_ep_ratio(ep: np.ndarray, funded_amnt: np.ndarray) -> np.ndarray:

    return ep / (funded_amnt + 1e-9)


class ApprovalModel:

    def __init__(self, hurdle_rate: float = 0.0):
        self.hurdle_rate = hurdle_rate

    def ep(self, pd_score, lgd_hat, ead_hat, funded_amnt, int_rate, term):
        return calculate_ep(pd_score, lgd_hat, ead_hat, funded_amnt, int_rate, term)

    def ep_ratio(self, pd_score, lgd_hat, ead_hat, funded_amnt, int_rate, term):
        ep_val = self.ep(pd_score, lgd_hat, ead_hat, funded_amnt, int_rate, term)
        return calculate_ep_ratio(ep_val, funded_amnt)

    def pd_threshold(self, funded_amnt, int_rate, lgd_hat, ead_hat, term):

        income = calculate_income(funded_amnt, int_rate, term)
        h_abs  = self.hurdle_rate * funded_amnt
        
        # Ecuación de corte: (Income - Hurdle_Abs) / (Income + LGD * EAD)
        # Se restringe entre 0 y 1 para evitar valores matemáticos fuera de probabilidad lógica.
        pd_max = (income - h_abs) / (income + lgd_hat * ead_hat + 1e-9)
        return np.clip(pd_max, 0, 1)

    def approve(self, pd_score, lgd_hat, ead_hat, funded_amnt, int_rate, term):
        ep_r = self.ep_ratio(pd_score, lgd_hat, ead_hat, funded_amnt, int_rate, term)
        return (ep_r > self.hurdle_rate).astype(int)

    def __repr__(self):
        return f'ApprovalModel(hurdle_rate={self.hurdle_rate:.4f})'