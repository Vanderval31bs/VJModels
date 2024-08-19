import numpy as np
import pandas as pd

import statsmodels.api as sm
from statstests.process import stepwise
from statstests.tests import shapiro_francia
from scipy.stats import shapiro, boxcox, chi2


class AdvancedLinearRegression:
    def __init__(self, alpha=0.05, verbose=False):
        self.alpha = alpha
        self.fitted = False
        self.verbose = verbose
        self.base_model = None
        self.stepwise_model = None
        self.lmbda = None
        self.heteroscedasticity = None
        
    def log(self, *args):
        if self.verbose:
            print(*args)
            

    def fit_stepwise_model(self, df: pd.DataFrame, target_label: str):
        columns = list(df.drop(columns=[target_label]).columns)
        formula = ' + '.join(columns)
        formula = target_label + " ~ " + formula

        self.base_model = sm.OLS.from_formula(formula, df).fit()
        self.stepwise_model = stepwise(self.base_model, pvalue_limit=self.alpha)

        self.log("Fitted stepwise and base model")

    def shapiro_francia_test(self):
        self.shapiro = shapiro_francia(self.stepwise_model.resid)
        self.log("Performed Shapiro-Francia test: ", self.shapiro)

    def shapiro_wilk_test(self):
        stat, p_value = shapiro(self.stepwise_model.resid)
        self.shapiro = {
            'method': 'Shapiro-Wilk normality test',
            'W': stat,
            'p-value': p_value
        }
        self.log("Performed Shapiro-Wilk test: ", self.shapiro)

    def is_distribution_normal(self):
        p_value = self.shapiro['p-value']

        if p_value > self.alpha:
            self.log('Não se rejeita H0 - Distribuição aderente à normalidade')
            return True
        else:
            self.log('Rejeita-se H0 - Distribuição não aderente à normalidade')
            return False

    def apply_box_cox(self, df: pd.DataFrame, target_label: str):
        yast, self.lmbda = boxcox(df[target_label])  
        df[target_label] = yast
        self.fit_stepwise_model(df, target_label)

    def inverse_box_cox_transform(self, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred * self.lmbda + 1) ** (1 / self.lmbda)

    def breusch_pagan_test(self):
        df = pd.DataFrame({'yhat':self.stepwise_model.fittedvalues,
                        'resid':self.stepwise_model.resid})
    
        df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
    
        modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
        anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
        anova_table['sum_sq'] = anova_table['sum_sq']/2
        chisq = anova_table['sum_sq'].iloc[0]
        p_value = chi2.pdf(chisq, 1)*2
        
        self.log(f"chisq: {chisq}")        
        self.log(f"p-value: {p_value}")
        
        return chisq, p_value
    
    def has_heteroscedasticity(self):
        _, p_value = self.breusch_pagan_test()
        
        if p_value > self.alpha:
            self.log('Não se rejeita H0 - Ausência de Heterocedasticidade')
            return False
        else:
            self.log('Rejeita-se H0 - Existência de Heterocedasticidade')
            return True

    def fit(self, df: pd.DataFrame, target_label: str):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        df_dummies = pd.get_dummies(df, columns=categorical_columns, dtype=int, drop_first=True)

        self.fit_stepwise_model(df_dummies, target_label)

        n = len(df)
        if (n >= 30):
            self.shapiro_francia_test()
        else:
            self.shapiro_wilk_test()

        if (not self.is_distribution_normal()):
            self.apply_box_cox(df_dummies, target_label)

        self.breusch_pagan_test()
        self.heteroscedasticity = self.has_heteroscedasticity()

        self.fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model is not fitted yet.")
        
        if (self.lmbda is not None):
            y_pred = self.stepwise_model.predict(X)
            return self.inverse_box_cox_transform(y_pred)
        
        return self.stepwise_model.predict(X)

    def summary(self) -> dict:
        if not self.fitted:
            raise ValueError("Model is not fitted yet.")
        
        coefficients = self.stepwise_model.params.to_dict()
        intercept = self.stepwise_model.params.get('Intercept', 0)

        r_squared = self.stepwise_model.rsquared
        adj_r_squared = self.stepwise_model.rsquared_adj
        p_values = self.stepwise_model.pvalues.to_dict()

        summary = {
            "coefficients": coefficients,
            "intercept": intercept,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "p_values": p_values,
            "boxcox_applied": self.lmbda is not None,
            "heteroscedasticity": self.heteroscedasticity,
        }
        
        return summary
    

if __name__ == "__main__":
    df = pd.read_csv('planosaude.csv')
    df.drop(columns=['id'], inplace=True)
    model = AdvancedLinearRegression(verbose=False)
    model.fit(df, 'despmed')
    print(model.summary())
