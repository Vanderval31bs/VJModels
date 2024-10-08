import numpy as np
import pandas as pd

import statsmodels.api as sm
from statstests.process import stepwise
from statstests.tests import shapiro_francia
from statstests.datasets import bebes
from scipy.stats import shapiro, boxcox, chi2


class AdvancedLinearRegression:
    def __init__(self, alpha=0.05, verbose=False):
        self.alpha = alpha
        self.verbose = verbose

        self.fitted = False

        self.df = None
        self.categorical_columns = None
        self.base_model = None
        self.stepwise_model = None
        self.lmbda = None
        self.heteroscedasticity = None
        self.breusch_pagan_p_value = None
        
    def log(self, *args):
        if self.verbose:
            print(*args)
            

    def fit_stepwise_model(self, df: pd.DataFrame, target_label: str):
        columns = list(df.drop(columns=[target_label]).columns)
        formula = ' + '.join(columns)
        formula = target_label + " ~ " + formula

        self.base_model = sm.OLS.from_formula(formula, df).fit()
        self.stepwise_model = stepwise(self.base_model, pvalue_limit=self.alpha)

        self.log("Fitted base model (with all predictor variables) and a model using the stepwise selection.")

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

    def is_resid_distribution_normal(self):
        p_value = self.shapiro['p-value']

        if p_value > self.alpha:
            self.log('Não se rejeita H0: Distribuição dos resíduos é aderente à normalidade')
            return True
        else:
            self.log('Rejeita-se H0: Distribuição dos resíduos não é aderente à normalidade')
            return False

    def apply_box_cox(self, df: pd.DataFrame, target_label: str):
        yast, self.lmbda = boxcox(df[target_label])  
        df[target_label] = yast
        self.log(f"Applied Box-Cox transformation with lambda = {self.lmbda:.5f}")

        self.fit_stepwise_model(df, target_label)
        self.log("Fitted stepwise model with transformed Box-Cox target variable")

    def inverse_box_cox_transform(self, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred * self.lmbda + 1) ** (1 / self.lmbda)    

    def breusch_pagan_test(self):
        residuals = self.stepwise_model.resid
        fitted_values = self.stepwise_model.fittedvalues

        data = pd.DataFrame({
            'fitted': fitted_values,
            'squared_residuals': np.square(residuals)
        })

        data['scaled_residuals'] = data['squared_residuals'] / np.mean(data['squared_residuals'])
        auxiliary_model = sm.OLS.from_formula('scaled_residuals ~ fitted', data).fit()

        anova_results = sm.stats.anova_lm(auxiliary_model, typ=2)
        anova_results['sum_sq'] /= 2

        chi2_stat = anova_results['sum_sq'].iloc[0]
        p_value = chi2.sf(chi2_stat, 1) * 2

        self.breusch_pagan_p_value = p_value
        self.log(f"Breusch-Pagan p-value: {self.breusch_pagan_p_value}")
    
    def has_heteroscedasticity(self):        
        if self.breusch_pagan_p_value > self.alpha:
            self.log('Não se rejeita H0: Ausência de Heterocedasticidade')
            return False
        else:
            self.log('Rejeita-se H0: Presença de Heterocedasticidade')
            return True

    def fit(self, df: pd.DataFrame, target_label: str):

        if df[target_label].dtype not in ['int', 'float']:
            raise ValueError("Target label must be a numerical column.")
            # In the future, implement a more general model that does logistic regression here

        self.df = df
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        df_dummies = pd.get_dummies(df, columns=self.categorical_columns, dtype=int, drop_first=True)

        self.fit_stepwise_model(df_dummies, target_label)

        n = len(df_dummies)
        if (n >= 30):
            self.shapiro_francia_test()
        else:
            self.shapiro_wilk_test()

        if (not self.is_resid_distribution_normal()):
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
    
    def clean_column_name(self, name: str) -> str:
        # Remove Q('...') to get just the column name
        if name.startswith("Q('") and name.endswith("')"):
            return name[3:-2]
        return name

    def summary(self) -> dict:
        if not self.fitted:
            raise ValueError("Model is not fitted yet.")
        
        # Step 1: Transform categorical variables
        if len(self.categorical_columns) == 0:
            step1 = "**STEP 1**\nNo categorical variables to transform."
        elif len(self.categorical_columns) == 1:
            categories = f"'{self.categorical_columns[0]}'"
            step1 = f"**STEP 1**\nTransformed category {categories} into dummy variables."
        else:
            categories = "', '".join(self.categorical_columns)
            step1 = f"**STEP 1**\nTransformed categories '{categories}' into dummy variables."

        # Step 2: Fit stepwise model
        step2 = "**STEP 2**\nFitted the first model using the stepwise technique."

        # Step 3: Perform normality test based on sample size
        n = len(self.df)
        test_type = "Shapiro-Francia test" if n >= 30 else "Shapiro-Wilk test"
        step3 = f"**STEP 3**\nAs the length of the dataframe is {n}, {test_type} was performed."
        step3 += f"\nThe p-value obtained is {self.shapiro['p-value']:.5f}."
        resid_dist_text = "" if self.is_resid_distribution_normal() else "NOT "
        step3 += f"\nThis means that the residuals are {resid_dist_text}normally distributed."

        # Step 4: Box-Cox transformation (if applied)
        step4 = ""
        if self.lmbda is not None:
            step4 = f"**STEP 4**\nApplied Box-Cox transformation with lambda = {self.lmbda:.5f}."
            step4 += "\nFitted stepwise model with transformed target variable."

        # Step 5: Perform Breusch-Pagan test for heteroscedasticity
        heteroscedasticity_text = "" if self.heteroscedasticity else "NOT "
        step5 = "Performed Breusch-Pagan test for heteroscedasticity."
        step5 += f"\nThe p-value obtained is {self.breusch_pagan_p_value:.5f}."
        step5 += f"\nThis means that the residuals are {heteroscedasticity_text}heteroscedastic."

        # Conclusion: Format parameters and p-values in a table-like structure
        params_table = "\n".join([f"{self.clean_column_name(key)}: {value:.5f}" for key, value in self.stepwise_model.params.items()])
        pvalues_table = "\n".join([f"{self.clean_column_name(key)}: {value:.5e}" for key, value in self.stepwise_model.pvalues.items()])

        if self.heteroscedasticity:
            conclusion = "The model has heteroscedasticity. Probably, there are relevant features for predicting the target variable that were omitted from the model."
        else:
            conclusion = "The final model parameters are:\n"
            conclusion += params_table
            conclusion += f"\n\nThe final model R² is: {self.stepwise_model.rsquared:.5f}."
            conclusion += f"\nThe final model adjusted R² is: {self.stepwise_model.rsquared_adj:.5f}."
            conclusion += f"\nThe final model F statistic p-value is: {self.stepwise_model.f_pvalue:.5f}."
            conclusion += "\n\nThe final model params p-values are:\n"
            conclusion += pvalues_table

        # Combine all steps into a summary
        summary_steps = [step1, step2, step3, step4, f"**STEP 4**\n{step5}" if step4 == "" else f"**STEP 5**\n{step5}"]
        summary = "\n\n".join([step for step in summary_steps if step]) + f"\n\n**CONCLUSION**:\n{conclusion}"

        return summary

    

if __name__ == "__main__":
    df = bebes.get_data()
    target = "comprimento"
    model = AdvancedLinearRegression(verbose=False)
    model.fit(df, target)


    print("-----------------------------------------\n\n")
    print(model.summary())
    print("-----------------------------------------\n\n")

    predictions = model.predict(df[['idade']])
    actual = df[target]
    idade = df['idade']

    result_table = pd.DataFrame({
        'Idade': idade,
        'Valor Real': actual,
        'Previsão': predictions
    })

    print(result_table.head(5))
