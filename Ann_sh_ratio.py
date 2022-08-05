import pandas as pd
import numpy as np

def annualized_sharpe_ratio(ruturn_series, riskfree_rate, periods_per_year):
    """
    Compute the annualized sharp ratio of a set of returns:
    1- First, convert the annual riskfree rate to per period.
    2- Second, use two other functions for annuallized returns and volatilities.
    3- We should infer the periods per year.
    4- Periods could be day, month or quarter in dataset.
    5- Adjust your data series before the calculation to look better.

    """
    def annualize_rets(ruturn_series, periods_per_year):
        """
        Annualizes a set of returns
        1- We should infer the periods per year, periods could be month, day or quarter
        2- Annual return is defined as the percentage change in an investment over a one-year period.
        3- Annualized return is the percentage change in an investment measured over periods longer than one year.
        """
        compounded_growth = (1 + ruturn_series).prod()
        n_periods = ruturn_series.shape[0]
        return compounded_growth ** (periods_per_year / n_periods) - 1
    
    def annualize_vol(return_series, periods_per_year):
        """
        Annualizes the volatilities of a set of returns:
        1- We should infer the periods per year.
        2- Periods could be day, month or quarter in dataset.
        3- Adjust your data series before the calculation to look better.
        """
        return return_series.std() * np.sqrt(periods_per_year)

    riskfree_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_return = ruturn_series - riskfree_per_period
    annualized_excess_return = annualize_rets(excess_return, periods_per_year)
    annualized_volotility = annualize_vol(ruturn_series, periods_per_year)
    return annualized_excess_return / annualized_volotility