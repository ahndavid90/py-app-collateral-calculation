# %% [markdown]
# # <center> SOA Session 3A - Storytelling with Data </center>

# %% [markdown]
# ## Collateral Monitoring Tool v1.01

# %% [markdown]
# <u>Developer:</u> David Ahn
# 
# <u>Date:</u> 5/24/2022
# 
# <u>Purpose:</u> To monitor daily collateral amount by calculating present value of future projected cashflows
# 
# <u>Summary:</u> 
# 
# Collateral amount is the future obligated cashflows the company needs to hold in the bank account to ensure solvency for company's longevity risk. 
# 
# The tool will calculate PVs of Best Estimate, Best Scenario, and Worst Scenario 
# - Best Scenario: More people are dying than expected. 
# - Worst Scenario: Less people are dying than expected. 
# 
# <u>Assumptions</u>:
# - CF = 100,000
# - annual_qx = 0.03 
# - base_mortality_shock = 50%
# 
# <u>Valuation Discount Factor</u>: Simple US Treasury Curve
# 
# <u>Procedure:</u>
# 1. Import python libraries
# 2. Import US treasury yield curve
# 3. Calculate valuation discount factor (vdf) using linear interpolation
# 4. Calculate survivorship factor for BE, Best, and Worst Scenarios
# 5. Calculate PVs using survivorship and vdf
# 6. Results Summary
# 7. Export output
# 
# <u>Resources:</u>
# https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield
# 
# 
# 
# 

# %% [markdown]
# ## 1. Library Import
# 
# The following libraries are used throughout the tool
# 
# - <u>numpy</u> - Used for numerical calculations working with arrays
# - <u>pandas</u> - Working with tabular data with rows and columns as dataframes to process data
# - <u>matplotlib.pyplot</u> - Create and work with data visualization
# - <u>datetime</u> - Convert date format
# - <u>scipy</u> - Used for linear interpolation
# 
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.interpolate

# %% [markdown]
# ## 2. Yield Curve
# 
# Import daily US treasury yield curve rates 

# %%
# Create dataframe from .csv file from the most recent valuation date
treasury_curve = pd.read_csv('daily-treasury-rates.csv', index_col=0, nrows=1)
treasury_curve_final = treasury_curve.transpose().reset_index()

# Generate valuation date
val_date = datetime.strptime(treasury_curve_final.columns[1], '%m/%d/%Y').strftime('%m-%d-%Y')

# Set up columns
columns = ['Tenors', 'Rate']
treasury_curve_final.columns = columns

# Print results
print(f'Valuation Date: {val_date}')
treasury_curve_final.plot(x='Tenors', y='Rate')
treasury_curve_final


# %% [markdown]
# ## 3. Valuation Discount Factor (VDF)
# 
# #### Calculate Valuation Discount Factor from the yield curve and applying linear interpolation method

# %% [markdown]
# Calculate Maturities on annual basis, which will be used in the linear interpolation

# %%
treasury_curve_final['Maturities'] = np.where(
    treasury_curve_final['Tenors'].str[-2:]=='Mo', 
    treasury_curve_final['Tenors'].str[:1].astype(int)/12,
    treasury_curve_final['Tenors'].str[:2].astype(int)
)

treasury_curve_final

# %% [markdown]
# Cashflows will project out for 30 years (360 months). Map the 30 year projection with Rates and Maturies from the yield curve above

# %%
# Create a new dataframe with months and years
df = pd.DataFrame({'Months': range(1,361)})
df['Year'] = df['Months']/12

# Map treasury yield curve
df_final = df.merge(treasury_curve_final, left_on='Year', right_on='Maturities', how='left').fillna(method='ffill')

df_final


# %% [markdown]
# ## <center>Linear Interpolation</center>
# 
# 

# %% [markdown]
# Linear interpolation function is defined as follows:
# 
# $$ y(x)  =  y_1  +  (x – x_1)  \frac{(y_2 – y_1) }{ (x_2 – x_1)} $$

# %% [markdown]
# Use scipy package to linearly interpolate the rates across 360 months

# %%
# Linear Interpolation
y_interp = scipy.interpolate.interp1d(df_final['Maturities'], df_final['Rate'])
df_final['Rate_Interpolated'] = y_interp(df_final['Year'])

df_final

# %% [markdown]
# <u>David's Comment:</u> This is much easier and cleaner than writing linear interpolation formula from scratch! As we've discussed, python is open-source where we can use libraries developed by 3rd party, where it has been fully tested and documented

# %% [markdown]
# Calculate Valuation Discount Factor (VDF)

# %%
df_final.drop(['Tenors', 'Rate', 'Maturities'], axis=1, inplace=True)

df_final['VDF'] = (1 + df_final['Rate_Interpolated'] / 100) ** (-df_final['Year'])

df_final

# %% [markdown]
# ## Assumptions and Inputs

# %%
CF = 100000
annual_qx = 0.03
monthly_qx = annual_qx/12
base_mortality_shock = 0.5

# %% [markdown]
# ## 4. Cumulative Survivorship using q(x)

# %% [markdown]
# Calculate cumulative survivorship for BE, Best, and Worst based on the 50% mortality shocks

# %%
df_final['CFs'] = CF
df_final['Survivorship'] = (1-monthly_qx)
df_final['Survivorship_Best'] = (1-monthly_qx*(1+base_mortality_shock))
df_final['Survivorship_Worst'] = (1-monthly_qx/(1+base_mortality_shock))

df_final['Cumulative_Survivorship'] = df_final['Survivorship'].cumprod()
df_final['Cumulative_Survivorship_Best'] = df_final['Survivorship_Best'].cumprod()
df_final['Cumulative_Survivorship_Worst'] = df_final['Survivorship_Worst'].cumprod()

df_final


# %% [markdown]
# ## 5. PVs

# %% [markdown]
# Calculate PV for BE, Best, and Worst scenarios

# %%
df_final['Decremented_CFs'] = CF * df_final['Cumulative_Survivorship'] 
df_final['Decremented_CFs_Best'] = CF * df_final['Cumulative_Survivorship_Best'] 
df_final['Decremented_CFs_Worst'] = CF * df_final['Cumulative_Survivorship_Worst'] 

df_final['PV'] = df_final['VDF'] * df_final['Decremented_CFs']
df_final['PV_Best'] = df_final['VDF'] * df_final['Decremented_CFs_Best']
df_final['PV_Worst'] = df_final['VDF'] * df_final['Decremented_CFs_Worst']

df_final

# %% [markdown]
# ## 6. Results Summary

# %%
df_final.plot(kind='line',x='Months',y=['PV', 'PV_Best', 'PV_Worst'])

plt.show()

# %%
total = pd.Series({'PV': df_final.PV.sum(), 'PV_Best': df_final.PV_Best.sum(), 'PV_Worst': df_final.PV_Worst.sum()})

total.name = 'Total'

df_final_total = df_final.append(total).fillna('')

print(f'PV: {df_final.PV.sum()}')
print(f'PV (Best Scenario): {df_final.PV_Best.sum()}')
print(f'PV (Worst Scenario): {df_final.PV_Worst.sum()}')

# %% [markdown]
# ## 7. Export the Output
# 
# 

# %%
df_final_total.to_csv(f'collateral_validation_{val_date}.csv')

# %%



