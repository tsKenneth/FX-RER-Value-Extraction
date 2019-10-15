"""
Extracting value premium from G10 Currencies using monthly real exchange rates
"""

# Import statements
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

"""
===============================================================================
Configurations
===============================================================================
"""

# Library Configs - Set how many rows and columns to display when using
# dataframe.display()
pd.options.display.max_columns = None
pd.options.display.max_rows = None

plt.style.use('seaborn')

# Data dictionaries for easy modification of RIC definitions
INTERVAL_DICT = {
	'weekly': 'P1W',
	'daily': 'P1D',
	'monthly': 'P1M',
	'quarterly': 'P3M'
}

TENOR_DICT = {
	'spotweek': 'SW',
	'1_month': '1M',
	'2_month': '2M',
	'3_month': '3M'
}

# Scope of analysis
DATA_START_DATE = '2016-11-01'
DATA_END_DATE = '2018-05-01'

"""
===============================================================================
RER Calculation
===============================================================================
"""


def calculate_RER(df_currency_spot, df_cpi, column_name_order):
	"""

	Calculate the real exchange rate using the spot rate and the countries' CPI.
	Ensure that the order in which the columns are arranged are for both dataframes
	to be the same as the order specified in column_name_order.

	:param df_currency_spot: dataframe contain spot prices for G10 currency
	:type df_currency_spot: Pandas dataframe
	:param df_cpi: dataframe containing the CPI% for G10 countries
	:type df_cpi: Pandas dataframe
	:param column_name_order: the ordered column names for the currency spot dataframe and CPI dataframe
	:return: Pandas dataframe containing RER for each G10 currency
	"""

	# Import monthly currency spot rates
	df_currency_spot_rer = df_currency_spot.copy()

	# Sort date from earliest to latest
	df_currency_spot_rer = df_currency_spot_rer.sort_index()

	# Get CPI List
	df_cpi_rer = df_cpi.copy()

	# Create CPI Levels
	df_cpi_levels = df_currency_spot_rer.copy()

	# Rename columns
	df_cpi_rer.columns = column_name_order
	df_cpi_levels.columns = column_name_order
	df_currency_spot_rer.columns = column_name_order

	# Set base level at reciprocal of currency spot at reference month
	df_cpi_levels.iloc[0,] = 1 / df_cpi_levels.iloc[0,]
	df_cpi_levels["USD"] = 1

	# Merge CPI frames
	df_cpi_levels = pd.merge(df_cpi_levels, df_cpi_rer, how="left", left_index=True, right_index=True, suffixes=("", "_chg"))

	# Calculate CPI data using CPI change data
	df_cpi_change = df_cpi_levels.copy()

	# Calculate CPI levels
	for curr in column_name_order:
		temp_list = []
		temp_list.append(df_cpi_change.iloc[0][curr])
		for chg in df_cpi_change[curr + "_chg"]:
			temp_list.append(temp_list[-1] * ((chg + 100) / 100))
		df_cpi_change[curr] = temp_list[:-1]

	df_cpi_change = df_cpi_change[column_name_order]

	# Getting overall data frame
	df_cpi_overall = pd.merge(df_currency_spot_rer, df_cpi_change, how="left", left_index=True, right_index=True,
							suffixes=("_fx", "_cpi"))

	# Calculating RER and RER 1m change
	column_name_order.remove('USD')
	for curr in column_name_order:
		df_cpi_overall[curr + "_RER"] = df_cpi_overall[curr + "_fx"] * df_cpi_overall[curr + "_cpi"] / df_cpi_overall["USD_cpi"]
		df_cpi_overall[curr + "_RER_1m_chg"] = df_cpi_overall[curr + "_RER"].pct_change()
		df_cpi_overall[curr + "_RER_2m_chg"] = df_cpi_overall[curr + "_RER"].pct_change(2)
		df_cpi_overall[curr + "_RER_3m_chg"] = df_cpi_overall[curr + "_RER"].pct_change(3)

	return df_cpi_overall.filter(regex='_RER', axis=1).copy()


"""
===============================================================================
RER Ranking
===============================================================================
"""

# Taking absolutes for value premium:
# send short signal when overvalue and long signal when undervalue


# Abs model input: element-wise
def return_abs(x):
	if x >= 1:
		return -1
	elif x < 1:
		return 1


# Abs rank model input: rows
def return_abs_rank(row):
	numNeg = 0
	for x in range(0, len(row)):
		if row[x] > 1:
			numNeg += 1

	row = row.rank(ascending=False)

	for x in range(0, len(row)):
		row[x] -= numNeg
		if row[x] <= 0:
			row[x] -= 1

	return row


# Abs rank 2 model input: rows
def return_abs_rank_2(row):
	absRow = abs(row - 1).rank()
	signRow = np.sign(1 - row)

	row = absRow * signRow
	return row

# Abs rank 3 model input: rows
def return_abs_diff_top(row):
	temp_row = np.absolute(1-row)
	sign_row = np.sign(1 - row)

	temp_row = temp_row.rank(ascending=False)

	for x in range(0, len(temp_row)):
		if temp_row[x] != 1:
			temp_row[x] = 0

	row = temp_row * sign_row
	return row


# For signal changes
def return_rank(row):
	max = np.max(row)
	mid = (max+1)//2
	for x in range(0, len(row)):
		if row[x] <= mid:
			row[x] = max - mid - row[x] + 1
		elif row[x] > mid:
			row[x] = -row[x]+mid
	return row

"""
===============================================================================
Backtesting module
===============================================================================
"""


# Get annualised returns
def get_annualised(col, interval=12):
	start = 100
	base = 100
	for x in range(1, len(col)):
		start *= (1 + col[x])
	total = (start - base) / base

	annualised = (1 + total) ** (interval / (len(col) - 1)) - 1

	return annualised


# Get Sharpe ratio
def get_sharpe_ratio(col, annual_rf):

	avg_return = np.mean(col)
	std = np.std(col)

	monthly_rf = (1 + annual_rf)**(1/12) - 1
	sharpe = (avg_return - monthly_rf)/std
	sharpe = sharpe * math.sqrt(12)

	return sharpe


# Get cumulative returns
def get_cumulative(col):
	start = [100]
	for x in range(1, len(col)):
		start.append(start[-1] * (1 + col[x]))

	return start


# Backtesting module
def backtest(signal_df, signal_fx_list, spot_df, forward_df, skiprows=0, backtest_shift = 1):
	"""

	Perform backtesting using the specified strategy's signal

	:param signal_df: signals to indicate a long(positive) or short(negative) position
	:param signal_fx_list: the column names for the signals
	:param interval: interval of backtest
	:return: Backtest returns contained in a dataframe
	"""

	# Rename columns
	spot_df.columns = signal_fx_list
	forward_df.columns = signal_fx_list
	signal_df = signal_df.add_suffix('_signal')

	# Merge required dfs into one mega df
	backtest_df = pd.merge(forward_df, spot_df, how="left", left_index=True, right_index=True, suffixes=("_forward", "_spot"))
	backtest_df = pd.merge(backtest_df, signal_df, how="left", left_index=True, right_index=True)

	backtest_df["total_units"] = backtest_df[[s + "_signal" for s in signal_fx_list]].apply(lambda x: abs(x)).sum(axis=1)

	# Sort dates
	backtest_df.sort_index(inplace=True)

	# Shift forward outrights by 1, shift signals as per required
	shift_forward_cols = backtest_df.filter(regex="_forward").columns
	shift_signal_cols = backtest_df.filter(regex="_signal").columns

	backtest_df[shift_forward_cols] = backtest_df[shift_forward_cols].shift(1, axis=0)
	backtest_df[shift_signal_cols] = backtest_df[shift_signal_cols].shift(backtest_shift, axis=0)
	backtest_df["total_units"] = backtest_df["total_units"].shift(backtest_shift, axis=0)

	# Apply row skip
	if skiprows > 0:
		backtest_df = backtest_df[skiprows+backtest_shift:]

	for curr in signal_fx_list:
		absolute_return = (backtest_df[curr + "_spot"] - backtest_df[curr + "_forward"]) / backtest_df[curr + "_forward"]
		weight = backtest_df[curr + "_signal"] / backtest_df["total_units"]
		backtest_df[curr + "_weighted_return"] = absolute_return * weight

	backtest_df["total_return"] = backtest_df[[s + "_weighted_return" for s in signal_fx_list]].sum(axis=1)
	backtest_df["cumulative_return"] = get_cumulative(backtest_df["total_return"])

	return backtest_df.filter(regex="return")

"""
===============================================================================
Model Base: RER Calculation for building signals and Base Strategy (Power Diff)
===============================================================================
"""

# Import monthly FX spot
fx_spot_monthly = pd.read_csv("collection_output/fxSpot_monthly_end.csv")
fx_spot_monthly.set_index('DATE', inplace=True)
fx_spot_monthly.index = pd.to_datetime(fx_spot_monthly.index)
fx_spot_monthly.index = fx_spot_monthly.index.to_period("m")

# Import CPI
df_cpi_monthly = pd.read_csv("collection_output/cpi_monthly.csv")
df_cpi_monthly.set_index('Date', inplace=True)
df_cpi_monthly.index = pd.to_datetime(df_cpi_monthly.index)
df_cpi_monthly.index = df_cpi_monthly.index.to_period("m")

# Import forward outright
forward_outright_monthly_start_dict = {'1_month': 'forwardOutright_1_month_end.csv',
									 '2_month': 'forwardOutright_2_month_end.csv',
									 '3_month': 'forwardOutright_3_month_end.csv'}

forward_outright_monthly_df_dict = {}

for tenor, filepath in forward_outright_monthly_start_dict.items():
	forward_outright_df = pd.read_csv("collection_output/" + filepath)
	forward_outright_df.set_index('DATE', inplace=True)
	forward_outright_df.index = pd.to_datetime(forward_outright_df.index)
	forward_outright_df.index = forward_outright_df.index.to_period("m")
	forward_outright_monthly_df_dict[tenor] = forward_outright_df


# Calculate RER using CPI and monthly fx spot
fx_spot_monthly_RER = fx_spot_monthly.copy()
fx_spot_monthly_RER["USD"] = 1.0
RERColList = ["EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "NOK", "SEK", "USD"]
df_rer = calculate_RER(fx_spot_monthly_RER, df_cpi_monthly, RERColList)

backtest_results_dict = {}

# Build power difference model
diff_power_dict = {}
for i in [1, 2, 10, 25, 100, 200]:
	if i == 0:
		continue

	df_rer_abs_difference_power_step = df_rer.copy().filter(regex="_RER$").apply(lambda x: np.power(abs(1 - x), i) * np.sign(1 - x))
	df_rer_abs_difference_power_step.columns = RERColList
	df_rer_abs_difference_power_step.to_csv("model_output/1MAbsDiffPower_" + str(i) + ".csv")
	diff_power_dict[str(i)] = backtest(df_rer_abs_difference_power_step, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 1)
	diff_power_dict[str(i)].to_csv("backtest_output/1MAbsDiffPower_" + str(i) + ".csv")


# Rank absolute difference, and only allocate to the biggest difference
df_rer_abs_difference_power_top = df_rer.copy().filter(regex="_RER$")
df_rer_abs_difference_power_top = df_rer_abs_difference_power_top.apply(lambda x:  return_abs_diff_top(x), axis=1).iloc[1:]
df_rer_abs_difference_power_top.columns = RERColList
df_rer_abs_difference_power_top.to_csv("model_output/1MAbsDiffPower_Top.csv")

backtest_results_dict['1MAbsDiffPower_Top'] = backtest(df_rer_abs_difference_power_top, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 1)

"""
===============================================================================
Model 1: Absolute value investing
===============================================================================
"""

backtest_results_dict['1MAbsDiffPower_200'] = diff_power_dict['200']

df_rer_model_1 = df_rer.copy()

# Extract only RER columns
rer_rank_model = df_rer_model_1[[s + "_RER" for s in RERColList]]

# Equal weight allocation, regardless of degree of overvaluation/undervaluation
rer_rank_model_1M_abs = rer_rank_model.copy()
rer_rank_model_1M_abs = rer_rank_model_1M_abs.applymap(lambda x: return_abs(x)).iloc[1:]
rer_rank_model_1M_abs.columns = RERColList
rer_rank_model_1M_abs.to_csv("model_output/1MAbs.csv")

# Split valuation into overvalue and undervalue and rank separately. Negative numbers signify overvalue
rer_rank_model_1M_abs_rank = rer_rank_model.copy()
rer_rank_model_1M_abs_rank = rer_rank_model_1M_abs_rank.apply(lambda x: return_abs_rank(x), axis=1).iloc[1:]
rer_rank_model_1M_abs_rank.columns = RERColList
rer_rank_model_1M_abs_rank.to_csv("model_output/1MAbsRank.csv")

# Split valuation into overvalue and undervalue and rank together. Negative numbers signify overvalue
rer_rank_model_1M_abs_rank_2 = rer_rank_model.copy()
rer_rank_model_1M_abs_rank_2 = rer_rank_model_1M_abs_rank_2.apply(lambda x: return_abs_rank_2(x), axis=1).iloc[1:]
rer_rank_model_1M_abs_rank_2.columns = RERColList
rer_rank_model_1M_abs_rank_2.to_csv("model_output/1MAbsRank2.csv")

backtest_results_dict['1MAbs'] = backtest(rer_rank_model_1M_abs, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 1)
backtest_results_dict['1MAbsRank1'] = backtest(rer_rank_model_1M_abs_rank, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 1)
backtest_results_dict['1MAbsRank2'] = backtest(rer_rank_model_1M_abs_rank_2, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 1)

"""
===============================================================================
Model 2: Signal change value investing
===============================================================================
"""

df_rer_model_2 = df_rer.copy()

signal_chg_3m = df_rer_model_2[[s + "_RER_3m_chg" for s in RERColList]].copy().rank(axis=1)
signal_chg_2m = df_rer_model_2[[s + "_RER_2m_chg" for s in RERColList]].copy().rank(axis=1)
signal_chg_1m = df_rer_model_2[[s + "_RER_1m_chg" for s in RERColList]].copy().rank(axis=1)
signal_raw = df_rer_model_2[[s + "_RER" for s in RERColList]].copy().rank(axis=1)

signal_chg_3m.columns = RERColList
signal_chg_2m.columns = RERColList
signal_chg_1m.columns = RERColList
signal_raw.columns = RERColList

signal_chg_3m = signal_chg_3m.apply(lambda x: return_rank(x), axis=1)
signal_chg_3m.to_csv("model_output/1MChg3mTop4.csv")

signal_chg_2m = signal_chg_2m.apply(lambda x: return_rank(x), axis=1)
signal_chg_2m.to_csv("model_output/1MChg2mTop4.csv")

signal_chg_1m.apply(lambda x: return_rank(x), axis=1)
signal_chg_1m.to_csv("model_output/1MChg1mTop4.csv")

signal_raw = signal_raw.apply(lambda x: return_rank(x), axis=1)
signal_raw.to_csv("model_output/1MRERRawTop4.csv")

backtest_results_dict['1MChg1mTop4'] = backtest(signal_chg_1m, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 1)
backtest_results_dict['2MChg2mTop4'] = backtest(signal_chg_2m, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 2)
backtest_results_dict['3MChg3mTop4'] = backtest(signal_chg_3m, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 3)
backtest_results_dict['1MRERRawTop4'] = backtest(signal_raw, RERColList, fx_spot_monthly, forward_outright_monthly_df_dict['1_month'], 1)

"""
===============================================================================
Plot Strategies
===============================================================================
"""

# Plot power difference strategy and get returns result in a matrix
plt.plot(figsize=(10, 5), title="Cumulative Returns (Power Optimization)")

power_returns_matrix = []
for i in [1, 2, 10, 25, 100, 200]:
	diff_power_dict[str(i)]["cumulative_return"][len(diff_power_dict[str(i)]):None:-1].plot(label=str(i))
	power_returns_matrix.append({'Power': str(i),
						 'Annualized_returns': str("{:.2f}".format(get_annualised(diff_power_dict[str(i)]["total_return"]) * 100) + "%"),
						 'Sharpe_ratio': str("{:.2f}".format(get_sharpe_ratio(diff_power_dict[str(i)]["total_return"], 0.015))),
						 'Cumulative_returns': str("{:.2f}".format(get_cumulative(diff_power_dict[str(i)]["total_return"])[-1]) + "%"),
						 'Std_dev': np.std(diff_power_dict[str(i)]["total_return"])})

power_returns_matrix = pd.DataFrame(power_returns_matrix)
power_returns_matrix.set_index('Power', inplace=True)
power_returns_matrix.to_csv('backtest_summary/power_diff_returns.csv')

plt.legend(loc='upper left')
plt.savefig('backtest_summary/n_optimization.png')
plt.clf()

# Plot all strategies in the same graph
plt.plot(figsize=(10, 5), title="Cumulative Returns (All strategies)")
for result_name, backtest_result in backtest_results_dict.items():
	backtest_result.to_csv("backtest_output/" + result_name + ".csv")
	backtest_result["cumulative_return"][len(backtest_result):None:-1].plot(label=result_name)

plt.legend(loc='upper left')
plt.savefig('backtest_summary/strat_comparison.png')
plt.clf()

# Form matrix of returns
returns_matrix = []
for result_name, backtest_result in backtest_results_dict.items():
	returns_matrix.append({'Name': result_name,
						   'Annualized_returns': str("{:.2f}".format(get_annualised(backtest_result["total_return"]) * 100) + "%"),
						   'Sharpe_ratio': str("{:.2f}".format(get_sharpe_ratio(backtest_result["total_return"], 0.015))),
						  'Cumulative_returns': str("{:.2f}".format(get_cumulative(backtest_result["total_return"])[-1]) + "%"),
							'Std_dev': np.std(backtest_result["total_return"])})
returns_matrix = pd.DataFrame(returns_matrix)
returns_matrix.set_index('Name', inplace=True)

returns_matrix.to_csv('backtest_summary/strat_returns.csv')

"""
===============================================================================
For UBS Quant Hackathon 2019
===============================================================================
"""

# Print out the final strategy values
dominant_strategy = '1MAbsDiffPower_Top'
print("Dominant_strategy:" + dominant_strategy)
print("Sharpe ratio for dominant strategy: " +
	  str(returns_matrix.iloc[returns_matrix.index.get_loc(dominant_strategy)]['Sharpe_ratio']))
print("Annualised for dominant strategy: " +
	  str(returns_matrix.iloc[returns_matrix.index.get_loc(dominant_strategy)]['Annualized_returns']))
print("Net cumulative return for dominant strategy: " +
	  str(returns_matrix.iloc[returns_matrix.index.get_loc(dominant_strategy)]['Cumulative_returns']))

# Export strategy values over time series (cumulative returns) for submission
backtest_results_dict[dominant_strategy]['Strategy value (Cumulative_returns)'] = get_cumulative(backtest_results_dict[dominant_strategy]["total_return"])
backtest_results_dict[dominant_strategy][['Strategy value (Cumulative_returns)']].to_csv('strategy_values_time_series.csv')
