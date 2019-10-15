"""
Extracting value premium from G10 Currencies using monthly real exchange rates
"""

# Import statements
import pandas as pd
import requests
import json

from datetime import datetime
from scipy import interpolate

"""
Configurations
"""

# Library Configs - Set how many rows and columns to display when using
# dataframe.display()
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# HTTP API for retrieving data on historical prices
PRICE_RESOURCE_ENDPOINT = "https://dsa-stg-edp-api.fr-nonprod.aws.thomsonreuters.com/data/historical-pricing/beta1/views/summaries/"

# IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT

# Personal key for Data Science Accelerator access to Historical Pricing Data
# Replace with your own personal key
ACCESS_TOKEN = 'Provide your own token here'

# IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT


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
DATA_END_DATE = '2018-10-31'


# RIC codes for the various asset classes
FX_Dict = {'EUR': 'EUR=', 'GBP': 'GBP=', 'JPY': 'JPY=', 'AUD': 'AUD=', 'NZD': 'NZD=', 'CAD': 'CAD=', 'CHF': 'CHF=',
		   'NOK': 'NOK=', 'SEK': 'SEK='}
commodities_Dict = {'Gold': 'XAU=', 'Silver': 'XAG='}
indices_Dict = {"DJIA": ".DJI", "NASDAQ": ".IXIC", "NYSE": ".NYA", "S&P500": ".SPX", "FTSE100": ".FTSE",
				"STOXX50": ".STOXX50E", "DAX": ".GDAXI", "NIKKEI": ".N225"}

reciprocal_currency_list = ['JPY', 'CAD', 'CHF', 'NOK', 'SEK']

IR_Dict = {'USD Rate', 'GBP Rate', 'JPY Rate', 'AUD Rate', 'NZD Rate', 'CAD Rate', 'CHF Rate', 'NOK Rate', 'SEK Rate'}

"""
===============================================================================
Data Import Functions
===============================================================================
"""


# Generic GET request function
def get_request(url, requested_data):
	"""
	HTTP GET request to Refinitiv API

	Retrieves data from Refinitiv Historical Pricing Data API

	:param url: str, the url of the API endpoint
	:param requested_data: dict, contains user-defined variables to specify requested data
	:return: DataFrame, containing the historical data. Returns None if response does not contain data
	"""
	data_response = requests.get(url, headers={'X-api-key': ACCESS_TOKEN},
								 params=requested_data);

	if data_response.status_code != 200:
		raise ValueError("get_request: Unable to get data. Code %s, Message: %s"
						 % (data_response.status_code, data_response.text))
	else:
		try:
			json_response = json.loads(data_response.text);
			data = json_response[0]['data']
			headers = json_response[0]['headers']
			names = [headers[x]['name'] for x in range(len(headers))]
			df = pd.DataFrame(data, columns=names)
			return df
		except KeyError:
			return None


# Get Price Data
def get_request_price(ric, interval, start_date, end_date, fields=None):
	"""
	HTTP GET price request to Refinitiv API

	Gets data on the historical prices of an asset or a list of assets

	:param ric: ric code(s) to retrieve historical data on
	:type ric: string or list of strings
	:param interval: interval of the data retrieved
	:type interval: string
	:param start_date: lower bounding date from which to retrieve data starting from
	:type start_date: string
	:param end_date: upper bounding date from which to retrieve data ending on
	:type end_date: string
	:param fields: the specific data fields to be retrieved
	:type fields: string or list of strings
	:return: DataFrame, containing the historical pricing data. If a list of RIC
		code is provided, then the columns will be merged on the date column
	"""

	# Check if fields is none, and append to requestData if it exists
	if fields is None:
		requested_data = {
			'interval': interval,
			'start': start_date,
			'end': end_date,
		}
	else:
		requested_data = {
			'interval': interval,
			'start': start_date,
			'end': end_date,
			'fields': fields
		}

	# If ric is a list, perform request iteratively and merge into a single dataframe
	if type(ric) is list:
		price_df = pd.DataFrame()
		for x in ric:
			x_df = get_request(str(PRICE_RESOURCE_ENDPOINT + x), requested_data)

			if x_df is not None:
				x_df.set_index("DATE", inplace=True)
				x_df = x_df.add_prefix(str(x + '_'))
				price_df = price_df.join(x_df, how='outer')
			else:
				print("get_request_price: %s is not a valid RIC." % x)
	elif type(ric) is str:
		price_df = get_request(str(PRICE_RESOURCE_ENDPOINT + ric), requested_data)
	else:
		raise ValueError("get_request_price: RIC is not a list or a string.")

	return price_df


"""
===============================================================================
Data Manipulation Functions
===============================================================================
"""


# Helper functions for Cubic Interpolation
def diff_month(d1, d2):
	return (d1.year - d2.year) * 12 + d1.month - d2.month


def next_month(date):
	year, month = date.split('-')
	year = int(year)
	month = int(month)

	month += 1
	if month > 12:
		month -= 12
		year += 1

	sm = str(month)
	sy = str(year)

	if len(sm) == 1:
		sm = '0' + sm

	return sy + '-' + sm


# Cubic interpolation for transforming quarterly data to monthly data
def cubic_interpolate(x_data, y_data, start, stop, timeframe):
	date_format = '%Y-%m'
	y_start = datetime.strptime(y_data[0], date_format)
	y_end = datetime.strptime(y_data[-1], date_format)

	y_data2 = [diff_month(datetime.strptime(m, date_format), y_start) for m in y_data]

	start_m = diff_month(datetime.strptime(start, date_format), y_start)
	stop_m = diff_month(datetime.strptime(stop, date_format), y_end) + 1

	tck = interpolate.splrep(y_data2, x_data)

	alist = []
	mlist = [start]

	for x in range(start_m, stop_m, timeframe):
		alist.append(interpolate.splev(x, tck))
		mlist.append(next_month(mlist[-1]))

	return alist, mlist[:-1]


"""
===============================================================================
Dataframes Creation Functions
===============================================================================
"""

# Convert daily to monthly by taking earliest day in the month
def get_interval(dataframe, grouper_frequency, take_first, period_type = None):
	dataframe = dataframe.copy()
	dataframe.index = pd.to_datetime(dataframe.index)
	if take_first is True:
		dataframe = dataframe.groupby(pd.Grouper(freq=grouper_frequency)).first()
	else:
		dataframe = dataframe.groupby(pd.Grouper(freq=grouper_frequency)).last()

	if period_type is not None:
		dataframe.index = dataframe.index.to_period(period_type)

	return dataframe


# Reciprocal
def reciprocate_currency(dataframe, reciprocal_list):
	"""

	Reciprocate the values of columns for currencies subordinate to USD

	:param dataframe: dataframe to reciprocate values
	:param reciprocal_list: list of currencies to reciprocate values for
	:return: dataframe with reciprocated values
	"""
	regex_reciprocal = "|".join(reciprocal_list)
	dataframe = dataframe.copy()
	reciprocal_columns = dataframe.filter(regex=regex_reciprocal).columns
	dataframe[reciprocal_columns] = dataframe[reciprocal_columns].apply(lambda x: 1 / x, axis=1)

	return dataframe


# CPI Information
def create_cpi_dataframe(cpi_file, period_type):
	"""

	Retrieves and format CPI data

	:param cpi_file: path to external cpi file
	:param period_type: period type for converting index to period index.
	:return: cpi dataframe
	"""
	df_cpi = pd.read_csv(cpi_file)
	df_cpi["Date"] = pd.to_datetime(df_cpi["Date"]).dt.to_period(period_type)
	df_cpi.set_index('Date', inplace=True)
	df_cpi = df_cpi.add_suffix('_CPI')
	return df_cpi


# Spot prices
def create_spot_dataframe(ric_dictionary, spot_interval_dict, price_column_name='MID_PRICE'):
	"""

	Retrieves spot prices on assets from DSA

	:param ric_dictionary: dictionary of ric codes to retrieve spot price from
	:param spot_interval_dict: list of intervals to obtain from the data
	:param price_column_name: column containing the price of the assets
	:return: spot dataframe
	"""

	df_spot = get_request_price(list(ric_dictionary.values()),
								INTERVAL_DICT['daily'], DATA_START_DATE, DATA_END_DATE, price_column_name)

	df_dict = {}
	for spot_interval in spot_interval_dict:
		if spot_interval == "weekly_start":
			df_dict[spot_interval] = get_interval(df_spot,'W-MON', True)
		elif spot_interval == "weekly_end":
			df_dict[spot_interval] = get_interval(df_spot, 'W-MON', False)
		elif spot_interval == 'monthly_start':
			df_dict[spot_interval] = get_interval(df_spot, 'MS', True, 'm')
		elif spot_interval == 'monthly_end':
			df_dict[spot_interval] = get_interval(df_spot, 'MS', False, 'm')
		elif spot_interval == 'daily':
			continue
		else:
			raise ValueError("Only daily, weekly_start, weekly_end, monthly_start, monthly_end")
	return df_dict


# Forward points
def create_forward_points_dataframe(forward_interval_dict, forward_tenor, price_col_name='MID_PRICE'):
	"""

	Retrieves forward points from DSA

	:param forward_interval_dict: list of intervals to obtain forward points
	:param forward_tenor: tenor of forward
	:param price_col_name: column containing the price of the forwards
	:return: forward points dataframe
	"""
	fx_list = [str(cur) + str(forward_tenor) + "=" for cur in FX_Dict.keys()]

	df_currency_forward = get_request_price(fx_list, INTERVAL_DICT['daily'], DATA_START_DATE, DATA_END_DATE, price_col_name)

	df_dict = {}
	for forward_interval in forward_interval_dict:
		if forward_interval == "weekly_start":
			df_dict[forward_interval] = get_interval(df_currency_forward, 'W-MON', True)
		elif forward_interval == "weekly_end":
			df_dict[forward_interval] = get_interval(df_currency_forward, 'W-MON', False)
		elif forward_interval == 'monthly_start':
			df_dict[forward_interval] = get_interval(df_currency_forward, 'MS', True, 'm')
		elif forward_interval == 'monthly_end':
			df_dict[forward_interval] = get_interval(df_currency_forward, 'MS', False, 'm')
		elif forward_interval == 'daily':
			continue
		else:
			raise ValueError("Only daily, weekly_start, weekly_end, monthly_start, monthly_end")

	return df_dict


# Forward Outright
def create_forward_outright_dataframe(fx_spot_df, fx_forward_df, forward_tenor, price_col_name='MID_PRICE'):
	"""

	Calculates forward outright using spot prices and forward points

	:param fx_spot_df: fx spot dataframe, interval needs to be the same as fx forward dataframe
	:param fx_forward_df: fx forward dataframe to calcuate outright from
	:param forward_tenor: tenor of the forwards
	:param price_col_name: column containing the spots in fx spot dataframe
	:return: forward outright dataframe
	"""
	df_currency_forward_outright = pd.merge(fx_spot_df.copy(), fx_forward_df.copy(),
											how="left", left_index=True, right_index=True)
	for column in FX_Dict.keys():
		df_currency_forward_outright[column + "_outright"] = df_currency_forward_outright[column + "=_" + price_col_name] + \
															 (df_currency_forward_outright[column + forward_tenor + "=_" + price_col_name] / 10000)

	df_currency_forward_outright["JPY_outright"] = df_currency_forward_outright["JPY" + "=_" + price_col_name] + \
												   (df_currency_forward_outright["JPY" + forward_tenor + "=_" + price_col_name] / 100)

	df_currency_forward_outright = df_currency_forward_outright.filter(regex="_outright")

	return df_currency_forward_outright

"""
===============================================================================
Dataframes Creation and Export
===============================================================================
"""

# CPI
print("Retrieving CPI:")
create_cpi_dataframe("external_data/CPI_monthly.csv", 'm').to_csv("collection_output/cpi_monthly.csv")
create_cpi_dataframe("external_data/CPI_quarterly.csv", 'q').to_csv("collection_output/cpi_quarterly.csv")

# FX Spot
print("Retrieving FX:")
fx_spot_interval_list = ['daily', 'monthly_start', 'monthly_end']
fx_spot_df_dict = create_spot_dataframe(FX_Dict, fx_spot_interval_list)
for name, df in fx_spot_df_dict.items():
	reciprocate_currency(df, reciprocal_currency_list).to_csv('collection_output/fxSpot_' + str(name) + '.csv')

# Commodities Spot
print("Retrieving Commodities:")
commodities_interval_list = ['daily', 'monthly_start', 'monthly_end']
commodities_spot_df_dict = create_spot_dataframe(commodities_Dict, commodities_interval_list)
for name, df in commodities_spot_df_dict.items():
	df.to_csv('collection_output/commoditiesSpot_' + str(name) + '.csv')

# Indices
print("Retrieving Indices:")
indices_interval_list = ['daily', 'monthly_start', 'monthly_end']
indices_df_dict = create_spot_dataframe(indices_Dict, indices_interval_list, 'TRDPRC_1')
for name, df in indices_df_dict.items():
	df.to_csv('collection_output/indicesSpot_' + str(name) + '.csv')

# Forward points
print("Retrieving Forward Points:")
forward_df_creation_dict = {'1_month': ['monthly_start', 'monthly_end'],
							'2_month': ['monthly_start', 'monthly_end'],
							'3_month': ['monthly_start', 'monthly_end']}
forward_df_dict = {}
for tenor, forward_interval_list in forward_df_creation_dict.items():
	temp_forward_df_dict = create_forward_points_dataframe(forward_interval_list, TENOR_DICT[tenor])
	forward_df_dict[tenor] = temp_forward_df_dict
	for name, df in temp_forward_df_dict.items():
		df.to_csv('collection_output/forwardPoints_' + str(tenor) + '_' + str(name) + '.csv')

# Forward outright
print("Retrieving Forward outrights:")
forward_outright_df_creation_dict = {'1_month_start': {'interval': 'monthly_start', 'tenor': '1_month'},
									 '2_month_start': {'interval': 'monthly_start', 'tenor': '2_month'},
									 '3_month_start': {'interval': 'monthly_start', 'tenor': '3_month'},
									 '1_month_end': {'interval': 'monthly_end', 'tenor': '1_month'},
									 '2_month_end': {'interval': 'monthly_end', 'tenor': '2_month'},
									 '3_month_end': {'interval': 'monthly_end', 'tenor': '3_month'}}

forward_outright_df_dict = {}

for name, params in forward_outright_df_creation_dict.items():
	forward_outright_df = create_forward_outright_dataframe(
		fx_spot_df_dict[params['interval']],
		forward_df_dict[params['tenor']][params['interval']],
		TENOR_DICT[params['tenor']])
	forward_outright_df_dict[name] = forward_outright_df
	reciprocate_currency(forward_outright_df_dict[name], reciprocal_currency_list).to_csv(
		'collection_output/forwardOutright_' + str(name) + '.csv')