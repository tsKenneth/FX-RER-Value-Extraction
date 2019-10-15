# FX-RER-Value-Extraction
Value premium extraction using Real Exchange Rate

## Analysis methodology and findings:
**Asset:** G10 currency
**Instrument:** 1 month forwards

**Sharpe Ratio:** 1.31
**Annualised Return:** 7.99%
**Period of analysis:** Jan 2017 to Oct 2018

## Requirements
Access token to Refinitiv DSA platform. Due to legal reasons, we're unable to provide a copy of the data we collected

## How to Run:
1. Download, unzip and place the application directory anywhere on your local machine with both read and write access
1. The script uses a external library dependencies to run various functions. To use these libraries you can:
	*a. On Windows OS, change directory to the root of the application and type "venv\Scripts\activate" to activate the virtual environment
	   On Linux OS, change directory to the root of the application and type "source /venv/Scripts/activate" to activate the virtual environment
	*b. Change directory to the root of the application and type "pip install -r requirements.txt" to install the libraries into your system python interpreter. Ensure that the system's python interpreter version you're running is >=3.7.4. Otherwise, please use the venv method.
1. Open data_collection.py and provide your access token to the DSA API. In the application root directory, type 'python data_collection.py"
1. In the application root directory, type "python quant_model.py" to run the script
1. The sharpe ratio, annualised returns and net cumulative returns of the dominant strategy will be printed.
1. (Optional) Type "deactivate" to deactivate the virtual environment (if step 2a. was used)

## Credits and Acknolwedgements
Analysis created using data provided by Thomson Reuters, using the Data Science Accelerator platform.
Submitted for the UBS Quant Hackathon 2019

Jonathan Kevin Chandra https://github.com/jkchandra
Teo Tian Shun Kenneth https://github.com/tsKenneth
Ye Fan (Steve) https://github.com/handsomejeff
