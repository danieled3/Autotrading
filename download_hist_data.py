# This code download 2 years of historical data of the 200 main stock index by using alpha_vantage

from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import pandas as pd
import time

apiKey = 'BEIUB17VP325LN89'

years = tuple(range(2, 0, -1))
months = tuple(range(12, 0, -1))
symbols = [
#'CLOV',
#'CIG',
#'MTSL',
#'AAPL',
#'NIO',
#'EBON',
#'BAC',
#'WFC',
#'PFE',
#'PLUG',
#'GE',
#'AMD',
#'PLTR',
#'AMC',
#'VIAC',
#'PCG',
#'FCEL',
#'PINS',
#'WKHS',
#'F',
#'ITUB',
#'PBR',
#'T',
#'IQ',
#'SOS',
#'MS',
#'C',
#'CSCO',
#'TSLA',
#'VALE',
#'AAL',
#'MSFT',
#'RIOT',
#'INTC',
#'MARA',
#'QS',
#'BNGO',
#'IDEX',
#'NOK',
#'XOM',
#'GOLD',
#'AZN',
#'CCL',
#'GGB',
#'VZ',
#'SNAP',
#'TIRX',
#'CMCSA',
#'MRNA',
#'FCX',
#'NUAN',
#'KO',
#'AA',
#'TME',
#'OCGN',
#'BBD',
#'CCIV',
#'PPD',
#'MRO',
#'VIPS',
#'MO',
#'OPEN',
#'NRZ',
#'ZNGA',
#'DISCA',
#'X',
#'DKNG',
#'NEE',
#'SKLZ',
#'INFY',
#'SIRI',
#'CLF',
#'HBAN',
#'BA',
#'BMY',
#'BABA',
#'NNDM',
#'JD',
#'JPM',
#'GSX',
#'XPEV',
#'ORCL',
#'NKLA',
#'ET',
#'FB',
#'MRVL',
#'CAN',
#'ENIA',
#'FUBO',
#'ABEV',
#'SPCE',
#'FSR',
#'MU',
#'TLRY',
#'NCLH',
#'DAL',
#'GEVO',
#'HOFV',
#'XEL',
#'MRK',
#'UBER',
#'LI',
#'SWN',
#'KMI',
#'OXY',
#'NLY',
#'KIM',
#'GM',
#'LSCC',
#'MAC',
#'BP',
#'UAL',
#'XL',
#'OGI',
#'RUN',
#'SCHW',
#'STON',
#'APHA',
#'BK',
#'GIS',
#'DQ',
#'KGC',
#'MGM',
#'M',
#'TSM',
#'DIS',
#'TWTR',
#'HPE',
#'SLB',
#'EBET',
#'JNJ',
#'BE',
#'INO',
#'BTG',
#'WMT',
#'VICI',
#'MDLZ',
#'CDEV',
#'SQ',
#'CVS',
#'HAL',
#'NVDA',
#'AY',
#'ABBV',
#'RIG',
#'RIDE',
#'AXTA',
#'KHC',
#'SU',
#'CL',
#'AUY',
#'GSK',
#'KEY',
#'USB',
#'VTRS',
#'BKR',
#'JMIA',
#'GILD',
#'PG',
#'AMAT',
#'COP',
#'RTX',
#'MVIS',
#'EBR',
#'BB',
#'CNP',
#'LUMN',
#'FTCH',
#'BSX',
#'LFMD',
#'PPL',
#'CVX',
#'CSIQ',
#'EDU',
#'ALXN',
#'ERIC',
#'HPQ',
#'ATVI',
#'PPG',
#'ING',
'EFOI',
'PLBY',
'SPLK',
'EW',
'NKE',
'NOVA',
'V',
'KR',
'EDIT',
'PBR.A',
'AMCR',
'AGC',
'QCOM',
'DVN',
'RMO',
'RKT',
'CERN',
'MP',
'PEP',
'NEM'
]

ts = TimeSeries(key=apiKey, output_format='csv')

# years = [1]
# months = [2]
# symbols = ['AAPL']
symbols_to_load = 20

now = datetime.now()  # current date and time
all_days = [(now - timedelta(i)).strftime("%Y-%m-%d") for i in range(0, 850) if
            (now - timedelta(i)).weekday() < 5]  # array of all days
df_tot = pd.DataFrame(all_days, columns=['time'])
counter = 1

for symbol in symbols:
    print(symbol)
    df_symbol = pd.DataFrame(columns=['time', symbol + '_close', symbol + '_volume'])
    for year in years:
        for month in months:

            print('Computation of year -' + str(year) + ' and month -' + str(month) + '...')
            slice = 'year' + str(year) + 'month' + str(month)

            totalData = ts.get_intraday_extended(symbol=symbol, interval='60min', slice=slice)  # download the csv
            df = pd.DataFrame(list(totalData[0]))  # csv --> dataframe

            header_row = 0
            df.columns = df.iloc[header_row]  # set column header
            df = df.drop(header_row)
            df = df.reset_index()
            df['time'] = [t[0:10] for t in df.time]  # extract date from datetime
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')  # transform string into integer
            df = df[['time', 'close', 'volume']].groupby('time').agg(
                {'close': ['first'], 'volume': ['sum']}).reset_index()
            df.columns = ['time', symbol + '_close', symbol + '_volume']  # rename columns
            df_symbol = df_symbol.append(df)  # append current df to full df_symbol

            if counter % 5 == 0:
                print('I am waiting for the API to reaload...')
                time.sleep(59)  # wait for API to reaload

            counter += 1

    df_tot = pd.merge(df_tot, df_symbol, on='time', how='left')
    if counter >= symbols_to_load * 24:
        drive.mount('drive')
        df_tot.to_csv('df_tot_11.csv')
        !cp
        df_tot_11.csv
        "drive/My Drive/"
        break



