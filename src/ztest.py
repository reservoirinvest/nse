my_dict = {'info': {'symbol': 'HDFCLIFE',
  'companyName': 'HDFC Life Insurance Company Limited',
  'activeSeries': ['EQ'],
  'debtSeries': [],
  'isFNOSec': True,
 'underlyingValue': 624.75,
 'vfq': 44001,
 'fut_timestamp': '09-Jul-2024 14:13:48',
 'stocks': [{'metadata': {'instrumentType': 'Stock Futures',
    'expiryDate': '25-Jul-2024',
    'optionType': '-',
    'strikePrice': 0,
    'identifier': 'FUTSTKHDFCLIFE25-07-2024XX0.00',
    'openPrice': 624.5,},
   'underlyingValue': 624.75,
   'volumeFreezeQuantity': 0,
   'marketDeptOrderBook': {'totalBuyQuantity': 684200,
    'totalSellQuantity': 1291400,
    'bid': [{'price': 626.05, 'quantity': 1100},
     {'price': 626, 'quantity': 15400},],
    'ask': [{'price': 626.2, 'quantity': 2200},
     {'price': 626.25, 'quantity': 4400},
     {'price': 626.3, 'quantity': 3300},],
    'carryOfCost': {'price': {'bestBuy': 626.05,
      'bestSell': 626.2,
      'lastPrice': 626.05},
     'carry': {'bestBuy': 4.741966849760131,
      'bestSell': 5.288483123086923,
      'lastPrice': 4.741966849760131}},
    'tradeInfo': {'tradedVolume': 6796,
     'value': 46783.8,},
    'otherInfo': {'settlementPrice': 622.3,
     'dailyvolatility': 1.52,}}},
  {'metadata': {'instrumentType': 'Stock Options',
    'expiryDate': '25-Jul-2024',
    'optionType': 'Call',
    'strikePrice': 630,},
   'underlyingValue': 624.75,
   'volumeFreezeQuantity': 44001,
   'marketDeptOrderBook': {'totalBuyQuantity': 394900,
    'totalSellQuantity': 371800,
    'bid': [{'price': 11.9, 'quantity': 11000},
     {'price': 11.85, 'quantity': 7700},],
    'ask': [{'price': 12, 'quantity': 5500},
     {'price': 12.05, 'quantity': 7700},
     {'price': 12.1, 'quantity': 14300},],
    'carryOfCost': {'price': {'bestBuy': 11.9,
      'bestSell': 12,
      'lastPrice': 12},
     'carry': {'bestBuy': -9035.605043144475,
      'bestSell': -9016.51497358361,
      'lastPrice': -9016.51497358361}},
    'tradeInfo': {'tradedVolume': 3763,
     'value': 524.04,
     'vmap': 12.66,},
    'otherInfo': {'settlementPrice': 0,
     'dailyvolatility': 1.52,}}},
  {'metadata': {'instrumentType': 'Stock Options',
    'expiryDate': '25-Jul-2024',
    'optionType': 'Call',
    'strikePrice': 630,},
   'underlyingValue': 624.75,
   'volumeFreezeQuantity': 44001,
   'marketDeptOrderBook': {'totalBuyQuantity': 394900,
    'totalSellQuantity': 371800,
    'bid': [{'price': 11.9, 'quantity': 11000},
     {'price': 11.85, 'quantity': 7700},],
    'ask': [{'price': 12, 'quantity': 5500},
     {'price': 12.05, 'quantity': 7700},
     {'price': 12.1, 'quantity': 14300},],
    'carryOfCost': {'price': {'bestBuy': 11.9,
      'bestSell': 12,
      'lastPrice': 12},
     'carry': {'bestBuy': -9035.605043144475,
      'bestSell': -9016.51497358361,
      'lastPrice': -9016.51497358361}},
    'tradeInfo': {'tradedVolume': 3763,
     'value': 524.04,
     'vmap': 12.66,},
    'otherInfo': {'settlementPrice': 0,
     'dailyvolatility': 1.52,}}},],
    'strikePrice': [0, 1, 2, 3, 5],
    'expiry': ['20240701', '20240702', '20240703']
}
}

import pandas as pd

# Function to extract data from nested dictionaries
def flatten_dict(data, prefix=''):
  items = []
  if isinstance(data, dict):
    for key, value in data.items():
      new_key = prefix + key if prefix else key
      if isinstance(value, dict):
        items.extend(flatten_dict(value, new_key + '_'))
      elif isinstance(value, list):
        # Handle lists by iterating and flattening elements
        for i, item in enumerate(value):
          items.extend(flatten_dict(item, new_key + f'_{i}'))
      else:
        items.append((new_key, value))
  return dict(items)

# Flatten the dictionary
flat_data = flatten_dict(my_dict)

# Extract field names from top level dictionary (if it's a dict)
field_names = []
if isinstance(flat_data, dict):
  field_names = [key for key in flat_data.keys() if not key.endswith('_')]

# Convert flattened dictionary to DataFrame (if flattening was successful)
df = pd.DataFrame()
if isinstance(flat_data, dict):
  df = pd.DataFrame(flat_data, index=[0])

# Select desired columns based on field names (if DataFrame exists)
if not df.empty:
  df = df[field_names]

# Print the DataFrame (if created)
if not df.empty:
  print(df.to_string())

