# from dataclasses import dataclass
# from typing import Any

def create_dataclass_from_dict(data_dict:dict, class_name:str):
    fields = {}
    
    for key, value in data_dict.items():
        if isinstance(value, dict):
            nested_class_name = key.capitalize()
            nested_class = create_dataclass_from_dict(value, class_name=nested_class_name)
            fields[key] = nested_class
        else:
            fields[key] = value
    
    return type(class_name, (), fields)

stock_price = {'lastPrice': 129.1,
 'change': 2.530000000000001,
 'pChange': 1.9988938927075939,
 'previousClose': 126.57,
 'open': 126.57,
 'close': 0,
 'vwap': 128.36,
 'lowerCP': '113.91',
 'upperCP': '139.22',
 'pPriceBand': 'No Band',
 'basePrice': 126.57,
 'intraDayHighLow': {'min': 126.4, 'max': 129.55, 'value': 129.1},
 'weekHighLow': {'min': 49.7,
  'minDate': '26-Jun-2023',
  'max': 142.9,
  'maxDate': '30-Apr-2024',
  'value': 129.1},
 'iNavValue': None,
 'checkINAV': False}

StockData = create_dataclass_from_dict(stock_price, 'stock_price')

# Create an instance of the dynamically created dataclass
stock = StockData()

# Accessing nested dictionary values using dot notation
print(stock.intraDayHighLow.min)  # Output: 126.4
print(stock.weekHighLow.maxDate)  # Output: 30-Apr-2024
print(stock.open)
