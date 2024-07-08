from dataclasses import make_dataclass, field, asdict

def create_dataclass_from_dict(data_dict: dict, class_name:str):

    """Creates dataclass type object from a nested dictionary.
    
    Usage:
    ---
    stock_price = {'lastPrice': 129.1,
    'open': 126.57,
    'close': 0,
    'weekHighLow': {'min': 49.7, 'minDate': '26-Jun-2023', ...}
    }

    # instantiate
    Stock = create_dataclass_from_dict(stock_price, 'Stock')
    my_stock = Stock()

    # check nested
    print(my_stock.weekHighLow.minDate) # works '26-Jun-2023'

    # Reconstruct to dict
    print(asdict(my_stock))
    """

    fields = []
    
    for key, value in data_dict.items():
        if isinstance(value, dict):
            nested_class_name = key.capitalize()
            nested_class = create_dataclass_from_dict(value, class_name=nested_class_name)
            fields.append((key, nested_class, field(default_factory=nested_class)))
        else:
            fields.append((key, type(value), field(default=value)))
    
    return make_dataclass(class_name, fields)