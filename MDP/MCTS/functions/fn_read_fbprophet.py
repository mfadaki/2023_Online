import random
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import math

# Read the data from the saved JSON file
with open("results_fbprophet.json", "r") as infile:
    results_dict = json.load(infile)

# Prepare actual demand data
actual_demands = np.array(results_dict["y_values"][1:])
# Prepare forecasted demand data
# forecasted_demands = np.array(results_dict["forecasted_demands"])

base_stock = np.array(results_dict["optimal_order_quantities"])


def demands_bstock_fbprophet(actual_demands, base_stock, iteration_number):
    _ = iteration_number * 10
    bstock = math.floor(base_stock[_])-1
    demands = []
    for ad in actual_demands[_]:
        ad_int = math.ceil(ad)
        x = tuple(np.random.randint(1, ad_int, size=(2,)))
        while sum(x) != ad_int:
            x = tuple(np.random.randint(1, ad_int, size=(2,)))
        demands.append(x)

    return demands, bstock


# demands, bstock = demands_bstock_fbprophet(actual_demands, base_stock, 0)
