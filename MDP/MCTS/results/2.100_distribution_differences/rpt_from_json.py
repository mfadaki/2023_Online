import numpy as np
# import scipy as cp
import pandas as pd
# import math
# import sys
# import os
import json
from json import JSONEncoder
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from bokeh.plotting import figure, output_file, show
from bokeh.models import NumeralTickFormatter
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, ranges, LabelSet
output_notebook()  # Show the graph just in the Jupyter notebook (inline)

# print(os.getcwd())

# path = "./results/0_No_Augm_in_MCTS/"
path = os.path.dirname(os.path.abspath(__file__)) + "/"

# Selct the most recent JSON file in the 'results' directory


def no_sol_fn():
    import glob
    import os

    list_of_files = glob.glob(path + "*.json")
    latest_json_file = max(list_of_files, key=os.path.getctime)
    print(latest_json_file)

    # No of Jsons
    latest_json_file_filename_with_extention = os.path.basename(latest_json_file)
    latest_json_file_filename_without_extention = os.path.splitext(
        latest_json_file_filename_with_extention
    )[0]
    no_solions = int(latest_json_file_filename_without_extention)
    return no_solions


no_solions = no_sol_fn()+1
print(no_solions)

no_solions = 100  # Use it if we want to manually set the no of solutions

# Get total time ...

fr_1_mcts_arr = []
fr_2_mcts_arr = []

fr_1_prop_arr = []
fr_2_prop_arr = []

deviations_arr = []

for c in range(no_solions):
    with open(path + str(c) + ".json", "r") as read_file:
        sol_json = json.load(read_file)
        fr_1_mcts_arr.append(sol_json['FR1'])
        fr_2_mcts_arr.append(sol_json['FR2'])
        fr_1_prop_arr.append(sol_json['FR1_prop'])
        fr_2_prop_arr.append(sol_json['FR2_prop'])

        dev = [0]*10
        for _ in range(10):
            if sum(sol_json['Demands'][_]) >= 10:
                if sum(sol_json['Allocations'][_]) == 10:
                    shortage = 0
                else:
                    shortage = 10 - sum(sol_json['Allocations'][_])
            else:
                if sum(sol_json['Demands'][_]) == sum(sol_json['Allocations'][_]):
                    shortage = 0
                else:
                    shortage = sum(sol_json['Demands'][_]) - sum(sol_json['Allocations'][_])
            dev[_] = shortage
        deviations_arr.append(dev)

        # deviations_arr.append([sum(sol_json['Demands'][x])-sum(sol_json['Allocations'][x]) for x in range(10)])

dev_frq = [0]*10
for y in range(10):
    for x in range(100):
        dev_frq[y] += deviations_arr[x][y]

print(dev_frq)

dev_frq_ready = [0]*10
for y in range(10):
    for x in range(100):
        if deviations_arr[x][y] > 0:
            dev_frq_ready[y] += 1

print(dev_frq_ready)
######################

dev_frq = [367, 261, 189, 143, 95, 65, 43, 16, 6, 0]
# dev_frq_prc = [round(x/sum(dev_frq), 2) for x in dev_frq]

x_categories = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# colors =  ["#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
colors = ["#D3D3D3"]*10

p = figure(x_range=x_categories, y_range=ranges.Range1d(start=0, end=400), width=400, height=300, title="")


p.xaxis.axis_label = 'Time'
p.yaxis.axis_label = 'MCST Inefficiency-Driven Shortage'

p.vbar(x=x_categories, top=dev_frq, width=0.4,
       line_color='black', color=colors, alpha=[0.99-x*0.055 for x in range(10)], line_width=1)

p.xgrid.grid_line_color = None
p.y_range.start = 0

p.xgrid.visible = False
p.ygrid.visible = False

p.yaxis.ticker = ticks = np.arange(0, 401, 100)
p.xaxis.major_tick_line_color = None

output_file("test.html", title="test example")
output_notebook()
show(p)

#####
