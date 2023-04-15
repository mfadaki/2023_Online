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
from bokeh.models import ColumnDataSource
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

no_solions = 10  # Use it if we want to manually set the no of solutions

# Get total time ...

fr_1_mcts_arr = []
fr_2_mcts_arr = []

fr_1_prop_arr = []
fr_2_prop_arr = []

for c in range(no_solions):
    with open(path + str(c) + ".json", "r") as read_file:
        sol_json = json.load(read_file)
        fr_1_mcts_arr.append(sol_json['FR1'])
        fr_2_mcts_arr.append(sol_json['FR2'])
        fr_1_prop_arr.append(sol_json['FR1_prop'])
        fr_2_prop_arr.append(sol_json['FR2_prop'])

######################

player = 2

if player == 1:
    source1 = ColumnDataSource(data=dict(x=list(range(1, no_solions+1)), y1=fr_1_mcts_arr))
    source2 = ColumnDataSource(data=dict(x=list(range(1, no_solions+1)), y2=fr_1_prop_arr))
else:
    source1 = ColumnDataSource(data=dict(x=list(range(1, no_solions+1)), y1=fr_2_mcts_arr))
    source2 = ColumnDataSource(data=dict(x=list(range(1, no_solions+1)), y2=fr_2_prop_arr))

# p = figure(plot_width=400, plot_height=300, y_range=(0, 1))
p = figure(width=400, height=300, y_range=(0, 1))

p.xaxis.axis_label = r"$${\tiny\#} \tau_s$$"
p.yaxis.axis_label = r"$$\beta_2$$"

# add a line renderer
p.line(x='x', y='y1',
       source=source1,
       color='black',
       line_width=2)

p.line(x='x', y='y2',
       source=source2,
       color='black',
       line_width=2,
       line_dash='dashed')

p.circle(x='x', y='y1',
         source=source1,
         color='black',
         size=7,
         fill_color='white',
         legend_label='MCTS')

p.triangle(x='x', y='y2',
           source=source2,
           color='black',
           size=7,
           fill_color='white',
           legend_label='Proportional')

p.xaxis.ticker = ticks = np.arange(0, 11, 1)
p.yaxis.ticker = ticks = np.arange(0, 1.1, 0.1)

p.xgrid.visible = False
p.ygrid.visible = False

output_file("test.html", title="test example")
output_notebook()

# p.xaxis.formatter = NumeralTickFormatter(format='0 %')

p.legend.location = "bottom_right"
p.legend.label_text_font_size = "8px"
p.legend.label_height = 8
p.legend.label_width = 8

show(p)
#####################
