# Instructions:
# Change the path
# If number of solutions in the first function is not right, set the number of json solutions manually in line after 40
# change the parameter name in the last function. for n-ij, we need to select nd_ijt


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import scipy as cp
# import pandas as pd
# import math
# import sys
# import os
import json
from json import JSONEncoder

from bokeh.plotting import figure, output_file, show
from bokeh.models import NumeralTickFormatter, Span
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, Label
output_notebook()  # Show the graph just in the Jupyter notebook (inline)


path = "./results/"

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

base_stock_arr = []

for c in range(no_solions):
    with open(path + str(c) + ".json", "r") as read_file:
        sol_json = json.load(read_file)
        fr_1_mcts_arr.append(sol_json['FR1'])
        fr_2_mcts_arr.append(sol_json['FR2'])
        base_stock_arr.append(sol_json['s'])


######################

source1 = ColumnDataSource(data=dict(x=list(range(no_solions)), y1=fr_1_mcts_arr))
source2 = ColumnDataSource(data=dict(x=list(range(no_solions)), y2=fr_2_mcts_arr))


# p = figure(plot_width=400, plot_height=300, y_range=(0, 1))
p = figure(width=400, height=300, y_range=(0, 1))

p.xaxis.axis_label = r"$${\tiny\#} \tau_s$$"
p.yaxis.axis_label = r"$$\beta$$"

# add a line renderer
p.line(x='x', y='y1',
       source=source1,
       color='black',
       line_width=2)

p.line(x='x', y='y2',
       source=source2,
       color='black',
       line_width=2)

p.circle(x='x', y='y1',
         source=source1,
         color='black',
         size=7,
         fill_color='white',
         legend_label='Retailer 1')

p.triangle(x='x', y='y2',
           source=source2,
           color='black',
           size=7,
           fill_color='white',
           legend_label='Retailer 2')

p.xaxis.ticker = ticks = np.arange(0, 10, 1)
p.yaxis.ticker = ticks = np.arange(0, 1.1, 0.1)

p.xgrid.visible = False
p.ygrid.visible = False

h_line = Span(location=0.85, dimension='width', line_color='green', line_width=1, line_dash='dashed')
p.add_layout(h_line)

label_avr = Label(x=0, y=0.68, text=r"$$\widehat{\beta} = 85\%$$", text_color="green", text_font_size="10pt")
p.add_layout(label_avr)

output_file("test.html", title="test example")
output_notebook()

# p.xaxis.formatter = NumeralTickFormatter(format='0 %')

p.legend.location = "bottom_right"
p.legend.label_text_font_size = "8px"
p.legend.label_height = 8
p.legend.label_width = 8

show(p)


#####################
# Graph Step wise Base Stock Level

x_axis_values = [0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 9]

base_stock_arr_updated = [13, 13, 14, 14, 14, 14, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18]

source1 = ColumnDataSource(data=dict(x=x_axis_values, y1=base_stock_arr_updated))
# source2 = ColumnDataSource(data=dict(x=list(range(no_solions)), y2=fr_2_mcts_arr))


# p = figure(plot_width=400, plot_height=300, y_range=(0, 1))
p = figure(width=400, height=300, y_range=(0, 20))

p.xaxis.axis_label = r"$${\tiny\#} \tau_s$$"
p.yaxis.axis_label = r"$$\text{Base-Stock Level} \, (S)$$"

# add a line renderer
p.line(x='x', y='y1',
       source=source1,
       color='black',
       line_width=2)

p.circle(x='x', y='y1',
         source=source1,
         color='black',
         size=7,
         fill_color='white'
         )


p.xaxis.ticker = ticks = np.arange(0, 10, 1)
p.yaxis.ticker = ticks = np.arange(0, 21, 5)

p.xgrid.visible = False
p.ygrid.visible = False

# h_line = Span(location=0.85, dimension='width', line_color='green', line_width=1, line_dash='dashed')
# p.add_layout(h_line)

# label_avr = Label(x=0, y=0.68, text=r"$$\widehat{\beta} = 85\%$$", text_color="green", text_font_size="10pt")
# p.add_layout(label_avr)

# output_file("test.html", title="test example")
# output_notebook()

# p.xaxis.formatter = NumeralTickFormatter(format='0 %')

p.legend.location = "bottom_right"
p.legend.label_text_font_size = "8px"
p.legend.label_height = 8
p.legend.label_width = 8

show(p)

#####################
# Graph Demands and FB Prophet

# Read the data from the saved JSON file
with open('results_fbprophet.json', 'r') as infile:
    results_dict = json.load(infile)

actual_demands = np.array(results_dict['y_values'][1:]).flatten()
forecasted_demands = np.array(results_dict['forecasted_demands']).flatten()

x_axis_val = list(range(len(actual_demands)))

source1 = ColumnDataSource(data=dict(x=x_axis_val, y1=actual_demands))
source2 = ColumnDataSource(data=dict(x=x_axis_val, y2=forecasted_demands))


# p = figure(plot_width=400, plot_height=300, y_range=(0, 1))
p = figure(width=400, height=300, y_range=(14, 20))

p.xaxis.axis_label = r"$$t$$"
p.yaxis.axis_label = "Aggregate Demand"

# add a line renderer

# add a line renderer
""" p.line(x='x', y='y1',
       source=source1,
       color='black',
       line_width=2) """


p.circle(x='x', y='y1',
         source=source1,
         color='grey',
         size=7,
         fill_color='white',
         legend_label='Aggregate Demand')

p.line(x='x', y='y2',
       source=source2,
       color='black',
       line_width=2,
       legend_label='Forecasted Demand')

""" p.triangle(x='x', y='y2',
           source=source2,
           color='black',
           size=7,
           fill_color='white',
           legend_label='Retailer 2') """


# p.xaxis.ticker = ticks = np.arange(0, 10, 1)
# p.yaxis.ticker = ticks = np.arange(0, 21, 5)

p.xgrid.visible = False
p.ygrid.visible = False

# h_line = Span(location=0.85, dimension='width', line_color='green', line_width=1, line_dash='dashed')
# p.add_layout(h_line)

# label_avr = Label(x=0, y=0.68, text=r"$$\widehat{\beta} = 85\%$$", text_color="green", text_font_size="10pt")
# p.add_layout(label_avr)

output_file("test.html", title="test example")
output_notebook()

# p.xaxis.formatter = NumeralTickFormatter(format='0 %')

p.legend.location = "bottom_right"
p.legend.label_text_font_size = "8px"
p.legend.label_height = 8
p.legend.label_width = 8

show(p)
