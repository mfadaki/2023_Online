import numpy as np
import pandas as pd
import json
from json import JSONEncoder
import glob
import math

from bokeh.plotting import figure, output_file, show
from bokeh.models import NumeralTickFormatter, Span
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, Label

output_notebook()  # Show the graph just in the Jupyter notebook (inline)

# print(os.getcwd())

# path = "./results/0_No_Augm_in_MCTS/"
path = os.path.dirname(os.path.abspath(__file__)) + "/"
print(path)

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

prp = 10

# Get total time ...

profit_supplier_mcts = []
profit_supplier_prop = []
price = 10
penalty_per_shortage_1 = 200
penalty_per_shortage_2 = 200
fill_rate_target = 0.9


def flatten_list(regular_list):
    return [item for sublist in regular_list for item in sublist]


def proportional(s, x1, x2):
    return math.floor(min(x1, s*x1/(x1+x2))), math.floor(min(x2, s*x2/(x1+x2)))


for c in range(no_solions):
    with open(path + str(c) + ".json", "r") as read_file:
        sol_json = json.load(read_file)
        prop_allocations = [proportional(10, sol_json['Demands'][x][0], sol_json['Demands'][x][1]) for x in range(10)]
        for _ in range(prp):
            if _ == 9:
                # mtcs
                sum_allocations_t = sum(sol_json['Allocations'][_])

                sum_allocations = sum(flatten_list(sol_json['Allocations']))
                sum_demands = sum(flatten_list(sol_json['Demands']))

                profit_supplier_mcts.append(sum_allocations_t * price - max(0, fill_rate_target-(sum_allocations/sum_demands)) * penalty_per_shortage_1)

                # Proportional
                sum_allocations_t_prop = sum(prop_allocations[_])

                sum_allocations_prop = sum(flatten_list(prop_allocations))
                profit_supplier_prop.append(sum_allocations_t_prop * price - max(0, fill_rate_target-(sum_allocations_prop/sum_demands)) * penalty_per_shortage_1)

            else:
                # mtcs
                sum_allocations_t = sum(sol_json['Allocations'][_])
                profit_supplier_mcts.append(sum_allocations_t * price)
                # Proportional
                sum_allocations_t_prop = sum(prop_allocations[_])
                profit_supplier_prop.append(sum_allocations_t_prop * price)


######################

# Graph for MCTS
source1 = ColumnDataSource(data=dict(x=list(range(1, 101)), y1=profit_supplier_mcts))

profit_supplier_mcts_end_prp = [profit_supplier_mcts[_] for _ in range(101) if (_-9) % 10 == 0]
source3 = ColumnDataSource(data=dict(x=list(range(10, 101, 10)), y3=profit_supplier_mcts_end_prp))

Average_profit_mcts = sum(profit_supplier_mcts) / len(profit_supplier_mcts)


# p = figure(plot_width=400, plot_height=300, y_range=(0, 1))
p = figure(width=800, height=200, y_range=(-10, 115))

p.xaxis.axis_label = r"$$t$$"
p.yaxis.axis_label = r"$$\pi_t$$"

# add a line renderer
p.line(x='x', y='y1',
       source=source1,
       color='grey',
       line_width=2)
""" 
p.line(x='x', y='y3',
       source=source3,
       color='black',
       line_width=2,
       line_dash='dashed') """

p.circle(x='x', y='y1',
         source=source1,
         color='grey',
         size=5,
         fill_color='white')
# legend_label='MCTS')

p.scatter(x='x', y='y3',
          source=source3,
          color='red',
          marker="star",
          size=12,
          fill_color='red',
          legend_label='End of PRP')

h_line = Span(location=Average_profit_mcts, dimension='width', line_color='green', line_width=1, line_dash='dashed')


label_avr = Label(x=-12, y=68, text=r"$$\bar\pi = 65.2$$", text_color="green", text_font_size="10pt")

# p.add_layout(h_line, label_avr)
p.add_layout(label_avr)
p.add_layout(h_line)

p.x_range.range_padding = 0.3

p.xaxis.ticker = ticks = np.arange(0, 101, 10)
p.yaxis.ticker = ticks = np.arange(-10, 111, 20)

p.xgrid.visible = False
p.ygrid.visible = False

output_file("test.html", title="test example")
output_notebook()

# p.xaxis.formatter = NumeralTickFormatter(format='0 %')

p.legend.location = "top_right"
p.legend.label_text_font_size = "8px"
p.legend.label_height = 8
p.legend.label_width = 8

show(p)


##################### #####################


# Graph for Proportional

source1 = ColumnDataSource(data=dict(x=list(range(1, 101)), y1=profit_supplier_prop))

profit_supplier_prop_end_prp = [profit_supplier_prop[_] for _ in range(101) if (_-9) % 10 == 0]
source3 = ColumnDataSource(data=dict(x=list(range(10, 101, 10)), y3=profit_supplier_prop_end_prp))

Average_profit_prop = sum(profit_supplier_prop) / len(profit_supplier_prop)


# p = figure(plot_width=400, plot_height=300, y_range=(0, 1))
p = figure(width=800, height=200, y_range=(-10, 115))

p.xaxis.axis_label = r"$$t$$"
p.yaxis.axis_label = r"$$\pi_t$$"

# add a line renderer
p.line(x='x', y='y1',
       source=source1,
       color='grey',
       line_width=2)
""" 
p.line(x='x', y='y3',
       source=source3,
       color='black',
       line_width=2,
       line_dash='dashed') """

p.circle(x='x', y='y1',
         source=source1,
         color='grey',
         size=5,
         fill_color='white')
# legend_label='MCTS')

p.scatter(x='x', y='y3',
          source=source3,
          color='red',
          marker="star",
          size=12,
          fill_color='red',
          legend_label='End of PRP')

h_line = Span(location=Average_profit_prop, dimension='width', line_color='green', line_width=1, line_dash='dashed')


label_avr = Label(x=-12, y=83, text=r"$$\bar\pi = 81.6$$", text_color="green", text_font_size="10pt")

# p.add_layout(h_line, label_avr)
p.add_layout(label_avr)
p.add_layout(h_line)

p.x_range.range_padding = 0.3

p.xaxis.ticker = ticks = np.arange(0, 101, 10)
p.yaxis.ticker = ticks = np.arange(-10, 111, 20)

p.xgrid.visible = False
p.ygrid.visible = False

# output_file("test.html", title="test example")
output_notebook()

# p.xaxis.formatter = NumeralTickFormatter(format='0 %')

p.legend.location = "top_right"
p.legend.label_text_font_size = "8px"
p.legend.label_height = 8
p.legend.label_width = 8

show(p)
#####################
