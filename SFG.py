import pandas as pd
import pm4py
from datetime import timedelta
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from matplotlib.path import Path
import mplcursors
import warnings

import sys
sys.stdout.reconfigure(encoding='utf-8') # python3.11.exe -u .\DFG_shows_Speed.py | tee output.txt

warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(
    description = "These flags allow you to specify a Window of Recency for the generated DFG's Average Edge Speeds, with all events older than their sum being disregarded."
)

# File Path
parser.add_argument(
    "-p", "--path", nargs="?", default="./data/example.xes", const=True,
    help="The file path of the event log (.xes) to be graphed."
)

# Graphing Parameters
parser.add_argument(
    "-col", "--color", nargs="?", default="b",
    help="Whether the edge colors should indicate bottlenecks 'b', speed 's' or the representativeness of the displayed temporal distance 't'. Note: For non-self-looping edges, 't' assumes an amplitude of 0."
)

parser.add_argument(
    "-se", "--seed", default=42, type=int,
    help="Used only for the initial positions in the algorithm. Defaults to 42."
)
parser.add_argument(
    "-k", "--node_distance", default=1.5, type=float,
    help="Optimal distance between nodes [for layout generation]. [...] Increase this value to move nodes farther apart. → Fiddle to find optimal length:duration ratio."
)
parser.add_argument(
    "-i", "--layout_iterations", default=500, type=int,
    help="Maximum number of iterations taken [whilst generating the node layout.] → Fiddle to find optimal length:duration ratio."
)
parser.add_argument(
    "-o", "--optimize", nargs="?", default=False, const=True,
    help="If set, the optimal layout for seeds ∈ [0, <seed>-1], k ∈ [0.5, 4.5], and i ∈ [50, 1000] will be chosen. Finding it might take a while."
)

parser.add_argument(
    "-cyl", "--cycles", default=10, type=int,
    help="How many cycles of the sine wave should be fitted onto each edge."
)
parser.add_argument(
    "-af", "--amplitude_factor", default=1, type=float,
    help="A factor to scale the edge amplitudes & self-loop radii. If set, the transition's edge lengths won't consistently equate to the temporal distances anymore."
)

# Miscellaneous
parser.add_argument(
    "-c", "--compare", nargs="?", default=False, const=True,
    help="If set, PM4PY's standard visualization of the input event log will be output as well."
)

# Recency Interval
parser.add_argument(
    "-a", "--years", metavar="years", default=0, type=int,
    required=False#, help="The number of years to be considered."
)
parser.add_argument(
    "-m", "--months", metavar="months", default=0, type=int,
    required=False#, help="The number of months to be considered."
)
parser.add_argument(
    "-d", "--days", metavar="days", default=0, type=int,
    required=False#, help="The number of days to be considered."
)
parser.add_argument(
    "-hr", "--hours", metavar="hours", default=0, type=int,
    required=False#, help="The number of hours to be considered."
)
parser.add_argument(
    "-min", "--minutes", metavar="minutes", default=0, type=int,
    required=False#, help="The number of minutes to be considered."
)
parser.add_argument(
    "-s", "--seconds", metavar="seconds", default=0, type=int,
    required=False, help="To set the recency interval to be considered for the speed calculations."
)

args = parser.parse_args()
args.color = args.color[0]
if args.color not in ['s', 't']:
    args.color = 'b'



# \/\/\/ # Helper Functions # \/\/\/ #

def print_recency():
    recency = ''
    if args.years > 0:
        recency += f"{args.years} years "
    if args.months > 0:
        recency += f"{args.months} months "
    if args.days > 0:
        recency += f"{args.days} days "
    if args.hours > 0:
        recency += f"{args.hours} hours "
    if args.minutes > 0:
        recency += f"{args.minutes} minutes "
    if args.seconds > 0:
        recency += f"{args.seconds} seconds "

    return recency.strip()

def format_duration(seconds):
    if pd.isna(seconds):
        return "N/A"

    seconds = float(seconds)
    a, rem = divmod(int(seconds), 31536000)
    d, rem = divmod(rem, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    s += seconds - int(seconds)
    ms = 1000*(seconds - int(seconds))

    parts = []
    if a > 0:
        parts.append(f"{a} a")
    if d > 0:
        parts.append(f"{d} d")
    if h > 0:
        parts.append(f"{h} h")
    if m > 0:
        parts.append(f"{m} min")
    if int(s) > 0:
        parts.append(f"{s:.1f} s")
    elif ms > 0 or not parts:
        parts.append(f"{ms:.1f} ms")

    return ' '.join(parts)

def sine_arc_length(A, length, period):
    k = 2 * np.pi / period
    integrand = lambda x: np.sqrt(1 + (A * k * np.cos(k * x))**2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        arc_len, _ = quad(integrand, 0, length) #, limit=100

    return arc_len

amplitude_failures = {'count': 0}
def find_amplitude(length, desired_len, period):
    # Objective function: difference between computed arc length and desired length
    def objective(A):
        return sine_arc_length(A, length, period) - desired_len
    
    lo, hi = 0, length * 20
    f_lo = objective(lo)
    f_hi = objective(hi)
    if f_lo * f_hi > 0:
        #print(f"Could not find amplitude for length={length}, desired={desired_len}. Using fallback A=1.")
        amplitude_failures['count'] += 1
        return 1.0

    # Initial guesses: amplitude should be positive, start small
    result = root_scalar(objective, bracket=[0, length*20], method='brentq')
    if result.converged:
        return result.root
    else:
        raise ValueError("Failed to find amplitude.")

def sine_wave_points(start, end, period, amplitude, steps=1000):
    """Generates sine wave points between two positions."""
    if args.amplitude_factor == 0:  # Save resources/improve performance 👉👈
        steps = 2

    x0, y0 = start
    x1, y1 = end

    # Vector from start to end
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)

    # Base direction
    line = np.linspace(0, 1, steps)
    xs = x0 + dx * line
    ys = y0 + dy * line

    # Perpendicular vector (for sine wave displacement)
    perp_dx, perp_dy = -dy, dx
    perp_norm = np.hypot(perp_dx, perp_dy)
    perp_dx, perp_dy = perp_dx / perp_norm, perp_dy / perp_norm

    # Wave displacement
    wave = amplitude * np.sin(2 * np.pi * line / period)
    xs += wave * perp_dx
    ys += wave * perp_dy

    # Close curve (sometimes necessary due to rounding error?)
    xs[-1] = x1
    ys[-1] = y1
    dx, dy = x1 - xs[-2], y1 - ys[-2]
    length = np.hypot(dx, dy)
    #ratio = length #/ (math.sqrt(node_size) / 72 * 20)
    #xs[-1] -= dx*(math.sqrt(node_size) / 72)
    #ys[-1] -= dy*(math.sqrt(node_size) / 72)

    return xs, ys

def sine_wave_patch(start, end, period, amplitude, color, width):
    # Generate sine wave points
    xs, ys = sine_wave_points(start, end, period, amplitude)
    
    # Build a Path object from the points
    vertices = list(zip(xs, ys))
    codes = [Path.MOVETO] + [Path.CURVE3]*(len(vertices)-2) + [Path.LINETO]
    path = Path(vertices, codes)

    # Create a FancyArrowPatch from the path
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle='-|>',
        mutation_scale=10 + max_width * 1.2,  # width * 1.2 # Scales arrowhead size with width
        linewidth=width,
        color=color,
        alpha=alpha
    )
    return arrow

# /\/\/\ # Helper Functions # /\/\/\ #



# # Check your PM4Py Version
# print(pm4py.__version__)

#path = "./PM4Py_tutorial_2025-05-09/data/repairExample.xes"
#args.path = "./PM4Py_tutorial_2025-05-09/data/running_example.xes"
try:
    base_log = pm4py.read_xes(args.path)
except:
    print(f"ERROR 404: Please provide a valid .xes file path.")
    sys.exit()

#print(base_log.head(5))
#print(base_log.columns)



# # Statistic(s) on pandas Dataframe
try:
    first_event = base_log['time:timestamp'].min()
    last_event = base_log['time:timestamp'].max()
except:
    print(f"ABORTED: The input event log '{args.path}' doesn't contain timestamps.\n")
    sys.exit()
recency_window = timedelta(
    days    = args.days + 30*args.months + 365*args.years,
    hours   = args.hours,
    minutes = args.minutes,
    seconds = args.seconds
)
cut_off_time = last_event - recency_window
if cut_off_time == last_event:
        cut_off_time = first_event

# Print (base) DFG
if args.compare:
    dfg, start_activities, end_activities = pm4py.discover_dfg(base_log)
    pm4py.view_dfg(dfg, start_activities, end_activities)



# Filter pandas Dataframe
log = base_log[[
    'case:concept:name', 
    'concept:name', 
    'time:timestamp'#, 
    #'lifecycle:transition', 
    #'org:resource'
]]

# Rename columns
log = log.rename(columns={'case:concept:name': 'id', 'concept:name': 'node', 'time:timestamp': 'start_time'})

# Sort by trace & timestamp(/start_time)
log = log.sort_values(by=['id', 'start_time'])


# Add following node info
log['next'] = log.groupby('id')['node'].shift(-1)
log['end_time'] = log.groupby('id')['start_time'].shift(-1)
log['duration'] = (log['end_time'] - log['start_time']).dt.total_seconds() #/ 60

# Reorder columns
log = log[['id', 'node', 'next', 'duration', 'start_time', 'end_time']]

# Rename Trace ends to <sink>
log['next'] = log['next'].fillna('<sink>')
log['end_time'] = log['end_time'].fillna(log['start_time'])

# Add Trace starts as <start>
first_events = log.groupby('id').first().reset_index()
start_edges = pd.DataFrame({
    'id': first_events['id'],
    'node': '<start>',
    'next': first_events['node'],
    'duration': None,
    'start_time': first_events['start_time'],   # - timedelta(microseconds=1),
    'end_time': first_events['start_time']
})
log = pd.concat([start_edges, log], ignore_index=True)

# Resort <start>s into correct positions
log = log.sort_values(by=['id', 'start_time'])  # <start> listed first due to 'start_edges' being first in the concatenation

# Set duration for <start> & <sink> edges
log['duration'] = log['duration'].fillna(0)

# Filter for Recency
recent_log = log.loc[log['start_time'] >= cut_off_time]


# # Print finished Event Log(s)
# #log = log.sort_values(by=['node', 'next'])
# print(f"\n{log.head(20)}\n")
# print(recent_log.head(20))
# print(recent_log.columns)



# Aggregate edges --> Capacity/Throughput & Speed/Avg. Duration
edge_list = log.groupby(['node', 'next']).agg(
    frequency=('id', 'count'),
    overall_avg_duration=('duration', 'mean')
).reset_index()

recent_edge_list = recent_log.groupby(['node', 'next']).agg(
    #frequency=('id', 'count'),
    recent_avg_duration=('duration', 'mean')
).reset_index()

edge_list = edge_list.merge(recent_edge_list, how='left', on=['node', 'next'])
#edge_list = edge_list.sort_values(by=['node', 'frequency'], ascending=[True, False])


# Finalize edge properties (1/2), which don't yet require the node layout
most_frequent = edge_list['frequency'].max()
max_width = 10
if most_frequent > 1:
    #edge_list['width'] = round(edge_list['frequency']/most_frequent * max_width, 1)
    edge_list['width'] = round((edge_list['frequency']-1)/(most_frequent-1) * (max_width-1) + 1, 1)
else:
    edge_list['width'] = edge_list['frequency']         # Protect against 'NaN'
edge_list['avg_duration'] = edge_list['recent_avg_duration'].fillna(edge_list['overall_avg_duration'])
# # TEST # Instantaneous Self-Loop
# edge_list.loc[(edge_list['node'] == 'Repair (Complex)') | (edge_list['next'] == 'Repair (Complex)'), 'avg_duration'] = 0


edge_list['speed'] = 1 / edge_list['avg_duration'].replace(0, edge_list['avg_duration'].loc[edge_list['avg_duration'] > 0].min()) # Cap infinite <start> & <sink> speeds in order not to skew Normilization...

# Normalize Speed & Frequency to [0, 1]
edge_list['norm_speed'] = (edge_list['speed'] - edge_list['speed'].min()) / (edge_list['speed'].max() - edge_list['speed'].min())
if most_frequent > 1:
    edge_list['norm_freq'] = (edge_list['frequency'] - edge_list['frequency'].min()) / (most_frequent - edge_list['frequency'].min())
else:
    edge_list['norm_freq'] = edge_list['frequency']     # Protect against 'NaN'

# Bottleneck Score = high when slow & high throughput
edge_list['bottleneck_score'] = (1 - edge_list['norm_speed']) * edge_list['norm_freq']  # Slow & Frequent => High (/bad) Score

def score_to_color(score):
    cmap = plt.get_cmap("RdYlGn_r")  # Reversed: green (low) to red (high)
    rgba = cmap(score)
    return "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

edge_list['color'] = edge_list['bottleneck_score'].apply(score_to_color)        # Color Mapping for Bottleneck Detection
edge_list.loc[edge_list['recent_avg_duration'].isna(), 'color'] = '#7f7f7f'   # Not recent color
edge_list.loc[edge_list['avg_duration'] == 0, 'color'] = "#1e6dbb"            # Differentiate instantaneous edges
edge_list.loc[(edge_list['node'] == '<start>') | (edge_list['next'] == '<sink>'), 'color'] = "#000000"  # Restore <start> & <sink> color



# Print Summary of the Input Event Log & Recency Constraints
print(f"\n{args.path}\n\nThe Event Log spans {last_event-first_event}\n\
From: {first_event}\n\
To:   {last_event}\n")
recency_str = print_recency()
if len(recency_str) == 0:
    print(f"All of which will be considered, because no Recency constraints have been given.")
elif cut_off_time <= first_event:
    print(f"All of which will be considered, because the last {recency_window} include the entire event log.") # {recency_str}
else:
    print(f"Of which only the average speeds of the last {recency_window} will be considered, i.e.\nFrom: {cut_off_time}")

print(f"\nIt contains (recently):")
print(f"#Traces/Cases:   {log['id'].nunique()} ({recent_log['id'].nunique()})")             # Traces of EventLog <=> Cases of Dataframe
print(f"#Events:         {len(base_log)} ({len(base_log.loc[base_log['time:timestamp'] >= cut_off_time])})")
print(f"=> #Transitions: {len(log)} ({len(recent_log)})")                                   # Events => Transitions
print(f"=> #Edges:       {len(edge_list)}")                                                 # incl. from <start> & to <sink>
print(f"#Activities:     {log['next'].nunique()-1} ({recent_log['next'].nunique()-1})")     # Activities -1 for <sink>
print(f"=> #Nodes:       {log['next'].nunique()+1}\n")                                      # incl. <start> & <sink>

shortest_tag, longest_tag = "", ""
shortest_edge = edge_list.loc[edge_list['avg_duration'] > 0].min()
if len(recency_str) > 0 and not pd.isna(shortest_edge['recent_avg_duration']):
    shortest_tag = ", recently"
print(f"Shortest Edge:   {format_duration(shortest_edge['avg_duration'])} ({shortest_edge['node']} → {shortest_edge['next']}{shortest_tag})")
longest_edge = edge_list.loc[edge_list['avg_duration'] > 0].max()
if len(recency_str) > 0 and not pd.isna(longest_edge['recent_avg_duration']):
    longest_tag = ", recently"
print(f"Longest Edge:    {format_duration(longest_edge['avg_duration'])} ({longest_edge['node']} → {longest_edge['next']}{longest_tag})\n")



# Start graphing the Event Log
G = nx.DiGraph()
node_size=1#600
alpha = 0.75

# Add all edges incl. their attributes from the 'edge_list'
for _, row in edge_list.iterrows():
    G.add_edge(
        row['node'], 
        row['next'], 
        width=row['width'], 
        color=row['color'], 
        length=row['avg_duration']
    )



# Compute node layout <~ edge lengths                                 # TODO: make weighting stronger...
# Convert 'length' to float for use in spring_layout weights
lengths = nx.get_edge_attributes(G, 'length')


if args.optimize:
    overall_best = None
    for s in range(0,args.seed):
        print(f"\n\n\n### SEED = {s} ###")
        overall_changes = 0
        for k in range(1,10):
            print(f"\n>>> SEED = {s}, k = {k/2} >>>")
            non_new = 0
            this_best = None
            this_changes = 0
            for it in range(1,21):
                try:
                    pos = nx.spring_layout(G, weight='weight', k=k/2, iterations=it*50, scale=2, seed=s)
                except:
                    print(f"k = {k/2}, it = {50*it}\n==> ERROR")

                # Edge lengths
                edge_list['length'] = edge_list.apply(
                    lambda row: math.dist(pos[row['node']], pos[row['next']]),
                    axis=1
                )
                edge_list['length_ratio'] = edge_list['avg_duration'] / edge_list['length']
                min_length_ratio = edge_list['length_ratio'].loc[edge_list['length_ratio'] > 0].min()
                edge_list['length_ratio'] /= min_length_ratio                               # Normalize
                edge_list.loc[edge_list['length_ratio'] == 0, 'length_ratio'] = 1           # Fix instantaneous edges

                #edge_list['desired_length'] = edge_list['length_ratio']*edge_list['length']
                edge_list['desired_length'] = edge_list['avg_duration']/min_length_ratio
                edge_list.loc[edge_list['avg_duration'] == 0, 'desired_length'] = edge_list['length']   # Fix instantaneous edges
                edge_list.loc[edge_list['desired_length'] == 0, 'desired_length'] = 1                   # Fix instantaneous self-loops

                edge_list.loc[edge_list['next'] == edge_list['node'], 'length_ratio'] = (               # Fix (scaled) self-loops
                    edge_list['desired_length'] / np.maximum(edge_list['desired_length']*args.amplitude_factor, 1.0)
                )

                # # Sine Wave parameters
                # edge_list['period'] = edge_list['length'] / args.cycles
                # try:
                #     edge_list['amplitude'] = edge_list.loc[edge_list['node'] != edge_list['next']].apply(
                #         lambda row: find_amplitude(row['length'], row['desired_length'], row['period']),
                #         axis=1
                #     )
                # except:
                #     print(f"ERR xxx k = {k/2}, it = {50*it}")
                #     continue

                max_row = edge_list.loc[edge_list['length_ratio'].idxmax()]
                if this_best is None or max_row['length_ratio'] < this_best[0]['length_ratio']:
                    this_best = (max_row, s, k, it)
                    label = "NEW"
                    if overall_best is None or max_row['length_ratio'] < overall_best[0]['length_ratio']:
                        overall_best = this_best
                        overall_changes = overall_changes + 1
                        this_changes = this_changes + 1
                        label = "OVERALL"
                    print(f"{label} <<< k = {k/2}, it = {50*it} ==> {max_row['node']} --> {max_row['next']} with {100/max_row['length_ratio']:.1f}% length")
                else:
                    print(f"OLD ||| k = {k/2}, it = {50*it} ==> {max_row['node']} --> {max_row['next']} with {100/max_row['length_ratio']:.1f}% length")
                    non_new = non_new + 1
                    # >= 5
                    if non_new >= 1\
                    or (this_changes < 1 and max_row['length_ratio'] > overall_best[0]['length_ratio']/0.9):
                        break

            best_row, best_seed, best_k, best_it = this_best
            print(f"seed = {s}, k = {best_k/2}, it = {50*best_it}\n==> {best_row['node']} --> {best_row['next']}\nwith {100/best_row['length_ratio']:.1f}% /// {100/overall_best[0]['length_ratio']:.1f}% length ")

            if overall_changes < 1 and this_best[0]['length_ratio'] > overall_best[0]['length_ratio']/0.5:
               continue

    _, args.seed, args.node_distance, args.layout_iterations = overall_best
    args.node_distance = args.node_distance/2
    args.layout_iterations = 50*args.layout_iterations
    print("\n")


# # Use inverse length as weight to increase spacing for longer durations
# inv_lengths = {k: 1 / v if v > 0 else 0.01 for k, v in lengths.items()}
# inv_squared_lengths = {k: 1 / (v ** 2) if v > 0 else 0.01 for k, v in lengths.items()}

# nx.set_edge_attributes(G, inv_lengths, 'weight')
# nx.set_edge_attributes(G, inv_squared_lengths, 'weight')

#pos = nx.spring_layout(G, weight='weight', k=0.5, iterations=100, scale=2, seed=42)
#pos = nx.spring_layout(G, weight='weight', k=1.5, iterations=500, scale=2, seed=42)
print(f"Rendering for seed = {args.seed}, k = {args.node_distance}, and i = {args.layout_iterations}...")
pos = nx.spring_layout(G, weight='weight', k=args.node_distance, iterations=args.layout_iterations, scale=2, seed=args.seed)


# # Normalize durations for better behavior (avoid huge values)
# max_len = max(lengths.values())
# lengths_normalized = {k: v / max_len for k, v in lengths.items()}
# pos = nx.kamada_kawai_layout(G, dist=lengths_normalized)



# Finalize edge properties (2/2), now based on node layout
#edge_list = edge_list[['node', 'next', 'width', 'avg_duration', 'color']]
# Edge lengths
edge_list['length'] = edge_list.apply(
    lambda row: math.dist(pos[row['node']], pos[row['next']]),
    axis=1
)
edge_list['length_ratio'] = edge_list['avg_duration'] / edge_list['length']
min_length_ratio = edge_list['length_ratio'].loc[edge_list['length_ratio'] > 0].min()
edge_list['length_ratio'] /= min_length_ratio                               # Normalize
edge_list.loc[edge_list['length_ratio'] == 0, 'length_ratio'] = 1           # Fix instantaneous edges

#edge_list['desired_length'] = edge_list['length_ratio']*edge_list['length']
edge_list['desired_length'] = edge_list['avg_duration']/min_length_ratio
edge_list.loc[edge_list['avg_duration'] == 0, 'desired_length'] = edge_list['length']   # Fix instantaneous edges
edge_list.loc[edge_list['desired_length'] == 0, 'desired_length'] = 1                   # Fix instantaneous self-loops

edge_list.loc[edge_list['next'] == edge_list['node'], 'length_ratio'] = (               # Fix (scaled) self-loops
    edge_list['desired_length'] / np.maximum(edge_list['desired_length']*args.amplitude_factor, 1.0)
)

# Sine Wave parameters
edge_list['period'] = edge_list['length'] / args.cycles
edge_list['amplitude'] = edge_list.loc[edge_list['node'] != edge_list['next']].apply(
    lambda row: find_amplitude(row['length'], row['desired_length'], row['period']),
    axis=1
)
if amplitude_failures['count'] > 0:
    print(f"ATTENTION: Could not compute amplitude for {amplitude_failures['count']}/{len(edge_list)} edges... ({(100*amplitude_failures['count']/len(edge_list)):.1f}%) → Using fallback A=1.")


# Apply '--amplitude_factor' / straighten edges
edge_list['amplitude'] *= args.amplitude_factor
if args.amplitude_factor == 0:
    edge_list['desired_length'] = 1

# Alternatively apply '--color' for Speed or Representativeness of Temp. Distance 
if args.color != 'b':
    if args.color == 's':   # Speed based on Duration
        min_duration = edge_list['recent_avg_duration'].loc[edge_list['recent_avg_duration'] > 0].min()
        edge_list['slowness_score'] = ((edge_list['recent_avg_duration'] - min_duration) / (edge_list['recent_avg_duration'].max() - min_duration))
        #edge_list['slowness_score'] = ((edge_list['avg_duration'] - edge_list['avg_duration'].min()) / (edge_list['avg_duration'].max() - edge_list['avg_duration'].min()))

        # I have no clue why this \/ looks so red...
        #edge_list['slowness_score'] = 1 - ((edge_list['speed'] - edge_list['speed'].min()) / (edge_list['speed'].max() - edge_list['speed'].min()))
        ##edge_list['color'] = edge_list['slowness_score'].apply(bottleneck_to_color)

        color_labels = ['──────────────', 'Fastest', 'Fast', 'Medium', 'Slow', 'Slowest']
    elif args.color == 't':
        max_length_ratio = edge_list['length_ratio'].loc[edge_list['length_ratio'] < math.inf].max()
        edge_list['slowness_score'] = ((edge_list['length_ratio'] - 1) / (max_length_ratio - 1))

        color_labels = ['─────────────────────', 'Fully representative', 'Pretty representative', 'Semi representative', 'Not very representative', 'Least representative']

    edge_list.loc[(edge_list['color'] != '#000000') & (edge_list['color'] != '#1e6dbb') & (edge_list['color'] != '#7f7f7f'), 'color'] = edge_list['slowness_score'].apply(score_to_color)   # Preserve special colors


# # Print combined Edge List
# edge_list = edge_list[['node', 'next', 'frequency', 'width', 'avg_duration', 'length', 'length_ratio', 'desired_length', 'period', 'amplitude', 'bottleneck_score','color']]
# #edge_list = edge_list[['node', 'next', 'width', 'desired_length', 'color', 'period', 'amplitude']]
# print(edge_list.head(40))
# print(edge_list.columns)



# # # Standard NetworkX Graphing # #
# # Draw nodes & labels
# nx.draw_networkx_nodes(G, pos, node_size=600, node_color='white', edgecolors='black')
# nx.draw_networkx_labels(G, pos, font_size=10)

# # Get edge attributes
# edges = G.edges(data=True)
# edge_colors = [d['color'] for _, _, d in edges]
# edge_widths = [d['width'] for _, _, d in edges]

# # Draw edges with custom width & color
# nx.draw_networkx_edges(
#     G, pos,
#     width=edge_widths,
#     edge_color=edge_colors,
#     arrows=True,
#     arrowstyle='-|>',
#     arrowsize=15,
#     alpha=alpha
# )

# plt.title('Task 5 – Capacity & Speed for DFGs')
# plt.axis('off')
# plt.gca().set_aspect('equal', adjustable='datalim') # Keep scaling of x- & y-axes consistent
# plt.tight_layout()
# plt.show()



# # Custom Edge Graphing # #
fig, ax = plt.subplots()

# Prepare edges
patches = []
for i, row in edge_list.iterrows():
    start = pos[row['node']].copy()
    end = pos[row['next']].copy()

    if args.amplitude_factor == 0 and row['node'] != row['next']:  # Shift 'there & back again' edge pairs     # TODO: Arrow heads may disconnect...
        reverse_exists = ((edge_list['node'] == row['next']) & (edge_list['next'] == row['node'])).any()
        
        if reverse_exists:
            shift_value = 0.01
            if i < edge_list[(edge_list['node'] == row['next']) & (edge_list['next'] == row['node'])].index[0]:
                start[0] -= shift_value
                end[0] -= shift_value
            else:
                start[0] += shift_value
                end[0] += shift_value

    if len(recency_str) > 0:
        recent = 'none'
        if pd.notna(row.get('recent_avg_duration')):
            recent = format_duration(row.get('recent_avg_duration')) # f"{row.get('recent_avg_duration'):.1f} s"
        label = f"\
{row['node']} → {row['next']}\n\n\
All-time Avg. Duration: {format_duration(row.get('overall_avg_duration', 'N/A'))}\n\
Recent Avg. Duration: {recent}\n\
Frequency: {row.get('frequency', 'N/A')}\
"
    else:
        label = f"\
{row['node']} → {row['next']}\n\n\
Avg. Duration: {format_duration(row.get('avg_duration', 'N/A'))}\n\
Frequency: {row.get('frequency', 'N/A')}\
"

    if row['node'] == row['next']:  # Self-Loop     => Circle
        radius = max([row['desired_length']*args.amplitude_factor, 1.0]) / (2 * np.pi)
        center_x, center_y = start[0] + radius, start[1]    # Shifted to the right for visibility

        # # Circles don't work with mplcursors' hover texts
        # loop = Circle(
        #     (center_x, center_y),   # Circle center (offset from node)
        #     radius=radius,
        #     edgecolor=row['color'],
        #     facecolor='none',
        #     linewidth=row['width'],
        #     alpha=alpha
        # )
        # ax.add_patch(loop)

        theta = np.linspace(0, 2 * np.pi, 100)
        xs = center_x + radius * np.cos(theta)
        ys = center_y + radius * np.sin(theta)

        loop_line = Line2D(xs, ys, color=row['color'], linewidth=row['width'], alpha=alpha)
        loop_line.set_picker(True)
        loop_line.tooltip_text = label
        ax.add_line(loop_line)
        patches.append(loop_line)
    else:                           # Regular Edge  => Sine Wave
        # Fancy Arrow Heads doesn't work with mplcursors' hover texts
        arrow = sine_wave_patch(start, end, row['period'], row['amplitude'], row['color'], row['width'])
        ax.add_patch(arrow)

        # Line2D does, however, work with mplcursors' hover texts --> same line but invisible
        xs, ys = sine_wave_points(start, end, row['period'], row['amplitude'])
        #print(f"{start} --> {end}:\n{xs[0]} & {ys[0]} -{len(xs)}-> {xs[-1]} & {ys[-1]}")
        line = Line2D(xs, ys, color=row['color'], linewidth=row['width'], alpha=0)
        line.set_picker(True)
        line.tooltip_text = label
        ax.add_line(line)
        patches.append(line)

        # # Fake arrowhead
        # dx = xs[-1] - xs[-2]
        # dy = ys[-1] - ys[-2]
        # norm = np.hypot(dx, dy)
        # dx /= norm
        # dy /= norm

        # arrow_length = 0.04
        # arrow_width = 0.02
        # x_base = xs[-1] - dx * arrow_length
        # y_base = ys[-1] - dy * arrow_length
        # px, py = -dy, dx

        # arrowhead = np.array([
        #     [xs[-1], ys[-1]],
        #     [x_base + px * arrow_width, y_base + py * arrow_width],
        #     [x_base - px * arrow_width, y_base - py * arrow_width]
        # ])
        # arrow_patch = plt.Polygon(arrowhead, closed=True, color=row['color'], alpha=alpha)
        # ax.add_patch(arrow_patch)

# Enable/show hover tooltips
cursor = mplcursors.cursor(patches, hover=True)
@cursor.connect("add")
def on_add(sel):
    sel.annotation.set_text(sel.artist.tooltip_text)
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
    sel.annotation.set_visible(True)

    # Save reference to this selection
    ax._last_annotation = sel.annotation

# .. and make them disappear again
def on_motion(event):
    # If the mouse is not over any patch, hide the annotation
    if not ax.contains(event)[0]:
        if hasattr(ax, "_last_annotation") and ax._last_annotation:
            ax._last_annotation.set_visible(False)
            event.canvas.draw_idle()
fig.canvas.mpl_connect("motion_notify_event", on_motion)


nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='white', edgecolors='black', ax=ax)
nx.draw_networkx_labels(
    G, pos, font_size=10,
    verticalalignment='center',
    horizontalalignment='center',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
    ax=ax
)


# Add legend for different Edge Colors
if args.color == 'b':
    color_labels = ['────────────────────────', 'Excellent (fast/infrequent)', 'Good', 'Medium', 'Bad', 'Bottleneck (slow & frequent)']
legend_elements = [
    Line2D([0], [0], color='#000000', lw=4, label='Start/Sink'),
    Line2D([0], [0], color='#1e6dbb', lw=4, label='Instantaneous'),
    #Line2D([0], [0], color='#7f7f7f', lw=4, label='No recent data'),
    #Line2D([0], [0], alpha=0),
    Line2D([0], [0], color='gray', lw=1, linestyle='--', label=color_labels[0]),
    Line2D([0], [0], color=score_to_color(0.0), lw=4, label=color_labels[1]),
    Line2D([0], [0], color=score_to_color(0.25), lw=4, label=color_labels[2]),
    Line2D([0], [0], color=score_to_color(0.5), lw=4, label=color_labels[3]),
    Line2D([0], [0], color=score_to_color(0.75), lw=4, label=color_labels[4]),
    Line2D([0], [0], color=score_to_color(1.0), lw=4, label=color_labels[5])
]
if len(recency_str) > 0:
    legend_elements += [
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label=color_labels[0]),
        Line2D([0], [0], color='#7f7f7f', lw=4, label='No recent data'),
        Line2D([0], [0], alpha=0, label=f"Last: {recency_window} ({min([100*recency_window/(last_event-first_event), 100]):.1f}%)")
    ]
max_row = edge_list.loc[edge_list['length_ratio'].idxmax()]
legend_elements += [
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label=color_labels[0]),
        Line2D([0], [0], alpha=0, label=f"seed = {args.seed}"),
        Line2D([0], [0], alpha=0, label=f"k = {args.node_distance}"),
        Line2D([0], [0], alpha=0, label=f"iterations = {args.layout_iterations}"),
        Line2D([0], [0], alpha=0, label=f"{max_row['node']} → {max_row['next']}\nwith {100/max_row['length_ratio']:.1f}% length"),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label=color_labels[0]),
        Line2D([0], [0], alpha=0, label=f"Cycles per Edge ≈ {args.cycles}"),
        Line2D([0], [0], alpha=0, label=f"Amplitude Factor = {args.amplitude_factor}")
    ]
plt.legend(handles=legend_elements, title='Edge Colors & More', loc='upper left')



# Final plot settings
ax.set_aspect('equal', adjustable='datalim')
ax.axis('off')
plt.title('Task 5 – Capacity & Speed for DFGs')
plt.tight_layout()
plt.show()



# TODO:
# * Calculating the period immensly suffers from rounding error
#   -> Sine Waves don't perfectly connect to 'next' nodes
# * Arrow heads are sometimes hardly visible
# * When zooming in, some arrow heads don't connect to nodes due to:
#   1. NetworkX node (labels) x custom plotted edge paths
#   2. 'there & back again' shifting