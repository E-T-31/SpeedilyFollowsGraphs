# Speedily Follows Graphs

_Speedily Follows Graphs (SFGs)_ are a novel variant of _Directly Follows Graphs (DFGs)_ for displaying event logs with the additional notion of **speed** alongside **capacity**, as in transition frequency. Additionally, potential **bottlenecks** can be detected by combining these two notions.

In practice, this implementation directly maps the temporal distances between events as the edge lengths between their respective nodes. To do this, it folds the edges periodically in the form of sine waves, if required by the node layout.
This implementation further uses edge color to highlight potential bottlenecks, i.e. slow transitions of high frequencies, where a lot of traces lose time.

## Usage
The Python script can be run with `python3.11.exe .\SFG.py` on Windows and will graph the event log provided in `./data/example.xes` by default.


### General Parameters
| Argument | Flag | Description | Default |
|-------|--------|---------|---------|
| `--path` | `-p` | The file path of the event log (.xes) to be graphed. | `./data/example.xes` |
| `--compare` | `-c` | If set, _PM4PY's_ standard visualization of the input event log will be output as well. | _False_ |

Example >>> `python3.11.exe .\SFG.py -p .\data\example.xes -c`


### Graphing Parameters

Used to fine-tune the appearance of the SFG in regard to its sine waves, as well as whether it should highlight bottlenecks, display speed directly, or show the representativeness of an edge's node-to-node length through its edge color.

| Argument | Flag | Description | Default |
|-------|--------|---------|---------|
| `--color` | `-col` | Whether the edge colors should indicate bottlenecks `b`, speed `s` or the representativeness of the displayed temporal distance `t`.<br>_Note:_ For non-self-looping edges, `t` assumes an amplitude of 0. | `b` |
| `--cycles` | `-cyl` | How many cycles of the sine wave should be fitted onto each edge. | `10` |
| `--amplitude_factor` | `-af` | A factor to scale the edge amplitudes & self-loop radii. If set, the transition's edge lengths won't consistently equate to the temporal distances anymore. | `1` |

Example >>> `python3.11.exe .\SFG.py -p .\data\example.xes -af 0 -col speed`


### Layouting Parameters

Used to guide the generation of a _NetworkX_ spring layout. The fourth flag allows a range of spring layouts to be probed for the best one, i.e. the one with the highest _lowest node distance to temporal distance ratio_. Said ratio determines how (un)representative the lengths of straight edges would be and, thus, how big the sine wave amplitudes need to be, with higher amplitudes leading to a less legible graph.

| Argument | Flag | Description | Default |
|-------|--------|---------|---------|
| `--seed` | `-se` | Used only for the initial positions in the algorithm. | `42` |
| `--node_distance` | `-k` | Optimal distance between nodes [for layout generation]. [...] Increase this value to move nodes farther apart. | `1.5` |
| `--layout_iterations` | `-i` | Maximum number of iterations taken [whilst generating the node layout.] | `500` |
| `--optimize` | `-o` | If set, the optimal layout for seeds ∈ [0, \<seed>-1], k ∈ [0.5, 4.5], and i ∈ [50, 1000] will be chosen. Finding it might take a while. | _False_ |

Example >>> `python3.11.exe .\SFG.py -p .\data\example.xes -o -se 100`

Example >>> `python3.11.exe .\SFG.py -p .\data\example.xes -se 79 -k 3 -i 100`


### Recency Parameters

The sum of the values provided through the flags below determines the recency window, for which the speed calculations will only be conducted. Transitions that did not occur within the defined time frame appear grayed out with a length according to their all-time average duration.
Throughput—and the existence of edges in general—is displayed unchanged for all traces of the event log.

| Argument | Flag | Description | Default |
|-------|--------|---------|---------|
| `--years` | `-a` | The number of years to be considered.  | `0` |
| `--months` | `-m` | The number of months to be considered. | `0` |
| `--days` | `-d` | The number of days to be considered.  | `0` |
| `--hours` | `-hr` | The number of hours to be considered. | `0` |
| `--minutes` | `-min` | The number of minutes to be considered.  | `0` |
| `--seconds` | `-s` | The number of seconds to be considered. | `0` |

Example >>> `python3.11.exe .\SFG.py -p .\data\example.xes -min 1 -s 57`
