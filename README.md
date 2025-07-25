# Speedily Follows Graphs

_Speedily Follows Graphs (SFGs)_ are a novel variant of _Directly Follows Graphs (DFGs)_ for displaying event logs with the additional notion of **speed** alongside **capacity** as in transition frequency. Additionally, potential **bottlenecks** can be detected by combining these two notions.

In practice, this implementation directly maps the temporal distances between events as the edge lengths between their respective nodes. To do this, it folds the edges periodically in the form of sine waves, if required by the node layout.
This implementation further uses edge color to highlight potential bottlenecks, i.e. slow transitions of high frequencies, where a lot of traces lose time.
