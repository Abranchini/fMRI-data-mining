# This file contains the functions for data visualitzation.
# The ML part of the code has been done in Python  
# But note that most of the visualization has been done in Matlab
# due to the fact that the visualization tools provided by the lab
# were implemented in Matlab.


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4



def plotGraph(A): # Plots classical graph from the adjacency matrix
    G = nx.Graph(A)
    pos = nx.layout.spring_layout(G)
    node_sizes = 10 
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                arrowsize=10, edge_color=edge_colors, width=2)
    val=nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    # show graph
    plt.show()

def plotGraphCircle(A): # Plots circular graph from the adjacency matrix
    G = nx.Graph(A)
    plot = Plot(plot_width=400, plot_height=400,
            x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
    plot.title.text = "Graph Interaction Demonstration"

    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

    graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0,0))

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)

    output_file("circularGraph.html")
    show(plot)

def plotCoefficients(coef,N):
    #fig, ax = plt.subplots()
    plt.imshow(coef, extent=[0, N-1, 0, N-1],cmap="jet")
    plt.colorbar()
    plt.show()
    # plt.savefig("Coefs" + '.png', bbox_inches='tight')

    