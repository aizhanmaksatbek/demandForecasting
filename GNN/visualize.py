import pandas as pd
from pyvis.network import Network


def vizualize_graph(file_path: str, result_path: str = "graph.html"):
    got_net = Network(
        notebook=True,
        cdn_resources="remote",
        height="950px",
        width="100%",
        bgcolor="#222222",
        font_color="white"
        )

    # set the physics layout of the network
    got_net.barnes_hut()
    got_data = pd.read_csv(file_path)
    columns = got_data.columns.str.strip()

    sources = got_data[columns[0]]
    targets = got_data[columns[1]]
    weights = got_data[columns[2]]

    edge_data = zip(sources, targets, weights)

    for e in edge_data:
        src = str(e[0])
        dst = str(e[1])
        w = e[2]

        got_net.add_node(src, src, title=src)
        got_net.add_node(dst, dst, title=dst)
        got_net.add_edge(src, dst, value=w, title=f"Weight: {w}")

    neighbor_map = got_net.get_adj_list()

    # add neighbor data to node hover data
    for node in got_net.nodes:
        node["title"] += "-node | Neighbors:" + str(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])

    got_net.show(result_path)


vizualize_graph(
    "GNN/data/family_graph_edges.csv",
    "GNN/checkpoints/graph_viz_prod_family.html"
    )

vizualize_graph(
    "GNN/data/store_graph_edges.csv",
    "GNN/checkpoints/graph_viz_store.html"
    )
