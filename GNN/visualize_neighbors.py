import pandas as pd
from pyvis.network import Network


def visualize_neighbors(
    center_node_id: int,
    neighbors_csv_path: str,
    result_path: str = "GNN/checkpoints/graph_node_neighbors.html",
    height: str = "900px",
    width: str = "100%",
    bgcolor: str = "#FFFFFF",
    font_color: str = "#222222",
):
    """
    Build an interactive star graph centered at `center_node_id` using a neighbors CSV
    with columns: neighbor_node_id, importance, store_nbr, family.

    - Edge thickness encodes `importance`.
    - Node hover shows `store_nbr` and `family`.
    - Nodes are grouped by `family` for color differentiation.
    """
    df = pd.read_csv(neighbors_csv_path)
    # Normalize column names and ensure expected schema
    df.columns = df.columns.str.strip()

    net = Network(
        notebook=True,
        cdn_resources="remote",
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color
        )

    center_id_str = str(center_node_id)
    center_node_store = 1
    center_node_prod = "EGGS"
    # Add center node with a distinct style
    net.add_node(
        center_id_str,
        label=f"Node-{center_id_str} ({center_node_prod}, {center_node_store})",
        title=f"Center Node {center_id_str}",
        color="#E5F5E5",
        size=12,
    )

    # Precompute scaling for sizes/widths and consistent colors
    imp = df["importance"].astype(float)
    imp_min, imp_max = float(imp.min()), float(imp.max())
    # Avoid div0
    span = (imp_max - imp_min) if (imp_max - imp_min) > 1e-9 else 1.0

    def scale(v: float, lo: float, hi: float) -> float:
        return lo + (max(min(v, imp_max), imp_min) - imp_min) * (hi - lo) / span

    # Deterministic palette; highlight EGGS distinctly
    base_palette = [
        "#118AB2", "#06D6A0", "#FFD166", "#073B4C", "#8ECAE6",
        "#FFB703", "#023047", "#8338EC", "#FB8500", "#90BE6D"
    ]
    families = [f for f in df["family"].dropna().astype(str).unique()]
    color_map = {}
    idx = 0
    for fam in families:
        if fam.upper() == "EGGS":
            color_map[fam] = "#287E4C"  # emphasized color
        else:
            color_map[fam] = base_palette[idx % len(base_palette)]
            idx += 1

    # Build edges from center to each neighbor
    for _, row in df.iterrows():
        neighbor_id = int(row["neighbor_node_id"])  # robust to CSV types
        importance = float(row["importance"])  # edge weight
        store_nbr = str(row["store_nbr"]) if not pd.isna(row["store_nbr"]) else ""
        family = str(row["family"]) if not pd.isna(row["family"]) else ""

        if neighbor_id == center_node_id:
            # Skip self-loop if present
            continue

        if importance > 0.0:
            neighbor_id_str = str(neighbor_id)

            node_size = scale(importance, 12.0, 36.0)
            edge_width = 1  # scale(importance, 0.0, 1.0)
            edge_length = 240 - scale(importance, 0.0, 160.0)  # higher importance => shorter edge

            color = color_map.get(family, "#95A5A6")

            # Add neighbor node with consistent color and scaled size
            net.add_node(
                neighbor_id_str,
                label=f"{neighbor_id_str} ({family}, {store_nbr})",
                title=f"Node {neighbor_id_str} | Store: {store_nbr} | Family: {family} | Importance: {importance:.3f}",
                color=color,
                value=node_size,  # value ties into scaling settings
            )

            # Add edge: width/length reflect importance; keep label off to reduce clutter
            net.add_edge(
                center_id_str,
                neighbor_id_str,
                # value=edge_width,
                length=edge_length,
                title=f"Importance: {importance:.3f} | Store: {store_nbr} | Family: {family}",
                # label=f"{importance:.3f}",
            )

    # Enrich hover data with neighbor counts
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node_id = node.get("id")
        node["title"] = (node.get("title", "") +
                          f" | Neighbors: {neighbor_map.get(node_id, [])}")
        node["value"] = len(neighbor_map.get(node_id, [])) or node.get("value", 1)

    # Give interactive controls to tweak physics/colors live
    net.show_buttons(filter_=["physics", "nodes", "edges"])
    net.show(result_path)


if __name__ == "__main__":
    # Defaults wired to the provided CSV and center node "10"
    visualize_neighbors(
        center_node_id=10,
        neighbors_csv_path="GNN/checkpoints/xai_custom_top100_neighbors_node10.csv",
        result_path="GNN/checkpoints/graph_node10_neighbors.html",
    )
