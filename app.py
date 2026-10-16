
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import networkx as nx
import matplotlib.pyplot as plt
import leidenalg as la
import igraph as ig
import streamlit as st
from io import StringIO, BytesIO
import sys

# --- 3D Molecular View Component Additions ---
import py3Dmol
from stmol import showmol

sys.setrecursionlimit(10000)

# --- Streamlit Caching ---
@st.cache_data
def process_pdb_to_coordinates(pdb_content):
    """
    Parses PDB content and extracts coordinates into a raw list of dictionaries.
    Returns standard Python primitives to fully bypass RecursionErrors.
    """
    parser = PDBParser()
    pdb_file_like = StringIO(pdb_content)
    structure = parser.get_structure('protein', pdb_file_like)

    extracted_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA') and residue.get_resname() not in ['HOH', 'WAT', 'UNL']:
                    ca_atom = residue['CA']
                    clean_res_num = int(residue.id[1])
                    extracted_coords.append({
                        'residue_id': clean_res_num,
                        'residue_name': str(residue.get_resname()),
                        'chain_id': str(chain.id),
                        'coordinates': ca_atom.get_coord().tolist()
                    })
    return extracted_coords

def build_contact_network(ca_coords, target_chains, cutoff=7.5):
    """Builds a protein contact network filtering by chains using flat data arrays."""
    G = nx.Graph()

    filtered_coords = [ 
        res for res in ca_coords 
        if target_chains is None or res['chain_id'] in target_chains 
    ]

    for res_data in filtered_coords:
        node_id = f"{res_data['chain_id']}_{res_data['residue_name']}_{res_data['residue_id']}"
        G.add_node(node_id,
                   residue_id=res_data['residue_id'],
                   residue_name=res_data['residue_name'],
                   chain_id=res_data['chain_id'],
                   coordinates=np.array(res_data['coordinates']))

    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        node_i_id, data_i = nodes[i]
        coord_i = data_i['coordinates']

        for j in range(i + 1, len(nodes)):
            node_j_id, data_j = nodes[j]
            coord_j = data_j['coordinates']

            distance = np.linalg.norm(coord_i - coord_j)
            if distance <= cutoff:
                G.add_edge(node_i_id, node_j_id, distance=distance)
    return G

# --- Excel Exporter Utility ---
def to_excel(df):
    """Converts a DataFrame to an Excel file binary format."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Network Analysis')
    return output.getvalue()

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
st.title("Protein Contact Network Analysis with Leiden Communities")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a PDB file", type=["pdb"])
    cutoff = st.slider("Distance Cutoff (Angstroms)", min_value=1.0, max_value=15.0, value=7.5, step=0.5)

if uploaded_file is not None:
    pdb_content = uploaded_file.read().decode("utf-8")

    # Process structural assets down to standard primitive blocks
    all_coordinates = process_pdb_to_coordinates(pdb_content)
    all_chains = sorted(list(set(res['chain_id'] for res in all_coordinates)))

    with st.sidebar:
        selected_chains = st.multiselect(
            "Select Chain(s) to Analyze (leave empty for all)",
            options=all_chains,
            default=all_chains
        )

    chains_filter = None if not selected_chains else selected_chains

    if not selected_chains:
        st.sidebar.warning("Please select at least one chain to proceed.")

    contact_network = build_contact_network(all_coordinates, target_chains=chains_filter, cutoff=cutoff)

    st.subheader("Protein Contact Network Analysis")
    st.write(f"Number of nodes in the network: {contact_network.number_of_nodes()}")
    st.write(f"Number of edges in the network: {contact_network.number_of_edges()}")

    if contact_network.number_of_nodes() > 1 and contact_network.number_of_edges() > 0:
        ig_graph = ig.Graph.from_networkx(contact_network)

        # Leiden Seed is fixed to 42 for calculation consistency across manual updates
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition, seed=42)

        st.write(f"Modularity score: {partition.modularity:.4f}")

        node_to_community = {ig_graph.vs[i]['_nx_name']: partition.membership[i] + 1 for i in range(ig_graph.vcount())}
        nx.set_node_attributes(contact_network, node_to_community, 'community')

        # Calculate Participation Coefficient (P_i)
        participation_coefficients = {}
        all_communities = set(nx.get_node_attributes(contact_network, 'community').values())

        for node_id in contact_network.nodes():
            node_degree = contact_network.degree(node_id)
            if node_degree == 0:
                participation_coefficients[node_id] = 0.0
                continue

            community_links = {comm: 0 for comm in all_communities}
            for neighbor in contact_network.neighbors(node_id):
                neighbor_community = contact_network.nodes[neighbor]['community']
                if neighbor_community in community_links:
                    community_links[neighbor_community] += 1

            sum_ratios_squared = 0
            for comm in all_communities:
                k_is = community_links[comm]
                if k_is > 0:
                    sum_ratios_squared += (k_is / node_degree)**2

            participation_coefficients[node_id] = 1 - sum_ratios_squared

        output_data = []
        for node_id, attributes in contact_network.nodes(data=True):
            output_data.append({
                'Node_ID': node_id,
                'Chain_ID': attributes['chain_id'],
                'Residue_Name': attributes['residue_name'],
                'Residue_Number': attributes['residue_id'],
                'Community': attributes['community'],
                'Participation_Coefficient': participation_coefficients[node_id]
            })

        df_results = pd.DataFrame(output_data)

        # --- Highlight Residues Selector ---
        all_residues_list = sorted(df_results['Node_ID'].unique())
        with st.sidebar:
            st.markdown("---")
            st.header("💡 Highlight Amino Acids")
            highlighted_nodes = st.multiselect(
                "Select Residues to Highlight in Visualizations",
                options=all_residues_list,
                default=[]
            )

        # --- SIDEBAR Hubs Ranking Box ---
        df_sorted_hubs = df_results.sort_values(by='Participation_Coefficient', ascending=False).head(5)
        with st.sidebar:
            st.markdown("---")
            st.header("🔑 Critical Cluster Hubs")
            st.caption("Top 5 residues spanning maximum interface communities (Highest Participation Coefficients).")
            for _, hub in df_sorted_hubs.iterrows():
                st.info(f"**{hub['Node_ID']}**\n\n*   Community: {hub['Community']}\n*   $P_i$: {hub['Participation_Coefficient']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Processed Results")
        with col2:
            excel_data = to_excel(df_results)
            st.download_button(
                label="📥 Download Excel File",
                data=excel_data,
                file_name="protein_network_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.dataframe(df_results, width='stretch')

        # --- Dual Visualization Window ---
        st.markdown("---")

        cmap = plt.colormaps['tab20']
        unique_communities = sorted(list(df_results['Community'].unique()))
        community_colors = {}
        for idx, comm in enumerate(unique_communities):
            r, g, b = cmap(idx % 20)[:3]
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            community_colors[comm] = hex_color

        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            st.markdown("### 3D Atom Sphere (Ball) Viewer")
            st.caption("Physical 3D structure showing amino acid residue balls colored by community. Highlighted residues have secondary yellow halos.")

            viewer = py3Dmol.view(width=450, height=450)
            viewer.addModel(pdb_content, 'pdb')
            viewer.setBackgroundColor('#111111')
            viewer.setStyle({}, {})

            for _, row in df_results.iterrows():
                node_id = row['Node_ID']
                # Highlight selection visual styling override
                if node_id in highlighted_nodes:
                    viewer.setStyle(
                        {'chain': row['Chain_ID'], 'resi': int(row['Residue_Number'])},
                        {'sphere': {'color': '#FFFF00', 'radius': 2.8}}
                    )
                else:
                    viewer.setStyle(
                        {'chain': row['Chain_ID'], 'resi': int(row['Residue_Number'])},
                        {'sphere': {'color': community_colors[row['Community']], 'radius': 1.8}}
                    )
            viewer.zoomTo()
            showmol(viewer, height=450, width=450)

        with vis_col2:
            st.markdown("### 3D Clustered Topological Network Layout")
            st.caption("Residue interaction topological graph. Highlighted residues are displayed as large glowing stars.")

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')

            node_list = list(contact_network.nodes())
            edge_list = list(contact_network.edges())

            for u, v in contact_network.edges():
                if contact_network.nodes[u]['community'] == contact_network.nodes[v]['community']:
                    contact_network[u][v]['weight'] = 8.0
                else:
                    contact_network[u][v]['weight'] = 0.2

            pos_3d = nx.spring_layout(contact_network, dim=3, weight='weight', k=0.5, seed=42, iterations=60)

            x_nodes = [pos_3d[node][0] for node in node_list]
            y_nodes = [pos_3d[node][1] for node in node_list]
            z_nodes = [pos_3d[node][2] for node in node_list]
            
            # Base node colors and sizes
            node_colors_list = []
            node_sizes_list = []
            for node in node_list:
                if node in highlighted_nodes:
                    node_colors_list.append('#FFFF00')
                    node_sizes_list.append(180) # Larger size for highlighting
                else:
                    node_colors_list.append(community_colors[contact_network.nodes[node]['community']])
                    node_sizes_list.append(40)

            # Draw normal nodes
            ax.scatter(x_nodes, y_nodes, z_nodes, c=node_colors_list, s=node_sizes_list, depthshade=True, zorder=3)

            # Highlight border for chosen nodes
            highlighted_indices = [node_list.index(n) for n in highlighted_nodes if n in node_list]
            if highlighted_indices:
                hx = [x_nodes[i] for i in highlighted_indices]
                hy = [y_nodes[i] for i in highlighted_indices]
                hz = [z_nodes[i] for i in highlighted_indices]
                ax.scatter(hx, hy, hz, facecolors='none', edgecolors='red', s=300, marker='o', linewidths=2.0, zorder=4)

            for edge in edge_list:
                u, v = edge
                x_line = [pos_3d[u][0], pos_3d[v][0]]
                y_line = [pos_3d[u][1], pos_3d[v][1]]
                z_line = [pos_3d[u][2], pos_3d[v][2]]
                ax.plot(x_line, y_line, z_line, color='#444444', alpha=0.15, linewidth=0.6, zorder=1)

            ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True, facecolor='white')
            plt.close(fig)

        st.markdown("---")
        st.success("Analysis complete! Clustered topological data structures populated successfully.")
