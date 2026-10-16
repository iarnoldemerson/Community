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
                    
                    # Target index 1 from the Biopython ID tuple to get the true integer number
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

    selected_chains = st.sidebar.multiselect(
        "Select Chain(s) to Analyze (leave empty for all)",
        options=all_chains,
        default=all_chains
    )

    chains_filter = None if not selected_chains else selected_chains
    
    if not selected_chains:
        st.warning("Please select at least one chain to proceed with the analysis, or leave empty to analyze all chains.")

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
        
        # Color unpacking configuration block fixes the float conversion TypeError
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
            st.caption("Amino acid residues are individual spheres colored by community mapping.")
            
            viewer = py3Dmol.view(width=450, height=450)
            viewer.addModel(pdb_content, 'pdb')
            viewer.setBackgroundColor('#111111')
            viewer.setStyle({}, {})  # Clears default parent structures
            
            for _, row in df_results.iterrows():
                viewer.setStyle(
                    {'chain': row['Chain_ID'], 'resi': int(row['Residue_Number'])},
                    {'sphere': {'color': community_colors[row['Community']], 'radius': 1.8}}
                )
            viewer.zoomTo()
            showmol(viewer, height=450, width=450)

        with vis_col2:
            st.markdown("### 2D Clustered Community Network Layout")
            st.caption("Residue interaction topology grouped explicitly into distinct spatial cluster pools.")
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            node_list = list(contact_network.nodes())
            edge_list = list(contact_network.edges())
            
            # --- Inject community attraction weights directly to force structural pooling ---
            for u, v in contact_network.edges():
                if contact_network.nodes[u]['community'] == contact_network.nodes[v]['community']:
                    contact_network[u][v]['weight'] = 8.0  # Tight attraction inside same community
                else:
                    contact_network[u][v]['weight'] = 0.2  # Soft repulsion between different communities

            # Run global spring solver pulling tightly weighted node clusters together
            pos = nx.spring_layout(contact_network, weight='weight', k=0.4, seed=42, iterations=50)
            
            node_colors_list = [community_colors[contact_network.nodes[node]['community']] for node in node_list]
            
            # Draw network nodes safely across the clustered coordinate map
            nx.draw_networkx_nodes(
                contact_network, pos, nodelist=node_list,
                node_size=50, node_color=node_colors_list, ax=ax
            )
            
            # Draw connection trails spanning between the local clusters
            nx.draw_networkx_edges(
                contact_network, pos, edgelist=edge_list,
                alpha=0.2, edge_color="#333333", width=0.5, ax=ax
            )
            
            ax.axis('off')
            plt.tight_layout()
            
            # Pass figure explicitly and override transparent backgrounds to prevent rendering invisibility
            st.pyplot(fig, clear_figure=True, facecolor='white')
            plt.close(fig)

        st.markdown("---")
        st.success("Analysis complete! Clustered topological data structures populated successfully.")
