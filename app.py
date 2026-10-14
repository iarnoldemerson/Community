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

sys.setrecursionlimit(10000)

# --- Streamlit Caching Redesign ---
# By doing the parsing AND extraction entirely inside the cached function, 
# the deep molecular 'Structure' objects never enter Streamlit's runtime memory loops.
@st.cache_data
def process_pdb_to_coordinates(pdb_content):
    """
    Parses PDB content and immediately extracts coordinates into a raw list of 
    dictionaries. Returns standard Python primitives to fully bypass RecursionErrors.
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
                    # Extract raw primitives (strings, integers, list of floats)
                    extracted_coords.append({
                        'residue_id': int(residue.id[1]), 
                        'residue_name': str(residue.get_resname()),
                        'chain_id': str(chain.id),
                        'coordinates': ca_atom.get_coord().tolist() # Convert numpy array to flat list
                    })
    return extracted_coords

def build_contact_network(ca_coords, target_chains, cutoff=7.5):
    """Builds a protein contact network filtering by chains using flat coordinate data."""
    G = nx.Graph()

    # Filter coordinates based on user sidebar selection
    filtered_coords = [
        res for res in ca_coords 
        if target_chains is None or res['chain_id'] in target_chains
    ]

    for res_data in filtered_coords:
        node_id = f"{res_data['chain_id']}_{res_data['residue_name']}_{res_data['residue_id']}"
        # Convert coordinates back to numpy array for vector math
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
    
    # Safely get raw flat coordinate payload 
    all_coordinates = process_pdb_to_coordinates(pdb_content)
    
    # Get available chains from our flat list
    all_chains = sorted(list(set(res['chain_id'] for res in all_coordinates)))

    selected_chains = st.sidebar.multiselect(
        "Select Chain(s) to Analyze (leave empty for all)",
        options=all_chains,
        default=all_chains
    )

    # Process based on selection map
    chains_filter = None if not selected_chains else selected_chains
    
    if not selected_chains:
        st.warning("Please select at least one chain to proceed with the analysis, or leave empty to analyze all chains.")

    contact_network = build_contact_network(all_coordinates, target_chains=chains_filter, cutoff=cutoff)

    st.subheader("Protein Contact Network Analysis")
    st.write(f"Number of nodes in the network: {contact_network.number_of_nodes()}")
    st.write(f"Number of edges in the network: {contact_network.number_of_edges()}")

    if contact_network.number_of_nodes() > 1 and contact_network.number_of_edges() > 0:
        ig_graph = ig.Graph.from_networkx(contact_network)
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)

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

        nx.set_node_attributes(contact_network, participation_coefficients, 'participation_coefficient')

        output_data = []
        for node_id, attributes in contact_network.nodes(data=True):
            output_data.append({
                'Node_ID': node_id,
                'Chain_ID': attributes['chain_id'],
                'Residue_Name': attributes['residue_name'],
                'Residue_Number': attributes['residue_id'],
                'Community': attributes['community'],
                'Participation_Coefficient': attributes['participation_coefficient']
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
        
        # FIX: Replaced deprecated use_container_width=True with width='stretch'
        st.dataframe(df_results, width='stretch')

        # --- Network Visualization Section ---
        st.markdown("---")
        st.markdown("### Protein Network Community Visualization")
        
        show_plot = st.checkbox("Render 2D Network Graph", value=True)
        
        if show_plot:
            with st.spinner("Generating network plot..."):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                node_list = list(contact_network.nodes())
                edge_list = list(contact_network.edges())
                
                pos = nx.spring_layout(contact_network, iterations=50, seed=42)
                node_communities = [contact_network.nodes[node]['community'] for node in node_list]
                cmap = plt.colormaps['tab20']
                
                nx.draw_networkx_nodes(
                    contact_network, pos, nodelist=node_list,
                    node_size=60, node_color=node_communities, cmap=cmap, ax=ax
                )
                nx.draw_networkx_edges(
                    contact_network, pos, edgelist=edge_list,
                    alpha=0.2, edge_color="gray", ax=ax
                )
                
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.success("Analysis complete! Data processed successfully.")
