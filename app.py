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

# --- Streamlit Structural Caching Bypass ---
@st.cache_resource(hash_funcs={
    nx.Graph: lambda _: None,
    ig.Graph: lambda _: None,
    "Bio.PDB.Structure.Structure": lambda _: None
})
def parse_and_extract_structure(pdb_content):
    """Parses PDB content safely inside a cached boundary."""
    parser = PDBParser()
    pdb_file_like = StringIO(pdb_content)
    structure = parser.get_structure('protein', pdb_file_like)
    return structure

def get_alpha_carbon_coords(structure, target_chains=None):
    """
    Extracts the alpha-carbon coordinates for each residue from a pre-parsed structure.
    Cleaned to fetch only the integer number from the Biopython sequence tuple.
    """
    coords = []
    for model in structure:
        all_chains_in_pdb = sorted(list(set(chain.id for chain in model)))
        chains_to_process = target_chains if target_chains is not None else all_chains_in_pdb

        for chain in model:
            if chain.id in chains_to_process:
                for residue in chain:
                    if residue.has_id('CA') and residue.get_resname() not in ['HOH', 'WAT', 'UNL']:
                        ca_atom = residue['CA']
                        
                        # --- FIX: Extract only the integer ID from the Biopython residue tuple ---
                        res_num = residue.id[1] 
                        
                        coords.append({
                            'residue_id': res_num, # Clean integer number
                            'residue_name': residue.get_resname(),
                            'chain_id': chain.id,
                            'coordinates': ca_atom.get_coord()
                        })
    return coords

def build_contact_network(ca_coords, cutoff=7.5):
    """Builds a protein contact network with clean, readable string names for Node_ID."""
    G = nx.Graph()

    for i, res_data in enumerate(ca_coords):
        # --- FIX: Node ID will now look clean (e.g., "A_MET_14") ---
        node_id = f"{res_data['chain_id']}_{res_data['residue_name']}_{res_data['residue_id']}"
        G.add_node(node_id, **res_data)

    for i in range(len(ca_coords)):
        node_i_data = ca_coords[i]
        node_i_id = f"{node_i_data['chain_id']}_{node_i_data['residue_name']}_{node_i_data['residue_id']}"
        coord_i = node_i_data['coordinates']

        for j in range(i + 1, len(ca_coords)):
            node_j_data = ca_coords[j]
            node_j_id = f"{node_j_data['chain_id']}_{node_j_data['residue_name']}_{node_j_data['residue_id']}"
            coord_j = node_j_data['coordinates']

            distance = np.linalg.norm(coord_i - coord_j)
            if distance <= cutoff:
                G.add_edge(node_i_id, node_j_id, distance=distance)
    return G

# --- Excel Exporter Utility Function ---
def to_excel(df):
    """Converts a DataFrame to an Excel file binary format."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Network Analysis')
    return output.getvalue()

# --- Streamlit Application Layout ---
st.set_page_config(layout="wide")
st.title("Protein Contact Network Analysis with Leiden Communities")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a PDB file", type=["pdb"])
    cutoff = st.slider("Distance Cutoff (Angstroms)", min_value=1.0, max_value=15.0, value=7.5, step=0.5)

if uploaded_file is not None:
    pdb_content = uploaded_file.read().decode("utf-8")
    
    structure = parse_and_extract_structure(pdb_content)
    all_chains = sorted(list(set(chain.id for model in structure for chain in model)))

    selected_chains = st.sidebar.multiselect(
        "Select Chain(s) to Analyze (leave empty for all)",
        options=all_chains,
        default=all_chains
    )

    if not selected_chains:
        st.warning("Please select at least one chain to proceed with the analysis, or leave empty to analyze all chains.")
        ca_coordinates = get_alpha_carbon_coords(structure, target_chains=None)
    else:
        ca_coordinates = get_alpha_carbon_coords(structure, target_chains=selected_chains)

    if not ca_coordinates:
        st.warning("No alpha carbon coordinates found for the selected chain(s) or PDB file.")
    else:
        st.subheader("Protein Contact Network Analysis")
        contact_network = build_contact_network(ca_coordinates, cutoff=cutoff)

        st.write(f"Number of nodes in the network: {contact_network.number_of_nodes()}")
        st.write(f"Number of edges in the network: {contact_network.number_of_edges()}")

        if contact_network.number_of_nodes() > 1 and contact_network.number_of_edges() > 0:
            ig_graph = ig.Graph.from_networkx(contact_network)
            partition = la.find_partition(ig_graph, la.ModularityVertexPartition)

            modularity_score = partition.modularity
            st.write(f"Modularity score: {modularity_score:.4f}")

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

                P_i = 1 - sum_ratios_squared
                participation_coefficients[node_id] = P_i

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
            
            # --- Added Download Button and Table Display ---
            col1, col2 = st.columns([8, 2])
            with col1:
                st.markdown("### Processed Results")
            with col2:
                # Convert the dataframe into an Excel byte-stream
                excel_data = to_excel(df_results)
                st.download_button(
                    label="📥 Download Excel File",
                    data=excel_data,
                    file_name="protein_network_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Display interactive dataframe below the header components
            st.dataframe(df_results, use_container_width=True)

            # --- Network Visualization Section ---
            st.markdown("---")
            st.markdown("### Protein Network Community Visualization")
            
            show_plot = st.checkbox("Render 2D Network Graph", value=True)
            
            if show_plot:
                with st.spinner("Generating network plot..."):
                    # Create the layout configuration figure container explicitly
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Convert graph elements into simple list maps to completely bypass deep recursion traps
                    node_list = list(contact_network.nodes())
                    edge_list = list(contact_network.edges())
                    
                    # Compute iterative visual coordinates with capped loops to optimize rendering speed
                    pos = nx.spring_layout(
                        contact_network, 
                        iterations=50,   
                        seed=42          
                    )
                    
                    # Match community index metrics with flat node lists
                    node_communities = [contact_network.nodes[node]['community'] for node in node_list]
                    
                    # Initialize stable colormap registration using modern Matplotlib API
                    cmap = plt.colormaps['tab20']
                    
                    # Draw nodes safely using sequence constraints
                    nx.draw_networkx_nodes(
                        contact_network, 
                        pos, 
                        nodelist=node_list,
                        node_size=60, 
                        node_color=node_communities, 
                        cmap=cmap, 
                        ax=ax
                    )
