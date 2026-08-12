import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter


conf_col = 'reviewer confidence \n(H, M, L)'
relevance_col = 'irrelevant to Translator'
def calculate_complex_relevance(row):
    """
    Implements the 6-rule mapping based on relationship and confidence.
    """
    rel = row[relevance_col]
    # 1. Normalize the Relationship value
    if pd.isna(rel):
        rel = 'nd'
    else:
        rel = str(rel).strip().lower()

    # 2. Handle Confidence specifically
    conf_val = row[conf_col]

    # Check if it is explicitly NaN
    if pd.isna(conf_val):
        conf_str = 'L'
    else:
        conf_str = str(conf_val).upper()

    if 'H' in conf_str:
        conf = 'H'
    elif 'M' in conf_str:
        conf = 'M'
    elif 'L' in conf_str:
        conf = 'L'
    else:
        return None # Return 'L' if confidence level is unrecognizable

    # 3. Implement the mapping logic
    if rel == 'n': #inversed because "irrelevant y/n"
        if conf == 'H': return 0
        if conf == 'M': return 1
        if conf == 'L': return 2
    if rel == 'nd': #in case nd -> potentially relevant but low confidence
        return 2
    elif rel == 'y':
        if conf == 'L': return 3
        if conf == 'M': return 4
        if conf == 'H': return 5
    
    return None # Return None if relationship is not 'y' or 'n'

rel_col = 'relationship extracted from snippet correctly (y/n)'
def calculate_complex_triple(row):
    """
    Implements the 6-rule mapping based on relationship and confidence.
    """
    # 1. Normalize the Relationship value
    rel = str(row[rel_col]).strip().lower()

    # 2. Handle Confidence specifically
    conf_val = row[conf_col]

    # Check if it is explicitly NaN
    if pd.isna(conf_val):
        conf_str = 'L'
    else:
        conf_str = str(conf_val).upper()

    if 'H' in conf_str:
        conf = 'H'
    elif 'M' in conf_str:
        conf = 'M'
    elif 'L' in conf_str:
        conf = 'L'
    else:
        return None # Return 'L' if confidence level is unrecognizable

    # 3. Implement the mapping logic
    if rel == 'y' or rel == 'x':
        if conf == 'H': return 0
        if conf == 'M': return 1
        if conf == 'L': return 2
    elif rel == 'n':
        if conf == 'L': return 3
        if conf == 'M': return 4
        if conf == 'H': return 5
    
    return None # Return None if relationship is not 'y' or 'n'

def plot_rater_pairing_graph(D,figure_extension = ".png"):
    # --- 4. Show rater pairing graph: ---

    # Find items with more than one reviewer
    item_counts = D.groupby('item_id')['rater_id'].nunique()
    multi_review_items = item_counts[item_counts > 1].index

    if multi_review_items.empty:
        print("No overlapping reviews found to plot graph.")
    else:
        # Extract edges from items with multiple reviewers
        edges = []
        for item in multi_review_items:
            # Get all unique raters for this specific item
            raters_in_item = D[D['item_id'] == item]['rater_id'].unique()
            if len(raters_in_item) > 1:
                # Create combinations of pairs (nchoosek equivalent)
                for pair in combinations(sorted(raters_in_item), 2):
                    edges.append(pair)

        if not edges:
            print("No reviewer overlap found.")
        else:
            # Aggregate weights (count how many items each pair shares)
            edge_counts = Counter(edges)
            
            # Create NetworkX Graph
            G = nx.Graph()
            for (u, v), weight in edge_counts.items():
                G.add_edge(u, v, weight=weight)

            # --- 5. Visualization ---
            plt.figure(figsize=(12, 8), facecolor='white')
            pos = nx.spring_layout(G, k=0.5, iterations=50) # 'force' layout equivalent

            # Node properties
            degrees = dict(G.degree())
            node_sizes = [5 + (degrees[n] / max(degrees.values())) * 20 *20 for n in G.nodes()]
            
            # Draw Edges
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(weights) if weights else 1
            edge_widths = [round((w / max_weight) * 20 + 1, 2) for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#999999', alpha=0.7)

            # Draw Nodes
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#3366CC')

            # Custom Node Labels (Rater IDs - BIG and BLUE)
            for node, (x, y) in pos.items():
                plt.text(x, y + 0.01, str(node), fontsize=24, fontweight='bold', 
                        color="#022C57", ha='center', va='bottom')

            # Custom Edge Labels (Weights - MEDIUM size)
            for u, v, d in G.edges(data=True):
                x_mid = (pos[u][0] + pos[v][0]) / 2
                y_mid = (pos[u][1] + pos[v][1]) / 2
                plt.text(x_mid, y_mid, str(d['weight']), fontsize=18, fontweight='bold',
                        color='#333333', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            plt.title("Reviewer Overlap Graph\nBlue: Reviewers | Numbers: Shared Items", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"data/reviewer_graph{figure_extension}")
            plt.close()

def prepare_dataset():
    # --- 1. Load Data ---
    # Replace these paths with your actual local paths
    path_metrics = r'data\KG_metrics_LLM_Checker_results_test_suite_samplesize4_reviewers-assigned.csv'
    path_deval = r'data\SemMedDB_evaluation_sheet - SemMedDB_evaluation_sheet.csv'
    path_rater_mapping = r'data\SemMedDB eval reviewer ID - rater mapping.csv'

    D3 = pd.read_csv(path_metrics)
    # MATLAB 'NumHeaderLines', 1 skips the first line. In pandas, we use skiprows=1.
    Deval = pd.read_csv(path_deval, delimiter=',', skiprows=1)
    rater_mapping = pd.read_csv(path_rater_mapping)

    # --- 2. Data Preparation ---

    # Rename 'x_' to 'index' in Deval if it exists
    if '#' in Deval.columns:
        Deval = Deval.rename(columns={'#': 'index'})

    # Join D3 and Deval on 'index'
    D = pd.merge(D3, Deval, on='index', how='left')

    # Rename reviewer_D3 to reviewer_ids if it exists
    if 'reviewer_x' in D.columns:
        D = D.rename(columns={'reviewer_x': 'reviewer_ids'})

    # Apply rater mapping logic
    for _, row in rater_mapping.iterrows(): # ISSUE WITH THE RATER MAPPING
        current_reviewer = row['reviewer_ids']
        current_start = row['rows start']
        current_end = row['rows end']
        current_rater = row['Rater ID']

        # Find rows matching reviewer ID
        mask = (D['reviewer_ids'] == current_reviewer.strip())

        # Refine mask if range is provided (not NaN)
        if pd.notna(current_start) and pd.notna(current_end):
            range_mask = (D['index'] >= current_start) & (D['index'] <= current_end)
            mask = mask & range_mask

        # Update metadata
        D.loc[mask, 'rater_id'] = current_rater
        D.loc[mask, 'rowsStart_mapped'] = current_start
        D.loc[mask, 'rowsEnd_mapped'] = current_end
    D['rater_id'] = D['rater_id'].astype(float).astype(int)


    # Check if review is complete (all three columns must be non-null/not NaN)
    cols_to_check = [
        'subject identified correctly in snippet (y/n)',
        'object identified correctly in snippet (y/n)',
        'relationship extracted from snippet correctly (y/n)'
    ]

    # Check if all columns in the list are not null for each row
    D['is_reviewed_complete'] = D[cols_to_check].notna().all(axis=1)
    D = D[D['is_reviewed_complete']] # Keep only rows where is_reviewed_complete is True


    # Convert rating to numeric: 1 if 'y', otherwise 0 (simulating MATLAB ismember logic)
    D['TripleRating'] = (D['relationship extracted from snippet correctly (y/n)'] == 'y').astype(int)

    # Create unique key for items
    D['item_id_str'] = D['PMID'].astype(str) + '_' + D['id'].astype(str)
    D['item_id'], uniques = pd.factorize(D['item_id_str'])

    # Create strata id:
    D['strata_str'] = D['semantic_complexity_classes'].astype(str) + '_' + D['degree_assymetry_classes'].astype(str) + '_' + D['biomedical_area_pair'].astype(str)
    D['strata'], uniques = pd.factorize(D['strata_str'])

    # Calculate percentage overlap (items with > 1 reviewer)
    overlap_count = D.groupby('item_id').size().value_counts()
    total_rows = len(D)
    percentage_overlap = (overlap_count.index > 1).sum() / total_rows # This is a simplified logic check
    # More accurately to your MATLAB:
    group_counts = D.groupby(['PMID', 'id']).size()
    overlap_val = (group_counts > 1).sum() / len(D)
    print(f"percentage overlap: {round(overlap_val, 2)}")

    # --- 3. Prepare rating ---
    # Define the exact column names from your dataset
    # conf_col = 'reviewer confidence \n(H, M, L)'
    # rel_col = 'relationship extracted from snippet correctly (y/n)'

    # --- 4. Implement complex rating from rating rules ---
    # TRIPLES (relationship)
    if rel_col in D.columns and conf_col in D.columns:
        D['triple rating'] = D.apply(calculate_complex_triple, axis=1)
    else:
        # Fallback if columns are missing (as per your previous structure)
        print("Warning: Required columns for triple rating calculation not found.")
        D['triple rating'] = np.nan 

    # RELEVANCE
    if relevance_col in D.columns and conf_col in D.columns:
        D['relevance rating'] = D.apply(calculate_complex_relevance, axis=1)
    else:
        # Fallback if columns are missing (as per your previous structure)
        print("Warning: Required columns for relevance rating calculation not found.")
        D['relevance rating'] = np.nan 

        


    D.to_csv('data/data_merged.csv', index=False)

    return D


