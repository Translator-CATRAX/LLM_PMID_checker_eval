import os
import KGX_node_metrics

def main():
    # Compute metrics:
    KGX_nodes_metrics = KGX_node_metrics.nodes_degree(input_KGX_file)
    print('bob')


if __name__ == "__main__":
    # load data (TO BE UPDATED AFTER AUTOMATION):
    input_KGX_file = './data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/normalized_edges.jsonl'
    
    output_file = './data/KGX_computed_degrees.json'
    main()
