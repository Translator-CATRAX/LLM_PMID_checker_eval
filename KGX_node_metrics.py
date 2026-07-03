import json
from collections import Counter




## Compute metrics:


def nodes_degree(input_KGX_file):
    degree_counter = Counter()
    with open(input_KGX_file, encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Update count for both nodes in the edge
                degree_counter[data['subject']] += 1
                degree_counter[data['object']] += 1
            except (json.JSONDecodeError, KeyError):
                continue

    # Save all nodes and their degrees to a JSON file
    all_degrees = dict(degree_counter)
    return all_degrees

def get_biolink_category_mapping(id_list,mapping_file):
    return []

def compute_KGX_node_metrics(input_KGX_file):
    node_metrics = dict()
    node_metrics["degree"] = nodes_degree(input_KGX_file)
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(all_degrees, f, indent=4)
