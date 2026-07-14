import json
from collections import Counter
from tqdm import tqdm




## Compute metrics:


def nodes_degree(kgx_dict):
    degree_counter = Counter()
    for edge in kgx_dict:
        try:
            degree_counter[kgx_dict[edge]['subject']] += 1
            degree_counter[kgx_dict[edge]['object']] += 1
        except (json.JSONDecodeError, KeyError):
            continue

    # Save all nodes and their degrees to a JSON file
    all_degrees = dict(degree_counter)
    return all_degrees

def get_biolink_category_mapping(id_list,category_mapping):

    node_category = dict()
            
    for id in id_list:
        if id in category_mapping.keys():
            node_category[id] = category_mapping[id]
        else:
            print(f'could not find id:{id} in mapping file')

    return node_category

def compute_KGX_node_metrics(kgx_dict,category_mapping_dict):
    node_degree = nodes_degree(kgx_dict)
    node_category = get_biolink_category_mapping(list(node_degree.keys()),category_mapping_dict)

    print('bob')
    return node_degree, node_category
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(all_degrees, f, indent=4)
