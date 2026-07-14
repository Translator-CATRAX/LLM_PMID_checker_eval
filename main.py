import os
import KGX_node_metrics
import json
from tqdm import tqdm
import bmt

def get_highest_parent_below_namedthing(toolkit, cat_name):
    """
    Calcule le parent le plus élevé qui n'est pas NamedThing et la profondeur.
    Utilise toolkit.get_parent de manière itérative.
    """
    current_node = cat_name
    depth = 0
    while True:
        parent = toolkit.get_parent(current_node, False)
        # Si pas de parent trouvé ou si le parent est NamedThing, on s'arrête
        if not parent or "NamedThing" in parent:
            return current_node, depth
        else:
            current_node = parent
            depth += 1

def read_KGX(input_KGX_file,headers_of_interest = ['subject','object','predicate','category','publications']):
    # need to verify that data['id'] is unique
    kgx_dict = dict()
     
    
    with open(input_KGX_file, encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(input_KGX_file, encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing mapping"):
            try:
                data = json.loads(line)
                if 'id' in data.keys():
                    additional_data = dict()
                    kgx_dict[data['id']] = dict()
                    for h in headers_of_interest: # init
                        kgx_dict[data['id']][h] = []
                    for k in data.keys():
                        if k in headers_of_interest and k != 'id':
                            kgx_dict[data['id']][k] = data[k]
                        elif k != 'id':
                            additional_data[k] = data[k]
                    kgx_dict[data['id']]['additional_data'] = additional_data
            except (json.JSONDecodeError, KeyError):
                continue

    return kgx_dict

def read_mapping(mapping_file):
    category_mapping = dict()
    toolkit = bmt.toolkit.Toolkit()
    # Cache pour éviter de recalculer la hiérarchie pour chaque ligne
    # Structure: { category_name: (biolink_branch, depth) }
    cache_branch = {}

    # line counter:
    with open(mapping_file, encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
  
    with open(mapping_file, encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing mapping"):
            try:
                data = json.loads(line)
                id = data['id']
                if len(id) != 0:
                    if id not in category_mapping.keys():
                        category_mapping[id] = {}
                        cat_name = data['category'][0]
                        category_mapping[id]['category'] = cat_name
                        category_mapping[id]['name'] = data['name']
                        
                        
                        branch, depth = get_highest_parent_below_namedthing(toolkit, cat_name)
                        category_mapping[id]['biolink_branch'] = branch
                        category_mapping[id]['depth'] = depth

            except (json.JSONDecodeError, KeyError):
                continue
    return category_mapping

def main(kgx_dict,category_mapping_dict):
    # Compute metrics:
    KGX_nodes_metrics = KGX_node_metrics.compute_KGX_node_metrics(kgx_dict,category_mapping_dict)

    print('bob')


if __name__ == "__main__":
    # load data (TO BE UPDATED AFTER AUTOMATION):
    input_KGX_file = './data/kg2.10.3_semmeddb_for_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/normalized_edges.jsonl'
    biolink_id_to_category_mapping = './data/kg2.10.3_semmeddb_for_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/merged_nodes.jsonl'
    output_file = './data/KGX_computed_degrees.json'

    print('load KGX:')
    kgx_dict = read_KGX(input_KGX_file)
    print('load node category IDs:')
    category_mapping_dict = read_mapping(biolink_id_to_category_mapping)

    main(kgx_dict,category_mapping_dict)
