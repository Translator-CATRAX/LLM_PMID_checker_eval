import os
import KGX_node_metrics
import json
from tqdm import tqdm
import bmt

def read_KGX(input_KGX_file):
    # need to verify that data['id'] is unique
    kgx_dict = dict()
    headers_of_interest = ['subject','object','predicate','category','publications']
    with open(input_KGX_file, encoding='utf-8') as f:
        for line in f:
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

    # line counter:
    with open(mapping_file, encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
  
    with open(mapping_file, encoding='utf-8') as f:
        category_mapping = dict()
        for line in tqdm(f, total=total_lines, desc="Processing mapping"):
            try:
                data = json.loads(line)
                id = data['id']
                if len(id) != 0:
                    if id not in category_mapping.keys():
                        category_mapping[id] = {}
                        category_mapping[id]['category']=data['category'][0]
                        category_mapping[id]['name']=data['name']
                        # using bmt, get highest parent of the node category that in not NamedThing and add it to category_mapping[id]['biolink_branch']. calculate the depth from the node category to the category_mapping[id]['biolink_branch']

            except (json.JSONDecodeError, KeyError):
                continue
    return category_mapping

def main(kgx_dict,category_mapping_dict):
    # Compute metrics:
    KGX_nodes_metrics = KGX_node_metrics.compute_KGX_node_metrics(kgx_dict,category_mapping_dict)

    print('bob')


if __name__ == "__main__":
    # load data (TO BE UPDATED AFTER AUTOMATION):
    input_KGX_file = './data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/normalized_edges.jsonl'
    biolink_id_to_category_mapping = './data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/merged_nodes.jsonl'
    output_file = './data/KGX_computed_degrees.json'

    print('load KGX:')
    kgx_dict = read_KGX(input_KGX_file)
    print('load node category IDs:')
    category_mapping_dict = read_mapping(biolink_id_to_category_mapping)

    main(kgx_dict,category_mapping_dict)
