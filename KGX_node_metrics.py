import json
from collections import Counter
import bmt

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

def map_id_to_biolink_category(id_list,category_mapping):

    node_category = dict()
            
    for id in id_list:
        if id in category_mapping.keys():
            node_category[id] = category_mapping[id]
        else:
            print(f'could not find id:{id} in mapping file')

    return node_category

def get_highest_parent_below_root(toolkit, cat_name):
    """
    Calcule le parent le plus élevé qui n'est pas NamedThing et la profondeur.
    Utilise toolkit.get_parent de manière itérative.
    """
    current_node = cat_name
    depth = 1
    while True:
        parent = toolkit.get_parent(current_node, True)
        # Si pas de parent trouvé ou si le parent est NamedThing, on s'arrête
        if not parent or "named thing" == parent or "related to" == parent or "biolink:NamedThing" == parent or "biolink:related_to" == parent:
            return current_node, depth
        else:
            current_node = parent
            depth += 1

def get_biolink_info():

    
    toolkit = bmt.toolkit.Toolkit()
    # Cache pour éviter de recalculer la hiérarchie pour chaque ligne
    # Structure: { category_name: (biolink_branch, depth) }

    # Get biolink elements:
    biolink_elements = toolkit.get_all_elements(formatted=True)
    biolink_category_mapping = {}
    for el in biolink_elements:
        biolink_category_mapping[el] = {}
        branch, depth = get_highest_parent_below_root(toolkit, el)
        # TO DO PROPERLY: need to check the classes composing the mixin
        # val1, val2 = extract_mixin_values(branch)
        # if val1 is not None and val2 is not None != 0:
        #     biolink_category_mapping[el]['biolink_branch'] = [val1,val2]
        # else:
        #     biolink_category_mapping[el]['biolink_branch'] = [branch]

        biolink_category_mapping[el]['biolink_branch'] = branch
        biolink_category_mapping[el]['biolink_depth'] = depth

        # Externalize mapping:
        # el = toolkit.get_element(cat_name)
        
        # # branch, depth = get_highest_parent_below_namedthing(toolkit, cat_name)
        # if 'name' in el:
        #     if el['name'] in biolink_category_mapping:
        #         branch = biolink_category_mapping[el['name']]['biolink_branch']
        #         depth = biolink_category_mapping[el['name']]['biolink_depth']
        #         category_mapping[id]['biolink_branch'] = branch
        #         category_mapping[id]['biolink_depth'] = depth
        #     else:
        #         print(f'{el['name'] not in biolink_category_mapping}')

    return biolink_category_mapping


def compute_KGX_node_metrics(kgx_dict,category_mapping_dict):
    node_degree = nodes_degree(kgx_dict)
    biolink_info = get_biolink_info() # this essentially has more than just the node info
    node_category = map_id_to_biolink_category(list(node_degree.keys()),category_mapping_dict)

    node_metrics = dict()
    for node_id,degree in node_degree.items():
        if node_id not in node_metrics:
            node_metrics[node_id] = {}
            node_metrics[node_id]['degree'] = degree
            node_metrics[node_id]['biolink_category'] = node_category[node_id]['category']
            node_metrics[node_id]['name'] = node_category[node_id]['name']
            node_metrics[node_id]['biolink_branch'] = biolink_info[node_metrics[node_id]['biolink_category']]['biolink_branch']
            node_metrics[node_id]['biolink_depth'] = biolink_info[node_metrics[node_id]['biolink_category']]['biolink_depth']
        else:
            print(f'Multiple nodes {node_id} have been found.')

    return node_metrics,biolink_info

