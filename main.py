import os
import KGX_node_metrics
import json
import csv
from tqdm import tqdm
import sys
import re

def extract_mixin_values(input_string):
    # Pattern explanation:
    # %s      -> matches literal '%s'
    # to      -> matches literal 'to'
    # mixin   -> matches literal 'mixin'
    pattern = r"(.*?)to(.*?)mixin"
    
    match = re.search(pattern, input_string)
    
    if match:
        # Extract the captured groups and convert to integers (or keep as strings)
        s1 = match.group(1)
        s2 = match.group(2)
        return s1, s2
    else:
        return None, None


def read_KGX(input_KGX_file,headers_of_interest = ['subject','object','predicate','publications']):
    # need to verify that data['id'] is unique
    kgx_dict = dict()
     
    
    with open(input_KGX_file, encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        print(f'Total lines:{total_lines}')
    cpt = 1
    with open(input_KGX_file, encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Import KGX"):
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
                        elif k != 'arg': # Note: user code had 'id' check, keeping logic as is
                            additional_data[k] = data[k]
                    kgx_dict[data['id']]['additional_data'] = additional_data
            except (json.JSONDecodeError, KeyError):
                print(f'Read line {cpt} issue')
                continue
            cpt += 1

    return kgx_dict




def read_KGX_category_mapping(mapping_file):

    category_mapping = dict()

    # line counter:
    with open(mapping_file, encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        print(f'Total lines:{total_lines}')
    
    cpt = 1
  
    with open(mapping_file, encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing category mapping and biolink model metrics"):
            try:
                data = json.loads(line)
                id = data['id']
                if len(id) != 0:
                    if id not in category_mapping.keys():
                        category_mapping[id] = {}
                        cat_name = data['category'][0]
                        category_mapping[id]['category'] = cat_name
                        category_mapping[id]['name'] = data['name']

            except (json.JSONDecodeError, KeyError):
                print(f'Read line {cpt} issue')
                continue
        cpt += 1
    return category_mapping

def save_to_csv(data, output_file):
    """
    Sauvegarde une liste de dictionnaires en CSV de manière optimisée.
    """
    if not data:
        print("Aucune donnée à sauvegarder dans le CSV.")
        return

    # On utilise les clés du premier dictionnaire comme noms de colonnes
    fieldnames = list(data[0].keys())

    try:
        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Fichier CSV sauvegardé avec succès : {output_file}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du CSV : {e}")

def main(kgx_dict,category_mapping_dict):
    # Compute metrics:
    KGX_nodes_metrics,biolink_info = KGX_node_metrics.compute_KGX_node_metrics(kgx_dict,category_mapping_dict)

    KGX_metrics = []
    total_lines = len(kgx_dict.keys())
    for edge_id in tqdm(kgx_dict.keys(), total=total_lines, desc="Mapping metrics"):
        subject = kgx_dict[edge_id]['subject']
        object = kgx_dict[edge_id]['object']
        predicate = kgx_dict[edge_id]['predicate']

        KGX_edge_metrics = {}
        # append subject metrics:
        KGX_edge_metrics['id'] = edge_id
        KGX_edge_metrics['subject'] = subject
        KGX_edge_metrics['subject_degree'] = KGX_nodes_metrics[subject]['degree']
        KGX_edge_metrics['subject_biolink_category'] = KGX_nodes_metrics[subject]['biolink_category']
        KGX_edge_metrics['subject_biolink_branch'] = KGX_nodes_metrics[subject]['biolink_branch']
        KGX_edge_metrics['subject_biolink_depth'] = KGX_nodes_metrics[subject]['biolink_depth']
        KGX_edge_metrics['subject_name'] = KGX_nodes_metrics[subject]['name']

        # append object metrics:
        KGX_edge_metrics['object'] = object
        KGX_edge_metrics['object_degree'] = KGX_nodes_metrics[object]['degree']
        KGX_edge_metrics['object_biolink_category'] = KGX_nodes_metrics[object]['biolink_category']
        KGX_edge_metrics['object_biolink_branch'] = KGX_nodes_metrics[object]['biolink_branch']
        KGX_edge_metrics['object_biolink_depth'] = KGX_nodes_metrics[object]['biolink_depth']
        KGX_edge_metrics['object_name'] = KGX_nodes_metrics[object]['name']

        # append predicate metrics:
        KGX_edge_metrics['predicate'] = predicate
        KGX_edge_metrics['predicate_biolink_branch'] = biolink_info[predicate]['biolink_branch']
        KGX_edge_metrics['predicate_biolink_depth'] = biolink_info[predicate]['biolink_depth']
        KGX_edge_metrics['publications_number'] = len(kgx_dict[edge_id])

        KGX_metrics.append(KGX_edge_metrics)


    return KGX_metrics


if __name__ == "__main__":
    # load data (TO BE UPDATED AFTER AUTOMATION):
    input_KGX_file = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/normalized_edges.jsonl'
    biolink_id_to_category_mapping = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/merged_nodes.jsonl'
    output_file_json = 'data/KGX_computed_edges_metrics.json'
    output_file_csv = 'data/KGX_computed_edges_metrics.csv'

    # Vérification de l'existence des fichiers avant de commencer
    required_files = [os.path.abspath(input_KGX_file), os.path.abspath(biolink_id_to_category_mapping)]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("ERREUR : Les fichiers suivants sont introuvables. Veuillez vérifier vos chemins :")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    kgx_dict = read_KGX(input_KGX_file)
    category_mapping_dict = read_KGX_category_mapping(biolink_id_to_category_mapping)

    KGX_edge_metrics = main(kgx_dict,category_mapping_dict)

    os.makedirs(os.path.dirname(output_file_json), exist_ok=True)
    
    # Sauvegarde JSON
    with open(output_file_json, 'w', encoding='utf-8') as f:
        json.dump(KGX_edge_metrics, f, indent=4)
    print(f"Fichier JSON sauvegardé : {output_file_json}")

    # Sauvegarde CSV
    save_to_csv(KGX_edge_metrics, output_file_csv)
