import os
import KGX_node_metrics
import KGX_metrics_design
import json
import csv
from tqdm import tqdm
import sys
import re
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


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


def read_KGX_to_parquet(input_KGX_file):
    """
    Reads a JSONL KGX file and writes a flattened Parquet file.
    Every key in the original JSON becomes a column in the Parquet file.
    The 'publications' list is exploded into multiple rows based on PMID.
    """
    # We use a temporary JSONL to avoid keeping everything in RAM
    temp_flat_jsonl = "temp_flattened_data.jsonl"
    root, ext = os.path.splitext(input_KGX_file)
    output_parquet_file = root + ".parquet"

    # 1. Count lines for tqdm progress bar
    with open(input_KGX_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"Total lines to process: {total_lines}")

    # 2. Stream through the input file
    with open(input_KGX_file, 'r', encoding='utf-8') as f_in, \
         open(temp_flat_jsonl, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Streaming & Flattening KGX"):
            try:
                data = json.loads(line)
                if 'id' not in data:
                    continue
                
                # Prepare the base record: 
                base_record = data.copy()
                # base_record['idx'] = base_record.pop('id')
                
                # Extract publications for the explosion
                publications = data.get('publications', [])
                
                if not publications:
                    # If there are no publications, we still want the record, 
                    # but with PMID as None so it can still join/exist.
                    base_record['PMID'] = None
                    f_out.write(json.dumps(base_record) + '\n')
                else:
                    # THE EXPLOSION: Create one row for every PMID found
                    for pmid in publications:
                        flat_record = base_record.copy()
                        flat_record['PMID'] = pmid
                        f_out.write(json.dumps(flat_record) + '\n')

            except (json.JSONDecoderonError, KeyError) as e:
                # It's better to see what went wrong during debugging
                # print(f"Error processing line: {e}")
                continue

    # 3. Conve6rt the flattened JSONL into a high-performance Parquet
    print("Converting flattened JSONL to Polars DataFrame...")
    try:
        df_final = pl.read_ndjson(temp_flat_jsonl)
        
        print(f"Writing to {output_parquet_file}...")
        df_final.write_parquet(output_parquet_file)
        
        # 4. Clean up the temporary file
        if os.path.exists(temp_flat_jsonl):
            os.remove(temp_flat_jsonl)
            
        print(f"Done! Successfully saved to {output_parquet_file}")
        return output_parquet_file

    except Exception as e:
        print(f"Error during Polars conversion: {e}")
        if os.path.exists(temp_flat_jsonl):
            os.remove(temp_flat_jsonl)
        raise

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

    # On utilise les clés du premier dictionnaire comme noms de columns
    fieldnames = list(data[0].keys())

    try:
        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Fichier CSV sauvegardé avec succès : {output_file}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du CSV : {e}")

def save_to_parquet(updated_data, output_file_path):
    """
    Optimized conversion of a large list of dictionaries to Parquet.
    Uses PyArrow's C++ implementation to avoid Python-level iteration overhead.
    
    Args:
        updated_data (list of dict): The KG edge data (1M+ rows).
        output_file_path (str): Destination path.
    """
    if not updated_data:
        print("Error: The data list is empty. Nothing to save.")
        return

    try:
        print(f"Starting conversion of {len(updated_data)} rows...")

        # 1. Use PyArrow to convert the list of dicts directly to a Table.
        # 'from_pylist' is implemented in C++ and is significantly faster 
        # and more memory-efficient than Polars or Pandas for this specific input type.
        table = pa.Table.from_pylist(updated_data)

        # 2. Write the Arrow Table directly to Parquet.
        # We skip the step of converting to Polars entirely to save memory.
        # Writing via PyArrow is the industry standard for high-performance Parquet.
        pq.write_table(table, output_file_path, compression='snappy')

        print(f"Successfully saved {len(updated_data)} edges to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred during saving: {e}")
        raise


def map_metrics_to_KGX(kgx_dict,category_mapping_dict):
    # Compute metrics:
    KGX_nodes_metrics,biolink_info = KGX_node_metrics.compute_KGX_node_metrics(kgx_dict,category_mapping_dict)

    ## Append to dict:
    KGX_metrics = []
    total_lines = len(kgx_dict.keys())
    for edge_id in tqdm(kgx_dict.keys(), total=total_lines, desc="Mapping metrics to KGX"):
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
        KGX_edge_metrics['publications'] = kgx_dict[edge_id]['publications']
        KGX_edge_metrics['publications_number'] = len(kgx_dict[edge_id]['publications'])

        KGX_metrics.append(KGX_edge_metrics)
        # ## Flatten dict:
        # for pub in kgx_dict[edge_id]['publications']:
        #     KGX_edge_metrics['publications'] = pub
        #     KGX_metrics.append(KGX_edge_metrics)
    

    return KGX_metrics

def build_edges_test_suite(kg_path, ml_results_path, sample_size=20):
    """
    Builds a stratified test suite by joining exploded KG evidence 
    with ML predictions and sampling PMIDs per stratum.
    
    Args:
        kg_path (str): Path to the Parquet file containing KG edges (with 'publications' list).
        ml_results_path (str): Path to the Parquet file with ML predictions (contains 'predicted').
        sample_szie (int): Number of PMIDs to sample per stratum.
        
    Returns:
        pl.DataFrame: The final sampled test suite.
    """
    
    # 1. Load the KG Data (Object A)
    # This contains your features: complexity, asymmetry, is_hub_edge, etc.
    kg_df = pl.read_parquet(kg_path)

    # 2. Flatten (Explode) the publications
    # Each row now represents one specific PMID for a specific edge.
    # All edge-level features (complexity, etc.) are duplicated across these rows.
    exploded_kg = kg_df.explode("publications")

    # 3. Load the ML Results
    # This file must contain: ['subject', 'predicate', 'object', 'predicted']
    ml_results = pl.read_parquet(ml_results_path)

    # 4. Perform the Join
    # We join on the triple identity. The 'predicted' column is brought into our exploded KG.
    # We use an 'inner' join to ensure we only test PMIDs that actually have a prediction.
    joined_df = exploded_kg.join(
        ml_results, 
        on=['subject', 'predicate', 'object'], 
        how='inner'
    )

    # 5. Stratified Sampling
    # We identify all columns that define our 'strata' (everything except the unique ID and PMID)
    # We group by these strata and take the first N PMIDs found in each group.
    
    # Identify grouping columns: everything except 'id' and 'publications'
    group_cols = [
        col for col in joined_df.columns 
        if col not in ['id', 'publications', 'predicted']
    ]

    # Perform the sampling
    # .head(sample_size) is extremely fast in Polars for this purpose.
    test_suite = (
        joined_df
        .group_by(group_cols)
        .head(sample_size)
    )

    return test_suite

def main(input_KGX_file,biolink_id_to_category_mapping,LLM_checker_results_file,save_files = True):


    # Vérification de l'existence des fichiers avant de commencer
    required_files = [os.path.abspath(input_KGX_file), os.path.abspath(biolink_id_to_category_mapping)]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("ERREUR : Les fichiers suivants sont introuvables. Veuillez vérifier vos chemins :")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    kgx_dict = read_KGX(input_KGX_file)
    # kgx_parquet = read_KGX_to_parquet(input_KGX_file)
    category_mapping_dict = read_KGX_category_mapping(biolink_id_to_category_mapping)

    # calculate metrics and map to KGX:
    KGX_edge_metrics = map_metrics_to_KGX(kgx_dict,category_mapping_dict)

    design_dict = {'id':'ignore',
                    'subject':'ignore',
                    'subject_degree': 'powerlaw',
                    'subject_biolink_category':'ignore',
                    'subject_biolink_branch':'discrete',
                    'subject_biolink_depth':'discrete',
                    'subject_name':'ignore',
                    'object':'ignore',
                    'object_degree': 'powerlaw',
                    'object_biolink_category':'ignore',
                    'object_biolink_branch':'discrete',
                    'object_biolink_depth':'discrete',
                    'object_name':'ignore',
                    'object_biolink_category':'ignore',
                    'predicate':'ignore',
                    'predicate_biolink_branch':'discrete',
                    'predicate_biolink_depth':'discrete',
                    'publications_number':'powerlaw'
                    }
    # metrics_transformed,sampled_data = KGX_metrics_design.main(KGX_edge_metrics,design_dict)
    metrics_transformed = KGX_metrics_design.main(KGX_edge_metrics,design_dict)

    output_file_parquet = 'data/KGX_computed_edges_transformed_metrics.parquet'
    output_file_csv = 'data/KGX_computed_edges_transformed_metrics.csv'
    if save_files:
        print('Save transformed metrics:')
        os.makedirs(os.path.dirname(output_file_parquet), exist_ok=True)
        os.makedirs(os.path.dirname(output_file_csv), exist_ok=True)

        save_to_csv(metrics_transformed, output_file_csv) # save csv
        save_to_parquet(metrics_transformed, output_file_parquet) # save parquet



    ## JOINTURE AVEC RESULTS
    #### Transform metrics_transformed into pl format
    LLM_checker_dataset = pl.read_parquet(LLM_checker_results_file)
    KGX_metrics_dataset = pl.read_parquet(LLM_checker_results_file)


    return metrics_transformed

if __name__ == "__main__":
    # load data (TO BE UPDATED AFTER AUTOMATION):
    input_KGX_file = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/normalized_edges.jsonl'
    biolink_id_to_category_mapping = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/merged_nodes.jsonl'
    LLM_checker_results_file = 'data/LLM_Pmid_Evaluation_SemMedDB_v1.0/results.parquet'
    main(input_KGX_file,biolink_id_to_category_mapping,LLM_checker_results_file,False)
