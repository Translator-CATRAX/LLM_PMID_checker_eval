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


def read_KGX(input_KGX_file,headers_of_interest = ['subject','original_subject','subject_form_or_variant_qualifier','object','original_object','object_aspect_qualifier','object_direction_qualifier','predicate','qualified_predicate','publications']):
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
                        kgx_dict[data['id']][h] = None
                    for k in data.keys():
                        if k in headers_of_interest and k != 'id':
                            kgx_dict[data['id']][k] = data[k]

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

def map_metrics_to_KGX(kgx_dict, category_mapping_dict):
    """
    Maps node and predicate metrics to KGX edges dynamically.
    
    This version is modular: it preserves all original keys in kgx_dict 
    and appends any available metrics from KGX_nodes_metrics and biolink_info
    without needing to know the specific key names in advance.
    """
    # Compute metrics (Assuming this function exists as per your snippet)
    KGX_nodes_metrics, biolink_info = KGX_node_metrics.compute_KGX_node_metrics(kgx_dict, category_mapping_dict)

    KGX_metrics = []
    total_lines = len(kgx_dict)

    for edge_id, edge_data in tqdm(kgx_dict.items(), total=total_lines, desc="Mapping metrics to KGX"):
        # 1. Start with all original keys from the kgx_dict entry
        # We use .copy() to avoid mutating the original input dictionary
        edge_metrics = edge_data.copy()
        edge_metrics['id'] = edge_id

        # 2. Dynamically append Subject metrics
        subject_id = edge_data.get('subject')
        if subject_id in KGX_nodes_metrics:
            for key, value in KGX_nodes_metrics[subject_id].items():
                edge_metrics[f'subject_{key}'] = value

        # 3. Dynamically append Object metrics
        object_id = edge_data.get('object')
        if object_id in KGX_nodes_metrics:
            for key, value in KGX_nodes_metrics[object_id].items():
                edge_metrics[f'object_{key}'] = value

        # 4. Dynamically append Predicate metrics
        predicate = edge_data.get('predicates') if 'predicates' in edge_data else edge_data.get('predicate')
        if predicate in biolink_info:
            for key, value in biolink_info[predicate].items():
                edge_metrics[f'predicate_{key}'] = value

        # 5. Add derived metrics (like publications count) if the source data exists
        if 'publications' in edge_data and isinstance(edge_data['publications'], list):
            edge_metrics['publications_number'] = len(edge_data['publications'])

        KGX_metrics.append(edge_metrics)

    return KGX_metrics

def build_edges_test_suite(kg_path, model_results_path, attribute_to_review = "publications", strata_cols=['semantic_complexity_classes','is_hub_edge','degree_assymetry_classes','biomedical_area_pair','predicted'], sample_size=4,joined_df=[]):
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
    if len(joined_df) == 0:
        # 1. Load the KG Data (Object A)
        # This contains your features: complexity, asymmetry, is_hub_edge, etc.
        kg_df = pl.read_parquet(kg_path)

        # 2. Flatten (Explode) the publications
        # Each row now represents one specific PMID for a specific edge.
        # All edge-level features (complexity, etc.) are duplicated across these rows.
        print(f'Explode by attribute: {attribute_to_review}:')
        exploded_kg = kg_df.explode(attribute_to_review,empty_as_null=True)
        exploded_kg = exploded_kg.rename({"subject": "subject_curie","object": "object_curie","publications": "PMID"}) # TO DO outsource mapping file
        print(f'Explode by attribute: {attribute_to_review}: Done.')

        # 3. Load the model results
        # This file must contain: ['subject', 'predicate', 'object', 'predicted'] # TO DO CHECK
        print(f'Load the model results in: {model_results_path}:')
        ml_results = pl.read_parquet(model_results_path)
        print(f'Load the model results in: {model_results_path}: Done.')

        # 4. Perform the Join
        # We join on the triple identity. The 'predicted' column is brought into our exploded KG.
        # We use an 'inner' join to ensure we only test PMIDs that actually have a prediction.
        print(f'Joining datasets:')
        joined_df = exploded_kg.join(
            ml_results, 
            on=['subject_curie', 'predicate', 'object_curie','PMID'], 
            how='inner'
        )
        print(f'Joining datasets: Done.')
        joined_df_path = 'data/KG_metrics_LLM_Checker_results.parquet'
        joined_df.write_parquet(joined_df_path, compression='snappy')
        print(f"Successfully saved {len(joined_df)} edges to: {joined_df_path}")


    # 5. Stratified sampling: identify all columns that define the strata from strata_cols 
    # Shuffle the entire DataFrame first
    # This ensures that when we pick the "top 20" from a group, 
    # they are random and not just the first 20 encountered in the original file (which is definitely NOR random).
    joined_df = joined_df.sample(fraction=1.0, shuffle=True)
    
    # Identify grouping columns: perform stratified sampling using window functions
    test_suite = (
        joined_df
        # 1. Handle Nulls first so they don't break the string concatenation
        .with_columns([
            pl.col(c).fill_null(strategy="zero") if joined_df[c].dtype.is_integer() 
            else pl.col(c).fill_null("Unknown") 
            for c in strata_cols
        ])
        .with_columns([
            # 2. Create the 'strata_id' (The human-readable ID)
            # This joins all features into one string: "Class1 | True | Class0 | AreaA"
            pl.concat_str(
                [pl.col(c).cast(pl.Utf8) for c in strata_cols], 
                separator=" | "
            ).alias("strata_id"),

            # 3. Create the 'group_idx' (The mathematical counter used for filtering)
            # This is the part that resets to 0, 1, 2... for every group
            pl.int_range(0, pl.len()).over(strata_cols).alias("group_idx")
        ])
        # 4. Filter using the counter
        .filter(pl.col("group_idx") < sample_size)
        # 5. Drop the counter so you only see the human-readable ID in your final CSV
        .drop("group_idx")
    )

    test_suite_path_parquet = f'data/KG_metrics_LLM_Checker_results_test_suite_{sample_size}.parquet'
    test_suite.write_parquet(test_suite_path_parquet, compression='snappy')
    print(f"Successfully saved {len(test_suite)} edges to: {test_suite_path_parquet}")

    test_suite_path_csv = f'data/KG_metrics_LLM_Checker_results_test_suite_{sample_size}.csv'
    test_suite.write_csv(test_suite_path_csv)
    print(f"Successfully saved {len(test_suite)} edges to: {test_suite_path_csv}")


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
    category_mapping_dict = read_KGX_category_mapping(biolink_id_to_category_mapping)

    # calculate metrics and map to KGX:
    KGX_edge_metrics = map_metrics_to_KGX(kgx_dict,category_mapping_dict)

    design_dict = {'subject_degree': 'powerlaw',
                    'subject_biolink_branch':'discrete',
                    'subject_biolink_depth':'discrete',
                    'object_degree': 'powerlaw',
                    'object_biolink_branch':'discrete',
                    'object_biolink_depth':'discrete',
                    'predicate_biolink_branch':'discrete',
                    'predicate_biolink_depth':'discrete',
                    'publications_number':'powerlaw'
                    }

    metrics_transformed = KGX_metrics_design.main(KGX_edge_metrics,design_dict)

    kgx_metrics_parquet = 'data/KGX_computed_edges_transformed_metrics.parquet'
    kgx_metrics_csv = 'data/KGX_computed_edges_transformed_metrics.csv'
    if save_files:
        print('Save transformed metrics:')
        os.makedirs(os.path.dirname(kgx_metrics_parquet), exist_ok=True)
        os.makedirs(os.path.dirname(kgx_metrics_csv), exist_ok=True)

        save_to_csv(metrics_transformed, kgx_metrics_csv) # save csv

        # debug_arrow_schema_conflicts(metrics_transformed)
        save_to_parquet(metrics_transformed, kgx_metrics_parquet) # save parquet


    ## Compute test suite with stratified sampling:
    test_suite = build_edges_test_suite(kgx_metrics_parquet, LLM_checker_results_file)

    return test_suite

if __name__ == "__main__":
    # load data (TO BE UPDATED AFTER AUTOMATION):
    input_KGX_file = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/normalized_edges.jsonl'
    biolink_id_to_category_mapping = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/merged_nodes.jsonl'
    LLM_checker_results_file = 'data/LLM_Pmid_Evaluation_SemMedDB_v1.0/results.parquet'
    main(input_KGX_file,biolink_id_to_category_mapping,LLM_checker_results_file)
