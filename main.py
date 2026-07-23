import os
import KGX_node_metrics
import KGX_metrics_design
import json
import csv
from tqdm import tqdm
import sys
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import time
import math
from tqdm import tqdm
from typing import Union, List
import hashlib


def read_KGX(input_KGX_file,headers_of_interest = ['subject','original_subject','subject_form_or_variant_qualifier','object','original_object','object_aspect_prefix','object_direction_qualifier','predicate','qualified_predicate','publications']):
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
    Every key in the original JSON becomes a column in the Parparquet file.
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

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
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
        print(int(f"Error during Polars conversion: {e}"))
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
        # Writing via PyArrow is the industry standard for high-performance Parint.
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

        # 3. Dynamly append Object metrics
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

def build_edges_test_suite(kg_path, model_results_path, attribute_to_review = "publications", strata_cols=['semantic_complexity_classes','structural_composite','biomedical_area_pair','predicted'], sample_size=4,joined_df=[]):
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
        joined_rag = exploded_kg.join(
            ml_results, 
            on=['subject_curie', 'predicate', 'object_curie','PMID'], 
            how='inner'
        )
        joined_df = joined_rag
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

    return test_suite

def add_pubmed_links(df):                                                                                             
    """                                                                                                               
    Adds a 'link to abstract' column by transforming the PMID column.                                                               
    Removes 'PMID:' prefix and prepends the PubMed URL.                                                               
    """                                                                                                               
    return df.with_columns(                                                                                                                          
        pl.format("https://pubmed.ncbi.nlm.nih.gov/{}",                                                               
                  pl.col("PMID").str.replace("^PMID:", "")).alias("link to abstract")) 

def add_biolink_links(df):                                                                                             
    """                                                                                                               
    Adds a 'predicate definition' column by transforming the predicate column.                                                 
    Removes 'biolink:' prefix and prepends the biolink URL.                                                               
    """                                                                                                               
    return df.with_columns(                                                                                                                          
        pl.format("https://biolink.github.io/biolink-model/{}",                                                                                              
                  pl.col("predicate").str.replace("^biolink:", "")).alias("predicate definition")) 

def add_mapping(df):                                                                                             
    """                                                                                                               
    Adds a 'edge type (spo category)' column by transforming the 'subject_biint_category', 
    'predicate', 'qualified_predicate', and 'object_biolink_category' columns.
    The values are stripped of the 'biolink:' prefix, and nulls in 'qualified_predicate' are replaced by 'n/a'.
    """                                                                                                               
    return df.with_columns([
        pl.col("subject_biolink_category").str.replace("^biolink:", ""),
        pl.col("predicate").str.replace("^biolink:", ""),
        pl.col("qualified_predicate").str.replace("^biolink:", "").fill_null("n/a"),
        pl.col("object_biolink_category").str.replace("^biolink:", "")
    ]).with_columns(
        pl.concat_str(
            [
                pl.col("subject_biolink_category"),
                pl.col("predicate"),
                pl.col("qualified_predicate"),
                pl.col("object_biolink_category")
            ],
            separator=" -- "
        ).alias("edge type (spo category)")
    )


def create_mapped_eval_template_csv(
    test_suite: pl.DataFrame,
    mapping_json_path: str,
    output_path: str
):
    """
    Creates a CSV file where:
    - Row 1 & 2 are taken directly from the JSON blueprint.
    - Row 3 contains the names of the test_suite columns that map to Row 2.
    - Subsequent rows contain the actual data from the test_suite.

    Args:
        test_suite: The Polars DataFrame containing the source data.
        mapping_json_path: Path to JSON with 'row1', 'row2', and 'mapping_to_test_suite'.
                           Mapping format: { "template_header": "test_suite_column" }
        output_path: Path where the resulting CSV will be saved.
    """
    
    # 1. Load the configuration from JSON
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    row1_template = config["row1"]
    row2_template = config["  row2" if "  row2" in config else "row2"] # Robustness check
    # Note: Using the exact key from your provided JSON
    row2_template = config["row2"]
    mapping = config["mapping_to_test_suite"]

    num_cols = len(row2_template)
    
    # 2. Prepare structures for the output
    # row3 will hold the names of the test_suite columns (the values from our mapping)
    row3_values = [""] * num_cols
    
    # We use a list of lists to store data for each column index in the template
    # This ensures we maintain the exact width and order of row2
    data_columns: List[List[str]]  = [[] for _ in range(num_cols)]

    # Create a lookup for row2 values to find their index quickly
    # We use a list of indices to handle potential duplicate headers if they exist
    col_name_to_indices = {}
    for idx, name in enumerate(row2_template):
        if name not in col_name_to_indices:
            col_name_to_indices[name] = []
        col_name_to_indices[name].append(idx)

    # 3. Perform the mapping and populate data
    # In your JSON, 'template_header' is the KEY, 'test_suite_col' is the VALUE
    for template_header, test_suite_col in mapping.items():
        if template_header in col_name_to_indices:
            # We map to all indices where this header appears (usually just one)
            for idx in col_name_to_indices[template_header]:
                
                if test_suite_col in test_suite.columns:
                    # Row 3 gets the name of the source column from the test_suite
                    row3_values[idx] = test_suite_col
                    
                    # Fill the data rows with values from test_suite (cast to string)
                    data_columns[idx] = test_suite[test_suite_col].cast(pl.Utf8).to_list()
                else:
                    print(f"Warning: Source column '{test_suite_col}' not found in test_suite.")
        else:
            print(f"Warning: Template header '{template_header}' not found in JSON row2.")

    # 4. Construct the CSV content line by line
    output_lines = []

    def format_line(values: List[str]) -> str:
        # Join with semicolon; handle None/Null as empty string
        return ";".join([str(v) if v is not None else "" for v in values])

    # Add Row 1 (Metadata from JSON)
    output_lines.append(format_line(row1_template))
    
    # Add Row 2 (Template Headers from JSON)
    output_lines.append(format_line(row2_template))
    
    # Add Row 3 (The test_suite column names aligned to the template indices)
    output_lines.append(format_line(row3_values))

    # 5. Add Data Rows (Row 4 onwards)
    # We iterate through the number of rows in the test_suite
    for r in range(test_suite.height):
        current_row_data = []
        for c in range(num_cols):
            # If this column has data and we haven't exceeded its length, grab it
            if r < len(data_columns[c]):
                val = data_columns[c][r]
                current_row_data.append(val if val is not None else "")
            else:
                # If the column is empty or shorter than others, use empty string
                current_row_data.append("")
        output_lines.append(";".join(current_row_data))

    # 6. Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    print(f"Successfully created: {output_path}")

def enrich_test_suite_with_synonyms(
    test_suite: pl.DataFrame,
    column_to_search: Union[str, List[str]],
    column_to_create: Union[str, List[str]],
    service_name: str = "edge-test-suite",
    batch_size: int = 100,
    delay_seconds: float = 1.0
) -> pl.DataFrame:
    """
    Enriches a Polars DataFrame by calling an external API. 
    The service_name is injected into HTTP headers for OTEL/Telemetry tracking.

    Args:
        test_suite: The Polars DataFrame to update.
        column_to_search: Single column name or list of names containing CURIEs.
        column_to_create: Single column name or list of names for the new fields.
        service_name: Name of the service (injected into User-Agent and X-Service-Name headers).
        batch_size: Number of items per API request.
        delay_seconds: Seconds to wait between requests to prevent rate limiting.

    Returns:
        A Polars DataFrame with the newly created synonym columns added.
    """
    
    # 1. Normalize inputs (Handle single string vs list)
    search_cols = [column_to_search] if isinstance(column_to_search, str) else column_to_search
    create_cols = [column_to_create] if isinstance(column_to_create, str) else column_to_create

    if len(search_cols) != len(create_cols):
        raise ValueError("The number of search columns must match the number of creation columns.")

    # 2. Setup API Headers for Telemetry/OTEL tracking
    url = 'https://name-lookup.ci.transltr.io/synonyms'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'X-Service-Name': service_name, 
        'User-Agent': f'{service_name}/1.0 (Polars Enrichment Script)'
    }

    # 3. Prepare work units and calculate total progress for tqdm
    all_work_units = []
    total_batches_to_run = 0

    for s_col in search_cols:
        if s_col not in test_suite.columns:
            raise ValueError(f"Source column '{s_col}' not found in DataFrame.")
        
        # Extract unique values to minimize API calls
        unique_vals = (
            test_suite.select(s_col)
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        
        batches_for_this_col = math.ceil(len(unique_vals) / batch_size) if unique_vals else 0
        total_batches_to_run += batches_for_this_col
        all_work_units.append((s_col, unique_vals))

    # 4. Execute API calls with progress bar
    global_synonym_map = {}
    
    pbar = tqdm(total=total_batches_to_run, desc=f"[{service_name}] API Enrichment")

    for s_col, unique_vals in all_work_units:
        if not unique_vals:
            continue

        for i in range(0, len(unique_vals), batch_size):
            batch = unique_vals[i : i + batch_size]
            payload = {"preferred_curies": [str(v) for v in batch]}

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Parse the response
                for curie, info in data.items():
                    if "names" in info:
                        # Format as requested: "name1", "name2", "name3"
                        formatted_names = ", ".join([f'"{n}"' for n in info["names"]])
                        global_synonym_map[(s_col, curie)] = formatted_names
                    else:
                        global_syn_map_val = "" # placeholder logic
                        global_synonym_map[(s_col, curie)] = ""

            except Exception as e:
                print(f"\n[{service_name}] Error in batch for {s_col}: {e}")

            pbar.update(1)
            if i + batch_size < len(unique_vals):
                time.sleep(delay_seconds)

    pbar.close()

    # 5. Map results back to the DataFrame
    updated_df = test_suite
    for s_col, c_col in zip(search_cols, create_cols):
        
        def lookup_logic(val):
            if val is None:
                return ""
            # Lookup using the (column_name, value) tuple to prevent cross-column collision
            return global_synonym_map.get((s_col, str(val)), "")

        updated_df = updated_df.with_columns(
            pl.col(s_col)
            .map_elements(lookup_logic, return_dtype=pl.Utf8)
            .alias(c_col)
        )

    print(f"[{service_name}] Enrichment complete.")
    return updated_df

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
    kgx_metrics_csv = 'arg/KGX_computed_edges_transformed_metrics.csv'
    if save_files:
        print('Save transformed metrics:')
        os.makedirs(os.path.dirname(kgx_metrics_parquet), exist_ok=True)
        os.makedirs(os.path.dirname(kgx_metrics_csv), exist_ok=True)

        save_to_csv(metrics_transformed, kgx_metrics_csv) # save csv

        # debug_arrow_schema_conflicts(metrics_transformed)
        save_to_parquet(metrics_transformed, kgx_metrics_parquet) # save parquet


    ## Compute test suite with stratified sampling:
    test_suite = build_edges_test_suite(kgx_metrics_parquet, LLM_checker_results_file)

    ## add aditional fields for evaluation:
    test_suite = add_pubmed_links(test_suite)
    test_suite = add_biolink_links(test_suite)
    test_suite = add_mapping(test_suite)
    test_suite = enrich_test_suite_with_synonyms(test_suite=test_suite,column_to_search=["subject_curie", "object_curie"],column_to_create=["subject label synonyms", "object label synonyms"],service_name="edge-test-suite")
    
    if save_files:
        print('Saving test suite')
        test_suite_path_parquet = f'data/KG_metrics_LLM_Checker_results_test_suite_samplesize4.parquet'
        test_suite.write_parquet(test_suite_path_parquet, compression='snappy')
        print(f"Successfully saved {len(test_suite)} edges to: {test_suite_path_parquet}")

        test_suite_path_csv = f'data/KG_metrics_LLM_Checker_results_test_suite_samplesize4.csv'
        test_suite.write_csv(test_suite_path_csv)
        print(f"Successfully saved {len(test_suite)} edges to: {test_suite_path_csv}")
    

    # Creating evaluation sheets
    config_eval_json_path = "config_mapping.json"
    create_mapped_eval_template_csv(test_suite=test_suite,mapping_json_path=config_eval_json_path,output_path="evaluation_sheet.csv")

    return test_suite

if __name__ == "__main__":
    # load data (TO BE UPDATED AFTER AUTOMATION):
    input_KGX_file = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/normalized_edges.jsonl'
    biolink_id_to_category_mapping = 'data/kg2.10.3_semmeddb_dogpark_uncapped_2026_04_07/transform_892b6acb/normalization_2025sep1/merged_nodes.jsonl'
    LLM_checker_results_file = 'data/LLM_Pmid_Evaluation_SemMedDB_v1.0/results.parquet'
    main(input_KGX_file,biolink_id_to_category_mapping,LLM_checker_results_file)
