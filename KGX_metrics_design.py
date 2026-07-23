import numpy as np
import random
import warnings
from tqdm import tqdm
import copy

# TO DO: merge classifications into 1 method with different distributions a priori

def KGX_edge_sampling(data,columns_type, sample_size=20):
    """
    Échantillonne les données en regroupant par toutes les propriétés 
    sauf l'identifiant unique 'id'.
    """
    if not data:
        return []

    groups = {}
    group_keys = [k for k,v in columns_type.items() if v != 'ignore'] # omit columns

    for row in data:
        # Construction d'une clé de groupe hashable
        key_values = []
        for k in group_keys:
            val = row[k]
            # Conversion des types non-hashables (set, list) en tuples pour le regroupement
            if isinstance(val, (set, list)):
                val = tuple(sorted(list(val)))
            elif isinstance(val, dict):
                val = tuple(sorted(val.items()))
            key_values.append(val)
        
        group_key = tuple(key_values)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(row)

    sampled_data = []
    for group in groups.values():
        # On échantillonne jusqu'à sample_size pour chaque groupe trouvé
        if len(prob_group := group) <= sample_size:
            sampled_data.extend(group)
        else:
            sampled_data.extend(random.sample(group, sample_size))

    return sampled_data

def classify_distribution(y):
    """
    Robust classification using Percentiles (Quantiles).
    This is immune to the 'inflation' caused by heavy tails.
    """
    y = np.array(y, dtype=float)
    if y.size == 0: return np.array([]), []
    
    if y.size < 2:
        return np.zeros_like(y), y

    # Define boundaries based on percentiles rather than Sigma
    # We define the 'Middle' as the bulk of the data (25th to 75th percentile)
    p05 = np.percentile(y, 5)   # 5th percentile (Extreme Low)
    p25 = np.percentile(y, 25)  # 25th percentile (Low)
    p75 = np.percentile(y, 75)  # 75th percentile (High)
    p95 = np.percentile(y, 95)  # 95th percentile (Extreme High)

    conditions = [
        (y <= p05),                          # -2: Extreme Low (Bottom 5%)
        (y > p05) & (y <= p25),             # -1: Low
        (y > p25) & (y <= p75),            #  0: Medium (The Interquartile Range)
        (y > p75) & (y < p95),              #  1: High
        (y >= p95)                           #  2: Extreme High (Top 5%)
    ]
    
    choices = [-2, -1, 0, 1, 2]
    results = np.select(conditions, choices, default=0)

    return results, y    

def log_distribution(x):
    """
    Log transformation
    """
    x = np.array(x, dtype=float)
    if x.size == 0: return np.log2(x + 1e-9)

    # Log transform as before
    y = np.log2(np.clip(x, 0, None) + 1e-9)

    return y

def transform_metrics(data, columns_type):                                                                                                                                                                                                      
    """                                                                                                                                                                                                                                                 
    Transforme les données en fonction des types de colonnes spécifiés.                                                                                                                                                                                 
    """                                                                                                                                                                                                                                                 
    if not data:                                                                                                                                                                                                                                        
        return []

    data_out = copy.deepcopy(data)                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                        
    # 1. Vérification de la cohérence des clés (tous les dictionnaires doivent avoir les primes clés)                                                                                                                                                   
    base_keys = set(data_out[0].keys())                                                                                                                                                                                                                     
    for i, row in enumerate(data_out):                                                                                                                                                                                                                      
        if set(row.keys()) != base_keys:                                                                                                                                                                                                                
            raise ValueError(f"Erreur de structure : le dictionnaire à l'index {i} n'a pas les mêmes clés que le premier.")                                                                                                                             
                                                                                                                                                    
    # Mais ici, on va modifier les valeurs des colonnes 'powerlaw' directement.                                                                                                                                                                         
                                                                                                                                                                                                                                                
    for column, col_type in tqdm(columns_type.items(), total=len(columns_type.keys()), desc="Data processing:"):                                                                                                                                                                                                       
        if column not in base_keys:                                                                                                                                                                                                                     
            continue                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                        
        if col_type == 'discrete':                                                                                                                                                                                                                      
            # Vérification si les valeurs sont "considérables" (seuil arbitraire de 100)                                                                                                                                                 
            values = set([row[column] for row in data_out])                                                                                                                                                                                                     
            if values and len(values) > 100:                                                                                                                                                                                                            
                warnings.warn(f"Attention : la colonne '{column}' est marquée comme 'discrete' mais contient un ensemble de valeurs élevées ({len(values)}).")                                                                                                     
                                                                                                                                                                                                                                                        
        elif col_type == 'powerlaw':                                                                                                                                                                                                                    
            # Extraction des valeurs actuelles pour la transformation                                                                                                                                                                                   
            original_values = [row[column] for row in data_out]                                                                                                                                                                                             
            # Transformation via classify_log_distribution (qui retourne (labels, y))                                                                                                                                                                   
            labels = log_distribution(original_values)             
                                                                                                                                                                                                                                                        
            # Mise à jour de chaque ligne avec le nouveau label                                                                                                                                                                                         
            for i, row in enumerate(data_out):                                                                                                                                                                                                              
                row[column] = labels[i]                                                                                                                                                                                                               
                                                                                                                                                                                                                                                        
        elif col_type == 'continuous':                                                                                                                                                                                                                  
            # On ne fait rien pour 'continuous'                                                                                                                                                                                                         
            pass                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                        
    return data_out

def feature_product(data, columns_to_multiply, new_column_name):
    """
    Calculates the product of values found in specific keys within a list of dictionaries.
    
    Args:
        data (list of dict): The dataset containing the edge attributes.
        columns_to_multiply (list of str): The keys whose values should be multiplied.
        new_column_name (str): The name of the new key to store the result in.
        
    Returns:
        list of dict: The updated dataset with the new product key.
    """
    for entry in data:
        product = 1.0
        valid_multiplication_performed = False
        
        for key in columns_to_multiply:
            # Check if the key exists in the current dictionary
            if key in entry:
                val = entry[key]
                
                # Ensure the value is a number (int or float) to avoid TypeError with strings
                if isinstance(val, (int, float)):
                    product *= val
                    valid_multiplication_performed = True
                else:
                    # If we encounter a string (like a branch name), we skip it 
                    # and warn the user, or you could treat it as 1.0
                    print(f"Warning: Value for '{key}' is not a number ({val}). Skipping in product.")
            else:
                # If a key is missing, we treat its contribution as 1 (neutral element)
                # so it doesn't zero out the entire product.
                pass
        
        # If no numeric keys were found at all, set product to 0 or None
        if not valid_multiplication_performed:
            entry[new_column_name] = 0.0
        else:
            entry[new_column_name] = product
            
    return data

def feature_substract(data, columns_to_substract, new_column_name):
    """
    Calculates the difference (first key minus second key) of values 
    found in specific keys within a list of dictionaries.
    
    Args:
        data (list of dict): The dataset containing the edge attributes.
        columns_to_substract (list of str): A list where index 0 is the minuend 
                                           and index 1 is the subtrahend.
        new_column_name (str): The name of the new key to store the result in.
        
    Returns:
        list of dict: The updated dataset with the new subtraction key.
    """
    # Safety check: Ensure we have at least two keys to perform subtraction
    if len(columns_to_substract) < 2:
        raise ValueError("Subtraction requires at least two keys in columns_to_substract.")

    key1 = columns_to_substract[0]
    key2 = columns_to_substract[1]

    for entry in data:
        # Retrieve values, defaulting to None if key is missing
        val1 = entry.get(key1)
        val2 = entry.get(key2)
        
        # Check if both values exist and are numeric
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            entry[new_column_name] = abs(val1 - val2)
        else:
            # If a value is missing or is a string (like a branch name), 
            # we set the result to 0.0 and log a warning.
            entry[new_column_name] = 0.0
            
            # Detailed error reporting for debugging your test suite
            if val1 is None or val2 is None:
                print(f"Warning: Missing key in entry {entry.get('id', 'unknown')}. "
                      f"({key1}: {val1}, {key2}: {val2})")
            elif not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                print(f"Warning: Non-numeric value in entry {entry.get('id', 'unknown')}. "
                      f"({key1}: {type(val1).__name__}, {key2}: {type(val2).__name__})")
            
    return data

def composite_feature_structural_impact(data, new_column_name):
    """
    Merge structural features in 1 (first key minus second key) of values 
    found in specific keys within a list of dictionaries.
    
    Args:
        data (list of dict): The dataset containing the edge attributes.
        new_column_name (str): The name of the new key to store the result in.
        
    Returns:
        list of dict: The updated dataset with the new subtraction key.
    """

    columns_to_combine = ['is_hub_edge','degree_assymetry_classes']

    # Safety check: Ensure we have at least two keys to perform subtraction
    if len(columns_to_combine) < 2:
        raise ValueError("Subtraction requires at least two keys in columns_to_combine.")

    key1 = columns_to_combine[0]
    key2 = columns_to_combine[1]

    for entry in data:
        # Retrieve values, defaulting to None if key is missing
        val1 = entry.get(key1)
        val2 = entry.get(key2)
        
        # Check if both values exist and are numeric
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Isolated – 0 0 or 1 0
            # Asymetric – 1 1
            # Dense cluster – 0 1
            if val1 == 1 and val2 == 1:
                entry[new_column_name] = "Asymetric"
            if val1 == 0 and val2 == 1:
                entry[new_column_name] = "Dense cluster"
            else:
                entry[new_column_name] = "Isolated"

        else:
            # If a value is missing or is a string (like a branch name), 
            # we set the result to 0.0 and log a warning.
            entry[new_column_name] = None
            
            # Detailed error reporting for debugging your test suite
            if val1 is None or val2 is None:
                print(f"Warning: Missing key in entry {entry.get('id', 'unknown')}. "
                      f"({key1}: {val1}, {key2}: {val2})")
            elif not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                print(f"Warning: Non-numeric value in entry {entry.get('id', 'unknown')}. "
                      f"({key1}: {type(val1).__name__}, {key2}: {type(val2).__name__})")
            
    return data

def compute_semantic_complexity(data):
    semantic_complexity_headers = ['subject_biolink_depth','object_biolink_depth','predicate_biolink_depth']


    data_semantic_complexity = feature_product(data, semantic_complexity_headers, 'semantic_complexity')
    data_semantic_complexity = feature_degree_classification(data_semantic_complexity, 'semantic_complexity', 'semantic_complexity_classes')

    return data_semantic_complexity

def feature_hub_classification(data, columns_to_consider, new_column_name):
    """
    Identifies if an edge contains at least one 'Extreme High' (Class 2) node.
    
    Args:
        data (list of dict): The dataset.
        columns_to_consider (list of str): [subject_degree_key, object_degree_key]
        new_column_name (str): The key to store the boolean/binary result.
        
    Returns:
        list of dict: Updated dataset.
    """
    if len(columns_to_consider) < 2:
        raise ValueError("You must provide exactly two keys: [subject_key, object_key]")

    s_key = columns_to_consider[0]
    o_key = columns_to_consider[1]

    # Collect all values to calculate global percentiles ---
    s_degrees = []
    o_degrees = []
    
    for entry in data:
        # We use .get(key, 0) to handle missing degrees safely
        s_degrees.append(entry.get(s_key, 0))
        o_degrees.append(entry.get(o_key, 0))

    # Calculate the distribution classes for the whole population ---
    s_classes, _ = classify_distribution(np.array(s_degrees))
    o_classes, _ = classify_distribution(np.array(o_degrees))

    # Map back to the dictionaries with the OR logic ---
    for i, entry in enumerate(data):
        # Check if Subject Class is 2 OR Object Class is 2
        if s_classes[i] == 2 or o_classes[i] == 2:
            entry[new_column_name] = 1
        else:
            entry[new_column_name] = 0
            
    return data

def feature_biomedical_area(data, columns_to_consider, new_column_name):
    """
    Implements Canonical Ordering for biological branches to create 
    a unified 'Relationship Type' feature.
    
    Args:
        data (list of dict): The dataset containing edge attributes.
        columns_to_consider (list of str): [subject_branch_key, object_branch_key]
        new_column_name (str): The name of the new key to store the pair string.
        
    Returns:
        list of dict: The updated dataset with the canonical pair key.
    """
    if len(columns_to_consider) < 2:
        raise ValueError("You must provide exactly two keys for the pair.")

    s_key = columns_to_consider[0]
    o_key = columns_to_consider[1]

    for entry in data:
        # 1. Extract values, using 'Unknown' if a key is missing or None
        # We cast to str() to ensure we can sort them even if types are mixed
        val_s = str(entry.get(s_key, 'Unknown'))
        val_o = str(entry.get(o_key, 'Unknown'))
        
        # Handle cases where the value might be explicitly None in the dict
        if val_s == 'None': val_s = 'Unknown'
        if val_o == 'None': val_o = 'Unknown'

        # 2. Canonical Ordering: Sort the two strings alphabetically
        # This ensures ('Gene', 'Disease') and ('Disease', 'Gene') are identical
        ordered_pair = sorted([val_s, val_o])

        # 3. Create a single string representation for stratification
        # We use a separator like ' -> ' to make it human-readable in your test reports
        entry[new_column_name] = f"{ordered_pair[0]} -> {ordered_pair[1]}"

    return data

def feature_degree_classification(data, column_to_read, new_column_name):
    """
    Classifies degree asymmetry into 3 strata (Low, Medium, High) 
    using quantiles to ensure balanced distribution for testing.
    
    Args:
        data (list of dict): The KG edge dataset.
        column_to_read (str): The key containing the asymmetry values.
        new_column_name (str): The name of the new classification key.
        
    Returns:
        list of dict: The updated dataset.
    """
    if not data:
        return []

    # 1. Extract all values for the calculation
    # We use a list comprehension to grab the values, defaulting to 0 if missing
    values = np.array([float(entry.get(column_to_read, 0)) for entry in data])

    if values.size == 0:
        return data

    # 2. Calculate Quantile Boundaries
    # We use the 25th and 75th percentiles to create 3 equal-sized groups (Tertiaries)
    # This prevents "strata explosion" while capturing the distribution shape.
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)

    # 3. Apply classification to each entry
    for entry in data:
        val = float(entry.get(column_to_read, 0))
        
        if val <= p25:
            label = 0      # Bottom 25% (Near zero/Symmetric)
        elif val <= p75:
            label = 1   # Middle 50% (Moderate asymmetry)
        else:
            label = 2     # Top 25% (Extreme asymmetry/Outliers)
            
        entry[new_column_name] = label

    return data

def feature_assymetry_classification(data, column_to_read, new_column_name):
    """
    Classifies degree asymmetry into 3 strata (Low, Medium, High) 
    using quantiles to ensure balanced distribution for testing.
    
    Args:
        data (list of dict): The KG edge dataset.
        column_to_read (str): The key containing the asymmetry values.
        new_column_name (str): The name of the new classification key.
        
    Returns:
        list of dict: The updated dataset.
    """
    if not data:
        return []

    # 1. Extract all values for the calculation
    # We use a list comprehension to grab the values, defaulting to 0 if missing
    values = np.array([float(entry.get(column_to_read, 0)) for entry in data])

    if values.size == 0:
        return data

    # 2. Calculate Quantile Boundaries
    # We use the 25th and 75th percentiles to create 3 equal-sized groups (Tertiaries)
    # This prevents "strata explosion" while capturing the distribution shape.
    p75 = np.percentile(values, 75)

    # 3. Apply classification to each entry
    for entry in data:
        val = float(entry.get(column_to_read, 0))
        
        if val <= p75:
            label = 0   # (Moderate asymmetry)
        else:
            label = 1     # Top 25% (Extreme asymmetry/Outliers)
            
        entry[new_column_name] = label

    return data

def compute_structural_impact(data):

    updated_data = feature_substract(data, ['subject_degree','object_degree'], 'degree_assymetry')
    updated_data = feature_hub_classification(updated_data, ['subject_degree', 'object_degree'], 'is_hub_edge')
    updated_data = feature_assymetry_classification(updated_data, 'degree_assymetry', 'degree_assymetry_classes')
    updated_data = composite_feature_structural_impact(updated_data, 'structural_composite')
    
    return updated_data


def main(data,columns_type):

    ## reequilibrate distributions based on data type
    updated_data = transform_metrics(data, columns_type)

    ## dimensionality reduction:
    ### semantic complexity
    updated_data = compute_semantic_complexity(updated_data) # one-sided from normal around 0
    
    ### structural impact:
    updated_data = compute_structural_impact(updated_data)
    
    ### vocabulary type:
    biomedical_area_headers = ['subject_biolink_branch','object_biolink_branch']
    updated_data = feature_biomedical_area(updated_data, biomedical_area_headers, 'biomedical_area_pair')


    return updated_data

if __name__ == "__main__":
    data = []
    columns_type = {'id':'ignore',
                    'subject':'ignore',
                    'subject_degree': 'powerlaw',
                    'subject_biolink_branch':'discrete',
                    'subject_biolink_depth':'discrete',
                    'object_degree': 'powerlaw',
                    'object_biolink_branch':'discrete',
                    'object_biolink_depth':'discrete',
                    'predicate_biolink_branch':'discrete',
                    'predicate_biolink_depth':'discrete',
                    'publications_number':'powerlaw'
                    }
    main(data, columns_type)