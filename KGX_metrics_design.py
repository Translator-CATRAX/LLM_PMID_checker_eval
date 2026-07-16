import math
import statistics
import random
import warnings
from tqdm import tqdm



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

def classify_log_distribution(x):
    if not x:
        return []

    # Transformation y = log(x)
    # Note: On utilise math.log (logarithme naturel). 
    # On suppose x > 0 pour éviter les erreurs de domaine.
    y = [math.log(val) for val in x]

    if len(y) < 2:
        # L'écart-type nécessite au moins deux points de données
        return ["Medium"] * len(x)

    mu_y = statistics.mean(y)
    sigma_y = statistics.stdev(y)

    results = []
    for val_y in y:
        if val_y < (mu_y - sigma_y):
            results.append("Low")
        elif (mu_y - sigma_y) <= val_y < mu_y:
            results.append("Medium")
        elif mu_y <= val_y < (mu_y + sigma_y):
            results.append("High")
        else:  # val_y >= (mu_y + sigma_y)
            results.append("Outliers")
            
    return results,y

def build_design_from_metrics(data, columns_type):                                                                                                                                                                                                      
    """                                                                                                                                                                                                                                                 
    Transforme les données en fonction des types de colonnes spécifiés.                                                                                                                                                                                 
    """                                                                                                                                                                                                                                                 
    if not data:                                                                                                                                                                                                                                        
        return []                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                        
    # 1. Vérification de la cohérence des clés (tous les dictionnaires doivent avoir les primes clés)                                                                                                                                                   
    base_keys = set(data[0].keys())                                                                                                                                                                                                                     
    for i, row in enumerate(data):                                                                                                                                                                                                                      
        if set(row.keys()) != base_keys:                                                                                                                                                                                                                
            raise ValueError(f"Erreur de structure : le dictionnaire à l'index {i} n'a pas les mêmes clés que le premier.")                                                                                                                             

    # On travaille sur une copie pour ne pas modifier l'original si on veut rester pure (optionnel)                                                                                                                                                     
    # Mais ici, on va modifier les valeurs des colonnes 'powerlaw' directement.                                                                                                                                                                         
                                                                                                                                                                                                                                                    
    for column, col_type in tqdm(columns_type.items(), total=len(columns_type.keys()), desc="Data transformation"):                                                                                                                                                                                                       
        if column not in base_keys:                                                                                                                                                                                                                     
            continue                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                        
        if col_type == 'discrete':                                                                                                                                                                                                                      
            # Vérification si les valeurs sont "considérables" (seuil arbitraire de 100)                                                                                                                                                 
            values = set([row[column] for row in data])                                                                                                                                                                                                     
            if values and len(values) > 100:                                                                                                                                                                                                            
                warnings.warn(f"Attention : la colonne '{column}' est marquée comme 'discrete' mais contient un ensemble de valeurs élevées ({len(values)}).")                                                                                                     
                                                                                                                                                                                                                                                        
        elif col_type == 'powerlaw':                                                                                                                                                                                                                    
            # Extraction des valeurs actuelles pour la transformation                                                                                                                                                                                   
            original_values = [row[column] for row in data]                                                                                                                                                                                             
            # Transformation via classify_log_distribution (qui retourne (labels, y))                                                                                                                                                                   
            labels, _ = classify_log_distribution(original_values)                                                                                                                                                                                      
                                                                                                                                                                                                                                                        
            # Mise à jour de chaque ligne avec le nouveau label                                                                                                                                                                                         
            for i, row in enumerate(data):                                                                                                                                                                                                              
                row[column] = labels[i]                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                        
        elif col_type == 'continuous':                                                                                                                                                                                                                  
            # On ne fait rien pour 'continuous'                                                                                                                                                                                                         
            pass                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                        
    return row

def main(data,columns_type):
    data_transformed = build_design_from_metrics(data, columns_type)
    sampled_data = KGX_edge_sampling(data,columns_type, sample_size=20)
    print('bob')

if __name__ == "__main__":
    data = []
    columns_type = {'id':'ignore',
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
    main(data, columns_type)