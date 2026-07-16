import math
import statistics
import random

def KGX_edge_sampling(data, sample_size=20):
    """
    Échantillonne les données en regroupant par toutes les propriétés 
    sauf l'identifiant unique 'id'.
    """
    if not data:
        return []

    groups = {}
    # On définit les clés de regroupement (toutes sauf 'id', 'subject', 'subject_name', 'object', 'object_name')

    group_keys = [k for k in data[0].keys() if k != 'id' and k != 'subject' and k != 'subject_name' and k != 'object' and k != 'object_name' and k!='subject_biolink_category' and k!='object_biolink_category' and k!='predicate']

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

