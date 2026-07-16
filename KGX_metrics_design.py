import math
import statistics

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
            
    return results
