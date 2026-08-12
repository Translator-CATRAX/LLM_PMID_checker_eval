import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import sys
import argparse
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import copy

import data_preparation

def save_trace(trace, filename="data/inter-rater_Gibbs_trace.nc"):
    """
    Saves the InferenceData object to a NetCDF file.
    """
    print(f"Saving trace to {filename}...")
    az.to_netcdf(trace, filename)
    print("Save complete.")

# --- LOADING THE TRACE ---
def load_trace(filename="inter-rater_Gibbs_trace.nc"):
    """
    Loads an InferenceData object from a NetCDF file.
    """
    print(f"Loading trace from {filename}...")
    trace = az.from_netcdf(filename)
    print("Load complete.")
    return trace

def run_gibbs_sampler_irr_optimized(obs_item, obs_rater, obs_rating):
    """
    Optimized Model: Marginalizes out the discrete Z to allow 
    pure NUTS sampling. This eliminates the Metropolis bottleneck.
    """
    n_items = len(np.unique(obs_item)) 
    n_raters = int(obs_rater.max())

    # Rater profiles (The "Anchor" Prior)
    alpha_base = np.stack([
        np.array([0.05, 0.1, 0.2, 0.4, 0.6, 1]), # False State
        np.array([1, 0.6, 0.4, 0.2, 0.1, 0.05])  # True State
    ])
    alpha_prior = np.repeat(alpha_base[np.newaxis, :, :], n_raters, axis=0)

    coords = {
        "items": np.arange(n_items),
        "raters": np.arange(n_raters),
        "truth_else": ["False_State", "True_State"],
        "categories": ["True-High","True-Medium","True-Low", "False-Low", "False-Medium", "False-High"] 
    }
    # Note: Ensure categories match your alpha_base shape exactly.

    with pm.Model(coords=coords) as optimized_model:
        # --- Priors ---
        # Global prevalence (pi)
        # pi = pm.Beta("pi", alpha=2, beta=2)

        # Use this:
        logit_pi = pm.Normal("logit_pi", mu=2, sigma=1) # A positive mu pushes pi towards 1
        pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi))


        # Rater Confusion Matrix
        theta = pm.Dirichlet("theta", a=alpha_prior, dims=("raters", "truth_else", "categories"))

        # --- Marginalized Likelihood ---
        # Instead of sampling Z, we calculate the weighted probability 
        # of each category based on pi.
        
        # theta_for_obs: shape (n_observations, 2, 6)
        theta_for_obs = theta[obs_rater - 1] 
        
        # Extract the two possible outcomes for each observation:
        # Prob if item is False (Z=0)
        p_false = theta_for_obs[:, 0, :] 
        # Prob if item is True (Z=1)
        p_true = theta_for_obs[:, 1, :]

        # The Marginal Probability: P(obs) = P(obs|Z=0)(1-pi) + P(obs|Z=1)(pi)
        # This is the "Magic" step that removes the need for Metropolis.
        p_marginal = (1 - pi) * p_false + pi * p_true

        # Likelihood
        pm.Categorical("likelihood", p=p_marginal, observed=obs_rating)

        # --- Sampling ---
        # Now we only use NUTS! No Metropolis needed.
        trace = pm.sample(
            draws=2000, 
            tune=1000, 
            chains=4, 
            target_accept=0.95,
            random_seed=77
        )

    return trace

def evaluate_rater_performance(results):
    """
    Evaluates rater performance for a 2x6 confusion matrix.
    Truth: [True, False] (2 rows)
    Response: [T-High, T-Med, T-Low, F-Low, F-Med, F-High] (6 columns)
    """
    rater_matrices = results['rater_matrices']
    performance_report = {}

    for r_id, matrix in rater_matrices.items():
        # matrix shape is (2, 6)
        if matrix.shape != (2, 6):
            raise ValueError(f"Rater {r_id} matrix must be (2, 6). Found {matrix.shape}")

        # --- 1. ACCURACY CALCULATION ---
        # Accuracy = P(Response is 'True-type' | Truth is True) 
        #           + P(Response is 'False-type' | Truth is False)
        
        # Sum probabilities in Row 0 (Truth=True) for columns 0, 1, 2
        acc_true_state = np.sum(matrix[1, 0:3])
        
        # Sum probabilities in Row 1 (Truth=False) for columns 3, 4, 5
        acc_false_state = np.sum(matrix[0, 3:6])
        
        # The overall accuracy is the average of these two state-specific accuracies
        total_accuracy = (acc_true_state + acc_false_state) / 2

        # --- 2. ERROR RATE ---
        # Error rate is the probability that a 'True' truth gets a 'False' response,
        # or a 'False' truth gets a 'True' response.
        error_true_state = np.sum(matrix[1, 3:6])
        error_false_state = np.sum(matrix[0, 0:3])
        total_error_rate = (error_true_state + error_false_state) / 2

        # --- 3. ENTROPY (Decisiveness) ---
        # We still use the flattened matrix to see how 'spread out' the rater is.
        flattened_probs = matrix.flatten()
        entropy = -np.sum(flattened_probs * np.log(flattened_probs + 1e-12))

        performance_report[r_id] = {
            "accuracy": total_accuracy,
            "error_rate": total_error_rate,
            "entropy": entropy,
            "confusion_matrix": matrix
        }

    return performance_report

def extract_posterior_results(trace, obs_item, obs_rater, obs_rating, hdi_prob=0.97):
    """
    Extracts posterior means and HDI bounds for item probabilities 
    and rater confusion matrices from a PyMC trace where Z is marginalized.
    """
    
    # --- STEP 0: Robust Trace Loading ---
    if isinstance(trace, tuple):
        trace = trace[0]

    if not hasattr(trace, "posterior"):
        raise TypeError(f"The provided trace is of type {type(trace)}, not InferenceData.")

    # Access the posterior group
    post_group = trace.posterior

    # --- STEP 1: Setup Dimensions & Bounds Checking ---
    # FIX: Use .sizes instead of .dims to allow string-based lookup
    # post_group["theta"].dims is a tuple, but .sizes is a mapping!
    n_raters_in_model = post_group["theta"].sizes["raters"]
    n_categories_in_model = post_group["theta"].sizes["categories"]

    # Check for Rater ID mismatch (Input must be 1-indexed to match your model)
    max_r_input = int(obs_rater.max())
    if max_r_input > n_raters_in_model:
        raise ValueError(f"Rater Mismatch: Input contains rater {max_r_input}, "
                         f"but model only knows up to rater {n_raters_in_model}.")

    # Check for Rating Value mismatch (Input must be 0-indexed)
    max_rating_input = int(obs_rating.max())
    if max_rating_input >= n_categories_in_model:
        raise ValueError(f"Rating Mismatch: Input contains rating {max_rating_input}, "
                         f"but model only has {n_categories_in_model} categories.")

    # Extract raw values for high-speed math
    post_pi_values = post_group["pi"].values  
    post_theta_values = post_group["theta"].values  
    
    n_chains = post_pi_values.shape[0]
    n_draws = post_pi_values.shape[1]
    
    unique_items = np.unique(obs_item)
    item_to_idx = {item_id: i for i, item_id in enumerate(unique_items)}
    
    obs_df = pd.DataFrame({
        'item': obs_item,
        'rater': obs_rater,
        'rating': obs_rating
    })

    # --- STEP 2: Reconstruct P(Z=1 | data) ---
    prob_z1_samples = np.zeros((n_chains, n_draws, len(obs_item)))

    for item_id in unique_items:
        item_idx = item_to_idx[item_id]
        item_obs = obs_df[obs_df['item'] == item_id]
        raters_for_item = item_obs['rater'].values - 1 # Convert to 0-indexed
        ratings_for_item = item_obs['rating'].values
        
        likelihood_z1 = np.ones((n_chains, n_draws))
        likelihood_z0 = np.ones((n_chains, n_draws))

        for i, r_idx in enumerate(raters_for_item):
            rating = ratings_for_item[i]
            # Using the raw numpy values for speed
            p_true = post_theta_values[:, :, r_idx, 1, rating]
            p_false = post_theta_values[:, :, r_idx, 0, rating]
            
            likelihood_z1 *= p_true
            likelihood_z0 *= p_false

        # Bayes Rule
        numerator = likelihood_z1 * post_pi_values
        denominator = (likelihood_z1 * post_pi_values) + (likelihood_z0 * (1 - post_pi_values))
        
        prob_for_this_item = numerator / denominator
        
        matching_indices = np.where(obs_item == item_id)[0]
        prob_z1_samples[:, :, matching_indices] = prob_for_this_item[:, :, np.newaxis]

    # --- STEP 3: Calculate Statistics ---
    item_probs_in_obs = prob_z1_samples.mean(axis=(0, 1))
    item_probs_hdi_low = np.quantile(prob_z1_samples, (1 - hdi_prob) / 2, axis=(0, 1))
    item_probs_hdi_high = np.quantile(prob_z1_samples, 1 - (1 - hdi_prob) / 2, axis=(0, 1))

    # Process Theta
    post_theta_mean = post_theta_values.mean(axis=(0, 1))
    post_theta_low = np.quantile(post_theta_values, (1 - hdi_prob) / 2, axis=(0, 1))
    post_theta_high = np.quantile(post_theta_values, 1 - (1 - hdi_prob) / 2, axis=(0, 1))

    rater_matrices = {}
    rater_matrices_low = {}
    rater_matrices_high = {}
    for r_id in range(n_raters_in_model):
        rater_matrices[r_id] = post_theta_mean[r_id]
        rater_matrices_low[r_id] = post_theta_low[r_id]
        rater_matrices_high[r_id] = post_theta_high[r_id]

    return {
        "item_probs_in_obs": item_probs_in_obs,
        "item_probs_hdi_low": item_probs_hdi_low,
        "item_probs_hdi_high": item_probs_hdi_high,
        "rater_matrices": rater_matrices,
        "rater_matrices_hdi_low": rater_matrices_low,
        "rater_matrices_hdi_high": rater_matrices_high
    }

def extract_hdi_tensor(hdi_data, var_name, mode='lower'):
    """Helper to extract theta bounds (Tensor: raters, truth, cat)"""
    if mode == 'lower':
        return hdi_data[var_name].sel(hdi='lower').values
    return hdi_data[var_name].sel(hdi='higher').values

def extract_hdi(hdi_data, var_name, index_array, mode='lower'):
    """Helper to extract Z bounds and map them to observations"""
    if mode == 'lower':
        # Get the lower bound for all items, then index by obs_item
        return extract_hdi_tensor(hdi_data, var_name, 'lower')[index_array]
    return extract_hdi_tensor(hdi_data, var_name, 'higher')[index_array]

def plot_rater_confusion(rater_matrices, rater_id, prefix, figure_extension):
    plt.figure(figsize=(6, 4))
    sns.heatmap(rater_matrices[rater_id], annot=True, cmap="Blues")
    plt.title(f"Confusion Matrix for Rater {rater_id}")
    plt.xlabel("Rater Response")
    plt.ylabel("True State")
    plt.savefig(f"data/{prefix}_confmat_{rater_id}{figure_extension}")
    plt.close()

def generate_mock_data(n_items=1320, n_raters=18, n_strata=400, n_obs=1600):
    """Generates random data and returns a joined DataFrame."""
    print("Generating mock data...")
    
    obs_item = np.random.randint(0, n_items, n_obs)
    obs_rater = np.random.randint(0, n_raters, n_obs)
    obs_rating = np.random.randint(0, 6, n_obs)
    item_strata = np.random.randint(0, n_strata, n_items)

    # Create the main observations dataframe
    df_obs = pd.DataFrame({
        'index': np.arange(n_obs), # Creating an index to match the join logic
        'item_id': obs_item,
        'rater_id': obs_rater,
        'rating': obs_rating
    })

    # Create a lookup table for strata (to simulate joining)
    df_strata = pd.DataFrame({
        'item_id': np.arange(n_items),
        'strata': item_strata
    })

    # Join them to provide a single unified dataframe as the output
    return pd.merge(df_obs, df_strata, on='item_id', how='left')

def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def rename_columns_from_config(df, column_mapping):
    """
    Renames columns in the DataFrame based on the config mapping.
    Mapping format expected: {'Standard_Name': 'Original_CSV_Name'}
    """
    # Create a dictionary for pandas .rename(): {'Original_CSV_Name': 'Standard_Name'}
    # We swap the key and value from your YAML structure
    rename_dict = {v: k for k, v in column_mapping.items()}
    
    # Identify which columns we are actually attempting to rename
    columns_to_rename = [v for v in rename_dict.keys() if v in df.columns]
    
    if not columns_to_rename:
        print("No matching columns found in CSV to rename.")
        return df

    print(f"Renaming columns: {rename_dict}")
    return df.rename(columns=rename_dict)

def load_csv_files(file_paths,join_on: str = 'index'):
    """Loads multiple CSV files and joins them by the 'index' column."""
    print(f"Loading and joining {len(file_paths)} CSV files...")
    
    combined_df = None

    for path in file_paths:
        try:
            # Changed from read_parquet to read_csv
            temp_df = pd.read_csv(path)
            
            if combined_df is None:
                combined_df = temp_df
            else:
                # We use outer join so no data is lost if indices don't perfectly overlap
                combined_df = pd.merge(combined_df, temp_df, on=join_on, how='outer')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            sys.exit(1)

    return combined_df

def run_model_diagnostics(trace, prefix, figure_extension, var_names=None):
    """
    Gener_ates a comprehensive diagnostic dashboard to verify model convergence.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        The trace object from pm.sample()
    var_names : list, optional
        Specific variables to plot. If None, plots all global parameters.
    """
    if var_names is None:
        # We focus on 'pi' for trace/rank because plotting 100s of theta 
        # parameters in a single trace plot creates an unreadable mess.
        var_names = ["pi"]

    # Create a figure with a grid layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # --- 1. TRACE PLOT (Stationarity Check) ---
    # Shows the value of parameters over the sampling iterations.
    az.plot_trace(trace, var_names=var_names, axes=axes[0, 0])
    axes[0, 0].set_title("1. Trace Plots (Stationarity)", fontsize=14, fontweight='bold')

    # --- 2. RANK PLOT (Mixing Check) ---
    # A modern way to see if chains are exploring the same space.
    az.plot_rank(trace, var_names=var_names, ax=axes[0, 1])
    axes[0, 1].set_title("2. Rank Plots (Chain Mixing)", fontsize=14, fontweight='bold')

    # --- 3. AUTOCORRELATION PLOT (Efficiency Check) ---
    # Shows how much a sample depends on the previous sample.
    az.plot_autocorr(trace, var_names=var_names, ax=axes[1, 0])
    axes[1, 0].set_title("3. Autocorrelation (Independence)", fontsize=14, fontweight='bold')

    # --- 4. FOREST PLOT (Posterior Summary) ---
    # Shows the mean and HDI for all parameters in one view.
    az.plot_forest(trace, var_names=var_names, ax=axes[1, 1])
    axes[1, 1].set_title("4. Forest Plot (Posterior Estimates)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"data/{prefix}_model_convergence{figure_extension}")
    plt.close()

    # Print the numerical summary for R-hat and ESS
    print("\n" + "="*50)
    print(f"{prefix} NUMERICAL CONVERGENCE SUMMARY")
    print("="*50)
    summary = az.summary(trace, var_names=var_names)
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])
    print("="*50)

def plot_model_diagnostics(trace, prefix, figure_extension, var_names=None):
    """
    Generates a comprehensive diagnostic dashboard to verify model convergence.
    """
    if var_names is None:
        # We focus on 'pi' for trace/rank because plotting 100s of theta 
        # parameters in a single trace plot creates an unreadable mess.
        var_names = ["pi"]

    # 1. Create a 3x2 grid (6 slots total)
    # We use the full 2D axes object so ArviZ can use its internal 2D indexing.
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # --- 1. TRACE PLOT (Stationarity Check) ---
    # IMPORTANT: Pass the FULL 2D axes object. 
    # ArviZ will automatically use axes[0, 0] and axes[0, 1] for the two plots.
    az.plot_trace(trace, var_names=var_names, axes=axes)
    
    # Since plot_trace occupies [0,0] and [0,1], we set titles on those specific indices
    axes[0, 0].set_title("1. Trace Plots (Stationary)", fontsize=14, fontweight='bold')
    axes[0, 1].set_title("", fontsize=14) # Hide the redundant title on the density side

    # --- 2. RANK PLOT (Mixing Check) ---
    # We manually point to the next available slot in the 2D grid
    az.plot_rank(trace, var_names=var_names, ax=axes[1, 0])
    axes[1, 0].set_title("2. Rank Plots (Chain Mixing)", fontsize=14, fontweight='bold')

    # --- 3. AUTOCORRELATION PLOT (Efficiency Check) ---
    az.plot_autocorr(trace, var_names=var_names, ax=axes[1, 1])
    axes[1, 1].set_title("3. Autocorrelation (Independence)", fontsize=14, fontweight='bold')

    # --- 4. FOREST PLOT (Posterior Summary) ---
    az.plot_forest(trace, var_names=var_names, ax=axes[2, 0])
    axes[2, 0].set_title("4. Forest Plot (Posterior Estimates)", fontsize=14, fontweight='bold')

    # --- 5. CLEANUP ---
    # We have one unused axis left at axes[2, 1]. Let's hide it so the plot looks clean.
    axes[2, 1].axis('off')

    plt.tight_layout()
    
    save_path = f"data/{prefix}_model_convergence{figure_extension}"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Diagnostic plot saved to: {save_path}")
    plt.close()

    # Print the numerical summary for R-hat and ESS
    print("\n" + "="*50)
    print(f"{prefix} NUMERICAL CONVERGENCE SUMMARY")
    print("="*50)
    summary = az.summary(trace, var_names=var_names)
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Gibbs sampler...")
    parser.add_argument('--files', nargs='+', help='List of path to CSV files')
    parser.add_argument('--config', default='config_gibbs_model.yml', help='Path to the YAML config file')

    args = parser.parse_args()


    if args.files:
        # OPTION 1: Load from CSV files
        df = load_csv_files(args.files)

        # 1. Load configuration
        config = load_config(args.config)
        column_mapping = config.get('columns', {})   

        # 2. MAP/RENAME columns using the config mapping
        # This transforms 'CSV_NAME' -> 'STANDARD_NAME'
        df = rename_columns_from_config(df, column_mapping)
        
        # 3. Final Verification: Check if our standard names exist after renaming
        standard_names = list(column_mapping.keys())
        missing = [name for name in standard_names if name not in df.columns]
        
        if missing:
            print(f"Warning: Standard columns {missing} are still missing from the DataFrame.")
    else:
        # OPTION 2: Use mock data (already uses standard names)
        # print("Generating Gibbs sampling from normal distributions")
        # df = generate_mock_data()
        df = data_preparation.prepare_dataset()

        data_preparation.plot_rater_pairing_graph(df,figure_extension = ".png")

        # 1. Load configuration
        config = load_config(args.config)
        column_mapping = config.get('columns', {})   


    # 2. Run Model for inter-rater correction on columns ratings
    for prefix in column_mapping["obs_rating"]:
        # 2. MAP/RENAME columns using the config mapping
        # This transforms 'CSV_NAME' -> 'STANDARD_NAME'
        column_mapping1 = copy.deepcopy(column_mapping)
        column_mapping1["obs_rating"]=prefix
        df1 = rename_columns_from_config(df, column_mapping1)
        standard_names = list(column_mapping1.keys())
        missing = [name for name in standard_names if name not in df.columns]
        if missing:
            print(f"Warning: Standard columns {missing} are still missing from the DataFrame.")

        # 3. Run model
        trace =  run_gibbs_sampler_irr_optimized(df1.obs_item, df1.obs_rater, df1.obs_rating)
        save_trace(trace, filename=f"data/{prefix}inter-rater_Gibbs_trace.nc")

        m_name = f"P(Z|M {prefix})"
        m_name_hdi_low = f"P(Z|M {prefix})  HDI low"
        m_name_hdi_high = f"P(Z|M {prefix})  HDI high"
        figure_diag_name = f"{prefix}_pi_"
        figure_diag_name2 = f"{prefix}_logit_pi_"

        # 4. Infer Z and compute results
        results =  extract_posterior_results(trace, df1.obs_item, df1.obs_rater,df1.obs_rating,0.97)
        df[m_name] = results["item_probs_in_obs"]
        df[m_name_hdi_low] = results["item_probs_hdi_low"]
        df[m_name_hdi_high] = results["item_probs_hdi_high"]


        rater_performance_report = evaluate_rater_performance(results)
        rater_performance_report = pd.DataFrame(evaluate_rater_performance(results)).T
        rater_performance_report.to_csv(f"data/{prefix}_rater_performance_report.csv")

        # 5. Plot
        for r_id, matrix in results["rater_matrices"].items():
            plot_rater_confusion(results["rater_matrices"], r_id,prefix,".png")

        
        plot_model_diagnostics(trace,figure_diag_name,".png", var_names=["pi"])
        plot_model_diagnostics(trace,figure_diag_name2,".png", var_names=["logit_pi"])

    df.to_csv("data/data_merged_ratings_corrected.csv", index=False) # save inferences



    # Analyses
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="P(Z|M triple rating)", hue="triple rating", split=True, inner="quart")


    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="P(Z|M triple rating)", hue="predicted", split=True, inner="quart")

    print("bob")
