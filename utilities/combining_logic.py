"""
combining_logic.py

Standalone, importable module bundling the fusion logic that combines the three
models' outputs into a single final maliciousness probability:

  - Product of Experts and Averaging Logic: fuses XGBoost + Transformer outputs
    into a base malicious probability, blending log-pooling and linear-pooling
    based on how much the two models disagree.
  - CNN Aggregation Evidence Logic: buckets/ranks CNN file-level evidence and
    turns it into a bounded nudge applied on top of the base probability.

Usage (after uploading this file as a Kaggle Dataset and attaching it to your notebook):

    import sys
    sys.path.append("/kaggle/input/<your-dataset-slug>")
    from combining_logic import total_maliciousness

    base_P_mal, cnn_effect, final_P_mal, disagreement, scored_files = total_maliciousness(
        xgb_results, transformer_results,
        cnn_all_results, cnn_class_weights_data,
        xgb_per_class_fscores, transformer_per_class_fscores,
        hierarchy, max_limit_files, per_class_limit
    )

total_maliciousness() is the single entry point meant to be called from the
notebook; the other functions in this module are its building blocks, exposed
in case you want to call/inspect them individually (e.g. bucketing() or
score_calc() on their own).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Product of Experts and Averaging Logic
# (only for the xgboost and transformer models)
# ---------------------------------------------------------------------------

def aggregrate_to_binary_classes(inference_results, fscores):
    """
    Power-scales each class's probability by its per-class F1 reliability score,
    renormalizes, then collapses to binary [p_benign, p_mal]. The first entry in
    Probability_Distribution / fscores must correspond to benign/goodware.
    """
    # power scaling of prob with respective class f1 scores
    probs = inference_results['Probability_Distribution']
    power_scaled = []
    for i in range(0, len(fscores)):
        power_scaled.append(pow(probs[i], fscores[i]))

    # re-normalize the terms
    s = sum(power_scaled)
    for i in range(0, len(power_scaled)):
        power_scaled[i] = power_scaled[i] / s

    # first entry corresponds to benign/goodware, even if others are there, 1-benign is malicious
    p_benign = power_scaled[0]
    p_mal = 1 - p_benign
    return [p_benign, p_mal]


def base_maliciousness(sc1, sc2):
    """
    Combines two binary [p_benign, p_mal] scores (only for the xgboost and
    transformer models) via a Product-of-Experts (log) pool, and separately via
    a linear pool (plain average). Also returns the disagreement (total variation
    distance) between the two models' malicious probabilities, which the caller
    uses to blend log-pool and linear-pool -- log-pool sharpens when models agree,
    linear-pool prevents one confident model from steamrolling the other when
    they conflict.
    """
    # disagreement score and linear pool
    disagree = abs(sc1[1] - sc2[1])
    p_linear_mal = (sc1[1] + sc2[1]) / 2

    unnorm_benign = sc1[0] * sc2[0]  # multiply the benign probs
    unnorm_mal = sc1[1] * sc2[1]  # multiply the malicious probs
    norm_const = unnorm_benign + unnorm_mal  # find sum

    # normalize wrt sum and return the log malicious prob, linear prob and disagreement
    return unnorm_mal / norm_const, disagree, p_linear_mal


# ---------------------------------------------------------------------------
# CNN Aggregation Evidence Logic
# ---------------------------------------------------------------------------

def extract_important_files(files_info, max_total, per_class_limit):
    """
    Greedily walks classes in severity order (most severe first) and takes up to
    `per_class_limit` most-confident files from each, until `max_total` files are
    selected overall. files_info values are assumed pre-sorted by confidence (desc).
    """
    selected_files = []
    for attack, files in files_info.items():
        count = 0
        for confidence, file_index in files:
            if len(selected_files) == max_total:
                break
            selected_files.append((file_index, confidence, attack))
            count += 1
            if count == per_class_limit:
                break
        if len(selected_files) == max_total:
            break
    return selected_files


def bucketing(hierarchy, cnn_all_results, max_limit_files, per_class_limit):
    """
    Groups CNN inference results by predicted class, sorts each group by confidence,
    then hands off to extract_important_files() to pick a diverse, high-confidence
    subset (at most `per_class_limit` per class, `max_limit_files` total).
    """
    # Walk classes in explicit severity order (by hierarchy VALUE, not dict insertion
    # order) so this stays correct even if `hierarchy` is redefined/reordered later.
    ordered_classes = sorted(hierarchy, key=hierarchy.get)
    files_info = {attack: [] for attack in ordered_classes}

    for result in cnn_all_results:
        predicted_class = result['Predicted_Class'][0]
        file_index = result['File_index']
        confidence = result['Calibrated_Confidence'][0]

        if predicted_class not in files_info:
            print(f"Warning: predicted class '{predicted_class}' not found in hierarchy, skipping file {file_index}.")
            continue

        files_info[predicted_class].append((confidence, file_index))

    # Sort each class's files by confidence, most confident first
    for attack in files_info:
        files_info[attack].sort(key=lambda item: item[0], reverse=True)

    influencial_files = extract_important_files(files_info, max_limit_files, per_class_limit)
    return influencial_files


def sample_size_factor(n_files, k0=3):
    """
    n_files: number of files actually contributing evidence (len(scored_files)).
    k0: smoothing constant -- roughly 'how many files before you trust this fully'.
    Classic Bayesian-style discounting: grows toward 1 as more evidence corroborates,
    stays low when there's only a file or two.
    """
    return n_files / (n_files + k0)


def score_calc(influencial_files, class_weights, beta=1.0):
    """
    influencial_files: list of (file_index, confidence, predicted_class) from bucketing().
    class_weights: loaded cnn_class_weights.json content (dict with a "w_normalized" key).
    beta: saturation-speed knob for tanh -- tune against real validation data before
          trusting this in production; 1.0 is a placeholder, not a calibrated value.

    Returns:
      result       -- the bounded [0, 1) CNN evidence signal, ready to plug into
                       final_P_mal = P_mal + (1 - P_mal) * result
      scored_files -- (file_index, predicted_class, confidence, raw_score) sorted
                       by raw_score desc, kept around for later reporting
                       (e.g. CNN_Summary in final_summary).
    """
    w_normalized = class_weights["w_normalized"]

    # raw_score_j = calibrated_confidence_j * w_normalized(predicted_class_j)
    # scored_files entries keep confidence alongside raw_score so downstream
    # reporting (e.g. final_summary's CNN_Summary) has the actual confidence
    # available, not just the composite confidence*weight score.
    scored_files = []
    for file_index, confidence, predicted_class in influencial_files:
        weight = w_normalized[predicted_class]
        raw_score = confidence * weight
        scored_files.append((file_index, predicted_class, confidence, raw_score))

    # Rank by raw_score itself (NOT by hierarchy/severity) -- strongest evidence,
    # regardless of which class it came from, gets rank 1.
    scored_files.sort(key=lambda item: item[3], reverse=True)

    # No renormalization: aggregate should grow/shrink with genuine evidence strength.
    # tanh alone provides the saturation ("limit after some extent").
    aggregate = sum(
        raw_score / np.sqrt(rank + 1)
        for rank, (_, _, _, raw_score) in enumerate(scored_files)
    )
    result = sample_size_factor(len(scored_files)) * np.tanh(beta * aggregate)

    return result, scored_files


# ---------------------------------------------------------------------------
# Entry point: combines everything above into the final malicious probability
# ---------------------------------------------------------------------------

def total_maliciousness(xgb_results, transformer_results,
                         cnn_all_results, cnn_class_weights_data,
                         xgb_per_class_fscores, transformer_per_class_fscores,
                         hierarchy, max_limit_files, per_class_limit):

    s1 = aggregrate_to_binary_classes(xgb_results, xgb_per_class_fscores)
    s2 = aggregrate_to_binary_classes(transformer_results, transformer_per_class_fscores)

    P_mal_log_pool, disagreement, P_mal_linear_pool = base_maliciousness(s1, s2)
    P_mal = (P_mal_log_pool) * (1 - disagreement) + (P_mal_linear_pool) * (disagreement)

    selected_files = bucketing(hierarchy, cnn_all_results, max_limit_files, per_class_limit)
    cnn_effect, scored_files = score_calc(selected_files, cnn_class_weights_data, beta=1.0)

    final_P_mal = P_mal + (1 - P_mal) * cnn_effect

    return P_mal, cnn_effect, final_P_mal, disagreement, scored_files
