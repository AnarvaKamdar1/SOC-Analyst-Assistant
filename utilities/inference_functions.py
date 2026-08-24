"""
inference_functions.py

Standalone, importable module bundling the three calibrated inference pipelines
(XGBoost, Transformer, CNN) used in the malware-detection multi-model system.

Usage (after uploading this file as a Kaggle Dataset and attaching it to your notebook):

    import sys
    sys.path.append("/kaggle/input/<your-dataset-slug>")
    from inference_functions import (
        xgboost_calibrated_inference,
        transformer_calibrated_inference,
        cnn_calibrated_inference,
        APITransformerClassifier,
        MaleVisCNN,
        device,
    )

Each *_calibrated_inference() function is self-contained: it loads its own model
weights/artifacts from the paths you pass in, runs inference, applies Platt
scaling calibration, and returns a results dict that includes a "Top_Factors"
(or "File_index") field for downstream reporting.
"""

import os
import json
import math
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import shap


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model Architectures
# (needed here so *_calibrated_inference() can reconstruct the model before
#  loading trained weights via load_state_dict -- architecture is unchanged
#  from training, just redefined so this module has no dependency on the
#  training notebooks)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class APITransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, num_classes=2, dropout=0.12):
        super(APITransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)
        out = self.classifier(out)
        return out


class MaleVisCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaleVisCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# XGBoost Inference (CIC-MalMem-2022)
# ---------------------------------------------------------------------------

def unwrap_base_xgb_model(calibrated_model):
    """
    CalibratedClassifierCV wraps the underlying XGBoost model (possibly behind an
    extra FrozenEstimator layer, depending on sklearn version). Walk down through
    wrapper layers until we reach the actual XGBoost estimator (identified by
    having a get_booster() method, which only the real XGBoost model has).
    """
    candidate = calibrated_model.calibrated_classifiers_[0]
    for attr in ("estimator", "base_estimator"):
        if hasattr(candidate, attr):
            candidate = getattr(candidate, attr)
            break
    while hasattr(candidate, "estimator") and not hasattr(candidate, "get_booster"):
        candidate = candidate.estimator
    return candidate


def get_shap_top_factors(calibrated_model, scaled_df, predicted_class_id, top_n=3):
    """
    Extracts the top_n most influential features (by SHAP value, no abs() --
    only features that genuinely supported the predicted class) for the
    predicted class, using the XGBoost model underlying the Platt-scaled wrapper.
    """
    base_model = unwrap_base_xgb_model(calibrated_model)
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(scaled_df)

    # Handle both SHAP output conventions across versions:
    # - list of per-class arrays: shap_values[class_id][sample_idx]
    # - single array shaped (samples, features, classes)
    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class_id][0]
    else:
        class_shap = shap_values[0, :, predicted_class_id]

    feature_names = scaled_df.columns
    top_indices = np.argsort(class_shap)[::-1][:top_n]

    return [
        {"feature": str(feature_names[i]), "shap_value": float(class_shap[i])}
        for i in top_indices
    ]


def xgboost_calibrated_inference(input_data, calibrated_model_path, pipeline_path, encoder_path):
    """
    Performs standalone inference on new sample data using the Platt Scaled model.
    """
    # 1. Load artifacts
    calibrated_model = joblib.load(calibrated_model_path)
    scaler_pipe = joblib.load(pipeline_path)
    encoder = joblib.load(encoder_path)

    # 2. Format input data
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("input_data must be a pandas DataFrame or dictionary.")

    # Ensure columns match training feature set
    if hasattr(scaler_pipe, "feature_names_in_"):
        expected_cols = scaler_pipe.feature_names_in_
        df_input = df_input[expected_cols]

    # 3. Preprocess / Scale
    scaled_data = scaler_pipe.transform(df_input)
    scaled_df = pd.DataFrame(scaled_data, columns=df_input.columns)

    # 4. Predict probabilities via Platt-scaled model
    probabilities = calibrated_model.predict_proba(scaled_df)
    predictions = np.argmax(probabilities, axis=1)

    # 5. Decode predicted labels
    class_names = encoder.inverse_transform(predictions)

    # 6. Top contributing factors for the predicted class (SHAP, on the underlying tree model)
    top_factors = get_shap_top_factors(calibrated_model, scaled_df, predictions[0], top_n=3)

    # order of prob dist: benign, ransomware, spyware, trojan
    results = {
        "Predicted_Class_ID": predictions,
        "Predicted_Class": class_names,
        "Calibrated_Confidence": np.max(probabilities, axis=1),
        "Probability_Distribution": probabilities[0],
        "Top_Factors": top_factors
    }

    return results


# ---------------------------------------------------------------------------
# Transformer Inference (API call sequences)
# ---------------------------------------------------------------------------

def get_occlusion_top_factors(inference_model, seq_tensor, predicted_class_id, api_mapping,
                               pad_token_id=0, top_n=10):
    """
    Per-sample occlusion importance: masks one timestep at a time (replacing its API
    call id with pad_token_id) and measures how much the predicted class's probability
    drops. Larger drop = that timestep mattered more to the prediction.

    NOTE: this is a perturbation-based, PER-SAMPLE importance (the transformer analog
    of what SHAP does for XGBoost above) -- not a precomputed, dataset-level "global"
    importance. Getting a true global ranking would need a separate training-time
    computation (e.g. averaged occlusion importance over the validation set, saved as
    an artifact, same pattern as cnn_class_weights.json).

    pad_token_id: should match whatever padding/OOV token id was used during training.
    Defaults to 0 -- verify this against your vocab before trusting the output.
    """
    with torch.no_grad():
        baseline_probs = torch.softmax(inference_model(seq_tensor), dim=1)
        baseline_score = baseline_probs[0, predicted_class_id].item()

    seq_len = seq_tensor.shape[1]
    importances = []
    for t in range(seq_len):
        occluded = seq_tensor.clone()
        occluded[0, t] = pad_token_id
        with torch.no_grad():
            occluded_probs = torch.softmax(inference_model(occluded), dim=1)
            occluded_score = occluded_probs[0, predicted_class_id].item()
        importances.append(baseline_score - occluded_score)

    # Sort by raw importance (no abs()), keep only calls that genuinely INCREASED
    # the predicted class's confidence -- can return fewer than top_n entries.
    importances = np.array(importances)
    sorted_indices = np.argsort(importances)[::-1]
    positive_indices = [i for i in sorted_indices if importances[i] > 0]
    top_indices = positive_indices[:top_n]

    return [
        {
            "timestep": int(t),
            "api_call_id": int(seq_tensor[0, t].item()),
            "api_call": api_mapping[int(seq_tensor[0, t].item())],
            "importance": float(importances[t])
        }
        for t in top_indices
    ]


def transformer_calibrated_inference(input_sequences, model_path, platt_scaler_path, config_path, api_mapping):
    """
    Performs standalone inference on new API sequence samples using the Transformer + Platt Scaled calibrator.
    """
    # 1. Load Preprocessing Config
    with open(config_path, 'r') as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    num_classes = config["num_classes"]

    # 2. Format input sequence
    if isinstance(input_sequences, pd.DataFrame):
        sequences = input_sequences.values.astype(np.int64)
    elif isinstance(input_sequences, np.ndarray):
        sequences = input_sequences.astype(np.int64)
    else:
        sequences = np.array(input_sequences, dtype=np.int64)

    if sequences.ndim == 1:
        sequences = np.expand_dims(sequences, axis=0)

    # 3. Reconstruct PyTorch Model Architecture & Load State Dict
    inference_model = APITransformerClassifier(vocab_size=vocab_size, num_classes=num_classes).to(device)
    inference_model.load_state_dict(torch.load(model_path, map_location=device))
    inference_model.eval()

    # 4. Forward Pass to Get Raw Logits
    seq_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    with torch.no_grad():
        raw_logits = inference_model(seq_tensor).cpu().numpy()

    # 5. Apply Platt Scaler
    platt_scaler = joblib.load(platt_scaler_path)
    calibrated_probs = platt_scaler.predict_proba(raw_logits)
    pred_class_ids = np.argmax(calibrated_probs, axis=1)

    # 6. Map predictions
    target_mapping = {0: "Goodware/Benign", 1: "Malware"}
    class_names = [target_mapping.get(cid, f"Class_{cid}") for cid in pred_class_ids]

    # 7. Top contributing timesteps for the predicted class (occlusion-based importance)
    top_factors = get_occlusion_top_factors(inference_model, seq_tensor, pred_class_ids[0], api_mapping, top_n=10)

    # order of prob dist : Goodware, Malware
    results = {
        "Predicted_Class_ID": pred_class_ids,
        "Predicted_Class": class_names,
        "Calibrated_Confidence": np.max(calibrated_probs, axis=1),
        "Probability_Distribution": calibrated_probs[0],
        "Top_Factors": top_factors,
    }

    # sort based on timestep (so calls read in the order they occurred in the sequence)
    results['Top_Factors'] = sorted(results['Top_Factors'], key=lambda x: x['timestep'])

    return results


# ---------------------------------------------------------------------------
# CNN Inference (MaleVis)
# ---------------------------------------------------------------------------

def cnn_calibrated_inference(input_data, model_path, pipeline_path, calibration_path, file_index,
                              model_class=MaleVisCNN, device=device, sample_lookup=None):
    """
    Performs standalone inference on new image sample(s) using the Platt Scaled CNN model.

    input_data: any of the following, or a list mixing them:
      - a file path (str / os.PathLike)
      - a PIL.Image
      - an int index into `sample_lookup` (e.g. id=4) -- requires sample_lookup to be passed.
        This lets you manually pick a specific dataset sample (e.g. from test_samples) by
        index without pre-extracting paths yourself.
    sample_lookup: a list of (path, label) tuples, e.g. `test_samples`, used to resolve
        integer indices in input_data. Only required if you pass int indices.
    """
    # 1. Load preprocessing config
    with open(pipeline_path, "r") as f:
        pipeline_config = json.load(f)

    img_size = tuple(pipeline_config["img_size"])
    norm_mean = pipeline_config["norm_mean"]
    norm_std = pipeline_config["norm_std"]
    classes = pipeline_config["classes"]

    inference_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 2. Load model architecture (unchanged) and trained weights
    inf_model = model_class(num_classes=len(classes)).to(device)
    inf_model.load_state_dict(torch.load(model_path, map_location=device))
    inf_model.eval()

    # 3. Load fitted Platt scaling calibrators
    calibrators = joblib.load(calibration_path)

    # 4. Normalize input into a list of (PIL.Image, ground_truth_label_or_None)
    def resolve(item):
        if isinstance(item, (str, os.PathLike)):
            return Image.open(item).convert('RGB'), None
        elif isinstance(item, Image.Image):
            return item.convert('RGB'), None
        elif isinstance(item, (int, np.integer)):
            if sample_lookup is None:
                raise ValueError(
                    "Got an integer index but no sample_lookup was provided. "
                    "Pass sample_lookup=test_samples (or similar) to resolve indices."
                )
            path, label = sample_lookup[item]
            return Image.open(path).convert('RGB'), label
        else:
            raise ValueError("input_data items must be a file path, PIL Image, or int index.")

    if isinstance(input_data, list):
        resolved = [resolve(item) for item in input_data]
    else:
        resolved = [resolve(input_data)]

    images = [r[0] for r in resolved]
    ground_truth_ids = [r[1] for r in resolved]

    # 5. Preprocess and batch
    batch = torch.stack([inference_transform(img) for img in images]).to(device)

    # 6. Forward pass to get raw logits
    with torch.no_grad():
        logits = inf_model(batch).cpu().numpy()

    # 7. Apply Platt scaling calibration
    n_samples = logits.shape[0]
    n_classes = len(classes)
    calibrated = np.zeros((n_samples, n_classes))
    for c, calibrator in enumerate(calibrators):
        calibrated[:, c] = calibrator.predict_proba(logits[:, c].reshape(-1, 1))[:, 1]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    calibrated_probs_out = calibrated / row_sums

    predictions = np.argmax(calibrated_probs_out, axis=1)
    predicted_classes = [classes[p] for p in predictions]

    # 8. Build results in dictionary form
    results = {
        "Predicted_Class_ID": predictions,
        "Predicted_Class": predicted_classes,
        "Calibrated_Confidence": np.max(calibrated_probs_out, axis=1),
        "Probability_Distribution": calibrated_probs_out,
        "File_index": file_index
        # Ground_Truth_ID / Ground_Truth_Class intentionally omitted -- at real
        # inference time there are no ground-truth labels to compare against.
    }

    return results
