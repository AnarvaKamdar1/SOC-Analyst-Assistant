"""
json_summary_generator.py

Builds the fusion system's final_summary dict (the structured JSON payload
handed to the SOC report LLM prompt) from the three detectors' raw results
plus the combining-logic outputs.

Intended usage (from the `utilities` Kaggle dataset):

    sys.path.append("/kaggle/input/datasets/anarvaaa/utilities")
    from json_summary_generator import build_final_summary

    final_summary = build_final_summary(
        base_P_mal, disagreement, final_P_mal,
        xgb_results, transformer_results, cnn_all_results, scored_files
    )
"""


def build_final_summary(base_P_mal, disagreement, final_P_mal,
                         xgb_results, transformer_results, cnn_all_results, scored_files):
    """Assemble the final_summary dict from detector outputs.

    Args:
        base_P_mal: base malicious probability (XGBoost + Transformer), from total_maliciousness().
        disagreement: disagreement score, from total_maliciousness().
        final_P_mal: final malicious probability (after CNN ceiling adjustment), from total_maliciousness().
        xgb_results: dict returned by xgboost_calibrated_inference().
        transformer_results: dict returned by transformer_calibrated_inference().
        cnn_all_results: list of dicts returned by cnn_calibrated_inference() (one per scanned file).
        scored_files: list of (file_index, predicted_class, confidence, raw_score) tuples, from total_maliciousness().

    Returns:
        dict -- the final_summary payload.
    """

    return {
        "Base_P_malicious": round(base_P_mal, 4),
        "Disagreement": round(disagreement, 4),

        "CNN_evidence_rise_pct": round((final_P_mal - base_P_mal) * 100, 4),
        "Final_P_malicious": round(final_P_mal, 4),

        "Xgboost_Summary": {
            "predicted_state": xgb_results['Predicted_Class'][0],
            "confidence": round(xgb_results['Calibrated_Confidence'][0], 3),
            "supporting_evidence": [xgb_results['Top_Factors'][i]['feature']
                        for i in range(0, len(xgb_results['Top_Factors']))]
                        if xgb_results['Predicted_Class'][0] != "Benign" else []
        },

        "Transformer_Summary": {
            "predicted_state": transformer_results['Predicted_Class'][0],
            "confidence": round(transformer_results['Calibrated_Confidence'][0], 3),
            "supporting_evidence": [transformer_results['Top_Factors'][i]['api_call']
                       for i in range(0, len(transformer_results['Top_Factors']))]
                        if transformer_results['Predicted_Class'][0] == "Malware" else []
        },

        "CNN_Summary": {
            "total_scanned_files": len(cnn_all_results),
            "influencial_files": [
                {"file_index": file_index, "predicted_class": predicted_class, "confidence": round(float(confidence), 3)}
                for file_index, predicted_class, confidence, raw_score in scored_files
            ]
        }
    }
