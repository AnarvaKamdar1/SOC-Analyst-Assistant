"""
timeline_runner.py

Runs the SOC-Analyst-Assistant detection pipeline (XGBoost + Transformer +
CNN -> fusion -> security-reference grounding -> LLM report) across a
sequence of independent evidence snapshots -- a "timeline".

Each entry in `evidence_sequence` is an independent snapshot of system
state observed at a given timestamp t:

    "t10": [xgb_idx, api_idx, [cnn_idx, cnn_idx, ...]]

    - xgb_idx:  row index into the MalMem2022 dataframe. Represents the
                memory/process snapshot taken AT time t.
    - api_idx:  row index into the API-call dataframe. Represents the
                sequence of Windows API/syscalls observed SINCE the
                previous snapshot (a windowed/aggregated behavioral trace,
                not a point sample).
    - cnn_idxs: list of indices into the MaleVis image candidates.
                Represents zero or more suspicious files that were
                flagged and scanned SINCE the previous snapshot. Can be
                empty if no new files were observed in that window.

Consistent with the v1 design, evidence channels at a given t are treated
as independent -- xgb/api/cnn readings at the same timestamp are not
assumed to originate from the same underlying event, only from the same
window of observation. This module does not accumulate maliciousness
across timestamps (that's v3's job, layered on top of the `history` this
produces); it only guarantees that every timestamp is run independently
and its full result is preserved for later use.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class PipelineContext:
    """
    Bundles every model, artifact path, and precomputed config the
    detector/fusion/report stages need, so run_timeline doesn't take
    fifteen positional arguments and so the same context can be reused
    across every snapshot without re-loading models per timestamp.

    Build this once in the notebook (after loading data/models/artifacts)
    and pass it into run_timeline.
    """
    # Data sources
    df_malmem: Any
    df_api: Any
    seq_cols: list
    sample_image_candidates: list

    # XGBoost artifacts
    xgb_model_path: str
    xgb_pipeline_path: str
    xgb_encoder_path: str

    # Transformer artifacts
    transformer_model_path: str
    transformer_platt_scalar_path: str
    transformer_pipeline_path: str
    api_calls_mapping: dict

    # CNN artifacts
    cnn_model_path: str
    cnn_pipeline_path: str
    cnn_calibration_path: str
    cnn_class_weights_data: dict

    # Fusion config
    xgb_per_class_fscores: dict
    transformer_per_class_fscores: dict
    hierarchy: Any
    max_limit_files: int
    per_class_limit: int

    # LLM
    tokenizer: Any
    model: Any
    device: Any


def _parse_t_label(t_label: str) -> int:
    """'t10' -> 10. Centralized so the label format only needs to change
    in one place if it ever does."""
    return int(t_label.lstrip("t"))


def run_single_snapshot(
    t_label: str,
    xgb_idx: int,
    api_idx: int,
    cnn_idxs: list,
    ctx: PipelineContext,
) -> dict:
    """
    Runs ONE independent snapshot through the full detection, fusion,
    grounding, and report-generation pipeline. This is the same logic
    v1's main.ipynb ran once; here it's isolated so run_timeline can
    call it repeatedly without duplicating pipeline code.
    """
    # Imported here (not at module top) so this file only depends on the
    # rest of utilities/ at call time, matching how main.ipynb wires
    # things up -- keeps this module import-order-agnostic.
    from inference_functions import (
        xgboost_calibrated_inference,
        transformer_calibrated_inference,
        cnn_calibrated_inference,
    )
    from combining_logic import total_maliciousness
    from json_summary_generator import build_final_summary
    from prompt_generation import generate_soc_report

    t_value = _parse_t_label(t_label)

    xgb_input = ctx.df_malmem.iloc[[xgb_idx]]
    transformer_input = ctx.df_api[ctx.seq_cols].iloc[[api_idx]]

    xgb_results = xgboost_calibrated_inference(
        input_data=xgb_input,
        calibrated_model_path=ctx.xgb_model_path,
        pipeline_path=ctx.xgb_pipeline_path,
        encoder_path=ctx.xgb_encoder_path,
    )

    transformer_results = transformer_calibrated_inference(
        input_sequences=transformer_input,
        model_path=ctx.transformer_model_path,
        platt_scaler_path=ctx.transformer_platt_scalar_path,
        config_path=ctx.transformer_pipeline_path,
        api_mapping=ctx.api_calls_mapping,
    )

    cnn_all_results = []
    for i in cnn_idxs:
        cnn_results = cnn_calibrated_inference(
            input_data=ctx.sample_image_candidates[i],
            model_path=ctx.cnn_model_path,
            pipeline_path=ctx.cnn_pipeline_path,
            calibration_path=ctx.cnn_calibration_path,
            file_index=i,
        )
        cnn_all_results.append(cnn_results)

    base_P_mal, cnn_effect, final_P_mal, disagreement, scored_files = total_maliciousness(
        xgb_results,
        transformer_results,
        cnn_all_results,
        ctx.cnn_class_weights_data,
        ctx.xgb_per_class_fscores,
        ctx.transformer_per_class_fscores,
        ctx.hierarchy,
        ctx.max_limit_files,
        ctx.per_class_limit,
    )

    final_summary = build_final_summary(
        base_P_mal,
        disagreement,
        final_P_mal,
        xgb_results,
        transformer_results,
        cnn_all_results,
        scored_files,
    )

    soc_report, matched_snippets, unmatched_labels = generate_soc_report(
        final_summary, ctx.tokenizer, ctx.model, ctx.device
    )

    return {
        "t_label": t_label,
        "t": t_value,
        "base_P_mal": base_P_mal,
        "cnn_effect": cnn_effect,
        "final_P_mal": final_P_mal,
        "disagreement": disagreement,
        "final_summary": final_summary,
        "soc_report": soc_report,
        "matched_snippets": matched_snippets,
        "unmatched_labels": unmatched_labels,
    }


def run_timeline(
    evidence_sequence: dict,
    ctx: PipelineContext,
    verbose: bool = True,
    on_step: Optional[Callable[[dict], None]] = None,
) -> list:
    """
    Runs the pipeline independently across every snapshot in
    evidence_sequence and returns the accumulated history.

    Args:
        evidence_sequence: dict of {"t<N>": [xgb_idx, api_idx, [cnn_idxs]]}.
            Keys are sorted numerically before iterating, so insertion
            order in the dict literal doesn't matter.
        ctx: a PipelineContext built once with all loaded models/artifacts.
        verbose: if True, prints each SOC report as it's generated
            (matches the current notebook's behavior).
        on_step: optional callback invoked with each result dict right
            after it's produced -- use this to persist incrementally
            (e.g. append to a JSON file or DB row) instead of only
            keeping results in memory. This is the hook v3's rolling
            EWMA computation, or a future container's replay loop, would
            plug into.

    Returns:
        list of result dicts, one per timestamp, in chronological order.
        Each dict has the same shape as run_single_snapshot's return value.
    """
    history = []

    ordered_labels = sorted(evidence_sequence.keys(), key=_parse_t_label)

    for t_label in ordered_labels:
        xgb_idx, api_idx, cnn_idxs = evidence_sequence[t_label]

        result = run_single_snapshot(t_label, xgb_idx, api_idx, cnn_idxs, ctx)

        if verbose:
            print(
                "=" * 80
                + f"\nSOC ANALYST REPORT at T={result['t']} units\n"
                + "=" * 80
                + "\n"
            )
            print(result["soc_report"])

        history.append(result)

        if on_step:
            on_step(result)

    return history
