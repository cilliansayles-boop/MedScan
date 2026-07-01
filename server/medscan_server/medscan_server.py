"""
PATCH — Symptom Contributions ("What influenced this result")
================================================================
This is NOT a replacement for medscan_server.py — it's three small,
additive changes for Taha to review and merge himself. Nothing here
overwrites his code; it only adds one new function and extends the
existing response with one new optional field.

WHAT IT DOES
------------
For each symptom the user answered "yes" to (non-zero in symp_input),
re-run inference with just that one symptom zeroed out, and measure
how much the predicted class's probability drops. A big drop means
that symptom was doing a lot of work; near-zero means it barely
mattered. This is the same "zero it out and measure the damage"
occlusion approach already used for feature importance in Cell 7 of
HybridCRNN_v6_FIXED.ipynb (Panel 6/7) — just applied per-prediction
instead of across the whole test set.

It is NOT SHAP or Integrated Gradients — it's a cheaper approximation
that reuses run_inference() as-is. The frontend explicitly labels it
as an approximation, not an exact breakdown, so it doesn't overclaim.

COST
----
Only re-runs inference for symptoms that were actually answered "yes"
(typically 0-6 of the 14 fields per request), and only symp_input
changes each time — spec_input/feat_input (the expensive spectrogram
+ librosa part) are computed once and reused. On the TFLite path this
is a handful of extra interpreter.invoke() calls, each on tiny (1,14)
input — should add well under 100ms even on Render's free tier CPU.
Worth Taha timing it before merging, but it shouldn't be noticeable.


STEP 1 — Add this function to medscan_server.py, directly after run_inference()
--------------------------------------------------------------------------------
"""

def compute_symptom_contributions(spec_input, feat_input, symp_input,
                                   baseline_probs: dict, condition: str,
                                   top_n: int = 5) -> list:
    """
    Occlusion-based, per-prediction symptom attribution.

    For each active (non-zero) symptom field, zero it out, re-run
    run_inference() with the SAME spec_input/feat_input (only symp_input
    changes), and record how much the predicted class's probability
    shifted. Returns the top_n by absolute magnitude.

    Positive delta  -> removing this symptom REDUCED confidence in the
                        predicted condition, i.e. it was evidence FOR it.
    Negative delta  -> removing this symptom INCREASED confidence, i.e.
                        it was mildly evidence AGAINST the predicted
                        condition (rare, but honest to show if it happens).
    """
    fields = MODEL_META['symptom_fields']
    baseline_target = baseline_probs[condition]
    contributions = []

    for i, field in enumerate(fields):
        if symp_input[0, i] == 0:
            continue  # nothing to attribute — this symptom wasn't present

        perturbed = symp_input.copy()
        perturbed[0, i] = 0.0

        try:
            perturbed_result = run_inference(spec_input, feat_input, perturbed)
        except Exception as e:
            logger.warning(f"Symptom contribution step failed for '{field}': {e}")
            continue

        perturbed_target = perturbed_result['probabilities'].get(condition, baseline_target)
        delta = float(baseline_target - perturbed_target)

        contributions.append({
            "field": field,
            "delta": round(delta, 4),
            "value": float(symp_input[0, i]),
        })

    contributions.sort(key=lambda c: abs(c["delta"]), reverse=True)
    return contributions[:top_n]


"""
STEP 2 — Extend PredictionResult (near the other response models)
--------------------------------------------------------------------------------
Add ONE line to the existing model — don't remove anything:

    class PredictionResult(BaseModel):
        condition:     str
        severity:      str
        confidence:    float
        explanation:   str
        actions:       list
        seeDoctor:     bool
        probabilities: dict = {}
        timestamp:     str  = ""
        symptom_contributions: list = []      # <-- ADD THIS LINE


STEP 3 — Call it inside process_prediction_sync(), right after run_inference()
--------------------------------------------------------------------------------
Current code (for reference, unchanged):

    result        = run_inference(spec, feat, symp)
    condition     = result['condition']
    confidence    = result['confidence']
    severity_info = generate_explanation(condition, confidence)

Add ONE call and ONE dict key — everything else in that function stays as-is:

    result        = run_inference(spec, feat, symp)
    condition     = result['condition']
    confidence    = result['confidence']
    severity_info = generate_explanation(condition, confidence)

    # NEW — per-scan symptom attribution (cheap: only re-runs the tiny
    # symp_input branch, spec/feat aren't recomputed)
    symptom_contributions = compute_symptom_contributions(
        spec, feat, symp, result['probabilities'], condition
    )

    logger.info(f"[{request_id}] PREDICTION: {condition} ({confidence:.1%}) | {result['probabilities']}")

    return {
        "condition": condition,
        "severity": severity_info['severity'],
        "confidence": confidence,
        "explanation": severity_info['explanation'],
        "actions": severity_info['actions'],
        "seeDoctor": severity_info['seeDoctor'],
        "probabilities": result['probabilities'],
        "timestamp": datetime.now().isoformat(),
        "symptom_contributions": symptom_contributions,   # <-- ADD THIS LINE
    }


THAT'S IT
---------
Nothing else in medscan_server.py needs to change. The frontend
(App.jsx) already reads `result.symptom_contributions` and quietly
shows nothing on the result card if the field is missing or empty —
so this can be merged whenever Taha's ready and won't break anything
if it ships later than the frontend does.

Worth Taha double-checking before merging:
  - That `symp_input`/`symp` really is a plain numpy array with
    .copy() available at that point in the pipeline (it is, per the
    current code — np.expand_dims returns an ndarray) — just flagging
    since this patch depends on that.
  - Whether 20 requests/minute (the existing rate limit) is still fine
    given each request now does a few extra tiny inferences.
"""
