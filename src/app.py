import streamlit as st
import numpy as np
from joblib import load
from pathlib import Path
import random

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "scaler.joblib"

# Full feature list including 'id' and excluding 'class' at inference time
ALL_FEATURES = [
    "id", "gender", "PPE", "DFA", "RPDE", "numPulses", "numPeriodsPulses",
    "meanPeriodPulses", "stdDevPeriodPulses", "locPctJitter", "locAbsJitter",
    "rapJitter", "ppq5Jitter", "ddpJitter", "locShimmer", "locDbShimmer",
    "ddaShimmer", "meanAutoCorrHarmonicity", "meanNoiseToHarmHarmonicity",
    "meanHarmToNoiseHarmonicity", "minIntensity", "maxIntensity",
    "meanIntensity", "GQ_prc5_95", "GNE_SNR_TKEO", "GNE_SNR_SEO",
    "GNE_NSR_TKEO", "GNE_NSR_SEO", "VFER_mean", "VFER_std", "VFER_entropy",
    "VFER_SNR_TKEO", "VFER_SNR_SEO", "VFER_NSR_TKEO", "VFER_NSR_SEO",
    "IMF_SNR_SEO", "IMF_SNR_TKEO", "IMF_SNR_entropy", "IMF_NSR_SEO",
    "IMF_NSR_TKEO", "IMF_NSR_entropy", "mean_Log_energy",
    "mean_delta_log_energy", "std_Log_energy", "std_delta_log_energy", "Ea",
    "det_entropy_shannon_coef", "det_TKEO_mean_coef",
    "app_entropy_shannon_coef", "app_TKEO_std_coef", "Ea2",
    "det_LT_entropy_shannon"
]

# Slider ranges (adjust as needed)
FEATURE_RANGES = {
    "id": (1, 10000),
    "gender": (0, 1),
    "PPE": (0.0, 1.0),
    "DFA": (0.0, 2.0),
    "RPDE": (0.0, 1.0),
    "numPulses": (50, 500),
    "numPeriodsPulses": (50, 500),
    "meanPeriodPulses": (0.001, 0.02),
    "stdDevPeriodPulses": (0.0, 0.01),
    "locPctJitter": (0.0, 0.1),
    "locAbsJitter": (0.0, 0.001),
    "rapJitter": (0.0, 0.1),
    "ppq5Jitter": (0.0, 0.1),
    "ddpJitter": (0.0, 0.3),
    "locShimmer": (0.0, 1.0),
    "locDbShimmer": (0.0, 5.0),
    "ddaShimmer": (0.0, 3.0),
    "meanAutoCorrHarmonicity": (0.0, 1.0),
    "meanNoiseToHarmHarmonicity": (0.0, 50.0),
    "meanHarmToNoiseHarmonicity": (0.0, 50.0),
    "minIntensity": (40.0, 80.0),
    "maxIntensity": (60.0, 120.0),
    "meanIntensity": (50.0, 100.0),
    "GQ_prc5_95": (0.0, 100.0),
    "GNE_SNR_TKEO": (0.0, 50.0),
    "GNE_SNR_SEO": (0.0, 50.0),
    "GNE_NSR_TKEO": (0.0, 1.0),
    "GNE_NSR_SEO": (0.0, 1.0),
    "VFER_mean": (0.0, 10.0),
    "VFER_std": (0.0, 5.0),
    "VFER_entropy": (0.0, 10.0),
    "VFER_SNR_TKEO": (0.0, 50.0),
    "VFER_SNR_SEO": (0.0, 50.0),
    "VFER_NSR_TKEO": (0.0, 1.0),
    "VFER_NSR_SEO": (0.0, 1.0),
    "IMF_SNR_SEO": (0.0, 50.0),
    "IMF_SNR_TKEO": (0.0, 50.0),
    "IMF_SNR_entropy": (0.0, 10.0),
    "IMF_NSR_SEO": (0.0, 1.0),
    "IMF_NSR_TKEO": (0.0, 1.0),
    "IMF_NSR_entropy": (0.0, 10.0),
    "mean_Log_energy": (-10.0, 10.0),
    "mean_delta_log_energy": (-5.0, 5.0),
    "std_Log_energy": (0.0, 10.0),
    "std_delta_log_energy": (0.0, 5.0),
    "Ea": (0.0, 1.0),
    "det_entropy_shannon_coef": (0.0, 10.0),
    "det_TKEO_mean_coef": (0.0, 100.0),
    "app_entropy_shannon_coef": (0.0, 10.0),
    "app_TKEO_std_coef": (0.0, 100.0),
    "Ea2": (0.0, 1.0),
    "det_LT_entropy_shannon": (0.0, 10.0),
}

@st.cache_resource
def load_assets():
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    return model, scaler

def initialize_session_state():
    if 'feature_values' not in st.session_state:
        st.session_state.feature_values = {feat: 0.0 for feat in ALL_FEATURES}

def randomize_features():
    for feat in ALL_FEATURES:
        min_val, max_val = FEATURE_RANGES[feat]
        if feat == "gender":
            st.session_state.feature_values[feat] = random.choice([0, 1])
        elif feat == "id":
            st.session_state.feature_values[feat] = random.randint(int(min_val), int(max_val))
        else:
            st.session_state.feature_values[feat] = round(random.uniform(float(min_val), float(max_val)), 6)

def main():
    st.title("🧠 Parkinson Disease Detection")
    st.markdown(
        "Use the sliders below to set feature values, or click “Randomize” to generate random values. "
        "The app scales features first and then applies the trained selector to match the training pipeline."
    )

    initialize_session_state()
    model, scaler = load_assets()

    col1, _ = st.columns([1, 4])
    with col1:
        if st.button("🎲 Randomize", type="secondary"):
            randomize_features()
            st.rerun()

    st.subheader("📊 Feature Values")

    user_input = {}
    num_cols = 3
    cols = st.columns(num_cols)

    for i, feat in enumerate(ALL_FEATURES):
        min_val, max_val = FEATURE_RANGES[feat]
        col = cols[i % num_cols]
        with col:
            if feat == "gender":
                user_input[feat] = st.slider(
                    feat, min_value=0, max_value=1,
                    value=int(st.session_state.feature_values[feat]),
                    step=1, key=f"slider_{feat}"
                )
            elif feat == "id":
                user_input[feat] = st.slider(
                    feat, min_value=int(min_val), max_value=int(max_val),
                    value=int(st.session_state.feature_values[feat]),
                    step=1, key=f"slider_{feat}"
                )
            else:
                user_input[feat] = st.slider(
                    feat, min_value=float(min_val), max_value=float(max_val),
                    value=float(st.session_state.feature_values[feat]),
                    format="%.6f", key=f"slider_{feat}"
                )

    st.session_state.feature_values = user_input.copy()

    st.subheader("🔮 Prediction")
    if st.button("Predict", type="primary"):
        try:
            # Build input vector (exclude any target column; ALL_FEATURES list already excludes 'class')
            vec = np.array([user_input[feat] for feat in ALL_FEATURES]).reshape(1, -1)

            # 1) Scale full feature vector
            vec_scaled_full = scaler.transform(vec)

            # 2) Select features using the trained selector mask
            selector = model.named_steps["selector"]
            mask = selector.get_support()
            vec_selected = vec_scaled_full[:, mask]

            clf = model.named_steps["clf"]

            # Predictions
            pred = clf.predict(vec_selected)[0]

            # Safe probability extraction using classes_
            prob = None
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(vec_selected)
                # If classifier learned only one class, probs shape may be (n,1)
                if probs.shape[1] == 1:
                    # Only one class present; map to probability of Parkinson (assumed class label 1)
                    if hasattr(clf, "classes_"):
                        only_class = clf.classes_[0]
                        prob = 1.0 if only_class == 1 else 0.0
                    else:
                        prob = float(pred == 1)
                else:
                    # Find index of positive class (assumed label 1)
                    if hasattr(clf, "classes_"):
                        classes = clf.classes_
                        if 1 in classes:
                            pos_idx = int(np.where(classes == 1)[0][0])
                            prob = float(probs[0, pos_idx])
                        else:
                            # If labels are not {0,1}, pick the class with higher label as "positive"
                            pos_idx = int(np.argmax(classes))
                            prob = float(probs[0, pos_idx])
                    else:
                        # Fallback: take the second column if available (binary)
                        prob = float(probs[0, min(1, probs.shape[1]-1)])
            else:
                # No predict_proba available: use decision_function if present to create a pseudo-probability
                if hasattr(clf, "decision_function"):
                    df = clf.decision_function(vec_selected)
                    # Sigmoid mapping as a rough score to [0,1]
                    prob = float(1 / (1 + np.exp(-df[0]))) if np.ndim(df) == 1 else float(1 / (1 + np.exp(-df[0, 0])))
                else:
                    prob = float(pred == 1)

            st.markdown("---")
            if pred == 1:
                st.error("🩺 Parkinson Disease Detected")
                st.error(f"Probability: {prob:.2%}")
            else:
                st.success("✅ Healthy")
                st.info(f"Probability of Parkinson: {prob:.2%}")

            confidence = max(prob, 1 - prob)
            st.progress(confidence)
            st.caption(f"Model Confidence: {confidence:.2%}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.error("Please check that the scaler and model were trained with the same feature set and order, and that the selector mask matches the scaled feature dimensions.")

if __name__ == "__main__":
    main()
