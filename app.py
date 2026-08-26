import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
import cv2
import joblib

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dyslexia Detection Web App",
    layout="wide"
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "handwriting_model.pkl"
)


# =========================================================
# BASIC CSS
# =========================================================

hide_menu_style = """
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { font-weight: 600; }
</style>
"""

st.markdown(
    hide_menu_style,
    unsafe_allow_html=True
)


# =========================================================
# HANDWRITING 13-FEATURE EXTRACTION
# =========================================================

def extract_handwriting_features(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Unable to read handwriting image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    image_height, image_width = binary.shape

    # 1. INK DENSITY
    ink_pixels = np.count_nonzero(binary)
    ink_density = (ink_pixels / (image_height * image_width)) * 100

    # 2. CONNECTED COMPONENTS
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 20 and h >= 5 and w >= 2:
            components.append({
                "x": x, "y": y, "w": w, "h": h, "area": area
            })

    if len(components) < 2:
        raise ValueError(
            "Not enough handwriting detected. Please upload a clearer handwriting image."
        )

    components.sort(key=lambda c: c["x"])

    # 3. MEAN LETTER HEIGHT & VARIATION
    heights = np.array([c["h"] for c in components], dtype=float)
    mean_height = np.mean(heights)
    height_variation = (np.std(heights) / (mean_height + 1e-6)) * 100

    # 4. MEAN LETTER WIDTH & VARIATION
    widths = np.array([c["w"] for c in components], dtype=float)
    mean_width = np.mean(widths)
    width_variation = (np.std(widths) / (mean_width + 1e-6)) * 100

    # 5. BASELINE DRIFT
    baseline_positions = np.array([c["y"] + c["h"] for c in components], dtype=float)
    baseline_drift = np.std(baseline_positions)

    # 6. SPACING & SPACING VARIATION
    spacings = []
    for i in range(len(components) - 1):
        current = components[i]
        next_component = components[i + 1]
        gap = next_component["x"] - (current["x"] + current["w"])
        if gap >= 0:
            spacings.append(gap)

    if len(spacings) > 0:
        spacings = np.array(spacings, dtype=float)
        mean_spacing = np.mean(spacings)
        spacing_variation = (np.std(spacings) / (mean_spacing + 1e-6)) * 100
    else:
        mean_spacing, spacing_variation = 0.0, 0.0

    # 7. STROKE WIDTH
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_region = distance[binary > 0]
    estimated_stroke_width = np.mean(ink_region) * 2 if len(ink_region) > 0 else 0.0

    # 8. SLANT VARIATION
    slant_angles = []
    for component in components:
        x, y, w, h = component["x"], component["y"], component["w"], component["h"]
        roi = binary[y:y + h, x:x + w]
        moments = cv2.moments(roi)
        if abs(moments["mu02"]) > 1e-6:
            angle = 0.5 * np.arctan2(
                2 * moments["mu11"],
                moments["mu20"] - moments["mu02"]
            )
            slant_angles.append(np.degrees(angle))

    slant_variation = np.std(slant_angles) if len(slant_angles) > 0 else 0.0

    # 9. MARGIN ALIGNMENT
    left_positions = np.array([c["x"] for c in components], dtype=float)
    margin_alignment = np.std(left_positions)

    # 10. OVERLAP / COLLISION
    overlap_count = sum(
        1 for i in range(len(components) - 1)
        if components[i + 1]["x"] < components[i]["x"] + components[i]["w"]
    )
    overlap_ratio = (overlap_count / (len(components) - 1)) * 100

    return [
        ink_density, float(len(components)), mean_height, height_variation,
        mean_width, width_variation, baseline_drift, mean_spacing,
        spacing_variation, estimated_stroke_width, slant_variation,
        margin_alignment, overlap_ratio
    ]


# =========================================================
# FEATURE NAMES & COLUMN SCHEMA
# =========================================================

FEATURE_NAMES = [
    "Ink Density (%)",
    "Connected Components",
    "Mean Letter Height",
    "Height Variation (%)",
    "Mean Letter Width",
    "Width Variation (%)",
    "Baseline Drift",
    "Mean Spacing",
    "Spacing Variation (%)",
    "Estimated Stroke Width",
    "Slant Variation",
    "Margin Alignment",
    "Overlap / Collision (%)"
]

MODEL_FEATURE_COLUMNS = [
    "ink_density",
    "connected_components",
    "mean_height",
    "height_variation",
    "mean_width",
    "width_variation",
    "baseline_drift",
    "mean_spacing",
    "spacing_variation",
    "estimated_stroke_width",
    "slant_variation",
    "margin_alignment",
    "overlap_ratio"
]


def get_feature_array(path):
    return extract_handwriting_features(path)


# =========================================================
# MODEL SCORING
# =========================================================

def score(input_features):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "handwriting_model.pkl was not found at:\n" + MODEL_PATH
        )

    model = joblib.load(MODEL_PATH)
    features = pd.DataFrame([input_features], columns=MODEL_FEATURE_COLUMNS)

    if hasattr(model, "feature_names_in_"):
        features = features[model.feature_names_in_]

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    dyslexia_probability = 0.0
    for cls, probability in zip(classes, probabilities):
        if int(cls) == 1:
            dyslexia_probability = float(probability) * 100

    return prediction, dyslexia_probability


# =========================================================
# STREAMLIT UI
# =========================================================

st.header("Dyslexia Detection Web APP")

tab1, tab2, tab3 = st.tabs(["Home", "Writing", "About"])


# =========================================================
# TAB 1 - HOME (DETAILED CONTEXT)
# =========================================================

with tab1:
    st.header("Home Page")

    st.write(
        """
        **Dyslexia** is a specific, neurodevelopmental learning condition characterized by difficulties
        with accurate and fluent word recognition, decoding abilities, and spelling. It originates from
        neurological differences in the brain areas responsible for phonological processing—the ability
        to isolate, manipulate, and map speech sounds (phonemes) to their written visual representations
        (graphemes).
        """
    )

    st.write(
        """
        Crucially, dyslexia is **independent of cognitive intelligence, motivation, sensory acuity, or socioeconomic background**.
        Individuals with dyslexia often exhibit normal or above-average intellectual capabilities, problem-solving skills,
        and creativity, but encounter persistent barriers when translating spoken language into orthographic symbols and vice-versa.
        """
    )

    home_image_path = os.path.join(BASE_DIR, "home_infographic.jpg")
    if os.path.exists(home_image_path):
        st.image(home_image_path, width=600)
    else:
        st.info("Home page image not found.")

    st.markdown("---")

    # -----------------------------------------------------
    # CLINICAL MANIFESTATIONS & NEUROLOGICAL CONTEXT
    # -----------------------------------------------------
    st.subheader("Clinical Manifestations & Neurological Basis")

    st.write(
        """
        Functional neuroimaging studies (fMRI) indicate that individuals with dyslexia show reduced neural activation
        in the left-hemisphere reading network (specifically the left parieto-temporal and occipito-temporal/visual word form areas)
        and compensatory over-activation in the inferior frontal gyrus (Broca’s area) and right hemisphere.
        This structural divergence leads to core behavioral indicators:
        """
    )

    st.markdown(
        """
        - **Phonological Awareness Deficits:** Severe difficulty segmenting multisyllabic words, blending phonemes, and isolating phonetic units.
        - **Orthographic Processing Lags:** Inability to build an automated visual memory bank of sight words, causing laborious letter-by-letter decoding.
        - **Rapid Automatized Naming (RAN) Challenges:** Slower retrieval speeds when naming familiar symbols, digits, and colors under time constraints.
        - **Working Memory Constraints:** Reduced capacity in the phonological loop, causing learners to forget early parts of a sentence before finishing reading it.
        """
    )

    # -----------------------------------------------------
    # DYSLEXIA & COMORBID DYSGRAPHIA IN HANDWRITING
    # -----------------------------------------------------
    st.subheader("The Handwriting Connection: Motor & Spatial Dysgraphia")

    st.write(
        """
        While dyslexia is primarily classified as a reading disability, it is heavily comorbid with **Dysgraphia**
        (impairment in written expression and motor execution). Handwriting is an intensive cognitive process requiring
        simultaneous phoneme-to-grapheme retrieval, spatial planning, working memory allocation, and fine-motor execution.
        """
    )

    st.write(
        """
        When phonological retrieval demands excessive cognitive load, handwriting execution degrades noticeably.
        Computational computer vision can extract quantifiable indicators from handwriting canvases:
        """
    )

    st.markdown(
        """
        1. **Spatial Disorganization:** Inability to maintain straight writing lines without ruled grids, resulting in **high baseline drift** and inconsistent left-margin alignment.
        2. **Sizing Dysregulation:** Inconsistent letter scaling (**high height & width variation**), where ascenders and descenders collide arbitrarily.
        3. **Rhythm & Spacing Breakdown:** Erratic inter-character and inter-word gaps, frequently accompanied by stroke collisions.
        4. **Motor Pressure & Hesitations:** Variations in pen pressure and stroke width caused by micro-hesitations during lexical retrieval.
        5. **Letter Reversals & Directional Instability:** Visual-spatial disorientation resulting in mirrored strokes and unstable slant angles.
        """
    )

    # -----------------------------------------------------
    # PREVALENCE & IMPORTANCE OF EARLY SCREENING
    # -----------------------------------------------------
    st.subheader("Prevalence, Demographics & Early Screening")

    st.write(
        """
        According to international developmental data, dyslexia affects approximately **10% to 15% of the global population**,
        representing the single most prevalent learning disability in school-aged children.
        In multilingual regions such as India, children learning multiple orthographic systems simultaneously face compounded orthographic challenges.
        """
    )

    st.write(
        """
        Traditional diagnostic protocols require exhaustive neuropsychological batteries conducted over multiple days by licensed clinicians.
        This application provides a rapid, non-invasive **AI-assisted preliminary screening framework** using Computer Vision
        to triage students who require formal clinical evaluations.
        """
    )

    st.subheader("Important Medical & Educational Notice")
    st.info(
        """
        This application is an educational, research-oriented screening tool. The generated dyslexia score represents
        a probabilistic pattern classification derived from computer-vision algorithms. It must not be interpreted
        as a definitive medical diagnosis. A formal dyslexia diagnosis must be administered by certified clinical psychologists
        or educational specialists using standardized clinical batteries.
        """
    )


# =========================================================
# TAB 2 - WRITING & PREDICTION (HANDWRITING BIOMECHANICS ONLY)
# =========================================================

with tab2:
    st.title("Dyslexia Detection Using Handwriting Samples")

    st.write(
        """
        Upload a handwriting sample and click **Predict**.
        The application processes the image using computer vision, extracts **13 biomechanical handwriting characteristics**,
        and evaluates classification likelihood using our trained machine learning model.
        """
    )

    image = st.file_uploader(
        "Upload the handwriting sample",
        type=["jpg", "jpeg", "png"],
        key="handwriting_upload"
    )

    if image is not None:
        image_uploaded = Image.open(image)
        temp_path = os.path.join(BASE_DIR, "temp_handwriting.jpg")
        image_uploaded.save(temp_path)

        st.write(f"Selected image: {image.name}")
        st.image(image_uploaded, width=400)

        if st.button("Predict", key="predict_handwriting"):
            try:
                # -------------------------------------------------
                # 1. EXTRACT 13 BIOMECHANICAL FEATURES
                # -------------------------------------------------
                feature_array = get_feature_array(temp_path)

                feature_df = pd.DataFrame(
                    {
                        "Biomechanical Parameter": FEATURE_NAMES,
                        "Calculated Value": [round(val, 2) for val in feature_array]
                    }
                )

                st.subheader("1. Extracted Handwriting Biomechanical Features")
                st.dataframe(feature_df)

                # -------------------------------------------------
                # 2. MODEL PREDICTION
                # -------------------------------------------------
                result, dyslexia_score = score(feature_array)

                st.subheader("2. Classification Verdict & Probability")
                if result == 1:
                    st.warning("Higher likelihood of dyslexia markers detected based on the trained handwriting model.")
                else:
                    st.success("Lower likelihood of dyslexia markers detected based on the trained handwriting model.")

                score_value = int(max(0, min(100, dyslexia_score)))
                st.progress(score_value)
                st.metric("Dyslexia Probability", f"{dyslexia_score:.2f}%")

                if dyslexia_score >= 70:
                    st.error("High Model Confidence (Score >= 70%)")
                elif dyslexia_score >= 40:
                    st.warning("Moderate Model Confidence (40% - 69%)")
                else:
                    st.success("Low Model Confidence (Score < 40%)")

                st.info(
                    """
                    This score reflects the ensemble model's posterior probability for the dyslexic class
                    based on spatial and structural handwriting traits.
                    """
                )

            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)


# =========================================================
# TAB 3 - ABOUT (DETAILED ARCHITECTURE & BENCHMARKS)
# =========================================================

with tab3:
    st.header("About the Application")

    st.write(
        """
        This application is an educational and experimental Artificial Intelligence system
        engineered to investigate whether quantifiable handwriting traits
        can serve as an accurate, objective preliminary screening modality for dyslexia.
        """
    )

    st.write(
        """
        By marrying **Computer Vision (OpenCV)** and **Machine Learning (Scikit-Learn)**,
        the platform assesses handwriting on spatial and biomechanical levels.
        """
    )

    # -----------------------------------------------------
    # SYSTEM PIPELINE & MATHEMATICAL FORMULATIONS
    # -----------------------------------------------------
    st.subheader("System Architecture & Processing Pipeline")

    st.markdown(
        """
        **Stage 1: Image Ingestion & Preprocessing**
        - Ingests user-submitted handwriting scans in `.jpg`, `.jpeg`, or `.png` formats.
        - Converts color channels to grayscale intensity: $I(x, y) = 0.299R + 0.587G + 0.114B$.
        - Applies a $3 \\times 3$ Gaussian Kernel to suppress high-frequency scanner speckle noise.

        **Stage 2: Adaptive Binarization & Morphological Cleaning**
        - Computes dynamic binarization via **Otsu’s Thresholding with Inverse Polarity**, separating ink foreground ($255$) from paper background ($0$).
        - Executes Morphological Opening using a $2 \\times 2$ structuring element to sever artificial pixel bridges and strip isolated noise pixels.

        **Stage 3: Connected Component Analysis (CCA)**
        - Evaluates 8-way pixel neighborhood connectivity to assign unique integer labels to discrete stroke regions.
        - Applies strict spatial filtering ($Area \\ge 20$, $Height \\ge 5$, $Width \\ge 2$) to isolate meaningful alphanumeric characters from dust and paper imperfections.

        **Stage 4: Biomechanical Feature Extraction**
        - Calculates 13 spatial, geometrical, and orientation features from segmented connected components.

        **Stage 5: Ensemble Machine Learning Classification**
        - The 13-dimensional feature vector is passed to a calibrated **Histogram Gradient Boosting Classifier** to output predicted class labels and class posterior probabilities.
        """
    )

    # -----------------------------------------------------
    # 13 BIOMECHANICAL FEATURES EXPLAINED
    # -----------------------------------------------------
    st.subheader("Handwriting Biomechanical Feature Formulations (13 Parameters)")

    biomech_features = {
        "1. Ink Density (%)": "Ratio of active ink pixels to the total canvas dimensions: $(N_{\\text{ink}} / (H \\times W)) \\times 100$.",
        "2. Connected Components": "Total count of isolated stroke regions identified during component labeling.",
        "3. Mean Letter Height": "Arithmetic mean of all detected bounding box heights: $\\mu_h = \\frac{1}{N}\\sum h_i$.",
        "4. Height Variation (%)": "Normalized standard deviation of character heights: $(\\sigma_h / (\\mu_h + 10^{-6})) \\times 100$. Measures sizing instability.",
        "5. Mean Letter Width": "Arithmetic mean of all detected bounding box widths: $\\mu_w = \\frac{1}{N}\\sum w_i$.",
        "6. Width Variation (%)": "Normalized standard deviation of character widths: $(\\sigma_w / (\\mu_w + 10^{-6})) \\times 100$.",
        "7. Baseline Drift": "Standard deviation of the vertical bottom coordinates ($y_i + h_i$) across all characters. Captures horizontal baseline instability.",
        "8. Mean Spacing": "Average horizontal gap between adjacent character bounding boxes ($x_{i+1} - (x_i + w_i)$).",
        "9. Spacing Variation (%)": "Normalized standard deviation of inter-character gaps: $(\\sigma_{\\text{gap}} / (\\mu_{\\text{gap}} + 10^{-6})) \\times 100$.",
        "10. Estimated Stroke Width": "Doubled mean value of the Euclidean Distance Transform ($L_2$) over the binary skeleton, estimating pen pressure.",
        "11. Slant Variation": "Standard deviation of character orientation angles derived from second-order central image moments: $\\theta = \\frac{1}{2}\\arctan\\left(\\frac{2\\mu_{11}}{\\mu_{20} - \\mu_{02}}\\right)$.",
        "12. Margin Alignment": "Standard deviation of the left-most character starting positions ($x_i$), assessing left margin adherence.",
        "13. Overlap / Collision Ratio (%)": "Percentage of consecutive components exhibiting horizontal coordinate collisions ($x_{i+1} < x_i + w_i$)."
    }

    for feat_name, feat_desc in biomech_features.items():
        st.markdown(f"**{feat_name}**")
        st.write(feat_desc)

    # -----------------------------------------------------
    # BENCHMARKS & EXPERIMENTAL VALIDATION
    # -----------------------------------------------------
    st.subheader("Model Performance & Experimental Benchmarks")

    st.write(
        """
        Our classifier was selected through a multi-model benchmark evaluated across 313 real-world handwriting samples
        using **Stratified 5-Fold Cross-Validation** and tested on an independent held-out test split (63 unseen samples).
        """
    )

    benchmark_data = pd.DataFrame(
        {
            "Evaluated Architecture": ["Default Random Forest", "GridSearch Tuned RF", "Tuned Extra Trees", "Hist Gradient Boosting (Final)"],
            "5-Fold CV Mean F1": ["0.8689", "0.8877", "0.8675", "0.8781"],
            "Test Accuracy": ["77.78%", "77.78%", "79.37%", "80.95%"],
            "ROC-AUC Score": ["0.8667", "0.8667", "0.9081", "0.8707"],
            "Dyslexic Recall": ["76.0%", "76.0%", "82.0%", "82.0%"]
        }
    )
    st.table(benchmark_data)

    st.write(
        """
        The finalized **Histogram Gradient Boosting** model achieved an overall **80.95% accuracy** with an **82.0% recall**
        for the dyslexic class, ensuring reliable sensitivity to minimize false-negative screening errors.
        """
    )

    # -----------------------------------------------------
    # TECHNOLOGIES USED
    # -----------------------------------------------------
    st.subheader("Technologies & Libraries Used")

    st.markdown(
        """
        - **Core Language:** Python 3.10+
        - **Web Application Framework:** Streamlit
        - **Image Processing & Computer Vision:** OpenCV (`cv2`), Pillow (PIL)
        - **Machine Learning & Modeling:** Scikit-Learn (HistGradientBoosting, RandomForest, ExtraTrees, GridSearchCV)
        - **Data Handling & Persistence:** NumPy, Pandas, Joblib
        """
    )

    # -----------------------------------------------------
    # LIMITATIONS & ETHICAL CONSIDERATIONS
    # -----------------------------------------------------
    st.subheader("Limitations & Ethical Considerations")

    st.write(
        """
        1. **Physical Artifact Dependencies:** Extracted metrics can be influenced by uneven lighting, writing instruments (fountain pen vs. pencil), paper textures, and camera skew.
        2. **Age & Motor Development:** Young children still developing fine motor skills may exhibit natural handwriting variations unrelated to dyslexia.
        3. **Ethical Guardrails:** This system is an automated triage tool. Decisions regarding educational interventions, accommodations, or clinical diagnoses must always involve qualified human educators and psychologists.
        """
    )

    st.subheader("Disclaimer")
    st.info(
        """
        This application is developed strictly for educational, experimental, and research demonstration purposes.
        The automated predictions generated by this software must not be used as a replacement for certified medical
        or educational dyslexia evaluations.
        """
    )