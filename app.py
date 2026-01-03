import streamlit as st
from PIL import Image
from utils import load_model, classify_image, remove_background
import base64
from io import BytesIO
import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Local Clothing Classifier", page_icon="👕", layout="centered"
)

# Initialize Session State for History
if "history" not in st.session_state:
    st.session_state["history"] = []

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Application Header
st.title("Local Clothing Classifier")
st.markdown("### Run Hugging Face models locally")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    model_options = {
        "Google ViT Base (General)": "google/vit-base-patch16-224",
        "Wargon Clothing Classifier": "wargoninnovation/wargon-clothing-classifier",
    }

    selected_model_name = st.selectbox(
        "Choose Model", list(model_options.keys()), index=0
    )

    selected_model_id = model_options[selected_model_name]

    st.info(
        f"**Current Model**:\n{selected_model_id}\n\n**Backend**: Hugging Face + PyTorch"
    )

# Modern Tab Navigation
tab_classify, tab_history = st.tabs(["Classify", "History"])

# --- TAB 1: CLASSIFY ---
with tab_classify:
    st.markdown("Upload an image to identify clothing items with precision.")

    uploaded_file = st.file_uploader(
        "Drop your image here", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            # Display image
            st.image(image, caption="Uploaded Image", width="stretch")

            # Background Removal Option
            remove_bg = st.checkbox(
                "Remove Background for better accuracy", value=False
            )

            image_to_classify = image
            if remove_bg:
                with st.spinner("Removing background..."):
                    image_to_classify = remove_background(image)
                    st.image(
                        image_to_classify,
                        caption="Processed Image",
                        width="stretch",
                    )

            with st.spinner(f"Analyzing with {selected_model_name}..."):
                # Load model and classify
                processor, model = load_model(selected_model_id)

                if processor and model:
                    predictions = classify_image(image_to_classify, processor, model)

                    # Create history item
                    history_item = {
                        "id": len(st.session_state["history"]),
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "filename": (
                            uploaded_file.name
                            if hasattr(uploaded_file, "name")
                            else "unknown.jpg"
                        ),
                        "model": selected_model_name,
                        "top_prediction": predictions[0],
                        "all_predictions": predictions,
                        "image": image.copy(),
                    }

                    # Prevent duplicates
                    if (
                        not st.session_state["history"]
                        or st.session_state["history"][-1]["filename"]
                        != history_item["filename"]
                    ):
                        st.session_state["history"].append(history_item)

                    st.markdown("### Analysis Results")

                    # Display results with custom styling
                    for pred in predictions:
                        score_pct = pred["score"] * 100
                        st.markdown(
                            f"""
                        <div class="result-card">
                            <div style="flex-grow: 1;">
                                <span style="font-weight: 600; font-size: 1.1rem;">{pred['label'].title()}</span>
                                <div class="result-progress">
                                    <div class="progress-bar" style="width: {score_pct}%;"></div>
                                </div>
                            </div>
                            <span style="font-weight: 800; font-size: 1.2rem; margin-left: 15px; color: #4ade80;">
                                {score_pct:.1f}%
                            </span>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning(
                        "⚠️ Model failed to load. Please try selecting a different model from the sidebar."
                    )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            st.error(
                "We ran into an issue analyzing your image. Please try another image or model."
            )
            with st.expander("Details (for developers)"):
                st.code(str(e))

# --- TAB 2: HISTORY ---
with tab_history:
    st.header("Upload History")

    if not st.session_state["history"]:
        st.info("No items in history yet. Go to 'Classify' to upload images.")
    else:
        # Clear history button
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()

        # Sort by latest first
        for item in reversed(st.session_state["history"]):
            with st.container():
                st.markdown(
                    f"**{item['filename']}** <span style='color:grey; font-size:0.8em'>({item['timestamp']})</span>",
                    unsafe_allow_html=True,
                )
                h_col1, h_col2 = st.columns([1, 2])

                with h_col1:
                    st.image(item["image"], width="stretch")

                with h_col2:
                    st.markdown(f"**Model**: {item['model']}")
                    top_label = item["top_prediction"]["label"].title()
                    top_score = item["top_prediction"]["score"] * 100
                    st.markdown(
                        f"**Result**: <span style='color:#4ade80; font-weight:bold'>{top_label} ({top_score:.1f}%)</span>",
                        unsafe_allow_html=True,
                    )

                    with st.expander("Full Details"):
                        for pred in item["all_predictions"]:
                            st.write(f"{pred['label']}: {pred['score']*100:.1f}%")

                st.divider()
