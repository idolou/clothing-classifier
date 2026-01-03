import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import streamlit as st
from rembg import remove
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_model(model_id):
    """
    Loads the ViT model and processor.
    Cached to avoid reloading on every interaction.
    """
    try:
        processor = ViTImageProcessor.from_pretrained(model_id)
        model = ViTForImageClassification.from_pretrained(model_id)

        # Patch for Wargon model missing labels
        if "wargon" in model_id.lower():
            wargon_labels = {
                0: "Blazer",
                1: "Blouse",
                2: "Cardigan",
                3: "Dress",
                4: "Hoodie",
                5: "Jacket",
                6: "Jeans",
                7: "Nightgown",
                8: "Outerwear",
                9: "Pajamas",
                10: "Rain jacket",
                11: "Rain trousers",
                12: "Robe",
                13: "Shirt",
                14: "Shorts",
                15: "Skirt",
                16: "Sweater",
                17: "T-shirt",
                18: "Tank top",
                19: "Tights",
                20: "Top",
                21: "Training top",
                22: "Trousers",
                23: "Tunic",
                24: "Vest",
                25: "Winter jacket",
                26: "Winter trousers",
            }
            model.config.id2label = wargon_labels
            model.config.label2id = {v: k for k, v in wargon_labels.items()}

        return processor, model
    except Exception as e:
        logger.error(f"Failed to load model {model_id}", exc_info=True)
        # Friendly error is handled by the caller or we can show a toast here
        return None, None


def classify_image(image, processor, model):
    """
    Classifies the input image using the loaded model.
    Returns: list of dicts with 'label' and 'score'.
    """
    # Ensure image is in RGB mode (remove alpha channel if present)
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probs, 5)

    results = []
    for prob, idx in zip(top5_prob, top5_indices):
        label = model.config.id2label[idx.item()]
        score = prob.item()
        results.append({"label": label, "score": score})

    return results


def remove_background(image):
    """
    Removes the background from the input image using rembg.
    Returns the processed image.
    """
    try:
        # rembg expects PIL Image
        output = remove(image)
        return output
    except Exception as e:
        logger.error("Error in remove_background", exc_info=True)
        raise e  # Re-raise to let the app handle how to show it, or we can just log here and ret invalid image
