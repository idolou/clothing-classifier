# Local Clothing Classifier 👕

A streamlined image classification application built with Streamlit and Hugging Face Transformers.

![App Preview](app_preview.png)

## Features

- **Local Inference**: Uses Vision Transformer (ViT) models to identify clothing items without sending data to the cloud.
- **Multiple Models**: Switch between `google/vit-base-patch16-224` (General) and `wargoninnovation/wargon-clothing-classifier` (Specialized).
- **Background Removal**: Integrated `rembg` tool to remove backgrounds for cleaner processing.
- **History**: locally tracks your session's uploaded images and results.
- **Simple UI**: Clean interface designed for efficiency.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Set up a virtual environment**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: This project requires `numpy<2.3` due to `numba` dependencies._

## Usage

1.  **Run the application**:

    ```bash
    streamlit run app.py
    ```

2.  **Using the App**:
    - **Classify**: Upload an image in the "Classify" tab. Optionally check "Remove Background" for better results on busy images.
    - **History**: View your past uploads and results in the "History" tab.
    - **Settings**: Use the sidebar to switch between different AI models.

## Project Structure

- `app.py`: Main Streamlit application entry point.
- `utils.py`: Utility functions for model loading, classification, and background removal.
- `style.css`: Custom CSS styling for the application.
- `requirements.txt`: Python package dependencies.

## Technologies

- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [Rembg](https://github.com/danielgatis/rembg)
