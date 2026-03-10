# Concrete Crack Segmentation

A U-Net based model for segmenting cracks in concrete images, with a Streamlit app for inference.

## Project Structure

```
concrete-crack-segmentation/
├── app/              # Streamlit app and inference pipeline
├── model/            # U-Net architecture, dataset, training, evaluation
├── data/             # Raw images/masks, processed data, train/val/test splits
├── notebooks/        # Exploratory notebooks
├── outputs/          # Predictions, logs, figures
├── weights/          # Saved model weights
├── config.yaml       # Hyperparameters and paths
└── requirements.txt  # Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python model/train.py
```

**Evaluate:**
```bash
python model/evaluate.py
```

**Run app:**
```bash
streamlit run app/streamlit_app.py
```
