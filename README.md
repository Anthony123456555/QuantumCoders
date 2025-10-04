# Exoplanet Detector

Repo skeleton for the NASA Exoplanet Detector challenge.

Structure:
- data/raw: put your raw CSV/FITS here (3 CSVs you mentioned)
- src: source code (loaders, preprocess, models, api)
- notebooks: exploratory notebooks (placeholders)
- models: saved models will go here

How to run (prototype):
1. Copy your raw CSV files into `data/raw/`
2. Create a Python venv, install requirements: `pip install -r requirements.txt`
3. Run baseline training: `python src/models/train_baseline.py --input data/raw/catalog.csv`
4. Start API: `uvicorn src.api.app:app --reload --port 8000`

