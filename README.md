# AircraftVerse Design Explorer

Interactive visualization project for exploring AircraftVerse, a simulated aircraft design dataset with 27,714 configurations.

The project focuses on how aircraft design choices relate to flight performance. The Streamlit dashboard helps users compare design families, identify viable aircraft configurations, and inspect trade-offs between maximum distance, hover time, speed, mass, and design complexity.

Original data source:

- AircraftVerse paper: <https://arxiv.org/abs/2306.05562>
- AircraftVerse dataset on Zenodo: <https://doi.org/10.5281/zenodo.6525446>

## Product

- Visualization product: Streamlit dashboard in `app.py`
- Data processing: Python pipeline in `aircraft_dashboard.py` and `data_acquisition/`
- Documentation: Quarto website in `docs/`
- Processed dataset: `data/aircraft_designs.csv.gz`

## Run The Dashboard

```bash
UV_CACHE_DIR=.uv-cache uv sync
UV_CACHE_DIR=.uv-cache uv run streamlit run app.py
```

The dashboard loads the compact processed dataset first. If `data/aircraft_designs.csv.gz` is missing, it can fall back to a local slim AircraftVerse extraction in `data_raw/aircraftverse_slim`.

## Rebuild The Processed Dataset

The full raw AircraftVerse downloads are large and are not committed. To rebuild the processed file:

1. Download the three AircraftVerse zip files from <https://doi.org/10.5281/zenodo.6525446>.
2. Create the slim JSON extraction:

```bash
python3 data_acquisition/prepare_slim_aircraftverse.py \
  --zip-files data_raw/AircraftVerse_1.zip data_raw/AircraftVerse_2.zip data_raw/AircraftVerse_3.zip \
  --target-dir data_raw/aircraftverse_slim
```

3. Build the compact dashboard table:

```bash
UV_CACHE_DIR=.uv-cache uv run python data_acquisition/build_processed_dataset.py \
  --source-dir data_raw/aircraftverse_slim \
  --output data/aircraft_designs.csv.gz
```

The expected output has 27,714 rows and 42 columns.

## Render Documentation

```bash
cd docs
UV_CACHE_DIR=../.uv-cache uv run quarto render
```

The rendered site is written to `docs/build`.

## Repository Notes

`data_raw/` is intentionally ignored because it contains large local downloads and extracted raw files. The committed `.csv.gz` file is the reproducible processed dataset used by the app and deployment.
