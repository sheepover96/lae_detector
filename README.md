# Subaru-HSC: LAE detector

## Requirements
- python3 or Docker

## Required dataset format
Dataset must be following CSV format.
```csv:dataset format
id, narrow_fits_data_path, g_fits_data_path
```

## How to use
### python3
1. `python3 -m venv venv`

1. `source venv/bin/activate`
1. `pip install -r requirements.txt`
1. `python3 main.py dataset_path`

### docker
under construction

## Output format
LAE detection result is saved in the fllowing format.

```csv:dataset format
id, narrow_fits_data_path, g_fits_data_path, narrow_band_prediction, g_band_prediction, LAE_or_Not(1 or 0), narrow_band_certainty, g_band_certainty
```
