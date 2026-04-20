# Air Quality ML: Real-Time CO Prediction Dashboard

This project provides a Streamlit dashboard that compares multiple machine learning model predictions for CO concentration against live OpenAQ sensor measurements.

The app:
- Fetches live environmental data from OpenAQ
- Engineers the required model features in real time
- Scales inputs using a saved scaler
- Runs multiple trained models from saved artifacts
- Compares each model prediction against the current CO ground-truth proxy

## Project Structure

```
Air Quality ML/
|-- app.py
|-- ML_Project_Clg.ipynb
|-- README.md
|-- saved_model_artifacts/
|   |-- all_models.pkl
|   |-- features.pkl
|   |-- medians.pkl
|   |-- model_metrics.pkl
|   `-- scaler.pkl
`-- saved_model_artifacts.zip
```

## Requirements

Python 3.9+ is recommended.

Install required packages:

```bash
pip install streamlit joblib numpy pandas python-dotenv openaq
```

## Environment Variables

Create a `.env` file in the project root with your OpenAQ API key:

```env
OPENAQ_API=your_openaq_api_key_here
```

The app reads this key using `python-dotenv` and authenticates OpenAQ requests.

## Run the App

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## How It Works

1. Artifacts are loaded once with Streamlit caching from `saved_model_artifacts/`.
2. On button click, latest measurements are fetched for station ID `8118`.
3. Missing live values are backfilled with historical medians.
4. Additional features are engineered:
	 - Absolute humidity (AH)
	 - Time cyclic features (`Hour_sin`, `Hour_cos`)
	 - Weekend indicator
5. Feature vector is ordered using `features.pkl` and scaled via `scaler.pkl`.
6. All models from `all_models.pkl` predict CO.
7. Dashboard displays:
	 - Predicted CO by model
	 - Actual CO proxy (from live CO value conversion)
	 - Absolute error per model
	 - Line chart for prediction variance

## Notes

- CO comparison baseline is computed as:

	CO (mg/m^3) = CO (ppm) x 1.15

- OpenAQ responses can vary by station and sensor availability. The app handles missing readings using median fallback values.

## Troubleshooting

- If the app fails to fetch data, verify:
	- `.env` exists and contains a valid `OPENAQ_API`
	- Internet connectivity is available
	- OpenAQ service/API access is active
- If artifact-loading errors occur, ensure all `.pkl` files exist in `saved_model_artifacts/`.

## Future Improvements

- Add a `requirements.txt` for one-command environment setup
- Parameterize station selection in the UI
- Add logging and API retry handling
- Show historical trend comparisons, not only latest snapshot
