# Disaster Tweet Classifier

A Flask web interface backed by the trained NLP pipeline in `models/disaster_tweet_classifier.joblib`. The app accepts one tweet at a time and returns the model label and its estimated disaster score.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. The model artifact must remain at `models/disaster_tweet_classifier.joblib`, or set `DISASTER_MODEL_PATH` to another location. The `/healthz` route reports whether the model loaded.

## Run the Streamlit app locally

```powershell
python -m streamlit run streamlit_app.py
```

The app uses the project-level `.streamlit/config.toml` theme and caches the trained model between reruns. Keep `models/disaster_tweet_classifier.joblib` alongside the app, or set `DISASTER_MODEL_PATH` to its deployed path.

## Deploy with Streamlit Community Cloud

1. Push this project to a GitHub repository with `streamlit_app.py`, `predictor.py`, `requirements.txt`, `.streamlit/config.toml`, and `models/disaster_tweet_classifier.joblib` included.
2. In Streamlit Community Cloud, create an app using that repository and select `streamlit_app.py` as the entrypoint.
3. In Advanced settings, select Python 3.13 to match the environment used to serialize the model, then deploy.

Community Cloud installs app dependencies from the repository's `requirements.txt`; the config file belongs at `.streamlit/config.toml`. No secrets file is needed for this app.

## Deploy to Heroku

The root `Procfile`, `requirements.txt`, and `.python-version` declare the web process, pinned Python dependencies, and Python runtime. Heroku's Python build process installs dependencies from `requirements.txt`; `.python-version` selects the runtime. The web process uses Gunicorn through the `Procfile`.

After installing and signing into the Heroku CLI, create an app in the Heroku dashboard or with the CLI, initialize/commit this project to Git if it is not already a repository, set the Heroku Git remote, and push the chosen branch. Keep the trained model file in the deployment source or configure `DISASTER_MODEL_PATH` for a securely managed model artifact.

## Project conclusion

The selected Logistic Regression classifier combines word and character TF-IDF with tweet structure and lightweight sentiment features. On the held-out test set it achieved 81.3% accuracy, 82.5% precision, 71.6% recall, and 76.7% F1. The supplied data has 7,613 labeled tweets, rather than the 10,000 stated in the brief. The Streamlit interface is ready for deployment configuration and can support human triage, but it is not an emergency alert service; predictions should be verified before action. The app has not yet been published to a cloud host.

## Notes

- The current model was trained on 7,613 labeled tweets; its result is an estimate and should not be treated as an emergency alert.
- Keep the installed scikit-learn version aligned with the version used to serialize the model.
- Use HTTPS and review the hosting provider's logging and retention settings before handling real user submissions.
