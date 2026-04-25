# NBA Postseason Swap Tool

This project is an interactive decision-support tool for exploring NBA roster swaps.

It uses a team-level logistic regression to estimate postseason probability, then applies that probability inside an expected value framework to compare same-position player replacements.

## What is included

- `app.py`: Streamlit app
- `_NBA.xlsx`: source data
- `nba_prescriptive_backend.py`: modeling and swap engine
- `Postseason_Model_Methodology.ipynb`: full standalone methodology notebook
- `Postseason_Model_For_Judges.ipynb`: judge-facing companion notebook

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app and point it to `app.py`.
3. Make sure `_NBA.xlsx` is committed to the repo as well.
4. Copy `share_links.example.json` to `share_links.json` and replace the placeholder URLs with your real GitHub/Colab links.

### Colab link format

For a notebook in GitHub, the browser-friendly Colab link looks like:

```text
https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/Postseason_Model_Methodology.ipynb
```

Use the same pattern for `Postseason_Model_For_Judges.ipynb`.

## Notes

- The app expects `_NBA.xlsx` to be present in the project root.
- The methodology notebook is self-contained and only needs `_NBA.xlsx` in the same folder.
