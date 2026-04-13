# Travel Tide - Rewards Program

In this project we use the Travel Tide bookings and sessions database to find interesing trends, draw insights, segment customers with rule based and machine learning approaches.

## Overview:

* **Goal:** Find relevant insights from the data, segment customers and assign perks to them.
* Devided in different notebooks, the ETL, EDA and Machine Learning preparation preceed the segmentation phases.
* The results shared in the PDF files under the folder `results`. The segmented users can be found in `data/assigned/users.csv`

## Structure:

```
├── data
│   ├── aggregated
│   │   ├── flights.csv
│   │   ├── hotels.csv
│   │   ├── sessions.csv
│   │   ├── users.csv
│   ├── assigned
│   │   ├── perks.csv
│   │   ├── users.csv
│   ├── segmented
│   │   ├── users_ml.csv
│   │   ├── users_rb.csv
├── notebooks
│   │   ├── EDA.ipynb
│   │   ├── ETL.ipynb
│   │   ├── ML_prep.ipynb
│   │   ├── ML_segmentation.ipynb
│   │   ├── Perk_assigment.ipynb
│   │   ├── RB_segmentation.ipynb
├── src
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── plots.py
│   │   ├── profile.py
│   │   ├── utils.py
├── README.md
├── setup.py

```

## Setup:

1. Install dependencies
2. Run notebooks in this order: ETL. EAD, ML_prep, ML_segmentation, RB_segmentation and Perk_assignment

## Notebooks description:

* **ETL:** data fetch, clean-up, inconsistency detection and type conversion.
* **EDA:** exploratory data analysis of tables.
* **ML_prep:** scaling and inputation for machine learning.
* **RB_segmentation:** rule-based segmentation
* **Perk_assigment:** perks assignment for segmented users