# vignette-tabular-nn

Vignette on building and tuning a neural network for yes/no prediction from a table of data; created as a class project for PSTAT 197A in Fall 2026.

## Contributors 

1. Emma Nguyen - EDA / preprocessing
2. Praveen
3. Varun
4. Iris 


## Vignette abstract: a brief description in a few sentences of your vignette topic, example data, and outcomes.

This vignette teaches how a small neural network works, and how to build, train, and tune one to make a yes/no prediction from a table of data. The example predicts whether a job's openings will increase or decrease, using the Kaggle dataset "AI Impact on Job Market: Increasing vs Decreasing Jobs (2024–2030)" (30,000 rows, 13 columns). The dataset is synthetic, and the vignette says so.

Readers prepare the data, fit a logistic regression baseline, train a neural network, compare the two, spot overfitting, and avoid common mistakes such as data leakage. In particular, "Job Openings (2024)" and "Projected Openings (2030)" can reveal the answer, so the vignette shows why those columns should not be used as predictors.

## Repository contents: an explanation of the directory structure of the repository

## Reference list: 2 or more references to learn more about your topic.

Kaggle Source:

This is a synthetic dataset generated using realistic modeling, public job data patterns (U.S. BLS, OECD, McKinsey, WEF reports), and AI simulation to reflect plausible scenarios from 2024 to 2030. Ideal for educational, research, and AI project purposes.

License:
MIT