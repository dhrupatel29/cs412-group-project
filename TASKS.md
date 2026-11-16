# Who's doing what

We each commit our own work from our own GitHub accounts.

**Dhru** set up the repo and the data pipeline, reproduced the pilot, and tuned Logistic Regression and XGBoost. He's running the final experiments now and will write the setup, baselines and results parts of the report.

**Pranay** built the FT-Transformer and the runtime script, and drafted the intro and related work. He still needs to push his tuned config, make the final runtime table and move his sections into the report.

**Vaibhav** built and tuned the MLP, did the shift analysis and wrote the summary script. He still needs to make the calibration plots, update the tables with the final results and write the data and MLP sections.

**Together** we'll write the abstract, conclusion and contributions, then put the slides together for the presentation.

## How we ran things

Every model got 20 tuning trials on the chronological split, picked by validation AP. The random split always uses seed 42, and missing values are filled in the same way for every model.

## Plan

Final runs, plots and runtime table by Nov 20. Slides done by Nov 22, and we present on Nov 23. Report finished by Dec 1.