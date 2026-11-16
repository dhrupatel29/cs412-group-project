# Data

We use the UCI Bank Marketing dataset (`bank-additional-full.csv`, 41,188 rows, `;`-separated). It's already in this folder, so there's nothing to download.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
(Moro, Cortez & Rita, 2014)

We drop the `duration` column because it's only known after the call ends, so keeping it would leak the answer.
