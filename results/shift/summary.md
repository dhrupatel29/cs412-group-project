# Temporal shift (chronological split)

pdays uses 999 for 'never contacted', so its mean is not meaningful; KS still is.


## prevalence

| split | partition | rows | positives | pos_rate |
|---|---|---|---|---|
| random | train | 28831 | 3248 | 0.1127 |
| random | val | 6178 | 696 | 0.1127 |
| random | test | 6179 | 696 | 0.1126 |
| chronological | train | 28831 | 1606 | 0.0557 |
| chronological | val | 6178 | 658 | 0.1065 |
| chronological | test | 6179 | 2376 | 0.3845 |


## numeric_drift

| feature | train_mean | val_mean | test_mean | smd | ks_stat | ks_p | focus |
|---|---|---|---|---|---|---|---|
| euribor3m | 4.685 | 1.328 | 0.9485 | -7.144 | 1 | 0 | True |
| emp.var.rate | 1 | -1.8 | -2.322 | -4.389 | 0.9604 | 0 | True |
| nr.employed | 5209 | 5099 | 5040 | -4.421 | 0.9604 | 0 | True |
| cons.price.idx | 93.81 | 92.94 | 93.12 | -1.152 | 0.6568 | 0 | False |
| cons.conf.idx | -39.79 | -46.43 | -37.88 | 0.3586 | 0.4255 | 0 | True |
| previous | 0.0359 | 0.336 | 0.6495 | 0.9173 | 0.3962 | 0 | False |
| pdays | 997.2 | 973.6 | 789.4 | -0.721 | 0.2092 | 8.149e-196 | False |
| age | 40.18 | 37.85 | 41.48 | 0.1045 | 0.1321 | 7.648e-78 | False |
| campaign | 2.768 | 2.197 | 2.002 | -0.3118 | 0.1153 | 2.072e-59 | False |


## categorical_drift

| feature | tvd | largest_change | train_prop | test_prop | focus |
|---|---|---|---|---|---|
| poutcome | 0.3962 | nonexistent | 0.9652 | 0.569 | True |
| contact | 0.3528 | cellular | 0.5216 | 0.8744 | True |
| month | 0.2568 | jul | 0.2319 | 0.07914 | True |
| job | 0.2403 | blue-collar | 0.2326 | 0.08836 | False |
| default | 0.2163 | no | 0.7466 | 0.9629 | False |
| marital | 0.1344 | single | 0.2498 | 0.384 | False |
| education | 0.1291 | university.degree | 0.2894 | 0.3936 | False |
| day_of_week | 0.09501 | mon | 0.1949 | 0.2523 | False |
| housing | 0.0464 | no | 0.4678 | 0.4214 | False |
| loan | 0.006647 | yes | 0.1513 | 0.1447 | False |


## category_props

| feature | value | train | val | test |
|---|---|---|---|---|
| month | mar | 0.009781 | 0 | 0.04273 |
| month | apr | 0.02979 | 0.2588 | 0.02816 |
| month | may | 0.2693 | 0.7412 | 0.2309 |
| month | jun | 0.1517 | 0 | 0.1528 |
| month | jul | 0.2319 | 0 | 0.07914 |
| month | aug | 0.1795 | 0 | 0.1623 |
| month | sep | 0 | 0 | 0.09225 |
| month | oct | 0.002324 | 0 | 0.1054 |
| month | nov | 0.1254 | 0 | 0.07849 |
| month | dec | 0.0003468 | 0 | 0.02784 |
| poutcome | nonexistent | 0.9652 | 0.6827 | 0.569 |
| poutcome | failure | 0.03302 | 0.2927 | 0.2415 |
| poutcome | success | 0.001734 | 0.0246 | 0.1895 |
| contact | cellular | 0.5216 | 0.9231 | 0.8744 |
| contact | telephone | 0.4784 | 0.07689 | 0.1256 |
