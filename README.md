# Neural Network-Based Rule Models With Truth Tables

This is the github for the submitted paper at ECAI 2023. The packages used to evaluate with a venv are under `./requirements.txt`
file.

```bash
pip3 install -r requirements.txt
```

You will find there a python file for evaluating an example model on the following datasets:

- california housing
- adult
- compas
- heloc
- diabetes

Each model with their associated truth tables and weights can be found in the `./models` directory, under each dataset name.

## Evaluation

You can evaluate our model with the following command:

```bash
python3 infer_truth_tables.py
```

To choose the dataset, you only have to change the `./config/default.yaml` file and write the name of the dataset you want to evaluate on.

/!\ california housing is refered by the word `house`. The file would then be `dataset: house`


## Don't care terms injection and complexity reduction

The file to reduce the number of terms with dont care terms injection and to name the rules is `./name_rules.py`. *We provided it for the adult, compas and diabetes dataset*.

```bash
python3 name_rules.py
```

It will save the expressions in dnf form in the corresponding dataset folder in `./models/{dataset}/human_expressions` folder and their associated truth tables.
It will also compute the correlation between each filter and save it in the same folder.


You can then verify that the metrics are the same than without dont care terms with the file `./infer_truth_tables_dontcares.py` :

```bash
python3 infer_truth_tables_dontcares.py
```

And that using correlated filters deacreases the metrics with the file `./infer_truth_tables_correlation.py` :

```bash
python3 infer_truth_tables_correlation.py
```