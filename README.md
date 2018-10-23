# Style Change Detection 1st Place Solution: Team SU-FMI ( Atanas, Didi, Daniel )

## Install dependencies
  * ```pip3 install -r requirements.txt```

## Optional dependencies
  * ```pip3 install pydot``` (keras model visualization)
  * ```apt install graphviz``` (keras model visualization)
  * ```pip3 install ipython``` (interactive python shell)
  * ```pip3 install jupyter```
  * ```pip3 install h5py``` (saving keras models to disk)
  * ```pip3 install autopep8``` (autoformat python code)
  * ```pip3 install textstat```

## External resources / Prerequisites
  * Add ```results.json``` to root directory
  * Add training/validation data to ```data/training``` / ```data/validation```
  * [Optional] Add external_feather file to ```data/feather```
  * Add pre-trained vectors to ```data/vectors```
  * Add file ```config.json``` to root with:
  ```{
    "persist_results": true,
    "results_file": "results/my_results_file.json",
    "n_jobs": 1
  }```

## Usage
  * ```python3 main.py```
# style-change-detection
