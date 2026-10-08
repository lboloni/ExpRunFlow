# ExpRunFlow
ExpRunFlow is a small, opinionated library that supports a model to parameterize and run scientific experiment flows.

It is shared by the [BerryPicker](https://github.com/lboloni/BerryPicker) and [WaterberryFarms](https://github.com/lboloni/WaterberryFarms) projects.

* `exprunflow/exp_run_config.py`: the `Config` singleton and the `Experiment` (exp/run) objects. See [docs/DESIGN-ExpRun.md](docs/DESIGN-ExpRun.md).
* `exprunflow/flow.py`: the helpers of the flow notebooks, which run a queue of stage notebooks with Papermill and report on the results. See [docs/DESIGN-Flows.md](docs/DESIGN-Flows.md).
* `exprunflow/reproducibility.py`: seeding the RNGs of a run.
* `templates/`: samples of the main and machine-specific settings files.

## Installation

Check out the repository next to the project, and install it in the project's virtual environment:

```shell
pip install -e ../ExpRunFlow[flow]
```

The `flow` extra installs Papermill, tqdm and IPython, which the flow helpers need.

## Use in a project

A project has its code and notebooks in `src/` and its exp/run templates in `data/expruns/`. Its `src/exp_run_config.py` re-exports the library and sets the project-specific values:

```python
import pathlib
from exprunflow.exp_run_config import Config, Experiment

Config.PROJECTNAME = "MyProject"
Config.SRC_ROOT = pathlib.Path(__file__).resolve().parent
```

## Tests

```shell
python -m pytest tests
```
