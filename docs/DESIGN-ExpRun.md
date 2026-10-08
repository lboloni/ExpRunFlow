# Experiment/run design

## Objective

An experiment/run, abbreviated **exp/run** or **exprun**, is the complete,
human-readable configuration for one reproducible unit of research work, such
as training a model, verifying it, or comparing a set of alternatives. It
answers three questions without requiring someone to read Python code:

1. What configuration should be used?
2. Which notebook performs the work?
3. Where are the resulting data and provenance stored?

An exprun is configuration, not an implementation. Models and experiment code
remain in Python in the project, orchestration belongs to flows, and
generated data belongs in the result tree. Keeping these responsibilities
separate makes experiments easy to inspect, rerun, archive, and move between
machines.

## General principles

- **Configuration is data.** Checked-in YAML describes a run; it does not
  contain Python code.
- **Resolved expruns are complete.** Code accesses fields as `exp["field"]`
  and assumes required values are present and correctly formatted. Invalid
  research configuration should fail rather than be silently repaired.
- **Templates and results are different trees.** Editing a template must not
  modify old results, and generating results must not dirty the source tree.
- **Machine differences stay out of the templates.** Paths and hardware
  details live in the machine-specific settings, not in checked-in runs.
- **Notebook paths are metadata.** An exprun declares its entry points, while
  loading the exprun has no execution side effect.
- **Provenance travels with results.** The resolved configuration saved into
  a result directory records what produced that directory.

## Project layout and settings

The exp/run framework is implemented by `exprunflow/exp_run_config.py`. A
project using it has the layout

```text
<project>/
  src/               the code and the notebooks
  data/expruns/      the built-in exp/run templates
```

and a small `src/exp_run_config.py` that re-exports the library and sets the
project-specific values before the first use of `Config`:

```python
import pathlib
from exprunflow.exp_run_config import Config, Experiment

Config.PROJECTNAME = "BerryPicker"
Config.SRC_ROOT = pathlib.Path(__file__).resolve().parent
Config.KERNEL_NAME = "berrypicker"   # optional, the kernel flows run with
```

The project's code then uses `from exp_run_config import Config`.

`Config()` is a singleton. It reads
`~/.config/<PROJECTNAME>/mainsettings.yaml`, whose only field, `configpath`,
points to a machine-specific settings file (see `templates/`). The settings
used by the framework are:

- `experiment_data`: the root of the result tree;
- `experiment_system_dependent_dir`: the root of the system-dependent
  overlays (see below);
- `flows_path`: where flow workspaces are created.

A project may add its own settings; they are accessed as
`Config()["setting"]`.

`Config().runtime` holds values that are computed at startup and never saved,
most importantly `Config().runtime["device"]`, the torch device (cuda, mps or
cpu).

## Template organization

Built-in templates are organized as

```text
data/expruns/
  <family>/
    _defaults_<family>.yaml
    <run>.yaml
```

The defaults file contains values shared by a family. A run file contains the
values specific to one named run. An optional system-dependent overlay is
read from:

```text
<experiment_system_dependent_dir>/<family>/<run>_sysdep.yaml
```

`Config().get_experiment(experiment, run)` merges these in order: the family
default, then the run, then the overlay. Later values override earlier ones.
The overlay is appropriate for paths or machine characteristics (for example
a camera or USB port), not for changing the scientific meaning of a run.

Two helpers create templates in an external exp/run tree. They refuse to
work while the built-in tree is active:

- `copy_experiment(family, run=None)` copies a run, or the whole family, from
  the built-in tree.
- `create_exprun_variant(family, run, changes, new_run_name)` resolves a
  built-in run, applies `changes`, and writes the result as a new run.

Flows do not use `copy_experiment`; `setup_flow()` copies families from the
currently active tree (see [DESIGN-Flows.md](DESIGN-Flows.md)).

## Configuration and result paths

`Config` exposes separate APIs for the two trees:

- `get_exprun_path()` and `set_exprun_path()` select the configuration
  templates. By default this is the built-in `data/expruns`.
- `get_results_path()` and `set_results_path()` select the result tree. By
  default this is the machine's `experiment_data`.

Both setters accept strings, as papermill passes the paths to stage
notebooks as strings.

## Creating the result directory

A run result lives below:

```text
<results_path>/<experiment>/<run>/             (or .../<run>/<subrun>/)
```

`get_experiment(experiment, run, subrun=None, creation_style="exist-ok",
create_data_dir=True)` resolves the exprun, sets `exp["data_dir"]`, and
prepares the directory according to `creation_style`:

- `exist-ok` reuses the directory, creating it only when absent;
- `version` moves an existing directory to a timestamped backup and starts
  fresh;
- `discard-old` deletes an existing directory and starts fresh.

Any other value raises an exception. `exist-ok` means reuse the directory; it
does not mean that the computation is skipped. A notebook decides for itself
whether an existing result (for example a trained model) can be reused.

Resolving an exprun for inspection uses `create_data_dir=False`. This returns
the exprun, including its `data_dir`, without creating or changing anything.
Flows use it to read the notebook entry points of the runs they queue.

## Notebook entry points

Every resolved exprun has an `input-to-notebook` list. Each entry is the POSIX
path of a notebook relative to `src` (`Config.SRC_ROOT`):

```yaml
input-to-notebook:
  - sensorprocessing/Train_Conv_VAE.ipynb
  - sensorprocessing/Verify_Conv_VAE.ipynb
```

A notebook belongs in the list when the exprun is one of its primary inputs:
the notebook loads that experiment and run and produces, verifies, compares,
or displays its result. A notebook that only loads the exprun as a component
of another experiment is not listed. Flow notebooks are not entry points.

The list order follows the normal workflow: a training or data-production
notebook precedes a verification notebook. An empty list means that the
exprun is a supporting configuration without its own notebook.

The field is inherited like any other field. Put it in the family default
when every run has the same entry points. Families containing different kinds
of runs have an empty default and override it in each run. Flows select
notebooks only through this field.

## Stage notebook contract

A notebook listed in some `input-to-notebook` is a **stage notebook**. It can
be run directly or by a flow, with the same interface. It has exactly one
cell tagged `parameters`, defining at least:

```python
creation_style = "exist-ok"
expruns_path = None   # if not None, an external exp/run tree
results_path = None   # if not None, an external result tree
experiment = "sensorprocessing_conv_vae"
run = "sp_vae_128"
```

Alternative runs are listed as commented assignments. The next cell points
`Config` at `expruns_path` and `results_path` when they are given, and loads
the primary exprun with
`Config().get_experiment(experiment, run, creation_style=creation_style)`.
Other expruns loaded as components use the default `exist-ok`.

The last cell of the notebook calls `exp.done()` on the primary exprun (see
below).

## Result provenance and completion

The result directory contains an `exprun.yaml` copy of the resolved
configuration alongside the generated data. It is written when the directory
is created, and `time_started` is recorded at that point. Timers
(`exp.start_timer(name)` and `exp.end_timer(name)`) add their start, end, and
duration to it.

Calling `exp.done()` adds `time_done` and saves. It is called only after the
notebook has produced every result it declares. The marker therefore means
that the stage completed, not merely that its directory exists.

Known limitations:

- When two entry points share an exprun (a training and its verification),
  they share one `exprun.yaml`, so `time_done` does not tell them apart.
- Under `exist-ok`, a `time_done` from an earlier successful run remains. This
  is intended, since the result is reused.
- `done()` on an existing directory rewrites `exprun.yaml` from the current
  configuration, which drops the earlier `time_started` and timer fields.

## Reproducibility

`exprunflow.reproducibility.configure_deterministic_run(seed=777)` seeds the
Python, numpy and torch RNGs and lets cuDNN pick the fastest algorithms.
Deterministic algorithms are not enforced, as they cost speed on CUDA and are
not supported for all operations on MPS.

## Relationship to flows

An exprun describes one unit of work. A flow generates or selects several
such units, isolates their configuration and result trees, executes their
declared notebooks, and reports on the results. That orchestration contract
is defined in [DESIGN-Flows.md](DESIGN-Flows.md).

## Tests

`tests/test_exp_run_config.py` tests the configuration precedence, the
creation styles, resolution without a data directory, and the path API.
Each project checks its own templates and notebooks against the contract
above (for BerryPicker, `src/test/test_exprun_notebooks.py`).
