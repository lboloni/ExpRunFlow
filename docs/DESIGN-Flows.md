# Experiment-flow design

## Objective

A flow turns a set of expruns into a repeatable research procedure. It
prepares an isolated workspace, selects or generates the expruns it needs,
runs the existing stage notebooks in an explicit order, stops on failure, and
leaves behind both the scientific products and the executed notebooks needed
to understand what happened.

The goal is not to build a general workflow engine. A flow is a small,
readable notebook for a known procedure. It should make the complete
procedure easy to start, inspect, rerun, and audit without hiding research
decisions behind dynamic scheduling machinery.

## General principles

- **The procedure is explicit.** Queue order and membership come from short
  queue-building code in the flow notebook or the project.
- **Stage notebooks remain independently runnable.** A flow supplies the
  same parameters a user would supply directly; it does not create a second
  implementation of an experiment.
- **Each execution is isolated.** Copied and generated expruns, results, and
  executed notebooks share one flow workspace outside the source tree.
- **Inspection has no side effects.** Queue construction resolves expruns
  without creating their result directories.
- **Failures remain failures.** A failed stage stops the queue and is
  re-raised after the final report has recorded partial progress.
- **Completion is evidence-based.** A stage is complete only when its result
  provenance contains `time_done`.
- **The design stays deliberately small.** There is no DAG database,
  component registry, automatic implementation selection, or hidden retry.

The configuration and result conventions used below are defined in
[DESIGN-ExpRun.md](DESIGN-ExpRun.md). The helpers are in `exprunflow/flow.py`;
a project re-exports them from its own `src/flow.py` (or similar), next to
any project-specific queue builders.

## Isolated workspace

`setup_flow(flow_name, families)` creates the following workspace below the
machine-specific `flows_path`:

```text
<flows_path>/<flow_name>/
  expruns/              copied families and generated expruns
  results/              experimental data and models
  executed-notebooks/   the stage notebooks as executed by Papermill
```

It copies each listed family from the currently active exprun path, not from
a hard-coded built-in location, and then points `Config` to the workspace
with `set_exprun_path()` and `set_results_path()`. It returns
`(expruns_path, results_path, notebooks_path)`. Copying a family also copies
its default and thus its `input-to-notebook` value.

Each flow lists the families it needs explicitly: the families whose runs it
queues or generates, and the families its stages load as components.

Generated data and executed notebooks never belong in the source repository.

## Building the execution queue

A queue is an ordered list of flow entries. An entry is a dict with a display
`name`, the `experiment`, the `run`, the `notebook` and the `creation_style`;
a project may add further keys for its own bookkeeping.

`flow_entry(name, experiment, run, index, creation_style)` resolves the
exprun with `create_data_dir=False` and takes the notebook from
`exp["input-to-notebook"][index]`. The exprun is therefore the only place
where the notebook is chosen.

The expruns of a queue are either checked-in runs, selected for example from
the list of runs of a comparison exprun, or **generated** by the flow: a
generator function writes one exprun YAML into the workspace and returns the
flow entry for it:

```python
def generate_run(params, exp_name, run_name):
    val = {...}
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    return flow_entry("Train", exp_name, run_name, 0, creation_style)
```

The usual order is: produce the shared inputs, train or run each
alternative, verify or visualize, and run the comparisons. A flow may also
have several phases, building the next part of the queue from the results of
the previous one.

## Creation styles and data ownership

Every stage receives one of the three creation styles of
[DESIGN-ExpRun.md](DESIGN-ExpRun.md). The flow's `creation_style` parameter
applies to producer stages: trainings, runs, and comparisons.

The second entry point of the same exprun, such as a verification after a
training on one run, always receives `exist-ok`. It reopens the directory of
its producer; `discard-old` or `version` would remove or move the result it
is supposed to verify.

Trained models and simulation results are authoritative inputs;
verification figures and comparisons are reproducible derived products. A
comparison exprun owns its own result directory and must not modify the runs
it reads.

## Execution and failure behavior

`run_flow(entries, expruns_path, results_path, notebooks_path)` executes the
queue in order through `run_notebook()`. Each stage receives the same
Papermill parameters:

- `experiment`
- `run`
- `creation_style`
- `expruns_path`
- `results_path`

The notebook path is relative to `Config.SRC_ROOT`. The notebook runs with
the kernel `Config.KERNEL_NAME` (its own kernel when this is `None`) and its
own directory as the working directory. Its executed copy is written to
`executed-notebooks/<notebook>_<experiment>_<run>.ipynb`. One overall
progress bar shows the queue total, the current stage, and the number of
stages remaining; Papermill's cell-level progress remains local to the
current stage.

Exceptions are not suppressed. The top-level flow catches a stage exception
only long enough to execute its final report cell:

```python
flow_error = None
try:
    run_flow(entries, expruns_path, results_path, notebooks_path)
except Exception as error:
    flow_error = error
```

The report records partial progress, after which the original exception is
re-raised. Papermill therefore marks the top-level flow as failed, and the
executed notebook of the failed stage remains available as the diagnostic
artifact.

## Completion and final report

Every top-level flow notebook ends with a report cell:

```python
final_results_path = pathlib.Path(results_path, <comparison experiment>, <comparison run>)
report = display_flow_report(
    entries, results_path, notebooks_path, final_results_path,
    flow_error=flow_error,
    preview_files=sorted(final_results_path.glob("*.png")))
if flow_error is not None:
    raise flow_error
if not report["all-results-present"]:
    raise Exception("Flow finished without all expected stage results.")
```

`get_flow_report()` checks every queued stage at
`results/<experiment>/<run>/exprun.yaml`. A stage is complete when that file
contains `time_done`. For each stage it also reads the executed notebook in
`executed-notebooks/` (named by `executed_notebook_path()`, the same rule
`run_notebook()` uses): papermill records the run time in
`metadata.papermill.duration` and marks the failing cell with
`metadata.papermill.exception`, whose error output gives the stage error.
The report distinguishes complete, partial, and no-result executions. If
execution returns without an exception but a marker is missing, the final
cell raises; incomplete results cannot be presented as success.

`display_flow_report()` shows clickable absolute paths for the flow
workspace, the complete results tree, and the final-results directory
(normally the directory of the main comparison). A stage table lists every
queued stage with its status (complete, failed, incomplete, or not run), its
duration, a link to its executed notebook, a link to its result directory,
and, for a failed stage, the error; the total stage time follows the table.
The report lists every PDF below the final-results directory as a link, and
it embeds the previews passed as `preview_files`, each either a path or a
dict with `path` and `description`. The paths remain visible and copyable
even if a notebook frontend refuses to open a local `file:` link.

Comparison notebooks should save every figure as a publication PDF and a
same-basename PNG. The PDF is the linked product; the PNG is the preview,
because notebook frontends render PNGs inline reliably and PDFs not.

The checked-in flow notebook contains no saved outputs; the rendered report
belongs to its executed copy.

## Tests

`tests/test_flow.py` tests the workspace setup, `flow_entry`, the Papermill
parameters and defaults of `run_notebook`, ordered fail-fast execution, and
the report. The projects test their own queue builders and flow notebooks.
