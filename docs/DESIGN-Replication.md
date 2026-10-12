# Replication design

**Status:** implemented in ExpRunFlow 0.2.0, in four modules:
- `exprunflow/replication.py`;
- `exprunflow/metrics.py`;
- `exprunflow/aggregate.py`;
- `exprunflow/plots.py`.

There is also a template, `templates/Aggregate.ipynb`. The tests are in `tests/test_replication.py`. WaterberryFarms uses it for the MAREX comparisons (`WaterberryFarms/src/papers/y2027_marex/DESIGN-MultiSeedEvaluation.md`).

The implementation differs from the proposal below in four points:
- **A separate metrics module:** `save_metrics` and `load_metrics` are in `exprunflow/metrics.py`, which needs no optional dependency, so a run stage can write its metrics without the `analysis` extra.
- **Labels in the replication block:** the block also records the labels of the base run. `save_metrics` takes `label_keys` for runs that are not replications.
- **Machine information:** every `metrics.json` records the machine it ran on (`host`), and the aggregation writes `hosts.csv`.
- **Measurements that may differ:** `check_reproducible` takes `ignore`, the measurements that legitimately differ between runs, such as computation times.

## Objective

An experiment whose outcome depends on random choices must be run several times, as **independent replications**: the same configuration, with only the random seeds varied. Its results are then reported as means with confidence intervals and ranges, not as single numbers.

The library should make this procedure as explicit and repeatable as a flow:
- declare in an exp/run which base runs are replicated, and with which seeds;
- generate one exprun per replication into the flow workspace, and queue them;
- have every replication record its results in a standard, plain-data form;
- aggregate the replications into tidy tables with their statistics;
- draw the standard figures of replicated results from those tables.

What a seed *means* is project-specific: a map seed in WaterberryFarms changes an environment exprun, a behavior seed changes the seed of every robot, and a training seed in BerryPicker would seed the RNGs of a training. The library defines the mechanism, and the project supplies these meanings as small functions.

The goal is not a general design-of-experiments engine. It is a small extension of the "generated expruns" pattern of [DESIGN-Flows.md](DESIGN-Flows.md), with the statistics that replications need.

## General principles

- **Replications are ordinary expruns.** A replication is a run variant written into the flow workspace. It can be inspected, rerun and opened by its name like any other run.
- **The declaration is data.** The base runs and the seed values are listed in one exp/run, the *aggregate*. The flow that generates the replications and the notebook that aggregates them read the same declaration, so they cannot disagree.
- **Names are determined by the seeds.** `clustered-marex-m3-b2` is the same replication in every flow that declares it. Seed lists can grow without renaming or rerunning the existing replications.
- **Seeds are derived, never shared by accident.** Per-entity seeds (for each robot, for each training) are derived from the replication seed with `numpy.random.SeedSequence`, not by adding offsets.
- **Common random numbers.** In a replication, every compared approach receives the same seeds, so the approaches can be compared replication by replication.
- **A replication is a function of its seeds.** The library cannot enforce this, but it provides a check (Section 8), and the documentation makes it a stated requirement of a replicated stage.
- **Aggregation reads plain data only.** Replications are aggregated from their `metrics.json`, never from pickles of project objects.
- **Tidy tables are the single interface to figures.** All figures and paper tables are drawn from the aggregated CSV tables.
- **The existing API does not change.** Everything here is new modules and optional extras (Section 9).

## Terminology

| Term | Meaning |
|---|---|
| base run | a checked-in run of the replicated family, e.g. `marex2027-run/clustered-marex` |
| factor | a named source of randomness that is varied, e.g. `map-seed`, `behavior-seed`, `training-seed` |
| factor value | one seed of a factor |
| replication | one base run with one value of every factor |
| applier | a project function that turns a factor value into the changes of the replication's exprun |
| labels | the project's classification of a base run, e.g. `scenario: clustered`, `approach: marex` |
| cell | the replications of one base run, i.e. one combination of labels; statistics are computed per cell |
| cluster factor | a factor whose replications are not independent of each other, e.g. the behavior seeds that share one map |

## 1. Declaration

The **aggregate exp/run**, in a family of the project (e.g. `marex2027-aggregate`), declares the replications:

```yaml
# data/expruns/marex2027-aggregate/icc-2027-replicated.yaml
source-experiment: marex2027-run      # the family of the base runs
runs:                                # the base runs to replicate
  - clustered-marex
  - clustered-mrse
  - unclustered-marex
factors:                             # crossed: every value of every factor
  map-seed: {code: m, values: [1, 2, 3]}
  behavior-seed: {code: b, values: [1, 2, 3]}
cluster-factor: map-seed             # optional, Section 5.2
labels: [scenario, approach]         # the keys of the base runs that identify a cell
paired-by: approach                  # optional: the label compared pairwise (Section 5.1)
input-to-notebook:
  - ../ExpRunFlow/templates/Aggregate.ipynb   # or a project copy
```

- **Crossed factors:** the factors are crossed, so with `|M|` map seeds and `|B|` behavior seeds every base run has `|M| x |B|` replications. A one-factor design lists one factor.
- **The code** of a factor is the letter used in the replication name.
- **Labels** are ordinary keys of the base runs. The library copies them into the metrics, so that no code parses run names.

The **flow exp/run** refers to the aggregate, as in WaterberryFarms today:

```yaml
run-experiment: marex2027-run
aggregate-experiment: marex2027-aggregate
aggregate: icc-2027-replicated
figure-experiment: marex2027-figure
figures: [replicated-comparison, replicated-ranges, replicated-over-time, replicated-paired]
```

## 2. Generating the replications (`exprunflow/replication.py`)

### 2.1 Functions

```python
derive_seed(base, *keys) -> int
    # int(np.random.SeedSequence([base, *keys]).generate_state(1)[0])

replication_name(base_run, factor_values, codes) -> str
    # "clustered-marex-m3-b2": the base name, then -<code><value> in the declaration order

replication_names(aggregate) -> list[str]
    # every base run with every combination of factor values, in a fixed order

write_exprun_variant(family, run, changes, new_run, config=None) -> str
    # writes the run's own YAML file with the top-level changes applied, as a new run of the
    # family in the active exprun path; refuses the built-in tree (like create_exprun_variant)

expand_replications(aggregate, appliers, config=None) -> list[Replication]
    # writes every replication (and the variants its appliers need) and returns their records

replication_entries(collection, appliers, creation_style, config=None) -> list[dict]
    # the flow entries: the dependency variants (e.g. precomputed environments), the replications,
    # the aggregation, and the figures, in this order
```

### 2.2 `write_exprun_variant` versus `create_exprun_variant`

The two functions write different things:
- **`create_exprun_variant`** (existing, unchanged) resolves the run and dumps all its merged values, family defaults included, under a timestamped name. The variant is a frozen snapshot.
- **`write_exprun_variant`** (new) copies the run's own file with the changes. The variant keeps inheriting the defaults of its family, which a replication needs, since the defaults of the flow workspace apply to it like to its base run. It writes a header comment naming its origin, and takes an explicit name.

### 2.3 Appliers

An applier is a project function, registered per factor:

```python
def applier(context: ApplyContext, value) -> dict:
    """The changes to the replication's exprun for one factor value"""
```

`ApplyContext` gives the applier:
- `base`: the resolved base run;
- `config`: the `Config`;
- `variant(family, run, changes, name)`: writes a dependency variant once per distinct name, and records it so that `replication_entries` queues it before the replications.

The two appliers of WaterberryFarms become:

```python
def apply_map_seed(context, m):
    env_family, env_run = context.base["exp_environment"], context.base["run_environment"]
    name = context.variant(env_family, env_run, {"tylcv-generated-seed": m}, f"{env_run}-m{m}",
                           queue=True)   # precomputed before the replications
    return {"run_environment": name}

def apply_behavior_seed(context, b):
    return {"robots": [with_seed(robot, derive_seed(b, i)) if has_seed(robot) else robot
                       for i, robot in enumerate(context.base["robots"])]}
```

A BerryPicker training would need only:

```python
def apply_training_seed(context, s):
    return {"seed": s}   # the training notebook calls configure_deterministic_run(exp["seed"])
```

### 2.4 The replication block

Every replication carries a block that identifies it, instead of the project-specific top-level keys that WaterberryFarms uses today (`base-run`, `behavior-seed`):

```yaml
replication:
  base-run: clustered-marex
  factors: {map-seed: 3, behavior-seed: 2}
```

A base run that is not a replication has no block. The metrics helper (Section 3) records its factor values as `null`.

### 2.5 Resuming: reusing completed replications

A long replicated flow may be interrupted, by a failure or by the user. As every replication is a function of its exprun (Section 8), the completed replications can be reused rather than recomputed. With `skip-completed: true` in the collection, set e.g. by a flow notebook parameter, `replication_entries` leaves out the replications and dependency variants that `is_completed(family, run)` accepts:
- the result's saved `exprun.yaml` records `time_done`, the completion evidence of [DESIGN-Flows.md](DESIGN-Flows.md);
- every value of the exp/run's current configuration equals the saved one. Execution-specific values (`data_dir`, the times, the timers) are excluded.

The aggregation and the figures always run on all the replications.

**Not detected:** changes of the code, and of other exp/runs that a replication reads. Reuse is therefore opt-in, for resuming the same experiment, and a full flow recomputes everything by default.

### 2.6 Partial experiments, continued to more replications

A partial experiment is a quick first pass, e.g. 3 replications per base run, to see the results and the figures in the paper. It is later continued to the full experiment, e.g. 20 replications, without repeating the runs it already has.

- **The count:** `replications: N` in the aggregate exp/run runs only the first N factor combinations of every base run. Without it, every combination runs.
- **The spread order:** with a count, `factor_combinations` lists the combinations with the **first factor varying fastest**. A prefix therefore covers as many values of the first factor as possible. With the factors `map-seed` (5 values) and `behavior-seed` (4 values), 3 replications are 3 different maps with the first behavior seed, rather than 3 behaviors on one map. Declare the factor whose spread matters most (usually the cluster factor) first.
- **Continuing:** a larger count extends a smaller one, i.e. its first N combinations are the same. The replication names depend only on the factor values, so the replications of the partial experiment are the first ones of the larger one.
- **Setting the count from a flow:** `replications: N` in the collection (e.g. from a flow notebook parameter) overrides the aggregate's count. `replication_entries` writes it into the aggregate in the flow workspace with `set_replications(family, run, count)`. The aggregation stage therefore loads exactly the replications that ran. The workspace's copy of the aggregate records the count of the last flow.
- **The continuation command:** the same flow with a larger `replications` and `skip-completed: true`. Only the new replications and their new dependency variants run (Section 2.5); the aggregation and the figures run again on all of them.
- **Other entries:** `without_completed(entries)` applies the same completion test to any flow entries, e.g. the single runs of a project's flow.

## 3. The metrics of a run (`exprunflow/aggregate.py`)

A replicated stage writes `metrics.json` into its result directory:

```python
save_metrics(exp, scalars, series=None, entities=None) -> path
load_metrics(data_dir) -> dict
```

`save_metrics` fills in the identification itself, from the exprun: the run name, the replication block, and the labels of the aggregate. The stage only supplies its measurements:

```json
{
  "run": "clustered-marex-m3-b2",
  "base-run": "clustered-marex",
  "factors": {"map-seed": 3, "behavior-seed": 2},
  "labels": {"scenario": "clustered", "approach": "marex"},
  "scalars": {"voi-absolute": 21099.0, "bytes-transmitted": 132056},
  "entities": {"con-1": {"group": "contractor", "voi-absolute": 8000.0}},
  "series": {"timestep": [9, 19, "..."], "voi-absolute": ["..."]}
}
```

- **`scalars`:** the final values of the run.
- **`entities`:** values per entity, such as robots or trainings. The `group` of an entity (e.g. the robot's role) is what is compared across replications, since entity names may not be comparable.
- **`series`:** sampled on a shared index, named by its first key (`timestep`). Replications of a cell must share it, so that the series can be averaged pointwise.

## 4. Tidy tables

`aggregate_tables(metrics, aggregate)` returns these tables, and `save_tables` / `load_tables` write and read them as CSV in the result directory of the aggregate exp/run. All are long ("tidy") tables, one observation per row. The key columns are the labels, then the factors, then `run`.

| Table | One row per | Columns besides the keys |
|---|---|---|
| `replications.csv` | replication and scalar | `metric`, `value` |
| `entities.csv` | replication, entity and metric | `entity`, `group`, `metric`, `value` |
| `series.csv` | replication, series metric and index | `metric`, `<index>`, `value` |
| `summary.csv` | cell and metric | the statistics of Section 5 |
| `summary-series.csv` | cell, metric and index | the statistics of Section 5 |
| `paired.csv` | the other labels, metric and pair of `paired-by` values | the statistics of the differences, `fraction_greater`, `fraction_less`, `p_ttest`, `p_wilcoxon` |
| `variance.csv` | cell and metric | the variance component of every factor (Section 5.3) |
| `hosts.csv` | machine and Python version | the replications run there: `processor`, `cpus`, `platform`, `python`, `replications` |

The factor and label column names are the declared names, with dashes, e.g. `map-seed`. WaterberryFarms currently writes `map_seed`, `behavior_seed`, `scenario` and `approach` as fixed columns (Section 9).

## 5. Statistics

### 5.1 Per cell

`describe(values)` gives:
- `n`;
- `mean`, `std`, `sem`;
- `ci95_low`, `ci95_high`: the Student t interval of the mean, NaN for `n = 1`;
- `min`, `q25`, `median`, `q75`, `max`.

`summarize(table, keys, cluster_factor)` adds the same interval computed over the **per-cluster means** (`n_clusters`, `ci95_low_clusters`, `ci95_high_clusters`). `confidence_interval(row)` returns the interval the figures show: the per-cluster interval if there are several clusters, otherwise the per-replication interval.

**Paired differences:** if the aggregate declares `paired-by`, `paired_differences` matches the replications of two values of that label (e.g. two approaches) on all their factor values, and describes the differences. This is meaningful only with common random numbers, which the factor appliers provide by construction.

### 5.2 The cluster factor

Replications that share a value of the cluster factor are not independent. For example, several behavior seeds on one map all share that map. An interval over all replications then overstates the evidence. The per-cluster interval is the conservative alternative, with `n` the number of clusters, and the figures use it when it exists.

### 5.3 Variance components

For a crossed design with several values of each factor, `variance_components` gives, per cell and metric:
- the variance between the cluster factor's values (of the per-cluster means);
- the mean within-cluster variance.

They show which factor's seeds would narrow the interval the most. For more than two factors, the components are computed per factor against the rest, which is approximate. A full random-effects model is out of scope.

## 6. Figures (`exprunflow/plots.py`)

The generic figures of replicated results, all drawn from the tidy tables. Each takes the tables, the metric(s), the label on the x axis, the label distinguished by color, the display names, the colors, and the output file. They are saved as PDF and PNG (the convention of the projects' figure helpers).

| Function | Shows |
|---|---|
| `plot_bars` | one panel per metric, grouped bars: mean, CI error bars, the replications as dots |
| `plot_ranges` | the box-and-whisker form of `plot_bars`: the box is the CI of the mean with a line at the mean, the whiskers the range, the dots the replications; box and whiskers are drawn independently |
| `plot_series` | one panel per value of a label, a mean curve per value of the color label, with a CI band and a min–max band |
| `plot_paired` | the paired differences with their CI and the zero line, with the fraction of replications won, one panel per value of a label |

**Shared axes:** within a figure, the values that are compared share their axis:
- the bars and boxes of a metric are in one panel;
- the panels of a series figure share the y axis, and those of a paired figure share the x axis.

Comparisons are therefore made within a figure, never across independently scaled figures.

Project-specific figures stay in the project. In WaterberryFarms these are the communication cost per message type and the VoI per robot role, both drawn from the same tables.

## 7. Notebooks

- **`templates/Aggregate.ipynb`:** the aggregation stage. It resolves the aggregate exp/run, loads the metrics of `replication_names(aggregate)`, writes the tables, displays the summary and the paired differences, and marks the exp/run done. Its parameters are the standard ones of a stage notebook ([DESIGN-ExpRun.md](DESIGN-ExpRun.md)). A project uses it directly or copies it.
- **Replicated figures:** a project notebook, as today in WaterberryFarms (`MAREX-Visualize-Replicated.ipynb`). The `figure-kind` of the figure exp/run selects a plot function of Section 6 or of the project.
- **Flow notebooks** call `replication_entries` when the collection has an `aggregate`, and otherwise build their queue as before.

## 8. Reproducibility

A replication must be a function of its exprun. Two kinds of hidden randomness break this:
- **global RNG state:** handled by `configure_deterministic_run` for the global generators, or by passing explicit generators;
- **wall-clock limits:** an anytime search that stops after a time limit returns results that depend on the machine and its load. WaterberryFarms found one in the MAREX planner, and replaced the time limit with a count of evaluations.

The library offers a check:

```python
check_reproducible(family, run, run_stage, compare=("scalars", "series", "entities"), ignore=())
    -> (bool, first metrics, second metrics)
    # runs the stage twice, each into a fresh temporary results path, and compares the chosen parts of
    # metrics.json, leaving out the measurements in ignore, which legitimately differ between runs
    # (computation times)
```

A project calls it in a test for one replication per replicated stage. Replications run sequentially in a flow. Running them in parallel is safe only once the check holds, and is not part of this design.

## 9. Compatibility

### 9.1 The library: backward compatible

The proposal only adds:
- `exprunflow/replication.py`: no new dependencies (yaml and numpy are already required);
- `exprunflow/aggregate.py`: needs pandas and scipy, in a new optional extra `analysis`;
- `exprunflow/plots.py`: needs matplotlib, in a new optional extra `plots`;
- `templates/Aggregate.ipynb` and this document.

**Unchanged:**
- `Config`, `Experiment`, `get_experiment` and its creation styles;
- `copy_experiment` and `create_exprun_variant`, with their current semantics. BerryPicker notebooks call `create_exprun_variant` (`Train_ProprioTuned_CNN.ipynb`, `Flow_PtunVsRandProj_128.ipynb`) and rely on its resolved, timestamped output;
- the flow helpers of `flow.py`, and `configure_deterministic_run`.

**Dependencies:** a plain `pip install -e ../ExpRunFlow` installs exactly what it installs today, and so does the `flow` extra. The new modules import pandas, scipy and matplotlib only when used, and say which extra is missing when they are not installed.

**Version:** 0.1.0 becomes 0.2.0, a minor version for an additive change.

### 9.2 WaterberryFarms: a one-time migration

Moving its replication code onto the library changes the project, not the library's other users:

| Today (`y2027_marex`) | With the library |
|---|---|
| `robot_seed`, `write_exprun_variant`, `replication_name(s)`, `build_marex_2027_replicated_entries` in `src/wbf_flow.py` | `derive_seed`, `write_exprun_variant`, `replication_names`, `replication_entries`, plus two appliers in the project |
| `map-seeds`, `behavior-seeds` in the aggregate exp/run | a `factors` section with codes, plus `labels` and `cluster-factor` |
| top-level `base-run`, `behavior-seed` in the replications | the `replication` block |
| `scenario`, `approach`, `map-seed`, `behavior-seed` at the top of `metrics.json` | `labels` and `factors` |
| `per-robot` with `role` | `entities` with `group` |
| columns `map_seed`, `behavior_seed`, `robots.csv` with `robot`, `role`; `n_maps`, `ci95_*_maps` | columns `map-seed`, `behavior-seed`, `entities.csv` with `entity`, `group`; `n_clusters`, `ci95_*_clusters` |
| `marex_aggregate.py` | removed: the library's `aggregate.py` |
| comparison, ranges, series and paired figures in `marex_graphics.py` | the library's `plots.py`; the communication and per-role figures stay |

**The migration:**
1. Rerun the replicated flow, rather than converting the existing results. The runs are deterministic, and the current 18 replications take about 3 minutes.
2. Adjust the paper's `scripts/sync_full_results.py` to the new column names.
3. Adjust `test_replications.py` to the new formats. Its tests of the statistics move to the library's tests, with the synthetic data set.

Until the migration, WaterberryFarms keeps working unchanged against ExpRunFlow 0.2, since it uses none of the new modules.

## 10. Tests (in the library)

- **Seeds and names:** `derive_seed` is stable and distinct across keys; `replication_names` gives the declared order and the names of Section 1.
- **Variants:** `write_exprun_variant` writes the changes and keeps the family defaults (the variant resolves with them), and refuses the built-in tree.
- **Expansion:** `expand_replications` and `replication_entries` on a toy family with two factors, one of which writes a queued dependency variant:
  - the order of the entries;
  - the replication blocks;
  - that every dependency variant is written once.
- **Metrics:** `save_metrics` fills in the identification from the exprun, for a replication and for a plain run.
- **Statistics:** on a synthetic set (two approaches, two clusters, three replications per cluster, values `100*cluster + 10*replication + offset`):
  - the means and t intervals against scipy;
  - the per-cluster interval;
  - a paired difference that is exactly the offset;
  - the two variance components;
  - NaN intervals for `n = 1`.
- **Plots:** every plot function writes a PDF and a PNG from the synthetic tables.
- **Reproducibility:** `check_reproducible` detects a stage that uses an unseeded generator.
- **Compatibility:** the existing tests of `Config`, `create_exprun_variant` and the flow helpers pass unchanged, and importing `exprunflow.replication` succeeds without pandas installed.

## 11. Open questions

1. **Factorial sweeps:** should `factors` also hold non-seed parameters, such as the number of robots or the budget? The mechanism is the same: an applier per factor and crossed values. The statistics then need to know which factors are sources of randomness, to be averaged over, and which are experimental conditions, to be kept as keys. A `kind: seed | condition` per factor would express this.
2. **Partial results:** should the aggregation tolerate missing replications, for example a failed run in a long sweep, with a warning and the actual `n` per cell? Or fail, as flows do today?
3. **The base runs:** should a base run itself count as a replication? It has hand-chosen seeds, so it would not be a sample of the same seed distribution. The proposal keeps base runs out of the aggregation.
4. **Where the figure notebook lives:** a library template like `Aggregate.ipynb`, or project-specific, as proposed, since every project adds its own figure kinds?
