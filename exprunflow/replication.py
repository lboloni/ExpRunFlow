"""Replications: running base runs several times with varied random seeds.

An aggregate exp/run declares the base runs and the factors (named sources of randomness with their
seed values); the project supplies one applier per factor, which turns a seed into the changes of an
exprun. This module writes the replications as exprun variants into the flow workspace and builds their
flow entries. See docs/DESIGN-Replication.md.

The functions are deliberately not part of flow.py: projects re-export flow.py with a star import.
"""

import itertools
import pathlib
import shutil
import tempfile

import numpy as np
import yaml

from exprunflow.exp_run_config import Config


def derive_seed(base, *keys):
    """An independent seed derived from a base seed and keys (e.g. the index of a robot), with
    numpy's SeedSequence: unlike offsets, derived seeds never collide across bases and keys"""
    return int(np.random.SeedSequence([int(base), *[int(k) for k in keys]]).generate_state(1)[0])


def factor_items(aggregate):
    """The factors of an aggregate exp/run, as (name, code, values) in declaration order"""
    return [(name, spec["code"], list(spec["values"])) for name, spec in aggregate["factors"].items()]


def replication_name(base_run, factor_values, codes):
    """The name of a replication: the base run, then -<code><value> for every factor in order"""
    return base_run + "".join(f"-{code}{value}" for code, value in zip(codes, factor_values))


def factor_combinations(aggregate):
    """Every combination of the factor values, as dicts, in a fixed order (the last factor fastest)"""
    items = factor_items(aggregate)
    names = [name for name, _, _ in items]
    return [dict(zip(names, values)) for values in itertools.product(*[values for _, _, values in items])]


def replication_names(aggregate):
    """The replications of an aggregate exp/run: every base run with every combination of factor values"""
    codes = [code for _, code, _ in factor_items(aggregate)]
    return [replication_name(run, list(combination.values()), codes)
            for run in aggregate["runs"] for combination in factor_combinations(aggregate)]


def write_exprun_variant(family, run, changes, new_run, config=None):
    """Write the run of the family, with the changes to its top-level values, as the new run into the active
    exprun path. Unlike create_exprun_variant, the run's own file is copied rather than its resolved values,
    so the variant keeps inheriting the defaults of its family. Refuses to write into the built-in tree."""
    if config is None:
        config = Config()
    exprun_path = pathlib.Path(config.get_exprun_path())
    internal = getattr(config, "exprun_path_internal", None)
    if internal is not None and exprun_path.resolve() == pathlib.Path(internal).resolve():
        raise Exception("write_exprun_variant does not write into the built-in exp/run tree")
    family_path = exprun_path / family
    with open(family_path / f"{run}.yaml") as f:
        values = yaml.safe_load(f) or {}
    values.update(changes)
    with open(family_path / f"{new_run}.yaml", "w") as f:
        f.write(f"# generated from {family}/{run} by exprunflow.replication.write_exprun_variant, do not edit\n")
        yaml.safe_dump(values, f, sort_keys=False)
    return new_run


class ApplyContext:
    """What an applier sees: the resolved base run, the Config, and a way to write dependency variants
    (written once per name; queued ones are run before the replications, e.g. a precomputation)"""

    def __init__(self, base, config, written, queued):
        self.base = base
        self.config = config
        self._written = written
        self._queued = queued

    def variant(self, family, run, changes, name, queue=False):
        if (family, name) not in self._written:
            write_exprun_variant(family, run, changes, name, self.config)
            self._written.add((family, name))
            if queue:
                self._queued.append((family, name))
        return name


def expand_replications(aggregate, appliers, config=None):
    """Write every replication declared by the aggregate exp/run, with the changes of the appliers
    (factor name -> function(context, value) -> dict of changes) and a replication block. Returns the
    replications, as dicts with name, base-run, factors and labels, and the queued dependency variants,
    as (family, name) pairs."""
    if config is None:
        config = Config()
    family = aggregate["source-experiment"]
    items = factor_items(aggregate)
    missing = [name for name, _, _ in items if name not in appliers]
    if missing:
        raise Exception(f"No applier for the factors {missing}")
    codes = [code for _, code, _ in items]
    written, queued, replications = set(), [], []
    for base_run in aggregate["runs"]:
        base = config.get_experiment(family, base_run, create_data_dir=False)
        labels = {key: base[key] for key in aggregate.get("labels", [])}
        context = ApplyContext(base, config, written, queued)
        for combination in factor_combinations(aggregate):
            changes = {}
            for name, value in combination.items():
                changes.update(appliers[name](context, value))
            name = replication_name(base_run, list(combination.values()), codes)
            changes["replication"] = {"base-run": base_run, "factors": combination, "labels": labels}
            write_exprun_variant(family, base_run, changes, name, config)
            replications.append({"name": name, "base-run": base_run, "factors": combination, "labels": labels})
    return replications, queued


# the values of a saved exprun.yaml that change with every execution, and are not part of the configuration
VOLATILE_KEYS = {"data_dir", "time_started", "time_done", "exp_run_sys_indep_file"}


def is_completed(family, run, config=None):
    """Whether the exp/run has a completed result that can be reused: its saved exprun.yaml records 
    time_done (the completion evidence of flows), and every value of the exp/run's current configuration 
    equals the saved one. Not detected: changes of the code, and of other exp/runs that the run reads 
    (e.g. its environment), which a project must rule out itself before reusing results."""
    if config is None:
        config = Config()
    exp = config.get_experiment(family, run, create_data_dir=False)
    path = pathlib.Path(exp["data_dir"], "exprun.yaml")
    if not path.exists():
        return False
    with open(path) as f:
        saved = yaml.safe_load(f) or {}
    if "time_done" not in saved:
        return False
    current = exp.values if hasattr(exp, "values") and isinstance(exp.values, dict) else dict(exp)
    return all(key in saved and saved[key] == value for key, value in current.items()
               if key not in VOLATILE_KEYS and not key.startswith("timer"))


def replication_entries(collection, appliers, creation_style, config=None):
    """The flow entries of a replicated collection: the queued dependency variants, the replications, the
    aggregation, and the figures, in this order. The collection names the run experiment, the aggregate
    (aggregate-experiment, aggregate), and the figures (figure-experiment, figures); the figures must draw
    the aggregate."""
    if config is None:
        config = Config()
    run_experiment = collection["run-experiment"]
    aggregate_experiment = collection["aggregate-experiment"]
    aggregate = config.get_experiment(aggregate_experiment, collection["aggregate"], create_data_dir=False)
    if aggregate["source-experiment"] != run_experiment:
        raise Exception(f"The aggregate {collection['aggregate']} replicates {aggregate['source-experiment']}, "
                        f"expected {run_experiment}")
    replications, queued = expand_replications(aggregate, appliers, config)
    # with skip-completed, the replications (and the dependency variants) that already have a completed 
    # result with the same configuration are not queued again, e.g. to resume an interrupted flow
    skip = bool(collection.get("skip-completed", False))
    if skip:
        queued = [(family, name) for family, name in queued if not is_completed(family, name, config)]
        replications = [r for r in replications if not is_completed(run_experiment, r["name"], config)]

    def entry(prefix, family, run):
        exp = config.get_experiment(family, run, create_data_dir=False)
        return {"name": f"{prefix} {family}/{run}", "notebook": exp["input-to-notebook"][0],
                "experiment": family, "run": run, "creation_style": creation_style}

    entries = [entry("Prepare", family, name) for family, name in queued]
    entries += [entry("Run", run_experiment, r["name"]) for r in replications]
    entries.append(entry("Aggregate", aggregate_experiment, collection["aggregate"]))
    figure_experiment = collection["figure-experiment"]
    for figure_name in collection.get("figures", []):
        figure = config.get_experiment(figure_experiment, figure_name, create_data_dir=False)
        if figure["source-experiment"] != aggregate_experiment or list(figure["source-runs"]) != [collection["aggregate"]]:
            raise Exception(f"Figure {figure_name} does not draw the aggregate {collection['aggregate']}")
        entries.append(entry("Figure", figure_experiment, figure_name))
    return entries


def check_reproducible(family, run, run_stage, compare=("scalars", "series", "entities"), ignore=(), config=None):
    """Run the stage of the exp/run twice, each time into a fresh temporary results path, and compare the
    chosen parts of the metrics.json it writes. run_stage(exp) runs the stage on a resolved exp/run.
    ignore: the measurements that legitimately differ between runs, such as computation times; they are
    left out of the scalars and series, and of the values of every entity.
    Returns (reproducible, first metrics, second metrics)."""
    from exprunflow.metrics import load_metrics
    if config is None:
        config = Config()
    previous = config.get_results_path()
    outcomes = []
    try:
        for _ in range(2):
            directory = tempfile.mkdtemp()
            try:
                config.set_results_path(pathlib.Path(directory))
                exp = config.get_experiment(family, run, creation_style="discard-old")
                run_stage(exp)
                outcomes.append(load_metrics(exp.data_dir()))
            finally:
                shutil.rmtree(directory, ignore_errors=True)
    finally:
        config.set_results_path(previous)

    def without_ignored(metrics, part):
        values = metrics.get(part) or {}
        if part == "entities":
            return {entity: {k: v for k, v in row.items() if k not in ignore} for entity, row in values.items()}
        return {k: v for k, v in values.items() if k not in ignore}

    first, second = outcomes
    same = all(without_ignored(first, part) == without_ignored(second, part) for part in compare)
    return same, first, second
