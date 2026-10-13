"""Tests for the replications: exprunflow/replication.py, metrics.py, aggregate.py and plots.py."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from exprunflow.exp_run_config import Config
from exprunflow.metrics import load_metrics, save_metrics
from exprunflow.replication import (
    check_reproducible, derive_seed, expand_replications, replication_entries, replication_names,
    write_exprun_variant)


def write_yaml(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(values, f, sort_keys=False)


class ToyConfig:
    """A Config over a temporary exp/run tree, resolving like Config.get_experiment (defaults, then run)"""

    def __init__(self, exprun_path, results_path, internal=None):
        self.exprun_path = pathlib.Path(exprun_path)
        self.exprun_path_internal = pathlib.Path(internal) if internal else pathlib.Path("/nonexistent")
        self.results_path = pathlib.Path(results_path)

    def get_exprun_path(self):
        return self.exprun_path

    def get_results_path(self):
        return self.results_path

    def set_results_path(self, path):
        self.results_path = pathlib.Path(path)

    def get_experiment(self, family, run, create_data_dir=True, creation_style="exist-ok"):
        values = {}
        defaults = self.exprun_path / family / f"_defaults_{family}.yaml"
        if defaults.exists():
            values.update(yaml.safe_load(defaults.read_text()) or {})
        values.update(yaml.safe_load((self.exprun_path / family / f"{run}.yaml").read_text()) or {})
        data_dir = self.results_path / family / run
        if create_data_dir:
            data_dir.mkdir(parents=True, exist_ok=True)
        return ToyExp(values, run, data_dir)


class ToyExp(dict):
    def __init__(self, values, run, data_dir):
        super().__init__(values)
        self["data_dir"] = str(data_dir)
        self._run = run

    def run(self):
        return self._run

    def data_dir(self):
        return pathlib.Path(self["data_dir"])


class TestSeedsAndNames(unittest.TestCase):
    def test_derived_seeds_are_stable_and_distinct(self):
        self.assertEqual(derive_seed(1, 0), derive_seed(1, 0))
        self.assertEqual(len({derive_seed(b, i) for b in range(1, 6) for i in range(4)}), 20)

    def test_replication_names_cross_the_factors_in_order(self):
        aggregate = {"runs": ["a", "b"], "factors": {"map-seed": {"code": "m", "values": [1, 2]},
                                                    "behavior-seed": {"code": "b", "values": [7]}}}
        self.assertEqual(replication_names(aggregate), ["a-m1-b7", "a-m2-b7", "b-m1-b7", "b-m2-b7"])

    def test_a_partial_experiment_spreads_over_the_first_factor(self):
        aggregate = {"runs": ["a"], "factors": {"map-seed": {"code": "m", "values": [1, 2, 3]},
                                               "behavior-seed": {"code": "b", "values": [7, 8]}}}
        self.assertEqual(replication_names({**aggregate, "replications": 4}), ["a-m1-b7", "a-m2-b7", "a-m3-b7", "a-m1-b8"])
        # a larger count extends a smaller one, and the full count has every combination
        self.assertEqual(replication_names({**aggregate, "replications": 2}), ["a-m1-b7", "a-m2-b7"])
        self.assertEqual(sorted(replication_names({**aggregate, "replications": 6})), sorted(replication_names(aggregate)))
        for count in [0, 7]:
            with self.assertRaises(ValueError):
                replication_names({**aggregate, "replications": count})


class TestExpansion(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.directory.name)
        self.expruns = root / "expruns"
        write_yaml(self.expruns / "run" / "_defaults_run.yaml", {"rounds": 3, "input-to-notebook": ["Run.ipynb"]})
        write_yaml(self.expruns / "run" / "a.yaml", {"approach": "x", "env": "e", "robots": [{"seed": 1}, {"seed": 2}]})
        write_yaml(self.expruns / "run" / "b.yaml", {"approach": "y", "env": "e", "robots": [{"seed": 1}, {}]})
        write_yaml(self.expruns / "env" / "_defaults_env.yaml", {"input-to-notebook": ["Prepare.ipynb"]})
        write_yaml(self.expruns / "env" / "e.yaml", {"map-seed": 0})
        write_yaml(self.expruns / "agg" / "_defaults_agg.yaml", {"input-to-notebook": ["Aggregate.ipynb"]})
        write_yaml(self.expruns / "agg" / "all.yaml", {
            "source-experiment": "run", "runs": ["a", "b"], "labels": ["approach"],
            "factors": {"map-seed": {"code": "m", "values": [1, 2]}, "behavior-seed": {"code": "b", "values": [5]}}})
        write_yaml(self.expruns / "fig" / "f.yaml", {"source-experiment": "agg", "source-runs": ["all"],
                                                    "input-to-notebook": ["Figure.ipynb"]})
        write_yaml(self.expruns / "flow" / "all.yaml", {"run-experiment": "run", "aggregate-experiment": "agg",
            "aggregate": "all", "figure-experiment": "fig", "figures": ["f"]})
        self.config = ToyConfig(self.expruns, root / "results")
        self.appliers = {
            "map-seed": lambda context, m: {"env": context.variant("env", context.base["env"], {"map-seed": m},
                                                                   f"e-m{m}", queue=True)},
            "behavior-seed": lambda context, b: {"robots": [
                {**robot, "seed": derive_seed(b, i)} if "seed" in robot else robot
                for i, robot in enumerate(context.base["robots"])]},
        }

    def tearDown(self):
        self.directory.cleanup()

    def load(self, family, run):
        return yaml.safe_load((self.expruns / family / f"{run}.yaml").read_text())

    def test_variant_keeps_the_family_defaults(self):
        write_exprun_variant("run", "a", {"approach": "z"}, "a-z", self.config)
        exp = self.config.get_experiment("run", "a-z", create_data_dir=False)
        self.assertEqual((exp["approach"], exp["rounds"]), ("z", 3))
        self.assertNotIn("rounds", self.load("run", "a-z"))  # inherited, not copied

    def test_variant_refuses_the_built_in_tree(self):
        config = ToyConfig(self.expruns, self.directory.name, internal=self.expruns)
        with self.assertRaisesRegex(Exception, "built-in"):
            write_exprun_variant("run", "a", {}, "a-copy", config)

    def test_entries_and_replication_blocks(self):
        flow = self.config.get_experiment("flow", "all", create_data_dir=False)
        entries = replication_entries(flow, self.appliers, "discard-old", self.config)
        self.assertEqual([(e["name"].split()[0], e["run"], e["notebook"]) for e in entries], [
            ("Prepare", "e-m1", "Prepare.ipynb"), ("Prepare", "e-m2", "Prepare.ipynb"),
            ("Run", "a-m1-b5", "Run.ipynb"), ("Run", "a-m2-b5", "Run.ipynb"),
            ("Run", "b-m1-b5", "Run.ipynb"), ("Run", "b-m2-b5", "Run.ipynb"),
            ("Aggregate", "all", "Aggregate.ipynb"), ("Figure", "f", "Figure.ipynb")])
        # the phases of a parallel flow
        self.assertEqual([e["phase"] for e in entries],
                         ["prepare"] * 2 + ["run"] * 4 + ["aggregate", "figure"])
        replication = self.load("run", "b-m2-b5")
        self.assertEqual(replication["replication"], {"base-run": "b", "factors": {"map-seed": 2, "behavior-seed": 5},
                                                      "labels": {"approach": "y"}})
        self.assertEqual(replication["env"], "e-m2")
        self.assertEqual(replication["robots"], [{"seed": derive_seed(5, 0)}, {}])
        self.assertEqual(self.load("env", "e-m2")["map-seed"], 2)

    def test_skip_completed_reuses_only_matching_completed_results(self):
        from exprunflow.replication import is_completed
        flow = self.config.get_experiment("flow", "all", create_data_dir=False)
        replication_entries(flow, self.appliers, "discard-old", self.config)

        def complete(family, run, **changes):
            exp = self.config.get_experiment(family, run)
            saved = {**{k: v for k, v in exp.items()}, "time_done": "2026-10-09 20:00:00", **changes}
            write_yaml(exp.data_dir() / "exprun.yaml", saved)

        complete("run", "a-m1-b5")                      # completed with the same configuration
        complete("run", "a-m2-b5", approach="changed")  # completed, but with another configuration
        exp = self.config.get_experiment("run", "b-m1-b5")
        write_yaml(exp.data_dir() / "exprun.yaml", dict(exp))  # started, never done
        complete("env", "e-m1")
        self.assertTrue(is_completed("run", "a-m1-b5", self.config))
        self.assertFalse(is_completed("run", "a-m2-b5", self.config))
        self.assertFalse(is_completed("run", "b-m1-b5", self.config))
        self.assertFalse(is_completed("run", "b-m2-b5", self.config))

        write_yaml(self.expruns / "flow" / "all.yaml", {**dict(flow), "skip-completed": True})
        flow = self.config.get_experiment("flow", "all", create_data_dir=False)
        entries = replication_entries(flow, self.appliers, "discard-old", self.config)
        self.assertEqual([(e["name"].split()[0], e["run"]) for e in entries], [
            ("Prepare", "e-m2"), ("Run", "a-m2-b5"), ("Run", "b-m1-b5"), ("Run", "b-m2-b5"),
            ("Aggregate", "all"), ("Figure", "f")])

    def test_a_partial_experiment_is_continued_without_repeating_its_runs(self):
        from exprunflow.replication import set_replications, without_completed
        write_yaml(self.expruns / "flow" / "all.yaml", {**self.load("flow", "all"), "replications": 1})
        flow = self.config.get_experiment("flow", "all", create_data_dir=False)
        entries = replication_entries(flow, self.appliers, "discard-old", self.config)
        self.assertEqual([(e["name"].split()[0], e["run"]) for e in entries], [
            ("Prepare", "e-m1"), ("Run", "a-m1-b5"), ("Run", "b-m1-b5"), ("Aggregate", "all"), ("Figure", "f")])
        # the count is written into the aggregate, with its other values, so the aggregation sees the same replications
        aggregate = self.load("agg", "all")
        self.assertEqual((aggregate["replications"], aggregate["runs"]), (1, ["a", "b"]))
        for family, run in [("env", "e-m1"), ("run", "a-m1-b5"), ("run", "b-m1-b5")]:
            exp = self.config.get_experiment(family, run)
            write_yaml(exp.data_dir() / "exprun.yaml", {**dict(exp), "time_done": "2026-10-12 20:00:00"})
        # continued to every replication: only the new ones run
        write_yaml(self.expruns / "flow" / "all.yaml", {**self.load("flow", "all"), "replications": 2, "skip-completed": True})
        flow = self.config.get_experiment("flow", "all", create_data_dir=False)
        entries = replication_entries(flow, self.appliers, "discard-old", self.config)
        self.assertEqual([(e["name"].split()[0], e["run"]) for e in entries], [
            ("Prepare", "e-m2"), ("Run", "a-m2-b5"), ("Run", "b-m2-b5"), ("Aggregate", "all"), ("Figure", "f")])
        self.assertEqual(self.load("agg", "all")["replications"], 2)
        # without_completed is the same test for any entries; None removes the count
        self.assertEqual([e["run"] for e in without_completed(
            [{"experiment": "run", "run": "a-m1-b5"}, {"experiment": "run", "run": "a-m2-b5"}], self.config)], ["a-m2-b5"])
        set_replications("agg", "all", None, self.config)
        self.assertNotIn("replications", self.load("agg", "all"))

    def test_figures_must_draw_the_aggregate(self):
        write_yaml(self.expruns / "fig" / "f.yaml", {"source-experiment": "run", "source-runs": ["a"],
                                                    "input-to-notebook": ["Figure.ipynb"]})
        flow = self.config.get_experiment("flow", "all", create_data_dir=False)
        with self.assertRaisesRegex(Exception, "does not draw"):
            replication_entries(flow, self.appliers, "discard-old", self.config)

    def test_metrics_identification_and_reproducibility_check(self):
        replications, _ = expand_replications(self.config.get_experiment("agg", "all", create_data_dir=False),
                                              self.appliers, self.config)
        exp = self.config.get_experiment("run", replications[0]["name"])
        save_metrics(exp, {"value": 1.0}, series={"t": [0, 1], "value": [0.5, 1.0]})
        metrics = load_metrics(exp.data_dir())
        self.assertEqual((metrics["base-run"], metrics["factors"], metrics["labels"]),
                         ("a", {"map-seed": 1, "behavior-seed": 5}, {"approach": "x"}))
        self.assertIn("processor", metrics["host"])
        plain = self.config.get_experiment("run", "a")
        save_metrics(plain, {"value": 2.0}, label_keys=["approach"])
        self.assertEqual(load_metrics(plain.data_dir())["labels"], {"approach": "x"})

        seeded = lambda exp: save_metrics(exp, {"value": float(np.random.default_rng(exp["robots"][0]["seed"]).random())})
        unseeded = lambda exp: save_metrics(exp, {"value": float(np.random.default_rng().random())})
        self.assertTrue(check_reproducible("run", replications[0]["name"], seeded, config=self.config)[0])
        self.assertFalse(check_reproducible("run", replications[0]["name"], unseeded, config=self.config)[0])
        timed = lambda exp: save_metrics(exp, {"value": 1.0, "seconds": float(np.random.default_rng().random())})
        self.assertFalse(check_reproducible("run", replications[0]["name"], timed, config=self.config)[0])
        self.assertTrue(check_reproducible("run", replications[0]["name"], timed, ignore=("seconds",), config=self.config)[0])


def synthetic_metrics():
    """Two approaches, two clusters (map seeds), three replications per cluster:
    value = 100 * map + 10 * behavior (+ 5 for approach a)"""
    metrics = []
    for approach, offset in [("a", 5), ("r", 0)]:
        for m in [1, 2]:
            for b in [1, 2, 3]:
                value = 100 * m + 10 * b + offset
                metrics.append({"run": f"{approach}-m{m}-b{b}", "base-run": approach,
                                "factors": {"map-seed": m, "behavior-seed": b}, "labels": {"approach": approach},
                                "scalars": {"v": value}, "entities": {"r1": {"group": "g", "v": value}},
                                "series": {"timestep": [9, 19], "v": [value / 2, value]}})
    return metrics


AGGREGATE = {"labels": ["approach"], "cluster-factor": "map-seed", "paired-by": "approach",
             "paired-order": ["a", "r"],
             "factors": {"map-seed": {"code": "m", "values": [1, 2]}, "behavior-seed": {"code": "b", "values": [1, 2, 3]}}}


class TestAggregation(unittest.TestCase):
    def test_statistics_on_a_synthetic_set(self):
        from scipy import stats
        from exprunflow.aggregate import aggregate_tables, confidence_interval, describe
        tables = aggregate_tables(synthetic_metrics(), AGGREGATE)
        self.assertEqual(len(tables["replications"]), 12)
        self.assertEqual(list(tables["replications"].columns[:4]), ["approach", "map-seed", "behavior-seed", "run"])
        row = tables["summary"].query("approach == 'a'").iloc[0]
        values = [100 * m + 10 * b + 5 for m in [1, 2] for b in [1, 2, 3]]
        low, high = stats.t.interval(0.95, 5, loc=np.mean(values), scale=stats.sem(values))
        self.assertAlmostEqual(row["mean"], np.mean(values))
        self.assertAlmostEqual(row["ci95_low"], low)
        self.assertAlmostEqual(row["ci95_high"], high)
        self.assertEqual(row["n_clusters"], 2)
        self.assertEqual(confidence_interval(row), (row["ci95_low_clusters"], row["ci95_high_clusters"]))
        paired = tables["paired"].iloc[0]
        self.assertEqual((paired["approach"], paired["reference"], paired["mean"], paired["fraction_greater"]),
                         ("a", "r", 5.0, 1.0))
        variance = tables["variance"].query("approach == 'a'").iloc[0]
        self.assertAlmostEqual(variance["variance_between"], np.var([120, 220], ddof=1))
        self.assertAlmostEqual(variance["variance_within"], np.var([10, 20, 30], ddof=1))
        self.assertEqual(len(tables["entities"]), 12)
        series = tables["summary-series"].query("approach == 'r' and timestep == 9").iloc[0]
        self.assertAlmostEqual(series["mean"], np.mean([(100 * m + 10 * b) / 2 for m in [1, 2] for b in [1, 2, 3]]))
        self.assertTrue(np.isnan(describe([3.0])["ci95_low"]))

    def test_plots_write_pdf_and_png(self):
        import matplotlib
        matplotlib.use("Agg")
        from exprunflow.aggregate import aggregate_tables
        from exprunflow.plots import plot_bars, plot_boxes, plot_paired, plot_ranges, plot_series
        metrics = synthetic_metrics()
        for m in metrics:
            m["labels"]["map"] = "one"
        aggregate = dict(AGGREGATE, labels=["map", "approach"])
        tables = aggregate_tables(metrics, aggregate)
        with tempfile.TemporaryDirectory() as directory:
            d = pathlib.Path(directory)
            plot_bars(tables, ["v"], "map", "approach", d / "bars.pdf")
            plot_ranges(tables, ["v"], "map", "approach", d / "ranges.pdf", where={"map": "one"})
            plot_boxes(tables, ["v"], "map", "approach", d / "boxes.pdf")
            plot_boxes(tables, ["v"], "map", "approach", d / "panels.pdf", start_at_zero=True,
                       panels=[{"where": {"map": "one"}, "title": "one"}, {"where": {}, "title": "all"}])
            plot_boxes(tables, ["v"], "map", "approach", d / "grid.pdf", start_at_zero=True,
                       panels=[[{"where": {"map": "one"}, "title": "one"}, {"where": {}, "title": "all"}],
                               [{"where": {}, "title": "all"}]])
            plot_series(tables, "v", "map", "approach", d / "series.pdf")
            plot_paired(tables, "v", "map", "approach", d / "paired.pdf")
            self.assertEqual(sorted(p.name for p in d.iterdir()), sorted(
                f"{n}.{e}" for n in ["bars", "ranges", "boxes", "panels", "grid", "series", "paired"] for e in ["pdf", "png"]))


class TestCompatibility(unittest.TestCase):
    def test_replication_imports_without_the_analysis_extra(self):
        """exprunflow.replication and exprunflow.metrics do not import pandas, scipy or matplotlib"""
        code = ("import sys; import exprunflow.replication, exprunflow.metrics; "
                "print(any(m in sys.modules for m in ['pandas', 'scipy', 'matplotlib']))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "False")

    def test_flow_does_not_export_the_replication_functions(self):
        import exprunflow.flow as flow
        for name in ["derive_seed", "write_exprun_variant", "replication_entries", "save_metrics"]:
            self.assertFalse(hasattr(flow, name), name)


if __name__ == "__main__":
    unittest.main()
