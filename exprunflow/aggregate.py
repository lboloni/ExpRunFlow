"""The aggregation of replications into tidy tables (one observation per row), with the statistics from
which figures draw means, confidence intervals and ranges. Needs the optional extra `analysis` (pandas,
scipy). See docs/DESIGN-Replication.md, Sections 4 and 5."""

import itertools
import pathlib

import numpy as np

try:
    import pandas as pd
    from scipy import stats
except ImportError as error:  # pragma: no cover
    raise ImportError("exprunflow.aggregate needs the optional extra: pip install -e ExpRunFlow[analysis]") from error

from exprunflow.exp_run_config import Config
from exprunflow.metrics import load_metrics
from exprunflow.replication import factor_items, replication_names

TABLES = ["replications", "entities", "series", "summary", "summary-series", "paired", "variance", "hosts"]


def load_replications(aggregate, config=None):
    """The metrics of the replications declared by an aggregate exp/run"""
    if config is None:
        config = Config()
    return [load_metrics(config.get_experiment(aggregate["source-experiment"], run, create_data_dir=False)["data_dir"])
            for run in replication_names(aggregate)]


def key_columns(aggregate):
    """The key columns of the tables: the labels, the factors, and the run"""
    return list(aggregate.get("labels", [])) + [name for name, _, _ in factor_items(aggregate)] + ["run"]


def tidy_tables(metrics, aggregate):
    """The replications, entities and series tables of a list of metrics"""
    labels = list(aggregate.get("labels", []))
    factors = [name for name, _, _ in factor_items(aggregate)]
    replications, entities, series = [], [], []
    for m in metrics:
        keys = {**{label: m["labels"].get(label) for label in labels},
                **{factor: m["factors"].get(factor) for factor in factors}, "run": m["run"]}
        replications += [{**keys, "metric": metric, "value": value} for metric, value in m["scalars"].items()]
        for entity, values in m.get("entities", {}).items():
            entities += [{**keys, "entity": entity, "group": values.get("group"), "metric": metric, "value": value}
                         for metric, value in values.items() if metric != "group"]
        if m.get("series"):
            index_name, index = next(iter(m["series"].items()))
            for metric, values in list(m["series"].items())[1:]:
                series += [{**keys, "metric": metric, index_name: i, "value": v} for i, v in zip(index, values)]
    return pd.DataFrame(replications), pd.DataFrame(entities), pd.DataFrame(series)


def describe(values):
    """n, mean, std, sem, the 95% Student t confidence interval of the mean, and the range and quartiles.
    With n = 1 the spread statistics are NaN."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = std / np.sqrt(n) if n > 1 else np.nan
    half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
    return {"n": n, "mean": mean, "std": std, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half,
            "min": float(np.min(values)), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values))}


def summarize(table, keys, cluster_factor=None):
    """The statistics per group of keys over the replications and, if there is a cluster factor, the
    interval over the per-cluster means (n_clusters, ci95_low_clusters, ci95_high_clusters)"""
    rows = []
    for group, frame in table.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        row.update(describe(frame["value"]))
        if cluster_factor is not None:
            clusters = describe(frame.groupby(cluster_factor, dropna=False)["value"].mean())
            row.update({"n_clusters": clusters["n"], "ci95_low_clusters": clusters["ci95_low"],
                        "ci95_high_clusters": clusters["ci95_high"]})
        else:
            row.update({"n_clusters": 1, "ci95_low_clusters": np.nan, "ci95_high_clusters": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_interval(row):
    """The confidence interval a figure shows: over the per-cluster means if there are several clusters
    (conservative, Section 5.2), otherwise over the replications"""
    if row["n_clusters"] > 1:
        return row["ci95_low_clusters"], row["ci95_high_clusters"]
    return row["ci95_low"], row["ci95_high"]


def paired_differences(replications, labels, factors, paired_by, order=None):
    """The statistics of the per-replication differences value(a) - value(r) between every pair of values
    of the paired_by label, matched on all the factor values (common random numbers)"""
    others = [label for label in labels if label != paired_by]
    rows = []
    for group, frame in replications.groupby(others + ["metric"], sort=False, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        wide = frame.pivot_table(index=factors, columns=paired_by, values="value")
        values = [v for v in (order or sorted(wide.columns)) if v in wide.columns]
        for a, r in itertools.combinations(values, 2):
            diff = (wide[a] - wide[r]).dropna()
            if diff.empty:
                continue
            row = dict(zip(others + ["metric"], group))
            row.update({paired_by: a, "reference": r})
            row.update(describe(diff))
            row["fraction_greater"] = float(np.mean(diff > 0))
            row["fraction_less"] = float(np.mean(diff < 0))
            row["p_ttest"] = float(stats.ttest_rel(wide.loc[diff.index, a], wide.loc[diff.index, r]).pvalue) \
                if len(diff) > 1 and np.any(diff != diff.iloc[0]) else np.nan
            row["p_wilcoxon"] = float(stats.wilcoxon(diff).pvalue) if len(diff) > 1 and np.any(diff != 0) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def variance_components(replications, cell_keys, cluster_factor):
    """Per cell and metric: the variance between clusters (of the per-cluster means) and the mean
    within-cluster variance. NaN unless there are several clusters, and several replications per cluster."""
    rows = []
    for group, frame in replications.groupby(cell_keys + ["metric"], sort=False, dropna=False):
        row = dict(zip(cell_keys + ["metric"], group if isinstance(group, tuple) else (group,)))
        if cluster_factor is None:
            row.update({"clusters": 1, "per_cluster": len(frame), "variance_between": np.nan,
                        "variance_within": float(frame["value"].var(ddof=1)) if len(frame) > 1 else np.nan})
        else:
            by_cluster = frame.groupby(cluster_factor)["value"]
            clusters, per_cluster = by_cluster.ngroups, int(by_cluster.size().min())
            row.update({"clusters": clusters, "per_cluster": per_cluster,
                        "variance_between": float(by_cluster.mean().var(ddof=1)) if clusters > 1 else np.nan,
                        "variance_within": float(by_cluster.var(ddof=1).mean()) if per_cluster > 1 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_tables(metrics, aggregate):
    """All the tables of Section 4, by name"""
    labels = list(aggregate.get("labels", []))
    factors = [name for name, _, _ in factor_items(aggregate)]
    cluster = aggregate.get("cluster-factor")
    replications, entities, series = tidy_tables(metrics, aggregate)
    index_name = next((c for c in series.columns if c not in labels + factors + ["run", "metric", "value"]), None)
    tables = {
        "replications": replications,
        "entities": entities,
        "series": series,
        "summary": summarize(replications, labels + ["metric"], cluster),
        "summary-series": summarize(series, labels + ["metric", index_name], cluster) if index_name else pd.DataFrame(),
        "variance": variance_components(replications, labels, cluster),
        "hosts": pd.DataFrame(host_summary(metrics)),
    }
    paired_by = aggregate.get("paired-by")
    tables["paired"] = paired_differences(replications, labels, factors, paired_by, aggregate.get("paired-order")) \
        if paired_by else pd.DataFrame()
    return tables


def save_tables(tables, directory):
    for name, table in tables.items():
        table.to_csv(pathlib.Path(directory, f"{name}.csv"), index=False)


def load_tables(directory):
    tables = {}
    for name in TABLES:
        path = pathlib.Path(directory, f"{name}.csv")
        if path.exists():
            try:
                tables[name] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                tables[name] = pd.DataFrame()
    return tables


def host_summary(metrics):
    """The hosts the replications ran on, with the number of replications per host"""
    hosts = {}
    for m in metrics:
        host = m.get("host") or {}
        key = (host.get("processor"), host.get("cpus"), host.get("platform"), host.get("python"))
        hosts[key] = hosts.get(key, 0) + 1
    return [{"processor": k[0], "cpus": k[1], "platform": k[2], "python": k[3], "replications": n}
            for k, n in hosts.items()]
