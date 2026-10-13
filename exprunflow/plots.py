"""The standard figures of replicated results, drawn from the tidy tables of exprunflow.aggregate. Values
that are compared share their axis within a figure. Needs the optional extras `analysis` and `plots`
(matplotlib). See docs/DESIGN-Replication.md, Section 6.

Every function takes the tables, a `where` filter ({column: value}, applied to every table that has the
column), display names (`lookup`), colors, and the output path; it saves the figure as PDF and PNG."""

import pathlib

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
except ImportError as error:  # pragma: no cover
    raise ImportError("exprunflow.plots needs the optional extra: pip install -e ExpRunFlow[plots]") from error

from exprunflow.aggregate import confidence_interval

DEFAULT_COLORS = ["#CC6666", "#9999CC", "#66CC99", "#E69F00", "#56B4E9"]


def save_figure(fig, output_path):
    """Save a figure as a PDF and a PNG preview next to it"""
    path = pathlib.Path(output_path)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path.with_suffix(".pdf")


def select(table, where=None, **equal):
    """The rows of the table matching where and equal, for the columns the table has"""
    for column, value in {**(where or {}), **equal}.items():
        if column in table.columns:
            table = table[table[column] == value]
    return table


def _values(table, column, values):
    return list(values) if values is not None else list(dict.fromkeys(table[column]))


def _error(row):
    low, high = confidence_interval(row)
    if np.isnan(low):
        return None
    return [[row["mean"] - low], [high - row["mean"]]]


def _grouped(tables, metrics, x, color, output_path, draw, x_values, color_values, lookup, colors, where,
             metric_labels, start_at_zero, panels=None):
    """The layout of plot_bars, plot_ranges and plot_boxes: one panel per metric, the values of x on the x
    axis, one element per value of color. With panels, a list of {"where": {...}, "title": ...}, one panel
    per entry instead, for a single metric: its where is added to the common one, and the panels share
    their vertical axis, so that they can be compared directly. With panels as a list of rows (lists of such
    entries), the panels form a grid, and each row shares its own vertical axis."""
    lookup, colors, metric_labels = lookup or {}, colors or DEFAULT_COLORS, metric_labels or {}
    if panels:
        if len(metrics) != 1:
            raise ValueError("Panels by where filter need exactly one metric")
        rows = panels if isinstance(panels[0], list) else [panels]
        specs = [[(metrics[0], {**(where or {}), **(panel.get("where") or {})}, panel.get("title")) for panel in row]
                 for row in rows]
    else:
        specs = [[(metric, where, None) for metric in metrics]]
    x_values = _values(select(tables["summary"], where), x, x_values)
    color_values = _values(select(tables["summary"], where), color, color_values)
    # with a single value on the x axis (e.g. one scenario per figure), the elements fill the panel and the
    # value is not labelled, as the figure states it
    single = len(x_values) == 1
    width = (0.96 if single else 0.8) / len(color_values)
    rng = np.random.default_rng(0)  # the jitter of the points
    columns = max(len(row) for row in specs)
    fig, axes = plt.subplots(len(specs), columns, figsize=(3.2 * columns, 3 * len(specs)), squeeze=False,
                             sharey="row" if panels else False)
    cells = []
    for r, row in enumerate(specs):
        for c in range(columns):
            if c < len(row):
                cells.append((axes[r][c], row[c], c))
            else:
                axes[r][c].set_visible(False)
    for ax, (metric, panel_where, title), index in cells:
        summary = select(tables["summary"], panel_where)
        replications = select(tables["replications"], panel_where)
        for i, c in enumerate(color_values):
            for k, xv in enumerate(x_values):
                rows = select(summary, metric=metric, **{x: xv, color: c})
                if rows.empty:
                    continue
                position = k + (i - (len(color_values) - 1) / 2) * width
                values = select(replications, metric=metric, **{x: xv, color: c})["value"]
                draw(ax, position, width, rows.iloc[0], colors[i % len(colors)], values)
                ax.scatter(position + rng.uniform(-width / 5, width / 5, len(values)), values, s=6,
                           color="black", zorder=3)
        if single:
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
        else:
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels([lookup.get(v, v) for v in x_values])
            ax.set_xlim(-0.5, len(x_values) - 0.5)
        if title:
            ax.set_title(title, fontsize=9)
        if not panels or index == 0:
            ax.set_ylabel(metric_labels.get(metric, metric))
    # after every panel is drawn, so that a shared axis keeps the data of all of them
    if start_at_zero:
        for ax, _, _ in cells:
            ax.set_ylim(bottom=0)
    # the legend above the panels, in one row, so that it never covers the data
    fig.legend(handles=[Patch(facecolor=colors[i % len(colors)], alpha=0.7, label=lookup.get(c, c))
                        for i, c in enumerate(color_values)], fontsize=8, loc="upper center",
               ncol=len(color_values), frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.08 / len(specs)))
    return save_figure(fig, output_path)


def plot_bars(tables, metrics, x, color, output_path, x_values=None, color_values=None, lookup=None,
              colors=None, where=None, metric_labels=None, panels=None):
    """Grouped bars: one panel per metric, the mean as a bar with its 95% CI, and the replications as dots"""
    def draw(ax, position, width, row, color_value, values):
        ax.bar(position, row["mean"], width, color=color_value, yerr=_error(row), capsize=3, alpha=0.85)
    return _grouped(tables, metrics, x, color, output_path, draw, x_values, color_values, lookup, colors,
                    where, metric_labels, start_at_zero=True, panels=panels)


def plot_ranges(tables, metrics, x, color, output_path, x_values=None, color_values=None, lookup=None,
                colors=None, where=None, metric_labels=None, panels=None, start_at_zero=False):
    """Box and whiskers in the layout of plot_bars: the box is the 95% CI of the mean, with a line at the
    mean; the whiskers are the range of the replications; the dots the replications. The box and the
    whiskers are drawn independently: with few replications the CI can be wider than the range."""
    def draw(ax, position, width, row, color_value, values):
        low, high = confidence_interval(row)
        if np.isnan(low):
            low = high = row["mean"]
        box = 0.7 * width
        ax.add_patch(Rectangle((position - box / 2, low), box, high - low, facecolor=color_value, alpha=0.5,
                               edgecolor=color_value, linewidth=1))
        ax.plot([position - box / 2, position + box / 2], [row["mean"]] * 2, color=color_value, linewidth=2)
        ax.plot([position, position], [row["min"], row["max"]], color="black", linewidth=1)
        for end in [row["min"], row["max"]]:
            ax.plot([position - box / 4, position + box / 4], [end, end], color="black", linewidth=1)
        ax.update_datalim([(position, low), (position, high), (position, row["min"]), (position, row["max"])])
    return _grouped(tables, metrics, x, color, output_path, draw, x_values, color_values, lookup, colors,
                    where, metric_labels, start_at_zero=start_at_zero, panels=panels)


def plot_boxes(tables, metrics, x, color, output_path, x_values=None, color_values=None, lookup=None,
               colors=None, where=None, metric_labels=None, panels=None, start_at_zero=False):
    """Standard box plots in the layout of plot_bars: the box spans the quartiles of the replications, with a
    line at the median; the whiskers reach the most extreme replications within 1.5 times the interquartile
    range of the box; the dots are the replications, the outliers among them. With panels, one panel per
    where filter, on a shared vertical axis (_grouped); start_at_zero starts the axis at zero."""
    def draw(ax, position, width, row, color_value, values):
        box = ax.boxplot([list(values)], positions=[position], widths=0.7 * width, patch_artist=True,
                         showfliers=False, manage_ticks=False)
        for patch in box["boxes"]:
            patch.set(facecolor=color_value, alpha=0.5, edgecolor=color_value)
        for line in box["medians"]:
            line.set(color=color_value, linewidth=2)
        for line in box["whiskers"] + box["caps"]:
            line.set(color="black", linewidth=1)
    return _grouped(tables, metrics, x, color, output_path, draw, x_values, color_values, lookup, colors,
                    where, metric_labels, start_at_zero=start_at_zero, panels=panels)


def plot_series(tables, metric, panel, color, output_path, panel_values=None, color_values=None, lookup=None,
                colors=None, where=None, metric_labels=None, index="timestep", scale=1.0):
    """One panel per value of panel, sharing the y axis: per value of color, the mean curve with its 95% CI
    band and, lighter, the min-max range of the replications"""
    lookup, colors, metric_labels = lookup or {}, colors or DEFAULT_COLORS, metric_labels or {}
    summary = select(tables["summary-series"], where, metric=metric)
    panel_values = _values(summary, panel, panel_values)
    color_values = _values(summary, color, color_values)
    fig, axes = plt.subplots(1, len(panel_values), figsize=(3.5 * len(panel_values), 3), sharey=True, squeeze=False)
    for ax, pv in zip(axes[0], panel_values):
        for i, c in enumerate(color_values):
            frame = select(summary, **{panel: pv, color: c}).sort_values(index)
            if frame.empty:
                continue
            col = colors[i % len(colors)]
            ci = np.array([confidence_interval(row) for _, row in frame.iterrows()], dtype=float) / scale
            ax.fill_between(frame[index], frame["min"] / scale, frame["max"] / scale, color=col, alpha=0.12, linewidth=0)
            ax.fill_between(frame[index], ci[:, 0], ci[:, 1], color=col, alpha=0.35, linewidth=0)
            ax.plot(frame[index], frame["mean"] / scale, color=col, label=lookup.get(c, c))
        ax.set_title(lookup.get(pv, pv), fontsize=10)
        ax.set_xlabel(lookup.get(index, index))
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_paired(tables, metric, panel, paired_by, output_path, panel_values=None, lookup=None, where=None,
                metric_labels=None):
    """The paired differences of the paired_by label, one panel per value of panel, sharing the x axis: the
    mean difference with its 95% CI, the zero line, and the fraction of replications in which the first
    value is greater"""
    lookup, metric_labels = lookup or {}, metric_labels or {}
    paired = select(tables["paired"], where, metric=metric)
    panel_values = _values(paired, panel, panel_values)
    frames = [select(paired, **{panel: pv}) for pv in panel_values]
    rows = max(len(f) for f in frames)
    fig, axes = plt.subplots(1, len(panel_values), figsize=(3.5 * len(panel_values), 0.8 + 0.5 * rows),
                             sharex=True, squeeze=False)
    for ax, pv, frame in zip(axes[0], panel_values, frames):
        names = []
        for i, (_, row) in enumerate(frame.iterrows()):
            ax.errorbar(row["mean"], i, xerr=[[row["mean"] - row["ci95_low"]], [row["ci95_high"] - row["mean"]]]
                        if not np.isnan(row["ci95_low"]) else None, fmt="o", color="black", capsize=4)
            ax.annotate(f"{row['fraction_greater']:.0%} greater", (row["mean"], i), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7)
            names.append(f"{lookup.get(row[paired_by], row[paired_by])} - {lookup.get(row['reference'], row['reference'])}")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(-0.6, rows - 0.2)
        ax.set_title(lookup.get(pv, pv), fontsize=10)
        ax.set_xlabel(f"Difference: {metric_labels.get(metric, metric)}")
    fig.tight_layout()
    return save_figure(fig, output_path)
