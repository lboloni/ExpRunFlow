"""Small helpers shared by the experiment-flow notebooks.

See docs/DESIGN-Flows.md for the flow contract.
"""

import concurrent.futures
import json
import os
import pathlib
import shutil
import signal
import threading
import time

import yaml

from exprunflow.exp_run_config import Config


def setup_flow(flow_name, experiment_families, flows_path=None, config=None):
    """Create an external flow workspace, copy the exp/run families into it
    from the currently active exp/run path, and point Config at it."""
    if config is None:
        config = Config()
    if flows_path is None:
        flows_path = config["flows_path"]

    source_path = pathlib.Path(config.get_exprun_path())
    flow_path = pathlib.Path(flows_path).expanduser() / flow_name
    expruns_path = flow_path / "expruns"
    results_path = flow_path / "results"
    notebooks_path = flow_path / "executed-notebooks"
    expruns_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    notebooks_path.mkdir(parents=True, exist_ok=True)

    for family in experiment_families:
        shutil.copytree(
            source_path / family,
            expruns_path / family,
            dirs_exist_ok=True,
        )

    config.set_exprun_path(expruns_path)
    config.set_results_path(results_path)
    return expruns_path, results_path, notebooks_path


def flow_entry(name, experiment, run, index, creation_style, config=None, phase=None):
    """Create a flow entry that runs the index-th notebook declared in the
    input-to-notebook field of the exp/run. The optional phase lets the entry
    run concurrently with the neighbouring entries of the same phase
    (docs/DESIGN-Parallelization.md)."""
    if config is None:
        config = Config()
    exp = config.get_experiment(experiment, run, create_data_dir=False)
    entry = {
        "name": name,
        "notebook": exp["input-to-notebook"][index],
        "experiment": experiment,
        "run": run,
        "creation_style": creation_style,
    }
    if phase is not None:
        entry["phase"] = phase
    return entry


def executed_notebook_path(entry, notebooks_path):
    """Return where run_notebook retains the executed copy of an entry."""
    stem = pathlib.Path(entry["notebook"]).stem
    return (pathlib.Path(notebooks_path)
            / f"{stem}_{entry['experiment']}_{entry['run']}.ipynb")


# the messages of the failures of a kernel start that are worth a retry, e.g. a
# port taken between its selection and its use when many kernels start at once
KERNEL_STARTUP_ERRORS = (
    "Address already in use", "Kernel died before replying",
    "Kernel didn't respond", "kernel_info")
KERNEL_START_ATTEMPTS = 3
# the seconds nbclient waits for the output of a cell after its reply
IOPUB_TIMEOUT = 60


def is_kernel_startup_error(error):
    """Whether an exception of papermill is a failure to start the kernel,
    rather than a failure inside the notebook."""
    if type(error).__name__ == "PapermillExecutionError":
        return False
    if type(error).__name__ == "ZMQError":
        return True
    return any(text in str(error) for text in KERNEL_STARTUP_ERRORS)


def run_notebook(
        entry, expruns_path, results_path, notebooks_path,
        notebook_root=None, executor=None, kernel_name=None):
    """Execute one flow entry and retain the executed notebook. The notebook
    path is relative to notebook_root, by default Config.SRC_ROOT; the kernel
    is by default Config.KERNEL_NAME. A failure to start the kernel is retried
    (up to KERNEL_START_ATTEMPTS attempts), a failure inside the notebook
    never."""
    if notebook_root is None:
        notebook_root = Config.SRC_ROOT
    if kernel_name is None:
        kernel_name = Config.KERNEL_NAME
    if executor is None:
        import papermill
        executor = papermill.execute_notebook

    notebook_path = pathlib.Path(notebook_root) / entry["notebook"]
    output_path = executed_notebook_path(entry, notebooks_path)
    parameters = {
        "experiment": entry["experiment"],
        "run": entry["run"],
        "creation_style": entry["creation_style"],
        "expruns_path": pathlib.Path(expruns_path).as_posix(),
        "results_path": pathlib.Path(results_path).as_posix(),
    }
    for attempt in range(1, KERNEL_START_ATTEMPTS + 1):
        try:
            # the kernel starts in the notebook's directory through the
            # resources of nbclient, not papermill's cwd: that changes the
            # working directory of the whole process, which the concurrent
            # notebooks of a parallel flow share
            executor(
                notebook_path,
                output_path,
                parameters=parameters,
                kernel_name=kernel_name,
                resources={"metadata": {"path": str(notebook_path.parent)}},
                # the cell progress bars of papermill (tqdm) are not thread
                # safe: only the serial flow shows them
                progress_bar=threading.current_thread() is threading.main_thread(),
                # the time nbclient waits for the output of a finished cell;
                # its default of 4 s is too short when many kernels share the
                # machine in a parallel flow
                iopub_timeout=IOPUB_TIMEOUT,
            )
            return output_path
        except Exception as error:
            if attempt == KERNEL_START_ATTEMPTS or not is_kernel_startup_error(error):
                raise
            time.sleep(attempt)
    return output_path


# the thread pools of the numerical libraries, limited in the kernels of a
# parallel flow
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def limit_threads(threads_per_worker, environ=None):
    """Set the thread variables that the user has not set, in the environment
    that the kernels inherit. Returns the variables it set."""
    if environ is None:
        environ = os.environ
    set_now = {}
    for variable in THREAD_VARIABLES:
        if variable not in environ:
            environ[variable] = str(threads_per_worker)
            set_now[variable] = environ[variable]
    return set_now


def resolve_workers(workers, phase=None):
    """The number of workers of a phase: an int, "auto" (the cores minus 2),
    or a dict by phase, with "default" for the other phases."""
    if isinstance(workers, dict):
        workers = workers.get(phase, workers.get("default", 1))
    if workers == "auto":
        workers = max(1, (os.cpu_count() or 1) - 2)
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"The number of workers must be at least 1, got {workers}")
    return workers


def phase_groups(entries):
    """Group the queue into maximal runs of consecutive entries with the same
    phase; an entry without a phase is a group of its own."""
    groups = []
    for entry in entries:
        phase = entry.get("phase")
        if phase is not None and groups and groups[-1][0] == phase:
            groups[-1][1].append(entry)
        else:
            groups.append((phase, [entry]))
    return groups


def check_distinct_results(phase, entries):
    """Entries that run concurrently must write into distinct result
    directories."""
    seen = {}
    for entry in entries:
        key = (entry["experiment"], entry["run"])
        if key in seen:
            raise ValueError(
                f"The entries {seen[key]!r} and {entry['name']!r} of phase "
                f"{phase!r} write into the same result directory "
                f"{key[0]}/{key[1]}, they cannot run concurrently")
        seen[key] = entry["name"]


def longest_first(entries, notebooks_path):
    """The entries ordered by the durations of their earlier executions,
    longest first; the entries without an executed notebook keep their
    position after them."""
    def duration(entry):
        path = executed_notebook_path(entry, notebooks_path)
        try:
            return read_executed_notebook(path)[0]
        except Exception:
            return None
    durations = [duration(entry) for entry in entries]
    known = sorted((i for i, d in enumerate(durations) if d is not None),
                   key=lambda i: -durations[i])
    unknown = [i for i, d in enumerate(durations) if d is None]
    return [entries[i] for i in known + unknown]


def interrupt_child_kernels():
    """Interrupt the kernels started by this process (SIGINT, as a kernel
    interrupt does), so that the running notebooks of a parallel flow stop.
    Needs psutil; without it, the running notebooks finish."""
    try:
        import psutil
    except ImportError:
        return 0
    count = 0
    for child in psutil.Process().children(recursive=True):
        try:
            if "ipykernel" in " ".join(child.cmdline()):
                child.send_signal(signal.SIGINT)
                count += 1
        except psutil.Error:
            pass
    return count


def run_flow(
        entries, expruns_path, results_path, notebooks_path,
        notebook_runner=run_notebook, progress_factory=None,
        workers=1, on_failure="stop", order="queue", threads_per_worker=1):
    """Run an ordered notebook queue with one overall progress bar. Before
    each stage it displays a link to the stage's executed notebook, which
    papermill updates while the stage runs.

    The entries of a phase (consecutive entries with the same "phase" key) run
    concurrently, at most workers at a time (an int, "auto", or a dict by
    phase); the phases run in queue order, with a barrier between them. With
    workers=1 (the default), or for entries without a phase, the entries run
    one at a time, in the queue order (docs/DESIGN-Parallelization.md).

    on_failure="stop" starts no new entry after a failure, "continue" finishes
    the phase of the failure first; then the first exception is raised, with
    all the failures as its flow_failures and the execution summary as its
    flow_execution. order="longest-first" starts the entries of a phase with
    the longest earlier executions first. In a parallel flow, the thread
    variables of the numerical libraries are set to threads_per_worker, unless
    already set.

    Returns the execution summary: the wall-clock seconds, the workers, the
    threads per worker."""
    from html import escape
    from IPython.display import HTML, display

    if on_failure not in ("stop", "continue"):
        raise ValueError(f"Unknown on_failure {on_failure!r}")
    if order not in ("queue", "longest-first"):
        raise ValueError(f"Unknown order {order!r}")

    def notebooks_left(count):
        unit = "notebook" if count == 1 else "notebooks"
        return f"{count} {unit} left"

    if progress_factory is None:
        from tqdm import tqdm
        progress_factory = tqdm

    groups = phase_groups(entries)
    for phase, group in groups:
        check_distinct_results(phase, group)
    parallel = any(phase is not None and resolve_workers(workers, phase) > 1
                   for phase, _ in groups)
    execution = {
        "workers": workers if isinstance(workers, (dict, int)) else resolve_workers(workers),
        "threads-per-worker": threads_per_worker if parallel else None,
        "wall-seconds": None}
    if parallel:
        limit_threads(threads_per_worker)
        display(HTML(
            f"<p>Parallel flow: {escape(str(execution['workers']))} workers, "
            f"{threads_per_worker} thread(s) per worker.</p>"))

    started = time.monotonic()
    failures = []
    progress = progress_factory(
        total=len(entries), desc="Overall flow", unit="notebook")
    progress.set_postfix_str(notebooks_left(len(entries)))

    def announce(entry):
        notebook = executed_notebook_path(entry, notebooks_path).resolve()
        display(HTML(
            f"<p>*** {escape(entry['name'])}: "
            f'<a href="{notebook.as_uri()}">{escape(str(notebook))}</a></p>'))

    def run_serially(group):
        for entry in group:
            remaining = len(entries) - progress.n
            progress.set_postfix_str(
                f"{notebooks_left(remaining)}; current: {entry['name']}")
            announce(entry)
            try:
                notebook_runner(
                    entry, expruns_path, results_path, notebooks_path)
            except Exception as error:
                failures.append((entry, error))
                if on_failure == "stop":
                    return
                continue
            progress.update(1)
            progress.set_postfix_str(notebooks_left(
                len(entries) - progress.n))

    def run_concurrently(phase, group, count):
        if order == "longest-first":
            group = longest_first(group, notebooks_path)
        pending = list(group)
        running = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            try:
                while pending or running:
                    while pending and len(running) < count and not (
                            failures and on_failure == "stop"):
                        entry = pending.pop(0)
                        announce(entry)
                        future = pool.submit(
                            notebook_runner, entry, expruns_path,
                            results_path, notebooks_path)
                        running[future] = entry
                        progress.set_postfix_str(
                            f"{notebooks_left(len(entries) - progress.n)}; "
                            f"phase {phase}: {len(running)} running; "
                            f"started: {entry['name']}")
                    if not running:
                        break
                    done, _ = concurrent.futures.wait(
                        running, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        entry = running.pop(future)
                        error = future.exception()
                        if error is not None:
                            failures.append((entry, error))
                        else:
                            progress.update(1)
                    progress.set_postfix_str(
                        f"{notebooks_left(len(entries) - progress.n)}; "
                        f"phase {phase}: {len(running)} running")
            except BaseException:
                # an interrupt: start nothing more, stop the running notebooks
                for future in running:
                    future.cancel()
                interrupt_child_kernels()
                raise

    try:
        for phase, group in groups:
            count = 1 if phase is None else resolve_workers(workers, phase)
            if count == 1 or len(group) == 1:
                run_serially(group)
            else:
                run_concurrently(phase, group, min(count, len(group)))
            if failures:
                break
    finally:
        execution["wall-seconds"] = time.monotonic() - started
        progress.close()
    if failures:
        error = failures[0][1]
        error.flow_failures = failures
        error.flow_execution = execution
        raise error
    return execution


def read_executed_notebook(path):
    """Return the papermill duration and error of an executed notebook."""
    with pathlib.Path(path).open() as handle:
        notebook = json.load(handle)
    error = None
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("papermill", {}).get("exception"):
            for output in cell.get("outputs", []):
                if output["output_type"] == "error":
                    error = f'{output["ename"]}: {output["evalue"]}'
    return notebook["metadata"]["papermill"].get("duration"), error


def get_flow_report(entries, results_path, notebooks_path, flow_error=None,
                    execution=None):
    """Return completion information for an executed notebook flow. A stage
    is complete if its exprun.yaml contains time_done. The duration and error
    of a stage come from the papermill metadata of its executed notebook. The
    execution summary of run_flow, if given, adds the wall-clock time and the
    effective parallelism (the summed stage time over the wall-clock time)."""
    results_path = pathlib.Path(results_path).resolve()
    stages = []
    for entry in entries:
        directory = results_path / entry["experiment"] / entry["run"]
        provenance = directory / "exprun.yaml"
        complete = False
        if provenance.exists():
            with provenance.open() as handle:
                values = yaml.safe_load(handle)
            complete = Config.TIME_DONE in values
        notebook = executed_notebook_path(entry, notebooks_path)
        duration, error = None, None
        if notebook.exists():
            duration, error = read_executed_notebook(notebook)
        else:
            notebook = None
        stages.append({
            "name": entry["name"],
            "directory": directory,
            "complete": complete,
            "notebook": notebook,
            "duration": duration,
            "error": error,
        })
    completed = [stage for stage in stages if stage["complete"]]
    missing = [stage for stage in stages if not stage["complete"]]
    total = sum(stage["duration"] for stage in stages
                if stage["duration"] is not None)
    if execution is None and flow_error is not None:
        execution = getattr(flow_error, "flow_execution", None)
    wall = None if execution is None else execution.get("wall-seconds")
    return {
        "stage-seconds": total,
        "wall-seconds": wall,
        "parallelism": total / wall if wall else None,
        "execution": execution,
        "successful": flow_error is None and not missing,
        "all-results-present": not missing,
        "stages": stages,
        "completed": completed,
        "missing": missing,
        "flow-directory": results_path.parent,
        "results-directory": results_path,
        "error": flow_error,
    }


def format_duration(seconds):
    """Format a duration in seconds as e.g. '1 min 48 s'."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {seconds} s"
    return f"{seconds} s"


def display_flow_report(
        entries, results_path, notebooks_path, final_results_path,
        flow_error=None, preview_files=(), execution=None):
    """Display flow status, a per-stage table, artifact links, and selected
    PNG previews."""
    from html import escape
    from IPython.display import HTML, Image, display, display_pdf

    report = get_flow_report(
        entries, results_path, notebooks_path, flow_error, execution)
    final_results_path = pathlib.Path(final_results_path).resolve()

    def directory_link(label, path):
        path = pathlib.Path(path).resolve()
        return (
            f'<p><strong>{escape(label)}:</strong> '
            f'<a href="{path.as_uri()}">{escape(str(path))}</a></p>')

    if report["successful"]:
        status = "Flow successfully executed."
    elif report["completed"]:
        status = (
            f'Flow has partial results: {len(report["completed"])} of '
            f'{len(entries)} stages completed.')
    else:
        status = "Flow did not produce any completed stage results."

    details = [f"<h2>Flow result</h2><p><strong>{escape(status)}</strong></p>"]
    if flow_error is not None:
        details.append(
            f'<p><strong>Error:</strong> {escape(repr(flow_error))}</p>')
    details.append(directory_link(
        "Flow data directory", report["flow-directory"]))
    details.append(directory_link(
        "All results", report["results-directory"]))
    details.append(directory_link("Final results", final_results_path))

    rows = []
    for stage in report["stages"]:
        if stage["complete"]:
            stage_status = "complete"
        elif stage["error"] is not None:
            stage_status = "failed"
        elif stage["notebook"] is not None:
            stage_status = "incomplete"
        else:
            stage_status = "not run"
        duration = "" if stage["duration"] is None \
            else format_duration(stage["duration"])
        notebook = "" if stage["notebook"] is None else (
            f'<a href="{stage["notebook"].as_uri()}">'
            f'{escape(stage["notebook"].name)}</a>')
        name = escape(stage["name"])
        if stage["error"] is not None:
            name += f'<br><small>{escape(stage["error"])}</small>'
        rows.append(
            f"<tr><td>{name}</td><td>{stage_status}</td>"
            f"<td>{duration}</td><td>{notebook}</td>"
            f'<td><a href="{stage["directory"].as_uri()}">results</a></td></tr>')
    total = report["stage-seconds"]
    wall = ""
    if report["wall-seconds"] is not None:
        wall = (f"<p><strong>Wall-clock time:</strong> "
                f'{format_duration(report["wall-seconds"])}, with '
                f'{escape(str(report["execution"]["workers"]))} worker(s); '
                f'effective parallelism {report["parallelism"]:.1f}</p>')
    details.append(
        "<h3>Stages</h3><table><tr><th>Stage</th><th>Status</th>"
        "<th>Duration</th><th>Executed notebook</th><th>Results</th></tr>"
        f'{"".join(rows)}</table>'
        f"<p><strong>Total stage time:</strong> {format_duration(total)}</p>"
        + wall)
    display(HTML("".join(details)))

    pdfs = sorted(final_results_path.rglob("*.pdf")) \
        if final_results_path.exists() else []
    if pdfs:
        links = "".join(
            f'<li><a href="{path.as_uri()}">'
            f'{escape(str(path.relative_to(final_results_path)))}</a></li>'
            for path in pdfs)
        display(HTML(
            f"<h3>Generated figures</h3><ul>{links}</ul>"))

    for preview in preview_files:
        if isinstance(preview, dict):
            path = pathlib.Path(preview["path"]).resolve()
            description = preview["description"]
        else:
            path = pathlib.Path(preview).resolve()
            description = None
        if path.exists():
            heading = f"<h3>{escape(path.name)}</h3>"
            if description is not None:
                heading += f"<p>{escape(description)}</p>"
            display(HTML(heading))
            if path.suffix.lower() == ".png":
                display(Image(filename=str(path)))
            else:
                display_pdf(path.read_bytes(), raw=True)
    return report
