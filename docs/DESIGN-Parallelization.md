# Parallel flow design

**Status:** implemented in `exprunflow/flow.py`:
- `run_flow` takes `workers`, `on_failure`, `order` and `threads_per_worker`, and returns the execution summary;
- `get_flow_report` and `display_flow_report` take that summary as `execution`;
- `replication_entries` sets the phases;
- the tests are in `tests/test_parallel_flow.py`.

The implementation differs from the proposal below in two points:
- **Interrupting the running kernels:** on an interrupt, `interrupt_child_kernels` sends SIGINT to the kernel processes that the flow started, found with `psutil`, instead of keeping a registry of kernel managers. Without `psutil`, the running notebooks finish.
- **No integration test with real notebooks:** the integration test of Section 9 is not in the library. A parallel run of real papermill kernels was checked by hand: 8 notebooks with 4 workers overlapped 4 at a time, and the kernels saw `OMP_NUM_THREADS=1`.

## Objective

`run_flow` executes the queue of a flow one notebook at a time. In a replicated flow, almost all the notebooks are independent of each other: the replications only read their own exprun and the prepared inputs. Their results do not depend on the order in which they run.

An example is the MAREX paper flow of WaterberryFarms (`marex2027-flow/icc-2027-paper`), on a 32-core machine:
- the full experiment has 84 base runs × 20 replications, i.e. 1,680 run notebooks;
- one replication of every base run takes about 19 minutes serially, so the full experiment takes about 6.3 hours;
- it uses one core.

With 28 workers, the same flow should take roughly 15–30 minutes. The range depends on the memory bandwidth and on the slowest runs at the end.

The library should run independent notebooks of a queue concurrently:
- the results do not change;
- a flow that does not ask for it runs exactly as now;
- failures are reported as they are now;
- projects need only small changes.

**Not a goal:** distribution over several machines. Section 10 lists it as an extension.

## General principles

- **The queue declares what may run together, the executor decides when.** An entry declares its *phase*. The entries of one phase are independent of each other. The phases run in the order of the queue, with a barrier between consecutive phases.
- **The serial flow is the special case `workers=1`.** It is also the default, so existing flows and notebooks behave identically.
- **Every notebook keeps its own process.** Papermill already starts a separate kernel per notebook. The parallel executor only starts several kernels at once. No project code runs in threads of the flow notebook.
- **A notebook's result is a function of its exprun.** This is already required of a replicated stage ([DESIGN-Replication.md](DESIGN-Replication.md), Section 8). In a parallel flow it must also hold for every entry of a phase: an entry must not read anything that another entry of the same phase writes.
- **Everything shared is written before execution starts.** The queue builders (`replication_entries`, `set_replications`, `write_exprun_variant`, the project's builders) run serially in the flow notebook, before `run_flow`. Workers never write expruns.
- **One core per notebook.** Numerical libraries do not start their own thread pools inside a parallel flow (Section 5).

## 1. Phases

### 1.1 The `phase` key

A flow entry may carry an integer or string `phase`. The executor groups the queue into maximal runs of **consecutive** entries with the same phase:
- a group starts when all the entries of the previous group have finished;
- the entries within a group run concurrently, at most `workers` at a time.

Only consecutive grouping is used, so the queue order stays the single source of the execution order. A phase label that appears again later in the queue starts a new group.

An entry **without** a `phase` is its own group. A queue without phases therefore runs serially, in its order, whatever the number of workers. This makes the extension backward compatible for projects that build their own entries.

### 1.2 The phases of the library's builders

`replication_entries` sets the phases:

| Phase | Entries | Why it is a barrier |
|---|---|---|
| `prepare` | the queued dependency variants (e.g. the environments of the map seeds) | the runs read them |
| `run` | the replications | the aggregate reads all of them |
| `aggregate` | the aggregation | the figures read its tables |
| `figure` | the figures of the aggregate | |

`flow_entry` takes an optional `phase=None` argument, and project builders set the same keys.

### 1.3 A dependency graph instead of phases (not proposed now)

A barrier waits for the slowest entry of a phase. A dependency graph would avoid this: each entry would list the entries it `requires`, so for example a replication could start as soon as its own environment is prepared.

For the flows we have, the gain is small:
- the prepare phase takes about 80 s of a multi-hour flow;
- the aggregate is a single notebook.

Phases are simpler to declare and to check, and they keep the queue readable. The executor is written so that phases can later be replaced by a `requires` relation (Section 10).

## 2. The executor

### 2.1 API

```python
run_flow(entries, expruns_path, results_path, notebooks_path,
         notebook_runner=run_notebook, progress_factory=None,
         workers=1, on_failure="stop", order="queue")
```

- **`workers`:** the maximum number of notebooks running at once. Use `workers=1` for serial execution (the default), or `workers="auto"` for `os.cpu_count() - 2`, which leaves room for the flow notebook and the system.
- **`on_failure`:** `"stop"` (the default) submits no new entries after the first failure, waits for the running ones, and raises the first exception. `"continue"` finishes the current phase, then stops before the next phase, because that phase may need the failed result.
- **`order`:**
  - `"queue"` submits the entries of a phase in queue order;
  - `"longest-first"` submits first the entries whose earlier executions took longest, read from their executed notebooks (`metadata.papermill.duration`) when they exist, which shortens the tail of a phase. Entries without a history keep their queue position.

### 2.2 Implementation

A `concurrent.futures.ThreadPoolExecutor` with `workers` threads calls `notebook_runner` for each entry:
- each thread only waits on its papermill call, and the work happens in the kernel subprocess, so threads are sufficient;
- `notebook_runner` remains injectable, so the tests use a fake runner (Section 9).

The main thread:
- submits the entries of one phase;
- collects the futures with `as_completed`;
- updates the progress bar, and does all `display` calls. Worker threads never touch IPython display.

At a barrier, it waits for all the futures of the phase before submitting the next phase.

With `workers=1`, the executor calls the runner directly in the main thread, as now, with no thread pool. The serial path therefore stays identical, including its tracebacks.

### 2.3 Progress display

There is still one overall progress bar. Its postfix shows:
- the current phase;
- the number of notebooks running and left;
- the name of the most recently started entry.

Instead of one link per stage as it starts, the executor displays the link to each executed notebook when the entry is submitted, so the running notebooks can be opened while they run.

### 2.4 Interruption

A `KeyboardInterrupt` in the flow notebook (or the termination of the papermill run of the flow notebook):
- cancels the entries that have not started (`executor.shutdown(cancel_futures=True)`);
- sends an interrupt to the running kernels, through a registry of the kernel managers that the default `run_notebook` keeps while papermill executes;
- leaves the partial results behind.

`skip-completed` already resumes such a flow (DESIGN-Replication.md, Section 2.5), so an interrupted parallel flow is resumed exactly like a serial one.

## 3. Failures and the report

The failure semantics of [DESIGN-Flows.md](DESIGN-Flows.md) are kept:
- the exception of a failed stage is re-raised after the report cell;
- its executed notebook is the diagnostic artifact.

In a parallel flow, several entries may fail in the same phase. `run_flow` raises the first one, and attaches all of them as `error.flow_failures`, a list of `(entry, exception)`.

`get_flow_report` needs no change: it already reads every stage from its `exprun.yaml` and executed notebook. It gains two numbers:
- the **wall-clock time** of the flow, measured by `run_flow` and passed to the report;
- the **sum of the stage times**, which it reports now.

Their ratio is the effective parallelism.

## 4. Correctness conditions for the entries of a phase

The library cannot check these conditions. It documents them as part of the stage contract of [DESIGN-ExpRun.md](DESIGN-ExpRun.md).

1. **Distinct result directories.** Every entry writes only into `results/<experiment>/<run>/`, and no two entries of a phase have the same experiment and run. The executor **checks** this before it starts a phase, and raises if two entries share a directory.
2. **No writes into shared inputs.** An entry may read the results of earlier phases, but must not modify them, and must not create them lazily.
   - Example from WaterberryFarms: an environment with `cache: true` is created and saved by the first notebook that needs it when its cache does not exist (`wbf_helper.create_env`). In a parallel flow, two runs of the same environment would both create it. The prepare phase must therefore precompute every cached environment. `single_run_entries` and `replication_entries` already queue the environments first, so the condition holds if they are put into the `prepare` phase.
3. **No shared temporary files.** Scratch files go into the run's own directory or into `tempfile` names, never into fixed paths. A project audits its notebooks for this once.
4. **No dependence on the execution order.** This holds automatically for seeded runs. The derived seeds of DESIGN-Replication.md do not depend on the queue position.
5. **The executed notebooks have distinct names.** `executed_notebook_path` already names them by notebook, experiment and run, which condition 1 makes unique.

## 5. Resources

- **Threads per kernel.**
  - **The problem:** NumPy, SciPy and GPy start a BLAS/OpenMP thread pool of one thread per core in every kernel. With 28 kernels on 32 cores, that is about 900 threads, which slows every run down instead of speeding the flow up.
  - **The fix:** for `workers > 1`, the executor sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to `threads_per_worker` (default 1) in the environment that the kernels inherit. It does this before the first kernel starts, and only for variables the user has not set.
  - **The parameter:** `threads_per_worker` is a parameter of `run_flow`. A project whose runs are themselves multi-threaded may trade fewer workers for more threads each.
- **Memory.** Peak memory is `workers` × the largest kernel. The executor does not measure it. The flow notebook may lower `workers` for memory-heavy phases, with a per-phase override `workers={"run": 28, "figure": 8}`.
  - For WaterberryFarms, a run kernel stays well below 1 GB, and the machine has 123 GB, so this is not a constraint there.
- **Kernel startup.** Papermill kernels communicate over local TCP ports. When many kernels start at the same moment, a port can occasionally be taken between its selection and its use ("Address already in use"). The default `run_notebook` therefore retries a kernel *startup* failure up to 3 times. It never retries a failure inside the notebook, which would hide real errors.
- **Disk.** The executed notebooks and the results are written as now. Parallelism does not change their total size.

## 6. Timing measurements

The runs record their computation times, for example the timers of the exprun and `metadata.papermill.duration`. Projects may report them, as the MAREX paper does with the time per run.

Under parallel load, a run is slower than alone, because of memory bandwidth and turbo clocks. Therefore:
- `run_flow` records `workers` and `threads_per_worker` in the executed flow notebook. Projects that report times should also put the worker count into the generated text, e.g. "on 28 parallel workers";
- the reproducibility check of DESIGN-Replication.md already ignores timing measurements (`ignore`), so the compared results are unaffected.

## 7. Flow notebooks and projects

The template flow notebook gains a parameter `workers = 1`, which it passes to `run_flow`.

For WaterberryFarms and the MAREX paper:
- `wbf_flow.single_run_entries` and `build_marex_2027_flow_entries` set the phases. The single-run part and the replicated part of a combined collection can share phases:

  | Phase | Entries |
  |---|---|
  | `prepare` | the environments of the example runs and of the replications |
  | `run` | the example runs and the replications |
  | `aggregate` | the aggregation |
  | `figure` | the figures of the example runs and of the aggregate |

  The example figures only read the example runs, so they could start earlier. Moving them to the end costs about 40 s and keeps the queue simple.
- `MAREX-Flow.ipynb` gains the parameter `workers`.
- `paper-MRMR/scripts/paper_flow.py` gains `--workers N` (default `auto`).
- The sync script's `numbers.tex`:
  - reports the wall-clock time of the flow next to the summed notebook time;
  - adds a macro with the worker count, so the paper's statement about the computation time stays true.

## 8. Expected speed-up

| Phase | Serial | 28 workers (estimate) |
|---|---|---|
| prepare (the environments of the map seeds) | ~80 s at 3 replications; more at 20 | < 1 min |
| run (1,680 notebooks) | ~6 h | ~15–25 min |
| aggregate | ~15 s | ~15 s (one notebook) |
| figures (14) | ~40 s | ~10 s |

The run phase is bounded below by:
- the summed notebook time divided by the workers;
- the longest single run (a 200×200 run with 10 robots, a few minutes at most).

The kernel startup of every notebook (about 2 s) is paid in parallel too, so it no longer adds up. Memory bandwidth may limit the speed-up below the core count; a first measurement with 8, 16 and 28 workers on the MAREX flow should settle the default.

## 9. Tests (in the library)

All the tests use a fake `notebook_runner` that sleeps briefly, records its start and end times and thread, and optionally raises.

They check that:
- **with `workers=1`,** the entries run in queue order, one at a time, in the main thread, and that a queue without phases also runs serially with any number of workers;
- **within a phase,** at most `workers` entries overlap, and more than one does when allowed;
- **at a barrier,** no entry of a later phase starts before every entry of the earlier phase has ended;
- **non-consecutive repeated phase labels** form separate groups;
- **`on_failure="stop"`** starts no new entry after a failure, raises the first exception, and attaches all the failures;
- **`on_failure="continue"`** finishes the phase and stops before the next one;
- **two entries of a phase with the same result directory** are rejected before anything runs;
- **the thread variables** are set for `workers > 1`, and the variables set by the user are kept;
- **`longest-first`** orders by the durations of existing executed notebooks;
- **`replication_entries`** assigns the four phases.

An integration test runs a small replicated flow (the existing test flow of `tests/test_replication.py`) with 1 and with 4 workers, and compares the aggregated tables, ignoring the timing columns.

## 10. Extensions (not proposed now)

- **A dependency graph:** `requires` instead of phases (Section 1.3). The executor would submit an entry when everything it requires has completed. The barrier executor is the special case where every entry requires the whole previous phase.
- **Several machines:** a runner that submits each entry to a batch system (e.g. Slurm), or over SSH to hosts that share the workspace. It would be another `notebook_runner`. The phase and failure semantics stay the same. The `host` that `metrics.json` already records shows where each replication ran.
- **Resource-aware scheduling:** a per-entry `weight` (e.g. its expected memory), for flows whose kernels differ a lot in size.

## 11. Open questions

1. **The default worker count.** Should `"auto"` be the default of the project flow scripts, or should every flow opt in explicitly? The proposal keeps the library default at 1 and makes the scripts default to `auto`.
2. **Interrupting the running kernels.** Should an interrupt stop the running notebooks immediately, or let them finish so that their results are complete? Immediate interruption is the proposal; `skip-completed` makes the loss small.
3. **Reported computation times in the paper.** Should the computation times in the paper come from a separate serial calibration run, rather than from the parallel flow?
