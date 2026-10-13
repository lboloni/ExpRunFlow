"""Tests for the parallel execution of a flow queue (docs/DESIGN-Parallelization.md)."""

import json
import pathlib
import tempfile
import threading
import time
import unittest
from unittest import mock

from exprunflow import flow
from exprunflow.flow import (
    THREAD_VARIABLES, executed_notebook_path, get_flow_report, is_kernel_startup_error, limit_threads,
    longest_first, phase_groups, resolve_workers, run_flow, run_notebook)


class Progress:
    def __init__(self, **options):
        self.n = 0

    def set_postfix_str(self, label):
        pass

    def update(self, count):
        self.n += count

    def close(self):
        pass


class Recorder:
    """A fake notebook runner: sleeps, records its interval and thread, fails for the named runs"""

    def __init__(self, sleep=0.05, failing=()):
        self.sleep, self.failing = sleep, set(failing)
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, entry, *paths):
        start = time.monotonic()
        time.sleep(self.sleep)
        with self.lock:
            self.calls.append({"run": entry["run"], "start": start, "end": time.monotonic(),
                               "thread": threading.get_ident()})
        if entry["run"] in self.failing:
            raise RuntimeError(f"failed {entry['run']}")

    def interval(self, run):
        return next((c["start"], c["end"]) for c in self.calls if c["run"] == run)

    def max_overlap(self):
        events = sorted([(c["start"], 1) for c in self.calls] + [(c["end"], -1) for c in self.calls])
        current = best = 0
        for _, change in events:
            current += change
            best = max(best, current)
        return best


def entries(*specs):
    """entries("a:run", "b:run", "c") - run name and optional phase"""
    result = []
    for spec in specs:
        run, _, phase = spec.partition(":")
        entry = {"name": run, "notebook": "x/N.ipynb", "experiment": "e", "run": run, "creation_style": "exist-ok"}
        if phase:
            entry["phase"] = phase
        result.append(entry)
    return result


class TestParallelFlow(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        patcher = mock.patch.dict("os.environ", {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.directory.cleanup()

    def run_queue(self, queue, runner, **options):
        return run_flow(queue, self.root, self.root / "results", self.root, notebook_runner=runner,
                        progress_factory=Progress, **options)

    def test_one_worker_runs_in_queue_order_in_the_main_thread(self):
        runner = Recorder(sleep=0.01)
        queue = entries("a:run", "b:run", "c:run")
        self.run_queue(queue, runner)
        self.assertEqual([c["run"] for c in runner.calls], ["a", "b", "c"])
        self.assertEqual(runner.max_overlap(), 1)
        self.assertEqual({c["thread"] for c in runner.calls}, {threading.get_ident()})

    def test_entries_without_a_phase_run_serially_with_any_workers(self):
        runner = Recorder(sleep=0.02)
        self.run_queue(entries("a", "b", "c"), runner, workers=4)
        self.assertEqual([c["run"] for c in runner.calls], ["a", "b", "c"])
        self.assertEqual(runner.max_overlap(), 1)

    def test_a_phase_runs_concurrently_within_the_worker_limit(self):
        runner = Recorder(sleep=0.1)
        execution = self.run_queue(entries(*[f"r{i}:run" for i in range(6)]), runner, workers=3)
        self.assertEqual(len(runner.calls), 6)
        self.assertEqual(runner.max_overlap(), 3)
        self.assertEqual(execution["workers"], 3)
        self.assertGreater(execution["wall-seconds"], 0)

    def test_no_entry_of_a_later_phase_starts_before_the_earlier_phase_ends(self):
        runner = Recorder(sleep=0.05)
        queue = entries("p1:prepare", "p2:prepare", "r1:run", "r2:run", "r3:run", "agg:aggregate", "f1:figure",
                        "f2:figure")
        self.run_queue(queue, runner, workers=4)
        phases = [["p1", "p2"], ["r1", "r2", "r3"], ["agg"], ["f1", "f2"]]
        for earlier, later in zip(phases, phases[1:]):
            end = max(runner.interval(run)[1] for run in earlier)
            start = min(runner.interval(run)[0] for run in later)
            self.assertLessEqual(end, start)

    def test_repeated_non_consecutive_phases_are_separate_groups(self):
        groups = phase_groups(entries("a:x", "b:x", "c:y", "d:x", "e", "f"))
        self.assertEqual([(phase, [e["run"] for e in group]) for phase, group in groups],
                         [("x", ["a", "b"]), ("y", ["c"]), ("x", ["d"]), (None, ["e"]), (None, ["f"])])

    def test_workers_by_phase_and_auto(self):
        self.assertEqual(resolve_workers({"run": 5, "default": 2}, "run"), 5)
        self.assertEqual(resolve_workers({"run": 5, "default": 2}, "figure"), 2)
        self.assertEqual(resolve_workers({"run": 5}, "figure"), 1)
        self.assertGreaterEqual(resolve_workers("auto"), 1)
        with self.assertRaises(ValueError):
            resolve_workers(0)

    def test_stop_starts_nothing_after_a_failure_and_attaches_all_failures(self):
        runner = Recorder(sleep=0.05, failing={"r0", "r1"})
        queue = entries(*[f"r{i}:run" for i in range(8)], "agg:aggregate")
        with self.assertRaisesRegex(RuntimeError, "failed r") as caught:
            self.run_queue(queue, runner, workers=2)
        ran = {c["run"] for c in runner.calls}
        self.assertNotIn("agg", ran)
        self.assertLessEqual(len(ran), 4)
        self.assertEqual({entry["run"] for entry, _ in caught.exception.flow_failures}, {"r0", "r1"})
        self.assertIsNotNone(caught.exception.flow_execution["wall-seconds"])

    def test_continue_finishes_the_phase_and_stops_before_the_next(self):
        runner = Recorder(sleep=0.02, failing={"r0"})
        queue = entries(*[f"r{i}:run" for i in range(6)], "agg:aggregate")
        with self.assertRaisesRegex(RuntimeError, "failed r0"):
            self.run_queue(queue, runner, workers=2, on_failure="continue")
        self.assertEqual({c["run"] for c in runner.calls}, {f"r{i}" for i in range(6)})

    def test_entries_of_a_phase_with_the_same_result_directory_are_rejected(self):
        runner = Recorder()
        queue = entries("a:run", "a:run")
        with self.assertRaisesRegex(ValueError, "same result directory"):
            self.run_queue(queue, runner, workers=2)
        self.assertEqual(runner.calls, [])

    def test_a_parallel_flow_limits_the_threads_but_keeps_the_users_setting(self):
        import os
        os.environ.pop("OPENBLAS_NUM_THREADS", None)
        os.environ["OMP_NUM_THREADS"] = "4"
        self.run_queue(entries("a:run", "b:run"), Recorder(sleep=0.01), workers=2)
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
        self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")

    def test_limit_threads_sets_only_the_missing_variables(self):
        environ = {"MKL_NUM_THREADS": "8"}
        set_now = limit_threads(2, environ)
        self.assertEqual(environ["MKL_NUM_THREADS"], "8")
        self.assertEqual(set(set_now), set(THREAD_VARIABLES) - {"MKL_NUM_THREADS"})

    def test_longest_first_orders_by_earlier_durations(self):
        queue = entries("short:run", "new:run", "long:run")
        for run, duration in (("short", 1.0), ("long", 9.0)):
            entry = next(e for e in queue if e["run"] == run)
            executed_notebook_path(entry, self.root).write_text(
                json.dumps({"cells": [], "metadata": {"papermill": {"duration": duration}}}))
        self.assertEqual([e["run"] for e in longest_first(queue, self.root)], ["long", "short", "new"])

    def test_the_report_gives_the_wall_clock_time_and_the_parallelism(self):
        queue = entries("a:run")
        executed_notebook_path(queue[0], self.root).write_text(
            json.dumps({"cells": [], "metadata": {"papermill": {"duration": 30.0}}}))
        report = get_flow_report(queue, self.root / "results", self.root,
                                 execution={"workers": 4, "wall-seconds": 10.0})
        self.assertEqual(report["wall-seconds"], 10.0)
        self.assertAlmostEqual(report["parallelism"], 3.0)


class TestKernelStartRetry(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.entry = entries("a")[0]
        patcher = mock.patch.object(flow.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.directory.cleanup()

    def test_a_kernel_startup_failure_is_retried(self):
        attempts = []

        def executor(*args, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("Kernel died before replying to kernel_info")

        run_notebook(self.entry, self.root, self.root, self.root, notebook_root=self.root, executor=executor,
                     kernel_name="k")
        self.assertEqual(len(attempts), 3)

    def test_a_failure_inside_the_notebook_is_not_retried(self):
        class PapermillExecutionError(Exception):
            pass

        attempts = []

        def executor(*args, **kwargs):
            attempts.append(1)
            raise PapermillExecutionError("Address already in use")  # a cell's error, whatever its text

        with self.assertRaises(PapermillExecutionError):
            run_notebook(self.entry, self.root, self.root, self.root, notebook_root=self.root,
                         executor=executor, kernel_name="k")
        self.assertEqual(len(attempts), 1)

    def test_the_cell_progress_bar_is_shown_only_in_the_main_thread(self):
        shown = []

        def executor(*args, progress_bar, **kwargs):
            shown.append(progress_bar)

        run = lambda: run_notebook(self.entry, self.root, self.root, self.root, notebook_root=self.root,
                                   executor=executor, kernel_name="k")
        run()
        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        self.assertEqual(shown, [True, False])

    def test_kernel_startup_errors(self):
        self.assertTrue(is_kernel_startup_error(RuntimeError("Kernel didn't respond in 60 seconds")))
        self.assertFalse(is_kernel_startup_error(ValueError("bad value")))


if __name__ == "__main__":
    unittest.main()
