"""Process CPU/RSS and device GPU telemetry, sampled every 200 ms."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil


class Monitor:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()
        self.stages = {}
        self.current = "startup"
        self.rows = []
        self.stop = threading.Event()
        self.gpu = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None
        self.started = time.perf_counter()
        self.process.cpu_percent()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            row = dict(t=time.perf_counter() - self.started, stage=self.current,
                       cpu_core_percent=self.process.cpu_percent(), rss_mb=self.process.memory_info().rss / 1e6,
                       gpu_global_percent=None, gpu_process_mb=None)
            if self.gpu is not None:
                try:
                    row["gpu_global_percent"] = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu
                    row["gpu_process_mb"] = sum(p.usedGpuMemory for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.gpu)
                                                 if p.pid == os.getpid()) / 1e6
                except Exception:
                    pass
            self.rows.append(row)
            self.stop.wait(.2)

    @contextmanager
    def stage(self, name):
        previous = self.current
        self.current = name
        start = time.perf_counter()
        cpu = self.process.cpu_times()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            endcpu = self.process.cpu_times()
            self.stages[name] = dict(seconds=elapsed, cpu_seconds=endcpu.user + endcpu.system - cpu.user - cpu.system)
            self.current = previous
            self.save()

    def save(self):
        (self.out / "timings.json").write_text(json.dumps(self.stages, indent=2))

    def finish(self):
        self.stop.set()
        self.thread.join()
        import csv
        if self.rows:
            with (self.out / "resources.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=self.rows[0])
                writer.writeheader()
                writer.writerows(self.rows)
        summary = {}
        for stage in ["all"] + list(self.stages):
            rows = self.rows if stage == "all" else [r for r in self.rows if r["stage"] == stage]
            entry = {"samples": len(rows)}
            for key in ("cpu_core_percent", "rss_mb", "gpu_global_percent", "gpu_process_mb"):
                values = [r[key] for r in rows if r[key] is not None]
                entry[key + "_mean"] = float(np.mean(values)) if values else None
                entry[key + "_peak"] = float(max(values)) if values else None
            summary[stage] = entry
        (self.out / "resources.json").write_text(json.dumps(summary, indent=2))
        self.save()
