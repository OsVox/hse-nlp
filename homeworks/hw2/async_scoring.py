"""Bounded CPU scoring processes that can overlap the next GPU training arm."""
import json
import os
import subprocess
import time


class ScoringJobs:
    def __init__(self):
        self.pending = []
        self.completed = []

    def start(self, arm, stage, command, output):
        log = output.with_suffix('.scoring.log').open('w')
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        except BaseException:
            log.close()
            raise
        self.pending.append(dict(arm=arm,stage=stage,process=process,log=log,
                                 output=output,started_utc=time.time(),began=time.monotonic()))

    def collect(self, results, *, wait=False):
        for job in self.pending[:]:
            process = job['process']
            if wait:
                process.wait(timeout=max(1,120-(time.monotonic()-job['began'])))
            code = process.poll()
            if code is None:
                continue
            job['log'].close()
            self.pending.remove(job)
            if code:
                raise subprocess.CalledProcessError(code,process.args)
            results[job['arm']][job['stage']] = json.loads(job['output'].read_text())
            self.completed.append(dict(arm=job['arm'],stage=job['stage'],
                                       started_utc=job['started_utc'],collected_utc=time.time()))

    def close(self):
        for job in self.pending:
            process = job['process']
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
            job['log'].close()
        self.pending.clear()
