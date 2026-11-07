"""File identities and bounded subprocess execution shared by reference tools."""

import hashlib
import os
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, directory, input_path, log_path, environment, timeout):
    start = time.perf_counter()
    with input_path.open("rb") as inp, log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=environment,
            stdin=inp,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            raise
    elapsed = time.perf_counter() - start
    if code:
        raise RuntimeError(f"Command failed ({code}): {log_path}")
    return elapsed
