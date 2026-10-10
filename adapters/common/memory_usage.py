"""Load the same native sampler used by the Rust adapters, only in memory replays."""

import ctypes as ct
import json
from pathlib import Path
import sys


class MemorySampler:
    def __init__(self, backend):
        name = {"darwin": "libphysmark_core.dylib", "win32": "physmark_core.dll"}.get(
            sys.platform, "libphysmark_core.so"
        )
        path = Path(__file__).resolve().parents[2] / "target/release" / name
        self.lib = ct.CDLL(str(path))
        self.lib.physmark_memory_new.argtypes = [ct.c_char_p]
        self.lib.physmark_memory_new.restype = ct.c_void_p
        self.lib.physmark_memory_sample.argtypes = [ct.c_void_p, ct.c_char_p]
        self.lib.physmark_memory_sample.restype = None
        self.lib.physmark_memory_finish.argtypes = [ct.c_void_p, ct.c_size_t]
        self.lib.physmark_memory_finish.restype = ct.c_void_p
        self.lib.physmark_memory_free.argtypes = [ct.c_void_p]
        self.lib.physmark_memory_free.restype = None
        self.handle = self.lib.physmark_memory_new(backend.encode())

    def sample(self, phase):
        self.lib.physmark_memory_sample(self.handle, phase.encode())

    def finish(self, steps):
        ptr = self.lib.physmark_memory_finish(self.handle, steps)
        self.handle = None
        try:
            return json.loads(ct.string_at(ptr))
        finally:
            self.lib.physmark_memory_free(ptr)

    def __del__(self):
        # Stop the native sampling thread if engine setup or stepping raises.
        if getattr(self, "handle", None):
            self.finish(0)
