import time
import sys
import contextlib
from io import StringIO

class VerboseContext:
    def __init__(self, name) -> None:
        self.name = name
        self._stdout = None
        self._buffer = None
    
    def __enter__(self):
        print(self.name, end="... ")
            
        self._stdout = sys.stdout
        self._buffer = StringIO()
        sys.stdout = self._buffer
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        print()
        print(self._buffer.getvalue(), end="")
        
        if exc_type is not None:
            print(f"failed: {exc_value}")
        
class TimerContext(VerboseContext):
    def __enter__(self, *args):
        super().__enter__(*args)
        self.start_time = time.monotonic()

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.write(f"Elapsed time: {time.monotonic() - self.start_time:.2f}s")
        super().__exit__(exc_type, exc_value, traceback)