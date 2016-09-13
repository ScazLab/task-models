import sys
import tempfile

if sys.version_info >= (3, 0):
    TemporaryDirectory = tempfile.TemporaryDirectory
    from queue import Queue

else:
    import shutil

    class TemporaryDirectory:

        def __init__(self):
            self.name = tempfile.mkdtemp()

        def __enter__(self):
            return self.name

        def __exit__(self, type, value, traceback):
            shutil.rmtree(self.name)

    from Queue import Queue
