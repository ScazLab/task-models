import sys


if sys.version_info >= (3,0):
    TemporaryDirectory = tempfile.TemporaryDirectory

else:
    import shutil
    import tempfile
    
    
    class TemporaryDirectory:
    
        def __init__(self):
            self.name = tempfile.mkdtemp()
    	
        def __enter__(self):
            return self.name
    
        def __exit__(self, type, value, traceback):
            shutil.rmtree(self.name)
