import importlib.util
import sys
from pathlib import Path

import pytest

REQUIRED_MODULES = [
    "numpy",
    "librosa",
    "faster_whisper",
    "moviepy",
    "PIL",
    "customtkinter",
]
missing_modules = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
if missing_modules:
    skip_reason = "Missing dependencies: " + ", ".join(missing_modules)
else:
    skip_reason = "All required dependencies are available."

pytestmark = pytest.mark.skipif(bool(missing_modules), reason=skip_reason)


def load_song_module():
    module_path = Path(__file__).resolve().parent.parent / "songGemini_final_v9.0_multiprocess.py"
    spec = importlib.util.spec_from_file_location("song_module_for_tests", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create module spec for songGemini_final_v9.0_multiprocess.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, spec.name


def test_main_uses_dummy_app():
    module, module_name = load_song_module()
    events = []

    class DummyApp:
        def __init__(self):
            events.append("init")

        def mainloop(self):
            events.append("mainloop")

    original_app = getattr(module, "App", None)
    original_name = module.__name__
    try:
        module.App = DummyApp
        module.__name__ = "__main__"
        module.main()
    finally:
        module.__name__ = original_name
        if original_app is not None:
            module.App = original_app
        sys.modules.pop(module_name, None)

    assert events == ["init", "mainloop"]
