import os
import sys

# The app and training code load files using relative paths like "src/scaler.pkl",
# which only resolve correctly when running from the project root (D:\CAN_IDS).
# pytest_configure runs before test modules are imported/collected, so this must
# happen here rather than in a fixture (fixtures run too late — after collection,
# which is when "from app import app" would already try to load src/scaler.pkl).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def pytest_configure(config):
    os.chdir(PROJECT_ROOT)
    notebooks_path = os.path.join(PROJECT_ROOT, "notebooks")
    if notebooks_path not in sys.path:
        sys.path.append(notebooks_path)
    src_path = os.path.join(PROJECT_ROOT, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
