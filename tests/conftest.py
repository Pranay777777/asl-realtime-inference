# tests/conftest.py
import pathlib
import sys

# add the project root (parent of /tests) to sys.path so "import train" works
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
