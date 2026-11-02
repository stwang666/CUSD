"""
Merging method implementations.

"""
from .task_arithmetic import merge as ta_merge

METHODS = {
    "ta": ta_merge,
}
