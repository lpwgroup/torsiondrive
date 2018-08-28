"""
Base import for the torsiondrive module.
"""

from ._version import get_versions as _get_versions
_versions = _get_versions()
__version__ = _versions['version']
__git_revision__ = _versions['full-revisionid']
del _get_versions, _versions
