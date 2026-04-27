"""embedding_dedupe -- deduplicate near-identical embedding records by cosine similarity.

Public surface (Python port of @mukundakatta/embedding-dedupe):

    from embedding_dedupe import dedupe, cosine

* ``dedupe(records, threshold=0.95, key='id', vector='embedding', keep='first')``
  -- collapse near-duplicate records into clusters and return one survivor per cluster.
* ``cosine(a, b)`` -- pure-Python cosine similarity helper.

Pure Python, zero runtime deps. Install the optional ``[numpy]`` extra for faster
math; the default impl uses stdlib loops.
"""

from .dedupe import (
    cosine,
    dedupe,
)

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "cosine",
    "dedupe",
]
