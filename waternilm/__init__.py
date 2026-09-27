"""Water disaggregation based on Ellert, Makonin and Popowich's research.

See README.md and docs/PROVENANCE.md for scientific and code attribution.
"""

from .model import Model, activity_ranges, capped_viterbi, train

__all__ = ["Model", "activity_ranges", "capped_viterbi", "train"]
