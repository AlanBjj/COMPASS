"""TRACE — Type-Routed Adaptive Consistency.

Unlike self-consistency, which uses ONE global aggregation operator over k samples, TRACE
selects the aggregation operator BY QUESTION TYPE: different question types require different
notions of "agreement as evidence". v1 ships two operators (math majority-by-number, and a
reasoning operator with a yes/no verdict vote and an open-ended STANCE vote), routed by
COMPASS's existing gate/route type.
"""

from .trace import TraceResult, trace_aggregate

__all__ = ["trace_aggregate", "TraceResult"]
