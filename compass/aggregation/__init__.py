"""TRACE — Type-Routed Adaptive Consistency.

Unlike self-consistency, which uses ONE global aggregation operator over k samples, TRACE
selects the aggregation operator BY QUESTION TYPE: different question types require different
notions of "agreement as evidence". Two operators: a math operator (majority by final number) and
a reasoning operator (a yes/no verdict vote, and CPC for open-ended answers), routed by the answer
type.
"""

from .trace import TraceResult, trace_aggregate

__all__ = ["trace_aggregate", "TraceResult"]
