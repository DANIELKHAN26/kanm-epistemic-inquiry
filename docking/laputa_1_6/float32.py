"""Single-precision emulation of the Laputa 1.6 trust arithmetic (post hoc diagnostic).

Laputa 1.6 sets ``t_float`` to ``double`` (``Prefix.h``: ``USE_DOUBLE_FLOAT``).
However, ``TrustFunction`` hard-codes ``float`` in all of the following:
- the 49 node values;
- ``update(float belief, bool pTrue)``;
- the running node position ``r``, which is accumulated by repeated ``float`` addition;
- the trapezoid total;
- ``expectation()``.
Some sub-expressions are promoted to ``double`` by ``double`` literals (``1.0``, ``.5``, ``.333…``) and by ``TRUST_FUNCTION_RESOLUTION_INV``, which is ``1.0f / (double)48``.

Not emulated: the credence that Laputa passes to ``update`` (``Amount::v()``, which is ``0.5 + 0.5 * (lower - upper)``). Its absolute precision is about 1e-16. At credences of about 1e-10 it can therefore round to a different ``float`` than KANM's credence. This is the source of the rare 1–2 ulp trust differences that remain under emulation (see ``README.md``).

The functions below reproduce that mix of precisions, operation by operation, for the non-SSE C++ path. That path is the one compiled: the SSE path is guarded by the undefined macro ``__SSE9__``. The emulation assumes IEEE-754 round-to-nearest with no extended precision (x86-64 SSE, ``FLT_EVAL_METHOD == 0``).

**Status.** This was added *after* the deterministic docking criterion fixed in advance failed in 1 of 200 scenarios (see ``README.md``). It is a diagnostic of that failure and not part of the acceptance criteria. KANM itself keeps double precision.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

import kanm.laputa16 as _l16

_F32, _F64 = np.float32, np.float64
_N = _l16.RESOLUTION
_INV = _F64(1.0) / _F64(_N)  # TRUST_FUNCTION_RESOLUTION_INV


def observe_float32(values: tuple[float, ...], belief: float, positive: bool) -> tuple[float, ...]:
    """``TrustFunction::update`` (non-SSE C++ path) with Laputa's precisions."""
    v = [_F32(x) for x in values]
    b = _F32(belief)
    nb = _F32(_F64(1.0) - _F64(b))
    r, dr = (_F32(0.0), _F32(_INV)) if positive else (_F32(1.0), _F32(-_INV))

    def factor(r: np.float32) -> np.float64:
        # r * belief is float * float; (1.0 - r) * negBelief is promoted to double.
        return _F64(_F32(r * b)) + (_F64(1.0) - _F64(r)) * _F64(nb)

    v[0] = _F32(_F64(v[0]) * factor(r))
    total = _F32(_F64(v[0]) * 0.5)
    r = _F32(r + dr)
    for t in range(1, _N):
        v[t] = _F32(_F64(v[t]) * factor(r))
        total = _F32(total + v[t])
        r = _F32(r + dr)
    v[_N] = _F32(_F64(v[_N]) * factor(r))
    total = _F32(_F64(total) + _F64(v[_N]) * 0.5)
    if total > 0:
        scale = _F32(_F32(_N) / total)
        v = [_F32(x * scale) for x in v]
    else:
        v = [_F32(1.0)] * (_N + 1)
    return tuple(float(x) for x in v)


def expectation_float32(values: tuple[float, ...]) -> float:
    """``TrustFunction::expectation`` (non-SSE C++ path) with Laputa's precisions."""
    v = [_F32(x) for x in values]
    val, r0, rs = _F32(0.0), _F32(0.0), _F32(_INV)
    for i in range(_N):
        f0 = v[i]
        fs = _F32(v[i + 1] - f0)
        term = (_F64(_F32(r0 * f0))
                + _F64(_F32(_F32(r0 * fs) + _F32(f0 * rs))) * 0.5
                + _F64(_F32(fs * rs)) * 0.333333333333333333)
        val = _F32(_F64(val) + term)
        r0 = _F32(r0 + rs)
    return float(_F32(_F64(val) * _INV))


@contextmanager
def emulated() -> Iterator[None]:
    """Temporarily make ``Laputa16Trust`` use the single-precision arithmetic."""
    cls = _l16.Laputa16Trust
    saved = cls.observe, cls.__dict__["expected_reliability"]

    def observe(self, prior_belief: float, positive: bool):  # type: ignore[no-untyped-def]
        if not 0.0 <= float(prior_belief) <= 1.0:
            raise ValueError("prior_belief must lie in [0, 1]")
        return cls(observe_float32(self.values, prior_belief, positive))

    cls.observe = observe
    cls.expected_reliability = property(lambda self: expectation_float32(self.values))
    try:
        yield
    finally:
        cls.observe, cls.expected_reliability = saved
