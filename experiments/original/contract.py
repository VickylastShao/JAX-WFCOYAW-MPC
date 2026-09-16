"""Host-only contracts for the prospectively frozen oracle diagnostic."""
import hashlib
import json
import math
from pathlib import Path

INITIALIZATION_STEPS = 2295
DT = 0.2
DELAY_STEPS = 900
HORIZON_STEPS = 2400
DECISION_STEPS = 1000
DECISIONS = tuple(range(0, 1800, 200))
SCORED_STEPS = 11400


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forecast_interval(t):
    require(t in DECISIONS, 'decision outside frozen schedule')
    start = INITIALIZATION_STEPS + int(round(t / DT))
    return start, start + DELAY_STEPS + HORIZON_STEPS


def validate_coverage(completed_planes):
    require(completed_planes >= INITIALIZATION_STEPS + SCORED_STEPS,
            'archive does not cover the complete scoring window')
    require(all(forecast_interval(t)[1] <= completed_planes for t in DECISIONS),
            'archive does not cover every forecast')


def command_check(candidate, delayed_yaw, seconds, finite):
    reasons = []
    if len(candidate) != 9 or len(delayed_yaw) != 9:
        return ['shape']
    if not finite or not all(math.isfinite(x) for x in candidate + delayed_yaw):
        reasons.append('nonfinite')
    if not math.isfinite(seconds) or seconds > 180:
        reasons.append('deadline')
    if any(abs(x) > 30.0001 for x in candidate):
        reasons.append('yaw_bounds')
    if sum(abs(a-b) for a,b in zip(candidate, delayed_yaw)) > 90.0001:
        reasons.append('scheduled_travel')
    return reasons


def adjudicate(rows):
    require(len(rows) in (4,12), 'fixed cell count')
    require(all(r['technical_pass'] for r in rows), 'technical failure')
    return {'status':'technical_pass_only','net_energy_adjudication':'separate postprocessing',
            'independent_confirmation':False,'causal_forecast_claim':False}
