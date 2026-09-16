"""Shared net-energy model: constant-speed motion until the target is reached.

Each LES step may contain a fractional movement interval. Motor input is zero
once the commanded yaw is reached. Startup, holding and wear costs are excluded.
"""
def motion_hours(yaw_trace, initial_yaw, *, xp, rate_deg_s=.3):
    """Mean batch total turbine movement hours for [steps,batch,turbines].

    Exact absolute movement is used in both optimization and reporting.
    The derivative at zero movement is the valid zero subgradient.
    """
    if rate_deg_s <= 0:
        raise ValueError('yaw rate must be positive')
    prior=xp.concatenate((initial_yaw[None],yaw_trace[:-1]),axis=0)
    delta=yaw_trace-prior
    distance=xp.where(delta>0,delta,xp.where(delta<0,-delta,xp.zeros_like(delta)))
    return xp.mean(xp.sum(distance,axis=(0,2)))/(rate_deg_s*3600.)


def motion_energy_kwh(yaw_trace, initial_yaw, *, xp, power_kw=30., rate_deg_s=.3):
    if power_kw < 0:
        raise ValueError('yaw input power must be nonnegative')
    return power_kw*motion_hours(yaw_trace,initial_yaw,xp=xp,rate_deg_s=rate_deg_s)


def net_mean_power_mw(result, initial_yaw, *, xp, horizon_s=480., power_kw=30.):
    if horizon_s <= 0:
        raise ValueError('horizon must be positive')
    return result.mean_farm_power_mw-motion_energy_kwh(result.yaw_deg,initial_yaw,xp=xp,power_kw=power_kw)*3.6/horizon_s
