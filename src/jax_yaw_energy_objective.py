"""Differentiable yaw-motion energy, explicitly conditional on electrical power."""
def motion_energy_kwh(yaw_trace, initial_yaw, *, xp, power_kw=30., rate_deg_s=.3, smoothing_deg=1e-4):
    """Constant-speed energy surrogate. Exact movement-time scoring is separate.

    yaw_trace shape: [steps,batch,turbines]. Returns mean batch kWh.
    Smooth abs is zero at rest; its bias is bounded by epsilon per increment.
    """
    if power_kw < 0 or rate_deg_s <= 0 or smoothing_deg <= 0:
        raise ValueError('invalid energy model parameters')
    prior=xp.concatenate((initial_yaw[None],yaw_trace[:-1]),axis=0)
    delta=yaw_trace-prior
    distance=xp.sqrt(delta*delta+smoothing_deg*smoothing_deg)-smoothing_deg
    return power_kw*xp.mean(xp.sum(distance,axis=(0,2)))/(rate_deg_s*3600.)


def net_mean_power_mw(result, initial_yaw, *, xp, horizon_s=480., power_kw=30.):
    if horizon_s<=0:
        raise ValueError('horizon must be positive')
    energy=motion_energy_kwh(result.yaw_deg,initial_yaw,xp=xp,power_kw=power_kw)
    return result.mean_farm_power_mw-energy*3.6/horizon_s
