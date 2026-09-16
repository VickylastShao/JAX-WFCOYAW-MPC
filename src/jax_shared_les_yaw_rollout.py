"""One checkpointed LES/actuator trajectory for prediction and plant execution."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from jax_controller_side_mpc import rate_limited_target_step


class SharedYawRollout(NamedTuple):
    farm: object
    final_yaw_deg: object
    power_watts: object
    yaw_deg: object
    mean_farm_power_mw: object


def make_shared_les_yaw_rollout(case, functions, *, steps, checkpoint_steps=20):
    if steps < 1 or checkpoint_steps < 1 or steps % checkpoint_steps:
        raise ValueError('checkpoint must divide positive rollout length')
    def raw_rollout(farm, yaw, target, planes):
        if planes.shape != (case.batch,steps,3,case.ny,case.nz):
            raise ValueError('inlet shape mismatch')
        if yaw.shape != target.shape or yaw.shape != (case.batch,case.num_turbines):
            raise ValueError('actuator shape mismatch')
        target=jnp.clip(target,np.float32(-30),np.float32(30))
        def block(carry, block_index):
            def step(current, local_index):
                flow,angle=current
                angle=rate_limited_target_step(angle,target,
                    max_increment_deg=.3*float(case.dt),min_yaw_deg=-30.,max_yaw_deg=30.)
                inlet=jax.lax.dynamic_index_in_dim(planes,block_index*checkpoint_steps+local_index,axis=1,keepdims=False)
                flow,power=functions['advance'](flow,angle,functions['disk_weights'](angle),inlet)
                return (flow,angle),(power,angle)
            return jax.lax.scan(step,carry,jnp.arange(checkpoint_steps,dtype=jnp.int32))
        (final,angle),(powers,angles)=jax.lax.scan(jax.checkpoint(block,prevent_cse=False),
            (farm,yaw),jnp.arange(steps//checkpoint_steps,dtype=jnp.int32))
        powers=powers.reshape((steps,case.batch,case.num_turbines))
        angles=angles.reshape((steps,case.batch,case.num_turbines))
        mean=jnp.mean(jnp.sum(powers,axis=-1))/np.float32(1e6)
        return SharedYawRollout(final,angle,powers,angles,mean)
    @jax.custom_vjp
    def rollout(farm, yaw, target, planes):
        return raw_rollout(farm, yaw, target, planes)

    def forward(farm, yaw, target, planes):
        # Keep the primal integration graph identical to the plant graph.
        # Reconstruct the derivative tape in backward, rather than exposing
        # its residual computations to compilation of the physical forward pass.
        return raw_rollout(farm, yaw, target, planes), (farm, yaw, target, planes)

    def backward(inputs, cotangents):
        _, pullback = jax.vjp(raw_rollout, *inputs)
        return pullback(cotangents)

    rollout.defvjp(forward, backward)
    return rollout
