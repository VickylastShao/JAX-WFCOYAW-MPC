"""Pure LES-MPC oracle experiment; never imports a lookup control policy."""
import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path
from contract import *

D = Path(os.environ["NET_MPC_CONFIG"])


def main(args):
    root = args.root
    p = json.loads((D/'PROTOCOL.json').read_text())
    allocation = json.loads((D/'ALLOCATION.json').read_text())
    a = json.loads((D/'GPU_ALIGNMENT.json').read_text())
    from pure_les_mpc_mainline_gate import adjudicate_preexperiment_alignment
    gate = adjudicate_preexperiment_alignment(a, mainline_path=root/p['mainline'])
    require(gate['authorized_to_launch'] and gate['status'] == 'pass', 'mainline gate denied')
    require(sha(D/'PROTOCOL.json') == a['input_scope']['protocol_sha256'], 'protocol binding')
    require(sha(D/'ALLOCATION.json') == a['resource_scope']['allocation_sha256'], 'allocation binding')
    require(allocation['confirmed'] is True and allocation['host'] == 'CB-H20'
            and allocation['gpu_index'] == 1 and allocation['experiment_id'] == p['experiment_id'],
            'this batch has no matching GPU allocation')
    for name, digest in p['bound_files'].items():
        require(sha(root/name) == digest, 'source/input mismatch: '+name)
    require(os.environ.get('PAPER2_CONTAINER_DIGEST') == p['image'], 'image binding')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'visible device mismatch')

    import jax
    import jaxlib
    import jax.numpy as jnp
    import numpy as np
    from controller_side_case_setup import block_tree
    from jax_mole_compact_sac_benchmark import (
        make_scaled_mole_training_case, build_flow_functions, make_initial_flow,
        _apply_velocity_boundaries, FlowState)
    from jax_mole_layouts import canonical_scale_layouts
    from jax_controller_side_mpc import rate_limited_target_step
    from jax_shared_les_yaw_rollout import make_shared_les_yaw_rollout
    from jax_yaw_net_energy import net_mean_power_mw
    from jax_pure_projected_optimizers import make_projected_adam_optimizer
    from mole_plane_archive import PlaneArchive
    from run_full_g2_unit import sha256_tree, sha256_array
    require(jax.default_backend() == 'gpu' and len(jax.devices()) == 1, 'one GPU required')
    require((jax.__version__, jaxlib.__version__, np.__version__) == ('0.9.0.1','0.9.0.1','2.3.5'),
            'runtime versions differ')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'GATE.json', gate)
    save(args.output/'RUNTIME.json', {'jax':jax.__version__, 'jaxlib':jaxlib.__version__,
         'numpy':np.__version__, 'python':platform.python_version(),
         'device':str(jax.devices()[0]), 'image':p['image'], 'protocol_sha256':sha(D/'PROTOCOL.json')})
    case = make_scaled_mole_training_case(canonical_scale_layouts()[9], batch=1, rhs_assembly='tensor')
    f = build_flow_functions(case, periodic_x=False, include_turbines=True)
    zero = jnp.zeros((1,9), dtype=jnp.float32)
    zero_weights = f['disk_weights'](zero)
    timings, executables, pairs = [], {}, []

    def compiled(fn, key, *values):
        if key not in executables:
            started = time.perf_counter()
            executables[key] = jax.jit(fn).lower(*values).compile()
            timings.append({'kind':'JIT', 'key':key, 'seconds':time.perf_counter()-started})
        return executables[key]

    def execute(fn, key, *values):
        executable = compiled(fn, key, *values)
        started = time.perf_counter()
        value = block_tree(executable(*values))
        timings.append({'kind':'synchronized_execution', 'key':key, 'seconds':time.perf_counter()-started})
        return value

    def initialize(farm, planes):
        def step(current, inlet): return f['advance'](current, zero, zero_weights, inlet)[0], None
        return jax.lax.scan(step, farm, jnp.swapaxes(planes,0,1))[0]

    def plant(farm,yaw,target,planes):
        return make_shared_les_yaw_rollout(case,f,steps=planes.shape[1])(farm,yaw,target,planes)

    def diagnostics(state,power,angles):
        div=f['diagnostic_divergence'](f['divergence'](state.velocity))[:,3:-3,3:-3]
        cfl=jnp.max(jnp.abs(state.velocity[:,0])*DT/case.dx+jnp.abs(state.velocity[:,1])*DT/case.dy+jnp.abs(state.velocity[:,2])*DT/case.dz)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((state,power,angles))]))
        return finite,cfl,jnp.sqrt(jnp.mean(div**2))

    prediction=make_shared_les_yaw_rollout(case,f,steps=HORIZON_STEPS)
    delay=make_shared_les_yaw_rollout(case,f,steps=DELAY_STEPS)
    def objective(target,state,planes):
        flow, yaw = state
        result = prediction(flow,yaw,target,planes)
        # Explicitly optimize electrical power; Ecomp's angle penalty is not used.
        return net_mean_power_mw(result,yaw,xp=jnp,horizon_s=480.,power_kw=30.), result

    def project(target, yaw):
        bounded = jnp.clip(target,-30.,30.)
        delta = bounded-yaw
        scale = jnp.minimum(1.,90./jnp.maximum(jnp.sum(jnp.abs(delta),axis=-1,keepdims=True),1e-12))
        return yaw+scale*delta

    optimizer = make_projected_adam_optimizer(jax.value_and_grad(objective,has_aux=True),
        projector=project, update_count=4, command_dimension=9, step_scale=1.)

    def control(warm, farm, yaw, active, planes):
        delayed = delay(farm,yaw,active,planes[:,:DELAY_STEPS])
        solution = optimizer(warm, delayed.final_yaw_deg,
                             (delayed.farm,delayed.final_yaw_deg),planes[:,DELAY_STEPS:])
        return delayed, solution

    def energy(power): return float(np.asarray(power,dtype=np.float64).sum()*DT/3.6e9)

    def compare(predicted, actual):
        maximum = 0.
        for x,y in zip(jax.tree.leaves(predicted),jax.tree.leaves(actual)):
            x,y = np.asarray(x),np.asarray(y)
            if x.dtype.kind in 'iu': require(np.array_equal(x,y), 'integer state mismatch')
            else: maximum = max(maximum,float(np.max(np.abs(x-y)))/max(1.,float(np.max(np.abs(y)))))
        require(maximum <= 1e-5, 'prediction/plant state mismatch')
        return maximum

    def advance_checked(flow,yaw,target,planes):
        raw = execute(plant,'plant'+str(planes.shape[1]),flow,yaw,target,planes)
        qc = execute(diagnostics,'diagnostics'+str(planes.shape[1]),raw.farm,raw.power_watts,raw.yaw_deg)
        result = (raw.farm,raw.final_yaw_deg,raw.power_watts,raw.yaw_deg,*qc)
        require(bool(result[4]) and float(result[5])<1 and float(result[6])<.1, 'plant numerical failure')
        angles = np.asarray(result[3])[:,0,:]
        previous = np.concatenate((np.asarray(yaw),angles[:-1]),axis=0)
        require(np.max(np.abs(angles)) <= 30.0001, 'executed yaw bound')
        require(np.max(np.abs(angles-previous))/DT <= .3001, 'executed yaw rate')
        return result

    qualification = p['stage'] == 'qualification'
    for c in p['cells']:
        cid = c['unit']+'__'+c['family']
        out = args.output/cid
        out.mkdir()
        archive = PlaneArchive(args.inputs/c['unit']/c['family']/'true')
        b = c['binding']
        require(archive.manifest_sha256==b['manifest'] and archive.manifest['archive_payload_sha256']==b['payload'],
                'true archive binding mismatch')
        archive.verify(hashes=True)
        validate_coverage(archive.completed_planes)
        host = np.ascontiguousarray(archive.read(0,INITIALIZATION_STEPS),dtype=np.float32)
        require(sha256_array(host)==b['initial_planes'], 'initial planes binding')
        planes = jnp.asarray(host[None])
        initial = make_initial_flow(case,f,c['seed'])
        initial = initial._replace(velocity=_apply_velocity_boundaries(initial.velocity,planes[:,0],periodic_x=False))
        initial = execute(initialize,'initialize',initial,planes)
        require(sha256_tree(initial)==b['initial_state'], 'initial state binding')
        mpc, baseline, yaw, active, warm = initial, initial, zero, zero, zero
        records, mpc_power, zero_power, yaw_trace = [], [], [], []
        representative = jnp.asarray(archive.read(INITIALIZATION_STEPS,3300)[None])
        if qualification:
            from qualification_checks import check
            save(args.output/'NUMERICAL_CHECKS.json', check(jax,jnp,np,objective,delay,mpc,yaw,active,representative,block_tree))
        controller = compiled(control,'controller_H480',warm,mpc,yaw,active,representative)
        # Compile both plant segment shapes before entering online timing.
        compiled(plant,'plant900',mpc,yaw,active,representative[:,:900])
        compiled(plant,'plant100',mpc,yaw,active,representative[:,900:1000])
        for t in (DECISIONS[:2] if qualification else DECISIONS):
            started = time.perf_counter()
            start, stop = forecast_interval(t)
            host = np.ascontiguousarray(archive.read(start,stop-start),dtype=np.float32)
            packet = jnp.asarray(host[None])
            delayed, solution = block_tree(controller(warm,mpc,yaw,active,packet))
            candidate = np.asarray(solution.best_targets,dtype=np.float32)
            delayed_yaw = np.asarray(delayed.final_yaw_deg,dtype=np.float32)
            finite = bool(solution.all_finite)
            provisional = time.perf_counter()-started
            reasons = command_check(candidate.reshape(-1).tolist(),delayed_yaw.reshape(-1).tolist(),provisional,finite)
            applied = jnp.asarray(candidate) if not reasons else active
            block_tree(applied)
            elapsed = time.perf_counter()-started
            if elapsed > 180 and 'deadline' not in reasons:
                reasons.append('deadline'); applied=active
            save(out/f'command_{t:04d}.json',{'controller_seconds':elapsed,'fallback_reasons':reasons,'candidate_deg':candidate.reshape(-1).tolist(),'start_state_sha256':sha256_tree(mpc),'actual_yaw_deg':np.asarray(yaw).reshape(-1).tolist(),'active_target_deg':np.asarray(active).reshape(-1).tolist()})
            # Auditing is outside the online controller, never influences selection.
            truth_host = np.ascontiguousarray(archive.read(start,DECISION_STEPS),dtype=np.float32)
            require(np.array_equal(host[:DECISION_STEPS],truth_host), 'oracle and plant inlet mismatch')
            true_packet = jnp.asarray(truth_host[None])
            first = advance_checked(mpc,yaw,active,true_packet[:,:900])
            consistency = compare(delayed.farm,first[0])
            yaw_error = float(np.max(np.abs(np.asarray(delayed.final_yaw_deg)-np.asarray(first[1]))))
            require(yaw_error<=1e-4, 'delay yaw mismatch')
            delay_e = float(delayed.mean_farm_power_mw)*180/3600
            delay_error = abs(delay_e-energy(first[2]))/max(abs(energy(first[2])),1e-12)
            require(delay_error<=1e-5, 'delay energy mismatch')
            second = advance_checked(first[0],first[1],applied,true_packet[:,900:1000])
            zfirst = advance_checked(baseline,zero,zero,true_packet[:,:900])
            zsecond = advance_checked(zfirst[0],zero,zero,true_packet[:,900:1000])
            row = {'decision_seconds':t, 'controller_seconds':elapsed,
                   'objective_initial_mw':float(solution.initial_value),
                   'objective_best_mw':float(solution.best_value),
                   'objective_gradient_evaluations':int(solution.objective_gradient_evaluations),
                   'candidate_deg':candidate.reshape(-1).tolist(), 'applied_deg':np.asarray(applied).reshape(-1).tolist(),
                   'fallback_reasons':reasons, 'delay_state_error':consistency,
                   'delay_yaw_error_deg':yaw_error, 'delay_energy_relative_error':delay_error,
                   'inlet_start':start, 'inlet_stop':stop,
                   'forecast_sha256':sha256_array(host), 'applied_inlet_sha256':sha256_array(truth_host),
                   'mpc_window_mwh':energy(first[2])+energy(second[2]),
                   'noyaw_window_mwh':energy(zfirst[2])+energy(zsecond[2]),
                   'segment_end_cfl_max':max(float(r[5]) for r in (first,second,zfirst,zsecond)),
                   'segment_end_divergence_rms_max':max(float(r[6]) for r in (first,second,zfirst,zsecond))}
            if t == 0 and not reasons:
                replay = advance_checked(first[0],first[1],applied,packet[:,900:])
                predicted = solution.best_aux
                row['h480_state_error'] = compare(predicted.farm,replay[0])
                predicted_e = float(predicted.mean_farm_power_mw)*480/3600
                row['h480_energy_relative_error'] = abs(predicted_e-energy(replay[2]))/max(abs(energy(replay[2])),1e-12)
                require(row['h480_energy_relative_error']<=1e-5,'H480 energy mismatch')
            # Check the optimizer's chosen value against the exact shared net model.
            chosen = solution.best_aux
            chosen_hours = float(np.abs(np.diff(np.concatenate((np.asarray(delayed.final_yaw_deg,dtype=np.float64)[None],np.asarray(chosen.yaw_deg,dtype=np.float64))),axis=0)).sum())/(.3*3600.)
            chosen_net = float(np.asarray(chosen.power_watts,dtype=np.float64).sum(axis=-1).mean()/1e6)-30*chosen_hours*3.6/480
            row['predicted_motion_energy_kwh'] = 30*chosen_hours
            row['exact_objective_audit_mw'] = chosen_net
            row['objective_accounting_abs_error_mw'] = abs(chosen_net-float(solution.best_value))
            row['optimizer_update_values_mw'] = np.asarray(solution.update_values).tolist()
            require(row['objective_accounting_abs_error_mw'] <= 5e-6, 'objective accounting mismatch')
            records.append(row)
            save(out/f'decision_{t:04d}.json',row)
            require(not reasons,'technical stop after recorded fallback: '+','.join(reasons))
            require(row['objective_gradient_evaluations']==5,'optimizer budget mismatch')
            require(row['objective_best_mw']+1e-6>=row['objective_initial_mw'],'optimizer best-value regression')
            for r in (first,second): mpc_power.append(np.asarray(r[2])[:,0,:]);yaw_trace.append(np.asarray(r[3])[:,0,:])
            for r in (zfirst,zsecond): zero_power.append(np.asarray(r[2])[:,0,:])
            mpc,yaw,baseline,active,warm = second[0],second[1],zsecond[0],applied,applied
            print(json.dumps({'cell':cid,'decision':t,'seconds':elapsed,'paired_window_mwh':row['mpc_window_mwh']-row['noyaw_window_mwh']}),flush=True)
        if qualification:
            save(args.output/'RESULT.json', {'status':'pass','decisions':records,'timings':timings,'protocol_sha256':sha(D/'PROTOCOL.json')})
            return
        tail = jnp.asarray(archive.read(INITIALIZATION_STEPS+9000,2400)[None])
        mr = advance_checked(mpc,yaw,active,tail)
        zr = advance_checked(baseline,zero,zero,tail)
        mpc_power.append(np.asarray(mr[2])[:,0,:]);zero_power.append(np.asarray(zr[2])[:,0,:]);yaw_trace.append(np.asarray(mr[3])[:,0,:])
        mp,zp,yt = map(np.concatenate,(mpc_power,zero_power,yaw_trace))
        require(mp.shape==zp.shape==yt.shape==(11400,9),'scoring trace length mismatch')
        delta = np.diff(np.concatenate((np.zeros((1,9)),yt)),axis=0)
        reversal=0
        for column in delta.T:
            signs=np.sign(column[np.abs(column)>1e-6]);reversal+=int(np.sum(signs[1:]!=signs[:-1]))
        row={'cell':cid,'mpc_mwh':energy(mp),'noyaw_mwh':energy(zp),'technical_pass':True,
             'main_mpc_mwh':energy(mp[:9000]),'main_noyaw_mwh':energy(zp[:9000]),
             'tail_mpc_mwh':energy(mp[9000:]),'tail_noyaw_mwh':energy(zp[9000:]),
             'row_mpc_mwh':[energy(mp[:,i:i+3]) for i in (0,3,6)],
             'row_noyaw_mwh':[energy(zp[:,i:i+3]) for i in (0,3,6)],
             'yaw_travel_deg':float(np.abs(delta).sum()),'yaw_reversals':reversal,
             'maximum_abs_yaw_deg':float(np.max(np.abs(yt))), 'maximum_yaw_rate_deg_per_s':float(np.max(np.abs(delta))/DT),
             'saturation_fraction':float(np.mean(np.abs(yt)>=29.9999)),
             'max_controller_seconds':max(r['controller_seconds'] for r in records),
             'initial_state_sha256':sha256_tree(initial), 'mpc_final_state_sha256':sha256_tree(mr[0]),
             'noyaw_final_state_sha256':sha256_tree(zr[0])}
        np.savez_compressed(out/'TRACES.npz',mpc_power_watts=mp,noyaw_power_watts=zp,mpc_yaw_deg=yt)
        for name,state in [('MPC',mr[0]),('NOYAW',zr[0])]:
            np.savez(out/(name+'_FINAL_STATE.npz'),**{k:np.asarray(v) for k,v in zip(FlowState._fields,state)})
        save(out/'RESULT.json',row);pairs.append(row)
        save(out/'TIMINGS.json',timings)
    result={'adjudication':adjudicate(pairs),'pairs':pairs,'timing':timings,
            'protocol_sha256':sha(D/'PROTOCOL.json'),'evidence_class':p['evidence_class']}
    save(args.output/'RESULT.json',result)
    save(args.output/'SEAL.json',{'result_sha256':sha(args.output/'RESULT.json')})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for key in ('root','inputs','output'): parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    try: main(args)
    except Exception as error:
        if args.output.exists() and not (args.output/'FAILURE.json').exists():
            save(args.output/'FAILURE.json',{'status':'failed','type':type(error).__name__,'message':str(error)})
        traceback.print_exc()
        raise SystemExit(1)
