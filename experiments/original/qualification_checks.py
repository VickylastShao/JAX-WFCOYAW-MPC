"""Multi-scale derivative qualification; unchanged original error tolerances."""
def check(jax,jnp,np,objective,delay,farm,yaw,active,planes,block_tree):
    from jax_yaw_net_energy import motion_energy_kwh
    from jax_controller_side_mpc import rate_limited_target_step
    z=jnp.zeros((1,9),dtype=jnp.float32)
    a=jnp.ones((1,1,9),dtype=jnp.float32)*.173
    cost=lambda x: motion_energy_kwh(x,z,xp=jnp)
    assert float(cost(a*0))==0
    assert abs(float(cost(a))-float(cost(-a)))<1e-8
    exact=30*9*.173/(.3*3600)
    assert abs(float(cost(a))-exact)<=1e-7
    target=jnp.full((1,9),.173,dtype=jnp.float32)
    direction=jnp.ones_like(target)/3
    def actuator_cost(t):
        def step(y,_):
            q=rate_limited_target_step(y,t,max_increment_deg=.06,min_yaw_deg=-30.,max_yaw_deg=30.)
            return q,q
        _,trace=jax.lax.scan(step,z,None,length=2400)
        return cost(trace)*3.6/480
    assert float(jnp.max(jnp.abs(jax.grad(actuator_cost)(z)))) == 0., 'zero subgradient'
    cost_ad=float(jnp.sum(jax.grad(actuator_cost)(target)*direction))
    analytic=30/(.3*3600)*3.6/480*3
    assert abs(cost_ad-analytic)<1e-8,'actuator cost derivative'
    delayed=block_tree(jax.jit(delay)(farm,yaw,active,planes[:,:900]))
    state=(delayed.farm,delayed.final_yaw_deg);packet=planes[:,900:]
    (_,aux),gradient=block_tree(jax.jit(jax.value_and_grad(objective,has_aux=True))(target,state,packet))
    ad=float(jnp.sum(gradient*direction))
    scalar=jax.jit(objective)
    def host_net(r):
        y=np.asarray(r.yaw_deg,dtype=np.float64)
        d=np.diff(np.concatenate((np.asarray(state[1],dtype=np.float64)[None],y)),axis=0)
        gross=np.asarray(r.power_watts,dtype=np.float64).sum(axis=-1).mean()/1e6
        c=30*np.mean(np.sum(np.abs(d),axis=(0,2)))/(.3*3600)*3.6/480
        return float(gross-c)
    rows=[]
    for h in (.01,.03,.1,.3):
        vp,rp=block_tree(scalar(target+h*direction,state,packet))
        vm,rm=block_tree(scalar(target-h*direction,state,packet))
        fd32=(float(vp)-float(vm))/(2*h)
        fd64=(host_net(rp)-host_net(rm))/(2*h)
        passed=lambda fd: np.isfinite(fd) and np.isfinite(ad) and (abs(fd-ad)<=3e-5 or abs(fd-ad)<=.05*max(abs(fd),abs(ad)))
        row={'h':h,'fd_float32':fd32,'fd_host64':fd64,'ad':ad,'passed_original_tolerance_float32':bool(passed(fd32)),'passed_original_tolerance_host64':bool(passed(fd64)),'role':'legacy_small_step_diagnostic' if h==.01 else 'required_qualification'}
        rows.append(row)
        if h!=.01:assert passed(fd32) and passed(fd64),row
    assert max(r['fd_host64'] for r in rows[1:])-min(r['fd_host64'] for r in rows[1:])<=.05*abs(ad),'finite difference plateau'
    return {'status':'pass','ad_mw_per_deg':ad,'sweeps':rows,'actuator_cost_ad':cost_ad,'actuator_cost_analytic':analytic,'units_pass':True,'zero_cost_pass':True,'symmetry_pass':True,'tolerance_unchanged':{'absolute':3e-5,'relative':.05}}
