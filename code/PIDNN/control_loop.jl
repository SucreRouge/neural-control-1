function control_loop(state, setpoint, controller, t0, te, gradient_source)
    cstate     = initial_state(controller)
    controller = deepcopy(controller)

    times = t0:dt:te

    # logging data: state
    states    = Array(typeof(state), length(times))
    controls  = Array(Float64,       length(times))
    setpoints = Array(Float64,       length(times))
    weights   = Array(Float64,       (length(times), 3))
    last      = 0

    try
        for i in 1:length(times)
            sp = setpoint(times[i])
            cp = observe(state)
            c, cstate = forward(controller, cstate, sp, cp)

            states[i]    = state
            controls[i]  = c
            setpoints[i] = sp
            weights[i, :] = controller.O

            newstate = update(state, c, times[i])

            δ = gradient_source(i, states, controls, setpoints)
            backprop!(controller, cstate, δ)
            
            state = newstate
            last = i
        end
    catch(e)
        print(e)
        for i in last+1:length(times)
            states[i]    = state
        end
    end

    return states, controls, setpoints, weights
end
