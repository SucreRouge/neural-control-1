const dt = 0.1

# This function calculates the gradient of the 
# loss function with respect to the control value using the difference quotient.
function control_grad_DQ(t::Integer, states::Vector, controls::Vector{Float64}, setpoints::Vector{Float64})
    if t < 2
        return 0.0
    end

    # get the relevant data
    x_n = observe(states[t])
    x_p = observe(states[t-1])

    v_n = controls[t]
    v_p = controls[t-1]

    y_n = setpoints[t]

    # calculate ∂x/∂v with difference quotient.
    dxdv = sign((x_n - x_p) / (v_n - v_p))
    δ = -(y_n - x_n) * dxdv

    return δ
end

# This function calculates the gradient of the 
# loss function with respect to the control value in Taylor approximation.
function control_grad_Taylor(t::Integer, states::Vector, controls::Vector{Float64}, setpoints::Vector{Float64})
    if t < 1
        return 0.0
    end

    # get the relevant data
    x_n = observe(states[t])
    y_n = setpoints[t]

    return -(y_n - x_n)
end

# This function calculates the gradient of the 
# loss function with respect to the control value in Taylor approximation.
function control_grad_RegTaylor(t::Integer, states::Vector, controls::Vector{Float64}, setpoints::Vector{Float64})
    if t < 2
        return 0.0
    end

    # get the relevant data
    x_n = observe(states[t])
    y_n = setpoints[t]
    v_n = controls[t]
    v_p = controls[t-1]

    return -(y_n - x_n) + 0.1*(v_p)
    #return 0.5*(v_p)
end

function control_grad_zero(t::Integer, states::Vector, controls::Vector{Float64}, setpoints::Vector{Float64})
    return 0.0
end



function control_loop(state, setpoint, controller, t0, te, gradient_source)
    cstate = initial_state(controller)

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
    catch
        for i in last+1:length(times)
            states[i]    = state
        end
    end

    return states, controls, setpoints, weights
end
