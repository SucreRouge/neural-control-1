include("experiment.jl")
include("system.jl")
include("pidnn.jl")

function change(setpoint, p, d, time, update)
    state    = State2Ord(0.0, 0.0)
    ctrl     = PIDNN(p, 0.1, d, 0.0001)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, update)
    return ctrl.O[1] - p, ctrl.O[3] - d
end

function error(setpoint, p, d, time, update)
    state    = State2Ord(0.0, 0.0)
    ctrl     = PIDNN(p, 0.1, d, 0.0)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, update)
    x = map(observe, x)
    error = sum((x - y).^2)
    return error
end

function make_data(file, setpoint)
    state    = State2Ord(0.0, 0.0)
    ctrl     = PIDNN(2.5, 0.1, 1.5, 0.1)
    x, v, y, w1 = control_loop(state, setpoint, ctrl, 0, 1200, control_grad_DQ)

    ctrl     = PIDNN(8.5, 0.1, 5.5, 0.1)
    x, v, y, w2 = control_loop(state, setpoint, ctrl, 0, 1200, control_grad_DQ)

    ctrl     = PIDNN(0.5, 0.1, 2.5, 0.1)
    x, v, y, w3 = control_loop(state, setpoint, ctrl, 0, 1200, control_grad_DQ)

    pidnn_data = [w1 w2 w3]
    writedlm(file, pidnn_data)
end

function setpoint(t)
    return (t/10) % 2 < 1 ? 1 : 0
end

function make_quiver(file, update)
    res = Vector{Float64}[]
    for p in linspace(0, 10, 100)
        for d in linspace(0, 10, 100)
            dp, dd = change(setpoint, p, d, 1000, update)
            e      = error(setpoint, p, d, 100, update)
            if dp*dp + dd*dd > 0.05
                dp = 0.0/0.0
                dd = 0.0/0.0
            end
            push!(res, [p, d, dp, dd, e])
        end
    end

    res = hcat(res...)
    writedlm(file, res')
end

make_quiver("quiver.txt", control_grad_DQ)
#make_data("hires.txt", setpoint)
make_quiver("quiver_taylor.txt", control_grad_RegTaylor)