include("NeuralPID.jl")
using NeuralPID


function change(setpoint, p, d, time, update)
    state    = State2Ord(0.0, 0.0)
    ctrl     = PIDNN(p, 0.1, d, 0.0001)
    ctrl.fixed_I = 0.1
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, update)
    return w[end, 1] - p, w[end, 3] - d
end

function error(setpoint, p, d, time, update)
    state    = State2Ord(0.0, 0.0)
    ctrl     = PIDNN(p, 0.1, d, 0.0)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, update)
    x = map(observe, x)
    error = sum((x - y).^2)
    return error
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

make_quiver("data/quiver.txt", gradients.DQ)