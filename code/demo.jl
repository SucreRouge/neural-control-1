include("experiment.jl")
include("system.jl")
include("pidnn.jl")

function make_data(file, setpoint, time)
    state    = State2Ord(0.0, 0.0)

    ctrl     = PIDNN(2.5, 0.1, 1.5, 0.0)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, control_grad_DQ)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pid_data = [x v y w]

    ctrl     = PIDNN(2.5, 0.1, 1.5, 2.0)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, control_grad_DQ)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pidnn_data = [x v y w]



    writedlm(file, [pid_data pidnn_data])
end

function setpoint(t)
    return 1
end

make_data("data_1.txt", setpoint, 10)

function setpoint(t)
    return (t/10) % 2 < 1 ? 1 : 0
end

make_data("data_2.txt", setpoint, 60)