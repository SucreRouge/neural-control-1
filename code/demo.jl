include("NeuralPID.jl")
using NeuralPID

function make_data(file, setpoint, time)
    state    = State2Ord(0.0, 0.0)

    ctrl     = PIDNN(2.5, 0.5, 1.5, 0.0)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, gradients.DQ)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pid_data = [x v y w]

    ctrl     = PIDNN(2.5, 0.5, 1.5, 1.5)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, gradients.DQ)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pidnn_data = [x v y w]

    writedlm(file, [pid_data pidnn_data])
end

function setpoint(t)
    return 1
end

make_data("data/data_1.txt", setpoint, 16)

function setpoint(t)
    return (t/10) % 2 < 1 ? 1 : 0
end

make_data("data/data_2.txt", setpoint, 60)