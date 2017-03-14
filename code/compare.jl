include("NeuralPID.jl")
using NeuralPID

function make_data(file, setpoint, time)
    state    = State2Ord(0.0, 0.0)

    ctrl     = PIDNN(2.5, 0.1, 1.5, 0.1)
    ctrl.fixed_I = 0.1
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, gradients.null)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pid_data = [x v y w]

    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, gradients.Taylor)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pidnnt_data = [x v y w]

    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, gradients.DQ)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pidnnr_data = [x v y w]

    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, gradients.RegTaylor)
    x = map(observe, x)
    error = sum((x - y).^2)
    println(error)

    pidnnrt_data = [x v y w]

    writedlm(file, [pid_data pidnnr_data pidnnt_data pidnnrt_data])
end

function setpoint(t)
    return (t/10) % 2 < 1 ? 1 : 0
end

make_data("data/compare_rect.txt", setpoint, 1010)

function geterr(setpoint, time, alg)
    state    = State2Ord(0.0, 0.0)
    ctrl     = PIDNN(2.5, 0.1, 1.5, 1.0)
    x, v, y, w = control_loop(state, setpoint, ctrl, 0, time, alg)
    x = map(observe, x)
    error = sum((x - y).^2)
    return error
end

#println(geterr(setpoint, 20, control_grad_zero))
#println(mean(map(x -> geterr(setpoint, 20, control_grad_rand), 1:1000)))


    