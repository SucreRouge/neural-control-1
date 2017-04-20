#
#   This script performs the comparison between the analytical solution and a numerical
#   solution for the L2 error of a PD controlled system of 2nd order
#

const pspace = linspace(0.5, 10.0, 100)

function analytic_loss(a, y, D, P)
    aD = a-D
    return -y*y/(4*aD) *(aD*aD/P + 1)
end

function numerical_loss(a, y, D, P, dt)
    state = 0.0
    velo  = 0.0
    err   = 0.0
    for t in 0:dt:1000
        force  = a * velo + P*(y - state) - D * velo
        state += 0.5*dt*dt*force + velo * dt
        velo  += dt * force
        err   += 0.5*(state - y)^2 *dt
    end
    return err
end

function compare(a, y, D, P)
    result = Float64[]
    for dt in linspace(1e-4, 1.0, 100)
        nl = numerical_loss(a, y, D, P, dt)
        push!(result, nl)
    end
    return result, analytic_loss(a, y, D, P)
end

res, a = compare(1.0, 1.0, 2.0, 1.5)

function compare_fixed_dt(a, y, D, dt)
    result_nl = Float64[]
    result_al = Float64[]
    for P in pspace
        nl = numerical_loss(a, y, D, P, dt)
        al = analytic_loss(a, y, D, P)
        push!(result_nl, nl)
        push!(result_al, al)
    end
    return result_nl, result_al
end

nl, al = compare_fixed_dt(1.0, 1.0, 2.0, 0.1)
nl2, al = compare_fixed_dt(1.0, 1.0, 2.0, 0.01)

writedlm("data/compare_l2.txt", [pspace al nl nl2])

