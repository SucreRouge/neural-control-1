type Config
    λ::Real
    r::Real
end

function error(config, t)
    λ = config.λ
    return exp(-λ * t) * cos(t)
end

function error_sin(config, t)
    λ = config.λ
    return exp(-λ * t) * sin(t)
end

function errord(config, t)
    λ = config.λ
    #return (error(config, t+0.001) - error(config, t)) / 0.001
    return -λ * error(config, t) - error_sin(config, t)
end

function errordd(config, t)
    λ = config.λ
    return (λ^2 - 1) * error(config, t) +2 * λ * error_sin(config, t)
end

function signum(config, t)
    λ = config.λ
    r = config.r
    ed = errord(config, t)
    if ed != 0
        sg = sign(1 + r * errordd(config, t)/ed)
    else
        sg = 0
    end
    return sg
end

function change(config, dvdw)
    c = 0.0
    et = 100
    steps = 0:0.005:et
    for t in steps
        d = -error(config, t) * signum(config, t) * dvdw(config, t)
        c += d
    end
    return c / length(steps) * et
end

for r in [0.5, 1, 2]
    lambdas = linspace(0.0, 3.0, 200)
    resultsp = Float64[]
    resultsd = Float64[]

    for lambda in lambdas
        cfg = Config(lambda, r)
        push!(resultsp, change(cfg, error))
        push!(resultsd, change(cfg, errord))
    end

    writedlm("wcc$r.txt", [lambdas resultsp resultsd])
end
