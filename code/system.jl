abstract SystemState

# State of a second order system
type State2Ord <: SystemState
    p::Float64
    v::Float64
end

# get state for the next time step
function update(state::State2Ord, ctrl::Float64, t::Float64)
    p = state.p + state.v * dt
    v = state.v + (ctrl - 0.1state.p) * dt
    return State2Ord(p, v)
end

# get the observable part of the state. This is what the 
# controller has to e based on.
function observe(state::State2Ord)
    return state.p
end

########################################################################################################################

# State of a second order system
type Pendulum <: SystemState
    p::Float64
    v::Float64
end

# get state for the next time step
function update(state::Pendulum, ctrl::Float64, t::Float64)
    p = state.p + state.v * dt
    cst = 1.0
    if t > 550
        cst = 2.0
    end
    es = rand() / 10
    v = state.v + (ctrl - cst * sin(state.p) + es) * dt
    return Pendulum(p, v)
end

# get the observable part of the state. This is what the 
# controller has to e based on.
function observe(state::Pendulum)
    return state.p
end