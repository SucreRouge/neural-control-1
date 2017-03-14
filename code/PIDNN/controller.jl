# Simple code for a PIDNN controller

################################################################################
#                                   types
################################################################################

# This class represents a PIDNN. It includes the input matrix
# (currently constant) and the output vector (P, I, D coefficients)
# as well as a saved learning rate. 
# It is implemented as a stateless object (in the sense that the control
# algorithm does not change any internal status), its state is saved
# as a PIDNN_State object that is passed as a separate parameter when
# needed.
type PIDNN
    W::Matrix{Float64}
    O::Matrix{Float64}

    learning_rate::Float64
    fixed_I::Nullable{Float64}
end

# This class represents the current, internal state of a PIDNN (PID even).
type PIDNN_State
    integral::Float64
    last_error::Float64
    out::Vector{Float64}
end

################################################################################
#                           constructor functions
################################################################################
function PIDNN(p, i, d, lr)
    return PIDNN(Float64[1.0 -1.0; 1.0 -1.0; 1.0 -1.0], Float64[p i d], lr, Nullable{Float64}())
end

function initial_state(P::PIDNN)
    return PIDNN_State(0,0, Float64[0,0,0])
end

################################################################################
#                           control and learning
################################################################################

function forward(P::PIDNN, state::PIDNN_State, d::Real, c::Real)
    # forward pass through the PIDNN. This calculates a new internal state and the current control signal.

    v = Float64[d, c]
    h = P.W * v                                                                 # this gives the generalized error vector
    ni = state.integral + h[2] * dt                                             # calculate error integral
    o = Float64[h[1], ni, (h[3] - state.last_error)/dt]                         # P, I, D terms

    newstate = PIDNN_State(ni, h[3], o)                                         # updated state

    control = P.O * o                                                           # new control signal

    return Float64(control[1]), newstate
end

# modifies P
function backprop!(P::PIDNN, state::PIDNN_State, err::Real)
    grad = err * state.out'
    P.O = P.O - P.learning_rate * dt * grad
    # if we fix the I value, reset that here.
    P.O[2] = get(P.fixed_I, P.O[2])
    return grad
end
