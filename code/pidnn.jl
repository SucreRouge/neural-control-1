# Simple code for a PIDNN controller

################################################################
#                          types
################################################################

# This class represents a PIDNN. It includes the input matrix
# (currentl constant) and the output vector (P, I, D coefficients)
# as well as a saved learning rate. 
# It is implemented as a stateless object (in the sense that the control
# algorithm does not change any internal status), its state is saved
# as a PIDNN_State object that is passed as a separate parameter when
# needed.
type PIDNN
    W::Matrix{Float64}
    O::Matrix{Float64}

    learning_rate::Float64
end

# This class represents the current, internal state of a PIDNN (PID even).
type PIDNN_State
    it::Float64
    h3::Float64
    out::Vector{Float64}
end

###############################################################
#                   constructor functions
###############################################################
function PIDNN(p, i, d, lr)
    return PIDNN(Float64[1.0 -1.0; 1.0 -1.0; 1.0 -1.0], Float64[p i d], lr)
end

function initial_state(P::PIDNN)
    return PIDNN_State(0,0, Float64[0,0,0])
end

################################################################
#             control and learning
################################################################

function forward(P::PIDNN, state::PIDNN_State, d::Real, c::Real)
    v = Float64[d, c]
    h = P.W * v
    ni = state.it + h[2] * dt
    o = Float64[h[1], ni, (h[3] - state.h3)/dt]

    newstate = PIDNN_State(ni, h[3], o)

    return Float64((P.O * o)[1]), newstate
end

# modifies P
function backprop!(P::PIDNN, state::PIDNN_State, err::Real)
    grad = err * state.out'
    P.O = P.O - P.learning_rate * dt * grad
    P.O[2] = 0.1
    return grad
end




