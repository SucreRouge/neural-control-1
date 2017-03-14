################################################################################
#                           backpropagation error
################################################################################

module gradients

using ..NeuralPID

# Different approaches for calculating the loss gradient wrt the pidnn output
# This function calculates the gradient of the 
# loss function with respect to the control value using the difference quotient.
function DQ(t::Integer, states::Vector, controls::Vector{Float64}, 
                         setpoints::Vector{Float64})
    if t < 2
        return 0.0
    end

    # get the relevant data
    x_n = observe(states[t])
    x_p = observe(states[t-1])

    v_n = controls[t]
    v_p = controls[t-1]

    y_n = setpoints[t]

    # calculate ∂x/∂v with difference quotient.
    dxdv = sign((x_n - x_p) / (v_n - v_p))
    δ = -(y_n - x_n) * dxdv

    return δ
end

# This function calculates the gradient of the 
# loss function with respect to the control value in Taylor approximation.
function Taylor(t::Integer, states::Vector, controls::Vector{Float64}, 
                             setpoints::Vector{Float64})
    if t < 1
        return 0.0
    end

    # get the relevant data
    x_n = observe(states[t])
    y_n = setpoints[t]

    return -(y_n - x_n)
end

# This function calculates the gradient of the 
# loss function with respect to the control value in Taylor approximation, 
# using additional regularization.
function RegTaylor(t::Integer, states::Vector, controls::Vector{Float64}, 
                                setpoints::Vector{Float64})
    if t < 2
        return 0.0
    end

    # get the relevant data
    x_n = observe(states[t])
    y_n = setpoints[t]
    v_n = controls[t]
    v_p = controls[t-1]

    return -(y_n - x_n) + 0.1*(v_p)
    #return 0.5*(v_p)
end

function null(t::Integer, states::Vector, controls::Vector{Float64}, 
                           setpoints::Vector{Float64})
    return 0.0
end

end