# define the PIDNN module
module NeuralPID
    const dt = 0.01
    include("PIDNN/controller.jl")
    include("PIDNN/system.jl")
    include("PIDNN/gradients.jl")
    include("PIDNN/control_loop.jl")

    export PIDNN
    export SystemState, State2Ord, observe, update
    export control_loop
    export gradients
end