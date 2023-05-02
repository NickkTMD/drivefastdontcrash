using Ipopt
using Symbolics
using LazySets
using Polyhedra
using LinearAlgebra
using SparseArrays
using Infiltrator
using StaticArrays
using SymPy
using Polynomials

include("map.jl")

function get_points(lb::LaneBoundary) 

    if (lb.curvature == 0)

        pt_a = lb.pt_a
        pt_b = lb.pt_b

        mdpt = (pt_a[1:2] + pt_b[1:2]) / 2
        q1pt = (pt_a[1:2] + mdpt) / 2
        q3pt = (pt_b[1:2] + mdpt) / 2
    
        lowerXs = [pt_a[1] - .05, q1pt[1] - .025, mdpt[1], q3pt[1] + .025, pt_b[1] + .05]
        lowerYs = [pt_a[2], q1pt[2], mdpt[2], q3pt[2], pt_b[2]]

        return lowerXs, lowerYs
    else
        # center = nothing
        # right = lb.curvature < 0
        # radius = 1 / lb.curvature

        # if (right)
        #     center = lb.pt_d - normalize(pt_b - pt_d) * inside_rad
        # else
        #     center = lb.pt_b - normalize(pt_d - pt_b) * inside_rad
        # end

        # TODO: add more points to circle
        pt_a = lb.pt_a
        pt_b = lb.pt_b
        mdpt = (pt_a + pt_b) / 2

        lowerXs = [pt_a[1], mdpt[1], pt_b[1]]
        lowerYs = [pt_a[2], mdpt[2], pt_b[2]]
    end
end

"""
Create functions which accepts X¹, X², X³, r¹, r², r³, a¹, b¹, a², b², as input, and each return
one of the 5 callbacks which constitute an IPOPT problem: 
1. eval_f
2. eval_g
3. eval_grad_f
4. eval_jac_g
5. eval_hess_lag

Xⁱ is the vehicle_state of vehicle i at the start of the trajectory (t=0)
rⁱ is the radius of the i-th vehicle.

The purpose of this function is to construct functions which can quickly turn 
updated world information into planning problems that IPOPT can solve.
"""
# trajectory_length=40, timestep=0.2, R = Diagonal([0.1, 0.5]), 
function create_callback_generator(; map=training_map(), max_vel=10.0, trajectory_length=8, R = Diagonal([0.1, 0.5]), timestep=0.2)

    # Define symbolic variables for all inputs, as well as trajectory
   
    # State = [x, y, velocity, yaw(steering_angle)]
    # inputs = [accel, angular velo angle]

    # define variable that has to be an integer
    # @variables x::Int y::Int

    X¹, lowerLane, upperLane, target_pos, Z = let
        @variables(X¹[1:4], lowerLane[1:3], upperLane[1:3], target_pos[1:2], Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    end

    states, controls = decompose_trajectory(Z)

    all_states = [[X¹,]; states]

    # vehicle_2_prediction = constant_velocity_prediction(X², trajectory_length, timestep, tc)
    # vehicle_3_prediction = constant_velocity_prediction(X³, trajectory_length, timestep, tc)
   
    # get target pos from target id

    cost_val = sum(stage_cost(x, u, target_pos) for (x,u) in zip(states, controls))

    cost_grad = Symbolics.gradient(cost_val, Z)

    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]

    # we want to look at current segment and next segment and generate a polynomial as an upper and lower bound for our position 

    for k in 1:trajectory_length
        # trajectory must obey physics

        evolv = all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep)

        # # @info "Evolv is $evolv"

        append!(constraints_val, evolv)
        append!(constraints_lb, zeros(4)) # fuck it maybe only mostly follow physics?
        append!(constraints_ub, zeros(4))

        # lane boundaries... stay in parent or a child's

        # trajectory must obey velocity constraints
        # maybe being near center line is actually a score function sorta thing?
        # also is it better if conditions are differentiable? 

        # stay within lane polynomials

        pos = states[k][1:2]

        poly = lowerLane[1] + lowerLane[2] * pos[1] + lowerLane[3] * pos[1] * pos[1]
        poly2 = upperLane[1] + upperLane[2] * pos[1] + upperLane[3] * pos[1] * pos[1]

        # append!(constraints_val, pos[2] - poly) 
        # append!(constraints_lb, 0)
        # append!(constraints_ub, Inf)

        # append!(constraints_val, poly2 - pos[2])
        # append!(constraints_lb, 0)
        # append!(constraints_ub, Inf) 

        # @info "controls $(controls[k][1])"
        
        append!(constraints_val, controls[k][1])
        append!(constraints_lb, -5.0)
        append!(constraints_ub, 5.0) # max velo

        append!(constraints_val, controls[k][2])
        append!(constraints_lb, -1)
        append!(constraints_ub, 1) # max steering angle
    end

    constraints_jac = Symbolics.sparsejacobian(constraints_val, Z)
    (jac_rows, jac_cols, jac_vals) = findnz(constraints_jac)
    num_constraints = length(constraints_val)

    λ, cost_scaling = let
        @variables(λ[1:num_constraints], cost_scaling) .|> Symbolics.scalarize
    end

    lag = (cost_scaling * cost_val + λ' * constraints_val)
    lag_grad = Symbolics.gradient(lag, Z)
    lag_hess = Symbolics.sparsejacobian(lag_grad, Z)
    (hess_rows, hess_cols, hess_vals) = findnz(lag_hess)
    
    expression = Val{false}

    full_cost_fn = let
        cost_fn = Symbolics.build_function(cost_val, [Z;X¹;lowerLane;upperLane;target_pos]; expression)
        (Z, X¹, lowerLane, upperLane, target_pos) -> cost_fn([Z;X¹;lowerLane;upperLane;target_pos])
    end

    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [Z;X¹;lowerLane;upperLane;target_pos]; expression)[2]

        (grad, Z, X¹, lowerLane, upperLane, target_pos) -> begin
            cost_grad_fn!(grad, [Z;X¹;lowerLane;upperLane;target_pos])
        end
    end

    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [Z;X¹;lowerLane;upperLane;target_pos]; expression)[2]
        (cons, Z, X¹, lowerLane, upperLane, target_pos) -> constraint_fn!(cons, [Z;X¹;lowerLane;upperLane;target_pos])
    end

    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [Z;X¹;lowerLane;upperLane;target_pos]; expression)[2]
        (vals, Z, X¹, lowerLane, upperLane, target_pos) -> constraint_jac_vals_fn!(vals, [Z;X¹;lowerLane;upperLane;target_pos])
    end
    
    full_hess_vals_fn = let
        hess_vals_fn! = Symbolics.build_function(hess_vals, [Z;X¹;lowerLane;upperLane;target_pos;λ;cost_scaling]; expression)[2]
        (vals, Z, X¹, lowerLane, upperLane, target_pos, λ, cost_scaling) -> hess_vals_fn!(vals, [Z;X¹;lowerLane;upperLane;target_pos;λ;cost_scaling])
    end

    full_constraint_jac_triplet = (; jac_rows, jac_cols, full_constraint_jac_vals_fn)
    full_lag_hess_triplet = (; hess_rows, hess_cols, full_hess_vals_fn)

    return (; full_cost_fn, 
            full_cost_grad_fn, 
            full_constraint_fn, 
            full_constraint_jac_triplet, 
            full_lag_hess_triplet,
            constraints_lb,
            constraints_ub)
end

"""
Assume z = [U[1];...;U[K];X[1];...;X[K]]
Return states = [X[1], X[2],..., X[K]], controls = [U[1],...,U[K]]
where K = trajectory_length
"""
function decompose_trajectory(z)
    K = Int(length(z) / 6)
    controls = [@view(z[(k-1)*2+1:k*2]) for k = 1:K]
    states = [@view(z[2K+(k-1)*4+1:2K+k*4]) for k = 1:K]
    return states, controls
end

function compose_trajectory(states, controls)
    K = length(states)
    z = [reduce(vcat, controls); reduce(vcat, states)]
end

"""
The physics model used for motion planning purposes.
Returns X[k] when inputs are X[k-1] and U[k]. 
Uses a slightly different vehicle model than presented in class for technical reasons.
"""
function evolve_state(X, U, Δ)

    # X is state [x, y, velocity, angle]
    # U is controls [acceleration, angular velo/yaw rate]

    V = X[3] + Δ * U[1] 
    θ = X[4] + Δ * U[2]
    X + Δ * [V*cos(θ), V*sin(θ), U[1], U[2]]
end

"""
Cost at each stage of the plan
"""
function stage_cost(X, U, target_pos)
    # for now, higher velocity is better

    # penalize being far from target location

    # targDistPenalty = 0.5 * norm(X[1:2] - target_pos[1:2])^2

    return -2 * U[1]^2 + 0.1 * U[2]^2 - X[3] # + targDistPenalty
end

# Don't call this function until we know where we are
function generate_trajectory(starting_state, path, target_id, callbacks; trajectory_length=8, map=training_map())
    X1 = starting_state

    @info "STARTING STATE: $X1"

    target_pos = nothing

    if (target_id in path)
        # we can get to dest
         
        target_seg = map[target_id]
        target_pos = target_seg.lane_boundaries[2].pt_a + target_seg.lane_boundaries[2]
        target_seg.lane_boundaries[3].pt_a + target_seg.lane_boundaries[3].pt_b
        
        target_pos = target_pos / 4.0
    else
        # if our target is not in range, we go to the next seg end. We may need a longer path
        target_seg = map[path[2]]
        target_pos = (target_seg.lane_boundaries[1].pt_b + target_seg.lane_boundaries[2].pt_b) / 2
    end

    # use lane boundaries of the starting segment fuck it
    starting_seg_id = path[1]
    next_seg_id = path[2] 

    starting_seg = map[starting_seg_id]
    next_seg = map[next_seg_id]

    # lower lane boundary
    firstXs, firstYs = get_points(starting_seg.lane_boundaries[1])
    secondXs, secondYs = get_points(next_seg.lane_boundaries[1])
    
    lowerXs = [firstXs; secondXs]
    lowerYs = [firstYs; secondYs]

    # upper lane boundary
    firstXs, firstYs = get_points(starting_seg.lane_boundaries[length(next_seg.lane_boundaries)])
    secondXs, secondYs = get_points(next_seg.lane_boundaries[length(next_seg.lane_boundaries)])
    
    upperXs = [firstXs; secondXs]
    upperYs = [firstYs; secondYs]

    lowerLane = fit(lowerXs, lowerYs, 2)
    upperLane = fit(upperXs, upperYs, 2)

    # convert lowerLane to symbolics expr
    

    sample = starting_seg.lane_boundaries[1].pt_b[1]
    if (lowerLane(sample) > upperLane(sample))
        
        # switch
        temp = lowerLane
        lowerLane = upperLane
        upperLane = temp
    end

    lowerLane = lowerLane.coeffs
    upperLane = upperLane.coeffs

    @info "What are polys"

    # TODO refine callbacks given current positions of vehicles, lane geometry,
    # etc.
    # refine callbacks with current values of parameters / problem inputs
    wrapper_f = function(z) 
        # callbacks.full_cost_fn(z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)
        callbacks.full_cost_fn(z, X1, lowerLane, upperLane, target_pos)
    end
    wrapper_grad_f = function(z, grad)
        # callbacks.full_cost_grad_fn(grad, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)

        callbacks.full_cost_grad_fn(grad, z, X1, lowerLane, upperLane, target_pos)
    end
    wrapper_con = function(z, con)
        # callbacks.full_constraint_fn(con, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)

        callbacks.full_constraint_fn(con, z, X1, lowerLane, upperLane, target_pos)
    end
    wrapper_con_jac = function(z, rows, cols, vals)
        if isnothing(vals)
            rows .= callbacks.full_constraint_jac_triplet.jac_rows
            cols .= callbacks.full_constraint_jac_triplet.jac_cols
        else
            # callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)
            callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, z, X1, lowerLane, upperLane, target_pos)
        end
        nothing
    end
    wrapper_lag_hess = function(z, rows, cols, cost_scaling, λ, vals)
        if isnothing(vals)
            rows .= callbacks.full_lag_hess_triplet.hess_rows
            cols .= callbacks.full_lag_hess_triplet.hess_cols
        else
            # callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center, λ, cost_scaling)
            callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z, X1, lowerLane, upperLane, target_pos, λ, cost_scaling)
        end
        nothing
    end

    n = trajectory_length*6
    m = length(callbacks.constraints_lb)

    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(-Inf, n),
        fill(Inf, n),
        length(callbacks.constraints_lb),
        callbacks.constraints_lb,
        callbacks.constraints_ub,
        length(callbacks.full_constraint_jac_triplet.jac_rows),
        length(callbacks.full_lag_hess_triplet.hess_rows),
        wrapper_f,
        wrapper_con,
        wrapper_grad_f,
        wrapper_con_jac,
        wrapper_lag_hess
    )

    controls = repeat([zeros(2),], trajectory_length)
    states = repeat([starting_state,], trajectory_length)
    zinit = compose_trajectory(states, controls)

    @info "Check zinit"

    prob.x = zinit

    @info "Solving with IPOPT"

    Ipopt.AddIpoptIntOption(prob, "print_level", 1)

    status = Ipopt.IpoptSolve(prob)

    states, controls = decompose_trajectory(prob.x)

    if status == 0
        @info "Ipopt found soln " * string(prob.x)
    else
        @warn "Problem not cleanly solved. IPOPT status is $(status)."
        @info "Controls are $(controls) and states are $(states)"
    end

    (; states, controls, status)
end