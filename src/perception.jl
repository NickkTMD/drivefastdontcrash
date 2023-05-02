struct MyPerceptionType
    last_update::Float64
    x::Vector{SimpleVehicleState}
end

function f(x, Δ)
    [x.p1 + Δ*cos(x.θ)*x.v;
     x.p2 + Δ*sin(x.θ)*x.v;
     x.θ;
     x.v; 
     x.l;
     x.w;
     x.h;]
end

function jac_fx(x, Δ)
    [1.0 0 -Δ*sin(x.θ)*x.v Δ*cos(x.θ) 0 0 0
     0 1.0 Δ*cos(x.θ)*x.v Δ*sin(x.θ) 0 0 0
     0 0 1.0 0 0 0 0
     0 0 0 1.0 0 0 0 
     0 0 0 0 1.0 0 0  
     0 0 0 0 0 1.0 0
     0 0 0 0 0 0 1.0]
end

function my_convert_to_pixel(num_pixels, pixel_len, px)
    min_val = -pixel_len*num_pixels/2
    pix_id = ((px - min_val) / pixel_len)+1
    return pix_id
end

function get_cam_transform_standard_matrix(camera_id)
    R_cam_to_body = [
        0.9998 0.0 0.0199987;
        0.0 1.0 0.0;
        -0.0199987 0.0 0.9998;
    ]
    t_cam_to_body = [1.35, 1.7, 2.4]
    if camera_id == 2
        t_cam_to_body[2] = -1.7
    end

    T = [R_cam_to_body t_cam_to_body]
end

function convert_from_pixel(num_pixels, pixel_len, pix_id)
    min_val = -pixel_len*num_pixels/2
    px = min_val + (pix_id-1)*pixel_len
    return px
end

function simple_state_from_camera_projection(bbox, ego_state, camera_id; focal_len = 0.01, pixel_len = 0.001, image_width = 640, image_height = 480)
    vehicle_size = SVector(13.2, 5.7, 5.3)

    # TODO: Might not need to convert from pixels, id might be fine? 
    top = bbox[1]   # convert_from_pixel(image_height, pixel_len, bbox[1])  
    bot = bbox[2]   # convert_from_pixel(image_height, pixel_len, bbox[3]) 
    left = bbox[3]  # convert_from_pixel(image_width, pixel_len, bbox[2]) 
    right = bbox[4] # convert_from_pixel(image_width, pixel_len, bbox[4]) 
    # top = convert_from_pixel(image_height, pixel_len, bbox[1])  
    # bot = convert_from_pixel(image_height, pixel_len, bbox[3]) 
    # left = convert_from_pixel(image_width, pixel_len, bbox[2]) 
    # right = convert_from_pixel(image_width, pixel_len, bbox[4]) 
     
    x_center = (left + right) / 2
    y_center = (top + bot) / 2

    box_pixel_width = right - left
    box_pixel_height = bot - top 

    # Get depths based on similar triangles using pixel shape
    # and actual vehicle shape. Then average the width and height
    # estimates to guess actual depth from camera.
    depth_from_width = focal_len * vehicle_size[2] / box_pixel_width 
    depth_from_height = focal_len * vehicle_size[3] / box_pixel_height
    depth_avg = (depth_from_height + depth_from_width) / 2

    T_body_cam = get_cam_transform(camera_id)
    T_cam_camrot = get_rotated_camera_transform()
    T_body_camrot = multiply_transforms(T_body_cam, T_cam_camrot)
    T_world_body = get_body_transform(ego_state.orientation, ego_state.position)
    T_world_camrot = multiply_transforms(T_world_body, T_body_camrot)
    # Do not invert this since we're going camera -> world

    x_normalized = (x_center - image_width / 2) * pixel_len / focal_len
    y_normalized = (y_center - image_height / 2) * pixel_len / focal_len

    # @info "normalized"
    # @info x_normalized
    # @info y_normalized

    x_cam = x_normalized * depth_avg
    y_cam = y_normalized * depth_avg
    z_cam = depth_avg

    cam_coords = SVector(x_cam, y_cam, z_cam, 1.0)
    world_coords = T_world_camrot * cam_coords
    x_world, y_world, z_world = Tuple(world_coords)

    # @info "world"
    # @info x_world
    # @info y_world
    # @info z_world

    SimpleVehicleState(
        x_world, 
        y_world, 
        extract_yaw_from_quaternion(ego_state.orientation), 
        ego_state.velocity[1], 
        vehicle_size[1], 
        vehicle_size[2], 
        vehicle_size[3]
    ) 
end

function get_3d_bbox_corners_from_simple_vehicle_state(state)
    len = state[5]
    wid = state[6]
    hgt = state[7]
    combinations = [
        (len, wid, -hgt), 
        (len, -wid, -hgt), 
        (-len, wid, hgt), 
        (-len, -wid, hgt), 
        (len, wid, hgt), 
        (len, -wid, hgt), 
        (-len, wid, -hgt), 
        (-len, -wid, -hgt), 
    ]
    corners = [
        [state[1]; state[2]; state[7]/2;] + (1/2) * [
            cos(state[3]) -sin(state[3]) 0;
            sin(state[3]) cos(state[3]) 0;
            0 0 1;
        ] * [l; w; h;]
        for (l, w, h) in combinations
    ]

    return [coord for corner in corners for coord in corner] 
end

function _corner_projections_from_simple_state(vehicle, ego_state, camera_id; focal_len = 0.01, pixel_len = 0.001, image_width = 640, image_height = 480)
    T_body_cam = get_cam_transform_standard_matrix(camera_id)
    T_cam_camrot = get_rotated_camera_transform()

    T_body_camrot = multiply_transforms(T_body_cam, T_cam_camrot)
    
    flat_corners = get_3d_bbox_corners_from_simple_vehicle_state(vehicle)

    corners = [
        [flat_corners[i], flat_corners[i+1], flat_corners[i+2]]
        for i in 1:3:length(flat_corners) - 1
    ]

    T_world_body = get_body_transform(ego_state.orientation, ego_state.position)
    T_world_camrot = multiply_transforms(T_world_body, T_body_camrot)
    T_camrot_world = invert_transform(T_world_camrot)

    other_vehicle_corners = [T_camrot_world * [pt;1] for pt in corners]
    
    scaled_corners = [
        [focal_len * corner[1] / corner[3], focal_len * corner[2] / corner[3]] 
        for corner in other_vehicle_corners if corner[3] >= focal_len
    ]

    flattened_points = vcat(scaled_corners...) 

    return flattened_points
end

function _bounding_box_from_projected_points(flattened_points; image_width=640, image_height=480, pixel_len=0.001)
    projected_points = [
        [flattened_points[i], flattened_points[i+1]]
        for i in 1:2:length(flattened_points) - 1
    ]

    left = image_width/2
    right = -image_width/2
    top = image_height/2
    bot = -image_height/2

    for pt in projected_points
        px, py = pt
        left = min(left, px)
        right = max(right, px)
        top = min(top, py)
        bot = max(bot, py)
    end

    if top ≈ bot || left ≈ right || top > bot || left > right
        # out of frame
        # We won't reach this in jacobian computation because 
        # we terminate the EKF before then if we see this
        return nothing 
    else 
        top = my_convert_to_pixel(image_height, pixel_len, top)
        bot = my_convert_to_pixel(image_height, pixel_len, bot)
        left = my_convert_to_pixel(image_width, pixel_len, left)
        right = my_convert_to_pixel(image_width, pixel_len, right)
        return [top, left, bot, right]
    end
end

function svs_to_vec(x)
    [x.p1 x.p2 x.θ x.v x.l x.w x.h]
end 

"""
Measurement func -> map real state to measurement
"""
function h(x_vec, ego, camera_id)
    points = _corner_projections_from_simple_state(x_vec, ego, camera_id)
    return _bounding_box_from_projected_points(points) 
end

function jac_hx(x_vec, ego, camera_id)
    return ForwardDiff.jacobian(st -> h(st, ego, camera_id), x_vec)
end

struct VehicleEKF 
    μ::SimpleVehicleState
    Σ::Matrix
end

function associate_bbox_with_vehicle(meas_pred, bboxes)
    max_iou = 0
    assoc_idx = nothing
    assoc_bbox = nothing

    for (bbox_idx, bbox) in enumerate(bboxes)
        # Compute intersection over union, per 
        # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        x_left = max(meas_pred[1], bbox[1])
        y_top = max(meas_pred[2], bbox[2])
        x_right = min(meas_pred[3], bbox[3])
        y_bottom = min(meas_pred[4], bbox[4])

        if x_right < x_left || y_bottom < y_top
            continue
        end

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        pred_area = (meas_pred[3] - meas_pred[1]) * (meas_pred[4] - meas_pred[2])
        bbox_area = (bbox[3] - bbox[1]) * (bbox[4] - bbox[2])
        union_area = pred_area + bbox_area - intersection_area

        iou = intersection_area / union_area 

        if iou > max_iou
            max_iou = iou
            assoc_idx = bbox_idx
            assoc_bbox = bbox
        end
    end 

    assoc_idx, assoc_bbox
end 

struct EgoState
    position::SVector{3, Float64} # position of center of vehicle
    orientation::SVector{4, Float64} # represented as quaternion
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
end


# Function to append data to CSV file for evalauting results 
# function append_to_csv(file, data)
#     row = DataFrame(data)
# 
#     open(file, "a") do io
#         CSV.write(io, row, writeheader=false, append=true)
#     end
# end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel, gt_channel_for_test)
    # set up stuff
    vehicles = VehicleEKF[]

    # Create a CSV for tracking results
    # csv_file = "results_2.csv"
    # 
    # try
    # # Create a new CSV file with the specified headers
    #     CSV.write(csv_file, DataFrame(iteration=Int[], x=Float64[], y=Float64[]))
    # catch e
    #     @warn e
    # end

    # @info "created CSV" 

    proc_cov = [
        .01 0 0 0 0 0 0; 
        0 .01 0 0 0 0 0;
        0 0 0.2 0 0 0 0; 
        0 0 0 .1 0 0 0; 
        0 0 0 0 .001 0 0; 
        0 0 0 0 0 .001 0;  
        0 0 0 0 0 0 .001; 
    ] 
    meas_var = [
        5 0 0 0;
        0 5 0 0;
        0 0 5 0;
        0 0 0 5;
    ]

    last_time = 0.0

    iteration = 0 

    try
        while true
            sleep(0.001)

            fresh_cam_meas = []
            while isready(cam_meas_channel)
                sleep(0.001)
                meas = take!(cam_meas_channel)
                push!(fresh_cam_meas, meas)
            end

            if length(fresh_cam_meas) == 0
                continue
            end

            fresh_cam_meas = sort(fresh_cam_meas, by=x -> x.time)

            latest_ego = nothing

            while isready(localization_state_channel)
                sleep(0.001)
                latest_localization_state = fetch(localization_state_channel)
                # For testing with ground truth, assumes ego is 1 
                if latest_localization_state.vehicle_id == 1 
                    latest_ego = latest_localization_state
                    break
                end 
            end

            if latest_ego == nothing
                continue
            end

            for cam_meas in fresh_cam_meas
                sleep(0.001)
                Δ = cam_meas.time - last_time

                if Δ < 0
                    continue 
                end
                
                # TODO: update to last_update
                Δ_local = latest_ego.time- cam_meas.time 

                ego_extrap_vec = rigid_body_dynamics(
                    latest_ego.position, 
                    latest_ego.orientation, 
                    latest_ego.velocity, 
                    latest_ego.angular_velocity, # TODO: change this to angular_vel, 
                    Δ_local
                )
                
                #TODO: ego_extrap = EgoState(ego_extrap_vec[1, :], ego_extrap_vec[2,:], ego_extrap_vec[3,:], ego_extrap_vec[4,:])
                ego_extrap = EgoState(latest_ego.position, latest_ego.orientation, latest_ego.velocity, latest_ego.angular_velocity)

                unassoc_bboxes = cam_meas.bounding_boxes

                for (veh_idx, veh) in enumerate(vehicles)
                    sleep(0.001)
                    # Do EKF on current vehicles until we need measurement
                    A = jac_fx(veh.μ, Δ)

                    μ̂ = f(veh.μ, Δ)
                    Σ̂ = A*veh.Σ*A' + proc_cov

                    # μ̂_vec = svs_to_vec(μ̂)
                    pred_z = h(μ̂, ego_extrap, cam_meas.camera_id)
                    if pred_z == nothing
                        deleteat!(vehicles, veh_idx)
                        continue
                    end 

                    C = jac_hx(μ̂, ego_extrap, cam_meas.camera_id)
                    d = pred_z - C*μ̂

                    # Associate vehicle measurement prediction with bbox from meas
                    assoc_idx, assoc_bbox = associate_bbox_with_vehicle(pred_z, unassoc_bboxes)

                    if assoc_bbox != nothing
                        deleteat!(unassoc_bboxes, assoc_idx) 

                        Σ = inv(inv(Σ̂) + C'*inv(meas_var)*C)
                        μ = Σ * (inv(Σ̂) * μ̂ + C'*inv(meas_var) * (assoc_bbox - d))

                        svs = SimpleVehicleState(
                            μ[1],
                            μ[2],
                            μ[3],
                            μ[4],
                            μ[5],
                            μ[6],
                            μ[7],
                        )
                        
                        vehicles[veh_idx] = VehicleEKF(svs, Σ)
                    else 
                        deleteat!(vehicles, veh_idx)
                    end
                end

                # Look through bounding boxes that haven't been associated
                # with current vehicles
                for bbox in unassoc_bboxes 
                    sleep(0.001)
                    μ = simple_state_from_camera_projection(bbox, ego_extrap, cam_meas.camera_id)
                    Σ = proc_cov 
                    push!(vehicles, VehicleEKF(μ, Σ))
                end

                last_time = cam_meas.time
            end

            states = [veh.μ for veh in vehicles]
            perception_state = MyPerceptionType(time(), states)

            freshest_gt_message = nothing
            latest_time = -Inf

            # Testing
            while isready(gt_channel_for_test)
                sleep(0.001)
                # @info "getting groundtruth"
                gt = take!(gt_channel_for_test)
                # Assumes ego_id is 1, since this is using fake localize
                if gt.vehicle_id != 1 && gt.time > latest_time
                    freshest_gt_message = gt
                    latest_time = gt.time
                end
            end

            data = Dict("iteration" => iteration,
                        "x" => perception_state.x[1].p1,
                        "y" => perception_state.x[1].p2)

            @info data

            iteration += 1

            # Append to CSV for tracking results
            # append_to_csv(csv_file, data)

            if isready(perception_state_channel)
                take!(perception_state_channel)
            end
            put!(perception_state_channel, perception_state)
        end
    catch e
        @error "Something went wrong" exception=(e, catch_backtrace())
    end
end
