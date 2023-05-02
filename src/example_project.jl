struct SimpleVehicleState
    p1::Float64
    p2::Float64
    θ::Float64
    v::Float64
    l::Float64
    w::Float64
    h::Float64
end

struct FullVehicleState
    position::SVector{3, Float64}
    velocity::SVector{3, Float64}
    orientation::SVector{3, Float64}
    angular_vel::SVector{3, Float64}
end

struct MyLocalizationType
    last_update::Float64
    x::FullVehicleState
end

struct MyPerceptionType
    last_update::Float64
    x::Vector{SimpleVehicleState}
end

function test_algorithms(gt_channel,
        localization_state_channel,
        perception_state_channel, 
        ego_vehicle_id)
    estimated_vehicle_states = Dict{Int, Tuple{Float64, Union{SimpleVehicleState, FullVehicleState}}}
    gt_vehicle_states = Dict{Int, GroundTruthMeasuremen}

    t = time()
    while true

        while isready(gt_channel)
            meas = take!(gt_channel)
            id = meas.vehicle_id
            if meas.time > gt_vehicle_states[id].time
                gt_vehicle_states[id] = meas
            end
        end

        latest_estimated_ego_state = fetch(localization_state_channel)
        latest_true_ego_state = gt_vehicle_states[ego_vehicle_id]
        if latest_estimated_ego_state.last_update < latest_true_ego_state.time - 0.5
            @warn "Localization algorithm stale."
        else
            estimated_xyz = latest_estimated_ego_state.position
            true_xyz = latest_true_ego_state.position
            position_error = norm(estimated_xyz - true_xyz)
            t2 = time()
            if t2 - t > 5.0
                @info "Localization position error: $position_error"
                t = t2
            end
        end

        latest_perception_state = fetch(perception_state_channel)
        last_perception_update = latest_perception_state.last_update
        vehicles = last_perception_state.x

        for vehicle in vehicles
            xy_position = [vehicle.p1, vehicle.p2]
            closest_id = 0
            closest_dist = Inf
            for (id, gt_vehicle) in gt_vehicle_states
                if id == ego_vehicle_id
                    continue
                else
                    gt_xy_position = gt_vehicle_position[1:2]
                    dist = norm(gt_xy_position-xy_position)
                    if dist < closest_dist
                        closest_id = id
                        closest_dist = dist
                    end
                end
            end
            paired_gt_vehicle = gt_vehicle_states[closest_id]

            # compare estimated to GT

            if last_perception_update < paired_gt_vehicle.time - 0.5
                @info "Perception upate stale"
            else
                # compare estimated to true size
                estimated_size = [vehicle.l, vehicle.w, vehicle.h]
                actual_size = paired_gt_vehicle.size
                @info "Estimated size error: $(norm(actual_size-estimated_size))"
            end
        end
    end

function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end
        
        # process measurements

        localization_state = MyLocalizationType(0,0.0)
        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        latest_localization_state = fetch(localization_state_channel)
        
        # process bounding boxes / run ekf / do what you think is good

        perception_state = MyPerceptionType(0,0.0)
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

using Rotations

function get_current_state(latest_gt::GroundTruthMeasurement)
    pos = latest_gt.position

    angle = latest_gt.orientation
    yaw = QuaternionToYaw(angle)
    velo = latest_gt.velocity
 
    [pos[1], pos[2], velo[1], yaw]
end

target_map_segment = 24 # -1 # (not a valid segment, will be overwritten by message)

function decision_making(localization_state_channel, 
        perception_state_channel, 
        gt_channel, # for testing
        map, 
        socket)
    # do some setup

    # wait 5 seconds for location to be set
    @info "Waiting 5 to get accurate location info"
    sleep(1) # TODO Wait 5

    timestep=0.5
    
    @info "Creating callback gen"
    callbacks = create_callback_generator(max_vel=5.0, timestep=timestep, trajectory_length=4)

    # get initial state
    latest_gt = fetch(gt_channel)
    state = get_current_state(latest_gt)

    # get our first trajectory
    lastPath = []
    trajectory = nothing
    current_step = 1
    helicopterMode = false

    # Continuously generate new trajectories and save them
    errormonitor(@async while true
        # get latest state
        latest_gt = fetch(gt_channel)
        state = get_current_state(latest_gt)

        curr_seg_id = nothing
        for seg_id in lastPath
            if inside_segment(state[1:2], map[seg_id])
                curr_seg_id = seg_id
                break
            end
        end

        if (curr_seg_id === nothing)

            current_segments = get_segments(map, state[1:2])

            if (length(current_segments) == 0)
                @info "Not on map, houston we are fucked"

                @info "INITIALIZING HELICOPTER MODE"
                helicopterMode = true

                curr_seg_id = lastPath[1]
            else

                # grab any element from map

                for (k, v) in current_segments
                    curr_seg_id = k
                    break
                end
            end
        end

        # we are not on the path, so we need to find a new path
        @info "Finding new path"
        lastPath = find_path(curr_seg_id, target_map_segment)

        if (length(lastPath) == 1)
            lastPath = [lastPath;lastPath]
        end

        if (length(lastPath) == 0)
            sleep(1)
            continue
        end

        trajectory = generate_trajectory(state, lastPath[1:2], target_map_segment, callbacks, map=map)

        sleep(2 * timestep)

        # could always sleep

        current_step = 1
    end
    )

    alternate = 1

    while true

        @info "loop "
        
        sleep(timestep)

        if (helicopterMode)

            @info "Helicopter mode"

            alternate = -alternate

            cmd = VehicleCommand(10, alternate, true)
            serialize(socket, cmd)

        else
            # No trajectory calculated yet
            if (trajectory === nothing) 
                @info "Waiting on trajectory..."
            else
                states, controls = decompose_trajectory(trajectory)

                @info "Decomposed trajectory"

                if length(states) >= current_step
                    # apply the current step controls
                    target_angle = controls[current_step][2]
                    target_velo = states[current_step][3]

                    @info "Sending command... $target_angle, $target_velo"

                    cmd = VehicleCommand(target_angle, target_velo, true)
                    serialize(socket, cmd)

                    current_step+=1
                end     
            end
        end
    end

end

# this is prob wrong but we rolling with it for now
function QuaternionToYaw(q::SVector{4, Float64} = SVector(1.0, 0.0, 0.0, 0.0))
    # roll (x-axis rotation)
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = atan(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = sqrt(1 + 2 * (q.w * q.y - q.x * q.z))
    cosp = sqrt(1 - 2 * (q.w * q.y - q.x * q.z))
    pitch = 2 * atan(sinp, cosp) - π / 2

    # yaw (z-axis rotation)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = atan(siny_cosp, cosy_cosp)

    (yaw)
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end

function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.training_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{GroundTruthMeasurement}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)

    # target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)


        sleep(0.001)
        local measurement_msg
        received = false
        while true

            # get ping from server
            

            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment

        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements

            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    @async localize(gps_channel, imu_channel, localization_state_channel)
    @async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map, socket)
    @async test_algorithms(gt_channel, localization_state_channel, perception_state_channel, ego_vehicle_id)
end
