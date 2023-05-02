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
    orientation::SVector{4, Float64}
    angular_vel::SVector{3, Float64}
end

struct MyLocalizationType
    last_update::Float64
    x::FullVehicleState
end


# camera stream is continuous, but localization is one time
# so you have to extrapolate the localization state to the 
# camera time that you want to process for
# localization is whatever it is

# channels are conveyer belts, coming in sorted
# but localization channels are just one at a time 

# Take everything off the camera channel, put in a box, 
# Sort them by time, if there are any measurements from before
# last measurement, throw that away because we're past that
# We take all of them in the box, see localization state, extend forward
# or backward, and then do EKF
# Might need to adjust EKF to only do it on most recent box

# WHat happens when thing disappears?

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

function fake_localize(gt_channel, localization_state_channel)
    while true
        sleep(0.001)
        freshest = fetch(perception_state_channel)

        localization_state = FullVehicleState(freshest.position, freshest.velocity, freshest.orientation, freshest.angular_velocity)

        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end
end 

function decision_making(localization_state_channel, 
        perception_state_channel, 
        map, 
        target_road_segment_id, 
        socket)
    # do some setup
    while true
        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
        steering_angle = 0.0
        target_vel = 0.0
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end

function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    @info socket
    @info "hello"
    msg = deserialize(socket) # Visualization info
    @info "bye"
    @info msg
    # map_segments = training_map()
    (; chevy_base) = load_mechanism()

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while isopen(socket)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            sleep(0.001)
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end

        !received && continue
        measurement_msg = deserialize(socket)
        target_map_segment = measurement_msg.target_segment
        ego_vehicle_id = measurement_msg.vehicle_id

        for meas in measurement_msg.measurements
            sleep(0.001)
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

    # @async localize(gps_channel, imu_channel, localization_state_channel)
    @async fake_localize(gt_channel, localization_state)
    @async perception(cam_channel, gt_channel, perception_state_channel) # localization_state_channel, perception_state_channel)
    @async test_perception(gt_channel, perception_state_channel, ego_vehicle_id)
    # @async decision_making(localization_state_channel, perception_state_channel, target_map_segment, map, socket)

    @info "Press 'q' at any time to terminate vehicle."
    while isopen(socket)
        sleep(0.001)
        key = get_c()
        if key == 'q'
            # terminate vehicle
            target_velocity = 0.0
            steering_angle = 0.0
            @info "Terminating Keyboard Client."
            cmd = VehicleCommand(steering_angle, target_velocity, false)
            serialize(socket, cmd)
        end
    end
end

function test_perception(gt_channel, perception_state_channel, ego_id)
    t = time()
    while true
        
        @info "testing perception"

        sleep(0.001)
        tn = time()
        freshest_gt_message = nothing
        latest_time = -Inf

        while isready!(gt_channel)
            sleep(0.001)
            @info "getting groundtruth"
            gt = take!(gt_channel)
            if gt.vehicle_id != ego_id && gt.time > latest_time
                freshest_gt_message = gt
                latest_time = gt.time
            end
        end

        @info "gt"
        @info freshest_gt_message 

        @info "perception"

        perception_state = fetch(perception_state_channel)

        @info perception_state 

        if tn - t > 1.0
            t = tn
            println(perception_state.p1 - freshest_gt_message.position[1]) 
            println(perception_state.p2 - freshest_gt_message.position[2]) 
            println(perception_state.θ - extract_yaw_from_quaternion(freshest_gt_message.orientation))
            println(perception_state.v - freshest_gt_message.velocity[1])
            # TODO: how to compare vel 
        end
    end
end
