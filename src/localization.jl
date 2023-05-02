mutable struct Particle 
    Position::Vector{Float64}
    Quaternion::Quaternion
    Velocity::Vector{Float64}
    AngularVelocity::Vector{Float64}
    weight::Float64
end

function rigid_body_dynamics_particle(particle::Particle, Δt)
    noise_dist = Distributions.Normal(0, 0.1);
    r = particle.AngularVelocity
    mag = norm(r)

    if mag < 1e-5
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*Δt / 2.0)
        vᵣ = sin(mag*Δt / 2.0) * (r / mag)
    end

    sₙ = particle.Quaternion[1]
    vₙ = particle.Quaternion[2:4]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    R = Rot_from_quat(particle.Quaternion) 

    new_position = particle.Position + Δt * particle.Velocity
    new_quaternion = Quaternion(SVector{4, Float64}(s, v[1], v[2], v[3]))
    new_velocity = particle.Velocity + rand(noise_dist, 3)
    new_angular_vel = particle.AngularVelocity + rand(noise_dist, 3)
    return Particle(new_position, new_quaternion, new_velocity, new_angular_vel, 0)
end

function gps_model(particle::Particle, gps_meas; sqrt_meas_cov = Diagonal([1.0, 1.0]), max_rate=20.0)
    T = get_gps_transform()
    gps_loc_body = T*[zeros(3); 1.0]
    # Predict measurement based on particle state
    xyz_body = particle.Position # position
    q_body = particle.Quaternion # quaternion
    Tbody = get_body_transform(q_body, xyz_body)
    xyz_gps = Tbody * [gps_loc_body; 1]
    pred_meas = xyz_gps[1:2]

    # Generate actual measurement and update particle weight
    particle_likelihood = gps_likelihood(pred_meas, gps_meas, sqrt_meas_cov)
    particle.weight *= particle_likelihood
end


function imu_model(particle::Particle, imu_meas; sqrt_meas_cov = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]), max_rate=20.0)
    T_body_imu = get_imu_transform()
    T_imu_body = invert_transform(T_body_imu)
    R = T_imu_body[1:3,1:3]
    p = T_imu_body[1:3,end]

    # Predict measurement based on particle state
    v_body = particle.Velocity # velocity
    ω_body = particle.AngularVelocity # angular_vel

    ω_imu = R * ω_body
    v_imu = R * v_body + p × ω_imu

    pred_meas = [v_imu; ω_imu]

    # Generate actual measurement and update particle weight
    particle_likelihood = likelihood(pred_meas, imu_meas, sqrt_meas_cov)
    particle.weight *= likelihood
end

function gps_likelihood(predicted_meas, actual_meas, noise_cov)
    # Compute the likelihood of the actual measurement given the predicted measurement and the measurement noise covariance matrix
    likelihood = 1.0
    lat_likelihood = pdf(Distributions.Normal(predicted_meas[1], sqrt(noise_cov[1,1])), actual_meas.lat)
    long_likelihood = pdf(Distributions.Normal(predicted_meas[2], sqrt(noise_cov[2,2])), actual_meas.long)
    likelihood *= lat_likelihood * long_likelihood
    return likelihood
end

function resample(particles)
    num_particles = length(particles)
    weights = [particle.weight for particle in particles]
    new_particles = Particle[]
    for i in 1:num_particles
        idx = Distributions.sample(1:num_particles, Weights(weights))
        push!(new_particles, deepcopy(particles[idx]))
        new_particles[i].weight = 1.0 / num_particles
    end
    return new_particles
end

function compute_state_estimate(particles::Vector{Particle})
    # Compute weighted average of particle states
    pos_mean = zeros(3)
    quat_mean = Quaternion(SVector{4, Float64}(1.0, 0.0, 0.0, 0.0))
    vel_mean = zeros(3)
    angvel_mean = zeros(3)
    weight_sum = 0.0
    for particle in particles
        pos_mean += particle.weight * particle.Position
        # quat_mean += particle.weight * [particles.Quaternion[1], particles.Quaternion[2:4]]
        vel_mean += particle.weight * particle.Velocity
        angvel_mean += particle.weight * particle.AngularVelocity
        weight_sum += particle.weight
    end
    pos_mean /= weight_sum
    # quat_mean /= weight_sum
    vel_mean /= weight_sum
    angvel_mean /= weight_sum

    # Normalize quaternion mean
    # quat_mean = normalize(quat_mean)

    # Create state estimate
    state_estimate = Particle(pos_mean, quat_mean, vel_mean, angvel_mean, 1.0)
    return state_estimate
end

# We don't really need to store these measurements but may be useful later
function flush_channel(channel::Channel)
    meas_dict = Dict{meas::Measurement}()
    count = 0
    while isready(channel) && count < 10
        meas = take!(channel)
        measurements_dict[count+1] = meas
        count += 1
    end
end

function localization(gps_channel::Channel{GPSMeasurement}, imu_channel::Channel{IMUMeasurement}, localization_state_channel::Channel{Particle})
### Step 1: Initialize particles with Gaussian distribution centered around first GPS Measurement
    println("Entering step 1")
    println("Initializing Particles")
    num_particles = 100
    # Get initial position
    initial_pos = [0.0; 0.0; 0.0]
    println("Waiting for GPS measurement")
    wait(gps_channel)
    initial_gps_meas = take!(gps_channel)
    println("Got GPS measurement", initial_gps_meas.lat, " ", initial_gps_meas.long)
    initial_pos[1] = initial_gps_meas.lat
    initial_pos[2] = initial_gps_meas.long

    # Initial Velocity to 0
    initial_lin_vel = [0.0; 0.0; 0.0]
    initial_ang_vel = [0.0; 0.0; 0.0]

    
    println("Exiting step 1")
    # Initialize particles with Gaussian distribution centered around first GPS Measurement
    println("Entering step 2")
    num_particles = 100
    particles = []
    lat_dist = Distributions.Normal(initial_pos[1], 1)
    long_dist = Distributions.Normal(initial_pos[2], 1)
    particle_lat = rand(lat_dist, num_particles)
    particle_long = rand(long_dist, num_particles)
    for i in 1:num_particles
        particle_pos = [particle_lat[i]; particle_long[i]; 0.0]
        # seg_id = get_segment(training_map(), particle.Position)
        particle_quat = Quaternion(SVector{4, Float64}(1.0, 0.0, 0.0, 0.0))
        particle_lin_vel = initial_lin_vel
        particle_ang_vel = initial_ang_vel
        particle = Particle(particle_pos, particle_quat, particle_lin_vel, particle_ang_vel, 1.0/num_particles)
        push!(particles, particle)
    end

    t = initial_gps_meas.time
    println("Entering step 3")
    while true
        wait(gps_channel)
        # if both channels are ready, default to GPS measurement
        if isready(gps_channel) && isready(imu_channel) 
            fresh_gps_meas = take!(gps_channel)
            println("fresh gps measurement" , fresh_gps_meas.lat, " ", fresh_gps_meas.long)
            # flush_channel(gps_channel)
            Δt = fresh_gps_meas.time - t # needs to be latest time
            t = fresh_gps_meas.time
            
            # Propagate forward
            for i in 1:num_particles
                particles[i] = rigid_body_dynamics_particle(particles[i], Δt)
            end

            # Update Step
            for i in 1:num_particles
                # Predict GPS measurement
                gps_likelihood = gps_model(particles[i],fresh_gps_meas; sqrt_meas_cov = Diagonal([1.0, 1.0]), max_rate=20.0)
                particles[i].weight *= gps_likelihood
            end

        elseif isready(imu_channel)
            fresh_imu_meas = take!(imu_channel)
            println("fresh imu measurement", fresh_imu_meas.linear_vel, " ", fresh_imu_meas.angular_vel)
            # flush_channel(imu_channel)
            Δt = fresh_imu_meas.time - t # needs to be latest time
            t = fresh_imu_meas.time

            # Propagate forward
            for i in 1:num_particles
                particles[i] = rigid_body_dynamics_particle(particles[i], Δt)
            end

            # Update Step
            for i in 1:num_particles
                # Predict GPS measurement
                imu_likelihood = imu_model(particles[i], fresh_imu_meas; sqrt_meas_cov = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]), max_rate=20.0)
                particles[i].weight *= imu_likelihood
            end
        else
            continue
        end

### Step 4: Normalize particle weights
        # Normalize particle weights
        println("Entering step 4")
        weights = [p.weight for p in particles]
        weights = weights ./ sum(weights)
        for i in 1:num_particles
            particles[i].weight = weights[i]
        end
        
### Step 5: Resample particles
        # Resample particles
        println("Entering step 5")
        particles = resample(particles)
        estimated_state = compute_state_estimate(particles)
        put!(localization_state_channel, estimated_state)
        println("Estimated State: ", estimated_state)
    end
end





