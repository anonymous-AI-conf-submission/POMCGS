include("./FSC.jl")
include("./Qlearning.jl")
using MultivariateStats
using POMDPModels
using POMDPs
using POMDPModelTools
using Clustering
using DataFrames, CSV


mutable struct LogResult
    _vec_episodes::Vector{Int64}
    _vec_evaluation_value::Vector{Float64}
    _vec_valid_value::Vector{Float64}
    _vec_unvalid_rate::Vector{Float64}
    _vec_upper_bound::Vector{Float64}
    _vec_fsc_size::Vector{Int64}
end

mutable struct ContinuousPlannerPOMCGS
    _nb_process_samples::Int64                   # default 10000
    _max_b_gap::Float64                          # default 0.05
    _max_graph_node_size::Int64                  # default 1e6
    _nb_iter::Int64                              # default 1e7
    _softmax_c::Float64                          # default 3.0
    _discount::Float64                           # default 0.9
    _nb_abstract_obs::Int64                      # default 10
    _bool_grid_state::Bool                       # default false
    _bool_PCA_observation::Bool                  # default false
    _out_dimension_PCA::Int64                    # default 2
    _state_grid_distance::Vector{Float64}        # default empty
    _max_min_r::Float64                         
    _epsilon::Float64                            # default 0.01
    _Q_learning_policy::Qlearning
    _Log_result::LogResult
    _C_star::Int64                              # default 50
    _k_a::Float64                                # default 2.0
    _alpha_a::Float64                            # default 0.5
    _bool_APW::Bool                              # default false
    _bool_discrete_obs::Bool                     # default false
    _max_planning_secs::Int64                     # default 1e5
    _nb_eval::Int64                              # default 100000
    _nb_sim::Int64                               # default 1000
    ContinuousPlannerPOMCGS(max_min_r, Q_learning_policy::Qlearning) = new(10000, 0.05, 1e6, 1e7, 3.0, 0.9, 10, false, false, 2, Vector{Float64}(), max_min_r, 0.01, Q_learning_policy, LogResult(Vector{Int64}(),Vector{Float64}(),Vector{Float64}(),Vector{Float64}(),Vector{Float64}(),Vector{Int64}()), 50, 2.0, 0.5, false, false, 1e5, 10000, 1000)
end

function InitPlannerParameters(planner::ContinuousPlannerPOMCGS,
                                nb_process_samples::Int64,
                                max_b_gap::Float64,
                                softmax_c::Float64,
                                discount::Float64,
                                nb_abstract_obs::Int64,
                                bool_grid_state::Bool,
                                state_grid_distance::Vector{Float64},
                                epsilon)
    planner._nb_process_samples = nb_process_samples
    planner._max_b_gap = max_b_gap
    planner._softmax_c = softmax_c
    planner._discount = discount
    planner._nb_abstract_obs = nb_abstract_obs
    planner._bool_grid_state = bool_grid_state
    planner._state_grid_distance = state_grid_distance
    planner._epsilon = epsilon
end


function ProcessActionWeightedParticle(bool_state_grid, 
                                        state_grid,
                                        bool_PCA::Bool,
                                        PCA_dim::Int64,
                                        pomdp, 
                                        fsc::FSC,
                                        nI::Int64,
                                        a,
                                        nb_process_samples::Int64,
                                        discount::Float64,
                                        Q_learning_policy::Qlearning,
                                        nb_abstract_obs::Int64)

    sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples = CollectSamplesAndBuildNewBeliefs(bool_state_grid, 
                                                                                                        state_grid,
                                                                                                        bool_PCA,
                                                                                                        PCA_dim,
                                                                                                        pomdp, 
                                                                                                        fsc, 
                                                                                                        nI, 
                                                                                                        a, 
                                                                                                        nb_process_samples, 
                                                                                                        nb_abstract_obs)

    # 4. Build new beliefs 
    fsc._nodes[nI]._R_action[a] = sum_R_a
    expected_future_V = 0.0
    for (key, value) in all_dict_weighted_samples
        NormalizeDict(all_dict_weighted_samples[key])
        sort!(all_dict_weighted_samples[key], rev=true, byvalue=true)
        distance_r = 0.8
        bool_search, n_nextI = SearchOrInsertBelief(fsc, all_dict_weighted_samples[key], -1, distance_r, 1)
        # bool_search, n_nextI = SearchOrInsertBelief(fsc, all_dict_weighted_samples[key], fsc._max_accept_belief_gap)
        if !bool_search
            max_value, actions, Q_actions = GetValueQMDP(fsc._nodes[n_nextI]._dict_weighted_samples, fsc._action_space, Q_learning_policy)
            ProcessActions(fsc._nodes[n_nextI], actions)
            fsc._nodes[n_nextI]._Q_action = Q_actions
            fsc._nodes[n_nextI]._V_node = max_value
        end
        fsc._eta[nI][Pair(a, key)] = n_nextI
        obs_weight = all_oI_weight[key]
        expected_future_V += (obs_weight / sum_all_weights) * fsc._nodes[n_nextI]._V_node
    end

    # --- Update Q(n, a) -----
    fsc._nodes[nI]._Q_action[a] = fsc._nodes[nI]._R_action[a] + discount * expected_future_V
    return fsc._nodes[nI]._Q_action[a]
end


function Step(fsc::FSC, pomdp, s, a, bool_state_grid::Bool, state_grid::Vector{Float64}, id_thread::Int64, map_discrete2continuous_states_threads)
   if bool_state_grid
        s = rand(fsc._map_discrete2continuous_states[s])
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        sp_vec = convert_s(Vector{Float64}, sp, pomdp)
        sp_processed = ProcessState(sp_vec, state_grid)

        if haskey(map_discrete2continuous_states_threads[id_thread], sp_processed)
            push!(map_discrete2continuous_states_threads[id_thread][sp_processed], sp)
        else
            map_discrete2continuous_states_threads[id_thread][sp_processed] = [sp]
        end
        return sp_processed, o, r
   else
        return @gen(:sp, :o, :r)(pomdp, s, a)
   end
end


function Simulate(pomdp, 
                    fsc::FSC,
                    s,
                    nI::Int64,
                    depth::Int64,
                    bool_state_grid::Bool,
                    state_grid::Vector{Float64},
                    bool_PCA::Bool,
                    PCA_dim::Int64,
                    nb_process_samples::Int64,
                    discount::Float64,
                    c::Float64,
                    k_a::Float64,
                    alpha_a::Float64,
                    C_star::Int64,
                    bool_APW::Bool,
                    Q_learning_policy::Qlearning,
                    nb_abstract_obs::Int64,
                    epsilon::Float64)



    # if (discount^depth)*(Q_learning_policy._R_max - Q_learning_policy._R_min) / ( 1 - discount) < epsilon || isterminal(pomdp, s)
    if (discount^depth)*(Q_learning_policy._R_max - Q_learning_policy._R_min) < planner._epsilon || isterminal(pomdp, s)
        return 0.0
    end

    a = rand(fsc._nodes[nI]._actions)
    if bool_APW
        a = ActionProgressiveWidening(fsc._nodes[nI], fsc._action_space, k_a, alpha_a, C_star)
    else
        a = UcbActionSelection(fsc, nI, c)
    end

    fsc._nodes[nI]._visits_node += 1
    fsc._nodes[nI]._visits_action[a] += 1


    if fsc._nodes[nI]._visits_action[a] == 1
        return ProcessActionWeightedParticle(bool_state_grid, state_grid, bool_PCA, PCA_dim, pomdp, fsc, nI, a, nb_process_samples, discount, Q_learning_policy, nb_abstract_obs)
    end

    nI_next = -1
    sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
    o_vec = convert_o(Vector{Float64}, o, pomdp)

    if bool_PCA 
        o_vec = predict(fsc._nodes[nI]._PCA_Ms[a], o_vec)
    end

    o_processed = FindMostCloseAbstractObs(o_vec, fsc._nodes[nI]._abstract_observations[a])


    pair_a_o = Pair(a, o_processed)
    nI_next = fsc._eta[nI][pair_a_o]

    # update r(n, a)
    sum_R_n_a = fsc._nodes[nI]._R_action[a] * (nb_process_samples + fsc._nodes[nI]._visits_action[a])
    sum_R_n_a += r
    fsc._nodes[nI]._R_action[a] = sum_R_n_a / (nb_process_samples + fsc._nodes[nI]._visits_action[a] + 1)


    esti_V = fsc._nodes[nI]._R_action[a] + discount * Simulate(pomdp, fsc, sp, nI_next, depth + 1,  bool_state_grid,  state_grid, bool_PCA, PCA_dim,nb_process_samples, discount, c, k_a, alpha_a, C_star, bool_APW, Q_learning_policy, nb_abstract_obs, epsilon)
    fsc._nodes[nI]._Q_action[a] = fsc._nodes[nI]._Q_action[a] + ((esti_V - fsc._nodes[nI]._Q_action[a]) / fsc._nodes[nI]._visits_action[a])
    fsc._nodes[nI]._V_node = esti_V

    return esti_V
end



function Search(pomdp, 
                b,
                dict_weighted_b::OrderedDict{Any,Float64},
                fsc::FSC,
                planner::ContinuousPlannerPOMCGS)
    # assume an empty fsc
    node_start = CreateNode(dict_weighted_b)
    ProcessActions(node_start, planner._Q_learning_policy._action_space)
    push!(fsc._nodes, node_start)
    push!(fsc._eta, Dict{Pair{Any, Int64},Int64}())
    node_start_index = 1

	sum_planning_time_secs = 0
    for i in 1:planner._nb_iter
        elapsed_time = @elapsed begin
            s = rand(b)
            # @time Simulate(pomdp, 
            Simulate(pomdp, 
                    fsc, 
                    s, 
                    node_start_index, 
                    0,
                    planner._bool_grid_state,
                    planner._state_grid_distance,
                    planner._bool_PCA_observation,
                    planner._out_dimension_PCA,
                    planner._nb_process_samples,
                    planner._discount,
                    planner._softmax_c,
                    planner._k_a,
                    planner._alpha_a,
                    planner._C_star,
                    planner._bool_APW,
                    planner._Q_learning_policy,
                    planner._nb_abstract_obs,
                    planner._epsilon)
        end    
            
        sum_planning_time_secs += elapsed_time
    
        if sum_planning_time_secs > planner._max_planning_secs
            println("Timeout reached")
            break
        end

        if i % planner._nb_sim == 0
            println("--- Iter ", i ÷ planner._nb_sim, " ---")
            # println("fsc size:", length(fsc._nodes))
            push!(planner._Log_result._vec_episodes, i)
            # estimate lower bound with full FSC policy
            elapsed_time = @elapsed begin
                avg_sim = EvaluationWithSimulationFSC(pomdp, fsc, discount(pomdp), planner._nb_eval, planner) 
                # estimate lower bound with partial FSC policy (N(n) > N*)
                U, L = EvaluateBounds(pomdp, planner._max_min_r, fsc, planner._Q_learning_policy, discount(pomdp), planner._nb_eval, planner._C_star, planner._bool_PCA_observation, planner._epsilon)
            end
            println("L and U evaluation time: $elapsed_time")
            L =  max(avg_sim, L)
            println("Upper bound value: ", U)
            println("Lower bound value: ", L)
            push!(planner._Log_result._vec_evaluation_value, L)
            push!(planner._Log_result._vec_fsc_size, length(fsc._nodes))
            push!(planner._Log_result._vec_upper_bound, U)
            if U - L < planner._epsilon
                break
            end
        end
    end
end


function ProcessState(s_vec::Vector{Float64}, state_grid::Vector{Float64})
    result = Vector{Int64}()
    for i in 1:length(s_vec)
        s_i = floor(Int64, s_vec[i] / state_grid[i])
        push!(result, s_i)
    end
    return result
end


function EvaluationWithSimulationFSC(pomdp, fsc::FSC, discount::Float64, nb_sim::Int64, planner::ContinuousPlannerPOMCGS)

    b0 = initialstate(pomdp)
    sum_r = 0.0
    sum_r_valid = 0.0
    sum_unvalid_search = 0

    num_threads = Threads.nthreads()
    sum_r_threads = zeros(Float64, num_threads)
    sum_r_valid_threads = zeros(Float64, num_threads)
    sum_unvalid_search_threads = zeros(Int64, num_threads)

    R_max = planner._Q_learning_policy._R_max
    R_min = planner._Q_learning_policy._R_min
    epsilon = planner._epsilon

    Threads.@threads for sim_i = 1:nb_sim
        id_thread = Threads.threadid()
        step = 0
        s = rand(b0)
        nI = 1
        bool_random_pi = false
        bool_sim_i_invalid = false

        while (discount^step)*(R_max - R_min) /(1 - discount)  > epsilon && isterminal(pomdp, s) == false
        # while (discount^step)*(R_max - R_min) > epsilon && isterminal(pomdp, s) == false
            if bool_random_pi == true && bool_sim_i_invalid == false
                bool_sim_i_invalid = true
                sum_unvalid_search_threads[id_thread] += 1
            end
            
            if nI == -1
                bool_random_pi = true
            elseif fsc._nodes[nI]._visits_node == 0 
                bool_random_pi = true
            end

            a = GetBestAction(fsc._nodes[1]) # should be a safe action, or a greedy action 
            if !bool_random_pi
                a = GetBestAction(fsc._nodes[nI])
            end
            
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            s = sp
            o_vec = convert_o(Vector{Float64}, o, pomdp) 
            if planner._bool_PCA_observation && haskey(fsc._nodes[nI]._PCA_Ms, a)
                o_vec = predict(fsc._nodes[nI]._PCA_Ms[a], o_vec)
            elseif planner._bool_PCA_observation && !haskey(fsc._nodes[nI]._PCA_Ms, a)
                bool_random_pi = true
            end

            abstract_oI = -1
            if haskey(fsc._nodes[nI]._abstract_observations, a)
                abstract_oI = FindMostCloseAbstractObs(o_vec, 
                fsc._nodes[nI]._abstract_observations[a])   
            end
         
            sum_r_threads[id_thread] += (discount^step) *  r 
            if bool_sim_i_invalid == false 
                sum_r_valid_threads[id_thread] += (discount^step) * r 
            end 
            if haskey(fsc._eta[nI], Pair(a, abstract_oI))
                nI = fsc._eta[nI][Pair(a, abstract_oI)]
            else
                bool_random_pi = true
            end

            step += 1
        end
    end

    for i in 1: num_threads
        sum_r += sum_r_threads[i]
        sum_r_valid += sum_r_valid_threads[i]
        sum_unvalid_search += sum_unvalid_search_threads[i]
    end
    
    ratio_unvalid_search = sum_unvalid_search / nb_sim
    avg_sum = sum_r / nb_sim
    avg_sum_valid_search = sum_r_valid / (nb_sim - sum_unvalid_search)
    # println("avg sum:", avg_sum)
    # println("avg sum valid search:", avg_sum_valid_search)
    # println("unvalid search:", ratio_unvalid_search)
    push!(planner._Log_result._vec_valid_value, avg_sum_valid_search)
    push!(planner._Log_result._vec_unvalid_rate, ratio_unvalid_search)

    return avg_sum
end


function EvaluateBounds(pomdp, R_lower_bound, fsc::FSC, Q_learning_policy::Qlearning, discount::Float64, nb_sim::Int64, C_star::Int64, bool_PCA::Bool, epsilon::Float64)
    b0 = initialstate(pomdp)
    sum_r_U = 0.0
    sum_r_L = 0.0
    R_max = Q_learning_policy._R_max
    R_min = Q_learning_policy._R_min

    num_threads = Threads.nthreads()
    sum_r_U_threads = zeros(Float64, num_threads)
    sum_r_L_threads = zeros(Float64, num_threads)

    Threads.@threads for sim_i = 1:nb_sim
        id_thread = Threads.threadid()        
        step = 0
        s = rand(b0)
        nI = 1

        # while (discount^step)*(R_max - R_min)/( 1 - discount) > epsilon && isterminal(pomdp, s) == false
        while (discount^step)*(R_max - R_min) > epsilon && isterminal(pomdp, s) == false
            # a = GetBestAction(fsc._nodes[nI]._actions, fsc._nodes[nI])
            a = GetBestAction(fsc._nodes[nI])
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            o_vec = convert_o(Vector{Float64}, o, pomdp)
            if bool_PCA && haskey(fsc._nodes[nI]._PCA_Ms, a)
                o_vec = predict(fsc._nodes[nI]._PCA_Ms[a], o_vec)
            elseif bool_PCA && !haskey(fsc._nodes[nI]._PCA_Ms, a)
                o_vec = []
            end

            abstract_obsI = FindMostCloseAbstractObs(o_vec, fsc._nodes[nI]._abstract_observations[a])
            s = sp
            if haskey(fsc._eta[nI], Pair(a, abstract_obsI)) && fsc._nodes[nI]._visits_node > C_star
                nI = fsc._eta[nI][Pair(a, abstract_obsI)]
                sum_r_U_threads[id_thread] += (discount^step) *  r 
                sum_r_L_threads[id_thread] += (discount^step) *  r 
            else
                # need check GetValueQMDP function
                max_Q = GetValueQMDP(fsc._nodes[nI]._dict_weighted_samples, Q_learning_policy)
                sum_r_U_threads[id_thread] += (discount^step)*max_Q
                sum_r_L_threads[id_thread] += (discount^step)*R_lower_bound
                break
            end
            step += 1
        end

    end

    for i in 1: num_threads
        sum_r_U += sum_r_U_threads[i]
        sum_r_L += sum_r_L_threads[i]
    end

    return sum_r_U / nb_sim, sum_r_L / nb_sim
end


function SimulationFSC(s, pomdp, fsc::FSC, discount::Float64, nI::Int64, R_lower_bound::Float64, step::Int64)
    sum_r = 0.0
    while (discount^step) > 0.01 && isterminal(pomdp, s) == false
        a = GetBestAction(fsc._nodes[nI])
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        s = sp
        o_vec = convert_o(Vector{Float64}, o, pomdp)
        if bool_PCA 
            o_vec = predict(fsc._nodes[nI]._PCA_Ms[a], o_vec)
        end
        abstract_oI = FindMostCloseAbstractObs(o_vec, 
                                                fsc._nodes[nI]._abstract_observations[a])        
        if haskey(fsc._eta[nI], Pair(a, abstract_oI))
            nI = fsc._eta[nI][Pair(a, abstract_oI)]
            sum_r += (discount^step) *  r 
        else
            sum_r += (discount^step) * R_lower_bound
            break
        end

        step += 1
    end

    return sum_r 
end


function SimulationFSC(b0, pomdp, fsc::FSC, discount::Float64, planner::ContinuousPlannerPOMCGS)

    sum_r = 0.0
    step = 0
    s = rand(b0)
    nI = 1
    while (discount^step) > 0.01 && isterminal(pomdp, s) == false
        a = GetBestAction(fsc._nodes[nI])
        println("--- step ", step , " ---")
        println("s:", s)
        println("nI:", nI)
        println("belief:", fsc._nodes[nI]._dict_weighted_samples)
        println("visit action:",fsc._nodes[nI]._visits_action)
        println("a:", a)
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        println("o:", o)
        println("r:", r)

        s = sp
        o_vec = convert_o(Vector{Float64}, o, pomdp)
        if planner._bool_PCA_observation 
            o_vec = predict(fsc._nodes[nI]._PCA_Ms[a], o_vec)
        end
        abstract_oI = FindMostCloseAbstractObs(o_vec, 
                                                fsc._nodes[nI]._abstract_observations[a])        
        if haskey(fsc._eta[nI], Pair(a, abstract_oI))
            nI = fsc._eta[nI][Pair(a, abstract_oI)]
            sum_r += (discount^step) *  r 
        else
            println("unexpected abstract_obsI:", abstract_oI)
            sum_r += (discount^step) * planner._max_min_r
            break
        end

        step += 1
    end

    return sum_r 
end

function RolloutNode(pomdp, node, action_space, depth::Int64)
    sum_result = 0.0
    for (s_key, s_weight) in node._dict_weighted_samples
        V_s_esti= s_weight*Rollout(pomdp, s_key, action_space, depth)
        sum_result += V_s_esti
    end
    
    return sum_result 
end

function Rollout(pomdp, s, action_space, depth::Int64)
    gamma = discount(pomdp)
    if (gamma^depth) < 0.01
        return 0
    end
   
    a = rand(action_space)
    sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
    action_space = collect(actions(pomdp, sp))
    
    return r + gamma*Rollout(pomdp, sp, action_space, depth+1) 
end

#  use weighted sample to accelerate the computation!
function RolloutNodeQlearning(pomdp, node::FscNode, Q_learning_policy::Qlearning, depth::Int64)
    sum_result = 0.0
    for (s_key, s_weight) in node._dict_weighted_samples
        V_s_esti= s_weight*RolloutQlearning(pomdp, s_key, Q_learning_policy, depth)
        sum_result += V_s_esti
    end
    
    return sum_result 
end

function RolloutQlearning(pomdp, s, Q_learning_policy::Qlearning, depth::Int64)
    gamma = discount(pomdp)
    return (gamma^depth)*MaxQ(Q_learning_policy, s)
end

function FindRLower(pomdp, b0, action_space)
    action_min_r = Dict{Any, Float64}()
    for a in action_space
        min_r = typemax(Float64)
        for i in 1:100
            s = rand(b0)
            step = 0
            while (discount(pomdp)^step) > 0.01 && isterminal(pomdp, s) == false
                sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
                s = sp
                if r < min_r
                    action_min_r[a] = r
                    min_r = r
                end
                step += 1
            end
        end
    end

    max_min_r = typemin(Float64)
    max_min_a = rand(action_space)
    for a in action_space
        if (action_min_r[a] > max_min_r)
            max_min_r = action_min_r[a]
            max_min_a = a
        end
    end

    return max_min_r/(1-discount(pomdp)), max_min_a
end

function InitContinuousActionPOMDP(pomdp, b0, nb_particles_b0::Int64)
    particles_b0 = []
    dict_weighted_b0 = OrderedDict{Any,Float64}()

    for i in 1:nb_particles_b0
        s = rand(b0)
        push!(particles_b0, s)
        if haskey(dict_weighted_b0, s)
            dict_weighted_b0[s] += 1.0/nb_particles_b0
        else
            dict_weighted_b0[s] = 1.0/nb_particles_b0
        end
    end

    sort!(dict_weighted_b0, byvalue=true, rev=true)

    return particles_b0, dict_weighted_b0
end

function InitPOMDP(pomdp, b0, nb_particles_b0::Int64)
    action_space = collect(actions(pomdp))
    particles_b0 = []
    dict_weighted_b0 = OrderedDict{Any,Float64}()

    for i in 1:nb_particles_b0
        s = rand(b0)
        push!(particles_b0, s)
        if haskey(dict_weighted_b0, s)
            dict_weighted_b0[s] += 1.0/nb_particles_b0
        else
            dict_weighted_b0[s] = 1.0/nb_particles_b0
        end
    end

    sort!(dict_weighted_b0, byvalue=true, rev=true)

    return particles_b0, dict_weighted_b0, action_space
end

function InitPOMDP(pomdp, b0, state_grid::Vector{Float64}, nb_particles_b0::Int64)
    action_space = collect(actions(pomdp))
    particles_b0 = []
    dict_weighted_b0 = OrderedDict{Any,Float64}()

    for i in 1:nb_particles_b0
        s = rand(b0)
        s_vec = convert_s(Vector{Float64}, s, pomdp)
        s_processed = ProcessState(s_vec, state_grid)
        push!(particles_b0, s)
        if haskey(dict_weighted_b0, s_processed)
            dict_weighted_b0[s_processed] += 1.0/nb_particles_b0
        else
            dict_weighted_b0[s_processed] = 1.0/nb_particles_b0
        end
    end

    return particles_b0, dict_weighted_b0, action_space
end

function InitElemPOMDP(pomdp, b0, state_grid::Vector{Float64}, nb_particles_b0::Int64)
    action_space = collect(actions(pomdp))
    particles_b0 = []
    init_map_d2continuous_states = Dict{Vector{Float64}, Any}()
    dict_weighted_b0 = OrderedDict{Any,Float64}()

    for i in 1:nb_particles_b0
        s = rand(b0)
        s_vec = convert_s(Vector{Float64}, s, pomdp)
        s_processed = ProcessState(s_vec, state_grid)
        push!(particles_b0, s)
        if haskey(dict_weighted_b0, s_processed)
            dict_weighted_b0[s_processed] += 1.0/nb_particles_b0
            push!(init_map_d2continuous_states[s_processed], s)
        else
            dict_weighted_b0[s_processed] = 1.0/nb_particles_b0
            init_map_d2continuous_states[s_processed] = [s]
        end
    end

    return particles_b0, init_map_d2continuous_states, dict_weighted_b0, action_space
end




function NormalizeDict(d::OrderedDict{Any, Float64})
    sum = 0.0
    for (key, value) in d
        sum += value
    end

    for (key, value) in d
        d[key] = value / sum
    end
end

function NormalizeDict(d::Dict{Int64, Float64})
    sum = 0.0
    for (key, value) in d
        sum += value
    end

    for (key, value) in d
        d[key] = value / sum
    end
end


function HeuristicNodeQ(node::FscNode, Q_learning_policy::Qlearning)
    max_value = typemin(Float64)
    for (a, value) in node._Q_action
        value = 0.0
        for (s, pb) in node._dict_weighted_samples
            value += pb*GetQ(Q_learning_policy, s, a)
        end
        if value > max_value
            max_value = value
        end
        # node._Q_action[a] = 0.1 * value
        node._Q_action[a] = value
    end
    return max_value

end


function GetValueQMDP(b::OrderedDict{Any, Float64}, actions, Q_learning_policy::Qlearning)
    max_value = typemin(Float64)
    Q_actions = Dict{Any, Float64}() 

    for a in Q_learning_policy._action_space
        temp_value = 0.0
        for (s, pb) in b
            temp_value += pb*GetQ(Q_learning_policy, s, a)
        end
        Q_actions[a] = temp_value
        if temp_value > max_value
            max_value = temp_value
        end
    end

    return max_value, Q_learning_policy._action_space, Q_actions
end

function GetValueQMDP(b::OrderedDict{Any, Float64}, Q_learning_policy::Qlearning)
    value = 0.0

    for (s, pb) in b
        value += pb*GetV(Q_learning_policy, s)
    end

    return value 

end


function CheckAllActionVisit(node::FscNode)
    for (key, value) in node._visits_action
        if value == 0
            return false
        end
    end
    return true
end






function CollectSamplesAndBuildNewBeliefs(bool_state_grid, 
                                            state_grid,
                                            bool_PCA::Bool,
                                            PCA_dim::Int64,
                                            pomdp, 
                                            fsc::FSC, 
                                            nI::Int64, 
                                            a::Any, 
                                            nb_samples::Int64,
                                            nb_abstract_obs::Int64)

    all_oI_weight = Dict{Int64, Float64}()
    all_dict_weighted_samples = Dict{Int64, OrderedDict{Any,Float64}}()
    sum_R_a = 0.0
    sum_all_weights = 0.0
    collected_obs_vec = Vector{Vector{Float64}}()
    collected_sp_w_pairs = Vector{Pair{Any, Float64}}()
    set_abstract_obs = Set{Vector{Float64}}()


    # prepare data for multi-thread computation 
    num_threads = Threads.nthreads()
    all_dict_weighted_samples_threads = Vector{Dict{Int64, OrderedDict{Any,Float64}}}()
    all_oI_weight_threads = Vector{Dict{Int64, Float64}}()
    sum_R_a_threads = zeros(Float64, num_threads)
    sum_all_weights_threads = zeros(Float64, num_threads)
    collected_obs_vec_threads = Vector{Vector{Vector{Float64}}}()
    collected_sp_w_pairs_threads = Vector{Vector{Pair{Any, Float64}}}()
    set_abstract_obs_threads = Vector{Set{Vector{Float64}}}()
    map_discrete2continuous_states_threads= Vector{Dict{Vector{Float64}, Any}}()

    for i in 1:num_threads
        push!(all_dict_weighted_samples_threads, Dict{Int64,OrderedDict{Any,Float64}}())
        push!(all_oI_weight_threads, Dict{Int64, Float64}())
        push!(collected_obs_vec_threads, Vector{Vector{Float64}}())
        push!(collected_sp_w_pairs_threads, Vector{Pair{Any, Float64}}())
        push!(set_abstract_obs_threads, Set{Vector{Float64}}())
        push!(map_discrete2continuous_states_threads, Dict{Vector{Float64}, Vector{Any}}())
    end

    all_keys = collect(keys(fsc._nodes[nI]._dict_weighted_samples))
    Threads.@threads for i in 1:length(all_keys)
        id_thread = Threads.threadid()
        # s = rand(all_keys)
        s = all_keys[i]
        w = fsc._nodes[nI]._dict_weighted_samples[s]
        nb_sim = ceil(w * nb_samples)
        w = 1.0
        for i in 1:nb_sim
            sp, o, r = Step(fsc, pomdp, s, a, bool_state_grid, state_grid, id_thread, map_discrete2continuous_states_threads)
            o_vec = convert_o(Vector{Float64}, o, pomdp)
            sum_R_a_threads[id_thread] += r*w
            sum_all_weights_threads[id_thread] += w
            push!(collected_obs_vec_threads[id_thread], o_vec)
            push!(collected_sp_w_pairs_threads[id_thread], Pair(sp, w))
        end
    end
    # merge threads data 
    for id_thread in 1:num_threads
        sum_R_a += sum_R_a_threads[id_thread]
        sum_all_weights += sum_all_weights_threads[id_thread]
        collected_obs_vec = vcat(collected_obs_vec, collected_obs_vec_threads[id_thread])
        collected_sp_w_pairs = vcat(collected_sp_w_pairs, collected_sp_w_pairs_threads[id_thread])
        if bool_state_grid
            for (k,v) in map_discrete2continuous_states_threads[id_thread]
                if !haskey(fsc._map_discrete2continuous_states, k)
                    fsc._map_discrete2continuous_states[k] = v
                end
            end
        end
    end
    sum_R_a = sum_R_a/sum_all_weights


    X = reduce(hcat, collected_obs_vec)

    # --- if a PCA is needed for dimension reduction ---
    if bool_PCA 
        PCA_M = fit(PCA, X; maxoutdim=PCA_dim)
        X = predict(PCA_M, X)
        collected_obs_vec = collect(eachcol(X))
        fsc._nodes[nI]._PCA_Ms[a] = PCA_M
    end

    # --- KMeans for observation clustering ---
    itr = 50 # parameter for kmeans
    result = kmeans(X, nb_abstract_obs; maxiter = itr)
    mu = result.centers

    for i in 1:nb_abstract_obs
        push!(set_abstract_obs, mu[:,i])
    end
 
    StoreAllAbstractObs(fsc, nI, a, set_abstract_obs)

    ## use collected samples to update belief 
    Threads.@threads for i in 1:length(collected_obs_vec)
        id_thread = Threads.threadid()
        # abstract_obsI = assignment_results[i]
        abstract_obsI = FindMostCloseAbstractObs(collected_obs_vec[i], fsc._nodes[nI]._abstract_observations[a])
        sp_w_pair = collected_sp_w_pairs[i]
        sp = sp_w_pair.first
        w = sp_w_pair.second
        if haskey(all_dict_weighted_samples_threads[id_thread], abstract_obsI)
            all_oI_weight_threads[id_thread][abstract_obsI] += w 
            if haskey(all_dict_weighted_samples_threads[id_thread][abstract_obsI], sp)
                all_dict_weighted_samples_threads[id_thread][abstract_obsI][sp] += w
            else 
                all_dict_weighted_samples_threads[id_thread][abstract_obsI][sp] = w
            end
        else
            all_dict_weighted_samples_threads[id_thread][abstract_obsI] = OrderedDict{Any,Float64}()
            all_oI_weight_threads[id_thread][abstract_obsI] = w 
            all_dict_weighted_samples_threads[id_thread][abstract_obsI][sp] = w
        end
    end


    # merge threads data 
    current_nb_abstract_obs = length(fsc._nodes[nI]._abstract_observations[a]) 
    for i in 1:current_nb_abstract_obs
        all_dict_weighted_samples[i] = OrderedDict{Int64,Float64}()
        all_oI_weight[i] = 0.0
    end

    for id_thread in 1:num_threads
        for abs_oI in 1:current_nb_abstract_obs
            if haskey(all_dict_weighted_samples_threads[id_thread], abs_oI)
                all_dict_weighted_samples[abs_oI] = merge(+, all_dict_weighted_samples[abs_oI], all_dict_weighted_samples_threads[id_thread][abs_oI])
                all_oI_weight[abs_oI] += all_oI_weight_threads[id_thread][abs_oI]
            end
        end
    end

    return sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples
end


function GenerateRandomActionSet(action_space, nb_init_actions::Int64)
    init_action_set = []
    for i in 1:nb_init_actions
        push!(init_action_set, rand(action_space))
    end
    return init_action_set
end


function ProcessActions(n::FscNode, actions::Vector{Any})
    for a in actions
        AddNewAction(n, a)
    end
end

function ProcessActions(n::FscNode, actions::Vector{Int64})
    for a in actions
        AddNewAction(n, a)
    end
end


function ProcessActions(n::FscNode, actions)
    for a in actions
        AddNewAction(n, a)
    end
end

function ExportLogData(planner::ContinuousPlannerPOMCGS, name::String)
    output_name = name * "-k" *string.(planner._nb_abstract_obs) *"xi" * string.(planner._max_b_gap)* "c" *string.(planner._softmax_c)* ".csv"

    min_length = min(length(planner._Log_result._vec_episodes),
                    length(planner._Log_result._vec_evaluation_value),
                    length(planner._Log_result._vec_upper_bound),
                    length(planner._Log_result._vec_valid_value),
                    length(planner._Log_result._vec_unvalid_rate),
                    length(planner._Log_result._vec_fsc_size))


    df = DataFrame(episode = planner._Log_result._vec_episodes[1:min_length],
                   lower = planner._Log_result._vec_evaluation_value[1:min_length],
                   upper = planner._Log_result._vec_upper_bound[1:min_length],
                   valid_value = planner._Log_result._vec_valid_value[1:min_length],
                   unvalid_rate = planner._Log_result._vec_unvalid_rate[1:min_length],
                   fsc_size = planner._Log_result._vec_fsc_size[1:min_length])
    CSV.write(output_name, string.(df))
end
