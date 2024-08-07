using OrderedCollections


mutable struct FscNode
    # _state_particles::Vector{Any}
    # _nb_particles::Int64
    _actions::Vector{Any}
    _Q_action::Dict{Any,Float64}
    _R_action::Dict{Any,Float64} # expected instant reward 
    _visits_action::Dict{Any,Int64}
    _visits_node::Int64
    _V_node::Float64
    _dict_weighted_samples::OrderedDict{Any, Float64}
    _PCA_Ms::Dict{Any, Any}  # a -> PCA model
    _best_action::Any
    _abstract_observations::Dict{Any, Vector{Vector{Float64}}} # a -> collection_obs_centroid
end

mutable struct BeliefSearchTreeNode
    _fsc_node_index::Int64
    _childs::Vector{BeliefSearchTreeNode}
end

mutable struct FSC
    _eta::Vector{Dict{Pair{Any, Int64},Int64}} # nI -> {a, abstract_obsI} -> next_nI
    _eta_search::Dict{Int64,Vector{Int64}} # nI -> childs 
    _nodes::Vector{FscNode}
    _max_accept_belief_gap::Float64
    _max_node_size::Int64
    _action_space
    _map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}}
    _belief_search_root::BeliefSearchTreeNode
    _flag_unexpected_obs::Int64
    _prunned_node_list::Vector{Int64}
end

function InitBeliefSearchTreeNode(nI::Int64)
    return BeliefSearchTreeNode(nI, Vector{BeliefSearchTreeNode}())
end


function InitFscNode(action_space)
    # --- init for actions ---
    init_actions = []
    init_Q_action = Dict{Any,Float64}()
    init_R_action = Dict{Any,Float64}()
    init_visits_action = Dict{Any,Int64}()
    init_abstract_observations = Dict{Any, Vector{Vector{Float64}}}()
    init_PCA_Ms = Dict{Any, Any}()

    for a in action_space
        # init_collection_obs_dict[a] = Vector{Vector{Float64}}()
        init_abstract_observations[a] = Vector{Vector{Float64}}()
        # init_collection_raw_sp_dict[a] = Dict{Int64,Vector{Any}}()
        init_Q_action[a] = 0.0
        init_R_action[a] = 0.0
        init_visits_action[a] = 0
    end
    # ------------------------
    init_visits_node = 0
    init_V_node = 0.0
    # nb_particles = 0
    # --- Weighted Particles ----
    init_dict_weighted_particles = OrderedDict{Any, Float64}()
    # --- abstract observations ---
    return FscNode( init_actions, 
                    # nb_particles,
                    init_Q_action,
                    init_R_action,
                    init_visits_action,
                    init_visits_node,
                    init_V_node,
                    init_dict_weighted_particles,
                    init_PCA_Ms,
                    # init_collection_raw_sp_dict,
                    # init_collection_obs_dict,
                    nothing,
                    init_abstract_observations)
                  
end



function InitFscNode()
    init_actions = []
    init_Q_action = Dict{Any,Float64}()
    init_R_action = Dict{Any,Float64}()
    init_visits_action = Dict{Any,Int64}()
    init_abstract_observations = Dict{Any, Vector{Vector{Float64}}}()
    init_PCA_Ms = Dict{Any, Any}()
    # ------------------------
    init_visits_node = 0
    init_V_node = 0.0
    # --- Weighted Particles ----
    init_dict_weighted_particles = OrderedDict{Any, Float64}()
    # --- abstract observations ---
    return FscNode( init_actions, 
                    init_Q_action,
                    init_R_action,
                    init_visits_action,
                    init_visits_node,
                    init_V_node,
                    init_dict_weighted_particles,
                    init_PCA_Ms,
                    nothing,
                    init_abstract_observations)    
end


function AddNewAction(n::FscNode, a)
    if !haskey(n._visits_action, a)
        push!(n._actions, a)
        n._abstract_observations[a] = Vector{Vector{Float64}}()
        n._Q_action[a] = 0.0
        n._R_action[a] = 0.0
        n._visits_action[a] = 0.0
    end
end

function CreateNode(weighted_b::OrderedDict{Any, Float64})
    node = InitFscNode()
    node._dict_weighted_samples = weighted_b
    return node
end

function CreatNode(b, weighted_b, action_space)
    node = InitFscNode(action_space)
    node._dict_weighted_samples = weighted_b
    return node
end

function CreatNode(weighted_b::Dict{Any, Float64}, action_space)
    node = InitFscNode(action_space)
    node._dict_weighted_samples = weighted_b
    return node
end

function InitFSC(max_accept_belief_gap::Float64, max_node_size::Int64, action_space)
    # init_eta = Vector{Dict{Pair{Any, Int64},Int64}}(undef, max_node_size)
    init_eta = Vector{Dict{Pair{Any, Int64},Int64}}()
    init_eta_search = Dict{Int64, Vector{Int64}}()
    init_eta_search[-1] = Vector{Int64}()
    init_nodes = Vector{FscNode}()
    init_collection_continuous_states = Dict{Vector{Float64}, Vector{Any}}()
    init_belief_search_root = InitBeliefSearchTreeNode(-1)
    flag_unexpected_obs = -999
    init_prunned_node_list = Vector{Int64}()
    return FSC(init_eta,
                init_eta_search,
                init_nodes,
                max_accept_belief_gap,
                max_node_size,
                action_space,
                init_collection_continuous_states,
                init_belief_search_root,
                flag_unexpected_obs,
                init_prunned_node_list)

end    

function GetBestAction(action_space, n::FscNode)
    Q_max = typemin(Float64)
    best_a = rand(action_space)
    for (key, value) in n._Q_action
        if value > Q_max && n._visits_action[key] != 0
            Q_max = value
            best_a = key
        end
    end
    
    return best_a
end

function GetBestAction(n::FscNode)
    Q_max = typemin(Float64)
    best_a = rand(n._actions)
    for (key, value) in n._Q_action
        if value > Q_max && n._visits_action[key] != 0
            Q_max = value
            best_a = key
        end
    end
    
    return best_a
end

# function GetBestAction(n::FscNode)
#     if isnothing(n._best_action)
#         Q_max = typemin(Float64)
#         best_a = rand(keys(n._Q_action))
#         for (key, value) in n._Q_action
#             if value > Q_max && n._visits_action[key] != 0
#                 Q_max = value
#                 best_a = key
#             end
#         end
    
#         n._best_action = best_a
#         return best_a
#     else 
#         return n._best_action
#     end
# end

function FindSimiliarBelief(fsc::FSC, new_weighted_particles::Dict{Any, Float64}, b_gap_max::Float64)

    min_distance_node_i = -1
    min_distance = typemax(Float64)


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads  = ones(Float64, num_threads)*typemax(Float64)


    Threads.@threads for i in 1:length(fsc._nodes)
        id_thread = Threads.threadid()
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[i]._dict_weighted_samples
        for (key, value) in new_weighted_particles
            if haskey(weighted_b_node_i, key)
                distance_i += abs(value - weighted_b_node_i[key])
            else
                distance_i += value
            end

            if distance_i > b_gap_max
                break
            end
        end

        if distance_i < min_distance_threads[id_thread]
            min_distance_threads[id_thread] = distance_i
            min_distance_node_i_threads[id_thread] = i
        end
    end

    for i in 1:num_threads
        if min_distance_threads[i] < min_distance
            min_distance = min_distance_threads[i]
            min_distance_node_i = min_distance_node_i_threads[i]
        end
    end


    return min_distance_node_i, min_distance
end


function ComputeDistanceWithSortedBelief(d1, sorted_b, collection_sorted_b, length_comapre::Int64)
    collection_keys_cmp = []
    for i in 0:length_comapre - 1
        key_i = collection_sorted_b[length(collection_sorted_b) - i].first
        push!(collection_keys_cmp, key_i)
    end
    
    distance = 0.0
    for key_i in collection_keys_cmp
        if haskey(d1, key_i)
            distance += abs(sorted_b[key_i] - d1[key_i])
        else
            distance += sorted_b[key_i]
        end
    end
    return distance
end

function UcbActionSelection(fsc::FSC, nI::Int64, c::Float64)
    node_visits = fsc._nodes[nI]._visits_node
    max_value = typemin(Float64)
    selected_a = rand(fsc._action_space)

    for a in fsc._action_space
        AddNewAction(fsc._nodes[nI], a)
        ratio_visit = 0.0
        node_a_visits = fsc._nodes[nI]._visits_action[a]
        if node_a_visits == 0
            ratio_visit = log(node_visits + 1) / 1.0 
        else
            ratio_visit = log(node_visits + 1) / node_a_visits
        end

        value = fsc._nodes[nI]._Q_action[a] + c * sqrt(ratio_visit)

        if value > max_value
            max_value = value
            selected_a = a
        end

    end

    return selected_a
end


function ActionProgressiveWidening(n::FscNode, action_space, K_a::Float64, alpha_a::Float64, C_star::Int64)
    node_visits = n._visits_node
    if length(n._actions) <= K_a*(node_visits^alpha_a) && node_visits < C_star
        a = rand(action_space)
        AddNewAction(n, a)
        return a
    else
        max_value = typemin(Float64)
        selected_a = rand(n._actions)
        for a in n._actions
            ratio_visit = 0.0
            node_a_visits = n._visits_action[a]
            if node_a_visits == 0
                ratio_visit = log(node_visits + 1) / 1.0 
            else
                ratio_visit = log(node_visits + 1) / node_a_visits
            end
            value = n._Q_action[a] + c * sqrt(ratio_visit)
            if value > max_value
                max_value = value
                selected_a = a
            end
        end
        return selected_a 
    end
end


function ProcessState(state_grid::Vector{Float64}, state_particle::Vector{Float64})
    result = Vector{Int64}()
    for i in 1:length(state_particle)
        s_i = floor(Int64, state_particle[i] / state_grid[i])
        push!(result, s_i)
    end
    return result
end


function CheckMinMaxValue(vec_in, vec_max, vec_min, nb_dimension::Int64)
    for i in 1:nb_dimension
        vec_max[i] = max(vec_in[i], vec_max[i])
        vec_min[i] = min(vec_in[i], vec_min[i])
    end
end

function ComputeDistance(vec_1, vec_2)
    distance = 0.0
    for i in 1:length(vec_2)
        distance_temp = abs(vec_2[i] - vec_1[i])
        distance += distance_temp
    end
    return distance
end

function ComputeDistance(dict_1::OrderedDict{Any, Float64}, dict_2::OrderedDict{Any, Float64})
    sum = 0.0
    for (key, value) in dict_1
        if haskey(dict_2, key)
            sum += abs(value - dict_2[key])
        else
            sum += value
        end
    end
    
    return sum
end

function FindMostCloseAbstractObs(obs_vec,
    all_abstract_obs::Vector{Vector{Float64}})
    result = -1
    min_distance = typemax(Float64)
    for i in 1:length(all_abstract_obs)
        distance_i = ComputeDistance(obs_vec, all_abstract_obs[i])
        if distance_i < min_distance
            min_distance = distance_i
            result = i
        end
    end

    return result
end

function GetStateParticlesFromAbstractObs(node::FscNode, a, abstract_oI::Int64)
    return node._collection_raw_sp_dict[a][abstract_oI]
end

function GetStateParticlesFromAction(node::FscNode, a)
    state_particles = []
    for (abs_oI, vec_particles) in  node._collection_raw_sp_dict[a]
        for sp in vec_particles
             push!(state_particles, sp)
        end
    end
    
    return state_particles
    
end

function AddStateWithAbstractObs(node::FscNode, a, abstract_obsI::Int64, sp)
    if haskey(node._collection_raw_sp_dict[a], abstract_obsI)
        push!(node._collection_raw_sp_dict[a][abstract_obsI], sp)
    else
        node._collection_raw_sp_dict[a][abstract_obsI] = [sp] 
    end
end 


function MergeDict(d1, d2)
    for (key, value) in d2
        if haskey(d1, key)
            d1[key] = (value + d1[key])/2
        else
            d1[key] = value
        end
    end
    
    # normalization
    sum = 0.0
    for (key, value) in d1
        sum += value
    end
    
    for (key, value) in d1
        value /= sum
    end
end

function SampleState(node::FscNode)
    p_rand = rand()
    sum_p = 0.0
    for (key, value) in node._dict_weighted_samples
        sum_p += value 
        if sum_p >= p_rand
            return key
        end
    end
end

function StoreAllAbstractObs(fsc::FSC, nI::Int64, a, all_abstract_obs::Set{Vector{Float64}})
    for elem in all_abstract_obs
        push!(fsc._nodes[nI]._abstract_observations[a], elem)
    end
end

function SearchOrInsertBelief(fsc::FSC, 
                              belief::OrderedDict{Any, Float64}, 
                              search_node_index::Int64, 
                              distance::Float64, 
                              depth::Int64)
   
    compare_distance = distance*0.8
    min_distance = typemax(Float64)
    min_node_index = -1


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads  = ones(Float64, num_threads)*typemax(Float64)

    Threads.@threads for i in 1:length(fsc._eta_search[search_node_index])
        id_thread = Threads.threadid()
        child_node_i = fsc._eta_search[search_node_index][i]
        current_distance = ComputeDistance(belief, fsc._nodes[child_node_i]._dict_weighted_samples)
        if current_distance < min_distance_threads[id_thread]
            min_distance_threads[id_thread] = current_distance
            min_distance_node_i_threads[id_thread] = child_node_i
        end
    end

    for id_thread in 1:num_threads
        if min_distance > min_distance_threads[id_thread] 
            min_distance = min_distance_threads[id_thread] 
            min_node_index = min_distance_node_i_threads[id_thread]
        end
    end


    if (min_distance < fsc._max_accept_belief_gap)
        return true, min_node_index
    end

    if (min_distance > compare_distance) 
        n_next = CreateNode(belief)
        push!(fsc._nodes, n_next)
        n_nextI = length(fsc._nodes)
        push!(fsc._eta, Dict{Pair{Any, Int64},Int64}()) #add one row for n_nextI in _eta
        fsc._eta_search[n_nextI] = Vector{Int64}() #add one row for n_nextI in _eta_search
        push!(fsc._eta_search[search_node_index] , n_nextI)
        return false, n_nextI
    else
        SearchOrInsertBelief(fsc, belief, min_node_index, distance, depth+1)
    end
end

function SearchOrInsertBelief(fsc::FSC, new_weighted_particles::OrderedDict{Any, Float64}, b_gap_max::Float64)

    min_distance_node_i = -1
    min_distance = typemax(Float64)


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads  = ones(Float64, num_threads)*typemax(Float64)


    Threads.@threads for i in 1:length(fsc._nodes)
        id_thread = Threads.threadid()
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[i]._dict_weighted_samples
        for (key, value) in new_weighted_particles
            if haskey(weighted_b_node_i, key)
                distance_i += abs(value - weighted_b_node_i[key])
            else
                distance_i += value
            end

            if distance_i > b_gap_max
                break
            end
        end

        if distance_i < min_distance_threads[id_thread]
            min_distance_threads[id_thread] = distance_i
            min_distance_node_i_threads[id_thread] = i
        end
    end

    for i in 1:num_threads
        if min_distance_threads[i] < min_distance
            min_distance = min_distance_threads[i]
            min_distance_node_i = min_distance_node_i_threads[i]
        end
    end

    # create new belief node if needed
    if min_distance > b_gap_max
        n_next = CreateNode(new_weighted_particles)
        push!(fsc._nodes, n_next)
        n_nextI = length(fsc._nodes)
        push!(fsc._eta, Dict{Pair{Any, Int64},Int64}()) #add one row for n_nextI in _eta
        return false, n_nextI
    else
        return true, min_distance_node_i
    end
end

function SearchOrInsertBeliefWithPrunnedNodes(fsc::FSC, new_weighted_particles::OrderedDict{Any, Float64}, b_gap_max::Float64)

    min_distance_node_i = -1
    min_distance = typemax(Float64)


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads  = ones(Float64, num_threads)*typemax(Float64)


    Threads.@threads for i in 1:length(fsc._prunned_node_list)
        node_i = fsc._prunned_node_list[i]
        id_thread = Threads.threadid()
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[node_i]._dict_weighted_samples
        for (key, value) in new_weighted_particles
            if haskey(weighted_b_node_i, key)
                distance_i += abs(value - weighted_b_node_i[key])
            else
                distance_i += value
            end

            if distance_i > b_gap_max
                break
            end
        end

        if distance_i < min_distance_threads[id_thread]
            min_distance_threads[id_thread] = distance_i
            min_distance_node_i_threads[id_thread] = node_i
        end
    end

    for i in 1:num_threads
        if min_distance_threads[i] < min_distance
            min_distance = min_distance_threads[i]
            min_distance_node_i = min_distance_node_i_threads[i]
        end
    end

    # create new belief node if needed
    if min_distance > b_gap_max
        n_next = CreateNode(new_weighted_particles)
        push!(fsc._nodes, n_next)
        n_nextI = length(fsc._nodes)
        push!(fsc._eta, Dict{Pair{Any, Int64},Int64}()) #add one row for n_nextI in _eta
        push!(fsc._prunned_node_list, n_nextI)
        return false, n_nextI
    else
        return true, min_distance_node_i
    end
end



function Prunning(fsc::FSC) 
    nI = 1
    open_list = [nI]
    result_list = [nI]
    while length(open_list) > 0
        nI = last(open_list)
        deleteat!(open_list, length(open_list))
        a = GetBestAction(fsc._nodes[nI])
        for (k,v) in fsc._eta[nI]
            if (k[1] == a) && !(v in result_list)
                push!(open_list, v)
                push!(result_list, v)  
            end
        end
    end

    return result_list
end
