# define Naive FSC
using OrderedCollections

mutable struct FscNode
    _state_particles
    _nb_particles::Int64
    _Q_action::Dict{Any,Float64}
    _R_action::Dict{Any,Float64} # expected instant reward 
    _visits_action::Dict{Any,Int64}
    _visits_node::Int64
    _V_node::Float64
    _best_action::Any
    _dict_weighted_samples::OrderedDict{Any,Float64}
end


mutable struct FSC
    _eta::Vector{Dict{Pair{Any,Int64},Int64}}
    # _eta::Vector{Dict{Pair{Any, Any},Int64}}
    # Fast search nodes
    _eta_search::Dict{Int64,Vector{Int64}} # nI -> child indices
    _nodes::Vector{FscNode}
    _max_accept_belief_gap::Float64
    _max_node_size::Int64
    _action_space
    _dict_trans_func::Dict{Any,Dict{Any,Dict{Any,Float64}}} # a->s->s'
    _dict_obs_func::Dict{Any,Dict{Any,Dict{Any,Float64}}} # a->s'->o
    _dict_process_a_s::Dict{Pair{Any,Any},Bool} # (a, s) -> bool
    _flag_unexpected_obs::Int64
    _prunned_node_list::Vector{Int64}
end

function InitFscNode(action_space)
    init_particles = []
    # --- init for actions ---
    init_Q_action = Dict{Any,Float64}()
    init_R_action = Dict{Any,Float64}()
    init_visits_action = Dict{Any,Int64}()
    for i in action_space
        init_Q_action[i] = 0.0
        init_R_action[i] = 0.0
        init_visits_action[i] = 0
    end
    # ------------------------
    init_visits_node = 0
    init_V_node = 0
    nb_particles = 0
    # --- Weighted Particles ----
    init_dict_weighted_particles = OrderedDict{Any,Float64}()
    return FscNode(init_particles,
        nb_particles,
        init_Q_action,
        init_R_action,
        init_visits_action,
        init_visits_node,
        init_V_node,
        nothing,
        init_dict_weighted_particles)

end

function CreatNode(b, weighted_b, action_space)
    node = InitFscNode(action_space)
    node._state_particles = b
    #     node._feature_vector = feature_b
    node._dict_weighted_samples = weighted_b
    node._nb_particles = length(b)
    return node
end


function InitFSC(max_accept_belief_gap::Float64, max_node_size::Int64, action_space)
    init_eta = Vector{Dict{Pair{Any,Int64},Int64}}(undef, max_node_size)
    for i in range(1, stop=max_node_size)
        init_eta[i] = Dict{Pair{Any,Int64},Int64}()
    end
    # init_eta = Vector{Dict{Pair{Any, Any},Int64}}(undef, max_node_size)
    # for i in range(1,stop=max_node_size)
    #     init_eta[i] = Dict{Pair{Any, Any},Int64}()
    # end
    # init_eta = Vector{Dict{Pair{Any, Int64},Int64}}()
    init_eta_search = Dict{Int64,Vector{Int64}}()
    init_eta_search[-1] = Vector{Int64}()
    init_nodes = Vector{FscNode}()
    init_dict_trans_func = Dict{Any,Dict{Any,Dict{Any,Float64}}}() # a->s->s'
    init_dict_obs_func = Dict{Any,Dict{Any,Dict{Any,Float64}}}() # a->s'->o
    init_dict_process_a_s = Dict{Pair{Any,Any},Bool}() # (a, s) -> bool
    flag_unexpected_obs = -999
    init_prunned_node_list = Vector{Int64}()
    return FSC(init_eta,
        init_eta_search,
        init_nodes,
        max_accept_belief_gap,
        max_node_size,
        action_space,
        init_dict_trans_func,
        init_dict_obs_func,
        init_dict_process_a_s,
        flag_unexpected_obs,
        init_prunned_node_list)

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

function GetBestAction(n::FscNode)
    Q_max = typemin(Float64)
    best_a = rand(keys(n._Q_action))
    for (key, value) in n._Q_action
        if value > Q_max && n._visits_action[key] != 0
            Q_max = value
            best_a = key
        end
    end

    n._best_action = best_a
    return best_a
end

# function FindSimiliarBelief(fsc::FSC, new_weighted_particles)
#     min_distance_node_i = -1
#     min_distance = typemax(Float64)
#     sorted_b = sort(new_weighted_particles; byvalue=true)
#     collection_sorted_b = collect(sorted_b)
#     largest_s_key = last(collection_sorted_b).first
#     for i in 1:length(fsc._nodes)
#         distance_i = 0.0
#         weighted_b_node_i = fsc._nodes[i]._dict_weighted_samples
#         if haskey(weighted_b_node_i, largest_s_key) || sorted_b[largest_s_key] < 0.05
#             for (key, value) in new_weighted_particles
#                 flag = haskey(weighted_b_node_i, key)
#                 if flag
#                     distance_i += abs(value - weighted_b_node_i[key])
#                 else
#                     distance_i += value
#                 end
#             end

#             if distance_i < min_distance
#                 min_distance = distance_i
#                 min_distance_node_i = i
#             end
#         end
#     end

#     return min_distance_node_i, min_distance
# end

function FindSimiliarBeliefKLD(fsc::FSC, new_weighted_particles)
    min_distance_node_i = -1
    min_distance = typemax(Float64)
    sorted_b = sort(new_weighted_particles; byvalue=true)
    collection_sorted_b = collect(sorted_b)
    largest_s_key = last(collection_sorted_b).first
    for i in 1:length(fsc._nodes)
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[i]._dict_weighted_samples
        if haskey(weighted_b_node_i, largest_s_key) || sorted_b[largest_s_key] < 0.05

            distance_i = ComputeKLD(new_weighted_particles, weighted_b_node_i)

            if distance_i < min_distance
                min_distance = distance_i
                min_distance_node_i = i
            end
        end
    end

    return min_distance_node_i, min_distance
end

function FindSimiliarBeliefKLD(fsc::FSC, new_weighted_particles, b_gap_required)
    min_distance_node_i = -1
    min_distance = b_gap_required
    sorted_b = sort(new_weighted_particles; byvalue=true)
    collection_sorted_b = collect(sorted_b)
    largest_s_key = last(collection_sorted_b).first
    for i in 1:length(fsc._nodes)
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[i]._dict_weighted_samples
        if haskey(weighted_b_node_i, largest_s_key) || sorted_b[largest_s_key] < 0.05

            distance_i = ComputeKLD(new_weighted_particles, weighted_b_node_i, b_gap_required, min_distance)

            if distance_i < min_distance
                min_distance = distance_i
                min_distance_node_i = i
            end
        end
    end

    return min_distance_node_i, min_distance
end

function ComputeKLD(p, q)
    res = 0.0
    for (x, px) in p
        if haskey(q, x)
            res += px * log(px / q[x])
        else
            res += px * log(px / 0.01)
        end
    end
    return res
end

function ComputeKLD(p, q, b_gap_required, b_gap_min)
    res = 0.0
    for (x, px) in p
        if res < b_gap_required || res < b_gap_min
            if haskey(q, x)
                res += px * log(px / q[x])
            else
                res += px * log(px / 0.01)
            end
        else
            res = b_gap_required
            break
        end
    end
    return res
end

function FindSimiliarBelief(fsc::FSC, new_weighted_particles, b_gap_required)
    min_distance_node_i = -1
    min_distance = typemax(Float64)
    for i in 1:length(fsc._nodes)
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[i]._dict_weighted_samples
        for (key, value) in new_weighted_particles
            flag = haskey(weighted_b_node_i, key)
            if flag
                distance_i += abs(value - weighted_b_node_i[key])
            else
                distance_i += value
            end


            if distance_i > b_gap_required
                break
            end

        end

        if distance_i < min_distance
            min_distance = distance_i
            min_distance_node_i = i
        end

    end

    return min_distance_node_i, min_distance
end


function UcbActionSelection(fsc::FSC, nI::Int64, c::Float64)
    node_visits = fsc._nodes[nI]._visits_node
    max_value = typemin(Float64)
    selected_a = rand(fsc._action_space)

    for a in fsc._action_space
        ratio_visit = 0
        node_a_visits = fsc._nodes[nI]._visits_action[a]
        if node_a_visits == 0
            ratio_visit = log(node_visits + 1) / 0.1
        else
            ratio_visit = log(node_visits + 1) / node_a_visits
        end

        # value = fsc._nodes[nI]._Q_action[a] + c * sqrt(ratio_visit) / log(node_visits + 2)
        value = fsc._nodes[nI]._Q_action[a] + c * sqrt(ratio_visit)

        if value > max_value
            max_value = value
            selected_a = a
        end

    end

    return selected_a
end


function SearchOrInsertBelief(fsc::FSC, new_weighted_particles::OrderedDict{Any,Float64}, b_gap_max::Float64)

    min_distance_node_i = -1
    min_distance = typemax(Float64)


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads = ones(Float64, num_threads) * typemax(Float64)


    Threads.@threads for i in 1:length(fsc._nodes)
        id_thread = Threads.threadid()
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[i]._dict_weighted_samples
        # if haskey(weighted_b_node_i, largest_s_key) || sorted_b[largest_s_key] < 0.05
        # for (key, value) in sorted_b
        for (key, value) in new_weighted_particles
            # for (key, value) in new_weighted_particles
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
        # end
    end

    for i in 1:num_threads
        if min_distance_threads[i] < min_distance
            min_distance = min_distance_threads[i]
            min_distance_node_i = min_distance_node_i_threads[i]
        end
    end

    # create new belief node if needed
    if min_distance > b_gap_max
        n_next = CreatNode([], new_weighted_particles, fsc._action_space)
        push!(fsc._nodes, n_next)
        n_nextI = length(fsc._nodes)
        # push!(fsc._eta, Dict{Pair{Any, Int64},Int64}()) #add one row for n_nextI in _eta
        # fsc._eta_search[n_nextI] = Vector{Int64}() #add one row for n_nextI in _eta_search
        # push!(fsc._eta_search[search_node_index] , n_nextI)
        return false, n_nextI
    else
        return true, min_distance_node_i
    end
end



function SearchOrInsertBelief(fsc::FSC, node_list::Vector{Int64}, new_weighted_particles::OrderedDict{Any,Float64}, b_gap_max::Float64)

    min_distance_node_i = -1
    min_distance = typemax(Float64)


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads = ones(Float64, num_threads) * typemax(Float64)


    Threads.@threads for i in 1:length(node_list)
        node_i = node_list[i]
        id_thread = Threads.threadid()
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[node_i]._dict_weighted_samples
        for (key, value) in new_weighted_particles
            # for (key, value) in new_weighted_particles
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
        # end
    end

    for i in 1:num_threads
        if min_distance_threads[i] < min_distance
            min_distance = min_distance_threads[i]
            min_distance_node_i = min_distance_node_i_threads[i]
        end
    end

    # create new belief node if needed
    if min_distance > b_gap_max
        n_next = CreatNode([], new_weighted_particles, fsc._action_space)
        push!(fsc._nodes, n_next)
        n_nextI = length(fsc._nodes)
        push!(node_list, n_nextI)
        # push!(fsc._eta, Dict{Pair{Any, Int64},Int64}()) #add one row for n_nextI in _eta
        # fsc._eta_search[n_nextI] = Vector{Int64}() #add one row for n_nextI in _eta_search
        # push!(fsc._eta_search[search_node_index] , n_nextI)
        return false, n_nextI
    else
        return true, min_distance_node_i
    end
end


function SearchOrInsertBelief(fsc::FSC,
                                belief::OrderedDict{Any,Float64},
                                search_node_index::Int64,
                                distance::Float64,
                                depth::Int64)

    compare_distance = distance * 0.8
    min_distance = typemax(Float64)
    min_node_index = -1


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads = ones(Float64, num_threads) * typemax(Float64)

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
        n_next = CreatNode([], belief, fsc._action_space)
        push!(fsc._nodes, n_next)
        n_nextI = length(fsc._nodes)
        push!(fsc._eta, Dict{Pair{Any,Int64},Int64}()) #add one row for n_nextI in _eta
        fsc._eta_search[n_nextI] = Vector{Int64}() #add one row for n_nextI in _eta_search
        push!(fsc._eta_search[search_node_index], n_nextI)
        return false, n_nextI
    else
        SearchOrInsertBelief(fsc, belief, min_node_index, distance, depth + 1)
    end
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
