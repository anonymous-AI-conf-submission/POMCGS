include("./Qlearning.jl")
include("./FSC.jl")
using POMDPModels
using POMDPs
using POMDPModelTools
using Random
using DataFrames, CSV

# define Naive POMCGS

mutable struct LogResult
	_vec_episodes::Vector{Int64}
	_vec_evaluation_value::Vector{Float64}
	_vec_valid_value::Vector{Float64}
	_vec_unvalid_rate::Vector{Float64}
	_vec_upper_bound::Vector{Float64}
	_vec_fsc_size::Vector{Int64}
end


mutable struct NaivePlannerPOMCGS
	_nb_process_action_samples::Int64                   # default 1000
	_max_b_gap::Float64                          # default 0.05
	_max_graph_node_size::Int64                  # default 1e5
	_nb_iter::Int64                              # default 1e7
	_softmax_c::Float64                          # default 3.0
	_discount::Float64                           # default 0.9
	_epsilon::Float64                            # default 0.01
	_C_star::Int64								 # default 100
	_max_planning_secs::Int64                     # default 1e5
	_nb_sim::Int64								 # default 1000
	_nb_eval::Int64                              # default 100000
	_Q_learning_policy::Qlearning
	_Log_result::LogResult
	_R_lower::Float64
	NaivePlannerPOMCGS(R_lower, Q_learning_policy::Qlearning) =
		new(1000, 0.05, 1e6, 1e7, 3.0, 0.9, 0.01, 100, 1e5, 1000, 10000, Q_learning_policy, LogResult(Vector{Int64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Int64}()), R_lower)
end


function InitPlannerParameters(planner::NaivePlannerPOMCGS,
	nb_process_action_samples::Int64,
	max_b_gap::Float64,
	softmax_c::Float64,
	discount::Float64,
	epsilon::Float64)
	planner._nb_process_action_samples = nb_process_action_samples
	planner._max_b_gap = max_b_gap
	planner._softmax_c = softmax_c
	planner._discount = discount
	planner._epsilon = epsilon
end



function InitPOMDP(pomdp, b0, nb_particles_b0)
	action_space = collect(actions(pomdp))
	particles_b0 = []
	dict_weighted_b0 = Dict{Any, Float64}()

	for i in 1:nb_particles_b0
		s = rand(b0)
		push!(particles_b0, s)
		if haskey(dict_weighted_b0, s)
			dict_weighted_b0[s] += 1.0 / nb_particles_b0
		else
			dict_weighted_b0[s] = 1.0 / nb_particles_b0
		end
	end


	return particles_b0, dict_weighted_b0, action_space
end

function SimulationWithFSC(b0, pomdp, fsc::FSC, steps::Int64)
	s = rand(b0)
	sum_r = 0.0
	nI = 1
	for i in 1:steps
		if (isterminal(pomdp, s))
			break
		end

		println("---------")
		println("step: ", i)
		println("state:", s)
		a = GetBestAction(fsc._nodes[nI])
		println("perform action:", a)
		sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
		s = sp
		sum_r += (discount(pomdp)^i) * r
		println("recieve obs:", o)
		println("nI:", nI)
		println("nI visits:", fsc._nodes[nI]._visits_node)
		println("nI value:", fsc._nodes[nI]._V_node)
		nI = fsc._eta[nI][Pair(a, o)]
		println("reward:", r)
	end

	println("sum_r:", sum_r)
end

function EvaluationWithSimulationFSC(b0, pomdp, fsc::FSC, discount::Float64, nb_sim::Int64)

	sum_r = 0.0
	sum_r_valid = 0.0
	sum_unvalid_search = 0
	a_safe = GetBestAction(fsc._nodes[1]) # should be a safe action, or a greedy action 

	for sim_i in 1:nb_sim
		step = 0
		sum_r_sim_i = 0.0
		s = rand(b0)
		nI = 1
		bool_random_pi = false
		bool_sim_i_invalid = false

		while (discount^step) > 0.01 && isterminal(pomdp, s) == false
			if bool_random_pi == true && bool_sim_i_invalid == false
				bool_sim_i_invalid = true
				sum_unvalid_search += 1
			end

			if nI == -1
				bool_random_pi = true
			elseif fsc._nodes[nI]._visits_node == 0
				bool_random_pi = true
			end

			if bool_random_pi
				# a = rand(fsc._action_space)
				a = a_safe
			else
				a = GetBestAction(fsc._nodes[nI])
			end

			sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
			s = sp
			sum_r_sim_i += (discount^step) * r

			if haskey(fsc._eta[nI], Pair(a, o))
				nI = fsc._eta[nI][Pair(a, o)]
			else
				bool_random_pi = true
			end
			step += 1
		end
		sum_r += sum_r_sim_i
		if bool_sim_i_invalid == false
			sum_r_valid += sum_r_sim_i
		end
	end

	println("avg sum:", sum_r / nb_sim)
	println("avg sum valid search:", sum_r_valid / (nb_sim - sum_unvalid_search))
	println("unvalid search:", sum_unvalid_search / nb_sim)
end

function EvaluationWithSimulationFSC(b0, pomdp, fsc::FSC, discount::Float64, nb_sim::Int64, vec_evaluation_value, vec_valid_value, vec_unvalid_rate)

	sum_r = 0.0
	sum_r_valid = 0.0
	sum_unvalid_search = 0

	# set a default action
	a_safe = GetBestAction(fsc._nodes[1]) # should be a safe action, or a greedy action 

	for sim_i in 1:nb_sim
		step = 0
		sum_r_sim_i = 0.0
		s = rand(b0)
		nI = 1
		bool_random_pi = false
		bool_sim_i_invalid = false

		while (discount^step) > 0.01 && isterminal(pomdp, s) == false
			if bool_random_pi == true && bool_sim_i_invalid == false
				bool_sim_i_invalid = true
				sum_unvalid_search += 1
			end

			if nI == -1
				bool_random_pi = true
			elseif fsc._nodes[nI]._visits_node == 0
				bool_random_pi = true
			end

			if bool_random_pi
				# a = rand(fsc._action_space)
				a = a_safe
			else
				a = GetBestAction(fsc._nodes[nI])
			end

			sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
			s = sp
			sum_r_sim_i += (discount^step) * r

			if haskey(fsc._eta[nI], Pair(a, o))
				nI = fsc._eta[nI][Pair(a, o)]
			else
				bool_random_pi = true
			end
			step += 1
		end
		sum_r += sum_r_sim_i
		if bool_sim_i_invalid == false
			sum_r_valid += sum_r_sim_i
		end
	end

	#sum_r is a lower bound
	# println("avg sum:", sum_r / nb_sim)
	# println("avg sum valid search:", sum_r_valid / (nb_sim - sum_unvalid_search))
	# println("unvalid search:", sum_unvalid_search / nb_sim)
	push!(vec_evaluation_value, sum_r / nb_sim)
	push!(vec_valid_value, sum_r_valid / (nb_sim - sum_unvalid_search))
	push!(vec_unvalid_rate, sum_unvalid_search / nb_sim)

	return sum_r / nb_sim

end

function EvaluateUpperBound(b0, pomdp, fsc::FSC, Q_learning_policy::Qlearning, discount::Float64, nb_sim::Int64, C_star::Int64)
	sum_r = 0.0

	for sim_i in 1:nb_sim
		step = 0
		sum_r_sim_i = 0.0
		s = rand(b0)
		nI = 1

		while (discount^step) > 0.01 && isterminal(pomdp, s) == false
			a = GetBestAction(fsc._nodes[nI])
			if (a == -1)
				sum_r_sim_i += (discount^step) * GetValueQMDP(fsc._nodes[nI], Q_learning_policy)
				break
			end

			sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
			s = sp
			if haskey(fsc._eta[nI], Pair(a, o)) && fsc._nodes[nI]._visits_node > C_star
				nI = fsc._eta[nI][Pair(a, o)]
				sum_r_sim_i += (discount^step) * r
			else
				sum_r_sim_i += (discount^step) * GetValueQMDP(fsc._nodes[nI], Q_learning_policy)
				break
			end
			step += 1
		end
		sum_r += sum_r_sim_i
	end

	return sum_r / nb_sim
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
	for a in action_space
		if (action_min_r[a] > max_min_r)
			max_min_r = action_min_r[a]
		end
	end

	return max_min_r / (1 - discount(pomdp))
end

function EvaluateBounds(b0, pomdp, R_lower_bound, fsc::FSC, Q_learning_policy::Qlearning, discount::Float64, nb_sim::Int64, C_star::Int64)
	sum_r_U = 0.0
	sum_r_L = 0.0

	for sim_i in 1:nb_sim
		step = 0
		sum_r_U_sim_i = 0.0
		sum_r_L_sim_i = 0.0
		s = rand(b0)
		nI = 1

		while (discount^step) > 0.01 && isterminal(pomdp, s) == false
			a = GetBestAction(fsc._nodes[nI])
			if (a == -1)
				sum_r_U_sim_i += (discount^step) * GetValueQMDP(fsc._nodes[nI], Q_learning_policy)
				break
			end

			sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
			s = sp
			if haskey(fsc._eta[nI], Pair(a, o)) && fsc._nodes[nI]._visits_node > C_star
				nI = fsc._eta[nI][Pair(a, o)]
				sum_r_U_sim_i += (discount^step) * r
				sum_r_L_sim_i += (discount^step) * r
			else
				sum_r_U_sim_i += (discount^step) * GetValueQMDP(fsc._nodes[nI], Q_learning_policy)
				# sum_r_L_sim_i += (discount^step)*R_lower_bound
				temp_sum_r = 0.0
				while (discount^step) > 0.01 && isterminal(pomdp, s) == false
					sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
					s = sp
					if haskey(fsc._eta[nI], Pair(a, o))
						nI = fsc._eta[nI][Pair(a, o)]
						temp_sum_r += (discount^step) * r
					else
						temp_sum_r += (discount^step) * R_lower_bound
						break
					end
					step += 1
				end

				sum_r_L_sim_i += (discount^step) * temp_sum_r
				# sum_r_L_sim_i += (discount^step)*SimulationFSC(s, pomdp, fsc, discount, nI, R_lower_bound, step)
				break
			end
			step += 1
		end
		sum_r_U += sum_r_U_sim_i
		sum_r_L += sum_r_L_sim_i

	end

	return sum_r_U / nb_sim, sum_r_L / nb_sim
end


function SimulationFSC(s, pomdp, fsc::FSC, discount::Float64, nI::Int64, R_lower_bound::Float64, step::Int64)
	sum_r = 0.0
	while (discount^step) > 0.01 && isterminal(pomdp, s) == false
		a = GetBestAction(fsc._nodes[nI])
		sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
		s = sp
		if haskey(fsc._eta[nI], Pair(a, o))
			nI = fsc._eta[nI][Pair(a, o)]
			sum_r += (discount^step) * r
		else
			sum_r += (discount^step) * R_lower_bound
			break
		end

		step += 1
	end

	return sum_r
end




function ProcessActionWeightedParticle(pomdp,
	fsc::FSC,
	nI::Int64,
	a,
	depth::Int64,
	nb_process_samples::Int64,
	discount::Float64,
	Q_learning_policy::Qlearning)

	# 1: Collect Samples and build new beliefs
	sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples = CollectSamplesAndBuildNewBeliefsWeightedParticles(pomdp,
		fsc,
		nI,
		a,
		nb_process_samples)

	# 4. Build new beliefs 
	fsc._nodes[nI]._R_action[a] = sum_R_a
	expected_future_V = 0.0
	for (key, value) in all_dict_weighted_samples
		NormalizeDict(all_dict_weighted_samples[key])
		sort!(all_dict_weighted_samples[key], rev = true, byvalue = true)
		bool_search, n_nextI = SearchOrInsertBelief(fsc, fsc._prunned_node_list, all_dict_weighted_samples[key], fsc._max_accept_belief_gap)
		if !bool_search
			max_Q = HeuristicNodeQ(fsc._nodes[n_nextI], Q_learning_policy)
			fsc._nodes[n_nextI]._V_node = max_Q
		end
		fsc._eta[nI][Pair(a, key)] = n_nextI
		obs_weight = all_oI_weight[key]
		expected_future_V += (obs_weight / sum_all_weights) * fsc._nodes[n_nextI]._V_node
	end

	# --- Update Q(n, a) -----
	fsc._nodes[nI]._Q_action[a] = fsc._nodes[nI]._R_action[a] + discount * expected_future_V
	return fsc._nodes[nI]._Q_action[a]
end


function Simulate(pomdp,
	fsc::FSC,
	s,
	nI::Int64,
	depth::Int64,
	nb_process_action_samples::Int64,
	discount::Float64,
	c::Float64,
	epsilon::Float64,
	Q_learning_policy::Qlearning)


	if (discount^depth) * (Q_learning_policy._R_max - Q_learning_policy._R_min) < epsilon || isterminal(pomdp, s) || nI == -1
		return 0
	end


	a = UcbActionSelection(fsc, nI, c)

	fsc._nodes[nI]._visits_node += 1
	fsc._nodes[nI]._visits_action[a] += 1

	if fsc._nodes[nI]._visits_action[a] == 1
		return ProcessActionWeightedParticle(pomdp, fsc, nI, a, depth, nb_process_action_samples, discount, Q_learning_policy)
	end

	nI_next = -1
	sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
	if haskey(fsc._eta[nI], Pair(a, o))
		nI_next = fsc._eta[nI][Pair(a, o)]
	else
		fsc._eta[nI][Pair(a, fsc._flag_unexpected_obs)] = nI
		nI_next = nI # could do something smarter..., may be a merged belief node with all possible observations
	end

	# update r(n, a)
	nb_process_samples = nb_process_action_samples
	sum_R_n_a = fsc._nodes[nI]._R_action[a] * (nb_process_samples + fsc._nodes[nI]._visits_action[a])
	sum_R_n_a += r
	fsc._nodes[nI]._R_action[a] = sum_R_n_a / (nb_process_samples + fsc._nodes[nI]._visits_action[a] + 1)

	esti_V = fsc._nodes[nI]._R_action[a] + discount * Simulate(pomdp, fsc, sp, nI_next, depth + 1, nb_process_samples, discount, c, epsilon, Q_learning_policy)
	fsc._nodes[nI]._Q_action[a] = fsc._nodes[nI]._Q_action[a] + ((esti_V - fsc._nodes[nI]._Q_action[a]) / fsc._nodes[nI]._visits_action[a])
	fsc._nodes[nI]._V_node = esti_V

	return esti_V
end



function Search(pomdp,
	b,
	dict_weighted_b::Dict{Any, Float64},
	fsc::FSC,
	planner::NaivePlannerPOMCGS)

	# assume an empty fsc
	node_start = CreatNode(b, dict_weighted_b, fsc._action_space)
	HeuristicNodeQ(node_start, planner._Q_learning_policy)
	push!(fsc._nodes, node_start)

	vec_episodes = Vector{Int64}()
	vec_evaluation_value = Vector{Float64}()
	vec_fsc_size = Vector{Int64}()

	sum_planning_time_secs = 0
	for i in 1:planner._nb_iter
		elapsed_time = @elapsed begin
			s = rand(b)
			Simulate(pomdp,
				fsc,
				s,
				1,
				0,
				planner._nb_process_action_samples,
				planner._discount,
				planner._softmax_c,
				planner._epsilon,
				planner._Q_learning_policy)
		end
        # println("Simulate time: $elapsed_time")

		sum_planning_time_secs += elapsed_time

		if sum_planning_time_secs > planner._max_planning_secs
			println("Timeout reached")
			break
		end

		if i % planner._nb_sim == 0
            println("--- Iter ", i ÷ planner._nb_sim, " ---")
			# add Prunning
			fsc._prunned_node_list = Prunning(fsc)
			# println("fsc size:", length(fsc._prunned_node_list))
			# println("Node Value: ", node_start._V_node)
			elapsed_time = @elapsed begin
				L = EvaluationWithSimulationFSC(b, pomdp, fsc, discount(pomdp), planner._nb_eval, planner._Log_result._vec_evaluation_value, planner._Log_result._vec_valid_value, planner._Log_result._vec_unvalid_rate)
				U = EvaluateUpperBound(b, pomdp, fsc, planner._Q_learning_policy, discount(pomdp), planner._nb_eval, planner._C_star)
			end
			println("L and U evaluation time: $elapsed_time")
			println("Upper bound value: ", U)
			println("Lower bound value: ", L)
			push!(planner._Log_result._vec_episodes, i)
			push!(planner._Log_result._vec_fsc_size, length(fsc._prunned_node_list))
			push!(planner._Log_result._vec_upper_bound, U)
			if U - L < planner._epsilon
				break
			end
		end
	end
	

	return vec_episodes, vec_evaluation_value, vec_fsc_size

end

#  use weighted sample to accelerate the computation!
function RolloutNode(pomdp, node::FscNode, action_space, depth::Int64)
	sum_result = 0.0

	for (s_key, s_weight) in node._dict_weighted_samples
		V_s_esti = s_weight * Rollout(pomdp, s_key, action_space, depth)
		sum_result += V_s_esti

	end

	return sum_result
	# return sum_result / length(node._state_particles)
end


#  use weighted sample to accelerate the computation!
function RolloutNodeQlearning(pomdp, node::FscNode, Q_learning_policy::Qlearning, depth::Int64)
	sum_result = 0.0

	for (s_key, s_weight) in node._dict_weighted_samples
		V_s_esti = s_weight * RolloutQlearning(pomdp, s_key, Q_learning_policy, depth)
		sum_result += V_s_esti

	end

	return sum_result
end


function RolloutQlearning(pomdp, s, Q_learning_policy::Qlearning, depth::Int64)
	gamma = discount(pomdp)
	return (gamma^depth) * MaxQ(Q_learning_policy, s)
end

function Rollout(pomdp, s, action_space, depth::Int64)
	gamma = discount(pomdp)
	if (gamma^depth) < 0.01 || isterminal(pomdp, s)
		return 0.0
	end

	sp, o, r = @gen(:sp, :o, :r)(pomdp, s, rand(action_space))

	return r + gamma * Rollout(pomdp, sp, action_space, depth + 1)
end

function HeuristicNodeQ(node::FscNode, Q_learning_policy::Qlearning)
	max_value = typemin(Float64)
	for (a, value) in node._Q_action
		value = 0.0
		for (s, pb) in node._dict_weighted_samples
			value += pb * GetQ(Q_learning_policy, s, a)
		end
		# node._Q_action[a] = 0.1 * value
		node._Q_action[a] = 0.1 * value
		if value > max_value
			max_value = value
		end
	end
	return max_value
end

function GetValueQMDP(node::FscNode, Q_learning_policy::Qlearning)
    value = 0.0

    for (s, pb) in node._dict_weighted_samples
        value += pb*GetV(Q_learning_policy, s)
    end

    return value 

end

function AddTransition(dict_trans_func::Dict{Any, Dict{Any, Dict{Any, Float64}}}, a, s, sp)
	if !haskey(dict_trans_func, a)
		dict_trans_func[a] = Dict{Any, Dict{Any, Float64}}()
		dict_trans_func[a][s] = Dict{Any, Float64}()
		dict_trans_func[a][s][sp] = 1.0
	elseif !haskey(dict_trans_func[a], s)
		dict_trans_func[a][s] = Dict{Any, Float64}()
		dict_trans_func[a][s][sp] = 1.0
	elseif !haskey(dict_trans_func[a][s], sp)
		dict_trans_func[a][s][sp] = 1.0
	else
		dict_trans_func[a][s][sp] += 1.0
	end
end

function EstimateTransAndObsFuncs(pomdp, s, a, nb_estimation::Int64,
	dict_trans_func::Dict{Any, Dict{Any, Dict{Any, Float64}}},
	dict_obs_func::Dict{Any, Dict{Any, Dict{Any, Float64}}})

	collection_sp_o = Dict{Any, Dict{Any, Int64}}()
	for i in 1:nb_estimation
		sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
		AddTransition(dict_trans_func, a, s, sp)
		# AddObservation(dict_obs_func, a, sp, o)
		if haskey(collection_sp_o, sp)
			if haskey(collection_sp_o[sp], o)
				collection_sp_o[sp][o] += 1
			else
				collection_sp_o[sp][o] = 1
			end
		else
			collection_sp_o[sp] = Dict{Any, Int64}()
			collection_sp_o[sp][o] = 1
		end
	end

	sum_trans_pb = 0.0
	for (key_sp, pb_sp) in dict_trans_func[a][s]
		sum_trans_pb += pb_sp
	end
	for (key_sp, pb_sp) in dict_trans_func[a][s]
		dict_trans_func[a][s][key_sp] = pb_sp / sum_trans_pb
	end


	for (key_sp, dict_pb_obs) in collection_sp_o
		sum_sp_obs_value = 0
		for (key_o, value_obs) in dict_pb_obs
			sum_sp_obs_value += value_obs
		end

		for (key_o, value_obs) in dict_pb_obs
			if !haskey(dict_obs_func, a)
				dict_obs_func[a] = Dict{Any, Dict{Any, Float64}}()
				dict_obs_func[a][key_sp] = Dict{Any, Float64}()
				dict_obs_func[a][key_sp][key_o] = value_obs / sum_sp_obs_value
			elseif !haskey(dict_obs_func[a], key_sp)
				dict_obs_func[a][key_sp] = Dict{Any, Float64}()
				dict_obs_func[a][key_sp][key_o] = value_obs / sum_sp_obs_value
			elseif !haskey(dict_obs_func[a][key_sp], key_o)
				dict_obs_func[a][key_sp][key_o] = value_obs / sum_sp_obs_value
				# else # danger, estimate existed obs
				#     println()
			end
		end
	end



end


function ComputeProbObs(pomdp, b, a, o, nb_estimation::Int64, dict_trans_func, dict_obs_func, dict_process_a_s)
	res = 0.0
	for (s, pb_s) in b
		if !haskey(dict_process_a_s, Pair(a, s))
			EstimateTransAndObsFuncs(pomdp, s, a, nb_estimation, dict_trans_func, dict_obs_func)
		end
		pb_sp_o = 0.0
		for (sp, pb_sp) in dict_trans_func[a][s]
			if haskey(dict_obs_func[a][sp], o)
				pb_sp_o += pb_sp * dict_obs_func[a][sp][o]
			end
		end

		res += pb_s * pb_sp_o

	end

	return res
end


function ComputeAllProbObs(pomdp, b, a, nb_estimation::Int64, dict_trans_func, dict_obs_func, dict_process_a_s)

	res = Dict{Any, Float64}()
	for (s, pb_s) in b
		if !haskey(dict_process_a_s, Pair(a, s))
			EstimateTransAndObsFuncs(pomdp, s, a, nb_estimation, dict_trans_func, dict_obs_func)
		end
		for (sp, pb_sp) in dict_trans_func[a][s]
			for (o, pb_o) in dict_obs_func[a][sp]
				if haskey(res, o)
					res[o] += pb_s * pb_sp * pb_o
				else
					res[o] = pb_s * pb_sp * pb_o
				end
			end

		end
	end

	return res
end

function UpdateBelief(b, a, o, pb_oba::Float64, dict_trans_func, dict_obs_func)
	b_next = Dict{Any, Float64}()
	for (s, pb_s) in b
		for (sp, pb_sp) in dict_trans_func[a][s]
			if haskey(dict_obs_func[a][sp], o)
				pb_o = dict_obs_func[a][sp][o]
				pb_s_next_obs = (pb_o * pb_sp * pb_s) / pb_oba
				if haskey(b_next, sp)
					b_next[sp] += pb_s_next_obs
				else
					b_next[sp] = pb_s_next_obs
				end
			end
		end
	end
	return b_next
end

function UpdateAllBelief(pomdp, b, a, nb_estimation::Int64, dict_trans_func, dict_obs_func, dict_process_a_s)

	all_pr_oba = ComputeAllProbObs(pomdp, b, a, nb_estimation, dict_trans_func, dict_obs_func, dict_process_a_s)
	all_next_beliefs = Dict{Any, Dict{Any, Float64}}()
	for (s, pb_s) in b
		for (sp, pb_sp) in dict_trans_func[a][s]
			for (o, pb_oba) in all_pr_oba
				if haskey(dict_obs_func[a][sp], o)
					pb_o = dict_obs_func[a][sp][o]
					pb_s_next_obs = (pb_o * pb_sp * pb_s) / pb_oba
					if haskey(all_next_beliefs, o)
						if haskey(all_next_beliefs[o], sp)
							all_next_beliefs[o][sp] += pb_s_next_obs
						else
							all_next_beliefs[o][sp] = pb_s_next_obs
						end
					else
						all_next_beliefs[o] = Dict{Any, Float64}()
						all_next_beliefs[o][sp] = pb_s_next_obs
					end
				end
			end
		end
	end

	return all_next_beliefs, all_pr_oba
end

function CollectSamplesAndBuildNewBeliefsWeightedParticles(pomdp,
	fsc::FSC,
	nI::Int64,
	a,
	nb_process_action_samples::Int64)

	all_oI_weight = Dict{Int64, Float64}()
	all_dict_weighted_samples = Dict{Int64, OrderedDict{Any, Float64}}()
	sum_R_a = 0.0
	sum_all_weights = 0.0

	# prepare data for multi-thread computation 
	num_threads = Threads.nthreads()
	all_dict_weighted_samples_threads = Vector{Dict{Int64, OrderedDict{Any, Float64}}}()
	all_oI_weight_threads = Vector{Dict{Int64, Float64}}()
	sum_R_a_threads = zeros(Float64, num_threads)
	sum_all_weights_threads = zeros(Float64, num_threads)
	all_obs = Set{Int64}()


	for i in 1:num_threads
		push!(all_dict_weighted_samples_threads, Dict{Int64, OrderedDict{Any, Float64}}())
		push!(all_oI_weight_threads, Dict{Int64, Float64}())
	end

	all_keys = collect(keys(fsc._nodes[nI]._dict_weighted_samples))
	Threads.@threads for i in 1:length(all_keys)
		id_thread = Threads.threadid()
		s = all_keys[i]
		w = fsc._nodes[nI]._dict_weighted_samples[s]
		nb_sim = ceil(w * nb_process_action_samples)
		w = 1.0
		for i in 1:nb_sim
			sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
			sum_R_a_threads[id_thread] += r * w
			sum_all_weights_threads[id_thread] += w
			if haskey(all_dict_weighted_samples_threads[id_thread], o)
				all_oI_weight_threads[id_thread][o] += w
				if haskey(all_dict_weighted_samples_threads[id_thread][o], sp)
					all_dict_weighted_samples_threads[id_thread][o][sp] += w
				else
					all_dict_weighted_samples_threads[id_thread][o][sp] = w
				end
			else
				all_dict_weighted_samples_threads[id_thread][o] = OrderedDict{Any, Float64}()
				all_oI_weight_threads[id_thread][o] = w
				all_dict_weighted_samples_threads[id_thread][o][sp] = w
				push!(all_obs, o)
			end
		end
	end
	# merge threads data 
	for id_thread in 1:num_threads
		sum_R_a += sum_R_a_threads[id_thread]
		sum_all_weights += sum_all_weights_threads[id_thread]
	end
	sum_R_a = sum_R_a / sum_all_weights


	# merge threads data 
	for o in all_obs
		all_dict_weighted_samples[o] = OrderedDict{Int64, Float64}()
		all_oI_weight[o] = 0.0
	end

	for id_thread in 1:num_threads
		for o in all_obs
			if haskey(all_dict_weighted_samples_threads[id_thread], o)
				all_dict_weighted_samples[o] = merge(+, all_dict_weighted_samples[o], all_dict_weighted_samples_threads[id_thread][o])
				all_oI_weight[o] += all_oI_weight_threads[id_thread][o]
			end
		end
	end

	return sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples
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


function ExportLogData(planner::NaivePlannerPOMCGS, name::String)
    output_name = name *"xi" * string.(planner._max_b_gap)* "c" *string.(planner._softmax_c)* ".csv"

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
