-module(map2rec).
-include("records.hrl").
-export([
	convert/2
]).

convert(constraint, Map)->
	M_ = maps:to_list(Map),
	lists:foldl(fun
		({morphology,Val},R)->
			R#constraint{morphology = Val};
		({connection_architecture,Val},R)->
			R#constraint{connection_architecture = Val};
		({neural_afs,Val},R)->
			R#constraint{neural_afs = Val};
		({neural_pfns,Val},R)->
			R#constraint{neural_pfns = Val};
		({substrate_plasticities,Val},R)->
			R#constraint{substrate_plasticities = Val};
		({substrate_linkforms,Val},R)->
			R#constraint{substrate_linkforms = Val};
		({neural_aggr_fs,Val},R)->
			R#constraint{neural_aggr_fs = Val};
		({tuning_selection_fs,Val},R)->
			R#constraint{tuning_selection_fs = Val};
		({tuning_duration_f,Val},R)->
			R#constraint{tuning_duration_f = Val};
		({annealing_parameters,Val},R)->
			R#constraint{annealing_parameters = Val};
		({perturbation_ranges,Val},R)->
			R#constraint{perturbation_ranges = Val};
		({agent_encoding_types,Val},R)->
			R#constraint{agent_encoding_types = Val};
		({heredity_types,Val},R)->
			R#constraint{heredity_types = Val};
		({mutation_operators,Val},R)->
			R#constraint{mutation_operators = Val};
		({tot_topological_mutations_fs,Val},R)->
			R#constraint{tot_topological_mutations_fs = Val};
		({population_evo_alg_f,Val},R)->
			R#constraint{population_evo_alg_f = Val};
		({population_fitness_postprocessor_f,Val},R)->
			R#constraint{population_fitness_postprocessor_f = Val};
		({population_selection_f,Val},R)->
			R#constraint{population_selection_f = Val};
		({specie_distinguishers,Val},R)->
			R#constraint{specie_distinguishers = Val};
		({hof_distinguishers,Val},R)->
			R#constraint{hof_distinguishers = Val};
		({objectives,Val},R)->
			R#constraint{objectives = Val};
		(_,R)->
			R
	end, #constraint{}, M_);

convert(pmp, Map)->
	M_ = maps:to_list(Map),
	lists:foldl(fun
		({op_mode,Val},R)->
			R#pmp{op_mode = Val};
		({population_id,Val},R)->
			R#pmp{population_id = Val};
		({survival_percentage,Val},R)->
			R#pmp{survival_percentage = Val};
		({specie_size_limit,Val},R)->
			R#pmp{specie_size_limit = Val};
		({init_specie_size,Val},R)->
			R#pmp{init_specie_size = Val};
		({polis_id,Val},R)->
			R#pmp{polis_id = Val};
		({generation_limit,Val},R)->
			R#pmp{generation_limit = Val};
		({evaluations_limit,Val},R)->
			R#pmp{evaluations_limit = Val};
		({fitness_goal,Val},R)->
			R#pmp{fitness_goal = Val};
		({benchmarker_pid,Val},R)->
			R#pmp{benchmarker_pid = Val};
		({committee_pid,Val},R)->
			R#pmp{committee_pid = Val};
		(_,R)->
			R
	end, #pmp{}, M_);

convert(sensor,Map)->
	M_ = maps:to_list(Map),
	lists:foldl(fun
		({id, Val}, R) ->
			R#sensor{id = Val};
		({name, Val}, R) ->
			R#sensor{name = Val};
		({type, Val}, R) ->
			R#sensor{type = Val};
		({cx_id, Val}, R) ->
			R#sensor{cx_id = Val};
		({scape, Val}, R) ->
			R#sensor{scape = Val};
		({vl, Val}, R) ->
			R#sensor{vl = Val};
		({fanout_ids, Val}, R) ->
			R#sensor{fanout_ids = Val};
		({generation, Val}, R) ->
			R#sensor{generation = Val};
		({format, Val}, R) ->
			R#sensor{format = Val};
		({gt_parameters, Val}, R) ->
			R#sensor{gt_parameters = Val};
		({phys_rep, Val}, R) ->
			R#sensor{phys_rep = Val};
		({vis_rep, Val}, R) ->
			R#sensor{vis_rep = Val};
		({pre_f, Val}, R) ->
			R#sensor{pre_f = Val};
		({post_f, Val}, R) ->
			R#sensor{post_f = Val};
		(_,R)->
			R
	end, #sensor{}, M_);

convert(actuator,Map)->
	M_ = maps:to_list(Map),
	lists:foldl(fun
		({id, Val}, R) ->
			R#actuator{id = Val};
		({name, Val}, R) ->
			R#actuator{name = Val};
		({type, Val}, R) ->
			R#actuator{type = Val};
		({cx_id, Val}, R) ->
			R#actuator{cx_id = Val};
		({scape, Val}, R) ->
			R#actuator{scape = Val};
		({vl, Val}, R) ->
			R#actuator{vl = Val};
		({fanin_ids, Val}, R) ->
			R#actuator{fanin_ids = Val};
		({generation, Val}, R) ->
			R#actuator{generation = Val};
		({format, Val}, R) ->
			R#actuator{format = Val};
		({gt_parameters, Val}, R) ->
			R#actuator{gt_parameters = Val};
		({phys_rep, Val}, R) ->
			R#actuator{phys_rep = Val};
		({vis_rep, Val}, R) ->
			R#actuator{vis_rep = Val};
		({pre_f, Val}, R) ->
			R#actuator{pre_f = Val};
		({post_f, Val}, R) ->
			R#actuator{post_f = Val};
		(_,R)->
			R
	end, #actuator{}, M_).
