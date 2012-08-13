%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This source code and work is provided and developed by Gene I. Sher & DXNN Research Group WWW.DXNNResearch.COM
%
%Copyright (C) 2009 by Gene Sher, DXNN Research Group, CorticalComputer@gmail.com
%All rights reserved.
%
%This code is licensed under the version 3 of the GNU General Public License. Please see the LICENSE file that accompanies this project for the terms of use.
%
%The original release of this source code and the DXNN MK2 system was introduced and explained (architecture and the logic behind it) in my book: Handbook of Neuroevolution Through Erlang. Springer 2012, print ISBN: 978-1-4614-4462-6 ebook ISBN: 978-1-4614-4463-6. 
%%%%%%%%%%%%%%%%%%%% Deus Ex Neural Network :: DXNN %%%%%%%%%%%%%%%%%%%%

-module(genome_mutator).
-compile(export_all).
-include("records.hrl").
-define(DELTA_MULTIPLIER,math:pi()*2).
-define(SAT_LIMIT,math:pi()*2).
-define(SEARCH_PARAMTERS_MUTATION_PROBABILITY,0.1).
-define(ES_MUTATORS,[
	mutate_tuning_selection,
%	mutate_tuning_duration, %This should not be mutated, since it must stay the same amongst all competing agents within a population.
	mutate_tuning_annealing,
	mutate_tot_topological_mutations,
	mutate_heredity_type
]).
-define(ACTUATORS,morphology:get_Actuators(A#agent.morphology)).
-define(SENSORS,morphology:get_Sensors(A#agent.morphology)).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
long_test(TotMutations) when TotMutations > 0->
	genotype:create_test(),
	short_test(TotMutations).

	short_test(0)->
		exoself:start(test,void);
	short_test(Index)->
		test(),
		short_test(Index-1).
%This is a simple function that executes the test() function the number of times with which the long_test/1 function was initially called. The test/0 function executes mutate(test), which applies a random number of mutation operators to the genotype, where that number ranges from 1 to sqrt(Tot_neurons). After all the mutation operators have been applied succesfully, the function executes exoself:start(test,void), mapping the genotype to phenotype, to test whether the resulting NN system is functional.

test()->
	Result = mutate(test),
	case Result of
		{atomic,_} ->
			io:format("******** Mutation Succesful.~n");
		_->
			io:format("******** Mutation Failure:~p~n",[Result])
	end.
%The test/1 function simply tests the mutate/1 function with an agent whose id is 'test'.

test(Agent_Id,Mutator)->
	F = fun()->
		genome_mutator:Mutator(Agent_Id)
	end,
	mnesia:transaction(F).
%test/2 function tests the mutation operator "Mutator" on the agent with an id Agent_Id.

mutate(Agent_Id)->
	random:seed(now()),
	F = fun()->
		mutate_SearchParameters(Agent_Id),
		A = genotype:read({agent,Agent_Id}),
		{TTM_Name,Parameter} = A#agent.tot_topological_mutations_f,
		TotMutations = tot_topological_mutations:TTM_Name(Parameter,Agent_Id),
		OldGeneration = A#agent.generation,
		NewGeneration = OldGeneration+1,
		genotype:write(A#agent{generation = NewGeneration}),
		apply_Mutators(Agent_Id,TotMutations),
		genotype:update_fingerprint(Agent_Id)
%		genotype:speciate(Agent_Id),
%		verify_network(Agent_Id),
%		technome_constructor:update_Stats(DX_Id),
%		technome_constructor:reset_fitness(DX_Id),
%		find_DeadEndNodes(Agent_Id)
	end,
	mnesia:transaction(F).
%The function mutate/1 first updates the generation of the agent to be mutated, then calculates the number of mutation operators to be applied to it by executing the tot_topological_mutations:TTM_Name/2 function, and then finally runs the apply_Mutators/2 function, which mutates the agent. Once the agent is mutated, the function updates its fingerprint by executing genotype:update_finrgerprint/1.

	mutate_SearchParameters(Agent_Id)->
		case random:uniform() < 1 of
			true ->
				TotMutations = random:uniform(length(?ES_MUTATORS)),
				apply_ESMutators(Agent_Id,TotMutations);
			false ->
				ok
		end.
%mutate_SearchParamters/1 with a probability of ?SEARCH_PARAMETERS_MUTATION_PROBABILITY applies a random number between 1 and length(?ES_MUTATORS) of evolutioanry strategy mutation operators from the ?ES_MUTATORS list.

		apply_ESMutators(_Agent_Id,0)->
			ok;
		apply_ESMutators(Agent_Id,MutationIndex)->
			ES_Mutators = ?ES_MUTATORS,
			ES_Mutator = lists:nth(random:uniform(length(ES_Mutators)),ES_Mutators),
			io:format("Evolutionary Strategy Mutation Operator:~p~n",[ES_Mutator]),
			F = fun()->
				genome_mutator:ES_Mutator(Agent_Id)
			end,
			Result=mnesia:transaction(F),
			case Result of
				{atomic,_} ->
					apply_ESMutators(Agent_Id,MutationIndex-1);
				Error ->
					io:format("******** Error:~p~nRetrying with new Mutation...~n",[Error]),
					apply_ESMutators(Agent_Id,MutationIndex-1)
			end.
%apply_ESMutators/2 with uniform distribution chooses a random evolutionary strategy mutation operator from the ?ES_MUTATORS list of such functions, and applies it to the agent. Whether the mutation succcessful or not, the function counts down the total number of mutation operators left to apply. This is to ensure that if the researcher set for each such evolutionary strategy to be static, having only one available mutatable parameter for every agent, the system will eventually try to mutate the strategy TotMutations number of times, and then return to the caller.

	apply_Mutators(_Agent_Id,0)->
		done;
	apply_Mutators(Agent_Id,MutationIndex)->
		Result = apply_NeuralMutator(Agent_Id),
		case Result of
			{atomic,_} ->
				apply_Mutators(Agent_Id,MutationIndex-1);
			Error ->
				io:format("******** Error:~p~nRetrying with new Mutation...~n",[Error]),
				apply_Mutators(Agent_Id,MutationIndex)
		end.
%apply_Mutators/2 applies the set number of successfull mutation operators to the Agent. If a mutaiton operator exits with an error, the function tries another mutaiton operator. It is only after a sucessfull mutation operator is applied that the MutationIndex is decremented.

		apply_NeuralMutator(Agent_Id)->
			F = fun()->
				A = genotype:read({agent,Agent_Id}),
				MutatorsP = A#agent.mutation_operators,
				Mutator = select_random_MO(MutatorsP),
				io:format("Mutation Operator:~p~n",[Mutator]),
				genome_mutator:Mutator(Agent_Id)
			end,
			mnesia:transaction(F).
%apply_NeuralMutator/1 applies the available mutation operators to the NN. Because the genotype is stored in mnesia, if the mutation operator function exits with an error, the database made changes are retracted, and a new mutation operator can then be applied to the agent, as if the previous unsuccessful mutation operator was never applied. The mutation operator to be applied to the agent is chosen randomly from the agent's mutation_operators list.
			
			select_random_MO(MutatorsP)->
				TotSize = lists:sum([SliceSize || {_MO,SliceSize} <- MutatorsP]),
				Choice=random:uniform(TotSize),
				select_random_MO(MutatorsP,Choice,0).
				
				select_random_MO([{MO,SliceSize}|MOs],Choice,Range_From)->
					Range_To = Range_From+SliceSize,
					case (Choice >= Range_From) and (Choice =< Range_To) of
						true ->
							MO;
						false ->
							select_random_MO(MOs,Choice,Range_To)
					end;
				select_random_MO([],_Choice,_Range_From)->
					exit("********ERROR:select_random_MO:: reached [] without selecting a mutation operator.").
%select_random_MO/1, using the analogy of a roulette wheel, first calculates the entire are of the wheel by summing together all the slice sizes of the parts. The function than chooses randomly a spot on the wheel, and through select_random_MO/3 calculates where that spot is located, with regards to the mutation operator that it falls on. Since some slices are larger then others, they will have uniformly larger probabilities of being selected.

mutate_tuning_selection(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	case (A#agent.constraint)#constraint.tuning_selection_fs -- [A#agent.tuning_selection_f] of
		[] ->
			exit("********ERROR:mutate_tuning_selection/1:: Nothing to mutate, only a single function available.");
		Tuning_Selection_Functions->
			New_TSF = lists:nth(random:uniform(length(Tuning_Selection_Functions)),Tuning_Selection_Functions),
			U_A = A#agent{tuning_selection_f = New_TSF},
			genotype:write(U_A)
	end.
%mutate_tuning_selection/1 function checks if there are any other than the currently used tuning selection functions available in the agent's constraint. If there is, then it chooses a random one from this list, and sets the agent's tuning_selection_f to it. If there are no other tuning selection functions, then it exits with an error.
	
mutate_tuning_annealing(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	case (A#agent.constraint)#constraint.annealing_parameters -- [A#agent.annealing_parameter] of
		[] ->
			exit("********ERROR:mutate_tuning_annealing/1:: Nothing to mutate, only a single function available.");
		Tuning_Annealing_Parameters->
			New_TAP = lists:nth(random:uniform(length(Tuning_Annealing_Parameters)),Tuning_Annealing_Parameters),
			U_A = A#agent{annealing_parameter = New_TAP},
			genotype:write(U_A)
	end.
%mutate_annealing_parameter/1 function checks if there are any other than the currently used tuning annealing parameters available in the agent's constraint. If there is, then it chooses a random one from this list, and sets the agent's annealing_parameter to it. If there are no other tuning annealing parameters, then it exits with an error.

mutate_tot_topological_mutations(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	case (A#agent.constraint)#constraint.tuning_selection_fs -- [A#agent.tuning_selection_f] of
		[] ->
			exit("********ERROR:mutate_tuning_selection/1:: Nothing to mutate, only a single function available.");
		Tuning_Selection_Functions->
			New_TSF = lists:nth(random:uniform(length(Tuning_Selection_Functions)),Tuning_Selection_Functions),
			U_A = A#agent{tuning_selection_f = New_TSF},
			genotype:write(U_A)
	end.
%mutate_tot_topological_mutations/1 function checks if there are any other than the currently used tuning tot topological mutation functions available in the agent's constraint. If there is, then it chooses a random one from this list, and sets the agent's tot_topological_mutations_f to it. If there are no other functions that can calculate tot topological mutations, then it exits with an error.

mutate_heredity_type(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	case (A#agent.constraint)#constraint.heredity_types -- [A#agent.heredity_type] of
		[] ->
			exit("********ERROR:mutate_heredity_type/1:: Nothing to mutate, only a single function available.");
		Heredity_Type_Pool->
			New_HT = lists:nth(random:uniform(length(Heredity_Type_Pool)),Heredity_Type_Pool),
			U_A = A#agent{heredity_type = New_HT},
			genotype:write(U_A)
	end.
%mutate_heredity_type/1 function checks if there are any other heredity types in the agent's constraint record. If any other than the one currently used by the agent are present, the agent exchanges the heredity type it currently uses to a random one from the remaining list. If no other heredity types are available, the mutation operator exits with an error, and the neuroevolutionary system tries another mutation operator.

mutate_weights(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,

	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	N = genotype:read({neuron,N_Id}),
	Input_IdPs = N#neuron.input_idps,
	U_Input_IdPs = perturb_IdPs(Input_IdPs),
	U_N = N#neuron{input_idps = U_Input_IdPs},
	EvoHist = A#agent.evo_hist,
	U_EvoHist = [{mutate_weights,N_Id}|EvoHist],
	U_A = A#agent{evo_hist = U_EvoHist},
	genotype:write(U_N),
	genotype:write(U_A).
%The mutate_weights/1 function accepts the Agent_Id parameter, extracts the NN's cortex, and then chooses a random neuron belonging to the NN with a uniform distribution probability. Then the neuron's input_idps list is extracted, and the function perturb_IdPs/1 is used to perturb/mutate the weights. Once the Input_IdPs have been perturbed, the agent's evolutionary history, EvoHist is updated to include the successfully applied mutate_weights mutation operator. Then the updated Agent and the updated neuron are written to the database.
	
	perturb_IdPs(Input_IdPs)->
		Tot_WeightsP=lists:sum([length(WeightsP) || {_Input_Id,WeightsP}<-Input_IdPs]),
		MP = 1/math:sqrt(Tot_WeightsP),
		perturb_IdPs(MP,Input_IdPs,[]).
	perturb_IdPs(MP,[{Input_Id,WeightsP}|Input_IdPs],Acc)->
		U_WeightsP = perturb_weightsP(MP,WeightsP,[]),
		perturb_IdPs(MP,Input_IdPs,[{Input_Id,U_WeightsP}|Acc]);
	perturb_IdPs(_MP,[],Acc)->
		lists:reverse(Acc).
%perturb_IdPs/1 accepts the Input_IdPs list of the format:[{Id,Weights}...], calculates the total number of weights in the Input_IdPs, and then calculates the mutation probability MP, which is 1/sqrt(Tot_Weights). Once the mutation probability is calculated, each weight in the Input_IdPs list has a chance of MP to be perturbed/mutated. Once all the weights in the Input_IdPs list had a chance of being mutated, the updated Input_IdPs is returned to the caller.

	perturb_weightsP(MP,[{W,PL}|Weights],Acc)->
		U_W = case random:uniform() < MP of
			true->
				sat((random:uniform()-0.5)*?DELTA_MULTIPLIER+W,-?SAT_LIMIT,?SAT_LIMIT);
			false ->
				W
		end,
		perturb_weightsP(MP,Weights,[{U_W,PL}|Acc]);
	perturb_weightsP(_MP,[],Acc)->
		lists:reverse(Acc).
%perturb_weightsP/3 is called with the mutation probability MP, a weights list, and an empty list, [], to be used as an accumulator. The function goes through every weight, where every weight has a chance of MP to be mutated/perturbed. The perturbations have a random intensity between -Pi/2 and Pi/2. Once all the weights in the weights list had a chance of being perturbed, the updated weights list is reversed back to its original order, and returned back to the caller.
		sat(Val,Min,Max)->
			if
				Val < Min -> Min;
				Val > Max -> Max;
				true -> Val
			end.
%sat/3 calculates whether Val is between Min and Max values, if it is, then val is returned as is. If Val is less than Min, then Min is returned, if Val is greater than Max, then Max is returned.

add_bias(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	Generation = A#agent.generation,
	
	N = genotype:read({neuron,N_Id}),
	SI_IdPs = N#neuron.input_idps,
	MI_IdPs = N#neuron.input_idps_modulation,
	{PFName,_NLParameters} = N#neuron.pf,
	case {lists:keymember(bias,1,SI_IdPs), lists:keymember(bias,1,MI_IdPs), PFName == neuromodulation, random:uniform(2)} of
		{_,false,true,2} ->
			U_MI_IdPs = lists:append(MI_IdPs,[{bias,[{random:uniform()-0.5,plasticity:PFName(weight_parameters)}]}]),
			U_N = N#neuron{
				input_idps_modulation = U_MI_IdPs,
				generation = Generation},
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{{add_bias,m},N_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_N),
			genotype:write(U_A);
		{true,_,_,_} ->
			exit("********ERROR:add_bias:: This Neuron already has a bias in input_idps.");
		{false,_,_,_} ->
			U_SI_IdPs = lists:append(SI_IdPs,[{bias,[{random:uniform()-0.5,plasticity:PFName(weight_parameters)}]}]),
			U_N = N#neuron{
				input_idps = U_SI_IdPs,
				generation = Generation},
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{{add_bias,s},N_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_N),
			genotype:write(U_A)
	end.
%The add_bias/1 function is called with the Agent_Id parameter. The function first extracts the neuron_ids list from the cortex element and chooses a random neuron from the id list. After the neuron is read from the database, we check whether input_idps and input_idps_modulation lists already have bias, and we randomly generate a value 1 or 2. If the value 1 is generated and the input_idps list does not have a bias, it is added. If the value 2 is generated, and the input_idps_modulation does not have a bias, it is added. Otherwise an error is returned. 

remove_bias(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	Generation = A#agent.generation,
	
	N = genotype:read({neuron,N_Id}),
	SI_IdPs = N#neuron.input_idps,
	MI_IdPs = N#neuron.input_idps_modulation,
	{PFName,_NLParameters} = N#neuron.pf,
	case {lists:keymember(bias,1,SI_IdPs), lists:keymember(bias,1,MI_IdPs), PFName == neuromodulation, random:uniform(2)} of
		{_,true,true,2} ->%Remove modulatory bias
			U_MI_IdPs = lists:keydelete(bias,1,MI_IdPs),
			U_N = N#neuron{
				input_idps_modulation = U_MI_IdPs,
				generation = Generation},
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{{remove_bias,m},N_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_N),
			genotype:write(U_A);
		{false,_,_,_} ->
			exit("********ERROR:remove_bias:: This Neuron does not have a bias in input_idps.");
		{true,_,_,_} ->%Remove synaptic bias
			U_SI_IdPs = lists:keydelete(bias,1,SI_IdPs),
			U_N = N#neuron{
				input_idps = U_SI_IdPs,
				generation = Generation},
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{{remove_bias,s},N_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_N),
			genotype:write(U_A)
	end.
%The remove_bias/1 function is called with the Agent_Id parameter. The function first extracts the neuron_ids list from the cortex element and chooses a random neuron from the id list. After the neuron is read from the database, we check whether input_idps and input_idps_modulation lists already have bias, and we randomly generate a value 1 or 2. If the value 1 is generated and the input_idps list has a bias, it is removed. If the value 2 is generated, and the input_idps_modulation has a bias, it is removed. Otherwise an error is returned. 

mutate_af(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	Generation = A#agent.generation,
	
	N = genotype:read({neuron,N_Id}),
	AF = N#neuron.af,
	case (A#agent.constraint)#constraint.neural_afs -- [AF] of
		[] ->
			exit("********ERROR:mutate_af:: There are no other activation functions to use.");
		Activation_Functions ->
			NewAF = lists:nth(random:uniform(length(Activation_Functions)),Activation_Functions),
			U_N = N#neuron{af=NewAF,generation=Generation},
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{mutate_af,N_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_N),
			genotype:write(U_A)
	end.
%The mutate_af/1 function chooses a random neuron, and then changes its currently used activation function into another one available from the neural_afs list of the agent's constraint record.

mutate_pf(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	Generation = A#agent.generation,
	
	N = genotype:read({neuron,N_Id}),
	{PFName,_NLParameters} = N#neuron.pf,
	case (A#agent.constraint)#constraint.neural_pfns -- [PFName] of
		[] ->
			exit("********ERROR:mutate_pf:: There are no other plasticity functions to use.");
		Other_PFNames ->
			New_PFName = lists:nth(random:uniform(length(Other_PFNames)),Other_PFNames),
			New_NLParameters = plasticity:New_PFName(neural_parameters),
			NewPF = {New_PFName,New_NLParameters},
			InputIdPs = N#neuron.input_idps,
			U_InputIdPs = [{Input_IdP,plasticity:New_PFName(weight_parameters)} || {Input_IdP,_OldPL} <- InputIdPs],
			U_N = N#neuron{pf=NewPF,input_idps = U_InputIdPs, generation=Generation},
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{mutate_pf,N_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_N),
			genotype:write(U_A)
	end.
%The mutate_pf/1 function chooses a random neuron, and then changes its currently used plasticity function into another one available from the neural_pfs list of the agent's constraint record.

mutate_plasticity_parameters(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	N = genotype:read({neuron,N_Id}),
	{PFName,_Parameters} = N#neuron.pf,
	U_N = plasticity:PFName({N_Id,mutate}),
	EvoHist = A#agent.evo_hist,
	U_EvoHist = [{mutate_plasticity_parameters,N_Id}|EvoHist],
	U_A = A#agent{evo_hist=U_EvoHist},
	genotype:write(U_N),
	genotype:write(U_A).
%The mutate_plasticity_parameters/1 chooses a random neuron from the NN, and mutates the parameters of its plasticity function, if present.

mutate_aggrf(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	Generation = A#agent.generation,
	
	N = genotype:read({neuron,N_Id}),
	AggrF = N#neuron.aggr_f,
	case (A#agent.constraint)#constraint.neural_aggr_fs -- [AggrF] of
		[] ->
			exit("********ERROR:mutate_aggrf:: There are no other aggregation functions to use.");
		Aggregation_Functions ->
			NewAggrF = lists:nth(random:uniform(length(Aggregation_Functions)),Aggregation_Functions),
			U_N = N#neuron{aggr_f=NewAggrF,generation=Generation},
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{mutate_aggrf,N_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_N),
			genotype:write(U_A)
	end.
%The mutate_aggrf/1 function chooses a random neuron, and then changes its currently used aggregation function into another one available from the neural_aggr_fs list of the agent's constraint record.
	
link_FromElementToElement(Generation,From_ElementId,To_ElementId)->
	case {From_ElementId,To_ElementId} of
		{{_FromSId,neuron},{_ToSId,neuron}} ->
			link_FromNeuronToNeuron(Generation,From_ElementId,To_ElementId);
		{{_FromSId,sensor},{_ToSId,neuron}} ->
			link_FromSensorToNeuron(Generation,From_ElementId,To_ElementId);
		{{_FromNId,neuron},{_ToAId,actuator}} ->
			link_FromNeuronToActuator(Generation,From_ElementId,To_ElementId)
	end.
%The function link_FromElementToElement/3 first calculates what type of link is going to be established (neuron to neuron, sensor to neuron, or neuron to actuator), and then calls the specific linking function based on that.

link_FromNeuronToNeuron(Generation,From_NeuronId,To_NeuronId)->
	%A = genotype:read({agent,Agent_Id}),
	%Generation = A#agent.generation,
%From Partf
	FromN = genotype:read({neuron,From_NeuronId}),
	U_FromN = link_FromNeuron(FromN,To_NeuronId,Generation),
	genotype:write(U_FromN),
%To Part
	ToN = genotype:read({neuron,To_NeuronId}),%We read it afterwards, in the case that it's the same Element. Thus we do not overwrite the earlier changes.
	FromOVL = 1,
	U_ToN = link_ToNeuron(From_NeuronId,FromOVL,ToN,Generation),
	genotype:write(U_ToN).
%link_FromNeuronToNeuron/3 establishes a link from neuron with id From_NeuronId, to a neuron with id To_NeuronId. The function then calls link_FromNeuron/4, which establishes the link on the From_NeuronId's side. The updated neuron associated with the From_NeuronId is then written to database. To decide how long the weight list that is going to be added to the To_NeuronId's input_idps, the function calculates From_NeuronId's output vector length. Since the connection is from a neuron, FromOVL is set to 1. link_ToNeuron/4 is then called, and the link is established on the To_NeuronId's side. Finally, the updated neuron associated with the id To_NeuronId is written to database. The order of reading the FromN and ToN neuron records from the database is important. It is essential that ToN is read after the U_FromN is written to database, in the case that From_NeuronId and To_NeuronId refer to the same neuron (a recurrent connection from the neuron to itself). If both neurons are read at the same time for example before the links are established, then the link established in the U_FromN will be overwritten when the U_ToN is written to file. Thus order is important in this function.

	link_FromNeuron(FromN,ToId,Generation)->
		{{FromLI,_},_} = FromN#neuron.id,
		{{ToLI,_},_} = ToId,
		FromOutput_Ids = FromN#neuron.output_ids,
		FromRO_Ids = FromN#neuron.ro_ids,
		case lists:member(ToId, FromOutput_Ids) of
			true ->
				exit("******** ERROR:add_NeuronO[can not add O_Id to Neuron]: ~p already a member of ~p~n",[ToId,FromN#neuron.id]);
			false ->
				{U_FromOutput_Ids,U_FromRO_Ids} = case FromLI >= ToLI of
					true ->
						{[ToId|FromOutput_Ids],[ToId|FromRO_Ids]};
					false ->
						{[ToId|FromOutput_Ids],FromRO_Ids}
				end,
				FromN#neuron{
					output_ids = U_FromOutput_Ids,
					ro_ids = U_FromRO_Ids,
					generation = Generation
				}
		end.
%link_FromNeuron/4 updates the record of the neuron from whom the link is being created. FromN is the record of the neuron from whom the link/connection eminates, and ToId is the id of the element to whom the link is headed towards. The function extracts the layer index of the neuron FromN, and the layer index of the element with the id ToId. Then the two layer indecies are compared, and the ToId is either added only to the FromN's output_ids list, or if the connection is recursive, ToLayerIndex =< FromLayerIndex, to output_ids and ro_ids lists. The FromN's generation is updated to the value Generation, which is the current, most recent generation, since this neuron has just been modified. Finally, the updated neuron record is then returned to the caller. On the other hand, if ToId, the id of the element to which the connection is being established, is already a member of the FromN's output_ids list, then the function exits with error.

	link_ToNeuron(FromId,FromOVL,ToN,Generation)->%TODO: Only allows a single connection from a presynaptic element.
		ToSI_IdPs = ToN#neuron.input_idps,
		ToMI_IdPs = ToN#neuron.input_idps_modulation,
		{PFName,_NLParameters}=ToN#neuron.pf,
		case {lists:keymember(FromId,1,ToSI_IdPs),lists:keymember(FromId,1,ToMI_IdPs)} of
			{false,false} ->
				case {PFName == neuromodulation, random:uniform(2)} of
					{true,2} ->
						U_ToMI_IdPs = [{FromId, genotype:create_NeuralWeightsP(PFName,FromOVL,[])}|ToMI_IdPs],
						ToN#neuron{
							input_idps = U_ToMI_IdPs,
							generation = Generation
						};
					_ ->
						U_ToSI_IdPs = [{FromId, genotype:create_NeuralWeightsP(PFName,FromOVL,[])}|ToSI_IdPs],
						ToN#neuron{
							input_idps = U_ToSI_IdPs,
							generation = Generation
						}
				end;
			_ ->
				exit("ERROR:add_NeuronI::[can not add I_Id]: ~p already connected to ~p~n",[FromId,ToN#neuron.id])
		end.
%link_ToNeuron/4 updates the record of ToN, so that its updated to receive a connection from the element FromId. The link eminates from element with the id FromId, whose output vector length is FromOVL, and the connection is made to the neuron ToN, the record whose is updated in this function. Randomly chosen, either the ToN's input_idps_modulation or input_idps list is updated with the tuple {FromId,[{W_1,WPs}...{W_FromOVL,WPs}]}, then the neuron's generation is updated to Generation (the current, most recent generation), and the updated ToN's record is returned to the caller. On the other hand, if the FromId is already part of the ToN's input_idps or input_idps_modulation list (dependent on which was randomly chosen), which means that the standard or modulatory link already exists between the neuron ToN and element FromId, the the function exits with an error.

link_FromSensorToNeuron(Generation,From_SensorId,To_NeuronId)->
	%A = genotype:read({agent,Agent_Id}),
	%Generation = A#agent.generation,
%From Part
	FromS = genotype:read({sensor,From_SensorId}),
	U_FromS = link_FromSensor(FromS,To_NeuronId,Generation),
	genotype:write(U_FromS),
%To Part
	ToN = genotype:read({neuron,To_NeuronId}),
	FromOVL = FromS#sensor.vl,
	U_ToN = link_ToNeuron(From_SensorId,FromOVL,ToN,Generation),
	genotype:write(U_ToN).
%The function link_FromSensorToNeuron/3 establishes a connection from the sensor with id From_SensorId, and the neuron with id To_NeuronId. First the sensor record is updated with the connection details using the function link_FromSensor, and the updated sensor record is written to database. Then the record of the neuron to whom the link is being established is updated using the function link_ToNeuron/4, after which the updated neuron is written to database.

	link_FromSensor(FromS,ToId,Generation)->
		FromFanout_Ids = FromS#sensor.fanout_ids,
		case lists:member(ToId, FromFanout_Ids) of
			true ->
				exit("******** ERROR:link_FromSensor[can not add ToId to Sensor]: ~p already a member of ~p~n",[ToId,FromS#sensor.id]);
			false ->
				FromS#sensor{
					fanout_ids = [ToId|FromFanout_Ids],
					generation=Generation
				}
		end.
%The function link_FromSensor/2 updates the record of the sensor FromS, from whom the link eminates towards the element with id ToId. First the function ensures that there is no connection yet established between FromS and ToId, if a connection between these two elements already exists, then the function exits with error. If there is no connection between the two elements, then ToId is added to the sensor's fanout_ids list, and the updated record of the sensor is returned to the caller.
		
link_FromNeuronToActuator(Generation,From_NeuronId,To_ActuatorId)->
	%A = genotype:read({agent,Agent_Id}),
	%Generation = A#agent.generation,
%From Part
	FromN = genotype:read({neuron,From_NeuronId}),
	U_FromN = link_FromNeuron(FromN,To_ActuatorId,Generation),
	genotype:write(U_FromN),
%To Part
	ToA = genotype:read({actuator,To_ActuatorId}),
	Fanin_Ids = ToA#actuator.fanin_ids,
	case length(Fanin_Ids) >= ToA#actuator.vl of
		true ->
			exit("******** ERROR:link_FromNeuronToActuator:: Actuator already fully connected");
		false ->
			U_Fanin_Ids = [From_NeuronId|Fanin_Ids],
			genotype:write(ToA#actuator{
				fanin_ids = U_Fanin_Ids,
				generation=Generation})
	end.
%The function Link_FromNeuronToActuator/4 establishes a link eminating from the neuron with an id From_NeuronId, to an actuator with the id To_ActuatorId. First the From_NeuronId's record is updated using the function link_FromNeuron/3, after which the updated neuron record is written to database. Then the function checks whether the actuator to which the neuron is establishing the link, still has space for that link (length(Fanin_Ids) is less than the actuator's vector length, vl). If there is no more room, then the function exits with error, if there is room, then the actuator's fanin_ids list is updated by appending to it the id of the neuron's id. The updated actuator is then written to the database.

cutlink_FromElementToElement(Generation,From_ElementId,To_ElementId)->
	case {From_ElementId,To_ElementId} of
		{{_FromId,neuron},{_ToId,neuron}} ->
			cutlink_FromNeuronToNeuron(Generation,From_ElementId,To_ElementId);
		{{_FromId,sensor},{_ToId,neuron}} ->
			cutlink_FromSensorToNeuron(Generation,From_ElementId,To_ElementId);
		{{_FromId,neuron},{_ToId,actuator}} ->
			cutlink_FromNeuronToActuator(Generation,From_ElementId,To_ElementId)
	end.
%cutlink_FromElementToElement/3 first checks which of the three types of connections is between the From_ElementId and To_ElementId (neuron to neuron, sensor to neuron, or neuron to actuator), and then disconnects the two elements using one of the three specialised cutlink_... functions.

cutlink_FromNeuronToNeuron(Generation,From_NeuronId,To_NeuronId)->
	%A = genotype:read({agent,Agent_Id}),
	%Generation = A#agent.generation,
%From Part
	FromN = genotype:read({neuron,From_NeuronId}),
	U_FromN = cutlink_FromNeuron(FromN,To_NeuronId,Generation),
	genotype:write(U_FromN),
%To Part
	ToN = genotype:read({neuron,To_NeuronId}),
	U_ToN = cutlink_ToNeuron(From_NeuronId,ToN,Generation),
	genotype:write(U_ToN).
%The cutlink_FromNeuronToNeuron/3 function disconnections the connection from the From_NeuronId to the To_NeuronId. The function first disconnects the neuron associated with From_NeuronId by calling the cutlink_FromNeuron/3, and then writing to database the updated neuron. The function then disconnects the neuron associated with the To_NeuronId from the connection using the cutlink_ToNeuron/3, and writes to database the updated ToN record. If the From_NeuronId and the To_NeuronId are ids of the same neuron, then it is important to first write U_FromN to database, before reading the ToN neuron from the database, so as not to lose the update made by the cutlink_FromNeuron/3, before reading the updated neuron from the database and calling the cutlink_ToNeuron. Thus this order of reading and writing the neurons from the database is essential to cover the corner cases.

	cutlink_FromNeuron(FromN,ToId,Generation)->
		FromOutput_Ids = FromN#neuron.output_ids,
		FromRO_Ids = FromN#neuron.ro_ids,
		case lists:member(ToId, FromOutput_Ids) of
			true ->
				U_FromOutput_Ids = FromOutput_Ids--[ToId],
				U_FromRO_Ids = FromRO_Ids--[ToId],%Not necessary if not recursive...
				FromN#neuron{
					output_ids = U_FromOutput_Ids,
					ro_ids = U_FromRO_Ids,
					generation = Generation};
			false ->
				exit("ERROR:: cutlink_FromNeuron [can not remove O_Id]: ~p not a member of ~p~n",[ToId,FromN#neuron.id])
		end.
%cutlink_FromNeuron/3 cuts the connection on the FromNeuron (FromN) side. The function first checks if the ToId is a member of the output_ids list, if its not then the function exits with an error. If the ToId is a member of the output_ids list, then the function removes the ToId from the FromOutput_Ids and from the FromRO_Ids. Even if the ToId is a recursive connection, then removing it from ro_ids updates the FromRO_Ids list, if its not, then no change is made to the ro_ids list. Once the lists are updated, the updated neuron record of FromN is returned to the caller.

	cutlink_ToNeuron(FromId,ToN,Generation)->
		ToSI_IdPs = ToN#neuron.input_idps,
		ToMI_IdPs = ToN#neuron.input_idps_modulation,
		Guard1 = lists:keymember(FromId, 1, ToSI_IdPs),
		Guard2 = lists:keymember(FromId, 1, ToMI_IdPs),
		if 
			Guard1->
				U_ToSI_IdPs = lists:keydelete(FromId,1,ToSI_IdPs),
				ToN#neuron{
					input_idps = U_ToSI_IdPs,
					generation = Generation};
			Guard2 ->
				U_ToMI_IdPs = lists:keydelete(FromId,1,ToMI_IdPs),
				ToN#neuron{
					input_idps = U_ToMI_IdPs,
					generation = Generation};
			true ->
				exit("ERROR[can not remove I_Id]: ~p not a member of ~p~n",[FromId,ToN#neuron.id])
		end.
%cutlink_ToNeuron/3 cuts the connection on the ToNeuron (ToN) side. The function first checks if the FromId is a member of the ToN's input_idps list, if its not, then the function checks if it is a member of the input_idps_modulation list. If it is not a member of either, the function exits with error. If FromId is a member of one of these lists, then that tuple is removed from that list, and the updated ToN record is returned to the caller.

cutlink_FromSensorToNeuron(Generation,From_SensorId,To_NeuronId)->
	%A = genotype:read({agent,Agent_Id}),
	%Generation = A#agent.generation,
%From Part
	FromS = genotype:read({sensor,From_SensorId}),
	U_FromS = cutlink_FromSensor(FromS,To_NeuronId,Generation),
	genotype:write(U_FromS),
%To Part
	ToN = genotype:read({neuron,To_NeuronId}),
	U_ToN = cutlink_ToNeuron(From_SensorId,ToN,Generation),
	genotype:write(U_ToN).
%The cutlink_FromSensorToNeuron/3 cuts the connection from the From_SensorId to To_NeuronId. The function first cuts the cunnection on the From_SensorId side using the cutlink_FromSensor/3 side, and writes the updated sensor to database. The function then cuts the connection on the To_NeuronId side using the cutlink_ToNeuron/3 function, and writes the updated neuron record to database.
	
	cutlink_FromSensor(FromS,ToId,Generation)->
		FromFanout_Ids = FromS#sensor.fanout_ids,
		case lists:member(ToId, FromFanout_Ids) of
			true ->
				U_FromFanout_Ids = FromFanout_Ids--[ToId],
				FromS#sensor{
					fanout_ids = U_FromFanout_Ids,
					generation=Generation};
			false ->
				exit("ERROR:: cutlink_FromSensor [can not remove ToId]: ~p not a member of ~p~n",[ToId,FromS#sensor.id])
		end.
%The cutlink_FromSensor/3 function first checks whether ToId is a member of the sensor's FromS fanout_ids list. If its not, then the function exits with an error. If ToId is a member of FromS's fanout_ids list, then it is removed from that list, and the updated sensor record of FromS is returned to the caller.

cutlink_FromNeuronToActuator(Generation,From_NeuronId,To_ActuatorId)->
	%A = genotype:read({agent,Agent_Id}),
	%Generation = A#agent.generation,
%From Part
	FromN = genotype:read({neuron,From_NeuronId}),
	U_FromN = cutlink_FromNeuron(FromN,To_ActuatorId,Generation),
	genotype:write(U_FromN),
%To Part
	ToA = genotype:read({actuator,To_ActuatorId}),
	U_ToA = cutlink_ToActuator(From_NeuronId,ToA,Generation),
	genotype:write(U_ToA).
%cutlink_FromNeuronToActuator/3 cuts the connection from the From_NeuronId to To_ActuatorId. The function first cuts the connection on the From_NeuronId side using the cutlink_FromNeuron/3 function, and writes the updated U_FromN to database. Then the connection on the To_ActuatorId is cut using the cutlink_ToActuator/3 function, after which the updated actuator record is written to the database.

	cutlink_ToActuator(FromId,ToA,Generation)->
		ToFanin_Ids = ToA#actuator.fanin_ids,
		case lists:member(FromId, ToFanin_Ids) of
			true ->
				U_ToFanin_Ids = ToFanin_Ids--[FromId],
				ToA#actuator{
					fanin_ids = U_ToFanin_Ids,
					generation=Generation};
			false ->
				exit("ERROR:: cutlink_ToActuator [can not remove FromId]: ~p not a member of ~p~n",[FromId,ToA])
		end.
%The cutlink_ToActuator/3 function cuts the connection on the ToActuator's side. The function first checks if the FromId is a member of the actuator ToA's fanin_ids list. If its not, the function exits with an error. If FromId is a member of the actuator's fanin_ids list, then the id is removed from the list, and the updated actuator record is returned to the caller.

add_outlink(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	%A_Ids = Cx#cortex.actuator_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	N = genotype:read({neuron,N_Id}),
	Output_Ids = N#neuron.output_ids,
	Outlink_NIdPool = filter_OutlinkIdPool(A#agent.constraint,N_Id,N_Ids),
	case Outlink_NIdPool -- Output_Ids of
		[] ->
			exit("********ERROR:add_outlink:: Neuron already connected to all ids");
		Available_Ids ->
			To_Id = lists:nth(random:uniform(length(Available_Ids)),Available_Ids),
			link_FromElementToElement(A#agent.generation,N_Id,To_Id),
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{add_outlink,N_Id,To_Id}|EvoHist],
			U_A = A#agent{evo_hist=U_EvoHist},
			genotype:write(U_A)
	end.
%The add_outlink/1 function reads the cortex record from the database based on the cortex id extracted from the agent record. The function then selects a random neuron from the neuron_ids stored in the cortex record. The function then subtracts the neuron's output_ids from the combined list of the actuator and neuron ids belonging to the neural network to get a list of ids belonging to the elements to which this neuron is not yet connected. If this list is empty, the function exits with error. If the list is not empty, it is given the name Available_Ids, from which a random id is chosen, and the neuron is then connected to the element to whom the id belongs. Finally, the agent's evo_hist list is updated, and the updated agent record is written to the database.

	filter_OutlinkIdPool(C,N_Id,N_Ids)->
		case C#constraint.connection_architecture of
			recurrent ->
				N_Ids;
			feedforward ->
				{{LI,_},neuron} = N_Id,
				[{{Outlink_LI,Outlink_UniqueId},neuron} || {{Outlink_LI,Outlink_UniqueId},neuron} <- N_Ids, Outlink_LI > LI]
		end.
%The function filter_OutlinkIdPool/3 uses the connection_architecture specification in the constraint record of the agent to return a filtered neuron id pool. For the feedforward connection_architecture, the function ensures that only the neurons in the forward facing layers are allowed in the id pool.

add_inlink(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	S_Ids = case A#agent.encoding_type of
		neural ->
			Cx#cortex.sensor_ids;
		substrate ->
			Substrate_Id=A#agent.substrate_id,
			Substrate=genotype:read({substrate,Substrate_Id}),
			Substrate#substrate.cpp_ids
	end,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	N = genotype:read({neuron,N_Id}),
	{I_Ids,_WeightPLists} = lists:unzip(N#neuron.input_idps),
	Inlink_NIdPool = filter_InlinkIdPool(A#agent.constraint,N_Id,N_Ids),
	case lists:append(S_Ids,Inlink_NIdPool) -- I_Ids of
		[] ->
			exit("********ERROR:add_INLink:: Neuron already connected from all ids");
		Available_Ids ->
			From_Id = lists:nth(random:uniform(length(Available_Ids)),Available_Ids),
			link_FromElementToElement(A#agent.generation,From_Id,N_Id),
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{add_inlink,From_Id,N_Id}|EvoHist],
			genotype:write(A#agent{evo_hist=U_EvoHist})
	end.
%The add_inlink/1 function extracts the list of neuron ids within the NN, and chooses a random id from this list. The input ids belonging to the neuron's input_Idps list are then subtracted from the combined neuron and sensor ids belonging to the NN. The result is a list of element ids from which the neuron is not yet connected. If this list is empty, the function exits with an error, otherwise the function chooses a random id from this list and establishes a connection between the neuron and the element correlated with randomly chosen id. Finally, the agent's evo_hist list is updated, and the updated agent is written to database.

	filter_InlinkIdPool(C,N_Id,N_Ids)->
		case C#constraint.connection_architecture of
			recurrent ->
				N_Ids;
			feedforward ->
				{{LI,_},neuron} = N_Id,
				[{{Inlink_LI,Inlink_UniqueId},neuron} || {{Inlink_LI,Inlink_UniqueId},neuron} <- N_Ids, Inlink_LI < LI]
		end.
%The function filter_InlinkIdPool/3 uses the connection_architecture specification in the constraint record of the agent to return a filtered neuron id pool. For the feedforward connection_architecture, the function ensures that  only the neurons in the previous layers are allowed in the filtered neuron id pool.

add_neuron(Agent_Id)->%Adds neuron and connects it to other neurons, not sensors or actuators.
	A = genotype:read({agent,Agent_Id}),
	Generation = A#agent.generation,
	Pattern = A#agent.pattern,
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	S_Ids = case A#agent.encoding_type of
		neural ->
			Cx#cortex.sensor_ids;
		substrate ->
			Substrate_Id=A#agent.substrate_id,
			Substrate=genotype:read({substrate,Substrate_Id}),
			Substrate#substrate.cpp_ids
	end,
	%A_Ids = Cx#cortex.actuator_ids,
	{TargetLayer,TargetNeuron_Ids} = lists:nth(random:uniform(length(Pattern)),Pattern),
	NewN_Id = {{TargetLayer,genotype:generate_UniqueId()},neuron},
	U_N_Ids = [NewN_Id|N_Ids],
	U_Pattern = lists:keyreplace(TargetLayer, 1, Pattern, {TargetLayer,[NewN_Id|TargetNeuron_Ids]}),
	SpecCon = A#agent.constraint,
	genotype:construct_Neuron(Cx_Id,Generation,SpecCon,NewN_Id,[],[]),
	Inlink_NIdPool = filter_InlinkIdPool(A#agent.constraint,NewN_Id,N_Ids),
	Outlink_NIdPool = filter_OutlinkIdPool(A#agent.constraint,NewN_Id,N_Ids),
	FromElementId_Pool = Inlink_NIdPool++S_Ids,
	ToElementId_Pool = Outlink_NIdPool,
	case (FromElementId_Pool == []) or (ToElementId_Pool == []) of
		true ->
			exit("********ERROR::add_neuron(Agent_Id)::Can't add new neuron here, Inlink_NIdPool or Outlink_NIdPool is empty.");
		false ->
			From_ElementId = lists:nth(random:uniform(length(FromElementId_Pool)),FromElementId_Pool),
			To_ElementId = lists:nth(random:uniform(length(ToElementId_Pool)),ToElementId_Pool),
			link_FromElementToElement(A#agent.generation,From_ElementId,NewN_Id),
			link_FromElementToElement(A#agent.generation,NewN_Id,To_ElementId),
			U_EvoHist = [{add_neuron,From_ElementId,NewN_Id,To_ElementId}|A#agent.evo_hist],
			genotype:write(Cx#cortex{neuron_ids = U_N_Ids}),
			genotype:write(A#agent{pattern=U_Pattern,evo_hist=U_EvoHist})
	end.
%The function add_neuron/1 creats a new neuron, and connects it to a randomly selected element in the NN, and form a randomly selected element in the NN. The function first reads the agent's pattern list, selects a random layer from the pattern, and then creates a new neuron id for that layer. Then, a new, unconnected neuron, is created with that neuron id. From the cortex's neuron_ids list, two random neuron ids are chosen: From_ElementId, and To_ElementId, (they can be the same ids). The function then establishes a connection from the neuron to To_ElemenId, and to the neuron from From_ElementId. Finally, the cortex's neuron_ids list is updated by appending to it the id of the newly created neuron, the agent's evo_hist is updated, and the updated cortex and agent records are written to database.

outsplice(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Generation = A#agent.generation,
	Pattern = A#agent.pattern,
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
	N = genotype:read({neuron,N_Id}),
	{{LayerIndex,_UId},neuron} = N_Id,
%Choose a random neuron in the output_ids for splicing, only forward facing.
	OId_Pool = [{{OL,O_UId},OT} || {{OL,O_UId},OT} <- N#neuron.output_ids],% OL > LayerIndex],
	io:format("OIds:~p~nOIdPool:~p~n",[N#neuron.output_ids,OId_Pool]),
	case OId_Pool of
		[] ->
			exit("********ERROR:neurolink_OutputSplice:: NeuroLink_OutputSplice O_IdPool == []");
		_->		
			O_Id = lists:nth(random:uniform(length(OId_Pool)),OId_Pool),
			{{OutputLayerIndex,_Output_UId},_OutputType} = O_Id,
%Create a new Layer, or select an existing one between N_Id and the O_Id, and create the new unlinked neuron.
		NewLI = case OutputLayerIndex >= LayerIndex of
			true ->
				get_NewLI(LayerIndex,OutputLayerIndex,next,Pattern);
			false ->
				get_NewLI(LayerIndex,OutputLayerIndex,prev,Pattern)
		end,
		NewN_Id={{NewLI,genotype:generate_UniqueId()},neuron},
		SpecCon = A#agent.constraint,
		genotype:construct_Neuron(Cx_Id,Generation,SpecCon,NewN_Id,[],[]),
%Update pattern.
		U_Pattern=case lists:keymember(NewLI,1,Pattern) of
			true->
				{NewLI,InLayerIds}=lists:keyfind(NewLI, 1, Pattern),
				lists:keyreplace(NewLI, 1, Pattern, {NewLI,[NewN_Id|InLayerIds]});
			false ->
				lists:sort([{NewLI,[NewN_Id]}|Pattern])
		end,
%Disconnect the N_Id from the O_Id, and reconnect through NewN_Id
		cutlink_FromElementToElement(A#agent.generation,N_Id,O_Id),
		link_FromElementToElement(A#agent.generation,N_Id,NewN_Id),
		link_FromElementToElement(A#agent.generation,NewN_Id,O_Id),
%Updated agent
		EvoHist = A#agent.evo_hist,
		U_EvoHist = [{outsplice,N_Id,NewN_Id,O_Id}|EvoHist],
		U_Cx = Cx#cortex{neuron_ids = [NewN_Id|Cx#cortex.neuron_ids]},
		genotype:write(U_Cx),
		genotype:write(A#agent{pattern=U_Pattern,evo_hist=U_EvoHist})
	end.
%The function outsplice/1 chooses a random neuron id from the cortex's neuron_ids list, disconnects it from a randomly chosen id in its output_ids list, and then reconnects it to the same element through a newly created neuron. The function first chooses a random neuron N with the neuron id N_Id from the cortex's neuron_ids list. Then the neuron N's output_ids list is extracted, and a new id list O_IdPool is created from the ids in the output_ids list that are located in the layer after the N_Id's layer (the ids of elements to whom the N_Id forms a feed forward connection). From that sublist of N's output_ids list, a random O_Id is chosen, and if the sublist is empty, then the function exits with an error. Then N_Id is disconnected from the O_Id. The function then creates or extracts a new layer index, NewLI, located between N_Id and O_Id. If there exists a layer between N_Id and O_Id, NewLI is simply that layer, if on the other hand O_Id's layer comes imedietly after N_Id's then a new layer is created between O_Id and N_Id, whose layer index is in the middle of the two elements. A new unconnected neuron is then created in that layer, with a neuron id NewN_Id, and connected to the O_Id, and from the N_Id, thus establishing a path from N_Id to O_Id through the NewN_Id. The cortex's neuron_ids is updated with the NewN_Id, and the agent's evo_hist list is updated with the new mutation operator tuple {outsplice,N_Id,Newn_Id,O_Id}. Finally, the updated cortex and agent are written to database.

get_NewLI(LI,LI,_Direction,_Pattern)->
	LI;
get_NewLI(FromLI,ToLI,Direction,Pattern)->
	NewLI = case Direction of
		next ->
			get_NextLI(Pattern,FromLI,ToLI);
		prev ->
			get_PrevLI(lists:reverse(Pattern),FromLI,ToLI)
	end,
	NewLI.
%get_NewLI/4 calculates or creats a new layer index located between FromLI and ToLI. The function calls get_NextLI/3 or get_PrevLI/3, depending on whether the direction of the connection is forward, from sensors towards actuators (Direction = next) or from actuators towards sensors (Direction = prev), which is the case when executing an insplice/1 function, which calculates or creates a new layer between the N_Id and one of the ids in its input_idps list. If the FromLI == ToLI, the function exits with an error.

	get_NextLI([{FromLI,_LastLayerNIds}],FromLI,ToLI)->
		(FromLI+ToLI)/2;
	get_NextLI([{LI,_LayerNIds}|Pattern],FromLI,ToLI)->
		case LI == FromLI of
			true ->
				[{NextLI,_NextLayerNIds}|_] = Pattern,
				case NextLI == ToLI of
					true ->
						(FromLI + ToLI)/2;
					false ->
						NextLI
				end;
			false ->
				get_NextLI(Pattern,FromLI,ToLI)
		end.
%get_NextLI checks whether the ToLI comes directly after FromLI, or whether there is another layer between them. If there is another layer between them, then that layer is returned, and the splice neuron should then be put into that layer. If there is no layer between FromLI and ToLI, then a new layer is created in the middle, the new layer index has the value of (FromLI+ToLI)/2.

	get_PrevLI([{FromLI,_FirstLayerNIds}],FromLI,ToLI)->
		(FromLI+ToLI)/2;
	get_PrevLI([{LI,_LayerNIds}|Pattern],FromLI,ToLI)->
		case LI == FromLI of
			true ->
				[{PrevLI,_PrevLayerNIds}|_] = Pattern,
				case PrevLI == ToLI of
					true ->
						(FromLI + ToLI)/2;
					false ->
						PrevLI
				end;
			false ->
				get_PrevLI(Pattern,FromLI,ToLI)
		end.
%get_PrevLI checks whether the The ToLI comes directly before FromLI, or whetehr there is another layer in between them. If there is another layer, then the function returns that layer, if no such layer is found, the the function creates a new layer indeex, (FromLI+ToLI)/2.

add_sensorlink(Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx_Id = A#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	S_Ids = case A#agent.encoding_type of
		neural ->
			Cx#cortex.sensor_ids;
		substrate ->
			Substrate_Id=A#agent.substrate_id,
			Substrate=genotype:read({substrate,Substrate_Id}),
			Substrate#substrate.cpp_ids
	end,
	S_Id = lists:nth(random:uniform(length(S_Ids)),S_Ids),
	S = genotype:read({sensor,S_Id}),
	case N_Ids -- S#sensor.fanout_ids of
		[] ->
			exit("********ERROR:add_sensorlink:: Sensor already connected to all N_Ids");
		Available_Ids ->
			N_Id = lists:nth(random:uniform(length(Available_Ids)),Available_Ids),
			link_FromElementToElement(A#agent.generation,S_Id,N_Id),
			EvoHist = A#agent.evo_hist,
			U_EvoHist = [{add_sensorlink,S_Id,N_Id}|EvoHist],
			genotype:write(A#agent{evo_hist=U_EvoHist})
	end.
%The function add_sensorlink/1, randomly selects a S_Id from the cortex's sensor_ids list, and then establishes from that sensor a connection to a still unlinked to this sensor, randomly selected neuron from the cortex's neuron_ids list. The function first selects a random sensor id S_Id from the cortex's sensor_ids list. Then a list of N_Ids to which S_Id is not yet connected is calculated by subtracting from the N_Ids the S_Ids fanout_ids list. If the resulting list is empty, then the function exits with an error since there is no other neurons to which the sensor can establish a new connection to. If the list is not empty, then a random neuron id, N_Id, is selected from this list, and a connection is established from S_Id to N_Id. Finally, the agent's evo_hist is then updated, and it is written to database.

add_actuatorlink(Agent_Id)->%TODO: There should be a preference towards non fully connected actuators.
	Agent = genotype:read({agent,Agent_Id}),
	Cx_Id = Agent#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	N_Ids = Cx#cortex.neuron_ids,
	A_Ids = case Agent#agent.encoding_type of
		neural ->
			Cx#cortex.actuator_ids;
		substrate ->
			Substrate_Id=Agent#agent.substrate_id,
			Substrate=genotype:read({substrate,Substrate_Id}),
			Substrate#substrate.cep_ids
	end,
	A_Id = lists:nth(random:uniform(length(A_Ids)),A_Ids),
	A = genotype:read({actuator,A_Id}),
	case N_Ids -- A#actuator.fanin_ids of
		[] ->
			exit("********ERROR:add_actuatorlink:: Actuator already connected from all N_Ids");
		Available_Ids ->
			N_Id = lists:nth(random:uniform(length(Available_Ids)),Available_Ids),
			link_FromElementToElement(Agent#agent.generation,N_Id,A_Id),
			EvoHist = Agent#agent.evo_hist,
			U_EvoHist = [{add_actuatorlink,N_Id,A_Id}|EvoHist],
			genotype:write(Agent#agent{evo_hist=U_EvoHist})
	end.
%The add_actuatorlink/1 selects a random actuator id A_Id from the cortex's actuator_ids list, and then connects A_Id to randomly selected neuron to which the A_Id is not yet connected to. The function first selects a random actuator id A_Id from the cortex's actuator_ids list. Then the function creates a list of neuron ids to which it is not yet connected by subtracting its fanin_ids list from the cortex's neuron_ids list. If the resulting id pool is empty, then the function exits with error. If the resulting id pool is not empty, a neuron id N_Id is randomly chosen from this id list, and the actuator is connected to this randomly chosen neuron. Finally, the agent's evo_hist is updated, and the updated agent is written to database.

add_sensor(Agent_Id)->%TODO: There should be a preference towards adding sensors not yet used.
	Agent = genotype:read({agent,Agent_Id}),
	Cx_Id = Agent#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	S_Ids = Cx#cortex.sensor_ids,
	SpeCon = Agent#agent.constraint,
	Morphology = SpeCon#constraint.morphology,
	case morphology:get_Sensors(Morphology)--[(genotype:read({sensor,S_Id}))#sensor{id=undefined,cx_id=undefined,fanout_ids=[],generation=undefined} || S_Id<-S_Ids] of
		[] ->
			exit("********ERROR:add_sensor(Agent_Id):: NN system is already using all available sensors");
		Available_Sensors ->
			NewS_Id = {{-1,genotype:generate_UniqueId()},sensor},
			NewSensor=(lists:nth(random:uniform(length(Available_Sensors)),Available_Sensors))#sensor{id=NewS_Id,cx_id=Cx_Id},
			EvoHist = Agent#agent.evo_hist,
			case Agent#agent.encoding_type of
				neural->
					genotype:write(NewSensor),
					N_Ids = Cx#cortex.neuron_ids,
					N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
					link_FromElementToElement(Agent#agent.generation,NewS_Id,N_Id),
					U_EvoHist = [{add_sensor,NewS_Id,N_Id}|EvoHist];
				substrate ->
					Substrate_Id = Agent#agent.substrate_id,
					genotype:write(NewSensor#sensor{fanout_ids=[Substrate_Id]}),
					U_EvoHist = [{add_sensor,NewS_Id,Substrate_Id}|EvoHist]
			end,
			U_Cx = Cx#cortex{sensor_ids=[NewS_Id|S_Ids]},
			genotype:write(U_Cx),
			genotype:write(Agent#agent{evo_hist=U_EvoHist})
	end.
%The add_sensor/1 function adds and connects a new sensor to the neural network, a sensor type to which the NN is not yet connected from. After retrieving the morphology name from the constraints record retrived from the agent, the complete set of available sensors is retrevied using the morphology:get_Sensors/1 function. From this complete sensor list we subtract the sensor tuples used by the NN based system, but first we revert those sensor's id and cx_id back to undefined, since that is what the initial state of the sensor tuples are. With the NN's sensors ids and cx_ids reverted back to undefined, they can be subtracted from the compelete set of sensors. If the resulting list is empty, then the function exits with an error. On the other hand if ther esulting list is not empty, then there are still sensors which the NN is not yet using (though it does not mean that using the sensors would make the NN better, these sensors might be simply useless, and hence not previously incorporated during evolution). From this resulting list we then select a random sensor, and create for it a unique sensor id NewS_Id. A random neuron id N_Id is then selected from the cortex's neuron_ids list, and a connection is established from NewS_Id to the N_Id. The cortex's sensor_ids is updated with the new sensor's id, and the agent's evo_hist is updated with the new tuple. The updated cortex and agent records are then written to database.

add_actuator(Agent_Id)->%TODO: There should be a preference towards adding actuators not yet used.
	Agent = genotype:read({agent,Agent_Id}),
	Cx_Id = Agent#agent.cx_id,
	Cx = genotype:read({cortex,Cx_Id}),
	A_Ids = Cx#cortex.actuator_ids,%TODO: Should we fill in all the fanin_ids locations, or just 1? and let evolution fill the rest?
	SpeCon = Agent#agent.constraint,
	Morphology = SpeCon#constraint.morphology,
	case morphology:get_Actuators(Morphology)--[(genotype:read({actuator,A_Id}))#actuator{cx_id=undefined,id=undefined,fanin_ids=[],generation=undefined} || A_Id<-A_Ids] of
		[] ->
			exit("********ERROR:add_actuator(Agent_Id):: NN system is already using all available actuators");
		Available_Actuators ->
			%io:format("New:~p~nUsd:~p~n Remaining:~p~n",[morphology:get_Actuators(Morphology),[(genotype:read({actuator,A_Id}))#actuator{cx_id=undefined,id=undefined,fanin_ids=[],generation=undefined} || A_Id<-A_Ids],Available_Actuators]),
			NewA_Id = {{1,genotype:generate_UniqueId()},actuator},
			NewActuator=(lists:nth(random:uniform(length(Available_Actuators)),Available_Actuators))#actuator{id=NewA_Id,cx_id=Cx_Id},
			EvoHist = Agent#agent.evo_hist,
			case Agent#agent.encoding_type of
				neural ->
					genotype:write(NewActuator),
					N_Ids = Cx#cortex.neuron_ids,
					N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
					link_FromElementToElement(Agent#agent.generation,N_Id,NewA_Id),
					U_EvoHist = [{add_actuator,N_Id,NewA_Id}|EvoHist];
				substrate ->
					Substrate_Id = Agent#agent.substrate_id,
					genotype:write(NewActuator#actuator{fanin_ids=[Substrate_Id]}),
					U_EvoHist = [{add_actuator,Substrate_Id,NewA_Id}|EvoHist]
			end,
			U_Cx = Cx#cortex{actuator_ids=[NewA_Id|A_Ids]},
			genotype:write(U_Cx),
			genotype:write(Agent#agent{evo_hist=U_EvoHist})
	end.
%The add_actuator/1 function adds and connects a new actuator to the neural network, an actuator type to which the NN is noet yet connected to. After the morphology name from the constraints record, a complete actuator list available to the NN from which to draw its actuators from during evolution is created. From that list the actuator list that the NN is already connected to is subtracted, after the ids and cx_ids of those actuators is set to undefined. The resulting list is the list of actuators to which the NN is not yet connected to. A random actuator is chosen from that list, and a random neuron id N_Id from cortex's neuron_ids is chosen and connected to the new actuator. The cortex's actuator_ids list is then updated with the id of the newly created actuator, the agent's evo_hist is updated with the new tuple, and then both the updated cortex and the agent are written to database.

add_cpp(Agent_Id)->%TODO: There should be a preference towards adding substrate_cpps not yet used.
	Agent = genotype:read({agent,Agent_Id}),
	case Agent#agent.encoding_type of
		neural->
			exit("********ERROR:add_cpp(Agent_Id):: NN is neural encoded, can not apply mutation operator.");
		substrate->
			Cx_Id = Agent#agent.cx_id,
			Cx = genotype:read({cortex,Cx_Id}),
			Substrate_Id = Agent#agent.substrate_id,
			Substrate=genotype:read({substrate,Substrate_Id}),
			Dimensions = length(Substrate#substrate.densities),
			Plasticity = Substrate#substrate.plasticity,
			CPP_Ids = Substrate#substrate.cpp_ids,
			case morphology:get_SubstrateCPPs(Dimensions,Plasticity)--[(genotype:read({sensor,CPP_Id}))#sensor{id=undefined,cx_id=undefined,fanout_ids=[],generation=undefined} || CPP_Id<-CPP_Ids] of
				[] ->
					exit("********ERROR:add_cpp(Agent_Id):: NN system is already using all available substrate_cpps");
				Available_CPPs ->
					NewCPP_Id = {{-1,genotype:generate_UniqueId()},sensor},
					NewCPP=(lists:nth(random:uniform(length(Available_CPPs)),Available_CPPs))#sensor{id=NewCPP_Id,cx_id=Cx_Id},
					EvoHist = Agent#agent.evo_hist,
					genotype:write(NewCPP),
					N_Ids = Cx#cortex.neuron_ids,
					N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
					link_FromElementToElement(Agent#agent.generation,NewCPP_Id,N_Id),
					U_EvoHist = [{add_cpp,NewCPP_Id,N_Id}|EvoHist],
					U_Substrate = Substrate#substrate{cpp_ids=[NewCPP_Id|CPP_Ids]},
					genotype:write(U_Substrate),
					genotype:write(Agent#agent{evo_hist=U_EvoHist})
			end
	end.
	
add_cep(Agent_Id)->
	Agent = genotype:read({agent,Agent_Id}),
	case Agent#agent.encoding_type of
		neural->
			exit("********ERROR:add_cep(Agent_Id):: NN is neural encoded, can not apply mutation operator.");
		substrate->
			Cx_Id = Agent#agent.cx_id,
			Cx = genotype:read({cortex,Cx_Id}),
			Substrate_Id = Agent#agent.substrate_id,
			Substrate=genotype:read({substrate,Substrate_Id}),
			Dimensions = length(Substrate#substrate.densities),
			Plasticity = Substrate#substrate.plasticity,
			CEP_Ids = Substrate#substrate.cep_ids,
			case morphology:get_SubstrateCEPs(Dimensions,Plasticity)--[(genotype:read({actuator,CEP_Id}))#actuator{id=undefined,cx_id=undefined,fanin_ids=[],generation=undefined} || CEP_Id<-CEP_Ids] of
				[] ->
					exit("********ERROR:add_cep(Agent_Id):: NN system is already using all available substrate_cpps");
				Available_CEPs ->
					NewCEP_Id = {{-1,genotype:generate_UniqueId()},actuator},
					NewCEP=(lists:nth(random:uniform(length(Available_CEPs)),Available_CEPs))#actuator{id=NewCEP_Id,cx_id=Cx_Id},
					EvoHist = Agent#agent.evo_hist,
					genotype:write(NewCEP),
					N_Ids = Cx#cortex.neuron_ids,
					N_Id = lists:nth(random:uniform(length(N_Ids)),N_Ids),
					link_FromElementToElement(Agent#agent.generation,N_Id,NewCEP_Id),
					U_EvoHist = [{add_cep,NewCEP_Id,N_Id}|EvoHist],
					U_Substrate = Substrate#substrate{cep_ids=[NewCEP_Id|CEP_Ids]},
					genotype:write(U_Substrate),
					genotype:write(Agent#agent{evo_hist=U_EvoHist})
			end
	end.
