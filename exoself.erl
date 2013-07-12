%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This source code and work is provided and developed by Gene I. Sher & DXNN Research Group WWW.DXNNResearch.COM
%
%The original release of this source code and the DXNN MK2 system was introduced and explained in my book: Handbook of Neuroevolution Through Erlang. Springer 2012, print ISBN: 978-1-4614-4462-6 ebook ISBN: 978-1-4614-4463-6. 
%
%Copyright (C) 2009 by Gene Sher, DXNN Research Group CorticalComputer@gmail.com
%
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.
%%%%%%%%%%%%%%%%%%%% Deus Ex Neural Network :: DXNN %%%%%%%%%%%%%%%%%%%%

-module(exoself).
-compile(export_all).
-include("records.hrl").
-record(state,{
	agent_id,
	morphology,
	generation,
	pm_pid,
	idsNpids,
	cx_pid,
	specie_id,
	spids=[],
	npids=[],
	nids=[],
	apids=[],
	private_scape_pids=[],
	public_scape_pids=[],
	highest_fitness,
	eval_acc=0,
	cycle_acc=0,
	time_acc=0,
	max_attempts=15,
	attempt=1,
	tuning_duration_f,
	tuning_selection_f,
	annealing_parameter,
	perturbation_range,
	substrate_pid,
	cpp_pids=[],
	cep_pids=[],
	opmode
}).

start(Agent_Id)->
	case whereis(monitor) of
		undefined ->
			io:format("start(Agent_Id):: 'monitor' is not registered~n");
		PId ->
			start(Agent_Id,PId,gt)
	end.

start(Agent_Id,PM_PId)->
	start(Agent_Id,PM_PId,gt).

start(Agent_Id,PM_PId,OpMode)->
	spawn(exoself,prep,[Agent_Id,PM_PId,OpMode]).
%The start/3 function spawns a new Agent_Id exoself process, belonging to the population_monitor process with the pid PM_PId, and using the OpMode with which it was spawned.

prep(Agent_Id,PM_PId,OpMode)->
	random:seed(now()),
	IdsNPIds = ets:new(idsNpids,[set,private]), 
	A = genotype:dirty_read({agent,Agent_Id}),
	HeredityType = A#agent.heredity_type,
	Cx = genotype:dirty_read({cortex,A#agent.cx_id}),
	SIds = Cx#cortex.sensor_ids,
	AIds = Cx#cortex.actuator_ids,
	NIds = Cx#cortex.neuron_ids,
	spawn_CerebralUnits(IdsNPIds,cortex,[Cx#cortex.id]),
	spawn_CerebralUnits(IdsNPIds,sensor,SIds),
	spawn_CerebralUnits(IdsNPIds,actuator,AIds),
	spawn_CerebralUnits(IdsNPIds,neuron,NIds),
	{Private_ScapePIds,Public_ScapePIds} = spawn_Scapes(IdsNPIds,SIds,AIds,Agent_Id),
	case A#agent.encoding_type of
		substrate ->
			Substrate_Id=A#agent.substrate_id,
			Substrate = genotype:dirty_read({substrate,Substrate_Id}),
			CPP_Ids = Substrate#substrate.cpp_ids,
			CEP_Ids = Substrate#substrate.cep_ids,
			spawn_CerebralUnits(IdsNPIds,substrate_cpp,CPP_Ids),
			spawn_CerebralUnits(IdsNPIds,substrate_cep,CEP_Ids),
			spawn_CerebralUnits(IdsNPIds,substrate,[Substrate_Id]),
			
			Substrate_PId=ets:lookup_element(IdsNPIds,Substrate_Id,2),
			link_SubstrateCPPs(CPP_Ids,IdsNPIds,Substrate_PId),
			link_SubstrateCEPs(CEP_Ids,IdsNPIds,Substrate_PId),
			SDensities = Substrate#substrate.densities,
			SPlasticity = Substrate#substrate.plasticity,
			SLinkform = Substrate#substrate.linkform,
			Sensors=[genotype:dirty_read({sensor,SId})||SId <- SIds],
			Actuators=[genotype:dirty_read({actuator,AId})||AId <- AIds],		
			CPP_PIds=[ets:lookup_element(IdsNPIds,Id,2)||Id<-CPP_Ids],
			CEP_PIds=[ets:lookup_element(IdsNPIds,Id,2)||Id<-CEP_Ids],
			Substrate_PId ! {self(),init,{Sensors,Actuators,[ets:lookup_element(IdsNPIds,Id,2)||Id<-SIds],[ets:lookup_element(IdsNPIds,Id,2)||Id<-AIds],CPP_PIds,CEP_PIds,SDensities,SPlasticity,SLinkform}};
		_ ->
			CPP_PIds=[],
			CEP_PIds=[],
			Substrate_PId = undefined
	end,
	link_Sensors(SIds,IdsNPIds,OpMode),
	link_Actuators(AIds,IdsNPIds,OpMode),
	link_Neurons(NIds,IdsNPIds,HeredityType),
	{SPIds,NPIds,APIds}=link_Cortex(Cx,IdsNPIds,OpMode),
	Cx_PId = ets:lookup_element(IdsNPIds,Cx#cortex.id,2),
	{TuningDurationFunction,Parameter} = A#agent.tuning_duration_f,
	Morphology = (A#agent.constraint)#constraint.morphology,
	S = #state{
		agent_id=Agent_Id,
		morphology=Morphology,
		generation=A#agent.generation,
		pm_pid=PM_PId,
		idsNpids=IdsNPIds,
		cx_pid=Cx_PId,
		specie_id=A#agent.specie_id,
		spids=SPIds,
		npids=NPIds,
		nids=NIds,
		apids=APIds,
		substrate_pid=Substrate_PId,
		cpp_pids = CPP_PIds,
		cep_pids = CEP_PIds,
		private_scape_pids=Private_ScapePIds,
		public_scape_pids=Public_ScapePIds,
		max_attempts= tuning_duration:TuningDurationFunction(Parameter,NIds,A#agent.generation),
		tuning_selection_f=A#agent.tuning_selection_f,
		annealing_parameter=A#agent.annealing_parameter,
		tuning_duration_f=A#agent.tuning_duration_f,
		perturbation_range=A#agent.perturbation_range,
%		max_attempts = 15 + round(math:sqrt(length(NIds) + length(SIds) + length(AIds))),
		opmode=OpMode
	},
	%io:format("S:~p~n",[{S,OpMode}]),
	loop(S,OpMode).
%The prep/2 function prepares and sets up the exoself's state before dropping into the main loop. The function first reads the agent and cortex records belonging to the Agent_Id NN based system. The function then reads the sensor, actuator, and neuron ids, then spawns the private scapes using the spawn_Scapes/3 function, spawns the cortex, sensor, actuator, and neuron processes, and then finally links up all these processes together using the link_.../2 processes. Once the phenotype has been generated from the genotype, the exoself drops into its main loop.
-define(MIN_PIMPROVEMENT,0).
loop(S,gt)->
	receive
		{Cx_PId,evaluation_completed,Fitness,Cycles,Time,GoalReachedFlag}->
			%io:format("E Msg:~p~n E S:~p~n",[{Cx_PId,evaluation_completed,Fitness,Cycles,Time,GoalReachedFlag},S]),
			IdsNPIds = S#state.idsNpids,
			HighestFitness = case S#state.highest_fitness of
				undefined ->
					[Val-1||Val <- Fitness];
				HiFi->
					HiFi
			end,	
			{U_HighestFitness,U_Attempt}=case vec1_dominates_vec2(Fitness,HighestFitness,?MIN_PIMPROVEMENT) of %case Fitness > S#state.highest_fitness of %vec1_dominates_vec2(Avg_Fitness,HighestFitness,?MIN_PIMPROVEMENT)
				true ->
					[NPId ! {self(),weight_backup} || NPId <- S#state.npids],
					A=genotype:dirty_read({agent,S#state.agent_id}),
					[Main_Fitness|_] = Fitness,
					genotype:write(A#agent{
						fitness=Fitness,
						main_fitness = Main_Fitness
					}),
					backup_genotype(S#state.idsNpids,S#state.npids),
					{Fitness,0};
				false ->
					Perturbed_NIdPs=get(perturbed),
					[ets:lookup_element(IdsNPIds,NId,2) ! {self(),weight_restore} || {NId,_Spread} <- Perturbed_NIdPs],
					{HighestFitness,S#state.attempt+1}
			end,
			[PId ! {self(), reset_prep} || PId <- S#state.npids],
			gather_acks(length(S#state.npids)),
			[PId ! {self(), reset} || PId <- S#state.npids],
			case S#state.substrate_pid of
				undefined ->
					ok;
				Substrate_PId ->
					Substrate_PId ! {self(),reset_substrate},
					receive
						{Substrate_PId,ready}->
							ok
					end
			end,
			%io:format("HighestFitness:~p U_Attempt:~p~n",[U_HighestFitness,U_Attempt]),
			U_CycleAcc = S#state.cycle_acc+Cycles,
			U_TimeAcc = S#state.time_acc+Time,
			U_EvalAcc = S#state.eval_acc+1,
			gen_server:cast(S#state.pm_pid,{self(),evaluations,S#state.specie_id,1,Cycles,Time}),
			case (U_Attempt >= S#state.max_attempts) or (GoalReachedFlag == true) of
				true ->	%End training
%					A=genotype:dirty_read({agent,S#state.agent_id}),
%					[Main_Fitness|_] = U_HighestFitness,
%					genotype:write(A#agent{
%						fitness=U_HighestFitness,
%						main_fitness = Main_Fitness
%					}),
%					backup_genotype(S#state.idsNpids,S#state.npids),
					terminate_phenotype(S#state.cx_pid,S#state.spids,S#state.npids,S#state.apids,S#state.private_scape_pids,S#state.cpp_pids,S#state.cep_pids,S#state.substrate_pid),
					case GoalReachedFlag of
						true ->
							io:format("Goal reached. Agent:~p terminating. Genotype has been backed up.~n Fitness:~p~n TotEvaluations:~p~n TotCycles:~p~n TimeAcc:~p~n",[self(),U_HighestFitness,U_EvalAcc,U_CycleAcc,U_TimeAcc]),
							gen_server:cast(S#state.pm_pid,{S#state.agent_id,goal_reached});
						_ ->
							io:format("Tuning complete. Agent:~p terminating. Genotype has been backed up.~n Fitness:~p~n TotEvaluations:~p~n TotCycles:~p~n TimeAcc:~p~n",[self(),U_HighestFitness,U_EvalAcc,U_CycleAcc,U_TimeAcc]),
							ok
					end,
					gen_server:cast(S#state.pm_pid,{S#state.agent_id,terminated,U_HighestFitness});
				false -> %Continue training
					%io:format("exoself state:~p~n",[S]),
					reenter_PublicScape(S#state.public_scape_pids,[genotype:dirty_read({sensor,ets:lookup_element(IdsNPIds,Id,2)})||Id<-S#state.spids],[genotype:dirty_read({actuator,ets:lookup_element(IdsNPIds,Id,2)})||Id<-S#state.apids],S#state.specie_id,S#state.morphology,length(S#state.nids)),
					TuningSelectionFunction=S#state.tuning_selection_f,
					PerturbationRange = S#state.perturbation_range,
					AnnealingParameter = S#state.annealing_parameter,
					ChosenNIdPs=tuning_selection:TuningSelectionFunction(S#state.nids,S#state.generation,PerturbationRange,AnnealingParameter),
					[ets:lookup_element(IdsNPIds,NId,2) ! {self(),weight_perturb,Spread} || {NId,Spread} <- ChosenNIdPs],
					%io:format("ChosenNPIds:~p~n",[ChosenNIdPs]),
					put(perturbed,ChosenNIdPs),
					Cx_PId ! {self(),reactivate},
					U_S =S#state{
						cycle_acc=U_CycleAcc,
						time_acc=U_TimeAcc,
						eval_acc=U_EvalAcc,
						attempt=U_Attempt,
						highest_fitness=U_HighestFitness
					},
					exoself:loop(U_S,gt)
			end
		%after 10000 ->
		%	io:format("exoself:~p stuck.~n",[S#state.agent_id])
	end;
loop(S,validation)->
	io:format("In the Validation loop~n"),
	receive
		{Cx_PId,evaluation_completed,Fitness,Cycles,Time,GoalReachedFlag}->
			terminate_phenotype(S#state.cx_pid,S#state.spids,S#state.npids,S#state.apids,S#state.private_scape_pids,S#state.cpp_pids,S#state.cep_pids,S#state.substrate_pid),
			io:format("Validation complete, agent:~p terminating. Fitness:~p~n TotCycles:~p~n TimeAcc:~p Goal:~p~n",[self(),Fitness,Cycles,Time,GoalReachedFlag]),
			S#state.pm_pid ! {S#state.agent_id,validation_complete,S#state.specie_id,Fitness,Cycles,Time}
	end;
loop(S,test)->
	io:format("In the Test loop~n"),
	receive
		{Cx_PId,evaluation_completed,Fitness,Cycles,Time,GoalReachedFlag}->
			terminate_phenotype(S#state.cx_pid,S#state.spids,S#state.npids,S#state.apids,S#state.private_scape_pids,S#state.cpp_pids,S#state.cep_pids,S#state.substrate_pid),
			io:format("Test complete, agent:~p terminating. Fitness:~p~n TotCycles:~p~n TimeAcc:~p Goal:~p~n",[self(),Fitness,Cycles,Time,GoalReachedFlag]),
			%ets:insert(testing,{self(),Fitness})%%TODO
			S#state.pm_pid ! {S#state.agent_id,test_complete,S#state.specie_id,Fitness,Cycles,Time}
	end.
%The exoself process' main loop awaits from its cortex proccess the evoluation_completed message. Once the message is received, based on the fitness achieved, exoself decides whether to continue tunning the weights or terminate the system. Exoself tries to improve the fitness by perturbing/tuning the weights of its neurons, after each tuning session, the Neural Network based system performs another evaluation by interacting with the scape until completion (the NN solves a problem, or dies within the scape or...). The order of events is important: When evaluation_completed message is received, the function first checks whether the newly achieved fitness is higher than the highest fitness achieved so far. If it is not, the function sends the neurons a message to restore their weights to previous state, during which it last acehived the highest fitness instead of their current state which yielded the current lower fitness score. If on the other hand the new fitness is higher than the previously highest achieved fitness, then the function tells the neurons to backup their current weights, as these weights represent the NN's best, most fit form yet. Exoself then tells all the neurons to prepare for a reset by sending each neuron the {self(),reset_prep} message. Since the NN can have recursive connections, and the manner in which initial recursive messages are sent, it is important for each neuron to flush their buffers to be reset into an initial fresh state, which is achieved after the neurons receive the reset_prep message. The function then sends the reset message to the neurons, which returns them into their main loop. Finally, the function checks whether exoself has already tried to improve the NN's fitness a maximum S#state.max_attempts number of times. If that is the case, the exoself process backs up the updated NN (the updated, tuned weights) to database using the backup_genotype/2 function, prints to screen that it is terminating, and sends to the population_monitor the acumulated statistics (highest fitness, evaluation count, cycle count...). On the other hand, if the exoself is not yet done tuning the neural weights, it has not yet reached its ending condition, it uses a tuning_selection_function to compose a list of tuples: [{NId,Spread}...] of neuron ids and the perturbation spread values, where the spread is the range from which the perturbation is randomly chosen. The spread itself is based on the age of the slected neuron, using the annealing_factor value, which when set to 1 implies that there is no annealing, and when set to a value less than 1, decreases the Spread. Once this list of elements is composed, the exoself sends each of the neurons a message to perturb their synaptic weights using the Spread value. The exoself then reactivates the cortex, and drops back into its main loop.

		vec1_dominates_vec2(A,B,MIP)->
			%io:format("A:~p~nB:~p~nMIP:~p~n",[A,B,MIP]),
			VecDif=vec1_dominates_vec2(A,B,MIP,[]),
			%io:format("VecDif:~p~n",[VecDif]),
			TotElems = length(VecDif),
			DifElems=length([Val || Val<-VecDif, Val > 0]),
			case DifElems of
				TotElems->%Complete Superiority
					true;
				0 ->%Complete Inferiority
					false;
				_ ->%Variation, pareto front TODO
					false
			end.
		vec1_dominates_vec2([Val1|Vec1],[Val2|Vec2],MIP,Acc)->
			vec1_dominates_vec2(Vec1,Vec2,MIP,[Val1-(Val2+Val2*MIP)|Acc]);
		vec1_dominates_vec2([],[],_MIP,Acc)->
			Acc.
		
		vector_avg(Vec,Length)->vector_avg(Vec,[],0,[],Length).
		vector_avg([Vec|Vectors],RemAcc,ValAcc,VecAcc,Length)->
			case Vec of
				[]->
					lists:reverse(VecAcc);
				[Val|Rem]->
					vector_avg(Vectors,[Rem|RemAcc],Val+ValAcc,VecAcc,Length)
			end;
		vector_avg([],RemAcc,ValAcc,VecAcc,Length)->
			vector_avg(RemAcc,[],0,[ValAcc/Length|VecAcc],Length).
			
		vector_basic_stats(VectorList)->
			T_VectorList = transpose(VectorList,[],[],[]),%Does not retain order
			[VecSample|_TVL] = T_VectorList,
			Length = length(VecSample),
			AvgVector = [lists:sum(V)/Length || V<-T_VectorList],
			io:format("AvgVector:~p Length:~p T_VectorList:~p~n",[AvgVector,Length,T_VectorList]),
			StdVector = std_vector(T_VectorList,AvgVector,[]),%[[functions:std(List,Avg,[])|| List<-T_VectorList] || Avg<- AvgVector],
			MaxVector = lists:max(VectorList),
			MinVector = lists:min(VectorList),
			{MaxVector,MinVector,AvgVector,StdVector}.
		
			std_vector([List|T_VectorList],[Avg|AvgVector],Acc)->
				std_vector(T_VectorList,AvgVector,[functions:std(List,Avg,[])|Acc]);
			std_vector([],[],Acc)->
				lists:reverse(Acc).
				
		transpose(VectorList)->transpose(VectorList,[],[],[]).
		transpose([V|VectorList],RemAcc,ValAcc,VecAcc)->
			case V of
				[] ->
					lists:reverse(VecAcc);
				[Val|Rem] ->
					transpose(VectorList,[Rem|RemAcc],[Val|ValAcc],VecAcc);
				UNKOWN_PATTERN ->
					exit("ERROR:~p~n",[UNKOWN_PATTERN])
			end;
		transpose([],RemAcc,ValAcc,VecAcc)->
			transpose(RemAcc,[],[],[ValAcc|VecAcc]).
			
	spawn_CerebralUnits(IdsNPIds,CerebralUnitType,[Id|Ids])-> 
		PId = CerebralUnitType:gen(self(),node()),
		ets:insert(IdsNPIds,{Id,PId}), 
		ets:insert(IdsNPIds,{PId,Id}), 
		spawn_CerebralUnits(IdsNPIds,CerebralUnitType,Ids); 
	spawn_CerebralUnits(IdsNPIds,_CerebralUnitType,[])-> 
		ets:insert(IdsNPIds,{bias,bias}).
%We spawn the process for each element based on its type: CerebralUnitType, and the gen function that belongs to the CerebralUnitType module. We then enter the {Id,PId} tuple into our ETS table for later use.

	spawn_Scapes(IdsNPIds,Sensor_Ids,Actuator_Ids,Agent_Id)->
		Sensor_Scapes = [(genotype:dirty_read({sensor,Id}))#sensor.scape || Id<-Sensor_Ids], 
		Actuator_Scapes = [(genotype:dirty_read({actuator,Id}))#actuator.scape || Id<-Actuator_Ids], 
		Unique_Scapes = Sensor_Scapes++(Actuator_Scapes--Sensor_Scapes), 
		Private_SN_Tuples=[{scape:gen(self(),node()),ScapeName} || {private,ScapeName}<-Unique_Scapes],
		[ets:insert(IdsNPIds,{ScapeName,PId}) || {PId,ScapeName} <- Private_SN_Tuples], 
		[ets:insert(IdsNPIds,{PId,ScapeName}) || {PId,ScapeName} <-Private_SN_Tuples],
		[PId ! {self(),ScapeName} || {PId,ScapeName} <- Private_SN_Tuples],
		PublicScapePIds=enter_PublicScape(IdsNPIds,Sensor_Ids,Actuator_Ids,Agent_Id),
		PrivateScapePIds=[PId || {PId,_ScapeName} <-Private_SN_Tuples],
		io:format("PublicScapes:~p PrivateScapes:~p~n",[PublicScapePIds,PrivateScapePIds]),
		{PrivateScapePIds,PublicScapePIds}.
%The spawn_Scapes/3 function first extracts all the scapes that the sensors and actuators interface with, it then creates a filtered scape list which only holds unique scape records, after which it further only selects those scapes that are private, and spawns them.

		enter_PublicScape(IdsNPIds,Sensor_Ids,Actuator_Ids,Agent_Id)->
			A = genotype:dirty_read({agent,Agent_Id}),
			Sensors = [genotype:dirty_read({sensor,Id}) || Id<-Sensor_Ids],
			Actuators = [genotype:dirty_read({actuator,Id}) || Id<-Actuator_Ids],
			TotNeurons = length((genotype:dirty_read({cortex,A#agent.cx_id}))#cortex.neuron_ids),
			Morphology = (A#agent.constraint)#constraint.morphology,
			Sensor_Scapes = [Sensor#sensor.scape || Sensor<-Sensors], 
			Actuator_Scapes = [Actuator#actuator.scape || Actuator<-Actuators], 
			Unique_Scapes = Sensor_Scapes++(Actuator_Scapes--Sensor_Scapes), 
			Public_SN_Tuples=[{gen_server:call(polis,{get_scape,ScapeName}),ScapeName} || {public,ScapeName}<-Unique_Scapes],
			[ets:insert(IdsNPIds,{ScapeName,PId}) || {PId,ScapeName} <- Public_SN_Tuples], 
			[ets:insert(IdsNPIds,{PId,ScapeName}) || {PId,ScapeName} <-Public_SN_Tuples],
			[gen_server:call(PId,{enter,Morphology,A#agent.specie_id,Sensors,Actuators,TotNeurons,self()}) || {PId,ScapeName} <- Public_SN_Tuples],
			[PId || {PId,_ScapeName} <-Public_SN_Tuples].
			
		reenter_PublicScape([PS_PId|PS_PIds],Sensors,Actuators,Specie_Id,Morphology,TotNeurons)->
			gen_server:call(PS_PId,{enter,Morphology,Specie_Id,Sensors,Actuators,TotNeurons,self()}),
			reenter_PublicScape(PS_PIds,Sensors,Actuators,Specie_Id,Morphology,TotNeurons);
		reenter_PublicScape([],_Sensors,_Actuators,_Specie_Id,_Morphology,_TotNeurons)->
			ok.
			
		leave_PublicScape([PS_PId|PS_PIds])->
			gen_server:call(PS_PId,{leave,self()}),
			leave_PublicScape(PS_PIds);
		leave_PublicScape([])->
			ok.

	link_Sensors([SId|Sensor_Ids],IdsNPIds,OpMode) ->
		S=genotype:dirty_read({sensor,SId}),
		SPId = ets:lookup_element(IdsNPIds,SId,2),
		Cx_PId = ets:lookup_element(IdsNPIds,S#sensor.cx_id,2),
		SName = S#sensor.name,
		Fanout_Ids = S#sensor.fanout_ids,
		Fanout_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Fanout_Ids],
		Scape=case S#sensor.scape of
			{private,ScapeName}->
				ets:lookup_element(IdsNPIds,ScapeName,2);
			{public,ScapeName}->
				ets:lookup_element(IdsNPIds,ScapeName,2);
			_ ->
				undefined
		end,
		SPId ! {self(),{SId,Cx_PId,Scape,SName,S#sensor.vl,S#sensor.parameters,Fanout_PIds,OpMode}},
		link_Sensors(Sensor_Ids,IdsNPIds,OpMode);
	link_Sensors([],_IdsNPIds,_OpMode)->
		ok.
%The link_Sensors/2 function sends to the already spawned and waiting sensors their states, composed of the PId lists and other information which are needed by the sensors to link up and interface with other elements in the distributed phenotype.

	link_Actuators([AId|Actuator_Ids],IdsNPIds,OpMode) ->
		A=genotype:dirty_read({actuator,AId}),
		APId = ets:lookup_element(IdsNPIds,AId,2),
		Cx_PId = ets:lookup_element(IdsNPIds,A#actuator.cx_id,2),
		AName = A#actuator.name,
		Fanin_Ids = A#actuator.fanin_ids,
		Fanin_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Fanin_Ids],
		Scape=case A#actuator.scape of
			{private,ScapeName}->
				ets:lookup_element(IdsNPIds,ScapeName,2);
			{public,ScapeName}->
				ets:lookup_element(IdsNPIds,ScapeName,2);
			_ ->
				undefined
		end,
		APId ! {self(),{AId,Cx_PId,Scape,AName,A#actuator.vl,A#actuator.parameters,Fanin_PIds,OpMode}},
		link_Actuators(Actuator_Ids,IdsNPIds,OpMode);
	link_Actuators([],_IdsNPIds,_OpMode)->
		ok.
%The link_Actuators2 function sends to the already spawned and waiting actuators their states, composed of the PId lists and other information which are needed by the actuators to link up and interface with other elements in the distributed phenotype.

	link_SubstrateCPPs([CPP_Id|CPP_Ids],IdsNPIds,Substrate_PId) ->
		CPP=genotype:dirty_read({sensor,CPP_Id}),
		CPP_PId = ets:lookup_element(IdsNPIds,CPP_Id,2),
		Cx_PId = ets:lookup_element(IdsNPIds,CPP#sensor.cx_id,2),
		CPPName = CPP#sensor.name,
		Fanout_Ids = CPP#sensor.fanout_ids,
		Fanout_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Fanout_Ids],
		CPP_PId ! {self(),{CPP_Id,Cx_PId,Substrate_PId,CPPName,CPP#sensor.vl,CPP#sensor.parameters,Fanout_PIds}},
		link_SubstrateCPPs(CPP_Ids,IdsNPIds,Substrate_PId);
	link_SubstrateCPPs([],_IdsNPIds,_Substrate_PId)->
		ok.
%The link_Sensors/2 function sends to the already spawned and waiting sensors their states, composed of the PId lists and other information which are needed by the sensors to link up and interface with other elements in the distributed phenotype.

	link_SubstrateCEPs([CEP_Id|CEP_Ids],IdsNPIds,Substrate_PId) ->
		CEP=genotype:dirty_read({actuator,CEP_Id}),
		CEP_PId = ets:lookup_element(IdsNPIds,CEP_Id,2),
		Cx_PId = ets:lookup_element(IdsNPIds,CEP#actuator.cx_id,2),
		CEPName = CEP#actuator.name,
		Fanin_Ids = CEP#actuator.fanin_ids,
		Fanin_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Fanin_Ids],
		CEP_PId ! {self(),{CEP_Id,Cx_PId,Substrate_PId,CEPName,CEP#actuator.parameters,Fanin_PIds}},
		link_SubstrateCEPs(CEP_Ids,IdsNPIds,Substrate_PId);
	link_SubstrateCEPs([],_IdsNPIds,_Substrate_PId)->
		ok.
%The link_SubstrateCEPs/2 function sends to the already spawned and waiting substrate_ceps their states, composed of the PId lists and other information which are needed by the substrate_ceps to link up and interface with other elements in the distributed phenotype.

	link_Neurons([NId|Neuron_Ids],IdsNPIds,HeredityType) ->
		N=genotype:dirty_read({neuron,NId}),
		NPId = ets:lookup_element(IdsNPIds,NId,2),
		Cx_PId = ets:lookup_element(IdsNPIds,N#neuron.cx_id,2),
		AFName = N#neuron.af,
		PFName = N#neuron.pf,
		AggrFName = N#neuron.aggr_f,
		Input_IdPs = N#neuron.input_idps,
		Input_IdPs_Modulation = N#neuron.input_idps_modulation,
		Output_Ids = N#neuron.output_ids,
		RO_Ids = N#neuron.ro_ids,
		SI_PIdPs = convert_IdPs2PIdPs(IdsNPIds,Input_IdPs,[]),
		MI_PIdPs = convert_IdPs2PIdPs(IdsNPIds,Input_IdPs_Modulation,[]),
		O_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Output_Ids],
		RO_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- RO_Ids],
		NPId ! {self(),{NId,Cx_PId,AFName,PFName,AggrFName,HeredityType,SI_PIdPs,MI_PIdPs,O_PIds,RO_PIds}},
		link_Neurons(Neuron_Ids,IdsNPIds,HeredityType);
	link_Neurons([],_IdsNPIds,HeredityType)->
		ok.
%The link_Neurons/2 function sends to the already spawned and waiting neurons their states, composed of the PId lists and other information needed by the neurons to link up and interface with other elements in the distributed phenotype.

		%convert_IdPs2PIdPs(_IdsNPIds,[{bias,[Bias]}],Acc)->
		%	lists:reverse([Bias|Acc]);
		convert_IdPs2PIdPs(IdsNPIds,[{Id,WeightsP}|Fanin_IdPs],Acc)->
			convert_IdPs2PIdPs(IdsNPIds,Fanin_IdPs,[{ets:lookup_element(IdsNPIds,Id,2),WeightsP}|Acc]);
		convert_IdPs2PIdPs(_IdsNPIds,[],Acc)->
			lists:reverse(Acc).
%The convert_IdPs2PIdPs/3 converts the IdPs tuples into tuples that use PIds instead of Ids, such that the Neuron will know which weights are to be associated with which incoming vector signals. The last element is the bias, which is added to the list in a non tuple form. Afterwards, the list is reversed to take its proper order.

	link_Cortex(Cx,IdsNPIds,OpMode) ->
		Cx_Id = Cx#cortex.id,
		Cx_PId = ets:lookup_element(IdsNPIds,Cx_Id,2),
		SIds = Cx#cortex.sensor_ids,
		AIds = Cx#cortex.actuator_ids,
		NIds = Cx#cortex.neuron_ids,
		SPIds = [ets:lookup_element(IdsNPIds,SId,2) || SId <- SIds],
		NPIds = [ets:lookup_element(IdsNPIds,NId,2) || NId <- NIds],
		APIds = [ets:lookup_element(IdsNPIds,AId,2) || AId <- AIds],
		Cx_PId ! {self(),Cx_Id,SPIds,NPIds,APIds,OpMode},
		{SPIds,NPIds,APIds}.
%The link_Cortex/2 function sends to the already spawned and waiting cortex its state, composed of the PId lists and other information which is needed by the cortex to link up and interface with other elements in the distributed phenotype.

backup_genotype(IdsNPIds,NPIds)->
	Neuron_IdsNWeights = get_backup(NPIds,[]),
	update_genotype(IdsNPIds,Neuron_IdsNWeights),
	io:format("Finished updating genotype~n").

	get_backup([NPId|NPIds],Acc)->
		NPId ! {self(),get_backup},
		receive
			{NPId,NId,SWeightTuples,MWeightTuples,PF}->
				get_backup(NPIds,[{NId,SWeightTuples,MWeightTuples,PF}|Acc])
		end;
	get_backup([],Acc)->
		Acc.
%The backup_genotype/2 uses get_backup/2 to contact all the neurons in its NN and request for the neuron's Ids and their Input_IdPs. Once the updated Input_IdPs from all the neurons have been accumulated, they are passed through the update_genotype/2 function to produce updated neurons, and write them to database.

	update_genotype(IdsNPIds,[{N_Id,SI_PIdPs,MI_PIdPs,PF}|WeightPs])->
		N = genotype:dirty_read({neuron,N_Id}),
		Updated_SI_IdPs = convert_PIdPs2IdPs(IdsNPIds,SI_PIdPs,[]),
		Updated_MI_IdPs = convert_PIdPs2IdPs(IdsNPIds,MI_PIdPs,[]),
		U_N = N#neuron{input_idps = Updated_SI_IdPs,input_idps_modulation=Updated_MI_IdPs,pf=PF},
		genotype:write(U_N),
		%io:format("N:~p~n U_N:~p~n Genotype:~p~n U_Genotype:~p~n",[N,U_N,Genotype,U_Genotype]),
		update_genotype(IdsNPIds,WeightPs);
	update_genotype(_IdsNPIds,[])->
		ok.
%For every {N_Id,PIdPs} tuple the update_genotype/3 function extracts the neuron with the id: N_Id, updates the neuron's input_IdPs, and writes the updated neuron to database.

		convert_PIdPs2IdPs(IdsNPIds,[{PId,WeightsP}|Input_PIdPs],Acc)->
			convert_PIdPs2IdPs(IdsNPIds,Input_PIdPs,[{ets:lookup_element(IdsNPIds,PId,2),WeightsP}|Acc]);
		convert_PIdPs2IdPs(_IdsNPIds,[],Acc)->
			lists:reverse(Acc).
%The convert_PIdPs2IdPs/3 performs the conversion from PIds to Ids of every {PId,Weights} tuple in the Input_PIdPs list. The updated Input_IdPs are then returned to the caller.
	
terminate_phenotype(Cx_PId,SPIds,NPIds,APIds,ScapePIds,CPP_PIds,CEP_PIds,Substrate_PId)->
	%io:format("Terminating the phenotype:~nCx_PId:~p~nSPIds:~p~nNPIds:~p~nAPIds:~p~nScapePids:~p~n",[Cx_PId,SPIds,NPIds,APIds,ScapePIds]),
	[PId ! {self(),terminate} || PId <- SPIds],
	[PId ! {self(),terminate} || PId <- APIds],
	[PId ! {self(),terminate} || PId <- NPIds],
	[PId ! {self(),terminate} || PId <- ScapePIds],
	case Substrate_PId == undefined of
		true ->
			ok;
		false ->
			[PId ! {self(),terminate} || PId <- CPP_PIds],
			[PId ! {self(),terminate} || PId <- CEP_PIds],
			Substrate_PId ! {self(),terminate}
	end,
	Cx_PId ! {self(),terminate}.
%The terminate_phenotype/5 function termiantes sensors, actuators, neurons, all private scapes, and the cortex which composes the NN based system.

gather_acks(0)->
	done;	
gather_acks(PId_Index)->
	receive
		{_From,ready}->
			gather_acks(PId_Index-1)
		after 100000 ->
			io:format("******** Not all acks received:~p~n",[PId_Index])
	end.
%gather_acks/1 ensures that the X number of {From,ready} messages are sent to it, before it returns with done. X is set by the caller of the function.
