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

-module(selection_algorithm).
-compile(export_all).
-include("records.hrl").
-define(SURVIVAL_PERCENTAGE,0.5).

-define(SHOF_RATIO,1).
-define(SPECIE_SIZE_LIMIT,10).
-define(REENTRY_PROBABILITY,0.0).
-define(EFF,1). %Efficiency., TODO: this should further be changed from absolute number of neurons, to diff in lowest or avg, and the highest number of neurons

		hof_competition(Specie_Id,RemainingChampionDesignators,Specie_Size_Limit)->%returns a list of new generation of agents for a single specie
			S = genotype:read({specie,Specie_Id}),
			%io:format("HOF_COMEPTITION S:~p~n",[S]),
			io:format("HOF_COMPETITION~n"),
			genotype:write(S#specie{agent_ids=[]}),
			SHOF = S#specie.hall_of_fame,
			NewGen_Ids=case ?SHOF_RATIO < 1 of
				true ->
					Agent_Ids = S#specie.agent_ids,
					Distinguishers = S#specie.hof_distinguishers,
					%Actives = to_champion_form(Agent_Ids,Distinguishers,[]) -- SHOF,
					Actives = RemainingChampionDesignators,
					io:format("SHOF:~w~n",[SHOF]),
					io:format("Actives:~w~n",[Actives]),
					SHOF_FitnessScaled=[{Champ#champion.fs*Champ#champion.main_fitness/math:pow(Champ#champion.tot_n,?EFF),Champ#champion.id}||Champ<-SHOF],
					Active_FitnessScaled=[{Ac#champion.fs*Ac#champion.main_fitness/math:pow(Ac#champion.tot_n,?EFF),Ac#champion.id}||Ac<-Actives],
					TotFitnessActives = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-Active_FitnessScaled]),
					TotFitnessSHOFs = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-SHOF_FitnessScaled]),
					%io:format("TotFitnessActives:~p~nTotFitnessSHOFs:~p~n",[TotFitnessActives,TotFitnessSHOFs]),
					choose_Winners(Specie_Id,Active_FitnessScaled,TotFitnessActives,[],[],round((1-?SHOF_RATIO)*Specie_Size_Limit))++
					choose_Winners(Specie_Id,SHOF_FitnessScaled,TotFitnessSHOFs,[],[],round(?SHOF_RATIO*Specie_Size_Limit));
				false ->
					io:format("SHOF:~w~n",[SHOF]),
					%io:format("RemainingChampionDesignators:~p~n",[RemainingChampionDesignators]),
					Allotments=[{Champ#champion.fs*(Champ#champion.main_fitness/math:pow(Champ#champion.tot_n,?EFF)),Champ#champion.id}||Champ<-SHOF],
%					io:format("Sum:~w~n",[lists:sum([Champ#champion.main_fitness/math:pow(Champ#champion.tot_n,?EFF)||Champ<-SHOF])]),
%					io:format("Denominators:~w~n",[lists:sort([math:pow(Champ#champion.tot_n,?EFF)||Champ<-SHOF])]),
%					io:format("Allotments:~w~n",[Allotments]),
					Tot = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-Allotments]),
					%io:format("Tot:~p~n",[Tot]),
					choose_Winners(Specie_Id,Allotments,Tot,[],[],Specie_Size_Limit)
			end,
			io:format("NewGen_Ids:~w~n",[NewGen_Ids]),
			U_S = genotype:read({specie,Specie_Id}),
			genotype:write(U_S#specie{agent_ids=NewGen_Ids,all_agent_ids=lists:append(NewGen_Ids,U_S#specie.all_agent_ids)}),
			NewGen_Ids.
		
		hof_rank(Specie_Id,RemainingChampionDesignators,Specie_Size_Limit)->
			S = genotype:read({specie,Specie_Id}),
			io:format("S:~p~n",[S]),
			genotype:write(S#specie{agent_ids=[]}),
			SHOF = S#specie.hall_of_fame,
			NewGen_Ids=case ?SHOF_RATIO < 1 of
				true ->
					Agent_Ids = S#specie.agent_ids,
					Distinguishers = S#specie.hof_distinguishers,
					%Actives = to_champion_form(Agent_Ids,Distinguishers,[]) -- SHOF,
					Actives = RemainingChampionDesignators,
					io:format("SHOF:~p~n",[SHOF]),
					io:format("Actives:~p~n",[Actives]),
					Actives_Ranked=assign_rank(lists:sort([{Ac#champion.fs*Ac#champion.main_fitness,Ac#champion.id}||Ac<-Actives]), lists:seq(1,length(Actives)),[]),
					SHOF_Ranked=assign_rank(lists:sort([{Champ#champion.fs*Champ#champion.main_fitness,Champ#champion.id}||Champ<-SHOF]), lists:seq(1,length(SHOF)),[]),
					TotFitnessActives = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-Actives_Ranked]),
					TotFitnessSHOFs = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-SHOF_Ranked]),
					io:format("Actives_Ranked:~p~nSHOF_Ranked:~p~n",[Actives_Ranked,SHOF_Ranked]),
					choose_Winners(Specie_Id,Actives_Ranked,TotFitnessActives,[],[],round((1-?SHOF_RATIO)*Specie_Size_Limit))++
					choose_Winners(Specie_Id,SHOF_Ranked,TotFitnessSHOFs,[],[],round(?SHOF_RATIO*Specie_Size_Limit));
					
				false ->
					SHOF = S#specie.hall_of_fame,
					Allotments=assign_rank(lists:sort([{Champ#champion.fs*Champ#champion.main_fitness,Champ#champion.id}||Champ<-SHOF]),lists:seq(1,length(SHOF)),[]),
					Tot = lists:sum([Val || {Val,_Id}<-Allotments]),
					choose_Winners(Specie_Id,Allotments,Tot,[],[],Specie_Size_Limit)
			end,
			io:format("NewGen_Ids:~p~n",[NewGen_Ids]),
			U_S = genotype:read({specie,Specie_Id}),
			genotype:write(U_S#specie{agent_ids=NewGen_Ids,all_agent_ids=lists:append(NewGen_Ids,U_S#specie.all_agent_ids)}),
			NewGen_Ids.
			
			assign_rank([{_MainFitness,Agent_Id}|Champions],[Rank|RankList],Acc)->
				assign_rank(Champions,RankList,[{Rank,Agent_Id}|Acc]);
			assign_rank([],[],Acc)->
				io:format("Rank:~p~n",[Acc]),
				Acc.
			
		hof_top3(Specie_Id,_RemainingChampionDesignators,Specie_Size_Limit)->
			S = genotype:read({specie,Specie_Id}),
			genotype:write(S#specie{agent_ids=[]}),
			SHOF = S#specie.hall_of_fame,
			Allotments = lists:sublist(lists:reverse(lists:sort([{Champ#champion.fs*Champ#champion.main_fitness,Champ#champion.id}||Champ<-SHOF])),3),
			Tot = lists:sum([Val || {Val,_Id}<-Allotments]),
			io:format("SHOF:~p~n",[SHOF]),
			io:format("Allotments:~p~n",[Allotments]),
			NewGen_Ids=choose_Winners(Specie_Id,Allotments,Tot,[],[],Specie_Size_Limit),
			io:format("NewGen_Ids:~p~n",[NewGen_Ids]),
			U_S = genotype:read({specie,Specie_Id}),
			genotype:write(U_S#specie{agent_ids=NewGen_Ids,all_agent_ids=lists:append(NewGen_Ids,U_S#specie.all_agent_ids)}),
			NewGen_Ids.
		
		hof_efficiency(Specie_Id,RemainingChampionDesignators,Specie_Size_Limit)->
			S = genotype:read({specie,Specie_Id}),
			genotype:write(S#specie{agent_ids=[]}),
			SHOF = S#specie.hall_of_fame,
			NewGen_Ids=case ?SHOF_RATIO < 1 of
				true ->
					Agent_Ids = S#specie.agent_ids,
					Distinguishers = S#specie.hof_distinguishers,
					%Actives = to_champion_form(Agent_Ids,Distinguishers,[]),
					Actives = RemainingChampionDesignators,
					io:format("SHOF:~p~n",[SHOF]),
					io:format("Actives:~p~n",[Actives]),
					Active_NeuralEfficiencyScaled=[{Ac#champion.fs*Ac#champion.main_fitness/Ac#champion.tot_n,Ac#champion.id}||Ac<-Actives],
					SHOF_NeuralEfficiencyScaled=[{Champ#champion.fs*Champ#champion.main_fitness/Champ#champion.tot_n,Champ#champion.id}||Champ<-SHOF],
					TotFitnessActives = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-Active_NeuralEfficiencyScaled]),
					TotFitnessSHOFs = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-SHOF_NeuralEfficiencyScaled]),
					io:format("TotFitnessActives:~p~nTotFitnessSHOFs:~p~n",[TotFitnessActives,TotFitnessSHOFs]),
					choose_Winners(Specie_Id,Active_NeuralEfficiencyScaled,TotFitnessActives,[],[],round((1-?SHOF_RATIO)*Specie_Size_Limit))++
					choose_Winners(Specie_Id,SHOF_NeuralEfficiencyScaled,TotFitnessSHOFs,[],[],round(?SHOF_RATIO*Specie_Size_Limit));
				false ->
					io:format("SHOF:~p~n",[SHOF]),
					SHOF_NeuralEfficiencyScaled=[{Champ#champion.fs*Champ#champion.main_fitness/Champ#champion.tot_n,Champ#champion.id}||Champ<-SHOF],
					io:format("SHOF_NeuralEfficiencyScaled:~p~n",[SHOF_NeuralEfficiencyScaled]),
					TotFitnessSHOFs = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-SHOF_NeuralEfficiencyScaled]),
					io:format("TotFitnessSHOFs:~p~n",[TotFitnessSHOFs]),
					choose_Winners(Specie_Id,SHOF_NeuralEfficiencyScaled,TotFitnessSHOFs,[],[],Specie_Size_Limit)
			end,
			io:format("NewGen_Ids:~p~n",[NewGen_Ids]),
			U_S = genotype:read({specie,Specie_Id}),
			genotype:write(U_S#specie{agent_ids=NewGen_Ids,all_agent_ids=lists:append(NewGen_Ids,U_S#specie.all_agent_ids)}),
			NewGen_Ids.
			
		hof_random(Specie_Id,RemainingChampionDesignators,Specie_Size_Limit)->
			S = genotype:read({specie,Specie_Id}),
			genotype:write(S#specie{agent_ids=[]}),
			SHOF = S#specie.hall_of_fame,
			NewGen_Ids=case ?SHOF_RATIO < 1 of
				true ->
					Agent_Ids = S#specie.agent_ids,
					Distinguishers = S#specie.hof_distinguishers,
					%Actives = to_champion_form(Agent_Ids,Distinguishers,[]),
					Actives = RemainingChampionDesignators,
					io:format("SHOF:~p~n",[SHOF]),
					io:format("Actives:~p~n",[Actives]),
					Active_RandomScaled=[{Ac#champion.fs*1,Ac#champion.id}||Ac<-Actives],
					SHOF_RandomScaled=[{Champ#champion.fs*1,Champ#champion.id}||Champ<-SHOF],
					TotFitnessActives = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-Active_RandomScaled]),
					TotFitnessSHOFs = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-SHOF_RandomScaled]),
					io:format("TotFitnessActives:~p~nTotFitnessSHOFs:~p~n",[TotFitnessActives,TotFitnessSHOFs]),
					choose_Winners(Specie_Id,Active_RandomScaled,TotFitnessActives,[],[],round((1-?SHOF_RATIO)*Specie_Size_Limit))++
					choose_Winners(Specie_Id,SHOF_RandomScaled,TotFitnessSHOFs,[],[],round(?SHOF_RATIO*Specie_Size_Limit));
				false ->
					SHOF = S#specie.hall_of_fame,
					io:format("SHOF:~p~n",[SHOF]),
					SHOF_RandomScaled=[{Champ#champion.fs*1,Champ#champion.id}||Champ<-SHOF],
					TotFitnessSHOFs = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-SHOF_RandomScaled]),
					io:format("TotalOffspring:~p~n",[round(?SHOF_RATIO*Specie_Size_Limit)]),
					choose_Winners(Specie_Id,SHOF_RandomScaled,TotFitnessSHOFs,[],[],Specie_Size_Limit)
			end,
			io:format("NewGen_Ids:~p~n",[NewGen_Ids]),
			U_S = genotype:read({specie,Specie_Id}),
			genotype:write(U_S#specie{agent_ids=NewGen_Ids,all_agent_ids=lists:append(NewGen_Ids,U_S#specie.all_agent_ids)}),
			NewGen_Ids.

			choose_Winners(Specie_Id,Agents,TotalFitness,OffspringAcc,ReentryAcc,0)->
				reenter(ReentryAcc,Specie_Id),
				OffspringAcc++ReentryAcc;
			choose_Winners(Specie_Id,Agents,TotalFitness,OffspringAcc,ReentryAcc,AgentIndex)->
				%io:format("1~n"),
				try choose_Winner(Specie_Id,Agents,(random:uniform(100)/100)*TotalFitness,0) of
					{OffspringId,offspring}->%io:format("2~n"),
						choose_Winners(Specie_Id,Agents,TotalFitness,[OffspringId|OffspringAcc],ReentryAcc,AgentIndex-1);
					{Agent_Id,reentry}->%io:format("3~n"),
						case lists:member(Agent_Id,ReentryAcc) of
							true ->%io:format("4~n"),
								choose_Winners(Specie_Id,Agents,TotalFitness,OffspringAcc,ReentryAcc,AgentIndex);
							false ->%io:format("5~n"),
								choose_Winners(Specie_Id,Agents,TotalFitness,OffspringAcc,[Agent_Id|ReentryAcc],AgentIndex-1)
						end
				catch
					_:_ ->
						io:format("CHOOSE WINNER CRASHING::BACKTRACE:~p~n",[erlang:get_stacktrace()]),
						timer:sleep(10000000),
						choose_Winners(Specie_Id,Agents,TotalFitness,OffspringAcc,ReentryAcc,AgentIndex)
				end.
				
				reenter([Agent_Id|ReentryIds],Specie_Id)->
					io:format("REENTERING:~p~n",[Agent_Id]),
					S = genotype:read({specie,Specie_Id}),
					SHOF = S#specie.hall_of_fame,
					U_SHOF = lists:keydelete(Agent_Id, 3, SHOF),
					U_S = S#specie{hall_of_fame=U_SHOF},
					%U_S = S#specie{hall_of_fame=U_SHOF,agent_ids=[Agent_Id|S#specie.agent_ids]},
					A = genotype:read({agent,Agent_Id}),
					U_A = A#agent{champion_flag=[rentered|A#agent.champion_flag]},%true, false, lost, rentered
					genotype:write(U_S),
					genotype:write(U_A),
					reenter(ReentryIds,Specie_Id);
					%Remove agent from phof and shof, tag re-entry (not lost)
				reenter([],_Specie_Id)->
					ok.
				
			choose_Winner(Specie_Id,[{_PortionSize,Agent_Id}],_Index,_Acc)->
				case random:uniform() =< ?REENTRY_PROBABILITY of
					true ->
						%io:format("CHOOSE REENTRY WINNER, Index:~p~n",[Agent_Id]),
						{Agent_Id,reentry};
					false ->%io:format("Starting here~n"),
						A = genotype:read({agent,Agent_Id}),
						OffspringAgent_Id = create_MutantAgentCopy(Agent_Id),
						%io:format("CHOOSE WINNER, Index:~p OffspringAgent_Id:~p~n",[Agent_Id,OffspringAgent_Id]),
						U_A = A#agent{offspring_ids=[OffspringAgent_Id|A#agent.offspring_ids]},%true, false, lost, rentered
						%io:format("OffspringAgent_Id:~p~n",[OffspringAgent_Id]),
						genotype:write(U_A),
						OffspringA = genotype:read({agent,OffspringAgent_Id}),
						U_OffspringA = OffspringA#agent{champion_flag=[false|OffspringA#agent.champion_flag]},
						%io:format("U_OffspringA:~p~n",[U_OffspringA]),
						genotype:write(U_OffspringA),
						{OffspringAgent_Id,offspring}
						%choose agent as parent
						%create clone, mutate clone, return offspring
				end;
			choose_Winner(Specie_Id,[{PortionSize,Agent_Id}|Allotments],Index,Acc)->
				case (Index >= Acc) and (Index =< (Acc+PortionSize)) of
					true ->%io:format("WIndex:~p~n",[Index]),
						case random:uniform() =< ?REENTRY_PROBABILITY of
							true ->
								%io:format("CHOOSE REENTRY WINNER, Index:~p~n",[{Index,PortionSize,Agent_Id}]),
								{Agent_Id,reentry};
							false ->%io:format("Starting here~n"),
								A = genotype:read({agent,Agent_Id}),
								OffspringAgent_Id = create_MutantAgentCopy(Agent_Id),
								%io:format("CHOOSE WINNER, Index:~p OffspringAgent_Id:~p~n",[{Index,PortionSize,Agent_Id},OffspringAgent_Id]),
								U_A = A#agent{offspring_ids=[OffspringAgent_Id|A#agent.offspring_ids]},%true, false, lost, rentered
								%io:format("OffspringAgent_Id:~p~n",[OffspringAgent_Id]),
								genotype:write(U_A),
								OffspringA = genotype:read({agent,OffspringAgent_Id}),
								U_OffspringA = OffspringA#agent{champion_flag=[false|OffspringA#agent.champion_flag]},
								%io:format("U_OffspringA:~p~n",[U_OffspringA]),
								genotype:write(U_OffspringA),
								{OffspringAgent_Id,offspring}
								%choose agent as parent
								%create clone, mutate clone, return offspring
						end;
					false ->
						choose_Winner(Specie_Id,Allotments,Index,Acc+PortionSize)
				end.
		
		construct_AgentSummaries([Agent_Id|Agent_Ids],Acc)->
			A = genotype:dirty_read({agent,Agent_Id}),
			construct_AgentSummaries(Agent_Ids,[{A#agent.fitness,length((genotype:dirty_read({cortex,A#agent.cx_id}))#cortex.neuron_ids),Agent_Id}|Acc]);
		construct_AgentSummaries([],Acc)->
			Acc.
%The construct_AgentSummaries/2 reads the agents in the Agent_Ids list, and composes a list of tuples of the following format: [{AgentFitness,AgentTotNeurons,Agent_Id}...]. This list of tuples is reffered to as AgentSummaries. Once the AgentSummaries list is composed, it is returned to the caller.

	create_MutantAgentCopy(Agent_Id)->
		AgentClone_Id = genotype:clone_Agent(Agent_Id),
		%io:format("AgentClone_Id:~p and now the entire agent:~p~n",[AgentClone_Id,mnesia:dirty_read({agent,AgentClone_Id})]),
		genome_mutator:mutate(AgentClone_Id),
		AgentClone_Id.
%The create_MutantAgentCopy/1 first creates a clone of the Agent_Id, and then uses the genome_mutator:mutate/1 function to mutate that clone, returning the id of the cloned agent to the caller.
				
	create_MutantAgentCopy(Agent_Id,safe)->%TODO
		A = genotype:dirty_read({agent,Agent_Id}),
		S = genotype:dirty_read({specie,A#agent.specie_id}),
		AgentClone_Id = genotype:clone_Agent(Agent_Id),
		Agent_Ids = S#specie.agent_ids,
		genotype:write(S#specie{agent_ids = [AgentClone_Id|Agent_Ids]}),
		%io:format("AgentClone_Id:~p~n",[AgentClone_Id]),
		genome_mutator:mutate(AgentClone_Id),
		AgentClone_Id.
%The create_MutantAgentCopy/2 is similar to arity 1 function of the same name, but it also adds the id of the cloned mutant agent to the specie record to which the original belonged. The specie with its updated agent_ids is then written to database, and the id of the mutant clone is returned to the caller.

choose_CompetitionWinner([{_MutantAlotment,Fitness,Profile,Agent_Id}],_Index,_Acc)->%TODO: Does this really work?
	{Fitness,Profile,Agent_Id};
choose_CompetitionWinner([{MutantAlotment,Fitness,Profile,Agent_Id}|AlotmentsP],Index,Acc)->
	case (Index > Acc) and (Index =< Acc+MutantAlotment) of
		true ->
			{Fitness,Profile,Agent_Id};
		false ->
			choose_CompetitionWinner(AlotmentsP,Index,Acc+MutantAlotment)
	end.
