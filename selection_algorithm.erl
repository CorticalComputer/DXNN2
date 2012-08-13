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

-module(selection_algorithm).
-compile(export_all).
-include("records.hrl").
-define(SURVIVAL_PERCENTAGE,0.5).

competition(ProperlySorted_AgentSummaries,NeuralEnergyCost,PopulationLimit)->
	TotSurvivors = round(length(ProperlySorted_AgentSummaries)*?SURVIVAL_PERCENTAGE),
	Valid_AgentSummaries = lists:sublist(ProperlySorted_AgentSummaries,TotSurvivors),
	Invalid_AgentSummaries = ProperlySorted_AgentSummaries -- Valid_AgentSummaries,
	{_,_,Invalid_AgentIds} = lists:unzip3(Invalid_AgentSummaries),
	[genotype:delete_Agent(Agent_Id) || Agent_Id <- Invalid_AgentIds],
	io:format("Valid_AgentSummaries:~p~n",[Valid_AgentSummaries]),
	io:format("Invalid_AgentSummaries:~p~n",[Invalid_AgentSummaries]),
	TopAgentSummaries = lists:sublist(Valid_AgentSummaries,3),
	{_TopFitnessList,_TopTotNs,TopAgent_Ids} = lists:unzip3(TopAgentSummaries),
	io:format("NeuralEnergyCost:~p~n",[NeuralEnergyCost]),

	{AlotmentsP,NextGenSize_Estimate} = calculate_alotments(Valid_AgentSummaries,NeuralEnergyCost,[],0),
	Normalizer = NextGenSize_Estimate/PopulationLimit,
	io:format("Population size normalizer:~p~n",[Normalizer]),
	NewGenAgent_Ids = gather_survivors(AlotmentsP,Normalizer,[]),
	{NewGenAgent_Ids,TopAgent_Ids}.
%The competition/3 is part of the selection algorithm dubbed "competition". The function first executes calculate_alotments/4 to calculate the number of offspring alloted for each agent in the Sorted_AgentSummaries list. The function then calculates the Normalizer value, which is used then used to proportionalize the alloted number of offspring to each agent, to ensure that the final specie size is within PopulationLimit. The function then drops into the gather_survivors/3 function which, using the normalized offspring allotment values, creates the actual mutant offspring.

competition_WithDiversifier(ProperlySorted_AgentSummaries,NeuralEnergyCost,PopulationLimit)->
	TotSurvivors = round(length(ProperlySorted_AgentSummaries)*?SURVIVAL_PERCENTAGE),
	Valid_AgentSummaries=uniquify(ProperlySorted_AgentSummaries,TotSurvivors),
	%Valid_AgentSummaries = lists:sublist(ProperlySorted_AgentSummaries,TotSurvivors),
	Invalid_AgentSummaries = ProperlySorted_AgentSummaries -- Valid_AgentSummaries,
	{_FitnessList,_TotNList,Invalid_AgentIds} = lists:unzip3(Invalid_AgentSummaries),
	[genotype:delete_Agent(Agent_Id) || Agent_Id <- Invalid_AgentIds],
	io:format("Valid_AgentSummaries:~p~n",[Valid_AgentSummaries]),
	io:format("Invalid_AgentSummaries:~p~n",[Invalid_AgentSummaries]),
	TopAgentSummaries = lists:sublist(Valid_AgentSummaries,3),
	{_TopFitnessList,_TopTotNs,TopAgent_Ids} = lists:unzip3(TopAgentSummaries),
	io:format("NeuralEnergyCost:~p~n",[NeuralEnergyCost]),

	{AlotmentsP,NextGenSize_Estimate} = calculate_alotments(Valid_AgentSummaries,NeuralEnergyCost,[],0),
	Normalizer = NextGenSize_Estimate/PopulationLimit,
	io:format("Population size normalizer:~p~n",[Normalizer]),
	NewGenAgent_Ids = gather_survivors(AlotmentsP,Normalizer,[]),
	{NewGenAgent_Ids,TopAgent_Ids}.
	
	
	uniquify(ProperlySorted_DXSummaries,TotSurvivors)->
	%[DX_Summary|Sorted_DXSummaries] = ProperlySorted_DXSummaries,%lists:sublist(ProperlySorted_DXSummaries,TotSurvivors),
	[DX_Summary|Sorted_DXSummaries] = lists:sublist(ProperlySorted_DXSummaries,TotSurvivors),
	{Fitness,TotN,_DX_Id} = DX_Summary,
	Diversified_DXSummaries = diversify([{Fitness,TotN}],Sorted_DXSummaries,TotSurvivors-1,[DX_Summary]).
			
		diversify(_TopProfiles,_Sorted_DXSummaries,0,Acc)->
			lists:reverse(Acc);
		diversify(Profiles,[DX_Summary|Sorted_DXSummaries],KeepTot,Acc)->
			{Fitness,TotN,DX_Id} = DX_Summary,
			%case compare_profiles(Profiles,Profile) of
			case compare_profilesf(Profiles,{Fitness,TotN}) of
				true->
					diversify([{Fitness,TotN}|Profiles],Sorted_DXSummaries,KeepTot-1,[DX_Summary|Acc]);
				false ->
					diversify(Profiles,Sorted_DXSummaries,KeepTot,Acc)
			end;
		diversify(_TopProfiles,[],KeepTot,Acc)->
			lists:reverse(Acc).
			
			compare_profilesf([{TopFitness,TopTotN}|TopProfiles],{Fitness,TotN})->%Better make Fitnes part of profile
				case (TopTotN == TotN) and (TopFitness == Fitness) of
					true ->
						false;
					false ->
						compare_profilesf(TopProfiles,{Fitness,TotN})
				end;
			compare_profilesf([],_ProfileP)->
				true.

	calculate_alotments([{Fitness,TotNeurons,Agent_Id}|Sorted_AgentSummaries],NeuralEnergyCost,Acc,NewPopAcc)->
		NeuralAlotment = Fitness/NeuralEnergyCost,
		MutantAlotment = NeuralAlotment/TotNeurons,
		U_NewPopAcc = NewPopAcc+MutantAlotment,
		calculate_alotments(Sorted_AgentSummaries,NeuralEnergyCost,[{MutantAlotment,Fitness,TotNeurons,Agent_Id}|Acc],U_NewPopAcc);
	calculate_alotments([],_NeuralEnergyCost,Acc,NewPopAcc)->
		%io:format("NewPopAcc:~p~n",[NewPopAcc]),
		{Acc,NewPopAcc}.
%The calculate_alotments/4 function accepts the AgentSummaries list and for each agent, using the NeuralEnergyCost, calcualtes how many offspring that agent can produce by using the agent's Fitness, TotNEurons, and NeuralEnergyCost values. The function first calculates how many neurons the agent is alloted, based on the agent's fitness and the cost of each neuron (which itself was calculated based on the average performance of the population). From the number of neurons alloted to the agent, the function then calculates how many offspring the agent should be alloted, by deviding the agent's NN size by the number of neurons it is alloted. The function also keeps track of how many offspring will be created from all these agents in general, by adding up all the offspring alotements. The calcualte_alotments/4 function does this for each tuple in the AgentSummaries, and then returns the calculated alotment list and NewPopAcc to the caller.

	gather_survivors([{MutantAlotment,Fitness,TotNeurons,Agent_Id}|AlotmentsP],Normalizer,Acc)->
		Normalized_MutantAlotment = round(MutantAlotment/Normalizer),
		io:format("Agent_Id:~p Normalized_MutantAlotment:~p~n",[Agent_Id,Normalized_MutantAlotment]),
		SurvivingAgent_Ids = case Normalized_MutantAlotment >= 1 of
			true ->
				MutantAgent_Ids = case Normalized_MutantAlotment >= 2 of
					true ->
						[population_monitor:create_MutantAgentCopy(Agent_Id)|| _ <-lists:seq(1,Normalized_MutantAlotment-1)];
					false ->
						[]
				end,
				[Agent_Id|MutantAgent_Ids];
			false ->
				io:format("Deleting agent:~p~n",[Agent_Id]),
				genotype:delete_Agent(Agent_Id),
				[]
		end,
		gather_survivors(AlotmentsP,Normalizer,lists:append(SurvivingAgent_Ids,Acc));
	gather_survivors([],_Normalizer,Acc)->
		io:format("New Population:~p PopSize:~p~n",[Acc,length(Acc)]),
		Acc.
%The gather_survivors/3 function accepts the list composed of the alotment tuples and a population normalizer value calculated by the competition/3 function, and from those values calculates the actual number of offspring that each agent should produce, creating those mutant offspring and accumulating the new generation agent ids. FOr each Agent_Id the function first calculates the noramlized offspring alotment value, to ensure that the final nubmer of agents in the specie is within the popualtion limit of that specie. If the offspring alotment value is less than 0, the agent is killed. If the offspring alotment is 1, the parent agent is allowed to survive to the next generation, but is not allowed to create any new offspring. If the offspring alotment is greater than one, then the Normalized_MutantAlotment-1 offspring are created from this fit agent, by calling upon the create_MutantAgentCopy/1 function, which rerns the id of the new mutant offspring. Once all the offspring have been created, the function returns to the caller a list of ids, composed of the surviving parent agent ids, and their offspring.

top3(ProperlySorted_AgentSummaries,NeuralEnergyCost,PopulationLimit)->
	TotSurvivors = 3,
	Valid_AgentSummaries = lists:sublist(ProperlySorted_AgentSummaries,TotSurvivors),
	Invalid_AgentSummaries = ProperlySorted_AgentSummaries -- Valid_AgentSummaries,
	{_,_,Invalid_AgentIds} = lists:unzip3(Invalid_AgentSummaries),
	{_,_,Valid_AgentIds} = lists:unzip3(Valid_AgentSummaries),
	[genotype:delete_Agent(Agent_Id) || Agent_Id <- Invalid_AgentIds],
	io:format("Valid_AgentSummaries:~p~n",[Valid_AgentSummaries]),
	io:format("Invalid_AgentSummaries:~p~n",[Invalid_AgentSummaries]),
	TopAgentSummaries = lists:sublist(Valid_AgentSummaries,3),
	{_TopFitnessList,_TopTotNs,TopAgent_Ids} = lists:unzip3(TopAgentSummaries),
	io:format("NeuralEnergyCost:~p~n",[NeuralEnergyCost]),
	NewGenAgent_Ids = breed(Valid_AgentIds,PopulationLimit-TotSurvivors,[]),
	{NewGenAgent_Ids,TopAgent_Ids}.
		
	breed(_Valid_AgentIds,0,Acc)->
		Acc;
	breed(Valid_AgentIds,OffspringIndex,Acc)->%TODO
		Parent_AgentId = lists:nth(random:uniform(length(Valid_AgentIds)),Valid_AgentIds),
		MutantAgent_Id = population_monitor:create_MutantAgentCopy(Parent_AgentId),
		breed(Valid_AgentIds,OffspringIndex-1,[MutantAgent_Id|Acc]).
%The breed/3 function is part of a very simple selection algorithm, which just selects the top 3 most fit agents, and then uses the create_MutantAgentCopy/1 function to create their offspring.

competition(ProperlySorted_AgentSummaries)->
	TotEnergy = lists:sum([Fitness || {Fitness,_TotN,_Agent_Id}<-ProperlySorted_AgentSummaries]),
	TotNeurons = lists:sum([TotN || {_Fitness,TotN,_Agent_Id} <- ProperlySorted_AgentSummaries]),
	NeuralEnergyCost = TotEnergy/TotNeurons,
	{AlotmentsP,Normalizer} = calculate_alotments(ProperlySorted_AgentSummaries,NeuralEnergyCost,[],0),
	Choice = random:uniform(),
	{WinnerFitness,WinnerTotN,WinnerAgent_Id}=choose_CompetitionWinner(AlotmentsP,Normalizer,Choice,0),
	{WinnerFitness,WinnerTotN,WinnerAgent_Id}.
		
	choose_CompetitionWinner([{MutantAlotment,Fitness,TotN,Agent_Id}|AlotmentsP],Normalizer,Choice,Range_From)->
		Range_To = Range_From+MutantAlotment/Normalizer,
		case (Choice >= Range_From) and (Choice =< Range_To) of
			true ->
				{Fitness,TotN,Agent_Id};
			false ->
				choose_CompetitionWinner(AlotmentsP,Normalizer,Choice,Range_To)
		end;
	choose_CompetitionWinner([],_Normalizer,_Choice,_Range_From)->
		exit("********ERROR:choose_CompetitionWinner:: reached [] without selecting a winner.").
