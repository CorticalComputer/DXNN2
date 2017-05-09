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

-module(tuning_duration).
-compile(export_all).
-include("records.hrl").

const(Parameter,_N_Ids,_Generation)->
	ConstMaxAttempts = Parameter,
	ConstMaxAttempts.
%const/3 returns the preset const max_attempts value.

wsize_proportional(Parameter,N_Ids,Generation)->
	Power = Parameter,
	Active_NIds = extract_RecGenNIds(N_Ids,Generation,3,[]),
	Tot_ActiveNeuron_Weights = extract_NWeightCount(Active_NIds,0),
	10 + functions:sat(round(math:pow(Tot_ActiveNeuron_Weights,Power)),100,0).
%wsize_proportional/3 calculats the max_attempts value based on the individual agent's parameters, in this case the max_attempts is proportional to the agent's number of weights belonging to the neurons which were added or mutated within the last 3 generations.

	extract_RecGenNIds([N_Id|N_Ids],Generation,AgeLimit,Acc)->
		N = genotype:dirty_read({neuron,N_Id}),
		NeuronGen = N#neuron.generation,
		case NeuronGen >= (Generation-AgeLimit) of
			true ->
				extract_RecGenNIds(N_Ids,Generation,AgeLimit,[N_Id|Acc]);
			false ->
				extract_RecGenNIds(N_Ids,Generation,AgeLimit,Acc)
		end;
	extract_RecGenNIds([],_Generation,_AgeLimit,Acc)->
		Acc.
%extract_RecGenNIds/4 extracts the NIds of all neurons whose age is lower or equal to the AgeLimit.

	extract_NWeightCount([N_Id|RecGenN_Ids],Acc)->
		N = genotype:dirty_read({neuron,N_Id}),
		TotWeights = case N#neuron.af of
			{circuit,_}->
				circuit:tot_weights(N#neuron.input_idps);
			_ ->
				Input_IdPs = N#neuron.input_idps,
				lists:sum([length(Weights) || {_IId,Weights} <- Input_IdPs])
		end,
		extract_NWeightCount(RecGenN_Ids,TotWeights+Acc);
	extract_NWeightCount([],Acc)->
		Acc.
%extract_NWeightCount/2 counts the number of weights in total belonging to the list of neuron ids that the function was called with.

nsize_proportional(Parameter,N_Ids,Generation)->
	Power = Parameter,
	Tot_Neurons = length(extract_RecGenNIds(N_Ids,Generation,3,[])),
	20 + functions:sat(round(math:pow(Tot_Neurons,Power)),100,0).
%nsize_proportional/3 calculates the max_attempts to be proportional to the number of neurons which were within the last 3 generations mutated or added to the NN.
