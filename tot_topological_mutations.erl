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

-module(tot_topological_mutations).
-compile(export_all).
-include("records.hrl").

%ncount_exponential/2 calculates TotMutations by putting the size of the NN to some power Power.
ncount_exponential(Power,Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx = genotype:read({cortex,A#agent.cx_id}),
	TotNeurons = length(Cx#cortex.neuron_ids),
	TotMutations = random:uniform(round(math:pow(TotNeurons,Power))),
	io:format("Tot neurons:~p Performing Tot mutations:~p on:~p~n",[TotNeurons,TotMutations,Agent_Id]),
	TotMutations.

%ncount_linear/2 calcualtes TotMutations by multiplying the size of the NN by the value Multiplier.
ncount_linear(Multiplier,Agent_Id)->
	A = genotype:read({agent,Agent_Id}),
	Cx = genotype:read({cortex,A#agent.cx_id}),
	TotNeurons = length(Cx#cortex.neuron_ids),
	TotMutations = TotNeurons*Multiplier,
	io:format("Tot neurons:~p Performing Tot mutations:~p on:~p~n",[TotNeurons,TotMutations,Agent_Id]),
	TotMutations.
