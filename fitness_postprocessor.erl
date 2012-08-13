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

-module(fitness_postprocessor).
-compile(export_all).
-include("records.hrl").
-define(EFF,0.05). %Efficiency.

none(Agent_Summaries)->
	lists:reverse(lists:sort(Agent_Summaries)).

size_proportional(Agent_Summaries)->
	SDX=lists:reverse(lists:sort([{Fitness/math:pow(TotN,?EFF),{Fitness,TotN,Agent_Id}}||{Fitness,TotN,Agent_Id}<-Agent_Summaries])),
	ProperlySorted_AgentSummaries = [Val || {_,Val}<-SDX],
	ProperlySorted_AgentSummaries.
	
novelty_proportional(Agent_Summeries)->
	void.
