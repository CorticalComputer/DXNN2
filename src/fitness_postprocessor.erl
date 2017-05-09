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
