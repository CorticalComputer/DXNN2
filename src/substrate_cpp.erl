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

-module(substrate_cpp).
-compile(export_all).
-include("records.hrl").

gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf_PId) ->
	receive 
		{ExoSelf_PId,{Id,Cx_PId,Substrate_PId,CPPName,VL,Parameters,Fanout_PIds}} ->
			loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CPPName,VL,Parameters,Fanout_PIds)
	end.
%When gen/2 is executed, it spawns the substrate_cpp element and immediately begins to wait for its initial state message.

loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CPPName,VL,Parameters,Fanout_PIds)->
	receive
		{Substrate_PId,Presynaptic_Coords,Postsynaptic_Coords}->
			SensoryVector = functions:CPPName(Presynaptic_Coords,Postsynaptic_Coords),
			[Pid ! {self(),forward,SensoryVector} || Pid <- Fanout_PIds],
			loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CPPName,VL,Parameters,Fanout_PIds);
		{Substrate_PId,Presynaptic_Coords,Postsynaptic_Coords,IOW}->
			SensoryVector = functions:CPPName(Presynaptic_Coords,Postsynaptic_Coords,IOW),
			%io:format("SensoryVector:~p~n",[SensoryVector]),
			[Pid ! {self(),forward,SensoryVector} || Pid <- Fanout_PIds],
			loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CPPName,VL,Parameters,Fanout_PIds);
		{ExoSelf_PId,terminate} ->
			%io:format("substrate_cpp:~p is terminating.~n",[Id]),
			ok
	end.
