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

-module(actuator).
-compile(export_all).
-include("records.hrl").

gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf_PId) -> 
	receive 
		{ExoSelf_PId,{Id,Cx_PId,Scape,ActuatorName,VL,Parameters,Fanin_PIds,OpMode}} ->
			put(opmode,OpMode),
			loop(Id,ExoSelf_PId,Cx_PId,Scape,ActuatorName,VL,Parameters,{Fanin_PIds,Fanin_PIds},[])
	end.
%When gen/2 is executed it spawns the actuator element and immediately begins to wait for its initial state message.

loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,VL,Parameters,{[From_PId|Fanin_PIds],MFanin_PIds},Acc) ->
	receive
		{From_PId,forward,Input} ->
			loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,VL,Parameters,{Fanin_PIds,MFanin_PIds},lists:append(Input,Acc));
		{ExoSelf_PId,terminate} ->
			%io:format("Actuator:~p is terminating.~n",[self()])
			ok
	end;
loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,VL,Parameters,{[],MFanin_PIds},Acc)->
	{Fitness,EndFlag} = actuator:AName(ExoSelf_PId,lists:reverse(Acc),Parameters,VL,Scape),
	Cx_PId ! {self(),sync,Fitness,EndFlag},
	loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,VL,Parameters,{MFanin_PIds,MFanin_PIds},[]).
%The actuator process gathers the control signals from the neurons, appending them to the accumulator. The order in which the signals are accumulated into a vector is in the same order as the neuron ids are stored within NIds. Once all the signals have been gathered, the actuator sends cortex the sync signal, executes its function, and then again begins to wait for the neural signals from the output layer by reseting the Fanin_PIds from the second copy of the list.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ACTUATORS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pts(ExoSelf_PId,Result,Parameters,VL,_Scape)->
	io:format("actuator:pts(Result): ~p~n",[Result]),
	case get(opmode) of
		test ->
			{[1,0,0],0};
		_ ->
			{1,0}
	end.
%The pts/2 actuation function simply prints to screen the vector passed to it.

xor_SendOutput(ExoSelf_PId,Output,_Parameters,VL,Scape)->
	Scape ! {self(),action,Output},
	receive 
		{Scape,Fitness,HaltFlag}->
			case get(opmode) of
				test ->
					{[Fitness,0,0],HaltFlag};
				_ ->
					{Fitness,HaltFlag}
			end
	end.
%xor_sim/2 function simply forwards the Output vector to the XOR simulator, and waits for the resulting Fitness and EndFlag from the simulation process.

pb_SendOutput(ExoSelf_PId,Output,Parameters,VL,Scape)->
	Scape ! {self(),push,Parameters,Output},
	receive 
		{Scape,Fitness,HaltFlag}->
			case get(opmode) of
				test ->
					{[Fitness,0,0],HaltFlag};
				_ ->
					{Fitness,HaltFlag}
			end
	end.
	
dtm_SendOutput(ExoSelf_PId,Output,Parameters,VL,Scape)->
	Scape ! {self(),move,Parameters,Output},
	receive 
		{Scape,Fitness,HaltFlag}->
			%io:format("self():~p Fitness:~p HaltFlag:~p~n",[self(),Fitness,HaltFlag]),
			case get(opmode) of
				test ->
					{[Fitness,0,0],HaltFlag};
				_ ->
					{Fitness,HaltFlag}
			end
	end.
	
two_wheels(ExoSelf_PId,Output,Parameters,VL,Scape)->
	%io:format("Scape:~p~n",[Scape]),
	OVL = length(Output),
	{Fitness,HaltFlag}=case OVL == VL of
		true ->
			gen_server:call(Scape,{actuator,ExoSelf_PId,two_wheels,Output});
		false ->
			gen_server:call(Scape,{actuator,ExoSelf_PId,two_wheels,lists:append(Output,lists:duplicate(OVL - VL,0))})
	end,
	%io:format("Ok actuator used~n"),
	case get(opmode) of
		test ->
			{[Fitness,0,0],HaltFlag};
		_ ->
			{Fitness,HaltFlag}
	end.
	
fx_Trade(ExoSelf_PId,Output,Parameters,VL,Scape)->
	%io:format("fx_trade:~p~n",[Output]),
	[TradeSignal] = Output,
	Scape ! {self(),trade,'EURUSD15',functions:trinary(TradeSignal)},
	receive 
		{Scape,Fitness,HaltFlag}->
			case get(opmode) of
				test ->
					{[Fitness,0,0],HaltFlag};
				_ ->
					{Fitness,HaltFlag}
			end
	end.
	
abc_pred(ExoSelf,[Output],Parameters,VL,Scape)->
	Scape ! {self(),classify,get(opmode),Parameters,Output},
	receive 
		{Scape,Fitness,HaltFlag}->
			case get(opmode) of
				test ->
					{[Fitness,0,0],HaltFlag};
				_ ->
					{Fitness,HaltFlag}
			end
	end.
