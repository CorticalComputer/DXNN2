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

-module(substrate_cep).
-compile(export_all).
-include("records.hrl").

gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf_PId) -> 
	receive 
		{ExoSelf_PId,{Id,Cx_PId,Substrate_PId,CEPName,Parameters,Fanin_PIds}} ->
			loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CEPName,Parameters,{Fanin_PIds,Fanin_PIds},[])
	end.
%When gen/2 is executed it spawns the actuator element and immediately begins to wait for its initial state message.

loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CEPName,Parameters,{[From_PId|Fanin_PIds],MFanin_PIds},Acc) ->
	receive
		{From_PId,forward,Input} ->
			loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CEPName,Parameters,{Fanin_PIds,MFanin_PIds},lists:append(Input,Acc));
		{ExoSelf_PId,terminate} ->
			%io:format("Substrate_CEP:~p is terminating.~n",[self()])
			ok
	end;
loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CEPName,Parameters,{[],MFanin_PIds},Acc)->
	ProperlyOrdered_Input=lists:reverse(Acc),
	substrate_cep:CEPName(ProperlyOrdered_Input,Parameters,Substrate_PId),
	loop(Id,ExoSelf_PId,Cx_PId,Substrate_PId,CEPName,Parameters,{MFanin_PIds,MFanin_PIds},[]).
%The substrate_cep process gathers the control signals from the neurons, appending them to the accumulator. The order in which the signals are accumulated into a vector is in the same order that the neuron ids are stored within NIds. Once all the signals have been gathered, the substrate_cep executes its function, forwards the processed signal to the substrate, and then again begins to wait for the neural signals from the output layer by reseting the Fanin_PIds from the second copy of the list.


%%%%%%%% Substrate_CEPs %%%%%%%% 
set_weight(Output,_Parameters,Substrate_PId)->
	[Val] = Output,
	Threshold = 0.33,
	Weight = if 
		Val > Threshold ->
			(functions:scale(Val,1,Threshold)+1)/2;
		Val < -Threshold ->
			(functions:scale(Val,-Threshold,-1)-1)/2;
		true ->
			0
	end,
	Substrate_PId ! {self(),set_weight,[Weight]}.
%

set_abcn(Output,_Parameters,Substrate_PId)->
	Substrate_PId ! {self(),set_abcn,Output}.
	
delta_weight(Output,_Parameters,Substrate_PId)->
	[Val] = Output,
	Threshold = 0.33,
	DW = if 
		Val > Threshold ->
			(functions:scale(Val,1,Threshold)+1)/2;
		Val < -Threshold ->
			(functions:scale(Val,-Threshold,-1)-1)/2;
		true ->
			0
	end,
	Substrate_PId ! {self(),set_iterative,[DW]}.
