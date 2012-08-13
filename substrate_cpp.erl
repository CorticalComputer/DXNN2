%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This source code and work is provided and developed by Gene I. Sher & DXNN Research Group WWW.DXNNResearch.COM
%
%Copyright (C) 2009 by Gene Sher, DXNN Research Group, CorticalComputer@gmail.com
%All rights reserved.
%
%This code is licensed under the version 3 of the GNU General Public License. Please see the LICENSE file that accompanies this project for the terms of use.
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
