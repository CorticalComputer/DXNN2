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

-module(sensor).
-compile(export_all).
-include("records.hrl").

gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf_PId) ->
	receive 
		{ExoSelf_PId,{Id,Cx_PId,Scape,SensorName,VL,Parameters,Fanout_PIds,OpMode}} ->
			put(opmode,OpMode),
			loop(Id,ExoSelf_PId,Cx_PId,Scape,SensorName,VL,Parameters,Fanout_PIds)
	end.
%When gen/2 is executed it spawns the sensor element and immediately begins to wait for its initial state message.

loop(Id,ExoSelf_PId,Cx_PId,Scape,SensorName,VL,Parameters,Fanout_PIds)->
	receive
		{Cx_PId,sync}->
			SensoryVector = sensor:SensorName(ExoSelf_PId,VL,Parameters,Scape),
			%advanced_fanout(PId,SensoryVector,Fanout_PIdPs)
			[Pid ! {self(),forward,SensoryVector} || Pid <- Fanout_PIds],
			loop(Id,ExoSelf_PId,Cx_PId,Scape,SensorName,VL,Parameters,Fanout_PIds);
		{ExoSelf_PId,terminate} ->
			%io:format("Sensor:~p is terminating.~n",[Id]),
			ok
	end.
%The sensor process accepts only 2 types of messages, both from the cortex. The sensor can either be triggered to begin gathering sensory data based on its sensory role, or terminate if the cortex requests so.

advanced_fanout(PId,SensoryVector,[{all,Fanout_PId}|Fanout_PIdPs])->
	Fanout_PId ! {self(),forward,SensoryVector},
	advanced_fanout(PId,SensoryVector,Fanout_PIdPs);
advanced_fanout(PId,SensoryVector,[{single,Index,Fanout_PId}|Fanout_PIdPs])->
	Fanout_PId ! {self(),forward, [lists:nth(Index,SensoryVector)]},
	advanced_fanout(PId,SensoryVector,Fanout_PIdPs).

rng(ExoSelf_PId,VL,_Scape)->
	rng1(VL,[]).
rng1(0,Acc)->
	Acc;
rng1(VL,Acc)-> 
	rng1(VL-1,[random:uniform()|Acc]).
%rng/2 is a simple random number generator that produces a vector of random values, each between 0 and 1. The length of the vector is defined by the VL, which itself is specified within the sensor record.

xor_GetInput(ExoSelf_PId,VL,_Parameters,Scape)->
	Scape ! {self(),sense},
	receive
		{Scape,percept,SensoryVector}->
			case length(SensoryVector)==VL of
				true ->
					SensoryVector;
				false ->
					io:format("Error in sensor:xor_sim/3, VL:~p SensoryVector:~p~n",[VL,SensoryVector]),
					lists:duplicate(VL,0)
			end
	end.
%xor_GetInput/2 contacts the XOR simulator and requests the sensory vector, which in this case should be a binary vector of length 2. The sensor checks that the incoming sensory signal, the percept, is indeed of length 2. If the vector length differs, then this is printed to the console and a dummy vector of appropriate length is constructed.

pb_GetInput(ExoSelf_PId,VL,Parameters,Scape)->
	Scape ! {self(),sense,Parameters},
	receive
		{Scape,percept,SensoryVector}->
			case length(SensoryVector)==VL of
				true ->
					SensoryVector;
				false ->
					io:format("Error in sensor:pb_GetInput/3, VL:~p SensoryVector:~p~n",[VL,SensoryVector]),
					lists:duplicate(VL,0)
			end
	end.
	
dtm_GetInput(ExoSelf_PId,VL,Parameters,Scape)->
	Scape ! {self(),sense,Parameters},
	receive
		{Scape,percept,SensoryVector}->
			%io:format("self():~p SensoryVector:~p~n",[self(),SensoryVector]),
			case length(SensoryVector)==VL of
				true ->
					SensoryVector;
				false ->
					io:format("Error in sensor:dtm_GetInput/3, VL:~p SensoryVector:~p~n",[VL,SensoryVector]),
					lists:duplicate(VL,0)
			end
	end.
	
%distance_scanner(Exoself_Id,VL,[Spread,Density,RadialOffset],Scape)->
%	case gen_server:call(Scape,{get_all,avatars}) of
%		destroyed->
%			lists:duplicate(VL,-1);
%		Avatars ->
%			Self = lists:keyfind(self(),2,Avatars),
%			Loc = Self#avatar.loc,
%			Direction = Self#avatar.direction,
%			distance_scanner(silent,{1,0,0},Density,Spread,Loc,Direction,lists:keydelete(self(), 2, Avatars))
%	end.
%
%color_scanner(Exoself_Id,VL,[Spread,Density,RadialOffset],Scape)->
%	case gen_server:call(Scape,{get_all,avatars}) of
%		destroyed->
%			lists:duplicate(VL,-1);
%		Avatars ->
%			Self = lists:keyfind(self(),2,Avatars),
%			Loc = Self#avatar.loc,
%			Direction = Self#avatar.direction,
%			color_scanner(silent,{1,0,0},Density,Spread,Loc,Direction,lists:keydelete(self(), 2, Avatars))
%	end.
	
fx_PCI(Exoself_Id,VL,Parameters,Scape)->
	[HRes,VRes] = Parameters,
	case get(opmode) of
		gt	->
			%Normal, assuming we have 10000 rows, we start from 1000 to 6000
			Scape ! {self(),sense,'EURUSD15',close,[HRes,VRes,graph_sensor],2000,1000};
		benchmark ->
			Scape ! {self(),sense,'EURUSD15',close,[HRes,VRes,graph_sensor],1001,500};
		test ->
			Scape ! {self(),sense,'EURUSD15',close,[HRes,VRes,graph_sensor],501,last}
	end,
	receive 
		{_From,Result}->
			Result
	end.

fx_PLI(Exoself_Id,VL,Parameters,Scape)->
	[HRes,Type] = Parameters,%Type=open|close|high|low
	case get(opmode) of
		gt	->
			%Normal, assuming we have 10000 rows, we start from 1000 to 6000
			Scape ! {self(),sense,'EURUSD15',close,[HRes,list_sensor],5000,1000};
		benchmark ->
			Scape ! {self(),sense,'EURUSD15',close,[HRes,list_sensor],1001,500};
		test ->
			Scape ! {self(),sense,'EURUSD15',close,[HRes,list_sensor],501,last}
	end,
	receive 
		{_From,Result}->
			normalize(Result)
	end.
	
	normalize(Vector)->
		Normalizer=math:sqrt(lists:sum([Val*Val||Val<-Vector])),
		[Val/Normalizer || Val <- Vector].
	
fx_Internals(Exoself_Id,VL,Parameters,Scape)->
	Scape ! {self(),sense,internals,Parameters},
	receive
		{PId,Result}->
			Result
	end.
	
abc_pred(Exoself_Id,VL,Parameters,Scape)->
	Scape ! {self(),sense,get(opmode),Parameters},
	receive
		{PId,percept,Result}->
			Result
	end.
	
	translate_seq(Char)->
		case Char of
			65 -> -1;
			82 -> -0.9;
			78 -> -0.8;
			68 -> -0.7;
			67 -> -0.6;
			69 -> -0.5;
			81 -> -0.4;
			71 -> -0.3;
			72 -> -0.2;
			73 -> -0.1;
			76 -> 0;
			75 -> 0.1;
			77 -> 0.2;
			70 -> 0.3;
			80 -> 0.4;
			83 -> 0.5;
			84 -> 0.6;
			87 -> 0.7;
			89 -> 0.8;
			86 -> 0.9;
			85 -> 1
		end.
