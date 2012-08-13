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

-module(epitopes).
-compile(export_all).
-include("records.hrl").

start()->
	spawn(epitopes,db,[]).
db()->
	ets:file2tab(abc_pred10),
	ets:file2tab(abc_pred12),
	ets:file2tab(abc_pred14),
	ets:file2tab(abc_pred16),
	ets:file2tab(abc_pred18),
	ets:file2tab(abc_pred20),
	receive
		terminate ->
			ok
	end.

sim(ExoSelf)->
	receive
		{From,sense,OpMode,Parameters}->
			%[TableName,StartIndex,EndIndex,StartBenchIndex,EndBenchIndex] = [abc_pred16,841,1120,841,1120],%%TODO
			[TableName,StartIndex,EndIndex,StartBenchIndex,EndBenchIndex] = Parameters,
			Out=case get(abc_pred) of
				undefined ->
					case OpMode of
						gt ->
							put(abc_pred,StartIndex),
							Sequence = ets:lookup_element(TableName,StartIndex,2),
							lists:flatten([translate_seq(Char) || Char <- Sequence]);
						benchmark ->
							put(abc_pred,StartBenchIndex),
							Sequence = ets:lookup_element(TableName,StartBenchIndex,2),
							lists:flatten([translate_seq(Char) || Char <- Sequence])
					end;
				Ind ->
					Index = case Ind == 0 of
						true -> 1;
						false -> Ind
					end,
					Sequence = ets:lookup_element(TableName,Index,2),
					lists:flatten([translate_seq(Char) || Char <- Sequence])
			end,
			From ! {self(),percept,Out},
			sim(ExoSelf);
		{From,classify,OpMode,Parameters,Output}->
			%[TableName,StartIndex,EndIndex,StartBenchIndex,EndBenchIndex] = [abc_pred16,841,1120,841,1120],%%TODO
			[TableName,StartIndex,EndIndex,StartBenchIndex,EndBenchIndex] = Parameters,
			case get(abc_pred) of
				undefined ->
					exit("Exit with error from sim epitopes~n");
				Ind ->
					Index = case Ind == 0 of
						true -> 1;
						false -> Ind
					end,
					Classification = ets:lookup_element(TableName,Index,3),
					HaltFlag = case OpMode of
						gt ->
							case Index == EndIndex of
								true -> erase(abc_pred),1;
								false -> put(abc_pred,(Index+1) rem 1401),0
							end;
						benchmark ->
							case Index == EndBenchIndex of
								true -> erase(abc_pred),1;
								false -> put(abc_pred,(Index+1) rem 1401),0
							end
					end,
					case (Classification == functions:bin(Output)) of%%TODO
					%case (Classification==0) and (0==functions:bin(Output)) of
						true ->
							From ! {self(),1,HaltFlag};
						false ->
							From ! {self(),0,HaltFlag}
					end
			end,
			sim(ExoSelf);
		{ExoSelf,terminate} ->
			ok
	%after 10000 ->
		%io:format("Exiting with error from epitopes sim~n")
	end.
	
	translate_seq1(Char)->
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
			88 -> 1
		end.
		
	translate_seq(Char)->
		case Char of
			65 -> [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
			82 -> [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
			78 -> [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
			68 -> [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
			67 -> [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
			69 -> [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
			81 -> [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
			71 -> [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0];
			72 -> [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0];
			73 -> [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0];
			76 -> [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0];
			75 -> [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0];
			77 -> [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0];
			70 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0];
			80 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0];
			83 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0];
			84 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0];
			87 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0];
			89 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0];
			86 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0];
			88 -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
		end.
		
test(ExperimentName)->
	ets:new(testing,[named_table,set,public]),
	[E] = mnesia:dirty_read({experiment,ExperimentName}),
	io:format("E:~p~n",[E]),
	Traces = E#experiment.trace_acc,
	BestGen_Champions = [get_best(Trace) || Trace <- Traces],
	[{BOTB_F,BOTB_Id}|_] = lists:reverse(lists:sort(BestGen_Champions)),
	[exoself:start(ExoselfId,void,benchmark) || {GenFitness,ExoselfId} <- BestGen_Champions],
	timer:sleep(5000),
	get_avg(ets:first(potato),[]),
	exoself:start(BOTB_Id,void,benchmark),
	ets:delete(testing).
	
	get_best(T)->
		Stats = T#trace.stats,
		GenTest_Champions=[Stat#stat.gentest_fitness || [Stat] <- Stats],
		[Best|_]=lists:reverse(lists:sort(GenTest_Champions)),
		Best.
	
	get_avg('$end_of_table',Acc)->
		io:format("~p~n",[{functions:avg(Acc),functions:std(Acc),lists:max(Acc),lists:min(Acc),functions:avg(Acc)/280}]);
	get_avg(Key,Acc)->
		Val = ets:lookup_element(testing,Key,2),
		get_avg(ets:next(testing,Key),[Val|Acc]).
