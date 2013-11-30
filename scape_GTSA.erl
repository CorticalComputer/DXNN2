%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright (C) 2009 by Gene Sher, DXNN Research Group CorticalComputer@gmail.com	%
%											%
%   Licensed under the Apache License, Version 2.0 (the "License");			%
%   you may not use this file except in compliance with the License.			%
%   You may obtain a copy of the License at						%
%											%
%     http://www.apache.org/licenses/LICENSE-2.0 					%
%											%
%   Unless required by applicable law or agreed to in writing, software			%
%   distributed under the License is distributed on an "AS IS" BASIS,			%
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.		%
%   See the License for the specific language governing permissions and			%
%   limitations under the License.							%
%%%%%%%%%%%%%%%%%%%% Deus Ex Neural Network :: DXNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-module(scape_GTSA).%General Time Series Analysis
-compile(export_all).
-record(state,{
	file_name,
	table_name,
	heart_beat,
	window_size,
	info
}).

-record(a_state,{
	index_start,
	index_current,
	index_end,
	tn,
	window_length,
	window,
	tot_rows
}).

-record(info,{
	name,
	ivl,
	ovl,
	trn_end,
	val_end,
	tst_end
}).

%%%%%%%%%%%%%%%%%%%%%%%%%%%% SCAPE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start(ETS_FileName,FileName)->
	%load the file
	%initialize the state
	%start the scape loop
	{ok,TN} = ets:file2tab(ETS_FileName),
	[{0,Info}] = ets:lookup(TN,0),
	%HeartBeat = create_HeartBeat(FileName,ETS_FileName,TN),
	Init_State = #state{
		file_name = ETS_FileName,
		table_name = TN,
		%heart_beat = HeartBeat,
		info=Info
	},
	loop(Init_State).
	
loop(S)->
	receive
		{From,get_percept,Command,Parameters}->
			?MODULE:Command(From,Parameters,S#state.info),
			?MODULE:loop(S);
		{From,act,Command,Parameters,Output}->
			?MODULE:Command(From,Output,Parameters,S#state.info),
			?MODULE:loop(S);
		{HeartBeat_PId,update_table,NewData}->
			update_table(S#state.file_name,S#state.table_name,NewData),
			?MODULE:loop(S);
		terminate->
			ok
	end.
	
create_HeartBeat(FileName,ETS_FileName,TN)->
	spawn(?MODULE,heart_beat,[FileName,ETS_FileName,TN]).
	
heart_beat(FN,ETS_FN,Info)->
	receive
		terminate->
			io:format("heart_beat for:~nFileName: ~p~nETS FileName: ~p~nTableName: ~p~n is terminating...~n",[FN,ETS_FN,Info#info.name])
	after 10000 ->
		%try reading the FN
		%if successfull write into TN, save it to ETS_FN.
		heart_beat(FN,ETS_FN,Info)
	end.
	
update_table(FileName,TableName,NewData)->
	ok.
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%% SENSORS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
get_window(Agent_PId,Parameters,Info)->
	case get(Agent_PId) of
		undefined ->
			AS=create_init_state(Agent_PId,Parameters,Info),
			put(Agent_PId,AS),
			io:format("AS1:~p~n",[AS]),
			Agent_PId ! {self(),percept,AS#a_state.window};	
		AS->
			Window = AS#a_state.window,
			Vector=ets:lookup_element(AS#a_state.tn,AS#a_state.index_current,2),
			U_Window = Vector ++ lists:sublist(Window,(AS#a_state.tot_rows-1)*Info#info.ivl),
			put(Agent_PId,AS#a_state{window=U_Window}),
			io:format("AS2:~p~n",[AS#a_state{window=U_Window}]),
			Agent_PId ! {self(),percept,U_Window}
	end.
	
	create_init_state(Agent_PId,Parameters,Info)->
		A_State=case Parameters of
			[train,TotRows] ->
				#a_state{
					tn = Info#info.name,
					index_start=1,
					index_current=1,
					index_end=Info#info.trn_end,
					window_length=TotRows*Info#info.ivl,
					tot_rows=TotRows
				};
			[validation,TotRows]->
				case Info#info.val_end of
					undefined ->						
						#a_state{
							tn = Info#info.name,
							index_start=1,
							index_current=1,
							index_end=Info#info.trn_end,
							window_length=TotRows*Info#info.ivl,
							tot_rows=TotRows
						};
					Val_End ->
						#a_state{
							tn = Info#info.name,
							index_start=Info#info.trn_end+1,
							index_current=Info#info.trn_end+1,
							index_end=Info#info.val_end,
							window_length=TotRows*Info#info.ivl,
							tot_rows=TotRows
						}
				end;
			[test,TotRows]->
				case Info#info.tst_end of
					undefined ->
						#a_state{
							tn = Info#info.name,
							index_start=1,
							index_current=1,
							window_length=TotRows*Info#info.ivl,
							tot_rows=TotRows
						};
					Tst_End ->
						#a_state{
							tn = Info#info.name,
							index_start=Info#info.val_end+1,
							index_current=Info#info.val_end+1,
							window_length=TotRows*Info#info.ivl,
							tot_rows=TotRows
						}
				end
		end,
		Vector = lists:reverse(lists:append([ets:lookup_element(A_State#a_state.tn,Index,2)||Index<-lists:seq(A_State#a_state.index_current,A_State#a_state.index_current+TotRows)])),
		A_State#a_state{
			window = Vector,
			index_current=A_State#a_state.index_current+TotRows
		}.

%%%%%%%%%%%%%%%%%%%%%%%%%%%% ACTUATORS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predict_value(Agent_PId,[Prediction],Parameters,Info)->			
	AS = get(Agent_PId),
	Window = AS#a_state.window,
	U_Index_Current = AS#a_state.index_current+1,
	io:format("Prediction:~p Parameters:~p~n",[Prediction,Parameters]),
	case ets:lookup(AS#a_state.tn,U_Index_Current) of
		[{U_Index_Current,Vector,_}]->
			[OpMod,ElementIndex]=Parameters,
			ExpectedValue=lists:nth(ElementIndex,Vector),
			Fitness = 1/(abs(ExpectedValue-Prediction)+0.0001),
			io:format("Fitness:~p~n",[Fitness]),
			put(Agent_PId,AS#a_state{index_current=U_Index_Current}),
			Agent_PId ! {self(),[Fitness],0};
		[] ->
			erase(Agent_PId),
			Agent_PId ! {self(),[0],1}
			
	end.
	
	
predict_direction(Agent_PId,Parameters,Info)->			
	ok.

make_trade(Agent_PId,Parameters,Info)->			
	ok.

