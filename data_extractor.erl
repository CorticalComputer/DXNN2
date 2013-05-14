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


-module(data_extractor).
-compile(export_all).
-include("records.hrl").

-define(PACKAGER,vowel_recognition).

check_table(TableName)->
	{ok,TN} = ets:file2tab(TableName),
	io:format("TN:~p~n",[TN]),
	table_dump(TN,ets:first(TN)),
	ets:delete(TN).
	
	table_dump(TN,'$end_of_table')->
		ok;
	table_dump(TN,Key)->
		io:format("~p~n",[ets:lookup(TN,Key)]),
		table_dump(TN,ets:next(TN,Key)).

start(URL,SplitVals,FileName)->
	start(URL,SplitVals,FileName,?PACKAGER).
start(URL,SplitVals,FileName,Packager)->
	case file:read_file(URL) of
		{ok,Data} ->
			file:close(URL),
			List = binary_to_list(Data),
			Extracted_Values = list_to_dvals(SplitVals,List,[]),
			store(Extracted_Values,FileName,Packager);
		{error,Error} ->
			io:format("Error:~p~n",[Error])
	end.
	
	list_to_dvals(_SplitVals,[],Acc)->
		lists:reverse(Acc);
	list_to_dvals(SplitVals,List,Acc)->
		{DVal_Line,Remainder} = splitter(SplitVals,List,[]),
		list_to_dvals(SplitVals,Remainder,[DVal_Line|Acc]).
		
		splitter([SplitVal|SplitVals],List,Acc)->
			case SplitVal of
				skip ->
					[_|Remainder] = List,
					splitter(SplitVals,Remainder,Acc);
				_ ->
					{Val,Remainder}=split_with(SplitVal,List),
					Number = list_to_number(Val),
					splitter(SplitVals,Remainder,[Number|Acc])
			end;
		splitter([],List,Acc)->
			{lists:reverse(Acc),List}.
	
		split_with(Seperator,List)->split_with(Seperator,List,[]).
		split_with(Seperator,[Char|List],ValAcc)->
			case Char of
				Seperator->
					{lists:reverse(ValAcc),List};
				_ ->
					split_with(Seperator,List,[Char|ValAcc])
			end;
		split_with(_Seperator,[],ValAcc)->
			{lists:reverse(ValAcc),[]}.	
		
list_to_number(List)->
	try list_to_float(List) of
		Float ->
			Float
	catch 
		_:_ ->
			try list_to_integer(List) of
				Int ->
					Int
			catch
				_:_ ->
					List
			end
	end.
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Packagers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
store(Extracted_Values,Name,Packager)->
	TableName = ets:new(Name,[set,public,named_table]),
	data_extractor:Packager(TableName,Extracted_Values,1),
	ets:tab2file(TableName,Name),
	ets:delete(TableName).
	
simple_store(TableName,[Line|Lines],Index)->
	ets:insert(TableName,{Index,Line}),
	simple_store(TableName,Lines,Index+1);
simple_store(TableName,[],Index)->
	io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).

%0 0 0 -3.639 0.418 -0.670 1.779 -0.168 1.627 -0.388 0.529 -0.874 -0.814  0
%data_extractor:start("Vowel_Recognition",[32,32,32,32,32,32,32,32,32,32,32,32,32,skip,10],vowel_recognition).  	
vowel_recognition(TableName,[Line|Lines],Index)->
	{Type,Remainder}=lists:split(3,Line),
	{Features,Classification} = lists:split(10,Remainder),
	ets:insert(TableName,{Index,Type,Features,Classification}),
	vowel_recognition(TableName,Lines,Index+1);
vowel_recognition(TableName,[],Index)->
	io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).

%*CM078: 0.0491 0.0279 0.0592 0.1270 0.1772 0.1908 0.2217 0.0768 0.1246 0.2028 0.0947 0.2497 0.2209 0.3195 0.3340 0.3323 0.2780 0.2975 0.2948 0.1729 0.3264 0.3834 0.3523 0.5410 0.5228 0.4475 0.5340 0.5323 0.3907 0.3456 0.4091 0.4639 0.5580 0.5727 0.6355 0.7563 0.6903 0.6176 0.5379 0.5622 0.6508 0.4797 0.3736 0.2804 0.1982 0.2438 0.1789 0.1706 0.0762 0.0238 0.0268 0.0081 0.0129 0.0161 0.0063 0.0119 0.0194 0.0140 0.0332 0.0439}
rw()->
	{ok,MT}=ets:file2tab(mines),
	{ok,RT}=ets:file2tab(rocks),
	TableName = ets:new(mines_vs_rocks,[set,public,named_table]),
	I1=mines_vs_rocks(TableName,ets:tab2list(MT),1,0),
	I2=mines_vs_rocks(TableName,ets:tab2list(RT),I1,0),
	I3=mines_vs_rocks(TableName,ets:tab2list(MT),I2,1),
	I4=mines_vs_rocks(TableName,ets:tab2list(RT),I3,1),
	ets:tab2file(TableName,mines_vs_rocks),
	ets:delete(TableName),
	{I2,I4}.

mines_vs_rocks(TableName,[{_,Line}|Lines],Index,TestFlag)->
	io:format("~p~n",[Line]),
	[Identifier|Features]=Line,
	U_Index=case {Identifier,TestFlag} of
		{[42,67,77|_],0} ->
			ets:insert(TableName,{Index,0,Features,[1,0]}),
			Index+1;
		{[42,67,82|_],0} ->
			ets:insert(TableName,{Index,0,Features,[0,1]}),
			Index+1;
		{[67,77|_],1} ->
			ets:insert(TableName,{Index,1,Features,[1,0]}),
			Index+1;
		{[67,82|_],1} ->
			ets:insert(TableName,{Index,1,Features,[0,1]}),
			Index+1;
			_ ->
				Index
	end,
	mines_vs_rocks(TableName,Lines,U_Index,TestFlag);
mines_vs_rocks(_TableName,[],Index,_TestFlag)->
	Index.
	
abc_pred1(TableName,[Line|Lines],Index)->
	[Sequence,Classification] = Line,
	ets:insert(TableName,{Index,Sequence,Classification}),
	abc_pred1(TableName,Lines,Index+1);
abc_pred1(TableName,[],Index)->
	io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).
