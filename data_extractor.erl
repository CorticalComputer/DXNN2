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

-define(PACKAGER,mnist).

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
	
%[$,||_<-lists:seq(1,784)]++[10].
mnist(TableName,[Line|Lines],Index)->
	[Classification|Sequence] = lists:reverse(Line),
	%io:format("Length:~p~n",[length(Sequence)]),
	ets:insert(TableName,{Index,lists:reverse(Sequence),[Classification]}),
	mnist(TableName,Lines,Index+1);
mnist(TableName,[],Index)->
	IVL = 28*28,
	OVL = 10,
	Info={info,IVL,OVL,TableName,50000,60000,70000},
	ets:insert(TableName,{0,Info}),
	io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).

	move()->
		{ok,Mnist} = ets:file2tab(mnist),
		{ok,Mnist_Test}=ets:file2tab(mnist_test),
		move(Mnist,60001,Mnist_Test,1),
		ets:tab2file(Mnist,mnist).
		
		move(To_TN,70001,From_TN,10001)->
			ok;
		move(To_TN,To_Index,From_TN,From_Index)->
			[{From_Index,Sequence,Classification}] = ets:lookup(From_TN,From_Index),
			ets:insert(To_TN,{To_Index,Sequence,Classification}),
			move(To_TN,To_Index+1,From_TN,From_Index+1).
			
	update()->
		{ok,Mnist} = ets:file2tab(mnist),
		update(Mnist,1,70001),
		IVL = 28*28,
		OVL = 10,
		Info={info,mnist,IVL,OVL,50000,60000,70000},
		ets:insert(Mnist,{0,Info}),
		ets:tab2file(Mnist,mnist).
		
		update(_TN,EndIndex,EndIndex)->
			ok;
		update(TN,Index,EndIndex)->
			io:format("Index:~p~n",[Index]),
			[{Index,Sequence,[Classification]}] = ets:lookup(TN,Index),
			Class=case Classification of
				0 -> [0,0,0,0,0,0,0,0,0,1];
				1 -> [0,0,0,0,0,0,0,0,1,0];
				2 -> [0,0,0,0,0,0,0,1,0,0];
				3 -> [0,0,0,0,0,0,1,0,0,0];
				4 -> [0,0,0,0,0,1,0,0,0,0];
				5 -> [0,0,0,0,1,0,0,0,0,0];
				6 -> [0,0,0,1,0,0,0,0,0,0];
				7 -> [0,0,1,0,0,0,0,0,0,0];
				8 -> [0,1,0,0,0,0,0,0,0,0];
				9 -> [1,0,0,0,0,0,0,0,0,0]
			end,
			ets:insert(TN,{Index,Sequence,Class}),
			update(TN,Index+1,EndIndex).
		
	mnist_ConvertBin()->
		{ok,Mnist} = ets:file2tab(mnist),
		mnist_ConvertBin(Mnist,1,70001),
		ets:rename(Mnist,mnist_bin),
		ets:tab2file(mnist_bin,mnist_bin).
		
		mnist_ConvertBin(_TN,EndIndex,EndIndex)->
			ok;
		mnist_ConvertBin(TN,Index,EndIndex)->
			[{Index,Sequence,Class}] = ets:lookup(TN,Index),
			U_Sequence = [case Val > 0 of true -> 1; false -> 0 end || Val <- Sequence],
			ets:insert(TN,{Index,U_Sequence,Class}),
			mnist_ConvertBin(TN,Index+1,EndIndex).
		
-record(info,{name,ivl,ovl,trn_end,val_end,tst_end}).
		mnist_SetInfo()->
			IVL = 28*28,
			OVL = 10,
			Info = #info{
				name=mnist,
				ivl=IVL,
				ovl=OVL,
				trn_end=50000,
				val_end=60000,
				tst_end=70000
			},
			{ok,Mnist} = ets:file2tab(mnist),
			ets:insert(Mnist,{0,Info}),
			ets:tab2file(Mnist,mnist).
		
%[$/,$/]++[$,||_<-lists:seq(1,3)]++[13,10].
%data_extractor:start("data.txt",[$/,$/]++[$,||_<-lists:seq(1,3)]++[10],hedge_fund,hedge_fund).	
hedge_fund(TableName,[DataNames|Lines],Index)->
	ets:insert(TableName,{0,DataNames}),
	hedge_fund1(TableName,Lines,1).
	
	hedge_fund1(TableName,[Line|Lines],Index)->
		[Date|Vector] = Line,
		[_|RV]=lists:reverse(Vector),
		io:format("Date:~p~n Vector:~p~n",[Date,lists:reverse(RV)]),
		ets:insert(TableName,{Index,lists:reverse(RV),Date}),
		hedge_fund1(TableName,Lines,Index+1);
	hedge_fund1(TableName,[],Index)->
		[{_Key,Vec,_Date}]=ets:lookup(TableName,Index-1),
		[{0,DataNaes}]=ets:lookup(TableName,0),
		Info = #info{
			name=TableName,
			ivl=length(Vec),
			ovl=1,
			trn_end=Index-1,
			val_end=Index-1,
			tst_end=Index-1
		},
		ets:insert(TableName,{0,Info}),
		io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).

%[9|| _<-lists:seq(1,8)]++[10]
%data_extractor:start("Gm12878.chr22.ChromHMM.bed",[9|| _<-lists:seq(1,8)]++[10],chrom_HMM,chrom_HMM).
chrom_HMM(TableName,Lines,Index)->
	Unique_Tags = find_unique_tags(Lines,[]),	
	%Tot_Tags = length(Unique_Tags),
	chrom_HMM(TableName,Lines,Index,Unique_Tags).
	
	find_unique_tags([Line|Lines],Acc)->
		[_,_,_,Tag|_] = Line,
		case lists:member(Tag,Acc) of
			true ->
				find_unique_tags(Lines,Acc);
			false ->
				find_unique_tags(Lines,[Tag|Acc])
		end;
	find_unique_tags([],Acc)->
		Acc.
	
	chrom_HMM(TableName,[Line|Lines],Index,Unique_Tags)->
		[ChrName,StartBPI,EndBPI,Tag|_] = Line,
		%io:format("StartBPI:~p EndBPI:~p Tag:~p~n",[StartBPI,EndBPI,Tag]),
		TotSteps = (EndBPI - StartBPI)/200,
		Vector = tag_it(Tag,Unique_Tags,[]),
		Entries = [{Index+I,Vector,StartBPI+I*200}||I<-lists:seq(0,round(TotSteps))],
		%io:format("Line:~p TotSteps:~p Vector:~p~n",[Line,TotSteps,Vector]),
		%[io:format("Entry:~p~n",[Entry])||Entry<-Entries],
		[ets:insert(TableName,Entry)||Entry<-Entries],
		chrom_HMM(TableName,Lines,round(Index+TotSteps+1),Unique_Tags);
	chrom_HMM(TableName,[],Index,Unique_Tags)->
		IVL = length(Unique_Tags),
		Info = #info{
			name=TableName,
			ivl=IVL,
			ovl=undefined,
			trn_end=Index-1
		},
		ets:insert(TableName,{0,Info}),
		io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]),
		io:format("Tag_Order:~p~n",[[{Tag,Index}|| {Tag,Index}<- lists:zip(Unique_Tags,lists:seq(1,length(Unique_Tags)))]]).
		%fit_to_chr22(TableName).
	
		tag_it(Target_Tag,[Target_Tag|Tags],Acc)->
			tag_it(Target_Tag,Tags,[1|Acc]);
		tag_it(Target_Tag,[Tag|Tags],Acc)->
			tag_it(Target_Tag,Tags,[0|Acc]);
		tag_it(_Target_Tag,[],Acc)->
			lists:reverse(Acc).

		chrom_HMMTags()->
			[{"ReprD",1},
			   {"EnhF",2},
			   {"PromP",3},
			   {"H4K20",4},
			   {"Enh",5},
			   {"Art",6},
			   {"Gen5'",7},
			   {"Gen3'",8},
			   {"ElonW",9},
			   {"Tss",10},
			   {"EnhW",11},
			   {"EnhWF",12},
			   {"CtcfO",13},
			   {"Repr",14},
			   {"ReprW",15},
			   {"Ctcf",16},
			   {"DnaseD",17},
			   {"Elon",18},
			   {"Pol2",19},
			   {"DnaseU",20},
			   {"Low",21},
			   {"FaireW",22},
			   {"TssF",23},
			   {"PromF",24},
			   {"Quies",25}].
	
%[9|| _<-lists:seq(1,14)]++[10]
deep_gene(TableName,[Line|Lines],Index)->
	[BasePairIndex|Vector]=Line,
	ets:insert(TableName,{Index,Vector,BasePairIndex}),
	deep_gene(TableName,Lines,Index+1);
deep_gene(TableName,[],Index)->
	IVL = 14,
	Info = #info{
		name=TableName,
		ivl=IVL,
		ovl=undefined,
		trn_end=Index-1
	},
	ets:insert(TableName,{0,Info}),
	io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).
	
	deep_gene_switch()->
		{ok,TN}=ets:file2tab(chr22),
		deep_gene_switch(TN,ets:first(TN),1),
		ets:tab2file(TN,chr22).
	deep_gene_switch(TN,'$end_of_table',Index)->
		IVL = 14,
		Info = #info{
			name=chr22,
			ivl=IVL,
			ovl=undefined,
			trn_end=Index-1
		},
		ets:insert(TN,{0,Info});
	deep_gene_switch(TN,Key,Index)->
		io:format("~p~n",[ets:lookup(TN,Key)]),
		[{Key,BasePairIndex,Vector}] = ets:lookup(TN,Key),
		ets:insert(TN,{Key,Vector,BasePairIndex}),
		deep_gene_switch(TN,ets:next(TN,Key),Index+1).
		
	deep_gene_AddInfo()->
		{ok,TN}=ets:file2tab(chr22),
		IVL = 14,
		Info = #info{
			name=chr22,
			ivl=IVL,
			ovl=undefined,
			trn_end=166035
		},
		ets:insert(TN,{0,Info}),
		ets:tab2file(TN,chr22).
		
	dg_clean()->
		{ok,TN} = ets:file2tab(chr22),
		{Acc,Acc1}=dg_clean(TN,1,166036,[],1),
		[ets:insert(TN,Tuple) || Tuple <- Acc],
		io:format("TotRemains:~p~n",[Acc1-1]),
		New_TN = chr22_clean,
		[{0,Info}] = ets:lookup(TN,0),
		U_Info = Info#info{trn_end = Acc1-1,name=New_TN},
		ets:rename(TN,New_TN),
		Result1=ets:insert(New_TN,{0,U_Info}),
		Result2=ets:tab2file(New_TN,New_TN),
		io:format("R1:~p R2:~p~n",[Result1,Result2]),
		ets:delete(New_TN).
		
		dg_clean(_TN,EndIndex,EndIndex,Acc,Acc1)->
			{Acc,Acc1};
		dg_clean(TN,Index,EndIndex,Acc,Acc1)->
			[{Index,Sequence,Class}] = ets:lookup(TN,Index),
			case lists:sum(Sequence) == 0 of
				true ->
					ets:delete(TN,Index),
					dg_clean(TN,Index+1,EndIndex,Acc,Acc1);
				false ->
					ets:delete(TN,Index),
					dg_clean(TN,Index+1,EndIndex,[{Acc1,Sequence,Class}|Acc],Acc1+1)
			end.
	
	dg_scale1()->
		{ok,TN} = ets:file2tab(chr22_clean),
		[{0,Info}] = ets:lookup(TN,0),
		Table_Statistics=find_DataStatistics(TN,Info),
		Index_End = case Info#info.tst_end of
			undefined ->
				Info#info.trn_end;
			Tst_End ->
				Tst_End
		end,
		dg_scale1(TN,1,Index_End+1,Table_Statistics),
		New_TN = chr22_clean_scaled1,
		ets:rename(TN,New_TN),
		ets:insert(New_TN,{0,Info#info{name=New_TN}}),
		ets:tab2file(New_TN,New_TN).
		
		dg_scale1(_TN,EndIndex,EndIndex,_TS)->
			ok;
		dg_scale1(TN,Index,EndIndex,TS)->
			[{Index,Sequence,Class}] = ets:lookup(TN,Index),
			U_Sequence = scale_vec(Sequence,TS,[]),
			ets:insert(TN,{Index,U_Sequence,Class}),
			dg_scale1(TN,Index+1,EndIndex,TS).
			
			scale_vec([Val|Sequence],[{Min,Avg,Max}|TS],Acc)->
				scale_vec(Sequence,TS,[Val/Max|Acc]);
			scale_vec([],[],Acc)->
				lists:reverse(Acc).
				
	dg_scale2()->
		{ok,TN} = ets:file2tab(chr22_clean),
		[{0,Info}] = ets:lookup(TN,0),
		Index_End = case Info#info.tst_end of
			undefined ->
				Info#info.trn_end;
			Tst_End ->
				Tst_End
		end,
		dg_scale2(TN,1,Index_End+1),
		New_TN = chr22_clean_scaled2,
		ets:rename(TN,New_TN),
		ets:insert(New_TN,{0,Info#info{name=New_TN}}),
		ets:tab2file(New_TN,New_TN).
		
		dg_scale2(_TN,EndIndex,EndIndex)->
			ok;
		dg_scale2(TN,Index,EndIndex)->
			[{Index,Sequence,Class}] = ets:lookup(TN,Index),
			U_Sequence = [math:log(Val+math:sqrt(Val*Val+1))||Val<-Sequence],
			ets:insert(TN,{Index,U_Sequence,Class}),
			dg_scale2(TN,Index+1,EndIndex).
	
	find_DataStatistics(FN)->
		{ok,TN} = ets:file2tab(FN),
		[{0,Info}] = ets:lookup(TN,0),
		find_DataStatistics(TN,Info).
	find_DataStatistics(TN,Info)->
		Index_End = case Info#info.tst_end of
			undefined ->
				Info#info.trn_end;
			Tst_End ->
				Tst_End
		end,
		[{1,Sequence,Class}] = ets:lookup(TN,1),
		InitStats=[{Val,Val,Val}||Val<-Sequence],%{Min,Avg,Max}
		find_DataStatistics(TN,2,Index_End+1,InitStats).
		
%[48.0055,53.782,156.853,145.479,618.019,64.63,19.552,17.845,2390.32,449.651,107.808,218.897,10.507,8.1865]

		find_DataStatistics(TN,EndIndex,EndIndex,Statistics)->
			U_Statistics = [{Min,Avg_Acc/(EndIndex-1),Max}||{Min,Avg_Acc,Max}<-Statistics],
			%io:format("Statistics:~p~n",[U_Statistics]),
			U_Statistics;
		find_DataStatistics(TN,Index,EndIndex,Statistics)->
			[{Index,Sequence,Class}] = ets:lookup(TN,Index),
			U_Statistics = update_min_avg_max(Statistics,Sequence,[]),
			find_DataStatistics(TN,Index+1,EndIndex,U_Statistics).

			update_min_avg_max([{Min,Avg_Acc,Max}|List1],[Val|List2],Acc)->
				update_min_avg_max(List1,List2,[{min(Min,Val),Avg_Acc+Val,max(Max,Val)}|Acc]);
			update_min_avg_max([],[],Acc)->
				lists:reverse(Acc).
	
	dg_bin()->
		{ok,TN} = ets:file2tab(chr22_clean),
		[{0,Info}] = ets:lookup(TN,0),
		Index_End = case Info#info.tst_end of
			undefined ->
				Info#info.trn_end;
			Tst_End ->
				Tst_End
		end,
		dg_bin(TN,1,Index_End+1),
		ets:rename(TN,chr22_bin),
		ets:tab2file(chr22_bin,chr22_bin).
	
		dg_bin(_TN,EndIndex,EndIndex)->
			ok;
		dg_bin(TN,Index,EndIndex)->
			[{Index,Sequence,Class}] = ets:lookup(TN,Index),
			U_Sequence = [case Val == 0 of true -> io:format("Zero~n"), 0; false -> 1 end||Val<-Sequence],
			ets:insert(TN,{Index,U_Sequence,Class}),
			dg_bin(TN,Index+1,EndIndex).
			
		count_zeroes(FN)->
			{ok,TN} = ets:file2tab(FN),
			[{0,Info}] = ets:lookup(TN,0),
			Index_End = case Info#info.tst_end of
				undefined ->
					Info#info.trn_end;
				Tst_End ->
					Tst_End
			end,
			count_zeroes(TN,1,Index_End+1,0,0),
			ets:delete(TN).
			
			count_zeroes(_TN,EndIndex,EndIndex,ZAcc,NZAcc)->
				io:format("Zeroes:~p NotZeroes:~p Ratio:~p~n",[ZAcc,NZAcc,ZAcc/NZAcc]);
			count_zeroes(TN,Index,EndIndex,ZAcc,NZAcc)->
				[{Index,Sequence,Class}] = ets:lookup(TN,Index),
				NZ = lists:sum(Sequence),
				Z = length(Sequence) - NZ,
				count_zeroes(TN,Index+1,EndIndex,ZAcc+Z,NZAcc+NZ).
	
change_id()->
	C=genotype:dirty_read({circuit,undefined}),
	genotype:write(C#circuit{id=mnist_3}).
	
%[$,|| _<-lists:seq(1,13)]++[10]
wine(TableName,[Line|Lines],Index)->
	[Classification|Vector]=Line,
	Class=case Classification of
		1 -> [0,0,1];
		2 -> [0,1,0];
		3 -> [1,0,0]
	end,
	ets:insert(TableName,{Index,Vector,Class}),
	wine(TableName,Lines,Index+1);
wine(TableName,[],Index)->
	IVL = 13,
	OVL = 3,
	Info = #info{
		name=wine,
		ivl=IVL,
		ovl=OVL,
		trn_end=Index-1
	},
	ets:insert(TableName,{0,Info}),
	io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).

%[9|| _<- lists:seq(1,9)]++[10]
chr_hmm(TableName,[Line|Lines],Index)->
	%io:format("~p~n",[{Line,Lines}]),
	[ChrNumber,From,To,Tag|Remainder] = Line,
	ets:insert(TableName,{Index,From,To,Tag,Remainder}),
	chr_hmm(TableName,Lines,Index+1);
chr_hmm(TableName,[],Index)->
	Info = #info{
		name=chr22_hmm
	},
	ets:insert(TableName,{0,Info}),
	io:format("Stored to ETS table:~p Index reached:~p~n",[TableName,Index-1]).

create_CircuitTestFiles()->
	%i10o20: IVL=10,OVL=20,
	I10O20 = ets:new(i10o20,[set,private,named_table]),
	Trni10o20=[{Index,[random:uniform(2)-1||_<-lists:seq(1,10)],[random:uniform(2)-1||_<-lists:seq(1,20)]} || Index <- lists:seq(1,500)],
	Vali10o20=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Trni10o20,100)],
	Tsti10o20=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Vali10o20,100)],
	[ets:insert(i10o20,{Index,I,O}) || {Index,I,O}<-Trni10o20++Vali10o20++Tsti10o20],
	ets:insert(i10o20,{0,#info{
		name=i10o20,
		ivl=10,
		ovl=20,
		trn_end=500,
		val_end=600,
		tst_end=700
	}}),
	ets:tab2file(i10o20,i10o20),
	ets:delete(i10o20),
	%i50o20: IVL=50,OVL=20,
	I50O20 = ets:new(i50o20,[set,private,named_table]),
	Trni50o20=[{Index,[random:uniform(2)-1||_<-lists:seq(1,50)],[random:uniform(2)-1||_<-lists:seq(1,20)]} || Index <- lists:seq(1,500)],
	Vali50o20=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Trni50o20,100)],
	Tsti50o20=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Vali50o20,100)],
	[ets:insert(i50o20,{Index,I,O}) || {Index,I,O}<-Trni50o20++Vali50o20++Tsti50o20],
	ets:insert(i50o20,{0,#info{
		name=i50o20,
		ivl=50,
		ovl=20,
		trn_end=500,
		val_end=600,
		tst_end=700
	}}),
	ets:tab2file(i50o20,i50o20),
	ets:delete(i50o20),
	%i100o20: IVL=100,OVL=20,
	I100O20 = ets:new(i100o20,[set,private,named_table]),
	Trni100o20=[{Index,[random:uniform(2)-1||_<-lists:seq(1,100)],[random:uniform(2)-1||_<-lists:seq(1,20)]} || Index <- lists:seq(1,500)],
	Vali100o20=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Trni100o20,100)],
	Tsti100o20=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Vali100o20,100)],
	[ets:insert(i100o20,{Index,I,O}) || {Index,I,O}<-Trni100o20++Vali100o20++Tsti100o20],
	ets:insert(i100o20,{0,#info{
		name=i100o20,
		ivl=100,
		ovl=20,
		trn_end=500,
		val_end=600,
		tst_end=700
	}}),
	ets:tab2file(i100o20,i100o20),
	ets:delete(i100o20),
	%i100o50: IVL=100,OVL=50,
	I100O50 = ets:new(i100o50,[set,private,named_table]),
	Trni100o50=[{Index,[random:uniform()-0.5||_<-lists:seq(1,100)],[random:uniform()-0.5||_<-lists:seq(1,50)]} || Index <- lists:seq(1,500)],
	Vali100o50=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Trni100o50,100)],
	Tsti100o50=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Vali100o50,100)],
	[ets:insert(i100o50,{Index,I,O}) || {Index,I,O}<-Trni100o50++Vali100o50++Tsti100o50],
	ets:insert(i100o50,{0,#info{
		name=i100o50,
		ivl=100,
		ovl=50,
		trn_end=500,
		val_end=600,
		tst_end=700
	}}),
	ets:tab2file(i100o50,i100o50),
	ets:delete(i100o50),
	%i100o200: IVL=100,OVL=200,
	I100O200 = ets:new(i100o200,[set,private,named_table]),
	Trni100o200=[{Index,[random:uniform()-0.5||_<-lists:seq(1,100)],[random:uniform()-0.5||_<-lists:seq(1,200)]} || Index <- lists:seq(1,500)],
	Vali100o200=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Trni100o200,100)],
	Tsti100o200=[{Index+500,I,O}||{Index,I,O}<-lists:sublist(Vali100o200,100)],
	[ets:insert(i100o200,{Index,I,O}) || {Index,I,O}<-Trni100o200++Vali100o200++Tsti100o200],
	ets:insert(i100o200,{0,#info{
		name=i100o200,
		ivl=100,
		ovl=200,
		trn_end=500,
		val_end=600,
		tst_end=700
	}}),
	ets:tab2file(i100o200,i100o200),
	ets:delete(i100o200),
	%i200o100: IVL=200,OVL=100
	I200O100not_test = ets:new(i200o100not_test,[set,private,named_table]),
	Trni200o100not_test=[{Index,[random:uniform()-0.5||_<-lists:seq(1,200)],[random:uniform()-0.5||_<-lists:seq(1,100)]} || Index <- lists:seq(1,500)],
	[ets:insert(i200o100not_test,{Index,I,O}) || {Index,I,O}<-Trni200o100not_test],
	ets:insert(i200o100not_test,{0,#info{
		name=i200o100not_test,
		ivl=200,
		ovl=undefined,
		trn_end=500
	}}),
	ets:tab2file(i200o100not_test,i200o100not_test),
	ets:delete(i200o100not_test),
	%i200o100short: IVL=200,OVL=100
	I200O100short = ets:new(i200o100short,[set,private,named_table]),
	Trni200o100short=[{Index,[random:uniform()-0.5||_<-lists:seq(1,200)],[random:uniform(2)-1||_<-lists:seq(1,100)]} || Index <- lists:seq(1,10)],
	[ets:insert(i200o100short,{Index,I,O}) || {Index,I,O}<-Trni200o100short],
	ets:insert(i200o100short,{0,#info{
		name=i200o100short,
		ivl=200,
		ovl=100,
		trn_end=10
	}}),
	ets:tab2file(i200o100short,i200o100short),
	ets:delete(i200o100short),
	%xor_bip: IVL=2,OVL=1
	XOR = ets:new(xor_bip,[set,private,named_table]),
	Trnxor_bip=[{1,[-1,-1],[-1]},{2,[1,1],[-1]},{3,[-1,1],[1]},{4,[1,-1],[1]}],
	[ets:insert(xor_bip,{Index,I,O}) || {Index,I,O}<-Trnxor_bip],
	ets:insert(xor_bip,{0,#info{
		name=xor_bip,
		ivl=2,
		ovl=1,
		trn_end=4
	}}),
	ets:tab2file(xor_bip,xor_bip),
	ets:delete(xor_bip).
