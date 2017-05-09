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

-module(scape_LLVMPhaseOrdering).%General Time Series Analysis
-compile(export_all).
-define(OPTIMIZATION_LIST,[
	done,
	"adce",
	"always-inline",
	"argpromotion",
	"bb-vectorize",
	"break-crit-edges",
	"codegenprepare",
	"constmerge",
	"constprop",
	"dce",
	"deadargelim",
	"die",
	"dse",
	"functionattrs",
	"globaldce",
	"globalopt",
	"gvn",
	"indvars",
	"inline",
	"instcombine",
	"internalize",
	"ipconstprop",
	"ipsccp",
	"jump-threading",
	"lcssa",
	"licm",
	"loop-deletion",
	"loop-extract",
	"loop-extract-single",
	"loop-reduce",
	"loop-rotate",
	"loop-simplify",
	"loop-unroll",
	"loop-unswitch",
	"loweratomic",
	"lowerinvoke",
	"lowerswitch",
	"mem2reg",
	"memcpyopt",
	"mergefunc",
	"mergereturn",
	"partial-inliner",
	"prune-eh",
	"reassociate",
	"reg2mem",
	"scalarrepl",
	"sccp",
	"simplifycfg",
	"sink",
	"strip",
	"strip-dead-debug-info",
	"strip-dead-prototypes",
	"strip-debug-declare",
	"strip-nondebug",
	"tailcallelim"]).

tot_optimizations()->
	length(?OPTIMIZATION_LIST).

%%%%%%%%%%%%%%%%%%%%%%%%%%%% SCAPE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start(Exoself_PId)->
	loop(Exoself_PId).
	
loop(Exoself_PId)->
	receive
		{From,get_percept,get_BitCodeStatistics,[OpMode,ProgramName]}->
			Percept = try scape_LLVMPhaseOrdering:ProgramName(Exoself_PId,sense) of
				Result ->
					Result
				catch
					Error:Reason ->
						io:format("PERCEPT Error:~p Reason:~p~n",[Error,Reason]),
						io:format("Backtrace:~p~n",[erlang:get_stacktrace()]),
						io:format("Data:~p~n",[{From,get_percept,get_BitCodeStatistics,[OpMode,ProgramName]}]),
						exit("CRASHED~n")
			end,
			%io:format("Percept:~p~n",[{Percept,length(Percept)}]),
			From ! {self(),percept,Percept},
			?MODULE:loop(Exoself_PId);
		{From,act,choose_OptimizationPhase,[OpMode,ProgramName],Output}->
			Optimization = find_optimization(Output),
			{Fitness,HaltFlag} = try scape_LLVMPhaseOrdering:ProgramName(Exoself_PId,{act,Optimization}) of
				Result -> 
					Result
				catch
					Error:Reason ->
						io:format("ACT Error:~p Reason:~p~n",[Error,Reason]),
						io:format("Backtrace:~p~n",[erlang:get_stacktrace()]),
						io:format("Data:~p~n",[{From,get_percept,get_BitCodeStatistics,[OpMode,ProgramName]}]),
						exit("CRASHED~n")
			end,
			From ! {self(),Fitness,HaltFlag},
			case OpMode of
				test ->
					case get({Exoself_PId,opts}) of
						undefined ->
							put({Exoself_PId,opts},[Optimization]);
						Opts ->%io:format("Opts:~p~n",[Opts]),
							put({Exoself_PId,opts},[Optimization|Opts])
					end,
					case HaltFlag of
						1 ->	%io:format("Haltflag:~p~n",[{ProgramName,Fitness}]),
							[T] = Fitness,
							{ok, File_Output} = file:open("llvm_Data/"++atom_to_list(ProgramName)++float_to_list(T)++".txt", write),
							%io:format("File_Output:~p pwd():~p~n",[File_Output,c:pwd()]),
							[io:format(File_Output,"~p~n",[Opt]) || Opt<-lists:reverse(get({Exoself_PId,opts}))],
							file:close(File_Output),
							erase({Exoself_PId,opts});
						_ ->ok
					end;
				_ ->ok
			end,
			?MODULE:loop(Exoself_PId);
		terminate->
			ok
	end.
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%% SENSORS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%% ACTUATORS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALL_BACK_FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Bzip2 input.random 2 2>&1 | tee bzipA_inclusion.out
%Bzip2 input.random 2 2>&1 | tee bzipB_inclusion.out
%Bzip2 input.random 2 2>&1 | tee bzipB
%cccp 166 200
%Twolf ./ref.input/ref
%benchmarks/gcc/gcc/Gcc benchmarks/gcc/gcc/cccp.i -o benchmarks/gcc/gcc/cccp1.s
%Bzip2 input.source 5

%A set of commands to compile, run, recompile, move between directories, for various benchmarks dealing with compilation.
%This will expect that all the programs are in a particular file, benchmark_llvm, and will deal with particular benchmarks only, so each function will be for a certain one.

find_optimization(Output)->
	{MaxVal,MaxIndex} = find_MaxIndex(Output),
	lists:nth(MaxIndex,?OPTIMIZATION_LIST).
	
	find_MaxIndex([Val|Output])->
		find_MaxIndex(Output,2,Val,1).
	find_MaxIndex([Val|Output],Index,MaxVal,MaxIndex)->
		case Val > MaxVal of
			true ->
				find_MaxIndex(Output,Index+1,Val,Index);
			false ->
				find_MaxIndex(Output,Index+1,MaxVal,MaxIndex)
		end;
	find_MaxIndex([],_Index,MaxVal,MaxIndex)->
		{MaxVal,MaxIndex}.
	
gcc(Exoself_PId,sense)->
	[FileName|_]=case get(Exoself_PId) of
		undefined ->
			FunctionName = "gcc",
			Unique_Extension = integer_to_list(round(genotype:generate_UniqueId() * math:pow(10,25))),
			List = os:cmd("ls llvm_Data/"++FunctionName++"/"++FunctionName++"/src/*.c"),
			Reference_FileNames = [FileName ||[FileName,Extension]<-data_extractor:list_to_dvals([46,10],List,[])],%--"llvm_Data/gcc/gcc/src/reorg",
			io:format("List:~p~n",[Reference_FileNames]),
			%Reference_FileNames = ["word-file","xalloc","utilities","strncasecmp","read-dict","prune","print","post-process","parse","massage","main","idiom","fast-match","extract-links","build-disjuncts","and","analyze-linkage"],
			Personal_FileNames = [Reference_FileName++Unique_Extension || Reference_FileName<-Reference_FileNames],
			PhaseIndex=1,
			put(Exoself_PId,{Personal_FileNames,Personal_FileNames,PhaseIndex,Unique_Extension}),
			Result = [os:cmd("clang -emit-llvm -O0 -c "++FN++".c -o "++FN++Unique_Extension++".bc") || FN <- Reference_FileNames],
			io:format("Result:~p~n",[Result]),
			Personal_FileNames;
		{Remaining_File_Names,_All_Files,PhaseIndex,Unique_Extension} ->
			Remaining_File_Names
	end,
	%FeatureVector = get_FileFeatures(FileName),
	FeatureVector = [Val/30||Val<-lists:seq(1,30)],
	[1/PhaseIndex|FeatureVector];
gcc(Exoself_PId,{act,done})->
	FunctionName = "gcc",
	{_,All_Files,PhaseIndex,Unique_Extension} = get(Exoself_PId),
	Result1=[os:cmd("llc-3.4 "++FN++".bc") || FN<-All_Files],
	Result2=os:cmd("gcc " ++ lists:flatten([FN++".s "||FN<-All_Files]) ++ " -o llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension),
	Result3=[file:delete(FN++".bc") || FN <- All_Files],
	Result4=[file:delete(FN++".s") || FN <- All_Files],
	StartTime = now(),
	Result = os:cmd("llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension++" llvm_Data/"++FunctionName++"/"++FunctionName++"/src/cccp.i -o "++"llvm_Data/"++FunctionName++"/"++FunctionName++"/src/cccp1.s"),
	file:delete("llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension),
	EndTime = now(),
	RunTime = time_dif(StartTime,EndTime),
	erase(Exoself_PId),
	io:format("Results:~p~n",[{Result1,Result2,Result3,Result4,RunTime}]),
	{[1/RunTime],1};
gcc(Exoself_PId,{act,Optimization})->
	FunctionName = "gcc",
	case get(Exoself_PId) of
		{All_Files,All_Files,51,Unique_Extension}->
			parser(Exoself_PId,{act,done});
		{[FileName],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{All_Files,All_Files,PhaseIndex+1,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
			io:format("Result:~p~n",[Result]),
			{[0],0};
		{[FileName|Remaining_File_Names],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{Remaining_File_Names,All_Files,PhaseIndex,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
			io:format("Result:~p~n",[Result]),
			{[0],0};
		unefined ->
			exit("parser Actuator files not defined!~n")
	end.
	
perl()->
	ok.

vpr()->
	ok.
	
gap()->
	ok.

parser(Exoself_PId,sense)->
	[FileName|_]=case get(Exoself_PId) of
		undefined ->
			FunctionName = "parser",
			Unique_Extension = integer_to_list(round(genotype:generate_UniqueId() * math:pow(10,25))),
			List = os:cmd("ls llvm_Data/"++FunctionName++"/"++FunctionName++"/src/*.c"),
			Reference_FileNames = [FileName ||[FileName,Extension]<-data_extractor:list_to_dvals([46,10],List,[])],%--"llvm_Data/gcc/gcc/src/reorg",
			%io:format("List:~p~n",[Reference_FileNames]),
			%Reference_FileNames = ["word-file","xalloc","utilities","strncasecmp","read-dict","prune","print","post-process","parse","massage","main","idiom","fast-match","extract-links","build-disjuncts","and","analyze-linkage"],
			Personal_FileNames = [Reference_FileName++Unique_Extension || Reference_FileName<-Reference_FileNames],
			PhaseIndex=1,
			put(Exoself_PId,{Personal_FileNames,Personal_FileNames,PhaseIndex,Unique_Extension}),
			Result = [os:cmd("clang -emit-llvm -O0 -c "++FN++".c -o "++FN++Unique_Extension++".bc") || FN <- Reference_FileNames],
			%io:format("Result:~p~n",[Result]),
			Personal_FileNames;
		{Remaining_File_Names,_All_Files,PhaseIndex,Unique_Extension} ->
			Remaining_File_Names
	end,
	%FeatureVector = get_FileFeatures(FileName),
	FeatureVector = [Val/30||Val<-lists:seq(1,30)],
	[1/PhaseIndex|FeatureVector];
parser(Exoself_PId,{act,done})->
	FunctionName = "parser",
	{_,All_Files,PhaseIndex,Unique_Extension} = get(Exoself_PId),
	Result1=[os:cmd("llc-3.4 "++FN++".bc") || FN<-All_Files],
	Result2=os:cmd("gcc " ++ lists:flatten([FN++".s "||FN<-All_Files]) ++ " -o llvm_Data/"++FunctionName++"/"++FunctionName++"/"++FunctionName++Unique_Extension),
	Result3=[file:delete(FN++".bc") || FN <- All_Files],
	Result4=[file:delete(FN++".s") || FN <- All_Files],
	c:cd("llvm_Data/parser/parser"),
	StartTime = now(),
	Result = os:cmd("time ./parser"++Unique_Extension++" 2.1.dict -batch"),
	EndTime = now(),
	%io:format("~p~n",[data_extractor:list_to_dvals([$:,$:],Result,[])]),
	file:delete("llvm_Data/"++FunctionName++"/"++FunctionName++"/"++FunctionName++Unique_Extension),
	c:cd("../../.."),
	%io:format("Dir:~p~n",[c:pwd()]),
	RunTime = time_dif(StartTime,EndTime),
	%io:format("Result:~p~n",[Result]),
	erase(Exoself_PId),
	%io:format("Results:~p~n",[{Result1,Result2,Result3,Result4,RunTime}]),
	{[1/RunTime],1};
parser(Exoself_PId,{act,Optimization})->
	FunctionName = "parser",
	case get(Exoself_PId) of
		{All_Files,All_Files,51,Unique_Extension}->
			parser(Exoself_PId,{act,done});
		{[FileName],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{All_Files,All_Files,PhaseIndex+1,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
			io:format("Result:~p~n",[Result]),
			{[0],0};
		{[FileName|Remaining_File_Names],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{Remaining_File_Names,All_Files,PhaseIndex,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
			io:format("Result:~p~n",[Result]),
			{[0],0};
		unefined ->
			exit("parser Actuator files not defined!~n")
	end.

bzip2(Exoself_PId,sense)->
	[FileName|_]=case get(Exoself_PId) of
		undefined ->
			FunctionName = "bzip2",
			Unique_Extension = integer_to_list(round(genotype:generate_UniqueId() * math:pow(10,25))),
			List = os:cmd("ls llvm_Data/"++FunctionName++"/"++FunctionName++"/src/*.c"),
			Reference_FileNames = [FileName ||[FileName,Extension]<-data_extractor:list_to_dvals([46,10],List,[])],%--"llvm_Data/gcc/gcc/src/reorg",
			%io:format("List:~p~n",[Reference_FileNames]),
			Personal_FileNames = [Reference_FileName++Unique_Extension || Reference_FileName<-Reference_FileNames],
			PhaseIndex=1,
			put(Exoself_PId,{Personal_FileNames,Personal_FileNames,PhaseIndex,Unique_Extension}),
			Result = [os:cmd("clang -emit-llvm -O0 -c "++FN++".c -o "++FN++Unique_Extension++".bc") || FN <- Reference_FileNames],
			%io:format("Result:~p~n",[Result]),
			Personal_FileNames;
		{Remaining_File_Names,_All_Files,PhaseIndex,Unique_Extension} ->
			Remaining_File_Names
	end,
	%FeatureVector = get_FileFeatures(FileName),
	FeatureVector = [Val/30||Val<-lists:seq(1,30)],
	[1/PhaseIndex|FeatureVector];
bzip2(Exoself_PId,{act,done})->
	FunctionName = "bzip2",
	{_,All_Files,PhaseIndex,Unique_Extension} = get(Exoself_PId),
%	os:cmd("clang -o "++"llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension++" "++lists:flatten([FN++".bc "||FN<-All_Files])++" -lm"),
	Result1=[os:cmd("llc-3.4 "++FN++".bc") || FN<-All_Files],
	Result2=os:cmd("gcc " ++ lists:flatten([FN++".s "||FN<-All_Files]) ++ " -o llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension),
	Result3=[file:delete(FN++".bc") || FN <- All_Files],
	Result4=[file:delete(FN++".s") || FN <- All_Files],
	StartTime = now(),
	Result = os:cmd("time llvm_Data/bzip2/bzip2/src/bzip2"++Unique_Extension++" llvm_Data/bzip2/bzip2/input.random 2 2 > /dev/null"),
	%Result = os:cmd("llvm_Data/bzip2/bzip2/src/bzip2"++Unique_Extension++" llvm_Data/bzip2/bzip2/input.random 2 2>&1 | tee llvm_Data/bzip2/bzip2/bzipB"),
	%clang -o ExedcutableFileNameIWant ALL_BCs -lm
	EndTime = now(),%
	
	RunTime = case time_dif(StartTime,EndTime) < 0.10 of
		true ->
			io:format("Result:~p~n~n~n~p~n~n~n~p~n~n~n~p~n~n~n~p~n",[Result1,Result2,Result3,Result4,Result]),
			10;
		false ->
			%[[_,_,_,User,_,Sys],_]=data_extractor:list_to_dvals([58,32,32,$s,58,$e],Result,[]),
			[[User,_,System,_,Wall],_]=data_extractor:list_to_dvals([$u,32, $s, $:, $e],Result,[]),%[58,32,32,$s,58,$e]
			AltRunTime=User+Sys
			%time_dif(StartTime,EndTime)
	end,
	
	file:delete("llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension),
%	RunTime = time_dif(StartTime,EndTime),
	io:format("What:~p~n",[{Result,RunTime}]),
	%io:format("Result:~p User:~p Sys:~p~n",[Result,User,Sys]),
	%AltRunTime=User+Sys,
	erase(Exoself_PId),
%	io:format("Results:~p~n",[{Result1,Result2,Result3,Result4,RunTime}]),
	{[1/RunTime],1};
bzip2(Exoself_PId,{act,Optimization})->
	FunctionName = "bzip2",
	case get(Exoself_PId) of
		{All_Files,All_Files,51,Unique_Extension}->
			bzip2(Exoself_PId,{act,done});
		{[FileName],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{All_Files,All_Files,PhaseIndex+1,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
			%io:format("Result:~p~n",[Result]),
			{[0],0};
		{[FileName|Remaining_File_Names],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{Remaining_File_Names,All_Files,PhaseIndex,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
%			io:format("Result:~p~n",[Result]),
			{[0],0};
		unefined ->
			exit("parser Actuator files not defined!~n")
	end.

get_FileFeatures(FileName)->
	List=os:cmd("llvm_Data/neuropo --module-features "++FileName++".bc"),
	SplitVals = [32,10],
	Tockenized_List = data_extractor:list_to_dvals(SplitVals,List,[]),
	FeatureVector = [math:log(Val+math:sqrt(Val*Val+1))||[FeatureName,Val]<-Tockenized_List],%TODO: Segway scaling used on Val
	FeatureVector.	

ccbench(Exoself_PId,sense)->
	[FileName|_]=case get(Exoself_PId) of
		undefined ->
			FunctionName = "ccbench",
			Unique_Extension = integer_to_list(round(genotype:generate_UniqueId() * math:pow(10,25))),
			List = os:cmd("ls llvm_Data/"++FunctionName++"/src/*.c"),
			%List_bzlib = os:cmd("ls llvm_Data/"++FunctionName++"/src/bzlib/*.c"),
			Reference_FileNames = [FileName ||[FileName,Extension]<-data_extractor:list_to_dvals([46,10],List,[])],
			%io:format("List:~p~n",[Reference_FileNames]),
			Personal_FileNames = [Reference_FileName++Unique_Extension || Reference_FileName<-Reference_FileNames] ++ ["llvm_Data/ccbench/src/test_almabench"++Unique_Extension],
			PhaseIndex=1,
			put(Exoself_PId,{Personal_FileNames,Personal_FileNames,PhaseIndex,Unique_Extension}),
			Result_CC = os:cmd("clang++ -emit-llvm -o ccbench/test_almabench"++Unique_Extension++".bc -c ccbench/src/test_almabench.cc"),
			Result = [os:cmd("clang -emit-llvm -O0 -c "++FN++".c -o "++FN++Unique_Extension++".bc") || FN <- Reference_FileNames],
			%io:format("Result:~p~n",[Result]),
			Personal_FileNames;
		{Remaining_File_Names,_All_Files,PhaseIndex,Unique_Extension} ->
			Remaining_File_Names
	end,
	%FeatureVector = get_FileFeatures(FileName),
	FeatureVector = [Val/30||Val<-lists:seq(1,30)],
	[1/PhaseIndex|FeatureVector];
ccbench(Exoself_PId,{act,done})->
	FunctionName = "ccbench",
	{_,All_Files,PhaseIndex,Unique_Extension} = get(Exoself_PId),
%	os:cmd("clang -o "++"llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension++" "++lists:flatten([FN++".bc "||FN<-All_Files])++" -lm"),
	%Result1=[os:cmd("llc-3.4 "++FN++".bc") || FN<-All_Files],
	Result2=os:cmd("clang++ " ++ "-o llvm_Data/"++FunctionName++"/src/"++FunctionName++Unique_Extension++" "++lists:flatten([FN++".bc "||FN<-All_Files])++" -lm"),%clang++ -o ccbench *.bc bzlib/*.bc -lm
	Result3=[file:delete(FN++".bc") || FN <- All_Files],
	%Result4=[file:delete(FN++".s") || FN <- All_Files],
	StartTime = now(),
	Result = os:cmd("time llvm_Data/ccbench/src/ccbench"++Unique_Extension++" > /dev/null"),
	%clang -o ExedcutableFileNameIWant ALL_BCs -lm
	EndTime = now(),%
	
	RunTime = case time_dif(StartTime,EndTime) < 0.10 of
		true ->
			io:format("Result:~p~n~n~n~p~n~n~n~p~n~n~n~p~n~n~n~p~n",[Result1,Result2,Result3,Result4,Result]),
			10;
		false ->
			%[[_,_,_,User,_,Sys],_]=data_extractor:list_to_dvals([58,32,32,$s,58,$e],Result,[]),
			[[User,_,System,_,Wall],_]=data_extractor:list_to_dvals([$u,32, $s, $:, $e],Result,[]),%[58,32,32,$s,58,$e]
			AltRunTime=User+Sys
			%time_dif(StartTime,EndTime)
	end,
	
	file:delete("llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension),
%	RunTime = time_dif(StartTime,EndTime),
	io:format("What:~p~n",[{Result,RunTime}]),
	%io:format("Result:~p User:~p Sys:~p~n",[Result,User,Sys]),
	%AltRunTime=User+Sys,
	erase(Exoself_PId),
%	io:format("Results:~p~n",[{Result1,Result2,Result3,Result4,RunTime}]),
	{[1/RunTime],1};
ccbench(Exoself_PId,{act,Optimization})->
	FunctionName = "ccbench",
	case get(Exoself_PId) of
		{All_Files,All_Files,51,Unique_Extension}->
			bzip2(Exoself_PId,{act,done});
		{[FileName],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{All_Files,All_Files,PhaseIndex+1,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
			%io:format("Result:~p~n",[Result]),
			{[0],0};
		{[FileName|Remaining_File_Names],All_Files,PhaseIndex,Unique_Extension}->
			put(Exoself_PId,{Remaining_File_Names,All_Files,PhaseIndex,Unique_Extension}),
			Result = os:cmd("opt-3.4 -"++Optimization++" -o "++FileName++".bc"++" "++FileName++".bc"),
%			io:format("Result:~p~n",[Result]),
			{[0],0};
		unefined ->
			exit("parser Actuator files not defined!~n")
	end.

time_dif(StartTime,EndTime)->
	{StartA,StartS,StartMS} = StartTime,
	{EndA,EndS,EndMS} = EndTime,
	DifA = EndA-StartA,
	case EndA > StartA of
		true ->
			DifS = EndS-StartS,
			DifMS= EndMS-StartMS;
		false->
			DifS = EndS-StartS-1,
			DifMS= EndMS+1000000-StartMS
	end,
	DifS+DifMS/1000000.
	
test_Opts()->
	bzip2(self(),sense),
	FunctionName = "bzip2",
	[_|Opts] = ?OPTIMIZATION_LIST,
	{_,All_Files,_,Unique_Extension}=get(self()),io:format("1~p~n",[All_Files]),
	Result = [{after_each(Opt,FN),Opt,FN} || Opt<-Opts,FN<-All_Files],% || _<-lists:seq(1,10)],io:format("2~n"),
	erase(self()),
	Result3=[file:delete(FN++".bc") || FN <- All_Files],io:format("5~n"),
	Result4=[file:delete(FN++".s") || FN <- All_Files],io:format("6~n"),
	file:delete("llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension),
	io:format("~p~n",[Result]).
	
	after_each(Opt,FN)->
		FunctionName = "bzip2",
		{_,All_Files,_,Unique_Extension}=get(self()),io:format("1~p~n",[All_Files]),
		os:cmd("opt-3.4 -verify -"++Opt++" -o "++FN++".bc"++" "++FN++".bc"),
		Result1=[os:cmd("llc-3.4 "++FN++".bc") || FN<-All_Files],io:format("3~n"),
		Result2=os:cmd("gcc " ++ lists:flatten([FN++".s "||FN<-All_Files]) ++ " -o llvm_Data/"++FunctionName++"/"++FunctionName++"/src/"++FunctionName++Unique_Extension),io:format("4~n"),
		StartTime = now(),
		Result5 = os:cmd("llvm_Data/bzip2/bzip2/src/bzip2"++Unique_Extension++" llvm_Data/bzip2/bzip2/input.random 2 2>&1 | tee llvm_Data/bzip2/bzip2/bzipB"),
		%Result = os:cmd("llvm_Data/bzip2/bzip2/src/bzip2"++Unique_Extension++" llvm_Data/bzip2/bzip2/input.random 2 2>&1 | tee llvm_Data/bzip2/bzip2/bzipB"),
		%clang -o ExedcutableFileNameIWant ALL_BCs -lm
		EndTime = now(),%
		%[[_,_,_,User,_,Sys],_]=data_extractor:list_to_dvals([58,32,32,$s,58,$e],Result,[]),
		RunTime = time_dif(StartTime,EndTime),
		%io:format("Result:~p User:~p Sys:~p~n",[Result,User,Sys]),
		%AltRunTime=User+Sys,
		io:format("~p~n",[{Result5,Result2,Opt,FN}]).
