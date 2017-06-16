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

-module(benchmarker).
-compile(export_all).
-include("records.hrl").
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Benchmark Options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-define(DIR,"benchmarks/").
-define(INIT_CONSTRAINTS,[
	#constraint{
		morphology=Morphology,
		connection_architecture=CA,
		population_selection_f=hof_competition,
		population_evo_alg_f=generational,
		neural_pfns=[none],
		agent_encoding_types=[neural],
		%neural_afs=[{circuit,{micro,[#layer{neurode_type=tanh,tot_neurodes=2,dynamics=dynamic},#layer{neurode_type=tanh,tot_neurodes=1,dynamics=static}]}}],
		neural_afs=[tanh],
%		neural_afs = [
%			{circuit,#layer{neurode_type=tanh,tot_neurodes=10,dynamics=dynamic,type=dae}},
%			{circuit,#layer{neurode_type=sigmoid,tot_neurodes=10,dynamics=dynamic,type=dae}},
%			{circuit,#layer{neurode_type=all,tot_neurodes=10,dynamics=dynamic,type=dae}},
%			{circuit,#layer{neurode_type=sin,tot_neurodes=10,dynamics=dynamic,type=dae}}
%		],
%		neural_afs = [{circuit,#layer{neurode_type=tanh,tot_neurodes=1,type=standard}}],
%		neural_afs = [{circuit,#layer{neurode_type=tanh,tot_neurodes=10,dynamics=dynamic,type=dae}}],
		tuning_selection_fs=[dynamic_random],
		mutation_operators= [
			%{mutate_weights,10000},
			{add_bias,10},
			%{remove_bias,1},
	%		{mutate_af,1},
			{add_outlink,40},
			{add_inlink,40},
			{add_neuron,40},
			{outsplice,40},
			%{insplice,40},
			{add_sensorlink,1},
			%{add_actuatorlink,1},
			{add_sensor,1},
			{add_actuator,1},
	%		{mutate_plasticity_parameters,1},
			{add_cpp,1},
			{add_cep,1}
		]
	}
	||
		Morphology<-[xorAndXor],
		CA<-[recurrent]

]).
%neural_types=[{circuit,{static,[{tanh,2,static},{tanh,1,static}]}}] %[{circuit,{static|dynamic,[{NeuronType::tanh|sin|rbf|gaussian|gabor_2d, LayerSize::integer(), static|dynamic}]}} | standard]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print_experiment(Experiment_Id)->
	io:format("********~n~p~n*******",[mnesia:dirty_read({experiment,Experiment_Id})]).

get_ekeys()->
	io:format("--- Currently Stored Experiments ---~n"),
	get_ekeys(mnesia:dirty_first(experiment)).

	get_ekeys('$end_of_table')->
		ok;
	get_ekeys(Key)->
		[E]=mnesia:dirty_read({experiment,Key}),
		io:format("*~p~n Notes: ~p~n",[Key,E#experiment.notes]),
		get_ekeys(mnesia:dirty_next(experiment,Key)).

%Starts and ends Neural Networks with various preset parameters and options, and polls the logger for information about each run.
start(Id)->
	start(Id,"No comment").
start(Id,Notes)->
	PMP = #pmp{
		op_mode=[gt,validation,test],
		population_id=test,
		survival_percentage=0.5,
		specie_size_limit=10,
		init_specie_size=10,
		polis_id = mathema,
		generation_limit = 10000,
		evaluations_limit = 100000,
		fitness_goal = inf
	},
	E=#experiment{
		id = Id,
		backup_flag = true,
		pm_parameters=PMP,
		init_constraints=?INIT_CONSTRAINTS,
		progress_flag=in_progress,
		run_index=1,
		tot_runs=100,
		started={date(),time()},
		interruptions=[]
	},
	genotype:write(E),
	register(benchmarker,spawn(benchmarker,prep,[E])).
start(Id, Pmp, Constraints, Runs) when is_integer(Runs) and is_atom(Id)->
	Pmp_ = map2rec:convert(pmp, Pmp),
	Con_ = lists:foldl(fun(C,Acc)-> [map2rec:convert(constraint, C)|Acc] end, [], Constraints),
	E=#experiment{
        id = Id,
        backup_flag = true,
        pm_parameters=Pmp_,
        init_constraints=Con_,
        progress_flag=in_progress,
        run_index=1,
        tot_runs=Runs,
        started={date(),time()},
        interruptions=[]
    },
    genotype:write(E),
    register(benchmarker,spawn(benchmarker,prep,[E])).

continue(Id)->
	case genotype:dirty_read({experiment,Id}) of
		undefined ->
			io:format("Can't continue experiment:~p, it's not present in the database.~n",[Id]);
		E ->
			case E#experiment.progress_flag of
				completed ->
					io:format("Experiment:~p already completed:~p~n",[Id,E#experiment.trace_acc]);
				in_progress ->
					Interruptions = E#experiment.interruptions,
					U_Interruptions = [now()|Interruptions],
					U_E = E#experiment{
						interruptions = U_Interruptions
					},
					genotype:write(U_E),
					register(benchmarker,spawn(benchmarker,prep,[U_E]))
			end
	end.

prep(E)->
	PMP = E#experiment.pm_parameters,
	U_PMP = PMP#pmp{benchmarker_pid=self()},
	Constraints = E#experiment.init_constraints,
	Population_Id = PMP#pmp.population_id,
	population_monitor:prep_PopState(U_PMP,Constraints),
	loop(E#experiment{pm_parameters=U_PMP},Population_Id).

loop(E,P_Id)->
	receive
		{P_Id,completed,Trace}->
			U_TraceAcc = [Trace|E#experiment.trace_acc],
			U_RunIndex = E#experiment.run_index+1,
			case U_RunIndex > E#experiment.tot_runs of
				true ->
					U_E = E#experiment{
						trace_acc = U_TraceAcc,
						run_index = U_RunIndex,
						completed = {date(),time()},
						progress_flag = completed
					},
					genotype:write(U_E),
					report(U_E#experiment.id,"report"),
					case lists:member(test,(E#experiment.pm_parameters)#pmp.op_mode) of
						true ->
							io:format("E:~p~n",[U_E]),
							Traces = E#experiment.trace_acc,
							BestGen_Champions = lists:reverse(lists:sort([get_best(Trace) || Trace <- Traces])),
							io:format("Validation Champions {ValFitness,TestFitness}:~p~n",[BestGen_Champions]),
							%[{BOTB_F,BOTB_Id}|_] = lists:reverse(lists:sort(BestGen_Champions)),
							%io:format("BOTB:~p~n",[{BOTB_F,BOTB_Id}]),
%							BestGen_PIdPs=[{exoself:start(ExoselfId,self(),test),ExoselfId} || {GenFitness,ExoselfId} <- BestGen_Champions],
%							BestGen_Results=receive_TestAcks(BestGen_PIdPs,[]),
%							BestGen_Avg = get_avg(BestGen_Results,[]),
%							io:format("Test results of Validation Champions:~p~n",[lists:keyfind(BOTB_Id,1,BestGen_Results)]),
							io:format("************************~n");
						false ->
							ok
					end;
				false ->
					U_E = E#experiment{
						trace_acc = U_TraceAcc,
						run_index = U_RunIndex
					},
					genotype:write(U_E),
					PMP = U_E#experiment.pm_parameters,
					Constraints = U_E#experiment.init_constraints,
					population_monitor:prep_PopState(PMP,Constraints),
					io:format("****Experiment:~p/~p completed.****~n",[U_RunIndex,E#experiment.tot_runs]),
					loop(U_E,P_Id)
			end;
		terminate ->
			ok
	end.

	receive_TestAcks([{PId,Id}|PIdPs],Acc)->
		receive
			{PId,test_complete,Id,Fitness,Cycles,Time} ->
				receive_TestAcks(PIdPs,[{Id,Fitness}|Acc])
		end;
	receive_TestAcks([],Acc)->
		Acc.


	get_best(T)->
		Stats = T#trace.stats,
		GenTest_Champions=[{Stat#stat.validation_fitness,Stat#stat.test_fitness} || [Stat] <- Stats],
		[Best|_]=lists:reverse(lists:sort(GenTest_Champions)),
		Best.

%	get_avg([{Id,FitnessP}|IdPs],Acc)->
%		get_avg(IdPs,[FitnessP|Acc]);
%	get_avg([],Acc)->
%		[io:format("~p~n",[{functions:avg(Fitness),functions:std(Fitness),lists:max(Fitness),lists:min(Fitness)}]) || Fitness <- Acc].
	get_avg([{Id,FitnessP}|IdPs],Acc)->
		get_avg(IdPs,[FitnessP|Acc]);
	get_avg([],Acc)->
		get_avg(Acc,[],[],[]).

		get_avg([Fitness|FitnessPs],Acc1,Acc2,Acc3)->
			%io:format("Fitness:~p~n",[{Fitness,FitnessPs}]),
			case Fitness of
				[Score|Scores] ->
			%		io:format("Score/Scores:~p~n",[{Score,Scores}]),
					get_avg(FitnessPs,[Score|Acc1],[Scores|Acc2],Acc3);
				[] ->
					get_avg(FitnessPs,Acc1,Acc2,Acc3)
			end;
		get_avg([],[],[],Acc3)->
			io:format("Top validation score based agent's test fitness:~n"),
			[io:format("~p~n",[{functions:avg(Score),functions:std(Score),lists:max(Score),lists:min(Score)}]) || Score <- lists:reverse(Acc3)];
		get_avg([],Acc1,Acc2,Acc3)->
			%io:format("Acc1:~p~n",[Acc1]),
			get_avg(Acc2,[],[],[Acc1|Acc3]).


report(Experiment_Id,FileName)->
	report(Experiment_Id,FileName,undefind).
report(Experiment_Id,FileName,EvalLimit)->
	E = genotype:dirty_read({experiment,Experiment_Id}),

	{ok, EFile} = file:open(?DIR++FileName++"_Experiment", write),
	io:format(EFile, "~p",[E]),
	file:close(EFile),
	io:format("******** Experiment written to file:~p~n",[?DIR++FileName++"_Experiment"]),

	Traces = E#experiment.trace_acc,
	{ok, File} = file:open(?DIR++FileName++"_Trace_Acc", write),
	Evaluations_Stats = get_evaluations(Experiment_Id,undefined,EvalLimit),
	io:format(File,"REPORT:~p~n",[Evaluations_Stats]),
	lists:foreach(fun(X) -> io:format(File, "~p.~n",[X]) end, Traces),
	file:close(File),
	io:format("******** Traces_Acc written to file:~p~n",[?DIR++FileName++"_Trace_Acc"]),

	Graphs = prepare_Graphs(Traces),
	write_Graphs(Graphs,FileName++"_Graphs"),
	Eval_List = [T#trace.tot_evaluations||T<-Traces],
	io:format("Tot Evaluations Avg:~p Std:~p~n",[functions:avg(Eval_List),functions:std(Eval_List)]),
	Evaluations_Stats,
	get_evaluations(Experiment_Id,undefined,EvalLimit).

-record(graph,{morphology,avg_neurons=[],neurons_std=[],avg_fitness=[],fitness_std=[],max_fitness=[],min_fitness=[],maxavg_fitness=[],maxavg_fitness_std=[],minavg_fitness=[],avg_diversity=[],diversity_std=[],evaluations=[],validation_fitness=[],validation_fitness_std=[],validationmax_fitness=[],validationmin_fitness=[],evaluation_Index=[]}).
-record(avg,{avg_neurons=[],neurons_std=[],avg_fitness=[],fitness_std=[],max_fitness=[],min_fitness=[],maxavg_fitness,maxavg_fitness_std=[],minavg_fitness,avg_diversity=[],diversity_std=[],evaluations=[],validation_fitness=[],validation_fitness_std=[],validationmax_fitness=[],validationmin_fitness=[]}).
%-record(stat,{avg_subcores,subcores_std,avg_neurons,neurons_std,avg_fitness,fitness_std,max_fitness,min_fitness,avg_diversity,evaluations,time_stamp}).

prepare_Graphs(Traces)->
%Each trace is composed of a list of lists of stats. Length of list of stats determines the number of species.... we need to graph that so that we can graph the features against evaluations.
%1. seperate into Traces
%2. Seperate Traces into stats
%3. Extract from each stats the various features against evaluations
%4. Combine the whatever from all stats from all traces into the averages.
	[T|_] = Traces,
	[Stats_List|_] = T#trace.stats,
	Morphologies = [S#stat.morphology || S<-Stats_List],
	Morphology_Graphs = [prep_Traces(Traces,Morphology,[])|| Morphology <- Morphologies],
	[io:format("Graph:~p~n",[Graph])|| Graph<-Morphology_Graphs],
	Morphology_Graphs.

	prep_Traces([T|Traces],Morphology,Acc)->
		Morphology_Trace = lists:flatten([[S||S<-Stats,S#stat.morphology == Morphology]||Stats<-T#trace.stats]),
		prep_Traces(Traces,Morphology,[Morphology_Trace|Acc]);
	prep_Traces([],Morphology,Acc)->
		Graph = avg_MorphologicalTraces(lists:reverse(Acc),[],[],[]),
		Graph#graph{morphology=Morphology}.

		avg_MorphologicalTraces([S_List|S_Lists],Acc1,Acc2,Acc3)->
			case S_List of
				[S|STail] ->
					avg_MorphologicalTraces(S_Lists,[STail|Acc1],[S|Acc2],Acc3);
				[] ->
					Graph = avg_statslists(Acc3,#graph{}),
					Graph
			end;
		avg_MorphologicalTraces([],Acc1,Acc2,Acc3)->
			avg_MorphologicalTraces(lists:reverse(Acc1),[],[],[lists:reverse(Acc2)|Acc3]).

			avg_statslists([S_List|S_Lists],Graph)->
				Avg = avg_stats(S_List,#avg{}),
				U_Graph = Graph#graph{
					avg_neurons = [Avg#avg.avg_neurons|Graph#graph.avg_neurons],
					neurons_std = [Avg#avg.neurons_std|Graph#graph.neurons_std],
					avg_fitness = [Avg#avg.avg_fitness|Graph#graph.avg_fitness],
					fitness_std = [Avg#avg.fitness_std|Graph#graph.fitness_std],
					max_fitness = [Avg#avg.max_fitness|Graph#graph.max_fitness],
					min_fitness = [Avg#avg.min_fitness|Graph#graph.min_fitness],
					maxavg_fitness = [Avg#avg.maxavg_fitness|Graph#graph.maxavg_fitness],
					maxavg_fitness_std = [Avg#avg.maxavg_fitness_std|Graph#graph.maxavg_fitness_std],
					minavg_fitness = [Avg#avg.minavg_fitness|Graph#graph.minavg_fitness],
					evaluations = [Avg#avg.evaluations|Graph#graph.evaluations],
					validation_fitness = [Avg#avg.validation_fitness|Graph#graph.validation_fitness],
					validation_fitness_std = [Avg#avg.validation_fitness_std|Graph#graph.validation_fitness_std],
					validationmax_fitness = [Avg#avg.validationmax_fitness|Graph#graph.validationmax_fitness],
					validationmin_fitness = [Avg#avg.validationmin_fitness|Graph#graph.validationmin_fitness],
					avg_diversity = [Avg#avg.avg_diversity|Graph#graph.avg_diversity],
					diversity_std = [Avg#avg.diversity_std|Graph#graph.diversity_std]
				},
				avg_statslists(S_Lists,U_Graph);
			avg_statslists([],Graph)->
				io:format("Validation Fitness:~p~n",[lists:reverse(Graph#graph.validation_fitness)]),
				Graph#graph{
					avg_neurons = lists:reverse(Graph#graph.avg_neurons),
					neurons_std = lists:reverse(Graph#graph.neurons_std),
					avg_fitness = lists:reverse(Graph#graph.avg_fitness),
					fitness_std = lists:reverse(Graph#graph.fitness_std),
					max_fitness = lists:reverse(Graph#graph.max_fitness),
					min_fitness = lists:reverse(Graph#graph.min_fitness),
					maxavg_fitness = lists:reverse(Graph#graph.maxavg_fitness),
					maxavg_fitness_std = lists:reverse(Graph#graph.maxavg_fitness_std),
					minavg_fitness = lists:reverse(Graph#graph.minavg_fitness),
					evaluations = lists:reverse(Graph#graph.evaluations),
					validation_fitness = lists:reverse(Graph#graph.validation_fitness),
					validation_fitness_std = lists:reverse(Graph#graph.validation_fitness_std),
					validationmax_fitness = lists:reverse(Graph#graph.validationmax_fitness),
					validationmin_fitness = lists:reverse(Graph#graph.validationmin_fitness),
					avg_diversity = lists:reverse(Graph#graph.avg_diversity),
					diversity_std = lists:reverse(Graph#graph.diversity_std)
				}.

				avg_stats([S|STail],Avg)->
					%io:format("S:~p~n",[S]),
					io:format("AVG_STATS:~p~n",[S#stat.validation_fitness]),
					{Validation_Fitness,ChampionId} = S#stat.validation_fitness,
					%io:format("Here1:~p~n",[{Validation_Fitness,Avg#avg.validation_fitness}]),
					%io:format("Here2:~p~n",[list_append(Validation_Fitness,Avg#avg.validation_fitness)]),
					U_Avg = Avg#avg{
						avg_neurons = [S#stat.avg_neurons|Avg#avg.avg_neurons],
						%neurons_std = [S#stat.neurons_std|Avg#avg.neurons_std],
						avg_fitness = list_append(S#stat.avg_fitness,Avg#avg.avg_fitness),
						%fitness_std = list_append(S#stat.fitness_std,Avg#avg.fitness_std),
						max_fitness = list_append(S#stat.max_fitness,Avg#avg.max_fitness),
						min_fitness = list_append(S#stat.min_fitness,Avg#avg.min_fitness),
						evaluations = [S#stat.evaluations|Avg#avg.evaluations],
						validation_fitness = list_append(Validation_Fitness,Avg#avg.validation_fitness),
						avg_diversity = [S#stat.avg_diversity|Avg#avg.avg_diversity]
					},
					avg_stats(STail,U_Avg);
				avg_stats([],Avg)->
					Avg#avg{
						avg_neurons=functions:avg(Avg#avg.avg_neurons),
						neurons_std=functions:std(Avg#avg.avg_neurons),
						avg_fitness=[functions:avg(Val)||Val<-Avg#avg.avg_fitness],
						fitness_std=[functions:std(Val)||Val<-Avg#avg.avg_fitness],
						max_fitness=[lists:max(Val)||Val<-Avg#avg.max_fitness],
						min_fitness=[lists:min(Val)||Val<-Avg#avg.min_fitness],
						maxavg_fitness=[functions:avg(Val)||Val<-Avg#avg.max_fitness],
						maxavg_fitness_std=[functions:std(Val)||Val<-Avg#avg.max_fitness],
						minavg_fitness=[functions:avg(Val)||Val<-Avg#avg.min_fitness],
						evaluations=functions:avg(Avg#avg.evaluations),
						validation_fitness=[functions:avg(Val)||Val<-Avg#avg.validation_fitness],
						validation_fitness_std=[functions:std(Val)||Val<-Avg#avg.validation_fitness],
						validationmax_fitness=[lists:max(Val)||Val<-Avg#avg.validation_fitness],
						validationmin_fitness=[lists:min(Val)||Val<-Avg#avg.validation_fitness],
						avg_diversity=functions:avg(Avg#avg.avg_diversity),
						diversity_std=functions:std(Avg#avg.avg_diversity)
					}.

			list_append([],[])->
				[];
%			list_append(0,[])->
%				[];
%			list_append(0,ListB)->
%				ListB;
			list_append(ListA,[])->
				[[Val]||Val<-ListA];
			list_append([],ListB)->
				ListB;
			list_append(ListA,ListB)->
				list_append(ListA,ListB,[]).
			list_append([Val|ListA],[AccB|ListB],Acc)->
				list_append(ListA,ListB,[[Val|AccB]|Acc]);
			list_append([],[],Acc)->
				%io:format("Acc:~p~n",[Acc]),
				lists:reverse(Acc).

write_Graphs([G|Graphs],Graph_Postfix)->
	Morphology = G#graph.morphology,
	U_G = G#graph{evaluation_Index=[500*Index || Index <-lists:seq(1,length(G#graph.avg_fitness))]},
	File = case Morphology of
		{M,F}->
			{ok, File} = file:open(?DIR++"graph_"++atom_to_list(M)++"_"++atom_to_list(F)++"_"++Graph_Postfix, write),
			File;
		_->
			{ok, File} = file:open(?DIR++"graph_"++atom_to_list(Morphology)++"_"++Graph_Postfix, write),
			File
	end,

	io:format(File,"#Avg Fitness Vs Evaluations, Morphology:~p",[Morphology]),
	print_MultiObjectiveFitness(File,U_G#graph.evaluation_Index,U_G#graph.avg_fitness,U_G#graph.fitness_std),

	io:format(File,"~n~n~n#Avg Neurons Vs Evaluations, Morphology:~p~n",[Morphology]),
	lists:foreach(fun({X,Y,Std}) -> io:format(File, "~p ~p ~p~n",[X,Y,Std]) end, lists:zip3(U_G#graph.evaluation_Index,U_G#graph.avg_neurons,U_G#graph.neurons_std)),

	io:format(File,"~n~n#Avg Diversity Vs Evaluations, Morphology:~p~n",[Morphology]),
	lists:foreach(fun({X,Y,Std}) -> io:format(File, "~p ~p ~p~n",[X,Y,Std]) end, lists:zip3(U_G#graph.evaluation_Index,U_G#graph.avg_diversity,U_G#graph.diversity_std)),

	io:format(File,"~n~n# Max Fitness Vs Evaluations, Morphology:~p",[Morphology]),
	print_MultiObjectiveFitness(File,U_G#graph.evaluation_Index,U_G#graph.max_fitness),

	io:format(File,"~n~n~n#Avg. Max Fitness Vs Evaluations, Morphology:~p",[Morphology]),
	print_MultiObjectiveFitness(File,U_G#graph.evaluation_Index,U_G#graph.maxavg_fitness),

	io:format(File,"~n~n~n#Avg. Min Fitness Vs Evaluations, Morphology:~p",[Morphology]),
	print_MultiObjectiveFitness(File,U_G#graph.evaluation_Index,U_G#graph.min_fitness),

	io:format(File,"~n~n~n#Specie-Population Turnover Vs Evaluations, Morphology:~p~n",[Morphology]),
	lists:foreach(fun({X,Y}) -> io:format(File, "~p ~p~n",[X,Y]) end, lists:zip(U_G#graph.evaluation_Index,U_G#graph.evaluations)),

	io:format(File,"~n~n#Validation Avg Fitness Vs Evaluations, Morphology:~p",[Morphology]),
	print_MultiObjectiveFitness(File,U_G#graph.evaluation_Index,U_G#graph.validation_fitness,U_G#graph.validation_fitness_std),

	io:format(File,"~n~n~n#Validation Max Fitness Vs Evaluations, Morphology:~p",[Morphology]),
	print_MultiObjectiveFitness(File,U_G#graph.evaluation_Index,U_G#graph.validationmax_fitness),

	io:format(File,"~n~n~n#Validation Min Fitness Vs Evaluations, Morphology:~p",[Morphology]),
	print_MultiObjectiveFitness(File,U_G#graph.evaluation_Index,U_G#graph.validationmin_fitness),

	file:close(File),
	write_Graphs(Graphs,Graph_Postfix);
write_Graphs([],_Graph_Postfix)->
	ok.

	print_MultiObjectiveFitness(File,[I|Index],[F|Fitness],[Std|StandardDiviation])->
		case (F==[]) or (Std==[]) of
			true ->
				ok;
			false ->
				io:format(File,"~n~p",[I]),
				print_FitnessAndStd(File,F,Std)
		end,
		print_MultiObjectiveFitness(File,Index,Fitness,StandardDiviation);
	print_MultiObjectiveFitness(_File,[],[],[])->
		ok.

		print_FitnessAndStd(File,[FE|FitnessElements],[SE|StdElements])->
			io:format(File," ~p ~p",[FE,SE]),
			print_FitnessAndStd(File,FitnessElements,StdElements);
		print_FitnessAndStd(_File,[],[])->
			ok.

	print_MultiObjectiveFitness(File,[I|Index],[F|Fitness])->
		case F == [] of
			true ->
				ok;
			false ->
				io:format(File,"~n~p",[I]),
				[io:format(File," ~p",[FE])||FE<-F]
		end,
		print_MultiObjectiveFitness(File,Index,Fitness);
	print_MultiObjectiveFitness(_File,[],[])->
		ok.

unconsult(List)->
	{ok, File} = file:open(?DIR++"alife_benchmark", write),
	lists:foreach(fun(X) -> io:format(File, "~p~n",[X]) end, List),
	file:close(File).

gen_plot(Lists)->gen_plot(Lists,[],[],[]).

gen_plot([List|Lists],Acc1,Acc2,Acc3)->
	case List of
		[Val|Rem] ->
			gen_plot(Lists,[Rem|Acc1],[Val|Acc2],Acc3);
		[] ->
			print_plot(500,Acc3)
	end;
gen_plot([],Acc1,Acc2,Acc3)->
	gen_plot(Acc1,[],[],[functions:avg(Acc2)|Acc3]).

	genplot(Lists)->genplot(Lists,[]).
	genplot([L|Lists],Acc)->
		genplot(Lists,[lists:max(L)|Acc]);
	genplot([],Acc)->
		print_plot(0,lists:reverse(Acc)).

	print_plot(Index,[Val|List])->
		io:format("~p  ~p~n",[Index,Val]),
		print_plot(Index+500,List);
	print_plot(_Index,[])->
		void.

trace2graph(TraceFileName)->
	{ok,Traces} = file:consult(TraceFileName),
	io:format("Traces:~p~n",[Traces]),
	Graphs = prepare_Graphs(Traces),
	write_Graphs(Graphs,"__Graph").

get_evaluations(E_Id)->
	get_evaluations(E_Id,undefined,undefined).
get_evaluations(E_Id,FitnessGoal)->
	get_evaluations(E_Id,FitnessGoal,undefined).
get_evaluations(E_Id,FitnessGoal,EvalLimit)->
	[E] = mnesia:dirty_read({experiment,E_Id}),
	Trace_Acc =E#experiment.trace_acc,
	EvaluationAcc=[analyze_stats(lists:reverse(T#trace.stats),FitnessGoal,EvalLimit,0)|| T<-Trace_Acc],
	TotEvoRuns = length(EvaluationAcc),
	SuccessAcc = [E|| E<-EvaluationAcc,E=/=undefined],
	TotSuccess = length(SuccessAcc),
	SuccessRate = TotSuccess/TotEvoRuns,
	io:format("SuccessAcc:~p~n",[SuccessAcc]),
	Avg = case SuccessAcc of
		[] ->
			undefined;
		SuccessAcc ->
			functions:avg(SuccessAcc)
	end,
	Std = functions:std(SuccessAcc,Avg,[]),
	io:format("Success Rate:{~p/~p,~p%} Avg:~p Std:~p~n",[TotSuccess,TotEvoRuns,SuccessRate*100,Avg,Std]).

		analyze_stats([[S]|Stats],FitnessGoal,EvalLimit,Acc)->
			io:format("S:~p~n",[S]),
			U_Acc = Acc+S#stat.evaluations,
			case S#stat.max_fitness of
				[] ->
					analyze_stats(Stats,FitnessGoal,EvalLimit,U_Acc);
				[MaxFitness|_]->
					case MaxFitness >= FitnessGoal of
						true ->
							U_Acc;
						false ->
							case U_Acc > EvalLimit of
								true ->
									undefined;
								false ->
									analyze_stats(Stats,FitnessGoal,EvalLimit,U_Acc)
							end
					end
			end;
%analyze_stats(Stats,U_Acc);
		analyze_stats([],FitnessGoal,EvalLimit,Acc)->
			case FitnessGoal of
				undefined ->
					Acc;
				_ ->
					undefined
			end.

chg_mrph(Id,NewMorph)->
	[A] = mnesia:dirty_read({agent,Id}),
	mnesia:dirty_write((A#agent.constraint)#constraint{morphology=NewMorph}),
	io:format("Ok: OldMorph:~p NewMorph:~p~n",[(A#agent.constraint)#constraint.morphology,NewMorph]).

vector_gt(V1,V2)->
	vector_gt(V1,V2,0).
vector_gt([A|V1],[B|V2],Acc)->
	case A >= B of
		true ->
			vector_gt(V1,V2,(A-B)+Acc);
		false ->
			false
	end;
vector_gt([],[],Acc)->
	Acc > 0;%If false then V1==V2, else V1 is at least as large as V2, and superior in some vector element.
vector_gt(_,undefined,_Acc)->
	false.

vector_lt(V1,V2)->
	vector_lt(V1,V2,0).
vector_lt([A|V1],[B|V2],Acc)->
	case A =< B of
		true ->
			vector_lt(V1,V2,(A-B)+Acc);
		false ->
			false
	end;
vector_lt([],[],Acc)->
	Acc<0;
vector_lt(_,undefined,_Acc)->
	false.

vector_eq([A|V1],[B|V2])->
	case A == B of
		true ->
			vector_eq(V1,V2);
		false ->
			false
	end;
vector_eq([],[])->
	true;
vector_eq(_,undefined)->
	false.
