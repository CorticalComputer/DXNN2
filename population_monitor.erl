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
-module(population_monitor).
-include("records.hrl").
%% API
-export([start_link/1,start_link/0,start/1,start/0,stop/0,init/2]).
%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3,test/0, create_specie/3, continue/0, continue/1, init_population/2, prep_PopState/2, extract_AgentIds/2,delete_population/1]).
-behaviour(gen_server).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Population Monitor Options & Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-record(state,{
	op_mode = [gt],
	population_id = test,
	activeAgent_IdPs = [],
	agent_ids = [],
	tot_agents,
	agents_left,
	op_tag,
	agent_summaries = [],
	pop_gen = 0,
	eval_acc = 0,
	cycle_acc = 0,
	time_acc = 0,
	tot_evaluations = 0,
	step_size,
	goal_status,
	evolutionary_algorithm,
	fitness_postprocessor,
	selection_algorithm,
	best_fitness,
	survival_percentage = 0.5,
	specie_size_limit = 10,
	init_specie_size = 10,
	polis_id = mathema,
	generation_limit = 100,
	evaluations_limit = 20000,
	fitness_goal = inf,
	benchmarker_pid,
	committee_pid,
	goal_reached=false
}).

-define(INIT_CONSTRAINTS,[
	#constraint{
		morphology=Morphology,
		connection_architecture=CA,
		population_evo_alg_f=steady_state,%steady_state
		neural_pfns=[none],
		agent_encoding_types=[neural],
		substrate_plasticities=[iterative],
		substrate_linkforms=[l2l_feedforward]
	} || 
		Morphology<-[prey],
		CA<-[recurrent]
]).
-define(INIT_PMP,#pmp{
		op_mode=[gt],
		population_id=test,
		survival_percentage=0.5,
		evolution_type = generational,
		selection_type = hof_competition,
		specie_constraint = ?INIT_CONSTRAINTS,
		specie_size_limit=10,
		init_specie_size=10,
		polis_id = mathema,
		generation_limit = inf,
		evaluations_limit = inf,
		fitness_goal = inf
	}).

-define(MIN_ACCEPTABLE_FITNESS_RATIO,1).
-define(FS,false).
-define(EFF,1). %Efficiency., TODO: this should further be changed from absolute number of neurons, to diff in lowest or avg, and the highest number of neurons

%%==================================================================== API
%%--------------------------------------------------------------------
%% Function: start_link() -> {ok,Pid} | ignore | {error,Error}
%% Description: Starts the server
%%--------------------------------------------------------------------
start_link(Start_Parameters) ->
	gen_server:start_link(?MODULE, Start_Parameters, []).

start(Start_Parameters) -> 
	gen_server:start(?MODULE, Start_Parameters, []).
	
start_link() ->
	gen_server:start_link(?MODULE, [], []).
    
start() -> 
	%gen_server:start(?MODULE, [], []).
	init_population(#state{op_mode = [gt,validation]},?INIT_CONSTRAINTS).

stop() ->
	gen_server:cast(monitor,{stop,normal}).
	
init(Pid,InitState)->
	gen_server:cast(Pid,{init,InitState}).

%%==================================================================== gen_server callbacks
%%--------------------------------------------------------------------
%% Function: init(Args) -> {ok, State} |
%%                         {ok, State, Timeout} |
%%                         ignore               |
%%                         {stop, Reason}
%% Description: Initiates the server
%%--------------------------------------------------------------------
init(S) ->
	process_flag(trap_exit,true),
	register(monitor,self()),
	Population_Id = S#state.population_id,
	OpMode = S#state.op_mode,
	io:format("******** Population monitor started with parameters:~p~n",[S]),
	Agent_Ids = extract_AgentIds(Population_Id,all),
	ActiveAgent_IdPs = summon_agents(OpMode,Agent_Ids),
	P = genotype:dirty_read({population,Population_Id}),
	[put({evaluations,Specie_Id},0) || Specie_Id<-P#population.specie_ids],
	T = P#population.trace,
	TotEvaluations=T#trace.tot_evaluations,
	io:format("Initial Tot Evaluations:~p~n",[TotEvaluations]),
	State = S#state{
		population_id = Population_Id,
		activeAgent_IdPs = ActiveAgent_IdPs,
		tot_agents = length(Agent_Ids),
		agents_left = length(Agent_Ids),
		op_tag = continue,
		evolutionary_algorithm = P#population.evo_alg_f,
		fitness_postprocessor = P#population.fitness_postprocessor_f,
		selection_algorithm = P#population.selection_f,
		best_fitness = 0,
		step_size = T#trace.step_size,
		tot_evaluations = TotEvaluations
	},
	{ok, State}.
%In init/1 the population_monitor proces registers itself with the node under the name monitor, and sets all the needed parameters within its #state record. The function first extracts all the Agent_Ids that belong to the population using the extract_AgentIds/2 function. Each agent is then spawned/activated, converted from genotype to phenotype in the summon_agents/2 function. The summon_agents/2 function summons the agents and returns to the caller a list of tuples with the following format: [{Agent_Id,Agent_PId}...]. Once the state record's parameters have been set, the function drops into the main gen_server loop.

%%--------------------------------------------------------------------
%% Function: %% handle_call(Request, From, State) -> {reply, Reply, State} |
%%                                      {reply, Reply, State, Timeout} |
%%                                      {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, Reply, State} |
%%                                      {stop, Reason, State}
%% Description: Handling call messages
%%--------------------------------------------------------------------
handle_call({stop,normal},_From, S)->
	ActiveAgent_IdPs = S#state.activeAgent_IdPs,
	[Agent_PId ! {self(),terminate} || {_DAgent_Id,Agent_PId}<-ActiveAgent_IdPs],
	{stop, normal, S};
handle_call({stop,shutdown},_From,State)->
	{stop, shutdown, State}.
%If the population_monitor process receives a {stop,normal} call, it checks if there are any still active agents. If there are any, it terminates them, and then itself terminates.

%%--------------------------------------------------------------------
%% Function: handle_cast(Msg, State) -> {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, State}
%% Description: Handling cast messages
%%--------------------------------------------------------------------
handle_cast({Agent_Id,terminated,Fitness},S) when S#state.evolutionary_algorithm == generational ->
	Population_Id = S#state.population_id,
	OpTag = S#state.op_tag,
	OpMode = S#state.op_mode,
	AgentsLeft = S#state.agents_left,
	case (AgentsLeft-1) =< 0 of
		true ->
%			benchmark ! {self(), tunning_phase,done},
			intrapopulation_selection(Population_Id,S#state.specie_size_limit,S#state.fitness_postprocessor,S#state.selection_algorithm),
			U_PopGen = S#state.pop_gen+1,
			io:format("Population Generation:~p Ended.~n~n~n",[U_PopGen]),
			case OpTag of
				continue ->
					Specie_Ids = (genotype:dirty_read({population,Population_Id}))#population.specie_ids,
					SpecFitList=[(genotype:dirty_read({specie,Specie_Id}))#specie.fitness || Specie_Id <- Specie_Ids],
					BestFitness=lists:nth(1,lists:reverse(lists:sort([MaxFitness || {_,_,MaxFitness,_} <- SpecFitList]))),
					io:format("BestFitness:~p~n",[BestFitness]),
					Generation_Limit=S#state.generation_limit,
					Evaluation_Limit=S#state.evaluations_limit,
					Fitness_Goal=S#state.fitness_goal,
					case (U_PopGen >= Generation_Limit) or (S#state.tot_evaluations >= Evaluation_Limit) or fitness_goal_reached(BestFitness,Fitness_Goal) or S#state.goal_reached of
						true ->%ENDING_CONDITION_REACHED
							%benchmark ! {self(),goal_failed,S#state.evaluations_acc,U_PopGen},
							io:format("Ending condition reached~n"),
							Agent_Ids = extract_AgentIds(Population_Id,all),
							TotAgents=length(Agent_Ids),
							U_S=S#state{agent_ids=Agent_Ids, tot_agents=TotAgents, agents_left=TotAgents, pop_gen=U_PopGen},
							{stop,normal,U_S};
						false ->%IN_PROGRESS
							Agent_Ids = extract_AgentIds(Population_Id,all),
							U_ActiveAgent_IdPs=summon_agents(OpMode,Agent_Ids),
							TotAgents=length(Agent_Ids),
							U_S=S#state{activeAgent_IdPs=U_ActiveAgent_IdPs, tot_agents=TotAgents, agents_left=TotAgents, pop_gen=U_PopGen},
							{noreply,U_S}
					end;
				done ->
					io:format("Shutting down Population Monitor~n"),
					U_S = S#state{agents_left = 0,pop_gen=U_PopGen},
					{stop,normal,U_S};
				pause ->
					io:format("Population Monitor has paused.~n"),
					%benchmark ! {self(), monitor, paused},
					U_S = S#state{agents_left=0,pop_gen=U_PopGen},
					{noreply,U_S}
			end;
		false ->
			io:format("Agents Left:~p~n",[AgentsLeft-1]),
			ActiveAgent_IdPs = S#state.activeAgent_IdPs,
			U_ActiveAgent_Ids = lists:keydelete(Agent_Id,1,ActiveAgent_IdPs),
			U_S = S#state{activeAgent_IdPs = U_ActiveAgent_Ids,agents_left = AgentsLeft-1},
			{noreply,U_S}
	end;
%This clause accepts the cast signals sent by the agents which terminate after finishing with their evaluations. The clause specialises in the "competition" selection algorithm, which is a generational selection algorithm. As a generation selection algorithm, it waits untill the entire population has finished being evaluated, and only then selects the fit from the unfit, and creates the updated population of the next generation. The OpTag can be set from the outsie to shutdown the population_monitor by setting it to done. Once an ending condition is reached, either through a generation limit, an evaluations limit, or fitness goal, the population_monitor exits normally. If the ending condition is not reached, the population_monitor spawns the new generation of agents and awaits again for all the agents in the population to complete their evaluations. If the OpTag is set to pause, it does not generate a new population, and instead goes into a waiting mode, and awaits to be restarted or terminated.

handle_cast({Agent_Id,terminated,Fitness},State) when State#state.evolutionary_algorithm == steady_state ->
	F = fun()->
		%io:format("Tot Evaluations:~p~n",[State#state.tot_evaluations]),
		io:format("Agent: ~p terminated.~n",[Agent_Id]),
		A = genotype:read({agent,Agent_Id}),
		Specie_Id = A#agent.specie_id,
		S = genotype:read({specie,Specie_Id}),
		U_ActiveAgent_Ids = S#specie.agent_ids -- [Agent_Id],
		Distinguishers = S#specie.hof_distinguishers,
		SHOF = S#specie.hall_of_fame,
		{U_SHOF,Losers}=update_SHOF(SHOF,[Agent_Id],Distinguishers,[]),
		io:format("SHOF:~p~nU_SHOF:~p~n",[SHOF,U_SHOF]),
		U_S = S#specie{hall_of_fame=U_SHOF,agent_ids=U_ActiveAgent_Ids},
		genotype:write(U_S),
		U_S
	end,
	{atomic,U_S}=mnesia:transaction(F),
	ActiveAgent_IdPs = lists:keydelete(Agent_Id,1,State#state.activeAgent_IdPs),
	case (State#state.tot_evaluations >= State#state.evaluations_limit) or (State#state.goal_status==reached) of
		true ->%DONE
			gather_STATS(State#state.population_id,State#state.tot_evaluations,State#state.op_mode),
			[gen_server:cast(PId,{stop,normal}) || {_Id,PId}<-ActiveAgent_IdPs],
			{stop,normal,State};
		false ->%CONTINUE
			F2 = fun()->
				U_SHOF = U_S#specie.hall_of_fame,
				Specie_Id = U_S#specie.id,
				FitnessScaled=[{Champ#champion.main_fitness/math:pow(Champ#champion.tot_n,?EFF),Champ#champion.id}||Champ<-U_SHOF],
				TotFitness = lists:sum([Main_Fitness || {Main_Fitness,_Id}<-FitnessScaled]),
				[Offspring_Id]=selection_algorithm:choose_Winners(Specie_Id,FitnessScaled,TotFitness,[],[],1),
				U2_S = genotype:read({specie,Specie_Id}),
				genotype:write(U2_S#specie{agent_ids=[Offspring_Id|U2_S#specie.agent_ids]}),
				Offspring_Id
			end,
			{atomic,Offspring_Id}=mnesia:transaction(F2),
			OpMode = case lists:member(gt,State#state.op_mode) of
				true ->
					gt;
				false ->
					exit("ERROR in population_monitor. OpModes does not contain gt in training stage~n")
			end,
			[{Offspring_Id,Offspring_PId}] = summon_agents(OpMode,[Offspring_Id]),%TODO: Uses a static gt opmode.
			{noreply,State#state{activeAgent_IdPs=[{Offspring_Id,Offspring_PId}|ActiveAgent_IdPs]}}
	end;
handle_cast({op_tag,pause},S) when S#state.op_tag == continue ->
	U_S = S#state{op_tag = pause},
	{noreply,U_S};
%The population_monitor process accepts a pause command cast, which if it recieves, it then goes into pause mode after all the agents have completed with their evaluations. The process can only go into pause mode if it is currently in the continue mode (its op_tag is set to continue).

handle_cast({op_tag,continue},S) when S#state.op_tag == pause ->
	Population_Id = S#state.population_id,
	OpMode = S#state.op_mode,
	Agent_Ids = extract_AgentIds(Population_Id,all),
	U_ActiveAgent_IdPs=summon_agents(OpMode,Agent_Ids),
	TotAgents=length(Agent_Ids),
	U_S=S#state{activeAgent_IdPs=U_ActiveAgent_IdPs,tot_agents=TotAgents,agents_left=TotAgents,op_tag=continue},
	{noreply,U_S};
%The population_monitor process can accept a continue command if its current op_tag is set to pause. When it receives a continue command, it summons all the agents in the population, and continues with its neuroevolution synchronization duties.

handle_cast({From,evaluations,Specie_Id,AEA,AgentCycleAcc,AgentTimeAcc},S)->
	AgentEvalAcc = case S#state.goal_reached of
		true ->
			0;
		false ->
			AEA
	end,
	Eval_Acc = S#state.eval_acc,
	U_EvalAcc = S#state.eval_acc+AgentEvalAcc,
	U_CycleAcc = S#state.cycle_acc+AgentCycleAcc,
	U_TimeAcc = S#state.time_acc+AgentTimeAcc,
	U_TotEvaluations = S#state.tot_evaluations + AgentEvalAcc,
	SEval_Acc=get({evaluations,Specie_Id}),
	put({evaluations,Specie_Id},SEval_Acc+AgentEvalAcc),
	
	case U_TotEvaluations rem 50 of
		0 ->
			io:format("Tot_Evaluations:~p~n",[U_TotEvaluations]);
		_ ->
			done
	end,
	U_S=case U_EvalAcc >= S#state.step_size of
		true ->
			io:format("Evaluations/Step:~p~n",[Eval_Acc]),
			gather_STATS(S#state.population_id,U_EvalAcc,S#state.op_mode),
			Population_Id = S#state.population_id,
			P = genotype:dirty_read({population,Population_Id}),
			T = P#population.trace,
			case S#state.committee_pid of
				undefined -> ok;
				Committee_PId ->
					Committee_PId ! {self(),trace_update,T}
			end,
			TotEvaluations=T#trace.tot_evaluations,
			io:format("Tot Evaluations:~p~n",[U_TotEvaluations]),
			S#state{eval_acc=0, cycle_acc=0, time_acc=0, tot_evaluations=U_TotEvaluations};
		false ->
			S#state{eval_acc=U_EvalAcc,cycle_acc=U_CycleAcc,time_acc=U_TimeAcc,tot_evaluations=U_TotEvaluations}
	end,
	{noreply,U_S};

handle_cast({_From,goal_reached},S)->
	U_S=S#state{goal_reached=true},
	{noreply,U_S};

handle_cast({_From,print_TRACE},S)->
	Population_Id = S#state.population_id,
	P = genotype:dirty_read({population,Population_Id}),
	io:format("******** TRACE ********:~n~p~n",[P#population.trace]),
	{noreply,S};

handle_cast({init,InitState},_State)->
	{noreply,InitState};
handle_cast({stop,normal},State)->
	{stop, normal,State};
handle_cast({stop,shutdown},State)->
	{stop, shutdown, State}.
%%--------------------------------------------------------------------
%% Function: handle_info(Info, State) -> {noreply, State} |
%%                                       {noreply, State, Timeout} |
%%                                       {stop, Reason, State}
%% Description: Handling all non call/cast messages
%%--------------------------------------------------------------------
handle_info(_Info, State) ->
    {noreply, State}.

terminate(Reason, S) ->
	case S of
		[] ->
			io:format("******** Population_Monitor shut down with Reason:~p, with State: []~n",[Reason]);
		_ ->
			OpMode = S#state.op_mode,
			OpTag = S#state.op_tag,
			TotEvaluations=S#state.tot_evaluations,
			Population_Id = S#state.population_id,
			gather_STATS(Population_Id,0,OpMode),
			P = genotype:dirty_read({population,Population_Id}),
			T = P#population.trace,
			U_T = T#trace{tot_evaluations=TotEvaluations},
			U_P = P#population{trace=U_T},
			genotype:write(U_P),
			io:format("******** TRACE START ********~n"),
			io:format("~p~n",[U_T]),
			io:format("******** ^^^^ TRACE END ^^^^ ********~n"),
			io:format("******** Population_Monitor:~p shut down with Reason:~p OpTag:~p, while in OpMode:~p~n",[Population_Id,Reason,OpTag,OpMode]),
			io:format("******** Tot Agents:~p Population Generation:~p Tot_Evals:~p~n",[S#state.tot_agents,S#state.pop_gen,S#state.tot_evaluations]),
			case S#state.benchmarker_pid of
				undefined ->
					ok;
				PId ->
					PId ! {S#state.population_id,completed,U_T}
			end
	end.
%When the population_monitor process terminates, it states so, notifies with what op_tag and op_mode it terminated, all the stats gathered, and then shuts down.

%%--------------------------------------------------------------------
%% Func: code_change(OldVsn, State, Extra) -> {ok, NewState}
%% Description: Convert process state when code is changed
%%--------------------------------------------------------------------
code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%--------------------------------------------------------------------
%% Internal functions
%%--------------------------------------------------------------------
extract_AgentIds(Population_Id,AgentType)->
	P = genotype:dirty_read({population,Population_Id}),
	Specie_Ids = P#population.specie_ids,
	%io:format("Specie_Ids:~p~n",[Specie_Ids]),
	case AgentType of
		champions ->
			extract_ChampionAgentIds(Specie_Ids,[]);
		all ->
			extract_AllAgentIds(Specie_Ids,[])
	end.
%The extract_AgentIds/2 function accepts the Population_Id and a parameter which specifies what type of agents (all agent ids, or just those of the champions) to extract from the population, after which it extracts those agents. Depending on the AgentType parameter, the function either calls extract_ChampionAgentIds/2 or extract_AllAgentIds/2, which return the list of agent ids to the caller.

	extract_ChampionAgentIds([Specie_Id|Specie_Ids],Acc)->
		S = genotype:dirty_read({specie,Specie_Id}),
		ChampionAgent_Ids = S#specie.champion_ids,
		extract_ChampionAgentIds(Specie_Ids,lists:append(ChampionAgent_Ids,Acc));
	extract_ChampionAgentIds([],Acc)->
		Acc.
%extract_ChampionAgentIds/2 accumulates the ids of champion agents from every specie in the Specie_Ids list, and then returns that list to the caller.
	
	extract_AllAgentIds([Specie_Id|Specie_Ids],Acc)->
		extract_AllAgentIds(Specie_Ids,lists:append(extract_SpecieAgentIds(Specie_Id),Acc));
	extract_AllAgentIds([],Acc)->
		Acc.
%extract_AllAgentIds/2 accumulates and returns to the caller an id list of all the agents belonging to the species in the Specie_Ids list.
		
extract_SpecieAgentIds(Specie_Id)->
	S = genotype:dirty_read({specie,Specie_Id}),
	S#specie.agent_ids.
%extract_SpecieAgentIds/1 returns a list of agent ids to the caller.

summon_agents(OpMode,Agent_Ids)->
	summon_agents(OpMode,Agent_Ids,[]).
summon_agents(OpMode,[Agent_Id|Agent_Ids],Acc)->
%	io:format("Agent_Id:~p~n",[Agent_Id]),
	Agent_PId = exoself:start(Agent_Id,self()),
	summon_agents(OpMode,Agent_Ids,[{Agent_Id,Agent_PId}|Acc]);
summon_agents(_OpMode,[],Acc)->
	Acc.
%The summon_agents/2 and summon_agents/3 spawns all the agents in the Agent_ids list, and returns to the caller a list of tuples as follows: [{Agent_Id,Agent_PId}...].

fitness_goal_reached(_BestFitness,inf)->
	false;
fitness_goal_reached([BestFit|BestFitness],[GoalFit|Fitness_Goal])->
	case BestFit > GoalFit of
		true ->
			fitness_goal_reached(BestFitness,Fitness_Goal);
		false ->
			false
	end;
fitness_goal_reached([],[_GoalFit|Fitness_Goal])->
	false;
fitness_goal_reached([_BestFit|BestFitness],[])->
	true.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test()->
	init_population(#state{op_mode = [gt,validation]},?INIT_CONSTRAINTS).
%The test/0 function starts the population monitor through init_population/1 with a set of default parameters specified by the macros of this module.

prep_PopState(PMP,Specie_Constraints)->
	S=#state{
		op_mode=PMP#pmp.op_mode,
		population_id = PMP#pmp.population_id,
		survival_percentage=PMP#pmp.survival_percentage,
		specie_size_limit=PMP#pmp.specie_size_limit,
		init_specie_size=PMP#pmp.init_specie_size,
		polis_id=PMP#pmp.polis_id,
		generation_limit=PMP#pmp.generation_limit,
		evaluations_limit=PMP#pmp.evaluations_limit,
		fitness_goal=PMP#pmp.fitness_goal,
		benchmarker_pid=PMP#pmp.benchmarker_pid
	},
	init_population(S,Specie_Constraints).

init_population(Init_State,Specie_Constraints)->
	random:seed(now()),
	Population_Id = Init_State#state.population_id,
	%OpMode = Init_State#state.op_mode,
	F = fun()->
		case genotype:read({population,Population_Id}) of
			undefined ->
				create_Population(Population_Id,Init_State#state.init_specie_size,Specie_Constraints);
			_ ->
				delete_population(Population_Id),
				create_Population(Population_Id,Init_State#state.init_specie_size,Specie_Constraints)
		end
	end,
	Result = mnesia:transaction(F),
	case Result of
		{atomic,_} ->
			population_monitor:start(Init_State);
		Error ->
			io:format("******** ERROR in PopulationMonitor:~p~n",[Error])
	end.
%The function init_population/1 creates a new population with the id Population_Id, composed of length(Specie_Constraints) species, where each specie uses the particular specie constraint specified within the Specie_Constraints list. The function first checks if a population with the noted Population_Id already exists, if a population does exist, then the function first delets it, and then creates a fresh one. Since the ids are usually generated with the genotype:create_UniqueId/0, the only way an already existing Population_Id is dropped into the function as a parameter is if it is intended, usually when runing tests, with the Population_Id = test.

	create_Population(Population_Id,SpecieSize,Specie_Constraints)->
		Specie_Ids = [create_specie(Population_Id,SpecCon,origin,SpecieSize) || SpecCon <- Specie_Constraints],
		[C|_]=Specie_Constraints,
		Seed_Agent_Ids = lists:flatten([(genotype:read({specie,Specie_Id}))#specie.agent_ids || Specie_Id<-Specie_Ids]),
		io:format("SeedIds:~p~n",[Seed_Agent_Ids]),
		Population = #population{
			id = Population_Id,
			specie_ids = Specie_Ids,
			evo_alg_f = C#constraint.population_evo_alg_f,
			fitness_postprocessor_f = C#constraint.population_fitness_postprocessor_f,
			selection_f = C#constraint.population_selection_f,
			seed_specie_ids=Specie_Ids,
			seed_agent_ids= Seed_Agent_Ids
		},
		genotype:write(Population).

		create_specie(Population_Id,SpeCon,Fingerprint)->
			Specie_Id = genotype:generate_UniqueId(),
			create_specie(Population_Id,Specie_Id,0,[],SpeCon,Fingerprint).
		create_specie(Population_Id,SpeCon,Fingerprint,SpecieSize)->
			Specie_Id = genotype:generate_UniqueId(),
			create_specie(Population_Id,Specie_Id,SpecieSize,[],SpeCon,Fingerprint).
		create_specie(Population_Id,Specie_Id,0,IdAcc,SpeCon,Fingerprint)->
			io:format("Specie_Id:~p Morphology:~p~n",[Specie_Id,SpeCon#constraint.morphology]),
			Specie = #specie{
				id = Specie_Id,
				population_id = Population_Id,
				fingerprint = Fingerprint,
				constraint = SpeCon,
				agent_ids = IdAcc,
				all_agent_ids = IdAcc,
				seed_agent_ids = IdAcc
			},
			genotype:write(Specie),
			Specie_Id;
		create_specie(Population_Id,Specie_Id,Agent_Index,IdAcc,SpeCon,Fingerprint)->
			Agent_Id = {genotype:generate_UniqueId(),agent},
			genotype:construct_Agent(Specie_Id,Agent_Id,SpeCon),
			create_specie(Population_Id,Specie_Id,Agent_Index-1,[Agent_Id|IdAcc],SpeCon,Fingerprint).
%The create_Population/3 generates length(Specie_Constraints) number of specie, each composed of ?INIT_SPECIE_SIZE number of agents. The function uses the create_specie/4 to generate the species. The create_specie/3 and create_specie/4 functions are simplified versions which use default parameters to call the create_specie/6 function. The create_specie/6 function constructs the agents using the genotype:construct_Agent/3 function, accumulating the Agent_Ids in the IdAcc list. Once all the agents have been created, the function creates the specie record, fills in the required elements, writes the specie to database, and then finally returns the Specie_Id to the caller.

continue()->
	random:seed(now()),
	population_monitor:start(#state{op_mode = [gt,validation]}).
continue(Population_Id)->
	random:seed(now()),
	S = #state{population_id=Population_Id,op_mode = [gt,validation]},
	population_monitor:start(S).
%The function continue/0 and continue/1 are used to summon an already existing population with either the default population Id, or the specified Population_Id.

%--------------------------------Evolve Top Technomes--------------------------------
%%%Notes: Species
%%%Function: 
%%%Interface:Input:() Output:
%%%MsgComunication: N/A
intrapopulation_selection(Population_Id,Specie_Size_Limit,Fitness_Postprocessor,Selection_Algorithm)->%,SelectionType)->%
	%NeuralEnergyCost = calculate_EnergyCost(Population_Id),
	F = fun()->
		P = genotype:read({population,Population_Id}),
		%Distinguishers = P#population.distinguishers,
		%Objectives = P#population.objectives,
		Specie_Ids = P#population.specie_ids,
		%update_HallOfFame_Population(Population_Id),
		%[intraspecie_selection(Specie_Id,SelectionType) || Specie_Id <- Specie_Ids]
		[intraspecie_selection(Specie_Id,Specie_Size_Limit,Fitness_Postprocessor,Selection_Algorithm) || Specie_Id <- Specie_Ids]
	end,
	{atomic,_} = mnesia:transaction(F).

	intraspecie_selection(Specie_Id,Specie_Size_Limit,Fitness_Postprocessor,Selection_Algorithm_Name)->
		S = genotype:dirty_read({specie,Specie_Id}),
		%Objectives = S#specie.objectives,
		Distinguishers = S#specie.hof_distinguishers,
		Agent_Ids = S#specie.agent_ids,
		SHOF = S#specie.hall_of_fame,
		{U_SHOF,Losers}=update_SHOF(SHOF,Agent_Ids,Distinguishers,[]),
		%io:format("SHOF:~p~nU_SHOF:~p~n",[SHOF,U_SHOF]),
		{AvgFitness,Std,MaxFitness,MinFitness} = calculate_SpecieFitness({specie,S}),
		{Factor,Fitness}=S#specie.innovation_factor,
		U_InnovationFactor = case MaxFitness > Fitness of
			true ->
				{0,MaxFitness};
			false ->
				{Factor-1,Fitness}
		end,
		U_S = S#specie{
			hall_of_fame=U_SHOF,
			fitness = {AvgFitness,Std,MaxFitness,MinFitness},
			innovation_factor = U_InnovationFactor
		},
		genotype:write(U_S),
		%io:format("Before seletion algorithm~n"),
		NewGen_SpecieAgents=case ?INTERACTIVE_SELECTION of
			false ->
				selection_algorithm:Selection_Algorithm_Name(Specie_Id,Losers,Specie_Size_Limit);
			true ->%io:format("U_SHOF:~p~n",[U_SHOF]),
				interactive_evolution:select(Specie_Id,Losers)
		end,
		%io:format("After selection algorithm~n"),
		%NewGen_SpecieAgents=?MODULE:evolvability_research(Specie_Id,RemainingChampionDesignators),%USE THIS TO TAKE into account the evolvability or robustness.
		%io:format("NewGen_SpecieAgents:~p~n",[NewGen_SpecieAgents]),
		NewGen_SpecieAgents.
		
		update_SHOF(SHOF,[Agent_Id|Agent_Ids],Distinguishers,Acc)->
			case update_SHOF(SHOF,Agent_Id,Distinguishers) of
				{U_SHOF,undefined} ->
					update_SHOF(U_SHOF,Agent_Ids,Distinguishers,Acc);
				{U_SHOF,Loser} ->
					update_SHOF(U_SHOF,Agent_Ids,Distinguishers,[Loser|Acc])
			end;
		update_SHOF(SHOF,[],_Distinguishers,Acc)->
			{SHOF,Acc}.

			update_SHOF(SHOF,Agent_Id,Distinguishers)->%Will return {U_SHOF,Champion|undefined}
				Agent = to_champion_form(SHOF,Agent_Id,Distinguishers),
				case [C|| C<-SHOF, Agent#champion.hof_fingerprint==C#champion.hof_fingerprint] of %lists:keyfind(Agent#champion.hof_fingerprint, 2, SHOF) of
						[] ->%Champion with such fingerprint does not exist, thus it is entered, as a stepping stone, into the HOF
							A = genotype:read({agent,Agent#champion.id}),
							U_A = A#agent{champion_flag=[true|A#agent.champion_flag]},
							genotype:write(U_A),
							%retrograde_update(A#agent.parent_ids,A#agent.main_fitness,A#agent.fitness,0),
							update_FitnessStagnation(Agent#champion.id,better,?FS),
							{[Agent|SHOF],undefined};
						Champs ->%Agents with this fingerprint exist, and new agent is either entered or not into HOF based on fitness dominance... or behavioral minimal difference.							
							SHOF_Remainder = SHOF -- Champs,
							case fitness_domination(Agent,Champs) of
								false ->
									%io:format("Agent not added:~p~n",[{Agent}]),
									update_FitnessStagnation(Agent#champion.id,worse,?FS),
									{SHOF,Agent};
								U_Champs ->
									update_FitnessStagnation(Agent#champion.id,better,?FS),
									{SHOF_Remainder++U_Champs,undefined}
							end									
					end.
			
					update_FitnessStagnation(_,_,false)->
						ok;
					update_FitnessStagnation(Id,worse,true)->
						A = genotype:read({agent,Id}),
						case A#agent.parent_ids of
							[AncestorId] ->
								Ancestor = genotype:read({agent,AncestorId}),
								FS = Ancestor#agent.fs,
								io:format("FS worse:~p~n",[{FS,AncestorId}]),
								genotype:write(Ancestor#agent{fs=FS - FS*0.1});
							[] ->
								ok
						end;
					update_FitnessStagnation(Id,better,true)->
						A = genotype:read({agent,Id}),
						case A#agent.parent_ids of
							[AncestorId] ->
								Ancestor = genotype:read({agent,AncestorId}),
								FS = Ancestor#agent.fs,
								io:format("FS better:~p~n",[{FS,AncestorId}]),
								genotype:write(Ancestor#agent{fs=FS + (1-FS)*0.1});
							[] ->
								ok
						end.
			
					fitness_domination(Agent,SHOF)->
						case fitness_domination(Agent,SHOF,[],[]) of
							dominated ->
								%io:format("NOT ADDING, fitness_domination:~p~n",[{Agent}]),
								false;
							{on_pareto,RemainingChamps}->%On Pareto
								A = genotype:read({agent,Agent#champion.id}),
								U_A = A#agent{champion_flag=[true|A#agent.champion_flag]},
								genotype:write(U_A),
								[Agent|RemainingChamps];
							dominating->
								A = genotype:read({agent,Agent#champion.id}),
								U_A = A#agent{champion_flag=[true|A#agent.champion_flag]},
								genotype:write(U_A),
								[Agent];
							{strange,LoserAcc,RemainingChamps}->
								io:format("******** ALGORITHMIC ERROR:: fitness_domination!~n"),
								%io:format("ADDING, fitness_domination:~p~n",[{Agent}]),
								A = genotype:read({agent,Agent#champion.id}),
								U_A = A#agent{champion_flag=[true|A#agent.champion_flag]},
								genotype:write(U_A),
								[Agent|RemainingChamps]
						end.
						
						fitness_domination(Agent,[Champ|Champs],LoserAcc,Acc)->
							case Agent#champion.hof_fingerprint == Champ#champion.hof_fingerprint of
								true ->
									VecDif=exoself:vec1_dominates_vec2(Agent#champion.fitness,Champ#champion.fitness,0,[]),
									%io:format("VecDif:~p~n",[VecDif]),
									TotDomElems=length([Val || Val<-VecDif, Val > 0]),
									TotElems=length(VecDif),
									case TotDomElems of
										TotElems ->%Dominating, could potentially check for behavioral difference here
											ChampA = genotype:read({agent,Champ#champion.id}),
											U_ChampA = ChampA#agent{champion_flag=[lost|ChampA#agent.champion_flag]},%true, false, lost, rentered
											genotype:write(U_ChampA),
											fitness_domination(Agent,Champs,[Champ|LoserAcc],Acc);
										0 ->%Dominated
											dominated;
										_ ->%On Pareto
											fitness_domination(Agent,Champs,LoserAcc,[Champ|Acc])
									end;
								false ->
									fitness_domination(Agent,Champs,LoserAcc,[Champ|Acc])
							end;
						fitness_domination(_Agent,[],LoserAcc,[])->
							dominating;
						fitness_domination(_Agent,[],[],Acc)->
							{on_pareto,Acc};
						fitness_domination(_Agent,[],LoserAcc,Acc)->
							{strange,LoserAcc,Acc}.
					
					on_pareto_front(Agent,SHOF)->
						case opf(Agent,SHOF) of
							false ->
								%io:format("NOT ADDING, on_pareto_front:~p~n",[{Agent}]),
								false;
							true ->
								%io:format("ADDING, on_pareto_front:~p~n",[{Agent}]),
								A = genotype:read({agent,Agent#champion.id}),
								U_A = A#agent{champion_flag=[true|A#agent.champion_flag]},
								genotype:write(U_A),
								[Agent|SHOF]
						end.
							
						opf(Agent,[Champ|Champs])->
							VecDif=exoself:vec1_dominates_vec2(Agent#champion.fitness,Champ#champion.fitness,0,[]),
							%io:format("VecDif:~p~n",[VecDif]),
							DifElems=length([Val || Val<-VecDif, Val > 0]),
							case DifElems of
								0 ->%Complete Inferiority
									false;
								_ ->%Variation, pareto front TODO
									opf(Agent,Champs)
							end;
						opf(_Agent,[])->
							true.
							
					novel_behavior(Agent,SHOF)->%TODO: Not really behavior, more just fitness, or fitness defined behavior?
						Minimal_Novelty=case get(minimal_novelty) of
							undefined ->
								put(minimal_novelty,2),
								2;
							Val ->
								Val
						end,
						MainFitness_List = [C#champion.main_fitness || C<-SHOF],
						SHOF_AvgFitness = functions:avg(MainFitness_List),
						SHOF_STD = functions:std(MainFitness_List,SHOF_AvgFitness,[]),
						Minimal_Fitness = SHOF_AvgFitness-0.1*abs(SHOF_AvgFitness),
						case (lists:min(Agent#champion.behavioral_differences) > Minimal_Novelty) and (Agent#champion.main_fitness > Minimal_Fitness) of
							true ->
								%io:format("ADDING, and the min different is:~p~n",[{lists:min(Agent#champion.behavioral_differences),Minimal_Novelty,Agent#champion.behavioral_differences}]),
								put(minimal_novelty,Minimal_Novelty+Minimal_Novelty*0.5+0.05),
								A = genotype:read({agent,Agent#champion.id}),
								U_A = A#agent{champion_flag=[true|A#agent.champion_flag]},
								genotype:write(U_A),
								[Agent|SHOF];
							false ->
								%io:format("NOT Adding, and the min different is:~p~n",[{lists:min(Agent#champion.behavioral_differences),Minimal_Novelty,Agent#champion.behavioral_differences}]),
								put(minimal_novelty,Minimal_Novelty-Minimal_Novelty*0.05+0.05),
								false
						end.
					
					to_champion_form(SHOF,Agent_Id,Distinguishers)->
						A=genotype:read({agent,Agent_Id}),
						Behavioral_Differences=case ?BEHAVIORAL_TRACE of
							true ->%io:format("behavioral trace:~p~n",[A#agent.behavioral_trace]),
								case phenotypic_diversity:compare_behavior(SHOF,A#agent.behavioral_trace) of
									[]->
										[0];
									Comparisons ->
										Comparisons
								end;
							false ->
								[0]
						end,
						#champion{
							hof_fingerprint=[specie_identifier:Distinguisher(Agent_Id)|| Distinguisher <- Distinguishers],
							fitness = A#agent.fitness,%A list of multi objective fitnesses and stuff.
							%validation_fitness=A#agent.validation_fitness,
							%test_fitness=A#agent.test_fitness,
							main_fitness = A#agent.main_fitness, % A single fitness value that we can use to decide on probability of offspring creation.
							tot_n = length(lists:flatten([NIds||{_LayerId,NIds}<-A#agent.pattern])),
							id = Agent_Id,
							evolvability=A#agent.evolvability,
							robustness=A#agent.robustness,
							brittleness=A#agent.brittleness,
							generation=A#agent.generation,
							behavioral_differences = Behavioral_Differences,
							fs = A#agent.fs
						}.
				
				vecdiff([A|V1],[B|V2],Acc)->
					vecdiff(V1,V2,[A-B|Acc]);
				vecdiff([],[],Acc)->
					Acc.
				
				retrograde_update(_AncestorIds,_Main_Fitness,_Fitness,10)->
					ok;
				retrograde_update([AncestorId|AncestorIds],Main_Fitness,Fitness,DepthIndex)->
					A = genotype:read({agent,AncestorId}),
					Ancestor_MainFitness = A#agent.main_fitness,
					Evolvability = A#agent.evolvability,
					Robustness = A#agent.robustness,
					Brittleness = A#agent.brittleness,
					case Main_Fitness > (Ancestor_MainFitness + Ancestor_MainFitness*0.05) of
						true ->%Update evolvability & robustness
							%AncestralDepthAmplifier = (math:pow(1.1,DepthIndex)),%TODO: This defines how the fitness is updated retroactively
							%PercentageBasedAddition = 0.1, 
							%U_Ancestor_MainFitness = AncestralDepthAmplifier*PercentageBasedAddition,
							U_Evolvability = Evolvability + 1,
							U_Robustness = Robustness,
							U_Brittleness = Brittleness;
						false ->%If within 10%, still update robustness
							case Main_Fitness > (Ancestor_MainFitness - Ancestor_MainFitness*0.05) of
								true -> 
									U_Evolvability = Evolvability,
									U_Robustness = Robustness+1,
									U_Brittleness = Brittleness;
								false ->
									U_Evolvability = Evolvability,
									U_Robustness = Robustness,
									U_Brittleness = Brittleness+1
							end
					end,
					U_A = A#agent{
						evolvability = U_Evolvability,
						robustness = U_Robustness,
						brittleness = U_Brittleness
					},
					genotype:write(U_A),
					retrograde_update(A#agent.parent_ids,Main_Fitness,Fitness,DepthIndex+1);
				retrograde_update([],_Main_Fitness,_Fitness,_Depth_Index)->
					ok.
	
				retrograde_update(AncestorId,Main_Fitness,Fitness)->
					A = genotype:read({agent,AncestorId}),
					Ancestor_MainFitness = A#agent.main_fitness,
					Evolvability = A#agent.evolvability,
					Robustness = A#agent.robustness,
					Brittleness = A#agent.brittleness,
					case Main_Fitness > (Ancestor_MainFitness + Ancestor_MainFitness*0.05) of
						true ->%Update evolvability & robustness
							%AncestralDepthAmplifier = (math:pow(1.1,DepthIndex)),%TODO: This defines how the fitness is updated retroactively
							%PercentageBasedAddition = 0.1, 
							%U_Ancestor_MainFitness = AncestralDepthAmplifier*PercentageBasedAddition,
							U_Evolvability = Evolvability + 1,
							U_Robustness = Robustness,
							U_Brittleness = Brittleness;
						false ->%If within 10%, still update robustness
							case Main_Fitness > (Ancestor_MainFitness - Ancestor_MainFitness*0.05) of
								true -> 
									U_Evolvability = Evolvability,
									U_Robustness = Robustness+1,
									U_Brittleness = Brittleness;
								false ->
									U_Evolvability = Evolvability,
									U_Robustness = Robustness,
									U_Brittleness = Brittleness+1
							end
					end,
					U_A = A#agent{
						evolvability = U_Evolvability,
						robustness = U_Robustness,
						brittleness = U_Brittleness
					},
					genotype:write(U_A).
	
		evolvability_research(Specie_Id,Specie_Size_Limit)->
			S = genotype:read({specie,Specie_Id}),
			%io:format("S:~p~n",[S]),
			genotype:write(S#specie{agent_ids=[]}),
			Distinguishers = S#specie.hof_distinguishers,
			Agent_Ids = get_SpecieAgentIds(Specie_Id),
			Champions=[to_champion_form(S#specie.hall_of_fame,Agent_Id,Distinguishers) || Agent_Id <-Agent_Ids],
			FitnessScaled=[{Champ#champion.main_fitness*math:pow(1.1,Champ#champion.evolvability),Champ#champion.id}||Champ<-Champions,Champ#champion.generation>0],
			TotFitness = lists:sum([Fitness || {Fitness,_Id}<-FitnessScaled]),
			NewGen_Ids=selection_algorithm:choose_Winners(Specie_Id,FitnessScaled,TotFitness,[],[],Specie_Size_Limit),
			%io:format("NewGen_Ids:~p~n",[NewGen_Ids]),
			genotype:write(S#specie{agent_ids=NewGen_Ids}),
			NewGen_Ids.
			
			get_SpecieAgentIds(Specie_Id)->
				S = genotype:dirty_read({specie,Specie_Id}),
				%io:format("****Specie_Id:~p****~n",[Specie_Id]),
				lists:append(S#specie.seed_agent_ids,[get_GeneticLineIds(Agent_Id,1)||Agent_Id<-S#specie.seed_agent_ids]).
		
				get_GeneticLineIds(Agent_Id,Generation)->
					A = genotype:dirty_read({agent,Agent_Id}),
					%io:format("Generation:~p Agent Id:~p~n",[Generation,Agent_Id]),
					lists:append(A#agent.offspring_ids,[get_GeneticLineIds(Id,Generation+1) || Id <- A#agent.offspring_ids]).

delete_population(Population_Id)->
	P = genotype:read({population,Population_Id}),
	Specie_Ids = P#population.specie_ids,
	io:format("delete_population ~p ::specie_ids:~p~n",[Population_Id,Specie_Ids]),
	[delete_specie(Specie_Id) || Specie_Id <- Specie_Ids],
	genotype:delete({population,Population_Id}).
	
	delete_specie(Specie_Id)->
		S = genotype:read({specie,Specie_Id}),
		%Every agent is an offspring of someone in the seed population
		%because there is only a single parent to an agent, starting from seed agents, it is possible to delete everyone, if we delete all offspring
		Seed_Agent_Ids = S#specie.seed_agent_ids,
		All_Agent_Ids = S#specie.all_agent_ids,
		io:format("delete_specie::seed_agent_ids:~p~n",[Seed_Agent_Ids]),
		io:format("delete_specie::agent_ids:~p~n",[All_Agent_Ids]),
		delete_Agents(All_Agent_Ids),
		%[genotype:delete_Agent(Agent_Id) || Agent_Id <- All_Agent_Ids],
%		[delete_genetic_line(Agent_Id) || Agent_Id <- Seed_Agent_Ids],
		genotype:delete({specie,Specie_Id}).
		
		delete_Agents([Agent_Id|Agent_Ids])->
			genotype:delete_Agent(Agent_Id),
			delete_Agents(Agent_Ids);
		delete_Agents([])->
			ok.
		
		delete_genetic_line(Agent_Id)->
			io:format("delete_genetic_line(~p)~n",[Agent_Id]),
			A = genotype:read({agent,Agent_Id}),
			genotype:delete_Agent(A#agent.id),
			Offspring_Ids = A#agent.offspring_ids,
			io:format("Offspring Ids:~p~n",[Offspring_Ids]),
			[delete_genetic_line(Id) || Id <- Offspring_Ids],
			ok.
			
calculate_EnergyCost(Population_Id)->
	Agent_Ids = extract_AgentIds(Population_Id,all),
	TotEnergy = lists:sum([extract_AgentFitness(Agent_Id) || Agent_Id<-Agent_Ids]),
	TotNeurons = lists:sum([extract_AgentTotNeurons(Agent_Id) || Agent_Id <- Agent_Ids]),
	EnergyCost = TotEnergy/TotNeurons,
	EnergyCost.
%The calculate_EnergyCost/1 calculates the average cost of each neuron, based on the fitness of each agent in the population, and the total number of neurons in the population. The value is calcualted by first adding up all the fitness scores of the agents belonging to the population. Then adding up the total number of neurons composing each agent in the population. And then finally producing the EnergyCost value by dividing the TotEnergy by TotNeurons, returning the value to the caller.

	extract_AgentTotNeurons(Agent_Id)->
		A = genotype:dirty_read({agent,Agent_Id}),
		Cx = genotype:dirty_read({cortex,A#agent.cx_id}),
		Neuron_Ids = Cx#cortex.neuron_ids,
		length(Neuron_Ids).
	
	extract_AgentFitness(Agent_Id)->
		A = genotype:dirty_read({agent,Agent_Id}),
		A#agent.fitness.
%The function extract_AgentTotNeurons simply extracts the neuron_ids list, and returns the length of that list, which is the total number of neurons belonging to the NN based system.

calculate_PopulationFitness(Population_Id,[Specie_Id|Specie_Ids],AvgFAcc,MaxFAcc,MinFAcc)->
	{AvgFitness,Std,MaxF,MinF}=calculate_SpecieFitness(Specie_Id),
	case get({fitness,Specie_Id}) of
		undefined ->
			put({fitness,Specie_Id},[{AvgFitness,Std}]);
		PrevGenFitness->
			put({fitness,Specie_Id},[{AvgFitness,Std}|PrevGenFitness])
	end,
	calculate_PopulationFitness(Population_Id,Specie_Ids,[{Specie_Id,AvgFitness}|AvgFAcc],[{Specie_Id,MaxF}|MaxFAcc],[{Specie_Id,MinF}|MinFAcc]);
calculate_PopulationFitness(_Population_Id,[],AvgFAcc,MaxFAcc,MinFAcc)->
	{AvgFAcc,MaxFAcc,MinFAcc}.

	calculate_SpecieFitness({specie,S})->
		Agent_Ids = S#specie.agent_ids,
		FitnessAcc = calculate_fitness(Agent_Ids),
		%Sorted_FitnessAcc=lists:sort(FitnessAcc),
		case FitnessAcc of
			[] ->
				MinFitness = [],
				MaxFitness = [],
				AvgFitness = [],
				StdFitness = inf;
			[AvgFitness] ->
				MaxFitness = AvgFitness,
				MinFitness = AvgFitness,
				StdFitness = inf;
			_ ->
				{MaxFitness,MinFitness,AvgFitness,StdFitness}=exoself:vector_basic_stats(FitnessAcc)
%				[MinFitness|_] = Sorted_FitnessAcc,
%				[MaxFitness|_] = lists:reverse(Sorted_FitnessAcc),
%				AvgFitness = functions:avg(FitnessAcc),
%				Std = functions:std(FitnessAcc)
		end,
		{AvgFitness,StdFitness,MaxFitness,MinFitness};
	calculate_SpecieFitness(Specie_Id)->
		S = genotype:dirty_read({specie,Specie_Id}),
		calculate_SpecieFitness({specie,S}).
%The calculate_SpecieFitness/1 function calculates the general fitness statistic of the specie, the averate, max, min, and standard deviation of the specie's fitness. The function first composes a fitness list by accessing the fitness scores of each agent belonging to it, and then calculates the noted above statistics from that list, returning the tuple to the caller.

	calculate_fitness(Agent_Ids)->
		calculate_fitness(Agent_Ids,[]).
	calculate_fitness([Agent_Id|Agent_Ids],FitnessAcc)->
		A = genotype:dirty_read({agent,Agent_Id}),
		case A#agent.fitness of
			undefined ->
				calculate_fitness(Agent_Ids,FitnessAcc);
			Fitness ->
				calculate_fitness(Agent_Ids,[Fitness|FitnessAcc])
		end;
	calculate_fitness([],FitnessAcc)->
		FitnessAcc.
%The calculate_fitness/1 composes a fitness list composed of the fitness values belonging to the agents in the Agent_Ids list. If the agent does not yet have a fitness score, if for example it has just been created/mutated but not yet evaluated, it is skipped. The composed fitness list is returned to the caller.

gather_STATS(Population_Id,EvaluationsAcc,OpMode)->
	io:format("Gathering Species STATS in progress~n"),
	TimeStamp = now(),
	F = fun() ->
		P = genotype:read({population,Population_Id}),
		T = P#population.trace,
		SpecieSTATS = [update_SpecieSTAT(Specie_Id,TimeStamp,OpMode) || Specie_Id<-P#population.specie_ids],
		PopulationSTATS = T#trace.stats,
		U_PopulationSTATS = [SpecieSTATS|PopulationSTATS],
		U_TotEvaluations = T#trace.tot_evaluations+EvaluationsAcc,
		U_Trace = T#trace{
			stats = U_PopulationSTATS,
			tot_evaluations=U_TotEvaluations
		},
		io:format("Population Trace:~p~n",[U_Trace]),
		genotype:write(P#population{trace=U_Trace})
	end,
	Result=mnesia:transaction(F),
	io:format("Result:~p~n",[Result]).
	
	update_SpecieSTAT(Specie_Id,TimeStamp,OpModes)->
		Specie_Evaluations = get({evaluations,Specie_Id}),
		put({evaluations,Specie_Id},0),
		S = genotype:read({specie,Specie_Id}),
		{Avg_Neurons,Neurons_Std} = calculate_SpecieAvgNodes({specie,S}),
		{AvgFitness,Fitness_Std,MaxFitness,MinFitness} = calculate_SpecieFitness({specie,S}),
		SpecieDiversity = calculate_SpecieDiversity({specie,S}),
		{ValFitness,TestFitness,Champion_Id}=validation_testing(Specie_Id,OpModes),
		STAT = #stat{
			morphology = (S#specie.constraint)#constraint.morphology,
			specie_id = Specie_Id,
			avg_neurons=Avg_Neurons,
			std_neurons=Neurons_Std,
			avg_fitness=AvgFitness,
			std_fitness=Fitness_Std,
			max_fitness=MaxFitness,
			min_fitness=MinFitness,
			avg_diversity=SpecieDiversity,
			evaluations = Specie_Evaluations,
			time_stamp=TimeStamp,
			validation_fitness = {ValFitness,Champion_Id},
			test_fitness = {TestFitness,Champion_Id}
		},
		STATS = S#specie.stats,
		U_STATS = [STAT|STATS],
		genotype:dirty_write(S#specie{stats=U_STATS}),
		STAT.

		validation_testing(Specie_Id,OpModes)->%Perhaps test all agents in the SHOF
			case lists:member(validation,OpModes) of
				true ->
					S = genotype:read({specie,Specie_Id}),
					SHOF = S#specie.hall_of_fame,
					U_SHOF=champion_ValTest(SHOF,[]),
					genotype:write(S#specie{hall_of_fame=U_SHOF}),
					SortedChampions=lists:reverse(lists:sort([{C#champion.main_fitness,C#champion.id} || C <- U_SHOF])),
					io:format("Sorted champions:~p~n",[SortedChampions]),
					case SortedChampions of
						[{Champ_TrnFitness,Champ_Id}|_] ->
							Champion=lists:keyfind(Champ_Id,3,U_SHOF),
							{Champion#champion.validation_fitness,Champion#champion.test_fitness,Champ_Id};
						[]->
							{[],[],void}
					end;
				false ->
					{[],[],void}
			end.
			
			champion_ValTest([C|Champions],Acc)->
				Champion_Id = C#champion.id,
				ValFitness=case C#champion.validation_fitness of
					undefined ->
						ValChampion_PId=exoself:start(Champion_Id,self(),validation),
						receive
							%{Champion_Id,ValFitness,FitnessProfile}->
							{ValChampion_Id,validation_complete,ValSpecie_Id,ValFitness,ValCycles,ValTime}->
							%io:format("Got Validation results:~p~n",[{Champion_Id,validation_complete,Specie_Id,ValFitness,Cycles,Time}]),
								ValFitness
						end;
					ValFitness->
						ValFitness
				end,
				TestFitness=case C#champion.test_fitness of
					undefined ->
						TestChampion_PId=exoself:start(Champion_Id,self(),test),
						receive
							%{Champion_Id,ValFitness,FitnessProfile}->
							{TestChampion_Id,test_complete,TestSpecie_Id,TestFitness,TestCycles,TestTime}->
								%io:format("Got Validation results:~p~n",[{Champion_Id,validation_complete,Specie_Id,ValFitness,Cycles,Time}]),
								TestFitness
						end;
					TestFitness ->
						TestFitness
				end,
				U_C = C#champion{validation_fitness=ValFitness,test_fitness=TestFitness},
				champion_ValTest(Champions,[U_C|Acc]);		
			champion_ValTest([],Acc)->
				lists:reverse(Acc).

calculate_SpecieAvgNodes({specie,S})->
	Agent_Ids = S#specie.agent_ids,
	calculate_AvgNodes(Agent_Ids,[]);
calculate_SpecieAvgNodes(Specie_Id)->
	io:format("calculate_SpecieAvgNodes(Specie_Id):~p~n",[Specie_Id]),
	S = genotype:read({specie,Specie_Id}),
	calculate_SpecieAvgNodes({specie,S}).
	
	calculate_AvgNodes([Agent_Id|Agent_Ids],NAcc)->
		io:format("calculate_AvgNodes/2 Agent_Id:~p~n",[Agent_Id]),
		A = genotype:read({agent,Agent_Id}),
		Cx = genotype:read({cortex,A#agent.cx_id}),
		Tot_Neurons = length(Cx#cortex.neuron_ids),
		calculate_AvgNodes(Agent_Ids,[Tot_Neurons|NAcc]);
	calculate_AvgNodes([],NAcc)->
		{functions:avg(NAcc),functions:std(NAcc)}.

calculate_PopulationDiversity(Population_Id,[Specie_Id|Specie_Ids],Acc)->
	Diversity = calculate_SpecieDiversity(Specie_Id),
	case get({diversity,Specie_Id}) of
		undefined->
			put({diversity,Specie_Id},[Diversity]);
		PrevGenDiversity ->
			put({diversity,Specie_Id},[Diversity|PrevGenDiversity])
	end,
	calculate_PopulationDiversity(Population_Id,Specie_Ids,[{Specie_Id,Diversity}|Acc]);
calculate_PopulationDiversity(_Tot_Population_Id,[],Acc)->
	Acc.

	calculate_SpecieDiversity({specie,S})->
		Agent_Ids = S#specie.agent_ids,
		Diversity = calculate_diversity(Agent_Ids);
	calculate_SpecieDiversity(Specie_Id)->
		S = genotype:dirty_read({specie,Specie_Id}),
		calculate_SpecieDiversity({specie,S}).
		
		calculate_diversity(Agent_Ids)->
			calculate_diversity(Agent_Ids,[]).
		calculate_diversity([Agent_Id|Agent_Ids],DiversityAcc)->
			A = genotype:read({agent,Agent_Id}),
			Fingerprint = A#agent.fingerprint,
			U_DiversityAcc = (DiversityAcc -- [Fingerprint]) ++ [Fingerprint],
			calculate_diversity(Agent_Ids,U_DiversityAcc);
		calculate_diversity([],DiversityAcc)->
			length(DiversityAcc).

print_SpecieDiversity([Specie_Id|Specie_Ids])->
	S = genotype:dirty_read({specie,Specie_Id}),
	Morphology=(S#specie.constraint)#constraint.morphology,
	io:format("Specie id:~p~n Specie morphology:~p~n Diversity:~p~n",[Specie_Id,Morphology,get({diversity,Specie_Id})]),
	print_SpecieDiversity(Specie_Ids);
print_SpecieDiversity([])->
	done.
