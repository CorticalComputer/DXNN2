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

-module(population_monitor).
-include("records.hrl").
%% API
-export([start_link/1,start_link/0,start/1,start/0,stop/0,init/2]).
%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3,create_MutantAgentCopy/1,test/0, create_specie/3, continue/2, continue/3, init_population/2, prep_PopState/2, extract_AgentIds/2,delete_population/1]).
-behaviour(gen_server).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Population Monitor Options & Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-define(INIT_CONSTRAINTS,[#constraint{morphology=Morphology,connection_architecture=CA, population_evo_alg_f=generational,neural_pfns=[none],agent_encoding_types=[neural],substrate_plasticities=[iterative],substrate_linkforms = [l2l_feedforward]} || Morphology<-[epitopes],CA<-[feedforward]]).
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
	evaluations_limit = 100000,
	fitness_goal = inf,
	benchmarker_pid,
	goal_reached=false
}).
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
	gen_server:start(?MODULE, [], []).

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
			mutate_population(Population_Id,S#state.specie_size_limit,S#state.fitness_postprocessor,S#state.selection_algorithm),
			U_PopGen = S#state.pop_gen+1,
%			io:format("Population Generation:~p Ended.~n~n~n",[U_PopGen]),
			case OpTag of
				continue ->
					Specie_Ids = (genotype:dirty_read({population,Population_Id}))#population.specie_ids,
					SpecFitList=[(genotype:dirty_read({specie,Specie_Id}))#specie.fitness || Specie_Id <- Specie_Ids],
					BestFitness=lists:nth(1,lists:reverse(lists:sort([MaxFitness || {_,_,MaxFitness,_} <- SpecFitList]))),
					Generation_Limit=S#state.generation_limit,
					Evaluation_Limit=S#state.evaluations_limit,
					Fitness_Goal=S#state.fitness_goal,
					case (U_PopGen >= Generation_Limit) or (S#state.tot_evaluations >= Evaluation_Limit) or (BestFitness >= Fitness_Goal) or S#state.goal_reached of
						true ->%ENDING_CONDITION_REACHED
							%benchmark ! {self(),goal_failed,S#state.evaluations_acc,U_PopGen},
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

handle_cast({Agent_Id,terminated,Fitness},S) when S#state.evolutionary_algorithm == steady_state ->
	Population_Id = S#state.population_id,
	Evaluation_Limit=S#state.evaluations_limit,
	Fitness_Goal=S#state.fitness_goal,
	NewBestFitness = case Fitness >= S#state.best_fitness of
		true ->
			Fitness;
		false ->
			S#state.best_fitness
	end,
	case (S#state.tot_evaluations >= Evaluation_Limit) or (NewBestFitness > Fitness_Goal) or S#state.goal_reached of
		true ->
			case lists:keydelete(Agent_Id,1,S#state.activeAgent_IdPs) of
				[] ->
					U_S=S#state{activeAgent_IdPs=[]},
					{stop,normal,U_S};
				U_ActiveAgent_IdPs ->
					U_S=S#state{activeAgent_IdPs=U_ActiveAgent_IdPs},
					{noreply,U_S}
			end;
		false ->
			FitnessPostprocessorName = S#state.fitness_postprocessor,
			SelectionAlgorithmName = S#state.selection_algorithm,
			A = genotype:dirty_read({agent,Agent_Id}),
			Morphology= (A#agent.constraint)#constraint.morphology,
			io:format("Agent_Id:~p of morphology:~p with fitness:~p terminated.~n",[Agent_Id,Morphology,Fitness]),
			Specie_Id = A#agent.specie_id,
			Specie = genotype:dirty_read({specie,Specie_Id}),
			Old_DeadPool_AgentSummaries = Specie#specie.dead_pool,
			Old_Agent_Ids = Specie#specie.agent_ids,
			[AgentSummary] = construct_AgentSummaries([Agent_Id],[]),
			DeadPool_AgentSummaries = [AgentSummary|Old_DeadPool_AgentSummaries],
			ProperlySorted_AgentSummaries = fitness_postprocessor:FitnessPostprocessorName(DeadPool_AgentSummaries),
	
			Valid_AgentSummaries = case length(ProperlySorted_AgentSummaries) >= S#state.specie_size_limit of
				true ->
					[{InvalidFitness,InvalidTotN,InvalidAgent_Id}|Remaining_AgentSummaries] = lists:reverse(ProperlySorted_AgentSummaries),
%					io:format("Informationtheoretic Death:~p::~p~n",[InvalidAgent_Id,{InvalidFitness,InvalidTotN,InvalidAgent_Id}]),
					genotype:delete_Agent(InvalidAgent_Id,safe),
					lists:reverse(Remaining_AgentSummaries);
				false ->
					ProperlySorted_AgentSummaries
			end,
			io:format("Valid_AgentSummaries:~p~n",[Valid_AgentSummaries]),
			{WinnerFitness,WinnerProfile,WinnerAgent_Id} = selection_algorithm:SelectionAlgorithmName(Valid_AgentSummaries),
			ActiveAgent_IdP = case random:uniform() < 0.1 of
				true ->
					U_DeadPool_AgentSummaries = lists:delete({WinnerFitness,WinnerProfile,WinnerAgent_Id},Valid_AgentSummaries),
					WinnerAgent_PId = Agent_PId = exoself:start(WinnerAgent_Id,self()),
					{WinnerAgent_Id,WinnerAgent_PId};
				false ->
					U_DeadPool_AgentSummaries = Valid_AgentSummaries,
					CloneAgent_Id = create_MutantAgentCopy(WinnerAgent_Id,safe),
					CloneAgent_PId = exoself:start(CloneAgent_Id,self()),
					{CloneAgent_Id,CloneAgent_PId}
			end,
			Top_AgentSummaries = lists:sublist(U_DeadPool_AgentSummaries,round(S#state.specie_size_limit*S#state.survival_percentage)),
			{_,_,TopAgent_Ids} = lists:unzip3(lists:sublist(Top_AgentSummaries,3)),
%			io:format("TopAgent_Ids:~p~n",[TopAgent_Ids]),
			USpecie=genotype:dirty_read({specie,Specie_Id}),
			genotype:write(USpecie#specie{dead_pool = U_DeadPool_AgentSummaries,champion_ids = TopAgent_Ids}),
			ActiveAgent_IdPs = S#state.activeAgent_IdPs,
			U_ActiveAgent_IdPs = [ActiveAgent_IdP|lists:keydelete(Agent_Id,1,ActiveAgent_IdPs)],
			U_S=S#state{
				activeAgent_IdPs=U_ActiveAgent_IdPs,
				best_fitness=NewBestFitness
			},
			{noreply,U_S}
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
	U_S=case U_EvalAcc >= S#state.step_size of
		true ->
			io:format("Evaluations/Step:~p~n",[Eval_Acc]),
			gather_STATS(S#state.population_id,U_EvalAcc,S#state.op_mode),
			Population_Id = S#state.population_id,
			P = genotype:dirty_read({population,Population_Id}),
			T = P#population.trace,
			TotEvaluations=T#trace.tot_evaluations,
			io:format("Tot Evaluations:~p~n",[TotEvaluations]),
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
			case TotEvaluations < 500 of
				true ->%So that there is at least one stat in the stats list.
					gather_STATS(Population_Id,0,OpMode);
				false ->
					ok
			end,
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test()->
	init_population(#state{op_mode = [gt,benchmark]},?INIT_CONSTRAINTS).
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
		Population = #population{
			id = Population_Id,
			specie_ids = Specie_Ids,
			evo_alg_f = C#constraint.population_evo_alg_f,
			fitness_postprocessor_f = C#constraint.population_fitness_postprocessor_f,
			selection_f = C#constraint.population_selection_f
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
				agent_ids = IdAcc
			},
			genotype:write(Specie),
			Specie_Id;
		create_specie(Population_Id,Specie_Id,Agent_Index,IdAcc,SpeCon,Fingerprint)->
			Agent_Id = {genotype:generate_UniqueId(),agent},
			genotype:construct_Agent(Specie_Id,Agent_Id,SpeCon),
			create_specie(Population_Id,Specie_Id,Agent_Index-1,[Agent_Id|IdAcc],SpeCon,Fingerprint).
%The create_Population/3 generates length(Specie_Constraints) number of specie, each composed of ?INIT_SPECIE_SIZE number of agents. The function uses the create_specie/4 to generate the species. The create_specie/3 and create_specie/4 functions are simplified versions which use default parameters to call the create_specie/6 function. The create_specie/6 function constructs the agents using the genotype:construct_Agent/3 function, accumulating the Agent_Ids in the IdAcc list. Once all the agents have been created, the function creates the specie record, fills in the required elements, writes the specie to database, and then finally returns the Specie_Id to the caller.

continue(OpMode,Selection_Algorithm)->
	Population_Id = test,
	population_monitor:start({OpMode,Population_Id,Selection_Algorithm}).
continue(OpMode,Selection_Algorithm,Population_Id)->
	population_monitor:start({OpMode,Population_Id,Selection_Algorithm}).
%The function continue/2 and continue/3 are used to summon an already existing population with Population_Id, and continue with the experiment using the Selection_Algorithm.

mutate_population(Population_Id,KeepTot,Fitness_Postprocessor,Selection_Algorithm)->
	NeuralEnergyCost = calculate_EnergyCost(Population_Id),
	F = fun()->
		P = genotype:read({population,Population_Id}),
		Specie_Ids = P#population.specie_ids,
		[mutate_Specie(Specie_Id,KeepTot,NeuralEnergyCost,Fitness_Postprocessor,Selection_Algorithm) || Specie_Id <- Specie_Ids]
	end,
	{atomic,_} = mnesia:transaction(F).
%The function mutate_population/3 mutates the agents within every specie in its specie_ids list, maintianing each specie within the size of KeepTot. The function first calculates the average cost of each neuron, and then calls each specie seperately with the Fitness_Postprocessor and Selection_Algorithm parameters, which are used to mutate the species.

	mutate_Specie(Specie_Id,PopulationLimit,NeuralEnergyCost,Fitness_Postprocessor_Name,Selection_Algorithm_Name)->
		S = genotype:dirty_read({specie,Specie_Id}),
		{AvgFitness,Std,MaxFitness,MinFitness} = calculate_SpecieFitness({specie,S}),
		Agent_Ids = S#specie.agent_ids,
		Sorted_AgentSummaries = lists:reverse(lists:sort(construct_AgentSummaries(Agent_Ids,[]))),
		io:format("Using: Fitness Postprocessor:~p Selection Algorirthm:~p~n",[Fitness_Postprocessor_Name,Selection_Algorithm_Name]),
		ProperlySorted_AgentSummaries = fitness_postprocessor:Fitness_Postprocessor_Name(Sorted_AgentSummaries),
		{NewGenAgent_Ids,TopAgent_Ids} = selection_algorithm:Selection_Algorithm_Name(ProperlySorted_AgentSummaries,NeuralEnergyCost,PopulationLimit),
		{FList,_TNList,_AgentIds}=lists:unzip3(ProperlySorted_AgentSummaries),
		[TopFitness|_] = FList,
		{Factor,Fitness}=S#specie.innovation_factor,
		U_InnovationFactor = case TopFitness > Fitness of
			true ->
				{0,TopFitness};
			false ->
				{Factor-1,Fitness}
		end,
		genotype:write(S#specie{
			agent_ids = NewGenAgent_Ids,
			champion_ids = [genotype:clone_Agent(TopAgent_Id)||TopAgent_Id <-TopAgent_Ids],
			fitness = {AvgFitness,Std,MaxFitness,MinFitness},
			innovation_factor = U_InnovationFactor}).
%The function mutate_Specie/5 calls the selection algorithm function to seperate the fit from the unfit organisms in the specie, and then mutates the fit organisms to produce offspring, maintaning the total specie size within PopulationLimit. The function first calls the fitness_postprocessor which sorts the agent summaries. Then, the resorted updated summaries are split into a valid (fit) and invalid (unfit) lists of agents by the selection algorithm. The invalid agents are deleted, and the valid agents are used to create offspring using the particular Selection_Algorithm_Name function. The agent ids belonging to the next generation (the valid agents and their offspring) are then produced by the selection function. Finally, the innovation factor (the last time the specie's top fitness improved) is updated, the ids of the 3 top agents within the species are noted, and the updated specie record is written to database.

	construct_AgentSummaries([Agent_Id|Agent_Ids],Acc)->
		A = genotype:dirty_read({agent,Agent_Id}),
		construct_AgentSummaries(Agent_Ids,[{A#agent.fitness,length((genotype:dirty_read({cortex,A#agent.cx_id}))#cortex.neuron_ids),Agent_Id}|Acc]);
	construct_AgentSummaries([],Acc)->
		Acc.
%The construct_AgentSummaries/2 reads the agents in the Agent_Ids list, and composes a list of tuples of the following format: [{AgentFitness,AgentTotNeurons,Agent_Id}...]. This list of tuples is reffered to as AgentSummaries. Once the AgentSummaries list is composed, it is returned to the caller.

	create_MutantAgentCopy(Agent_Id)->
		AgentClone_Id = genotype:clone_Agent(Agent_Id),
		io:format("AgentClone_Id:~p~n",[AgentClone_Id]),
		genome_mutator:mutate(AgentClone_Id),
		AgentClone_Id.
%The create_MutantAgentCopy/1 first creates a clone of the Agent_Id, and then uses the genome_mutator:mutate/1 function to mutate that clone, returning the id of the cloned agent to the caller.
				
	create_MutantAgentCopy(Agent_Id,safe)->%TODO
		A = genotype:dirty_read({agent,Agent_Id}),
		S = genotype:dirty_read({specie,A#agent.specie_id}),
		AgentClone_Id = genotype:clone_Agent(Agent_Id),
		Agent_Ids = S#specie.agent_ids,
		genotype:write(S#specie{agent_ids = [AgentClone_Id|Agent_Ids]}),
		io:format("AgentClone_Id:~p~n",[AgentClone_Id]),
		genome_mutator:mutate(AgentClone_Id),
		AgentClone_Id.
%The create_MutantAgentCopy/2 is similar to arity 1 function of the same name, but it also adds the id of the cloned mutant agent to the specie record to which the original belonged. The specie with its updated agent_ids is then written to database, and the id of the mutant clone is returned to the caller.

delete_population(Population_Id)->
	P = genotype:dirty_read({population,Population_Id}),
	Specie_Ids = P#population.specie_ids,
	[delete_specie(Specie_Id) || Specie_Id <- Specie_Ids],
	mnesia:delete({population,Population_Id}).
%The delete_population/1 function delets the entire population, by deleting the specie records belonging to the Population_Id, deleting the agent records belonging to those species, and then deleting the population record itself.
	
	delete_specie(Specie_Id)->
		S = genotype:dirty_read({specie,Specie_Id}),
		Agent_Ids = S#specie.agent_ids,
		[genotype:delete_Agent(Agent_Id) || Agent_Id <- Agent_Ids],
		mnesia:delete({specie,Specie_Id}).
%The delete_specie/1 function delets the agents associated with the Specie_Id, and then delets the specie record itself.
			
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
		Sorted_FitnessAcc=lists:sort(FitnessAcc),
		case Sorted_FitnessAcc of
			[] ->
				MinFitness = 0,
				MaxFitness = 0,
				AvgFitness = 0,
				Std = inf;
			[MinFitness] ->
				MaxFitness = MinFitness,
				AvgFitness = MinFitness,
				Std = inf;
			_ ->
				[MinFitness|_] = Sorted_FitnessAcc,
				[MaxFitness|_] = lists:reverse(Sorted_FitnessAcc),
				AvgFitness = functions:avg(FitnessAcc),
				Std = functions:std(FitnessAcc)
		end,
		{AvgFitness,Std,MaxFitness,MinFitness};
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
		mnesia:write(P#population{trace=U_Trace})
	end,
	Result=mnesia:transaction(F),
	io:format("Result:~p~n",[Result]).
	
	update_SpecieSTAT(Specie_Id,TimeStamp,OpMode)->
		Specie_Evaluations = get({evaluations,Specie_Id}),
		put({evaluations,Specie_Id},0),
		S = genotype:read({specie,Specie_Id}),
		{Avg_Neurons,Neurons_Std} = calculate_SpecieAvgNodes({specie,S}),
		{AvgFitness,Fitness_Std,MaxFitness,MinFitness} = calculate_SpecieFitness({specie,S}),
		SpecieDiversity = calculate_SpecieDiversity({specie,S}),
		
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
			gentest_fitness = run_GenTest(S,OpMode)
		},
		STATS = S#specie.stats,
		U_STATS = [STAT|STATS],
		mnesia:dirty_write(S#specie{stats=U_STATS}),
		STAT.

	run_GenTest(S,OpMode)->
		case lists:member(benchmark,OpMode) of
			true ->
				ChampionAgent_Id = case S#specie.champion_ids of%TODO:Update the manner in which champions are stored and added, currently, they are not deleted with the population... acumulate over time, are 3 at a time...
					[Id] ->
						Id;
					[Id|_] ->
						Id;
					[]->
						void
				end,
				case ChampionAgent_Id of
					void ->
						{0,void};
					_ ->
						ChampionAgent_PId=exoself:start(ChampionAgent_Id,self(),benchmark),
						receive
							{ChampionAgent_PId,benchmark_complete,Specie_Id,Fitness,Cycles,Time}->
								genotype:print(ChampionAgent_Id),
								{Fitness,ChampionAgent_Id}
							%Msg ->
							%	io:format("Msg:~p~n",[Msg])
						end
				end;
			false ->
				{0,void}
		end.

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
