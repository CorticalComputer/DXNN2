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

-module(polis). 
%% API 
-export([start/1,start/0,stop/0,init/2,create/0,reset/0,sync/0]). 
%% gen_server callbacks 
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,terminate/2, code_change/3]). 
-behaviour(gen_server). 
-include("records.hrl"). 
%%=========================================== Polis Configuration Options 
-record(state,{active_mods=[],active_scapes=[]}). 
-record(scape_summary,{address,type,parameters=[],metabolics,physics}).

-define(MODS,[]).
-define(PUBLIC_SCAPES,[
	#scape_summary{type=flatland,metabolics=static} %Public Forum
]). 
%The MODS list contains the names of the processes, functions, or other databases that also need to be executed and started when we start our neuroevolutionary platform. In the same manner, when we have created a new public scape, we can add a scape_summary tuple with this scape's information to the PUBLIC_SCAPES list, so that it is initialized and started with the system. The state record for the polis has all the elements needed to track the currently active mods and public scapes, which were either present during the startup of the neuroevolutionary platform, or latter added while the polis was already online.

%%=========================================== API 
sync()-> 
	make:all([native,load]).
% A sync/1 function can compile and reload all the modules pertaining to the project within the folder.

start() -> 
	case whereis(polis) of 
		undefined -> 
			gen_server:start(?MODULE, {?MODS,?PUBLIC_SCAPES}, []); 
		Polis_PId -> 
			io:format("Polis:~p is already running on this node.~n",[Polis_PId]) 
	end. 

start(Start_Parameters) -> 
	gen_server:start(?MODULE, Start_Parameters, []). 
init(Pid,InitState)-> 
	gen_server:cast(Pid,{init,InitState}). 
%The start/0 first checks whether a polis process has already been spawned, by checking if one is registered. If it's not, then the start/1 function starts up the neuroevolutionary platform.

stop()-> 
	case whereis(polis) of 
		undefined -> 
			io:format("Polis cannot be stopped, it is not online~n"); 
		Polis_PId -> 
			gen_server:cast(Polis_PId,{stop,normal})
	end. 
%The stop/0 function first checks whether a polis process is online. If there is an online polis process running on the node, then the stop function sends a signal to it requesting it to stop.
	 
%%============================================ gen_server callbacks 
init({Mods,PublicScapes}) -> 
	{A,B,C} = now(), 
	random:seed(A,B,C), 
	process_flag(trap_exit,true), 
	register(polis,self()), 
	io:format("Parameters:~p~n",[{Mods,PublicScapes}]), 
	mnesia:start(), 
	start_supmods(Mods), 
	Active_PublicScapes = start_scapes(PublicScapes,[]), 
	io:format("******** Polis: ##MATHEMA## is now online.~n"), 
	InitState = #state{active_mods=Mods,active_scapes=Active_PublicScapes}, 
	%Scape_PIdsP = [{Scape_PId,Scape_Type}...] 
	{ok, InitState}. 
%The init/1 function first seeds random with a new seed, in the case a random number generator will be needed. The polis process is then registered, the mnesia database is started, and the supporting modules, if any, are then started through the start_supmods/1 function. Then all the specified public scapes, if any, are activated. Having called our neuroevolutionary platform polis, we give this polis a name “MATHEMA”, which is a greek word for knowledge, and learning. Finally we create the initial state, which contains the Pids of the currently active public scapes, and the names of the activated mods. The function then drops into the main gen_server loop.

handle_call({get_scape,Type},{Cx_PId,_Ref},S)->
	Active_PublicScapes = S#state.active_scapes, 
	Scape_PId = case lists:keyfind(Type,3,Active_PublicScapes) of 
		false -> 
			undefined; 
		PS -> 
			PS#scape_summary.address
	end, 
	{reply,Scape_PId,S}; 
handle_call({stop,normal},_From, State)-> 
	{stop, normal, State}; 
handle_call({stop,shutdown},_From,State)-> 
	{stop, shutdown, State}. 
%At this point polis only accepts a get_scape call, to which it replies with the Pid or undefined message, and the two standard {stop,normal} and {stop,shutdown} calls.

handle_cast({init,InitState},_State)-> 
	{noreply,InitState}; 
handle_cast({stop,normal},State)-> 
	{stop, normal,State}; 
handle_cast({stop,shutdown},State)-> 
	{stop, shutdown, State}. 
%At this point polis allows only for 3 standard casts: {init,InitState}, {stop,normal} and {stop,shutdown}.

handle_info(_Info, State) ->
    {noreply, State}.
%The standard, still unused handle_info/2 function.

terminate(Reason, S) -> 
	Active_Mods = S#state.active_mods, 
	stop_supmods(Active_Mods), 
	stop_scapes(S#state.active_scapes),
	io:format("******** Polis: ##MATHEMA## is now offline, terminated with reason:~p~n",[Reason]), 
	ok. 

code_change(_OldVsn, State, _Extra) -> 
    {ok, State}. 
%When polis is terminated, it first shuts down all the scapes by calling stop_scapes/1, and then all the supporting mods, by calling stop_supmods/1.

%%-------------------------------------------------------------------- 
%%% Internal functions 
%%-------------------------------------------------------------------- 
create()-> 
	mnesia:create_schema([node()]), 
	mnesia:start(), 
	mnesia:create_table(population,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,population)}]), 
	mnesia:create_table(specie,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,specie)}]),
	mnesia:create_table(agent,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,agent)}]), 
	mnesia:create_table(cortex,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,cortex)}]), 
	mnesia:create_table(neuron,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,neuron)}]),
	mnesia:create_table(sensor,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,sensor)}]), 
	mnesia:create_table(actuator,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,actuator)}]),
	mnesia:create_table(substrate,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,substrate)}]),
	mnesia:create_table(experiment,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,experiment)}]).

reset()-> 
	mnesia:stop(), 
	ok = mnesia:delete_schema([node()]), 
	polis:create(). 
%The create/0 function sets up new mnesia databases composed of the dx, cortex, neuron, sensor, actuator, polis, and population, and specie tables. The reset/0 function deletes the schema, and recreates a fresh database from scratch.

%Start/Stop enviromental modules: DBs, Environments, Network Access systems and tools... 
start_supmods([ModName|ActiveMods])-> 
	ModName:start(), 
	start_supmods(ActiveMods); 
start_supmods([])-> 
	done. 
%The start_supmods/1 function expects a list of module names of the mods that are to be started with the startup of the neuroeovlutionary platform. Each module must have a start/0 function that starts up the supporting mod process.

stop_supmods([ModName|ActiveMods])-> 
	ModName:stop(), 
	stop_supmods(ActiveMods); 
stop_supmods([])-> 
	done. 
%The stop_supmods/1 expects a list of supporting mod names, the mod's name must be the name of its module, and that module must have a stop/0 function tat stops the module. stop_supmods/1 goes through the list of the mods, and stops each one.

start_scapes([S|Scapes],Acc)-> 
	Type = S#scape_summary.type, 
	Parameters = S#scape_summary.parameters,
	Physics = S#scape_summary.physics,
	Metabolics = S#scape_summary.metabolics,
	{ok,PId} = Type:start_link({self(),Type,Physics,Metabolics}),
	start_scapes(Scapes,[S#scape_summary{address=PId}|Acc]); 
start_scapes([],Acc)-> 
	lists:reverse(Acc). 
%The start_scapes/2 function accepts a list of scape_summary records, which specify the names of the public scapes and any parameters that with which those scapes should be started. What specifies what scape that is going to be created by the scape module is the Type that is dropped into the function. Ofcourse the scape module should already be able to create the Type of scape that is dropped into the start_link function. Once the scape is started, we record the Pid in that scape_summary's record. When all the public scapes have been started, the function outputs a list of updated scape_summary records.

stop_scapes([S|Scapes])-> 
	PId = S#scape_summary.address,
	gen_server:cast(PId,{self(),stop,normal}),
	stop_scapes(Scapes); 
stop_scapes([])-> 
	ok.
%The stop_scapes/1 function accepts a list of scape_summary records. The function extracts the Pid of the scape from the scape_summary, and requests for that scape to terminate itself.
