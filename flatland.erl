%% This source code and work is provided and developed by DXNN Research Group WWW.DXNNResearch.COM
%%
%Copyright (C) 2009 by Gene Sher, DXNN Research Group, CorticalComputer@gmail.com
%All rights reserved.
%
%This code is licensed under the version 3 of the GNU General Public License. Please see the LICENSE file that accompanies this project for the terms of use.

-module(flatland).
-compile(export_all).
%% API
%-export([start_link/1,start_link/0,start/1,start/0,init/2]).
%% gen_server callbacks
%-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

%-record(state, {}).
-behaviour(gen_server).
-include("records.hrl").
-define(NEURAL_COST,100).
-define(PLANT_AGELIMIT,2000).
-define(FL_AGELIMIT,10000).
-define(PLANT_GROWTH,off).
-define(SPAWN_LOC,[{1,[0,0]},{2,[2500,0]},{3,[5000,0]},{4,[5000,2500]},{5,[5000,5000]},{6,[2500,5000]},{7,[0,5000]},{8,[0,2500]}]).
-define(SECTOR_SIZE,10).
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
%-record(polis,{id,scape_ids=[],population_ids=[],specie_ids=[],dx_ids=[],parameters=[]}).
%-record(scape,{id,scape_id,type,physics,metabolics,avatars=[],plants=[],walls=[],laws=[],anomolies=[],artifacts=[],objects=[],elements=[],atoms=[],borders=[]}).
%-record(avatar,{id,morphology,type,team,energy,sound,gestalt,age=0,kills=0,loc,direction,r,objects,state,stats,actuators,sensors}).
init(Parameters) ->
	{A,B,C} = now(),
	random:seed(A,B,C),
	process_flag(trap_exit,true),
%	io:format("Scape Parameters:~p~n",[Parameters]),
	spawn(flatland,heartbeat,[self()]),
	InitState = case Parameters of
		{Polis_PId,Scape_Type,Physics,Metabolics} ->
			Init_Avatars = world_init(Scape_Type,Physics,Metabolics),
%			io:format("InitAvatars:~p~n",[InitAvatars]),
			#scape{id=self(),
				type = Scape_Type,
				avatars = Init_Avatars,
				physics = Physics,
				metabolics = Metabolics
			}
	end,
	{ok, InitState}.

%%--------------------------------------------------------------------
%% Function: %% handle_call(Request, From, State) -> {reply, Reply, State} |
%%                                      {reply, Reply, State, Timeout} |
%%                                      {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, Reply, State} |
%%                                      {stop, Reason, State}
%% Description: Handling call messages
%%--------------------------------------------------------------------
handle_call({actuator,Exoself_PId,Command,Output},{From_PId,_Ref},State)->
	%timer:sleep(10),
	%io:format("########################: ~p~n",[now()]),
	{FitnessP,U_State}=case get(Exoself_PId) of
		undefined ->
			io:format("Unregistered Citizen:~p~n",[Exoself_PId]),
			{{0,0},State};
		destroyed ->
			erase(Exoself_PId),
			io:format("Avatar:~p destroyed.~n",[Exoself_PId]),
			{{0,1},State};
		_ ->
			Avatars = State#scape.avatars,
			Avatar = lists:keyfind(Exoself_PId, 2, Avatars),
			U_Avatar = flatland:Command(Avatar#avatar{kills=0},Output),
			case (U_Avatar#avatar.energy > 0) and (U_Avatar#avatar.age < 20000) of
				true ->
					Age = U_Avatar#avatar.age,
					Fitness = case Age > 1000 of
						true ->
							0.001+Avatar#avatar.kills*1;
						false ->
							0.001+Avatar#avatar.kills*1
					end,
					{{Fitness,0},State#scape{avatars = collision_detection(U_Avatar,lists:keyreplace(Exoself_PId, 2, Avatars, U_Avatar))}};
				false ->
					io:format("Avatar:~p died at age:~p~n",[U_Avatar#avatar.id,U_Avatar#avatar.age]),
					{{0,1},destroy_avatar(Exoself_PId,State)}
			end
	end,
	{reply,FitnessP,U_State};
handle_call({multi_agent,update_agents,U_Avatars},{Exoself_PId,_Ref},State)->
	{reply,ok,State#scape{avatars=U_Avatars}};
handle_call({get_all,avatars,Exoself_PId},{From_PId,_Ref},State)->
	Reply =case get(Exoself_PId) of
		destroyed ->
			destroyed;
		_ ->
			State#scape.avatars
	end,
	{reply,Reply,State};
handle_call(tick,{From_PId,_Ref},State)->
	Avatars = State#scape.avatars,
	U_Avatars = flatland:metabolics(Avatars,[]),
	{reply,done,State#scape{avatars = U_Avatars}};
handle_call({enter,Morphology,Specie_Id,CF,CT,TotNeurons,Exoself_PId},{From_PId,_Ref},State)->
	{Reply,U_State}=case get(Exoself_PId) of
		entered ->
			io:format("Already Registered Citizen:~p~n",[Exoself_PId]),
			{undefined,State};
		undefined ->
			Stats = {CF,CT,TotNeurons},
			Avatars = State#scape.avatars,
			put(Exoself_PId,entered),
%			io:format("Avatar:~p entered~n",[Exoself_PId]),
			Avatar=case get(visor) of
				undefined ->
					create_avatar(Morphology,Specie_Id,Exoself_PId,Stats,void);
				{Visor_PId,Canvas} ->
					visor:draw_avatar(Canvas,create_avatar(Morphology,Specie_Id,Exoself_PId,Stats,void))
			end,
			io:format("New agent entered scape:~p~n",[Exoself_PId]),
			{done,State#scape{avatars = [Avatar|Avatars]}}
	end,
	{reply,Reply,U_State};
handle_call({leave,Exoself_PId},{From_PId,_Ref},State)->
	U_State=destroy_avatar(Exoself_PId,State),
%	io:format("Avatar left:~p~n",[Exoself_PId]),
	{reply,done,U_State};
handle_call(get_canvas,{From_PId,_Ref},State)->
	Reply=case get(visor) of
		{_Visor_PId,Canvas}->
			Canvas;
		undefined ->
			undefined
	end,
	{reply,Reply,State};
handle_call({Visor_PId,unsubscribe},{From_PId,_Ref},State)->
	erase(visor),
	{reply,done,State};
handle_call({stop,normal},_From, State)->
	{stop, normal, State};
handle_call({stop,shutdown},_From,State)->
	{stop, shutdown, State}.

%%--------------------------------------------------------------------
%% Function: handle_cast(Msg, State) -> {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, State}
%% Description: Handling cast messages
%%--------------------------------------------------------------------
handle_cast({Visor_PId,subscribe,Canvas},State)->
	put(visor,{Visor_PId,Canvas}),
	U_Avatars = visor:draw_avatars(Canvas,State#scape.avatars,[]),
	U_Objects = visor:draw_objects(Canvas,State#scape.objects,[]),
	io:format("Visor:~p subscribed with canvas:~p~n",[Visor_PId,Canvas]),
	{noreply,State#scape{avatars = U_Avatars,objects = U_Objects}};
handle_cast({Visor_PId,redraw,Filter},State)->
	%io:format("~p~n",[now()]),
	case get(visor) of
		undefined ->
			io:format("Scape:~p can't redraw, Visor:~p is not subscribed.~n",[State#scape.type,Visor_PId]);
		{Visor_PId,_Canvas}->
			visor:redraw_avatars(Filter,State#scape.avatars),
			visor:redraw_objects(Filter,State#scape.objects)
	end,
	{noreply,State};
handle_cast({Visor_PId,unsubscribe},State)->
	erase(visor),
	{noreply,State};
handle_cast(tick,State)->
%	Avatars = State#scape.avatars,
%	Scheduler = State#scape.scheduler,
%	io:format("Scape Type:~p Scheduler:~p~n",[State#scape.type,Scheduler]),
%	{U_Avatars,U_Scheduler}=case Scheduler of
%		10 ->
%			%Plant=create_avatar(plant,plant,genotype:generate_UniqueId(),void,no_respawn),
%			Plant=case get(visor) of
%				undefined ->
%					create_avatar(plant,plant,genotype:generate_UniqueId(),void,no_respawn);
%				{Visor_PId,Canvas} ->
%					visor:draw_avatar(Canvas,create_avatar(plant,plant,genotype:generate_UniqueId(),void,no_respawn))
%			end,
%			io:format("Plant:~p~n",[Plant]),
%			{[Plant|metabolics(Avatars,[])],0};
%		_ ->
%			{metabolics(Avatars,[]),Scheduler+1}
%	end,

	Avatars = State#scape.avatars,
	U_Avatars = flatland:metabolics(Avatars,[]),
	{noreply,State#scape{avatars = U_Avatars}};%#scape{avatars = U_Avatars,scheduler=U_Scheduler}};
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

%%--------------------------------------------------------------------
%% Function: terminate(Reason, State) -> void()
%% Description: This function is called by a gen_server when it is about to
%% terminate. It should be the opposite of Module:init/1 and do any necessary
%% cleaning up. When it returns, the gen_server terminates with Reason.
%% The return value is ignored.
%%--------------------------------------------------------------------
terminate(Reason, State) ->
	io:format("Scape:~p~n Terminating with reason:~p~n",[self(),Reason]),
	ok.

%%--------------------------------------------------------------------
%% Func: code_change(OldVsn, State, Extra) -> {ok, NewState}
%% Description: Convert process state when code is changed
%%--------------------------------------------------------------------
code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%--------------------------------------------------------------------
%%% Internal functions
%%--------------------------------------------------------------------
heartbeat(Scape_PId)->
	heartbeat(Scape_PId,100,0).
heartbeat(Scape_PId,Tau,Time)->
	receive 
		{update_tau,NewTau}->
			flatland:heartbeat(Scape_PId,Tau,Time)
	after Tau ->
		gen_server:cast(Scape_PId,tick),
		flatland:heartbeat(Scape_PId,Tau,Time+Tau)
	end.

metabolics([Avatar|Avatars],Acc)->
%	io:format("Avatar:~p~n",[Avatar]),
	case Avatar#avatar.type of
		plant ->
			case Avatar#avatar.state of
				no_respawn->
					case Avatar#avatar.age < 1000 of
						true ->
							U_Avatar=ripen(Avatar),
							metabolics(Avatars,[U_Avatar|Acc]);
						false ->
							case get(visor) of
								undefined->
									done;
								{_Visor_PId,_Canvas} ->
									[gs:destroy(Id) || {_ObjType,Id,_Color,_Pivot,_Coords,_Parameter} <- Avatar#avatar.objects]
							end,
							metabolics(Avatars,Acc)
					end;
				respawn ->
					%RespawnedAvatar = respawn_avatar(Avatar),
					metabolics(Avatars,[Avatar|Acc])
			end;	
		predator ->
			Energy = Avatar#avatar.energy,
			%io:format("Predator:~p~n",[Energy]),
			U_Avatar=Avatar#avatar{energy = Energy -0.01},
			metabolics(Avatars,[U_Avatar|Acc]);
		prey ->
			Energy = Avatar#avatar.energy,
			%io:format("Prey:~p~n",[Energy]),
			U_Avatar=Avatar#avatar{energy = Energy -0.01},
			metabolics(Avatars,[U_Avatar|Acc]);
		_ ->
			metabolics(Avatars,[Avatar|Acc])
	end;
metabolics([],Acc)->
%	io:format("Time:~p Avatar_Count:~p~n",[now(),length(Acc)]),
	Acc.
	
	ripen(Avatar)->
		%io:format("Here~n"),
		Energy = Avatar#avatar.energy,
		Age = Avatar#avatar.age,
		New_Color=case Energy of
			-2000 -> black;
			-500 -> cyan;
			0 -> grey;
			500 -> green;
			1300 -> yellow;
			1500 -> white;
			_ -> no_change
		end,
		%io:format("Plant:~p~n",[Energy]),
		case New_Color of
			no_change ->
				%io:format("not ripe:~n"),
				Avatar#avatar{energy = functions:saturation(Energy+2,1000),age = Age+1};
			_ ->
				%io:format("ripe~n"),
				U_Energy = functions:saturation(Energy+2,1000),
			     Avatar#avatar{energy=U_Energy,age = Age+1,objects=[{circle,Id,New_Color,Loc,Coords,R}||{circle,Id,_Color,Loc,Coords,R}<-Avatar#avatar.objects]}
		end.
	
new_loc()->
	X = random:uniform(5000),
	Y = random:uniform(5000),
	{X,Y}.
	
new_loc(XMin,XMax,YMin,YMax)->
	X = random:uniform(XMax-XMin)+XMin,
	Y = random:uniform(YMax-YMin)+YMin,
	{X,Y}.
	
remove(Avatar_PId,Avatar_PIdsP,Acc)->
	void.

check_borders(Avatar,[{XMin,XMax},{YMin,YMax}])->
	{X,Y} = Avatar#avatar.loc,
	R = Avatar#avatar.r,
	DX = if
		(X-R) < XMin -> XMin-(X-R);
		(X+R) > XMax -> XMax-(X+R);
		true -> 0
	end,
	DY = if
		(Y-R) < YMin -> YMin-(Y-R);
		(Y+R) > YMax -> YMax-(Y+R);
		true -> 0
	end,
	case {DX,DY} of
		{0,0} ->
			Avatar;
		{DX,DY} ->
			U_Objects=[{ObjName,Id,Color,{PX+DX,PY+DY},[{X+DX,Y+DY}||{X,Y}<-Coords],P}||{ObjName,Id,Color,{PX,PY},Coords,P}<-Avatar#avatar.objects],
			Avatar#avatar{loc = {X+DX,Y+DY},objects = U_Objects}
	end.

collision_detection(OperatorAvatar,Avatars)->
	collision_detection(OperatorAvatar,0,0,Avatars,[]).
collision_detection(OperatorAvatar,EnergyAcc,Kills,[Avatar|Avatars],Acc)->
	%io:format("OAvatar:~p Avatar:~p~n",[OperatorAvatar,Avatar]),
	if
		(Avatar#avatar.id == OperatorAvatar#avatar.id)->
			collision_detection(OperatorAvatar,EnergyAcc,Kills,Avatars,[Avatar|Acc]);
		(Avatar#avatar.type == wall) ->
			U_OperatorAvatar = world_wall_collision(OperatorAvatar,Avatar),
			collision_detection(U_OperatorAvatar,EnergyAcc,Kills,Avatars,[Avatar|Acc]);
		true ->
			{X,Y}= OperatorAvatar#avatar.loc,
			{DX,DY} = OperatorAvatar#avatar.direction,
			{Xav,Yav}=Avatar#avatar.loc,
			Penetration=case (OperatorAvatar#avatar.type == predator) or (OperatorAvatar#avatar.spear == true) of
				true ->
					Spear_UnitRay = {DX*math:cos(0) - DY*math:sin(0), DX*math:sin(0) + DY*math:cos(0)},
					{InterDist,_Color}=sensor:shortest_intrLine({{X,Y},Spear_UnitRay},[Avatar],{inf,void}),
					(InterDist =/= -1) and (InterDist < (2 + OperatorAvatar#avatar.r));%TODO, spear length should be avatar defined, not 2.
				false ->
					false
			end,
			Distance = math:sqrt(math:pow(X-Xav,2)+math:pow(Y-Yav,2)),
			Collision = (Distance < (OperatorAvatar#avatar.r+Avatar#avatar.r)),
			{Energy,Order,U_OperatorAvatar,U_Avatar}= case Collision or Penetration of
				true ->
					world_behavior(Collision,Penetration,OperatorAvatar,Avatar);
				false ->
					{0,void,OperatorAvatar,Avatar}
			end,
			case Order of
				destroy ->
					case get(visor) of
						undefined->
							done;
						{_Visor_PId,_Canvas} ->
							[gs:destroy(Id) || {_ObjType,Id,_Color,_Pivot,_Coords,_Parameter} <- U_Avatar#avatar.objects]
					end,
					put(U_Avatar#avatar.id,destroyed),
					collision_detection(U_OperatorAvatar,EnergyAcc+Energy,Kills+1,Avatars,Acc);
				plant_eaten->
					Kill_Score = case Energy > 0 of
						true ->
							1;
						false ->
							0
					end,
					case U_Avatar#avatar.state of
						no_respawn->
							case get(visor) of
								undefined->
									done;
								{_Visor_PId,_Canvas} ->
									[gs:destroy(Id) || {_ObjType,Id,_Color,_Pivot,_Coords,_Parameter} <- U_Avatar#avatar.objects]
							end,
							collision_detection(U_OperatorAvatar,EnergyAcc+Energy,Kills+Kill_Score,Avatars,Acc);
						respawn ->
							RespawnedAvatar = respawn_avatar([Avatar|Avatars]++Acc,U_Avatar),
							collision_detection(U_OperatorAvatar,EnergyAcc+Energy,Kills+Kill_Score,Avatars,[RespawnedAvatar|Acc])
					end;
				poison_eaten->
					case U_Avatar#avatar.state of
						no_respawn->
							case get(visor) of
								undefined->
									done;
								{_Visor_PId,_Canvas} ->
									[gs:destroy(Id) || {_ObjType,Id,_Color,_Pivot,_Coords,_Parameter} <- U_Avatar#avatar.objects]
							end,
							collision_detection(U_OperatorAvatar,EnergyAcc+Energy,Kills,Avatars,Acc);
						respawn ->
							RespawnedAvatar = respawn_avatar([Avatar|Avatars]++Acc,U_Avatar),
							collision_detection(U_OperatorAvatar,EnergyAcc+Energy,Kills,Avatars,[RespawnedAvatar|Acc])
					end;
				void ->
					collision_detection(U_OperatorAvatar,EnergyAcc+Energy,Kills,Avatars,[U_Avatar|Acc])
			end
	end;
collision_detection(OperatorAvatar,EnergyAcc,KillsAcc,[],Acc)->
	case EnergyAcc =/= 0 of
		true ->
			Energy = OperatorAvatar#avatar.energy,
			Kills = OperatorAvatar#avatar.kills,
			U_OperatorAvatar = OperatorAvatar#avatar{energy=functions:saturation(Energy+EnergyAcc,10000), kills = KillsAcc+Kills},
			lists:keyreplace(U_OperatorAvatar#avatar.id, 2, Acc, U_OperatorAvatar);
		false ->
			lists:keyreplace(OperatorAvatar#avatar.id, 2, Acc, OperatorAvatar)
	end.

create_avatar(Morphology)->
	create_avatar(Morphology,Morphology,genotype:generate_UniqueId(),{cf,ct,-1},respawn,undefined).
create_avatar(Morphology,Specie_Id)->
	Stats = {cf,ct,-1},
	create_avatar(Morphology,Specie_Id,genotype:generate_UniqueId(),Stats,respawn,undefined).
create_avatar(Morphology,Specie_Id,Id,Stats,Parameters)->
	create_avatar(Morphology,Specie_Id,Id,Stats,Parameters,undefined).
create_avatar(Morphology,Specie_Id,Id,{CF,CT,TotNeurons},void,InitEnergy) when (Morphology == predator)  or (Morphology == prey) or (Morphology == automaton)->
	case Morphology of
		predator->
			io:format("Creating Predator:~p~n",[{CF,CT,Id}]),
			%{CF,CT,TotNeurons} = Stats,
			Color = red,
			%Color=visor:ct2color(CT),
			Loc = {X,Y} = {random:uniform(400)+300,random:uniform(400)+300},
			Direction ={DX,DY} = {-1/math:sqrt(2),-1/math:sqrt(2)},
			%Loc = {X,Y} = {random:uniform(round(XMax/2)) + XMax/2 -R,random:uniform(round(YMax/2))+YMax/2 -R},
			Energy =case InitEnergy of
				undefined -> 
					1000;
				InitEnergy ->
					InitEnergy
			end,
			Metabolic_Package = static,
			case Metabolic_Package of
				static ->
					Mass = 6,
					R = 6;
				_ ->
					Mass = 6+Energy/1000,
					R = math:sqrt(Mass*10)
			end,
			Objects = [{circle,undefined,Color,{X,Y},[{X,Y}],R},{line,undefined,red,{X,Y},[{X,Y},{X+DX*R*2,Y+DY*R*2}],void}],
			#avatar{
				id = Id,
				type = Morphology,
				energy = Energy,
				loc = Loc,
				direction = Direction,
				r = R,
				mass = Mass,
				objects = Objects,
				actuators = CF,
				sensors =CT,
				stats = TotNeurons
			};
		prey ->
			io:format("Creating Prey:~p~n",[{CF,CT,Id}]),
			%{CF,CT,TotNeurons} = Stats,
			Direction = {DX,DY} = {1/math:sqrt(2),1/math:sqrt(2)},
			{X,Y} = {random:uniform(800),random:uniform(500)},
			Energy =case InitEnergy of
				undefined -> 
					1000;
				InitEnergy ->
					InitEnergy
			end,
			Metabolic_Package = static,
			case Metabolic_Package of
				static ->
					Mass = 10,
					R = 10;
				_ ->
					Mass = 10+Energy/1000,
					R = math:sqrt(Mass*10)
			end,
			case lists:keymember(spear,2,CF) of
				true ->
					Color = red,
					Objects = [{circle,undefined,Color,{X,Y},[{X,Y}],R},{line,undefined,Color,{X,Y},[{X,Y},{X+DX*R*2,Y+DY*R*2}],void}];
				false ->
					Color = blue,
					Objects = [{circle,undefined,Color,{X,Y},[{X,Y}],R}]
			end,
			#avatar{
				id = Id,
				type = Morphology,
				energy = Energy,
				loc = {X,Y},
				direction = Direction,
				r = R,
				mass = Mass,
				objects = Objects,
				actuators = CF,
				sensors =CT,
				stats = TotNeurons
			};
		automaton ->
			Angle = random:uniform()*2*math:pi(),
			Direction = {DX,DY}={(1/math:sqrt(2))*math:cos(Angle) - (1/math:sqrt(2))*math:sin(Angle),(1/math:sqrt(2))*math:sin(Angle) + (1/math:sqrt(2))*math:cos(Angle)},
			{X,Y} = {400+random:uniform(1120),200+random:uniform(580)},
			Mass = 10,
			R = 10,
			Color = blue,
			Objects = [{circle,undefined,Color,{X,Y},[{X,Y}],R}],
			#avatar{
				id = Id,
				type = Morphology,
				loc = {X,Y},
				direction = Direction,
				r = R,
				mass = Mass,
				objects = Objects
			}
	end;
create_avatar(Morphology,Specie_Id,Avatar_Id,{InitEnergy,InitLoc},RespawnFlag,Metabolics) when (Morphology == plant) or (Morphology == poison) ->
	case Morphology of
		plant ->
			io:format("Creating Plant~n"),
			Direction={1/math:sqrt(2),1/math:sqrt(2)},
			%{X,Y} = Loc = {random:uniform(5000),random:uniform(5000)},
			case InitLoc of
				undefined ->
					{X,Y} = Loc = {random:uniform(800),random:uniform(500)};
				Val ->
					{X,Y} = Loc = Val
			end,
			Energy =case InitEnergy of
				undefined -> 
					500;
				InitEnergy ->
					InitEnergy
			end,
			case Metabolics of
				static ->
					Mass = 3,
					R = 3;
				_ ->
					Mass = 3+Energy/1000,
					R = math:sqrt(Mass*3)
			end,
			%Objects = [{line,undefined,green,{0+X,0+Y},[{-R+X,-R+Y},{R+X,R+Y}],void},{line,undefined,green,{0+X,0+Y},[{-R+X,R+Y},{R+X,-R+Y}],void}],
			Objects = [{circle,undefined,green,{X,Y},[{X,Y}],R}],
			#avatar{
				id = Avatar_Id,
				type = Morphology,
				energy = Energy,
				food = 0,
				health = 0,
				mass = Mass,
				loc = Loc,
				direction = Direction,
				r = R,
				objects = Objects,
				state = RespawnFlag %[respawn| no_respawn]
			};
		poison ->
			io:format("Creating Poison~n"),
			Direction={1/math:sqrt(2),1/math:sqrt(2)},
			case InitLoc of
				undefined ->
					{X,Y} = Loc = {random:uniform(800),random:uniform(500)};
				Val ->
					{X,Y} = Loc = Val
			end,
			Energy =case InitEnergy of
				undefined -> 
					-2000;
				InitEnergy ->
					InitEnergy
			end,
			case Metabolics of
				static ->
					Mass = 3,
					R = 3;
				_ ->
					Mass = 3+abs(Energy)/1000,
					R = math:sqrt(Mass*3)
			end,
			%Objects = [{line,undefined,green,{0+X,0+Y},[{-R+X,-R+Y},{R+X,R+Y}],void},{line,undefined,green,{0+X,0+Y},[{-R+X,R+Y},{R+X,-R+Y}],void}],
			Objects = [{circle,undefined,black,{X,Y},[{X,Y}],R}],
			#avatar{
				id = Avatar_Id,
				type = Morphology,
				energy = Energy,
				loc = Loc,
				direction = Direction,
				r = R,
				objects = Objects,
				state = RespawnFlag
			}
	end;
create_avatar(Morphology,Specie_Id,Id,undefined,Parameters,undefined) when (Morphology == rock) or (Morphology == wall) or (Morphology == fire_pit) or (Morphology == beacon)->
	case Morphology of
		rock ->
			io:format("Creating Rock~n"),
			Direction={1/math:sqrt(2),1/math:sqrt(2)},
			{X,Y,R,Energy} = Parameters,
			Objects = [{circle,undefined,brown,{X,Y},[{X,Y}],R}],
			#avatar{
				id = Id,
				type = Morphology,
				energy = Energy,
				loc = {X,Y},
				direction = Direction,
				r = R,
				objects = Objects
			};
		wall ->
			io:format("Creating Wall~n"),
			case Parameters of
				{x_wall,{Y,XMin,XMax}}->
					YMin = YMax = Y;
				{y_wall,{X,YMin,YMax}}->
					XMin = XMax = X
			end,
			Pivot = {(XMin+XMax)/2,(YMin+YMax)/2},
			Objects = [{line,undefined,brown,Pivot,[{XMin,YMin},{XMax,YMax}],void}],
			#avatar{
				id = Id,
				type = Morphology,
				energy = 10000,
				loc = void,
				direction = void,
				r = void,
				objects = Objects,
				state = Parameters
			};
		pilar ->
			{Loc,R} = Parameters,
			Objects = [{circle,undefined,green,Loc,[Loc],R}],
			#avatar{
				id = Id,
				type = Morphology,
				loc = Loc,
				direction = void,
				r = R,
				objects = Objects,
				state = Parameters
			};
		fire_pit ->
			io:format("Creating FirePit~n"),
			Direction={1/math:sqrt(2),1/math:sqrt(2)},
			{X,Y,R,Energy} = Parameters,
			Objects = [{circle,undefined,red,{X,Y},[{X,Y}],R}],
			#avatar{
				id = Id,
				type = Morphology,
				energy = Energy,
				loc = {X,Y},
				direction = Direction,
				r = R,
				objects = Objects
			};
		beacon ->
			io:format("Creating Beacon~n"),
			Direction={1/math:sqrt(2),1/math:sqrt(2)},
			{X,Y,R,Energy} = Parameters,
			Objects = [{circle,undefined,white,{X,Y},[{X,Y}],R}],
			#avatar{
				id = beacon,
				type = Morphology,
				energy = Energy,
				loc = {X,Y},
				direction = Direction,
				r = R,
				objects = Objects
			}
	end.

destroy_avatar(ExoSelf_PId,State)->
	Avatars = State#scape.avatars,
	case get(ExoSelf_PId) of
		undefined ->
			io:format("Destroy Avatar in Scape:: Undefined:~p~n",[ExoSelf_PId]);
		entered->io:format("Destroying avatar associated with:~p~n",[ExoSelf_PId]),
			Avatar = lists:keyfind(ExoSelf_PId, 2, Avatars),
			erase(ExoSelf_PId),
			%io:format("Visor:~p~n",[get(visor)]),
			case get(visor) of
				undefined->
					done;
				{Visor_PId,_Canvas} ->
					io:format("Avatar:~p~n",[Avatar]),
					[gs:destroy(Id) || {_ObjType,Id,_Color,_Pivot,_Coords,_Parameter} <- Avatar#avatar.objects]
			end
	end,
	State#scape{avatars = lists:keydelete(ExoSelf_PId, 2, Avatars)}.

respawn_avatar(A)->
%	io:format("Respawning:~p~n",[A]),
	{X,Y} = {random:uniform(800),random:uniform(500)},
	case A#avatar.type of
		plant ->
			A#avatar{
				loc = {X,Y},
				energy = 500,
				objects = [{circle,Id,green,{X,Y},[{X,Y}],R} ||{circle,Id,_Color,{OldX,OldY},[{OldX,OldY}],R}<-A#avatar.objects]
			};
		poison ->
			A#avatar{
				loc = {X,Y},
				energy = -2000,
				objects = [{circle,Id,black,{X,Y},[{X,Y}],R} ||{circle,Id,_Color,{OldX,OldY},[{OldX,OldY}],R}<-A#avatar.objects]
			}
	end.

respawn_avatar(Avatars,A)->
%	io:format("Respawning:~p~n",[A]),
	%{X,Y} = {random:uniform(800),random:uniform(800)},
	OAvatars = [A || A <- Avatars, (A#avatar.type==rock) or (A#avatar.type==pillar) or (A#avatar.type==fire_pit)],
	{X,Y} = return_valid(OAvatars),
	case A#avatar.type of
		plant ->
			A#avatar{
				loc = {X,Y},
				energy = 500,
				objects = [{circle,Id,green,{X,Y},[{X,Y}],R} ||{circle,Id,_Color,{OldX,OldY},[{OldX,OldY}],R}<-A#avatar.objects]
			};
		poison ->
			A#avatar{
				loc = {X,Y},
				energy = -2000,
				objects = [{circle,Id,black,{X,Y},[{X,Y}],R} ||{circle,Id,_Color,{OldX,OldY},[{OldX,OldY}],R}<-A#avatar.objects]
			}
	end.

	return_valid(OAvatars)->
		case return_valid(OAvatars,{random:uniform(800),random:uniform(500)}) of
			undefined ->
				return_valid(OAvatars,{random:uniform(800),random:uniform(500)});
			Loc ->
				Loc
		end.
	
		return_valid([OA|OAvatars],{X,Y})->
			{Xav,Yav} = OA#avatar.loc,
			Distance = math:sqrt(math:pow(X-Xav,2)+math:pow(Y-Yav,2)),
			Collision = (Distance < (OA#avatar.r+2)),
			case Collision of
				true ->
					undefined;
				false ->
					return_valid(OAvatars,{X,Y})
			end;
		return_valid([],{X,Y})->
			{X,Y}.

move(Avatar,S)->
	{LX,LY} = Avatar#avatar.loc,
	{DX,DY} = Avatar#avatar.direction,
	Speed=case Avatar#avatar.type of
		prey ->
			S;
		_ ->
			S*0.9
	end,
	Energy = Avatar#avatar.energy,
	TotNeurons = Avatar#avatar.stats,
	U_Energy = Energy - 0.1*(math:sqrt(math:pow(DX*Speed,2)+math:pow(DY*Speed,2)))-0.1,%TODO
	%io:format("self():~p Energy burned:~p~n",[self(),abs(Speed)]),
	U_Loc = {LX+(DX*Speed),LY+(DY*Speed)},
	U_Objects=[{ObjName,Id,C,{PX+(DX*Speed),PY+(DY*Speed)},[{X+(DX*Speed),Y+(DY*Speed)}||{X,Y}<-Coords],P}||{ObjName,Id,C,{PX,PY},Coords,P}<-Avatar#avatar.objects],
	Avatar#avatar{energy = U_Energy,loc = U_Loc,objects=U_Objects}.
	
translate(Avatar,{DX,DY})->
	{LX,LY} = Avatar#avatar.loc,
	Energy = Avatar#avatar.energy,
	TotNeurons = Avatar#avatar.stats,
	U_Energy = Energy - 0.1*(math:sqrt(math:pow(DX,2),math:pow(DY,2))) - 0.1,%TODO
	U_Loc = {LX+DX,LY+DY},
	U_Objects=[{ObjName,Id,Color,{PX+DX,PY+DY},[{X+DX,Y+DY}||{X,Y}<-Coords],P}||{ObjName,Id,Color,{PX,PY},Coords,P}<-Avatar#avatar.objects],
	Avatar#avatar{loc = U_Loc,energy = U_Energy,objects=U_Objects}.
	
rotate(Avatar,A)->
	Ratio=math:pi()/4,
	Angle = A*Ratio,
	{DX,DY} =Avatar#avatar.direction,
	Energy = Avatar#avatar.energy,
	TotNeurons = Avatar#avatar.stats,
	U_Energy = Energy  - 0.1*(abs(Angle))-0.1,%TODO
	U_Direction = {DX*math:cos(Angle) - DY*math:sin(Angle),DX*math:sin(Angle) + DY*math:cos(Angle)},
	U_Objects = rotation(Avatar#avatar.objects,Angle,[]),
	Avatar#avatar{energy = U_Energy,direction=U_Direction,objects=U_Objects}.
	
	rotation([Object|Objects],Angle,Acc)->
		{ObjName,Id,Color,{PX,PY},Coords,Parameter} = Object,
		U_Coords = [{(X-PX)*math:cos(Angle) - (Y-PY)*math:sin(Angle) + PX, (X-PX)*math:sin(Angle) + (Y-PY)*math:cos(Angle) +PY} || {X,Y} <- Coords],
		U_Object = {ObjName,Id,Color,{PX,PY},U_Coords,Parameter},
		rotation(Objects,Angle,[U_Object|Acc]);
	rotation([],_Angle,Acc)->
		Acc.

	move_and_rotate(Avatar,[Speed,Angle])->
		Moved_Avatar = move(Avatar,Speed),
		Rotated_Avatar = rotate(Moved_Avatar,Angle).
		
	rotate_and_move(Avatar,[Speed,Angle])->
		Rotated_Avatar = rotate(Avatar,Angle),
		Moved_Avatar = move(Rotated_Avatar,Speed).
		
	rotate_and_translate(Avatar,[Translation,Angle])->
		Rotated_Avatar=rotate(Avatar,Angle),
	Translated_Avatar = translate(Rotated_Avatar,Translation).
	
two_wheels(Avatar,[SWheel1,SWheel2])->
	{Speed,Angle}=twowheel_to_moverotate(SWheel1,SWheel2),
	Rotated_Avatar = rotate(Avatar,Angle),
	Moved_Avatar = move(Rotated_Avatar,Speed),
	AgeAcc = Moved_Avatar#avatar.age,
	Moved_Avatar#avatar{age = AgeAcc+1}.
	
	twowheel_to_moverotate(Wr,Wl)->
		Speed = (Wr + Wl)/2,
		Angle = Wr - Wl,
		{Speed,Angle}.
		
	differential_drive(Wr,Wl)->
		R = 1,
		L = 1,
		differential_drive(R,L,Wr,Wl).
	differential_drive(R,L,Wr,Wl)->
		Uw = (Wr+wl)/2,
		Ua = (Wr-Wl),
		DTheta = (R/L)*Ua,
		Theta = DTheta*1,
		DX = R*Uw*math:cos(Theta),
		DY = R*Uw*math:sin(Theta),
		{DX,DY}.
		
speak(Avatar,[Val])->
%	io:format("Avatar:~p Speak:~p~n",[Avatar#avatar.id,Val]),
	Avatar#avatar{sound=Val}.

gestalt_output(Avatar,Gestalt)->
	Avatar#avatar{gestalt=Gestalt}.

create_offspring(Avatar,[ExoSelf_PId,CreateVal])->
	case CreateVal > 0 of
		true ->
			OffspringCost = Avatar#avatar.stats*?NEURAL_COST,
			Energy = Avatar#avatar.energy,
			case Energy > (OffspringCost+1000) of
				true ->io:format("Avatar Energy:~p OffspringCost:~p~n",[Energy,OffspringCost]),
					gen_server:cast(ExoSelf_PId,{self(),mutant_clone,granted,Avatar#avatar.id}),
					Avatar#avatar{energy=Energy-(OffspringCost+1000)};
				false ->
					Avatar#avatar{energy=Energy-50}
			end;
		false ->
			Avatar
	end.
	
spear(Avatar,[Val])->
	case Val > 0 of
		true ->
			Energy = Avatar#avatar.energy,
			case Energy > 100 of
				true ->
					Avatar#avatar{energy=Energy-10,spear=true};
				false ->
					Avatar#avatar{energy=Energy-1,spear=false}
			end;
		false ->
			Avatar#avatar{spear=false}
	end.
	
shoot(Avatar,[Val])->
	case Val > 0 of
		true ->
			Energy = Avatar#avatar.energy,
			case Energy > 100 of
				true ->
					shoot_50,%draw when shot is fired, different color from spear
					Avatar#avatar{energy=Energy-20};
				false ->
					Avatar#avatar{energy=Energy-1}
			end;
		false ->
			Avatar
	end.

%Sectors are 10 by 10.
%[{3,3}] XSec = X div 10, YSec = Y div 10...
put_sector(Coord)->
	void.
get_sector(Coord)->
	void.
loc2sector({X,Y})->
	{trunc(X/?SECTOR_SIZE),trunc(Y/?SECTOR_SIZE)}.
	
command_generator()->
	%quiet time in steps (100-1000)
	%A sequence of commands is generated
	%generated commands are set to sensors, and calcualted fitness sent to actuators
	void.
	
obedience_fitness(Command,Beacon)->
	%Fitness = ImportanceA*CommandObedience - ImportanceB*CollisionAvoidenceFirePit - ImportanceC*CollisionAvoidencePillar - ImportanceC*CollisionAvoidenceWall;
	%1/abs(Command-DrivingDecision)=CommandObedience, CollisionAvoidence = FirePitIntersectionCount, PIllarIntersectionCount, WallIntersectionCount
	%BEACON:
	%distance to beacon, angle to beacon, 1/abs(DistanceFromBeacon-RequestedDistance)
	void.
	
epitopes()->
	spawn(flatland,db,[]).
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






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-define(COLLISIONS,on).

world_init(World_Type,Physics,Metabolics)->
	XMin = -5000,
	XMax = 5000,
	YMin = -5000,
	YMax = 5000,
	WorldPivot = {(XMin+XMax)/2,(YMin+YMax)/2},
	World_Border = [{XMin,XMax},{YMin,YMax}],

	case World_Type of
		duel ->	
			Id=genotype:generate_UniqueId(),
			Plants=[create_avatar(plant,plant,Id,{undefined,undefined},respawn,Metabolics)|| _<-lists:duplicate(10,1)],
			Walls = lists:append(create_walls(),create_pillars()),
			Scape_Physics = [],
			Plants++Walls;
		hunt ->
			[];
		dangerous_hunt ->
			[];
		flatland ->
			Walls = lists:append(create_walls(),create_pillars()),
			Rocks = create_rocks(),
			FirePits = create_firepits(),
			Beacons = create_beacons(),
			Scape_Physics = [],
			%io:format("Plants:~p~n",[Plants]),
			Plants=[create_avatar(plant,plant,gen_id(),{undefined,return_valid(Rocks++FirePits)},respawn,Metabolics)||_<-lists:duplicate(10,1)],
			Poisons=[create_avatar(poison,poison,gen_id(),{undefined,return_valid(Rocks++FirePits)},respawn,Metabolics)||_<-lists:duplicate(10,1)],
			Plants;%++Rocks++Walls++Poisons++FirePits++Beacons;
		dynamic ->
			[];
		baator ->
			Id=genotype:generate_UniqueId(),
			Plants=[create_avatar(plant,plant,Id,{undefined,undefined},respawn,Metabolics)|| _<-lists:duplicate(10,1)],
			Poisons=[create_avatar(poison,poison,Id,{undefined,undefined},respawn,Metabolics)|| _<-lists:duplicate(10,1)],
			Walls = lists:append(create_walls(),create_pillars()),
			Scape_Physics = [],
			Plants++Walls;
		multi_agent ->
			[]
	end.

world_behavior(Collision,Penetration,OAvatar,Avatar)->
	OAType = OAvatar#avatar.type,
	AType = Avatar#avatar.type,
	if
		(OAType == prey) and (AType == plant) and (OAvatar#avatar.spear =/= undefined)->%OAvatar is a predator
			%io:format("Prey: ~p ate a plant: ~p gained energy:~p~n",[OAvatar#avatar.id,Avatar#avatar.id,Avatar#avatar.energy*0.2]),
			{Avatar#avatar.energy*0.2,plant_eaten,OAvatar,Avatar};
		(OAType == prey) and (AType == plant) ->
			%io:format("Prey: ~p ate a plant: ~p~n",[OAvatar#avatar.id,Avatar#avatar.id]),
			{Avatar#avatar.energy,plant_eaten,OAvatar,Avatar};
		(OAType == prey) and (AType == poison) ->
			%io:format("Prey: ~p ate a plant: ~p~n",[OAvatar#avatar.id,Avatar#avatar.id]),
			{Avatar#avatar.energy,poison_eaten,OAvatar,Avatar};
		(OAType == prey) and (AType == prey) and (Penetration == true) and (Avatar#avatar.spear == undefined)->%OAvatar spears Avatar who is a prey
			{500,destroy,OAvatar,Avatar};
		(OAType == prey) and (AType == prey) and (Penetration == true) and (Avatar#avatar.spear =/= undefined)->%OAvatar/predator spears another predator
			io:format("########Predator killed another predator~n"),
			{100,destroy,OAvatar,Avatar};
		(OAType == prey) and (AType == prey) and (Collision == true)->
			U_Avatar=case ?COLLISIONS of
				on ->
					PushStrength = 0.1,%Push
					push(OAvatar,Avatar,PushStrength);
				off ->
					Avatar
			end,
			{0,void,OAvatar,U_Avatar};
		(OAType == predator) and (AType == prey) and (Penetration == true)->
%			io:format("Hunter: ~p ate a Prey: ~p~n",[OAvatar#avatar.id,Avatar#avatar.id]),
			{500,destroy,OAvatar,Avatar};
		(OAType == predator) and (AType == prey) and (Collision == true)->
			U_Avatar=case ?COLLISIONS of
				on ->
					PushStrength = 1,%Push_Hard
					push(OAvatar,Avatar,PushStrength);
				off ->
					Avatar
			end,
			{0,void,OAvatar,U_Avatar};
		(OAType == predator) and (AType == predator)  and (Penetration == true)->
			U_Avatar=case ?COLLISIONS of
				on ->
					PushStrength = 1,%Push_Very_Hard %TODO, pushing must be done from the tip of the sword.
					push(OAvatar,Avatar,PushStrength);
				off ->
					Avatar
			end,
			{0,void,OAvatar,U_Avatar};
		(OAType == predator) and (AType == predator)  and (Collision == true)->
			U_Avatar=case ?COLLISIONS of
				on ->
					PushStrength = 0.1,%Push_Hard
					push(OAvatar,Avatar,PushStrength);
				off ->
					Avatar
			end,
			{0,void,OAvatar,U_Avatar};
		(OAType == predator) and ((AType == plant) or (AType == poison)) and (Collision == true)->
			U_Avatar=case ?COLLISIONS of
				on ->
					PushStrength = 0,%Push_asside
					push(OAvatar,Avatar,PushStrength);
				off ->
					Avatar
			end,
			{0,void,OAvatar,U_Avatar};
		(AType == rock) ->
			case ?COLLISIONS of
				on ->
					case OAvatar#avatar.energy > Avatar#avatar.energy of
						true ->%TODO
							PushStrength = 1,%Push
							U_Avatar = push(OAvatar,Avatar,PushStrength),
							{-1,void,OAvatar,U_Avatar};
						false ->
							PushStrength = 0,
							U_OAvatar=push(Avatar,OAvatar,PushStrength),
							{-1,void,U_OAvatar,Avatar}
					end;
				off ->
					{0,void,OAvatar,Avatar}
			end;
		(AType == fire_pit) ->
			case ?COLLISIONS of
				on ->
					case OAvatar#avatar.energy > Avatar#avatar.energy of
						true ->%TODO
							PushStrength = 1,%Push
							U_Avatar = push(OAvatar,Avatar,PushStrength),
							{-100,void,OAvatar,U_Avatar};
						false ->
							PushStrength = 0,
							U_OAvatar=push(Avatar,OAvatar,PushStrength),
							{-100,void,U_OAvatar,Avatar}
					end;
				off ->
					{0,void,OAvatar,Avatar}
			end;
		(AType == beacon) ->
			case ?COLLISIONS of
				on ->
					case OAvatar#avatar.energy > Avatar#avatar.energy of
						true ->%TODO
							PushStrength = 1,%Push
							U_Avatar = push(OAvatar,Avatar,PushStrength),
							{0,void,OAvatar,U_Avatar};
						false ->
							PushStrength = 0,
							U_OAvatar=push(Avatar,OAvatar,PushStrength),
							{0,void,U_OAvatar,Avatar}
					end;
				off ->
					{0,void,OAvatar,Avatar}
			end;
		true ->
			{0,void,OAvatar,Avatar}
	end.

push(OAvatar,Avatar,PushStrength)->
	OAEnergy = OAvatar#avatar.energy,
	AEnergy = Avatar#avatar.energy,
	case OAEnergy > AEnergy of
		true ->
			{OX,OY} = OAvatar#avatar.loc,
			{X,Y} = Avatar#avatar.loc,
			DX = X-OX,
			DY = Y-OY,
			Distance = math:sqrt(math:pow(OX-X,2)+math:pow(OY-Y,2)),
%	io:format("OAvatar Type:~p Avatar Type:~p OX:~p OY:~p X:~p Y:~p DX:~p DY:~p Distance:~p~n",[OAvatar#avatar.type,Avatar#avatar.type,OX,OY,X,Y,DX,DY,Distance]),
			Min_Distance = OAvatar#avatar.r + Avatar#avatar.r,
			case Distance == 0 of
				true ->
					MinPushX = -DX,
					MinPushY = -DY;
				false ->
					MinPushX = (Min_Distance/Distance)*DX - DX,
					MinPushY = (Min_Distance/Distance)*DY - DY
			end,
			PushX = MinPushX + case DX == 0 of
				true -> 0;
				false -> (DX/abs(DX))*PushStrength
			end,
			PushY = MinPushY + case DY == 0 of
				true -> 0;
				false -> (DY/abs(DY))*PushStrength
			end,
%			io:format("MinPushX:~p MinPushY:~p~n",[MinPushX,MinPushY]),
%			io:format("Pusher:~p Pushee:~p~n Push:~p~n NewLoc:~p~n Distance:~p NewDistance:~p~n",[{OX,OY},{X,Y},{PushX,PushY},NewLoc,Distance,NewDistance]),
			U_Loc = {X+PushX,Y+PushY},
		U_Objects=[{ObjName,Id,Color,{PX+PushX,PY+PushY},[{CX+PushX,CY+PushY}||{CX,CY}<-Coords],P}||{ObjName,Id,Color,{PX,PY},Coords,P}<-Avatar#avatar.objects],
			Avatar#avatar{loc=U_Loc,energy=AEnergy-10*PushStrength,objects=U_Objects};
		false ->
			Avatar
	end.

resist({OX,OY},Avatar)->
	PushStrength = 0,
%	{OX,OY} = OAvatar#avatar.loc,
	{X,Y} = Avatar#avatar.loc,
	DX = X-OX,
	DY = Y-OY,
	Distance = math:sqrt(math:pow(OX-X,2)+math:pow(OY-Y,2)),
	Min_Distance = Avatar#avatar.r,
	MinPushX = (Min_Distance/Distance)*DX - DX,
	MinPushY = (Min_Distance/Distance)*DY - DY,
	PushX = MinPushX,%+(DX/abs(DX))*PushStrength,
	PushY = MinPushY,%+(DY/abs(DY))*PushStrength,
%	io:format("MinPushX:~p MinPushY:~p~n",[MinPushX,MinPushY]),
%	io:format("Pusher:~p Pushee:~p~n Push:~p~n NewLoc:~p~n Distance:~p NewDistance:~p~n",[{OX,OY},{X,Y},{PushX,PushY},NewLoc,Distance,NewDistance]),
	U_Loc = {X+PushX,Y+PushY},
	U_Objects=[{ObjName,Id,Color,{PX+PushX,PY+PushY},[{CX+PushX,CY+PushY}||{CX,CY}<-Coords],P}||{ObjName,Id,Color,{PX,PY},Coords,P}<-Avatar#avatar.objects],
	Avatar#avatar{loc=U_Loc,objects=U_Objects}.

world_wall_collision(OperatorAvatar,Avatar)->
	{X,Y} = OperatorAvatar#avatar.loc,
	R = OperatorAvatar#avatar.r,
	{WallType,WallParam} = Avatar#avatar.state,
	case WallType of
		x_wall ->
			{WY,WXMin,WXMax} = WallParam,
			case (WY =< (Y+R)) and (WY >= (Y-R)) of
				true ->
					case (WXMin =< X) and (WXMax >= X) of
						true ->
							case Y > WY of
								true ->
									DY = R-(Y-WY),
									U_Loc = {X,Y+DY},
									U_Objects = update_objects(OperatorAvatar#avatar.objects,0,DY),
									OperatorAvatar#avatar{loc = U_Loc,objects=U_Objects};
								false ->
									DY = -R-(Y-WY),
									U_Loc = {X,Y+DY},
									U_Objects = update_objects(OperatorAvatar#avatar.objects,0,DY),
									OperatorAvatar#avatar{loc = U_Loc,objects=U_Objects}
							end;
						false ->
							case X < WXMin of
								true ->
									Distance = math:sqrt(math:pow(X-WXMin,2)+math:pow(Y-WY,2)),
									case Distance < R of
										true ->
											resist({WXMin,WY},OperatorAvatar);
										false ->
											OperatorAvatar
									end;
								false ->
									Distance = math:sqrt(math:pow(X-WXMax,2)+math:pow(Y-WY,2)),
									case Distance < R of
										true ->
											resist({WXMax,WY},OperatorAvatar);
										false ->
											OperatorAvatar
									end
							end
					end;
				false ->
					OperatorAvatar
			end;
		y_wall ->
			{WX,WYMin,WYMax} = WallParam,
			case (WX =< (X+R)) and (WX >= (X-R)) of
				true ->
					case (WYMin =< Y) and (WYMax >= Y) of
						true ->
							case X > WX of
								true ->
									DX = R-(X-WX),
									U_Loc = {X+DX,Y},
									U_Objects = update_objects(OperatorAvatar#avatar.objects,DX,0),
									OperatorAvatar#avatar{loc = U_Loc,objects=U_Objects};
								false ->
									DX = -R-(X-WX),
									U_Loc = {X+DX,Y},
									U_Objects = update_objects(OperatorAvatar#avatar.objects,DX,0),
									OperatorAvatar#avatar{loc = U_Loc,objects=U_Objects}
							end;
						false ->
							case Y < WYMin of
								true ->
									Distance = math:sqrt(math:pow(Y-WYMin,2)+math:pow(X-WX,2)),
									case Distance < R of
										true ->
											resist({WX,WYMin},OperatorAvatar);
										false ->
											OperatorAvatar
									end;
								false ->
									Distance = math:sqrt(math:pow(Y-WYMax,2)+math:pow(X-WX,2)),
									case Distance < R of
										true ->
											resist({WX,WYMax},OperatorAvatar);
										false ->
											OperatorAvatar
									end
							end
					end;
				false ->
					OperatorAvatar
			end
	end.
	
	update_objects(Objects,DX,DY)->
		[{ObjName,Id,Color,{PX+DX,PY+DY},[{CX+DX,CY+DY}||{CX,CY}<-Coords],P}||{ObjName,Id,Color,{PX,PY},Coords,P}<-Objects].
		
wc({X,Y},R,WallType,WallParam)->
	case WallType of
		x_wall ->
			{WY,WXMin,WXMax} = WallParam,
			case (WY =< (Y+R)) and (WY >= (Y-R)) and ((WXMin-R) =< X) and ((WXMax+R)>= X) of
				true ->
					case Y > WY of
						true ->
							io:format("1:~p~n",[{X,WY+R}]);
						false ->
							io:format("2:~p~n",[{X,WY-R}])
					end;
				false ->
					io:format("3:~p~n",[{X,Y}])
			end;
		y_wall ->
			{WX,WYMin,WYMax} = WallParam,
			case (WX =< (X+R)) and (WX >= (X-R)) and ((WYMin-R) =< Y) and ((WYMax+R)>= Y) of
				true ->
					case X > WX of
						true ->
							io:format("4:~p~n",[{WX+R,Y}]);
						false ->
							io:format("5:~p~n",[{WX-R,Y}])
					end;
				false ->
					io:format("6:~p~n",[{X,Y}])
			end
	end.

pt({OX,OY},{X,Y})->
	DX = X-OX,
	DY = Y-OY,
	Distance = math:sqrt(math:pow(DX,2)+math:pow(DY,2)),
	Min_Distance = 1 + 1,
	%Min_Dif=Min_Distance-Distance,
	MinPushX = (Min_Distance/Distance)*DX - DX,
	MinPushY = (Min_Distance/Distance)*DY - DY,
	io:format("MinPushX:~p MinPushY:~p~n",[MinPushX,MinPushY]),
	PushStrength = 1.05,
	PushX = MinPushX*PushStrength,
	PushY = MinPushY*PushStrength,

	NewLoc = {NewX,NewY}={X+PushX,Y+PushY},
	NewDistance = math:sqrt(math:pow(NewX-OX,2)+math:pow(NewY-OY,2)),
	io:format("Pusher:~p Pushee:~p~n Push:~p~n NewLoc:~p~n Distance:~p NewDistance:~p~n",[{OX,OY},{X,Y},{PushX,PushY},NewLoc,Distance,NewDistance]).

gen_id()->
	genotype:generate_UniqueId().
	
create_seeds()->
	[].
	
create_rocks()->
	Rock_Locs = [
		{100,100,40,inf},
		{200,400,20,inf},
		{300,500,20,inf},
		{200,300,60,inf},
		{200,450,15,inf},
		{300,100,50,inf},
	%	{400,500,20,inf},
	%	{450,400,20,inf},
		{1000,400,20,inf}
		%{400,200,100,inf}
	],
	[create_avatar(rock,rock,gen_id(),undefined,Rock_Loc,undefined) || Rock_Loc <- Rock_Locs].

create_pillars()->
	[].

create_walls()->
	Sections =[
		{x_wall,{300,100,200}},
		{y_wall,{250,100,500}},
		{x_wall,{200,400,450}},
		{y_wall,{400,200,300}},
		%{x_wall,{300,400,450}},
		%{x_wall,{250,450,600}},
		%{y_wall,{700,200,400}},
		%{x_wall,{400,450,500}},
		{y_wall,{500,400,500}},
		{x_wall,{500,450,500}}
	],	
	[create_avatar(wall,wall,gen_id(),undefined,Section_Loc,undefined) || Section_Loc <- Sections].
	
create_firepits()->
	FirePits=[
		%{100,100,50,inf},
		%{400,400,50,inf},
		{600,100,50,inf},
		{900,300,50,inf},
		{800,200,50,inf},
		{150,800,50,inf},
		{50,500,50,inf},
		{600,800,50,inf},
		{500,300,50,inf}
	],
	[create_avatar(fire_pit,fire_pit,gen_id(),undefined,FirePit,undefined) || FirePit <- FirePits].
	
create_water()->
	Waters=[
	],
	[create_avatar(water,water,gen_id(),undefined,Water,undefined) || Water <- Waters].
	
create_beacons()->
	Beacons = [
		{500,500,3,inf}
	],
	[create_avatar(beacon,beacon,gen_id(),undefined,Beacon,undefined) || Beacon <- Beacons].
