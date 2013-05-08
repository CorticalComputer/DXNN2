%% This source code and work is provided and developed by DXNN Research Group WWW.DXNNResearch.COM
%%
%Copyright (C) 2009 by Gene Sher, DXNN Research Group, CorticalComputer@gmail.com
%All rights reserved.
%
%This code is licensed under the version 3 of the GNU General Public License. Please see the LICENSE file that accompanies this project for the terms of use.

-module(visor).
-include("records.hrl").
-compile(export_all).
-record(state,{window,canvas,update_rate,filter,scape,scape_type}).
-define(WIDTH,1440).
-define(HEIGHT,900).
-define(INIT_SCAPE_TYPE,flatland).
zo()->
	visor ! {self(),new_Filter,{0.2,200,200}}.

zo(Filter)->
	visor ! {self(),new_Filter,Filter}.

legend()->
	visor ! ctlegend_on.
	
start()->
	spawn(visor,loop,[]).
	
stop()->
	visor ! {self(),terminate}.
	
loop()->
	GS = gs:start(),
	Window = gs:create(window,GS,[{title,"Visor"},{width,?WIDTH},{height,?HEIGHT}]),
	Canvas = gs:create(canvas,Window,[{width,?WIDTH},{height,?HEIGHT}]),
	register(visor,self()),
	put(canvas,Canvas),
	gs:config(Window,{map,true}),
	ScapeType = ?INIT_SCAPE_TYPE,
	Scape_PId = gen_server:call(polis,{get_scape,ScapeType}),
%	io:format("Inside Visor:: Scape_PId:~p~n",[Scape_PId]),
	gen_server:cast(Scape_PId,{self(),subscribe,Canvas}),
	InitState = #state{
		window = Window,
		canvas = Canvas,
		update_rate = 10,
		filter = {1.0,0,0},
		scape = Scape_PId,
		scape_type = ScapeType
	},
	io:format("Visor started. Scape_PId:~p ScapeType:~p~n",[Scape_PId,ScapeType]),
	loop(InitState).
	
loop(S)->
	Scape_PId = S#state.scape,
	receive
		{From,new_Filter,NewFilter}->
			visor:loop(S#state{filter=NewFilter});
		{From,new_UpdateRate,NewUpdateRate}->
			visor:loop(S#state{update_rate=NewUpdateRate});
		{From,change_scape,ScapeType}->
			gen_server:cast(S#state.scape,{self(),unsubscribe}),
			New_ScapePId = gen_server:call(polis,{get_scape,ScapeType}),
			visor:loop(S#state{scape=New_ScapePId});
		{Scape_PId,draw_object,Object}->
			%io:format("~p~n",[Object]),
			Id=draw_object(S#state.canvas,Object),
			Scape_PId ! {self(),Id},
			visor:loop(S);
		{Scape_PId,destroy_object,Id}->
			gs:destroy(Id),
			visor:loop(S);
		ctlegend_on ->
			Legend=ct_legend(),
			Canvas = S#state.canvas,
			io:format("Legend:~p~n",[Legend]),
			put(legend,gs:create(text,Canvas,[{fg,red},{coords,[{50,50}]},{text,Legend}])),
			visor:loop(S);
		ctlegend_off ->
			gs:destroy(get(legend)),
			erase(legend),
			io:format("Legend off.~n"),
			visor:loop(S);
		{From,terminate}->
			done = gen_server:call(S#state.scape,{self(),unsubscribe}),
			io:format("Visor unsubscribed from:~p and terminated.~n",[S#state.scape])
	after S#state.update_rate ->
		%io:format("Visor is refreshing.~n"),
		gen_server:cast(S#state.scape,{self(),redraw,S#state.filter}),
		visor:loop(S)
	end.

draw_avatars(Canvas,[Avatar|Avatars],Acc)->
	U_Avatar = draw_avatar(Canvas,Avatar),
	draw_avatars(Canvas,Avatars,[U_Avatar|Acc]);
draw_avatars(Canvas,[],Acc)->
	Acc.
		
draw_avatar(Canvas,Avatar)->
	Objects = Avatar#avatar.objects,
	U_Objects = draw_objects(Canvas,Objects,[]),
	Avatar#avatar{objects=U_Objects}.
	
	draw_objects(Canvas,[Object|Objects],Acc)->
		{ObjName,_IdPlaceHolder,Color,Pivot,Coords,Parameter} = Object,		
		Id = case ObjName of
			circle ->
				[{Cx,Cy}] = Coords,
				R = Parameter,
				Draw_Coords = [{Cx-R,Cy-R},{Cx+R,Cy+R}],
				gs:create(oval,Canvas,[{coords,Draw_Coords},{fill,Color},{fg,Color}]);
			arrow ->
				gs:create(line,Canvas,[{coords,Coords},{arrow,last},{fg,Color}]);
			polygon ->
				gs:create(polygon,Canvas,[{coords,Coords},{fill,Color},{fg,Color}]);
			_ ->
				gs:create(ObjName,Canvas,[{coords,Coords},{fg,Color}])
		end,
		U_Object = {ObjName,Id,Color,Pivot,Coords,Parameter},
		draw_objects(Canvas,Objects,[U_Object|Acc]);
	draw_objects(_Canvas,[],Acc)->
		Acc.
			
	draw_object(Canvas,Object)->
		{ObjName,undefined,Color,Pivot,Coords,Parameter} = Object,
		Id = case ObjName of
			circle ->
				[{Cx,Cy}] = Coords,
				R = Parameter,
				Draw_Coords = [{Cx-R,Cy-R},{Cx+R,Cy+R}],
				gs:create(oval,Canvas,[{coords,Draw_Coords},{fill,Color},{fg,Color}]);
			arrow ->
				gs:create(line,Canvas,[{coords,Coords},{arrow,last},{fg,Color}]);
			_ ->
				gs:create(ObjName,Canvas,[{coords,Coords},{fg,Color}])
		end.
		
	draw_stats(Canvas,Avatar) when (Avatar#avatar.type == flatlander) or (Avatar#avatar.type == prey)->
		CF = Avatar#avatar.actuators,
		CT = Avatar#avatar.sensors,
		Loc = Avatar#avatar.loc,
		CFStatIds = [gs:create(text,Canvas,[{coords,[Loc]},{text,atom_to_list(Actuator)}]) || {CF_PId,CFVL,{actuator,Actuator,ActuatorId,Parameters}}<-CF],
		CTStatIds = [gs:create(text,Canvas,[{coords,[Loc]},{text,atom_to_list(Sensor)}]) || {CT_PId,CTVL,{sensor,Sensor,SensorId,Parameters}}<-CT],
		[gs:destroy(Id) || Id<- CFStatIds],
		[gs:destroy(Id) || Id<- CTStatIds];
	draw_stats(_,_)->
		done.

redraw_avatars(Filter,[Avatar|Avatars])->
	Objects = Avatar#avatar.objects,
	redraw_objects(Filter,Objects),
	redraw_avatars(Filter,Avatars);
redraw_avatars(_Filter,[])->
	done.
				
	redraw_objects({Zoom,PanX,PanY},[Object|Objects])->
		{ObjName,Id,Color,Pivot,Coords,Parameter} = Object,
		Draw_Coords =case ObjName of
			circle ->
				[{Cx,Cy}] = Coords,
				R = Parameter,
				[{Cx-R,Cy-R},{Cx+R,Cy+R}];
			_ ->
				Coords
		end,
		Filtered_Coords = [{X*Zoom+PanX,Y*Zoom+PanY} || {X,Y} <- Draw_Coords],
		if
			(ObjName  == circle) -> gs:config(Id,[{coords,Filtered_Coords},{fill,Color},{fg,Color}]);
			true -> gs:config(Id,[{coords,Filtered_Coords},{fg,Color}])
		end,
		redraw_objects({Zoom,PanX,PanY},Objects);
	redraw_objects(_Filter,[])->
		done.

%color_scaner(Op,{Zoom,PanX,PanY},Density,Spread,Loc,Direction,Avatars)
redraw_SensorVisualization(Filter,[Avatar|Avatars],AllAvatars)->
	sensors:color_scaner(draw,Filter,45,math:pi()*2,Avatar#avatar.loc,Avatar#avatar.direction,AllAvatars--[Avatar]),
	redraw_SensorVisualization(Filter,Avatars,AllAvatars);
redraw_SensorVisualization(_Filter,[],_AllAvatars)->
	done.
	
ct2color(Sensors)->
	io:format("Sensors:~p~n",[Sensors]),
	SensorNames = [Name ||{sensor,Name,SensorId,VL,Parameters}<-Sensors],
	SensorTuple = count_SensorTypes(SensorNames,{0,0,0,0}),
	case SensorTuple of
		{_,0,0,0}-> red;%prey_sensor
		{0,_,0,0}-> blue;%plant_sensor
		{_,_,0,0}-> white;%prey_sensor and plant_sensor
		{0,_,_,0}-> cyan; %plant_sensor, predator_sensor
		{_,0,_,0}-> grey; %prey_sensor, predator_sensor
		{_,_,_,_}-> black %prey_sensor, plant_sensor, predator_sensor
	end.
	%[coned_prey_sensor,coned_plant_sensor,coned_flatlander_sensor]]
	count_SensorTypes([Sensor|Sensors],{PreySensors,PlantSensors,FlatlanderSensors,OtherSensors})->
		case Sensor of
			coned_prey_sensor->count_SensorTypes(Sensors,{PreySensors+1,PlantSensors,FlatlanderSensors,OtherSensors});
			coned_plant_sensor->count_SensorTypes(Sensors,{PreySensors,PlantSensors+1,FlatlanderSensors,OtherSensors});
			coned_flatlander_sensor->count_SensorTypes(Sensors,{PreySensors,PlantSensors,FlatlanderSensors+1,OtherSensors});
			_-> count_SensorTypes(Sensors,{PreySensors,PlantSensors,FlatlanderSensors,OtherSensors+1})
		end;
	count_SensorTypes([],Tuple)->
		Tuple.
		
ct_legend()->
	["Red -> prey_sensor\n",
	"Blue -> plant_sensor\n", 
	"White -> prey_sensor & plant_sensor\n", 
	"Cyan -> plant_sensor & predator_sensor\n",
	"Grey -> prey_sensor & predator_sensor\n",
	"Black -> prey_sensor & plant_sensor & predator_sensor\n"].
						
			check_pivot(Avatar)->
				Objects = Avatar#avatar.objects,
				[{Pivot,Coords}] = [{Pivot,Coords} || {arrow,Id,Color,Pivot,Coords,Parameter}<-Objects],
				case length([1||Coord<-Coords, Pivot==Coord])of
					1 ->
						io:format("hit~n");
					0 ->
						io:format("mis~n")
				end.
				
t()->	spawn(visor,test,[]).
test()->		
	GS = gs:start(),
	Window = gs:create(window,GS,[{title,"Visor"},{width,1400},{height,900}]),
	Canvas = gs:create(canvas,Window,[{width,1400},{height,900}]),
	register(test,self()),
	put(canvas,Canvas),
	gs:config(Window,{map,true}),
	CT = [{self(),4,{sensor,coned_plant_sensor,id,[4]}}],
	Avatars = [scape:create_avatar(flatlander,self(),self(),{cf,CT,1},void) || _<- lists:duplicate(20,1)],
	U_Avatars = draw_avatars(Canvas,Avatars,[]),
	test(U_Avatars,Window,Canvas,100000,random).
		
		test(_,_,_,0,_)->
			done;
		test(Avatars,Window,Canvas,Step,Mode)->
			timer:sleep(4),
			U_Avatars = case Mode of
				manual ->
					receive 
						{move, Speed}->
							[scape:move(Avatar,Speed) || Avatar <- Avatars];
						{rotate,Angle}->
							[scape:rotate(Avatar,Angle) || Avatar <- Avatars]
					after 4 ->
						Avatars
					end;
				random ->
					case random:uniform(2) of
						1 ->%io:format("move~n"),
							[scape:move(Avatar,random:uniform()) || Avatar <- Avatars];
						2 ->%io:format("rotate~n"),
							[scape:rotate(Avatar,(random:uniform()-0.5)/2*math:pi()) || Avatar <- Avatars]
					end
			end,
			Zoom = 1.0,
			PanX = Zoom*0,
			PanY = Zoom*0,
			Filter = {Zoom,PanX,PanY},
			redraw_avatars(Filter,U_Avatars),
			redraw_SensorVisualization(Filter,U_Avatars,U_Avatars),
			visor:test(U_Avatars,Window,Canvas,Step-1,Mode).
