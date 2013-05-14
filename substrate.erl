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

-module(substrate).
-compile(export_all).
-include("records.hrl").
-define(SAT_LIMIT,math:pi()).
-record(state,{
	type,
	plasticity=none,
	morphology,
	specie_id,
	sensors,
	actuators,
	spids=[],
	apids=[],
	cpp_pids=[],
	cep_pids=[],
	densities,
	substrate_state_flag,
	old_substrate,
	cur_substrate,
	link_form
}).

gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf)->
	random:seed(now()),
	receive
		{ExoSelf,init,InitState}->
			{Sensors,Actuators,SPIds,APIds,CPP_PIds,CEP_PIds,Densities,Plasticity,LinkForm}=InitState,
			%io:format("InitState:~p~n",[InitState]),
			S = #state{
				sensors=Sensors,
				actuators=Actuators,
				spids=SPIds,
				apids=APIds,
				cpp_pids=CPP_PIds,
				cep_pids=CEP_PIds,
				densities = Densities,
				substrate_state_flag=reset,
				old_substrate=void,
				cur_substrate=init,
				plasticity=Plasticity,
				link_form = LinkForm
			},
			substrate:loop(ExoSelf,S,SPIds,[])
	end.

loop(ExoSelf,S,[SPId|SPIds],SAcc)->
	%io:format("~p~n",[S]),
	receive
		{SPId,forward,Sensory_Signal}->
			loop(ExoSelf,S,SPIds,[Sensory_Signal|SAcc]);
		{ExoSelf,reset_substrate}->
			U_S = S#state{
				old_substrate=S#state.cur_substrate,
				substrate_state_flag=reset
			},
			ExoSelf ! {self(),ready},
			loop(ExoSelf,U_S,[SPId|SPIds],SAcc);
		{ExoSelf,backup_substrate} ->
%			io:format("reseting:~n"),
			U_S = S#state{
				old_substrate=S#state.cur_substrate,
				substrate_state_flag=reset
			},
			ExoSelf ! {self(),ready},
			loop(ExoSelf,U_S,[SPId|SPIds],SAcc);
		{ExoSelf,revert_substrate} ->
%			io:format("reverting:~n"),
			U_S = S#state{
				cur_substrate = S#state.old_substrate,
				substrate_state_flag=reset
			},
			ExoSelf ! {self(),ready},
			loop(ExoSelf,U_S,[SPId|SPIds],SAcc);
		{ExoSelf,terminate}->
			%io:format("Resulting substrate:~p~n",[S]),
			void;
		Msg ->
			io:format("Unknown Msg:~p~n",[Msg]),
			loop(ExoSelf,S,[SPId|SPIds],SAcc)
%		after 20000 ->
%			io:format("********ERROR: Substrate Crashed:~p~n",[S])
	end;
loop(ExoSelf,S,[],SAcc)->%All sensory signals received
	{U_Substrate,U_SMode,OAcc} = reason(SAcc,S),
	advanced_fanout(OAcc,S#state.actuators,S#state.apids),
	U_S = S#state{
		cur_substrate=U_Substrate,
		substrate_state_flag=U_SMode
	},
	loop(ExoSelf,U_S,S#state.spids,[]).

reason(Input,S)->
	Densities = S#state.densities,
	Substrate = S#state.cur_substrate,
	SMode = S#state.substrate_state_flag,
	CPP_PIds = S#state.cpp_pids,
	CEP_PIds = S#state.cep_pids,
	Plasticity = S#state.plasticity,
	case SMode of
		reset ->%io:format("reset~n"),
			Sensors=S#state.sensors,
			Actuators=S#state.actuators,
			New_Substrate = create_substrate(Sensors,Densities,Actuators,S#state.link_form),
			%io:format("New_Substrate:~p~n Output:~p~n Populated_Substrate:~p~n",[New_Substrate,Output,Populated_Substrate]),
			U_SMode=case Plasticity of
				iterative ->
					{Output,Populated_Substrate} = calculate_ResetOutput(Densities,New_Substrate,Input,CPP_PIds,CEP_PIds,Plasticity,S#state.link_form),
					iterative;
				_ ->
					{Output,Populated_Substrate} = calculate_ResetOutput(Densities,New_Substrate,Input,CPP_PIds,CEP_PIds,Plasticity,S#state.link_form),
					hold
			end,
			{Populated_Substrate,U_SMode,Output};
		iterative ->%io:format("Iterative~n"),
			{Output,U_Substrate} = calculate_ResetOutput(Densities,Substrate,Input,CPP_PIds,CEP_PIds,Plasticity,S#state.link_form),
%			io:format("Output:~p~n Densities:~p~n Substrate:~p~n U_Substrate:~p~n CT:~p~n CF:~p~n",[Output,Densities,Substrate,U_Substrate,CT,CF]),
			{U_Substrate,SMode,Output};
		hold ->%io:format("hold~n"),
			{Output,U_Substrate} = calculate_HoldOutput(Densities,Substrate,Input,S#state.link_form,Plasticity,CPP_PIds,CEP_PIds),
			%io:format("Substrate:~p~n U_Substrate:~p~n",[Substrate,U_Substrate]),
			%io:format("Output1:~p Output:~p~n",[Output,Output]),
			{U_Substrate,SMode,Output}
	end.

advanced_fanout(OAcc,[Actuator|Actuators],[APId|APIds])->
	{Output,OAccRem}=lists:split(Actuator#actuator.vl,OAcc),
	APId ! {self(),forward,Output},
	advanced_fanout(OAccRem,Actuators,APIds);
advanced_fanout([],[],[])->
	ok.
%%==================================================================== Internal Functions
fanout([Pid|Pids],Msg)->
	Pid ! Msg,
	fanout(Pids,Msg);
fanout([],_Msg)->
	true.

flush_buffer()->
	receive 
		ANY -> %io:format("ANY:~p~n",[ANY]),
		flush_buffer()
	after 0 ->
		done
end.

%	no_geo
%	{symmetric,[R1,R2...Rk],[Val1...Valn]} where n == R1*R2*...Dk and k = dimension
%	{asymmetric,[[R1..Rp],[R1..Rt]],[Val1...Valn]} where lists:sum(lists:flatten([[R1...Rp],[R1..Rt]])) == n, and depth = Dimension.
%	coorded, every val comes with its own coord tuple: {Coord,Val}. The coord is a list, thus specifying the dimensionality.
test_cs()->
	Sensors = [
		#sensor{format=no_geo,vl=3},
		#sensor{format={symmetric,lists:reverse([2,3])},vl=6}
	],
	Actuators = [
		#actuator{format=no_geo,vl=2},
		#actuator{format={symmetric,lists:reverse([3,2])},vl=6}
	],
	create_substrate(Sensors,[3,2,3,2],Actuators,l2l_feedforward).
	
test_IS(SubstrateDimension)->
	Sensors = [
		#sensor{format=no_geo,vl=10},
		#sensor{format={symmetric,lists:reverse([3,4])},vl=[
		1,-1,-1,-1,
		1,-1,-1,-1,
		1,1,1,1]}
	],
	compose_ISubstrate(Sensors,SubstrateDimension).

test_OS(SubstrateDimension)->
	Actuators = [
		#actuator{format=no_geo,vl=10},
		#actuator{format={symmetric,lists:reverse([3,4])},vl=[
		1,-1,-1,-1,
		1,-1,-1,-1,
		1,1,1,1]}
	],
	compose_OSubstrate(Actuators,SubstrateDimension,[w1,w2,w3]).
	
create_substrate(Sensors,Densities,Actuators,LinkForm)->
	[Depth|SubDensities] = Densities,
	Substrate_I = compose_ISubstrate(Sensors,length(Densities)),
	I_VL = length(Substrate_I),
%	io:format("I_VL:~p~n",[I_VL]),
	case LinkForm of
		l2l_feedforward ->
			Weight = 0,
			H = mult(SubDensities),
			IWeights = lists:duplicate(I_VL,Weight),
			HWeights = lists:duplicate(H,Weight);
		%ff_fully_interconnected->%TODO: Variable number of weights for each layer.
		%	;
		fully_interconnected ->
			Output_Neurodes = tot_ONeurodes(Actuators,0),
			Weight = 0,
			Tot_HiddenNeurodes = mult([Depth-1|SubDensities]),
			Tot_Weights = Tot_HiddenNeurodes + I_VL + Output_Neurodes,
			IWeights = lists:duplicate(Tot_Weights,Weight),
			HWeights = lists:duplicate(Tot_Weights,Weight);
		jordan_recurrent ->
			Output_Neurodes = tot_ONeurodes(Actuators,0),
			Weight = 0,
			H = mult(SubDensities),
			IWeights = lists:duplicate(I_VL+Output_Neurodes,Weight),
			HWeights = lists:duplicate(H,Weight);
		neuronself_recurrent ->
			Weight = 0,
			H = mult(SubDensities),
			IWeights = lists:duplicate(I_VL+1,Weight),
			HWeights = lists:duplicate(H+1,Weight)
		%planeself_recurrent ->%TODO: Variable number of weights for each layer
		%	;
		%olb_recurrent ->%TODO: One Layer Back recurrency, variable number of weights for each layer.
		%	void
	end,	
	case Depth of
		0 ->
			Substrate_O=compose_OSubstrate(Actuators,length(Densities),IWeights),
			[Substrate_I,Substrate_O];
		1 ->
			Substrate_R = cs(SubDensities,IWeights),
			Substrate_O=compose_OSubstrate(Actuators,length(Densities),HWeights),
%			io:format("Substrate_I:~n~p~n Substrate_R:~n~p~n Substrate_O:~n~p~n",[Substrate_I,Substrate_R,Substrate_O]),
			[Substrate_I,extrude(0,Substrate_R),Substrate_O];
		_ ->
			Substrate_R = cs(SubDensities,IWeights),
			Substrate_H = cs(SubDensities,HWeights),
			Substrate_O=compose_OSubstrate(Actuators,length(Densities),HWeights),
			%io:format("OResolutions:~p Substrate_O:~p~n",[OResolutions,Substrate_O]),
			[_,RCoord|C1] = build_CoordList(Depth+1),
			[_|C2] = lists:reverse(C1),
			HCoords = lists:reverse(C2),
			%io:format("RCoord:~p HCoord:~p~n",[RCoord,HCoords]),
			ESubstrate_R = extrude(RCoord,Substrate_R),
			ESubstrates_H = [extrude(HCoord,Substrate_H) || HCoord<-HCoords],
			%io:format("ESubstrate_R:~p ESubstrates_H:~p~n",[ESubstrate_R,ESubstrates_H]),
			lists:append([[Substrate_I,ESubstrate_R],ESubstrates_H,[Substrate_O]])
	end.

	compose_ISubstrate(Sensors,SubstrateDimension)->
		compose_ISubstrate(Sensors,[],1,SubstrateDimension-2).
	compose_ISubstrate([S|Sensors],Acc,Max_Dim,Required_Dim)->
		case S#sensor.format of
			undefined ->
				Dim=1,
				CoordLists = create_CoordLists([S#sensor.vl]),
				ISubstrate_Part=[{Coord,0,void}|| Coord<-CoordLists],
				{Dim,ISubstrate_Part};
			no_geo ->
				Dim=1,
				CoordLists = create_CoordLists([S#sensor.vl]),
				ISubstrate_Part=[{Coord,0,void}|| Coord<-CoordLists],
				{Dim,ISubstrate_Part};
			{symmetric,Resolutions}->
				Dim = length(Resolutions),
				Signal_Length = mult(Resolutions),
				CoordLists = create_CoordLists(Resolutions),
				ISubstrate_Part=[{Coord,0,void}|| Coord<-CoordLists],
				{Dim,ISubstrate_Part};
	%		{asymmetric,Resolutions}->
	%			Dim = find_depth(Resolutions),
	%			ISubstrate_Part = void,
	%			{Dim,ISubstrate_Part};
			{coorded,Dim,Resolutions,ISubstrate_Part} ->
				{Dim,ISubstrate_Part}
		end,
		U_Dim = case Max_Dim > Dim of
			true ->
				Max_Dim;
			false ->
				Dim
		end,
		compose_ISubstrate(Sensors,[ISubstrate_Part|Acc],U_Dim,Required_Dim);
	compose_ISubstrate([],Acc,ISubstratePart_MaxDim,Required_Dim)->
		case Required_Dim >= ISubstratePart_MaxDim of
			true ->
				ISubstrate_Depth = length(Acc),
				ISubstrate_DepthCoords = build_CoordList(ISubstrate_Depth),
				adv_extrude(Acc,Required_Dim,lists:reverse(ISubstrate_DepthCoords),-1,[]);%Passed in inverted,reversed inside adv_extrude, same for depth coords.
			false ->
				exit("Error in adv_extrude, Required_Depth < ISubstratePart_MaxDepth~n")
		end.

		adv_extrude([ISubstrate_Part|ISubstrate],Required_Dim,[IDepthCoord|ISubstrate_DepthCoords],LeadCoord,Acc)->
			Extruded_ISP = [{[LeadCoord,IDepthCoord|lists:append(lists:duplicate(Required_Dim - length(Coord),0),Coord)],O,W} || {Coord,O,W}<-ISubstrate_Part],
			extrude(ISubstrate_Part,Required_Dim,IDepthCoord,[]),
			adv_extrude(ISubstrate,Required_Dim,ISubstrate_DepthCoords,LeadCoord,lists:append(Extruded_ISP,Acc));
		adv_extrude([],_Required_Dim,[],_LeadCoord,Acc)->
			Acc.
			
			extrude([{Coord,O,W}|ISubstrate_Part],Required_Dim,DepthCoord,Acc)->
				Dim_Dif = Required_Dim - length(Coord),
				U_Coord= [1,DepthCoord|lists:append(lists:duplicate(Dim_Dif,0),Coord)],
				extrude(ISubstrate_Part,Required_Dim,DepthCoord,[{U_Coord,O,W}|Acc]);
			extrude([],_Required_Dim,_DepthCoord,Acc)->
				Acc.

	compose_OSubstrate(Actuators,SubstrateDimension,Weights)->
		compose_OSubstrate(Actuators,[],1,SubstrateDimension-2,Weights).
	compose_OSubstrate([A|Actuators],Acc,Max_Dim,Required_Dim,Weights)->
		case A#actuator.format of
			undefined ->%Dim=void,OSubstrate_Part=void,
				Dim=1,
				CoordLists = create_CoordLists([A#actuator.vl]),
				OSubstrate_Part=[{Coord,0,Weights}|| Coord<-CoordLists],
				{Dim,OSubstrate_Part};
			no_geo ->%Dim=void,OSubstrate_Part=void,
				Dim=1,
				CoordLists = create_CoordLists([A#actuator.vl]),
				OSubstrate_Part=[{Coord,0,Weights}|| Coord<-CoordLists],
				{Dim,OSubstrate_Part};
			{symmetric,Resolutions}->%Dim=void,OSubstrate_Part=void,
				Dim = length(Resolutions),
				Signal_Length = mult(Resolutions),
				CoordLists = create_CoordLists(Resolutions),
				OSubstrate_Part=[{Coord,0,Weights}|| Coord<-CoordLists],
				{Dim,OSubstrate_Part};
	%		{asymmetric,Resolutions}->%Dim=void,OSubstrate_Part=void,
	%			Dim = find_depth(Resolutions),
	%			OSubstrate_Part = void,
	%			{Dim,OSubstrate_Part};
			{coorded,Dim,Resolutions,Unadjusted_OSubstrate_Part} ->
				OSubstrate_Part=[{Coord,O,Weights}|| {Coord,O,_}<-Unadjusted_OSubstrate_Part],
				{Dim,OSubstrate_Part}
		end,
		U_Dim = case Max_Dim > Dim of
			true ->
				Max_Dim;
			false ->
				Dim
		end,
		compose_OSubstrate(Actuators,[OSubstrate_Part|Acc],U_Dim,Required_Dim,Weights);
	compose_OSubstrate([],Acc,OSubstratePart_MaxDim,Required_Dim,_Weights)->
		case Required_Dim >= OSubstratePart_MaxDim of
			true ->%done;
				ISubstrate_Depth = length(Acc),
				ISubstrate_DepthCoords = build_CoordList(ISubstrate_Depth),
				adv_extrude(Acc,Required_Dim,lists:reverse(ISubstrate_DepthCoords),1,[]);%Passed in inverted,reversed inside adv_extrude, same for depth coord
			false ->
				exit("Error in adv_extrude, Required_Depth < OSubstratePart_MaxDepth~n")
		end.

		find_depth(Resolutions)->find_depth(Resolutions,0).
		find_depth(Resolutions,Acc)->
			case is_list(Resolutions) of
				true ->
					[_Head|Tail] = Resolutions,
					find_depth(Tail,Acc+1);
				false ->
					Acc
			end.

%Substrate encoding: X density = n, Y density = k, Z density = p, T density = l
%Weights = [W1,W2...WI],
%[[{[Z1,Y,X],o,[W1...Wn]}...{[Z1,Yn,Xk],o,[W1...Wn]}]...[{[Zs,Y,X],o,[W1...Wn]}...]],
		build_CoordList(Density)->
			case Density == 1 of
				true ->
					[0.0];
				false ->
					DensityDividers = Density - 1,
					Resolution = 2/DensityDividers,
					build_CoordList(Resolution,DensityDividers,1,[])
			end.

			extend(I,DI,D,Substrate)->
				void.
				
			mult(List)->
				mult(List,1).
			mult([Val|List],Acc)->
				mult(List,Val*Acc);
			mult([],Acc)->
				Acc.

tot_ONeurodes([A|Actuators],Acc)->
	Tot_ANeurodes=case A#actuator.format of
		undefined ->
			A#actuator.vl;
		no_geo ->
			A#actuator.vl;
		{symmetric,Resolutions}->
			mult(Resolutions);
		{coorded,Dim,Resolutions,Unadjusted_OSubstrate_Part} ->
			length(Unadjusted_OSubstrate_Part)
	end,
	tot_ONeurodes(Actuators,Tot_ANeurodes+Acc);
tot_ONeurodes([],Acc)->
	Acc.


%[{[D3,D2,D1],o,[W1,W2,W3...]}...]
	cs(Densities,Weights)->
		RDensities = lists:reverse(Densities),
		Substrate = create_CoordLists(RDensities,[]),
		attach(Substrate,0,Weights).
	
		create_CoordLists(Densities)->
			create_CoordLists(Densities,[]).	
		create_CoordLists([Density|RDensities],[])->
			CoordList = build_CoordList(Density),
			XtendedCoordList = [[Coord]||Coord <- CoordList],
			create_CoordLists(RDensities,XtendedCoordList);
		create_CoordLists([Density|RDensities],Acc)->
			CoordList = build_CoordList(Density),
			XtendedCoordList = [[Coord|Sub_Coord]||Coord <- CoordList,Sub_Coord <- Acc],
			create_CoordLists(RDensities,XtendedCoordList);
		create_CoordLists([],Acc)->
			Acc.
			
			build_CoordList(Resolution,0,Coord,Acc)->
				[-1|Acc];
			build_CoordList(Resolution,DensityDividers,Coord,Acc)->
				build_CoordList(Resolution,DensityDividers-1,Coord-Resolution,[Coord|Acc]).
		
attach(List,E1,E2)->
	attach(List,E1,E2,[]).
attach([Val|List],E1,E2,Acc)->
	attach(List,E1,E2,[{Val,E1,E2}|Acc]);
attach([],_E1,_E2,Acc)->
	lists:reverse(Acc).
	
extrude(NewDimension_Coord,Substrate)->
	extrude(NewDimension_Coord,Substrate,[]).
extrude(NewDimension_Coord,[{Coord,O,W}|Substrate],Acc)->
	extrude(NewDimension_Coord,Substrate,[{[NewDimension_Coord|Coord],O,W}|Acc]);
extrude(_Coord,[],Acc)->
	lists:reverse(Acc).

calculate_HoldOutput(Densities,Substrate,Input,LinkForm,Plasticity,CPP_PIds,CEP_PIds)->
	[IHyperlayer|Populated_PHyperlayers] = Substrate,
	Populated_IHyperlayer = populate_InputHyperlayer(IHyperlayer,lists:flatten(Input),[]),
	{Output,U_PHyperlayers}=calculate_substrate_output(Populated_IHyperlayer,Populated_PHyperlayers,LinkForm,Plasticity,CPP_PIds,CEP_PIds),
	{Output,[IHyperlayer|U_PHyperlayers]}.

calculate_ResetOutput(Densities,Substrate,Input,CPP_PIds,CEP_PIds,Plasticity,LinkForm)->
	[IHyperlayer|PHyperlayers] = Substrate,
	%io:format("IHyperlayer:~p~n PHyperlayers:~p~n",[IHyperlayer,PHyperlayers]),
	Populated_IHyperlayer = populate_InputHyperlayer(IHyperlayer,lists:flatten(Input),[]),
	case Plasticity of
		iterative ->
			%Populated_Substrate = [Populated_IHyperlayer|PHyperlayers],
			%io:format("PHyperlayers:~p~n LinkForm:~p~n Plasticity:~p~n",[PHyperlayers,LinkForm,Plasticity]),
			{Output,U_PHyperlayers}=calculate_substrate_output(Populated_IHyperlayer,PHyperlayers,LinkForm,Plasticity,CPP_PIds,CEP_PIds),
			{Output,[IHyperlayer|U_PHyperlayers]};
		_->
			Populated_PHyperlayers = populate_PHyperlayers(Substrate,CPP_PIds,CEP_PIds,LinkForm,Plasticity),
			%Populated_Substrate = lists:append([Populated_IHyperlayer],Populated_PHyperlayers),
			%io:format("Populated_PHyperlayers:~p~n LinkForm:~p~n Plasticity:~p~n",[Populated_PHyperlayers,LinkForm,Plasticity]),
			{Output,U_PHyperlayers}=calculate_substrate_output(Populated_IHyperlayer,Populated_PHyperlayers,LinkForm,Plasticity,CPP_PIds,CEP_PIds),
			{Output,[IHyperlayer|U_PHyperlayers]}
	end.

	populate_InputHyperlayer([{Coord,PrevO,void}|Substrate],[I|Input],Acc)->
		populate_InputHyperlayer(Substrate,Input,[{Coord,I,void}|Acc]);
	populate_InputHyperlayer([],[],Acc)->
		lists:reverse(Acc).
		
	populate_PHyperlayers(Substrate,CPP_PIds,CEP_PIds,LinkForm,Plasticity)->
		case LinkForm of
			l2l_feedforward ->
				[IHyperlayer,PHyperlayer|RemSubstrate] = Substrate,
				%io:format("Substrate:~p~n",[Substrate]),
				populate_PHyperlayers_l2l(IHyperlayer,PHyperlayer,RemSubstrate,CPP_PIds,CEP_PIds,Plasticity,[],[]);
			%ff_fully_interconnected->%TODO: Variable number of weights for each layer.
			%	;
			fully_interconnected ->
				[_IHyperlayer,PHyperlayer|RemSubstrate] = Substrate,
				I_Neurodes = lists:flatten(Substrate),
				populate_PHyperlayers_fi(I_Neurodes,PHyperlayer,RemSubstrate,CPP_PIds,CEP_PIds,Plasticity,[],[]);
			jordan_recurrent ->
				[IHyperlayer,PHyperlayer|RemSubstrate] = Substrate,
				[OHyperlayer|_]=lists:reverse(Substrate),
				I_Neurodes=lists:flatten([IHyperlayer,OHyperlayer]),
				populate_PHyperlayers_l2l(I_Neurodes,PHyperlayer,RemSubstrate,CPP_PIds,CEP_PIds,Plasticity,[],[]);
			neuronself_recurrent ->
				[IHyperlayer,PHyperlayer|RemSubstrate] = Substrate,
				populate_PHyperlayers_nsr(IHyperlayer,PHyperlayer,RemSubstrate,CPP_PIds,CEP_PIds,Plasticity,[],[])
			%planeself_recurrent ->%TODO: Variable number of weights for each layer
			%	;
			%olb_recurrent ->%TODO: One Layer Back recurrency, variable number of weights for each layer.
			%	void
		end.
	
		populate_PHyperlayers_l2l(PrevHyperlayer,[{Coord,PrevO,PrevWeights}|CurHyperlayer],Substrate,CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			NewWeights = case Plasticity of
				none -> 
					get_weights(PrevHyperlayer,Coord,CPP_PIds,CEP_PIds,[]);
				_ ->
					get_weights(PrevHyperlayer,Coord,CPP_PIds,CEP_PIds,[],PrevWeights,PrevO)
			end,
			populate_PHyperlayers_l2l(PrevHyperlayer,CurHyperlayer,Substrate,CPP_PIds,CEP_PIds,Plasticity,[{Coord,PrevO,NewWeights}|Acc1],Acc2);
		populate_PHyperlayers_l2l(_PrevHyperlayer,[],[CurHyperlayer|Substrate],CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			PrevHyperlayer = lists:reverse(Acc1),
			populate_PHyperlayers_l2l(PrevHyperlayer,CurHyperlayer,Substrate,CPP_PIds,CEP_PIds,Plasticity,[],[PrevHyperlayer|Acc2]);
		populate_PHyperlayers_l2l(_PrevHyperlayer,[],[],CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			lists:reverse([lists:reverse(Acc1)|Acc2]).

		populate_PHyperlayers_fi(FlatSubstrate,[{Coord,PrevO,PrevWeights}|CurHyperlayer],Substrate,CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			NewWeights = case Plasticity of
				none -> 
					get_weights(FlatSubstrate,Coord,CPP_PIds,CEP_PIds,[]);
				_ ->
					get_weights(FlatSubstrate,Coord,CPP_PIds,CEP_PIds,[],PrevWeights,PrevO)
			end,
			populate_PHyperlayers_fi(FlatSubstrate,CurHyperlayer,Substrate,CPP_PIds,CEP_PIds,Plasticity,[{Coord,PrevO,NewWeights}|Acc1],Acc2);
		populate_PHyperlayers_fi(FlatSubstrate,[],[CurHyperlayer|Substrate],CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			populate_PHyperlayers_fi(FlatSubstrate,CurHyperlayer,Substrate,CPP_PIds,CEP_PIds,Plasticity,[],[lists:reverse(Acc1)|Acc2]);
		populate_PHyperlayers_fi(_FlatSubstrate,[],[],CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			lists:reverse([lists:reverse(Acc1)|Acc2]).

		populate_PHyperlayers_nsr(PrevHyperlayer,[{Coord,PrevO,PrevWeights}|CurHyperlayer],Substrate,CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			NewWeights = case Plasticity of
				none -> 
					get_weights([{Coord,PrevO,PrevWeights}|PrevHyperlayer],Coord,CPP_PIds,CEP_PIds,[]);
				_ ->
					get_weights([{Coord,PrevO,PrevWeights}|PrevHyperlayer],Coord,CPP_PIds,CEP_PIds,[],PrevWeights,PrevO)
			end,
			populate_PHyperlayers_nsr(PrevHyperlayer,CurHyperlayer,Substrate,CPP_PIds,CEP_PIds,Plasticity,[{Coord,PrevO,NewWeights}|Acc1],Acc2);
		populate_PHyperlayers_nsr(_PrevHyperlayer,[],[CurHyperlayer|Substrate],CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			PrevHyperlayer = lists:reverse(Acc1),
			populate_PHyperlayers_nsr(PrevHyperlayer,CurHyperlayer,Substrate,CPP_PIds,CEP_PIds,Plasticity,[],[PrevHyperlayer|Acc2]);
		populate_PHyperlayers_nsr(_PrevHyperlayer,[],[],CPP_PIds,CEP_PIds,Plasticity,Acc1,Acc2)->
			lists:reverse([lists:reverse(Acc1)|Acc2]).
						
			get_weights([{I_Coord,I,_I_Weights}|I_Neurodes],Coord,CPP_PIds,CEP_PIds,Acc)->
				static_fanout(CPP_PIds,I_Coord,Coord),
				U_W=fanin(CEP_PIds,void),
				get_weights(I_Neurodes,Coord,CPP_PIds,CEP_PIds,[U_W|Acc]);
			get_weights([],_Coord,_CPP_PIds,_CEP_PIds,Acc)->
				lists:reverse(Acc).

				static_fanout([CPP_PId|CPP_PIds],I_Coord,Coord)->
					%io:format("CPP_PId:~p~n",[CPP_PId]),
					CPP_PId ! {self(),I_Coord,Coord},
					static_fanout(CPP_PIds,I_Coord,Coord);
				static_fanout([],_I_Coord,_Coord)->
					done.
					
				fanin([CEP_PId|CEP_PIds],W)->
					receive
						{CEP_PId,Command,Signal}->
							U_W=substrate:Command(Signal,W)
					end,
					fanin(CEP_PIds,U_W);
				fanin([],W)->
					W.
				
			get_weights([{I_Coord,I,_I_Weights}|I_Neurodes],Coord,CPP_PIds,CEP_PIds,Acc,[W|Weights],O)->
				plasticity_fanout(CPP_PIds,I_Coord,Coord,[I,O,W]),
				U_W=fanin(CEP_PIds,W),
				get_weights(I_Neurodes,Coord,CPP_PIds,CEP_PIds,[U_W|Acc],Weights,O);
			get_weights([],_Coord,CPP_PIds,CEP_PIds,Acc,[],_O)->
				lists:reverse(Acc).

				plasticity_fanout([CPP_PId|CPP_PIds],I_Coord,Coord,IOW)->
					CPP_PId ! {self(),I_Coord,Coord,IOW},
					plasticity_fanout(CPP_PIds,I_Coord,Coord,IOW);
				plasticity_fanout([],_I_Coord,_Coord,_IOW)->
					done.
							
					set_weight(Signal,_WP)->
						[U_W] = Signal,
						functions:sat(U_W,3.1415,-3.1415).
						
					weight_expression(Signal,_WP) ->
						[U_W,Expression]=Signal,
						case Expression > 0 of
							true ->
								functions:sat(U_W,3.1415,-3.1415);
							false ->
								0
						end.
					
					set_abcn(Signal,_WP)->
						[U_W,A,B,C,N] = Signal,
						{functions:sat(U_W,3.1415,-3.1415),abcn,[A,B,C,N]}.
						
					set_iterative(Signal,W)->
						[Delta_Weight] = Signal,
						functions:sat(W + Delta_Weight,3.1415,-3.1415).
						
		calculate_substrate_output(IHyperlayer,PHyperlayer,LinkForm,Plasticity,CPP_PIds,CEP_PIds)->
			case LinkForm of
				l2l_feedforward ->
					calculate_output_std(IHyperlayer,PHyperlayer,Plasticity,CPP_PIds,CEP_PIds,[]);
				%ff_fully_interconnected->%TODO: Variable number of weights for each layer.
				%	;
				fully_interconnected ->
					calculate_output_fi(lists:flatten([IHyperlayer|PHyperlayer]),PHyperlayer,Plasticity,CPP_PIds,CEP_PIds,[]);
				jordan_recurrent ->
					[OHyperlayer|_] = lists:reverse(PHyperlayer,Plasticity),
					calculate_output_std(lists:flatten([IHyperlayer|OHyperlayer]),PHyperlayer,Plasticity,CPP_PIds,CEP_PIds,[]);
				neuronself_recurrent ->
					calculate_output_nsr(IHyperlayer,PHyperlayer,Plasticity,CPP_PIds,CEP_PIds,[])
				%planeself_recurrent ->%TODO: Variable number of weights for each layer
				%	;
				%olb_recurrent ->%TODO: One Layer Back recurrency, variable number of weights for each layer.
				%	void
			end.
			
		calculate_output_std(I_Neurodes,[Cur_Hyperlayer|Substrate],Plasticity,CPP_PIds,CEP_PIds,Acc)->
%			io:format("Prev_Hypercube:~p~n",[Prev_Hypercube]),
%			Val=length(Prev_Hypercube),
%			Normalizer = case Val == 0 of
%				true ->
%					1;
%				false ->
%					Val
%			end,
%			U_PH=[{Coord,O,Weights}  || {Coord,O,Weights} <- Prev_Hypercube],
			%U_PH=[{Coord,O+(random:uniform()-0.5)*O*0.5,Weights}  || {Coord,O,Weights} <- Prev_Hypercube],	
			U_CurHyperlayer = [calculate_output(I_Neurodes,Neurode,Plasticity,CPP_PIds,CEP_PIds) || Neurode <- Cur_Hyperlayer],
%			io:format("Updated_CurHypercube:~p~n",[Updated_CurHypercube]),
			calculate_output_std(U_CurHyperlayer,Substrate,Plasticity,CPP_PIds,CEP_PIds,[U_CurHyperlayer|Acc]);
		calculate_output_std(Output_Hyperlayer,[],_Plasticity,CPP_PIds,CEP_PIds,Acc)->
			{[Output || {_Coord,Output,_Weights} <- Output_Hyperlayer],lists:reverse(Acc)}.
			
			calculate_output(I_Neurodes,Neurode,Plasticity,CPP_PIds,CEP_PIds)->
				{Coord,_Prev_O,Weights} = Neurode,
				case Plasticity of
					none ->
						Output=calculate_neurode_output_std(I_Neurodes,Neurode,0),
						{Coord,Output,Weights};
					iterative ->
						Output=calculate_neurode_output_std(I_Neurodes,Neurode,0),
						U_Weights = get_weights(I_Neurodes,Coord,CPP_PIds,CEP_PIds,[],Weights,Output),
						{Coord,Output,U_Weights};
					abcn ->
						%io:format("I_Neurodes:~p~n Neurode:~p~n",[I_Neurodes,Neurode]),
						Output=calculate_neurode_output_plast(I_Neurodes,Neurode,0),
						%io:format("Output:~p~n",[Output]),
						update_neurode(I_Neurodes,{Coord,Output,Weights},[])
				end.
			
					calculate_neurode_output_std([{_I_Coord,O,_I_Weights}|I_Neurodes],{Coord,Prev_O,[Weight|Weights]},Acc)->
						calculate_neurode_output_std(I_Neurodes,{Coord,Prev_O,Weights},O*Weight+Acc);
					calculate_neurode_output_std([],{Coord,Prev_O,[]},Acc)->
						functions:tanh(Acc).
					%calculate_output(A,B,C)->
					%	io:format("{A,B,C}:~p~n",[{A,B,C}]).
					
					calculate_neurode_output_plast([{_I_Coord,O,_I_Weights}|I_Neurodes],{Coord,Prev_O,[{W,_LF,_Parameters}|WPs]},Acc)->
						calculate_neurode_output_plast(I_Neurodes,{Coord,Prev_O,WPs},O*W+Acc);
					calculate_neurode_output_plast([],{Coord,Prev_O,[]},Acc)->
						functions:tanh(Acc).
						
					update_neurode([{_I_Coord,I_O,_I_Weights}|I_Neurodes],{Coord,O,[{W,LF,Parameters}|WPs]},Acc)->
						U_W = substrate:LF(I_O,O,W,Parameters),
						update_neurode(I_Neurodes,{Coord,O,WPs},[{U_W,LF,Parameters}|Acc]);
					update_neurode([],{Coord,O,[]},Acc)->
						{Coord,O,lists:reverse(Acc)}.
			
						abcn(Input,Output,W,[A,B,C,N])->
							Delta_Weight = N*(A*Input*Output + B*Input + C*Output),
							%io:format("~p~n",[{Input,Output,W,Delta_Weight,[A,B,C,N]}]),
							W+Delta_Weight.
			
		calculate_output_fi(I_Neurodes,[Cur_Hyperlayer|Substrate],Plasticity,CPP_PIds,CEP_PIds,Acc)->
			%I_Neurodes = lists:flatten([I_Hyperlayer,Cur_Hyperlayer|Substrate]),
			U_CurHyperlayer = [calculate_output(I_Neurodes,Neurode,Plasticity,CPP_PIds,CEP_PIds) || Neurode <- Cur_Hyperlayer],
			calculate_output_fi([I_Neurodes|U_CurHyperlayer],Substrate,Plasticity,CPP_PIds,CEP_PIds,[U_CurHyperlayer|Acc]);
		calculate_output_fi(Output_Hyperlayer,[],_Plasticity,CPP_PIds,CEP_PIds,Acc)->
			{[Output || {_Coord,Output,_Weights} <- Output_Hyperlayer],lists:reverse(Acc)}.
			
		calculate_output_nsr(I_Neurodes,[Cur_Hyperlayer|Substrate],Plasticity,CPP_PIds,CEP_PIds,Acc)->
			U_CurHyperlayer = [calculate_output([Neurode|I_Neurodes],Neurode,Plasticity,CPP_PIds,CEP_PIds) || Neurode <- Cur_Hyperlayer],
			calculate_output_nsr(U_CurHyperlayer,Substrate,Plasticity,CPP_PIds,CEP_PIds,[U_CurHyperlayer|Acc]);
		calculate_output_nsr(Output_Hyperlayer,[],_Plasticity,CPP_PIds,CEP_PIds,Acc)->
			{[Output || {_Coord,Output,_Weights} <- Output_Hyperlayer],lists:reverse(Acc)}.
