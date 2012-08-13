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

-module(functions).
-compile(export_all).
	
saturation(Val)->
	case Val > 1000 of
		true ->
			1000;
		false ->
			case Val < -1000 of
				true ->
					-1000;
				false ->
					Val
			end
	end.
	
saturation(Val,Spread)->
	case Val > Spread of
		true ->
			Spread;
		false ->
			case Val < -Spread of
				true ->
					-Spread;
				false ->
					Val
			end
	end.

scale([H|T],Max,Min)->
	[scale(Val,Max,Min)||Val<-[H|T]];
scale(Val,Max,Min)-> %Nm = (Y*2 - (Max + Min))/(Max-Min)
	case Max == Min of
		true ->
			0;
		false ->
			(Val*2 - (Max+Min))/(Max-Min)
	end.

sat(Val,Max,Min)->
	case Val > Max of
		true ->
			Max;
		false ->
			case Val < Min of
				true ->
					Min;
				false ->
					Val
			end
	end.

sat_dzone(Val,Max,Min,DZMax,DZMin)->
	case (Val < DZMax) and (Val > DZMin) of
		true ->
			0;
		false ->
			sat(Val,Max,Min)
	end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Activation Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tanh(Val)->		
	math:tanh(Val).
		
cos(Val)->
	math:cos(Val).

sin(Val)->
	math:sin(Val).

sgn(0)->
	0;
sgn(Val)->
	case Val > 0 of
		true -> 1;
		false -> -1
	end.

bin(Val)->
	case Val > 0 of
		true -> 1;
		false -> 0
	end.

trinary(Val)->
	if
		(Val < 0.33) and (Val > -0.33) -> 0;
		Val >= 0.33 -> 1;
		Val =< -0.33 -> -1
	end.
	
multiquadric(Val)->
	math:pow(Val*Val + 0.01,0.5).

absolute(Val)->
	abs(Val).
	
linear(Val)->
	Val.

quadratic(Val)->
	sgn(Val)*Val*Val.

gaussian(Val)->
	gaussian(2.71828183,Val).

gaussian(Const,Val)->
	V = case Val > 10 of
		true ->
			10;
		false ->
			case Val < -10 of
				true ->
					-10;
				false ->
					Val
			end
	end,
	math:pow(Const,-V*V).

sqrt(Val)->
	sgn(Val)*math:sqrt(abs(Val)).
	
log(Val)->
	case Val == 0 of
		true ->
			0;
		false ->
			sgn(Val)*math:log(abs(Val))
	end.

sigmoid(Val)-> %(-1 : 1)--Der:Y*(1-Y)
	V = case Val > 10 of
		true ->
			10;
		false ->
			case Val < -10 of
				true ->
					-10;
				false ->
					Val
			end
	end,
	2/(1+math:pow(2.71828183,-V)) - 1.

sigmoid1(Val)-> %(-1 : 1) -- Der:1/((1+abs(val))*(1+abs(val)))
	Val/(1+abs(Val)).

avg(List)->
	lists:sum(List)/length(List).
std(List)->
	Avg = avg(List),
	std(List,Avg,[]).
	
	std([Val|List],Avg,Acc)->
		std(List,Avg,[math:pow(Avg-Val,2)|Acc]);
	std([],_Avg,Acc)->
		Variance = lists:sum(Acc)/length(Acc),
		math:sqrt(Variance).
		
%A's Synaptic Weights:
%	xor_GetInput to A: [-4.3986,-2.3223]
%	A to A: [6.2832]
%	Bias: [1.3463]

%B's Synaptic Weights:
%	A to B: [-4.9582]
%	Bias: [-2.4443]
%XOR = [{[-1,-1],[-1]},{[1,-1],[1]},{[-1,1],[1]},{[1,1],[-1]}],
s([V1,V2])->
	RC=case get(rec) of
		undefined ->
			0;
		RecSig ->
			RecSig
	end,
	Output1 = math:tanh(V1*-4.3986 + V2*-2.3223 + RC*6.2832 + 1.3463),
	put(rec,Output1),
	io:format("Output1:~p~n",[Output1]),
	Output2 = math:tanh(Output1*-4.9582 -2.4443),
	io:format("Output2:~p~n",[Output2]).
	
%HYPERCUBE:
%Specifications:[I,H,O]
%I,O:[LL1...LLn] Layer length, where LL is on the X axis, and 1-n is on the Y axis, making this a 2d specification.
% [[LL1...LLx],[LL1...LLx]] Depth 2 list, 3d I_Hypercube.
% [{Dimension,[{Coord1...CoordN}]},{Dimension,[{Coord1...CoordN}]}]
%H:[N..Z,Y,X] Where N is the depth, and Z,Y,X specifies the dimensions of a symetric hypercube. Z by Y by X.
%abc,none
cartesian(I_Coord,Coord)->
	lists:append(I_Coord,Coord).
polar(I_Coord,Coord)->
	lists:append(cart2pol(I_Coord),cart2pol(Coord)).
spherical(I_Coord,Coord)->
	lists:append(cart2spher(I_Coord),cart2spher(Coord)).
centripital_distances(I_Coord,Coord)->
	[centripital_distance(I_Coord,0),centripital_distance(Coord,0)].
cartesian_distance(I_Coord,Coord)->
	[calculate_distance(I_Coord,Coord,0)].
cartesian_CoordDiffs(I_Coord,Coord)->%I:[X1,Y1,Z1] [X2,Y2,Z2] O:[X2-X1,Y2-Y1,Z2-Z1]
	cartesian_CoordDiffs1(I_Coord,Coord,[]).
	
	cartesian_CoordDiffs1([FromCoord|FromCoords],[ToCoord|ToCoords],Acc)->
		cartesian_CoordDiffs1(FromCoords,ToCoords,[ToCoord-FromCoord|Acc]);
	cartesian_CoordDiffs1([],[],Acc)->
		lists:reverse(Acc).

cartesian_GaussedCoordDiffs(FromCoords,ToCoords)->%I:[X1,Y1,Z1] [X2,Y2,Z2] O:[gauss(X2-X1),gauss(Y2-Y1),gauss(Z2-Z1)]
	cartesian_GaussedCoordDiffs1(FromCoords,ToCoords,[]).
	
	cartesian_GaussedCoordDiffs1([FromCoord|FromCoords],[ToCoord|ToCoords],Acc)->
		cartesian_GaussedCoordDiffs1(FromCoords,ToCoords,[functions:gaussian(ToCoord-FromCoord)|Acc]);
	cartesian_GaussedCoordDiffs1([],[],Acc)->
		lists:reverse(Acc).
		
%Iterative
cartesian(I_Coord,Coord,[I,O,W])->
	[I,O,W|lists:append(I_Coord,Coord)].
polar(I_Coord,Coord,[I,O,W])->
	[I,O,W|lists:append(cart2pol(I_Coord),cart2pol(Coord))].
spherical(I_Coord,Coord,[I,O,W])->
	[I,O,W|lists:append(cart2spher(I_Coord),cart2spher(Coord))].
centripital_distances(I_Coord,Coord,[I,O,W])->
	[I,O,W,centripital_distance(I_Coord,0),centripital_distance(Coord,0)].
cartesian_distance(I_Coord,Coord,[I,O,W])->
	[I,O,W,calculate_distance(I_Coord,Coord,0)].
cartesian_CoordDiffs(FromCoords,ToCoords,[I,O,W])->
	[I,O,W|cartesian_CoordDiffs(FromCoords,ToCoords)].
cartesian_GaussedCoordDiffs(FromCoords,ToCoords,[I,O,W])->
	[I,O,W|cartesian_GaussedCoordDiffs(FromCoords,ToCoords)].
iow(_I_Coord,_Coord,IOW)->
	IOW.
		
	cart2pol([Y,X])->
		R = math:sqrt(X*X + Y*Y),
		Theta = case R == 0 of
			true ->
				0;
			false ->
				if
					(X>0)	and	(Y>=0)	-> math:atan(Y/X);
					(X>0)	and	(Y<0)	-> math:atan(Y/X) + 2*math:pi();
					(X<0)			-> math:atan(Y/X) + math:pi();
					(X==0)	and	(Y>0)	-> math:pi()/2;
					(X==0)	and	(Y<0)	-> 3*math:pi()/2
				end
		end,
		[R,Theta].
		
	cart2spher([Z,Y,X])->
		%Pi = math:pi(),
		PreR = X*X + Y*Y,
		R = math:sqrt(PreR),
		P = math:sqrt(PreR + Z*Z),
		Theta = case R == 0 of
			true ->
				0;
			false ->
				if
					(X>0)	and	(Y>=0)	-> math:atan(Y/X);
					(X>0)	and	(Y<0)	-> math:atan(Y/X) + 2*math:pi();
					(X<0)			-> math:atan(Y/X) + math:pi();
					(X==0)	and	(Y>0)	-> math:pi()/2;
					(X==0)	and	(Y<0)	-> 3*math:pi()/2
				end
		end,
		Phi = case P == 0 of
			false ->
				math:acos(Z/P);
			true ->
				0
		end,
		[P,Theta,Phi].
					centripital_distance([Val|Coord],Acc)->
						centripital_distance(Coord,Val*Val+Acc);
					centripital_distance([],Acc)->
						math:sqrt(Acc).
						
					calculate_distance([Val1|Coord1],[Val2|Coord2],Acc)->
						Distance = Val2 - Val1,
						calculate_distance(Coord1,Coord2,Distance*Distance+Acc);
					calculate_distance([],[],Acc)->
						math:sqrt(Acc).

to_cartesian(Direction)->
	case Direction of
		{spherical,Coordinates}->
			{cartesian,spherical2cartesian(Coordinates)};
		{polar,Coordinates}->
			{cartesian,polar2cartesian(Coordinates)};
		{cartesian,Coordinates}->
			{cartesian,Coordinates}
	end.
						
normalize(Vector)->
	Normalizer = calculate_normalizer(Vector,0),
	normalize(Vector,Normalizer,[]).
					
	calculate_normalizer([Val|Vector],Acc)->
		calculate_normalizer(Vector,Val*Val+Acc);
		calculate_normalizer([],Acc)->
		math:sqrt(Acc).
					
	normalize([Val|Vector],Normalizer,Acc)->
		normalize(Vector,Normalizer,[Val/Normalizer|Acc]);
		normalize([],_Normalizer,Acc)->
		lists:reverse(Acc).
				
spherical2cartesian({P,Theta,Phi})->
	X = P*math:sin(Phi)*math:cos(Theta),
	Y = P*math:sin(Phi)*math:sin(Theta),
	Z = P*math:cos(Phi),
	{X,Y,Z}.

%Theta: 0-2Pi, Phi:0-Pi, R: 0+, P: 0+		
cartesian2spherical({X,Y})->
	cartesian2spherical({X,Y,0});
cartesian2spherical({X,Y,Z})->
	%Pi = math:pi(),
	PreR = X*X + Y*Y,
	R = math:sqrt(PreR),
	P = math:sqrt(PreR + Z*Z),
	Theta = case R == 0 of
		true ->
			0;
		false ->
			if
				(X>0)	and	(Y>=0)	-> math:atan(Y/X);
				(X>0)	and	(Y<0)	-> math:atan(Y/X) + 2*math:pi();
				(X<0)			-> math:atan(Y/X) + math:pi();
				(X==0)	and	(Y>0)	-> math:pi()/2;
				(X==0)	and	(Y<0)	-> 3*math:pi()/2
			end
	end,
	Phi = case P == 0 of
		false ->
			math:acos(Z/P);
		true ->
			0
	end,
	{P,Theta,Phi}.
				
polar2cartesian({R,Theta})->
	X = R*math:cos(Theta),
	Y = R*math:sin(Theta),
	{X,Y,0}.

%Theta: 0-2Pi, R: 0+			
cartesian2polar({X,Y})->
	cartesian2polar({X,Y,0});
cartesian2polar({X,Y,_Z})->
	R = math:sqrt(X*X + Y*Y),
	Theta = case R == 0 of
		true ->
			0;
		false ->
			if
				(X>0)	and	(Y>=0)	-> math:atan(Y/X);
				(X>0)	and	(Y<0)	-> math:atan(Y/X) + 2*math:pi();
				(X<0)			-> math:atan(Y/X) + math:pi();
				(X==0)	and	(Y>0)	-> math:pi()/2;
				(X==0)	and	(Y<0)	-> 3*math:pi()/2
			end
	end,
	{R,Theta}.

distance(Vector1,Vector2)->
	distance(Vector1,Vector2,0).	
distance([Val1|Vector1],[Val2|Vector2],Acc)->
	distance(Vector1,Vector2,Acc+math:pow(Val2-Val1,2));
distance([],[],Acc)->
	math:sqrt(Acc).
	
vector_difference(Vector1,Vector2)->
	vector_difference(Vector1,Vector2,[]).
vector_difference([Val1|Vector1],[Val2|Vector2],Acc)->
	vector_difference(Vector1,Vector2,[Val2-Val1|Acc]);
vector_difference([],[],Acc)->
	lists:reverse(Acc).
