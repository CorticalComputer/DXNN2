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

-module(signal_aggregator).
-compile(export_all).
-include("records.hrl").

dot_product(IAcc,IPIdPs)->
	%io:format("Iacc:~p IPIdPs:~p~n",[IAcc,IPIdPs]),
	dot_product(IAcc,IPIdPs,0).
dot_product([{IPId,Input}|IAcc],[{IPId,WeightsP}|IPIdPs],Acc)->
	Dot = dot(Input,WeightsP,0),
	dot_product(IAcc,IPIdPs,Dot+Acc);
dot_product([],[{bias,[{Bias,_LPs}]}],Acc)->
	Acc + Bias;
dot_product([],[],Acc)->
	Acc.
	
%	dot(Input,[{Weights,MWeights}|WeightsP])->
%		dot(Input,MWeights,0),
%		dot(Input,WeightsP);
%	dot(Input,[_|WeightsP])->
%		dot(Input,WeightsP);
%	dot(_Input,[])->
%		0.
		
		dot([I|Input],[{W,_LPs}|WeightsP],Acc) ->
			dot(Input,WeightsP,I*W+Acc);
		dot([],[],Acc)->
			Acc.
%The dot/3 function accepts an input vector and a weight list, and computes the dot product of the two vectors.

diff_product(IAcc,IPIdPs)->
	case get(diff_product) of
		undefined ->
			put(diff_product,IAcc),
			dot_product(IAcc,IPIdPs,0);
		Prev_IAcc ->
			put(diff_product,IAcc),
			Diff_IAcc = input_diff(IAcc,Prev_IAcc,[]),
			dot_product(Diff_IAcc,IPIdPs,0)
	end.
	
	input_diff([{IPId,Input}|IAcc],[{IPId,Prev_Input}|Prev_IAcc],Acc)->
		Vector_Diff = diff(Input,Prev_Input,[]),
		input_diff(IAcc,Prev_IAcc,[{IPId,Vector_Diff}|Acc]);
	input_diff([],[],Acc)->
		lists:reverse(Acc).
	
		diff([A|Input],[B|Prev_Input],Acc)->
			diff(Input,Prev_Input,[A-B|Acc]);
		diff([],[],Acc)->
			lists:reverse(Acc).
			
mult_product(IAcc,IPIdPs)->
	mult_product(IAcc,IPIdPs,1).
mult_product([{IPId,Input}|IAcc],[{IPId,WeightsP}|IPIdPs],Acc)->
	Dot = mult(Input,WeightsP,1),
	mult_product(IAcc,IPIdPs,Dot*Acc);
mult_product([],[{bias,[{Bias,_LPs}]}],Acc)->
	Acc * Bias;
mult_product([],[],Acc)->
	Acc.

	mult([I|Input],[{W,_LPs}|Weights],Acc) ->
		mult(Input,Weights,I*W*Acc);
	mult([],[],Acc)->
		Acc.
