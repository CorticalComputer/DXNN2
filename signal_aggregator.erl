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

-module(signal_aggregator).
-compile(export_all).
-include("records.hrl").

dot_product(IAcc,IPIdPs)->
	%io:format("Iacc:~p IPIdPs:~p~n",[IAcc,IPIdPs]),
	dot_product(IAcc,IPIdPs,0).
dot_product([{IPId,Input}|IAcc],[{IPId,WeightsP}|IPIdPs],Acc)->
	Dot = dot(Input,WeightsP,0),
	dot_product(IAcc,IPIdPs,Dot+Acc);
dot_product([],[{bias,[{Bias,_PDB,_LP,_LPs}]}],Acc)->
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
		
		dot([I|Input],[{W,_PDW,_LP,_LPs}|WeightsP],Acc) ->
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
mult_product([],[{bias,[{Bias,_PDB,_LP,_LPs}]}],Acc)->
	Acc * Bias;
mult_product([],[],Acc)->
	Acc.

	mult([I|Input],[{W,_PDW,_LP,_LPs}|Weights],Acc) ->
		mult(Input,Weights,I*W*Acc);
	mult([],[],Acc)->
		Acc.
