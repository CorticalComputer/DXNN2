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

-module(plasticity).
-compile(export_all).
-include("records.hrl").
-define(SAT_LIMIT,math:pi()*2).

none({_N_Id,mutate})->
	exit("Neuron does not support plasticity.");
none(neural_parameters)->
	[];
none(weight_parameters)->
	[].
%none/1 returns a set of learning parameters needed by the none/1 plasticity function. Since this function specifies that the neuron has no plasticity, the parameter lists are empty. When executed with the {N_Id,mutate} parameter, the function exits, since there is nothing ot mutate. The exit allows for the neuroevolutionary system to try another mutation operator on the NN system.

none(_NeuralParameters,_IAcc,Input_PIdPs,_Output)->
	Input_PIdPs.
%none/3 returns the original Input_PIdPs to the caller.

hebbian_w({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{input_idps = U_InputIdPs};
hebbian_w(neural_parameters)->
	[];
hebbian_w(weight_parameters)->
	[(random:uniform()-0.5)].
%hebbian_w/1 function produces the necessary parameter list for the hebbian_w learning rule to operate. The parameter list for the simple hebbian_w learning rule is a parameter list composed of a single parameter H: [H], for every synaptic weight of the neuron. When hebbian_w/1 is called with the parameter neural_parameters, it returns []. When hebbian_w/1 is executed with the {N_Id,mutate} parameter, the function goes through every parameter in the neuron's input_idps, and perturbs the parameter value using the specified spread (?SAT_LIMIT).

	perturb_parameters(InputIdPs,Spread)->
		TotParameters = lists:sum([ lists:sum([length(Ps) || {_W,Ps} <- WPs]) || {_Input_Id,WPs} <- InputIdPs]),
		MutationProb = 1/math:sqrt(TotParameters),
		[{Input_Id,[{W,perturb(Ps,MutationProb,Spread,[])}|| {W,Ps} <- WPs]} || {Input_Id,WPs} <- InputIdPs].
	
		perturb([Val|Vals],MutationProb,Spread,Acc)->
			case random:uniform() < MutationProb of
				true ->
					U_Val = sat((random:uniform()-0.5)*2*Spread+Val,Spread,Spread),
					perturb(Vals,MutationProb,Spread,[U_Val|Acc]);
				false ->
					perturb(Vals,MutationProb,Spread,[Val|Acc])
			end;
		perturb([],_MutationProb,_Spread,Acc)->
			lists:reverse(Acc).

hebbian_w(_NeuralParameters,IAcc,Input_PIdPs,Output)->
	hebbian_w1(IAcc,Input_PIdPs,Output,[]).

	hebbian_w1([{IPId,Is}|IAcc],[{IPId,WPs}|Input_PIdPs],Output,Acc)->
		Updated_WPs = hebbrule_w(Is,WPs,Output,[]),
		hebbian_w1(IAcc,Input_PIdPs,Output,[{IPId,Updated_WPs}|Acc]);
	hebbian_w1([],[],_Output,Acc)->
		lists:reverse(Acc);
	hebbian_w1([],[{bias,WPs}],_Output,Acc)->
		lists:reverse([{bias,WPs}|Acc]).
%hebbian_w/4 function operates on each Input_PIdP, calling the hebbian_w1/4 function which processes each of the complementary Is and WPs lists, producing the Updated_WPs list in return, with the updated/adapted weights based on the hebbian_w learning rule. 

	hebbrule_w([I|Is],[{W,DW,LP,[H]}|WPs],[Output],Acc)->
		Updated_W = functions:saturation(W + H*I*Output,?SAT_LIMIT),
		hebbrule_w(Is,WPs,[Output],[{Updated_W,DW,LP,[H]}|Acc]);
	hebbrule_w([],[],_Output,Acc)->
		lists:reverse(Acc).
%hebbrule_w/4 applies the hebbian learning rule to each weight, using the input value I, the neuron's calculated output Output, and its own distinct learning parameter H associated with each synaptic weight.

hebbian({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	{PFName,ParameterList} = N#neuron.pf,
	Spread = ?SAT_LIMIT*10,
	MutationProb = 1/math:sqrt(length(ParameterList)),
	U_ParameterList = perturb(ParameterList,MutationProb,Spread,[]),
	U_PF = {PFName,U_ParameterList},
	N#neuron{pf=U_PF};
hebbian(neural_parameters)->
	[(random:uniform()-0.5)];
hebbian(weight_parameters)->
	[].
%hebbian/1 function produces the necessary parameter list for the hebbian learning rule to operate. The parameter list for the standard hebbian learning rule is a parameter list composed of a single parameter H: [H], used by the neuron for all its synaptic weights. When hebbian/1 is called with the parameter weight_parameters, it returns []. When the function is executed with the {N_Id,mutate} parameter, it uses the perturb/4 function to perturb the parameter list, which in this case is a list composed of a single floating point parameter.

hebbian([_M,H],IAcc,Input_PIdPs,Output)->
	hebbian(H,IAcc,Input_PIdPs,Output,[]).

	hebbian(H,[{IPId,Is}|IAcc],[{IPId,WPs}|Input_PIdPs],Output,Acc)->
		Updated_WPs = hebbrule(H,Is,WPs,Output,[]),
		hebbian(H,IAcc,Input_PIdPs,Output,[{IPId,Updated_WPs}|Acc]);
	hebbian(_H,[],[],_Output,Acc)->
		lists:reverse(Acc);
	hebbian(_H,[],[{bias,WPs}],_Output,Acc)->
		lists:reverse([{bias,WPs}|Acc]).
%hebbian/4 function operates on each Input_PIdP, calling the hebbian/5 function which processes each of the complementary Is and WPs lists, producing the Updated_WPs list in return, with the updated/adapted weights based on the standard hebbian learning rule, using the neuron's single learning parameter H. 

	hebbrule(H,[I|Is],[{W,DW,LP,[]}|WPs],[Output],Acc)->
		Updated_W = functions:saturation(W + H*I*Output,?SAT_LIMIT),
		hebbrule(H,Is,WPs,[Output],[{Updated_W,DW,LP,[]}|Acc]);
	hebbrule(_H,[],[],_Output,Acc)->
		lists:reverse(Acc).
%hebbrule/5 applies the hebbian learning rule to each weight, using the input value I, the neuron's calculated output Output, and the neuron's leraning parameter H.

ojas_w({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{input_idps = U_InputIdPs};
ojas_w(neural_parameters)->
	[];
ojas_w(weight_parameters)->
	[(random:uniform()-0.5)].
%oja/1 function produces the necessary parameter list for the oja's learning rule to operate. The parameter list for oja's learning rule is a list composed of a single parameter H: [H] per synaptic weight. If the learning parameter is positive, then the postsynaptic neuron's synaptic weight increases if the two connected neurons produce output signals of the same sign. If the learning parameter is negative, and the two connected neurons produce output signals of the same sign, then the synaptic weight of the postsynaptic neuron, decreases in magnitude. Otherwise it increases.

ojas_w(_Neural_Parameters,IAcc,Input_PIdPs,Output)->
	ojas_w1(IAcc,Input_PIdPs,Output,[]).
ojas_w1([{IPId,Is}|IAcc],[{IPId,WPs}|Input_PIdPs],Output,Acc)->
	Updated_WPs = ojas_rule_w(Is,WPs,Output,[]),
	ojas_w1(IAcc,Input_PIdPs,Output,[{IPId,Updated_WPs}|Acc]);
ojas_w1([],[],_Output,Acc)->
	lists:reverse(Acc);
ojas_w1([],[{bias,WPs}],_Output,Acc)->
	lists:reverse([{bias,WPs}|Acc]).
%ojas_w/4 function operates on each Input_PIdP, calling the ojas_rule_w/4 function which processes each of the complementary Is and WPs lists, producing the Updated_WPs list in return, with the updated/adapted weights based on the oja's learning rule, using each synaptic weight's distinct learning parameter. 

	ojas_rule_w([I|Is],[{W,DW,LP,[H]}|WPs],[Output],Acc)->
		Updated_W = functions:saturation(W + H*Output*(I - Output*W),?SAT_LIMIT),
		ojas_rule_w(Is,WPs,[Output],[{Updated_W,DW,LP,[H]}|Acc]);
	ojas_rule_w([],[],_Output,Acc)->
		lists:reverse(Acc).
%ojas_weights/4 applies the ojas learning rule to each weight, using the input value I, the neuron's calculated output Output, and each weight's learning parameter H.

ojas({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	{PFName,ParameterList} = N#neuron.pf,
	Spread = ?SAT_LIMIT*10,
	MutationProb = 1/math:sqrt(length(ParameterList)),
	U_ParameterList = perturb(ParameterList,MutationProb,Spread,[]),
	U_PF = {PFName,U_ParameterList},
	N#neuron{pf=U_PF};
ojas(neural_parameters)->
	[(random:uniform()-0.5)];
ojas(weight_parameters)->
	[].
%oja/1 function produces the necessary parameter list for the oja's learning rule to operate. The parameter list for oja's learning rule is a list composed of a single parameter H: [H], used by the neuron for all its synaptic weights. If the learning parameter is positive, and the two connected neurons produce output signals of the same sign, then the postsynaptic neuron's synaptic weight increases. Otherwise it decreases.

ojas([_M,H],IAcc,Input_PIdPs,Output)->
	ojas(H,IAcc,Input_PIdPs,Output,[]).
ojas(H,[{IPId,Is}|IAcc],[{IPId,WPs}|Input_PIdPs],Output,Acc)->
	%io:format("ojas:~p~n",[{H,[{IPId,Is}|IAcc],[{IPId,WPs}|Input_PIdPs],Output,Acc}]),
	Updated_WPs = ojas_rule(H,Is,WPs,Output,[]),
	ojas(H,IAcc,Input_PIdPs,Output,[{IPId,Updated_WPs}|Acc]);
ojas(_H,[],[],_Output,Acc)->
	lists:reverse(Acc);
ojas(_H,[],[{bias,WPs}],_Output,Acc)->
	lists:reverse([{bias,WPs}|Acc]).
%ojas/5 function operates on each Input_PIdP, calling the ojas_rule/5 function which processes each of the complementary Is and WPs lists, producing the Updated_WPs list in return, with the updated/adapted weights based on the standard oja's learning rule. 

	ojas_rule(H,[I|Is],[{W,DW,LP,[]}|WPs],[Output],Acc)->
		%io:format("ojas:~p~n",[{H,I,W,Output}]),
		Updated_W = functions:saturation(W + H*Output*(I - Output*W),?SAT_LIMIT),
		ojas_rule(H,Is,WPs,[Output],[{Updated_W,DW,LP,[]}|Acc]);
	ojas_rule(_H,[],[],_Output,Acc)->
		lists:reverse(Acc).
%ojas_rule/5 updates every synaptic weight using Oja's learning rule.

self_modulationV1({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{input_idps=U_InputIdPs};
self_modulationV1(neural_parameters)->
	A=0.1,
	B=0,
	C=0,
	D=0,
	[A,B,C,D];
self_modulationV1(weight_parameters)->
	[(random:uniform()-0.5)].
	
self_modulationV1([_M,A,B,C,D],IAcc,Input_PIdPs,Output)->
	%io:format("[_M,A,B,C,D]:~p~n",[[_M,A,B,C,D]]),
	H = math:tanh(dot_productV1(IAcc,Input_PIdPs)),
	neuromodulation([H,A,B,C,D],IAcc,Input_PIdPs,Output,[]).
	
	dot_productV1(IAcc,IPIdPs)->
		dot_productV1(IAcc,IPIdPs,0).
	dot_productV1([{IPId,Input}|IAcc],[{IPId,WeightsP}|IPIdPs],Acc)->
		Dot = dotV1(Input,WeightsP,0),
		dot_productV1(IAcc,IPIdPs,Dot+Acc);
	dot_productV1([],[{bias,[{_Bias,[H_Bias]}]}],Acc)->
		Acc + H_Bias;
	dot_productV1([],[],Acc)->
		Acc.
	
		dotV1([I|Input],[{_W,DW,LP,[H_W]}|Weights],Acc) ->
			dotV1(Input,Weights,I*H_W+Acc);
		dotV1([],[],Acc)->
			Acc.
	
neuromodulation([H,A,B,C,D],[{IPId,Is}|IAcc],[{IPId,WPs}|Input_PIdPs],Output,Acc)->
	%io:format("[H,A,B,C,D]:~p~n",[[H,A,B,C,D]]),
	Updated_WPs = genheb_rule([H,A,B,C,D],Is,WPs,Output,[]),
	neuromodulation([H,A,B,C,D],IAcc,Input_PIdPs,Output,[{IPId,Updated_WPs}|Acc]);
neuromodulation(_NeuralParameters,[],[],_Output,Acc)->
	lists:reverse(Acc);
neuromodulation([H,A,B,C,D],[],[{bias,WPs}],Output,Acc)->
	Updated_WPs = genheb_rule([H,A,B,C,D],[1],WPs,Output,[]),
	lists:reverse([{bias,Updated_WPs}|Acc]).

	genheb_rule([H,A,B,C,D],[I|Is],[{W,DW,LP,Ps}|WPs],[Output],Acc)->
		%io:format("GenHeb[H,A,B,C,D]:~p~n",[[H,A,B,C,D]]),
		Updated_W = functions:saturation(W + H*(A*I*Output + B*I + C*Output + D),?SAT_LIMIT),
		genheb_rule([H,A,B,C,D],Is,WPs,[Output],[{Updated_W,DW,LP,Ps}|Acc]);
	genheb_rule(_NeuralLearningParameters,[],[],_Output,Acc)->
		lists:reverse(Acc).
%Updated_W(i)= W(i) + H*(A*I(i)*Output + B*I(i) + C*Output + D)

self_modulationV2({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	{PFName,[A|ParameterList]} = N#neuron.pf,
	[U_A] = perturb([A],0.5,?SAT_LIMIT*10,[]),
	U_PF = {PFName,[U_A|ParameterList]},
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{pf=U_PF,input_idps=U_InputIdPs};
self_modulationV2(neural_parameters)->
	A=(random:uniform()-0.5),
	B=0,
	C=0,
	D=0,
	[A,B,C,D];
self_modulationV2(weight_parameters)->
	[(random:uniform()-0.5)].
	
self_modulationV2([_M,A,B,C,D],IAcc,Input_PIdPs,Output)->
	H = math:tanh(dot_productV1(IAcc,Input_PIdPs)),
	neuromodulation([H,A,B,C,D],IAcc,Input_PIdPs,Output,[]).

self_modulationV3({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	{PFName,ParameterList} = N#neuron.pf,
	MSpread = ?SAT_LIMIT*10,
	MutationProb = 1/math:sqrt(length(ParameterList)),
	U_ParameterList = perturb(ParameterList,MutationProb,MSpread,[]),
	U_PF = {PFName,U_ParameterList},
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{pf=U_PF,input_idps=U_InputIdPs};
self_modulationV3(neural_parameters)->
	A=(random:uniform()-0.5),
	B=(random:uniform()-0.5),
	C=(random:uniform()-0.5),
	D=(random:uniform()-0.5),
	[A,B,C,D];
self_modulationV3(weight_parameters)->
	[(random:uniform()-0.5)].

self_modulationV3([_M,A,B,C,D],IAcc,Input_PIdPs,Output)->
	H = math:tanh(dot_productV1(IAcc,Input_PIdPs)),
	neuromodulation([H,A,B,C,D],IAcc,Input_PIdPs,Output,[]).

self_modulationV4({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{input_idps=U_InputIdPs};
self_modulationV4(neural_parameters)->
	B=0,
	C=0,
	D=0,
	[B,C,D];
self_modulationV4(weight_parameters)->
	[(random:uniform()-0.5),(random:uniform()-0.5)].

self_modulationV4([_M,B,C,D],IAcc,Input_PIdPs,Output)->
	{AccH,AccA} = dot_productV4(IAcc,Input_PIdPs),
	H = math:tanh(AccH),
	A = math:tanh(AccA),
	neuromodulation([H,A,B,C,D],IAcc,Input_PIdPs,Output,[]).
	
	dot_productV4(IAcc,IPIdPs)->
		dot_productV4(IAcc,IPIdPs,0,0).
	dot_productV4([{IPId,Input}|IAcc],[{IPId,WeightsP}|IPIdPs],AccH,AccA)->
		{DotH,DotA} = dotV4(Input,WeightsP,0,0),
		dot_productV4(IAcc,IPIdPs,DotH+AccH,DotA+AccA);
	dot_productV4([],[{bias,[{_Bias,[H_Bias,A_Bias]}]}],AccH,AccA)->
		{AccH + H_Bias,AccA+A_Bias};
	dot_productV4([],[],AccH,AccA)->
		{AccH,AccA}.
	
		dotV4([I|Input],[{_W,DW,LP,[H_W,A_W]}|Weights],AccH,AccA) ->
			dotV4(Input,Weights,I*H_W+AccH,I*A_W+AccA);
		dotV4([],[],AccH,AccA)->
			{AccH,AccA}.

self_modulationV5({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	{PFName,ParameterList} = N#neuron.pf,
	MSpread = ?SAT_LIMIT*10,
	MutationProb = 1/math:sqrt(length(ParameterList)),
	U_ParameterList = perturb(ParameterList,MutationProb,MSpread,[]),
	U_PF = {PFName,U_ParameterList},
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{pf=U_PF,input_idps=U_InputIdPs};
self_modulationV5(neural_parameters)->
	B=(random:uniform()-0.5),
	C=(random:uniform()-0.5),
	D=(random:uniform()-0.5),
	[B,C,D];
self_modulationV5(weight_parameters)->
	[(random:uniform()-0.5),(random:uniform()-0.5)].

self_modulationV5([_M,B,C,D],IAcc,Input_PIdPs,Output)->
	{AccH,AccA} = dot_productV4(IAcc,Input_PIdPs),
	H = math:tanh(AccH),
	A = math:tanh(AccA),
	neuromodulation([H,A,B,C,D],IAcc,Input_PIdPs,Output,[]).

self_modulationV6({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	InputIdPs = N#neuron.input_idps,
	U_InputIdPs=perturb_parameters(InputIdPs,?SAT_LIMIT),
	N#neuron{input_idps=U_InputIdPs};
self_modulationV6(neural_parameters)->
	[];
self_modulationV6(weight_parameters)->
	H = (random:uniform()-0.5),
	A = (random:uniform()-0.5),
	B = (random:uniform()-0.5),
	C = (random:uniform()-0.5),
	D = (random:uniform()-0.5),
	[H,A,B,C,D].

self_modulationV6([_M],IAcc,Input_PIdPs,Output)->
	{AccH,AccA,AccB,AccC,AccD} = dot_productV6(IAcc,Input_PIdPs),
	H = math:tanh(AccH),
	A = math:tanh(AccA),
	B = math:tanh(AccB),
	C = math:tanh(AccC),
	D = math:tanh(AccD),
	neuromodulation([H,A,B,C,D],IAcc,Input_PIdPs,Output,[]).
	
	dot_productV6(IAcc,IPIdPs)->
		dot_productV6(IAcc,IPIdPs,0,0,0,0,0).
	dot_productV6([{IPId,Input}|IAcc],[{IPId,WeightsP}|IPIdPs],AccH,AccA,AccB,AccC,AccD)->
		{DotH,DotA,DotB,DotC,DotD} = dotV6(Input,WeightsP,0,0,0,0,0),
		dot_productV6(IAcc,IPIdPs,DotH+AccH,DotA+AccA,DotB+AccB,DotC+AccC,DotD+AccD);
	dot_productV6([],[{bias,[{_Bias,[H_Bias,A_Bias,B_Bias,C_Bias,D_Bias]}]}],AccH,AccA,AccB,AccC,AccD)->
		{AccH+H_Bias,AccA+A_Bias,AccB+B_Bias,AccC+C_Bias,AccD+D_Bias};
	dot_productV6([],[],AccH,AccA,AccB,AccC,AccD)->
		{AccH,AccA,AccB,AccC,AccD}.
	
		dotV6([I|Input],[{_W,_DW,_LP,[H_W,A_W,B_W,C_W,D_W]}|Weights],AccH,AccA,AccB,AccC,AccD) ->
			dotV6(Input,Weights,I*H_W+AccH,I*A_W+AccA,I*B_W+AccB,I*C_W+AccC,I*D_W+AccD);
		dotV6([],[],AccH,AccA,AccB,AccC,AccD)->
			{AccH,AccA,AccB,AccC,AccD}.

neuromodulation({N_Id,mutate})->
	random:seed(now()),
	N = genotype:read({neuron,N_Id}),
	{PFName,ParameterList} = N#neuron.pf,
	MSpread = ?SAT_LIMIT*10,
	MutationProb = 1/math:sqrt(length(ParameterList)),
	U_ParameterList = perturb(ParameterList,MutationProb,MSpread,[]),
	U_PF = {PFName,U_ParameterList},
	N#neuron{pf=U_PF};	
neuromodulation(neural_parameters)->
	H = (random:uniform()-0.5),
	A = (random:uniform()-0.5),
	B = (random:uniform()-0.5),
	C = (random:uniform()-0.5),
	D = (random:uniform()-0.5),
	[H,A,B,C,D];
neuromodulation(weight_parameters)->
	[].
	
neuromodulation([M,H,A,B,C,D],IAcc,Input_PIdPs,Output)->
	%io:format("Neuromodulation:~p~n",[{M,H,A,B,C,D,IAcc,Input_PIdPs,Output}]),
	Modulator = scale_dzone(M,0.33,?SAT_LIMIT),
	neuromodulation([Modulator*H,A,B,C,D],IAcc,Input_PIdPs,Output,[]).
	
scale_dzone(Val,Threshold,MaxMagnitude)->
	if 
		Val > Threshold ->
			(functions:scale(Val,MaxMagnitude,Threshold)+1)*MaxMagnitude/2;
		Val < -Threshold ->
			(functions:scale(Val,-Threshold,-MaxMagnitude)-1)*MaxMagnitude/2;
		true ->
			0
	end.
								
	scale(Val,Max,Min)-> %Nm = (Y*2 - (Max + Min))/(Max-Min)
		case Max == Min of
			true ->
				0;
			false ->
				(Val*2 - (Max+Min))/(Max-Min)
		end.

sat_dzone(Val,Max,Min,DZMax,DZMin)->
	case (Val < DZMax) and (Val > DZMin) of
		true ->
			0;
		false ->
			sat(Val,Max,Min)
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
