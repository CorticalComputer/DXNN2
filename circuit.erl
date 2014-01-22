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

-module(circuit).
-compile(export_all).
-include("records.hrl").
-define(OUTPUT_SAT_LIMIT,math:pi()).
-define(BACKPROP,true).
-record(info,{name,ivl,ovl,trn_end,val_end,tst_end}).	
-define(f(X), is_float(X)).	
%[{circuit,tanh2tanh1}] %[{circuit,{static|dynamic,[{NeuronType::tanh|sin|rbf|gaussian|gabor_2d, LayerSize::integer(), static|dynamic}]}}]
%dynamic: {dynamic,[{tanh|sin|gaussian|linear,2,dynamic},{tanh|sin|gaussian|linear,1,static}]}
%perceptron_2l: {static,[{tanh,2,dynamic},{tanh,1,static}]}
%rbf: {static,[{rbf,2,dynamic},{linear,1,static}]}
%fourier: {static,[{sin,2,dynamic},{linear,1,static}]}
%gabor_2d:{static,[{gabor_2d,2,dynamic},{linear,1,static}]}
%mlffnn: {dynamic,[{tanh,2,dynamic},{tanh,1,static}]}
%[competitive,tanh]: {static,[{competitive,2,dynamic},{tanh,1,static}]}
%[hebbian,tanh]: {static,[{hebbian,2,dynamic},{tanh,1,static}]}
%[hebbian,competitive,tanh]: {static,[{hebbian,2,dynamic},{competitive,2,dynamic},{tanh,1,static}]}

%-record(circuit,{
%	id,
%	i,%input_idps::[{id(),vl}]
%	ovl,%int()
%	ivl,
%	training,%{TrainingType::bp|rbm|ga,TrainingLength::int()|{validation_goal,float()}}
%	output,%[float()]
%	parameters,%list()
%	dynamics,%static|dynamic
%	layers,%[#neurode|#layer|#circuit]
%	type=standard,%standard|dae|ae|sdae|sae|{pooling,max|avg|min}
%	noise,%float()|undefined
%	noise_type=zero_mask,%zero_mask|gaussian|saltnpepper|undefined
%	lp_decay=0.999999,%float()
%	lp_min=0.0000001,%float()
%	lp_max=0.1,%float()
%	memory=[],%[list()]
%	memory_size={0,100000},%{int(),int()}
%	validation,%[float()]
%	testing,%[float()]
%	receptive_field=full,%full|int()
%	step=0,%int()
%	block_size=100,%int()
%	err_acc=0,
%	backprop_tuning=off,
%	training_length=1000
%}).
%-record(layer,{
%	id,%Z::float()
%	type,
%	noise,
%	neurode_type=tanh,%tanh|sin|cos|rbf|cplx1/2/3/4/5/6/7|gabor_2d
%	dynamics=dynamic,
%	neurodes=[],%[#neurode]
%	tot_neurodes,%int()
%	input,%[float()]
%	output,%[float()]
%	ivl,%int()
%	encoder=[],%[neurode]
%	decoder=[],%[#neurode]
%	backprop_tuning=off,%off|on
%	index_start,%int()
%	index_end,%int()
%	parameters=[]%[any()]
%}).
%-record(neurode,{
%	id,%[X::float(),Y::float()]
%	weights,%[float()]|[{float(),float(),float()}]
%	af,%tanh|sin|cos|rbf|cplx
%	bias,%float()|{float(),float(),float()}
%	parameters=[],%list()
%	dot_product%[float()]
%}).
%-record(layer_spec,{type,af,ivl,dynamics,receptive_field,step}).%[{dae|pooling|standard,af(),IVL::int(),static|dynamic,Receptive_Field::int(),Step::int()}]

tot_weights(C)->
	lists:sum(lists:flatten([[length(Neurode#neurode.weights)||Neurode<-Layer#layer.neurodes]||Layer <-C#circuit.layers])).%Called from exoself for neural count.

create_Circuit(I_IdPs,{micro,CircuitLayerSpec})->
	%io:format("I_IdPs:~p CircuitLayerSpec:~p~n",[I_IdPs,CircuitLayerSpec]),
	create_InitCircuit(I_IdPs,{static,CircuitLayerSpec});
create_Circuit(I_IdPs,CircuitLayerSpec)->
	Noise=0.25,
	TrainingSetup = {bp,10000},
	TestingSetup = {1000,1000},
	CircuitDynamics = static,
	Circuit=case CircuitLayerSpec#layer.type of 
		dae -> 
			create_DAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		cdae -> 
			create_CDAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		lrf_dae -> 
			create_LRF_DAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		dnn ->
			create_DNN(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		competitive ->
			create_Competitive(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		standard ->
			create_Standard(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)
	end.

create_InitCircuit(Input_Specs,{CircuitDynamics,CircuitLayersSpec})->
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-Input_Specs]),
	InitLayers = create_InitLayers(CircuitLayersSpec,IVL,[]),
	[OutputLayer|_]=lists:reverse(CircuitLayersSpec),
	%io:format("InitLayers:~p~n",[InitLayers]),
	#circuit{i=Input_Specs,ivl=IVL,dynamics=CircuitDynamics,layers=InitLayers,ovl=OutputLayer#layer.tot_neurodes}.
	
	create_InitLayers([L|CircuitLayersSpec],IVL,Acc)->
		AF = create_af(L#layer.neurode_type),
		%io:format("LS:~p NIVL:~p~n",[L#layer.tot_neurodes,{Neurode_IVL,IVL,L#layer.receptive_field}]),
		Weights = create_weights(AF,IVL),
		U_L=L#layer{
			neurodes=[#neurode{
				id=genotype:generate_UniqueId(),
				af=AF,
				weights=case ?BACKPROP of 
					true -> create_weightsp(AF,IVL);
					false -> create_weights(AF,IVL)
				end,
				bias = case ?BACKPROP of
					true ->
						%[Bias]=create_weightsp(AF,1),
						%Bias;
						undefined;
					false -> undefined
				end,
				parameters=create_parameters(AF)} || _<-lists:seq(1,L#layer.tot_neurodes)
			],
			ivl = IVL
			%receptive_field=Receptive_Field,
			%step=Step
		},
		create_InitLayers(CircuitLayersSpec,U_L#layer.tot_neurodes,[U_L|Acc]);
	create_InitLayers([],_IVL,Acc)->
		lists:reverse(Acc).
	
	create_af(NeuronType)->
		case NeuronType of
			all ->
				AFs = [tanh,sin,cos,absolute,gaussian,linear,sigmoid],
				lists:nth(random:uniform(length(AFs)),AFs);
			fourier ->
				AFs = [sin,cos],
				lists:nth(random:uniform(length(AFs)),AFs);
			AF ->
				AF
		end.
		
	create_weights(AF,VL)->
		case AF of
			gabor_2d ->
				[random:uniform()-0.5|| _<-lists:seq(1,VL*2)];
			cplx1 ->
				[{random:uniform()-0.5,random:uniform()-0.5}|| _<-lists:seq(1,VL*2)];
			cplx2 ->
				[{random:uniform()-0.5,random:uniform()-0.5}|| _<-lists:seq(1,VL*2)];
			cplx3 ->
				[{random:uniform()-0.5,random:uniform()-0.5}|| _<-lists:seq(1,VL*2)];
			cplx4 ->
				[{random:uniform()-0.5,random:uniform()-0.5}|| _<-lists:seq(1,VL*2)];
			cplx5 ->
				[{random:uniform()-0.5,random:uniform()-0.5}|| _<-lists:seq(1,VL*2)];
			cplx6 ->
				[{random:uniform()-0.5,random:uniform()-0.5}|| _<-lists:seq(1,VL*2)];
			cplx7 ->
				[{random:uniform()-0.5,random:uniform()-0.5}|| _<-lists:seq(1,VL*2)];
			avg_pooling->
				undefined;
			max_pooling->
				undefined;
			competitive ->
				functions:normalize([random:uniform()-0.5|| _<-lists:seq(1,VL)]);
			_->
				[random:uniform()-0.5|| _<-lists:seq(1,VL)]
		end.
	
	create_weightsp(AF,VL)->
		case AF of
			cplx1->
				[{{random:uniform()-0.5,random:uniform()-0.5},{0,0},{0.1,0.1}}|| _<-lists:seq(1,VL)];
			cplx2->
				[{{random:uniform()-0.5,random:uniform()-0.5},{0,0},{0.1,0.1}}|| _<-lists:seq(1,VL)];
			cplx3->
				[{{random:uniform()-0.5,random:uniform()-0.5},{0,0},{0.1,0.1}}|| _<-lists:seq(1,VL)];
			cplx4->
				[{{random:uniform()-0.5,random:uniform()-0.5},{0,0},{0.1,0.1}}|| _<-lists:seq(1,VL)];
			cplx5 ->
				[{{random:uniform()-0.5,random:uniform()-0.5},{0,0},{0.1,0.1}}|| _<-lists:seq(1,VL)];
			cplx6 ->
				[{{random:uniform()-0.5,random:uniform()-0.5},{0,0},{0.1,0.1}}|| _<-lists:seq(1,VL)];
			cplx7 ->
				[{{random:uniform()-0.5,random:uniform()-0.5},{0,0},{0.1,0.1}}|| _<-lists:seq(1,VL)];
			competitive ->
				%normalizep([{random:uniform()-0.5,0.0,0.1}|| _<-lists:seq(1,VL)]);
				functions:normalize([random:uniform()-0.5|| _<-lists:seq(1,VL)]);
				%[{random:uniform()-0.5,0.0,0.1}|| _<-lists:seq(1,VL)]
			avg_pooling->
				undefined;
			max_pooling->
				undefined;
			_->
				[{random:uniform()-0.5,0.0,0.1}|| _<-lists:seq(1,VL)]
		end.
	
	create_parameters(AF)->
		case AF of
			gabor_2d -> [random:uniform()-0.5,random:uniform()-0.5,random:uniform()-0.5,random:uniform()-0.5,random:uniform()-0.5];
			rbf -> [random:uniform()-0.5];
			_ -> []
		end.

transfer_function(IAcc,C,_Plasticity)->
	%io:format("C:~p~n",[C]),
	IVector=lists:flatten([Input||{_From,Input}<-IAcc]),
	case (C#circuit.training_length == 0) of
		true ->%pre_training is done, output_???_short is used...
			case C#circuit.type of
				standard ->
					calculate_output_std(IVector,C);
				dae->
					{calculate_output_dae_short(IVector,C),C};
				lrf_dae->
					{calculate_output_lrf_dae_short(IVector,C),C};
				dnn ->
					ok
			end;
		false ->%pre_training in proggress
			case C#circuit.type of%TODO THE OUTPUTS ARE NOISY, but no noise would require extra calculations, it's important to know if there is or isnt recurrent calculations. If none, then output void is ok.
				lrf_dae ->
					U_C=backprop_LRF_DAE(IVector,C),
					[Encoder|_]=U_C#circuit.layers,
					{Encoder#layer.output,U_C};
				dae ->
					%io:format("IAcc:~p IVector:~p~n",[IAcc,IVector]),
					U_C=backprop_DAE(IVector,C),
					%io:format("U_C:~p~n",[U_C]),
					[Encoder|_]=U_C#circuit.layers,
					%io:format("~p~n",[{Encoder#layer.output,U_C}]),
					{Encoder#layer.output,U_C};
				cdae ->
					U_C=backprop_CDAE(IVector,C),
					[Encoder|_]=U_C#circuit.layers,
					{Encoder#layer.output,U_C};
				standard ->%No pretraining in standard. 
					calculate_output_std(IVector,C);
				dnn ->
					ok
			end
	end.

validate(C)->
	case get(sleep) of
		true ->
			C;
		false ->
			TrainingLength=C#circuit.training_length,
			C#circuit{training_length= TrainingLength-1}
	end.

calculate_output_dae(IVector,C)->%Calculates output, and returns it and the updated Encoder/Decoder layers.
	{_Output,[Encoder,Decoder]} = calculate_output_std(IVector,C#circuit.layers,[]),
	U_C = C#circuit{layers=[Encoder,Decoder]},
	{Encoder#layer.output,U_C}.

calculate_output_dae_short(IVector,C)->%Calculates just the output of the DAE, no layers updated.
	[Encoder,_Decoder] = C#circuit.layers,
	{Output,_} = calculate_output_std(IVector,[Encoder],[]),
	Output.

calculate_output_lrf_dae(IVector,C)->%Calculates just the output of the DAE, no layers updated.
	[Encoder,Decoder] = C#circuit.layers,
	{Output,[U_Encoder]} = calculate_output_LRF(IVector,C#circuit.receptive_field,[Encoder],[]),
	U_C = C#circuit{
		layers=[U_Encoder,Decoder]
	},
	{Output,U_C}.

calculate_output_lrf_dae_short(IVector,C)->%Calculates just the output of the DAE, no layers updated.
	[Encoder,_Decoder] = C#circuit.layers,
	{Output,_} = calculate_output_LRF(IVector,C#circuit.receptive_field,[Encoder],[]),
	Output.

calculate_output_sdae(IVector,SDAE)->
	DAEs = SDAE#circuit.layers,
	{Output,U_DAEs}=calculate_output_sdae(IVector,DAEs,[]),
	{Output,SDAE#circuit{layers=U_DAEs}}.

	calculate_output_sdae(IVector,[DAE|DAEs],Acc)->
		{Output,U_DAE} = calculate_output_dae(IVector,DAE),
		calculate_output_sdae(Output,DAEs,[U_DAE|Acc]);
	calculate_output_sdae(Output,[],Acc)->
		{Output,lists:reverse(Acc)}.

calculate_output_sdaes_short(IVector,[PDAE|PDAEs])->%Calculates just output, DAEs are not updated with regards to their layers.
	Output=calculate_output_dae_short(IVector,PDAE),
	calculate_output_sdaes_short(Output,PDAEs);
calculate_output_sdaes_short(Output,[])->
	Output.

calculate_output_dnn(IVector,DNN)->
	Circuits = DNN#circuit.layers,
	{Output,U_Circuits}=calculate_output_dnn(IVector,Circuits,[]),
	{Output,DNN#circuit{layers=U_Circuits}}.
	
calculate_output_dnn(I,[C|Cs],Acc)->
	{Output,U_C}=case C#circuit.type of
		dae->
			calculate_output_dae(I,C);
		lrf_dae->
			calculate_output_lrf_dae(I,C);
		standard ->
			calculate_output_std(I,C)
	end,
	calculate_output_dnn(Output,Cs,[U_C|Acc]);
calculate_output_dnn(Output,[],Acc)->
	{Output,lists:reverse(Acc)}.

calculate_output_dnn_short(I,[C|Cs])->
	Output=case C#circuit.type of
		dae->
			calculate_output_dae_short(I,C);
		lrf_dae->
			calculate_output_lrf_dae_short(I,C);
		standard ->
			calculate_output_std_short(I,C)
	end,
	calculate_output_dnn_short(Output,Cs);
calculate_output_dnn_short(Output,[])->
	Output.
	
calculate_output_std_short(IVector,C)->
	{Output,_} = calculate_output_std(IVector,C#circuit.layers,[]),
	Output.
	
calculate_output_std(IVector,C)->
	{Output,U_Layers} = calculate_output_std(IVector,C#circuit.layers,[]),
	U_C = C#circuit{
		layers=U_Layers
	},
	{Output,U_C}.
	
calculate_output_std(IVector,[L|Layers],Acc)->
	%io:format("self():~p {IVector,L}:~p~n self:~p~n",[self(),{IVector,L},self()]),
	Neurodes = L#layer.neurodes,
	%io:format("Neurodes:~p~n",[Neurodes]),
	U_L=case ?BACKPROP of%TODO
		true ->
			[N|_]=Neurodes,
			%io:format("N:~p~n",[N]),
			case N#neurode.weights of
				[{{RW,IW},_,_}|_] ->
					Preprocessed_IVector=conditional_preprocessing(IVector),
					{Output,U_Neurodes} = calculate_neurodes_output(Preprocessed_IVector,Neurodes,[],[]),
					L#layer{
						input=Preprocessed_IVector,
						neurodes=U_Neurodes,
						output=Output
					};
				_ ->
					{Output,U_Neurodes} = calculate_neurodes_output(IVector,Neurodes,[],[]),
					%io:format("Output:~p~n",[Output]),
					L#layer{
						input=IVector,
						neurodes=U_Neurodes,
						output=Output
					}
			end;
		false ->
			Output = [calculate_neurode_output(IVector,N#neurode.weights,N#neurode.bias,N#neurode.af,N#neurode.parameters) || N <- Neurodes],
			L#layer{
				input=IVector,
				output=Output
			}
	end,
%	io:format("U_IVector:~p~n",[U_IVector]),
	calculate_output_std(Output,Layers,[U_L|Acc]);
calculate_output_std(Output,[],Acc)->
	%io:format("Output:~p~n",[Output]),
	{Output,lists:reverse(Acc)}.

calculate_output_competitive(IVector,C)->
	calculate_output_competitive(IVector,C#circuit.layers,[]).
calculate_output_competitive(IVector,[L],Acc)->%TODO: Nothing is done for the multi layer competitive network, or the Acc variable.
	%io:format("self():~p {IVector,L}:~p~n self:~p~n",[self(),{IVector,L},self()]),
	Neurodes = L#layer.neurodes,
	%io:format("Neurodes:~p~n",[Neurodes]),
	{_Dot,Index,N}=closest(IVector,Neurodes),
	%N#neurode.weights.
	compose_competitive_output(Index,length(Neurodes),[]).
	
	compose_competitive_output(Target_Index,0,Acc)->
		Acc;
	compose_competitive_output(Target_Index,Target_Index,Acc)->
		compose_competitive_output(Target_Index,Target_Index-1,[1|Acc]);
	compose_competitive_output(Target_Index,Index,Acc)->
		compose_competitive_output(Target_Index,Index-1,[0|Acc]).
	
	
	closest(Vector,[N|Neurodes])->
		Weights = N#neurode.weights,
		Dot = calculate_dot(Weights,Vector,0),
		closest(Vector,Neurodes,2,{Dot,1,N}).
	closest(Vector,[N|Neurodes],Index,{Closest,ClosestIndex,ClosestN})->
		Weights = N#neurode.weights,
		Dot = calculate_dot(Weights,Vector,0),
		case Dot > Closest of
			true ->
				closest(Vector,Neurodes,Index+1,{Dot,Index,N});
			false ->
				closest(Vector,Neurodes,Index+1,{Closest,ClosestIndex,ClosestN})
		end;
	closest(_Vector,[],Index,{ClosestDistance,ClosestIndex,ClosestN})->
		{ClosestDistance,ClosestIndex,ClosestN}.
		
		calculate_dot([A|As],[B|Bs],Acc)->
			calculate_dot(As,Bs,A*B+Acc);
		calculate_dot([],[],Acc)->
			Acc.
		
		calculate_distance([Val|Vector],[{W,_LP,_PDW}|Weights],Acc)->
			calculate_distance(Vector,Weights,Acc+abs(Val-W));
		calculate_distance([],[],Acc)->
			Acc.
			
calculate_output_LRF(IVector,Receptive_Field,[L|Decoder],Acc)->
	%io:format("{IVector,L}:~p~n self:~p~n",[{IVector,L},self()]),
	Neurodes = L#layer.neurodes,
	U_L=case ?BACKPROP of%TODO
		true ->
			[N|_]=Neurodes,
			%io:format("N:~p~n",[N]),
			case N#neurode.weights of
				[{{RW,IW},_,_}|_] ->
					Preprocessed_IVector=conditional_preprocessing(IVector),
					{IV_Head,IV_Tail} = lists:split(Receptive_Field,Preprocessed_IVector),
					{Output,U_Neurodes} = calculate_neurodes_output_LRF(IV_Head,IV_Tail,Neurodes,[],[]),
					L#layer{
						input=Preprocessed_IVector,
						neurodes=U_Neurodes,
						output=Output
					};
				_ ->
					{IV_Head,IV_Tail} = lists:split(Receptive_Field,IVector),
					{Output,U_Neurodes} = calculate_neurodes_output_LRF(IV_Head,IV_Tail,Neurodes,[],[]),
					L#layer{
						input=IVector,
						neurodes=U_Neurodes,
						output=Output
					}
				end;
		false ->
			Output = [calculate_neurode_output(IVector,N#neurode.weights,N#neurode.bias,N#neurode.af,N#neurode.parameters) || N <- Neurodes],
			L#layer{
				input=IVector,
				output=Output
			}
	end,
	calculate_output_std(Output,Decoder,[U_L|Acc]).

	calculate_neurode_output(IVector,Weights,Bias,rbf,Parameters)->
		calculate_neurode_output_rbf(IVector,Weights,Bias,Parameters,0);
	calculate_neurode_output(IVector,Weights,Bias,gabor_2d,Parameters)->
		calculate_neurode_output_gabor_2d(IVector,Weights,Bias,Parameters,0,0);
	calculate_neurode_output(IVector,Weights,Bias,AF,Parameters)->
		calculate_neurode_output_std(IVector,Weights,Bias,AF,0).
		
	calculate_neurodes_output(IVector,[N|Neurodes],OAcc,NAcc)->
		%io:format("IVector:~p Length:~p~n",[IVector,length(IVector)]),
		AF = N#neurode.af,
		DotProduct = dot_product(IVector,N#neurode.weights,N#neurode.bias),
		Output=functions:AF(DotProduct),
		calculate_neurodes_output(IVector,Neurodes,[Output|OAcc],[N#neurode{dot_product=DotProduct}|NAcc]);
	calculate_neurodes_output(_IVector,[],OAcc,NAcc)->
		{lists:reverse(OAcc),lists:reverse(NAcc)}.
	
	calculate_neurodes_output_LRF(IV_Head,[E|IV_Tail],[N|Neurodes],OAcc,NAcc)->
		AF = N#neurode.af,
		DotProduct = dot_product(IV_Head,N#neurode.weights,N#neurode.bias),
		Output=functions:AF(DotProduct),
		[_|U_IV_Head]=IV_Head,
		calculate_neurodes_output_LRF(lists:append(U_IV_Head,[E]),IV_Tail,Neurodes,[Output|OAcc],[N#neurode{dot_product=DotProduct,i=IV_Head}|NAcc]);
	calculate_neurodes_output_LRF(IV_Head,[],[N],OAcc,NAcc)->
		AF = N#neurode.af,
		DotProduct = dot_product(IV_Head,N#neurode.weights,N#neurode.bias),
		Output=functions:AF(DotProduct),
		{lists:reverse([Output|OAcc]),lists:reverse([N#neurode{dot_product=DotProduct,i=IV_Head}|NAcc])}.		

		dot_product([{RI,II}|IVector],CplxWeights,CplxBias)->
			calculate_cplx_dot([{RI,II}|IVector],CplxWeights,CplxBias,{0.0,0.0});
		dot_product(IVector,Weights,Bias)->
			DotProduct = dot_product(IVector,Weights,Bias,0.0).
			
		dot_product([I|IVector],[{W,_PDW,_LP}|Weights],Bias,Acc) when ?f(I), ?f(W), ?f(Acc)->
			%io:format("I1:~p~n",[I]),
			dot_product(IVector,Weights,Bias,I*W+Acc);
		dot_product([I|IVector],[{W,_PDW,_LP}|Weights],Bias,Acc) ->
			%io:format("I2:~p~n",[I]),
			dot_product(IVector,Weights,Bias,I*W+Acc);
		dot_product([],[],undefined,Acc)->
			Acc;
		dot_product([],[],{Bias,_PB,_LP},Acc)->
			Bias+Acc.
		
		calculate_neurode_output_rbf([I|IVector],[Weight|Weights],Bias,Parameters,Acc)->
			calculate_neurode_output_rbf(IVector,Weights,Bias,Parameters,math:pow(I-Weight,2)+Acc);
		calculate_neurode_output_rbf([],[],undefined,Parameters,Acc)->
			[Diligence] = Parameters,
			math:exp((Acc)/math:pow(Diligence,2));
		calculate_neurode_output_rbf([],[],Bias,Parameters,Acc)->
			[Diligence] = Parameters,
			math:exp((-Acc+Bias)/math:pow(Diligence,2)).
			
		calculate_neurode_output_gabor_2d([I|IVector],[WeightX,WeightY|Weights],Bias,Parameters,AccX,AccY)->
			calculate_neurode_output_gabor_2d(IVector,Weights,Bias,Parameters,I*WeightX+AccX,I*WeightY+AccY);
		calculate_neurode_output_gabor_2d([],[],undefined,Parameters,AccX,AccY)->
			[Sigma,Theta,Lambda,Psi,Gamma]=Parameters,
			wavelet:gabor_2d(AccX,AccY,Sigma,Theta,Lambda,Psi,Gamma);	
		calculate_neurode_output_gabor_2d([],[],[XBias],Parameters,AccX,AccY)->
			[Sigma,Theta,Lambda,Psi,Gamma]=Parameters,
			wavelet:gabor_2d(AccX+XBias,AccY,Sigma,Theta,Lambda,Psi,Gamma);
		calculate_neurode_output_gabor_2d([],[],[XBias,YBias],Parameters,AccX,AccY)->
			[Sigma,Theta,Lambda,Psi,Gamma]=Parameters,
			wavelet:gabor_2d(AccX+XBias,AccY+YBias,Sigma,Theta,Lambda,Psi,Gamma).

		calculate_neurode_output_std([I|IVector],[Weight|Weights],Bias,AF,Acc)->
			calculate_neurode_output_std(IVector,Weights,Bias,AF,I*Weight+Acc);
		calculate_neurode_output_std([],[],undefined,AF,Acc)->
			functions:AF(Acc);
		calculate_neurode_output_std([],[],Bias,AF,Acc)->
			functions:AF(Acc+Bias).

plasticity_function(DIV,Output,Circuit)->
	ok.
	
perturb_circuit(Circuit,DMultiplier)->
	%io:format("Circuit:~p~n DMultiplier:~p~n",[Circuit,DMultiplier]),
	random:seed(now()),
	CircuitDynamics = Circuit#circuit.dynamics,
%	CircuitLayersSpecs = Circuit#circuit.layers_spec,
	%{add_neurode,0},{add_layer,0},
	MutationOperators = case CircuitDynamics of
		static -> [{perturb_weights,100}];%,{add_bias,2},{remove_bias,2}];
		dynamic -> [{add_layer,1},{perturb_weights,95},{add_bias,3},{remove_bias,3},{add_neurode,2}]
	end,
	Tot=lists:sum([RelativeProbability||{_,RelativeProbability}<-MutationOperators]),
%	io:format("MutationOperators:~p~n Tot:~p~n",[MutationOperators,Tot]),
	Mutagen=get_RandomMutagen(0,MutationOperators,random:uniform(Tot)),
%	io:format("Applying Mutagen:~p to Circuit:~p in neuron:~p~n",[Mutagen,Circuit,self()]),
	case circuit:Mutagen(Circuit#circuit.layers,DMultiplier) of
		not_supported->
			perturb_circuit(Circuit,DMultiplier);
		U_Layers ->
			%io:format("U_CircuitSpecs:~p U_Circuit:~p in neuron:~p~n",[U_CircuitSpecs,U_Circuit,self()]),
			Circuit#circuit{layers=U_Layers}
	end.
				
	get_RandomMutagen(Range_From,[{Mutagen,Prob}|Mutagens],Choice)->
%		io:format("Choice:~p~n",[Choice]),
		Range_To = Range_From+Prob,
		case (Choice >= Range_From) and (Choice =< Range_To) of
			true ->
				Mutagen;
			false ->
				get_RandomMutagen(Range_To,Mutagens,Choice)
		end;
	get_RandomMutagen(_Rage_From,[],_Choice)->
		exit("********ERROR:get_RandomMutagen:: in get_RandomMutagen(Mutagens,Choice), Mutagens reached []").

	perturb_weights(Layers,DMultiplier)->
		TotWeights=lists:sum(lists:flatten([[length(N#neurode.weights)||N<-NeurodeLayer#layer.neurodes] || NeurodeLayer<-Layers])),
		MP = 1/math:sqrt(TotWeights),
		U_Layers = [perturb_Neurodes(NeurodeLayer,DMultiplier,MP,[])||NeurodeLayer<-Layers],
		U_Layers.

		perturb_Neurodes(NeurodeLayer,DMultiplier,MP,Acc)->
			U_Neurodes=[N#neurode{
				weights=[perturb(W,MP,DMultiplier)||W<-N#neurode.weights],
				parameters=[perturb(W,MP,DMultiplier)||W<-N#neurode.parameters],
				bias=perturb(N#neurode.bias,MP,DMultiplier)} 
			|| N <- NeurodeLayer#layer.neurodes],
			NeurodeLayer#layer{neurodes=U_Neurodes}.

			perturb_Weights([{W,PDW,LP}|Weights],MP,DMultiplier,Acc)->
				%[perturb(W,MP,DMultiplier) || W<-Weights]
				WLimit = ?OUTPUT_SAT_LIMIT,
				%io:format("Perturb_W:~p~n",[{W,MP,DMultiplier}]),
				U_W=case random:uniform() < MP of
					true ->
						DW = (random:uniform()-0.5)*DMultiplier,
						%io:format("DW:~p~n",[DW]),
						{functions:sat(W + DW +0.5*PDW,WLimit,-WLimit),DW,LP};
					false ->
						{W,PDW,LP}
				end;
			perturb_Weights([W|Weights],MP,DMultiplier,Acc)->
				%[perturb(W,MP,DMultiplier) || W<-Weights]
				WLimit = ?OUTPUT_SAT_LIMIT,
				%io:format("Perturb_W:~p~n",[{W,MP,DMultiplier}]),
				U_W=case random:uniform() < MP of
					true ->
						DW = (random:uniform()-0.5)*DMultiplier,
						%io:format("DW:~p~n",[DW]),
						functions:sat(W + DW,WLimit,-WLimit);
					false ->
						W
				end,
				perturb_Weights(Weights,MP,DMultiplier,[U_W|Acc]);
			perturb_Weights({W,PDW,LP},MP,DMultiplier,[])->%BIAS
				%[perturb(W,MP,DMultiplier) || W<-Weights]
				WLimit = ?OUTPUT_SAT_LIMIT,
				%io:format("Perturb_W:~p~n",[{W,MP,DMultiplier}]),
				U_W=case random:uniform() < MP of
					true ->
						DW = (random:uniform()-0.5)*DMultiplier,
						%io:format("DW:~p~n",[DW]),
						{functions:sat(W + DW +0.5*PDW,WLimit,-WLimit),DW,LP};
					false ->
						{W,PDW,LP}
				end;
			perturb_Weights(W,MP,DMultiplier,[])->%BIAS
				%[perturb(W,MP,DMultiplier) || W<-Weights]
				WLimit = ?OUTPUT_SAT_LIMIT,
				%io:format("Perturb_W:~p~n",[{W,MP,DMultiplier}]),
				U_W=case random:uniform() < MP of
					true ->
						DW = (random:uniform()-0.5)*DMultiplier,
						%io:format("DW:~p~n",[DW]),
						functions:sat(W + DW,WLimit,-WLimit);
					false ->
						W
				end;
			perturb_Weights(undefined,_MP,_DMultiplier,[])->
				undefined;
			perturb_Weights([],_MP,_DMultiplier,Acc)->
				lists:reverse(Acc).
			
			perturb(undefined,_MP,_DMultiplier)->
				undefined;
			perturb({W,PDW,LP},MP,DMultiplier)->
				WLimit = ?OUTPUT_SAT_LIMIT,
				case random:uniform() < MP of
					true ->
						DW = (random:uniform()-0.5)*DMultiplier,
						{functions:sat(W + DW +0.5*PDW,WLimit,-WLimit),DW,LP};
					false ->
						{W,PDW,LP}
				end;
			perturb(W,MP,DMultiplier)->
				WLimit = ?OUTPUT_SAT_LIMIT,
				case random:uniform() < MP of
					true ->
						DW = (random:uniform()-0.5)*DMultiplier,
						functions:sat(W + DW,WLimit,-WLimit);
					false ->
						W
				end.
				
	add_neurode(Layers,_DMultiplier)->
		TotLayers = length(Layers),
		LayerIndex = random:uniform(TotLayers),
		L = lists:nth(LayerIndex,Layers),
		case L#layer.dynamics of
			dynamic ->%perform add_neurode
				AF = create_af(L#layer.neurode_type),
				U_Layers = circuit:add_neurode(Layers,LayerIndex,L#layer.ivl,AF),
				U_Layers;
			static ->%No new neurodes allowed in this layer.
				not_supported
		end.
		
		insert_after(Index,Element,List)->
			insert_after(Index,1,Element,List,[]).
		insert_after(Index,Index,NewElement,[Element|List],Acc)->
			insert_after(Index,Index+1,NewElement,List,[NewElement,Element|Acc]);
		insert_after(TargetIndex,Index,NewElement,[Element|List],Acc)->
			insert_after(TargetIndex,Index+1,NewElement,List,[Element|Acc]);
		insert_after(_TargetIndex,_Index,_NewElement,[],Acc)->
			lists:reverse(Acc).
			
		replace(Index,Element,List)->
			replace(Index,1,Element,List,[]).
		replace(TargetIndex,TargetIndex,NewElement,[TargetElement|List],Acc)->
			replace(TargetIndex,TargetIndex+1,NewElement,List,[NewElement|Acc]);
		replace(TargetIndex,Index,NewElement,[Element|List],Acc)->
			replace(TargetIndex,Index+1,NewElement,List,[Element|Acc]);
		replace(_TargetIndex,_Index,_NewElement,[],Acc)->
			lists:reverse(Acc).
		
	add_layer(Layers,_DMultiplier)->%TODO:New layer is default
		TotLayers = length(Layers),
		LayerIndex = random:uniform(TotLayers),
		LayerSize = case LayerIndex == TotLayers of
			true ->
				[NeurodeLayer|_] = lists:reverse(Layers),
				length(NeurodeLayer#layer.neurodes);
			false ->
				random:uniform(3)
		end,
		U_Layers = add_layer(Layers,LayerIndex,LayerSize,[]),
		U_Layers.
		
		
		
		delete_weights(TargetIndex,TotWeights,Weights)->
			%io:format("TargetIndex:~p~n TotWeight:~p~n Weights:~p~n",[TargetIndex,TotWeights,Weights]),
			delete_weights(1,TargetIndex,TotWeights,Weights,[]).
		delete_weights(_TargetIndex,_TargetIndex,0,Weights,Acc)->
			lists:reverse(Acc)++Weights;
		delete_weights(TargetIndex,TargetIndex,WeightIndex,[_W|Weights],Acc)->
			delete_weights(TargetIndex,TargetIndex,WeightIndex-1,Weights,Acc);
		delete_weights(Index,TargetIndex,WeightIndex,[W|Weights],Acc)->
			delete_weights(Index+1,TargetIndex,WeightIndex,Weights,[W|Acc]).
		
		add_weights(TargetIndex,VL,AF,Weights)->add_weights(1,TargetIndex,VL,AF,Weights,[]).
		%add_weights(_TargetIndex,_TargetIndex,0,Weights,Acc)->
		%	lists:reverse(Acc)++Weights;
		add_weights(TargetIndex,TargetIndex,VL,AF,Weights,Acc)->
			%add_weights(TargetIndex,TargetIndex,WeightIndex-1,Weights,[random:uniform()-0.5|Acc]);
			NewWeights = case ?BACKPROP of 
				true -> create_weightsp(AF,VL);
				false -> create_weights(AF,VL)
			end,
			lists:reverse(lists:append(NewWeights,Acc))++Weights;
		add_weights(Index,TargetIndex,VL,AF,[W|Weights],Acc)->
			add_weights(Index+1,TargetIndex,VL,AF,Weights,[W|Acc]).	
		
	add_bias(Layers,_DMultiplier)->
		MP = 1/math:sqrt(lists:sum([length(L#layer.neurodes)||L<-Layers])),
		U_Layers=[L#layer{neurodes=[N#neurode{bias=add_bias1(N#neurode.af,N#neurode.bias,MP)}|| N <- L#layer.neurodes]}||L<-Layers],
		U_Layers.
		
		add_bias1(AF,Val,MP)->
			case Val of
				undefined ->
					case random:uniform() < MP of
						true ->
							case ?BACKPROP of
								true ->
									[Bias]=create_weightsp(AF,1),
									Bias;
								false -> 
									random:uniform()-0.5
							end;
						false ->
							Val
					end;
				_ ->
					Val
			end.
			
	remove_bias(Layers,_DMultiplier)->
		MP = 1/math:sqrt(lists:sum([length(L#layer.neurodes)||L<-Layers])),
		U_Layers=[L#layer{neurodes=[N#neurode{bias=remove_bias1(N#neurode.bias,MP)}|| N <- L#layer.neurodes]}||L<-Layers],
		U_Layers.
		
		remove_bias1(Val,MP)->
			case Val of
				undefined ->
					undefined;
				Val ->
					case random:uniform() < MP of
						true ->
							undefined;
						false ->
							Val
					end
			end.
		
		add_neurode(Layers,LayerIndex,VL,AF)->%The "TrailingLayer" is actually the one ahead, sin signals go from back to front
			add_neurode(Layers,LayerIndex,VL,AF,[]).
		add_neurode([L|Layers],1,VL,AF,Acc)->
			case Layers of
				[] ->
					U_TotNeurodes = L#layer.tot_neurodes+1,
					U_L=L#layer{neurodes=[#neurode{id=genotype:generate_UniqueId(),af=AF,weights=add_weights(1,VL,AF,[])}|L#layer.neurodes],tot_neurodes=U_TotNeurodes},
					lists:reverse(Acc)++[U_L];
				[TL|LayersRemainder] ->
					U_TotNeurodes = L#layer.tot_neurodes+1,
					U_L=L#layer{neurodes=[#neurode{id=genotype:generate_UniqueId(),af=AF,weights=add_weights(1,VL,AF,[])}|L#layer.neurodes],tot_neurodes=U_TotNeurodes},
					U_TL=TL#layer{neurodes=[N#neurode{weights=add_weights(1,1,N#neurode.af,N#neurode.weights)}||N<-TL#layer.neurodes]},
					lists:reverse(Acc)++[U_L]++[U_TL]++LayersRemainder
			end;
		add_neurode([NeurodeLayer|Layer],LayerIndex,VL,AF,Acc)->
			add_neurode(Layer,LayerIndex-1,VL,AF,[NeurodeLayer|Acc]).
			
		delete_neurode(Layers,LayerIndex)->
			delete_neurode(Layers,LayerIndex,[]).
		delete_neurode([L|Layers],1,Acc)->
			case Layers of
				[] ->
					[_|U_Neurodes] = L#layer.neurodes,
					U_TotNeurodes = L#layer.tot_neurodes-1,
					U_L=L#layer{neurodes=U_Neurodes,tot_neurodes=U_TotNeurodes},
					case U_Neurodes of
						[] ->
							lists:reverse(Acc)++[L];
						_ ->
							lists:reverse(Acc)++[U_L]
					end;
				[TL|LayersRemainder] ->
					%[_|U_NeurodeLayer] = NeurodeLayer,
					[_|U_Neurodes] = L#layer.neurodes,
					U_TotNeurodes = L#layer.tot_neurodes-1,
					case U_Neurodes of
						[] ->
							lists:reverse(Acc)++[L]++[TL]++LayersRemainder;
						_ ->
							U_TL=TL#layer{neurodes=[N#neurode{weights=delete_weights(1,1,1,N#neurode.weights,[])}||N<-TL#layer.neurodes]},
							U_L=L#layer{neurodes=U_Neurodes,tot_neurodes=U_TotNeurodes},
							
							lists:reverse(Acc)++[U_L]++[U_TL]++LayersRemainder
					end
			end;
		delete_neurode([NeurodeLayer|Layers],LayerIndex,Acc)->
			delete_neurode(Layers,LayerIndex-1,[NeurodeLayer|Acc]).
			
		add_layer([L|Layers],1,LayerSize,Acc)->%New_LayerSpecs = {any,LayerSize,dynamic,full,0},TODO: adding a default neurode layer, need to be able to add custom
			TotWeights=length(L#layer.neurodes),
			NewNeurodes=[#neurode{id=genotype:generate_UniqueId(),af=tanh,weights=add_weights(1,TotWeights,tanh,[])}||_<-lists:seq(1,LayerSize)],
			NewLayer = #layer{ivl=TotWeights,neurodes=NewNeurodes,tot_neurodes=LayerSize},
			case Layers of
				[FL|RemainingLayer]->
					Diff = LayerSize-TotWeights,
					case Diff < 0 of
						true ->
							U_FL=FL#layer{neurodes=[N#neurode{weights=delete_weights(1,abs(Diff),N#neurode.weights)}||N<-FL#layer.neurodes]},
							case RemainingLayer of
								[] ->
									lists:reverse([L|Acc])++[NewLayer]++[U_FL];
								_->
									lists:reverse([L|Acc])++[NewLayer]++[U_FL]++RemainingLayer
							end;
						false ->
							U_FL=FL#layer{neurodes=[N#neurode{weights=add_weights(1,Diff,N#neurode.af,N#neurode.weights)}||N<-FL#layer.neurodes]},
							case RemainingLayer of
								[] ->
									lists:reverse([L|Acc])++[NewLayer]++[U_FL];
								_ ->
									lists:reverse([L|Acc])++[NewLayer]++[U_FL]++RemainingLayer
							end
					end;
				[] ->
					lists:reverse([L|Acc])++[NewLayer]
			end;			
		add_layer([Layer|Layers],LayerIndex,LayerSize,Acc)->
			add_layer(Layers,LayerIndex-1,LayerSize,[Layer|Acc]).

link_ToCircuit(C,{FromId,FromOVL})->
	%io:format("link_ToNeuron::~n AF:~p~n C#circuit.i:~p~n",[ToN#neuron.af,C#circuit.i]),
	case C#circuit.type of
		standard ->
			[NeurodeLayer|Substrate]=C#circuit.layers,
			U_Neurodes=[Neurode#neurode{weights=circuit:add_weights(1,FromOVL,Neurode#neurode.af,Neurode#neurode.weights)}||Neurode<-NeurodeLayer#layer.neurodes],
			U_NeurodeLayer = NeurodeLayer#layer{neurodes=U_Neurodes},
			U_ToI = [{FromId,FromOVL}|C#circuit.i],
			C#circuit{
				i = U_ToI,
				layers=[U_NeurodeLayer|Substrate]
			};
		dae ->
			[Encoder,Decoder]=C#circuit.layers,
			U_ENeurodes=[Neurode#neurode{weights=circuit:add_weights(1,FromOVL,Neurode#neurode.af,Neurode#neurode.weights)}||Neurode<-Encoder#layer.neurodes],
			U_Encoder = Encoder#layer{neurodes=U_ENeurodes},
			U_TotNeurodes = Decoder#layer.tot_neurodes+FromOVL,
			U_ToI = [{FromId,FromOVL}|C#circuit.i],
			New_Neurodes = [#neurode{id=genotype:generate_UniqueId(),af=create_af(Decoder#layer.neurode_type),weights=add_weights(1,U_Encoder#layer.tot_neurodes,create_af(Decoder#layer.neurode_type),[])}||_<-lists:seq(1,FromOVL)],
			U_Decoder=Decoder#layer{
				neurodes=New_Neurodes++Decoder#layer.neurodes,
				tot_neurodes=U_TotNeurodes
			},
			C#circuit{
				i = U_ToI,
				layers=[U_Encoder,U_Decoder]
			};
		lrf_dae ->
			exit("link_ToCircuit:~p not imlemented yet.~n",[C#circuit.type]);
		cdae ->
			exit("link_ToCircuit:~p not imlemented yet.~n",[C#circuit.type]);
		dnn ->
			exit("link_ToCircuit:~p not imlemented yet.~n",[C#circuit.type]);
		so_lrf_dae ->
			exit("link_ToCircuit:~p not imlemented yet.~n",[C#circuit.type])
	end.

cutlink_ToCircuit(C,FromId)->
	ToI = C#circuit.i,
	%{TargetIndex,TargetVL}=lists:keyfind(FromId,1,ToI),
	%io:format("FromId:~p ToI~p~n",[FromId,ToI]),
	case C#circuit.type of
		standard ->
			{I_Index,IVL} = extract_WeightIndex(ToI,FromId,1),
			[NeurodeLayer|Substrate] = C#circuit.layers,
			%io:format("TargetIndex:~p~n TargetVL:~p~n NeurodeLayer#layer.neurodes:~p~n",[TargetIndex,TargetVL,NeurodeLayer#layer.neurodes]),
			U_Neurodes=[Neurode#neurode{weights=circuit:delete_weights(I_Index,IVL,Neurode#neurode.weights)}||Neurode<-NeurodeLayer#layer.neurodes],
			U_NeurodeLayer = NeurodeLayer#layer{neurodes=U_Neurodes},
			U_ToI = lists:keydelete(FromId,1,ToI),
			C#circuit{
				i=U_ToI,
				layers=[U_NeurodeLayer|Substrate]
			};
		dae ->
			exit("cutlink_ToCircuit:~p not imlemented yet.~n",[C#circuit.type]);
		lrf_dae ->
			exit("cutlink_ToCircuit:~p not imlemented yet.~n",[C#circuit.type]);
		cdae ->
			exit("cutlink_ToCircuit:~p not imlemented yet.~n",[C#circuit.type]);
		dnn ->
			exit("link_ToCircuit:~p not imlemented yet.~n",[C#circuit.type]);
		so_lrf_dae ->
			exit("link_ToCircuit:~p not imlemented yet.~n",[C#circuit.type])
	end.
	
	extract_WeightIndex([{TargetKey,TargetVL}|_],TargetKey,WeightIndexAcc)->
		{WeightIndexAcc,TargetVL};
	extract_WeightIndex([{_Key,VL}|ToI],TargetKey,WeightIndexAcc)->
		extract_WeightIndex(ToI,TargetKey,WeightIndexAcc+VL).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST CIRCUIT CREATION AND MUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_create_InitCircuit(Type)->
	IVL = 2,
	CircuitDynamics = static,
	case Type of
		micro ->
			CircuitLayersSpec = create_TestLayers(),
			create_Circuit([{input_id,IVL}],{micro,CircuitLayersSpec});
		micro_cplx ->
			CircuitLayersSpec = create_TestCplxLayers(),
			create_Circuit([{input_id,IVL}],{micro,CircuitLayersSpec});
		Type ->
			CircuitLayerSpec = #layer{neurode_type=tanh,tot_neurodes=3,type=Type},
			create_Circuit([{input_id,IVL}],CircuitLayerSpec)
	end.

	test_create_CPLX_InitCircuit()->
		IVL = 2,
		CircuitDynamics = static,
		CircuitLayersSpec =[
			#layer{neurode_type=cplx5,tot_neurodes=1,dynamics=dynamic}
		],
		create_InitCircuit([{input_id,IVL}],{CircuitDynamics,CircuitLayersSpec}).
		
	create_TestLayers()->
		[
			#layer{ivl=5,neurodes=[#neurode{id=a,weights=[1,2,3,4,5]},#neurode{id=b,weights=[2,3,4,5,6]},#neurode{id=c,weights=[3,4,5,6,7]}],tot_neurodes=3},
			#layer{ivl=3,neurodes=[#neurode{id=d,weights=[1,2,3]},#neurode{id=e,weights=[2,3,4]},#neurode{id=f,weights=[3,4,5]}],tot_neurodes=3},
			#layer{ivl=3,neurodes=[#neurode{id=g,weights=[2,2,2]},#neurode{id=h,weights=[3,3,3]}],tot_neurodes=2}
		].

	create_TestCplxLayers()->
		[
			#layer{neurode_type=cplx5,tot_neurodes=2,dynamics=dynamic},
			#layer{neurode_type=cplx5,tot_neurodes=1,dynamics=dynamic}
		].

test_DeleteWeights(TargetIndex,TargetVL)->%TODO:To delete weights use the delete weight function to test it
	Layers = create_TestLayers(),
	[NeurodeLayer|RemainderLayers] = Layers,
	U_NeurodeLayer=NeurodeLayer#layer{ivl=NeurodeLayer#layer.ivl-TargetVL,neurodes=[N#neurode{weights=delete_weights(1,TargetIndex,TargetVL,N#neurode.weights,[])}||N<-NeurodeLayer#layer.neurodes]},
	U_Layers=[U_NeurodeLayer|RemainderLayers],
	io:format("Layers:~n~p~n",[Layers]),
	io:format("U_Layers:~n~p~n",[U_Layers]).
			
test_AddWeights(TargetIndex,TargetVL)->
	Layers = create_TestLayers(),
	[NeurodeLayer|RemainderLayers] = Layers,
	U_NeurodeLayer=NeurodeLayer#layer{ivl=NeurodeLayer#layer.ivl+TargetVL,neurodes=[N#neurode{weights=add_weights(TargetIndex,TargetVL,N#neurode.af,N#neurode.weights)}||N<-NeurodeLayer#layer.neurodes]},
	U_Layers=[U_NeurodeLayer|RemainderLayers],
	io:format("Layers:~n~p~n",[Layers]),
	io:format("U_Layers:~n~p~n",[U_Layers]).
			
test_AddNeurode()->
	Layers = create_TestLayers(),
	%CircuitLayerSpecs = [{tanh,3,dynamic,5,0},{tanh,3,dynamic,3,0},{tanh,2,dynamic,3,0}],
	add_neurode(Layers,void).
	
test_DeleteNeurode(LayerIndex)->
	Layers = create_TestLayers(),
	U_Layers=delete_neurode(Layers,LayerIndex),
	io:format("Layers:~n~p~n",[Layers]),
	io:format("U_Layers:~n~p~n",[U_Layers]).

test_AddLayer()->
	Layers = create_TestLayers(),
	%CircuitLayerSpecs = [{tanh,3,dynamic,5,0},{tanh,3,dynamic,3,0},{tanh,2,dynamic,3,0}],
	U_Layers=add_layer(Layers,void),
	io:format("Old_Layers:~p~nNew_Layers:~p~n",[Layers,U_Layers]).
			
test_Mutation(Mutagen,DMult)->
	% [{add_layer,1},{perturb_weights,95},{add_bias,3},{remove_bias,3},{add_neurode,2}]
	Layers = create_TestLayers(),
	%CircuitLayerSpecs = [{tanh,3,dynamic,5,0},{tanh,3,dynamic,3,0},{tanh,2,dynamic,3,0}],
	Circuit = #circuit{layers=Layers,dynamics=dynamic},
	circuit:Mutagen(Layers,DMult).
			
test_perturb_circuit()->
	Layers = create_TestLayers(),
	%CircuitLayerSpecs = [{tanh,3,dynamic,5,0},{tanh,3,dynamic,3,0},{tanh,2,dynamic,3,0}],
	Circuit = #circuit{layers=Layers,dynamics=dynamic},
	perturb_circuit(Circuit,4).
			
test_std_output()->%TODO: not yet implemented
	Layers = create_TestLayers().
			
test_rbf_output()->%TODO: not yet implemented
	Layers = create_TestLayers().

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP SDAE IMPLEMENTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_create_Standard()->
	CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=3,dynamics=dynamic,type=standard},
	create_Standard([{void_id,5}],CircuitLayerSpec,static,0.5,{bp,1000}).

test_create_DNN()->
	CircuitLayersSpec=[
		#layer{neurode_type=tanh,tot_neurodes=3,dynamics=dynamic,type=dae},
		#layer{neurode_type=sin,tot_neurodes=4,dynamics=static,type=dae},
		#layer{neurode_type=sin,tot_neurodes=2,dynamics=static,type=standard}
		
	],
	create_DNN([{void_id,5}],CircuitLayersSpec,static,0.5,{bp,1000}).
	
test_create_SDAE()->
	CircuitLayersSpec=[
		#layer{neurode_type=tanh,tot_neurodes=3,dynamics=dynamic,type=dae},
		#layer{neurode_type=sin,tot_neurodes=4,dynamics=static,type=dae}
	],
	create_SDAE([{void_id,5}],CircuitLayersSpec,static,0.5,{bp,1000}).

test_create_DAE()->
	CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=3,dynamics=dynamic,type=dae},
	create_DAE([{void_id,5}],CircuitLayerSpec,static,0.5,{bp,1000}).

test_create_CAE()->
	CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=2,dynamics=dynamic,type=cae},
	create_CDAE([{void_id,10}],CircuitLayerSpec,static,0,{bp,1000}).
	
test_create_CDAE()->
	CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=2,dynamics=dynamic,type=cdae},
	create_CDAE([{void_id,10}],CircuitLayerSpec,static,0.5,{bp,1000}).

test_create_LRF_DAE()->
	CircuitLayerSpec=#layer{neurode_type=tanh,dynamics=dynamic,type=lrf_dae},
	create_LRF_DAE([{void_id,10}],CircuitLayerSpec,static,0,{bp,1000}).

test_create_Competitive()->
	CircuitLayerSpec=#layer{neurode_type=undefined,tot_neurodes=3,dynamics=dynamic,type=competitive},
	create_Competitive([{void_id,5}],CircuitLayerSpec,static,void,void).

create_DNN(I_IdPs,CircuitLayersSpec,CircuitDynamics,Noise,TrainingSetup)->
%Input must be one that defines a list of circuits to stack.
%In some sense, it makes no sense to have multi-layered circuits, each layer can be parallelized.
%In this manner, perhaps the idea of circuits of circuit is not good, and the layer should again become the basic element.
%The layer can be denoising, or standard, either backpropable, or not. Or perhaps remove the layer and just stick with the circuits.
%DNN must accept a list of layers, each layer must state what kind it is. We now must also test backprop.
	create_DNN(I_IdPs,CircuitLayersSpec,CircuitDynamics,Noise,TrainingSetup,[]).
create_DNN(I_IdPs,[CircuitLayerSpec|CircuitLayersSpec],CircuitDynamics,Noise,TrainingSetup,Acc)->
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
	%io:format("CircuitLayer:~p~n",[CircuitLayerSpec]),
	Circuit=case CircuitLayerSpec#layer.type of 
		dae -> 
			create_DAE([{undefined,IVL}],CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		cdae -> 
			create_CDAE([{undefined,IVL}],CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		lrf_dae -> 
			create_LRF_DAE([{undefined,IVL}],CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		competitive -> 
			create_Competitive([{undefined,IVL}],CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup);
		standard ->
			create_Standard([{undefined,IVL}],CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)
	end,
	OVL=Circuit#circuit.ovl,
	create_DNN([{undefined,OVL}],CircuitLayersSpec,CircuitDynamics,Noise,TrainingSetup,[Circuit|Acc]);
create_DNN(I_IdPs,[],CircuitDynamics,Noise,TrainingSetup,Acc)->
	Circuits=lists:reverse(Acc),
%	io:format("DAE_Circuits:~p~n",[Circuits]),
	[OutputCircuit|_] = Acc,
	#circuit{
		i=I_IdPs,
		ovl=OutputCircuit#circuit.ovl,
		dynamics=CircuitDynamics,
		layers=Circuits,
		noise=Noise,
		type = dnn,
		training=TrainingSetup
	}.
		
create_SDAE(I_IdPs,CircuitLayersSpec,CircuitDynamics,Noise,TrainingSetup)->
	create_SDAE(I_IdPs,CircuitLayersSpec,CircuitDynamics,Noise,TrainingSetup,[]).	
create_SDAE(I_IdPs,[CircuitLayerSpec|CircuitLayersSpec],CircuitDynamics,Noise,TrainingSetup,Acc)->
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
	DAE_Circuit=create_DAE([{undefined,IVL}],CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup),
	OVL=DAE_Circuit#circuit.ovl,
	create_SDAE([{undefined,OVL}],CircuitLayersSpec,CircuitDynamics,Noise,TrainingSetup,[DAE_Circuit|Acc]);
create_SDAE(I_IdPs,[],CircuitDynamics,Noise,TrainingSetup,Acc)->
	DAE_Circuits=lists:reverse(Acc),
%	io:format("DAE_Circuits:~p~n",[DAE_Circuits]),
	[OutputDAE|_] = Acc,
%	[OutputEncoder|_] = OutputDAE#circuit.layers,
	#circuit{
		i=I_IdPs,
		ovl=OutputDAE#circuit.ovl,
		dynamics=CircuitDynamics,
		layers=DAE_Circuits,
		noise=Noise,
		type = sdae,
		training=TrainingSetup
	}.
	
create_DAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)->
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
	DecoderLayer = #layer{neurode_type=linear,tot_neurodes=IVL,dynamics=static},
	InitLayers = create_InitLayers([CircuitLayerSpec,DecoderLayer],IVL,[]),
	%io:format("InitLayers created:~p~n",[InitLayers]),
	#circuit{
		i=I_IdPs,
		ivl=IVL,
		ovl=CircuitLayerSpec#layer.tot_neurodes,
		dynamics=CircuitDynamics,
		layers=InitLayers,
		noise=Noise,
		type = dae,
		training=TrainingSetup
	}.

create_Standard(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)->
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
%	DecoderLayer = #layer{neurode_type=tanh,tot_neurodes=IVL,dynamics=static},
	InitLayers = create_InitLayers([CircuitLayerSpec],IVL,[]),
	%io:format("InitLayers created:~p~n",[InitLayers]),
	#circuit{
		i=I_IdPs,
		ivl=IVL,
		ovl=CircuitLayerSpec#layer.tot_neurodes,
		dynamics=CircuitDynamics,
		layers=InitLayers,
		noise=Noise,
		type = standard,
		training=TrainingSetup
	}.

create_Competitive(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)->
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
	InitLayers = create_InitLayers([CircuitLayerSpec],IVL,[]),
	%io:format("InitLayers created:~p~n",[InitLayers]),
	#circuit{
		i=I_IdPs,
		ivl=IVL,
		ovl=CircuitLayerSpec#layer.tot_neurodes,
		dynamics=CircuitDynamics,
		layers=InitLayers,
		noise=Noise,
		type = standard,
		training=TrainingSetup
	}.
	
create_CDAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)->%
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
	ReceptiveField = round(math:sqrt(IVL)),
	OVL = (IVL-ReceptiveField+1)*CircuitLayerSpec#layer.tot_neurodes,
	DecoderLayer = #layer{neurode_type=tanh,tot_neurodes=ReceptiveField,dynamics=static},
	InitLayers = create_InitLayers([CircuitLayerSpec,DecoderLayer],ReceptiveField,[]),
	%io:format("InitLayers created:~p~n",[InitLayers]),
	#circuit{
		i=I_IdPs,
		ivl=IVL,
		ovl=OVL,
		dynamics=CircuitDynamics,
		layers=InitLayers,
		noise=Noise,
		type = cdae,
		receptive_field=ReceptiveField,
		step=1,
		training=TrainingSetup
	}.

create_LRF_DAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)->
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
	ReceptiveField = round(math:sqrt(IVL)),
	LL=round((IVL+1-ReceptiveField)/1),
	OVL = LL,
	EncoderLayer= CircuitLayerSpec#layer{tot_neurodes=LL},
	DecoderLayer = #layer{neurode_type=tanh,tot_neurodes=IVL,dynamics=static},
	InitLayers = create_InitLayers([EncoderLayer,DecoderLayer],ReceptiveField,[]),
	%io:format("InitLayers created:~p~n",[InitLayers]),
	#circuit{
		i=I_IdPs,
		ivl=IVL,
		ovl=OVL,
		dynamics=CircuitDynamics,
		layers=InitLayers,
		noise=Noise,
		type = lrf_dae,
		receptive_field=ReceptiveField,
		step=1,
		training=TrainingSetup
	}.

create_SO_LRF_DAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)->
	Circuit = create_LRF_DAE(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup),
	OVL = Circuit#circuit.ovl,
	Circuits=[Circuit,void],
	OutputCircuit = void,
	%TODO: THIS IS NOT YET DONE< FINISH IT ASAP
	#circuit{
		i=I_IdPs,
		ovl=OutputCircuit#circuit.ovl,
		dynamics=CircuitDynamics,
		layers=Circuits,
		noise=Noise,
		type = dnn,
		training=TrainingSetup
	}.

create_Sparse_DAE()->
	ok.

ovl(IVL,ReceptiveField)->
	(IVL-ReceptiveField+1).

create_PoolingCircuit(I_IdPs,CircuitLayerSpec,CircuitDynamics,Noise,TrainingSetup)->%TODO
	IVL = lists:sum([IVL||{_Input_Id,IVL}<-I_IdPs]),
	InitLayers = create_InitLayers([CircuitLayerSpec],IVL,[]),
	%io:format("InitLayers created:~p~n",[InitLayers]),
	#circuit{
		i=I_IdPs,
		ovl=CircuitLayerSpec#layer.tot_neurodes,
		dynamics=CircuitDynamics,
		layers=InitLayers,
		noise=Noise,
		type = pooler,
		training=TrainingSetup
	}.
	
	receptive_field(undefined,1,Receptive_Field,Coverage)->
		LL=(Coverage+1-Receptive_Field)/1,
		io:format("For: StepSize:~p Receptive_Field:~p Coverage:~p -> LayerLength:~p~n",[1,Receptive_Field,Coverage,LL]);
	receptive_field(LL,1,undefined,Coverage)->
		Receptive_Field=Coverage+1-LL*1,
		io:format("For: LayerLength:~p StepSize:~p Coverage:~p -> Receptive_Field:~p~n",[LL,1,Coverage,Receptive_Field]);
	receptive_field(LL,StepSize,Receptive_Field,undefined)->%Coverage = LL*StepSize + Receptive_Field-StepSize
		Coverage=LL*StepSize+Receptive_Field-StepSize,
		io:format("For: LayerLength:~p StepSize:~p Receptive_Field:~p ->Coverage:~p~n",[LL,1,Receptive_Field,Coverage]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP TRAINING IMPLEMENTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_DAE(_PCircuits,DAE,_PrevDAE,_Info,0,0)->
	DAE;
train_DAE(PCircuits,DAE,PrevDAE,Info,0,Index)->
	io:format("self():~w ErrAcc:~w~n",[self(),DAE#circuit.err_acc]),
	case DAE#circuit.err_acc < PrevDAE#circuit.err_acc of		
		true ->
			case is_integer(Index) of
				true ->
					train_DAE(PCircuits,DAE#circuit{err_acc=0},DAE,Info,Info#info.trn_end,Index-1);
				false ->
					train_DAE(PCircuits,DAE#circuit{err_acc=0},DAE,Info,Info#info.trn_end,Index)
			end;
		false ->
			case get(self()) == 1 of
				true ->
					erase(self()),
					PrevDAE;
				false ->
					put(self(),1),
					case is_integer(Index) of
						true ->
							train_DAE(PCircuits,DAE#circuit{err_acc=0},PrevDAE,Info,Info#info.trn_end,Index-1);
						false ->
							train_DAE(PCircuits,DAE#circuit{err_acc=0},PrevDAE,Info,Info#info.trn_end,Index)
					end
			end
	end;
train_DAE(PCircuits,DAE,PrevDAE,Info,SampleIndex,Index)->
%	io:format("Index:~p~n",[SampleIndex]),
	%io:format("PId:~p DAE_SampleIndex:~p~n",[self(),SampleIndex]),
%	[{Key,I,DO}] = ets:lookup(Info#info.name,random:uniform(Info#info.trn_end)),%lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
	KeyStart = random:uniform(Info#info.trn_end - 10),
	I = lists:flatten([Vec ||[{_,Vec,_}]<- [ets:lookup(Info#info.name,Key) || Key <- lists:seq(KeyStart,KeyStart+9)]]),
	%io:format("I:~p DO:~p~n",[I,DO]),
	U_I=calculate_output_dnn_short(I,PCircuits),
	%io:format("I:~p~nU_I:~p~n",[I,U_I]),
	U_DAE=backprop_DAE(U_I,DAE),
	?MODULE:train_DAE(PCircuits,U_DAE,PrevDAE,Info,SampleIndex-1,Index).

	backprop_DAE(IVector,C)->
		NoiseProb=C#circuit.noise,
		NoisyInput=case NoiseProb == 0 of
			true -> 
				IVector;
			false ->
				[case random:uniform()<NoiseProb of true -> 0.0; false -> Val end || Val<-IVector]
		end,
		%io:format("Layers:~p~n",[C#circuit.layers]),
		%io:format("NoisyInput:~p~n",[NoisyInput]),
		{Output,U_Layers1} = calculate_output_std(NoisyInput,C#circuit.layers,[]),
		%io:format("I:~p~nO:~p~n",[IVector,Output]),
		{OutputError_List,U_ErrAcc} = calculate_OutputError(Output,IVector,[],C#circuit.err_acc),
		{U_Layers2,_ErrorList} = layers_backprop(lists:reverse(U_Layers1),OutputError_List,[]),
%		io:format("Layers:~p~nU_Layers:~p~n",[Circuit#circuit.layers,U_Layers2]).
		C#circuit{layers=U_Layers2,err_acc=U_ErrAcc}.

train_CDAE(_PCircuits,CDAE,_PrevCDAE,_Info,0,0)->
	CDAE;
train_CDAE(PCircuits,CDAE,PrevCDAE,Info,0,Index)->
	io:format("ErrAcc:~w~n",[CDAE#circuit.err_acc]),
	case CDAE#circuit.err_acc < PrevCDAE#circuit.err_acc of		
		true ->
			case is_integer(Index) of
				true ->				
					train_CDAE(PCircuits,CDAE#circuit{err_acc=0},CDAE,Info,Info#info.trn_end,Index-1);
				false ->
					train_CDAE(PCircuits,CDAE#circuit{err_acc=0},CDAE,Info,Info#info.trn_end,Index)
			end;
		false ->
			case get(self()) == 1 of
				true ->
					erase(self()),
					PrevCDAE;
				false ->
					put(self(),1),
					case is_integer(Index) of
						true ->
							train_CDAE(PCircuits,CDAE#circuit{err_acc=0},PrevCDAE,Info,Info#info.trn_end,Index-1);
						false ->
							train_CDAE(PCircuits,CDAE#circuit{err_acc=0},PrevCDAE,Info,Info#info.trn_end,Index)
					end
			end
	end;
train_CDAE(PCircuits,CDAE,PrevCDAE,Info,SampleIndex,Index)->
	%{I,DO} = lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
	[{Key,I,DO}] = ets:lookup(Info#info.name,random:uniform(Info#info.trn_end)),
%	io:format("I:~p~n",[I]),
%	{Head,Tail} = lists:split(CDAE#circuit.receptive_field,I),
	U_I=calculate_output_dnn_short(I,PCircuits),
	U_CDAE=backprop_CDAE(U_I,CDAE),
	train_CDAE(PCircuits,U_CDAE,PrevCDAE,Info,SampleIndex-1,Index).
	
	backprop_CDAE(IVector,CDAE)->
		{Head,Tail} = lists:split(CDAE#circuit.receptive_field,IVector),
		backprop_CDAE(Head,Tail,CDAE).
	backprop_CDAE(Head,Tail,CDAE)->
%		io:format("Head:~p~n  Tail:~p~n",[Head,Tail]),
		%timer:sleep(1000),
		U_CDAE=backprop_DAE(Head,CDAE),
		case Tail of
			[E|U_Tail] ->
				[_|U_Head]=Head,
				backprop_CDAE(lists:reverse([E|lists:reverse(U_Head)]),U_Tail,U_CDAE);
			[] ->
				U_CDAE
		end.

train_LRF_DAE(_PCircuits,LRF_DAE,_PrevLRF_DAE,_Info,0,0)->
	LRF_DAE;
train_LRF_DAE(PCircuits,LRF_DAE,PrevLRF_DAE,Info,0,Index)->
	io:format("ErrAcc:~w~n",[LRF_DAE#circuit.err_acc]),
	case LRF_DAE#circuit.err_acc < PrevLRF_DAE#circuit.err_acc of		
		true ->
			case is_integer(Index) of
				true ->
					train_LRF_DAE(PCircuits,LRF_DAE#circuit{err_acc=0},LRF_DAE,Info,Info#info.trn_end,Index-1);
				false ->
					train_LRF_DAE(PCircuits,LRF_DAE#circuit{err_acc=0},LRF_DAE,Info,Info#info.trn_end,Index)
			end;
		false ->
			case get(self()) == 1 of
				true ->
					erase(self()),
					PrevLRF_DAE;
				false ->
					put(self(),1),
					case is_integer(Index) of
						true ->
							train_DAE(PCircuits,LRF_DAE#circuit{err_acc=0},PrevLRF_DAE,Info,Info#info.trn_end,Index-1);
						false ->
							train_DAE(PCircuits,LRF_DAE#circuit{err_acc=0},PrevLRF_DAE,Info,Info#info.trn_end,Index)
					end
			end
	end;
train_LRF_DAE(PCircuits,LRF_DAE,PrevLRF_DAE,Info,SampleIndex,Index)->
	%{I,DO} = lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
%	io:format("PId:~p LRF_DAE_SampleIndex:~p~n",[self(),SampleIndex]),
	[{Key,I,DO}] = ets:lookup(Info#info.name,random:uniform(Info#info.trn_end)),
	U_I=calculate_output_dnn_short(I,PCircuits),
	%io:format("I:~p~n",[I]),
%	{Head,Tail} = lists:split(LRF_DAE#circuit.receptive_field,U_I),
%	U_LRF_DAE=backprop_LRF_DAE(Head,Tail,LRF_DAE),
	U_LRF_DAE=backprop_LRF_DAE(U_I,LRF_DAE),
	train_LRF_DAE(PCircuits,U_LRF_DAE,PrevLRF_DAE,Info,SampleIndex-1,Index).
		
	backprop_LRF_DAE(Head,Tail,LRF_DAE)->
		%io:format("Head:~p~n  Tail:~p~n",[Head,Tail]),
		%timer:sleep(1000),
		U_LRF_DAE=backprop_LRF_DAE(Head,LRF_DAE),
		case Tail of
			[E|U_Tail] ->
				[_|U_Head]=Head,
				backprop_LRF_DAE(lists:reverse([E|lists:reverse(U_Head)]),U_Tail,U_LRF_DAE);
			[] ->
				U_LRF_DAE
		end.

		backprop_LRF_DAE(IVector,C)->
			NoiseProb=C#circuit.noise,
			NoisyInput=case NoiseProb == 0 of
				true ->
					IVector;
				false ->
					[case random:uniform()<NoiseProb of true -> 0.0; false -> Val end || Val<-IVector]
			end,
			{Output,U_Layers1} = calculate_output_LRF(NoisyInput,C#circuit.receptive_field,C#circuit.layers,[]),
	%		io:format("I:~p~nO:~p~n",[IVector,Output]),
			{OutputError_List,U_ErrAcc} = calculate_OutputError(Output,IVector,[],C#circuit.err_acc),
			U_Layers2 = layers_backprop_LRF(lists:reverse(U_Layers1),OutputError_List,[]),
	%		io:format("Layers:~p~nU_Layers:~p~n",[Circuit#circuit.layers,U_Layers2]).
			C#circuit{layers=U_Layers2,err_acc=U_ErrAcc}.

train_Standard(_PCircuits,C,_PrevC,_Info,0,0)->
	C;
train_Standard(PCircuits,C,PrevC,Info,0,Index)->
	io:format("ErrAcc:~w~n",[C#circuit.err_acc]),
	case C#circuit.err_acc < PrevC#circuit.err_acc of		
		true ->
			case is_integer(Index) of
				true ->
					train_Standard(PCircuits,C#circuit{err_acc=0},C,Info,Info#info.trn_end,Index-1);
				false ->
					train_Standard(PCircuits,C#circuit{err_acc=0},C,Info,Info#info.trn_end,Index)
			end;
		false ->
			case get(self()) == 1 of
				true ->
					erase(self()),
					PrevC;
				false ->
					put(self(),1),
					case is_integer(Index) of
						true ->
							train_DAE(PCircuits,C#circuit{err_acc=0},PrevC,Info,Info#info.trn_end,Index-1);
						false ->
							train_DAE(PCircuits,C#circuit{err_acc=0},PrevC,Info,Info#info.trn_end,Index)
					end
			end
	end;
train_Standard(PCircuits,C,PrevC,Info,SampleIndex,Index)->
	%io:format("SampleIndex:~p~n",[SampleIndex]),
	[{Key,I,DO}] = ets:lookup(Info#info.name,random:uniform(Info#info.trn_end)),%lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
	U_I=calculate_output_dnn_short(I,PCircuits),
	U_C=backprop_Standard(U_I,DO,C),
	train_Standard(PCircuits,U_C,PrevC,Info,SampleIndex-1,Index).

	backprop_Standard(IVector,DO,C)->
		{Output,U_Layers1} = calculate_output_std(IVector,C#circuit.layers,[]),
%		io:format("Output:~p~n",[Output]),
		{OutputError_List,U_ErrAcc} = calculate_OutputError(Output,DO,[],C#circuit.err_acc),
		{U_Layers2,_ErrorList} = layers_backprop(lists:reverse(U_Layers1),OutputError_List,[]),
		C#circuit{layers=U_Layers2,err_acc=U_ErrAcc}.

train_Competitive(PCircuits,C,PrevC,Info,0,0)->
	C;
train_Competitive(PCircuits,C,PrevC,Info,0,Index)->
	train_Competitive(PCircuits,C,PrevC,Info,Info#info.trn_end,Index-1);
train_Competitive(PCircuits,C,PrevC,Info,SampleIndex,Index)->
	[{Key,I,DO}] = ets:lookup(Info#info.name,random:uniform(Info#info.trn_end)),%lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
	U_I=calculate_output_dnn_short(I,PCircuits),
	[L] = C#circuit.layers,
	Neurodes = L#layer.neurodes,
	{TargetDistance,TargetIndex,TargetN}=closest(U_I,Neurodes),
	%io:format("TargetDistance:~p TargetIndex:~p TargetN:~p~n",[TargetDistance,TargetIndex,TargetN]),
	U_Neurodes=update_CompetitiveNeurodes(Neurodes,1,{TargetDistance,TargetIndex,U_I},[]),
	%U_N = N#neurode{weights=train_(N#weights,
	%Neurodes = L#layer.neurodes,
	%U_Neurodes = lists:keyreplace(N#neurode.id, 2, Neurodes, U_N)
	%io:format("Neurodes:~p~nU_Neurodes:~p~n~n",[Neurodes,U_Neurodes]),
	U_C = C#circuit{layers=[L#layer{neurodes=U_Neurodes}]},
	train_Competitive(PCircuits,U_C,PrevC,Info,SampleIndex-1,Index).
	
	update_CompetitiveNeurodes([N|Neurodes],TargetIndex,{Distance,TargetIndex,IVector},Acc)->
		U_Weights=competitive_update(N#neurode.weights,IVector,1,[]),
		U_N=N#neurode{weights=U_Weights},
		%io:format("N:~p~nU_N:~p~n~n",[N,U_N]),
		update_CompetitiveNeurodes(Neurodes,TargetIndex+1,{Distance,TargetIndex,IVector},[U_N|Acc]);
	update_CompetitiveNeurodes([N|Neurodes],Index,{Distance,TargetIndex,IVector},Acc)->
		%U_Weights=competitive_update(N#neurode.weights,IVector,-0.1,[]),
		%U_N=N#neurode{weights=U_Weights},
		update_CompetitiveNeurodes(Neurodes,Index,{Distance,TargetIndex,IVector},[N|Acc]);
	update_CompetitiveNeurodes([],_Index,{_Distance,_TargetIndex,_IVector},Acc)->
		lists:reverse(Acc).

		%competitive_update([{W,_PDW,_LP}|Weights],[Val|IVector],LP,Acc)->
		%	U_W = W+(Val-W)*LP,
		%	competitive_update(Weights,IVector,LP,[{U_W,PDW,LP}|Acc]);
		competitive_update([W|Weights],[Val|IVector],LP,Acc)->
			%U_W = W+(Val-W)*LP,
			U_W = W+LP*Val,
			competitive_update(Weights,IVector,LP,[U_W|Acc]);
		competitive_update([],[],_LP,Acc)->
			functions:normalize(lists:reverse(Acc)).
			
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP TESTING OF VARIOUS CIRCUITS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
check(Info,IVL,TL)->
	random:seed(now()),
	TotRuns = 1000,
	Passed=check(Info,IVL,TL,1000,0),
	io:format("SuccessPercentage:~p%~n",[(Passed/TotRuns)*100]).
check(Info,IVL,TL,0,Passed)->
	Passed;
check(Info,IVL,TL,Index,Passed)->
	RMS = case IVL == 2 of
		true ->
			test_backprop(TL);
		false ->
			bp_DAE(IVL*2,TL,Info,0.25)
	end,
	io:format("RMS:~p~n",[RMS]),
	case RMS < 0.01 of
		true ->
			check(Info,IVL,TL,Index-1,Passed+1);
		false ->
			check(Info,IVL,TL,Index-1,Passed)
	end.

test_bp_DAE(LayerLength,TrainingLength,TN)->
%	{ok,TN} = ets:file2tab(FileName),
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	NoiseLevel=0.25,%Set to 0 to make it AE
	bp_DAE(LayerLength,TrainingLength,Info,NoiseLevel).
%	ets:delete(FileName).
	
	bp_DAE(LayerLength,TrainingLength,Info,NoiseLevel)->
		%A special 2 layer (or 1 layer if using tied weights) NN
		%Function trains the NN to output its noised input for either X num of evaluations, or until Err no longer decreases.
		%Technically, it is something like test_backprop, which create a new circuit, and then trains it X amount of time.	
		%Create a circuit with 2 layers
		%The circuit specifies that it's an autoencoder, and thus uses a particular trianing, with the second layer being the size of IVL of the circuit, and the trianing is through backprip (or GA).
		IVL = Info#info.ivl,
		case undefined of% Info#info.ovl of
			undefined ->
				OVL = LayerLength;
			OVL -> 
				OVL
		end,
		{SM,SS,SMS}=now(),
		random:seed({SM,SS,SMS}),
		CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=OVL,dynamics=dynamic},
		DAE=create_DAE([{void_id,IVL}],CircuitLayerSpec,static,NoiseLevel,{bp,TrainingLength}),
		U_DAE = train_DAE([],DAE,DAE,Info,Info#info.trn_end,TrainingLength),
		{FM,FS,FMS}=now(),
		io:format("Parameters:~w Start/Finish:~w~n",[{IVL,OVL,TrainingLength},{FM-SM,FS-SS,FMS-SMS}]).
	
test_bp_CDAE(LayerLength,TrainingLength,FileName)->
	{ok,TN} = ets:file2tab(FileName),
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	NoiseLevel=0.25,%Set to 0 to make it AE
	bp_CDAE(LayerLength,TrainingLength,Info,NoiseLevel).
	
	bp_CDAE(LayerLength,TrainingLength,Info,NoiseLevel)->
		IVL=Info#info.ivl,
		CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=LayerLength,dynamics=dynamic},
		CDAE=create_CDAE([{void_id,IVL}],CircuitLayerSpec,static,NoiseLevel,{bp,TrainingLength}),
		U_CDAE = train_CDAE([],CDAE,CDAE,Info,Info#info.trn_end,TrainingLength).

test_bp_CAE(LayerLength,TrainingLength,FileName)->
	{ok,TN} = ets:file2tab(FileName),
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	NoiseLevel=0.25,%Set to 0 to make it AE
	bp_CAE(LayerLength,TrainingLength,Info,NoiseLevel).

	bp_CAE(LayerLength,TrainingLength,Info,NoiseLevel)->
		IVL=Info#info.ivl,
		CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=LayerLength,dynamics=dynamic},
		CAE=create_CDAE([{void_id,IVL}],CircuitLayerSpec,static,NoiseLevel,{bp,TrainingLength}),
		U_CAE = train_CDAE([],CAE,CAE,Info,Info#info.trn_end,TrainingLength).

test_bp_LRF_DAE(LayerLength,TrainingLength,FileName)->
	{ok,TN} = ets:file2tab(FileName),
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	NoiseLevel=0.25,%Set to 0 to make it AE
	bp_LRF_DAE(LayerLength,TrainingLength,Info,NoiseLevel).

	bp_LRF_DAE(_ReceptiveField,TrainingLength,Info,NoiseLevel)->
		IVL=Info#info.ivl,
		{SM,SS,SMS}=now(),
		CircuitLayerSpec=#layer{neurode_type=tanh,dynamics=dynamic},
		LRF_DAE=create_LRF_DAE([{void_id,IVL}],CircuitLayerSpec,static,NoiseLevel,{bp,TrainingLength}),
		U_LRF_DAE = train_LRF_DAE([],LRF_DAE,LRF_DAE,Info,Info#info.trn_end,TrainingLength),
		ReceptiveField=U_LRF_DAE#circuit.receptive_field,
		[Encoder,Decoder] = U_LRF_DAE#circuit.layers,
		LayerLengths=[Encoder#layer.tot_neurodes,Decoder#layer.tot_neurodes],
		{FM,FS,FMS}=now(),
		io:format("Parameters:~w Start/Finish:~w s/eval:~w~n",[{IVL,LayerLengths,TrainingLength,ReceptiveField},{FM-SM,FS-SS,FMS-SMS},(FS-SS)/1000]).

test_bp_SDAE(LayerLength,TrainingLength,FileName)->
	{ok,TN} = ets:file2tab(FileName),
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	NoiseLevel=0.25,%Set to 0 to make it AE
	bp_SDAE(LayerLength,TrainingLength,Info,NoiseLevel).

	bp_SDAE(LayerLengths,TrainingLength,Info,NoiseLevel)->%TODO
		IVL=Info#info.ivl,
		{SM,SS,SMS}=now(),
		DAE_List = [#layer{neurode_type=tanh,tot_neurodes=LL,dynamics=dynamic} || LL <- LayerLengths],
		SDAE=create_SDAE([{void_id,IVL}],DAE_List,static,0.5,{bp,TrainingLength}),
	%	io:format("SDAE:~p~n",[SDAE]),
		U_SDAE = train_SDAE(SDAE,Info,TrainingLength),
		{FM,FS,FMS}=now(),
		io:format("Parameters:~w Start/Finish:~w s/eval:~w~n",[{IVL,LayerLengths,TrainingLength},{FM-SM,FS-SS,FMS-SMS},(FS-SS)/1000]).

		train_SDAE(SDAE,Info,TrainingLength)->
			DAEs = SDAE#circuit.layers,
			U_DAEs=train_SDAE([],DAEs,Info,TrainingLength),
			SDAE#circuit{layers=U_DAEs}.
		train_SDAE(PDAEs,[DAE|DAEs],Info,TrainingLength)->
			U_DAE=train_DAE([],DAE,DAE,Info,Info#info.trn_end,TrainingLength),
	%		io:format("U_DAE:~p~n",[U_DAE]),
			train_SDAE(PDAEs++[U_DAE],DAEs,Info,TrainingLength);
		train_SDAE(PDAEs,[],_Info,_TrainingLength)->
			PDAEs.

test_bp_Standard(LayerLength,TrainingLength,TN)->
	%{ok,TN} = ets:file2tab(FileName),
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	NoiseLevel=0.25,%Set to 0 to make it AE
	bp_Standard(TrainingLength,Info).
	
	bp_Standard(TrainingLength,Info)->
		{SM,SS,SMS}=now(),
		random:seed({SM,SS,SMS}),
		IVL = Info#info.ivl,
		OVL = Info#info.ovl,
		CircuitLayersSpec =[
			%#layer{neurode_type=linear,tot_neurodes=50,dynamics=static,type=standard},
			#layer{neurode_type=tanh,tot_neurodes=10,dynamics=static,type=standard},
			#layer{neurode_type=tanh,tot_neurodes=OVL,dynamics=static,type=standard}
		],
		CircuitDynamics = static,
		C=create_InitCircuit([{input_id,IVL}],{CircuitDynamics,CircuitLayersSpec}),
	%	{U_Circuit,RMS}=backprop_train(0,Circuit,TrainingSet,length(TrainingSet),0,TrainingLength),
		U_C=train_Standard([],C,C,Info,Info#info.trn_end,TrainingLength),
		{FM,FS,FMS}=now(),
		io:format("Parameters:~w Start/Finish:~w~n",[{IVL,OVL,TrainingLength},{FM-SM,FS-SS,FMS-SMS}]),
		List=[ets:lookup(Info#info.name,Key)||Key <- lists:seq(1,Info#info.trn_end)],
		O_DO=[{calculate_output_std_short(I,U_C),DO}||[{_Key,I,DO}]<-List],
		TotErr=lists:sum([calculate_OutputError(O,DO,0)||{O,DO}<-O_DO]),
		ErrAvg=TotErr/(Info#info.trn_end),
		io:format("O_DO:~p~nErrAvg:~p~n",[O_DO,ErrAvg]).

test_Competitive(LayerLength,TrainingLength,TN)->
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	IVL = Info#info.ivl,
	case undefined of
		undefined ->
			OVL = LayerLength;
		OVL -> 
			OVL
	end,
	Index_End = case Info#info.tst_end of
		undefined ->
			Info#info.trn_end;
		Tst_End ->
			Tst_End
	end,
	random:seed(now()),
	CircuitLayerSpec=#layer{neurode_type=competitive,tot_neurodes=OVL,dynamics=dynamic,type=competitive},
	NoiseLevel=0,
	Competitive_Circuit=create_Competitive([{void_id,IVL}],CircuitLayerSpec,static,NoiseLevel,{competitive,TrainingLength}),
	io:format("New Competitive Circuit:~p~n Init parameters:~p~n",[Competitive_Circuit,{LayerLength,TrainingLength,TN,IVL,OVL}]),
	{SM,SS,SMS}=now(),
	U_Competitive_Circuit = train_Competitive([],Competitive_Circuit,void,Info,Info#info.trn_end,TrainingLength),
	{FM,FS,FMS}=now(),
	io:format("Parameters:~w Start/Finish:~w~n",[{IVL,OVL,TrainingLength},{FM-SM,FS-SS,FMS-SMS}]),
	io:format("LayerLength:~p TrainingLength:~p TN:~p~n",[{LayerLength,is_integer(LayerLength)},{TrainingLength,is_integer(TrainingLength)},{TN,is_atom(TN)}]),
	CCircuit = Competitive_Circuit#circuit{id=integer_to_list(LayerLength)++"_"++integer_to_list(TrainingLength)++"_"++atom_to_list(TN)}.
	
%	ETS_Vectors = [ets:lookup_element(TN,Index,2) || Index<-lists:seq(1,Index_End)],
%	CCircuit_Vectors = [calculate_output_competitive(Vector,CCircuit#circuit.layers,[]) || Vector <- ETS_Vectors],
%	{ok, File_Output} = file:open(CCircuit#circuit.id++"_CompetitiveCluster_Output"++".txt", write),
%	write_Matrix(File_Output,CCircuit_Vectors),
%	file:close.

%	ETS_Vectors = [ets:lookup_element(TN,I,2) || I<-lists:seq(1,Index_End)],
%	CCircuit_Vectors = [calculate_output_competitive(Vector,CCircuit#circuit.layers,[]) || Vector <- ETS_Vectors],
%	io:format("CCircuit_Vectors:~p~n",[CCircuit_Vectors]),
%	U_Competitive_Circuit#circuit.id.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP TESTING OF DEEP NEURAL NETWORK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
create_CircuitTable()->
	mnesia:create_table(circuit,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,circuit)}]).
	
ets2TrainingSet(_TN,StartIndex,StartIndex,Acc)->
	Acc;
ets2TrainingSet(TN,StartIndex,Index,Acc)->
	[{Index,I,O}] = ets:lookup(TN,Index),
	ets2TrainingSet(TN,StartIndex,Index-1,[{I,O}|Acc]).

benchmark_ListElementExtraction(TotSamples,IVL,OVL)->
	{SM,SS,SMS}=now(),
	TrainingSamples = [{[random:uniform()||_<-lists:seq(1,IVL)],[random:uniform()||_<-lists:seq(1,OVL)]} || _<- lists:seq(1,TotSamples)],
	Result = [lists:nth(random:uniform(TotSamples),TrainingSamples)||_<-lists:seq(1,TotSamples)],
	{FM,FS,FMS}=now(),
	io:format("Start/Finish:~w SecondsPerSample:~p~n",[{FM-SM,FS-SS,FMS-SMS},(FS-SS)/TotSamples]).
	
continue_DNN(Id,FileName,TrainingLength)->
	case genotype:dirty_read({circuit,Id}) of
		undefined ->
			io:format("No circuit with Id:~p exists.~n",[Id]);
		DNN ->
			{ok,TN} = ets:file2tab(FileName),
			[{0,Info}]=ets:lookup(TN,0),
			io:format("Info:~p~n",[Info]),
			{SM,SS,SMS}=now(),
			random:seed({SM,SS,SMS}),
			U_DNN = train_DNN(DNN,Info,TrainingLength),
			{FM,FS,FMS}=now(),
			io:format("TrainingLength:~s Parameters:~w Start/Finish:~w~n",[TrainingLength,{FM-SM,FS-SS,FMS-SMS}]),
			genotype:write(U_DNN),
			ErrAvg=test_phase(U_DNN,Info)
	end.

write_Bed(C_Id,TN,Description)->
	DNN = genotype:dirty_read({circuit,C_Id}),
	[{0,Info}]=ets:lookup(TN,0),
	{ok, File_Output} = file:open(atom_to_list(C_Id)++"_OutputBed.bed", write),
	List=[ets:lookup(Info#info.name,Key)||Key <- lists:seq(1,Info#info.trn_end)],
	[{1,_,ChrStart}]=ets:lookup(Info#info.name,1),
	[{_,_,ChrEnd}]=ets:lookup(Info#info.name,Info#info.trn_end),
	io:format(File_Output,"track name=bed description=\"Clone Paired Reads\" useScore=1~n",[]),
	write_Bed(File_Output,TN,Info#info.trn_end,DNN),
	file:close(File_Output).
	%{ok, File_Circuit} = file:open(atom_to_list(C_Id)++"_Circuit", write),
	%io:format(File_Circuit,"#Circuit Genotype:~n~p",[DNN]),
	%file:close(File_Circuit).

	write_Bed(FN,TN,Index_End,DNN)->
		write_Bed(FN,TN,1,Index_End+1,DNN).
	write_Bed(_FN,_TN,Index_End,Index_End,_DNN)->
		ok;
	write_Bed(FN,TN,Index,Index_End,DNN)->
		[{Index,I,ChrIndex}] = ets:lookup(TN,Index),
		Output = calculate_output_dnn_short(I,DNN#circuit.layers),
		write_Bed1(FN,Output,1,ChrIndex),
		write_Bed(FN,TN,Index+1,Index_End,DNN).

		write_Bed1(File_Output,[Val|List],Index,ChrIndex)->
			io:format(File_Output,"chr22 ~p ~p ~p~n",[ChrIndex*200,ChrIndex*200+199,(Val+1)*500]),
			write_Bed1(File_Output,List,Index+1,ChrIndex);
		write_Bed1(_File_Output,[],_Index,_ChrIndex)->
			ok.
		
write_BedGraph(C_Id,TN,Description)->
	DNN = genotype:dirty_read({circuit,C_Id}),
	[{0,Info}]=ets:lookup(TN,0),
	TotFiles = DNN#circuit.ovl,
	FNs = [FN ||{ok,FN}<- [file:open(atom_to_list(C_Id)++"_OutputBedGraph" ++ integer_to_list(Index) ++ ".bed", write) || Index <- lists:seq(0,TotFiles)]],
	List=[ets:lookup(Info#info.name,Key)||Key <- lists:seq(1,Info#info.trn_end)],
	[{1,_,ChrStart}]=ets:lookup(Info#info.name,1),
	[{_,_,ChrEnd}]=ets:lookup(Info#info.name,Info#info.trn_end),
	[{io:format(File_Output,"browser position chr22:~p-~p~n",[ChrStart*200,ChrEnd*200]),
	io:format(File_Output,"browser hide all~n",[]),
	io:format(File_Output,"browser pack refGene encodeRegions~n",[]),
	io:format(File_Output,"browser full altGraph~n",[]),
	io:format(File_Output,"track type=bedGraph name=\"~p\" description=\"BedGraph format\" visibility=full color=200,100,0 altColor=0,100,200 priority=20~n",[File_Output])} || File_Output<-FNs],
	write_BedGraph(FNs,TN,Info#info.trn_end,DNN),
	[file:close(FN)|| FN<-FNs].

	write_BedGraph(FileNames,TN,Index_End,DNN)->
		io:format("FIleNames:~p TN:~p Index_End:~p~n",[FileNames,TN,Index_End]),
		write_BedGraph(FileNames,TN,1,Index_End+1,DNN).
	write_BedGraph(_FileNames,_TN,Index_End,Index_End,_DNN)->
		ok;
	write_BedGraph([FNSP|FileNames],TN,Index,Index_End,DNN)->
		[{Index,I,ChrIndex}] = ets:lookup(TN,Index),
		Output = calculate_output_dnn_short(I,DNN#circuit.layers),
		write_BedGraph1(FNSP,Output,ChrIndex),
		write_BedGraph2(FileNames,Output,ChrIndex),
		write_BedGraph([FNSP|FileNames],TN,Index+1,Index_End,DNN).
		
		write_BedGraph1(File_Output,[Val|List],ChrIndex)->
			io:format(File_Output,"chr22 ~p ~p ~p~n",[ChrIndex*200,ChrIndex*200+199,Val]),
			write_BedGraph1(File_Output,List,ChrIndex);
		write_BedGraph1(_File_Output,[],_ChrIndex)->
			ok.
		
		write_BedGraph2([FN|FNs],[Val|List],ChrIndex)->
			io:format(FN,"chr22 ~p ~p ~p~n",[ChrIndex*200,ChrIndex*200+199,Val]),
			write_BedGraph2(FNs,List,ChrIndex);
		write_BedGraph2([],[],_ChrIndex)->
			ok.

get_OutputStatistics(C_Id,TN)->
	DNN = genotype:dirty_read({circuit,C_Id}),
	[{0,Info}]=ets:lookup(TN,0),
	TotFiles = DNN#circuit.ovl,
	[{1,_,ChrStart}]=ets:lookup(Info#info.name,1),
	[{_,_,ChrEnd}]=ets:lookup(Info#info.name,Info#info.trn_end),
	Index_End = case Info#info.tst_end of
			undefined ->
				Info#info.trn_end;
			Tst_End ->
				Tst_End
		end,
	[{1,I,ChrIndex}] = ets:lookup(TN,1),
	Output = calculate_output_dnn_short(I,DNN#circuit.layers),
	InitStats=[{Val,Val,Val}||Val<-Output],%{Min,Avg,Max}
	find_DataStatistics(TN,DNN,2,Index_End+1,InitStats).

	find_DataStatistics(TN,DNN,EndIndex,EndIndex,Statistics)->
		U_Statistics = [{Min,Avg_Acc/(EndIndex-1),Max}||{Min,Avg_Acc,Max}<-Statistics],
		%io:format("Statistics:~p~n",[U_Statistics]),
		U_Statistics;
	find_DataStatistics(TN,DNN,Index,EndIndex,Statistics)->
		[{Index,I,ChrIndex}] = ets:lookup(TN,Index),
		%io:format("I:~p~n",[I]),
		Output = calculate_output_dnn_short(I,DNN#circuit.layers),
		%io:format("I:~p~n",[{I,Output}]),
		U_Statistics = update_min_avg_max(Statistics,Output,[]),
		find_DataStatistics(TN,DNN,Index+1,EndIndex,U_Statistics).

		update_min_avg_max([{Min,Avg_Acc,Max}|List1],[Val|List2],Acc)->
			update_min_avg_max(List1,List2,[{min(Min,Val),Avg_Acc+Val,max(Max,Val)}|Acc]);
		update_min_avg_max([],[],Acc)->
			lists:reverse(Acc).

find_correlation(C_Id,TN)->
	DNN = genotype:dirty_read({circuit,C_Id}),
	[{0,Info}]=ets:lookup(TN,0),
	TotFiles = DNN#circuit.ovl,
	Index_End = case Info#info.tst_end of
		undefined ->
			Info#info.trn_end;
		Tst_End ->
			Tst_End
	end,
	ETS_Vectors = [ets:lookup_element(TN,Index,2) || Index<-lists:seq(1,Index_End)],
	DNN_Vectors = [calculate_output_dnn_short(Vector,DNN#circuit.layers) || Vector<-ETS_Vectors],
	ETS_AvgVector = get_VecAvg(ETS_Vectors),
	DNN_AvgVector = get_VecAvg(DNN_Vectors),
	ETS_Difs = get_VecDifs(ETS_Vectors,ETS_AvgVector),
	DNN_Difs = get_VecDifs(DNN_Vectors,DNN_AvgVector),
	Pearson_CorVec = get_Pearson(ETS_Difs,DNN_Difs),
	
	Ranked_ETS_Vectors= rank(ETS_Vectors),
	Ranked_DNN_Vectors= rank(DNN_Vectors),
	Ranked_ETS_AvgVector= get_VecAvg(Ranked_ETS_Vectors),
	Ranked_DNN_AvgVector= get_VecAvg(Ranked_DNN_Vectors),
	Ranked_ETS_Difs= get_VecDifs(Ranked_ETS_Vectors,Ranked_ETS_AvgVector),
	Ranked_DNN_Difs= get_VecDifs(Ranked_DNN_Vectors,Ranked_DNN_AvgVector),
	Spearman_CorVec = get_Pearson(Ranked_ETS_Difs,Ranked_DNN_Difs),
	io:format("Pearson_CorVec:~p~nSpearman_CorVec:~p~n",[Pearson_CorVec,Spearman_CorVec]),
	{ok, File_Output} = file:open(atom_to_list(C_Id)++"_Correlations"++".txt", write),
	io:format(File_Output,"Pearson Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Pearson_CorVec],
	io:format(File_Output,"Spearman Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Spearman_CorVec],
	file:close(File_Output).
	
get_ckeys()->
	io:format("--- Currently Stored Experiments ---~n"),
	get_ckeys(mnesia:dirty_first(circuit)).
	
	get_ckeys('$end_of_table')->
		ok;
	get_ckeys(Key)->
		%[E]=mnesia:dirty_read({experiment,Key}),
		io:format("*~p~n",[Key]),
		get_ckeys(mnesia:dirty_next(circuit,Key)).

find_Chrom_correlation(C_Id,TN,ChromHMM_TN)->
	DNN = genotype:dirty_read({circuit,C_Id}),
	[{0,Info}]=ets:lookup(TN,0),
	TotFiles = DNN#circuit.ovl,
	Index_End = case Info#info.tst_end of
		undefined ->
			Info#info.trn_end;
		Tst_End ->
			Tst_End
	end,
	ETS_and_ChromHMM_Vectors = syncup(TN,ChromHMM_TN,1,Index_End,1,[]),
	ETS_Vectors = [V||{V,_}<-ETS_and_ChromHMM_Vectors],
	ChromHMM_Vectors = [V||{_,V}<-ETS_and_ChromHMM_Vectors],
	%ETS_Vectors = [ets:lookup_element(TN,Index,2) || Index<-lists:seq(1,Index_End)],
	DNN_Vectors = [calculate_output_dnn_short(Vector,DNN#circuit.layers) || Vector<-ETS_Vectors],
	ChromHMM_AvgVector = get_VecAvg(ChromHMM_Vectors),
	DNN_AvgVector = get_VecAvg(DNN_Vectors),
	ETS_Difs = get_VecDifs(ChromHMM_Vectors,ChromHMM_AvgVector),
	DNN_Difs = get_VecDifs(DNN_Vectors,DNN_AvgVector),
	Pearson_CorVec = get_Pearson(ETS_Difs,DNN_Difs),
	
	Ranked_ETS_Vectors= rank(ChromHMM_Vectors),
	Ranked_DNN_Vectors= rank(DNN_Vectors),
	Ranked_ETS_AvgVector= get_VecAvg(Ranked_ETS_Vectors),
	Ranked_DNN_AvgVector= get_VecAvg(Ranked_DNN_Vectors),
	Ranked_ETS_Difs= get_VecDifs(Ranked_ETS_Vectors,Ranked_ETS_AvgVector),
	Ranked_DNN_Difs= get_VecDifs(Ranked_DNN_Vectors,Ranked_DNN_AvgVector),
	Spearman_CorVec = get_Pearson(Ranked_ETS_Difs,Ranked_DNN_Difs),
	io:format("Pearson_CorVec:~p~nSpearman_CorVec:~p~n",[Pearson_CorVec,Spearman_CorVec]),
	{ok, File_Output} = file:open(atom_to_list(C_Id)++"_ChromHMM_Correlations"++".txt", write),
	io:format(File_Output,"Pearson Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Pearson_CorVec],
	io:format(File_Output,"Spearman Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Spearman_CorVec],
	io:format(File_Output,"Top3 by ChromHMM TAG:~n",[]),
	TotTop = 1,
	Top_Pearsons = find_top(Pearson_CorVec,TotTop,data_extractor:chrom_HMMTags(),[]),
	Top_Spearmans = find_top(Spearman_CorVec,TotTop,data_extractor:chrom_HMMTags(),[]),
	[io:format(File_Output,"GMC Pearson top ~p: ~p~n",[TotTop,Top3]) || Top3 <-Top_Pearsons],
	[io:format(File_Output,"GMC Spearman top ~p: ~p~n",[TotTop,Top3])|| Top3 <- Top_Spearmans],
	Top_Unique_Pearsons = find_TopUnique(lists:flatten(Top_Pearsons),[]),
	Top_Unique_Spearmans = find_TopUnique(lists:flatten(Top_Spearmans),[]),
	[io:format(File_Output,"GMC Pearson top unique ~p: ~p~n",[TotTop,Top3]) || Top3 <-Top_Unique_Pearsons],
	[io:format(File_Output,"GMC Spearman top unique ~p: ~p~n",[TotTop,Top3])|| Top3 <- Top_Unique_Spearmans],
	io:format(File_Output,"Plot:~p~n",[bin_vectors(ETS_and_ChromHMM_Vectors,[],[])]),
	file:close(File_Output).
	%write_PearsonMatrix(C_Id,Pearson_CorVec),
	%write_SpearmanMatrix(C_Id,Spearman_CorVec).
	%write_All(ETS_and_ChromHMM_Vectors,DNN_Vectors,C_Id).

find_Competitive_Chrom_correlation(LL,TN,ChromHMM_TN)->%TODO
%	DNN = genotype:dirty_read({circuit,C_Id}),
	CCircuit = test_Competitive(LL,75,TN),%Creates the competitive network and trains it on the table with 10 full table runs.
	C_Id = CCircuit#circuit.id,
	[{0,Info}]=ets:lookup(TN,0),
	TotFiles = CCircuit#circuit.ovl,
	Index_End = case Info#info.tst_end of
		undefined ->
			Info#info.trn_end;
		Tst_End ->
			Tst_End
	end,
	ETS_and_ChromHMM_Vectors = syncup(TN,ChromHMM_TN,1,Index_End,1,[]),
	ETS_Vectors = [V||{V,_}<-ETS_and_ChromHMM_Vectors],
	ChromHMM_Vectors = [V||{_,V}<-ETS_and_ChromHMM_Vectors],
	%ETS_Vectors = [ets:lookup_element(TN,Index,2) || Index<-lists:seq(1,Index_End)],
%	DNN_Vectors = [calculate_output_dnn_short(Vector,DNN#circuit.layers) || Vector<-ETS_Vectors],
	DNN_Vectors = [calculate_output_competitive(Vector,CCircuit#circuit.layers,[]) || Vector <- ETS_Vectors],
	ChromHMM_AvgVector = get_VecAvg(ChromHMM_Vectors),
	DNN_AvgVector = get_VecAvg(DNN_Vectors),
	ETS_Difs = get_VecDifs(ChromHMM_Vectors,ChromHMM_AvgVector),
	DNN_Difs = get_VecDifs(DNN_Vectors,DNN_AvgVector),
	Pearson_CorVec = get_Pearson(ETS_Difs,DNN_Difs),
	
	Ranked_ETS_Vectors= rank(ChromHMM_Vectors),
	Ranked_DNN_Vectors= rank(DNN_Vectors),
	Ranked_ETS_AvgVector= get_VecAvg(Ranked_ETS_Vectors),
	Ranked_DNN_AvgVector= get_VecAvg(Ranked_DNN_Vectors),
	Ranked_ETS_Difs= get_VecDifs(Ranked_ETS_Vectors,Ranked_ETS_AvgVector),
	Ranked_DNN_Difs= get_VecDifs(Ranked_DNN_Vectors,Ranked_DNN_AvgVector),
	Spearman_CorVec = get_Pearson(Ranked_ETS_Difs,Ranked_DNN_Difs),
	io:format("Pearson_CorVec:~p~nSpearman_CorVec:~p~n",[Pearson_CorVec,Spearman_CorVec]),
	{ok, File_Output} = file:open(C_Id++"_ChromHMM_Correlations"++".txt", write),
	io:format(File_Output,"Pearson Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Pearson_CorVec],
	io:format(File_Output,"Spearman Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Spearman_CorVec],
	io:format(File_Output,"Top3 by ChromHMM TAG:~n",[]),
	TotTop = 1,
	Top_Pearsons = find_top(Pearson_CorVec,TotTop,data_extractor:chrom_HMMTags(),[]),
	Top_Spearmans = find_top(Spearman_CorVec,TotTop,data_extractor:chrom_HMMTags(),[]),
	[io:format(File_Output,"GMC Pearson top ~p: ~p~n",[TotTop,Top3]) || Top3 <-Top_Pearsons],
	[io:format(File_Output,"GMC Spearman top ~p: ~p~n",[TotTop,Top3])|| Top3 <- Top_Spearmans],
	Top_Unique_Pearsons = find_TopUnique(lists:flatten(Top_Pearsons),[]),
	Top_Unique_Spearmans = find_TopUnique(lists:flatten(Top_Spearmans),[]),
	[io:format(File_Output,"GMC Pearson top unique ~p: ~p~n",[TotTop,Top3]) || Top3 <-Top_Unique_Pearsons],
	[io:format(File_Output,"GMC Spearman top unique ~p: ~p~n",[TotTop,Top3])|| Top3 <- Top_Unique_Spearmans],
	io:format(File_Output,"Plot:~p~n",[bin_vectors(ETS_and_ChromHMM_Vectors,[],[])]),
	file:close(File_Output),
	write_PearsonMatrix(C_Id,Pearson_CorVec),
	write_SpearmanMatrix(C_Id,Spearman_CorVec).
	%write_All(ETS_and_ChromHMM_Vectors,DNN_Vectors,C_Id).

find_Competitive_correlation(LL,TN)->
	%DNN = genotype:dirty_read({circuit,C_Id}),
	CCircuit = test_Competitive(LL,2,TN),%Creates the competitive network and trains it on the table with 10 full table runs.
	C_Id = CCircuit#circuit.id,
	[{0,Info}]=ets:lookup(TN,0),
	TotFiles = CCircuit#circuit.ovl,
	Index_End = case Info#info.tst_end of
		undefined ->
			Info#info.trn_end;
		Tst_End ->
			Tst_End
	end,
	ETS_Vectors = [ets:lookup_element(TN,Index,2) || Index<-lists:seq(1,Index_End)],
	%DNN_Vectors = [calculate_output_dnn_short(Vector,DNN#circuit.layers) || Vector<-ETS_Vectors],
	DNN_Vectors = [calculate_output_competitive(Vector,CCircuit#circuit.layers,[]) || Vector <- ETS_Vectors],
	io:format("CCircuit_Vectors:~p~n",[DNN_Vectors]),
	ETS_AvgVector = get_VecAvg(ETS_Vectors),
	DNN_AvgVector = get_VecAvg(DNN_Vectors),
	ETS_Difs = get_VecDifs(ETS_Vectors,ETS_AvgVector),
	DNN_Difs = get_VecDifs(DNN_Vectors,DNN_AvgVector),
	Pearson_CorVec = get_Pearson(ETS_Difs,DNN_Difs),
	
	Ranked_ETS_Vectors= rank(ETS_Vectors),
	Ranked_DNN_Vectors= rank(DNN_Vectors),
	Ranked_ETS_AvgVector= get_VecAvg(Ranked_ETS_Vectors),
	Ranked_DNN_AvgVector= get_VecAvg(Ranked_DNN_Vectors),
	Ranked_ETS_Difs= get_VecDifs(Ranked_ETS_Vectors,Ranked_ETS_AvgVector),
	Ranked_DNN_Difs= get_VecDifs(Ranked_DNN_Vectors,Ranked_DNN_AvgVector),
	Spearman_CorVec = get_Pearson(Ranked_ETS_Difs,Ranked_DNN_Difs),
	io:format("Pearson_CorVec:~p~nSpearman_CorVec:~p~n",[Pearson_CorVec,Spearman_CorVec]),
	{ok, File_Output} = file:open(C_Id++"_Correlations"++".txt", write),
	io:format(File_Output,"Pearson Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Pearson_CorVec],
	io:format(File_Output,"Spearman Correlation:~n",[]),
	[io:format(File_Output,"GMC:~p~n",[Cor])||Cor<-Spearman_CorVec],
	file:close(File_Output).

	write_PearsonMatrix(C_Id,Pearson_CorVec)->
		case is_atom(C_Id) of 
			true ->
				{ok, File_Output} = file:open(atom_to_list(C_Id)++"_ChromHMM_PearsonMatrix"++".txt", write);
			false ->
				{ok, File_Output} = file:open(C_Id++"_ChromHMM_PearsonMatrix"++".txt", write)
		end,
		write_Matrix(File_Output,Pearson_CorVec),
		file:close(File_Output).
	
	write_SpearmanMatrix(C_Id,Spearman_CorVec)->
		case is_atom(C_Id) of
			true ->
				{ok, File_Output} = file:open(atom_to_list(C_Id)++"_ChromHMM_SpearmanMatrix"++".txt", write);
			false ->
				{ok, File_Output} = file:open(C_Id++"_ChromHMM_SpearmanMatrix"++".txt", write)
		end,
		write_Matrix(File_Output,Spearman_CorVec),
		file:close(File_Output).
	
		write_Matrix(File_Output,[Vector])->
			[io:format(File_Output,"~p ",[Val])||Val<-Vector];
		write_Matrix(File_Output,[Vector|VectorList])->
			[io:format(File_Output,"~p ",[Val])||Val<-Vector],
			io:format(File_Output,"~n",[]),
			write_Matrix(File_Output,VectorList).
			
	
	find_top([CVector|Correlations],TotTop,Tags,Acc)->
		Tagged_Correlations = lists:zip(CVector,Tags),
		Top_Tagged_Correlations = lists:sublist(lists:reverse(lists:sort(Tagged_Correlations)),TotTop),
		find_top(Correlations,TotTop,Tags,[Top_Tagged_Correlations|Acc]);
	find_top([],_TotTop,_Tags,Acc)->
		lists:reverse(Acc).
		
	find_TopUnique([{NewVal,Tag}|Tuples],Acc)->
		case lists:keyfind(Tag,2,Acc) of
			{OldVal,Tag} ->
				case NewVal > OldVal of
					true ->
						U_Acc = lists:keyreplace(Tag,2,Acc,{NewVal,Tag}),
						find_TopUnique(Tuples,U_Acc);
					false ->
						find_TopUnique(Tuples,Acc)
				end;
			false ->
				find_TopUnique(Tuples,[{NewVal,Tag}|Acc])
		end;
	find_TopUnique([],Acc)->
		Acc.
	
	bin_vectors([{ETS_Vector,ChromHMM_Vector}|ETS_and_ChromHMM_Vectors],ETS_Acc,ChromHMM_Acc)->
		bin_vectors(ETS_and_ChromHMM_Vectors,vector_add(ETS_Acc,ETS_Vector),vector_add(ChromHMM_Acc,ChromHMM_Vector));
	bin_vectors([],ETS_Acc,ChromHMM_Acc)->
		{ETS_Acc,lists:zip(ChromHMM_Acc,data_extractor:chrom_HMMTags())}.
	
	write_All([{Input_Vector,ChromHMMTag_Vector}|ETS_and_ChromHMM_Vectors],[DNN_Vector|DNN_Vectors],C_Id)->
		case is_atom(C_Id) of
			true ->
				{ok, File_Output} = file:open(atom_to_list(C_Id)++"_InputV_ChromHMMV_DNNV"++".txt", write);
			false ->
				{ok, File_Output} = file:open(C_Id++"_InputV_ChromHMMV_DNNV"++".txt", write)
		end,
		io:format(File_Output,"%%% Concattenated: InputVector of length:~p, ChromHMM Tag Vector:~p DNN OutputVector:~p~n",[length(Input_Vector),length(ChromHMMTag_Vector),length(DNN_Vector)]),
		write_All2([{Input_Vector,ChromHMMTag_Vector}|ETS_and_ChromHMM_Vectors],[DNN_Vector|DNN_Vectors],File_Output),
		file:close(File_Output).
		
		write_All2([{Input_Vector,ChromHMMTag_Vector}|ETS_and_ChromHMM_Vectors],[DNN_Vector|DNN_Vectors],File_Output)->
			Full_Vector = Input_Vector++ChromHMMTag_Vector++DNN_Vector,
			[io:format(File_Output," ~p",[Cor])||Cor<-Full_Vector],
			case DNN_Vectors of
				[] ->
					ok;
				_ ->
					io:format(File_Output,"~n",[])
			end,
			write_All2(ETS_and_ChromHMM_Vectors,DNN_Vectors,File_Output);
		write_All2([],[],_File_Output)->
			ok.

	syncup(_TN,_ChromHMM_TN,Index_End,Index_End,_CIndex,Acc)->
		lists:reverse(Acc);
	syncup(TN,ChromHMM_TN,Index,Index_End,CIndex,Acc)->
		case {ets:lookup(TN,Index),ets:lookup(ChromHMM_TN,CIndex)} of
			{[{Index,Vector,BPPos}],[{CIndex,CVector,CBPPos}]}->
				case BPPos*200 == CBPPos of
					true ->
						syncup(TN,ChromHMM_TN,Index+1,Index_End,CIndex+1,[{Vector,CVector}|Acc]);
					false ->
						case BPPos*200 > CBPPos of
							true ->
								syncup(TN,ChromHMM_TN,Index,Index_End,CIndex+1,Acc);
							false ->
								syncup(TN,ChromHMM_TN,Index+1,Index_End,CIndex,Acc)
						end
				end;
			{[],_}->
				lists:reverse(Acc);
			{_,[]}->
				lists:reverse(Acc)
		end.

	get_VecAvg(Vectors)->
		get_VecAvg(Vectors,0,[]).
	get_VecAvg([Vector|Vectors],Index,Acc)->
		get_VecAvg(Vectors,Index+1,vector_add(Acc,Vector));
	get_VecAvg([],Index,Acc)->
		[Val/Index||Val<-Acc].

	get_VecDifs(Vectors,AvgVector)->
		get_VecDifs(Vectors,[-Val||Val<-AvgVector],[]).
	get_VecDifs([Vector|Vectors],NegAvgVector,Acc)->
		get_VecDifs(Vectors,NegAvgVector,[vector_add(Vector,NegAvgVector)|Acc]);
	get_VecDifs([],_NegAvgVector,Acc)->
		lists:reverse(Acc).
	
	get_Pearson(ETS_Difs,DNN_Difs)->
		ETS_DifsSquared = [[Dif*Dif||Dif<-ETS_Dif]||ETS_Dif<-ETS_Difs],
		DNN_DifsSquared = [[Dif*Dif||Dif<-DNN_Dif]||DNN_Dif<-DNN_Difs],
		ETS_DifsSquaredSummedVec = sum_Rows(ETS_DifsSquared),
		DNN_DifsSquaredSummedVec = sum_Rows(DNN_DifsSquared),
		io:format("ETS_Difs:~p~n DNN_Difs:~p~n",[lists:sublist(ETS_Difs,10),lists:sublist(DNN_Difs,10)]),
		Numerator_Vecs = [doot(ETS_Difs,DNN_DifVec) || DNN_DifVec <- invert_matrix(DNN_Difs)],
		io:format("Numerator_Vecs:~p~n",[lists:sublist(Numerator_Vecs,10)]),
		[get_Pearson(ETS_DifsSquaredSummedVec,DNN_DifsSquaredSummed,Numerator_Vec,[]) || {DNN_DifsSquaredSummed,Numerator_Vec}<-lists:zip(DNN_DifsSquaredSummedVec,Numerator_Vecs)].
		
		get_Pearson([ETS_DifsSquaredSummed|ETS_DifsSquaredSummedVec],DNN_DifsSquaredSummed,[Numerator|Numerator_Vec],Acc)->
			io:format("ETS_DifsSquaredSummed*DNN_DifsSquaredSummed:~p~n",[{ETS_DifsSquaredSummed,DNN_DifsSquaredSummed}]),
			Result=case ((ETS_DifsSquaredSummed*DNN_DifsSquaredSummed == 0) or (ETS_DifsSquaredSummed*DNN_DifsSquaredSummed == 0.0)) and (Numerator == 0.0) of
				true ->
					0;
				false ->
					Numerator/math:sqrt((ETS_DifsSquaredSummed*DNN_DifsSquaredSummed))
			end,
			get_Pearson(ETS_DifsSquaredSummedVec,DNN_DifsSquaredSummed,Numerator_Vec,[Result|Acc]);
		get_Pearson([],_DNN_DifsSquaredSummed,[],Acc)->
			lists:reverse(Acc).
		
		sum_Rows([Vec|Rows])->
			sum_Rows(Rows,Vec).
		sum_Rows([Vec|Rows],Acc)->
			sum_Rows(Rows,vector_add(Vec,Acc));
		sum_Rows([],Acc)->
			Acc.
	
		doot([ETS_Vec|ETS_Vecs],[DNN_Val|DNN_Vec])->
			Acc=[ETS_Val*DNN_Val||ETS_Val<-ETS_Vec],
			doot(ETS_Vecs,DNN_Vec,Acc).
		doot([ETS_Vec|ETS_Vecs],[DNN_Val|DNN_Vec],Acc)->
			U_Acc=vector_add([ETS_Val*DNN_Val||ETS_Val<-ETS_Vec],Acc),
			doot(ETS_Vecs,DNN_Vec,U_Acc);
		doot([],[],Acc)->
			Acc.
		
	rank(Vectors)->
		%Split/invert matrix
		%rank eachvector
		%fuse into original matrix /invert matrix again
		Inverted_Matrix = invert_matrix(Vectors),
		Ranked_Inverted_Matrix = [rank1(Vector) || Vector<-Inverted_Matrix],
		invert_matrix(Ranked_Inverted_Matrix).
		
		invert_matrix(Vectors)->
			invert_matrix(Vectors,[],[],[]).
		invert_matrix([[Val|Vector]|Vectors],Acc1,Acc2,Acc3)->
			invert_matrix(Vectors,[Vector|Acc1],[Val|Acc2],Acc3);
		invert_matrix([],Acc1,Acc2,Acc3)->
			invert_matrix(lists:reverse(Acc1),[],[],[lists:reverse(Acc2)|Acc3]);
		invert_matrix([[]|_Vectors],[],[],Acc3)->
			lists:reverse(Acc3).
			
		rank1(Vector)->
			Sorted_Vector = lists:sort(index(Vector,1,[])),
			Sorted_Indexed_Vector = index2(Sorted_Vector),
			Ranked_Vectors = [Rank||{Index,Rank}<-lists:sort([{Index,Rank}||{{Val,Index},Rank} <-Sorted_Indexed_Vector])].
		
			index([Val|Vector],Index,Acc)->
				index(Vector,Index+1,[{Val,Index}|Acc]);
			index([],Index,Acc)->
				Acc.
				
			index2([{Val,Index}|Sorted_Vector])->
				index2(Sorted_Vector,[{Val,Index}],1,1,[]).
			index2([{Val,Index}|Sorted_Vector],[{LastVal,LastIndex}|Other],RankStart,RankCur,Acc)->
				case Val > LastVal of
					true ->
						Ranked_Tuples = [{{Val,Ind},(RankStart+RankCur)/2}||{Val,Ind}<-[{LastVal,LastIndex}|Other]],
						index2(Sorted_Vector,[{Val,Index}],RankCur+1,RankCur+1,lists:append(Ranked_Tuples,Acc));
					false ->
						index2(Sorted_Vector,[{Val,Index},{LastVal,LastIndex}|Other],RankStart,RankCur+1,Acc)
				end;
			index2([],Other,RankStart,RankCur,Acc)->
				Ranked_Tuples = [{{Val,Ind},(RankStart+RankCur)/2}||{Val,Ind}<-Other],
				lists:reverse(lists:append(Ranked_Tuples,Acc)).
	
great_excitations(C_Id,TN,OVL)->
	DNN = genotype:dirty_read({circuit,C_Id}),
	[{0,Info}]=ets:lookup(TN,0),
	{ok, File_Output} = file:open(atom_to_list(C_Id)++"_GCellBest", write),
%	OVL = Info#info.ivl,
	MinAvgMaxList=data_extractor:find_DataStatistics(TN,Info),
	GCells = [find_triggers(DNN,Index,Info#info.ivl,10,[],MinAvgMaxList) || Index <- lists:seq(1,OVL)],
	io:format("GCells:~p~n",[GCells]),
	[write_list(File_Output,Max_I)||{MaxInput,Max_I}<-GCells],
	file:close(File_Output).

	write_list(File_Output,TN,Index_End,DNN)->
		write_list(File_Output,TN,1,Index_End+1,DNN).
	write_list(_File_Output,_TN,Index_End,Index_End,_DNN)->
		ok;
	write_list(File_Output,TN,Index,Index_End,DNN)->
		[{Index,I,DO}] = ets:lookup(TN,Index),
		Output = calculate_output_dnn_short(I,DNN#circuit.layers),
		write_list(File_Output,Output),
		write_list(File_Output,TN,Index+1,Index_End,DNN).
		
		write_list(File_Output,[Val|List])->
			case List of
				[]->
					io:format(File_Output,"~p~n",[Val]);
				_ ->
					io:format(File_Output,"~p, ",[Val]),
					write_list(File_Output,List)
			end.

	find_triggers(_DNN,_Index,_IVL,0,Acc,_MinAvgMaxList)->
		lists:max(Acc);
	find_triggers(DNN,Index,IVL,Attempt,Acc,MinAvgMaxList)->
		Init_I = [random:uniform()/2|| _<-lists:seq(1,IVL)],
		Output = calculate_output_dnn_short(Init_I,DNN#circuit.layers),
		Init_Max = lists:nth(Index,Output),
%		MinAvgMax = lists:nth(Index,MinAvgMaxList),
		Result = find_trigger(DNN,Index,Init_I,Init_Max,{10000,10000},MinAvgMaxList),
		find_triggers(DNN,Index,IVL,Attempt-1,[Result|Acc],MinAvgMaxList).
	
		find_trigger(_DNN,_Index,Best_I,Best_Max,{0,_MaxAttempts},_MinMaxList)->
			io:format("Best_I:~p Best_Max:~p~n",[Best_I,Best_Max]),
			{Best_Max,Best_I};
		find_trigger(DNN,Index,Best_I,Best_Max,{AttemptIndex,MaxAttempts},MinAvgMaxList)->
			New_I = perturb_Val(Best_I,1/math:sqrt(length(Best_I)),MinAvgMaxList,[]),
			New_Max = lists:nth(Index,calculate_output_dnn_short(New_I,DNN#circuit.layers)),
			case New_Max > Best_Max of
				true ->
					io:format("NewBest:~p~n",[{New_I,New_Max}]),
					find_trigger(DNN,Index,New_I,New_Max,{MaxAttempts,MaxAttempts},MinAvgMaxList);
				false ->
					find_trigger(DNN,Index,Best_I,Best_Max,{AttemptIndex-1,MaxAttempts},MinAvgMaxList)
			end.

			perturb_Val([Val|Input],MP,[{Min,Avg,Max}|MinAvgMaxList],Acc)->
				U_Val=case random:uniform() < MP of
					true ->
						DMultiplier  = 1,
						DVal = (random:uniform()-0.5)*DMultiplier,
						functions:sat(Val + DVal,Max,Min);
					false ->
						Val
				end,
				perturb_Val(Input,MP,MinAvgMaxList,[U_Val|Acc]);
			perturb_Val([],_MP,[],Acc)->
				lists:reverse(Acc).

test_push_DNN_Layer()->
	%get dnn to modify-
	%check the number of neurodes in the encoder, that's the IVL to new circuit-
	%decide on new circuit type: DAE, CDAE, LRF_DAE, Standard
	%set up the new circuit with X number of neurodes and IVL for each neurode
	%add the new circuit to DNN
	Test_DNN = test_create_DNN(),
	io:format("Original DNN:~n~p~n",[Test_DNN]),
	Layers = Test_DNN#circuit.layers,
	[OutputCircuit|_] = lists:reverse(Layers),
	IVL=OutputCircuit#circuit.ovl,
	OVL = 3,
	CircuitLayerSpec=#layer{neurode_type=sigmoid,tot_neurodes=OVL,dynamics=dynamic,type=cae},
	DAE=create_DAE([{void_id,IVL}],CircuitLayerSpec,static,0.25,{bp,1000}),
	U_Layers = lists:reverse([DAE|lists:reverse(Layers)]),
	U_Test_DNN = Test_DNN#circuit{layers=U_Layers,ovl=OVL},
	io:format("New DNN:~n~p~n",[U_Test_DNN]),
	io:format("IVL:~p OVL:~p~n",[IVL,OVL]).
	
push_DNN_Layer(Id,OVL,New_Id)->
	DNN = genotype:dirty_read({circuit,Id}),
	Layers = DNN#circuit.layers,
	[OutputCircuit|_] = lists:reverse(Layers),
	IVL=OutputCircuit#circuit.ovl,
	CircuitLayerSpec=#layer{neurode_type=sigmoid,tot_neurodes=OVL,dynamics=dynamic,type=dae},
	DAE=create_DAE([{void_id,IVL}],CircuitLayerSpec,static,0.25,{bp,1000}),
	U_Layers = lists:reverse([DAE|lists:reverse(Layers)]),
	U_DNN = DNN#circuit{layers=U_Layers,ovl=OVL,id=New_Id},
	genotype:write(U_DNN).

test_pop_DNN_Layer()->
	Test_DNN = test_create_DNN(),
	io:format("Original DNN:~n~p~n",[Test_DNN]),
	Layers = Test_DNN#circuit.layers,
	[_CurrentOutputCircuit,NewOutputCircuit|RemainingLayers] = lists:reverse(Layers),
	IVL=NewOutputCircuit#circuit.ovl,
	OVL = 3,
	%CircuitLayerSpec=#layer{neurode_type=sigmoid,tot_neurodes=OVL,dynamics=dynamic,type=cae},
	%DAE=create_DAE([{void_id,IVL}],CircuitLayerSpec,static,0.25,{bp,1000}),
	U_Layers = lists:reverse([NewOutputCircuit|RemainingLayers]),
	U_Test_DNN = Test_DNN#circuit{layers=U_Layers,ovl=OVL},
	io:format("New DNN:~n~p~n",[U_Test_DNN]),
	io:format("IVL:~p OVL:~p~n",[IVL,OVL]).
	
pop_DNN_Layer(Id,New_Id)->
	DNN = genotype:dirty_read({circuit,Id}),
	Layers = DNN#circuit.layers,
	[CurrentOutputCircuit,NewOutputCircuit|RemainingLayers] = lists:reverse(Layers),
	New_OVL=NewOutputCircuit#circuit.ovl,
	U_Layers = lists:reverse([NewOutputCircuit|RemainingLayers]),
	U_DNN = DNN#circuit{layers=U_Layers,ovl=New_OVL,id=New_Id},
	genotype:write(U_DNN),
	io:format("Old OVL:~p New OVL:~p~n",[CurrentOutputCircuit#circuit.ovl,New_OVL]).

continue_training_DNN(Id,TN,TrainingLength)->
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	DNN = genotype:dirty_read({circuit,Id}),
	Circuits = DNN#circuit.layers,
	{PCircuits,RCircuits}=lists:split(length(Circuits)-1,Circuits),
	U_Circuits=train_DNN(PCircuits,RCircuits,Info,TrainingLength),
	U_DNN = DNN#circuit{layers=U_Circuits},
	genotype:write(U_DNN),
	ErrAvg=test_phase(U_DNN,Info).
	
bp_DNN(Id,TN,TrainingLength,LL_List,E_AF)->
	Ls = [#layer{neurode_type=E_AF,tot_neurodes=LL,dynamics=static,type=dae,backprop_tuning=on} || LL<-LL_List],
	bp_DNN(Id,TN,TrainingLength,Ls).
	
bp_DNN(Id,TN,TrainingLength,Ls)->
	[{0,Info}]=ets:lookup(TN,0),
	io:format("Info:~p~n",[Info]),
	IVL = Info#info.ivl*10,
	case Info#info.ovl of
		undefined -> OVL = 30;
		OVL -> OVL
	end,
	{SM,SS,SMS}=now(),
	random:seed(now()),
	%io:format("Here2~n"),
	case Ls of
		undefined ->
			Layers=[
				%#layer{neurode_type=sigmoid,tot_neurodes=500,dynamics=static,type=dae,backprop_tuning=on},
				%#layer{neurode_type=sigmoid,tot_neurodes=400,dynamics=static,type=dae,backprop_tuning=on},
				%#layer{neurode_type=sigmoid,tot_neurodes=100,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=50,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=50,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=50,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=50,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=50,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=OVL,dynamics=static,type=dae,backprop_tuning=on}
			];
		Layers->
			Layers
	end,
	%io:format("here3~n"),
	DNN=create_DNN([{void_id,IVL}],Layers,static,0.25,{bp,TrainingLength}),
	%io:format("DNN:~p~n",[DNN]),
	U_DNN = train_DNN(DNN,Info,TrainingLength),
	{FM,FS,FMS}=now(),
	io:format("Parameters:~w Start/Finish:~w~n",[{IVL,OVL,TrainingLength},{FM-SM,FS-SS,FMS-SMS}]),
	genotype:write(U_DNN#circuit{id=Id}),
	ErrAvg=test_phase(U_DNN,Info).

	test_DNN(C_Id,FileName)->
		%FileName = mnist,
		%Circuit_Id = mnist,
		{ok,TN} = ets:file2tab(FileName),
		[{0,Info}]=ets:lookup(TN,0),
		DNN=genotype:dirty_read({circuit,C_Id}),
		test_phase(DNN,Info).

	test_phase(DNN,Info)->
		[C|_]=lists:reverse(DNN#circuit.layers),
		case (Info#info.tst_end == undefined) and (C#circuit.type =/= standard) of
			true ->
				%List=[ets:lookup(Info#info.name,Key)||Key <- lists:seq(1,Info#info.trn_end)],
				%O_DO=[{calculate_output_dnn_short(I,DNN#circuit.layers),DO}||[{_Key,I,DO}] <-List],
				%io:format("O_DO:~p~n",[O_DO]);
				io:format("test_phase(DNN,Info) complete. tst_end == undefined and circuit.type =/= standard~n");
			false ->
				case (Info#info.tst_end == undefined) of
					true->
%						List=[ets:lookup(Info#info.name,Key)||Key <- lists:seq(1,Info#info.trn_end)],
%						O_DO=[{calculate_output_dnn_short(I,DNN#circuit.layers),DO}||[{_Key,I,DO}] <-List],
%						TotErr=lists:sum([calculate_OutputError(O,DO,0)||{O,DO}<-O_DO]),
						TotErr=get_TotErr(Info#info.name,DNN,1,Info#info.trn_end+1,0),
						ErrAvg=TotErr/Info#info.trn_end,
						%io:format("O_DO:~p~nErrAvg:~p~n",[O_DO,ErrAvg]),
						ErrAvg;
					false ->
%						List = [ets:lookup(Info#info.name,Key)||Key <- lists:seq(Info#info.val_end+1,Info#info.tst_end)],
%						O_DO=[{calculate_output_dnn_short(I,DNN#circuit.layers),DO}||[{_Key,I,DO}] <- List],
%						TotErr=lists:sum([calculate_OutputError(O,DO,0)||{O,DO}<-O_DO]),
						TotErr=get_TotErr(Info#info.name,DNN,Info#info.val_end+1,Info#info.tst_end+1,0),
						ErrAvg=TotErr/(Info#info.tst_end-Info#info.val_end),
						%io:format("O_DO:~p~nErrAvg:~p~n",[O_DO,ErrAvg]),
						ErrAvg
				end
		end.
		
		get_TotErr(_TN,_DNN,Index_End,Index_End,Acc)->
			Acc;
		get_TotErr(TN,DNN,Index,Index_End,Acc)->
			[{Index,I,DO}] = ets:lookup(TN,Index),
			O = calculate_output_dnn_short(I,DNN#circuit.layers),
			TotErr = calculate_OutputError(O,DO,0),
			get_TotErr(TN,DNN,Index+1,Index_End,Acc+TotErr).

	train_DNN(DNN,Info,TrainingLength)->
		Circuits = DNN#circuit.layers,
		U_Circuits=train_DNN([],Circuits,Info,TrainingLength),
		case off of %DNN#circuit.backprop_tuning of
			off ->
				DNN#circuit{layers=U_Circuits};
			on ->
				DNN#circuit{layers=tune_dnn(U_Circuits,Info,TrainingLength)}
		end.
		
		train_DNN(PCircuits,[C|Circuits],Info,TrainingLength)->%TODO: We use calculate_output_dnn_short, so if there is more than the last standard one, then the other ones are processed through short and thus without dot_product values.
			U_C=case C#circuit.type of
				lrf_dae ->
					train_LRF_DAE(PCircuits,C,C,Info,Info#info.trn_end,TrainingLength);
				dae ->
					train_DAE(PCircuits,C,C,Info,Info#info.trn_end,TrainingLength);
				standard when Circuits == []->%TODO: This part should train the entire thing.
					train_Standard(PCircuits,C,C,Info,Info#info.trn_end,TrainingLength);
				standard ->%TODO: Can not have backprop_tuning = on.
					C
			end,
	%		io:format("U_C:~p~n",[U_C]),
			?MODULE:train_DNN(PCircuits++[U_C],Circuits,Info,TrainingLength);
		train_DNN(PCircuits,[],_Info,_TrainingLength)->
			PCircuits.
	
	tune_dnn(Circuits,_Info,0)->
		Circuits;
	tune_dnn(Circuits,Info,TrainingLength)->
		U_Circuits=backprop_dnn(Circuits,Info,Info#info.trn_end),
		tune_dnn(U_Circuits,Info,TrainingLength-1).
		
		backprop_dnn(Circuits,_Info,0)->
			Circuits;
		backprop_dnn(Circuits,Info,SampleIndex)->
			[{_Key,I,DO}] = ets:lookup(Info#info.name,random:uniform(Info#info.trn_end)),
			{Output,U_Layers1} = calculate_output_dnn(I,Circuits,[]),
			%io:format("Output:~p~n",[Output]),
			{OutputError_List,ErrAcc} = calculate_OutputError(Output,DO,[],0),%TODO
			U_Circuits=backprop_through_circuits(lists:reverse(Circuits),OutputError_List,[]),
			backprop_dnn(U_Circuits,Info,SampleIndex-1).
	
			backprop_through_circuits([C|Circuits],OutputError_List,Acc)->
				{U_C,U_OutputError_List} = case C#circuit.type of
					dae ->
						tune_DAE(C,OutputError_List);
					lrf_dae->
						tune_LRF_DAE(C,OutputError_List);
					standard->
						{U_Layers,ErrList}=layers_backprop(lists:reverse(C#circuit.layers),OutputError_List,[]),
						{C#circuit{layers=U_Layers},ErrList}
				end,
				backprop_through_circuits(Circuits,U_OutputError_List,[U_C|Acc]);
			backprop_through_circuits([],_OutputError_List,Acc)->
				Acc.
				
				tune_DAE(C,ErrorList)->
					[Encoder,Decoder]=C#circuit.layers,
					{[U_Encoder],U_ErrorList} = layers_backprop([Encoder],ErrorList,[]),
				%	io:format("Layers:~p~nU_Layers:~p~n",[Circuit#circuit.layers,U_Layers2]).
					U_C = C#circuit{layers=[U_Encoder,Decoder]},
					{U_C,U_ErrorList}.
					
				tune_LRF_DAE(C,ErrorList)->
					[Encoder,Decoder]=C#circuit.layers,
					{U_ENeurodes,U_ErrorList} = layer_backprop_LRF(Encoder#layer.neurodes,ErrorList),
					U_Encoder = Encoder#layer{neurodes=U_ENeurodes},
					U_C = C#circuit{layers=[U_Encoder,Decoder]},
					{U_C,U_ErrorList}.

%Prep: 
%   Create a new table, DeepGeneExperiments (dg_experiment).
%{id,
%   active_circuits=[circuit_id,...],
%   trained_circuits=[circuit_id,...],
%   gmcs=[{circuit_id,gmc_index,pattern}...],
%   top_unique_correlations=[{circuit_id,gmc_index,spearman_cor,pearson_cor},...],
%   circuit_patterns=[layers,layers...]
%}
-record(dg_experiment,{
    active_circuits=[],
    training_circuits=[],
    gmcs=[],
    top_unique_correlations=[],
    circuit_patterns=[],
    population_size=10
    }).
    
-record(unique_cor,{
    circuit_id,
    gmc_index,
    spearman_cor,
    pearson_cor
    }).
    
-record(gmc,{
    circuit_id,
    gmc_index,
    pattern
    }).

create_DGExperiment_Table()->
    mnesia:create_table(dg_experiment,[{disc_copies, [node()]},{type,set},{attributes, record_info(fields,dg_experiment)}]).
    
new_DG_Experiment(Experiment_Id)->
    random:seed(now()),
    CircuitArchitectures=[
	    [#layer{neurode_type=E_AF,tot_neurodes=LL,dynamics=static,type=dae,backprop_tuning=on} || LL<-[50,50,50,50,50]],
	    [#layer{neurode_type=E_AF,tot_neurodes=LL,dynamics=static,type=dae,backprop_tuning=on} || LL<-[30,40,50,40,30]],
	    [#layer{neurode_type=E_AF,tot_neurodes=LL,dynamics=static,type=dae,backprop_tuning=on} || LL<-[100,50,100]],
	    [#layer{neurode_type=E_AF,tot_neurodes=LL,dynamics=static,type=dae,backprop_tuning=on} || LL<-[100,50,40,30,20]],
	    [#layer{neurode_type=E_AF,tot_neurodes=LL,dynamics=static,type=dae,backprop_tuning=on} || LL<-[300,150,75,50]],
	    [
				#layer{neurode_type=sigmoid,tot_neurodes=60,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=60,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=60,dynamics=static,type=dae,backprop_tuning=on},xxx
				#layer{neurode_type=sigmoid,tot_neurodes=60,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=60,dynamics=static,type=dae,backprop_tuning=on},
				#layer{neurode_type=sigmoid,tot_neurodes=60,dynamics=static,type=dae,backprop_tuning=on}
	    ]
	],
	TotCircuits = length(CircuitArchitectures),
	Init_Circuits = [(create_DNN([{void_id,IVL}],Layers,static,0.25,{bp,TrainingLength}))#circuit{id=genotype:generate_UniqueId()} || Layers <- CircuitArchitectures]
	[genotype:dirty_write(Init_Circuit) || Init_Circuit<-Init_Circuits],
	Init_ActiveCircuitIds=[C#circuit.id|| C<-Init_Circuits],
	DG_Experiment=#dg_experiment{
	    id = Experiment_Id,
	    active_circuits = Init_ActiveCircuitIds,
	    trained_circuits = [],
	    gmcs=[],
	    top_unique_correlations=[],
	    circuit_patterns=[],
	    population_size=10,
	},
	genotype:dirty_write(DG_Experiment),
	continue_DG_Experiment(Experiment_Id).

continue_DG_Experiment(Experiment_Id,0)->
    print_DG_Experiment_Report(Experiment_Id);
continue_DG_Experiment(Experiment_Id,PhaseIndex)->
    %1. Experiment contains the Ids of the circuits, their stages, and their results.
    %2. spawn TotCircuits number of circuits, of some default, or qually spaced out based sizes/depths
    %3. Train the systems
    %4. Run the correlation finder (and gmc finder?)
    %5. Compare the current new found correlations to the already existing ones, replace with higher correlations, add new ones.
    %6. Compare new GMCs to the old ones, and add them to the list. (The ID of the circuit, and the Index of the GMC)
    %7. clone the circuits, pop top 1-2 layers, add new ones, and go to step 2.
    
    DG_Experiment = genotype:dirty_read({dg_experiment,Experiment_Id}),
    DNN_IdPIdMap = [{DNN_Id,spawn(circuit,train_DNN,[genotype:dirty_read({circuit,DNN_Id}),Info,undefined,self()])} || DNN_Id<-DX_Experiment#dg_experiment.active_circuits],
    gather_DNN_Completion_Acks(DNN_IdPIdMap),
    U1_DG_Experiment=update_Correlations(DG_Experiment),
    U2_DG_Experiment=update_GMCs(U1_DG_Experiment),
    New_Circuit_Ids=generate_NewCircuits(U2_DG_Experiment#experiment.active_circuits,U2_DG_Experiment#experiment.population_size),
    genotype:dirty_write(U2_DG_Experiment#experiment{active_circuits=New_Circuit_Ids,trained_circuits = lists:append(DG_Experiment#experiment.active_ids,U2_DG_Experiment#experiment.trained_circuits)}),
    continue_DG_Experiment(Experiment_Id,PhaseIndex-1).
    
    update_Correlations(DG_Experiment)->
        %1. Calculate spearmen and pearson correlation
        %2. Compare the best unique ones to the existing ones in the top_unique_correlations list of tuples (f(O^2)?)
        %3. Add new ones, ore replace if better.
        %4. Update DG_Experiment with the new top_unique_correlations, and return the updated DG_Experiment
        ok.%TODO
        
    update_GMCs(DG_Experiment)->
        %1. Find GMC activation patterns
        %2. Perform eucleadian distance comparison between new active_circuits gmcs and existing circuit_patterns, add new ones if they differ by significant enough margin.
        %3. Return updated DG_Experiment
        ok.%TODO
        
    generate_NewCircuits(Circuits,Pop_Size)->
        %1. Randomly choose a circuit from Circuits
        %2. Pop randomly between 1 and 3 layers from the top of the circuit
        %3. Push to the circuit a random set of layers, between 1 and 3, of random type.
        %4. Decrement Pop_Size, and recur.
        ok.%TODO
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP ALGORITHM IMPLEMENTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_backprop(TrainingLength)->
	random:seed(now()),
	Circuit = test_create_InitCircuit(micro),
	TrainingSet = [{[-1,-1],[-1]},{[-1,1],[1]},{[1,-1],[1]},{[1,1],[-1]}],
	%Circuit = test_CPLX_Circuit(),
	%TrainingSet=[{conditional_preprocessing(I),O} || {I,O}<- [{[0,0],[0]},{[0,1],[1]},{[1,0],[1]},{[1,1],[0]}]],
	{U_Circuit,RMS}=backprop_train(0,Circuit,TrainingSet,length(TrainingSet),0,TrainingLength),
%	{U_Circuit,RMS}=optimize(0,Circuit,TrainingSet,length(TrainingSet),0,TrainingLength),

	Outputs=[calculate_output_std_short(I,U_Circuit)||{I,DO}<-TrainingSet],
%	[io:format("I:~p DO:~p O:~p~n",[I,DO,O])||{{I,DO},{O,_}}<-lists:zip(TrainingSet,Outputs)],
	RMS/length(TrainingSet).

	backprop_train(_NoiseProb,Circuit,_TrainingSet,0,ErrAcc,1)->
		{Circuit,ErrAcc};
	backprop_train(NoiseProb,Circuit,TrainingSet,0,ErrAcc,Index)->
		io:format("self():~p ErrAcc:~w~n",[self(),ErrAcc]),
		backprop_train(NoiseProb,Circuit,TrainingSet,length(TrainingSet),0,Index-1);
	backprop_train(NoiseProb,Circuit,TrainingSet,SampleIndex,ErrAcc,Index)->
		{I,DO} = lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
		U_I =case NoiseProb == 0 of
			true ->
				I;
			false ->
				[case random:uniform()<NoiseProb of true -> 0; false -> Val end || Val<-I]
		end,
		{Output,U_Layers1} = calculate_output_std(U_I,Circuit#circuit.layers,[]),
%		io:format("Output:~p~n",[Output]),
		{OutputError_List,U_ErrAcc} = calculate_OutputError(Output,DO,[],ErrAcc),
		{U_Layers2,_ErrorList} = layers_backprop(lists:reverse(U_Layers1),OutputError_List,[]),
%		io:format("Layers:~p~nU_Layers:~p~n",[Circuit#circuit.layers,U_Layers2]),
		backprop_train(NoiseProb,Circuit#circuit{layers=U_Layers2},TrainingSet,SampleIndex-1,U_ErrAcc,Index).
		
%circuit_backprop(Circuit,Input,Output,ExpectedOutput)->%Beta = Delta
	layers_backprop([L|Layers],ErrorList,Acc)->
%		io:format("L~p~n",[L]),
		{U_Neurodes,New_ErrorList} = layer_backprop(L#layer.input,L#layer.neurodes,ErrorList),
		layers_backprop(Layers,New_ErrorList,[L#layer{neurodes=U_Neurodes}|Acc]);
	layers_backprop([],ErrorList,Acc)->
		{Acc,ErrorList}.
	
	layers_backprop_LRF([Decoder,Encoder],ErrorList,Acc)->
%		io:format("L~p~n",[L]),
		{U_DNeurodes,DErrorList} = layer_backprop(Decoder#layer.input,Decoder#layer.neurodes,ErrorList),
		U_Decoder = Decoder#layer{neurodes=U_DNeurodes},
		{U_ENeurodes,EErrorList} = layer_backprop_LRF(Encoder#layer.neurodes,DErrorList),
		U_Encoder = Encoder#layer{neurodes=U_ENeurodes},
		[U_Encoder,U_Decoder].
%		layers_backprop_LRF(Layers,New_ErrorList,[L#layer{neurodes=U_Neurodes}|Acc]);
%	layers_backprop_LRF([],_ErrorList,Acc)->
%		Acc.
	
	calculate_OutputError([O|Output],[DO|DesiredOutput],RMSAcc)->
		calculate_OutputError(Output,DesiredOutput,math:pow(DO-O,2)+RMSAcc);
	calculate_OutputError([],[],RMSAcc)->
		%io:format("RMSAcc:~p~n",[RMSAcc]),
		RMSAcc.

	calculate_OutputError([O|Output],[DO|DesiredOutput],Acc,RMSAcc)->
		%io:format("DO:~p O:~p DO-O:~p math:pow(DO-O,2):~p~n",[DO,O,DO-O,math:pow(DO-O,2)]),
		calculate_OutputError(Output,DesiredOutput,[DO-O|Acc],math:pow(DO-O,2)+RMSAcc);
	calculate_OutputError([],[],Acc,RMSAcc)->
		%io:format("RMSAcc:~p~n",[RMSAcc]),
		{lists:reverse(Acc),RMSAcc}.
	
	layer_backprop(Input,Layer,ErrorList)->
		layer_backprop(Input,Layer,ErrorList,[],[]).
	layer_backprop(Input,[Neurode|Layer],[Error|ErrorList],NeurodeAcc,ErrorAcc)->
		{U_Neurode,Err} = neurode_backprop(Neurode,Input,Error),
		layer_backprop(Input,Layer,ErrorList,[U_Neurode|NeurodeAcc],vector_add(Err,ErrorAcc));
	layer_backprop(_Input,[],[],NeurodeAcc,ErrorAcc)->
		{lists:reverse(NeurodeAcc),ErrorAcc}.

	layer_backprop_LRF(Layer,ErrorList)->
		layer_backprop_LRF(Layer,ErrorList,[],[]).
	layer_backprop_LRF([Neurode|Layer],[Error|ErrorList],NeurodeAcc,ErrorAcc)->
		{U_Neurode,Err} = neurode_backprop(Neurode,Neurode#neurode.i,Error),
		layer_backprop_LRF(Layer,ErrorList,[U_Neurode|NeurodeAcc],vector_add(Err,ErrorAcc));
	layer_backprop_LRF([],[],NeurodeAcc,ErrorAcc)->
		{lists:reverse(NeurodeAcc),ErrorAcc}.

		vector_add(A,[])->
			A;
		vector_add([],B)->
			B;
		vector_add(A,B)->
			vector_add(A,B,[]).
		vector_add([A|AL],[B|BL],Acc)->
			vector_add(AL,BL,[A+B|Acc]);
		vector_add([],[],Acc)->
			lists:reverse(Acc).
		%ErrAcc+lists:sum(Error)...

	neurode_backprop(N,Input,Error)->
%		io:format("N:~p Input:~p Error:~p~n",[N,Input,Error]),
		AF = N#neurode.af,
		Delta = delta(Error,N#neurode.dot_product,AF),
%		io:format("Delta:~p~nInput:~p~nWeights:~p~nBias:~p~n",[Delta,Input,N#neurode.weights,N#neurode.bias]),
		{U_WP,U_BiasP,U_Err} = backpropogate(Delta,Input,N#neurode.weights,N#neurode.bias,[],[]),
		{N#neurode{weights=U_WP,bias=U_BiasP},U_Err}.
		
backpropogate(Delta,Input,DWP,Bias)->backpropogate(Delta,Input,DWP,Bias,[],[]).
backpropogate(Delta,[I|Input],[WP|DWP],Bias,DWPAcc,ErrAcc)->
	%io:format("Delta:~p I:~p WP:~p Bias:~p DWPAcc:~p ErrAcc:~p~n",[Delta,I,WP,Bias,DWPAcc,ErrAcc]),
	{U_WP,Err} = update_WP(Delta,I,WP),
%	I_Pid ! {self(),backprop,Error},
	backpropogate(Delta,Input,DWP,Bias,[U_WP|DWPAcc],[Err|ErrAcc]);
backpropogate(Delta,[],[],undefined,DWPAcc,ErrAcc)->
	{lists:reverse(DWPAcc),undefined,lists:reverse(ErrAcc)};
backpropogate(Delta,[],[],BiasP,DWPAcc,ErrAcc)->
	{U_BiasP,_Err} = update_WP(Delta,1,BiasP),
	{lists:reverse(DWPAcc),U_BiasP,lists:reverse(ErrAcc)}.
	
	update_WP({RDelta,IDelta},I,{{RW,IW},{RPDW,IPDW},{RLP,ILP}}) when ?f(RDelta), ?f(IDelta), ?f(I), ?f(RW), ?f(IW), ?f(RPDW), ?f(IPDW), ?f(RLP), ?f(ILP) ->
		%io:format("RDelta,IDelta},I,{{RW,IW},{RPDW,IPDW},{RLP,ILP}}:~p~n",[{{RDelta,IDelta},I,{{RW,IW},{RPDW,IPDW},{RLP,ILP}}}]),
		case I of
			{RI,II} -> ok;
			1 -> RI = -1, II = 0
		end,
		RNDW = RLP*RDelta*RI,
		INDW = -ILP*IDelta*II,
	%	U_RLP = RLP*0.999,
	%	U_ILP = ILP*0.999,
		U_RW = RW+RNDW+0.5*RPDW-RW*abs(RW)/1000000,
		U_IW = IW+INDW+0.5*IPDW-IW*abs(IW)/1000000,
		{{{U_RW,U_IW},{RNDW,INDW},{RLP,ILP}},0};%{RDelta*RW,IDelta*IW}};
	update_WP(Delta,I,{W,PDW,LP}) when ?f(Delta), ?f(I), ?f(W), ?f(PDW), ?f(LP) -> %%%WUpdateRule: W + LP*NDW + PDW, PDW = 0.9*NLP*NWD, NDW = LP*NWD, NWD = Beta*I
%		NWD = Delta*I, %%%New Weight Direction NWD
%		NLP = calculate_NLP(PDW,NWD,LP),
%		NDW = LP*NWD,
%		UpdatedW = W+NDW+PDW,
%		WDecay = -(UpdatedW*abs(UpdatedW))/1000000, %%%Possible to add decay towards 0 to W.
%		%io:format("~p~n",[{Delta,W,NDW,NLP,I}]),
%		{{W+NDW+PDW+WDecay,0.9*NLP*NWD,NLP},Delta*W}.
		%WDecay = -(UpdatedW*abs(UpdatedW))/1000000,
		%NWD = Delta*I,
		%io:format("W:~p~n",[{W,PDW,LP}]),
		NDW = LP*Delta*I, %%%New Weight Direction NWD
		NLP = LP*0.999999,%calculate_NLP(PDW,NWD,LP),%*100*(math:pow(PDW-NDW,2)),
		%io:format("NLP:~p~n",[NLP]),
		UpdatedW = functions:saturation(W+NDW+0.5*PDW-W*abs(W)/1000000,100),
		{{UpdatedW,NDW,NLP},Delta*W};
	update_WP(Delta,I,{W,PDW,LP}) ->
		NDW = LP*Delta*I,
		NLP = LP*0.999999,
		UpdatedW = functions:saturation(W+NDW+0.5*PDW-W*abs(W)/1000000,100),
		{{UpdatedW,NDW,NLP},Delta*W};
	update_WP(Delta,I,W) when ?f(Delta), ?f(I), ?f(W)->
%		io:format("Delta,I,W:~p~n",[{Delta,I,W}]),
		LP=0.1,
		NDW = LP*Delta*I,
		UpdatedW = W+NDW+W*abs(W)/1000000,%Err = Delta*W, NDW = LP*Delta*I, Delta = TotErr*derivative(AF,DotProduct)
		{{UpdatedW,NDW,LP},Delta*W}.
		
		calculate_NLP(PDW,NDW,LP) when ?f(PDW), ?f(NDW), ?f(LP)->
			case PDW*NDW >= 0 of
				true->
					LP + (1-LP)*0.01;
				false-> 
					LP * 0.01
			end.

delta(TotError,{DPR,DPI},AF) when ?f(TotError), ?f(DPR), ?f(DPI)->
	{DerR,DerI}=derivatives:AF({DPR,DPI}),
	{TotError*DerR,TotError*DerI};
delta(TotError,DotProduct,AF) when ?f(TotError), ?f(DotProduct) ->
	TotError*derivatives:AF(DotProduct).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP AUTO-ENCODER IMPLEMENTATIONS TIED-WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bp_Tied_DAE(IVL,LayerLength,TrainingLength)->
	Circuit=create_TDAE(IVL,LayerLength),
	TrainingSet = [{A,A}||A<-[[random:uniform(2)-1||_<-lists:seq(1,IVL)]||_<-lists:seq(1,10)]],
	{U_Circuit,RMS}=tied_backprop_train(0.5,Circuit,TrainingSet,length(TrainingSet),0,TrainingLength),
	Outputs=[calculate_output_std(I,U_Circuit#circuit.layers,[])||{I,DO}<-TrainingSet],
%	[io:format("I:~p DO:~p O:~p~n",[I,DO,O])||{{I,DO},{O,_}}<-lists:zip(TrainingSet,Outputs)],
	RMS/length(TrainingSet).

	create_TDAE(IVL,LL)->
		CircuitDynamics = static,
		CircuitLayersSpec =[
			#layer{neurode_type=tanh,tot_neurodes=LL,dynamics=static}
		],
		Circuit=create_InitCircuit([{input_id,IVL}],{CircuitDynamics,CircuitLayersSpec}),
		[L] = Circuit#circuit.layers,
		Circuit#circuit{layers=[L#layer{parameters=[random:uniform()/10||_<-lists:seq(1,IVL)]}]}.
		
	tied_backprop_train(_NoiseProb,Circuit,_TrainingSet,0,ErrAcc,1)->
		{Circuit,ErrAcc};
	tied_backprop_train(NoiseProb,Circuit,TrainingSet,0,ErrAcc,Index)->
		io:format("ErrAcc:~w~n",[ErrAcc]),
		tied_backprop_train(NoiseProb,Circuit,TrainingSet,length(TrainingSet),0,Index-1);
	tied_backprop_train(NoiseProb,Circuit,TrainingSet,SampleIndex,ErrAcc,Index)->
		{I,DO} = lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
		{Hidden_Output,[U_Layer1]} = calculate_output_std([case random:uniform()<NoiseProb of true -> 0; false -> Val end || Val<-I],Circuit#circuit.layers,[]),
%		io:format("Hidden_Output:~p~n",[Hidden_Output]),
		Pseudo_Weights_List = get_pseudoweights([N#neurode.weights || N<-U_Layer1#layer.neurodes],U_Layer1#layer.parameters),%basically transpositionov a matrix.
		Reconstructed_Dot = [dot(Hidden_Output,Pseudo_Weights,Bias)||{Pseudo_Weights,Bias}<-Pseudo_Weights_List],
		Reconstructed_Input = [math:tanh(Val)||Val <-Reconstructed_Dot],
%		io:format("Reconstructed_Dot:~p~n",[Reconstructed_Dot]),
		{OutputError_List,U_ErrAcc} = calculate_OutputError(Reconstructed_Input,DO,[],ErrAcc),
%		io:format("{OutputError_List,U_ErrAcc}:~p~n",[{OutputError_List,U_ErrAcc}]),
%		io:format("zip3:~p~n",[lists:zip3(Pseudo_Weights_List,OutputError_List,Reconstructed_Dot)]),
		{BackpropErrors,U_L1Parameters} = tied_backprop(Hidden_Output,Pseudo_Weights_List,OutputError_List,Reconstructed_Dot,[],[]),
%		io:format("BackpropErrors:~p~n",[BackpropErrors]),
		{U_Neurodes,ErrBackprop2} = layer_backprop(U_Layer1#layer.input,U_Layer1#layer.neurodes,BackpropErrors),
%		io:format("Layers:~p~nU_Layers:~p~n",[Circuit#circuit.layers,U_Layers2]),
		U_Layer2=U_Layer1#layer{neurodes=U_Neurodes,parameters=U_L1Parameters},
		tied_backprop_train(NoiseProb,Circuit#circuit{layers=[U_Layer2]},TrainingSet,SampleIndex-1,U_ErrAcc,Index).

		tied_backprop(Hidden_Output,[{DWP,Bias}|Pseudo_Weights_List],[Error|OutputError_List],[DotProduct|Reconstructed_Dot],ErrAcc,BiasAcc)->
			{_,U_Bias,Err}=backpropogate(delta(Error,DotProduct,tanh),Hidden_Output,DWP,Bias),
			tied_backprop(Hidden_Output,Pseudo_Weights_List,OutputError_List,Reconstructed_Dot,vector_add(Err,ErrAcc),[U_Bias|BiasAcc]);
		tied_backprop(_Hidden_Output,[],[],[],ErrAcc,BiasAcc)->
			{ErrAcc,lists:reverse(BiasAcc)}.

test_GetPseudoWeights(IVL,LayerLength)->
	Circuit=create_TDAE(IVL,LayerLength),
	[Layer] = Circuit#circuit.layers,
	Pseudo_Weights_List = get_pseudoweights([N#neurode.weights || N<-Layer#layer.neurodes],Layer#layer.parameters),%basically transpositionov a matrix.
	io:format("Layer:~p~nPseudoWeights:~p~n",[Layer,Pseudo_Weights_List]).
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP COMPLEX-VALUED AUTO-ENCODER IMPLEMENTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cplx_DAE(IVL,LayerLength,TrainingLength)->
	Circuit=cplx_create_autoencoder(IVL,LayerLength),
	TrainingSet = [{A,A}||A<-[[random:uniform(2)-1||_<-lists:seq(1,IVL)]||_<-lists:seq(1,10)]],
	{U_Circuit,RMS}=backprop_train(0.5,Circuit,TrainingSet,length(TrainingSet),0,TrainingLength),
	Outputs=[calculate_output_std(I,U_Circuit#circuit.layers,[])||{I,DO}<-TrainingSet],
%	[io:format("I:~p DO:~p O:~p~n",[I,DO,O])||{{I,DO},{O,_}}<-lists:zip(TrainingSet,Outputs)],
	RMS/length(TrainingSet).
	
	cplx_create_autoencoder(IVL,LL)->
		CircuitDynamics = static,
		CircuitLayersSpec =[
			#layer{neurode_type=cplx4,tot_neurodes=LL,dynamics=dynamic},
			#layer{neurode_type=tanh,tot_neurodes=IVL,dynamics=static}
		],
		create_InitCircuit([{input_id,IVL}],{CircuitDynamics,CircuitLayersSpec}).

	conditional_preprocessing([Val|Input])->
		case Val of
			{I,R}->
				[Val|Input];
			_ ->
				[complex:encode_input(-1,1,Val)|[complex:encode_input(-1,1,X)||X<-Input]]
		end.

	calculate_cplx_dot([{RX,IX}|IV],[{{RW,IW},{PDR,PDI},{LPR,LPI}}|WeightsP],Bias,{RAcc,IAcc})->
		calculate_cplx_dot(IV,WeightsP,Bias,{RW*RX-IW*IX+RAcc,IW*RX+RW*IX+IAcc});
	calculate_cplx_dot([],[],undefined,{RAcc,IAcc})->
		{RAcc,IAcc};
	calculate_cplx_dot([],[],{{RB,IB},{PDR,PDI},{LPR,LPI}},{RAcc,IAcc})->
		{RB+RAcc,IB+IAcc}.
	%calculate_dot(A,B,C)->
		%exit("A1:~p B1:~p C1:~p~n",[A,B,C]).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BP MULTI-VALUED NEURON AUTO-ENCODER IMPLEMENTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHC AUTO-ENCODER IMPLEMENTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shc_DAE(IVL,LayerLength,TrainingLength)->
	Noise = 0.25,
	OVL=IVL,
	CircuitLayerSpec=#layer{neurode_type=tanh,tot_neurodes=OVL,dynamics=dynamic},
	Circuit=create_DAE([{undefined,IVL}],CircuitLayerSpec,Noise,static,{bp,TrainingLength}),
	TrainingSet = [{A,A}||A<-[[random:uniform(2)-1||_<-lists:seq(1,IVL)]||_<-lists:seq(1,10)]],
	{U_Circuit,RMS}=optimize(0.5,Circuit,TrainingSet,length(TrainingSet),0,TrainingLength),
	Outputs=[calculate_output_std(I,U_Circuit#circuit.layers,[])||{I,DO}<-TrainingSet],
%	[io:format("I:~p DO:~p O:~p~n",[I,DO,O])||{{I,DO},{O,_}}<-lists:zip(TrainingSet,Outputs)],
	RMS/length(TrainingSet).

	optimize(_NoiseProb,Circuit,_TrainingSet,0,ErrAcc,1)->
		{Prev_ErrAcc, PrevCircuit} = get(err_acc),
		erase(err_acc),
		case Prev_ErrAcc < ErrAcc of
			true ->
				{PrevCircuit,Prev_ErrAcc};
			false ->
				{Circuit,ErrAcc}
		end;
	optimize(NoiseProb,Circuit,TrainingSet,0,ErrAcc,Index)->
%		io:format("ErrAcc:~w~n",[ErrAcc]),
		Perturbed_Circuit=case get(err_acc) of
			undefined ->
				put(err_acc,{ErrAcc,Circuit}),
				%io:format("ErrAcc:~w~n",[ErrAcc]),
				perturb(Circuit);
			{Prev_ErrAcc,PrevCircuit} ->
				case Prev_ErrAcc < ErrAcc of
					true ->
						%io:format("PrevErrAcc:~w~n",[Prev_ErrAcc]),
						perturb(PrevCircuit);
					false ->
						put(err_acc,{ErrAcc,Circuit}),
						io:format("NewErrAcc:~w~n",[ErrAcc]),
						perturb(Circuit)
				end
		end,
		optimize(NoiseProb,Perturbed_Circuit,TrainingSet,length(TrainingSet),0,Index-1);
	optimize(NoiseProb,Circuit,TrainingSet,SampleIndex,ErrAcc,Index)->
		{I,DO} = lists:nth(random:uniform(length(TrainingSet)),TrainingSet),
		NoisyInput=[case random:uniform()<NoiseProb of true -> 0; false -> Val end || Val<-I],
%		io:format("Circuit:~p~n",[Circuit]),
		{Output,U_Layers1} = calculate_output_std(NoisyInput,Circuit#circuit.layers,[]),
%		io:format("Output:~p~n",[Output]),
		{OutputError_List,U_ErrAcc} = calculate_OutputError(Output,DO,[],ErrAcc),
%		io:format("Layers:~p~nU_Layers:~p~n",[Circuit#circuit.layers,U_Layers2]),
		optimize(NoiseProb,Circuit#circuit{layers=U_Layers1},TrainingSet,SampleIndex-1,U_ErrAcc,Index).
		
		perturb(C)->
			Layers = C#circuit.layers,
			TotWeights=lists:sum(lists:flatten([[[length(N#neurode.weights)]||N<-L#layer.neurodes]||L<-Layers])),
			MP = 1/math:sqrt(TotWeights),
			U_Layers=[L#layer{neurodes=[N#neurode{weights=[perturb_wp(WP,MP,1)||WP<-N#neurode.weights],bias=perturb_wp(N#neurode.bias,MP,1)}||N<-L#layer.neurodes]}||L<-Layers],
			C#circuit{layers=U_Layers}.
			
		perturb_wp(undefined,_MP,_DMultiplier)->
			undefined;
		perturb_wp({W,PDW,LP},MP,DMultiplier)->
			WLimit = ?OUTPUT_SAT_LIMIT,
			case random:uniform() < MP of
				true ->
					DW = (random:uniform()-0.5)*DMultiplier,
					{functions:sat(W + DW,WLimit,-WLimit)+PDW*0.5,DW,LP};
				false ->
					{W,PDW,LP}
			end.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHC AUTO-ENCODER IMPLEMENTATIONS TIED_WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RESTRICTED BOLTZMAN MACHINE IMPLEMENTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rbm_test(IVL,LL)->
	TrainingSet = [{A,A}||A<-[[random:uniform(2)-1||_<-lists:seq(1,IVL)]||_<-lists:seq(1,10)]],
	Circuit=create_rbm(IVL,LL),
%	LType = rbm,
%	LParameters=[LLearningRate=0.1],
%	NType = tanh,
%	Weights = [random:uniform()/10||_<-lists:seq(1,IVL)],
%	NParameters = [NLearningRate=0.1],
%	Neurons = [{neurode,NType,Weights,NParameters,random:uniform()/10}||_<-lists:seq(1,TotNeurons)],
%	VisibleBias = [random:uniform()/10||_<-lists:seq(1,TotNeurons)],
%	Layer = {layer,LType,LParameters,Neurons},
%	Pseudo_Bias_List = [random:uniform()/10||_<-lists:seq(1,IVL)],
	U_Circuit=rbm_train_test(TrainingSet,Circuit).
	
	
	create_rbm(IVL,LL)->
		CircuitDynamics = static,
		CircuitLayersSpec =[
		#layer{neurode_type=tanh,tot_neurodes=LL,dynamics=static}
		],
		Circuit=create_InitCircuit([{input_id,IVL}],{CircuitDynamics,CircuitLayersSpec}),
		[L] = Circuit#circuit.layers,
		Circuit#circuit{layers=[L#layer{parameters=[random:uniform()/10||_<-lists:seq(1,IVL)]}]}.
		
	
	rbm_train_test(IV,Circuit)->
		[Layer] = Circuit#circuit.layers,
%		{layer,rbm,[LearningRate],Neurons}=Layer,
		io:format("RBM:~p~n",[Layer]),
		U_L=rbm_train(IV,IV,Layer,1000),
		Circuit#circuit{layers=[U_L]}.
		
		rbm_train(_MIL,_MIL,L,0)->
			L;
		rbm_train([{Input,DO}|TrainingSet],MIL,L,Index)->
			{U_Neurodes,U_Pseudo_Bias_List} = rbm_train(Input,L#layer.neurodes,L#layer.parameters),
			U_L = L#layer{neurodes=U_Neurodes,parameters=U_Pseudo_Bias_List},
			rbm_train(TrainingSet,MIL,U_L,Index);
		rbm_train([],MIL,L,Index)->
			rbm_train(MIL,MIL,L,Index-1).
			
		rbm_train(Input,Neurodes,Pseudo_Bias_List)->
			LearningRate = 0.1,
			Hidden_Output = [rbm_dot(Input,N#neurode.weights,N#neurode.bias) || N <-Neurodes],%
			Sampled_Hidden_Output = [case random:uniform() < 0.5 of true -> Val; false -> 0 end || Val <-Hidden_Output],%h1
			
			Pseudo_Weights_List = get_pseudoweights([N#neurode.weights || N<-Neurodes],Pseudo_Bias_List),%basically transpositionov a matrix.
			Reconstructed_Input = [rbm_dot(Sampled_Hidden_Output,Pseudo_Weights,Bias)||{Pseudo_Weights,Bias}<-Pseudo_Weights_List],%q_v2
			Sampled_Reconstructed_Input = [case random:uniform() < 0.5 of true -> Val; false -> 0 end || Val <-Reconstructed_Input],%v2
			
			Reconstructed_Output = [rbm_dot(Sampled_Reconstructed_Input,N#neurode.weights,N#neurode.bias) || N<-Neurodes],%q_h2
			%Sampled_Reconstructed_Output = [case random:uniform() < 0.5 of true -> Val; false -> 0 end || Val <-Reconstructed_Output],
			%io:format("Here~p~n",[{Pseudo_Bias_List,LearningRate,Input,Sampled_Reconstructed_Input}]),
			
			U_Pseudo_Bias_List = update_bias(LearningRate,Input,Sampled_Reconstructed_Input,Pseudo_Bias_List,[]),
			%[Bias + LearningRate*(Input-Sampled_Reconstructed_Input) || Bias<-Pseudo_Bias_List],
			
			Positive=[[I*O||I<-Input]||O<-Sampled_Hidden_Output],%a
			Negative=[[I*O||I<-Sampled_Reconstructed_Input]||O<-Reconstructed_Output],%b

			U_Neurodes=rbm_update_neurodes(Neurodes,Positive,Negative,LearningRate,Sampled_Hidden_Output,Reconstructed_Output,[]),
			%io:format("U_Neurodes:~p~n",[U_Neurodes]),
			io:format("Input:~p~nReconstructed_Input:~p~n",[Input,Reconstructed_Input]),
			{U_Neurodes,U_Pseudo_Bias_List}.
			
			update_bias(LR,[I|Input],[SRI|Sampled_Reconstructed_Input],[B|Bias_List],Acc)->
				U_B = B+LR*(I-SRI),
				update_bias(LR,Input,Sampled_Reconstructed_Input,Bias_List,[U_B|Acc]);
			update_bias(_LR,[],[],[],Acc)->
				lists:reverse(Acc).
			
			rbm_dot(Input,[{W,PDW,LP}|Weights],Bias)->
				rbm_dot1(Input,[{W,PDW,LP}|Weights],Bias,0);
			rbm_dot(Input,Weights,Bias)->
				rbm_dot2(Input,Weights,Bias,0).

			rbm_dot1([I|Input],[{W,_PDW,_LP}|Weights],Bias,Acc)->
				%io:format("{~p,~p}~n",[I,W]),
				rbm_dot1(Input,Weights,Bias,I*W+Acc);
			rbm_dot1([],[],undefined,Acc)->
				math:tanh(Acc);
			rbm_dot1([],[],{Bias,_PDB,_LP},Acc)->
				%io:format("Bias:~p Acc:~p~n",[Bias,Acc]),
				math:tanh(Acc+Bias);
			rbm_dot1([],[],Bias,Acc)->
				math:tanh(Acc+Bias).
			
			rbm_dot2([I|Input],[W|Weights],Bias,Acc)->
				%io:format("{~p,~p}~n",[I,W]),
				rbm_dot2(Input,Weights,Bias,I*W+Acc);
			rbm_dot2([],[],undefined,Acc)->
				math:tanh(Acc);
			rbm_dot2([],[],Bias,Acc)->
				%io:format("Acc:~p~n",[Acc]),
				math:tanh(Acc+Bias).
				
			dot(Input,[{W,PDW,LP}|Weights],Bias)->
				dot1(Input,[{W,PDW,LP}|Weights],Bias,0);
			dot(Input,Weights,Bias)->
				dot2(Input,Weights,Bias,0).

			dot1([I|Input],[{W,_PDW,_LP}|Weights],Bias,Acc)->
				%io:format("{~p,~p}~n",[I,W]),
				dot1(Input,Weights,Bias,I*W+Acc);
			dot1([],[],undefined,Acc)->
				Acc;
			dot1([],[],{Bias,_PDB,_LP},Acc)->
				%io:format("Bias:~p Acc:~p~n",[Bias,Acc]),
				Acc+Bias;
			dot1([],[],Bias,Acc)->
				Acc+Bias.
			
			dot2([I|Input],[W|Weights],Bias,Acc)->
				%io:format("{~p,~p}~n",[I,W]),
				dot2(Input,Weights,Bias,I*W+Acc);
			dot2([],[],undefined,Acc)->
				Acc;
			dot2([],[],Bias,Acc)->
				%io:format("Acc:~p~n",[Acc]),
				Acc+Bias.
				
				
				sigmoid(X)->
					1/(1+math:exp(-X)).
				
			get_pseudoweights([[{W,PDW,LP}|Weights]|Weights_List],PBL)->
				get_pseudoweights1([[{W,PDW,LP}|Weights]|Weights_List],[],[],[],PBL);
			get_pseudoweights(Weights_List,PBL)->
				get_pseudoweights2(Weights_List,[],[],[],PBL).
				
			get_pseudoweights1([[{W,PDW,LP}|Weights]|Weights_List],WAcc,PWAcc,Acc,PBL)->
				get_pseudoweights1(Weights_List,[Weights|WAcc],[{W,PDW,LP}|PWAcc],Acc,PBL);
			get_pseudoweights1([],WAcc,PWAcc,Acc,PBL)->
				%io:format("PBL:~p~n WAcc:~p~n PWAcc:~p~n Acc:~p~n",[PBL,WAcc,PWAcc,Acc]),
				case WAcc of
					[[]|_] ->
						[PB] = PBL,
						[{lists:reverse(Weights),Bias}||{Weights,Bias}<-lists:reverse([{PWAcc,PB}|Acc])];
					_ ->
						[PB|PBLr] = PBL,
						get_pseudoweights1(lists:reverse(WAcc),[],[],[{PWAcc,PB}|Acc],PBLr)
				end.
				
			get_pseudoweights2([[W|Weights]|Weights_List],WAcc,PWAcc,Acc,PBL)->
				get_pseudoweights2(Weights_List,[Weights|WAcc],[W|PWAcc],Acc,PBL);
			get_pseudoweights2([],WAcc,PWAcc,Acc,PBL)->
				%io:format("PBL:~p~n",[PBL]),
				case WAcc of
					[[]|_] ->
						[PB] = PBL,
						lists:reverse([{lists:reverse(Weights),Bias}||{Weights,Bias}<-lists:reverse([{PWAcc,PB}|Acc])]);
					_ ->
						[PB|PBLr] = PBL,
						get_pseudoweights2(lists:reverse(WAcc),[],[],[{PWAcc,PB}|Acc],PBLr)
				end.
				
			rbm_update_neurodes([N|Neurodes],[Pos|Positive],[Neg|Negative],LearningRate,[SHO|Sampled_Hidden_Output],[RHO|Reconstructed_Hidden_Output],Acc)->
				%U_Hidden_Biases = [Bias + LearningRate*(Sampled_Hidden_Output-Reconstructed_Output) || Bias<-Hidden_Bias_List],
%				{neurode,NType,Weights,NParameters,Bias}=Neuron,
				U_Weights=rbm_update_weights(N#neurode.weights,Pos,Neg,LearningRate),
				U_Bias = case N#neurode.bias of
					{B,PBD,BLP}-> {B+LearningRate*(SHO-RHO),PBD,BLP};
					B -> B + LearningRate*(SHO-RHO)
				end,
				%io:format("U_Weights:~p~n",[U_Weights]),
				rbm_update_neurodes(Neurodes,Positive,Negative,LearningRate,Sampled_Hidden_Output,Reconstructed_Hidden_Output,[N#neurode{weights=U_Weights,bias=U_Bias}|Acc]);
			rbm_update_neurodes([],[],[],_LearningRate,[],[],Acc)->
				lists:reverse(Acc).

				rbm_update_weights([{W,PDW,LP}|Weights],Pos,Neg,LearningRate)->
					rbm_update_weights1([{W,PDW,LP}|Weights],Pos,Neg,LearningRate,[]);
				rbm_update_weights(Weights,Pos,Neg,LearningRate)->
					rbm_update_weights2(Weights,Pos,Neg,LearningRate,[]).
					
				rbm_update_weights1([{W,PDW,LP}|Weights],[PVal|P],[NVal|N],LearningRate,Acc)->
					U_W = W+LearningRate*(PVal-NVal),
					rbm_update_weights1(Weights,P,N,LearningRate,[{functions:sat(U_W,10,-10),PDW,LP}|Acc]);
				rbm_update_weights1([],[],[],LearningRate,Acc)->
					lists:reverse(Acc).
					
				rbm_update_weights2([W|Weights],[PVal|P],[NVal|N],LearningRate,Acc)->
					U_W = W+LearningRate*(PVal-NVal),
					rbm_update_weights2(Weights,P,N,LearningRate,[functions:sat(U_W,10,-10)|Acc]);
				rbm_update_weights2([],[],[],LearningRate,Acc)->
					lists:reverse(Acc).
					
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPETITIVE LAYER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HEBBIAN LAYER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OJAS LAYER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KOHONAN MAP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
