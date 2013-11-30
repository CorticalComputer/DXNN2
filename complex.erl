-module(complex).
-include("records.hrl").
-compile(export_all).

conjugate({R,I})->
	{R,-1*I}.
add({R1,I1},{R2,I2})->
	{R1+R2,I1+I2}.
subtract({R1,I1},{R2,I2})->
	{R1-R2,I1-I2}.
multiply({A,B},{C,D})->
	R = A*C-B*D,
	I = B*C+A*D,
	{R,I}.
divide({A,B},{C,D})->
	case C*C+D*D of
		0 ->
			{0,0};
		Denom ->
			R = (A*C+B*D)/Denom,
			I = (B*C-A*D)/Denom,
			{R,I}
	end.
sqrt({A,0})->
	{math:sqrt(A),0};
sqrt({A,B})->
	Modulus = math:sqrt(A*A+B*B),
	R = math:sqrt((A+Modulus)/2),
	I = functions:sgn(B)*math:sqrt((-A+Modulus)/2),
	A1 = {R,I},
	A2 = {-R,-I},
	A1.

-define(BP,[1,-1]).
make_tables(IL)->
	Combinations=make_tables(1.0*IL-1,[[Val]||Val<-?BP]),
	Truths=make_tables(math:pow(2,IL)-1,[[Val]||Val<-?BP]),
	Result=[lists:zip(Combinations,Sublist)||Sublist<-Truths],
	%io:format("Truth Tables:~n~p~n",[Result]),
	Result.
	
	make_tables(0.0,Acc)->
		Acc;
	make_tables(IL,Acc)->
		U_Acc=[[Val|List] || Val<-?BP, List<-Acc],
		make_tables(IL-1,U_Acc).

gen_weights(0,Acc)->
	Acc;
gen_weights(Index,Acc)->
	gen_weights(Index-1,[{random:uniform()-0.5,random:uniform()-0.5}|Acc]).

encode_input(Min,Max,X)->
	Phi = math:pi()*(X-Min)/(Max-Min),
	%io:format("Phi:~p~n",[Phi]),
	Z = {math:cos(Phi),math:sin(Phi)}.

decode_output(Min,Max,X)->
	Phi = 2*(X-Min)/(Max-Min) - 1.

scale(Min,Max,X)->
	Phi = math:pi()*(X-Min)/(Max-Min).
descale(Min,Max,X)->
	Phi = 2*(X-Min)/(Max-Min) - 1.


cplx_dot([{RW,IW}|Weights],Bias,Input)->
	case Bias of
		undefined ->
			dot([{RW,IW}|Weights],Input,{0,0});
		_ ->
			dot([{RW,IW}|Weights],Input,Bias)
	end;
cplx_dot([{W,PDW,LP}|Weights],Bias,Input)->
	case Bias of
		undefined ->
			dot2([{W,PDW,LP}|Weights],Input,{0,0});
		_ ->
			{B,_PDB,BLP}=Bias,
			dot2([{W,PDW,LP}|Weights],Input,B)
	end.
	
	dot([{RW,IW}|Weights],[{RX,IX}|Input],{RAcc,IAcc})->
		dot(Weights,Input,{RW*RX-IW*IX+RAcc,IW*RX+RW*IX+IAcc});
	dot([],[],{RAcc,IAcc})->
		{RAcc,IAcc}.

	dot2([{{RW,IW},_,_}|Weights],[{RX,IX}|Input],{RAcc,IAcc})->
		dot2(Weights,Input,{RW*RX-IW*IX+RAcc,IW*RX+RW*IX+IAcc});
	dot2([],[],{RAcc,IAcc})->
		{RAcc,IAcc}.


tanh(X)->
	1/(1+math:exp(-X)).


cplx1({R,I})->
	math:sqrt(math:pow(tanh(R),2)+math:pow(tanh(I),2)).
	
cplx2({R,I})->
	math:pow(tanh(R)-tanh(I),2).%Testing complete, failure rate:11800/65536

cplx3({R,I})->
	math:sqrt(math:pow(sigmoid(R),2)+math:pow(sigmoid(I),2)).

cplx4({R,I})->
	math:pow(sigmoid(R)-sigmoid(I),2).%Testing complete, failure rate:11800/65536

sigmoid(Val)->
	1/(1+math:exp(-Val)).

cplx5({R,I})->
	(tanh(R) + tanh(I))/2.%DRO=derivatives:tanh(R)/2, DIO=derivatives:tanh(I)/2
	
cplx6({R,I})->
	math:sqrt(R*R+I*I).%DRO=(1/2*(math:sqrt(R*R+I*I))*2*R, DIO=(1/2*(math:sqrt(R*R+I*I))*2*I
	
cplx7({R,I})->
	tanh(R)-tanh(I).%DRO=derivatives:tanh(R), DIO=-derivatives:tanh(I)

create_neurons(0,_IVL,Acc)->
	Acc;
create_neurons(Neuron_Index,IVL,Acc)->
	create_neurons(Neuron_Index-1,IVL,[{Neuron_Index,gen_weights(IVL,[])}|Acc]).

forward(Neurons,IV,Min,Max)->
	[apply_neuron(Neuron,IV,Min,Max) || Neuron<-Neurons].
	
apply_neuron(Neuron,IV,Min,Max)->
	{Neuron_Index,[Bias|Weights]} = Neuron,
	%io:format("Neuron:~p~n IV:~p~n",[Neuron,IV]),
	Preprocessed_IV=[encode_input(Min,Max,X) || X<-IV],
	%io:format("IV:~p~nPreprocessed_IV:~p~n",[IV,Preprocessed_IV]),
	Cplx_DotProduct=dot(Weights,Preprocessed_IV,Bias),
	[cplx2(Cplx_DotProduct)*2-1].
	
test_neurons(TotNeurons,IV)->
	IVL = length(IV),
	Neurons=create_neurons(TotNeurons,IVL,[]),
	forward(Neurons,IV,-1,1).
	
test_XOR()->
	XOR = [
		{[0,1],[1]},
		{[1,0],[1]},
		{[1,1],[0]},
		{[0,0],[0]}
	],
	Neuron=create_neurons(1,3,[]),
	%io:format("XOR:~p Neuron:~p~n",[XOR,Neuron]),
	train(Neuron,XOR,undefined,10000).
	
	train_with_restarts(Neuron,Problem,MaxEvals,0)->
		failed;
	train_with_restarts(Neuron,Problem,MaxEvals,RestartIndex)->
		case train(Neuron,Problem,undefined,MaxEvals) of
			failed ->
				train_with_restarts(Neuron,Problem,MaxEvals,RestartIndex-1);
			U_Neuron ->
				U_Neuron
		end.
		
	train(Neuron,Problem,Error,0)->
		%io:format("Problem:~p~n Failed!!!, final neuron:~p~n final error:~p~n",[Problem,Neuron,Error]),
		failed;
	train(Neuron,Problem,Error,MaxEvals)->
		[PerturbedNeuron] = perturb(Neuron),
		Outputs=[apply_neuron(PerturbedNeuron,IV,-1,1)||{IV,_EO}<-Problem],
		%io:format("Problem:~p Outputs:~p~n",[Problem,Outputs]),
		RMSE = get_RMSE(Outputs,[EO||{_IV,EO}<-Problem],0),
		case RMSE < 0.1 of
			true -> 
				Neuron;
			false ->
				case RMSE < Error of
					true ->
						io:format("RMSE:~p~n",[RMSE]),
						train([PerturbedNeuron],Problem,RMSE,MaxEvals-1);
					false ->
						train(Neuron,Problem,Error,MaxEvals-1)
				end
		end.
		
	perturb(Neurons)->
		perturb(Neurons,[]).
	perturb([{Id,Weights}|Neurons],Acc)->
		U_Weights=[case random:uniform() < 0.5 of true -> {R+random:uniform()-0.5,I+random:uniform()-0.5}; false -> {R,I} end||{R,I}<-Weights],
		perturb(Neurons,[{Id,U_Weights}|Acc]);
	perturb([],Acc)->
		lists:reverse(Acc).
		
	get_RMSE([[O]|Output],[[E]|Expected],Acc)->
		%io:format("O:~p E:~p~n",[O,E]),
		get_RMSE(Output,Expected,math:pow(O-E,2)+Acc);
	get_RMSE([[O]|Output],[E|Expected],Acc)->
		%io:format("O:~p E:~p~n",[O,E]),
		get_RMSE(Output,Expected,math:pow(O-E,2)+Acc);
	get_RMSE([],[],Acc)->
		math:sqrt(Acc).
		
test_all(VL)->
	Tests=make_tables(VL),
	TotFailures=test_all(VL,Tests,length(Tests),10000),
	io:format("Testing complete, failure rate:~p/~p~n",[TotFailures,length(Tests)]).
	
	test_all(VL,[T|Tests],TestIndex,MaxEvals)->
		Neuron = create_neurons(1,VL+1,[]),
		case train_with_restarts(Neuron,T,MaxEvals,10) of
			failed ->
				io:format("FAILURE!!!~n"),
				test_all(VL,Tests,TestIndex,MaxEvals);
			U_Neuron->
				io:format("SUCCESS!!!~n"),
				test_all(VL,Tests,TestIndex-1,MaxEvals)
		end;
	test_all(_VL,[],TestIndex,_MaxEvals)->
		TestIndex.

test_all2(VL)->
	Tests=make_tables(VL),
	io:format("Tests:~p~n",[Tests]),
	TotFailures=test_all2(VL,Tests,length(Tests),10000),
	io:format("Testing complete, failure rate:~p/~p~n",[TotFailures,length(Tests)]).
	
	test_all2(VL,[T|Tests],TestIndex,MaxEvals)->
		Circuit = [[circuit:add_bias(Neurode) || Neurode<-Layer]||Layer<-technome_constructor:create_circuit(VL,[2,1],tanh)],
		case train_with_restarts2(Circuit,T,MaxEvals,10) of
			failed ->
				io:format("FAILURE!!!~n"),
				test_all2(VL,Tests,TestIndex,MaxEvals);
			U_Circuit->
				io:format("SUCCESS!!!~n"),
				test_all2(VL,Tests,TestIndex-1,MaxEvals)
		end;
	test_all2(_VL,[],TestIndex,_MaxEvals)->
		TestIndex.
		
	train_with_restarts2(Circuit,Problem,MaxEvals,0)->
		failed;
	train_with_restarts2(Circuit,Problem,MaxEvals,RestartIndex)->
		case train2(Circuit,Problem,undefined,MaxEvals) of
			failed ->
				train_with_restarts2(Circuit,Problem,MaxEvals,RestartIndex-1);
			U_Neuron ->
				U_Neuron
		end.
		
	train2(Circuit,Problem,Error,0)->
		%io:format("Problem:~p~n Failed!!!, final neuron:~p~n final error:~p~n",[Problem,Neuron,Error]),
		failed;
	train2(Circuit,Problem,Error,MaxEvals)->
		PerturbedCircuit = circuit:perturb_circuit(Circuit,math:pi()),
		Outputs=[[circuit:calculate_output_std(IV,PerturbedCircuit)]||{IV,_EO}<-Problem],
		%io:format("Problem:~p Outputs:~p~n",[Problem,Outputs]),
		RMSE = get_RMSE(Outputs,[EO||{_IV,EO}<-Problem],0),
		case RMSE < 0.1 of
			true -> 
				PerturbedCircuit;
			false ->
				case RMSE < Error of
					true ->
						%io:format("RMSE:~p~n",[RMSE]),
						train2(PerturbedCircuit,Problem,RMSE,MaxEvals-1);
					false ->
						train2(Circuit,Problem,Error,MaxEvals-1)
				end
		end.

calculate_output(DIV,WeightsP)->
	Preprocessed_IV=[{From,conditional_preprocessing(Input)} || {From,Input}<-DIV],
	%io:format("DIV:~p~nPreprocessed_IV:~p~nWeightsP:~p~n",[DIV,Preprocessed_IV,WeightsP]),
	Cplx_DotProduct=calculate_dot(WeightsP,Preprocessed_IV,{0,0}),
	cplx2(Cplx_DotProduct)*2-1.
	
	conditional_preprocessing([Val|Input])->
		case Val of
			{I,R}->
				[Val|Input];
			_ ->
				[encode_input(-1,1,Val)|[encode_input(-1,1,X)||X<-Input]]
		end.

	calculate_dot([{From,WPC}|WeightsP],[{From,Input}|DIV],{RAcc,IAcc})->
		{R,I}=calculate_dot2(WPC,Input,{0,0}),
		calculate_dot(WeightsP,DIV,{RAcc+R,I+IAcc});
	calculate_dot([],[],{RAcc,IAcc})->
		{RAcc,IAcc};
	calculate_dot([{threshold,[{RThreshold,IThreshold,_}]}],[],{RAcc,IAcc})->
		{RThreshold+RAcc,IThreshold+IAcc}.
	%calculate_dot(A,B,C)->
		%exit("A1:~p B1:~p C1:~p~n",[A,B,C]).
		
		calculate_dot2([{RW,IW,_}|Weights],[{RX,IX}|Input],{RAcc,IAcc})->
			calculate_dot2(Weights,Input,{RW*RX-IW*IX+RAcc,IW*RX+RW*IX+IAcc});
		calculate_dot2([],[],{RAcc,IAcc})->
			{RAcc,IAcc}.
		%calculate_dot2(A,B,C)->
			%exit("A:~p B:~p C:~p~n",[A,B,C]).
			
calculate_output2(DIV,WeightsP)->
	%io:format("DIV:~p WeightsP:~p~n",[DIV,WeightsP]),
	Cplx_DotProduct=calculate_dot(WeightsP,DIV,{0,0}),
	cplx2(Cplx_DotProduct).


	bo2bi(true)->1;
	bo2bi(false)->0.
		
test_Neuron(VL)->
	Tests=make_tables(VL),
	TF = [true,false],
	%Tests = [[{[bo2bi(A),bo2bi(B),bo2bi(C),bo2bi(D)],bo2bi((A xor B) and (C xor D))} || A<-TF,B<-TF,C<-TF,D<-TF]],
	io:format("Tests:~p~n",[Tests]),
	TotFailures=test_all3(VL,Tests,length(Tests),10000),
	io:format("Testing complete, failure rate:~p/~p~n",[TotFailures,length(Tests)]).
	
	test_all3(VL,[T|Tests],TestIndex,MaxEvals)->
		Neuron = create_Neuron(VL),
		N_PId=spawn_Neuron(Neuron),
	
		case train_with_restarts3(VL,N_PId,T,MaxEvals,10) of
			failed ->
				io:format("FAILURE!!!~n"),
				test_all3(VL,Tests,TestIndex,MaxEvals);
			U_Circuit->
				io:format("SUCCESS!!!~n"),
				test_all3(VL,Tests,TestIndex-1,MaxEvals)
		end;
	test_all3(_VL,[],TestIndex,_MaxEvals)->
		TestIndex.
		
	train_with_restarts3(VL,N_PId,Problem,MaxEvals,0)->
		failed;
	train_with_restarts3(VL,N_PId,Problem,MaxEvals,RestartIndex)->
		case train3(N_PId,Problem,undefined,MaxEvals) of
			failed ->
				NewNeuron = create_Neuron(VL),
				New_N_PId=spawn_Neuron(NewNeuron),
				train_with_restarts3(VL,New_N_PId,Problem,MaxEvals,RestartIndex-1);
			U_Neuron ->
				U_Neuron
		end.
		
	train3(N_PId,Problem,Error,0)->
		%io:format("Problem:~p~n Failed!!!, final neuron:~p~n final error:~p~n",[Problem,N_PId,Error]),
		N_PId ! {self(),terminate},
		failed;
	train3(N_PId,Problem,Error,MaxEvals)->
		%Send signal to N_PId to perturb itself.
		%gather outputs
		N_PId ! {self(),gt,weight_mutate,math:pi()},
		%PerturbedCircuit = circuit:perturb_circuit(Circuit,math:pi()),
		Outputs=debrief_Neuron(N_PId,Problem,[]),
		%Outputs=[[circuit:calculate_output_std(IV,PerturbedCircuit)]||{IV,_EO}<-Problem],
		%io:format("Problem:~p Outputs:~p~n",[Problem,Outputs]),
		RMSE = get_RMSE(Outputs,[EO||{_IV,EO}<-Problem],0),
		case RMSE < 0.1 of
			true -> 
				N_PId ! {self(),terminate},
				ok;
			false ->
				case RMSE < Error of
					true ->
						io:format("RMSE:~p~n",[RMSE]),
						N_PId ! {self(),gt,weight_save},
						train3(N_PId,Problem,RMSE,MaxEvals-1);
					false ->
						N_PId ! {self(),gt,weight_revert},
						train3(N_PId,Problem,Error,MaxEvals-1)
				end
		end.	
	
		debrief_Neuron(N_PId,[{IV,_EO}|Problem],Acc)->
			N_PId ! {self(),forward,IV},
			receive
				{N_PId,forward,Output}->
					debrief_Neuron(N_PId,Problem,[Output|Acc])
			end;
		debrief_Neuron(_N_PId,[],Acc)->
			lists:reverse(Acc).
	
	create_Neuron(IVL)->
		N_TotIVL = IVL,
		N_I = [{self(),IVL}],
		N_TotOVL = 1,
		N_O = [self()],
		SU_Id=self(),
		Generation=1,
		N_Id = test,
		Neural_Type=standard,
		Heredity_Type=darwinian,
		SpeCon = #constraint{neural_afs =[cplx], neural_aggr_fs=[dot_product]},
		AF=technome_constructor:generate_NeuralAF(SpeCon#constraint.neural_afs,Neural_Type,N_Id),
		DWP=case Neural_Type of
			circuit ->
				technome_constructor:create_circuit(N_TotIVL,[2,1],AF);
				%create_circuit(N_TotIVL,[1+random:uniform(round(math:sqrt(N_TotIVL))),1]);
			standard ->
				Val = technome_constructor:create_NWP(N_I,[]),
				lists:append(Val,[{threshold,[technome_constructor:weight_tuple()]}])
		end,
		Neuron = #neuron{
			id = N_Id,
			%type = Neural_Type,
			heredity_type = Heredity_Type,
			ivl = N_TotIVL,
			i = N_I,
			ovl = N_TotOVL,
			o = N_O,
			%lt = LearningType,
			preprocessor = technome_constructor:generate_NeuralPreprocessor(SpeCon#constraint.neural_preprocessors,Neural_Type,N_Id),
			signal_integrator = technome_constructor:generate_NeuralSignalIntegrator(SpeCon#constraint.neural_signal_integrators,Neural_Type,N_Id),
			activation_function=AF,
			postprocessor = technome_constructor:generate_NeuralPostprocessor(SpeCon#constraint.neural_postprocessors,Neural_Type,N_Id),
			plasticity=technome_constructor:generate_NeuralPF(SpeCon#constraint.neural_pfs,Neural_Type,N_Id),
			ro = technome_constructor:calculate_RO(SU_Id,N_Id,N_O,[]),
			dwp = DWP,
			su_id = SU_Id,
			generation = Generation
		}.
	
	spawn_Neuron(N)->
		{ok,N_PId}=neuron:gen(self(),node()),
		%[N] = mnesia:dirty_read({neuron,N_Id}),
		%io:format("~p~n",[N]),
		%N_PId = ets:lookup_element(IdsNPids,N_Id,2),
		I_PIds = [{self(),IVL}|| {I_Id,IVL} <- N#neuron.i], %I from Cortex xor Subcore or Neurons
		O_PIds = [self()], %O to Cortex xor Subcore or Neurons
		RO_PIds =[],
		NDWP = case N#neuron.type of
			standard ->
				[{case Id of threshold -> threshold; _ -> self() end,WPC} || {Id,WPC} <- N#neuron.dwp];
			circuit ->
				N#neuron.dwp
		end,
		%io:format("I_PIds:~p~n NDWP:~p~n",[I_PIds,NDWP]),
		TotIVL = N#neuron.ivl,
		TotOVL = N#neuron.ovl,
		%LearningType = N#neuron.lt,
		SU_Id = N#neuron.su_id,
		
		Parameters=N#neuron.parameters,
		PreProcessors=N#neuron.preprocessor,
		SignalIntegrator=N#neuron.signal_integrator,
		ActivationFunction=N#neuron.activation_function,
		PostProcessor=N#neuron.postprocessor,
		Plasticity=N#neuron.plasticity,
		MLFFNN_Module=N#neuron.mlffnn_module,
		Type=N#neuron.type,
		Heredity_Type=N#neuron.heredity_type,
		State = {self(),N#neuron.id,SU_Id,TotIVL,I_PIds,TotOVL,O_PIds,RO_PIds,NDWP,Parameters,PreProcessors,SignalIntegrator,ActivationFunction,PostProcessor,Plasticity,MLFFNN_Module,Type,Heredity_Type},
		N_PId ! {self(),init,State},
		N_PId.
