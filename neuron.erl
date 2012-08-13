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
-module(neuron).
-compile(export_all).
-include("records.hrl").
-define(DELTA_MULTIPLIER,math:pi()*2).
-define(SAT_LIMIT,math:pi()*2).
-define(RO_SIGNAL,0).
-record(state,{
	id,
	cx_pid,
	af,
	pf,
	aggrf,
	heredity_type,
	si_pids=[],
	si_pidps_bl = [],
	si_pidps_current=[],
	si_pidps_backup=[],
	mi_pids=[],
	mi_pidps_current=[],
	mi_pidps_backup=[],
	output_pids=[],
	ro_pids=[]
}).
gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf_PId) ->
	random:seed(now()),
	receive 
		{ExoSelf_PId,{Id,Cx_PId,AF,PF,AggrF,HeredityType,SI_PIdPs,MI_PIdPs,Output_PIds,RO_PIds}} ->
			fanout(RO_PIds,{self(),forward,[?RO_SIGNAL]}),
			SI_PIds = lists:append([IPId || {IPId,_W} <- SI_PIdPs, IPId =/= bias],[ok]),
			MI_PIds = lists:append([IPId || {IPId,_W} <- MI_PIdPs, IPId =/= bias],[ok]),
			%io:format("SI_PIdPs:~p ~nMI_PIdPs:~p~n",[SI_PIdPs,MI_PIdPs]),
			S=#state{
				id=Id,
				cx_pid=Cx_PId,
				af=AF,
				pf=PF,
				aggrf=AggrF,
				heredity_type = HeredityType,
				si_pids=SI_PIds,
				si_pidps_bl = SI_PIdPs,
				si_pidps_current=SI_PIdPs,
				si_pidps_backup=SI_PIdPs,
				mi_pids=MI_PIds,
				mi_pidps_current=MI_PIdPs,
				mi_pidps_backup=MI_PIdPs,
				output_pids=Output_PIds,
				ro_pids=RO_PIds
			},
			loop(S,ExoSelf_PId,SI_PIds,MI_PIds,[],[])
	end.
%When gen/2 is executed, it spawns the neuron element and immediately begins to wait for its initial state message from the exoself. Once the state message arrives, the neuron sends out the default forward signals to any elements in its ro_ids list, if any. Afterwards, prep drops into the neuron's main loop.

loop(S,ExoSelf_PId,[ok],[ok],SIAcc,MIAcc)->
	PF = S#state.pf,
	%PreProcessors=S#state.pre_processors,
	%SignalIntegrator=S#state.signal_integrator,
	AF = S#state.af,
	%PostProcessors=S#state.post_processors,
	AggrF = S#state.aggrf,
	{PFName,PFParameters} = PF,
	%io:format("self:~p~n SIAcc:~p~n MIAcc:~p~n",[self(), SIAcc,MIAcc]),
	Ordered_SIAcc = lists:reverse(SIAcc),
	SI_PIdPs = S#state.si_pidps_current,
	SAggregation_Product = sat(signal_aggregator:AggrF(Ordered_SIAcc,SI_PIdPs),?SAT_LIMIT),
	SOutput = functions:AF(SAggregation_Product),
	
	Output_PIds = S#state.output_pids,
	[Output_PId ! {self(),forward,[SOutput]} || Output_PId <- Output_PIds],
	
	case PFName of
		none ->
			U_S=S;
		_ ->%io:format("MIAcc:~p, S:~p~n",[MIAcc,S]),
			Ordered_MIAcc = lists:reverse(MIAcc),
			MI_PIdPs = S#state.mi_pidps_current,
			MAggregation_Product = sat(signal_aggregator:dot_product(Ordered_MIAcc,MI_PIdPs),?SAT_LIMIT),
			MOutput = functions:tanh(MAggregation_Product),
			U_SI_PIdPs = plasticity:PFName([MOutput|PFParameters],Ordered_SIAcc,SI_PIdPs,SOutput),
			%io:format("U_SI_PIdPs:~p~n",[U_SI_PIdPs]),
			U_S=S#state{
				si_pidps_current = U_SI_PIdPs
			}
	end,
	SI_PIds = S#state.si_pids,
	MI_PIds = S#state.mi_pids,
	neuron:loop(U_S,ExoSelf_PId,SI_PIds,MI_PIds,[],[]);
loop(S,ExoSelf_PId,[SI_PId|SI_PIds],[MI_PId|MI_PIds],SIAcc,MIAcc)->
	receive
		{SI_PId,forward,Input}->
			loop(S,ExoSelf_PId,SI_PIds,[MI_PId|MI_PIds],[{SI_PId,Input}|SIAcc],MIAcc);
		{MI_PId,forward,Input}->
			loop(S,ExoSelf_PId,[SI_PId|SI_PIds],MI_PIds,SIAcc,[{MI_PId,Input}|MIAcc]);
		{ExoSelf_PId,weight_backup}->
			U_S=case S#state.heredity_type of
				darwinian ->
					S#state{
						si_pidps_backup=S#state.si_pidps_bl,
						mi_pidps_backup=S#state.mi_pidps_current
					};
				lamarckian ->
					S#state{
						si_pidps_backup=S#state.si_pidps_current,
						mi_pidps_backup=S#state.mi_pidps_current
					}
			end,
			loop(U_S,ExoSelf_PId,[SI_PId|SI_PIds],[MI_PId|MI_PIds],SIAcc,MIAcc);
		{ExoSelf_PId,weight_restore}->
			U_S = S#state{
				si_pidps_bl=S#state.si_pidps_backup,
				si_pidps_current=S#state.si_pidps_backup,
				mi_pidps_current=S#state.mi_pidps_backup
			},
			loop(U_S,ExoSelf_PId,[SI_PId|SI_PIds],[MI_PId|MI_PIds],SIAcc,MIAcc);
		{ExoSelf_PId,weight_perturb,Spread}->
			Perturbed_SIPIdPs=perturb_IPIdPs(Spread,S#state.si_pidps_backup),
			Perturbed_MIPIdPs=perturb_IPIdPs(Spread,S#state.mi_pidps_backup),
			U_S=S#state{
				si_pidps_bl=Perturbed_SIPIdPs,
				si_pidps_current=Perturbed_SIPIdPs,
				mi_pidps_current=Perturbed_MIPIdPs
			},
			loop(U_S,ExoSelf_PId,[SI_PId|SI_PIds],[MI_PId|MI_PIds],SIAcc,MIAcc);
		{ExoSelf_PId,reset_prep}->
			neuron:flush_buffer(),
			ExoSelf_PId ! {self(),ready},
			RO_PIds = S#state.ro_pids,
			receive 
				{ExoSelf_PId, reset}->
					fanout(RO_PIds,{self(),forward,[?RO_SIGNAL]})
			end,
			loop(S,ExoSelf_PId,S#state.si_pids,S#state.mi_pids,[],[]);
		{ExoSelf_PId,get_backup}->
			NId = S#state.id,
			ExoSelf_PId ! {self(),NId,S#state.si_pidps_backup,S#state.mi_pidps_backup},
			loop(S,ExoSelf_PId,[SI_PId|SI_PIds],[MI_PId|MI_PIds],SIAcc,MIAcc);
		{ExoSelf_PId,terminate}->
			%io:format("Neuron:~p is terminating.~n",[self()])
			ok
		%after 10000 ->
			%io:format("neuron:~p stuck.~n",[S#state.id])
	end.
%The neuron process waits for vector signals from all the processes that it's connected from, taking the dot product of the input and weight vectors, and then adding it to the accumulator. Once all the signals from Input_PIds are received, the accumulator contains the dot product to which the neuron then adds the bias and executes the activation function. After fanning out the output signal, the neuron again returns to waiting for incoming signals. When the neuron receives the {ExoSelf_PId,get_backup} message, it forwards to the exoself its full MInput_PIdPs list, and its Id. The MInput_PIdPs contains the modified, tuned and most effective version of the input_idps. The neuron process is also accepts weight_backup signal, when receiving it the neuron saves to process dictionary the current MInput_PIdPs. When the neuron receives the weight_restore signal, it reads back from the process dictionary the stored Input_PIdPs, and switches over to using it as its active Input_PIdPs list. When the neuron receives the weight_perturb signal from the exoself, it perturbs the weights by executing the perturb_Lipids/1 function, which returns the updated list. Finally, the neuron can also accept a reset_prep signal, which makes the neuron flush its buffer in the off chance that it has a recursively sent signal in its inbox. After flushing its buffer, the neuron waits for the exoself to send it the reset signal, at which point the neuron, now fully refreshed after the flush_buffer/0, outputs a default forward signal to its recursively connected elements, if any, and then drops back into the main loop.

	fanout([Pid|Pids],Msg)->
		Pid ! Msg,
		fanout(Pids,Msg);
	fanout([],_Msg)->
		true.
%The fanout/2 function fans out th Msg to all the PIds in its list.

	flush_buffer()->
		receive 
			_ ->
				flush_buffer()
		after 0 ->
			done
	end.
%The flush_buffer/0 cleans out the element's inbox.
perturb_IPIdPs(Spread,[])->[];
perturb_IPIdPs(Spread,Input_PIdPs)->
	Tot_Weights=lists:sum([length(WeightsP) || {_Input_PId,WeightsP}<-Input_PIdPs]),
	MP = 1/math:sqrt(Tot_Weights),
	perturb_IPIdPs(Spread,MP,Input_PIdPs,[]).
perturb_IPIdPs(Spread,MP,[{Input_PId,WeightsP}|Input_PIdPs],Acc)->
	%MP = 1/math:sqrt(length(WeightsP)),
	U_WeightsP = perturb_weightsP(Spread,MP,WeightsP,[]),
	perturb_IPIdPs(Spread,MP,Input_PIdPs,[{Input_PId,U_WeightsP}|Acc]);
perturb_IPIdPs(_Spread,_MP,[],Acc)->
	lists:reverse(Acc).
%The perturb_IPIdPs/1 function calculates the probability with which each neuron in the Input_PIdPs is chosen to be perturbed. The probablity is based on the total number of weights in the Input_PIdPs list, with the actual mutation probablity equating to the inverse of square root of total number of weights. The perturb_IPIdPs/3 function goes through each weights block and calls the perturb_weights/3 to perturb the weights.

	perturb_weightsP(Spread,MP,[{W,LPs}|Weights],Acc)->
		%io:format("Spread:~p~n",[Spread]),
		U_W = case random:uniform() < MP of
			true->
				DW = (random:uniform()-0.5)*Spread,
				%io:format("self:~p DW:~p~n",[DW,self()]),
				sat(W+DW,-?SAT_LIMIT,?SAT_LIMIT);
			false ->
				W
		end,
		perturb_weightsP(Spread,MP,Weights,[{U_W,LPs}|Acc]);
	perturb_weightsP(_Spread,_MP,[],Acc)->
		lists:reverse(Acc).
%The perturb_weights/3 function is the function that actually goes through each weight block, and perturbs each weight with a probablity of MP. If the weight is chosen to be perturbed, the perturbation intensity is chosen uniformly between -Spread and Spread.

		sat(Val,Limit)->
			sat(Val,-abs(Limit),abs(Limit)).
		sat(Val,Min,Max)->
			if
				Val < Min -> Min;
				Val > Max -> Max;
				true -> Val
			end.
%sat/3 function simply ensures that the Val is neither less than min or greater than max.
