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

-record(sensor,{id,name,type,cx_id,scape,vl,fanout_ids=[],generation,format,parameters,gt_parameters,phys_rep,vis_rep,pre_f,post_f}). 
-record(actuator,{id,name,type,cx_id,scape,vl,fanin_ids=[],generation,format,parameters,gt_parameters,phys_rep,vis_rep,pre_f,post_f}).
-record(neuron, {id, generation, cx_id, pre_processor,signal_integrator,af, post_processor, pf, aggr_f, input_idps=[], input_idps_modulation=[], output_ids=[], ro_ids=[]}).
-record(cortex, {id, agent_id, neuron_ids=[], sensor_ids=[], actuator_ids=[]}).
-record(substrate, {id, agent_id, densities, linkform, plasticity=none, cpp_ids=[],cep_ids=[]}). 
-record(agent,{id, encoding_type, generation, population_id, specie_id, cx_id, fingerprint, constraint, evo_hist=[], fitness=0, innovation_factor=0, pattern=[], tuning_selection_f, annealing_parameter, tuning_duration_f, perturbation_range, mutation_operators,tot_topological_mutations_f,heredity_type,substrate_id}).
-record(specie,{id, population_id, fingerprint, constraint, agent_ids=[], dead_pool=[], champion_ids=[], fitness, innovation_factor={0,0},stats=[]}).
-record(trace,{stats=[],tot_evaluations=0,step_size=500}).
-record(population,{id, polis_id, specie_ids=[], morphologies=[], innovation_factor, evo_alg_f, fitness_postprocessor_f, selection_f, trace=#trace{}}).
-record(stat,{morphology,specie_id,avg_neurons,std_neurons,avg_fitness,std_fitness,max_fitness,min_fitness,gentest_fitness,avg_diversity,evaluations,time_stamp}).
-record(topology_summary,{type,tot_neurons,tot_n_ils,tot_n_ols,tot_n_ros,af_distribution}).
-record(avatar,{id,sector,morphology,energy=0,health=0,food=0, age=0, kills=0, loc, direction, r, mass, objects=[], state,actuators,sensors}).
-record(object,{id,sector,type,color,loc,pivot,parameters=[]}).

-record(constraint,{
	morphology=xor_mimic, %xor_mimic 
	connection_architecture = recurrent, %recurrent|feedforward 
	neural_afs=[
		tanh,
		cos,
		gaussian
		%sqrt
		%absolute
	], %[tanh,cos,gaussian,absolute,sin,sqrt,sigmoid],
	neural_pfns=[none], %[none,hebbian_w,hebbian,ojas_w,ojas,self_modulationV1,self_modulationV2,self_modulationV2,self_modulationV3,self_modulationV4,self_modulationV5,self_modulationV6,neuromodulation]
	substrate_plasticities=[none],
	substrate_linkforms = [l2l_feedforward],%[l2l_feedfrward,jordan_recurrent,fully_connected]
	neural_aggr_fs=[dot_product], %[dot_product, mult_product, diff]
	tuning_selection_fs=[dynamic_random], %[all,all_random, recent,recent_random, lastgen,lastgen_random]
	tuning_duration_f={wsize_proportional,0.5}, %[{const,20},{nsize_proportional,0.5},{wsize_proportional,0.5}...]
	annealing_parameters=[0.5], %[1,0.9]
	perturbation_ranges=[1], %[0.5,1,2,3...]
	agent_encoding_types= [neural], %[neural,substrate]
	heredity_types = [darwinian], %[darwinian,lamarckian]
	mutation_operators= [
		%{mutate_weights,1},
		{add_bias,1}, 
		%{remove_bias,1}, 
%		{mutate_af,1}, 
		{add_outlink,1}, 
		{add_inlink,1}, 
		{add_neuron,1}, 
		{outsplice,1},
		{add_sensorlink,1},
		{add_actuatorlink,1}
%		{add_sensor,1}, 
%		{add_actuator,1},
%		{mutate_plasticity_parameters,1},
%		{add_cpp,1},
%		{add_cep,1}
	], %[{mutate_weights,1}, {add_bias,1}, {remove_bias,1}, {mutate_af,1}, {add_outlink,1}, {remove_outLink,1}, {add_inlink,1}, {remove_inlink,1}, {add_sensorlink,1}, {add_actuatorlink,1}, {add_neuron,1}, {remove_neuron,1}, {outsplice,1}, {insplice,1}, {add_sensor,1}, {remove_sensor,1}, {add_actuator,1}, {remove_actuator,1},{mutate_plasticity_parameters,1}]
	tot_topological_mutations_fs = [{ncount_exponential,0.5}], %[{ncount_exponential,0.5},{ncount_linear,1}]
	population_evo_alg_f=generational, %[generational, steady_state]
	population_fitness_postprocessor_f=size_proportional, %[none,nsize_proportional]
	population_selection_f=competition %[competition,top3]
}).
-record(experiment,{
	id,
	backup_flag = true,
	pm_parameters,
	init_constraints,
	progress_flag=in_progress,
	trace_acc=[],
	run_index=1,
	tot_runs=10,
	notes,
	started={date(),time()},
	completed,
	interruptions=[]
}).

-record(pmp,{
	op_mode=gt,
	population_id=test,
	survival_percentage=0.5,
	specie_size_limit=10,
	init_specie_size=10,
	polis_id = mathema,
	generation_limit = 100,
	evaluations_limit = 100000,
	fitness_goal = inf,
	benchmarker_pid
}).
%%% sensor:
%id= {{-1::LayerCoordinate, float()::Unique_Id()}, sensor}
%name= atom()
%cx_id= cortex.id
%scape= {private|public, atom()::ScapeName}
%vl= int()
%fanout_ids= [neuron.id...]
%	fanout_idps= [{all,neuron.id}|{single,Index,neuron.Id}...]
%generation=int()
%format= {no_geo|geo,[int()::Resolution...]}
%parameters= [any()...]
%phys_rep= [any()...]
%vis_rep= [any()...]
%pre_f= atom()::FunctionName
%post_f= atom()::FunctionName

%%%actuator:
%id= {{1::LayerCoordinate,generate_UniqueId()},actuator}
%name= atom()
%cx_id= cortex.id
%scape= {private|public, atom()::ScapeName}
%vl= int()
%fanout_ids= [neuron.id...]
%generation=int()
%format= {no_geo|geo,[int()::Resolution...]}
%parameters= [any()...]
%phys_rep= [any()...]
%vis_rep= [any()...]
%pre_f= atom()::FunctionName
%post_f= atom()::FunctionName

%%%neuron:
%id= {{float()::LayerCoordinate, float()::Unique_Id},neuron}
%generation= int()
%cx_id= cortex.id
%af= atom()::FunctionName
%pf= {atom()::FunctionName,any()::ParameterList}
%aggr_f= atom()::FunctionName
%input_idps= [{Input_Id,WeightsP},{neuron.id|sensor.id,[{float()::Weight,any()::ParameterList}...]}...]
%output_ids= [neuron.id|actuator.id...]
%ro_ids= [neuron.id...]

%%%cortex:
%id= {{origin, float()::Unique_Id()},cortex}
%agent_id= agent.id
%neuron_ids= [neuron.id...]
%sensor_ids= [sensor.id...]
%actuator_ids= [actuator.id...]

%%%agent:
%id= {float()::Unique_Id(),agent}
%encoding_type= atom()::neural|substrate
%generation= int()
%population_id= population.id
%specie_id= specie.id
%cx_id= cortex.id
%fingerprint= fingerprint()
%constraint= constraint()
%evo_hist= [OperatorAppllied...]
%	{atom()::MO_Name, ElementA.id, ElementB.id, ElementC.id}
%	{atom()::MO_Name, ElementA.id, ElementB.id}
%	{atom()::MO_Name, ElementA.id}
%fitness= float()
%innovation_factor= int()
%pattern= [{float()::LayerCoordinate, N_Ids}...]
%tuning_selection_f= atom()::FunctionName
%annealing_parameter= float()
%tuning_duration_f= {atom()::FunctionName,any()::Parameter}
%perturbation_range= float()
%mutation_operators= [{atom()::FunctionName,float()}...]
%tot_topological_mutations_f= {atom()::FunctionName,float()}

%%%specie:
%id= atom()|{float()::Unique_Id,specie}
%population_id= population.id
%fingerprint= fingerprint()
%constraint= constraint()
%agent_ids= [agent.id...]
%dead_pool= [agent.id...]
%champion_ids= [agent.id..]
%fitness= float()
%innovation_factor= int()

%%%population:
%id= atom()|{float()::Unique_Id,population}
%polis_id= polis.id
%specie_ids= [specie.id...]
%morphologies= [atom()::Morphology_Name...]
%innovation_factor= int()
%evo_alg_f= atom()::FunctionName
%fitness_postprocessor_f= atom()::FunctionName
%selection_f= atom()::FunctionName

%%%fingerprint:
%generalized_sensors= [sensor()::init...]
%	sensor.id = undefined
%	sensor.cx_id = undefined
%	sensor.fanout_ids = []
%generlized_actuators= [actuator()::init...]
%	actuator.id = undefined
%	actuator.cx_id = undefined
%	actuator.fanin_ids = []
%generalized_pattern= [{float()::LayerCoordinate,int()::TotNeurons}...]
%generalized_evohist= [GeneralizedOperatorApplied...]
%	{atom()::MO_Name,{float()::ElementA_LayerCoordinate,atom()::ElementA_Type},{ElementB_LayerCoordinate,ElementB_Type},{ElementC_LayerCoordinate,ElementC_Type}},
%	{atom()::MO_Name,{float()::ElementA_LayerCoordinate,atom()::ElementA_Type},{ElementB_LayerCoordinate,ElementB_Type}},
%	{atom()::MO_Name,{float()::ElementA_LayerCoordinate,atom()::ElementA_Type}},
%	{atom()::MO_Name},

%%%constraint:
%morphology=xor_mimic, %xor_mimic 
%connection_architecture = recurrent, %recurrent|feedforward 
%neural_afs=[tanh,cos,gaussian,absolute] %[tanh,cos,gaussian,absolute,sin,sqrt,sigmoid],
%neural_pfs=[none], %[none,hebbian,neuro_modulated]
%neural_aggr_fs=[dot_product], %[dot_product, mult_product, diff]
%tuning_selection_fs=[all], %[all,all_random, recent, recent_random, lastgen,lastgen_random]
%tuning_duration_f={const,any()::Parameter}, %[{const,any()::Parameter},{size_proportional,any()::Parameter}...]
%annealing_parameters=[0], %[0,1,2,0.5...]
%perturbation_ranges=[1], %[0.5,1,2,3...]
%agent_encoding_types= [neural], %[neural,substrate]
%mutation_operators= [{atom()::FunctionName,float()}...]
%tot_topological_mutations_fs = [{ncount_exponential,0.5}], %[{ncount_exponential,0.5},{ncount_linear,1}]
%population_evo_alg_f=generational_default, %[generational, steady_state]
%population_fitness_postprocessor_f=size_proportional, %[none,size_proportional]
%population_selection_f=competition %[competition,top3]

%%%polis
%id= atom()|float()|{float()::Unique_Id,polis}|{atom()::PolisName,polis}

%%%scape
%id= atom()|float()|{float()::Unique_Id,scape}|{atom()::ScapeName,scape}

%export_population(Population_Id)->
	%create an ets file
	%backup population to ets
	%give it a timestamp
	%save to disk
%	ok.
	
%export_champion(Champion_Id)->
	%create ets file.
	%backup champion to file.
	%give it a timestamp
	%save to disk
%	ok.
	
%updated_trace(Experiment_Id)->
	%stores more information
	%comes with the population where agents don't get deleted
%	ok.
	
%updated_population(Population_Id)->
%	ok.
	%agents no longer get deleted, only noted if active or inactive
	%agents not who their parents were
	%
