DXNN MK2 Erlang,
By Gene Sher
http://DXNN.org
http://DXNNResearch.com
CorticalComputer@gmail.com

Brief documentation for this package is included in this README file.  

-------------
1. LICENSE
-------------

Copyright (C) 2009 by Gene Sher, DXNN Research Group, CorticalComputer@gmail.com
All rights reserved.

This code is licensed under the version 3 of the GNU General Public License. Please see the LICENSE file that accompanies this project for the terms of use.

---------------------
2. USAGE and SUPPORT
---------------------

I hope that this software will be a useful starting point for your own
explorations in the creation of Computational Intelligence. The software is provided 
as is; however, I will do my best to maintain it and accommodate
suggestions. If you want to be notified of future releases of the
software or have questions, comments, bug reports or suggestions, send
an email to CorticalComputer@gmail.com, or request a user account at www.DXNNResearch.com

Alternatively, you may post your questions on dxnn.org or dxnnresearch.com

The following explains how to use DXNN.

INTRO
-----
DXNN is a fully distributed Topology and Weight Evolving Artificial Neural Network system created and invented by Gene Sher. 
Originaly introduced in the publication preprint of 2010 available on arXiv:1011.6022v3
The current state of the project is composed of the following:

Cortex synchronizes Neurons, Sensors, and Actuators. A population monitor controls a population of agents, it is the population_monitor process that spawns agents, waits for them to be evaluated, and then performs the selection, replication, and variation (application of various mutation operators). Furthermore, due to this being a memetic algorithm based system (although can be changed to genetic, by simply switching max_attempts variable to 1), each NN based agent system comes with an exoself process, which performs the synaptic weight tuning.

Scapes are simulations, not necessarily of 3d environments. It is through scapes that problems are presented to the agents, and it is the agent's morphology which defines the agent's sensors and actuators, and thus defines which scapes it can interface with. The system itself, the mnesia database, and the top most system, is called polis (It's Greek for city state).

This version is 1.0, and it does not have a lot of comments (or any), since I built it primarily for myself. But v2.0, which I will release very soon after this, is better and cleaner implemented. Has more functionality, and is fully documented. It is also the version built inside my upcoming book: Neuroevolution Through Erlang, to be released towards the end of this year (just recently submitted my manuscript to my editor). But if you want to give this version a try (feel free to send me an email if you can't get it to work), then by all means go for it. You might find strange comments somewhere in the source code, since I did not take out any notes I wrote for myself.

STARTING POINT
--------------
From inside Erlang.
1. %%%%Compilation%%%%
	make:all().
2. %%%%Initialize All Databases%%%%
	First create a folder called “benchmarks”, the system expects it to exist, and writes files to it when performing benchmarks.
	polis:create(). % This creates the database
3. %%%%Start The Polis Databases%%%%
	polis:start(). % This starts the polis process, the whole thing, the infrastructure (it runs the scapes...)
4. At this point you can summon NN based agents or populations of agents, construct Sensors and Actuators and provide them to the NNs... This section will be expanded in future additions.

To set the population and agents to the preferred sensors and actuators, modify the INIT_CONSTRAINTS (in population_monitor module), and use the particular morphology you want (check the morphology module, different morphologies are for different problems), and then execute population_monitor:start(), which will create the population of size decided by you of agents using the specified morphology and thus the sensors and actuators. If you want the agents to discover and explore the available sensors and actuators (perform feature selection in a sense) then modify modular_constructor, ensuring that you use the get_InitSensors() and get_InitActuators() function, instead of the get_Sensors() and get_Actuators() function used within the module. The get_Init.. starts the population off with the NN based agents using just a single sensor and actuator, exploring other available sensors and actuators within the morphology as they evolve. You can add new sensors and actuators by specifying those sensors and actuators in the morphology module, and then creating those functions in the sensors and actuators modules. New mutation operators, activation functions... all can be added within the records.hrl, as long as those functions are realized/implemented in their respective modules so that they can be executed when called upon.
