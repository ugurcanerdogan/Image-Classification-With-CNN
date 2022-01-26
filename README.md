# Image-Classification-With-CNN
Image Classification Using Custom CNN with PyTorch

## Dataset

Dataset Link: <a href="https://drive.google.com/file/d/1VKmG8Wg2TFCsPFkJRIcdYDpdwOt5FuqQ/view">Link<a/>

## Requirements
	Modules:
		numpy
		pytorch
		torchvision
		math
		matplotlib
	Python Version 3.8.0

## Running
	- Should run program as "python main.py"
	
	- By default program trains the model with residual connections applied, dropout=0.45, 
	batch size=64, learning rate=0.0005 and saves it on same location with "main.py" after
	training
	
	- Should pass "False" as a boolean parameter to function call on line 434 and also 
	uncomment line 446 and 447 for testing trained model(path is given as 3rd parameter 
	on same function call) according to defined hyper parameters on lines 417,418,419; 
	besides, by default program trains model with applied residual connection and 
	dropouts as "0.45".
	
	- To change trained CNN, should change line 295:
		-MyNet() : no residual, no dropout
		-DropMyNet(): no residual, with dropout
		-ResidualMyNet(): with residual, no dropout
		-DropResidualMyNet(): with residual, with dropout
