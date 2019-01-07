# Reinforcement Learning (Cartpole-v1)

Running Steps:

Agent training: 
  - git clone my repository.
  - find the document called main.py.
  - open any python IDE, such as pycharm, spyder, etc.
  - import the necessary python library, including numpy, pandas, and gym. Particularly, import my_lunarlander.py file. All of them should already be imported at the beginning of main.py file.
  - In the python file of cartpole_v1.py, there is the train and run_trained_model functions, used for Cartpole-v1 agent training and trained agent testing.
  - Excpet importing the necessary libraries of  numpy, tensorflow and collections, my_lunarlander.py needs to import another python file called DDQN, which implements the algorithms of normal DQN, Double DQN and Dueling DQN.
  - With imported numpy and tensorflow library, DDQN contains the init function, record_step function, select_action function and the learn function, which implements the core algorithms. 
  - DDQN.py file contains another python file called builtNN, which implements and constructs the Neural Network using tensorflow library. 
  - Conclusively, to train the Cartpole-v1 model, go to the main.py file in the python IDE and click run.
  - Some episode reward and averaged reward will automatically print in the console, to see the agent landing, you need to uncomment env.render in my code.


Test trained Agent:
  - To test a trained agent, simply uncomment in main.py a line code of "cartpole.run_trained_model()".
  - After you train the agent, a session should be saved in the same directory of main.py, which will get reloaded during the testing.
  - Agent test is unecessary to run after training process. Since the session file was saved during training, you can comment out the training steps and only run the trained model.
  - When the test is done, the episode number and reward will be printed in the console.

Run Advanced Algorithms:
  - Both Double DQN and Dueling DQN have already been implemented in DDQN.py. 
  - Each algorithm can be used by turing on its corresponding flag in main.py.
  - For example, to run Double DQN, you can simply change DOUBLE_DQN to True and run main.py.
