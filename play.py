#!/usr/bin/env python

"""
Actions in the kungfu domain:

0: no action
1: jump
2: move right
3: move left
4: duck
5: duck turn right
6: duck turn left 

kicks: an agent cannot do multiple attacks in a row (it needs to move or jump first)

7: right kick -- cannot be repeated
8: left kick -- cannot be repeated
9: low kick left
10: punch right 
11: punch left
12: duck and punch right
13: duck and punch left

"""


""" Trains an agent with (stochastic) Policy Gradients on Kung Fu. Uses OpenAI Gym. """
import numpy as np
#import _pickle as pickle
import gym

#from chainer import cuda
import time, threading
import sys
import math
import copy


# for visualization
#from scipy.misc import toimage
from scipy.stats import entropy

# import our dqn agent
from simple_dqn_agent import SimpleDQN

# some common util functions
from mdl_utils import *
from buffer_queue import QueueBuffer
from steps_queue import Buffer


print("Number of arguments: %i" % len(sys.argv))

trial_num = int(sys.argv[1])
max_num_game_steps = int(sys.argv[2])
output_dir = sys.argv[3]
exp_tag = sys.argv[4]

# train mode -- either baseline, regular, or subtask
mdl_train_mode = sys.argv[5]#"baseline"
backtracking = sys.argv[6]#"normal" "random" "non-mistake", "neg", "alwaysneg"
prob = sys.argv[7]
mdl_prob = float(prob) # with what probability do we go into subtask mode
domain = sys.argv[8]
env = gym.make(domain)
env.test_function()
# feature type
feature_type = FEATURE_CHANGE_BUFFER

# grayscale or not
grayscale = False

# what to print and whether to render
print_out = 0
render = 0
 # whether to render


# create buffer for last k images
image_buffer_queue = QueueBuffer(buffer_image_k)
num_of_subs = 24

total_mistakes = 0

downsample_factor = 2	

# hack
if domain == "KungFuMaster-v0":
	#img_width = 159
	#img_height = 67
	downsample_factor = 1
elif domain == "Boxing-v0":
	#img_width = 40
	#img_height =40
	downsample_factor = 2

# used for some games
train_reward_multiplyer = 1
if domain == "Boxing-v0":
	train_reward_multiplyer = 100
elif domain == "IceHockey-v0":
	train_reward_multiplyer = 500
elif domain == "Tennis-v0":
	train_reward_multiplyer = 300
elif domain == "DoubleDunk-v0":
	train_reward_multiplyer = 300
elif domain == "Breakout-v0":
	train_reward_multiplyer = 300
elif domain == "FishingDerby-v0":
	train_reward_multiplyer = 5000
elif domain == "Frostbite-v0":
	train_reward_multiplyer = 10
elif domain == "Phoenix-v0":
	train_reward_multiplyer = 50

# get the environment's image size
img_start = env.reset()


#figure out how many actions
num_a = 0
for i in range(0,20) :
	if env.action_space.contains(i) :
		num_a += 1
	else :
		break
	   
print("Found %i actions in the domain" % num_a)

# input image params
n_color_channels = 3
if grayscale:
	n_color_channels = 1

# find out what the input size is for this domain
img_processed = prepro(img_start,domain,downsample_factor,grayscale)


print("Original image size: "+(str(len(img_start)))+" by "+str(len(img_start[0])))
img_width = len(img_processed) 
img_height = len(img_processed[0]) 

print("processed img size: "+str(img_width) +" x "+str(img_height))



# hyperparameters for NN
A = num_a # 2, 3 for no-ops
H = 200 # number of hidden layer neurons
gamma = 0.95 # discount factor for reward
learning_rate = 1e-3
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
D = int(img_width * img_height  * n_color_channels )  # input dimensionality

if feature_type == FEATURE_CHANGE_PLUS_RAW:
	D = D * 2


print("Input dimensionality: " + str(D))

# training parameters
update_freq = int(sys.argv[9]) # every how many episodes to do a param update?
device = 1

# constants for greedy exploration
greedy_e_epsilon = 0.1 # probability of taking a random action


# the weights of the NN is initialized to random values using this seed
# different random seeds can lead to quite different initial behabior of the agent
random_seed = trial_num

# save filename
curve_file_name = exp_tag+"_"+str(random_seed)+".txt"
episode_file_name = "episode" + exp_tag+"_"+str(random_seed)+".txt"

  
# create NN model
dqn_agent = SimpleDQN(A,D,H,learning_rate,gamma,decay_rate,greedy_e_epsilon,random_seed)

print(env.action_space)
observation = env.reset()
prev_x = None # used in computing the difference frame
prev_I = None # previous image
running_reward = None
reward_sum = 0
episode_number = 0
game_step_number = 0
game_step_total = 0

# used for mistake-detection for kungfu
life_before = 1.0
life_after = 1.0
negative_life_reward = -100
life_decreased = False
life_decreased_counter = 0
life_decreased_threshold = 2

# used for mistake detection for frostbite, etc.
num_life_pixels_before = -1
num_life_pixels_now = -1

# currently only used for UpDown
num_lives_before = 3
num_lives_after = 3

# used for mistake detection for frostbite, updown, etc.
lost_life = False

# used for Atlantis-v4
has_mid_gun = True

# after each subtask, we add 0 or 1 depending on whether the mistake occured again
mistake_detected_indicator_array = []

# some learning progress params
lp_window_short = 20
lp_window_long = 40
lp_progress_values = []
gs_per_ep = []
# counts how many mistakes have been detected in the current subtask
mistake_subtask_counter = 0

# where we store the curve
ep_reward_list = []

last_action = 0
# used to hold queue of states
mdl_subtask_k = float(sys.argv[10]) # how many subtasks to play after making a mistake
mdl_subtask_counter = 0
mdl_gamesteps_k = int(sys.argv[11])
mdl_buffer_size = int(sys.argv[12]) # when choosing a subtask, pick 1 of N previous states
mdl_min_back = int(sys.argv[13])
#40 10

if domain == "UpNDown-v0":
	mdl_gamesteps_k = 400
	mdl_buffer_size = 400 # when choosing a subtask, pick 1 of N previous states
	mdl_min_back = 100
elif domain == "Frostbite-v0":
	mdl_gamesteps_k = 300
	mdl_buffer_size = 5
	mdl_min_back = 70
elif domain == "Atlantis-v0":
	mdl_buffer_size = 100 # go back slight longer in the past
	mdl_min_back = 70
	

state_buffer_queue = QueueBuffer(mdl_buffer_size * 3)
if backtracking != "random":
	online_buffer_queue = QueueBuffer(500) # SIZEOFBUFFER

if backtracking == "random" or backtracking == "normal" or backtracking=="sub":
	step_queue = Buffer(update_freq)

number_of_mistakes = 0

paused_state = None

subtask_to_normal = False

def goBack():
	# print ("step back")
	r_index = 0
	num_stored_states = len(state_buffer_queue.get_elements())
	if num_stored_states > 0:
		r_index = np.random.randint(0,num_stored_states)
	#env.restore_state(state_buffer[r_index]) # randomly go to a saved state
	if state_buffer_queue.current_size() > 0:
		env.restore_state(state_buffer_queue.get_elements()[num_stored_states - 1])

while True:
	t0  = time.time()
  
	if render: 
		t  = time.time()
		env.render()
		#print((time.time()-t)*1000, ' ms, @rendering')

	t  = time.time()
  
	# get the life (0.0 to 1.0)
	if domain == "KungFuMaster-v0":
		life_after = compute_life_kungfu(observation)
		if life_after < life_before:
			life_decreased = True
			life_decreased_counter += 1
			if backtracking == "neg":
				reward_sum -= 30
		#print "Life lost!
		else :
			life_decreased = False
			life_decreased_counter = 0
		life_before = life_after
	elif domain == "UpNDown-v0":
		num_lives_after = compute_lives_updown(observation)
		if num_lives_after < num_lives_before:
			lost_life = True
		else :
			lost_life = False
		num_lives_before = num_lives_after
	elif domain == "Frostbite-v0":
		num_life_pixels_now =get_num_live_text_pixels_frostbite(observation)
		
		if num_life_pixels_before != -1 and num_life_pixels_before != num_life_pixels_now:
			lost_life = True
			# print("Lost life!")
			num_life_pixels_before = num_life_pixels_now
		else:
			num_life_pixels_before = num_life_pixels_now
  
  
	# preprocess the observation, set input to network to be difference image
	cur_I = prepro(observation,domain,downsample_factor,grayscale)
	
	# store image
	

	cur_x = cur_I.astype(np.float).ravel()
  
	x = cur_x
	
	#print(get_rr_numplanepixels(observation))
	
	if feature_type == FEATURE_CHANGE:
		# feature 1: difference from last frame
		x = cur_x - prev_x if prev_x is not None else np.zeros(D)
		prev_x = cur_x
		prev_I = cur_I

		x = x / 255
	elif feature_type == FEATURE_CHANGE_PLUS_RAW:
		change_x = cur_x - prev_x if prev_x is not None else np.zeros(int(D/2))
		
		prev_x = cur_x
		prev_I = cur_I
		
		combined_x = np.append(change_x,cur_x).astype(np.float).ravel()

		x = combined_x
		x = x / 255
	elif feature_type == FEATURE_CHANGE_BUFFER:
		# difference from buffer avg
		image_buffer_queue.add_element(cur_I)
		avg_I = image_average(image_buffer_queue.get_elements(),grayscale)
		
 
		avg_x = avg_I.astype(np.float).ravel()
		x = cur_x - avg_x if avg_x is not None else np.zeros(D)
		x = x / 255
	
	
	# forward the policy network and sample an action from the returned probability
	t  = time.time()
  
	# process input and choose action with exploration
	action = dqn_agent.process_step(x,True)

	observation, reward, done, info = env.step(action)
	game_step_number = game_step_number + 1	  
	game_step_total += 1 
	
	#if mdl_train_mode == "subtask":
	#	input("Press enter to continue")
  
	# save last action
	last_action = action
  
	# apply modifier
	if reward != 0:
		reward = reward * train_reward_multiplyer
		
	#if reward != 0:
	#	print(reward)
	
	# check if reward total is 0...this results in a problem
	if reward == 0: reward = 1
  	
	# feed reward to agent
	dqn_agent.give_reward(reward)

	# here we do not using shaping reward as this is just used for reporting game score
	reward_sum += reward 
		
	# store state

	# look for mistake
	mdl_mistake_detected = False
	
	# for kungfu
	if domain == "Atlantis-v4":
		# check if we have gun
		detect_gun_result = has_middle_gun(observation, "middle") # "middle" "left" "right"
		if has_mid_gun and detect_gun_result == -1:
			# print("Lost gun!")
			has_mid_gun = False
			mdl_mistake_detected = True
		elif has_mid_gun == False and detect_gun_result == 1:
			has_mid_gun = True

		if backtracking == "neg" and mdl_mistake_detected:
			reward_sum -= 4000

	elif domain == "KungFuMaster-v0":
		if life_decreased and life_decreased_counter > life_decreased_threshold:
			mdl_mistake_detected = True
			if backtracking == "neg":
				reward_sum -= 100
			# if mdl_train_mode == "regular": print("Life decreased more than %f frames in a row" % (life_decreased_counter))
	elif domain =="Boxing-v0":
		if reward == -2 * train_reward_multiplyer:
			mdl_mistake_detected = True
	
	if backtracking == "random":
		if game_step_total % 50 == 0:
			state_buffer_queue.add_element(env.clone_state())
	else:
		online_buffer_queue.add_element(env.clone_state()) 

	if mdl_mistake_detected:
		# print ("mistaked")
		if backtracking != "random":
			r_index = np.random.randint(mdl_min_back,mdl_buffer_size)			
			if r_index < online_buffer_queue.current_size():
				state_buffer_queue.add_element(online_buffer_queue.get_elements()[online_buffer_queue.current_size() - r_index])
			else:
				mdl_mistake_detected = False
		total_mistakes += 1
		# time.sleep(1)
		if backtracking == "special" and np.random.uniform() < mdl_prob:
			goBack()
	
	# handle train mode
	if mdl_train_mode == "regular" and backtracking == "sub" and mdl_mistake_detected == True and np.random.uniform() < mdl_prob:
		# mdl_train_mode = "subtask"
		mdl_subtask_counter = 0
		done = True
		paused_state = env.clone_state()

	elif mdl_train_mode == "subtask":
		# check that we are only in the subtask for a given # of gamesteps
		if game_step_number >= mdl_gamesteps_k:
			# print("Subtask finished after %f train steps, subtassk: %f"% (game_step_number, mdl_subtask_counter) )
			done = True

		# we can be done either if the agent dies or if the # of steps in the subtask is over the threshold
		if done:
			print("done subtask number: " + str(mdl_subtask_counter) + " " + str(step_queue.get_steps()))
			mdl_subtask_counter += 1
			mistake_detected_indicator_array.append(mistake_subtask_counter)

	
	if done: # an episode finished
		#gs_per_ep.append(game_step_number)
		# print("Episode " + str(episode_number) + " finished after "+str(game_step_number)+" steps and " + str(reward_sum)+" train reward.")
		
		
		if mdl_train_mode != "subtask" and (backtracking == "normal" or backtracking == "random" or backtracking == "sub"):
			step_queue.add_element(game_step_number)
			episode_number += 1
			print("done " + mdl_train_mode + " " + str(episode_number))

		if mdl_train_mode == "subtask" and (backtracking == "normal" or backtracking == "random" or backtracking == "sub") and (mdl_subtask_counter * mdl_gamesteps_k >= (step_queue.get_steps() / update_freq) * mdl_subtask_k):
			mdl_train_mode = "regular"
			print("switchin to regular")
			episode_number += 1
			if backtracking == "sub":
				subtask_to_normal = True

		game_step_number = 0
		
		# call finish episode function
		dqn_agent.finish_episode()
		
		# reset reward sum for episode
		reward_sum = 0
		
		# perform rmsprop parameter update every batch_size episodes
		if ((episode_number % update_freq == 0) or episode_number == 1) and mdl_train_mode != "subtask": #update_freq used to be batch_size	
			updated_params = False
			if (episode_number % update_freq == 0):
				print("Updating Model Parameters... episode: " + str(episode_number))
				dqn_agent.update_parameters()
				updated_params = True
			
			eval_reward = evaluate_policy_kungfu(False,env,dqn_agent,D,domain,downsample_factor,feature_type,grayscale)

			running_reward = eval_reward if ((running_reward is None) or (running_reward < 0)) else running_reward * 0.9 + eval_reward * 0.1
					
			# print('ep %f: resetting env. episode reward total was %f. running mean: %f' % (episode_number, eval_reward, running_reward))
    
			# hack to make first evaluation appear as if done at 0 
			if updated_params:
				ep_reward_list.append([episode_number,game_step_total,eval_reward,running_reward])
			else:
				ep_reward_list.append([episode_number-1,0,eval_reward,running_reward])
			
			if print_out:
				for row in ep_reward_list:
					print("%f,%f,%f,%f" % (row[0],row[1],row[2],row[3]))

			save_curve(curve_file_name,ep_reward_list)
			reward_sum = 0
			
			if episode_number != 1:
				if (backtracking == "normal" or backtracking == "random") and mdl_train_mode != "baseline":
					mdl_train_mode = "subtask"
					# print("switchin to subtask")
					mdl_subtask_counter = 0
					done = True
				
		if backtracking == "sub" and subtask_to_normal:
			env.restore_state(paused_state)
			subtask_to_normal = False
		# how to init the next episode
		elif mdl_train_mode == "baseline" or mdl_train_mode == "regular":
			# print("Reseting env")
			observation = env.reset() # reset env

		elif mdl_train_mode == "subtask":
		
			# randomly decide how far back to rewind
			
			r_index = 0
			if backtracking == "random" or backtracking == "normal":
				num_stored_states = len(state_buffer_queue.get_elements())
				if num_stored_states > 0:
					r_index = np.random.randint(0,num_stored_states)
			else:
				continue

			env.restore_state(state_buffer_queue.get_elements()[r_index]) # randomly go to a saved state
			  
		# reset previous x
		prev_x = None    
		num_life_pixels_before = -1

print (total_mistakes)



