import numpy as np
from scipy.misc import toimage
#from scipy.misc import imread

from buffer_queue import QueueBuffer


FEATURE_CHANGE = 1
FEATURE_CHANGE_BUFFER = 2
FEATURE_CHANGE_PLUS_RAW = 3

buffer_image_k = 5

# compute average of images
def image_average(images,make_gray):
	
	o = np.zeros([len(images[0]),len(images[0][0]),3])
	if make_gray == True:
		o = np.zeros([len(images[0]),len(images[0][0])])

	#print(o[97][10])
	#print("computing avg...")
	for k in range(0,len(images)):
		o += images[k]
		#print(images[k][97][10])
	#print(o[97][10])
	
	
	#for img_k in images:
	#	for i in range(0,len(img_k)):
	#		for j in range(0,len(img_k[0])):
	#			output[i][j][0]=0
	#			output[i][j][1]=0
	#			output[i][j][2]=0
	#output.fill(0)
	#print("compting avg image")
	#for img_k in images:
	#	print("adding image")
	#	print(img_k[97][10])
	#	for i in range(0,len(img_k)):
	#		for j in range(0,len(img_k[0])):
	#			for c in range(0,3):
	#				output[i][j][c]+=img_k[i][j][c]
	#	#output += img_k
	#print(output[97][10])
	
	o = o / len(images)
	#print(o[97][10])
	#
	return o


### Some functions related to Atlantis-v4

# whether or not the middle gun is there
def has_middle_gun(orig_I, type):
	
	# crop where the gun should be
	I = orig_I
	if type == "middle":
		I = orig_I[110:120,72:80]
	elif type == "left":
		I = orig_I[122:130,0:10]
	elif type == "right":
		I = orig_I[110:120,150:160]

	#toimage(I).save("test.png")
	
	# count the # of black pixels
	I[I == [0,0,0]] = 0
	
	num_nonbg = sum(sum(sum(I)))
	
	if num_nonbg == 461: # the gun is there
		return 1
	elif num_nonbg == 0: # nothing is there
		return -1
	else:
		return 0 # we don't know, could be an explosion or a beam

def has_right_gun(orig_I):
	
	# crop where the gun should be
	I = orig_I[110:120,150:160]
	toimage(I).save("test.png")
	
	# count the # of black pixels
	I[I == [0,0,0]] = 0
	
	num_nonbg = sum(sum(sum(I)))
	
	if num_nonbg == 418: # the gun is there
		return 1
	elif num_nonbg == 0: # nothing is there
		return -1
	else:
		return 0 # we don't know, could be an explosion or a beam

def has_left_gun(orig_I):
	
	# crop where the gun should be
	I = orig_I[122:130,0:10]
	toimage(I).save("test.png")
	
	# count the # of black pixels
	I[I == [0,0,0]] = 0
	
	num_nonbg = sum(sum(sum(I)))
	
	if num_nonbg == 370: # the gun is there
		return 1
	elif num_nonbg == 0: # nothing is there
		return -1
	else:
		return 0 # we don't know, could be an explosion or a beam



# preprocess for atalantis-v4
def prepro_atlantis(I,downsample_factor,make_gray):
	
	#has_middle_gun(I)
	
	#toimage(I).save("test_orig.png")
	
	# crop to region of interest
	I = I[10:135,1:160]
	
	# downsample width and height
	if downsample_factor != 1:
		I = I[::downsample_factor,::downsample_factor,:]
	
	#toimage(I).save("test_crop.png")
	
	return I


# preprocess kung fu image
def prepro_kungfu(I):
	#img = toimage(I)
	#img.save("test_raw.png")
 
	# crop for kung fu master: 160 (height) x 167
	I =I[98:165,1:160]
	#toimage(I).save("test_crop.png")
  
	dk = 1
	#I = I[::dk,::dk,3] # downsample by factor of dk and make gray
  
	#toimage(I).save("test_crop_gray.png")

	#return I.astype(np.float).ravel()
	return I
	
def prepro_hockey(I):
	#img = toimage(I)
	#img.save("test_raw.png")
 
	# crop for kung fu master: 160 (height) x 167
	I =I[30:190,30:130]
	#toimage(I).save("test_crop.png")
  
	dk = 1
	#I = I[::dk,::dk,3] # downsample by factor of dk and make gray
  
	#toimage(I).save("test_crop_gray.png")

	#return I.astype(np.float).ravel()
	return I

def get_rr_numplanepixels(I):
	I_copy = I.copy()
	I_roi = I_copy[141:164,1:160]
	
	#I_roi = I_roi[::1,::1,0] # downsample by factor of dk and make gray

	#img = toimage(I_roi)
	#img.save("test_roi.png")
	


	I_roi[I_roi != [232,232,74]] = 0
	I_roi[I_roi == [232,232,74]] = 1
	#I_roi[I_roi == 232] = 1
	
	#img = toimage(I_roi)
	#img.save("test_seg.png")
	
	return sum(sum(sum(I_roi)))

def get_rr_plane_xy(I):
	I_copy = I.copy()
	I_roi =I_copy[141:164,1:160]
	I_roi = I_roi[::1,::1,0] # downsample by factor of dk and make gray

	#img = toimage(I_roi)
	#img.save("test_roi.png")
	
	I_roi[I_roi != 232] = 0
	
	#img = toimage(I_roi)
	#img.save("test_seg.png")
	
	x = (np.average(I_roi,axis=0))
	y = (np.average(I_roi,axis=1))
	
	avg_x = 0
	sum_x = 0
	for p in range(0,len(x)):
		avg_x += p * x[p]
		sum_x += x[p]
		
	if sum_x != 0:
		avg_x = int(avg_x / sum_x)
	else:
		avg_x = 0
	
	avg_y = 0
	sum_y = 0
	for p in range(0,len(y)):
		avg_y += p * y[p]
		sum_y += y[p]
		
		
	if sum_y != 0:
		avg_y = int(avg_y / sum_y)
	else: 
		avg_y = 0
	
	return [int(avg_x),int(avg_y)]
	
		
	
	#print(str(avg_x)+","+str(avg_y))
	
	#x = (np.average(I_roi,axis=1))
	#x = np.average(x)
	#print(x)
	
	
def get_boxer_xy(I):
	
	I =I[30:180,30:130]
	
	I = I[::1,::1,0] # downsample by factor of dk and make gray
	

	I[I!=214]=0
	#print(str(len(I))+","+str(len(I[0])))
	x = (np.average(I,axis=0))
	y = (np.average(I,axis=1))
	#print(x)
	#print(len(x))
	#print(y)
	#print(len(y))
	
	avg_x = 0
	sum_x = 0
	for p in range(0,len(x)):
		avg_x += p * x[p]
		sum_x += x[p]
	avg_x = int(avg_x / sum_x)
	
	
	avg_y = 0
	sum_y = 0
	for p in range(0,len(y)):
		avg_y += p * y[p]
		sum_y += y[p]
	avg_y = int(avg_y / sum_y)
	#print(str(avg_x)+","+str(avg_y))
	
	
	# add 30 to each to compensate for cropping
	return [int(avg_x+30),int(avg_y+30)]
	
	#toimage(I).save("test_crop_gray.png")
	

# preprocess kung fu image
def prepro_boxing(I,downsample_factor,b_xy,x_win,y_win):
	# crop around center
	y_min = b_xy[1]-y_win
	y_max = b_xy[1]+y_win
	if y_min < 1:
		y_min = 1
		y_max = y_min+2*y_win
	elif y_max > len(I):
		y_max = len(I)-1
		y_min = y_max - 2*y_win	
	
	x_min = b_xy[0]-x_win
	x_max = b_xy[0]+x_win
	if x_min < 1:
		x_min = 1
		x_max = x_min+2*x_win
	elif x_max > len(I[0]):
		x_max = len(I[0])-1
		x_min = x_max - 2*x_win	
	
	
	
	I =I[y_min:y_max,x_min:x_max]
 
	#print(str(len(I))+" by "+str(len(I[0])))
	
	#toimage(I).save("boxer.png")
	if downsample_factor > 1:
		dk = downsample_factor
		I = I[::dk,::dk,:] # downsample by factor of dk and make gray

	#return I.astype(np.float).ravel()
	return I

def prepro_riverraid(I,downsample_factor):
	
	y_win_above = 90
	y_win_below = 10
	x_win = 40
	
	
	b_xy = get_rr_plane_xy(I)
	b_xy[1]=150
	
	#img = toimage(I)
	#img.save("test_raw.png")
	
	
	# crop around center
	y_min = b_xy[1]-y_win_above
	y_max = b_xy[1]+y_win_below
	if y_min < 1:
		y_min = 1
		y_max = y_min+2*y_win
	elif y_max > len(I):
		y_max = len(I)-1
		y_min = y_max - 2*y_win	
	
	x_min = b_xy[0]-x_win
	x_max = b_xy[0]+x_win
	if x_min < 1:
		x_min = 1
		x_max = x_min+2*x_win
	elif x_max > len(I[0]):
		x_max = len(I[0])-1
		x_min = x_max - 2*x_win	
	
	
	I_cropped =I[y_min:y_max,x_min:x_max]
	#img = toimage(I_cropped)
	#img.save("test_cropped.png")
	
	
	
	
	#if downsample_factor > 1:
	#	dk = downsample_factor
	#	I = I[::dk,::dk,:] 
	
	return I_cropped

def prepro_default(I,downsample_factor,make_gray):
	#img = toimage(I)
	#img.save("test_raw.png")
 
	# crop for kung fu master: 160 (height) x 167
	#I =I[98:165,1:160]
	#toimage(I).save("test_crop.png")
	#print(len(I[4][1]))
	if downsample_factor > 1:
		dk = downsample_factor
		I = I[::dk,::dk,:] # downsample by factor of dk and make gray
	#print(len(I[4][1]))
	#toimage(I).save("test_crop_gray.png")

	if make_gray:
		I = I[::1,::1,0] # downsample by factor of dk and make gray
	#toimage(I).save("test_gray.png")

	#return I.astype(np.float).ravel()
	return I

# to do: implement change tracker for these pixels to specify mistakes
def prepro_frostbite(I,downsample_factor,make_gray):
	#img = toimage(I)
	#img.save("test_raw.png")
	
	#I_lives =I[21:31,63:70]
	
	#img = toimage(I_lives)
	#img.save("test_raw_lives.png")
	
	#I_lives[I_lives == [132,144,252]] = 1
	#I_lives[I_lives == [45,50,184]] = 0
	
	#print(sum(sum(sum(I_lives))))
	
	return prepro_default(I,downsample_factor,make_gray)

def get_num_live_text_pixels_frostbite(I):
	I_lives =I[21:31,63:70]
	I_lives[I_lives == [132,144,252]] = 1
	I_lives[I_lives == [45,50,184]] = 0
	return sum(sum(sum(I_lives)))

def prepro(I,domain,downsample_factor,make_gray):
	if domain == "KungFuMaster-v0":
		return prepro_kungfu(I)
	elif domain == "Boxing-v0":
		b_xy = get_boxer_xy(I)
		return prepro_boxing(I,downsample_factor,b_xy,40,40)
	elif domain == "Riverraid-v0":
		return prepro_riverraid(I,downsample_factor)
	elif domain == "IceHockey-v0":
		return prepro_hockey(I)
	elif domain == "Frostbite-v0":
		return prepro_frostbite(I,downsample_factor,make_gray)
	elif domain == "Atlantis-v4":
		return prepro_atlantis(I,downsample_factor,make_gray)
	else:
		return prepro_default(I,downsample_factor,make_gray)

def compute_lives_updown(I):
	I_cr = I[194:206,13:40]
	#toimage(I_cr).save("test_crop.png")
	
	I_cr[I_cr == [198,108,58]] = 1
	I_cr[I_cr == [0,0,0]] = 0
	
	# 198, 132,66,0
	sum_pixels = sum(sum(sum(I_cr)))
	if sum_pixels == 198:
		return 3
	elif sum_pixels == 132:
		return 2
	elif sum_pixels == 66:
		return 1
	else:
		return 0
	
	#print (sum(sum(sum(I_cr))))

# ecperimental function to compute the "life" of the player from the image
# the idea is to see if giving negative reward when life decreases improves 
# learning (i.e., makes learning faster or converges to higher score)
def compute_life_kungfu(I):
  
	# crop image to region of interest
	I_life = I[41:45,49:88]
  
  
	# convert to binary image where 1 indicates life and 0 background
	I_life[I_life == [232,232,74]] = 1
	I_life[I_life == [45,50,184]] = 0

	#toimage(I_game).save("test_crop.png")
  
	# count the proportion of pixels that are set to life pixel
	return (sum(sum(sum(I_life)))/468.00)

# save a learning curve
def save_curve(filename,ep_reward_list):
	f = open(filename, 'w')
	for row in ep_reward_list:
		line = str("%i,%i,%f,%f\n" % (row[0],row[1],row[2],row[3]))
		f.write(line)
	f.close()
	
	

def evaluate_policy_kungfu(render_eval,env,dqn_agent,D,domain,downsample_factor,feature_type,make_gray):
	
	# keep track of how many steps it tooks
	eval_run_steps = 0
	
	# reset environment
	observation = env.reset()
	
	# outcome reward
	eval_reward = 0
	
	prev_x = None
	
	image_buffer_queue = QueueBuffer(buffer_image_k)


	
	while True:
		if render_eval: 
			t  = time.time()
			env.render()
		# preprocess the observation, set input to network to be difference image
		cur_I = prepro(observation,domain,downsample_factor,make_gray)
		cur_x = cur_I.astype(np.float).ravel()
	  
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
			#print("sizes:")
			#print(len(x))
			#print(len(combined_x))
			x = combined_x
			x = x / 255
		elif feature_type == FEATURE_CHANGE_BUFFER:
			# difference from buffer avg
			image_buffer_queue.add_element(cur_I)
			avg_I = image_average(image_buffer_queue.get_elements(),make_gray)
		
		
			avg_x = avg_I.astype(np.float).ravel()
			x = cur_x - avg_x if avg_x is not None else np.zeros(D)
			x = x / 255
		
		# process input and choose action with exploration
		action = dqn_agent.process_step(x,False)
		
		observation, reward, done, info = env.step(action)
		if reward > 0:
			eval_reward += reward 
		eval_run_steps += 1
		
		# feed reward to agent
		dqn_agent.give_reward(reward)


		
		# for hockey, game ends when someone scores
		#if domain == "IceHockey-v0" and reward != 0:
		#	done = True
		
		if done:
			print("eval reward " + str(eval_reward))	
			
			# call finish episode function
			dqn_agent.finish_episode()
			
			return eval_reward
