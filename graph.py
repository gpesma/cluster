import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd
import sys
import math
import numpy as np

files = []

# files.append("workings/atl_b_")
# files.append("atl_b_")
# files.append("atl_neg_")
# files.append("workings/atl_neg_")
# files.append("workings/atl_01_")
# files.append("atl_ran_")
# files.append("atl_sub_01_")
# files.append("atl_s_01_")
# files.append("atl_s_3_")
# files.append("atl_neg_")

# files.append("workings_atl/atl_sub_")
# files.append("workings/atl_sub_75_")
# files.append("workings/atl_s_")


# files.append("box_b_")
# files.append("box_n_05_")
# files.append("box_n_3_")
# files.append("workings/box_b_")
# files.append("workings/box_n_1_")
# files.append("workings/box_ran_1_")
# files.append("workings/box_sub_05_")
# files.append("stash/box_s_07_")
# files.append("box_psub_05_")
# files.append("box_s_07_")
# files.append("box_s_05_")
# files.append("box_s_03_")
# files.append("box_ran_1_")

# files.append("one_and_bust/kung_b__")
# files.append("kung_b_")
files.append("workings/kung_b_")
# files.append("workings/kung_s_")
# files.append("workings/kung_sub_1_")
files.append("workings/kung_n_200_")
files.append("kung_ran_")
# files.append("kung_sub_1_")
# files.append("workings/kung_neg_")
# files.append("kung_n_200_")
# files.append("workings/kung_n_200_")
# files.append("kung_03_")
# files.append("kung_05_")
# files.append("one_and_done/kung_s_")

rewards = []
steps = []
r = []

colors = ['blue', 'orange', 'green', 'red', 'yellow']
labels = ['Baseline', 'Subtask when Mistake', 'Random Subtask', 'one more', 'two more']

rolling_window = 50

def make_rolling(ar):
	new_ar = []
	for j in range(0, len(ar) - rolling_window):
		sum = 0
		for k in range(0, rolling_window):
			sum += ar[j + k] / rolling_window
		new_ar.append(sum)
	return new_ar


i = 0

for file in files:
	for j in range(0, 6):
		f = file + str(j) + ".txt"
		read = open (f, "r")

		max_i = 0
		max_steps = 0
		n = ""
		#calculate main graph
		with read as filestream:
			for line in filestream:
				for line in filestream:
					if len(line) >= 3:
						l = line.split(",")
						r.append((int(l[1]), float(l[2])))
						# rewards.append(float(l[2]))
						# steps.append(int(l[1]))
		read.close()
	# plt.scatter(steps, rewards, color = colors[i], label = labels[i])
	# plt.plot(steps, (np.polyfit(np.log(steps), rewards, 1)))
	r.sort(key=lambda tup: tup[0])
	steps = [j[0] for j in r]
	rewards = [j[1] for j in r]
	# steps = make_rolling(steps)
	rewards = make_rolling(rewards)
	plt.plot(steps[:len(rewards)], rewards, color = colors[i], label = labels[i])
	# plt.plot(steps[:60], rewards[: 60], color = colors[i], label = labels[i])
	i += 1

	steps = []
	rewards = []
	r= []



plt.axis([0,500000, 3000, 25000])
plt.legend()
plt.title("Algorithms 1 & 2")
plt.ylabel("Reward")
plt.xlabel("Game Steps")
plt.savefig('kung_nor.png')
plt.show()
