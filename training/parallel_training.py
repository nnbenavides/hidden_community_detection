import os
import numpy as np
from utils import create_args
from threading import Thread
from time import sleep
from train import main


max_tests = 1000
num_tests = max_tests
seen = {}
threads = [[]]*num_devices
max_threads_per_device = 1
num_devices = 2
curr_device = 0
# device_counters = [0]*num_devices

# devices_pointer = 0; devices = [int(max_threads/num_devices)]

def check_threads(threads):
	# count = 0
	devices = []
	for i in range(len(threads)):
		for j in reversed(range(len(threads[inner]))):
			if not threads[i][j].is_alive():
				del threads[i][j]
				devices.append(i)


	# for i in reversed(range(len(threads))):
		# if not threads[i].is_alive():
			# del threads[i]
			# count += 1
	
	return devices

# def run_new(args, curr_device):
	# os.system('CUDA_VISIBLE_DEVICES=%d; python3 train.py %s' % (curr_device, args))

print("Starting %d threads" % max_threads)
# for t in range(max_threads):
for device in range(len(threads)):
	for j in range(max_threads_per_device):
		new_args[:-1] = create_args(device=device)
		while new_args[:-1] in seen:
			new_args = create_args(device=device)

		seen[new_args[:-1]] = True
		threads[device].append(Thread(target=main, args=new_args))
		# curr_device += 1
		# if curr_device == num_devices: curr_device = 0
		threads[device][-1].start()
		# device_counters[device]+=1
		num_tests-=1
		
		# device = -1
		# for i in range(len(threads)):
			# if device_counters[i] < max_threads_per_device:
				# device = i
	
	

print("Finished starting %d threads, will spawn more as they die" % max_threads)
while(num_tests > 0):
	sleep(15)
	devices = check_threads(threads)
	if len(devices) > 0:
		for device in devices:
			if num_tests == 0: break
			# device = -1
			# while(device==-1):
				# for i in range(len(device_counters)):
					# if device_counters[i] < max_threads_per_device:
						# device = i
			new_args = create_args(device=device)
			while new_args[:-1] in seen:
				new_args = create_args(device=device)

			seen[new_args[:-1]] = True
			threads[device].append(Thread(target=run_new, args=new_args))
			print("Starting thread")
			threads[device][-1].start()
			num_tests-=1

	else:
		continue
		
for device in threads:
	for t in threads[device]
		t.join()