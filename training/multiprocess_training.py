import os
import numpy as np
from utils import create_args
from multiprocessing import Process
from time import sleep
from train import main


max_tests = 1000
num_tests = max_tests
seen = {}
num_devices = 2
processes = [[]]*num_devices
max_processes_per_device = 1

curr_device = 0
# device_counters = [0]*num_devices

# devices_pointer = 0; devices = [int(max_processes/num_devices)]

def check_processes(processes):
	# count = 0
	devices = []
	for i in range(len(processes)):
		for j in reversed(range(len(processes[i]))):
			if not processes[i][j].is_alive():
				del processes[i][j]
				devices.append(i)


	# for i in reversed(range(len(processes))):
		# if not processes[i].is_alive():
			# del processes[i]
			# count += 1
	
	return devices

# def run_new(args, curr_device):
	# os.system('CUDA_VISIBLE_DEVICES=%d; python3 train.py %s' % (curr_device, args))

print("Starting processes")
# for t in range(max_processes):
for device in range(len(processes)):
	for j in range(max_processes_per_device):
		new_args[:-1] = create_args(device=device)
		while new_args[:-1] in seen:
			new_args = create_args(device=device)

		seen[new_args[:-1]] = True
		processes[device].append(Process(target=main, args=new_args))
		# curr_device += 1
		# if curr_device == num_devices: curr_device = 0
		processes[device][-1].start()
		# device_counters[device]+=1
		num_tests-=1
		
		# device = -1
		# for i in range(len(processes)):
			# if device_counters[i] < max_processes_per_device:
				# device = i
	
	

print("Finished starting processes, will spawn more as they die")
while(num_tests > 0):
	sleep(15)
	devices = check_processes(processes)
	if len(devices) > 0:
		for device in devices:
			if num_tests == 0: break
			# device = -1
			# while(device==-1):
				# for i in range(len(device_counters)):
					# if device_counters[i] < max_processes_per_device:
						# device = i
			new_args = create_args(device=device)
			while new_args[:-1] in seen:
				new_args = create_args(device=device)

			seen[new_args[:-1]] = True
			processes[device].append(Process(target=run_new, args=new_args))
			print("Starting process")
			processes[device][-1].start()
			num_tests-=1

	else:
		continue
		
for device in processes:
	for p in processes[device]
		p.join()