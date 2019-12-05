import os
import numpy as np
from utils import gen_layers, create_arg_string
from threading import Thread


max_tests = 1000
num_tests = max_tests
seen = {}
threads = []
# workers = int(num_cpus/max_threads)
max_threads = 4
# device1 = int(max_threads/2)
# device2 = int(max_threads/2)
num_devices = 2
curr_device = 0

# devices_pointer = 0; devices = [int(max_threads/num_devices)]

def check_threads(threads):
	count = 0
	for i in reversed(range(len(threads))):
		if not threads[i].is_alive():
			del threads[i]
			count += 1
	
	return count

def run_new(args, curr_device):
	os.system('CUDA_VISIBLE_DEVICES=%d; python3 train.py %s' % (args, curr_device))

print("Starting %d threads" % max_threads)
for t in range(max_threads):
	
	new_args = create_arg_string()
	while new_args in seen:
		new_args = create_arg_string()

	seen[new_args] = True

	threads.append(Thread(target=run_new, args=(new_args, curr_device,)))
	curr_device += 1
	if curr_device == num_devices; curr_device = 0
	threads[-1].start()
	num_tests-=1

print("Finished starting %d threads, will spawn more as they die" % max_threads)

while(num_tests > 0):
	count = check_threads(threads)
	if count > 0:
		for c in range(count):
			if num_tests == 0: break
			new_args = create_arg_string()
			while new_args in seen:
				new_args = create_arg_string()

			seen[new_args] = True
			threads.append(Thread(target=run_new, args=(new_args, curr_device,)))
			curr_device += 1
			if curr_device == num_devices; curr_device = 0
			print("Starting thread")
			threads[-1].start()
			num_tests-=1
	else:
		continue
		
for t in threads:
	t.join()


