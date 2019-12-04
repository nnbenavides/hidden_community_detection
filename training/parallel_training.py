import os
import numpy as np
from utils import gen_layers, create_arg_string
from multiprocessing import Process


max_tests = 1000
num_tests = max_tests
seen = {}
processes = []
# workers = int(num_cpus/max_threads)
max_processes = 1

def check_processes(processes):
	count = 0
	for i in reversed(range(len(processes))):
		if not processes[i].is_alive():
			del processes[i]
			count += 1
	
	return count

def run_new(args):
	os.system('python3 train.py %s' % args)

print("Starting %d processes" % max_processes)
for t in range(max_processes):
	
	new_args = create_arg_string()
	while new_args in seen:
		new_args = create_arg_string()

	seen[new_args] = True
	processes.append(Process(target=run_new, args=(new_args,)))

	processes[-1].start()
	num_tests-=1

print("Finished starting %d processes" % max_processes)

while(num_tests > 0):
	count = check_processes(processes)
	if count > 0:
		for c in range(count):
			if num_tests == 0: break
			new_args = create_arg_string()
			while new_args in seen:
				new_args = create_arg_string()

			seen[new_args] = True
			processes.append(Process(target=run_new, args=(new_args,)))
			print("Starting process")
			processes[-1].start()
			num_tests-=1
	else:
		continue
		
for p in processes:
	p.join()


