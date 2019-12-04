import os
import numpy as np
from utils import gen_layers, create_arg_string
from threading import Thread


max_tests = 1000
num_tests = max_tests
seen = {}
threads = []
# workers = int(num_cpus/max_threads)
max_threads = 1

def check_threads(threads):
	count = 0
	for i in reversed(range(len(threads))):
		if not threads[i].is_alive():
			del threads[i]
			count += 1
	
	return count

def run_new(args):
	os.system('python3 train.py %s' % args)

print("Starting %d threads" % max_threads)
for t in range(max_threads):
	
	new_args = create_arg_string()
	while new_args in seen:
		new_args = create_arg_string()

	seen[new_args] = True
	threads.append(Thread(target=run_new, args=(new_args,)))

	threads[-1].start()
	num_tests-=1

print("Finished starting %d threads" % max_threads)

while(num_tests > 0):
	count = check_threads(threads)
	if count > 0:
		for c in range(count):
			if num_tests == 0: break
			new_args = create_arg_string()
			while new_args in seen:
				new_args = create_arg_string()

			seen[new_args] = True
			threads.append(Thread(target=run_new, args=(new_args,)))
			print("Starting thread")
			threads[-1].start()
			num_tests-=1
	else:
		continue
		
for t in threads:
	t.join()


