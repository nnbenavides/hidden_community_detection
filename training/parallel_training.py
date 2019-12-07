import os
import numpy as np
from utils import create_args
from threading import Thread
from time import sleep
from node2vectrainer import main


max_tests = 1000
num_tests = max_tests
seen = {}
threads = []
max_threads = 1
curr_device = 0

def check_threads(threads):
	count = 0

	for i in reversed(range(len(threads))):
		if not threads[i].is_alive():
			del threads[i]
			count += 1
	
	return count

print("Starting threads")
for i in range(max_threads):
	new_args = create_args()
	while str(new_args[:-1]) in seen:
		new_args = create_args()

	seen[str(new_args[:-1])] = True
	threads.append(Thread(target=main, args=new_args))
	threads[-1].start()
	num_tests-=1
	
	

print("Finished starting threads, will spawn more as they die")
while(num_tests > 0):
	sleep(15)
	count = check_threads(threads)
	if count > 0:
		for i in range(count):
			if num_tests == 0: break
			new_args = create_args()
			while str(new_args[:-1]) in seen:
				new_args = create_args()

			seen[str(new_args[:-1])] = True
			threads.append(Thread(target=main, args=new_args))
			print("Starting thread")
			threads[-1].start()
			num_tests-=1
	else:
		continue
		
for device in threads:
	for t in threads[device]:
		t.join()