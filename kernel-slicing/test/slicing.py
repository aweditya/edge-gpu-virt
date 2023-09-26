import numpy as np

# After queues are equalized; Distribute load such that 
# all GPUs finish at same time

# Generate random per gpu profiles for a function
np.random.seed(5)
points = np.random.choice(10000, 10, replace='False')
points.sort()

gpus = 5
profiles = np.zeros([gpus, points.shape[0]])
profiles[0, :] = [3* x**(1.2) + 2  for x in points]
profiles[1, :] = [4* x**(1.05) + 2  for x in points]
profiles[2, :] = [2* x**(1.03) + 5  for x in points]
profiles[3, :] = [1* x**(1.01) + 2  for x in points]
profiles[4, :] = [x + 3  for x in points]

# number of gpus of each type
n_gpus = [1, 4, 2, 3, 1]



# Linearly Interpolate/Extrapolate points given function values at few points
def inter_extra(x, points, y):
	if x < points[0]:
		return y[0]+(x-points[0])*(y[1]-y[0])/(points[1]-points[0])
	elif x > points[-1]:
		return y[-1]+(x-points[-1])*(y[-1]-y[-2])/(points[-1]-points[-2])
	return np.interp(x, points, y)




# Data size to evenly distribute
L_rem = 20000

print('Data to distribute  %d' % (L_rem))



# algorithm to evenl distribute so that total completion time is minimized
i = 0
time = inter_extra(L_rem / n_gpus[i], points, profiles[i])
prev = 0


C = n_gpus[i]
i = i + 1
while i < gpus and np.abs(time - prev) > 1e-3:
	a = [n_gpus[j] * inter_extra(time, profiles[j], points) for j in range(i)]
	prev = time
	t = time / 2
	time = 0
	while t > 1e-3:
		rem = L_rem - np.sum([n_gpus[j] * inter_extra(time + t, profiles[j], points) for j in range(i)])
		if inter_extra(rem/n_gpus[i], points, profiles[i]) > time + t:	
			time = time + t	
		t = t / 2
	if time > prev:
		break
	else:
		C = C + n_gpus[i]
		i = i + 1


a = [n_gpus[i] * inter_extra(time, profiles[i], points) for i in range(gpus)]
print('Time: ', time)
print('Distribution: ' )

print("gpu\tn_gpus\tdata to assigned to single gpu")
for i in range(gpus):
	d = inter_extra(time, profiles[i], points)
	print("%d\t%d\t* %.3f data = \t%.3f data\t %.3f time" %(i + 1, n_gpus[i], d, n_gpus[i] * d, inter_extra(d, points, profiles[i]) ))

print(np.sum(a))
