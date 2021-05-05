import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation

from envs import rastrigin, quadratic, sphere, griewangk, styblinski_tang



def visualize(env, path, fpath, title):
	writergif = animation.PillowWriter(fps=30)
	fpath = os.path.split(os.path.split(fpath)[0])[0]
	path = np.array(path).T
	minima = env.minima
	minima_ = minima.reshape(-1, 1)

	xmin, xmax, xstep = env.min_bound, env.max_bound, .2
	ymin, ymax, ystep = env.min_bound, env.max_bound, .2

	x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
	z = env.plot_eval_func([x,y])

	# 3d plot
	fig = plt.figure(figsize=(8, 5))
	ax = plt.axes(projection='3d', elev=50, azim=-50)

	ax.plot_surface(x, y, z, rstride=1, cstride=1, 
	                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
	ax.plot(*minima_, env.plot_eval_func(minima_), 'r*', markersize=10)

	line, = ax.plot([], [], [], 'b', label='Newton-CG', lw=2)
	point, = ax.plot([], [], [], 'bo')

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_zlabel('$z$')

	ax.set_xlim((xmin, xmax))
	ax.set_ylim((ymin, ymax))

	def animate3d(i):
	    line.set_data(path[0,:i], path[1,:i])
	    line.set_3d_properties(env.plot_eval_func(path[::,:i]))
	    point.set_data(path[0,i-1:i], path[1,i-1:i])
	    point.set_3d_properties(env.plot_eval_func(path[::,i-1:i]))
	    return line, point

	anim = animation.FuncAnimation(fig, animate3d,
	                               frames=path.shape[1], interval=60, 
	                               repeat_delay=5, blit=True)
	anim.save(os.path.join(fpath, 'Surface ' + title + '.gif'), writer=writergif)
	
	ax.quiver(path[0,:-1], path[1,:-1], env.plot_eval_func(path[::,:-1]), 
	          path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], env.plot_eval_func((path[::,1:]-path[::,:-1])), 
	          color='k')
	plt.savefig(os.path.join(fpath, 'Surface ' + title + '.jpg'))

	# contour plot
	fig, ax = plt.subplots(figsize=(10, 6))

	ax.contour(x, y, z, cmap=plt.cm.jet)
	ax.plot(*minima_, 'r*', markersize=18)

	line, = ax.plot([], [], 'b', label='Newton-CG', lw=2)
	point, = ax.plot([], [], 'bo')

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	ax.set_xlim((xmin, xmax))
	ax.set_ylim((ymin, ymax))

	def animate2d(i):
	    line.set_data(*path[::,:i])
	    point.set_data(*path[::,i-1:i])
	    return line, point

	anim = animation.FuncAnimation(fig, animate2d,
	                               frames=path.shape[1], interval=60, 
	                               repeat_delay=5, blit=True)
	anim.save(os.path.join(fpath, 'Contour ' + title + '.gif'), writer=writergif)
	
	ax.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
	plt.savefig(os.path.join(fpath, 'Contour ' + title + '.jpg'))


if __name__ == '__main__':
	path = np.array([(5.,5.), (4., 4.,), (3., 3.), (2., 2.), (1., 1.),]).T
	env = quadratic.Quadratic(dimension=2)
	visualize(env, path)