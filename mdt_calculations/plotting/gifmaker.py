import imageio
import glob
images = []
filenames = glob.glob("../../gif_imgs/*.png") 
print(filenames)
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('test.gif', images, duration=0.5)