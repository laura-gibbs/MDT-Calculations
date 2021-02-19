import imageio
import glob
images = []
filenames = glob.glob("./gif_imgs/*.png") 
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('test.gif', images)