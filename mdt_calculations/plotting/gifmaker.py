import imageio
import glob


def gen_gif(imgs_dir, name):
    images = []
    filenames = glob.glob('gifs/gif_imgs/'+imgs_dir+'/*.png') 
    print(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('gifs/'+name+'.gif', images, duration=0.5)
