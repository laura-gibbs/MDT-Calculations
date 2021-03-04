import imageio
import glob


def gen_gif(imgs_dir, name):
    images = []
    filenames = glob.glob('figs/gif_imgs/'+imgs_dir+'/*.png') 
    print(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('figs/gifs/'+name+'.gif', images, duration=1)
