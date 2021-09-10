import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from model import UNet
#from utils.data_vis import plot_img_and_mask
from dataloader import InfektaDataset
from tqdm import tqdm

IMAGE_SIZE = 16
FRAMES_COUNT = 120
torch.set_default_dtype(torch.float64)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float64)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = output
            probs = F.relu(output)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        full_mask = probs.squeeze().cpu().numpy()

    return full_mask 


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=FRAMES_COUNT*8, n_classes=8)

    logging.info("Loading model {}".format(args.model))

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
        j= 250
        img = np.load(fn+str(j+0)+".npy")
        people = np.sum(img)
        factor = 1.0/np.sum(img)
        print(factor)
        ims = []
        timeseries = []
        #ims = [img*factor for i in range(FRAMES_COUNT)]
        for k in range(j):
            img1 = np.load(fn+str(k)+".npy")*factor
            ims += [img1]
            timeseries += [[np.sum(np.floor(j*people)) for j in img1 ]]
        for k in range(FRAMES_COUNT):
            img1 = np.load(fn+str(j+k)+".npy")*factor
            ims += [img1]
            timeseries += [[np.sum(np.floor(j*people)) for j in img1 ]]
        frames = []
        fig, ax = plt.subplots()
        lines = ax.plot([k-j-FRAMES_COUNT for k in range(len(timeseries))],timeseries)
        ax.legend(lines,["DEAD","IMMUNE","RECOVERED","SUSCEPTIBLE","EXPOSED","ASYMPTOTIC","SERIOUSLY","CRITICAL"])
        plt.show(block=False)
        for i in tqdm(range(60*24-FRAMES_COUNT)):
            img4  = np.vstack(ims[len(ims)-FRAMES_COUNT:])
            mask = predict_img(net=net,
                            full_img=img4,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
            factor = 1.0/np.sum(mask)
            ims += [mask*factor]
            frames += [np.reshape(mask/np.sum(mask),(IMAGE_SIZE*8,IMAGE_SIZE))]
            timeseries += [[np.sum(np.floor((j/np.sum(mask))*people)) for j in mask ]]
            img = mask
            #ax.plot([k-j-FRAMES_COUNT for k in range(len(timeseries))],timeseries)
            #fig.canvas.draw_idle()
            #plt.pause(.001)
            
        fig, ax = plt.subplots()
        lines = plt.plot([k-j-FRAMES_COUNT for k in range(len(timeseries))],timeseries)
        plt.legend(lines,["DEAD","IMMUNE","RECOVERED","SUSCEPTIBLE","EXPOSED","ASYMPTOTIC","SERIOUSLY","CRITICAL"])
        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left=0.25, bottom=0.25)

        axcolor = 'lightgoldenrodyellow'
        # Make a horizontal slider to control the frequency.
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        freq_slider = Slider(
            ax=axfreq,
            label='Frame',
            valmin=0,
            valmax=60*24-1,
            valinit=1,
        )
        def update(val):
            frame = frames[int(val)]
            ax.imshow(frame)
            fig.canvas.draw_idle()
        freq_slider.on_changed(update)
        plt.show()
        if not args.no_save:
            out_fn = out_files[i]
            with open(out_files[i],"wb") as f:
                print(mask)
                np.save(f,mask)
                print(np.sum(mask))
                for z in range(8):
                    matplotlib.image.imsave(out_files[i]+ str(z).zfill(5) + ".png", mask[z])

            logging.info("Result saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
