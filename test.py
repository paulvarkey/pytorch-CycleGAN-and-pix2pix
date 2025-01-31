"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import numpy as np
import pandas as pd


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if opt.dataset_mode == 'maveric_csv':
            for label, im_tensor in visuals.items():
                im_numpy = im_tensor[0].cpu().detach().float().numpy()
                if label.endswith("A"):
                    im_numpy = ((im_numpy + 1) / 2 * (dataset.dataset.a_max - dataset.dataset.a_min)) + dataset.dataset.a_min
                elif label.endswith("B"):
                    im_numpy = ((im_numpy + 1) / 2 * (dataset.dataset.b_max - dataset.dataset.b_min)) + dataset.dataset.b_min
                    if label == "real_B":
                        real_b = im_numpy
                    elif label == "fake_B":
                        fake_b = im_numpy
                visuals[label] = np.where(im_numpy == -500, np.nan, im_numpy)

            mae = np.mean(np.abs(real_b - fake_b))
            print('processing (%04d)-th image... %s / MAE: %.3f' % (i, img_path, mae))

            fake_b_filename = os.path.join(web_dir, os.path.basename(img_path[0]))
            fake_b_colname = dataset.dataset.cols[i % len(dataset.dataset.cols)]
            try:
                fake_b_df = pd.read_csv(fake_b_filename)
                fake_b_df[fake_b_colname] = fake_b.reshape(-1)
            except FileNotFoundError:
                _, x, y = fake_b.shape
                yv, xv = np.meshgrid(np.arange(y), np.arange(x))
                fake_b_df = pd.DataFrame(
                    {
                        "x": xv.reshape(-1),
                        "y": yv.reshape(-1),
                        fake_b_colname: fake_b.reshape(-1),
                    }
                )
            fake_b_df["rsrp_dbm"] = fake_b_df.filter(regex=dataset.dataset.opt.col_regex).max(axis=1)
            fake_b_df.to_csv(fake_b_filename, index=False)
        else:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)  # save images to an HTML file
    webpage.save()  # save the HTML
