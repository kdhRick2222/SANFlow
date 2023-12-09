import os
import datetime
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd

from utils import *

OUT_DIR = './viz/'

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300

def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def export_hist(c, gts, scores, threshold):
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    # image_dirs = os.path.join(OUT_DIR, c.model)
    image_dirs = os.path.join(OUT_DIR, c.enc_arch)
    os.makedirs(image_dirs, exist_ok=True)
    Y = scores.flatten().clip(0, 2)
    Y_label = gts.flatten()
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [-1., 0., 2., 3.])
    # ay = plt.Axes(fig, [-1., 0., 2., 3.])
    fig.add_axes(ax)
    plt.hist([Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['r', 'g'], label=['ANO', 'TYP'], alpha=0.75, histtype='barstacked')
    # image_file = os.path.join(image_dirs, 'hist_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.png')
    image_file = os.path.join(image_dirs, 'hist_images.png')
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()


def export_groundtruth(c, test_img, gts):
    # image_dirs = os.path.join(OUT_DIR, c.model, 'gt_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    image_dirs = os.path.join(OUT_DIR, c.enc_arch, 'gt_images')
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting grountruth...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            #
            fig = plt.figure(figsize=(2*cm, 2*cm), dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(gt_img)
            image_file = os.path.join(image_dirs, '{:08d}.png'.format(i))
            fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def export_scores(c, test_img, scores, gts, threshold, name_):
    # image_dirs = os.path.join(OUT_DIR, c.model, 'sc_images_' + name_ + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    image_dirs = os.path.join(OUT_DIR, c.enc_arch, 'sc_images_')
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting scores...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            # score_map = (255.0*(scores[i]-scores.min()) / (scores.max() - scores.min())).astype(np.uint8)
            # gtmask
            gt_mask = gts[i].astype(np.float64)
            #
            fig_img, ax_img = plt.subplots(2, 1, figsize=(2*cm, 4*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(img, cmap='gray', interpolation='none')
            # ax_img[0].imshow(img, interpolation='none')
            ax_img[0].imshow(score_map, cmap='inferno', norm=norm, alpha=0.9, interpolation='none')
            ax_img[1].imshow(score_img)
            image_file = os.path.join(image_dirs, '{:08d}.png'.format(i))
            fig_img.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def export_test_images(c, test_img, gts, var_maps, scores, threshold):
    # image_dirs = os.path.join(OUT_DIR, c.model, 'images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    image_dirs = os.path.join(OUT_DIR, c.enc_arch, 'images')
    cm = 1/2.54
    var_maps = var_maps.squeeze(1)
    var_dirs = os.path.join(OUT_DIR, c.class_name)
    var_list = var_maps.ravel()
    fig = plt.figure(figsize=(10*cm, 10*cm), dpi=dpi)
    # fig = plt.figure(dpi=dpi)
    # ax = plt.Axes(fig)
    ax = plt.Axes(fig, [0., 0.1, 0.2, 0.3])
    fig.add_axes(ax)
    plt.hist(var_list, bins=np.arange(0, 0.3+0.001, step=0.001))
    fig.savefig(var_dirs+ '.png', dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

    # print(scores[1].shape, var_maps[1].shape)
    # print(scores[1].min(), scores[1].max(), var_maps[1].min(), var_maps[1].max())
    # print(var_maps[1])
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting images...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        # scores_norm = 1.0/ (scores.max() - scores.min())
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            # print('GT:', i, gt_mask.sum())
            # gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            # print('SC:', i, score_mask.sum())
            # score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            # print(var_maps[i].max(), var_maps[i].min(), var_maps[i].mean(), var_maps[i].std())
            max_ = var_maps[i].max()
            var_norm = var_maps[i].clip(0, 0.2)
            var_norm1 = (var_norm - var_norm.mean())/var_norm.std()
            # var_norm1 = (var_norm1 - var_norm1.mean())/var_norm1.std()
            var_norm2 = var_norm1
            # var_norm2 = morphology.opening(var_norm1, kernel)
            var_map = (255.0*var_norm2).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(3, 1, figsize=(4*cm, 12*cm))
            # fig_img, ax_img = plt.subplots(4, 1, figsize=(2*cm, 8*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(gt_img)
            # ax_img[1].imshow(gt_mask, cmap='gray')
            ax_img[1].imshow(score_map, cmap='jet') #, norm=norm)
            ax_img[2].imshow(score_img)
            # ax_img[3].imshow(var_map, cmap='gray', norm=norm)

            image_file = os.path.join(image_dirs, '{:08d}.png'.format(i))

            mask_file = os.path.join(image_dirs, '{:08d}_mask.png'.format(i))
            input_file = os.path.join(image_dirs, '{:08d}_input.png'.format(i))
            output_file = os.path.join(image_dirs, '{:08d}_output.png'.format(i))
            varmap_file = os.path.join(image_dirs, '{:08d}_varmap.png'.format(i))

            fig_img.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # plt.plot(gt_img)
            # print(gt_mask.shape)
            # ## separately saving
            # mask_img, ax_mask = plt.subplots(1, 1, figsize=(2*cm, 2*cm))
            # ax_mask.axes.xaxis.set_visible(False)
            # ax_mask.axes.yaxis.set_visible(False)
            # ax_mask.spines['top'].set_visible(False)
            # ax_mask.spines['right'].set_visible(False)
            # ax_mask.spines['bottom'].set_visible(False)
            # ax_mask.spines['left'].set_visible(False)
            # ax_mask.imshow(gt_mask, cmap='gray')
            # mask_img.savefig(mask_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # input_img, ax_input = plt.subplots(1, 1, figsize=(2*cm, 2*cm))
            # ax_input.axes.xaxis.set_visible(False)
            # ax_input.axes.yaxis.set_visible(False)
            # ax_input.spines['top'].set_visible(False)
            # ax_input.spines['right'].set_visible(False)
            # ax_input.spines['bottom'].set_visible(False)
            # ax_input.spines['left'].set_visible(False)
            # ax_input.imshow(img)
            # input_img.savefig(input_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # input_score, ax_score = plt.subplots(1, 1, figsize=(2*cm, 2*cm))
            # ax_score.axes.xaxis.set_visible(False)
            # ax_score.axes.yaxis.set_visible(False)
            # ax_score.spines['top'].set_visible(False)
            # ax_score.spines['right'].set_visible(False)
            # ax_score.spines['bottom'].set_visible(False)
            # ax_score.spines['left'].set_visible(False)
            # ax_score.imshow(score_img)
            # input_score.savefig(output_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # var_img, ax_var = plt.subplots(1, 1)#, figsize=(4*cm, 4*cm))
            # ax_var.axes.xaxis.set_visible(False)
            # ax_var.axes.yaxis.set_visible(False)
            # ax_var.spines['top'].set_visible(False)
            # ax_var.spines['right'].set_visible(False)
            # ax_var.spines['bottom'].set_visible(False)
            # ax_var.spines['left'].set_visible(False)

            # img = ax_var.imshow(var_map, cmap='gray', norm=norm)
            # # img = ax_var.imshow(var_map)
            # cbar = plt.colorbar(img, ax=ax_var, orientation='vertical', norm=norm)
            # # cbar.set_ticks([])
            # # ax_var.colorbar()
            # # plt.show()
            # var_img.savefig(varmap_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()


def export_test_means(c, test_img, gts, N_maps, A_maps, scores, threshold):
    image_dirs = os.path.join(OUT_DIR, c.model, 'images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    cm = 1/2.54
    # print(N_maps.shape)
    N_maps = N_maps.squeeze(1)
    A_maps = A_maps.squeeze(1)
    var_dirs = os.path.join(OUT_DIR, c.class_name)
    var_list = N_maps.ravel()
    fig = plt.figure(figsize=(10*cm, 10*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0.1, 0.2, 0.3])
    fig.add_axes(ax)
    plt.hist(var_list, bins=np.arange(0, 0.3+0.001, step=0.001))
    fig.savefig(var_dirs+ '.png', dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

    if not os.path.isdir(image_dirs):
        print('Exporting images...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            # print('SC:', i, score_mask.sum())
            # score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            # print(var_maps[i].max(), var_maps[i].min(), var_maps[i].mean(), var_maps[i].std())
            N_norm = N_maps[i].clip(-2, 2)
            N_norm2 = (N_norm - N_norm.min())/(N_norm.max() - N_norm.min())
            N_map = (255.0*N_norm2).astype(np.uint8)
            A_norm = A_maps[i].clip(-1, 3)
            A_norm2 = (A_norm - A_norm.min())/(A_norm.max() - A_norm.min())
            A_map = (255.0*A_norm2).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(5, 1, figsize=(4*cm, 12*cm))
            # fig_img, ax_img = plt.subplots(4, 1, figsize=(2*cm, 8*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(gt_img)
            # ax_img[1].imshow(gt_mask, cmap='gray')
            ax_img[1].imshow(score_map, cmap='jet') #, norm=norm)
            ax_img[2].imshow(score_img)
            ax_img[3].imshow(N_map, cmap='gray', norm=norm)
            ax_img[4].imshow(A_map, cmap='gray', norm=norm)

            image_file = os.path.join(image_dirs, '{:08d}.png'.format(i))

            mask_file = os.path.join(image_dirs, '{:08d}_mask.png'.format(i))
            input_file = os.path.join(image_dirs, '{:08d}_input.png'.format(i))
            output_file = os.path.join(image_dirs, '{:08d}_output.png'.format(i))
            varmap_file = os.path.join(image_dirs, '{:08d}_varmap.png'.format(i))

            fig_img.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # plt.plot(gt_img)
            # print(gt_mask.shape)
            # ## separately saving
            # mask_img, ax_mask = plt.subplots(1, 1, figsize=(2*cm, 2*cm))
            # ax_mask.axes.xaxis.set_visible(False)
            # ax_mask.axes.yaxis.set_visible(False)
            # ax_mask.spines['top'].set_visible(False)
            # ax_mask.spines['right'].set_visible(False)
            # ax_mask.spines['bottom'].set_visible(False)
            # ax_mask.spines['left'].set_visible(False)
            # ax_mask.imshow(gt_mask, cmap='gray')
            # mask_img.savefig(mask_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # input_img, ax_input = plt.subplots(1, 1, figsize=(2*cm, 2*cm))
            # ax_input.axes.xaxis.set_visible(False)
            # ax_input.axes.yaxis.set_visible(False)
            # ax_input.spines['top'].set_visible(False)
            # ax_input.spines['right'].set_visible(False)
            # ax_input.spines['bottom'].set_visible(False)
            # ax_input.spines['left'].set_visible(False)
            # ax_input.imshow(img)
            # input_img.savefig(input_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # input_score, ax_score = plt.subplots(1, 1, figsize=(2*cm, 2*cm))
            # ax_score.axes.xaxis.set_visible(False)
            # ax_score.axes.yaxis.set_visible(False)
            # ax_score.spines['top'].set_visible(False)
            # ax_score.spines['right'].set_visible(False)
            # ax_score.spines['bottom'].set_visible(False)
            # ax_score.spines['left'].set_visible(False)
            # ax_score.imshow(score_img)
            # input_score.savefig(output_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)

            # var_img, ax_var = plt.subplots(1, 1)#, figsize=(4*cm, 4*cm))
            # ax_var.axes.xaxis.set_visible(False)
            # ax_var.axes.yaxis.set_visible(False)
            # ax_var.spines['top'].set_visible(False)
            # ax_var.spines['right'].set_visible(False)
            # ax_var.spines['bottom'].set_visible(False)
            # ax_var.spines['left'].set_visible(False)

            # img = ax_var.imshow(var_map, cmap='gray', norm=norm)
            # # img = ax_var.imshow(var_map)
            # cbar = plt.colorbar(img, ax=ax_var, orientation='vertical', norm=norm)
            # # cbar.set_ticks([])
            # # ax_var.colorbar()
            # # plt.show()
            # var_img.savefig(varmap_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
            
            # print(N_map.shape)
            # weight = N_maps[i].reshape(-1).tolist() + A_maps[i].reshape(-1).tolist()
            weightN = N_maps[i].reshape(-1).tolist()
            A_maps2 = A_maps[i]+0.8
            weightA = A_maps2.reshape(-1).tolist()
            plt.hist(weightN, bins=50)
            plt.hist(weightA, bins=50)
            plt.savefig(image_dirs + '/hist_{:08d}.png'.format(i))
            plt.close()


def distribution_fig(x, label, class_name, num, L):
    OUT_DIR = './distribution/' + class_name + '/' + 'num{}/'.format(num)
    image_dirs = os.path.join(OUT_DIR)
    os.makedirs(image_dirs, exist_ok=True)
    tsne = TSNE(n_components=2, verbose=1) #, perplexity=40, n_iter=1000)
    z_ = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = label
    df["comp-1"] = z_[:,0]
    df["comp-2"] = z_[:,1]
    plt.figure(figsize = ( 8 , 6 ))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 2), data=df)
    plt.xlabel( "x" , size = 12 )
    plt.ylabel( "y" , size = 12 )
    plt.title( "distribution" , size = 12)
    plt.savefig(OUT_DIR + 'pooling_layer{}.png'.format(L))


def distribution_histo(x, label, class_name, num, L):
    OUT_DIR = './distribution/' + class_name + '/' + 'num{}/'.format(num)
    image_dirs = os.path.join(OUT_DIR)
    os.makedirs(image_dirs, exist_ok=True)
    # tsne = TSNE(n_components=2, verbose=1) #, perplexity=40, n_iter=1000)
    # z_ = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = label
    df["comp-1"] = x[:,0]
    df["comp-2"] = x[:,1]
    plt.figure(figsize = (8, 6))
    # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 2), data=df)
    plt.hist([df["comp-1"][label==0], df["comp-1"][label==1]], 100, density=True, color=['g', 'r'], label=['NORMAL', 'DEFECTS'], alpha=0.5, histtype='barstacked')
    plt.legend(loc='upper left')
    plt.xlabel( "x" , size = 12 )
    plt.ylabel( "y" , size = 12 )
    plt.title( "distribution" , size = 12)
    plt.savefig(OUT_DIR + 'pooling_layer{}.png'.format(L))

# def distribution_histo(x, label, class_name, num, L):
#     OUT_DIR = './distribution/' + class_name + '/' + 'num{}/'.format(num)
#     image_dirs = os.path.join(OUT_DIR)
#     os.makedirs(image_dirs, exist_ok=True)
#     tsne = TSNE(n_components=2, verbose=1) #, perplexity=40, n_iter=1000)
#     z_ = tsne.fit_transform(x)
#     df = pd.DataFrame()
#     df["y"] = label
#     df["comp-1"] = z_[:,0]
#     df["comp-2"] = z_[:,1]
#     plt.figure(figsize = ( 8 , 6 ))
#     sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 2), data=df)
#     plt.xlabel( "x" , size = 12 )
#     plt.ylabel( "y" , size = 12 )
#     plt.title( "distribution" , size = 12)
#     plt.savefig(OUT_DIR + 'pooling_layer{}.png'.format(L))