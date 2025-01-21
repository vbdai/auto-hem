import torch, os, errno
import matplotlib.pyplot as plt
import random, collections
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
import io, cv2

import datetime
import timeit

def save_file(data, file_path, mode='np'):
    if mode=='np':
        #numpy
        # create folder if does not exist
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        file = open(file_path, 'wb')
        np.save(file, data)
        print(f'\033[34m Saved {file_path} \033[0m')

def load_file(file_path, mode='np'):
    if mode == 'np':
        file = open(file_path, 'rb')
        data = np.load(file, allow_pickle=True)
        print(f'\033[34m Loaded {file_path} \033[0m')
    return data

class Timer:
    """Measure time used."""
    # Ref: https://stackoverflow.com/a/57931660/

    def __init__(self, round_ndigits: int = 0):
        self._round_ndigits = round_ndigits
        self._start_time = timeit.default_timer()

    def reset(self):
        self._start_time = timeit.default_timer()

    def __call__(self) -> float:
        return timeit.default_timer() - self._start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), self._round_ndigits)))


def hist(data, display=True, plot=False):
    counter = collections.Counter(data)
    list1 = [counter[i] for i in range(10)]
    list2 = [round(counter[i] / len(data) * 100, 2) for i in range(10)]

    if plot==True:
        plt.plot(range(len(list2)), list2, 'o-')
        plt.ylim([0,11])
        plt.title('Real-world Data distribution')
        plt.ylabel('Percentage of data (%)')
        plt.xlabel('Class label')


    list1 = ','.join(map(str, list1+ ['T', len(data)]))
    list2 = ','.join(map(str, list2))
    if display:
        print(list1)
        print(list2)

    return list1, list2

# keywords -> list of keywords in the saved model
# objects -> list of all objects to load the state dict into
def load_model(keywords, objects, current_epoch, filename):
    output_objs = tuple()
    print('\033[34m Loading model: ', filename + str(current_epoch) + '\033[0m')
    checkpoint = torch.load(filename + str(current_epoch))
    for idx, keyword in enumerate(keywords):
        objects[idx].load_state_dict(checkpoint[keyword])
        del checkpoint[keyword]
        output_objs = output_objs + (objects[idx],)
    return output_objs

def save_model(keywords, objects, epoch, filename):
    results = {}
    results['epoch'] = epoch
    for idx, keyword in enumerate(keywords):
        results[keyword] = objects[idx].state_dict()

    # create folder if does not exist
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    torch.save(results, filename + str(epoch))

def get_param_stats(net, is_print=False):
    if is_print:
        print('Name | Size | Norm | RequiresGrad')
    norm_dict = {}
    for name, parameter in net.named_parameters():
        if is_print:
            print(name,
              ' | ', list(parameter.size()),
              ' |', parameter.norm().cpu().data.numpy(),
              ' | ', parameter.requires_grad)
        norm_dict[name] = parameter.norm().cpu().data.numpy()

    if is_print:
        print('Module training?')
        for name, module in net.named_modules():
            print(name, module.training)

    return norm_dict

def _subsample_feats(feats, random=True, cap_samples=10000):
    new_feats = collections.OrderedDict()
    for feat in feats.keys():
        feats_key = feats[feat]
        if random == True:
            new_feats[feat] = feats_key[np.random.choice(feats_key.shape[0], cap_samples, replace=False), :]
        elif random == False:
            new_feats[feat] = feats_key[:10000, :]
        else:
            print('Not implemented')
            exit()
    return new_feats

def get_img_from_fig(fig, dpi=45):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def pcaplot_binary(train_data, adv_data, type = 'together', title='PCA'):
    pca_obj = decomposition.PCA(n_components=2)
    ss = preprocessing.StandardScaler()

    if type == 'together':
        features = np.concatenate([train_data, adv_data], 0)
        features = ss.fit_transform(features)
        new_feats = pca_obj.fit_transform(features)
        new_feats_train = new_feats[0:train_data.shape[0], :]
        new_feats_adv = new_feats[train_data.shape[0]:, :]
    else:
        new_feats_train = ss.fit_transform(train_data)
        new_feats_train = pca_obj.fit_transform(new_feats_train)
        new_feats_adv = ss.transform(adv_data)
        new_feats_adv = pca_obj.transform(new_feats_adv)

    start = 2000
    num_points = 500
    fig = plt.figure()
    plt.scatter(new_feats_train[start:start + num_points, 0], new_feats_train[start:start + num_points, 1], marker='_')
    plt.scatter(new_feats_adv[start:start + num_points, 0], new_feats_adv[start:start + num_points, 1], marker='|')
    plt.legend(['first', 'second'])
    plt.title(title)
    plt.show()

def visualize_clusters(activations, labels, title='Scatter', type='pca'):
    # activations -> Nx50 or Nx2 or ..
    # labels -> 1,1,2,3,5 ...

    markers = ['|', '_', 'x', '*', '1', '2', '3', '4', '.', 'd']
    scatter_count = 4000
    dim = activations.shape[1]
    total_num = activations.shape[0]
    uniq_labels = np.unique(labels)

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("bright", len(uniq_labels))

    sp = StratifiedShuffleSplit(n_splits=1, test_size=1-scatter_count/total_num)
    for train_index, _ in sp.split(activations, labels):
        x_train, y_train = activations[train_index], labels[train_index]
        break

    if dim > 2:
        if type == 'pca':
            print('Performing PCA')
            ss = preprocessing.StandardScaler()
            pca_obj = decomposition.PCA(n_components=2)
            x_train = ss.fit_transform(x_train)
            x_train = pca_obj.fit_transform(x_train)
            title = title + ' | PCAed ' + str(dim) + '->2'
        elif type == 'tsne':
            print('Performing TSNE')
            tsne = TSNE()
            x_train = tsne.fit_transform(x_train)
            title = title + ' | TSNEed ' + str(dim) + '->2'


    print(np.sum(y_train == 0))
    fig, ax = plt.subplots()
    sns.scatterplot(x_train[:, 0], x_train[:, 1], hue=y_train, legend='full', y_jitter=True, x_jitter=True, edgecolor='none', alpha=.40, palette=palette)
    plt.title(title)
    # plt.show()
    return fig

# Draw a list/ np/ tensor of images in a kxk grid
def drawImages(samples):
    if isinstance(samples, list):
        samples = np.concatenate(samples,0).squeeze()
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy().squeeze()
    count = samples.shape[0]
    if count == 1:
        plt.imshow(samples.squeeze())
    elif count>1 and count <=4:
        fig, ax = plt.subplots(2,2)
        for im_id in range(count):
            image = samples[im_id]
            ax[int(im_id/2), int(im_id%2)].imshow(image)
    elif count> 4 and count <=9:
        fig, ax = plt.subplots(3,3, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
        for im_id in range(count):
            image = samples[im_id]
            ax[int(im_id/3), int(im_id%3)].imshow(image)
    plt.show()

def viewArray(imgs, gt, preds):
    count = len(imgs)
    if count <=3:
        img = torch.cat(imgs, 2)
        img = img.squeeze().cpu().detach().numpy()
        pred = ', '.join([str(i.cpu().detach().numpy()) for i in preds])
        gt = gt.cpu().detach().numpy()
    plt.imshow(img)
    title = 'GT - ' + str(gt) + ' |||  Pred - ' + str(pred)
    plt.title(title)
    plt.xlabel('Benign, Fast Gradient Method, Basic Iterative Method')
    plt.show()

def flipimg(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = img.swapaxes(0,1).swapaxes(1,2)
    return img

def viewRandom(imgs, y, preds):
    [x, x_fgm, x_pgd] = imgs
    [y_pred, y_pred_fgm, y_pred_pgd] = preds

    ind = random.randint(0,len(x)-1)
    viewArray([x[ind], x_fgm[ind], x_pgd[ind]],
              y[ind],
              [y_pred[ind], y_pred_fgm[ind], y_pred_pgd[ind]])

def viewSingle(img, gt, pred):
  img = img.squeeze().cpu().detach().numpy()
  gt = gt.cpu().detach().numpy()
  pred = pred.cpu().detach().numpy()
  plt.imshow(img)
  title = 'GT - ' + str(gt) + ' |||  Pred - ' + str(pred)
  plt.title(title)
  plt.show()
