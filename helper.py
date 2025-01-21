import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import sys, pickle
from common_functions import *
import pandas as pd
import torch.nn.functional as F
from collections import OrderedDict
from easydict import EasyDict

# Some initializations
torch.backends.cudnn.enabled = False
random.seed(32)
np.random.seed(32)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
os.environ['CUDA_VISIBLE_DEVICES']= '1'

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        raise NotImplementedError()

    def close(self):
        self.hook.remove()

class AppendHook(Hook):
    def __init__(self, module):
        super(self.__class__, self).__init__(module)
        self.current_activations = []

    def hook_fn(self, module, input, output):
        assert (torch.sum(output < 0) == 0)  # make sure its ReLu output
        self.current_activations.append(output.clone().detach().cpu().numpy())

    def close(self):
        del self.current_activations
        super().close()

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        # For observed provenances
        self.dynamic_fc_layers = []
        self.derived_models = {}

        # For observed values
        self.hooks = collections.OrderedDict()

    ## Functions for value invaraints
    def register_hooks(self, hook_cls, file=None, data_size=None):
        self.hooks = collections.OrderedDict()
        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                layer_name = name.split('.')[0]
                if hook_cls == AppendHook:
                    self.hooks[layer_name] = hook_cls(module)
                # print ('Regsitered hook ', layer_name, ' ', type(module))

    def unregister_hooks(self):
        for name, hook in self.hooks.items():
            hook.close()
            del hook
        del self.hooks

    ## Functions for prov features
    def op_hook_fn(self, module, input, output):
        out_features = np.prod(output.shape[1:])
        # Condition is to avoid the last layer which is already present
        if list(self._modules.values())[-1].in_features != out_features:
            new_fc_layer = nn.Linear(out_features, self.out_channels)
            new_reshape_layer = Reshape(-1, out_features)
            new_fc_module = nn.Sequential(new_reshape_layer, new_fc_layer)
            if list(self.parameters())[0].is_cuda:
                new_fc_module = new_fc_module.cuda()
            # print ('New layer registered ', str(new_fc_module))
            self.dynamic_fc_layers.append(new_fc_module)
        else:
            # the condition is not needed when layer layer is not seq
            # print('No new layer registered for the second last layer')
            pass

    # Creates a new layer for each sequential layer expect the last layer
    def generate_reduced_models(self, input_size):
        temp_hooks = []
        layer_names = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Sequential):
                layer_names.append('RM_' + name)
                temp_hooks.append(module.register_forward_hook(self.op_hook_fn))
        temp_inp = torch.randn(10, *input_size)
        if list(self.parameters())[0].is_cuda:
            temp_inp = temp_inp.cuda()
        self.forward(temp_inp)
        for hook in temp_hooks:
            hook.remove()

        # Create reduced models for all but last one which is the true model
        reduced_models = collections.OrderedDict()
        optimizers = collections.OrderedDict()
        module_list = list(self._modules.values())
        for layer_id in range(len(module_list) - 2):
            included_layers = module_list[:layer_id + 1]
            dynamic_fc_layer_i = self.dynamic_fc_layers[layer_id]
            reduced_model_i = nn.Sequential(*included_layers, dynamic_fc_layer_i)
            reduced_models[layer_names[layer_id]] = reduced_model_i
            optimizer = torch.optim.Adam(dynamic_fc_layer_i.parameters(), lr=1e-3)
            optimizers[layer_names[layer_id]] = optimizer
        self.derived_models = reduced_models
        self.derived_optimizers = optimizers
        # reduced_modeldict = nn.ModuleDict(reduced_models)
        # return nn.ModuleList(self.dynamic_fc_layers)


class CIFAR10_SMALLCNN(BaseModel):
    def __init__(self):
        super(CIFAR10_SMALLCNN, self).__init__()
        self.input_size = [3, 32, 32]
        self.out_channels = 10

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
            Reshape(-1, 4096),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(256, self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc3(x)
        return x

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class TrainableModel(torch.nn.Module):
    def __init__(self, FLAGS, data_loader, nn_model, model_tags=[]):
        super(TrainableModel, self).__init__()
        self.FLAGS = FLAGS
        self.net = nn_model
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=FLAGS.lr)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = data_loader

        # Setup RC Layer
        self.new_fc_layer = nn.Linear(256, 10)
        self.rc_optimizer = torch.optim.Adam(self.new_fc_layer.parameters(), lr=1e-3)

    def saveMyself(self):
        save_model([self.__class__], [self], 0, os.path.join(os.getcwd(), 'saved_model'))

    def loadMyself(self, epoch=0):
        load_model([self.__class__], [self], epoch, os.path.join(os.getcwd(), 'saved_model'))

    def trainMyself(self, start_epoch=1):
        data = self.data
        device = self.device
        net = self.net
        optimizer = self.optimizer
        loss_fn = self.loss_fn

        print(f'Training on {data.train.dataset.__len__()} samples')
        # Train vanilla model
        train_losses = []
        test_losses = []
        timer = Timer()
        for epoch in range(start_epoch, self.FLAGS.nb_epochs + 1):
            epoch_time = timer()
            train_loss = 0.0
            test_loss = 0.0
            train_corr = 0
            test_corr = 0
            net.train()
            for x, y in data.train:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = net(x)
                _, y_pred = logits.max(1)
                train_corr += torch.sum(y_pred == y).cpu().data.numpy()
                loss = loss_fn(logits, y)
                loss.mean().backward()
                optimizer.step()
                train_loss += loss.cpu().data.numpy().sum()

            net.eval()
            with torch.no_grad():
                for x, y in data.test:
                    x, y = x.to(device), y.to(device)
                    logits = net(x)
                    _, y_pred = logits.max(1)
                    test_corr += torch.sum(y_pred == y).cpu().data.numpy()
                    loss = loss_fn(logits, y)
                    test_loss += loss.cpu().data.numpy().sum()

            train_loss = train_loss / data.train.dataset.__len__()
            test_loss = test_loss / data.test.dataset.__len__()
            train_acc = train_corr / data.train.dataset.__len__()
            test_acc = test_corr / data.test.dataset.__len__()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print('Epoch: {}/{}, Train: L({:.3f}) A({:.2f}), Test: L({:.3f}) A({:.2f}), Time(s): PE({:.2f}) EL({:.2f})'.format(epoch, self.FLAGS.nb_epochs, train_loss, train_acc, test_loss, test_acc, timer() - epoch_time, timer()))
        self.losses = np.array([train_losses, test_losses]).T

        net.eval()
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        with torch.no_grad():
            for x, y in data.test:
                x, y = x.to(device), y.to(device)
                _, y_pred = net(x).max(1)  # model prediction on clean examples
                report.nb_test += y.size(0)
                report.correct += y_pred.eq(y).sum().item()
        print('Model Acc on Test (%): {:.3f}'.format(report.correct / report.nb_test * 100.))

    def get_rcnet(self):
        module_list = list(self.net._modules.values())
        included_layers = module_list[:-1]
        rc_net = nn.Sequential(*included_layers, self.new_fc_layer)
        return rc_net

    def train_RCLayer(self):
        data = self.data
        device = self.device
        loss_fn = self.loss_fn

        # Setup RC Network
        rc_net = self.get_rcnet()

        # Freeze all the layers expect the new layers.
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        self.net.eval()

        print(f'Training RCLayer on {data.train.dataset.__len__()} samples')
        for epoch in range(1, self.FLAGS.rc_epochs+1):
            train_loss = []
            test_loss = []

            self.new_fc_layer.train()
            for x, y in data.train:
                x, y = x.to(device), y.to(device)
                self.rc_optimizer.zero_grad()
                loss = loss_fn(rc_net(x), y)
                loss.mean().backward()
                self.rc_optimizer.step()
                # train_loss += [loss.cpu().data.numpy()]
                train_loss += list(loss.cpu().data.numpy())

            rc_net.eval()
            with torch.no_grad():
                for x, y in data.test:
                    x, y = x.to(device), y.to(device)
                    loss = loss_fn(rc_net(x), y)
                    # test_loss += [loss.cpu().data.numpy()]
                    test_loss += list(loss.cpu().data.numpy())

            train_loss = np.mean(train_loss)
            test_loss = np.mean(test_loss)

            print('Epoch: {}/{}, train loss: {:.3f}, test loss: {:.3f}'.format(epoch, self.FLAGS.rc_epochs,
                                                                               train_loss, test_loss))

        # evaluate
        rc_net.eval()
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        with torch.no_grad():
            for x, y in data.test:
                x, y = x.to(device), y.to(device)
                _, y_pred = rc_net(x).max(1)  # model prediction on clean examples
                report.nb_test += y.size(0)
                report.correct += y_pred.eq(y).sum().item()
        print('Derived model Acc on Test (%): {:.3f}'.format(report.correct / report.nb_test * 100.))

    def short_evaluate(self, dataloader):
        net = self.net
        rc_net = self.get_rcnet()
        device = self.device

        net.eval()
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                _, y_pred = net(x).max(1)  # model prediction on clean examples
                report.nb_test += y.size(0)
                report.correct += y_pred.eq(y).sum().item()
        print('Model Acc on Test (%): {:.3f}'.format(report.correct / report.nb_test * 100.))


        rc_net.eval()
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                _, y_pred = rc_net(x).max(1)  # model prediction on clean examples
                report.nb_test += y.size(0)
                report.correct += y_pred.eq(y).sum().item()
        print('Derived model Acc on Test (%): {:.3f}'.format(report.correct / report.nb_test * 100.))

    def evaluate(self, dataloader):
        net = self.net
        device = self.device
        net.eval()
        with torch.no_grad():
            y_preds = []
            y_true = []
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                _, y_pred = net(x).max(1)
                y_preds.append(y_pred.cpu().numpy())
                y_true.append(y.cpu().numpy())
            y_preds = np.concatenate(y_preds, 0)
            y_true = np.concatenate(y_true, 0)
            verdicts = y_preds == y_true
            acc = np.mean(verdicts)
            print(f'Overall accuracy {acc}')

            # compute class wise accuracy
            all_classes  = np.sort(np.unique(dataloader.dataset.targets))
            acc_dict = {}
            for class_i in all_classes:
                y_idx = y_true == class_i
                verdicts_i = verdicts[y_idx]
                acc_i = np.mean(verdicts_i)
                print(f'Acc for class {class_i} is {acc_i}')
                acc_dict[class_i] = acc_i
            return acc_dict

class ModelDataset():
    def __init__(self, model, dataset_source):
        self.model = model
        self.dataset_source = dataset_source
        self.rc_net = model.get_rcnet()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return self.dataset_source.__len__()

    def __getdim__(self):
        x, _ = self.__getitem__(1)
        return x.shape[0]

    def __getitem__(self, idx):
        raw_item, target = self.dataset_source.__getitem__(idx)

        with torch.no_grad():
            raw_item = raw_item.to(self.device).unsqueeze(0)
            logits_i = self.rc_net(raw_item)
            # probs = F.normalize(logits_i, dim=1)
            probs = F.softmax(logits_i, dim=1)
        return probs.squeeze(), target


class AE_INV_TINY(torch.nn.Module):
    def __init__(self, input_dim, train_loader, test_loader, epochs=20):
        super(self.__class__, self).__init__()

        # input is 28x28
        self.recognizer = nn.Sequential(nn.Linear(input_dim, 10),
                                        nn.ReLU(),
                                        nn.Linear(10, 9),
                                        nn.ReLU(),
                                        )
        self.generator = nn.Sequential(nn.Linear(9, 10),
                                        nn.ReLU(),
                                        nn.Linear(10, input_dim),
                                        # nn.Sigmoid(),
                                        )

        self.finalLosses = OrderedDict()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fn = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.epochs = epochs

    def saveMyself(self):
        save_model([self.__class__], [self], 0, os.path.join(os.getcwd(), 'ae_model'))

    def loadMyself(self):
        load_model([self.__class__], [self], 0, os.path.join(os.getcwd(), 'ae_model'))

    def forward(self, x):
        x = self.recognizer(x)
        x = self.generator(x)
        return x

    def train_myself(self):
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        device = self.device
        model = self
        train_loader = self.train_loader
        test_loader = self.test_loader

        meantrainloss = []
        meantestloss = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = []
            for x, _ in train_loader:
                x = x.to(device)
                pred = model(x)
                optimizer.zero_grad()
                loss = loss_fn(pred, x)
                loss.mean().backward()
                optimizer.step()
                train_loss += loss.detach().flatten(1).mean(1).cpu().numpy().tolist()

            model.eval()
            test_loss = []
            with torch.no_grad():
                for x, _ in test_loader:
                    # x = x[:, :10]
                    x = x.to(device)
                    pred = model(x)
                    loss = loss_fn(pred, x)
                    test_loss += loss.detach().flatten(1).mean(1).cpu().numpy().tolist()
            meantrainloss.append(np.mean(train_loss))
            meantestloss.append(np.mean(test_loss))
            print('Epoch: {}/{}, Train loss {:.3f}, Test loss: {:.3f}'.format(epoch, self.epochs, meantrainloss[-1], meantestloss[-1]))

        self.finalLosses['train'] = meantrainloss[-1]
        self.losses = np.array([meantrainloss, meantestloss]).T

    def compute_scores(self, infer_loader):
        loss_fn = self.loss_fn
        model = self
        device = self.device

        reconstructions = OrderedDict()
        test_losses = OrderedDict()
        model.eval()
        with torch.no_grad():
            reconstruction = []
            for x, _ in infer_loader:
                x = x.to(device)
                # x = x[:, :10]
                pred = model(x)
                loss = loss_fn(pred, x)
                reconstruction += loss.detach().flatten(1).mean(1).cpu().numpy().tolist()
            test_loss = np.mean(reconstruction)
            self.finalLosses['infer'] = test_loss
            # print('Test loss of ', loader_name, ' is ', test_loss)
            # reconstructions[loader_name] = reconstruction# np.array(reconstruction).reshape(1,-1)
        return reconstruction

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


    list1 = ','.join(map(str, list1 ))#+ ['T', len(data)]))
    list2 = ','.join(map(str, list2))
    if display:
        print(list1)
        print(list2)

    return list1, list2

class CIFAR_skewed(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None, name=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.targets = targets
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def get_data_array(self):
        results = []
        for i in range(self.__len__()):
            results.append(np.expand_dims(self.__getitem__(i)[0],0))
        results = np.concatenate(results, 0)
        results = np.reshape(results, (results.shape[0], -1))
        return results

class CIFAR10DataLoaders():
    def __init__(self, FLAGS):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.FLAGS = FLAGS
        batch_size = self.FLAGS.batch_size

        self.splitDataSigmoid()

        # hist(self.Ay_train)
        # hist(self.Ay_val)
        # hist(self.Ay_test)
        # hist(self.By_train)
        # hist(self.By_val)
        # hist(self.By_test)

        transform_train = self.transform_train
        transform_test = self.transform_test
        self.source_train = torch.utils.data.DataLoader(
            CIFAR_skewed(data=self.AX_train, targets=self.Ay_train, transform=transform_train, name='A_Train'),
            shuffle=True,
            batch_size=batch_size)
        self.source_test = torch.utils.data.DataLoader(
            CIFAR_skewed(data=self.AX_test, targets=self.Ay_test, transform=transform_test, name='A_Test'),
            shuffle=False,
            batch_size=batch_size)
        self.source_val = torch.utils.data.DataLoader(
            CIFAR_skewed(data=self.AX_val, targets=self.Ay_val, transform=transform_test, name='A_Val'),
            shuffle=False,
            batch_size=batch_size)
        self.shifted_train = torch.utils.data.DataLoader(
            CIFAR_skewed(data=self.BX_train, targets=self.By_train, transform=transform_train, name='B_Train'),
            shuffle=True,
            batch_size=batch_size)
        self.shifted_test = torch.utils.data.DataLoader(
            CIFAR_skewed(data=self.BX_test, targets=self.By_test, transform=transform_test, name='B_Test'),
            shuffle=False,
            batch_size=batch_size)
        self.shifted_val = torch.utils.data.DataLoader(
            CIFAR_skewed(data=self.BX_val, targets=self.By_val, transform=transform_test, name='B_Val'),
            shuffle=False,
            batch_size=batch_size)

    def splitDataSigmoid(self, sigma=0.4, origin=3.35):
        # data_dir = os.path.join(self.FLAGS.project_dir, 'data')
        self.root = self.FLAGS.data_dir
        self.base_folder = self.FLAGS.datadirname
        traintest_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]

        self.data = []
        self.targets = []

        for file_name, checksum in traintest_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)

        d = 0.6  # percentage split between source(d) and shifted(1-d)
        k = 2 / 3  # percentage split between val(k) and test(1-k)
        r = k / (1 + k)  # percentage of split between train(r) and rest(1-r)

        sigma = 0.4
        origin = 3.45 # original value before paper
        origin = 2.4 # for paper
        sig = lambda x: 1 / (1 + (np.exp(-(x - origin) / sigma)))
        sigvals = [sig(x) for x in range(10)]
        sigsum = np.sum(sigvals)
        alpha = 10 * d / sigsum
        norm_sigvals = [x * alpha for x in sigvals]

        proportion = {a: b for a, b in zip(range(10), norm_sigvals)}
        assert (np.sum(list(proportion.values())) - 10 * d < 0.1)
        print('P:', [round(proportion[i], 2) for i in range(10)])

        source_idx = []
        shifted_idx = []
        for el in np.sort(np.unique(self.targets)):
            idxs = np.where(self.targets == el)[0]
            src, shf = train_test_split(idxs, train_size=proportion[el])
            source_idx += src.tolist()
            shifted_idx += shf.tolist()

        source_labels = self.targets[source_idx]
        source_data = self.data[source_idx]
        shifted_labels = self.targets[shifted_idx]
        shifted_data = self.data[shifted_idx]

        self.AX_train, AX_rest, self.Ay_train, Ay_rest = train_test_split(source_data, source_labels, train_size=r,
                                                                          random_state=42)
        self.AX_val, self.AX_test, self.Ay_val, self.Ay_test = train_test_split(AX_rest, Ay_rest, train_size=k,
                                                                                random_state=42)
        self.BX_train, BX_rest, self.By_train, By_rest = train_test_split(shifted_data, shifted_labels,
                                                                          train_size=r,
                                                                          random_state=42)
        self.BX_val, self.BX_test, self.By_val, self.By_test = train_test_split(BX_rest, By_rest, train_size=k,
                                                                                random_state=42)

    def combine(self, data1, data2):
        shuffle = isinstance(data1.sampler, torch.utils.data.sampler.RandomSampler)
        transform = self.transform_train if shuffle == True else self.transform_test
        data_combined = np.concatenate([data1.dataset.data, data2.dataset.data], 0)
        targets_combined = np.concatenate([data1.dataset.targets, data2.dataset.targets], 0)
        datalord = torch.utils.data.DataLoader(
            CIFAR_skewed(data=data_combined, targets=targets_combined, transform=transform),
            shuffle=shuffle,
            batch_size=self.FLAGS.batch_size)
        return datalord

    def subsample(self, data, size):
        shuffle = isinstance(data.sampler, torch.utils.data.sampler.RandomSampler)
        transform = self.transform_train if shuffle == True else self.transform_test
        data_orig = data.dataset.data
        data_size = data_orig.shape[0]
        idx = np.random.choice(data_size, size=size, replace=False)
        data_subsampled = data_orig[idx]
        targets_subsampled = data.dataset.targets[idx]
        datalord = torch.utils.data.DataLoader(
            CIFAR_skewed(data=data_subsampled, targets=targets_subsampled, transform=transform),
            shuffle=shuffle,
            batch_size=self.FLAGS.batch_size)
        return datalord

    def np_subsample(self, data, size, file_name='auto_enc_idx.npy'):
        file = os.path.join(self.FLAGS.save_dir, file_name)
        print(f'Loading np file idx: {file}')
        idx = np.load(file)[:size]
        shuffle = isinstance(data.sampler, torch.utils.data.sampler.RandomSampler)
        transform = self.transform_train if shuffle == True else self.transform_test
        data_subsampled = data.dataset.data[idx]
        targets_subsampled = data.dataset.targets[idx]
        datalord = torch.utils.data.DataLoader(
            CIFAR_skewed(data=data_subsampled, targets=targets_subsampled, transform=transform),
            shuffle=shuffle,
            batch_size=self.FLAGS.batch_size)
        return datalord

    def changeSampler(self, dataloader_ins, shuffle):
        data = dataloader_ins.dataset.data
        targets = dataloader_ins.dataset.targets
        transform = self.transform_train if shuffle == True else self.transform_test
        datalord = torch.utils.data.DataLoader(CIFAR_skewed(data=data, targets=targets, transform=transform),
                                               shuffle=shuffle,
                                               batch_size=self.FLAGS.batch_size)
        return datalord
