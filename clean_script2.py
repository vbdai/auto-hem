from helper import *
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

# Some initializations
torch.backends.cudnn.enabled = False
random.seed(32)
np.random.seed(32)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
os.environ['CUDA_VISIBLE_DEVICES']= '1'

# Flags
FLAGS = EasyDict()
FLAGS.data_dir = '<path_to_cifar>'
FLAGS.datadirname = '<cifar_dir>'
FLAGS.nb_epochs = 20
FLAGS.batch_size = 64
FLAGS.lr = 1e-3

#%% DATALAODERS
dataloader = CIFAR10DataLoaders(FLAGS)
rl_test = dataloader.combine(dataloader.source_test, dataloader.shifted_test)
rl_input = dataloader.combine(dataloader.source_val, dataloader.shifted_train)
print('Distributions: Train, Pool, Test')
hist(dataloader.source_train.dataset.targets)
hist(rl_input.dataset.targets)
hist(rl_test.dataset.targets)

#%% TARGET MODEL
nn_model = CIFAR10_SMALLCNN()
source_dataloader = EasyDict(train=dataloader.source_train, test=rl_test)
model = TrainableModel(FLAGS, source_dataloader, nn_model)
if model.device == 'cuda':
    model = model.cuda()
# model.trainMyself(start_epoch=1)
# model.train_RCLayer()
# model.saveMyself()
model.loadMyself(epoch=1)
# model.evaluate(rl_test)
model.short_evaluate(rl_test)

#%% AUTOENCODER
print('AUTOENCODER TRAINING STARTED')
train_dataset = ModelDataset(model, dataloader.source_train.dataset)
test_dataset = ModelDataset(model, dataloader.source_val.dataset)
infer_dataset = ModelDataset(model, rl_input.dataset)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=FLAGS.batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=FLAGS.batch_size)
infer_loader = DataLoader(infer_dataset, shuffle=False, batch_size=FLAGS.batch_size)

ae_model = AE_INV_TINY(train_dataset.__getdim__(), train_loader, test_loader, epochs=15)
if ae_model.device == 'cuda':
    ae_model = ae_model.cuda()
ae_model.train_myself()
# ae_model.saveMyself()
scores = ae_model.compute_scores(infer_loader)
ranks = np.argsort(scores)[::-1]
gts = rl_input.dataset.targets
anno_classes  = [0,1]
sample_counts = [50, 100, 500, 1000,2000,3000,4000,5000,6000,7000,8000,9000]
print(ae_model.getStats(sample_counts, anno_classes, ranks, gts))

exit()
