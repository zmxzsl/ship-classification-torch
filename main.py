import warnings
import argparse
from torch import optim
from configs.config import DefaultConfigs 
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from models.model import *
from utils.utils import *
from utils.progress_bar import *


def evaluate(model_name, val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()
    val_progressor = ProgressBar(mode=model_name+" Val  ", epoch=epoch,
                                 total_epoch=config.epochs, model_name=model_name, total=len(val_loader))
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, (img, target) in enumerate(val_loader):
            val_progressor.current = idx
            img = Variable(img).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(img)
            loss = criterion(output, target)

            precision = accuracy(output, target, topk=(1, 5, 10))
            losses.update(loss.item(), img.size(0))
            top1.update(precision[0], img.size(0))
            top5.update(precision[1], img.size(0))
            top10.update(precision[2], img.size(0))

            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()

        val_progressor.done()
    return [losses.avg, top1.avg, top5.avg, top10.avg]

def run_train():
    model_name = args.model_name
    model = get_net(config, model_name)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = torch.load(os.path.join(config.best_models, "model_best.pth.tar"))
        start_epoch = checkpoint["epoch"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        top1_list_train = checkpoint["result_train"]
        top1_list_val = checkpoint["result_val"]
    else:
        start_epoch = 0
        best_precision1 = 0
        top1_list_train = []
        top1_list_val = []

    for folder in [config.path_weights, config.path_best_models, config.path_logs]:
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(folder + model_name):
            os.makedirs(folder + model_name)
        if not os.path.exists(folder + model_name + '/' + args.alpha):
            os.makedirs(folder + model_name + '/' + args.alpha)

    train_data_list = get_files(config.path_images, "train")
    val_data_list = get_files(config.path_images, "val")

    train_dataloader = DataLoader(ShipDataset(train_data_list, config, args, test=False), batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_fn1, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(ShipDataset(val_data_list, config, args, test=True), batch_size=config.batch_size * 2,
                                shuffle=True, collate_fn=collate_fn1, pin_memory=False, num_workers=4)
    #import ipdb;ipdb.set_trace()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    model.cuda()
    model.train()

    for epoch in range(start_epoch, config.epochs):
        train_progressor = ProgressBar(mode=model_name+" Train", epoch=epoch, total_epoch=config.epochs,
                                       model_name=model_name, total=len(train_dataloader))
        for idx, (img, target) in enumerate(train_dataloader):
          
            # forward
            train_progressor.current = idx
            model.train()
            img = Variable(img).float().cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(img)
            loss = criterion(output, target)

            precision1_train, _, _ = accuracy(output, target, topk=(1, 5, 10))
            train_losses.update(loss.item(), img.size(0))
            train_top1.update(precision1_train[0],img.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            train_progressor()

        #evaluate
        train_progressor.done()
        valid_loss = evaluate(model_name, val_dataloader, model, criterion, epoch)
        top1_list_train.append(evaluate(model_name, train_dataloader, model, criterion, epoch))
        top1_list_val.append(valid_loss)
        is_best = valid_loss[1] > best_precision1
        if is_best:
            best_precision1 = valid_loss[1]

        save_checkpoint({
                    "epoch": epoch + 1,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "best_precision": best_precision1,
                    "optimizer": optimizer.state_dict(),
                    "valid_loss": valid_loss,
                    "result_train": top1_list_train,
                    "result_val": top1_list_val,
        }, is_best, args.alpha, config)


if __name__ == "__main__":
    # set random.seed and cudnn performance
    config = DefaultConfigs()
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Train ship class network')
    parser.add_argument('--model_name', help='model directory', type=str, default='resnet50')
    parser.add_argument('--gpus', help='gpus id', type=str, default='0')
    parser.add_argument('--seed', help='random seed', type=int, default=888)
    parser.add_argument('--resume', help='load exist checkpoint', action='store_true')
    parser.add_argument('--train_mode', help='The way to train networks patterns', type=str, default='Vanilla')
    parser.add_argument('--alpha', help='Parameters for adjusting the highlighting level of key areas', type=str, default='02')
    parser.add_argument('--save_frequency', help='Frequency of saving network parameters during training', type=int, default=101)
    args = parser.parse_args()
    
    # Update parameters
    config.save_frequency = args.save_frequency
    config.model_name = args.model_name
    if args.train_mode != 'Vanilla':
        config.path_images = './data/add_info/{}/'.format(args.alpha) 
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_train()
