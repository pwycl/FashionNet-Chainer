import argparse
import chainer
from chainer.training import extensions
from chainer.training import updaters
from DeepfashionDatasets import DeepFashionLandmarkDatasets
from fashionNet import FashionNet

def main():
    parser=argparse.ArgumentParser(
        description='training FashionNet from your dataset in deepfashion format')
    parser.add_argument('train_catogory')
    parser.add_argument('train_landmark')
    parser.add_argument('val_catogory')
    parser.add_argument('val_landmark')
    parser.add_argument('--epoch','-E',type=int,default=10)
    parser.add_argument('--root','-R')
    parser.add_argument('--batch_size','-B',type=int,default=32)
    parser.add_argument('--out','-o',default='result')
    args=parser.parse_args()

    model=FashionNet()
    model.setTrain_1()

    train=DeepFashionLandmarkDatasets(
        args.train_catogory,args.train_landmark,args.root)
    val=DeepFashionLandmarkDatasets(
        args.val_catogory,args.val_landmark,args.root)

    train_iter=chainer.iterators.SerialIterator(train,args.batch_size)
    val_iter=chainer.iterators.SerialIterator(val,args.batch_size)

    optimizer=chainer.optimizers.MomentumSGD(lr=0.01,momentum=0.9)
    optimizer.setup(model)

    updater=chainer.training.StandardUpdater(train_iter,optimizer)

    trainer=chainer.training.Trainer(updater,(args.epoch,'epoch'),out=args.out)

    val_interval=100,'iteration'
    log_interval=100,'iteration'

    trainer.extend(extensions.Evaluator(val_iter,model),trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(),trigger=val_interval)
    trainer.exend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]),trigger=log_interval)

    trainer.run()

if __name__=='__main__':
    main()

