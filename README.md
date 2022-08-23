<h1 align="center"><b>GSAM Optimizer</b></h1>
<h3 align="center"><b>Surrogate Gap Guided Sharpness-Aware Minimization </b></h3>
<a href="https://openreview.net/pdf?id=edONMAnhLu-" a> [Paper, ICLR 2022]</a>   <a href="https://sites.google.com/view/gsam-iclr22/home" a> [Webpate] </a>

Official implementation in jax: https://github.com/google-research/big_vision/tree/main/big_vision/trainers/proj/gsam 

Discussion on reproducing results with jax code: https://github.com/google-research/big_vision/pull/8

Disclaimer: This repository is a re-implmentation in PyTorch tested only on a Cifar10 experiment, not tested by reproduction of results in the paper.

Acknowledgement: This repository is based on https://github.com/davda54/sam 

## (Potentially) Unresolved issues with PyTorch code on per-worker unsynchronized gradient and weight perturbation
Since the SAM family works best when```each worker has its own (different) gradient and weight perturbation```, but in DataParallel mode in PyTorch the gradient is synchronized across workers hence perturbation is also synchronized across workers.

In order to let each worker use its own gradient, I use ```model.no_sync()``` in the [code](https://github.com/juntang-zhuang/GSAM/blob/bab6bbe65612d6080f522870b9dd79f5957882d6/gsam/gsam.py#L160), perform the gradient decomposition in GSAM for each worker separately, then synchronize the <img src="https://latex.codecogs.com/svg.latex?&space;\nabla{f}^{GSAM}" />
 [here](https://github.com/juntang-zhuang/GSAM/blob/bab6bbe65612d6080f522870b9dd79f5957882d6/gsam/gsam.py#L180) before feeding it to the base optimizer. However, I'm not sure if ```model.no_sync()``` only works in ```DistributedDataParallel``` mode but not in ```DataParallel``` mode. 

I suppose the training script needs to be set as ```DistributedDataParallel``` in order to replicate my experiments with Jax, but I have quite limited experimence with PyTorch distributed training.
Please feel free to create a PR if you are an expert on this.

## Algorithm and results on ImageNet in the paper
![](img/gsam_algo2.png)
![](img/gsam_results.png)
![](img/gsam_hyperparam2.png)
## How to use GSAM in code
For readability the essential code is highlighted (at a cost of an extra "+" sign at the beginning of line). Please ***remove the beginning "+"*** when using GSAM in your project. Each step of code is marked with notes, please read before using.

```diff
# import GSAM class and scheduler
from gsam import GSAM, LinearScheduler

# Step 0): set up base optimizer, e.g. SGD, Adam, AdaBelief ...
+base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

# Step 1): set up learning rate scheduler. See [below](https://github.com/juntang-zhuang/GSAM/edit/main/README.md#notes-on-rho_scheduler)
# If you pass base_optimizer to lr_scheduler, lr_scheduler.step() will update lr for all trainable parameters in base_optimizer. 
# Otherwise, it only returns the value, and you need to manually assign lr to parameters in base_optimizer.
# Currently LinearScheduler, CosineScheduler and PolyScheduler are re-implemented, all have support for warmup and user-specified min value.
# You can also use torch.optim.lr_scheduler to adjust learning rate, however, in this case, it's recommended to use ProportionScheduler for rho_t.

+lr_scheduler = LinearScheduler(T_max=args.epochs*len(dataset.train), max_value=args.learning_rate, min_value=args.learning_rate*0.01, optimizer=base_optimizer)

# Step 2): set up rho_t scheduler. 
# There are two ways to set up rho_t decays proportional to lr, e.g. (lr - lr_min) / (lr_max - lr_min) = (rho - rho_min) / (rho_max - rho_min)
#
# Method a), call same scheduler twice with different ```max_value``` and ```min_value```:
#            lr_scheduler = CosineScheduler(T_max=args.epochs*len(dataset.train), max_value=args.learning_rate, min_value=args.learning_rate*0.01, optimizer=base_optimizer)
#            rho_scheduler = CosineScheduler(T_max=args.epochs*len(dataset.train), max_value=args.rho_max, min_value=args.rho_min)
#
# Method b), call the ```ProportionScheduler``` class:
#            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False)
#            rho_scheduler = ProportionScheduler(lr_scheduler, max_lr=args.learning_rate, min_lr=args.min_lr, max_value=args.rho_max, min_value=args.rho_min)
+rho_scheduler = LinearScheduler(T_max=args.epochs*len(dataset.train), max_value=args.rho_max, min_value=args.rho_min)

# Step 3): configure GSAM
+gsam_optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)

# ============================================================================================
# training loop

for batch in dataset.train:
    inputs, targets = (b.cuda() for b in batch)
        
    # Step 4): Define loss function, so that loss_fn only takes two inputs (predictions, targets), and outputs a scalar valued loss.
    # If you have auxialliary parameters e.g. arg1, arg2, arg3 ..., please define as:
    #           criterion = nn.CrossEntropyLoss()
    #           loss_fn = lambda predictions, targets: criterion(predictions, targets, arg1=arg1, arg2=arg2, arg3=arg3 ...)
    
+   def loss_fn(predictions, targets):
+       return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()
    
    # Step 5): Set closure, GSAM automatically sets the closure as
    #            predictions = model(inputs), loss = loss_fn(predictions, targets), loss.backward()
    # Note: need to set_closure for each (inputs, targets) pair
    
+   gsam_optimizer.set_closure(loss_fn, inputs, targets)
    
    # Step 6): Update model parameters. 
    # optimizer.step() internally does the following: 
    #            (a) zero grad (b) get gradients (c) get rho_t from rho_scheduler (d) perturb weights (e) zero grad (f) get gradients at perturbed location
    #            (g) decompose gradients and update gradients (h) apply new gradients with base_optimizer
    # Note: zero_grad is called internally for every step of GSAM.step(), gradient accumulation is currently not supported
    
+   predictions, loss = gsam_optimizer.step()

    # Step 7): Upate lr and rho_t
+   lr_scheduler.step()
+   gsam_optimizer.update_rho_t()
# ============================================================================================
```

# Notes on rho_scheduler

### Case 1 (same as in paper for ImageNet experiments):
If you use the same type for lr_scheduler and rho_scheduler, it's equivalent to let rho_t evolves proportionally with 
learning rate, 
```(lr - lr_min) / (lr_max - lr_min) = (rho - rho_min) / (rho_max - rho_min)```

Example to use the same type of scheduler for rho and lr:
#### Method 1.1) Call the same Scheduler class twice for lr_scheduler and rho_scheduler
```
from gsam.scheduler import LinearScheduler
lr_scheduler = LinearScheduler(T_max=args.epochs*len(dataset.train), max_value=args.learning_rate, min_value=args.learning_rate*0.01, optimizer=base_optimizer, warmup_step=2000)
rho_scheduler = LinearScheduler(T_max=args.epochs*len(dataset.train), max_value=args.rho_max, min_value=args.rho_min, warmup_step=2000)
```
```
from gsam.scheduler import CosineScheduler
lr_scheduler = CosineScheduler(T_max=args.epochs*len(dataset.train), max_value=args.learning_rate, min_value=args.learning_rate*0.01, optimizer=base_optimizer, warmup_step=2000)
rho_scheduler = CosineScheduler(T_max=args.epochs*len(dataset.train), max_value=args.rho_max, min_value=args.rho_min, warmup_step=2000)
```
#### Method 1.2) Create an lr_scheduler from torch.optim.lr_scheduler, then call gsam.scheduler.ProportionScheduler
```
from torch.optim.lr_scheduler import CosineAnnealingLR
from gsam.scheduler import ProportionScheduler
base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
lr_scheduler = CosineAnnealingLR(optimizer=base_optimizer, T_max=args.epochs*len(dataset.train), eta_min=args.learning_rate*0.01)
rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.learning_rate, min_lr=args.learning_rate*0.01, max_value=args.rho_max, min_value=args.rho_min)
```

### Case 2 (write your own scheduler)
1) You can also write your own shceduler by inherit ```gsam.scheduler.SchedulerBase``` class and define ```step_func```.
2) You can write your own lr scheduler by inheriting ```torch.optim.lr_scheduler._LRScheduler```, or combining several schedulers using ```torch.optim.lr_scheduler.SequentialLR```. After creating your own lr_scheduler, call ```gsam.ProportionScheduler``` to create ```rho_scheduler```.

## Citation
```
@inproceedings{
zhuang2022surrogate,
title={Surrogate Gap Minimization Improves Sharpness-Aware Training},
author={Juntang Zhuang and Boqing Gong and Liangzhe Yuan and Yin Cui and Hartwig Adam and Nicha C Dvornek and sekhar tatikonda and James s Duncan and Ting Liu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=edONMAnhLu-}
}
```
