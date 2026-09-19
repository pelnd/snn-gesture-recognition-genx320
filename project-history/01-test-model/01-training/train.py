"""
Trains SpikingJelly's DVSGestureNet on the DVS128Gesture dataset.
Runs a training loop with periodic test-set evaluation, and saves
checkpoints (latest + best by test accuracy) after each epoch.

This is the initial training run, with reduced settings to keep it fast. 
32 channels, 8 timesteps, batch size 2, 2 epochs
"""


import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DATA_DIR = '../../../data/DVS128Gesture'
DEVICE = 'cpu'
T = 8               # number of simulated timesteps per sample
BATCH_SIZE = 2
CHANNELS = 32       # width of the conv layers in the network
EPOCHS = 2
LR = 1e-3
NUM_CLASSES = 11    # DVS128 Gesture has 11 gesture classes
CHECKPOINT_DIR = '../02-weights-and-logs'
RESUME_PATH = None  # set to '../02-weights-and-logs/checkpoint_latest.pth' to resume
PRINT_EVERY = 20 

NUM_WORKERS = 0  # >0 causes segfaults for me


def main():
    net = parametric_lif_net.DVSGestureNet(
        channels=  CHANNELS,                    # How many feature maps we produce, by how many kernels we slide
        spiking_neuron= neuron.LIFNode,         # Neuron model
        surrogate_function = surrogate.ATan(),  # Surrogate function
        detach_reset = True
    )

    functional.set_step_mode(net, 'm')  # This means network can receive all timesteps at once

    net.to(DEVICE) # actully default is cpu anyway

    # Training and set split is predefined in the dataset, here we split our event data into T timesteps, each timestep having equal number of events
    train_set = DVS128Gesture(DATA_DIR, train=True, data_type='frame', frames_number=T, split_by='number')
    test_set = DVS128Gesture(DATA_DIR, train=False, data_type='frame', frames_number=T, split_by='number')

    # Preparing batches, drop_last ensures each training batch has same size
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers= NUM_WORKERS, pin_memory= True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers= NUM_WORKERS, pin_memory= True)

    # optimizer that will update your network's weights during training
    # This uses Adam algorithm, net.parameters() are params to be updated, lr (learning rate) is how big steps to take when learning
    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)

    # to decrease learning rate as we move
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  #create dir for checkpoints
    max_test_acc = -1  #track best checkpoint

    # if resumed
    start_epoch = 0
    if RESUME_PATH is not None:
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print(f'resumed from {RESUME_PATH}, starting at epoch {start_epoch}')


    #Training Loop
    for epoch in range(start_epoch, EPOCHS):

        net.train()  #training mode on, for good practices
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (frame, label) in enumerate(train_loader):  #so for each batch [N, T, C, H, W]

            # moving data to whatever device as well
            frame = frame.to(DEVICE).transpose(0, 1)  # multi-step mode expects [T, N, C, H, W]
            label = label.to(DEVICE)
            label_onehot = F.one_hot(label, NUM_CLASSES).float() # one-hotting labels

            optimizer.zero_grad() # clears gradients from previous batch, so they dont accumulate

            # this is the whole forward pass in one line
            # the output is firing rates time averaged for each of the classes
            # so with batch_size =2, num_classes = 11; output has the shape [2, 11] ( before averaging: [T, 2, 11])
            # predictions based on the firing rate -> rate coding, more spikes means stronger signal
            out_firing_rate = net(frame).mean(0)

            loss = F.mse_loss(out_firing_rate, label_onehot) #loss [2,11], tensor-like object with connection to the computations

            # we follow the graph backwards, calculate gradients, and store them as model parameters in net.parameters()
            loss.backward() 
            optimizer.step()  #updates weights based on the gradients

            train_total += label.numel()   #basically batch size here
            train_loss += loss.item() * label.numel()   #loss is averaged so makes sense to multiply, .item() makes it a number
            train_correct += (out_firing_rate.argmax(1) == label).sum().item()

            functional.reset_net(net)  # our manual voltage reset

            if batch_idx % PRINT_EVERY == 0:
                print(f'  epoch {epoch}, batch {batch_idx}, '    #print progress
                      f'running loss={loss.item():.4f}')

        train_loss /= train_total
        train_acc = train_correct / train_total

        scheduler.step()  # step the scheduler once per epoch

        # Test
        net.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():  # says no need to track gradients

            for frame, label in test_loader:
                frame = frame.to(DEVICE).transpose(0, 1)
                label = label.to(DEVICE)
                label_onehot = F.one_hot(label, NUM_CLASSES).float()
 
                out_firing_rate = net(frame).mean(0)
                loss = F.mse_loss(out_firing_rate, label_onehot)
 
                test_total += label.numel()
                test_loss += loss.item() * label.numel()
                test_correct += (out_firing_rate.argmax(1) == label).sum().item()
                functional.reset_net(net)  # same reason as above

        test_loss /= test_total
        test_acc = test_correct / test_total
 
        print(f'epoch={epoch}  '
              f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
              f'test_loss={test_loss:.4f}  test_acc={test_acc:.4f}')

        #checkpointing
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc,
        }

        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth'))

        # checkpoint the best
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint['max_test_acc'] = max_test_acc
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pth'))
            print(f'  -> new best test_acc={max_test_acc:.4f}, saved checkpoint_best.pth')


if __name__ == '__main__':
    main()
