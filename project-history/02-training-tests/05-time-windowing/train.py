"""
Trains SpikingJelly's DVSGestureNet on the DVS128Gesture dataset.
Runs a training loop with periodic test-set evaluation, saving checkpoints
(latest + best by test accuracy), a training curve log (CSV), and the run's
hyperparameters (JSON) after each epoch.

Time-based windowing variant: instead of the eventcount runs' fixed- or
adaptive-event-count frames, this splits each recording into fixed 125ms
frames (split_by='time', duration=125,000), then slices non-overlapping
T=8-frame windows (SlidingWindowDataset). No augmentation, no weight decay.

Adapted from SpikingJelly's classify_dvsg.py example
(https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/classify_dvsg.py),
licensed under the Qǐzhì Open Source License 1.0 (see third-party-licenses/).
"""

import os
import csv
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DATA_DIR = '../../../data/DVS128Gesture'
DEVICE = 'cuda'
T = 8               # number of simulated timesteps per sample
DURATION = 125_000
BATCH_SIZE = 16
CHANNELS = 64       # width of the conv layers in the network
EPOCHS = 64
LR = 1e-3
NUM_CLASSES = 11    # DVS128 Gesture has 11 gesture classes

RUN_NAME = 'time_windowing'
CHECKPOINT_DIR = './weights-and-logs'
RESUME_PATH = None  # set to './weights-and-logs/checkpoint_latest.pth' to resume
PRINT_EVERY = 100 

NUM_WORKERS = 0

class SlidingWindowDataset(torch.utils.data.Dataset):
    # Slices Dataset into T-frame windows

    def __init__(self, base_dataset, T=8, stride=None):
        self.base = base_dataset
        self.T = T
        self.stride = stride if stride is not None else T
        self.index = []  # (recording_idx, start_frame)
        for i in range(len(base_dataset)):
            frames, _ = base_dataset[i]        # [N, C, H, W], N variable
            n = frames.shape[0]
            for start in range(0, n - T + 1, self.stride):
                self.index.append((i, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec_idx, start = self.index[idx]
        frames, label = self.base[rec_idx]
        return frames[start:start + self.T], label


def main():

    net = parametric_lif_net.DVSGestureNet(
        channels=  CHANNELS,                    # How many feature maps we produce, by how many kernels we slide
        spiking_neuron= neuron.LIFNode,         # Neuron model
        surrogate_function = surrogate.ATan(),  # Surrogate function
        detach_reset = True
    )

    functional.set_step_mode(net, 'm')  # This means network can receive all timesteps at once

    net.to(DEVICE) 

    # Training and set split is predefined in the dataset, here we split our event data into 125 ms frames, then wrap to get T=8
    train_base = DVS128Gesture(DATA_DIR, train=True, data_type='frame', split_by='time', duration=DURATION)
    test_base = DVS128Gesture(DATA_DIR, train=False, data_type='frame', split_by='time', duration=DURATION)

    train_set = SlidingWindowDataset(train_base, T=T)
    test_set = SlidingWindowDataset(test_base, T=T)

    print(f'train samples: {len(train_set)}  (from {len(train_base)} recordings)')
    print(f'test samples: {len(test_set)}  (from {len(test_base)} recordings)')

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

    # hyperparameter logging
    with open(os.path.join(CHECKPOINT_DIR, 'run_config.json'), 'w') as f:
        json.dump({'T': T, 'duration': DURATION, 'channels': CHANNELS, 'batch_size': BATCH_SIZE,
               'lr': LR, 'epochs': EPOCHS, 'run_name': RUN_NAME}, f, indent=2)
    
    # training log
    log_path = os.path.join(CHECKPOINT_DIR, 'training_log.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

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

        functional.reset_net(net)

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
            # so with batch_size =n, num_classes = 11; output has the shape [n, 11] ( before averaging: [T, n, 11])
            # predictions based on the firing rate -> rate coding, more spikes means stronger signal
            out_firing_rate = net(frame).mean(0)

            loss = F.mse_loss(out_firing_rate, label_onehot) #loss [n,11], tensor-like object with connection to the computations

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

        

        log_writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        log_file.flush()

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


    log_file.close()

if __name__ == '__main__':
    main()
