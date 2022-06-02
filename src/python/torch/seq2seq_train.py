import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from util import *

class Seq2Seq_Train:
    def __init__(self,
                encoder,
                decoder,
                input_tensor,
                target_tensor,
                n_epochs,
                target_len,
                batch_size,
                learning_rate=0.01,
                opt_alg='adam',
                print_step=1,
                strategy = 'recursive',
                teacher_forcing_ratio=0.5,
                device='cpu',
                loss_figure_name='loss.png'):

        self.encoder = encoder
        self.decoder = decoder
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.n_epochs = n_epochs
        self.target_len = target_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt_alg = opt_alg
        self.print_step = print_step
        self.strategy = strategy
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.loss_figure_name = loss_figure_name

    def start_train(self):
        print('>>> Start training... (be patient: training time varies)')
        if self.strategy == 'recursive':
            self.train_model_recursive()

        elif self.strategy == 'teacher_forcing':
            self.train_model_teacher_forcing()

        elif self.strategy == 'mixed':
            self.train_model_mixed()

        else:
            assert False, "Please select one of them---[recursive, teacher_forcing, mixed]!"

        print('>>> Finish training!')

    def train_model_recursive(self):

        ### move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        ### get the learnable parameters
        params = []
        params += [x for x in self.encoder.parameters()]
        params += [x for x in self.decoder.parameters()]

        ### define optimizer method
        if self.opt_alg.upper() == 'ADAM':
            optimizer = optim.Adam(params=params, lr=self.learning_rate)
        elif self.opt_alg.upper() == 'SGD':
            optimizer = optim.SGD(params=params, lr=self.learning_rate)
        else:
            assert False, 'This version only supports ADAM and SGD!'

        ### define loss function
        criterion = nn.MSELoss()

        ### calculate number of batch iterations
        n_batches = int(self.input_tensor.shape[1] / self.batch_size)
        ### save loss
        losses = []

        for epoch in range(self.n_epochs):
            self.encoder.train()
            self.decoder.train()
            batch_loss = []

            for batch in range(n_batches):
                # select data
                input_batch = self.input_tensor[:, batch: batch + self.batch_size, :]

                # target_batch_input means the "luminosity delta", which is given to us
                # we will combine this information with "calibration" as input to decoder each time
                target_batch_input = self.target_tensor[:, batch: batch + self.batch_size, 0:1]

                # target_batch means the "calibration", which is the value we want to predict
                # so target_batch is the real target
                target_batch = self.target_tensor[:, batch: batch + self.batch_size, 1:]

                # move data to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                target_batch_input = target_batch_input.to(self.device)

                # outputs tensor
                outputs = torch.zeros(self.target_len, self.batch_size, target_batch.shape[2])

                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                # different training strategies
                # predict recursively
                # make prediction step by step
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    ### adding the other features
                    lumi_feature = target_batch_input[t, :, :]
                    decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                    decoder_input = decoder_output

                # compute the loss
                outputs = outputs.to(self.device)
                loss = criterion(outputs, target_batch)
                batch_loss.append(loss.item())

                # backpropagation
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(batch_loss)
            #print('>>>>>> {}/{} Epoch; Loss={}'.format(epoch, self.n_epochs, epoch_loss))
            losses.append(epoch_loss)

            ### we save its loss every print_step
            if epoch % self.print_step == 0:
                plot_loss(losses, self.loss_figure_name)
        show_loss(losses)

    def train_model_teacher_forcing(self):
        ### move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        ### get the learnable parameters
        params = []
        params += [x for x in self.encoder.parameters()]
        params += [x for x in self.decoder.parameters()]

        ### define optimizer method
        if self.opt_alg.upper == 'ADAM':
            optimizer = optim.Adam(params=params, lr=self.learning_rate)
        if self.opt_alg.upper == 'SGD':
            optimizer = optim.SGD(params=params, lr=self.learning_rate)
        else:
            assert False, 'This version only supports ADAM and SGD!'

        ### define loss function
        criterion = nn.MSELoss()

        ### calculate number of batch iterations
        n_batches = int(self.input_tensor.shape[1] / self.batch_size)

        ### save loss
        losses = []

        for epoch in range(self.n_epochs):
            self.encoder.train()
            self.decoder.train()
            batch_loss = []

            for batch in range(n_batches):
                # select data
                input_batch = self.input_tensor[:, batch: batch + self.batch_size, :]

                # target_batch_input means the "luminosity delta", which is given to us
                # we will combine this information with "calibration" as input to decoder each time
                target_batch_input = self.target_tensor[:, batch: batch + self.batch_size, 0:1]

                # target_batch means the "calibration", which is the value we want to predict
                # so target_batch is the real target
                target_batch = self.target_tensor[:, batch: batch + self.batch_size, 1:]

                # move data to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                target_batch_input = target_batch_input.to(self.device)

                # outputs tensor
                outputs = torch.zeros(self.target_len, self.batch_size, target_batch.shape[2])

                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                # different training strategies
                # use teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = target_batch[t, :, :]
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_input = torch.cat((lumi_feature, decoder_input), dim=1)

                # predict recursively
                else:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                        decoder_input = decoder_output

                # compute the loss
                outputs = outputs.to(self.device)
                loss = criterion(outputs, target_batch)
                batch_loss.append(loss.item())

                # backpropagation
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(batch_loss)
            #print('>>>>>> {}/{} Epoch; Loss={}'.format(epoch, self.n_epochs, epoch_loss))
            losses.append(epoch_loss)

            ### we save its loss every print_step
            if epoch % self.print_step == 0:
                plot_loss(losses, self.loss_figure_name)
        show_loss(losses)

    def train_model_mixed(self):

        ### move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        ### get the learnable parameters
        params = []
        params += [x for x in self.encoder.parameters()]
        params += [x for x in self.decoder.parameters()]

        ### define optimizer method
        if self.opt_alg.upper == 'ADAM':
            optimizer = optim.Adam(params=params, lr=self.learning_rate)
        if self.opt_alg.upper == 'SGD':
            optimizer = optim.SGD(params=params, lr=self.learning_rate)
        else:
            assert False, 'This version only supports ADAM and SGD!'

        ### define loss function
        criterion = nn.MSELoss()

        ### calculate number of batch iterations
        n_batches = int(self.input_tensor.shape[1] / self.batch_size)

        ### save loss
        losses = []

        for epoch in range(self.n_epochs):
            self.encoder.train()
            self.decoder.train()
            batch_loss = []

            for batch in range(n_batches):
                # select data
                input_batch = self.input_tensor[:, batch: batch + self.batch_size, :]

                # target_batch_input means the "luminosity delta", which is given to us
                # we will combine this information with "calibration" as input to decoder each time
                target_batch_input = self.target_tensor[:, batch: batch + self.batch_size, 0:1]

                # target_batch means the "calibration", which is the value we want to predict
                # so target_batch is the real target
                target_batch = self.target_tensor[:, batch: batch + self.batch_size, 1:]

                # move data to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                target_batch_input = target_batch_input.to(self.device)

                # outputs tensor
                outputs = torch.zeros(self.target_len, self.batch_size, target_batch.shape[2])

                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                # different training strategies
                # predict using mixed teacher forcing
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output

                    # predict with teacher forcing
                    if random.random() < self.teacher_forcing_ratio:
                        decoder_input = target_batch[t, :, :]
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_input = torch.cat((lumi_feature, decoder_input), dim=1)
                    # predict recursively
                    else:
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                        decoder_input = decoder_output

                # compute the loss
                outputs = outputs.to(self.device)
                loss = criterion(outputs, target_batch)
                batch_loss.append(loss.item())

                # backpropagation
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(batch_loss)
            #print('>>>>>> {}/{} Epoch; Loss={}'.format(epoch, self.n_epochs, epoch_loss))
            losses.append(epoch_loss)

            ### we save its loss every print_step
            if epoch % self.print_step == 0:
                plot_loss(losses, self.loss_figure_name)
        show_loss(losses)