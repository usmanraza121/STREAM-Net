"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import pandas as pd

class TscanTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("TS-CAN trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            mean_var = []       # for mean variance of MC-dropout
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test) # exclude when using MC-drop
                
                # print('for loop:', _, 'pred_ppg_test=', pred_ppg_test[1])
            #================MC_dropout==========================================#
                # self.model.module.enable_dropout()
                # samples, sample_mean, sample_var = self.model.module.mc_sample(data_test)
                # # print(f'sample:{samples.shape},sample_mean:{sample_mean.shape}, sample_var:{sample_var.shape}')
                # # print('sample_mean=', sample_mean[1])

                # mean_var.append(sample_var.detach())
                # # print('variance=', mean_var)
                # # print('all variance:', mean_var.shape)

                # pred_ppg_test = sample_mean.view(-1,1)    # include when using MC-drop------------------
                
                # # self.plots(samples, sample_mean, sample_var, labels_test) # to plot and save results
                # # ------------------------------------------------
                # # pre3 = self.mc_dropout(data_test, label= labels_test)  # old method
            #-------------------------loop inside-------------------
                # pred_ppg_test = self.model.module.mc_sample2(data_test)
                # self.model.train()
                # with torch.no_grad():
                #     for i in range(10):
                #         pred_ppg_test = self.model.module.mc_sample2(data_test)
                #         for idx in range(batch_size):
                #             subj_index = test_batch[2][idx]
                #             sort_index = int(test_batch[3][idx])
                #             if subj_index not in predictions.keys():
                #                 predictions[subj_index] = dict()
                #                 labels[subj_index] = dict()
                #             predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                #             labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                #         print('---calculat MC metrics-------')
                #         calculate_metrics(predictions, labels, self.config)
            #================MC_dropout==========================================#
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)
        # mean_var = torch.cat(mean_var, dim=0)   # mean variance for MC-dropout
        # mean_vars = mean_var.mean()             # mean variance for MC-dropout
        # print(f"Mean_Variance: {mean_vars.item()}")     # mean variance for MC-dropout
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    #================MC_dropout==========previous data=========================#
    def mc_dropout(self, data_test, label, num_samples = 30, save=False):
        preds = []
        self.model.train()
        with torch.no_grad():
            for s in range(num_samples):
                output = self.model(data_test)
                preds.append(output.detach().cpu().numpy().flatten())
           
        pred = torch.tensor(preds)
        pred = pred.reshape(720, -1)
        label = label.cpu()
        # print('pred:', pred.shape)
        
        if save:
            df1 = pd.DataFrame(pred)
            df2= pd.DataFrame(label)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            df1.to_excel(f'results/predictions_{timestamp}.xlsx', index=False)
            # df1.to_excel(f'results/predictions1.xlsx', index=True)
            df2.to_excel(f'results/labels_{timestamp}.xlsx', index=False)
            print("Results saved to predictions_results.xlsx") 

        return pred
    
    def plots(self, samples, sample_mean, sample_var, labels ):
        samples = samples.cpu().numpy()
        sample_mean = sample_mean.cpu().numpy()
        sample_var = sample_var.cpu().numpy()
        label = labels.cpu().numpy()
        idd = 'no-dropout'   # keep name

        plt.figure(figsize=(12, 12))
        print('shape', sample_mean.shape)
        epistemic_uncertainty = np.var(sample_mean, axis=0)#.mean(0)
        print("epistemic_uncertainty",epistemic_uncertainty)
        #------------------------Plots--------------------------------------
        print('plots output----------samples_shape', samples.shape)
        plt.subplot(3, 1, 1)
        for i in range(samples.shape[1]):
            plt.plot(range(300), samples[:300,i], label=f"Data Point {i+1}", alpha=0.7)
            # if i ==1:
                # plt.plot(range(samples.shape[0]), label, color='b')
        plt.title('Data Points')
        plt.xlabel('Data Points Index')
        plt.ylabel('Prediction')
        # plt.legend()

        # 2. Plot the sample means (the "best guess" prediction)
        plt.subplot(3, 1, 2)
        plt.plot(range(300), sample_mean[0:300], linestyle='-', color='b', label="Sample Mean")
        # plt.plot(range(samples.shape[0]), sample_mean, marker='o', linestyle='-', color='b', label="Sample Mean")
        plt.title('Mean MC Samples')
        plt.xlabel('Data Point Index')
        plt.ylabel('Mean Prediction')
        # plt.legend()

        # 3. Plot the sample variances (uncertainty)
        plt.subplot(3, 1, 3)
        plt.plot(range(300), sample_var[0:300], linestyle='-', color='r', label="Sample Variance")
        # plt.plot(range(samples.shape[0]), sample_var, marker='x', linestyle='-', color='r', label="Sample Variance")
        plt.title('Variance MC Samples')
        plt.xlabel('Data Point Index')
        plt.ylabel('Prediction Variance')
        # plt.legend()

        plt.tight_layout()
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs('./results', exist_ok=True)
        filename = f"monte_carlo_plots_{idd}_{timestamp}.png"
        filepath = os.path.join('results', filename)
        plt.savefig(filepath, dpi=300)  # Save with high resolution

# plot for predictions and labels-----------------------------------
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        for i in range(samples.shape[1]):
            plt.plot(range(300), samples[:300,i], label=f"Data Point {i+1}", alpha=0.7)
            # if i ==1:
                # plt.plot(range(samples.shape[0]), label, color='b')
        plt.title('Data Points')
        plt.xlabel('Data Point Index')
        plt.ylabel('Prediction')
        # plt.legend()

        # 2. Plot the sample means (the "best guess" prediction)
        plt.subplot(2, 1, 2)
        plt.plot(range(300), label[:300], color='b')
        # plt.plot(range(samples.shape[0]), sample_mean, marker='o', linestyle='-', color='b', label="Sample Mean")
        plt.title('Predictions and Labels')
        plt.xlabel('Data Point Index')
        plt.ylabel('Ground Truth')
        plt.tight_layout()

        filename = f"Monte_Carlo_predictions_vs_labels_{idd}_{timestamp}.png"
        filepath = os.path.join('results', filename)
        plt.savefig(filepath, dpi=300)  # Save with high resolution

        print(f"Plot saved successfully at: {filepath}")
        plt.close()

      
        prediction_columns = [f"Prediction {i+1}" for i in range(samples.shape[1])]
        label_column = ["Label"]

        df1 = pd.DataFrame(samples, columns=prediction_columns)  # Shape (720, 30)
        df2 = pd.DataFrame(label, columns=label_column)        ## Shape (720, 1)
        df3 = pd.concat([df1, df2], axis=1)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        df3.to_excel(f'results/predictions{idd}_{timestamp}.xlsx', index=False)
        #------------------------------------------------
    
    # def save_results(self):
    #     """
    #     Combine results into single dataframe and save to disk as .csv file
    #     """
    #     results = pd.concat([
    #         pd.DataFrame(self.IDs.cpu().numpy(), columns= ['ID']),  
    #         pd.DataFrame(self.predicted_labels.cpu().numpy(), columns= ['predicted_label']),
    #         pd.DataFrame(self.correct_predictions.cpu().numpy(), columns= ['correct_prediction']),
    #         pd.DataFrame(self.epistemic_uncertainty.cpu().numpy(), columns= ['epistemic_uncertainty']), 
    #     ], axis=1)

    #     create_results_directory()
    #     results.to_csv('results/{}_{}_results.csv'.format(self.__class__.__name__, datetime.datetime.now().replace(microsecond=0).isoformat()), index=False)
