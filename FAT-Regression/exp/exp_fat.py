from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, get_record
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter


warnings.filterwarnings('ignore')


class Exp_fresim(Exp_Basic):
    def __init__(self, args):
        super(Exp_fresim, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs/{args.data}/{args.model}/{args.pretrain_mode}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            if self.args.trs:
                print("train from scratch")
            else:
              print("Loading ckpt: {}".format(self.args.load_checkpoints))
              model = transfer_weights(self.args.load_checkpoints, model, device=self.device, freeze=self.args.freeze)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data, self.args.exp_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.pretrain_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.pretrain_epochs):
            start_time = time.time()
            if self.args.task_type == "clf":
                train_loss, train_cl_loss, train_rb_loss, train_w_loss = self.pretrain_one_epoch(train_loader,
                                                                                                 model_optim,
                                                                                                 model_scheduler)
                vali_loss, valid_cl_loss, valid_rb_loss, valid_w_loss = self.valid_one_epoch(vali_loader)

            else:
                train_loss, train_cl_loss, train_rb_loss, train_w_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
                vali_loss, valid_cl_loss, valid_rb_loss, valid_w_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()

            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f}/{6:.4f} Val Loss: {7:.4f}/{8:.4f}/{9:.4f}/{10:.4f}"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,
                        train_rb_loss, train_w_loss,
                        vali_loss, valid_cl_loss, valid_rb_loss, valid_w_loss))

            pretrain_txt = path + "/" + "pretrain_loss.txt"
            pretrain_content = "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f}/{6:.4f} Val Loss: {7:.4f}/{8:.4f}/{9:.4f}/{10:.4f}\n".format(
                epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,
                train_rb_loss, train_w_loss,
                vali_loss, valid_cl_loss, valid_rb_loss, valid_w_loss)
            get_record(pretrain_txt, pretrain_content)

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_cl_loss': train_cl_loss,
                'train_rb_loss': train_rb_loss,
                'vali_loss': vali_loss,
                'valid_cl_loss': valid_cl_loss,
                'valid_rb_loss': valid_rb_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))
                save_info = "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...\n".format(min_vali_loss, vali_loss, epoch)
                get_record(pretrain_txt, save_info)

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best_ptmode:{self.args.pretrain_mode}.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))
                get_record(pretrain_txt, "Saving model at epoch {}...\n".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))


    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []
        train_cl_loss = []
        train_rb_loss = []
        train_w_loss = []

        self.model.train()
        for i, (batch_x, batch_y, *others) in enumerate(train_loader):
            model_optim.zero_grad()


            batch_x = batch_x.float().to(self.device)
            loss, loss_cl, loss_rb, loss_w, _, _, _ = self.model(batch_x)

            # backward
            loss.backward()
            model_optim.step()

            # record
            train_loss.append(loss.item())
            train_cl_loss.append(loss_cl.item())
            train_rb_loss.append(loss_rb.item())
            train_w_loss.append(loss_w)

        model_scheduler.step()

        train_loss = np.average(train_loss)
        train_cl_loss = np.average(train_cl_loss)
        train_rb_loss = np.average(train_rb_loss)
        train_w_loss = np.average(train_w_loss)

        return train_loss, train_cl_loss, train_rb_loss, train_w_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_cl_loss = []
        valid_rb_loss = []
        valid_w_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, *others) in enumerate(vali_loader):


            batch_x = batch_x.float().to(self.device)
            # encoder
            loss, loss_cl, loss_rb, loss_w, _, _, _ = self.model(batch_x)

            # Record
            valid_loss.append(loss.item())
            valid_cl_loss.append(loss_cl.item())
            valid_rb_loss.append(loss_rb.item())
            valid_w_loss.append(loss_w)

        vali_loss = np.average(valid_loss)
        valid_cl_loss = np.average(valid_cl_loss)
        valid_rb_loss = np.average(valid_rb_loss)
        valid_w_loss = np.average(valid_w_loss)

        self.model.train()
        return vali_loss, valid_cl_loss, valid_rb_loss, valid_w_loss

    def train(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, self.args.data, self.args.exp_name)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.freeze == 0:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, *others) in enumerate(test_loader):
                iter_count += 1
                model_optim.zero_grad()

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                # loss
                loss = criterion(outputs, batch_y)

                # print("loss", loss)
                loss.backward()
                model_optim.step()

                # record
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss))

            finetune_txt = path + '/' + 'finetune_loss.txt'
            finetune_content = "lr_{0}_dp_{1}_{2}_Epoch: {3}, Steps: {4}, Time: {5:.2f}s | Train Loss: {6:.7f} Vali Loss: {7:.7f} Test Loss: {8:.7f} \n".format(
                self.args.learning_rate, self.args.dropout, self.args.pred_len, epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss)
            get_record(finetune_txt, finetune_content)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.freeze == 0:
               adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, *others) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder
                outputs = self.model(batch_x)

                # loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, *others) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()


                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))

        path = os.path.join(self.args.checkpoints, self.args.data, self.args.exp_name)
        text_txt = path + '/' + 'test_loss.txt'
        text_content = 'lr_{0}_dp_{1}_{2}___{3}->{4}, {5:.5f}, {6:.5f} \n'.format(self.args.learning_rate, self.args.dropout, self.args.exp_name, self.args.seq_len, self.args.pred_len, mse, mae)
        get_record(text_txt, text_content)


        if self.args.trs:
            filename = "{}/score_{}_train_from_scratch.txt".format(folder_path,self.args.model,)
        else:
            filename = "{}/score_{}_{}_{}.txt".format(folder_path,self.args.model,self.args.lm,self.args.pretrain_mode)
        f = open(filename,'a')
        f.write('lr_{0}_dp_{1}_{2}->{3}, {4:.3f}, {5:.3f} \n'.format(self.args.learning_rate, self.args.dropout, self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()