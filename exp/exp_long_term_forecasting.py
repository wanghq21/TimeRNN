from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import math
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')


class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean', beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        diff = torch.abs(input - target)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction mode: {}".format(self.reduction))



class SmoothL1Loss2(nn.Module):
    def __init__(self, reduction='mean', beta=1.0):
        super(SmoothL1Loss2, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        diff = torch.abs(input - target)
        loss = torch.where(diff < self.beta, 
                           0.5 * diff ** 2 / self.beta, 
                           torch.sqrt(diff + 1e-8) - math.sqrt(self.beta) + 0.5 * self.beta)
                           
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction mode: {}".format(self.reduction))




class SmoothL1Loss3(nn.Module):
    def __init__(self, reduction='mean', beta=1.0, beta1=3.0):
        super(SmoothL1Loss3, self).__init__()
        self.beta = beta
        self.beta1 = beta1
        self.reduction = reduction

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        diff = torch.abs(input - target)
        loss = torch.where(diff < self.beta, 
                           0.5 * diff ** 2 / self.beta, 
                           torch.where(diff < self.beta1,
                           diff - 0.5 * self.beta, 
                           torch.sqrt(diff + 1e-8) - math.sqrt(self.beta1) + 0.5 * self.beta1 - 0.5 * self.beta))
              
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction mode: {}".format(self.reduction))


# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """

#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # self.mov = moving_avg(kernel_size=3,stride=1)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        # criterion = torch.nn.SmoothL1Loss()
        return criterion

    def _select_criterion_train(self):
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        # criterion = SmoothL1Loss(beta=0.1)
        # criterion = SmoothL1Loss2(beta=0.1)
        # criterion = SmoothL1Loss3(beta=0.1, beta1=1)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_cycle = batch_cycle.int().to(self.device)

                if 'custom' in self.args.data or 'ETTh' in self.args.data or 'ETTm' in self.args.data: 
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_train = self._select_criterion_train()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
  
                if 'custom' in self.args.data or 'ETTh' in self.args.data or 'ETTm' in self.args.data: 
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # batch_y = self.mov(batch_y)
                    # batch_y = torch.cat((ba tch_x, batch_y),dim=-2)
                    loss = criterion_train(outputs, batch_y)  
                    # loss = torch.sqrt(loss)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
  
                if 'custom' in self.args.data or 'ETTh' in self.args.data or 'ETTm' in self.args.data: 
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    else:
                        if self.args.output_attention:
                            outputs  = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs  = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                out = outputs

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                
                # if i == 10:

                #     x = batch_x[0, :, :] 
                #     m = out[0, :, :] 
                    
                #     # 计算相关系数矩阵
                #     # 转置后，每一行为一个变量，每一列为一个观测值
                #     data_for_corr = x.T  # shape: (10, 100)
                #     corr_matrix = torch.corrcoef(data_for_corr)

                #     # 将 tensor 转为 numpy 数组
                #     corr_np = corr_matrix.cpu().numpy()

                #     # 绘制热力图
                #     plt.figure(figsize=(8, 6))
                #     im = plt.imshow(corr_np, cmap='coolwarm', vmin=-1, vmax=1)
                #     plt.colorbar(im, fraction=0.046, pad=0.04)
                #     plt.title('Correlation Matrix Heatmap')
                #     plt.xlabel('Channel')
                #     plt.ylabel('Channel')
                #     # plt.xticks(range(self.args.enc_in))
                #     # plt.yticks(range(self.args.enc_in))
                #     # plt.show()
                #     plt.savefig('{}_{}_correlation_x.pdf'.format(self.args.model, self.args.enc_in))

                #     data_for_corr = m.T  # shape: (10, 100)
                #     corr_matrix = torch.corrcoef(data_for_corr)

                #     # 将 tensor 转为 numpy 数组
                #     corr_np = corr_matrix.cpu().numpy()

                #     # 绘制热力图
                #     plt.figure(figsize=(8, 6))
                #     im = plt.imshow(corr_np, cmap='coolwarm', vmin=-1, vmax=1)
                #     plt.colorbar(im, fraction=0.046, pad=0.04)
                #     plt.title('Correlation Matrix Heatmap')
                #     plt.xlabel('Channel')
                #     plt.ylabel('Channel')
                #     # plt.xticks(range(self.args.enc_in))
                #     # plt.yticks(range(self.args.enc_in))
                #     # plt.show()
                #     plt.savefig('{}_{}_correlation_m.pdf'.format(self.args.model, self.args.enc_in))


                #     x = batch_x.detach().cpu().numpy()[0, :, :].transpose()
                #     m = out.detach().cpu().numpy()[0, :, :].transpose()
                #     corr_matrix = np.corrcoef(x)
                #     corr_matrix = np.abs(corr_matrix)
                #     triu_corr_matrix = np.triu(corr_matrix, k=1)  # 只保留上三角
                #     mean_corr_x = np.nanmean(triu_corr_matrix)  # 计算上三角部分的平均值
                #     print("绝对平均相关系数:", mean_corr_x)
                #     corr_values = triu_corr_matrix[~np.isnan(triu_corr_matrix)]  # 获取非 NaN 值
                #     corr_variance_x = np.var(corr_values)
                #     print("绝对相关性强度的方差:", corr_variance_x)
                #     corr_matrix = np.corrcoef(m)
                #     corr_matrix = np.abs(corr_matrix)
                #     triu_corr_matrix = np.triu(corr_matrix, k=1)  # 只保留上三角
                #     mean_corr_m = np.nanmean(triu_corr_matrix)  # 计算上三角部分的平均值
                #     print("绝对平均相关系数:", mean_corr_m)
                #     corr_values = triu_corr_matrix[~np.isnan(triu_corr_matrix)]  # 获取非 NaN 值
                #     corr_variance_m = np.var(corr_values)
                #     print("绝对相关性强度的方差:", corr_variance_m)

                #     # corr_matrix = np.corrcoef(m)
                #     # corr_matrix = np.abs(corr_matrix)
                #     # dist_matrix = 1 - corr_matrix  # 相关性越高，距离越近
                #     # linkage_matrix = linkage(squareform(dist_matrix), method='ward')
                #     # cluster_labels = fcluster(linkage_matrix, t=2, criterion='maxclust')  # 设定2个簇
                #     # print('聚类方差', np.var(cluster_labels))  # 计算聚类标签的方差，值越小表示聚类越紧密


                #     # x = batch_x.detach().cpu().numpy()[0, :, :].transpose()
                #     # m = out.detach().cpu().numpy()[0, :, :].transpose()
                #     # tsne = TSNE(n_components=2, perplexity=self.args.enc_in//10, random_state=42)
                #     # data_tsne = tsne.fit_transform(x)

                #     # # 可视化降维后的数据
                #     # plt.figure(figsize=(8, 6))
                #     # plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='b', label='Channels')
                #     # plt.title('t-SNE Visualization of Channels')
                #     # plt.xlabel('t-SNE Component 1')
                #     # plt.ylabel('t-SNE Component 2')
                #     # plt.grid(True)
                #     # # plt.show()
                #     # plt.savefig('x.pdf')

                #     # tsne = TSNE(n_components=2, perplexity=self.args.enc_in//10, random_state=42)
                #     # data_tsne = tsne.fit_transform(m)

                #     # # 可视化降维后的数据
                #     # plt.figure(figsize=(8, 6))
                #     # plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='b', label='Channels')
                #     # plt.title('t-SNE Visualization of Channels')
                #     # plt.xlabel('t-SNE Component 1')
                #     # plt.ylabel('t-SNE Component 2')
                #     # plt.grid(True)
                #     # # plt.show()
                #     # plt.savefig('middle.pdf')



        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)



        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mean_corr_x:{}, mean_corr_m:{}, corr_variance_x:{}, corr_variance_m:{}, alpha1:{}, alpha2:{}, patch:{}, dropout:{}, d2:{}, learning_rate:{}'.format(mse, mae, rmse, mape, mean_corr_x, mean_corr_m, corr_variance_x, corr_variance_m, self.args.alpha1, self.args.alpha2, self.args.patch, self.args.dropout, self.args.d2, self.args.learning_rate))
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, patch:{}, dropout:{}, d2:{}, learning_rate:{}'.format(mse, mae, rmse, mape, self.args.patch, self.args.dropout, self.args.d2, self.args.learning_rate))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
