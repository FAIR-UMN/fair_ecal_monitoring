# This script includes the functions to make predictions on data
from ecal_dataset_prep import *

class Seq2Seq_Prediction:
    def __init__(self,
                encoder,
                decoder,
                Xtrain,
                Ytrain,
                df,
                scaler_cali,
                device,
                fig_name_mape,
                fig_name_mse,
                metric_file,
                strategy):

        self.encoder = encoder
        self.decoder = decoder
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.df = df
        self.scaler_cali = scaler_cali
        self.device = device
        self.fig_name_mape = fig_name_mape
        self.fig_name_mse = fig_name_mse
        self.metric_file = metric_file
        self.strategy = strategy

    def start_prediction(self):

        if self.strategy == 'case1':  # do not use prediction as input to help the next-round prediction
            print('>>> ', self.strategy, ': start prediction...(be patient)')
            self.prediction_case1()
            print('>>> Finish prediction!')

        elif self.strategy == 'case2':  # use prediction as input to help the next-round prediction
            print('>>> ', self.strategy, ': start prediction...(be patient)')
            self.prediction_case2()
            print('>>> Finish prediction!')

        else:
            assert False, "Please select one of them---[case1, case2]!"

    # case1: we do not use prediction as input to help next-round prediction
    def prediction_case1(self):

        # get the correct scaler
        # if 'train' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['train']
        # elif 'val' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['val']
        # elif 'test' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['test']

        iw = self.Xtrain.shape[0]
        ow = self.Ytrain.shape[0]

        batches = self.Xtrain.shape[1]

        pred_Ytrain = np.zeros((self.Ytrain.shape[0], self.Ytrain.shape[1], 1))

        # plot training/test predictions
        for ii in range(batches):
            # train set
            X_train_temp = self.Xtrain[:, ii, :]
            Y_train_temp = self.Ytrain[:, ii, :]
            input_tensor = torch.from_numpy(X_train_temp).type(torch.Tensor).to(self.device)
            target_tensor = torch.from_numpy(Y_train_temp).type(torch.Tensor).to(self.device)
            Y_train_pred = self.predict(self.encoder, self.decoder, input_tensor, target_tensor, target_len=ow)
            pred_Ytrain[:, ii:ii + 1, :] = Y_train_pred

        ### after we have all the predictions, we want to plot the fig and compute its performance metric
        ### first, we want to flatten our windowed data
        Ytrain = self.Ytrain[:, :, 1:]
        GT_np = self.from_seq2norm(Ytrain)
        GT_np = np.asarray(GT_np).reshape(-1, 1)
        GT_np_org_scale = self.scaler_cali.inverse_transform(GT_np)
        Pred_np = self.from_seq2norm(pred_Ytrain)
        Pred_np = np.asarray(Pred_np).reshape(-1, 1)
        Pred_np_org_scale = self.scaler_cali.inverse_transform(Pred_np)

        meanAPE = self.MAPE_Metric(GT_np_org_scale, Pred_np_org_scale)
        fig_title = 'MAPE = {}'.format(meanAPE)
        T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mape, fig_title)
        # plot_prediction(GT_np, GT_np, lumi_info, T_info, fig_name, fig_title)

        mse = self.MSE_Metric(GT_np_org_scale, Pred_np_org_scale)
        # fig_title = 'MSE = {}'.format(mse)
        # T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        # lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        # self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mse, fig_title)

        # now, we want to save the metrics into the metric file
        metric_dict = {'MAP': [meanAPE], 'MSE': [mse]}
        metric_df = pd.DataFrame.from_dict(metric_dict)
        metric_df.to_csv(self.metric_file, index=False)
        return


    # case2: we use prediction as input to help next-round prediction
    def prediction_case2(self):

        # get the correct scaler
        # if 'train' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['train']
        # elif 'val' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['val']
        # elif 'test' in self.metric_file:
        #     scaler_cali = self.scaler_cali_dict['test']

        iw = self.Xtrain.shape[0]
        ow = self.Ytrain.shape[0]

        batches = self.Xtrain.shape[1]

        pred_Ytrain = np.zeros((self.Ytrain.shape[0], self.Ytrain.shape[1], 1))

        Y_train_pred = []
        for ii in range(batches):
            # train set
            if ii == 0:
                X_train_temp = self.Xtrain[:, ii, :]
            else:
                X_train_temp = self.Xtrain[:, ii, :]
                X_train_temp[:, 1] = Y_train_pred.reshape(-1)
            Y_train_plt = self.Ytrain[:, ii, :]
            input_tensor = torch.from_numpy(X_train_temp).type(torch.Tensor).to(self.device)
            target_tensor = torch.from_numpy(Y_train_plt).type(torch.Tensor).to(self.device)
            Y_train_pred = self.predict(self.encoder, self.decoder, input_tensor, target_tensor, target_len=ow)
            pred_Ytrain[:, ii:ii + 1, :] = Y_train_pred

        ### after we have all the predictions, we want to plot the fig and compute its performance metric
        ### first, we want to flatten our windowed data
        Ytrain = self.Ytrain[:, :, 1:]
        GT_np = self.from_seq2norm(Ytrain)
        GT_np = np.asarray(GT_np).reshape(-1, 1)
        GT_np_org_scale = self.scaler_cali.inverse_transform(GT_np)
        Pred_np = self.from_seq2norm(pred_Ytrain)
        Pred_np = np.asarray(Pred_np).reshape(-1, 1)
        Pred_np_org_scale = self.scaler_cali.inverse_transform(Pred_np)

        meanAPE = self.MAPE_Metric(GT_np_org_scale, Pred_np_org_scale)
        fig_title = 'MAPE = {}'.format(meanAPE)
        T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mape, fig_title)

        mse = self.MSE_Metric(GT_np_org_scale, Pred_np_org_scale)
        # fig_title = 'MSE = {}'.format(mse)
        # T_info = self.df.index[iw:iw + len(GT_np_org_scale)]
        # lumi_info = self.df['delta_lumi'][iw:iw + len(GT_np_org_scale)]
        # self.plot_prediction(GT_np_org_scale, Pred_np_org_scale, lumi_info, T_info, self.fig_name_mse, fig_title)

        # now, we want to save the metrics into the metric file
        metric_dict = {'MAP': [meanAPE], 'MSE': [mse]}
        metric_df = pd.DataFrame.from_dict(metric_dict)
        metric_df.to_csv(self.metric_file, index=False)

        return

    def predict(self, encoder, decoder, input_tensor, target_tensor, target_len):
        # This function is used to make prediction once the model is trained
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            # extend the input tensor to correct dim
            input_tensor = input_tensor.unsqueeze(1)
            target_tensor_2features = target_tensor.unsqueeze(1)

            # target_tensor_input is the "luminosity delta", which will be used as input to the decoder
            target_tensor_input = target_tensor_2features[:, :, 0:1]

            # target_tensor is the "calibration", which is the value we want to predict
            target_tensor = target_tensor_2features[:, :, 1:]

            encoder_output, encoder_hidden = encoder(input_tensor)

            # initialize tensor for predictions
            outputs = torch.zeros(target_len, target_tensor.shape[2])

            # decode input_tensor
            decoder_input = input_tensor[-1, :, :]  # the initialization, can be any values
            decoder_hidden = encoder_hidden

            # make prediction step by step
            for t in range(target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output.squeeze(0)
                lumi_feature = target_tensor_input[t, :, :]
                decoder_output = torch.cat((lumi_feature, decoder_output), dim=1)
                decoder_input = decoder_output

            np_outputs = outputs.detach().unsqueeze(1)
            np_outputs = np_outputs.numpy()

        return np_outputs

    # This function converts the seq to normal format
    def from_seq2norm(self, input_np):
        result = []

        total_batch_num = input_np.shape[1]

        for cur_b in range(total_batch_num):
            cur_batch_data = input_np[:, cur_b:cur_b + 1, :]
            sample_num = cur_batch_data.shape[1]
            for cur_idx in range(sample_num):
                result.extend((cur_batch_data[:, cur_idx, :]).flatten().tolist())
        return result

    def plot_prediction(self, target, pred, lumi_info, time_info, fig_name, fig_title):
        #### double Y figure
        fig, ax1 = plt.subplots(figsize=(16, 9))  # fig, ax1 = plt.subplots(figsize=(25, 5))
        plt.title(fig_title, fontsize=20)
        plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=14)
        plot1 = ax1.plot(time_info, target, color='b', linewidth=2, label='Calibration (true)')
        plot2 = ax1.plot(time_info, pred, color='r', linewidth=2, label='Calibration (prediction)')
        ax1.set_ylabel('Calibration', fontsize=18)
        ax1.yaxis.label.set_color('b')
        ax1.set_xlabel('Time Info', fontsize=18)
        ax1.set_ylim(0.7, 1)
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ### now, start the other y plot
        ax2 = ax1.twinx()
        plot3 = ax2.plot(time_info, lumi_info, label='Luminosity', color='grey', linewidth=1, linestyle='dashed')
        ax2.set_ylabel('Luminosity', fontsize=18)
        ax2.yaxis.label.set_color('grey')
        # ax2.set_ylim(0, 0.08)
        # ax2.set_xlim(1966, 2014.15)
        # ax2.tick_params(axis='y', labelsize=14)
        for tl in ax2.get_yticklabels():
            tl.set_color('grey')

        lines = plot1 + plot2 + plot3
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.tight_layout()
        plt.savefig(fig_name, dpi=300)
        plt.show()
        plt.close()

    def MAPE_Metric(self, GT_np, Pred_np):
        if len(GT_np) != len(Pred_np):
            assert False, 'GT_np and Pred_np must have the same length!'
        APES = []
        for i in range(len(GT_np)):
            ape = abs((Pred_np[i] - GT_np[i]) / (GT_np[i]))
            if np.isnan(ape):
                continue
            APES.append(ape)
        meanAPE = (sum(APES) * 100 / len(APES))
        meanAPE = np.round(meanAPE, 3)[0]
        return meanAPE

    def MSE_Metric(self, GT_np, Pred_np):
        if len(GT_np) != len(Pred_np):
            assert False, 'GT_np and Pred_np must have the same length!'

        GT_np_arr = np.asarray(GT_np)
        Pred_np_arr = np.asarray(Pred_np)
        sum_pow = np.sum(np.power(GT_np_arr - Pred_np_arr, 2))
        # sqrt_sum = np.sqrt(sum_pow)
        mse = sum_pow / len(GT_np)
        return mse








if __name__ == '__main__':
    pass