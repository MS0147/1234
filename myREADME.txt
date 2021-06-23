<changed>

model.py
1) Error appear

Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

-> class BeatGan
-> self.bce_criterion = nn.BCELoss().cuda()
   self.mse_criterion = nn.MSELoss().cuda()

-> def update_netd
->
        self.err_d_real = self.bce_criterion(self.out_d_real, torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.FloatTensor).cuda())

        self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full((self.batchsize,), self.fake_label, device=self.device).type(torch.FloatTensor).cuda())


data.py
1) if you use 2D signal data, the data shape must (num_of_data, 1, width, height)

2) I changed "np.load()" parts for utilize.

3) normalize part is changed.

network.py
1) Conv2D(input channel, output channel, kernel_size, stride, padding)
  output size = output channel * (input_size-kernel_size+2*padding)/stride+1

  We use 128*128 data so we set "Encoder"'s Conv2d's parameters like that.
  The result is 50*1*1

2) Decoder is inverse of Encoder

3) Third model accuracy is 0.9460. the 5th and 6th layer parameter is saved in that file.


option.py
1) change batchsize from 128 to 64, because Memory allocate issue was happened.

etc)
1) 2D-data must have the shape (n,1,w,h)
   1D-data is (n,1,w)

1-1) So, you should go to experiments/ecg/dataset/preprocessed.
    Push change.py and change2.py to ano0.
    Diretory ano0 should have _ _samples.npy file.

2) Data augment can exec in run_ecg.sh n_aug
