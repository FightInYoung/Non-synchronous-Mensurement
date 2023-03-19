function [mix_signal,snr1]=mix_noise(signal,snr)
[sampling,micro_channel]=size(signal);
db = snr;
n=wgn(sampling,1,1);
for channel=1:1:micro_channel
alpha = sqrt( sum(signal(:,channel).^2)/(sum(n.^2)*10^(db/10)) );

%check SNR
snr1 = 10*log10(sum(signal(:,channel).^2)/sum((alpha.*n).*(alpha.*n)));

mix_signal(:,channel) =signal(:,channel) + alpha*n;
end