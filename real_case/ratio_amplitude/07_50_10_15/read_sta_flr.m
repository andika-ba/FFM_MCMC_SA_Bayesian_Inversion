function [t_datp,res_10_flr] = read_sta_flr(sta_flr)
% load seismic data;

load(sta_flr)

test_flr = crise(:,2)/(1.4966*800*10^6);  %convert counts to velocity


% filter test between 5 AND 15

% RESHAPE and median -> smooth 1s dATA
% RESHAPE and median -> smooth 10s dATA
[a,b]  = butter(4,5/50,'high');
[c,d]  = butter(4,15/50,'low');
ENV    = abs(hilbert(filtfilt(c,d,filtfilt(a,b,test_flr)))); % Calculate the envelope
crise_10sec = median(reshape(ENV(1:(floor(length(ENV)/1000)*1000)),...
                     1000,floor(length(ENV)/1000)),1); % Downsample the data to 10 sec, i.e. 1 ob-
                                                       % servation corresponds to 10 sec.
        
res = movmedian(crise_10sec,30); % smoothing 5 minutes


idx_init    = (7*3600+(50*60))/10;    % Index at 7:50 am for 10 sec      

idx_1100    = (11*3600+(0*60))/10;         % Final time (11:00) 
res_10_flr  = res(idx_init:idx_1100);     % Picked data array
t_datp      = (idx_init:1:idx_1100);  % Picked time array 10 second increment


end