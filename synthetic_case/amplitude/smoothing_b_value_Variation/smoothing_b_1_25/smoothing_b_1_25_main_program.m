clear;
close all;
clc;
tic

%% In a Nutshell

% This code is about fitting two different models. The models are the body wave attenuation model
% and the Failure Forecast Method (FFM). The body wave attenuation model is used to generate
% synthetic data representing seismic migration that preceeds volcanic eruption. The synthetic data
% set is then fitted with the FFM model using MCMC Simulated Annealing to obtain the best fit values
% of the FFM parameters (A, tf, and alpha). The uncertainties of these parameters are calculated
% using MCMC Bayesian Inversion.

%% Generating Synthetic Data

% True model param and simulated data
% The model used to generate synthetic data is based on the body wave attenuation model
% (Battaglia and Aki,2003), with some modifications.
% I0n1 is Seismic Amplitude (square root of Seismic Energy) of the source according to Lay
% and Wallace (1995)
% In this variation, we used b-value = 1.25, and smooth the time series using moving median
% technique (use movmedian built-in function).

rng('default')
v    = 1;              % Magma velocity/upward vertical velocity of the migrating source (m/s)
d1   = 100;            % Distance between the seismic station and the eruptive site (meters)
z0   = -2000;          % Initial depth of migrating magma/migrating seismic source (meters)
t0   = 0;              % Initial time (second)
tend = 1900;           % Time at the end of seismic amplitude recording (seconds)
dt   = 1;              % incretment of time
t    = (t0:dt:tend);   % Time array (seconds)
z    = z0+(v.*t);      % Depth array (meters)
r1   = sqrt((z.^2)+(d1^2)); % Radial distance array (meters), using Pythagorean Theorem

f       = 10;             % Frequency of the seismic wave (Hz)
Q       = 170;            % Quality factor for attenuation
beta    = 1000;           % Shear wave velocity (m/s)
n       = 1;              % Coefficient for body wave attenuation equation (Battaglia and Aki, 2003)
Ntot    = 2000;           % Total number of earthquake (Gutenberg Richter Law)
M_max   = 8;              % Maximum earthquake magnitude
M       = 0:0.001:M_max;  % Array of earthquake magnitude
a       = log10(Ntot);    % a (Gutenberg - Richter Law)
b       = 1.25;           % b-value Gutenberg - Richter Law
N       = 10.^(a-(b.*M)); % Gutenberg - Richter Law
aa      = min(N);
bb      = max(N);
rand_N  = (bb-aa).*rand(1,Ntot) + aa; % Randomly select earthquake magnitudes
my_M_GR = interp1(N,M,rand_N);
my_M_GR = my_M_GR(1:length(t));
Magn    = my_M_GR;

I0n1 = sqrt(10.^((1.5.*Magn)+11.8)); % Seismic amplitude of the source (Lay and Wallace, 1995)

y_in_ns = (I0n1.*exp(-pi*f.*r1./(Q*beta)))./(r1.^n); % Synthetic dataset generated

y_in    = movmedian(y_in_ns,60); %final synthetic data

nd   = 3;              % Number of parameters

%% MCMC Simulated Annealing

rng('default');
n_metro  = 1500;          % number of iterations at each temperature
T_start  = 1E3;           % initial cooling temperature
T_end    = 1E-7;          % final cooling temperature
alpha    = 0.99;          % cooling constant --> T_new = T_old*alpha
par_init = [2E-7 1900 1.10];  % initial guess
bounds   = [1E-8 1000 1.00;1E-2 3000 3.00];  % lower-upper limit (boundaries)
Temp     = T_start;      
mcur     = par_init;      % guess/initial values
sd1      = 1.05E-4;       % stepsize for parameter A (related to proposal distribution)
sd2      = 10;            % stepsize for parameter tf (related to proposal distribution)
sd3      = 0.04;          % stepsize for parameter alpha (related to proposal distribution)

k        = 1;             % For mplot and fplot indexing

l        = ceil(log(T_end/T_start)/log(alpha));
mplot    = zeros(l,nd);
fplot    = zeros(k,1);

prop_ok  = -1;

while Temp >= T_end
    
    for i = 1:n_metro     % Metropolis Loop
        while prop_ok < 0    
            mnew(1) = normrnd(mcur(1),sd1,1); % Generating new parameters using proposal distributions
            mnew(2) = normrnd(mcur(2),sd2,1); % Generating new parameters using proposal distributions
            mnew(3) = normrnd(mcur(3),sd3,1); % Generating new parameters using proposal distributions
            mnew    = [mnew(1) mnew(2) mnew(3)];
            fnew    = obj_func(mnew,y_in,t);  % Calculate objective function using proposed/new values
            fcur    = obj_func(mcur,y_in,t);  % Calculate objective function using old/current values
            deltae  = abs(fnew-fcur);         % Calculate the difference between fnew and fcur
            c1      = isreal(fnew);           % Check if the objective function is real value
            c2      = isreal(fcur);           % Check if the objective function is real value

            % Make sure that the proposed values fall within lower and upper bounds
            % Also make sure that the proposed values return real values
            if (((nd==sum(mnew > bounds(1,:))) && (nd==sum(mnew < bounds(2,:))) && (c1==1) && (c2==1) ))
                prop_ok = 1;
            else
                prop_ok = -1;
            end
        end               % End of loop for generating new parameters
        
        % Metropolis Criteria
        if (fnew>fcur)
            p = exp(-deltae/Temp); % Compute probability of acceptance
            if (p>rand())          % determine whether to accept worse point
                accept = true;     % accept the worse solution
            else
                accept = false;    % don't accept the worse solution
            end
        else
            accept = true;         % always accept the better solution (fnew<fcur)
        end
        
        if (accept == true)
            mcur = mnew;
        end
        
    end % End loop of Metropolis
    
    mplot(k,:) = mcur;    % record solution (mcmc sa sample) to see the convergence
    fplot(k)   = fcur;    % record objective function to see the convergence
    k          = k+1;
    Temp       = Temp*alpha;   % Decrease temperature
    prop_ok    = -1;

    
end % End loop of Temperature

mopt = mplot(end,:);
figure(1)
subplot(4,1,1)
plot(log10(fplot))
subplot(4,1,2)
plot(mplot(:,1))
subplot(4,1,3)
plot(mplot(:,2))
subplot(4,1,4)
plot(mplot(:,3))

%% MCMC Bayesian Inversion - Generating Posterior Distribution

rng('default');
n_metro_2     = 5E6;      % Number of sample (MCMC Bayesian Inversion) 
sd1_2         = 1E-10;    % stepsize for parameter A (related to proposal distribution)
sd2_2         = 0.001;    % stepsize for parameter tf (related to proposal distribution)
sd3_2         = 0.001;    % stepsize for parameter alpha (related to proposal distribution)
ac_cntr       = 0;        % To calculate the number of accepted samples
rej_cntr      = 0;        % To calculate the number of rejected samples
prop_ok       = -1;       
mcur          = par_init;              % guess/initiation values
mdist         = zeros(n_metro_2,nd);   % Empty matrix for mcmc samples
fdist         = zeros(n_metro_2,1);    
ac_rat        = zeros(n_metro_2,1);    % Empty matrix for acceptance rate
beta          = 0.02;                  % Constant for adapting stepsize

for iter = 1:n_metro_2     % Metropolis Loop to generate parameter distribution
    
    while prop_ok < 0
        if mod(iter,50000) == 0 % Every 50000 iterations adapt stepsizes
            sd1_2   = std(mdist((1:iter),1)); % Adapting the stepsize
            sd2_2   = std(mdist((1:iter),2)); % Adapting the stepsize
            sd3_2   = std(mdist((1:iter),3)); % Adapting the stepsize
            mnew(1) = (1 - beta)*normrnd(mcur(1),(2.38*sd1_2/sqrt(nd)),1);  % Generating new parameters using proposal distributions
            mnew(2) = (1 - beta)*normrnd(mcur(2),(2.38*sd2_2/sqrt(nd)),1);  % Generating new parameters using proposal distributions
            mnew(3) = (1 - beta)*normrnd(mcur(3),(2.38*sd3_2/sqrt(nd)),1);  % Generating new parameters using proposal distributions
        else
            mnew(1) = normrnd(mcur(1),sd1_2,1); % Generating new parameters using proposal distributions
            mnew(2) = normrnd(mcur(2),sd2_2,1); % Generating new parameters using proposal distributions
            mnew(3) = normrnd(mcur(3),sd3_2,1); % Generating new parameters using proposal distributions
        end
        
        mnew    = [mnew(1) mnew(2) mnew(3)];
        fnew    = obj_func(mnew,y_in,t); % Calculate objective function using proposed/new values
        fcur    = obj_func(mcur,y_in,t); % Calculate objective function using current/old values
        deltae  = abs(fnew-fcur); % Calculate the difference between fnew and fcur
        c1      = isreal(fnew); % Check if the objective function is real value
        c2      = isreal(fcur); % Check if the objective function is real value
        
        % Make sure that the proposed values fall within lower and upper bounds
        % Also make sure that the proposed values return real values
        
        if (((nd==sum(mnew > bounds(1,:))) && (nd==sum(mnew < bounds(2,:))) && (c1==1) && (c2==1) ))
            prop_ok = 1;
        else
            prop_ok = -1;
        end
        
    end               % End of loop for generating new parameters
        
    
    if (fnew>fcur)
        p = exp(-deltae/1); % Compute probability of acceptance
        if (p>rand())       % determine whether to accept worse point
            accept = true;  % accept the worse solution
        else
            accept = false; % don't accept the worse solution
        end
    else
        accept = true;      % always accept the better solution (fnew<fcur)
    end
    
    if (accept == true)
        mcur = mnew;
        ac_cntr = ac_cntr+1;   % accepted samples
    else
        rej_cntr = rej_cntr+1; % recjected samples
    end
    
    prop_ok       = -1;
    mdist(iter,:) = mcur;      % store samples
    fdist(iter)   = fcur;
    ac_rat(iter)  = 100*(ac_cntr/(rej_cntr+ac_cntr)); % calculate acceptance rate

    if mod(iter,20000) == 0
        fprintf('Iteration = %8d\n',iter)
        fprintf('Acceptance Rate (Percent) = %6.2f\n',ac_rat(iter))
    end


end % End loop of Metropolis

burnin_samp = 100000; % the number of burn in samples
mdist       = mdist((burnin_samp+1:end),:); % remaining samples
toc
save('smooth_b_1_25_seismic_intensity_try_hybrid_1900.mat') % save results as a mat file


%% Plotting MCMC Bayesian Inversion

pos_plt      = zeros(nd,2); % Create empty matrix for subplot position
start_num    = [1,2];       % Initiation
pos_plt(1,:) = start_num;

if nd >= 2
    for j = 2:nd
        pos_plt(j,:) = [start_num(1)+3,start_num(2)+3];
        start_num    = pos_plt(j,:);
    end
end

% Plotting MCMC Bayesian Inversion samples
figure()
sgtitle('Bayesian Inversion - Check Convergence')

for j = 1:nd
    subplot(nd,3,[pos_plt(j,1),pos_plt(j,2)])
    plot(mdist(:,j))
    subplot(nd,3,j*3)
    histogram(mdist(:,j),50)
    hold on
    xline(mopt(j),'--r','LineWidth',2)
end

%% Visualized PDF

figure(3)
FigHandle = figure(3);
set(FigHandle, 'Position', [100, 100, 1000, 800]);

np = nd;
for i=1:np
for j=1:np
    if i==j
        subplot(np,np,(i-1)*np+j)
        histogram(mdist(:,i),50,'EdgeColor','none')
        hold on
        line([mplot(end,i),mplot(end,i)],ylim ,'LineWidth',2,'Color','r')
        hold off
    end
    if j>i
        subplot(np,np,(i-1)*np+j)
        histogram2(mdist(:,j),mdist(:,i),[100,100],'displaystyle','tile','Normalization','probability')
        colormap(jet)
        hold on
        line([mplot(end,j),mplot(end,j)], ylim ,'LineWidth',2,'Color','r')
        line(xlim, [mplot(end,i),mplot(end,i)] ,'LineWidth',2,'Color','r')
        hold off
    end
end
end

%% Quantile Calculation

% Calculate 95% Confidence Interval for each parameter

signif_lev                    = 0.05;
[sample_mcmc_qu,bayes_intrvl] = quant_calc(mdist,signif_lev);

% Printing out min, max, and optimum values of each parameter 
fprintf('====================================================\n')
fprintf('|              Bayesian (Min Max)                  |\n')
fprintf('====================================================\n')
fprintf('| Parameter |     Min    |    Best    |     Max    |\n')
fprintf('====================================================\n')
fprintf('|  Param %d  |  %5.2E  |  %5.2E  |  %5.2E  |\n',1,min(mdist(:,1)),mopt(1),...
         max(mdist(:,1)))
fprintf('|  Param %d  |    %5.0f   |   %5.0f    |    %5.0f   |\n',2,min(mdist(:,2)),mopt(2),...
         max(mdist(:,2)))
fprintf('|  Param %d  |    %5.2f   |   %5.2f    |    %5.2f   |\n',3,min(mdist(:,3)),mopt(3),...
         max(mdist(:,3)))
fprintf('====================================================\n')
fprintf('\n')

% Printing out 95% confidence interval and optimum values of each parameter 
fprintf('====================================================\n')
fprintf('|             Bayesian (95 percent CI)             |\n')
fprintf('====================================================\n')
fprintf('| Parameter |    Lower   |     Best   |    Upper   |\n')
fprintf('====================================================\n')
fprintf('|  Param %d  |  %5.2E  |  %5.2E  |  %5.2E  |\n',1,bayes_intrvl(1,1),mopt(1),...
        bayes_intrvl(1,2))
fprintf('|  Param %d  |    %5.0f   |   %5.0f    |    %5.0f   |\n',2,bayes_intrvl(2,1),mopt(2),...
        bayes_intrvl(2,2))
fprintf('|  Param %d  |    %5.2f   |   %5.2f    |    %5.2f   |\n',3,bayes_intrvl(3,1),mopt(3),...
        bayes_intrvl(3,2))
fprintf('====================================================\n')
fprintf('\n')

%% Objective Function (Likelihood)

function [obj_val] = obj_func(var,ydat,tdat)

A_ana     = var(1);
tf_ana    = var(2);
alpha_ana = var(3);

ymods     = (A_ana*(alpha_ana-1)*tf_ana)^(1/(1-alpha_ana)).*...
            ((1-(tdat./tf_ana)).^(1/(1-alpha_ana)));              % The FFM model

obj_val   = (sum(((ydat-ymods).^2)))/2/(std(ydat)^2);
end

%% Quantile Function

function [smpl_mcmc_q,param_quant] = quant_calc(sample_mcmc,sig_level)

sz          = size(sample_mcmc);
n_param     = sz(2); 
param_quant = zeros(n_param,2);

for j = 1:n_param
    param_quant(j,:) = quantile(sample_mcmc(:,j),[sig_level/2 1-sig_level/2]);
end

idx_all     = 1:1:length(sample_mcmc);
idx_95      = false(length(sample_mcmc),1);

check_hist = zeros(sz(1),n_param);

for j = 1:sz(1)
    for k = 1:n_param
        if param_quant(k,1) <= sample_mcmc(j,k) && sample_mcmc(j,k) <= param_quant(k,2)
            check_hist(j,k) = 1;
        else
            check_hist(j,k) = 0;
        end
    end
end

for kk = 1:length(idx_95)
    if sum(check_hist(kk,:)) == n_param
        idx_95(kk) = true;
    else
        idx_95(kk) = false;
    end
end

idx_final   = idx_all(idx_95);
smpl_mcmc_q = sample_mcmc(idx_final,:);

end