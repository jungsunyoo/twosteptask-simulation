function data = wrapper_nstates_decay_recovery(nrits, nstates, nrtrials)

% Function that constructs a variable with simulated data for a variety of
% two-step paradigms, as reported in Kool, Cushman, & Gershman (2016).
% Particularly, the function estimates the linear relationship between
% model-based control and reward rate across a range of the reinforcement
% learning parameters inverse temperature and learning rate.
%
% The results are most easily assessed in a surface plot that can be
% constructed using the function plot_grid(data), which takes the output of
% this function as an input.
%
% When the function is run, you will answer a series of questions in order
% to determine which two-step task you would like to simulate.
% To reconstruct the figures in the paper, you should provide the following
% answers:
%
%           bounds      sd      structure   choices     rewards
% Fig 3A    [.25 .75]   .025    stoch       two         probabilities
% Fig 5A    [.25 .75]   .025    deter       two         probabilities
% Fig 6B    [0 1]       .025    stoch       two         probabilities
% Fig 7B    [0 1]       0.2     stoch       two         probabilities
% Fig 8A    [0 1]       0.2     deter       two         probabilities
% Fig 9     [0 1]       0.2     deter       one         probabilities
% Fig 11    [0 1]       0.2     deter       one         points
%
% 
% USAGE: data = wrapper(nrits), where nrits is the number of iterations
% that each linear coefficient should be estimated across the range of RL
% parameters. In the paper, nrits = 1000.
% 
% FIELDS:
%   .simulationFunction - String that indicates which simulation function
%       was used.
%   .bounds - The upper and lower bounds of the reward distributions
%   .choices - The number of choices
%   .rewardrate - reward rates for each participant
%   .slope - [11 x 11 x nrits] matrix with estimated regression
%       coefficients between model-based control and reward rate. Rows
%       indicate inverse temperatures, rows learning rates.
%   .slope - [11 x 11 x 11 x nrits] matrix with reward rates for different
%       RL agents. Rows indicate inverse temperatures, rows learning rates.
%       The third dimension indexes levels of w.
%   .lrs - Array with learning rates used in simulations (0:0.1:1)
%   .bs - Array with inverse temperatures  used in simulations (0:1:10)
%   .lambda - Eligibity trace parameter (default = 0.5)
%
% NOTES:
%
%   1. The output of this function is needed for the function plot_grid
%
% Wouter Kool, Aug 2016

rng shuffle

data = determineTask;

% ws = 0:1/10:1;
nrbins = 11;
bs = 0:10/(nrbins-1):10;
lrs = 0:1/(nrbins-1):1;
% yjs added for introducing decay parameter
gamma = 0:1/(nrbins-1):1;
ws = 0:1/(nrbins-1):1;

% lambda = 0.5;
% nrtrials = 200; %201;


% Add for 5stage
% data structure: [planet 1, planet 2, first stage index]
% states_total = zeros(10,3)
% states_total = [[1,2,1]; [1,3,2]; [1,4,3]; [1,5,4]; 
%     [2,3,5]; [2,4,6]; [2,5,7]; [3,4,8]; [3,5,9]; [4,5,10]];

states_total = nchoosek(1:nstates,2);
states_total = [states_total [1:nchoosek(nstates,2)]'];


corr_table = zeros(nrits, 10);
CI_table = zeros(nrits, 10);
raw_CI_table = zeros(nrits, 5);
raw_recovered = zeros(nrits, 5);
    for i = 1:nrits
% counter=0;
% corr_table = zeros(length(bs) * length(lrs) * length(gamma) * length(ws),8);
% CI_table = zeros(length(bs) * length(lrs) * length(gamma) * length(ws),4);
% for b_i = 1:length(bs) % iterate over different learning rates
%     
%     for lr_i = 1:length(lrs) % and inverse temperatures
%         
% %         rewardrate = zeros(1,length(ws));
%         for g_i = 1:length(gamma)
%             for w_i = 1:length(ws)
%                 counter=counter+1;
%                 avg_param = zeros(nrits, 4);
        lr_i = rand;
        b_i = rand * 10;        
        g_i = rand;
        w_i = rand;
        l_i = rand;
        
%         alpha_sample = rand;
%         alpha_evoked = rand;
%         beta = rand*10;%round(rand*10);%randsample(10, 1);
% %         lambda = 0.5;
%         w = rand;
        

        disp(['Iteration ' num2str(i) ' recovering for true parameters: alpha: ', num2str(lr_i), ' beta: ', num2str(b_i),' w: ', num2str(w_i), ' gamma: ', num2str(g_i), ' lambda: ', num2str(l_i) ])
    %         for i = 1:nrits

    %                     disp(['Iteration ', num2str(i), ' of ', num2str(nrits) ' of nstate=' num2str(nstates)])

        rewards = generate_rewards_nstates(nrtrials,data.bounds,data.sd,data.choices, nstates); %#ok<NASGU>
        
        
        
        
        % Simulation
        
%         % If TD model
        x = [lr_i b_i l_i w_i g_i ];
        output = MBMF_stochastic_1choice_rew_nstates_decay_sim(x,rewards, states_total, nstates);

        % If sampler model
%         x = [alpha_sample alpha_evoked beta lambda w]; 
%         output = MBMF_stochastic_1choice_rew_nstates_sampler_decay_sim(x,rewards, states_total, nstates);

        % Parameter recovery
        results = fit_model(x, output, 'likefun_TD');
    %     avg_param(i,:) = results.x_recovered;
        % making 

    %                     rewardrate(w_i) = sum(output.rewards)/length(output.rewards); % store reward rate for each value of w
    %         end
    %     avg = mean(avg_param,1); % will result in four params
%         corr_table(i, :) = [b_i lr_i w_i g_i results.x_recovered];
        corr_table(i, :) = [lr_i results.alpha b_i results.beta w_i results.w g_i results.gamma l_i results.lambda];
        raw_recovered(i,:) = results.x_recovered;
        rawInterval = sqrt(diag(inv(squeeze(results.hessian))));
%         CI_table(i, :) = rawInterval';
        
        upper_bound = results.x_recovered + 2 * rawInterval';
        lower_bound = results.x_recovered - 2 * rawInterval';
        
        CI_table(i,1) = 1./[1+exp(-upper_bound(1))]; % transformed upper bound for alpha
        CI_table(i,2) = 1./[1+exp(-lower_bound(1))]; % transformed lower bound for alpha
        CI_table(i,3) = 10./[1+exp(-upper_bound(2))]; % transformed upper bound for beta
        CI_table(i,4) = 10./[1+exp(-lower_bound(2))]; % transformed lower bound for beta
        CI_table(i,5) = 1./[1+exp(-upper_bound(3))]; % transformed upper bound for w
        CI_table(i,6) = 1./[1+exp(-lower_bound(3))]; % transformed lower bound for w
        CI_table(i,7) = 1./[1+exp(-upper_bound(4))]; % transformed upper bound for gamma
        CI_table(i,8) = 1./[1+exp(-lower_bound(4))]; % transformed lower bound for gamma   
        CI_table(i,9) = 1./[1+exp(-upper_bound(5))]; % transformed upper bound for lambda
        CI_table(i,10) = 1./[1+exp(-lower_bound(5))]; % transformed lower bound for lambda 

                
    end
%                 params = polyfit(zscore(ws),zscore(rewardrate),1); % estimate linear effect between w and reward rate
                
%                 data.slope(b_i,lr_i, g_i, i) = params(1);
%                 data.rewardrate(:,b_i,lr_i,g_i, i) = rewardrate;
%             end
%             
%             
    

%          
%         end
%     end
% end

% data.lrs = lrs;
% data.bs = bs;
% data.lambda = lambda;
% 
% cd /Users/yoojungsun0/simulation/tradeoffs/simulations/results/with_decay
% 
% eval(['save MBMF_stochastic_1choice_rew_', num2str(nstates) ,'states_decay_sim data']);
data.corrtable = corr_table;
data.CI_table = CI_table;
data.raw_recovered = raw_recovered;
end

