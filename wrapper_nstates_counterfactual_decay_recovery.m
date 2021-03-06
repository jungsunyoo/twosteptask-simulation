function data = wrapper_nstates_counterfactual_decay_recovery(nrits, nstates, nrtrials)

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

lambda = 0.5;
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

        lr_i = rand;
        k_i = rand;
        b_i = rand * 10; %randsample(10,1);        
        g_i = rand;
        w_i = rand;
        
%         alpha_sample = rand;
%         alpha_evoked = rand;
%         beta = round(rand*10);%randsample(10, 1);
%         w = rand;
        

        disp(['Iteration ' num2str(i) ' recovering for true parameters: alpha : ', num2str(lr_i), ' kappa : ', num2str(k_i) ' beta: ', num2str(b_i),' w: ', num2str(w_i), ' gamma: ', num2str(g_i) ])

        rewards = generate_rewards_nstates(nrtrials,data.bounds,data.sd,data.choices, nstates); %#ok<NASGU>
                
        % Simulation
        
%         % If TD model
        x = [lr_i k_i b_i lambda w_i g_i];
%         output = MBMF_stochastic_1choice_rew_nstates_decay_sim(x,rewards, states_total, nstates);
        output = MBMF_stochastic_1choice_rew_nstates_counterfactual_decay_sim(x,rewards, states_total, nstates);

        % Parameter recovery
%         results = fit_model(x, output, 'likefun_TD', 'testing');
        results = fit_model_counterfactual(x, output);

        corr_table(i, :) = [lr_i results.alpha k_i results.kappa b_i results.beta w_i results.w g_i results.gamma];
        raw_recovered(i,:) = results.x_recovered;
        rawInterval = sqrt(diag(inv(squeeze(results.hessian))));

        
        upper_bound = results.x_recovered + 2 * rawInterval';
        lower_bound = results.x_recovered - 2 * rawInterval';
        
        CI_table(i,1) = 1./[1+exp(-upper_bound(1))]; % transformed upper bound for alpha1
        CI_table(i,2) = 1./[1+exp(-lower_bound(1))]; % transformed lower bound for alpha1
        CI_table(i,3) = 1./[1+exp(-upper_bound(2))]; % transformed upper bound for alpha2
        CI_table(i,4) = 1./[1+exp(-lower_bound(2))]; % transformed lower bound for alpha2        
        CI_table(i,5) = 10./[1+exp(-upper_bound(3))]; % transformed upper bound for beta
        CI_table(i,6) = 10./[1+exp(-lower_bound(3))]; % transformed lower bound for beta
        CI_table(i,7) = 1./[1+exp(-upper_bound(4))]; % transformed upper bound for w
        CI_table(i,8) = 1./[1+exp(-lower_bound(4))]; % transformed lower bound for w
        CI_table(i,9) = 1./[1+exp(-upper_bound(5))]; % transformed upper bound for gamma
        CI_table(i,10) = 1./[1+exp(-lower_bound(5))]; % transformed lower bound for gamma        

                
    end

data.corrtable = corr_table;
data.CI_table = CI_table;
data.raw_recovered = raw_recovered;
end

