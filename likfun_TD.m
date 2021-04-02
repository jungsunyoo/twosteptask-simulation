function [nloglik]= likfun_TD(params, output, flags)
% likelihood function for TD model

% Specify task structure
numTrials = length(output.rewards);
bandits   = [1 2];

% Parameters

lr = params(1);
beta = params(2);
w = params(3);
gamma = params(4);
lambda = 0.5;

tr = 0.7;                   % common transition probability

nstates = max(output.second_state);
states_total = max(output.first_state(:,3));



% Lower_bound + ((Upper_bound-Lower_bound) * inverse logit(param))
lr              =  0 + (1 * (1./[1+exp(-lr)]));
beta            =  0 + (10 * (1./[1+exp(-beta)]));
w               =  0 + (1 * (1./[1+exp(-w)]));
gamma           =  0 + (1 * (1./[1+exp(-gamma)]));


%   Initialize variables
% vals = [0 0]; %initial expected values for the bandits
% choiceProbs = [.5 .5]; %initial probabilities of choosing each bandi
% yjs added
Qmf = zeros(states_total,2);
Q2 = zeros(nstates,1);
Tm = [.7 .3; .3 .7];
N = numTrials;

% Initialize value and probability matrices
ValV = zeros(numTrials,2);
PV   = zeros(numTrials,2);

for trial = 1:N%numTrials %for each trial

    a = output.choices(trial); %get the participant's choice
    r = output.rewards(trial); %get the reward that was received
    % yjs added
    current_state = output.first_state(trial, 1:2);
    current_state_index = output.first_state(trial,3);
    s = output.second_state(trial);
    
    planet1 = current_state(1);
    planet2 = current_state(2);
    
    Qmb = Tm * [Q2(planet1); Q2(planet2)]; % compute model-based value function
    Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';
       
    % compute likelihood of the observed choices given the params
    vals = Q;    
    ev = exp(beta .* vals);
    sev = sum(ev);
    choiceProbs = ev/sev;
    
    % store values and choice probabilities
    ValV(trial,:) = vals;
    PV(trial,:) = choiceProbs;    
    
    % update values for next trial
    dtQ(1) = Q2(s) - Qmf(current_state_index,a);
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lr*dtQ(1); % update
    
    dtQ(2) = r - Q2(s);                                   % prediction error (2nd choice)
    Q2(s) = Q2(s) + lr*dtQ(2);                          % update TD value function
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lambda*lr*dtQ(2);                     % eligibility trace

    % Decaying unchosen states and/or action pairs for this trial
    for s_ = 1:nstates
        if s_ ~= s
            Q2(s_) = Q2(s_) * (1-gamma);
        end
    end
    
    for s_ = 1:length(states_total)
        for a_ = 1:2
            if s_~=current_state_index || a_~=a
                Qmf(s_,a_) = Qmf(s_,a_) * (1-gamma);
            end
        end
    end    
end

% compute the likelihood: sum of logs of p(observed choice) for all trials

%create matrix with all the probabilities of choosing bandit 1 on trials when the
%participant chose bandit 1
probs1 = PV(output.choices==1, 1); 

%repeat for bandit 2
probs2 = PV(output.choices==2, 2); 

loglik = sum(log(probs1)) + sum(log(probs2));

%compute negative log likelihood
nloglik = loglik*-1;

% % yjs added priors
% flags.pp_gamma =  @(x)(pdf('beta', x, 1.1, 1.1));
% flags.pp_w =  @(x)(pdf('beta', x, 1.1, 1.1));

if (true)
    % add in the log prior probability of the parameters
    nloglik = nloglik - log(flags.pp_alpha(lr));
    nloglik = nloglik - log(flags.pp_beta(beta));
    nloglik = nloglik - log(flags.pp_gamma(gamma));
    nloglik = nloglik - log(flags.pp_w(w));
%     nloglik = nloglik - log(flags.pp_betaC(beta_c));
end

end
