function [nloglik]= likfun_TD_counterfactual(params, output, flags)
% likelihood function for TD model

% Specify task structure
numTrials = length(output.rewards);
bandits   = [1 2];

% Parameters

lr1 = params(1);
lr2 = params(2);
beta = params(3);
w = params(4);
gamma = params(5);
lambda = 0.5;

tr = 0.7;                   % common transition probability

nstates = max(output.second_state);
states_total = max(output.first_state(:,3));



% Lower_bound + ((Upper_bound-Lower_bound) * inverse logit(param))
lr1              =  0 + (1 * (1./[1+exp(-lr1)]));
lr2              =  0 + (1 * (1./[1+exp(-lr2)]));
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
    a_cf = output.cf_choice(trial);
    r = output.rewards(trial); %get the reward that was received
    r_cf = output.cf_reward(trial);
    % yjs added
    current_state = output.first_state(trial, 1:2);
    current_state_index = output.first_state(trial,3);
    s = output.second_state(trial);
    s_cf = output.cf_state(trial);
    
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
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lr1*dtQ(1); % update
    
    dtQ(2) = r - Q2(s);                                   % prediction error (2nd choice)
    Q2(s) = Q2(s) + lr1*dtQ(2);                          % update TD value function
    
    % Update counterfactual
    Q2(s_cf) = Q2(s_cf) + lr2 * (r_cf - Q2(s_cf));
    
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lambda*lr1*dtQ(2);                     % eligibility trace
     % Jungsun added for counterfactual eligiility trace
    Qmf(current_state_index,a_cf) = Qmf(current_state_index,a_cf) + lambda * lr2 * (r_cf - Q2(s_cf));
    
    % Decaying unchosen states and/or action pairs for this trial
    for s_ = 1:nstates
        if s_ ~= s && s_ ~= s_cf
            Q2(s_) = Q2(s_) * (1-gamma);
        end
    end
    
    for s_ = 1:states_total
        if s_~=current_state_index % commented this out to enable counterfactual eligibility trace || a_~=a
            for a_ = 1:2           
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

if (true)
    % add in the log prior probability of the parameters
    nloglik = nloglik - log(flags.pp_alpha1(lr1));
    nloglik = nloglik - log(flags.pp_alpha2(lr2));
    nloglik = nloglik - log(flags.pp_beta(beta));
    nloglik = nloglik - log(flags.pp_gamma(gamma));
    nloglik = nloglik - log(flags.pp_w(w));

end

end
