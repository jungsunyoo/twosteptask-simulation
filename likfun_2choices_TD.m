function [nloglik]= likfun_2choices_TD(params, output, flags)
% likelihood function for TD model

% Specify task structure
numTrials = length(output.rewards);
bandits   = [1 2];

% Parameters

lr = params(1);
beta = params(2);
w = params(3);
gamma = params(4);
% lambda = params(5);
lambda=0.5;

tr = 0.7;                   % common transition probability

nstates = max(output.second_state);
states_total = max(output.first_state(:,3));



% Lower_bound + ((Upper_bound-Lower_bound) * inverse logit(param))
lr              =  0 + (1 * (1./[1+exp(-lr)]));
beta            =  0 + (10 * (1./[1+exp(-beta)]));
w               =  0 + (1 * (1./[1+exp(-w)]));
gamma           =  0 + (1 * (1./[1+exp(-gamma)]));
% lambda           =  0 + (1 * (1./[1+exp(-lambda)]));

%   Initialize variables
% vals = [0 0]; %initial expected values for the bandits
% choiceProbs = [.5 .5]; %initial probabilities of choosing each bandi
% yjs added
Qmf = zeros(states_total,2);
Q2 = zeros(nstates,2);
Tm = [.7 .3; .3 .7];
N = numTrials;

% Initialize value and probability matrices
ValV = zeros(numTrials,2);
PV   = zeros(numTrials,2);

for trial = 1:N%numTrials %for each trial

    a(1) = output.choices(trial,1); %get the participant's choice
    a(2) = output.choices(trial,2);
    r = output.rewards(trial); %get the reward that was received
    % yjs added
    current_state = output.first_state(trial, 1:2);
    current_state_index = output.first_state(trial,3);
    s = output.second_state(trial);
    
    planet1 = current_state(1);
    planet2 = current_state(2);

    maxQ = max([Q2(planet1,:); Q2(planet2,:)],[], 2);
    
%     maxQ = max(Qd(2:3,:),[],2);                                     % optimal reward at second level
    Qmb = Tm'*maxQ;                                                  % compute model-based value function
    
%     tr_prob = rand;
    Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';    
    
    
    
%     Qmb = Tm * [Q2(planet1); Q2(planet2)]; % compute model-based value function
%     Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';
       
    % compute likelihood of the observed choices given the params
    vals = Q;    
    ev = exp(beta .* vals);
    sev = sum(ev);
    choiceProbs = ev/sev;
    % YJS added: compute likelihood of the observed choices in step 2 given
    % the params
    
    
    
%     if rand < exp(b*Q2(s,1))/sum(exp(b*Q2(s,:)))                    % make choice using softmax and observe transition
%         a(2) = 1;
%     else
%         a(2) = 2;
%     end    
    
    
    % store values and choice probabilities
    ValV(trial,:) = vals;
    PV(trial,1:2) = choiceProbs;
    PV(trial, 3) = exp(beta*Q2(s,1))/sum(exp(beta*Q2(s,:)));
    PV(trial,4) = exp(beta*Q2(s,2))/sum(exp(beta*Q2(s,:)));
 
    dtQ(1) = Q2(s,a(2)) - Qmf(current_state_index,a(1));
%     dtQ(1) = Qd(s,a(2)) - Qd(1,a(1));                               % backup with actual choice (i.e., sarsa)
    Qmf(current_state_index,a(1)) = Qmf(current_state_index,a(1)) + lr*dtQ(1);
%     Qd(1,a(1)) = Qd(1,a(1)) + lr*dtQ(1);                            % update TD value function
    
%     r = rand < rew(t,s,a(2));                                       % sample reward
    dtQ(2) = r - Q2(s,a(2));                                        % prediction error (2nd choice)
    
    Q2(s,a(2)) = Q2(s,a(2)) + lr*dtQ(2);                            % update TD value function
%     Qd(1,a(1)) = Qd(1,a(1)) + lambda*lr*dtQ(2);                     % eligibility trace
    Qmf(current_state_index,a(1)) = Qmf(current_state_index,a(1)) + lambda*lr*dtQ(2);                     % eligibility trace
    
    

    % Decaying unchosen states and/or action pairs for this trial
    %following is for step 2
    for s_ = 1:nstates
        for a_ = 1:2
            if s_ ~= s || a_ ~= a(2)
                Q2(s_,a_) = Q2(s_,a_) * (1-gamma);
            end
        end
    end
    
    % following is for stage 1
    for s_ = 1:states_total
        for a_ = 1:2
            if s_~=current_state_index || a_~=a(1)
                Qmf(s_,a_) = Qmf(s_,a_) * (1-gamma);
            end
        end
    end    
end

% compute the likelihood: sum of logs of p(observed choice) for all trials

%create matrix with all the probabilities of choosing bandit 1 on trials when the
%participant chose bandit 1 in stage 1
probs1_1 = PV(output.choices(:,1)==1, 1); 

%repeat for bandit 2 in stage 1
probs1_2 = PV(output.choices(:,1)==2, 2); 

% YJS added: STAGE 2
probs2_1 = PV(output.choices(:,2)==1, 3);
probs2_2 = PV(output.choices(:,2)==2, 4);

loglik = sum(log(probs1_1)) + sum(log(probs1_2)) + sum(log(probs2_1)) + sum(log(probs2_2));

%compute negative log likelihood
nloglik = loglik*-1;

if (true)
    % add in the log prior probability of the parameters
    nloglik = nloglik - log(flags.pp_alpha(lr));
    nloglik = nloglik - log(flags.pp_beta(beta));
    nloglik = nloglik - log(flags.pp_gamma(gamma));
    nloglik = nloglik - log(flags.pp_w(w));
%     nloglik = nloglik - log(flags.pp_lambda(lambda));

end

end
