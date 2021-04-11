function output = MBMF_stochastic_1choice_rew_nstates_counter_1alpha_decay_sim(x,rew, states_total, nstates)

% Mixed model-based / model-free simulation code for a task with stochastic
% transitions, one choice at the second level, and points for the
% second-level bandits.
%
% USAGE: output = MBMF_stochastic_1choice_rew_sim(x,rews)
%
% INPUTS:
%   x - [1 x 4] vector of parameters, where:
%       x(1) - learning rate
%       x(2) - softmax inverse temperature
%       x(3) - eligibility trace decay
%       x(4) - mixing weight
%       x(5) - decay rate gamma
%   rews - [N x 2] array storing the rewards, where
%       rews(n,s) is the payoff on trial n in second-level state s after 
%       taking action a, where N is the number of trials
%
% OUTPUTS:
%   output.A - [N x 1] chosen actions at first and second levels
%   R - [N x 1] second level rewards
%   S - [N x 1] second level states
%   C - [N x 1] type of transition (common [1] or rare [0])
%
% Wouter Kool, Aug 2016, based on code written by Sam Gershman

% parameters
lr1 = x(1);                  % learning rate for real options
% lr2 = x(2);                 % learning rate for counterfactual options
b = x(2);                   % softmax inverse temperature

lambda = x(3);              % eligibility trace decay
w = x(4);                   % mixing weight
gamma = x(5);

tr = 0.7;                   % common transition probability

% initialization
% Qmf = zeros(1,2);           % Q(s,a): state-action value function for Q-learning
Qmf = zeros(size(states_total,1),2);
Q2 = zeros(nstates,1);
% Tm = [.3 .7; .7 .3];        % transition matrix
Tm = [.7 .3; .3 .7];
N = size(rew,1);
output.choices = zeros(N,1);
output.rewards = zeros(N,1);
output.first_state = zeros(N,3);
output.second_state = zeros(N,1);
output.C = zeros(N,1);

% Jungsun Yoo added for multiple number of second states


d = floor(N/nchoosek(nstates,2));
r = N - nchoosek(nstates,2)*d; % remainder
pairs = repmat(states_total,d, 1);
pairs = [pairs; states_total(1:r,:)];
pairs = pairs(randperm(size(pairs,1)),:);

states = pairs(:,1:2);
states_index = pairs(:,3);

% loop through trials
for t = 1:N
    
    current_state = states(t,:);
    current_state_index = states_index(t);
    
    planet1 = current_state(1);
    planet2 = current_state(2);
    
    Qmb = Tm * [Q2(planet1); Q2(planet2)];
    
    tr_prob = rand;
    Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';

    if rand < exp(b*Q(1))/sum(exp(b*Q))                     % make choice using softmax
        a = 1;
        a_cf = 2;
        if tr_prob < tr
            % This is common transition
            s = planet1; %round(double(tr_prob<tr))+2;
            s_cf = planet2; %s_cf stands for counterfactual state
        else
            s = planet2;
            s_cf = planet1;
        end
    else
        a = 2;
        a_cf = 1;
        if tr_prob < tr
            % This is common transition
            s = planet2; %round(double(tr_prob<tr))+2;
            s_cf = planet1;
        else
            s = planet1;
            s_cf = planet2;
        end        
%         s = current_state(a); %round(double(tr_prob>tr))+2;
    end
    
    
    dtQ(1) = Q2(s) - Qmf(current_state_index,a);    % backup with actual choice (i.e., sarsa)                    
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lr1*dtQ(1); % update TD value function
                         
    r = rew(t,s);                                         % sample reward
    r_cf = rew(t, s_cf); % counterfactual reward
    
    dtQ(2) = r - Q2(s);                                   % prediction error (2nd choice)

    Q2(s) = Q2(s) + lr1*dtQ(2);                          % update TD value function
    
    % Update counterfacual TD value
    Q2(s_cf) = Q2(s_cf) + lr1 * (r_cf - Q2(s_cf));
    
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lambda*lr1*dtQ(2);                     % eligibility trace
    % Jungsun added for counterfactual eligiility trace
    Qmf(current_state_index,a_cf) = Qmf(current_state_index,a_cf) + lambda * lr1 * (r_cf - Q2(s_cf));
    
    
    % eligibility trace for cf too???
    
    
    % Decaying unchosen states and/or action pairs for this trial
    % don't decay counterfacually observed 
    for s_ = 1:nstates
        if s_ ~= s && s_ ~= s_cf % this part changed 
            Q2(s_) = Q2(s_) * (1-gamma);
        end
    end
    
    for s_ = 1:size(states_total,1)%length(states_total)
        if s_~=current_state_index % commented this out to enable counterfactual eligibility trace || a_~=a
            for a_ = 1:2
           
                Qmf(s_,a_) = Qmf(s_,a_) * (1-gamma);
            end
        end
    end
            
    output.choices(t,1) = a;
    output.rewards(t,1) = r;
    output.second_state(t,1) = s;
    output.cf_state(t,1) = s_cf;
    output.cf_reward(t,1) = r_cf;
    output.cf_choice(t,1) = a_cf;
    output.C(t,1) = tr_prob<tr;
    output.first_state(t,:) = [current_state current_state_index];
    
end

end
