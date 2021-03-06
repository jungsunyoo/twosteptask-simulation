function output = MBMF_stochastic_2choices_p_nstates_decay_sim(x,rew, states_total, nstates)

% Mixed model-based / model-free simulation code for a task with stochastic
% transitions, two choices at the second level, and reward probabilities 
% for the second-level bandits
%
% USAGE: output = MBMF_stochastic_2choices_p_sim(x,rews)
%
% INPUTS:
%   x - [1 x 4] vector of parameters, where:
%       x(1) - softmax inverse temperature
%       x(2) - learning rate
%       x(3) - eligibility trace decay
%       x(4) - mixing weight
%   P - [N x 2 x 2] array storing the rewards, where
%       P(n,s,a) is the probability of observing a binary reward on trial n
%       in second-level state s after taking action a, where N is the
%       number of trials
%
% OUTPUTS:
%   output.A - [N x 2] chosen actions at first and second levels
%   R - [N x 1] second level rewards
%   S - [N x 1] second level states
%
% Wouter Kool, Aug 2016, based on code written by Sam Gershman

% parameters

lr = x(1);                  % learning rate
b = x(2);                   % softmax inverse temperature                
lambda = x(3);              % eligibility trace decay
w = x(4);                   % mixing weight
gamma = x(5);               % decay rate
tr = 0.7;                   % common transition probability

% initialization
% Qd = zeros(3,2);            % Q(s,a): state-action value function for Q-learning
% Tm = [.3 .7; .7 .3];        % transition matrix

Tm = [.7 .3; .3 .7];        % transition matrix

Qmf = zeros(size(states_total,1),2);
Q2 = zeros(nstates,2);
% Tm = [.3 .7; .7 .3];        % transition matrix
% Tm = [.7 .3; .3 .7];
N = size(rew,1);
output.choices = zeros(N,2);
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



% N = size(P,1);
% output.A = zeros(N,2);
% output.R = zeros(N,1);
% output.S = zeros(N,1);
% output.C = zeros(N,1);

% loop through trials
for t = 1:N
    
    current_state = states(t,:);
    current_state_index = states_index(t);
    
    planet1 = current_state(1);
    planet2 = current_state(2);
    
    maxQ = max([Q2(planet1,:); Q2(planet2,:)],[], 2);
    
%     maxQ = max(Qd(2:3,:),[],2);                                     % optimal reward at second level
    Qmb = Tm'*maxQ;                                                  % compute model-based value function
    
    tr_prob = rand;
    Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';
%     Q = w*Qm + (1-w)*Qd(1,:)';                                      % mix TD and model value
    if rand < exp(b*Q(1))/sum(exp(b*Q))                             % make choice using softmax
        a(1) = 1;
%         s = round(double(tr_prob<tr))+2;
        if tr_prob < tr
            s = planet1;
        else
            s = planet2;
        end
        
    else
        a(1) = 2;
%         s = round(double(tr_prob>tr))+2;
        if tr_prob < tr
            s = planet2;
        else
            s = planet1;
        end
    end
    
    if rand < exp(b*Q2(s,1))/sum(exp(b*Q2(s,:)))                    % make choice using softmax and observe transition
        a(2) = 1;
    else
        a(2) = 2;
    end
    
    dtQ(1) = Q2(s,a(2)) - Qmf(current_state_index,a(1));
%     dtQ(1) = Qd(s,a(2)) - Qd(1,a(1));                               % backup with actual choice (i.e., sarsa)
    Qmf(current_state_index,a(1)) = Qmf(current_state_index,a(1)) + lr*dtQ(1);
%     Qd(1,a(1)) = Qd(1,a(1)) + lr*dtQ(1);                            % update TD value function
    
    r = rand < rew(t,s,a(2));                                       % sample reward
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
    
    % store stuff
%     output.A(t,:) = a;
%     output.R(t,1) = r;
%     output.S(t,1) = s-1;
%     output.C(t,1) = tr_prob<tr;
    output.choices(t,1) = a(1);
    output.choices(t,2) = a(2);
    output.rewards(t,1) = r;
    output.first_state(t,:) = [current_state current_state_index];
    output.second_state(t,1) = s;
    output.C(t,1) = tr_prob < tr;
    
    
end

end