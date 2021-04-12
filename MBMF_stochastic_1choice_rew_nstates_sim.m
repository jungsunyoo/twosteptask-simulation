function output = MBMF_stochastic_1choice_rew_nstates_sim(x,rew, states_total, nstates)

% Mixed model-based / model-free simulation code for a task with stochastic
% transitions, one choice at the second level, and points for the
% second-level bandits.
%
% USAGE: output = MBMF_stochastic_1choice_rew_sim(x,rews)
%
% INPUTS:
%   x - [1 x 4] vector of parameters, where:
%       x(1) - softmax inverse temperature
%       x(2) - learning rate
%       x(3) - eligibility trace decay
%       x(4) - mixing weight
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
lr = x(1);                  % learning rate
b = x(2);                   % softmax inverse temperature
lambda = x(3);              % eligibility trace decay
w = x(4);                   % mixing weight

tr = 0.7;                   % common transition probability

% initialization
% Qmf = zeros(1,2);           % Q(s,a): state-action value function for Q-learning
Qmf = zeros(size(states_total,1),2);
Q2 = zeros(nstates,1);
% Tm = [.3 .7; .7 .3];        % transition matrix
Tm = [.7 .3; .3 .7];
N = size(rew,1);
output.A = zeros(N,1);
output.R = zeros(N,1);
output.S = zeros(N,1);
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
    
    
%     Qmb = Tm' * Q2;
%     Qmb = Tm'*[Q2(planet1); Q2(planet2)];                                           % compute model-based value function
    Qmb = Tm * [Q2(planet1); Q2(planet2)];
    
    tr_prob = rand;
    Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';
%     Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';
%     Q = w*Qmb + (1-w)*Qmf';                                 % mix TD and model value
    if rand < exp(b*Q(1))/sum(exp(b*Q))                     % make choice using softmax
        a = 1;
        if tr_prob < tr
            % This is common transition
            s = planet1; %round(double(tr_prob<tr))+2;
        else
            s = planet2;
        end
    else
        a = 2;
        if tr_prob < tr
            % This is common transition
            s = planet2; %round(double(tr_prob<tr))+2;
        else
            s = planet1;
        end        
%         s = current_state(a); %round(double(tr_prob>tr))+2;
    end
    
    dtQ(1) = Q2(s) - Qmf(current_state_index,a);
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lr*dtQ(1);
%     dtQ(1) = Q2(s-1) - Qmf(a);                              % backup with actual choice (i.e., sarsa)
%     Qmf(a) = Qmf(a) + lr*dtQ(1);                            % update TD value function

    r = rew(t,s);                                         % sample reward
    dtQ(2) = r - Q2(s);                                   % prediction error (2nd choice)
%     r = rew(t,s-1);                                         % sample reward
%     dtQ(2) = r - Q2(s-1);                                   % prediction error (2nd choice)

    Q2(s) = Q2(s) + lr*dtQ(2);                          % update TD value function
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lambda*lr*dtQ(2);                     % eligibility trace
    
%     Q2(s-1) = Q2(s-1) + lr*dtQ(2);                          % update TD value function
%     Qmf(a) = Qmf(a) + lambda*lr*dtQ(2);                     % eligibility trace
    
    % store stuff
    output.A(t,1) = a;
    output.R(t,1) = r;
%     output.S(t,1) = s-1;
    output.S(t,1) = s;
    output.C(t,1) = tr_prob<tr;
    
end

end
