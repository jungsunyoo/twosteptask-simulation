function output = MBMF_stochastic_1choice_rew_nstates_sampler_decay_sim(x, rew, states_total, nstates)

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
% Jungsun Yoo, Apr 2021, based on code written by WOouter Kool and Sam
% Gershman

% Jungsun Yoo added for determining trial order (images, probes)
% number of probe trials: approximately N/3 (Bornstein et. al 2017: 130
% choices, 32 probes (26 valid / 6 novel), average 39 trials in the past)

% For now, just use predefined TaskStructure
%read in task structure - to get reward probabilities (columns 9 and 10)
%and whether the trial is a probe trial (column 3)

% structIdx = randi([1,10],1);
structIdx = randsample(10,1); % randomly select trialStruct to start on
taskStruct = csvread(['taskStructures/' num2str(structIdx) '_TaskStructure_0.csv']);

numBandits = 2;
numTrials = length(taskStruct); %length(taskStructAll{1});
numInvalidProbes = 6;
numProbes = sum(taskStruct(:,3));
numChoices = numTrials - numProbes;

% for each simulated participant with same reward probabilities

% Needed parameters:
% alpha_sample, alpha_evoked, beta, lambda, w, (beta_c),

% parameters

alpha_sample = x(1);        % decay rate for choice trials
alpha_evoked = x(2);        % decay rate for evoked trials
beta = x(3);                % softmax inverse temperature
lambda = x(4);              % eleigibility trace decay
w = x(5);                   % mixing weight between model-based and model-free
% gamma is not needed here because alpha represents decay rate

tr = 0.7;                   % common transition probability

% initialization
% Qmf = zeros(1,2);           % Q(s,a): state-action value function for Q-learning
Qmf = zeros(size(states_total,1),2);
Q2 = zeros(nstates,1);
% Tm = [.3 .7; .7 .3];        % transition matrix
Tm = [.7 .3; .3 .7];
% N = size(rew,1); % Trial number
output.choices = zeros(numTrials,1);
output.rewards = zeros(numTrials,1);
output.first_state = zeros(numTrials,3);
output.second_state = zeros(numTrials,1);
output.C = zeros(numTrials,1);

% Jungsun Yoo added for multiple number of second states


d = floor(numTrials/nchoosek(nstates,2));
r = numTrials - nchoosek(nstates,2)*d; % remainder
pairs = repmat(states_total,d, 1);
pairs = [pairs; states_total(1:r,:)];
pairs = pairs(randperm(size(pairs,1)),:);

states = pairs(:,1:2);
states_index = pairs(:,3);




%Task structure - this is what data from a real participant will look like
%1: Subject ID
%2: Trial Number
%3: Choice Trial (0) or Memory Probe (1)
%4: Choice
%5: Choice RT
%6: Deep encoding response
%7: Reward
%8: Trial-unique image
%specify task structure



%make vector of image labels
numImgs = numChoices + numInvalidProbes; %
images = linspace(1,numImgs,numImgs);

images = images(randperm(length(images))); % randomize list of images
invalidProbes = images(1:numInvalidProbes); % assign 6 as invalid probe trials
trialImages = images(numInvalidProbes+1:end); % assign the rest as choice trial images

% Randomize valid-invalid probe order
valid_invalidOrd = [zeros(numProbes - numInvalidProbes,1); ones(numInvalidProbes, 1)];
valid_invalidOrd = valid_invalidOrd(randperm(length(valid_invalidOrd)));

%create vector of images that are available to probe based on previous choice trials
availableForProbe = [];
imageVecIndices = [1 1 1]; %generate indices for image ID, invalid probe ID, probe order

%% Initialize choice variables
choose1prob = .5; %initial probability for choosing bandit 1
combs{1} = {}; %initialize combs structure for bandit 1
combs{2} = {}; %initialize combs structure for bandit 2

trialOrder = taskStruct(1:end,3);

% loop through trials
for t = 1:numTrials
    if trialOrder(t) == 0 % choice trial
        current_state = states(t,:);
        current_state_index = states_index(t);
        
        planet1 = current_state(1);
        planet2 = current_state(2);


            
        Qmb = Tm * [Q2(planet1); Q2(planet2)];  % compute model-based value function
        
        tr_prob = rand;
        Q = w*Qmb + (1-w)*Qmf(current_state_index,:)';
        
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
        
        
        r = rew(t,s);                                         % sample reward
        dtQ(2) = r - Q2(s);                                   % prediction error (2nd choice)
        
        
        Q2(s) = Q2(s) + lr*dtQ(2);                          % update TD value function
        Qmf(current_state_index,a) = Qmf(current_state_index,a) + lambda*lr*dtQ(2);                     % eligibility trace

        %====
        %update the task structure with the choice and reward
        taskStruct(t, 4) = choice;
        taskStruct(t, 7) = reward;

        %Update image indices
        taskStruct(t,8) = trialImages(imageVecIndices(1)); %save image that was presented in the trial structure
        imageVecIndices(1) = imageVecIndices(1) + 1; %add 1 to choice trial index so that the next image gets saved on the next trial
        availableForProbe = [availableForProbe taskStruct(t,8)]; %add choice trial so it's available to be evoked        
        %====        
        
    else % Evoked trial (probe trial)
        if valid_invalidOrd(imageVecIndices(3)) == 0 % and if it's a valid probe
            probedIdx = randi([1 length(availableForProbe)],1); % select a past choice trial to evoke
            probed = availableForProbe(probedIdx); %retrieve evoked trial image
            taskStruct(trialIdx,8) = probed; %label evoked image
            availableForProbe = [availableForProbe(1:probedIdx-1) availableForProbe(probedIdx+1:end)]; % remove probed trial so that it's no longer available to be evoked again
            probedTrial = taskStruct(taskStruct(1:t-1,8)==taskStruct(t,8),2); %find probed trial
            choice = taskStruct(probedTrial, 4); %store the choice made on the evoked trial
            reward = taskStruct(probedTrial, 7); %store the reward from the evoked trial
        else
            taskStruct(trialIdx,8) = invalidProbes(imageVecIndices(2)); % select invalid probe image
            imageVecIndices(2) = imageVecIndices(2) + 1; %update invalid probe index
            choice = 0; %label invalid probe evoked "choices" as 0
            reward = 0; %label invalid probe evoked "rewards" as 0
        end
        imageVecIndices(3) = imageVecIndices(3) + 1; %update index that determines what type of probe trial it is        
    end
    
    %% Update choice probability for next trial
    for bi = 1:numBandits %loop through bandits
        % Calculate bandit combinations that can be sampled on next
        % trial
        combs{bi}{trialIdx} = taskStruct((taskStruct(1:(trialIdx),4) == bi),2); %find previous trials when this bandit was selected
        banditTmp = combs{bi}{trialIdx}; %get previous trials when this bandit was selected

        regTrials    = [taskStruct(banditTmp, 3)==0]; %split based on whether they were choice trials
        evokedTrials = [taskStruct(banditTmp, 3)==1]; %or evoked trials

        if (any(evokedTrials)) %if there were any evoked trials
            rwdval{bi}  = [taskStruct(banditTmp(regTrials), 7)' ... %get the reward values from the choice trials for this bandit
                           taskStruct(banditTmp(evokedTrials), 7)']; %and the reward values from the evoked trials for this bandit
            pval{bi}    = [alpha.*((1-alpha).^(trialIdx-banditTmp(regTrials)))' ... %compute the probabilities of sampling each choice trial for this bandit
                           alpha_evoked.*((1-alpha_evoked).^(trialIdx-banditTmp(evokedTrials)))']; %and the probabilities of sampling each evoked trial for this bandit
        else %otherwise do the same thing for the choice trials only
            rwdval{bi}  = [taskStruct(banditTmp(regTrials), 7)'];
            pval{bi}    = [alpha.*((1-alpha).^(trialIdx-banditTmp(regTrials)))'];
        end

        if (length(rwdval{bi}) < 1) %if there are no trials for this bandit
           rwdval{bi} = [0]; %set the reward at 0
           pval{bi}   = [1]; %and the probability at 1
        end
        pval{bi}    = pval{bi}./sum(pval{bi}); %normalize the probabilities by dividing by their sum
    end %stop looping through bandits

     % compute choice probabilities
      congruentChoice = [0 0]; %initialize congruent choice vector at [0 0]
      if taskStruct(trialIdx, 3)== 0 %if it's a choice trial, compute congruent choice
        congruentChoice(choice) = 1; %if they choose this bandit again on the next trial, it will be repeated
      end

      % Now, for all combinations of rwdvals, compute choice probability
        rvmat = []; %initialize value matrix
        pmat  = []; %initialize probability matrix
        for r = 1:length(rwdval{1}) %for bandit 1
            rvmat = [rvmat; rwdval{1}(r) - rwdval{2}(:)]; %update the reward matrix with the difference between the value of bandit 1 and bandit 2
            pmat  = [pmat;  pval{1}(r).*pval{2}(:)]; %update the probability matrix with the probability of sampling these reward combinations
        end

    %put values into softmax choice function
    choose1prob = sum(pmat .* (1-1./(1+ exp(beta_c .* (congruentChoice(1) - congruentChoice(2)) + beta .* rvmat))));

    %store choice probabilities
    simData(sub).choiceProbs(trialIdx,:) = [choose1prob 1-choose1prob];
    
    output.choices(t,1) = a;
    output.rewards(t,1) = r;
    output.second_state(t,1) = s;
    output.C(t,1) = tr_prob<tr;
    output.first_state(t,:) = [current_state current_state_index];
    
end

end
