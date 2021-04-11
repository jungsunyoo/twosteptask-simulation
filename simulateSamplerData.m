%%  Simulate Sampler Data
% This script generates simulated data from participants using memory
% samples to make decisions

%% Variables to change
export = 0; %save data?
uniformPars = 1; %draw parameters from uniform distributions?
paramSpace = 0; %cover entire parameter space?
numParticipants = 50; % number of participants to simulate
dataName = 'samplerData'; %name of output

%% Parameter values
if uniformPars == 1 
    if paramSpace == 1 %if we want to generate all possible parameter combinations
    alphas  = linspace(0,1,50);
    betas   = linspace(0,5,50);
    beta_Cs = linspace(-3,3,50);
    alpha_evokeds = linspace(0,1,50);
    paramCombs = combvec(alphas,betas,beta_Cs,alpha_evokeds);
    numCombs = length(paramCombs);
    end
else % Specify average parameters based on Bornstein et al. 2017
    meanAlpha = .5393; %learning rate
    sdAlpha = .0583;
    meanAlphaEvoked = .4386;
    sdAlphaEvoked = .099;
    meanBeta = 2.2869; %explore parameter
    sdBeta = 0.4943;
    meanBeta_c = .5855; %perseverative parameter
    sdBeta_c = .3215;
end

%% Set up

%read in task structure - to get reward probabilities (columns 9 and 10)
%and whether the trial is a probe trial (column 3)
numStructs  = 10;
for structIdx = 1:numStructs
    taskStructAll{structIdx} = csvread(['taskStructures/' num2str(structIdx) '_TaskStructure_0.csv']);
end
% for each simulated participant with same reward probabilities


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
numBandits = 2;
numTrials = length(taskStructAll{1});
numInvalidProbes = 6;
numProbes = sum(taskStructAll{1}(:,3));
numChoices = numTrials - numProbes;

%make vector of image labels
numImgs = numChoices + numInvalidProbes; %
images = linspace(1,numImgs,numImgs);

structIdx = randsample(10,1); % randomly select trialStruct to start on
%% Simulate choices for each participant
for sub = 1:numParticipants
    
    if structIdx > 10; structIdx = 1; end 
    taskStruct = taskStructAll{structIdx}; % select task structure for subject
    structIdx = structIdx + 1; % update index
    
    trialOrder = taskStruct(1:end,3); %order of choice vs. probe trials
    
    images = images(randperm(length(images))); % randomize list of images
    invalidProbes = images(1:numInvalidProbes); % assign 6 as invalid probe trials
    trialImages = images(numInvalidProbes+1:end); % assign the rest as choice trial images

    % Randomize valid-invalid probe order
    valid_invalidOrd = [zeros(numProbes - numInvalidProbes,1); ones(numInvalidProbes, 1)];
    valid_invalidOrd = valid_invalidOrd(randperm(length(valid_invalidOrd)));

    %create vector of images that are available to probe based on previous choice trials
    availableForProbe = [];
    imageVecIndices = [1 1 1]; %generate indices for image ID, invalid probe ID, probe order

    %% Determine parameter values for each participant
    %determine alpha, beta, beta_c, & alpha_evoked for each simulated participant
    if uniformPars == 1 && paramSpace == 1
        alpha        = paramCombs(1,sub);
        beta         = paramCombs(2,sub);
        beta_c       = paramCombs(3,sub);
        alpha_evoked = paramCombs(4,sub);
    elseif uniformPars == 1 && paramSpace == 0
        alpha = rand; %[0 1]
        alpha_evoked = rand; %[0 1]
        beta = 3*rand; %[0 3]
        beta_c = -3 + 6*rand; %[-3 3]
    else
        alpha = meanAlpha + randn(1) * sdAlpha;
             if alpha > 1
                 alpha = 1; %set to 1 if it's above 1
             elseif alpha < 0
                 alpha = .001; %set to .001 if it's below 0
             end
        alpha_evoked = meanAlphaEvoked + randn(1) * sdAlphaEvoked;
             if alpha_evoked > 1
                 alpha_evoked = 1; %set to 1 if it's above 1
             elseif alpha_evoked < 0
                 alpha_evoked = .001; %set to .001 if it's below 0
             end
        beta = meanBeta + randn(1) * sdBeta;
             if beta < 0
                 beta = 0; %set to 0
             end
        beta_c = meanBeta_c + randn(1) * sdBeta_c;
    end

    %% Initialize choice variables
    choose1prob = .5; %initial probability for choosing bandit 1
    combs{1} = {}; %initialize combs structure for bandit 1
    combs{2} = {}; %initialize combs structure for bandit 2

    %% Loop through trials
    for trialIdx = 1:numTrials
        if taskStruct(trialIdx, 3)== 0 % If this trial is a choice trial, make a choice
            if rand(1) < choose1prob %coin flip to determine the choice on this trial
                choice = 1; %choose bandit 1
                if rand(1) < taskStruct(trialIdx,9) %get reward probability for bandit 1
                reward = 1; %determine reward
                else
                reward = -1;
                end
            else
                choice = 2; %choose bandit 2
                if rand(1)  < taskStruct(trialIdx,10) %get reward probability for bandit 2
                    reward  = 1; %determine reward
                else
                    reward  = -1;
                end
            end

             %update the task structure with the choice and reward
            taskStruct(trialIdx, 4) = choice;
            taskStruct(trialIdx, 7) = reward;

            %Update image indices
            taskStruct(trialIdx,8) = trialImages(imageVecIndices(1)); %save image that was presented in the trial structure
            imageVecIndices(1) = imageVecIndices(1) + 1; %add 1 to choice trial index so that the next image gets saved on the next trial
            availableForProbe = [availableForProbe taskStruct(trialIdx,8)]; %add choice trial so it's available to be evoked

        elseif taskStruct(trialIdx, 3)== 1 %if it's a probe trial
            if valid_invalidOrd(imageVecIndices(3)) == 0 % and if it's a valid probe
                probedIdx = randi([1 length(availableForProbe)],1); % select a past choice trial to evoke
                probed = availableForProbe(probedIdx); %retrieve evoked trial image
                taskStruct(trialIdx,8) = probed; %label evoked image
                availableForProbe = [availableForProbe(1:probedIdx-1) availableForProbe(probedIdx+1:end)]; % remove probed trial so that it's no longer available to be evoked again
                probedTrial = taskStruct(taskStruct(1:trialIdx-1,8)==taskStruct(trialIdx,8),2); %find probed trial
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

            %update the task structure with the choice and reward
            taskStruct(trialIdx, 4) = choice;
            taskStruct(trialIdx, 7) = reward;

            %store trial data for the subject
            simData(sub).probet(trialIdx,:) = taskStruct(trialIdx, 3); %store whether it was a choice trial or a probe trial
            simData(sub).choices(trialIdx) = choice; %store the choice made or evoked
            simData(sub).rewards(trialIdx) = reward; %store the reward made or evoked
            simData(sub).trial(trialIdx) = trialIdx; % store the trial number

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

    end
    %store participant data
    simData(sub).alpha = alpha;
    simData(sub).beta = beta;
    simData(sub).alpha_evoked = alpha_evoked;
    simData(sub).beta_c = beta_c;
    simData(sub).images = taskStruct(1:end,8);
end

%% Save data
if export
    save([dataName num2str(numTrials)], 'simData');
end
