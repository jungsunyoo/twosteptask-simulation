%% Simulate TD Data
% This script generates simulated data from participants using standard temporal
% difference learning to make decisions

%% Variables to change
export = 1; %save data?
uniformPars = 1; %draw parameters from uniform distributions over whole parameter space?
paramSpace = 0; %cover entire parameter space?
numParticipants = 1000; %number of participants to simulate
dataName = 'TdData_new'; %name of output

%% Parameter Values
if uniformPars == 1
    if paramSpace == 1
    alphas  = linspace(0,1,50);
    betas   = linspace(0,5,50);
    beta_Cs = linspace(-3,3,50);
    paramCombs = combvec(alphas,betas,beta_Cs);
    numCombs = length(paramCombs);
    end
else
    % Specify average parameters based on Bornstein et al. 2017
    meanAlpha = .5552; %learning rate
    sdAlpha = .0862;
    meanBeta = 1.7551; %explore parameter
    sdBeta = 0.6845;
    meanBeta_c = -.093; %perseverative parameter
    sdBeta_c = .2354;
end

%% Set up

%read in task structures - to get reward probabilities (columns 9 and 10)
%and whether the trial is a probe trial (column 3)
numStructs  = 10;
for structIdx = 1:numStructs
    taskStructAll{structIdx} = csvread(['taskStructures/' num2str(structIdx) '_TaskStructure_0.csv']);
end

%Task structure - this is what data from a real participant will look like
%1: Subject ID
%2: Trial Number
%3: Choice Trial (0) or Memory Probe (1)
%4: Choice
%5: Choice RT
%6: Deep encoding response
%7: Reward
%8: Trial-unique image

% Specify task structure
bandits = [1 2];
numTrials = length(taskStructAll{1});
numInvalidProbes = 6;
numProbes = sum(taskStructAll{1}(:,3));
numChoices = numTrials - numProbes;

%make vector of image labels
numImgs = numChoices + numInvalidProbes; %
images = linspace(1,numImgs,numImgs);

structIdx = randsample(10,1); % randomly select trialStruct to start on
%% Simulate participant choices
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
    %determine alpha, beta, & beta_c for each simulated participant
    if uniformPars == 1 && paramSpace == 1
        alpha        = paramCombs(1,sub);
        beta         = paramCombs(2,sub);
        beta_c       = paramCombs(3,sub);
    elseif uniformPars == 1 && paramSpace == 0
        alpha = rand; %[0 1]
        beta = 3*rand; %[0 3]
        beta_c = -3 + 6*rand; %[-3 3]
    else
        alpha = meanAlpha + randn(1) * sdAlpha;
             if alpha > 1
                 alpha = 1; %set to 1 if it's above 1
             elseif alpha < 0
                 alpha = .001; %set to .001 if it's below 0
             end
        beta = meanBeta + randn(1) * sdBeta; 
             if beta < 0
                 beta = 0; %set to 0
             end
        beta_c = meanBeta_c + randn(1) * sdBeta_c; 
    end
    

    %% Initialize choice variables
    vals = [0 0]; %initial expected values for the bandits
    choiceProbs = [.5 .5]; %initial probabilities of choosing each bandit
    
    %% Loop through trials
        for trialIdx = 1:numTrials
            if taskStruct(trialIdx,3) == 0 %if it's a choice trial
                if rand(1) < choiceProbs(1) %coin flip to determine the choice on this trial
                    choice = 1;
                    if rand(1)  < taskStruct(trialIdx,9) %get reward probability for bandit 1
                        reward = 1; %determine reward
                    else
                        reward = -1;
                    end
                        
                else
                    choice = 2;
                    if rand(1) < taskStruct(trialIdx,10) %get reward probability for bandit 2
                       reward = 1;
                    else
                        reward = -1;
                    end
                end
                
                %Update image indices
                taskStruct(trialIdx,8) = trialImages(imageVecIndices(1)); %save image that was presented in the trial structure
                imageVecIndices(1) = imageVecIndices(1) + 1; %add 1 to choice trial index so that the next image gets saved on the next trial
                availableForProbe = [availableForProbe taskStruct(trialIdx,8)]; %add choice trial so it's available to be evoked
            
                %store trial data            
                simData(sub).probet(trialIdx,:) = 0; %store that it was not a probe trial
                simData(sub).choices(trialIdx) = choice; %store the choice made on each trial
                simData(sub).rewards(trialIdx) = reward; %store the reward on each trial
                simData(sub).trial(trialIdx) = trialIdx; %store trial number

                %Update values for next trial
                predError = reward - vals(choice); %compute prediction error
                vals(choice) = vals(choice) + (alpha * predError); % update value of chosen bandit
                
                % compute choice probabilities based on values
                congruentChoice = [0 0]; %initialize congruent choice vector at [0 0]
                congruentChoice(choice) = 1; %if they choose this bandit again, it's repeated
                
            %exponentiate the value, adding in the perseverative
            %variable if the choice would be repeated
            congruentChoiceDiff = congruentChoice(1) - congruentChoice(2);
            valDiff = vals(1) - vals(2);
            choose1prob = (1 - 1./(1 + exp(beta_c.*congruentChoiceDiff + beta.*valDiff)));
            choiceProbs = [choose1prob 1-choose1prob];

            
            elseif taskStruct(trialIdx,3) == 1 %if it's a probe trial
                if valid_invalidOrd(imageVecIndices(3)) == 0 % and a valid probe trial
                    probedIdx = randi([1 length(availableForProbe)],1); % select a past choice trial to probe
                    probed = availableForProbe(probedIdx); %retrieve evoked trial image 
                    taskStruct(trialIdx,8) = probed; %label evoked image
                    availableForProbe = [availableForProbe(1:probedIdx-1) availableForProbe(probedIdx+1:end)]; % remove probed trial from available
                    choice = simData(sub).choices(probedIdx); %retrieve the choice made on the evoked trial
                    reward = simData(sub).rewards(probedIdx); %retrieve the reward from the evoked trial
                else
                    % invalid probe
                    taskStruct(trialIdx,8) = invalidProbes(imageVecIndices(2)); % invalid probe
                    imageVecIndices(2) = imageVecIndices(2) + 1;
                    choice = 0; %store evoked choice as 0
                    reward = 0; %store evoked reward as 0
                end
                
                %updated probed image index so that the next one gets
                %selected on the next probe trial
                imageVecIndices(3) = imageVecIndices(3) + 1; 
                
                %store trial data without updating choice probabilities          
                simData(sub).probet(trialIdx,:) = 1; %store that it was a probe trial
                simData(sub).choices(trialIdx) = choice; %store the choice made on each trial
                simData(sub).rewards(trialIdx) = reward; %store the reward on each trial
                simData(sub).trial(trialIdx) = trialIdx; %store trial number
                 
           % compute choice probabilities without updating values
            congruentChoice = [0 0]; %initialize congruent choice vector at [0 0]
            %no repeated choices after evoked trials
                
            %exponentiate the value, adding in the perseverative
            %variable if the choice would be repeated
            congruentChoiceDiff = congruentChoice(1) - congruentChoice(2);
            valDiff = vals(1) - vals(2);
            choose1prob = (1 - 1./(1 + exp(beta_c.*congruentChoiceDiff + beta.*valDiff)));
            choiceProbs = [choose1prob 1-choose1prob];
            end
           
        end
        
    %store subject parameters
    simData(sub).alpha = alpha;
    simData(sub).beta = beta;
    simData(sub).beta_c = beta_c;
    simData(sub).images = taskStruct(1:end,8);
end

%% Save data
if export
    save([dataName num2str(numTrials)], 'simData');
end
