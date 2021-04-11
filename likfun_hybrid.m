function [nloglik]= likefun_hybrid(params, data, flags)

% Likelihood function for memory sampling model


% Specify task structure
numProbes = 33;
numTrials = 165;
bandits   = [1 2];
numBandits = 2;

%initialize variables
combs{1} = {};
combs{2} = {};
choose1prob = .5;

% Get parameters to fit
alpha_Samp   = params(1);
beta_Samp    = params(2);
beta_c       = params(3);
alpha_evoked = params(4);
alpha_TD     = params(5);
beta_TD      = params(6);

% Lower_bound + ((Upper_bound-Lower_bound) * inverse logit(param))
alpha_Samp   =  0 + (1 * (1./[1+exp(-alpha_Samp)]));
beta_Samp    =  0 + (5 * (1./[1+exp(-beta_Samp)]));
alpha_evoked =  0 + (1 * (1./[1+exp(-alpha_evoked)]));
alpha_TD     =  0 + (1 * (1./[1+exp(-alpha_TD)]));
beta_TD      =  0 + (5 * (1./[1+exp(-beta_TD)]));
beta_c       = -3 + (6 * (1./[1+exp(-beta_c)]));

vals = [0 0]; %initial expected values for the bandits
ValV = zeros(numTrials,2);

for trialIdx = 1:numTrials %for each trial
    if data.probet(trialIdx) == 0 % if it's a choice trial, compute choice probability
        chosenBandit = data.choices(trialIdx); %get the participant's choice
        otherBandit  = ~(chosenBandit-1)+1; %label the other bandit
        reward       = data.rewards(trialIdx); %get the reward that was received

            for bi = 1:numBandits
                % Calculate bandit combinations for previous trials
                combs{bi}{trialIdx} = data.trial(data.choices(1:trialIdx-1) == bi); %get the trials on which this bandit was chosen or evoked
                banditTmp = combs{bi}{trialIdx}; %create vector of trials on which this bandit was chosen or evoked

                regTrials    = [data.probet(banditTmp)==0]; %split based on whether they were choice trials
                evokedTrials = [data.probet(banditTmp)==1]; %or evoked trials

                if (any(evokedTrials)) %if there were any evoked trials
                    rwdval{bi}  = [data.rewards(banditTmp(regTrials))'; ... %get the reward values from the choice trials for this bandit
                                   data.rewards(banditTmp(evokedTrials))']; %and the reward values from the evoked trials for this bandit
                    pval{bi}    = [alpha_Samp.*((1-alpha_Samp).^(trialIdx-banditTmp(regTrials)))'; ... %compute the probabilities of sampling each choice trial for this bandit
                                   alpha_evoked.*((1-alpha_evoked).^(trialIdx-banditTmp(evokedTrials)))']; %and the probabilities of sampling each evoked trial for this bandit
                else
                    rwdval{bi}  = [data.rewards(banditTmp(regTrials))'];
                    pval{bi}    = [alpha_Samp.*((1-alpha_Samp).^(trialIdx-banditTmp(regTrials)))'];
                end

                if (length(rwdval{bi}) < 1) %if there are no trials for this bandit
                    rwdval{bi} = [0]; %set the reward at 0
                    pval{bi}   = [1]; %and the probability at 1
                end

                pval{bi}    = pval{bi}./sum(pval{bi}); %normalize the probabilities by dividing by their sum
            end %stop looping through bandits

            congruentChoice = 0; %set congruent choice to 0
            if trialIdx > 1 && (data.probet(trialIdx-1) == 0) %if there were bandits on the previous trial and the previous trial was a choice trial
                    congruentChoice = ((data.choices(trialIdx-1) == chosenBandit).*2)-1; %and they chose the same bandit, set this value to 1. If they chose a different bandit, set to -1
            end

           % store values and choice probabilities
           ValV(trialIdx,:) = vals;

           % update values for next trial
           predError = reward - vals(chosenBandit); %compute prediction error
           vals(chosenBandit) = vals(chosenBandit) + (alpha_TD * predError); % update value of chosen bandit

            % Now, for all combinations of rwdvals, compute choice probability
                rvmat = []; %initialize value matrix
                pmat  = []; %initialize probability matrix
                for r = 1:length(rwdval{chosenBandit}) %for the chosen bandit
                    rvmat = [rvmat; rwdval{chosenBandit}(r) - rwdval{otherBandit}(:)]; %update the reward matrix with the difference between the value of the chosen bandit and the other bandits
                    pmat  = [pmat;  pval{chosenBandit}(r).*pval{otherBandit}(:)]; %update the probability matrix with the probability of sampling the chosen bandit multiplied by the probability of sampling the other bandit 
                end

                %compute probability of participant making the choice they
                %made on this trial
                choiceProbs(trialIdx) = sum(pmat.*(1 - 1./(1 + exp(beta_c.*congruentChoice + beta_TD.*vals(chosenBandit) + beta_Samp.*rvmat))));
    end
end

%% compute the likelihood: sum of logs of p(observed choice) for all trials
%compute log likelihood
relevantChoiceProbs = choiceProbs(data.probet == 0);
loglik = sum(log(relevantChoiceProbs));

%compute negative log likelihood
nloglik = loglik*-1;


if (true)
    % add in the log prior probability of the parameters
    nloglik = nloglik - log(flags.pp_alpha(alpha_Samp));
    nloglik = nloglik - log(flags.pp_alpha(alpha_evoked));
    nloglik = nloglik - log(flags.pp_beta(beta_Samp));
    nloglik = nloglik - log(flags.pp_betaC(beta_c));
    nloglik = nloglik - log(flags.pp_alpha(alpha_TD));
    nloglik = nloglik - log(flags.pp_beta(beta_TD));
end

end
