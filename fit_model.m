% Environment for testing model fit function
function results = fit_model(true_params, output, likfunToUse, outputName)

% This function takes as input a data set, a likelihood function (0, 1 or 2 -
% 0 for null, 1 for TD and 2 for sampler, and the name where the data should be saved

% if nargin < 2
%     likfunToUse = 1;
% end
% 
% if nargin < 3
%     likfunToUse = 1;
%     outputName = ['results_' datestr(date)];
% end

% true_params = true parameters
lr = true_params(1);                  % learning rate
b = true_params(2);                   % softmax inverse temperature
lambda = true_params(3);              % eligibility trace decay
w = true_params(4);                   % mixing weight
gamma = true_params(5);

% Prior distributions for parameters
flags.pp_alpha = @(x)(pdf('beta', x, 1.1, 1.1));                  % Beta prior for \alphas (from Daw et al 2011 Neuron)
flags.pp_pi = @(x)(pdf('beta', abs(x), 1.1, 1.1));                % symmetric Beta prior for \alpha bump (from Daw et al 2011 Neuron)
% flags.pp_beta = @(x)(pdf('gamma', x, 1.2, 5));                  % Gamma prior for softmax \beta (from Daw et al 2011 Neuron)
flags.pp_beta = @(x)(pdf('beta', x/10, 1.1, 1.1));                % Beta prior for \alphas (from Daw et al 2011 Neuron)
% flags.pp_betaC = @(x)(pdf('beta', (x+3)/6, 1.1, 1.1));            % Beta prior for \alphas (from Daw et al 2011 Neuron)

% yjs added priors
flags.pp_gamma =  @(x)(pdf('beta', x, 1.1, 1.1));
flags.pp_w =  @(x)(pdf('beta', x, 1.1, 1.1));


% Load simulated data
% simData = load(data);
% simData = struct2cell(output);
% simData = simData{1};
% nSubs = length(simData); %number of subjects to fit
nSubs = 1;
dataToFit = output;

%% Set up parameter space

% Alpha - learning /decay rate
param(1).name = 'alpha';
param(1).lb = 0;
param(1).ub = 1;

%Beta - explore/exploit tradeoff
param(2).name = 'beta';
param(2).lb = .001;
param(2).ub = 10;

% w - model-based vs. model-free
param(3).name = 'w';
param(3).lb = 0;
param(3).ub = 1;

% gamma - decay rate of unchosen options
param(4).name = 'gamma';
param(4).lb = 0;
param(4).ub = 1;


%% Important things to pass to fmincon
numParams = length(param); %specify number of parameters
lb = [param.lb]; %specify lower and upper bounds of parameters
ub = [param.ub];

% set number of starts
nStarts = 5;

% define options for fmincon
options = optimset('Display','off');

%% Run fmincon: loop through each subject

% Set up results structure
results.params = numParams;

% Set up results matrix
resultsMat     = zeros(nSubs,6+numParams);

% disp('Fitting TD Model');
%         transformParams = @(x)([1./[1+exp(-x(1))] ... %alpha [0 1]
%                                 5./[1+exp(-x(2))] ... %beta [0 5]
%                                 1./[1+exp(-x(3))] ... %w [0 1]
%                                 1./[1+exp(-x(4))]]); %gamma [0 1]

for sub = 1:nSubs
%     disp(['Fitting subject ' int2str(sub)]);
    
    resultsMat(sub,1) = sub;%dataToFit(sub).subID; % save subID
    
    trueX = [lr b w gamma];
%     try  %display true parameters
%         trueX = 
%         %[dataToFit(sub).alpha dataToFit(sub).beta dataToFit(sub).beta_c dataToFit(sub).alpha_evoked];
%     catch %display true parameters (no alpha evoked)
%         trueX = 
%         [dataToFit(sub).alpha dataToFit(sub).beta dataToFit(sub).beta_c];
%     end

%     disp(['True params: ' num2str(trueX)]);

    nUnchanged = 0;
    starts = 0;
    while nUnchanged < 4     % "Convergence" test -
        starts = starts + 1; %add 1 to starts

        %define likelihood function
        f = @(x) likfun_TD(x,dataToFit, flags);
%         if likfunToUse == 1
%             f = @(x) likfun_TD(x,dataToFit(sub), flags);
%         elseif likfunToUse == 2
%             f = @(x) likfun_sampler(x,dataToFit(sub),flags);
%         elseif likfunToUse == 3
%             f = @(x) likfun_hybrid(x,dataToFit(sub), flags);
%         elseif likfunToUse == 0
%             f = @(x) likfun_null(x, dataToFit(sub), flags);
%         end

        %set fmincon starting values
        x0 = zeros(1,numParams); % initialize at zero
        for p = 1:numParams
            x0(p) = unifrnd(param(p).lb, param(p).ub); %pick random starting values
        end

            % find min negative log likelihood = maximum likelihood for each
            % subject
        [x_recovered, nloglik,exitflag,output,grad,hessian] = fminunc(f, x0, options);
%         [x_recovered, nloglik,exitflag,output,lambda_, grad,hessian] = fmincon(f, x0, options);
            
%             if likfunToUse > 0
%         disp(['subject ' num2str(sub) ': start ' num2str(starts) '(' num2str(nUnchanged) '): params [', num2str(transformParams(x_recovered)) ']']);
%             end
            
            % store min negative log likelihood and associated parameter values
            if starts == 1 || nloglik < results.nLogLik(sub)
                nUnchanged = 0; %reset to 0 if likelihood changes
                results.nLogLik(sub) = nloglik;
%                 resultsMat(sub,2)    = nloglik;

%                 if likfunToUse > 1
%                 results.alphaEvoked(sub)      = 1./[1+exp(-x(4))];
%                 resultsMat(sub,6+numParams)   = 1./[1+exp(-x(4))]; % alphaEvoked
%                 end
                
%                 if likfunToUse == 0
%                     results.betaC(sub) = -3 + 6./[1+exp(-x(1))];
%                     resultsMat(sub,7)  = -3 + 6./[1+exp(-x(1))]; % betaC
%                 end
                    
                
%                 if likfunToUse > 0
% %                 results.betaC(sub) = -3 + 6./[1+exp(-x(3))];
%                 resultsMat(sub,7)  = -3 + 6./[1+exp(-x(3))]; % betaC
%                 end

%                 if likfunToUse < 3
%                     if likfunToUse > 0
                results.alpha(sub) = 1./[1+exp(-x_recovered(1))];
                results.beta(sub)  = 10./[1+exp(-x_recovered(2))];
                results.w(sub) = 1./[1+exp(-x_recovered(3))];
                results.gamma(sub) = 1./[1+exp(-x_recovered(4))];
                results.x_recovered(sub,:) = x_recovered;
%                 resultsMat(sub,8) = 1./[1+exp(-x(1))]; % alpha
%                 resultsMat(sub,9) = 5./[1+exp(-x(2))]; % beta
%                     end
%                 else
%                     results.alphaSamp(sub) = 1./[1+exp(-x(1))];
%                     results.betaSamp(sub)  = 5./[1+exp(-x(2))];
%                     results.alphaTD(sub)   = 1./[1+exp(-x(5))];
%                     results.betaTD(sub)    = 5./[1+exp(-x(6))];
%                     resultsMat(sub,8)     = 1./[1+exp(-x(1))]; % alphaSamp
%                     resultsMat(sub,9)     = 5./[1+exp(-x(2))]; % betaSamp
%                     resultsMat(sub,10)     = 1./[1+exp(-x(5))]; % alphaTD
%                     resultsMat(sub,11)     = 5./[1+exp(-x(6))]; % betaTD
%                 end
% 
%                 % When computing AIC/BIC we have to take back out the prior probabilities of the parameters.
%                 useLogLik = nloglik;
%                 if likfunToUse > 1
%                     if (~isinf(log(flags.pp_alpha(x(4)))) && ~isnan(log(flags.pp_alpha(x(4))))) %alpha_evoked
%                         useLogLik = useLogLik + log(flags.pp_alpha(x(4)));
%                     end
%                 end
%                 if likfunToUse > 0
%                     if (~isinf(log(flags.pp_betaC(x(3)))) && ~isnan(log(flags.pp_betaC(x(3))))) %betaC
%                         useLogLik = useLogLik + log(flags.pp_betaC(x(3)));
%                     end
%                 end
%                 if likfunToUse == 0
%                     if (~isinf(log(flags.pp_betaC(x(1)))) && ~isnan(log(flags.pp_betaC(x(1))))) %betaC
%                         useLogLik = useLogLik + log(flags.pp_betaC(x(1)));
%                     end
%                 end
%                 if likfunToUse < 3
%                     if likfunToUse > 0
%                     if (~isinf(log(flags.pp_alpha(x(1)))) && ~isnan(log(flags.pp_alpha(x(1)))))
%                         useLogLik = useLogLik + log(flags.pp_alpha(x(1)));
%                     end
%                     if (~isinf(log(flags.pp_beta(x(2)))) && ~isnan(log(flags.pp_beta(x(2)))))
%                         useLogLik = useLogLik + log(flags.pp_beta(x(2)));
%                     end
%                     end
%                 else
%                      if (~isinf(log(flags.pp_alpha(x(1)))) && ~isnan(log(flags.pp_alpha(x(1))))) %alphaSamp
%                         useLogLik = useLogLik + log(flags.pp_alpha(x(1)));
%                      end
%                      if (~isinf(log(flags.pp_beta(x(2)))) && ~isnan(log(flags.pp_beta(x(2))))) %betaSamp
%                         useLogLik = useLogLik + log(flags.pp_beta(x(2)));
%                      end
%                      if (~isinf(log(flags.pp_alpha(x(5)))) && ~isnan(log(flags.pp_alpha(x(5))))) %alphaTD
%                         useLogLik = useLogLik + log(flags.pp_alpha(x(5)));
%                      end
%                      if (~isinf(log(flags.pp_beta(x(6)))) && ~isnan(log(flags.pp_beta(x(6))))) %betaTD
%                         useLogLik = useLogLik + log(flags.pp_beta(x(6)));
%                      end
%                 end

%                 results.AIC(sub) = 2*length(x) + 2*useLogLik;
%                 results.BIC(sub) = 0.5*(log(length(dataToFit(sub).probet)*.8) * length(x)) + useLogLik;
                results.model(sub,:) = likfunToUse;
                results.exitflag(sub) = exitflag;
                results.output(sub) = output;
                results.grad(sub,:) = grad;
                results.hessian(sub,:,:,:,:) = hessian;
                results.laplace(sub) = nloglik + (0.5*length(x_recovered)*log(2*pi)) - 0.5*log(det(hessian));

%                 resultsMat(sub,3)     = 2*length(x) + 2*useLogLik; % AIC
%                 resultsMat(sub,4)     = 0.5*(log(length(dataToFit(sub).probet)*.8) * length(x)) + useLogLik; % BIC
%                 resultsMat(sub,5)     = likfunToUse; % model
%                 resultsMat(sub,6)     = nloglik + (0.5*length(x)*log(2*pi)) - 0.5*log(det(hessian)); % laplace
% 
            else
                nUnchanged = nUnchanged + 1;
            end
    end

%     disp(['subject ' num2str(sub) ': Final MAP ' num2str(results.nLogLik(sub)) ...
%           ', final BIC ' num2str(results.BIC(sub)) ...
%           ', final Laplace ' num2str(results.laplace(sub))]);

%     save(outputName, 'results');
    
%     if likfunToUse == 1
%         cHeader = {'subID','nloglik','AIC','BIC','model','laplace','betaC','alpha','beta'}; %dummy header
%     elseif likfunToUse == 2
%         cHeader = {'subID','nloglik','AIC','BIC','model','laplace','betaC','alpha','beta','alphaEvoked'}; %dummy header
%     elseif likfunToUse == 0
%         cHeader = {'subID','nloglik','AIC','BIC','model','laplace','betaC'}; %dummy header
%     else
%         cHeader = {'subID','nloglik','AIC','BIC','model','laplace','betaC','alphaSamp','betaSamp','alphaTD','betaTD','alphaEvoked'}; %dummy header
%     end
%     commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
%     commaHeader = commaHeader(:)';
%     textHeader = cell2mat(commaHeader); %cHeader in text with commas
%     
%     fid = fopen([outputName '.csv'],'w'); 
%     fprintf(fid,'%s\n',textHeader);
%     fclose(fid);
%     
%     dlmwrite([outputName '.csv'],resultsMat,'-append');
    
end

% save(outputName, 'results');

end
