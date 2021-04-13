clear;clc;



% Meta-wrapper for executing simulations
nrtrials = 200;
nrits = 1000;
for nstates = 3:5
    clearvars -except nstates nrtrials nrits
%     data = wrapper_nstates(1000,nstates, nrtrials);
    data = wrapper_nstates_decay(nrits,nstates, nrtrials);
end

% % Meta-wrapper for visualizing & saving data
% for nstates = 2:20
%     clearvars -except nstates 
%     currname = ['MBMF_stochastic_1choice_rew_', num2str(nstates) ,'states_sim'];
%     load(currname)
%     plot_grid(data)
%     savefig(currname)
% end

