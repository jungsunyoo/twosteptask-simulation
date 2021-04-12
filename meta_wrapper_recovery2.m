clear;clc;



% Meta-wrapper for executing simulations

nrits = 1000;
for nstates = 2:5
    nrtrials = 100 * size(nchoosek(1:nstates,2),1);
%     nrtrials = 200;
    clearvars -except nstates nrtrials nrits
%     data = wrapper_nstates(1000,nstates, nrtrials);
    data = wrapper_nstates_decay_recovery(nrits,nstates, nrtrials);
    savename = ['recovery_moretrials_',num2str(nstates),'states_4params_nrits',num2str(nrits),'.mat'];
    save(savename,'data');
end

% % Meta-wrapper for visualizing & saving data
% for nstates = 2:20
%     clearvars -except nstates 
%     currname = ['MBMF_stochastic_1choice_rew_', num2str(nstates) ,'states_sim'];
%     load(currname)
%     plot_grid(data)
%     savefig(currname)
% end

