

clear;clc;
close all;
params = {'alpha', 'beta'};
% params = {'alpha'};
% Meta-wrapper for visualizing & saving data
for par = 1:length(params)
    for nstates = 2:5
        clearvars -except nstates par params
%         cd /Users/yoojungsun0/simulation/tradeoffs/simulations/results/with_decay
        cd /Users/yoojungsun0/simulation/tradeoffs/simulations/results/TD
        currname = ['MBMF_stochastic_1choice_rew_', num2str(nstates) ,'states_decay_sim'];
        load(currname)
        plot_grid_decay(data, params{par})
%         savefig([currname, '_' params{par}])
    end
end