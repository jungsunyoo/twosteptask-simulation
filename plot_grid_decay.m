function plot_grid_decay(data, includevar)

% Function that plots a surface plot of the linear relationship between
% model-based control and reward rate in a two-step task across a range of 
% the reinforcement learning parameters inverse temperature and learning 
% rate. Most figures reported in Kool, Cushman, & Gershman (2016) were
% created using this function.
%
% USAGE: plot_grid(data), with data the output of the wrapper.m function.
%
% Wouter Kool, Aug 2016
nrbins = 11;
gamma = 0:1/(nrbins-1):1;
% Preprocessing data
% data.slope(b_i,lr_i, g_i, i) = params(1);
if strcmp(includevar, 'beta')
    xax = data.bs;
    xlab = 'Inverse temperature';
    yax = gamma;
    p_data = squeeze(data.slope(:, 11, :, :));
elseif strcmp(includevar, 'alpha')
    xax = data.lrs;
    xlab = 'Learning rate';
    yax = gamma;
    p_data = squeeze(data.slope(11, :, :, :)); 
else
    disp('Error')
end

figure;
% surface(data.lrs,data.bs,mean(data.slope,3));
surface(xax,yax,mean(p_data,3));
caxis([-0.05 0.6]);
zlim([-0.05 0.6]);
colormap('jet');
xlabel(xlab);
ylabel('Remember parameter (gamma)');
zlabel('Standardized linear effect of w on reward');
c=colorbar;
% c.Label.String='Standardized linear effect of w on reward';

view(-33,22);

end

