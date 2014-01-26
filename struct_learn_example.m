% THINGS TO TRY:
% When the loss function is uniformly 0 it seems like the algorithm should
% pick w_star such as to minimize the maximum energy difference between any
% y and the gold_standard_y.

% Try larger LAMBDAs to see what happens when the data term is favoured
% more.

% Clear
clear all

% ======= SETTINGS ========================================================

% 'Script behaviour'
PLOT_L_FUNCTION = true;
PLOT_LOSS_FUNCTION = true;

% Define parameters

NUMBER_OF_FEATURES = 2;
LENGTH_OF_Y = 5;
LAMBDA = 100;

% =========================================================================

% Generate ys
ys = de2bi(0:2^LENGTH_OF_Y-1);


% Randomly pick good and bad edges
good_edges = rand(1,LENGTH_OF_Y) < .5;


% Gold standard is the one with only good edges
gold_standard_y = double(good_edges);


% Randomly pick means for each feature: one for good edges, one for bad
% ones.
% Convention: good: first column, bad: second column
feature_means = myUtilities.scale(rand(NUMBER_OF_FEATURES-1,2),5,15);


% For each edge generate a number that represents the value of a particular
% feature (alpha numbers in paper)
% Feature one should be constant
% Convention: dimension 1: edge, dimension 2: feature
alphas = randn(LENGTH_OF_Y,NUMBER_OF_FEATURES-1); % zeros(LENGTH_OF_Y,NUMBER_OF_FEATURES);
feature_means_extended = cat(3,...
                        repmat(feature_means(:,1)',LENGTH_OF_Y,1),...
                        repmat(feature_means(:,2)',LENGTH_OF_Y,1));
good_edges_extended = repmat(good_edges',1,NUMBER_OF_FEATURES-1);
alphas_mean_adjusted = alphas + ...
    good_edges_extended .* feature_means_extended(:,:,1) - ...
    (good_edges_extended - 1) .* feature_means_extended(:,:,2);
alphas_mean_adjusted = cat(2,ones(LENGTH_OF_Y,1),alphas_mean_adjusted);

% Plot alphas
alpha_handle = figure();
hold on
if size(alphas_mean_adjusted,2) == 1
    scatter(alphas_mean_adjusted(good_edges,1),ones(size(alphas_mean_adjusted(good_edges,:))),'g');
    scatter(alphas_mean_adjusted(~good_edges,1),ones(size(alphas_mean_adjusted(~good_edges,:))),'r');
    title('Feature space (edges)');
    xlabel('Feature 1');
    ylabel('Constant');
elseif size(alphas_mean_adjusted,2) == 2
    scatter(alphas_mean_adjusted(good_edges,1),alphas_mean_adjusted(good_edges,2),'g');
    scatter(alphas_mean_adjusted(~good_edges,1),alphas_mean_adjusted(~good_edges,2),'r');
    title('Feature space (edges)');
    xlabel('Feature 1');
    ylabel('Feature 2');
elseif size(alphas_mean_adjusted,2) == 3
    scatter3(alphas_mean_adjusted(good_edges,1),alphas_mean_adjusted(good_edges,2),alphas_mean_adjusted(good_edges,3),'g');
    scatter3(alphas_mean_adjusted(~good_edges,1),alphas_mean_adjusted(~good_edges,2),alphas_mean_adjusted(~good_edges,3),'r');
    title('Feature space (edges)');
    xlabel('Feature 1');
    ylabel('Feature 2');
    zlabel('Feature 3');
end


% Calculate feature vectors for all ys from alphas
features = ys * alphas_mean_adjusted;

gold_standard_features = gold_standard_y * alphas_mean_adjusted;

% Training

min_search_function_w_handle = @(w)min_search_function(LAMBDA, gold_standard_y, ys, gold_standard_features, features, w);

w_star = fminsearch(min_search_function_w_handle,zeros(1,NUMBER_OF_FEATURES));
%w_star = [-1,-1];

% Add w_star to alpha plot; add 'descision boundary'
figure(alpha_handle)
if size(alphas_mean_adjusted,2) == 2
    plot([0;w_star(1)],[0,w_star(2)]);
    plot(0,0,'b*');
    % descision boundary
    boundary_vector = null(w_star);
    xlimits = xlim();
    point_1 = boundary_vector * xlimits(1) / boundary_vector(1);
    point_2 = boundary_vector * xlimits(2) / boundary_vector(1);
    plot([point_1(1),point_2(1)],[point_1(2),point_2(2)],'r');
elseif size(alphas_mean_adjusted,2) == 3
    plot3([0;w_star(1)],[0,w_star(2)],[0;w_star(3)]);
    plot3(0,0,0,'b*');
end


% Get an idea of what the l function looks like:
if PLOT_L_FUNCTION
    if NUMBER_OF_FEATURES == 2
        l_w_handle = @(w)l(gold_standard_y, ys, gold_standard_features, features, w);
        [w_probes_1,w_probes_2] = meshgrid(-1.5:0.1:1.5);
        l_res = zeros(size(w_probes_1));
        for i = 1 : size(w_probes_1,1)
            for j = 1 : size(w_probes_1,2)
                l_res(i,j) = l_w_handle([w_probes_1(i,j),w_probes_2(i,j)]);
            end
        end
        figure();
        mesh(l_res);
    else
        disp('Can only plot the l function if the number of features is two (or less, but not implemented yet)');
    end
end


% Get an idea of what the loss function looks like:
if PLOT_LOSS_FUNCTION
    if NUMBER_OF_FEATURES == 2
        min_search_function_res = zeros(size(w_probes_1));
        for i = 1 : size(w_probes_1,1)
            for j = 1 : size(w_probes_1,2)
                min_search_function_res(i,j) = min_search_function_w_handle([w_probes_1(i,j),w_probes_2(i,j)]);
            end
        end
        figure();
        mesh(min_search_function_res);
    else
        disp('Can only plot the loss function if the number of features is two (or less, but not implemented yet)');
    end
end


% Do we allways get the same w_star? --> yes, more or less.
same = true;
for i = 1 : 10
    w_star_dash = fminsearch(min_search_function_w_handle,rand(1,NUMBER_OF_FEATURES));
    same = same && all(w_star_dash == w_star);
end

% Calculate y_star
y_stars = find(features*w_star' == min(features*w_star'));

% Visualization
if NUMBER_OF_FEATURES == 1
    figure()
    % Plot features with losses encoded in size
    losses = bsxfun(@loss_function,repmat(gold_standard_y,size(ys,1),1)',ys')';
    energies = features * w_star';
    scatter(features,zeros(size(features)),myUtilities.scale(losses,20,300),...
        repmat(myUtilities.scale(energies,0,1),1,3).*ones(size(ys,1),3));
    % Plot y_stars in green
    hold on
    for y_star_num = 1 : length(y_stars)
        plot(features(y_stars(y_star_num),1),0,'r*');
    end
    % Plot features of gold standard in red
    plot(gold_standard_features,0,'g*');
    
    % Plot direction of w_star
    plot([0;w_star(1)],[0,0]);
    plot(0,0,'b*');
    
    title('loss size coded, energy color coded; green: gold st., red: picked');
    xlabel('Feature 1');
    ylabel('Constant');
    
elseif NUMBER_OF_FEATURES == 2
    figure()
    % Plot features with losses encoded in size
    losses = bsxfun(@loss_function,repmat(gold_standard_y,size(ys,1),1)',ys')';
    energies = features * w_star';
    scatter(features(:,1),features(:,2),myUtilities.scale(losses,20,300),...
        repmat(myUtilities.scale(energies,0,1),1,3).*ones(size(ys,1),3));
    % Plot y_stars in green
    hold on
    for y_star_num = 1 : length(y_stars)
        plot(features(y_stars(y_star_num),1),features(y_stars(y_star_num),2),'r*');
    end
    % Plot features of gold standard in red
    plot(gold_standard_features(1),gold_standard_features(2),'g*');
    
    % Plot direction of w_star
    plot([0;w_star(1)],[0,w_star(2)]);
    plot(0,0,'b*');
    
    title('loss size coded, energy color coded; green: gold st., red: picked');
    xlabel('Feature 1');
    ylabel('Feature 2');
    
elseif NUMBER_OF_FEATURES == 3
    figure()
    % Plot features with losses encoded in size
    losses = bsxfun(@loss_function,repmat(gold_standard_y,size(ys,1),1)',ys')';
    energies = features * w_star';
    scatter3(features(:,1),features(:,2),features(:,3),myUtilities.scale(losses,20,300),...
        repmat(myUtilities.scale(energies,0,1),1,3).*ones(size(ys,1),3));
    % Plot y_stars in green
    hold on
    for y_star_num = 1 : length(y_stars)
        plot3(features(y_stars(y_star_num),1),features(y_stars(y_star_num),2),features(y_stars(y_star_num),3),'r*');
    end
    % Plot features of gold standard in red
    plot3(gold_standard_features(1),gold_standard_features(2),gold_standard_features(3),'g*');
    
    % Plot direction of w_star
    plot3([0;w_star(1)],[0,w_star(2)],[0;w_star(3)]);
    plot3(0,0,0,'b*');
    
    title('loss size coded, energy color coded; green: gold st., red: picked');
    xlabel('Feature 1');
    ylabel('Feature 2');
    zlabel('Feature 3');
end

% Energy vs. loss function --> doesn't depend on dimensionality: can be
% plottet either way.
figure();
scatter(losses,energies);
xlabel('Losses');
ylabel('Energies');
set(gca,'XDir','reverse');

% Align figures
%figHandles = findobj('Type','figure'); % get all figure handles
%for i = 2 : length(figHandles)
%    iptwindowalign(figHandles(i-1),'right',figHandles(i),'left');
%end

%% Close all
close all