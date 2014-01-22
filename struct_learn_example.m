% THINGS TO TRY:
% When the loss function is uniformly 0 it seems like the algorithm should
% pick w_star such as to minimize the maximum energy difference between any
% y and the gold_standard_y.

% Try larger LAMBDAs to see what happens when the data term is favoured
% more.

% Define parameters

NUMBER_OF_FEATURES = 2;
LENGTH_OF_Y = 3;
LAMBDA = 25;

% Generate ys, pick gold standard y

ys = de2bi(0:2^LENGTH_OF_Y-1);
gold_standard_y_num = ceil(size(ys,1)*rand(1));
gold_standard_y = ys(gold_standard_y_num,:);

% 'Calculate' features (i.e. make them up for the purpose of the example)

features = rand(size(ys,1),NUMBER_OF_FEATURES);
gold_standard_feature = features(gold_standard_y_num,:);

%% Training

min_search_function_w_handle = @(w)min_search_function(LAMBDA, gold_standard_y, ys, gold_standard_feature, features, w);

w_star = fminsearch(min_search_function_w_handle,zeros(1,NUMBER_OF_FEATURES));
%w_star = [-1,-1];

% Get an idea of what the l function looks like:

l_w_handle = @(w)l(gold_standard_y, ys, gold_standard_feature, features, w);
[w_probes_1,w_probes_2] = meshgrid(-1.5:0.1:1.5);
l_res = zeros(size(w_probes_1));
for i = 1 : size(w_probes_1,1)
    for j = 1 : size(w_probes_1,2)
        l_res(i,j) = l_w_handle([w_probes_1(i,j),w_probes_2(i,j)]);
    end
end
figure(1);
mesh(l_res);

% Get an idea of what the loss function looks like:
min_search_function_res = zeros(size(w_probes_1));
for i = 1 : size(w_probes_1,1)
    for j = 1 : size(w_probes_1,2)
        min_search_function_res(i,j) = min_search_function_w_handle([w_probes_1(i,j),w_probes_2(i,j)]);
    end
end
figure(2);
mesh(min_search_function_res);


% Do we allways get the same w_star? --> yes, more or less.
same = true;
for i = 1 : 10
    w_star_dash = fminsearch(min_search_function_w_handle,rand(1,NUMBER_OF_FEATURES));
    same = same && all(w_star_dash == w_star);
end

% Calculate y_star
y_stars = find(features*w_star' == min(features*w_star'));

% Visualization

if NUMBER_OF_FEATURES == 2
    figure(3)
    % Plot features with losses encoded in size
    losses = bsxfun(@loss_function,repmat(gold_standard_y,size(ys,1),1)',ys')';
    energies = features * w_star';
    scatter(features(:,1),features(:,2),myUtilities.scale(losses,20,300),repmat(myUtilities.scale(energies,0,1),1,3).*ones(size(ys,1),3));
    % Plot y_stars in green
    hold on
    for y_star_num = 1 : length(y_stars)
        plot(features(y_stars(y_star_num),1),features(y_stars(y_star_num),2),'g*');
    end
    % Plot features of gold standard in red
    plot(features(gold_standard_y_num,1),features(gold_standard_y_num,2),'r*');
    
    % Plot direction of w_star
    plot([0;w_star(1)],[0,w_star(2)]);
    plot(0,0,'b*');
    
    title('loss size coded, energy color coded; red: gold st., green: picked');
    
    % Energy vs. loss function
    figure(4);
    scatter(losses,energies);
    xlabel('Losses');
    ylabel('Energies');
    set(gca,'XDir','reverse');
end