function [loss] = loss_function( gold_standard_y, some_other_y )

% Count number of deviations
loss = sum(abs(gold_standard_y - some_other_y));

% Uniform loss, with requirement loss(gold_standard_y,gold_standard_y) == 0
% fullfilled
% loss = 0;

% Loss 1 for every y except the gold standard.
% if all(gold_standard_y == some_other_y)
%     loss = 0;
% else
%     loss = 1;
% end

end

