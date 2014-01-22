function [ ret ] = min_search_function( LAMBDA, gold_standard_y, ys, gold_standard_feature, features, w )
%MIN_SEARCH_FUNCTION The function to minimize

ret = 0.5*norm(w)^2 + LAMBDA/size(ys,1)*l(gold_standard_y, ys, gold_standard_feature, features,w);

end

