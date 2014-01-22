function [ ret ] = l( gold_standard_y, ys, gold_standard_feature, features, w )
%L Function of the same name in Kroeger paper.

ret = -inf;

for i = 1 : size(ys,1)
    ret = max(loss_function(gold_standard_y,ys(i,:)) + gold_standard_feature*w' - features(i,:)*w',ret);
end

end