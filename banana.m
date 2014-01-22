function [ b ] = banana( x )
%BANANA Test for minimization

b = 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
end

