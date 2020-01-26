function [lambda] = piT(x, xp, xpp)

lambda=2*(xpp-xp)+2*(dot(xp-x,xpp-xp)/(norm(xpp-xp)^2))*(xpp-xp)+x;

end