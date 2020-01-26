function [center] = circumcenter(ptA, ptB, ptC)

        

        a = ptA-ptC;
		b = ptB-ptC;
        
		denominator = (2*((norm(a)*norm(b))^2-dot(a,b)^2));
		inner = (norm(a)^2)*b-(norm(b)^2)*a;
		numerator = (dot(inner,b))*a - dot(inner,a)*b;
		center = (1/denominator)*numerator + ptC;
        
end
