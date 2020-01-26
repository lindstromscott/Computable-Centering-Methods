function [answer] = colinear_check(ptA, ptB, ptC, thresh)

        

        a = ptA-ptC;
		b = ptB-ptC;
        
        if abs(dot(a,b)/(norm(a)*norm(b))) < 1-thresh %check to make sure not colinear
            answer = 1;
        else
            answer = 0;
        end
        
end
