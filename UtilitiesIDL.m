classdef UtilitiesIDL
    properties
        precision=10^-3;
    end
    methods
        function self=UtilitiesIDL
        end
        function D=infty_norm_projection(self,D)
            [h,~]=size(D);
            for j=1:h
                D(j,:)=(1-self.precision)*proj_b1(1/(1-self.precision)*D(j,:));
            end
        end
    end
end