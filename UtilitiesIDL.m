classdef UtilitiesIDL
    properties
        wp_precision=10^-3;
        precision=10^-6;
    end
    methods
        function self=UtilitiesIDL
        end
        function D=infty_norm_projection(self,D)
            [h,~]=size(D);
            for j=1:h
                D(j,:)=(1-self.wp_precision)*proj_b1(1/(1-self.wp_precision)*D(j,:));
            end
        end
        
        function M=RandOrthMat(self,n)
            % M = RANDORTHMAT(n)
            % generates a random n x n orthogonal real matrix.
            %
            % M = RANDORTHMAT(n,tol)
            % explicitly specifies a thresh value that measures linear dependence
            % of a newly formed column with the existing columns. Defaults to 1e-6.
            %
            % In this version the generated matrix distribution *is* uniform over the manifold
            % O(n) w.r.t. the induced R^(n^2) Lebesgue measure, at a slight computational
            % overhead (randn + normalization, as opposed to rand ).
            %
            % (c) Ofek Shilon , 2006.
            tol=self.precision;
            if nargin==1
                tol=1e-6;
            end
            
            M = zeros(n); % prealloc
            
            % gram-schmidt on random column vectors
            
            vi = randn(n,1);
            % the n-dimensional normal distribution has spherical symmetry, which implies
            % that after normalization the drawn vectors would be uniformly distributed on the
            % n-dimensional unit sphere.
            M(:,1) = vi ./ norm(vi);
            
            for i=2:n
                nrm = 0;
                while nrm<tol
                    vi = randn(n,1);
                    vi = vi -  M(:,1:i-1)  * ( M(:,1:i-1).' * vi )  ;
                    nrm = norm(vi);
                end
                M(:,i) = vi ./ nrm;
            end
        end
        
    end
end