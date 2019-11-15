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
        
        function X=picard_iterations(self,U,D,E,f)
            [h,~]=size(D);
            [~,m]=size(U);
            X=rand(h,m)-0.5;
            k=1;
            while k<10^3 && norm(X-max(0,D*X+E*U+f*ones(1,m)),'fro')>self.precision
                X=max(0,D*X+E*U+f*ones(1,m));
                k=k+1;
            end
            if k>=10^3
                disp('picard iterations were not able to find a solution X to the implicit model within 1,000 iterations')
            end
        end
        
        function M=RandOrthMat(self,n)
            % (c) Ofek Shilon , 2006.
            tol=self.precision;
            if nargin==1
                tol=1e-6;
            end
            M = zeros(n); 
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
    
    methods(Static)
        function out=RMSE(Y_1,Y_2)
            [~,m]=size(Y_1);
            out=(1/sqrt(m))*norm(Y_1-Y_2,'fro');
        end
        
        function fval=scalar_fenchel_divergence(U,V)
            [~,m]=size(U);
            fval=(1/m)*(0.5*norm(U,'fro')^2+0.5*norm(max(0,V),'fro')^2-trace(U'*V));
        end
    end
end