classdef UtilitiesIDL
    properties
        wp_precision=10^-3;
        precision=10^-6;
    end
    methods
        % constructor
        function s=UtilitiesIDL
        end
        
        function D=infty_norm_projection(s,D)
            [h,~]=size(D);
            for j=1:h
                D(j,:)=(1-s.wp_precision)*proj_b1(1/(1-s.wp_precision)*D(j,:));
            end
        end
        
        function [A,D]=lmi_projection(s,A,D,lambda)
            if norm(lambda)>0
                A_init=A;
                D_init=D;
                Lam=diag(lambda);
                [h,~]=size(D);
                [p,~]=size(A);
                prec=s.wp_precision;
                cvx_begin sdp quiet
                variables A(p,h)
                variables D(h,h)
                minimize(norm([A;D]-[A_init;D_init]))
                Lam*D+D'*Lam<=Lam+A'*A
                norm(D)<=1-prec
                cvx_end
            end
        end
        
        function X=picard_iterations(s,U,D,E,f)
            [h,~]=size(D);
            [~,m]=size(U);
            X=rand(h,m)-0.5;
            k=1;
            while k<10^3 && norm(X-max(0,D*X+E*U+f*ones(1,m)),'fro')>s.precision
                X=max(0,D*X+E*U+f*ones(1,m));
                k=k+1;
            end
            if k>=10^3
                disp('picard iterations were not able to find a solution X to the implicit model within 1,000 iterations')
            end
        end
        
        function M=RandOrthMat(s,n)
            % (c) Ofek Shilon , 2006.
            tol=s.precision;
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
        
        % compute the MSE given model parameters
        function out=RMSE_actual_implicit(s,A,B,c,D,E,f,U,Y)
            X=s.picard_iterations(U,D,E,f);
            out=sqrt(s.MSE_implicit_objective(X,A,B,c,U,Y));
        end
        
    end
    
    methods(Static)
        % just compute RMSE
        function out=RMSE(Y_1, Y_2)
            [~,m]=size(Y_1);
            out=(1/sqrt(m))*norm(Y_1-Y_2,'fro');
        end
        
        % compute the MSE, given X (not considering satisfaction of the implicit constraint)
        function fval=MSE_implicit_objective(X, A, B, c, U, Y)
            [~,m]=size(U);
            fval=(1/m)*norm(A*X+B*U+c*ones(1,m)-Y,'fro')^2;
        end
        
        function fval=scalar_fenchel_divergence(U, V)
            [~,m]=size(U);
            fval=(1/m)*(0.5*norm(U,'fro')^2+0.5*norm(max(0,V),'fro')^2-trace(U'*V));
        end
        
        function fval = fenchel_divergence(U, V)
            [~,m]=size(U);
            fval=(1/m)*(0.5*sum(U,2).^2+0.5*sum(max(0,V),2).^2-sum(U.*V,2));
        end
        
        function fval=implicit_objective(X, A, B, c, D, E, f, U, Y, lambda)
            fval=MSE_implicit_objective(X, A, B, c, U, Y)+lambda'* fenchel_divergence(X, D*X+E*U+f*ones(1,m));
        end
        
    end
end