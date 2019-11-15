classdef ImplicitDeepLearning
    properties
        U_train % input of training data
        Y_train % output
        X %     hidden features
        h %     # of hidden variables
        m%     # of datapoints
        n %     # of features for the input
        p %     # of outputs
           %     the following matrices and vectors correspond to the implicit
           %     prediction rule:
           %                            y=Ax+Bu+c; x=max(0,Dx+Eu+f)
        A 
        B
        c
        D
        E
        f
        % precision used in the solver
        precision=10^-5;
        % dual variable for fenchel
        lambda=0
        % well_posedness specification
        well_posedness='infty'
        fval_reg
        fval_fenchel_divergence
        funcs
    end
    
    methods
        function s=ImplicitDeepLearning(U,Y,h)
            s.U_train=U;
            s.Y_train=Y;
            s.h=h;
            [s.n,s.m]=size(U);
            [s.p,~]=size(Y);
            s.funcs=UtilitiesIDL;
            %TODO: include checks of inputs   
        end
        
        function s=train(s)
            s=s.initialization;
            s.X=s.picard_iterations;
            s.fval_reg=NaN*zeros(100,1);
            % initial implicit problem (lambda=0) start with (A,B,c,X)...
            for k=1:100
                [grad_A,grad_B,grad_c]=s.gradient_parameters_reg;
                grad_X=s.gradient_hidden_var;
                step_theta_reg=s.step_size_parameters_reg;
                step_X=s.step_size_X;
                s.A=s.A-step_theta_reg*grad_A;
                s.B=s.B-step_theta_reg*grad_B;
                s.c=s.c-step_theta_reg*grad_c;
                s.X=max(0,s.X-step_X*grad_X);
                s.fval_reg(k)=s.objective_reg;
            end
            
            % ... then continue with (D,E,f)
            s.fval_fenchel_divergence=NaN*zeros(200,1);
            for k=1:200
                [grad_D,grad_E,grad_f]=s.gradient_parameters_hid;
                step_theta_hid=s.step_size_parameters_hid;
                s.D=s.well_posedness_projection(s.D-step_theta_hid*grad_D);
                s.E=s.E-step_theta_hid*grad_E;
                s.f=s.f-step_theta_hid*grad_f;   
                s.fval_fenchel_divergence(k)=s.funcs.scalar_fenchel_divergence(s.X,s.D*s.X+s.E*s.U_train+s.f*ones(1,s.m));
            end
        end
        
        function X=picard_iterations(s,X)
            if nargin==1
                X=rand(s.h,s.m);
            end
            k=1;
            while k<10^3 && norm(X-max(0,s.D*X+s.E*s.U_train+s.f*ones(1,s.m)),'fro')>s.precision
                X=max(0,s.D*X+s.E*s.U_train+s.f*ones(1,s.m));
                k=k+1;
            end
            if k>=10^3
                disp('picard iterations were not able to find a solution X to the implicit model')
            end
        end
        
        % gradients
        function grad_X=gradient_hidden_var(s)
            if norm(s.lambda)>0
                grad_X=(1/s.m)*( s.A'*(s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m))+ ...
                    (diag(s.lambda)-diag(s.lambda)*s.D-s.D'*diag(s.lambda))*s.X+ ...
                    s.D'*diag(s.lambda)*max(0,s.D*s.X+s.E*s.U_train+s.f*ones(1,s.m))- ...
                    diag(s.lambda)*(s.E*s.U_train+s.f*ones(1,s.m)) );
            else
                 grad_X=(1/s.m)*( s.A'*(s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m)));
            end
        end
        
        function [grad_A,grad_B,grad_c]=gradient_parameters_reg(s)
            Cst=(1/s.m)*(s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m)-s.Y_train);
            grad_A=Cst*s.X';
            grad_B=Cst*s.U_train';
            grad_c=Cst*ones(s.m,1);
        end
        
        function [grad_D,grad_E,grad_f]=gradient_parameters_hid(s)
            if norm(s.lambda)>0
                Cst=diag(s.lambda)*(1/s.m)*(max(0,s.D*s.X+s.E*s.U_train+s.f*ones(1,s.m))-s.X);
                grad_D=Cst*s.X';
                grad_E=Cst*s.U_train';
                grad_f=Cst*ones(s.m,1);
            else
                Cst=(1/s.m)*(max(0,s.D*s.X+s.E*s.U_train+s.f*ones(1,s.m))-s.X);
                grad_D=Cst*s.X';
                grad_E=Cst*s.U_train';
                grad_f=Cst*ones(s.m,1);
            end
        end
        
        % step sizes
     
        function  out=step_size_parameters_reg(s)
            out=s.m/max([s.m,norm(s.X)^2,norm(s.U_train)^2,norm(s.X*s.U_train')]);
        end
        
        function  out=step_size_parameters_hid(s)
            if norm(s.lambda)>0
                out=s.m/(max(s.lambda)*max([s.m,norm(s.X)^2,norm(s.U_train)^2,norm(s.U_train)*norm(s.X)]));
            else
                out=s.m/(max([s.m,norm(s.X)^2,norm(s.U_train)^2,norm(s.U_train)*norm(s.X)]));
            end
        end
        
        function  out=step_size_X(s)
            if norm(s.lambda)>0
                out=s.m/(norm(s.A'*s.A+diag(s.lambda)-diag(s.lambda)*s.D+s.D'*diag(s.lambda))+max(s.lambda)*norm(s.D)^2);
            else
                out=s.m/(norm(s.A'*s.A));
            end
        end
        
        function fval=objective_reg(s)
            fval=(1/sqrt(s.m))*norm(s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m)-s.Y_train,'fro');
        end
        
        function [A,D]=lmi_projection(s,A,D,lambda)
            if norm(lambda)>0
                A_init=A;
                D_init=D;
                Lam=diag(lambda);
                h=s.h;
                p=s.p;
                prec=s.precision;
                cvx_begin sdp quiet
                variables A(p,h)
                variables D(h,h)
                minimize(norm([A;D]-[A_init;D_init]))
                Lam*D+D'*Lam<=Lam+A'*A
                norm(D)<=1-prec
                cvx_end
            end
        end
        
        function D=infty_norm_projection(s,D)
            for j=1:s.h
                D(j,:)=(1-s.precision)*proj_b1(1/(1-s.precision)*D(j,:));
            end
        end
        
        function D=well_posedness_projection(s,D)
            if strcmp(s.well_posedness,'infty')
                D=s.infty_norm_projection(D);
            elseif strcmp(s.well_posedness,'LMI')
                D=s.lmi_projection(s.A,D,s.lambda);
            end
        end
        
        function s=initialization(s)
            s.A=rand(s.p,s.h)-0.5;
            s.B=rand(s.p,s.n)-0.5;
            s.c=rand(s.p,1)-0.5;
            s.D=s.well_posedness_projection(rand(s.h,s.h)-0.5);
            s.E=rand(s.h,s.n)-0.5;
            s.f=rand(s.h,1)-0.5;
        end
  
    end
end