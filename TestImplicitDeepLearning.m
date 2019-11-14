classdef TestImplicitDeepLearning
    properties
        test_pass
    end

    methods
        function self=TestImplicitDeepLearning()
            test_1=self.test_infty_matrix_proj;
            test_2=self.test_picard_iterations;
            self.test_pass=test_1*test_2;
            if self.test_pass==1
                disp('all tests pass')
            else
                disp('tests failed')
            end
        end
        

    end
    
    methods(Static)
        function out=test_infty_matrix_proj
            out=0;
            U=rand(10,10^3);
            Y=rand(5,10^3);
            idl=ImplicitDeepLearning(U,Y,10);
            idl.precision=10^-3;
            D=rand(10,10);
            D_new=idl.infty_norm_projection(D);
            if norm(D_new,Inf)<=1-idl.precision+10^-9
                out=1;
            end
        end
        
        function out=test_picard_iterations
            U=rand(10,10^3);
            Y=rand(5,10^3);
            idl=ImplicitDeepLearning(U,Y,10);
            idl.well_posedness_projection(rand(10,10));
            idl=idl.initialization;
            X=idl.picard_iterations;
            if norm(X-max(0,idl.D*X+idl.E*idl.input+idl.f*ones(1,idl.m)),'fro')<idl.precision
                out=1;
            else
                out=0;
            end
        end
    end
end