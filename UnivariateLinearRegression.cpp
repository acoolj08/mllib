#include <iostream>

namespace jagetiya 
{
    template <class T>
    struct UnivariateLinearRegression 
    {
    private:
        static const int n_ = 1;  // number of features
        int m_;                   // number of training sets
        double theta0_;           // theta parameter 0
        double theta1_;           // theta parameter 1
        double alpha_;            // learning rate alpha    
        
        double cost_function(const T x[], const T y[]) const
        {
            double J = 0;
            for(int i = 0 ; i < m_ ; i++) 
            {
                J += (theta0_ + theta1_*x[i] - y[i]) *
                (theta0_ + theta1_*x[i] - y[i]);
            }
            J = J / (2 * m_);
            return J;
        }
        
        //repeat until convergence
        void gradient_descent(const T x[], const T y[])
        {
            double res1 = 0;
            double res2 = 0;
            for(int i = 0 ; i < m_ ; i++) 
            {
                res1 += (theta0_ + theta1_*x[i] - y[i]) / m_;
                res2 += ((theta0_ + theta1_*x[i] - y[i]) * x[i]) / m_;
            }
            double temp0 = theta0_ - alpha_ * res1;
            double temp1 = theta1_ - alpha_ * res2;
            theta0_ = temp0;
            theta1_ = temp1;    
        }
        
    public:
        UnivariateLinearRegression() = default;
        UnivariateLinearRegression(int m = 10, 
            double theta0 = 0, double theta1 = 0, double alpha = 0.5)
            : m_(m), theta0_(theta0), theta1_(theta1), alpha_(alpha)
        {}

        //train the system
        double train(const T x[], const T y[])
        {
            double prev_result = cost_function(x, y);
            gradient_descent(x, y);
            double current_result = cost_function(x, y);
            double tmp = 0;
            while( !(current_result - prev_result < 0.01
                &&  current_result - prev_result > -0.01))
            {
                gradient_descent(x, y);
                tmp = current_result;
                current_result = cost_function(x, y);
                prev_result = tmp;
                std::cout << current_result <<std::endl;
            }
            return current_result;
        }

        double calculate_value(int x) { return theta0_ + theta1_ * x;}
    };
}

int main()
{
    int v = 1000000;
    int *x = (int *) malloc(sizeof(int) * v);
    int *y = (int *) malloc(sizeof(int) * v);
    for(int i = 0 ; i < v ; i++) 
    {
        x[i] = i+1;
        y[i] = rand() % 100;
    }
    

    jagetiya::UnivariateLinearRegression<int> t(v, 0, 0, 0.0000000000005);
    t.train(x, y);
    std::cout << x[0] << " " <<y[0] << " " <<t.calculate_value(x[0]) <<std::endl;
    std::cout << x[1] << " " <<y[1] << " " <<t.calculate_value(x[1]) <<std::endl;
    std::cout << x[2] << " " <<y[2] << " " <<t.calculate_value(x[2]) <<std::endl;
    std::cout << x[3] << " " <<y[3] << " " <<t.calculate_value(x[3]) <<std::endl;
    std::cout << x[4] << " " <<y[4] << " " <<t.calculate_value(x[4]) <<std::endl;
    std::cout << x[5] << " " <<y[5] << " " <<t.calculate_value(x[5]) <<std::endl;
}
