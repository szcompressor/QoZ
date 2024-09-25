

#ifndef SZ_TRANSFORM_HPP
#define SZ_TRANSFORM_HPP
#include<cmath>
namespace QoZ {
    template<class T>
    inline T sigmoid(T x){
        if (x>=0){
            return ((double)(1.0))/((double)(1.0)+exp(-x));

        }
        else{
            double t=exp(x);
            return t/(t+(double)(1.0));
        }
    }
    template<class T>
    inline T logit(T x){
        if(x>=1.0-1e-20)
            return log(1e20);
        else if (x<=1e-20)
            return log(1e-20);
        return log( x /( (double)(1.0)-x ) );
    }
   
    template<class T>
    inline T tanh(T x){

        if(x>=0){
            double t=exp(-2*x);
            return ((double)(1.0)-t)/((double)(1.0)+t);
        }
        else{
            double t=exp(2*x);
            return (t-(double)(1.0))/(t+(double)(1.0));
        }

    }
    template<class T>
    inline T arctanh(T x){
        if(x>=1.0-1e-20)
           return 0.5*log(2e20);
        else if (x<=-1.0+1e-20)
            return 0.5*log(0.5e-20);

        return 0.5* ( log ( ( (double)(1.0)+x )/( (double)(1.0)-x ) ) );
    }

}
#endif //SZ_INTERPOLATORS_HPP
