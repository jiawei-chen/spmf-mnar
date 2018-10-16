#include "mex.h"
#include<vector>
using namespace std;
double ans[1000005];
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
     int  nn=mxGetM(prhs[0]); //获得行数
     int n=*(mxGetPr(prhs[2]));
     double *p=mxGetPr(prhs[0]);
    double *pa=mxGetPr(prhs[1]);
        for(int i=1;i<=n;i++)
            ans[i]=0;
     for(int i=1;i<=nn;i++)
     {
        int x=*(p+i-1);
        double y=*(pa+i-1);
        ans[x]+=y;

     }

     plhs[0]=mxCreateDoubleMatrix(n,1,mxREAL);
      double *a = mxGetPr(plhs[0]);
     for(int i=1;i<=n;i++)
     {
        a[i-1]=ans[i];
     }
}








