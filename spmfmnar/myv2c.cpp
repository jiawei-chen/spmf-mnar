#include "mex.h"
#include<vector>
using namespace std;
    vector<int>q[1000005];
    vector<double>w[1000005];
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
     int  nn=mxGetM(prhs[0]); //获得行数
     int n=*(mxGetPr(prhs[2]));
     double *p=mxGetPr(prhs[0]);
    double *pa=mxGetPr(prhs[1]);
     for(int i=1;i<=n;i++)
     {
        q[i].clear();
        w[i].clear();
     }
     for(int i=1;i<=nn;i++)
     {
        int x=*(p+i-1);
        double y=*(pa+i-1);
        q[x].push_back(i);
        w[x].push_back(y);
     }

     plhs[0]=mxCreateCellMatrix(n,1);
     plhs[1]=mxCreateCellMatrix(n,1);
     for(int i=1;i<=n;i++)
     {

        mwIndex subs[2];
        int nsubs=2;
        subs[0]=i-1;
        subs[1]=0;
      int  index=mxCalcSingleSubscript(plhs[0],nsubs,subs);

         mxArray* now=mxCreateDoubleMatrix(1,q[i].size(),mxREAL);
          mxArray* nowa=mxCreateDoubleMatrix(1,q[i].size(),mxREAL);
         double* buf=mxGetPr(now);
          double* bufa=mxGetPr(nowa);
         for(int j=0;j<q[i].size();j++)
         {
            *(buf+j)=q[i][j];
            *(bufa+j)=w[i][j];
         //   mexPrintf("%d\n",q[i][j]);
         }
         mxSetCell(plhs[0],index,now);
         mxSetCell(plhs[1],index,nowa);
     }
}








