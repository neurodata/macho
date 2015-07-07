#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "Image/image.h"
#include "Util/cmdLineParser.h"

void ShowUsage(char* ex){
	printf("Usage %s:\n",ex);
	printf("\t--in <input image> --out <output image>\n");
	printf("\t[--axoplasmicreticula]\n"); 
}
void mexFunction( int nlhs , mxArray *plhs[] , int argc , const mxArray *prhs[] ) {
	
	if (argc == 0)
	{
		ShowUsage("FindReticula") ;
		return ;
	}
/*      
	char** argv = new char*[argc] ;
	for ( int i = 0 ; i < argc ; i++ )
	{
		char* arg = (char *)mxArrayToString(prhs[i]) ;
		if (arg && arg[0] == '\0')
			argv[i] = '\0';
		else
			argv[i] = strdup(arg) ;
		mxFree(arg);
	}
        	
	cmdLineString In , Out ;

	cmdLineReadable  APR  ;
	
	char* paramNames[]={
		"in","out",
		"axoplasmicreticula"
	};

	cmdLineReadable* params[]=
	{
		&In,&Out,
		&APR
	};

        //cmdLineParse(argc-1,&argv[1],paramNames,sizeof(paramNames)/sizeof(char*),params); // <-- CPP
	cmdLineParse(argc,&argv[0],paramNames,sizeof(paramNames)/sizeof(char*),params); // <-- MATLAB
        
	// Check that the input and output files have been set
	if(!In.set || !Out.set){
		if(!In.set)	{printf("Input image was not set\n");}
		else		{printf("Output image was not set\n");}
		ShowUsage("FindReticula") ;
		return ; //EXIT_FAILURE;
	}
        
	if(APR.set){*/

		MatlabImage image(prhs[1]) ;
		MatlabImage out(prhs[3]) ;

		if (!image.AxoplasmicReticula(image , out)) {
			printf("Could not compute axoplasmic reticula in the image\n");
		}
/*		return ;
	}
        */

	return ; //EXIT_SUCCESS;
}
