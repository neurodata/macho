#include "image.h"
#include <stdlib.h>
#include <math.h>
#include <stack>
#include <queue>
#include <sstream>
//#include <direct.h>
#include <omp.h>

//Need to adjust this to fix MATLAB needs
//int intensity ( const Image32& im , int& i , int& b , int& l ) ;

////////////////////////////
// Image processing stuff //
////////////////////////////

float clamp ( float value ) ;
float min ( float a , float b ) ;
float max ( float a , float b ) ;

float clamp(float value) {
	if (value < 0)
		return 0;
	else if (value > 255)
		return 255;
	else
		return value;
}

float min(float a, float b) {
	if (a < b)
		return a;
	else
		return b;
}

float max(float a, float b) {
	if (a > b)
		return a;
	else
		return b;
}

int MatlabImage::AxoplasmicReticula(MatlabImage image , MatlabImage output) const {
	
	// look for pixels that are dark enough
	// look at neighbours to see if they are dark enough
		// if dark - push to stack until surrounded by light pixels
		// if neighbourhood radius crosses threshold, discard all pixels collected from that neighbourhood.
	// color all pixels in stack
	
	/*int blk = 0 , lght = 0 , intnsty = 0 ;
	intensity( outputImage , intnsty , blk , lght );
	printf( "Total intensity = %d, Black pixel = %d, Light pixel = %d\n" , intnsty , blk , lght+2 );*/
	
	int blk = 45 ; int lght = 85 ;

	int anno = 1;							
	string id, annoid, dir;
	int previous_i = 0, previous_j = 0;
	
	int r = image.rows; int c = image.cols ; int s = image.slices ;
	for (int ns=0;ns<s;ns++){ 
		for (int j=0;j<r;j++){
			for(int i=0;i<c;i++){
				if (output.get(i,j,ns) == 0) {
					double bl = (double)blk ;
					int black = (int)bl; 
					if (image.get(i,j,ns) < black) {
						bool light = false;
						stack<int> cluster_i;
						stack<int> cluster_j;
						cluster_i.push(i);
						cluster_j.push(j);
						int diametre = 3, radius = 1, step = 2;
						while (light == false && radius < 17) {												 
							int light_p = 0;
							int count = 0;
							int row = 0;
							bl += 0.5 ;
							black = (int)bl ;
							for (int rj=j+1-radius;rj<=j+1+radius;rj++){
								if (row%diametre == 0){
									for (int ri=i+1-radius;ri<=i+1+radius;ri++){
										if (rj < 0 || ri < 0 || rj >= r || ri >= c)
											continue;
										if (image.get(ri,rj,ns) < black) {
											cluster_i.push(ri);
											cluster_j.push(rj);
										}
										else if (image.get(ri,rj,ns) > lght+2) 
											light_p++;
										count++;
									}
								}
								else {
									for (int ri=i+1-radius;ri<=i+1+radius;ri+=step){
										if (rj < 0 || ri < 0 || rj >= r || ri >= c)
											continue;
										if (image.get(ri,rj,ns) < black) {	
											cluster_i.push(ri);
											cluster_j.push(rj);
										}
										else if (image.get(ri,rj,ns) > lght+2) 
											light_p++;
										count++;
									}
								}
								row++;
							}
							if ( light_p >= count-5 && light_p <= count ) 
									light = true;
							diametre++;
							radius++;
							step += 2;
						}
						if (radius < 3 || radius > 16) { 
							cluster_i = stack<int>();
							cluster_j = stack<int>();
						}
						else if (cluster_i.top() == previous_i && cluster_j.top() == previous_j) {
							cluster_i = stack<int>();
							cluster_j = stack<int>();
						}
						else {					
							stringstream s;
							s << anno;
							id = s.str();

							previous_i = cluster_i.top();
							previous_j = cluster_j.top();
							while(cluster_i.size() != 0) {
								int new_i = cluster_i.top();
								int new_j = cluster_j.top();
								output.set(new_i,new_j,ns,255);
								cluster_i.pop();
								cluster_j.pop();
							}
							anno++;
							s.clear();
						}
					}
				}
			}
		}
		if ((ns+1)%10==0)
			mexPrintf("%d%% complete...\n",(ns+1)*100/s) ;
	}
	
	return 1;
}
