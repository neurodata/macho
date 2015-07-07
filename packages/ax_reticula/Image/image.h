#ifndef IMAGE_INCLUDED
#define IMAGE_INCLUDED

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <matrix.h>
#include "mex.h"
using namespace std;

/** The value of pi to many significant digits/*/
#define PI 3.1415926535897932384

class MatlabImage {
public:
    MatlabImage(const mxArray *input) {
        this->data = (double*)mxGetData(input);
        this->rows = mxGetDimensions(input)[0];
        this->cols = mxGetDimensions(input)[1];
        this->slices = mxGetDimensions(input)[2];
    }
    double *data;
    int rows, cols, slices;
    int get(int row, int col, int slice) {
        return this->data[row + rows*(col + cols*slice)];
    }
	void set(int row, int col, int slice, int val) {
        this->data[row + rows*(col + cols*slice)] = val;
    }
	
	int AxoplasmicReticula( MatlabImage , MatlabImage ) const;
};

#endif // IMAGE_INCLUDED

