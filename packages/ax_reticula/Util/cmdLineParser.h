#ifndef CMD_LINE_PARSER_INCLUDED
#define CMD_LINE_PARSER_INCLUDED
#include <stdarg.h>
#include <string.h>

#include "mex.h"

//#ifdef WIN32
int strcasecmp(char* c1,char* c2);
//#endif

/** This base class represents the type of information that can be read from the command line.*/
class cmdLineReadable{
public:
	/** This value is set to 1 if the parameter was successfully read from the command line.*/
	int set;
	cmdLineReadable(void);
	/** This method attempts to read in the parameter from the command line. */
	virtual int read(char** argv,int argc);
	int setval(int argc);
};

/** This class represents an integer that is to be read from the command line.*/
class cmdLineInt : public cmdLineReadable {
public:
	/** The value read in.*/
	int value;
	cmdLineInt();
	int read(char** argv,int argc);
};
/** This class represents an array of integer that is to be read from the command line.*/
class cmdLineIntArray : public cmdLineReadable {
public:
	/** The number of integers read in */
	int count;
	/** The values of the integers */
	int* value;
	cmdLineIntArray(int);
	~cmdLineIntArray();
	int read(char** argv,int argc);
};
/** This class represents a floating point number that is to be read from the command line.*/
class cmdLineFloat : public cmdLineReadable {
public:
	/** The value of the floating point */
	float value;
	cmdLineFloat();
	int read(char** argv,int argc);
};
/** This class represents an array of floating point number that is to be read from the command line.*/
class cmdLineFloatArray : public cmdLineReadable {
public:
	/** The number of floating point numbers read in */
	int count;
	/** The values of the floating point numbers */
	float* value;
	cmdLineFloatArray(int);
	~cmdLineFloatArray();
	int read(char** argv,int argc);
};

/** This class represents a string (word) that is to be read from the command line.*/
class cmdLineString : public cmdLineReadable {
public:
	/** The value of the string */
	char* value;
	cmdLineString();
	~cmdLineString();
	int read(char** argv,int argc);
};

/** This class represents an array of strings (words) that is to be read from the command line.*/
class cmdLineStringArray : public cmdLineReadable {
public:
	/** The number of strings read in */
	int count;
	/** The values of the strings */
	char** value;
	cmdLineStringArray(int);
	~cmdLineStringArray();
	int read(char** argv,int argc);
};

/** This function parses the command line arguments, using the array of names to look up the parameter names
  * and sets the values of the cmdLineReadable parameters appropriately. (Note that paramers must start with
  * a "--", so if I would like to set the value of a parameter with name "foo", it has to look like "--foo ..." in
  * the command line.)
  */
void cmdLineParse(int argc,  char **argv,char** names,int num,cmdLineReadable** r);

/** This file returns the file extension of the specified file name (i.e. the part of the file name proceeding the final "." */
char* GetFileExtension(char* fileName);

#endif // CMD_LINE_PARSER_INCLUDED
