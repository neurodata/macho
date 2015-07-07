#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "cmdLineParser.h"

//#ifdef WIN32
int strcasecmp(char* c1,char* c2){return strcmp(c1,c2);}
//#endif

cmdLineReadable::cmdLineReadable(void){set=0;}
int cmdLineReadable::read(char**,int){
	set=1;
	return 0;
}

int cmdLineReadable::setval(int){
	set = 1;
	return 1;
}

cmdLineInt::cmdLineInt(void){value=0;}
int cmdLineInt::read(char** argv,int argc){
	if(argc>0){
		value=atoi(argv[0]);
		set=1;
		return 1;
	}
	else{return 0;}
}

cmdLineIntArray::cmdLineIntArray(int cnt){
	value=new int[cnt];
	assert(value);
	count=cnt;
	for(int i=0;i<count;i++){value[i]=0;}
}
cmdLineIntArray::~cmdLineIntArray(void){
	if(value){
		delete[] value;
		value=NULL;
	}
}
int cmdLineIntArray::read(char** argv,int argc){
	if(argc>=count){
		set=1;
		for(int i=0;i<count;i++){
			value[i]=atoi(argv[i]);
		}
		return count;
	}
	else{return 0;}
}

cmdLineFloat::cmdLineFloat(void){value=0;}
int cmdLineFloat::read(char** argv,int argc){
	if(argc>0){
		value=(float)atof(argv[0]);
		set=1;
		return 1;
	}
	else{return 0;}
}

cmdLineFloatArray::cmdLineFloatArray(int cnt){
	value=new float[cnt];
	assert(value);
	count=cnt;
	for(int i=0;i<cnt;i++){value[i]=0;}
}
cmdLineFloatArray::~cmdLineFloatArray(void){
	if(value){
		delete[] value;
		value=NULL;
	}
}
int cmdLineFloatArray::read(char** argv,int argc){
	if(argc>=count){
		set=1;
		for(int i=0;i<count;i++){
			value[i]=(float)atof(argv[i]);
		}
		return count;
	}
	else{return 0;}
}

cmdLineString::cmdLineString(void){value=NULL;}
cmdLineString::~cmdLineString(void){
	if(value){
		delete[] value;
		value=NULL;
	}
}
int cmdLineString::read(char** argv,int argc){
	if(argc>0){
		value=new char[strlen(argv[0])+1];
		strcpy(value,argv[0]);
		set=1;
		return 1;
	}
	else{return 0;}
}

cmdLineStringArray::cmdLineStringArray(int cnt){
	value=new char*[cnt];
	assert(value);
	count=cnt;
	for(int i=0;i<count;i++){value[i]=NULL;}
}
cmdLineStringArray::~cmdLineStringArray(void){
	if(value){
		for(int i=0;i<count;i++){if(value[i]){delete[] value[i];}}
		delete[] value;
		value=NULL;
	}
}
int cmdLineStringArray::read(char** argv,int argc){
	if(argc>=count){
		set=1;
		for(int i=0;i<count;i++){
			value[i]=new char[strlen(argv[i])+1];
			strcpy(value[i],argv[i]);
		}
		return count;
	}
	else{return 0;}
}

void cmdLineParse(int argc, char **argv,char** names,int num,cmdLineReadable** readable){
	int i,j;

	while (argc > 0) {
		if (argv[0][0] == '-' && argv[0][1]=='-') {
			for(i=0;i<num;i++){
				if (!strcmp(&argv[0][2],names[i])){
					argv++, argc--;
					if (argv[0] != NULL) 
						j=readable[i]->read(argv,argc);
					else {
						j=readable[i]->setval(1) ;
					}
					argv+=j,argc-=j;
					break;
				}
			}
			if(i==num){argv++, argc--;}
		}
		else {argv++, argc--;}
	}
}

char* GetFileExtension(char* fileName){
	char* fileNameCopy;
	char* ext=NULL;
	char* temp;

	fileNameCopy=new char[strlen(fileName)+1];
	assert(fileNameCopy);
	strcpy(fileNameCopy,fileName);
	temp=strtok(fileNameCopy,".");
	while(temp!=NULL){
		if(ext!=NULL){delete[] ext;}
		ext=new char[strlen(temp)+1];
		assert(ext);
		strcpy(ext,temp);
		temp=strtok(NULL,".");
	}
	delete[] fileNameCopy;
	return ext;
}
