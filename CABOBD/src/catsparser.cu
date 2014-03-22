#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "catsparser.cuh"



FILE * open_file( char const * file) {
	FILE * fp;
	fp = fopen(file,"r");
	if(!fp) {
		fprintf(stderr, "Error opening file %s \n",file);
		exit(EXIT_FAILURE);
	}
	return fp;
}

void check_config_error(struct config * conf, int line, char * file) {
	unsigned int status = 0;
	status |= (conf->nbids <= 0); // if no bids
	status |= (conf->goods <= 0);
	status |= (conf->bids == NULL); // if no pointer
	if(status) {
		fprintf(stderr, "Error parsing auction at line %d in %s\n Aborting execution!\n",line,file);
		exit(EXIT_FAILURE);
	}
}

unsigned int match_string_toint(const char * string, FILE * fp) {
	unsigned int ret = 0;

	char * line = NULL;
	size_t len = 0;
	size_t slen = strlen(string);

	while (getline(&line, &len, fp) != -1) {
			if(strncmp(line,string,slen) == 0) {
				line = (line+slen);
				while(*line == ' ') line++;
				char * start = line;
				while(isdigit(*line)) line++;
				ret = strtol(start, &line,10);
				break;
			}

	}
	return ret;
}


void parse_bid_line(FILE * fp, struct config * conf) {

	struct fbid * bid = conf->bids;
	char * line = NULL;
	size_t len = 0;

	//skip all lines which is not a bid line, they start woth a number
	while (getline(&line, &len, fp) != -1 && !isdigit(*line));

	// ex bidline
	// id	val		goods
	// 0	597101	17	23	24	#

	do{
		printf("\n");
		bid->bidstr = 0;
		char * head,*tail;
		head = tail = line;


		while(*head != '\t') head++;
		bid->id = strtol(tail, &head,10);
		tail = head;
		printf("id:%u\t",bid->id);

		while(*head != '\t') head++;
		bid->value = strtol(tail, &head,10);
		tail = head;
		printf("value:%u\t",bid->value);

		head++;
		while(*head != '#') { // parse goods
			if(*head == '\t') {
				bid->bidstr |= (1 << strtol(tail,&head,10));
				printf("%lu\t",strtol(tail,&head,10));
				tail = head;
			}
			head++;
		}
		bid++;
	} while (getline(&line, &len, fp) != -1);

}



void parse_auction_size(FILE * fp, struct config * conf) {

	conf->goods = match_string_toint("goods",fp);
	printf("Goods: %d\n",conf->goods);
	conf->nbids = match_string_toint("bids",fp);
	printf("Bids: %d\n",conf->nbids);
	conf->dummy_bids = match_string_toint("dummy",fp);
	printf("Dummy goods: %d\n",conf->dummy_bids);

	conf->bids =(struct fbid *) malloc(sizeof(struct fbid)*conf->nbids);
	memset(conf->bids,0,sizeof(struct fbid)*conf->nbids); // set everything to 0
	check_config_error(conf,__LINE__,__FILE__);
}

struct config * parse_file(char const* file) {
	struct config * conf = (struct config *) malloc(sizeof(struct config));
	if(!conf) {
		fprintf(stderr, "Could not allocate memory at line %d in %s\n Aborting execution!\n",__LINE__,__FILE__);
		exit(EXIT_FAILURE);
	}
	printf("Opening file %s\n",file);
	FILE * fp = open_file(file);
	printf("Parsing auction size\n");
	parse_auction_size(fp,conf);
	printf("Parsing bid lines\n");
	parse_bid_line(fp,conf);
	return conf;
}


