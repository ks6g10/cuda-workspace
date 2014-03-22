
#include <inttypes.h>

#ifndef CATS_PARSER_H
#define CATS_PARSER_H


struct fbid {
	uint32_t bidstr;
	uint32_t value;
	uint16_t dummy;
	uint16_t id;
};

struct config {
	struct fbid * bids;
	unsigned int nbids;
	unsigned int dummy_bids;
	unsigned int goods;
};

struct config * parse_file(char const *file);

#endif
