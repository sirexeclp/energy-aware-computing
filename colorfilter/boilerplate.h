#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define BMP_CORE_HEADER_SIZE 14
#define BMP_V5_HEADER_SIZE	124

#define BMP_BI_BITFIELDS 3  // uncompressed, masks are specified

#define BMP_ARGB8888_A_MASK 0xFF000000
#define BMP_ARGB8888_R_MASK 0x00FF0000
#define BMP_ARGB8888_G_MASK 0x0000FF00
#define BMP_ARGB8888_B_MASK 0x000000FF

typedef union _pixel {
	struct {
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	};
	char bgra[4];
} pixel_t;

typedef struct _ciexyz_triple
{
	int32_t redX;
	int32_t redY;
	int32_t redZ;
	int32_t greenX;
	int32_t greenY;
	int32_t greenZ;
	int32_t blueX;
	int32_t blueY;
	int32_t blueZ;
} ciexyz_triple_t;

typedef struct _bmp {
	union {
		char header[BMP_CORE_HEADER_SIZE + BMP_V5_HEADER_SIZE];
		struct __attribute__((packed)) {
			uint16_t signature;
			uint32_t bytes;
			uint32_t reserved;
			uint32_t offset;
			// V5 Header
			uint32_t infoHeaderSize;
			int32_t  width;
			int32_t  height;
			uint16_t planes;
			uint16_t bitCount;
			uint32_t compression;
			uint32_t sizeImage;
			int32_t  xPPM;
			int32_t  yPPM;
			uint32_t clrUsed;
			uint32_t clrImportant;
			uint32_t redMask;
			uint32_t greenMask;
			uint32_t blueMask;
			uint32_t alphaMask;
			uint32_t csType;
			ciexyz_triple_t endpoints;
			uint32_t gammaRed;
			uint32_t gammaGreen;
			uint32_t gammaBlue;
			uint32_t intent;
			uint32_t profileData;
			uint32_t profileSize;
			uint32_t reserved_v5;
		};
	};
	int isTopDown;
	int yOffset;
	int yDirection;
	int channels;
	uint8_t *data;
} bmp_t;

/*----------------------------------------------------------------------------*/

void bmp_read(bmp_t *bmp, const char *filename)
{
	FILE *f = fopen(filename, "r");

	if (!f) {
		printf("Cannot open %s for reading.\n", filename);
		exit(2);
	}

	fread(bmp->header, sizeof(bmp->header), 1, f);

	if (!(bmp->header[0] == 'B' && bmp->header[1] == 'M' && bmp->infoHeaderSize == BMP_V5_HEADER_SIZE)) {
		fprintf(stderr, "Unsupported bitmap format.\n");
		exit(3);
	}

	/* We are storing 4 channel bmp in the end. Override old value */
	bmp->channels = bmp->bitCount / 8;
	bmp->bitCount = 32;

	/* top-down bitmaps (top row first) are stored with negative height */
	bmp->isTopDown = bmp->height < 0;
	bmp->height = abs(bmp->height);

	/* To calculate right row in GET_PIXEL/SET_PIXEL */
	bmp->yDirection = bmp->isTopDown ? 1 : -1;
	bmp->yOffset = bmp->isTopDown ? 0 : bmp->height - 1;

	/* Allocate memory for our pixel data and read file */
	bmp->sizeImage = bmp->width * bmp->height * bmp->channels;
	bmp->data = malloc(bmp->sizeImage);
	fread(bmp->data, bmp->sizeImage, 1, f);

	fclose(f);
}

void bmp_copyHeader(bmp_t *bmp, bmp_t *other)
{
	memcpy(bmp, other, sizeof(bmp_t));
	bmp->data = calloc(bmp->sizeImage, 1);
}

void bmp_write(bmp_t *bmp, const char *filename)
{
	int x, y;
	pixel_t output;
	FILE *f = fopen(filename, "w");

	fwrite(bmp->header, sizeof(bmp->header), 1, f);

	/* Store bitmap bottom-up (thus positive height) */
	for (y = bmp->height - 1; y >= 0; y--) {
		for (x = 0; x < bmp->width; x++) {
			output = ((pixel_t *)bmp->data)[(y * bmp->yDirection + bmp->yOffset) * bmp->width + x];
			if (fwrite(output.bgra, sizeof(output.bgra), 1, f) != 1) {
				fprintf(stderr, "Cannot write to file %s.\n", filename);
				exit(6);
			}
		}
	}

	fclose(f);
}

void bmp_free(bmp_t *bmp)
{
	free(bmp->data);
	bmp->data = NULL;
}

/*----------------------------------------------------------------------------*/

int is_apt_for_exercise(bmp_t *bmp)
{
	int is_argb =
		(bmp->channels == 4) &&
		(bmp->compression == BMP_BI_BITFIELDS) &&
		(bmp->redMask     == BMP_ARGB8888_R_MASK) &&
		(bmp->greenMask   == BMP_ARGB8888_G_MASK) &&
		(bmp->blueMask    == BMP_ARGB8888_B_MASK) &&
		(bmp->alphaMask   == BMP_ARGB8888_A_MASK);
	int is_simdable =
		((bmp->width * bmp->height) % 4 == 0);
	return is_argb && is_simdable;
}