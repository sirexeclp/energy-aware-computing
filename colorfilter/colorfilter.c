#include "boilerplate.h"
void taskA(uint8_t *in, uint8_t *out, int32_t width, int32_t height)
{
	pixel_t *pin = (pixel_t *)in;
	pixel_t *pout = (pixel_t *)out;
	pixel_t value;
	int p;

#pragma omp parallel for private(p, value)
	for (p = 0; p < width * height; p++)
	{
		value = pin[p];
		value.b = ~value.b;
		value.r = ~value.r;
		value.g = ~value.g;
		pout[p] = value;
	}
}

void taskB(uint8_t *in, uint8_t *out, int32_t width, int32_t height)
{
	pixel_t *pin = (pixel_t *)in;
	pixel_t *pout = (pixel_t *)out;
	pixel_t value;
	int p;

#pragma omp parallel for private(p, value)
	for (p = 0; p < width * height; p++)
	{
		value = pin[p];
		if ((value.r < value.g) || (value.r < value.b))
		{
			uint8_t grey = (value.r + value.g + value.g + value.b) / 4;
			value.r = grey;
			value.g = grey;
			value.b = grey;
		}
		pout[p] = value;
	}
}

void taskC(uint8_t *in, uint8_t *out, int32_t width, int32_t height)
{
	const int rounds = 100; // Even numbers mean simpler implementation
	pixel_t *pin = (pixel_t *)in;
	pixel_t *pout = (pixel_t *)out;

	for (int round = 0; round < rounds; round++)
	{
#pragma omp parallel for
		for (int y = 0; y < height; y++)
#pragma omp parallel for
			for (int x = 0; x < width; x++)
			{
				int accR = 0.;
				int accG = 0.;
				int accB = 0.;

				accR += ((y - 1 < 0) || x - 1 < 0) ? 0 : pin[(y - 1) * width + x - 1].r;
				accG += ((y - 1 < 0) || x - 1 < 0) ? 0 : pin[(y - 1) * width + x - 1].g;
				accB += ((y - 1 < 0) || x - 1 < 0) ? 0 : pin[(y - 1) * width + x - 1].b;

				accR += (y - 1 < 0) ? 0 : pin[(y - 1) * width + x].r;
				accG += (y - 1 < 0) ? 0 : pin[(y - 1) * width + x].g;
				accB += (y - 1 < 0) ? 0 : pin[(y - 1) * width + x].b;

				accR += ((y - 1 < 0) || x + 1 > width) ? 0 : pin[(y - 1) * width + x + 1].r;
				accG += ((y - 1 < 0) || x + 1 > width) ? 0 : pin[(y - 1) * width + x + 1].g;
				accB += ((y - 1 < 0) || x + 1 > width) ? 0 : pin[(y - 1) * width + x + 1].b;

				accR += (x - 1 < 0) ? 0 : pin[y * width + x - 1].r;
				accG += (x - 1 < 0) ? 0 : pin[y * width + x - 1].g;
				accB += (x - 1 < 0) ? 0 : pin[y * width + x - 1].b;

				accR += pin[y * width + x].r;
				accG += pin[y * width + x].g;
				accB += pin[y * width + x].b;

				accR += (x + 1 > width) ? 0 : pin[y * width + x + 1].r;
				accG += (x + 1 > width) ? 0 : pin[y * width + x + 1].g;
				accB += (x + 1 > width) ? 0 : pin[y * width + x + 1].b;

				accR += ((y + 1 > height) || x - 1 < 0) ? 0 : pin[(y + 1) * width + x - 1].r;
				accG += ((y + 1 > height) || x - 1 < 0) ? 0 : pin[(y + 1) * width + x - 1].g;
				accB += ((y + 1 > height) || x - 1 < 0) ? 0 : pin[(y + 1) * width + x - 1].b;

				accR += ((y + 1 > height)) ? 0 : pin[(y + 1) * width + x].r;
				accG += ((y + 1 > height)) ? 0 : pin[(y + 1) * width + x].g;
				accB += ((y + 1 > height)) ? 0 : pin[(y + 1) * width + x].b;

				accR += ((y + 1 > height) || x + 1 > width) ? 0 : pin[(y + 1) * width + x + 1].r;
				accG += ((y + 1 > height) || x + 1 > width) ? 0 : pin[(y + 1) * width + x + 1].g;
				accB += ((y + 1 > height) || x + 1 > width) ? 0 : pin[(y + 1) * width + x + 1].b;

				pout[y * width + x].r = (uint8_t)(accR / 9);
				pout[y * width + x].g = (uint8_t)(accG / 9);
				pout[y * width + x].b = (uint8_t)(accB / 9);
				pout[y * width + x].a = pin[y * width + x].a;
			}
#pragma omp barrier
		uint8_t *tmp = in;
		in = out;
		out = tmp;
	}
}

void taskD(uint8_t *in, uint8_t *out, int32_t width, int32_t height)
{
	pixel_t *pin = (pixel_t *)in;
	pixel_t *pout = (pixel_t *)out;
	pixel_t value;
	int p;

#pragma omp parallel for private(p, value)
	for (p = 0; p < width * height; p++)
	{
		value = pin[p];
		if ((value.r < value.g) || (value.r < value.b))
		{
			uint8_t grey = (value.r + value.g + value.g + value.b) / 4;
			value.r = grey;
			value.b = grey;
			//its not a bug; it's art :D
			value.b = grey;
		}
		pout[p] = value;
	}
}
#include "moreboilerplate.h"