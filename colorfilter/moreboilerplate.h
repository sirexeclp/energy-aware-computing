
/*----------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
	bmp_t bmp_in, bmp_out;
	if (argc < 2 || argc > 3) {
		fprintf(stderr, "Usage: %s [task = a] bmp-file\n", argv[0]);
		exit(1);
	}

	char *filename = argv[1];
	char task = 'a';
	if (argc == 3) {
		filename = argv[2];
		task = argv[1][0];
	}

	bmp_read(&bmp_in, filename);

	if (!is_apt_for_exercise(&bmp_in)) {
		fprintf(stderr, "For the sake simplicity please provide a ARGB8888 image with a pixel count divisible by four.\n");
		exit(4);
	}

	bmp_copyHeader(&bmp_out, &bmp_in);

	switch (task) {
		case 'a':
			taskA(bmp_in.data, bmp_out.data, bmp_in.width, bmp_in.height);
			break;
		case 'b':
			taskB(bmp_in.data, bmp_out.data, bmp_in.width, bmp_in.height);
			break;
		case 'c':
			taskC(bmp_in.data, bmp_out.data, bmp_in.width, bmp_in.height);
			break;
		default:
			fprintf(stderr, "Invalid task.\n");
			exit(5);
	}

	bmp_write(&bmp_out, "output.bmp");
	bmp_free(&bmp_in);
	bmp_free(&bmp_out);

	return 0;
}