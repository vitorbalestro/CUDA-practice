// TO COMPILE: nvcc open-image.c -o (name of the executable) -ljpeg

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

// Function to read a JPEG image and convert it to a pixel matrix
int readJPEG(const char* filename, unsigned char** image_data, int* width, int* height, int *num_components) {
    // Declare the JPEG decompression object
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Open the JPEG file
    FILE* infile = fopen(filename, "rb");
    if (!infile) {
        fprintf(stderr, "Error opening JPEG file.\n");
        return 1;
    }

    // Initialize the JPEG decompression object error handler
    cinfo.err = jpeg_std_error(&jerr);

    // Create the JPEG decompression object
    jpeg_create_decompress(&cinfo);

    // Specify the source file
    jpeg_stdio_src(&cinfo, infile);

    // Read header information from the JPEG file
    jpeg_read_header(&cinfo, TRUE);

    // Start the decompression process
    jpeg_start_decompress(&cinfo);

    // Allocate memory for the pixel data
    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *num_components = cinfo.num_components;
    *image_data = (unsigned char*)malloc(*width * *height * *num_components);

    // Read scanlines and fill the pixel matrix
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row = *image_data + cinfo.output_scanline * *width * *num_components;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    // Finish the decompression process
    jpeg_finish_decompress(&cinfo);

    // Clean up and release resources
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return 0;
}

void array2JPEG(const char *outputFileName, unsigned char *image_data, int width, int height, int num_components) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *outfile;
    if ((outfile = fopen(outputFileName, "wb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", outputFileName);
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = num_components; 
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    int row_stride = width * num_components; 
    JSAMPROW row_pointer;
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = &image_data[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}


int main() {

    const char* filename = "JF.jpeg"; // Change to your JPEG file
    const char *output_name = "JF(test).jpeg";
    unsigned char* image_data;
    unsigned char* output_image;
    int width, height, num_components;

    if (readJPEG(filename, &image_data, &width, &height, &num_components) != 0) {
        fprintf(stderr, "Error reading JPEG image.\n");
        return 1;
    }
    int size = width*height*num_components;
    output_image = (unsigned char*)malloc(size);

    // copy array of input image to output_image array
    for(int i = 0; i < size; i++){
        output_image[i] = image_data[i];
    }
    
    array2JPEG(output_name,output_image,width,height,num_components);
    
    free(image_data);
    free(output_image);

    return 0;
}
