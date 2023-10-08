#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

/* #define _WIDTH 1600 // number of lines
#define _HEIGHT 1200 // number of columns
#define _TYPE 3 // RGB */

__host__ int readJPEG(const char* filename, unsigned char** image_data, int* width, int* height) {
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
    int numChannels = cinfo.num_components;
    *image_data = (unsigned char*)malloc(*width * *height * numChannels);

    // Read scanlines and fill the pixel matrix
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row = *image_data + cinfo.output_scanline * *width * numChannels;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    // Finish the decompression process
    jpeg_finish_decompress(&cinfo);

    // Clean up and release resources
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return 0;
}

void saveJPEG(const char *outputFileName, unsigned char *image_data, int width, int height) {
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
    cinfo.input_components = 3; // Assuming RGB format
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    int row_stride = width * 3; // RGB format
    JSAMPROW row_pointer;
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = &image_data[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}


__global__ void rescaling(int *d_pin, int *d_pout, int L, int C, int t){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < t*L && col < t*C){
        if(row >= t*(L/4) && row < 3*t*(L/4) && col >= t*(L/4) && col < 3*t*(L/4))
        d_pout[col + row * t*C] = (d_pout[(col-1) + row * t*C] + d_pout[(col+1) + row * t*C] + d_pout[col + (row-1) * t*C]+ d_pout[col + (row+1) * t*C])/4;

    }

}

int main() {

    unsigned char *pout;
    int *d_pin, *d_pout;
    int width, height;
    const char* filename = "JF.jpeg"; // Change to your JPEG file
    const char *output_name = "JF(test).jpeg";
    unsigned char* image_data;

    if (readJPEG(filename, &image_data, &width, &height) != 0) {
        fprintf(stderr, "Error reading JPEG image.\n");
        return 1;
    }

    int size_pin = 3 * width * height * sizeof(unsigned char);
    int size_pout = 3 * (width) * (height) * sizeof(unsigned char);

    cudaMalloc((void**)&d_pin,size_pin);
    cudaMalloc((void**)&d_pout,size_pout);

    cudaMemcpy(d_pin,&image_data,size_pin,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = (16,16);
    dim3 gridDim = ((height-1)/16 + 1, (width-1)/16 + 1);

    rescaling<<<gridDim,threadsPerBlock>>>(d_pin, d_pout, width, height, 3);

    pout = (unsigned char*)malloc(size_pout);
    
    cudaMemcpy(&pout, d_pout, size_pout, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    saveJPEG(output_name,pout,1600,1200);
    
    free(image_data);

    cudaFree(d_pin);
    cudaFree(d_pout);

    
}