// TO COMPILE: nvcc image-to-gpu.cu -o (name of the executable) -ljpeg
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

__host__ int readJPEG(const char* filename, unsigned char** image_data, int* width, int* height, int *num_components) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE* infile = fopen(filename, "rb");
    if (!infile) {
        fprintf(stderr, "Error opening JPEG file.\n");
        return 1;
    }

    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);

    jpeg_read_header(&cinfo, TRUE);

    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *num_components = cinfo.num_components;
    *image_data = (unsigned char*)malloc(*width * *height * *num_components);

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row = *image_data + cinfo.output_scanline * *width * *num_components;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return 0;
}

__host__ void array2JPEG(const char *outputFileName, unsigned char *image_data, int width, int height, int num_components) {
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

__global__ void send_image_to_GPU(unsigned char *d_pin, unsigned char *d_pout, int width, int height, int num_components){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < height && col < width * num_components){
        d_pout[col + row * width * num_components] = d_pin[col + row * width * num_components];
    }
}
int main() {

    const char* filename = "JF.jpeg"; // Change to your JPEG file
    const char* output_name = "JF(test).jpeg";
    unsigned char* image_data;
    int width, height,num_components;
    
    
    if (readJPEG(filename, &image_data, &width, &height, &num_components) != 0) {
        fprintf(stderr, "Error reading JPEG image.\n");
        return 1;
    }
    unsigned char *d_pin, *d_pout;
    int size = num_components * width * height;
    unsigned char* output_image;
    output_image = (unsigned char*)malloc(size);

    cudaMalloc((void**)&d_pin,size);
    cudaMalloc((void**)&d_pout,size);
    cudaMemcpy(d_pin, image_data, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock (16,16);
    dim3 gridDim ((height-1)/16 + 1, (width * num_components-1)/16 + 1);

    send_image_to_GPU<<<gridDim, threadsPerBlock>>>(d_pin, d_pout, width, height, num_components);
    
    cudaMemcpy(output_image,d_pout,size,cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();


    array2JPEG(output_name,output_image,width,height,num_components);

    free(image_data);
    free(output_image);
    cudaFree(d_pin);
    cudaFree(d_pout);

    return 0;
}
