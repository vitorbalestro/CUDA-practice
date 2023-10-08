#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

__host__ int readJPEG(const char *filename, unsigned char **image_data, int *width, int *height) {
    // ... (Your existing code for reading JPEG)

    return 0;
}
__host__ void saveJPEG(const char *outputFileName, unsigned char *image_data, int width, int height) {
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

__global__ void rescaling(unsigned char *d_pin, unsigned char *d_pout, int L, int C, int t) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < t*L && col < t*C){
        
        if(row >= t*(L/4) && row < 3*t*(L/4) && col >= t*(L/4) && col < 3*t*(L/4))
        d_pout[col + row * t*C] = (d_pin[(col-1) + row * t*C] + d_pin[(col+1) + row * t*C] + d_pin[col + (row-1) * t*C]+ d_pin[col + (row+1) * t*C])/4;
        
       
    }
}

int main() {
    unsigned char *pout;
    unsigned char *d_pin, *d_pout;
    int width, height;
    const char *filename = "JF.jpeg"; 
    const char *output_name = "JF(test2).jpeg";
    unsigned char *image_data;

    if (readJPEG(filename, &image_data, &width, &height) != 0) {
        fprintf(stderr, "Error reading JPEG image.\n");
        return 1;
    }

    int size_pin = 3 * width * height * sizeof(unsigned char);
    int size_pout = 3 * width * height * sizeof(unsigned char);

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_pin, size_pin);
    cudaMalloc((void **)&d_pout, size_pout);

    // Correct cudaMemcpy to copy data from host to device
    cudaMemcpy(d_pin, image_data, size_pin, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 gridDim((width - 1) / 16 + 1, (height - 1) / 16 + 1);

    rescaling<<<gridDim, threadsPerBlock>>>(d_pin, d_pout, width, height, 3);

    // Allocate memory for pout on the host
    pout = (unsigned char *)malloc(size_pout);

    cudaMemcpy(&pout, d_pout, size_pout, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    saveJPEG(output_name, pout, width , height); // Corrected width and image size

    free(image_data);
    free(pout);

    cudaFree(d_pin);
    cudaFree(d_pout);

    return 0;
}
