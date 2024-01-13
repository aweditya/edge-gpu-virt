// C standard includes
#include <stdio.h>

// OpenCL includes
#include <CL/cl.h>

typedef struct platform_info
{
    char profile[64];
    char version[64];
    char name[64];
    char vendor[64];
    char extensions[64];
} platform_info_t;

typedef struct device_info
{
    cl_device_type type;
    cl_uint vendor_id;
    cl_uint max_compute_units;
    cl_uint max_work_item_dims;
    size_t max_work_group_size;
    cl_uint max_clk_freq;
    cl_uint device_addr_bits;
} device_info_t;

int main()
{
    size_t param_value_size_ret = 0;

    cl_int CL_err = CL_SUCCESS;
    cl_uint num_platforms = 0;

    // Get all platforms available (platforms are OpenCL implementations)
    CL_err = clGetPlatformIDs(0, NULL, &num_platforms);

    if (CL_err == CL_SUCCESS)
    {
        printf("%u platform(s) found\n", num_platforms);
    }
    else
    {
        printf("clGetPlatformIDs(%i)\n", CL_err);
    }

    cl_platform_id *platforms;
    platforms = malloc(sizeof(cl_platform_id) * num_platforms);
    CL_err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (CL_err == CL_SUCCESS)
    {
        printf("Read all platforms\n");
    }
    else
    {
        printf("clGetPlatformIDs(%i)\n", CL_err);
    }

    // Info about each platform
    platform_info_t plat_info;
    for (int i = 0; i < num_platforms; ++i)
    {
        printf("Platform %d:\n", i);
        CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 64, plat_info.profile, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_PLATFORM_PROFILE: %s\n", param_value_size_ret, plat_info.profile);
        }
        else
        {
            printf("clGetPlatformInfo(%i)\n", CL_err);
        }

        CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 64, plat_info.version, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_PLATFORM_VERSION: %s\n", param_value_size_ret, plat_info.version);
        }
        else
        {
            printf("clGetPlatformInfo(%i)\n", CL_err);
        }

        CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 64, plat_info.name, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_PLATFORM_NAME: %s\n", param_value_size_ret, plat_info.name);
        }
        else
        {
            printf("clGetPlatformInfo(%i)\n", CL_err);
        }

        CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 64, plat_info.vendor, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_PLATFORM_VENDOR: %s\n", param_value_size_ret, plat_info.vendor);
        }
        else
        {
            printf("clGetPlatformInfo(%i)\n", CL_err);
        }
    }

    // Devices on platform 0
    cl_uint num_devices;
    CL_err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (CL_err == CL_SUCCESS)
    {
        printf("%u device(s) found in platform 0\n", num_devices);
    }
    else
    {
        printf("clGetDeviceIDs(%i)\n", CL_err);
    }

    cl_device_id *devices;
    devices = malloc(sizeof(cl_device_id) * num_devices);
    CL_err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (CL_err == CL_SUCCESS)
    {
        printf("Read all devices on platform 0\n");
    }
    else
    {
        printf("clGetDeviceIDs(%i)\n", CL_err);
    }

    // Info about each device on platform 0
    device_info_t dev_info;
    for (int i = 0; i < num_devices; ++i)
    {
        printf("Device %d on platform 0:\n", i);
        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_info.type, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_DEVICE_TYPE: %ld\n", param_value_size_ret, dev_info.type);
        }
        else
        {
            printf("clGetDeviceInfo(%i)\n", CL_err);
        }

        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &dev_info.vendor_id, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_DEVICE_VENDOR_ID: %d\n", param_value_size_ret, dev_info.vendor_id);
        }
        else
        {
            printf("clGetDeviceInfo(%i)\n", CL_err);
        }

        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &dev_info.max_compute_units, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", param_value_size_ret, dev_info.max_compute_units);
        }
        else
        {
            printf("clGetDeviceInfo(%i)\n", CL_err);
        }

        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dev_info.max_work_item_dims, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_DEVICE_MAX_WORK_ITEMS_DIMENSION: %d\n", param_value_size_ret, dev_info.max_work_item_dims);
        }
        else
        {
            printf("clGetDeviceInfo(%i)\n", CL_err);
        }

        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &dev_info.max_work_group_size, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", param_value_size_ret, dev_info.max_work_group_size);
        }
        else
        {
            printf("clGetDeviceInfo(%i)\n", CL_err);
        }

        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &dev_info.max_clk_freq, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_DEVICE_MAX_CLOCK_FREQUENCY: %d MHz\n", param_value_size_ret, dev_info.max_clk_freq);
        }
        else
        {
            printf("clGetDeviceInfo(%i)\n", CL_err);
        }

        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &dev_info.device_addr_bits, &param_value_size_ret);
        if (CL_err == CL_SUCCESS)
        {
            printf("[Read %lu bytes] CL_DEVICE_ADDRESS_BITS: %d bits\n", param_value_size_ret, dev_info.device_addr_bits);
        }
        else
        {
            printf("clGetDeviceInfo(%i)\n", CL_err);
        }
    }

    // Create context
    cl_context context;
    context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &CL_err);
    if (CL_err == CL_SUCCESS)
    {
        printf("Created context\n");
    }
    else
    {
        printf("clCreateContext(%i)\n", CL_err);
    }

    // Create a command queue
    cl_command_queue queue;
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, &CL_err);
    if (CL_err == CL_SUCCESS)
    {
        printf("Created a command queue\n");
    }
    else
    {
        printf("clCreateCommandQueueWithProperties(%i)\n", CL_err);
    }
#else
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &CL_err);
    if (CL_err == CL_SUCCESS)
    {
        printf("Created a command queue\n");
    }
    else
    {
        printf("clCreateCommandQueue(%i)\n", CL_err);
    }
#endif

    CL_err = clReleaseCommandQueue(queue);
    if (CL_err == CL_SUCCESS)
    {
        printf("Released command queue\n");
    }
    else
    {
        printf("clReleaseCommandQueue(%i)\n", CL_err);
    }

    CL_err = clReleaseContext(context);
    if (CL_err == CL_SUCCESS)
    {
        printf("Released context\n");
    }
    else
    {
        printf("clReleaseContext(%i)\n", CL_err);
    }

    free(platforms);
    free(devices);
    return 0;
}
