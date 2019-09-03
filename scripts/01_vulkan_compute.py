'''
Python script to allocate buffer and perform simple compute operation in GPU using Vulkan API.
Summary of the script:
    - Vulkan is very low level API and often requires a lot of boilerplate coding
    - Allocate two buffer memories (in and out) of equal size in GPU
    - Map the buffer memory with Host/CPU in order to initialize int values
    - Use numpy to modify values in the buffers for fun
    - Create GLSL code that copies 2 times values 'input' buffer to 'output' buffer
      (GLSL code has already been precompiled to SPIR-V binary format) 
    - Map the buffer memory with Host to check the computed result

Based on Neil Henning's following blog post about sample C++ code for 
GPU computation using (low level) Vulkan API.

http://www.duskborn.com/posts/a-simple-vulkan-compute-example/

Notes:
 - In functions starting with "vk", argument expecting pointer can be supplied with
   object. The function internally computes the pointer. The converse, however, is
   not true.
 - This script requires installation of:
   1. numpy
   2. vulkan (https://github.com/realitix/vulkan)
   3. Vulkan SDK or runtime (https://vulkan.lunarg.com/)  
'''

import logging
logging.basicConfig(level = logging.DEBUG)
import ctypes
import random
import numpy as np
from vulkan import *
from vulkan._vulkancache import ffi

# ************************************************************************************************
# The 3-D compute space is divided into a number of work groups in each dimension. 
# Each work group is broken down into a number of invocations aka subgroups or local work groups.
# This problem uses 1-D buffers with 256 work groups and 64 invocations per each for computation.

buffer_length = 16384
total_thread_per_invocation = 64 # this is hardcoded in compiled compute shader code 
total_work_groups =  int(buffer_length/total_thread_per_invocation)

# ************************************************************************************************
# Boiler plate code to select appropiate GPU device, create logical device, compute queue,
# allocate memory for data, etc.

appInfo = VkApplicationInfo(sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            pApplicationName = "Simple GPU computation using Python",
                            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
                            pEngineName = "Gyan Basyal",
                            engineVersion = VK_MAKE_VERSION(1, 0, 0),
                            apiVersion = VK_MAKE_VERSION(1, 1, 114)
                            )

createInfo = VkInstanceCreateInfo(sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 flags = 0,
                                 pApplicationInfo = appInfo,
                                 )

instance = vkCreateInstance(createInfo, None)

logging.info('query physical devices, use the first device')
physical_device = vkEnumeratePhysicalDevices(instance)[0]
physical_device_feature = vkGetPhysicalDeviceFeatures(physical_device)
physical_device_property = vkGetPhysicalDeviceProperties(physical_device) # not used?
# limits local_size_x, ... in shade code (total invocation per work group)
maxWorkGroupSizeX,maxWorkGroupSizeY,maxWorkGroupSizeZ = physical_device_property.limits.maxComputeWorkGroupSize  
# Max work group number
maxWorkGroupCountX,maxWorkGroupCountY,maxWorkGroupCountZ = physical_device_property.limits.maxComputeWorkGroupCount  

assert(maxWorkGroupSizeX >= total_thread_per_invocation), 'Too large work group invocation size'
assert(maxWorkGroupCountX >= total_work_groups), 'Too large work group size'

logging.debug('query compute queue family')
queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=physical_device)
queue_family_compute_index = -1

# find unit within the device that supports computation operation
for i, queue_family in enumerate(queue_families):
    if (queue_family.queueCount > 0 and queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT):
        queue_family_compute_index = i
        # found it!
        break

device_queue_create = [VkDeviceQueueCreateInfo(sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                               queueFamilyIndex = queue_family_compute_index,
                                               queueCount = 1,
                                               pQueuePriorities = [1.0],
                                               flags = 0
                                               )]      

device_create = VkDeviceCreateInfo(sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                   pQueueCreateInfos = device_queue_create,
                                   queueCreateInfoCount = len(device_queue_create),
                                   pEnabledFeatures = physical_device_feature,
                                   flags= 0
                                   )

logging.info('create logical device in the physical device') 
logical_device = vkCreateDevice(physical_device, device_create, None)

logging.info('Allocate total 2*4*16384 bytes (_ppData) in gpu for two buffers') 
# The memory will be divided equally between two buffers

buffer_size = buffer_length * ctypes.sizeof(ctypes.c_int)  
mem_size = buffer_size * 2  
mem_props = vkGetPhysicalDeviceMemoryProperties(physical_device)
mem_type_index = -1
for k in range(mem_props.memoryTypeCount):
    if  ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & mem_props.memoryTypes[k].propertyFlags) and
        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & mem_props.memoryTypes[k].propertyFlags) and                
        (mem_size < mem_props.memoryHeaps[mem_props.memoryTypes[k].heapIndex].size)):
        mem_type_index = k
        break

if mem_type_index == -1:
    print('Memory allocation error')
    sys.exit(1)
elif mem_type_index == VK_MAX_MEMORY_TYPES:
    print('Insufficient Host Memory')
    sys.exit(1)

memAllocInfo = VkMemoryAllocateInfo(sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    allocationSize = mem_size,
                                    memoryTypeIndex = mem_type_index 
                                    )

device_memory = vkAllocateMemory(device = logical_device,
                                 pAllocateInfo = memAllocInfo,
                                 pAllocator = 0,
                                 )
                                   
logging.info('Make the GPU memory visible to CPU') 
_ppData = vkMapMemory(device = logical_device,
                      memory = device_memory,
                      offset = 0,
                      size = mem_size,
                      flags = 0)

# ************************************************************************************************
logging.info('Initialize the memory with random integer using _ppData, which is visible to CPU') 

ppData = ffi.from_buffer("int[]",_ppData)
print('** Assign random int between -1000 and 1000')
for x in range(buffer_length):
    ppData[x] = random.randint(-1000,1000)

# Further modify gpu memory (using numpy) from CPU. Finally, assign 4000 to input buffer and
# 0 to output buffer. 

arr = np.frombuffer(_ppData,dtype=np.int32)
print('** Numpy array:')
print(arr)

arr[2] = 100
print('** Buffer after assigning 100 to index 2:')
print('Numpy array: %r'%arr)
print(ppData[2])
if not arr[2] == ppData[2]:
    print('Error')

arr[:buffer_length] = 4000
arr[buffer_length:] = 0
print('** Buffer after assigning 4000 and 0 to in_buffer and out_buffer, respectively')
print('Numpy array: %r'%arr)
print(ppData[2])

arr_out_init = arr[buffer_length:].copy()

# ************************************************************************************************
logging.info('Create input and output buffers in gpu for use in compute shader')
# No more need to map GPU memory to Host after initialization of values 
vkUnmapMemory(device = logical_device,
              memory = device_memory)

# Bind two buffers with gpu memory
bufferInfo = VkBufferCreateInfo(sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                size = mem_size,
                                usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  
                                sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                queueFamilyIndexCount = 1,
                                pQueueFamilyIndices = [mem_type_index]   
                                )

in_buffer = vkCreateBuffer(device = logical_device,
                           pCreateInfo = bufferInfo,
                           pAllocator = 0)

vkBindBufferMemory(device = logical_device,
                   buffer = in_buffer,
                   memory = device_memory,
                   memoryOffset = 0  
                   )

out_buffer = vkCreateBuffer(device = logical_device,
                            pCreateInfo = bufferInfo,
                            pAllocator = 0)

vkBindBufferMemory(device = logical_device,
                   buffer = out_buffer,
                   memory = device_memory,
                   memoryOffset = buffer_size  
                   )
# ************************************************************************************************
# ************************************************************************************************
# COMPUTE SHADER: Copy 2 times in_buffer values to out_buffer
# GPU computation code is called shader (i.e., vertex shader, fragment shader or compute shader).
# Vulkan requires shader code in SPIR-V format, which is sequence of 32-bit word instructions (e.g., int32_t shader[] in C/C++).
# It is much easier to write compute shader code in OpenGL Shading Language (GLSL) and convert it to SPIR-V binary
# using glslangValidator command that is available after installing Vulkan SDK.
# The GLSL code for this problem is as follows. This has already been precompiled as shader\01_vulkan_compute.comp.spv.
'''
#version 450
layout (local_size_x = 64) in;
layout (std430,binding = 0) buffer  lay0 {
    int in_buffer[];
};
layout (std430,binding = 1) buffer  lay1 {
    int out_buffer[];
};
void main() 
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx < in_buffer.length()) {
        out_buffer[idx] = 2*in_buffer[idx];
    }
}
'''
# Note that the above GLSL code refers to buffer variables using binding index 0 and 1 respectively.
# Vulkan API function will be used later to bind in_buffer and out_buffer to these indices.
logging.info('Load SPIR-V shader binary code')
with open(r'shader\01_vulkan_compute.comp.spv','rb') as fid:
    shader_data = fid.read()

comp_shader_spirv = shader_data
comp_shader_spirv_size = len(comp_shader_spirv)

logging.debug('Create shader')

shaderInfo = VkShaderModuleCreateInfo(sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      codeSize = comp_shader_spirv_size,
                                      pCode = comp_shader_spirv
                                     )

p_shader_module = vkCreateShaderModule(device = logical_device, 
                                     pCreateInfo = shaderInfo,   #cffi turns this to pointer?? 
                                     pAllocator = 0
                                    )

shader_main = ffi.new('char[]',b'main')

# Now bind indices to the two buffers for use in shader stage
binding0 = VkDescriptorSetLayoutBinding(binding = 0,
                                        descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        descriptorCount = 1,
                                        stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
                                        )
binding1 = VkDescriptorSetLayoutBinding(binding = 1,
                                        descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        descriptorCount = 1,
                                        stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
                                        )

descriptorInfo = VkDescriptorSetLayoutCreateInfo(sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                 bindingCount = 2,
                                                 pBindings = [binding0,binding1]
                                                )

p_descriptorSetLayout = vkCreateDescriptorSetLayout(device = logical_device, 
                                                    pCreateInfo = descriptorInfo, 
                                                    pAllocator = 0
                                                   )

logging.info('Create shader stage and compute pipelines')
# boilerplate code to run the compute command ...

pipeLayoutInfo = VkPipelineLayoutCreateInfo(sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            setLayoutCount = 1,
                                            pSetLayouts = [p_descriptorSetLayout]
                                           )

pipeLayout = vkCreatePipelineLayout(device = logical_device,
                                    pCreateInfo = pipeLayoutInfo,
                                    pAllocator = 0
                                    )

shaderStageInfo = VkPipelineShaderStageCreateInfo(sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                  stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                  module = p_shader_module,
                                                  pName = shader_main,
                                                  pSpecializationInfo = 0)

pipeComputeInfo = VkComputePipelineCreateInfo(sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                              stage = shaderStageInfo,
                                              layout = pipeLayout
                                              )
                                            
p_pipeline = vkCreateComputePipelines(device = logical_device, 
                                      pipelineCache = 0, 
                                      createInfoCount = 1, 
                                      pCreateInfos = [pipeComputeInfo],
                                      pAllocator = 0
                                      )

cmdPoolInfo = VkCommandPoolCreateInfo(sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      queueFamilyIndex = queue_family_compute_index
                                      )

p_descriptorPoolSize = VkDescriptorPoolSize(type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          descriptorCount = 2)

descriptorPoolInfo = VkDescriptorPoolCreateInfo(sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                maxSets = 1,
                                                poolSizeCount = 1,
                                                pPoolSizes = p_descriptorPoolSize)

descriptorPool = vkCreateDescriptorPool(device = logical_device, 
                                        pCreateInfo = descriptorPoolInfo, 
                                        pAllocator = 0
                                        )


descriptorAllocInfo = VkDescriptorSetAllocateInfo(sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                  descriptorPool = descriptorPool,
                                                  descriptorSetCount = 1,
                                                  pSetLayouts = [p_descriptorSetLayout]
                                                  )

p_descriptorSet = vkAllocateDescriptorSets(device = logical_device, 
                                           pAllocateInfo = descriptorAllocInfo
                                           )

in_descBufferInfo =  VkDescriptorBufferInfo(buffer = in_buffer,
                                            offset = 0,
                                            range = buffer_size
                                            )


out_descBufferInfo =  VkDescriptorBufferInfo(buffer = out_buffer,
                                             offset = 0,
                                             range = buffer_size
                                             )

writeDescSet0 = VkWriteDescriptorSet(sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                     dstSet = p_descriptorSet[0],
                                     dstBinding = 0,
                                     dstArrayElement = 0,
                                     descriptorCount = 1,
                                     descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     pBufferInfo = in_descBufferInfo,
                                     )

writeDescSet1 = VkWriteDescriptorSet(sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                     dstSet = p_descriptorSet[0],
                                     dstBinding = 1,
                                     dstArrayElement = 0,
                                     descriptorCount = 1,
                                     descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     pBufferInfo = out_descBufferInfo,
                                     )

vkUpdateDescriptorSets(device = logical_device, 
                       descriptorWriteCount = 2, 
                       pDescriptorWrites = [writeDescSet0, writeDescSet1], 
                       descriptorCopyCount = 0,
                       pDescriptorCopies = 0
                       )

cmdPool = vkCreateCommandPool(device = logical_device, 
                              pCreateInfo = cmdPoolInfo, 
                              pAllocator = 0
                              )

cmdBufferAllocInfo = VkCommandBufferAllocateInfo(sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                            commandPool = cmdPool,
                                            level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                            commandBufferCount = 1
                                            )

p_cmdBuffer = vkAllocateCommandBuffers(device = logical_device,
                                     pAllocateInfo = cmdBufferAllocInfo
                                     )

cmdBufferBeginInfo = VkCommandBufferBeginInfo(sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                                              )

vkBeginCommandBuffer(commandBuffer = p_cmdBuffer[0], 
                     pBeginInfo = cmdBufferBeginInfo
                     )

vkCmdBindPipeline(commandBuffer = p_cmdBuffer[0], 
                  pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE, 
                  pipeline = p_pipeline[0])

vkCmdBindDescriptorSets(commandBuffer = p_cmdBuffer[0],
                        pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE,
                        layout = pipeLayout,
                        firstSet = 0,
                        descriptorSetCount = 1,
                        pDescriptorSets = p_descriptorSet,
                        dynamicOffsetCount = 0,
                        pDynamicOffsets = 0
                        )

vkCmdDispatch(commandBuffer = p_cmdBuffer[0], 
              groupCountX = total_work_groups, 
              groupCountY = 1, 
              groupCountZ = 1
              )

vkEndCommandBuffer(commandBuffer = p_cmdBuffer[0])

# get device queue used by compute command
compute_queue = vkGetDeviceQueue(device = logical_device ,
                                 queueFamilyIndex = queue_family_compute_index,
                                 queueIndex = 0
                                 )

submitInfo = VkSubmitInfo(sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          commandBufferCount = 1,
                          pCommandBuffers = p_cmdBuffer,
                          )

logging.info('Computing command ...')
vkQueueSubmit(queue = compute_queue, 
              submitCount = 1, 
              pSubmits = [submitInfo], 
              fence = 0
              )

vkQueueWaitIdle(queue = compute_queue)
logging.info('Finished Computation')

_ppData = vkMapMemory(logical_device,device_memory,0,mem_size,0)
arr = np.frombuffer(_ppData,dtype=np.int32)
arr_out_final = arr[buffer_length:]

print('\n------ Results ------------\n')
print('**Input buffer values**')
print(arr[0:buffer_length])
print('\n**Initial output array**')
print(arr_out_init)
print('\n**Final output array**')
print(arr_out_final)
print('\n------ The End ------------')


# ************************************************************************************************
''' Console Output:
INFO:root:query physical devices, use the first device
DEBUG:root:query compute queue family
INFO:root:create logical device in the physical device
INFO:root:Allocate total 2*4*16384 bytes (_ppData) in gpu for two buffers
INFO:root:Make the GPU memory visible to CPU
INFO:root:Initialize the memory with random integer using _ppData, which is visible to CPU
** Assign random int between -1000 and 1000
** Numpy array:
[-838  235  822 ...    0    0    0]
** Buffer after assigning 100 to index 2:
Numpy array: array([-838,  235,  100, ...,    0,    0,    0])
100
** Buffer after assigning 4000 and 0 to in_buffer and out_buffer, respectively
Numpy array: array([4000, 4000, 4000, ...,    0,    0,    0])
4000
INFO:root:Create input and output buffers in gpu for use in compute shader
INFO:root:Load SPIR-V shader binary code
DEBUG:root:Create shader
INFO:root:Create shader stage and compute pipelines
INFO:root:Computing command ...
INFO:root:Finished Computation

------ Results ------------

**Input buffer values**
[4000 4000 4000 ... 4000 4000 4000]

**Initial output array**
[0 0 0 ... 0 0 0]

**Final output array**
[8000 8000 8000 ... 8000 8000 8000]

------ The End -----------
'''
# ************************************************************************************************
