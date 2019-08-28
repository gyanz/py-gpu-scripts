'''
Part 1 - Python script to allocate numpy array in GPU. 

Based on Neil Henning's following blog post, describing sample C++ code for 
GPU computation using (low level) Vulkan API.

http://www.duskborn.com/posts/a-simple-vulkan-compute-example/

'''
import ctypes
import random
import numpy as np
from vulkan import *
from vulkan._vulkancache import ffi

# ************************************************************************************************
# Boiler plate code to select appropiate GPU device, create logical device, compute queue,
# allocate memory for data, etc.

appInfo = VkApplicationInfo(sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            pApplicationName = "Allocate array in GPU",
                            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
                            pEngineName = "Gyan Basyal",
                            engineVersion = VK_MAKE_VERSION(1, 0, 0),
                            apiVersion = VK_API_VERSION_1_0
                            )

createInfo = VkInstanceCreateInfo(sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 flags = 0,
                                 pApplicationInfo = appInfo,
                                 )

instance = vkCreateInstance(createInfo, None)

# query physical devices
physical_devices = vkEnumeratePhysicalDevices(instance)

physical_devices_features = [vkGetPhysicalDeviceFeatures(physical_device) for physical_device in physical_devices]

physical_devices_properties = [vkGetPhysicalDeviceProperties(physical_device) for physical_device in physical_devices]

# select the first device in the list
physical_device = physical_devices[0]

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
                                   pEnabledFeatures = physical_devices_features[0],
                                   flags= 0
                                   )

# create logical device in the physical device 
logical_device = vkCreateDevice(physical_device, device_create, None)

# Allocate 4*16384 bytes buffer memory (_ppData) in gpu

buffer_length = 16384
mem_size = buffer_length * ctypes.sizeof(ctypes.c_int)  
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
device_memory = vkAllocateMemory(logical_device,memAllocInfo,0) 

_ppData = vkMapMemory(logical_device,device_memory,0,mem_size,0)

# ************************************************************************************************
# Assign random integer value to gpu buffer allocate above (using cffi) 

ppData = ffi.from_buffer("int[]",_ppData)
print('** Assign random int between -1000 and 1000')
for x in range(buffer_length):
    ppData[x] = random.randint(-1000,1000)

# ************************************************************************************************
# Further modify gpu buffer values (using numpy)

arr = np.frombuffer(_ppData,dtype=np.int32)
print('** Numpy array:')
print(arr)

arr[2] = 100
print('** Buffer after assigning 100 to index 2:')
print('Numpy array: %r'%arr)
print(ppData[2])
if not arr[2] == ppData[2]:
    print('Error')

arr[:] = 2000
print('** Buffer after assigning 2000 to each element')
print('Numpy array: %r'%arr)
print(ppData[2])