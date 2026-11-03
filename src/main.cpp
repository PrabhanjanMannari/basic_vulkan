#include <iostream>
#include <stdexcept>
#include <cassert>
#include <array>
#include <vector>
#include <string_view>
#include <string>
#include <fstream>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vma/vk_mem_alloc.h>

constexpr int WIN_WIDTH{1280};
constexpr int WIN_HEIGHT{760};

#define NDEBUG 
constexpr void chk(VkResult result) {
#ifdef NDEBUG 
    assert(result == VK_SUCCESS);
#endif 
}

class VulkanApp {
public:
    GLFWwindow *m_win{nullptr};

    VkInstance m_instance{nullptr};

    VkPhysicalDevice m_physicalDevice{nullptr};
    uint32_t m_queueFamilyIdx{0};

    VkDevice m_device{nullptr};
    VkQueue m_deviceQueue{nullptr};

    VkSurfaceKHR m_surface{nullptr};
    VkSurfaceCapabilitiesKHR m_surfaceCaps{};

    struct Swapchain {
        VkSwapchainKHR handle{nullptr};
        VkExtent2D extent{};
        VkFormat format{};
        std::vector<VkImage> images{};
        std::vector<VkImageView> imageViews{};
    };
    Swapchain m_swapchain{};

    VkPipelineLayout m_graphicsPipelineLayout{nullptr};
    VkPipeline m_graphicsPipeline{nullptr};

    VkCommandPool m_commandPool{nullptr};
    VkCommandBuffer m_commandBuffer{nullptr};

    VkFence m_drawFence{nullptr};
    VkSemaphore m_acquireSemaphore{nullptr};
    std::vector<VkSemaphore> m_renderFinishedSemaphoreList{};

    inline static constexpr std::array<char const*, 1> m_reqdDevExtensions {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    VulkanApp(void) {
        init();
    }

    ~VulkanApp(void) {
        cleanup();
    }

    void initWindow(void) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_win = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "Window", nullptr, nullptr);
        if (m_win == nullptr) {
            glfwTerminate();
            throw std::runtime_error("Unable to create GLFW window");
        }
    }

    void createInstance(void) {
        VkApplicationInfo appInfo {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "How to Vulkan",
            .apiVersion = VK_API_VERSION_1_4,
        };

        uint32_t instanceExtensionCount{0};
        char const* const* instanceExtensions = glfwGetRequiredInstanceExtensions(&instanceExtensionCount);
        
        constexpr uint32_t LAYER_COUNT{1};
        constexpr std::array<char const*, LAYER_COUNT> instanceLayers{
            "VK_LAYER_KHRONOS_validation",
        };

        VkInstanceCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo, 
#ifdef NDEBUG //@TODO This works only for validation layers. Should change this when other layers are enabled 
            .enabledLayerCount = instanceLayers.size(),
            .ppEnabledLayerNames = instanceLayers.data(),
#endif 
            .enabledExtensionCount = instanceExtensionCount, 
            .ppEnabledExtensionNames = instanceExtensions,
        };

        chk(vkCreateInstance(&createInfo, nullptr, &m_instance));
    }

    void selectPhysicalDevice(void) {
        uint32_t deviceCount{0};
        chk(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr));
        std::vector<VkPhysicalDevice> physicalDeviceList{};
        physicalDeviceList.resize(deviceCount);
        chk(vkEnumeratePhysicalDevices(m_instance, &deviceCount, physicalDeviceList.data()));

        bool didSelectPhysicalDevice{false};
        for (uint32_t idx{0}; idx < deviceCount; ++idx) {
            const auto &physicalDevice{physicalDeviceList[idx]};

            VkPhysicalDeviceProperties2 devProps{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            vkGetPhysicalDeviceProperties2(physicalDevice, &devProps);

            std::vector<VkQueueFamilyProperties2> queueFamilyProperties{};
            
            if (devProps.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                uint32_t queueFamilyCount{0};
                vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &queueFamilyCount, nullptr);
                queueFamilyProperties.clear();
                queueFamilyProperties.resize(queueFamilyCount);
                for (auto &qProps : queueFamilyProperties) {
                    qProps.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
                }
                vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
                bool didSelectQueueFamily{false};
                for (uint32_t qIdx{0}; qIdx < queueFamilyCount; ++qIdx) {
                    const bool isQueueFamilyEligible{
                        checkQueueFamilySupportGraphicsAndPresentation(queueFamilyProperties, physicalDevice, qIdx)
                    };
                    if (isQueueFamilyEligible) {
                        // @TODO Breaking after one queue family satisfies the constraints 
                        didSelectQueueFamily = true;
                        m_queueFamilyIdx = qIdx;
                        break;
                    }
                }
                
                bool doesSupportReqdExtensions{false};
                uint32_t devExtensionCount{0};
                vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtensionCount, nullptr);
                std::vector<VkExtensionProperties> devExtensionProps(devExtensionCount);
                vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtensionCount, devExtensionProps.data());
                for (const auto &devExt : devExtensionProps) {
                    // @TODO Need to change this when other extensions are added 
                    if (std::string_view{devExt.extensionName} == std::string_view{m_reqdDevExtensions[0]}) {
                        doesSupportReqdExtensions = true;
                        // @TODO Don't break unless all reqd extensions are supported 
                        break;
                    }
                }

                bool doesSupportReqdFeatures{checkPhysicalDeviceFeatures(physicalDevice)};
                if (didSelectQueueFamily && doesSupportReqdExtensions && doesSupportReqdFeatures) {
                    didSelectPhysicalDevice = true;
                    m_physicalDevice = physicalDeviceList[idx];
                    std::cout << "Found discrete GPU with graphics, presentation and swapchain extension support\n";
                }
            }
        }

        // Select physical device at idx 0 and check if it supports graphics and presentation 
        if (!didSelectPhysicalDevice) {
            m_physicalDevice = physicalDeviceList[0];
            m_queueFamilyIdx = 0;
            std::cerr << "Selecting physical device at idx 0 and queue family at idx 0 without checking for presentation or graphics support\n";
        }
        chk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &m_surfaceCaps));
    }

    // @TODO Connect this with the features being enabled when creating a logical device 
    bool checkPhysicalDeviceFeatures(const VkPhysicalDevice &physicalDevice) {
        VkPhysicalDeviceVulkan11Features vk11Features {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        };
        VkPhysicalDeviceVulkan12Features vk12Features {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &vk11Features,
        };
        VkPhysicalDeviceVulkan13Features vk13Features {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &vk12Features,
        };
        VkPhysicalDeviceFeatures2 vkFeatures {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vk13Features,
            .features = VkPhysicalDeviceFeatures{},
        };

        vkGetPhysicalDeviceFeatures2(physicalDevice, &vkFeatures);

        bool areVk10FeaturesSupported{vkFeatures.features.samplerAnisotropy == VK_TRUE};
        bool areVk11FeaturesSupported{vk11Features.shaderDrawParameters == VK_TRUE};
        bool areVk12FeaturesSupported{
            vk12Features.descriptorIndexing                        && 
            vk12Features.shaderSampledImageArrayNonUniformIndexing && 
            vk12Features.descriptorBindingVariableDescriptorCount  && 
            vk12Features.runtimeDescriptorArray                    &&
            vk12Features.bufferDeviceAddress
        };
        bool areVk13FeaturesSupported{
            vk13Features.synchronization2 && vk13Features.dynamicRendering
        };

        return areVk10FeaturesSupported && areVk11FeaturesSupported && areVk12FeaturesSupported && areVk13FeaturesSupported;
    }

    bool checkQueueFamilySupportGraphicsAndPresentation(const std::vector<VkQueueFamilyProperties2> &queueFamilyProperties, 
                                                       const VkPhysicalDevice &physicalDevice,
                                                       const uint32_t qIdx) {
        const auto &qProps{queueFamilyProperties[qIdx]};
        const auto queueCount{qProps.queueFamilyProperties.queueCount};
        const auto qFlags{qProps.queueFamilyProperties.queueFlags};
        const bool doesSupportGraphics{(qFlags & VK_QUEUE_GRAPHICS_BIT) != 0};
        VkBool32 doesSupportPresentation{};

        chk(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, qIdx, m_surface, &doesSupportPresentation));
        return doesSupportGraphics && (doesSupportPresentation == VK_TRUE);
    }

    void createLogicalDevice(void) {
        VkPhysicalDeviceVulkan11Features vk11Features {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .shaderDrawParameters = true,
        };
        VkPhysicalDeviceVulkan12Features vk12Features {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &vk11Features,
            .descriptorIndexing                         = true, 
            .shaderSampledImageArrayNonUniformIndexing  = true, 
            .descriptorBindingVariableDescriptorCount   = true, 
            .runtimeDescriptorArray                     = true,
            .bufferDeviceAddress                        = true,
        };

        VkPhysicalDeviceVulkan13Features vk13Features {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &vk12Features,
            .synchronization2 = true, 
            .dynamicRendering = true,
        };

        VkPhysicalDeviceFeatures vk10Features{
            .samplerAnisotropy = VK_TRUE
        };

        const float qPriorities{1.0f};
        VkDeviceQueueCreateInfo queueCreateInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = m_queueFamilyIdx,
            .queueCount = 1,
            .pQueuePriorities = &qPriorities,
        };

        VkDeviceCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &vk13Features,
            .queueCreateInfoCount = 1, 
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<uint32_t>(m_reqdDevExtensions.size()),
            .ppEnabledExtensionNames = m_reqdDevExtensions.data(),
            .pEnabledFeatures = &vk10Features,
        };

        chk(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device));
        vkGetDeviceQueue(m_device, m_queueFamilyIdx, 0, &m_deviceQueue);
    }

    void setupVma(void) {
    }

    void createSurface(void) {
        chk(glfwCreateWindowSurface(m_instance, m_win, nullptr, &m_surface));
    }

    void createSwapchain(void) {
        if (m_surfaceCaps.currentExtent.width == 0xFFFFFFFF) {
            int width{}, height{};
            glfwGetWindowSize(m_win, &width, &height);
            m_swapchain.extent =  VkExtent2D {
                .width = static_cast<uint32_t>(width), 
                .height = static_cast<uint32_t>(height),
            };
        }
        else {
            m_swapchain.extent = m_surfaceCaps.currentExtent;
        }
        //@TODO Query surface capabilities, formats and present modes 
        m_swapchain.format = VkFormat{VK_FORMAT_B8G8R8A8_SRGB};
        VkSwapchainCreateInfoKHR createInfo {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = m_surface, 
            .minImageCount = m_surfaceCaps.minImageCount, 
            .imageFormat = m_swapchain.format, 
            .imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR, 
            .imageExtent = VkExtent2D{.width = m_swapchain.extent.width, .height = m_swapchain.extent.height},
            .imageArrayLayers = 1, 
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
            .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, 
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        };
        chk(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain.handle));

        uint32_t imageCount{0};
        chk(vkGetSwapchainImagesKHR(m_device, m_swapchain.handle, &imageCount, nullptr));
        m_swapchain.images.resize(imageCount);
        chk(vkGetSwapchainImagesKHR(m_device, m_swapchain.handle, &imageCount, m_swapchain.images.data()));
        m_swapchain.imageViews.resize(imageCount);
        for (size_t idx{0}; idx < imageCount; ++idx) {
            const auto &image{m_swapchain.images[idx]};
            VkImageViewCreateInfo imgViewCreateInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = m_swapchain.format,
                .components = {
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, 
                    .levelCount = 1,
                    .layerCount = 1,
                },
            };
            chk(vkCreateImageView(m_device, &imgViewCreateInfo, nullptr, &m_swapchain.imageViews[idx]));
        }
    }

    // @TODO Use std::byte instead?
    std::vector<char> readFile(const std::string &filename) {
        std::ifstream file{filename};
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open shader file at: " + filename);
        }

        std::vector<char> shaderBytecode{
            std::istreambuf_iterator<char>{file},
            std::istreambuf_iterator<char>{}
        };
        return shaderBytecode;
    }

    void createGraphicsPipeline(void) {
        /*
         * CREATE SHADER MODULE 
         */
        const std::vector<char> shaderBytecode{readFile("shaders/slang.spv")};
        VkShaderModuleCreateInfo shaderCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 
            .codeSize = shaderBytecode.size() * sizeof(char),
            .pCode = reinterpret_cast<const uint32_t*>(shaderBytecode.data()), 
        };
        VkShaderModule shaderModule{};
        vkCreateShaderModule(m_device, &shaderCreateInfo, nullptr, &shaderModule);
        VkPipelineShaderStageCreateInfo vertShaderCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shaderModule,
            .pName = "vertMain",
        };
        VkPipelineShaderStageCreateInfo fragShaderCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = shaderModule,
            .pName = "fragMain",
        };
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages {
            vertShaderCreateInfo,
            fragShaderCreateInfo,
        };

        /**
         * FIXED FUNCTION DESCRIPTION 
         */
        VkPipelineVertexInputStateCreateInfo vertexInputInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
        };
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        VkViewport viewport{
            .x = 0,
            .y = 0,
            .width  = static_cast<float>(m_swapchain.extent.width), 
            .height = static_cast<float>(m_swapchain.extent.height), 
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        VkRect2D scissor{
            .offset = VkOffset2D{0, 0},
            .extent = m_swapchain.extent,
        };
        VkPipelineViewportStateCreateInfo viewportInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, 
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };
        VkPipelineRasterizationStateCreateInfo rasterInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo multisampleInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };
        VkPipelineColorBlendAttachmentState blendAttachment {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo blendInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1, 
            .pAttachments = &blendAttachment,
        };
        std::array<VkDynamicState, 2> dynamicStates {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        VkPipelineDynamicStateCreateInfo dynamicInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, 
            .dynamicStateCount = dynamicStates.size(),
            .pDynamicStates = dynamicStates.data(),
        };

        VkPipelineRenderingCreateInfo renderInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1, 
            .pColorAttachmentFormats = &m_swapchain.format,
        };


        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0,
        };
        chk(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_graphicsPipelineLayout));

        VkGraphicsPipelineCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &renderInfo,
            .stageCount = 2,
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportInfo,
            .pRasterizationState = &rasterInfo, 
            .pMultisampleState = &multisampleInfo,
            .pColorBlendState = &blendInfo,
            .pDynamicState = &dynamicInfo,
            .layout = m_graphicsPipelineLayout,
            .renderPass = nullptr,
        };
        chk(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_graphicsPipeline));
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    void createCommandPool(void) {
        VkCommandPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = m_queueFamilyIdx,
        };
        chk(vkCreateCommandPool(m_device, &createInfo, nullptr, &m_commandPool));
    }

    void createCommandBuffer(void) {
        VkCommandBufferAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 
            .commandPool = m_commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        chk(vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer));
    }

    /**
     * @TODO Dont do this for one image at a time 
     */
    void transitionImageLayout(uint32_t              imageIdx,
                               VkPipelineStageFlags2 srcStageMask, 
                               VkPipelineStageFlags2 dstStageMask,
                               VkAccessFlags2        srcAccessMask, 
                               VkAccessFlags2        dstAccessMask,
                               VkImageLayout         oldLayout, 
                               VkImageLayout         newLayout) {
        VkImageMemoryBarrier2 barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask, 
            .oldLayout = oldLayout, 
            .newLayout = newLayout,
            .image = m_swapchain.images[imageIdx],
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, 
                .baseMipLevel = 0,
                .levelCount = 1, 
                .baseArrayLayer = 0, 
                .layerCount = 1,
            },
        };
        VkDependencyInfo depInfo {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1, 
            .pImageMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(m_commandBuffer, &depInfo);
    }
                            

    void recordCommandBuffer(uint32_t imageIdx) {
        VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        chk(vkBeginCommandBuffer(m_commandBuffer, &beginInfo));
        
        transitionImageLayout(
            imageIdx,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_NONE,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL
        );

        VkClearValue clearColor{
            .color = {0.0f, 0.0f, 0.0f, 1.0f},
        };
        VkRenderingAttachmentInfo attachmentInfo {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = m_swapchain.imageViews[imageIdx],
            .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, 
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clearColor,
        };
        
        VkRenderingInfo renderInfo {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {
                .offset = {0, 0},
                .extent = m_swapchain.extent,
            },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentInfo,
        };

        vkCmdBeginRendering(m_commandBuffer, &renderInfo);
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
        /**
         * @TODO: 
         * @TODO: Dynamic Rendering Viewport and Scissor
         */
        VkViewport viewport{
            .x = 0,
            .y = 0,
            .width  = static_cast<float>(m_swapchain.extent.width), 
            .height = static_cast<float>(m_swapchain.extent.height), 
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(m_commandBuffer, 0, 1, &viewport);
        VkRect2D scissor{
            .offset = VkOffset2D{0, 0},
            .extent = m_swapchain.extent,
        };
        vkCmdSetScissor(m_commandBuffer, 0, 1, &scissor);

        vkCmdDraw(m_commandBuffer, 3, 1, 0, 0);
        vkCmdEndRendering(m_commandBuffer);

        transitionImageLayout(
            imageIdx,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_2_NONE,
            VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        );
        chk(vkEndCommandBuffer(m_commandBuffer));
    }

    void createSyncPrimitives(void) {
        VkSemaphoreCreateInfo acquireImageInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        chk(vkCreateSemaphore(m_device, &acquireImageInfo, nullptr, &m_acquireSemaphore));

        VkSemaphoreCreateInfo renderFinishedInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        m_renderFinishedSemaphoreList.resize(m_swapchain.images.size());
        for (auto &semaphore : m_renderFinishedSemaphoreList) {
            chk(vkCreateSemaphore(m_device, &renderFinishedInfo, nullptr, &semaphore));
        }

        VkFenceCreateInfo fenceInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        chk(vkCreateFence(m_device, &fenceInfo, nullptr, &m_drawFence));
    }

    void drawFrame(void) {
        chk(vkWaitForFences(m_device, 1, &m_drawFence, VK_TRUE, UINT64_MAX));
        vkResetFences(m_device, 1, &m_drawFence);
        
        uint32_t curImgIdx{~0u};
        chk(vkAcquireNextImageKHR(m_device, m_swapchain.handle, UINT64_MAX, m_acquireSemaphore, VK_NULL_HANDLE, &curImgIdx));

        recordCommandBuffer(curImgIdx);

        VkPipelineStageFlags waitDstStageMask {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSubmitInfo submitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, 
            .waitSemaphoreCount = 1, 
            .pWaitSemaphores = &m_acquireSemaphore, 
            .pWaitDstStageMask = &waitDstStageMask, 
            .commandBufferCount = 1, 
            .pCommandBuffers = &m_commandBuffer, 
            .signalSemaphoreCount = 1, 
            .pSignalSemaphores = &m_renderFinishedSemaphoreList[curImgIdx],
        };
        chk(vkQueueSubmit(m_deviceQueue, 1, &submitInfo, m_drawFence));

        VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, 
            .waitSemaphoreCount = 1, 
            .pWaitSemaphores = &m_renderFinishedSemaphoreList[curImgIdx], 
            .swapchainCount = 1, 
            .pSwapchains = &m_swapchain.handle, 
            .pImageIndices = &curImgIdx,
        };
        chk(vkQueuePresentKHR(m_deviceQueue, &presentInfo));
    }

    void initVulkan(void) {
        createInstance();
        createSurface();
        selectPhysicalDevice();
        createLogicalDevice();
        setupVma();
        createSwapchain();
        createGraphicsPipeline();
        createCommandPool();
        createCommandBuffer();
        createSyncPrimitives();
    }

    void init(void) {
        initWindow();
        initVulkan();
    }

    void cleanup(void) {
        vkDeviceWaitIdle(m_device);
        vkDestroySemaphore(m_device, m_acquireSemaphore, nullptr);
        for (auto &semaphore : m_renderFinishedSemaphoreList) {
            vkDestroySemaphore(m_device, semaphore, nullptr);
        }
        vkDestroyFence(m_device, m_drawFence, nullptr);
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_graphicsPipelineLayout, nullptr);
        for (auto &imageView : m_swapchain.imageViews) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(m_device, m_swapchain.handle, nullptr);
        vkDestroyDevice(m_device, nullptr);
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyInstance(m_instance, nullptr);
        glfwDestroyWindow(m_win);
        glfwTerminate();
    }
};

int main() {
    
    VulkanApp app{};

    while (!glfwWindowShouldClose(app.m_win)) {
        glfwPollEvents();
        app.drawFrame();
    }

    return 0;
}
