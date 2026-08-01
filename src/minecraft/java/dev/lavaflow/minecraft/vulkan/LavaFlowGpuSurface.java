package dev.lavaflow.minecraft.vulkan;

import com.mojang.blaze3d.systems.*;
import com.mojang.blaze3d.textures.GpuTextureView;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK10.*;

/** LavaFlow-owned swapchain with explicit acquire, blit, and present synchronization. */
final class LavaFlowGpuSurface implements GpuSurfaceBackend {
    private static final int NO_IMAGE = -1;
    private static final int ACQUIRE_SEMAPHORES = 3;

    private final LavaFlowVulkanContext context;
    private final long surface;
    private final Set<GpuSurface.PresentMode> supportedPresentModes;
    private final long[] acquireSemaphores = new long[ACQUIRE_SEMAPHORES];
    private int acquireSemaphoreIndex = ACQUIRE_SEMAPHORES - 1;

    private long swapchain;
    private long[] images = new long[0];
    private int[] imageLayouts = new int[0];
    private long[] presentSemaphores = new long[0];
    private int width;
    private int height;
    private int currentImage = NO_IMAGE;
    private boolean suboptimal;
    private boolean outOfDate;
    private boolean closed;

    LavaFlowGpuSurface(LavaFlowDevice device, long window) {
        context = device.context();
        surface = context.surface();
        supportedPresentModes = Collections.unmodifiableSet(queryPresentModes());
        createAcquireSemaphores();
    }

    private static void check(int result, String operation) {
        if (result < 0) throw new IllegalStateException(operation + " failed with VkResult " + result);
    }

    private Set<GpuSurface.PresentMode> queryPresentModes() {
        try (MemoryStack stack = stackPush()) {
            IntBuffer count = stack.ints(0);
            check(vkGetPhysicalDeviceSurfacePresentModesKHR(context.physicalDevice(), surface, count, null),
                    "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
            IntBuffer modes = stack.mallocInt(count.get(0));
            check(vkGetPhysicalDeviceSurfacePresentModesKHR(context.physicalDevice(), surface, count, modes),
                    "vkGetPhysicalDeviceSurfacePresentModesKHR");
            EnumSet<GpuSurface.PresentMode> result = EnumSet.noneOf(GpuSurface.PresentMode.class);
            for (int i = 0; i < modes.capacity(); i++) {
                switch (modes.get(i)) {
                    case VK_PRESENT_MODE_IMMEDIATE_KHR -> result.add(GpuSurface.PresentMode.IMMEDIATE);
                    case VK_PRESENT_MODE_MAILBOX_KHR -> result.add(GpuSurface.PresentMode.MAILBOX);
                    case VK_PRESENT_MODE_FIFO_KHR -> result.add(GpuSurface.PresentMode.FIFO);
                    case VK_PRESENT_MODE_FIFO_RELAXED_KHR -> result.add(GpuSurface.PresentMode.FIFO_RELAXED);
                    default -> { }
                }
            }
            return result;
        }
    }

    private void createAcquireSemaphores() {
        try (MemoryStack stack = stackPush()) {
            VkSemaphoreCreateInfo info = VkSemaphoreCreateInfo.calloc(stack).sType$Default();
            LongBuffer out = stack.mallocLong(1);
            for (int i = 0; i < acquireSemaphores.length; i++) {
                check(vkCreateSemaphore(context.device(), info, null, out),
                        "vkCreateSemaphore(acquire)");
                acquireSemaphores[i] = out.get(0);
            }
        }
    }

    @Override
    public void configure(GpuSurface.Configuration configuration) throws SurfaceException {
        ensureOpen();
        try (MemoryStack stack = stackPush()) {
            VkSurfaceCapabilitiesKHR capabilities = VkSurfaceCapabilitiesKHR.calloc(stack);
            check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.physicalDevice(), surface, capabilities),
                    "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
            int requestedWidth;
            int requestedHeight;
            if (capabilities.currentExtent().width() != -1) {
                requestedWidth = capabilities.currentExtent().width();
                requestedHeight = capabilities.currentExtent().height();
            } else {
                requestedWidth = Math.clamp(configuration.width(),
                        capabilities.minImageExtent().width(), capabilities.maxImageExtent().width());
                requestedHeight = Math.clamp(configuration.height(),
                        capabilities.minImageExtent().height(), capabilities.maxImageExtent().height());
            }

            // Keep the old swapchain alive until after the new one is created so the
            // presentation engine can hand off smoothly.  Destroying it first (and passing
            // NULL as oldSwapchain) forces a momentary black frame on every resize/reconfigure.
            long previousSwapchain = swapchain;
            if (previousSwapchain != NULL) {
                // Wait for all in-flight work referencing the old images to finish before we
                // retire those images, then release the per-image present semaphores.
                vkDeviceWaitIdle(context.device());
                for (long semaphore : presentSemaphores)
                    if (semaphore != NULL) vkDestroySemaphore(context.device(), semaphore, null);
            }
            swapchain = NULL;
            images = new long[0];
            imageLayouts = new int[0];
            presentSemaphores = new long[0];
            currentImage = NO_IMAGE;

            int[] format = chooseSurfaceFormat(stack);
            int imageCount = Math.max(3, capabilities.minImageCount());
            if (capabilities.maxImageCount() > 0) imageCount = Math.min(imageCount, capabilities.maxImageCount());
            int presentMode = supportedPresentModes.contains(configuration.presentMode())
                    ? toVk(configuration.presentMode()) : VK_PRESENT_MODE_FIFO_KHR;

            VkSwapchainCreateInfoKHR info = VkSwapchainCreateInfoKHR.calloc(stack).sType$Default()
                    .surface(surface).minImageCount(imageCount).imageFormat(format[0]).imageColorSpace(format[1])
                    .imageExtent(VkExtent2D.calloc(stack).set(requestedWidth, requestedHeight))
                    .imageArrayLayers(1).imageUsage(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .preTransform(capabilities.currentTransform())
                    .compositeAlpha(chooseCompositeAlpha(capabilities.supportedCompositeAlpha()))
                    .presentMode(presentMode).clipped(true).oldSwapchain(previousSwapchain);
            if (context.graphicsFamily() == context.presentFamily()) {
                info.imageSharingMode(VK_SHARING_MODE_EXCLUSIVE);
            } else {
                info.imageSharingMode(VK_SHARING_MODE_CONCURRENT)
                        .pQueueFamilyIndices(stack.ints(context.graphicsFamily(), context.presentFamily()));
            }

            LongBuffer out = stack.mallocLong(1);
            check(vkCreateSwapchainKHR(context.device(), info, null, out), "vkCreateSwapchainKHR");
            swapchain = out.get(0);

            // Old swapchain is now retired; destroy it now that the new one is live.
            if (previousSwapchain != NULL) {
                vkDestroySwapchainKHR(context.device(), previousSwapchain, null);
            }

            IntBuffer count = stack.ints(0);
            check(vkGetSwapchainImagesKHR(context.device(), swapchain, count, null),
                    "vkGetSwapchainImagesKHR(count)");
            LongBuffer handles = stack.mallocLong(count.get(0));
            check(vkGetSwapchainImagesKHR(context.device(), swapchain, count, handles),
                    "vkGetSwapchainImagesKHR");
            images = new long[count.get(0)];
            imageLayouts = new int[images.length];
            presentSemaphores = new long[images.length];
            for (int i = 0; i < images.length; i++) images[i] = handles.get(i);

            VkSemaphoreCreateInfo semaphoreInfo = VkSemaphoreCreateInfo.calloc(stack).sType$Default();
            for (int i = 0; i < presentSemaphores.length; i++) {
                check(vkCreateSemaphore(context.device(), semaphoreInfo, null, out), "vkCreateSemaphore(present)");
                presentSemaphores[i] = out.get(0);
            }
            width = requestedWidth;
            height = requestedHeight;
            currentImage = NO_IMAGE;
            suboptimal = false;
            outOfDate = false;
        } catch (RuntimeException exception) {
            suboptimal = outOfDate = true;
            throw new SurfaceException(exception);
        }
    }

    private int[] chooseSurfaceFormat(MemoryStack stack) {
        IntBuffer count = stack.ints(0);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(context.physicalDevice(), surface, count, null),
                "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
        VkSurfaceFormatKHR.Buffer formats = VkSurfaceFormatKHR.malloc(count.get(0));
        try {
            check(vkGetPhysicalDeviceSurfaceFormatsKHR(context.physicalDevice(), surface, count, formats),
                    "vkGetPhysicalDeviceSurfaceFormatsKHR");
            for (int i = 0; i < formats.capacity(); i++) {
                VkSurfaceFormatKHR candidate = formats.get(i);
                if (candidate.colorSpace() == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
                        && (candidate.format() == VK_FORMAT_R8G8B8A8_UNORM
                        || candidate.format() == VK_FORMAT_B8G8R8A8_UNORM)) {
                    return new int[]{candidate.format(), candidate.colorSpace()};
                }
            }
            VkSurfaceFormatKHR fallback = formats.get(0);
            return new int[]{fallback.format(), fallback.colorSpace()};
        } finally {
            formats.free();
        }
    }

    private static int chooseCompositeAlpha(int supported) {
        int[] candidates = {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
        for (int candidate : candidates) if ((supported & candidate) != 0) return candidate;
        throw new IllegalStateException("Surface supports no composite alpha mode");
    }

    private static int toVk(GpuSurface.PresentMode mode) {
        return switch (mode) {
            case IMMEDIATE -> VK_PRESENT_MODE_IMMEDIATE_KHR;
            case MAILBOX -> VK_PRESENT_MODE_MAILBOX_KHR;
            case FIFO -> VK_PRESENT_MODE_FIFO_KHR;
            case FIFO_RELAXED -> VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        };
    }

    @Override public boolean isSuboptimal() { return suboptimal; }

    @Override
    public void acquireNextTexture() throws SurfaceException {
        ensureConfigured();
        if (currentImage != NO_IMAGE) throw new IllegalStateException("A swapchain image is already acquired");
        try (MemoryStack stack = stackPush()) {
            IntBuffer index = stack.ints(NO_IMAGE);
            acquireSemaphoreIndex = (acquireSemaphoreIndex + 1) % acquireSemaphores.length;
            int result = vkAcquireNextImageKHR(context.device(), swapchain, 5_000_000_000L,
                    acquireSemaphores[acquireSemaphoreIndex], NULL, index);
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                suboptimal = outOfDate = true;
                throw new SurfaceException("Swapchain became out of date while acquiring an image");
            }
            if (result == VK_SUBOPTIMAL_KHR) suboptimal = true;
            else check(result, "vkAcquireNextImageKHR");
            currentImage = index.get(0);
        } catch (SurfaceException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            throw new SurfaceException(exception);
        }
    }

    @Override
    public void blitFromTexture(CommandEncoderBackend backend, GpuTextureView sourceView) {
        ensureConfigured();
        if (currentImage == NO_IMAGE) throw new IllegalStateException("No swapchain image has been acquired");
        LavaFlowCommandEncoder encoder = (LavaFlowCommandEncoder)backend;
        encoder.waitFor(acquireSemaphores[acquireSemaphoreIndex]);
        LavaFlowGpuTexture source = (LavaFlowGpuTexture)sourceView.texture();
        encoder.transition(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        long destination = images[currentImage];
        try (MemoryStack stack = stackPush()) {
            VkImageMemoryBarrier.Buffer toTransfer = VkImageMemoryBarrier.calloc(1, stack).sType$Default()
                    .oldLayout(imageLayouts[currentImage]).newLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(destination).srcAccessMask(0).dstAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT);
            toTransfer.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1)
                    .baseArrayLayer(0).layerCount(1);
            // When the image was previously presented (PRESENT_SRC_KHR), use
            // COLOR_ATTACHMENT_OUTPUT_BIT as the source stage.  This matches the stage at
            // which the acquire semaphore is waited (see LavaFlowCommandEncoder.submit) and
            // forces Mali's TBDR to flush its L1 tile cache before the blit begins.
            // TOP_OF_PIPE_BIT is sufficient only for the very first use (UNDEFINED layout)
            // where there is no prior GPU work referencing the image.
            int toTransferSrcStage = (imageLayouts[currentImage] == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
                    ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                    : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            vkCmdPipelineBarrier(encoder.commandBuffer(), toTransferSrcStage,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, null, toTransfer);

            VkImageBlit.Buffer blit = VkImageBlit.calloc(1, stack);
            int blitWidth = Math.min(width, sourceView.getWidth(0));
            int blitHeight = Math.min(height, sourceView.getHeight(0));
            blit.srcSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(sourceView.baseMipLevel())
                    .baseArrayLayer(0).layerCount(1);
            blit.srcOffsets(0).set(0, 0, 0);
            blit.srcOffsets(1).set(blitWidth, blitHeight, 1);
            blit.dstSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
            blit.dstOffsets(0).set(0, blitHeight, 0);
            blit.dstOffsets(1).set(blitWidth, 0, 1);
            vkCmdBlitImage(encoder.commandBuffer(), source.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit, VK_FILTER_NEAREST);

            VkImageMemoryBarrier.Buffer toPresent = VkImageMemoryBarrier.calloc(1, stack).sType$Default()
                    .oldLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(destination).srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(0);
            toPresent.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(1)
                    .baseArrayLayer(0).layerCount(1);
            vkCmdPipelineBarrier(encoder.commandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, null, null, toPresent);
        }
        imageLayouts[currentImage] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        encoder.signal(presentSemaphores[currentImage]);
    }

    @Override
    public void present() {
        ensureConfigured();
        if (currentImage == NO_IMAGE) throw new IllegalStateException("No swapchain image has been acquired");
        int image = currentImage;
        currentImage = NO_IMAGE;
        try (MemoryStack stack = stackPush()) {
            VkPresentInfoKHR info = VkPresentInfoKHR.calloc(stack).sType$Default()
                    .pWaitSemaphores(stack.longs(presentSemaphores[image]))
                    .swapchainCount(1)
                    .pSwapchains(stack.longs(swapchain)).pImageIndices(stack.ints(image));
            int result = vkQueuePresentKHR(context.presentQueue(), info);
            if (result == VK_ERROR_OUT_OF_DATE_KHR) { suboptimal = outOfDate = true; return; }
            if (result == VK_SUBOPTIMAL_KHR) suboptimal = true;
            else check(result, "vkQueuePresentKHR");
        }
    }

    @Override public Collection<GpuSurface.PresentMode> supportedPresentModes() { return supportedPresentModes; }

    private void ensureOpen() { if (closed) throw new IllegalStateException("Surface is closed"); }

    private void ensureConfigured() {
        ensureOpen();
        if (swapchain == NULL || outOfDate) throw new IllegalStateException("Swapchain is not configured or is out of date");
    }

    private void destroySwapchain() {
        if (swapchain == NULL) return;
        vkDeviceWaitIdle(context.device());
        for (long semaphore : presentSemaphores)
            if (semaphore != NULL) vkDestroySemaphore(context.device(), semaphore, null);
        vkDestroySwapchainKHR(context.device(), swapchain, null);
        swapchain = NULL;
        images = new long[0];
        imageLayouts = new int[0];
        presentSemaphores = new long[0];
        currentImage = NO_IMAGE;
    }

    @Override
    public void close() {
        if (closed) return;
        destroySwapchain();
        for (int i = 0; i < acquireSemaphores.length; i++) {
            if (acquireSemaphores[i] != NULL) {
                vkDestroySemaphore(context.device(), acquireSemaphores[i], null);
                acquireSemaphores[i] = NULL;
            }
        }
        closed = true;
    }
}
