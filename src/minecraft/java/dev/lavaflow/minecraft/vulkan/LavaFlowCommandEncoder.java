package dev.lavaflow.minecraft.vulkan;

import com.mojang.blaze3d.buffers.*;
import com.mojang.blaze3d.systems.*;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.textures.GpuTextureView;
import org.joml.Vector4fc;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.KHRDynamicRendering.vkCmdEndRenderingKHR;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK11.VK_ERROR_OUT_OF_POOL_MEMORY;

final class LavaFlowCommandEncoder implements CommandEncoderBackend {
    private static final int SUBMIT_SLOTS = 3;

    private final LavaFlowVulkanContext context;
    private final LavaFlowDevice device;
    private final SubmitSlot[] slots = new SubmitSlot[SUBMIT_SLOTS];
    private int slotIndex;
    private SubmitSlot slot;
    private LavaFlowTransientMemory transientMemory;
    private VkCommandBuffer commandBuffer;
    private boolean renderPassOpen;
    private long waitSemaphore;
    private long signalSemaphore;
    private LavaFlowRenderPass activeRenderPass;
    private long recordingSerial = 1;
    private long completedSerial;
    private boolean destroyed;
    private long readbackScratch;
    private int readbackScratchBytes;

    private static final class SubmitSlot {
        final VkCommandBuffer commandBuffer;
        final long fence;
        final LavaFlowTransientMemory transientMemory;
        LavaFlowDevice.SubmitBatch batch;
        long serial;
        boolean inFlight;

        SubmitSlot(VkCommandBuffer commandBuffer, long fence, LavaFlowTransientMemory transientMemory) {
            this.commandBuffer = commandBuffer;
            this.fence = fence;
            this.transientMemory = transientMemory;
        }
    }

    LavaFlowCommandEncoder(LavaFlowDevice device) {
        this.device = device;
        this.context = device.context();
        createSlots();
        prepareSlot(0);
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) throw new IllegalStateException(operation + " failed with VkResult " + result);
    }

    private void createSlots() {
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferAllocateInfo allocation = VkCommandBufferAllocateInfo.calloc(stack).sType$Default()
                    .commandPool(context.commandPool()).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(SUBMIT_SLOTS);
            PointerBuffer out = stack.mallocPointer(SUBMIT_SLOTS);
            check(vkAllocateCommandBuffers(context.device(), allocation, out), "vkAllocateCommandBuffers");
            VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack).sType$Default();
            LongBuffer fenceOut = stack.mallocLong(1);
            for (int i = 0; i < SUBMIT_SLOTS; i++) {
                check(vkCreateFence(context.device(), fenceInfo, null, fenceOut), "vkCreateFence(submit)");
                slots[i] = new SubmitSlot(new VkCommandBuffer(out.get(i), context.device()),
                        fenceOut.get(0), new LavaFlowTransientMemory(device, this));
            }
        }
    }

    private void prepareSlot(int index) {
        if (destroyed) throw new IllegalStateException("LavaFlow command encoder is destroyed");
        slotIndex = index;
        slot = slots[index];
        if (slot.inFlight) {
            check(vkWaitForFences(context.device(), slot.fence, true, 5_000_000_000L),
                    "vkWaitForFences(submit slot)");
            check(vkResetFences(context.device(), slot.fence), "vkResetFences(submit slot)");
            completedSerial = Math.max(completedSerial, slot.serial);
            slot.inFlight = false;
            device.completeSubmit(slot.batch);
            slot.batch = null;
        }
        slot.transientMemory.recycle();
        check(vkResetCommandBuffer(slot.commandBuffer, 0), "vkResetCommandBuffer");
        try (MemoryStack stack = stackPush()) {
            VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack).sType$Default()
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            check(vkBeginCommandBuffer(slot.commandBuffer, begin), "vkBeginCommandBuffer");
        }
        commandBuffer = slot.commandBuffer;
        transientMemory = slot.transientMemory;
    }

    VkCommandBuffer commandBuffer() { return commandBuffer; }
    LavaFlowDevice device() { return device; }
    void waitFor(long semaphore) {
        if (waitSemaphore != NULL) throw new IllegalStateException("Only one surface acquire is supported per submit");
        waitSemaphore = semaphore;
    }

    void signal(long semaphore) {
        if (signalSemaphore != NULL) throw new IllegalStateException("Only one surface present is supported per submit");
        signalSemaphore = semaphore;
    }

    @Override public void submit() {
        if (destroyed) throw new IllegalStateException("LavaFlow command encoder is destroyed");
        if (renderPassOpen) submitRenderPass();
        check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        try (MemoryStack stack = stackPush()) {
            VkSubmitInfo submit = VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(stack.pointers(commandBuffer));
            if (waitSemaphore != NULL) {
                submit.waitSemaphoreCount(1).pWaitSemaphores(stack.longs(waitSemaphore));
                // Mali GPU (TBDR) requires the acquire semaphore to be waited at
                // COLOR_ATTACHMENT_OUTPUT_BIT. Waiting only at TRANSFER_BIT leaves Mali's
                // tile buffer unflushed from prior frame work, causing flickering.
                submit.pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT));
            }
            if (signalSemaphore != NULL) submit.pSignalSemaphores(stack.longs(signalSemaphore));
            check(vkQueueSubmit(context.graphicsQueue(), submit, slot.fence), "vkQueueSubmit");
        }
        LavaFlowFrameStats.workSubmitted();
        slot.batch = device.detachSubmitBatch(transientMemory.retire());
        slot.serial = recordingSerial++;
        slot.inFlight = true;
        waitSemaphore = signalSemaphore = NULL;
        prepareSlot((slotIndex + 1) % SUBMIT_SLOTS);
    }

    /**
     * Returns a scratch allocation of at least {@code bytes}, reused across calls.
     *
     * <p>Host-visible device memory is typically write-combined, where scattered narrow reads are far
     * more expensive than one sequential bulk copy. Callers that need to inspect such memory copy the
     * whole range here first and read from the result, which is ordinary cached memory. Grows
     * monotonically so the render loop performs no allocation once warm.
     */
    long readbackScratch(int bytes) {
        if (bytes > readbackScratchBytes) {
            int capacity = Math.max(bytes, Math.max(readbackScratchBytes * 2, 4096));
            long grown = MemoryUtil.nmemAlignedAlloc(64, capacity);
            if (grown == NULL) throw new IllegalStateException("Out of memory growing readback scratch");
            if (readbackScratch != NULL) MemoryUtil.nmemAlignedFree(readbackScratch);
            readbackScratch = grown;
            readbackScratchBytes = capacity;
        }
        return readbackScratch;
    }

    void destroy() {
        if (destroyed) return;
        if (renderPassOpen) submitRenderPass();
        destroyed = true;
        if (readbackScratch != NULL) {
            MemoryUtil.nmemAlignedFree(readbackScratch);
            readbackScratch = NULL;
            readbackScratchBytes = 0;
        }
        for (SubmitSlot submitSlot : slots) {
            if (submitSlot == null) continue;
            if (submitSlot.inFlight) {
                check(vkWaitForFences(context.device(), submitSlot.fence, true, 5_000_000_000L),
                        "vkWaitForFences(destroy)");
                completedSerial = Math.max(completedSerial, submitSlot.serial);
                submitSlot.inFlight = false;
                device.completeSubmit(submitSlot.batch);
                submitSlot.batch = null;
            }
            submitSlot.transientMemory.destroy();
            vkDestroyFence(context.device(), submitSlot.fence, null);
        }
        for (SubmitSlot submitSlot : slots) {
            if (submitSlot != null) {
                vkFreeCommandBuffers(context.device(), context.commandPool(), submitSlot.commandBuffer);
            }
        }
        commandBuffer = null;
    }

    @Override public TransientMemory transientMemory() { return transientMemory; }
    @Override public RenderPassBackend createRenderPass(RenderPassDescriptor descriptor) {
        if (renderPassOpen) throw new IllegalStateException("A render pass is already open");
        renderPassOpen = true;
        activeRenderPass = new LavaFlowRenderPass(this, descriptor);
        return activeRenderPass;
    }
    @Override public void submitRenderPass() {
        if (renderPassOpen) {
            renderPassOpen = false;
            LavaFlowRenderPass pass = activeRenderPass;
            activeRenderPass = null;
            pass.end();
        }
    }

    void markRenderPassClosed() { renderPassOpen = false; }

    void transition(LavaFlowGpuTexture texture, int targetLayout) {
        int sourceLayout = texture.layout();
        if (sourceLayout == targetLayout) return;
        LavaFlowFrameStats.barrierRecorded();
        try (MemoryStack stack = stackPush()) {
            VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack).sType$Default()
                    .oldLayout(sourceLayout).newLayout(targetLayout)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .image(texture.handle());
            barrier.subresourceRange().aspectMask(LavaFlowVk.aspect(texture.getFormat())).baseMipLevel(0)
                    .levelCount(texture.getMipLevels()).baseArrayLayer(0).layerCount(texture.getDepthOrLayers());
            barrier.srcAccessMask(sourceAccessForLayout(sourceLayout));
            barrier.dstAccessMask(destinationAccessForLayout(targetLayout));
            vkCmdPipelineBarrier(commandBuffer, sourceStageForLayout(sourceLayout), destinationStageForLayout(targetLayout),
                    0, null, null, barrier);
            texture.layout(targetLayout);
        }
    }

    @Override public void clearColorTexture(GpuTexture gpuTexture, Vector4fc color) {
        LavaFlowGpuTexture texture = texture(gpuTexture); transition(texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        try (MemoryStack stack = stackPush()) {
            VkClearColorValue clear = VkClearColorValue.calloc(stack).float32(stack.floats(color.x(), color.y(), color.z(), color.w()));
            VkImageSubresourceRange range = VkImageSubresourceRange.calloc(stack).aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(texture.getMipLevels()).baseArrayLayer(0).layerCount(texture.getDepthOrLayers());
            vkCmdClearColorImage(commandBuffer, texture.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clear, range);
        }
        restoreSampledLayout(texture);
    }
    @Override public void clearColorAndDepthTextures(GpuTexture color, Vector4fc value, GpuTexture depth, double depthValue) {
        clearColorTexture(color, value); clearDepthTexture(depth, depthValue);
    }
    @Override public void clearColorAndDepthTextures(GpuTexture color, Vector4fc value, GpuTexture depth, double depthValue, int x, int y, int width, int height) {
        LavaFlowGpuTexture colorTexture = texture(color);
        LavaFlowGpuTexture depthTexture = texture(depth);
        if (width <= 0 || height <= 0) return;
        if (x < 0 || y < 0 || x + width > color.getWidth(0) || y + height > color.getHeight(0)) {
            throw new IllegalArgumentException("Clear rectangle exceeds texture bounds");
        }
        GpuTextureView colorView = device.createTextureView(color);
        GpuTextureView depthView = device.createTextureView(depth);
        try {
            RenderPassDescriptor descriptor = RenderPassDescriptor.create(() -> "LavaFlow partial color/depth clear")
                    .withColorAttachment(colorView)
                    .withDepthAttachment(depthView)
                    .withRenderArea(new RenderPass.RenderArea(0, 0, color.getWidth(0), color.getHeight(0)));
            createRenderPass(descriptor);
            // vkCmdClearAttachments is recorded directly rather than through a draw, so the pass has
            // to be begun explicitly.
            activeRenderPass.ensureBegun();
            try (MemoryStack stack = stackPush()) {
                VkClearRect.Buffer rect = VkClearRect.calloc(1, stack).baseArrayLayer(0).layerCount(1);
                rect.rect().offset().set(x, y);
                rect.rect().extent().set(width, height);

                VkClearAttachment.Buffer attachments = VkClearAttachment.calloc(2, stack);
                attachments.get(0).aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).colorAttachment(0);
                attachments.get(0).clearValue().color()
                        .float32(stack.floats(value.x(), value.y(), value.z(), value.w()));
                attachments.get(1).aspectMask(LavaFlowVk.aspect(depthTexture.getFormat()));
                attachments.get(1).clearValue().depthStencil().depth((float) depthValue).stencil(0);
                vkCmdClearAttachments(commandBuffer, attachments, rect);
            } finally {
                submitRenderPass();
            }
        } finally {
            depthView.close();
            colorView.close();
        }
    }
    @Override public void clearDepthTexture(GpuTexture gpuTexture, double value) {
        LavaFlowGpuTexture texture = texture(gpuTexture); transition(texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        try (MemoryStack stack = stackPush()) {
            VkClearDepthStencilValue clear = VkClearDepthStencilValue.calloc(stack).depth((float)value).stencil(0);
            VkImageSubresourceRange range = VkImageSubresourceRange.calloc(stack).aspectMask(LavaFlowVk.aspect(texture.getFormat()))
                    .baseMipLevel(0).levelCount(texture.getMipLevels()).baseArrayLayer(0).layerCount(texture.getDepthOrLayers());
            vkCmdClearDepthStencilImage(commandBuffer, texture.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clear, range);
        }
        restoreSampledLayout(texture);
    }

    @Override public void writeToBuffer(GpuBufferSlice destination, ByteBuffer source) {
        long size = source.remaining();
        if (size == 0) return;
        if (size > destination.length()) throw new IllegalArgumentException("Destination buffer slice is too small");
        GpuBufferSlice staging = transientMemory.uploadStaging(source, 1);
        copyToBuffer(staging, destination.slice(0, size));
    }
    @Override public void copyToBuffer(GpuBufferSlice source, GpuBufferSlice destination) {
        if (source.length() > destination.length()) throw new IllegalArgumentException("Destination buffer slice is too small");
        try (MemoryStack stack = stackPush()) {
            VkBufferCopy.Buffer copy = VkBufferCopy.calloc(1, stack).srcOffset(source.offset())
                    .dstOffset(destination.offset()).size(source.length());
            vkCmdCopyBuffer(commandBuffer, buffer(source).handle(), buffer(destination).handle(), copy);
            int usage = destination.buffer().usage();
            int destinationStages = bufferDestinationStages(usage);
            int destinationAccess = bufferDestinationAccess(usage);
            VkBufferMemoryBarrier.Buffer barrier = VkBufferMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(destinationAccess)
                    .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                    .buffer(buffer(destination).handle()).offset(destination.offset()).size(source.length());
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    destinationStages, 0, null, barrier, null);
        }
    }
    @Override public void writeToTexture(GpuTexture target, ByteBuffer source, int mip, int layer, int x, int y, int width, int height) {
        LavaFlowGpuBuffer staging = new LavaFlowGpuBuffer(device, GpuBuffer.USAGE_MAP_WRITE | GpuBuffer.USAGE_COPY_SRC, source.remaining());
        staging.write(0, source);
        copyBufferToTexture(staging.slice(), 0, 0, width, height, target, x, y, width, height, mip, layer);
        device.defer(staging);
    }
    @Override public void copyBufferToTexture(GpuBufferSlice source, int rowOffset, int imageOffset, int rowLength, int imageHeight, GpuTexture target, int x, int y, int width, int height, int mip, int layer) {
        LavaFlowGpuTexture texture = texture(target); transition(texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        long offset = source.offset() + (long)(rowOffset + imageOffset * rowLength) * target.getFormat().blockSize();
        try (MemoryStack stack = stackPush()) {
            VkBufferImageCopy.Buffer copy = VkBufferImageCopy.calloc(1, stack).bufferOffset(offset)
                    .bufferRowLength(rowLength).bufferImageHeight(imageHeight);
            copy.imageSubresource().aspectMask(LavaFlowVk.aspect(target.getFormat())).mipLevel(mip).baseArrayLayer(layer).layerCount(1);
            copy.imageOffset().set(x, y, 0); copy.imageExtent().set(width, height, 1);
            vkCmdCopyBufferToImage(commandBuffer, buffer(source).handle(), texture.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);
        }
        if ((target.usage() & GpuTexture.USAGE_TEXTURE_BINDING) != 0) {
            transition(texture, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
    @Override public void copyTextureToBuffer(GpuTexture source, GpuBuffer target, long offset, Runnable callback, int mip) {
        copyTextureToBuffer(source, target, offset, callback, mip, 0, 0, source.getWidth(mip), source.getHeight(mip));
    }
    @Override public void copyTextureToBuffer(GpuTexture source, GpuBuffer target, long offset, Runnable callback, int mip, int x, int y, int width, int height) {
        LavaFlowGpuTexture texture = texture(source); transition(texture, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        try (MemoryStack stack = stackPush()) {
            VkBufferImageCopy.Buffer copy = VkBufferImageCopy.calloc(1, stack).bufferOffset(offset);
            copy.imageSubresource().aspectMask(LavaFlowVk.aspect(source.getFormat())).mipLevel(mip).baseArrayLayer(0).layerCount(1);
            copy.imageOffset().set(x, y, 0); copy.imageExtent().set(width, height, 1);
            vkCmdCopyImageToBuffer(commandBuffer, texture.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    ((LavaFlowGpuBuffer)target).handle(), copy);
        }
        restoreSampledLayout(texture);
        if (callback != null) device.afterSubmit(callback);
    }
    @Override public void copyTextureToTexture(GpuTexture source, GpuTexture target, int mip, int targetX, int targetY, int sourceX, int sourceY, int width, int height) {
        LavaFlowGpuTexture src = texture(source), dst = texture(target);
        transition(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); transition(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        try (MemoryStack stack = stackPush()) {
            VkImageCopy.Buffer copy = VkImageCopy.calloc(1, stack);
            copy.srcSubresource().aspectMask(LavaFlowVk.aspect(source.getFormat())).mipLevel(mip).baseArrayLayer(0).layerCount(1);
            copy.dstSubresource().aspectMask(LavaFlowVk.aspect(target.getFormat())).mipLevel(mip).baseArrayLayer(0).layerCount(1);
            copy.srcOffset().set(sourceX, sourceY, 0); copy.dstOffset().set(targetX, targetY, 0); copy.extent().set(width, height, 1);
            vkCmdCopyImage(commandBuffer, src.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.handle(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);
        }
        restoreSampledLayout(src);
        restoreSampledLayout(dst);
    }
    @Override public GpuFence createFence() { return new LavaFlowFence(this, recordingSerial); }

    boolean awaitSubmitCompletion(long serial, long timeout) {
        if (serial <= completedSerial) return true;
        if (serial >= recordingSerial) {
            if (timeout == 0) return false;
            throw new IllegalStateException("Cannot wait for work that has not been submitted");
        }
        for (SubmitSlot submitSlot : slots) {
            if (submitSlot.inFlight && submitSlot.serial == serial) {
                int result = vkWaitForFences(context.device(), submitSlot.fence, true, timeout);
                if (result == VK_TIMEOUT) return false;
                check(result, "vkWaitForFences(logical fence)");
                completedSerial = Math.max(completedSerial, serial);
                return true;
            }
        }
        return serial <= completedSerial;
    }
    @Override public void writeTimestamp(GpuQueryPool pool, int index) {
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ((LavaFlowQueryPool)pool).handle(), index);
    }

    private static LavaFlowGpuBuffer buffer(GpuBufferSlice slice) { return (LavaFlowGpuBuffer)slice.buffer(); }
    private static LavaFlowGpuTexture texture(GpuTexture texture) { return (LavaFlowGpuTexture)texture; }

    private void restoreSampledLayout(LavaFlowGpuTexture texture) {
        if ((texture.usage() & GpuTexture.USAGE_TEXTURE_BINDING) != 0) {
            transition(texture, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    private static int sourceStageForLayout(int layout) {
        return switch (layout) {
            case VK_IMAGE_LAYOUT_UNDEFINED -> VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ->
                    VK_PIPELINE_STAGE_TRANSFER_BIT;
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ->
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL ->
                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            default -> VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        };
    }

    private static int destinationStageForLayout(int layout) {
        return switch (layout) {
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ->
                    VK_PIPELINE_STAGE_TRANSFER_BIT;
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL -> VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ->
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL ->
                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            default -> VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        };
    }

    private static int sourceAccessForLayout(int layout) {
        return switch (layout) {
            case VK_IMAGE_LAYOUT_UNDEFINED -> 0;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL -> VK_ACCESS_TRANSFER_READ_BIT;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL -> VK_ACCESS_TRANSFER_WRITE_BIT;
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ->
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ->
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL ->
                    VK_ACCESS_SHADER_READ_BIT;
            default -> VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        };
    }

    private static int destinationAccessForLayout(int layout) {
        return switch (layout) {
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL -> VK_ACCESS_TRANSFER_READ_BIT;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL -> VK_ACCESS_TRANSFER_WRITE_BIT;
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ->
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ->
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL ->
                    VK_ACCESS_SHADER_READ_BIT;
            default -> VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        };
    }

    private static int bufferDestinationStages(int usage) {
        int stages = 0;
        if ((usage & (GpuBuffer.USAGE_VERTEX | GpuBuffer.USAGE_INDEX)) != 0) {
            stages |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        }
        if ((usage & (GpuBuffer.USAGE_UNIFORM | GpuBuffer.USAGE_UNIFORM_TEXEL_BUFFER)) != 0) {
            stages |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        if ((usage & GpuBuffer.USAGE_INDIRECT_PARAMETERS) != 0) stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        if ((usage & (GpuBuffer.USAGE_COPY_SRC | GpuBuffer.USAGE_COPY_DST)) != 0) stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
        if ((usage & GpuBuffer.USAGE_MAP_READ) != 0) stages |= VK_PIPELINE_STAGE_HOST_BIT;
        return stages == 0 ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : stages;
    }

    private static int bufferDestinationAccess(int usage) {
        int access = 0;
        if ((usage & GpuBuffer.USAGE_VERTEX) != 0) access |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        if ((usage & GpuBuffer.USAGE_INDEX) != 0) access |= VK_ACCESS_INDEX_READ_BIT;
        if ((usage & GpuBuffer.USAGE_UNIFORM) != 0) access |= VK_ACCESS_UNIFORM_READ_BIT;
        if ((usage & GpuBuffer.USAGE_UNIFORM_TEXEL_BUFFER) != 0) access |= VK_ACCESS_SHADER_READ_BIT;
        if ((usage & GpuBuffer.USAGE_INDIRECT_PARAMETERS) != 0) access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        if ((usage & GpuBuffer.USAGE_COPY_SRC) != 0) access |= VK_ACCESS_TRANSFER_READ_BIT;
        if ((usage & GpuBuffer.USAGE_COPY_DST) != 0) access |= VK_ACCESS_TRANSFER_WRITE_BIT;
        if ((usage & GpuBuffer.USAGE_MAP_READ) != 0) access |= VK_ACCESS_HOST_READ_BIT;
        return access == 0 ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : access;
    }
}
