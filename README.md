# Pytorch Vulkan backend based on ported dlprimitives, ported CLBlast, and other odds and ends

Out-of-tree backend for pytorch supporting any GPU with a Vulkan driver!

Originally based [pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim), but will likely have even more functionality in the future.

Currently pytorch version 2.4.0 is required, but this will change in the future.

# Installation

See [README-build.md](README-build.md)
    
## How to Use

Example:
```
import pytorch_vk
import torch
a = torch.randn(256).to("vk:0")
```

## Known Issues

1. Many operators not implemented and there may be fallbacks to CPU. Sometimes it is minor but sometimes it may hamper the performance, some may just fail
2. When you save/restore the model move it to CPU. Currently there is an issue with loading back saved state dictionary if it was saved from vk device
3. Efficiency is currently abysmal due to GPU under-utilization. Optimization has been put on the backburner in order to focus on making sure all the computations pass correctness tests, but hopefully this will soon change.

## To-do list

In order of priority:
- Refactor codebases of upstream dependencies. They have far too many goofy abstractions that only exist because of limitations inherent to OpenCL and/or the ability to support multiple GPGPU APIs, and this makes maintenance and optimization difficult.
- Optimize as much as possible.
- Implement missing operators.

## `pytorch_vk` specific API

Some functions specific to `pytorch_vk`. When using pytorch >= 2.4 they are accessible from `torch.vk` and `pytorch_vk`, for 1.13 you must use `pytorch_vk`

- `torch.vk.empty_cache()`: Same as `torch.cuda.empty_cache()` remove all cached GPU memory
- `torch.vk.synchronize(device=None)`: synchronize all operations queue on the device, if device is None - all of them same as `torch.cuda.synchonize`
- `torch.vk.manual_seed_all(seed)`: reset random number generator state. `torch.manual_seed` - it calls automatically for pytorch >= 2.4. Note for pytorch 1.13 you must call `pytorch_vk.manual_seed_all`


