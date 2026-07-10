## Building on Linux

To build on Linux, you will need the following:
- A compiler toolchain as specified [here](https://devguide.python.org/getting-started/setup-building/index.html#build-dependencies).
- Vulkan development headers. The safest way to get these is from the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#linux), but most package repositories will also have them.

### Building with pip

If you just want to use the package, run `pip install git+https://github.com/softcookiepp/pytorch_dlprim.git`

### Building the wheel

Simply run the following to build the wheel for this package:
```
python -m build --wheel
```
This will generate a wheel inside the `dist` folder for the python version you used.

### Building for development purposes

Ensure your python virtual environment is active if applicable, then execute the following
```
git clone https://github.com/softcookiepp/pytorch_dlprim.git --recurse-submodules
cd pytorch_dlprim
mkdir build
cd build
cmake .. && make
```
Then during testing, ensure `PYTHONPATH` includes the build directory.

## Building on Windows

No idea yet. It would be greatly appreciated if someone familiar with Vulkan development on Windows could help!
