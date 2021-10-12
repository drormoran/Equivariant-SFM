# Python Bundle Adjustment


## Conda envorinment
Use the <a href="https://github.com/drormoran/Equivariant-SFM/blob/main/environment.yml"> ESFM</a> environment.
```
conda activate ESFM
export PYBIND11_PYTHON_VERSION="3.8"
export PYTHON_VERSION="3.8"
```

## Folders
After set up, the order of the folders would be:
```
Equivariant-SFM
├── bundle_adjustment
│   ├── ceres-solver
│   │   ├── ceres-bin
│   │   |   └── lib
│   │   |       └── PyCeres.cpython-38-x86_64-linux-gnu.so
│   │   ├── ceres_python_bindings
│   │   |   └── python_bindings
│   │   |       └── custom_cpp_cost_functions.cpp
│   │   └── CMakeLists.txt
│   └── custom_cpp_cost_functions.cpp
├── code
├── datasets
├── results
```
## Set up
1. Download the <a href="http://ceres-solver.org/installation.html">Ceres-Solver</a> package to the bundle_adjustment folder:

```
cd bundle_adjustment
git clone https://ceres-solver.googlesource.com/ceres-solver
```


2. Clone the <a href="https://github.com/Edwinem/ceres_python_bindings">ceres_python_bindings</a> package inside the ceres-solver folder:

```
cd ceres-solver
git clone https://github.com/Edwinem/ceres_python_bindings
```


3. Copy the file "custom_cpp_cost_functions.cpp" and replace the file "ceres-solver/ceres_python_bindings/python_bindings/custom_cpp_cost_functions.cpp".
This file contains our projective and euclidean custom bundle adjustment functions.

```
cp ../custom_cpp_cost_functions.cpp ceres_python_bindings/python_bindings/custom_cpp_cost_functions.cpp
```

Next, you need to build ceres_python_bindings and ceres-solver and create a shared object file that python can call.
You can either continue with the instructions here or follow the instructions at the <a href="https://github.com/Edwinem/ceres_python_bindings">ceres_python_bindings</a> repository.

4. run:

```
cd ceres_python_bindings
git submodule init
git submodule update
```


5. Add to the end of the file ceres-solver/CMakeLists.txt the line: "include(ceres_python_bindings/AddToCeres.cmake)":

```
cd ..
echo "include(ceres_python_bindings/AddToCeres.cmake)" >> CMakeLists.txt
```


6. Inside ceres-solver folder run:


```
mkdir ceres-bin
cd ceres-bin
cmake ..
make -j8
make test
```

7. If everything worked you should see the following file:

```
bundle_adjustment/ceres-solver/ceres-bin/lib/PyCeres.cpython-38-x86_64-linux-gnu.so
```

8. If you want to use this bundle adjustment implementation for a different project make sure to add the path of the shared object to linux PATH (in our code this is done for you). In the python project this would be for example:

```
import sys
sys.path.append('../bundle_adjustment/ceres-solver/ceres-bin/lib/PyCeres.cpython-38-x86_64-linux-gnu.so')
import PyCeres
```


To see the usage of the PyCeres functions go to code/utils/ceres_utils and code/utils/ba_functions.
