# Oceananigans On Sherlock
A collection of tips and example projects for running Oceananigans on Stanford's Sherlock cluster

# This repository

- Contains ``Project.toml``, ``Manifest.toml`` and ``LocalPreferences.toml`` set up to run Oceananigans on _Sherlock_.
- The Oceananigans github repository contains many examples. This repository contains a script to download them so you can run them yourself.
    - For example to download the internal wave example, from the examples directory run
        ```bash
        $ ./download_example internal_wave
        ```
    - This will create an ``internal_wave`` directory with the ``internal_wave.jl`` script.

# Set up

- We require CUDA to run on Sherlock's GPUs.
- MPI is optional.
    - It is required for distributed simulations.
    - There is a substantial overhead and it can be finicky.
        - For example, the Sherlock version of OpenMPI is quite old and incompatible with the Julia NCDatasets package. Although Oceananigans supports writing output to netcdf via NCDatasets, this is not available to us for distributed computations. Not that this is a particularly big loss, JLD2 is the recommended output format in any case.
- Refer to the [general Julia advice](https://github.com/Stanford-Thomas-Group/julia-best-practices) for instructions on how to set up CUDA and MPI in a new project.
- The ``Project.toml`` and ``LocalPreferences.toml`` are already set up for _Sherlock_ so you should be able to simply load the openmpi module, set the environment variables and start a Julia instance
    ```bash
    $ module load openmpi/5.0.5
    $ export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
    $ julia --project
    ```
    and then instantiate the project. In **Package Mode** run 
    ```julia
    (oceananigans-on-sherlock) pkg> instantiate
    ```

# Examples from the Oceananigans GitHub

The Oceananigans GitHub contains the following examples (as of the `main` branch, 2026-10-16). "Runnable" means every package the script uses is listed in this project's `Project.toml` (or a Julia standard library).

| Example | Runnable? | Missing packages |
|---|---|---|
| `baroclinic_adjustment` | ✅ | |
| `convecting_plankton` | ✅ | |
| `global_wind_driven_gyres` | ❌ | NCDatasets, ConservativeRegridding |
| `horizontal_convection` | ✅ | |
| `hydrostatic_lock_exchange` | ✅ | |
| `internal_tide` | ✅ | |
| `internal_wave` | ✅ | |
| `kelvin_helmholtz_instability` | ✅ | |
| `langmuir_turbulence` | ✅ | |
| `ocean_wind_mixing_and_convection` | ✅ | |
| `one_dimensional_diffusion` | ✅ | |
| `polar_vortex_crystal` | ✅ | |
| `shallow_water_Bickley_jet` | ❌ | NCDatasets |
| `single_column_parameter_estimation` | ❌ | Enzyme, Reactant, Optim |
| `spherical_baroclinic_instability` | ✅ | |
| `tilted_bottom_boundary_layer` | ❌ | NCDatasets |
| `two_dimensional_turbulence` | ✅ | |

- NCDatasets is intentionally not installed here rather than simply missing — see the note above about its incompatibility with Sherlock's OpenMPI version. The examples requiring NCDatasets should be runnable but will require you to reset the project without the MPI dependency. 
- `ConservativeRegridding` is missing because the only example that requires it also requires `NCDatasets`.
- `Enzyme`, `Reactant`, `Optim` are missing because they are only required by one example and are heavy builds.

# Additional examples

## Mixed Layer Instability
- None of the above examples actually require MPI - they all run on a single CPU or GPU.
- Here we set up a simple distributed simulation running on 2 GPUs.
- We also have the ``sbatch`` script for submitting the job on _Sherlock_.
