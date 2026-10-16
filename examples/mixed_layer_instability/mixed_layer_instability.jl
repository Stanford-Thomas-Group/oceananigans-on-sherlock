# Mixed layer instability (MLI) on two distributed GPUs
#
# A horizontal density front is confined to a mixed layer and left to adjust under
# rotation. The front goes baroclinically unstable (Boccaletti et al. 2007, Fox-Kemper
# et al. 2008), forming submesoscale eddies that slump the front and restratify the
# mixed layer, competing against a surface cooling flux that destratifies it.
#
# Stratification and the front are set independently via a two-component linear
# equation of state: temperature alone sets the background vertical stratification,
# salinity alone carries the horizontal front. The domain is partitioned in the
# periodic x-direction and run across two GPUs via MPI.

using MPI
MPI.Init()

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations: Distributed, Partition, @root
using Oceananigans.TurbulenceClosures: TKEDissipationVerticalDiffusivity
using CUDA
using Printf
using CairoMakie

arch = Distributed(GPU(); partition=Partition(x=2))
rank = arch.local_rank

@info @sprintf("Rank %d on %s: CUDA_VISIBLE_DEVICES=%s, using %s (uuid %s)",
               rank, gethostname(), get(ENV, "CUDA_VISIBLE_DEVICES", "unset"), CUDA.device(), CUDA.uuid(CUDA.device()))

# Grid: periodic along the front, bounded across it and in depth.
# Halo is the minimum required by 5th-order WENO everywhere -- halo width sets the
# per-step MPI communication volume between the two ranks, so it isn't padded.
# (4, 4, 3) rather than (3, 3, 3): WENOVectorInvariant's vorticity reconstruction uses
# a VelocityStencil, which needs one extra halo point in the horizontal directions
# beyond a bare order-5 WENO reconstruction.
Nx, Ny, Nz = 192, 192, 48
Lx, Ly, Lz = 10kilometers, 10kilometers, 150meters

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz),
                        x=(0, Lx), y=(-Ly/2, Ly/2), z=(-Lz, 0),
                        halo=(4, 4, 3),
                        topology=(Periodic, Bounded, Bounded))

# Surface cooling: a destabilizing heat flux that competes with MLI restratification.
Q  = 300    # cooling, W m⁻²
ρ₀ = 1026   # kg m⁻³
cᴾ = 3991   # J (kg K)⁻¹
Jᵀ = Q / (ρ₀ * cᴾ)

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Jᵀ))

equation_of_state = LinearEquationOfState(thermal_expansion=2e-4, haline_contraction=7.8e-4)

# Largest Δt the time step wizard is allowed to reach (set below) -- the free surface's
# barotropic substeps are sized for this value so it stays stable even as Δt grows from
# its small initial value toward this cap.
max_Δt = 5minutes

model = HydrostaticFreeSurfaceModel(grid;
                                     free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=max_Δt),
                                     coriolis = FPlane(f=1e-4),
                                     buoyancy = SeawaterBuoyancy(; equation_of_state),
                                     tracers = (:T, :S),  # :e, :ϵ are added automatically by the closure
                                     closure = TKEDissipationVerticalDiffusivity(),
                                     momentum_advection = WENOVectorInvariant(order=5),
                                     tracer_advection = WENO(order=5),
                                     boundary_conditions = (T=T_bcs,))

# Initial conditions: T(z) alone sets the background stratification N², S(x,y,z) alone
# carries the horizontal front, confined to the mixed layer depth. No initial velocity --
# geostrophic adjustment and a small noise seed grow the instability from rest.
g = model.buoyancy.formulation.gravitational_acceleration
α = equation_of_state.thermal_expansion

N²   = 1e-5          # background stratification, s⁻²
dTdz = N² / (g * α)

Δy_front = 2kilometers
ΔS       = 0.8
mld      = 75meters
noise    = 1e-3

ramp(y, Δy) = min(max(0, y / Δy + 1/2), 1)
depth_mask(z, h) = min(max(1 + z / h, 0), 1)

Tᵢ(x, y, z) = 20 + dTdz * z + noise * randn()
Sᵢ(x, y, z) = 35 + ΔS * ramp(y, Δy_front) * depth_mask(z, mld) + noise * randn()

set!(model, T=Tᵢ, S=Sᵢ, e=1e-6)

simulation = Simulation(model, Δt=30seconds, stop_time=12hours)
conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.3, max_Δt=max_Δt)

wall_clock = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.velocities
    @printf("rank %d, iter: %d, time: %s, wall time: %s, max|u|: (%.2e, %.2e, %.2e) m/s, next Δt: %s\n",
            rank, iteration(sim), prettytime(sim), prettytime(1e-9 * (time_ns() - wall_clock[])),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), prettytime(sim.Δt))
    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# Each rank writes its own surface-only output file (JLD2Writer automatically appends
# "_rank$(rank)" for a Distributed architecture); we stitch the pieces back into a
# single global field in post-processing below, following the standard Oceananigans
# distributed convention of file-per-rank output.
u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))

outputs = (; T=model.tracers.T, S=model.tracers.S, ζ)

simulation.output_writers[:surface] = JLD2Writer(model, outputs;
                                                  filename = "mli_surface.jld2",
                                                  indices = (:, :, grid.Nz),
                                                  schedule = TimeInterval(15minutes),
                                                  overwrite_files = true)

@root @info "Running MLI simulation on $(MPI.Comm_size(MPI.COMM_WORLD)) ranks..."

run!(simulation)

MPI.Barrier(MPI.COMM_WORLD)

# Stitch the per-rank surface fields into a single global field and render the movie.
# Only rank 0 does this -- @root also barriers the other ranks so MPI doesn't tear down
# while rank 0 is still rendering.
@root begin
    nranks = MPI.Comm_size(MPI.COMM_WORLD)

    T_fts = [FieldTimeSeries("mli_surface_rank$(r).jld2", "T") for r in 0:nranks-1]
    S_fts = [FieldTimeSeries("mli_surface_rank$(r).jld2", "S") for r in 0:nranks-1]
    ζ_fts = [FieldTimeSeries("mli_surface_rank$(r).jld2", "ζ") for r in 0:nranks-1]

    # Ranks are ordered left-to-right by their true global x-position (each rank's local
    # grid retains its actual position in the domain), then concatenated along x.
    order = sortperm([xnodes(fts.grid, Center())[1] for fts in T_fts])

    times = T_fts[1].times
    Nt = length(times)

    stitch(fts, n) = vcat((Array(interior(fts[r], :, :, 1, n)) for r in order)...)

    T_global = [stitch(T_fts, n) for n in 1:Nt]
    S_global = [stitch(S_fts, n) for n in 1:Nt]
    ζ_global = [stitch(ζ_fts, n) for n in 1:Nt]

    x = 1e-3 .* vcat((xnodes(T_fts[r].grid, Center()) for r in order)...)
    y = 1e-3 .* ynodes(T_fts[order[1]].grid, Center())

    n = Observable(1)
    S_n = @lift S_global[$n]
    ζ_n = @lift ζ_global[$n]
    title = @lift "Mixed layer instability, t = " * prettytime(times[$n])

    fig = Figure(size=(1000, 500))

    axS = Axis(fig[2, 1], xlabel="x (km)", ylabel="y (km)", title="Surface salinity (g/kg)", aspect=1)
    hmS = heatmap!(axS, x, y, S_n, colormap=:haline)
    Colorbar(fig[2, 0], hmS, flipaxis=false)

    axζ = Axis(fig[2, 2], xlabel="x (km)", ylabel="y (km)", title="Surface vorticity (s⁻¹)", aspect=1)
    hmζ = heatmap!(axζ, x, y, ζ_n, colormap=:balance, colorrange=(-2e-4, 2e-4))
    Colorbar(fig[2, 3], hmζ)

    Label(fig[1, 1:2], title, fontsize=20, tellwidth=false)

    CairoMakie.record(fig, "mixed_layer_instability.mp4", 1:Nt, framerate=8) do i
        n[] = i
    end

    @info "Wrote mixed_layer_instability.mp4"
end
