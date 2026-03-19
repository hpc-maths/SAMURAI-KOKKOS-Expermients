#include <fmt/format.h>

#include <benchmark/benchmark.h>

#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/subset/nary_set_operator.hpp>
#include <samurai/uniform_mesh.hpp>

#include <Kokkos_Core.hpp>

#include <samurai_kokkos_all_offsets_environment.hpp>
#include <samurai_kokkos_environment.hpp>
#include <samurai_kokkos_scope.hpp>


////////////////////////////////////////////////////////////////////////
//// Helper functions
////////////////////////////////////////////////////////////////////////

template <std::size_t dim>
auto init_uniform_mesh()
{
    std::size_t start_level = 8;

    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
    min_corner.fill(0);
    max_corner.fill(1);
    const samurai::Box<double, dim> box(min_corner, max_corner);
    return samurai::UniformMesh<samurai::UniformConfig<dim, 1>>(box, start_level);
}

template <std::size_t dim>
auto init_mesh(double eps, std::size_t direction, std::size_t nb)
{
    std::size_t min_level   = 4;
    std::size_t start_level = 8;
    std::size_t max_level   = (dim == 2) ? 14 : 8;
    std::size_t jump        = max_level - start_level;

    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
    min_corner.fill(0);
    max_corner.fill(1);
    const samurai::Box<double, dim> box(min_corner, max_corner);
    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(start_level).max_stencil_size(2).disable_minimal_ghost_width();
    auto mesh   = samurai::mra::make_mesh(box, config);
    auto u      = samurai::make_scalar_field<double>("u", mesh);

    auto init_fct = [&](auto& cell)
    {
        auto center = cell.center();
        u[cell]     = 0;
        for (std::size_t i = 1; i <= nb; ++i)
        {
            u[cell] += tanh(1000 * std::abs(center[direction] - static_cast<double>(i) / static_cast<double>(nb + 1)));
        }
        u[cell] -= static_cast<double>(nb);
    };

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               init_fct(cell);
                           });

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(eps);
    MRadaptation(mra_config);

    using cl_type = std::decay_t<decltype(mesh)>::cl_type;
    while (jump > 0)
    {
        cl_type cl;
        for_each_interval(mesh,
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              samurai::static_nested_loop<dim - 1, 0, 2>(
                                  [&](const auto& stencil)
                                  {
                                      auto new_index = 2 * index + stencil;
                                      cl[level + 1][new_index].add_interval(i << 1);
                                  });
                          });
        config.max_level()++;
        mesh = {cl, config};

        u.resize();
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   init_fct(cell);
                               });
        MRadaptation(mra_config);
        jump--;
    }
    samurai::save(std::filesystem::current_path(), fmt::format("initial_mesh_{}_{}_{}", eps, direction, nb), mesh);
    return mesh;
}

template<samurai::mesh_like Mesh>
auto get_subset(const Mesh& mesh, const std::size_t level)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    
    return samurai::intersection(
        mesh[mesh_id_t::all_cells][level],
        samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2])).on(level);    
}

////////////////////////////////////////////////////////////////////////
//// axpy on the whole mesh
////////////////////////////////////////////////////////////////////////

inline void axpy_samurai(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);

    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);

    const double alpha = 2.;

    for (auto _ : state)
    {
        samurai::for_each_interval(mesh, [&](const auto level, const auto& interval, const auto& index)
        {
            const auto offset = samurai::memory_offset(mesh, {level, interval.start, index});
            for (size_t i=0; i!=interval.size(); ++i)
            {
                b[i + offset] += alpha*a[i + offset];
            }
        });
    }
}

inline void axpy_samurai_stl(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);

    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);

    const double alpha = 2.;

    for (auto _ : state)
    {
        samurai::for_each_interval(mesh, [&](const auto level, const auto& interval, const auto& index)
        {
            const auto offset = samurai::memory_offset(mesh, {level, interval.start, index});
            
            std::transform(a.data() + offset, a.data() + offset + interval.size(), b.data() + offset, b.data() + offset, [alpha](const double ai, const double bi)
            {
                return alpha*ai + bi;
            });
        });
    }
}

inline void axpy_full(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);

    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);

    const Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_a(a.data(), mesh.nb_cells());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_b(b.data(), mesh.nb_cells());

    // Create device views
    auto device_a = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_a);
    auto device_b = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_b);

    const double alpha = 2.;

    for (auto _ : state)
    {   
        Kokkos::parallel_for("axpy", mesh.nb_cells(), KOKKOS_LAMBDA (const int i)
        {
            device_b(i) += alpha * device_a(i);
        });
        Kokkos::fence();
    }
}

inline void axpy_all_offsets(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);
    
    SamuraiKokkosAllOffsetsEnvironment& env = Scope<SamuraiKokkosAllOffsetsEnvironment>::getContex();
    
    env.clear();
    env.reserve(mesh.nb_cells());
    
    samurai::for_each_interval(mesh, [&](const size_t level, const auto& interval, const auto& index)
    {
        const int offset = int(samurai::memory_offset(mesh, {level, interval.start, index}));

        for (int i=0; i!=int(interval.size()); ++i)
        {
            env.add_offset(offset + i);
        }
    });
    
    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);
    
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_a(a.data(), mesh.nb_cells());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_b(b.data(), mesh.nb_cells());
    
    auto device_a = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_a);
    auto device_b = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_b);
    
    const double alpha = 2.;

    for (auto _ : state)
    {
		// step 1. fill the offsets
		env.clear();
		samurai::for_each_interval(mesh, [&](const size_t level, const auto& interval, const auto& index)
		{
			const int offset = int(samurai::memory_offset(mesh, {level, interval.start, index}));
	
			for (int i=0; i!=int(interval.size()); ++i)
			{
				env.add_offset(offset + i);
			}
		});
		// step 2. copy inputs host -> device
		env.copy_data_to_host();
		// step 3. perform axpy
		const auto device_offsets = env.get_device_offsets();
		
		Kokkos::parallel_for("axpy", env.size(), KOKKOS_LAMBDA (const int i)
        {
            device_b(device_offsets[i]) += alpha*device_a(device_offsets[i]);
        });
	}
}

inline void axpy_offsets_and_sizes(benchmark::State& state)
{
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = Kokkos::TeamPolicy<>::member_type;

    auto mesh = init_mesh<2>(1.e-4, 0, 1);
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    
    std::size_t nb_xintervals = 0;
    samurai::for_each_level(mesh, [&](const size_t level)
    {
        nb_xintervals += mesh[mesh_id_t::cells][level][0].size();
    });
    
    SamuraiKokkosEnvironment& env = Scope<SamuraiKokkosEnvironment>::getContex();
    
    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);
    
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_a(a.data(), mesh.nb_cells());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_b(b.data(), mesh.nb_cells());
    
    auto device_a = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_a);
    auto device_b = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_b);
    
    const double alpha = 2.;

    for (auto _ : state)
    {
		// step 1. fill the offsets
		samurai::for_each_interval(mesh, [&](const size_t level, const auto& interval, const auto& index)
		{
			const int offset = int(samurai::memory_offset(mesh, {level, interval.start, index}));
			env.add_offset_and_interval_size(offset, interval.size());
		});
		// step 2. copy inputs host -> device
		env.copy_data_to_host();
		// step 3. perform axpy
		const auto device_offsets        = env.get_device_offsets();
		const auto device_interval_sizes = env.get_device_interval_sizes();
		
		Kokkos::parallel_for("axpy_outer", team_policy(nb_xintervals, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& member)
        {
            const auto offset = device_offsets(member.league_rank());
            const auto size   = device_interval_sizes(member.league_rank());
                        
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, size), [&](const int i)
            {
                const int ii = i + offset;

                device_b(ii) += alpha*device_a(ii);
            });
		});
	}
}

BENCHMARK(axpy_samurai);
BENCHMARK(axpy_samurai_stl);
BENCHMARK(axpy_full);
BENCHMARK(axpy_all_offsets);
BENCHMARK(axpy_offsets_and_sizes);

////////////////////////////////////////////////////////////////////////
//// axpy on subset
////////////////////////////////////////////////////////////////////////

inline void subset_axpy_samurai(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);

    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);

    const double alpha = 2.;

    for (auto _ : state)
    {
		for (std::size_t level = ((mesh.min_level() > 0) ? mesh.min_level() - 1 : 0); level < mesh.max_level(); ++level)
		{
			const auto subset = get_subset(mesh, level);
		
			subset([&](const auto& interval, const auto& index)
			{
				const auto offset = samurai::memory_offset(mesh, {level, interval.start, index});
				for (size_t i=0; i!=interval.size(); ++i)
				{
					b[i + offset] += alpha*a[i + offset];
				}
			});
		}
    }
}

inline void subset_axpy_samurai_stl(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);

    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);

    const double alpha = 2.;

    for (auto _ : state)
    {
		for (std::size_t level = ((mesh.min_level() > 0) ? mesh.min_level() - 1 : 0); level < mesh.max_level(); ++level)
		{
			const auto subset = get_subset(mesh, level);
		
			subset([&](const auto& interval, const auto& index)
			{
				const auto offset = samurai::memory_offset(mesh, {level, interval.start, index});
				
				std::transform(a.data() + offset, a.data() + offset + interval.size(), b.data() + offset, b.data() + offset, [alpha](const double ai, const double bi)
				{
					return alpha*ai + bi;
				});
			});
		}
    }
}

inline void subset_axpy_full(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);

    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);

    const Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_a(a.data(), mesh.nb_cells());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_b(b.data(), mesh.nb_cells());
    
    Kokkos::View<bool*, Kokkos::HostSpace> host_mask("mask", mesh.nb_cells());

    // Create device views
    auto device_a = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_a);
    auto device_b = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_b);
    
    auto device_mask = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_mask);

    const double alpha = 2.;

    for (auto _ : state)
    {   
		// set 1 fill the mask
		Kokkos::deep_copy(host_mask, false);
		for (std::size_t level = ((mesh.min_level() > 0) ? mesh.min_level() - 1 : 0); level < mesh.max_level(); ++level)
		{
			const auto subset = get_subset(mesh, level);
		
			subset([&](const auto& interval, const auto& index)
			{
				const auto offset = samurai::memory_offset(mesh, {level, interval.start, index});
				
				for (int i=0; i!=int(interval.size()); ++i)
				{
					host_mask(offset + i) = true;
				}
			});
		}
		Kokkos::deep_copy(device_mask, host_mask);
		
        Kokkos::parallel_for("axpy", mesh.nb_cells(), KOKKOS_LAMBDA (const int i)
        {
            device_b(i) += device_mask(i) * alpha * device_a(i);
        });
        Kokkos::fence();
    }
}

inline void subset_axpy_all_offsets(benchmark::State& state)
{
    auto mesh = init_mesh<2>(1.e-4, 0, 1);
    
    SamuraiKokkosAllOffsetsEnvironment& env = Scope<SamuraiKokkosAllOffsetsEnvironment>::getContex();
    
    env.reserve(mesh.nb_cells());
    
    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);
    
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_a(a.data(), mesh.nb_cells());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_b(b.data(), mesh.nb_cells());
    
    auto device_a = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_a);
    auto device_b = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_b);
    
    const double alpha = 2.;

    for (auto _ : state)
    {
		// step 1. fill the offsets
		env.clear();
		for (std::size_t level = ((mesh.min_level() > 0) ? mesh.min_level() - 1 : 0); level < mesh.max_level(); ++level)
		{
			const auto subset = get_subset(mesh, level);
			
			subset([&](const auto& interval, const auto& index)
			{
				const int offset = int(samurai::memory_offset(mesh, {level, interval.start, index}));
		
				for (int i=0; i!=int(interval.size()); ++i)
				{
					env.add_offset(offset + i);
				}
			});
		}
		// step 2. copy inputs host -> device
		env.copy_data_to_host();
		// step 3. perform subset_axpy
		const auto device_offsets = env.get_device_offsets();
		
		Kokkos::parallel_for("subset_axpy", env.size(), KOKKOS_LAMBDA (const int i)
        {
            device_b(device_offsets[i]) += alpha*device_a(device_offsets[i]);
        });
	}
}

inline void subset_axpy_offsets_and_sizes(benchmark::State& state)
{
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = Kokkos::TeamPolicy<>::member_type;

    auto mesh = init_mesh<2>(1.e-4, 0, 1);
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    
    std::size_t nb_xintervals = 0;
    samurai::for_each_level(mesh, [&](const size_t level)
    {
        nb_xintervals += mesh[mesh_id_t::cells][level][0].size();
    });
    
    SamuraiKokkosEnvironment& env = Scope<SamuraiKokkosEnvironment>::getContex();
    
    env.reserve(nb_xintervals);
    
    auto a = samurai::make_scalar_field("a", mesh);
    auto b = samurai::make_scalar_field("b", mesh);
    
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_a(a.data(), mesh.nb_cells());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_b(b.data(), mesh.nb_cells());
    
    auto device_a = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_a);
    auto device_b = create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_b);
    
    const double alpha = 2.;

    for (auto _ : state)
    {
		// step 1. fill the offsets
		env.clear();
		for (std::size_t level = ((mesh.min_level() > 0) ? mesh.min_level() - 1 : 0); level < mesh.max_level(); ++level)
		{
			const auto subset = get_subset(mesh, level);
			
			subset([&](const auto& interval, const auto& index)
			{
				const int offset = int(samurai::memory_offset(mesh, {level, interval.start, index}));
				env.add_offset_and_interval_size(offset, interval.size());
			});
		}
		// step 2. copy inputs host -> device
		env.copy_data_to_host();
		// step 3. perform subset_axpy
		const auto device_offsets        = env.get_device_offsets();
		const auto device_interval_sizes = env.get_device_interval_sizes();
		
		Kokkos::parallel_for("subset_axpy_outer", team_policy(nb_xintervals, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& member)
        {
            const auto offset = device_offsets(member.league_rank());
            const auto size   = device_interval_sizes(member.league_rank());
                        
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, size), [&](const int i)
            {
                const int ii = i + offset;

                device_b(ii) += alpha*device_a(ii);
            });
		});
	}
}

BENCHMARK(subset_axpy_samurai);
BENCHMARK(subset_axpy_samurai_stl);
BENCHMARK(subset_axpy_full);
BENCHMARK(subset_axpy_all_offsets);
BENCHMARK(subset_axpy_offsets_and_sizes);

////////////////////////////////////////////////////////////////////////
//// main
////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) 
{
  Kokkos::initialize(argc, argv);
  {
	fmt::println("Kokkos default execution space : {}", Kokkos::DefaultExecutionSpace::name());
	
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) { return 1; }
	
	SamuraiKokkosEnvironment samuraiKokkosEnv;
	SamuraiKokkosAllOffsetsEnvironment samuraiKokkosAllOffsetsEnv;
	
	Scope<SamuraiKokkosEnvironment> env1(samuraiKokkosEnv);
	Scope<SamuraiKokkosAllOffsetsEnvironment> env2(samuraiKokkosAllOffsetsEnv);
	
	::benchmark::RunSpecifiedBenchmarks();
  }
  Kokkos::finalize();
  
  return EXIT_SUCCESS;
}
