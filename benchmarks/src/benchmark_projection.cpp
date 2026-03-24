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
auto get_projection_subset(const Mesh& mesh, const std::size_t level)
{
	using mesh_id_t = typename Mesh::mesh_id_t;
	
	return samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells_and_ghosts][level - 1]).on(level - 1);
}

template<samurai::mesh_like Mesh, typename Index>
std::array<std::size_t, 1ULL << (Mesh::dim - 1)> get_src_offsets(const Mesh& src_mesh, const std::size_t level, const std::size_t i_start, const Index& index)
{
	constexpr std::size_t dim = Mesh::dim;
	
	std::array<std::size_t, 1ULL << (dim - 1)> src_offsets;
	
	if constexpr (dim == 1)
	{
		src_offsets[0] = samurai::memory_offset(src_mesh, {level + 1, 2 * i_start});
	}
	else
	{
		std::size_t ind = 0;
		samurai::static_nested_loop<dim - 1, 0, 2>(
			[&](const auto& stencil)
			{
				auto new_index     = 2 * index + stencil;
				src_offsets[ind++] = samurai::memory_offset(src_mesh, {level + 1, 2 * i_start, new_index});
			});
	}
	
	return src_offsets;
}

////////////////////////////////////////////////////////////////////////
//// projection
////////////////////////////////////////////////////////////////////////

template<std::size_t dim, std::size_t n_comp>
inline void projection_samurai(benchmark::State& state)
{
	auto mesh = init_mesh<dim>(1.e-4, 0, 1);

    auto src = samurai::make_vector_field<double, n_comp>("src", mesh);
    auto dst = samurai::make_vector_field<double, n_comp>("dst", mesh);	
    
    for (auto _ : state)
    {
		for (std::size_t level = mesh.max_level(); level >= mesh.min_level(); --level)
		{
			const auto projection_subset = get_projection_subset(mesh, level);
			
			projection_subset([&](const auto& interval, const auto& index)
			{
				const auto& src_offsets = get_src_offsets(mesh, level, interval.start, index);
				const auto  dst_offsets = samurai::memory_offset(mesh, {level, interval.start, index});
				
				const auto* src_data = src.data();
				auto* dst_data       = dst.data();
				
				constexpr double inv = 1.0 / static_cast<double>(1ULL << dim);
				
				for (std::size_t i=0; i!=interval.size(); ++i)
				{
					std::array<double, n_comp> sum;
					sum.fill(0);
					
					for (const auto& src_offset : src_offsets)
					{
						for (std::size_t n = 0; n != n_comp; ++n)
						{
							assert((src_offset + 2*i) * n_comp + n < mesh.nb_cells()*n_comp);
							
							sum[n] += src_data[(src_offset + 2*i) * n_comp + n] + src_data[(src_offset + 2*i + 1) * n_comp + n];
						}
					}
					for (std::size_t n = 0; n != n_comp; ++n)
					{
						assert((dst_offsets + i) * n_comp + n < mesh.nb_cells()*n_comp);
						
						dst_data[(dst_offsets + i) * n_comp + n] = sum[n] * inv;
					}
				}
			});
		}
	}
}

BENCHMARK(projection_samurai<1, 1>);
//~ BENCHMARK(projection_samurai<2, 1>);
//~ BENCHMARK(projection_samurai<2, 2>);
//~ BENCHMARK(projection_samurai<3, 1>);
//~ BENCHMARK(projection_samurai<3, 3>);

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
