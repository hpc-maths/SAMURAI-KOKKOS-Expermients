#include <samurai/box.hpp> 
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/schemes/fv.hpp>

#include <Kokkos_Core.hpp>

//~ std::vector<int> offset;
//~ std::vector<std::pair<int,int>> offset_size;

int main(int argc, char** argv)
{
	Kokkos::initialize(argc, argv);
	
	using Scalar = double;
	
	constexpr size_t dim = 2;
	
	const samurai::Box<Scalar, dim> box({0, 0}, {1, 1});

	auto config = samurai::mesh_config<dim>().min_level(0).max_level(4).max_stencil_size(2);
	auto mesh   = samurai::mra::make_mesh(box, config);
	
	//~ using mesh_id_t = typename decltype(mesh)::mesh_id_t;
	
	Kokkos::View<double*> a("a", mesh.nb_cells());
	Kokkos::View<double*> b("b", mesh.nb_cells());
	Kokkos::View<double*> c("c", mesh.nb_cells());
	
	Kokkos::parallel_for("vector_add", mesh.nb_cells(), KOKKOS_LAMBDA (const int i) 
	{
		a(i) = 1; 
		b(i) = 2;
	});
	
	using team_policy = Kokkos::TeamPolicy<>;
  using member_type = Kokkos::TeamPolicy<>::member_type;
	
	samurai::for_each_interval(mesh, [&](const size_t level, const auto& interval, const auto& index)
	{
		const int offset = int(samurai::memory_offset(mesh, {level, interval.start, index}));
		
		Kokkos::parallel_for("vector_add", team_policy(int(interval.size()), Kokkos::AUTO), KOKKOS_LAMBDA (const member_type& member)
		{
			const int i = member.league_rank();
			c(i + offset) = a(i + offset) + b(i + offset);
		}); 
	}); 
	
	//// 	version naive with more memory overhead
	//Kokkos::parallel_for("tuc", offset.size(), KOKKOS_LAMBDA(const int i)
	//{
	//	c(i) = a(i) + b(i);
	//});
	//// 	version with less memory overhead
	//Kokkos::parallel_for("tuc", offset_size.size(), KOKKOS_LAMBDA(const std::pair<int, int> i_size)
	//{
	//	const auto& [i, size] = i_size;
	//	
	//	for (int j=0; j!=size; ++j)
	//	{
	//		c(i + j) = a(i + j) + b(i + j);
	//	}
	//});
	//
	//Kokkos::parallel_for("tuc", team_policy(offset_and_size.size(), Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& member)
	//{
	//	const int i  = member.league_rank();
	//	const auto& [offset, size] = offset_and_size[i];
	//	
	//	Kokkos::parallel_for("inner_truc", Kokkos::TeamThreadRange(member, size), [&](const int j)
	//	{
	//		const int o = j + offset;
	//		
	//		c(o) = a(o) + b(o);
	//	}); 
	//	
	//	Kokkos::parallel_for("inner_truc", 
	//});
	
	return EXIT_SUCCESS;
}
