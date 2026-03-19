#pragma once

#include <samurai_kokkos_environment.hpp>

template<typename Context>
class Scope
{
public:
	explicit Scope(Context& context) { m_previousContext = s_activeContext; s_activeContext = std::addressof(context); }
	
	~Scope() { s_activeContext = m_previousContext; }
	
	Scope(const Scope&) = delete;
	
	Scope(Scope&&) = delete;
	
	Scope& operator=(const Scope&) = delete;
	
	Scope& operator=(Scope&&) = delete;
	
	[[nodiscard]] 
	inline constexpr static Context& getContex()
	{
		if (s_activeContext == nullptr)
		{
			throw std::runtime_error("No active tape — wrap your call in a TapeScope");
		}
		return *s_activeContext; 
	}
private:
	Context* m_previousContext = nullptr;
	
	inline static thread_local Context* s_activeContext = nullptr;
};
