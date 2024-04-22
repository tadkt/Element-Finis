# # # # # # # # # # # #
#  MESH of a 2D house vertical cut of a hut
# # # # # # # # # # # # 
import fenics as fe
from fenics import *
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
from mshr import Polygon, generate_mesh
# mshr is the mesh generation component of FEniCS. It generates simplicial DOLFIN meshes in 2D and 3D
# see e.g.https://bitbucket.org/fenics-project/mshr/

# Define the thermal diffusivity, Stefan-Boltzmann constant, and external field temperature
mu = Constant(1.0)  # Replace with your value for thermal diffusivity
sigma = Constant(5.67e-8)  # Replace with your value for the Stefan-Boltzmann constant
u_f = Constant(300.0)  # Replace with the external field temperature in Kelvin
u_fire = Constant(500.0)  # Replace with the fireplace temperature in Kelvin

# Create empty Mesh 
mesh = Mesh()

# Create list of polygonal domain vertices
domain_vertices = [
    Point(0.0, 0.0), # edge of house
    Point(4.0, 0.0), # edge fireplace
    Point(4.0, 1.0), # edge fireplace
    Point(6.0, 1.0), # edge fireplace
    Point(6.0, 0.0), # edge fireplace
    Point(10.0, 0.0), # edge of house
    Point(10.0, 1.0), # edge window
    Point(10.0, 2.0), # edge window
    Point(10.0, 5.0), # edge of house
    Point(8.0, 5.0), # edge chimney
    Point(8.0, 7.0), # edge chimney
    Point(7.0, 7.0), # edge chimney
    Point(7.0, 5.0), # edge chimney
    Point(0.0, 5.0), # edge of house
]

domain = Polygon(domain_vertices)
mesh = generate_mesh(domain, 20)
plot(mesh)
plt.title("Mesh of a simple Hut")
plt.show()

# Initialize mesh function for interior domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())  # CellFunction
domains.set_all(0)

# Define new measures associated with the interior domains
dx = Measure("dx", domain=mesh, subdomain_data=domains)


# Define the boundaries
wall_thick = 1e-7 # 0.5
roof_thick = 1e-7 # 0.8
chimney_thick = 1e-7 # 0.2
floor_thick = 1e-7 # 0.5
window_thick = 1e-7 # 0.06
fireplace_thick = 1e-7 # 0.10

class Wall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            near(x[0], 0)
            or (near(x[0], 10) and x[1] <= 1)
            or (near(x[0], 10) and x[1] >= 2)
        )

class Roof(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (near(x[1], 5) and x[0] <= 7) or (near(x[1], 5) and x[0] >= 8)
        )

class Chimney(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (near(x[0], 7) and x[1] >= 5)
            or (near(x[0], 8) and x[1] >= 5)
            or (near(x[1], 7))
        )

class Floor(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0) and (x[0] <= 4 or x[0] >= 6)

class Window(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 10) and between(x[1], (1, 3.5)))

class Fire(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 1) and between(x[0], (4, 6)))

class Brick(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 4) and x[1] <= 1) or (near(x[0], 6) and x[1] <= 1)

class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (0.5, 0.7)) and between(x[0], (0.2, 1.0))


# create a cell function over the boundaries edges
sub_boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1) # mesh.topology().dim()-1
# set marker to 6
sub_boundaries.set_all(0)

wall = Wall()
wall.mark(sub_boundaries, 1)

roof = Roof()
roof.mark(sub_boundaries, 2)

chimney = Chimney()
chimney.mark(sub_boundaries, 3)

floor = Floor()
floor.mark(sub_boundaries, 4)

window = Window()
window.mark(sub_boundaries, 5)

fire = Fire()
fire.mark(sub_boundaries, 6)

brick = Brick()
brick.mark(sub_boundaries, 7)

# redefining integrals over boundaries
ds = Measure('ds', domain=mesh, subdomain_data=sub_boundaries)

# Define new measures associated with the interior domains
dx = Measure("dx", domain=mesh, subdomain_data=domains)

    
# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Material property and constants
mu = Constant(1.0)  # Replace with the actual value for thermal diffusivity
sigma = Constant(5.67e-8)  # Stefan-Boltzmann constant
u_f = Constant(300.0)  # Temperature of the field
u_fire = Constant(500.0)  # Temperature of the fire
u_0 = Constant(3.0)  # Initial temperature of 3 degrees Celsius

# Initial condition
u_n = interpolate(u_0, V)


class GammaFlux(SubDomain):
    def inside(self, x, on_boundary):
        # Combine all the marked boundaries for Γflux
        return on_boundary and (
            Wall().inside(x, on_boundary) or
            Roof().inside(x, on_boundary) or
            Chimney().inside(x, on_boundary) or
            Floor().inside(x, on_boundary) or
            Brick().inside(x, on_boundary)
        )

# We already have the Fire class defined as Γfire:
class Fire(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 1) and between(x[0], (4, 6)))

sub_boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_boundaries.set_all(0)

gamma_flux = GammaFlux()
gamma_flux.mark(sub_boundaries, 1)

gamma_fire = Fire()
gamma_fire.mark(sub_boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=sub_boundaries)
# Constants and material properties
mu = 1.0  # Thermal conductivity
sigma = 5.67e-8  # Stefan-Boltzmann constant
u_f = 300  # External temperature field
u_fire = 500  # Fire temperature
# c_expression = "x - 3"  # Placeholder for c(x) expression

# Neumann condition for Γflux
g_flux = Expression('mu * uf', mu=mu, uf=u_f, degree=2)

# Neumann condition for Γfire
g_fire = Expression('mu * sigma * (pow(u, 4) - pow(u_fire, 4))',
                    mu=mu, sigma=sigma, u_fire=u_fire, degree=2)

# Now, we define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = mu * dot(grad(u), grad(v)) * dx
L = g_flux * v * ds(1) + g_fire * v * ds(2)

# Initial condition
u_n = interpolate(Constant(3.0), V)  # Initial temperature uniformly at 3 degrees Celsius
u = Function(V)
solve(a == L, u)

# # # # # # # # # # # # 
# # MULTIPLE NEUMAN BCs
# # # # # # # # # # # #

# # Create classes for defining parts of the boundaries
# class Left(SubDomain):
#     def inside(self, x, on_boundary):
#         return near(x[0], 0.0, tol_bc)

# class Right(SubDomain):
#     def inside(self, x, on_boundary):
#         return near(x[0], 1.0, tol_bc)
    
# left = Left()
# top = Top()

# # Initialize mesh function for boundary domains
# boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# boundaries.set_all(0)
# left.mark(boundaries, 1)
# top.mark(boundaries, 2)

# ds = ds(subdomain_data = boundaries)
# #ds(1) = Gamma_in ; ds(2) = Gamma_wall
    
# a = inner(grad(u_n),grad(v)) * dx + c * u_n*v * ds(2) 

# solve(a == F, u_n, bc0)
    
