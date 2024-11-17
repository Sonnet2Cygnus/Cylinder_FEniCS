from fenics import *
from mshr import *
import matplotlib.pyplot as plt

# parameters
D = 1                       ## radius
H = 4                       ## H of farfield
L = 15                      ## L of farfield
A = 3                       ## the position of cylinder from inlet
U = 1                       ## inlet velocity 
Re = 1000
f = Constant((0., 0.))      ## body force

# build the mesh
box = Rectangle(Point(-A, -H/2), Point(L-A, H/2))
cylinder = Circle(Point(0., 0.), D/2)
domain = box - cylinder # computational domain
n = 100
mesh = generate_mesh(domain, n)

# plot the mesh
plot(mesh)
plt.title("Mesh")
plt.show()

# define function spaces
V = VectorElement('P', mesh.ufl_cell(), 2)       # quadratic vector function space --veclotiy
Q = FiniteElement('P', mesh.ufl_cell(), 1)       # 1 ci de 
X = FunctionSpace(mesh, MixedElement([V, Q]))

# define boundary conditions
def dirichlet_boundary(x, on_boundary):
    return on_boundary and not near(x[0], L-A)

# define inlet conditions
g_D = Expression(('near(x[0], -A) ? U : 0.', '0.'), degree=3, U=U, A=A)
bc_U = DirichletBC(X.sub(0), g_D, dirichlet_boundary)

# define reference pressure point 0
def left_corner(x, on_boundary):
    return near(x[0], -A) and near(x[1], -H/2)

# define pressure boundary conditions
bc_P = DirichletBC(X.sub(1), Constant(0.), left_corner, 'pointwise')

bc = [bc_U, bc_P]

# define Newton step
def newton_step(X, u_old, u_pre, Re, bc, f, dt):

    u, p = TrialFunctions(X)
    v, q = TestFunctions(X)

    # in weak form, discrete matrix A and l RHS teerm 

    a = (1/dt * dot(u, v) * dx   # dt 
         + dot(dot(grad(u), u_old), v) * dx  # linear convection 
         + dot(dot(grad(u_old), u), v) * dx  # no-linear convection
         + 1/Re * inner(grad(u), grad(v)) * dx # viscous diffusion
         - p * div(v) * dx + q * div(u) * dx) # pressure and incompressible 

    L = (1/dt * dot(u_pre, v) * dx # dt
         + dot(f, v) * dx          # extra force
         + dot(dot(grad(u_old), u_old), v) * dx) #convection 

    x = Function(X)
    solve(a == L, x, bc)
    u, p = x.split()
    return u, p

# output files for visualization
velocity_file = File('velocity.pvd')
pressure_file = File('pressure.pvd')

# initial conditions
u_pre = project(Constant((U, 0.)), VectorFunctionSpace(mesh, 'P', 2))
p_pre = project(Constant(0.), FunctionSpace(mesh, 'P', 1))

u_pre.rename('velocity', 'velocity')
p_pre.rename('pressure', 'pressure')

velocity_file << (u_pre, 0)
pressure_file << (p_pre, 0)

# time-stepping parameters
t = 0
T = 30.0
dt = 0.2
time_index = 0
restart = 5

# newton parameters
tol = 1e-6
maxit = 10

# time-stepping loop
while t + dt < T:
    t += dt
    u_old = u_pre
    p_old = p_pre
    it = 0
    err = 2 * tol
    
    print(f"\nTime t = {t}")
    
    # Newton iteration
    while err > tol and it < maxit:
        u, p = newton_step(X, u_old, u_pre, Re, bc, f, dt)
        err = errornorm(u, u_old, 'H1')
        u_old = u
        p_old = p
        print(f"Newton iteration = {it}, error = {err}")
        it += 1
    
    # update solutions
    u_pre = u
    p_pre = p

    # save results for visualization
    u_pre.rename('velocity', 'velocity')
    p_pre.rename('pressure', 'pressure')
    time_index += 1
    if time_index % restart == 0:
        velocity_file << (u_pre, t)
        pressure_file << (p_pre, t)

# visualization
plt.figure()
q = plot(u_pre)
plt.colorbar(q)
plt.title("Velocity")

plt.figure()
q = plot(p_pre)
plt.colorbar(q)
plt.title("Pressure")

plt.show()
