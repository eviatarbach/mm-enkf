module Models

using Zygote

struct System
   func
   jac
   params
end

function System(func_generic, params)
   func = (t, u)->func_generic(t, u, params)
   jac = (t, u)->jacobian(func_generic, t, u, params)[2]
   return System(func, jac, params)
end

zygote = false
sim_func = zygote ? Zygote.Buffer : similar

function colpitts(t, u, p)
   M = 2

   c = [p["c21"], p["c32"], p["c13"]]

   du = zeros(3*M)

   for i=0:M-1
      x1, x2, x3 = u[i*3 + 1:i*3 + 3]
      du[3*i + 1] = p["p1"]*x2 + (c[i%3 + 1])*(u[(3*(i + 1) + 1) % (3*M)] - x1)
      du[3*i + 2] = -p["p2"]*(x1 + x3) - p["p4"]*x2
      du[3*i + 3] = p["p3"][i%3 + 1]*(x2 + 1 - exp(-x1))
   end
   return du
end

colpitts_true = (t, u)->colpitts(t, u, Dict("p1" => 5.0, "p2" => 0.0797,
                                            "p3" => 3*[3.0, 3.5, 4.0],
                                            "p4" => 0.6898, "c21" => 0.05,
                                            "c32" => 0.1, "c13" => 0.15))

colpitts_err = (t, u)->colpitts(t, u, Dict("p1" => 5.0 + 0.1,
                                           "p2" => 0.0797 + 0.01,
                                           "p3" => 3*[3.0, 3.5, 4.0],
                                           "p4" => 0.6898, "c21" => 0.05,
                                           "c32" => 0.1, "c13" => 0.15))

function chua(t, u, p)
   x, y, z = u

   du = zeros(3)

   f = p["m_1"]*x + 0.5*(p["m_0"] - p["m_1"])*(abs(x + 1) - abs(x - 1))

   du[1] = p["α"]*(y - x - f)
   du[2] = x - y + z
   du[3] = -p["β"]*y

   return du
end

chua_true = (t, u)->chua(t, u, Dict("α" => 15.6, "β" => 25.58, "m_1" => -5/7,
                                    "m_0" => -8/7))

chua_err = (t, u)->chua(t, u, Dict("α" => 15.7, "β" => 24.58, "m_1" => -5/7,
                                    "m_0" => -8/7))

function lorenz63(t, u, p)
   du = sim_func(u)
   du[1] = p["σ"]*(u[2]-u[1])
   du[2] = u[1]*(p["ρ"]-u[3]) - u[2]
   du[3] = u[1]*u[2] - p["β"]*u[3]

   return copy(du)
end

# function lorenz63_stochastic(t, u, p)
#    x, y, z = u

#    du = zeros(3)
#    du[1] = p["σ"]*(u[2]-u[1])
#    du[2] = u[1]*(p["ρ"]-u[3]) - u[2]
#    du[3] = u[1]*u[2] - p["β"]*u[3]

#    return du + p["r"]*randn(3)
# end

lorenz63_true = System(lorenz63, Dict("σ" => 10, "β" => 8/3, "ρ" => 28))

lorenz63_err = System(lorenz63, Dict("σ" => 10, "β" => 8/3 - 0.1, "ρ" => 28))
lorenz63_err2 = System(lorenz63, Dict("σ" => 10 - 0.3, "β" => 8/3, "ρ" => 28.1))
lorenz63_err3 = System(lorenz63, Dict("σ" => 10.00001, "β" => 8/3, "ρ" => 28))
lorenz63_err4 = System(lorenz63, Dict("σ" => 10, "β" => (1 + 0.1)*(8/3), "ρ" => 28))

function lorenz96(t, u, p)
   N = 40

   # compute state derivatives
   du = sim_func(u)

   # first the 3 edge cases: i=1,2,N
   du[1] = (u[2] - u[N-1])*u[N] - u[1] + p["F1"]
   du[2] = (u[3] - u[N])*u[1] - u[2] + p["F1"]
   du[N] = (u[1] - u[N-2])*u[N-1] - u[N] + p["F4"]

   # then the general case
   for i=3:N-1
      if i <= 10
         F = p["F1"]
      elseif i <= 20
         F = p["F2"]
      elseif i <= 30
         F = p["F3"]
      else
         F = p["F4"]
      end
      du[i] = (u[i+1] - u[i-2])*u[i-1] - u[i] + F
   end

   #du = du .+ p["F"]

   return copy(du)
end

lorenz96_true = System(lorenz96, Dict("F1" => 8, "F2" => 8, "F3" => 12, "F4" => 12))

lorenz96_err = System(lorenz96, Dict("F1" => 8, "F2" => 8, "F3" => 8, "F4" => 8))
lorenz96_err2 = System(lorenz96, Dict("F1" => 12, "F2" => 12, "F3" => 12, "F4" => 12))
lorenz96_err3 = System(lorenz96, Dict("F1" => 14, "F2" => 14, "F3" => 14, "F4" => 14))
lorenz96_err4 = System(lorenz96, Dict("F1" => 10, "F2" => 10, "F3" => 10, "F4" => 10))

end