module Models

function System(func_generic, params)
   func = (t, u)->func_generic(t, u, params)
   return func
end

function lorenz63(t, u, p)
   du = similar(u)
   du[1] = p["σ"]*(u[2]-u[1])
   du[2] = u[1]*(p["ρ"]-u[3]) - u[2]
   du[3] = u[1]*u[2] - p["β"]*u[3]

   return copy(du)
end

lorenz63_true = System(lorenz63, Dict("σ" => 10, "β" => 8/3, "ρ" => 28))

lorenz63_err = System(lorenz63, Dict("σ" => 10, "β" => 8/3 - 0.1, "ρ" => 28))
lorenz63_err2 = System(lorenz63, Dict("σ" => 10 - 0.3, "β" => 8/3, "ρ" => 28.1))
lorenz63_err3 = System(lorenz63, Dict("σ" => 10.1, "β" => 8/3, "ρ" => 28))
lorenz63_err4 = System(lorenz63, Dict("σ" => 10, "β" => (1 + 0.1)*(8/3), "ρ" => 28))

function lorenz96(t, u, p)
   N = p["N"]

   du = similar(u)

   for i=1:N
      if i <= 10
         F = p["F1"]
      elseif i <= 20
         F = p["F2"]
      elseif i <= 30
         F = p["F3"]
      else
         F = p["F4"]
      end
      du[i] = (u[mod(i+1, 1:N)] - u[mod(i-2, 1:N)])*u[mod(i-1, 1:N)] - u[i] + F
   end

   return copy(du)
end

lorenz96_true = System(lorenz96, Dict("F1" => 8, "F2" => 10, "F3" => 12, "F4" => 14, "N" => 40))

lorenz96_err = System(lorenz96, Dict("F1" => 8, "F2" => 8, "F3" => 8, "F4" => 8, "N" => 40))
lorenz96_err2 = System(lorenz96, Dict("F1" => 12, "F2" => 12, "F3" => 12, "F4" => 12, "N" => 40))
lorenz96_err3 = System(lorenz96, Dict("F1" => 14, "F2" => 14, "F3" => 14, "F4" => 14, "N" => 40))
lorenz96_err4 = System(lorenz96, Dict("F1" => 10, "F2" => 10, "F3" => 10, "F4" => 10, "N" => 40))

lorenz96_half_true = System(lorenz96, Dict("F1" => 8, "F2" => 10, "F3" => 12, "F4" => 14, "N" => 20))

function lorenz96_twoscale(t, u, p)
   N = p["N"]
   n = p["n"]

   dx = zeros(N)
   dy = zeros(n, N)

   u = reshape(u, n + 1, N)
   x = u[1, :]
   y = u[2:end, :]

   for i=1:N
      if i <= 10
         F = p["F1"]
      elseif i <= 20
         F = p["F2"]
      elseif i <= 30
         F = p["F3"]
      else
         F = p["F4"]
      end
      dx[i] = (x[mod(i+1, 1:N)] - x[mod(i-2, 1:N)])*x[mod(i-1, 1:N)] - x[i] + F - p["h"]*p["c"]/p["b"]*sum(y[:, i])

      for j=1:n
         dy[j, i] = p["c"]*p["b"]*y[mod(j+1, 1:n), i]*(y[mod(j-1, 1:n), i] - y[mod(j+2, 1:n), i]) - p["c"]*y[j, i] + p["h"]*p["c"]/p["b"]*x[i]
      end
   end

   du = vec([dx dy']')

   return du
end

lorenz96_twoscale_true = System(lorenz96_twoscale, Dict("F1" => 8, "F2" => 10, "F3" => 12, "F4" => 14, "b" => 10, "c" => 10, "h" => 1.0, "N" => 20, "n" => 10))
lorenz96_twoscale_err = System(lorenz96_twoscale, Dict("F1" => 8.5, "F2" => 9.5, "F3" => 8, "F4" => 8, "b" => 10, "c" => 10, "h" => 1.0, "N" => 20, "n" => 10))

end
