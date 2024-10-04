import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.ContinuedFractions.Computation
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.NormedSpace.InnerProduct
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Log
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Pi
import Mathlib.Data.Set.Basic
import Mathlib.Data.Tuple
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Eigenspace
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Basic
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace minimum_dwarfs_l644_644048

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l644_644048


namespace hotel_digit_packages_needed_l644_644440

theorem hotel_digit_packages_needed :
  (∀ f : ℕ → ℕ, 
   (∀ n, (n ≥ 210 ∧ n <= 240 ∨ n ≥ 310 ∧ n <= 340) → 
    ∃ j, j ∈ (set.range (λ i, i % 10)) ∧ 
          f n = (nat.digits 10 n).count j)) →
  ∃ p, ∀ d ∈ (set.range (λ i, i % 10)),
    list.count d ((list.range' 210 (240-210+1)) ++ (list.range' 310 (340-310+1))) <= p * 10 ∧ 
    list.count 3 ((list.range' 210 (240-210+1)) ++ (list.range' 310 (340-310+1))) > (p - 1) * 10 :=
sorry

end hotel_digit_packages_needed_l644_644440


namespace sin_sq_alpha_l644_644637

theorem sin_sq_alpha (α β : ℝ) (h0: 0 < α) (h1: α < π/2)
  (h2: cos(α - β) = 4/5) : sin(α) ^ 2 = 144/169 :=
sorry

end sin_sq_alpha_l644_644637


namespace integral_ln_squared_over_cubic_root_l644_644964

theorem integral_ln_squared_over_cubic_root 
  : ∫ x in (1 : ℝ)..8, (ln x) ^ 2 / (x ^ (2 / 3)) = 6 * (ln 8) ^ 2 - 36 * ln 8 + 54 :=
by
  sorry

end integral_ln_squared_over_cubic_root_l644_644964


namespace total_savings_l644_644383

theorem total_savings (savings_sep savings_oct : ℕ) 
  (h1 : savings_sep = 260)
  (h2 : savings_oct = savings_sep + 30) :
  savings_sep + savings_oct = 550 := 
sorry

end total_savings_l644_644383


namespace polygon_not_all_lattice_points_l644_644103

theorem polygon_not_all_lattice_points (n : ℕ) (h_n : n = 1994) (a : ℕ → ℝ) 
  (h_a : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i = real.sqrt (i^2 + 4)) :
  ¬ ∃ (v : ℕ → ℕ × ℕ), ∀ i : ℕ, 1 ≤ i ∧ i < n → 
    a i = real.sqrt ((v (i+1)).fst - (v i).fst)^2 + ((v (i+1)).snd - (v i).snd)^2 := 
sorry

end polygon_not_all_lattice_points_l644_644103


namespace monotonicity_f_range_of_k_for_p_eq_1_log_inequality_l644_644306

def f (x : ℝ) (p : ℝ) : ℝ := p * log x + (p - 1) * x^2 + 1

-- (I) Monotonicity of f(x)
theorem monotonicity_f (p : ℝ) :
  (∀ x : ℝ, 0 < x → (p > 1 → deriv ( λ (x : ℝ), f x p ) x > 0) ∧ 
          (p ≤ 0 → deriv ( λ (x : ℝ), f x p ) x < 0) ∧
          (-1 < p ∧ p < 0 → ∃ c : ℝ, 0 < c ∧ 
          ∀ x : ℝ, 0 < x ∧ x < c → deriv ( λ (x : ℝ), f x p ) x > 0 ∧ 
                    c < x  → deriv ( λ (x : ℝ), f x p ) x < 0)) := sorry

-- (II) Range of k when p = 1
theorem range_of_k_for_p_eq_1 :
  ∀ x : ℝ, 0 < x → ∀ k : ℝ, ( ∀ x : ℝ, f x 1 ≤ k * x ) ↔ (1 ≤ k) := sorry

-- (III) Inequality ln(n + 1) < 1 + 2 + ... + n for n ∈ ℕ^*
theorem log_inequality (n : ℕ) (hn : n > 0) :
  log (n + 1) < (finset.range n).sum (λ i, (i + 1)) := sorry

end monotonicity_f_range_of_k_for_p_eq_1_log_inequality_l644_644306


namespace domain_of_g_comp_f_l644_644973

-- Define the functions as per the conditions
def f (x : ℝ) := x^2 - x - 4
def g (x : ℝ) := real.log x / real.log 2

-- Define the domain of g ∘ f
def domain_g_comp_f : set ℝ :=
  {x | f x > 2}

-- Prove that domain of g ∘ f is (-∞, -2) ∪ (3, ∞) given conditions
theorem domain_of_g_comp_f :
  domain_g_comp_f = {x | x < -2 ∨ x > 3} :=
sorry

end domain_of_g_comp_f_l644_644973


namespace sequence_comparison_l644_644681

-- Conditions
def domain_R (f : ℝ → ℝ) : Prop := ∀ x, x ∈ ℝ
def f_pos_when_x_neg (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x < 0 → f x > 1
def functional_eq (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f(x) * f(y) = f(x + y)
def sequence_condition (f : ℝ → ℝ) (a : ℕ → ℝ) : Prop := ∀ n : ℕ, n > 0 → f(a(n + 1)) * f(1 / (1 + a n)) = 1
def initial_condition (f : ℝ → ℝ) (a : ℕ → ℝ) : Prop := a 1 = f 0

-- Proof problem stating the desired comparison
theorem sequence_comparison 
  (f : ℝ → ℝ) 
  (a : ℕ → ℝ) 
  (h1 : domain_R f)
  (h2 : f_pos_when_x_neg f)
  (h3 : functional_eq f)
  (h4 : sequence_condition f a)
  (h5 : initial_condition f a) : 
  f (a 2016) < f (a 2015) := 
sorry

end sequence_comparison_l644_644681


namespace generating_function_solution_count_l644_644780

noncomputable theory

def a_n (n k : ℕ) : ℕ :=
  ((Polynomial.X : Polynomial ℚ)^(n + k - 1)).coeff n

def F (x : ℚ) (k : ℕ) : ℚ :=
  (1 - x) ^ -k

theorem generating_function (k : ℕ) : 
  generate_function (λ n, a_n n k) = λ x, F x k :=
sorry

theorem solution_count (n k : ℕ) :
  a_n n k = nat.choose (n + k - 1) n :=
sorry

end generating_function_solution_count_l644_644780


namespace f_is_odd_function_f_monotonicity_domain_interval_exists_l644_644310

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log m ((x - 3) / (x + 3))

-- Problem 1: Parity of the function
theorem f_is_odd_function (m : ℝ) : (∀ x : ℝ, f m (-x) = - (f m x)) :=
by
  sorry

-- Problem 2: Monotonicity of the function
theorem f_monotonicity (m : ℝ) (α β : ℝ) (hα : α > 0) (hβ : β > α) :
  if 0 < m ∧ m < 1 then ∀ (x1 x2 : ℝ), x1 ∈ Icc α β ∧ x2 ∈ Icc α β ∧ x1 < x2 → f m x1 > f m x2
  else ∀ (x1 x2 : ℝ), x1 ∈ Icc α β ∧ x2 ∈ Icc α β ∧ x1 < x2 → f m x1 < f m x2 :=
by
  sorry

-- Problem 3: Existence of domain interval [α, β]
theorem domain_interval_exists (m : ℝ) (α β : ℝ) (hα : α > 0) (hβ : β > α) :
  if 0 < m ∧ m < (2 - real.sqrt 3) / 4 then 
    (log m (m * (β - 1)) ∈ Icc (f m α) (f m β)) ∧ 
    (log m (m * (α - 1)) ∈ Icc (f m α) (f m β))
  else 
    ∀ (x : ℝ), ¬ (log m (m * (x - 1)) ∈ Icc (f m α) (f m β)) :=
by
  sorry

end f_is_odd_function_f_monotonicity_domain_interval_exists_l644_644310


namespace sin_arith_seq_max_elements_l644_644784

theorem sin_arith_seq_max_elements (d k : ℝ) (h_d : d = 2 * Real.pi / k) (h_k_ge_2 : 2 ≤ k) (h_k_nat : ∃ (n : ℕ), k = n) :
  ∃ (T : Finset ℝ), (∀ (n : ℕ), T ∋ (Real.sin (n * d)))
  → T.card ≤ Int.to_nat k ∧ T.sum = 0 :=
by
  -- Proof-related code goes here
  sorry

end sin_arith_seq_max_elements_l644_644784


namespace at_least_one_male_probability_l644_644172

/-- 
Given a total of 4 female doctors and 3 male doctors, 
if we randomly select 3 doctors from these 7, 
the probability of selecting at least 1 male doctor is 31/35. 
-/
theorem at_least_one_male_probability 
  (total_females: ℕ) 
  (total_males: ℕ) 
  (selection: ℕ) 
  (probability: ℚ) 
  (total_females = 4) 
  (total_males = 3) 
  (selection = 3) 
  (total_doctors = total_females + total_males) 
  (total_ways := (nat.choose total_doctors selection)) : 
  probability = 31/35 := 
sorry

end at_least_one_male_probability_l644_644172


namespace connie_tickets_l644_644221

theorem connie_tickets (total_tickets spent_on_koala spent_on_earbuds spent_on_glow_bracelets : ℕ)
  (h1 : total_tickets = 50)
  (h2 : spent_on_koala = total_tickets / 2)
  (h3 : spent_on_earbuds = 10)
  (h4 : total_tickets = spent_on_koala + spent_on_earbuds + spent_on_glow_bracelets) :
  spent_on_glow_bracelets = 15 :=
by
  sorry

end connie_tickets_l644_644221


namespace integral_problem_1_integral_problem_2_integral_problem_3_l644_644256

-- Problem 1
theorem integral_problem_1 (x : ℝ) :
  ∫ (λ x => 1 / (sqrt(x) + sqrt(sqrt(x+1)))) dx =
  4 * (sqrt(x) / 2 - sqrt(sqrt(x)) + log(1 + sqrt(sqrt(x)))) + C := 
sorry

-- Problem 2
theorem integral_problem_2 (x : ℝ) :
  ∫ (λ x => (x + sqrt(1 + x)) / cbrt(1 + x)) dx =
  6 * (x^(10/6) / 10 + x^(7/6) / 7 - x^(4/6) / 4) + C :=
sorry

-- Problem 3
theorem integral_problem_3 (x : ℝ) :
  ∫ (λ x => sqrt(x) / (x^2 * sqrt(x - 1))) dx =
  2 * sqrt((x - 1) / x) + C :=
sorry

end integral_problem_1_integral_problem_2_integral_problem_3_l644_644256


namespace arithmetic_sequence_properties_sum_bn_l644_644295

noncomputable def Sn (n : ℕ) (a_1 d : ℤ) : ℤ :=
  n * (a_1 + (n-1) * d)

noncomputable def a_n (n : ℕ) (a_1 d : ℤ) : ℤ :=
  a_1 + (n - 1) * d

noncomputable def b_n (n : ℕ) (a_n : ℕ → ℤ) : ℤ :=
  2 * a_n n + 2 ^ a_n n

noncomputable def T_n (n : ℕ) (b_n : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum (λ k, b_n (k + 1))

theorem arithmetic_sequence_properties (a_1 : ℤ) (d : ℤ) (h1 : a_1 = 2) (h2 : Sn 7 a_1 d = 4 * (a_n 2 a_1 d + a_n 5 a_1 d)) :
  ∀ n : ℕ, a_n n a_1 d = 2 * n := by
  sorry

theorem sum_bn (a_1 : ℤ) (d : ℤ) (h1 : a_1 = 2) (h2 : Sn 7 a_1 d = 4 * (a_n 2 a_1 d + a_n 5 a_1 d)) (h3 : ∀ n : ℕ, a_n n a_1 d = 2 * n) :
  ∀ n : ℕ, T_n n (b_n (a_n n a_1 d)) = 2 * n * (n + 1) + (4 ^ n - 1) * 4 / 3 := by
  sorry

end arithmetic_sequence_properties_sum_bn_l644_644295


namespace red_squares_in_9x11_l644_644411

theorem red_squares_in_9x11 (sheet : ℕ → ℕ → Prop)
  (H1 : ∀ i j, (i < 2 → j < 3 → sheet i j = true) → (finset.card (finset.filter sheet (finset.range 6)) = 2))
  (H2 : ∀ i j, (i < 3 → j < 2 → sheet i j = true) → (finset.card (finset.filter sheet (finset.range 6)) = 2)) :
  finset.card (finset.filter sheet (finset.range (9 * 11))) = 33 := 
sorry

end red_squares_in_9x11_l644_644411


namespace special_matrix_field_exists_square_roots_l644_644796

def is_commutative_field (K : Type) [field K] : Prop := sorry

structure special_matrix :=
(a b : ℝ)

def K : set (matrix (fin 2) (fin 2) ℝ) :=
{ M | ∃ x : special_matrix, M = ![![x.a, x.b], ![-x.b, x.a]] }

theorem special_matrix_field : is_commutative_field K :=
sorry

theorem exists_square_roots (A : matrix (fin 2) (fin 2) ℝ) 
  (hA : A ∈ K) : ∃ B C : matrix (fin 2) (fin 2) ℝ, B ∈ K ∧ C ∈ K ∧ B * B = A ∧ C * C = A :=
sorry

end special_matrix_field_exists_square_roots_l644_644796


namespace hawks_margin_of_victory_l644_644341

theorem hawks_margin_of_victory :
  let hawks_touchdowns := 4 
  let hawks_extra_points := 2
  let hawks_two_point_conversions := 1
  let hawks_missed_extra_points := 0
  let hawks_field_goals := 2
  let hawks_safeties := 1
  let eagles_touchdowns := 3
  let eagles_extra_points := 3
  let eagles_two_point_conversions := 1
  let eagles_safeties := 1
  let eagles_field_goals := 3
  let hawks_total_points := (hawks_touchdowns * 6) + (hawks_extra_points * 1) + (hawks_two_point_conversions * 2) + (hawks_missed_extra_points * 0) + (hawks_field_goals * 3) + (hawks_safeties * 2)
  let eagles_total_points := (eagles_touchdowns * 6) + (eagles_extra_points * 1) + (eagles_two_point_conversions * 2) + (eagles_safeties * 2) + (eagles_field_goals * 3)
  margin_of_victory = hawks_total_points - eagles_total_points
  margin_of_victory = 2 :=
by
  sorry

end hawks_margin_of_victory_l644_644341


namespace simplify_G_l644_644769

variable (x : ℝ)

def F : ℝ := Real.log ((1 + x) / (1 - x))

def G : ℝ := Real.log ((1 + (2 * x + x^2) / (1 + 2 * x)) / (1 - (2 * x + x^2) / (1 + 2 * x)))

theorem simplify_G : G x = 2 * Real.log (1 + 2 * x) - F x := by
  sorry

end simplify_G_l644_644769


namespace value_of_a_plus_b_l644_644331

theorem value_of_a_plus_b (a b : ℝ) (h : sqrt (a + 3) + abs (b - 5) = 0) : a + b = 2 :=
sorry

end value_of_a_plus_b_l644_644331


namespace juanitas_dessert_cost_l644_644181

theorem juanitas_dessert_cost :
  let brownie_cost := 2.50
  let ice_cream_cost := 1.00
  let syrup_cost := 0.50
  let nuts_cost := 1.50
  let num_scoops_ice_cream := 2
  let num_syrups := 2
  let total_cost := brownie_cost + num_scoops_ice_cream * ice_cream_cost + num_syrups * syrup_cost + nuts_cost
  total_cost = 7.00 :=
by
  sorry

end juanitas_dessert_cost_l644_644181


namespace lambda_range_l644_644091

theorem lambda_range (λ : ℝ) (n : ℕ) (h_pos : 0 < n) (h_dec : ∀ n : ℕ, 0 < n → -2 * n^2 + λ * n > -2 * (n + 1)^2 + λ * (n + 1)) : λ < 6 := 
by 
  sorry

end lambda_range_l644_644091


namespace photos_per_week_in_february_l644_644037

def january_photos : ℕ := 31 * 2

def total_photos (jan_feb_photos : ℕ) : ℕ := jan_feb_photos - january_photos

theorem photos_per_week_in_february (jan_feb_photos : ℕ) (weeks_in_february : ℕ)
  (h1 : jan_feb_photos = 146)
  (h2 : weeks_in_february = 4) :
  total_photos jan_feb_photos / weeks_in_february = 21 := by
  sorry

end photos_per_week_in_february_l644_644037


namespace sum_of_possible_perimeters_l644_644396

open Real

-- Define the given lengths
def AB : ℝ := 12
def BC : ℝ := 16
def AC : ℝ := AB + BC

-- Define the lengths of AD, CD, and BD as integers
variables (AD CD BD : ℕ)

-- Define the conditions
axiom ad_cd_eq : AD = CD
axiom pythagorean_bd : ∃ h : ℝ, h^2 = BD^2 - 4^2
axiom pythagorean_ad : ∃ h : ℝ, h^2 = AD^2 - 8^2

-- Define the main theorem
theorem sum_of_possible_perimeters : (AD, BD : ℕ) → 
  AD = 13 ∧ BD = 11 ∨ AD = 8 ∧ BD = 4 → 
  s = (2 * AD + AC + 2 * 8 + AC) = 98 :=
begin
  sorry -- Proof not required
end

end sum_of_possible_perimeters_l644_644396


namespace symmetric_line_equation_l644_644666

-- Define points P and Q
def P : ℝ × ℝ := (3, 2)
def Q : ℝ × ℝ := (1, 4)

-- Define the midpoint function
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the symmetric line condition
def symmetric_about_line (P Q : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  let M := midpoint P Q in
  l M ∧ ∀ R : ℝ × ℝ, l R → R = (2 * M.1 - P.1, 2 * M.2 - P.2) ∨ R = (2 * M.1 - Q.1, 2 * M.2 - Q.2)

-- Define the line l
def line_l := λ p : ℝ × ℝ, p.1 - p.2 + 1 = 0

-- Theorem stating the property we need to prove
theorem symmetric_line_equation :
  symmetric_about_line P Q line_l → (∀ p : ℝ × ℝ, line_l p ↔ p.1 - p.2 + 1 = 0) :=
by
  sorry

end symmetric_line_equation_l644_644666


namespace max_value_expr_bound_l644_644264

noncomputable def max_value_expr (x : ℝ) : ℝ := 
  x^6 / (x^10 + x^8 - 6 * x^6 + 27 * x^4 + 64)

theorem max_value_expr_bound : 
  ∃ x : ℝ, max_value_expr x ≤ 1 / 8.38 := sorry

end max_value_expr_bound_l644_644264


namespace find_n_l644_644007

theorem find_n 
  (n : ℕ) 
  (b : ℕ → ℝ)
  (h₀ : b 0 = 28)
  (h₁ : b 1 = 81)
  (hn : b n = 0)
  (h_rec : ∀ j : ℕ, 1 ≤ j → j < n → b (j+1) = b (j-1) - 5 / b j)
  : n = 455 := 
sorry

end find_n_l644_644007


namespace sqrt_is_nat_for_n_gt_1_l644_644382

def a_sequence : ℕ → ℝ
| 0     := 1
| (n+1) := a_sequence n / 2 + 1 / (4 * a_sequence n)

theorem sqrt_is_nat_for_n_gt_1 : ∀ n > 1, ∃ k : ℕ, k = Int.floor (Real.sqrt (2 / (2 * (a_sequence n)^2 - 1))) :=
by sorry

end sqrt_is_nat_for_n_gt_1_l644_644382


namespace g_512_minus_g_256_l644_644263

-- Definition of g(n) as described in the problem
def g (n : ℕ) : ℚ :=
  (nat.sigma (2 * n) (λ d, (d % 2 = 0))) / (n : ℚ)

-- Statement of the problem in Lean
theorem g_512_minus_g_256 : g 512 - g 256 = 1 / 512 :=
by
  sorry

end g_512_minus_g_256_l644_644263


namespace solution_l644_644368

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution (x : ℝ) : g (g x) = g x ↔ x = 0 ∨ x = 4 ∨ x = 5 ∨ x = -1 :=
by
  sorry

end solution_l644_644368


namespace number_of_solutions_l644_644654

def f (x : ℝ) : ℝ := |1 - 2 * x|

def iterate_f (n : ℕ) (x : ℝ) : ℝ :=
  Nat.recOn n x (λ _ y, f y)

theorem number_of_solutions :
  { x : ℝ | 0 ≤ x ∧ x ≤ 1 ∧ iterate_f 3 x = (1/2) * x }.finite.toFinset.card = 8 := sorry

end number_of_solutions_l644_644654


namespace original_triangle_angles_l644_644572

theorem original_triangle_angles:
  ∀ (T : Triangle),
    (∃ T1 T2 : Triangle, 
      is_partition T [T1, T2] ∧
      ((is_isosceles T1 ∧ ¬is_equilateral T1) ∧ ∀ Ti ∈ [T2], is_equilateral Ti)) →
    angles T = [30, 60, 90] :=
by
  intro T
  intro h
  -- Here we would normally provide the proof but we omit it as per the problem statement
  sorry

end original_triangle_angles_l644_644572


namespace number_of_friends_l644_644018

theorem number_of_friends (total_crackers initial_friends : ℕ) (each_ate crackers_left : ℕ) : 
  total_crackers = 32 → each_ate = 8 → crackers_left = total_crackers / each_ate → crackers_left = 4 :=
by
  assume h1 : total_crackers = 32
  assume h2 : each_ate = 8
  assume h3 : crackers_left = total_crackers / each_ate
  sorry

end number_of_friends_l644_644018


namespace angle_measure_sector_ABC_l644_644900

theorem angle_measure_sector_ABC 
  (r_cone : ℝ) (radius_circle : ℝ) (vol_cone : ℝ) 
  (h_cone : ℝ) (slant_height : ℝ) (angle_sector_used : ℝ) :
  r_cone = 15 → radius_circle = 31 →
  vol_cone = 2025 * Real.pi →
  h_cone = 27 → slant_height = 31 →
  angle_sector_used ≈ (30 / (2 * 31)) * 360 →
  360 - angle_sector_used = 185.81 :=
by
  intros
  sorry

end angle_measure_sector_ABC_l644_644900


namespace solve_quadratic_eqn_l644_644451

theorem solve_quadratic_eqn (x : ℝ) : 2 * (x + 1) = x * (x + 1) → x = -1 ∨ x = 2 := by
  intro h
  have : (x + 1) * (2 - x) = 0 := by
    sorry
  cases this with
  | inl h1 =>
    -- Proof that x + 1 = 0 implies x = -1
    sorry
  | inr h2 =>
    -- Proof that 2 - x = 0 implies x = 2
    sorry

end solve_quadratic_eqn_l644_644451


namespace Billy_sleep_hours_l644_644952

-- Define the given problem in Lean 4 statement
theorem Billy_sleep_hours :
  ∃ x : ℝ, (6 + x + (x / 2) + (3 * x / 2) = 30) ∧ (x - 6 = 2) :=
begin
  -- Proof step is not required as per the instruction
  sorry
end

end Billy_sleep_hours_l644_644952


namespace log_ratios_l644_644078

noncomputable def ratio_eq : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem log_ratios
  {a b : ℝ}
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : Real.log a / Real.log 8 = Real.log b / Real.log 18)
  (h4 : Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = ratio_eq :=
sorry

end log_ratios_l644_644078


namespace coloring_count_420_l644_644945

/-- Define the problem conditions -/
variables {V : Type} [fintype V]

/-- The vertices of the quadrilateral prism V -/
variables (vertices : set V) (edges : set (V × V))

/-- Only 5 different colors are available -/
def colors := fin 5

/-- Each vertex is colored such that the two endpoints of the same edge have different colors -/
def valid_coloring (c : V → colors) : Prop :=
  ∀ {v1 v2 : V}, (v1, v2) ∈ edges → c v1 ≠ c v2

/-- The number of valid colorings of the quadrilateral prism -/
def count_valid_colorings : nat :=
  fintype.card {c : V → colors // valid_coloring c}

/-- Prove the total number of different ways to color the vertices of the quadrilateral prism is 420 -/
theorem coloring_count_420 : count_valid_colorings = 420 :=
sorry

end coloring_count_420_l644_644945


namespace parabola_tangent_line_l644_644736

theorem parabola_tangent_line (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b * x + 2 = 2 * x + 3 → a = -1 ∧ b = 4) :=
sorry

end parabola_tangent_line_l644_644736


namespace distribute_peanuts_l644_644237

theorem distribute_peanuts (n m : ℕ) (hn : n = 1600) (hm : m = 100) :
  (∃ k, 1 ≤ k ∧ k ≤ 1600 / 100 ∧ (∃ d : ℕ, d = 4 ∧ (count_ℕ k d m) d >= 4)) ∧
  (∃ (g1 g2 g3 : Finset ℕ) (g4 : ℕ),
    (g1.card = 31 ∧ g2.card = 31 ∧ g3.card = 31 ∧ g4 = 7) ∧
    (∀ i ∈ g1, ∃ d : ℕ, d = 3 ∧ (count_ℕ i d m) d >= 3) ∧
    (∀ i ∈ g2, ∃ d : ℕ, d = 2 ∧ (count_ℕ i d m) d >= 2) ∧
    (∀ i ∈ g3, i = 1) ∧
    (g1.sum + g2.sum + g3.sum + g4 = 1600)) :=
by
  sorry

end distribute_peanuts_l644_644237


namespace ratio_of_areas_l644_644206

variables {V : Type*} [inner_product_space ℝ V]
variables {A E D C F B: V}
variables (area : fin 2 → set V → ℝ)

-- Define vector relations
variables (h1 : 6 • (E - A) + 3 • (E - D) = 2 • (C - A))
variables (h2 : 4 • (B - F) + 5 • (D - F) = 2 • (D - A))

-- Define the areas of structures
noncomputable def area_ACDB : ℝ := sorry
noncomputable def area_AEDF : ℝ := sorry

-- Prove the ratio of areas
theorem ratio_of_areas : area_ACDB / area_AEDF = 27 / 8 :=
sorry

end ratio_of_areas_l644_644206


namespace expected_value_of_heads_l644_644558

theorem expected_value_of_heads :
  let penny_value := 1 
  let nickel_value := 5 
  let dime_value := 10 
  let quarter_value := 25 
  let half_dollar_value := 50 
  let prob_heads := 1/2
  let expected_value := prob_heads * (penny_value + nickel_value + dime_value + quarter_value + half_dollar_value)
  in expected_value = 45.5 :=
by
  let penny_value := 1 
  let nickel_value := 5 
  let dime_value := 10 
  let quarter_value := 25 
  let half_dollar_value := 50 
  let prob_heads := 1/2
  let expected_sum := penny_value + nickel_value + dime_value + quarter_value + half_dollar_value
  let expected_value := prob_heads * expected_sum
  have h : expected_sum = 91 := by simp [penny_value, nickel_value, dime_value, quarter_value, half_dollar_value]
  have h2 : expected_value = prob_heads * 91 := by rw h
  norm_num [prob_heads] at h2
  exact h2.symm

end expected_value_of_heads_l644_644558


namespace soccer_team_problem_l644_644394

theorem soccer_team_problem :
  let quadruplets := ({Brian, Brad, Bill, Bob} : Finset string)
  let all_players := (quadruplets ∪ (range 12).map (λ n, "Player" ++ toString n) : Finset string)
  let starters_quadruplets := quadruplets.powerset.filter (λ s, s.card = 3)
  let starters_remaining := (all_players \ quadruplets).powerset.filter (λ s, s.card = 3)
  (quadruplets.card = 4) ∧ (all_players.card = 16) ∧ (starters_quadruplets.card = 4) 
  ∧ (starters_remaining.card = 220) → 
  (starters_quadruplets.card * starters_remaining.card = 880) :=
by
  sorry

end soccer_team_problem_l644_644394


namespace find_y_from_exponentiation_l644_644724

theorem find_y_from_exponentiation (y : ℝ) (h : 3^6 = 9^y) : y = 3 := by
  sorry

end find_y_from_exponentiation_l644_644724


namespace probability_ace_of_spades_l644_644159

theorem probability_ace_of_spades (total_cards : ℕ) (black_cards : ℕ) (removed_black_cards : ℕ)
  (total_cards = 52) (black_cards = 26) (removed_black_cards = 12)
  : (1 : ℚ) / (total_cards - removed_black_cards) = 1 / 40 := sorry

end probability_ace_of_spades_l644_644159


namespace fill_bucket_time_l644_644079

def rate_tap_A : ℝ := 3 -- Rate of Tap A in liters per minute
def total_volume : ℝ := 50 -- Total volume of the bucket in liters
def initial_volume : ℝ := 8 -- Initial volume of the bucket in liters
def one_third_bucket : ℝ := total_volume / 3 -- One third of the bucket's volume
def half_bucket : ℝ := total_volume / 2 -- Half of the bucket's volume
def time_tap_B : ℝ := 20 -- Time in minutes for Tap B to fill one third of the bucket
def time_tap_C : ℝ := 30 -- Time in minutes for Tap C to fill half of the bucket

def rate_tap_B : ℝ := one_third_bucket / time_tap_B -- Rate of Tap B in liters per minute
def rate_tap_C : ℝ := half_bucket / time_tap_C -- Rate of Tap C in liters per minute

def combined_rate : ℝ := rate_tap_A + rate_tap_B + rate_tap_C -- Combined rate in liters per minute
def remaining_volume : ℝ := total_volume - initial_volume -- Remaining volume to fill in liters

def time_to_fill_bucket : ℝ := remaining_volume / combined_rate -- Time to fill the remaining volume

theorem fill_bucket_time :
  time_to_fill_bucket = 9 :=
by
  sorry

end fill_bucket_time_l644_644079


namespace possible_values_of_n_l644_644844

theorem possible_values_of_n (P1 P2 P3 : Plane)
    (h1 : ¬(P1 = P2)) (h2 : ¬(P1 = P3)) (h3 : ¬(P2 = P3)):
  ∃ n : ℕ, n ∈ {4, 6, 7, 8} :=
by sorry

end possible_values_of_n_l644_644844


namespace pipe_cistern_problem_l644_644151

theorem pipe_cistern_problem:
  ∀ (rate_p rate_q : ℝ),
    rate_p = 1 / 10 →
    rate_q = 1 / 15 →
    ∀ (filled_in_4_minutes : ℝ),
      filled_in_4_minutes = 4 * (rate_p + rate_q) →
      ∀ (remaining : ℝ),
        remaining = 1 - filled_in_4_minutes →
        ∀ (time_to_fill : ℝ),
          time_to_fill = remaining / rate_q →
          time_to_fill = 5 :=
by
  intros rate_p rate_q Hp Hq filled_in_4_minutes H4 remaining Hr time_to_fill Ht
  sorry

end pipe_cistern_problem_l644_644151


namespace find_tan_phi_l644_644930

-- Define the known sides of the triangle
def side_a : ℝ := 13
def side_b : ℝ := 14
def side_c : ℝ := 15

-- Definition for the semi-perimeter of the triangle
def semi_perimeter : ℝ := (side_a + side_b + side_c) / 2

-- Definition of the area of the triangle using Heron's formula
noncomputable def area : ℝ := 
  Real.sqrt (semi_perimeter * (semi_perimeter - side_a) * (semi_perimeter - side_b) * (semi_perimeter - side_c))

-- Calculation of cos(D) using the Law of Cosines
noncomputable def cos_angle_D : ℝ := 
  (side_a^2 + side_c^2 - side_b^2) / (2 * side_a * side_c)

-- Calculation of sin(D) from cos(D)
noncomputable def sin_angle_D : ℝ := 
  Real.sqrt (1 - cos_angle_D^2)

-- The given problem to find tan(φ)
theorem find_tan_phi 
  (tan_phi : ℝ) 
  (h : tan_phi == exact_value) : 
  (∃ φ, is_acute_angle φ ∧ φ = angle_between_bisectors side_a side_b side_c) → 
  ∃ tan_phi, tan φ = tan_phi := 
begin
  sorry
end

end find_tan_phi_l644_644930


namespace smallest_positive_multiple_17_more_11_multiple_43_l644_644136

theorem smallest_positive_multiple_17_more_11_multiple_43 :
  ∃ a : ℕ, 17 * a = 204 ∧ ∀ b : ℕ, (b > 0) → (17 * b ≡ 11 [MOD 43]) → (17 * b ≥ 17 * a) :=
begin
  sorry
end

end smallest_positive_multiple_17_more_11_multiple_43_l644_644136


namespace point_B_represents_2_l644_644792

theorem point_B_represents_2 (A B : ℤ) (hA : A = -2) (hMove : B = A + 4) : B = 2 :=
by
  rw [hA] at hMove
  rw [←hMove]
  norm_num
  -- the proof is skipped as requested
  sorry

end point_B_represents_2_l644_644792


namespace sum_m_n_l644_644000

noncomputable def f (x : ℝ) : ℝ := real.logb 2 (-|x| + 4)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 ^ (|x - 1|) + m + 1

theorem sum_m_n (m n : ℤ)
  (hf_domain : ∀ x ∈ set.Icc (m : ℝ) (n : ℝ), real.logb 2 (-|x| + 4) ∈ set.Icc 0 2)
  (hg_zero : ∃! x, g x m = 0) :
  m + n = 1 :=
sorry

end sum_m_n_l644_644000


namespace even_function_order_l644_644675

noncomputable def f (m : ℝ) (x : ℝ) := (m - 1) * x^2 + 6 * m * x + 2

theorem even_function_order (m : ℝ) (h_even : ∀ x : ℝ, f m (-x) = f m x) : 
  m = 0 ∧ f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry

end even_function_order_l644_644675


namespace ratio_y_share_to_total_l644_644561

theorem ratio_y_share_to_total
  (total_profit : ℝ)
  (diff_share : ℝ)
  (h_total : total_profit = 800)
  (h_diff : diff_share = 160) :
  ∃ (a b : ℝ), (b / (a + b) = 2 / 5) ∧ (|a - b| = (a + b) / 5) :=
by
  sorry

end ratio_y_share_to_total_l644_644561


namespace count_jianzhan_count_gift_boxes_l644_644529

-- Definitions based on given conditions
def firewood_red_clay : Int := 90
def firewood_white_clay : Int := 60
def electric_red_clay : Int := 75
def electric_white_clay : Int := 75
def total_red_clay : Int := 1530
def total_white_clay : Int := 1170

-- Proof problem 1: Number of "firewood firing" and "electric firing" Jianzhan produced
theorem count_jianzhan (x y : Int) (hx : firewood_red_clay * x + electric_red_clay * y = total_red_clay)
  (hy : firewood_white_clay * x + electric_white_clay * y = total_white_clay) : 
  x = 12 ∧ y = 6 :=
sorry

-- Definitions based on given conditions for Part 2
def total_jianzhan : Int := 18
def box_a_capacity : Int := 2
def box_b_capacity : Int := 6

-- Proof problem 2: Number of purchasing plans for gift boxes
theorem count_gift_boxes (m n : Int) (h : box_a_capacity * m + box_b_capacity * n = total_jianzhan) : 
  ∃ s : Finset (Int × Int), s.card = 4 ∧ ∀ (p : Int × Int), p ∈ s ↔ (p = (9, 0) ∨ p = (6, 1) ∨ p = (3, 2) ∨ p = (0, 3)) :=
sorry

end count_jianzhan_count_gift_boxes_l644_644529


namespace lowest_painting_cost_l644_644115

theorem lowest_painting_cost (x y z a b c : ℝ) (h1 : x < y) (h2 : y < z) (h3 : a < b) (h4 : b < c) :
  ∃ (cost : ℝ), cost = az + by + cx ∧ cost = min (ax + by + cz) (min (az + by + cx) (min (ay + bz + cx) (ay + bx + cz)))
  :=
sorry

end lowest_painting_cost_l644_644115


namespace percentage_increase_chef_vs_dishwasher_l644_644581

-- Define the conditions
def manager_wage : ℝ := 8.50
def dishwasher_wage : ℝ := manager_wage / 2
def chef_wage : ℝ := manager_wage - 3.40

-- Define the percentage increase calculation
def percentage_increase : ℝ := ((chef_wage - dishwasher_wage) / dishwasher_wage) * 100

-- The theorem to prove
theorem percentage_increase_chef_vs_dishwasher : percentage_increase = 20 := 
by 
  sorry

end percentage_increase_chef_vs_dishwasher_l644_644581


namespace correct_operation_l644_644507

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l644_644507


namespace number_of_pieces_l644_644036

def area_of_pan (length : ℕ) (width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

theorem number_of_pieces (length width side : ℕ) (h_length : length = 24) (h_width : width = 15) (h_side : side = 3) :
  (area_of_pan length width) / (area_of_piece side) = 40 :=
by
  rw [h_length, h_width, h_side]
  sorry

end number_of_pieces_l644_644036


namespace mating_season_male_alligators_l644_644392

theorem mating_season_male_alligators
  (ratio_m_af_jf : 2 = 2) 
  (af_non_season : ℕ) (h_af_non_season : af_non_season = 15) 
  (double_af : af_non_season * 2 = 30)
  (max_alligator_population : ℕ) (h_max : max_alligator_population = 200)
  (ratio_turtles : 3 = 3)
  : let m := (2 * af_non_season) / 3 -- given ratio M / AF = 2 / 3 in non-mating season
        jf := (5 * af_non_season) / 3 -- given ratio JF / AF = 5 / 3 in non-mating season
        total_mating_season := m + double_af + jf
in total_mating_season ≤ max_alligator_population ∧ m = 10 :=
by sorry

end mating_season_male_alligators_l644_644392


namespace cost_of_300_pencils_in_dollars_l644_644818

theorem cost_of_300_pencils_in_dollars :
  ∀ (cost_per_pencil : ℕ) (number_of_pencils : ℕ) (cents_in_dollar : ℕ),
    cost_per_pencil = 5 →
    number_of_pencils = 300 →
    cents_in_dollar = 200 →
    (number_of_pencils * cost_per_pencil) / cents_in_dollar = 7.5 :=
by
  assume cost_per_pencil number_of_pencils cents_in_dollar
  assume h1 : cost_per_pencil = 5
  assume h2 : number_of_pencils = 300
  assume h3 : cents_in_dollar = 200
  sorry

end cost_of_300_pencils_in_dollars_l644_644818


namespace largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l644_644554

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l644_644554


namespace pow_div_pow_eq_result_l644_644217

theorem pow_div_pow_eq_result : 13^8 / 13^5 = 2197 := by
  sorry

end pow_div_pow_eq_result_l644_644217


namespace isosceles_triangle_l644_644402

theorem isosceles_triangle {a b R : ℝ} {α β : ℝ} 
  (h : a * Real.tan α + b * Real.tan β = (a + b) * Real.tan ((α + β) / 2))
  (ha : a = 2 * R * Real.sin α) (hb : b = 2 * R * Real.sin β) :
  α = β := 
sorry

end isosceles_triangle_l644_644402


namespace average_pages_per_hour_l644_644564

theorem average_pages_per_hour 
  (P : ℕ) (H : ℕ) (hP : P = 30000) (hH : H = 150) : 
  P / H = 200 := 
by 
  sorry

end average_pages_per_hour_l644_644564


namespace solve_congruences_l644_644810

theorem solve_congruences (x : ℤ) (h1 : x ≡ 2 [MOD 7]) (h2: x ≡ 3 [MOD 6]) : x ≡ 9 [MOD 42] :=
sorry

end solve_congruences_l644_644810


namespace estimate_M_l644_644597

noncomputable def M : ℝ := 8 + 8 * real.root 3 4

theorem estimate_M : M ≈ 18.528592 := sorry

end estimate_M_l644_644597


namespace room_height_is_12_l644_644820

theorem room_height_is_12
  (length width cost_per_sqft total_cost door_height door_width window_height window_width window_count : ℝ)
  (H : ℝ) 
  (room_dims : length = 25 ∧ width = 15)
  (cost_info : cost_per_sqft = 5 ∧ total_cost = 4530)
  (door_dims : door_height = 6 ∧ door_width = 3)
  (window_dims : window_height = 4 ∧ window_width = 3 ∧ window_count = 3):
  H = 12 :=
by
  have perimeter : ℝ := 2 * (length + width)
  have total_wall_area : ℝ := perimeter * H
  have door_area : ℝ := door_height * door_width
  have window_area : ℝ := window_height * window_width
  have total_window_area : ℝ := window_count * window_area
  have area_to_be_whitewashed : ℝ := total_wall_area - (door_area + total_window_area)
  have total_cost_eq : total_cost = area_to_be_whitewashed * cost_per_sqft
  sorry

end room_height_is_12_l644_644820


namespace speed_of_faster_train_l644_644122

theorem speed_of_faster_train 
  (equal_length : ∀ (a b : ℝ), a = b)
  (slower_speed : ℝ)
  (passing_time_seconds : ℝ)
  (train_length_meters : ℝ) :
  slower_speed = 36 ∧ passing_time_seconds = 72 ∧ train_length_meters = 100 → 
  ∃ (v : ℝ), v = 46 :=
by
  intro h
  let slower_speed := 36
  let passing_time_seconds := 72
  let train_length_meters := 100
  have train_length_km := (train_length_meters / 1000)
  have total_distance_km := 2 * train_length_km
  have passing_time_hours := (passing_time_seconds / 3600)
  have relative_speed := total_distance_km / passing_time_hours
  have relative_speed_km_hr := 10
  have faster_speed := relative_speed_km_hr + slower_speed
  use faster_speed
  exact eq_of_heq (by decide) -- This step can simply assert the correctness sho.

  sorry -- Placeholder for additional proof steps necessary for Lean check.

end speed_of_faster_train_l644_644122


namespace prod_seq_2014_l644_644660

variable {a : ℕ → ℝ}

def a₁ : ℝ := 2

def a_(n+1) (n : ℕ) (a_n : ℝ) : ℝ := (1 + a_n) / (1 - a_n)

noncomputable def prod_seq (n : ℕ) : ℝ :=
  ∏ i in Finset.range (n + 1), a (i + 1)

theorem prod_seq_2014 : prod_seq 2014 = -6 := by
  sorry

end prod_seq_2014_l644_644660


namespace time_for_b_to_complete_lap_l644_644169

theorem time_for_b_to_complete_lap
  (speed_A : ℝ) (time_to_meet : ℝ)
  (time_A_per_lap : ℝ := 6)
  (time_B_rest : ℝ := 8)
  (meet_time : time_to_meet = 4) :
  speed_A = 1 / time_A_per_lap →
  (∃ time_per_lap_B : ℝ, time_per_lap_B = time_to_meet + time_B_rest) :=
by
  intro speed_A_def
  use (time_to_meet + time_B_rest)
  sorry

end time_for_b_to_complete_lap_l644_644169


namespace problem_statement_l644_644689

variables (a b : ℝ^3) (n n1 n2 : ℝ^3) (A B C : ℝ^3) (u t : ℝ)

def dot_product (v1 v2 : ℝ^3) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def line_plane_perpendicular (a : ℝ^3) (n : ℝ^3) : Prop :=
  dot_product a n = 0

def planes_parallel (n1 n2 : ℝ^3) : Prop :=
  ¬ ∃ λ : ℝ, n1 = λ • n2

def plane_contains_points (n : ℝ^3) (A B C : ℝ^3) : Prop :=
  let ab := B - A in
  let bc := C - B in
  dot_product n ab = 0 ∧ dot_product n bc = 0

def line_perpendicular_to_line (a b : ℝ^3) : Prop :=
  dot_product a b = 0

theorem problem_statement :
  forall (a b : ℝ^3) (n n1 n2 : ℝ^3) (A B C : ℝ^3) (u t : ℝ),
  (a = ⟨1, -1, 2⟩ → b = ⟨2, 1, -1/2⟩ → line_perpendicular_to_line a b) ∧
  (a = ⟨0, 1, -1⟩ → n = ⟨1, -1, -1⟩ → ¬ line_plane_perpendicular a n) ∧
  (n1 = ⟨0, 1, 3⟩ → n2 = ⟨1, 0, 2⟩ → planes_parallel n1 n2) ∧
  (A = ⟨1, 0, -1⟩ → B = ⟨0, -1, 0⟩ → C = ⟨-1, 2, 0⟩ →
    u = 1/3 → t = 4/3 → (u + t = 5/3) → plane_contains_points ⟨1, u, t⟩ A B C)
: 0 :=
sorry

end problem_statement_l644_644689


namespace range_of_a_l644_644330

theorem range_of_a (a : ℝ) (h : log a 3 < 1) : a > 3 ∨ (0 < a ∧ a < 1) := 
    sorry

end range_of_a_l644_644330


namespace min_dwarfs_for_no_empty_neighbor_l644_644061

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l644_644061


namespace flower_pots_sequence_count_l644_644946

-- Define the conditions
def initial_pots_left : ℕ := 2  -- Initial number of pots on the left
def initial_pots_right : ℕ := 2  -- Initial number of pots on the right

-- Statement of the proof problem
theorem flower_pots_sequence_count :
  ∃ (count : ℕ), count = 6 ∧
  let pots_left := initial_pots_left in
  let pots_right := initial_pots_right in
  -- Define the rule that Han Mei can only move the nearest pot from the chosen side
  true := -- Omitting further details for rule formalization
sorry

end flower_pots_sequence_count_l644_644946


namespace possible_values_of_f_2021_l644_644365

noncomputable def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2021}

def f (n : ℕ) : ℕ := sorry -- the actual function is not needed for the statement

def functionPower (f : ℕ → ℕ) (n : ℕ) (k : ℕ) : ℕ := Nat.iterate f k n 

theorem possible_values_of_f_2021 (h : ∀ n ∈ S, functionPower f n n = n) :
  ∃ t, t ∈ {43, 86, 129, 172, 215, 258, 301, 344, 387, 430, 473, 516, 559, 602, 645, 688, 731, 774, 817, 860, 903, 946, 989, 1032, 1075, 1118, 1161, 1204, 1247, 1290, 1333, 1376, 1419, 1462, 1505, 1548, 1591, 1634, 1677, 1720, 1763, 1806, 1849, 1892, 1935, 1978, 2021} ∧ f 2021 = t :=
sorry

end possible_values_of_f_2021_l644_644365


namespace range_of_a_l644_644699

theorem range_of_a (a : ℝ) (f g : ℝ → ℝ)
  (h₁ : ∀ x ∈ Icc 1 3, f x = x + a)
  (h₂ : ∀ x ∈ Icc 1 4, g x = x + 4 / x)
  (h₃ : ∀ x₁ ∈ Icc 1 3, ∃ x₂ ∈ Icc 1 4, f x₁ ≥ g x₂) :
  a ≥ 3 :=
sorry

end range_of_a_l644_644699


namespace sufficient_not_necessary_l644_644010

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) : (a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) :=
by
  sorry

end sufficient_not_necessary_l644_644010


namespace quadratic_real_roots_condition_l644_644641

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by
  sorry

end quadratic_real_roots_condition_l644_644641


namespace firefighters_time_to_extinguish_fire_l644_644545

theorem firefighters_time_to_extinguish_fire (gallons_per_minute_per_hose : ℕ) (total_gallons : ℕ) (number_of_firefighters : ℕ)
  (H1 : gallons_per_minute_per_hose = 20)
  (H2 : total_gallons = 4000)
  (H3 : number_of_firefighters = 5): 
  (total_gallons / (gallons_per_minute_per_hose * number_of_firefighters)) = 40 := 
by 
  sorry

end firefighters_time_to_extinguish_fire_l644_644545


namespace minimum_PA_PF_l644_644312

noncomputable def parabola_focus : Point := ⟨0, 1⟩
noncomputable def point_A : Point := ⟨-1, 8⟩
def parabola (p : Point) : Prop := p.1 ^ 2 = 4 * p.2

theorem minimum_PA_PF {P : Point} (hP : parabola P) :
  let PA := dist P point_A,
      PF := dist P parabola_focus in
  PA + PF >= 9 :=
by sorry

end minimum_PA_PF_l644_644312


namespace same_graph_log_eq_x_l644_644872

noncomputable def eqn_graph_same (a : ℝ) (x : ℝ) : Prop :=
    ∀ (x : ℝ), y = x <-> y = log a (a^x)

theorem same_graph_log_eq_x (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
    eqn_graph_same a x := 
by
  sorry

end same_graph_log_eq_x_l644_644872


namespace curveC1_cartesian_pointP_cartesian_coordinates_line_l_inclination_intersection_l644_644748

noncomputable def curveC1_parametric (m : ℝ) : ℝ × ℝ :=
  (sqrt m + 1 / sqrt m, sqrt m - 1 / sqrt m)

theorem curveC1_cartesian (m : ℝ) : ∃ m, (fst (curveC1_parametric m)) ^ 2 - (snd (curveC1_parametric m)) ^ 2 = 4 :=
begin
  use m,
  sorry
end

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * cos theta, rho * sin theta)

theorem pointP_cartesian_coordinates (θ := π / 3) 
  (ρ_calculated := 4 * sin (θ - π / 6)) :
  polar_to_cartesian ρ_calculated θ = (1, sqrt 3) :=
begin
  have hρ := calc 4 * sin (π / 3 - π / 6) = 4 * sin (π / 6) : by simp
                           ... = 4 * 1 / 2 : by simp
                           ... = 2 : by ring,
  unfold polar_to_cartesian,
  rw [hρ, cos_pi_div_three, sin_pi_div_three],
  simp,
  sorry
end

theorem line_l_inclination_intersection 
  (P : ℝ × ℝ := (1, sqrt 3))
  (inclination : ℝ := 2 * π / 3)
  : (∃ (A B : ℝ × ℝ),  curveC1_cartesian A → curveC1_cartesian B → 
  |dist P A + dist P B| = 8) :=
begin
  sorry
end

end curveC1_cartesian_pointP_cartesian_coordinates_line_l_inclination_intersection_l644_644748


namespace cone_surface_area_volume_ineq_l644_644836

theorem cone_surface_area_volume_ineq
  (A V r a m : ℝ)
  (hA : A = π * r * (r + a))
  (hV : V = (1/3) * π * r^2 * m)
  (hPythagoras : a^2 = r^2 + m^2) :
  A^3 ≥ 72 * π * V^2 := 
by
  sorry

end cone_surface_area_volume_ineq_l644_644836


namespace second_derivative_correct_l644_644274

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a - sin x

theorem second_derivative_correct (a x : ℝ) : 
  (derivative (derivative (f a))) x = -cos x :=
by
  sorry

end second_derivative_correct_l644_644274


namespace find_a_l644_644305

noncomputable def f (x : ℝ) : ℝ := Math.log 2 (x - 1/x)

theorem find_a 
  (a : ℝ) 
  (h1 : ∀ x, x ∈ Icc a (Real.infinity) → f x ∈ Icc 0 (Real.infinity))
  (h2 : ∀ x y, x ∈ Icc a (Real.infinity) → y ∈ Icc a (Real.infinity) → x ≤ y → f x ≤ f y)
  (h3 : a > 0)
  (h4 : a - 1/a = 1) :
  a = (Math.sqrt 5 + 1) / 2 :=
sorry

end find_a_l644_644305


namespace angle_bisector_intersections_form_rhombus_l644_644090

variable (A B C D P Q M N R S : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P] [Inhabited Q]
variable [Inhabited M] [Inhabited N] [Inhabited R] [Inhabited S]

-- Definitions for points and inscribed quadrilateral
def inscribed_quadrilateral (A B C D : Type) := sorry

-- Points P and Q as intersections of extensions
def intersect_extensions (A B C D P Q : Type) := sorry

-- Points of intersection of angle bisectors forming rhombus
def form_rhombus (M N R S : Type) := sorry

-- Conditions involved in the problem
variable h1: inscribed_quadrilateral A B C D
variable h2: intersect_extensions A B C D P Q
variable h3: form_rhombus M N R S

-- Final theorem statement
theorem angle_bisector_intersections_form_rhombus :
  inscribed_quadrilateral A B C D →
  intersect_extensions A B C D P Q →
  form_rhombus M N R S :=
by
  assume h1 h2 h3
  sorry

end angle_bisector_intersections_form_rhombus_l644_644090


namespace ap_nth_term_linear_gp_nth_term_exponential_l644_644131

theorem ap_nth_term_linear (a₁ d : ℝ) : ∀ (n : ℕ), ∃ A B, a₁ + (n-1)*d = A*n + B :=
by
  sorry

theorem gp_nth_term_exponential (a₁ r : ℝ) : ∀ (n : ℕ), ∃ C D, a₁ * r^(n-1) = C*r^n := 
by
  sorry

end ap_nth_term_linear_gp_nth_term_exponential_l644_644131


namespace factorization1_factorization2_l644_644619

-- Definition of the first polynomial and its factorization
def polynomial1 := 5 * x^2 + 6 * x * y - 8 * y^2
def factorized_poly1 := (5 * x - 4 * y) * (x + 2 * y)

theorem factorization1 (x y : ℝ) : polynomial1 = factorized_poly1 :=
by
  sorry

-- Definition of the second polynomial and its factorization
def polynomial2 (a : ℝ) := x^2 + 2 * x - 15 - a * x - 5 * a
def factorized_poly2 (a : ℝ) := (x + 5) * (x - (3 + a))

theorem factorization2 (x a : ℝ) : polynomial2 a = factorized_poly2 a :=
by
  sorry

end factorization1_factorization2_l644_644619


namespace sum_of_two_squares_iff_double_sum_of_two_squares_l644_644398

theorem sum_of_two_squares_iff_double_sum_of_two_squares (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_two_squares_iff_double_sum_of_two_squares_l644_644398


namespace min_dwarfs_l644_644069

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l644_644069


namespace correct_operation_l644_644489

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l644_644489


namespace biographies_percentage_before_purchase_l644_644363

variable (B T : ℕ)

-- Conditions
def condition1 : Prop :=
  B + 0.8823529411764707 * B = 0.32 * (T + 0.8823529411764707 * B)

def condition2 : Prop :=
  0.8823529411764707 * B = 0.8823529411764707 * B

-- Statement to be proved
theorem biographies_percentage_before_purchase (h1 : condition1 B T) (h2 : condition2 B T) :
  (B / T) = 0.2 :=
sorry

end biographies_percentage_before_purchase_l644_644363


namespace trains_meet_time_approx_l644_644123

noncomputable def length_train_1 := 90 -- in meters
noncomputable def length_train_2 := 100 -- in meters
noncomputable def initial_distance := 200 -- in meters
noncomputable def speed_train_1 := 71 * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def speed_train_2 := 89 * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def time_to_meet := (length_train_1 + length_train_2 + initial_distance) / (speed_train_1 + speed_train_2)

theorem trains_meet_time_approx : time_to_meet ≈ 8.7755 := 
by sorry

end trains_meet_time_approx_l644_644123


namespace individual_wages_l644_644535

-- Define individual work rates
def work_rate_A := 1 / 10
def work_rate_B := 1 / 15
def work_rate_C := 1 / 20

-- Define total wage
def total_wage := 6600

-- Define the wages distribution percentages
def wage_percentage_A := 0.50
def wage_percentage_B := 0.30
def wage_percentage_C := 0.20

-- Define the expected individual wages
def wage_A := wage_percentage_A * total_wage
def wage_B := wage_percentage_B * total_wage
def wage_C := wage_percentage_C * total_wage

-- Theorem to prove the individual wages
theorem individual_wages : 
  wage_A = 3300 ∧ wage_B = 1980 ∧ wage_C = 1320 :=
by
  unfold wage_A wage_B wage_C
  unfold wage_percentage_A wage_percentage_B wage_percentage_C total_wage
  norm_num
  split; norm_num
  split; norm_num
  sorry

end individual_wages_l644_644535


namespace average_marks_of_all_students_l644_644086

theorem average_marks_of_all_students (n1 n2 : ℕ) (m1 m2 : ℕ) (h1 : n1 = 30) (h2 : n2 = 50) (h3 : m1 = 50) (h4 : m2 = 60) :
  (n1 * m1 + n2 * m2) / (n1 + n2) = 56.25 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_marks_of_all_students_l644_644086


namespace average_marks_l644_644085

theorem average_marks (A : ℝ) :
  let marks_first_class := 25 * A
  let marks_second_class := 30 * 60
  let total_marks := 55 * 50.90909090909091
  marks_first_class + marks_second_class = total_marks → A = 40 :=
by
  sorry

end average_marks_l644_644085


namespace age_inconsistency_l644_644011

variable (x y z w : ℕ)

axiom cond1 : x + y = y + z + 19
axiom cond2 : x + w = z + w - 12
axiom cond3 : w = y + 5

theorem age_inconsistency : False :=
by
  have h1 : x = z + 19 := by linarith [cond1]
  have h2 : x = z - 12 := by linarith [cond2]
  have h : z + 19 = z - 12 := by linarith [h1, h2]
  linarith

end age_inconsistency_l644_644011


namespace min_dwarfs_for_no_empty_neighbor_l644_644059

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l644_644059


namespace win_probability_greater_than_half_l644_644462

-- Define the probability space and conditions
variable (boxes : Type) [Fintype boxes] [DecidableEq boxes]
variable (prize boxA boxB boxC : boxes)
variable (initial_choice : boxes)
variable (is_prize : boxes → Prop)
variable [DecidablePred is_prize]

-- Conditions
axiom h1 : boxA ≠ boxB ∧ boxA ≠ boxC ∧ boxB ≠ boxC
axiom h2 : is_prize prize
axiom h3 : ∀ b, is_prize b ↔ (b = prize)

-- Probability that you initially pick the box with the prize
def prob_initial_choice : ℝ := 1 / 3

-- Probability that the prize is in one of the other two boxes if not in initial choice
def prob_other_boxes : ℝ :=
  if initial_choice = boxA then 2 / 3 else
  if initial_choice = boxB then 2 / 3 else
  2 / 3

-- Host's action of opening an empty box
axiom h4 : ∃ b, b ≠ initial_choice ∧ ¬ is_prize b ∧ b ≠ prize

-- Revised probability after host opens an empty box
def prob_switch_win : ℝ :=
  if initial_choice = boxA then 2 / 3 else
  if initial_choice = boxB then 2 / 3 else
  2 / 3

-- Prove that switching gives a higher probability of winning than sticking
theorem win_probability_greater_than_half : prob_switch_win > 1 / 2 :=
by
  intro boxes prize boxA boxB boxC initial_choice is_prize h1 h2 h3 h4
  simp [prob_switch_win]
  linarith

end win_probability_greater_than_half_l644_644462


namespace external_diagonal_not_lengths_5_6_9_l644_644480

theorem external_diagonal_not_lengths_5_6_9 (a b c : ℕ) :
  ¬(a^2 + b^2 = 5^2 ∧ b^2 + c^2 = 6^2 ∧ a^2 + c^2 = 9^2) → 
  ∀ a b c : ℕ, ¬(a^2 + b^2 ∈ {5^2, 6^2, 9^2} ∧ b^2 + c^2 ∈ {5^2, 6^2, 9^2} ∧ a^2 + c^2 ∈ {5^2, 6^2, 9^2}) :=
by sorry

end external_diagonal_not_lengths_5_6_9_l644_644480


namespace functional_equation_solution_l644_644997

noncomputable def f : ℕ+ → ℕ+ := sorry

theorem functional_equation_solution :
  (∀ m n : ℕ+, f (m^2 + f n) = f m^2 + n) →
  (∀ x : ℕ+, f x = x) := 
begin
  intros h x,
  sorry
end

end functional_equation_solution_l644_644997


namespace smallest_positive_multiple_of_32_l644_644475

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n = 32 * k ∧ n = 32 := by
  use 32
  constructor
  · exact Nat.zero_lt_succ 31
  · use 1
    constructor
    · exact Nat.zero_lt_succ 0
    · constructor
      · rfl
      · rfl

end smallest_positive_multiple_of_32_l644_644475


namespace integral_solution_l644_644526

noncomputable def integral_problem : ℝ :=
  ∫ x in (0 : ℝ)..(Real.pi / 2), sin x / (1 + sin x + cos x)

theorem integral_solution : integral_problem = - (1 / 2) * Real.log 2 + (Real.pi / 4) :=
by
  sorry

end integral_solution_l644_644526


namespace solve_abs_equation_l644_644075

theorem solve_abs_equation :
  (|y - 8| + 3 * y = 15) → (y = 23 / 4 ∨ y = 7 / 2) :=
by
  sorry

end solve_abs_equation_l644_644075


namespace unique_triangle_condition_l644_644455

theorem unique_triangle_condition (AB BC AC : ℝ) (angle_A angle_B angle_C : ℝ) :
  (\(AB = 2 \land BC = 6 \land AC = 9 \implies False\)) \and
  (\(AB = 7 \land BC = 5 \land angle_A = 30\implies False\)) \and
  (\(angle_A = 50 \land angle_B = 60 \land angle_C = 70 \implies False\)) \and
  (\(AC = 3.5 \land BC = 4.8 \land angle_C = 70\))

end unique_triangle_condition_l644_644455


namespace math_problem_l644_644494

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l644_644494


namespace container_weight_l644_644550

noncomputable def weight_in_pounds : ℝ := 57 + 3/8
noncomputable def weight_in_ounces : ℝ := weight_in_pounds * 16
noncomputable def number_of_containers : ℝ := 7
noncomputable def ounces_per_container : ℝ := weight_in_ounces / number_of_containers

theorem container_weight :
  ounces_per_container = 131.142857 :=
by sorry

end container_weight_l644_644550


namespace inequality_proof_l644_644273

variable (x : ℝ)

theorem inequality_proof (h1 : 3/2 ≤ x) (h2 : x ≤ 5) : 
  2 * real.sqrt(x + 1) + real.sqrt(2 * x - 3) + real.sqrt(15 - 3 * x) < 2 * real.sqrt 19 :=
sorry

end inequality_proof_l644_644273


namespace muffins_total_is_83_l644_644205

-- Define the given conditions.
def initial_muffins : Nat := 35
def additional_muffins : Nat := 48

-- Define the total number of muffins.
def total_muffins : Nat := initial_muffins + additional_muffins

-- Statement to prove.
theorem muffins_total_is_83 : total_muffins = 83 := by
  -- Proof is omitted.
  sorry

end muffins_total_is_83_l644_644205


namespace houses_with_white_mailboxes_l644_644549

theorem houses_with_white_mailboxes (total_mail : ℕ) (total_houses : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ)
    (h1 : total_mail = 48) (h2 : total_houses = 8) (h3 : red_mailboxes = 3) (h4 : mail_per_house = 6) :
  total_houses - red_mailboxes = 5 :=
by
  sorry

end houses_with_white_mailboxes_l644_644549


namespace heads_of_lettuce_l644_644470

-- Define the conditions as constants.
constant customers_per_month : Nat := 500
constant lettuce_revenue : Nat := 2000
constant price_per_lettuce : Nat := 1

-- Define the function to calculate the number of heads of lettuce per customer.
def heads_of_lettuce_per_customer (customers_per_month : Nat) (lettuce_revenue : Nat) (price_per_lettuce : Nat) : Nat :=
  lettuce_revenue / (customers_per_month * price_per_lettuce)

-- Prove the result.
theorem heads_of_lettuce (customers_per_month lettuce_revenue price_per_lettuce : Nat) :
  heads_of_lettuce_per_customer customers_per_month lettuce_revenue price_per_lettuce = 4 :=
by
  unfold heads_of_lettuce_per_customer
  simp
  sorry

end heads_of_lettuce_l644_644470


namespace theta_in_first_quadrant_l644_644648

noncomputable def quadrant_of_theta (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) : ℕ :=
  if 0 < Real.sin theta ∧ 0 < Real.cos theta then 1 else sorry

theorem theta_in_first_quadrant (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) :
  quadrant_of_theta theta h1 h2 = 1 :=
by
  sorry

end theta_in_first_quadrant_l644_644648


namespace limes_given_l644_644603

theorem limes_given (original_limes now_limes : ℕ) (h1 : original_limes = 9) (h2 : now_limes = 5) : (original_limes - now_limes = 4) := 
by
  sorry

end limes_given_l644_644603


namespace number_of_students_above_120_l644_644927

noncomputable theory
open_locale classical

-- Given conditions
def total_students : ℕ := 1000
def score_distribution (ξ : ℝ) : Prop := ∀ (μ σ : ℝ), ξ ~ N(μ = 100, σ^2)
def probability_interval : Prop := P(80 ≤ ξ ∧ ξ ≤ 100) = 0.45

-- Problem statement
theorem number_of_students_above_120 :
  (∀ (ξ : ℝ), score_distribution ξ) →
  probability_interval →
  ∑ x in (multiset.filter (≥ 120) (multiset.range total_students)), 1 = 50 :=
by
  intro score_distribution probability_interval
  sorry

end number_of_students_above_120_l644_644927


namespace win_probability_greater_than_half_l644_644461

-- Define the probability space and conditions
variable (boxes : Type) [Fintype boxes] [DecidableEq boxes]
variable (prize boxA boxB boxC : boxes)
variable (initial_choice : boxes)
variable (is_prize : boxes → Prop)
variable [DecidablePred is_prize]

-- Conditions
axiom h1 : boxA ≠ boxB ∧ boxA ≠ boxC ∧ boxB ≠ boxC
axiom h2 : is_prize prize
axiom h3 : ∀ b, is_prize b ↔ (b = prize)

-- Probability that you initially pick the box with the prize
def prob_initial_choice : ℝ := 1 / 3

-- Probability that the prize is in one of the other two boxes if not in initial choice
def prob_other_boxes : ℝ :=
  if initial_choice = boxA then 2 / 3 else
  if initial_choice = boxB then 2 / 3 else
  2 / 3

-- Host's action of opening an empty box
axiom h4 : ∃ b, b ≠ initial_choice ∧ ¬ is_prize b ∧ b ≠ prize

-- Revised probability after host opens an empty box
def prob_switch_win : ℝ :=
  if initial_choice = boxA then 2 / 3 else
  if initial_choice = boxB then 2 / 3 else
  2 / 3

-- Prove that switching gives a higher probability of winning than sticking
theorem win_probability_greater_than_half : prob_switch_win > 1 / 2 :=
by
  intro boxes prize boxA boxB boxC initial_choice is_prize h1 h2 h3 h4
  simp [prob_switch_win]
  linarith

end win_probability_greater_than_half_l644_644461


namespace range_of_k_l644_644703

noncomputable def line (k : ℝ) : ℝ → ℝ := λ x, k * x + 3
noncomputable def circle : ℝ × ℝ → Prop := 
  λ p, (p.1 - 2) ^ 2 + (p.2 - 3) ^ 2 = 4

theorem range_of_k (M N : ℝ × ℝ) (k : ℝ) : 
  (line k M.1 = M.2) ∧ (line k N.1 = N.2) ∧ 
  (circle M) ∧ (circle N) ∧ 
  dist M N ≥ 2 * Real.sqrt 3 → 
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by sorry

end range_of_k_l644_644703


namespace An_is_integer_l644_644676

-- Define the conditions for the problem
variables (a b : ℕ) (θ : ℝ)
variable (n : ℕ)

-- Assume the given conditions
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom a_greater_than_b : a > b
axiom sin_theta_def : sin θ = 2 * a * b / (a^2 + b^2)
axiom theta_range : 0 < θ ∧ θ < π / 2

-- The theorem to be proved
theorem An_is_integer : ∀ n : ℕ, ∃ k : ℤ, (a^2 + b^2)^n * sin θ = k :=
by
  sorry

end An_is_integer_l644_644676


namespace smallest_possible_value_l644_644093

theorem smallest_possible_value (x : ℕ) (m : ℕ) :
  (x > 0) →
  (Nat.gcd 36 m = x + 3) →
  (Nat.lcm 36 m = x * (x + 3)) →
  m = 12 :=
by
  sorry

end smallest_possible_value_l644_644093


namespace sum_of_positive_integers_l644_644965

open Real

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n

def expression_is_integer (n : ℕ) : Prop :=
  is_integer (9 * sqrt n + 4 * sqrt (n + 2) - 3 * sqrt (n + 16))

theorem sum_of_positive_integers :
  ∑ k in {n : ℕ | expression_is_integer n}, n = 18 :=
sorry

end sum_of_positive_integers_l644_644965


namespace equilateral_triangle_fixed_area_equilateral_triangle_max_area_l644_644734

theorem equilateral_triangle_fixed_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = minimized ∨ a + b + c = minimized ∨ a^2 + b^2 + c^2 = minimized ∨ R = minimized) →
    (a = b ∧ b = c) :=
by
  sorry

theorem equilateral_triangle_max_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = fixed ∨ a + b + c = fixed ∨ a^2 + b^2 + c^2 = fixed ∨ R = fixed) →
  (Δ = maximized) →
    (a = b ∧ b = c) :=
by
  sorry

end equilateral_triangle_fixed_area_equilateral_triangle_max_area_l644_644734


namespace rectangle_area_at_stage_8_l644_644189

-- Declare constants for the conditions.
def square_side_length : ℕ := 4
def number_of_stages : ℕ := 8
def area_of_single_square : ℕ := square_side_length * square_side_length

-- The statement to prove
theorem rectangle_area_at_stage_8 : number_of_stages * area_of_single_square = 128 := by
  sorry

end rectangle_area_at_stage_8_l644_644189


namespace sin_cos_value_sin_minus_cos_value_l644_644672

open Real

-- Condition 
variable (x : ℝ) 
hypothesis h1 : sin(x + π) + cos(x - π) = 1 / 2
hypothesis h2 : 0 < x ∧ x < π

theorem sin_cos_value (h1 : sin (x + π) + cos (x - π) = 1 / 2) (h2 : 0 < x ∧ x < π) : 
  sin x * cos x = -3 / 8 :=
by 
  sorry

theorem sin_minus_cos_value (h1 : sin (x + π) + cos (x - π) = 1 / 2) (h2 : 0 < x ∧ x < π) : 
  sin x - cos x = sqrt 7 / 2 :=
by 
  sorry

end sin_cos_value_sin_minus_cos_value_l644_644672


namespace mowing_time_l644_644800

/-- 
Rena uses a mower to trim her "L"-shaped lawn which consists of two rectangular sections 
sharing one $50$-foot side. One section is $120$-foot by $50$-foot and the other is $70$-foot by 
$50$-foot. The mower has a swath width of $35$ inches with overlaps by $5$ inches. 
Rena walks at the rate of $4000$ feet per hour. 
Prove that it takes 0.95 hours for Rena to mow the entire lawn.
-/
theorem mowing_time 
  (length1 length2 width mower_swath overlap : ℝ) 
  (Rena_speed : ℝ) (effective_swath : ℝ) (total_area total_strips total_distance : ℝ)
  (h1 : length1 = 120)
  (h2 : length2 = 70)
  (h3 : width = 50)
  (h4 : mower_swath = 35 / 12)
  (h5 : overlap = 5 / 12)
  (h6 : effective_swath = mower_swath - overlap)
  (h7 : Rena_speed = 4000)
  (h8 : total_area = length1 * width + length2 * width)
  (h9 : total_strips = (length1 + length2) / effective_swath)
  (h10 : total_distance = total_strips * width) : 
  (total_distance / Rena_speed = 0.95) :=
by sorry

end mowing_time_l644_644800


namespace correct_operation_l644_644506

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l644_644506


namespace cashew_price_per_pound_l644_644216

theorem cashew_price_per_pound :
  ∀ (peanut_price cashew_weight total_weight total_price : ℝ),
  peanut_price = 2 ∧
  total_weight = 25 ∧
  total_price = 92 ∧
  cashew_weight = 11 →
  (64 / 11) = (total_price - (total_weight - cashew_weight) * peanut_price) / cashew_weight :=
by
  intros peanut_price cashew_weight total_weight total_price h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cashew_price_per_pound_l644_644216


namespace find_min_max_A_l644_644552

-- Define a 9-digit number B
def is_9_digit (B : ℕ) : Prop := B ≥ 100000000 ∧ B < 1000000000

-- Define a function that checks if a number is coprime with 24
def coprime_with_24 (B : ℕ) : Prop := Nat.gcd B 24 = 1

-- Define the transformation from B to A
def transform (B : ℕ) : ℕ := let b := B % 10 in b * 100000000 + (B / 10)

-- Lean 4 statement for the problem
theorem find_min_max_A :
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 999999998) ∧
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 166666667) :=
  by
    sorry -- Proof is omitted

end find_min_max_A_l644_644552


namespace fibonacci_inequality_l644_644833

def Fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | 2     => 2
  | n + 2 => Fibonacci (n + 1) + Fibonacci n

theorem fibonacci_inequality (n : ℕ) (h : n > 0) : 
  Real.sqrt (Fibonacci (n+1)) > 1 + 1 / Real.sqrt (Fibonacci n) := 
sorry

end fibonacci_inequality_l644_644833


namespace tennis_players_count_l644_644347

theorem tennis_players_count
  (total_members : ℕ)
  (badminton_players : ℕ)
  (non_players : ℕ)
  (both_sports_players : ℕ)
  (membersPlayingAtLeastOneSport := total_members - non_players)
  (only_badminton_players := badminton_players - both_sports_players) :
  total_members = 42 ∧ badminton_players = 20 ∧ non_players = 6 ∧ both_sports_players = 7 →
  (only_badminton_players + (membersPlayingAtLeastOneSport - only_badminton_players - both_sports_players) + both_sports_players = membersPlayingAtLeastOneSport) →
  (both_sports_players + (membersPlayingAtLeastOneSport - only_badminton_players - both_sports_players) = 23) :=
begin
  sorry
end

end tennis_players_count_l644_644347


namespace cosine_power_identity_l644_644403

theorem cosine_power_identity (n : ℕ) (h : 0 < n) (θ : ℝ) :
  (Real.cos θ) ^ n = (1 / (2 ^ n)) * ∑ k in Finset.range (n + 1), Nat.choose n k * Real.cos ((n - 2 * k) * θ) :=
by sorry

end cosine_power_identity_l644_644403


namespace polynomial_solution_l644_644626

noncomputable def is_solution (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2)

theorem polynomial_solution (P : ℝ → ℝ) : is_solution P → ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
  sorry

end polynomial_solution_l644_644626


namespace area_gray_region_l644_644961

-- Circle data
def centerC : ℝ × ℝ := (4, 4)
def radiusC : ℝ := 5
def centerD : ℝ × ℝ := (10, 4)
def radiusD : ℝ := 3

-- Main theorem statement
theorem area_gray_region : 
  24 - (25 * real.arccos (4 / 5) - 12 + 9 * real.arccos (1 / 3) - 8.484) 
  = 44.484 - 25 * real.arccos (4 / 5) - 9 * real.arccos (1 / 3) := 
by sorry

end area_gray_region_l644_644961


namespace win_probability_greater_than_half_l644_644460

/-- Define the boxes and initial choice -/
inductive Box
| A | B | C

def initial_choice : Box := Box.A

/-- Define the probabilities -/
def probability_of_box (b : Box) : ℝ :=
  match b with
  | Box.A => 1 / 3
  | Box.B => 1 / 3
  | Box.C => 1 / 3

/-- Host's action: reveals an empty box that is not initially chosen -/
def host_action (initial : Box) (prize : Box) : Box :=
  if initial = Box.A then
    if prize = Box.B then Box.C else Box.B
  else if initial = Box.B then
    if prize = Box.A then Box.C else Box.A
  else
    if prize = Box.A then Box.B else Box.A

/-- Probability after switching -/
def probability_after_switch (initial : Box) (prize : Box) : ℝ :=
  if initial = prize then 0 else 1

/-- Theorem stating the probability of winning after switching is greater than 1/2 -/
theorem win_probability_greater_than_half :
  ∀ (prize : Box), (probability_after_switch initial_choice prize) > 1 / 2 := by
  sorry

end win_probability_greater_than_half_l644_644460


namespace weight_of_new_person_is_correct_l644_644150

noncomputable def weight_new_person (increase_per_person : ℝ) (old_weight : ℝ) (group_size : ℝ) : ℝ :=
  old_weight + group_size * increase_per_person

theorem weight_of_new_person_is_correct :
  weight_new_person 7.2 65 10 = 137 :=
by
  sorry

end weight_of_new_person_is_correct_l644_644150


namespace problem_equivalence_l644_644315

variables (x : ℝ) (k : ℤ)

def a : ℝ × ℝ := (real.sqrt 3, real.cos(2 * x))

def b (n : ℝ) : ℝ × ℝ := (real.sin (2 * x), n)

def f (m n : ℝ) : ℝ := m * real.sin (2 * x) + n * real.cos (2 * x)

noncomputable def g : ℝ → ℝ := λ x, 2 * real.cos (2 * x)

def passes_through (p1 p2 : ℝ × ℝ) := 
  f ((real.sqrt 3) : ℝ) (1 : ℝ) (p1.1) = p1.2 ∧ f ((real.sqrt 3) : ℝ) (1 : ℝ) (p2.1) = p2.2

theorem problem_equivalence :
  passes_through (π / 12, real.sqrt 3) (2 * π / 3, -2) →
  ∃ (k : ℤ), ∀ (x : ℝ), 
    (-π / 2 + k * π < x ∧ x < k * π) ↔ (0 < (g x) ∧ g (x + 1 / (2 * π)) > g (x - 1 / (2 * π)))
:=
by
  intros h,
  sorry

end problem_equivalence_l644_644315


namespace strategy_exists_l644_644939

def chosen_integer : set ℤ := {n | 1 ≤ n ∧ n ≤ 2002}

def is_odd (n : ℤ) : Prop := n % 2 = 1

def probability_of_odd_guesses (winning_numbers total_numbers : ℤ) : ℚ :=
  (if total_numbers ≠ 0 then (winning_numbers : ℚ) / (total_numbers : ℚ) else 0)

theorem strategy_exists (winning_numbers : ℤ) (total_numbers : ℤ) :
  (probability_of_odd_guesses winning_numbers total_numbers > (2 / 3)) := by
  sorry

end strategy_exists_l644_644939


namespace smallest_value_other_integer_l644_644096

noncomputable def smallest_possible_value_b : ℕ :=
  by sorry

theorem smallest_value_other_integer (x : ℕ) (h_pos : x > 0) (b : ℕ) 
  (h_gcd : Nat.gcd 36 b = x + 3) (h_lcm : Nat.lcm 36 b = x * (x + 3)) :
  b = 108 :=
  by sorry

end smallest_value_other_integer_l644_644096


namespace solve_for_z_l644_644163

variable {x y z : ℝ}

theorem solve_for_z (h : 0.65 * x * y - z = 0.20 * 747.50) : z = 0.65 * x * y - 149.50 :=
by
  have h1 : 0.20 * 747.50 = 149.50 := by norm_num
  rw [h1] at h
  linarith

end solve_for_z_l644_644163


namespace circle_rectangle_positive_integers_l644_644087

theorem circle_rectangle_positive_integers (π : ℝ) (R l r : ℕ) : 
  (2 * π * R - 2 * (l + 6) = 2012) ∧ (l = R + r) 
  → ∃ c : ℕ, c = 1011 :=
by
  intro h,
  sorry

end circle_rectangle_positive_integers_l644_644087


namespace square_area_and_position_l644_644920

/-!
A square is given in the plane with consecutively placed vertices A, B, C, D and a point O.
It is known that OA = OC = 10, OD = 6√2, and the side length of the square does not exceed 3.
Find the area of the square. Is point O located outside or inside the square?
-/

-- Define the coordinates of the points and the required properties.
def point (α : Type*) := (x y : α)

variables {α : Type*} [linear_ordered_field α]
variables (A B C D O : point α)
variables (s : α)

-- Given conditions
def sq_vertices := -- (Angle and vertices construction)
s ≤ 3

def OA := (A.x - O.x)^2 + (A.y - O.y)^2 = 10^2
def OC := (C.x - O.x)^2 + (C.y - O.y)^2 = 10^2
def OD := (D.x - O.x)^2 + (D.y - O.y)^2 = (6 * real.sqrt 2)^2

-- Question to determine the area of the square and position of O
theorem square_area_and_position 
  (A B C D O : point ℝ)
  (s : ℝ)
  (OA : (A.x - O.x)^2 + (A.y - O.y)^2 = 100)
  (OC : (C.x - O.x)^2 + (C.y - O.y)^2 = 100)
  (OD : (D.x - O.x)^2 + (D.y - O.y)^2 = 72)
  (sq_vertices : s ≤ 3) :
  s^2 = 4 ∧ -- Area of the square
  (O.x, O.y) ∉ set.univ := -- O is outside the square
sorry

end square_area_and_position_l644_644920


namespace find_base_b_l644_644338

-- Define the base b and the relationship given in the problem
def is_base (b : ℕ) : Prop :=
  (2 * b + 2) ^ 2 = 5 * b^2 + b + 4

-- State the main theorem that we want to prove
theorem find_base_b : ∃ b : ℕ, is_base b ∧ b = 7 :=
by
  existsi 7
  split
  sorry
  rfl

end find_base_b_l644_644338


namespace math_problem_l644_644499

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l644_644499


namespace monotonic_intervals_max_value_of_a_l644_644692

noncomputable theory

open Real

-- Define the function f(x) = ln x + a * x^2 + (a + 2) * x + 1
def f (a x : ℝ) : ℝ := ln x + a * x^2 + (a + 2) * x + 1

-- Prove the monotonic interval of the function f(x)
theorem monotonic_intervals (a : ℝ) :
  (a ≥ 0 → ∀ x : ℝ, 0 < x → f a x is increasing on (0, ∞)) ∧
  (a < 0 → (∀ x : ℝ, 0 < x ∧ x < -1/a → f a x is increasing on (0, -1/a)) ∧
            (∀ x : ℝ, x > -1/a → f a x is decreasing on (-1/a, ∞))) :=
sorry

-- Prove the maximum value of a given that f(x) ≤ 0 for all x > 0 and a ∈ Z
theorem max_value_of_a :
  (∀ a : ℤ, (∀ x : ℝ, 0 < x → f a x ≤ 0) → a ≤ -2) :=
sorry

end monotonic_intervals_max_value_of_a_l644_644692


namespace min_dwarfs_l644_644068

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l644_644068


namespace cut_rectangles_into_square_and_rectangle_l644_644185

theorem cut_rectangles_into_square_and_rectangle (a b : ℝ) (N : ℕ) (h_a_lt_b : a < b) (h_N_pos : 0 < N)
  (h_non_square : a ≠ b) :
  ∃ (cuts : ℕ → ℤ), 
    (∀ i, 1 ≤ cuts i ≤ 2) ∧ 
    (∀ j, 1 ≤ cuts j ≤ V j) ∧ 
    (cuts.sum = N) :=
begin
  sorry,
end

end cut_rectangles_into_square_and_rectangle_l644_644185


namespace quadratic_inequality_solution_l644_644683

theorem quadratic_inequality_solution :
  ∀ (a b : ℝ),
    (∀ x : ℝ, -5 * x ^ 2 - 5 * x + b > 0 ↔ (-3 < x ∧ x < 2)) →
    (b = 30 ∧ a = -5) →
    (∀ x : ℝ, 30 * x ^ 2 - 5 * x - 5 > 0 ↔ (x < -1 / 3 ∨ 1 / 2 < x)) :=
by
  intro a b hsol hab
  rw [←hab.1, ←hab.2]
  sorry

end quadratic_inequality_solution_l644_644683


namespace minimum_dwarfs_l644_644046

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l644_644046


namespace remaining_integers_in_T_l644_644835

-- Defining the set T
def T : Set ℕ := { n | 1 ≤ n ∧ n ≤ 100 }

-- Predicate to check for multiples of a given number
def multiple_of (m n : ℕ) : Prop := ∃ k, n = k * m

-- Set of multiples of 3 in T
def multiples_of_3 : Set ℕ := { n | n ∈ T ∧ multiple_of 3 n }

-- Set of multiples of 4 in T
def multiples_of_4 : Set ℕ := { n | n ∈ T ∧ multiple_of 4 n }

-- Set of multiples of 5 in T
def multiples_of_5 : Set ℕ := { n | n ∈ T ∧ multiple_of 5 n }

-- Set of all multiples of 3, 4, or 5 in T
def multiples_of_3_4_or_5 : Set ℕ := multiples_of_3 ∪ multiples_of_4 ∪ multiples_of_5

-- The number of elements in a finite set
noncomputable def card (s : Set ℕ) : ℕ :=
  (Finset.filter (λ x, x ∈ s) (Finset.range 101)).card

-- The proof problem statement
theorem remaining_integers_in_T : card T - card multiples_of_3_4_or_5 = 40 :=
  by sorry

end remaining_integers_in_T_l644_644835


namespace intervals_of_monotonicity_harmonic_log_inequalities_l644_644690

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - (a * x - 1) / x

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x : ℝ, (1 <= x → deriv (fun x => (f x a)) x ≥ 0) ∧ (0 < x ∧ x < 1 → deriv (fun x => (f x a)) x ≤ 0)) :=
sorry

theorem harmonic_log_inequalities (n : ℕ) (h : 0 < n) :
  (∑ k in Finset.range n, 1 / (k + 2)) < Real.log(n + 1) ∧ Real.log(n + 1) < 1 + ∑ k in Finset.range n, 1 / (k + 1) := 
sorry

end intervals_of_monotonicity_harmonic_log_inequalities_l644_644690


namespace math_problem_l644_644497

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l644_644497


namespace intersection_point_x_value_l644_644235

theorem intersection_point_x_value : 
  (∃ (x y : ℝ), y = 3 * x + 4 ∧ y = 5 * x - 41) → x = 22.5 :=
by
  intro h
  cases' h with x hxy
  cases' hxy with y hy
  use y
  sorry

end intersection_point_x_value_l644_644235


namespace min_dwarfs_for_no_empty_neighbor_l644_644060

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l644_644060


namespace max_min_of_f_area_of_closed_figure_l644_644309

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * x^2 + 1

theorem max_min_of_f :
  (∀ x, f x ≤ 1) ∧ f 0 = 1 ∧ (∀ x, f x ≥ 5 / 6) ∧ f 1 = 5 / 6 :=
sorry

theorem area_of_closed_figure :
  ∫ x in 0..3 / 2, ((1 : ℝ) - f x) = 9 / 64 :=
sorry

end max_min_of_f_area_of_closed_figure_l644_644309


namespace tan_alpha_plus_pi_over_3_sin_cos_ratio_l644_644673

theorem tan_alpha_plus_pi_over_3
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  Real.tan (α + Real.pi / 3) = (48 - 25 * Real.sqrt 3) / 11 := 
sorry

theorem sin_cos_ratio
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17 :=
sorry

end tan_alpha_plus_pi_over_3_sin_cos_ratio_l644_644673


namespace ConjugateOfComplexNumber_l644_644420

noncomputable def conj (z : ℂ) : ℂ := complex.conj z

theorem ConjugateOfComplexNumber (z : ℂ) (h : z * (1 + complex.I) = 2 - 2 * complex.I) : 
  conj z = 2 * complex.I :=
  sorry

end ConjugateOfComplexNumber_l644_644420


namespace speed_in_still_water_l644_644182

theorem speed_in_still_water (upstream downstream : ℝ) 
  (h_up : upstream = 25) 
  (h_down : downstream = 45) : 
  (upstream + downstream) / 2 = 35 := 
by 
  -- Proof will go here
  sorry

end speed_in_still_water_l644_644182


namespace find_a2_is_arithmetic_find_k_l644_644280

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom a1 : a 1 = 1
axiom cond : ∀ n, (a (n + 1) - a n) / (a n * a (n + 1)) = 2 / (4 * S n - 1)

-- Statement for part (1)
theorem find_a2 : a 2 = 3 := sorry

-- Statement for part (2)
def b (n : ℕ) : ℝ := a n / (a (n + 1) - a n)

theorem is_arithmetic : ∃ d : ℝ, ∀ n, b (n + 1) - b n = d := sorry

-- Statement for part (3)
def c (n : ℕ) : ℝ := 2^(b n) * a n

theorem find_k (λ : ℝ) (hλ: 1 ≤ λ ∧ λ ≤ Real.sqrt 2) : 
  ∃ k : ℝ, (2 + 2 * Real.sqrt 2) < k ∧ ∀ n : ℕ, 2 * λ^2 - k * λ + 3 * Real.sqrt 2 < c n := sorry

end find_a2_is_arithmetic_find_k_l644_644280


namespace toll_for_18_wheel_truck_l644_644838

variable (t : ℝ) (x : ℕ)

def toll_formula (x : ℕ) : ℝ := 1.50 + 1.50 * (x - 2)

theorem toll_for_18_wheel_truck :
  let wheels_total := 18
  let wheels_front := 2
  let wheels_remaining := wheels_total - wheels_front
  let wheels_per_axle := 4
  let other_axles := wheels_remaining / wheels_per_axle
  let axles_total := other_axles + 1
  toll_formula axles_total = 6.00 := 
by
  let wheels_total := 18
  let wheels_front := 2
  let wheels_remaining := wheels_total - wheels_front
  let wheels_per_axle := 4
  let other_axles := wheels_remaining / wheels_per_axle
  let axles_total := other_axles + 1
  have h : toll_formula axles_total = 1.50 + 1.50 * (axles_total - 2) := rfl
  rw h
  have ha : axles_total = 5 := by sorry
  rw ha
  norm_num
  rw [show 1.50 * (5 - 2) = 4.50, by norm_num]
  norm_num
  sorry

end toll_for_18_wheel_truck_l644_644838


namespace min_distance_from_point_to_line_l644_644298

theorem min_distance_from_point_to_line : 
  ∀ (x₀ y₀ : Real), 3 * x₀ - 4 * y₀ - 10 = 0 → Real.sqrt (x₀^2 + y₀^2) = 2 :=
by sorry

end min_distance_from_point_to_line_l644_644298


namespace polygon_perimeter_l644_644890

theorem polygon_perimeter :
  let AB := 2
  let BC := 2
  let CD := 2
  let DE := 2
  let EF := 2
  let FG := 3
  let GH := 3
  let HI := 3
  let IJ := 3
  let JA := 4
  AB + BC + CD + DE + EF + FG + GH + HI + IJ + JA = 26 :=
by {
  sorry
}

end polygon_perimeter_l644_644890


namespace solve_for_x_l644_644074

theorem solve_for_x (x : ℝ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 := 
  sorry

end solve_for_x_l644_644074


namespace probability_equilateral_triangle_l644_644914

open Real

noncomputable def distance_from_origin (p : Point) : ℝ := sqrt (p.1^2 + p.2^2)

noncomputable def X := (sqrt 2 - sqrt 6 / 3, 0)

def is_within_radius (p : Point) (r : ℝ) : Prop :=
  distance_from_origin p < r

def equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

theorem probability_equilateral_triangle :
  ∃ (a b c : ℕ), (a.gcd c = 1) ∧ (probability X (within_radius 4)  = (a * π + b) / (c * π)) ∧ (a + b + c = 34) :=
begin
  sorry
end

end probability_equilateral_triangle_l644_644914


namespace find_x_l644_644888

variables {ℝ : Type*} [linear_ordered_field ℝ] {x : ℝ}

def vec_add (v1 v2 : ℝ × ℝ) := (v1.1 + v2.1, v1.2 + v2.2)
def scalar_mult (k : ℝ) (v : ℝ × ℝ) := (k * v.1, k * v.2)
def parallel (v1 v2 : ℝ × ℝ) := ∃ (k : ℝ), scalar_mult k v2 = v1

theorem find_x (a b c d : ℝ × ℝ) (h₀ : a = (1, 1)) (h₁ : b = (4, x))
               (h₂ : c = vec_add a b) (h₃ : d = vec_add (scalar_mult 2 a) b)
               (h₄: parallel c d) : x = 4 :=
begin
  sorry
end

end find_x_l644_644888


namespace differential_equation_solution_exists_l644_644631

open Real

noncomputable def general_solution (x : ℝ) (C1 C2 : ℝ) : ℝ :=
  C1 + C2 * exp (-x) + (x^3) / 3

noncomputable def particular_solution (x : ℝ) : ℝ :=
  -2 + 3 * exp (-x) + (x^3) / 3

theorem differential_equation_solution_exists :
  ∃ (y : ℝ → ℝ), (∀ x, y x = -2 + 3 * exp (-x) + (x^3) / 3) ∧
  (∀ y, deriv y = λ x, (-3 * exp (-x) + x^2) ∧ y 0 = 1 ∧ deriv y 0 = -3) :=
by
  sorry

end differential_equation_solution_exists_l644_644631


namespace distance_between_points_l644_644630

noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 + (B.3 - A.3) ^ 2)

theorem distance_between_points :
  let A := (3, 2, -5) in
  let B := (6, 9, -2) in
  distance A B = real.sqrt 67 := 
by {
  -- proof goes here
  sorry
}

end distance_between_points_l644_644630


namespace rhombus_diagonals_perpendicular_not_in_rectangle_l644_644512

-- Definitions for the rhombus
structure Rhombus :=
  (diagonals_perpendicular : Prop)

-- Definitions for the rectangle
structure Rectangle :=
  (diagonals_not_perpendicular : Prop)

-- The main proof statement
theorem rhombus_diagonals_perpendicular_not_in_rectangle 
  (R : Rhombus) 
  (Rec : Rectangle) : 
  R.diagonals_perpendicular ∧ Rec.diagonals_not_perpendicular :=
by sorry

end rhombus_diagonals_perpendicular_not_in_rectangle_l644_644512


namespace lacy_correct_percentage_l644_644023

theorem lacy_correct_percentage (x : ℕ) (h : x > 0) : (6 / 7 : ℚ) * 100 = 857.142857 % :=
by
  -- This is a placeholder for the proof which we are not required to provide
  sorry

end lacy_correct_percentage_l644_644023


namespace compound_interest_1200_20percent_3years_l644_644250

noncomputable def compoundInterest (P r : ℚ) (n t : ℕ) : ℚ :=
  let A := P * (1 + r / n) ^ (n * t)
  A - P

theorem compound_interest_1200_20percent_3years :
  compoundInterest 1200 0.20 1 3 = 873.6 :=
by
  sorry

end compound_interest_1200_20percent_3years_l644_644250


namespace min_dwarfs_l644_644072

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l644_644072


namespace range_of_f_on_interval_l644_644696

-- Define the function f
def f (t : ℝ) : ℝ := 4 * t^2 - 8 * t + 3

-- Define the transformation from x to t
def t (x : ℝ) : ℝ := (x + 1) / 2

-- State the theorem about the range of f(x)
theorem range_of_f_on_interval :
  (∀ x : ℝ, x ∈ Ico (-1) 2 → (f (t x) ∈ Icc (-1) 15)) ∧
  (∀ y : ℝ, y ∈ Icc (-1) 15 → ∃ x : ℝ, x ∈ Ico (-1) 2 ∧ f (t x) = y) :=
by {
  sorry
}

end range_of_f_on_interval_l644_644696


namespace factorize_difference_of_squares_l644_644993

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2 * x) * (3 + 2 * x) :=
sorry

end factorize_difference_of_squares_l644_644993


namespace transformed_mean_and_variance_l644_644663

variable (x : Fin 5 → ℝ)
variable (mean_var : (∑ i, x i = 10) ∧ (∑ i, (x i - 2) ^ 2 / 5 = 1 / 3))

theorem transformed_mean_and_variance (x': Fin 5 → ℝ := λ i, 3 * (x i) - 2) 
: (∑ i, x' i) / 5 = 4 ∧ (∑ i, (x' i - 4) ^ 2 / 5 = 3) :=
by
  sorry

end transformed_mean_and_variance_l644_644663


namespace f_value_l644_644729

variables {α : Type*} [Trigonometric α]

noncomputable def f : α → α := sorry

-- Definitions based on conditions
def odd_function (f : α → α) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : α → α) (p : α) : Prop := ∀ x k : ℤ, f (x + p * k) = f x

-- Given data
axiom f_is_odd : odd_function f
axiom f_is_periodic : periodic_function f 5
axiom f_at_neg3 : f (-3) = 1
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Goal to prove
theorem f_value :
  f (20 * Real.sin α * Real.cos α) = -1 :=
sorry

end f_value_l644_644729


namespace proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l644_644158

variable (x y : ℤ)

def proposition_A := (x ≠ 1000 ∨ y ≠ 1002)
def proposition_B := (x + y ≠ 2002)

theorem proposition_A_necessary_for_B : proposition_B x y → proposition_A x y := by
  sorry

theorem proposition_A_not_sufficient_for_B : ¬ (proposition_A x y → proposition_B x y) := by
  sorry

end proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l644_644158


namespace evaluate_expression_l644_644616

theorem evaluate_expression : (↑7 ^ (1/4) / ↑7 ^ (1/6)) = (↑7 ^ (1/12)) :=
by
  sorry

end evaluate_expression_l644_644616


namespace not_right_triangle_l644_644754

theorem not_right_triangle (a b c : ℝ) (h : a / b = 1 / 2 ∧ b / c = 2 / 3) :
  ¬(a^2 = b^2 + c^2) :=
by sorry

end not_right_triangle_l644_644754


namespace cupric_cyanide_formation_l644_644322

/--
Given:
1 mole of CuSO₄ 
2 moles of HCN

Prove:
The number of moles of Cu(CN)₂ formed is 0.
-/
theorem cupric_cyanide_formation (CuSO₄ HCN : ℕ) (h₁ : CuSO₄ = 1) (h₂ : HCN = 2) : 0 = 0 :=
by
  -- Proof goes here
  sorry

end cupric_cyanide_formation_l644_644322


namespace bobby_finishes_candies_in_weeks_l644_644583

def total_candies (packets: Nat) (candies_per_packet: Nat) : Nat := packets * candies_per_packet

def candies_eaten_per_week (candies_per_day_mon_fri: Nat) (days_mon_fri: Nat) (candies_per_day_weekend: Nat) (days_weekend: Nat) : Nat :=
  (candies_per_day_mon_fri * days_mon_fri) + (candies_per_day_weekend * days_weekend)

theorem bobby_finishes_candies_in_weeks :
  let packets := 2
  let candies_per_packet := 18
  let candies_per_day_mon_fri := 2
  let days_mon_fri := 5
  let candies_per_day_weekend := 1
  let days_weekend := 2

  total_candies packets candies_per_packet / candies_eaten_per_week candies_per_day_mon_fri days_mon_fri candies_per_day_weekend days_weekend = 3 :=
by
  sorry

end bobby_finishes_candies_in_weeks_l644_644583


namespace domino_chessboard_cut_l644_644164

theorem domino_chessboard_cut (n : ℕ) (h : 1 ≤ n ∧ n ≤ 3) :
  (∃ line : ℤ, ∀ (r c : ℕ), ((r < line ∧ r + 1 ≥ line) ∨ 
                               (c < line ∧ c + 1 ≥ line)) → 
                      ¬ cuts_domino r c 2 1 n)
  ↔ n = 1 ∨ n = 2 :=
by sorry

-- Auxiliary definition to signify cutting through domino.
def cuts_domino (r c height width n : ℕ) : Prop := 
  ∃ k : ℕ, r < k ∧ k < (r + height) ∧ 
           ∃ l : ℕ, c < l ∧ l < (c + width)

end domino_chessboard_cut_l644_644164


namespace integer_part_of_s_l644_644655

theorem integer_part_of_s :
  let s := ∑ n in Finset.range(1, 10^6 + 1), 1 / Real.sqrt n
  ⌊s⌋ = 1998 := by
  let s := ∑ n in Finset.range (1, 10^6 + 1), 1 / Real.sqrt n
  have : 1998 < s ∧ s < 1999 := sorry
  sorry

end integer_part_of_s_l644_644655


namespace space_diagonals_count_l644_644541

theorem space_diagonals_count (Q : Type) [finite Q] (vertices : ℕ) (edges : ℕ) (tri_faces : ℕ) (quad_faces : ℕ) 
  (h_vertices : vertices = 30)
  (h_edges : edges = 70)
  (h_faces : tri_faces = 30)
  (h_quad_faces : quad_faces = 12) :
  let total_segments := vertices * (vertices - 1) / 2 in
  let face_diagonals := quad_faces * 2 in
  let space_diagonals := total_segments - edges - face_diagonals in
  space_diagonals = 341 :=
by
  have h_total_segments : total_segments = 435 := by
    calc
      vertices * (vertices - 1) / 2 = 30 * 29 / 2 := by rw h_vertices
      ... = 435 by norm_num
  have h_face_diagonals : face_diagonals = 24 := by
    calc
      quad_faces * 2 = 12 * 2 := by rw h_quad_faces
      ... = 24 by norm_num
  have h_space_diagonals : space_diagonals = 435 - 70 - 24 := by
    calc
      total_segments - edges - face_diagonals = 435 - 70 - 24 := by rw [h_total_segments, h_edges, h_face_diagonals]
  show space_diagonals = 341 from by
    calc
      435 - 70 - 24 = 341 by norm_num 
      ... = 341 by refl

end space_diagonals_count_l644_644541


namespace correct_order_l644_644781

noncomputable def f (x : ℝ) : ℝ := x^2 - real.pi * x

noncomputable def α : ℝ := real.arcsin (1 / 3)
noncomputable def β : ℝ := real.arctan (5 / 4)
noncomputable def γ : ℝ := real.arccos (-1 / 3)
noncomputable def δ : ℝ := real.arccot (-5 / 4)

theorem correct_order :
  f α > f δ ∧ f δ > f β ∧ f β > f γ :=
sorry

end correct_order_l644_644781


namespace solve_problem_l644_644225

-- Define the operation @ as given in the problem
def operation (x y : ℝ) : ℝ := real.sqrt (x * y + 4)

-- State the theorem to be proven
theorem solve_problem : operation (operation 2 6) 8 = 6 :=
by 
  sorry

end solve_problem_l644_644225


namespace area_of_pentagon_m_n_l644_644620

noncomputable def m : ℤ := 12
noncomputable def n : ℤ := 11

theorem area_of_pentagon_m_n :
  let pentagon_area := (Real.sqrt m) + (Real.sqrt n)
  m + n = 23 :=
by
  have m_pos : m > 0 := by sorry
  have n_pos : n > 0 := by sorry
  sorry

end area_of_pentagon_m_n_l644_644620


namespace bee_distance_to_P2015_l644_644894

noncomputable def bee_flight_distance : ℝ :=
  let P_0 := (0 : ℂ)
  let P_1 := (1 : ℂ)
  let ω := complex.exp (complex.I * real.pi / 6)  -- ω = e^(iπ/6)
  let z := ∑ k in finset.range 2015, (k + 1) * ω^k
  abs z

theorem bee_distance_to_P2015 : bee_flight_distance = 1008 * real.sqrt 6 + 1008 * real.sqrt 2 :=
by
  -- Proof is omitted
  sorry

end bee_distance_to_P2015_l644_644894


namespace total_employees_l644_644175

variable (x y m : ℕ)
variable (A_employees : x = 20)
variable (C_employees : y = 10)
variable (B_employees : 300)
variable (sample_total : 45)
variable (B_selected : 15)

theorem total_employees 
  (A_employees : ∀ x, WorkshopA x)
  (C_employees : ∀ y, WorkshopC y)
  (sample_total : 45)
  (B_selected : 15)
  (proportion : 45 / m = 15 / 300) : 
  m = 900 := 
sorry

end total_employees_l644_644175


namespace least_n_for_1987_trailing_zeros_l644_644231

theorem least_n_for_1987_trailing_zeros :
  ∃ (n : ℕ), (n = 7960) ∧ (∑ k in (range $ nat.ceil (real.logb 5 n)), n / 5^k = 1987) :=
by
  sorry

end least_n_for_1987_trailing_zeros_l644_644231


namespace subset_disjointness_l644_644661

theorem subset_disjointness (n m : ℕ) {S : Finset ℕ} (hS : S.card = (n + 1) * m - 1)
  (partition : ∀ (T : Finset (Finset ℕ)), T.card = n → (T ⊆ S → T ∈ A ∨ T ∈ B))
  (disjoint : ∀ (T1 T2 : Finset (Finset ℕ)), T1 ∈ A → T2 ∈ A → (T1 ≠ T2 → T1 ∩ T2 = ∅)) :
  ∃ K : ℕ, K ≥ m ∧ ∃ T : Finset (Finset ℕ), T.card = K ∧ (∀ t ∈ T, t.card = n ∧ t ⊆ S) ∧ pairwise T disjoint :=
sorry

end subset_disjointness_l644_644661


namespace breeding_number_seventh_year_l644_644816

noncomputable def breeding_number (a x : ℕ) : ℝ :=
  a * Real.log2 (x + 1)

theorem breeding_number_seventh_year :
  ∀ a, breeding_number a 1 = 100 → breeding_number a 7 = 300 :=
by
  intros a h
  have h1 : a = 100 := sorry
  have h2 : breeding_number a 7 = 300 := sorry
  exact h2


end breeding_number_seventh_year_l644_644816


namespace proper_subset_count_l644_644314

open Finset

variable (U : Finset ℕ) (complement_A : Finset ℕ)
variable (hU : U = {1, 2, 3}) (h_complement_A : complement_A = {2})

noncomputable def A : Finset ℕ := U \ complement_A

theorem proper_subset_count (U : Finset ℕ) (complement_A : Finset ℕ)
  (hU : U = {1, 2, 3}) (h_complement_A : complement_A = {2}) :
  (U \ complement_A).card = 2 ∧ (U \ complement_A).powerset.card - 1 = 3 := 
sorry

end proper_subset_count_l644_644314


namespace train_length_l644_644548

noncomputable def jogger_speed_kmh : ℝ := 9
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def head_start : ℝ := 270
noncomputable def passing_time : ℝ := 39

noncomputable def kmh_to_ms (speed: ℝ) : ℝ := speed * (1000 / 3600)

theorem train_length (l : ℝ) 
  (v_j := kmh_to_ms jogger_speed_kmh)
  (v_t := kmh_to_ms train_speed_kmh)
  (d_h := head_start)
  (t := passing_time) :
  l = 120 :=
by 
  sorry

end train_length_l644_644548


namespace calculate_3_to_5_mul_7_to_5_l644_644214

theorem calculate_3_to_5_mul_7_to_5 : 3^5 * 7^5 = 4084101 :=
by {
  -- Sorry is added to skip the proof; assuming the proof is done following standard arithmetic calculations
  sorry
}

end calculate_3_to_5_mul_7_to_5_l644_644214


namespace card_arrangements_count_l644_644269

-- Define the types and numbers associated with the cards
inductive Suit
| Hearts
| Clubs

inductive Rank
| Two
| Three
| Four
| Five

structure Card where
  suit : Suit
  rank : Rank

def card_value (c : Card) : ℕ :=
  match c.rank with
  | Rank.Two => 2
  | Rank.Three => 3
  | Rank.Four => 4
  | Rank.Five => 5

-- Define the set of all eight cards
def deck : List Card :=
  [⟨Suit.Hearts, Rank.Two⟩, ⟨Suit.Hearts, Rank.Three⟩, ⟨Suit.Hearts, Rank.Four⟩,
   ⟨Suit.Hearts, Rank.Five⟩, ⟨Suit.Clubs, Rank.Two⟩, ⟨Suit.Clubs, Rank.Three⟩,
   ⟨Suit.Clubs, Rank.Four⟩, ⟨Suit.Clubs, Rank.Five⟩]

-- Define the conditions
def valid_draw (cards : List Card) : Prop :=
  cards.length = 4 ∧ (cards.map card_value).sum = 14

-- Prove the number of different arrangements
theorem card_arrangements_count (cards : List Card)
  (h1 : valid_draw cards) : (cards.toFinset.card ) = 864 := by
  sorry

end card_arrangements_count_l644_644269


namespace compute_inverse_10_mod_1729_l644_644594

def inverse_of_10_mod_1729 : ℕ :=
  1537

theorem compute_inverse_10_mod_1729 :
  (10 * inverse_of_10_mod_1729) % 1729 = 1 :=
by
  sorry

end compute_inverse_10_mod_1729_l644_644594


namespace find_a7_l644_644750

variable {a : ℕ → ℕ}  -- Define the geometric sequence as a function from natural numbers to natural numbers.
variable (h_geo_seq : ∀ (n k : ℕ), a n ^ 2 = a (n - k) * a (n + k)) -- property of geometric sequences
variable (h_a3 : a 3 = 2) -- given a₃ = 2
variable (h_a5 : a 5 = 8) -- given a₅ = 8

theorem find_a7 : a 7 = 32 :=
by
  sorry

end find_a7_l644_644750


namespace inverse_proportion_quadrants_l644_644267

theorem inverse_proportion_quadrants (m : ℝ) : (∀ (x : ℝ), x ≠ 0 → y = (m - 2) / x → (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) ↔ m > 2 :=
by
  sorry

end inverse_proportion_quadrants_l644_644267


namespace rain_on_tuesday_l644_644743

-- Define the probabilities as constants
def P_RM : ℝ := 0.6
def P_RMT : ℝ := 0.4
def P_RN : ℝ := 0.25

-- Define the problem statement
theorem rain_on_tuesday : ∃ P_RT : ℝ, P_RT = 0.55 :=
by
  let P_RT : ℝ := 1 - P_RM + P_RMT - P_RN
  suffices : P_RT = 0.55
  from Exists.intro P_RT this
  sorry

end rain_on_tuesday_l644_644743


namespace max_three_digit_sum_l644_644725

theorem max_three_digit_sum (A B C : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) :
  110 * A + 10 * B + 3 * C ≤ 981 :=
sorry

end max_three_digit_sum_l644_644725


namespace sum_problem_l644_644962

theorem sum_problem :
  ∑ n in Finset.range 99 \ Finset.singleton 0 \ Finset.singleton 1, (1 : ℝ) / (n + 2) * sqrt (n + 1) + (n + 1) * sqrt (n + 2) = 9 / 10 := 
by
  sorry

end sum_problem_l644_644962


namespace replaced_solution_percentage_l644_644187

theorem replaced_solution_percentage (y x z w : ℝ) 
  (h1 : x = 0.5)
  (h2 : y = 80)
  (h3 : z = 0.5 * y)
  (h4 : w = 50) 
  :
  (40 + 0.5 * x) = 50 → x = 20 :=
by
  sorry

end replaced_solution_percentage_l644_644187


namespace anthony_pencils_total_l644_644580

def pencils_initial : Nat := 9
def pencils_kathryn : Nat := 56
def pencils_greg : Nat := 84
def pencils_maria : Nat := 138

theorem anthony_pencils_total : 
  pencils_initial + pencils_kathryn + pencils_greg + pencils_maria = 287 := 
by
  sorry

end anthony_pencils_total_l644_644580


namespace quadratic_solution_l644_644033

theorem quadratic_solution (x : ℝ) :
  2 * x^2 - 8 * x + 5 = 0 ↔ x = 2 + sqrt (3/2) / 2 ∨ x = 2 - sqrt (3/2) / 2 :=
by
  sorry

end quadratic_solution_l644_644033


namespace polynomial_solution_l644_644981

open Polynomial

noncomputable def p (x : ℝ) : ℝ := -4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2

theorem polynomial_solution (x : ℝ) :
  4 * x^5 + 3 * x^3 + 2 * x^2 + (-4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2) = 6 * x^3 - 5 * x^2 + 4 * x - 2 :=
by
  -- Verification of the equality
  sorry

end polynomial_solution_l644_644981


namespace assignment_for_points_l644_644706

def point := (ℤ × ℤ)
def S : set point := { p | ∃(x y: ℤ), 1 ≤ x ∧ x ≤ 2016 ∧ 1 ≤ y ∧ y ≤ 2016 ∧ p = (x, y) }

def are_collinear (p1 p2 p3 : point) : Prop :=
∃ (a b c : ℤ), a * (p1.1) + b * (p1.2) + c = 0 ∧
                a * (p2.1) + b * (p2.2) + c = 0 ∧
                a * (p3.1) + b * (p3.2) + c = 0

def not_pairwise_coprime (a b c : ℕ) : Prop :=
∃ d > 1, d ∣ a ∧ d ∣ b ∧ d ∣ c

theorem assignment_for_points :
  ∃ (f : point → ℕ), (∀ p1 p2 p3 ∈ S,
    are_collinear p1 p2 p3 ↔ not_pairwise_coprime (f p1) (f p2) (f p3)) :=
sorry

end assignment_for_points_l644_644706


namespace B_2_2_eq_9_l644_644605

def B : ℕ → ℕ → ℕ
| 0, n     := n + 1
| (m+1), 0 := B m 2
| (m+1), (n+1) := B m (B (m+1) n)

theorem B_2_2_eq_9 : B 2 2 = 9 := 
by
  sorry

end B_2_2_eq_9_l644_644605


namespace max_annual_profit_at_16_annual_profit_function_l644_644905

noncomputable def annual_profit (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then
    (- (x - 16) ^ 2 + 156.9)
  else if x > 20 then
    160 - (x / 100)
  else
    0

theorem max_annual_profit_at_16 :
  ∀ x : ℕ, x = 16 → annual_profit x = 156.9 :=
sorry

theorem annual_profit_function :
  ∀ x : ℕ, x > 0 → annual_profit x =
    if x ≤ 20 then
      -x^2 + 32 * x - 1.1
    else
      160 - x / 100 :=
sorry

end max_annual_profit_at_16_annual_profit_function_l644_644905


namespace grasshoppers_no_overlap_l644_644268

/-- 
Mathematical problem stating that no two grasshoppers can land on the same spot 
given the conditions provided.
-/
theorem grasshoppers_no_overlap :
  (∀ n : ℕ, ∀ a b : Fin 4, a ≠ b → 
    let positions := 
      match a, b with
      | 0, 0 => (0,0)
      | 0, 1 => (0, 3^n)
      | 0, 2 => (3^n, 3^n)
      | 0, 3 => (3^n, 0)
      | 1, 0 => (0, 3^n)
      | 1, 1 => (0,0)
      | 1, 2 => (3^n, 3^n)
      | 1, 3 => (3^n, 0)
      | 2, 0 => (3^n, 0)
      | 2, 1 => (0, 3^n)
      | 2, 2 => (0, 0)
      | 2, 3 => (3^n, 3^n)
      | 3, 0 => (3^n, 0)
      | 3, 1 => (0, 3^n)
      | 3, 2 => (3^n, 3^n)
      | 3, 3 => (0,0)
      end;
    positions a ≠ positions b) :=
sorry

end grasshoppers_no_overlap_l644_644268


namespace green_tea_leaves_needed_l644_644942

-- Constants for the given conditions
constant sprigs_of_mint : ℕ := 3
constant green_tea_leaves_per_sprig : ℕ := 2
constant efficacy_factor : ℕ := 2

-- The theorem to prove
theorem green_tea_leaves_needed : 
  (sprigs_of_mint * green_tea_leaves_per_sprig) * efficacy_factor = 12 := 
by {
  sorry
}

end green_tea_leaves_needed_l644_644942


namespace min_value_a_b_l644_644674

open Real

theorem min_value_a_b {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (parallel : a * (1 - b) - b * (a - 4) = 0) : a + b = 9 / 2 :=
by sorry

end min_value_a_b_l644_644674


namespace rectangle_square_division_l644_644915

theorem rectangle_square_division (a b : ℝ) (n : ℕ) (h1 : (∃ (s1 : ℝ), s1^2 * (n : ℝ) = a * b))
                                            (h2 : (∃ (s2 : ℝ), s2^2 * (n + 76 : ℝ) = a * b)) :
    n = 324 := 
by
  sorry

end rectangle_square_division_l644_644915


namespace purely_imaginary_complex_number_l644_644336

theorem purely_imaginary_complex_number (a : ℝ) (h : (a^2 - 3 * a + 2) = 0 ∧ (a - 2) ≠ 0) : a = 1 :=
by {
  sorry
}

end purely_imaginary_complex_number_l644_644336


namespace selling_price_correct_l644_644032

-- Define the given conditions
def purchase_price : ℝ := 42000
def repair_costs : ℝ := 13000
def profit_percent : ℝ := 21.636363636363637 / 100

-- Define the total cost
def total_cost : ℝ := purchase_price + repair_costs

-- Define the profit
def profit : ℝ := total_cost * profit_percent

-- Define the selling price
def selling_price : ℝ := total_cost + profit

-- Prove that the selling price is 66900
theorem selling_price_correct : selling_price = 66900 :=
by
  sorry

end selling_price_correct_l644_644032


namespace initial_percentage_of_water_l644_644388

/-
Initial conditions:
- Let W be the initial percentage of water in 10 liters of milk.
- The mixture should become 2% water after adding 15 liters of pure milk to it.
-/

theorem initial_percentage_of_water (W : ℚ) 
  (H1 : 0 < W) (H2 : W < 100) 
  (H3 : (10 * (100 - W) / 100 + 15) / 25 = 0.98) : 
  W = 5 := 
sorry

end initial_percentage_of_water_l644_644388


namespace problem_lemma_l644_644854

def sqrt4 (x : ℕ) : ℝ := x ^ (1 / 4 : ℝ)
def sqrt3 (x : ℕ) : ℝ := x ^ (1 / 3 : ℝ)
def sqrt2 (x : ℕ) : ℝ := x ^ (1 / 2 : ℝ)

theorem problem_lemma : sqrt4 16 * sqrt3 8 * sqrt2 4 = 8 := 
by
  -- The proof steps from the solution would go here
  sorry

end problem_lemma_l644_644854


namespace abs_diff_squares_l644_644125

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end abs_diff_squares_l644_644125


namespace initial_rope_length_correct_l644_644191

noncomputable def initial_rope_length (A_additional : ℝ) (new_length : ℝ) : ℝ :=
let π := Real.pi in
let new_area := π * new_length^2 in
let initial_area := new_area - A_additional in
let initial_length_squared := initial_area / π in
Real.sqrt initial_length_squared

theorem initial_rope_length_correct :
  initial_rope_length 1348.2857142857142 23 ≈ 9.99 :=
by
  sorry

end initial_rope_length_correct_l644_644191


namespace harmonic_mean_2_3_6_l644_644215

def harmonic_mean (a b c : ℕ) : ℚ := 3 / ((1 / a) + (1 / b) + (1 / c))

theorem harmonic_mean_2_3_6 : harmonic_mean 2 3 6 = 3 := 
by
  sorry

end harmonic_mean_2_3_6_l644_644215


namespace f_correct_l644_644372

noncomputable def f : ℕ → ℝ
| 0       => 0 -- undefined for 0, start from 1
| (n + 1) => if n = 0 then 1/2 else sorry -- recursion undefined for now

theorem f_correct : ∀ n ≥ 1, f n = (3^(n-1) / (3^(n-1) + 1)) :=
by
  -- Initial conditions
  have h0 : f 1 = 1/2 := sorry
  -- Recurrence relations
  have h1 : ∀ n, n ≥ 1 → f (n + 1) ≥ (3 * f n) / (2 * f n + 1) := sorry
  -- Prove the function form
  sorry

end f_correct_l644_644372


namespace value_of_a_l644_644783

theorem value_of_a {f : ℝ → ℝ} (a b c : ℝ) 
  (h1 :  ∀ x, f x = real.sqrt (a * x^2 + b * x + c))
  (h2 : a < 0)
  (h3 : ∀ s t : Set.Icc (-b - real.sqrt(b^2 - 4 * a * c) / (2 * a)) 
    (b + real.sqrt(b^2 - 4 * a * c) / (2 * a)), 
    ∃ sq, (∀ x ∈ sq, ∃ y ∈ sq, (s, f t) = x ∧ (x, f t) = y)) :
  a = -4 :=
sorry

end value_of_a_l644_644783


namespace part_a_length_A5A1_part_b_length_A_n_A_1_l644_644913

structure BrazilianFigure (n : ℕ) :=
  (vertices : Fin n → ℂ)
  (lengths : ∀ j : Fin (n-1), complex.abs (vertices j.succ - vertices j) = (Real.sqrt 2)^(j : ℕ))
  (angles : ∀ k : Fin (n-2), Complex.arg (vertices k.succ.succ - vertices k.succ) - Complex.arg (vertices k.succ - vertices k) = Real.pi * (3/4))

theorem part_a_length_A5A1 : 
  length_A_n_A_1 {n := 5, vertices := vertices, lengths := lengths, angles := angles} = 5 :=
sorry

theorem part_b_length_A_n_A_1 (n : ℕ) (h : n % 4 = 1) :
  length_A_n_A_1 {n := n, vertices := vertices, lengths := lengths, angles := angles} = 4^((n - 1) / 4) - (-1)^((n - 1) / 4) :=
sorry

end part_a_length_A5A1_part_b_length_A_n_A_1_l644_644913


namespace number_of_smaller_triangles_l644_644908

-- Define the conditions of the problem
def side_length_large : ℝ := 15
def side_length_small : ℝ := 3
def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

-- The key hypothesis stating what we need to prove.
theorem number_of_smaller_triangles :
  let A_large := area_equilateral_triangle side_length_large in
  let A_small := area_equilateral_triangle side_length_small in
  A_large / A_small = 25 :=
by
  sorry

end number_of_smaller_triangles_l644_644908


namespace min_dwarfs_for_no_empty_neighbor_l644_644057

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l644_644057


namespace sum_of_coefficients_polynomial_expansion_l644_644721

theorem sum_of_coefficients_polynomial_expansion :
  let polynomial := (2 * (1 : ℤ) + 3)^5
  ∃ b_5 b_4 b_3 b_2 b_1 b_0 : ℤ,
  polynomial = b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0 ∧
  (b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 3125 :=
by
  sorry

end sum_of_coefficients_polynomial_expansion_l644_644721


namespace tangent_lines_through_point_to_circle_l644_644089

theorem tangent_lines_through_point_to_circle : 
  ∀ (x y : ℝ), (x = 0 ∨ y + 1 = 0) ↔ 
  (∃ k : ℝ, P(0, -1) ∧ (x^2 + y^2 - 2*x + 4*y + 4 = 0) ∧ (y + 1 = k*(x - 0))) ∧ 
  (∀ (k : ℝ), (|k + 1| / Real.sqrt (k^2 + 1) = 1) → (y + 1 = 0 ∨ x = 0)) :=
by
  intro x y
  split
  { intro hx
    cases hx with h1 h2
    { left
      exact h1 }
    { right
      exact h2 } }
  { intro hy
    cases hy
    { left
      exact hy.left }
    { right
      exact hy.left } }
  sorry

end tangent_lines_through_point_to_circle_l644_644089


namespace distance_between_foci_of_tangent_ellipse_l644_644203

theorem distance_between_foci_of_tangent_ellipse
  (center : ℝ × ℝ) (a b : ℝ) (h1 : 6 = center.1)
  (h2 : 2 = center.2) (h3 : 2 * a = 12)
  (h4 : 2 * b = 4) : 
  real.sqrt (a^2 - b^2) = 4 * real.sqrt 2 :=
by sorry

end distance_between_foci_of_tangent_ellipse_l644_644203


namespace union_of_equilateral_triangles_area_l644_644875

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

noncomputable def total_area_without_overlaps (n : ℕ) (s : ℝ) : ℝ := n * equilateral_triangle_area s

noncomputable def overlap_area (s : ℝ) : ℝ := (sqrt 3 / 4) * (s / 2)^2

noncomputable def total_overlap_area (n : ℕ) (s : ℝ) : ℝ := (n - 1) * overlap_area s

noncomputable def net_area (n : ℕ) (s : ℝ) : ℝ := total_area_without_overlaps n s - total_overlap_area n s

theorem union_of_equilateral_triangles_area :
  net_area 5 (2 * sqrt 3) = 12 * sqrt 3 := by
  sorry

end union_of_equilateral_triangles_area_l644_644875


namespace possible_integer_roots_l644_644601

noncomputable def polynomial : Type := ℤ[X]

theorem possible_integer_roots (b c d e f : ℤ) :
    ∃ (m : ℕ), m ∈ {0, 1, 2, 3, 4, 5} ∧ (∃ g : polynomial, g = X^5 + C b * X^4 + C c * X^3 + C d * X^2 + C e * X + C f) := 
sorry

end possible_integer_roots_l644_644601


namespace min_numbers_to_ensure_product_105_l644_644644

theorem min_numbers_to_ensure_product_105 : 
  ∃ n : ℕ, n = 7 ∧ ∀ (S : Finset ℕ), (∀ x ∈ S, 1 ≤ x ∧ x ≤ 100) → S.card ≥ n → 
  ∃ (x y ∈ S), x * y = 105 :=
by
  sorry

end min_numbers_to_ensure_product_105_l644_644644


namespace trapezoid_area_l644_644912

theorem trapezoid_area (u l h : ℕ) (hu : u = 12) (hl : l = u + 4) (hh : h = 10) : 
  (1 / 2 : ℚ) * (u + l) * h = 140 := by
  sorry

end trapezoid_area_l644_644912


namespace product_repeating_decimal_l644_644220

theorem product_repeating_decimal (p : ℚ) (h₁ : p = 152 / 333) : 
  p * 7 = 1064 / 333 :=
  by
    sorry

end product_repeating_decimal_l644_644220


namespace ratio_of_prices_l644_644202

variable (C : ℝ)
def P1 := 1.35 * C
def P2 := 0.90 * C

theorem ratio_of_prices : P2 / P1 = (2 : ℝ) / 3 := by
  sorry

end ratio_of_prices_l644_644202


namespace find_params_l644_644628

theorem find_params (a b c : ℝ) :
    (∀ x : ℝ, x = 2 ∨ x = -2 → x^5 + 4 * x^4 + a * x = b * x^2 + 4 * c) 
    → a = 16 ∧ b = 48 ∧ c = -32 :=
by
  sorry

end find_params_l644_644628


namespace cut_difference_l644_644923

-- define the conditions
def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

-- theorem to prove the correctness of the difference
theorem cut_difference : (skirt_cut - pants_cut = 0.25) :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end cut_difference_l644_644923


namespace necessary_but_not_sufficient_condition_l644_644652

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (hpq : p ∨ q) (h : p ∧ q) : p ∧ q ↔ (p ∨ q) := by
  sorry

end necessary_but_not_sufficient_condition_l644_644652


namespace most_students_can_attend_l644_644239

def availability_table : list (string × list (option bool)) :=
  [("Alice", [some true, none, some true, none, none, some true]),
   ("Bob", [none, some true, none, some true, some true, none]),
   ("Cindy", [some true, some true, none, some true, some true, none]),
   ("David", [none, none, some true, some true, none, some true]),
   ("Eva", [some true, none, none, none, some true, some true])]
   
def count_unavailabilities_per_day : list (list (option bool)) → list ℕ :=
  λ days, days.foldr (λ day acc, (day.filter_map id).length :: acc) []

def most_available_days (availabilities: list ℕ) : list ℕ :=
  let min_unavailable := availabilities.foldr min availabilities.head! in
  availabilities.zip_with_index.filter_map (λ (x : ℕ × ℕ), if x.1 = min_unavailable then some x.2 else none)

theorem most_students_can_attend (availabilities_table : list (string × list (option bool))) :
  most_available_days (count_unavailabilities_per_day (transpose (availabilities_table.map prod.snd))) = [1, 2] :=
by
  sorry

end most_students_can_attend_l644_644239


namespace real_numbers_theorem_l644_644332

theorem real_numbers_theorem (a b : ℝ)
  (h : |a - 1| + real.sqrt (b + 2) = 0) :
  (a + b)^2022 = 1 :=
sorry

end real_numbers_theorem_l644_644332


namespace real_and_imaginary_solutions_l644_644258

theorem real_and_imaginary_solutions :
  ∃ (x y : ℂ), (y = (x + 1)^4) ∧ (x * y + y = 5) ∧
  ((∃ (xr : ℝ), x = xr ∧ y = (xr + 1)^4) ∧
   (∃ (z1 z2 z3 z4 : ℂ), z1 ≠ x ∧ z2 ≠ x ∧ z3 ≠ x ∧ z4 ≠ x ∧
    (y = (z1 + 1)^4 ∨ y = (z2 + 1)^4 ∨ y = (z3 + 1)^4 ∨ y = (z4 + 1)^4))) := sorry

end real_and_imaginary_solutions_l644_644258


namespace polynomial_solution_l644_644625

noncomputable def P (x : ℝ) : ℝ := sorry

theorem polynomial_solution (P : ℝ → ℝ) (h1 : P 0 = 0)
  (h2 : ∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ (a : ℝ), ∀ x : ℝ, P x = a * x :=
begin
  sorry
end

end polynomial_solution_l644_644625


namespace samantha_trip_l644_644406

theorem samantha_trip (a b c d x : ℕ)
  (h1 : 1 ≤ a) (h2 : a + b + c + d ≤ 10) 
  (h3 : 1000 * d + 100 * c + 10 * b + a - (1000 * a + 100 * b + 10 * c + d) = 60 * x)
  : a^2 + b^2 + c^2 + d^2 = 83 :=
sorry

end samantha_trip_l644_644406


namespace mailing_ways_l644_644333

-- Definitions based on the problem conditions
def countWays (letters mailboxes : ℕ) : ℕ := mailboxes^letters

-- The theorem to prove the mathematically equivalent proof problem
theorem mailing_ways (letters mailboxes : ℕ) (h_letters : letters = 3) (h_mailboxes : mailboxes = 4) : countWays letters mailboxes = 4^3 := 
by
  rw [h_letters, h_mailboxes]
  rfl

end mailing_ways_l644_644333


namespace paired_attractions_consecutive_l644_644762

noncomputable def attraction_arrangements : ℕ :=
  let units := [1, 2, 3, 4] -- {AB}, C, D, E seen as 4 units
  let arrangements := (list.permutations units).length
  arrangements

theorem paired_attractions_consecutive :
  attraction_arrangements = 24 :=
by {
  -- Acknowledge the permuations of 4 distinct units {AB, C, D, E}
  have h : (list.permutations [1, 2, 3, 4]).length = 24,
  -- Use fact that permutation of 4 items is 4!
  exact (fact_perms_4).length
}

end paired_attractions_consecutive_l644_644762


namespace smallest_positive_multiple_of_32_l644_644476

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n = 32 * k ∧ n = 32 := by
  use 32
  constructor
  · exact Nat.zero_lt_succ 31
  · use 1
    constructor
    · exact Nat.zero_lt_succ 0
    · constructor
      · rfl
      · rfl

end smallest_positive_multiple_of_32_l644_644476


namespace trig_identity_proof_l644_644519

theorem trig_identity_proof :
  (∃ (cos68 cos8 cos82 cos22 cos53 cos23 cos67 cos37 : ℝ),
    cos68 = real.cos (68 * real.pi / 180) ∧
    cos8 = real.cos (8 * real.pi / 180) ∧
    cos82 = real.cos (82 * real.pi / 180) ∧
    cos22 = real.cos (22 * real.pi / 180) ∧
    cos53 = real.cos (53 * real.pi / 180) ∧
    cos23 = real.cos (23 * real.pi / 180) ∧
    cos67 = real.cos (67 * real.pi / 180) ∧
    cos37 = real.cos (37 * real.pi / 180) ∧ 
    ((cos68 * cos8 - cos82 * cos22) / (cos53 * cos23 - cos67 * cos37) = 1)) :=
by sorry

end trig_identity_proof_l644_644519


namespace internal_curve_convex_l644_644030

noncomputable def is_convex (C : set (ℝ × ℝ)) : Prop := 
  ∀ (x y : ℝ × ℝ), x ∈ C → y ∈ C → ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → t • x + (1 - t) • y ∈ C

noncomputable def r_neighborhood (r : ℝ) (C : set (ℝ × ℝ)) : set (ℝ × ℝ) := 
  { p | ∃ x ∈ C, dist p x < r }

theorem internal_curve_convex 
  (K : set (ℝ × ℝ)) 
  (r : ℝ)
  (hK_convex : is_convex K)
  (h_internal_curve : ∃ C, ∀ (p : ℝ × ℝ), p ∈ C ↔ (∃ x ∈ K, dist p x < r)) :
  ∃ C, is_convex C ∧ ∀ (p : ℝ × ℝ), p ∈ C ↔ (∃ x ∈ K, dist p x < r) :=
sorry

end internal_curve_convex_l644_644030


namespace rectangle_symmetry_l644_644916

-- Define basic geometric terms and the notion of symmetry
structure Rectangle where
  length : ℝ
  width : ℝ
  (length_pos : 0 < length)
  (width_pos : 0 < width)

def is_axes_of_symmetry (r : Rectangle) (n : ℕ) : Prop :=
  -- A hypothetical function that determines whether a rectangle r has n axes of symmetry
  sorry

theorem rectangle_symmetry (r : Rectangle) : is_axes_of_symmetry r 2 := 
  -- This theorem states that a rectangle has exactly 2 axes of symmetry
  sorry

end rectangle_symmetry_l644_644916


namespace minimum_dwarfs_l644_644054

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l644_644054


namespace problem_solution_l644_644097

theorem problem_solution (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end problem_solution_l644_644097


namespace term_101_is_two_l644_644192

def next_term (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2) + 1 else (n + 1) / 2

def sequence : ℕ → ℕ
| 0     := 16
| (n+1) := next_term (sequence n)

theorem term_101_is_two : sequence 100 = 2 := 
sorry

end term_101_is_two_l644_644192


namespace round_robin_cyclic_sets_l644_644741

theorem round_robin_cyclic_sets
  (teams : ℕ)
  (wins losses ties : ℕ)
  (h_teams : teams = 21)
  (h_wins : wins = 8)
  (h_losses : losses = 7)
  (h_ties : ties = 5) :
  ∃ sets_of_three, sets_of_three = 742 :=
by {
  use 742,
  sorry
}

end round_robin_cyclic_sets_l644_644741


namespace inverse_mod_l644_644592

theorem inverse_mod (a b n : ℕ) (h : (a * b) % n = 1) : b % n = a⁻¹ % n := sorry

example : ∃ x : ℕ, (10 * x) % 1729 = 1 ∧ x < 1729 :=
by
  use 1585
  have h₁ : (10 * 1585) % 1729 = 1 := by norm_num
  exact ⟨h₁, by norm_num⟩

end inverse_mod_l644_644592


namespace centipede_sock_shoe_order_l644_644898

theorem centipede_sock_shoe_order :
  let legs := 10
  let items_per_leg := 2
  let total_items := legs * items_per_leg
  ∃ (valid_permutations : ℕ),
    valid_permutations = (Nat.factorial total_items) / 2^legs :=
begin
  let legs := 10,
  let items_per_leg := 2,
  let total_items := legs * items_per_leg,
  use (Nat.factorial total_items) / 2^legs,
  sorry
end

end centipede_sock_shoe_order_l644_644898


namespace arithmetic_sequence_sum_l644_644866

theorem arithmetic_sequence_sum :
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  Sₙ = 1830 :=
by
  intros
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  sorry

end arithmetic_sequence_sum_l644_644866


namespace product_divisible_by_294_l644_644446

theorem product_divisible_by_294 (k : ℤ) : 
  let a := 7 * k,
      b := a + 7,
      c := b + 7
  in 294 ∣ (a * b * c) :=
by 
  let a := 7 * k,
      b := a + 7,
      c := b + 7
  show 294 ∣ (a * b * c)
  sorry

end product_divisible_by_294_l644_644446


namespace problem1_problem2_problem3_l644_644994

-- Problem 1
theorem problem1 (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B :=
by sorry

-- Problem 2
theorem problem2 (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 :=
by sorry

end problem1_problem2_problem3_l644_644994


namespace fraction_of_eggs_hatched_l644_644947

def E : ℕ := _
def survived_first_year : ℕ := 120
def survived_fraction : ℝ := (4:ℝ) / 5 * (2:ℝ) / 5

theorem fraction_of_eggs_hatched (F : ℝ) (H : F = (survived_first_year : ℝ) / (E * survived_fraction)) : F = 1 :=
by
  sorry

end fraction_of_eggs_hatched_l644_644947


namespace cosine_value_of_alpha_l644_644698

theorem cosine_value_of_alpha (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  let P := (1, -1) in
  let α := real.angle.fromPoints P (0, 0) in
  real.cos α = real.sqrt 2 / 2 :=
sorry

end cosine_value_of_alpha_l644_644698


namespace win_probability_greater_than_half_l644_644459

/-- Define the boxes and initial choice -/
inductive Box
| A | B | C

def initial_choice : Box := Box.A

/-- Define the probabilities -/
def probability_of_box (b : Box) : ℝ :=
  match b with
  | Box.A => 1 / 3
  | Box.B => 1 / 3
  | Box.C => 1 / 3

/-- Host's action: reveals an empty box that is not initially chosen -/
def host_action (initial : Box) (prize : Box) : Box :=
  if initial = Box.A then
    if prize = Box.B then Box.C else Box.B
  else if initial = Box.B then
    if prize = Box.A then Box.C else Box.A
  else
    if prize = Box.A then Box.B else Box.A

/-- Probability after switching -/
def probability_after_switch (initial : Box) (prize : Box) : ℝ :=
  if initial = prize then 0 else 1

/-- Theorem stating the probability of winning after switching is greater than 1/2 -/
theorem win_probability_greater_than_half :
  ∀ (prize : Box), (probability_after_switch initial_choice prize) > 1 / 2 := by
  sorry

end win_probability_greater_than_half_l644_644459


namespace juice_can_problem_l644_644843

theorem juice_can_problem :
  ∀ (cans : ℕ) (fill_fraction : ℚ) (cans_needed : ℕ),
  cans = 3 → fill_fraction = 2 / 3 →
  (cans_needed = 36 ↔ 8 = (8 * 1 / (cans * fill_fraction)) * cans) :=
by
  intros
  split
  sorry
  sorry

end juice_can_problem_l644_644843


namespace water_formed_amount_limiting_reactant_naoh_unreacted_h3po4_amount_l644_644228

def h3po4_initial := 2.5 -- initial moles of H3PO4
def naoh_initial := 3.0 -- initial moles of NaOH

def balanced_equation := 
  ∀ (h3po4 naoh na3po4 h2o : ℝ), 
    h3po4 + 3 * naoh = na3po4 + 3 * h2o -- balanced chemical equation

theorem water_formed_amount : 
  ∀ (h3po4_initial naoh_initial : ℝ), 
  let limiting_reactant := min (h3po4_initial / 1) (naoh_initial / 3) in
  (limiting_reactant = naoh_initial / 3) →
  (3 * limiting_reactant = 3) :=
by
  intros h3po4_initial naoh_initial
  let limiting_reactant := min (h3po4_initial / 1) (naoh_initial / 3)
  have h : limiting_reactant = naoh_initial / 3 := by sorry
  have water_formed := 3 * limiting_reactant
  have correct_amt := water_formed = 3
  exact correct_amt

theorem limiting_reactant_naoh : 
  ∀ (h3po4_initial naoh_initial : ℝ), 
  let limiting_reactant := min (h3po4_initial / 1) (naoh_initial / 3) in
  (limiting_reactant = naoh_initial / 3) :=
by
  intros h3po4_initial naoh_initial
  let limiting_reactant := min (h3po4_initial / 1) (naoh_initial / 3)
  have correct_limiting := limiting_reactant = naoh_initial / 3
  exact correct_limiting

theorem unreacted_h3po4_amount : 
  ∀ (h3po4_initial naoh_initial : ℝ), 
  let limiting_reactant := min (h3po4_initial / 1) (naoh_initial / 3) in
  (limiting_reactant = naoh_initial / 3) →
  (h3po4_initial - (limiting_reactant * (1 / 3)) = 1.5) :=
by
  intros h3po4_initial naoh_initial
  let limiting_reactant := min (h3po4_initial / 1) (naoh_initial / 3)
  have h : limiting_reactant = naoh_initial / 3 := by sorry
  have unreacted_h3po4 := h3po4_initial - (1 * limiting_reactant)
  have correct_unreacted_amt := unreacted_h3po4 = 1.5
  exact correct_unreacted_amt

end water_formed_amount_limiting_reactant_naoh_unreacted_h3po4_amount_l644_644228


namespace inverse_true_l644_644513

theorem inverse_true : 
  (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop), 
    (∀ a b, supplementary a b → a = b) ∧ (∀ l1 l2, parallel l1 l2)) ↔ 
    (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop),
    (∀ l1 l2, parallel l1 l2) ∧ (∀ a b, supplementary a b → a = b)) :=
sorry

end inverse_true_l644_644513


namespace tetrahedron_volume_EFGH_l644_644077

def volume_of_tetrahedron (EF EG EH FG FH GH : ℝ) : ℝ :=
  let matrix_det := by 
    sorry -- Placeholder for determinant computation
  in matrix_det / 288

theorem tetrahedron_volume_EFGH :
  volume_of_tetrahedron 3 4 5 (Real.sqrt 17) (2 * Real.sqrt 6) 6 = 3 * Real.sqrt 2 :=
by 
  sorry

end tetrahedron_volume_EFGH_l644_644077


namespace general_formula_a_general_formula_b_sum_first_n_terms_l644_644771

-- Definitions of the sequences
def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ := (list.range n).sum

def T_n (b : ℕ → ℕ) (n : ℕ) : ℕ := (list.range n).sum

-- Conditions
axiom a_n_eq_condition (a : ℕ → ℕ) (n : ℕ) : 3 * a n = 2 * S_n a n + 3

axiom b_n_arithmetic (b : ℕ → ℕ) (d : ℕ) : ∀ n, b (n + 1) - b n = d

axiom T5_eq_25 (T : ℕ → ℕ) : T 5 = 25

axiom b10_eq_19 (b : ℕ → ℕ) : b 10 = 19

-- Proofs to be implemented
theorem general_formula_a (a : ℕ → ℕ) :
  (∀ n, a n = 3^n) :=
begin
  -- proof goes here
  sorry
end

theorem general_formula_b (b : ℕ → ℕ) :
  (∀ n, b n = 2 * n - 1) :=
begin
  -- proof goes here
  sorry
end

def c_n (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (a n * b n) / (n * (n + 1))

def R_n (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (list.range n).sum

-- Sum of first n terms for the sequence {c_n}
theorem sum_first_n_terms (a : ℕ → ℕ) (b : ℕ → ℕ) :
  ∀ n, R_n a b n = (3^(n + 1)) / (n + 1) - 3 :=
begin
  -- proof goes here
  sorry
end

end general_formula_a_general_formula_b_sum_first_n_terms_l644_644771


namespace compute_inverse_10_mod_1729_l644_644595

def inverse_of_10_mod_1729 : ℕ :=
  1537

theorem compute_inverse_10_mod_1729 :
  (10 * inverse_of_10_mod_1729) % 1729 = 1 :=
by
  sorry

end compute_inverse_10_mod_1729_l644_644595


namespace tangent_slope_is_4_l644_644104

theorem tangent_slope_is_4 (x y : ℝ) (h_curve : y = x^4) (h_slope : (deriv (fun x => x^4) x) = 4) :
    (x, y) = (1, 1) :=
by
  -- Place proof here
  sorry

end tangent_slope_is_4_l644_644104


namespace range_of_a_l644_644006

theorem range_of_a (a : ℝ) (x : ℤ) (h1 : ∀ x, x > 0 → ⌊(x + a) / 3⌋ = 2) : a < 8 :=
sorry

end range_of_a_l644_644006


namespace smallest_positive_period_and_monotonic_increase_max_min_in_interval_l644_644012

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem smallest_positive_period_and_monotonic_increase :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∃ a b : ℝ, (k * π - π / 3 ≤ a ∧ a ≤ x) ∧ (x ≤ b ∧ b ≤ k * π + π / 6) → f x = 1) := sorry

theorem max_min_in_interval :
  (∀ x : ℝ, (-π / 4 ≤ x ∧ x ≤ π / 6) → (1 - Real.sqrt 3 ≤ f x ∧ f x ≤ 3)) := sorry

end smallest_positive_period_and_monotonic_increase_max_min_in_interval_l644_644012


namespace fifth_grade_total_students_l644_644353

-- Define the conditions given in the problem
def total_boys : ℕ := 350
def total_playing_soccer : ℕ := 250
def percentage_boys_playing_soccer : ℝ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- Define the total number of students
def total_students : ℕ := 500

-- Prove that the total number of students is 500
theorem fifth_grade_total_students 
  (H1 : total_boys = 350) 
  (H2 : total_playing_soccer = 250) 
  (H3 : percentage_boys_playing_soccer = 0.86) 
  (H4 : girls_not_playing_soccer = 115) :
  total_students = 500 := 
sorry

end fifth_grade_total_students_l644_644353


namespace lower_limit_prime_between_a_and_25_l644_644805

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem lower_limit_prime_between_a_and_25 (D : Set ℕ) (h1 : ∀ n, n ∈ D ↔ is_prime n ∧ 13 ≤ n ∧ n ≤ 25) (h2 : 25 - 13 = 12) : ∃ a, a = 13 :=
by
  use 13
  sorry

end lower_limit_prime_between_a_and_25_l644_644805


namespace probability_of_odd_five_digit_number_l644_644423

theorem probability_of_odd_five_digit_number :
  let digits := {2, 3, 5, 7, 9}
  let n := 5
  let odd_digits := {3, 5, 7, 9}
  (|odd_digits|: ℝ) / (|digits|: ℝ) = 4 / 5 :=
by
  sorry

end probability_of_odd_five_digit_number_l644_644423


namespace cyclic_sum_inequality_l644_644377

open BigOperators

theorem cyclic_sum_inequality (n : ℕ) (x : Fin n → ℝ) (hpos : ∀ i, 0 < x i) (hsum : ∑ i, x i = 1) :
  (∑ i, (x i)^2 / (x i + x ((i + 1) % n))) ≥ (1/2) :=
by
  sorry

end cyclic_sum_inequality_l644_644377


namespace sum_of_possible_f_l644_644688

-- Definition: magic square conditions and establishing fg = 36
def is_magic_square (a b c d e f g : ℕ) (P : ℕ) :=
  (72 * a * b = P) ∧ (c * d * e = P) ∧ (f * g * 4 = P)

theorem sum_of_possible_f (a b c d e f g P : ℕ) (h : is_magic_square a b c d e f g P) (hP : P = 144) (hfg : f * g = 36) :
  (∑ x in {x | ∃ g, x * g = 36 ∧ x > 0}, x) = 91 :=
by
  sorry

end sum_of_possible_f_l644_644688


namespace minimum_value_l644_644775

-- Define the expression E(a, b, c)
def E (a b c : ℝ) : ℝ := a^2 + 8 * a * b + 24 * b^2 + 16 * b * c + 6 * c^2

-- State the minimum value theorem
theorem minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  E a b c = 18 :=
sorry

end minimum_value_l644_644775


namespace complex_plane_quadrant_l644_644749

theorem complex_plane_quadrant :
  let z := (2 - 3 * complex.I) / (complex.I ^ 3)
  in (z.re > 0) ∧ (z.im > 0) :=
by
  let z := (2 - 3 * complex.I) / (complex.I ^ 3)
  sorry

end complex_plane_quadrant_l644_644749


namespace quadrilateral_with_all_sides_equal_is_rhombus_l644_644514

theorem quadrilateral_with_all_sides_equal_is_rhombus
  (Q : Type)
  [quadrilateral Q]
  (equal_sides : ∀ (a b c d : Q), a = b ∧ b = c ∧ c = d ∧ d = a) :
  (∀ (a b c d : Q), quadrilateral a b c d) → 
  (∀ (a b c d : Q), rhombus a b c d) :=
by
  intro Q
  assume equal_sides
  sorry

end quadrilateral_with_all_sides_equal_is_rhombus_l644_644514


namespace find_p_l644_644016

theorem find_p (f : ℂ) (w : ℂ) (p : ℂ) (h_f : f = 10) (h_w : w = 10 + 250 * complex.I) :
  (f * p - w = 20000) → (p = 2001 + 25 * complex.I) :=
by
  intros h_eq
  rw [h_f, h_w] at h_eq
  sorry

end find_p_l644_644016


namespace inverse_of_f_l644_644098

noncomputable def f (x : ℝ) : ℝ := 2 * real.sqrt x

theorem inverse_of_f :
  (∀ y (x : ℝ), x ≥ 0 → f x = y ↔ x = (y / 2) ^ 2) := 
by 
  sorry

end inverse_of_f_l644_644098


namespace g_100_equals_79_l644_644974

def g (x : ℕ) : ℕ :=
  if (Nat.log 3 x = Int.floor (Nat.log 3 x)) then
    Nat.log 3 x
  else if x % 5 ≠ 0 then
    1 + g(x + 2)
  else
    2 + g(x + 1)

theorem g_100_equals_79 : g 100 = 79 :=
  by sorry

end g_100_equals_79_l644_644974


namespace total_columns_l644_644102

variables (N L : ℕ)

theorem total_columns (h1 : L > 1500) (h2 : L = 30 * (N - 70)) : N = 180 :=
by
  sorry

end total_columns_l644_644102


namespace math_problem_l644_644498

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l644_644498


namespace tan_sum_eq_l644_644647

-- Define the conditions as hypotheses
variables (x y : ℝ)

-- The problem statement as Lean theorem
theorem tan_sum_eq :
  (sin x + sin y = 15 / 17) →
  (cos x + cos y = 8 / 17) →
  (sin (x - y) = 1 / 5) →
  (tan x + tan y = 195 * real.sqrt 6 / 328) := by
  sorry

end tan_sum_eq_l644_644647


namespace distance_from_sphere_center_to_triangle_plane_l644_644919

theorem distance_from_sphere_center_to_triangle_plane :
  ∀ (O : EuclideanSpace ℝ 3) (r : ℝ) (A B C : EuclideanSpace ℝ 3),
    let triangle := {a := A, b := B, c := C},
    dist O A = r →
    dist O B = r →
    dist O C = r →
    dist A B = 13 →
    dist B C = 13 →
    dist C A = 10 →
    r = 8 →
    dist_to_plane O triangle = (2 * sqrt 119) / 3 :=
by {
  intros,
  sorry
}

end distance_from_sphere_center_to_triangle_plane_l644_644919


namespace quadratic_has_two_distinct_real_roots_l644_644871

theorem quadratic_has_two_distinct_real_roots :
  let a := (1 : ℝ)
  let b := (-5 : ℝ)
  let c := (-1 : ℝ)
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l644_644871


namespace range_of_omega_l644_644436

theorem range_of_omega (ω : ℝ) (hω_pos : ω > 0) :
  (∃ x : ℝ, x ∈ set.Icc (0 : ℝ) 1 ∧ f x = inf (set.range f x) ∧ f x = sup (set.range f x)) ↔
  (ω ∈ set.Ico (13 * π / 6) (25 * π / 6)) :=
by
  have h1 : ∀ x, f x = 2 * sin (ω * x + π / 3), 
    { intro x, simple }
  have h2 : ∀ x, 0 ≤ x ∧ x ≤ 1, 
    { intro x, simple }
  have h3 : 
    (∃ x : ℝ, x ∈ set.Icc (0 : ℝ) 1 ∧ f x = inf (set.range f x) ∧ f x = sup (set.range f x)) 
    ↔ 
    (∃ x : ℝ, x ∈ set.Icc (0 : ℝ) 1 ∧ f x = f 0 ∧ f x = f 1),
    { 
      sorry 
    } 
  apply h3,

end range_of_omega_l644_644436


namespace trig_solution_l644_644808

noncomputable def solve_trig_system (x y : ℝ) : Prop :=
  (3 * Real.cos x + 4 * Real.sin x = -1.4) ∧ 
  (13 * Real.cos x - 41 * Real.cos y = -45) ∧ 
  (13 * Real.sin x + 41 * Real.sin y = 3)

theorem trig_solution :
  solve_trig_system (112.64 * Real.pi / 180) (347.32 * Real.pi / 180) ∧ 
  solve_trig_system (239.75 * Real.pi / 180) (20.31 * Real.pi / 180) :=
by {
    repeat { sorry }
  }

end trig_solution_l644_644808


namespace find_other_root_and_k_value_l644_644303

-- Defining the equation
def quadratic_eq (x k : ℝ) := 5 * x^2 + k * x - 10

-- Given condition: one root is -5
def one_root (k : ℝ) := quadratic_eq (-5) k = 0

noncomputable def other_root_and_k_value : ℝ × ℝ :=
  let x1 := 2 / 5 in
  let k := 23 in
  (x1, k)

theorem find_other_root_and_k_value :
  ∃ x1 k : ℝ, quadratic_eq (-5) k = 0 ∧ 5 * x1^2 + k * x1 - 10 = 0 ∧ x1 = 2 / 5 ∧ k = 23 :=
by
  let x1 := 2 / 5 in
  let k := 23 in
  existsi [x1, k]
  split
  · -- Prove the given root condition -5 satisfies the equation
    unfold quadratic_eq
    norm_num
    sorry
  split
  · -- Prove that the other root x1 satisfies the equation with k=23
    unfold quadratic_eq
    norm_num
    sorry
  split
  · -- The other root value
    refl
  · -- Value of k
    refl

end find_other_root_and_k_value_l644_644303


namespace area_of_bounded_region_l644_644955

noncomputable def integral_bounded_area : ℝ :=
∫ x in (0 : ℝ) .. Real.ln 2, Real.sqrt (Real.exp x - 1)

theorem area_of_bounded_region :
  ∫ x in (0 : ℝ) .. Real.ln 2, Real.sqrt (Real.exp x - 1) = 2 - Real.pi / 2 :=
sorry

end area_of_bounded_region_l644_644955


namespace triangle_side_a_l644_644664

-- Definitions based on given conditions
def cosB : ℚ := 5 / 13
def cosC : ℚ := 4 / 5
def sideC : ℚ := 1

-- Define sideA as the side to be proven equal to 21/13
def sideA : ℚ := 21 / 13

-- Lean statement for the proof problem
theorem triangle_side_a (cosB cosC sideC : ℚ) (h1 : cosB = 5 / 13) (h2 : cosC = 4 / 5) (h3 : sideC = 1) : 
  sideA = 21 / 13 :=
by {
  -- Placeholder for the proof
  sorry,
}

end triangle_side_a_l644_644664


namespace moles_of_NaHCO3_used_l644_644257

/-- Chemical reaction conditions -/
variables (n_HNO3 n_NaHCO3 n_CO2 : ℚ)

/-- The balanced chemical equation for the reaction: HNO₃ + NaHCO₃ → CO₂ + H₂O + NaNO₃ -/
def balanced_reaction : Prop :=
  n_HNO3 = 1 ∧ n_CO2 = 1 ∧ n_HNO3 = n_NaHCO3

/-- Prove the number of moles of Sodium bicarbonate used -/
theorem moles_of_NaHCO3_used (h : balanced_reaction n_HNO3 n_NaHCO3 n_CO2) : n_NaHCO3 = 1 :=
by
  sorry


end moles_of_NaHCO3_used_l644_644257


namespace sum_of_squares_iff_double_l644_644399

theorem sum_of_squares_iff_double (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_squares_iff_double_l644_644399


namespace apples_pyramid_l644_644546

noncomputable def total_apples (base_length base_width : ℕ) : ℕ :=
  let rec helper (l w : ℕ) : ℕ :=
    if l = 0 ∨ w = 0 then 0
    else l * w + helper (l - 1) (w - 1)
  helper base_length base_width

theorem apples_pyramid (base_length base_width : ℕ) (h1 : base_length = 6) (h2 : base_width = 9) :
  total_apples base_length base_width = 154 :=
by {
  subst h1,
  subst h2,
  simp [total_apples],
  sorry
}

end apples_pyramid_l644_644546


namespace given_problem_l644_644213

noncomputable def improper_fraction_5_2_7 : ℚ := 37 / 7
noncomputable def improper_fraction_6_1_3 : ℚ := 19 / 3
noncomputable def improper_fraction_3_1_2 : ℚ := 7 / 2
noncomputable def improper_fraction_2_1_5 : ℚ := 11 / 5

theorem given_problem :
  71 * (improper_fraction_5_2_7 - improper_fraction_6_1_3) / (improper_fraction_3_1_2 + improper_fraction_2_1_5) = -13 - 37 / 1197 := 
  sorry

end given_problem_l644_644213


namespace set_intersection_l644_644335

theorem set_intersection (x : ℝ) :
  (x ∈ {x : ℝ | x * (x - 2) < 0}) ∩ (x ∈ {x : ℝ | |x| ≤ 1}) ↔ (0 < x ∧ x ≤ 1) := by
  sorry

end set_intersection_l644_644335


namespace daily_milk_production_l644_644390

-- Definitions of the conditions
def expenses := 3000
def price_per_gallon := 3.55
def total_income_june := 18300
def days_in_june := 30

-- The theorem statement
theorem daily_milk_production : 
  ∃ daily_production, daily_production = 171 ∧
    total_income_june / price_per_gallon ∼ 5154 ∧
    5154 / days_in_june ∼ 171 :=
by
  exists 171
  split
  sorry
  split
  sorry
  sorry

end daily_milk_production_l644_644390


namespace scientific_notation_of_0_0000012_l644_644444

theorem scientific_notation_of_0_0000012 :
  0.0000012 = 1.2 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000012_l644_644444


namespace Sn_formula_l644_644281

noncomputable def sequence (n : ℕ) : ℚ := 
  if n = 1 then 1 else sorry -- Placeholder for the actual definition

noncomputable def Sn (n : ℕ) : ℚ := 
  if n = 1 then sequence 1 else 
    ∑ i in finset.range n, sequence (i+1)

theorem Sn_formula (n : ℕ) (hn : n ≥ 1) : Sn n = 1 / (2 * n - 1) := by
  sorry

end Sn_formula_l644_644281


namespace point_on_x_axis_l644_644201

theorem point_on_x_axis : ∃ p, (p = (-2, 0) ∧ p.snd = 0) ∧
  ((p ≠ (0, 2)) ∧ (p ≠ (-2, -3)) ∧ (p ≠ (-1, -2))) :=
by
  sorry

end point_on_x_axis_l644_644201


namespace sin_value_l644_644282

variable (α m : Real)
variable (P : Real × Real)
variable (cos_α : Real)

-- Given conditions
axiom terminal_side : P = (sqrt 3, m)
axiom cos_condition : cos_α = m / 6

-- The statement we want to prove
theorem sin_value : sin α = sqrt 3 / 2 := sorry

end sin_value_l644_644282


namespace bananas_to_oranges_l644_644208

variables (banana apple orange : Type) 
variables (cost_banana : banana → ℕ) 
variables (cost_apple : apple → ℕ)
variables (cost_orange : orange → ℕ)

-- Conditions given in the problem
axiom cond1 : ∀ (b1 b2 b3 : banana) (a1 a2 : apple), cost_banana b1 = cost_banana b2 → cost_banana b2 = cost_banana b3 → 3 * cost_banana b1 = 2 * cost_apple a1
axiom cond2 : ∀ (a3 a4 a5 a6 : apple) (o1 o2 : orange), cost_apple a3 = cost_apple a4 → cost_apple a4 = cost_apple a5 → cost_apple a5 = cost_apple a6 → 6 * cost_apple a3 = 4 * cost_orange o1

-- Prove that 8 oranges cost as much as 18 bananas
theorem bananas_to_oranges (b1 b2 b3 : banana) (a1 a2 a3 a4 a5 a6 : apple) (o1 o2 : orange) :
    3 * cost_banana b1 = 2 * cost_apple a1 →
    6 * cost_apple a3 = 4 * cost_orange o1 →
    18 * cost_banana b1 = 8 * cost_orange o2 := 
sorry

end bananas_to_oranges_l644_644208


namespace sunflower_seeds_contest_l644_644582

theorem sunflower_seeds_contest 
  (first_player_seeds : ℕ) (second_player_seeds : ℕ) (total_seeds : ℕ) 
  (third_player_seeds : ℕ) (third_more : ℕ) 
  (h1 : first_player_seeds = 78) 
  (h2 : second_player_seeds = 53) 
  (h3 : total_seeds = 214) 
  (h4 : first_player_seeds + second_player_seeds + third_player_seeds = total_seeds) 
  (h5 : third_more = third_player_seeds - second_player_seeds) : 
  third_more = 30 :=
by
  sorry

end sunflower_seeds_contest_l644_644582


namespace proof_sum_f_inv_l644_644434

def g (x : ℝ) : ℝ := if x < 5 then x + 3 else 0
def h (x : ℝ) : ℝ := if x >= 5 then real.sqrt (x - 1) else 0
def f (x : ℝ) : ℝ := if x < 5 then g x else h x

def g_inv (y : ℝ) : ℝ := y - 3
def h_inv (y : ℝ) : ℝ := y^2 + 1
def f_inv (y : ℝ) : ℝ := if y < 8 then g_inv y else h_inv y

theorem proof_sum_f_inv :
  (∑ i in list.range' (-8) (7 + 1 - (-8)), f_inv i) = 182 := by
  sorry

end proof_sum_f_inv_l644_644434


namespace domain_of_tan_arcsin_l644_644252

theorem domain_of_tan_arcsin :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 ↔ is_defined (λ x, Real.tan (Real.arcsin x)) x :=
sorry

end domain_of_tan_arcsin_l644_644252


namespace removed_number_is_24_l644_644300

theorem removed_number_is_24
  (S9 : ℕ) (S8 : ℕ) (avg_9 : ℕ) (avg_8 : ℕ) (h1 : avg_9 = 72) (h2 : avg_8 = 78) (h3 : S9 = avg_9 * 9) (h4 : S8 = avg_8 * 8) :
  S9 - S8 = 24 :=
by
  sorry

end removed_number_is_24_l644_644300


namespace second_largest_of_10_11_12_l644_644841

theorem second_largest_of_10_11_12 : ∃ (n : ℕ), n = 11 ∧ ∀ m ∈ {10, 11, 12}, m ≠ n → m < 12 ∧ m > 10 :=
by
  sorry

end second_largest_of_10_11_12_l644_644841


namespace valid_tree_arrangements_l644_644851

-- Define the types of trees
inductive TreeType
| Birch
| Oak

-- Define the condition that each tree must be adjacent to a tree of the other type
def isValidArrangement (trees : List TreeType) : Prop :=
  ∀ (i : ℕ), i < trees.length - 1 → trees.nthLe i sorry ≠ trees.nthLe (i + 1) sorry

-- Define the main problem
theorem valid_tree_arrangements : ∃ (ways : Nat), ways = 16 ∧
  ∃ (arrangements : List (List TreeType)), arrangements.length = ways ∧
    ∀ arrangement ∈ arrangements, arrangement.length = 7 ∧ isValidArrangement arrangement :=
sorry

end valid_tree_arrangements_l644_644851


namespace perimeter_of_square_36_l644_644814

variable (a s P : ℕ)

def is_square_area : Prop := a = s * s
def is_square_perimeter : Prop := P = 4 * s
def condition : Prop := 5 * a = 10 * P + 45

theorem perimeter_of_square_36 (h1 : is_square_area a s) (h2 : is_square_perimeter P s) (h3 : condition a P) : P = 36 := 
by
  sorry

end perimeter_of_square_36_l644_644814


namespace derivative_of_y_is_tan54x_sec54x_l644_644884
-- Import the necessary library

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (real.tan (real.cos 2))^(1/7) + (real.sin(27*x))^2 / (27 * real.cos(54*x))

-- Prove that the derivative of y is given by the desired expression
theorem derivative_of_y_is_tan54x_sec54x (x : ℝ) : deriv y x = real.tan (54 * x) * real.sec (54 * x) := 
sorry

end derivative_of_y_is_tan54x_sec54x_l644_644884


namespace initial_water_percentage_l644_644386

variable (W : ℝ) -- Initial percentage of water in the milk

theorem initial_water_percentage 
  (final_water_content : ℝ := 2) 
  (pure_milk_added : ℝ := 15) 
  (initial_milk_volume : ℝ := 10)
  (final_mixture_volume : ℝ := initial_milk_volume + pure_milk_added)
  (water_equation : W / 100 * initial_milk_volume = final_water_content / 100 * final_mixture_volume) 
  : W = 5 :=
by
  sorry

end initial_water_percentage_l644_644386


namespace performance_interpretation_l644_644883

-- Define what "Great performance and only a musical genius" implies in context.
def only_musical_genius_perform_successfully (performance: Prop) : Prop :=
  performance → (only (λ x: Type, x = "musical genius") performs_successfully)

-- Define the given performance scenario
def performance_today : Prop := True

-- The correct answer must validate that the correct interpretation follows
theorem performance_interpretation :
  only_musical_genius_perform_successfully performance_today =
  (True → ∀ (x: Type), x ≠ "musical genius" → x does_not_perform_successfully) :=
by {
  sorry
}

end performance_interpretation_l644_644883


namespace exists_two_tangent_circles_l644_644284

-- Define the geometric conditions and tangency
variables {Γ₁ Γ₂ : Circle} (O₁ O₂ : Point)
variables (P : Point)

-- Definition of tangency between circles Γ₁ and Γ₂ at point of tangency T
def tangent_circles (Γ₁ Γ₂ : Circle) (O₁ O₂ : Point) :=
  Γ₁.is_tangent_at O₁ (Γ₂, O₂)

-- Definition of the common tangent line where P lies
def common_tangent (Γ₁ Γ₂ : Circle) (O₁ O₂ : Point) (P : Point) :=
  line_through P.is_perpendicular_to (line_through O₁ O₂)

-- Proof statement: There exist exactly two circles tangent to Γ₁, Γ₂, and passing through P.
theorem exists_two_tangent_circles (Γ₁ Γ₂ : Circle) (O₁ O₂ : Point) (T P : Point) 
  (h1 : tangent_circles Γ₁ Γ₂ O₁ O₂) (h2 : common_tangent Γ₁ Γ₂ O₁ O₂ P) :
  ∃! ω₁ ω₂ : Circle, (ω₁.is_tangent_to Γ₁) ∧ (ω₁.is_tangent_to Γ₂) ∧ (P ∈ ω₁) ∧
                 (ω₂.is_tangent_to Γ₁) ∧ (ω₂.is_tangent_to Γ₂) ∧ (P ∈ ω₂) ∧ (ω₁ ≠ ω₂) :=
sorry

end exists_two_tangent_circles_l644_644284


namespace necessary_but_not_sufficient_l644_644786

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def P : Set ℝ := {x | x ≤ -1}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧ (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
by
  sorry

end necessary_but_not_sufficient_l644_644786


namespace altitude_length_l644_644739

theorem altitude_length 
    {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (AB BC AC : ℝ) (hAC : 𝕜) 
    (h₀ : AB = 8)
    (h₁ : BC = 7)
    (h₂ : AC = 5) :
  h = (5 * Real.sqrt 3) / 2 :=
sorry

end altitude_length_l644_644739


namespace four_played_games_l644_644107

theorem four_played_games
  (A B C D E : Prop)
  (A_answer : ¬A)
  (B_answer : A ∧ ¬B)
  (C_answer : B ∧ ¬C)
  (D_answer : C ∧ ¬D)
  (E_answer : D ∧ ¬E)
  (truth_condition : (¬A ∧ ¬B) ∨ (¬B ∧ ¬C) ∨ (¬C ∧ ¬D) ∨ (¬D ∧ ¬E)) :
  A ∨ B ∨ C ∨ D ∧ E := sorry

end four_played_games_l644_644107


namespace concentration_time_within_bounds_l644_644438

-- Define the time bounds for the highest concentration of the drug in the blood
def highest_concentration_time_lower (base : ℝ) (tolerance : ℝ) : ℝ := base - tolerance
def highest_concentration_time_upper (base : ℝ) (tolerance : ℝ) : ℝ := base + tolerance

-- Define the base and tolerance values
def base_time : ℝ := 0.65
def tolerance_time : ℝ := 0.15

-- Define the specific time we want to prove is within the bounds
def specific_time : ℝ := 0.8

-- Theorem statement
theorem concentration_time_within_bounds : 
  highest_concentration_time_lower base_time tolerance_time ≤ specific_time ∧ 
  specific_time ≤ highest_concentration_time_upper base_time tolerance_time :=
by sorry

end concentration_time_within_bounds_l644_644438


namespace incorrect_order_count_is_eleven_l644_644735

noncomputable def word := "good"
def incorrect_orders_count (word: String) : Nat :=
  let permutations := word.toList.permutations
  let incorrect_permutations := permutations.filter (λ perm, String.mk perm ≠ word)
  incorrect_permutations.length

theorem incorrect_order_count_is_eleven (word : String) (h_word_good : word = "good") :
  incorrect_orders_count word = 11 := by
  sorry

end incorrect_order_count_is_eleven_l644_644735


namespace parabola_vertex_point_sum_l644_644622

theorem parabola_vertex_point_sum (a b c : ℚ) 
  (h1 : ∃ (a b c : ℚ), ∀ x : ℚ, (y = a * x ^ 2 + b * x + c) = (y = - (1 / 3) * (x - 5) ^ 2 + 3)) 
  (h2 : ∀ x : ℚ, ((x = 2) ∧ (y = 0)) → (0 = a * 2 ^ 2 + b * 2 + c)) :
  a + b + c = -7 / 3 := 
sorry

end parabola_vertex_point_sum_l644_644622


namespace shipping_packages_min_lcm_l644_644803

theorem shipping_packages_min_lcm : 
  let sarah := 18;
  let ryan := 11;
  let emily := 15;
  Nat.lcm (Nat.lcm sarah ryan) emily = 990 :=
by
  let sarah := 18;
  let ryan := 11;
  let emily := 15;
  have h1 : Nat.prime 11 := by sorry
  have h2 : Nat.gcd_succ 18 11 = 1 := by sorry
  have h3 : (2^(1:ℕ) * 3^(2:ℕ) * 5^(1:ℕ) * 11^(1:ℕ)) = 990 := by sorry
  exact sorry

end shipping_packages_min_lcm_l644_644803


namespace internal_diagonal_cubes_l644_644891

def gcd (a b : ℕ) : ℕ :=
if b = 0 then a else gcd b (a % b)

def inclusion_exclusion (a b c : ℕ) (ab bc ca abc : ℕ) : ℕ :=
a + b + c - ab - bc - ca + abc

theorem internal_diagonal_cubes :
  let a := 120
  let b := 270
  let c := 300
  let ab := gcd a b
  let bc := gcd b c
  let ca := gcd c a
  let abc := gcd (gcd a b) c
  inclusion_exclusion a b c ab bc ca abc = 600 := by
  sorry

end internal_diagonal_cubes_l644_644891


namespace find_f2_l644_644731

theorem find_f2 
  (f : ℕ → ℝ)
  (h : ∀ x : ℕ, f(x + 1) = x^3 - x + 1) : 
  f 2 = 1 := 
sorry

end find_f2_l644_644731


namespace correct_operation_l644_644490

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l644_644490


namespace inequality_am_gm_l644_644677

theorem inequality_am_gm 
  (a b c d : ℝ) 
  (h_nonneg_a : 0 ≤ a) 
  (h_nonneg_b : 0 ≤ b) 
  (h_nonneg_c : 0 ≤ c) 
  (h_nonneg_d : 0 ≤ d) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 :=
by
  sorry


end inequality_am_gm_l644_644677


namespace sum_solutions_l644_644865

theorem sum_solutions (x : ℝ) (h : 6 * x / 18 = 9 / x) : Σ S, S = { y |  y^2 = 27 } -> ∑ y in S, y = 0 := by
sorry

end sum_solutions_l644_644865


namespace probability_of_odd_number_l644_644426

-- Define the total number of digits
def digits : List ℕ := [2, 3, 5, 7, 9]

-- Define what it means for a number to be odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the problem of arranging the digits into a five-digit number that is odd
def num_of_arrangements (d : List ℕ) : ℕ := 5.factorial

def num_of_odd_arrangements (d : List ℕ) : ℕ := 4 * 4.factorial

-- Define the probability 
def probability_odd (d : List ℕ) : ℚ := num_of_odd_arrangements d / num_of_arrangements d

-- Statement: Prove that the probability is 4/5
theorem probability_of_odd_number : probability_odd digits = 4 / 5 := by
  sorry

end probability_of_odd_number_l644_644426


namespace number_of_ways_to_choose_officers_l644_644901

noncomputable def ways_to_choose_officers (members : ℕ) (founding_members : ℕ) : ℕ :=
  let num_positions := 5
  let choices := [founding_members, members-1, members-2, members-3, members-4]
  choices.foldl (*) 1

theorem number_of_ways_to_choose_officers :
  ways_to_choose_officers 12 4 = 25920 :=
by
  simp [ways_to_choose_officers]
  sorry

end number_of_ways_to_choose_officers_l644_644901


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644064

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644064


namespace arithmetic_seq_geom_eq_div_l644_644665

noncomputable def a (n : ℕ) (a1 d : ℝ) : ℝ := a1 + n * d

theorem arithmetic_seq_geom_eq_div (a1 d : ℝ) (h1 : d ≠ 0) (h2 : a1 ≠ 0) 
    (h_geom : (a 3 a1 d) ^ 2 = (a 1 a1 d) * (a 7 a1 d)) :
    (a 2 a1 d + a 5 a1 d + a 8 a1 d) / (a 3 a1 d + a 4 a1 d) = 2 := 
by
  sorry

end arithmetic_seq_geom_eq_div_l644_644665


namespace pyramid_base_length_l644_644562

theorem pyramid_base_length
  (AB BC AD AE BE : ℝ)
  (h1 : AB = 4)
  (h2 : ∠ADE = 45)
  (h3 : ∠BCE = 60) :
  BC = 2 * sqrt 2 :=
sorry

end pyramid_base_length_l644_644562


namespace eccentricity_of_curve_l644_644821

theorem eccentricity_of_curve : 
  (∃ ρ θ : ℝ, ρ^2 * cos (2 * θ) = 1) → ∃ e : ℝ, e = sqrt 2 :=
by
  intro h
  exists sqrt 2
  sorry

end eccentricity_of_curve_l644_644821


namespace math_problem_l644_644496

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l644_644496


namespace simplified_fraction_sum_l644_644141

theorem simplified_fraction_sum : 
  let numerator := 64
  let denominator := 96
  let gcd := Nat.gcd numerator denominator
  let simplest_numerator := numerator / gcd
  let simplest_denominator := denominator / gcd
in simplest_numerator + simplest_denominator = 5 := 
by
  let numerator := 64
  let denominator := 96
  let gcd := Nat.gcd numerator denominator
  let simplest_numerator := numerator / gcd
  let simplest_denominator := denominator / gcd
  have h : simplest_numerator + simplest_denominator = 5 := sorry
  exact h

end simplified_fraction_sum_l644_644141


namespace bill_harry_combined_l644_644950

-- Definitions based on the given conditions
def sue_nuts := 48
def harry_nuts := 2 * sue_nuts
def bill_nuts := 6 * harry_nuts

-- The theorem we want to prove
theorem bill_harry_combined : bill_nuts + harry_nuts = 672 :=
by
  sorry

end bill_harry_combined_l644_644950


namespace sin_square_general_proposition_l644_644292

-- Definitions for the given conditions
def sin_square_sum_30_90_150 : Prop :=
  (Real.sin (30 * Real.pi / 180))^2 + (Real.sin (90 * Real.pi / 180))^2 + (Real.sin (150 * Real.pi / 180))^2 = 3/2

def sin_square_sum_5_65_125 : Prop :=
  (Real.sin (5 * Real.pi / 180))^2 + (Real.sin (65 * Real.pi / 180))^2 + (Real.sin (125 * Real.pi / 180))^2 = 3/2

-- The general proposition we want to prove
theorem sin_square_general_proposition (α : ℝ) : 
  sin_square_sum_30_90_150 ∧ sin_square_sum_5_65_125 →
  (Real.sin (α * Real.pi / 180 - 60 * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180 + 60 * Real.pi / 180))^2 = 3/2 :=
by
  intro h
  -- Proof goes here
  sorry

end sin_square_general_proposition_l644_644292


namespace shane_photos_per_week_l644_644040

theorem shane_photos_per_week (jan_days feb_weeks photos_per_day total_photos : ℕ) :
  (jan_days = 31) →
  (feb_weeks = 4) →
  (photos_per_day = 2) →
  (total_photos = 146) →
  let photos_jan := photos_per_day * jan_days in
  let photos_feb := total_photos - photos_jan in
  let photos_per_week := photos_feb / feb_weeks in
  photos_per_week = 21 :=
by
  intros h1 h2 h3 h4
  let photos_jan := photos_per_day * jan_days
  let photos_feb := total_photos - photos_jan
  let photos_per_week := photos_feb / feb_weeks
  sorry

end shane_photos_per_week_l644_644040


namespace prob_abs_diff_gt_one_is_three_over_eight_l644_644405

noncomputable def prob_abs_diff_gt_one : ℝ := sorry

theorem prob_abs_diff_gt_one_is_three_over_eight :
  prob_abs_diff_gt_one = 3 / 8 :=
by sorry

end prob_abs_diff_gt_one_is_three_over_eight_l644_644405


namespace solve_double_inequality_l644_644807

/-- Proof problem: Solve the double inequality for x:
    -2 < (x^2 - 16x + 15) / (x^2 - 2x + 5) < 1. -/
theorem solve_double_inequality (x : ℝ) :
  -2 < (x^2 - 16x + 15) / (x^2 - 2x + 5) ∧ (x^2 - 16x + 15) / (x^2 - 2x + 5) < 1 ↔
  (5 / 7 < x ∧ x < 5 / 3) ∨ 5 < x :=
sorry

end solve_double_inequality_l644_644807


namespace inequality_sqrt_sum_l644_644028

theorem inequality_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (1 / a) + (1 / b) + (1 / c) = 3) :
  (1 / (sqrt (a^3 + b))) + (1 / (sqrt (b^3 + c))) + (1 / (sqrt (c^3 + a))) ≤ (3 / (sqrt 2)) :=
by
  sorry

end inequality_sqrt_sum_l644_644028


namespace units_digit_division_l644_644479

theorem units_digit_division (a b c d e denom : ℕ)
  (h30 : a = 30) (h31 : b = 31) (h32 : c = 32) (h33 : d = 33) (h34 : e = 34)
  (h120 : denom = 120) :
  ((a * b * c * d * e) / denom) % 10 = 4 :=
by
  sorry

end units_digit_division_l644_644479


namespace lucas_units_digit_L_L10_eq_4_l644_644813

def L : ℕ → ℕ
| 0       := 3
| 1       := 1
| (n + 2) := (L (n + 1)) + (L n)

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem lucas_units_digit_L_L10_eq_4 : units_digit (L (L 10)) = 4 := by
  sorry

end lucas_units_digit_L_L10_eq_4_l644_644813


namespace ian_lottery_win_l644_644328

theorem ian_lottery_win 
  (amount_paid_to_colin : ℕ)
  (amount_left : ℕ)
  (amount_paid_to_helen : ℕ := 2 * amount_paid_to_colin)
  (amount_paid_to_benedict : ℕ := amount_paid_to_helen / 2)
  (total_debts_paid : ℕ := amount_paid_to_colin + amount_paid_to_helen + amount_paid_to_benedict)
  (total_money_won : ℕ := total_debts_paid + amount_left)
  (h1 : amount_paid_to_colin = 20)
  (h2 : amount_left = 20) :
  total_money_won = 100 := 
sorry

end ian_lottery_win_l644_644328


namespace number_of_poly_lines_l644_644114

def nonSelfIntersectingPolyLines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n - 3)
  else 0

theorem number_of_poly_lines (n : ℕ) (h : n > 1) :
  nonSelfIntersectingPolyLines n =
  if n = 2 then 1 else n * 2^(n - 3) :=
by sorry

end number_of_poly_lines_l644_644114


namespace trigonometric_value_l644_644650

theorem trigonometric_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - Real.pi / 4)) = 13 / 4 := 
sorry

end trigonometric_value_l644_644650


namespace more_cats_than_dogs_l644_644590

def initial_counts :=
  {cats: Nat // cats = 48} ×
  {dogs: Nat // dogs = 36} ×
  {rabbits: Nat // rabbits = 10} ×
  {parrots: Nat // parrots = 5}

def round1 (c: {cats: Nat // cats = 48}) (r: {rabbits: Nat // rabbits = 10}) :=
  {cats := c.val - 6, rabbits := r.val - 2}

def round2 (c: Nat) (d: {dogs: Nat // dogs = 36}) (p: {parrots: Nat // parrots = 5}) :=
  {cats := c - 12, dogs := d.val - 8, parrots := p.val - 2}

def round3 (c: Nat) (r: Nat) (p: Nat) :=
  {cats := c - 8, rabbits := r - 4, parrots := p - 1}

def round4 (d: Nat) (r: Nat) :=
  {dogs := d - 5, rabbits := r - 2}

theorem more_cats_than_dogs :
  let c_0 := (48 : Nat);
  let d_0 := (36 : Nat);
  let r_0 := (10 : Nat);
  let p_0 := (5 : Nat);
  let ⟨_, _, r1⟩ := round1 ⟨c_0, rfl⟩ ⟨r_0, rfl⟩;
  let ⟨c2, d1, p1⟩ := round2 r1.cats ⟨d_0, rfl⟩ ⟨p_0, rfl⟩;
  let ⟨c3, _, p2⟩ := round3 c2 d1.cats p1;
  let ⟨df, _⟩ := round4 d1.cats p2.rabbits
  in c3 - df.dogs = -1 :=
by
  sorry

end more_cats_than_dogs_l644_644590


namespace find_diagonal_squared_l644_644768

variable {E F G H X Y Z W : Type}

-- Conditions
def parallelogram (A B C D : Type) := True -- A place-holder definition (should be defined properly in a complete formalization)

-- Hypotheses
hypotheses
  (h1 : parallelogram E F G H)
  (h2 : area E F G H = 24)
  (h3 : XY = 8)
  (h4 : ZW = 10)

-- Problem statement
theorem find_diagonal_squared (d : ℝ) (h : d^2 = 37) : ∃ d, d^2 = 37 :=
by
  sorry

end find_diagonal_squared_l644_644768


namespace quadratic_factorization_sum_l644_644432

theorem quadratic_factorization_sum (d e f : ℤ) (h1 : ∀ x, x^2 + 18 * x + 80 = (x + d) * (x + e)) 
                                     (h2 : ∀ x, x^2 - 20 * x + 96 = (x - e) * (x - f)) : 
                                     d + e + f = 30 :=
by
  sorry

end quadratic_factorization_sum_l644_644432


namespace compute_3_l644_644621

theorem compute_3^3_mul_6^3 : 3^3 * 6^3 = 5832 :=
  by
  sorry

end compute_3_l644_644621


namespace good_number_intervals_sum_l644_644271

noncomputable def a_n (x : ℝ) (n : ℕ) : ℝ :=
  n * x / List.prod (List.range n).map (λ k, 1 - (k + 1) * x)

def is_good_number (x : ℝ) : Prop :=
  (x ∈ set.Ioo 0 1) ∧
  (¬ ∃ k : ℕ, 1 / x = k) ∧
  (∑ i in Finset.range 10, a_n x (i + 1) > -1) ∧
  (∏ i in Finset.range 10, a_n x (i + 1) > 0)

theorem good_number_intervals_sum :
  let intervals := [
    set.Ioo 0 (1/10 : ℝ), 
    set.Ioo (1/7) (1/6), 
    set.Ioo (1/3) (1/2)]
  in
  (∑ (l, u) in intervals.map (λ I, (I.lower, I.upper)), (u - l)) = (61 / 210 : ℝ) :=
sorry

end good_number_intervals_sum_l644_644271


namespace trajectory_of_P_line_AC_fixed_point_l644_644287

-- Problem (Ⅰ)
theorem trajectory_of_P :
  let F1 : ℝ × ℝ := (-1, 0)
  let F2 : ℝ × ℝ := (1, 0)
  ∃ G : set (ℝ × ℝ), (∀ M : ℝ × ℝ, dist M F2 = 2 * real.sqrt 2 → 
                       ∃ P : ℝ × ℝ, P ∈ G ∧
                       (dist P M = dist P F1 ∧ dist P F2 + dist P F1 = 2 * real.sqrt 2))
  → G = {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} := sorry

-- Problem (Ⅱ)
theorem line_AC_fixed_point (G : set (ℝ × ℝ))
  (hG : G = {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}) :
  ∀ (L : ℝ × ℝ → ℝ × ℝ → Prop) (A B : ℝ × ℝ),
  (L F2 A ∧ A ∈ G) ∧ (L F2 B ∧ B ∈ G) ∧ (∃ x : ℝ, (B.1 = x ∧ B.2 = 0)) →
  ∃ C : ℝ × ℝ, C.1 = 2 ∧ C.2 = B.2 ∧ 
  ∃ F : ℝ × ℝ → ℝ × ℝ, F A C → F (1.5, 0) :=
sorry

end trajectory_of_P_line_AC_fixed_point_l644_644287


namespace tank_fraction_full_l644_644715

theorem tank_fraction_full 
  (initial_fraction : ℚ)
  (full_capacity : ℚ)
  (added_water : ℚ)
  (initial_fraction_eq : initial_fraction = 3/4)
  (full_capacity_eq : full_capacity = 40)
  (added_water_eq : added_water = 5) :
  ((initial_fraction * full_capacity + added_water) / full_capacity) = 7/8 :=
by 
  sorry

end tank_fraction_full_l644_644715


namespace frog_escape_probability_l644_644457

-- Define the probability function
def prob_escape : ℕ → ℚ
| 0     := 0
| 12    := 1
| n     := if (0 < n ∧ n < 12) then (n / 12) * prob_escape (n - 1) + (1 - n / 12) * prob_escape (n + 1) else 0

-- The theorem to be proven
noncomputable def P2 : ℚ := (prob_escape 2)

theorem frog_escape_probability :
  P2 = 279 / 598 := sorry

end frog_escape_probability_l644_644457


namespace intersection_is_correct_l644_644366

def M := {x : ℝ | log x > 0}
def N := {x : ℝ | x^2 ≤ 4}
def intersection := {x : ℝ | x > 1 ∧ x ≤ 2}

theorem intersection_is_correct : M ∩ N = intersection := 
by {
  sorry
}

end intersection_is_correct_l644_644366


namespace playerB_wins_l644_644155

/- Define the basic concepts and the problem -/
def outcome := List Char -- outcome of the coin flips
def flip := outcome -- the flipping sequence

def O := 'O' -- heads
def P := 'P' -- tails

def "OOO (three times O)" := [O, O, O]
def "OPO" := [O, P, O]

/-- Define event occurrences in sequences -/
noncomputable def occurs_before (seq1 seq2 : outcome) (flips : flip) : Prop :=
  ∃ i j, i < j ∧ flips.drop(i).take(seq1.length) = seq1 ∧ flips.drop(j).take(seq2.length) = seq2

/-- Statement to prove that the probability of "OPO" occurring before "OOO" is
    greater than the probability of "OOO" occurring before "OPO" -/
theorem playerB_wins (flips : flip) :
  (∃ i, occurs_before "OPO" "OOO" flips) →
  (∃ j, occurs_before "OOO" "OPO" flips) →
  (∃ i, occurs_before "OPO" "OOO" flips) ∧ ¬ (∃ j, occurs_before "OOO" "OPO" flips) :=
sorry

end playerB_wins_l644_644155


namespace inequality_solution_m_range_l644_644532

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a = 1 → f x + a - 1 > 0 ↔ x ≠ 2) ∧
  (a > 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ True) ∧
  (a < 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ x < a + 1 ∨ x > 3 - a) :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 5 :=
by
  sorry

end inequality_solution_m_range_l644_644532


namespace probability_of_odd_five_digit_number_l644_644424

theorem probability_of_odd_five_digit_number :
  let digits := {2, 3, 5, 7, 9}
  let n := 5
  let odd_digits := {3, 5, 7, 9}
  (|odd_digits|: ℝ) / (|digits|: ℝ) = 4 / 5 :=
by
  sorry

end probability_of_odd_five_digit_number_l644_644424


namespace sufficient_but_not_necessary_condition_converse_not_true_x_sufficient_but_not_necessary_for_x_squared_l644_644528

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x < -1) : x^2 - 1 > 0 :=
by {
  have hx_pos : x^2 > 1, {
    calc x^2 = (x * x): by norm_num
      ...  > 1 : by {
        apply mul_self_pos,
        linarith,
        },
    },
  linarith,
}

-- Now, we also need to show the converse does not hold.
theorem converse_not_true (x : ℝ) (h : x^2 - 1 > 0) : ¬ (x < -1) -> ( 1 < x) :=
by {
  contrapose!,
  intro hn,
  have hp: 0 <= x^2,
  {
    apply pow_two_nonneg,
  },
  apply lt_irrefl 1,
  linarith,
}

-- Combining the two results to state that x < -1 is a sufficient but not necessary condition for x^2 - 1 > 0
theorem x_sufficient_but_not_necessary_for_x_squared (x : ℝ) : (x < -1 -> x^2 - 1 > 0) ∧ ¬ (x^2 - 1 > 0 -> x < -1) :=
by {
  split,
  { exact sufficient_but_not_necessary_condition x, },
  { exact converse_not_true x, },
}

end sufficient_but_not_necessary_condition_converse_not_true_x_sufficient_but_not_necessary_for_x_squared_l644_644528


namespace midpoint_correct_l644_644230

open Complex

def midpoint_of_segment (z1 z2 : ℂ) : ℂ :=
  (z1 + z2) / 2

theorem midpoint_correct :
  let z1 := -7 + 6 * Complex.i
  let z2 := 5 - 3 * Complex.i
  let midpoint := midpoint_of_segment z1 z2
  midpoint = -1 + 1.5 * Complex.i :=
by
  let z1 := -7 + 6 * i
  let z2 := 5 - 3 * i
  let midpoint := midpoint_of_segment z1 z2
  show midpoint = -1 + 1.5 * i
  sorry

end midpoint_correct_l644_644230


namespace count_triples_is_correct_l644_644770

def S : set ℕ := {n | n ∈ (set.Icc 1 29)}

def relation (a b : ℕ) : Prop := 
  (0 < a - b ∧ a - b ≤ 14) ∨ (b - a > 14)

def count_ordered_triples : ℕ :=
  (set.prod (set.prod S S) S).to_finset.filter (λ t, relation t.1.1 t.1.2 ∧ relation t.1.2 t.2 ∧ relation t.2 t.1.1).card

theorem count_triples_is_correct : count_ordered_triples = 4278 :=
  sorry

end count_triples_is_correct_l644_644770


namespace largest_angle_in_triangle_with_altitudes_l644_644571

theorem largest_angle_in_triangle_with_altitudes (a b c : ℝ) (h₁ : 9 * a = 12 * b = 18 * c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (C : ℝ), C = 109.47 ∧ C = real.acos ((b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) :=
begin
  sorry
end

end largest_angle_in_triangle_with_altitudes_l644_644571


namespace ellipse_trajectory_condition_l644_644886

theorem ellipse_trajectory_condition {F1 F2 M : Type*}  [metric_space M]
  (c : ℝ) (h1 : ∀ p, dist p F1 + dist p F2 = c) : 
  (∃ a, ∀ q, (dist q F1 + dist q F2 = 2 * a) ↔ (dist q F1 + dist q F2 = c)) :=
sorry

end ellipse_trajectory_condition_l644_644886


namespace sum_of_coordinates_D_l644_644795

def point (x y : ℝ) := (x, y)
def midpoint (p1 p2 : ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def A := point 2 8
def B := point 0 0
def C := point 4 2

theorem sum_of_coordinates_D (a b : ℝ)
  (D := point a b)
  (in_first_quadrant : 0 < a ∧ 0 < b)
  (mid_AB := midpoint A B)
  (mid_BC := midpoint B C)
  (mid_CD := midpoint C D)
  (mid_DA := midpoint D A)
  (is_rectangle : True) : -- This is a place holder, you can define the condition that the midpoints form a rectangle
  a + b = 8 :=
sorry

end sum_of_coordinates_D_l644_644795


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644067

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644067


namespace scientific_notation_of_0_0000012_l644_644443

theorem scientific_notation_of_0_0000012 :
  0.0000012 = 1.2 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000012_l644_644443


namespace percentage_increase_is_50_l644_644100

-- Define the conditions
variables {P : ℝ} {x : ℝ}

-- Define the main statement (goal)
theorem percentage_increase_is_50 (h : 0.80 * P + (0.008 * x * P) = 1.20 * P) : x = 50 :=
sorry  -- Skip the proof as per instruction

end percentage_increase_is_50_l644_644100


namespace probability_odd_number_l644_644429

theorem probability_odd_number 
  (digits : Finset ℕ) 
  (odd_digits : Finset ℕ)
  (total_digits : ∀d, d ∈ digits → d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9)
  (total_odd_digits : ∀d, d ∈ odd_digits → d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9) :
  card odd_digits / card digits = 4 / 5 :=
by sorry

end probability_odd_number_l644_644429


namespace ratio_u_v_l644_644345

variable (x y z u v : ℝ)
variable (k : ℝ) (h_pos : k > 0)
variable (x_eq : x = 2 * k) (y_eq : y = 5 * k)
variable (right_triangle : z = Real.sqrt (x^2 + y^2))
variable (perpendicular : u = x^2 / z ∧ v = y^2 / z)

theorem ratio_u_v :
  x : y = 2 : 5 → (u / v) = 4 / 25 :=
by
  sorry

end ratio_u_v_l644_644345


namespace problem_solution_l644_644686

def Point := (ℝ × ℝ)

def A : Point := (1, 1)
def B : Point := (3, 2)
def C : Point := (5, 4)

def slope (p1 p2 : Point) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

def k_AB : ℝ := slope A B

def perpendicular_slope (k : ℝ) : ℝ := -1 / k

def equation_of_line (p : Point) (m : ℝ) : (ℝ × ℝ × ℝ) :=
  -- Standard form ax + by + c = 0
  let a := -m
  let b := 1
  let c := -(p.2 - m * p.1)
  (a, b, c)

def line_altitude_AB := equation_of_line C (perpendicular_slope k_AB)

def parallel_line (l : Point) (slope : ℝ) : (ℝ × ℝ × ℝ) :=
  let a := -slope
  let b := 1
  let c := -l.2 + slope * l.1
  (a, b, c)

def k_AC : ℝ := slope A C

noncomputable def find_parallel_line_equation_and_check_intercepts : Bool :=
  let a := -3 / 7
  let x_intercept := -a / (a + 1)
  let y_intercept := a
  let intercept_greater := x_intercept > y_intercept
  intercept_greater

def perimeter_of_triangle : ℝ := (find_parallel_line_equation_and_check_intercepts) ? 
  (((4/7) + (3/7) + 5/7)) sorry

theorem problem_solution :
  (line_altitude_AB = (2, 1, -14)) ∧ (perimeter_of_triangle = 12 / 7) :=
by
  sorry

end problem_solution_l644_644686


namespace interest_rate_first_part_l644_644034

theorem interest_rate_first_part :
  ∃ r : ℝ, 
    ∀ (P1 P2 : ℝ),
      (P1 + P2 = 3000) →
      (P1 ≈ 299.99999999999994) →
      (P2 = 3000 - P1) →
      (P1 * r / 100 + P2 * 5 / 100 = 144) →
      r = 3 :=
by
  sorry

end interest_rate_first_part_l644_644034


namespace real_values_satisfying_condition_l644_644265

noncomputable def num_real_values_satisfying_condition : ℝ :=
  number_of_reals (λ d : ℝ, abs (1/3 - d * complex.I) = 2/3) = 2

theorem real_values_satisfying_condition : num_real_values_satisfying_condition = 2 :=
sorry

end real_values_satisfying_condition_l644_644265


namespace problem_1_problem_2_l644_644153

open Real

variables {a b c d p S : ℝ} {A C : ℝ}

-- Definition of the semiperimeter
def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- Problem Part 1
theorem problem_1 (a b c d S A C : ℝ)
  (h1 : p = semiperimeter a b c d)
  (h2 : S = 1/2 * (a * d * sin A + b * c * sin C)) :
  16 * S^2 = 4 * (b^2 * c^2 + a^2 * d^2) - 
             8 * a * b * c * d * cos (A + C) - 
             (b^2 + c^2 - a^2 - d^2)^2 := sorry

-- Problem Part 2
theorem problem_2 (a b c d S A C : ℝ)
  (h1 : p = semiperimeter a b c d)
  (h2 : S = sqrt ((p - a)* (p - b) * (p - c) * (p - d) - a * b * c * d * cos^2((A + C) / 2))) :
  S = sqrt ((p - a)* (p - b) * (p - c) * (p - d) - a * b * c * d * cos^2((A + C) / 2)) := sorry

end problem_1_problem_2_l644_644153


namespace total_savings_correct_l644_644852

-- Define the savings of Sam, Victory and Alex according to the given conditions
def sam_savings : ℕ := 1200
def victory_savings : ℕ := sam_savings - 200
def alex_savings : ℕ := 2 * victory_savings

-- Define the total savings
def total_savings : ℕ := sam_savings + victory_savings + alex_savings

-- The theorem to prove the total savings
theorem total_savings_correct : total_savings = 4200 :=
by
  sorry

end total_savings_correct_l644_644852


namespace factorize_difference_of_squares_l644_644990

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4*x^2 = (3 - 2*x) * (3 + 2*x) :=
by
  sorry

end factorize_difference_of_squares_l644_644990


namespace hyperbola_k_range_l644_644435

theorem hyperbola_k_range (k : ℝ) :
  (∃ x y : ℝ, ((x^2 / (2 - k)) + (y^2 / (k - 1)) = 1)) ∧ 
  (∃ a b : ℝ, (a ≠ b)) →
  (k < 1 ∨ k > 2) :=
begin
  intro h,
  -- skipping proof for now
  sorry
end

end hyperbola_k_range_l644_644435


namespace xyz_value_l644_644009

-- Variables for Complex numbers
variables {x y z : ℂ}

-- Definitions based on given conditions
def cond1 := x * y + 5 * y = -20
def cond2 := y * z + 5 * z = -20
def cond3 := z * x + 5 * x = -20

-- Theorem stating the result we need to prove
theorem xyz_value : 
  cond1 ∧ cond2 ∧ cond3 → x * y * z = 100 :=
by
  sorry -- proof goes here

end xyz_value_l644_644009


namespace factorize_difference_of_squares_l644_644992

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2 * x) * (3 + 2 * x) :=
sorry

end factorize_difference_of_squares_l644_644992


namespace cos_of_7pi_over_4_l644_644995

theorem cos_of_7pi_over_4 : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 :=
by
  sorry

end cos_of_7pi_over_4_l644_644995


namespace no_valid_a_l644_644227

theorem no_valid_a : ¬ ∃ (a : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 
  ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 2 * x₁^2 + (3 * a + 1) * x₁ + a^2 = 0 ∧ 2 * x₂^2 + (3 * a + 1) * x₂ + a^2 = 0 :=
by {
  sorry
}

end no_valid_a_l644_644227


namespace function_satisfies_equation_l644_644408

theorem function_satisfies_equation (y : ℝ → ℝ) (h : ∀ x : ℝ, y x = Real.exp (x + x^2) + 2 * Real.exp x) :
  ∀ x : ℝ, deriv y x - y x = 2 * x * Real.exp (x + x^2) :=
by {
  sorry
}

end function_satisfies_equation_l644_644408


namespace sqrt_sum_of_4_cube_eq_8_sqrt_3_l644_644138

theorem sqrt_sum_of_4_cube_eq_8_sqrt_3 : sqrt (4^3 + 4^3 + 4^3) = 8 * sqrt 3 := 
by 
suffices h : 4^3 = 64, 
begin
  rw [h, h, h],    -- replace 4^3 with 64 three times
  suffices h2 : 3 * 64 = 192, 
    by 
    rw [← h2, Nat.sqrt_mul, Nat.sqrt_eq_iff_mul_self_eq, Nat.sqrt_eq_iff_mul_self_eq],
    norm_num,
  simp [h],
  norm_num,
end

end sqrt_sum_of_4_cube_eq_8_sqrt_3_l644_644138


namespace xiao_ming_first_error_in_step_one_l644_644874

theorem xiao_ming_first_error_in_step_one 
  (x : ℝ)
  (H_initial : -((3*x + 1) / 2) - ((2*x - 5) / 6) > 1) :
  (∀ a b, a * b = 6 → (a ≠ 2 ∨ b ≠ 3) ∧ (a ≠ 3 ∨ b ≠ 2) → false)
  ∧ (∀ c, 3 * (3*x + 1) + (2*x - 5) ≠ -6)
  := by sorry

end xiao_ming_first_error_in_step_one_l644_644874


namespace example_theorem_l644_644004

noncomputable def f (a : Fin (n + 1) → ℝ) := 
  ∑ i in Finset.range (n + 1), a i * (X ^ i)

noncomputable def f_squared (a : Fin (n + 1) → ℝ) := 
  (f a)^2

theorem example_theorem (a : Fin (n + 1) → ℝ) (h : ∀ (i : Fin (n + 1)), 0 ≤ a i ∧ a i ≤ a 0) :
  let b := λ i, (f_squared a).coeff i in
  b (n + 1) ≤ (1/2) * (f a).eval 1 ^ 2 :=
by
  sorry

end example_theorem_l644_644004


namespace probability_spinner_lands_on_non_prime_is_one_third_l644_644569

-- Definitions for the spinner sections
def sections : List ℕ := [7, 8, 11, 13, 14, 17]

-- Definition for checking if a number is prime
def is_prime (n : ℕ) : Bool :=
  if n < 2 then false
  else List.range' 2 (n - 2 + 1) |>.all (fun m => n % m ≠ 0)

-- Definition for checking if a number is not prime
def is_non_prime (n : ℕ) : Bool :=
  ¬ is_prime n

-- Count of non-prime numbers in the sections
def count_non_prime : ℕ :=
  sections.countp is_non_prime

-- Total number of sections
def total_sections : ℕ := sections.length

-- The probability to be proved
def probability_non_prime : ℚ :=
  count_non_prime / total_sections

theorem probability_spinner_lands_on_non_prime_is_one_third :
  probability_non_prime = 1 / 3 :=
by sorry

end probability_spinner_lands_on_non_prime_is_one_third_l644_644569


namespace parabola_minimum_distance_l644_644658

theorem parabola_minimum_distance:
  let d1 (P : ℝ × ℝ) := real.dist P (real.sqrt (-(4 * P.1)))
  let d2 (P : ℝ × ℝ) := real.dist P (4 - P.1 - P.2)
  ∃ P, P ∈ parabola ∧ (d1 P + d2 P) = 5 * real.sqrt 2 / 2
:= sorry

end parabola_minimum_distance_l644_644658


namespace equal_elements_l644_644279

theorem equal_elements {n : ℕ} (a : ℕ → ℝ) (h₁ : n ≥ 2) (h₂ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≠ -1) 
  (h₃ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1)) 
  (hn1 : a (n + 1) = a 1) (hn2 : a (n + 2) = a 2) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i = a 1 := by
  sorry

end equal_elements_l644_644279


namespace marked_price_l644_644547

theorem marked_price (initial_price : ℝ) (discount_percent : ℝ) (profit_margin_percent : ℝ) (final_discount_percent : ℝ) (marked_price : ℝ) :
  initial_price = 40 → 
  discount_percent = 0.25 → 
  profit_margin_percent = 0.50 → 
  final_discount_percent = 0.10 → 
  marked_price = 50 := by
  sorry

end marked_price_l644_644547


namespace is_odd_function_l644_644609

def f (x : ℝ) : ℝ := x^3 - x

theorem is_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end is_odd_function_l644_644609


namespace minimum_value_of_a_l644_644679

theorem minimum_value_of_a (x : ℝ) (a : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ 20) (ha : 0 < a) (h : (20 - x) / 4 + a / 2 * Real.sqrt x ≥ 5) : 
  a ≥ Real.sqrt 5 := 
sorry

end minimum_value_of_a_l644_644679


namespace find_f_9_l644_644178

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

variables (f : ℝ → ℝ)
variable (h_odd : odd_function f)
variable (h_period : ∀ x, f (x - 2) = f (x + 2))
variable (h_def : ∀ x, x ∈ set.Icc (-2 : ℝ) (0 : ℝ) → f x = 3^x - 1)

theorem find_f_9 : f 9 = 2 / 3 :=
by
  sorry

end find_f_9_l644_644178


namespace zero_of_function_l644_644113

def f (x : ℝ) : ℝ := log x / log 2 - 2

theorem zero_of_function : f 4 = 0 :=
by
  -- proof here
  sorry

end zero_of_function_l644_644113


namespace prove_correct_option_C_l644_644501

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l644_644501


namespace minimum_dwarfs_l644_644050

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l644_644050


namespace length_of_platform_l644_644167

theorem length_of_platform (T : ℝ) (t_p : ℝ) (t_pl : ℝ) (L : ℝ) (V : ℝ) 
  (h1 : T = 300)
  (h2 : t_p = 18)
  (h3 : t_pl = 54)
  (h4 : V = T / t_p)
  (h5 : V = (T + L) / t_pl) : 
  L = 600 := 
by 
  -- Assume speed is consistent
  rw [h1, h2] at h4 
  have V_val : V = 300 / 18 := h4
  
  -- Calculate speed of train
  have speed_val : V = 16.6667 := calc
    V = 300 / 18 : by rw V_val
    ... = 16.6667 : by norm_num
  
  -- Use speed to calculate length of platform
  rw [h1, h3, speed_val] at h5 
  have length_calc : 16.6667 = (300 + L) / 54 := h5
    
  -- Solve for L
  have L_val : 16.6667 * 54 = 300 + L := by rw length_calc; norm_num
  have L_eq : L = 600 := by linarith
  
  exact L_eq

#check length_of_platform

end length_of_platform_l644_644167


namespace no_integer_coeff_poly_exists_l644_644238

theorem no_integer_coeff_poly_exists (P : ℤ[X])
  (h₁ : P.eval 7 = 11)
  (h₂ : P.eval 11 = 13) :
  ¬ ∃ (P : ℤ[X]), P.eval 7 = 11 ∧ P.eval 11 = 13 :=
by
  sorry

end no_integer_coeff_poly_exists_l644_644238


namespace new_ratio_petrol_kerosene_l644_644448

/-- 
If the initial ratio of petrol to kerosene in a container is 3:2, 
after removing 10 liters of the mixture and replacing it with 10 liters of kerosene, 
the new ratio of petrol to kerosene is 2:3. 
/-- 
theorem new_ratio_petrol_kerosene 
    (initial_P : ℝ) (initial_K : ℝ) (total : ℝ)
    (h1 : initial_P / initial_K = 3 / 2)
    (h2 : initial_P + initial_K = 29.999999999999996)
    (removed_mixture : ℝ := 10)
    (added_kerosene : ℝ := 10) : 
    (initial_P - (3 / 5) * removed_mixture) / (initial_K - (2 / 5) * removed_mixture + added_kerosene) = 2 / 3 :=
by 
  sorry

end new_ratio_petrol_kerosene_l644_644448


namespace minimum_dwarfs_l644_644047

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l644_644047


namespace part_a_cube_edge_length_part_b_cube_edge_length_l644_644584

-- Part (a)
theorem part_a_cube_edge_length (small_cubes : ℕ) (edge_length_original : ℤ) :
  small_cubes = 512 → edge_length_original^3 = small_cubes → edge_length_original = 8 :=
by
  intros h1 h2
  sorry

-- Part (b)
theorem part_b_cube_edge_length (small_cubes_internal : ℕ) (edge_length_inner : ℤ) (edge_length_original : ℤ) :
  small_cubes_internal = 512 →
  edge_length_inner^3 = small_cubes_internal → 
  edge_length_original = edge_length_inner + 2 →
  edge_length_original = 10 :=
by
  intros h1 h2 h3
  sorry

end part_a_cube_edge_length_part_b_cube_edge_length_l644_644584


namespace problem_lemma_l644_644855

def sqrt4 (x : ℕ) : ℝ := x ^ (1 / 4 : ℝ)
def sqrt3 (x : ℕ) : ℝ := x ^ (1 / 3 : ℝ)
def sqrt2 (x : ℕ) : ℝ := x ^ (1 / 2 : ℝ)

theorem problem_lemma : sqrt4 16 * sqrt3 8 * sqrt2 4 = 8 := 
by
  -- The proof steps from the solution would go here
  sorry

end problem_lemma_l644_644855


namespace rubber_band_area_correct_l644_644026

theorem rubber_band_area_correct :
  let rect := ((0, 0), (5, 4)) in
  let points := [(0,1), (2,4), (5,3), (3,0)] in
  area_of_polygon_in_rectangle rect points = 1.5 :=
by
  sorry

end rubber_band_area_correct_l644_644026


namespace range_of_m_l644_644337

theorem range_of_m (m : ℝ) (x y : ℝ) :
  (∃ x y, (\frac{x^2}{m+2} -\frac{y^2}{m+1} = 1)) → (m ∈ Ioo (-2 : ℝ) (-3/2 : ℝ) ∪ Ioo (-3/2 : ℝ) (-1 : ℝ)) := sorry

end range_of_m_l644_644337


namespace magnitude_of_linear_combination_l644_644711

open Real

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hab_perp : inner a b = 0)

theorem magnitude_of_linear_combination :
  ‖2 • a - b‖ = 2 * sqrt 2 :=
by
  sorry

end magnitude_of_linear_combination_l644_644711


namespace weekly_sales_correct_l644_644788

open Real

noncomputable def cost_left_handed_mouse (cost_normal_mouse : ℝ) : ℝ :=
  cost_normal_mouse * 1.3

noncomputable def cost_left_handed_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  cost_normal_keyboard * 1.2

noncomputable def cost_left_handed_scissors (cost_normal_scissors : ℝ) : ℝ :=
  cost_normal_scissors * 1.5

noncomputable def daily_sales_mouse (cost_normal_mouse : ℝ) : ℝ :=
  25 * cost_left_handed_mouse cost_normal_mouse

noncomputable def daily_sales_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  10 * cost_left_handed_keyboard cost_normal_keyboard

noncomputable def daily_sales_scissors (cost_normal_scissors : ℝ) : ℝ :=
  15 * cost_left_handed_scissors cost_normal_scissors

noncomputable def bundle_price (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  (cost_left_handed_mouse cost_normal_mouse + cost_left_handed_keyboard cost_normal_keyboard + cost_left_handed_scissors cost_normal_scissors) * 0.9

noncomputable def daily_sales_bundle (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  5 * bundle_price cost_normal_mouse cost_normal_keyboard cost_normal_scissors

noncomputable def weekly_sales (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  3 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors) +
  1.5 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors)

theorem weekly_sales_correct :
  weekly_sales 120 80 30 = 29922.25 := sorry

end weekly_sales_correct_l644_644788


namespace remainder_55_57_mod_7_l644_644864

theorem remainder_55_57_mod_7 :
  (55 * 57) % 7 = 6 :=
by
  have h1 : 55 % 7 = 6 := rfl
  have h2 : 57 % 7 = 1 := rfl
  rw [←h1, ←h2]
  -- This line expounds the equivalence in modular arithmetic
  exact Nat.mul_mod h1 h2

end remainder_55_57_mod_7_l644_644864


namespace min_pieces_chessboard_l644_644474

theorem min_pieces_chessboard (n : ℕ) : 
  let min_pieces := if n % 2 = 0 then 2 * n else 2 * n + 1 in
  ∀ (board :Π (i j : ℕ), 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n → Prop),
    (∀ i, board i i ∧ board i (n - i - 1)) ↔ min_pieces = (if n % 2 = 0 then 2 * n else 2 * n + 1) := 
by
  sorry

end min_pieces_chessboard_l644_644474


namespace green_tea_leaves_required_l644_644941

theorem green_tea_leaves_required (S_m : ℕ) (L_s : ℕ) (halved : ℕ) : 
  S_m = 3 → 
  L_s = 2 → 
  halved = 2 → 
  2 * (S_m * L_s) = 12 :=
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  simp 
  norm_num 
  sorry

end green_tea_leaves_required_l644_644941


namespace totalCandy_l644_644639

-- Define the number of pieces of candy each person had
def TaquonCandy : ℕ := 171
def MackCandy : ℕ := 171
def JafariCandy : ℕ := 76

-- Prove that the total number of pieces of candy they had together is 418
theorem totalCandy : TaquonCandy + MackCandy + JafariCandy = 418 := by
  sorry

end totalCandy_l644_644639


namespace mandy_yoga_time_l644_644359

def gym_to_bicycle_ratio := 2 / 3
def yoga_to_exercise_ratio := 2 / 3
def total_time := 100
def bicycle_time := 18

-- We need to find yoga_time such that the total time spent on these activities matches the given constraints.
theorem mandy_yoga_time :
  let gym_time := (2 / 3) * bicycle_time in
  let total_exercise_time := gym_time + bicycle_time in
  let yoga_time := (2 / 3) * total_exercise_time in
  yoga_time = 20 :=
by
  sorry

end mandy_yoga_time_l644_644359


namespace correct_operation_l644_644511

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l644_644511


namespace abs_difference_of_squares_l644_644130

theorem abs_difference_of_squares : 
  let a := 105 
  let b := 103
  abs (a^2 - b^2) = 416 := 
by 
  let a := 105
  let b := 103
  sorry

end abs_difference_of_squares_l644_644130


namespace constant_term_expansion_l644_644251

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem constant_term_expansion (x : ℝ) :
  let term := (sqrt x - 1 / (x ^ (1 / 3))) ^ 10
  C(10, 6) * (-1)^6 = 210 :=
by
  sorry

end constant_term_expansion_l644_644251


namespace evaluate_expression_l644_644241

theorem evaluate_expression : 8^(Real.log 8 (5 + 3)) = 8 := by
  sorry

end evaluate_expression_l644_644241


namespace marble_arrangement_remainder_is_correct_l644_644384

noncomputable def marble_arrangement_remainder : ℕ :=
  let N := 924 in -- Calculated number of ways Lucas can achieve such arrangements
  N % 1000       -- Find the remainder when N is divided by 1000

theorem marble_arrangement_remainder_is_correct : marble_arrangement_remainder = 924 :=
by
  unfold marble_arrangement_remainder
  sorry

end marble_arrangement_remainder_is_correct_l644_644384


namespace find_length_QJ_in_triangle_PQR_l644_644755

theorem find_length_QJ_in_triangle_PQR
  (PQ PR QR : ℝ)
  (h1 : PQ = 12) (h2 : PR = 16) (h3 : QR = 20) :
  let s := (PQ + PR + QR) / 2,
      K := √(s * (s - PQ) * (s - PR) * (s - QR)),
      r := K / s,
      y := QR + PR - PQ,
      QJ := √(y^2 + r^2)
  in QJ = 4 * √6 :=
by
  sorry

end find_length_QJ_in_triangle_PQR_l644_644755


namespace min_dwarfs_for_no_empty_neighbor_l644_644056

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l644_644056


namespace unique_solution_for_k_l644_644966

noncomputable def k := -1/12

theorem unique_solution_for_k :
  ∃ x : ℝ, (x - 3) / (k * x + 2) = x ∧ ∀ y : ℝ, (y - 3) / (k * y + 2) = y → y = x :=
by
  let discriminant (a b c : ℝ) := b^2 - 4 * a * c
  have h1 : discriminant k 1 (-3) = 0 := by
    simp [discriminant, k]
    norm_num
  have h2 : ∀ x, k * x + 2 ≠ 0 := by
    intro x hx
    simp [k] at hx
    field_simp at hx
    linarith
  sorry

end unique_solution_for_k_l644_644966


namespace portion_spent_in_first_store_l644_644385

/-
Mark went to a store where he spent some of his money "x", and then spent an additional $14. 
He then went to another store where he spent one-third of his starting money $180, and then spent $16 more. 
When he had no money left, he had $180 when he entered the first store. 
Given these conditions, prove that Mark spent approximately 57.77% of his money in the first store.
-/
theorem portion_spent_in_first_store (x : ℝ) : 
  let total_money := 180
  let spent_in_store1 := x + 14
  let remaining_after_store1 := total_money - spent_in_store1
  let spent_in_store2 := (1 / 3) * total_money + 16
  remaining_after_store1 - spent_in_store2 = 0 → (spent_in_store1 / total_money) ≈ 0.5777 :=
by
  sorry

end portion_spent_in_first_store_l644_644385


namespace find_natural_numbers_l644_644977

-- Define the conditions
def is_not_square (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ≠ n

def divides_floor_sqrt_cubed (n : ℕ) : Prop :=
  let k := nat.floor (real.sqrt n)
  in k ^ 3 ∣ n ^ 2

-- Define the set of solutions
def solutions : set ℕ :=
{2, 3, 8, 24}

-- The final theorem
theorem find_natural_numbers (n : ℕ) :
  is_not_square n ∧ divides_floor_sqrt_cubed n ↔ n ∈ solutions :=
sorry

end find_natural_numbers_l644_644977


namespace correct_operation_l644_644509

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l644_644509


namespace part_a_part_b_l644_644598

-- Part (a)
theorem part_a (n : ℕ) (hn : 0 < n) :
  ∃ a b : ℕ, 0 < b ∧ 
             a / b = sqrt n ∧ 
             b ≤ sqrt n + 1 ∧
             sqrt n ≤ a / b ∧ a / b ≤ sqrt (n + 1) := sorry

-- Part (b)
theorem part_b :
  ∃ᶠ (n : ℕ) in at_top, 
  ¬ (∃ a b : ℕ, 0 < b ∧ 
               a / b = sqrt n ∧ 
               b ≤ sqrt n ∧ 
               sqrt n ≤ a / b ∧ 
               a / b ≤ sqrt (n + 1)) := sorry

end part_a_part_b_l644_644598


namespace polynomial_expansion_l644_644987

variable (x : ℝ)

theorem polynomial_expansion : 
  (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 :=
by
  sorry

end polynomial_expansion_l644_644987


namespace polynomial_remainder_l644_644633

theorem polynomial_remainder (x : ℕ) : 
  let q := (11175 * x^2 + 22200 * x + 11026) in
  (x^150) % (x + 1)^3 = q := 
by
  -- skipping the proof as instructed
  sorry

end polynomial_remainder_l644_644633


namespace oranges_in_buckets_l644_644840

theorem oranges_in_buckets :
  ∀ (x : ℕ),
  (22 + x + (x - 11) = 89) →
  (x - 22 = 17) :=
by
  intro x h
  sorry

end oranges_in_buckets_l644_644840


namespace all_lines_XY_pass_through_C_l644_644899

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def is_perpendicular (p₁ p₂ : Point) (c : Circle) : Prop :=
  p₁.x * p₂.x + p₁.y * p₂.y = 0

def projection (p : Point) (a b : Point) : Point := 
  let t := ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / ((b.x - a.x)^2 + (b.y - a.y)^2)
  {x := a.x + t * (b.x - a.x), y := a.y + t * (b.y - a.y)}

def tangent_intersection (a b : Point) (c : Circle) : Point := 
  -- The formula for the intersection of tangents at points A and B is omitted for brevity.
  sorry

def fixed_point (p : Point) (r : ℝ) : Point :=
  let scale := r^2 / (p.x^2 + p.y^2 + r^2)
  {x := 2 * p.x * scale, y := 2 * p.y * scale}

theorem all_lines_XY_pass_through_C (O : Point) (r : ℝ) (p : Point)
  (hP : (p.x^2 + p.y^2) < r^2)
  (A : Point) (B : Point)
  (hA : A.x^2 + A.y^2 = r^2)
  (hB : B.x^2 + B.y^2 = r^2)
  (hperp : is_perpendicular (Point.mk (A.x - p.x) (A.y - p.y)) 
                            (Point.mk (B.x - p.x) (B.y - p.y)) (Circle.mk O r)) :
  let X := projection p A B,
      Y := tangent_intersection A B (Circle.mk O r),
      C := fixed_point p r in
  ∃ C, ∀ (A : Point) (B : Point),
    (is_perpendicular (Point.mk (A.x - p.x) (A.y - p.y))
                      (Point.mk (B.x - p.x) (B.y - p.y)) (Circle.mk O r)) →
    line_through_XY X Y C :=
begin
  sorry
end

end all_lines_XY_pass_through_C_l644_644899


namespace values_of_y_satisfy_quadratic_l644_644611

theorem values_of_y_satisfy_quadratic :
  (∃ (x y : ℝ), 3 * x^2 + 4 * x + 7 * y + 2 = 0 ∧ 3 * x + 2 * y + 4 = 0) →
  (∃ (y : ℝ), 4 * y^2 + 29 * y + 6 = 0) :=
by sorry

end values_of_y_satisfy_quadratic_l644_644611


namespace isosceles_triangle_vertex_angle_l644_644747

theorem isosceles_triangle_vertex_angle (A B C : Type) [angle A] [angle B] [angle C] 
  (isosceles_triangle : ∀ {ABC : triangle}, base_angle = A ∧ base_angle = B ∧ vertex_angle = C)
  (base_angle_is_80 : ∀ {A B}, angle A = 80 ∧ angle B = 80 ∧ base_angle A B)
  : ∃ {C : angle}, angle C = 20 := by
sorry

end isosceles_triangle_vertex_angle_l644_644747


namespace num_possible_house_numbers_l644_644984

-- Define the conditions in Lean
def is_two_digit_prime (n : ℕ) : Prop :=
  nat.prime n ∧ 10 ≤ n ∧ n < 60 ∧ n % 10 ≠ 0

def valid_house_numbers : ℕ :=
  {p : ℕ // is_two_digit_prime p}.card * ({p : ℕ // is_two_digit_prime p}.card - 1)

-- The main statement that checks the total number of distinct house numbers ABCD
theorem num_possible_house_numbers : valid_house_numbers = 156 := by
  sorry

end num_possible_house_numbers_l644_644984


namespace sum_of_sequences_eq_720_l644_644834

theorem sum_of_sequences_eq_720 :
  ∃ (a b : ℕ → ℕ) (S : ℕ),
  (∀ n, a n = a 1 + (n - 1) * 5) ∧
  (∀ n, b n = b 1 + (n - 1) * 7) ∧
  a 1 = 5 ∧
  b 1 = 7 ∧
  a 20 + b 20 = 60 ∧
  S = ∑ i in finset.range 20, (a i + b i) ∧
  S = 720 :=
begin
  let a := λ n, 5 + (n - 1) * (a 2 - a 1),
  let b := λ n, 7 + (n - 1) * (b 2 - b 1),
  use [a, b, 720],
  split,
  { intros n,
    unfold a,
    sorry },
  split,
  { intros n,
    unfold b,
    sorry },
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { unfold a b,
    sorry },
  split,
  { unfold a b,
    sorry },
  { exact rfl }
end

end sum_of_sequences_eq_720_l644_644834


namespace k_polygonal_intersects_fermat_l644_644659

theorem k_polygonal_intersects_fermat (k : ℕ) (n m : ℕ) (h1: k > 2) 
  (h2 : ∃ n m, (k - 2) * n * (n - 1) / 2 + n = 2 ^ (2 ^ m) + 1) : 
  k = 3 ∨ k = 5 :=
  sorry

end k_polygonal_intersects_fermat_l644_644659


namespace non_zero_a_l644_644708

noncomputable def M (a : ℝ) : set ℝ := {0, a}
noncomputable def N : set ℤ := {x : ℤ | x^2 - 2 * x - 3 < 0}

-- Auxiliary lemma: Prove the actual elements of the set N
lemma N_elements : N = {0, 1, 2} :=
by {
  -- Proof omitted for brevity
  sorry
}

theorem non_zero_a (a : ℝ) (h : (0:int) ∈ N ∨ a ∈ N) : a ≠ 0 :=
by {
  -- Proof omitted for brevity
  sorry
}

end non_zero_a_l644_644708


namespace theresa_crayons_initial_l644_644463

-- Let T_initial represent the number of crayons Theresa had initially.
-- Conditions imply that Theresa having 19 crayons initially is true, considering Janice's actions don't affect Theresa.

theorem theresa_crayons_initial (Janice_crayons : ℕ) (Nancy_crayons : ℕ) (T_initial : ℕ) :
  Janice_crayons = 12 → 
  (Janice_crayons + Nancy_crayons = 25) → 
  T_initial = 19 :=
by {
  -- Use the information provided to infer that T_initial is indeed 19.
  intros hJanice hShare,
  (assume hTheresa : T_initial = 19),
  -- Given conditions:
  have hJanice' : Janice_crayons = 12 := hJanice,
  have hShare' : Janice_crayons + Nancy_crayons = 25 := hShare,
  have hTheresa' : T_initial = 19 := hTheresa,
  -- Conclude the proof
  exact hTheresa',
}

end theresa_crayons_initial_l644_644463


namespace find_n_l644_644340

variables (a b n : ℝ)

-- Conditions
def condition1 : Prop := (150 * a * 1 + 150 * b * 1 = 450 ∧ 150 * a * 1 + 150 * b * 1 = 300)
def condition2 : Prop := (90 * a * 2 + 90 * b * 2 = 360 ∧ 90 * a * 2 + 90 * b * 2 = 450)
def condition3 : Prop := (300 * a * 4 + 75 * b * 4 = n)

-- Theorem to prove
theorem find_n (h1 : condition1) (h2 : condition2) (h3 : condition3) : n = 150 :=
by sorry

end find_n_l644_644340


namespace minimum_alliances_per_country_minimum_number_of_alliances_l644_644024

section MilitaryAlliances
variables (Country : Type) (Alliance : Type)
variables (membership : Country → Alliance → Prop)
variables [fintype Country] [fintype Alliance]

-- Assume there are 100 countries
axiom country_count : fintype.card Country = 100

-- Assume each alliance contains no more than 50 countries
axiom alliance_size : ∀ a : Alliance, (finset.card {c : Country | membership c a}) ≤ 50

-- Assume every two countries must share at least one alliance
axiom common_alliance : ∀ (c1 c2 : Country), c1 ≠ c2 → ∃ a : Alliance, membership c1 a ∧ membership c2 a

-- Part (a): Prove that each country must participate in at least three alliances
theorem minimum_alliances_per_country (c : Country) : 
  (finset.card {a : Alliance | membership c a}) ≥ 3 := 
sorry

-- Part (b): Prove that the minimum number of alliances required is 6
theorem minimum_number_of_alliances : 
  fintype.card Alliance ≥ 6 := 
sorry

end MilitaryAlliances

end minimum_alliances_per_country_minimum_number_of_alliances_l644_644024


namespace total_cost_of_lollipops_l644_644802

/-- Given Sarah bought 12 lollipops and shared one-quarter of them, 
    and Julie reimbursed Sarah 75 cents for the shared lollipops,
    Prove that the total cost of the lollipops in dollars is $3. --/
theorem total_cost_of_lollipops 
(Sarah_lollipops : ℕ) 
(shared_fraction : ℚ) 
(Julie_paid : ℚ) 
(total_lollipops_cost : ℚ)
(h1 : Sarah_lollipops = 12) 
(h2 : shared_fraction = 1/4) 
(h3 : Julie_paid = 75 / 100) 
(h4 : total_lollipops_cost = 
        ((Julie_paid / (Sarah_lollipops * shared_fraction)) * Sarah_lollipops / 100)) :
total_lollipops_cost = 3 := 
sorry

end total_cost_of_lollipops_l644_644802


namespace direction_opposite_l644_644318

noncomputable def a (m : ℝ) : ℝ × ℝ := (m, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (1, m + 1)

theorem direction_opposite (m : ℝ) : (a m).fst * (b m).snd = (a m).snd * (b m).fst → (a m).fst / abs((a m).fst) * (b m).fst / abs((b m).fst) < 0 → m = -2  :=
by
  intro h
  intro habs
  sorry

end direction_opposite_l644_644318


namespace selection_probabilities_l644_644464

-- Define the probabilities of selection for Ram, Ravi, and Rani
def prob_ram : ℚ := 5 / 7
def prob_ravi : ℚ := 1 / 5
def prob_rani : ℚ := 3 / 4

-- State the theorem that combines these probabilities
theorem selection_probabilities : prob_ram * prob_ravi * prob_rani = 3 / 28 :=
by
  sorry


end selection_probabilities_l644_644464


namespace k_faster_than_l_by_percentage_l644_644149

def percentage_faster_than (v_k v_l : ℕ) : ℕ := ((v_k - v_l) * 100) / v_l

theorem k_faster_than_l_by_percentage :
  ∀ (v_k v_l l_start_time k_start_time meet_time initial_distance l_distance k_distance : ℕ),
  l_start_time = 9 →
  k_start_time = 10 →
  meet_time = 12 →
  v_l = 50 →
  initial_distance = 300 →
  l_distance = (meet_time - l_start_time) * v_l →
  k_distance = initial_distance - l_distance →
  v_k = k_distance / (meet_time - k_start_time) →
  percentage_faster_than v_k v_l = 50 :=
by
  intros v_k v_l l_start_time k_start_time meet_time initial_distance l_distance k_distance
  assume h_l_start : l_start_time = 9
  assume h_k_start : k_start_time = 10
  assume h_meet : meet_time = 12
  assume h_vl : v_l = 50
  assume h_initial_dist : initial_distance = 300
  assume h_l_dist : l_distance = (meet_time - l_start_time) * v_l
  assume h_k_dist : k_distance = initial_distance - l_distance
  assume h_vk : v_k = k_distance / (meet_time - k_start_time)
  sorry

end k_faster_than_l_by_percentage_l644_644149


namespace factorize_difference_of_squares_l644_644991

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4*x^2 = (3 - 2*x) * (3 + 2*x) :=
by
  sorry

end factorize_difference_of_squares_l644_644991


namespace triangle_properties_l644_644357

open Real

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (b = c)

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def area (a b c : ℝ) (A : ℝ) : ℝ :=
  1/2 * b * c * sin A

theorem triangle_properties 
  (A B C a b c : ℝ) 
  (h1 : sin B * sin C = 1/4) 
  (h2 : tan B * tan C = 1/3) 
  (h3 : a = 4 * sqrt 3) 
  (h4 : A + B + C = π) 
  (isosceles : is_isosceles_triangle A B C a b c) :
  is_isosceles_triangle A B C a b c ∧ 
  perimeter a b c = 8 + 4 * sqrt 3 ∧ 
  area a b c A = 4 * sqrt 3 :=
sorry

end triangle_properties_l644_644357


namespace parallelogram_angle_bisectors_divide_l644_644450

theorem parallelogram_angle_bisectors_divide :
  ∀ A B C D : Type, ∀ (AB BC CD DA : ℝ), 
  (parallelogram A B C D) → 
  (AB = 8) → (BC = 3) → (CD = 8) → (DA = 3) → 
  ∃ K M : Type, 
    angle_bisector B K D A → 
    angle_bisector C M D A → 
    (segment_length A K = 3 ∧
     segment_length K M = 2 ∧
     segment_length M D = 3) := 
begin 
  sorry 
end

end parallelogram_angle_bisectors_divide_l644_644450


namespace w_squared_roots_l644_644722

theorem w_squared_roots :
  ∀ w : ℝ, 
  (w + 15)^2 = (4 * w + 5) * (3 * w + 9) →
  (w^2 ≈ 40.4967) ∨ (w^2 ≈ 103.6694) :=
by
  intro w
  intro h
  sorry

end w_squared_roots_l644_644722


namespace non_congruent_triangles_perimeter_18_l644_644323

theorem non_congruent_triangles_perimeter_18 :
  ∃ (triangles : Finset (Finset ℕ)), triangles.card = 11 ∧
  (∀ t ∈ triangles, t.card = 3 ∧ (∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 18 ∧ a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end non_congruent_triangles_perimeter_18_l644_644323


namespace ranking_possible_sequences_l644_644339

theorem ranking_possible_sequences 
  (students : Fin 5)
  (A B : students)
  (first_place_ne_A_and_B : ∀ (i : students), (i ≠ A) → (i ≠ B) → (i = first))
  (B_not_last : ∀ (j : students), j ≠ B → (j = last)) :
  54 = number_of_different_possible_sequences students A B :=
sorry

end ranking_possible_sequences_l644_644339


namespace variance_of_scores_l644_644193

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (λ x => (x - m) ^ 2)).sum / data.length

theorem variance_of_scores :
  let data := [9.7, 9.9, 10.1, 10.2, 10.1]
  variance data = 0.032 :=
by
  let data := [9.7, 9.9, 10.1, 10.2, 10.1]
  -- Calculate the mean
  let m := mean data
  have h1 : m = 10 := by norm_num [mean, data]
  -- Calculate the variance
  have h2 : variance data = 0.032 := by norm_num [variance, data, h1]
  exact h2

end variance_of_scores_l644_644193


namespace rectangle_diagonal_PO2_sum_l644_644794

-- Main statement of the problem translated into Lean 4 code
theorem rectangle_diagonal_PO2_sum (AB BC : ℝ) (AP CP : ℝ)
    (O1 O2 : Point) (P: Point) (AP_gt_CP : AP > CP)
    (rect_len_cond : AB = 15 ∧ BC = 8)
    (O1P_eq_angle_cond : ∠O1PO2 = 90°) :
    ∃ (a b : ℝ), AP = Real.sqrt a + Real.sqrt b ∧ a + b = 41 :=
by
  sorry

end rectangle_diagonal_PO2_sum_l644_644794


namespace grunters_win_all_five_l644_644812

theorem grunters_win_all_five (p : ℚ) (games : ℕ) (win_prob : ℚ) :
  games = 5 ∧ win_prob = 3 / 5 → 
  p = (win_prob) ^ games ∧ p = 243 / 3125 := 
by
  intros h
  cases h
  sorry

end grunters_win_all_five_l644_644812


namespace prove_correct_option_C_l644_644502

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l644_644502


namespace find_b_l644_644829

theorem find_b (b p : ℚ) :
  (∀ x : ℚ, (2 * x^3 + b * x + 7 = (x^2 + p * x + 1) * (2 * x + 7))) →
  b = -45 / 2 :=
sorry

end find_b_l644_644829


namespace solve_eq_l644_644642

def Min (a b : ℝ) := if a < b then a else b

theorem solve_eq (x : ℝ) (h_distinct : x ≠ 0) :
  Min (1 / x) (2 / x) = (3 / x) - 1 ↔ x = 2 :=
begin
  sorry
end

end solve_eq_l644_644642


namespace work_done_by_student_l644_644524

theorem work_done_by_student
  (M : ℝ)  -- mass of the student
  (m : ℝ)  -- mass of the stone
  (h : ℝ)  -- height from which the stone is thrown
  (L : ℝ)  -- distance on the ice where the stone lands
  (g : ℝ)  -- acceleration due to gravity
  (t : ℝ := Real.sqrt (2 * h / g))  -- time it takes for the stone to hit the ice derived from free fall equation
  (Vk : ℝ := L / t)  -- initial speed of the stone derived from horizontal motion
  (Vu : ℝ := m / M * Vk)  -- initial speed of the student derived from conservation of momentum
  : (1/2 * m * Vk^2 + (1/2) * M * Vu^2) = 126.74 :=
by
  sorry

end work_done_by_student_l644_644524


namespace angle_between_vectors_l644_644317

def vector (α : Type*) := (α × α)

noncomputable def magnitude {α : Type*} [RealField α] (v : vector α) : α :=
  (v.1 ^ 2 + v.2 ^ 2).sqrt

noncomputable def dot_product {α : Type*} [Ring α] (v w : vector α) : α :=
  v.1 * w.1 + v.2 * w.2

noncomputable def angle_between {α : Type*} [RealField α] (v w : vector α) : α :=
  let cos_theta := dot_product v w / (magnitude v * magnitude w) in
  Real.acos cos_theta

theorem angle_between_vectors : angle_between (1, 1) ((-1, 2).1 + 1, (-1, 2).2 + 1) = Real.pi / 4 :=
by sorry

end angle_between_vectors_l644_644317


namespace percentage_of_candidates_selected_in_State_A_is_6_l644_644740

-- Definitions based on conditions
def candidates_appeared : ℕ := 8400
def candidates_selected_B : ℕ := (7 * candidates_appeared) / 100 -- 7% of 8400
def extra_candidates_selected : ℕ := 84
def candidates_selected_A : ℕ := candidates_selected_B - extra_candidates_selected

-- Definition based on the goal proof
def percentage_selected_A : ℕ := (candidates_selected_A * 100) / candidates_appeared

-- The theorem we need to prove
theorem percentage_of_candidates_selected_in_State_A_is_6 :
  percentage_selected_A = 6 :=
by
  sorry

end percentage_of_candidates_selected_in_State_A_is_6_l644_644740


namespace common_difference_l644_644758

-- Definitions
variable (a₁ d : ℝ) -- First term and common difference of the arithmetic sequence

-- Conditions
def mean_nine_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 8 * d)) = 10

def mean_ten_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 9 * d)) = 13

-- Theorem to prove the common difference is 6
theorem common_difference (a₁ d : ℝ) :
  mean_nine_terms a₁ d → 
  mean_ten_terms a₁ d → 
  d = 6 := by
  intros
  sorry

end common_difference_l644_644758


namespace max_consecutive_sum_lt_500_l644_644863

theorem max_consecutive_sum_lt_500 :
  ∃ n : ℕ, (∀ m : ℕ, 2 ≤ m → m ≤ n + 1 → ∑ k in finset.range (m + 1), k < 500) ∧
           (∀ m : ℕ, 2 ≤ m → m ≤ n + 2 → ∑ k in finset.range (m + 1), k ≥ 500) :=
begin
  sorry
end

end max_consecutive_sum_lt_500_l644_644863


namespace hugo_probability_l644_644742

noncomputable def P_hugo_first_roll_seven_given_win (P_Hugo_wins : ℚ) (P_first_roll_seven : ℚ)
  (P_all_others_roll_less_than_seven : ℚ) : ℚ :=
(P_first_roll_seven * P_all_others_roll_less_than_seven) / P_Hugo_wins

theorem hugo_probability :
  let P_Hugo_wins := (1 : ℚ) / 4
  let P_first_roll_seven := (1 : ℚ) / 8
  let P_all_others_roll_less_than_seven := (27 : ℚ) / 64
  P_hugo_first_roll_seven_given_win P_Hugo_wins P_first_roll_seven P_all_others_roll_less_than_seven = (27 : ℚ) / 128 :=
by
  sorry

end hugo_probability_l644_644742


namespace average_distance_per_day_l644_644081

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_distance_per_day :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 :=
by
  sorry

end average_distance_per_day_l644_644081


namespace largest_prime_factor_of_combined_expr_l644_644978

-- Define the given mathematical expressions
def expr1 : ℕ := 18^4
def expr2 : ℕ := 3 * 18^2
def expr3 : ℕ := 1
def expr4 : ℕ := 17^4

-- Define the combined expression
def combined_expr : ℕ := expr1 + expr2 + expr3 - expr4

-- Prove that the largest prime factor of the combined expression is 307
theorem largest_prime_factor_of_combined_expr : 
  ∃ p : ℕ, prime p ∧ p = 307 ∧ (∀ q : ℕ, prime q ∧ q ∣ combined_expr → q ≤ p) :=
begin
  sorry
end

end largest_prime_factor_of_combined_expr_l644_644978


namespace cos_angle_planes_l644_644773

noncomputable def cos_angle_between_planes 
  (normal1 normal2 : ℝ × ℝ × ℝ)
  (h1 : normal1 = (3, -4, 6))
  (h2 : normal2 = (9, 2, -3)) : ℝ :=
  let dot := normal1.1 * normal2.1 + normal1.2 * normal2.2 + normal1.3 * normal2.3 in
  let norm1 := Real.sqrt (normal1.1 ^ 2 + normal1.2 ^ 2 + normal1.3 ^ 2) in
  let norm2 := Real.sqrt (normal2.1 ^ 2 + normal2.2 ^ 2 + normal2.3 ^ 2) in
  dot / (norm1 * norm2)

theorem cos_angle_planes :
  ∃ θ : ℝ,
    cos_angle_between_planes (3, -4, 6) (9, 2, -3) = 1 / Real.sqrt 5734 :=
by
  use 1 / Real.sqrt 5734
  sorry

end cos_angle_planes_l644_644773


namespace solve_abs_inequality_l644_644809

theorem solve_abs_inequality (x : ℝ) (h : 1 < |x - 1| ∧ |x - 1| < 4) : (-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5) :=
by
  sorry

end solve_abs_inequality_l644_644809


namespace option_C_correct_l644_644487

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l644_644487


namespace max_min_expression_value_l644_644296

theorem max_min_expression_value (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) :
  let m := (if a > 0 then 1 else -1) + (if b > 0 then 1 else -1) + (if c > 0 then 1 else -1) - (if a * b * c > 0 then 1 else -1)
  let n := (if a < 0 then -1 else 1) + (if b < 0 then -1 else 1) + (if c < 0 then -1 else 1) - (if a * b * c < 0 then -1 else 1)
  (n ^ m) / (m * n) = -16 :=
by
  sorry

end max_min_expression_value_l644_644296


namespace explanation_for_dna_content_l644_644960

-- Definitions based on the problem conditions
def same_organism (a b : Type) : Prop := a = b

def dna_content_twice (a b : Type) [dna : Type] (dna_content : dna → ℕ) : Prop :=
  dna_content a = 2 * dna_content b

-- Hypotheses from solution interpretation
def normal_somatic_cell (a : Type) : Prop := sorry
def end_first_meiotic_division (b : Type) : Prop := sorry
def late_mitosis (a : Type) : Prop := sorry
def early_mitosis (b : Type) : Prop := sorry
def secondary_spermatocyte (b : Type) : Prop := sorry
def chromosomes_move_to_poles (a b : Type) : Prop := sorry

axiom cell_dna_content_equivalence_p1 (a b : Type) : normal_somatic_cell a ∧ end_first_meiotic_division b → dna_content_twice a b
axiom cell_dna_content_equivalence_p2 (a b : Type) : late_mitosis a ∧ early_mitosis b → dna_content_twice a b
axiom cell_dna_content_equivalence_p3 (a b : Type) : early_mitosis a ∧ end_first_meiotic_division b → dna_content_twice a b

-- The theorem we need to prove
theorem explanation_for_dna_content (a b : Type) [dna : Type] (dna_content : dna → ℕ)
    (H1 : same_organism a b)
    (H2 : dna_content_twice a b) :
  early_mitosis a ∧ end_first_meiotic_division b :=
by
  sorry

end explanation_for_dna_content_l644_644960


namespace rounding_6865_65_to_nearest_0_1_l644_644043

theorem rounding_6865_65_to_nearest_0_1 :
  Real.round (6865.65 * 10) / 10 = 6865.7 := 
by
  sorry

end rounding_6865_65_to_nearest_0_1_l644_644043


namespace triangle_angle_proof_l644_644753

theorem triangle_angle_proof (P Q R M N O S : Type) [IsTriangle P Q R] 
  (PM ⦉ P) (QN ⦉ Q) (RO ⦉ R) (S_is_orthocenter : orthocenter S P Q R)
  (angle_PQR : angle P Q R = 60)
  (angle_PRQ : angle P R Q = 15) : 
  angle Q S R = 75 := 
sorry

end triangle_angle_proof_l644_644753


namespace most_likely_units_digit_l644_644614

theorem most_likely_units_digit :
  ∃ m n : Fin 11, ∀ (M N : Fin 11), (∃ k : Nat, k * 11 + M + N = m + n) → 
    (m + n) % 10 = 0 :=
by
  sorry

end most_likely_units_digit_l644_644614


namespace problem_statement_l644_644713

-- Define vectors and function
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + Real.sqrt 3 / 2

-- Problem Statement
theorem problem_statement : 
  (∀ x, x = k * Real.pi / 2 + 5 * Real.pi / 12 → ∃ k : ℤ) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3 / 2) :=
sorry

end problem_statement_l644_644713


namespace minimum_dwarfs_l644_644044

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l644_644044


namespace sum_last_two_digits_modified_fibonacci_factorial_series_l644_644137

theorem sum_last_two_digits_modified_fibonacci_factorial_series :
  (1! % 100 + 2! % 100 + 3! % 100 + 4! % 100 + 7! % 100 + 11! % 100 + 18! % 100 + 29! % 100 + 47! % 100 + 76! % 100) = 73 :=
by {
  sorry
}

end sum_last_two_digits_modified_fibonacci_factorial_series_l644_644137


namespace constant_term_binomial_l644_644088

/-- The constant term in the expansion of the binomial (∛x + 1/(2x))^8 is 7. -/
theorem constant_term_binomial (x : ℝ) :
  let T_r (r : ℕ) := (Nat.choose 8 r) * (x ^ (8 - r) / (2 ^ r * x ^ r)) in
  (∃ r : ℕ, 0 ≤ r ∧ r ≤ 8 ∧ T_r r = 7) :=
begin
  sorry
end

end constant_term_binomial_l644_644088


namespace more_oranges_than_apples_l644_644766

-- Definitions based on conditions
def apples : ℕ := 14
def oranges : ℕ := 2 * 12  -- 2 dozen oranges

-- Statement to prove
theorem more_oranges_than_apples : oranges - apples = 10 := by
  sorry

end more_oranges_than_apples_l644_644766


namespace repeating_decimal_fraction_correct_l644_644243

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := 3.36363636 in  -- representing 3.\overline{36}
  x

theorem repeating_decimal_fraction_correct :
  repeating_decimal_to_fraction = 10 / 3 :=
sorry

end repeating_decimal_fraction_correct_l644_644243


namespace roots_polynomial_pq_sum_l644_644776

theorem roots_polynomial_pq_sum :
  ∀ p q : ℝ, 
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) * (x - 4) = x^4 - 10 * x^3 + p * x^2 - q * x + 24) 
  → p + q = 85 :=
by 
  sorry

end roots_polynomial_pq_sum_l644_644776


namespace find_mass_of_water_vapor_l644_644902

noncomputable def heat_balance_problem : Prop :=
  ∃ (m_s : ℝ), m_s * 536 + m_s * 80 = 
  (50 * 80 + 50 * 20 + 300 * 20 + 100 * 0.5 * 20)
  ∧ m_s = 19.48

theorem find_mass_of_water_vapor : heat_balance_problem := by
  sorry

end find_mass_of_water_vapor_l644_644902


namespace binom_2023_2_eq_l644_644963

theorem binom_2023_2_eq : Nat.choose 2023 2 = 2045323 := by
  sorry

end binom_2023_2_eq_l644_644963


namespace planes_contain_at_least_three_midpoints_l644_644719

-- Define the cube structure and edge midpoints
structure Cube where
  edges : Fin 12

def midpoints (c : Cube) : Set (Fin 12) := { e | true }

-- Define the total planes considering the constraints
noncomputable def planes : ℕ := 4 + 18 + 56

-- The proof goal
theorem planes_contain_at_least_three_midpoints :
  planes = 81 := by
  sorry

end planes_contain_at_least_three_midpoints_l644_644719


namespace correct_operation_l644_644488

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l644_644488


namespace sum_of_nine_two_digit_numbers_l644_644757

theorem sum_of_nine_two_digit_numbers (n : ℕ) (S : ℕ) 
  (h_n : n = 9) 
  (h_S : S = 240) 
  (h_two_digit : ∀ (x : ℕ), x ∈ (range 10 100)) 
  (h_contains_nine : ∀ (x : ℕ), '9' ∈ (x.digits 10)) :
  ¬ ∃ (numbers : list ℕ), (numbers.length = n ∧ numbers.sum = S ∧ ∀ (x : ℕ), x ∈ numbers → x ∈ (range 10 100) ∧ '9' ∈ (x.digits 10)) :=
  sorry

end sum_of_nine_two_digit_numbers_l644_644757


namespace max_cube_sum_l644_644008

theorem max_cube_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) : x^3 + y^3 + z^3 ≤ 27 :=
sorry

end max_cube_sum_l644_644008


namespace find_min_max_A_l644_644551

-- Define a 9-digit number B
def is_9_digit (B : ℕ) : Prop := B ≥ 100000000 ∧ B < 1000000000

-- Define a function that checks if a number is coprime with 24
def coprime_with_24 (B : ℕ) : Prop := Nat.gcd B 24 = 1

-- Define the transformation from B to A
def transform (B : ℕ) : ℕ := let b := B % 10 in b * 100000000 + (B / 10)

-- Lean 4 statement for the problem
theorem find_min_max_A :
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 999999998) ∧
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 166666667) :=
  by
    sorry -- Proof is omitted

end find_min_max_A_l644_644551


namespace bill_harry_combined_l644_644951

-- Definitions based on the given conditions
def sue_nuts := 48
def harry_nuts := 2 * sue_nuts
def bill_nuts := 6 * harry_nuts

-- The theorem we want to prove
theorem bill_harry_combined : bill_nuts + harry_nuts = 672 :=
by
  sorry

end bill_harry_combined_l644_644951


namespace sqrt_14_plus_2_range_l644_644985

theorem sqrt_14_plus_2_range :
  5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 :=
by
  sorry

end sqrt_14_plus_2_range_l644_644985


namespace nth_monomial_l644_644346

variable (a : ℝ)

def monomial_seq (n : ℕ) : ℝ :=
  (n + 1) * a ^ n

theorem nth_monomial (n : ℕ) : monomial_seq a n = (n + 1) * a ^ n :=
by
  sorry

end nth_monomial_l644_644346


namespace or_true_necessary_but_not_sufficient_l644_644531

variable (p q : Prop)

theorem or_true_necessary_but_not_sufficient (h : p ∨ q) : 
  (p ∨ q → p ∧ q) ↔ False ∧ (p ∧ q → p ∨ q) ↔ True :=
by
  sorry

end or_true_necessary_but_not_sufficient_l644_644531


namespace tv_cost_l644_644017

theorem tv_cost (savings original_savings furniture_spent : ℝ) (hs : original_savings = 1000) (hf : furniture_spent = (3/4) * original_savings) (remaining_spent : savings = original_savings - furniture_spent) : savings = 250 := 
by
  sorry

end tv_cost_l644_644017


namespace probability_of_odd_five_digit_number_l644_644425

theorem probability_of_odd_five_digit_number :
  let digits := {2, 3, 5, 7, 9}
  let n := 5
  let odd_digits := {3, 5, 7, 9}
  (|odd_digits|: ℝ) / (|digits|: ℝ) = 4 / 5 :=
by
  sorry

end probability_of_odd_five_digit_number_l644_644425


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644062

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644062


namespace unit_prices_min_selling_price_l644_644416

-- Problem 1: Unit price determination
theorem unit_prices (x y : ℕ) (hx : 3600 / x * 2 = 5400 / y) (hy : y = x - 5) : x = 20 ∧ y = 15 := 
by 
  sorry

-- Problem 2: Minimum selling price for 50% profit margin
theorem min_selling_price (a : ℕ) (hx : 3600 / 20 = 180) (hy : 180 * 2 = 360) (hz : 540 * a ≥ 13500) : a ≥ 25 := 
by 
  sorry

end unit_prices_min_selling_price_l644_644416


namespace sum_odd_numbers_to_2019_is_correct_l644_644022

-- Define the sequence sum
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Define the specific problem
theorem sum_odd_numbers_to_2019_is_correct : sum_first_n_odd 1010 = 1020100 :=
by
  -- Sorry placeholder for the proof
  sorry

end sum_odd_numbers_to_2019_is_correct_l644_644022


namespace correct_operation_l644_644493

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l644_644493


namespace valid_numbers_count_l644_644720

noncomputable def count_valid_numbers : ℕ :=
  (filter (λ n, n < 1000 ∧
                  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧
                  n = 100 * a + 10 * b + c ∧
                  (a + b + c) % 7 = 0 ∧
                  (a + b + c) % 3 = 0)
          (range 1000)).length

theorem valid_numbers_count : count_valid_numbers = 33 :=
by sorry

end valid_numbers_count_l644_644720


namespace blueprint_to_real_world_l644_644895

-- Define the scale factor and the length on the blueprint
def scale_factor := 25
def length_on_blueprint := 9.8

-- Define the correct length in the real world to be proved
def real_world_length := 245

-- Formulate the theorem to prove the equivalence
theorem blueprint_to_real_world :
  length_on_blueprint * scale_factor = real_world_length :=
by
  sorry

end blueprint_to_real_world_l644_644895


namespace polynomial_has_complex_zero_l644_644186

theorem polynomial_has_complex_zero (r s : ℤ) (p q d : ℤ) :
  ∃ α β : ℤ, let ζ1 := (3 : ℝ) / 2 + (19 : ℝ).sqrt / 2 * I,
                 ζ2 := (3 : ℝ) / 2 - (19 : ℝ).sqrt / 2 * I in
  Q(x) = (x - r) * (x - s) * ((x^3 : ℤ) + p*x^2 + q*x + d) ∧ 
  x^2 + α*x + β = (x - ζ1) * (x - ζ2) :=
sorry

end polynomial_has_complex_zero_l644_644186


namespace krishan_money_l644_644831

theorem krishan_money (R G K : ℕ) 
  (h_ratio1 : R * 17 = G * 7) 
  (h_ratio2 : G * 17 = K * 7) 
  (h_R : R = 735) : 
  K = 4335 := 
sorry

end krishan_money_l644_644831


namespace digit_nine_possibilities_l644_644544

theorem digit_nine_possibilities :
  {N : ℕ // N < 10 ∧ (864 * 10 + N) % 9 = 0}.card = 2 :=
by
  sorry

end digit_nine_possibilities_l644_644544


namespace unique_hexagon_angles_sides_identity_1_identity_2_l644_644099

noncomputable def lengths_angles_determined 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : Prop :=
  -- Assuming this is the expression we need to handle:
  ∀ (δ ε ζ : ℝ),
    δ = 180 - α ∧
    ε = 180 - β ∧
    ζ = 180 - γ →
  ∃ (angles_determined : Prop),
    angles_determined

theorem unique_hexagon_angles_sides 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : 
  lengths_angles_determined a b c d e f α β γ h₀ :=
sorry

theorem identity_1 
  (a b c d : ℝ) 
  (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  (1 / a + 1 / c = 1 / b + 1 / d) :=
sorry

theorem identity_2 
  (a b c d e f : ℝ) 
  (h₀ : true) : 
  ((a + f) * (b + d) * (c + e) = (a + e) * (b + f) * (c + d)) :=
sorry

end unique_hexagon_angles_sides_identity_1_identity_2_l644_644099


namespace sum_of_first_40_digits_l644_644139

noncomputable def first_40_digits_sum (x : ℚ) : ℕ :=
  let digits := (Real.toDigits 10 x).2.toList;
  (digits.take 40).map (λ d, Nat.ofDigitChars [d]).sum

theorem sum_of_first_40_digits : first_40_digits_sum (1 / 1234 : ℚ) = 218 := by
  sorry

end sum_of_first_40_digits_l644_644139


namespace geometric_sum_S6_l644_644354

open Real

-- Define a geometric sequence
noncomputable def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a * n else a * (1 - q ^ n) / (1 - q)

-- Given conditions
variables (a q : ℝ) (n : ℕ)
variable (S3 : ℝ)
variable (q : ℝ) (h_q : q = 2)
variable (h_S3 : S3 = 7)

theorem geometric_sum_S6 :
  sum_geometric a 2 6 = 63 :=
  by
    sorry

end geometric_sum_S6_l644_644354


namespace same_terminal_side_angle_l644_644145

theorem same_terminal_side_angle : 
  ∃ α ∈ (set.Icc 0 360), α = -120 + 360 * 1 :=
by
  use 240
  split
  { norm_num }
  { norm_num }

end same_terminal_side_angle_l644_644145


namespace correct_understanding_president_l644_644515

/-- Definitions according to the problem's conditions --/
def occupies_highest_position : Prop := false
def exercises_powers_accordingly : Prop := true
def important_state_organ : Prop := true
def decide_major_affairs_independently : Prop := false

/-- Lean Statement that the correct understandings about the President are ② and ③ --/
theorem correct_understanding_president :
  exercises_powers_accordingly ∧ important_state_organ ∧ 
  ¬occupies_highest_position ∧ ¬decide_major_affairs_independently :=
by 
  split;
  try {exact true.intro};
  try {exact false.elim sorry}

end correct_understanding_president_l644_644515


namespace sequence_binomial_sum_l644_644705

variable {n : ℕ}

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 2^(n-1) + 1

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| (n+1), (k+1) => binom n k + binom n (k+1)

theorem sequence_binomial_sum (n : ℕ) : 
  (∑ k in Finset.range (n + 1), a (k + 1) * binom n k) = 3^n + 2^n := 
by 
  sorry

end sequence_binomial_sum_l644_644705


namespace find_m_plus_n_l644_644190

def point := (ℝ × ℝ × ℝ)

def hexagon (center : point) (side_length : ℝ) : list point :=
  [(6, 0, 0), 
   (3, 3 * Real.sqrt 3, 0),
   (-3, 3 * Real.sqrt 3, 0),
   (-6, 0, 0),
   (-3, -3 * Real.sqrt 3, 0),
   (3, -3 * Real.sqrt 3, 0)]

def prism_base := hexagon (0, 0, 0) 6
def prism_top := hexagon (0, 0, 5) 6

def plane_p (A C' E : point) : (ℝ × ℝ × ℝ × ℝ) :=
  -- This would compute the plane equation based on points ℝ × ℝ × ℝ)
  let n := ((15 * Real.sqrt 3), -45, 54 * Real.sqrt 3) in
  let d := 30 in
  (5, -15, 18, d)

def area_of_portion (base : list point) (top : list point) (pl_eq : (ℝ × ℝ × ℝ × ℝ)) : ℝ :=
  6 * Real.sqrt 399

theorem find_m_plus_n : 
  let base := prism_base,
      top := prism_top,
      plane := plane_p (6, 0, 0) (-3, 3 * Real.sqrt 3, 5) (-3, -3 * Real.sqrt 3, 0)
  in
  let area := area_of_portion base top plane
  in area = 6 * Real.sqrt 399 → 6 + 399 = 405 :=
by
  intros base top plane area
  rw [plane, base, top, area]
  exact sorry

end find_m_plus_n_l644_644190


namespace probability_of_odd_number_l644_644427

-- Define the total number of digits
def digits : List ℕ := [2, 3, 5, 7, 9]

-- Define what it means for a number to be odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the problem of arranging the digits into a five-digit number that is odd
def num_of_arrangements (d : List ℕ) : ℕ := 5.factorial

def num_of_odd_arrangements (d : List ℕ) : ℕ := 4 * 4.factorial

-- Define the probability 
def probability_odd (d : List ℕ) : ℚ := num_of_odd_arrangements d / num_of_arrangements d

-- Statement: Prove that the probability is 4/5
theorem probability_of_odd_number : probability_odd digits = 4 / 5 := by
  sorry

end probability_of_odd_number_l644_644427


namespace find_y_l644_644909

open Real

theorem find_y : ∃ y : ℝ, (sqrt ((3 - (-5))^2 + (y - 4)^2) = 12) ∧ (y > 0) ∧ (y = 4 + 4 * sqrt 5) :=
by
  use 4 + 4 * sqrt 5
  -- The proof steps would go here.
  sorry

end find_y_l644_644909


namespace max_possible_value_of_y_l644_644076

theorem max_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 :=
sorry

end max_possible_value_of_y_l644_644076


namespace shane_photos_per_week_l644_644039

theorem shane_photos_per_week (jan_days feb_weeks photos_per_day total_photos : ℕ) :
  (jan_days = 31) →
  (feb_weeks = 4) →
  (photos_per_day = 2) →
  (total_photos = 146) →
  let photos_jan := photos_per_day * jan_days in
  let photos_feb := total_photos - photos_jan in
  let photos_per_week := photos_feb / feb_weeks in
  photos_per_week = 21 :=
by
  intros h1 h2 h3 h4
  let photos_jan := photos_per_day * jan_days
  let photos_feb := total_photos - photos_jan
  let photos_per_week := photos_feb / feb_weeks
  sorry

end shane_photos_per_week_l644_644039


namespace range_of_a_l644_644707

noncomputable def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : a ≤ -1 / 2 ∨ a ≥ 2 :=
  sorry

end range_of_a_l644_644707


namespace fixed_point_exists_l644_644602

variable {f g : ℝ → ℝ}

theorem fixed_point_exists (h_cont_f : Continuous f) (h_cont_g : Continuous g)
  (h_comp : ∀ x ∈ (Set.Icc 0 1), f (g x) = g (f x)) 
  (h_f_increasing : ∀ x y ∈ (Set.Icc 0 1), x ≤ y → f x ≤ f y) 
  (h_f_range : ∀ x ∈ (Set.Icc 0 1), f x ∈ Set.Icc 0 1) 
  (h_g_range : ∀ x ∈ (Set.Icc 0 1), g x ∈ Set.Icc 0 1) : 
  ∃ a ∈ (Set.Icc 0 1), f a = a ∧ g a = a := 
sorry

end fixed_point_exists_l644_644602


namespace anne_gave_sweettarts_to_three_friends_l644_644204

theorem anne_gave_sweettarts_to_three_friends (sweettarts : ℕ) (eaten : ℕ) (friends : ℕ) 
  (h1 : sweettarts = 15) (h2 : eaten = 5) (h3 : sweettarts = friends * eaten) :
  friends = 3 := 
by 
  sorry

end anne_gave_sweettarts_to_three_friends_l644_644204


namespace chord_length_of_intersecting_circles_l644_644467

theorem chord_length_of_intersecting_circles :
  ∀ (O1 O2 A B C : Point),
    distance O1 O2 = 2 →
    radius O1 = 1 →
    radius O2 = sqrt 2 →
    (B ∈ chord_of_circle O2 A C) →
    (B ∈ circle O1) →
    (A ∈ circle O1) →
    (A ∈ circle O2) →
    midpoint B A C →
    distance A C = 2 * sqrt (7 / 2) :=
by
  sorry

end chord_length_of_intersecting_circles_l644_644467


namespace large_ball_radius_final_radius_l644_644832

-- Define the radius of small balls
def small_ball_radius : ℝ := 0.5

-- Number of small balls
def small_ball_count : ℕ := 12

-- Volume of one small ball using the formula V = (4 / 3) * pi * r^3
def small_ball_volume : ℝ := (4 / 3) * Real.pi * (small_ball_radius ^ 3)

-- Total volume of all small balls
def total_volume : ℝ := small_ball_count * small_ball_volume

-- The formula for the volume of the large ball with radius R
def large_ball_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * (R ^ 3)

-- The radius of the larger ball is R such that the total volume is equal to the volume of the large ball
theorem large_ball_radius :
  ∃ R : ℝ, large_ball_volume R = total_volume ∧ R = (3 / 2) ^ (1 / 3) := 
by
  existsi (3 / 2) ^ (1 / 3)
  split
  · -- Proof that the volume formula holds
    unfold large_ball_volume total_volume small_ball_volume
    rw [mul_comm (_ : ℝ)]
    norm_num
    ring -- In general you should show the equality holds but we use ring due to large calculation.
  · -- Proof that R matches the desired radius
    rfl -- Since we chose exactly (3 / 2)^(1/3)

-- The desired theorem statement without proof
theorem final_radius : 
  ∃ R : ℝ, (4 / 3) * Real.pi * (R ^ 3) = 12 * (4 / 3) * Real.pi * (0.5 ^ 3) 
    ∧ R = (3 / 2) ^ (1 / 3) := sorry

end large_ball_radius_final_radius_l644_644832


namespace continued_fraction_Pn_Qn_eq_Fibonacci_l644_644380

def P (n : ℕ) : ℕ := (Fibonacci (n+1))
def Q (n : ℕ) : ℕ := Fibonacci n

theorem continued_fraction_Pn_Qn_eq_Fibonacci (n : ℕ) :
  ∃ (P Q : ℕ → ℕ), (P = λ n, Fibonacci (n + 1)) ∧ (Q = λ n, Fibonacci n) ∧ 
  (∀ n, (P (n+1), Q (n+1)) = (Q n, P n + Q n)) ∧ (P 1 = 1) ∧ (Q 1 = 1) :=
by {
  let P := λ n, Fibonacci (n + 1),
  let Q := λ n, Fibonacci n,
  use [P, Q],
  split,
  { refl },
  split,
  { refl },
  split,
  { intro n,
    exact ⟨Q n, P n + Q n⟩ },
  split,
  { exact rfl },
  { exact rfl }
}

end continued_fraction_Pn_Qn_eq_Fibonacci_l644_644380


namespace shadow_boundary_function_correct_l644_644452

noncomputable def sphereShadowFunction : ℝ → ℝ :=
  λ x => (x + 1) / 2

theorem shadow_boundary_function_correct :
  ∀ (x y : ℝ), 
    -- Conditions: 
    -- The sphere with center (0,0,2) and radius 2
    -- A light source at point P = (1, -2, 3)
    -- The shadow must lie on the xy-plane, so z-coordinate is 0
    (sphereShadowFunction x = y) ↔ (- x + 2 * y - 1 = 0) :=
by
  intros x y
  sorry

end shadow_boundary_function_correct_l644_644452


namespace prime_sum_diff_condition_unique_l644_644249

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def can_be_written_as_sum_of_two_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime (p + q)

def can_be_written_as_difference_of_two_primes (p r : ℕ) : Prop :=
  is_prime p ∧ is_prime r ∧ is_prime (p - r)

-- Question rewritten as Lean statement
theorem prime_sum_diff_condition_unique (p q r : ℕ) :
  is_prime p →
  can_be_written_as_sum_of_two_primes (p - 2) p →
  can_be_written_as_difference_of_two_primes (p + 2) p →
  p = 5 :=
sorry

end prime_sum_diff_condition_unique_l644_644249


namespace largest_three_digit_multiple_of_six_with_sum_fifteen_l644_644861

theorem largest_three_digit_multiple_of_six_with_sum_fifteen : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n % 6 = 0) ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ (m % 6 = 0) ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
  sorry

end largest_three_digit_multiple_of_six_with_sum_fifteen_l644_644861


namespace no_b_for_221_square_l644_644222

theorem no_b_for_221_square (b : ℕ) (h : b ≥ 3) :
  ¬ ∃ n : ℕ, 2 * b^2 + 2 * b + 1 = n^2 :=
by
  sorry

end no_b_for_221_square_l644_644222


namespace Shelby_rainy_time_l644_644041

-- Define the problem parameters and conditions
def sunny_speed : ℝ := 35 / 60  -- miles per minute
def rainy_speed : ℝ := 25 / 60  -- miles per minute
def total_distance : ℝ := 22.5  -- miles
def total_time : ℝ := 50  -- minutes

theorem Shelby_rainy_time :
  ∃ (y : ℝ), (y = 40) ∧ (sunny_speed * (total_time - y) + rainy_speed * y = total_distance) :=
begin
  -- Existential statement with a condition to prove
  sorry
end

end Shelby_rainy_time_l644_644041


namespace identify_random_event_l644_644577

-- Definitions of the events
def event1 : String := "Tossing a coin twice in a row and getting heads both times"
def event2 : String := "Opposite charges attract each other"
def event3 : String := "Water freezes at 1 ℃ under standard atmospheric pressure"

-- Statements about the type of events
def is_random_event (e : String) : Prop := 
  e = event1 -- We are directly identifying event1 as the random event here.

theorem identify_random_event : is_random_event event1 :=
by
  sorry

end identify_random_event_l644_644577


namespace arithmetic_sequence_sum_l644_644105

variable (S : ℕ → ℕ) -- Define a function S that gives the sum of the first n terms.
variable (n : ℕ)     -- Define a natural number n.

-- Conditions based on the problem statement
axiom h1 : S n = 3
axiom h2 : S (2 * n) = 10

-- The theorem we need to prove
theorem arithmetic_sequence_sum : S (3 * n) = 21 :=
by
  sorry

end arithmetic_sequence_sum_l644_644105


namespace part_a_inequality_l644_644520

theorem part_a_inequality {n : ℕ} (hn : 0 < n) : 
    (1 + 1 / (n : ℝ)) ^ n < 2 + ∑ i in finset.range n \ 1, 1 / (i + 1)! :=
sorry

end part_a_inequality_l644_644520


namespace water_layer_thickness_after_removing_sphere_l644_644543

theorem water_layer_thickness_after_removing_sphere :
  let R := 4
  let r := 3
  let H := 2 * r
  let V := (Math.pi) * R^2 * H
  let V1 := (4/3) * Math.pi * r^3
  let V0 := V - V1
  h = V0 / (Math.pi * R^2)
  ∧ h = 3.75
  :
  h = 3.75 := 
by
  sorry

end water_layer_thickness_after_removing_sphere_l644_644543


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644063

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644063


namespace no_pq_product_x5_plus_2x_plus_1_l644_644806

theorem no_pq_product_x5_plus_2x_plus_1 :
  ¬ ∃ (p q : Polynomial ℤ), (p.degree ≥ 1) ∧ (q.degree ≥ 1) ∧ (p * q = Polynomial.C 1 * X^5 + Polynomial.C 2 * X + Polynomial.C 1) :=
begin
  sorry
end

end no_pq_product_x5_plus_2x_plus_1_l644_644806


namespace convergence_of_series_l644_644877

noncomputable section

open Classical

variable {α : Type*}

-- Definitions as per the conditions
def a_n (n : ℕ) : ℝ := -- Sequence of positive reals, to be defined
sorry

def s_n (n : ℕ) : ℝ :=
∑ i in finset.range n, 1 / a_n i

-- The hypothesis that the series ∑ 1/a_n converges
axiom H_a : ∃ l, HasSum (λ n, 1 / a_n n) l

-- The statement to prove
theorem convergence_of_series :
  ∃ l, HasSum (λ n, n^2 * a_n n / s_n n^2) l :=
sorry

end convergence_of_series_l644_644877


namespace green_tea_leaves_required_l644_644940

theorem green_tea_leaves_required (S_m : ℕ) (L_s : ℕ) (halved : ℕ) : 
  S_m = 3 → 
  L_s = 2 → 
  halved = 2 → 
  2 * (S_m * L_s) = 12 :=
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  simp 
  norm_num 
  sorry

end green_tea_leaves_required_l644_644940


namespace smallest_b_in_AP_l644_644774

theorem smallest_b_in_AP (a b c : ℝ) (d : ℝ) (ha : a = b - d) (hc : c = b + d) (habc : a * b * c = 125) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) : 
    b = 5 :=
by
  -- Proof needed here
  sorry

end smallest_b_in_AP_l644_644774


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644066

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644066


namespace UnionMathInstitute_students_l644_644083

theorem UnionMathInstitute_students :
  ∃ n : ℤ, n < 500 ∧ 
    n % 17 = 15 ∧ 
    n % 19 = 18 ∧ 
    n % 16 = 7 ∧ 
    n = 417 :=
by
  -- Problem setup and constraints
  sorry

end UnionMathInstitute_students_l644_644083


namespace toothpicks_required_l644_644969

theorem toothpicks_required (n : ℕ) : n = 1002 → 
  let T := (n * (n + 1)) / 2 in
  let side_boundary := n * 3 in
  let total_initial_sides := 3 * T in
  let total_shared_sides := total_initial_sides - side_boundary in
  let total_toothpicks := total_shared_sides / 2 + side_boundary in
  total_toothpicks = 752253 := 
by
  sorry

end toothpicks_required_l644_644969


namespace sequence_general_form_sum_first_n_terms_l644_644704

-- Given Conditions
def S_n (a : ℕ → ℕ) (n : ℕ) [h : fact (0 < n)] : ℕ := 2 * a n - 2

-- Problem 1: Prove the sequence {a_n} has the general form a_n = 2^n
theorem sequence_general_form (a : ℕ → ℕ) (h₀ : ∀ n, 0 < n → S_n a n = 2 * a n - 2) 
    (h₁ : a 1 = 2) (hrec : ∀ n, 1 < n → a (n + 1) = 2 * a n) : 
  ∀ n, a n = 2 ^ n := 
sorry

-- Problem 2: Prove the sum of the first n terms T_n of the sequence {b_n} is T_n = (n - 2) * 2^(n + 1) + 4
theorem sum_first_n_terms (a : ℕ → ℕ) (b : ℕ → ℕ) 
    (h₀ : ∀ n, b n = (n - 1) * a n) (h₁ : a 1 = 2) 
    (hrec : ∀ n, 1 < n → a (n + 1) = 2 * a n) : 
  ∀ n, ∑ i in finset.range n, b (i + 1) = (n - 2) * 2 ^ (n + 1) + 4 := 
sorry

end sequence_general_form_sum_first_n_terms_l644_644704


namespace distance_from_point_to_plane_is_correct_l644_644301

def point (x y z : ℝ) := (x, y, z)

def normal_vector (a b c : ℝ) := (a, b, c)

noncomputable def distance_from_point_to_plane (P A : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) :=
  let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3)
  in abs (AP.1 * n.1 + AP.2 * n.2 + AP.3 * n.3) / real.sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2)

theorem distance_from_point_to_plane_is_correct :
  distance_from_point_to_plane (point -2 3 1) (point 1 2 3) (normal_vector 1 (-1) 1) = 2 * real.sqrt 3 :=
by
  sorry

end distance_from_point_to_plane_is_correct_l644_644301


namespace tan_inequality_l644_644876

theorem tan_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (real.arctan ((a*d - b*c) / (a*c + b*d)))^2 ≥ 
  2 * (1 - (a*c + b*d) / real.sqrt ((a^2 + b^2) * (c^2 + d^2))) := 
sorry

end tan_inequality_l644_644876


namespace equipment_cannot_pass_through_corridor_l644_644896

theorem equipment_cannot_pass_through_corridor 
  (corridor_width equipment_width equipment_length : ℝ)
  (hw : corridor_width = 3)
  (ew : equipment_width = 1)
  (el : equipment_length = 7) 
  : ¬ (∃ (pass_position : ℝ), pass_position + equipment_length ≤ 2 * corridor_width) := 
begin
  sorry
end

end equipment_cannot_pass_through_corridor_l644_644896


namespace part_a_l644_644527

theorem part_a (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x * y ≠ 1) :
  (x * y) / (1 - x * y) = x / (1 - x) + y / (1 - y) :=
sorry

end part_a_l644_644527


namespace inradius_bound_l644_644015

noncomputable def circumradius (ABC : Triangle) : ℝ := 1
def inradius (ABC : Triangle) : ℝ
def orthicInradius (ABC : Triangle) : ℝ

theorem inradius_bound
    (ABC : Triangle)
    (R : circumradius ABC = 1)
    (r : inradius ABC)
    (p : orthicInradius ABC) :
  p ≤ 1 - (1 + r) ^ 2 := sorry

end inradius_bound_l644_644015


namespace smallest_constant_obtuse_triangle_l644_644234

theorem smallest_constant_obtuse_triangle (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^2 > b^2 + c^2) → (b^2 + c^2) / (a^2) ≥ 1 / 2 :=
by 
  sorry

end smallest_constant_obtuse_triangle_l644_644234


namespace find_k_l644_644369

noncomputable def k : ℝ :=
  let k := Real.sqrt 6236 in (68 + k) / 62

theorem find_k (k : ℝ) (h1 : k > 2)
  (h2 : ∑' n : ℕ, (6 * (n + 1) - 2) * k^(-(n + 1)) = 31 / 9) : k = 147 / 62 := sorry

end find_k_l644_644369


namespace percentage_decrease_in_speed_l644_644559

variable (S : ℝ) (S' : ℝ) (T T' : ℝ)

noncomputable def percentageDecrease (originalSpeed decreasedSpeed : ℝ) : ℝ :=
  ((originalSpeed - decreasedSpeed) / originalSpeed) * 100

theorem percentage_decrease_in_speed :
  T = 40 ∧ T' = 50 ∧ S' = (4 / 5) * S →
  percentageDecrease S S' = 20 :=
by sorry

end percentage_decrease_in_speed_l644_644559


namespace alex_jellybeans_l644_644934

theorem alex_jellybeans (x : ℕ) : x = 254 → x ≥ 150 ∧ x % 15 = 14 ∧ x % 17 = 16 :=
by
  sorry

end alex_jellybeans_l644_644934


namespace probability_all_four_same_n_flips_l644_644847

theorem probability_all_four_same_n_flips :
  let P (n : ℕ) := (1/2)^n in
  ∑' n, P n ^ 4 = 1/15 := 
sorry

end probability_all_four_same_n_flips_l644_644847


namespace shopkeeper_percentage_profit_l644_644918

open Real

def total_weight : ℝ := 280
def percentage_sold_at_20_profit : ℝ := 40 / 100
def percentage_sold_at_30_profit : ℝ := 60 / 100
def profit_at_20_percent : ℝ := 20 / 100
def profit_at_30_percent : ℝ := 30 / 100

theorem shopkeeper_percentage_profit :
  let total_weight : ℝ := 280
      part_1_weight := total_weight * percentage_sold_at_20_profit
      part_2_weight := total_weight * percentage_sold_at_30_profit
      part_1_profit := profit_at_20_percent * part_1_weight
      part_2_profit := profit_at_30_percent * part_2_weight
      total_profit := part_1_profit + part_2_profit
      total_cost_price := total_weight  -- assuming cost price per kg as $1
      percentage_profit := (total_profit / total_cost_price) * 100
  in percentage_profit = 26 :=
by
  sorry

end shopkeeper_percentage_profit_l644_644918


namespace chocolate_cake_eggs_l644_644173

theorem chocolate_cake_eggs :
  ∃ (x : ℕ), (9 * 8) - 57 = 5 * x := 
begin
  use 3,
  norm_num,
end

end chocolate_cake_eggs_l644_644173


namespace cupcake_frosting_l644_644212

theorem cupcake_frosting :
  (let cagney_rate := (1 : ℝ) / 24
   let lacey_rate := (1 : ℝ) / 40
   let sammy_rate := (1 : ℝ) / 30
   let total_time := 12 * 60
   let combined_rate := cagney_rate + lacey_rate + sammy_rate
   total_time * combined_rate = 72) :=
by 
   -- Proof goes here
   sorry

end cupcake_frosting_l644_644212


namespace not_factorable_l644_644798

open Polynomial

def P (x y : ℝ) := x ^ 200 * y ^ 200 + 1

theorem not_factorable (f : Polynomial ℝ) (g : Polynomial ℝ) :
  ¬(P x y = f * g) := sorry

end not_factorable_l644_644798


namespace abs_diff_squares_l644_644126

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end abs_diff_squares_l644_644126


namespace smallest_positive_multiple_of_32_l644_644478

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ n % 32 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 32 = 0 → n ≤ m :=
by
  sorry

end smallest_positive_multiple_of_32_l644_644478


namespace peter_bought_6_ounces_l644_644395

-- Given conditions
def soda_cost_per_ounce : ℝ := 0.25
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.08
def money_spent : ℝ := 1.50

-- Needed definitions and calculations
def original_price (n_ounces : ℝ) : ℝ := soda_cost_per_ounce * n_ounces
def discounted_price (P : ℝ) : ℝ := P * (1 - discount_rate)
def total_price_after_tax (discounted_P : ℝ) : ℝ := discounted_P * (1 + tax_rate)

-- Statement to be proven
theorem peter_bought_6_ounces : 
  ∃ n : ℝ, n ≈ 6 ∧ total_price_after_tax (discounted_price (original_price n)) = money_spent :=
sorry

end peter_bought_6_ounces_l644_644395


namespace initial_money_is_10_l644_644760

-- Definition for the initial amount of money
def initial_money (X : ℝ) : Prop :=
  let spent_on_cupcakes := (1 / 5) * X
  let remaining_after_cupcakes := X - spent_on_cupcakes
  let spent_on_milkshake := 5
  let remaining_after_milkshake := remaining_after_cupcakes - spent_on_milkshake
  remaining_after_milkshake = 3

-- The statement proving that Ivan initially had $10
theorem initial_money_is_10 (X : ℝ) (h : initial_money X) : X = 10 :=
by sorry

end initial_money_is_10_l644_644760


namespace at_least_six_like_all_three_l644_644612

-- Define the number of respondents and percentages for each type of ice cream
def respondents : ℕ := 500
def like_strawberry : ℕ := 230
def like_vanilla : ℕ := 355
def like_chocolate : ℕ := 425

-- The goal is to show that there are at least 6 people who like all three types of ice cream
theorem at_least_six_like_all_three (h_st : like_strawberry = 0.46 * respondents)
                                    (h_vn : like_vanilla = 0.71 * respondents)
                                    (h_ch : like_chocolate = 0.85 * respondents)
                                    (h_total_respondents : respondents = 500) :
  ∃ x, x ≥ 6 ∧ x ≤ respondents ∧ 
  x * 3 + (respondents - x) * 2 ≥ like_strawberry + like_vanilla + like_chocolate :=
sorry

end at_least_six_like_all_three_l644_644612


namespace g_is_polynomial_l644_644585

-- Defining the function g_{k,l}(x)
noncomputable def g (k l : ℕ) (x : ℝ) : ℝ := sorry -- Placeholder for the function definition

-- Conditions for the problem
def in_bounds (k l : ℕ) : Prop := (0 ≤ k + l) ∧ (k + l ≤ 4)

-- Stating the theorem
theorem g_is_polynomial (k l : ℕ) (x : ℝ) (h : in_bounds k l) : 
  ∃ (p : polynomial ℝ), ∀ x, g k l x = p.eval x := sorry

end g_is_polynomial_l644_644585


namespace prove_correct_option_C_l644_644503

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l644_644503


namespace problem_1_problem_2_problem_3_l644_644589

theorem problem_1 : 2^100 - 2^99 = 2^99 :=
by
  sorry

theorem problem_2 (n : ℕ) : 2^n - 2^(n-1) = 2^(n-1) :=
by
  sorry

theorem problem_3 : (2 - 2^2 - 2^3 - 2^4 - … - 2^2013 + 2^2014) = 6 :=
by
  sorry

end problem_1_problem_2_problem_3_l644_644589


namespace v_2008_eq_10109_l644_644975

def sequence_v (n : ℕ) : ℕ :=
if n = 0 then 0 else
  let ⟨k, h⟩ := Nat.find_greatest fun k => k * (k + 1) / 2 < n in
  let group_start := (k * (k + 1) * 5) / 2 + 2 * k in
  group_start + 5 * (n - (k * (k + 1) / 2) - 1)

/-- Prove that the 2008th term of the sequence v_n is 10109. -/
theorem v_2008_eq_10109 :
  sequence_v 2008 = 10109 :=
sorry

end v_2008_eq_10109_l644_644975


namespace area_enclosed_by_circle_below_line_l644_644856

open Real

-- Definition of the problem and necessary conditions
def equation_circle (x y : ℝ) : Prop := x^2 - 18 * x + y^2 - 6 * y = -57
def line_condition (x y : ℝ) : Prop := y = x - 5

-- Representation of the area below the line
noncomputable def area_below_line : ℝ :=
  pi * 33 / 2

-- Theorem statement
theorem area_enclosed_by_circle_below_line :
  (∃ x y : ℝ, equation_circle x y ∧ line_condition x y) →
  ∃ area : ℝ, area = area_below_line :=
sorry

end area_enclosed_by_circle_below_line_l644_644856


namespace prove_correct_option_C_l644_644505

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l644_644505


namespace largest_three_digit_multiple_of_six_with_sum_fifteen_l644_644862

theorem largest_three_digit_multiple_of_six_with_sum_fifteen : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n % 6 = 0) ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ (m % 6 = 0) ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
  sorry

end largest_three_digit_multiple_of_six_with_sum_fifteen_l644_644862


namespace equilateral_triangle_side_length_l644_644767

theorem equilateral_triangle_side_length (A B C D F E : Type) [triangle A B C] [equilateral A B C]
  (FA : ℝ) (CD : ℝ) (DEF_area : ℝ) (DEF_angle : ℝ) (k : ℝ) (s : ℝ)
  (h_FA : FA = 5) (h_CD : CD = 2) (h_DEF_angle : DEF_angle = 60)
  (h_DEF_area : DEF_area = 14 * Real.sqrt 3) :
  ∃ (p q r : ℝ), (r = 989) ∧ (s = p + q * Real.sqrt r ∨ s = p - q * Real.sqrt r) := by
  sorry

end equilateral_triangle_side_length_l644_644767


namespace minimum_marked_cells_l644_644132

def grid := fin 20 × fin 20

def strip := fin 12

def horizontal_strips : set (set grid) :=
  { s | ∃ x : fin 20, ∃ y : fin 9, s = { (x, y + i) | i ∈ fin 12 } }

def vertical_strips : set (set grid) :=
  { s | ∃ x : fin 9, ∃ y : fin 20, s = { (x + i, y) | i ∈ fin 12 } }

def marked_cells (S : set grid) (n : ℕ) := ∀ s ∈ horizontal_strips ∪ vertical_strips,
  ∃ cell ∈ s, cell ∈ S

theorem minimum_marked_cells (n : ℕ) :
  (∃ S : set grid, marked_cells S n) → n = 32 :=
sorry

end minimum_marked_cells_l644_644132


namespace irreducible_polynomial_l644_644042

open Polynomial

theorem irreducible_polynomial (n : ℕ) : Irreducible ((X^2 + X)^(2^n) + 1 : ℤ[X]) := sorry

end irreducible_polynomial_l644_644042


namespace arithmetic_sequence_difference_l644_644133

theorem arithmetic_sequence_difference :
  let a := -8 
  let d := -2 - a
  abs ((a + 2004 * d) - (a + 1999 * d)) = 30 := 
by 
  let a := -8
  let d := 6
  show abs ((a + 2004 * d) - (a + 1999 * d)) = 30 from sorry

end arithmetic_sequence_difference_l644_644133


namespace evaluate_g_f_l644_644733

def f (a b : ℤ) : ℤ × ℤ := (-a, b)

def g (m n : ℤ) : ℤ × ℤ := (m, -n)

theorem evaluate_g_f : g (f 2 (-3)).1 (f 2 (-3)).2 = (-2, 3) := by
  sorry

end evaluate_g_f_l644_644733


namespace minimum_translation_l644_644848

theorem minimum_translation {f : ℝ → ℝ} (h1 : ∀ x, f x = (sin x)^2 - 2 * (sin x) * (cos x) + 3 * (cos x)^2) {m : ℝ} (h2 : m > 0) :
  let g (x : ℝ) := f (x + m / (2 * π)) in
  (∀ x, g (x + π / 8) = g (π / 8 - x)) → m = π / 4 :=
sorry

end minimum_translation_l644_644848


namespace next_number_in_sequence_l644_644021

theorem next_number_in_sequence :
  let numerator := 6
  let denominator := 2 + 3 + 5 + 7 + 9 + 11
  numerator / denominator = -6 / 37 := by
  let numerator := 6
  let denominator := 2 + 3 + 5 + 7 + 9 + 11
  have h : denominator = 37 := by
    calc
      denominator = 2 + 3 + 5 + 7 + 9 + 11 : by rfl
      ... = 37 : by norm_num
  rw [h]
  exact (eq.refl (-6 / 37))


end next_number_in_sequence_l644_644021


namespace phase_shift_of_combined_function_l644_644610

theorem phase_shift_of_combined_function :
  let f := λ x : ℝ, 3 * sin (3 * x + π / 4) + 2 * cos (3 * x + π / 4)
  ∃ phase_shift : ℝ, phase_shift = -π / 12 :=
by
  sorry

end phase_shift_of_combined_function_l644_644610


namespace domain_of_sqrt_sin_l644_644253

open Real Set

noncomputable def domain_sqrt_sine : Set ℝ :=
  {x | ∃ (k : ℤ), 2 * π * k + π / 6 ≤ x ∧ x ≤ 2 * π * k + 5 * π / 6}

theorem domain_of_sqrt_sin (x : ℝ) :
  (∃ y, y = sqrt (2 * sin x - 1)) ↔ x ∈ domain_sqrt_sine :=
sorry

end domain_of_sqrt_sin_l644_644253


namespace evaluate_expression_l644_644349

variable (m n p q s : ℝ)

theorem evaluate_expression :
  m / (n - (p + q * s)) = m / (n - p - q * s) :=
by
  sorry

end evaluate_expression_l644_644349


namespace factorize_expr_l644_644988

theorem factorize_expr (x : ℝ) : x^3 - 16 * x = x * (x + 4) * (x - 4) :=
sorry

end factorize_expr_l644_644988


namespace Pascal_concurrent_l644_644378

/-- Begin of the problem statement -/
variables {A B C D E F K L : Type*} [incircle : InscribedHexagon A B C D E F]
variables {K_intersection : K = AC ∩ BF}
variables {L_intersection : L = CE ∩ FD}

theorem Pascal_concurrent : ConcurrentLines A D K L B E := sorry
/-- End of the problem statement -/

end Pascal_concurrent_l644_644378


namespace find_abc_l644_644329

-- Given conditions: a, b, c are positive real numbers and satisfy the given equations.
variables (a b c : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)
variable (h1 : a * (b + c) = 152)
variable (h2 : b * (c + a) = 162)
variable (h3 : c * (a + b) = 170)

theorem find_abc : a * b * c = 720 := 
  sorry

end find_abc_l644_644329


namespace arithmetic_sequence_value_l644_644351

-- Defining the arithmetic sequence and conditions
variable {a : ℕ → ℝ} -- assuming sequence over natural numbers

-- Condition given in the problem
def condition := (a 3 + a 8 = 6)

-- Proving the main statement with the condition
theorem arithmetic_sequence_value (h: condition) : 3 * a 2 + a 16 = 12 :=
by
  sorry

end arithmetic_sequence_value_l644_644351


namespace problem_I_solved_problem_II_solved_problem_III_solved_l644_644906

noncomputable def arrangements_students_together : ℕ := 
  fact 5 * fact 4

noncomputable def arrangements_no_adjacent_students : ℕ := 
  (fact 4) * (nat.choose 5 4) * (fact 4)

noncomputable def arrangements_alternating : ℕ := 
  2 * (fact 4) * (fact 4)

theorem problem_I_solved : arrangements_students_together = 2880 :=
by sorry

theorem problem_II_solved : arrangements_no_adjacent_students = 2880 :=
by sorry

theorem problem_III_solved : arrangements_alternating = 1152 :=
by sorry

end problem_I_solved_problem_II_solved_problem_III_solved_l644_644906


namespace part_one_part_two_l644_644308

open Real

noncomputable def f (x : ℝ) : ℝ := (exp x) / x

theorem part_one (m : ℝ) (h1 : 0 < m) :
  (m ≥ 1 → ∀ x ∈ set.Icc m (m + 1), f x ≥ f m) ∧
  (m < 1 → ∀ x ∈ set.Icc m (m + 1), f x ≥ f 1) :=
by
  sorry

noncomputable def g (x : ℝ) : ℝ := (exp x) / x + x + 1 / x

theorem part_two (λ : ℝ) :
  (∀ x > 0, x * f x > -x^2 + λ * x - 1) ↔ λ < exp 1 + 2 :=
by
  sorry

end part_one_part_two_l644_644308


namespace math_problem_l644_644495

theorem math_problem (a b m : ℝ) : 
  ¬((-2 * a) ^ 2 = -4 * a ^ 2) ∧ 
  ¬((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧ 
  ((-m + 2) * (-m - 2) = m ^ 2 - 4) ∧ 
  ¬((a ^ 5) ^ 2 = a ^ 7) :=
by 
  split ; 
  sorry ;
  split ;
  sorry ;
  split ; 
  sorry ;
  sorry

end math_problem_l644_644495


namespace max_value_of_f_smallest_positive_period_of_f_no_acute_angle_alpha_l644_644412

def f (x : ℝ) : ℝ := 6 * (cos x) ^ 2 - sin (2 * x)

-- (1) Prove that the maximum value of f(x) is 6
theorem max_value_of_f : ∀ x : ℝ, f x ≤ 6 :=
by
  sorry

-- (2) Prove that the smallest positive period of f(x) is π
theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x :=
by
  sorry

-- (3) Prove that there is no acute angle α such that f(α) = 1
theorem no_acute_angle_alpha : ∀ α : ℝ, (0 < α ∧ α < π / 2) → f α ≠ 1 :=
by
  sorry

end max_value_of_f_smallest_positive_period_of_f_no_acute_angle_alpha_l644_644412


namespace option_C_correct_l644_644484

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l644_644484


namespace probability_of_odd_number_l644_644428

-- Define the total number of digits
def digits : List ℕ := [2, 3, 5, 7, 9]

-- Define what it means for a number to be odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the problem of arranging the digits into a five-digit number that is odd
def num_of_arrangements (d : List ℕ) : ℕ := 5.factorial

def num_of_odd_arrangements (d : List ℕ) : ℕ := 4 * 4.factorial

-- Define the probability 
def probability_odd (d : List ℕ) : ℚ := num_of_odd_arrangements d / num_of_arrangements d

-- Statement: Prove that the probability is 4/5
theorem probability_of_odd_number : probability_odd digits = 4 / 5 := by
  sorry

end probability_of_odd_number_l644_644428


namespace sum_of_smallest_and_largest_in_last_row_l644_644892

-- Define the grid size
def grid_size : ℕ := 10

-- Define the list of numbers from 1 to 100
def numbers_list : list ℕ := list.range' 1 100 -- list.range' starts at 1 up to 100

-- Function to determine the last row numbers in a 10x10 grid
def last_row_numbers (grid_size : ℕ) : list ℕ :=
  let start_idx := grid_size * (grid_size - 1) + 1 in
  list.range' start_idx grid_size

-- The main theorem that needs to be proven
theorem sum_of_smallest_and_largest_in_last_row : 
  (last_row_numbers grid_size).minimum = some 91 ∧ 
  (last_row_numbers grid_size).maximum = some 100 →
  list.sum (option.to_list (last_row_numbers grid_size).minimum ++ 
            option.to_list (last_row_numbers grid_size).maximum) = 191 :=
by
  intro h
  cases h with h_min h_max
  rw [option.to_list_some h_min, option.to_list_some h_max, list.sum_cons, list.sum_cons, list.sum_nil]
  exact dec_trivial

-- Helper lemma to identify the contents of the last_row_numbers
lemma last_row_numbers_contents :
  last_row_numbers grid_size = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100] :=
by sorry

end sum_of_smallest_and_largest_in_last_row_l644_644892


namespace integral_bounds_l644_644373

theorem integral_bounds (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_f0 : f 0 = 0) (h_f1 : f 1 = 1)
  (h_f'_bound : ∀ x, |derivative f x| ≤ 2) :
  ∃ a b, (∀ y, y = ∫ x in 0..1, f x → y ∈ Set.Ioo a b) ∧ (b - a = 3/4) := 
sorry

end integral_bounds_l644_644373


namespace four_painters_work_days_l644_644759

theorem four_painters_work_days
  (five_painters_time : ℝ)
  (rate : ℝ)
  (same_rate : ∀ (a b : ℝ), a / rate = b / rate → a = b)
  (work_done_by_five_painters : 5 * rate * five_painters_time = 4) :
  ∃ (D : ℝ), D = 1 :=
by
  -- Define the number of work-days for four painters
  let D := 4 / rate
  -- Show that D equals 1
  have h1 : 4 * D = 4 := by
    simp [D]
    ring
  existsi D
  exact same_rate (4 * D) 4 h1


end four_painters_work_days_l644_644759


namespace range_of_a_for_decreasing_exponential_l644_644157

theorem range_of_a_for_decreasing_exponential :
  ∀ (a : ℝ), (∀ (x1 x2 : ℝ), x1 < x2 → (2 - a)^x1 > (2 - a)^x2) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_for_decreasing_exponential_l644_644157


namespace platform_length_correct_l644_644195

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_cross_platform : ℝ := 30
noncomputable def time_cross_man : ℝ := 19
noncomputable def length_train : ℝ := train_speed_mps * time_cross_man
noncomputable def total_distance_cross_platform : ℝ := train_speed_mps * time_cross_platform
noncomputable def length_platform : ℝ := total_distance_cross_platform - length_train

theorem platform_length_correct : length_platform = 220 := by
  sorry

end platform_length_correct_l644_644195


namespace factorize_expr_l644_644989

theorem factorize_expr (x : ℝ) : x^3 - 16 * x = x * (x + 4) * (x - 4) :=
sorry

end factorize_expr_l644_644989


namespace permutation_6_2_l644_644868

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def permutation (n k : ℕ) : ℕ := factorial n / factorial (n - k)

theorem permutation_6_2 : permutation 6 2 = 30 := by
  have fact6 : factorial 6 = 720 := by
    sorry
  have fact4 : factorial 4 = 24 := by
    sorry
  show permutation 6 2 = 30
  calc
    permutation 6 2 
        = 720 / 24 : by rw [permutation, fact6, fact4]
    ... = 30        : by norm_num

end permutation_6_2_l644_644868


namespace purchasing_methods_l644_644540

theorem purchasing_methods :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 7 ∧
    ∀ (x y : ℕ), (x, y) ∈ s ↔ 60 * x + 70 * y ≤ 500 ∧ 3 ≤ x ∧ 2 ≤ y :=
sorry

end purchasing_methods_l644_644540


namespace decreasing_interval_of_ln_l644_644422

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f y ≤ f x

def ln_decreasing_interval : set ℝ :=
  {x | (5/2 : ℝ) < x ∧ x < 3}

theorem decreasing_interval_of_ln : is_decreasing_on (λ x, Real.log (-x^2 + 5 * x - 6)) ln_decreasing_interval :=
sorry

end decreasing_interval_of_ln_l644_644422


namespace intersection_of_lines_l644_644607

theorem intersection_of_lines : 
  (∃ x y : ℚ, y = -3 * x + 1 ∧ y = 5 * x + 4) ↔ 
  (∃ x y : ℚ, x = -3 / 8 ∧ y = 17 / 8) :=
by
  sorry

end intersection_of_lines_l644_644607


namespace maximize_annual_profit_l644_644299

noncomputable def profit_function (x : ℝ) : ℝ :=
  - (1 / 3) * x^3 + 81 * x - 234

theorem maximize_annual_profit :
  ∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function x :=
sorry

end maximize_annual_profit_l644_644299


namespace number_of_students_above_120_l644_644928

noncomputable theory
open_locale classical

-- Given conditions
def total_students : ℕ := 1000
def score_distribution (ξ : ℝ) : Prop := ∀ (μ σ : ℝ), ξ ~ N(μ = 100, σ^2)
def probability_interval : Prop := P(80 ≤ ξ ∧ ξ ≤ 100) = 0.45

-- Problem statement
theorem number_of_students_above_120 :
  (∀ (ξ : ℝ), score_distribution ξ) →
  probability_interval →
  ∑ x in (multiset.filter (≥ 120) (multiset.range total_students)), 1 = 50 :=
by
  intro score_distribution probability_interval
  sorry

end number_of_students_above_120_l644_644928


namespace probability_x_gt_8y_in_rectangle_l644_644791

open Real

theorem probability_x_gt_8y_in_rectangle :
  let A_triangle := (3013 * 3013) / 16
  let A_rectangle := 3013 * 3014
  let P := A_triangle / A_rectangle
  P = (3013 : ℝ) / 48224 := 
by
  let A_triangle := (3013 * 3013) / 16
  let A_rectangle := 3013 * 3014
  let P := A_triangle / A_rectangle
  have area_eq : A_triangle = (3013 : ℝ) * (3013 : ℝ) / 16 := by sorry
  have rect_eq : A_rectangle = (3013 : ℝ) * (3014 : ℝ) := by sorry
  have prob_eq : P = (3013 : ℝ) / (16 * 3014) := by 
    rw [area_eq, rect_eq]
    sorry
  show P = (3013 : ℝ) / 48224 from
    calc P = (3013 : ℝ) / (16 * 3014) : prob_eq
    ... = (3013 : ℝ) / 48224 : by norm_num

end probability_x_gt_8y_in_rectangle_l644_644791


namespace total_profit_l644_644574

-- Define the variables for the subscriptions and profits
variables {A B C : ℕ} -- Subscription amounts
variables {profit : ℕ} -- Total profit

-- Given conditions
def conditions (A B C : ℕ) (profit : ℕ) :=
  50000 = A + B + C ∧
  A = B + 4000 ∧
  B = C + 5000 ∧
  A * profit = 29400 * 50000

-- Statement of the theorem
theorem total_profit (A B C : ℕ) (profit : ℕ) (h : conditions A B C profit) :
  profit = 70000 :=
sorry

end total_profit_l644_644574


namespace smallest_value_of_x_l644_644439

-- Conditions
def has_eight_factors (x : ℕ) : Prop :=
  nat.factors_count x = 8

def is_factor_of (a b : ℕ) : Prop :=
  b % a = 0

-- Conditional variable
def x := 1134

-- Proof (Statement only, proof not required)
theorem smallest_value_of_x (hx1 : has_eight_factors x) (hx2 : is_factor_of 14 x) (hx3 : is_factor_of 18 x) : x = 1134 := 
sorry

end smallest_value_of_x_l644_644439


namespace max_distance_MN_l644_644687

noncomputable def C1_as_rectangular_coords : ∀ (x y : ℝ), Prop :=
λ x y, x^2 + y^2 - 2 * y = 0

noncomputable def C2_as_general_form (t : ℝ) : (ℝ × ℝ) :=
  (-3 / 5 * t + 2, 4 / 5 * t)

noncomputable def point_M := (2 : ℝ, 0 : ℝ)

theorem max_distance_MN : 
  ∀ N : ℝ × ℝ, 
    (C1_as_rectangular_coords N.1 N.2) → 
      (sqrt ((N.1 - point_M.1)^2 + (N.2 - point_M.2)^2) ≤ sqrt 5 + 1) :=
begin
  sorry
end

end max_distance_MN_l644_644687


namespace initial_percentage_of_water_l644_644389

/-
Initial conditions:
- Let W be the initial percentage of water in 10 liters of milk.
- The mixture should become 2% water after adding 15 liters of pure milk to it.
-/

theorem initial_percentage_of_water (W : ℚ) 
  (H1 : 0 < W) (H2 : W < 100) 
  (H3 : (10 * (100 - W) / 100 + 15) / 25 = 0.98) : 
  W = 5 := 
sorry

end initial_percentage_of_water_l644_644389


namespace find_cos_theta_l644_644367

variable {V : Type} [InnerProductSpace ℝ V] (a b c : V)

-- Conditions
axiom no_parallel_nonzero (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) : ¬Collinear ℝ (set_of (λ v, v ∈ {a, b, c}))
axiom given_identity : (a × b) × c = (1 / 4) * ‖b‖ * ‖c‖ • a 

-- Question to prove
theorem find_cos_theta (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_no_parallel : ¬Collinear ℝ (set_of (λ v, v ∈ {a, b, c})))
    (g_identity : (a × b) × c = (1 / 4) * ‖b‖ * ‖c‖ • a) : 
  cos_angle b c = -1 / 4 :=
sorry

end find_cos_theta_l644_644367


namespace work_completion_time_l644_644522

theorem work_completion_time (P W : ℕ) (h : P * 8 = W) : 2 * P * 2 = W / 2 := by
  sorry

end work_completion_time_l644_644522


namespace division_of_repeating_decimal_l644_644472

theorem division_of_repeating_decimal :
  let x := 142857 / 999999 in
  7 / x = 49 :=
by {
  assert x_def : x = 142857 / 999999 := by simp,
  have : 7 / (1 / 7) = 49,
  rw [x_def],
  linarith,
  simp,
  sorry
}

end division_of_repeating_decimal_l644_644472


namespace complement_of_singleton_in_intervals_l644_644710

def U : Set ℝ := set.Icc 0 1
def A : Set ℝ := {1}

theorem complement_of_singleton_in_intervals :
  (U \ A) = set.Ico 0 1 :=
sorry

end complement_of_singleton_in_intervals_l644_644710


namespace two_pow_neg_y_l644_644723

theorem two_pow_neg_y (y : ℝ) (h : 2^(4 * y) = 16) : 2^(-y) = 1/2 :=
sorry

end two_pow_neg_y_l644_644723


namespace sum_lucky_numbers_divisible_by_2002_l644_644534

-- Define a structure for six-digit numbers
structure LuckyNumber where
  a b c d e f : ℕ
  h₁ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f
  h₂ : a + b + c = d + e + f

theorem sum_lucky_numbers_divisible_by_2002 : 
  let S := Finset.univ.filter (λ N : LuckyNumber, True)
  (S.sum (λ N, 100000 * N.a + 10000 * N.b + 1000 * N.c + 100 * N.d + 10 * N.e + N.f)) % 2002 = 0 :=
sorry

end sum_lucky_numbers_divisible_by_2002_l644_644534


namespace sum_phi_fractional_parts_l644_644640

-- Definition of the function phi(y)
def phi (y : ℝ) : ℝ :=
  ∑' (m : ℕ) in Finset.range (m + 1), 1 / (m ^ y)

-- Definition of the fractional part of x
def frac_part (x : ℝ) : ℝ :=
  x - x.floor

-- The main theorem to prove
theorem sum_phi_fractional_parts :
  ∑' (j : ℕ) in Finset.range (j + 2), frac_part (phi (2 * j)) = 1 / 4 :=
begin
  sorry
end

end sum_phi_fractional_parts_l644_644640


namespace inequality_correct_transformation_l644_644870

-- Definitions of the conditions
variables (a b : ℝ)

-- The equivalent proof problem
theorem inequality_correct_transformation (h : a > b) : -a < -b :=
by sorry

end inequality_correct_transformation_l644_644870


namespace third_square_length_l644_644953

theorem third_square_length 
  (A1 : 8 * 5 = 40) 
  (A2 : 10 * 7 = 70) 
  (A3 : 15 * 9 = 135) 
  (L : ℕ) 
  (A4 : 40 + 70 + L * 5 = 135) 
  : L = 5 := 
sorry

end third_square_length_l644_644953


namespace goldfish_left_number_of_goldfish_l644_644262

theorem goldfish_left (initial_goldfish : ℕ) (died_goldfish : ℕ) (remaining_goldfish : ℕ) 
  (h1 : initial_goldfish = 89) (h2 : died_goldfish = 32) : 
  remaining_goldfish = initial_goldfish - died_goldfish :=
sorry

theorem number_of_goldfish : goldfish_left 89 32 57 :=
by
  simp [goldfish_left]
  refl

end goldfish_left_number_of_goldfish_l644_644262


namespace length_of_one_side_of_box_l644_644140

def cost_per_box : ℝ := 0.50
def total_cost : ℝ := 225.0
def total_volume : ℝ := 2160000.0

theorem length_of_one_side_of_box :
  let number_of_boxes := total_cost / cost_per_box in
  let volume_per_box := total_volume / number_of_boxes in
  (volume_per_box ^ (1 / 3.0)) ≈ 16.89 :=
by
  sorry

end length_of_one_side_of_box_l644_644140


namespace sum_f_up_to_2017_l644_644728

def f (n : ℕ) : ℝ :=
  if n ≠ 0 then Real.tan (n * Real.pi / 3) else 0

theorem sum_f_up_to_2017 :
  (∑ n in Finset.range 2017, f (n + 1)) = Real.sqrt 3 :=
by
  sorry

end sum_f_up_to_2017_l644_644728


namespace arithmetic_seq_sum_l644_644737

variable {α : Type*} [linear_ordered_field α]

noncomputable def arithmetic_sequence (a d : α) (n : ℕ) : α := a + d * n

theorem arithmetic_seq_sum
  (a : α) (d : α)
  (h_arith : ∀ n : ℕ, ∃ a_n, a_n = arithmetic_sequence a d n)
  (h_eq : ∀ a3 a10 : α, (a3 = arithmetic_sequence a d 3) ∧ (a10 = arithmetic_sequence a d 10) ∧ 
           a3^2 - 3 * a3 - 5 = 0 ∧ a10^2 - 3 * a10 - 5 = 0) :
  arithmetic_sequence a d 5 + arithmetic_sequence a d 8 = 3 :=
by {
  sorry
}

end arithmetic_seq_sum_l644_644737


namespace minimally_intersecting_triples_count_remainder_minimally_intersecting_triples_mod_1000_l644_644224

open Set

-- Define the conditions
def minimally_intersecting (A B C : Set ℕ) : Prop :=
  (|A ∩ B| = 1) ∧ (|B ∩ C| = 1) ∧ (|C ∩ A| = 1) ∧ (A ∩ B ∩ C = ∅)

-- Define the function to count such triples
noncomputable def count_minimally_intersecting_triples : ℕ := sorry

theorem minimally_intersecting_triples_count :
  count_minimally_intersecting_triples = 344064 :=
sorry

theorem remainder_minimally_intersecting_triples_mod_1000 :
  count_minimally_intersecting_triples % 1000 = 64 :=
sorry

end minimally_intersecting_triples_count_remainder_minimally_intersecting_triples_mod_1000_l644_644224


namespace correct_operation_l644_644491

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l644_644491


namespace repeating_decimals_sum_l644_644618

theorem repeating_decimals_sum :
  (0.66666666666666 : ℝ) + (0.22222222222222 : ℝ) - (0.44444444444444 : ℝ) = 4 / 9 :=
by {
  -- Rationalize repeating decimals
  let a : ℚ := 6 / 9 -- Equivalent to 0.66666666666666...
  let b : ℚ := 2 / 9 -- Equivalent to 0.22222222222222...
  let c : ℚ := 4 / 9 -- Equivalent to 0.44444444444444...
  -- Convert to real numbers
  let ra : ℝ := a
  let rb : ℝ := b
  let rc : ℝ := c
  -- Show sum of real numbers is the expected result
  have h : ra + rb - rc = 4 / 9, by {
    norm_num,
  },
  -- Conclude the proof
  exact h
}

end repeating_decimals_sum_l644_644618


namespace max_triangles_in_right_triangle_l644_644789

theorem max_triangles_in_right_triangle (a b : ℕ) (ha : a = 7) (hb : b = 7) : 
  ∃ (n : ℕ), n = 28 :=
begin
  sorry
end

end max_triangles_in_right_triangle_l644_644789


namespace option_C_correct_l644_644483

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l644_644483


namespace henry_has_more_than_500_seeds_on_saturday_l644_644320

theorem henry_has_more_than_500_seeds_on_saturday :
  (∃ k : ℕ, 5 * 3^k > 500 ∧ k + 1 = 6) :=
sorry

end henry_has_more_than_500_seeds_on_saturday_l644_644320


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644065

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l644_644065


namespace inclination_angle_of_line_l644_644823

theorem inclination_angle_of_line (a : ℝ) : 
  let l := (λ x : ℝ, (sqrt 3) * x - 3 * ((λ y, y) x) + a = 0) in 
  ∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ tan (α) = (sqrt 3 / 3) ∧ α = 30 :=
by
  sorry

end inclination_angle_of_line_l644_644823


namespace sum_coordinates_B_l644_644027

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B (x : ℝ) : (ℝ × ℝ) := (x, 4)

theorem sum_coordinates_B 
  (x : ℝ) 
  (h_slope : (4 - 0)/(x - 0) = 3/4) : x + 4 = 28 / 3 := by
sorry

end sum_coordinates_B_l644_644027


namespace sphere_surface_area_in_cube_l644_644358

theorem sphere_surface_area_in_cube :
  let C : Type := Cube 1 -- Cube with side length 1
  let O1 : Sphere := Sphere.inscribed C  -- Sphere O1 inscribed in cube C
  let O2 : Sphere := Sphere.tangent_to_faces_and_sphere C O1 -- Sphere O2 tangent to O1 and three faces of C
  ∃ r : ℝ, -- Existence of radius r of O2
  (r = (2 - real.sqrt 3) / 2) → -- Given radius of O2
  let A := 4 * real.pi * r^2 -- Surface area of O2
  A = (7 - 4 * real.sqrt 3) * real.pi := by sorry

end sphere_surface_area_in_cube_l644_644358


namespace find_max_value_l644_644002

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x

theorem find_max_value (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 = 1) : 
  maximum_value x y z ≤ Real.sqrt 3 := sorry

end find_max_value_l644_644002


namespace branches_after_ten_weeks_l644_644573

theorem branches_after_ten_weeks : (2 + ∑ i in (Finset.range 10), 2^(i+1)) = 2046 :=
by
  sorry

end branches_after_ten_weeks_l644_644573


namespace red_points_even_l644_644410

theorem red_points_even (P : Fin 6 → Point) (general_position : ∀ (i j k l : Fin 6), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
  ¬Coplanar ({P i, P j, P k, P l} : Set Point)) :
  ∀ (red_points : Fin 6 → Fin 6 → Set Point),
  (∀ i j : Fin 6, i ≠ j → red_points i j = intersection_points (segment (P i) (P j)) (tetrahedron ({P k | k ≠ i ∧ k ≠ j} : Set Point))) →
  Even (card { q | ∃ i j : Fin 6, i ≠ j ∧ q ∈ red_points i j }) :=
sorry

end red_points_even_l644_644410


namespace positive_difference_between_median_and_mode_l644_644134

-- Define the dataset based on the provided stem and leaf plot.
def dataset : List ℕ := [21, 25, 25, 26, 26, 26, 33, 33, 33, 37, 37, 37, 37, 40, 42, 45, 48, 48, 51, 55, 55, 59, 59, 59, 59, 59]

-- Define a function to compute the mode of a list
def mode (l : List ℕ) : ℕ :=
  l.foldr (λ x b, if l.count x > l.count b then x else b) 0

-- Define a function to compute the median of a list
noncomputable def median (l : List ℕ) : ℕ :=
  let sorted_l := l.qsort (· ≤ ·)
  if sorted_l.length % 2 = 0 then
    (sorted_l.get (sorted_l.length / 2 - 1) + sorted_l.get (sorted_l.length / 2)) / 2
  else
    sorted_l.get (sorted_l.length / 2)

-- Define a function to compute the positive difference
def positive_diff (a b : ℕ) : ℕ := abs (a - b)

-- The final theorem statement
theorem positive_difference_between_median_and_mode :
  positive_diff (median dataset) (mode dataset) = 22 :=
by
  sorry

end positive_difference_between_median_and_mode_l644_644134


namespace cricket_bat_cost_l644_644565

variable (CP_A : ℝ) (CP_B : ℝ) (CP_C : ℝ)

-- Conditions
def CP_B_def : Prop := CP_B = 1.20 * CP_A
def CP_C_def : Prop := CP_C = 1.25 * CP_B
def CP_C_val : Prop := CP_C = 234

-- Theorem statement
theorem cricket_bat_cost (h1 : CP_B_def CP_A CP_B) (h2 : CP_C_def CP_B CP_C) (h3 : CP_C_val CP_C) : CP_A = 156 :=by
  sorry

end cricket_bat_cost_l644_644565


namespace inequality_multiplication_l644_644270

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end inequality_multiplication_l644_644270


namespace total_students_l644_644744

variable (B G : ℕ) (avg_age_boys avg_age_girls avg_age_school : ℚ)

def school_problem_conditions :=
  avg_age_boys = 12 ∧
  avg_age_girls = 11 ∧
  avg_age_school = 11.75 ∧
  G = 150

theorem total_students (h : school_problem_conditions B G avg_age_boys avg_age_girls avg_age_school) :
  B + G = 600 :=
by
  rcases h with ⟨h_avg_boys, h_avg_girls, h_avg_school, h_G⟩
  sorry

end total_students_l644_644744


namespace collinear_midpoint_incenter_excircle_tangency_l644_644797

-- Define the triangle and its vertices
variables {A B C D M I E : Point}

-- Definitions for the conditions
def isTriangle (A B C : Point) : Prop := -- Assuming some definition of a triangle
sorry

def altitude (A B C : Point) (D : Point) : Prop :=
vertical_perpendicular A B C D -- Assuming vertical_perpendicular defines altitude

def midpoint (A D : Point) (M : Point) : Prop :=
is_midpoint A D M -- Assuming is_midpoint correctly defines a midpoint

def incenter (A B C : Point) (I : Point) : Prop :=
is_incenter A B C I -- Assuming is_incenter correctly identifies the incenter

def excircle_tangency (A B C : Point) (E : Point) : Prop :=
is_excircle_tangent_point A B C E -- Assuming is_excircle_tangent_point identifies the tangency point

-- The main theorem to state
theorem collinear_midpoint_incenter_excircle_tangency
  (hTri : isTriangle A B C)
  (hAlt : altitude A B C D)
  (hMid : midpoint A D M)
  (hInc : incenter A B C I)
  (hExc : excircle_tangency A B C E) : collinear M I E :=
sorry

end collinear_midpoint_incenter_excircle_tangency_l644_644797


namespace hyperbola_eccentricity_l644_644670

open Real

theorem hyperbola_eccentricity (a b : ℝ) (c : ℝ) (F : ℝ × ℝ) (M N : ℝ × ℝ) 
  (hF_foci : F = (c, 0)) 
  (h_hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → True) 
  (h_MN_on_line : ∀ l (h_l_origin : l.1 = 0 ∧ l.2 = 0), M ∈ M ∧ N ∈ N) 
  (h_MN_dot : (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * ( N.2 - F.2) = 0) 
  (h_area : abs ((M.1 * (N.2 - F.2) + N.1 * (F.2 - M.2) + F.1 * (M.2 - N.2)) / 2) = a * b):
  let e := sqrt (1 + (b^2 / a^2)) in 
  e = sqrt 2 := sorry

end hyperbola_eccentricity_l644_644670


namespace sum_of_reversible_digits_in_both_bases_l644_644260

def is_reversible_in_bases (n : ℕ) : Prop :=
  let base5 := n.digits 5
  let base13 := n.digits 13
  base5.reverse = base13

theorem sum_of_reversible_digits_in_both_bases : 
  (Finset.range 101).filter is_reversible_in_bases).sum = 18 :=
by
  sorry

end sum_of_reversible_digits_in_both_bases_l644_644260


namespace num_valid_pairs_proof_l644_644447

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def quadraticDiscriminant (A B : ℕ) : ℕ := 4 * A * A - 4 * A * B

noncomputable def num_valid_pairs : ℕ :=
  let pairs := [(a, b) | a ← List.range 9, b ← List.range 9, a > 0, b > 0]
  pairs.filter (λ (A, B) => isPerfectSquare (quadraticDiscriminant A B)).length

theorem num_valid_pairs_proof : ∃ n : ℕ, n = num_valid_pairs :=
sorry

end num_valid_pairs_proof_l644_644447


namespace team_problem_solved_probability_l644_644207

-- Defining the probabilities
def P_A : ℚ := 1 / 5
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Defining the probability that the problem is solved
def P_s : ℚ := 3 / 5

-- Lean 4 statement to prove that the calculated probability matches the expected solution
theorem team_problem_solved_probability :
  1 - (1 - P_A) * (1 - P_B) * (1 - P_C) = P_s :=
by
  sorry

end team_problem_solved_probability_l644_644207


namespace number_of_cards_le_0_3_l644_644764

theorem number_of_cards_le_0_3 :
  let jungkook_card := 0.8
  let yoongi_card := 1 / 2
  let yoojung_card := 0.9
  let yuna_card := 1 / 3
  in jungkook_card > 0.3 ∧ yoongi_card > 0.3 ∧ yoojung_card > 0.3 ∧ yuna_card > 0.3 → 0 = 0 :=
by
  intros
  sorry

end number_of_cards_le_0_3_l644_644764


namespace find_c_l644_644180

-- Definitions for the problem conditions
def point1 : ℝ × ℝ := (-1, -3)
def point2 : ℝ × ℝ := (2, 1)
def direction_vec_given (c : ℝ) : ℝ × ℝ := (3, c)
def direction_vec_calculated : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)

-- Statement of the proof problem
theorem find_c :
  ∃ c : ℝ, direction_vec_given c = direction_vec_calculated :=
sorry

end find_c_l644_644180


namespace determine_a_l644_644236

theorem determine_a (f : ℝ → ℝ) (a y : ℝ) (h1 : ∀ x, f x = x + (1 / (x + a))) 
  (h2 : ∀ x, f x has_min_on [0, ∞] at x = y) 
  (h3 : ∀ x, f x has_max_on [0, ∞] at x = y / 2) 
  : a = 1 := 
begin
  sorry
end

end determine_a_l644_644236


namespace largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l644_644556

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l644_644556


namespace photos_per_week_in_february_l644_644038

def january_photos : ℕ := 31 * 2

def total_photos (jan_feb_photos : ℕ) : ℕ := jan_feb_photos - january_photos

theorem photos_per_week_in_february (jan_feb_photos : ℕ) (weeks_in_february : ℕ)
  (h1 : jan_feb_photos = 146)
  (h2 : weeks_in_february = 4) :
  total_photos jan_feb_photos / weeks_in_february = 21 := by
  sorry

end photos_per_week_in_february_l644_644038


namespace probability_of_P_eq_neg1_l644_644110

-- Define complex vertices as given in the problem
def V : Set ℂ := {√3 * Complex.I, -√3 * Complex.I, 1 + Complex.I, 1 - Complex.I, -1 + Complex.I, -1 - Complex.I, Complex.I, -Complex.I}

-- Define the random choice of elements and product condition
def chosen_elements (n : ℕ) : Vector ℂ 14 := sorry -- This represents choosing 14 elements from V randomly

def P (z : Vector ℂ 14) : ℂ := z.toList.prod

-- Define the problem statement
theorem probability_of_P_eq_neg1 :
  ( ∃ (a b p : ℕ), Prime p ∧ p = 2 ∧
    let expr := a / (p ^ b) in
    a ≠ 0 ∧ a % p ≠ 0 ∧ 
    (a + b + p = 132) ∧ 
    (let prob := (91 / (2 ^ 39)) in expr = prob)) :=
by
  sorry

end probability_of_P_eq_neg1_l644_644110


namespace initial_water_percentage_l644_644387

variable (W : ℝ) -- Initial percentage of water in the milk

theorem initial_water_percentage 
  (final_water_content : ℝ := 2) 
  (pure_milk_added : ℝ := 15) 
  (initial_milk_volume : ℝ := 10)
  (final_mixture_volume : ℝ := initial_milk_volume + pure_milk_added)
  (water_equation : W / 100 * initial_milk_volume = final_water_content / 100 * final_mixture_volume) 
  : W = 5 :=
by
  sorry

end initial_water_percentage_l644_644387


namespace triangle_DEF_t_squared_l644_644530

-- Definitions from the conditions:
def radius : ℝ := 10
def D : ℝ := 0  -- assuming the semicircle is centered at the origin
def E : ℝ := 2 * radius
def F : ℝ × ℝ := (f_x, sqrt (radius^2 - f_x^2)) -- a point on the semicircle, different from D and E
def t : ℝ := sqrt ((D - F.1)^2 + F.2^2) + sqrt ((E - F.1)^2 + F.2^2)

theorem triangle_DEF_t_squared :
  t^2 = (2 * radius)^2 := by
  sorry

end triangle_DEF_t_squared_l644_644530


namespace angle_in_second_quadrant_l644_644454

theorem angle_in_second_quadrant (θ : ℝ) (h : θ = 3) : θ > (Real.pi * 0.5) ∧ θ < Real.pi :=
by
  rw h
  exact ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos, Real.pi_pos.zero_le]⟩
  sorry

end angle_in_second_quadrant_l644_644454


namespace angle_between_clock_hands_at_five_thirty_is_fifteen_degrees_l644_644954

-- Define the given conditions:
def hour_hand_degrees_per_hour : ℝ := 30
def minute_hand_degrees_per_minute : ℝ := 6
def hour_at_5 : ℝ := 5
def minute_at_30 : ℝ := 30

-- Define calculations:
def hour_hand_position_at_five_thirty : ℝ :=
  (hour_at_5 * hour_hand_degrees_per_hour) + ((minute_at_30 / 60) * hour_hand_degrees_per_hour)

def minute_hand_position_at_thirty : ℝ :=
  minute_at_30 * minute_hand_degrees_per_minute

def angle_between_hands : ℝ :=
  |minute_hand_position_at_thirty - hour_hand_position_at_five_thirty|

-- The proof problem statement:
theorem angle_between_clock_hands_at_five_thirty_is_fifteen_degrees :
  angle_between_hands = 15 := by
  sorry

end angle_between_clock_hands_at_five_thirty_is_fifteen_degrees_l644_644954


namespace complex_magnitude_l644_644226

theorem complex_magnitude (z : ℂ) (n : ℕ) : |(1 + complex.I)^10| = 32 := by
  sorry

end complex_magnitude_l644_644226


namespace ash_cloud_ratio_l644_644931

theorem ash_cloud_ratio
  (distance_ashes_shot_up : ℕ)
  (radius_ash_cloud : ℕ)
  (h1 : distance_ashes_shot_up = 300)
  (h2 : radius_ash_cloud = 2700) :
  (2 * radius_ash_cloud) / distance_ashes_shot_up = 18 :=
by
  sorry

end ash_cloud_ratio_l644_644931


namespace length_of_platform_l644_644165

-- Define the conditions
def trainLength : ℝ := 200
def timePole : ℝ := 42
def timePlatform : ℝ := 50

-- Define the speed of the train
def trainSpeed : ℝ := trainLength / timePole

-- State the theorem to find the length of the platform
theorem length_of_platform : 
  ∃ (lengthPlatform : ℝ), trainLength + lengthPlatform = trainSpeed * timePlatform ∧ lengthPlatform = 38 := by
  sorry

end length_of_platform_l644_644165


namespace smallest_n_partition_l644_644001

theorem smallest_n_partition (n : ℕ) (h : n ≥ 2) :
  (∀ (A B : Set ℕ), A ∪ B = {i | 2 ≤ i ∧ i ≤ n} ∧ A ∩ B = ∅ → 
    (∃ a b c ∈ A, a + b = c) ∨ (∃ a b c ∈ B, a + b = c)) ↔ n = 7 :=
sorry

end smallest_n_partition_l644_644001


namespace exist_points_no_three_collinear_integer_distances_l644_644401

theorem exist_points_no_three_collinear_integer_distances
  (N : ℕ) : ∃ (points : Fin N → ℝ × ℝ), 
  (∀ i j k : Fin N, i ≠ j → i ≠ k → j ≠ k → 
     ¬ collinear ℝ [{| x := points i, y := points j, z := points k |}]) ∧
  (∀ i j : Fin N, i ≠ j → ∃ d : ℕ, dist (points i) (points j) = (d : ℝ)) :=
by
  sorry

end exist_points_no_three_collinear_integer_distances_l644_644401


namespace die_sum_13_not_visible_die_sums_not_visible_is_only_13_l644_644827

noncomputable def die_sums_not_visible : Set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 15 ∧ (∀f1 f2 f3 : ℕ, 
    (f1 ∈ {1, 2, 3, 4, 5, 6}) ∧
    (f2 ∈ {1, 2, 3, 4, 5, 6}) ∧
    (f3 ∈ {1, 2, 3, 4, 5, 6}) ∧
    (f1 ≠ f2) ∧ (f2 ≠ f3) ∧ (f1 ≠ f3) ∧
    (f1 + f2 ≠ n) ∧ (f2 + f3 ≠ n) ∧ (f1 + f3 ≠ n) ∧
    (f1 + f2 + f3 ≠ n))}

theorem die_sum_13_not_visible : 13 ∈ die_sums_not_visible := 
  sorry

theorem die_sums_not_visible_is_only_13 : 
  ∀ n, 1 ≤ n ∧ n ≤ 15 → (n ∈ die_sums_not_visible ↔ n = 13) :=
  sorry

end die_sum_13_not_visible_die_sums_not_visible_is_only_13_l644_644827


namespace part_I_part_II_1_part_II_2_l644_644441

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 1) * x + a
noncomputable def g (x : ℝ) : ℝ := (2 * x) / (3 - x)

-- Part (Ⅰ): Prove the sum of f at specified points
theorem part_I (f : ℝ → ℝ) (h_f_symm : ∀ x, f(x) + f(2 - x) = 4) :
  f 0 + f (1 / 2) + f 1 + f (3 / 2) + f 2 = 10 :=
by
  sorry

-- Part (Ⅱ)(1): Prove the symmetry of g around the point (3, -2)
theorem part_II_1 : ∀ x, g(x) + g(6 - x) = -4 :=
by
  sorry

-- Part (Ⅱ)(2): Range of a such that for any x₁ ∈ [0, 2], there exists x₂ ∈ [-3, 2] so that f(x₁) = g(x₂)
theorem part_II_2 (a : ℝ) :
  (∀ x₁ ∈ set.Icc 0 2, ∃ x₂ ∈ set.Icc (-3) 2, f(x₁) a = g(x₂)) ↔ a ∈ set.Icc 0 3 :=
by
  sorry

end part_I_part_II_1_part_II_2_l644_644441


namespace min_dwarfs_l644_644070

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l644_644070


namespace percent_increase_in_area_l644_644179

theorem percent_increase_in_area (s : ℝ) (h_s : s > 0) :
  let medium_area := s^2
  let large_length := 1.20 * s
  let large_width := 1.25 * s
  let large_area := large_length * large_width 
  let percent_increase := ((large_area - medium_area) / medium_area) * 100
  percent_increase = 50 := by
    sorry

end percent_increase_in_area_l644_644179


namespace tutors_work_together_again_in_360_days_l644_644615

theorem tutors_work_together_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end tutors_work_together_again_in_360_days_l644_644615


namespace small_paintings_sold_l644_644019

theorem small_paintings_sold 
  (price_large : ℕ)
  (price_small : ℕ)
  (large_paintings : ℕ)
  (total_earnings : ℕ)
  (price_large_eq : price_large = 100)
  (price_small_eq : price_small = 80)
  (large_paintings_eq : large_paintings = 5)
  (total_earnings_eq : total_earnings = 1140) :
  ∃ (small_paintings : ℕ), small_paintings = 8 :=
by
  let earnings_large := large_paintings * price_large
  have earnings_large_eq : earnings_large = 500 := by
    rw [large_paintings_eq, price_large_eq]
    refl
  let earnings_small := total_earnings - earnings_large
  have earnings_small_eq : earnings_small = 640 := by
    rw [total_earnings_eq, earnings_large_eq]
    refl
  let small_paintings := earnings_small / price_small
  have small_paintings_eq : small_paintings = 8 := by
    rw [earnings_small_eq, price_small_eq]
    norm_num
  exact ⟨small_paintings, small_paintings_eq⟩

end small_paintings_sold_l644_644019


namespace percent_decrease_second_year_l644_644170

theorem percent_decrease_second_year 
    (initial_value : ℝ) 
    (first_year_decrease : ℝ) 
    (total_decrease : ℝ) : 
    (first_year_decrease = 0.4 * initial_value) → 
    (total_decrease = 0.46 * initial_value) → 
    let second_year_start_value := initial_value - first_year_decrease in
    let second_year_final_value := initial_value - total_decrease in
    let second_year_decrease := second_year_start_value - second_year_final_value in
    (second_year_decrease / second_year_start_value) * 100 = 10 :=
begin
    intros,
    sorry
end

end percent_decrease_second_year_l644_644170


namespace intersection_is_correct_l644_644290

theorem intersection_is_correct :
  let M := {-2, -1, 0, 1}
  let N := {x : ℕ | x^2 - x - 6 < 0}
  M ∩ N = {0, 1} :=
by
  sorry

end intersection_is_correct_l644_644290


namespace function_solution_l644_644276

def f : ℝ → ℝ := sorry -- Placeholder for the function definition

axiom condition1 : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → f(x * f(y)) * f(y) = f(x + y)
axiom condition2 : f 2 = 0
axiom condition3 : ∀ x : ℝ, 0 ≤ x → x < 2 → f x ≠ 0

theorem function_solution (x : ℝ) (hx : 0 ≤ x) : 
  f x = if x < 2 then 2 / (2 - x) else 0 :=
sorry -- Proof to be provided

end function_solution_l644_644276


namespace perfect_numbers_10_45_29_quadratic_trinomial_perfect_number_k_value_for_s_perfect_number_min_value_x_y_l644_644146

-- 1. Proof that 10, 45, 29 are perfect numbers
theorem perfect_numbers_10_45_29 : ∀ n ∈ {10, 45, 29}, ∃ a b : ℤ, n = a^2 + b^2 :=
begin
  sorry
end

-- 2. Showing that x^2 - 6x + 13 can be expressed as (x-m)^2 + n and calculate mn = 12
theorem quadratic_trinomial_perfect_number : ∃ m n : ℤ, (∃ x : ℤ, (x^2 - 6x + 13 = (x - m)^2 + n)) ∧ (m * n = 12) :=
begin
  sorry
end

-- 3. Determine values of k for S = x^2 + 9y^2 + 8x - 12y + k to be a perfect number
theorem k_value_for_s_perfect_number : ∃ k : ℤ, ∀ x y : ℤ, ∃ a b : ℤ, (x^2 + 9y^2 + 8x - 12y + k = a^2 + b^2) ↔ (k = 20) :=
begin
  sorry
end

-- 4. Find the minimum value of x + y given -x^2 + 7x + y - 10 = 0
theorem min_value_x_y : ∃ m : ℝ, ∀ x y : ℝ, (-x^2 + 7x + y - 10 = 0) → (m = x + y) ∧ (m = 1) :=
begin
  sorry
end

end perfect_numbers_10_45_29_quadratic_trinomial_perfect_number_k_value_for_s_perfect_number_min_value_x_y_l644_644146


namespace prove_correct_option_C_l644_644504

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l644_644504


namespace set_intersection_complement_eq_l644_644313

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Complement of B in U
def complement_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- The theorem statement
theorem set_intersection_complement_eq :
  A ∩ complement_B = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_complement_eq_l644_644313


namespace find_x_sum_of_digits_eq_22_l644_644248

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s == s.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c => c.toNat - '0'.toNat)).sum

theorem find_x_sum_of_digits_eq_22 :
  ∃ x : ℕ, (100 ≤ x ∧ x ≤ 999) ∧ is_palindrome x ∧ is_palindrome (x + 34) ∧ sum_of_digits x = 22 :=
by
  -- The proof has been intentionally skipped:
  sorry

end find_x_sum_of_digits_eq_22_l644_644248


namespace integral_bounds_l644_644374

theorem integral_bounds (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_f0 : f 0 = 0) (h_f1 : f 1 = 1)
  (h_f'_bound : ∀ x, |derivative f x| ≤ 2) :
  ∃ a b, (∀ y, y = ∫ x in 0..1, f x → y ∈ Set.Ioo a b) ∧ (b - a = 3/4) := 
sorry

end integral_bounds_l644_644374


namespace ratio_b_over_a_l644_644651

-- Given triangle ABC with specified conditions
variables (A B C a b c : ℝ)

-- Conditions
def C_eq_pi_over_3 : Prop := C = Real.pi / 3
def c_eq_2 : Prop := c = 2

-- Maximize the dot product of vectors AC and AB
def max_dot_product : Prop := ∀ (A : ℝ), 
  let B := (2 * Real.pi) / 3 - A in
  let sin_a := Real.sin A in
  let sin_b := Real.sin B in
  let cos_a := Real.cos A in
  let sin_c := Real.sin (Real.pi / 3) in
  b = (2 / sin_c) * sin_b →
  2 * b * cos_a = max (2 * b * cos_a)

-- Statement to prove
theorem ratio_b_over_a : C_eq_pi_over_3 → c_eq_2 → max_dot_product → (b / a = 2 + Real.sqrt 3) :=
by
  intro h1 h2 h3
  sorry

end ratio_b_over_a_l644_644651


namespace range_of_b_l644_644632

theorem range_of_b (b : ℝ) : (∃ x : ℝ, |x - 2| + |x - 5| < b) → b > 3 :=
by 
-- This is where the proof would go.
sorry

end range_of_b_l644_644632


namespace B2F_base16_to_base10_l644_644972

theorem B2F_base16_to_base10 :
  let d2 := 11
  let d1 := 2
  let d0 := 15
  d2 * 16^2 + d1 * 16^1 + d0 * 16^0 = 2863 :=
by
  let d2 := 11
  let d1 := 2
  let d0 := 15
  sorry

end B2F_base16_to_base10_l644_644972


namespace abs_difference_of_squares_l644_644129

theorem abs_difference_of_squares : 
  let a := 105 
  let b := 103
  abs (a^2 - b^2) = 416 := 
by 
  let a := 105
  let b := 103
  sorry

end abs_difference_of_squares_l644_644129


namespace find_a_l644_644289

def set_A : Set ℝ := {x | x^2 + x - 6 = 0}

def set_B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_a (a : ℝ) : set_A ∪ set_B a = set_A ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) := 
by
  sorry

end find_a_l644_644289


namespace problem_statement_l644_644976

variable {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonpos (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂

theorem problem_statement
  (h_even : is_even_function f)
  (h_increasing : is_increasing_on_nonpos f)
  (n : ℕ) (h_pos : 1 ≤ n) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) :=
begin
  sorry
end

end problem_statement_l644_644976


namespace pizza_slices_l644_644166

theorem pizza_slices (x y z total : ℕ) (h1 : y - x + z - x + x = total) (h_y : y = 15) (h_z : z = 16) (h_total : total = 24) :
  x = 7 :=
by
  rw [h_y, h_z, h_total] at h1
  simp at h1
  exact h1

end pizza_slices_l644_644166


namespace number_of_collections_l644_644356

def num_vowels : ℕ := 3
def num_consonants : ℕ := 7
def num_indistinguishable_t : ℕ := 2

def binom : ℕ → ℕ → ℕ
| n, 0 := 1
| 0, k := 0
| n, k := if k > n then 0 else binom (n - 1) (k - 1) + binom (n - 1) k

def case_1 : ℕ :=
  binom num_vowels 3 * binom (num_consonants - num_indistinguishable_t) 2

def case_2 : ℕ :=
  binom num_vowels 3 * binom (num_consonants - num_indistinguishable_t) 1

def case_3 : ℕ :=
  binom num_vowels 3 * binom (num_consonants - num_indistinguishable_t) 0

def total_combinations : ℕ :=
  case_1 + case_2 + case_3

theorem number_of_collections :
  total_combinations = 16 :=
by sorry

end number_of_collections_l644_644356


namespace relationship_withdrawn_leftover_l644_644211

-- Definitions based on the problem conditions
def pie_cost : ℝ := 6
def sandwich_cost : ℝ := 3
def book_cost : ℝ := 10
def book_discount : ℝ := 0.2 * book_cost
def book_price_with_discount : ℝ := book_cost - book_discount
def total_spent_before_tax : ℝ := pie_cost + sandwich_cost + book_price_with_discount
def sales_tax_rate : ℝ := 0.05
def sales_tax : ℝ := sales_tax_rate * total_spent_before_tax
def total_spent_with_tax : ℝ := total_spent_before_tax + sales_tax

-- Given amount withdrawn and amount left after shopping
variables (X Y : ℝ)

-- Theorem statement
theorem relationship_withdrawn_leftover :
  Y = X - total_spent_with_tax :=
sorry

end relationship_withdrawn_leftover_l644_644211


namespace total_selling_price_of_toys_l644_644184

/-
  Prove that the total selling price (TSP) for 18 toys,
  given that each toy costs Rs. 1100 and the man gains the cost price of 3 toys, is Rs. 23100.
-/
theorem total_selling_price_of_toys :
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  TSP = 23100 :=
by
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  sorry

end total_selling_price_of_toys_l644_644184


namespace hyperbola_focal_length_l644_644700

-- Definitions based on the conditions
def hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

def distance_focus_asymptote (c b d : ℝ) : Prop :=
  d = |(b * c) / (real.sqrt (1 + (b^2 / a^2)))|

-- The main theorem
theorem hyperbola_focal_length
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (a_eq : a = real.sqrt 5) (d_eq : d = 2)
  (h : hyperbola a b) (h' : distance_focus_asymptote c b d) :
  2 * c = 6 :=
sorry

end hyperbola_focal_length_l644_644700


namespace time_difference_l644_644617

-- Definitions
def time_chinese : ℕ := 5
def time_english : ℕ := 7

-- Statement to prove
theorem time_difference : time_english - time_chinese = 2 := by
  -- Proof goes here
  sorry

end time_difference_l644_644617


namespace area_maximized_when_angle_is_90_l644_644746

-- Definitions of given conditions
variable (a b : ℝ) (θ : ℝ)
def area_of_triangle : ℝ := (1 / 2) * a * b * Real.sin θ

-- Lean statement for the proof problem
theorem area_maximized_when_angle_is_90 :
  (∀ a b : ℝ, ∃ θ : ℝ, area_of_triangle a b θ = (1 / 2) * a * b * 1 ↔ θ = Real.pi / 2) :=
sorry

end area_maximized_when_angle_is_90_l644_644746


namespace solution_set_of_inequality_l644_644714

theorem solution_set_of_inequality
  (a : ℝ)
  (h : {1, a} ⊆ {a ^ 2 - 2 * a + 2, a - 1, 0}) :
  {x : ℝ | a * x^2 - 5 * x + a > 0} = set.Ioo (⊥) (1 / 2 : ℝ) ∪ set.Ioo (2 : ℝ) (⊤) :=
by
  sorry

end solution_set_of_inequality_l644_644714


namespace school_visitation_arrangements_l644_644537

theorem school_visitation_arrangements : 
    let week_days := 7
    let visit_days := (A: ℕ → ℕ) → ℕ
    let school_A_days := 2
    let school_B_days := 1
    let school_C_days := 1
    school_A_visits : ℕ := school_A_days < week_days
    school_B_visits : ℕ := school_B_days + school_A_days <= week_days
    school_C_visits : ℕ := school_C_days + school_A_days + school_B_days <= week_days
    let arrangements := (A 5 2) + (A 4 2) + (A 3 2) + (A 2 2)
  in
    arrangements = 40 := 
sorry

end school_visitation_arrangements_l644_644537


namespace value_of_logarithm_expression_l644_644456

theorem value_of_logarithm_expression : 10 ^ (Real.log 7 / Real.log 10) = 7 := by
  sorry

end value_of_logarithm_expression_l644_644456


namespace ProbabilityChile_l644_644360

open Classical

variable (P : Set String → ℝ)

-- Define the settings and conditions
def JenVisitsMadagascar := "Madagascar"
def JenVisitsChile := "Chile"

axiom ProbabilityMadagascar : P {JenVisitsMadagascar} = 0.5
axiom ProbabilityExclusiveOr : P {JenVisitsChile, JenVisitsMadagascar} - P {JenVisitsChile ∩ JenVisitsMadagascar} = 0.5

-- The required proof statement
theorem ProbabilityChile : P {JenVisitsChile} = 0.5 :=
by
  sorry

end ProbabilityChile_l644_644360


namespace average_remaining_checks_l644_644929

def travelers_checks_average 
  (x y : ℕ) 
  (total_value := 1800)
  (total_checks := 30) 
  (checks_50_pending : 6)
  (checks_100_pending : 6)
  (remaining_checks := 12) : ℕ :=
  if x + y = total_checks ∧ 50 * x + 100 * y = total_value ∧ checks_50_pending = x - 18 ∧ checks_100_pending = y then
    total_value - (50 * 18) // remaining_checks 
  else 
    0

theorem average_remaining_checks : travelers_checks_average 24 6 = 75 :=
by 
  simp [travelers_checks_average]; 
  sorry

end average_remaining_checks_l644_644929


namespace prime_quadruples_l644_644606

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_quadruples {p₁ p₂ p₃ p₄ : ℕ} (prime_p₁ : is_prime p₁) (prime_p₂ : is_prime p₂) (prime_p₃ : is_prime p₃) (prime_p₄ : is_prime p₄)
  (h1 : p₁ < p₂) (h2 : p₂ < p₃) (h3 : p₃ < p₄) (eq_condition : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  (p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
  (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
  (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29) :=
sorry

end prime_quadruples_l644_644606


namespace circumcircle_identity_l644_644756

open EuclideanGeometry
open Metric

noncomputable theory
variable {α : Type*} [MetricSpace α] [InnerProductSpace ℝ α] [hd2 : Fact (finrank ℝ α = 2)]
include hd2

theorem circumcircle_identity {A B C O P Q : α}
  (hO : O = circumcenter A B C)
  (hP : P ∈ (circle B O C) ∩ line_through A C)
  (hQ : Q ∈ (circle B O C) ∩ line_through A B) :
  circle A P Q = circle B O C := sorry

end circumcircle_identity_l644_644756


namespace scaling_transformation_correct_l644_644817

-- Define the scaling transformation for x and y.
def scaling_transform_x (x : ℝ) : ℝ := (1 / 2) * x
def scaling_transform_y (y : ℝ) : ℝ := (1 / 3) * y

-- Define the original point coordinates.
def original_point : ℝ × ℝ := (1, 2)

-- The proof statement that original_point after the scaling transformation results in the new coordinates.
theorem scaling_transformation_correct :
  scaling_transform_x original_point.1 = 1 / 2 ∧ scaling_transform_y original_point.2 = 2 / 3 :=
by
  split
  sorry
  sorry

end scaling_transformation_correct_l644_644817


namespace sum_of_two_squares_iff_double_sum_of_two_squares_l644_644397

theorem sum_of_two_squares_iff_double_sum_of_two_squares (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_two_squares_iff_double_sum_of_two_squares_l644_644397


namespace value_of_f_l644_644223

noncomputable def f : ℝ → ℝ := sorry

axiom even_f (x : ℝ) : f x = f (-x)
axiom periodic_f (x : ℝ) : f (x + π) = f x
axiom sin_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) : f x = sin x

theorem value_of_f : f (5 * π / 3) = sqrt 3 / 2 :=
by
  sorry

end value_of_f_l644_644223


namespace visitors_surveyed_l644_644112

-- Given definitions
def total_visitors : ℕ := 400
def visitors_not_enjoyed_nor_understood : ℕ := 100
def E := total_visitors / 2
def U := total_visitors / 2

-- Using condition that 3/4th visitors enjoyed and understood
def enjoys_and_understands := (3 * total_visitors) / 4

-- Assert the equivalence of total number of visitors calculation
theorem visitors_surveyed:
  total_visitors = enjoys_and_understands + visitors_not_enjoyed_nor_understood :=
by
  sorry

end visitors_surveyed_l644_644112


namespace meaningful_sqrt_range_l644_644738

theorem meaningful_sqrt_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end meaningful_sqrt_range_l644_644738


namespace speed_in_still_water_l644_644183

noncomputable def V_u (t : ℝ) : ℝ := 32 * (1 - 0.05 * t)
noncomputable def V_d (t : ℝ) : ℝ := 48 * (1 + 0.04 * t)
noncomputable def C (t : ℝ) : ℝ := 3 + 2 * sin(π * t / 6)

theorem speed_in_still_water (t : ℝ) : 
  (32 * (1 - 0.05 * t) + 48 * (1 + 0.04 * t)) / 2 = 40 + 0.16 * t :=
by
  sorry

end speed_in_still_water_l644_644183


namespace quadrilateral_area_correct_l644_644903

-- Defining the conditions
variables (A C B D : Point) (side_length : ℝ)
-- Assume that the points A, C, B, and D define the required plane intersections
-- Points A and C are diagonally opposite vertices and B, D are midpoints of edges
-- such that all the conditions hold for the specified problem.

noncomputable def quadrilateral_area := 
  let AC : ℝ := sqrt (2^2 + 2^2 + 2^2) in -- Diagonal AC length
  let BD : ℝ := sqrt (2^2 + 2^2) in       -- Diagonal BD length
  0.5 * AC * BD

theorem quadrilateral_area_correct (h₁ : AC = 2 * sqrt 3) (h₂ : BD = 2 * sqrt 2) :
  quadrilateral_area = 2 * sqrt 6 :=
by
  sorry

end quadrilateral_area_correct_l644_644903


namespace largest_three_digit_multiple_of_6_sum_15_l644_644859

-- Statement of the problem in Lean
theorem largest_three_digit_multiple_of_6_sum_15 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 6 = 0 ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 6 = 0 ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
by
  sorry -- proof not required

end largest_three_digit_multiple_of_6_sum_15_l644_644859


namespace rectangular_equation_of_C_length_of_chord_AB_l644_644811

def curve_C_polar_equation (rho theta : ℝ) : Prop :=
  (rho^2 * (Real.cos theta)^2) / 4 + (rho^2 * (Real.sin theta)^2) = 1

def curve_C_rectangular_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def line_l_parametric (x y t : ℝ) : Prop :=
  x = Real.sqrt 3 + t * Real.cos (Real.pi / 4) ∧ y = t * Real.sin (Real.pi / 4)

theorem rectangular_equation_of_C (rho theta : ℝ) (hC_polar : curve_C_polar_equation rho theta) :
  ∃ x y : ℝ, x = rho * Real.cos theta ∧ y = rho * Real.sin theta ∧ curve_C_rectangular_equation x y :=
by
  sorry

theorem length_of_chord_AB :
  let x := λ t : ℝ, Real.sqrt 3 + t * Real.cos (Real.pi / 4),
      y := λ t : ℝ, t * Real.sin (Real.pi / 4),
      C_rect := λ x y : ℝ, x^2 / 4 + y^2 = 1,
      quad_eq := λ t : ℝ, (5/2) * t^2 + Real.sqrt 6 * t - 1 in
  (∀ t₁ t₂ : ℝ, quad_eq t₁ = 0 ∧ quad_eq t₂ = 0 → |t₁ - t₂| = 8 / 5) ∧
  (∀ t, curve_C_rectangular_equation (x t) (y t)) :=
by
  sorry

end rectangular_equation_of_C_length_of_chord_AB_l644_644811


namespace cos_A_plus_B_l644_644671

theorem cos_A_plus_B (A B : ℝ) 
  (h1 : sin A + sin B = real.sqrt 2)
  (h2 : cos A + cos B = 1) : cos (A + B) = 0 := 
sorry

end cos_A_plus_B_l644_644671


namespace neither_necessary_nor_sufficient_condition_l644_644662

def red_balls := 5
def yellow_balls := 3
def white_balls := 2
def total_balls := red_balls + yellow_balls + white_balls

def event_A_occurs := ∃ (r : ℕ) (y : ℕ), (r ≤ red_balls) ∧ (y ≤ yellow_balls) ∧ (r = 1) ∧ (y = 1)
def event_B_occurs := ∃ (x y : ℕ), (x ≤ total_balls) ∧ (y ≤ total_balls) ∧ (x ≠ y)

theorem neither_necessary_nor_sufficient_condition :
  ¬(¬event_A_occurs → ¬event_B_occurs) ∧ ¬(¬event_B_occurs → ¬event_A_occurs) := 
sorry

end neither_necessary_nor_sufficient_condition_l644_644662


namespace bus_probability_l644_644171

-- Conditions given in the problem
def bus_arrival : set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 60 ∧ 0 ≤ p.2 ∧ p.2 ≤ 60 ∧ p.1 ≤ p.2 ∧ p.2 ≤ p.1 + 15}

def total_area := 60 * 60

theorem bus_probability :
  ∑ in bus_arrival, 1 / total_area = 137 / 1200 := sorry

end bus_probability_l644_644171


namespace sum_of_squares_iff_double_l644_644400

theorem sum_of_squares_iff_double (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_squares_iff_double_l644_644400


namespace annulus_area_sufficient_linear_element_l644_644162

theorem annulus_area_sufficient_linear_element (R r : ℝ) (hR : R > 0) (hr : r > 0) (hrR : r < R):
  (∃ d : ℝ, d = R - r ∨ d = R + r) → ∃ A : ℝ, A = π * (R ^ 2 - r ^ 2) :=
by
  sorry

end annulus_area_sufficient_linear_element_l644_644162


namespace arithmetic_sequence_a5_value_l644_644277

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_cond : (a 5)^2 - a 3 - a 7 = 0) 
  : a 5 = 2 := 
sorry

end arithmetic_sequence_a5_value_l644_644277


namespace five_tuples_count_l644_644608

theorem five_tuples_count :
  (∃ (x : Fin 5 → ℕ), 
    (∀ (i : Fin 5), x i ≥ i.val + 1) ∧ 
    (∑ i, x i = 25)) →
  Finset.univ.prod (λ i : Fin 5, (x i) - i.val - 1) = 1001 :=
by
  sorry

end five_tuples_count_l644_644608


namespace decreasing_interval_of_log_square_l644_644826

noncomputable def f (x : ℝ) := log (x^2)

theorem decreasing_interval_of_log_square :
  ∃ I : set ℝ, I = set.Ioo (⊥:ℝ) 0 ∧   -- (-∞, 0)
  monotonically_decreasing_on f I :=
sorry

end decreasing_interval_of_log_square_l644_644826


namespace find_standard_equation_of_ellipse_find_line_equation_that_intersects_ellipse_l644_644578

-- Given conditions in the problem
variables {a b : ℝ}
variable {C : set (ℝ × ℝ)} 
variable {l : Set (ℝ × ℝ)}

-- Condition: Ellipse equation
def ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.fst ^ 2) / (a ^ 2) + (p.snd ^ 2) / (b ^ 2) = 1

-- Condition: a > b > 0
axiom h1 : a > b ∧ b > 0

-- Condition: Two foci and one of its vertices form an isosceles right triangle
axiom h2 : (∀ {c : ℝ}, c^2 = a^2 - b^2 → b = c)

-- Condition: The chord formed by the intersection of the line x + y = 0 and the circle with center at the right vertex of the ellipse C and radius 2b has a length of 2√3.
def chord_length : Prop :=
  ∀ {chord_len : ℝ}, chord_len = 2 * real.sqrt 3 → 
    (∀ p : ℝ × ℝ, p.fst = a → (chord_len = 
      (real.sqrt ((p.fst - a) ^ 2 + (p.snd) ^ 2)) * 2 * b))

noncomputable def ellipse_standard_equation : Prop :=
  ellipse (real.sqrt 2) 1 (2,0)

noncomputable def line_equation : Prop :=
  ∃ k : ℝ, k = real.sqrt 2 ∧
    (∀ p : ℝ × ℝ, p.fst = 1 ∧ p.snd = k * (p.fst - 1) →
      l = {p | p.snd = ± real.sqrt 2 * (p.fst - 1)})

-- Proof problems
theorem find_standard_equation_of_ellipse : ellipse_standard_equation :=
sorry

theorem find_line_equation_that_intersects_ellipse : line_equation :=
sorry

end find_standard_equation_of_ellipse_find_line_equation_that_intersects_ellipse_l644_644578


namespace time_after_2021_hours_l644_644124

-- Definition of starting time and day
def start_time : Nat := 20 * 60 + 21  -- converting 20:21 to minutes
def hours_per_day : Nat := 24
def minutes_per_hour : Nat := 60
def days_per_week : Nat := 7

-- Define the main statement
theorem time_after_2021_hours :
  let total_minutes := 2021 * minutes_per_hour
  let total_days := total_minutes / (hours_per_day * minutes_per_hour)
  let remaining_minutes := total_minutes % (hours_per_day * minutes_per_hour)
  let final_minutes := start_time + remaining_minutes
  let final_day := (total_days + 1) % days_per_week -- start on Monday (0), hence +1 for Tuesday
  final_minutes / minutes_per_hour = 1 ∧ final_minutes % minutes_per_hour = 21 ∧ final_day = 2 :=
by
  sorry

end time_after_2021_hours_l644_644124


namespace pu_and_guan_equal_lengths_l644_644417

-- Define the growth functions of Pu and Guan
def pu_growth (n : ℝ) : ℝ := 3 * (1 - 1 / 2^n) / (1 - 1 / 2)
def guan_growth (n : ℝ) : ℝ := (2^n - 1) / (2 - 1)

-- Prove that Pu and Guan will have lengths equal after approximately 2.6 days
theorem pu_and_guan_equal_lengths (n : ℝ) (h : guan_growth n = pu_growth n) : n = 2.6 :=
by
  -- Use logarithm properties to show that 2^n = 6
  have eq₁ : guan_growth n = 2^n - 1 := sorry,
  have eq₂ : pu_growth n = 6 := sorry,
  -- Proving n = log_2 (6)
  have eq₃ : n = log 6 / log 2 := sorry,
  show n = 1 + log 3 / log 2 from by sorry

#check pu_and_guan_equal_lengths

end pu_and_guan_equal_lengths_l644_644417


namespace problem_l644_644656

open Set

variable {α : Type*}
variables {A B : Finset (Finset α)} {S : Finset α}
variables {m n : ℕ}
variables (A B S) in {A_1 A_2 ... A_n : Finset α} (B_1 B_2 ... B_n : Finset α)

def single_representation (A : Finset (Finset α)) (X : Finset α) : Prop :=
  ∀ i, ∃ x ∈ (Finset.to_list A).nth i, x ∈ X

theorem problem
  (hA : ∀ i, (Finset.to_list A).nth i ≠ none ∧ |(Finset.to_list A).nth i| = m)
  (hB : ∀ i, (Finset.to_list B).nth i ≠ none ∧ |(Finset.to_list B).nth i| = m)
  (hS : S = Finset.univ ⋃₀ {A | (A ∈ (Finset.to_list A).nth i ∨ A ∈ (Finset.to_list B).nth i)})
  : ∃ (C : Finset (Finset α)), (∀ i, single_representation (Finset.to_list A) (C.nth i) ∧ single_representation (Finset.to_list B) (C.nth i)) ∧ (Finset.univ ⋃₀ (Finset.to_list C)) = S ∧ ∀ i j, i ≠ j → (C.nth i ∩ C.nth j = ∅) :=
sorry

end problem_l644_644656


namespace necessary_and_sufficient_condition_l644_644272

theorem necessary_and_sufficient_condition (x y : ℝ) (h1 : 0 ≤ x ∧ x ≤ π / 2) (h2 : 0 ≤ y ∧ y ≤ π / 2)
  (h3 : sin x ^ 6 + 3 * sin x ^ 2 * cos y ^ 2 + cos y ^ 6 = 1) : x = y := by
  sorry

end necessary_and_sufficient_condition_l644_644272


namespace employees_reshuffle_l644_644343

theorem employees_reshuffle 
    (total_employees : ℕ)
    (current_senior_percentage current_junior_percentage current_engineer_percentage current_marketing_percentage
     desired_senior_percentage desired_junior_percentage desired_engineer_percentage desired_marketing_percentage : ℝ)
    (h_current_sum : current_senior_percentage + current_junior_percentage + current_engineer_percentage + current_marketing_percentage = 1)
    (h_desired_sum : desired_senior_percentage + desired_junior_percentage + desired_engineer_percentage + desired_marketing_percentage = 1)
    (total_employees_positive : total_employees > 0) :
    let current_senior := current_senior_percentage * total_employees,
        current_junior := current_junior_percentage * total_employees,
        current_engineer := current_engineer_percentage * total_employees,
        current_marketing := current_marketing_percentage * total_employees,
        desired_senior := desired_senior_percentage * total_employees,
        desired_junior := desired_junior_percentage * total_employees,
        desired_engineer := desired_engineer_percentage * total_employees,
        desired_marketing := desired_marketing_percentage * total_employees
    in current_senior - desired_senior = 500 ∧
       current_junior - desired_junior = 1000 ∧
       desired_engineer - current_engineer = 500 ∧
       desired_marketing - current_marketing = 1000 := 
begin
  sorry
end

end employees_reshuffle_l644_644343


namespace smallest_positive_integer_arithmetic_mean_squared_l644_644636

theorem smallest_positive_integer_arithmetic_mean_squared (n : ℕ) (h₁ : n > 1) 
(h₂ : ∃ k : ℕ, (1 + 2^2 + 3^2 + ... + n^2) / n = k^2) : 
  n = 337 := sorry

end smallest_positive_integer_arithmetic_mean_squared_l644_644636


namespace remainder_modulo_l644_644778

theorem remainder_modulo (y : ℕ) (hy : 5 * y ≡ 1 [MOD 17]) : (7 + y) % 17 = 14 :=
sorry

end remainder_modulo_l644_644778


namespace value_of_expression_l644_644799

variable (a b c : ℝ)

theorem value_of_expression (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1)
                            (h2 : abc = 1)
                            (h3 : a^2 + b^2 + c^2 - ((1 / (a^2)) + (1 / (b^2)) + (1 / (c^2))) = 8 * (a + b + c) - 8 * (ab + bc + ca)) :
                            (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) = -3/2 :=
by
  sorry

end value_of_expression_l644_644799


namespace compare_apothems_l644_644188

open Real

-- Define the rectangle and its properties
def rectangle (w : ℝ) : Prop :=
  let length := 2 * w
  let area := 2 * w * w
  let perimeter := 2 * (w + length)
  area = 2 * perimeter

-- Define the hexagon and its properties
def hexagon (s : ℝ) : Prop :=
  let area := (3 * sqrt 3 / 2) * s^2
  let perimeter := 6 * s
  area = 3 * perimeter

-- Define the apothems
def rect_apothem (w : ℝ) := w / 2
def hex_apothem (s : ℝ) := (sqrt 3 / 2) * s

-- The main theorem to prove
theorem compare_apothems (w s : ℝ) (hw : w ≠ 0) (hs : s ≠ 0) :
  rectangle w → hexagon s → (rect_apothem w = 1 / 2 * hex_apothem s) :=
by
  sorry

end compare_apothems_l644_644188


namespace complex_identity_l644_644701

noncomputable def z := sorry

theorem complex_identity (z : ℂ) (h1 : z^3 + 1 = 0) (h2 : z ≠ -1) :
  (z / (z - 1)) ^ 2018 + (1 / (z - 1)) ^ 2018 = -1 :=
sorry

end complex_identity_l644_644701


namespace repeating_decimal_fraction_correct_l644_644244

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := 3.36363636 in  -- representing 3.\overline{36}
  x

theorem repeating_decimal_fraction_correct :
  repeating_decimal_to_fraction = 10 / 3 :=
sorry

end repeating_decimal_fraction_correct_l644_644244


namespace probability_even_sums_l644_644080

open Finset
open Classical

namespace TileGame

def tiles : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def odd_tiles : Finset ℕ := tiles.filter is_odd
def even_tiles : Finset ℕ := tiles.filter is_even

def even_sum_sets : Finset (Finset ℕ) :=
  (powersetLen 3 tiles).filter (λ s, (s.sum id).is_even)

def total_ways_to_distribute (players : ℕ) : ℕ :=
  powersetLen (players * 3) tiles.card

theorem probability_even_sums :
  (70 / 120) ^ 3 = 343 / 1728 := by
  sorry

end TileGame

end probability_even_sums_l644_644080


namespace minimum_dwarfs_l644_644053

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l644_644053


namespace zumblian_word_count_l644_644790

theorem zumblian_word_count :
  let alphabet_size : ℕ := 7 in
  let words_of_length (n : ℕ) : ℕ := alphabet_size ^ n in
  words_of_length 1 + words_of_length 2 + words_of_length 3 + words_of_length 4 = 2800 :=
by
  let alphabet_size : ℕ := 7
  let words_of_length (n : ℕ) : ℕ := alphabet_size ^ n
  have h1: words_of_length 1 = 7 := rfl
  have h2: words_of_length 2 = 49 := rfl
  have h3: words_of_length 3 = 343 := rfl
  have h4: words_of_length 4 = 2401 := rfl
  calc
    words_of_length 1 + words_of_length 2 + words_of_length 3 + words_of_length 4
    = 7 + 49 + 343 + 2401 : by rw [h1, h2, h3, h4]
    ... = 2800 : rfl

end zumblian_word_count_l644_644790


namespace ratio_hema_rahul_l644_644003

-- Let Ba, Ravi, Hema, and Rahul be humans with certain age constraints
variables (R Ra H Rh : ℕ) 

-- Raj is 3 years older than Ravi
axiom Raj_older (R Ra: ℕ) : Ra = R + 3

-- Hema is 2 years younger than Ravi
axiom Hema_younger (R H: ℕ) : H = R - 2

-- Raj is 3 times as old as Rahul
axiom Raj_times_rahul (Ra Rh: ℕ) : 3 * Rh = Ra

-- When Raj is 20 years old, Raj's age is 33.33333333333333% more than Hema's age
axiom Raj_20_hem (Ra H: ℕ) : Ra = 20 → 20 = (4 * H) / 3 

theorem ratio_hema_rahul (R Ra H Rh: ℕ) 
  (Raj_time: Raj_older R Ra) 
  (Hema_Age: Hema_younger R H) 
  (Raj_time2: Raj_times_rahul Ra Rh)
  (Raj_age: Ra = 20) :
  (H : ℕ )* ((4:ℕ)*H = 4/3 * (15:ℕ)) :=
  begin
    sorry
  end

end ratio_hema_rahul_l644_644003


namespace find_y_l644_644732

theorem find_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 := by
  sorry

end find_y_l644_644732


namespace label_sum_l644_644853

theorem label_sum (n : ℕ) : 
  (∃ S : ℕ → ℕ, S 1 = 2 ∧ (∀ k, k > 1 → (S (k + 1) = 2 * S k)) ∧ S n = 2 * 3 ^ (n - 1)) := 
sorry

end label_sum_l644_644853


namespace smallest_value_other_integer_l644_644095

noncomputable def smallest_possible_value_b : ℕ :=
  by sorry

theorem smallest_value_other_integer (x : ℕ) (h_pos : x > 0) (b : ℕ) 
  (h_gcd : Nat.gcd 36 b = x + 3) (h_lcm : Nat.lcm 36 b = x * (x + 3)) :
  b = 108 :=
  by sorry

end smallest_value_other_integer_l644_644095


namespace fn_less_than_nsquared_div_4_l644_644645

theorem fn_less_than_nsquared_div_4
  (n : ℕ) (hn : 4 ≤ n)
  (r : ℝ) (hr : 0 < r)
  (a : ℕ → ℝ) 
  (ha : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → a n > a (n-1) ∧ ... ∧ a 2 > a 1 ∧ a 1 > 0)
  (fn : ℕ → ℕ → ℕ → ℝ → ℕ) 
  (hfn : ∀ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n → fn i j k r = if (a j - a i) / (a k - a j) = r then 1 else 0):
  (∑ i (H : 1 ≤ i) 
      (j (H : i < j) 
      (k (H : j < k) 
      (H : k ≤ n)), fn i j k r) < (n^2)/4 := 
sorry

end fn_less_than_nsquared_div_4_l644_644645


namespace probability_of_p_satisfying_eq_l644_644327

theorem probability_of_p_satisfying_eq :
  let S := { p : ℕ | 1 ≤ p ∧ p ≤ 20 ∧ ∃ q : ℤ, 2*p*q - 6*p - 4*q = 8 } in
  (S.card : ℚ) / 20 = 9/20 := by
-- sorry is used to indicate that the proof is omitted.
sorry

end probability_of_p_satisfying_eq_l644_644327


namespace t_shirt_cost_l644_644361

theorem t_shirt_cost (T : ℕ) 
  (h1 : 3 * T + 50 = 110) : T = 20 := 
by
  sorry

end t_shirt_cost_l644_644361


namespace parameterized_to_ordinary_equation_l644_644255

theorem parameterized_to_ordinary_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = sqrt t) 
  (h2 : y = 2 * sqrt (1 - t)) 
  (h3 : 0 ≤ t ∧ t ≤ 1) : 
  x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2 :=
by {
  sorry
}

end parameterized_to_ordinary_equation_l644_644255


namespace sphere_surface_area_from_box_l644_644824

/--
Given a rectangular box with length = 2, width = 2, and height = 1,
prove that if all vertices of the rectangular box lie on the surface of a sphere,
then the surface area of the sphere is 9π.
--/
theorem sphere_surface_area_from_box :
  let length := 2
  let width := 2
  let height := 1
  ∃ (r : ℝ), ∀ (d := Real.sqrt (length^2 + width^2 + height^2)),
  r = d / 2 → 4 * Real.pi * r^2 = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_from_box_l644_644824


namespace continuity_at_point_l644_644119

theorem continuity_at_point : 
  (∀ x ≠ 1, (∃ h1 : x ≠ 1, (∃ h2 : x ≠ 1, (((x + 1) / (x^2 + x + 1)) : ℝ)))) ↔ 
  (∀ L, L = 2/3 → 
    tendsto (λ x : ℝ, (x^2 - 1)/(x^3 - 1)) (nhds 1) (nhds L)) :=
begin
  sorry
end

end continuity_at_point_l644_644119


namespace max_and_min_values_of_f_l644_644669

noncomputable def f (x : ℝ) : ℝ := 9^x - 3^(x + 1) - 1

theorem max_and_min_values_of_f :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f(x) ≤ 647) ∧ (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f(x) ≥ -13 / 4) :=
begin
  sorry -- Proof omitted
end

end max_and_min_values_of_f_l644_644669


namespace sandy_books_l644_644801

theorem sandy_books (x : ℕ) (h1 : 1380 + 900 = 2280)
  (h2 : ∀ total_books, total_books = x + 55 → 2280 / total_books = 19) : x = 65 :=
by
  have h3 : x + 55 = 2280 / 19, from sorry
  have h4 : x + 55 = 120, from sorry
  have h5 : x = 65, from sorry
  exact h5

end sandy_books_l644_644801


namespace express_train_speed_ratio_l644_644938

noncomputable def speed_ratio (c h : ℝ) (x : ℝ) : Prop :=
  let t1 := h / ((1 + x) * c)
  let t2 := h / ((x - 1) * c)
  x = t2 / t1

theorem express_train_speed_ratio 
  (c h : ℝ) (x : ℝ) 
  (hc : c > 0) (hh : h > 0) (hx : x > 1) : 
  speed_ratio c h (1 + Real.sqrt 2) := 
by
  sorry

end express_train_speed_ratio_l644_644938


namespace determine_f_16_l644_644092

theorem determine_f_16 (a : ℝ) (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  (∀ x, a ^ (x - 4) + 1 = 2) →
  f 4 = 2 →
  f 16 = 4 :=
by
  sorry

end determine_f_16_l644_644092


namespace exists_integer_divisible_by_18_and_sqrt_between_24_7_and_25_l644_644996

theorem exists_integer_divisible_by_18_and_sqrt_between_24_7_and_25 : 
  ∃ n : ℕ, (18 ∣ n) ∧ 24.7 < real.sqrt n ∧ real.sqrt n < 25 ∧ n = 612 :=
by
  sorry

end exists_integer_divisible_by_18_and_sqrt_between_24_7_and_25_l644_644996


namespace area_ratio_convex_pentagon_l644_644968

noncomputable def centroid (A B C D : ℝ × ℝ) : ℝ × ℝ :=
((A.1 + B.1 + C.1 + D.1) / 4, (A.2 + B.2 + C.2 + D.2) / 4)

theorem area_ratio_convex_pentagon
  (A B C D E : ℝ × ℝ)
  (convex_ABCDE : convex_hull ℝ ({A, B, C, D, E} : set (ℝ × ℝ)))
  : let P_A := centroid B C D E
        P_B := centroid A C D E
        P_C := centroid A B D E
        P_D := centroid A B C E
        P_E := centroid A B C D 
    in area ({P_A, P_B, P_C, P_D, P_E} : set (ℝ × ℝ)) / area ({A, B, C, D, E} : set (ℝ × ℝ)) = 1/16 :=
by
  sorry

end area_ratio_convex_pentagon_l644_644968


namespace find_x0_l644_644730

open Real

noncomputable def f : ℝ → ℝ := λ x, x^3

theorem find_x0 (x0 : ℝ) (h : (deriv f x0) = 3) : x0 = 1 ∨ x0 = -1 :=
by
  sorry

end find_x0_l644_644730


namespace integer_solutions_of_cubic_equation_l644_644999

theorem integer_solutions_of_cubic_equation :
  ∀ (n m : ℤ),
    n ^ 6 + 3 * n ^ 5 + 3 * n ^ 4 + 2 * n ^ 3 + 3 * n ^ 2 + 3 * n + 1 = m ^ 3 ↔
    (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
by
  intro n m
  apply Iff.intro
  { intro h
    sorry }
  { intro h
    sorry }

end integer_solutions_of_cubic_equation_l644_644999


namespace shopkeeper_total_cards_l644_644566

-- Definition of the number of cards in a complete deck
def cards_in_deck : Nat := 52

-- Definition of the number of complete decks the shopkeeper has
def number_of_decks : Nat := 3

-- Definition of the additional cards the shopkeeper has
def additional_cards : Nat := 4

-- The total number of cards the shopkeeper should have
def total_cards : Nat := number_of_decks * cards_in_deck + additional_cards

-- Theorem statement to prove the total number of cards is 160
theorem shopkeeper_total_cards : total_cards = 160 := by
  sorry

end shopkeeper_total_cards_l644_644566


namespace abs_diff_squares_l644_644127

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end abs_diff_squares_l644_644127


namespace option_D_not_necessarily_true_l644_644727

variable {a b c : ℝ}

theorem option_D_not_necessarily_true 
  (h1 : c < b)
  (h2 : b < a)
  (h3 : a * c < 0) : ¬((c * b^2 < a * b^2) ↔ (b ≠ 0 ∨ b = 0 ∧ (c * b^2 < a * b^2))) := 
sorry

end option_D_not_necessarily_true_l644_644727


namespace game_is_not_fair_l644_644362

noncomputable def expected_winnings : ℚ := 
  let p_1 := 1 / 8
  let p_2 := 7 / 8
  let gain_case_1 := 2
  let loss_case_2 := -1 / 7
  (p_1 * gain_case_1) + (p_2 * loss_case_2)

theorem game_is_not_fair : expected_winnings = 1 / 8 :=
sorry

end game_is_not_fair_l644_644362


namespace bob_always_wins_l644_644846

theorem bob_always_wins :
  ∀ a b : ℕ, a = 47 → b = 2016 →
  ∃ w : string, w = "Bob wins" := 
by
  assume a b,
  assume ha : a = 47,
  assume hb : b = 2016,
  have h_wins : string := "Bob wins",
  existsi h_wins,
  exact h_wins

end bob_always_wins_l644_644846


namespace factorable_polynomial_l644_644286

open Nat

-- Definitions for the prime numbers p and q, and the natural number n ≥ 3
variables {p q : ℕ} [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)]
variable {n : ℕ} 

-- Condition that n is at least 3
axiom h_n : n ≥ 3

-- Prove that if f(x) can be factored, then a must be either -1 - pq or 1 + (-1)^n pq
theorem factorable_polynomial {a : ℤ}
  (h_distinct_primes : p ≠ q)
  (h_factorable : ∃ f g : polynomial ℤ, f.degree ≥ 1 ∧ g.degree ≥ 1 ∧ f * g = polynomial.C x^n + polynomial.C a * x^(n - 1) + polynomial.C (↑p * ↑q)) :
  a = -1 - ↑p * ↑q ∨ a = 1 + (-1)^n * p * q :=
sorry

end factorable_polynomial_l644_644286


namespace square_presses_exceed_1000_l644_644536

theorem square_presses_exceed_1000:
  ∃ n : ℕ, (n = 3) ∧ (3 ^ (2^n) > 1000) :=
by
  sorry

end square_presses_exceed_1000_l644_644536


namespace num_valid_n_l644_644325

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_valid_n (n : Nat) (h_multiple_of_7 : n % 7 = 0)
  (h_lcm_gcd : Nat.lcm (factorial 7) n = 7 * Nat.gcd (factorial 12) n) :
  Nat := sorry

example : num_valid_n 36 := sorry

end num_valid_n_l644_644325


namespace area_of_triangle_l644_644469

noncomputable def point := ℝ × ℝ

def line1 := {p : point | p.snd = (1/3)*p.fst + 2}
def line2 := {p : point | p.snd = 3*p.fst - 6}
def line3 := {p : point | p.fst + p.snd = 12}

def A : point := (3, 3)
def B : point := (4.5, 7.5)
def C : point := (7.5, 4.5)

def triangle_area (A B C : point) : ℝ :=
  (1/2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem area_of_triangle : triangle_area A B C = 9 := by
  sorry

end area_of_triangle_l644_644469


namespace minimum_dwarfs_l644_644052

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l644_644052


namespace number_of_sixes_l644_644893

theorem number_of_sixes (total_runs : ℕ) (boundaries : ℕ) (running_percentage : ℝ) :
  total_runs = 120 →
  boundaries = 5 →
  running_percentage = 58.333333333333336 →
  let running_runs := total_runs * (running_percentage / 100)
  let boundary_runs := boundaries * 4
  let six_runs := total_runs - (running_runs + boundary_runs)
  let num_sixes := six_runs / 6
  num_sixes = 5 := 
by
  intros h_total h_boundaries h_percentage
  rw [h_total, h_boundaries, h_percentage]
  let running_runs := 120 * (58.333333333333336 / 100)
  let boundary_runs := 5 * 4
  let six_runs := 120 - (running_runs + boundary_runs)
  let num_sixes := six_runs / 6
  have h_running_runs : running_runs = 70 := by sorry
  have h_boundary_runs : boundary_runs = 20 := by sorry
  have h_six_runs : six_runs = 30 := by sorry
  rw [h_running_runs, h_boundary_runs, h_six_runs]
  have h_num_sixes : num_sixes = 5 := by sorry
  exact h_num_sixes

end number_of_sixes_l644_644893


namespace ellipse_foci_distance_l644_644117

noncomputable def distance_between_foci (p1 p2 p3 : ℝ × ℝ) : ℝ :=
let x1 := p1.1, y1 := p1.2,
    x2 := p2.1, y2 := p2.2,
    x3 := p3.1, y3 := p3.2 in
let hAxis := |(x2 - x1) / 2|,
    vAxis := |(y3 - y1) / 2| in
2 * real.sqrt (hAxis ^ 2 - vAxis ^ 2)

theorem ellipse_foci_distance :
  distance_between_foci (10, -3) (25, -3) (15, 7) = 11.18 :=
sorry

end ellipse_foci_distance_l644_644117


namespace number_of_candidates_is_9_l644_644921

-- Defining the problem
def num_ways_to_select_president_and_vp (n : ℕ) : ℕ :=
  n * (n - 1)

-- Main theorem statement
theorem number_of_candidates_is_9 (n : ℕ) (h : num_ways_to_select_president_and_vp n = 72) : n = 9 :=
by
  sorry

end number_of_candidates_is_9_l644_644921


namespace coefficient_x3y3_in_expr_l644_644857

theorem coefficient_x3y3_in_expr : 
  ∀ (x y z : ℝ), 
  (coeff (expand ((x + y)^6 * (z + 2/z)^8)) x^3 * y^3 = 22400) :=
sorry

end coefficient_x3y3_in_expr_l644_644857


namespace determinant_of_projection_matrix_l644_644772

-- Define the projection matrix Q
noncomputable def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  1 / (3 ^ 2 + 2 ^ 2 + (-6 : ℝ) ^ 2) •
    ![
      ![9, 6, -18],
      ![6, 4, -12],
      ![-18, -12, 36]
    ]

-- State the theorem
theorem determinant_of_projection_matrix : det Q = 0 :=
  sorry

end determinant_of_projection_matrix_l644_644772


namespace students_with_both_l644_644752

theorem students_with_both (n : ℕ) 
  (h_problem_set : fraction students have a problem set = 2 / 3)
  (h_calculator : fraction students have a calculator = 4 / 5)
  (common_ratio_condition : common ratio = (students who brought a calculator but did not bring a problem set) / (students who did not bring both) = (students who did not have a problem set) / (students who didn't bring a calculator)) :
  fraction of students who have both a problem set and a calculator = 8 / 15 := 
begin
  -- sorry
end

end students_with_both_l644_644752


namespace part_i_part_ii_part_iii_l644_644316

-- Define the sequences and initial values
def a (n : ℕ) : ℕ := if n = 1 then 1 else if n = 2 then 3 else sorry
def b (n : ℕ) : ℕ := if n = 1 then 4 else if n = 2 then 9 else sorry

-- Define the sums of the first n terms
def S (n : ℕ) : ℕ := ∑ i in finset.range n, a i

-- Define the T sequence
def T (n : ℕ) : ℕ := ∑ i in finset.range n, (-1) ^ a i * b i 

-- Proving point (Ⅰ)
theorem part_i : a 2 = 3 ∧ b 2 = 9 :=
by sorry

-- Proving point (Ⅱ)
theorem part_ii : (∀ n, a n = (n * (n + 1)) / 2) ∧ (∀ n, b n = (n + 1) ^ 2) :=
by sorry

-- Proving point (Ⅲ)
theorem part_iii : ∀ n ≥ 3, |T n| < 2 * n ^ 2 :=
by sorry

end part_i_part_ii_part_iii_l644_644316


namespace melanie_picked_plums_l644_644787

variable (picked_plums : ℕ)
variable (given_plums : ℕ := 3)
variable (total_plums : ℕ := 10)

theorem melanie_picked_plums :
  picked_plums + given_plums = total_plums → picked_plums = 7 := by
  sorry

end melanie_picked_plums_l644_644787


namespace horse_food_per_day_l644_644209

theorem horse_food_per_day 
  (ratio_sheep_horses: ℕ × ℕ)
  (total_horse_food: ℕ)
  (num_sheep: ℕ)
  (ratio_sheep_horses = (2, 7))
  (total_horse_food = 12880)
  (num_sheep = 16)
  : (total_horse_food / (7 * num_sheep / 2) = 230) :=
by 
  sorry

end horse_food_per_day_l644_644209


namespace albert_money_l644_644199

-- Define the costs
def cost_paintbrush : ℝ := 1.50
def cost_set_of_paints : ℝ := 4.35
def cost_wooden_easel : ℝ := 12.65
def additional_amount_needed : ℝ := 12

-- Calculate the total cost of the items
def total_cost : ℝ :=
  cost_paintbrush + cost_set_of_paints + cost_wooden_easel

-- The amount of money Albert already has
def amount_albert_has : ℝ :=
  total_cost - additional_amount_needed

-- Theorem to prove the main question
theorem albert_money : amount_albert_has = 6.50 :=
by
  unfold amount_albert_has total_cost cost_paintbrush cost_set_of_paints cost_wooden_easel additional_amount_needed
  norm_num
  sorry

end albert_money_l644_644199


namespace number_of_rheas_l644_644959

theorem number_of_rheas 
  (num_wombats : ℕ) 
  (wombat_claws : ℕ) 
  (total_claws : ℕ) 
  (each_rhea_claws : ℕ)
  (H1 : num_wombats = 9)
  (H2 : wombat_claws = 4)
  (H3 : total_claws = 39)
  (H4 : each_rhea_claws = 1) 
  : ℕ :=
by
  have wombat_total_claws : ℕ := num_wombats * wombat_claws
  have claws_by_rheas : ℕ := total_claws - wombat_total_claws
  have num_rheas : ℕ := claws_by_rheas / each_rhea_claws
  exact num_rheas

-- The final Lean statement:
#check number_of_rheas 9 4 39 1 9 4 39 1 -- this should evaluate to 3

end number_of_rheas_l644_644959


namespace mean_score_seniors_is_180_l644_644468

variable (Students_Seniors NonSeniors_Seniors : ℕ)
variable (MeanStudents : ℚ := 120)

variable h1 : Students_Seniors + NonSeniors_Seniors = 200
variable h2 : NonSeniors_Seniors = 2 * Students_Seniors
variable (MeanSeniors MeanNonSeniors : ℚ)
variable h3 : MeanSeniors = 2 * MeanNonSeniors
variable h4 : (200 * MeanStudents = Students_Seniors * MeanSeniors + NonSeniors_Seniors * MeanNonSeniors)

theorem mean_score_seniors_is_180 : MeanSeniors = 180 := by
  sorry

end mean_score_seniors_is_180_l644_644468


namespace jogger_ahead_distance_l644_644907

def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def train_length_m : ℝ := 120
def passing_time_s : ℝ := 31

theorem jogger_ahead_distance :
  let V_rel := (train_speed_kmh - jogger_speed_kmh) * (1000 / 3600)
  let Distance_train := V_rel * passing_time_s 
  Distance_train = 310 → 
  Distance_train = 190 + train_length_m :=
by
  intros
  sorry

end jogger_ahead_distance_l644_644907


namespace area_of_focus_triangle_l644_644291

-- Definitions based on conditions
def ellipse : set (ℝ × ℝ) :=
  { p | let (x, y) := p in (x^2 / 100) + (y^2 / 64) = 1 }

def foci : ((ℝ × ℝ) × (ℝ × ℝ)) :=
  let a := 10
  let b := 8
  let c := real.sqrt (a^2 - b^2)
  ((-c, 0), (c, 0))  -- Assuming the foci are at (-c, 0) and (c, 0)

noncomputable def dist (p1 p2: ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Given an element is on the ellipse and the angle condition
def on_ellipse (p: ℝ × ℝ) : Prop :=
  p ∈ ellipse

def angle_condition (P F1 F2: ℝ × ℝ) : Prop :=
  ( ∃ θ: ℝ, θ = real.pi / 3 ∧ (∠F1PF2 := θ) )

-- This is the proof statement we need to show
theorem area_of_focus_triangle (P F1 F2: ℝ × ℝ)
  (hP: on_ellipse P) (hangle: angle_condition P F1 F2) :
  area (triangle F1 P F2) = 64 * real.sqrt 3 / 3 :=
sorry

end area_of_focus_triangle_l644_644291


namespace problem_statement_l644_644726

theorem problem_statement (P : ℕ → ℝ) (A B : ℕ) (AB : ℕ → ℕ)
  (hP_AB : P (AB A B) = 1 / 6)
  (hP_not_A : P (λ x, ¬A x) = 2 / 3)
  (hP_B : P B = 1 / 2) :
  (P A = 1 / 3) ∧
  (∃ x, AB A B x ≠ 0) ∧
  (∀ x, P (AB A B x) = P A * P B) ∧
  (∀ x, P (λ y, ¬A y) B x = P (λ x, ¬A x) * P B) :=
begin
  sorry
end

end problem_statement_l644_644726


namespace temperature_on_friday_l644_644881

theorem temperature_on_friday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 42) : 
  F = 34 :=
by
  sorry

end temperature_on_friday_l644_644881


namespace number_of_posts_l644_644533

theorem number_of_posts (length width post_interval : ℕ) (h_length : length = 6) (h_width : width = 8) (h_post_interval : post_interval = 2) : 
  (4 + 2 * ((length / post_interval) - 1) + 2 * ((width / post_interval) - 1)) = 14 := 
by
  -- condition definitions
  have h_l_pi : length / post_interval = 6 / 2 := by rw [h_length, h_post_interval]
  have h_w_pi : width / post_interval = 8 / 2 := by rw [h_width, h_post_interval]
  -- leverage conditions to compute the number of posts
  rw [← h_l_pi, ← h_w_pi]
  -- compute and simplify
  sorry

end number_of_posts_l644_644533


namespace expected_volunteers_2008_l644_644355

theorem expected_volunteers_2008 (initial_volunteers: ℕ) (annual_increase: ℚ) (h1: initial_volunteers = 500) (h2: annual_increase = 1.2) : 
  let volunteers_2006 := initial_volunteers * annual_increase
  let volunteers_2007 := volunteers_2006 * annual_increase
  let volunteers_2008 := volunteers_2007 * annual_increase
  volunteers_2008 = 864 := 
by
  sorry

end expected_volunteers_2008_l644_644355


namespace full_price_ticket_revenue_l644_644538

-- Given conditions
variable {f d p : ℕ}
variable (h1 : f + d = 160)
variable (h2 : f * p + d * (2 * p / 3) = 2800)

-- Goal: Prove the full-price ticket revenue is 1680.
theorem full_price_ticket_revenue : f * p = 1680 :=
sorry

end full_price_ticket_revenue_l644_644538


namespace parking_arrangements_l644_644917

-- Definitions based on the conditions
def total_spaces : ℕ := 12
def cars : ℕ := 8
def empty_spaces : ℕ := 4
def total_units : ℕ := cars + 1  -- 8 cars + 1 unit for 4 adjacent empty spaces

theorem parking_arrangements : total_units! = 9! := sorry

end parking_arrangements_l644_644917


namespace b2012_equals_1_l644_644293

def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

def b (n : ℕ) : ℕ := (fib n) % 4

theorem b2012_equals_1 : b 2012 = 1 :=
by
  sorry

end b2012_equals_1_l644_644293


namespace abs_difference_of_squares_l644_644128

theorem abs_difference_of_squares : 
  let a := 105 
  let b := 103
  abs (a^2 - b^2) = 416 := 
by 
  let a := 105
  let b := 103
  sorry

end abs_difference_of_squares_l644_644128


namespace correct_operation_l644_644508

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l644_644508


namespace am_eq_bn_l644_644887

-- Given conditions:
-- 1. Points D and C on a circle tangent to segment AB at its midpoint
-- 2. AD and BC intersect the circle at points X and Y respectively
-- 3. CX and DY intersect AB at points M and N respectively

noncomputable def problem_statement (A B C D X Y M N : Point) (circle : Circle) (AB_tangent : TangentCircle AB circle)
    (AD : Line) (BC : Line) (CX : Line) (DY : Line) : Prop :=
  (D ∈ circle) ∧ (C ∈ circle) ∧
  (AD ∩ circle = {X}) ∧ (BC ∩ circle = {Y}) ∧
  (CX ∩ AB = {M}) ∧ (DY ∩ AB = {N}) → (A.distance M = B.distance N)

theorem am_eq_bn (A B C D X Y M N : Point) (circle : Circle) (AB_tangent : TangentCircle AB circle)
    (AD : Line) (BC : Line) (CX : Line) (DY : Line) : problem_statement A B C D X Y M N circle AB_tangent AD BC CX DY := 
by
  -- sketch of proof setup
  sorry

end am_eq_bn_l644_644887


namespace measure_of_angle_Q_l644_644465

theorem measure_of_angle_Q (P Q R : Type) 
  [triangle P Q R] 
  (is_isosceles : angles_congruent Q R)
  (angle_R_eq_3_angle_P : ∃ x : ℝ, angle R = 3 * x ∧ angle P = x)
  : angle Q = 540 / 7 := 
sorry

end measure_of_angle_Q_l644_644465


namespace tip_per_person_l644_644031

-- Define the necessary conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def total_amount_made : ℝ := 37

-- Define the problem statement
theorem tip_per_person : (total_amount_made - hourly_wage) / people_served = 1.25 :=
by
  sorry

end tip_per_person_l644_644031


namespace minimum_dwarfs_l644_644049

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l644_644049


namespace number_of_students_above_120_l644_644925

def normal_distribution (μ σ : ℝ) : Prop := sorry -- Assumes the definition of a normal distribution

-- Given conditions
def number_of_students := 1000
def μ := 100
def some_distribution (ξ : ℝ) := normal_distribution μ (σ^2)
def probability_between (a b : ℝ) : ℝ := sorry -- Assumes the definition to calculate probabilities in normal distribution
def given_probability := (probability_between 80 100) = 0.45

-- Question and proof problem
theorem number_of_students_above_120 (σ : ℝ) (h : some_distribution ξ) (hp : given_probability) :
  ∃ n : ℕ, n = 50 := by
    sorry

end number_of_students_above_120_l644_644925


namespace find_ϕ_l644_644691

noncomputable def f (ω ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem find_ϕ (ω ϕ : ℝ) (h1 : 0 < ω) (h2 : abs ϕ < Real.pi / 2) (h3 : ∀ x : ℝ, f ω ϕ (x + Real.pi / 6) = g ω x) 
  (h4 : 2 * Real.pi / ω = Real.pi) : ϕ = Real.pi / 3 :=
by sorry

end find_ϕ_l644_644691


namespace inverse_mod_l644_644593

theorem inverse_mod (a b n : ℕ) (h : (a * b) % n = 1) : b % n = a⁻¹ % n := sorry

example : ∃ x : ℕ, (10 * x) % 1729 = 1 ∧ x < 1729 :=
by
  use 1585
  have h₁ : (10 * 1585) % 1729 = 1 := by norm_num
  exact ⟨h₁, by norm_num⟩

end inverse_mod_l644_644593


namespace blue_beads_l644_644020

-- Variables to denote the number of blue, red, white, and silver beads
variables (B R W S : ℕ)

-- Conditions derived from the problem statement
def conditions : Prop :=
  (R = 2 * B) ∧
  (W = B + R) ∧
  (S = 10) ∧
  (B + R + W + S = 40)

-- The theorem to prove
theorem blue_beads (B R W S : ℕ) (h : conditions B R W S) : B = 5 :=
by
  sorry

end blue_beads_l644_644020


namespace equilateral_triangle_side_length_l644_644793

theorem equilateral_triangle_side_length
  (P G H I : Point) (D E F : Point)
  (t : ℝ)
  (h_equilateral : is_equilateral_triangle D E F)
  (h_inside : is_inside P D E F)
  (h_perpendicular_PG : is_perpendicular P G D E)
  (h_perpendicular_PH : is_perpendicular P H E F)
  (h_perpendicular_PI : is_perpendicular P I F D)
  (h_PG : dist P G = 2)
  (h_PH : dist P H = 4)
  (h_PI : dist P I = 5) :
  dist D E = 6 * sqrt 2 := sorry

end equilateral_triangle_side_length_l644_644793


namespace gear_rotations_l644_644849

-- Definitions from the conditions
def gearA_teeth : ℕ := 12
def gearB_teeth : ℕ := 54

-- The main problem: prove that gear A needs 9 rotations and gear B needs 2 rotations
theorem gear_rotations :
  ∃ x y : ℕ, 12 * x = 54 * y ∧ x = 9 ∧ y = 2 := by
  sorry

end gear_rotations_l644_644849


namespace min_dwarfs_l644_644071

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l644_644071


namespace alice_bob_numbers_count_101_l644_644935

theorem alice_bob_numbers_count_101 : 
  ∃ n : ℕ, (∀ x, 3 ≤ x ∧ x ≤ 2021 → (∃ k l, x = 3 + 5 * k ∧ x = 2021 - 4 * l)) → n = 101 :=
by
  sorry

end alice_bob_numbers_count_101_l644_644935


namespace min_dwarfs_for_no_empty_neighbor_l644_644058

theorem min_dwarfs_for_no_empty_neighbor (n : ℕ) (h : n = 30) : ∃ k : ℕ, k = 10 ∧
  (∀ seats : Fin n → Prop, (∀ i : Fin n, seats i ∨ seats ((i + 1) % n) ∨ seats ((i + 2) % n))
   → ∃ j : Fin n, seats j ∧ j = 10) :=
by
  sorry

end min_dwarfs_for_no_empty_neighbor_l644_644058


namespace probability_odd_number_l644_644430

theorem probability_odd_number 
  (digits : Finset ℕ) 
  (odd_digits : Finset ℕ)
  (total_digits : ∀d, d ∈ digits → d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9)
  (total_odd_digits : ∀d, d ∈ odd_digits → d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9) :
  card odd_digits / card digits = 4 / 5 :=
by sorry

end probability_odd_number_l644_644430


namespace triangle_area_depends_only_on_PO_l644_644575

noncomputable theory

variables (A B C P O : ℂ)

def is_equilateral_triangle (A B C : ℂ) : Prop :=
  abs (A - B) = abs (B - C) ∧ abs (B - C) = abs (C - A)

def circumcenter (A B C : ℂ) : ℂ := sorry  -- Definition of circumcenter

def is_inside_circumcircle (P A B C : ℂ) : Prop :=
  abs (P - circumcenter A B C) < abs (A - circumcenter A B C)

theorem triangle_area_depends_only_on_PO
  (hABC : is_equilateral_triangle A B C)
  (hO : O = circumcenter A B C)
  (hP : is_inside_circumcircle P A B C) :
  ∃ (Δ : triangle), 
    Δ.side_lengths = (abs (P - A), abs (P - B), abs (P - C)) ∧ 
    Δ.area_depends_only_on_PO :=
sorry

end triangle_area_depends_only_on_PO_l644_644575


namespace limit_complex_sequence_l644_644152

variable (a b : ℝ) (n : ℕ)

theorem limit_complex_sequence (h : n → ℕ) :
  ∀ a b : ℝ,
  (tendsto (λ n, (n ^ a + (complex.I * n ^ b)) / (n ^ b + (complex.I * n ^ a))) at_top
    (nhds $
      if a = b then 1
      else if a < b then complex.I
      else -complex.I)) := sorry

end limit_complex_sequence_l644_644152


namespace total_amount_paid_l644_644570

theorem total_amount_paid
    (peaches_price : ℝ)
    (apples_price : ℝ)
    (oranges_price : ℝ)
    (grapefruits_price : ℝ)
    (peaches_quantity : ℕ)
    (apples_quantity : ℕ)
    (oranges_quantity : ℕ)
    (grapefruits_quantity : ℕ) :
    peaches_price = 0.4 →
    apples_price = 0.6 →
    oranges_price = 0.5 →
    grapefruits_price = 1.0 →
    peaches_quantity = 400 →
    apples_quantity = 150 →
    oranges_quantity = 200 →
    grapefruits_quantity = 80 →
    (peaches_quantity * peaches_price + apples_quantity * apples_price + oranges_quantity * oranges_price + grapefruits_quantity * grapefruits_price)
    - ((peaches_quantity * peaches_price) / 10 * 2 + (apples_quantity * apples_price) / 15 * 3 + (oranges_quantity * oranges_price) / 7 * 1.5 + (grapefruits_quantity * grapefruits_price) / 20 * 4)
    - if peaches_quantity >= 100 ∧ apples_quantity >= 50 ∧ oranges_quantity >= 100 then 10 else 0 = 333 := by
  sorry

end total_amount_paid_l644_644570


namespace cos_sin_315_l644_644219

theorem cos_sin_315 : 
  cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 ∧ 
  sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_sin_315_l644_644219


namespace schools_in_Pythagoras_city_l644_644613

-- Conditions
def isMedian (numSchools : ℕ) (rank : ℕ) : Prop :=
  rank = (4 * numSchools + 1) / 2

def correctRanking (numSchools : ℕ) : Prop :=
  ((4 * numSchools + 1) / 2 < 50)

-- Given conditions implying number of schools
theorem schools_in_Pythagoras_city : ∃ (n : ℕ), 4 * n + 1 < 100 ∧ n = 24 :=
begin
  use 24,
  split,
  { norm_num, },
  { refl, },
end

end schools_in_Pythagoras_city_l644_644613


namespace lines_intersect_at_circumcenter_of_triangle_l644_644643

variable (A B C P A₁ B₁ C₁ : Point)
variable (PA₁ PB₁ PC₁ : Line)
variable (PA PB PC : Segment)
variable (lₐ l_b l_c : Line)
variable (M : Point)

/-- Given a triangle ABC and a point P, let PA₁, PB₁, and PC₁ be perpendiculars from P to sides BC, CA, and AB, respectively.
Define lₐ as the line connecting the midpoint of PA and B₁C₁, l_b and l_c similarly.
Prove that lₐ, l_b, and l_c intersect at a single point, the circumcenter of the triangle formed by A₁, B₁, C₁. -/
theorem lines_intersect_at_circumcenter_of_triangle :
  (Are_perpendicular PA₁ (Line.mk P A₁)) →
  (Are_perpendicular PB₁ (Line.mk P B₁)) →
  (Are_perpendicular PC₁ (Line.mk P C₁)) →
  (Midpoint PA = M) →
  (OnePoint lₐ l_b l_c) :=
sorry

end lines_intersect_at_circumcenter_of_triangle_l644_644643


namespace mode_of_dataset_l644_644415

theorem mode_of_dataset :
  ∃ m : ℕ, m = 3 ∧ 
          ∀ n : ℕ, (n = 5 ∨ n = 4 ∨ n = 6 ∨ n = 3) → 
                    (list.count n [3, 5, 4, 6, 3, 3, 4] ≤ list.count 3 [3, 5, 4, 6, 3, 3, 4]) := 
by 
  sorry

end mode_of_dataset_l644_644415


namespace bill_and_harry_nuts_l644_644948

theorem bill_and_harry_nuts {Bill Harry Sue : ℕ} 
    (h1 : Bill = 6 * Harry) 
    (h2 : Harry = 2 * Sue) 
    (h3 : Sue = 48) : 
    Bill + Harry = 672 := 
by
  sorry

end bill_and_harry_nuts_l644_644948


namespace smallest_N_l644_644568

theorem smallest_N (l m n : ℕ) (N : ℕ) (h_block : N = l * m * n)
  (h_invisible : (l - 1) * (m - 1) * (n - 1) = 120) :
  N = 216 :=
sorry

end smallest_N_l644_644568


namespace math_problem_l644_644668

def A (x : ℝ) : Prop := -1 < x ∧ x < 7
def B (x : ℝ) (m : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 3m + 1

def p (m : ℝ) : Prop := ∀ x, B x m → A x
def q (m : ℝ) : Prop := ∃! x, x^2 + 2*m*x + 2*m ≤ 0

theorem math_problem (m : ℝ) : ¬ (p m ∨ q m) → m ≥ 2 := by
  sorry

end math_problem_l644_644668


namespace bus_total_distance_l644_644897

/-- 
The bus initially travels 30 feet in the first second, and then it decelerates by 6 feet per second
each subsequent second until it comes to a stop. We are to determine the total distance traveled 
by the bus until it stops.
-/
theorem bus_total_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) (Sₙ : ℕ) : a₁ = 30 → d = -6 → n = 6 → 
Sₙ = (n * (2 * a₁ + (n - 1) * d)) / 2 → Sₙ = 90 := by
  intros h₀ h₁ h₂ h₃
  rw [h₀, h₁, h₂] at h₃
  simp at h₃
  exact h₃

end bus_total_distance_l644_644897


namespace count_digit_1_in_sum_l644_644154

theorem count_digit_1_in_sum : 
  (finset.sum (finset.range 2017) (λ k, 9 * (10^k - 1)) = 2013) :=
sorry

end count_digit_1_in_sum_l644_644154


namespace arithmetic_sequence_properties_l644_644350

-- Define initial conditions: arithmetic sequence
def a (n : ℕ) := 2 * n + 5

-- Define S_n: sum of the first n terms of a(n)
def S (n : ℕ) := n^2 + 6 * n

-- Define b_n and T_n
def b (n : ℕ) := 5 / ((2 * n + 5) * (2 * n + 7))
def T (n : ℕ) := (5 * n) / (14 * n + 49)

-- Proof problem statement
theorem arithmetic_sequence_properties :
  (∀ n, a n = 2 * n + 5) ∧
  (∀ n, S n = n^2 + 6 * n) ∧
  (∀ n, (∑ i in finset.range n, b i) = T n) :=
sorry

end arithmetic_sequence_properties_l644_644350


namespace sin_2x_plus_pi_period_odd_l644_644433

noncomputable def period_of_function : ℝ := by
  let ω := 2
  exact 2 * π / ω

theorem sin_2x_plus_pi_period_odd :
  (∀ x: ℝ, sin (2 * x + π) = -sin (2 * x)) ∧ (∀ x: ℝ, sin (2 * x + π) = sin (2 * (x + period_of_function))) := 
by
  sorry

end sin_2x_plus_pi_period_odd_l644_644433


namespace mike_maximum_marks_l644_644521

theorem mike_maximum_marks
    (score : ℕ) (shortfall : ℕ) (passing_mark : ℕ) (total_marks : ℕ) (pass_rate : ℝ)
    (h1 : pass_rate = 30 / 100)
    (h2 : score = 212)
    (h3 : shortfall = 28)
    (h4 : passing_mark = score + shortfall)
    (h5 : passing_mark = (pass_rate * total_marks : ℝ).to_nat)
    : total_marks = 800 :=
sorry

end mike_maximum_marks_l644_644521


namespace pow_div_pow_eq_result_l644_644218

theorem pow_div_pow_eq_result : 13^8 / 13^5 = 2197 := by
  sorry

end pow_div_pow_eq_result_l644_644218


namespace area_of_bounded_region_l644_644629

def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def csc (θ : ℝ) : ℝ := 1 / (Real.sin θ)

theorem area_of_bounded_region : 
  let region := {
    p : ℝ × ℝ | 
      (∃ θ, p.1 = 2) ∨ 
      (∃ θ, p.2 = 3) ∨ 
      (p.1 = 0) ∨ 
      (p.2 = 0)
  } in
  (∃ A, A = 6) := 
by
  sorry

end area_of_bounded_region_l644_644629


namespace only_prime_in_K_l644_644371

def alternating_digits_0_and_1 (n : ℕ) : Prop :=
  ∀ i < n.bits, if n.testBit i then (¬ n.testBit (i + 1)) else n.testBit (i + 1)

def first_and_last_digit_1 (n : ℕ) : Prop :=
  n.testBit 0 ∧ n.testBit (n.bits - 1)

def K (n : ℕ) : Prop :=
  alternating_digits_0_and_1 n ∧ first_and_last_digit_1 n

theorem only_prime_in_K :
  ∃ (p : ℕ), prime p ∧ K p ∧ (∀ n, K n → prime n → n = p) :=
sorry

end only_prime_in_K_l644_644371


namespace number_of_terms_in_expansion_l644_644956

theorem number_of_terms_in_expansion (A B : Finset ℕ) (h1 : A.card = 4) (h2 : B.card = 5) :
  (A.product B).card = 20 :=
by
  sorry

end number_of_terms_in_expansion_l644_644956


namespace function_is_decreasing_l644_644694

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 2

theorem function_is_decreasing (a b : ℝ) (f_even : ∀ x : ℝ, f a b x = f a b (-x))
  (domain_condition : 1 + a + 2 = 0) :
  ∀ x y : ℝ, 1 ≤ x → x < y → y ≤ 2 → f a 0 x > f a 0 y :=
by
  sorry

end function_is_decreasing_l644_644694


namespace work_days_l644_644147

theorem work_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
    1 / (A + B) = 6 :=
by
  sorry

end work_days_l644_644147


namespace minimum_dwarfs_l644_644051

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l644_644051


namespace line_shift_up_l644_644822

theorem line_shift_up (x y : ℝ) (k : ℝ) (h : y = -2 * x - 4) : 
    y + k = -2 * x - 1 := by
  sorry

end line_shift_up_l644_644822


namespace greatest_divisor_of_630_lt_35_and_factor_of_90_l644_644473

theorem greatest_divisor_of_630_lt_35_and_factor_of_90 : ∃ d : ℕ, d < 35 ∧ d ∣ 630 ∧ d ∣ 90 ∧ ∀ e : ℕ, (e < 35 ∧ e ∣ 630 ∧ e ∣ 90) → e ≤ d := 
sorry

end greatest_divisor_of_630_lt_35_and_factor_of_90_l644_644473


namespace gcd_lcm_eq_lcm_gcd_l644_644013
open Nat

theorem gcd_lcm_eq_lcm_gcd (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  gcd (lcm a b) (gcd (lcm b c) (lcm c a)) = lcm (gcd a b) (gcd (gcd b c) (gcd c a)) :=
sorry

end gcd_lcm_eq_lcm_gcd_l644_644013


namespace inches_per_foot_l644_644391

-- Definition of the conditions in the problem.
def feet_last_week := 6
def feet_less_this_week := 4
def total_inches := 96

-- Lean statement that proves the number of inches in a foot
theorem inches_per_foot : 
    (total_inches / (feet_last_week + (feet_last_week - feet_less_this_week))) = 12 := 
by sorry

end inches_per_foot_l644_644391


namespace find_initial_pencils_l644_644986

theorem find_initial_pencils (p_total : ℕ) (j : ℕ) (p_initial : ℕ) (h_total : p_total = 57) (h_j: j = 6) (h_eq : p_total = p_initial + j) : p_initial = 51 :=
by {
    rw [h_total, h_j] at h_eq,
    linarith,
}

end find_initial_pencils_l644_644986


namespace smallest_positive_multiple_of_32_l644_644477

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ n % 32 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 32 = 0 → n ≤ m :=
by
  sorry

end smallest_positive_multiple_of_32_l644_644477


namespace find_n_l644_644084

theorem find_n {n : ℕ} (avg1 : ℕ) (avg2 : ℕ) (S : ℕ) :
  avg1 = 7 →
  avg2 = 6 →
  S = 7 * n →
  6 = (S - 11) / (n + 1) →
  n = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end find_n_l644_644084


namespace equilateral_triangle_of_rotation_l644_644779

-- Define the premise conditions
variables {A0 A1 A2 P0 : ℂ} -- assuming points are in the complex plane

-- Rotation by 2π/3 around point A_m
def rotation (z A : ℂ) : ℂ :=
  A + (z - A) * complex.exp (complex.I * (2 * π / 3))

-- Sequence definition: P_n is the image of P_{n-1} by the rotation about A_m
def seq (P : ℕ → ℂ) : Prop :=
  ∀ n, ∃ m, m = (n - 1) % 3 ∧ P (n + 1) = rotation (P n) (if m = 0 then A0 else if m = 1 then A1 else A2)

-- The statement to prove
theorem equilateral_triangle_of_rotation (h : seq (λ n, P0) ∧ (P0 1986 = P0)) : 
  A0 = A1 ∨ A1 = A2 ∨ A2 = A0 := 
sorry

end equilateral_triangle_of_rotation_l644_644779


namespace find_solutions_l644_644998

noncomputable
def is_solution (a b c d : ℝ) : Prop :=
  a + b + c = d ∧ (1 / a + 1 / b + 1 / c = 1 / d)

theorem find_solutions (a b c d : ℝ) :
  is_solution a b c d ↔ (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) :=
by
  sorry

end find_solutions_l644_644998


namespace plane_crash_probabilities_eq_l644_644910

noncomputable def crashing_probability_3_engines (p : ℝ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

noncomputable def crashing_probability_5_engines (p : ℝ) : ℝ :=
  10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

theorem plane_crash_probabilities_eq (p : ℝ) :
  crashing_probability_3_engines p = crashing_probability_5_engines p ↔ p = 0 ∨ p = 1/2 ∨ p = 1 :=
by
  sorry

end plane_crash_probabilities_eq_l644_644910


namespace max_neg_ints_eq_0_l644_644599

theorem max_neg_ints_eq_0 (a b c d : ℤ) (h : 2^a + 2^b + 5 = 3^c + 3^d) : 
  (a < 0 → false) ∧ (b < 0 → false) ∧ (c < 0 → false) ∧ (d < 0 → false) :=
by
  sorry

end max_neg_ints_eq_0_l644_644599


namespace difference_of_numbers_l644_644160

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := 
by
  sorry

end difference_of_numbers_l644_644160


namespace pentagon_area_larger_than_octagon_l644_644591

-- Define the side length as a constant
def side_length : ℝ := 3

-- Define helper functions for radii and apothems
noncomputable def apothem (sides : ℝ) (angle : ℝ) (side_length : ℝ) : ℝ :=
  side_length / (2 * Math.tan (angle / 2))

noncomputable def circumradius (sides : ℝ) (angle : ℝ) (side_length : ℝ) : ℝ :=
  side_length / (2 * Math.sin (angle / 2))

-- Conditions and values for the pentagon
def pentagon_sides : ℝ := 5
def pentagon_angle : ℝ := 360 / pentagon_sides

noncomputable def A_pentagon := apothem (pentagon_sides) (pentagon_angle) side_length
noncomputable def R_pentagon := circumradius (pentagon_sides) (pentagon_angle) side_length

-- Conditions and values for the octagon
def octagon_sides : ℝ := 8
def octagon_angle : ℝ := 360 / octagon_sides

noncomputable def A_octagon := apothem (octagon_sides) (octagon_angle) side_length
noncomputable def R_octagon := circumradius (octagon_sides) (octagon_angle) side_length

-- Area differences
noncomputable def area_diff (R : ℝ) (A : ℝ) : ℝ :=
  π * (R ^ 2 - A ^ 2)

-- Compare areas
theorem pentagon_area_larger_than_octagon :
  area_diff R_pentagon A_pentagon > area_diff R_octagon A_octagon :=
sorry

end pentagon_area_larger_than_octagon_l644_644591


namespace additional_grassy_ground_l644_644148

theorem additional_grassy_ground (r1 r2 : ℝ) (h1: r1 = 16) (h2: r2 = 23) :
  (π * r2 ^ 2) - (π * r1 ^ 2) = 273 * π :=
by
  sorry

end additional_grassy_ground_l644_644148


namespace polynomial_solution_l644_644627

noncomputable def is_solution (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2)

theorem polynomial_solution (P : ℝ → ℝ) : is_solution P → ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
  sorry

end polynomial_solution_l644_644627


namespace probability_particle_at_23_l644_644557

noncomputable def probability_at_point : ℚ :=
  let prob_move := (1 / 2 : ℚ)
  let num_moves := 5
  let num_rights := 2
  ∑ k in finset.range (num_moves + 1), if k = num_rights then (nat.choose num_moves k) * prob_move ^ num_moves else 0

theorem probability_particle_at_23 :
  probability_at_point = (nat.choose 5 2) * (1/2 : ℚ) ^ 5 :=
by
  sorry

end probability_particle_at_23_l644_644557


namespace sum_of_natural_numbers_l644_644586

theorem sum_of_natural_numbers (n : ℕ) (h : n * (n + 1) = 812) : n = 28 := by
  sorry

end sum_of_natural_numbers_l644_644586


namespace prove_correct_option_C_l644_644500

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l644_644500


namespace nine_values_of_x_l644_644716

theorem nine_values_of_x : ∃! (n : ℕ), ∃! (xs : Finset ℕ), xs.card = n ∧ 
  (∀ x ∈ xs, 3 * x < 100 ∧ 4 * x ≥ 100) ∧ 
  (xs.image (λ x => x)).val = ({25, 26, 27, 28, 29, 30, 31, 32, 33} : Finset ℕ).val :=
sorry

end nine_values_of_x_l644_644716


namespace quadrilateral_intersection_point_of_midlines_l644_644563

noncomputable def point (α : Type*) : Type* := sorry
noncomputable def line (α : Type*) : Type* := sorry
noncomputable def quadrilateral (α : Type*) : Type* := sorry

variables {α : Type*} [metric_space α]

-- Define the quadrilateral
def quadrilateral_circumscribed_about_circle (ABCD : quadrilateral α) (O : point α) : Prop := sorry
-- Define the midpoints of the opposite sides
def midpoint (P Q : point α) : point α := sorry

-- Define the lines connecting midpoints
def line_through (P Q : point α) : line α := sorry

-- Define the intersection of two lines
def intersection (l1 l2 : line α) (O : point α) : Prop := sorry

-- Define the condition of the multiplicative relation
def multiplicative_condition (O A B C D : point α) : Prop :=
  (dist O A) * (dist O C) = (dist O B) * (dist O D)

-- Main theorem statement
theorem quadrilateral_intersection_point_of_midlines (ABCD : quadrilateral α) (O A B C D X Y Z W : point α) :
  quadrilateral_circumscribed_about_circle ABCD O →
  midpoint A B = X →
  midpoint C D = Y →
  midpoint B C = Z →
  midpoint D A = W →
  (intersection (line_through X Y) (line_through Z W) O ↔ multiplicative_condition O A B C D) :=
sorry

end quadrilateral_intersection_point_of_midlines_l644_644563


namespace bill_and_harry_nuts_l644_644949

theorem bill_and_harry_nuts {Bill Harry Sue : ℕ} 
    (h1 : Bill = 6 * Harry) 
    (h2 : Harry = 2 * Sue) 
    (h3 : Sue = 48) : 
    Bill + Harry = 672 := 
by
  sorry

end bill_and_harry_nuts_l644_644949


namespace alice_single_letter_probability_l644_644936

theorem alice_single_letter_probability :
  (let S := {a, b, c}
   let n := 1001
   let prob_all_same := 3^(- (n - 1))
   let x_0 := 1
   let x_m : ℕ → ℚ := λ m, (1 / 4) * (1 + 3 * (-3)^(-m))
   let x_1001 := x_m 1001
   let target_prob := 1 - (prob_all_same + x_1001)
   (target_prob : ℚ) = (3 - 3^(-999)) / 4) :=
by
  sorry

end alice_single_letter_probability_l644_644936


namespace equidistant_point_quadrants_l644_644702

theorem equidistant_point_quadrants :
  ∀ (x y : ℝ), 3 * x + 5 * y = 15 → (|x| = |y| → (x > 0 → y > 0 ∧ x = y ∧ y = x) ∧ (x < 0 → y > 0 ∧ x = -y ∧ -x = y)) := 
by
  sorry

end equidistant_point_quadrants_l644_644702


namespace distance_travelled_from_accident_optimal_speed_to_minimize_fuel_l644_644321

-- Part I
def vehicle_speed (t : ℝ) : ℝ := (100 / (3 * (t + 1))) - (5 / 3 * t)

theorem distance_travelled_from_accident (reaction_time : ℝ) (speed_at_reaction : ℝ) (ln5 : ℝ) : 
  ∃ s : ℝ, s = 70 :=
  let distance_travelled := reaction_time * speed_at_reaction * (1000 / 3600) +
                            ∫ (t : ℝ) in (0..4), vehicle_speed t 
  ⟨70, by sorry⟩

-- Part II
def fuel_cost (v : ℝ) : ℝ := (v^2 / 250) + 40

theorem optimal_speed_to_minimize_fuel (distance : ℝ) (v_min v_max : ℝ) (h_min : v_min = 60) (h_max : v_max = 120) : 
  ∃ v_opt : ℝ, v_opt = 100 :=
  let fuel_cost_given_distance := (S / v) * (fuel_cost v)
  ⟨100, by sorry⟩

end distance_travelled_from_accident_optimal_speed_to_minimize_fuel_l644_644321


namespace part_I_part_II_l644_644804

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |2 * x - 1|

theorem part_I (a : ℝ) (x : ℝ) (h : a = 2) :
    f x a + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2 := 
by
    -- problem restatement
    sorry

theorem part_II (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 3) :
    -3 ≤ a ∧ a ≤ 5 := 
by
    -- problem restatement
    sorry

end part_I_part_II_l644_644804


namespace remainder_when_divided_by_22_l644_644142

theorem remainder_when_divided_by_22 (y : ℤ) (k : ℤ) (h : y = 264 * k + 42) : y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l644_644142


namespace num_unpronounceable_words_of_length_7_l644_644970

def alphabet : set char := {'A', 'B'}

def is_unpronounceable (word : list char) : Prop :=
  ∃ (a : char), a ∈ alphabet ∧ (list.repeat a 3).is_infix_of word

theorem num_unpronounceable_words_of_length_7 :
  let words := list.replicateM 7 ['A', 'B'] in
  (words.filter is_unpronounceable).length = 86 :=
by sorry

end num_unpronounceable_words_of_length_7_l644_644970


namespace possible_perimeters_outer_square_l644_644880

theorem possible_perimeters_outer_square (a b : ℕ) (h1 : b < 10) (h2 : a < 10) (h3 : a > b) (h4 : Nat.gcd a b = 1) (h5 : 8 * b = 16) :
  ∃ perimeters : Finset ℕ, perimeters = {24, 40, 56, 72} :=
by
  have b_eq_2 : b = 2 :=
    by linarith
  have a_values : Finset ℕ := {3, 5, 7, 9}
  have perimeter_set : Finset ℕ := (a_values.image (λ a, 8 * a))
  have perimeters_goal : Finset ℕ := {24, 40, 56, 72}
  use perimeter_set
  sorry

end possible_perimeters_outer_square_l644_644880


namespace problem_statement_l644_644667

noncomputable def proposition_p (x : ℝ) : Prop := ∃ x0 : ℝ, x0 - 2 > 0
noncomputable def proposition_q (x : ℝ) : Prop := ∀ x : ℝ, (2:ℝ)^x > x^2

theorem problem_statement : ∃ (p q : Prop), (∃ x0 : ℝ, x0 - 2 > 0) ∧ (¬ (∀ x : ℝ, (2:ℝ)^x > x^2)) :=
by
  sorry

end problem_statement_l644_644667


namespace abs_diff_26th_term_l644_644466

def C (n : ℕ) : ℤ := 50 + 15 * (n - 1)
def D (n : ℕ) : ℤ := 85 - 20 * (n - 1)

theorem abs_diff_26th_term :
  |(C 26) - (D 26)| = 840 := by
  sorry

end abs_diff_26th_term_l644_644466


namespace repeating_decimal_as_fraction_l644_644246

def repeating_decimal := ∀ (x : ℝ), x = 3.363636...

theorem repeating_decimal_as_fraction (x : ℝ) (hx : repeating_decimal x) : x = 37 / 11 :=
by sorry

end repeating_decimal_as_fraction_l644_644246


namespace maximum_profit_at_4_l644_644176

-- Definition of the cost function G(x)
def G (x : ℝ) : ℝ := 15 + 5 * x

-- Definition of the revenue function R(x)
def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -2 * x^2 + 21 * x + 1 else 56

-- Definition of the profit function f(x)
def f (x : ℝ) : ℝ := R(x) - G(x)

-- Prove that the maximum profit f(x) is 18 when x = 4
theorem maximum_profit_at_4 : ∃ (x : ℝ), x = 4 ∧ f(x) = 18 :=
by
  sorry

end maximum_profit_at_4_l644_644176


namespace largest_three_digit_multiple_of_6_sum_15_l644_644860

-- Statement of the problem in Lean
theorem largest_three_digit_multiple_of_6_sum_15 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 6 = 0 ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 6 = 0 ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
by
  sorry -- proof not required

end largest_three_digit_multiple_of_6_sum_15_l644_644860


namespace william_riding_hours_l644_644873

theorem william_riding_hours :
  ∃ x : ℝ, 
    (∀ (max_hours : ℝ) (days : ℤ) (max_used_days : ℤ) (half_days : ℤ) (total_hours : ℝ), 
      max_hours = 6 →
      days = 6 →
      max_used_days = 2 →
      half_days = 2 →
      total_hours = 21 →
      2 * max_hours + 2 * x + 2 * (max_hours / 2) = total_hours) →
    x = 1.5 :=
begin
  use 1.5,
  intros max_hours days max_used_days half_days total_hours,
  intros hm hd hmd hh ht,
  -- proof goes here
  sorry
end

end william_riding_hours_l644_644873


namespace ellipse_chord_length_l644_644579

variables (a b : ℝ)

-- Definitions based on given conditions
def ellipse_standard_eq (a b : ℝ) : Prop := (a = 2) ∧ (c = 1) ∧ (b^2 = a^2 - c^2)
def line_eq : Prop := ∀ x y : ℝ, y = x + 1 ↔ (7 * x^2 + 8 * x - 8) = 0
def chord_length : Prop := ∀ x₁ x₂ : ℝ, 
  x₁ + x₂ = -8 / 7 ∧ x₁ * x₂ = -8 / 7 → 
  abs ((x₁ - x₂) * sqrt 2) = 24 / 7

-- The Lean statement 
theorem ellipse_chord_length :
  ellipse_standard_eq a b ∧ line_eq ∧ chord_length :=
by sorry

end ellipse_chord_length_l644_644579


namespace repeating_decimal_as_fraction_l644_644245

def repeating_decimal := ∀ (x : ℝ), x = 3.363636...

theorem repeating_decimal_as_fraction (x : ℝ) (hx : repeating_decimal x) : x = 37 / 11 :=
by sorry

end repeating_decimal_as_fraction_l644_644245


namespace least_N_bench_sections_l644_644567

-- First, define the problem conditions
def bench_capacity_adult (N : ℕ) : ℕ := 7 * N
def bench_capacity_child (N : ℕ) : ℕ := 11 * N

-- Define the problem statement to be proven
theorem least_N_bench_sections :
  ∃ N : ℕ, (N > 0) ∧ (bench_capacity_adult N = bench_capacity_child N → N = 77) :=
sorry

end least_N_bench_sections_l644_644567


namespace concyclic_iff_directed_angle_eq_l644_644005

variables {A B C D: Type*} [ordered_field A]

-- Define points and angles
variable (α : Type*) [has_angle α]

/-- Definition for points on a plane. -/
noncomputable def are_concyclic (a b c d : α) : Prop :=
∃ (C : set α), is_circle C ∧ a ∈ C ∧ b ∈ C ∧ c ∈ C ∧ d ∈ C

/-- We'll use the directed angle notation as per the problem's conditions. -/
variable directed_angle: α → α → α → angle

noncomputable def directed_angle_eq (a b c d : α) : Prop :=
directed_angle a b c = directed_angle d b c

theorem concyclic_iff_directed_angle_eq (a b c d : α) :
  are_concyclic a b c d ↔ directed_angle_eq a b c d :=
by sorry

end concyclic_iff_directed_angle_eq_l644_644005


namespace part1_probability_real_roots_part2_probability_real_roots_l644_644971

theorem part1_probability_real_roots :
  (let A := { x : ℚ × ℚ // (x.1 ∈ {0, 1, 2, 3} ∧ x.2 ∈ {0, 1, 2, 3}) }
   in (∃ a b : ℚ, a ∈ {0, 1, 2, 3} ∧ b ∈ {0, 1, 2, 3} ∧ (2*a)^2 - 4*(b^2) ≥ 0) →
      (finset.card (finset.filter (λ ab, (ab.1 ≥ ab.2)) A.to_finset)) / (finset.card A.to_finset) = 5 / 8) 
:= by sorry

theorem part2_probability_real_roots :
  (let A := { x : ℚ × ℚ // (0 ≤ x.1 ∧ x.1 ≤ 3) ∧ (0 ≤ x.2 ∧ x.2 ≤ 2) }
   in ∃ a b : ℚ, (0 ≤ a ∧ a ≤ 3) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (2*a)^2 - 4*(b^2) ≥ 0 →
      (finset.card (finset.filter (λ ab, (ab.1 ≥ ab.2)) A.to_finset)) / (finset.card A.to_finset) = 2 / 3) 
:= by sorry

end part1_probability_real_roots_part2_probability_real_roots_l644_644971


namespace calculate_shaded_area_l644_644944

noncomputable def square_shaded_area : ℝ := 
  let a := 10 -- side length of the square
  let s := a / 2 -- half side length, used for midpoints
  let total_area := a * a / 2 -- total area of a right triangle with legs a and a
  let triangle_DMA := total_area / 2 -- area of triangle DAM
  let triangle_DNG := triangle_DMA / 5 -- area of triangle DNG
  let triangle_CDM := total_area -- area of triangle CDM
  let shaded_area := triangle_CDM + triangle_DNG - triangle_DMA -- area of shaded region
  shaded_area

theorem calculate_shaded_area : square_shaded_area = 35 := 
by 
sorry

end calculate_shaded_area_l644_644944


namespace option_C_correct_l644_644486

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l644_644486


namespace angle_bisector_equation_intersection_l644_644254

noncomputable def slope_of_angle_bisector (m1 m2 : ℝ) : ℝ :=
  (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)

noncomputable def equation_of_angle_bisector (x : ℝ) : ℝ :=
  (Real.sqrt 21 - 6) / 7 * x

theorem angle_bisector_equation_intersection :
  let m1 := 2
  let m2 := 4
  slope_of_angle_bisector m1 m2 = (Real.sqrt 21 - 6) / 7 ∧
  equation_of_angle_bisector 1 = (Real.sqrt 21 - 6) / 7 :=
by
  sorry

end angle_bisector_equation_intersection_l644_644254


namespace simplest_fractions_count_l644_644937

/--
Among the following fractions:
  (1) (x^2 - 2 * x) / (2 * y - x * y),
  (2) (x + 1) / (x^2 + 1),
  (3) (-2) / (a^2 - 2 * a),
  (4) (12 * x * y) / (9 * z^3),
there are 2 fractions that are in their simplest form.
-/
theorem simplest_fractions_count 
  (x y z a : ℕ) :
  let F1 := (x^2 - 2 * x) / (2 * y - x * y),
      F2 := (x + 1) / (x^2 + 1),
      F3 := (-2) / (a^2 - 2 * a),
      F4 := (12 * x * y) / (9 * z^3) in
  (is_simplified F2) ∧ (is_simplified F3) ∧ ¬(is_simplified F1) ∧ ¬(is_simplified F4) →
  (count_simplified [F1, F2, F3, F4] = 2)
  :=
by { sorry } -- Proof omitted

end simplest_fractions_count_l644_644937


namespace inequality_solution_range_of_a_l644_644695

def f (x a : ℝ) : ℝ := |x - 3| - |x - a|

-- Statement 1
theorem inequality_solution (x : ℝ) : f x 2 ≤ -1 / 2 ↔ x ≥ 11 / 4 :=
by
  sorry

-- Statement 2
theorem range_of_a (a : ℝ) : (∃ x : ℝ, f x a ≥ a) ↔ a ≤ 3 / 2 :=
by
  sorry

end inequality_solution_range_of_a_l644_644695


namespace series_value_l644_644109

theorem series_value : 
  (∑ n in finset.range 128, 1 / ((2 * n + 1) * (2 * n + 3))) = 128 / 257 := 
  sorry

end series_value_l644_644109


namespace keith_attended_games_l644_644842

-- Definitions from the conditions
def total_games : ℕ := 20
def missed_games : ℕ := 9

-- The statement to prove
theorem keith_attended_games : (total_games - missed_games) = 11 :=
by
  sorry

end keith_attended_games_l644_644842


namespace second_derivative_parametric_l644_644634

theorem second_derivative_parametric:
  (∀ t : ℝ, (x'' t = 2)) := 
by
  let x := λ t : ℝ, cos t + sin t
  let y := λ t : ℝ, sin (2 * t)
  have h1 : ∀ t : ℝ, ∂²/∂x² y (x t) = 2 := sorry 
  exact h1

end second_derivative_parametric_l644_644634


namespace shortest_distance_from_point_to_circle_is_five_sqrt_two_minus_one_l644_644135

noncomputable def shortest_distance (P : ℝ × ℝ) (a b r : ℝ) : ℝ :=
  let (x_p, y_p) := P
  let distance_center := real.sqrt ((x_p - a)^2 + (y_p - b)^2)
  distance_center - r

theorem shortest_distance_from_point_to_circle_is_five_sqrt_two_minus_one :
  shortest_distance (4, -3) 5 4 1 = 5 * real.sqrt 2 - 1 := 
sorry

end shortest_distance_from_point_to_circle_is_five_sqrt_two_minus_one_l644_644135


namespace part_I_part_II_monotonicity_part_II_range_part_III_l644_644697

-- Part (Ⅰ)
theorem part_I (b : ℝ) : 2 * sqrt (2^b) = 6 → b = 2 * log 3 / log 2 := sorry

-- Part (Ⅱ)
def f (x : ℝ) : ℝ := (4 * x^2 - 12 * x - 3) / (2 * x + 1)

theorem part_II_monotonicity : 
  (∀ x ∈ set.Icc 0 (1 / 2), f'' x ≤ 0) ∧ (∀ x ∈ set.Icc (1 / 2) 1, f'' x ≥ 0) := sorry

theorem part_II_range : 
  set.image f (set.Icc 0 1) = set.Icc (-4) (-3) := sorry

-- Part (Ⅲ)
def g (x : ℝ) (c : ℝ) : ℝ := -x - 2 * c

theorem part_III (c : ℝ) : 
  (∀ x₁ ∈ set.Icc 0 1, ∃ x₂ ∈ set.Icc 0 1, g x₂ c = f x₁) → c = 3 / 2 := sorry

end part_I_part_II_monotonicity_part_II_range_part_III_l644_644697


namespace minimum_value_cubic_func_l644_644275

noncomputable def cubic_monotonically_increasing_min_value (a b c d : ℝ) (h1 : a < b) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : ℝ :=
let t := b / a in
if t > 1 then 8 + 6 * Real.sqrt 2 else sorry

theorem minimum_value_cubic_func (a b c d : ℝ) (h1 : a < b) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  cubic_monotonically_increasing_min_value a b c d h1 h2 = 8 + 6 * Real.sqrt 2 :=
sorry

end minimum_value_cubic_func_l644_644275


namespace isosceles_triangle_perimeter_l644_644348

/-- 
Prove that the perimeter of an isosceles triangle with sides 6 cm and 8 cm, 
and an area of 12 cm², is 20 cm.
--/
theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (S : ℝ) (h3 : S = 12) :
  a ≠ b →
  a = c ∨ b = c →
  ∃ P : ℝ, P = 20 := sorry

end isosceles_triangle_perimeter_l644_644348


namespace real_values_satisfying_condition_l644_644266

noncomputable def num_real_values_satisfying_condition : ℝ :=
  number_of_reals (λ d : ℝ, abs (1/3 - d * complex.I) = 2/3) = 2

theorem real_values_satisfying_condition : num_real_values_satisfying_condition = 2 :=
sorry

end real_values_satisfying_condition_l644_644266


namespace find_nonnegative_real_numbers_l644_644623

noncomputable def integer_function (x : ℝ) : ℝ :=
  (13 + real.sqrt x)^(1/3) + (13 - real.sqrt x)^(1/3)

theorem find_nonnegative_real_numbers (x : ℝ) (hx : 0 ≤ x) (h : ∃ (k : ℤ), integer_function x = k) :
  x = 137 + 53 / 216 ∨ x = 168 + 728 / 729 ∨ x = 196 ∨ x = 747 + 19 / 27 :=
sorry

end find_nonnegative_real_numbers_l644_644623


namespace exists_n_2_pow_k_divides_n_n_minus_m_l644_644375

theorem exists_n_2_pow_k_divides_n_n_minus_m 
  (k : ℕ) (m : ℤ) (h1 : 0 < k) (h2 : Odd m) : 
  ∃ n : ℕ, 0 < n ∧ 2^k ∣ (n^n - m) :=
sorry

end exists_n_2_pow_k_divides_n_n_minus_m_l644_644375


namespace arithmetic_sequence_transformation_l644_644885

theorem arithmetic_sequence_transformation (x : ℤ) (a b : ℤ) : 
  (a = x + 6) ∧ (b = x + 12) ∧ (x = 2013) → a = 2019 ∧ b = 2025 :=
by 
  intro h,
  cases h with ha hb,
  cases hb with hb hx,
  rw hx at ha hb,
  exact ⟨ha, hb⟩

end arithmetic_sequence_transformation_l644_644885


namespace good_sequences_count_l644_644288

-- Definitions for the mathematical problem
def is_good_sequence (m n : ℕ) (a : Fin m → ℕ) : Prop :=
  (∃ k, 1 ≤ k ∧ k ≤ m ∧ (a ⟨k - 1, sorry⟩ + k) % 2 = 1) ∨
  (∃ k l, 1 ≤ k ∧ k < l ∧ l ≤ m ∧ a ⟨k - 1, sorry⟩ > a ⟨l - 1, sorry⟩)

def total_permutations (n m : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - m)

def number_of_bad_sequences (m n : ℕ) : ℕ :=
  -- This should be the calculation of bad sequences as per the solution
  sorry

def number_of_good_sequences (m n : ℕ) : ℕ :=
  total_permutations n m - number_of_bad_sequences m n

theorem good_sequences_count (m n : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ n) :
  ∃ P, P = number_of_good_sequences m n :=
begin
  use total_permutations n m - number_of_bad_sequences m n,
  sorry,
end

end good_sequences_count_l644_644288


namespace quadratic_inequality_solution_l644_644982

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + m > 0) ↔ m > 1 :=
by
  sorry

end quadratic_inequality_solution_l644_644982


namespace complex_problem_l644_644297

noncomputable def z (θ : ℝ) : ℂ :=
  ℂ.cos θ + ℂ.sin θ * complex.I

theorem complex_problem 
  (z : ℂ) 
  (h : z + 1/z = 2 * ℂ.cos (Real.pi / 36)) : 
  z^500 + 1/z^500 = 2 * ℂ.cos (500 * Real.pi / 180) :=
by
  sorry

end complex_problem_l644_644297


namespace at_least_one_makes_shot_l644_644932

-- Define the probabilities for player A and B making their shots
def probA : ℝ := 0.5
def probB : ℝ := 0.4

-- Define the events of player A and B not making their shots
def probNotA : ℝ := 1 - probA
def probNotB : ℝ := 1 - probB

-- Calculation of the probability that both miss their shots
def probBothMiss : ℝ := probNotA * probNotB

-- The probability that at least one of them makes the shot
theorem at_least_one_makes_shot : probBothMiss = 0.3 → 1 - probBothMiss = 0.7 :=
by
  intro h
  rw h
  norm_num

end at_least_one_makes_shot_l644_644932


namespace minimum_score_for_advanced_course_l644_644845

theorem minimum_score_for_advanced_course (q1 q2 q3 q4 : ℕ) (H1 : q1 = 88) (H2 : q2 = 84) (H3 : q3 = 82) :
  (q1 + q2 + q3 + q4) / 4 ≥ 85 → q4 = 86 := by
  sorry

end minimum_score_for_advanced_course_l644_644845


namespace minimum_dwarfs_l644_644055

-- Define the problem conditions
def chairs := fin 30 → Prop
def occupied (C : chairs) := ∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))

-- State the theorem that provides the solution
theorem minimum_dwarfs (C : chairs) : (∀ i : fin 30, C i → ¬(C (i + 1) ∧ C (i + 2))) → ∃ n : ℕ, n = 10 :=
by
  sorry

end minimum_dwarfs_l644_644055


namespace count_valid_n_l644_644324

def factorial : (ℕ → ℕ) 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem count_valid_n :
  let n_values := {n : ℕ | n % 4 = 0 ∧ Nat.lcm (factorial 4) n = 4 * Nat.gcd (factorial 8) n} in
  Fintype.card n_values = 4 :=
  by
  sorry

end count_valid_n_l644_644324


namespace arithmetic_sequence_a₄_l644_644106

open Int

noncomputable def S (a₁ d n : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a₄ {a₁ d : ℤ}
  (h₁ : S a₁ d 5 = 15) (h₂ : S a₁ d 9 = 63) :
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a₄_l644_644106


namespace intersection_of_sets_l644_644709

theorem intersection_of_sets :
  let M := { x : ℝ | |2 * x - 1| < 1 }
  let N := { x : ℝ | 3 ^ x > 1 }
  M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } := by
  sorry

end intersection_of_sets_l644_644709


namespace area_enclosed_by_cosine_curve_l644_644229

theorem area_enclosed_by_cosine_curve :
  let a := -Real.pi / 3
  let b := Real.pi / 3
  ∫ x in a..b, Real.cos x = Real.sqrt 3 :=
by
  let a := -Real.pi / 3
  let b := Real.pi / 3
  have h := (interval_integral.integral_cos_sub).2
  sorry

end area_enclosed_by_cosine_curve_l644_644229


namespace unique_plane_skew_lines_l644_644334

-- Define the concept of skew lines, parallelism, and subset in 3D space
variables {α : Type*} [Nonempty α] [LinearOrder α] [AddCommGroup α]

-- Definitions for line in 3D space
structure Line (α : Type*) := 
  (point : α) 
  (direction : α)

-- Define skew lines: two lines that do not intersect and are not parallel
def skew_lines (a b : Line α) : Prop := 
  ¬ (∃ p, p ∈ a ∧ p ∈ b) ∧ ¬ (parallel a b)

-- Define a plane and the required relationships
structure Plane (α : Type*) :=
  (contains_line : Line α → Prop)

def line_in_plane (L : Line α) (Π : Plane α) : Prop := Π.contains_line L

def parallel_to_plane (L : Line α) (Π : Plane α) : Prop :=
  -- Definition of line parallel to plane will need precise geometric formulation in Lean
  sorry

-- The theorem we want to prove
theorem unique_plane_skew_lines (a b : Line α) (h_skew : skew_lines a b) :
  ∃! β : Plane α,
    line_in_plane a β ∧ parallel_to_plane b β :=
by 
  sorry

end unique_plane_skew_lines_l644_644334


namespace same_color_neighbors_l644_644342

-- Define the parameters based on the conditions.
variables (n : ℕ) (blue_red_pairs : ℕ)

-- Declare the conditions in terms of Lean's variables.
def condition1 : Prop := n = 150
def condition2 : Prop := blue_red_pairs = 12

-- Define the theorem to be proven.
theorem same_color_neighbors (h1 : condition1) (h2 : condition2) : ∃ same_color : ℕ, same_color = 126 :=
by {
  -- Prove the theorem given the conditions.
  sorry
}

end same_color_neighbors_l644_644342


namespace chemistry_marks_l644_644922

-- Definitions based on given conditions
def total_marks (P C M : ℕ) : Prop := P + C + M = 210
def avg_physics_math (P M : ℕ) : Prop := (P + M) / 2 = 90
def physics_marks (P : ℕ) : Prop := P = 110
def avg_physics_other_subject (P C : ℕ) : Prop := (P + C) / 2 = 70

-- The proof problem statement
theorem chemistry_marks {P C M : ℕ} (h1 : total_marks P C M) (h2 : avg_physics_math P M) (h3 : physics_marks P) : C = 30 ∧ avg_physics_other_subject P C :=
by 
  -- Proof goes here
  sorry

end chemistry_marks_l644_644922


namespace slope_range_trajectory_equation_slope_range_m_l644_644285

noncomputable def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 + 3 = 0}

theorem slope_range {k : ℝ} (h : ∃ A B : ℝ × ℝ, A ∈ circle ∧ B ∈ circle ∧ A ≠ B ∧ A ≠ 0 ∧ B ≠ 0 ∧ A.2 = k * A.1 ∧ B.2 = k * B.1) :
  -real.sqrt (3) / 3 < k ∧ k < real.sqrt (3) / 3 :=
sorry

noncomputable def trajectory := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 = 0 ∧ 3 / 2 < p.1 ∧ p.1 ≤ 2}

theorem trajectory_equation (P : ℝ × ℝ) (h : ∃ A B : ℝ × ℝ, A ≠ B ∧ A.2 = k * A.1 ∧ B.2 = k * B.1 ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ P ∈ circle ∧ trajectory) :
  P ∈ trajectory :=
sorry

theorem slope_range_m (a : ℝ) (h : ∃ P ∈ trajectory, (P.2 = a * P.1 + 4) ∧ ∃! P : (P ∈ trajectory ∧ P.2 = a * P.1 + 4)):
  a = -15 / 8 ∨ (-real.sqrt (3) - 8) / 3 < a ∧ a ≤ (real.sqrt (3) - 8) / 3 :=
sorry

end slope_range_trajectory_equation_slope_range_m_l644_644285


namespace problem_solution_l644_644376

noncomputable def polynomial_roots := { p q r : ℝ // (p + q + r = 8) ∧ (pq + pr + qr = 10) ∧ (pqr = 3) }

theorem problem_solution (p q r : polynomial_roots) :
    (p * q * r : ℝ) = 3 →
    (p + q + r : ℝ) = 8 →
    (p * q + p * r + q * r : ℝ) = 10 →
    (p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 38) :=
by
  sorry

end problem_solution_l644_644376


namespace total_expenditure_correct_l644_644879

-- Define the dimensions of the hall in meters
def length : ℝ := 20
def width : ℝ := 15
def height : ℝ := 5

-- Define cost per square meter in Rs.
def cost_per_m2 : ℝ := 20

-- Total cost calculation
noncomputable def total_expenditure : ℝ := 
  let floor_ceiling_area := 2 * (length * width)
  let longer_walls_area := 2 * (height * length)
  let shorter_walls_area := 2 * (height * width)
  let total_area := floor_ceiling_area + longer_walls_area + shorter_walls_area
  total_area * cost_per_m2

theorem total_expenditure_correct : total_expenditure = 19000 :=
by
  -- Notice this is satisfied as it matches the problem's correct answer Rs. 19,000.
  sorry

end total_expenditure_correct_l644_644879


namespace Sheila_attend_probability_l644_644407

noncomputable def prob_rain := 0.3
noncomputable def prob_sunny := 0.4
noncomputable def prob_cloudy := 0.3

noncomputable def prob_attend_if_rain := 0.25
noncomputable def prob_attend_if_sunny := 0.9
noncomputable def prob_attend_if_cloudy := 0.5

noncomputable def prob_attend :=
  prob_rain * prob_attend_if_rain +
  prob_sunny * prob_attend_if_sunny +
  prob_cloudy * prob_attend_if_cloudy

theorem Sheila_attend_probability : prob_attend = 0.585 := by
  sorry

end Sheila_attend_probability_l644_644407


namespace number_of_true_propositions_solution_correct_l644_644980

def negation_proposition_condition : Prop := 
  ¬(∀ x : ℝ, sin x ≤ 1) = ∃ x : ℝ, sin x > 1

def converse_proposition_condition : Prop := 
  ∀ (a b m : ℝ), a < b → a * m^2 < b * m^2

def p (x : ℝ) : Prop := x ≥ 1 → log x ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

def proposition_3 : Prop := (∀ x : ℝ, p x) ∨ q

theorem number_of_true_propositions : ℕ :=
  if negation_proposition_condition then 1 else 0 +
  if converse_proposition_condition then 1 else 0 +
  if proposition_3 then 1 else 0

theorem solution_correct : number_of_true_propositions = 2 := sorry

end number_of_true_propositions_solution_correct_l644_644980


namespace solution_of_system_eq_l644_644449

-- Define the conditions: the system of equations 
def system_eq (x y : ℝ) : Prop := 
  (x + y = 2) ∧ (x - y = 0)

-- Define the expected solution set
def solution_set : set (ℝ × ℝ) := {p | p = (1, 1)}

-- The theorem to prove:
theorem solution_of_system_eq : {p : ℝ × ℝ | system_eq p.1 p.2} = solution_set := 
by 
  sorry

end solution_of_system_eq_l644_644449


namespace infinite_similar_polygons_l644_644278

theorem infinite_similar_polygons (P : Type) [polygon P] :
  ∃ (P1 P2 : Type) [polygon P1] [polygon P2], 
  (similar P1 P) ∧ (similar P2 P) ∧ 
  (area P1 + area P2 = area P) ∧ 
  (∀ k1 k2 : ℝ, (k1^2 + k2^2 = 1) → similar (scale P k1) P1 ∧ similar (scale P k2) P2) :=
sorry

end infinite_similar_polygons_l644_644278


namespace total_cost_function_optimal_speed_l644_644419

noncomputable def energy_cost (v : ℝ) : ℝ := 6 * 10^(-8) * v^3

noncomputable def total_cost (v : ℝ) : ℝ := 1750 * (6 * 10^(-8) * v^2 + 3.24 / v)

theorem total_cost_function (v : ℝ) (C : ℝ) (h1 : 0 < v) (h2 : v ≤ C) (h3 : 0 < C ∧ C ≤ 400) :
  total_cost v = 105 * (10^(-6) * v^2 + 54 / v) :=
by
  sorry

theorem optimal_speed (C : ℝ) (h1 : 0 < C ∧ C ≤ 400) :
  (∀ v, 0 < v ∧ v ≤ C → total_cost v ≥ total_cost C) ∧ (C >= 300 → total_cost 300 ≤ total_cost C) :=
by
  sorry

end total_cost_function_optimal_speed_l644_644419


namespace find_a_and_lambda_find_inverse_of_A_l644_644684

open Matrix

def A (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, a], ![-1, 4]]

def v : Vector (Fin 2) ℝ := ![1, 1]

noncomputable def λvalue (a : ℝ) : ℝ :=
  (A a).mulVec v 0 -- Extract the λ value

theorem find_a_and_lambda : 
  ∀ (a : ℝ), A a.mulVec v = (λvalue a) • v → a = 2 ∧ λvalue 2 = 3 :=
by sorry

def A_2 : Matrix (Fin 2) (Fin 2) ℝ := A 2

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ := inverse (A 2)

theorem find_inverse_of_A : A_inv = ![
  ![(2 / 3), (-1 / 3)],
  ![(1 / 6), (1 / 6)]
] :=
by sorry

end find_a_and_lambda_find_inverse_of_A_l644_644684


namespace perimeter_after_adding_tiles_l644_644414

-- Initial perimeter given
def initial_perimeter : ℕ := 20

-- Number of initial tiles
def initial_tiles : ℕ := 10

-- Number of additional tiles to be added
def additional_tiles : ℕ := 2

-- New tile side must be adjacent to an existing tile
def adjacent_tile_side : Prop := true

-- Condition about the tiles being 1x1 squares
def sq_tile (n : ℕ) : Prop := n = 1

-- The perimeter should be calculated after adding the tiles
def new_perimeter_after_addition : ℕ := 19

theorem perimeter_after_adding_tiles :
  ∃ (new_perimeter : ℕ), 
    new_perimeter = 19 ∧ 
    initial_perimeter = 20 ∧ 
    initial_tiles = 10 ∧ 
    additional_tiles = 2 ∧ 
    adjacent_tile_side ∧ 
    sq_tile 1 :=
sorry

end perimeter_after_adding_tiles_l644_644414


namespace range_of_a_l644_644101

theorem range_of_a (a x : ℝ) (h : x - a = 1 - 2*x) (non_neg_x : x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l644_644101


namespace rectangular_prism_parallel_edges_l644_644718

theorem rectangular_prism_parallel_edges :
  ∀ (l w h : ℕ), l ≠ w ∧ w ≠ h ∧ l ≠ h → 
  let prism_edges := {l, w, h} in 
  (number_of_parallel_pairs prism_edges = 12) :=
by
  sorry

end rectangular_prism_parallel_edges_l644_644718


namespace arithmetic_geometric_sequence_a4_value_l644_644283

theorem arithmetic_geometric_sequence_a4_value 
  (a : ℕ → ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4) : 
  a 4 = 1 := 
sorry

end arithmetic_geometric_sequence_a4_value_l644_644283


namespace sum_of_leftmost_rightmost_l644_644638

theorem sum_of_leftmost_rightmost (A B C D E : ℕ) 
  (h1 : A + B + C + D + E = 35)
  (h2 : B = 7)
  (h3 : B + C = 13)
  (h4 : A + D + 6 + 7 = 31) 
  (h5 : E + 6 + 7 = 21)
  (h6 : h_AE : A + E = 18) 
  : A + B = 11 :=
by
  sorry

end sum_of_leftmost_rightmost_l644_644638


namespace min_dwarfs_l644_644073

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l644_644073


namespace sum_of_squares_of_roots_l644_644830

theorem sum_of_squares_of_roots : 
  ∃ x1 x2 : ℝ, 
    (5 * x1^2 + 8 * x1 - 7 = 0) ∧ 
    (5 * x2^2 + 8 * x2 - 7 = 0) ∧ 
    (5 * (x1 ≠ x2)) ∧
    (x1^2 + x2^2 = 134/25) :=
begin
  sorry
end

end sum_of_squares_of_roots_l644_644830


namespace AX_plus_XB_eq_a_l644_644657

noncomputable def construct_point_X (l : Line) (A B : Point) (a : ℝ) : Point :=
  sorry

theorem AX_plus_XB_eq_a (l : Line) (A B : Point) (a : ℝ) :
  ∃ X : Point, X ∈ l ∧ dist A X + dist X B = a :=
sorry

end AX_plus_XB_eq_a_l644_644657


namespace finite_zero_addition_l644_644782

theorem finite_zero_addition (a b : ℕ) (h_distinct : a ≠ b) :
  ∃ k : ℕ, ∀ n > k, ¬(∃ seq : List ℕ, count_occurrences seq 0 = n) :=
by
  sorry

/--
Helper function to count occurrences of a number in a list.
-/
def count_occurrences {α : Type} [DecidableEq α] (l : List α) (a : α) : ℕ :=
  l.count a

end finite_zero_addition_l644_644782


namespace minimum_dwarfs_l644_644045

theorem minimum_dwarfs (n : ℕ) (C : ℕ → Prop) (h_nonempty : ∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :
  ∃ m, 10 ≤ m ∧ (∀ i, ∃ j, j = (i + 1) % 30 ∨ j = (i + 2) % 30 ∨ j = (i + 3) % 30 → C j) :=
sorry

end minimum_dwarfs_l644_644045


namespace carmen_parsley_left_l644_644958

theorem carmen_parsley_left (plates_whole_sprig : ℕ) (plates_half_sprig : ℕ) (initial_sprigs : ℕ) :
  plates_whole_sprig = 8 →
  plates_half_sprig = 12 →
  initial_sprigs = 25 →
  initial_sprigs - (plates_whole_sprig + plates_half_sprig / 2) = 11 := by
  intros
  sorry

end carmen_parsley_left_l644_644958


namespace factor_polynomial_l644_644247

variable (x : ℝ)

theorem factor_polynomial : (270 * x^3 - 90 * x^2 + 18 * x) = 18 * x * (15 * x^2 - 5 * x + 1) :=
by 
  sorry

end factor_polynomial_l644_644247


namespace locus_of_centers_is_perpendicular_bisector_l644_644979

-- Declare the fixed points A and B and the radius r
variables (A B : Point) (r : ℝ)

-- Assume the distance between A and B is 2r
axiom dist_AB_eq_2r : dist A B = 2 * r

-- The locus is defined as the perpendicular bisector of segment AB
def perpendicular_bisector (A B : Point) : Set Point := { P | dist P A = dist P B }

-- The main statement
theorem locus_of_centers_is_perpendicular_bisector : 
  ∀ (C : Circle), C.radius = r ∧ C.contains A ∧ C.contains B → ∃ P, P ∈ perpendicular_bisector A B := sorry

end locus_of_centers_is_perpendicular_bisector_l644_644979


namespace part1_part2_l644_644682

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then 4 * x - x^2 else if x ≤ -1 then 4 * x + x^2 else 0

theorem part1 : ∀ x, x ≤ -1 → f x = 4 * x + x^2 := 
by
  intro x hx
  simp [f, hx]
  have h1 : f (-x) = -(4 * x - x^2), from sorry
  have h2 : f (-x) = 4 * (-x) + (-x)^2, from sorry
  rw [yes h1, h2]
  sorry

noncomputable def g (x : ℝ) : ℝ := (f x - 9) / x

theorem part2 : ∀ x, 
  (x ≥ 1 ∨ x ≤ -1) → 
  (∃ y, g y ∈ (@Set.Icc ℝ _ (-∞) 12)) ∧
  (∀ x, 
     x < 1 → x > -1 → g x ∉ (@Set.Icc ℝ _ (-∞) 12)) :=
by
  intro x hx
  use sorry
  split
  use sorry
  intro x hx1 hx2
  simp [g, f, hx1, hx2]
  sorry

end part1_part2_l644_644682


namespace rudolph_miles_traveled_l644_644393

def stop_signs : ℕ := 17 - 3
def stop_signs_per_mile : ℕ := 2

theorem rudolph_miles_traveled : stop_signs / stop_signs_per_mile = 7 :=
by
  have h : stop_signs = 14 := by rfl
  have h1 : stop_signs_per_mile = 2 := by rfl
  calc
    14 / 2 = 7 : by norm_num

end rudolph_miles_traveled_l644_644393


namespace hyper_ant_returns_l644_644082

-- Define the problem conditions
variable (R4 : Type) [AddCommGroup R4] [Module ℤ R4]

def step : R4 := sorry -- Placeholder for the Δ vector, each move ±2 in one dimension

-- Define the main statement we want to prove
theorem hyper_ant_returns : 
  let steps := 6
  let origin := (0 : R4)
  let condition (point : R4) := sorry -- Placeholder for the condition: point is at the origin after exactly 6 steps, each move 2 units away
  (number_of_ways_to_return steps origin condition = 725568) :=
sorry

end hyper_ant_returns_l644_644082


namespace correct_equation_l644_644481

theorem correct_equation:
  (∀ x y : ℝ, -5 * (x - y) = -5 * x + 5 * y) ∧ 
  (∀ a c : ℝ, ¬ (-2 * (-a + c) = -2 * a - 2 * c)) ∧ 
  (∀ x y z : ℝ, ¬ (3 - (x + y + z) = -x + y - z)) ∧ 
  (∀ a b : ℝ, ¬ (3 * (a + 2 * b) = 3 * a + 2 * b)) :=
by
  sorry

end correct_equation_l644_644481


namespace largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l644_644555

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l644_644555


namespace nonneg_integer_solutions_l644_644516

theorem nonneg_integer_solutions :
  { x : ℕ | 5 * x + 3 < 3 * (2 + x) } = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_l644_644516


namespace projection_in_direction_neg_a_l644_644712

variables {V : Type*} [inner_product_space ℝ V] {a b : V}

theorem projection_in_direction_neg_a (h1 : ∥a∥ = 1) (h2 : ⟪a, b⟫ = 0) :
  (-(⟪a - (2 : ℝ) • b, a⟫ / ∥a∥)) = -1 :=
by sorry

end projection_in_direction_neg_a_l644_644712


namespace height_of_triangle_is_5_l644_644815

def base : ℝ := 4
def area : ℝ := 10

theorem height_of_triangle_is_5 :
  ∃ (height : ℝ), (base * height) / 2 = area ∧ height = 5 :=
by
  sorry

end height_of_triangle_is_5_l644_644815


namespace rearrange_segments_to_form_segment_of_naturals_l644_644121

theorem rearrange_segments_to_form_segment_of_naturals (a b : ℕ) :
  ∃ (c : ℕ) (s1 s2 : Fin 1961 → ℕ), 
    (∀ i : Fin 1961, s1 i = a + i) ∧ 
    (∀ i : Fin 1961, b + i ∈ Finset.univ (Fin 1961) → ∃ j, s2 j = b + i) ∧ 
    (∀ i : Fin 1961, s1 i + s2 i = c + i) :=
by
  sorry

end rearrange_segments_to_form_segment_of_naturals_l644_644121


namespace non_similar_triangles_in_decagon_l644_644344

-- Define what a regular decagon is
structure RegularDecagon :=
  (vertices : Fin 10 → Point)

-- Define the problem
theorem non_similar_triangles_in_decagon (D : RegularDecagon) : 
  ∃ n, n = 8 ∧ 
  (∀ t1 t2 : Triangle, t1 ≠ t2 → ¬ (similar t1 t2)) :=
sorry

end non_similar_triangles_in_decagon_l644_644344


namespace distinct_primes_not_equal_to_three_l644_644777

theorem distinct_primes_not_equal_to_three {k : ℕ} (p : ℕ → ℕ)
  (hp : ∀ i, p i ∈ Prime) (h_distinct : ∀ i j, i ≠ j → p i ≠ p j) :
  (∏ i in Finset.range k, (1 + 1 / (p i) + 1 / (p i ^ 2))) ≠ 3 :=
by sorry

end distinct_primes_not_equal_to_three_l644_644777


namespace sum_y1_y2_l644_644837

noncomputable def log_base := λ (b x : ℝ) : ℝ, real.log x / real.log b

theorem sum_y1_y2 :
  ∃ (x1 y1 z1 x2 y2 z2 : ℝ), 
    log_base 10 (2000 * x1 * y1) - (log_base 10 x1) * (log_base 10 y1) = 4 ∧
    log_base 10 (2 * y1 * z1) - (log_base 10 y1) * (log_base 10 z1) = 1 ∧
    log_base 10 (z1 * x1) - (log_base 10 z1) * (log_base 10 x1) = 0 ∧
    log_base 10 (2000 * x2 * y2) - (log_base 10 x2) * (log_base 10 y2) = 4 ∧
    log_base 10 (2 * y2 * z2) - (log_base 10 y2) * (log_base 10 z2) = 1 ∧
    log_base 10 (z2 * x2) - (log_base 10 z2) * (log_base 10 x2) = 0 ∧
    y1 + y2 = 25 :=
sorry

end sum_y1_y2_l644_644837


namespace smallest_possible_value_l644_644094

theorem smallest_possible_value (x : ℕ) (m : ℕ) :
  (x > 0) →
  (Nat.gcd 36 m = x + 3) →
  (Nat.lcm 36 m = x * (x + 3)) →
  m = 12 :=
by
  sorry

end smallest_possible_value_l644_644094


namespace avg_speed_correct_l644_644326

noncomputable def avg_speed_round_trip
  (flight_up_speed : ℝ)
  (tailwind_speed : ℝ)
  (tailwind_angle : ℝ)
  (flight_home_speed : ℝ)
  (headwind_speed : ℝ)
  (headwind_angle : ℝ) : ℝ :=
  let effective_tailwind_speed := tailwind_speed * Real.cos (tailwind_angle * Real.pi / 180)
  let ground_speed_to_mother := flight_up_speed + effective_tailwind_speed
  let effective_headwind_speed := headwind_speed * Real.cos (headwind_angle * Real.pi / 180)
  let ground_speed_back_home := flight_home_speed - effective_headwind_speed
  (ground_speed_to_mother + ground_speed_back_home) / 2

theorem avg_speed_correct :
  avg_speed_round_trip 96 12 30 88 15 60 = 93.446 :=
by
  sorry

end avg_speed_correct_l644_644326


namespace second_interest_rate_l644_644035

-- Define the given conditions
def total_amount : ℝ := 2500
def P1 : ℝ := 1000.0000000000005
def P2 : ℝ := total_amount - P1
def total_interest : ℝ := 140
def interest_rate1 : ℝ := 0.05
def interest1 : ℝ := P1 * interest_rate1
def interest2 (r: ℝ) : ℝ := P2 * r / 100

-- Define the second interest rate problem
theorem second_interest_rate :
  ∃ r : ℝ, interest1 + interest2 r = total_interest ∧ r = 6 :=
by
  sorry

end second_interest_rate_l644_644035


namespace horizontal_rows_same_suit_l644_644904

/-- Suppose we have a standard deck of 52 cards arranged in a 13x4 rectangle. 
    If two cards are adjacent either vertically or horizontally, 
    they are either of the same suit or the same rank. 
    Prove that in each horizontal row, all cards are of the same suit. -/
theorem horizontal_rows_same_suit 
  (cards : Matrix (Fin 4) (Fin 13) Card)
  (h_adjacent : ∀ i j, 
                (0 < i ∧ i < 4 ∧ ∃ c1 c2, cards ⟨i - 1, j⟩ = c1 ∧ cards ⟨i, j⟩ = c2 ∧ (c1.rank = c2.rank ∨ c1.suit = c2.suit)) ∨ 
                (i < 3 ∧ ∃ c1 c2, cards ⟨i, j⟩ = c1 ∧ cards ⟨i + 1, j⟩ = c2 ∧ (c1.rank = c2.rank ∨ c1.suit = c2.suit)) ∨ 
                (0 < j ∧ j < 13 ∧ ∃ c1 c2, cards ⟨i, j - 1⟩ = c1 ∧ cards ⟨i, j⟩ = c2 ∧ (c1.rank = c2.rank ∨ c1.suit = c2.suit)) ∨ 
                (j < 12 ∧ ∃ c1 c2, cards ⟨i, j⟩ = c1 ∧ cards ⟨i, j + 1⟩ = c2 ∧ (c1.rank = c2.rank ∨ c1.suit = c2.suit))) : 
  ∀ i : Fin 4, ∃ s : Suit, ∀ j : Fin 13, (cards ⟨i, j⟩).suit = s :=
begin
  sorry
end

end horizontal_rows_same_suit_l644_644904


namespace age_difference_is_18_l644_644108

variable (A B C : ℤ)
variable (h1 : A + B > B + C)
variable (h2 : C = A - 18)

theorem age_difference_is_18 : (A + B) - (B + C) = 18 :=
by
  sorry

end age_difference_is_18_l644_644108


namespace monochromatic_triangle_exists_l644_644471

/-- 
  Given a set of 6 points in the plane where no three points are collinear,
  and each line segment between any two points is colored either candy pink or
  canary yellow, there exists a monochromatic triangle.
 -/
theorem monochromatic_triangle_exists (points : Finset (ℝ × ℝ)) (h_points : points.card = 6)
  (no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → collinear ℝ {p1, p2, p3} → False)
  (coloring : (ℝ × ℝ) → (ℝ × ℝ) → Prop) :
  ∃ (a b c : ℝ × ℝ), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧ 
    coloring a b = coloring b c ∧ coloring b c = coloring c a := 
sorry

end monochromatic_triangle_exists_l644_644471


namespace inequality_proof_l644_644014

-- Definitions for conditions
variable {n : ℕ} (a : ℕ → ℝ) (hsum : ∀ n, 0 < n → (∑ j in Finset.range n, a j) ≥ Real.sqrt n)

-- Positivity of the sequence
variable (apos : ∀ n, 0 ≤ a n)

-- Theorem statement
theorem inequality_proof (n : ℕ) (hn : 1 ≤ n) 
  (h_sum_sqrt : ∀ n, 0 < n → (∑ j in Finset.range n, a j) ≥ Real.sqrt n)
  (a_pos : ∀ n, 0 ≤ a n) :
  (∑ j in Finset.range n, a j ^ 2) > (1 / 4) * (∑ k in Finset.range n, 1 / (k + 1)) :=
sorry

end inequality_proof_l644_644014


namespace calculate_value_expression_l644_644587

theorem calculate_value_expression :
  3000 * (3000 ^ 3000 + 3000 ^ 2999) = 3001 * 3000 ^ 3000 := 
by
  sorry

end calculate_value_expression_l644_644587


namespace green_tea_leaves_needed_l644_644943

-- Constants for the given conditions
constant sprigs_of_mint : ℕ := 3
constant green_tea_leaves_per_sprig : ℕ := 2
constant efficacy_factor : ℕ := 2

-- The theorem to prove
theorem green_tea_leaves_needed : 
  (sprigs_of_mint * green_tea_leaves_per_sprig) * efficacy_factor = 12 := 
by {
  sorry
}

end green_tea_leaves_needed_l644_644943


namespace smallest_perimeter_1000_l644_644518

noncomputable def is_smallest_perimeter (a b : ℕ) (h_area : a * b = 1000) (P : ℕ) :=
  ∀ (a' b' : ℕ), (a' * b' = 1000) → (2 * (a' + b') ≥ P)

theorem smallest_perimeter_1000 :
  ∃ a b : ℕ, a * b = 1000 ∧ is_smallest_perimeter a b 130 :=
begin
  sorry -- Proof will be filled in later
end

end smallest_perimeter_1000_l644_644518


namespace problem1_problem2_l644_644957

-- Problem 1: Calculation
theorem problem1 : -1^(2024) + |(-3)| - (π + 1)^0 = 1 := 
by 
  sorry

-- Problem 2: Solve the Equation
theorem problem2 (x : ℝ) (h : x ≠ -2 ∧ x ≠ 2) : (2 / (x + 2)) = (4 / (x^2 - 4)) → x = 4 := 
by 
  intro h_eq,
  field_simp [h] at h_eq,
  linarith

end problem1_problem2_l644_644957


namespace radius_of_circle_complex_roots_l644_644576

theorem radius_of_circle_complex_roots : ∀ z : ℂ, (z - 1)^6 = 64 * z^6 → abs (z - 1) = sqrt (2 / 3) * abs z :=
by
  intro z h
  sorry

end radius_of_circle_complex_roots_l644_644576


namespace minimum_value_f_maximum_value_f_l644_644653

-- Problem 1: Minimum value of f(x) = 12/x + 3x for x > 0
theorem minimum_value_f (x : ℝ) (h : x > 0) : 
  (12 / x + 3 * x) ≥ 12 :=
sorry

-- Problem 2: Maximum value of f(x) = x(1 - 3x) for 0 < x < 1/3
theorem maximum_value_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 3) :
  x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

end minimum_value_f_maximum_value_f_l644_644653


namespace f_positive_l644_644381

-- condition 1: defining that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)

-- condition 2: defining the function f for negative x
def f (x : ℝ) : ℝ := if x < 0 then 3^x + x else 0 -- we will only use this for x < 0

-- defining the property of f for positive x, which we want to prove
theorem f_positive (f : ℝ → ℝ)
  (h1 : is_odd_function f)
  (h2 : ∀ x, x < 0 → f x = 3^x + x) :
  ∀ x, 0 < x → f x = -3^(-x) + x :=
by
  sorry

end f_positive_l644_644381


namespace water_leakage_distance_l644_644196

/-
  The bucket has an initial side hole, causing it to lose half of the water when 
  travelling distance x, and a bottom hole causing it to lose two-thirds of the water 
  when travelling the same distance x. When both holes are active simultaneously, 
  and she is 1 meter away from the kitchen, only 1/40 of the water is left in the bucket.
  Prove that the distance from the well to the kitchen is approximately 50.91 meters.
-/

theorem water_leakage_distance
  (x : ℝ) -- distance from the well to the kitchen
  (h₁ : 0 < x) -- the distance is positive
  (h₂ : ∀ t ≤ x, volume t = volume 0 * (1 - (1/2) * (t/x))) -- water leaked through the side hole
  (h₃ : ∀ t ≤ x, volume t = volume 0 * (1 - (2/3) * (t/x))) -- water leaked through bottom hole
  (h₄ : volume (x - 1) = volume 0 * (1/40))  -- volume left 1 meter from the kitchen
  : x ≈ 50.91 :=
sorry

end water_leakage_distance_l644_644196


namespace cylinder_lateral_surface_area_l644_644819

-- Define the condition that the cross-section of the cylinder is a square with area 5
def is_square_cross_section (a : ℝ) : Prop :=
a^2 = 5

-- Define the lateral surface area of a cylinder given the height and diameter
def lateral_surface_area (d h : ℝ) : ℝ :=
π * d * h

-- The main theorem to prove
theorem cylinder_lateral_surface_area (d h : ℝ) (h_square : is_square_cross_section h) :
  lateral_surface_area d h = 5 * π :=
by
  -- Assuming the diameter d equals h as both sides of the square are equal to the height in this problem
  rw is_square_cross_section at h_square
  subst_vars
  simp [lateral_surface_area, h_square]
  linarith

end cylinder_lateral_surface_area_l644_644819


namespace Alex_is_26_l644_644200

-- Define the ages as integers
variable (Alex Jose Zack Inez : ℤ)

-- Conditions of the problem
variable (h1 : Alex = Jose + 6)
variable (h2 : Zack = Inez + 5)
variable (h3 : Inez = 18)
variable (h4 : Jose = Zack - 3)

-- Theorem we need to prove
theorem Alex_is_26 (h1: Alex = Jose + 6) (h2 : Zack = Inez + 5) (h3 : Inez = 18) (h4 : Jose = Zack - 3) : Alex = 26 :=
by
  sorry

end Alex_is_26_l644_644200


namespace percentage_left_during_panic_l644_644198

theorem percentage_left_during_panic
  (original_inhabitants remaining_inhabitants_after_panic : ℕ)
  (percentage_disappeared : ℝ)
  (initial_loss : original_inhabitants * percentage_disappeared = 10 / 100 * original_inhabitants)
  (remaining_after_initial_loss : original_inhabitants - original_inhabitants * percentage_disappeared = 7800 - 780)
  (final_population_after_panic : remaining_after_initial_loss - (original_inhabitants - original_inhabitants * percentage_disappeared - 5265) = 5265) :
  ((original_inhabitants - original_inhabitants * percentage_disappeared - 5265) / (original_inhabitants - original_inhabitants * percentage_disappeared) * 100 = 25) :=
sorry

end percentage_left_during_panic_l644_644198


namespace intersection_correct_l644_644785

open Set

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def intersection := (M ∩ N) = {x : ℝ | 1 ≤ x ∧ x < 2}

theorem intersection_correct : intersection := by
  sorry

end intersection_correct_l644_644785


namespace correct_number_of_arrangements_l644_644409

def arrangements_with_conditions (n : ℕ) : ℕ := 
  if n = 6 then
    let case1 := 120  -- when B is at the far right
    let case2 := 96   -- when A is at the far right
    case1 + case2
  else 0

theorem correct_number_of_arrangements : arrangements_with_conditions 6 = 216 :=
by {
  -- The detailed proof is omitted here
  sorry
}

end correct_number_of_arrangements_l644_644409


namespace proof_problem_l644_644693

-- Given function f
def f (x : ℝ) : ℝ := sin (x + π / 4) ^ 2

-- Define a and b as per problem statement
def a : ℝ := f (real.log 5)
def b : ℝ := f (real.log (1 / 5))

-- Proof statement encapsulating the questions from the problem
theorem proof_problem
  (h1 : a = f (real.log 5))
  (h2 : b = f (real.log (1 / 5))) :
  a + b = 1 ∧ a - b = sin (2 * real.log 5) :=
sorry

end proof_problem_l644_644693


namespace initial_amount_deposit_l644_644168

variable (A : ℝ := 306.18) (r : ℝ := 0.06) (n : ℕ := 12) (t : ℝ := 1/4)

theorem initial_amount_deposit : ∃ P : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ P ≈ 301.63 :=
by
  sorry

end initial_amount_deposit_l644_644168


namespace compute_expression_l644_644596

theorem compute_expression : 45 * 25 + 55 * 45 + 20 * 45 = 4500 :=
by
  calc
    45 * 25 + 55 * 45 + 20 * 45
      = 45 * (25 + 55 + 20) : by rw [← mul_add, ← mul_add]
      ... = 45 * 100       : by norm_num
      ... = 4500           : by norm_num

end compute_expression_l644_644596


namespace triangle_third_side_length_l644_644745

theorem triangle_third_side_length
  (AC BC : ℝ)
  (h_a h_b h_c : ℝ)
  (half_sum_heights_eq : (h_a + h_b) / 2 = h_c) :
  AC = 6 → BC = 3 → AB = 4 :=
by
  sorry

end triangle_third_side_length_l644_644745


namespace polynomial_solution_l644_644624

noncomputable def P (x : ℝ) : ℝ := sorry

theorem polynomial_solution (P : ℝ → ℝ) (h1 : P 0 = 0)
  (h2 : ∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ (a : ℝ), ∀ x : ℝ, P x = a * x :=
begin
  sorry
end

end polynomial_solution_l644_644624


namespace count_ordered_pairs_l644_644717

theorem count_ordered_pairs :
  (card {p : ℕ × ℕ | 0 < p.1 ∧ p.1 < p.2 ∧ p.2 < 2008 ∧ 2008^2 + p.1^2 = 2007^2 + p.2^2}) = 3 :=
sorry

end count_ordered_pairs_l644_644717


namespace num_persons_initially_l644_644418

theorem num_persons_initially (N : ℕ) (avg_weight : ℝ) 
  (h_increase_avg : avg_weight + 5 = avg_weight + 40 / N) :
  N = 8 := by
    sorry

end num_persons_initially_l644_644418


namespace isosceles_triangle_base_length_l644_644751

theorem isosceles_triangle_base_length (A B C D : Point)
  (h_iso : A.dist B = A.dist C)
  (h_median : B.dist D = C.dist D)
  (h_perimeter : A.dist B + B.dist C + C.dist A = 27)
  (h_parts : (A.dist B + B.dist D + D.dist A = 15) ∨ (A.dist C + C.dist D + D.dist A = 15)) :
  B.dist C = 7 ∨ B.dist C = 11 := by
  sorry

end isosceles_triangle_base_length_l644_644751


namespace proof_ab_lt_1_l644_644311

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem proof_ab_lt_1 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a > f b) : a * b < 1 :=
by
  -- Sorry to skip the proof
  sorry

end proof_ab_lt_1_l644_644311


namespace number_of_students_above_120_l644_644926

def normal_distribution (μ σ : ℝ) : Prop := sorry -- Assumes the definition of a normal distribution

-- Given conditions
def number_of_students := 1000
def μ := 100
def some_distribution (ξ : ℝ) := normal_distribution μ (σ^2)
def probability_between (a b : ℝ) : ℝ := sorry -- Assumes the definition to calculate probabilities in normal distribution
def given_probability := (probability_between 80 100) = 0.45

-- Question and proof problem
theorem number_of_students_above_120 (σ : ℝ) (h : some_distribution ξ) (hp : given_probability) :
  ∃ n : ℕ, n = 50 := by
    sorry

end number_of_students_above_120_l644_644926


namespace total_cost_in_dollars_l644_644421

theorem total_cost_in_dollars :
  (500 * 3 + 300 * 2) / 100 = 21 := 
by
  sorry

end total_cost_in_dollars_l644_644421


namespace not_all_zero_iff_at_least_one_non_zero_l644_644442

theorem not_all_zero_iff_at_least_one_non_zero (a b c : ℝ) : ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by
  sorry

end not_all_zero_iff_at_least_one_non_zero_l644_644442


namespace number_of_integer_solutions_l644_644445

-- Define the conditions as separate hypotheses
def condition1 (x : ℤ) : Prop := 12 - 4 * x > -8
def condition2 (x : ℤ) : Prop := x + 3 ≥ 5

-- Define the main theorem to prove the number of integer solutions
theorem number_of_integer_solutions : 
  {x : ℤ // condition1 x ∧ condition2 x}.to_finset.card = 3 :=
by
  sorry

end number_of_integer_solutions_l644_644445


namespace correct_option_c_l644_644869

theorem correct_option_c :
  ( ∀ (x : ℤ), (- ( -  3 ) ^ 2 = 9) ∧
    (-6 / 6 * (1 / 6 : ℚ) = -6) ∧
    ((-3) ^ 2 * | - (1 / 3) | = 3) ∧
    (3 ^ 2 / 2 = 9 / 4) ) → ((-3) ^ 2 * | - (1 / 3 : ℚ) | = 3):=
begin
  intros h,
  exact h.right.left.right.left.right
end

end correct_option_c_l644_644869


namespace probability_of_slope_condition_l644_644379

def is_in_unit_square (Q : (ℝ × ℝ)) : Prop :=
  (0 ≤ Q.1 ∧ Q.1 ≤ 1 ∧ 0 ≤ Q.2 ∧ Q.2 ≤ 1)

def slope_condition_met (Q : (ℝ × ℝ)) : Prop :=
  (Q.2 - (3 / 4)) / (Q.1 - (1 / 4)) ≥ 1

noncomputable def satisfying_region_area_unit_square : ℝ := (1 / 2) * (1 / 2) * (1 / 2)

theorem probability_of_slope_condition :
  ∫ (Q : ℝ × ℝ) in (set_of (λ Q, is_in_unit_square Q)),
    (if slope_condition_met Q then 1 else 0) = (1 / 8) / 1 := sorry

end probability_of_slope_condition_l644_644379


namespace Eva_numbers_l644_644240

theorem Eva_numbers : ∃ (a b : ℕ), a + b = 43 ∧ a - b = 15 ∧ a = 29 ∧ b = 14 :=
by
  sorry

end Eva_numbers_l644_644240


namespace smallest_n_with_conditions_l644_644635

/-- Define divisors function to count the number of divisors a number has -/
def divisors_count (n : ℕ) : ℕ :=
  (List.range (n + 1)).count (λ d, d > 0 ∧ n % d = 0)

/-- Define the function to check if a number has 10 consecutive divisors -/
def has_ten_consecutive_divisors (a : List ℕ) : Prop :=
  ∃ l, l.length = 10 ∧ l = List.range (l.head + 10) ∧ List.all (λ x, x ∈ a)

theorem smallest_n_with_conditions :
  ∃ n, divisors_count n = 144 ∧ has_ten_consecutive_divisors (List.range (n + 1)) ∧ ∀ m, (m < n) → (divisors_count m ≠ 144 ∨ ¬ has_ten_consecutive_divisors (List.range (m + 1))) :=
begin
  use 110880,
  sorry
end

end smallest_n_with_conditions_l644_644635


namespace find_integers_l644_644233

theorem find_integers (n : ℤ) : (n^2 - 13 * n + 36 < 0) ↔ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 :=
by
  sorry

end find_integers_l644_644233


namespace even_function_implies_a_zero_l644_644889

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, (x^2 - |x + a|) = (x^2 - |x - a|)) → a = 0 :=
by
  sorry

end even_function_implies_a_zero_l644_644889


namespace seq_coprime_l644_644156

noncomputable def seq (k : ℕ) : ℕ → ℕ
| 0 := k + 1
| (n + 1) := seq k n ^ 2 - k * seq k n + k

theorem seq_coprime (k : ℕ) (hk : k > 0) (m n : ℕ) (hmn : m ≠ n) : 
  Nat.gcd (seq k m) (seq k n) = 1 :=
by
  sorry

end seq_coprime_l644_644156


namespace skaters_total_hours_l644_644319

-- Define the practice hours based on the conditions
def hannah_weekend_hours := 8
def hannah_weekday_extra_hours := 17
def sarah_weekday_hours := 12
def sarah_weekend_hours := 6
def emma_weekday_hour_multiplier := 2
def emma_weekend_hour_extra := 5

-- Hannah's total hours
def hannah_weekday_hours := hannah_weekend_hours + hannah_weekday_extra_hours
def hannah_total_hours := hannah_weekend_hours + hannah_weekday_hours

-- Sarah's total hours
def sarah_total_hours := sarah_weekday_hours + sarah_weekend_hours

-- Emma's total hours
def emma_weekday_hours := emma_weekday_hour_multiplier * sarah_weekday_hours
def emma_weekend_hours := sarah_weekend_hours + emma_weekend_hour_extra
def emma_total_hours := emma_weekday_hours + emma_weekend_hours

-- Total hours for all three skaters combined
def total_hours := hannah_total_hours + sarah_total_hours + emma_total_hours

-- Lean statement version only, no proof required
theorem skaters_total_hours : total_hours = 86 := by
  sorry

end skaters_total_hours_l644_644319


namespace cyclic_sequence_period_sixteen_l644_644542

theorem cyclic_sequence_period_sixteen (a : ℝ) (h_a_pos : 0 < a) :
  ( ∃ u : ℕ → ℝ, u 1 = a ∧ (∀ n : ℕ, u (n + 1) = -1 / (u n + 1)) ∧ u 16 = a ) :=
begin
  sorry
end

end cyclic_sequence_period_sixteen_l644_644542


namespace marble_remainder_l644_644867

theorem marble_remainder
  (r p : ℕ)
  (h_r : r % 5 = 2)
  (h_p : p % 5 = 4) :
  (r + p) % 5 = 1 :=
by
  sorry

end marble_remainder_l644_644867


namespace sum_geometric_sequence_l644_644302

theorem sum_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h_1 : ∀ k : ℕ, a (k + 1) = a k * r)
(h_2 : a 2 = 2) (h_5 : a 5 = 16) :
  (∑ k in finset.range n, (a k) * (a (k + 1))) = (2 / 3) * ((4:ℝ)^n - 1) :=
sorry

end sum_geometric_sequence_l644_644302


namespace alice_sequence_1200th_digit_l644_644413

theorem alice_sequence_1200th_digit :
  let sequence := [2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] ++ -- one and two-digit numbers
                  [200, 201, ..., 299] ++ -- three-digit numbers (inclusive of all such numbers)
                  [2000, 2001, ..., 2999] -- four-digit numbers
  in (sequence.drop 1197).take 3 = [2, 2, 0] :=
by
  sorry

end alice_sequence_1200th_digit_l644_644413


namespace alcohol_percentage_after_adding_water_l644_644161

variable (original_volume : ℝ) (original_alcohol_percentage : ℝ) (added_water_volume : ℝ)

def initial_alcohol_volume (original_volume original_alcohol_percentage : ℝ) : ℝ :=
  original_volume * original_alcohol_percentage / 100

def initial_water_volume (original_volume original_alcohol_percentage : ℝ) : ℝ :=
  original_volume - initial_alcohol_volume original_volume original_alcohol_percentage

def new_total_volume (original_volume added_water_volume : ℝ) : ℝ :=
  original_volume + added_water_volume

def new_alcohol_percentage (original_volume original_alcohol_percentage added_water_volume : ℝ) : ℝ :=
  (initial_alcohol_volume original_volume original_alcohol_percentage / new_total_volume original_volume added_water_volume) * 100

theorem alcohol_percentage_after_adding_water :
  original_volume = 15 → original_alcohol_percentage = 20 → added_water_volume = 2 →
  new_alcohol_percentage original_volume original_alcohol_percentage added_water_volume = 17.65 :=
by
  intros
  sorry

end alcohol_percentage_after_adding_water_l644_644161


namespace probability_not_D_l644_644882

noncomputable def probability : Type := ℝ

-- Definitions of probabilities
def P_D : probability := 0.60
def P_D_and_not_C : probability := 0.20

-- Proof statement
theorem probability_not_D (P_D : probability) : probability :=
by
  -- Using the fact P(D) + P(not D) = 1
  let P_not_D := 1 - P_D
  have h : P_not_D = 0.40 := sorry
  exact P_not_D

end probability_not_D_l644_644882


namespace median_games_l644_644924

def games : List ℕ := [2, 6, 8, 9, 13, 16, 16, 17, 20, 20]

theorem median_games :
  let sorted_games := games -- games are already sorted
  sorted_games.length = 10 →
  (sorted_games.nth 4).getD 0 = 13 →
  (sorted_games.nth 5).getD 0 = 16 →
  ((sorted_games.nth 4).getD 0 + (sorted_games.nth 5).getD 0) / 2 = 14.5 :=
begin
  intro sorted_games, simp,
  intros hlen h5th h6th,
  rw [h5th, h6th],
  norm_num,
end

end median_games_l644_644924


namespace intersection_angle_l644_644210

theorem intersection_angle : ∃ (ω : ℝ), 
  let x := 2 in
  let y := 2 * Real.sqrt 3 in
  let m1 := (3 / y) in
  let m2 := -(x / y) in
  ω = Real.atan (abs ((m1 - m2) / (1 + m1 * m2))) * (180 / Real.pi) 
  ∧ ω ≈ 70.9 := -- 70 degrees and 54 minutes approx 70.9 degrees
by 
  sorry

end intersection_angle_l644_644210


namespace chains_of_sets_l644_644364

open Finset

noncomputable def M : Finset ℕ := range 6

theorem chains_of_sets (M: Finset ℕ) (h : M.card = 6):
    ∃ chains : Finset (Finset ℕ), chains.card = 43200 ∧ 
               (∀ c ∈ chains, ∃ (A B C D : Finset ℕ), {∅, A, B, C, D, M}.to_finset = c) := 
sorry

end chains_of_sets_l644_644364


namespace johnny_marble_choice_l644_644763

/-- Johnny has 9 different colored marbles and always chooses 1 specific red marble.
    Prove that the number of ways to choose four marbles from his bag is 56. -/
theorem johnny_marble_choice : (Nat.choose 8 3) = 56 := 
by
  sorry

end johnny_marble_choice_l644_644763


namespace find_other_endpoint_l644_644825

theorem find_other_endpoint (x1 y1 x2 y2 xm ym : ℝ)
  (midpoint_formula_x : xm = (x1 + x2) / 2)
  (midpoint_formula_y : ym = (y1 + y2) / 2)
  (h_midpoint : xm = -3 ∧ ym = 2)
  (h_endpoint : x1 = -7 ∧ y1 = 6) :
  x2 = 1 ∧ y2 = -2 := 
sorry

end find_other_endpoint_l644_644825


namespace domain_f_x_minus_1_l644_644680

theorem domain_f_x_minus_1 {f : ℝ → ℝ} (h : ∀ x, (2 * x + 1) ∈ set.Icc (-3 : ℝ) 3 → x ∈ set.Icc (-4 : ℝ) 8):
  ∀ x, (x - 1) ∈ set.Icc (-3 : ℝ) 3 → x ∈ set.Icc (-4 : ℝ) 8 := by
  sorry

end domain_f_x_minus_1_l644_644680


namespace probability_odd_number_l644_644431

theorem probability_odd_number 
  (digits : Finset ℕ) 
  (odd_digits : Finset ℕ)
  (total_digits : ∀d, d ∈ digits → d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9)
  (total_odd_digits : ∀d, d ∈ odd_digits → d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9) :
  card odd_digits / card digits = 4 / 5 :=
by sorry

end probability_odd_number_l644_644431


namespace attended_college_percentage_l644_644197

variable (total_boys : ℕ) (total_girls : ℕ) (percent_not_attend_boys : ℕ) (percent_not_attend_girls : ℕ)

def total_boys_attended_college (total_boys percent_not_attend_boys : ℕ) : ℕ :=
  total_boys - percent_not_attend_boys * total_boys / 100

def total_girls_attended_college (total_girls percent_not_attend_girls : ℕ) : ℕ :=
  total_girls - percent_not_attend_girls * total_girls / 100

noncomputable def total_student_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_boys_attended_college total_boys percent_not_attend_boys +
  total_girls_attended_college total_girls percent_not_attend_girls

noncomputable def percent_class_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_student_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls * 100 /
  (total_boys + total_girls)

theorem attended_college_percentage :
  total_boys = 300 → total_girls = 240 → percent_not_attend_boys = 30 → percent_not_attend_girls = 30 →
  percent_class_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls = 70 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end attended_college_percentage_l644_644197


namespace time_for_B_alone_l644_644878

variable (W : ℝ)

-- Conditions
def A_rate := W / 4
def B_and_C_rate := W / 3
def A_and_C_rate := W / 2

theorem time_for_B_alone :
  ∃ (B_time : ℝ), B_time = 12 :=
by
  -- Define B_rate and C_rate based on given conditions
  let A_rate := A_rate W
  let B_and_C_rate := B_and_C_rate W
  let A_and_C_rate := A_and_C_rate W

  -- Calculate individual rates
  let C_rate := A_and_C_rate - A_rate
  let B_rate := B_and_C_rate - C_rate

  -- Calculate time for B to complete the work alone
  let B_time := W / B_rate

  -- Prove that B_time is 12
  have : B_time = 12 := sorry

  exact ⟨B_time, this⟩

end time_for_B_alone_l644_644878


namespace option_C_correct_l644_644485

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l644_644485


namespace machinery_spending_l644_644765

theorem machinery_spending (total_amount raw_materials_cash : ℝ) (cash_spent_percentage : ℝ) :
    total_amount = 1000 → 
    raw_materials_cash = 500 → 
    cash_spent_percentage = 0.1 → 
    ∃ machinery_spending : ℝ, 
        machinery_spending = total_amount - (raw_materials_cash + cash_spent_percentage * total_amount) ∧ 
        machinery_spending = 400 :=
by 
    intros h1 h2 h3
    use total_amount - (raw_materials_cash + cash_spent_percentage * total_amount)
    split
    case_left
    { 
        exact rfl 
    }
    case_right
    { 
        rw [h1, h2, h3]
        norm_num 
    }

end machinery_spending_l644_644765


namespace probability_of_divisible_palindrome_l644_644177

def is_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000   -- a
  let d2 := (n % 10000) / 1000  -- b
  let d3 := (n % 1000) / 100    -- c
  let d4 := (n % 100) / 10      -- b
  let d5 := n % 10              -- a
  (d1 = d5) ∧ (d2 = d4) ∧ (d1 ≠ 0)

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_divisible_by_3 (n : ℕ) : Prop :=
  (n % 3) = 0

def count_valid_palindromes : ℕ :=
  let count := (Finset.range 10).sum (λ a,
    if a ≠ 0 then (Finset.range 10).sum (λ b,
      (Finset.filter is_prime_digit (Finset.range 10)).count (λ c,
        is_divisible_by_3 (2 * a + 2 * b + c)))
    else 0)
  count

def total_palindromes : ℕ :=
  900

noncomputable def probability_valid_palindrome : ℚ :=
  count_valid_palindromes.to_rat / total_palindromes.to_rat

theorem probability_of_divisible_palindrome :
  probability_valid_palindrome = 2 / 15 := by
  sorry

end probability_of_divisible_palindrome_l644_644177


namespace count_cubes_with_icing_on_two_sides_l644_644967

-- Define the parameters of the problem
def cake_side_length : Nat := 5

def is_icing_on_side (x y z : Nat) : Prop :=
  x = 0 ∨ x = 4 ∨ y = 0 ∨ y = 4 ∨ z = 0 ∨ z = 4

def has_icing_on_two_sides (x y z : Nat) : Prop :=
  (cond1 ∧ cond2 ∧ ¬cond3)
  where cond1 := is_icing_on_side x y z
        cond2 := if x = 0 ∨ x = 4 then y ≠ 0 ∧ y ≠ 4 ∧ z ≠ 0 ∧ z ≠ 4 else if y = 0 ∨ y = 4 then x ≠ 0 ∧ x ≠ 4 ∧ z ≠ 0 ∧ z ≠ 4 else x ≠ 0 ∧ x ≠ 4 ∧ y ≠ 0 ∧ y ≠ 4
        cond3 := x = 0 ∨ x = 4 ∧ y = 0 ∨ y = 4 ∧ z = 0 ∨ z = 4

theorem count_cubes_with_icing_on_two_sides : 
  (Finset.univ.filter (λ (x y z : Fin Nat cake_side_length), has_icing_on_two_sides x.val y.val z.val)).card = 40 
:= 
  sorry

end count_cubes_with_icing_on_two_sides_l644_644967


namespace ellipse_standard_equation_1_ellipse_standard_equation_2_l644_644259

-- Problem 1
theorem ellipse_standard_equation_1 : 
  (minor_axis : ℝ) (foci_distance : ℝ)
  (h_minor_axis : minor_axis = 6)
  (h_foci_distance : foci_distance = 8) : 
  (eqn_standard : String) :=
  eqn_standard = "x^2 / 25 + y^2 / 9 = 1" :=
sorry

-- Problem 2
theorem ellipse_standard_equation_2 : 
  (e : ℝ) (P : ℝ × ℝ)
  (h_eccentricity : e = sqrt 3 / 2)
  (h_point : P = (4, 2 * sqrt 3)) : 
  (eqn_standard : String) :=
  eqn_standard = "x^2 / 8 + y^2 / 2 = 1" :=
sorry

end ellipse_standard_equation_1_ellipse_standard_equation_2_l644_644259


namespace find_f_f_neg2_l644_644307

def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 3 then 2 ^ x
  else Real.log x

theorem find_f_f_neg2 : f (f (-2)) = 1 := by
  sorry

end find_f_f_neg2_l644_644307


namespace trig_identity_l644_644983

theorem trig_identity :
  (2 * real.cos (10 * real.pi / 180) - real.sin (20 * real.pi / 180)) / real.sin (70 * real.pi / 180) = real.sqrt 3 :=
by {
  sorry
}

end trig_identity_l644_644983


namespace determine_weights_l644_644839

-- Define conditions of the problem using Lean syntax.
variable {W : Type} [linear_ordered_add_comm_group W]
variable (weights : Fin 5 → W)
variable (same_weight : Fin 3 → W)
variable (heaviest_weight lightest_weight : W)

-- Conditions: Three weights are the same, one heavier, one lighter
axiom (h1 : weights 0 = same_weight 0)
axiom (h2 : weights 1 = same_weight 1)
axiom (h3 : weights 2 = same_weight 2)
axiom (h4 : heaviest_weight > same_weight 0)
axiom (h5 : lightest_weight < same_weight 0)

-- Proposition: Determine the heaviest and the lightest weights in three weighings.
theorem determine_weights : ∃ l h : W, 
  (∀i, (weights i = lightest_weight → i ∈ {0, 1, 2, 3, 4} ∧ ∀j ∈ {0, 1, 2, 3, 4}, weights j ≠ lightest_weight)) ∧
  (∀i, (weights i = heaviest_weight → i ∈ {0, 1, 2, 3, 4} ∧ ∀j ∈ {0, 1, 2, 3, 4}, weights j ≠ heaviest_weight)).
Proof. sorry

end determine_weights_l644_644839


namespace necessary_condition_l644_644911

theorem necessary_condition (a : ℝ) : (0 ≤ a ∧ a < 1) → (discriminant (2 * a) (-1) 1) < 0 := sorry

end necessary_condition_l644_644911


namespace ratio_of_areas_l644_644174

theorem ratio_of_areas :
  ∀ (r : ℝ),
  let original_square_area := (2 * (sqrt 2) * r)^2,
      smallest_circle_radius := r * (sqrt 3) / 2,
      smallest_circle_area := π * smallest_circle_radius^2,
      ratio := smallest_circle_area / original_square_area
  in
  ratio = 3 * π / 32 :=
by sorry

end ratio_of_areas_l644_644174


namespace simplify_P_eq_l644_644604

noncomputable def P (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y) - (x * y - y^2) / (x * y - x^2)

theorem simplify_P_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy: x ≠ y) : P x y = x / y := 
by
  -- Insert proof here
  sorry

end simplify_P_eq_l644_644604


namespace find_parabola_equation_l644_644685

noncomputable def circle_C1 := ∀ x y : ℝ, x^2 + (y - 2)^2 = 4
noncomputable def parabola_C2 (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x

theorem find_parabola_equation
  (p : ℝ)
  (h1 : circle_C1 0 0)
  (h2 : circle_C1 (8/5) (16/5))
  (h3 : (2 * 8/5 - 0)^2 + (16/5 - 0)^2 = (8*sqrt(5)/5)^2) 
  : p = 16/5 → parabola_C2 (32/5) :=
by
  intro hp_val
  rw [parabola_C2]
  sorry

end find_parabola_equation_l644_644685


namespace maple_ridge_problem_l644_644242

theorem maple_ridge_problem :
  let students_per_class := 15
  let rabbits_per_class := 3
  let guinea_pigs_per_class := 5
  let number_of_classes := 5
  let total_students := students_per_class * number_of_classes
  let total_rabbits := rabbits_per_class * number_of_classes
  let total_guinea_pigs := guinea_pigs_per_class * number_of_classes
  (total_students - total_rabbits - total_guinea_pigs) = 35 :=
by
  let students_per_class := 15
  let rabbits_per_class := 3
  let guinea_pigs_per_class := 5
  let number_of_classes := 5
  let total_students := students_per_class * number_of_classes
  let total_rabbits := rabbits_per_class * number_of_classes
  let total_guinea_pigs := guinea_pigs_per_class * number_of_classes
  calc
    (total_students - total_rabbits - total_guinea_pigs) = 
      (15 * 5 - 3 * 5 - 5 * 5) := by rw [total_students, total_rabbits, total_guinea_pigs]
    ... = 35 : by norm_num

end maple_ridge_problem_l644_644242


namespace gcd_is_not_one_for_two_segments_l644_644144

theorem gcd_is_not_one_for_two_segments (a : ℕ) (b : ℕ) (h1 : a = 19) (h2 : b = 190) : 
  ¬ (Nat.gcd a b = 1 ∧ 1 = Nat.greatestCommonMeasure a b) :=
by
  sorry

end gcd_is_not_one_for_two_segments_l644_644144


namespace rational_root_divides_value_l644_644029

noncomputable def is_irreducible_fraction (p q : ℤ) : Prop :=
  nat.coprime (int.natAbs p) (int.natAbs q)

variables {a : ℕ → ℤ} {n : ℕ} {p q k : ℤ} (hk : k ∈ ℤ)

theorem rational_root_divides_value 
  (h_fraction_irreducible : is_irreducible_fraction p q) 
  (h_root : (∑ i in finset.range (n + 1), a i * (p ^ (n - i)) * (q ^ i)) = 0)
  : (p - k * q) ∣ (∑ i in finset.range (n + 1), a i * k ^ (n - i)) :=
sorry

end rational_root_divides_value_l644_644029


namespace total_bad_carrots_and_tomatoes_l644_644850

theorem total_bad_carrots_and_tomatoes 
  (vanessa_carrots : ℕ := 17)
  (vanessa_tomatoes : ℕ := 12)
  (mother_carrots : ℕ := 14)
  (mother_tomatoes : ℕ := 22)
  (brother_carrots : ℕ := 6)
  (brother_tomatoes : ℕ := 8)
  (good_carrots : ℕ := 28)
  (good_tomatoes : ℕ := 35) :
  (vanessa_carrots + mother_carrots + brother_carrots - good_carrots) + 
  (vanessa_tomatoes + mother_tomatoes + brother_tomatoes - good_tomatoes) = 16 := 
by
  sorry

end total_bad_carrots_and_tomatoes_l644_644850


namespace terminating_decimal_expansion_7_over_625_l644_644261

theorem terminating_decimal_expansion_7_over_625 : (7 / 625 : ℚ) = 112 / 10000 := by
  sorry

end terminating_decimal_expansion_7_over_625_l644_644261


namespace option_C_correct_l644_644482

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l644_644482


namespace sqrt_2_inv_add_cos_45_eq_sqrt_2_l644_644588

theorem sqrt_2_inv_add_cos_45_eq_sqrt_2 :
  {(\sqrt{2})}^{-1} + Real.cos (Float.pi / 4) = \sqrt{2} := sorry

end sqrt_2_inv_add_cos_45_eq_sqrt_2_l644_644588


namespace sum_first_10_terms_sequence_l644_644453

theorem sum_first_10_terms_sequence:
  let sequence (n : ℕ) := (2 * n - 1) / (2: ℝ)^n 
  in (∑ n in finset.range 10, sequence (n+1)) = 3049 / 1024 := 
by
  sorry

end sum_first_10_terms_sequence_l644_644453


namespace area_of_RPSY_l644_644120

-- Definitions of the points and calculations based on the problem conditions
variables (X Y Z P Q R S : Point) (XY XZ XQ YZ : ℝ)
variable (area_XYZ : ℝ)
variable (m : ℝ)
variable (n : ℝ)

-- Given conditions
axiom XY_eq_60 : XY = 60
axiom XZ_eq_20 : XZ = 20
axiom area_XYZ_eq_240 : area_XYZ = 240
axiom P_midpoint_XY : P = midpoint X Y
axiom XQ_eq_3QZ : XQ = 3 * (XZ - XQ)
axiom angle_bisector_XYZ_intersects_PQ_at_R : 
  is_angle_bisector (X, Y, Z) (P, Q) R
axiom angle_bisector_XYZ_intersects_YZ_at_S : 
  is_angle_bisector (X, Y, Z) (Y, Z) S

-- Desired result: to prove the area of the quadrilateral RPSY is 202.5
theorem area_of_RPSY (XY : ℝ) (XZ : ℝ) (area_XYZ : ℝ)
  (P Q R S : Point) : 
  XY = 60 → 
  XZ = 20 → 
  area_XYZ = 240 → 
  P = midpoint X Y → 
  XQ = 3 * (XZ - XQ) → 
  is_angle_bisector (X, Y, Z) (P, Q) R → 
  is_angle_bisector (X, Y, Z) (Y, Z) S → 
  area_RPSY = 202.5 := 
by 
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end area_of_RPSY_l644_644120


namespace peanuts_remaining_l644_644116

def initial_peanuts : ℕ := 675
def brock_fraction_eaten : ℚ := 1 / 3
def families : ℕ := 3
def bonita_fraction_given : ℚ := 2 / 5
def carlos_fraction_eaten : ℚ := 1 / 5

theorem peanuts_remaining : 
  let brock_peanuts_eaten := (initial_peanuts : ℚ) * brock_fraction_eaten,
      peanuts_left_after_brock := initial_peanuts - brock_peanuts_eaten,
      peanuts_per_family := peanuts_left_after_brock / families,
      bonita_peanuts_given := peanuts_per_family * bonita_fraction_given,
      peanuts_left_per_family := peanuts_per_family - bonita_peanuts_given,
      total_peanuts_left := peanuts_left_per_family * families,
      carlos_peanuts_eaten := total_peanuts_left * carlos_fraction_eaten,
      peanuts_remaining := total_peanuts_left - carlos_peanuts_eaten
  in peanuts_remaining = 216 :=
by
  sorry

end peanuts_remaining_l644_644116


namespace work_done_by_student_l644_644525

theorem work_done_by_student
  (M : ℝ)  -- mass of the student
  (m : ℝ)  -- mass of the stone
  (h : ℝ)  -- height from which the stone is thrown
  (L : ℝ)  -- distance on the ice where the stone lands
  (g : ℝ)  -- acceleration due to gravity
  (t : ℝ := Real.sqrt (2 * h / g))  -- time it takes for the stone to hit the ice derived from free fall equation
  (Vk : ℝ := L / t)  -- initial speed of the stone derived from horizontal motion
  (Vu : ℝ := m / M * Vk)  -- initial speed of the student derived from conservation of momentum
  : (1/2 * m * Vk^2 + (1/2) * M * Vu^2) = 126.74 :=
by
  sorry

end work_done_by_student_l644_644525


namespace points_concyclic_iff_l644_644370

variables {A B C H O' N D : Type}
variables (a b c R : ℝ)

-- Definitions based on conditions:
def is_orthocenter (H : Type) (△ : Type) : Prop := sorry
def is_circumcenter (O' : Type) (△ : Type) : Prop := sorry
def is_midpoint (N : Type) (A O' : Type) : Prop := sorry
def is_reflection (D : Type) (N BC : Type) : Prop := sorry
def are_concyclic (A B D C : Type) : Prop := sorry

-- Formal statement of the theorem:
theorem points_concyclic_iff (h_orthocenter : is_orthocenter H (triangle A B C))
                             (o'_circumcenter : is_circumcenter O' (triangle B H C))
                             (n_midpoint : is_midpoint N A O')
                             (d_reflection : is_reflection D N (segment B C))
                             (a_bc : a = segment B C)
                             (b_ca : b = segment C A)
                             (c_ab : c = segment A B)
                             (R_circumradius : R = circumradius (triangle A B C)) :
  (are_concyclic A B D C) ↔ (b^2 + c^2 - a^2 = 3 * R^2) :=
  sorry

end points_concyclic_iff_l644_644370


namespace right_triangle_circum_inradius_sum_l644_644111

theorem right_triangle_circum_inradius_sum
  (a b : ℕ)
  (h1 : a = 16)
  (h2 : b = 30)
  (h_triangle : a^2 + b^2 = 34^2) :
  let c := 34
  let R := c / 2
  let A := a * b / 2
  let s := (a + b + c) / 2
  let r := A / s
  R + r = 23 :=
by
  sorry

end right_triangle_circum_inradius_sum_l644_644111


namespace digits_of_9984_l644_644858

theorem digits_of_9984 : (nat_digits 9984).length = 4 :=
sorry

end digits_of_9984_l644_644858


namespace black_hole_2004_l644_644458

def count_even_digits (n : ℕ) : ℕ :=
  (to_string n).to_list.count (λ c, c.is_digit && (c - '0').to_nat % 2 = 0)

def count_odd_digits (n : ℕ) : ℕ :=
  (to_string n).to_list.count (λ c, c.is_digit && (c - '0').to_nat % 2 = 1)

def total_digits (n : ℕ) : ℕ :=
  (to_string n).to_list.count (λ c, c.is_digit)

def transform (n : ℕ) : ℕ :=
  100 * (count_even_digits n) + 10 * (count_odd_digits n) + total_digits n

def is_black_hole_number (n : ℕ) : Prop :=
  ∀ m, transform (transform ... (transform n) ... ) = 123

theorem black_hole_2004 : is_black_hole_number 2004 :=
sorry

end black_hole_2004_l644_644458


namespace not_symmetric_wrt_x_axis_l644_644143

theorem not_symmetric_wrt_x_axis (x y : ℝ) :
  ¬ (∀ x y : ℝ, (x^2 - x + y^2 = 1) = (x^2 - x + (-y)^2 = 1) ∧
                (x + y^2 = -1) = (x + (-y)^2 = -1) ∧
                (2x^2 - y^2 = 1) = (2x^2 - (-y)^2 = 1) ∧
                (x^2y + xy^2 = 1) = (x^2 * (-y) + x * (-y)^2 = 1)) :=
by
  sorry

end not_symmetric_wrt_x_axis_l644_644143


namespace smallest_number_with_19_factors_l644_644761

theorem smallest_number_with_19_factors : ∃ n : ℕ, n = 262144 ∧ (∀ m : ℕ, m < n → (number_of_factors m ≠ 19)) ∧ number_of_factors n = 19 := 
by
  sorry

def number_of_factors (n : ℕ) : ℕ :=
  (nat.divisors n).length

-- The helper function calculating the number of factors
#eval number_of_factors 262144 -- should evaluate to 19

end smallest_number_with_19_factors_l644_644761


namespace cross_section_area_of_right_triangular_prism_l644_644437

theorem cross_section_area_of_right_triangular_prism 
  (H : ℝ) (α : ℝ) (hH_pos : 0 < H) (hα_acute : 0 < α ∧ α < π / 2) : 
  let area := (H^2 * real.sqrt 3 * real.cot α) / real.sin α in
  ∃ S, S = area := 
sorry

end cross_section_area_of_right_triangular_prism_l644_644437


namespace cost_per_item_first_batch_l644_644828

theorem cost_per_item_first_batch : 
  ∃ x : ℝ, 
    x > 0 ∧ 
    (∃ n1 n2 : ℝ, 
      n1 = 600 / x ∧ 
      n2 = 2 * n1 ∧ 
      n2 = 1250 / (x + 5)) → x = 120 :=
by
  existsi 120
  split
  · exact zero_lt_one -- We need x to be positive, e.g., x > 0. Here, x > 0.
  · existsi 600 / 120, 1250 / 125
    split
    · norm_num
    split
    · exact sorry -- this establishes the relationship for the first batch
    · exact sorry -- this establishes the relationship for the second batch

end cost_per_item_first_batch_l644_644828


namespace measured_weight_loss_l644_644523

variable (W : ℝ) (hW : W > 0)

noncomputable def final_weigh_in (initial_weight : ℝ) : ℝ :=
  (0.90 * initial_weight) * 1.02

theorem measured_weight_loss :
  final_weigh_in W = 0.918 * W → (W - final_weigh_in W) / W * 100 = 8.2 := 
by
  intro h
  unfold final_weigh_in at h
  -- skip detailed proof steps, focus on the statement
  sorry

end measured_weight_loss_l644_644523


namespace gunny_bag_fill_l644_644025

noncomputable def tons_to_pounds : ℝ := 2200
noncomputable def pounds_to_ounces : ℝ := 16
noncomputable def ounces_to_grams : ℝ := 28.3495
noncomputable def gunny_bag_capacity_tons : ℝ := 13.75
noncomputable def packet_weight_pounds : ℝ := 16
noncomputable def packet_weight_ounces : ℝ := 4
noncomputable def packet_weight_grams : ℝ := 250

-- Prove that the number of packets needed to completely fill the gunny bag is 1375
theorem gunny_bag_fill (
  tons_to_pounds : ℝ := 2200,
  pounds_to_ounces : ℝ := 16,
  ounces_to_grams : ℝ := 28.3495,
  gunny_bag_capacity_tons : ℝ := 13.75,
  packet_weight_pounds : ℝ := 16,
  packet_weight_ounces : ℝ := 4,
  packet_weight_grams : ℝ := 250
) : 
  let 
    gunny_bag_capacity_ounces := gunny_bag_capacity_tons * tons_to_pounds * pounds_to_ounces,
    packet_weight_total_ounces := (packet_weight_pounds * pounds_to_ounces) + packet_weight_ounces + (packet_weight_grams / ounces_to_grams)
  in
  (gunny_bag_capacity_ounces / packet_weight_total_ounces).ceil = 1375 := 
by
  sorry

end gunny_bag_fill_l644_644025


namespace arithmetic_sequence_equality_l644_644352

theorem arithmetic_sequence_equality (a_n : ℕ → ℝ) (a_1 d : ℝ)
  (h1 : ∀ n, a_n n = a_1 + (n - 1) * d)
  (h2 : ∑ i in Finset.range 15, a_n (i + 1) = 90) :
  a_n 8 = 6 :=
by
  sorry

end arithmetic_sequence_equality_l644_644352


namespace tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l644_644646

variable (α : ℝ)
variable (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1 / 4)

theorem tan_alpha_eq_neg2 : Real.tan α = -2 :=
  sorry

theorem sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2 :
  (Real.sin (2 * α) + 1) / (1 + Real.sin (2 * α) + Real.cos (2 * α)) = -1 / 2 :=
  sorry

end tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l644_644646


namespace find_min_max_A_l644_644553

-- Define a 9-digit number B
def is_9_digit (B : ℕ) : Prop := B ≥ 100000000 ∧ B < 1000000000

-- Define a function that checks if a number is coprime with 24
def coprime_with_24 (B : ℕ) : Prop := Nat.gcd B 24 = 1

-- Define the transformation from B to A
def transform (B : ℕ) : ℕ := let b := B % 10 in b * 100000000 + (B / 10)

-- Lean 4 statement for the problem
theorem find_min_max_A :
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 999999998) ∧
  (∃ (B : ℕ), is_9_digit B ∧ coprime_with_24 B ∧ B > 666666666 ∧ transform B = 166666667) :=
  by
    sorry -- Proof is omitted

end find_min_max_A_l644_644553


namespace fixed_point_exists_minimum_quadrilateral_area_l644_644600

open Real

-- Definitions and conditions
def parabola (p : ℝ) (hp : p > 0) (x y : ℝ) := y^2 = 2 * p * x
def line (k m x y : ℝ) := y = k * x + m
def intersects (p k m : ℝ) (hp : p > 0) (x y : ℝ) :=
  parabola p hp x y ∧ line k m x y

def vertex (x y : ℝ) : Prop := x = 0 ∧ y = 0
def fixed_point := ∃ p : ℝ, p > 0 ∧ ∀ (k m x y : ℝ), 
  intersects p k m x y → ∃ fx fy : ℝ, fx = p / 2 ∧ fy = 0 ∧ line k m fx fy

def quadrilateral_area_min (p : ℝ) (hp : p > 0) : Prop :=
  ∀ k m x₁ y₁ x₂ y₂, (intersects p k m x₁ y₁ ∧ intersects p k m x₂ y₂) →
    2 * p^2

-- Lean theorem statement for the given proof problem
theorem fixed_point_exists (p : ℝ) (hp : p > 0) : fixed_point := sorry

theorem minimum_quadrilateral_area (p : ℝ) (hp : p > 0) : quadrilateral_area_min p hp := sorry

end fixed_point_exists_minimum_quadrilateral_area_l644_644600


namespace ineq_sum_of_squares_l644_644678

theorem ineq_sum_of_squares 
  (n : ℕ) 
  (x : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < x i) :
  (Finset.univ.sum (fun i => (x i)^2 / x ((i + 1) % n))) ≥ Finset.univ.sum (fun i => x i) := by 
sorry

end ineq_sum_of_squares_l644_644678


namespace cos_half_pi_plus_alpha_correct_l644_644649

noncomputable def cos_half_pi_plus_alpha
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) : Real :=
  Real.cos (Real.pi / 2 + α)

theorem cos_half_pi_plus_alpha_correct
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  cos_half_pi_plus_alpha α h1 h2 = 3/5 := by
  sorry

end cos_half_pi_plus_alpha_correct_l644_644649


namespace b_ne_d_l644_644404

-- Conditions
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

def PQ_eq_QP_no_real_roots (a b c d : ℝ) : Prop := 
  ∀ (x : ℝ), P (Q x c d) a b ≠ Q (P x a b) c d

-- Goal
theorem b_ne_d (a b c d : ℝ) (h : PQ_eq_QP_no_real_roots a b c d) : b ≠ d := 
sorry

end b_ne_d_l644_644404


namespace roots_of_polynomial_l644_644232

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^2 - 5*x + 6)*(x)*(x-5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 :=
by
  sorry

end roots_of_polynomial_l644_644232


namespace eccentricity_hyperbola_l644_644118

-- Define the hyperbola and its properties
variables {a b m n : ℝ}
variable (P : ℝ × ℝ)
variable (e : ℝ)

-- Assertions based on problem conditions
def hyperbola (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
  ∃ (m n : ℝ), P = (m, n)

def intersects_asymptotes (a b m : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (m, b * m / a) ∧ B = (m, -b * m / a)

def dot_product_condition (a m n b : ℝ) : Prop :=
  (0 * 0 + (b * m / a - n) * (-(b * m / a) - n)) = -a^2 / 4

theorem eccentricity_hyperbola :
  hyperbola a b (m, n) →
  intersects_asymptotes a b m (m, b * m / a) (m, -b * m / a) →
  dot_product_condition a m n b →
  e = sqrt (1 + 1/4) →
  e = sqrt 5 / 2 :=
by {
  intros h1 h2 h3,
  sorry
}

end eccentricity_hyperbola_l644_644118


namespace gen_formulas_sum_first_n_terms_l644_644294

-- Definitions based on conditions
def geom_seq (a : ℕ → ℝ) := ∃ q > 0, ∀ n, a (n + 1) = q * (a n)
def arith_seq (b : ℕ → ℝ) := ∃ d, ∀ n, b (n + 1) = b n + d

-- Given conditions
def a (n : ℕ) := if n = 0 then 1 else 2^(n-1)
def b (n : ℕ) : ℕ → ℝ := if n = 0 then 0 else (2 * n) - 1

-- Proposition to be proved for general formulas
theorem gen_formulas :
  (∀ m, a 0 = 1 ∧ b 1 = 1 ∧ b 2 + b 3 = 2 * a 2 ∧ a 4 - 3 * b 1 = 7) →
  ∀ n, a n = 2^(n-1) ∧ b n = 2 * n - 1 :=
sorry

-- Proposition to be proved for the sum of the sequence
theorem sum_first_n_terms :
  (∀ n, let c := λ n, a n * b n in let S := λ n, (finset.range n).sum c in S n = (2 * n - 3) * 2^n + 3) :=
sorry

end gen_formulas_sum_first_n_terms_l644_644294


namespace correct_operation_l644_644510

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l644_644510


namespace center_of_circle_tangent_to_parallel_lines_l644_644539

-- Define the line equations
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 40
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -20
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- The proof problem
theorem center_of_circle_tangent_to_parallel_lines
  (x y : ℝ)
  (h1 : line1 x y → false)
  (h2 : line2 x y → false)
  (h3 : line3 x y) :
  x = 10 ∧ y = 5 := by
  sorry

end center_of_circle_tangent_to_parallel_lines_l644_644539


namespace outfit_count_l644_644517

theorem outfit_count (shirts pants shoes : ℕ) (h_shirts : shirts = 4) (h_pants : pants = 5) (h_shoes : shoes = 2) : 
  shirts * pants * shoes = 40 :=
by
  rw [h_shirts, h_pants, h_shoes]
  norm_num

end outfit_count_l644_644517


namespace always_piece_with_sum_leq_three_l644_644560

theorem always_piece_with_sum_leq_three (exists_piece : ∀ (n m : ℕ), ∃ (i j : ℕ), (i + j = n + m ∧ (i, j) = (0, 0) ∨ ((i - 1, j) ∈ pieces ∨ (i, j - 1) ∈ pieces) ∧ ∀ k l, (k, l) ≠ (i, j) → (k, l) ∉ pieces) :
      ∃ (a b : ℕ), a + b ≤ 3 :=
  sorry

end always_piece_with_sum_leq_three_l644_644560


namespace verify_highest_possible_value_for_A_l644_644194

noncomputable def highest_possible_value_for_A : ℕ :=
  let is_even (n : ℕ) := n % 2 = 0
  let is_odd (n : ℕ) := n % 2 = 1
  let decreasing_list : List ℕ → Prop := fun l => ∀ i j, i < j → l.nth i > l.nth j
  ∃ (A B C D E F G H I J : ℕ),
    Decreasing (List.of [A, B, C]) ∧ A + B + C = 16 ∧
    ({D, E, F} = {y | is_even y} ∧ D = E + 2 ∧ E = F + 2) ∧
    ({G, H, I, J} = {z | is_odd z} ∧ G = H + 2 ∧ H = I + 2 ∧ I = J + 2)
    ∧ (∀ x ∈ {A,B,C,D,E,F,G,H,I,J}, x ∈ {0,1,2,3,4,5,6,7,8,9}) 
    ∧ A = 9

theorem verify_highest_possible_value_for_A (h : noncomputable def highest_possible_value_for_A : ℕ) :
  h = 9 := sorry

end verify_highest_possible_value_for_A_l644_644194


namespace correct_operation_l644_644492

theorem correct_operation : ∀ (m : ℤ), (-m + 2) * (-m - 2) = m^2 - 4 :=
by
  intro m
  sorry

end correct_operation_l644_644492


namespace value_of_k_if_two_equal_real_roots_l644_644304

theorem value_of_k_if_two_equal_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + k = 0 → x^2 - 2 * x + k = 0) → k = 1 :=
by
  sorry

end value_of_k_if_two_equal_real_roots_l644_644304


namespace adam_and_simon_distance_l644_644933

theorem adam_and_simon_distance :
  ∀ (t : ℝ), (10 * t)^2 + (12 * t)^2 = 16900 → t = 65 / Real.sqrt 61 :=
by
  sorry

end adam_and_simon_distance_l644_644933
