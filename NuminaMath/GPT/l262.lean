import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearAlgebra
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.NormalDistribution
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Subgroup
import Mathlib.LinearAlgebra.Basic
import Mathlib.Logarithm
import Mathlib.NumberTheory.Divisors
import Mathlib.NumberTheory.Floor
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Distribution
import Mathlib.Probability.Independent
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic

namespace range_of_k_l262_262484

theorem range_of_k {k : ℝ} (a_n : ℕ → ℝ) (c_n : ℕ → ℝ) :
  (∀ n, log k (a_n n) = 2 * n + 2) →
  (k > 0 ∧ k ≠ 1) →
  (∀ n, c_n n = a_n n * log k (a_n n)) →
  (∀ n, c_n n < c_n (n + 1)) →
  k ∈ (Set.Ioo 0 (Real.sqrt 6 / 3) ∪ Set.Ioi 1) := sorry

end range_of_k_l262_262484


namespace frame_width_l262_262205

-- Define the given conditions
variables (width height : ℝ) (area_frame : ℝ)
def photo_dimensions := width = 12 ∧ height = 18
def area_photo := width * height = 216
def frame_area := area_frame = width * height

-- Define the proof problem
theorem frame_width (x : ℝ) (h : photo_dimensions width height) 
  (h₁ : area_photo width height) (h₂ : frame_area area_frame) :
  (12 + 2 * x) * (18 + 2 * x) - 216 = 216 →
  x = 3 :=
begin
  -- Proof is required here
  sorry,
end

end frame_width_l262_262205


namespace dot_product_eq_one_l262_262690

open Real

variables (a b : ℝ)

-- Define the norm of a vector
def norm (v : ℝ) : ℝ := sqrt (v * v)

-- Define the given conditions
axiom norm_add_ab : norm (a + b) = sqrt 6
axiom norm_sub_ab : norm (a - b) = sqrt 2

-- Prove the dot product
theorem dot_product_eq_one : a * b = 1 := 
by
  have h1 : (norm (a + b))^2 = 6 := by sorry
  have h2 : (norm (a - b))^2 = 2 := by sorry
  rw [norm, sqrt_mul_self] at h1 h2,
  apply sorry

end dot_product_eq_one_l262_262690


namespace paul_spending_l262_262785

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end paul_spending_l262_262785


namespace sqrt_d_eq_sqrt_a_add_sqrt_b_l262_262506

theorem sqrt_d_eq_sqrt_a_add_sqrt_b
  (a b d : ℝ)
  (h1 : d = (a + b + 2 * real.sqrt (a * b))): real.sqrt d = real.sqrt a + real.sqrt b :=
by 
  sorry

end sqrt_d_eq_sqrt_a_add_sqrt_b_l262_262506


namespace find_y_l262_262701

theorem find_y (y : ℝ) (h : 9 * y^3 = y * 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end find_y_l262_262701


namespace breadth_increase_l262_262116

theorem breadth_increase (L B : ℝ) (A : ℝ := L * B) 
  (L' : ℝ := 1.10 * L) (A' : ℝ := 1.375 * A) 
  (B' : ℝ := B * (1 + p / 100)) 
  (h1 : A = L * B)
  (h2 : A' = L' * B')
  (h3 : A' = 1.375 * A) 
  (h4 : L' = 1.10 * L) :
  p = 25 := 
begin 
  sorry 
end

end breadth_increase_l262_262116


namespace mean_median_difference_l262_262490

namespace RollerCoasters

def vertical_drops : List ℝ := [210, 130, 180, 250, 190, 220]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

noncomputable def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  if l.length % 2 = 0 then
    (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) / 2
  else
    sorted.get! (l.length / 2)

theorem mean_median_difference : 
  abs (mean vertical_drops - median vertical_drops) = 3.33 :=
by
  sorry

end RollerCoasters

end mean_median_difference_l262_262490


namespace common_ratio_of_geometric_series_l262_262855

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l262_262855


namespace hyperbola_eccentricity_l262_262618

theorem hyperbola_eccentricity (a b : ℝ) (h : (√5 * a) / 2 = b) :
  let e := real.sqrt (1 + (b^2 / a^2)) in
  e = 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_l262_262618


namespace find_y_l262_262700

theorem find_y (y : ℝ) (h : 9 * y^3 = y * 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end find_y_l262_262700


namespace eval_complex_div_l262_262279

theorem eval_complex_div : 
  (i / (Real.sqrt 7 + 3 * I) = (3 / 16) + (Real.sqrt 7 / 16) * I) := 
by 
  sorry

end eval_complex_div_l262_262279


namespace existence_of_epsilon_and_u_l262_262315

theorem existence_of_epsilon_and_u (n : ℕ) (h : 0 < n) :
  ∀ k ≥ 1, ∃ ε : ℝ, (0 < ε ∧ ε < 1 / k) ∧
  (∀ (a : Fin n → ℝ), (∀ i, 0 < a i) → ∃ u > 0, ∀ i, ε < (u * a i - ⌊u * a i⌋) ∧ (u * a i - ⌊u * a i⌋) < 1 / k) :=
by {
  sorry
}

end existence_of_epsilon_and_u_l262_262315


namespace remainder_7n_mod_4_l262_262911

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l262_262911


namespace combined_work_time_l262_262169

def Worker_A_time : ℝ := 10
def Worker_B_time : ℝ := 15

theorem combined_work_time :
  (1 / Worker_A_time + 1 / Worker_B_time)⁻¹ = 6 := by
  sorry

end combined_work_time_l262_262169


namespace number_less_than_reciprocal_l262_262919

theorem number_less_than_reciprocal :
  (∀ x ∈ {-3, -1/2, 0, 1, 3/2}, (x = -3 → x < (1 / x)) ∨ (x ≠ -3 → ¬(x < (1 / x)))) :=
by {
  sorry
}

end number_less_than_reciprocal_l262_262919


namespace max_value_of_function_on_interval_l262_262469

theorem max_value_of_function_on_interval :
  let f := λ x : ℝ, 3 * x^3 - 9 * x + 5 in
  ∃ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ set.Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x ∧ f x = 11 :=
by
  let f := λ x : ℝ, 3 * x^3 - 9 * x + 5
  have h_max_value : ∃ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ set.Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x,
  { sorry }
  obtain ⟨x, hx1, hx2⟩ := h_max_value
  have hx_max_11 : f x = 11,
  { sorry }
  exact ⟨x, hx1, hx2, hx_max_11⟩

end max_value_of_function_on_interval_l262_262469


namespace find_point_P_l262_262739

structure Triangle (A B C : Type) where
  AC BC : ℝ
  angleACB : ℝ

axiom IsoscelesTriangle (A B C : Type) (t : Triangle A B C) : t.AC = t.BC

def BD := 3
def DC := 6

noncomputable def DistanceFromC (x : ℝ) : ℝ :=
  3 * (Real.sqrt 3)

theorem find_point_P (A B C P : Type) (t : Triangle A B C) (angleAPB : ℝ) :
  IsoscelesTriangle A B C t →
  t.angleACB = 30 →
  t.AC = 9 →
  t.BC = 9 →
  angleAPB * 1.5 = t.angleACB →
  (exists (P : Type), (DistanceFromC 3 = 3 * Real.sqrt 3)) :=
by
  intros
  sorry

end find_point_P_l262_262739


namespace find_P_1_lt_X_lt_5_l262_262667

noncomputable def X : Type := sorry -- Random variable X
def μ : ℝ := 3
def σ² : ℝ := sorry -- Variance of the normal distribution
def P (A : set ℝ) : ℝ := sorry -- Probability measure

-- Defining the normal distribution condition
def X_follows_normal : Prop := sorry  -- X follows N(3, σ²)

-- Given condition: P(X ≥ 5) = 0.2
def P_X_geq_5 : Prop := P {x : ℝ | x ≥ 5} = 0.2

-- Theorem we need to prove
theorem find_P_1_lt_X_lt_5 
  (h1 : X_follows_normal)
  (h2 : P_X_geq_5) : P {x : ℝ | 1 < x ∧ x < 5} = 0.6 :=
sorry

end find_P_1_lt_X_lt_5_l262_262667


namespace julie_total_earnings_l262_262401

noncomputable def julies_hourly_rates : ℕ → ℕ
| 0 := 4 -- Mowing lawns
| 1 := 8 -- Pulling weeds
| 2 := 10 -- Pruning trees
| 3 := 12 -- Laying mulch
| _ := 0

def september_hours : ℕ → ℕ
| 0 := 25 -- Mowing lawns
| 1 := 3 -- Pulling weeds
| 2 := 10 -- Pruning trees
| 3 := 5  -- Laying mulch
| _ := 0

def october_hours (n : ℕ) : ℕ :=
(september_hours n * 3) / 2 -- 50% more than September, equivalent to multiplying by 1.5

def earnings (hours : ℕ → ℕ) : ℕ :=
(julies_hourly_rates 0 * hours 0) + 
(julies_hourly_rates 1 * hours 1) + 
(julies_hourly_rates 2 * hours 2) + 
(julies_hourly_rates 3 * hours 3)

theorem julie_total_earnings : 
  earnings september_hours + earnings october_hours = 710 := 
by
  sorry

end julie_total_earnings_l262_262401


namespace m_range_circle_l262_262657

noncomputable def circle_equation (m : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 2 * x + 4 * y + m = 0

theorem m_range_circle (m : ℝ) : circle_equation m → m < 5 := by
  sorry

end m_range_circle_l262_262657


namespace smallest_positive_period_range_of_f_on_interval_l262_262672

def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * sin x * cos x - cos (2 * x)

theorem smallest_positive_period : ∀ x, f (x + π) = f x :=
by
  intro x
  sorry

theorem range_of_f_on_interval (x : ℝ) : 0 ≤ x ∧ x ≤ 2 * π / 3 → -1 ≤ f x ∧ f x ≤ 2 :=
by
  intro h
  sorry

end smallest_positive_period_range_of_f_on_interval_l262_262672


namespace soccer_league_total_games_l262_262876

theorem soccer_league_total_games :
  let teams := 20
  let regular_games_per_team := 19 * 3
  let total_regular_games := (regular_games_per_team * teams) / 2
  let promotional_games_per_team := 3
  let total_promotional_games := promotional_games_per_team * teams
  let total_games := total_regular_games + total_promotional_games
  total_games = 1200 :=
by
  sorry

end soccer_league_total_games_l262_262876


namespace angle_MQB_l262_262063

-- Definitions based on the problem conditions
variables {A B C M N P Q : Type*}
variables [LinearOrderField A B C M N P Q]
variables (ABC : isosceles_triangle A B C) (hAB_BC : AB = BC)
variables (hM : M B A l AM = MN) (hN : MN = NC)
variables (hM_parallel : MQ ∥ BC) (hN_parallel : NP ∥ AB) (hPQ_BM : PQ = BM)

-- The angle problem and the required proof statement
theorem angle_MQB (hABC : isosceles_triangle AB BC):
  angle MQB = 36 :=
sorry

end angle_MQB_l262_262063


namespace shaniqua_income_per_haircut_l262_262092

theorem shaniqua_income_per_haircut (H : ℝ) :
  (8 * H + 5 * 25 = 221) → (H = 12) :=
by
  intro h
  sorry

end shaniqua_income_per_haircut_l262_262092


namespace pipe_C_drain_rate_l262_262442

variables (A B C : ℕ → ℕ)

noncomputable def pipe_A_rate := 40
noncomputable def pipe_B_rate := 30
noncomputable def tank_capacity := 750
noncomputable def total_time := 45
noncomputable def cycles := total_time / 3
noncomputable def net_rate (x : ℕ) := pipe_A_rate + pipe_B_rate - x

theorem pipe_C_drain_rate : ∃ x : ℕ, 15 * (70 - x) = 750 :=
by
  have h : 15 * (70 - 20) = 750 := by
    calc
      15 * (70 - 20) = 15 * 50 : by
        rw [←(show 70 - 20 = 50 by rfl)]
      ... = 750 : by norm_num
  exact ⟨20, h⟩

end pipe_C_drain_rate_l262_262442


namespace set_of_odd_numbers_l262_262138

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k + 1

def in_range (n : ℤ) : Prop := -8 < n ∧ n < 20

theorem set_of_odd_numbers {S : set ℤ} :
  S = {x : ℤ | -8 < x ∧ x < 20 ∧ ∃ k : ℤ, x = 2*k + 1} :=
sorry

end set_of_odd_numbers_l262_262138


namespace sum_x_coords_P3_l262_262411

noncomputable def sum_x_coords (coords : List ℝ) : ℝ :=
coords.sum

theorem sum_x_coords_P3 (x_coords : List ℝ) (h_len : x_coords.length = 150) 
(h_sum : sum_x_coords x_coords = 3024) : 
  let P1_coords := x_coords,
      P2_coords := List.map₂ (· + · / 2) P1_coords (P1_coords.tail ++ [P1_coords.head]),
      P3_coords := List.map₂ (· + · / 2) P2_coords (P2_coords.tail ++ [P2_coords.head]) in
  sum_x_coords P3_coords = 3024 :=
by
  sorry

end sum_x_coords_P3_l262_262411


namespace ninety_eight_squared_l262_262247

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l262_262247


namespace jenna_total_profit_l262_262394

theorem jenna_total_profit
  (buy_price : ℕ)         -- $3 per widget
  (sell_price : ℕ)        -- $8 per widget
  (rent : ℕ)              -- $10,000 monthly rent
  (worker_salary : ℕ)     -- $2,500 per worker
  (num_workers : ℕ)       -- 4 workers
  (tax_rate : ℚ)          -- 20% of total profit
  (widgets_sold : ℕ)      -- 5000 widgets sold
  (total_profit : ℤ)      -- Expected total profit $4,000
  (h_buy_price : buy_price = 3)
  (h_sell_price : sell_price = 8)
  (h_rent : rent = 10000)
  (h_worker_salary : worker_salary = 2500)
  (h_num_workers : num_workers = 4)
  (h_tax_rate : tax_rate = 0.2)
  (h_widgets_sold : widgets_sold = 5000)
  (h_total_profit : total_profit = 4000) :
  let total_salaries := num_workers * worker_salary in
  let total_fixed_costs := rent + total_salaries in
  let profit_per_widget := sell_price - buy_price in
  let total_profit_from_sales := widgets_sold * profit_per_widget in
  let profit_before_taxes := total_profit_from_sales - total_fixed_costs in
  let taxes_owed := profit_before_taxes * tax_rate in
  let net_profit := profit_before_taxes - taxes_owed in
  net_profit = total_profit :=
by
  -- The proof part is not required according to the instructions
  sorry

end jenna_total_profit_l262_262394


namespace paul_spending_l262_262786

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end paul_spending_l262_262786


namespace find_radius_of_circle_l262_262472

theorem find_radius_of_circle 
  (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (dist_AB : dist A C = 2)
  (dist_AC : dist A B = 1)
  (dist_CD : dist C D = 3)
  : ∃ r, r = 2 := 
sorry

end find_radius_of_circle_l262_262472


namespace arithmetic_identity_l262_262598

theorem arithmetic_identity : (139 + 27) * 2 + (23 + 11) = 366 := by
  have h1 : 139 + 27 = 166 := by norm_num
  have h2 : 166 * 2 = 332 := by norm_num
  have h3 : 23 + 11 = 34 := by norm_num
  have h4 : 332 + 34 = 366 := by norm_num
  rw [h1, h2, h3]
  exact h4

end arithmetic_identity_l262_262598


namespace multiply_24_99_l262_262173

theorem multiply_24_99 : 24 * 99 = 2376 :=
by
  sorry

end multiply_24_99_l262_262173


namespace combination_n_2_l262_262519

theorem combination_n_2 (n : ℕ) (h : n > 0) : 
  nat.choose n 2 = n * (n - 1) / 2 :=
sorry

end combination_n_2_l262_262519


namespace area_PARVT_l262_262541

structure Point : Type :=
(x : ℝ) (y : ℝ)

structure Parallelogram : Type :=
(P Q R S : Point)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def area_parallelogram (P Q R S : Point) : ℝ :=
abs ((Q.x - P.x) * (S.y - P.y) - (Q.y - P.y) * (S.x - P.x))

def area_triangle (A B C : Point) : ℝ :=
abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)) / 2

theorem area_PARVT (P Q R S T V : Point)
  (hPS : T = midpoint P S)
  (hRS : V = midpoint R S)
  (hParallelogram : Parallelogram P Q R S)
  (hArea : area_parallelogram P Q R S = 40) :
  area_triangle P R V + area_triangle P R T = 15 := 
sorry

end area_PARVT_l262_262541


namespace find_translation_unit_l262_262156

noncomputable def a_translation_unit (a : ℝ) : Prop :=
  (a > 0) ∧
  (∀ x, sin (2 * x - 2 * a + (π / 3)) = sin (2 * x + (3 * π / 4))) →
  a = (19 * π / 24)

theorem find_translation_unit (a > 0) :
  ∃ a, a_translation_unit a :=
begin
  sorry
end

end find_translation_unit_l262_262156


namespace distinct_possible_values_pq_plus_p_plus_q_l262_262263

theorem distinct_possible_values_pq_plus_p_plus_q :
  ∃ S : Finset ℕ, 
    (∀ p q ∈ ({1, 3, 5, 7, 9, 11, 13} : Finset ℕ), (p + 1) * (q + 1) - 1 ∈ S) ∧ 
    S.card = 27 :=
sorry

end distinct_possible_values_pq_plus_p_plus_q_l262_262263


namespace jenna_profit_l262_262392

noncomputable def total_profit (cost_price sell_price rent : ℝ) (tax_rate : ℝ) (worker_count : ℕ) (worker_salary : ℝ) (widgets_sold : ℕ) : ℝ :=
  let salaries := worker_count * worker_salary in
  let fixed_costs := salaries + rent in
  let profit_per_widget := sell_price - cost_price in
  let total_sales_profit := widgets_sold * profit_per_widget in
  let profit_before_taxes := total_sales_profit - fixed_costs in
  let taxes := profit_before_taxes * tax_rate in
  profit_before_taxes - taxes

theorem jenna_profit :
  total_profit 3 8 10000 0.2 4 2500 5000 = 4000 :=
by
  sorry

end jenna_profit_l262_262392


namespace degrees_to_minutes_l262_262227

theorem degrees_to_minutes (deg: ℝ): 
    deg = 29.5 →
    deg = 29 + 30 / 60 := 
by
  intro h
  rw h
  norm_num
  sorry

end degrees_to_minutes_l262_262227


namespace seq_b_formula_T_n_formula_l262_262042

noncomputable def S (b : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, b i

def a (n : ℕ) : ℝ := 3 * n - 1

def b (n : ℕ) : ℝ := 2 * (1 / 3^n)

def c (n : ℕ) : ℝ := a n * b n

noncomputable def T (n : ℕ) : ℝ := ∑ i in finset.range n, c i

theorem seq_b_formula :
  ∀ n : ℕ, b n = 2 * (1 / 3^n) :=
begin
  intros n,
  sorry
end

theorem T_n_formula :
  ∀ n : ℕ, T n = (7 / 2) - (1 / (2 * 3^(n - 2))) - ((3 * n - 1) / 3^n) :=
begin
  intros n,
  sorry
end

end seq_b_formula_T_n_formula_l262_262042


namespace geometric_series_common_ratio_l262_262841

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l262_262841


namespace express_in_scientific_notation_l262_262277

theorem express_in_scientific_notation :
  102200 = 1.022 * 10^5 :=
sorry

end express_in_scientific_notation_l262_262277


namespace youseff_time_difference_l262_262539

noncomputable def walking_time (blocks : ℕ) (time_per_block : ℕ) : ℕ := blocks * time_per_block
noncomputable def biking_time (blocks : ℕ) (time_per_block_seconds : ℕ) : ℕ := (blocks * time_per_block_seconds) / 60

theorem youseff_time_difference : walking_time 6 1 - biking_time 6 20 = 4 := by
  sorry

end youseff_time_difference_l262_262539


namespace fraction_first_to_second_l262_262498

def digit_fraction_proof_problem (a b c d : ℕ) (number : ℕ) :=
  number = 1349 ∧
  a = b / 3 ∧
  c = a + b ∧
  d = 3 * b

theorem fraction_first_to_second (a b c d : ℕ) (number : ℕ) :
  digit_fraction_proof_problem a b c d number → a / b = 1 / 3 :=
by
  intro problem
  sorry

end fraction_first_to_second_l262_262498


namespace parallel_lines_l262_262030

variables (Γ₁ Γ₂ : Set Point) -- Defining two circles.
variables (A B C D E F : Point) -- Defining points.
variables (chord₁ : Chord Γ₁ C D) -- Defining a chord of Γ₁.
variables (lineCA : Line (C, A)) -- Line passing through C and A.
variables (lineBD : Line (B, D)) -- Line passing through B and D.

axiom intersection_of_two_circles : intersects Γ₁ Γ₂ A ∧ intersects Γ₁ Γ₂ B
axiom second_intersection_CA : ∃ E, intersects (lineCA) Γ₂ E ∧ E ≠ A
axiom second_intersection_BD : ∃ F, intersects (lineBD) Γ₂ F ∧ F ≠ B

theorem parallel_lines : 
  (intersects Γ₁ Γ₂ A) ∧ 
  (intersects Γ₁ Γ₂ B) ∧ 
  (∃ C D, chord₁ = Chord Γ₁ C D) ∧ 
  (∃ E, intersects (lineCA C A) Γ₂ E ∧ E ≠ A) ∧ 
  (∃ F, intersects (lineBD B D) Γ₂ F ∧ F ≠ B) → 
  parallel (Line (C, D)) (Line (E, F)) :=
  by 
    sorry

end parallel_lines_l262_262030


namespace range_of_a_l262_262344

noncomputable def f (a x : ℝ) := a / x - 1 + Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ f a x ≤ 0) → a ≤ 1 := 
sorry

end range_of_a_l262_262344


namespace solve_for_y_l262_262097

theorem solve_for_y (y : ℝ) : 16^(3*y - 4) = 4^(-y - 6) → y = 2 / 7 :=
by
  sorry

end solve_for_y_l262_262097


namespace value_of_a_l262_262001

theorem value_of_a (a b : ℝ) (h1 : b = 2 * a) (h2 : b = 15 - 4 * a) : a = 5 / 2 :=
by
  sorry

end value_of_a_l262_262001


namespace painting_time_l262_262691

/-- 
Grandma Wang has 6 stools that need to be painted by a painter. Each stool needs to be painted twice.
The first coat takes 2 minutes, but there must be a 10-minute wait before applying the second coat.
We need to prove that it will take 24 minutes to paint all 6 stools.
-/
theorem painting_time : 
  (stools : ℕ) 
  (first_coat_time : ℕ) 
  (wait_time : ℕ) 
  (second_coat_time : ℕ) 
  (total_time : ℕ) 
  (h_stools : stools = 6) 
  (h_first_coat_time : first_coat_time = 2) 
  (h_wait_time : wait_time = 10) 
  (h_second_coat_time : second_coat_time = 2) 
  : total_time = 24 := 
by 
  sorry

end painting_time_l262_262691


namespace no_preimage_implies_p_gt_1_l262_262436

   noncomputable def f (x : ℝ) : ℝ :=
     -x^2 + 2 * x

   theorem no_preimage_implies_p_gt_1 (p : ℝ) (hp : ∀ x : ℝ, f x ≠ p) : p > 1 :=
   sorry
   
end no_preimage_implies_p_gt_1_l262_262436


namespace even_integers_with_inverse_mod_11_l262_262694

theorem even_integers_with_inverse_mod_11 :
  ∃ n : ℕ, n = 5 ∧ (∀ a ∈ {2, 4, 6, 8, 10}, Nat.gcd a 11 = 1) :=
by
  sorry

end even_integers_with_inverse_mod_11_l262_262694


namespace shortest_path_on_sphere_correct_l262_262074

section SphereShortestPath

variables {O : Type*} -- Sphere's center 
variables {A B C : Type*} -- Points on the sphere's surface
variables [metric_space O] [metric_space A] [metric_space B] [metric_space C]
variables (r : ℝ) -- Sphere's radius
variables (arc : set A) -- Shorter arc of the great circle

noncomputable def shortest_path_on_sphere (O : Type*) (A B : Type*) [metric_space O] 
  [metric_space A] [metric_space B] (r : ℝ) (arc : set A) : Prop :=
∀ (A B: Type*) [metric_space A] [metric_space B], ∃ C ∈ arc, 
shortest_path (on_sphere A B arc) = shorter_arc (great_circle_through O A B)

theorem shortest_path_on_sphere_correct : 
  shortest_path_on_sphere O A B r arc :=
sorry

end SphereShortestPath

end shortest_path_on_sphere_correct_l262_262074


namespace relationship_f1_f6_l262_262324

variable (f : ℝ → ℝ)

-- Assumption 1: f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)

-- Assumption 2: f is monotonically decreasing on [0, +∞)
def decreasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

theorem relationship_f1_f6
  (h1 : even_function f)
  (h2 : decreasing_on_nonneg f) :
  f 1 > f (-6) :=
by
  sorry

end relationship_f1_f6_l262_262324


namespace calculate_area_difference_l262_262485

def area_difference : ℝ :=
  let diagonal1 := 21
  let ar1_w := 4
  let ar1_h := 3
  let diagonal2 := 17
  let ar2_w := 16
  let ar2_h := 9

  let h1 := (diagonal1^2 * ar1_h^2 / (ar1_w^2 + ar1_h^2)).sqrt
  let w1 := (ar1_w * h1 / ar1_h)
  let area1 := w1 * h1

  let h2 := (diagonal2^2 * ar2_h^2 / (ar2_w^2 + ar2_h^2)).sqrt
  let w2 := (ar2_w * h2 / ar2_h)
  let area2 := w2 * h2

  area1 - area2

theorem calculate_area_difference :
  area_difference ≈ 723.67 :=
by
  sorry

end calculate_area_difference_l262_262485


namespace diplomatic_relations_eliminated_l262_262407

theorem diplomatic_relations_eliminated (n k : ℕ) (h1 : n ≥ k + 1) (G : SimpleGraph (Fin n)) (h2 : ∀ v : Fin n, G.degree v ≥ k) :
  ∃ (A B : Fin n), A ≠ B ∧ (∃ E : Finset (Sym2 (Fin n)), E.card ≥ k ∧ ∀ e ∈ E, e.inst Sym2Symmetric (Sym2.Rel.refl_sym2 _) → False) :=
by
  sorry

end diplomatic_relations_eliminated_l262_262407


namespace sin_theta_result_l262_262770

-- Define the vector types 
def vector := ℝ × ℝ × ℝ

-- Define the line's direction vector based on the given line equation
def d : vector := (3, 5, 4)

-- Define the plane's normal vector based on the given plane equation
def n : vector := (5, -3, 4)

-- Define the dot product for vectors
noncomputable def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the magnitude of a vector
noncomputable def magnitude (v : vector) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the sine of the angle between the line direction and the plane normal
noncomputable def sin_theta (d n : vector) : ℝ :=
  dot_product d n / (magnitude d * magnitude n)

-- Prove that the sine of the angle between the line and the plane is 8/25
theorem sin_theta_result : sin_theta d n = 8 / 25 :=
by
  sorry

end sin_theta_result_l262_262770


namespace min_value_M_l262_262419

noncomputable def a (x y z : ℝ) : ℝ := log z + log (x / (y * z) + 1)
noncomputable def b (x y z : ℝ) : ℝ := log (1 / x) + log (x * y * z + 1)
noncomputable def c (x y z : ℝ) : ℝ := log y + log (1 / (x * y * z) + 1)
noncomputable def M (x y z : ℝ) : ℝ := max (a x y z) (max (b x y z) (c x y z))

theorem min_value_M : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ M x y z = log 2 := by
  sorry

end min_value_M_l262_262419


namespace friend_gain_is_20_percent_l262_262577

def percentage_gain (original_cost: ℝ) (loss_percentage: ℝ) (selling_price: ℝ) : ℝ := 
  ((selling_price - (original_cost * (1 - loss_percentage / 100))) / (original_cost * (1 - loss_percentage / 100))) * 100

theorem friend_gain_is_20_percent :
  percentage_gain 50561.80 11 54000 = 20 :=
by
  sorry

end friend_gain_is_20_percent_l262_262577


namespace find_c2_given_d4_l262_262101

theorem find_c2_given_d4 (c d k : ℝ) (h : c^2 * d^4 = k) (hc8 : c = 8) (hd2 : d = 2) (hd4 : d = 4):
  c^2 = 4 :=
by
  sorry

end find_c2_given_d4_l262_262101


namespace rationalize_denominator_l262_262078

theorem rationalize_denominator (A B C : ℕ) 
  (h₁ : (4 : ℝ) / (3 * (7 : ℝ)^(1/4)) = (A * (B^(1/4) : ℝ)) / (C : ℝ)) 
  (h₂ : C > 0) 
  (h₃ : ∀ p : ℕ, p.prime → ¬ (p^4 ∣ B)) :
  A + B + C = 368 :=
sorry

end rationalize_denominator_l262_262078


namespace paul_mowing_lawns_l262_262067

theorem paul_mowing_lawns : 
  ∃ M : ℕ, 
    (∃ money_made_weeating : ℕ, money_made_weeating = 13) ∧
    (∃ spending_per_week : ℕ, spending_per_week = 9) ∧
    (∃ weeks_last : ℕ, weeks_last = 9) ∧
    (M + 13 = 9 * 9) → 
    M = 68 := by
sorry

end paul_mowing_lawns_l262_262067


namespace quadratic_roots_ratio_l262_262768

theorem quadratic_roots_ratio (k : ℝ) (k1 k2 : ℝ) (a b : ℝ) 
  (h_roots : ∀ x : ℝ, k * x * x + (1 - 6 * k) * x + 8 = 0 ↔ (x = a ∨ x = b))
  (h_ab : a ≠ b)
  (h_cond : a / b + b / a = 3 / 7)
  (h_ks : k^1 - 6 * (k1 + k2) + 8 = 0)
  (h_vieta : k1 + k2 = 200 / 36 ∧ k1 * k2 = 49 / 36) : 
  (k1 / k2 + k2 / k1 = 6.25) :=
by sorry

end quadratic_roots_ratio_l262_262768


namespace reduce_to_one_l262_262537

def transform (a b : ℕ) : ℕ := a + 2 * b

theorem reduce_to_one :
  let initial_num := concat_sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ..., 998, 999, 1000] in
  ∃ n : ℕ, (n = 1) ∧ (∀ num: ℕ, num = initial_num → (∃ m : ℕ, transform num m = 1)) :=
sorry

end reduce_to_one_l262_262537


namespace jenna_total_profit_l262_262398

-- Definitions from conditions
def widget_cost := 3
def widget_price := 8
def rent := 10000
def tax_rate := 0.2
def salary_per_worker := 2500
def number_of_workers := 4
def widgets_sold := 5000

-- Calculate intermediate values
def total_revenue := widget_price * widgets_sold
def total_cost_of_widgets := widget_cost * widgets_sold
def gross_profit := total_revenue - total_cost_of_widgets
def total_expenses := rent + salary_per_worker * number_of_workers
def net_profit_before_taxes := gross_profit - total_expenses
def taxes := tax_rate * net_profit_before_taxes
def total_profit_after_taxes := net_profit_before_taxes - taxes

-- Theorem to be proven
theorem jenna_total_profit : total_profit_after_taxes = 4000 := by sorry

end jenna_total_profit_l262_262398


namespace athena_spent_l262_262728

theorem athena_spent :
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  total_cost = 14 :=
by
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  sorry

end athena_spent_l262_262728


namespace cos_B_value_l262_262365

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cos : ℝ → ℝ)

-- Conditions
axiom h1 : a = b ∧ b = c
axiom h2 : cos A = 2 / 3
axiom h3 : cos B = (a^2 + c^2 - b^2) / (2 * a * c)
axiom h4 : cos C = (a^2 + b^2 - c^2) / (2 * a * b)
axiom h5 : b / cos B = c / cos C

-- Proof goal
theorem cos_B_value : cos B = sqrt 6 / 6 := by
  sorry

end cos_B_value_l262_262365


namespace parabola_focus_l262_262833

theorem parabola_focus (a : ℝ) (p : ℝ) (x y : ℝ) :
  a = -3 ∧ p = 6 →
  (y^2 = -2 * p * x) → 
  (y^2 = -12 * x) := 
by sorry

end parabola_focus_l262_262833


namespace proportion_y_c_l262_262366

variables {a b c x y : ℝ}
variables {A B C D : Point}
variables {BC AD : Line}
variable [triangle ABC]

-- Conditions
axiom side_a : side_of_tr ABC a
axiom side_b : side_of_tr ABC b
axiom side_c : side_of_tr ABC c
axiom angle_bisector_AD : angle_bisector AD ∠A
axiom meets_BC_at_D : meet_at AD BC D
axiom x_def : x = length CD
axiom y_def : y = length BD

theorem proportion_y_c (h : y / c = a / (b + c)) : y / c = a / (b + c) := 
  sorry

end proportion_y_c_l262_262366


namespace wall_building_time_l262_262742

variable (r : ℝ) -- rate at which one worker can build the wall
variable (W : ℝ) -- the wall in units, let’s denote one whole wall as 1 unit

theorem wall_building_time:
  (∀ (w t : ℝ), W = (60 * r) * t → W = (30 * r) * 6) :=
by
  sorry

end wall_building_time_l262_262742


namespace roots_of_polynomial_satisfy_l262_262450

noncomputable def poly_roots_satisfy : Prop :=
  ∀ p : ℝ, p ≠ 0 → 
    let (x1, x2) := 
      let sq := real.sqrt (p^2 + 2 / (p^2)) in
      ((-p + sq) / 2, (-p - sq) / 2) 
    in 
    x1^4 + x2^4 ≥ 2 + real.sqrt 2

theorem roots_of_polynomial_satisfy : poly_roots_satisfy :=
by sorry

end roots_of_polynomial_satisfy_l262_262450


namespace min_total_viewers_l262_262562

def four_movie_ticket_prices := [50, 55, 60, 65]
def permissible_two_movie_combinations := [105, 110, 115, 120, 125]
def distinct_amounts := four_movie_ticket_prices ++ permissible_two_movie_combinations

theorem min_total_viewers (H_1: four_movie_ticket_prices = [50, 55, 60, 65])
  (H_2: permissible_two_movie_combinations = [105, 110, 115, 120, 125])
  (H_3: distinct_amounts = [50, 55, 60, 65, 105, 110, 115, 120, 125])
  (H_4: ∃ n, 200 people spent n amount of money) :
  (1792 <= ∑ v in (distinct_amounts), v.count) := 
  sorry

end min_total_viewers_l262_262562


namespace parallel_planes_then_intersecting_lines_parallel_l262_262299

variables {Plane : Type} {Line : Type} [AffineSpace Plane Line]

-- Definitions from conditions
variables (α β γ : Plane) (a b : Line)
variables (h_parallel : α ∥ β) 
          (h_intersect1 : α ∩ γ = a) 
          (h_intersect2 : β ∩ γ = b)

-- Statement to prove
theorem parallel_planes_then_intersecting_lines_parallel :
  a ∥ b :=
sorry

end parallel_planes_then_intersecting_lines_parallel_l262_262299


namespace find_c_l262_262166

-- Define the polynomial P(x)
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4 * x^2 + c * x + 20

-- Given that x - 3 is a factor of P(x), prove that c = -83/3
theorem find_c (c : ℚ) (h : P c 3 = 0) : c = -83 / 3 :=
by
  sorry

end find_c_l262_262166


namespace area_of_triangle_ABC_l262_262669

structure Point :=
(x : ℝ)
(y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem area_of_triangle_ABC :
  area_of_triangle {x := 1, y := 2} {x := 2, y := 3} {x := 4, y := -1} = 3 :=
by
  -- Proof here
  sorry

end area_of_triangle_ABC_l262_262669


namespace find_U_l262_262465

-- Declare the variables and conditions
def digits : Set ℤ := {1, 2, 3, 4, 5, 6}

theorem find_U (P Q R S T U : ℤ) :
  -- Condition: Digits are distinct and each is in {1, 2, 3, 4, 5, 6}
  (P ∈ digits) ∧ (Q ∈ digits) ∧ (R ∈ digits) ∧ (S ∈ digits) ∧ (T ∈ digits) ∧ (U ∈ digits) ∧
  (P ≠ Q) ∧ (P ≠ R) ∧ (P ≠ S) ∧ (P ≠ T) ∧ (P ≠ U) ∧
  (Q ≠ R) ∧ (Q ≠ S) ∧ (Q ≠ T) ∧ (Q ≠ U) ∧
  (R ≠ S) ∧ (R ≠ T) ∧ (R ≠ U) ∧ (S ≠ T) ∧ (S ≠ U) ∧ (T ≠ U) ∧
  -- Condition: The three-digit number PQR is divisible by 9
  (100 * P + 10 * Q + R) % 9 = 0 ∧
  -- Condition: The three-digit number QRS is divisible by 4
  (10 * Q + R) % 4 = 0 ∧
  -- Condition: The three-digit number RST is divisible by 3
  (10 * R + S) % 3 = 0 ∧
  -- Condition: The sum of the digits is divisible by 5
  (P + Q + R + S + T + U) % 5 = 0
  -- Conclusion: U = 4
  → U = 4 :=
by sorry

end find_U_l262_262465


namespace new_numbers_are_reciprocals_l262_262158

variable {x y : ℝ}

theorem new_numbers_are_reciprocals (h : (1 / x) + (1 / y) = 1) : 
  (x - 1 = 1 / (y - 1)) ∧ (y - 1 = 1 / (x - 1)) := 
by
  sorry

end new_numbers_are_reciprocals_l262_262158


namespace cos_215_deg_minus_1_l262_262180

theorem cos_215_deg_minus_1 : cos (215 * real.pi / 180) - 1 = sqrt 3 / 2 :=
by sorry

end cos_215_deg_minus_1_l262_262180


namespace find_m_l262_262459

def arithmetic_sequence (a_n : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a_n n - a_n 0)) / 2

variable (a_n : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ)

def conditions : Prop :=
  S (m - 1) = -2 ∧ S m = 0 ∧ S (m + 1) = 3

theorem find_m (h : conditions a_n S m) : m = 5 := sorry

end find_m_l262_262459


namespace max_extra_packages_l262_262055

/-- Max's delivery performance --/
def max_daily_packages : Nat := 35

/-- (1) Max delivered the maximum number of packages on two days --/
def max_2_days : Nat := 2 * max_daily_packages

/-- (2) On two other days, Max unloaded a total of 50 packages --/
def two_days_50 : Nat := 50

/-- (3) On one day, Max delivered one-seventh of the maximum possible daily performance --/
def one_seventh_day : Nat := max_daily_packages / 7

/-- (4) On the last two days, the sum of packages was four-fifths of the maximum daily performance --/
def last_2_days : Nat := 2 * (4 * max_daily_packages / 5)

/-- (5) Total packages delivered in the week --/
def total_delivered : Nat := max_2_days + two_days_50 + one_seventh_day + last_2_days

/-- (6) Total possible packages in a week if worked at maximum performance --/
def total_possible : Nat := 7 * max_daily_packages

/-- (7) Difference between total possible and total delivered packages --/
def difference : Nat := total_possible - total_delivered

/-- Proof problem: Prove the difference is 64 --/
theorem max_extra_packages : difference = 64 := by
  sorry

end max_extra_packages_l262_262055


namespace cost_per_string_cheese_l262_262746

/-- Given three facts:
  1. Josh buys 3 packs of string cheese.
  2. Each pack has 20 string cheeses.
  3. He paid $6.

Prove that the cost per string cheese in cents is 10. -/
theorem cost_per_string_cheese (packs : ℕ) (cheese_per_pack : ℕ) (payment : ℝ) :
  packs = 3 →
  cheese_per_pack = 20 →
  payment = 6 →
  (payment / (packs * cheese_per_pack) * 100 = 10) :=
by {
  intros h1 h2 h3,
  sorry
}

end cost_per_string_cheese_l262_262746


namespace cleaning_time_l262_262935

theorem cleaning_time (total_area cleaned_area : ℕ) (cleaned_in_time : ℕ) (remaining_area : ℕ) :
  total_area = 72 →
  cleaned_area = 9 →
  cleaned_in_time = 23 →
  remaining_area = total_area - cleaned_area →
  (remaining_area / cleaned_area) * cleaned_in_time = 161 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end cleaning_time_l262_262935


namespace f_is_decreasing_intervals_g_range_on_interval_l262_262470

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := sin x ^ 4 + 2 * sqrt 3 * sin x * cos x - cos x ^ 4
def g (x : ℝ) : ℝ := 2 * sin (4 * x + π / 6)

-- Part (1): Proving the intervals where the function f(x) is decreasing
theorem f_is_decreasing_intervals (k : ℤ) :
  ∀ x, k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 → f' x < 0 :=
sorry

-- Part (2): Proving the range of the function g(x) over the interval [0, π/4]
theorem g_range_on_interval :
  ∀ x, 0 ≤ x ∧ x ≤ π / 4 → -1 ≤ g x ∧ g x ≤ 2 :=
sorry

end f_is_decreasing_intervals_g_range_on_interval_l262_262470


namespace perpendicular_lines_find_a_l262_262707

theorem perpendicular_lines_find_a (a : ℝ) :
  let l_1 := ∀ x y : ℝ, x + a * y - 2 = 0,
      l_2 := ∀ x y : ℝ, 2 * a * x + (a - 1) * y + 3 = 0 in
  (∀ x y : ℝ, l_1 x y → l_2 x y → 2 * a + a * (a - 1) = 0) → (a = 0 ∨ a = -1) := 
by
  sorry

end perpendicular_lines_find_a_l262_262707


namespace calculate_t_l262_262060

theorem calculate_t (t : ℚ) (natasha_hours : ℚ) (natasha_rate : ℚ) (maria_hours : ℚ) (maria_rate : ℚ) :
  natasha_hours = t - 4 →
  natasha_rate = 3t - 4 →
  maria_hours = 3t - 12 →
  maria_rate = t + 2 →
  natasha_hours * natasha_rate = maria_hours * maria_rate →
  t = 20 / 11 :=
by
  intros hNHours hNRate hMHours hMRate hEquality
  sorry

end calculate_t_l262_262060


namespace stratified_sampling_l262_262170

theorem stratified_sampling {total_students top_class_students experimental_class_students regular_class_students sample_size : ℕ}
  (h1 : total_students = 300)
  (h2 : top_class_students = 30)
  (h3 : experimental_class_students = 90)
  (h4 : regular_class_students = 180)
  (h5 : sample_size = 30) : 
  let proportion := sample_size / total_students in
  let sampled_top := top_class_students * proportion in
  let sampled_experimental := experimental_class_students * proportion in
  let sampled_regular := regular_class_students * proportion in
  (sampled_top, sampled_experimental, sampled_regular) = (3, 9, 18) :=
by
  sorry

end stratified_sampling_l262_262170


namespace intersection_A_B_l262_262684

def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 2x ≤ 0}
def A_inter_B : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_A_B : A ∩ B = A_inter_B := by
  sorry

end intersection_A_B_l262_262684


namespace area_of_shaded_region_l262_262211

def side_length_of_square : ℝ := 12
def radius_of_quarter_circle : ℝ := 6

theorem area_of_shaded_region :
  let area_square := side_length_of_square ^ 2
  let area_full_circle := π * radius_of_quarter_circle ^ 2
  (area_square - area_full_circle) = 144 - 36 * π :=
by
  sorry

end area_of_shaded_region_l262_262211


namespace value_of_m_l262_262662

theorem value_of_m (m : ℤ) : (|m| = 1) ∧ (m + 1 ≠ 0) → m = 1 := by
  sorry

end value_of_m_l262_262662


namespace percentage_decrease_in_breadth_l262_262979

-- Define the given conditions
variables {L B : ℝ} -- original length and breadth
variable (P : ℝ) -- percentage decrease in breadth

-- Define the new length and area calculations
def new_length : ℝ := 0.80 * L
def new_area : ℝ := 0.72 * (L * B)

-- Assume the new breadth after decrease
def new_breadth : ℝ := B * (1 - P)

-- Define the new area using the calculated values
def calculated_new_area : ℝ := (0.80 * L) * (B * (1 - P))

-- The theorem to be proved: The percentage of decrease in breadth is 0.1
theorem percentage_decrease_in_breadth :
  calculated_new_area = new_area → P = 0.1 :=
by
  sorry

end percentage_decrease_in_breadth_l262_262979


namespace triangle_bisector_AC1_BC1_l262_262216

theorem triangle_bisector_AC1_BC1 (A B C C1 : Point) 
  (AB BC CA : ℝ) (h_AB : dist A B = 7) (h_BC : dist B C = 6) (h_CA : dist C A = 5)
  (h_C1_on_AB : ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ C1 = t • A + (1 - t) • B) 
  (h_angle_condition : ∠A C1 C = ∠C1 B C ) :
  dist A C1 = 42 / 11 ∧ dist C1 B = 35 / 11 :=
by sorry

end triangle_bisector_AC1_BC1_l262_262216


namespace find_a_l262_262654

noncomputable def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_a (a : ℕ) (h : collinear (a, 0) (0, a + 4) (1, 3)) : a = 4 :=
by
  sorry

end find_a_l262_262654


namespace product_of_four_consecutive_naturals_is_square_l262_262451

theorem product_of_four_consecutive_naturals_is_square (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2) := 
by
  sorry

end product_of_four_consecutive_naturals_is_square_l262_262451


namespace isosceles_triangle_largest_angle_l262_262719

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : A = B) (h2 : C = 50) (h3 : A + B + C = 180) : max A C = 80 :=
by 
  -- Define additional facts about the triangle, e.g., A = B = 50, and sum of angles = 180.
  have h4 : A = 50 := sorry
  rw [h4, h2] at h3
  -- Prove the final result using the given conditions.
  sorry

end isosceles_triangle_largest_angle_l262_262719


namespace linear_function_positive_in_interval_abc_sum_greater_negative_one_l262_262182

-- Problem 1
theorem linear_function_positive_in_interval (f : ℝ → ℝ) (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n)
  (hf_m : f m > 0) (hf_n : f n > 0) : (∀ x : ℝ, m < x ∧ x < n → f x > 0) :=
sorry

-- Problem 2
theorem abc_sum_greater_negative_one (a b c : ℝ)
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : a * b + b * c + c * a > -1 :=
sorry

end linear_function_positive_in_interval_abc_sum_greater_negative_one_l262_262182


namespace matchstick_triangles_l262_262510

/-- Using 12 equal-length matchsticks, it is possible to form an isosceles triangle, an equilateral triangle, and a right-angled triangle without breaking or overlapping the matchsticks. --/
theorem matchstick_triangles :
  ∃ a b c : ℕ, a + b + c = 12 ∧ (a = b ∨ b = c ∨ a = c) ∧ (a * a + b * b = c * c ∨ a = b ∧ b = c) :=
by
  sorry

end matchstick_triangles_l262_262510


namespace a_share_of_profit_is_60_l262_262925

-- Define the conditions under given constraints
def a_investment : ℝ := 150
def b_investment : ℝ := 200
def total_profit : ℝ := 100
def a_investment_time : ℝ := 12 -- months
def b_investment_time : ℝ := 6 -- months

-- Calculate A's and B's investment in time.
def a_time_investment : ℝ := a_investment * a_investment_time
def b_time_investment : ℝ := b_investment * b_investment_time

-- Calculate the total investment in terms of time
def total_time_investment : ℝ := a_time_investment + b_time_investment
  
-- Calculate A's share of the profit
def a_share_of_profit (a_time_investment total_time_investment total_profit : ℝ) : ℝ :=
  (a_time_investment / total_time_investment) * total_profit

-- Theorem to prove A's share of the profit
theorem a_share_of_profit_is_60 : a_share_of_profit a_time_investment total_time_investment total_profit = 60 := by
  sorry

end a_share_of_profit_is_60_l262_262925


namespace divisible_numbers_in_set_l262_262305

open Nat

theorem divisible_numbers_in_set (n : ℕ) (h : 0 < n) (S : Finset ℕ) 
  (h_card : S.card = n + 1) (h_subset : ∀ x ∈ S, x ≤ 2 * n) :
  ∃ a b ∈ S, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
  by
  sorry

end divisible_numbers_in_set_l262_262305


namespace area_of_triangle_l262_262650

-- Definitions of the given conditions
def a : ℝ := Real.sqrt 2
def A : ℝ := Real.pi / 4  -- 45 degrees in radians
def B : ℝ := Real.pi / 3  -- 60 degrees in radians
def sinA : ℝ := Real.sin A
def sinB : ℝ := Real.sin B

-- Definition of b using the Law of Sines
def b : ℝ := (a * sinB) / sinA

-- Calculation of angle C
def C : ℝ := Real.pi - A - B  -- 180 degrees (pi radians) minus A and B in radians
def sinC : ℝ := Real.sin C

-- Expected value of the area of the triangle
def expected_area : ℝ := (3 + Real.sqrt 3) / 4

-- The theorem we want to prove
theorem area_of_triangle :
  (1 / 2) * a * b * sinC = expected_area := by
  sorry

end area_of_triangle_l262_262650


namespace vertex_below_x_axis_l262_262364

theorem vertex_below_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + a < 0) → a < 1 :=
by 
  sorry

end vertex_below_x_axis_l262_262364


namespace julio_fish_count_l262_262749

theorem julio_fish_count : 
  ∀ (catches_per_hour : ℕ) (hours_fishing : ℕ) (fish_lost : ℕ) (total_fish : ℕ), 
  catches_per_hour = 7 →
  hours_fishing = 9 →
  fish_lost = 15 →
  total_fish = (catches_per_hour * hours_fishing) - fish_lost →
  total_fish = 48 :=
by
  intros catches_per_hour hours_fishing fish_lost total_fish
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

# Jason had jotted down the following proof statement:

end julio_fish_count_l262_262749


namespace determinant_inequality_l262_262083

variable (x : ℝ)

def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem determinant_inequality (h : determinant 2 (3 - x) 1 x > 0) : x > 1 := by
  sorry

end determinant_inequality_l262_262083


namespace average_of_multiples_l262_262617

theorem average_of_multiples :
  let sum_of_first_7_multiples_of_9 := 9 + 18 + 27 + 36 + 45 + 54 + 63
  let sum_of_first_5_multiples_of_11 := 11 + 22 + 33 + 44 + 55
  let sum_of_first_3_negative_multiples_of_13 := -13 + -26 + -39
  let total_sum := sum_of_first_7_multiples_of_9 + sum_of_first_5_multiples_of_11 + sum_of_first_3_negative_multiples_of_13
  let average := total_sum / 3
  average = 113 :=
by
  sorry

end average_of_multiples_l262_262617


namespace metallic_sheet_first_dimension_l262_262198

-- Given Conditions
variable (x : ℝ) (height width : ℝ)
def metallic_sheet :=
  (x > 0) ∧ (height = 8) ∧ (width = 36 - 2 * height)

-- Volume of the resulting box should be 5760 m³
def volume_box :=
  (width - 2 * height) * (x - 2 * height) * height = 5760

-- Prove the first dimension of the metallic sheet
theorem metallic_sheet_first_dimension (h1 : metallic_sheet x height width) (h2 : volume_box x height width) : 
  x = 52 :=
  sorry

end metallic_sheet_first_dimension_l262_262198


namespace general_term_a_general_term_b_sum_b_l262_262682

theorem general_term_a (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1)
  (h2 : a 3 = 2) :
  ∀ n : ℕ, n > 0 → a n = n - 1 :=
by
  sorry

theorem general_term_b (b : ℕ → ℝ)
  (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1)
  (h2 : a 3 = 2)
  (h3 : ∀ n : ℕ, n > 0 → b n = (1 / 3) ^ a n + n) :
  ∀ n : ℕ, n > 0 → b n = (1 / 3) ^ (n - 1) + n :=
by
  sorry

theorem sum_b (b : ℕ → ℝ)
  (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1)
  (h2 : a 3 = 2)
  (h3 : ∀ n : ℕ, n > 0 → b n = (1 / 3) ^ a n + n) :
  ∀ n : ℕ, n > 0 → (∑ k in finset.range n, b (k + 1)) = (3 - 3 ^ (1 - n)) / 2 + n * (n + 1) / 2 :=
by
  sorry

end general_term_a_general_term_b_sum_b_l262_262682


namespace sum_of_prime_factors_l262_262901

theorem sum_of_prime_factors {N : ℕ} (h1 : 8679030 = 8679921 + N)
  (h2 : 8679030 % 330 = 0) :
  Nat.sum_prime_factors 8679030 = 284 :=
sorry

end sum_of_prime_factors_l262_262901


namespace ajay_income_l262_262172

theorem ajay_income
  (I : ℝ)
  (h₁ : I * 0.45 + I * 0.25 + I * 0.075 + 9000 = I) :
  I = 40000 :=
by
  sorry

end ajay_income_l262_262172


namespace max_single_player_salary_l262_262199

theorem max_single_player_salary :
  ∃ y : ℕ, (team_size = 25 ∧ min_salary = 20000 ∧ max_total_salary = 900000) →
  ((24 * min_salary + y ≤ max_total_salary) ∧ y ≥ min_salary ∧ y = 420000) :=
by {
  let team_size := 25,
  let min_salary := 20000,
  let max_total_salary := 900000,
  have y := max_total_salary - 24 * min_salary,
  existsi y,
  sorry
}

end max_single_player_salary_l262_262199


namespace assign_questions_to_students_l262_262493

theorem assign_questions_to_students:
  ∃ (assignment : Fin 20 → Fin 20), 
  (∀ s : Fin 20, ∃ q1 q2 : Fin 20, (assignment s = q1 ∨ assignment s = q2) ∧ q1 ≠ q2 ∧ ∀ q : Fin 20, ∃ s1 s2 : Fin 20, (assignment s1 = q ∧ assignment s2 = q) ∧ s1 ≠ s2) :=
by
  sorry

end assign_questions_to_students_l262_262493


namespace geometric_series_ratio_l262_262863

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l262_262863


namespace geometric_series_common_ratio_l262_262846

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l262_262846


namespace length_of_room_l262_262473

theorem length_of_room (b : ℕ) (t : ℕ) (L : ℕ) (blue_tiles : ℕ) (tile_area : ℕ) (total_area : ℕ) (effective_area : ℕ) (blue_area : ℕ) :
  b = 10 →
  t = 2 →
  blue_tiles = 16 →
  tile_area = t * t →
  total_area = (L - 4) * (b - 4) →
  blue_area = blue_tiles * tile_area →
  2 * blue_area = 3 * total_area →
  L = 20 :=
by
  intros h_b h_t h_blue_tiles h_tile_area h_total_area h_blue_area h_proportion
  sorry

end length_of_room_l262_262473


namespace fill_cistern_7_2_hours_l262_262950

theorem fill_cistern_7_2_hours :
  let R_fill := 1 / 4
  let R_empty := 1 / 9
  R_fill - R_empty = 5 / 36 →
  1 / (R_fill - R_empty) = 7.2 := 
by
  intros
  sorry

end fill_cistern_7_2_hours_l262_262950


namespace betty_cookies_brownies_l262_262234

theorem betty_cookies_brownies :
  let initial_cookies := 60
  let initial_brownies := 10
  let cookies_per_day := 3
  let brownies_per_day := 1
  let days := 7
  let remaining_cookies := initial_cookies - cookies_per_day * days
  let remaining_brownies := initial_brownies - brownies_per_day * days
  remaining_cookies - remaining_brownies = 36 :=
by
  sorry

end betty_cookies_brownies_l262_262234


namespace jenna_total_profit_l262_262393

theorem jenna_total_profit
  (buy_price : ℕ)         -- $3 per widget
  (sell_price : ℕ)        -- $8 per widget
  (rent : ℕ)              -- $10,000 monthly rent
  (worker_salary : ℕ)     -- $2,500 per worker
  (num_workers : ℕ)       -- 4 workers
  (tax_rate : ℚ)          -- 20% of total profit
  (widgets_sold : ℕ)      -- 5000 widgets sold
  (total_profit : ℤ)      -- Expected total profit $4,000
  (h_buy_price : buy_price = 3)
  (h_sell_price : sell_price = 8)
  (h_rent : rent = 10000)
  (h_worker_salary : worker_salary = 2500)
  (h_num_workers : num_workers = 4)
  (h_tax_rate : tax_rate = 0.2)
  (h_widgets_sold : widgets_sold = 5000)
  (h_total_profit : total_profit = 4000) :
  let total_salaries := num_workers * worker_salary in
  let total_fixed_costs := rent + total_salaries in
  let profit_per_widget := sell_price - buy_price in
  let total_profit_from_sales := widgets_sold * profit_per_widget in
  let profit_before_taxes := total_profit_from_sales - total_fixed_costs in
  let taxes_owed := profit_before_taxes * tax_rate in
  let net_profit := profit_before_taxes - taxes_owed in
  net_profit = total_profit :=
by
  -- The proof part is not required according to the instructions
  sorry

end jenna_total_profit_l262_262393


namespace Peter_drew_more_l262_262800

theorem Peter_drew_more :
  ∃ (P : ℕ), 5 + P + (P + 20) = 41 ∧ (P - 5 = 3) :=
sorry

end Peter_drew_more_l262_262800


namespace reduced_price_l262_262544

theorem reduced_price (
  P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 9 = 900 / R - 900 / P)
  (h3 : P = 42.8571) :
  R = 30 :=
by {
  sorry
}

end reduced_price_l262_262544


namespace binom_n_2_l262_262526

theorem binom_n_2 (n : ℕ) (h : 1 < n) : (nat.choose n 2) = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262526


namespace find_B_l262_262321

variables (A B C a b c : ℝ)
-- Assuming that the angles A, B, C are positive and their sum is π since it's a triangle.
-- Acute triangle implies \in (0, π/2)
-- Let's define each condition as hypotheses for the theorem

hypothesis (h1 : 0 < A ∧ A < π / 2)
hypothesis (h2 : 0 < B ∧ B < π / 2)
hypothesis (h3 : 0 < C ∧ C < π / 2)
hypothesis (h4 : A + B + C = π)

hypothesis (h5 : sqrt 3 * a = 2 * b * sin B * cos C + 2 * b * sin C * cos B)

theorem find_B : 
  B = π / 3 :=
sorry

end find_B_l262_262321


namespace AI_length_in_triangle_l262_262428

noncomputable def find_AI (D A N E I : Point) (C : Circle) (angle_DAN : Angle)
  (DA AN AE DI IN DN AI : Length) : Length := sorry

theorem AI_length_in_triangle
  (D A N E I : Point)
  (C : Circle)
  (angle_DAN : Angle)
  (inscribed : InscribedTriangle C (Triangle.mk D A N))
  (DA_length : DA = 2)
  (AN_length : AN = 1)
  (AE_length : AE = 2.5)
  (bisector : AngleBisector AE angle_DAN)
  (intersection : Intersects AE DI I)
  (BIS : AI = find_AI D A N E I C angle_DAN DA AN AE DI IN DN AI)
: AI = 4 / 5 := 
sorry

end AI_length_in_triangle_l262_262428


namespace seating_arrangements_l262_262009

theorem seating_arrangements :
  let total_seat_ways := Nat.factorial 10,
      john_wilma_paul_together := Nat.factorial 8 * Nat.factorial 3,
      alice_bob_together := Nat.factorial 9 * Nat.factorial 2,
      overlap := Nat.factorial 7 * Nat.factorial 3 * Nat.factorial 2
  in
  total_seat_ways - john_wilma_paul_together - alice_bob_together + overlap = 2685600 :=
by
  sorry

end seating_arrangements_l262_262009


namespace difficult_vs_easy_l262_262613

theorem difficult_vs_easy (x y z : ℕ) (h1 : x + y + z = 100) (h2 : x + 3 * y + 2 * z = 180) :
  x - y = 20 :=
by sorry

end difficult_vs_easy_l262_262613


namespace problem_solution_l262_262343

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_solution (a m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) →
  a = 1 ∧ (∃ n : ℝ, f n 1 ≤ m - f (-n) 1) → 4 ≤ m := 
by
  sorry

end problem_solution_l262_262343


namespace probability_A_not_work_jan1_l262_262956

open Classical

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def comb (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem probability_A_not_work_jan1 : 
  let total_ways := comb 6 2 * comb 4 2 * comb 2 2,
      favorable_ways := comb 5 2 * comb 4 2 * comb 2 2
  in (favorable_ways : ℚ) / total_ways = 2 / 3 := 
by
  obtain rfl : comb 5 2 = 10 := by sorry,
  obtain rfl : comb 6 2 = 15 := by sorry,
  have total_ways : ℚ := 15 * comb 4 2 * comb 2 2,
  have favorable_ways : ℚ := 10 * comb 4 2 * comb 2 2,
  have eq1 : (10 * comb 4 2 * comb 2 2) = (10 : ℚ) * (comb 4 2 * comb 2 2) := by sorry,
  have eq2 : (15 * comb 4 2 * comb 2 2) = (15 : ℚ) * (comb 4 2 * comb 2 2) := by sorry,
  rw [eq1, eq2],
  field_simp,
  exact eq.trans (10 / 15) (by norm_num),
  done

end probability_A_not_work_jan1_l262_262956


namespace athena_total_spent_l262_262733

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end athena_total_spent_l262_262733


namespace ticket_price_possible_values_l262_262196

theorem ticket_price_possible_values :
  let y := λ y : ℕ, y ∣ 210 ∧ y ∣ 280 in
  (finset.filter y (finset.Icc 1 280)).card = 8 :=
by 
  sorry

end ticket_price_possible_values_l262_262196


namespace xiao_ming_waiting_probability_l262_262971

noncomputable def arrival_time_distribution : ℝ → ℝ := sorry

theorem xiao_ming_waiting_probability :
  (∫ t in 6.5..7.5, arrival_time_distribution t * if t ∈ (6.5..7.0) ∨ t ∈ (7.333..7.5) then 1 else 0) / 
  (∫ t in 6.5..7.5, arrival_time_distribution t)
  = 1/2 := by sorry

end xiao_ming_waiting_probability_l262_262971


namespace new_median_is_five_l262_262953

-- Definitions and conditions
def collection : List ℕ := [4, 4, 5, 5, 6, 9]
def mean (lst : List ℕ) : ℚ := ((list.sum lst).toRat) / (list.length lst).toRat
def mode (lst : List ℕ) : ℕ := (list.filter (λ n, (list.count n lst) = (list.countp (λ m, (list.count m lst) > 1) lst)) lst).head

-- Theorem to prove
theorem new_median_is_five (c : List ℕ)
  (hc : c = [4, 4, 5, 5, 6, 9])
  (hmean : mean c = 5.5)
  (hmode : mode c = 4)
  (hmedian : median_nth c = 5) : 
  let new_collection := c ++ [9] in 
  median_nth new_collection = 5 := 
  by sorry

end new_median_is_five_l262_262953


namespace cody_final_ticket_count_l262_262589

theorem cody_final_ticket_count :
  ∀ (initial_tickets won_tickets beanie_cost tokens_trade_cost tokens_games_tickets games_played : ℕ),
    initial_tickets = 50 →
    won_tickets = 49 →
    beanie_cost = 25 →
    tokens_trade_cost = 10 →
    tokens_games_tickets = 6 →
    games_played = 3 →
    let final_tickets := (initial_tickets + won_tickets) - beanie_cost - tokens_trade_cost + (tokens_games_tickets * games_played)
    in final_tickets = 82 :=
by
  intros initial_tickets won_tickets beanie_cost tokens_trade_cost tokens_games_tickets games_played
  intros h_initial h_won h_beanie h_tokens_trade h_tokens_games h_games_played
  let final_tickets := (initial_tickets + won_tickets) - beanie_cost - tokens_trade_cost + (tokens_games_tickets * games_played)
  have : final_tickets = ((50 + 49) - 25 - 10 + (6 * 3)) :=
    by rw [h_initial, h_won, h_beanie, h_tokens_trade, h_tokens_games, h_games_played]
  have : final_tickets = 82 := by norm_num at this; exact this
  exact this

end cody_final_ticket_count_l262_262589


namespace polygon_sides_in_arithmetic_progression_l262_262461

theorem polygon_sides_in_arithmetic_progression 
  (a : ℕ → ℝ) (n : ℕ) (h1: ∀ i, 1 ≤ i ∧ i ≤ n → a i = a 1 + (i - 1) * 10) 
  (h2 : a n = 150) : n = 12 :=
sorry

end polygon_sides_in_arithmetic_progression_l262_262461


namespace election_votes_calculation_l262_262717

theorem election_votes_calculation (V : ℝ) (w_votes : ℝ) (s_votes : ℝ) (t_votes : ℝ) :
  (0.05 * V + 0.95 * V = V) ∧ 
  (w_votes = 0.45 * 0.95 * V) ∧ 
  (s_votes = 0.35 * 0.95 * V) ∧ 
  (w_votes - s_votes = 150) ∧ 
  (t_votes = 0.95 * V - (w_votes + s_votes)) → 
  V ≈ 1579 ∧ w_votes ≈ 675 ∧ s_votes ≈ 525 ∧ t_votes ≈ 300 := 
sorry

end election_votes_calculation_l262_262717


namespace mabel_total_walk_l262_262044

theorem mabel_total_walk (steps_mabel : ℝ) (ratio_helen : ℝ) (steps_helen : ℝ) (total_steps : ℝ) :
  steps_mabel = 4500 →
  ratio_helen = 3 / 4 →
  steps_helen = ratio_helen * steps_mabel →
  total_steps = steps_mabel + steps_helen →
  total_steps = 7875 := 
by
  intros h_steps_mabel h_ratio_helen h_steps_helen h_total_steps
  rw [h_steps_mabel, h_ratio_helen] at h_steps_helen
  rw [h_steps_helen, h_steps_mabel] at h_total_steps
  rw h_total_steps
  rw [h_steps_mabel, h_ratio_helen]
  linarith

end mabel_total_walk_l262_262044


namespace prob_divisible_by_18_l262_262109

/--
Given that the digits 1 to 9 are used exactly once to form a 9-digit number,
we need to show that the probability that the resulting number is divisible by 18
is equal to 4/9.
-/
theorem prob_divisible_by_18 (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  let total_permutations := (digits.card.factorial)
  let even_last_digit_permutations := (∑ d in (digits.filter even), (digits.erase d).card.factorial)
  let p := (even_last_digit_permutations : ℚ) / (total_permutations : ℚ)
  p = 4 / 9 :=
by {
  sorry
}

end prob_divisible_by_18_l262_262109


namespace company_pays_per_month_after_new_hires_l262_262954

theorem company_pays_per_month_after_new_hires :
  let initial_employees := 500 in
  let new_employees := 200 in
  let hourly_pay := 12 in
  let hours_per_day := 10 in
  let days_per_week := 5 in
  let weeks_per_month := 4 in
  let total_employees := initial_employees + new_employees in
  let daily_pay := hourly_pay * hours_per_day in
  let working_days_in_month := days_per_week * weeks_per_month in
  let monthly_pay_per_employee := daily_pay * working_days_in_month in
  let total_monthly_payment := total_employees * monthly_pay_per_employee in
  total_monthly_payment = 1680000 := 
by
  sorry

end company_pays_per_month_after_new_hires_l262_262954


namespace find_m_l262_262337

noncomputable def circleEquation (x y m : ℝ) : ℝ := x^2 + y^2 + m * x - 4
def lineEquation (x y : ℝ) : ℝ := x - y + 4
def center (m : ℝ) : ℝ × ℝ := (-m/2, 0)

theorem find_m (x y : ℝ) (h1 : ∃ (x y : ℝ), circleEquation x y m = 0)
    (h2 : ∃ (x1 y1 : ℝ), circleEquation x1 y1 m = 0 ∧ (x1 ≠ x ∨ y1 ≠ y))
    (h3 : lineEquation (center m).fst (center m).snd = 0) :
    m = 8 :=
by {
  sorry
}

end find_m_l262_262337


namespace first_four_eq_last_four_l262_262313

variable (s : List ℕ) (n : ℕ)
variable [Inhabited ℕ]

-- Condition (a): uniqueness of any 5 consecutive digits
def unique_5_consecutive_digits (s : List ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i ≤ s.length - 5 ∧ 0 ≤ j ∧ j ≤ s.length - 5 ∧ i ≠ j →
    (s.get (i + 0), s.get (i + 1), s.get (i + 2), s.get (i + 3), s.get (i + 4)) ≠ 
    (s.get (j + 0), s.get (j + 1), s.get (j + 2), s.get (j + 3), s.get (j + 4))

-- Condition (b): appending any digit to the sequence violates property (a)
def append_violates_property (s : List ℕ) : Prop :=
  ∀ x, ¬unique_5_consecutive_digits (s ++ [x])

-- The main theorem to prove
theorem first_four_eq_last_four (s : List ℕ) (h_length : s.length ≥ 5)
  (h_unique : unique_5_consecutive_digits s) 
  (h_append : append_violates_property s) 
  : (s.get 0, s.get 1, s.get 2, s.get 3) = 
    (s.get (s.length - 4), s.get (s.length - 3), s.get (s.length - 2), s.get (s.length - 1)) := 
  sorry

end first_four_eq_last_four_l262_262313


namespace cubic_polynomial_unique_solution_l262_262458

theorem cubic_polynomial_unique_solution :
  ∃ q : Polynomial ℝ, 
    (q.monic ∧ 
     q.eval (2 - 3 * Complex.I) = 0 ∧ 
     q.eval 0 = 40 ∧ 
     q.derivative.eval 1 = 0 ∧ 
     q = Polynomial.Cubic (1 : ℝ) (27 / 4) (-23) (-351 / 4)) :=
by {
  use Polynomial.Cubic (1 : ℝ) (27 / 4) (-23) (-351 / 4),
  split,
  { exact Polynomial.monic_Cubic 1 (27 / 4) (-23) (-351 / 4) },

  split,
  { have : q.eval (2 - 3 * Complex.I) = 0,
    sorry },

  split,
  { exact Polynomial.eval_zero_eq (Polynomial.Cubic (1 : ℝ) (27 / 4) (-23) (-351 / 4)) 40 },

  split,
  { exact Polynomial.derivative.Cubic_eq 1 (27 / 4) (-23) (-351 / 4) },

  repeat { try { sorry }}
}

end cubic_polynomial_unique_solution_l262_262458


namespace DE_perp_EF_l262_262314

variables {A B C D E F : Type*}
variables [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C] [NormedSpace ℝ D] [NormedSpace ℝ E] [NormedSpace ℝ F]
variables [InnerProductSpace ℝ B]

-- Assume the points and conditions in the problem
variables {triangle_ABC : Triangle A B C}
variable (D : Point A)
variable (E : Point B)
variable (F : Point C)

-- Conditions given in the problem
axiom angle_DAC_eq_30 : ∠ D A C = 30
axiom angle_DCA_eq_30 : ∠ D C A = 30
axiom angle_DBA_eq_60 : ∠ D B A = 60
axiom E_mid_BC : midpoint E B C
axiom F_trisection_AC : trisection_point F A C 2

-- The proof target
theorem DE_perp_EF : ∥E - D∥ * ∥F - E∥ * (E - D) ⬝ (F - E) = 0 := sorry

end DE_perp_EF_l262_262314


namespace cyclic_quadrilateral_proj_inscribed_l262_262449

noncomputable def cyclic_quadrilateral (A B C D : Type) [euclidean_geometry A]
    (AB BC CD DA : LineSegment A) 
    (cyclic : CyclicQuadrilateral AB BC CD DA) 
    (O : Type) (proj_A1 : Proj O AB) (proj_B1 : Proj O BC) 
    (proj_C1 : Proj O CD) (proj_D1 : Proj O DA) := 
    ∃ A1 B1 C1 D1: Point A,
    projections O AB A1 ∧ projections O BC B1 ∧ 
    projections O CD C1 ∧ projections O DA D1 ∧ 
    ¬(falls_on_extension A1 AB) ∧ ¬(falls_on_extension B1 BC) ∧ 
    ¬(falls_on_extension C1 CD) ∧ ¬(falls_on_extension D1 DA) ∧ 
    InscribedQuadrilateral A1 B1 C1 D1

theorem cyclic_quadrilateral_proj_inscribed {A B C D : Type} [euclidean_geometry A]
    {AB BC CD DA : LineSegment A} 
    (cyclic : CyclicQuadrilateral AB BC CD DA) 
    {O : Type} (proj_A1 : Proj O AB) (proj_B1 : Proj O BC) 
    (proj_C1 : Proj O CD) (proj_D1 : Proj O DA) :
    ∃ A1 B1 C1 D1: Point A,
    projections O AB A1 ∧ projections O BC B1 ∧ 
    projections O CD C1 ∧ projections O DA D1 ∧ 
    ¬(falls_on_extension A1 AB) ∧ ¬(falls_on_extension B1 BC) ∧ 
    ¬(falls_on_extension C1 CD) ∧ ¬(falls_on_extension D1 DA) ∧ 
    InscribedQuadrilateral A1 B1 C1 D1 := sorry

end cyclic_quadrilateral_proj_inscribed_l262_262449


namespace scalar_product_of_AE_and_BD_l262_262642

-- Definitions and conditions
variable (A B C D E : Type) [InnerProductSpace ℝ A]
variable (side_length : ℝ) (h_side : side_length = 2)
variable (midpoint_condition : E = midpoint C D)
variable (vAD vAB : A)
variable (hBD : vBD = vAD - vAB)
variable (hAE : vAE = vAD + (1 / 2) • vAB)

-- The proof problem
theorem scalar_product_of_AE_and_BD :
  inner_product vAE vBD = 2 :=
sorry

end scalar_product_of_AE_and_BD_l262_262642


namespace vector_sum_half_l262_262384

variable (A B C E F : Type)
variable [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup E] [AddCommGroup F]
variable (vector_space : Type) [Field vector_space]
variables {vector_space : Type} [VectorSpace vector_space A] [VectorSpace vector_space B] [VectorSpace vector_space C] [VectorSpace vector_space E] [VectorSpace vector_space F]

/-- Given a triangle ABC with points E and F such that 
    vector AE is equal to half of vector AB and vector CF is equal to twice vector FA.
    If vector EF can be expressed as x times vector AB plus y times vector AC, 
    then x + y is equal to -1/6. -/
theorem vector_sum_half : 
  (AE : vector_space → A) = (1/2 : vector_space) • (AB : vector_space → B) →
  (CF : vector_space → C) = (2 : vector_space) • (FA : vector_space → F) →
  (EF : vector_space → A) = (x : vector_space) • (AB : vector_space → B) + (y : vector_space) • (AC : vector_space → C) →
  x + y = (-1/6 : vector_space) :=
 by
  intros h1 h2 h3
  sorry

end vector_sum_half_l262_262384


namespace people_remaining_at_end_l262_262881

def total_people_start : ℕ := 600
def girls_start : ℕ := 240
def boys_start : ℕ := total_people_start - girls_start
def boys_left_early : ℕ := boys_start / 4
def girls_left_early : ℕ := girls_start / 8
def total_left_early : ℕ := boys_left_early + girls_left_early
def people_remaining : ℕ := total_people_start - total_left_early

theorem people_remaining_at_end : people_remaining = 480 := by
  sorry

end people_remaining_at_end_l262_262881


namespace infinite_x0_finite_values_l262_262420

noncomputable def f (x : ℝ) : ℝ := 5 * x - x^2

theorem infinite_x0_finite_values :
  ∃ S : Set ℝ, S = {x0 | ∀ n : ℕ, let xn : ℕ → ℝ := λ n, if h : n = 0 then x0 else f (xn (n - 1)) in (Set.finite (Set.range xn))} ∧ 
  infinite S :=
by sorry

end infinite_x0_finite_values_l262_262420


namespace common_ratio_of_geometric_series_l262_262869

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l262_262869


namespace largest_quotient_card_42819_l262_262619

theorem largest_quotient_card_42819 :
  ∃ (a b : ℕ), 
    (a ∈ {984} ∧ b ∈ {12}) ∧ 
    a / b = 82 :=
by
  sorry

end largest_quotient_card_42819_l262_262619


namespace sum_of_five_integers_l262_262478

theorem sum_of_five_integers :
  ∃ (n m : ℕ), (n * (n + 1) = 336) ∧ ((m - 1) * m * (m + 1) = 336) ∧ ((n + (n + 1) + (m - 1) + m + (m + 1)) = 51) := 
sorry

end sum_of_five_integers_l262_262478


namespace monotonically_increasing_interval_l262_262131

theorem monotonically_increasing_interval (x : ℝ) (h : ∀ y, y = -x^2 + 4 * x) :
  {x | x ≤ 2} = Iic 2 :=
sorry

end monotonically_increasing_interval_l262_262131


namespace find_a_given_inverse_l262_262679

theorem find_a_given_inverse (a : ℝ) (f : ℝ → ℝ) (h_inv : Function.LeftInverse (λ y, 2^(y - log2(1+y))) f)
  (h_inv_value : f⁻¹ 2 = 1) : a = 3 :=
sorry

end find_a_given_inverse_l262_262679


namespace free_fall_height_and_last_second_distance_l262_262559

theorem free_fall_height_and_last_second_distance :
  let time := 11
  let initial_distance := 4.9
  let increment := 9.8
  let total_height := (initial_distance * time + increment * (time * (time - 1)) / 2)
  let last_second_distance := initial_distance + increment * (time - 1)
  total_height = 592.9 ∧ last_second_distance = 102.9 :=
by
  sorry

end free_fall_height_and_last_second_distance_l262_262559


namespace line_slope_angle_l262_262486

theorem line_slope_angle (α : ℝ) : 
  (∀ x y : ℝ, √3 * x + y + 1 = 0 → y = -√3 * x - 1) → 
  (∃ α ∈ (0 : ℝ, 180), tan α = -√3 ∧ α = 120) :=
by
  intros h
  sorry

end line_slope_angle_l262_262486


namespace johns_previous_salary_l262_262745

-- Conditions
def johns_new_salary : ℝ := 70
def percent_increase : ℝ := 0.16666666666666664

-- Statement
theorem johns_previous_salary :
  ∃ x : ℝ, x + percent_increase * x = johns_new_salary ∧ x = 60 :=
by
  sorry

end johns_previous_salary_l262_262745


namespace circles_intersect_l262_262351

-- Define the parameters and conditions given in the problem.
def r1 : ℝ := 5  -- Radius of circle O1
def r2 : ℝ := 8  -- Radius of circle O2
def d : ℝ := 8   -- Distance between the centers of O1 and O2

-- The main theorem that needs to be proven.
theorem circles_intersect (r1 r2 d : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 8) (h_d : d = 8) :
  r2 - r1 < d ∧ d < r1 + r2 :=
by
  sorry

end circles_intersect_l262_262351


namespace cattle_train_speed_is_correct_l262_262189

-- Given conditions as definitions
def cattle_train_speed (x : ℝ) : ℝ := x
def diesel_train_speed (x : ℝ) : ℝ := x - 33
def cattle_train_distance (x : ℝ) : ℝ := 6 * x
def diesel_train_distance (x : ℝ) : ℝ := 12 * (x - 33)

-- Statement to prove
theorem cattle_train_speed_is_correct (x : ℝ) :
  cattle_train_distance x + diesel_train_distance x = 1284 → 
  x = 93.33 :=
by
  intros h
  sorry

end cattle_train_speed_is_correct_l262_262189


namespace pow3_mod8_l262_262606

theorem pow3_mod8 (n : ℕ) : (3^n) % 8 = 1 :=
by {
  -- Definitions used as conditions
  have base_case : 3 % 8 = 3,
  have step_case : (3^2) % 8 = 1,
  have pattern : ∀ m : ℕ, (3^(2*m)) % 8 = 1, from sorry,

  -- Specific instance for n = 1234, even, simplify using pattern
  sorry
}

end pow3_mod8_l262_262606


namespace determine_colors_2022_gon_l262_262426

noncomputable def minimum_queries (n : ℕ) : ℕ :=
if n = 2022 then 22 else 0

theorem determine_colors_2022_gon (A : Fin 2022 → Fin 2022)
  (determine_colors : (Fin 2022 → Bool) → Bool)
  (color_points : Fin 2022 → Bool) :
  (∃ Q : ℕ, (Q = minimum_queries 2022) ∧
   (∀ strategy : (Fin 2022 → Bool) → Bool, ∃ colors : Fin 2022 → Bool,
      strategy = determine_colors colors) ∧ Q = 22) :=
begin
  let Q := 22,
  use Q,
  split,
  { refl, },
  split,
  { intros strategy,
    use color_points,
    assume h,
    sorry, -- The proof itself is omitted.
  },
  { refl, },
end

end determine_colors_2022_gon_l262_262426


namespace no_four_intersections_l262_262681

theorem no_four_intersections (m : ℝ) :
  ¬ ∃ (t θ : ℝ),
    (x = 1.5 + t^2) ∧ (y = sqrt(6) * t) ∧ 
    (x = m + 2 * cos(θ)) ∧ (y = sqrt(3) * sin(θ)) ∧ 
    distinct_points x y :=
sorry

end no_four_intersections_l262_262681


namespace count_squares_in_H_l262_262410

-- Define the grid points H
def H : set (ℤ × ℤ) := 
  {p | let (x, y) := p in 
    (2 ≤ abs x ∧ abs x ≤ 6) ∧ (2 ≤ abs y ∧ abs y ≤ 6)}

-- Define squares in H with side lengths at least 4 but at most 5
noncomputable def squaresInH : set (set (ℤ × ℤ)) :=
  {s | ∃ p q r t : ℤ × ℤ, 
       p ∈ H ∧ q ∈ H ∧ r ∈ H ∧ t ∈ H ∧ 
       (∃ (d : ℕ), 4 ≤ d ∧ d ≤ 5 ∧ 
       ((p.1 = q.1 ∧ p.2 + d = q.2 ∧ r.1 = p.1 + d ∧ r.2 = p.2 ∧ t.1 = q.1 + d ∧ t.2 = q.2) ∨ 
        (p.1 + d = q.1 ∧ p.2 = q.2 ∧ r.1 = p.1 ∧ r.2 = p.2 + d ∧ t.1 = q.1 ∧ t.2 = q.2 + d)))}

-- Assert the number of such squares is 4
theorem count_squares_in_H :
  finset.card (finset.filter (λ s, s ∈ squaresInH) (finset.powerset H)) = 4 :=
sorry

end count_squares_in_H_l262_262410


namespace not_all_equal_values_l262_262240

/-- Defining the regular n-gon vertices and corresponding values. -/
variables (n : ℕ) (a : Fin n → ℝ) (O : EuclideanSpace ℝ (Fin 2))

/-- Defining the vector sum S -/
noncomputable def vector_sum (vertices: Fin n → EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  ∑ i in (Finset.univ : Finset (Fin n)), (a i) • (vertices i - O)

/-- Statement of the theorem proving the impossibility -/
theorem not_all_equal_values (vertices : Fin n → EuclideanSpace ℝ (Fin 2)) 
  (h1 : vector_sum n a O vertices ≠ 0) : 
  ¬ ∃ (b : ℝ), ∀ i, a i = b :=
by
  sorry

end not_all_equal_values_l262_262240


namespace geometric_series_common_ratio_l262_262226

theorem geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h_a : a = 500) (h_S : S = 3000) :
  ∃ r : ℝ, r = 5 / 6 :=
by
  sorry

end geometric_series_common_ratio_l262_262226


namespace geometric_series_common_ratio_l262_262842

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l262_262842


namespace parallel_vectors_have_proportional_components_l262_262631

theorem parallel_vectors_have_proportional_components :
  ∀ (x : ℝ), let a := (-2, 1) in let b := (x, -2) in
  (∃ k : ℝ, a = k • b) → x = 4 :=
by
  intro x
  let a := (-2, 1)
  let b := (x, -2)
  intro h
  sorry

end parallel_vectors_have_proportional_components_l262_262631


namespace geometric_series_common_ratio_l262_262847

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l262_262847


namespace sum_of_first_six_terms_of_arithmetic_sequence_is_neg_24_l262_262986

noncomputable def arithmetic_sequence_sum_6 (a₁ : ℕ → ℤ) (d : ℤ) (h₁ : a₁ 1 = 1) (h₂ : d ≠ 0) (h₃ : (a₁ 2 * a₁ 6) = (a₁ 3 ^ 2)) : ℤ :=
  6 * a₁ 1 + (6 * 5) * d / 2

theorem sum_of_first_six_terms_of_arithmetic_sequence_is_neg_24
    (h₁ : a₁ 1 = 1)
    (h₂ : d ≠ 0)
    (h₃ : (a₁ 2 * a₁ 6) = (a₁ 3 ^ 2))
    : arithmetic_sequence_sum_6 (λ n, 1 + (n-1) * d) d h₁ h₂ h₃ = -24 := by
    sorry

end sum_of_first_six_terms_of_arithmetic_sequence_is_neg_24_l262_262986


namespace angle_ABC_l262_262029

noncomputable def A := (-3 : ℝ, 1 : ℝ, 5 : ℝ)
noncomputable def B := (-4 : ℝ, 0 : ℝ, 3 : ℝ)
noncomputable def C := (-5 : ℝ, 0 : ℝ, 4 : ℝ)

def distance (P Q : ℝ × ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

def angle_cosine (A B C : ℝ × ℝ × ℝ) := 
  let AB := distance A B
  let AC := distance A C
  let BC := distance B C
  Real.acos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC))

theorem angle_ABC :
  angle_cosine A B C = 33.557 :=
sorry

end angle_ABC_l262_262029


namespace find_ordered_pair_l262_262254

-- Definitions based on the conditions
variable (a c : ℝ)
def has_exactly_one_solution :=
  (-6)^2 - 4 * a * c = 0

def sum_is_twelve :=
  a + c = 12

def a_less_than_c :=
  a < c

-- The proof statement
theorem find_ordered_pair
  (h₁ : has_exactly_one_solution a c)
  (h₂ : sum_is_twelve a c)
  (h₃ : a_less_than_c a c) :
  a = 3 ∧ c = 9 := 
sorry

end find_ordered_pair_l262_262254


namespace cans_of_beans_is_two_l262_262595

-- Define the problem parameters
variable (C B T : ℕ)

-- Conditions based on the problem statement
axiom chili_can : C = 1
axiom tomato_to_bean_ratio : T = 3 * B / 2
axiom quadruple_batch_cans : 4 * (C + B + T) = 24

-- Prove the number of cans of beans is 2
theorem cans_of_beans_is_two : B = 2 :=
by
  -- Include conditions
  have h1 : C = 1 := by sorry
  have h2 : T = 3 * B / 2 := by sorry
  have h3 : 4 * (C + B + T) = 24 := by sorry
  -- Derive the answer (Proof omitted)
  sorry

end cans_of_beans_is_two_l262_262595


namespace sum_of_highest_powers_10_and_6_dividing_12_l262_262608

noncomputable def legendre (n p : ℕ) : ℕ :=
  let rec aux (q acc : ℕ) :=
    if q = 0 then acc
    else aux (q / p) (acc + q / p)
  aux n 0

theorem sum_of_highest_powers_10_and_6_dividing_12! :
  let b2 := legendre 12 2
  let b5 := legendre 12 5
  let b3 := legendre 12 3
  let highest_power_10 := min b2 b5
  let highest_power_6 := min b2 b3
  highest_power_10 + highest_power_6 = 7 :=
by
  sorry

end sum_of_highest_powers_10_and_6_dividing_12_l262_262608


namespace minimum_speed_increase_l262_262882

def car_speed_increase_required (vA vB vC : ℝ) (dAB dAC : ℝ) (safety_distance new_safety_margin : ℝ) : ℝ :=
  let relative_speed_AB := vA - vB
  let relative_speed_AC := vA + vC
  let time_to_overtake_B := dAB / relative_speed_AB
  let time_to_meet_C := dAC / relative_speed_AC
  let remaining_time := time_to_meet_C - time_to_overtake_B
  let new_distance := vA * remaining_time
  let required_speed := (dAB / (remaining_time * 5280 / new_distance)) - vA
  required_speed

theorem minimum_speed_increase (vA vB vC : ℝ) (dAB dAC : ℝ) (safety_distance new_safety_margin : ℝ) :
  vA = 65 →
  vB = 50 →
  vC = 70 →
  dAB = 50 →
  dAC = 300 →
  safety_distance = 100 →
  new_safety_margin > safety_distance →
  car_speed_increase_required vA vB vC dAB dAC safety_distance new_safety_margin = 20 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end minimum_speed_increase_l262_262882


namespace cost_per_liter_of_gas_today_l262_262132

-- Definition of the conditions
def oil_price_rollback : ℝ := 0.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters := liters_today + liters_friday
def total_cost : ℝ := 39

-- The theorem to prove
theorem cost_per_liter_of_gas_today (C : ℝ) :
  (liters_today * C) + (liters_friday * (C - oil_price_rollback)) = total_cost →
  C = 1.4 := 
by 
  sorry

end cost_per_liter_of_gas_today_l262_262132


namespace range_of_f_l262_262620

noncomputable def f (x : ℝ) : ℝ :=
  (1/4)^x - 3 * (1/2)^x + 2

def in_range (x : ℝ) : Prop :=
  -2 ≤ x ∧ x ≤ 2

theorem range_of_f : 
  (∀ y, (∃ x, in_range x ∧ f x = y) ↔ y ∈ set.interval (-(1/4 : ℝ)) 6) :=
sorry

end range_of_f_l262_262620


namespace max_value_of_expression_l262_262340

noncomputable def maximum_value (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4 * x + 1

def condition (z : ℂ) : Prop :=
  complex.abs (z - 3) = 1

theorem max_value_of_expression :
  ∃ (x y : ℝ), condition (x + y * complex.I) ∧ maximum_value x y = 33 :=
sorry

end max_value_of_expression_l262_262340


namespace percentage_saved_approx_l262_262543

-- Define the conditions: amount saved and amount spent.
def amount_saved : ℝ := 10
def amount_spent : ℝ := 100

-- Define the statement to prove the percentage saved.
theorem percentage_saved_approx :
  (amount_saved / (amount_saved + amount_spent)) * 100 ≈ 9 :=
by
  sorry

end percentage_saved_approx_l262_262543


namespace determinant_range_l262_262082

theorem determinant_range (x : ℝ) : 
  (2 * x - (3 - x) > 0) ↔ (x > 1) :=
by
  sorry

end determinant_range_l262_262082


namespace botanical_guide_limit_l262_262712

theorem botanical_guide_limit (m : ℕ) (h : ∀ (plants : Fin m → Fin 100 → Bool), 
  ∀ i j : Fin m, i ≠ j → (Finset.filter 
  (λ k : Fin 100, plants i k ≠ plants j k) Finset.univ).card > 50) : 
  m ≤ 50 :=
sorry

end botanical_guide_limit_l262_262712


namespace average_student_headcount_fall_terms_l262_262593

theorem average_student_headcount_fall_terms :
  let headcount_03_04 := 11500
  let headcount_04_05 := 11300
  let headcount_05_06 := 11400
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11400 :=
by
  -- Defining the headcounts
  let headcount_03_04 := 11500
  let headcount_04_05 := 11300
  let headcount_05_06 := 11400
  -- The average calculation
  have avg_headcount : (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11400, from sorry,
  exact avg_headcount

end average_student_headcount_fall_terms_l262_262593


namespace right_triangle_unique_perimeter_18_l262_262695

theorem right_triangle_unique_perimeter_18 :
  ∃! (a b c : ℤ), a^2 + b^2 = c^2 ∧ a + b + c = 18 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end right_triangle_unique_perimeter_18_l262_262695


namespace tan_theta_minus_pi_over_4_l262_262703

variable (θ : ℝ)

def z (θ : ℝ) : ℂ := (Real.sin θ - 3/5) + (Real.cos θ - 4/5) * Complex.I

theorem tan_theta_minus_pi_over_4 (h : ∀ θ, z θ = (0 : ℝ) + (z θ).im * Complex.I) :
  Real.tan (θ - Real.pi / 4) = -7 :=
by
  sorry

end tan_theta_minus_pi_over_4_l262_262703


namespace combination_n_2_l262_262521

theorem combination_n_2 (n : ℕ) (h : n > 0) : 
  nat.choose n 2 = n * (n - 1) / 2 :=
sorry

end combination_n_2_l262_262521


namespace part_a_part_b_l262_262027

variables {Point : Type} [EuclideanGeometry Point]
variables (A B C P : Point)
variables (inside_BAC : inside ∠A B C P) (outside_ABC : outside ΔA B C P)

def circumcenter (X Y Z : Point) : Point := PlaneEuclideanGeometry.circumcenter X Y Z  -- Placeholder definition

variables {O_A O_B O_C : Point}
variables (circumcenter_PBC_PA : O_A = circumcenter P B C) (on_ray_PA : O_A ∈ ray P A)
variables (circumcenter_CPA_PB : O_B = circumcenter C P A) (on_ray_PB : O_B ∈ ray P B)
variables (circumcenter_APB_PC : O_C = circumcenter A P B) (on_ray_PC : O_C ∈ ray P C)

-- Statement of part (a):
theorem part_a :
  (on_ray_PA ∧ on_ray_PB) → on_ray_PC :=
sorry

-- Statement of part (b):
theorem part_b :
  (on_ray_PA ∧ on_ray_PB ∧ on_ray_PC) →
  ( ∃ (circumcircle_ABC : Circle A B C),
    O_A ∈ circumcircle_ABC ∧ O_B ∈ circumcircle_ABC ∧ O_C ∈ circumcircle_ABC ) :=
sorry

end part_a_part_b_l262_262027


namespace equal_product_of_distances_l262_262829

theorem equal_product_of_distances
  (A B C D O : Point)
  (circle : Circle)
  (H K L M : Point)
  (h1 : InscribedQuadrilateral A B C D circle center O)
  (h2 : CenterInQuadrilateral O A B C D)
  (h3 : TangentsIntersect A C (SymmetricLine B D O)) :
  ProductDistances O H L M K :=
sorry

end equal_product_of_distances_l262_262829


namespace distance_between_points_l262_262819

theorem distance_between_points : abs (3 - (-2)) = 5 := 
by
  sorry

end distance_between_points_l262_262819


namespace radius_of_sphere_eq_6_l262_262546

def surface_area_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2
def curved_surface_area_cylinder (r_cylinder h : ℝ) : ℝ := 2 * Real.pi * r_cylinder * h

theorem radius_of_sphere_eq_6 : 
  (∃ r : ℝ, surface_area_sphere r = curved_surface_area_cylinder 6 12 ∧ r = 6) :=
by 
  use 6
  unfold surface_area_sphere curved_surface_area_cylinder
  rw [Real.pi_mul 4 (6^2), Real.pi_mul 2 (6 * 12)]
  exact And.intro rfl rfl
  -- sorry -- Skipping the detailed proof steps

end radius_of_sphere_eq_6_l262_262546


namespace determinant_range_l262_262081

theorem determinant_range (x : ℝ) : 
  (2 * x - (3 - x) > 0) ↔ (x > 1) :=
by
  sorry

end determinant_range_l262_262081


namespace flash_catchup_distance_l262_262294

variable (v x c y : ℝ)
variable (h1 : c > 0) (h2 : x > 1)

theorem flash_catchup_distance (v x c y : ℝ) (h1 : c > 0) (h2 : x > 1) : 
  let flash_speed := x * v + c
  let ace_speed := v
  let ace_head_start := 2 * y
  (flash_speed * (2 * y / (v * (x - 1) + c)) = 2 * y) :=
begin
  sorry,
end

end flash_catchup_distance_l262_262294


namespace find_m_independent_quadratic_term_l262_262302

def quadratic_poly (m : ℝ) (x : ℝ) : ℝ :=
  -3 * x^2 + m * x^2 - x + 3

theorem find_m_independent_quadratic_term (m : ℝ) :
  (∀ x, quadratic_poly m x = -x + 3) → m = 3 :=
by 
  sorry

end find_m_independent_quadratic_term_l262_262302


namespace chess_tournament_ratio_l262_262368

theorem chess_tournament_ratio:
  ∃ n : ℕ, (n * (n - 1)) / 2 = 231 ∧ (n - 1) = 21 := 
sorry

end chess_tournament_ratio_l262_262368


namespace sum_of_distances_minimized_l262_262716

theorem sum_of_distances_minimized (x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) : 
  abs (x - 0) + abs (x - 50) = 50 := 
by
  sorry

end sum_of_distances_minimized_l262_262716


namespace xiao_hua_seat_correct_l262_262014

-- Define the classroom setup
def classroom : Type := ℤ × ℤ

-- Define the total number of rows and columns in the classroom.
def total_rows : ℤ := 7
def total_columns : ℤ := 8

-- Define the position of Xiao Ming's seat.
def xiao_ming_seat : classroom := (3, 7)

-- Define the position of Xiao Hua's seat.
def xiao_hua_seat : classroom := (5, 2)

-- Prove that Xiao Hua's seat is designated as (5, 2)
theorem xiao_hua_seat_correct : xiao_hua_seat = (5, 2) := by
  -- The proof would go here
  sorry

end xiao_hua_seat_correct_l262_262014


namespace greatest_int_leq_expr_l262_262163

theorem greatest_int_leq_expr : 
  (⌊ (5^100 + 3^100) / (5^96 + 3^96) ⌋ = 624) :=
by sorry

end greatest_int_leq_expr_l262_262163


namespace distance_from_R_to_midpoint_of_PQ_l262_262724

-- Definitions and assumptions based on the conditions given
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

variables (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (PQ PR QR: ℝ)

-- Assign given values
def PQ_val : ℝ := 15
def PR_val : ℝ := 9
def QR_val : ℝ := 12

-- Define the midpoint of segment PQ
def midpoint (P Q : ℝ) := (P + Q) / 2

-- The theorem statement
theorem distance_from_R_to_midpoint_of_PQ :
  is_right_triangle PR_val QR_val PQ_val → 
  midpoint PQ_val 0 = PQ_val / 2 → 
  dist (0 : ℝ) (midpoint PQ_val 0) = 7.5 :=
by
  sorry

end distance_from_R_to_midpoint_of_PQ_l262_262724


namespace sequence_value_x_l262_262016

theorem sequence_value_x (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 5) 
  (h3 : a3 = 11) 
  (h4 : a4 = 20) 
  (h5 : a6 = 47)
  (h6 : a2 - a1 = 3) 
  (h7 : a3 - a2 = 6) 
  (h8 : a4 - a3 = 9) 
  (h9 : a6 - a5 = 15) : 
  a5 = 32 :=
sorry

end sequence_value_x_l262_262016


namespace false_statement_C_l262_262423

def is_real (z : ℂ) : Prop := ∃ a : ℝ, z = a
def is_imaginary (z : ℂ) : Prop := ∃ b : ℝ, z = b * complex.I
def is_pure_imaginary (z : ℂ) : Prop := ∃ b : ℝ, z = b * complex.I ∧ 0 < b
def is_real_or_pure_imaginary (z : ℂ) : Prop := is_real z ∨ is_pure_imaginary z

theorem false_statement_C (z : ℂ) : (∃ b : ℝ, z = b * complex.I → z^ 2 ≥ 0) = false := by
  -- Placeholder for proof
  sorry

end false_statement_C_l262_262423


namespace calculate_pentagons_l262_262591

theorem calculate_pentagons 
  (triangles : ℕ) (squares : ℕ) (total_lines : ℕ) (triangles_lines : ℕ) (squares_lines : ℕ) :
  triangles = 12 ∧ squares = 8 ∧ total_lines = 88 ∧ triangles_lines = 3 ∧ squares_lines = 4 →
  let lines_triangles := triangles * triangles_lines,
      lines_squares := squares * squares_lines,
      lines_pentagons := total_lines - (lines_triangles + lines_squares),
      pentagons := lines_pentagons / 5 in
  pentagons = 4 :=
  by 
    sorry

end calculate_pentagons_l262_262591


namespace perpendicular_inequality_l262_262872

theorem perpendicular_inequality
  (ABC : Triangle)
  (circumcircle : Circle)
  (tangent_B : Tangent)
  (tangent_C : Tangent)
  (F : Point)
  (M : Point)
  (L : Point)
  (N : Point)
  (h_tangents_intersect: tangents_intersect_at tangents_B tangents_C F)
  (h_M_foot : foot_of_perpendicular A tangent_B M)
  (h_L_foot : foot_of_perpendicular A tangent_C L)
  (h_N_foot : foot_of_perpendicular A (BC ABC) N) :
  length_segment A M + length_segment A L >= 2 * length_segment A N :=
sorry

end perpendicular_inequality_l262_262872


namespace rational_sqrt2_l262_262134

variables (M : Set ℝ)
variables [fintype M]
variables (h_distinct : card M = 2003)
variables (h_condition : ∀ (a b : ℝ), a ∈ M → b ∈ M → a ≠ b → is_rat (a ^ 2 + b * real.sqrt 2))

theorem rational_sqrt2 (a : ℝ) (ha : a ∈ M) : is_rat (a * real.sqrt 2) :=
sorry

end rational_sqrt2_l262_262134


namespace caught_fish_l262_262996

variables (cost_of_game earnings_last_week additional_needed earnings_from_trout earnings_from_bluegill : ℝ)
variables (fraction_trout fraction_bluegill : ℝ)
variables (total_earnings_needed total_earnings_this_sunday : ℝ)

-- Given conditions
def conditions := 
  cost_of_game = 60 ∧
  earnings_last_week = 35 ∧
  additional_needed = 2 ∧
  earnings_from_trout = 5 ∧
  earnings_from_bluegill = 4 ∧
  fraction_trout = 0.6 ∧
  fraction_bluegill = 0.4 ∧
  total_earnings_needed = cost_of_game - earnings_last_week - additional_needed ∧
  total_earnings_this_sunday = 23

-- Question rephrased as a theorem
theorem caught_fish (total_fish : ℝ) (h : conditions) :
  fraction_trout * total_fish * earnings_from_trout + fraction_bluegill * total_fish * earnings_from_bluegill = total_earnings_this_sunday →
  total_fish = 5 :=
begin
  intros h_eq,
  sorry
end

end caught_fish_l262_262996


namespace square_cut_into_five_triangles_l262_262265

-- Given a square with side length s
def square_area (s : ℝ) : ℝ := s * s

-- Define the area of a triangle formed by cutting along the diagonal
def triangle_area (s : ℝ) : ℝ := (1 / 2) * square_area(s)

-- Define the condition of the problem
theorem square_cut_into_five_triangles (s : ℝ) : 
  ∃ (A₁ A₂ A₃ A₄ A₅ : ℝ), 
    A₁ = triangle_area s ∧ 
    (A₂ + A₃ + A₄ + A₅ = triangle_area s) := by
  sorry

end square_cut_into_five_triangles_l262_262265


namespace vertical_shift_d_l262_262994

variable (a b c d : ℝ)

theorem vertical_shift_d (h1: d + a = 5) (h2: d - a = 1) : d = 3 := 
by
  sorry

end vertical_shift_d_l262_262994


namespace f_increasing_exists_a_odd_f_l262_262674

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: Monotonicity of the function f
theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  sorry

-- Theorem 2: Existence of a real number a such that f is an odd function and finding that a
theorem exists_a_odd_f : ∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x ∧ a = 1 :=
by
  use 1
  split
  intro x
  have h1 : f 1 (-x) = 1 - 2 / (2^(-x) + 1) := rfl
  have h2 : f 1 x = 1 - 2 / (2^x + 1) := rfl
  have h3 : 2^(-x) = 1 / (2^x) := sorry
  have h4 : 1 - 2 / (1 / (2^x) + 1) = - (1 - 2 / (2^x + 1)) := sorry
  rw [h1, h2, ←h3, h4]
  trivial
  exact rfl

end f_increasing_exists_a_odd_f_l262_262674


namespace sum_of_nth_row_largest_n_for_sum_1988_specific_case_proof_no_larger_n_possible_l262_262936

noncomputable def a : ℕ := 2
noncomputable def d : ℕ := 30

-- Statement for Part 1
theorem sum_of_nth_row (a d : ℕ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  let sum := (λ n : ℕ, (finset.range (n + 1)).sum (λ i, (a + i * d) * (nat.choose n i)))
  in sum n = 2^n * a + (2^n - 2) * d :=
sorry

-- Statement for Part 2
theorem largest_n_for_sum_1988 (n : ℕ) (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  (2^n * a + (2^n - 2) * d = 1988) → (∀ m, (2^m * a + (2^m - 2) * d = 1988) → m ≤ n) :=
sorry

-- Definitions for Part 2 specific values
def largest_possible_n := 6  -- based on the solution provided
def corresponding_a := 2
def corresponding_d := 30

-- Prove these specific cases
theorem specific_case_proof :
  2^largest_possible_n * corresponding_a + (2^largest_possible_n - 2) * corresponding_d = 1988 :=
sorry

theorem no_larger_n_possible (n : ℕ) :
  (2^n * corresponding_a + (2^n - 2) * corresponding_d = 1988) → n <= largest_possible_n :=
sorry

end sum_of_nth_row_largest_n_for_sum_1988_specific_case_proof_no_larger_n_possible_l262_262936


namespace increase_80_by_135_percent_l262_262903

theorem increase_80_by_135_percent : 
  let original := 80 
  let increase := 1.35 
  original + (increase * original) = 188 := 
by
  sorry

end increase_80_by_135_percent_l262_262903


namespace binom_n_2_l262_262524

theorem binom_n_2 (n : ℕ) (h : 1 < n) : (nat.choose n 2) = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262524


namespace parallelogram_ABCD_l262_262427

-- Define the geometrical structures and conditions.
variables {Point : Type} [AffineSpace Point]

-- Given points and segments.
variables {A B C D E F G H : Point}
variable {BD : Line Point}
variable {quadrilateral : ConvexQuadrilateral A B C D}
variable {segment_AB : Segment A B}
variable {segment_AD : Segment A D}
variable {segment_EF : Segment E F}
variable {segment_CE : Segment C E}
variable {segment_CF : Segment C F}

-- Conditions for the problem
axiom point_on_segment_AB : E ∈ segment_AB
axiom point_on_segment_AD : F ∈ segment_AD
axiom segment_parallel_BD : segment_EF ∥ BD
axiom segment_intersect_BD_CE : G ∈ Intersection BD (LineThrough C E)
axiom segment_intersect_BD_CF : H ∈ Intersection BD (LineThrough C F)
axiom quadrilateral_AGCH : Parallelogram A G C H

-- The theorem statement
theorem parallelogram_ABCD (h1 : E ∈ segment_AB) (h2 : F ∈ segment_AD) (h3 : segment_EF ∥ BD)
  (h4 : G ∈ Intersection BD (LineThrough C E)) (h5 : H ∈ Intersection BD (LineThrough C F))
  (h6 : Parallelogram A G C H) : Parallelogram A B C D :=
sorry

end parallelogram_ABCD_l262_262427


namespace initial_blue_balls_eq_nine_l262_262176

-- Definitions of the conditions and what needs to be proved
variable (B : ℕ) (h_initial : -5 < B) (h_total : B < 26)
variable (h_prob : (B - 5 : ℚ) / 20 = 1 / 5)

theorem initial_blue_balls_eq_nine 
  (h_initial : B ≥ 5)
  (h_total : B ≤ 25) 
  (h_prob : (B - 5 : ℚ) / 20 = 1 / 5) :
  B = 9 :=
by
  sorry

end initial_blue_balls_eq_nine_l262_262176


namespace proof_problem_l262_262702

-- Define the complex number
def i := Complex.i
def x := (1 + i * Real.sqrt 3) / 2

-- State the theorem
theorem proof_problem : (1 : ℂ) / (x^2 - x) = -1 := by
  sorry

end proof_problem_l262_262702


namespace equilateral_triangle_l262_262644

-- Definitions based on the conditions
structure Point where
  x : ℝ
  y : ℝ

def rotate_120 (A : Point) (P : Point) : Point :=
  -- Simplified rotation for demonstration; actual implementation requires proper trigonometric rotation logic
  sorry

def seq_points (A : ℕ → Point) (P₀ : Point) : ℕ → Point
| 0       => P₀
| (n + 1) => rotate_120 (A (n + 1)) (seq_points A n)

-- Main statement
theorem equilateral_triangle
  (A₁ A₂ A₃ : Point)
  (P₀ : Point)
  (A : ℕ → Point)
  (hA : ∀ s ≥ 4, A s = A (s - 3))
  (hP : seq_points A P₀ 1986 = P₀) :
  -- Prove that the triangle A₁ A₂ A₃ is equilateral
  is_equilateral (A₁, A₂, A₃) :=
sorry

end equilateral_triangle_l262_262644


namespace find_number_l262_262186

theorem find_number (x : ℝ) :
  (real.sqrt ((x + 10)^3) - 2 = 54) → x = 5 :=
by
  intro h
  sorry

end find_number_l262_262186


namespace minimum_die_rolls_needed_l262_262356

theorem minimum_die_rolls_needed (p q : ℝ) (ε : ℝ) (n : ℕ) (h : p = 1 / 6 ∧ q = 5 / 6 ∧ ε = 0.01) :
  (2 * (Real.cdf 0.6745) ≥ 1 - 2 * (Real.cdf 0.6745)) → n ≥ 632 :=
by
  intros
  sorry

end minimum_die_rolls_needed_l262_262356


namespace homework_total_l262_262453

theorem homework_total :
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  math_pages + reading_pages + science_pages = 62 :=
by
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  show math_pages + reading_pages + science_pages = 62
  sorry

end homework_total_l262_262453


namespace cuboid_surface_area_500_l262_262880

def surface_area (w l h : ℝ) : ℝ :=
  2 * l * w + 2 * l * h + 2 * w * h

theorem cuboid_surface_area_500 :
  ∀ (w l h : ℝ), w = 4 → l = w + 6 → h = l + 5 →
  surface_area w l h = 500 :=
by
  intros w l h hw hl hh
  unfold surface_area
  rw [hw, hl, hh]
  norm_num
  sorry

end cuboid_surface_area_500_l262_262880


namespace trains_cross_time_l262_262946

def time_to_cross (length_train1 speed_train1 length_train2 speed_train2 : ℕ) : ℕ :=
  let speed_train1_mps := speed_train1 * (5 / 18)
  let speed_train2_mps := speed_train2 * (5 / 18)
  let relative_speed   := speed_train1_mps + speed_train2_mps
  let total_length     := length_train1 + length_train2
  total_length / relative_speed

theorem trains_cross_time : 
  time_to_cross 300 120 200.04 80 = 9 := 
by
  sorry

end trains_cross_time_l262_262946


namespace part1_part2_l262_262594

-- Part 1
theorem part1 (x y : ℝ) : (2 * x - 3 * y) ^ 2 - (y + 3 * x) * (3 * x - y) = -5 * x ^ 2 - 12 * x * y + 10 * y ^ 2 := 
sorry

-- Part 2
theorem part2 : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) - 2 ^ 16 = -1 := 
sorry

end part1_part2_l262_262594


namespace subset_inequality_l262_262425

theorem subset_inequality (X : Finset α) (m n : ℕ) (A : Fin n → Finset α)
  (h1 : ∀ i, A i ⊆ X)
  (h2 : ∀ i, ¬ A i = ∅)
  (h3 : (Finset.bUnion Finset.univ A) = X) :
  ∑ i in Finset.univ, ∑ j in Finset.univ, (A i).card * ((A i) ∩ (A j)).card ≥
  (1 : ℝ) / (m * n) * (∑ i in Finset.univ, (A i).card)^3 := 
sorry

end subset_inequality_l262_262425


namespace segment_lengths_l262_262006

noncomputable def radius : ℝ := 5
noncomputable def diameter : ℝ := 2 * radius
noncomputable def chord_length : ℝ := 8

-- The lengths of the segments AK and KB
theorem segment_lengths (x : ℝ) (y : ℝ) 
  (hx : 0 < x ∧ x < diameter) 
  (hy : 0 < y ∧ y < diameter) 
  (h1 : x + y = diameter) 
  (h2 : x * y = (diameter^2) / 4 - 16 / 4) : 
  x = 2.5 ∧ y = 7.5 := 
sorry

end segment_lengths_l262_262006


namespace allison_craft_supplies_l262_262220

theorem allison_craft_supplies :
  ∀ (glue_sticks_marie packs_paper_marie : ℕ)
    (h_glue_sticks_diff : ∃ glue_sticks_allison, glue_sticks_allison = glue_sticks_marie + 8)
    (h_packs_paper_ratio : ∃ packs_paper_allison, packs_paper_marie = 6 * packs_paper_allison),
    glue_sticks_marie = 15 →
    packs_paper_marie = 30 →
    ∃ glue_sticks_allison packs_paper_allison,
      glue_sticks_allison = 23 ∧ packs_paper_allison = 5 ∧ glue_sticks_allison + packs_paper_allison = 28 :=
by
  intros glue_sticks_marie packs_paper_marie h_glue_sticks_diff h_packs_paper_ratio h_marie_glue h_marie_paper
  obtain ⟨glue_sticks_allison, h_glue_sticks_allison⟩ := h_glue_sticks_diff
  obtain ⟨packs_paper_allison, h_packs_paper_allison⟩ := h_packs_paper_ratio
  use glue_sticks_allison, packs_paper_allison
  rw [h_marie_glue, h_marie_paper] at *
  have h_glue_sticks_allison_eq : glue_sticks_allison = 15 + 8 := h_glue_sticks_allison
  have h_packs_paper_allison_eq : packs_paper_allison = 30 / 6 := (eq.symm (nat.div_exact (by norm_num) (by norm_num 6))).mp h_packs_paper_allison
  rw [h_glue_sticks_allison_eq, h_packs_paper_allison_eq]
  exact ⟨rfl, rfl, by norm_num⟩

end allison_craft_supplies_l262_262220


namespace left_term_right_term_l262_262026

def seq_construct (R : ℕ → list ℕ) : ℕ → list ℕ
  | 0 => []
  | 1 => [1]
  | n + 1 => 
    let prev := R n
    (prev.foldr (λ x acc => acc ++ [1, 2, ... , x]) []).append [n + 1]

def left_term (k : ℕ) (seq : list ℕ) : ℕ :=
  seq.nth k ∘ Option.getD 1

def right_term (k : ℕ) (seq : list ℕ) : ℕ :=
  seq.reverse.nth k ∘ Option.getD 1

theorem left_term_right_term (n k : ℕ) (h : n > 1) (R : ℕ → list ℕ) :
  (left_term k (R n) = 1 ↔ right_term k (R n) ≠ 1) :=
sorry

end left_term_right_term_l262_262026


namespace product_of_two_numbers_ratio_l262_262159

theorem product_of_two_numbers_ratio (x y : ℝ)
  (h1 : x - y ≠ 0)
  (h2 : x + y = 4 * (x - y))
  (h3 : x * y = 18 * (x - y)) :
  x * y = 86.4 :=
by
  sorry

end product_of_two_numbers_ratio_l262_262159


namespace exists_n_2000_prime_divisors_and_divides_two_power_n_plus_1_l262_262272

theorem exists_n_2000_prime_divisors_and_divides_two_power_n_plus_1 :
  ∃ (n : ℕ), (∀ p : ℕ, p.prime → n.factors.count p = 2000) ∧ n ∣ (2^n + 1) :=
sorry

end exists_n_2000_prime_divisors_and_divides_two_power_n_plus_1_l262_262272


namespace betty_cookies_brownies_l262_262235

theorem betty_cookies_brownies :
  let initial_cookies := 60
  let initial_brownies := 10
  let cookies_per_day := 3
  let brownies_per_day := 1
  let days := 7
  let remaining_cookies := initial_cookies - cookies_per_day * days
  let remaining_brownies := initial_brownies - brownies_per_day * days
  remaining_cookies - remaining_brownies = 36 :=
by
  sorry

end betty_cookies_brownies_l262_262235


namespace triangle_shape_l262_262003

-- Definitions for conditions
def Triangle (A B C : Type) := A ∧ B ∧ C  -- representing a triangle type
variables {α : Type} [AddGroup α] [MulAction ℝ α]

def cos : α → ℝ := sorry
def sin : α → ℝ := sorry

-- Theorem statement
theorem triangle_shape 
  (A B C : α) -- Angles of triangle 
  (h1 : Triangle A B C)
  (h2 : (cos A + 2 * cos C) / (cos A + 2 * cos B) = sin B / sin C) : 
  -- Conclusion that ABC is either isosceles or right
  (isosceles A B C ∨ right_triangle A B C) :=
sorry

end triangle_shape_l262_262003


namespace mean_and_median_of_sequence_l262_262974

def arithmetic_sequence (n : ℕ) (a d : ℤ) : ℤ := a + (n - 1) * d

def S (n : ℕ) (a d : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem mean_and_median_of_sequence (a₃ : ℤ) (S₄ : ℤ) (h₃ : a₃ = 8) (h₄ : S₄ = 28) :
  ∃ a d : ℤ, d ≠ 0 ∧ arithmetic_sequence 3 a d = a₃ ∧ S 4 a d = S₄ ∧ 
  let sample := (finset.range 20).image (λ n, arithmetic_sequence (n+1) a d)
  ∑ x in sample, x / 20 = 23 ∧
  (sample.to_list.nth 9 + sample.to_list.nth 10) / 2 = 23 :=
by
  sorry

end mean_and_median_of_sequence_l262_262974


namespace two_zeros_in_interval_l262_262346

theorem two_zeros_in_interval (ω : ℝ) (hω : ω > 0) :
  (∃ x₁ x₂ ∈ set.Icc (0 : ℝ) (Real.pi / 2), 
    x₁ ≠ x₂ ∧ (sin (2 * ω * x₁ + 2 * Real.pi / 3) = sqrt 3 / 2) ∧ 
    (sin (2 * ω * x₂ + 2 * Real.pi / 3) = sqrt 3 / 2)) ↔ 
    (5 / 3 ≤ ω ∧ ω < 2) :=
sorry

end two_zeros_in_interval_l262_262346


namespace ellipse_properties_range_OP_OQ_MP_MQ_l262_262646

theorem ellipse_properties 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
  (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ C)
  (major_axis_len : 2 * a = 8)
  (T : ℝ × ℝ) (T_on_ellipse : T.1^2 / 16 + T.2^2 / 12 = 1)
  (k1 k2 : ℝ) (k1_eq : k1 = T.2 / (T.1 + 4)) (k2_eq : k2 = T.2 / (T.1 - 4)) :
  k1 * k2 = -3 / 4 := sorry

theorem range_OP_OQ_MP_MQ
  (ellipse_eq : ∀ (x y : ℝ), x^2 / 16 + y^2 / 12 = 1 ↔ (x, y) ∈ C)
  (O : ℝ × ℝ) (O_eq : O = (0, 0))
  (M : ℝ × ℝ) (M_eq : M = (0, 2))
  (P Q : ℝ × ℝ) (P_on_ellipse : P.1^2 / 16 + P.2^2 / 12 = 1) (Q_on_ellipse : Q.1^2 / 16 + Q.2^2 / 12 = 1)
  (line_through_M : ∀ (x y : ℝ), y = k * x + 2 ↔ (x, y) ∈ PQ)
  (intersect_PQ : ∃ PQ : set (ℝ × ℝ), ∃ k : ℝ, line_through_M PQ)
  : -20 ≤ (O.1 * P.1 + O.2 * P.2) * (O.1 * Q.1 + O.2 * Q.2) + (M.1 * P.1 + M.2 * P.2) * (M.1 * Q.1 + M.2 * Q.2) ∧ 
    (O.1 * P.1 + O.2 * P.2) * (O.1 * Q.1 + O.2 * Q.2) + (M.1 * P.1 + M.2 * P.2) * (M.1 * Q.1 + M.2 * Q.2) ≤ -52 / 3 := sorry

end ellipse_properties_range_OP_OQ_MP_MQ_l262_262646


namespace median_length_of_shorter_arc_l262_262207

noncomputable def median_shorter_arc_length (track_length : ℝ) (A B : ℝ) : ℝ :=
if |A - B| ≤ track_length / 2 then |A - B| else track_length - |A - B|

theorem median_length_of_shorter_arc (track_length : ℝ) (h : track_length = 1) :
  ∀ A B, (A ∈ [0, track_length) ∧ B ∈ [0, track_length)) →
  ∃ m, m = 0.25 ∧ 
       (∀ (x : ℝ), (x > m → 0.5 < (SetOf (λ P : ℝ × ℝ, 
         (median_shorter_arc_length track_length P.fst P.snd) > x)).measure) ∧ 
       (x = m → 0.5 = (SetOf (λ P : ℝ × ℝ, 
         (median_shorter_arc_length track_length P.fst P.snd) > x)).measure)) :=
sorry

end median_length_of_shorter_arc_l262_262207


namespace func_eq_condition_l262_262283

variable (a : ℝ)

theorem func_eq_condition (f : ℝ → ℝ) :
  (∀ x : ℝ, f (Real.sin x) + a * f (Real.cos x) = Real.cos (2 * x)) ↔ a ∈ (Set.univ \ {1} : Set ℝ) :=
by
  sorry

end func_eq_condition_l262_262283


namespace three_pow_max_factorial_l262_262175

theorem three_pow_max_factorial (k : ℕ) : 
  (∃ k : ℕ, ∀ m : ℕ, k ≥ m → ¬ (3^m ∣ fact 30)) → k = 14 := by
  sorry

end three_pow_max_factorial_l262_262175


namespace minimum_value_l262_262658

open Real

noncomputable def min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (condition : 2 * x + y = 2) : ℝ :=
  (1 / x) + (2 / y)

theorem minimum_value : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ min_value x y (by linarith) (by linarith) (by linarith) = 4 :=
begin
  use [1/4, 1/2],
  split, {exact by norm_num1},
  split, {exact by norm_num1},
  split, {exact by norm_num},
  sorry
end

end minimum_value_l262_262658


namespace acute_angled_triangles_in_convex_ngon_l262_262569

theorem acute_angled_triangles_in_convex_ngon
  (n : ℕ)
  (h_n_ge_3 : n ≥ 3)
  (polygon : ∀ i j, i ≠ j → dist (A i) (A j) ≠ 0)
  (inscribed_circle : ∀ i, ∃ O r, dist O (A i) = r)
  (no_diametric_opposite : ∀ i j, i ≠ j → dist (A i) (A j) ≠ 2 * r)
  (exists_acute_triangle : ∃ (i j k : ℕ), i < j ∧ j < k ∧ acute_angle (A i) (A j) (A k)) :
  ∃ (acute_triangle_count : ℕ), acute_triangle_count ≥ n - 2 :=
begin
  sorry
end

end acute_angled_triangles_in_convex_ngon_l262_262569


namespace tiling_problem_l262_262985

noncomputable def number_of_tilings (n : ℕ) (colors : ℕ) (board : ℕ) :=
  -- Assume an abstract function that calculates the number of valid tilings
  sorry

theorem tiling_problem :
  let m := 2 in
  let colors := 3 in
  let board := 8 in
  number_of_tilings m colors board % 1000 = 78 :=
by
  sorry

end tiling_problem_l262_262985


namespace united_call_charge_l262_262507

theorem united_call_charge 
  (base_united : ℝ)
  (base_atlantic : ℝ)
  (charge_atlantic : ℝ)
  (minutes : ℝ)
  (h_equal_bills : base_united + minutes * (charge_united) = base_atlantic + minutes * charge_atlantic)
  (base_united = 8)
  (base_atlantic = 12)
  (charge_atlantic = 0.2)
  (minutes = 80) : charge_united = 0.25 := 
  sorry

end united_call_charge_l262_262507


namespace distinct_sums_products_under_15_l262_262258

def is_positive_odd (n : ℕ) : Prop := n % 2 = 1 ∧ n > 0

noncomputable def possible_values : ℕ → ℕ → ℕ := λ p q, p * q + p + q

theorem distinct_sums_products_under_15 : 
  {pq_sum | ∃ p q : ℕ, is_positive_odd p ∧ is_positive_odd q ∧ p < 15 ∧ q < 15 ∧ pq_sum = possible_values p q}.to_finset.card = 28 :=
sorry

end distinct_sums_products_under_15_l262_262258


namespace min_value_ineq_inequality_proof_l262_262632

variable (a b x1 x2 : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx1_pos : 0 < x1) (hx2_pos : 0 < x2) (hab_sum : a + b = 1)

-- First problem: Prove that the minimum value of the given expression is 6.
theorem min_value_ineq : (x1 / a) + (x2 / b) + (2 / (x1 * x2)) ≥ 6 := by
  sorry

-- Second problem: Prove the given inequality.
theorem inequality_proof : (a * x1 + b * x2) * (a * x2 + b * x1) ≥ x1 * x2 := by
  sorry

end min_value_ineq_inequality_proof_l262_262632


namespace max_consecutive_sum_l262_262888

theorem max_consecutive_sum (a N : ℤ) (h₀ : N > 0) (h₁ : N * (2 * a + N - 1) = 90) : N = 90 :=
by
  -- Proof to be provided
  sorry

end max_consecutive_sum_l262_262888


namespace find_triples_l262_262616

theorem find_triples (a m n : ℕ) (k : ℕ):
  a ≥ 2 ∧ m ≥ 2 ∧ a^n + 203 ≡ 0 [MOD a^m + 1] ↔ 
  (a = 2 ∧ ((n = 4 * k + 1 ∧ m = 2) ∨ (n = 6 * k + 2 ∧ m = 3) ∨ (n = 8 * k + 8 ∧ m = 4) ∨ (n = 12 * k + 9 ∧ m = 6))) ∨
  (a = 3 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 4 ∧ n = 4 * k + 4 ∧ m = 2) ∨
  (a = 5 ∧ n = 4 * k + 1 ∧ m = 2) ∨
  (a = 8 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 10 ∧ n = 4 * k + 2 ∧ m = 2) ∨
  (a = 203 ∧ n = (2 * k + 1) * m + 1 ∧ m ≥ 2) := by sorry

end find_triples_l262_262616


namespace athena_total_spent_l262_262736

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end athena_total_spent_l262_262736


namespace problem_statement_l262_262309

variables {Point Line Plane : Type}
-- Assuming basic definitions related to geometric concepts
variable (m n : Line)
variable (α β : Plane)

-- Assuming the existence of perpendicularity and parallelism between lines and planes
variables (perpendicular parallel : Line → Plane → Prop)

-- The given conditions from the problem
variables (m_perp_α : perpendicular m α)
variables (n_perp_β : perpendicular n β)
variables (m_parallel_n : parallel m n)

theorem problem_statement : (∃ (perpendicular parallel : Plane → Plane → Prop), ∀ (α β : Plane), perpendicular α β ↔ parallel α β) →
  α ∥ β :=
by
  sorry

end problem_statement_l262_262309


namespace problem_statement_l262_262769

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_statement :
  (∀ x y : ℝ, f ((x - y)^2 + 1) = f x^2 - 3 * x * f y + y^2) →
  let n := (card { y : ℝ | ∃ x : ℝ, f x = y }).card in
  let s := Σ' y in { y : ℝ | ∃ x : ℝ, f x = y }, y in
  n * s = (22 : ℝ) / 3 :=
sorry

end problem_statement_l262_262769


namespace median_moons_is_2_l262_262303

-- Definition of the list of moons
def moon_counts : List ℕ := [0, 0, 1, 2, 20, 22, 14, 2, 5, 0, 1, 2, 3]

-- Function to compute median for a list of natural numbers
def median (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (· ≤ ·)
  sorted_lst[(sorted_lst.length / 2)]

-- Proof statement
theorem median_moons_is_2 : median moon_counts = 2 := 
  sorry

end median_moons_is_2_l262_262303


namespace speed_of_stream_l262_262966

theorem speed_of_stream
  (b s : ℝ)
  (H1 : 120 = 2 * (b + s))
  (H2 : 60 = 2 * (b - s)) :
  s = 15 :=
by
  sorry

end speed_of_stream_l262_262966


namespace shaded_region_area_l262_262370

noncomputable def radius_large : ℝ := 10
noncomputable def radius_small : ℝ := 4

theorem shaded_region_area :
  let area_large := Real.pi * radius_large^2 
  let area_small := Real.pi * radius_small^2 
  (area_large - 2 * area_small) = 68 * Real.pi :=
by
  sorry

end shaded_region_area_l262_262370


namespace athena_total_spent_l262_262732

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end athena_total_spent_l262_262732


namespace vendor_second_day_sale_l262_262585

theorem vendor_second_day_sale (n : ℕ) :
  let sold_first_day := (50 * n) / 100
  let remaining_after_first_sale := n - sold_first_day
  let thrown_away_first_day := (20 * remaining_after_first_sale) / 100
  let remaining_after_first_day := remaining_after_first_sale - thrown_away_first_day
  let total_thrown_away := (30 * n) / 100
  let thrown_away_second_day := total_thrown_away - thrown_away_first_day
  let sold_second_day := remaining_after_first_day - thrown_away_second_day
  let percent_sold_second_day := (sold_second_day * 100) / remaining_after_first_day
  percent_sold_second_day = 50 :=
sorry

end vendor_second_day_sale_l262_262585


namespace collinear_points_l262_262269

axiom collinear (A B C : ℝ × ℝ × ℝ) : Prop

theorem collinear_points (c d : ℝ) (h : collinear (2, c, d) (c, 3, d) (c, d, 4)) : c + d = 6 :=
sorry

end collinear_points_l262_262269


namespace magnitude_correct_l262_262107

open Real

noncomputable def magnitude_of_vector_addition
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) : ℝ :=
  ‖3 • a + b‖

theorem magnitude_correct 
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) :
  magnitude_of_vector_addition a b theta ha hb h_angle = sqrt 34 :=
sorry

end magnitude_correct_l262_262107


namespace art_club_proof_art_club_with_values_l262_262561

variable (a b : ℕ)

-- Number of students in the recitation club
def recitation_club : ℕ := (1/2 : ℝ) * a + b

-- Number of students in the dance club
def dance_club : ℕ := a + 2 * b - 40

-- Number of students in the art club
def art_club : ℕ := 260 - (5/2 : ℝ) * a - 3 * b

theorem art_club_proof (a b : ℕ) (h1 : 220 = a + recitation_club a b + dance_club a b + art_club a b) :
  art_club a b = 260 - (5/2 : ℝ) * a - 3 * b := sorry

theorem art_club_with_values : art_club 60 25 = 35 := sorry

end art_club_proof_art_club_with_values_l262_262561


namespace wilted_flowers_are_18_l262_262066

def picked_flowers := 53
def flowers_per_bouquet := 7
def bouquets_after_wilted := 5

def flowers_left := bouquets_after_wilted * flowers_per_bouquet
def flowers_wilted : ℕ := picked_flowers - flowers_left

theorem wilted_flowers_are_18 : flowers_wilted = 18 := by
  sorry

end wilted_flowers_are_18_l262_262066


namespace sign_pyramid_minus_top_l262_262372

def sign_pyramid_combinations : ℕ :=
  sorry -- Placeholder for the definition where the proof will confirm it equals 16

theorem sign_pyramid_minus_top (a b c d e : ℤ) (h : a ∈ {1, -1} ∧ b ∈ {1, -1} ∧ c ∈ {1, -1} ∧ d ∈ {1, -1} ∧ e ∈ {1, -1})
  (prod_eq_neg_one : a * b * c * d * e = -1) 
  : sign_pyramid_combinations = 16 := 
sorry

end sign_pyramid_minus_top_l262_262372


namespace remainder_7n_mod_4_l262_262906

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end remainder_7n_mod_4_l262_262906


namespace pencils_in_stock_at_end_of_week_l262_262068

def pencils_per_day : ℕ := 100
def days_per_week : ℕ := 5
def initial_pencils : ℕ := 80
def sold_pencils : ℕ := 350

theorem pencils_in_stock_at_end_of_week :
  (pencils_per_day * days_per_week + initial_pencils - sold_pencils) = 230 :=
by sorry  -- Proof will be filled in later

end pencils_in_stock_at_end_of_week_l262_262068


namespace problem_l262_262637

def f : ℝ → ℝ := sorry

def domain := {x : ℝ | 0 < x}

axiom cond1 (x : ℝ) (h : x ∈ domain) : f (2 * x) = 2 * f x
axiom cond2 (x : ℝ) (h : 1 < x ∧ x ≤ 2) : f x = 2 - x

theorem problem :
  (∀ m : ℤ, f (2^m) = 0) ∧
  (set.range f = set.Ici 0) ∧
  (¬(∃ n : ℤ, f (2^n + 1) = 9)) ∧
  (∀ a b : ℝ, (∀ x ∈ set.Ioo a b, f' x < 0) ↔ (∃ k : ℤ, a > 2^k ∧ b ≤ 2^(k+1))) :=
by
  sorry

end problem_l262_262637


namespace count_different_numerators_in_T_l262_262034

theorem count_different_numerators_in_T : 
  let T := { r : ℚ | ∃ (a b c d : ℕ), r = (1000 * a + 100 * b + 10 * c + d) / 9999 ∧ 0 < r ∧ r < 1 ∧ r.denom = 9999 ∧ Nat.gcd (1000 * a + 100 * b + 10 * c + d) 9999 = 1 } in
  T.count ≈ 5800 :=
by
  sorry

end count_different_numerators_in_T_l262_262034


namespace find_congruence_l262_262358

theorem find_congruence (x : ℤ) (h : 4 * x + 9 ≡ 3 [ZMOD 17]) : 3 * x + 12 ≡ 16 [ZMOD 17] :=
sorry

end find_congruence_l262_262358


namespace num_consecutive_odd_integers_l262_262150

theorem num_consecutive_odd_integers (n : ℕ) (h1 : (sum (range n).map (fun k => 399 + 2 * k) / n = 414)) : n = 16 := 
sorry

end num_consecutive_odd_integers_l262_262150


namespace ninety_eight_squared_l262_262245

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l262_262245


namespace inequality_of_sum_l262_262633

variable {n : ℕ} (hn : n ≥ 2)
variable {x : ℕ → ℝ}

noncomputable def sum_abs_eq_one : Prop := (∑ i in Finset.range n, abs (x i)) = 1
noncomputable def sum_eq_zero : Prop := (∑ i in Finset.range n, x i) = 0

theorem inequality_of_sum (h1 : sum_abs_eq_one x) (h2 : sum_eq_zero x) :
  abs (∑ i in Finset.range n, x i / (i + 1)) ≤ 1 / 2 - 1 / 2 ^ n :=
sorry

end inequality_of_sum_l262_262633


namespace find_k_l262_262601

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k * x + 1

theorem find_k (k : ℝ) : f 10 - g 10 k = 20 → k = -21.4 :=
by {
  intro h,
  -- The proof is omitted because the only requirement is the statement.
  sorry
}

end find_k_l262_262601


namespace minimum_moves_for_7_rings_l262_262813

def a : ℕ → ℕ 
| 1     := 1
| 2     := 2
| (n+3) := a (n+2) + 2 * a (n+1) + 1

theorem minimum_moves_for_7_rings : a 7 = 85 :=
by sorry

end minimum_moves_for_7_rings_l262_262813


namespace question1_question2_question3_l262_262435

/-
Given conditions:
1. f(x) = ln(1 + x)
2. g(x) = x * f'(x)
3. x >= 0
4. f'(x) is the derivative of f(x)
5. g_1(x) = g(x)
6. g_(n + 1)(x) = g(g_n(x))
7. n ∈ ℕ⁺
-/

/- Proof for Question 1: Prove g_n(x) = x / (1 + n * x) -/
theorem question1 (x : ℝ) (n : ℕ) (hxn : x ≥ 0) (hnn_pos : n > 0):
  let f := fun (x : ℝ) => Real.log (1 + x),
      g := fun (x : ℝ) => x * (f x).derivative in
  let rec g_n : ℕ → (ℝ → ℝ)
    | 0     := g
    | (n+1) := fun x => g (g_n n x) in
  g_n n x = x / (1 + n * x) := sorry

/- Proof for Question 2: Prove the range of a such that f(x) ≥ a * g(x) is a <= 1 -/
theorem question2 (a : ℝ) (x : ℝ) (hxn : x ≥ 0) :
  let f := fun (x : ℝ) => Real.log (1 + x),
      g := fun (x : ℝ) => x * (f x).derivative in
  (∀ x, f x ≥ a * g x) ↔ a ≤ 1 := sorry

/- Proof for Question 3: Prove g(1) + g(2) + ... + g(n) > n - ln (n + 1) -/
theorem question3 (n : ℕ) (hnn_pos : n > 0):
  let f := fun (x : ℝ) => Real.log (1 + x),
      g := fun (x : ℝ) => x * (f x).derivative in
  (∑ i in Finset.range (n + 1).filter (fun i => i > 0), g i) > n - Real.log (n + 1) := sorry

end question1_question2_question3_l262_262435


namespace fraction_of_speedsters_l262_262190

/-- Let S denote the total number of Speedsters and T denote the total inventory. 
    Given the following conditions:
    1. 54 Speedster convertibles constitute 3/5 of all Speedsters (S).
    2. There are 30 vehicles that are not Speedsters.

    Prove that the fraction of the current inventory that is Speedsters is 3/4.
-/
theorem fraction_of_speedsters (S T : ℕ)
  (h1 : 3 / 5 * S = 54)
  (h2 : T = S + 30) :
  (S : ℚ) / T = 3 / 4 :=
by
  sorry

end fraction_of_speedsters_l262_262190


namespace john_profit_l262_262025

/-- Define constants and conditions for the problem -/
def num_newspapers : ℕ := 500
def num_magazines : ℕ := 300
def num_books : ℕ := 200

def price_newspaper : ℕ := 2
def price_magazine : ℕ := 4
def price_book : ℕ := 10

def sell_ratio_newspaper : ℝ := 0.80
def sell_ratio_magazine : ℝ := 0.75
def sell_ratio_book : ℝ := 0.60

def discount_newspaper : ℝ := 0.75
def discount_magazine : ℝ := 0.60
def discount_book : ℝ := 0.45

def tax_rate : ℝ := 0.08
def shipping_fee : ℝ := 25
def commission_rate : ℝ := 0.05

noncomputable def profit : ℝ :=
  let cost_newspapers := (price_newspaper * (1 - discount_newspaper)) * num_newspapers
  let cost_magazines := (price_magazine * (1 - discount_magazine)) * num_magazines
  let cost_books := (price_book * (1 - discount_book)) * num_books
  let total_cost := cost_newspapers + cost_magazines + cost_books
  let total_cost_with_tax_shipping := total_cost * (1 + tax_rate) + shipping_fee

  let revenue_newspapers := (price_newspaper * num_newspapers * sell_ratio_newspaper.to_nat)
  let revenue_magazines := (price_magazine * num_magazines * sell_ratio_magazine.to_nat)
  let revenue_books := (price_book * num_books * sell_ratio_book.to_nat)
  let total_revenue := revenue_newspapers + revenue_magazines + revenue_books
  let total_revenue_after_commission := total_revenue * (1 - commission_rate)

  total_revenue_after_commission - total_cost_with_tax_shipping

theorem john_profit : profit = 753.60 :=
by sorry

end john_profit_l262_262025


namespace two_thirds_of_5_times_9_l262_262624

theorem two_thirds_of_5_times_9 : (2 / 3) * (5 * 9) = 30 :=
by
  sorry

end two_thirds_of_5_times_9_l262_262624


namespace line_through_point_equal_intercepts_l262_262823

theorem line_through_point_equal_intercepts (x y a b : ℝ) :
  ∀ (x y : ℝ), 
    (x - 1) = a → 
    (y - 2) = b →
    (a = -1 ∨ a = 2) → 
    ((x + y - 3 = 0) ∨ (2 * x - y = 0)) := by
  sorry

end line_through_point_equal_intercepts_l262_262823


namespace remainder_2001_to_2005_mod_19_l262_262530

theorem remainder_2001_to_2005_mod_19 :
  (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 :=
by
  -- Use modular arithmetic properties to convert each factor
  have h2001 : 2001 % 19 = 6 := by sorry
  have h2002 : 2002 % 19 = 7 := by sorry
  have h2003 : 2003 % 19 = 8 := by sorry
  have h2004 : 2004 % 19 = 9 := by sorry
  have h2005 : 2005 % 19 = 10 := by sorry

  -- Compute the product modulo 19
  have h_prod : (6 * 7 * 8 * 9 * 10) % 19 = 11 := by sorry

  -- Combining these results
  have h_final : ((2001 * 2002 * 2003 * 2004 * 2005) % 19) = (6 * 7 * 8 * 9 * 10) % 19 := by sorry
  exact Eq.trans h_final h_prod

end remainder_2001_to_2005_mod_19_l262_262530


namespace ninety_eight_squared_l262_262248

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l262_262248


namespace find_FG_l262_262814

noncomputable def triangle_ABC := 
  ∃ (A B C : Type) [triangle A B C],       -- there exists a triangle ABC
  ∃ (AD BE : Type) [altitudes AD BE],      -- with altitudes AD and BE
  ∃ (H : Type) [intersection AD BE H],     -- that intersect at orthocenter H
  ∃ (circumcircle : Type) [circleABH circumcircle], -- and a circumcircle of triangle ABH
  ∃ (F G : Type) [intersect circumcircle AC F] [intersect circumcircle BC G], -- that intersects AC and BC at points F and G
  ∃ (DE : ℝ) (DE_val : DE = 5),            -- given DE = 5 cm
  FG = 10                                  -- prove FG = 10 cm

theorem find_FG :
  triangle_ABC → FG = 10 :=
by
  intro conditions,
  sorry


end find_FG_l262_262814


namespace jenna_profit_l262_262390

noncomputable def total_profit (cost_price sell_price rent : ℝ) (tax_rate : ℝ) (worker_count : ℕ) (worker_salary : ℝ) (widgets_sold : ℕ) : ℝ :=
  let salaries := worker_count * worker_salary in
  let fixed_costs := salaries + rent in
  let profit_per_widget := sell_price - cost_price in
  let total_sales_profit := widgets_sold * profit_per_widget in
  let profit_before_taxes := total_sales_profit - fixed_costs in
  let taxes := profit_before_taxes * tax_rate in
  profit_before_taxes - taxes

theorem jenna_profit :
  total_profit 3 8 10000 0.2 4 2500 5000 = 4000 :=
by
  sorry

end jenna_profit_l262_262390


namespace pi_bounds_exists_constants_l262_262452

theorem pi_bounds_exists_constants :
  ∃ (A B : ℝ), (A = 0.1) ∧ (B = 4) ∧ ∀ (N : ℕ), A * N / Real.log N < Nat.primePi N ∧ Nat.primePi N < B * N / Real.log N :=
by
  sorry

end pi_bounds_exists_constants_l262_262452


namespace value_of_g_at_2_l262_262035

def g (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

theorem value_of_g_at_2 : g 2 = 11 := 
by
  sorry

end value_of_g_at_2_l262_262035


namespace diameter_of_circle_l262_262325

theorem diameter_of_circle 
  (m : ℝ)
  (O : Set Point)
  (H_circle : ∀ (p : Point), p ∈ O ↔ (p.x^2 + p.y^2 - 2 * p.x + m * p.y - 4 = 0))
  (H_symmetry : ∀ (M N : Point), M ∈ O → N ∈ O → (M + N) = (1, -m / 2)) :
  diameter_of_circle = 6 :=
by
  sorry

end diameter_of_circle_l262_262325


namespace income_increase_percentage_l262_262794

theorem income_increase_percentage (I : ℝ) (P : ℝ) (h1 : 0 < I)
  (h2 : 0 ≤ P) (h3 : 0.75 * I + 0.075 * I = 0.825 * I) 
  (h4 : 1.5 * (0.25 * I) = ((I * (1 + P / 100)) - 0.825 * I)) 
  : P = 20 := by
sorry

end income_increase_percentage_l262_262794


namespace max_a_value_l262_262962

def lattice_point (x y : ℤ) : Prop := true -- since both x and y being integers defines a lattice point

def line (m : ℚ) (x : ℤ) : ℚ := m * (x : ℚ) + 3

def not_lattice_point (m : ℚ) : Prop :=
  ∀ x : ℤ, (0 < x ∧ x ≤ 150) → ¬(∃ y : ℤ, line m x = y)

def valid_m (m : ℚ) : Prop := 1 / 2 < m

theorem max_a_value (a : ℚ) : (∀ m : ℚ, valid_m m → m < a → not_lattice_point m) → a = 75 / 149 :=
begin
  sorry
end

end max_a_value_l262_262962


namespace find_z_coordinate_l262_262197

theorem find_z_coordinate (x : ℚ) (z : ℚ) (t : ℚ) :
  (3 + 5 * t = 7) ∧ (z = 2 - 5 * t) → z = -2 := 
by
  assume h : (3 + 5 * t = 7) ∧ (z = 2 - 5 * t)
  let t_value : ℚ := 4 / 5
  have ht : t = t_value := by
    sorry
  rw ht at h
  sorry
  have hz : z = -2 := by
    sorry
  exact hz

end find_z_coordinate_l262_262197


namespace triangle_middle_segments_ratio_l262_262501

theorem triangle_middle_segments_ratio 
    (a b c a' b' c' : ℝ) 
    (h_a : a' = a / 3) 
    (h_b : b' = b / 3) 
    (h_c : c' = c / 3) :
    (a' / a) + (b' / b) + (c' / c) = 1 :=
by 
  -- From the conditions
  have h1 : a' / a = 1 / 3, from by rwa [h_a] at *,
  have h2 : b' / b = 1 / 3, from by rwa [h_b] at *,
  have h3 : c' / c = 1 / 3, from by rwa [h_c] at *,
  -- Summing up the ratios
  exact calc 
    (a' / a) + (b' / b) + (c' / c)
        = (1 / 3) + (1 / 3) + (1 / 3) : by rw [h1, h2, h3]
    ... = 1 : by norm_num

end triangle_middle_segments_ratio_l262_262501


namespace common_ratio_of_geometric_series_l262_262864

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l262_262864


namespace tiling_theorem_l262_262406

noncomputable def minimal_square_tiling (a b : ℕ) (a_decomp b_decomp : list ℕ)
  (h1 : a = list.sum (a_decomp.map (λ x, 2^x)))
  (h2 : b = list.sum (b_decomp.map (λ x, 2^x)))
  (h3 : a_decomp.sorted (>) ∧ b_decomp.sorted (>)) : ℕ :=
list.sum (a_decomp.bind (λ ai, b_decomp.map (λ bj, 2^(int.natAbs (ai - bj)))))

theorem tiling_theorem 
  (a b : ℕ) (a_decomp b_decomp : list ℕ)
  (h1 : a = list.sum (a_decomp.map (λ x, 2^x)))
  (h2 : b = list.sum (b_decomp.map (λ x, 2^x)))
  (h3 : a_decomp.sorted (>) ∧ b_decomp.sorted (>)) :
  minimal_square_tiling a b a_decomp b_decomp h1 h2 h3 = 
    (list.sum (a_decomp.bind (λ ai, b_decomp.map (λ bj, 2^(int.natAbs (ai - bj)))))) :=
by sorry

end tiling_theorem_l262_262406


namespace seating_arrangements_l262_262057

theorem seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 7) :
  let choose_stand := Nat.choose n (n - k)
  let factorial := ∀ (m : ℕ), Nat.factorial m
  choose_stand * (factorial k / k) = 25920 :=
by
  sorry

end seating_arrangements_l262_262057


namespace cos_double_angle_l262_262328

theorem cos_double_angle (x : ℝ) (h : cos x = 3 / 4) : cos (2 * x) = 1 / 8 :=
by 
  sorry

end cos_double_angle_l262_262328


namespace remainder_7n_mod_4_l262_262910

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l262_262910


namespace find_x_parallel_vectors_l262_262686

theorem find_x_parallel_vectors : 
  ∀ x : ℝ, (∀ a b : ℝ × ℝ, a = (2, 3) → b = (x, -6) → (∃ k : ℝ, a = k • b)) → x = -4 :=
by
  intros x h a b ha hb
  specialize h a b ha hb
  sorry

end find_x_parallel_vectors_l262_262686


namespace value_of_2p_plus_q_l262_262362

theorem value_of_2p_plus_q (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p :=
by
  sorry

end value_of_2p_plus_q_l262_262362


namespace ordering_of_abc_l262_262418

noncomputable def a : ℝ := 2 * Real.log (3 / 2)
noncomputable def b : ℝ := Real.log (1 / 3) / Real.log 2
noncomputable def c : ℝ := (1 / 2) ^ (-0.3)

theorem ordering_of_abc : b < a ∧ a < c := by
  -- Proof steps would go here
  sorry

end ordering_of_abc_l262_262418


namespace donut_distribution_l262_262981

theorem donut_distribution :
  ∃ (Alpha Beta Gamma Delta Epsilon : ℕ), 
    Delta = 8 ∧ 
    Beta = 3 * Gamma ∧ 
    Alpha = 2 * Delta ∧ 
    Epsilon = Gamma - 4 ∧ 
    Alpha + Beta + Gamma + Delta + Epsilon = 60 ∧ 
    Alpha = 16 ∧ 
    Beta = 24 ∧ 
    Gamma = 8 ∧ 
    Delta = 8 ∧ 
    Epsilon = 4 :=
by
  sorry

end donut_distribution_l262_262981


namespace multiplier_for_a_to_equal_30_l262_262363

theorem multiplier_for_a_to_equal_30 (a b : ℝ) (h1 : a = 5) (h2 : b = 30) (h3 : 40 * a * b = 1800) :
  30 / a = 6 :=
by
  rw h1
  norm_num
  sorry

end multiplier_for_a_to_equal_30_l262_262363


namespace polynomial_expansion_a6_l262_262765

theorem polynomial_expansion_a6 :
  let p := x^2 + x^7
  ∃ (a : ℕ → ℝ), p = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 ∧ a 6 = -7 := 
sorry

end polynomial_expansion_a6_l262_262765


namespace volume_of_tetrahedron_SABC_l262_262382

noncomputable def calculateVolumeOfTetrahedron (S A B C A' B' C' D O : ℝ) :=
  let AB : ℝ := 4
  let side_length := 4
  let volume := (side_length^3) / (6 * Real.sqrt 2)
  volume

theorem volume_of_tetrahedron_SABC 
  (S A B C A' B' C' D O : ℝ) 
  (h1 : ¬ (S = A))
  (h2 : ¬ (S = B))
  (h3 : ¬ (S = C))
  (h4 : A ≠ B) 
  (h5 : B ≠ C) 
  (h6 : C ≠ A)
  (h7 : Real.norm (B - A) = 4)
  (h8 : isEquilateralTriangle A B C)
  (h9 : midpoint S A = A')
  (h10 : midpoint S B = B')
  (h11 : midpoint S C = C')
  (h12 : midpoint A' B' = D)
  (h13 : Circumcenter {S,A,B,C} = O)
  (h14 : (Real.norm (C - O))^2 - (Real.norm (D - O))^2 = (Real.norm (B - A))^2) : 
  calculateVolumeOfTetrahedron S A B C A' B' C' D O = 32 * Real.sqrt 2 / 3 := 
sorry

end volume_of_tetrahedron_SABC_l262_262382


namespace zealand_fraction_of_total_l262_262512

theorem zealand_fraction_of_total
  (w x y z : ℝ)  -- w is Wanda's original money, x is Xander's original money, y is Yusuf's original money, z is Zealand's original money
  (x_donation : ℝ)  -- x_donation represents the amount each person gave
  (hw : w = 6 * x_donation)
  (hx : x = 5 * x_donation)
  (hy : y = 4 * x_donation)
  (hz : z = 0)
  (zw : z + x_donation = 3 * x_donation)  -- Zealand's new amount
  : let total = w + x + y in
    (3 * x_donation) / total = 1 / 5 :=
by
  sorry

end zealand_fraction_of_total_l262_262512


namespace no_common_terms_except_one_l262_262177

noncomputable def x_seq : ℕ → ℤ
| 0       := 1
| 1       := 1
| (n + 2) := x_seq n + 2 * x_seq (n + 1)

noncomputable def y_seq : ℕ → ℤ
| 0       := 1
| 1       := 7
| (n + 2) := 2 * y_seq (n + 1) + 3 * y_seq n

theorem no_common_terms_except_one :
  ∀ m n : ℕ, m > 1 → n > 1 → x_seq m ≠ y_seq n := 
sorry

end no_common_terms_except_one_l262_262177


namespace solve_equation_l262_262171

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_equation:
    (7.331 * ((log_base 3 x - 1) / (log_base 3 (x / 3))) - 
    2 * (log_base 3 (Real.sqrt x)) + (log_base 3 x)^2 = 3) → 
    (x = 1 / 3 ∨ x = 9) := by
  sorry

end solve_equation_l262_262171


namespace red_grapes_count_l262_262714

-- Definitions of variables and conditions
variables (G R Ra B P : ℕ)
variables (cond1 : R = 3 * G + 7)
variables (cond2 : Ra = G - 5)
variables (cond3 : B = 4 * Ra)
variables (cond4 : P = (1 / 2) * B + 5)
variables (cond5 : G + R + Ra + B + P = 350)

-- Theorem statement
theorem red_grapes_count : R = 100 :=
by sorry

end red_grapes_count_l262_262714


namespace ellipse_equation_correct_l262_262110

noncomputable def ellipse_shared_foci : Prop :=
  let E1 := 3 * (x^2) + 8 * (y^2) = 24 in
  let foci := (λ (c: ℝ), c = sqrt 5) in  -- Extracting foci from first ellipse
  let point := (3, 2) in
  let E2 := (x^2)/15 + (y^2)/10 = 1 in 
  (foci (sqrt 5)) ∧ E2.point = point

theorem ellipse_equation_correct : ellipse_shared_foci :=
sorry

end ellipse_equation_correct_l262_262110


namespace no_silver_matrix_1997_l262_262976

def is_silver_matrix {n : ℕ} (A : matrix (fin n) (fin n) ℕ) :=
  (∀ i : fin n, (finset.univ.image (λ j, A i j)) ∪ (finset.univ.image (λ j, A j i)) = finset.range (2 * n - 1))

theorem no_silver_matrix_1997 :
  ¬ ∃ (A : matrix (fin 1997) (fin 1997) ℕ), is_silver_matrix A :=
sorry

end no_silver_matrix_1997_l262_262976


namespace letters_in_mailboxes_l262_262940

theorem letters_in_mailboxes :
  ∃ n : ℕ, n = 3^4 ∧ n = 81 :=
by {
  use 81,
  split,
  {
    sorry, -- here, we would prove that 3^4 = 81
  },
  {
    refl, -- 81 is indeed 81
  }
}

end letters_in_mailboxes_l262_262940


namespace correct_function_l262_262221

theorem correct_function (f : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) = f x * f y)
  (h2 : ∀ x, 0 < f x) : 
  (f = λ x, 3^x) ∨ (f = λ x, 0) :=
sorry

end correct_function_l262_262221


namespace prove_cannot_form_2x2_square_l262_262625

noncomputable def cannot_form_2x2_square : Prop :=
  let pieces := [ (1, 1), (1, 1), (2, 1), (1, 2), (3, 1)]
  ∀ (formation : List (ℕ × ℕ)), -- all possible formations of the pieces
    (formation = [(2, 2)]) → 
    (list.sum (formation.map (λ p => p.1 * p.2))) ≠ 4 -- ensure total area is not 4

theorem prove_cannot_form_2x2_square : cannot_form_2x2_square :=
sorry

end prove_cannot_form_2x2_square_l262_262625


namespace range_of_dot_product_l262_262670

noncomputable def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def vec_dot (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 * x2) + (y1 * y2)

theorem range_of_dot_product :
  ∀ (P: ℝ × ℝ)
  (x := P.1)
  (y := P.2),
  ellipse_eq x y 2 (real.sqrt 3) →
  0 ≤ vec_dot (x + 2) y (x + 1) y ∧
  vec_dot (x + 2) y (x + 1) y ≤ 12 :=
sorry

end range_of_dot_product_l262_262670


namespace q1_q2a_q2b_q3_l262_262154

namespace MathProblem

noncomputable def f : ℝ → ℝ 
| x => if x > 0 && x <= 10 then -0.1 * x^2 + 2.6 * x + 43
       else if x > 10 && x <= 16 then 59
       else if x > 16 && x <= 30 then -3 * x + 107
       else 0

theorem q1 : f 5 > f 20 :=
by 
  sorry

theorem q2a : ∃ x, 0 < x ∧ x ≤ 10 ∧ f x = 59 :=
by 
  sorry

theorem q2b : ∀ x, (x > 10 ∧ x ≤ 16) → f x = 59 :=
by 
  sorry

theorem q3 : ¬ ∃ I : set ℝ, (∀ x ∈ I, f x ≥ 55) ∧ (measure_theory.volume I = 13) :=
by 
  sorry

end MathProblem

end q1_q2a_q2b_q3_l262_262154


namespace nickels_left_l262_262090

theorem nickels_left (n b : ℕ) (h₁ : n = 31) (h₂ : b = 20) : n - b = 11 :=
by
  sorry

end nickels_left_l262_262090


namespace point_quadrant_l262_262796

theorem point_quadrant :
  let P := (Real.cos (2017 * Real.pi / 180), Real.sin (2017 * Real.pi / 180))
  in P.1 < 0 ∧ P.2 < 0 := sorry

end point_quadrant_l262_262796


namespace unique_intersection_point_l262_262251

noncomputable def point_intersection : Prop := 
  let A := (0, 0) in
  let B := (0, 4) in
  let C := (6, 4) in
  let D := (6, 0) in
  let lines_from_A := [fun (x: ℝ) => x, fun (x: ℝ) => -x] in
  let lines_from_B := [fun (x: ℝ) => 4 - x, fun (x: ℝ) => 4 + x] in
  let intersections := 
    [ (2, 2), 
      (2, 2) ] in
  ∀ (P: (ℝ × ℝ)), P ∈ intersections → P = (2, 2)

theorem unique_intersection_point : point_intersection := 
  sorry

end unique_intersection_point_l262_262251


namespace prob_all_same_room_correct_prob_at_least_two_same_room_correct_l262_262879

-- Given conditions: three people and four rooms, each equally likely for each person.
def people := {1, 2, 3}
def rooms := {1, 2, 3, 4}
def assignments := {f : people → rooms // ∀ p ∈ people, f p ∈ rooms}

-- Define the probability space
noncomputable def prob_space : ProbabilityMassFunction assignments := sorry

-- Probability that all three people are assigned to the same room.
noncomputable def prob_all_same_room : ℚ :=
  let event := {f : assignments // ∃ r ∈ rooms, ∀ p ∈ people, f.val p = r}
  ProbabilityMassFunction.probability prob_space event

-- Probability that at least two people are assigned to the same room.
noncomputable def prob_at_least_two_same_room : ℚ :=
  1 - ProbabilityMassFunction.probability prob_space
    {f : assignments // ∀ p1 p2 ∈ people, p1 ≠ p2 → f.val p1 ≠ f.val p2}

-- Theorems
theorem prob_all_same_room_correct : prob_all_same_room = 1 / 16 := sorry

theorem prob_at_least_two_same_room_correct : prob_at_least_two_same_room = 5 / 8 := sorry

end prob_all_same_room_correct_prob_at_least_two_same_room_correct_l262_262879


namespace corresponding_pairs_4_l262_262031

noncomputable def num_corresponding_pairs : ℕ :=
  let A (a : ℝ) := {x : ℝ | |x - a| = 1}
  let B (b : ℝ) := ({1, -3, b} : Set ℝ)
  let valid_pairs := {p : ℝ × ℝ | ∃ a b, p = (a, b) ∧ A a ⊆ B b}
  Finset.card valid_pairs.to_finset

theorem corresponding_pairs_4 : num_corresponding_pairs = 4 := sorry

end corresponding_pairs_4_l262_262031


namespace computer_on_time_l262_262751

def days_in_hours : nat := 24

def total_hours : nat := 100

def end_time : nat := 5 + days_in_hours * 5 -- 5 p.m. Friday

def start_time : nat := end_time - total_hours

theorem computer_on_time : start_time = 1 + days_in_hours * 1 -- 1 p.m. Monday
:= by
    sorry

end computer_on_time_l262_262751


namespace angle_QRP_l262_262242

theorem angle_QRP
  (Ω : Circle)
  (D E F P Q R : Point)
  (h_Ω_inc : Ω.inc_of_triangle DEF)
  (h_Ω_circ : Ω.circ_of_triangle PQR)
  (h_P_on_EF : P ∈ line_segment EF)
  (h_Q_on_DE : Q ∈ line_segment DE)
  (h_R_on_DF : R ∈ line_segment DF)
  (angle_D : ∠DEF = 50)
  (angle_E : ∠EDF = 70)
  (angle_F : ∠EFD = 60) :
  ∠QRP = 40 :=
by
  sorry

end angle_QRP_l262_262242


namespace jenna_total_profit_l262_262396

-- Definitions from conditions
def widget_cost := 3
def widget_price := 8
def rent := 10000
def tax_rate := 0.2
def salary_per_worker := 2500
def number_of_workers := 4
def widgets_sold := 5000

-- Calculate intermediate values
def total_revenue := widget_price * widgets_sold
def total_cost_of_widgets := widget_cost * widgets_sold
def gross_profit := total_revenue - total_cost_of_widgets
def total_expenses := rent + salary_per_worker * number_of_workers
def net_profit_before_taxes := gross_profit - total_expenses
def taxes := tax_rate * net_profit_before_taxes
def total_profit_after_taxes := net_profit_before_taxes - taxes

-- Theorem to be proven
theorem jenna_total_profit : total_profit_after_taxes = 4000 := by sorry

end jenna_total_profit_l262_262396


namespace problem_part1_problem_part2_problem_part3_l262_262454

variable (a b x : ℝ) (p q : ℝ) (n x1 x2 : ℝ)
variable (h1 : x1 = -2) (h2 : x2 = 3)
variable (h3 : x1 < x2)

def equation1 := x + p / x = q
def solution1_p := p = -6
def solution1_q := q = 1

def equation2 := x + 7 / x = 8
def solution2 := x1 = 7

def equation3 := 2 * x + (n^2 - n) / (2 * x - 1) = 2 * n
def solution3 := (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)

theorem problem_part1 : ∀ (x : ℝ), (x + -6 / x = 1) → (p = -6 ∧ q = 1) := by
  sorry

theorem problem_part2 : (max 7 1 = 7) := by
  sorry

theorem problem_part3 : ∀ (n : ℝ), (∃ x1 x2, x1 < x2 ∧ (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)) := by
  sorry

end problem_part1_problem_part2_problem_part3_l262_262454


namespace maria_total_eggs_l262_262778

def total_eggs (boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  boxes * eggs_per_box

theorem maria_total_eggs :
  total_eggs 3 7 = 21 :=
by
  -- Here, you would normally show the steps of computation
  -- which we can skip with sorry
  sorry

end maria_total_eggs_l262_262778


namespace cone_surface_area_l262_262466

theorem cone_surface_area (d α : ℝ) (h₁ : 0 < d) (h₀ : 0 < α ∧ α < π / 2) :
  let R := d / cos α in
  let l := d / (sin α * cos α) in
  let S := π * R * l + π * R^2 in
  S = π * d^2 / (2 * sin α * sin^2 (π / 4 - α / 2)) :=
by
  sorry

end cone_surface_area_l262_262466


namespace base_layer_spheres_base_layer_count_l262_262942

-- Define the k-th triangular number function
def triangular_number (k : ℕ) : ℕ :=
  k * (k + 1) / 2

-- Prove that the sum of the first n triangular numbers for n = 8
-- equals 120
theorem base_layer_spheres (n : ℕ) (h : n = 8) :
  (∑ k in Finset.range (n + 1), triangular_number k) = 120 :=
  by
    sorry

-- Prove that the number of spheres in the base layer (8th triangular number) is 36
theorem base_layer_count : triangular_number 8 = 36 :=
  by
    rw [triangular_number]
    norm_num

end base_layer_spheres_base_layer_count_l262_262942


namespace hyperbola_eccentricity_l262_262341

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 / Real.exp x

theorem hyperbola_eccentricity :
  let a := SorryVariable,
      b := 2 * a,
      c := Real.sqrt (a ^ 2 + b ^ 2)
  in c = Real.sqrt 5 :=
by {
  -- Assume that the necessary expressions for a, b, and c are derived correctly.
  sorry
}

end hyperbola_eccentricity_l262_262341


namespace binom_n_2_l262_262517

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262517


namespace bob_regular_hourly_wage_l262_262995

theorem bob_regular_hourly_wage :
  ∃ R : ℝ, 
    (R > 0) ∧ 
    (let total_regular_hours := 80 in
     let total_overtime_pay := 72 in
     let total_pay := 472 in
     total_regular_hours * R + total_overtime_pay = total_pay ∧ R = 5) :=
begin
  use 5,
  split,
  { linarith, },
  { split,
    { linarith, },
    { refl, }, },
end

end bob_regular_hourly_wage_l262_262995


namespace incorrect_statement_about_f_l262_262300

def f (x : ℝ) : ℝ := π ^ (-(x ^ 2) / 2)

theorem incorrect_statement_about_f :
  (∀ x, f (-x) = f x) ∧
  (∀ x < 0, f x < f (x + ε) ∧ f x > f (x - ε)) ∧
  (∀ x > 0, f x > f (x + ε) ∧ f x < f (x - ε)) ∧
  (∀ y, ∃ x, y = f x → y ∈ (0, 1]) ∧
  ((∀ x, f x > π^(-2) ↔ -2 < x ∧ x < 2) →
    ¬((∀ x, f x > π^(-2) ↔ x < -2 ∨ x > 2))) 
:= sorry

end incorrect_statement_about_f_l262_262300


namespace find_ratio_l262_262444

noncomputable def collinear_points (a b c d e f g h : Point) : Prop :=
  collinear ({a, b, c, d, e, f, g, h} : set Point)

noncomputable def equal_segments (a b c d e f g h : Point) : Prop :=
  (dist a b = 2) ∧ (dist b c = 2) ∧ (dist c d = 2) ∧ (dist d e = 2) ∧ (dist e f = 2) ∧ (dist f g = 2) ∧ (dist g h = 2)

noncomputable def parallel_lines (j c k e a i : Point) : Prop :=
  parallel (Line.mk j c) (Line.mk a i) ∧ parallel (Line.mk k e) (Line.mk a i)

noncomputable def point_not_on_line (i : Point) (a h : Point) : Prop :=
  ∀ p ∈ (Line.mk a h).carrier, p ≠ i

noncomputable def point_on_line (p : Point) (a b : Point) : Prop :=
  p ∈ (Line.mk a b).carrier

theorem find_ratio
  (A B C D E F G H I J K : Point)
  (h1 : collinear_points A B C D E F G H)
  (h2 : equal_segments A B C D E F G H)
  (h3 : point_not_on_line I A H)
  (h4 : point_on_line J I D)
  (h5 : point_on_line K I H)
  (h6 : parallel_lines J C K E A I) :
  dist J C / dist K E = 7 / 4 :=
sorry

end find_ratio_l262_262444


namespace digging_project_depth_l262_262191

theorem digging_project_depth : 
  ∀ (P : ℕ) (D : ℝ), 
  (12 * P) * (25 * 30 * D) / 12 = (12 * P) * (75 * 20 * 50) / 12 → 
  D = 100 :=
by
  intros P D h
  sorry

end digging_project_depth_l262_262191


namespace function_elements_l262_262195

theorem function_elements (f : Type) :
  ∃ (correspondence : Type) (domain : Type) (range : Type), 
  (∀ (x : domain), ∃ (y : range), correspondence x y) :=
sorry

end function_elements_l262_262195


namespace g_240_minus_g_120_eq_0_l262_262295

-- Define the sum of even divisors of a number
def sum_of_even_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0).sum

-- Define the function g(n) as the quotient obtained when the sum of all even positive divisors of n is divided by n
noncomputable def g (n : ℕ) : ℚ :=
  if n = 0 then 0 else (sum_of_even_divisors n : ℚ) / n

-- The theorem stating that g(240) - g(120) = 0
theorem g_240_minus_g_120_eq_0 : g 240 - g 120 = 0 :=
  sorry

end g_240_minus_g_120_eq_0_l262_262295


namespace num_maps_from_set_a_to_set_b_l262_262185

theorem num_maps_from_set_a_to_set_b : 
  let A := {D, U, K, E}
  let B := {M, A, T, H}
  ∃ n : ℕ, n = 256 ∧ cardinality (A → B) = n :=
by
  let A := {D, U, K, E}
  let B := {M, A, T, H}
  use 256
  split
  exact rfl
  sorry

end num_maps_from_set_a_to_set_b_l262_262185


namespace irrational_2pi_l262_262918

theorem irrational_2pi : irrational (2 * real.pi) :=
sorry

end irrational_2pi_l262_262918


namespace solve_quadratic_1_solve_quadratic_2_l262_262456

theorem solve_quadratic_1 (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by sorry

theorem solve_quadratic_2 (x : ℝ) : x^2 - 5 * x + 6 = 0 → x = 2 ∨ x = 3 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l262_262456


namespace meeting_time_l262_262920

theorem meeting_time (xiaoDong_time grandfather_time : ℝ) (h1 : xiaoDong_time = 10) (h2 : grandfather_time = 16) : 
  (1 / (1 / xiaoDong_time + 1 / grandfather_time)) = 80 / 13 := 
by
  -- We state the conditions from the problem
  have xiaoDong_inv := (1 / xiaoDong_time),
  have grandfather_inv := (1 / grandfather_time),
  -- Now we use the given conditions
  rw [h1, h2] at *,
  -- The calculation is mostly here to demonstrate what we're proving
  sorry

end meeting_time_l262_262920


namespace editors_min_count_l262_262993

theorem editors_min_count
  (writers : ℕ)
  (P : ℕ)
  (S : ℕ)
  (W : ℕ)
  (H1 : writers = 45)
  (H2 : P = 90)
  (H3 : ∀ x : ℕ, x ≤ 6 → (90 = (writers + W - x) + 2 * x) → W ≥ P - 51)
  : W = 39 := by
  sorry

end editors_min_count_l262_262993


namespace geometric_series_ratio_l262_262859

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l262_262859


namespace amplitude_of_combined_wave_l262_262377

noncomputable def y1 (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) : ℝ := 3 * Real.cos (100 * Real.pi * t + Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y1 t + y2 t

theorem amplitude_of_combined_wave : ∀ t : ℝ, Real.I (enorm (0,abs y t)) 3 :=
by sorry

end amplitude_of_combined_wave_l262_262377


namespace tournament_max_matches_l262_262943

theorem tournament_max_matches (num_participants: ℕ) (max_fights: ℕ) : 
  num_participants = 55 → 
  (∀ (match: ℕ → ℕ → Prop), (
    ∀ x y, match x y → |x - y| ≤ 1
  )) → 
  max_fights = 8 :=
by
  intros h_participants h_match
  sorry

end tournament_max_matches_l262_262943


namespace base_of_number_l262_262367

theorem base_of_number (b : ℕ) : 
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 :=
by
  sorry

end base_of_number_l262_262367


namespace train_passes_jogger_time_l262_262575

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 75
noncomputable def jogger_head_start_m : ℝ := 500
noncomputable def train_length_m : ℝ := 300

noncomputable def km_per_hr_to_m_per_s (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def jogger_speed_m_per_s := km_per_hr_to_m_per_s jogger_speed_km_per_hr
noncomputable def train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr

noncomputable def relative_speed_m_per_s := train_speed_m_per_s - jogger_speed_m_per_s

noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m

theorem train_passes_jogger_time :
  let time_to_pass := total_distance_to_cover_m / relative_speed_m_per_s
  abs (time_to_pass - 43.64) < 0.01 :=
by
  sorry

end train_passes_jogger_time_l262_262575


namespace problem_l262_262071

noncomputable def length_of_AB (A B C D E F G : Point) : ℝ :=
let AB := 2 * (2 * (2 * (2 * (2 * (AG : ℝ)))))
in AB

theorem problem (A B C D E F G : Point) (AG : ℝ) (h1 : C = midpoint A B)
  (h2 : D = midpoint A C) (h3 : E = midpoint A D) (h4 : F = midpoint A E) 
  (h5 : G = midpoint A F) (h6 : AG = 5) : 
  length_of_AB A B C D E F G = 160 := 
sorry

end problem_l262_262071


namespace geometric_series_common_ratio_l262_262843

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l262_262843


namespace lateral_side_of_isosceles_triangle_l262_262564

theorem lateral_side_of_isosceles_triangle (x : ℝ) :
  is_isosceles_triangle ABC → 
  base_length BC = 12 →
  inscribed_circle_with_tangents ABC →
  sum_perimeters_of_smaller_triangles ABC = 48 →
  x = 18 :=
begin
  sorry
end

end lateral_side_of_isosceles_triangle_l262_262564


namespace dozen_Pokemon_cards_per_friend_l262_262808

theorem dozen_Pokemon_cards_per_friend
  (total_cards : ℕ) (num_friends : ℕ) (cards_per_dozen : ℕ)
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : cards_per_dozen = 12) :
  (total_cards / num_friends) / cards_per_dozen = 9 := 
sorry

end dozen_Pokemon_cards_per_friend_l262_262808


namespace remainder_when_dividing_928927_by_6_l262_262897

theorem remainder_when_dividing_928927_by_6 :
  928927 % 6 = 1 :=
by
  sorry

end remainder_when_dividing_928927_by_6_l262_262897


namespace least_possible_perimeter_l262_262122

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end least_possible_perimeter_l262_262122


namespace no_fixed_point_in_temperature_conversion_l262_262825

def convert_to_celsius (F : ℤ) : ℤ := Int.round ((5 * (F - 32) : ℚ) / 9)

def convert_to_fahrenheit (C : ℤ) : ℤ := Int.round ((9 * C : ℚ) / 5 + 33)

theorem no_fixed_point_in_temperature_conversion :
  ∀ (F : ℤ), 34 ≤ F ∧ F ≤ 1024 → F ≠ convert_to_fahrenheit (convert_to_celsius F) :=
by
  intros F hF
  have h₁ : Int.round ((5 * (F - 32) : ℚ) / 9) = convert_to_celsius F,
  by simp [convert_to_celsius]
  have h₂ : Int.round ((9 * (convert_to_celsius F) : ℚ) / 5 + 33) = convert_to_fahrenheit (convert_to_celsius F),
  by simp [convert_to_fahrenheit]
  sorry

end no_fixed_point_in_temperature_conversion_l262_262825


namespace largest_prime_up_to_50_l262_262155

theorem largest_prime_up_to_50 : 
  ∀ (n : ℕ), 2500 ≤ n ∧ n ≤ 2600 → ∀ p, nat.prime p ∧ p ≤ nat.sqrt 2600 → p ≤ 47 :=
by
  intros n hn p hp
  have h : nat.sqrt 2600 ≤ 50 := sorry
  have primes_up_to_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  have h_prime_list : ∀ q, q ∈ primes_up_to_50 → nat.prime q := sorry
  have h_max_prime : ∀ q, q ∈ primes_up_to_50 → q ≤ 47 := sorry
  cases hp with hp_prime hp_le
  cases hn with hn_lower hn_upper
  specialize h_max_prime p (sorry)
  exact h_max_prime

end largest_prime_up_to_50_l262_262155


namespace solution_to_problem_l262_262760

noncomputable def f₁ (x : ℝ) : ℝ :=
  real.sqrt (1 - (x - 1) ^ 2)

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then f₁ x
  else if 2 ≤ n then
    match (f (n - 1)) (real.sqrt ((n : ℝ) ^ 2 - (x - (n : ℝ)) ^ 2)) with
    | y := y
    end
  else 0

theorem solution_to_problem : ∃ c : ℝ, ∃ N : ℕ, N = 5 ∧ (∀ x, f N x = c) :=
by
  sorry

end solution_to_problem_l262_262760


namespace ellipse_m_gt_5_l262_262738

theorem ellipse_m_gt_5 (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m > 5 :=
by
  intros h
  sorry

end ellipse_m_gt_5_l262_262738


namespace set_assignment_possible_iff_l262_262431

noncomputable theory

-- Define assumptions about sets and elements
variables {α : Type*} (n : ℕ) [even n] 
          (A : fin (n + 1) → finset α) 
          [∀ i, (A i).card = n] 
          [∀ i j, i ≠ j → (A i ∩ A j).card = 1] 
          [∀ x, ∃ s t, s ≠ t ∧ x ∈ A s ∧ x ∈ A t]

-- The main theorem statement
theorem set_assignment_possible_iff (k : ℕ) (hk : n = 2 * k) : 
  (∃ f : α → bool, ∀ i, (A i).filter (λ x, f x = ff) .card = k) ↔ even k :=
sorry

end set_assignment_possible_iff_l262_262431


namespace common_sum_tesseract_faces_l262_262133

theorem common_sum_tesseract_faces : 
  let vertices := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  (∀ faces, sum faces = (55 : ℚ) / 36) :=
sorry

end common_sum_tesseract_faces_l262_262133


namespace common_ratio_of_geometric_series_l262_262866

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l262_262866


namespace find_certain_number_l262_262557

theorem find_certain_number (x : ℝ) (h : 0.80 * x = (4 / 5 * 20) + 16) : x = 40 :=
by sorry

end find_certain_number_l262_262557


namespace geometric_series_ratio_l262_262858

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l262_262858


namespace find_parallel_line_l262_262285

variables {x y : ℝ}

def line1 (x y : ℝ) : Prop := 2 * x + y - 3 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def point (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem find_parallel_line (hx : point 0 1) : line2 0 1 :=
by
  dsimp [line2, point] at *,
  sorry

end find_parallel_line_l262_262285


namespace donny_remaining_money_l262_262275

theorem donny_remaining_money :
  let initial_amount := 78
  let kite_cost := 8
  let frisbee_cost := 9
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  sorry

end donny_remaining_money_l262_262275


namespace return_cards_min_cost_l262_262208

theorem return_cards_min_cost (n : ℕ) (a : ℕ → ℕ) (h_perm : ∀ i, 1 ≤ a i ∧ a i ≤ n ∧ function.injective a) :
  ∃ swaps : list (ℕ × ℕ), 
    (∀ (x y : ℕ), (x, y) ∈ swaps → 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n) ∧ 
    (∀ (x y : ℕ), (x, y) ∈ swaps → x ≠ y ∧ list.count (swaps.map prod.swap) (y, x) = 0) ∧
    let total_cost := swaps.sum (λ ⟨x, y⟩, 2 * |x - y|)
    in total_cost ≤ (finset.range n).sum (λ i, |a i - (i + 1)|) ∧ 
       function.bijective (function.swap_compose a (list.prod.snd ⊟ (swaps.map prod.swap))) :=
by exact sorry

end return_cards_min_cost_l262_262208


namespace find_journey_length_l262_262547

def train_journey_length (L : ℝ) : Prop :=
  L = 400 / 3

theorem find_journey_length : 
  ∃ (L : ℝ), ∃ (T : ℝ),
    T = L / 100 ∧
    T + 1 / 3 = L / 80 ∧
    L = 400 / 3 ∧
    Float.round (L * 100) / 100 = 133.33 
:= by
  sorry

end find_journey_length_l262_262547


namespace cosine_of_angle_between_diagonals_l262_262599

noncomputable def a : ℝ × ℝ × ℝ := (3, 2, -2)
noncomputable def b : ℝ × ℝ × ℝ := (2, 3, 3)

theorem cosine_of_angle_between_diagonals :
  let d1 := (a.1 + b.1, a.2 + b.2, a.3 + b.3)
  let d2 := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let dot_product := d1.1 * d2.1 + d1.2 * d2.2 + d1.3 * d2.3
  let norm1 := Real.sqrt (d1.1 * d1.1 + d1.2 * d1.2 + d1.3 * d1.3)
  let norm2 := Real.sqrt (d2.1 * d2.1 + d2.2 * d2.2 + d2.3 * d2.3)
  ∃ θ : ℝ, Real.cos θ = dot_product / (norm1 * norm2) → Real.cos θ = 5 / Real.sqrt 1377 :=
by
  sorry

end cosine_of_angle_between_diagonals_l262_262599


namespace similarity_of_triangles_l262_262386

variables {A B C H B₁ C₁ : Type} 

-- Assuming triangles and altitude AH, perpendiculars BB₁ and CC₁
def is_altitude (A B C H : Type) : Prop := sorry 
def is_perpendicular (B C B₁ C₁ A : Type) : Prop := sorry 

-- Problem Statement in Lean
theorem similarity_of_triangles (h1: is_altitude A B C H) (h2: is_perpendicular B C B₁ C₁ A) :
  similar (triangle A B C) (triangle H B₁ C₁) :=
begin
  sorry
end

end similarity_of_triangles_l262_262386


namespace min_lines_same_quadrants_l262_262383

def line (k b : ℝ) : ℝ × ℝ → Prop := fun (k b : ℝ) => ∃ x : ℝ, k ≠ 0 ∧ y = k * x + b

theorem min_lines_same_quadrants (k b : ℝ) (h₁ : k ≠ 0) :
  ∃ (n : ℕ), n > 6 ∧ ∀ (lines : list (ℝ × ℝ)), lines.length = n →
    ∃ (l₁ l₂ : line k b), quadrant_passed l₁ = quadrant_passed l₂ := 
sorry

end min_lines_same_quadrants_l262_262383


namespace XF_XG_value_l262_262075

universe u

variable {K : Type u} [field K]

/-- Given that quadrilateral ABCD is inscribed in a circle with 
side lengths AB=4, BC=3, CD=7, and DA=9, and points X and Y 
on BD such that DX/BD = 1/3 and BY/BD = 5/18,
E is the intersection of line AX and the line through Y 
parallel to AD, F is the intersection of line CX 
and the line through E parallel to AC, G is the point on the 
circle other than C that lies on line CX, 
then the value of XF * XG is (13/27) * (55/An) ^ 2.
-/
theorem XF_XG_value (A B C D X Y E F G : K)
  (circle : Set K) (hA : A ∈ circle) (hB : B ∈ circle) (hC : C ∈ circle)
  (hD : D ∈ circle)
  (AB : K) (BC : K) (CD : K) (DA : K) (hAB : AB = 4) (hBC : BC = 3)
  (hCD : CD = 7) (hDA : DA = 9)
  (BD : K) (DX : K) (BX : K) (BY : K) (h_BD : BD > 0)
  (hDX_BD : DX / BD = 1 / 3) (hBY_BD : BY / BD = 5 / 18)
  (hXY_BD : BY = BD - Y * BD / BD)
  (hDX : DX = BD * 1 / 3) (hBX : BY = BD * 5 / 18) 
  (h_parallel_YE_AD : (line_through Y E) ∥ (line_through A D)) 
  (h_parallel_EF_AC : (line_through E F) ∥ (line_through A C))
  (h_AC_BD : ∀ (AC : K), AB * CD + BC * DA = 55) :
  let AC := 4 in
  XF * XG = (13 / 27) * (55 / AC) ^ 2 :=
sorry

end XF_XG_value_l262_262075


namespace gdp_scientific_notation_l262_262379

-- Given the gross domestic product of Lantian County from January to August 2023 is 7413000000 yuan,
-- prove that the scientific notation of 7413000000 is 7.413 * 10^9.
theorem gdp_scientific_notation :
    ∃ a : ℕ, 7413000000 = 7.413 * 10^a :=
begin
  use 9,
  sorry
end

end gdp_scientific_notation_l262_262379


namespace correct_operation_l262_262534

theorem correct_operation (a : ℝ) : 
    (a ^ 2 + a ^ 4 ≠ a ^ 6) ∧ 
    (a ^ 2 * a ^ 3 ≠ a ^ 6) ∧ 
    (a ^ 3 / a ^ 2 = a) ∧ 
    ((a ^ 2) ^ 3 ≠ a ^ 5) :=
by
  sorry

end correct_operation_l262_262534


namespace find_lambda_l262_262685

def vector (λ : ℝ) : ℝ × ℝ := (λ, 1)

def norm_squared (v : ℝ × ℝ) : ℝ := v.1 ^ 2 + v.2 ^ 2

theorem find_lambda (λ : ℝ) :
  let a := vector λ
  let b := vector (λ + 2)
  norm_squared (a.1 + b.1, a.2 + b.2) = norm_squared (a.1 - b.1, a.2 - b.2) →
  λ = -1 :=
by
  sorry

end find_lambda_l262_262685


namespace number_of_skirts_l262_262496

theorem number_of_skirts (pants skirts total_ways : Nat) (h1 : pants = 4) (h2 : total_ways = 7) : skirts = 3 :=
by
  -- Initial definitions from the conditions
  let pants := 4
  let total_ways := 7

  -- Calculate skirts based on conditions
  let skirts := total_ways - pants

  -- Requirement to prove
  show skirts = 3
  from sorry

end number_of_skirts_l262_262496


namespace circle_cannot_be_rearranged_to_square_l262_262640

-- Definitions based on conditions
def is_cuttable_to_square (circle : Set Point) (square : Set Point) : Prop := 
  ∃ parts : List (Set Point), 
    (∀ part ∈ parts, is_line_or_arc part) ∧
    (⋃ part ∈ parts, part = circle) ∧
    (area (⋃ part ∈ parts, part) = area square)

-- Main theorem statement
theorem circle_cannot_be_rearranged_to_square (circle square : Set Point) :
  ¬ is_cuttable_to_square circle square :=
sorry

end circle_cannot_be_rearranged_to_square_l262_262640


namespace MN_leq_R_or_MN_leq_AB_l262_262740

-- Definitions of the geometric entities
variables {R : ℝ} {A B O M N : Type} [MetricSpace A] [MetricSpace B] [MetricSpace O] [MetricSpace M] [MetricSpace N]
variables [are_finite_measure A B O M N]

noncomputable def sector (A O B : Type) := { angle_AOB : ℝ // angle_AOB < 180 }

-- The radius condition
noncomputable def radius (O A : Type) (R : ℝ) := dist O A = R

-- The given line segment MN
variables (M N : Type)

-- The proof statement
theorem MN_leq_R_or_MN_leq_AB
  (s : sector A O B)
  (R_pos : 0 < R)
  (radius_OA : radius O A R)
  (radius_OB : radius O B R)
  (MN_within_sector : dist O M ≤ R ∧ dist O N ≤ R) : dist M N ≤ R ∨ dist M N ≤ dist A B :=
sorry

end MN_leq_R_or_MN_leq_AB_l262_262740


namespace correct_answer_is_D_l262_262984

def is_set (A : Type) : Prop :=
  -- Add the precise definition of what it means for a collection to be a set
  sorry

def example_A : Set Real := {x | x > 10^10} -- Just an example criterion
def example_B : Set Real := {x | x ≈ 0}     -- Infinitely close to zero
noncomputable def example_C : Set string := sorry  -- "beautiful little girls" is too ambiguous
def example_D : Set Real := {x | x^2 - 1 = 0}

theorem correct_answer_is_D :
  is_set example_D ∧ ¬ is_set example_A ∧ ¬ is_set example_B ∧ ¬ is_set example_C :=
sorry

end correct_answer_is_D_l262_262984


namespace probability_no_consecutive_ones_l262_262975

open Nat

-- Define the function to count valid sequences without consecutive 1s
def a_n (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else if n = 2 then 3
  else a_n (n - 1) + a_n (n - 2)

-- Define the problem statement
theorem probability_no_consecutive_ones : 
  let p := (a_n 10).to_rat / 2^10 in p = 9 / 64 := by
  sorry

end probability_no_consecutive_ones_l262_262975


namespace removed_tetrahedron_volume_l262_262218

theorem removed_tetrahedron_volume :
  let unit_cube_vertex := (0, 0, 0)
  let point_A := (1 / 3, 0, 0)
  let point_B := (0, 1 / 3, 0)
  let point_C := (0, 0, 1 / 3)
  let tetrahedron_vertices := [unit_cube_vertex, point_A, point_B, point_C]
  let volume_tetrahedron := 1 / 108
  volume_of_tetrahedron tetrahedron_vertices = volume_tetrahedron :=
by
  sorry

end removed_tetrahedron_volume_l262_262218


namespace binom_n_2_l262_262525

theorem binom_n_2 (n : ℕ) (h : 1 < n) : (nat.choose n 2) = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262525


namespace addichiffre_1998_pow_1998_eq_9_l262_262776

/-- Definition of addichiffre: Adding all the digits of a number -/
def addichiffre (n : ℕ) : ℕ :=
  n.digits.sum

/-- Three times application of addichiffre to 1998^1998 -/
def three_times_addichiffre (n : ℕ) : ℕ :=
  addichiffre (addichiffre (addichiffre n))

/-- The mathematical problem -/
theorem addichiffre_1998_pow_1998_eq_9 :
  three_times_addichiffre (1998 ^ 1998) = 9 :=
sorry

end addichiffre_1998_pow_1998_eq_9_l262_262776


namespace transformation_correct_l262_262804

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

-- Define the transformation functions
noncomputable def shift_right_by_pi_over_10 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - Real.pi / 10)
noncomputable def stretch_x_by_factor_of_2 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x / 2)

-- Define the transformed function
noncomputable def transformed_function : ℝ → ℝ :=
  stretch_x_by_factor_of_2 (shift_right_by_pi_over_10 original_function)

-- Define the expected resulting function
noncomputable def expected_function (x : ℝ) : ℝ := Real.sin (x / 2 - Real.pi / 10)

-- State the theorem
theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = expected_function x :=
by
  sorry

end transformation_correct_l262_262804


namespace solve_for_y_l262_262698

theorem solve_for_y (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end solve_for_y_l262_262698


namespace probability_value_at_least_75_cents_l262_262187

-- Given conditions
def box_contains (pennies nickels quarters : ℕ) : Prop :=
  pennies = 4 ∧ nickels = 3 ∧ quarters = 5

def draw_without_replacement (total_coins : ℕ) (drawn_coins : ℕ) : Prop :=
  total_coins = 12 ∧ drawn_coins = 5

def equal_probability (chosen_probability : ℚ) (total_coins : ℕ) : Prop :=
  chosen_probability = 1/total_coins

-- Probability that the value of coins drawn is at least 75 cents
theorem probability_value_at_least_75_cents
  (pennies nickels quarters total_coins drawn_coins : ℕ)
  (chosen_probability : ℚ) :
  box_contains pennies nickels quarters →
  draw_without_replacement total_coins drawn_coins →
  equal_probability chosen_probability total_coins →
  chosen_probability = 1/792 :=
by
  intros
  sorry

end probability_value_at_least_75_cents_l262_262187


namespace trajectory_and_min_value_l262_262013

variables {x y : ℝ}

-- Conditions from the problem
def foci_F1 : ℝ × ℝ := (0, -Real.sqrt 3)
def foci_F2 : ℝ × ℝ := (0, Real.sqrt 3)
def eccentricity : ℝ := Real.sqrt 3 / 2

-- Definitions related to the ellipse
def ellipse_eq (a b x y : ℝ) := (y^2)/(a^2) + (x^2)/(b^2) = 1
def a := Real.sqrt 4
def b := Real.sqrt 1

-- Trajectory equation of point M
def trajectory_eq (x y : ℝ) := (1 / x^2) + (4 / y^2) = 1

-- Prove the correct answers given the conditions
theorem trajectory_and_min_value :
  (∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ ellipse_eq a b x y ∧ trajectory_eq x y) ∧ 
  (∃ (x : ℝ), 1 < x ∧ let y := (Real.sqrt 4) / Real.sqrt (1 - (1 / x^2)) in 3 = Real.sqrt (x^2 + y^2)) :=
by
  -- Proof omitted
  sorry

end trajectory_and_min_value_l262_262013


namespace find_n_l262_262288

theorem find_n (n : ℤ) : 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 16427 [MOD 15] ↔ n = 2 := 
by sorry

end find_n_l262_262288


namespace fundraiser_full_price_revenue_l262_262209

theorem fundraiser_full_price_revenue :
  ∃ (f h p : ℕ), f + h = 200 ∧ 
                f * p + h * (p / 2) = 2700 ∧ 
                f * p = 600 :=
by 
  sorry

end fundraiser_full_price_revenue_l262_262209


namespace cistern_filling_time_l262_262951

theorem cistern_filling_time{F E : ℝ} (hF: F = 1 / 4) (hE: E = 1 / 9) :
  (1 / (F - E) = 7.2) :=
by
  rw [hF, hE]
  have net_rate := 0.25 - 1 / 9
  rw net_rate
  exact (1 / (0.25 - 1 / 9)) = 7.2
  sorry

end cistern_filling_time_l262_262951


namespace student_math_percentage_l262_262213

-- Define the variables for the percentages
variables (M : ℝ) 

-- Given conditions
def history_percentage := 84 / 100
def third_subject_percentage := 67 / 100
def average_percentage := 75 / 100

-- Defining the proof problem
theorem student_math_percentage :
  (\(M : ℝ)),
  ((M + history_percentage + third_subject_percentage) / 3 = average_percentage) -> 
  (M = 74 / 100) := 
sorry

end student_math_percentage_l262_262213


namespace house_number_units_digit_is_five_l262_262440

/-- Define the house number as a two-digit number -/
def is_two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- Define the properties for the statements -/
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_power_of_prime (n : ℕ) : Prop := ∃ p : ℕ, Nat.Prime p ∧ p ^ Nat.log p n = n
def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def has_digit_seven (n : ℕ) : Prop := (n / 10 = 7 ∨ n % 10 = 7)

/-- The theorem stating that the units digit of the house number is 5 -/
theorem house_number_units_digit_is_five (n : ℕ) 
  (h1 : is_two_digit_number n)
  (h2 : (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n) ∨ 
        (¬is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n))
  : n % 10 = 5 := 
sorry

end house_number_units_digit_is_five_l262_262440


namespace percentOfNonUnionWomenIs90_l262_262011

variable (totalEmployees : ℕ) (percentMen : ℚ) (percentUnionized : ℚ) (percentUnionizedMen : ℚ)

noncomputable def percentNonUnionWomen : ℚ :=
  let numberOfMen := percentMen * totalEmployees
  let numberOfUnionEmployees := percentUnionized * totalEmployees
  let numberOfUnionMen := percentUnionizedMen * numberOfUnionEmployees
  let numberOfNonUnionEmployees := totalEmployees - numberOfUnionEmployees
  let numberOfNonUnionMen := numberOfMen - numberOfUnionMen
  let numberOfNonUnionWomen := numberOfNonUnionEmployees - numberOfNonUnionMen
  (numberOfNonUnionWomen / numberOfNonUnionEmployees) * 100

theorem percentOfNonUnionWomenIs90
  (h1 : percentMen = 46 / 100)
  (h2 : percentUnionized = 60 / 100)
  (h3 : percentUnionizedMen = 70 / 100) : percentNonUnionWomen 100 46 60 70 = 90 :=
sorry

end percentOfNonUnionWomenIs90_l262_262011


namespace distance_between_neg2_and_3_l262_262821
-- Import the necessary Lean libraries

-- State the theorem to prove the distance between -2 and 3 is 5
theorem distance_between_neg2_and_3 : abs (3 - (-2)) = 5 := by
  sorry

end distance_between_neg2_and_3_l262_262821


namespace donny_remaining_money_l262_262276

theorem donny_remaining_money :
  let initial_amount := 78
  let kite_cost := 8
  let frisbee_cost := 9
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  sorry

end donny_remaining_money_l262_262276


namespace incenter_x_coordinate_eq_l262_262579

theorem incenter_x_coordinate_eq (x y : ℝ) :
  (x = y) ∧ 
  (y = -x + 3) → 
  x = 3 / 2 := 
sorry

end incenter_x_coordinate_eq_l262_262579


namespace eccentricity_of_ellipse_l262_262336

noncomputable def ellipse (a b c : ℝ) :=
  (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + c^2) ∧ (b = 2 * c)

theorem eccentricity_of_ellipse (a b c : ℝ) (h : ellipse a b c) :
  (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l262_262336


namespace barry_magic_increase_l262_262784

theorem barry_magic_increase (n : ℕ) : 
  (∀ (x : ℝ), x > 0 → (x * (∏ k in range n, (k + 4) / (k + 3)) = 50 * x)) → n = 147 :=
by
  sorry

end barry_magic_increase_l262_262784


namespace n_equals_23_l262_262408

open Finset

theorem n_equals_23 {k n : ℕ} (hk : k ≥ 6) (hn : n = 2 * k - 1)
  (T : Finset (Vector ℕ n)) (x y : Vector ℕ n)
  (d : Vector ℕ n → Vector ℕ n → ℕ)
  (h_dist : ∀ x y : Vector ℕ n, d x y = x.toList.zipWith (λ a b, if a ≠ b then 1 else 0) y.toList |>.sum)
  (S : Finset (Vector ℕ n))
  (hS : S.card = 2 ^ k)
  (h_unique : ∀ x ∈ T, ∃! y ∈ S, d x y ≤ 3) :
  n = 23 :=
sorry

end n_equals_23_l262_262408


namespace gumball_difference_l262_262241

theorem gumball_difference :
  let c := 17
  let l := 12
  let a := 24
  let t := 8
  let n := c + l + a + t
  let low := 14
  let high := 32
  ∃ x : ℕ, (low ≤ (n + x) / 7 ∧ (n + x) / 7 ≤ high) →
  (∃ x_min x_max, x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126) :=
by
  sorry

end gumball_difference_l262_262241


namespace girl_needs_120_oranges_l262_262573

-- Define the cost and selling prices per pack
def cost_per_pack : ℤ := 15   -- cents
def oranges_per_pack_cost : ℤ := 4
def sell_per_pack : ℤ := 30   -- cents
def oranges_per_pack_sell : ℤ := 6

-- Define the target profit
def target_profit : ℤ := 150  -- cents

-- Calculate the cost price per orange
def cost_per_orange : ℚ := cost_per_pack / oranges_per_pack_cost

-- Calculate the selling price per orange
def sell_per_orange : ℚ := sell_per_pack / oranges_per_pack_sell

-- Calculate the profit per orange
def profit_per_orange : ℚ := sell_per_orange - cost_per_orange

-- Calculate the number of oranges needed to achieve the target profit
def oranges_needed : ℚ := target_profit / profit_per_orange

-- Lean theorem statement
theorem girl_needs_120_oranges :
  oranges_needed = 120 :=
  sorry

end girl_needs_120_oranges_l262_262573


namespace number_of_starting_positions_l262_262757

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 - 4 * x^2 = 4

-- Define the sequence of points and their recurrence relation
noncomputable def next_point (x_n : ℝ) : ℝ := (x_n^2 - 2) / (2 * x_n)

-- Define the theorem we want to prove
theorem number_of_starting_positions : 
  (∃ P_0 : ℝ, ∃ n : ℕ, P_0 = (λ (P : ℕ → ℝ), if P = 0 then 1 else next_point (P (P-1))) 2023 P_0) → 
  ∃ k : ℕ, k = 2^2023 - 2 :=
sorry

end number_of_starting_positions_l262_262757


namespace geometric_series_common_ratio_l262_262840

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l262_262840


namespace betty_cookies_brownies_l262_262233

theorem betty_cookies_brownies (cookies_per_day brownies_per_day initial_cookies initial_brownies days : ℕ) :
  cookies_per_day = 3 → brownies_per_day = 1 → initial_cookies = 60 → initial_brownies = 10 → days = 7 →
  initial_cookies - days * cookies_per_day - (initial_brownies - days * brownies_per_day) = 36 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_cookies_brownies_l262_262233


namespace product_of_midpoint_is_40_l262_262894

-- Define the coordinates of the endpoints
def point1 : ℝ × ℝ × ℝ := (3, -5, 2)
def point2 : ℝ × ℝ × ℝ := (7, -3, -6)

-- Define the calculation of the midpoint
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the product of the coordinates of the midpoint
def product_of_midpoint_coordinates (p : ℝ × ℝ × ℝ) : ℝ :=
  p.1 * p.2 * p.3

-- Prove that the product of the midpoint coordinates is 40
theorem product_of_midpoint_is_40 : 
  product_of_midpoint_coordinates (midpoint point1 point2) = 40 :=
by
  sorry

end product_of_midpoint_is_40_l262_262894


namespace largest_possible_triangle_area_l262_262217

noncomputable def triangle_max_area (median_AD median_BE : ℝ) : ℝ :=
let AG := (2 / 3) * median_AD in
let BG := (2 / 3) * median_BE in
let area_AGB := (1 / 2) * AG * BG in
3 * area_AGB

theorem largest_possible_triangle_area (median_AD median_BE : ℝ) (hAD : median_AD = 9) (hBE : median_BE = 12) : 
  triangle_max_area median_AD median_BE = 72 := 
by
  have hAG : ℝ := 6 := by rw [←hAD]; exact ((2 : ℝ) / 3) * 9
  have hBG : ℝ := 8 := by rw [←hBE]; exact ((2 : ℝ) / 3) * 12
  have h_area_AGB : ℝ := 24 := (1 / 2) * 6 * 8
  exact (3 : ℝ) * 24 == 72
  sorry

end largest_possible_triangle_area_l262_262217


namespace total_discount_is_15_l262_262831

structure Item :=
  (price : ℝ)      -- Regular price
  (discount_rate : ℝ) -- Discount rate in decimal form

def t_shirt : Item := {price := 25, discount_rate := 0.3}
def jeans : Item := {price := 75, discount_rate := 0.1}

def discount (item : Item) : ℝ :=
  item.discount_rate * item.price

def total_discount (items : List Item) : ℝ :=
  items.map discount |>.sum

theorem total_discount_is_15 :
  total_discount [t_shirt, jeans] = 15 := by
  sorry

end total_discount_is_15_l262_262831


namespace dot_product_OM_ON_l262_262032

noncomputable def point := (ℝ × ℝ)

def M : point := (1, 1)

def line_eq (p : point) : Prop := p.1 + p.2 = 2

def distance (p1 p2 : point) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def dot_product (p1 p2 : point) : ℝ := p1.1 * p2.1 + p1.2 * p2.2

axiom distance_M_N : ∀ N : point, line_eq N → distance M N = real.sqrt 2

theorem dot_product_OM_ON : ∀ (N : point), line_eq N → distance M N = real.sqrt 2 → dot_product M N = 2 :=
by
  intro N hN hdist
  sorry

end dot_product_OM_ON_l262_262032


namespace isosceles_triangle_largest_angle_l262_262720

theorem isosceles_triangle_largest_angle (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = 50) :
  A + B + C = 180 →
  C = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l262_262720


namespace contribution_amount_l262_262104

-- Definitions based on conditions
def initial_members : ℕ := 10
def new_members : ℕ := 15
def amount_diff : ℤ := 100

-- The initial contribution each member made
def initial_contribution (x : ℤ) : Prop := 
  initial_members * x = new_members * (x - amount_diff)

-- The problem as a Lean 4 statement
theorem contribution_amount : ∃ x : ℤ, initial_contribution x ∧ x = 300 :=
by
  have h : initial_contribution 300 := by
    unfold initial_contribution
    calc
      10 * 300 = 3000     : by norm_num
      ...    = 15 * 200   : by norm_num
      ...    = 15 * (300 - 100) : by norm_num
  exact ⟨300, h, rfl⟩

end contribution_amount_l262_262104


namespace max_expression_value_l262_262766

theorem max_expression_value :
  ∃ A B C D : ℕ,
    (A ∈ {12, 14, 16, 18}) ∧ (B ∈ {12, 14, 16, 18}) ∧ (C ∈ {12, 14, 16, 18}) ∧ (D ∈ {12, 14, 16, 18}) ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (A * B + B * C + B * D + C * D = 1116) :=
  sorry

end max_expression_value_l262_262766


namespace least_possible_perimeter_l262_262124

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end least_possible_perimeter_l262_262124


namespace find_x_l262_262622

theorem find_x (x : ℝ) (h : sqrt (x - 3) = 10) : x = 103 := 
by 
  sorry

end find_x_l262_262622


namespace remainder_of_7n_div_4_l262_262914

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l262_262914


namespace wanda_can_eat_100000_numbers_l262_262162

-- Define the main theorem
theorem wanda_can_eat_100000_numbers :
  ∃ (n : ℕ), n ≤ 2011 ∧ ∃ (S : Finset (ℕ × ℕ)), S.card ≥ 100000 ∧
  (∀ ⟨i, j⟩ ∈ S, i ≤ n ∧ j ≤ i) ∧
  (∀ ⟨a, b⟩ ⟨c, d⟩ ⟨e, f⟩ ∈ S, (Nat.choose a b) + (Nat.choose c d) ≠ (Nat.choose e f)) :=
sorry

end wanda_can_eat_100000_numbers_l262_262162


namespace principal_amount_is_correct_l262_262528

-- We define the conditions given in the problem.
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- Let's define the given problem.
def principal_condition (SI R T P : ℝ) : Prop :=
  SI = simple_interest P R T

-- Define the known values.
def given_SI : ℝ := 192
def given_R : ℝ := 6
def given_T : ℝ := 4
def correct_answer : ℝ := 800

-- The theorem to prove that under the given conditions, the principal amount (P) is 800.
theorem principal_amount_is_correct : principal_condition given_SI given_R given_T correct_answer :=
by
  sorry

end principal_amount_is_correct_l262_262528


namespace rectangular_prism_surface_area_l262_262468

theorem rectangular_prism_surface_area :
  ∀ (l w h : ℕ), (l = 10 ∧ w = 6 ∧ h = 5) → (2 * ((l * w) + (l * h) + (w * h)) = 280) :=
by
  intros l w h h_conditions
  rcases h_conditions with ⟨h_l, h_w, h_h⟩
  rw [h_l, h_w, h_h]
  sorry

end rectangular_prism_surface_area_l262_262468


namespace nine_sided_polygon_diagonals_l262_262202

theorem nine_sided_polygon_diagonals (n : ℕ) (h_n : n = 9) (convex : Prop) (obtuse_angle : Prop) : (n * (n - 3)) / 2 = 27 :=
by
  -- Given conditions
  have h1 : n = 9 := h_n
  have h2 : convex := convex
  have h3 : obtuse_angle := obtuse_angle
  -- The proof goes here
  sorry

end nine_sided_polygon_diagonals_l262_262202


namespace find_cos_A_l262_262385

-- Define the problem context
variables {A B C : ℝ} {a b c : ℝ}
variable (ABC_is_triangle : True)  -- Assuming this is a triangle, with sides opposite A, B, C being a, b, c respectively.

-- Define the given conditions
variables (h1 : sin B - sin C = (1/4) * sin A)
variables (h2 : 2 * b = 3 * c)

-- Define the proof goal
theorem find_cos_A : cos A = -1/4 :=
by sorry

end find_cos_A_l262_262385


namespace shiftedParabolaIsCorrect_l262_262709

def originalFunction (x : ℝ) : ℝ := 2 * x^2

def shiftLeft (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ :=
  fun x => f (x + h)

def shiftUp (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  fun x => f x + k

theorem shiftedParabolaIsCorrect :
  shiftUp (shiftLeft originalFunction 1) 3 = (λ x, 2*(x+1)^2 + 3) :=
by
  ext x
  unfold shiftUp shiftLeft originalFunction
  sorry

end shiftedParabolaIsCorrect_l262_262709


namespace quadratic_no_real_roots_implies_inequality_l262_262552

theorem quadratic_no_real_roots_implies_inequality (a b c : ℝ) :
  let A := b + c
  let B := a + c
  let C := a + b
  (B^2 - 4 * A * C < 0) → 4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by
  intro h
  sorry

end quadratic_no_real_roots_implies_inequality_l262_262552


namespace batsman_total_score_eq_120_l262_262947

/-- A batsman's runs calculation including boundaries, sixes, and running between wickets. -/
def batsman_runs_calculation (T : ℝ) : Prop :=
  let runs_from_boundaries := 5 * 4
  let runs_from_sixes := 5 * 6
  let runs_from_total := runs_from_boundaries + runs_from_sixes
  let runs_from_running := 0.5833333333333334 * T
  T = runs_from_total + runs_from_running

theorem batsman_total_score_eq_120 :
  ∃ T : ℝ, batsman_runs_calculation T ∧ T = 120 :=
sorry

end batsman_total_score_eq_120_l262_262947


namespace triangle_min_perimeter_l262_262125

theorem triangle_min_perimeter:
  ∃ x : ℤ, 27 < x ∧ x < 75 ∧ (24 + 51 + x) = 103 :=
begin
  sorry
end

end triangle_min_perimeter_l262_262125


namespace transformation_composition_l262_262727

def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

theorem transformation_composition :
  f (g (-1, 2)) = (1, -3) :=
by {
  sorry
}

end transformation_composition_l262_262727


namespace magnitude_of_sum_of_vectors_l262_262352

open Real

variables (a b : ℝ^3)
variables (h_a : ‖a‖ = 1) (h_b : ‖b‖ = 2)
variables (θ : ℝ) (h_angle : θ = π / 3)

theorem magnitude_of_sum_of_vectors (h_a : ‖a‖ = 1) (h_b : ‖b‖ = 2) (h_angle : θ = π / 3) :
  ‖a + b‖ = sqrt 7 :=
by
  sorry

end magnitude_of_sum_of_vectors_l262_262352


namespace fraction_beans_remain_l262_262873

theorem fraction_beans_remain (J B B_remain : ℝ) 
  (h1 : J = 0.10 * (J + B)) 
  (h2 : J + B_remain = 0.60 * (J + B)) : 
  B_remain / B = 5 / 9 := 
by 
  sorry

end fraction_beans_remain_l262_262873


namespace common_ratio_of_geometric_series_l262_262867

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l262_262867


namespace expression_equals_13_div_2_l262_262998

noncomputable def expression : ℝ :=
  log (sqrt 2) + log (sqrt 5) + (2:ℝ)^0 + (5^(1/3))^2 * (5^(1/3))

theorem expression_equals_13_div_2 : expression = 13/2 :=
  sorry

end expression_equals_13_div_2_l262_262998


namespace find_parallel_line_l262_262287

-- Definition of the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Definition of the original line equation
def original_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Definition of the desired line equation
def desired_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement: defining the desired line based on the point and parallelism condition
theorem find_parallel_line (x y : ℝ) (hx : point.fst = 0) (hy : point.snd = 1) :
  ∃ m : ℝ, (2 * x + y + m = 0) ∧ (2 * 0 + 1 + m = 0) → desired_line x y :=
sorry

end find_parallel_line_l262_262287


namespace midpoint_EF_on_A0C0_l262_262225

-- Given: Triangle ABC exists with points of tangency A0 and C0, point D on AC,
-- and points E and F symmetric to D with respect to angle bisectors of ∠A and ∠C respectively.
variable (A B C A0 C0 D E F : Point)
variable (ABC : Triangle A B C)

-- Conditions
axiom incircle_tangency_A0 : A0 = ABC.incircle_tangency_point B C
axiom incircle_tangency_C0 : C0 = ABC.incircle_tangency_point C A
axiom point_D_on_AC : D ∈ Line A C
axiom E_symmetric_to_D : E = symmetric_point D (angle_bisector A ABC)
axiom F_symmetric_to_D : F = symmetric_point D (angle_bisector C ABC)

-- Proof to show the midpoint of segment EF lies on A0C0
theorem midpoint_EF_on_A0C0 :
  let M := midpoint E F in
  M ∈ Line A0 C0 :=
sorry

end midpoint_EF_on_A0C0_l262_262225


namespace probability_novels_consecutive_l262_262781

theorem probability_novels_consecutive (books : Fin 12 → Prop) (novels : Fin 4 → Prop) (h_distinct : Function.Injective books) (h_novels_by_author : ∃ f, ∀ i, novels i = books (f i)) :
  (number_of_arrangements (λ n, ∃ k, (∀ i, novels i = books (k i))) / number_of_arrangements books) = 1 / 55 :=
by
  sorry

def number_of_arrangements {α : Type*} (arrangement : α → Prop) : ℕ := sorry

end probability_novels_consecutive_l262_262781


namespace problem_solution_correct_l262_262342

-- Definition of the function f as given
def f (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Conditions provided in the problem
variables {a b c : ℝ}
-- a ≠ 0
h1 : a ≠ 0
-- f(x) = x has no real roots
h2 : ∀ x : ℝ, f a b c x ≠ x

-- Prove the correctness of the statements
theorem problem_solution_correct :
  (∀ (x : ℝ), f a b c (f a b c x) ≠ x) ∧
  (∀ (x : ℝ), a > 0 → f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ (x : ℝ), f a b c (f a b c x) < x) :=
by {
  -- Proof omitted
  sorry
}

end problem_solution_correct_l262_262342


namespace find_f_l262_262774

theorem find_f (f : ℝ → ℝ) (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → x ≤ y → f x ≤ f y)
  (h₂ : ∀ x : ℝ, 0 < x → f (x ^ 4) + f (x ^ 2) + f x + f 1 = x ^ 4 + x ^ 2 + x + 1) :
  ∀ x : ℝ, 0 < x → f x = x := 
sorry

end find_f_l262_262774


namespace exists_non_intersected_side_l262_262018

theorem exists_non_intersected_side (n : ℕ) (P : Point) (polygon : polygon (2 * n)) 
  (h_convex : convex polygon) (h_interior : interior polygon P) : 
  ∃ side : set Point, side ∈ sides polygon ∧ ∀ vertex : Point, vertex ∈ vertices polygon → line_through P vertex ∉ side := 
sorry

end exists_non_intersected_side_l262_262018


namespace find_temperature_l262_262228

variable (T : ℝ)

namespace SkidConditions
  def skidChanceIncreasePerDegreeDrop := 5 / 3 
  def controlRegainingChance := 0.40
  def seriousAccidentOverallChance := 0.24

  def skidChanceBelowTemp32 (d : ℝ) : ℝ := d * skidChanceIncreasePerDegreeDrop
  def seriousAccidentChanceWhenSkidding (d : ℝ) : ℝ := skidChanceBelowTemp32 d * (1 - controlRegainingChance)
end SkidConditions

open SkidConditions

theorem find_temperature (h: seriousAccidentOverallChance = seriousAccidentChanceWhenSkidding (32 - T)) : 
  T = 8 := 
sorry

end find_temperature_l262_262228


namespace televisions_bought_l262_262229

theorem televisions_bought (T : ℕ)
  (television_cost : ℕ := 50)
  (figurine_cost : ℕ := 1)
  (num_figurines : ℕ := 10)
  (total_spent : ℕ := 260) :
  television_cost * T + figurine_cost * num_figurines = total_spent → T = 5 :=
by
  intros h
  sorry

end televisions_bought_l262_262229


namespace greatest_multiple_of_4_l262_262102

theorem greatest_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x > 0) (h3 : x^3 < 500) : x ≤ 4 :=
by sorry

end greatest_multiple_of_4_l262_262102


namespace cover_cube_with_strip_l262_262206

theorem cover_cube_with_strip :
  ∃ strip : ℝ × ℝ, strip = (12, 1) →
  ∃ cube : ℝ, cube = 1 →
  can_cover_in_two_layers strip cube :=
sorry

def can_cover_in_two_layers (strip : ℝ × ℝ) (cube : ℝ) : Prop :=
  let (length, width) := strip in
  (length * width = 12) ∧ (cube = 1) ∧ 
  ((6 * cube^2) * 2 = length * width)


end cover_cube_with_strip_l262_262206


namespace maximum_area_of_flower_bed_l262_262779

-- Definitions based on conditions
def length_of_flower_bed : ℝ := 150
def total_fencing : ℝ := 450

-- Question reframed as a proof statement
theorem maximum_area_of_flower_bed :
  ∀ (w : ℝ), 2 * w + length_of_flower_bed = total_fencing → (length_of_flower_bed * w = 22500) :=
by
  intro w h
  sorry

end maximum_area_of_flower_bed_l262_262779


namespace rational_iff_geometric_sequence_l262_262797

theorem rational_iff_geometric_sequence (x : ℚ) :
  (∃ a b c : ℕ, 0 ≤ a ∧ a < b ∧ b < c ∧ (x + a : ℚ) * (x + c : ℚ) = (x + b : ℚ)^2) ↔ x ∈ ℚ :=
by
  sorry

end rational_iff_geometric_sequence_l262_262797


namespace units_digit_problem_l262_262902

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : Nat) : Nat :=
  n % 10

noncomputable def problem_units_digit_sum : Nat :=
  (List.sum (List.map (λ n, units_digit (factorial n)) (List.range 25).map (+1)) + 7) % 10

theorem units_digit_problem :
  problem_units_digit_sum = 0 := by
  sorry

end units_digit_problem_l262_262902


namespace geometric_series_common_ratio_l262_262844

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l262_262844


namespace max_colors_49_to_94_l262_262290

def coloring_property (χ : ℕ → ℕ) (a b c : ℕ) : Prop :=
  if χ(a) = χ(b) ∧ χ(c) ≠ χ(a) then ¬ c ∣ (a + b) else True

theorem max_colors_49_to_94 :
  (∀ χ : ℕ → ℕ,
    (∀ a b c : ℕ, 49 ≤ a ∧ a ≤ 94 ∧ 49 ≤ b ∧ b ≤ 94 ∧ 49 ≤ c ∧ c ≤ 94 →
      coloring_property χ a b c) →
    ∃ k : ℕ, k = 2 
  ) :=
sorry

end max_colors_49_to_94_l262_262290


namespace probability_of_valid_triples_l262_262883

noncomputable def probability_valid_triples : ℚ :=
  let choices := (Finset.range 30).filter (λ x, x ≠ 0),
  let count_valid (Triple : (ℕ × ℕ × ℕ) → Prop) : ℕ :=
    (Finset.filter (λ t, Triple t) (choices.product (choices.product choices))).card in
  count_valid (λ t, ∃ x y z: ℕ, t = (x, y, z) ∧ x ≠ y ∧ y ≠ z ∧ x ∣ y ∧ y ∣ z) / (4060 : ℚ)

theorem probability_of_valid_triples :
  ∃ V : ℕ, probability_valid_triples = V / 4060 :=
sorry

end probability_of_valid_triples_l262_262883


namespace units_sold_to_customer_c_l262_262977

theorem units_sold_to_customer_c 
  (initial_units : ℕ)
  (defective_units : ℕ)
  (units_a : ℕ)
  (units_b : ℕ)
  (units_c : ℕ)
  (h_initial : initial_units = 20)
  (h_defective : defective_units = 5)
  (h_units_a : units_a = 3)
  (h_units_b : units_b = 5)
  (h_non_defective : initial_units - defective_units = 15)
  (h_sold_all : units_a + units_b + units_c = 15) :
  units_c = 7 := by
  -- use sorry to skip the proof
  sorry

end units_sold_to_customer_c_l262_262977


namespace sum_P_equals_1024_l262_262969

-- Definition of number of paths P(a, b) from (0,0) to (a,b) with 10 steps
def P (a b : ℕ) : ℕ := Nat.choose 10 a

-- Theorem stating the desired result
theorem sum_P_equals_1024 : (∑ i in Finset.range 11, P i (10 - i)) = 1024 := by
  sorry

end sum_P_equals_1024_l262_262969


namespace initial_fund_correct_l262_262152

-- Define the conditions given
def planned_amount_per_employee : ℝ := 50
def received_amount_per_employee : ℝ := 45
def undistributed_amount : ℝ := 95
def number_of_employees : ℝ := 95 / (planned_amount_per_employee - received_amount_per_employee)

-- Define the initial fund calculation
def initial_fund : ℝ := number_of_employees * planned_amount_per_employee

-- Define the correct answer
def correct_initial_fund : ℝ := 950

-- State the theorem
theorem initial_fund_correct : initial_fund = correct_initial_fund := 
by
  -- Proof is not required, so it's omitted
  sorry

end initial_fund_correct_l262_262152


namespace min_value_l262_262683

-- Define the set of numbers M
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2020 }

-- Define the three colors
inductive Color
| Red | Yellow | Blue

open Color

-- Define the colors for each number in M
def color : ℕ → Color
| n => sorry  -- coloring function that ensures each color is used at least once

-- Define set S1
def S1 : Set (ℕ × ℕ × ℕ) := 
  { (x, y, z) | x ∈ M ∧ y ∈ M ∧ z ∈ M ∧ color x = color y ∧ color y = color z ∧ (x + y + z) % 2020 = 0 }

-- Define set S2
def S2 : Set (ℕ × ℕ × ℕ) := 
  { (x, y, z) | x ∈ M ∧ y ∈ M ∧ z ∈ M ∧ color x ≠ color y ∧ color y ≠ color z ∧ color x ≠ color z ∧ (x + y + z) % 2020 = 0 }

-- The proof problem
theorem min_value : 2 * S1.to_finset.card - S2.to_finset.card = 2 :=
by
  -- Proof is omitted
  sorry

end min_value_l262_262683


namespace cube_diagonal_equidistant_points_form_hexagon_l262_262291

noncomputable section

def point_of_surface_equidistant_hexagon (a : ℝ) : Prop :=
  ∀ (P : EuclideanGeometry.Point ℝ 3),
  (is_on_cube_surface P a ∧ is_equidistant_from_diagonal_endpoints P a) →
    is_on_regular_hexagon P (a * Real.sqrt 2 / 2)

/--
The points on the surface of a cube that are equidistant from the endpoints 
of a space diagonal form a regular hexagon with side length \( \frac{a \sqrt{2}}{2} \).
-/
theorem cube_diagonal_equidistant_points_form_hexagon (a : ℝ) :
  point_of_surface_equidistant_hexagon a :=
sorry

end cube_diagonal_equidistant_points_form_hexagon_l262_262291


namespace power_sum_inequality_l262_262330

theorem power_sum_inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
by sorry

end power_sum_inequality_l262_262330


namespace greatest_percentage_increase_is_C_l262_262062

noncomputable def City := Type
def P1970 (c : City) : ℕ
def P1980 (c : City) : ℕ

axiom City_A : City
axiom City_B : City
axiom City_C : City
axiom City_D : City
axiom City_E : City

-- Population data for 1970
axiom P1970_A : P1970 City_A = 40
axiom P1970_B : P1970 City_B = 50
axiom P1970_C : P1970 City_C = 70
axiom P1970_D : P1970 City_D = 100
axiom P1970_E : P1970 City_E = 120

-- Population data for 1980
axiom P1980_A : P1980 City_A = 50
axiom P1980_B : P1980 City_B = 70
axiom P1980_C : P1980 City_C = 100
axiom P1980_D : P1980 City_D = 130
axiom P1980_E : P1980 City_E = 160

theorem greatest_percentage_increase_is_C :
  ∀ x ∈ ({City_A, City_B, City_C, City_D, City_E} : set City), 
    (P1980 City_C : ℚ) / (P1970 City_C) ≥ (P1980 x) / (P1970 x) := 
by
  sorry

end greatest_percentage_increase_is_C_l262_262062


namespace area_of_circle_l262_262600

theorem area_of_circle 
  (r : ℝ → ℝ)
  (h : ∀ θ : ℝ, r θ = 3 * Real.cos θ - 4 * Real.sin θ) :
  ∃ A : ℝ, A = (25 / 4) * Real.pi :=
by
  sorry

end area_of_circle_l262_262600


namespace inequality_solution_eq_l262_262626

theorem inequality_solution_eq :
  ∀ y : ℝ, 2 ≤ |y - 5| ∧ |y - 5| ≤ 8 ↔ (-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13) :=
by
  sorry

end inequality_solution_eq_l262_262626


namespace problem_lean_statement_l262_262460

theorem problem_lean_statement (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 * x ^ 2 + 5 * x + 3)
  (h2 : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 0 :=
by sorry

end problem_lean_statement_l262_262460


namespace possible_values_of_k_l262_262103

variable {R : Type*} [CommRing R] [IsDomain R] [NormedSpace ℂ R]

theorem possible_values_of_k 
  {a b c d k : ℂ} 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hd : d ≠ 0) 
  (hk1 : a * k^2 + b * k + c = 0) 
  (hk2 : b * k^2 + c * k + d = 0) 
  : k = 1 ∨ k = (Complex.ofReal (-1) + Complex.I * Complex.ofReal (Real.sqrt 3)) / Complex.ofReal 2 ∨ k = (Complex.ofReal (-1) - Complex.I * Complex.ofReal (Real.sqrt 3)) / Complex.ofReal 2 := 
sorry

end possible_values_of_k_l262_262103


namespace remainder_square_division_l262_262141

theorem remainder_square_division : ((9^2) % (Int.ofNat (Nat.cbrt 125))) = 1 := by
  sorry

end remainder_square_division_l262_262141


namespace two_conditions_valid_l262_262168

-- Definitions for each condition:
def condition1 : Prop := {x | x^2 + 1 = 0} = {x | (2x + 4 > 0 ∧ x + 3 < 0)}
def condition2 : Prop := {y | ∃ x, y = 2x^2 + 1} = {x | ∃ y, y = 2x^2 + 1}
def condition3 : Prop := {x | ∃ n : ℕ, x = (1 - (-1)^n) / 2} = {x | x ∈ ℕ ∧ -1 < x ∧ x < 2}
def condition4 : Prop := {p | ∃ x y, p = (x, y) ∧ y = Real.sqrt (x - 1) + Real.sqrt (1 - x)} = {(0, 1)}

-- Proof that exactly two conditions are valid
theorem two_conditions_valid : 
  ((condition1 → True) ∧ (condition1 → False)) +
  ((condition3 → True) ∧ (condition3 → False)) =
  2 := 
by
  -- Skipping the detailed proof steps
  sorry

end two_conditions_valid_l262_262168


namespace smallest_possible_sum_S_l262_262809

theorem smallest_possible_sum_S (n : ℕ) (S : ℕ) :
  ∃ n S, (prob_sum n 2027 = prob_sum n S) → (S = 339 ∧ 6 * n ≥ 2027 ∧ n > 0) :=
begin
  use 338,
  use 339,
  sorry
end

end smallest_possible_sum_S_l262_262809


namespace collinear_vectors_l262_262687

theorem collinear_vectors (λ : ℝ) 
  (h : ∃ k : ℝ, (\lambda, λ) = (k * 3\lambda, k * 1)) : 
  λ = 0 ∨ λ = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end collinear_vectors_l262_262687


namespace math_problem_proof_l262_262350

open Set

noncomputable def U : Set ℝ := set.univ

noncomputable def M : Set ℕ := {x | x^2 - 2 * x ≤ 0}

noncomputable def A : Set ℝ := {y | ∃ x : ℕ, y = 2^x + 1}

noncomputable def C_U_A : Set ℝ := U \ A

theorem math_problem_proof :
  M ∩ (C_U_A) = {0, 1} :=
by
  sorry

end math_problem_proof_l262_262350


namespace find_size_of_some_ounce_glass_l262_262597

variable (S : ℕ)

def water_used_by_S_glasses := 6 * S
def water_used_by_8_ounce_glasses := 4 * 8
def water_used_by_4_ounce_glasses := 15 * 4
def total_water := 122

theorem find_size_of_some_ounce_glass (h : water_used_by_S_glasses + water_used_by_8_ounce_glasses + water_used_by_4_ounce_glasses = total_water) : S = 5 := by
  sorry

end find_size_of_some_ounce_glass_l262_262597


namespace shaded_regions_area_l262_262737

theorem shaded_regions_area (r1 r2 : ℝ) (h1 : r1 = 3) (h2 : r2 = 6) : 
  let left_area : ℝ := 18
  let right_area : ℝ := 72
  let left_semicircle_area : ℝ := 4.5 * Real.pi
  let right_semicircle_area : ℝ := 18 * Real.pi
  let total_shaded_area : ℝ := left_area - left_semicircle_area + right_area - right_semicircle_area
  total_shaded_area ≈ 19.3 :=
by
  have left_area := 3 * 6
  have right_area := 6 * 12
  have left_semicircle_area := 0.5 * Real.pi * 3^2
  have right_semicircle_area := 0.5 * Real.pi * 6^2
  have total_shaded_area := (left_area - left_semicircle_area + right_area - right_semicircle_area)
  have approx := total_shaded_area ≈ 19.3
  sorry

end shaded_regions_area_l262_262737


namespace min_value_expression_l262_262480

theorem min_value_expression (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 1/x + 1/y = 1) : 
  (∃ x y, 0 < x ∧ 0 < y ∧ (1 / x + 1 / y = 1) ∧ 
  (3 * x / (x - 1) + 8 * y / (y - 1) = 11 + 4 * Real.sqrt 6)) :=
begin
  sorry
end

end min_value_expression_l262_262480


namespace abs_expr1_abs_expr2_l262_262806

theorem abs_expr1 (x : ℝ) : 
  (|3 * x - 2| + |2 * x + 3| = 
    if x < -3 / 2 then -5 * x - 1 
    else if -3 / 2 ≤ x ∧ x < 2 / 3 then -x + 5 
    else if x ≥ 2 / 3 then 5 * x + 1 
    else 0) := sorry

theorem abs_expr2 (x : ℝ) : 
  (||x - 1| - 3| + |3 * x + 1| =
    if x < -2 then -4 * x - 3 
    else if -2 ≤ x ∧ x < -1 / 3 then -2 * x + 1 
    else if -1 / 3 ≤ x ∧ x < 1 then 4 * x + 3 
    else if 1 ≤ x ∧ x < 4 then 2 * x + 5 
    else if x ≥ 4 then 4 * x - 3 
    else 0) := sorry

end abs_expr1_abs_expr2_l262_262806


namespace rectangle_area_y_value_l262_262815

theorem rectangle_area_y_value :
  ∀ (y : ℝ), 
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  (y > 1) → 
  (abs (R.1 - P.1) * abs (Q.2 - P.2) = 36) → 
  y = 13 :=
by
  intros y P Q R S hy harea
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  sorry

end rectangle_area_y_value_l262_262815


namespace lines_perpendicular_l262_262308

variable {Point : Type} [AffineSpace ℝ Point]
variables {l m : Line Point} {α β : Plane Point}

-- Definitions for perpendicularity and parallelism
def Line.parallel (l1 l2 : Line Point) := ∃ p1 p2, p1 ∈ l1 ∧ p2 ∈ l2 ∧ (p2 -ᵥ p1) ∈ line_direction l1
def Line.perpendicular (l1 l2 : Line Point) := ∃ v1 v2, (v1 ∈ line_direction l1) ∧ (v2 ∈ line_direction l2) ∧ (v1 ⬝ v2 = 0)
def Line.parallel_plane (l : Line Point) (α : Plane Point) := ∃ (p1 : Point), p1 ∈ l ∧ ⟦p1⟧ = line_direction l
def Line.perpendicular_plane (l : Line Point) (α : Plane Point) := ∃ (p1 : Point), p1 ∈ l ∧ ⟦p1⟧ ⊥ α

-- Conditions
variable (hl : l.parallel_plane α)
variable (hm : l.perpendicular_plane α)

-- Statement to prove
theorem lines_perpendicular : Line.perpendicular l m :=
sorry

end lines_perpendicular_l262_262308


namespace changes_in_population_l262_262886

-- Define the conditions.
variable (environmental_factors : Type)
variable (human_factors : Type)
variable (population_numbers : Type)

-- Define the possible changes in population numbers.
inductive PopulationChange
| increase : PopulationChange
| decrease : PopulationChange
| fluctuation : PopulationChange
| extinction : PopulationChange

-- Define the main theorem statement.
theorem changes_in_population (env : environmental_factors) (hum : human_factors) 
: PopulationChange → PopulationChange.increase ∨ PopulationChange.decrease ∨ PopulationChange.fluctuation ∨ PopulationChange.extinction :=
sorry

end changes_in_population_l262_262886


namespace general_term_formula_l262_262323

def sequence_an (n : ℕ) : ℝ := (2 : ℝ) / (3 : ℝ)^(n : ℝ)

def sum_Sn (n : ℕ) : ℝ := (seq_an n).sum

def sequence_bn (n : ℕ) : ℝ := -(1 : ℝ) / (n + 1)

theorem general_term_formula 
    (a_n : ℕ → ℝ) 
    (S_n : ℕ → ℝ) 
    (b_n : ℕ → ℝ)
    (a_1_pos : a_n 1 = 2 / 3)
    (arith_seq_cond : - (3 / a_n 2) + 1 / a_n 4 = 2 / a_n 3)
    (sum_cond : ∀ n, S_n n = 1 - (1 / 3)^(n + 1))
    (bn_log_cond : ∀ n, b_n n * real.log 3 (1 - S_n (n + 1)) = 1)
    (product_sum : b_n 1 * b_n 2 + b_n 2 * b_n 3 + ... + b_n n * b_n (n + 1) = 504 / 1009)
    : (∀ n, a_n n = (2 : ℝ) / (3^(n : ℝ))) ∧ (n = 2016) :=
by
  sorry

end general_term_formula_l262_262323


namespace total_peaches_l262_262148

theorem total_peaches (ab1 ab2: Nat):
  ab1 = 5 * 20 → ab2 = 4 * 25 → ab1 + ab2 = 200 := by {
    assume h1 h2,
    sorry
}

end total_peaches_l262_262148


namespace systematic_sampling_l262_262782

theorem systematic_sampling (total : ℕ) (start : ℕ) (size : ℕ) (interval : ℕ) 
  (h_total : total = 60) (h_size : size = 5) (h_start : start = 4) (h_interval : interval = total / size) :
  [start + interval, start + 2 * interval, start + 3 * interval, start + 4 * interval] = [16, 28, 40, 52] :=
by
  have h_interval_value : interval = 12, from by {
    rw [h_interval, h_total, h_size],
    norm_num,
  },
  rw h_interval_value,
  norm_num,
  sorry

end systematic_sampling_l262_262782


namespace tan_pi_minus_alpha_l262_262307

theorem tan_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 1 / 3) (h2 : π / 2 < α ∧ α < π) :
  Real.tan (π - α) = √2 / 4 :=
by
  sorry

end tan_pi_minus_alpha_l262_262307


namespace common_ratio_of_geometric_series_l262_262854

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l262_262854


namespace problem_A_problem_B_problem_D_l262_262335

variable {R : Type*} [LinearOrderedField R]

-- Given function and its property
variable {f : R → R}
hypothesis H : ∀ (x1 x2 : R), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < -1

-- Prove the statements about the function
theorem problem_A : f (-2) > f 2 + 4 :=
sorry

theorem problem_B : ∀ (x : R), f x > f (x + 1) + 1 :=
sorry

theorem problem_D : ∀ (a : R), a ≠ 0 → f (|a| + 1 / |a|) + (|a| + 1 / |a|) < f 2 + 3 :=
sorry

end problem_A_problem_B_problem_D_l262_262335


namespace HCF_of_A_and_B_l262_262106

-- Definitions of numbers and their LCM
def A : ℕ := 210
def B : ℕ := 330
def L : ℕ := 2310

-- Condition using the relationship between LCM and HCF
theorem HCF_of_A_and_B : Nat.gcd A B = 30 :=
by
  have h : A * B = L * (Nat.gcd A B) := sorry
  have eq1 : A * B = 69300 := sorry
  have eq2 : L = 2310 := sorry
  have H := 69300 / 2310
  show Nat.gcd A B = 30 from sorry

end HCF_of_A_and_B_l262_262106


namespace mabel_visits_helen_l262_262051

-- Define the number of steps Mabel lives from Lake High school
def MabelSteps : ℕ := 4500

-- Define the number of steps Helen lives from the school
def HelenSteps : ℕ := (3 * MabelSteps) / 4

-- Define the total number of steps Mabel will walk to visit Helen
def TotalSteps : ℕ := MabelSteps + HelenSteps

-- Prove that the total number of steps Mabel walks to visit Helen is 7875
theorem mabel_visits_helen :
  TotalSteps = 7875 :=
sorry

end mabel_visits_helen_l262_262051


namespace common_tangents_of_circles_l262_262463

def circle1 : set (ℝ × ℝ) := {p | (p.1^2 + p.2^2 + 4 * p.1 - 4 * p.2 + 7 = 0)}
def circle2 : set (ℝ × ℝ) := {p | (p.1^2 + p.2^2 - 4 * p.1 - 10 * p.2 + 13 = 0)}

theorem common_tangents_of_circles :
  (set.count {l : set (ℝ × ℝ) | l.is_line ∧ ∃ p ∈ l, p ∈ circle1 ∧ p ∈ circle2}) = 3 :=
begin
  sorry
end

end common_tangents_of_circles_l262_262463


namespace min_n_for_factors_l262_262927

theorem min_n_for_factors (n : ℕ) : (∀ k : ℤ, (k ∈ {4, 8} → k ∣ (60 * n))) → n = 8 := 
by
  intro h
  -- prove the theorem here
  sorry

end min_n_for_factors_l262_262927


namespace determine_g_l262_262421

variable (g : ℕ → ℕ)

theorem determine_g (h : ∀ x, g (x + 1) = 2 * x + 3) : ∀ x, g x = 2 * x + 1 :=
by
  sorry

end determine_g_l262_262421


namespace expressions_equal_when_a_plus_b_plus_c_eq_1_l262_262111

theorem expressions_equal_when_a_plus_b_plus_c_eq_1
  (a b c : ℝ) (h : a + b + c = 1) :
  a + b * c = (a + b) * (a + c) :=
sorry

end expressions_equal_when_a_plus_b_plus_c_eq_1_l262_262111


namespace selection_methods_correct_l262_262494

-- Define the number of students in each year
def first_year_students : ℕ := 3
def second_year_students : ℕ := 5
def third_year_students : ℕ := 4

-- Define the total number of different selection methods
def total_selection_methods : ℕ := first_year_students + second_year_students + third_year_students

-- Lean statement to prove the question is equivalent to the answer
theorem selection_methods_correct :
  total_selection_methods = 12 := by
  sorry

end selection_methods_correct_l262_262494


namespace imaginary_part_of_conjugate_z_l262_262339

noncomputable def z : ℂ := (1 - complex.i) / (1 + complex.i)

theorem imaginary_part_of_conjugate_z : complex.im (complex.conj z) = 1 :=
sorry

end imaginary_part_of_conjugate_z_l262_262339


namespace f_at_neg_100_l262_262430

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : ∀ x : ℝ, f(x + 3) * f(x - 4) = -1
axiom f_condition2 : ∀ x : ℝ, 0 ≤ x ∧ x < 7 → f(x) = Real.logb 2 (9 - x)

theorem f_at_neg_100 : f(-100) = -1 / 2 := sorry

end f_at_neg_100_l262_262430


namespace sum_of_abs_coeffs_in_ellipse_eq_21486_l262_262987

noncomputable def parametric_ellipse (t : ℝ) : ℝ × ℝ :=
  ((3 * (Math.sin t - 2)) / (3 - Math.cos t), (4 * (Math.cos t - 6)) / (3 - Math.cos t))

theorem sum_of_abs_coeffs_in_ellipse_eq_21486 :
  ∃ A B C D E F : ℤ,
    (∀ (x y : ℝ),
      (x, y) = parametric_ellipse t →
      A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C)
    (Int.gcd (Int.natAbs D) (Int.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E +
    Int.natAbs F = 21486) := 
sorry

end sum_of_abs_coeffs_in_ellipse_eq_21486_l262_262987


namespace sum_of_common_divisors_l262_262605

theorem sum_of_common_divisors : 
  let d := [48, 144, -24, 180, 192]
  ∃ (a b c d : ℕ), 
  (a | 48) ∧ (a | 144) ∧ (a | -24) ∧ (a | 180) ∧ (a | 192) ∧
  (b | 48) ∧ (b | 144) ∧ (b | -24) ∧ (b | 180) ∧ (b | 192) ∧
  (c | 48) ∧ (c | 144) ∧ (c | -24) ∧ (c | 180) ∧ (c | 192) ∧
  (d | 48) ∧ (d | 144) ∧ (d | -24) ∧ (d | 180) ∧ (d | 192) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b + c + d = 12 :=
by
  sorry

end sum_of_common_divisors_l262_262605


namespace number_of_cows_is_six_l262_262928

variable (C H : Nat) -- C for cows and H for chickens

-- Number of legs is 12 more than twice the number of heads.
def cows_count_condition : Prop :=
  4 * C + 2 * H = 2 * (C + H) + 12

theorem number_of_cows_is_six (h : cows_count_condition C H) : C = 6 :=
sorry

end number_of_cows_is_six_l262_262928


namespace pauls_total_cost_is_252_l262_262789

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end pauls_total_cost_is_252_l262_262789


namespace problem_statement_l262_262298

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

theorem problem_statement :
  (∃ m : ℝ, ∀ x : ℝ, f x = g (x + m)) ∧
  (∃ k : ℝ, ∀ x : ℝ, f x = g (k * x - 1)) ∧
  (∀ x > (2 / 3 : ℝ), ¬Monotonic (λ x : ℝ, g x - f x)) ∧
  (∀ x ∈ (Set.Ioc 0 1), g x - f x > (1 / 6 : ℝ)) :=
begin
  sorry  -- Proof omitted
end

end problem_statement_l262_262298


namespace range_of_omega_l262_262348

theorem range_of_omega (ω : ℝ) (h : 0 < ω) :
  (∀ x y : ℝ, -π/5 ≤ x ∧ x < y ∧ y ≤ π/4 → x < y → sin (ω * x) < sin (ω * y)) →
  0 < ω ∧ ω ≤ 2 :=
by
  intro h1
  split
  { exact h }
  {
    have h2 : ω ≤ 2.5 := sorry
    have h3 : ω ≤ 2 := sorry
    exact lt_of_le_of_lt h3 h2
  }

end range_of_omega_l262_262348


namespace balloons_problem_l262_262147

theorem balloons_problem :
  ∃ (b y : ℕ), y = 3414 ∧ b + y = 8590 ∧ b - y = 1762 := 
by
  sorry

end balloons_problem_l262_262147


namespace range_of_f_l262_262250

noncomputable def f (x : ℝ) : ℝ := (x - 2)^2 - 1

theorem range_of_f : set.range (λ x, f x) ∩ set.Icc (-1 : ℝ) 8 = set.Icc (-1 : ℝ) 8 :=
by
  sorry

end range_of_f_l262_262250


namespace lemon_cost_is_correct_l262_262353

theorem lemon_cost_is_correct
  (servings : ℕ)
  (cost_per_serving : ℕ)
  (apple_pounds : ℕ)
  (cost_per_pound_apple : ℕ)
  (pie_crust_cost : ℕ)
  (butter_cost : ℕ)
  (total_servings_cost : ℕ)
  (total_ingredients_cost_without_lemon : ℕ)
  (lemon_cost : ℕ) :
  servings = 8 ∧
  cost_per_serving = 1 ∧
  apple_pounds = 2 ∧
  cost_per_pound_apple = 2 ∧
  pie_crust_cost = 2 ∧
  butter_cost = 1 ∧
  total_servings_cost = servings * cost_per_serving ∧
  total_ingredients_cost_without_lemon = apple_pounds * cost_per_pound_apple + pie_crust_cost + butter_cost ∧
  lemon_cost = total_servings_cost - total_ingredients_cost_without_lemon →
  lemon_cost = 0.5 :=
by
  sorry

end lemon_cost_is_correct_l262_262353


namespace burt_speed_uphill_l262_262549

-- Variables and conditions
variables (d : ℝ) (v_up v_down v_avg : ℝ)

-- Given conditions
axiom H1 : v_down = 42
axiom H2 : v_avg = 21

-- Definition of the total distance and average speed equation
noncomputable def t_up := d / v_up
noncomputable def t_down := d / v_down
noncomputable def t_total := t_up + t_down
noncomputable def v_avg_calc := 2 * d / t_total

-- The proof statement
theorem burt_speed_uphill : v_avg_calc = v_avg → v_up = 14 := 
by
  intros h,
  sorry

end burt_speed_uphill_l262_262549


namespace hexagon_angle_R_l262_262376

theorem hexagon_angle_R (F I G U R E : ℝ) 
  (h1 : F = I ∧ I = R ∧ R = E)
  (h2 : G + U = 180) 
  (sum_angles_hexagon : F + I + G + U + R + E = 720) : 
  R = 135 :=
by sorry

end hexagon_angle_R_l262_262376


namespace value_of_expression_l262_262938

theorem value_of_expression :
  (10^2 - 10) / 9 = 10 :=
by
  sorry

end value_of_expression_l262_262938


namespace store_profit_loss_l262_262212

theorem store_profit_loss :
  ∃ (x y : ℝ), (1 + 0.25) * x = 135 ∧ (1 - 0.25) * y = 135 ∧ (135 - x) + (135 - y) = -18 :=
by
  sorry

end store_profit_loss_l262_262212


namespace arrangements_with_AB_next_to_each_other_l262_262877

theorem arrangements_with_AB_next_to_each_other :
  ∀ (persons : Finset ℕ), (A B : ℕ) (hA : A ∈ persons) (hB : B ∈ persons),
  persons.card = 5 ∧ ∃ (i j : ℕ), i ≠ j ∧ (i = 2 ∨ i = 3) ∧ (j = (i + 1) % 5) → 
  ∃ (AB_arrangements : ℕ) (others_arrangements : ℕ), 
  AB_arrangements * others_arrangements = 24 :=
by
  sorry

end arrangements_with_AB_next_to_each_other_l262_262877


namespace sum_of_squares_equal_l262_262812

-- Given conditions
def num_players : ℕ := 10
def games_per_player : ℕ := num_players - 1

-- Player indices
def Player := Fin num_players

-- Wins and losses for each player
variable (wins losses : Player → ℕ)

-- Each player plays exactly 9 games
axiom games_played (i : Player) : wins i + losses i = games_per_player

-- The total number of wins equals the total number of losses
axiom total_equal : (∑ i, wins i) = (∑ i, losses i)

-- The goal to prove
theorem sum_of_squares_equal : (∑ i, (wins i)^2) = (∑ i, (losses i)^2) :=
sorry

end sum_of_squares_equal_l262_262812


namespace ellipse_equation_slope_of_line_max_area_of_triangle_l262_262647
-- Import necessary libraries

-- Part 1: Prove the equation of the ellipse
theorem ellipse_equation {a b : ℝ} (h : a > b ∧ b > 0)
  (A : ℝ × ℝ) (hA : A = (sqrt 3, sqrt 13 / 2))
  (F : ℝ × ℝ) (hF : F = (-2 * sqrt 3, 0))
  (heq : ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ A = (x,y)) :
  a^2 = 16 ∧ b^2 = 4 :=
begin
  sorry
end

-- Part 2: Prove the slope of the line l
theorem slope_of_line (P : ℝ × ℝ) (hP : P = (0, 1))
  (hmn : ∀ ⦃x1 x2 y1 y2 : ℝ⦄, (x1, y1) = M ∧ (x2,y2) = N → 
     (Mx, My) = M ∧ (Nx, Ny) = N → 
     Mx^2 / 16 + My^2 / 4 = 1 ∧ Nx^2 / 16 + Ny^2 / 4 = 1 →
     (∃ k : ℝ, y = k * x + 1 ∧ (2 * (0,1) - N) = M ∧ (0, 1) = P) :
  (k = ((sqrt 15) / 10) ∨ k = (- (sqrt 15) / 10)) :=
begin
  sorry
end

-- Part 3: Prove the maximum area of the triangle MON
theorem max_area_of_triangle (O : ℝ × ℝ) (hO : O = (0, 0))
  (H : | (O + M) + (O + N) | = 4)
  (hMn : ∃ (x y : ℝ), x^2 / 16 + y^2 / 4 = 1 ∧ M = (x,y) ∧ N = (-x, -y)) :
  (∃ A : ℝ, A = 4) :=
begin
  sorry
end

end ellipse_equation_slope_of_line_max_area_of_triangle_l262_262647


namespace common_ratio_of_geometric_series_l262_262835

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l262_262835


namespace probability_of_black_given_not_white_l262_262875

variable (total_balls white_balls black_balls red_balls : ℕ)
variable (ball_is_not_white : Prop)

theorem probability_of_black_given_not_white 
  (h1 : total_balls = 10)
  (h2 : white_balls = 5)
  (h3 : black_balls = 3)
  (h4 : red_balls = 2)
  (h5 : ball_is_not_white) :
  (3 : ℚ) / 5 = (black_balls : ℚ) / (total_balls - white_balls) :=
by
  simp only [h1, h2, h3, h4]
  sorry

end probability_of_black_given_not_white_l262_262875


namespace discard_some_triples_divisible_by_three_l262_262037

theorem discard_some_triples_divisible_by_three
  (a b c : Fin 7 → ℤ) :
  ∃ S : Finset (Fin 7), 0 < S.card ∧
    (∑ i in S, a i) % 3 = 0 ∧
    (∑ i in S, b i) % 3 = 0 ∧
    (∑ i in S, c i) % 3 = 0 :=
sorry

end discard_some_triples_divisible_by_three_l262_262037


namespace tan_double_angle_l262_262653

theorem tan_double_angle (θ : ℝ) 
  (h1 : sin (π - θ) = 1 / 2) 
  (h2 : π / 2 < θ ∧ θ < π) : tan (2 * θ) = -real.sqrt 3 :=
by
  sorry

end tan_double_angle_l262_262653


namespace total_revenue_correct_l262_262266

noncomputable def calculateRevenue : ℤ :=
  let samsung_start := 14
  let samsung_end := 10
  let samsung_damaged := 2
  let samsungs_sold := samsung_start - samsung_end - samsung_damaged

  let iphone_start := 8
  let iphone_end := 5
  let iphone_damaged := 1
  let iphones_sold := iphone_start - iphone_end - iphone_damaged

  let samsung_retail := 800
  let samsung_discount := 0.10
  let samsung_discounted_price := samsung_retail - (samsung_retail * samsung_discount)
  let samsung_tax_rate := 0.12
  let samsung_final_price := samsung_discounted_price + (samsung_discounted_price * samsung_tax_rate)

  let iphone_retail := 1000
  let iphone_discount := 0.15
  let iphone_discounted_price := iphone_retail - (iphone_retail * iphone_discount)
  let iphone_tax_rate := 0.10
  let iphone_final_price := iphone_discounted_price + (iphone_discounted_price * iphone_tax_rate)

  let total_revenue := (samsungs_sold * samsung_final_price) + (iphones_sold * iphone_final_price)
  total_revenue

theorem total_revenue_correct : calculateRevenue = 3482.80 :=
by
  -- use sorry to skip the proof
  sorry

end total_revenue_correct_l262_262266


namespace pauls_total_cost_is_252_l262_262788

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end pauls_total_cost_is_252_l262_262788


namespace brand_a_millet_percentage_l262_262961

theorem brand_a_millet_percentage :
  let Am := 40 / 100 in
  let As := 60 / 100 in
  let Bm := 65 / 100 in
  let Bs := 35 / 100 in
  let mix_sunflower := 50 / 100 in
  let mix_brand_a := 60 / 100 in
  let mix_brand_b := 40 / 100 in
  mix_brand_a * As + mix_brand_b * Bs = mix_sunflower → As = 60 / 100 → Am = 1 - As → Am = 40 / 100 :=
by
  sorry

end brand_a_millet_percentage_l262_262961


namespace analogical_reasoning_ineq_l262_262331

-- Formalization of the conditions and the theorem to be proved

def positive (a : ℕ → ℝ) (n : ℕ) := ∀ i, 1 ≤ i → i ≤ n → a i > 0

theorem analogical_reasoning_ineq {a : ℕ → ℝ} (hpos : positive a 4) (hsum : a 1 + a 2 + a 3 + a 4 = 1) : 
  (1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4) ≥ 16 := 
sorry

end analogical_reasoning_ineq_l262_262331


namespace surface_area_of_4cm_cube_after_corner_removal_l262_262610

noncomputable def surface_area_after_corner_removal (cube_side original_surface_length corner_cube_side : ℝ) : ℝ := 
  let num_faces : ℕ := 6
  let num_corners : ℕ := 8
  let surface_area_one_face := cube_side * cube_side
  let original_surface_area := num_faces * surface_area_one_face
  let corner_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let exposed_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let net_change_per_corner_cube := -corner_surface_area_one_face + exposed_surface_area_one_face
  let total_change := num_corners * net_change_per_corner_cube
  original_surface_area + total_change

theorem surface_area_of_4cm_cube_after_corner_removal : 
  ∀ (cube_side original_surface_length corner_cube_side : ℝ), 
  cube_side = 4 ∧ original_surface_length = 4 ∧ corner_cube_side = 2 →
  surface_area_after_corner_removal cube_side original_surface_length corner_cube_side = 96 :=
by
  intros cube_side original_surface_length corner_cube_side h
  rcases h with ⟨hs, ho, hc⟩
  rw [hs, ho, hc]
  sorry

end surface_area_of_4cm_cube_after_corner_removal_l262_262610


namespace complex_pow_is_complex_l262_262447

theorem complex_pow_is_complex (a b : ℝ) (n : ℕ) (h : 0 < n) : 
    (a + b * complex.I) ^ n ∈ ℂ :=
sorry

end complex_pow_is_complex_l262_262447


namespace tortoise_wins_by_12_5_meters_l262_262007

def race_distance : ℕ := 100
def quarter_distance : ℕ := 25

def hare_speed_dashing (v_T : ℕ) : ℕ := 8 * v_T
def hare_speed_walking (v_T : ℕ) : ℕ := 2 * v_T

def tortoise_time (v_T : ℕ) : ℕ := race_distance / v_T
def hare_dashing_time (v_T : ℕ) : ℕ := quarter_distance / hare_speed_dashing v_T
def hare_walking_time (v_T : ℕ) : ℕ := quarter_distance / hare_speed_walking v_T
def hare_napping_time (v_T : ℕ) : ℕ := 2 * hare_walking_time v_T
def hare_total_time (v_T : ℕ) : ℕ := 2 * hare_dashing_time v_T + hare_walking_time v_T + hare_napping_time v_T

theorem tortoise_wins_by_12_5_meters (v_T : ℕ) : tortoise_time v_T < hare_total_time v_T ∧ 
  100 - (100 * hare_total_time v_T / tortoise_time v_T) = 12.5 := 
by
  sorry

end tortoise_wins_by_12_5_meters_l262_262007


namespace expression_value_at_2_l262_262165

theorem expression_value_at_2 : (2^2 - 3 * 2 + 2) = 0 :=
by
  sorry

end expression_value_at_2_l262_262165


namespace distance_Q_to_EH_l262_262725

-- Square and point definitions
def Square (side_length : ℝ) : Type :=
{ E F G H : Point
  -- Side length condition
  (size : ∀ {p₁ p₂ : Point}, (p₁ = E ∨ p₁ = G ∨ p₁ = F ∨ p₁ = H) → (p₂ = E ∨ p₂ = G ∨ p₂ = F ∨ p₂ = H) → dist p₁ p₂ = side_length)

  -- Ordering of points
  (square_structure : ¬(E = F ∧ G = H)) }

noncomputable def example_square := Square.mk 6 sorry sorry

-- Points
def N := midpoint example_square.G example_square.H

-- Circle definitions centered at N and E
def circle_N : Circle :=
{ center := N
  radius := 3 }

def circle_E : Circle :=
{ center := example_square.E
  radius := 5 }

-- Intersection point Q and required distance calculation
theorem distance_Q_to_EH (Q : Point) (H : Point) (E : Point) (y_Q : ℝ)
  (H : H = ⟨0, 0⟩) (G : example_square.G = ⟨6, 0⟩) (F : example_square.F = ⟨6, 6⟩) (E : example_square.E = ⟨0, 6⟩)
  -- Conditions for Q as the intersection point
  (Q_intersect : on_circle Q circle_N ∧ on_circle Q circle_E) :
    dist_to(EH) Q = 6 - (20 - 2 * sqrt 46) / 12 :=
   sorry

end distance_Q_to_EH_l262_262725


namespace final_answer_l262_262243

noncomputable def product_eq_one : Prop := 
  (∏ k in finset.range 12, ∏ j in finset.range 10, 
   (complex.exp (2 * real.pi * complex.I * j / 11) - 
    complex.exp (2 * real.pi * complex.I * k / 13))) = 1

theorem final_answer : product_eq_one :=
sorry

end final_answer_l262_262243


namespace hyperbolaEquation_l262_262680

noncomputable def equationOfHyperbola (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbolaEquation :
  ∃ a b : ℝ, (π / 3 = real.arctan (b / a)) ∧
             (a ≠ 0 ∧ b ≠ 0) ∧ 
             ∃ f : (p : ℝ × ℝ), distance from f to (y = (b / a) * x) = √3 ∧ 
              a = 1 ∧ b = √3 ∧ equationOfHyperbola a b x y = x^2 - y^2 / 3 = 1 :=
  sorry

end hyperbolaEquation_l262_262680


namespace triangle_min_perimeter_l262_262126

theorem triangle_min_perimeter:
  ∃ x : ℤ, 27 < x ∧ x < 75 ∧ (24 + 51 + x) = 103 :=
begin
  sorry
end

end triangle_min_perimeter_l262_262126


namespace jake_last_10_shots_l262_262219

-- conditions
variable (total_shots_initially : ℕ) (shots_made_initially : ℕ) (percentage_initial : ℝ)
variable (total_shots_finally : ℕ) (shots_made_finally : ℕ) (percentage_final : ℝ)

axiom initial_conditions : shots_made_initially = percentage_initial * total_shots_initially
axiom final_conditions : shots_made_finally = percentage_final * total_shots_finally
axiom shots_difference : total_shots_finally - total_shots_initially = 10

-- prove that Jake made 7 out of the last 10 shots
theorem jake_last_10_shots : total_shots_initially = 30 → 
                             percentage_initial = 0.60 →
                             total_shots_finally = 40 → 
                             percentage_final = 0.62 →
                             shots_made_finally - shots_made_initially = 7 :=
by
  -- proofs to be filled in
  sorry

end jake_last_10_shots_l262_262219


namespace quadratic_complete_square_l262_262482

theorem quadratic_complete_square:
  ∃ (a b c : ℝ), (∀ (x : ℝ), 3 * x^2 + 9 * x - 81 = a * (x + b) * (x + b) + c) ∧ a + b + c = -83.25 :=
by {
  sorry
}

end quadratic_complete_square_l262_262482


namespace binom_n_2_l262_262516

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262516


namespace common_ratio_of_geometric_series_l262_262868

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l262_262868


namespace tangent_circle_length_l262_262565

/--
Two circles with centers A and B are given.
Circle with center A has radius 10 units.
Circle with center B has radius 3 units.
The circles are externally tangent to each other at point X.
There is a common external tangent to both circles at points J (on circle A) and K (on circle B).

Prove that the length of segment AB is 13 units.
-/
theorem tangent_circle_length 
  (A B X J K : Type)
  (rA rB : ℝ) 
  (hA : rA = 10)
  (hB : rB = 3)
  (tangent_point : A ≠ B ∧ ∀ pt, pt ≠ X → ∃ P Q : Type, circle P 10 ∧ circle Q 3)
  (common_tangent : ∃ (J : Type) (K : Type), tangent_point J K) :
  d A B = 13 := 
begin
  -- sorry is used to skip the proof
  sorry
end

end tangent_circle_length_l262_262565


namespace kimberly_loan_l262_262752

theorem kimberly_loan :
  ∃ (t : ℕ), (1.06 : ℝ)^t > 3 ∧ ∀ (t' : ℕ), t' < t → (1.06 : ℝ)^t' ≤ 3 :=
by
sorry

end kimberly_loan_l262_262752


namespace proof_CD_eq_2OF_l262_262135

noncomputable def circle_center : Type := sorry -- Assuming the type of circle's center
noncomputable def A : circle_center := sorry
noncomputable def B : circle_center := sorry
noncomputable def C : circle_center := sorry
noncomputable def D : circle_center := sorry
noncomputable def O : circle_center := sorry -- Center of the circle
noncomputable def F : circle_center := sorry -- Foot of perpendicular from O to line AB
noncomputable def radius (O : circle_center) : ℝ := sorry -- Radius of the circle

axiom points_on_circle : ∀ (P : circle_center), P = A ∨ P = B ∨ P = C ∨ P = D

axiom perpendicular_AC_BD : ∀ {A C B D : circle_center}, A ≠ C ∧ B ≠ D → (A, C) ⊥ (B, D)
axiom foot_perpendicular : ∀ {A B O F : circle_center}, F = foot O A B → 
  (O, F) ⊥ line A B

def sine (θ : ℝ) : ℝ := sorry -- Assume a sine function is defined

def angle (P Q R : circle_center) : ℝ := sorry -- Assume an angle function is defined

theorem proof_CD_eq_2OF (A B C D O F : circle_center)
  (points_on_circle: ∀ (P : circle_center), P = A ∨ P = B ∨ P = C ∨ P = D)
  (perpendicular_AC_BD : ∀ {A C B D : circle_center}, A ≠ C ∧ B ≠ D → (A, C) ⊥ (B, D))
  (foot_perpendicular: ∀ {A B O F : circle_center}, F = foot O A B → (O, F) ⊥ line A B)
  (R : ℝ := radius O)
  (angle_FOA_eq_angle_BDA: angle F O A = angle B D A)
  (angle_FAO_eq_angle_CAD: angle F A O = angle C A D) :
  (dist C D = 2 * dist O F) :=
sorry

end proof_CD_eq_2OF_l262_262135


namespace time_to_see_each_other_again_l262_262023

variable (t : ℝ) (t_frac : ℚ)
variable (kenny_speed jenny_speed : ℝ)
variable (kenny_initial jenny_initial : ℝ)
variable (building_side distance_between_paths : ℝ)

def kenny_position (t : ℝ) : ℝ := kenny_initial + kenny_speed * t
def jenny_position (t : ℝ) : ℝ := jenny_initial + jenny_speed * t

theorem time_to_see_each_other_again
  (kenny_speed_eq : kenny_speed = 4)
  (jenny_speed_eq : jenny_speed = 2)
  (kenny_initial_eq : kenny_initial = -50)
  (jenny_initial_eq : jenny_initial = -50)
  (building_side_eq : building_side = 100)
  (distance_between_paths_eq : distance_between_paths = 300)
  (t_gt_50 : t > 50)
  (t_frac_eq : t_frac = 50) :
  (t == t_frac) :=
  sorry

end time_to_see_each_other_again_l262_262023


namespace odd_function_expression_l262_262112

noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then -x + 1 else sorry

theorem odd_function_expression (x : ℝ) (h : f (-x) = -f x) (hx : x < 0) : f x = -x - 1 := by
  unfold f
  split_ifs
  · sorry
  · sorry

end odd_function_expression_l262_262112


namespace Roe_saved_15_per_month_aug_nov_l262_262089

-- Step 1: Define the given conditions
def savings_per_month_jan_jul : ℕ := 10
def months_jan_jul : ℕ := 7
def savings_dec : ℕ := 20
def total_savings_needed : ℕ := 150
def months_aug_nov : ℕ := 4

-- Step 2: Define the intermediary calculations based on the conditions
def total_saved_jan_jul := savings_per_month_jan_jul * months_jan_jul
def total_savings_aug_nov := total_savings_needed - total_saved_jan_jul - savings_dec

-- Step 3: Define what we need to prove
def savings_per_month_aug_nov : ℕ := total_savings_aug_nov / months_aug_nov

-- Step 4: State the proof goal
theorem Roe_saved_15_per_month_aug_nov :
  savings_per_month_aug_nov = 15 :=
by
  sorry

end Roe_saved_15_per_month_aug_nov_l262_262089


namespace smallest_m_value_l262_262763

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.toInt

noncomputable def g (x : ℝ) : ℝ := abs (3 * fractional_part x - 1.5)

noncomputable def equation_holds (m : ℕ) (x : ℝ) : Prop :=
  m * g (x * g x) = x

noncomputable def has_at_least_3000_solutions (m : ℕ) : Prop :=
  ∃ (S : set ℝ), set.card S ≥ 3000 ∧ (∀ x ∈ S, equation_holds m x)

theorem smallest_m_value : ∃ (m : ℕ), has_at_least_3000_solutions m ∧ (∀ k : ℕ, k < m → ¬ has_at_least_3000_solutions k) :=
begin
  use 35,
  split,
  sorry,  -- Proof that 35 has at least 3000 solutions
  sorry   -- Proof that no smaller number than 35 has
end

end smallest_m_value_l262_262763


namespace probability_3_closer_0_0_to_6_l262_262970

noncomputable def probability_closer_to_3_than_0 (a b c : ℝ) : ℝ :=
  if h₁ : a < b ∧ b < c then
    (c - ((a + b) / 2)) / (c - a)
  else 0

theorem probability_3_closer_0_0_to_6 : probability_closer_to_3_than_0 0 3 6 = 0.75 := by
  sorry

end probability_3_closer_0_0_to_6_l262_262970


namespace isosceles_triangle_largest_angle_l262_262718

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : A = B) (h2 : C = 50) (h3 : A + B + C = 180) : max A C = 80 :=
by 
  -- Define additional facts about the triangle, e.g., A = B = 50, and sum of angles = 180.
  have h4 : A = 50 := sorry
  rw [h4, h2] at h3
  -- Prove the final result using the given conditions.
  sorry

end isosceles_triangle_largest_angle_l262_262718


namespace coin_value_difference_l262_262087

theorem coin_value_difference (p n d : ℕ) (h : p + n + d = 3000) (hp : p ≥ 1) (hn : n ≥ 1) (hd : d ≥ 1) : 
  (p + 5 * n + 10 * d).max - (p + 5 * n + 10 * d).min = 26973 := 
sorry

end coin_value_difference_l262_262087


namespace arc_length_SQ_l262_262369

theorem arc_length_SQ (O S Q : Point) (r : ℝ)
  (h1 : angle S O Q = 45)
  (h2 : dist O S = 12)
  (h3 : ∀ P1 P2 P3 : Point, angle P1 P2 P3 = angle P3 P2 P1) :
  ∃ l, length_arc S Q O = 6 * pi :=
by {
  sorry
}

end arc_length_SQ_l262_262369


namespace area_ratio_eq_l262_262513

-- Define the parameters used in the problem
variables (t t1 r ρ : ℝ)

-- Define the conditions given in the problem
def area_triangle_ABC : ℝ := t
def area_triangle_A1B1C1 : ℝ := t1
def circumradius_ABC : ℝ := r
def inradius_A1B1C1 : ℝ := ρ

-- Problem statement: Prove the given equation
theorem area_ratio_eq : t / t1 = 2 * ρ / r :=
sorry

end area_ratio_eq_l262_262513


namespace total_ice_cream_sales_l262_262058

theorem total_ice_cream_sales :
  let monday := 10000
  let tuesday := 12000
  let wednesday := 2 * tuesday
  let thursday := 1.5 * wednesday
  let friday := 0.75 * thursday
  let saturday := friday
  let sunday := friday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 163000 :=
by
  let monday := 10000
  let tuesday := 12000
  let wednesday := 2 * tuesday
  let thursday := 1.5 * wednesday
  let friday := 0.75 * thursday
  let saturday := friday
  let sunday := friday
  calc
    monday + tuesday + wednesday + thursday + friday + saturday + sunday
      = 10000 + 12000 + 2 * 12000 + 1.5 * (2 * 12000) + 0.75 * (1.5 * (2 * 12000)) + 0.75 * (1.5 * (2 * 12000)) + 0.75 * (1.5 * (2 * 12000)) : by sorry
      = 163000 : by sorry

end total_ice_cream_sales_l262_262058


namespace range_of_t_l262_262630

theorem range_of_t {S : ℕ → ℝ} {a : ℕ → ℝ} (h₀ : a 1 = 1)
  (h₁ : ∀ n, 2 * S n = (n + 1) * a n)
  (h₂ : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) :
  { t : ℝ // 1 ≤ t ∧ t < 3 / 2 } :=
by
  sorry

end range_of_t_l262_262630


namespace three_digit_integers_mod_eq_l262_262355

theorem three_digit_integers_mod_eq :
  let S := { x : ℕ | (9745 * x + 625) % 17 = 2000 % 17 ∧ 100 ≤ x ∧ x ≤ 999 }
  in S.card = 53 :=
by
  sorry

end three_digit_integers_mod_eq_l262_262355


namespace toy_cost_before_discount_l262_262400

-- Defining the problem
variable (x : ℝ)

-- Defining the conditions
axiom buys_5_toys : ℝ := 5
axiom discount_rate : ℝ := 0.20
axiom amount_paid : ℝ := 12

-- Defining the proof problem
theorem toy_cost_before_discount :
  0.80 * (buys_5_toys * x) = amount_paid → x = 3 :=
by
  intro h
  sorry

end toy_cost_before_discount_l262_262400


namespace nonagon_diagonal_perpendicular_l262_262405

noncomputable def is_regular_nonagon (s : list (ℝ × ℝ)) : Prop :=
  s.length = 9 ∧ ∀ (i j : ℕ), i < 9 → j < 9 → dist (s.nth_le i sorry) (s.nth_le ((i + 1) % 9) sorry) = dist (s.nth_le j sorry) (s.nth_le ((j + 1) % 9) sorry)

noncomputable def intersects_at (s : list (ℝ × ℝ)) (a b c d : ℕ) (P : ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, 0 ≤ t1 ∧ t1 ≤ 1 ∧ 0 ≤ t2 ∧ t2 ≤ 1 ∧
    s.nth_le ((a + 4) % 9) sorry = (1 - t1) • s.nth_le ((a + 4) % 9) sorry + t1 • s.nth_le ((b + 4) % 9) sorry ∧
    s.nth_le ((c + 4) % 9) sorry = (1 - t2) • s.nth_le ((c + 4) % 9) sorry + t2 • s.nth_le ((d + 4) % 9) sorry ∧
    s.nth_le ((a + 4) % 9) sorry = P ∧ s.nth_le ((c + 4) % 9) sorry = P

noncomputable def perpendicular (s : list (ℝ × ℝ)) (P G F A : ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, (P.1 - G.1) * (F.1 - A.1) + (P.2 - G.2) * (F.2 - A.2) = 0

theorem nonagon_diagonal_perpendicular (s : list (ℝ × ℝ)) :
  is_regular_nonagon s ∧ intersects_at s 0 4 3 5 P → perpendicular s P s[6 % 9] s[5 % 9] s[0 % 9] :=
by
  sorry

end nonagon_diagonal_perpendicular_l262_262405


namespace eval_exponents_l262_262143

theorem eval_exponents : (2^3)^2 - 4^3 = 0 := by
  sorry

end eval_exponents_l262_262143


namespace angle_between_vectors_l262_262688

variables (a b : ℝ^3) (angle : ℝ)

def norm (v : ℝ^3) : ℝ := real.sqrt (v ⬝ v)

def angle_between (u v : ℝ^3) : ℝ := 
  real.arccos ((u ⬝ v) / (norm u * norm v))

theorem angle_between_vectors : 
  norm a = 1 → 
  norm b = 2 → 
  norm (2 • a + b) = 2 * real.sqrt 3 → 
  angle_between a b = real.pi / 3 :=
by
  sorry

end angle_between_vectors_l262_262688


namespace expected_value_dice_sum_l262_262923

theorem expected_value_dice_sum :
  ∀ (d1 d2 d3 : ℕ), 
    (1 ≤ d1 ∧ d1 ≤ 6) → 
    (1 ≤ d2 ∧ d2 ≤ 6) → 
    (1 ≤ d3 ∧ d3 ≤ 6) → 
    max d1 (max d2 d3) = 5 → 
    ∃ (a b : ℕ), 
    (a = 645) ∧ (b = 61) ∧ (a + b = 706) ∧ 
    (expected_value (sum_dice d1 d2 d3) = (645 / 61) : ℚ) :=
by
  sorry

end expected_value_dice_sum_l262_262923


namespace part_1_solution_set_part_2_a_values_range_l262_262678

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 2 * a|

theorem part_1_solution_set (x : ℝ) :
  ∃ S : set ℝ, S = {x | f x 1 > 2} ∧ S = (set.Ioo ⊥ (1/2)) ∪ (set.Ioo ((5 : ℝ) / 2) ⊤) :=
by
    sorry

theorem part_2_a_values_range :
  ∃ a_range : set ℝ, (∀ x : ℝ, f x a ≥ a ^ 2 - 3 * a - 3) ↔ a ∈ { y | y ∈ set.Icc (-1) (2 + real.sqrt (7)) } :=
by
    sorry

end part_1_solution_set_part_2_a_values_range_l262_262678


namespace shortest_wire_length_l262_262160

theorem shortest_wire_length
  (d1 d2 : ℝ)
  (r1 r2 : ℝ)
  (h1 : d1 = 8)
  (h2 : d2 = 20)
  (h_r1 : r1 = d1 / 2)
  (h_r2 : r2 = d2 / 2)
  (h_distance_between_centers : r1 + r2 = 14)
  (h_straight_section_length : 2 * (sqrt (14^2 - (r2 - r1)^2)) = 8 * sqrt 10)
  (h_arc_length_smaller : (2 * arc_length (4 / 14) * r1 = 4.10))
  (h_arc_length_larger : (2 * arc_length (10 / 14) * r2 = 11.75)) :
  shortest_distance := 8 * sqrt 10 + 15.85 :=
sorry

end shortest_wire_length_l262_262160


namespace classroom_volume_classroom_paint_area_l262_262354

-- Define the dimensions of the classroom
def length := 10
def width := 6
def height := 3.5
def doors_windows_area := 6

-- Prove the volume of the classroom.
theorem classroom_volume : length * width * height = 210 := by
  sorry

-- Prove the area that needs to be painted.
theorem classroom_paint_area :
  (length * width) + 2 * (length * height) + 2 * (width * height) - doors_windows_area = 166 := by
  sorry

end classroom_volume_classroom_paint_area_l262_262354


namespace athena_total_spent_l262_262734

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end athena_total_spent_l262_262734


namespace configuration_count_l262_262571

theorem configuration_count :
  (∃ (w h s : ℕ), 2 * (w + h + 2 * s) = 120 ∧ w < h ∧ s % 2 = 0) →
  ∃ n, n = 196 := 
sorry

end configuration_count_l262_262571


namespace part1_range_a_part2_range_a_l262_262311

-- Definitions of the propositions
def p (a : ℝ) := ∃ x : ℝ, x^2 + a * x + 2 = 0

def q (a : ℝ) := ∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - a < 0

-- Part 1: If p is true, find the range of values for a
theorem part1_range_a (a : ℝ) :
  p a → (a ≤ -2*Real.sqrt 2 ∨ a ≥ 2*Real.sqrt 2) := sorry

-- Part 2: If one of p or q is true and the other is false, find the range of values for a
theorem part2_range_a (a : ℝ) :
  (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a) →
  (a ≤ -2*Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2*Real.sqrt 2)) := sorry

end part1_range_a_part2_range_a_l262_262311


namespace seq_geometric_sum_bn_l262_262652

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * (a(1) + a(n)) / 2
def a1 := 1
def d := 2
def a (n : ℕ) : ℕ := 2 * n - 1
def b (n : ℕ) : ℕ := 1 / (a n * a (n + 1))

theorem seq_geometric (hs1 : S 1 a = 1) (hs2 : S 2 a = 3) (hs4 : S 4 a = 16)
  (hgeom : hs2^2 = hs1 * hs4) : 
  a n = 2n - 1 := 
sorry

theorem sum_bn (sn : ℕ → ℕ) (hbn : b n = 1 / (a n * a (n + 1))) : 
  (finset.range n).sum b = n / (2 * n + 1) := 
sorry

end seq_geometric_sum_bn_l262_262652


namespace part1_l262_262548

theorem part1 (a b : ℝ) : 3*(a - b)^2 - 6*(a - b)^2 + 2*(a - b)^2 = - (a - b)^2 :=
by
  sorry

end part1_l262_262548


namespace find_angle_l262_262689

open Real
open EuclideanSpace

variables {n : ℕ} -- Dimension of the vector space
variables (a b : EuclideanSpace ℝ (Fin n)) -- Vector variables
variables (α : ℝ) -- Angle variable

noncomputable def vector_lengths (a b : EuclideanSpace ℝ (Fin n)) (h1 : ∥a∥ = sqrt 2) (h2 : ∥b∥ = 2)
  (h3 : inner (a - b) a = 0) : α := acos (inner a b / (∥a∥ * ∥b∥))

theorem find_angle (a b : EuclideanSpace ℝ (Fin n)) (α : ℝ) (h1 : ∥a∥ = sqrt 2) (h2 : ∥b∥ = 2)
  (h3 : inner (a - b) a = 0) : α = π / 4 :=
by sorry

end find_angle_l262_262689


namespace angle_BOC_in_circle_quadrilateral_l262_262550

open EuclideanGeometry

theorem angle_BOC_in_circle_quadrilateral
  (A B C D O : Point)
  (h_inscribed : is_cyclic_quadrilateral A B C D)
  (h_BC_CD : distance B C = distance C D)
  (h_angle_BCA : ∠ B C A = 64)
  (h_angle_ACD : ∠ A C D = 70)
  (h_point_O : O ∈ line_segment A C)
  (h_angle_ADO : ∠ A D O = 32) :
  ∠ B O C = 58 :=
sorry

end angle_BOC_in_circle_quadrilateral_l262_262550


namespace max_packages_delivered_l262_262052

/-- Max's performance conditions and delivery problem -/
theorem max_packages_delivered
  (max_daily_capacity : ℕ) (num_days : ℕ)
  (days_max_performance : ℕ) (max_deliveries_days1 : ℕ)
  (days_half_performance : ℕ) (half_deliveries_days2 : ℕ)
  (days_fraction_performance : ℕ) (fraction_deliveries_days3 : ℕ)
  (last_two_days_fraction : ℕ) (fraction_last_two_days : ℕ):
  ∀ (remaining_capacity : ℕ), remaining_capacity = 
  max_daily_capacity * num_days - 
  (days_max_performance * max_deliveries_days1 + 
  days_half_performance * half_deliveries_days2 + 
  days_fraction_performance * fraction_deliveries_days3 * (1/7) + 
  last_two_days_fraction * fraction_last_two_days * (4/5)) := sorry

#eval max_packages_delivered 35 7 2 35 2 50 1 35 (2 * 28)

end max_packages_delivered_l262_262052


namespace range_of_a_fx1_neg_fx2_pos_l262_262347

noncomputable def f (x : ℝ) (a : ℝ) := x * (Real.log x - a * x)

theorem range_of_a (x1 x2 a : ℝ) (h1 : x1 < x2)
    (h2 : ∀ x, 0 < x → Real.log x + 1 - 2 * a * x = 0 → x ∈ {x1, x2})
    : 0 < a ∧ a < (1 / 2) :=
sorry

theorem fx1_neg (x1 a : ℝ) (h1 : ∀ x, 0 < x → Real.log x + 1 - 2 * a * x = 0 → x ∈ {x1})
    : f x1 a < 0 :=
sorry

theorem fx2_pos (x2 a : ℝ) (h1 : ∀ x, 0 < x → Real.log x + 1 - 2 * a * x = 0 → x ∈ {x2})
    : f x2 a > - (1 / 2) :=
sorry

end range_of_a_fx1_neg_fx2_pos_l262_262347


namespace intersection_length_l262_262726

theorem intersection_length (t : ℝ) :
  let x := (1 / 2) - (Real.sqrt 2 / 2) * t,
      y := (Real.sqrt 2 / 2) * t,
      C := x^2 + y^2 = x + y,
      l1 := x = (1 / 2) - (Real.sqrt 2 / 2) * t,
      l2 := y = (Real.sqrt 2 / 2) * t,
      l_intersections := ∃ t1 t2 : ℝ, 
                            ((1 / 2) - (Real.sqrt 2 / 2) * t1)^2 + ((Real.sqrt 2 / 2) * t1)^2 = ((1 / 2) - ((Real.sqrt 2 / 2) * t1)) + ((Real.sqrt 2 / 2) * t1) ∧
                            ((1 / 2) - (Real.sqrt 2 / 2) * t2)^2 + ((Real.sqrt 2 / 2) * t2)^2 = ((1 / 2) - ((Real.sqrt 2 / 2) * t2)) + ((Real.sqrt 2 / 2) * t2) 
  in
  ∃ t1 t2 : ℝ, l_intersections →
  (abs (t1 - t2) = Real.sqrt(3) / 2) := 
sorry

end intersection_length_l262_262726


namespace mabel_visits_helen_l262_262049

-- Define the number of steps Mabel lives from Lake High school
def MabelSteps : ℕ := 4500

-- Define the number of steps Helen lives from the school
def HelenSteps : ℕ := (3 * MabelSteps) / 4

-- Define the total number of steps Mabel will walk to visit Helen
def TotalSteps : ℕ := MabelSteps + HelenSteps

-- Prove that the total number of steps Mabel walks to visit Helen is 7875
theorem mabel_visits_helen :
  TotalSteps = 7875 :=
sorry

end mabel_visits_helen_l262_262049


namespace distance_to_place_is_96_l262_262203

-- Define the conditions
def row_speed := 10 -- kmph in still water
def current_speed := 2 -- kmph
def round_trip_time := 20 -- hours

-- Prove the distance D to the place is 96 kilometers
theorem distance_to_place_is_96
  (row_speed : ℝ := 10)
  (current_speed : ℝ := 2)
  (round_trip_time : ℝ := 20) :
  ∃ (D : ℝ), (let against_current_speed := row_speed - current_speed;
                   with_current_speed := row_speed + current_speed;
                   t1 := D / against_current_speed;
                   t2 := D / with_current_speed in
               t1 + t2 = round_trip_time ∧ D = 96) :=
by
  sorry

end distance_to_place_is_96_l262_262203


namespace minimum_value_of_f_l262_262675

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ -1 / Real.exp 1) ∧ (∃ x : ℝ, x > 0 ∧ f x = -1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_f_l262_262675


namespace exists_point_X_l262_262638

open EuclideanGeometry

-- Definitions for the geometric objects
variables {M N A B : Point} -- Points on the plane

-- Line MN
noncomputable def line_MN := line_through M N

-- Point X on line MN such that angle AXM = 2 * angle BXN
theorem exists_point_X (M N A B: Point) (hMN: M ≠ N) (hA: ¬collinear ({M, N, A}) ∧ ¬collinear ({M, N, B})) : 
  ∃ (X : Point), X ∈ line_MN ∧ ∠AXM = 2 * ∠BXN := 
by 
  sorry

end exists_point_X_l262_262638


namespace kernel_is_subgroup_l262_262409

variables {G H : Type*} [Group G] [Group H]
variable (ϕ : G →* H)

theorem kernel_is_subgroup : (ϕ.ker).subgroup G :=
by sorry

end kernel_is_subgroup_l262_262409


namespace digit_100_of_3_div_26_is_3_l262_262887

theorem digit_100_of_3_div_26_is_3 :
  let dec_repr := 0.1 + ((1 / 1) * (0.153846)) -- decimal representation of 3/26
  ∀ n, (n > 0) → 
    let seq := "153846" in 
      ((n - 1) % 6 + 1) = 3 
      → seq.get !((n - 1) % 6) = '3' → 
  (seq.get !((100 - 1) % 6) = '3') :=
begin
  -- The exact proof is not supplied here
  sorry
end

end digit_100_of_3_div_26_is_3_l262_262887


namespace binom_n_2_l262_262515

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262515


namespace jenna_profit_l262_262391

noncomputable def total_profit (cost_price sell_price rent : ℝ) (tax_rate : ℝ) (worker_count : ℕ) (worker_salary : ℝ) (widgets_sold : ℕ) : ℝ :=
  let salaries := worker_count * worker_salary in
  let fixed_costs := salaries + rent in
  let profit_per_widget := sell_price - cost_price in
  let total_sales_profit := widgets_sold * profit_per_widget in
  let profit_before_taxes := total_sales_profit - fixed_costs in
  let taxes := profit_before_taxes * tax_rate in
  profit_before_taxes - taxes

theorem jenna_profit :
  total_profit 3 8 10000 0.2 4 2500 5000 = 4000 :=
by
  sorry

end jenna_profit_l262_262391


namespace pencils_in_stock_at_end_of_week_l262_262069

def pencils_per_day : ℕ := 100
def days_per_week : ℕ := 5
def initial_pencils : ℕ := 80
def sold_pencils : ℕ := 350

theorem pencils_in_stock_at_end_of_week :
  (pencils_per_day * days_per_week + initial_pencils - sold_pencils) = 230 :=
by sorry  -- Proof will be filled in later

end pencils_in_stock_at_end_of_week_l262_262069


namespace not_possible_to_collect_all_pieces_in_one_sector_l262_262563

noncomputable def initial_sum : ℕ := 1 + 2 + 3 + 4 + 5 + 6

theorem not_possible_to_collect_all_pieces_in_one_sector
  (sectors : Finset ℕ) -- Sectors numbered 1 to 6
  (pieces : ℕ → ℕ) -- Function mapping sector numbers to count of pieces in that sector
  (h_pieces : ∀ i ∈ sectors, pieces i = 1) -- Initial condition: each sector contains exactly one piece
  : 
  (∃(moves : ℕ → Finset (ℕ × ℕ)), ∀ n, move ∈ moves n → 
    (∃ i j, (pieces i).swap j = pieces i + 1 ∧ (pieces j).swap i = pieces j + 1) 
  ) → ∀ k ∈ sectors, pieces k ≠ 6 := 
sorry

end not_possible_to_collect_all_pieces_in_one_sector_l262_262563


namespace sin_alpha_value_cos_expression_value_l262_262755

variables {α : ℝ} {a b : ℝ × ℝ}

def vector_a : ℝ × ℝ := (Real.sqrt 3 * Real.sin α, 1)
def vector_b : ℝ × ℝ := (2, 2 * Real.cos α)

def orthogonal (x y : ℝ × ℝ) : Prop :=
  x.1 * y.1 + x.2 * y.2 = 0

theorem sin_alpha_value (hα : 0 < α ∧ α < Real.pi)
  (h_orth : orthogonal vector_a vector_b) :
  Real.sin α = 1 / 2 :=
sorry

theorem cos_expression_value (hα : 0 < α ∧ α < Real.pi)
  (h_orth : orthogonal vector_a vector_b) :
  Real.cos (2 * α + 7 * Real.pi / 12) = Real.sqrt 2 / 2 :=
sorry

end sin_alpha_value_cos_expression_value_l262_262755


namespace general_term_formula_l262_262483

def sequence (n : Nat) : ℚ :=
  (-1 : ℚ)^n * (1 / 2)^n

theorem general_term_formula :
  ∀ (n : ℕ), sequence (n + 1) = 
    (-1 : ℚ)^(n + 1) * (1 / 2)^(n + 1) := by
  sorry

end general_term_formula_l262_262483


namespace expression_value_l262_262164

theorem expression_value : 
  ((
    ( (3 + 2)⁻¹ + 2 )⁻¹ + 2 )⁻¹ + 2 
  ) = 65 / 27 := 
by 
  -- Proof omitted
  sorry

end expression_value_l262_262164


namespace triangulation_traverse_one_stroke_l262_262570

theorem triangulation_traverse_one_stroke (n : ℕ) (h_n : n ≥ 3) (h_triangulation : non_intersecting_diagonals_form_triangulation n) :
  (∃ eulerian_cycle : eulerian_cycle_exists n, eulerian_cycle) ↔ (n % 3 = 0) :=
sorry

-- Define the condition of non-intersecting diagonals forming a triangulation.
def non_intersecting_diagonals_form_triangulation (n : ℕ) : Prop :=
  ∃ diagonals : set (fin n × fin n), 
    (∀ (d₁ d₂ : fin n × fin n), d₁ ∈ diagonals → d₂ ∈ diagonals → d₁ ≠ d₂ → ¬ intersects d₁ d₂) ∧ 
    (card diagonals = n - 3)

-- Define existence of an Eulerian cycle in a graph derived from the triangulation.
def eulerian_cycle_exists (n : ℕ) : Prop :=
  ∃ g : graph (fin n), 
    (∀ v ∈ g.vertices, even (g.degree v)) ∧
    (is_connected g)

end triangulation_traverse_one_stroke_l262_262570


namespace vans_capacity_l262_262088

-- Definitions based on the conditions
def num_students : ℕ := 22
def num_adults : ℕ := 2
def num_vans : ℕ := 3

-- The Lean statement (theorem to be proved)
theorem vans_capacity :
  (num_students + num_adults) / num_vans = 8 := 
by
  sorry

end vans_capacity_l262_262088


namespace company_pays_per_month_after_new_hires_l262_262955

theorem company_pays_per_month_after_new_hires :
  let initial_employees := 500 in
  let new_employees := 200 in
  let hourly_pay := 12 in
  let hours_per_day := 10 in
  let days_per_week := 5 in
  let weeks_per_month := 4 in
  let total_employees := initial_employees + new_employees in
  let daily_pay := hourly_pay * hours_per_day in
  let working_days_in_month := days_per_week * weeks_per_month in
  let monthly_pay_per_employee := daily_pay * working_days_in_month in
  let total_monthly_payment := total_employees * monthly_pay_per_employee in
  total_monthly_payment = 1680000 := 
by
  sorry

end company_pays_per_month_after_new_hires_l262_262955


namespace angle_E_in_quadrilateral_EFGH_l262_262378

theorem angle_E_in_quadrilateral_EFGH 
  (angle_E angle_F angle_G angle_H : ℝ) 
  (h1 : angle_E = 2 * angle_F)
  (h2 : angle_E = 3 * angle_G)
  (h3 : angle_E = 6 * angle_H)
  (sum_angles : angle_E + angle_F + angle_G + angle_H = 360) : 
  angle_E = 180 :=
by
  sorry

end angle_E_in_quadrilateral_EFGH_l262_262378


namespace least_possible_perimeter_l262_262119

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end least_possible_perimeter_l262_262119


namespace percentage_four_petals_l262_262402

def total_clovers : ℝ := 200
def percentage_three_petals : ℝ := 0.75
def percentage_two_petals : ℝ := 0.24
def earnings : ℝ := 554 -- cents

theorem percentage_four_petals :
  (total_clovers - (percentage_three_petals * total_clovers + percentage_two_petals * total_clovers)) / total_clovers * 100 = 1 := 
by sorry

end percentage_four_petals_l262_262402


namespace tenth_student_score_l262_262374

-- Define the conditions as Lean definitions
def is_int (n : ℤ) := n ∈ Int

def is_bounded (n : ℤ) := 0 ≤ n ∧ n ≤ 100

def top_10_scores_arith_seq (a d : ℤ) (n : ℤ) := a + n * d

def scores_sum_3rd_to_6th (a d : ℤ) := 
  (top_10_scores_arith_seq a d 7) + (top_10_scores_arith_seq a d 6) + 
  (top_10_scores_arith_seq a d 5) + (top_10_scores_arith_seq a d 4) = 354

def xiaoyue_score (a d m : ℤ) := top_10_scores_arith_seq a d (10 - m) = 96

-- Main theorem to prove
theorem tenth_student_score : 
  ∃ (a d : ℤ), is_int a ∧ is_bounded a ∧ is_int d ∧ is_bounded d ∧ 
  scores_sum_3rd_to_6th a d ∧ ∃ (m : ℤ), is_int m ∧ xiaoyue_score a d m ∧ (top_10_scores_arith_seq a d 10 = 61 ∨ top_10_scores_arith_seq a d 10 = 72) :=
begin
  sorry
end

end tenth_student_score_l262_262374


namespace geometric_series_common_ratio_l262_262849

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l262_262849


namespace find_parallel_line_l262_262286

-- Definition of the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Definition of the original line equation
def original_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Definition of the desired line equation
def desired_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement: defining the desired line based on the point and parallelism condition
theorem find_parallel_line (x y : ℝ) (hx : point.fst = 0) (hy : point.snd = 1) :
  ∃ m : ℝ, (2 * x + y + m = 0) ∧ (2 * 0 + 1 + m = 0) → desired_line x y :=
sorry

end find_parallel_line_l262_262286


namespace original_price_of_shirt_l262_262612

variables (S C P : ℝ)

def shirt_condition := S = C / 3
def pants_condition := S = P / 2
def total_paid := 0.90 * S + 0.95 * C + P = 900

theorem original_price_of_shirt :
  shirt_condition S C →
  pants_condition S P →
  total_paid S C P →
  S = 900 / 5.75 :=
by
  sorry

end original_price_of_shirt_l262_262612


namespace base8_subtraction_l262_262281

theorem base8_subtraction : (325 : Nat) - (237 : Nat) = 66 :=
by 
  sorry

end base8_subtraction_l262_262281


namespace perimeter_of_first_square_l262_262479

theorem perimeter_of_first_square (p1 p2 p3 : ℕ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24) :
  p1 = 40 := 
  sorry

end perimeter_of_first_square_l262_262479


namespace find_parallel_line_l262_262284

variables {x y : ℝ}

def line1 (x y : ℝ) : Prop := 2 * x + y - 3 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def point (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem find_parallel_line (hx : point 0 1) : line2 0 1 :=
by
  dsimp [line2, point] at *,
  sorry

end find_parallel_line_l262_262284


namespace binom_n_2_l262_262523

theorem binom_n_2 (n : ℕ) (h : 1 < n) : (nat.choose n 2) = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262523


namespace function_symmetry_property_l262_262677

noncomputable def f (x : ℝ) : ℝ :=
  x ^ 2

def symmetry_property := 
  ∀ (x : ℝ), (-1 < x ∧ x ≤ 1) →
    (¬ (f (-x) = f x) ∧ ¬ (f (-x) = -f x))

theorem function_symmetry_property :
  symmetry_property :=
by
  sorry

end function_symmetry_property_l262_262677


namespace pauls_total_cost_is_252_l262_262790

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end pauls_total_cost_is_252_l262_262790


namespace common_ratio_of_geometric_series_l262_262857

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l262_262857


namespace initial_amount_l262_262777

-- Definitions of the conditions
def spent_on_sweets : ℝ := 1.05
def given_to_friends : ℝ := 1.00 + 1.00
def amount_left : ℝ := 17.05

-- The main statement: Prove that the initial amount equals $20.10
theorem initial_amount : ∃ (initial_money : ℝ), initial_money = amount_left + spent_on_sweets + given_to_friends :=
by
  have initial_money := amount_left + spent_on_sweets + given_to_friends
  use initial_money
  simp; linarith; sorry

end initial_amount_l262_262777


namespace sophomores_stratified_sampling_l262_262560

theorem sophomores_stratified_sampling 
  (total_students freshmen sophomores seniors selected_total : ℕ) 
  (H1 : total_students = 2800) 
  (H2 : freshmen = 970) 
  (H3 : sophomores = 930) 
  (H4 : seniors = 900) 
  (H_selected_total : selected_total = 280) : 
  (sophomores / total_students) * selected_total = 93 :=
by sorry

end sophomores_stratified_sampling_l262_262560


namespace f_neg_3_eq_2_l262_262708

noncomputable def f : ℤ → ℤ
| x := if x >= 0 then x + 1 else f (x + 2)

theorem f_neg_3_eq_2 : f (-3) = 2 :=
by sorry

end f_neg_3_eq_2_l262_262708


namespace remainder_7n_mod_4_l262_262908

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end remainder_7n_mod_4_l262_262908


namespace krishan_nandan_investment_ratio_l262_262403

theorem krishan_nandan_investment_ratio
    (X t : ℝ) (k : ℝ)
    (h1 : X * t = 6000)
    (h2 : X * t + k * X * 2 * t = 78000) :
    k = 6 := by
  sorry

end krishan_nandan_investment_ratio_l262_262403


namespace smallest_N_l262_262780

def leftmost_digit_1 (n : ℕ) : Prop :=
  n % 10^(n.digits 10).length = 1

def condition (N : ℕ) : Prop := 
  (N > 2017) ∧ ((∃ c : ℝ, c = (N / 2.5)) ∧ (c / N = 0.4))

theorem smallest_N : ∃ N, condition N ∧ N = 1481480 := by
  sorry

end smallest_N_l262_262780


namespace total_watermelons_l262_262021

def watermelons_grown_by_jason : ℕ := 37
def watermelons_grown_by_sandy : ℕ := 11

theorem total_watermelons : watermelons_grown_by_jason + watermelons_grown_by_sandy = 48 := by
  sorry

end total_watermelons_l262_262021


namespace speed_of_man_rowing_upstream_l262_262576

-- Variables declaration
variable (V_m V_downstream V_upstream V_s : ℝ)

-- Given conditions
axiom h1 : V_m = 24
axiom h2 : V_downstream = 28

-- Proof statement
theorem speed_of_man_rowing_upstream : V_upstream = V_m - (V_downstream - V_m) ↔ V_upstream = 20 := 
by
  -- Define the speed of the stream
  let V_s := V_downstream - V_m
  have hV_s : V_s = 4 := by
    rw [h1, h2]
    simp [V_s]
    -- speed of the stream calculation
  -- State the upstream speed 
  have : V_upstream = V_m - V_s := by
    simp [V_upstream, V_m, V_s]
  -- Conclude the theorem
  rw [h1, hV_s]
  simp [V_upstream]
  sorry  -- Proof steps to finalize omitted

end speed_of_man_rowing_upstream_l262_262576


namespace smallest_m_value_l262_262764

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.toInt

noncomputable def g (x : ℝ) : ℝ := abs (3 * fractional_part x - 1.5)

noncomputable def equation_holds (m : ℕ) (x : ℝ) : Prop :=
  m * g (x * g x) = x

noncomputable def has_at_least_3000_solutions (m : ℕ) : Prop :=
  ∃ (S : set ℝ), set.card S ≥ 3000 ∧ (∀ x ∈ S, equation_holds m x)

theorem smallest_m_value : ∃ (m : ℕ), has_at_least_3000_solutions m ∧ (∀ k : ℕ, k < m → ¬ has_at_least_3000_solutions k) :=
begin
  use 35,
  split,
  sorry,  -- Proof that 35 has at least 3000 solutions
  sorry   -- Proof that no smaller number than 35 has
end

end smallest_m_value_l262_262764


namespace sin_theta_of_arithmetic_progression_l262_262414

theorem sin_theta_of_arithmetic_progression (θ : ℝ) (h1 : cos θ, cos (2 * θ), and cos (3 * θ) form an arithmetic progression) (h2 : 0 < θ ∧ θ < π / 2) : sin θ = sqrt 3 / 2 :=
sorry

end sin_theta_of_arithmetic_progression_l262_262414


namespace horner_method_v5_value_l262_262509

def polynomial (x : ℤ) : ℤ := 2 * x^7 + x^6 - 3 * x^5 + 4 * x^3 - 8 * x^2 - 5 * x + 6

def v0 := 2
def v1 (x : ℤ) := v0 * x + 1
def v2 (x : ℤ) := v1 x * x - 3
def v3 (x : ℤ) := v2 x * x
def v4 (x : ℤ) := v3 x * x + 4

theorem horner_method_v5_value (x : ℤ) : let v5 := v4 x * x + ?m in
  v5 = v4 x * x - 8 :=
by
  sorry

end horner_method_v5_value_l262_262509


namespace sum_of_possible_k_sum_of_common_roots_k_l262_262899

theorem sum_of_possible_k
  (k : ℝ)
  (h1 : ∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) :
  k = 5 ∨ k = 9 :=
by sorry

theorem sum_of_common_roots_k :
  ∑ k in ({5, 9} : finset ℝ), k = 14 :=
by norm_num

end sum_of_possible_k_sum_of_common_roots_k_l262_262899


namespace determine_solutions_l262_262756

noncomputable def functional_eq_solutions (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + α * f x * y

theorem determine_solutions (f : ℝ → ℝ) (α : ℝ) :
  (α ≠ 0 → (f = (λ x, 0)) ∨ (α = 4 ∧ f = (λ x, x^2))) ∧
  (α = 0 → (∃ c : ℝ, f = (λ x, c)) ∨ f = (λ x, -x^2)) :=
by
  sorry

end determine_solutions_l262_262756


namespace radius_of_inscribed_circle_correct_l262_262980

noncomputable def radius_of_inscribed_circle 
  (A B C H : Type) 
  (AB CH : ℝ) 
  (AH_to_HB : Ratio) 
  (HA_eq_two_HB : AH_to_HB = Ratio.mk 2 1)
  (H_on_AB : true) 
  (center_on_CH : true) 
  (base_eq_sqrt3_over2 : AB = (Real.sqrt 3) / 2)
  (altitude_eq_sqrt6_over3 : CH = (Real.sqrt 6) / 3)
  : ℝ :=
  let BH := AB / 3
  let BC := Real.sqrt ((CH^2) + (BH^2))
  let OH := CH / 4
  let r := OH
  in r

theorem radius_of_inscribed_circle_correct
  (A B C H : Type) 
  (AB CH : ℝ) 
  (AH_to_HB : Ratio) 
  (HA_eq_two_HB : AH_to_HB = Ratio.mk 2 1)
  (H_on_AB : true) 
  (center_on_CH : true) 
  (base_eq_sqrt3_over2 : AB = (Real.sqrt 3) / 2)
  (altitude_eq_sqrt6_over3 : CH = (Real.sqrt 6) / 3)
  : radius_of_inscribed_circle A B C H AB CH AH_to_HB HA_eq_two_HB H_on_AB center_on_CH base_eq_sqrt3_over2 altitude_eq_sqrt6_over3 = (Real.sqrt 6) / 12 := by
  -- Proof is skipped
  sorry

end radius_of_inscribed_circle_correct_l262_262980


namespace combination_n_2_l262_262522

theorem combination_n_2 (n : ℕ) (h : n > 0) : 
  nat.choose n 2 = n * (n - 1) / 2 :=
sorry

end combination_n_2_l262_262522


namespace number_of_valid_five_digit_numbers_l262_262161

open Finset

def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

def countFiveDigitNumbersDivisibleByFive : ℕ :=
  let numbers := (digits.powerset.filter (λ s => s.card = 5)).filter (λ s => (s.max' sorry sorry = 0 ∨ s.max' sorry sorry = 5)) 
  5 * 4 * 3 * 2 -- Case when last digit is 0
  + 4 * 4 * 3 * 2 * 1 -- Case when last digit is 5

theorem number_of_valid_five_digit_numbers : countFiveDigitNumbersDivisibleByFive = 216 := 
sorry

end number_of_valid_five_digit_numbers_l262_262161


namespace fill_cistern_7_2_hours_l262_262949

theorem fill_cistern_7_2_hours :
  let R_fill := 1 / 4
  let R_empty := 1 / 9
  R_fill - R_empty = 5 / 36 →
  1 / (R_fill - R_empty) = 7.2 := 
by
  intros
  sorry

end fill_cistern_7_2_hours_l262_262949


namespace pete_walked_3000_miles_l262_262445

theorem pete_walked_3000_miles (flip_count last_day_steps total_steps_per_flip steps_per_mile : ℕ) (h1 : total_steps_per_flip = 80000) (h2 : flip_count = 55) (h3 : last_day_steps = 30000) (h4 : steps_per_mile = 1600) : 
    (nat.floor ((flip_count * total_steps_per_flip + last_day_steps) / steps_per_mile : ℝ)) = 3000 := 
by 
  -- Transformations and calculations would go here
  sorry

end pete_walked_3000_miles_l262_262445


namespace inner_prod_sum_real_inner_prod_modulus_l262_262297

open Complex

-- Define the given mathematical expressions
noncomputable def pair (α β : ℂ) : ℝ := (1 / 4) * (norm (α + β) ^ 2 - norm (α - β) ^ 2)

noncomputable def inner_prod (α β : ℂ) : ℂ := pair α β + Complex.I * pair α (Complex.I * β)

-- Prove the given mathematical statements

-- 1. Prove that ⟨α, β⟩ + ⟨β, α⟩ is a real number
theorem inner_prod_sum_real (α β : ℂ) : (inner_prod α β + inner_prod β α).im = 0 := sorry

-- 2. Prove that |⟨α, β⟩| = |α| * |β|
theorem inner_prod_modulus (α β : ℂ) : Complex.abs (inner_prod α β) = Complex.abs α * Complex.abs β := sorry

end inner_prod_sum_real_inner_prod_modulus_l262_262297


namespace part1_part2_l262_262329

-- Definition of sides opposite to angles and given conditions
variables {A B C a b c : ℝ}
variables (h_side_eq : a = 2 * b * Real.cos B) (h_diff : b ≠ c) 
variables (h_triangle : a^2 + c^2 = b^2 + 2 * a * c * Real.sin C)

-- Part 1: Prove that A = 2B
theorem part1 : h_side_eq ∧ h_diff → A = 2 * B :=
begin
  sorry
end

-- Part 2: If additional condition a^2 + c^2 = b^2 + 2ac sin C is given, find the value of A
theorem part2 : h_side_eq ∧ h_diff ∧ h_triangle → A = π / 3 :=
begin
  sorry
end

end part1_part2_l262_262329


namespace common_ratio_of_geometric_series_l262_262853

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l262_262853


namespace volunteers_arrangement_l262_262495

theorem volunteers_arrangement : 
  ∃ (n : ℕ), 
    let saturday_arrangements := Nat.choose 7 3 in
    let sunday_arrangements := Nat.choose 4 3 in
    n = saturday_arrangements * sunday_arrangements ∧ n = 140 :=
begin
  sorry
end

end volunteers_arrangement_l262_262495


namespace projections_lie_on_circle_l262_262446

noncomputable def projection_points_on_circle (A B K K' : Point)
  (tangents : Fin 4 (Line) )
  (proj_A : Fin 4 (Point))
  (proj_B : Fin 4 (Point)) :=
forall i, is_projection_on_tangent (A B) (K K') tangents[i] proj_A[i] proj_B[i]
  ∧ circle (find_center (A B K K' proj_A proj_B)) (find_radius (A B K K' proj_A proj_B))

/- Theorem to prove:
Given two intersecting circles with centers A and B intersecting at points K and K', and the projections of A and B onto the four tangents at K and K',
show that these projections lie on a circle.
-/
theorem projections_lie_on_circle (A B K K' : Point)
  (tangents : Fin 4 (Line))
  (proj_A : Fin 4 (Point)) 
  (proj_B : Fin 4 (Point)) 
  (hT: projection_points_on_circle A B K K' tangents proj_A proj_B): 
  ∀ (i : Fin 4), is_on_circle (find_center (A B K K' proj_A proj_B)) (find_radius (A B K K' proj_A proj_B)) (proj_A[i]) 
    ∧ is_on_circle (find_center (A B K K' proj_A proj_B)) (find_radius (A B K K' proj_A proj_B)) (proj_B[i]) :=
sorry

end projections_lie_on_circle_l262_262446


namespace least_possible_perimeter_l262_262121

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end least_possible_perimeter_l262_262121


namespace compute_distance_sum_l262_262753

-- Define the conditions
variables {T : ℝ} [ht : fact (0 < T)]
variables (λ : ℝ) (y : ℝ)

-- Definitions based on conditions
def ZO := 2 * T
def ZR := T
def B := (λ, 0)
def E := (2 * λ, T)
def O' := (2 * T, y)

-- Distance conditions
def BR := real.sqrt (λ^2 + T^2)
def BE := real.sqrt (λ^2 + T^2)
def EO' := real.sqrt ((2 * T - 2 * λ)^2 + (y - T)^2)

-- Slope condition based on 90-degree angle θ
def BE_slope := T / λ
def EO'_slope := (y - T) / (2 * T - 2 * λ)

-- Proof statement
theorem compute_distance_sum :
  BR = BE ∧ BE = EO' ∧ BE_slope * EO'_slope = -1 → 2 * (ZO + real.sqrt ((2 * T - 2 * T)^2 + (T - y)^2) + T) = 7 * T :=
sorry

end compute_distance_sum_l262_262753


namespace range_of_eccentricity_for_8_points_l262_262179

-- Definitions of the ellipse and eccentricity
def ellipse (a b : ℝ) (h : a > b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Theorem statement regarding the range of eccentricity
theorem range_of_eccentricity_for_8_points 
  (a b : ℝ) (h : a > b > 0) :
  ∃ (e : ℝ), 
  (eccentricity a b = e ∧ 
   ((Real.sqrt 2 / 2 < e ∧ e < Real.sqrt 3 - 1) ∨ 
    (Real.sqrt 3 - 1 < e ∧ e < 1))) :=
sorry

end range_of_eccentricity_for_8_points_l262_262179


namespace probability_sum_eq_16777222_l262_262153

noncomputable def calculate_probability : ℚ := sorry

theorem probability_sum_eq_16777222 :
  let p := 6
  let q := 16777216
  p + q = 16777222 := by
  rw [calculate_probability]
  sorry

end probability_sum_eq_16777222_l262_262153


namespace q_poly_decreasing_order_l262_262810

variable (x : ℝ)

def q (x : ℝ) := -2 * x^6 + 5 * x^4 + 35 * x^3 + 28 * x^2 + 6 * x + 3

theorem q_poly_decreasing_order :
  q(x) + (2 * x^6 + 5 * x^4 + 12 * x^2) = 10 * x^4 + 35 * x^3 + 40 * x^2 + 6 * x + 3 :=
by
  unfold q
  sorry

end q_poly_decreasing_order_l262_262810


namespace example_l262_262916

theorem example : (-3)^2 = 9 :=
by
  sorry

end example_l262_262916


namespace areas_are_equal_l262_262491

open EuclideanGeometry

structure Quadrilateral (V : Type*) :=
(A B C D : V)

variables {V : Type*} [inner_product_space ℝ V] [plane V]

def translate (p : V) (v : V) : V := p + v

def area (Q : Quadrilateral V) : ℝ :=
sorry -- Define the area calculation method

theorem areas_are_equal (ABCD A'B'C'D' : Quadrilateral V)
  (u v : V) 
  (hA' : A' = translate A u) 
  (hC' : C' = translate C u)
  (hB' : B' = translate B v)
  (hD' : D' = translate D v)
  (non_intersecting_ABCD : ¬ (self_intersecting ABCD))
  (non_intersecting_A'B'C'D' : ¬ (self_intersecting A'B'C'D')) :
  area ABCD = area A'B'C'D' :=
sorry

end areas_are_equal_l262_262491


namespace bus_trip_distance_l262_262188

variable (D : ℝ) (S : ℝ := 50)

theorem bus_trip_distance :
  (D / (S + 5) = D / S - 1) → D = 550 := by
  sorry

end bus_trip_distance_l262_262188


namespace mabel_total_walk_l262_262045

theorem mabel_total_walk (steps_mabel : ℝ) (ratio_helen : ℝ) (steps_helen : ℝ) (total_steps : ℝ) :
  steps_mabel = 4500 →
  ratio_helen = 3 / 4 →
  steps_helen = ratio_helen * steps_mabel →
  total_steps = steps_mabel + steps_helen →
  total_steps = 7875 := 
by
  intros h_steps_mabel h_ratio_helen h_steps_helen h_total_steps
  rw [h_steps_mabel, h_ratio_helen] at h_steps_helen
  rw [h_steps_helen, h_steps_mabel] at h_total_steps
  rw h_total_steps
  rw [h_steps_mabel, h_ratio_helen]
  linarith

end mabel_total_walk_l262_262045


namespace sufficient_not_necessary_l262_262553

-- Define set A and set B
def setA (x : ℝ) := x > 5
def setB (x : ℝ) := x > 3

-- Statement:
theorem sufficient_not_necessary (x : ℝ) : setA x → setB x :=
by
  intro h
  exact sorry

end sufficient_not_necessary_l262_262553


namespace candy_totals_l262_262802

-- Definitions of the conditions
def sandra_bags := 2
def sandra_pieces_per_bag := 6

def roger_bags1 := 11
def roger_bags2 := 3

def emily_bags1 := 4
def emily_bags2 := 7
def emily_bags3 := 5

-- Definitions of total pieces of candy
def sandra_total_candy := sandra_bags * sandra_pieces_per_bag
def roger_total_candy := roger_bags1 + roger_bags2
def emily_total_candy := emily_bags1 + emily_bags2 + emily_bags3

-- The proof statement
theorem candy_totals :
  sandra_total_candy = 12 ∧ roger_total_candy = 14 ∧ emily_total_candy = 16 :=
by
  -- Here we would provide the proof but we'll use sorry to skip it
  sorry

end candy_totals_l262_262802


namespace decimal_33_to_quaternary_l262_262264

def decimal_to_quaternary (n : ℕ) : list ℕ := 
  let rec to_quat (n : ℕ) (acc : list ℕ) : list ℕ :=
    if n = 0 then acc else to_quat (n / 4) ((n % 4) :: acc)
  to_quat n []

theorem decimal_33_to_quaternary : decimal_to_quaternary 33 = [2, 0, 1] :=
by 
  apply List.eq_cons_of_eq cons_singular_eq <|-- Cons 1
  apply List.eq_cons_of_eq cons_singular_eq <|-- Cons 0
  apply List.eq_cons_of_eq cons_nil_eq <|-- Cons 2 
  sorry

end decimal_33_to_quaternary_l262_262264


namespace north_village_conscript_count_l262_262064

theorem north_village_conscript_count 
    (people_north : ℕ) 
    (people_west : ℕ) 
    (people_south : ℕ) 
    (total_conscript : ℕ) 
    (H1 : people_north = 8100) 
    (H2 : people_west = 7488) 
    (H3 : people_south = 6912) 
    (H4 : total_conscript = 300) : 
    (people_north : ℕ) := 108 :=
begin
    sorry
end

end north_village_conscript_count_l262_262064


namespace find_a_l262_262312

-- Define the conditions
def given_complex_number (a : ℝ) : ℂ := (a * complex.I) / (1 - 2 * complex.I)
def magnitude_condition (a : ℝ) : Prop := complex.abs (given_complex_number a) = real.sqrt 5

-- Main theorem to prove
theorem find_a (a : ℝ) (h : a < 0) (h_mag : magnitude_condition a) : a = -5 :=
sorry

end find_a_l262_262312


namespace covered_area_of_strips_l262_262293

/-- There are five rectangular paper strips, each of length 12 and width 1.
Each strip overlaps with two others perpendicularly at two different points.
Each overlapping section is exactly 2 units in length equally shared.
Prove that the total area covered by the strips on the table is 40 units. -/
theorem covered_area_of_strips :
  ∃ (n : ℕ) (l w : ℝ) (overlap_area : ℝ), 
  n = 5 ∧ l = 12 ∧ w = 1 ∧ overlap_area = 2 → 
  (total_area := n * (l * w)) - (overlap_sections := (n * (n - 1) / 2) * overlap_area) = 40 :=
by
  sorry

end covered_area_of_strips_l262_262293


namespace clock_time_after_307_58_59_l262_262741

def twelve_hour_clock_time (hours mins secs : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_hours := (hours + 3) % 12
  let total_mins := (mins + 0) % 60
  let total_secs := (secs + 0) % 60
  (total_hours, total_mins, total_secs)

def time_after (left_hours left_mins left_secs : ℕ) : ℕ × ℕ × ℕ :=
  let (hrs, mins, secs) := twelve_hour_clock_time (left_hours % 12) (left_mins % 60) (left_secs % 60)
  let extra_min_carry := (left_secs / 60) + (mins + (left_mins % 60)) / 60
  let extra_hr_carry := ((hrs + ((left_hours % 12)) / 12) + 
                        (extra_min_carry + (mins + (left_mins % 60)) / 60) / 60) % 12
  let final_secs := (left_secs + 0) % 60
  let final_mins := (left_mins + extra_min_carry) % 60
  let final_hrs := (hrs + extra_hr_carry) % 12
  (final_hrs, final_mins, final_secs)

theorem clock_time_after_307_58_59 : let (x, y, z) := time_after 307 58 59 in x + y + z = 127 :=
by
  sorry

end clock_time_after_307_58_59_l262_262741


namespace remainder_7n_mod_4_l262_262909

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l262_262909


namespace breadth_increase_25_percent_l262_262117

variable (L B : ℝ) 

-- Conditions
def original_area := L * B
def increased_length := 1.10 * L
def increased_area := 1.375 * (original_area L B)

-- The breadth increase percentage (to be proven as 25)
def percentage_increase_breadth (p : ℝ) := 
  increased_area L B = increased_length L * (B * (1 + p/100))

-- The statement to be proven
theorem breadth_increase_25_percent : 
  percentage_increase_breadth L B 25 := 
sorry

end breadth_increase_25_percent_l262_262117


namespace locus_of_lines_l262_262803

-- Define the given data and conditions
variables {A : Point} {n : ℕ} (a b c : Fin n → ℝ) -- Coordinates of unit vectors parallel to given lines
def unit_vector (x y z : ℝ) := x^2 + y^2 + z^2 = 1 -- Condition for being a unit vector

-- The mathematical condition: sum of the absolute values of dot products equal to a constant
def sum_cosines_condition (x y z : ℝ) (k : ℝ) : Prop :=
  (finset.univ.sum (λ i, abs (a i * x + b i * y + c i * z))) = k

-- The set of vectors satisfying this condition form circles or segments on the unit sphere
theorem locus_of_lines (x y z : ℝ) (k : ℝ) :
  sum_cosines_condition a b c x y z k → unit_vector x y z :=
sorry  -- proof goes here

end locus_of_lines_l262_262803


namespace divide_PRIME_subunits_l262_262065

-- Definition of the problem
def number_of_ways_to_divide_subunits (total_members subunit1 subunit2 subunit3 : ℕ) : ℕ :=
  if total_members != 7 ∨ (subunit1 != 2 ∧ subunit1 != 3) ∨
     (subunit2 != 2 ∧ subunit2 != 3) ∨ (subunit3 != 2 ∧ subunit3 != 3) ∨
     subunit1 + subunit2 + subunit3 != total_members
  then 0
  else
    let ways_big_subunit := Nat.choose total_members subunit1 in
    let remaining_members := total_members - subunit1 in
    let ways_small_subunit := Nat.choose remaining_members subunit2 in
    let last_subunit := remaining_members - subunit2 in
    ways_big_subunit * ways_small_subunit * ((Nat.choose last_subunit last_subunit) * 3)

theorem divide_PRIME_subunits : number_of_ways_to_divide_subunits 7 3 2 2 = 630 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end divide_PRIME_subunits_l262_262065


namespace common_ratio_of_geometric_series_l262_262865

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l262_262865


namespace beadshop_profit_on_wednesday_l262_262948

theorem beadshop_profit_on_wednesday (total_profit profit_on_monday profit_on_tuesday profit_on_wednesday : ℝ)
  (h1 : total_profit = 1200)
  (h2 : profit_on_monday = total_profit / 3)
  (h3 : profit_on_tuesday = total_profit / 4)
  (h4 : profit_on_wednesday = total_profit - profit_on_monday - profit_on_tuesday) :
  profit_on_wednesday = 500 := 
sorry

end beadshop_profit_on_wednesday_l262_262948


namespace cylinder_surface_area_l262_262193

theorem cylinder_surface_area (h r : ℝ) (h_h : h = 12) (h_r : r = 5) : 
  let total_surface_area := 2 * Real.pi * r^2 + 2 * Real.pi * r * h in
  total_surface_area = 170 * Real.pi :=
by
  sorry

end cylinder_surface_area_l262_262193


namespace determinant_zero_implies_sum_l262_262771

open Matrix

noncomputable def matrix_example (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![2, 5, 8],
    ![4, a, b],
    ![4, b, a]
  ]

theorem determinant_zero_implies_sum (a b : ℝ) (h : a ≠ b) (h_det : det (matrix_example a b) = 0) : a + b = 26 :=
by
  sorry

end determinant_zero_implies_sum_l262_262771


namespace betty_cookies_brownies_l262_262232

theorem betty_cookies_brownies (cookies_per_day brownies_per_day initial_cookies initial_brownies days : ℕ) :
  cookies_per_day = 3 → brownies_per_day = 1 → initial_cookies = 60 → initial_brownies = 10 → days = 7 →
  initial_cookies - days * cookies_per_day - (initial_brownies - days * brownies_per_day) = 36 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_cookies_brownies_l262_262232


namespace determine_a_for_quadratic_l262_262904

theorem determine_a_for_quadratic (a : ℝ) : 
  (∃ x : ℝ, 3 * x ^ (a - 1) - x = 5 ∧ a - 1 = 2) → a = 3 := 
sorry

end determine_a_for_quadratic_l262_262904


namespace negate_exists_l262_262477

theorem negate_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔ (∀ x : ℝ, x ≥ Real.sin x ∨ x ≤ Real.tan x) :=
by
  sorry

end negate_exists_l262_262477


namespace digits_in_base_3_l262_262580

theorem digits_in_base_3 (N : ℕ) (h_pos : 0 < N)
  (h_base9 : 9^19 ≤ N ∧ N < 9^20)
  (h_base27 : 27^12 ≤ N ∧ N < 27^13) :
  ((3 : ℕ) ^ 38 ≤ N ∧ N < (3 : ℕ) ^ 39) :=
begin
  sorry,
end

end digits_in_base_3_l262_262580


namespace six_people_theorem_l262_262799

theorem six_people_theorem  {G : SimpleGraph (Fin 6)} :
  ∀ (color : G.Edge → Prop),
  (∀ e, color e ∨ ¬ color e) →
  ∃ (x y z : Fin 6), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  ((color (G.edge x y) ∧ color (G.edge y z) ∧ color (G.edge x z)) ∨
   (¬color (G.edge x y) ∧ ¬color (G.edge y z) ∧ ¬color (G.edge x z))) :=
begin
  sorry
end

end six_people_theorem_l262_262799


namespace minimum_value_of_quadratic_l262_262130

theorem minimum_value_of_quadratic : ∀ x : ℝ, (∃ y : ℝ, y = (x-2)^2 - 3) → ∃ m : ℝ, (∀ x : ℝ, (x-2)^2 - 3 ≥ m) ∧ m = -3 :=
by
  sorry

end minimum_value_of_quadratic_l262_262130


namespace total_steps_walked_l262_262047

theorem total_steps_walked (d_mabel : ℕ) (d_helen : ℕ) (h1 : d_mabel = 4500) (h2 : d_helen = 3 * d_mabel / 4) : 
  d_mabel + d_helen = 7875 :=
by
  rw [h1, h2]
  have : 3 * 4500 / 4 = 3375 := by norm_num
  rw this
  norm_num
  sorry

end total_steps_walked_l262_262047


namespace range_f_on_interval_l262_262333

open Function

def f (x : ℝ) : ℝ := |x^2 - 4|

theorem range_f_on_interval : 
  (λ x, f x) '' (set.Icc (-2 : ℝ) (2 : ℝ)) = set.Icc 0 4 := 
sorry

end range_f_on_interval_l262_262333


namespace speed_of_stream_l262_262965

variables (V_d V_u V_m V_s : ℝ)
variables (h1 : V_d = V_m + V_s) (h2 : V_u = V_m - V_s) (h3 : V_d = 18) (h4 : V_u = 6) (h5 : V_m = 12)

theorem speed_of_stream : V_s = 6 :=
by
  sorry

end speed_of_stream_l262_262965


namespace correct_statements_about_algorithms_l262_262588

def valid_algorithm (unique : Prop) (finite_steps : Prop) (clear_steps : Prop) (definite_result : Prop) : Prop :=
  ¬ unique ∧ finite_steps ∧ clear_steps ∧ definite_result

theorem correct_statements_about_algorithms :
  let unique := false
  let finite_steps := true
  let clear_steps := true
  let definite_result := true
  valid_algorithm unique finite_steps clear_steps definite_result :=
by
  intros
  unfold valid_algorithm
  triv
  sorry -- Proof goes here.

end correct_statements_about_algorithms_l262_262588


namespace income_increase_percentage_l262_262795

theorem income_increase_percentage (I : ℝ) (P : ℝ) (h1 : 0 < I)
  (h2 : 0 ≤ P) (h3 : 0.75 * I + 0.075 * I = 0.825 * I) 
  (h4 : 1.5 * (0.25 * I) = ((I * (1 + P / 100)) - 0.825 * I)) 
  : P = 20 := by
sorry

end income_increase_percentage_l262_262795


namespace max_items_one_student_can_receive_l262_262556

theorem max_items_one_student_can_receive (N : ℕ) : 
  ∀ (students : finset ℕ) (votes : ℕ → ℕ → ℕ),
  students.card = 2019 →
  (∀ i item, 0 ≤ votes i item ∧ votes i item ≤ 1) →
  (∀ i j, i ≠ j → votes i i = 0) →
  (∀ item, (∑ i in students, votes i item = ∑ j in students, votes j item) ∧ (∃! i, votes i item > votes j item) ∨ ′≥ 1 i ≠ j , 0  = votes item) →
  (∃ student, ∑ item in 1..N, if is_max_vote student item ∧ true (votes student item > 0) then 1 else 0) ≤ 1009 :=
by
  sorry

end max_items_one_student_can_receive_l262_262556


namespace p_is_sufficient_but_not_necessary_for_q_l262_262422

variables (x y : ℝ)

def p : Prop := (x - 2) * (y - 5) ≠ 0
def q : Prop := x ≠ 2 ∨ y ≠ 5

theorem p_is_sufficient_but_not_necessary_for_q : (p → q) ∧ ¬(q → p) :=
by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l262_262422


namespace length_of_trajectory_l262_262567

-- Define the geometrical setup and result
structure EquilateralTriangle (A B S : Point) : Prop :=
(side_length : dist A B = 2 ∧ dist B S = 2 ∧ dist S A = 2)

structure IsMidpoint (S O M : Point) : Prop :=
(midpoint : M = (S + O) / 2)

structure OnBaseCircle (O P : Point) : Prop :=
(on_circle : dist O P ≤ 1)

structure Orthogonal (A M P : Point) : Prop :=
(orth : let AM := M - A in let MP := P - M in dot AM MP = 0)

def trajectory_length (A B S O M P : Point) (h_triangle : EquilateralTriangle A B S)
  (h_midpoint : IsMidpoint S O M) (h_on_circle : OnBaseCircle O P) 
  (h_orthogonal : Orthogonal A M P) : ℝ :=
dist A B    -- Placeholder expression, should define length as calculated in solution steps

theorem length_of_trajectory : ∀ (A B S O M P : Point), 
  EquilateralTriangle A B S →
  IsMidpoint S O M →
  OnBaseCircle O P →
  Orthogonal A M P →
  trajectory_length A B S O M P = (sqrt 7) / 2 :=
by
  intros A B S O M P ht hm hoc ho
  -- Proof details omitted
  sorry

end length_of_trajectory_l262_262567


namespace tan_double_angle_subtraction_l262_262332

theorem tan_double_angle_subtraction
  (α β : ℝ)
  (h1 : π / 2 < α ∧ α < π)
  (h2 : sin α = 1 / 2)
  (h3 : 0 < β ∧ β < π / 2)
  (h4 : cos β = 3 / 5) :
  tan (2 * α - β) = (48 + 25 * real.sqrt 3) / 39 :=
by
  sorry

end tan_double_angle_subtraction_l262_262332


namespace geometric_series_common_ratio_l262_262845

-- Definitions of the conditions
def infinite_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

def infinite_geometric_sum_without_first_four_terms (a : ℝ) (r : ℝ) : ℝ :=
  (a * r^4) / (1 - r)

-- Statement of the theorem
theorem geometric_series_common_ratio
  (a r : ℝ)
  (h : infinite_geometric_sum a r = 81 * infinite_geometric_sum_without_first_four_terms a r) :
  r = 1 / 3 :=
by
  sorry

end geometric_series_common_ratio_l262_262845


namespace Matt_worked_minutes_l262_262059

theorem Matt_worked_minutes (monday_minutes tuesday_minutes certain_day_minutes : ℕ) :
  monday_minutes = 450 →
  tuesday_minutes = monday_minutes / 2 →
  certain_day_minutes = tuesday_minutes + 75 →
  certain_day_minutes = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Matt_worked_minutes_l262_262059


namespace consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l262_262926

-- 6(a): Prove that the product of two consecutive integers is either divisible by 6 or gives a remainder of 2 when divided by 18.
theorem consecutive_integers_product (n : ℕ) : n * (n + 1) % 18 = 0 ∨ n * (n + 1) % 18 = 2 := 
sorry

-- 6(b): Prove that there does not exist an integer n such that the number 3n + 1 is the product of two consecutive integers.
theorem no_3n_plus_1_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, 3 * m + 1 = m * (m + 1) := 
sorry

-- 6(c): Prove that for no integer n, the number n^3 + 5n + 4 can be the product of two consecutive integers.
theorem no_n_cubed_plus_5n_plus_4_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, n^3 + 5 * n + 4 = m * (m + 1) := 
sorry

-- 6(d): Prove that none of the numbers resulting from the rearrangement of the digits in 23456780 is the product of two consecutive integers.
def is_permutation (m : ℕ) (n : ℕ) : Prop := 
-- This function definition should check that m is a permutation of the digits of n
sorry

theorem no_permutation_23456780_product_consecutive : 
  ∀ m : ℕ, is_permutation m 23456780 → ¬ ∃ n : ℕ, m = n * (n + 1) := 
sorry

end consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l262_262926


namespace ninety_eight_squared_l262_262249

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l262_262249


namespace line_perpendicular_to_plane_l262_262002

theorem line_perpendicular_to_plane {A B C O : Type} [Point O A B C]
  (h1 : Perpendicular O A B)
  (h2 : Perpendicular O A C)
  (h3 : Intersection O B C) :
  Perpendicular A (Plane O B C) :=
by sorry

end line_perpendicular_to_plane_l262_262002


namespace time_for_nth_mile_l262_262968

noncomputable def speed (k : ℝ) (d : ℝ) : ℝ := k / (d * d)

noncomputable def time_for_mile (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2 * (n - 1) * (n - 1)

theorem time_for_nth_mile (n : ℕ) (h₁ : ∀ d : ℝ, d ≥ 1 → speed (1/2) d = 1 / (2 * d * d))
  (h₂ : time_for_mile 1 = 1)
  (h₃ : time_for_mile 2 = 2) :
  time_for_mile n = 2 * (n - 1) * (n - 1) := sorry

end time_for_nth_mile_l262_262968


namespace distinct_possible_values_pq_plus_p_plus_q_l262_262262

theorem distinct_possible_values_pq_plus_p_plus_q :
  ∃ S : Finset ℕ, 
    (∀ p q ∈ ({1, 3, 5, 7, 9, 11, 13} : Finset ℕ), (p + 1) * (q + 1) - 1 ∈ S) ∧ 
    S.card = 27 :=
sorry

end distinct_possible_values_pq_plus_p_plus_q_l262_262262


namespace overall_gain_is_10_percent_l262_262538

-- Conditions
def gain_A : ℝ := 70
def cost_A : ℝ := 700
def gain_B : ℝ := 50
def cost_B : ℝ := 500
def gain_C : ℝ := 30
def cost_C : ℝ := 300

-- Total values
def total_gain : ℝ := gain_A + gain_B + gain_C
def total_cost : ℝ := cost_A + cost_B + cost_C

-- Overall gain percentage
def overall_gain_percentage : ℝ := (total_gain / total_cost) * 100

-- Problem statement: Prove that the overall gain percentage is 10%
theorem overall_gain_is_10_percent : overall_gain_percentage = 10 := by
  sorry

end overall_gain_is_10_percent_l262_262538


namespace max_consecutive_sum_l262_262889

theorem max_consecutive_sum (a N : ℤ) (h₀ : N > 0) (h₁ : N * (2 * a + N - 1) = 90) : N = 90 :=
by
  -- Proof to be provided
  sorry

end max_consecutive_sum_l262_262889


namespace playgroup_count_l262_262145

-- Definitions based on the conditions
def total_people (girls boys parents : ℕ) := girls + boys + parents
def playgroups (total size_per_group : ℕ) := total / size_per_group

-- Statement of the problem
theorem playgroup_count (girls boys parents size_per_group : ℕ)
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_parents : parents = 50)
  (h_size_per_group : size_per_group = 25) :
  playgroups (total_people girls boys parents) size_per_group = 3 :=
by {
  -- This is just the statement, the proof is skipped with sorry
  sorry
}

end playgroup_count_l262_262145


namespace dishonest_shopkeeper_weight_l262_262194

noncomputable def weight_used (gain_percent : ℝ) (correct_weight : ℝ) : ℝ :=
  correct_weight / (1 + gain_percent / 100)

theorem dishonest_shopkeeper_weight :
  weight_used 5.263157894736836 1000 = 950 := 
by
  sorry

end dishonest_shopkeeper_weight_l262_262194


namespace infinite_angles_cover_plane_l262_262649

theorem infinite_angles_cover_plane 
  (S : ℕ → set ℚ) (hS : ∀ n, ∃ θ ∈ S n, true) 
  : Union (λ n, S n ) = set.univ := 
by
   sorry

end infinite_angles_cover_plane_l262_262649


namespace ratio_of_numbers_l262_262932

noncomputable theory

theorem ratio_of_numbers (x y : ℝ) (h1 : x > y) (h2 : (x + y) / (x - y) = 4 / 3) : x = 7 * y :=
by
  sorry

end ratio_of_numbers_l262_262932


namespace jasmine_max_stickers_l262_262020

-- Given conditions and data
def sticker_cost : ℝ := 0.75
def jasmine_budget : ℝ := 10.0

-- Proof statement
theorem jasmine_max_stickers : ∃ n : ℕ, (n : ℝ) * sticker_cost ≤ jasmine_budget ∧ (∀ m : ℕ, (m > n) → (m : ℝ) * sticker_cost > jasmine_budget) :=
sorry

end jasmine_max_stickers_l262_262020


namespace ratio_of_55_to_11_l262_262895

theorem ratio_of_55_to_11 : (55 / 11) = 5 := 
by
  sorry

end ratio_of_55_to_11_l262_262895


namespace area_of_ABCD_l262_262010

variable (A B C D E F : Type) [rect : Rectangle A B C D]
variable (h_trisect : trisect_angle C (line_segment C E) (line_segment C F))
variable (h_E_on_AB : E ∈ line_segment A B)
variable (h_F_on_AD : F ∈ line_segment A D)
variable (h_BE : length (line_segment B E) = 9)
variable (h_AF : length (line_segment A F) = 3)

theorem area_of_ABCD :
  area (rectangle A B C D) = 108 * real.sqrt 3 + 54 :=
by
  sorry

end area_of_ABCD_l262_262010


namespace area_difference_square_circle_l262_262503

theorem area_difference_square_circle (r : ℝ) (π_approx : ℝ) (hπ_approx : π_approx = 3.14159) : 
  let x := r * Real.sqrt 2
  let t := 16 / 5 * r^2
  let A_circle := π_approx * r^2
  let ΔA := (t - A_circle)
  ΔA = (r^2 * 0.05841) ∧ Δ : ΔA / A_circle ≈ 1 / 54 :=
by
  sorry

end area_difference_square_circle_l262_262503


namespace determinant_inequality_l262_262084

variable (x : ℝ)

def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem determinant_inequality (h : determinant 2 (3 - x) 1 x > 0) : x > 1 := by
  sorry

end determinant_inequality_l262_262084


namespace ninety_eight_squared_l262_262246

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l262_262246


namespace athena_total_spent_l262_262731

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end athena_total_spent_l262_262731


namespace common_ratio_of_geometric_series_l262_262856

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l262_262856


namespace transformation_U_l262_262319

def U : ℝ^3 → ℝ^3 := sorry

theorem transformation_U :
  (∀ (u v : ℝ^3) (c d : ℝ), (U (c • u + d • v) = c • (U u) + d • (U v))) → 
  (∀ (u v : ℝ^3), U (u × v) = (U u) × (U v)) → 
  (U ⟨4, 4, 2⟩ = ⟨3, -1, 6⟩) → 
  (U ⟨-4, 2, 4⟩ = ⟨3, 6, -1⟩) → 
  (U ⟨2, 6, 8⟩ = ⟨4, 6, 8⟩) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end transformation_U_l262_262319


namespace remainder_7n_mod_4_l262_262907

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end remainder_7n_mod_4_l262_262907


namespace neg_P_l262_262434

def P := ∃ x : ℝ, (0 < x) ∧ (3^x < x^3)

theorem neg_P : ¬P ↔ ∀ x : ℝ, (0 < x) → (3^x ≥ x^3) :=
by
  sorry

end neg_P_l262_262434


namespace f_2_eq_neg2_l262_262673

def f : ℤ → ℤ 
| x := if x > 0 then f(x-1)
       else if x = 0 then -2
       else 3^x

theorem f_2_eq_neg2 : f 2 = -2 :=
by 
  sorry

end f_2_eq_neg2_l262_262673


namespace mike_remaining_cards_l262_262438

def initial_cards (mike_cards : ℕ) : ℕ := 87
def sam_cards (sam_bought : ℕ) : ℕ := 13
def alex_cards (alex_bought : ℕ) : ℕ := 15

theorem mike_remaining_cards (mike_cards sam_bought alex_bought : ℕ) :
  mike_cards - (sam_bought + alex_bought) = 59 :=
by
  let mike_cards := initial_cards 87
  let sam_cards := sam_bought
  let alex_cards := alex_bought
  sorry

end mike_remaining_cards_l262_262438


namespace common_ratio_of_geometric_series_l262_262836

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l262_262836


namespace oranges_to_apples_equiv_apples_for_36_oranges_l262_262399

-- Conditions
def weight_equiv (oranges apples : ℕ) : Prop :=
  9 * oranges = 6 * apples

-- Question (Theorem to Prove)
theorem oranges_to_apples_equiv_apples_for_36_oranges:
  ∃ (apples : ℕ), apples = 24 ∧ weight_equiv 36 apples :=
by
  use 24
  sorry

end oranges_to_apples_equiv_apples_for_36_oranges_l262_262399


namespace julio_fish_count_l262_262750

theorem julio_fish_count : 
  ∀ (catches_per_hour : ℕ) (hours_fishing : ℕ) (fish_lost : ℕ) (total_fish : ℕ), 
  catches_per_hour = 7 →
  hours_fishing = 9 →
  fish_lost = 15 →
  total_fish = (catches_per_hour * hours_fishing) - fish_lost →
  total_fish = 48 :=
by
  intros catches_per_hour hours_fishing fish_lost total_fish
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

# Jason had jotted down the following proof statement:

end julio_fish_count_l262_262750


namespace percentage_decrease_in_breadth_l262_262978

-- Define the given conditions
variables {L B : ℝ} -- original length and breadth
variable (P : ℝ) -- percentage decrease in breadth

-- Define the new length and area calculations
def new_length : ℝ := 0.80 * L
def new_area : ℝ := 0.72 * (L * B)

-- Assume the new breadth after decrease
def new_breadth : ℝ := B * (1 - P)

-- Define the new area using the calculated values
def calculated_new_area : ℝ := (0.80 * L) * (B * (1 - P))

-- The theorem to be proved: The percentage of decrease in breadth is 0.1
theorem percentage_decrease_in_breadth :
  calculated_new_area = new_area → P = 0.1 :=
by
  sorry

end percentage_decrease_in_breadth_l262_262978


namespace range_of_a_l262_262349

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 3 then a - x else a * log x / log 2

theorem range_of_a (a : ℝ) (h : f a 2 < f a 4) : a > -2 := by
  sorry

end range_of_a_l262_262349


namespace S_n_eq_2n_minus_2_l262_262759

noncomputable def f (x : ℝ) := (4^(x+1)) / (4^x + 2)

def S (n : ℕ) := ∑ k in finset.range (n-1), f ((k+1 : ℝ) / n)

theorem S_n_eq_2n_minus_2 {n : ℕ} (hn : n ≥ 2) :
  S n = 2 * n - 2 := by sorry

end S_n_eq_2n_minus_2_l262_262759


namespace relationship_among_a_b_c_l262_262655

noncomputable def a : ℝ := (1/2) ^ (1/5)
noncomputable def b : ℝ := (1/5) ^ (-1/2)
noncomputable def c : ℝ := Real.logb (1/5) 10

theorem relationship_among_a_b_c : b > a ∧ a > c := by
  sorry

end relationship_among_a_b_c_l262_262655


namespace probability_successful_pairs_expectation_successful_pairs_gt_half_l262_262885

open Finset

-- Definitions based on the conditions of the problem
def total_socks (n : ℕ) : ℕ := 2 * n
def n_days (n : ℕ) : ℕ := n
def successful_pair (socks : ℕ → bool) (day : ℕ) : Prop := ∀ i : ℕ, i < day → socks i = true

theorem probability_successful_pairs (n : ℕ) :
  (∏ k in range n, (1 : ℝ) / (2 * n - (2 * k + 1))) = 2^n * fact n / fact (2 * n) := sorry

theorem expectation_successful_pairs_gt_half (n : ℕ) :
  n / (2 * n - 1 : ℝ) > 0.5 := sorry

end probability_successful_pairs_expectation_successful_pairs_gt_half_l262_262885


namespace find_abc_sum_l262_262113

theorem find_abc_sum (A B C : ℤ) (h : ∀ x : ℝ, x^3 + A * x^2 + B * x + C = (x + 1) * (x - 3) * (x - 4)) : A + B + C = 11 :=
by {
  -- This statement asserts that, given the conditions, the sum A + B + C equals 11
  sorry
}

end find_abc_sum_l262_262113


namespace find_other_x_intercept_l262_262301

theorem find_other_x_intercept (a b c : ℝ) (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9)
  (h2 : a * 0^2 + b * 0 + c = 0) : ∃ x, x ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=
by
  sorry

end find_other_x_intercept_l262_262301


namespace alyssa_games_last_year_l262_262982

theorem alyssa_games_last_year (games_this_year games_next_year games_total games_last_year : ℕ) (h1 : games_this_year = 11) (h2 : games_next_year = 15) (h3 : games_total = 39) (h4 : games_last_year + games_this_year + games_next_year = games_total) : games_last_year = 13 :=
by
  rw [h1, h2, h3] at h4
  sorry

end alyssa_games_last_year_l262_262982


namespace find_tunnel_length_l262_262871

-- Define the conditions
variable (speed : ℝ) -- Train's speed in miles per hour
variable (train_length : ℝ) -- Train's length in miles
variable (time_tail_after_front_enters : ℝ) -- Time in minutes for the tail to exit after front enters
variable (time_tail_after_front_exits : ℝ) -- Time in minutes for the tail to exit after front exits

-- Assume the given values from the problem
noncomputable def train_scenario : Prop :=
  speed = 30 / 60 ∧ -- converting speed to miles per minute
  train_length = 1 ∧
  time_tail_after_front_enters = 5 ∧
  time_tail_after_front_exits = 2

-- Define the goal: length of the tunnel in miles
def tunnel_length (length : ℝ) : Prop :=
  length = time_tail_after_front_enters * speed - (time_tail_after_front_exits + time_tail_after_front_enters - time_tail_after_front_exits) * speed / 2 - train_length / 2

-- The theorem statement
theorem find_tunnel_length : train_scenario →
  ∃ length, tunnel_length length :=
begin
  assume h,
  use 1,
  simp [train_scenario, tunnel_length] at *,
  sorry
end

end find_tunnel_length_l262_262871


namespace digit_for_divisibility_by_9_l262_262514

theorem digit_for_divisibility_by_9 (A : ℕ) (hA : A < 10) : 
  (∃ k : ℕ, 83 * 1000 + A * 10 + 5 = 9 * k) ↔ A = 2 :=
by
  sorry

end digit_for_divisibility_by_9_l262_262514


namespace workers_contribution_l262_262933

theorem workers_contribution (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 360000) : W = 1200 :=
by
  sorry

end workers_contribution_l262_262933


namespace bob_weight_l262_262489

variable (j b : ℕ)

theorem bob_weight :
  j + b = 210 →
  b - j = b / 3 →
  b = 126 :=
by
  intros h1 h2
  sorry

end bob_weight_l262_262489


namespace sin_theta_of_arithmetic_progression_l262_262415

theorem sin_theta_of_arithmetic_progression (θ : ℝ) (h1 : cos θ, cos (2 * θ), and cos (3 * θ) form an arithmetic progression) (h2 : 0 < θ ∧ θ < π / 2) : sin θ = sqrt 3 / 2 :=
sorry

end sin_theta_of_arithmetic_progression_l262_262415


namespace simplify_expression_l262_262671

   theorem simplify_expression (x : ℝ) (hx : 0 ≤ x) :
     (Real.sqrt (45 * x) * Real.sqrt (32 * x) * Real.sqrt (18 * x) * Real.cbrt (27 * x)) =
     72 * x^(1/3) * Real.sqrt (5 * x) :=
   by
     sorry
   
end simplify_expression_l262_262671


namespace tg_and_ctg_relations_l262_262388

theorem tg_and_ctg_relations (α γ : ℝ) :
  (tan (α + 2 * γ) + 2 * tan α - 4 * tan (2 * γ) = 0) ∧ (tan γ = 1 / 3) →
  (cot α = 1 / 3 ∨ cot α = 2) :=
by
  intros h,
  sorry

end tg_and_ctg_relations_l262_262388


namespace cistern_filling_time_l262_262952

theorem cistern_filling_time{F E : ℝ} (hF: F = 1 / 4) (hE: E = 1 / 9) :
  (1 / (F - E) = 7.2) :=
by
  rw [hF, hE]
  have net_rate := 0.25 - 1 / 9
  rw net_rate
  exact (1 / (0.25 - 1 / 9)) = 7.2
  sorry

end cistern_filling_time_l262_262952


namespace factorial_expression_l262_262532

theorem factorial_expression : (10! * 4! * 3!) / (9! * 5!) = 12 := by
  sorry

end factorial_expression_l262_262532


namespace bad_carrots_count_l262_262692

theorem bad_carrots_count
    (haley_picked : ℕ)
    (mother_picked : ℕ)
    (good_carrots : ℕ)
    (total_picked := haley_picked + mother_picked)
    (bad_carrots := total_picked - good_carrots) :
    haley_picked = 39 → mother_picked = 38 → good_carrots = 64 → bad_carrots = 13 := by
  intros h_pick m_pick g_carrots
  rw [h_pick, m_pick, g_carrots]
  simp
  exact rfl

end bad_carrots_count_l262_262692


namespace common_ratio_of_geometric_series_l262_262839

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l262_262839


namespace min_value_4a2_b2_plus_1_div_2a_minus_b_l262_262651

variable (a b : ℝ)

theorem min_value_4a2_b2_plus_1_div_2a_minus_b (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a > b) (h4 : a * b = 1 / 2) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x > y → x * y = 1 / 2 → (4 * x^2 + y^2 + 1) / (2 * x - y) ≥ c) :=
sorry

end min_value_4a2_b2_plus_1_div_2a_minus_b_l262_262651


namespace spring_length_relationship_maximum_mass_l262_262614

theorem spring_length_relationship (x y : ℝ) : 
  (y = 0.5 * x + 12) ↔ y = 12 + 0.5 * x := 
by sorry

theorem maximum_mass (x y : ℝ) : 
  (y = 0.5 * x + 12) → (y ≤ 20) → (x ≤ 16) :=
by sorry

end spring_length_relationship_maximum_mass_l262_262614


namespace three_digit_integers_count_l262_262696

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def no_zero_digits (n : ℕ) : Prop := 
  let digits := n.digits 10 in
  (digits.all (λ d, d ≠ 0))

def less_than_700 (n : ℕ) : Prop := n < 700

def at_least_two_same (n : ℕ) : Prop := 
  let digits := n.digits 10 in
  (digits.counts.toList.any (λ ⟨_, k⟩, k > 1))
  
theorem three_digit_integers_count : 
  {n : ℕ | is_three_digit n ∧ less_than_700 n ∧ no_zero_digits n ∧ at_least_two_same n}.card = 171 :=
by sorry

end three_digit_integers_count_l262_262696


namespace megan_bottles_l262_262437

theorem megan_bottles (initial_bottles drank gave_away remaining_bottles : ℕ) 
  (h1 : initial_bottles = 45)
  (h2 : drank = 8)
  (h3 : gave_away = 12) :
  remaining_bottles = initial_bottles - (drank + gave_away) :=
by 
  sorry

end megan_bottles_l262_262437


namespace box_dimensions_l262_262238

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  -- We assume the proof is correct based on given conditions
  sorry

end box_dimensions_l262_262238


namespace simplify_expr1_simplify_expr2_simplify_expr3_l262_262511

theorem simplify_expr1 : -2.48 + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem simplify_expr2 : (7/13) * (-9) + (7/13) * (-18) + (7/13) = -14 := by
  sorry

theorem simplify_expr3 : -((20 + 1/19) * 38) = -762 := by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l262_262511


namespace correct_calculation_l262_262533

theorem correct_calculation :
  (-7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2) ∧
  ¬ (2 * x + 3 * y = 5 * x * y) ∧
  ¬ (6 * x^2 - (-x^2) = 5 * x^2) ∧
  ¬ (4 * m * n - 3 * m * n = 1) :=
by
  sorry

end correct_calculation_l262_262533


namespace remaining_amount_after_purchase_l262_262274

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem remaining_amount_after_purchase : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end remaining_amount_after_purchase_l262_262274


namespace solve_for_x_l262_262096

theorem solve_for_x (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4)) : x = -14 := 
by 
  sorry

end solve_for_x_l262_262096


namespace find_sin_of_smallest_acute_angle_l262_262417

noncomputable def smallest_acute_angle (θ : ℝ) : Prop :=
θ > 0 ∧ θ < (Real.pi / 2) ∧
(∃ a b c : ℝ, a, b, c ∈ {Real.cos θ, Real.cos (2 * θ), Real.cos (3 * θ)} ∧ b - a = c - b)

theorem find_sin_of_smallest_acute_angle (θ : ℝ) (h : smallest_acute_angle θ) : Real.sin θ = 0 := 
sorry

end find_sin_of_smallest_acute_angle_l262_262417


namespace f_2018_eq_2017_l262_262572

-- Define f(1) and f(2)
def f : ℕ → ℕ 
| 1 => 1
| 2 => 1
| n => if h : n ≥ 3 then (f (n - 1) - f (n - 2) + n) else 0

-- State the theorem to prove f(2018) = 2017
theorem f_2018_eq_2017 : f 2018 = 2017 := 
by 
  sorry

end f_2018_eq_2017_l262_262572


namespace solution_set_of_inequality_l262_262139

theorem solution_set_of_inequality : { x : ℝ | x^2 - 2 * x + 1 ≤ 0 } = {1} :=
sorry

end solution_set_of_inequality_l262_262139


namespace number_of_distinct_values_l262_262255

-- Define the set of positive odd integers less than 15
def odd_integers_less_15 : set ℕ := {1, 3, 5, 7, 9, 11, 13}

-- Define the function to calculate the given expression
def expression (p q : ℕ) : ℕ := p * q + p + q

-- Define the transformed function using Simon's Favorite Factoring Trick
def transformed_expression (p q : ℕ) : ℕ := (p + 1) * (q + 1) - 1

-- Prove that the number of distinct possible values of the expression is 27
theorem number_of_distinct_values :
  finset.card
    (finset.image (λ p q : ℕ, expression p q)
      (finset.product (finset.filter (λ x : ℕ, x ∈ odd_integers_less_15) finset.univ)
                      (finset.filter (λ y : ℕ, y ∈ odd_integers_less_15) finset.univ))) = 27 :=
by
  sorry

end number_of_distinct_values_l262_262255


namespace three_pow_124_mod_7_l262_262896

theorem three_pow_124_mod_7 : (3^124) % 7 = 4 := by
  sorry

end three_pow_124_mod_7_l262_262896


namespace athena_spent_l262_262730

theorem athena_spent :
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  total_cost = 14 :=
by
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  sorry

end athena_spent_l262_262730


namespace average_mark_of_remaining_students_l262_262817

theorem average_mark_of_remaining_students 
  (N A E A_E : ℝ)
  (N_eq : N = 10)
  (A_eq : A = 70)
  (E_eq : E = 5)
  (A_E_eq : A_E = 50) :
  let total_marks := N * A
  let total_marks_excluded := E * A_E
  let total_marks_remaining := total_marks - total_marks_excluded
  let students_remaining := N - E
  let avg_remaining := total_marks_remaining / students_remaining
  avg_remaining = 90 :=
by 
  dsimp [total_marks, total_marks_excluded, total_marks_remaining, students_remaining, avg_remaining]
  rw [N_eq, A_eq, E_eq, A_E_eq]
  norm_num
  sorry

end average_mark_of_remaining_students_l262_262817


namespace induction_sum_formula_l262_262508

theorem induction_sum_formula :
  ∀ n : ℕ, (∑ i in finset.range (n+1), (2*i + 1)) = (n + 1) * (2*n + 1) :=
by
  sorry

end induction_sum_formula_l262_262508


namespace fraction_evaporated_is_3_over_4_l262_262304

noncomputable def fraction_evaporated_on_second_day {V : ℝ} (x : ℝ) : Prop :=
  (1 - x) * (2 / 3) * V = (1 / 6) * V

theorem fraction_evaporated_is_3_over_4 {V : ℝ} (hV : V ≠ 0) :
  ∃ x : ℝ, fraction_evaporated_on_second_day x ∧ x = 3 / 4 :=
by
  use 3 / 4
  unfold fraction_evaporated_on_second_day
  sorry

end fraction_evaporated_is_3_over_4_l262_262304


namespace jerry_feathers_left_l262_262024

theorem jerry_feathers_left : 
  (let hawk_feathers := 23 in
  let eagle_feathers := 24 * hawk_feathers in
  let total_feathers := hawk_feathers + eagle_feathers in
  let feathers_after_gift := total_feathers - 25 in
  let feathers_sold := Integer.floor (0.75 * feathers_after_gift) in
  feathers_after_gift - feathers_sold = 138) :=
sorry

end jerry_feathers_left_l262_262024


namespace range_of_x_function_l262_262136

open Real

theorem range_of_x_function : 
  ∀ x : ℝ, (x + 1 >= 0) ∧ (x - 3 ≠ 0) ↔ (x >= -1) ∧ (x ≠ 3) := 
by 
  sorry 

end range_of_x_function_l262_262136


namespace calculate_sqrt_expr_l262_262997

theorem calculate_sqrt_expr : 
  sqrt ((16^12 + 2^24) / (16^5 + 2^30)) = 512 := 
by
  sorry

end calculate_sqrt_expr_l262_262997


namespace trig_identity_l262_262623

theorem trig_identity (α : ℝ) :
  (cos α - cos (3 * α) + cos (5 * α) - cos (7 * α)) / (sin α + sin (3 * α) + sin (5 * α) + sin (7 * α)) = -tan α :=
by
  sorry

end trig_identity_l262_262623


namespace rolles_theorem_l262_262471

theorem rolles_theorem {f : ℝ → ℝ} {a b : ℝ} (h_diff : ∀ x ∈ set.Icc a b, differentiable_at ℝ f x)
  (h_eq : f a = f b) : ∃ x₀ ∈ set.Ioo a b, deriv f x₀ = 0 :=
sorry

end rolles_theorem_l262_262471


namespace minimum_m_value_l262_262345

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_m_value :
  (∃ m, ∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m) ∧
  ∀ m', (∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m') → 3 + Real.sqrt 3 / 2 ≤ m' :=
by
  sorry

end minimum_m_value_l262_262345


namespace sum_A_tensor_B_l262_262267

open Set

def A : Set ℕ := {2, 0}
def B : Set ℕ := {0, 8}

def A_tensor_B : Set ℕ := {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y}

theorem sum_A_tensor_B : (∑ z in A_tensor_B, z) = 16 := by
  have hA : A = {2, 0} := rfl
  have hB : B = {0, 8} := rfl
  have hAB : A_tensor_B = {0, 16} :=
    Set.ext (λ z,
      ⟨λ h, by
        rcases h with ⟨x, hxA, y, hyB, hz⟩
        cases hxA with hA2 hA0;
        cases hyB with hB0 hB8;
        simp_all [hxA, hyB, hz]⟩,
      λ h, Set.mem_def.mpr (
        or.inr (by cases h; simp_all [Set.mem_def, h] : 0 = 0 ∨ 0 = 16) )
    )
  sorry

end sum_A_tensor_B_l262_262267


namespace result_of_dividing_295_by_5_and_adding_6_is_65_l262_262500

theorem result_of_dividing_295_by_5_and_adding_6_is_65 : (295 / 5) + 6 = 65 := by
  sorry

end result_of_dividing_295_by_5_and_adding_6_is_65_l262_262500


namespace range_of_a_l262_262310

theorem range_of_a (a : ℝ) :
  (∀ x, 2 * x^2 - x - 1 ≤ 0 → x ∈ set.Icc (-1/2) 1) ∧
  (∀ x, x^2 - (2 * a - 1) * x + a * (a - 1) ≤ 0 → x ∈ set.Icc (a - 1) a) ∧
  ((∀ x, (x^2 - (2 * a - 1) * x + a * (a - 1) ≤ 0) → (2 * x^2 - x - 1 ≤ 0)) ∧ ¬(∀ x, (2 * x^2 - x - 1 ≤ 0) → (x^2 - (2 * a - 1) * x + a * (a - 1) ≤ 0)))
  → (1/2 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l262_262310


namespace num_ways_to_select_cubes_l262_262944

-- Define the \(3 \times 3 \times 1\) block
def block_3x3x1 : list (ℕ × ℕ × ℕ) :=
  [(x, y, 0) | x, y ∈ [1, 2, 3]]

-- Define what it means for two unit cubes to make a 45-degree angle with the horizontal plane
def cubes_make_45_degrees (c1 c2 : ℕ × ℕ × ℕ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = 1 ∧ c1.3 = c2.3

-- Main theorem statement
theorem num_ways_to_select_cubes : 
  ∃ (n : ℕ), n = 30 ∧ 
  (∀ (b1 b2 : ℕ × ℕ × ℕ), b1 ∈ block_3x3x1 → b2 ∈ block_3x3x1 → b1 ≠ b2 → cubes_make_45_degrees b1 b2 → b1.3 = 0 ∧ b2.3 = 0) :=
sorry

end num_ways_to_select_cubes_l262_262944


namespace num_O_atoms_correct_l262_262957

-- Conditions
def atomic_weight_H : ℕ := 1
def atomic_weight_Cr : ℕ := 52
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_Cr_atoms : ℕ := 1
def molecular_weight : ℕ := 118

-- Calculations
def weight_H : ℕ := num_H_atoms * atomic_weight_H
def weight_Cr : ℕ := num_Cr_atoms * atomic_weight_Cr
def total_weight_H_Cr : ℕ := weight_H + weight_Cr
def weight_O : ℕ := molecular_weight - total_weight_H_Cr
def num_O_atoms : ℕ := weight_O / atomic_weight_O

-- Theorem to prove the number of Oxygen atoms is 4
theorem num_O_atoms_correct : num_O_atoms = 4 :=
by {
  sorry -- Proof not provided.
}

end num_O_atoms_correct_l262_262957


namespace fisherman_gets_8_red_snappers_l262_262824

noncomputable def num_red_snappers (R : ℕ) : Prop :=
  let cost_red_snapper := 3
  let cost_tuna := 2
  let num_tunas := 14
  let total_earnings := 52
  (R * cost_red_snapper) + (num_tunas * cost_tuna) = total_earnings

theorem fisherman_gets_8_red_snappers : num_red_snappers 8 :=
by
  sorry

end fisherman_gets_8_red_snappers_l262_262824


namespace distinct_sums_products_under_15_l262_262260

def is_positive_odd (n : ℕ) : Prop := n % 2 = 1 ∧ n > 0

noncomputable def possible_values : ℕ → ℕ → ℕ := λ p q, p * q + p + q

theorem distinct_sums_products_under_15 : 
  {pq_sum | ∃ p q : ℕ, is_positive_odd p ∧ is_positive_odd q ∧ p < 15 ∧ q < 15 ∧ pq_sum = possible_values p q}.to_finset.card = 28 :=
sorry

end distinct_sums_products_under_15_l262_262260


namespace probability_white_given_popped_l262_262558

theorem probability_white_given_popped :
  (3/4 : ℚ) * (3/5 : ℚ) / ((3/4 : ℚ) * (3/5 : ℚ) + (1/4 : ℚ) * (1/2 : ℚ)) = 18/23 := by
  sorry

end probability_white_given_popped_l262_262558


namespace geometric_series_common_ratio_l262_262851

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l262_262851


namespace problem_correct_answer_l262_262413

def floor (x : ℝ) : ℤ := Int.floor x

def x_seq (a : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => Int.toNat ∘ floor $ (x_seq a n + floor (a / x_seq a n)) / 2

theorem problem_correct_answer (a : ℕ) (h1 : x_seq 5 0 = 5)
  (h2 : x_seq 5 1 = 3) (h3 : x_seq 5 2 = 2)
  (h4 : ¬ ∃ k, ∀ n ≥ k, x_seq a n = x_seq a k)
  (h5 : ∀ n ≥ 1, (x_seq a n : ℝ) > Real.sqrt a - 1)
  (h6 : ∃ k, (x_seq a (k+1) : ℝ) ≥ x_seq a k → ∀ n ≥ k, x_seq a n = Int.toNat ∘ floor (Real.sqrt a)) :
  3 = 3 := 
sorry

end problem_correct_answer_l262_262413


namespace find_a_l262_262659

-- Definitions given in the conditions
def f (x : ℝ) : ℝ := x^2 - 2
def g (x : ℝ) : ℝ := x^2 + 6

-- The main theorem to show
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 18) : a = Real.sqrt 14 := sorry

end find_a_l262_262659


namespace pepins_theorem_l262_262183

theorem pepins_theorem (n : ℕ) (hn : n > 0) (Fn : ℕ := 2^(2^n) + 1) :
  (3^((Fn - 1) / 2) ≡ -1 [MOD Fn]) ↔ prime Fn := sorry

end pepins_theorem_l262_262183


namespace largest_number_is_correct_l262_262142

theorem largest_number_is_correct (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 3) : c = 33.25 :=
by
  sorry

end largest_number_is_correct_l262_262142


namespace solution_exists_l262_262432

noncomputable def proof_problem (x y t : ℝ) : Prop :=
  (x * (t + 0.2) + 0.4 * y = 12) ∧
  (y * t = 12) ∧
  (t < 1) →

  (x = 6 ∧ y = 15)

theorem solution_exists : ∃ (x y t : ℝ), proof_problem x y t :=
by
  use 6, 15, 12 / 15
  split
  { ring_nf,
    exact 12, },
  split
  { exact 12, },
  { norm_num, }

end solution_exists_l262_262432


namespace ninety_eight_squared_l262_262244

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l262_262244


namespace B_N_C_collinear_l262_262230

-- Define points A and B
variables (A B : Point)

-- Define a point M on the segment AB
variable (M : Point)
axiom M_between_A_and_B : M ∈ segment A B

-- Define the squares AMCD and BEHM
variable (C D : Point)
axiom AMCD_square : square A M C D
variable (E H : Point)
axiom BEHM_square : square B E H M

-- Define the point N as the intersection of the circumcircles of the squares
variable (N : Point)
axiom N_intersection : N ∈ circumcircle A M C D ∧ N ∈ circumcircle B E H M

-- Prove that B, N, and C are collinear
theorem B_N_C_collinear (A B M C D E H N : Point)
  (M_between_A_and_B : M ∈ segment A B)
  (AMCD_square : square A M C D)
  (BEHM_square : square B E H M)
  (N_intersection : N ∈ circumcircle A M C D ∧ N ∈ circumcircle B E H M) :
  collinear ({B, N, C} : set Point) :=
sorry

end B_N_C_collinear_l262_262230


namespace factorial_division_example_l262_262592

theorem factorial_division_example : (fact (fact 4)) / (fact 4) = fact 23 :=
by 
  sorry

end factorial_division_example_l262_262592


namespace value_of_some_fraction_l262_262628

theorem value_of_some_fraction :
  let int_floor (x : ℝ) := ⌊x⌋₊ in
  let some_fraction := 0 in
  int_floor(6.5) * some_fraction + int_floor(2) * 7.2 + int_floor(8.4) - 6.6 = 15.8
:=
sorry

end value_of_some_fraction_l262_262628


namespace num_of_multiples_of_73_l262_262583

def a (n k : ℕ) : ℕ := 2^(n-1)*(n + 2*k - 2)

def is_multiple_of_73 (n k : ℕ) : Prop := 73 ∣ a n k

theorem num_of_multiples_of_73 :
  let total_multiples := ∑ n in Finset.range 36 | n % 2 = 1, (1 : ℕ).range (53 - n)
  total_multiples.count (λ k, is_multiple_of_73 n k) = 18 :=
sorry

end num_of_multiples_of_73_l262_262583


namespace common_ratio_of_geometric_series_l262_262837

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l262_262837


namespace paulsons_income_increase_percentage_l262_262793

-- Definitions of conditions
variables {I : ℝ} -- Original income
def E := 0.75 * I -- Original expenditure
def S := I - E -- Original savings

-- New income defined by percentage increase
variables {x : ℝ} -- percentage increase in income in fraction (0.20 for 20%)
def I_new := I * (1 + x)

-- New expenditure
def E_new := E * 1.10

-- New savings
def S_new := I_new - E_new

-- Percentage increase in savings given as 49.99999999999996% ~= 50%
def percentage_increase_in_savings := 0.50

-- New savings in terms of original savings
def S_new_from_percentage := S * (1 + percentage_increase_in_savings)

-- Theorem to prove that the percentage increase in income is 20%
theorem paulsons_income_increase_percentage (h : S_new = S_new_from_percentage) : x = 0.20 :=
by sorry

end paulsons_income_increase_percentage_l262_262793


namespace find_g_of_3_l262_262457

def f (x : ℝ) : ℝ := 3 / (2 - x)
def f_inv (x : ℝ) : ℝ := 2 - (3 / x)
def g (x : ℝ) : ℝ := 1 / (f_inv x) + 9

theorem find_g_of_3 : g 3 = 10 := by
  sorry

end find_g_of_3_l262_262457


namespace minimum_fully_acquainted_l262_262715

theorem minimum_fully_acquainted (n : ℕ) (h_n : n = 1982)
    (h_cond : ∀ (A B C D : ℕ), A ≠ B → B ≠ C → C ≠ D → D ≠ A → A ≠ C → B ≠ D → 
              (A ≤ n ∧ B ≤ n ∧ C ≤ n ∧ D ≤ n) →
              (∃ (E : ℕ), E ≤ n ∧ (E = A ∨ E = B ∨ E = C ∨ E = D) ∧ 
              (E = A → B ≤ n ∧ C ≤ n ∧ D ≤ n) ∧ 
              (E = B → A ≤ n ∧ C ≤ n ∧ D ≤ n) ∧ 
              (E = C → A ≤ n ∧ B ≤ n ∧ D ≤ n) ∧ 
              (E = D → A ≤ n ∧ B ≤ n ∧ C ≤ n) ∧
              (E = A → acquaintance E B ∧ acquaintance E C ∧ acquaintance E D) ∧
              (E = B → acquaintance E A ∧ acquaintance E C ∧ acquaintance E D) ∧
              (E = C → acquaintance E A ∧ acquaintance E B ∧ acquaintance E D) ∧
              (E = D → acquaintance E A ∧ acquaintance E B ∧ acquaintance E C))) :
  ∃ (m : ℕ), m = 1979 ∧ ∀ (x : ℕ), x ≤ n → acquaintance x x :=
sorry

end minimum_fully_acquainted_l262_262715


namespace smallest_m_solutions_l262_262762

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

noncomputable def g (x : ℝ) : ℝ := abs (3 * (fractional_part x) - 1.5)

def num_real_solutions (m : ℕ) : ℕ :=
  -- This is a placeholder for the function counting solutions to mg(xg(x)) = x
  sorry

theorem smallest_m_solutions (m : ℕ) (h : num_real_solutions 39 >= 3000) : m = 39 :=
  sorry

end smallest_m_solutions_l262_262762


namespace product_comparison_l262_262934

theorem product_comparison :
    (∏ n in Finset.range (2015 + 1) \ Finset.range 1755, (1 + (1 / (n + 1755)))) > Real.sqrt (8 / 7) :=
sorry

end product_comparison_l262_262934


namespace average_age_of_new_students_l262_262545

theorem average_age_of_new_students :
  ∀ (initial_group_avg_age new_group_avg_age : ℝ) (initial_students new_students total_students : ℕ),
  initial_group_avg_age = 14 →
  initial_students = 10 →
  new_group_avg_age = 15 →
  new_students = 5 →
  total_students = initial_students + new_students →
  (new_group_avg_age * total_students - initial_group_avg_age * initial_students) / new_students = 17 :=
by
  intros initial_group_avg_age new_group_avg_age initial_students new_students total_students
  sorry

end average_age_of_new_students_l262_262545


namespace range_fraction_l262_262085

theorem range_fraction {x y : ℝ} (h : x^2 + y^2 + 2 * x = 0) :
  ∃ a b : ℝ, a = -1 ∧ b = 1 / 3 ∧ ∀ z, z = (y - x) / (x - 1) → a ≤ z ∧ z ≤ b :=
by 
  sorry

end range_fraction_l262_262085


namespace find_q_l262_262327

noncomputable def geometric_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem find_q (a1 : ℝ) (h : 0 < q) (k : ℕ) (hk : k > 0) 
  (H : ∀ k : ℕ, k > 0 → (tendsto (λ n, geometric_sum a1 q n - geometric_sum a1 q (k+1)) at_top (nhds (a1 * q^(k-1)))) ) :
  q = (real.sqrt 5 - 1) / 2 := 
begin
  sorry
end

end find_q_l262_262327


namespace athena_spent_l262_262729

theorem athena_spent :
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  total_cost = 14 :=
by
  let sandwich_cost := 3
  let num_sandwiches := 3
  let drink_cost := 2.5
  let num_drinks := 2
  let total_cost := num_sandwiches * sandwich_cost + num_drinks * drink_cost
  sorry

end athena_spent_l262_262729


namespace mabel_total_walk_l262_262043

theorem mabel_total_walk (steps_mabel : ℝ) (ratio_helen : ℝ) (steps_helen : ℝ) (total_steps : ℝ) :
  steps_mabel = 4500 →
  ratio_helen = 3 / 4 →
  steps_helen = ratio_helen * steps_mabel →
  total_steps = steps_mabel + steps_helen →
  total_steps = 7875 := 
by
  intros h_steps_mabel h_ratio_helen h_steps_helen h_total_steps
  rw [h_steps_mabel, h_ratio_helen] at h_steps_helen
  rw [h_steps_helen, h_steps_mabel] at h_total_steps
  rw h_total_steps
  rw [h_steps_mabel, h_ratio_helen]
  linarith

end mabel_total_walk_l262_262043


namespace percentage_y_less_than_x_l262_262200

def percentage_decrease_from_x_to_y (x y : ℝ) (h: x = 7 * y) : ℝ :=
  (6 / 7) * 100

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 7 * y) : percentage_decrease_from_x_to_y x y h = 85.71 :=
  sorry

end percentage_y_less_than_x_l262_262200


namespace jennie_jisoo_rohee_juice_l262_262744

theorem jennie_jisoo_rohee_juice :
  ∀ (jennie jisoo rohee : ℚ), 
  jennie = 9/5 ∧ 
  jennie = jisoo + 2/10 ∧ 
  jisoo = rohee - 3/10 → 
  jisoo = 16/10 ∧ 
  jisoo < jennie ∧ 
  jisoo < rohee :=
by
  intros jennie jisoo rohee,
  assume h,
  cases h with h1 h_rest,
  cases h_rest with h2 h3,
  sorry

end jennie_jisoo_rohee_juice_l262_262744


namespace peaches_initial_l262_262056

theorem peaches_initial (picked : ℕ) (total : ℕ) (h1 : picked = 52) (h2 : total = 86) : 
  total - picked = 34 :=
by 
  rw [h1, h2]
  rfl

end peaches_initial_l262_262056


namespace max_marked_cells_in_20x20_board_l262_262157

def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 = x2 - 1))

def is_marked (board : ℕ × ℕ → Prop) (cells : List (ℕ × ℕ)) : Prop :=
  ∀ (c ∈ cells), board c

def at_most_one_adjacent_marked (board : ℕ × ℕ → Prop) (cells : List (ℕ × ℕ)) : Prop :=
  ∀ (c ∈ cells), ∃ (count : ℕ), count = List.countp (λ x, is_adjacent c.fst c.snd x.fst x.snd) cells ∧ count ≤ 1

noncomputable def max_marked_cells (board_size : ℕ) (board : ℕ × ℕ → Prop) : ℕ :=
  classical.some (nat.exists_max (λ n, ∃ (cells : List (ℕ × ℕ)),
    is_marked board cells ∧
    at_most_one_adjacent_marked board cells ∧
    List.length cells = n))

theorem max_marked_cells_in_20x20_board : max_marked_cells 20 (λ _, true) = 100 :=
sorry

end max_marked_cells_in_20x20_board_l262_262157


namespace geometric_series_common_ratio_l262_262850

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l262_262850


namespace triangle_inequality_for_min_segments_l262_262320

theorem triangle_inequality_for_min_segments
  (a b c d : ℝ)
  (a1 b1 c1 : ℝ)
  (h1 : a1 = min a d)
  (h2 : b1 = min b d)
  (h3 : c1 = min c d)
  (h_triangle : c < a + b) :
  a1 + b1 > c1 ∧ a1 + c1 > b1 ∧ b1 + c1 > a1 := sorry

end triangle_inequality_for_min_segments_l262_262320


namespace centers_form_square_l262_262100

-- Define a square constructed on a side of a parallelogram
structure Square (V : Type) [Add V] [Neg V] [Module ℝ V] :=
  (center : V)
  (side_length : ℝ)
  (vertices : Fin 4 → V)

-- Define a parallelogram
structure Parallelogram (V : Type) [Add V] [Neg V] [Module ℝ V] :=
  (A B C D : V)
  (opposite_sides_equal : (B - A = D - C) ∧ (C - B = A - D))

-- Given parallelogram and the squares constructed on its sides, prove the centers form a square
theorem centers_form_square {V : Type} [InnerProductSpace ℝ V] (P : Parallelogram V)
  (Sq_AB : Square V) (Sq_BC : Square V) (Sq_CD : Square V) (Sq_DA : Square V)
  (h_AB : Sq_AB.vertices 0 = P.A ∧ Sq_AB.vertices 1 = P.B)
  (h_BC : Sq_BC.vertices 0 = P.B ∧ Sq_BC.vertices 1 = P.C)
  (h_CD : Sq_CD.vertices 0 = P.C ∧ Sq_CD.vertices 1 = P.D)
  (h_DA : Sq_DA.vertices 0 = P.D ∧ Sq_DA.vertices 1 = P.A) :
  ∃ Square_centers : Square V, 
    (Square_centers.center = Sq_AB.center ∧
     Square_centers.center = Sq_BC.center ∧
     Square_centers.center = Sq_CD.center ∧
     Square_centers.center = Sq_DA.center) ∧
    (Square_centers.side_length = dist Sq_AB.center Sq_BC.center ∧
     Square_centers.side_length = dist Sq_BC.center Sq_CD.center ∧
     Square_centers.side_length = dist Sq_CD.center Sq_DA.center ∧
     Square_centers.side_length = dist Sq_DA.center Sq_AB.center) ∧
    (angle Sq_AB.center Sq_BC.center = π / 2 ∧
     angle Sq_BC.center Sq_CD.center = π / 2 ∧
     angle Sq_CD.center Sq_DA.center = π / 2 ∧
     angle Sq_DA.center Sq_AB.center = π / 2) :=
sorry

end centers_form_square_l262_262100


namespace sum_seq_formula_l262_262870

def a (n : ℕ) : ℚ := (n^2 + 2^n) / 2^(n-1)

def sum_seq (n : ℕ) : ℚ := ∑ k in Finset.range n, a (k+1)

theorem sum_seq_formula (n : ℕ) : 
  sum_seq n = (n * (n + 1) / 2) + 2^(n-2) := 
by
  sorry

end sum_seq_formula_l262_262870


namespace group_c_right_angled_triangle_l262_262587

theorem group_c_right_angled_triangle :
  (3^2 + 4^2 = 5^2) := by
  sorry

end group_c_right_angled_triangle_l262_262587


namespace jaylen_bell_peppers_ratio_l262_262022

theorem jaylen_bell_peppers_ratio :
  ∃ j_bell_p, ∃ k_bell_p, ∃ j_green_b, ∃ k_green_b, ∃ j_carrots, ∃ j_cucumbers, ∃ j_total_veg,
  j_carrots = 5 ∧
  j_cucumbers = 2 ∧
  k_bell_p = 2 ∧
  k_green_b = 20 ∧
  j_green_b = 20 / 2 - 3 ∧
  j_total_veg = 18 ∧
  j_carrots + j_cucumbers + j_green_b + j_bell_p = j_total_veg ∧
  j_bell_p / k_bell_p = 2 :=
sorry

end jaylen_bell_peppers_ratio_l262_262022


namespace time_to_cross_signal_pole_l262_262945

-- Given conditions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 39
def length_of_platform : ℝ := 1162.5

-- The question to prove
theorem time_to_cross_signal_pole :
  (length_of_train / ((length_of_train + length_of_platform) / time_to_cross_platform)) = 8 :=
by
  sorry

end time_to_cross_signal_pole_l262_262945


namespace number_of_solutions_fractional_equation_l262_262091

open Real

def fractional_part (x : ℝ) : ℝ := x - floor x

theorem number_of_solutions_fractional_equation : 
  (∃ sols : ℕ, sols = 181 ∧ ∀ x : ℝ, |x| ≤ 10 → fractional_part x + fractional_part (x * x) = 1 → sols = 181) :=
by
  -- Define the fractional part function
  let f := λ x : ℝ, x - floor x
  -- Given conditions
  sorry

end number_of_solutions_fractional_equation_l262_262091


namespace X_Y_independent_normal_l262_262937

open real measure_theory probability_theory

noncomputable def rayleigh_pdf (σ : ℝ) (r : ℝ) : ℝ :=
  if r > 0 then (r / σ^2) * exp (-r^2 / (2 * σ^2)) else 0

def uniform_pdf (α : ℝ) (k : ℕ) (θ : ℝ) : ℝ :=
  if α ≤ θ ∧ θ < α + 2 * π * k then 1 / (2 * π * k) else 0

def cos_density (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then 1 / (π * sqrt (1 - x^2)) else 0

def sin_density (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then 1 / (π * sqrt (1 - x^2)) else 0

theorem X_Y_independent_normal (σ : ℝ) (α : ℝ) (k : ℕ) :
  (∀ r > 0, pdf (rayleigh_pdf σ) r) →
  (∀ θ, pdf (uniform_pdf α k) θ) →
  independent X Y →
  ∀ x y, has_pdf (λ X = R * (cos θ)) :=
    sorry

end X_Y_independent_normal_l262_262937


namespace example_number_is_not_octal_l262_262983

-- Define a predicate that checks if a digit is valid in the octal system
def is_octal_digit (d : ℕ) : Prop :=
  d < 8

-- Define a predicate that checks if all digits in a number represented as list of ℕ are valid octal digits
def is_octal_number (n : List ℕ) : Prop :=
  ∀ d ∈ n, is_octal_digit d

-- Example number represented as a list of its digits
def example_number : List ℕ := [2, 8, 5, 3]

-- The statement we aim to prove
theorem example_number_is_not_octal : ¬ is_octal_number example_number := by
  -- Proof goes here
  sorry

end example_number_is_not_octal_l262_262983


namespace valid_votes_calculation_l262_262375

noncomputable def total_votes : ℕ := 700000
noncomputable def invalid_percentage : ℚ := 0.12
noncomputable def valid_percentage : ℚ := 0.88
noncomputable def valid_votes : ℕ := (valid_percentage * total_votes).toNat

noncomputable def candidate_A_percentage : ℚ := 0.45
noncomputable def candidate_B_percentage : ℚ := 0.25
noncomputable def candidate_C_percentage : ℚ := 0.15
noncomputable def candidate_D_percentage : ℚ := 0.10
noncomputable def candidate_E_percentage : ℚ := 0.05

noncomputable def votes_A : ℕ := (candidate_A_percentage * valid_votes).toNat
noncomputable def votes_B : ℕ := (candidate_B_percentage * valid_votes).toNat
noncomputable def votes_C : ℕ := (candidate_C_percentage * valid_votes).toNat
noncomputable def votes_D : ℕ := (candidate_D_percentage * valid_votes).toNat
noncomputable def votes_E : ℕ := (candidate_E_percentage * valid_votes).toNat

theorem valid_votes_calculation : 
  votes_A = 277200 ∧ 
  votes_B = 154000 ∧ 
  votes_C = 92400 ∧ 
  votes_D = 61600 ∧ 
  votes_E = 30800 :=
by {
  have hv : valid_votes = 616000 := by sorry,
  rw hv,
  have h_A : votes_A = (candidate_A_percentage * 616000).toNat := by sorry,
  rw h_A,
  have h_A_goal : (candidate_A_percentage * 616000).toNat = 277200 := by sorry,
  rw h_A_goal,
  have h_B : votes_B = (candidate_B_percentage * 616000).toNat := by sorry,
  rw h_B,
  have h_B_goal : (candidate_B_percentage * 616000).toNat = 154000 := by sorry,
  rw h_B_goal,
  have h_C : votes_C = (candidate_C_percentage * 616000).toNat := by sorry,
  rw h_C,
  have h_C_goal : (candidate_C_percentage * 616000).toNat = 92400 := by sorry,
  rw h_C_goal,
  have h_D : votes_D = (candidate_D_percentage * 616000).toNat := by sorry,
  rw h_D,
  have h_D_goal : (candidate_D_percentage * 616000).toNat = 61600 := by sorry,
  rw h_D_goal,
  have h_E : votes_E = (candidate_E_percentage * 616000).toNat := by sorry,
  rw h_E,
  have h_E_goal : (candidate_E_percentage * 616000).toNat = 30800 := by sorry,
  rw h_E_goal,
  tauto
}

end valid_votes_calculation_l262_262375


namespace number_of_distinct_values_l262_262256

-- Define the set of positive odd integers less than 15
def odd_integers_less_15 : set ℕ := {1, 3, 5, 7, 9, 11, 13}

-- Define the function to calculate the given expression
def expression (p q : ℕ) : ℕ := p * q + p + q

-- Define the transformed function using Simon's Favorite Factoring Trick
def transformed_expression (p q : ℕ) : ℕ := (p + 1) * (q + 1) - 1

-- Prove that the number of distinct possible values of the expression is 27
theorem number_of_distinct_values :
  finset.card
    (finset.image (λ p q : ℕ, expression p q)
      (finset.product (finset.filter (λ x : ℕ, x ∈ odd_integers_less_15) finset.univ)
                      (finset.filter (λ y : ℕ, y ∈ odd_integers_less_15) finset.univ))) = 27 :=
by
  sorry

end number_of_distinct_values_l262_262256


namespace reading_pages_distribution_l262_262578

theorem reading_pages_distribution :
  ∃ (pages_alice pages_bob pages_chandra : ℕ),
    pages_alice + pages_bob + pages_chandra = 1000 ∧
    (pages_alice * 10 = pages_bob * 50 ∧ pages_bob * 50 = pages_chandra * 25) ∧
    pages_alice = 625 ∧ pages_bob = 125 ∧ pages_chandra = 250 :=
by
  use 625, 125, 250
  simp
  exact ⟨rfl, ⟨rfl, rfl, rfl⟩⟩

end reading_pages_distribution_l262_262578


namespace simplify_expression_correct_l262_262093

def simplify_expression (x : ℝ) : Prop :=
  (5 - 2 * x) - (7 + 3 * x) = -2 - 5 * x

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
  by
    sorry

end simplify_expression_correct_l262_262093


namespace cyclic_quadrilateral_AD_correct_l262_262635

noncomputable def cyclic_quadrilateral_AD_length : ℝ :=
  let R := 200 * Real.sqrt 2
  let AB := 200
  let BC := 200
  let CD := 200
  let AD := 500
  sorry

theorem cyclic_quadrilateral_AD_correct (R AB BC CD AD : ℝ) (hR : R = 200 * Real.sqrt 2) 
  (hAB : AB = 200) (hBC : BC = 200) (hCD : CD = 200) : AD = 500 :=
by
  have hRABBCDC: R = 200 * Real.sqrt 2 ∧ AB = 200 ∧ BC = 200 ∧ CD = 200 := ⟨hR, hAB, hBC, hCD⟩
  sorry

end cyclic_quadrilateral_AD_correct_l262_262635


namespace minimum_value_expression_l262_262412

theorem minimum_value_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 :=
by
  sorry

end minimum_value_expression_l262_262412


namespace quadratic_function_properties_l262_262316

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_function_properties (b c a : ℝ) :
  (∀ x : ℝ, f x b c = f (2 - x) b c) 
  ∧ (f 3 b c = 0) 
  ∧ (∀ x : ℝ, 3 * x ≤ 4) 
  → ((b = -2) ∧ (c = -3) 
  ∧ (∀ x : ℝ, g x b c a = f (a ^ x) b c → (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), g x b c a ≤ 5)
  ∧ (a = 4))) :=
begin
  sorry
end

end quadratic_function_properties_l262_262316


namespace simplify_complex_expression_l262_262094

theorem simplify_complex_expression :
  ( let z := (1 + complex.i * real.sqrt 7) / 3
    let conj_z := (1 - complex.i * real.sqrt 7) / 3
  in z^7 + conj_z^7 ) = 
  (256 * real.sqrt 8 * real.cos (7 * real.atan (real.sqrt 7))) / 2187 := by
  sorry

end simplify_complex_expression_l262_262094


namespace find_sin_of_smallest_acute_angle_l262_262416

noncomputable def smallest_acute_angle (θ : ℝ) : Prop :=
θ > 0 ∧ θ < (Real.pi / 2) ∧
(∃ a b c : ℝ, a, b, c ∈ {Real.cos θ, Real.cos (2 * θ), Real.cos (3 * θ)} ∧ b - a = c - b)

theorem find_sin_of_smallest_acute_angle (θ : ℝ) (h : smallest_acute_angle θ) : Real.sin θ = 0 := 
sorry

end find_sin_of_smallest_acute_angle_l262_262416


namespace optimal_pill_combination_l262_262801

theorem optimal_pill_combination
  (pills : ℕ → ℕ)
  (VitaminA_per_pill : ℕ := 50)
  (VitaminB_per_pill : ℕ := 20)
  (VitaminC_per_pill : ℕ := 10)
  (weekly_VitaminA_requirement : ℕ := 1400)
  (weekly_VitaminB_requirement : ℕ := 700)
  (weekly_VitaminC_requirement : ℕ := 280) :
  pills 35 * VitaminA_per_pill >= weekly_VitaminA_requirement ∧
  pills 35 * VitaminB_per_pill = weekly_VitaminB_requirement ∧
  pills 35 * VitaminC_per_pill >= weekly_VitaminC_requirement :=
by
  unfold pills
  sorry

end optimal_pill_combination_l262_262801


namespace binom_n_2_l262_262518

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l262_262518


namespace airplane_seats_l262_262224

theorem airplane_seats (F : ℕ) (h : F + 4 * F + 2 = 387) : F = 77 := by
  -- Proof goes here
  sorry

end airplane_seats_l262_262224


namespace friend_spent_more_l262_262922

variable (total_spent : ℕ)
variable (friend_spent : ℕ)
variable (you_spent : ℕ)

-- Conditions
axiom total_is_11 : total_spent = 11
axiom friend_is_7 : friend_spent = 7
axiom spending_relation : total_spent = friend_spent + you_spent

-- Question
theorem friend_spent_more : friend_spent - you_spent = 3 :=
by
  sorry -- Here should be the formal proof

end friend_spent_more_l262_262922


namespace sum_of_odd_div_by_3_400_to_600_l262_262900

theorem sum_of_odd_div_by_3_400_to_600 : 
  let sequence := {n : ℕ | 400 < n ∧ n < 600 ∧ n % 2 = 1 ∧ n % 3 = 0} in
  (∑ n in sequence, n) = 16500 := 
by
  sorry

end sum_of_odd_div_by_3_400_to_600_l262_262900


namespace number_4_div_p_equals_l262_262704

-- Assume the necessary conditions
variables (p q : ℝ)
variables (h1 : 4 / q = 18) (h2 : p - q = 0.2777777777777778)

-- Define the proof problem
theorem number_4_div_p_equals (N : ℝ) (hN : 4 / p = N) : N = 8 :=
by 
  sorry

end number_4_div_p_equals_l262_262704


namespace b_geometric_l262_262317

def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

axiom a1 : a 1 = 1
axiom a_n_recurrence (n : ℕ) : a n + a (n + 1) = 1 / (3^n)
axiom b_def (n : ℕ) : b n = 3^(n - 1) * a n - 1/4

theorem b_geometric (n : ℕ) : b (n + 1) = -3 * b n := sorry

end b_geometric_l262_262317


namespace problem_statement_l262_262387

noncomputable def area_of_triangle_ABC : ℚ :=
  let BC : ℚ := 2
  let KC : ℚ := 1
  let BK : ℚ := 3 * real.sqrt 2 / 2
  let area : ℚ := 15 * real.sqrt 7 / 16
  area

theorem problem_statement : area_of_triangle_ABC = 15 * real.sqrt 7 / 16 := by
  sorry

end problem_statement_l262_262387


namespace geometric_series_common_ratio_l262_262848

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l262_262848


namespace jane_earnings_in_2_weeks_l262_262743

def chickens : ℕ := 10
def eggs_per_chicken_per_week : ℕ := 6
def price_per_dozen : ℕ := 2
def weeks : ℕ := 2

theorem jane_earnings_in_2_weeks :
  let total_eggs := chickens * eggs_per_chicken_per_week * weeks in
  let dozens := total_eggs / 12 in
  dozens * price_per_dozen = 20 :=
by
  sorry

end jane_earnings_in_2_weeks_l262_262743


namespace trajectory_of_M_is_ellipse_l262_262338

-- Conditions
def circle (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def point_A : (ℝ × ℝ) := (1, 0)
def point_Q (x y : ℝ) : Prop := circle x y
def point_M (x y : ℝ) : Prop := sorry  -- placeholder for the property of M

-- Trajectory of point M
theorem trajectory_of_M_is_ellipse :
  ∃ (x y : ℝ), point_M x y → (x^2 / 4 + y^2 / 3 = 1) :=
sorry

end trajectory_of_M_is_ellipse_l262_262338


namespace cost_of_one_dozen_pens_l262_262464

-- Define the initial conditions
def cost_pen : ℕ := 65
def cost_pencil := cost_pen / 5
def total_cost (pencils : ℕ) := 3 * cost_pen + pencils * cost_pencil

-- State the theorem
theorem cost_of_one_dozen_pens (pencils : ℕ) (h : total_cost pencils = 260) :
  12 * cost_pen = 780 :=
by
  -- Preamble to show/conclude that the proofs are given
  sorry

end cost_of_one_dozen_pens_l262_262464


namespace rationalize_denominator_l262_262079

theorem rationalize_denominator :
  ∃ (A B C : ℕ),  4 * real.root 343 (4:ℕ) / 21 = (A * real.root B (4:ℕ)) / C ∧ C > 0 ∧ B ≠ 0 ∧ ¬ (∃ p : ℕ, prime p ∧ p^4 ∣ B) ∧ (A + B + C = 368) :=
by
  use 4, 343, 21
  split
  { calc 4 * real.root 343 4 / 21 = (4 * real.root 343 4) / 21 : by rw mul_div_assoc
                               ... = (4 / 21) * real.root 343 4 : by rw [mul_comm, mul_assoc, one_mul]
                               ... = (4 * real.root 343 4) / 21 : by rw mul_comm }
  split
  { norm_num }
  split
  { norm_num }
  split
  { intro h,
    obtain ⟨p, hp, hdiv⟩ := h,
    have : ¬ prime p := nat.prime.not_dvd_one hp,
    contradiction }
  { norm_num }

end rationalize_denominator_l262_262079


namespace part_a_part_b_l262_262404

-- Part (a)
theorem part_a (AB BC CD DA : ℝ) (AC BD : ℝ) 
  (h1 : ∀ {a : ℝ}, a = 0) 
  (h2 : AC^2 + BD^2 = AB^2 + BC^2 + CD^2 + DA^2) : 
  AB^2 + CD^2 = BC^2 + DA^2 :=
sorry

-- Part (b)
theorem part_b (AB BC CD DA PQ QR RS SP : ℝ) (h1 : PQ = AB) 
  (h2 : QR = BC) (h3 : RS = CD) (h4 : SP = DA) : 
  ⊥ PR QS :=
sorry

end part_a_part_b_l262_262404


namespace remainder_division_of_product_l262_262529

theorem remainder_division_of_product
  (h1 : 1225 % 12 = 1)
  (h2 : 1227 % 12 = 3) :
  ((1225 * 1227 * 1) % 12) = 3 :=
by
  sorry

end remainder_division_of_product_l262_262529


namespace min_k_delete_increasing_or_decreasing_l262_262028

theorem min_k_delete_increasing_or_decreasing (n : ℕ) (h : 2 ≤ n) :
  ∃ k : ℕ, ∀ (seq : list ℕ), (seq.length = n) → 
  ∃ subseq : list ℕ, (subseq ⊆ seq) ∧ 
    (subseq.sorted nat.lt ∨ subseq.sorted (λ x y, y < x)) ∧ 
    subseq.length = (n - ⌈ real.sqrt n ⌉₊) :=
sorry

end min_k_delete_increasing_or_decreasing_l262_262028


namespace even_candies_on_white_cells_l262_262061

-- Define the chessboard as an 8x8 grid
def chessboard : Type := fin 8 × fin 8

-- Define a function that determines if a cell has a candy
variable (has_candy : chessboard → Prop)

-- Define conditions: there is an even number of candies in each row and column
def even_number_of_candies_in_row (i : fin 8) : Prop :=
  ∃ n, n % 2 = 0 ∧ (finset.card (finset.filter (λ j : fin 8, has_candy (i, j)) finset.univ) = n)

def even_number_of_candies_in_column (j : fin 8) : Prop :=
  ∃ n, n % 2 = 0 ∧ (finset.card (finset.filter (λ i : fin 8, has_candy (i, j)) finset.univ) = n)

-- Define the property that there is an even number of candies on all white cells
def white_cell (pos : chessboard) : Prop :=
  let (i, j) := pos in (i.val + j.val) % 2 = 1

def even_number_of_candies_on_white_cells : Prop :=
  ∃ n, n % 2 = 0 ∧
  (finset.card (finset.filter white_cell (finset.univ : finset chessboard))) = n 

-- The main theorem to prove
theorem even_candies_on_white_cells (h_row : ∀ i, even_number_of_candies_in_row has_candy i)
  (h_col : ∀ j, even_number_of_candies_in_column has_candy j) :
  even_number_of_candies_on_white_cells has_candy :=
sorry

end even_candies_on_white_cells_l262_262061


namespace garden_perimeter_l262_262972

theorem garden_perimeter
  (a b : ℝ)
  (h1: a^2 + b^2 = 225)
  (h2: a * b = 54) :
  2 * (a + b) = 2 * Real.sqrt 333 :=
by
  sorry

end garden_perimeter_l262_262972


namespace area_of_park_circle_l262_262959

noncomputable def mid_point_distance : ℝ  := 10
noncomputable def plank_length : ℝ := 15
noncomputable def distance_AC : ℝ := real.sqrt (mid_point_distance^2 + plank_length^2)
noncomputable def area_of_circle : ℝ := π * distance_AC^2

theorem area_of_park_circle :
  area_of_circle = 325 * π := 
by 
  unfold area_of_circle distance_AC plank_length mid_point_distance
  norm_num
  sorry

end area_of_park_circle_l262_262959


namespace complement_of_irreducible_proper_fraction_is_irreducible_l262_262798

theorem complement_of_irreducible_proper_fraction_is_irreducible 
  (a b : ℤ) (h0 : 0 < a) (h1 : a < b) (h2 : Int.gcd a b = 1) : Int.gcd (b - a) b = 1 :=
sorry

end complement_of_irreducible_proper_fraction_is_irreducible_l262_262798


namespace sqrt_81_eq_9_l262_262816

theorem sqrt_81_eq_9 : Real.sqrt 81 = 9 :=
by
  sorry

end sqrt_81_eq_9_l262_262816


namespace least_possible_perimeter_l262_262123

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end least_possible_perimeter_l262_262123


namespace fish_remaining_l262_262747

def fish_caught_per_hour := 7
def hours_fished := 9
def fish_lost := 15

theorem fish_remaining : 
  (fish_caught_per_hour * hours_fished - fish_lost) = 48 :=
by
  sorry

end fish_remaining_l262_262747


namespace reflection_eqn_l262_262128

theorem reflection_eqn 
  (x y : ℝ)
  (h : y = 2 * x + 3) : 
  -y = 2 * x + 3 :=
sorry

end reflection_eqn_l262_262128


namespace tax_percentage_correct_l262_262504

noncomputable def total_tshirts : ℕ := 5 * 4
noncomputable def price_after_discount : ℝ := 15 - 0.20 * 15
noncomputable def total_cost_before_tax : ℝ := total_tshirts * price_after_discount
noncomputable def total_paid : ℝ := 264
noncomputable def tax_amount : ℝ := total_paid - total_cost_before_tax
noncomputable def tax_percentage : ℝ := (tax_amount / total_cost_before_tax) * 100

theorem tax_percentage_correct : tax_percentage = 10 := 
by 
  have h1 : total_tshirts = 20 := by norm_num
  rw h1
  have h2 : price_after_discount = 12 := by norm_num
  rw h2
  have h3 : total_cost_before_tax = 240 := by norm_num
  rw h3
  have h4 : tax_amount = 24 := by norm_num
  rw h4
  have h5 : tax_percentage = 10 := by norm_num
  exact h5

end tax_percentage_correct_l262_262504


namespace isosceles_triangle_area_ratio_l262_262722

theorem isosceles_triangle_area_ratio 
  (a b : ℝ) 
  (h : a > b / 2) 
  (r : ℝ)
  (h_tri : IsoscelesTriangle a a b)
  (h_circ : r = (1 / 2 * b * real.sqrt (a^2 - (b / 2)^2)) / (a + b / 2)) :
  (π * r^2) / (1 / 2 * b * real.sqrt (a^2 - (b / 2)^2)) = 
  (π * (1 / 2 * b * real.sqrt (a^2 - (b / 2)^2) / (a + b / 2))^2) / (1 / 2 * b * real.sqrt (a^2 - (b / 2)^2)) :=
begin
  sorry
end

end isosceles_triangle_area_ratio_l262_262722


namespace measure_of_angle4_l262_262893

def angle1 := 62
def angle2 := 36
def angle3 := 24
def angle4 : ℕ := 122

theorem measure_of_angle4 (d e : ℕ) (h1 : angle1 + angle2 + angle3 + d + e = 180) (h2 : d + e = 58) :
  angle4 = 180 - (angle1 + angle2 + angle3 + d + e) :=
by
  sorry

end measure_of_angle4_l262_262893


namespace common_ratio_of_geometric_series_l262_262852

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end common_ratio_of_geometric_series_l262_262852


namespace max_consecutive_integers_sum_45_l262_262890

theorem max_consecutive_integers_sum_45 :
  ∃ N : ℕ, (∃ a : ℤ, 45 = N * a + (N * (N - 1)) / 2) ∧ N ∈ divisors 90 ∧ (∀ M : ℕ, (∃ b : ℤ, 45 = M * b + (M * (M - 1)) / 2) ∧ M ∈ divisors 90 → M ≤ N) :=
begin
  existsi 90,
  split,
  { existsi -44,
    -- Proof part omitted
    sorry },
  split,
  { -- Proof part omitted
    sorry },
  { intros M hM,
    -- Proof part omitted
    sorry }
end

end max_consecutive_integers_sum_45_l262_262890


namespace value_of_expr_l262_262181

noncomputable theory

variables {a b c : ℝ}

def expr (a b : ℝ) : ℝ := a * sqrt (1 - 1 / b^2) + b * sqrt (1 - 1 / a^2)

theorem value_of_expr (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0) 
                     (h1 : a^2 + b^2 = c) (h2 : a^2 * b^2 = c) :
  expr a b = 2 ∨ expr a b = -2 ∨ expr a b = 0 :=
by
  sorry

end value_of_expr_l262_262181


namespace triangle_ABC_is_obtuse_l262_262711

variables {A B C : ℝ}

theorem triangle_ABC_is_obtuse
  (hA_acute : 0 < A ∧ A < 90)
  (hB_acute : 0 < B ∧ B < 90)
  (h_cosA_gt_sinB : cos A > sin B) :
  90 < (180 - A - B) :=
by {
  sorry
}

end triangle_ABC_is_obtuse_l262_262711


namespace remainder_of_7n_div_4_l262_262913

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l262_262913


namespace sum_of_eccentricities_l262_262318

noncomputable def a (n : ℕ) : ℝ := if n = 1 then 1 else sqrt 3 * a (n - 1)
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i
noncomputable def e (n : ℕ) : ℝ := sqrt (1 + a n ^ 2)

theorem sum_of_eccentricities (n : ℕ) (h_q_pos : (sqrt 3) > 0) (h_n_pos : n > 0) :
  (∑ i in Finset.range n, (e i) ^ 2) = n + 1/2 * (3 ^ n - 1) :=
sorry

end sum_of_eccentricities_l262_262318


namespace real_imaginary_part_above_x_axis_polynomial_solutions_l262_262555

-- Question 1: For what values of the real number m is (m^2 - 2m - 15) > 0
theorem real_imaginary_part_above_x_axis (m : ℝ) : 
  (m^2 - 2 * m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

-- Question 2: For what values of the real number m does 2m^2 + 3m - 4=0?
theorem polynomial_solutions (m : ℝ) : 
  (2 * m^2 + 3 * m - 4 = 0) ↔ (m = -3 ∨ m = 2) :=
sorry

end real_imaginary_part_above_x_axis_polynomial_solutions_l262_262555


namespace smallest_m_solutions_l262_262761

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

noncomputable def g (x : ℝ) : ℝ := abs (3 * (fractional_part x) - 1.5)

def num_real_solutions (m : ℕ) : ℕ :=
  -- This is a placeholder for the function counting solutions to mg(xg(x)) = x
  sorry

theorem smallest_m_solutions (m : ℕ) (h : num_real_solutions 39 >= 3000) : m = 39 :=
  sorry

end smallest_m_solutions_l262_262761


namespace total_steps_walked_l262_262048

theorem total_steps_walked (d_mabel : ℕ) (d_helen : ℕ) (h1 : d_mabel = 4500) (h2 : d_helen = 3 * d_mabel / 4) : 
  d_mabel + d_helen = 7875 :=
by
  rw [h1, h2]
  have : 3 * 4500 / 4 = 3375 := by norm_num
  rw this
  norm_num
  sorry

end total_steps_walked_l262_262048


namespace isosceles_triangle_largest_angle_l262_262721

theorem isosceles_triangle_largest_angle (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = 50) :
  A + B + C = 180 →
  C = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l262_262721


namespace max_consecutive_integers_sum_45_l262_262891

theorem max_consecutive_integers_sum_45 :
  ∃ N : ℕ, (∃ a : ℤ, 45 = N * a + (N * (N - 1)) / 2) ∧ N ∈ divisors 90 ∧ (∀ M : ℕ, (∃ b : ℤ, 45 = M * b + (M * (M - 1)) / 2) ∧ M ∈ divisors 90 → M ≤ N) :=
begin
  existsi 90,
  split,
  { existsi -44,
    -- Proof part omitted
    sorry },
  split,
  { -- Proof part omitted
    sorry },
  { intros M hM,
    -- Proof part omitted
    sorry }
end

end max_consecutive_integers_sum_45_l262_262891


namespace parabola_directrix_tangent_circle_l262_262334

theorem parabola_directrix_tangent_circle (p : ℝ) (h_pos : 0 < p) (h_tangent: ∃ x : ℝ, (x = p/2) ∧ (x-5)^2 + (0:ℝ)^2 = 25) : p = 20 :=
sorry

end parabola_directrix_tangent_circle_l262_262334


namespace solution_set_inequality_l262_262487

theorem solution_set_inequality : 
  {x : ℝ | 2^(x - 3/x + 1) ≤ 1/2} = {x : ℝ | x ≤ -3 ∨ (0 < x ∧ x ≤ 1)} := 
by
  sorry

end solution_set_inequality_l262_262487


namespace slope_of_given_line_eq_l262_262607

theorem slope_of_given_line_eq : (∀ x y : ℝ, (4 / x + 5 / y = 0) → (x ≠ 0 ∧ y ≠ 0) → ∀ y x : ℝ, y = - (5 * x / 4) → ∃ m, m = -5/4) :=
by
  sorry

end slope_of_given_line_eq_l262_262607


namespace main_theorem_l262_262648

noncomputable def distance_from_M_to_yaxis
  (x y : ℝ)
  (h_ellipse : x^2 / 4 + y^2 = 1)
  (h_dot_product : (-√3 - x) * (√3 - x) + (-y) * (-y) = 0) : ℝ :=
|x|

theorem main_theorem : 
  ∀ (x y : ℝ)
  (h_ellipse : x^2 / 4 + y^2 = 1)
  (h_dot_product : (-√3 - x) * (√3 - x) + (-y) * (-y) = 0),
  distance_from_M_to_yaxis x y h_ellipse h_dot_product = 2 * √6 / 3 :=
by
  sorry

end main_theorem_l262_262648


namespace minimum_value_l262_262665

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (∃ (m : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → m ≤ (y / x + 1 / y)) ∧
   m = 3 ∧ (∀ (x : ℝ), 0 < x → 0 < (1 - x) → (1 - x) + x = 1 → (y / x + 1 / y = m) ↔ x = 1 / 2)) :=
by
  sorry

end minimum_value_l262_262665


namespace common_ratio_of_geometric_series_l262_262834

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l262_262834


namespace shortest_side_is_5_l262_262475

-- Define the input conditions
def triangle_sides (a b c : ℕ) : Prop :=
  a + b + c = 42 ∧ a = 18 ∧ b + c = 24

def is_integer_area (a b c : ℕ) : Prop :=
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c) : ℝ)
  A.den = 0

-- The main theorem statement
theorem shortest_side_is_5 (b c : ℕ) (h_sides : triangle_sides 18 b c)
  (h_area : is_integer_area 18 b c) : b = 5 :=
sorry

end shortest_side_is_5_l262_262475


namespace correct_operation_l262_262167

theorem correct_operation (x : ℝ) :
  2 * x^4 / x^3 = 2 * x :=
begin
  have div_rule := (2 * x^4) / x^3 = 2 * (x^4 / x^3),
  have pow_rule : x^4 / x^3 = x^(4 - 3),
  rw [div_rule, pow_rule],
  norm_num,
end

end correct_operation_l262_262167


namespace paulsons_income_increase_percentage_l262_262792

-- Definitions of conditions
variables {I : ℝ} -- Original income
def E := 0.75 * I -- Original expenditure
def S := I - E -- Original savings

-- New income defined by percentage increase
variables {x : ℝ} -- percentage increase in income in fraction (0.20 for 20%)
def I_new := I * (1 + x)

-- New expenditure
def E_new := E * 1.10

-- New savings
def S_new := I_new - E_new

-- Percentage increase in savings given as 49.99999999999996% ~= 50%
def percentage_increase_in_savings := 0.50

-- New savings in terms of original savings
def S_new_from_percentage := S * (1 + percentage_increase_in_savings)

-- Theorem to prove that the percentage increase in income is 20%
theorem paulsons_income_increase_percentage (h : S_new = S_new_from_percentage) : x = 0.20 :=
by sorry

end paulsons_income_increase_percentage_l262_262792


namespace sum_of_squares_of_roots_l262_262481

theorem sum_of_squares_of_roots :
  ∃ x1 x2 : ℝ, (10 * x1 ^ 2 + 15 * x1 - 20 = 0) ∧ (10 * x2 ^ 2 + 15 * x2 - 20 = 0) ∧ (x1 ≠ x2) ∧ x1^2 + x2^2 = 25/4 :=
sorry

end sum_of_squares_of_roots_l262_262481


namespace original_radius_is_correct_l262_262017

-- Define the conditions as Lean definitions
def height : ℝ := 4

def volume (r h : ℝ) : ℝ := π * r^2 * h

def volume_increase_by_radius (r : ℝ) : ℝ := 
  volume (r + 4) height - volume r height

def volume_increase_by_height (r : ℝ) : ℝ := 
  volume r (height + 4) - volume r height

def y (r : ℝ) : ℝ := volume_increase_by_radius r

-- The main statement translating the proof problem
theorem original_radius_is_correct (r : ℝ) (h_eq : height = 4)
  (vol_eq1 : volume_increase_by_radius r = y r)
  (vol_eq2 : volume_increase_by_height r = y r) :
  r = 2 + 2 * Real.sqrt 2 :=
  sorry

end original_radius_is_correct_l262_262017


namespace vector_magnitude_bounded_l262_262710

noncomputable def vector_magnitude_range : Prop :=
  ∀ (a b : ℝ × ℝ), (∥a∥ = 2) ∧ ((2 • a + b) • b = 12) → (2 ≤ ∥b∥ ∧ ∥b∥ ≤ 6)

theorem vector_magnitude_bounded : vector_magnitude_range :=
  sorry

end vector_magnitude_bounded_l262_262710


namespace find_y_l262_262636

def B (T : List ℝ) : List ℝ :=
  match T with
  | [] => []
  | [_] => []
  | [_, _] => []
  | x₁::x₂::x₃::xs => (x₁ + x₂ + x₃) / 3 :: B (x₂::x₃::xs)

def B_m (m : ℕ) (T : List ℝ) : List ℝ :=
  match m with
  | 0 => T
  | n + 1 => B (B_m n T)

theorem find_y (y : ℝ) (hy : y > 0) :
  let T := List.range 51 |>.map (λ i => y ^ i)
  in B_m 49 T = [1 / 3 ^ 24] → y = Math.sqrt 3 - 1 :=
by
  intro hT
  sorry

end find_y_l262_262636


namespace arrangements_not_adjacent_l262_262874

theorem arrangements_not_adjacent (n : ℕ) (h : n ≥ 3) :
  ∃ (A B : ℕ), let total_arrangements := (n-1)!
              let adjacent_arrangements := 2 * (n-2)!
              let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  in non_adjacent_arrangements = (n-3) * (n-2)! :=
sorry

end arrangements_not_adjacent_l262_262874


namespace union_A_B_l262_262041

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_A_B : A ∪ B = {1, 2, 3} := 
by
  sorry

end union_A_B_l262_262041


namespace parabola_equation_l262_262621

theorem parabola_equation :
  (∃ h k : ℝ, h^2 = 3 ∧ k^2 = 6) →
  (∃ c : ℝ, c^2 = (3 + 6)) →
  (∃ x y : ℝ, x = 3 ∧ y = 0) →
  (y^2 = 12 * x) :=
sorry

end parabola_equation_l262_262621


namespace image_of_planes_intersecting_l262_262443

-- Define planes \alpha and \beta
variables (α β : Type) 

-- Define a line L such that the line is parallel to the projecting line
variable (L : Type)
variable [line_projecting : is_parallel_to_projecting_line L]

-- Define a projection of planes \alpha and \beta
def project_image_of_planes (α β : Type) (L : Type) [is_parallel_to_projecting_line L] : Type := sorry

-- Theorem stating that the image of the intersecting planes can be represented as two intersecting lines at an angle θ
theorem image_of_planes_intersecting (α β : Type) (L : Type) [is_parallel_to_projecting_line L] : 
  exists (θ : ℝ), project_image_of_planes α β L ≈ two_intersecting_lines θ := 
sorry

end image_of_planes_intersecting_l262_262443


namespace combination_n_2_l262_262520

theorem combination_n_2 (n : ℕ) (h : n > 0) : 
  nat.choose n 2 = n * (n - 1) / 2 :=
sorry

end combination_n_2_l262_262520


namespace distinct_arrangements_l262_262361

-- Definitions based on the conditions
def boys : ℕ := 4
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def arrangements : ℕ := Nat.factorial boys * Nat.factorial (total_people - 2) * Nat.factorial 6

-- Main statement: Verify the number of distinct arrangements
theorem distinct_arrangements : arrangements = 8640 := by
  -- We will replace this proof with our Lean steps (which is currently omitted)
  sorry

end distinct_arrangements_l262_262361


namespace test_score_estimation_l262_262502

theorem test_score_estimation
  (μ : ℝ)
  (σ : ℝ)
  (hμ : μ = 78)
  (hσ : σ = 4)
  (h1 : ∀ X : ℝ, P(μ - σ < X ∧ X ≤ μ + σ) = 0.683)
  (h2 : ∀ X : ℝ, P(μ - 2 * σ < X ∧ X ≤ μ + 2 * σ) = 0.954)
  (h3 : ∀ X : ℝ, P(μ - 3 * σ < X ∧ X ≤ μ + 3 * σ) = 0.997) :
  P(λ X : ℝ, X ≥ 90) = 0.0013 :=
by sorry

end test_score_estimation_l262_262502


namespace ram_efficiency_eq_27_l262_262076

theorem ram_efficiency_eq_27 (R : ℕ) (h1 : ∀ Krish, 2 * (1 / (R : ℝ)) = 1 / Krish) 
  (h2 : ∀ s, 3 * (1 / (R : ℝ)) * s = 1 ↔ s = (9 : ℝ)) : R = 27 :=
sorry

end ram_efficiency_eq_27_l262_262076


namespace z_in_first_quadrant_l262_262668

-- Define the complex number z that satisfies the given condition
def z : ℂ := complex.mk 1 2

-- The key condition that z * i = -2 + i
def condition : Prop := z * complex.I = complex.mk (-2) 1

-- The goal is to prove that the point (1, 2) is in the first quadrant
theorem z_in_first_quadrant : condition → (z.re > 0 ∧ z.im > 0) :=
by
  intro h
  -- The proof would go here
  sorry

end z_in_first_quadrant_l262_262668


namespace common_ratio_of_geometric_series_l262_262838

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l262_262838


namespace colored_paper_distribution_l262_262146

theorem colored_paper_distribution (F M : ℕ) (h1 : F + M = 24) (h2 : M = 2 * F) (total_sheets : ℕ) (distributed_sheets : total_sheets = 48) : 
  (48 / F) = 6 := by
  sorry

end colored_paper_distribution_l262_262146


namespace solve_for_y_l262_262699

theorem solve_for_y (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end solve_for_y_l262_262699


namespace equation_of_line_l_symmetry_A_D_l262_262645

theorem equation_of_line_l 
    (C : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)
    (P : (1, -1 / 2))
    (A B : ℝ × ℝ)
    (midpoint_AB_P : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1 / 2)
    (non_zero_slope : ∃ m : ℝ, (A.1 ≠ B.1 ∨ A.2 ≠ B.2) → m ≠ 0) :
    3 * (B.1 - A.1) - 2 * (B.2 - A.2) - 4 = 0 :=
sorry

theorem symmetry_A_D
    (C : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)
    (l : 3 * x - 2 * y - 4 = 0)
    (Q : (4, 0) ∈ l)
    (A B D F : ℝ × ℝ)
    (line_BF_intersects_C_at_D : ∃ m_BF : ℝ, line_through (B, F) ∩ C = {B, D} ∧ m_BF ≠ 0)
    (between_A_B_Q : between A B Q)
    (symmetric_wrt_x_axis : ∀ A D, A.y = -D.y ∧ A.x = D.x → symmetric A D) :
    symmetric (A, D) :=
sorry

end equation_of_line_l_symmetry_A_D_l262_262645


namespace cubic_roots_arithmetic_progression_l262_262268

theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x : ℝ, x^3 + a * x^2 + b * x + c = 0) ∧ 
  (∀ x : ℝ, x^3 + a * x^2 + b * x + c = 0 → 
    (x = p - t ∨ x = p ∨ x = p + t) ∧ 
    (a ≠ 0)) ↔ 
  ((a * b / 3) - 2 * (a^3) / 27 - c = 0 ∧ (a^3 / 3) - b ≥ 0) := 
by sorry

end cubic_roots_arithmetic_progression_l262_262268


namespace athena_total_spent_l262_262735

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end athena_total_spent_l262_262735


namespace bisect_angle_l262_262990

noncomputable theory

-- Define the points and angles
variables (P A B C D Q : point)
variable circle : set point
variable tangent : point → point → Prop
variable secant : point → point → point → point → Prop
variable cyclic_quadrilateral : point → point → point → point → Prop

-- Assume the conditions
axiom condition1 : ¬P ∈ circle
axiom condition2 : tangent P A ∧ tangent P B
axiom condition3 : secant P C D ∧ C ∈ segment P D
axiom condition4 : angle_eq Q A ∘ angle_eq D A = angle_eq P B ∘ angle_eq C

-- Define what needs to be proved
theorem bisect_angle (P A B C D Q : point) (circle : set point)
    [tangent P A] [tangent P B] [secant P C D] 
    (h1 : cyclic_quadrilateral A B C D) 
    (h2 : angle_eq Q A D A = angle_eq P B C) :
  bisects (QP) (angle A Q B) :=
sorry

end bisect_angle_l262_262990


namespace num_pigs_on_farm_l262_262693

variables (P : ℕ)
def cows := 2 * P - 3
def goats := (2 * P - 3) + 6
def total_animals := P + cows P + goats P

theorem num_pigs_on_farm (h : total_animals P = 50) : P = 10 :=
sorry

end num_pigs_on_farm_l262_262693


namespace complex_product_conjugate_l262_262039

-- Definition of a complex number and its conjugate
def is_complex_number (z : ℂ) : Prop := true

-- Given condition
def magnitude_eight (z : ℂ) : Prop := abs z = 8

-- Main theorem to prove
theorem complex_product_conjugate (z : ℂ) (h1 : is_complex_number z) (h2 : magnitude_eight z) : z * conj z = 64 :=
by
  sorry

end complex_product_conjugate_l262_262039


namespace geometric_series_ratio_l262_262862

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l262_262862


namespace distance_on_wednesday_l262_262584

theorem distance_on_wednesday 
  (d_Mon d_Tue : ℕ) 
  (avg_distance_per_day : ℕ) :
  d_Mon = 12 → 
  d_Tue = 18 → 
  avg_distance_per_day = 17 → 
  ∃ d_Wed : ℕ, (d_Mon + d_Tue + d_Wed) / 3 = avg_distance_per_day ∧ d_Wed = 21 :=
by
  intros h0 h1 h2
  use 21
  split
  sorry

end distance_on_wednesday_l262_262584


namespace normal_distribution_prob_l262_262830

/-- Given a random variable ξ that follows a normal distribution N(1, σ²) 
    and P(ξ < 2) = 0.8, prove that P(0 < ξ < 1) = 0.3. -/
theorem normal_distribution_prob (ξ : ℝ → ℝ) (σ : ℝ) (h_normal : ∀ x, ξ x = x) 
  (hx : ∫ x in -∞..∞, (ξ x - 1) ^ 2 * exp (- (ξ x - 1)^2 / (2 * σ^2)) = 1)
  (hP2 : ∫ x in -∞..2, (1 / (σ * sqrt (2 * π))) * exp (- (x - 1)^2 / (2 * σ^2)) = 0.8) :
  ∫ x in 0..1, (1 / (σ * sqrt (2 * π))) * exp (- (x - 1)^2 / (2 * σ^2)) = 0.3 :=
sorry

end normal_distribution_prob_l262_262830


namespace tulip_gift_ways_l262_262441

theorem tulip_gift_ways (rubles : ℕ) (cost_per_tulip : ℕ) (num_shades : ℕ) 
  (max_tulips : ℕ) (ways : ℕ) :
  rubles = 1000 →
  cost_per_tulip = 49 →
  num_shades = 20 →
  max_tulips = 20 →
  ways = 2^(num_shades - 1) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end tulip_gift_ways_l262_262441


namespace no_real_solutions_l262_262455

theorem no_real_solutions : ¬ ∃ (r s : ℝ),
  (r - 50) / 3 = (s - 2 * r) / 4 ∧
  r^2 + 3 * s = 50 :=
by {
  -- sorry, proof steps would go here
  sorry
}

end no_real_solutions_l262_262455


namespace continuity_at_3_l262_262772

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 3 then x^3 - 2 * x^2 - 3 else b * x + 7

theorem continuity_at_3 (b : ℝ) : 
  (∀ x, x ≤ 3 → f x b = x^3 - 2 * x^2 - 3) ∧ 
  (∀ x, x > 3 → f x b = b * x + 7) → 
  (continuous_at (λ x, f x b) 3 ↔ b = -1/3) :=
by
  intros h
  sorry

end continuity_at_3_l262_262772


namespace base5_division_l262_262615

/- Definitions of conversion functions for the problem -/
def base5_to_base10 (n : ℕ) : ℕ := 
  -- This is a placeholder definition,
  -- actual implementation will convert base 5 number to base 10
  sorry

def base10_to_base5 (n : ℕ) : ℕ := 
  -- This is a placeholder definition,
  -- actual implementation will convert base 10 number to base 5
  sorry

/- Problem statement -/
theorem base5_division (a b result : ℕ) 
  (h1 : base5_to_base10 a = 334)
  (h2 : base5_to_base10 b = 11)
  (h3 : base10_to_base5 (334 / 11) = result) : 
  result = 110 := 
sorry

/- Instantiating the given problem into the theorem -/
example : base5_division 2314 21 110 
  (by norm_num)  -- h1: 2314_5 = 334
  (by norm_num)  -- h2: 21_5 = 11
  (by norm_num)  -- h3: 334 / 11 = 30 & 30 in base 5 is 110
  := by norm_num

end base5_division_l262_262615


namespace velocity_zero_at_1_and_2_l262_262201

def displacement (t : ℝ) : ℝ := (1 / 3) * t^3 - (3 / 2) * t^2 + 2 * t

def velocity (t : ℝ) : ℝ := derivative displacement t

theorem velocity_zero_at_1_and_2 :
  (velocity 1 = 0) ∧ (velocity 2 = 0) :=
by
  -- Placeholder for proof steps
  sorry

end velocity_zero_at_1_and_2_l262_262201


namespace trajectory_of_P_l262_262639

open Set

def point := ℝ × ℝ

def f₁ : point := (-5, 0)
def f₂ : point := (5, 0)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_trajectory (P : point) : Prop :=
  Real.abs (distance P f₁ - distance P f₂) = 6

theorem trajectory_of_P (P : point) (h : is_trajectory P) :
  ∃ x y, (P = (x, y) ∧ (x^2 / 9 - y^2 / 16 = 1 ∧ x ≥ 3)) :=
sorry

end trajectory_of_P_l262_262639


namespace min_value_a_div_b_plus_b_div_c_plus_c_div_a_l262_262038

noncomputable theory

theorem min_value_a_div_b_plus_b_div_c_plus_c_div_a
  (a b c : ℝ)
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h2 : ∃ x : ℝ, a * x ^ 3 + b * x + c = 0 ∧ b * x ^ 3 + c * x + a = 0 ∧ c * x ^ 3 + a * x + b = 0)
  (h3 : ∃ x : ℝ, ∀ Q : (ℝ × ℝ × ℝ), let (k, m, n) := Q in k * x ^ 3 + m * x + n = 0 → (k ≠ a ∧ m ≠ a ∧ n ≠ a → k * x ^ 3 + (a - k) * x + n = 0 ∨ k * x ^ 3 + m * x + (a - n) = 0)) :
  (a / b) + (b / c) + (c / a) ≥ 17/12 :=
by 
  sorry

end min_value_a_div_b_plus_b_div_c_plus_c_div_a_l262_262038


namespace prime_pow_minus_one_has_factor_congruent_one_mod_p_l262_262448

theorem prime_pow_minus_one_has_factor_congruent_one_mod_p
  (p : ℕ) (hp : p.prime) : ∃ q : ℕ, q.prime ∧ q ∣ p^p - 1 ∧ q ≡ 1 [MOD p] := 
by
  sorry

end prime_pow_minus_one_has_factor_congruent_one_mod_p_l262_262448


namespace league_games_count_l262_262105

theorem league_games_count :
  let num_divisions := 2
  let teams_per_division := 9
  let intra_division_games (teams_per_div : ℕ) := (teams_per_div * (teams_per_div - 1) / 2) * 3
  let inter_division_games (teams_per_div : ℕ) (num_div : ℕ) := teams_per_div * teams_per_div * 2
  intra_division_games teams_per_division * num_divisions + inter_division_games teams_per_division num_divisions = 378 :=
by
  sorry

end league_games_count_l262_262105


namespace roots_multisets_count_l262_262462

theorem roots_multisets_count :
  let s : Fin 8 → ℤ := [s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈]
  in (∀ i, s i ∈ {-1, 0, 1}) ∧ (∑ i, s i = 0) →
  (∃ T : Finset (Fin 8 → ℤ), T.card = 9) :=
begin
  sorry -- Proof goes here
end

end roots_multisets_count_l262_262462


namespace happy_children_count_l262_262783

theorem happy_children_count
  (total children : ℕ)
  (sad children : ℕ)
  (neither happy nor sad children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy boys : ℕ)
  (sad girls : ℕ)
  (boys neither happy nor sad : ℕ) : 
  children = 60 →
  sad children = 10 →
  neither happy nor sad children = 20 →
  boys = 16 →
  girls = 44 →
  happy boys = 6 →
  sad girls = 4 →
  boys neither happy nor sad = 4 →
  children - neither happy nor sad children - sad children = 30 :=
by {
  intro h1, intro h2, intro h3, intro h4, intro h5, intro h6, intro h7, intro h8,
  rw h1, rw h2, rw h3,
  linarith,
  sorry
}

end happy_children_count_l262_262783


namespace problem_statement_l262_262429

noncomputable def a_n (n : ℕ) : ℕ := (n.choose 2) * 3^(n-2)

theorem problem_statement (h : ∀ n, n ≥ 2 → n ∈ ℕ) :
  ∀ n ≥ 2, ∑ k in finset.range 2008 \ + 2, (3^k / a_n k) = 18 := by
  sorry

end problem_statement_l262_262429


namespace smallest_possible_positive_sum_is_12_l262_262278

noncomputable def smallest_possible_positive_sum : ℤ :=
  let b : Fin 97 → ℤ := sorry -- We omit the exact construction of b_i here
  ∑ i j in Finset.univ.off_diag, b i * b j

theorem smallest_possible_positive_sum_is_12 : smallest_possible_positive_sum = 12 :=
sorry

end smallest_possible_positive_sum_is_12_l262_262278


namespace transformation_matrix_dilation_reflection_l262_262289

theorem transformation_matrix_dilation_reflection :
  ∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
    M = ![
        [-2, 0],
        [0, -1]
      ] ∧
      (∀ (v : Vector (Fin 2) ℝ), M.mulVec v = dilation_then_reflection v) :=
sorry

end transformation_matrix_dilation_reflection_l262_262289


namespace mutually_exclusive_heads_not_heads_l262_262204

open Finset

-- Conditions
def flips := {("H", "H"), ("H", "T"), ("T", "H"), ("T", "T")}

def event_heads_at_least_once := {("H", "H"), ("H", "T"), ("T", "H")}

def mutually_exclusive_event (A B : Finset (String × String)) := 
  (A ∩ B) = ∅

-- Question to Prove
theorem mutually_exclusive_heads_not_heads :
  ∀ A B, A = event_heads_at_least_once → B = (singleton ("T", "T")) → 
    mutually_exclusive_event A B :=
by
  intros A B hA hB
  rw [hA, hB]
  sorry

end mutually_exclusive_heads_not_heads_l262_262204


namespace max_packages_delivered_l262_262053

/-- Max's performance conditions and delivery problem -/
theorem max_packages_delivered
  (max_daily_capacity : ℕ) (num_days : ℕ)
  (days_max_performance : ℕ) (max_deliveries_days1 : ℕ)
  (days_half_performance : ℕ) (half_deliveries_days2 : ℕ)
  (days_fraction_performance : ℕ) (fraction_deliveries_days3 : ℕ)
  (last_two_days_fraction : ℕ) (fraction_last_two_days : ℕ):
  ∀ (remaining_capacity : ℕ), remaining_capacity = 
  max_daily_capacity * num_days - 
  (days_max_performance * max_deliveries_days1 + 
  days_half_performance * half_deliveries_days2 + 
  days_fraction_performance * fraction_deliveries_days3 * (1/7) + 
  last_two_days_fraction * fraction_last_two_days * (4/5)) := sorry

#eval max_packages_delivered 35 7 2 35 2 50 1 35 (2 * 28)

end max_packages_delivered_l262_262053


namespace b_minus_c_eq_neg_log5005_3876_l262_262296

noncomputable def a (n : ℕ) (h : 1 < n) := 1 / Real.logBase 5005 n

noncomputable def b := a 3 (by norm_num) + a 4 (by norm_num) + a 5 (by norm_num) + a 6 (by norm_num)
noncomputable def c := a 15 (by norm_num) + a 16 (by norm_num) + a 17 (by norm_num) + a 18 (by norm_num) + a 19 (by norm_num)

theorem b_minus_c_eq_neg_log5005_3876 : b - c = -Real.logBase 5005 3876 := sorry

end b_minus_c_eq_neg_log5005_3876_l262_262296


namespace calc_128_pow_3_over_7_l262_262999

theorem calc_128_pow_3_over_7 : (128 : ℝ)^(3/7) = 8 := 
by 
  -- lemma for multiplying exponents
  have h1 : (128 : ℝ) = (2 : ℝ)^7 := by norm_num,
  -- combining the exponents
  have h2 : (2 : ℝ)^(7 * (3 / 7)) = (2 : ℝ)^3 := by ring,
  -- applying exponents
  have h3 : (2 : ℝ)^3 = 8 := by norm_num,
  -- combining all
  calc (128 : ℝ)^(3/7) = ((2 : ℝ)^7)^(3/7) : by rw h1
                     ... = (2 : ℝ)^(7 * (3 / 7)) : by rw [real.rpow_def, mul_div_cancel']
                     ... = (2 : ℝ)^3 : by rw h2
                     ... = 8 : by rw h3

-- We can replace 'sorry' with the actual proof steps once identified
-- sorry can be used as placeholder

end calc_128_pow_3_over_7_l262_262999


namespace rationalize_denominator_l262_262077

theorem rationalize_denominator (A B C : ℕ) 
  (h₁ : (4 : ℝ) / (3 * (7 : ℝ)^(1/4)) = (A * (B^(1/4) : ℝ)) / (C : ℝ)) 
  (h₂ : C > 0) 
  (h₃ : ∀ p : ℕ, p.prime → ¬ (p^4 ∣ B)) :
  A + B + C = 368 :=
sorry

end rationalize_denominator_l262_262077


namespace coefficient_of_x_squared_term_in_binomial_expansion_l262_262758

noncomputable def a : ℝ := ∫ x in 0..3, (2 * x - 1)

theorem coefficient_of_x_squared_term_in_binomial_expansion :
  a = 6 →
  let b := ∫ x in 0..3, (2 * x - 1) / (2 * x) in
  (x - b)^6 = (x - 3)^6 ∧
  ∀ x : ℝ, (6.choose 2 * (-3 : ℝ)^2 = 135) :=
by
  intro h1
  have h2 : a = 6 := h1
  rw h2 at *
  sorry

end coefficient_of_x_squared_term_in_binomial_expansion_l262_262758


namespace triangle_min_perimeter_l262_262127

theorem triangle_min_perimeter:
  ∃ x : ℤ, 27 < x ∧ x < 75 ∧ (24 + 51 + x) = 103 :=
begin
  sorry
end

end triangle_min_perimeter_l262_262127


namespace average_speed_of_car_l262_262140

/-- The car's average speed given it travels 65 km in the first hour and 45 km in the second hour. -/
theorem average_speed_of_car (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 65) (h2 : d2 = 45) (h3 : t = 2) :
  (d1 + d2) / t = 55 :=
by
  sorry

end average_speed_of_car_l262_262140


namespace count_last_digit_three_in_sequence_l262_262210

noncomputable def last_digit (n : ℕ) : ℕ :=
  (7 ^ n) % 10

theorem count_last_digit_three_in_sequence :
  (finset.range 2001).card (λ n, last_digit (n + 1) = 3) = 500 :=
sorry

end count_last_digit_three_in_sequence_l262_262210


namespace prove_determinant_l262_262433

noncomputable def problem_statement (a b c d p q r : ℝ) : Prop :=
  (∀ x : ℝ, (x - a) * (x - b) * (x - c) * (x - d) = x^4 + p * x^2 + q * x + r) →
  Matrix.det !![
    [a, b, c, d],
    [d, a, b, c],
    [c, d, a, b],
    [b, c, d, a]] = 0

theorem prove_determinant (a b c d p q r : ℝ) : problem_statement a b c d p q r :=
  by sorry

end prove_determinant_l262_262433


namespace value_of_m_l262_262663

theorem value_of_m (m : ℤ) : (|m| = 1) ∧ (m + 1 ≠ 0) → m = 1 := by
  sorry

end value_of_m_l262_262663


namespace cone_height_l262_262566

theorem cone_height (r : ℝ) (n : ℕ) (circumference : ℝ) 
  (sector_circumference : ℝ) (base_radius : ℝ) (slant_height : ℝ) 
  (h : ℝ) : 
  r = 8 →
  n = 4 →
  circumference = 2 * Real.pi * r →
  sector_circumference = circumference / n →
  base_radius = sector_circumference / (2 * Real.pi) →
  slant_height = r →
  h = Real.sqrt (slant_height^2 - base_radius^2) →
  h = 2 * Real.sqrt 15 := 
by
  intros
  sorry

end cone_height_l262_262566


namespace perpendicular_m_n_angle_m_n_l262_262661

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (k : ℝ)

def angle_between (u v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.acos ((inner u v) / (u.norm * v.norm))

noncomputable def m := 3 • a - 2 • b
noncomputable def n := 2 • a + k • b

axiom angle_ab : angle_between a b = 2 * Real.pi / 3
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 3

-- 1. Prove that if m is perpendicular to n, then k = 4/3
theorem perpendicular_m_n (h : inner m n = 0) : k = 4 / 3 := sorry

-- 2. Prove that when k = 8/3, the angle between m and n is 2π/3
theorem angle_m_n (hk : k = 8 / 3) : angle_between m n = 2 * Real.pi / 3 := sorry

end perpendicular_m_n_angle_m_n_l262_262661


namespace parliamentary_committee_l262_262611

theorem parliamentary_committee 
  (M : ℕ) (n : ℕ)
  (hM : M = 1994)
  (hn : n = 665)
  (slap : fin M → fin M)
  (unique_slap : ∀ i : fin M, ∃! j : fin M, slap i = j) :
  ∃ (committee : finset (fin M)), committee.card = n ∧ ∀ i j ∈ committee, slap i ≠ j := 
sorry

end parliamentary_committee_l262_262611


namespace gcd_sequence_l262_262754

noncomputable def p (x : ℤ) : ℤ := sorry
axiom p_int_coeffs : ∀ n : ℤ, p n ∈ ℤ
axiom p_0 : p 0 = 1
axiom p_1 : p 1 = 1

def a : ℕ → ℤ
| 0       := arbitrary_nonzero_integer
| (n + 1) := p (a n)

theorem gcd_sequence (i j : ℕ) : Nat.gcd (a i).natAbs (a j).natAbs = 1 :=
sorry

end gcd_sequence_l262_262754


namespace number_of_distinct_values_l262_262257

-- Define the set of positive odd integers less than 15
def odd_integers_less_15 : set ℕ := {1, 3, 5, 7, 9, 11, 13}

-- Define the function to calculate the given expression
def expression (p q : ℕ) : ℕ := p * q + p + q

-- Define the transformed function using Simon's Favorite Factoring Trick
def transformed_expression (p q : ℕ) : ℕ := (p + 1) * (q + 1) - 1

-- Prove that the number of distinct possible values of the expression is 27
theorem number_of_distinct_values :
  finset.card
    (finset.image (λ p q : ℕ, expression p q)
      (finset.product (finset.filter (λ x : ℕ, x ∈ odd_integers_less_15) finset.univ)
                      (finset.filter (λ y : ℕ, y ∈ odd_integers_less_15) finset.univ))) = 27 :=
by
  sorry

end number_of_distinct_values_l262_262257


namespace sum_of_seq_l262_262641

noncomputable def seq (n : ℕ) : ℤ :=
  if n = 1 then -2
  else if n = 2 then 3
  else seq (n-1) * 3 + 3 ^ (n-2) * (3 ^ (n - 1) - seq n)

noncomputable def sum_seq (n : ℕ) : ℤ :=
  (13 + (6 * n - 13) * 3 ^ n) / 4

theorem sum_of_seq (n : ℕ) :
  S_n = ∑ i in range n, seq i :=
  (13 + (6 * n - 13) * 3 ^ n) / 4 := sorry

end sum_of_seq_l262_262641


namespace maple_trees_initial_count_l262_262497

def initial_number_of_maple_trees (total_after_planting : ℕ) (planted_today : ℕ) : ℕ :=
  total_after_planting - planted_today

theorem maple_trees_initial_count 
  (total_after_planting : ℕ) 
  (planted_today : ℕ) 
  (h_total : total_after_planting = 64) 
  (h_planted : planted_today = 11) : 
  initial_number_of_maple_trees total_after_planting planted_today = 53 :=
by {
  have h : initial_number_of_maple_trees 64 11 = 64 - 11 := rfl,
  rw [h_total, h_planted],
  exact h,
  sorry
}

end maple_trees_initial_count_l262_262497


namespace solve_for_a_l262_262656

-- Define the function being odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)
  
theorem solve_for_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_odd_function f) 
  (h2 : ∀ x, x > 0 → f x = 2 ^ (x - a) - 1) 
  (h3 : f (-1) = 3 / 4) : a = 3 :=
sorry

end solve_for_a_l262_262656


namespace functional_equation_solution_l262_262282

theorem functional_equation_solution (f : ℚ → ℕ) :
  (∀ (x y : ℚ) (hx : 0 < x) (hy : 0 < y),
    f (x * y) * Nat.gcd (f x * f y) (f (x⁻¹) * f (y⁻¹)) = (x * y) * f (x⁻¹) * f (y⁻¹))
  → (∀ (x : ℚ) (hx : 0 < x), f x = x.num) :=
sorry

end functional_equation_solution_l262_262282


namespace arithmetic_sequence_a6_value_l262_262012

theorem arithmetic_sequence_a6_value (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_roots : ∀ x, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) :
  a 6 = -6 :=
by
  -- Definitions and given conditions would go here in a fully elaborated proof.
  sorry

end arithmetic_sequence_a6_value_l262_262012


namespace remainder_of_S_l262_262033

-- Define the set R as the set of distinct remainders when 2^n is divided by 500.
def R : Set ℕ := {m | ∃ n : ℕ, m = 2^n % 500}

-- Define the sum S of all elements in the set R.
def S : ℕ := ∑ x in R.toFinset, x

-- The theorem we need to prove: the remainder of S when divided by 500 is 499.
theorem remainder_of_S (hR : ∀ m ∈ R.toFinset, m < 500) : S % 500 = 499 :=
by
  -- (Proof omitted)
  sorry

end remainder_of_S_l262_262033


namespace total_steps_walked_l262_262046

theorem total_steps_walked (d_mabel : ℕ) (d_helen : ℕ) (h1 : d_mabel = 4500) (h2 : d_helen = 3 * d_mabel / 4) : 
  d_mabel + d_helen = 7875 :=
by
  rw [h1, h2]
  have : 3 * 4500 / 4 = 3375 := by norm_num
  rw this
  norm_num
  sorry

end total_steps_walked_l262_262046


namespace polynomial_factorization_l262_262467

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2
  = (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) :=
sorry

end polynomial_factorization_l262_262467


namespace calculate_sheep_l262_262924

-- Conditions as definitions
def cows : Nat := 24
def goats : Nat := 113
def total_animals_to_transport (groups size_per_group : Nat) : Nat := groups * size_per_group
def cows_and_goats (cows goats : Nat) : Nat := cows + goats

-- The problem statement: Calculate the number of sheep such that the total number of animals matches the target.
theorem calculate_sheep
  (groups : Nat) (size_per_group : Nat) (cows goats : Nat) (transportation_total animals_present : Nat) 
  (h1 : groups = 3) (h2 : size_per_group = 48) (h3 : cows = 24) (h4 : goats = 113) 
  (h5 : animals_present = cows + goats) (h6 : transportation_total = groups * size_per_group) :
  transportation_total - animals_present = 7 :=
by 
  -- To be proven 
  sorry

end calculate_sheep_l262_262924


namespace polygon_area_divisible_by_n_l262_262036

variables {n : ℕ} [fact (odd n)] {k : ℕ}
variables (A : fin k → ℤ × ℤ)

noncomputable def isPolygon (P : fin k → ℤ × ℤ) : Prop :=
  ∀ i : fin k, ∃ x y : ℤ, P i = (x, y)

noncomputable def isConvex (P : fin k → ℤ × ℤ) : Prop := sorry

noncomputable def onCircle (P : fin k → ℤ × ℤ) : Prop := sorry

noncomputable def sideLengthsDivisibleByN (P : fin k → ℤ × ℤ) (n : ℕ) : Prop :=
  ∀ i : fin k, ∃ a : ℤ, n ∣ (fst (P (i + 1 % k)) - fst (P i))^2 + (snd (P (i + 1 % k)) - snd (P i))^2

theorem polygon_area_divisible_by_n
  (h_convex : isConvex A) (h_integral_coords : isPolygon A) (h_on_circle : onCircle A) 
  (h_side_lengths_divisible : sideLengthsDivisibleByN A n) :
  n ∣ 2 * sorry :=
sorry

end polygon_area_divisible_by_n_l262_262036


namespace gcd_probability_l262_262988
open Nat

-- Definitions based on problem conditions
def is_uniform (n : ℕ) : Prop := 
  n ∈ { i | 1 ≤ i ∧ i ≤ fact 2023 }

def gcd_condition (n : ℕ) : Prop :=
  gcd (n^n + 50) (n + 1) = 1

def euler_phi (n : ℕ) : ℕ := 
  (1 to n).filter (λ x, gcd x n = 1).length

-- The final probability result
def final_probability : ℚ :=  \frac{265}{357}

theorem gcd_probability :
  (∀ n : ℕ, is_uniform n →  gcd_condition n) ↔ (final_probability = \frac{265}{357}) :=
sorry

end gcd_probability_l262_262988


namespace find_J_salary_l262_262931

variable (J F M A : ℝ)

theorem find_J_salary (h1 : (J + F + M + A) / 4 = 8000) (h2 : (F + M + A + 6500) / 4 = 8900) :
  J = 2900 := by
  sorry

end find_J_salary_l262_262931


namespace probability_of_selection_of_Ram_l262_262505

noncomputable def P_Ravi : ℚ := 1 / 5
noncomputable def P_Ram_and_Ravi : ℚ := 57 / 1000  -- This is the exact form of 0.05714285714285714

axiom independent_selection : ∀ (P_Ram P_Ravi : ℚ), P_Ram_and_Ravi = P_Ram * P_Ravi

theorem probability_of_selection_of_Ram (P_Ram : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ram = 2 / 7 := by
  intro h
  have h1 : P_Ram = P_Ram_and_Ravi / P_Ravi := sorry
  rw [h1, P_Ram_and_Ravi, P_Ravi]
  norm_num
  exact sorry

end probability_of_selection_of_Ram_l262_262505


namespace hypotenuse_of_right_triangle_l262_262973

theorem hypotenuse_of_right_triangle (a b : ℕ) (h : ℕ)
  (h1 : a = 15) (h2 : b = 36) (right_triangle : a^2 + b^2 = h^2) : h = 39 :=
by
  sorry

end hypotenuse_of_right_triangle_l262_262973


namespace percentage_passed_in_both_l262_262008

theorem percentage_passed_in_both (total_students : ℕ) (hindi_failed english_failed both_failed : ℕ)
  (h_failed : hindi_failed = (0.25 * total_students : ℕ))
  (e_failed : english_failed = (0.50 * total_students : ℕ))
  (b_failed : both_failed = (0.25 * total_students : ℕ)) :
  (total_students - (hindi_failed + english_failed - both_failed) = (0.50 * total_students : ℕ)) :=
by
  sorry

end percentage_passed_in_both_l262_262008


namespace area_of_30_60_90_triangle_l262_262827

-- Define the sides of a 30-60-90 triangle given the hypotenuse
def side_ratio_30_60_90 (hypotenuse : ℝ) : ℝ × ℝ × ℝ :=
  let x := hypotenuse / 2
  let y := x * sqrt 3
  (x, y, hypotenuse)

-- Prove the area calculation of the 30-60-90 triangle
theorem area_of_30_60_90_triangle (hypotenuse : ℝ) (area : ℝ)
  (h_hypotenuse : hypotenuse = 6 * sqrt 2)
  (h_area : area = 9 * sqrt 3) :
  let (x, y, _) := side_ratio_30_60_90 hypotenuse in
  (1 / 2) * x * y = area :=
by
  let (x, y, _) := side_ratio_30_60_90 hypotenuse
  have hx : x = 3 * sqrt 2 :=
    calc
      x = hypotenuse / 2 : by rw [side_ratio_30_60_90, mul_div_cancel_left]
      ... = 6 * sqrt 2 / 2 : by rw h_hypotenuse
      ... = 3 * sqrt 2 : by ring
  have hy : y = 3 * sqrt 6 :=
    calc
      y = x * sqrt 3 : by rw side_ratio_30_60_90
      ... = (3 * sqrt 2) * sqrt 3 : by rw hx
      ... = 3 * sqrt 6 : by ring
  calc
    (1 / 2) * x * y = (1 / 2) * (3 * sqrt 2) * (3 * sqrt 6) : by rw [hx, hy]
    ... = (1 / 2) * 9 * sqrt 12 : by ring
    ... = (1 / 2) * 9 * (2 * sqrt 3) : by rw sqrt_mul
    ... = (1 / 2) * 18 * sqrt 3 : by ring
    ... = 9 * sqrt 3 : by ring
  exact h_area


end area_of_30_60_90_triangle_l262_262827


namespace geometric_series_ratio_l262_262861

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l262_262861


namespace antibiotic_residual_coefficient_l262_262236

theorem antibiotic_residual_coefficient (λ : ℝ) (y : ℝ) (t : ℝ) :
  (∀ (t : ℝ), y = λ * (1 - 3^(-λ * t))) →
  y = (8/9) * λ →
  t = 8 →
  λ ≠ 0 →
  λ = 1/4 :=
by
  intros h1 h2 h3 h4
  sorry

end antibiotic_residual_coefficient_l262_262236


namespace breadth_increase_25_percent_l262_262118

variable (L B : ℝ) 

-- Conditions
def original_area := L * B
def increased_length := 1.10 * L
def increased_area := 1.375 * (original_area L B)

-- The breadth increase percentage (to be proven as 25)
def percentage_increase_breadth (p : ℝ) := 
  increased_area L B = increased_length L * (B * (1 + p/100))

-- The statement to be proven
theorem breadth_increase_25_percent : 
  percentage_increase_breadth L B 25 := 
sorry

end breadth_increase_25_percent_l262_262118


namespace college_students_count_l262_262713

theorem college_students_count (girls boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
(h_ratio : ratio_boys = 6) (h_ratio_girls : ratio_girls = 5)
(h_girls : girls = 200)
(h_boys : boys = ratio_boys * (girls / ratio_girls)) :
  boys + girls = 440 := by
  sorry

end college_students_count_l262_262713


namespace distance_between_points_l262_262820

theorem distance_between_points : abs (3 - (-2)) = 5 := 
by
  sorry

end distance_between_points_l262_262820


namespace remainder_of_7n_div_4_l262_262912

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l262_262912


namespace length_of_longest_side_l262_262129

-- Define the conditions
def medians (y₁ y₂ y₃ : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (y₁ x = x) ∧ (y₂ x = 2 * x) ∧ (y₃ x = 3 * x)

noncomputable def perimeter_triangle : ℝ := 1

-- Rewrite the proof problem in Lean 4 statement
theorem length_of_longest_side
  (m₁ m₂ m₃ : ℝ → ℝ)
  (hmedians : medians m₁ m₂ m₃)
  (hperimeter : perimeter_triangle = 1) :
    let a := 1 / (Real.sqrt 34 + 2 + Real.sqrt 58) in
    ∃ l : ℝ, l = a * Real.sqrt 58 :=
  by
    sorry

end length_of_longest_side_l262_262129


namespace quadratic_problem_l262_262775

def quadratic_condition (a b c : ℝ) :=
  let f := λ x:ℝ, a * x^2 + b * x + c in
  b = 4 * a ∧ f (-2) = a^2 ∧ f (-1) = 6

theorem quadratic_problem (a b c : ℝ) : quadratic_condition a b c → (a + c) / b = 1 / 2 := by
  intro h
  have hb : b = 4 * a := h.1
  have max_cond : (a * (-2)^2 + b * (-2) + c = a^2) := h.2.1
  have point_cond : (a * (-1)^2 + b * (-1) + c = 6) := h.2.2
  sorry

end quadratic_problem_l262_262775


namespace simplify_expr_proof_l262_262805

noncomputable def simplify_expr : ℚ :=
  (real.sqrt 2 / real.sqrt 5) * (real.sqrt 3 / real.sqrt 6) * (real.sqrt 4 / real.sqrt 8) * (real.sqrt 5 / real.sqrt 9)

theorem simplify_expr_proof : simplify_expr = 1 / 3 := by
  sorry

end simplify_expr_proof_l262_262805


namespace z_sum_in_fourth_quadrant_l262_262424

-- Define complex numbers z1 and z2
def z1 : ℂ := 3 - 4 * Complex.i
def z2 : ℂ := -2 + 3 * Complex.i

-- Define the sum of z1 and z2
def z_sum : ℂ := z1 + z2

-- Define the coordinates of z_sum
def z_sum_coordinates : ℝ × ℝ := (z_sum.re, z_sum.im)

-- Prove that the point corresponding to z_sum is in the fourth quadrant
theorem z_sum_in_fourth_quadrant : z_sum_coordinates.1 > 0 ∧ z_sum_coordinates.2 < 0 :=
by 
  -- skipping proof steps
  sorry

end z_sum_in_fourth_quadrant_l262_262424


namespace coeff_sum_l262_262357

variable (a_0 a_1 a_2 a_3 a_4 a_5 : ℚ)

def poly_expansion : (2 + x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 := sorry

theorem coeff_sum : a_1 - a_0 + a_3 - a_2 + a_5 - a_4 = -1 :=
by {
  have h := poly_expansion a_0 a_1 a_2 a_3 a_4 a_5,
  -- Substituting h result would be shown proof steps
  sorry
}

end coeff_sum_l262_262357


namespace max_abs_xk_l262_262137

-- Definitions for the conditions
variables {n : ℕ} (x : fin n → ℝ)

-- The core condition
def main_condition (x : fin n → ℝ) := 
  (finset.univ.sum (λ i : fin n, x i ^ 2)) + 
  (finset.univ.erase (fin n.last).sum (λ i : fin n, x i * x (i + 1))) = 1

-- The proof statement
theorem max_abs_xk (x : fin n → ℝ) (k : fin n) (h : main_condition x) : 
  |x k| ≤ sqrt (2 * k * (n - k + 1) / (n + 1)) :=
sorry

end max_abs_xk_l262_262137


namespace paula_aunt_money_l262_262791

theorem paula_aunt_money
  (shirts_cost : ℕ := 2 * 11)
  (pants_cost : ℕ := 13)
  (money_left : ℕ := 74) : 
  shirts_cost + pants_cost + money_left = 109 :=
by
  sorry

end paula_aunt_money_l262_262791


namespace jenna_total_profit_l262_262395

theorem jenna_total_profit
  (buy_price : ℕ)         -- $3 per widget
  (sell_price : ℕ)        -- $8 per widget
  (rent : ℕ)              -- $10,000 monthly rent
  (worker_salary : ℕ)     -- $2,500 per worker
  (num_workers : ℕ)       -- 4 workers
  (tax_rate : ℚ)          -- 20% of total profit
  (widgets_sold : ℕ)      -- 5000 widgets sold
  (total_profit : ℤ)      -- Expected total profit $4,000
  (h_buy_price : buy_price = 3)
  (h_sell_price : sell_price = 8)
  (h_rent : rent = 10000)
  (h_worker_salary : worker_salary = 2500)
  (h_num_workers : num_workers = 4)
  (h_tax_rate : tax_rate = 0.2)
  (h_widgets_sold : widgets_sold = 5000)
  (h_total_profit : total_profit = 4000) :
  let total_salaries := num_workers * worker_salary in
  let total_fixed_costs := rent + total_salaries in
  let profit_per_widget := sell_price - buy_price in
  let total_profit_from_sales := widgets_sold * profit_per_widget in
  let profit_before_taxes := total_profit_from_sales - total_fixed_costs in
  let taxes_owed := profit_before_taxes * tax_rate in
  let net_profit := profit_before_taxes - taxes_owed in
  net_profit = total_profit :=
by
  -- The proof part is not required according to the instructions
  sorry

end jenna_total_profit_l262_262395


namespace faster_train_speed_l262_262884

/- Definitions for the given conditions -/
def train_length : ℝ := 475 -- meters
def speed_slower_train : ℝ := 40 -- km/hr
def passing_time : ℝ := 36 -- seconds

/- Conversion factors -/
def dist_km : ℝ := (475 * 2) / 1000 -- km
def time_hr : ℝ := 36 / 3600 -- hours

/- Proof statement -/
theorem faster_train_speed :
  let V_f := (dist_km / time_hr) - speed_slower_train in
  V_f = 55 :=
by
  sorry

end faster_train_speed_l262_262884


namespace marks_in_math_l262_262602

theorem marks_in_math (e p c b : ℕ) (avg : ℚ) (n : ℕ) (total_marks_other_subjects : ℚ) :
  e = 45 →
  p = 52 →
  c = 47 →
  b = 55 →
  avg = 46.8 →
  n = 5 →
  total_marks_other_subjects = (e + p + c + b : ℕ) →
  (avg * n) - total_marks_other_subjects = 35 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_in_math_l262_262602


namespace spherical_shell_gravitational_attraction_l262_262554

/-- Prove that the gravitational attraction of a thin spherical shell with constant
    thickness and density at a point outside the shell is the same as the gravitational
    attraction if the entire mass were concentrated at the center of the shell. -/
theorem spherical_shell_gravitational_attraction
  (S : set Point)
  (O : Point)
  (P : Point)
  (r : ℝ)
  (M : ℝ)
  (G : ℝ)
  (h1 : ThinSphericalShell S)
  (h2 : Center S O)
  (h3 : TotalMass S M)
  (h4 : ConstantDensity S)
  (h5 : Thickness S = r)
  (h6 : Outside P S) :
  GravitationalAttraction S P = GravitationalAttraction (PointMass M O) P := sorry

end spherical_shell_gravitational_attraction_l262_262554


namespace quotient_of_larger_divided_by_smaller_l262_262476

theorem quotient_of_larger_divided_by_smaller
  (x y : ℕ)
  (h1 : x * y = 9375)
  (h2 : x + y = 400)
  (h3 : x > y) :
  x / y = 15 :=
sorry

end quotient_of_larger_divided_by_smaller_l262_262476


namespace remaining_amount_after_purchase_l262_262273

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem remaining_amount_after_purchase : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end remaining_amount_after_purchase_l262_262273


namespace aluminium_atoms_in_compound_l262_262958

def num_aluminium_atoms (num_sulfur_atoms molecular_weight atomic_weight_al atomic_weight_s : ℕ × ℕ × ℕ × ℕ) : ℕ :=
  let wt_sulfur := num_sulfur_atoms * atomic_weight_s
  let wt_aluminum := molecular_weight - wt_sulfur
  wt_aluminum / atomic_weight_al

theorem aluminium_atoms_in_compound :
  ∀ (molecular_weight : ℕ) (atomic_weight_al : ℕ) (atomic_weight_s : ℕ) (num_sulfur_atoms : ℕ),
    molecular_weight = 150 →
    atomic_weight_al = 26.98 →
    atomic_weight_s = 32.06 →
    num_sulfur_atoms = 3 →
    num_aluminium_atoms (num_sulfur_atoms, molecular_weight, atomic_weight_al, atomic_weight_s) = 2 :=
by
  sorry

end aluminium_atoms_in_compound_l262_262958


namespace fermat_little_theorem_l262_262551

theorem fermat_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a ^ p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l262_262551


namespace distinct_sums_products_under_15_l262_262259

def is_positive_odd (n : ℕ) : Prop := n % 2 = 1 ∧ n > 0

noncomputable def possible_values : ℕ → ℕ → ℕ := λ p q, p * q + p + q

theorem distinct_sums_products_under_15 : 
  {pq_sum | ∃ p q : ℕ, is_positive_odd p ∧ is_positive_odd q ∧ p < 15 ∧ q < 15 ∧ pq_sum = possible_values p q}.to_finset.card = 28 :=
sorry

end distinct_sums_products_under_15_l262_262259


namespace correct_statement_is_D_l262_262373

noncomputable def sample_size_Xiaoming := 200
noncomputable def sample_size_Xiaohua := 100

noncomputable def avg_height_Xiaoming := 166.2
noncomputable def avg_height_Xiaohua := 164.7

def is_correct_statement : Prop := 
  ∀ (option : string), option ∈ ["A", "B", "C", "D"] →
    (option = "D" ↔ 
     (∀ n : ℕ, n = sample_size_Xiaoming → avg_height_Xiaoming = 166.2) ∧
     (∀ n : ℕ, n = sample_size_Xiaohua → avg_height_Xiaohua = 164.7))

theorem correct_statement_is_D : is_correct_statement :=
sorry

end correct_statement_is_D_l262_262373


namespace infinite_solutions_xyza_l262_262073

theorem infinite_solutions_xyza (k : ℕ) : 
  let x := 2^(15*k + 10), y := 2^(6*k + 4), z := 2^(10*k + 7) in
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^5 = z^3 :=
by
  let x := 2^(15*k + 10)
  let y := 2^(6*k + 4)
  let z := 2^(10*k + 7)
  have h : 2^n ≠ 0 := pow_ne_zero n two_ne_zero
  { 
    have x_ne_zero : x ≠ 0 := h (15*k + 10)
    have y_ne_zero : y ≠ 0 := h (6*k + 4)
    have z_ne_zero : z ≠ 0 := h (10*k + 7)
    sorry
  }

end infinite_solutions_xyza_l262_262073


namespace num_lockers_is_2479_l262_262488

-- Define the total cost value
def total_cost : ℝ := 440.40

-- Define the cost per digit
def cost_per_digit : ℝ := 0.05

-- Define a function to calculate the cost of labeling lockers from 1 to n
def labeling_cost (n : ℕ) : ℝ :=
  let one_digit := min n 9
  let two_digits := if n > 9 then min (n - 9) 90 else 0
  let three_digits := if n > 99 then min (n - 99) 900 else 0
  let four_digits := if n > 999 then n - 999 else 0
  in (one_digit * 1 + two_digits * 2 + three_digits * 3 + four_digits * 4) * cost_per_digit

-- The proof objective: Prove that the number of lockers n is 2479
theorem num_lockers_is_2479 : ∃ (n : ℕ), labeling_cost n = total_cost ∧ n = 2479 :=
by
  use 2479
  -- The conditions and calculations will go here to complete the proof
  sorry

end num_lockers_is_2479_l262_262488


namespace rng_baseball_correct_expected_points_l262_262005

def rng_baseball_expected_points : ℚ :=
  -- Defining the probabilities P_i
  let P₃ := 4 / 5 in
  let P₂ := 1 / 5 * (P₃ + 3) in
  let P₁ := 1 / 5 * (P₂ + P₃ + 2) in
  let P₀ := 1 / 5 * (P₁ + P₂ + P₃ + 1) in
  -- The expected number of points
  P₀ * 5

theorem rng_baseball_correct_expected_points : 
  rng_baseball_expected_points = 409 / 125 := 
  by
  sorry

end rng_baseball_correct_expected_points_l262_262005


namespace paul_spending_l262_262787

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end paul_spending_l262_262787


namespace ivan_peter_max_distance_l262_262389

def time_to_max_distance (T1 T2 : ℕ) : ℕ :=
  let relative_speed := (1 / T1.toRat - 1 / T2.toRat).abs
  have h_relative_speed : relative_speed ≠ 0 := sorry
  (1 / relative_speed / 2).toNat

theorem ivan_peter_max_distance (T1 T2 : ℕ) (hT1 : T1 = 20) (hT2 : T2 = 28) :
  time_to_max_distance T1 T2 = 35 :=
by
  rw [hT1, hT2]
  -- Details omitted, proof steps will be filled in with algebraic simplifications and derivations
  sorry

end ivan_peter_max_distance_l262_262389


namespace same_color_probability_is_correct_l262_262697

-- Define the variables and conditions
def total_sides : ℕ := 12
def pink_sides : ℕ := 3
def green_sides : ℕ := 4
def blue_sides : ℕ := 5

-- Calculate individual probabilities
def pink_probability : ℚ := (pink_sides : ℚ) / total_sides
def green_probability : ℚ := (green_sides : ℚ) / total_sides
def blue_probability : ℚ := (blue_sides : ℚ) / total_sides

-- Calculate the probabilities that both dice show the same color
def both_pink_probability : ℚ := pink_probability ^ 2
def both_green_probability : ℚ := green_probability ^ 2
def both_blue_probability : ℚ := blue_probability ^ 2

-- The final probability that both dice come up the same color
def same_color_probability : ℚ := both_pink_probability + both_green_probability + both_blue_probability

theorem same_color_probability_is_correct : same_color_probability = 25 / 72 := by
  sorry

end same_color_probability_is_correct_l262_262697


namespace leading_coefficient_of_g_l262_262828

theorem leading_coefficient_of_g (g : ℝ → ℝ) (h : ∀ x : ℝ, g (x + 1) - g x = 8*x + 6) :
  leading_coeff g = 4 :=
sorry

end leading_coefficient_of_g_l262_262828


namespace minor_arc_MB_eq_60_degrees_l262_262015

theorem minor_arc_MB_eq_60_degrees
  (angle_MBC : ℝ)
  (h1 : angle_MBC = 60) -- Condition 1: \( \angle MBC \) measures 60 degrees
  (h2 : is_inscribed_angle angle_MBC) -- Condition 2: \( \angle MBC \) is an inscribed angle
  (h3 : subtends angle_MBC (-1,0) dir(120)) -- Condition 3: \( \angle MBC \) subtends arc \( MC \)
  (h4 : is_semicircle (-1,0) dir(120) (1,0)) -- Condition 4: Arc \( MBC \) is a semicircle
  : arc_measure (-1,0) dir(120) = 60 :=
sorry

end minor_arc_MB_eq_60_degrees_l262_262015


namespace second_player_wins_with_optimal_play_l262_262144

/-- There are 11 empty boxes. In one move, you can place one coin in any 10 of them.
Two players take turns. The winner is the one who, after their move, first places
the 21st coin in any one of the boxes. Prove that the second player wins with optimal play. -/
theorem second_player_wins_with_optimal_play :
  ∃ (strategy : ℕ → ℕ), (∀ n, 1 ≤ strategy n ∧ strategy n ≤ 11) → 
  ∃ (moves_played : ℕ → ℕ → ℕ) (coins : ℕ → ℕ),
  (∀ n, coins n < 21) → 
  (coins (moves_played 2) = 21 ∧ ∀ m, coins m < 21) :=
sorry

end second_player_wins_with_optimal_play_l262_262144


namespace smallest_c_cos_zero_at_zero_l262_262231

theorem smallest_c_cos_zero_at_zero (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (x y : ℝ) :
  (forall (x : ℝ), y = a * cos (b * x + c)) → (y = 0) → (x = 0) →
  c = π / 2 := 
sorry

end smallest_c_cos_zero_at_zero_l262_262231


namespace number_of_5card_hands_with_4_of_a_kind_l262_262960

-- Definitions based on the given conditions
def deck_size : Nat := 52
def num_values : Nat := 13
def suits_per_value : Nat := 4

-- The function to count the number of 5-card hands with exactly four cards of the same value
def count_hands_with_four_of_a_kind : Nat :=
  num_values * (deck_size - suits_per_value)

-- Proof statement
theorem number_of_5card_hands_with_4_of_a_kind : count_hands_with_four_of_a_kind = 624 :=
by
  -- Steps to show the computation results may be added here
  -- We use the formula: 13 * (52 - 4)
  sorry

end number_of_5card_hands_with_4_of_a_kind_l262_262960


namespace maximum_value_expression_l262_262892

theorem maximum_value_expression (a b c : ℕ) (ha : 0 < a ∧ a ≤ 9) (hb : 0 < b ∧ b ≤ 9) (hc : 0 < c ∧ c ≤ 9) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (v : ℚ), v = (1 / (a + 2010 / (b + 1 / c : ℚ))) ∧ v ≤ (1 / 203) :=
sorry

end maximum_value_expression_l262_262892


namespace percent_deficit_for_second_side_l262_262723

variable (L W : ℝ)
variable (x : ℝ)

-- Given conditions
def measured_length := 1.09 * L
def area_error := 0.0028 * (L * W)
def calculated_area := measured_length * (W - (x / 100) * W)

-- Proof goal
theorem percent_deficit_for_second_side (h : calculated_area = (L * W) + area_error) : x = 8 :=
by
  sorry

end percent_deficit_for_second_side_l262_262723


namespace coefficients_binomial_expansion_l262_262178

theorem coefficients_binomial_expansion : 
  (∀ (x : ℝ), (sqrt 3 * x - 1)^3 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3) → 
  (a_0 + a_2)^2 - (a_1 + a_3)^2 = -8 :=
by
  sorry

end coefficients_binomial_expansion_l262_262178


namespace volume_parallelepiped_is_half_l262_262359

-- Define the conditions: a and b are unit vectors with an angle of π/4 between them
variable (a b : ℝ^3)
variable (h_unit_a : ‖a‖ = 1)
variable (h_unit_b : ‖b‖ = 1)
variable (h_angle : real.angle a b = real.pi / 4)

-- Define the vectors generating the parallelepiped
def v1 : ℝ^3 := a
def v2 : ℝ^3 := 2 • b + vector.cross b a
def v3 : ℝ^3 := b

-- State the theorem we need to prove: the volume of the parallelepiped is 1/2
theorem volume_parallelepiped_is_half
  (a b : ℝ^3)
  (h_unit_a : ‖a‖ = 1)
  (h_unit_b : ‖b‖ = 1)
  (h_angle : real.angle a b = real.pi / 4) :
  |vector.dot_product a (vector.cross_product (2 • b + vector.cross_product b a) b)| = 1 / 2 :=
sorry

end volume_parallelepiped_is_half_l262_262359


namespace circle_tangent_to_directrix_and_yaxis_on_parabola_l262_262292

noncomputable def circle1_eq (x y : ℝ) := (x - 1)^2 + (y - 1 / 2)^2 = 1
noncomputable def circle2_eq (x y : ℝ) := (x + 1)^2 + (y - 1 / 2)^2 = 1

theorem circle_tangent_to_directrix_and_yaxis_on_parabola :
  ∀ (x y : ℝ), (x^2 = 2 * y) → 
  ((y = -1 / 2 → circle1_eq x y) ∨ (y = -1 / 2 → circle2_eq x y)) :=
by
  intro x y h_parabola
  sorry

end circle_tangent_to_directrix_and_yaxis_on_parabola_l262_262292


namespace solve_problem_l262_262941

noncomputable def problem_statement (x : ℝ) : Prop :=
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * Real.cos (3 * x / 2) ^ 2

theorem solve_problem (x : ℝ) :
  problem_statement x ↔
  (∃ k : ℤ, x = (Real.pi / 8) * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = (Real.pi / 4) * (4 * n - 1)) :=
by
  sorry

end solve_problem_l262_262941


namespace circle_parallel_lines_same_point_l262_262773

theorem circle_parallel_lines_same_point
(P1 P2 P3 P4 : Point) (circle : Circle) (h1 : P1 ∈ circle) (h2 : P2 ∈ circle) 
(h3 : P3 ∈ circle) (h4 : P4 ∈ circle) 
(h_order : clockwise_order [P1, P2, P3, P4])
(parallel1 : ∃ P5, P4P5 ∈ parallel_line_to (P1P2) ∧ P5 ∈ circle)
(parallel2 : ∃ P6, P5P6 ∈ parallel_line_to (P2P3) ∧ P6 ∈ circle)
(parallel3 : ∃ P7, P6P7 ∈ parallel_line_to (P3P4) ∧ P7 ∈ circle)
: P7 = P1 :=
by
  sorry

end circle_parallel_lines_same_point_l262_262773


namespace sum_of_digits_of_greatest_prime_divisor_l262_262531

-- Define the number 32767
def number : ℕ := 32767

-- Assert that 32767 is 2^15 - 1
lemma number_def : number = 2^15 - 1 := by
  sorry

-- State that 151 is the greatest prime divisor of 32767
lemma greatest_prime_divisor : Nat.Prime 151 ∧ ∀ p : ℕ, Nat.Prime p → p ∣ number → p ≤ 151 := by
  sorry

-- Calculate the sum of the digits of 151
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Conclude the sum of the digits of the greatest prime divisor is 7
theorem sum_of_digits_of_greatest_prime_divisor : sum_of_digits 151 = 7 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_l262_262531


namespace sequence_not_periodic_l262_262832

def sequence (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → a (2 * n) = a n) ∧
  (∀ n : ℕ, a (4 * n + 1) = 1) ∧
  (∀ n : ℕ, a (4 * n + 3) = 0)

theorem sequence_not_periodic (a : ℕ → ℕ) (h : sequence a) :
  ¬ ∃ T : ℕ, T > 0 ∧ ∀ n : ℕ, a n = a (n + T) :=
sorry

end sequence_not_periodic_l262_262832


namespace train_speed_54_kmh_l262_262215

theorem train_speed_54_kmh
  (train_length : ℕ)
  (tunnel_length : ℕ)
  (time_seconds : ℕ)
  (total_distance : ℕ := train_length + tunnel_length)
  (speed_mps : ℚ := total_distance / time_seconds)
  (conversion_factor : ℚ := 3.6) :
  train_length = 300 →
  tunnel_length = 1200 →
  time_seconds = 100 →
  speed_mps * conversion_factor = 54 := 
by
  intros h_train_length h_tunnel_length h_time_seconds
  sorry

end train_speed_54_kmh_l262_262215


namespace eval_transformed_function_at_pi_12_l262_262114

def original_function : ℝ → ℝ := λ x => Real.sin (2 * x - Real.pi / 4)
def transformed_function : ℝ → ℝ := λ x => Real.sin (2 * (x + Real.pi / 6) - Real.pi / 4)

theorem eval_transformed_function_at_pi_12 :
  transformed_function (Real.pi / 12) = Real.sqrt 2 / 2 :=
by
  sorry

end eval_transformed_function_at_pi_12_l262_262114


namespace trihedral_angle_planes_intersect_at_line_l262_262322

-- Definitions for the vertex, edges of the trihedral angle, etc.
variable {S A B C : Point}
variable (SA SB SC : Line)

-- Definitions for planes passing through points and lines; angle bisectors, etc.
variable (plane_SA_bisector : Plane)
variable (plane_SB_bisector : Plane)
variable (plane_SC_bisector : Plane)

-- Statement of the theorem
theorem trihedral_angle_planes_intersect_at_line (h1 : lies_on S SA)
                                                (h2 : lies_on S SB)
                                                (h3 : lies_on S SC)
                                                (h4 : plane_SA_bisector = Plane_through SA (angle_bisector (A, S, B)))
                                                (h5 : plane_SB_bisector = Plane_through SB (angle_bisector (B, S, C)))
                                                (h6 : plane_SC_bisector = Plane_through SC (angle_bisector (C, S, A))) :
  ∃ (L : Line), intersection_line plane_SA_bisector plane_SB_bisector plane_SC_bisector = L := sorry

end trihedral_angle_planes_intersect_at_line_l262_262322


namespace polynomial_nonzero_at_int_l262_262540

theorem polynomial_nonzero_at_int 
  (p : ℤ[X])
  (c : ℕ) 
  (hc : 0 < c) 
  (hdiv : ∀ i, 1 ≤ i ∧ i ≤ c → ¬ (c ∣ p.eval i)) :
  ∀ b : ℤ, p.eval b ≠ 0 :=
by 
  sorry

end polynomial_nonzero_at_int_l262_262540


namespace max_extra_packages_l262_262054

/-- Max's delivery performance --/
def max_daily_packages : Nat := 35

/-- (1) Max delivered the maximum number of packages on two days --/
def max_2_days : Nat := 2 * max_daily_packages

/-- (2) On two other days, Max unloaded a total of 50 packages --/
def two_days_50 : Nat := 50

/-- (3) On one day, Max delivered one-seventh of the maximum possible daily performance --/
def one_seventh_day : Nat := max_daily_packages / 7

/-- (4) On the last two days, the sum of packages was four-fifths of the maximum daily performance --/
def last_2_days : Nat := 2 * (4 * max_daily_packages / 5)

/-- (5) Total packages delivered in the week --/
def total_delivered : Nat := max_2_days + two_days_50 + one_seventh_day + last_2_days

/-- (6) Total possible packages in a week if worked at maximum performance --/
def total_possible : Nat := 7 * max_daily_packages

/-- (7) Difference between total possible and total delivered packages --/
def difference : Nat := total_possible - total_delivered

/-- Proof problem: Prove the difference is 64 --/
theorem max_extra_packages : difference = 64 := by
  sorry

end max_extra_packages_l262_262054


namespace addition_round_to_tenth_l262_262586

-- Define the numbers to be added
def num1 : ℝ := 725.9431
def num2 : ℝ := 84.379

-- Define the sum of the numbers
def sum := num1 + num2

-- Define the function to round to the nearest tenth
def round_nearest_tenth (x : ℝ) : ℝ :=
  (Real.floor (x * 10 + 0.5)) / 10

-- State the theorem to be proved
theorem addition_round_to_tenth :
  round_nearest_tenth sum = 810.3 := by
sorry

end addition_round_to_tenth_l262_262586


namespace value_of_expression_l262_262992

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 9*x2 + 25*x3 + 49*x4 + 81*x5 + 121*x6 + 169*x7 = 2)
  (h2 : 9*x1 + 25*x2 + 49*x3 + 81*x4 + 121*x5 + 169*x6 + 225*x7 = 24)
  (h3 : 25*x1 + 49*x2 + 81*x3 + 121*x4 + 169*x5 + 225*x6 + 289*x7 = 246) : 
  49*x1 + 81*x2 + 121*x3 + 169*x4 + 225*x5 + 289*x6 + 361*x7 = 668 := 
sorry

end value_of_expression_l262_262992


namespace dihedral_angles_acute_l262_262072

theorem dihedral_angles_acute (a b c : ℝ) : 
  let x := a^2 + b^2,
      y := b^2 + c^2,
      z := c^2 + a^2
  in
  x + y > z ∧ y + z > x ∧ z + x > y :=
by
  let x := a^2 + b^2
  let y := b^2 + c^2
  let z := c^2 + a^2
  sorry

end dihedral_angles_acute_l262_262072


namespace mabel_visits_helen_l262_262050

-- Define the number of steps Mabel lives from Lake High school
def MabelSteps : ℕ := 4500

-- Define the number of steps Helen lives from the school
def HelenSteps : ℕ := (3 * MabelSteps) / 4

-- Define the total number of steps Mabel will walk to visit Helen
def TotalSteps : ℕ := MabelSteps + HelenSteps

-- Prove that the total number of steps Mabel walks to visit Helen is 7875
theorem mabel_visits_helen :
  TotalSteps = 7875 :=
sorry

end mabel_visits_helen_l262_262050


namespace lily_pads_half_covered_in_57_days_l262_262929

/-- 
Given that a patch of lily pads doubles in size every day, 
and it takes 58 days to cover the entire lake, 
prove that it takes 57 days to cover half of the lake.
-/
theorem lily_pads_half_covered_in_57_days (double_size_each_day : ∀ (s : ℕ), s > 0 → 2 * s = s.next) :
  ∀ {days : ℕ}, days = 58 → ∀ (cover_entire : ℕ), (cover_entire * 2) = cover_entire -> (days - 1) = 57 :=
sorry

end lily_pads_half_covered_in_57_days_l262_262929


namespace polynomial_f_nonzero_l262_262253

-- Define the polynomial Q(x) = x^7 + ax^6 + bx^5 + cx^4 + dx^3 + ex^2 + fx + g
-- with distinct x-intercepts including a double root at x = 0
noncomputable def Q (a b c d e f g : ℝ) (x p q r s t u : ℝ) : ℝ :=
  x^7 + a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g

-- Define the problem conditions
theorem polynomial_f_nonzero (a b c d e f g : ℝ)
  (p q r s t u : ℝ) (h_dist : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ 
                              q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ 
                              r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ 
                              s ≠ t ∧ s ≠ u ∧ t ≠ u ∧ 
                              p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0 ∧ u ≠ 0) :
  let Q := λ x, x^2 * (x - p) * (x - q) * (x - r) * (x - s) * (x - t) * (x - u) in
  (∀ x, Q x = x^2 * (x - p) * (x - q) * (x - r) * (x - s) * (x - t) * (x - u)) →
  f ≠ 0 :=
sorry

end polynomial_f_nonzero_l262_262253


namespace min_value_fraction_sum_l262_262381

theorem min_value_fraction_sum
  (a : ℕ → ℝ)
  (q : ℝ)
  (hq : q > 0)
  (ha_geom : ∀ n m : ℕ, a (n + 1) = a n * q)
  (h_sqrt : ∀ m n : ℕ, sqrt (a m * a n) = 4 * a 1)
  (h_eq : a 7 = a 6 + 2 * a 5) :
  ∃ (m n : ℕ), (1 ≤ m ∧ 1 ≤ n ∧ (m + n = 6) → (1 / m + 5 / n = 7 / 4)) :=
sorry

end min_value_fraction_sum_l262_262381


namespace average_headcount_is_correct_l262_262527

/-- The student headcount data for the specified semesters -/
def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

noncomputable def average_headcount : ℕ :=
  (student_headcount.sum) / student_headcount.length

theorem average_headcount_is_correct : average_headcount = 11029 := by
  sorry

end average_headcount_is_correct_l262_262527


namespace problem_a_problem_b_l262_262184

-- Proof Problem (I)
theorem problem_a (a : ℝ) (h : a + a⁻¹ = 11) : a^(1/2) - a^(-1/2) = 3 ∨ a^(1/2) - a^(-1/2) = -3 :=
sorry

-- Proof Problem (II)
theorem problem_b (x : ℝ) (h : (Real.log2 x)^2 - 2 * Real.log2 x - 3 = 0) : x = 1/2 ∨ x = 8 :=
sorry

end problem_a_problem_b_l262_262184


namespace geometric_series_ratio_l262_262860

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l262_262860


namespace breadth_increase_l262_262115

theorem breadth_increase (L B : ℝ) (A : ℝ := L * B) 
  (L' : ℝ := 1.10 * L) (A' : ℝ := 1.375 * A) 
  (B' : ℝ := B * (1 + p / 100)) 
  (h1 : A = L * B)
  (h2 : A' = L' * B')
  (h3 : A' = 1.375 * A) 
  (h4 : L' = 1.10 * L) :
  p = 25 := 
begin 
  sorry 
end

end breadth_increase_l262_262115


namespace solve_for_n_l262_262706

theorem solve_for_n (n : ℕ) (h : 2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n) : n = 6 := by
  sorry

end solve_for_n_l262_262706


namespace Diana_total_earnings_l262_262270

def July : ℝ := 150
def August : ℝ := 3 * July
def September : ℝ := 2 * August
def October : ℝ := September + 0.1 * September
def November : ℝ := 0.95 * October
def Total_earnings : ℝ := July + August + September + October + November

theorem Diana_total_earnings : Total_earnings = 3430.50 := by
  sorry

end Diana_total_earnings_l262_262270


namespace jake_sister_weight_ratio_l262_262705

theorem jake_sister_weight_ratio
  (jake_present_weight : ℕ)
  (total_weight : ℕ)
  (weight_lost : ℕ)
  (sister_weight : ℕ)
  (jake_weight_after_loss : ℕ)
  (ratio : ℕ) :
  jake_present_weight = 188 →
  total_weight = 278 →
  weight_lost = 8 →
  jake_weight_after_loss = jake_present_weight - weight_lost →
  sister_weight = total_weight - jake_present_weight →
  ratio = jake_weight_after_loss / sister_weight →
  ratio = 2 := by
  sorry

end jake_sister_weight_ratio_l262_262705


namespace class_a_winning_probability_best_of_three_l262_262596

theorem class_a_winning_probability_best_of_three :
  let p := (3 : ℚ) / 5
  let win_first_two := p * p
  let win_first_and_third := p * ((1 - p) * p)
  let win_last_two := (1 - p) * (p * p)
  p * p + p * ((1 - p) * p) + (1 - p) * (p * p) = 81 / 125 :=
by
  sorry

end class_a_winning_probability_best_of_three_l262_262596


namespace log_prime_factor_inequality_l262_262627

open Real

-- Define p(n) such that it returns the number of prime factors of n.
noncomputable def p (n: ℕ) : ℕ := sorry  -- This will be defined contextually for now

theorem log_prime_factor_inequality (n : ℕ) (hn : n > 0) : 
  log n ≥ (p n) * log 2 :=
by 
  sorry

end log_prime_factor_inequality_l262_262627


namespace people_in_club_M_l262_262149

theorem people_in_club_M (m s z n : ℕ) (h1 : s = 18) (h2 : z = 11) (h3 : m + s + z + n = 60) (h4 : n ≤ 26) : m = 5 :=
sorry

end people_in_club_M_l262_262149


namespace systematic_first_choice_l262_262967

theorem systematic_first_choice :
  ∃ n : ℕ, n = 9 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → 39 - 3 * 10 = n) :=
by
  use 9
  split
  . rfl
  . intro k hk
    exact rfl

end systematic_first_choice_l262_262967


namespace zeros_of_quadratic_l262_262676

theorem zeros_of_quadratic (m : ℝ) :
  let f := λ x : ℝ, x^2 - m * x - m^2 in
  ∃ a b : ℝ, (a = b ∨ a ≠ b) :=
by
  let f := λ x : ℝ, x^2 - m * x - m^2
  let Δ := m^2 + 4 * m^2
  have hΔ : Δ ≥ 0 := by linarith
  sorry

end zeros_of_quadratic_l262_262676


namespace num_correct_equations_l262_262252

-- Condition 1: Equation to check (sqrt[n](a^n) == a)
def equation1 (n : Nat) (a : ℝ) : Prop :=
  Real.sqrt[n] (a^n) = a

-- Condition 2: For real a, (a^2 - a + 1)^0 = 1
def equation2 (a : ℝ) : Prop :=
  (a^2 - a + 1) ^ 0 = 1

-- Condition 3: Checking sqrt[3](x^4 + y^3) == x^(4/3) + y
def equation3 (x y : ℝ) : Prop :=
  Real.cbrt(x^4 + y^3) = x^(4/3) + y

-- Condition 4: Checking if sqrt[3](5) == sqrt[6]((-5)^2)
def equation4 : Prop :=
  Real.cbrt(5) = Real.root 6 ((-5)^2)

-- Final proof to count how many equations are correct
theorem num_correct_equations :
  (¬∃ n a, equation1 n a) ∧
  (∀ a, equation2 a) ∧
  (¬∀ x y, equation3 x y) ∧
  equation4 →
  2 = 2 := by
  sorry

end num_correct_equations_l262_262252


namespace trigonometric_range_l262_262306

theorem trigonometric_range (x : ℝ) (h1: 0 ≤ x) (h2: x < 2 * Real.pi) (h3: sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) : 
    Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4 :=
by
  sorry

end trigonometric_range_l262_262306


namespace b_bound_for_tangent_parallel_l262_262664

theorem b_bound_for_tangent_parallel (b : ℝ) (c : ℝ) :
  (∃ x : ℝ, 3 * x^2 - x + b = 0) → b ≤ 1/12 :=
by
  intros h
  -- Placeholder proof
  sorry

end b_bound_for_tangent_parallel_l262_262664


namespace problem1_problem2_l262_262629

open Real

-- Definitions
variables {α x : ℝ}
def OM := (cos α, sin α)
def ON := (cos x, sin x)
def PQ := (cos x, -sin x + 4 / (5 * cos α))

-- Problem statements
theorem problem1 (h1: cos α = 4 / (5 * sin x)) :
  ∃ T > 0, ∀ t, (y := dot_product ON PQ) → (y t + T = y t) → T = π := sorry

theorem problem2 (h2 : dot_product OM ON = 12 / 13)
(h3 : OM.1 / OM.2 = PQ.1 / PQ.2)
(h4 : 0 < α - x ∧ α - x < π / 2)
(h5 : 0 < α + x ∧ α + x < π / 2) :
  cos (2 * α) = 16 / 65 := sorry

end problem1_problem2_l262_262629


namespace markup_percentage_l262_262915

-- Define the wholesale cost
def wholesale_cost : ℝ := sorry

-- Define the retail cost
def retail_cost : ℝ := sorry

-- Condition given in the problem: selling at 60% discount nets a 20% profit
def discount_condition (W R : ℝ) : Prop :=
  0.40 * R = 1.20 * W

-- We need to prove the markup percentage is 200%
theorem markup_percentage (W R : ℝ) (h : discount_condition W R) : 
  ((R - W) / W) * 100 = 200 :=
by sorry

end markup_percentage_l262_262915


namespace proof_problem_l262_262040

-- Given definitions and conditions
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

def focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

def foci_positions (c : ℝ) : (ℝ × ℝ) := (-c, c)

def point_P_on_ellipse (x y : ℝ) : Prop := ellipse x y

def angle_F1_P_F2_half_pi_over_three (F1 P F2 : ℝ × ℝ) : Prop := 
  ∠ (F1 - P) (F2 - P) = Real.pi / 3

-- Goal is to prove the dot product
def dot_product_correct (F1 F2 P : ℝ × ℝ) : Prop :=
  let (x1, y1) := F1 in
  let (x2, y2) := F2 in
  let (px, py) := P in
  (px - x1) * (px - x2) + (py - y1) * (py - y2) = 32 / 3

-- Putting it all together to state the theorem
theorem proof_problem (x y : ℝ) (h1 : point_P_on_ellipse x y)
  (a b c : ℝ) (h2 : a = 5) (h3 : b = 4) (h4 : c = focal_distance a b)
  (h5 : F1 F2 := foci_positions c)
  (h6 : angle_F1_P_F2_half_pi_over_three F1 (x, y) F2) :
  dot_product_correct F1 F2 (x, y) :=
sorry

end proof_problem_l262_262040


namespace forecast_accuracy_probability_l262_262380

theorem forecast_accuracy_probability :
  let p := 0.8 in
  (p^3 + p^2 * (1 - p) + (1 - p) * p^2) = 0.768 :=
by
  -- Use by to initiate the proof. Here, we'll use sorry to skip the proof steps.
  sorry

end forecast_accuracy_probability_l262_262380


namespace pipe_tank_overflow_l262_262070

theorem pipe_tank_overflow (t : ℕ) :
  let rateA := 1 / 30
  let rateB := 1 / 60
  let combined_rate := rateA + rateB
  let workA := rateA * (t - 15)
  let workB := rateB * t
  (workA + workB = 1) ↔ (t = 25) := by
  sorry

end pipe_tank_overflow_l262_262070


namespace students_spend_15_50_l262_262099

noncomputable def pencil_price := 0.20
noncomputable def pen_price := 0.50
noncomputable def notebook_price := 1.50

noncomputable def Tolu_pencils := 3
noncomputable def Tolu_pens := 2
noncomputable def Tolu_notebooks := 1
noncomputable def Robert_pencils := 5
noncomputable def Robert_pens := 4
noncomputable def Robert_notebooks := 2
noncomputable def Melissa_pencils := 2
noncomputable def Melissa_pens := 3
noncomputable def Melissa_notebooks := 3

noncomputable def Tolu_total := (Tolu_pencils * pencil_price) +
                                (Tolu_pens * pen_price) + 
                                (Tolu_notebooks * notebook_price)

noncomputable def Robert_total := (Robert_pencils * pencil_price) +
                                  (Robert_pens * pen_price) + 
                                  (Robert_notebooks * notebook_price)

noncomputable def Melissa_total := (Melissa_pencils * pencil_price) +
                                   (Melissa_pens * pen_price) + 
                                   (Melissa_notebooks * notebook_price)

noncomputable def total_spend := Tolu_total + Robert_total + Melissa_total

theorem students_spend_15_50 :
  total_spend = 15.50 :=
by 
  sorry

end students_spend_15_50_l262_262099


namespace number_of_unique_triangles_l262_262991

-- Defining the geometry context
variables {Point Triangle : Type}
variables (A B C D E F H : Point)
variables (ABC AFH FHB BHD HDC CEH AEH : Triangle)
variables (ABE ACF BEC BFC ABD ADC AHB AHC BHC : Triangle)

-- Conditions
def conditions : Prop :=
  acute_triangle ABC ∧ 
  sides_unequal ABC ∧
  intersect_at H AD BE CF

-- Statement
theorem number_of_unique_triangles (h : conditions) : 
  number_of_unique_shapes = 7 := 
sorry

end number_of_unique_triangles_l262_262991


namespace hyperbola_eccentricity_l262_262660

variables {a c b e : ℝ}

theorem hyperbola_eccentricity (h₁ : ∀ x y : ℝ, (a > 0) → (y = (1/a) * x ∨ y = -(1/a) * x) → x^2 / a^2 - y^2 = 1)
                                (h₂ : ∀ θ : ℝ, θ = 30 → tan θ = 1 / a)
                                (h3 : b = 1)
                                (h4 : c = sqrt (a^2 + b^2)) :
                                e = c / a := by
  sorry

end hyperbola_eccentricity_l262_262660


namespace problem_proof_l262_262223

variables {α β γ : Type*} [plane α] [plane β] [plane γ]
variables {a b m n : Type*} [line a] [line b] [line m] [line n]
variables (parallel : α → β → Prop)
variables (perpendicular : α → γ → Prop)
variables (subset : m → α → Prop)
variables (not_subset : n → α → Prop)

-- Assume necessary definitions and axioms
axiom parallel_def : ∀ (α β : Type*), parallel α β ↔ ∀ {a b : Type*} [line a] [line b], parallel a b
axiom perpendicular_def : ∀ (α γ : Type*), perpendicular α γ ↔ ∀ {a b : Type*} [line a] [line b], perpendicular a b
axiom subset_def : ∀ (m α : Type*), subset m α ↔ ∀ {a : Type*} [line a], subset a α
axiom not_subset_def : ∀ (n α : Type*), not_subset n α ↔ ∀ {a : Type*} [line a], ¬subset a α

theorem problem_proof (h1 : perpendicular m α) 
                      (h2 : perpendicular n m) 
                      (h3 : not_subset_def n α) : 
                      ∀ {a b : Type*} [line a] [line b], parallel a b := 
by sorry

end problem_proof_l262_262223


namespace area_product_equality_l262_262767

variables {ABC A1 B1 C1 A2 B2 C2 : Type}
variables {t_a1 t_b1 t_c1 t_a2 t_b2 t_c2 : ℝ}

-- The feet of the altitudes in triangle ABC are A1, B1, and C1
-- The midpoints of the sides of triangle ABC are A2, B2, and C2
-- The areas of the triangles AB1C2, BC1A2, and CA1B2 are t_a1, t_b1, and t_c1, respectively
-- The areas of the triangles AC1B2, BA1C2, and CB1A2 are t_a2, t_b2, and t_c2, respectively

theorem area_product_equality
  (ABC_triang : ∃ (ABC : Type) (A1 B1 C1 A2 B2 C2 : ABC), 
                (t_a1 t_b1 t_c1 t_a2 t_b2 t_c2 : ℝ))
  (given_area_values :
    t_a1 = area_of_triangle (A B1 C2) ∧
    t_b1 = area_of_triangle (B C1 A2) ∧  
    t_c1 = area_of_triangle (C A1 B2) ∧
    t_a2 = area_of_triangle (A C1 B2) ∧
    t_b2 = area_of_triangle (B A1 C2) ∧
    t_c2 = area_of_triangle (C B1 A2)) :
    t_a1 * t_b1 * t_c1 = t_a2 * t_b2 * t_c2 :=
by
  -- Proof to be implemented
  sorry

end area_product_equality_l262_262767


namespace rationalize_denominator_l262_262080

theorem rationalize_denominator :
  ∃ (A B C : ℕ),  4 * real.root 343 (4:ℕ) / 21 = (A * real.root B (4:ℕ)) / C ∧ C > 0 ∧ B ≠ 0 ∧ ¬ (∃ p : ℕ, prime p ∧ p^4 ∣ B) ∧ (A + B + C = 368) :=
by
  use 4, 343, 21
  split
  { calc 4 * real.root 343 4 / 21 = (4 * real.root 343 4) / 21 : by rw mul_div_assoc
                               ... = (4 / 21) * real.root 343 4 : by rw [mul_comm, mul_assoc, one_mul]
                               ... = (4 * real.root 343 4) / 21 : by rw mul_comm }
  split
  { norm_num }
  split
  { norm_num }
  split
  { intro h,
    obtain ⟨p, hp, hdiv⟩ := h,
    have : ¬ prime p := nat.prime.not_dvd_one hp,
    contradiction }
  { norm_num }

end rationalize_denominator_l262_262080


namespace contest_end_time_l262_262568

theorem contest_end_time (start_time : Nat) (contest_duration : Nat) :
  start_time = 12 * 60 → contest_duration = 1500 → 
  let end_time := (start_time + contest_duration) % (24 * 60) in 
  end_time = 13 * 60 :=
by
  intros h1 h2
  let end_time := (start_time + contest_duration) % (24 * 60)
  have : end_time = 13 * 60 := sorry
  exact this

end contest_end_time_l262_262568


namespace f_10_is_358_l262_262603

def f : ℕ → ℤ
| 1       := 1
| 2       := 2
| (n + 3) := 2 * f (n + 2) - f (n + 1) + (n + 3)^2

theorem f_10_is_358 : f 10 = 358 := 
by {
  sorry
}

end f_10_is_358_l262_262603


namespace min_sum_min_expression_l262_262634

def sum_min_expression (x : Fin 2020 → ℝ) : ℝ :=
  (Finset.univ : Finset (Fin 2020)).sum (λ i, 
    (Finset.univ : Finset (Fin 2020)).sum (λ j, min i j * x i * x j)
  )

theorem min_sum_min_expression (x : Fin 2020 → ℝ) (h : x 1009 = 1) : 
  sum_min_expression x = 2 :=
sorry

end min_sum_min_expression_l262_262634


namespace lineup_students_l262_262609

def student := ℕ -- Representing the students as natural numbers for simplicity. A ⇔ 0, B ⇔ 1, C ⇔ 2, D ⇔ 3, E ⇔ 4.

def AB_next_to (arrangement : list student) : Prop :=
  (list.index_of 0 arrangement = list.index_of 1 arrangement + 1) ∨ 
  (list.index_of 0 arrangement + 1 = list.index_of 1 arrangement)

def C_not_next_to_D (arrangement : list student) : Prop :=
  (list.index_of 2 arrangement ≠ list.index_of 3 arrangement + 1) ∧ 
  (list.index_of 2 arrangement + 1 ≠ list.index_of 3 arrangement)

def count_valid_arrangements (arrangements : list (list student)) : ℕ :=
  (arrangements.filter (λ a, AB_next_to a ∧ C_not_next_to_D a)).length

noncomputable def total_number_of_ways : ℕ :=
  count_valid_arrangements (list.permutations [0, 1, 2, 3, 4])

theorem lineup_students :
  total_number_of_ways = 24 :=
by
  sorry

end lineup_students_l262_262609


namespace length_of_DE_l262_262108

-- Define the appropriate geometric setup
def isosceles_triangle (A B C P: Point) (a b c: ℝ) : Prop :=
  dist A B = a ∧ dist A C = b ∧ dist B C = c

def parallel (L1 L2: Line) : Prop := sorry -- Assume a definition of parallel lines

def area_fraction (T U: Triangle) (f: ℝ) : Prop :=
  area U = f * area T

-- Define the points involved
variables (A B C D E: Point)
variables (a b c: ℝ)

-- Conditions from (a)
variables (h_iso: isosceles_triangle A B C 15 20 20)
variables (h_parallel: parallel (segment D E) (segment A B))
variables (h_area_fraction: area_fraction (triangle D E B) (triangle A B) 0.25)

-- Main theorem to prove
theorem length_of_DE : dist D E = 7.5 :=
sorry  -- Proof to be provided later

end length_of_DE_l262_262108


namespace find_starting_number_l262_262878

theorem find_starting_number (x : ℕ) (hx : 2 ∣ x) (h_multiples : ∃ l, list.range l ≤ 100 ∧ list.length (filter (λ n, 2 ∣ n) (list.range l)) = 46) : x = 10 :=
sorry

end find_starting_number_l262_262878


namespace correct_sum_104th_parenthesis_l262_262271

noncomputable def sum_104th_parenthesis : ℕ := sorry

theorem correct_sum_104th_parenthesis :
  sum_104th_parenthesis = 2072 := 
by 
  sorry

end correct_sum_104th_parenthesis_l262_262271


namespace response_rate_increase_l262_262214

def response_rate (respondents : ℕ) (surveyed : ℕ) : ℚ := (respondents / surveyed : ℚ) * 100

def increase_in_response_rate (original_respondents : ℕ) (original_surveyed : ℕ)
    (redesigned_respondents : ℕ) (redesigned_surveyed : ℕ) : ℚ :=
  response_rate (redesigned_respondents) (redesigned_surveyed) - 
  response_rate (original_respondents) (original_surveyed)

theorem response_rate_increase :
  (increase_in_response_rate 7 70 9 63 ≈ 4.29) :=
sorry

end response_rate_increase_l262_262214


namespace release_10_non_pecking_budgerigars_l262_262371

noncomputable def aviary_pecking : Prop :=
  ∃ (s : Finset ℕ) (h₁ : s.card = 10), 
    (∀ a b ∈ s, 
      ∀ (has_pecked : ℕ → ℕ → Prop),
        ∀ (hcond : ∀ x, ∃ n, x.has_pecked n),
          ¬ has_pecked a b)

theorem release_10_non_pecking_budgerigars :
    aviary_pecking :=
  sorry

end release_10_non_pecking_budgerigars_l262_262371


namespace polynomial_identity_l262_262360

theorem polynomial_identity (x : ℝ) (hx : x^2 + x - 1 = 0) : x^4 + 2*x^3 - 3*x^2 - 4*x + 5 = 2 :=
sorry

end polynomial_identity_l262_262360


namespace find_a_l262_262921

def possible_scores : List ℕ := [103, 104, 105, 106, 107, 108, 109, 110]

def is_possible_score (a : ℕ) (n : ℕ) : Prop :=
  ∃ (k8 k0 ka : ℕ), k8 * 8 + ka * a + k0 * 0 = n

def is_impossible_score (a : ℕ) (n : ℕ) : Prop :=
  ¬ is_possible_score a n

theorem find_a : ∀ (a : ℕ), a ≠ 0 → a ≠ 8 →
  (∀ n ∈ possible_scores, is_possible_score a n) →
  is_impossible_score a 83 →
  a = 13 := by
  intros a ha1 ha2 hpossible himpossible
  sorry

end find_a_l262_262921


namespace peculiar_four_digit_number_l262_262151

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def digits_are_distinct (n : ℕ) : Prop :=
  n.digits.nodup

def num_divisors (n : ℕ) : ℕ :=
  (filter (λ d, d ∣ n) (range (n+1))).length

noncomputable def N : ℕ :=
  2601

theorem peculiar_four_digit_number :
  is_four_digit N ∧
  is_perfect_square N ∧
  is_perfect_square (sum_of_digits N) ∧
  is_perfect_square (N / sum_of_digits N) ∧
  num_divisors N = sum_of_digits N ∧
  digits_are_distinct N :=
by {
  let N := 2601
  show is_four_digit N,
  show is_perfect_square N,
  show is_perfect_square (sum_of_digits N),
  show is_perfect_square (N / sum_of_digits N),
  show num_divisors N = sum_of_digits N,
  show digits_are_distinct N,
  sorry
}

end peculiar_four_digit_number_l262_262151


namespace problem_solves_to_17_or_18_l262_262939

/-- Given problem conditions and the solution, prove that n = 17 or n = 18 -/
theorem problem_solves_to_17_or_18 (n : ℕ) (h1: 31 * 13 * n > 0) 
  (h2: ∃ s : finset (ℕ × ℕ × ℕ), (∀ t ∈ s, ∃ x y z : ℕ, t = (x, y, z) ∧ 2*x + 2*y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) ∧ s.card = 28 ) : 
  n = 17 ∨ n = 18 := 
sorry

end problem_solves_to_17_or_18_l262_262939


namespace sum_of_alternating_series_l262_262239

theorem sum_of_alternating_series : ∑ k in (finset.range 2019).map (add_right 3), (-1 : ℤ) ^ k = 1 :=
by
  sorry

end sum_of_alternating_series_l262_262239


namespace triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l262_262917

def triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b

def right_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_condition_A (a b c : ℝ) (h : triangle a b c) : 
  b^2 = (a + c) * (c - a) → right_triangle a c b := 
sorry

theorem triangle_condition_B (A B C : ℝ) (h : A + B + C = 180) : 
  A = B + C → 90 = A :=
sorry

theorem triangle_condition_C (A B C : ℝ) (h : A + B + C = 180) : 
  3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → 
  ¬ (right_triangle A B C) :=
sorry

theorem triangle_condition_D : 
  right_triangle 6 8 10 := 
sorry

theorem problem_solution (a b c : ℝ) (A B C : ℝ) (hABC : triangle a b c) : 
  (b^2 = (a + c) * (c - a) → right_triangle a c b) ∧
  ((A + B + C = 180) ∧ (A = B + C) → 90 = A) ∧
  (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → ¬ right_triangle a b c) ∧
  (right_triangle 6 8 10) → 
  ∃ (cond : Prop), cond = (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C) := 
sorry

end triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l262_262917


namespace least_possible_perimeter_l262_262120

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end least_possible_perimeter_l262_262120


namespace four_digit_property_l262_262019

-- Define the problem conditions and statement
theorem four_digit_property (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 0 ≤ y ∧ y < 100) :
  (100 * x + y = (x + y) ^ 2) ↔ (100 * x + y = 3025 ∨ 100 * x + y = 2025 ∨ 100 * x + y = 9801) := by
sorry

end four_digit_property_l262_262019


namespace cos_5alpha_eq_sin_5alpha_eq_l262_262604

noncomputable def cos_five_alpha (α : ℝ) : ℝ := 16 * (Real.cos α) ^ 5 - 20 * (Real.cos α) ^ 3 + 5 * (Real.cos α)
noncomputable def sin_five_alpha (α : ℝ) : ℝ := 16 * (Real.sin α) ^ 5 - 20 * (Real.sin α) ^ 3 + 5 * (Real.sin α)

theorem cos_5alpha_eq (α : ℝ) : Real.cos (5 * α) = cos_five_alpha α :=
by sorry

theorem sin_5alpha_eq (α : ℝ) : Real.sin (5 * α) = sin_five_alpha α :=
by sorry

end cos_5alpha_eq_sin_5alpha_eq_l262_262604


namespace solution_triple_root_system_l262_262807

theorem solution_triple_root_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  intro h
  sorry

end solution_triple_root_system_l262_262807


namespace brian_has_78_white_stones_l262_262237

-- Given conditions
variables (W B : ℕ) (R Bl : ℕ)
variables (x : ℕ)
variables (total_stones : ℕ := 330)
variables (total_collection1 : ℕ := 100)
variables (total_collection3 : ℕ := 130)

-- Condition: First collection stones sum to 100
#check W + B = 100

-- Condition: Brian has more white stones than black ones
#check W > B

-- Condition: Ratio of red to blue stones is 3:2 in the third collection
#check R + Bl = 130
#check R = 3 * x
#check Bl = 2 * x

-- Condition: Total number of stones in all three collections is 330
#check total_stones = total_collection1 + total_collection1 + total_collection3

-- New collection's magnetic stones ratio condition
#check 2 * W / 78 = 2

-- Prove that Brian has 78 white stones
theorem brian_has_78_white_stones
  (h1 : W + B = 100)
  (h2 : W > B)
  (h3 : R + Bl = 130)
  (h4 : R = 3 * x)
  (h5 : Bl = 2 * x)
  (h6 : 2 * W / 78 = 2) :
  W = 78 :=
sorry

end brian_has_78_white_stones_l262_262237


namespace smallest_n_terminating_decimal_l262_262898

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (n > 0) ∧ (∀ d : ℕ, (n + 110 = d) → (∀ p : ℕ, Prime p → p ∣ d → (p = 2 ∨ p = 5)) ∧ n = 15) :=
begin
  sorry
end

end smallest_n_terminating_decimal_l262_262898


namespace prob_X_gt_2_l262_262666

noncomputable def X : ℝ → ℝ := sorry
axiom X_normal : ∀ (x : ℝ), X(x) = pdf (NormalDist.mk 1 4) x

theorem prob_X_gt_2 :
  (∀ x, P(0 ≤ X ≤ 2) = 0.68) →
  P (X > 2) = 0.16 := 
by sorry

end prob_X_gt_2_l262_262666


namespace max_distinct_inscribed_angles_l262_262492

theorem max_distinct_inscribed_angles (n : ℕ) (h : n = 10) : 
  max_distinct_inscribed_angles_on_circle n = 80 :=
sorry

end max_distinct_inscribed_angles_l262_262492


namespace probability_abs_diff_gt_half_l262_262086

structure IndependentRandomNumbers where
  (x y : ℝ)
  (coin_flip1 coin_flip2_x coin_flip2_y : Bool)
  (hx : coin_flip1 → x ∈ Set.Icc 0 0.5) -- First flip heads
  (hx' : ¬coin_flip1 → coin_flip2_x → x = 0.75) -- First tails, second heads
  (hx'' : ¬coin_flip1 → ¬coin_flip2_x → x = 0.25) -- Both tails
  (hy : coin_flip1 → y ∈ Set.Icc 0 0.5) -- First flip heads
  (hy' : ¬coin_flip1 → coin_flip2_y → y = 0.75) -- First tails, second heads
  (hy'' : ¬coin_flip1 → ¬coin_flip2_y → y = 0.25) -- Both tails

def probability_condition (n m : ℕ) : ℝ :=
  if n = m then 0.5 else 1.0

theorem probability_abs_diff_gt_half :
  ∃ (n m : ℕ → IndependentRandomNumbers), 
  probability_condition n m = 0.5 :=
sorry

end probability_abs_diff_gt_half_l262_262086


namespace depth_of_channel_l262_262818

theorem depth_of_channel (a b A : ℝ) (h : ℝ) (h_area : A = (1 / 2) * (a + b) * h)
  (ha : a = 12) (hb : b = 6) (hA : A = 630) : h = 70 :=
by
  sorry

end depth_of_channel_l262_262818


namespace area_of_union_of_six_triangles_l262_262095

-- Define the side length of the equilateral triangles
def side_length : ℝ := 3

-- Define the total number of triangles
def num_triangles : ℕ := 6

-- Define the expected area covered by the union of the six triangles
def expected_area : ℝ := 10.6875 * Real.sqrt 3

-- Statement requiring proof: the area covered by the union of the six triangles
theorem area_of_union_of_six_triangles (s : ℝ) (n : ℕ) (area : ℝ) :
  s = side_length → n = num_triangles → area = expected_area :=
by
  intros h1 h2
  rw [h1, h2]
  exact sorry -- Proof goes here

end area_of_union_of_six_triangles_l262_262095


namespace grid_coloring_remainder_mod_2018_l262_262499

theorem grid_coloring_remainder_mod_2018 :
  ∃ M : ℕ, (∀ (grid : ℕ → ℕ → bool),
    (∀ i, (finset.univ.filter (λ j, grid i j = tt)).card = (finset.univ.filter (λ j, grid i j = ff)).card) →
    (∀ j, (finset.univ.filter (λ i, grid i j = tt)).card = (finset.univ.filter (λ i, grid i j = ff)).card) →
    M = finset.card {grid | (∀ i, (finset.univ.filter (λ j, grid i j = tt)).card = (finset.univ.filter (λ j, grid i j = ff)).card) ∧ 
      (∀ j, (finset.univ.filter (λ i, grid i j = tt)).card = (finset.univ.filter (λ i, grid i j = ff)).card)}) % 2018 = 6) :=
sorry

end grid_coloring_remainder_mod_2018_l262_262499


namespace ana_additional_payment_l262_262989

theorem ana_additional_payment (A B L : ℝ) (h₁ : A < B) (h₂ : A < L) : 
  (A + (B + L - 2 * A) / 3 = ((A + B + L) / 3)) :=
by
  sorry

end ana_additional_payment_l262_262989


namespace sum_of_solutions_l262_262098

theorem sum_of_solutions : 
  let S := { x : ℤ | 3^(x^2 + 3*x + 2) = 27^(x + 1) } in
  ∑ x in S, x = 0 :=
by
  sorry

end sum_of_solutions_l262_262098


namespace fish_remaining_l262_262748

def fish_caught_per_hour := 7
def hours_fished := 9
def fish_lost := 15

theorem fish_remaining : 
  (fish_caught_per_hour * hours_fished - fish_lost) = 48 :=
by
  sorry

end fish_remaining_l262_262748


namespace distinct_possible_values_pq_plus_p_plus_q_l262_262261

theorem distinct_possible_values_pq_plus_p_plus_q :
  ∃ S : Finset ℕ, 
    (∀ p q ∈ ({1, 3, 5, 7, 9, 11, 13} : Finset ℕ), (p + 1) * (q + 1) - 1 ∈ S) ∧ 
    S.card = 27 :=
sorry

end distinct_possible_values_pq_plus_p_plus_q_l262_262261


namespace time_to_pass_platform_l262_262582

-- Define the constants used in the problem
def length_of_train : ℝ := 250
def time_to_pass_pole : ℝ := 10
def length_of_platform : ℝ := 1250

-- Define the statement to be proved: it takes 60 seconds to pass the platform
theorem time_to_pass_platform :
  let speed := length_of_train / time_to_pass_pole in
  let combined_length := length_of_train + length_of_platform in
  combined_length / speed = 60 :=
by
  -- Proof steps would be here
  sorry

end time_to_pass_platform_l262_262582


namespace diagonal_bisects_angles_l262_262535

-- Define rhombus and its properties
structure Rhombus (A B C D : Type) :=
  (sides : ∀ (a b : A), a = b) -- all sides are equal
  (diagonals : ∀ (d1 d2 : B), ∃ (p : C), d1 ≠ d2 ∧ p ∈ d1 ∧ p ∈ d2) -- diagonals intersect at a point

-- Prove that the diagonal of a rhombus bisects its angles
theorem diagonal_bisects_angles (A B C D : Type) (r : Rhombus A B C D) : (∀ (angle : C), r.diagonals.angle = angle/2) :=
sorry

end diagonal_bisects_angles_l262_262535


namespace net_gain_is_2160_l262_262439

-- Define initial conditions
def home_worth : ℝ := 12000 -- Mr. X's home worth in dollars

def sale_price_X_to_Y (worth: ℝ) (profit_percent : ℝ) : ℝ := 
  worth * (1 + profit_percent / 100)

def sale_price_Y_to_X (sale_price : ℝ) (loss_percent : ℝ) : ℝ := 
  sale_price * (1 - loss_percent / 100)

-- Define the profit calculations
def initial_sale_price : ℝ := sale_price_X_to_Y home_worth 20
def buy_back_price : ℝ := sale_price_Y_to_X initial_sale_price 15

-- Calculate the net gain for Mr. X
def net_gain_X : ℝ := initial_sale_price - buy_back_price

theorem net_gain_is_2160 : 
  net_gain_X = 2160 := 
by
  sorry

end net_gain_is_2160_l262_262439


namespace range_of_f_l262_262905

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc (-3 : ℝ) 3) : 
  ∃ y, y ∈ Set.Icc (0 : ℝ) 25 ∧ ∀ z, z = (x^2 - 4*x + 4) → y = z :=
sorry

end range_of_f_l262_262905


namespace probability_blue_or_green_l262_262811

theorem probability_blue_or_green 
  (total_faces : ℕ) 
  (blue_faces : ℕ) 
  (green_faces : ℕ)
  (total_faces_eq : total_faces = 10)
  (blue_faces_eq : blue_faces = 1)
  (green_faces_eq : green_faces = 1) :
  (blue_faces + green_faces) / total_faces.toRat = 1 / 5 :=
by
  sorry

end probability_blue_or_green_l262_262811


namespace least_sub_number_l262_262174

theorem least_sub_number (x : ℕ) (divisor : ℕ) (remainder : ℕ) (correct_subtracted_number : ℕ) :
    x = 105829 → divisor = 21 → remainder = 105829 % 21 →
    correct_subtracted_number = remainder →
    (x - correct_subtracted_number) % divisor = 0 :=
by
  intros x_eq d_eq r_eq s_eq
  rw [x_eq, d_eq, s_eq, r_eq]
  sorry

end least_sub_number_l262_262174


namespace suitable_survey_l262_262536

def survey_suitable_for_census (A B C D : Prop) : Prop :=
  A ∧ ¬B ∧ ¬C ∧ ¬D

theorem suitable_survey {A B C D : Prop} (h_A : A) (h_B : ¬B) (h_C : ¬C) (h_D : ¬D) : survey_suitable_for_census A B C D :=
by
  unfold survey_suitable_for_census
  exact ⟨h_A, h_B, h_C, h_D⟩

end suitable_survey_l262_262536


namespace smallest_positive_period_symmetry_l262_262222

theorem smallest_positive_period_symmetry (f : ℝ → ℝ) :
  (∀ x, f x = sin (2 * x - π / 6)) →
  (∃ p > 0, (∀ x, f (x + p) = f x) ∧ p = π) ∧
  (∀ x, f (π / 3 - x) = f (π / 3 + x)) :=
sorry

end smallest_positive_period_symmetry_l262_262222


namespace max_unmarried_women_l262_262930

theorem max_unmarried_women (total_people : ℕ) (women_ratio : ℚ) (married_ratio : ℚ) 
(h1 : total_people = 80) 
(h2 : women_ratio = 2/5) 
(h3 : married_ratio = 1/2) 
: ∃ (max_unmarried_women : ℕ), max_unmarried_women = 32 :=
by
  have h_women : ℕ := (women_ratio * total_people).toNat
  have h_married : ℕ := (married_ratio * total_people).toNat
  have h_women_eq : h_women = 32 := by 
    rw [h2, h1]
    norm_num
  have h_married_eq : h_married = 40 := by 
    rw [h3, h1]
    norm_num
  exact ⟨32, by rw h_women_eq; rfl⟩

end max_unmarried_women_l262_262930


namespace geometric_sequence_problem_l262_262000

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

def condition (a : ℕ → ℝ) : Prop :=
a 4 + a 8 = -3

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : condition a) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end geometric_sequence_problem_l262_262000


namespace person_speed_kmph_l262_262542

theorem person_speed_kmph (distance_m : ℕ) (time_min : ℕ)
  (hdist : distance_m = 720)
  (htime : time_min = 12) : 
  let distance_km := (distance_m : ℝ) / 1000 in
  let time_hr := (time_min : ℝ) / 60 in
  let speed := distance_km / time_hr in
  speed = 3.6 :=
by
  sorry

end person_speed_kmph_l262_262542


namespace angle_GIH_gt_90_l262_262643

noncomputable theory

open EuclideanGeometry

def triangle : Type := Point
def incenter (A B C : Point) : Point := sorry
def centroid (A B C : Point) : Point := sorry
def orthocenter (A B C : Point) : Point := sorry

theorem angle_GIH_gt_90
  (A B C : triangle)
  (G := centroid A B C)
  (I := incenter A B C)
  (H := orthocenter A B C) :
  angle G I H > 90 :=
by
  sorry

end angle_GIH_gt_90_l262_262643


namespace ticket_price_divisors_count_l262_262574

theorem ticket_price_divisors_count :
  ∃ (x : ℕ), (36 % x = 0) ∧ (60 % x = 0) ∧ (Nat.divisors (Nat.gcd 36 60)).card = 6 := 
by
  sorry

end ticket_price_divisors_count_l262_262574


namespace magician_assistant_trick_l262_262964

theorem magician_assistant_trick (k : ℕ) (h_k : k = 2017) :
  (∃ n, n = k + 1 ∧ trick_works n) ∧ (∀ n, n ≤ k → ¬ trick_works n) :=
by sorry

end magician_assistant_trick_l262_262964


namespace max_points_in_equilateral_property_set_l262_262581

theorem max_points_in_equilateral_property_set (Γ : Finset (ℝ × ℝ)) :
  (∀ (A B : (ℝ × ℝ)), A ∈ Γ → B ∈ Γ → 
    ∃ C : (ℝ × ℝ), C ∈ Γ ∧ 
    dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B) → Γ.card ≤ 3 :=
by
  intro h
  sorry

end max_points_in_equilateral_property_set_l262_262581


namespace translate_cos_to_sin_l262_262826

def cos_function (x : ℝ) (φ : ℝ) : ℝ := cos (2 * x + φ)
def sin_function (x : ℝ) : ℝ := sin (2 * x + π / 3)

theorem translate_cos_to_sin (φ : ℝ) (h₁ : -π ≤ φ ∧ φ ≤ π) 
  (h₂ : ∀ x, cos_function (x - π / 2) φ = sin_function x) : φ = 5 * π / 6 :=
by
  sorry

end translate_cos_to_sin_l262_262826


namespace approx_five_l262_262280

theorem approx_five : 
  (10^100 + 10^102) / (10^101 + 10^101) ≈ 5 := 
by
  sorry

end approx_five_l262_262280


namespace length_more_than_breadth_by_200_percent_l262_262474

noncomputable def length: ℝ := 19.595917942265423
noncomputable def total_cost: ℝ := 640
noncomputable def rate_per_sq_meter: ℝ := 5

theorem length_more_than_breadth_by_200_percent
  (area : ℝ := total_cost / rate_per_sq_meter)
  (breadth : ℝ := area / length) :
  ((length - breadth) / breadth) * 100 = 200 := by
  have h1 : area = 128 := by sorry
  have h2 : breadth = 128 / 19.595917942265423 := by sorry
  rw [h1, h2]
  sorry

end length_more_than_breadth_by_200_percent_l262_262474


namespace proposition_p_iff_proposition_q_l262_262004

-- Variables for the triangle sides and angles
variables {a b c : ℝ} {A B C : ℝ}

-- Conditions from the problem statement
def proposition_p := (a / sin B = b / sin C) ∧ (b / sin C = c / sin A)
def proposition_q := A = B ∧ B = C

-- Main theorem stating equivalence
theorem proposition_p_iff_proposition_q :
  proposition_p ↔ proposition_q := 
sorry  -- proof goes here

end proposition_p_iff_proposition_q_l262_262004


namespace student_problem_correct_l262_262590

def student_problem_expr : ℝ :=
  -1^4 - (1 - 0.4) * (1 / 3) * (2 - 3^2)

theorem student_problem_correct : student_problem_expr = 0.4 :=
by
  -- Placeholder for the actual proof
  sorry

end student_problem_correct_l262_262590


namespace jenna_total_profit_l262_262397

-- Definitions from conditions
def widget_cost := 3
def widget_price := 8
def rent := 10000
def tax_rate := 0.2
def salary_per_worker := 2500
def number_of_workers := 4
def widgets_sold := 5000

-- Calculate intermediate values
def total_revenue := widget_price * widgets_sold
def total_cost_of_widgets := widget_cost * widgets_sold
def gross_profit := total_revenue - total_cost_of_widgets
def total_expenses := rent + salary_per_worker * number_of_workers
def net_profit_before_taxes := gross_profit - total_expenses
def taxes := tax_rate * net_profit_before_taxes
def total_profit_after_taxes := net_profit_before_taxes - taxes

-- Theorem to be proven
theorem jenna_total_profit : total_profit_after_taxes = 4000 := by sorry

end jenna_total_profit_l262_262397


namespace intersection_A_B_l262_262326

-- Define set A: {x | x ≤ 1}
def A : Set ℝ := {x | x ≤ 1}

-- Define set B: {x | x^2 - 2x < 0}
def B : Set ℝ := {x | x^2 - 2x < 0}

-- Define the intersection A ∩ B
def intersection : Set ℝ := {x | x ≤ 1} ∩ {x | 0 < x ∧ x < 2}

-- The theorem to prove
theorem intersection_A_B : intersection = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_l262_262326


namespace club_selection_ways_l262_262192

theorem club_selection_ways : 
  ∀ (num_members : Nat),
  num_members = 20 → 
  (∃ num_ways : Nat, num_ways = (num_members * (num_members - 1) * (num_members - 2)) / 2) ∧
  num_ways = 3420 :=
by
  intro num_members
  intro h
  use (num_members * (num_members - 1) * (num_members - 2)) / 2
  split
  . exact h
  . sorry

end club_selection_ways_l262_262192


namespace maximized_arc_difference_line_eq_l262_262963

theorem maximized_arc_difference_line_eq :
  (∃ (l : ℝ → ℝ) (x y : ℝ), l x = y ∧ (x - 1 = y - 1) ∧ (x^2 + y^2 = 4) ∧
  (∀ l x y, l x = y ∧ (x - 1 = y - 1) ∧ (x^2 + y^2 = 4) → l x = y ∧ x + y - 2 = 0)) :=
sorry

end maximized_arc_difference_line_eq_l262_262963


namespace distance_between_neg2_and_3_l262_262822
-- Import the necessary Lean libraries

-- State the theorem to prove the distance between -2 and 3 is 5
theorem distance_between_neg2_and_3 : abs (3 - (-2)) = 5 := by
  sorry

end distance_between_neg2_and_3_l262_262822
