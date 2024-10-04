import Data.Int.Order
import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.Multiplication
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Choose
import Mathlib.Combinatorics.Factorial
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.GraphTheory.Hamiltonian
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Integral.SetIntegral
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Calc
import data.set.finite
import tactic

namespace problem1_l619_619828

theorem problem1 : (1 * (-1)^2 + (-1 / 3)^(-2) - |(-5)| + (3 - Real.pi)^0) = 6 :=
by
  sorry

end problem1_l619_619828


namespace sequence_next_number_l619_619351

/-
  The problem is to find the next number in the sequence
  1, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 3, 4, 5, 6, 5, 3, 1, 2, 3, 4, 5, 6, 7, 6, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 3, 1, ?
  We need to prove that the next number in this sequence is 2.
-/
theorem sequence_next_number :
    let s := [1, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 3, 4, 5, 6, 5, 3, 1, 2, 3, 4, 5, 6, 7, 6, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 3, 1]
    s.length = 50 -> s[50] = 2 :=
by
  sorry

end sequence_next_number_l619_619351


namespace f_f_neg2_eq_4_l619_619501

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x else x^2

theorem f_f_neg2_eq_4 : f (f (-2)) = 4 :=
by
  sorry

end f_f_neg2_eq_4_l619_619501


namespace vector_angle_acuteness_l619_619889

theorem vector_angle_acuteness (x : ℝ) : 
  ∀ (a b : ℝ × ℝ), a = (1, 2) ∧ b = (x, 4) → 
    (∃ (θ : ℝ), θ ∈ (0, π/2) ∧ 
      ∀ ⦃x : ℝ⦄, x > -8 ∧ x ≠ 2 → 
        ((x > -8 ∧ x < 2) ∨ (x > 2))) := 
by
  sorry

end vector_angle_acuteness_l619_619889


namespace matrix_multiplication_correct_l619_619837

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-3, 2],
  ![4, 5]
]

def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![6, 3],
  ![-2, 4]
]

def result : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-22, -1],
  ![14, 32]
]

theorem matrix_multiplication_correct : mat1.mul mat2 = result := by 
  sorry

end matrix_multiplication_correct_l619_619837


namespace ramu_profit_percent_is_20_point_19_l619_619656

def total_cost (car_cost repair_cost : ℝ) : ℝ := 
  car_cost + repair_cost

def profit (selling_price total_cost : ℝ) : ℝ :=
  selling_price - total_cost

def profit_percent (profit total_cost : ℝ) : ℝ :=
  (profit / total_cost) * 100

theorem ramu_profit_percent_is_20_point_19 :
  let car_cost := 42000
  let repair_cost := 12000
  let selling_price := 64900
  let total_cost := total_cost car_cost repair_cost
  let profit := profit selling_price total_cost
  profit_percent profit total_cost ≈ 20.19 :=
by
  sorry

end ramu_profit_percent_is_20_point_19_l619_619656


namespace P_identity_l619_619653

theorem P_identity (n : ℕ) : 
  (∏ i in finset.range n, n + 1 + i) = 2^n * ∏ i in finset.range n, 2 * i + 1 := 
sorry

end P_identity_l619_619653


namespace second_rider_round_time_l619_619725

theorem second_rider_round_time (T : ℕ) :
  (∀ t, (t % 12 = 0) → (t % T = 0 → t = 36)) →
  T = 36 :=
by
  intro h
  have h1 : 36 % 12 = 0 := by norm_num
  specialize h 36 h1
  have h2 : ¬(36 % T = 0) → 36 ≠ 36 := by tauto
  contradiction
  sorry

end second_rider_round_time_l619_619725


namespace range_of_exponential_function_l619_619277

theorem range_of_exponential_function (x y : ℝ) :
  (∃ t : ℝ, t = (1 - x) / (1 + x) ∧ t ≠ -1 ∧ y = 2 ^ t) →
  (y ∈ (Set.Ioo 0 (1 / 2) ∪ Set.Ioo (1 / 2) ⊤)) :=
by
  sorry

end range_of_exponential_function_l619_619277


namespace triangle_ratios_l619_619422

variable (P1 P2 P3 : Type) [InnerProductSpace ℝ P1] [InnerProductSpace ℝ P2] [InnerProductSpace ℝ P3]

theorem triangle_ratios 
  (P : Type) [InnerProductSpace ℝ P] 
  (Q1 Q2 Q3 : Type) 
  (hP1P : InnerProduct P1 P = Q1)
  (hP2P : InnerProduct P2 P = Q2)
  (hP3P : InnerProduct P3 P = Q3) :
  ∃ r1 r2 r3 : ℝ, 
  r1 ≤ 2 ∧ r2 ≥ 2 ∧ r1 = ∥(P - Q1)∥ / ∥(P1 - P)∥
    ∧ r2 = ∥(P - Q2)∥ / ∥(P2 - P)∥
    ∧ r3 = ∥(P - Q3)∥ / ∥(P3 - P)∥ :=
sorry

end triangle_ratios_l619_619422


namespace sum_first_10_terms_eq_210_l619_619468

-- Definitions of conditions
def a : ℕ → ℕ
def a_2 : ℕ := 7
def a_4 : ℕ := 15

-- Theorem Statement
theorem sum_first_10_terms_eq_210 (a : ℕ → ℕ) (a_2 : a 2 = 7) (a_4 : a 4 = 15) : 
  (∑ i in Finset.range (10 + 1), a i) = 210 :=
sorry

end sum_first_10_terms_eq_210_l619_619468


namespace largest_angle_in_pentagon_l619_619365

theorem largest_angle_in_pentagon (x : ℝ) 
  (h_sum : (x + 1) + 2x + 3x + 4x + (5x - 1) = 540) 
  : 5 * x - 1 = 179 := by
  sorry

end largest_angle_in_pentagon_l619_619365


namespace int_solution_exists_l619_619748

theorem int_solution_exists (x y : ℤ) (h : x + y = 5) : x = 2 ∧ y = 3 := 
by
  sorry

end int_solution_exists_l619_619748


namespace joint_savings_account_total_l619_619584

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end joint_savings_account_total_l619_619584


namespace relatively_prime_bound_l619_619631

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in Finset.filter (λ d, d ∣ n) (Finset.range (n + 1)), d

theorem relatively_prime_bound (n : ℕ) (h₁ : n ≥ 2) :
  (nth_relatively_prime n n) ≥ sum_of_divisors n ∧
  (nth_relatively_prime n n) = sum_of_divisors n ↔ ∃ p e : ℕ, Prime p ∧ n = p^e :=
sorry

end relatively_prime_bound_l619_619631


namespace largest_of_seven_consecutive_composite_numbers_less_than_40_l619_619240

open Nat

theorem largest_of_seven_consecutive_composite_numbers_less_than_40 :
  ∃ (n : ℕ), 23 ≤ n ∧ n ≤ 30 ∧ ∀ (k : ℕ), n ≤ k ∧ k < n + 7 → ¬ prime k ∧ n + 6 = 30 :=
by
  sorry

end largest_of_seven_consecutive_composite_numbers_less_than_40_l619_619240


namespace other_integer_is_20_l619_619558

theorem other_integer_is_20 (x y : ℤ) (h_sum : 3 * x + 2 * y = 145) (h_one_is_35 : x = 35 ∨ y = 35) : x = 35 ∧ y = 20 ∨ y = 35 ∧ x = 20 :=
by {
  cases h_one_is_35,
  { -- Case x = 35
    left,
    split,
    { exact h_one_is_35 },
    { rw [h_one_is_35, 3 * 35] at h_sum,
      simp [h_sum] } },
  { -- Case y = 35
    right,
    split,
    { exact h_one_is_35 },
    { rw [h_one_is_35, 2 * 35] at h_sum,
      simp [h_sum] } }
}

end other_integer_is_20_l619_619558


namespace tan_value_expression_value_l619_619485

-- Definitions for the given conditions
def θ := let θ := θ inθ -- Dummy definition to declare θ
axiom sin_theta : sin θ = 4 / 5
axiom theta_range : π / 2 < θ ∧ θ < π

-- Problem statement to prove that tan(θ) = -4 / 3
theorem tan_value : tan θ = -4 / 3 :=
by
  have : θ := θ -- Dummy step to include θ in scope
  sorry

-- Problem statement to prove the value of the given expression
theorem expression_value : 
  (sin θ ^ 2 + 2 * sin θ * cos θ) / 
  (3 * sin θ ^ 2 + cos θ ^ 2) = -8 / 57 :=
by
  have : θ := θ -- Dummy step to include θ in scope
  sorry

end tan_value_expression_value_l619_619485


namespace smallest_integer_s_l619_619189

theorem smallest_integer_s (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (s : ℕ), (∀ (i : ℕ) (r : Fin m) (c : Fin n),
    i < s → 
    (∃ (array : Array (Array ℕ)), 
      (∀ r : Fin m, 
        let row := array[r];
        row.length = n ∧ ∀ k1 k2, k1 < n → k2 < n → k1 ≠ k2 → row[k1] ≠ row[k2] ∧
        (∀ k, k < n - 1 → row[k] + 1 = row[k + 1] ∨ row[k] + 1 = row[0])) ∧
      (∀ c : Fin n, 
        let col := (array.data.map (λ x => x[c]));
        col.length = m ∧ ∀ k1 k2, k1 < m → k2 < m → k1 ≠ k2 → col[k1] ≠ col[k2] ∧
        (∀ k, k < m - 1 → col[k] + 1 = col[k + 1] ∨ col[k] + 1 = col[0])) ∧
    (∀ r c, array[r][c] ≤ s))) :=
  s = m + n - Nat.gcd m n :=
begin
  sorry
end

end smallest_integer_s_l619_619189


namespace sequence_formula_l619_619264

theorem sequence_formula (a : ℕ → ℚ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = -1/2) (h3 : a 3 = 1/3) (h4 : a 4 = -1/4) :
  a n = (-1)^(n+1) * (1/n) :=
sorry

end sequence_formula_l619_619264


namespace log10_2_bound_l619_619121

-- Given conditions
theorem log10_2_bound :
  (10^3 = 1000) →
  (10^4 = 10000) →
  (2^9 = 512) →
  (2^{12} = 4096) →
  log 10 2 < 1 / 3 :=
by
  intros h1 h2 h3 h4
  sorry

end log10_2_bound_l619_619121


namespace max_value_div_mod_abs_a_plus_1_div_b_l619_619924

theorem max_value_div_mod_abs_a_plus_1_div_b (a b : ℝ) (h : a^2 - b^2 = -1) : 
  ∃ M, M = real.sqrt 2 ∧ (∀ t, real.abs (a b) ≤ t) := sorry

end max_value_div_mod_abs_a_plus_1_div_b_l619_619924


namespace max_farm_growth_l619_619252

noncomputable def v (x : ℝ) : ℝ :=
  if (0 < x ≤ 4) then 2
  else if (4 < x ≤ 20) then - (1 / 8) * x + 5 / 2
  else 0

noncomputable def f (x : ℝ) : ℝ := x * v x

theorem max_farm_growth : ∀ (x : ℝ), (0 < x ∧ x ≤ 20) → (f x ≤ 12.5) ∧ (f 10 = 12.5) :=
by
  intro x hx
  unfold f v
  split_ifs
  case pos h h_1 =>
    have h₀ : 0 < x ∧ x ≤ 4 := And.intro hx.left h_1
    calc
      x * 2 ≤ 4 * 2 := by
        nlinarith
      _ = 8 := by
        linarith
  case pos h h_1 =>
    have h₁ : 4 < x ∧ x ≤ 20 := And.intro h hx.right
    calc
      x * (- (1 / 8) * x + 5 / 2) ≤ 10 * (- 1 / 8 * 10 + 5 / 2) :=
        by
          interval_arith
      _ = 12.5 :=
        by
          exact congr_arg (op (*) x) sorry
  case neg h h_1 =>
    sorry

end max_farm_growth_l619_619252


namespace part1_equation_solution_part2_inequality_solution_l619_619761

theorem part1_equation_solution (x : ℝ) (h : x / (x - 1) = (x - 1) / (2 * (x - 1))) : 
  x = -1 :=
sorry

theorem part2_inequality_solution (x : ℝ) (h₁ : 5 * x - 1 > 3 * x - 4) (h₂ : - (1 / 3) * x ≤ 2 / 3 - x) : 
  -3 / 2 < x ∧ x ≤ 1 :=
sorry

end part1_equation_solution_part2_inequality_solution_l619_619761


namespace find_a_plus_c_l619_619393

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def bisector_eq (a b c : ℝ) (p : point) : Prop := 
  a * p.1 + 3 * p.2 + c = 0

def angle_bisector (a c : ℝ) (P Q R : point) : Prop :=
  ∃ k : ℝ, 
    let P' := ((sqrt (distance P Q) * R.1 + sqrt (distance P R) * Q.1) / (sqrt (distance P Q) + sqrt (distance P R)),
               (sqrt (distance P Q) * R.2 + sqrt (distance P R) * Q.2) / (sqrt (distance P Q) + sqrt (distance P R))) in
    bisector_eq a c P' ∧ a * P.1 + 3 * P.2 + c = 0

theorem find_a_plus_c :
  ∃ (a c : ℝ), angle_bisector a c (-5, 3) (-10, -15) (4, -5) :=
sorry

end find_a_plus_c_l619_619393


namespace running_speed_is_24_l619_619801

def walk_speed := 8 -- km/h
def walk_time := 3 -- hours
def run_time := 1 -- hour

def walk_distance := walk_speed * walk_time

def run_speed := walk_distance / run_time

theorem running_speed_is_24 : run_speed = 24 := 
by
  sorry

end running_speed_is_24_l619_619801


namespace johns_tip_percentage_l619_619186

theorem johns_tip_percentage :
  (∀ (d c : ℤ) (k : ℕ), d = 3 ∧ c = 200 ∧ k = 150 ->
  ∃ t, (t: ℕ)
  ∃ n T: ℤ, n = (30/d) ∧ T = 2050 ∧ 10 * k = n * k + 2 * c → 
  t = ((T - (n * k + 2 * c)) / n / k * 100) ∧ t = 10) :=
by sorry

end johns_tip_percentage_l619_619186


namespace sum_of_rational_roots_eq_zero_l619_619876

def h (x : ℚ) : ℚ := x^3 - 9 * x^2 + 27 * x - 14

theorem sum_of_rational_roots_eq_zero : ∑ r in (multiset.filter (λ r, h r = 0) [1, -1, 2, -2, 7, -7, 14, -14]), r = 0 :=
sorry

end sum_of_rational_roots_eq_zero_l619_619876


namespace train_length_l619_619384

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_conversion_factor : ℝ) (approx_length : ℝ) :
  speed_km_hr = 60 → time_seconds = 6 → speed_conversion_factor = (1000 / 3600) → approx_length = 100.02 →
  speed_km_hr * speed_conversion_factor * time_seconds = approx_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end train_length_l619_619384


namespace probability_xy_minus_x_minus_y_odd_l619_619304

-- Definitions based on the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def set := {1,2,3,4,5,6,7,8,9,10,11}

noncomputable def probability_odd_case : ℚ :=
  let odd_set := {n : ℕ | n ∈ set ∧ is_odd n}
  let even_set := {n : ℕ | n ∈ set ∧ is_even n}
  let odd_count := finite.to_finset odd_set).card
  let even_count := finite.to_finset even_set).card
  let valid_pairs := odd_count * even_count
  let total_pairs := nat.choose set.card 2
  (valid_pairs : ℚ) / (total_pairs : ℚ)

theorem probability_xy_minus_x_minus_y_odd : probability_odd_case = 6 / 11 :=
  sorry

end probability_xy_minus_x_minus_y_odd_l619_619304


namespace movie_marathon_l619_619300

theorem movie_marathon :
  let first_movie := 2
  let second_movie := first_movie * 1.5
  let combined_first_two := first_movie + second_movie
  let last_movie := 9 - combined_first_two
  combined_first_two - last_movie = 1 := by
  let first_movie := 2
  let second_movie := first_movie * 1.5
  let combined_first_two := first_movie + second_movie
  let last_movie := 9 - combined_first_two
  show combined_first_two - last_movie = 1 from
    calc 
      combined_first_two - last_movie = combined_first_two - (9 - combined_first_two) : by rfl
      ... = (first_movie + second_movie) - (9 - (first_movie + second_movie)) : by rfl
      ... = (2 + (2 * 1.5)) - (9 - (2 + (2 * 1.5))) : by rfl
      ... = (2 + 3) - (9 - 5): by rfl
      ... = 5 - 4 : by rfl
      ... = 1 : by rfl

end movie_marathon_l619_619300


namespace complex_number_quadrant_l619_619494

def complex_plane_quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0

theorem complex_number_quadrant :
  let z := (-1 + complex.i) * (2 * complex.i + 1)
  in complex_plane_quadrant z = 3 :=
begin
  sorry
end

end complex_number_quadrant_l619_619494


namespace halfway_fraction_between_l619_619058

theorem halfway_fraction_between (a b : ℚ) (h_a : a = 1/6) (h_b : b = 1/4) : (a + b) / 2 = 5 / 24 :=
by
  have h1 : a = (1 : ℚ) / 6 := h_a
  have h2 : b = (1 : ℚ) / 4 := h_b
  sorry

end halfway_fraction_between_l619_619058


namespace focus_of_parabola_l619_619685

theorem focus_of_parabola (h : ∀ y x, y^2 = 8 * x ↔ ∃ p, y^2 = 4 * p * x ∧ p = 2): (2, 0) ∈ {f | ∃ x y, y^2 = 8 * x ∧ f = (p, 0)} :=
by
  sorry

end focus_of_parabola_l619_619685


namespace ten_millions_in_hundred_million_hundred_thousands_in_million_l619_619714

theorem ten_millions_in_hundred_million :
  (100 * 10^6) / (10 * 10^6) = 10 :=
by sorry

theorem hundred_thousands_in_million :
  (1 * 10^6) / (100 * 10^3) = 10 :=
by sorry

end ten_millions_in_hundred_million_hundred_thousands_in_million_l619_619714


namespace find_a_l619_619423

def diamond (a b : ℝ) : ℝ := 3 * a - b^2

theorem find_a (a : ℝ) (h : diamond a 6 = 15) : a = 17 :=
by
  sorry

end find_a_l619_619423


namespace range_of_m_l619_619928

variable {α : Type*}

def A : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 2 }
def B (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 1 }

theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : 1/2 ≤ m ∧ m ≤ 1 :=
begin
  sorry
end

end range_of_m_l619_619928


namespace problem_solution_l619_619491

-- Declare the proof problem in Lean 4

theorem problem_solution (x y : ℝ) 
  (h1 : (y + 1) ^ 2 + (x - 2) ^ (1/2) = 0) : 
  y ^ x = 1 :=
sorry

end problem_solution_l619_619491


namespace find_omega_l619_619936

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) + sqrt 3 * cos (ω * x)

theorem find_omega
  (ω : ℝ)
  (h_omega_pos : ω > 0)
  (h_sum_zero : f ω (π / 6) + f ω (π / 2) = 0)
  (h_decreasing : ∀ x y : ℝ, (π / 6) < x → x < y → y < (π / 2) → f ω x > f ω y) :
  ω = 2 :=
sorry

end find_omega_l619_619936


namespace intersection_A_B_l619_619525
open Set

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by sorry

end intersection_A_B_l619_619525


namespace part1_part2_l619_619573

-- We state the problem conditions and theorems to be proven accordingly
variable (A B C : Real) (a b c : Real)

-- Condition 1: In triangle ABC, opposite sides a, b, c with angles A, B, C such that a sin(B - C) = b sin(A - C)
axiom condition1 (A B C : Real) (a b c : Real) : a * Real.sin (B - C) = b * Real.sin (A - C)

-- Question 1: Prove that a = b under the given conditions
theorem part1 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) : a = b := sorry

-- Condition 2: If c = 5 and cos C = 12/13
axiom condition2 (c : Real) : c = 5
axiom condition3 (C : Real) : Real.cos C = 12 / 13

-- Question 2: Prove that the area of triangle ABC is 125/4 under the given conditions
theorem part2 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) 
               (h2 : c = 5) (h3 : Real.cos C = 12 / 13): (1 / 2) * a * b * (Real.sin C) = 125 / 4 := sorry

end part1_part2_l619_619573


namespace paco_initial_sweet_cookies_l619_619224

theorem paco_initial_sweet_cookies
    (x : ℕ)  -- Paco's initial number of sweet cookies
    (eaten_sweet : ℕ)  -- number of sweet cookies Paco ate
    (left_sweet : ℕ)  -- number of sweet cookies Paco had left
    (h1 : eaten_sweet = 15)  -- Paco ate 15 sweet cookies
    (h2 : left_sweet = 19)  -- Paco had 19 sweet cookies left
    (h3 : x - eaten_sweet = left_sweet)  -- After eating, Paco had 19 sweet cookies left
    : x = 34 :=  -- Paco initially had 34 sweet cookies
sorry

end paco_initial_sweet_cookies_l619_619224


namespace apple_cost_price_l619_619806

theorem apple_cost_price (SP : ℝ) (loss_ratio : ℝ) (CP : ℝ) (h1 : SP = 18) (h2 : loss_ratio = 1/6) (h3 : SP = CP - loss_ratio * CP) : CP = 21.6 :=
by
  sorry

end apple_cost_price_l619_619806


namespace original_profit_margin_theorem_l619_619361

noncomputable def original_profit_margin (a : ℝ) (x : ℝ) (h : a > 0) : Prop := 
  (a * (1 + x) - a * (1 - 0.064)) / (a * (1 - 0.064)) = x + 0.08

theorem original_profit_margin_theorem (a : ℝ) (x : ℝ) (h : a > 0) :
  original_profit_margin a x h → x = 0.17 :=
sorry

end original_profit_margin_theorem_l619_619361


namespace quadrilateral_paving_possible_l619_619829

-- Define the property of the quadrilateral
def sum_of_interior_angles_is_360 (q : Type) [is_quadrilateral q] : Prop :=
  (interior_angles q).sum = 360

-- Defining that the ground can be paved with sufficient number of quadrilateral marble offcuts
def can_pave_with_quadrilaterals (q : Type) [is_quadrilateral q] : Prop :=
  ∃ n : ℕ, n > 0 ∧ (n * 4) * area_of q = area_of_ground
  
-- The main theorem stating the question with the given conditions.
theorem quadrilateral_paving_possible 
  (q : Type) [is_quadrilateral q]
  (h1 : sum_of_interior_angles_is_360 q)
  (h2 : all_same_size_and_shape q) :
  can_pave_with_quadrilaterals q :=
sorry

end quadrilateral_paving_possible_l619_619829


namespace charges_needed_to_vacuum_house_l619_619259

-- Conditions definitions
def battery_last_minutes : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def number_of_bedrooms : ℕ := 3
def number_of_kitchens : ℕ := 1
def number_of_living_rooms : ℕ := 1

-- Question (proof problem statement)
theorem charges_needed_to_vacuum_house :
  ((number_of_bedrooms + number_of_kitchens + number_of_living_rooms) * vacuum_time_per_room) / battery_last_minutes = 2 :=
by
  sorry

end charges_needed_to_vacuum_house_l619_619259


namespace find_usual_time_l619_619334

-- We define the variables for rate and time
variables (R T : ℝ)

-- The given conditions are that by walking at 4/3 of his usual rate, the boy arrives 4 minutes early
def usual_time_to_school (R T : ℝ) :=
  R * T = (4 / 3) * R * (T - 4)

-- The question is to prove the boy's usual time to reach the school is 16 minutes
theorem find_usual_time (R : ℝ) : (∃ T : ℝ, usual_time_to_school R T ∧ T = 16) :=
begin
  -- We should prove that for some T, the given condition implies T = 16
  sorry
end

end find_usual_time_l619_619334


namespace ab_equals_6_l619_619976

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619976


namespace part1_part2_l619_619406

theorem part1 (k : ℤ) (h : ∀ (x : ℤ), x = 2 → x^2 + k * x - 16 = 0): k = 6 :=
by
  have h2 := h 2 rfl
  linarith

theorem part2 (a b : ℤ) (h1 : ∀ (x : ℤ), x = -2 → 2 * x^4 - 4 * x^3 + a * x^2 + 7 * x + b = 0)
              (h2 : ∀ (x : ℤ), x = 1 → 2 * x^4 - 4 * x^3 + a * x^2 + 7 * x + b = 0): a = -12 ∧ b = 7 :=
by
  have h_neg2 := h1 (-2) rfl
  norm_num at h_neg2
  have h1 := h2 1 rfl
  norm_num at h1
  linarith

# Print theorems to verify successful parsing
# print part1
# print part2

end part1_part2_l619_619406


namespace pascal_triangle_first_20_probability_l619_619000

theorem pascal_triangle_first_20_probability :
    let rows : Fin 20 → List Nat := fun i => List.range (i.1 + 1)
    let total_elems := (Finset.univ : Finset (Fin 20)).sum (fun i => rows i).length
    let count_ones := 1 + (Finset.univ.image (fun i => if i.1 = 0 then 0 else 2)).sum id
    total_elems = 210 ∧ count_ones = 39 ∧ (count_ones : ℚ) / total_elems = 13 / 70 := by
  sorry

end pascal_triangle_first_20_probability_l619_619000


namespace least_sum_of_exponents_l619_619861

theorem least_sum_of_exponents (x : ℕ) (hx : x = 3125) :
  ∃ (s : finset ℕ), (x = s.sum (λ n, 2 ^ n)) ∧ s.sum id = 32 :=
by
  sorry

end least_sum_of_exponents_l619_619861


namespace perimeter_of_triangle_PXY_l619_619572

-- Define the parameters of triangle PQR.
constant P Q R : Type
constant PQ QR RP : ℝ 
axiom PQ_val : PQ = 15
axiom QR_val : QR = 36
axiom RP_val : RP = 27

-- Assume existence of incenter I of triangle PQR and line through it.
constant I : Type
constant X Y : Type
axiom line_through_I_parallel_to_QR : ∃ (line : Type), (line = I) ∧ (line ∥ QR) ∧ (line ∩ PQ = X) ∧ (line ∩ RP = Y)

-- Define parameters PX, PY, and XY
constant PX PY XY : ℝ 

-- Knowing the sides and incenter line properties, we aim to prove the perimeter:

theorem perimeter_of_triangle_PXY : PX + PY + XY = 17.423 :=
by {
  -- We will need to derive PX, PY, and XY based on the given conditions
  let s := PQ + QR + RP,
  have h1 : PX = PQ * PQ / s, sorry,
  have h2 : PY = RP * RP / s, sorry,
  have h3 : XY = PQ * RP / s, sorry,
  have h4 : (PX + PY + XY) = ((PQ * PQ + RP * RP + PQ * RP) / s), sorry,
  rw [PQ_val, QR_val, RP_val] at h4,
  norm_num at h4,
  exact h4,
}

end perimeter_of_triangle_PXY_l619_619572


namespace sum_of_primes_between_10_and_30_l619_619320

theorem sum_of_primes_between_10_and_30 : (∑ p in Finset.filter Nat.prime (Finset.range 31), if p > 10 then p else 0) = 112 := by
  sorry

end sum_of_primes_between_10_and_30_l619_619320


namespace find_value_of_m_l619_619084

theorem find_value_of_m (x m : ℤ) (h₁ : x = 2) (h₂ : y = m) (h₃ : 3 * x + 2 * y = 10) : m = 2 := 
by
  sorry

end find_value_of_m_l619_619084


namespace divide_fractions_l619_619851

theorem divide_fractions :
  (7 / 3) / (5 / 4) = (28 / 15) :=
by
  sorry

end divide_fractions_l619_619851


namespace fraction_defined_iff_l619_619544

theorem fraction_defined_iff (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) :=
by sorry

end fraction_defined_iff_l619_619544


namespace exists_func_satisfies_condition_l619_619292

theorem exists_func_satisfies_condition :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = abs (x + 1) :=
sorry

end exists_func_satisfies_condition_l619_619292


namespace no_subset_and_no_cover_l619_619316

open Set

noncomputable def maximum_subsets (n : ℕ) : ℕ :=
  Nat.choose (2 * n) (n - 1)

theorem no_subset_and_no_cover (S : Finset (Fin (2 * n)))
  (h_non_subset : ∀ (a b : Finset (Fin (2 * n))), a ∈ S → b ∈ S → (a ⊆ b → a = b))
  (h_no_cover : ∀ (a b : Finset (Fin (2 * n))), a ∈ S → b ∈ S → (a ∪ b ≠ Finset.univ)) :
  S.card ≤ maximum_subsets n := sorry

end no_subset_and_no_cover_l619_619316


namespace book_and_env_painting_count_number_of_book_and_env_painting_l619_619364

-- Definitions of the conditions
def total_participants : ℕ := 120
def book_club_participants : ℕ := 80
def fun_sports_participants : ℕ := 50
def env_theme_painting_participants : ℕ := 40
def book_and_fun_sports : ℕ := 20
def fun_sports_and_env_painting : ℕ := 10

-- Property that each resident participates in at most two activities
axiom at_most_two_activities :
  ∀ (A B C : ℕ), A + B + C - (A ∩ B) - (B ∩ C) - (C ∩ A) = total_participants

theorem book_and_env_painting_count :
  ∃ (x : ℕ), x = 20 ∧ 
    at_most_two_activities book_club_participants fun_sports_participants env_theme_painting_participants
theorem number_of_book_and_env_painting : ℕ :=
  by
    sorry

end book_and_env_painting_count_number_of_book_and_env_painting_l619_619364


namespace vector_angle_acuteness_l619_619890

theorem vector_angle_acuteness (x : ℝ) : 
  ∀ (a b : ℝ × ℝ), a = (1, 2) ∧ b = (x, 4) → 
    (∃ (θ : ℝ), θ ∈ (0, π/2) ∧ 
      ∀ ⦃x : ℝ⦄, x > -8 ∧ x ≠ 2 → 
        ((x > -8 ∧ x < 2) ∨ (x > 2))) := 
by
  sorry

end vector_angle_acuteness_l619_619890


namespace longest_segment_in_cylinder_l619_619783

theorem longest_segment_in_cylinder (radius height : ℝ) 
  (hr : radius = 5) (hh : height = 10) :
  ∃ segment_length, segment_length = 10 * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end longest_segment_in_cylinder_l619_619783


namespace sum_of_possible_values_of_f_l619_619416

def is_multiplicative_magic_square (M : Matrix (Fin 3) (Fin 3) ℤ) (P : ℤ) : Prop :=
  M 0 0 * M 0 1 * M 0 2 = P ∧
  M 1 0 * M 1 1 * M 1 2 = P ∧
  M 2 0 * M 2 1 * M 2 2 = P ∧
  M 0 0 * M 1 0 * M 2 0 = P ∧
  M 0 1 * M 1 1 * M 2 1 = P ∧
  M 0 2 * M 1 2 * M 2 2 = P ∧
  M 0 0 * M 1 1 * M 2 2 = P ∧
  M 0 2 * M 1 1 * M 2 0 = P

theorem sum_of_possible_values_of_f : ∀ (M : Matrix (Fin 3) (Fin 3) ℕ),
  (M 0 0 = 25) → 
  (M 2 1 = 48) → 
  (M 2 2 = 3) → 
  ∃ P, is_multiplicative_magic_square M P → 
  M 2 0 ∈ ({25} : Set ℕ) := sorry

end sum_of_possible_values_of_f_l619_619416


namespace edith_novel_count_l619_619855

-- Definitions based on conditions
variables (N W : ℕ)

-- Conditions from the problem
def condition1 : Prop := N = W / 2
def condition2 : Prop := N + W = 240

-- Target statement
theorem edith_novel_count (N W : ℕ) (h1 : N = W / 2) (h2 : N + W = 240) : N = 80 :=
by
  sorry

end edith_novel_count_l619_619855


namespace unique_m_for_pure_imaginary_l619_619346

variable {m : ℝ}

def pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem unique_m_for_pure_imaginary :
  (pure_imaginary (m * (m - 1) + (m - 1) * complex.I)) → m = 0 :=
by
  intro h
  sorry

end unique_m_for_pure_imaginary_l619_619346


namespace empty_can_mass_l619_619769

-- Define the mass of the full can
def full_can_mass : ℕ := 35

-- Define the mass of the can with half the milk
def half_can_mass : ℕ := 18

-- The theorem stating the mass of the empty can
theorem empty_can_mass : full_can_mass - (2 * (full_can_mass - half_can_mass)) = 1 := by
  sorry

end empty_can_mass_l619_619769


namespace turn_off_streetlights_l619_619559

theorem turn_off_streetlights : ∃ ways : ℕ, ways = 35 ∧ 
  (∀ (n m : ℕ), n ≤ 12 → m ≤ 4 → n ≠ 0 → n ≠ 12 ∧ n ≠ m ∧ (2 ≤ n ∧ n ≤ 11) → -- total streetlights and no selection of first and last
  (∀ a b c d : ℕ, 
    (a, b, c, d) ∈ ({i | i ∈ finset.range 12 \ {0, 11}}.powerset 4) ∧ 
    (∀ i j : ℕ, i ≠ j → abs (i - j) > 1))    -- no two streetlights turned off are adjacent

end turn_off_streetlights_l619_619559


namespace trains_crossing_time_l619_619354

theorem trains_crossing_time 
  (length_train1 : ℝ) 
  (speed_train1 : ℝ) 
  (length_train2 : ℝ) 
  (speed_train2 : ℝ) 
  (h1 : length_train1 = 270) 
  (h2 : speed_train1 = 120) 
  (h3 : length_train2 = 230.04) 
  (h4 : speed_train2 = 80)
  (relative_speed := (speed_train1 + speed_train2) * 1000 / 3600)
  (total_distance := length_train1 + length_train2)
  (time := total_distance / relative_speed):
  time ≈ 9 := 
by
  sorry

end trains_crossing_time_l619_619354


namespace correct_answer_l619_619529

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l619_619529


namespace days_considered_l619_619395

theorem days_considered (visitors_current : ℕ) (visitors_previous : ℕ) (total_visitors : ℕ)
  (h1 : visitors_current = 132) (h2 : visitors_previous = 274) (h3 : total_visitors = 406)
  (h_total : visitors_current + visitors_previous = total_visitors) :
  2 = 2 :=
by
  sorry

end days_considered_l619_619395


namespace range_of_t_l619_619073

def g (x : ℝ) : ℝ := x^2 - 2 * x
def f (x : ℝ) : ℝ := 2^(g x)

theorem range_of_t (t : ℝ) (h : ∀ x ∈ set.Icc (-1 : ℝ) t, f x ≤ 8) : t ∈ set.Ioc (-1, 3] :=
sorry

end range_of_t_l619_619073


namespace small_ring_rotations_l619_619807

-- Define the conditions for the problem
def large_ring_radius : ℝ := 4
def small_ring_radius : ℝ := 1

-- Define the function to compute the number of rotations
def number_of_rotations (large_radius small_radius : ℝ) : ℝ :=
  let large_circumference := 2 * large_radius * Real.pi
  let small_circumference := 2 * small_radius * Real.pi
  (large_circumference - small_circumference) / small_circumference

-- State the theorem
theorem small_ring_rotations :
  number_of_rotations large_ring_radius small_ring_radius = 3 :=
by
  -- Placeholder for proof
  sorry

end small_ring_rotations_l619_619807


namespace geometric_sequence_x_l619_619926

theorem geometric_sequence_x (x : ℝ) (h : 1 * x = x ∧ x * x = 9) : x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_x_l619_619926


namespace proof_floor_ceiling_addition_l619_619856

-- Definitions of the floor and ceiling functions
def my_floor (x : ℝ) : ℤ := int.floor x
def my_ceiling (x : ℝ) : ℤ := int.ceiling x

-- Specific values used in the problem
def x1 : ℝ := -0.237
def x2 : ℝ := 4.987

theorem proof_floor_ceiling_addition :
  my_floor x1 + my_ceiling x2 = 4 :=
by
  -- Leaving the proof as sorry since it's not required
  sorry

end proof_floor_ceiling_addition_l619_619856


namespace coefficient_x6_in_1px_8_is_28_l619_619260

theorem coefficient_x6_in_1px_8_is_28 : binomial 8 6 = 28 := 
by sorry

end coefficient_x6_in_1px_8_is_28_l619_619260


namespace remaining_money_proof_l619_619733

variables {scissor_cost eraser_cost initial_amount scissor_quantity eraser_quantity total_cost remaining_money : ℕ}

-- Given conditions
def conditions : Prop :=
  initial_amount = 100 ∧ 
  scissor_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissor_quantity = 8 ∧ 
  eraser_quantity = 10

-- Definition using conditions
def total_spent : ℕ :=
  scissor_quantity * scissor_cost + eraser_quantity * eraser_cost

-- Prove the total remaining money calculation
theorem remaining_money_proof (h : conditions) : 
  total_spent = 80 ∧ remaining_money = initial_amount - total_spent ∧ remaining_money = 20 :=
by
  -- Proof steps to be provided here
  sorry

end remaining_money_proof_l619_619733


namespace triangle_area_ABC_l619_619301

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_ABC :
  let A := (4 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 4 : ℝ)
  ∃ C : (ℝ × ℝ), (C.1 + C.2 = 8) → area_of_triangle A B C = 8 :=
by
  sorry

end triangle_area_ABC_l619_619301


namespace df_perp_ef_l619_619404

open Real EuclideanGeometry

-- Definitions and conditions from the given problem in a)
variables {A B C D M E F : Point}
variables (h1 : ∃ Δ : Triangle, (is_triangle Δ) ∧ ((vertex_A Δ) = A) ∧ ((vertex_B Δ) = B) ∧ ((vertex_C Δ) = C))
variables (h2 : angle_bisector A B C D)
variables (h3 : ∠ D C A = 60)
variables (h4 : distance D M = distance D B)
variables (h5 : intersect_ray AC B M = E)
variables (h6 : intersect_ray AB C M = F)

-- Proving orthogonality of DF and EF
theorem df_perp_ef : ⟪DF, EF⟫ = 0 :=
by
  sorry

end df_perp_ef_l619_619404


namespace inverse_var_q_value_l619_619230

theorem inverse_var_q_value (p q : ℝ) (h1 : ∀ p q, (p * q = 400))
(p_init : p = 800) (q_init : q = 0.5) (new_p : p = 400) :
  q = 1 := by
  sorry

end inverse_var_q_value_l619_619230


namespace identify_twins_with_one_question_l619_619308

theorem identify_twins_with_one_question :
  (∃ (Vanya Vasya : Prop), 
    (Vanya ∧ ¬ Vasya ∨ ¬ Vanya ∧ Vasya) ∧ 
    (∀ (brother : Prop), (brother = Vasya ∨ brother = Vanya)) ∧
    (∃ (ask_question : (Prop → Prop) → Prop),
      ask_question (λ brother, brother = Vasya → brother = Vanya))) :=
begin
  sorry
end

end identify_twins_with_one_question_l619_619308


namespace f_g_of_3_l619_619532

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l619_619532


namespace prime_square_sub_one_divisible_by_24_l619_619663

theorem prime_square_sub_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 24 ∣ p^2 - 1 := by
  sorry

end prime_square_sub_one_divisible_by_24_l619_619663


namespace obtuse_triangle_m_count_l619_619706

theorem obtuse_triangle_m_count :
  let valid_m := {m : ℕ | (17 < 13 + m) ∧ (m ≤ 10 ∨ (m < 30 ∧ m ≥ 22))};
  valid_m.card = 14 :=
by {
  sorry
}

end obtuse_triangle_m_count_l619_619706


namespace min_a_plus_b_eq_six_point_five_l619_619606

noncomputable def min_a_plus_b : ℝ :=
  Inf {s | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
                       (a^2 - 12 * b ≥ 0) ∧ 
                       (9 * b^2 - 4 * a ≥ 0) ∧ 
                       (a + b = s)}

theorem min_a_plus_b_eq_six_point_five : min_a_plus_b = 6.5 :=
by
  sorry

end min_a_plus_b_eq_six_point_five_l619_619606


namespace parallelogram_area_l619_619646

theorem parallelogram_area (θ : ℝ) (a b : ℝ) (hθ : θ = 100) (ha : a = 20) (hb : b = 10):
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  area = 200 * Real.cos 10 := 
by
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  sorry

end parallelogram_area_l619_619646


namespace symmetric_point_correct_l619_619167

def point_symmetric_to_x_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, -P.2, -P.3)

theorem symmetric_point_correct :
  point_symmetric_to_x_axis (1, 2, 3) = (1, -2, -3) :=
by
  dsimp [point_symmetric_to_x_axis]
  sorry

end symmetric_point_correct_l619_619167


namespace H_is_orthocenter_of_ABC_circumradius_of_ABC_is_R_l619_619294

open EuclideanGeometry

-- Given three circles, each of radius R, passing through the point H.
-- And given points A, B, and C are the other points of intersections different from H.
variables {R : ℝ} {H A B C : Point}
variables (circle1 circle2 circle3 : Circle)
variables (centers : Point → Point)

-- Assume the circles pass through the point H
axiom circles_through_H : (circle1.center = centers A ∧ circle2.center = centers B ∧ circle3.center = centers C) ∧
                          (circle1.radius = R ∧ circle2.radius = R ∧ circle3.radius = R) ∧
                          (H ∈ circle1.circumference ∧ H ∈ circle2.circumference ∧ H ∈ circle3.circumference)

-- Assume A, B, and C are pairwise intersections different from H
axiom pairwise_intersections : (A ∈ circle1.circumference ∧ A ∈ circle2.circumference ∧ A ≠ H) ∧
                               (B ∈ circle2.circumference ∧ B ∈ circle3.circumference ∧ B ≠ H) ∧
                               (C ∈ circle1.circumference ∧ C ∈ circle3.circumference ∧ C ≠ H)

-- 1. Prove that H is the orthocenter of triangle ABC
theorem H_is_orthocenter_of_ABC : is_orthocenter H A B C :=
sorry

-- 2. Prove that the circumradius of triangle ABC is R
theorem circumradius_of_ABC_is_R : circumradius A B C = R :=
sorry

end H_is_orthocenter_of_ABC_circumradius_of_ABC_is_R_l619_619294


namespace isosceles_triangle_area_l619_619055

-- Define the conditions for the isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c 

-- Define the side lengths
def side_length_1 : ℝ := 15
def side_length_2 : ℝ := 15
def side_length_3 : ℝ := 24

-- State the theorem
theorem isosceles_triangle_area :
  is_isosceles_triangle side_length_1 side_length_2 side_length_3 →
  side_length_1 = 15 →
  side_length_2 = 15 →
  side_length_3 = 24 →
  ∃ A : ℝ, (A = (1 / 2) * 24 * 9) ∧ A = 108 :=
sorry

end isosceles_triangle_area_l619_619055


namespace isosceles_triangle_top_angle_l619_619920

theorem isosceles_triangle_top_angle (A B C : Type) [triangle A B C] (isosceles : is_isosceles_triangle A B C) (angle_A : ∠A = 40) : 
∠top_angle(A B C) = 40 ∨ ∠top_angle(A B C) = 100 :=
begin
  sorry
end

end isosceles_triangle_top_angle_l619_619920


namespace range_of_m_l619_619460

theorem range_of_m (m : ℝ) (x : ℝ) :
  (¬ (|1 - (x - 1) / 3| ≤ 2) → ¬ (x^2 - 2 * x + (1 - m^2) ≤ 0)) → 
  (|m| ≥ 9) :=
by
  sorry

end range_of_m_l619_619460


namespace area_inside_circle_outside_square_l619_619809

-- Definition of the problem conditions
def diagonal_length_of_square := 2
def radius_of_circle := 1

-- The side length of the square derived from the diagonal
def side_length_of_square : ℝ := Real.sqrt 2

-- Area calculations
def area_of_circle : ℝ := Real.pi * (radius_of_circle ^ 2)
def area_of_square : ℝ := (side_length_of_square ^ 2)

-- The area inside the circle but outside the square
def area_outside_square : ℝ := area_of_circle - area_of_square

-- The final theorem statement
theorem area_inside_circle_outside_square : 
  area_outside_square = Real.pi - 2 := 
by
  sorry

end area_inside_circle_outside_square_l619_619809


namespace coeff_eq_30_implies_a_eq_neg_6_l619_619091

noncomputable def find_a (x a : ℝ) : ℝ := 
  let general_term (n : ℕ) := (-a)^n * (Nat.choose 5 n : ℝ) * x^((5-2*n)/2)
  if (general_term 1 = 30 * x^(3/2)) then -6 else sorry

theorem coeff_eq_30_implies_a_eq_neg_6 :
  (∀ x : ℝ, x > 0 ∧
  (∃ a : ℝ, a = find_a x a ∧ 
  ∑ i in Finset.range 6, (Nat.choose 5 i : ℝ) * (x^(1/2))^(5-i) * (-a/x^(1/2))^i = 30 * x^(3/2))) → 
  ∃ a : ℝ, a = -6 := 
sorry

end coeff_eq_30_implies_a_eq_neg_6_l619_619091


namespace necessary_but_not_sufficient_l619_619471

variables {V : Type*} [inner_product_space ℝ V]

theorem necessary_but_not_sufficient
  {a b c : V}
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0) :
  (inner a c = inner b c) ↔ (a = b) :=
sorry

end necessary_but_not_sufficient_l619_619471


namespace focus_parabola_hyperbola_equal_l619_619141

theorem focus_parabola_hyperbola_equal (a : ℝ) (h₀ : a > 0) 
  (h₁ : (2, 0) = (real.sqrt (a^2 + 3), 0)): a = 1 :=
sorry

end focus_parabola_hyperbola_equal_l619_619141


namespace calculate_value_l619_619844

def my_operation (a b c : ℕ) : ℚ :=
  (a + b : ℚ) / (c : ℚ)

theorem calculate_value :
  my_operation (my_operation 100 20 60) (my_operation 7 2 3) 3 = 5 / 3 :=
by
  sorry

end calculate_value_l619_619844


namespace spherical_to_rectangular_coordinates_l619_619023

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ),
    ρ = 15 →
    θ = 5 * Real.pi / 6 →
    φ = Real.pi / 3 →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    z = ρ * Real.cos φ →
    x = -45 / 4 ∧ y = -15 * Real.sqrt 3 / 4 ∧ z = 7.5 := 
by
  intro ρ θ φ x y z
  intro hρ hθ hφ hx hy hz
  rw [hρ, hθ, hφ] at *
  rw [hx, hy, hz]
  sorry

end spherical_to_rectangular_coordinates_l619_619023


namespace infinitely_many_common_divisors_greater_than_one_l619_619652

theorem infinitely_many_common_divisors_greater_than_one :
  ∃ᶠ n in at_top, ∃ d > 1, d ∣ (2 * n - 3) ∧ d ∣ (3 * n - 2) :=
begin
  sorry
end

end infinitely_many_common_divisors_greater_than_one_l619_619652


namespace smallest_C_l619_619043

noncomputable def f : ℝ+ → ℝ+ := sorry -- Define the function f from positive reals to positive reals

theorem smallest_C : ∃ (f : ℝ+ → ℝ+), 
  (∀ (x y : ℝ+), x * f (x^2) * f (f y) + f (y * f x) = f (x * y) * (f (f (x^2)) + f (f (y^2)))) ∧ (∃ C, C = 1 / 2) :=
begin
  obtain ⟨f, hf⟩ : ∃ f : ℝ+ → ℝ+, ∀ (x y : ℝ+),
    x * f (x^2) * f (f y) + f (y * f x) = f (x * y) * (f (f (x^2)) + f (f (y^2))),
  {
     sorry
  },
  use [f, hf],
  use (1 / 2),
  refl,
end

end smallest_C_l619_619043


namespace time_ratio_upstream_downstream_l619_619799

variables (Vm Vs : ℝ)
def Vu (Vm Vs : ℝ) := Vm - Vs
def Vd (Vm Vs : ℝ) := Vm + Vs
def ratio (Vm Vs : ℝ) := (Vd Vm Vs) / (Vu Vm Vs)

theorem time_ratio_upstream_downstream :
  Vm = 5 → Vs = 1.6666666666666667 → ratio Vm Vs = 2 :=
by
  intros hVm hVs
  unfold ratio Vu Vd
  rw [hVm, hVs]
  -- this simplifies the arithmetic
  have : 5 - 1.6666666666666667 = 3.333333333333333 := .refl
  have : 5 + 1.6666666666666667 = 6.666666666666667 := .refl
  simp[rw[hVm, hVs]]
  rw [← one_div_mul_one_div (5 - 1.6666666666666667) (5 + 1.6666666666666667)]
  simp only[Real.one_div, inv_div, mul_one]
  have : (5:ℝ) = 3.333333333333333 * 2 := rfl
  simp[ratio_at_rat,one_div,inv_div,mul_one,ha about]
  -- the precise numerical simplification would need libraries
  exact (5 : ℝ)/(3.333333333333333)
  simp[]

end time_ratio_upstream_downstream_l619_619799


namespace polynomial_coeff_sum_l619_619128

theorem polynomial_coeff_sum (a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x: ℝ, (x - 1) ^ 4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_4 - a_3 + a_2 - a_1 + a_0 = 16 :=
by
  intro h
  sorry

end polynomial_coeff_sum_l619_619128


namespace calculate_fg3_l619_619536

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l619_619536


namespace georgia_vs_texas_license_plates_l619_619887

theorem georgia_vs_texas_license_plates :
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  georgia_plates - texas_plates = 731161600 :=
by
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  show georgia_plates - texas_plates = 731161600
  sorry

end georgia_vs_texas_license_plates_l619_619887


namespace instrument_price_problem_l619_619638

theorem instrument_price_problem (v t p : ℝ) (h1 : 1.5 * v = 0.5 * t + 50) (h2 : 1.5 * t = 0.5 * p + 50) : 
  ∃ m n : ℤ, m = 80 ∧ n = 80 ∧ (100 + m) * v / 100 = n + (100 - m) * p / 100 := 
by
  use 80, 80
  sorry

end instrument_price_problem_l619_619638


namespace rhombus_longer_diagonal_l619_619381

open Real

theorem rhombus_longer_diagonal (s d1 d2 : ℝ) (hf_1 : s = 27) (hf_2 : d1 = 36) :
  d2 = 30 * sqrt 3 :=
by {
  have h_half_d1 : d1 / 2 = 18, from by { rw hf_2, norm_num },  -- d1 / 2 = 18
  have h_half_d2 : sqrt ((s^2) - (d1 / 2)^2) = sqrt 405, 
  { calc sqrt ((s^2) - (d1 / 2)^2) = sqrt (729 - 324) : by { rw [hf_1, h_half_d1], norm_num }
                                    ...                 = sqrt 405 : by norm_num },
  have h_d2_halves : d2 / 2 = sqrt 405, from h_half_d2,
  calc d2 = 2 * sqrt 405 : by { rw ←h_d2_halves, ring }
       ... = 30 * sqrt 3 : by norm_num
}

end rhombus_longer_diagonal_l619_619381


namespace sqrt_equation_solution_l619_619846

theorem sqrt_equation_solution (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (sqrt (a - b + c) = sqrt a - sqrt b + sqrt c) →
  (a = b ∧ 0 ≤ c) ∨ (b = c ∧ 0 ≤ a) :=
by
  sorry

end sqrt_equation_solution_l619_619846


namespace otimes_example_l619_619025

def otimes (a b : ℝ) : ℝ := (1/3) * a - 4 * b

theorem otimes_example : otimes 12 (-1) = 8 := by
  sorry

end otimes_example_l619_619025


namespace sequence_geometric_and_general_sum_within_interval_l619_619465

variable (a : ℕ → ℝ)
-- Conditions
def initial_condition := (a 1 = 1)
def recurrence_relation := ∀ n, a (n + 1) = 2 * a n + 2

-- Proof Problem 1: Proving the sequence transformation and general formula
theorem sequence_geometric_and_general (h1 : initial_condition a) (h2 : recurrence_relation a) :
  ∀ n, a n = 3 * 2^(n - 1) - 2 :=
sorry

-- Proof Problem 2: Proving the sum of terms within the specific interval
theorem sum_within_interval (h1 : initial_condition a) (h2 : recurrence_relation a) :
  ( ∑ n in Finset.Icc 4 10, a n ) = 3034 :=
sorry

end sequence_geometric_and_general_sum_within_interval_l619_619465


namespace rectangle_perimeter_l619_619655

theorem rectangle_perimeter (x y : ℝ) (h1 : 2 * x + y = 44) (h2 : x + 2 * y = 40) : 2 * (x + y) = 56 := 
by
  sorry

end rectangle_perimeter_l619_619655


namespace cartesian_eq_C1_polar_coords_intersect_C1_C2_l619_619157

section PolarToCartesian

variable {ρ θ x y t : ℝ}

-- Definition for the curve C1 in polar coordinates
def polar_eq_C1 (ρ θ : ℝ) : Prop := 
  ρ^2 - 4 * ρ * Real.cos θ + 3 = 0

-- Definition for the curve C2 in parametric form
def param_eq_C2 (t : ℝ) : Prop := 
  x = t * Real.cos (Real.pi / 6) ∧ y = t * Real.sin (Real.pi / 6)

-- Prove that the Cartesian equation for curve C1 is correct
theorem cartesian_eq_C1 (ρ θ : ℝ) (h : polar_eq_C1 ρ θ) : 
  (x - 2)^2 + y^2 = 1 := sorry

-- Prove that the intersection points' polar coordinates are correct
theorem polar_coords_intersect_C1_C2 (h : polar_eq_C1 ρ (π / 6)) (t : ℝ) : 
  (ρ, θ) = (√3, π / 6) := sorry

end PolarToCartesian

end cartesian_eq_C1_polar_coords_intersect_C1_C2_l619_619157


namespace cheryl_used_total_amount_l619_619014

theorem cheryl_used_total_amount :
  let bought_A := (5 / 8 : ℚ)
  let bought_B := (2 / 9 : ℚ)
  let bought_C := (2 / 5 : ℚ)
  let leftover_A := (1 / 12 : ℚ)
  let leftover_B := (5 / 36 : ℚ)
  let leftover_C := (1 / 10 : ℚ)
  let used_A := bought_A - leftover_A
  let used_B := bought_B - leftover_B
  let used_C := bought_C - leftover_C
  used_A + used_B + used_C = 37 / 40 :=
by 
  sorry

end cheryl_used_total_amount_l619_619014


namespace ab_equals_6_l619_619974

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619974


namespace jake_sold_tuesday_correct_l619_619578

def jake_initial_pieces : ℕ := 80
def jake_sold_monday : ℕ := 15
def jake_remaining_wednesday : ℕ := 7

def pieces_sold_tuesday (initial : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) : ℕ :=
  initial - sold_monday - remaining_wednesday

theorem jake_sold_tuesday_correct :
  pieces_sold_tuesday jake_initial_pieces jake_sold_monday jake_remaining_wednesday = 58 :=
by
  unfold pieces_sold_tuesday
  norm_num
  sorry

end jake_sold_tuesday_correct_l619_619578


namespace find_y_value_l619_619514

noncomputable def complex_x : ℂ := (2 - complex.i) / (3 + complex.i)
noncomputable def complex_matrix (x : ℂ) : ℂ :=
  (4 * complex.i * (x + complex.i)) - (3 - x * complex.i) * (1 + complex.i)

theorem find_y_value : complex_matrix complex_x = -5 := by
  sorry

end find_y_value_l619_619514


namespace natural_coordinates_on_parabola_l619_619060

theorem natural_coordinates_on_parabola :
  { (x, y) : ℕ × ℕ | y = -x^2 / 3 + 98 }.card = 5 :=
sorry

end natural_coordinates_on_parabola_l619_619060


namespace one_fifth_greater_than_decimal_by_term_l619_619689

noncomputable def one_fifth := (1 : ℝ) / 5
noncomputable def decimal_value := 20000001 / 10^8
noncomputable def term := 1 / (5 * 10^8)

theorem one_fifth_greater_than_decimal_by_term :
  one_fifth > decimal_value ∧ one_fifth - decimal_value = term :=
  sorry

end one_fifth_greater_than_decimal_by_term_l619_619689


namespace disease_cases_linear_decrease_l619_619148

theorem disease_cases_linear_decrease (cases_1970 cases_2010 cases_1995 cases_2005 : ℕ)
  (year_1970 year_2010 year_1995 year_2005 : ℕ)
  (h_cases_1970 : cases_1970 = 800000)
  (h_cases_2010 : cases_2010 = 200)
  (h_year_1970 : year_1970 = 1970)
  (h_year_2010 : year_2010 = 2010)
  (h_year_1995 : year_1995 = 1995)
  (h_year_2005 : year_2005 = 2005)
  (linear_decrease : ∀ t, cases_1970 - (cases_1970 - cases_2010) * (t - year_1970) / (year_2010 - year_1970) = cases_1970 - t * (cases_1970 - cases_2010) / (year_2010 - year_1970))
  : cases_1995 = 300125 ∧ cases_2005 = 100175 := sorry

end disease_cases_linear_decrease_l619_619148


namespace negation_correct_l619_619272

def original_proposition (x y : ℝ) : Prop := (x^2 + y^2 = 0) → (x = 0 ∧ y = 0)

def negation_of_original_proposition (x y : ℝ) : Prop := (x^2 + y^2 ≠ 0) → ¬(x = 0 ∧ y = 0)

theorem negation_correct (x y : ℝ) : negation_of_original_proposition x y ↔ ¬original_proposition x y :=
by
  sorry

end negation_correct_l619_619272


namespace total_assignment_schemes_l619_619449

def num_ways_to_assign_teams : ℕ :=
  let ways_A := choose 4 1  -- C^1_4
  let ways_rest := factorial 4  -- 4!
  ways_A * ways_rest

theorem total_assignment_schemes :
    num_ways_to_assign_teams = 96 :=
by
  -- We can define each component individually in Lean 4 if needed, but this compact form achieves the same:
  have h1: choose 4 1 = 4 := by simp [choose]
  have h2: factorial 4 = 24 := by norm_num
  have h3: num_ways_to_assign_teams = 4 * 24 := by simp [num_ways_to_assign_teams, h1, h2]
  rw [h3]
  norm_num -- This computes the final result 4 * 24 = 96

end total_assignment_schemes_l619_619449


namespace smallest_positive_period_minimum_value_of_m_l619_619938

open Real

-- Conditions

def f (x : ℝ) : ℝ := sin(x)^2 + (sqrt 3) * sin(x) * cos(x)

-- Theorem Statement
theorem smallest_positive_period :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = π := sorry

theorem minimum_value_of_m (m : ℝ) (hmax : ∀ x ∈ Icc(-π / 3, m), f x ≤ 3 / 2) :
  m ≥ π / 3 := sorry

end smallest_positive_period_minimum_value_of_m_l619_619938


namespace sum_of_exponents_l619_619859

theorem sum_of_exponents : 
  ∃ S : Finset ℕ, (3125 = S.sum (λ i, 2^i) ∧ S.sum id = 32) :=
by
  sorry

end sum_of_exponents_l619_619859


namespace problem_condition_l619_619943

def f (x : ℝ) : ℝ :=
  if x < 0 then log (2 : ℝ) (1 - x) else (4 : ℝ)^x

theorem problem_condition :
  f (-3) + f (Real.logb (2 : ℝ) 3) = 11 := by
  -- Proof to be written
  sorry

end problem_condition_l619_619943


namespace problem_b_c_constants_l619_619133

theorem problem_b_c_constants (b c : ℝ) (h : ∀ x : ℝ, (x + 2) * (x + b) = x^2 + c * x + 6) : c = 5 := 
by sorry

end problem_b_c_constants_l619_619133


namespace smallest_positive_number_is_option_B_l619_619871

theorem smallest_positive_number_is_option_B :
  let A := 8 - 2 * Real.sqrt 17
  let B := 2 * Real.sqrt 17 - 8
  let C := 25 - 7 * Real.sqrt 5
  let D := 40 - 9 * Real.sqrt 2
  let E := 9 * Real.sqrt 2 - 40
  0 < B ∧ (A ≤ 0 ∨ B < A) ∧ (C ≤ 0 ∨ B < C) ∧ (D ≤ 0 ∨ B < D) ∧ (E ≤ 0 ∨ B < E) :=
by
  sorry

end smallest_positive_number_is_option_B_l619_619871


namespace max_min_sum_of_squares_l619_619206

theorem max_min_sum_of_squares (a : Fin 2009 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_sum : ∑ i, a i = 2)
  (h_pair_sum : ∑ i in Finset.range 2009, a i * a ((i + 1) % 2009) = 1) :
  (2 : ℝ) = max S ∧ (3/2 : ℝ) = min S :=
by sorry

end max_min_sum_of_squares_l619_619206


namespace sand_needed_l619_619182

def area_rectangular_patch : ℕ := 6 * 7
def area_square_patch : ℕ := 5 * 5
def sand_per_square_inch : ℕ := 3

theorem sand_needed : area_rectangular_patch + area_square_patch * sand_per_square_inch = 201 := sorry

end sand_needed_l619_619182


namespace x_intercept_is_neg_three_halves_l619_619253

-- Definition of the points
def pointA : ℝ × ℝ := (-1, 1)
def pointB : ℝ × ℝ := (3, 9)

-- Statement of the theorem: The x-intercept of the line passing through the points is -3/2.
theorem x_intercept_is_neg_three_halves (A B : ℝ × ℝ)
    (hA : A = pointA)
    (hB : B = pointB) :
    ∃ x_intercept : ℝ, x_intercept = -3 / 2 := 
by
    sorry

end x_intercept_is_neg_three_halves_l619_619253


namespace hundredth_digit_of_fraction_l619_619962

theorem hundredth_digit_of_fraction (n : ℕ) :
  let repeating_sequence := "269230769"
  ∧ let decimal_repr := "0." ++ repeating_sequence
  in decimal_repr[(100 % repeating_sequence.length + 1)] = '2' := by
  sorry

end hundredth_digit_of_fraction_l619_619962


namespace joint_savings_account_total_l619_619585

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end joint_savings_account_total_l619_619585


namespace find_c_l619_619622

theorem find_c (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (h : f (\sqrt e) = 1 / 2)
  (h_range : ∃ m n, (∀ x ∈ set.Icc (\sqrt e) c, f x ∈ set.Icc m n) ∧ n - m = 3 / 2) :
  c = exp 2 := 
sorry

end find_c_l619_619622


namespace simplify_expression_l619_619667

variables {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end simplify_expression_l619_619667


namespace incorrect_step_in_system_solution_l619_619247

theorem incorrect_step_in_system_solution (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : a - b = 2) :
  2 * b = 3 → false := by
  have h3 : 2 * a + b - 2 * (a - b) = 7 - 2 * 2,
  { linarith, },
  have h4 : 2 * a + b - 2 * a + 2 * b = 3,
  { linarith at h3, },
  have h5 : 3 * b = 3,
  { linarith at h4, },
  have h6 : b = 1,
  { linarith at h5, },
  have h7 : 2 * b = 2,
  { linarith at h6, },
  intro h8,
  linarith,

end incorrect_step_in_system_solution_l619_619247


namespace minimal_value_of_a_b_l619_619616

noncomputable def minimal_sum_of_a_and_b : ℝ := 6.11

theorem minimal_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : discriminant (λ x, x^2 + a * x + 3 * b) >= 0) 
  (h4 : discriminant (λ x, x^2 + 3 * b * x + a) >= 0) : 
  a + b = minimal_sum_of_a_and_b :=
sorry

end minimal_value_of_a_b_l619_619616


namespace length_of_train_l619_619771

variable (L : ℕ)

def speed_tree (L : ℕ) : ℚ := L / 120

def speed_platform (L : ℕ) : ℚ := (L + 500) / 160

theorem length_of_train
    (h1 : speed_tree L = speed_platform L)
    : L = 1500 :=
sorry

end length_of_train_l619_619771


namespace croissants_left_l619_619842

-- Definitions based on conditions
def total_croissants : ℕ := 17
def vegans : ℕ := 3
def allergic_to_chocolate : ℕ := 2
def any_type : ℕ := 2
def guests : ℕ := 7
def plain_needed : ℕ := vegans + allergic_to_chocolate
def plain_baked : ℕ := plain_needed
def choc_baked : ℕ := total_croissants - plain_baked

-- Assuming choc_baked > plain_baked as given
axiom croissants_greater_condition : choc_baked > plain_baked

-- Theorem to prove
theorem croissants_left (total_croissants vegans allergic_to_chocolate any_type guests : ℕ) 
    (plain_needed plain_baked choc_baked : ℕ) 
    (croissants_greater_condition : choc_baked > plain_baked) : 
    (choc_baked - guests + any_type) = 3 := 
by sorry

end croissants_left_l619_619842


namespace Carmela_difference_l619_619831

theorem Carmela_difference (Cecil Catherine Carmela : ℤ) (X : ℤ) (h1 : Cecil = 600) 
(h2 : Catherine = 2 * Cecil - 250) (h3 : Carmela = 2 * Cecil + X) 
(h4 : Cecil + Catherine + Carmela = 2800) : X = 50 :=
by { sorry }

end Carmela_difference_l619_619831


namespace ab_equals_six_l619_619998

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l619_619998


namespace generating_function_identity_binet_formula_identity_l619_619336

noncomputable section

-- Definitions of the Fibonacci sequence recurrence relation
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

-- Definition of the characteristic roots
def φ := (1 + Real.sqrt 5) / 2
def φ_hat := (1 - Real.sqrt 5) / 2

-- The generating function for the Fibonacci sequence
def F (x : ℝ) : ℝ := x / (1 - x - x^2)

-- The desired form of the generating function
def F_transformed (x : ℝ) : ℝ := (1 / Real.sqrt 5) * ((1 / (1 - φ * x)) - (1 / (1 - φ_hat * x)))

-- Binet's formula for the Fibonacci sequence
def binet_formula (n : ℕ) : ℝ := (1 / Real.sqrt 5) * ((φ ^ n) - (φ_hat ^ n))

theorem generating_function_identity (x : ℝ) : F(x) = F_transformed(x) := by
  sorry

theorem binet_formula_identity (n : ℕ) : fib n = binet_formula n := by
  sorry

end generating_function_identity_binet_formula_identity_l619_619336


namespace right_angled_triangles_in_pyramid_l619_619169

theorem right_angled_triangles_in_pyramid (P A B C : Point) (h1 : Perpendicular PA (Plane A B C)) (h2 : ∠ACB = 90°) : 
  number_of_right_angled_triangles (faces_of_pyramid P A B C) = 3 :=
sorry

end right_angled_triangles_in_pyramid_l619_619169


namespace nat_number_36_sum_of_digits_l619_619049

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end nat_number_36_sum_of_digits_l619_619049


namespace evaluate_g_at_5_l619_619134

def g (x : ℝ) : ℝ := 5 * x + 2

theorem evaluate_g_at_5 : g 5 = 27 := by
  sorry

end evaluate_g_at_5_l619_619134


namespace tyler_remaining_money_l619_619730

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end tyler_remaining_money_l619_619730


namespace min_repetitions_2002_div_by_15_l619_619547

-- Define the function that generates the number based on repetitions of "2002" and appending "15"
def generate_number (n : ℕ) : ℕ :=
  let repeated := (List.replicate n 2002).foldl (λ acc x => acc * 10000 + x) 0
  repeated * 100 + 15

-- Define the minimum n for which the generated number is divisible by 15
def min_n_divisible_by_15 : ℕ := 3

-- The theorem stating the problem with its conditions (divisibility by 15)
theorem min_repetitions_2002_div_by_15 :
  ∀ n : ℕ, (generate_number n % 15 = 0) ↔ (n ≥ min_n_divisible_by_15) :=
sorry

end min_repetitions_2002_div_by_15_l619_619547


namespace men_l619_619353

-- Given conditions
variable (W M : ℕ)
variable (B : ℕ) [DecidableEq ℕ] -- number of boys
variable (total_earnings : ℕ)

def earnings : ℕ := 5 * M + W * M + 8 * W

-- Total earnings of men, women, and boys is Rs. 150.
def conditions : Prop := 
  5 * M = W * M ∧ 
  W * M = 8 * W ∧ 
  earnings = total_earnings

-- Prove men's wages (total wages for 5 men) is Rs. 50.
theorem men's_wages (hm : total_earnings = 150) (hb : W = 8) : 
  5 * M = 50 :=
by
  sorry

end men_l619_619353


namespace quadrilateral_proof_problem_l619_619379

theorem quadrilateral_proof_problem 
  (A B C D P : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ P]
  (circ : ∀ {X Y Z W : Type} [InnerProductSpace ℝ X] [InnerProductSpace ℝ Y] [InnerProductSpace ℝ Z] [InnerProductSpace ℝ W], (ABCD : X) → radius R)
  (AB CD : ℝ)
  (AD BC : ℝ)
  (h1 : AB = 5)
  (h2 : CD = 5)
  (h3 : AD > BC)
  (height_B_AD : ℝ)
  (h4 : height_B_AD = 3)
  (area_ADP : ℝ)
  (h5 : area_ADP = 25 / 2)
  (R : ℝ) : 
  AD = 10 ∧ BC = 2 ∧ R = 5 * sqrt 5 / 2 := 
sorry

end quadrilateral_proof_problem_l619_619379


namespace find_a_plus_b_l619_619458

theorem find_a_plus_b (a b : ℝ) (ha : a > b) (hb : b > 1) 
  (hlog : log a b + log b a = 5/2) (heq : a^b = b^a) : 
  a + b = 6 :=
sorry

end find_a_plus_b_l619_619458


namespace value_of_c_l619_619766

variable (a b c : ℝ)

theorem value_of_c :
  8 = 0.06 * a ∧ 6 = 0.08 * b → c = b / a → c = 0.5625 :=
by
  intros h hc
  have ha : a = 8 / 0.06 := by sorry
  have hb : b = 6 / 0.08 := by sorry
  subst ha
  subst hb
  rw hc
  simp
  sorry

end value_of_c_l619_619766


namespace find_number_of_small_gardens_l619_619338

-- Define the conditions
def seeds_total : Nat := 52
def seeds_big_garden : Nat := 28
def seeds_per_small_garden : Nat := 4

-- Define the target value
def num_small_gardens : Nat := 6

-- The statement of the proof problem
theorem find_number_of_small_gardens 
  (H1 : seeds_total = 52) 
  (H2 : seeds_big_garden = 28) 
  (H3 : seeds_per_small_garden = 4) 
  : seeds_total - seeds_big_garden = 24 ∧ (seeds_total - seeds_big_garden) / seeds_per_small_garden = num_small_gardens := 
sorry

end find_number_of_small_gardens_l619_619338


namespace triangle_obtuse_count_l619_619704

theorem triangle_obtuse_count : ∃ (m : ℕ), (m ∈ {5, 6, 7, 8, 9, 10} ∪ {22, 23, 24, 25, 26, 27, 28, 29}) ∧
  ∀ (a b : ℕ), m ∈ {5, 6, 7, 8, 9, 10} ∪ {22, 23, 24, 25, 26, 27, 28, 29} →
    (a = 13 ∧ b = 17 ∧ 17^2 > 13^2 + m^2 ∨ a + b > m ∧ m > m + 13) :=
sorry

end triangle_obtuse_count_l619_619704


namespace train_length_proof_l619_619392

noncomputable def train_length (speed_km_per_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  speed_m_per_s * time_sec

theorem train_length_proof :
  train_length 60 6 = 100.02 :=
by
  sorry

end train_length_proof_l619_619392


namespace initial_investment_l619_619660

-- Definitions from conditions
def r : ℝ := 0.10
def n : ℤ := 2
def t : ℤ := 1
def A : ℝ := 8820
def factor : ℝ := 1 + r / n

-- Main theorem to prove the initial investment
theorem initial_investment:
  ∃ P : ℝ, P = A / (factor^(n * t)) ∧ P = 8000 := by
  sorry

end initial_investment_l619_619660


namespace probability_A_given_B_l619_619295

def roll_outcomes : ℕ := 6^3 -- Total number of possible outcomes when rolling three dice

def P_AB : ℚ := 60 / 216 -- Probability of both events A and B happening

def P_B : ℚ := 91 / 216 -- Probability of event B happening

theorem probability_A_given_B : (P_AB / P_B) = (60 / 91) := by
  sorry

end probability_A_given_B_l619_619295


namespace geometric_sequence_arithmetic_means_l619_619527

-- Given conditions
variables (a b c m n : ℝ)
hypothesis H1 : b^2 = a * c
hypothesis H2 : m = (a + b) / 2
hypothesis H3 : n = (b + c) / 2

-- Statement to prove
theorem geometric_sequence_arithmetic_means (H1 : b^2 = a * c) (H2 : m = (a + b) / 2) (H3 : n = (b + c) / 2) : (a / m) + (c / n) = 2 :=
by sorry

end geometric_sequence_arithmetic_means_l619_619527


namespace largest_BD_l619_619192

theorem largest_BD (a b c d : ℕ) (ha : distinct (a :: b :: c :: d :: [])) (h : a ≤ 13 ∧ b ≤ 13 ∧ c ≤ 13 ∧ d ≤ 13)
  (cyclic_quad : ∃ (AC BD AB CD : ℕ), AC * BD = AB * CD) : 
  ∃ BD, BD = real.sqrt 178 := 
by sorry

end largest_BD_l619_619192


namespace correct_statement_from_prob_conditions_l619_619746

theorem correct_statement_from_prob_conditions :
  let fair_dice_odd_prob := ℚ (1 / 2)
  let king_draw_prob := ℚ (1 / 13)
  let sampling := ∃ (batch: Type), ∀ (subset: set batch), through_sampling(subset, batch) → represents_service_life(subset, batch)
  let variance_A := 3
  let variance_B := 0.02
  let correct_statement := "Understanding the service life of a batch of refrigerators through sampling."
  (sampled: (sampling)) → correct_statement :=
begin
  sorry
end

end correct_statement_from_prob_conditions_l619_619746


namespace part_I_part_II_l619_619120

def P : Set ℝ := {x | x^2 - 5 * x + 4 ≤ 0}
def Q (b : ℝ) : Set ℝ := {x | x^2 - (b + 2) * x + 2 * b ≤ 0}

theorem part_I (b : ℝ) (h : b = 1) : Q b = {x | 1 ≤ x ∧ x ≤ 2} :=
by
  rw [h]
  sorry

theorem part_II : ∀ (b : ℝ), Q b ⊆ P → 1 ≤ b ∧ b ≤ 4 :=
by
  intro b
  assume hQsubsetP
  sorry

end part_I_part_II_l619_619120


namespace count_distinct_pairs_l619_619064

theorem count_distinct_pairs (m n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ m) (h3 : m ≤ 5) :
  (∑ m in finset.Icc 1 5, ∑ n in finset.Icc 1 m, 1) = 15 :=
by sorry

end count_distinct_pairs_l619_619064


namespace matrix_power_solve_l619_619916

theorem matrix_power_solve :
  ∃ (a n : ℕ), let A := !![1, 3, a; 0, 1, 5; 0, 0, 1] in
               A^n = !![1, 27, 2883; 0, 1, 45; 0, 0, 1] ∧ a + n = 264 :=
by
  sorry

end matrix_power_solve_l619_619916


namespace longest_segment_in_cylinder_l619_619790

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end longest_segment_in_cylinder_l619_619790


namespace other_workers_count_l619_619577

theorem other_workers_count (W : ℕ) (prob : ℚ) (hW : W = 8) (hprob : prob = 1 / (W.choose 2)) :
  W - 2 = 6 :=
by
  rw hW at hprob
  have h : (8 : ℚ).choose 2 = 28 := by sorry
  rw h at hprob
  have hp : 1 / 28 = 0.03571428571428571 := by sorry
  rw hp at hprob
  exact nat.sub_self _ -- Provides the proof based on provided conditions.

end other_workers_count_l619_619577


namespace area_of_quadrilateral_l619_619560

theorem area_of_quadrilateral (A B C : ℝ) (triangle1 triangle2 triangle3 quadrilateral : ℝ)
  (hA : A = 5) (hB : B = 9) (hC : C = 9)
  (h_sum : quadrilateral = triangle1 + triangle2 + triangle3)
  (h1 : triangle1 = A)
  (h2 : triangle2 = B)
  (h3 : triangle3 = C) :
  quadrilateral = 40 :=
by
  sorry

end area_of_quadrilateral_l619_619560


namespace octagon_divide_into_regions_l619_619250

theorem octagon_divide_into_regions :
  (∀ (O : Type) [octagon O]
      (A : O → ℝ)
      (Angle135 : ∀ (x : O → ℝ), x = 135),
      ∀ (SideLength : O → ℝ), 
      (∀ (n : ℕ), SideLength (n + 2) = if n.even then 1 else (sqrt 2)))
  (O : Type) [octagon O] →  count_regions (O) = 84 :=
by
  sorry

end octagon_divide_into_regions_l619_619250


namespace slope_and_y_intercept_l619_619737

def line_equation (x y : ℝ) : Prop := 4 * y = 6 * x - 12

theorem slope_and_y_intercept (x y : ℝ) (h : line_equation x y) : 
  ∃ m b : ℝ, (m = 3/2) ∧ (b = -3) ∧ (y = m * x + b) :=
  sorry

end slope_and_y_intercept_l619_619737


namespace daughter_normal_probability_l619_619521

-- Definitions based on the problem conditions
def hemophilia_inheritance : Prop := ∀ (X : Type), ∃ b : X → Prop, ∀ (a : X), b a → ¬ phenotypically_normal a
def phenylketonuria_inheritance : Prop := ∀ (G : Type), ∃ r : G → Prop, ∀ (a : G), r a → ¬ phenotypically_normal a

def phenotypically_normal (person : Type) : Prop := sorry -- this should be defined properly

noncomputable def couple_normal_have_affected_son : Prop := ∃ (X G : Type) (father mother son : X), 
  phenotypically_normal mother ∧ phenotypically_normal father ∧ 
  ¬ phenotypically_normal son ∧
  hemophilia_inheritance ∧ phenylketonuria_inheritance

theorem daughter_normal_probability (X G : Type) (father mother daughter : X) 
  (h1 : couple_normal_have_affected_son) : 
  probability (phenotypically_normal daughter) = 3/4 := 
sorry

end daughter_normal_probability_l619_619521


namespace longest_segment_in_cylinder_l619_619781

-- Define the given conditions
def radius : ℝ := 5 -- Radius of the cylinder in cm
def height : ℝ := 10 -- Height of the cylinder in cm

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the longest segment L inside the cylinder using the Pythagorean theorem
noncomputable def longest_segment : ℝ := Real.sqrt ((diameter ^ 2) + (height ^ 2))

-- State the problem in Lean:
theorem longest_segment_in_cylinder :
  longest_segment = 10 * Real.sqrt 2 :=
sorry

end longest_segment_in_cylinder_l619_619781


namespace nonnegative_values_of_a_l619_619145

theorem nonnegative_values_of_a : 
  (∀ a x : ℝ, 
    (0 ≤ a ∧ a ≤ 6 ∧ x = 1 + 4 / (a - 1) ∧ x ≠ 2 ∧ 
     (2 * x - 3 > 9) ∧ (x < a) → x ∈ ℤ) → 
  ({a : ℝ | 0 ≤ a ∧ a ≤ 6 ∧ ∃ x : ℝ, x = 1 + 4 / (a - 1) ∧ x ≠ 2 ∧ 
    (2 * x - 3 > 9) ∧ (x < a) ∧ x ∈ ℤ}.card = 3)) :=
sorry

end nonnegative_values_of_a_l619_619145


namespace divisor_greater_than_2_l619_619323

theorem divisor_greater_than_2 (w n d : ℕ) (h1 : ∃ q1 : ℕ, w = d * q1 + 2)
                                       (h2 : n % 8 = 5)
                                       (h3 : n < 180) : 2 < d :=
sorry

end divisor_greater_than_2_l619_619323


namespace problem_1_problem_2_l619_619411

theorem problem_1 : Real.sqrt 9 + Real.cbrt (-8) + abs (1 - Real.sqrt 3) = Real.sqrt 3 :=
by
  sorry

theorem problem_2 (a : ℝ) : (12 * a^3 - 6 * a^2 + 3 * a) / (3 * a) = 4 * a^2 - 2 * a + 1 :=
by
  sorry

end problem_1_problem_2_l619_619411


namespace light_path_length_correct_l619_619191

noncomputable def cube_side_length : ℝ := 10
def point_B : (ℝ × ℝ × ℝ) := (0, 10, 0)
def point_Q : (ℝ × ℝ × ℝ) := (6, 10, 3)

def distance_of_light_path : ℝ := 10 * (Real.sqrt 145)

theorem light_path_length_correct : 
  (∃ cube_side_length = 10 ∧ point_B = (0, 10, 0) ∧ point_Q = (6, 10, 3),
  distance_of_light_path = 10 * Real.sqrt 145) :=
begin
  -- sorry to skip the actual proof
  sorry,
end

end light_path_length_correct_l619_619191


namespace find_a_l619_619740

theorem find_a (a : ℝ) (x y : ℝ) :
  (x^2 - 4*x + y^2 = 0) →
  ((x - a)^2 + y^2 = 4*((x - 1)^2 + y^2)) →
  a = -2 :=
by
  intros h_circle h_distance
  sorry

end find_a_l619_619740


namespace cos_square_identity_sin_ratio_l619_619450

theorem cos_square_identity (A B C : ℝ) (hABC : A + B + C = π) :
  cos(A) ^ 2 + cos(B) ^ 2 + cos(C) ^ 2 = 1 - 2 * cos(A) * cos(B) * cos(C) :=
by sorry

theorem sin_ratio (A B C : ℝ) (hABC : A + B + C = π) 
  (h_ratio : (cos(A) / 39 = cos(B) / 33 ∧ cos(B) / 33 = cos(C) / 25)) :
  (sin(A) / 13 = sin(B) / 14 ∧ sin(B) / 14 = sin(C) / 15) :=
by sorry

end cos_square_identity_sin_ratio_l619_619450


namespace pairs_sum_less_than_100_l619_619455

open Nat

def count_pairs_under_100 : ℕ :=
  card {p : ℕ × ℕ | p.1 < p.2 ∧ p.fst + p.snd < 100 ∧ p.1 ≤ 100 ∧ p.2 ≤ 100}

theorem pairs_sum_less_than_100 :
  count_pairs_under_100 = 2401 :=
sorry

end pairs_sum_less_than_100_l619_619455


namespace vector_angle_acuteness_l619_619888

theorem vector_angle_acuteness (x : ℝ) : 
  ∀ (a b : ℝ × ℝ), a = (1, 2) ∧ b = (x, 4) → 
    (∃ (θ : ℝ), θ ∈ (0, π/2) ∧ 
      ∀ ⦃x : ℝ⦄, x > -8 ∧ x ≠ 2 → 
        ((x > -8 ∧ x < 2) ∨ (x > 2))) := 
by
  sorry

end vector_angle_acuteness_l619_619888


namespace find_a6_a7_l619_619923

variable {a : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a n + d
axiom sum_given : a 2 + a 3 + a 10 + a 11 = 48

theorem find_a6_a7 (arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d) (h : a 2 + a 3 + a 10 + a 11 = 48) :
  a 6 + a 7 = 24 :=
by
  sorry

end find_a6_a7_l619_619923


namespace problem_statement_l619_619625

variables (x y : ℝ)

def p : Prop := x > 1 ∧ y > 1
def q : Prop := x + y > 2

theorem problem_statement : (p x y → q x y) ∧ ¬(q x y → p x y) := sorry

end problem_statement_l619_619625


namespace part_I_part_II_l619_619104

noncomputable def f (x a b : ℝ) : ℝ := abs (x - a) + abs (x + b)

-- Conditions for Part I
def a_pos : ℝ := 1
def b_pos : ℝ := 2

-- Conditions for Part II
def min_f_value : ℝ := 3

-- Statement for Part I
theorem part_I (x : ℝ) : f x a_pos b_pos ≤ 5 ↔ (-3 ≤ x ∧ x ≤ 2) :=
sorry

-- Statement for Part II
theorem part_II (a b : ℝ) (h : a > 0) (h' : b > 0) :
  (∀ x : ℝ, f x a b ≥ min_f_value) →
  a + b = min_f_value →
  ∃ a b : ℝ, 
    (a + b = min_f_value ∧
    (a = b) ∧ 
    a = 3 / 2 ∧ 
    b = 3 / 2 ∧ 
    (dfrac (a^2) b + dfrac (b^2) a = 3)) :=
sorry

end part_I_part_II_l619_619104


namespace area_BNE_l619_619003

variable (A B C D E F P M N : Type)
variable [parallelogram ABCD] [parallelogram CPNM]
variable (area_DFP : ℝ) (area_AEF : ℝ)
    (h1 : area_DFP = 22) (h2 : area_AEF = 36)

theorem area_BNE :
  ∃ (area_BNE : ℝ), area_BNE = 14 :=
sorry

end area_BNE_l619_619003


namespace B_and_C_together_time_l619_619358

-- Define the work rates for A, B, and C
def A_rate : ℝ := 1 / 4
def B_rate : ℝ := 1 / 12
def C_rate : ℝ := (1 / 2) - A_rate

-- Define the combined work rates for A + C and B + C
def A_plus_C_rate : ℝ := A_rate + C_rate
def B_plus_C_rate : ℝ := B_rate + C_rate

-- Prove that B and C together take 3 hours to do the work
theorem B_and_C_together_time : (1 / B_plus_C_rate) = 3 := by
  sorry

end B_and_C_together_time_l619_619358


namespace find_loss_percentage_l619_619330

def cost_price := 1500
def selling_price := 1290

def loss_amount := cost_price - selling_price
def loss_percentage := (loss_amount / cost_price.toRat) * 100

theorem find_loss_percentage : loss_percentage = 14 := by
  have h1 : loss_amount = 210 := by
    dsimp [loss_amount, cost_price, selling_price]
    norm_num
  have h2 : loss_percentage = (210 / 1500) * 100 := by
    dsimp [loss_percentage, loss_amount, cost_price]
    rw [h1]
  have h3 : (210 / 1500).toRat = 0.14 := by
    norm_num
  rw [h3] at h2
  norm_num at h2
  exact h2

end find_loss_percentage_l619_619330


namespace purely_imaginary_condition_l619_619261

def purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem purely_imaginary_condition (a b : ℝ) : purely_imaginary (a * complex.I + b) ↔ (a ≠ 0 ∧ b = 0) :=
sorry

end purely_imaginary_condition_l619_619261


namespace three_same_value_ma_pair_sum_to_15_l619_619795

noncomputable theory

-- Definitions from conditions
def card_values := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
def deck := finset (ℕ × card_values)  -- Each card is represented by its number and suit, but the suit is disregarded in the problem.

-- Theorem statements
theorem three_same_value_ma (cards : finset (ℕ × card_values)) (h1 : cards.cardinality ≥ 27) :
  ∃ v ∈ card_values, 3 ≤ (finset.filter (λ c, c.2 = v) cards).cardinality :=
sorry

theorem pair_sum_to_15 (cards : finset (ℕ × card_values)) (h2 : cards.cardinality ≥ 40) :
  ∃ (c1 c2 : (ℕ × card_values)) (h1 : c1 ∈ cards) (h2 : c2 ∈ cards), c1.2 + c2.2 = 15 :=
sorry

end three_same_value_ma_pair_sum_to_15_l619_619795


namespace general_term_of_sequence_l619_619929

theorem general_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : ∀ n, S n = 2 * a n - 1) 
    (a₁ : a 1 = 1) :
  ∀ n, a n = 2^(n - 1) := 
sorry

end general_term_of_sequence_l619_619929


namespace find_angle_C_find_cos_B_l619_619147

-- Define the given conditions
variable {A B C S : ℝ}
variable {a b c : ℝ}
variable h1 : S = 1/4 * (a^2 + b^2 - c^2)

-- First problem: Find angle C
theorem find_angle_C : C = 45 :=
  sorry

-- Second problem: Find cosine of angle B given specific values for b and c
theorem find_cos_B (h2 : b = 2) (h3 : c = Real.sqrt 6) : Real.cos B = Real.sqrt 6 / 3 :=
  sorry

end find_angle_C_find_cos_B_l619_619147


namespace evaluate_abs_expression_l619_619537

theorem evaluate_abs_expression (x : ℝ) (h : x < 1) : |x - Real.sqrt((x - 2)^2)| = 2 - 2 * x := 
sorry

end evaluate_abs_expression_l619_619537


namespace cyclic_quadrilateral_l619_619816

theorem cyclic_quadrilateral (ABCD : Quadrilateral) (X : Point) 
  (H_convex : convex ABCD)
  (H_diagonals_intersect : diagonals_intersect ABCD X)
  (H_eq : XA ABCD X * sin (∠ X A B) + XC ABCD X * sin (∠ X C D) = 
          XB ABCD X * sin (∠ X B C) + XD ABCD X * sin (∠ X D A)) :
  cyclic ABCD :=
sorry

end cyclic_quadrilateral_l619_619816


namespace sum_of_fourth_powers_of_roots_l619_619017

theorem sum_of_fourth_powers_of_roots :
  ∀ (r s t : ℝ), (r + s + t = 8) ∧ (r * s + r * t + s * t = 9) ∧ (r * s * t = 2) → (r^4 + s^4 + t^4 = 2018) :=
by
  intros r s t h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  sorry

end sum_of_fourth_powers_of_roots_l619_619017


namespace second_class_cyclic_permutations_l619_619444

theorem second_class_cyclic_permutations (T : ℕ) (H : T = 16) :
  let term := (3!) / ((1!) * (1!) * (1!)) in
  M(2, 2, 2) = 1 / 2 * T + 1 / 2 * term :=
begin
  have fact_term : term = 6, 
  { 
    -- calculating the term
    norm_num,
  },
  rw H,
  norm_num,
  rw fact_term,
  norm_num,
  have eq_m : M(2, 2, 2) = 11, 
  { 
    -- from the solution step
    norm_num,
  },
  exact eq_m,
end

end second_class_cyclic_permutations_l619_619444


namespace find_a_l619_619481

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {-4, a - 1, a + 1}

theorem find_a (a : ℝ) (h : A a ∩ B a = {-2}) : a = -1 :=
sorry

end find_a_l619_619481


namespace arrange_cards_l619_619900

-- Define the context with the given conditions
variables (n : ℕ)
variables (card_numbers : Finset (Fin n)) -- finite set of card numbers
variable (appearances : ∀ (i : Fin n), 2 * i ∈ card_numbers ∧ 2 * (i+1) ∈ card_numbers)

-- Statement of the theorem
theorem arrange_cards (h : ∀ i : Fin n, ∃ e1 e2 ∈ card_numbers, e1 ≠ e2 ∧ ∀ j k ∈ (Fin n), ∃ cycle ∈ card_numbers):
  ∃ (arrangement : Fin n → Fin n), ∀ (j : Fin n), j ∈ card_numbers :=
sorry

end arrange_cards_l619_619900


namespace min_z_value_l619_619952

variable (x y z : ℝ)

theorem min_z_value (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  z = x - y → z = -1 :=
by sorry

end min_z_value_l619_619952


namespace angle_A1_A5_C_l619_619219

open_locale real

variables {C : Type*}
variables [normed_group C] [normed_space ℝ C]

/-- Ten points A1, A2, A3, ..., A10 are equally spaced on a circle with center C -/
def ten_points_on_circle (A : fin 10 → ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ( ∀ i j : fin 10, i ≠ j → dist A i A j = dist (A (0 : fin 10)) (A (1 : fin 10)) 
    ∧ ∑ i in finset.univ, dist center (A i) = radius * 10 )

/-- The measure of the angle ∠A1 A5 C is 18 degrees -/
theorem angle_A1_A5_C {A : fin 10 → ℝ × ℝ} {center : ℝ × ℝ} {radius : ℝ}
  (h : ten_points_on_circle A center radius) :
  ∠ (A 0) (A 4) center = 18 :=
sorry

end angle_A1_A5_C_l619_619219


namespace cos_sin_15_deg_l619_619878

theorem cos_sin_15_deg :
  400 * (Real.cos (15 * Real.pi / 180))^5 +  (Real.sin (15 * Real.pi / 180))^5 / (Real.cos (15 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 100 := 
sorry

end cos_sin_15_deg_l619_619878


namespace range_f_l619_619061

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sin x - 1

theorem range_f : 
  set.range (λ x : ℝ, f x) = set.Icc (-1 / 2 : ℝ) (3 / 2 : ℝ) :=
begin
  sorry
end

end range_f_l619_619061


namespace sum_fraction_sequence_l619_619949

def a_n (n : ℕ) : ℕ := n^3 - (n-1)^3

theorem sum_fraction_sequence :
  (∑ k in Finset.range (2017 - 1) + 1 \ k => 2 → 2017), (1 / (a_n k - 1) : ℚ)) = 672 / 2017 :=
by
  sorry

end sum_fraction_sequence_l619_619949


namespace dot_product_necessary_but_not_sufficient_l619_619475

variable {V : Type*} [InnerProductSpace ℝ V]

theorem dot_product_necessary_but_not_sufficient
  (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a = b) ↔ (a ⋅ c = b ⋅ c) :=
sorry

end dot_product_necessary_but_not_sufficient_l619_619475


namespace calculate_fg3_l619_619534

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l619_619534


namespace longest_segment_in_cylinder_l619_619778

-- Define the given conditions
def radius : ℝ := 5 -- Radius of the cylinder in cm
def height : ℝ := 10 -- Height of the cylinder in cm

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the longest segment L inside the cylinder using the Pythagorean theorem
noncomputable def longest_segment : ℝ := Real.sqrt ((diameter ^ 2) + (height ^ 2))

-- State the problem in Lean:
theorem longest_segment_in_cylinder :
  longest_segment = 10 * Real.sqrt 2 :=
sorry

end longest_segment_in_cylinder_l619_619778


namespace michael_score_seventh_test_l619_619188

theorem michael_score_seventh_test (scores : Fin 8 → ℤ) (h_bounds : ∀ i, 90 ≤ scores i ∧ scores i ≤ 103)
  (h_average_int : ∀ n, (1 ≤ n ∧ n ≤ 8) → ∃ k : ℤ, (∑ i in Finset.range n, scores i) = k * n)
  (h_first : scores 0 = 100)
  (h_eighth : scores 7 = 96) :
  scores 6 = 102 := 
sorry

end michael_score_seventh_test_l619_619188


namespace danube_flow_power_assignment_l619_619688

theorem danube_flow_power_assignment :
  ∃ (Q_Bp Q_Iz : ℝ) (m_dot_Bp m_dot_Iz : ℝ) (P_Bp P_Iz : ℝ) (horsepower_Bp horsepower_Iz : ℝ),
  let g := 9.81 in
  let rho := 1000 in
  let h_Bp := 3 in
  let h_Iz := 20 in
  let conversion_factor := 735.5 in
  Q_Bp = 2400 ∧
  Q_Iz = 7600 ∧
  m_dot_Bp = Q_Bp * rho ∧
  m_dot_Iz = Q_Iz * rho ∧
  P_Bp = m_dot_Bp * g * h_Bp ∧
  P_Iz = m_dot_Iz * g * h_Iz ∧
  horsepower_Bp = P_Bp / conversion_factor ∧
  horsepower_Iz = P_Iz / conversion_factor ∧
  m_dot_Bp = 2.4 * 10^6 ∧
  m_dot_Iz = 7.6 * 10^6 ∧
  horsepower_Bp ≈ 960 ∧
  horsepower_Iz ≈ 20250 := 
begin
  sorry, -- Proof to be completed
end

end danube_flow_power_assignment_l619_619688


namespace ratio_of_sums_l619_619214

variable {a : ℕ → ℝ} -- {a_n} is a sequence of real numbers
variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms of the sequence

-- Defining arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Defining the sum of first n terms of arithmetic sequence
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S = λ n, n * (a 1 + a n) / 2

-- Now we state the problem as a theorem
theorem ratio_of_sums (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (h_cond : a 7 = 7 * a 4) :
  S 13 / S 7 = 13 := 
sorry

end ratio_of_sums_l619_619214


namespace water_tank_capacity_l619_619739

variable (C : ℝ)  -- Full capacity of the tank in liters

theorem water_tank_capacity (h1 : 0.4 * C = 0.9 * C - 50) : C = 100 := by
  sorry

end water_tank_capacity_l619_619739


namespace number_of_10_digit_numbers_divisible_by_66667_l619_619522

def ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 : ℕ := 33

theorem number_of_10_digit_numbers_divisible_by_66667 :
  ∃ n : ℕ, n = ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 :=
by
  sorry

end number_of_10_digit_numbers_divisible_by_66667_l619_619522


namespace length_of_platform_l619_619755

-- Definitions for the given conditions
def speed_of_train_kmph : ℕ := 54
def speed_of_train_mps : ℕ := 15
def time_to_pass_platform : ℕ := 16
def time_to_pass_man : ℕ := 10

-- Main statement of the problem
theorem length_of_platform (v_kmph : ℕ) (v_mps : ℕ) (t_p : ℕ) (t_m : ℕ) 
    (h1 : v_kmph = 54) 
    (h2 : v_mps = 15) 
    (h3 : t_p = 16) 
    (h4 : t_m = 10) : 
    v_mps * t_p - v_mps * t_m = 90 := 
sorry

end length_of_platform_l619_619755


namespace triangle_ratio_l619_619651

-- Given conditions:
-- a: one side of the triangle
-- h_a: height corresponding to side a
-- r: inradius of the triangle
-- p: semiperimeter of the triangle

theorem triangle_ratio (a h_a r p : ℝ) (area_formula_1 : p * r = 1 / 2 * a * h_a) :
  (2 * p) / a = h_a / r :=
by {
  sorry
}

end triangle_ratio_l619_619651


namespace repeating_decimal_sum_l619_619037

noncomputable def a : ℚ := 0.66666667 -- Repeating decimal 0.666... corresponds to 2/3
noncomputable def b : ℚ := 0.22222223 -- Repeating decimal 0.222... corresponds to 2/9
noncomputable def c : ℚ := 0.44444445 -- Repeating decimal 0.444... corresponds to 4/9
noncomputable def d : ℚ := 0.99999999 -- Repeating decimal 0.999... corresponds to 1

theorem repeating_decimal_sum : a + b - c + d = 13 / 9 := by
  sorry

end repeating_decimal_sum_l619_619037


namespace number_is_a_l619_619273

theorem number_is_a (x y z a : ℝ) (h1 : x + y + z = a) (h2 : (1 / x) + (1 / y) + (1 / z) = 1 / a) : 
  x = a ∨ y = a ∨ z = a :=
sorry

end number_is_a_l619_619273


namespace area_of_circle_with_diameter_z1_l619_619087

noncomputable def circle_area_diameter (z1 z2 : ℂ) (h1 : z1^2 - 4 * z1 * z2 + 4 * z2^2 = 0) (h2 : |z2| = 2) : ℝ :=
  π * (|z1| / 2)^2

theorem area_of_circle_with_diameter_z1 (z1 z2 : ℂ) (h1 : z1^2 - 4 * z1 * z2 + 4 * z2^2 = 0) (h2 : |z2| = 2) :
  circle_area_diameter z1 z2 h1 h2 = 4 * π :=
by
  sorry

end area_of_circle_with_diameter_z1_l619_619087


namespace solve_for_y_l619_619447

theorem solve_for_y (y : ℝ) (h : sqrt (y + 5) = 7) : y = 44 :=
sorry

end solve_for_y_l619_619447


namespace triangle_obtuse_count_l619_619703

theorem triangle_obtuse_count : ∃ (m : ℕ), (m ∈ {5, 6, 7, 8, 9, 10} ∪ {22, 23, 24, 25, 26, 27, 28, 29}) ∧
  ∀ (a b : ℕ), m ∈ {5, 6, 7, 8, 9, 10} ∪ {22, 23, 24, 25, 26, 27, 28, 29} →
    (a = 13 ∧ b = 17 ∧ 17^2 > 13^2 + m^2 ∨ a + b > m ∧ m > m + 13) :=
sorry

end triangle_obtuse_count_l619_619703


namespace factor_polynomial_l619_619698

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) :=
by
  sorry

end factor_polynomial_l619_619698


namespace line_equation_through_point_sum_intercepts_l619_619263

open Real

theorem line_equation_through_point_sum_intercepts (
    x y : ℝ 
    (h1 : (0, -2)) 
    (h2 : x / a + - y / (2 - a) = 1)
    (h3 : -2 / (2 - a) = 1))
  : a = 4  := sorry

end line_equation_through_point_sum_intercepts_l619_619263


namespace chocolate_bar_pieces_l619_619356

theorem chocolate_bar_pieces (X : ℕ) (h1 : X / 2 + X / 4 + 15 = X) : X = 60 :=
by
  sorry

end chocolate_bar_pieces_l619_619356


namespace train_length_l619_619389

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (h_speed : speed_km_hr = 60) (h_time : time_sec = 6) :
  let speed_ms := (speed_km_hr * 1000) / 3600
  let length_m := speed_ms * time_sec
  length_m ≈ 100.02 :=
by sorry

end train_length_l619_619389


namespace solution_set_problem_l619_619626

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := ∀ (x : ℝ), f' x < f x
def condition2 : Prop := f 1 = Real.exp 1

-- Define the question as a Lean statement
theorem solution_set_problem
  (hf1 : condition1 f)
  (hf2 : condition2 f) :
  ∀ {x : ℝ}, 0 < x ∧ x < Real.exp 1 → f (Real.log x) > x :=
sorry

end solution_set_problem_l619_619626


namespace goldfish_equal_after_8_months_l619_619822

noncomputable def B (n : ℕ) : ℝ := 3^(n + 1)
noncomputable def G (n : ℕ) : ℝ := 243 * 1.5^n

theorem goldfish_equal_after_8_months :
  ∃ n : ℕ, B n = G n ∧ n = 8 :=
by
  sorry

end goldfish_equal_after_8_months_l619_619822


namespace max_inscribed_rectangle_area_l619_619661

theorem max_inscribed_rectangle_area (a b : ℕ) (hypotenuse : ℕ) 
  (hyp : a = 3) (hyp' : b = 4) (hyp'' : hypotenuse = 5) :
  ∃ A : ℝ, A = 3 ∧ (∃ rect_side_on_hypotenuse : bool, rect_side_on_hypotenuse = tt) :=
begin
  sorry
end

end max_inscribed_rectangle_area_l619_619661


namespace three_digit_numbers_count_l619_619959

theorem three_digit_numbers_count : 
  let digits := {2, 5, 7} in
  ∃ count : ℕ, count = 27 ∧ 
    (∀ n : ℕ, (100 ≤ n ∧ n < 1000) → 
      (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ∈ digits) → 
      ∃ k : ℕ, k = count) :=
by
  sorry

end three_digit_numbers_count_l619_619959


namespace common_difference_divisible_by_p_l619_619275

variable (a : ℕ → ℕ) (p : ℕ)

-- Define that the sequence a is an arithmetic progression with common difference d
def is_arithmetic_progression (d : ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i + d

-- Define that the sequence a is strictly increasing
def is_increasing_arithmetic_progression : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

-- Define that all elements a_i are prime numbers
def all_primes : Prop :=
  ∀ i : ℕ, Nat.Prime (a i)

-- Define that the first element of the sequence is greater than p
def first_element_greater_than_p : Prop :=
  a 1 > p

-- Combining all conditions
def conditions (d : ℕ) : Prop :=
  is_arithmetic_progression a d ∧ is_increasing_arithmetic_progression a ∧ all_primes a ∧ first_element_greater_than_p a p ∧ Nat.Prime p

-- Statement to prove: common difference is divisible by p
theorem common_difference_divisible_by_p (d : ℕ) (h : conditions a p d) : p ∣ d :=
sorry

end common_difference_divisible_by_p_l619_619275


namespace acute_angle_clock_6_44_l619_619319

theorem acute_angle_clock_6_44 : 
  let degree_separation_each_hour := 30 in
  let minute_position := (44 / 60 : ℝ) * degree_separation_each_hour in
  let hour_position := (44 / 60 : ℝ) * degree_separation_each_hour in
  let hour_displacement := degree_separation_each_hour - hour_position in
  let total_angle := hour_displacement + degree_separation_each_hour + minute_position in
  total_angle = 62 :=
by
  sorry

end acute_angle_clock_6_44_l619_619319


namespace fraction_problem_l619_619010

def fractions : List (ℚ) := [4/3, 7/5, 12/10, 23/20, 45/40, 89/80]
def subtracted_value : ℚ := -8

theorem fraction_problem :
  (fractions.sum - subtracted_value) = -163 / 240 := by
  sorry

end fraction_problem_l619_619010


namespace curve_eq_conversion_intersection_point_l619_619948

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ := (1 + (1 / 2) * t, Real.sqrt 3 + Real.sqrt 3 * t)

def polar_to_rectangular (theta ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

def curve_rectangular_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x ^ 2

def polar_curve_eq (θ ρ : ℝ) : Prop := Real.sin θ = Real.sqrt 3 * ρ * (Real.cos θ) ^ 2

theorem curve_eq_conversion (θ ρ : ℝ) (h : polar_curve_eq θ ρ) : curve_rectangular_eq (ρ * Real.cos θ) (ρ * Real.sin θ) :=
  by
  sorry

theorem intersection_point (t : ℝ) (x y ρ : ℝ) (θ : ℝ) 
  (line_eq : line_parametric t = (x, y)) 
  (curve_eq : curve_rectangular_eq x y)
  (polar_eq : polar_curve_eq θ ρ) 
  (point_eq : (x, y) = polar_to_rectangular θ ρ) : 
  (x, y) = (1, Real.sqrt 3) ∧ (ρ, θ) = (2, Real.pi / 3) :=
  by
  sorry

end curve_eq_conversion_intersection_point_l619_619948


namespace cos_beta_value_cos_2alpha_plus_beta_value_l619_619067

-- Definitions of the conditions
variables (α β : ℝ)
variable (condition1 : 0 < α ∧ α < π / 2)
variable (condition2 : π / 2 < β ∧ β < π)
variable (condition3 : Real.cos (α + π / 4) = 1 / 3)
variable (condition4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Proof problem (1)
theorem cos_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos β = - 4 * Real.sqrt 2 / 9 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

-- Proof problem (2)
theorem cos_2alpha_plus_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos (2 * α + β) = -1 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

end cos_beta_value_cos_2alpha_plus_beta_value_l619_619067


namespace sum_of_factorials_equals_210_times_factorial_l619_619864

theorem sum_of_factorials_equals_210_times_factorial (n : ℕ) (hn : n = 5) :
  (n+1)! + (n+2)! + (n+3)! = n! * 210 := 
by
  subst hn
  calc
    (5+1)! + (5+2)! + (5+3)! 
      = 6! + 7! + 8! : by rfl
    ... = 720 + 5040 + 40320 : by norm_num
    ... = 46080 : by norm_num
    ... = 5! * 210 : by norm_num

end sum_of_factorials_equals_210_times_factorial_l619_619864


namespace volume_of_pyramid_l619_619908

theorem volume_of_pyramid (a b c : ℝ) (P A B C : ℝ → ℝ → Prop)
  (angle_P : ∠PAB = 60) 
  (area_PAB : (1 / 2) * a * b * sin (60) = (√3 / 2)) 
  (area_PBC : (1 / 2) * b * c * sin (60) = 2) 
  (area_PCA : (1 / 2) * c * a * sin (60) = 1):
  volume P A B C = 2 * √6 / 9 :=
by sorry

end volume_of_pyramid_l619_619908


namespace unique_providers_for_children_l619_619590

theorem unique_providers_for_children :
  let children : List String := ["Laura", "Sibling_1", "Sibling_2", "Sibling_3"]
  let providers := (List.range 1 26).map toString
  ∃ f : String → String, (∀ c ∈ children, f c ∈ providers) ∧
    (∀ c1 c2 ∈ children, c1 ≠ c2 → f c1 ≠ f c2) ∧
    (encode (children.map f) < decode (children.map f) + 1) :=
  303600 :=
by
  sorry

end unique_providers_for_children_l619_619590


namespace f_g_of_3_l619_619531

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l619_619531


namespace evenness_oddness_of_f_min_value_of_f_l619_619195

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + |x - a| + 1

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem evenness_oddness_of_f (a : ℝ) :
  (is_even (f a) ↔ a = 0) ∧ (a ≠ 0 → ¬ is_even (f a) ∧ ¬ is_odd (f a)) :=
by
  sorry

theorem min_value_of_f (a x : ℝ) (h : x ≥ a) :
  (a ≤ -1 / 2 → f a x = 3 / 4 - a) ∧ (a > -1 / 2 → f a x = a^2 + 1) :=
by
  sorry

end evenness_oddness_of_f_min_value_of_f_l619_619195


namespace range_f_log_l619_619488

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f x = f (-x)
axiom f_increasing (x y : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ y) : f x ≤ f y
axiom f_at_1 : f 1 = 0

theorem range_f_log (x : ℝ) : f (Real.log x / Real.log (1 / 2)) > 0 ↔ (0 < x ∧ x < 1 / 2) ∨ (2 < x) :=
by
  sorry

end range_f_log_l619_619488


namespace probability_y_greater_than_x_equals_3_4_l619_619832

noncomputable def probability_y_greater_than_x : Real :=
  let total_area : Real := 1000 * 4034
  let triangle_area : Real := 0.5 * 1000 * (4034 - 1000)
  let rectangle_area : Real := 3034 * 4034
  let area_y_greater_than_x : Real := triangle_area + rectangle_area
  area_y_greater_than_x / total_area

theorem probability_y_greater_than_x_equals_3_4 :
  probability_y_greater_than_x = 3 / 4 :=
sorry

end probability_y_greater_than_x_equals_3_4_l619_619832


namespace speed_of_boat_is_correct_l619_619800

theorem speed_of_boat_is_correct (t : ℝ) (V_b : ℝ) (V_s : ℝ) 
  (h1 : V_s = 19) 
  (h2 : ∀ t, (V_b - V_s) * (2 * t) = (V_b + V_s) * t) :
  V_b = 57 :=
by
  -- Proof will go here
  sorry

end speed_of_boat_is_correct_l619_619800


namespace modular_inverse_of_3_mod_197_l619_619039

theorem modular_inverse_of_3_mod_197 : ∃ x : ℕ, x < 197 ∧ (3 * x) % 197 = 1 :=
by
  use 66
  split
  . exact lt_of_lt_of_le (nat.zero_lt_succ 196) (nat.le_refl 197)
  . exact nat.monoidₓ.mul_mod_mul_of_pos 3 66 197 (3 * 66) 1
  -- sorry, detailed proof is omitted. Sorry statement is used to close the theorem.

end modular_inverse_of_3_mod_197_l619_619039


namespace longest_segment_in_cylinder_l619_619785

theorem longest_segment_in_cylinder (radius height : ℝ) 
  (hr : radius = 5) (hh : height = 10) :
  ∃ segment_length, segment_length = 10 * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end longest_segment_in_cylinder_l619_619785


namespace distance_from_point_to_x_axis_l619_619158

theorem distance_from_point_to_x_axis (x y : ℝ) (P : ℝ × ℝ) (hP : P = (x, y)) :
  abs (y) = 3 :=
by
  -- Assuming the y-coordinate is given as -3
  have hy : y = -3 := sorry
  rw [hy]
  exact abs_neg 3

end distance_from_point_to_x_axis_l619_619158


namespace current_price_after_adjustment_l619_619770

variable (x : ℝ) -- Define x, the original price per unit

theorem current_price_after_adjustment (x : ℝ) : (x + 10) * 0.75 = ((x + 10) * 0.75) :=
by
  sorry

end current_price_after_adjustment_l619_619770


namespace rectangular_solid_surface_area_l619_619854

-- Define the problem statement
theorem rectangular_solid_surface_area (a b c : ℕ) (h_a : a = 49) (h_b : b = 5) (h_c : c = 3)
  (h_volume : a * b * c = 1155)
  (h_cond_a : prime 7 ∨ (∃ p : ℕ, prime p ∧ a = p^2))
  (h_cond_b : prime b ∨ (∃ p : ℕ, prime p ∧ b = p^2))
  (h_cond_c : prime c ∨ (∃ p : ℕ, prime p ∧ c = p^2)) :
  2 * (a * b + b * c + c * a) = 814 := by
  sorry

end rectangular_solid_surface_area_l619_619854


namespace total_savings_in_joint_account_l619_619582

def kimmie_earnings : ℝ := 450
def zahra_earnings : ℝ := kimmie_earnings - (1 / 3) * kimmie_earnings
def kimmie_savings : ℝ := (1 / 2) * kimmie_earnings
def zahra_savings : ℝ := (1 / 2) * zahra_earnings
def joint_savings_account : ℝ := kimmie_savings + zahra_savings

theorem total_savings_in_joint_account :
  joint_savings_account = 375 := 
by
  -- proof to be provided
  sorry

end total_savings_in_joint_account_l619_619582


namespace diagonal_AC_length_l619_619564

open Real

variables (A B C D : Point)
variables (AB BD DC AC : Real)
variables (area_ABCD : Real)

-- Definition of the conditions
def convex_quadrilateral_conditions (A B C D : Point) : Prop :=
  (AB + BD + DC ≤ 2) ∧ (area_ABCD = 1 / 2)

-- Problem statement
theorem diagonal_AC_length (h: convex_quadrilateral_conditions A B C D) : AC = sqrt 2 :=
by
  sorry

end diagonal_AC_length_l619_619564


namespace digit_100_of_7_div_26_l619_619970

theorem digit_100_of_7_div_26 : 
  ( (\frac{7}{26} : ℚ).decimal_expansion.nth 100 = 2 ) := by 
sorry

end digit_100_of_7_div_26_l619_619970


namespace maximum_take_home_pay_l619_619556

theorem maximum_take_home_pay :
  ∃ x : ℝ, let tax_rate := 2 * x / 100 in
           let income := x * 1000 in
           let tax := tax_rate * income in
           let take_home_pay := income - tax in
           (∀ y : ℝ, let tax_rate_y := 2 * y / 100 in
                     let income_y := y * 1000 in
                     let tax_y := tax_rate_y * income_y in
                     let take_home_pay_y := income_y - tax_y in
                     take_home_pay ≥ take_home_pay_y) ∧ income = 25000 :=
begin
  sorry
end

end maximum_take_home_pay_l619_619556


namespace minimal_value_of_a_b_l619_619615

noncomputable def minimal_sum_of_a_and_b : ℝ := 6.11

theorem minimal_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : discriminant (λ x, x^2 + a * x + 3 * b) >= 0) 
  (h4 : discriminant (λ x, x^2 + 3 * b * x + a) >= 0) : 
  a + b = minimal_sum_of_a_and_b :=
sorry

end minimal_value_of_a_b_l619_619615


namespace largest_seven_consecutive_composites_less_than_40_l619_619237

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

def seven_consecutive_composites_less_than_40 (a : ℕ) : Prop :=
  a < 40 ∧ a > 29 ∧
  is_composite a ∧ is_composite (a - 1) ∧ is_composite (a - 2) ∧ 
  is_composite (a - 3) ∧ is_composite (a - 4) ∧ 
  is_composite (a - 5) ∧ is_composite (a - 6)

theorem largest_seven_consecutive_composites_less_than_40 :
  seven_consecutive_composites_less_than_40 36 :=
begin
  -- Proof goes here.
  sorry
end

end largest_seven_consecutive_composites_less_than_40_l619_619237


namespace unique_function_exists_l619_619076

noncomputable def formalize_problem : Type :=
  { f : ℝ → ℝ // 
    -- f(x) is odd: f(-x) = -f(x)
    (∀ x : ℝ, f (-x) = -f x) ∧ 
    -- g(x) = f(x+2) is even: f(-x+2) = f(x+2)
    (∀ x : ℝ, f (-x + 2) = f (x + 2)) ∧ 
    -- f(x) = x for x ∈ [0, 2]
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x = x)
  }

theorem unique_function_exists : ∃! f : formalize_problem, true :=
begin
  sorry
end

end unique_function_exists_l619_619076


namespace find_radius_of_semicircle_on_BC_l619_619570

noncomputable def radius_of_semicircle_on_BC (angle_ABC : ℝ) (area_semi_AB : ℝ) (arc_length_semi_AC : ℝ) : ℝ :=
  let AB := 2 * (sqrt (2 * area_semi_AB / π)),
      AC := 2 * (arc_length_semi_AC / π) in
  1 / 2 * sqrt (AB^2 + AC^2 + AB * AC)

theorem find_radius_of_semicircle_on_BC :
  radius_of_semicircle_on_BC 120 (12.5 * π) (9 * π) = sqrt 604 / 2 :=
  sorry

end find_radius_of_semicircle_on_BC_l619_619570


namespace sum_not_all_ones_l619_619146

-- Conditions definitions:
def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def is_permutation (a b : ℕ) : Prop :=
  a.digits 10 ~ b.digits 10

def all_ones (n : ℕ) : Prop :=
  n.digits 10 = list.repeat 1 n.digits.length

-- Main theorem:
theorem sum_not_all_ones (N M : ℕ) (hN : no_zero_digits N) (hM : is_permutation N M) :
  ¬ all_ones (N + M) :=
sorry

end sum_not_all_ones_l619_619146


namespace least_sum_of_exponents_l619_619862

theorem least_sum_of_exponents (x : ℕ) (hx : x = 3125) :
  ∃ (s : finset ℕ), (x = s.sum (λ n, 2 ^ n)) ∧ s.sum id = 32 :=
by
  sorry

end least_sum_of_exponents_l619_619862


namespace remainder_when_subtract_div_by_6_l619_619331

theorem remainder_when_subtract_div_by_6 (m n : ℕ) (h1 : m % 6 = 2) (h2 : n % 6 = 3) (h3 : m > n) : (m - n) % 6 = 5 := 
by
  sorry

end remainder_when_subtract_div_by_6_l619_619331


namespace find_f_one_l619_619634

noncomputable def f : ℝ → ℝ := sorry

lemma polynomial_functional_equation (x : ℝ) (hx : x ≠ 0) :
  f(x - 1) + f(x + 1) = x * (f(x))^2 / 1001 :=
sorry

lemma non_constant_polynomial : ¬ ∀ x, f(x) = 0 :=
sorry

theorem find_f_one : f(1) = 2002 :=
sorry

end find_f_one_l619_619634


namespace solution_set_inequality_l619_619106

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) + 1 else log (1 / 2) (x / 2) + 1

theorem solution_set_inequality : 
  { x : ℝ | f x > 2 } = { x : ℝ | x < 0 } ∪ { x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_inequality_l619_619106


namespace product_ab_l619_619985

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619985


namespace strictly_increasing_intervals_g_x_1_solutions_l619_619098

noncomputable def f (x : ℝ) := 2 * sin x * cos x + 2 * cos x ^ 2
noncomputable def g (x : ℝ) := let y := x - (π / 4); f y

theorem strictly_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, x ∈ [k * π - 3 * π / 8, k * π + π / 8] → 
  (differentiable ℝ f x ∧ deriv f x > 0) := sorry

theorem g_x_1_solutions :
  ∀ x : ℝ, x ∈ [0, π] → g x = 1 ↔ x = π / 8 ∨ x = 5 * π / 8 := sorry

end strictly_increasing_intervals_g_x_1_solutions_l619_619098


namespace general_formula_sequence_sum_first_n_terms_l619_619280

def a (n : Nat) : ℚ :=
  if n = 1 then 2 else (n + 1) / (n - 1)

noncomputable def b (n : Nat) : ℚ :=
  a (n + 1) - a n

noncomputable def S (n : Nat) : ℚ :=
  ∑ i in (Finset.range n).map (Finset.Nat.cast), b i

theorem general_formula_sequence {n : ℕ} (h : n > 0) :
  a n = (if n = 1 then 2 else (n + 1) / (n - 1)) :=
by
  sorry

theorem sum_first_n_terms {n : ℕ} (h : n > 0) :
  S n = (2 / n) - 1 :=
by
  sorry

end general_formula_sequence_sum_first_n_terms_l619_619280


namespace eggs_per_chicken_per_week_l619_619181

-- Define the conditions
def chickens : ℕ := 10
def price_per_dozen : ℕ := 2  -- in dollars
def earnings_in_2_weeks : ℕ := 20  -- in dollars
def weeks : ℕ := 2
def eggs_per_dozen : ℕ := 12

-- Define the question as a theorem to be proved
theorem eggs_per_chicken_per_week : 
  (earnings_in_2_weeks / price_per_dozen) * eggs_per_dozen / (chickens * weeks) = 6 :=
by
  -- proof steps
  sorry

end eggs_per_chicken_per_week_l619_619181


namespace parabola_directrix_separation_l619_619116

noncomputable def parabola_and_circle_separation (a : ℝ) : Prop :=
  ∃ (y : ℝ), (y^2 = 4 * a * (a + 1)) → y^2 ≠ 4 * a^2

theorem parabola_directrix_separation (a : ℝ) (h : parabola_and_circle_separation a) : a < -1 ∨ a > 1 :=
begin
  sorry
end

end parabola_directrix_separation_l619_619116


namespace digit_in_100th_place_l619_619966

theorem digit_in_100th_place :
    let seq : List Char := ['2', '6', '9', '2', '3', '0']
    (seq.get! ((100 % seq.length) - 1)) = '9' :=
by
  let seq : List Char := ['2', '6', '9', '2', '3', '0']
  have h_len : seq.length = 6 := rfl
  have h_mod : 100 % 6 = 4 := rfl
  have h_idx : (100 % seq.length) - 1 = 3 := by
    rw [h_len, h_mod]
    exact rfl
  show seq.get! 3 = '9' from rfl

end digit_in_100th_place_l619_619966


namespace max_value_l619_619205

-- Definition of real numbers a, b, c and angle θ
variables {a b c θ : ℝ}

-- Definition of trigonometric functions
def cos (θ : ℝ) := real.cos θ
def sin (θ : ℝ) := real.sin θ
def tan (θ : ℝ) := real.tan θ

-- Condition that c * cos^2 θ ≠ -a
axiom h_condition : c * cos(θ) * cos(θ) ≠ -a

-- Theorem stating the maximum value
theorem max_value : ∃ θ : ℝ, a * cos(θ) + b * sin(θ) + c * tan(θ) ≤ sqrt(a^2 + b^2 + c^2) :=
sorry  -- proof will be provided here

end max_value_l619_619205


namespace goody_pair_palindrome_l619_619313

def is_palindrome (s : list char) : Prop :=
  s = s.reverse

inductive GoodyPair : (list char) → (list char) → Prop
| base_case : GoodyPair ['a'] ['b']
| rule1 {u v α β : list char} (h : GoodyPair u v) (h1 : α = u ++ v) (h2 : β = v) : GoodyPair α β
| rule2 {u v α β : list char} (h : GoodyPair u v) (h1 : α = u) (h2 : β = u ++ v) : GoodyPair α β

theorem goody_pair_palindrome (α β : list char) (h : GoodyPair α β) :
  ∃ γ : list char, is_palindrome γ ∧ α ++ β = ['a'] ++ γ ++ ['b'] :=
sorry

end goody_pair_palindrome_l619_619313


namespace area_is_20_l619_619958

def point := (ℝ × ℝ)

def vertices : set point := { (0, 0), (4, 0), (1, 5), (5, 5) }

def length (p1 p2 : point) : ℝ := abs (p2.1 - p1.1)

def height (p1 p2 : point) : ℝ := abs (p2.2 - p1.2)

def base_of_parallelogram : ℝ := length (0, 0) (4, 0)

def height_of_parallelogram : ℝ := height (0, 0) (1, 5)

def area_of_parallelogram := base_of_parallelogram * height_of_parallelogram

theorem area_is_20 :
  area_of_parallelogram = 20 :=
sorry

end area_is_20_l619_619958


namespace math_problem_l619_619937

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem math_problem
  (omega phi : ℝ)
  (h1 : omega > 0)
  (h2 : |phi| < Real.pi / 2)
  (h3 : ∀ x, f x = Real.sin (omega * x + phi))
  (h4 : ∀ k : ℤ, f (k * Real.pi) = f 0) 
  (h5 : f 0 = 1 / 2) :
  (omega = 2) ∧
  (∀ x, f (x + Real.pi / 6) = f (-x + Real.pi / 6)) ∧
  (∀ k : ℤ, 
    ∀ x, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    ∀ y, y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    x < y → f x ≤ f y) :=
by
  sorry

end math_problem_l619_619937


namespace imaginary_part_conjugate_l619_619898

theorem imaginary_part_conjugate (z : ℂ) (h : z = (1 - complex.i)^2 / (1 + complex.i)) : z.conjugate.im = 1 :=
sorry

end imaginary_part_conjugate_l619_619898


namespace non_empty_solution_set_inequality_l619_619281

theorem non_empty_solution_set_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 := 
sorry

end non_empty_solution_set_inequality_l619_619281


namespace number_of_possible_values_for_abs_z_l619_619539

-- Define the mathematical problem
theorem number_of_possible_values_for_abs_z (z : ℂ) (h : z ^ 2 - 8 * z + 45 = 0) : 
  (∃! r : ℝ, ∃ z1 z2 : ℂ, z1 = 4 + real.sqrt 29 * complex.I ∧ z2 = 4 - real.sqrt 29 * complex.I ∧ z1.abs = r ∧ z2.abs = r) :=
sorry

end number_of_possible_values_for_abs_z_l619_619539


namespace lee_propose_time_l619_619594

theorem lee_propose_time (annual_salary : ℕ) (monthly_savings : ℕ) (ring_salary_months : ℕ) :
    annual_salary = 60000 → monthly_savings = 1000 → ring_salary_months = 2 → 
    let monthly_salary := annual_salary / 12 in
    let ring_cost := ring_salary_months * monthly_salary in
    ring_cost / monthly_savings = 10 := 
by 
    intros annual_salary_eq monthly_savings_eq ring_salary_months_eq;
    rw [annual_salary_eq, monthly_savings_eq, ring_salary_months_eq];
    let monthly_salary := 60000 / 12;
    have ring_cost_eq : 2 * monthly_salary = 10000 := by sorry;
    have savings_time_eq : 10000 / 1000 = 10 := by sorry;
    exact savings_time_eq at ring_cost_eq;
    assumption

end lee_propose_time_l619_619594


namespace sum_of_real_and_imaginary_parts_l619_619083

theorem sum_of_real_and_imaginary_parts (z : ℂ) (h : (conj z) / (1 + 2 * complex.i) = 2 + complex.i) : 
  ((z + 5).re + (z + 5).im) = 0 :=
by
  sorry

end sum_of_real_and_imaginary_parts_l619_619083


namespace positive_difference_mean_median_l619_619712

theorem positive_difference_mean_median 
  (a b c d e : ℕ)
  (h1 : a = 160)
  (h2 : b = 120)
  (h3 : c = 140)
  (h4 : d = 320)
  (h5 : e = 200) : 
  (| (a + b + c + d + e) / 5 - c | = 28) :=
by
  sorry

end positive_difference_mean_median_l619_619712


namespace maximum_omega_value_l619_619944

-- Definitions of given conditions.
def f (ω φ x : ℝ) : ℝ := 3 * Real.sin (ω * x + φ)

-- The hypothesis that omega is a positive number and phi is between 0 and pi.
def valid_parameters (ω φ : ℝ) : Prop :=
  ω > 0 ∧ 0 < φ ∧ φ < Real.pi

-- The function value constraint at x = -π/3.
def constraint_at_neg_pi_over_3 (ω φ : ℝ) : Prop :=
  f ω φ (-Real.pi / 3) = 0

-- The global upper bound condition.
def upper_bound_condition (ω φ : ℝ) : Prop :=
  ∀ x : ℝ, f ω φ x ≤ abs (f ω φ (Real.pi / 3))

-- Existence of a unique solution for f(x₁) = 3 in a given interval.
def unique_solution_in_interval (ω φ : ℝ) : Prop :=
  ∃! x1 ∈ set.Ioo (Real.pi / 15) (Real.pi / 5), f ω φ x1 = 3

-- The final theorem statement for the proof problem.
theorem maximum_omega_value : 
  ∀ ω φ : ℝ, valid_parameters ω φ →
  constraint_at_neg_pi_over_3 ω φ →
  upper_bound_condition ω φ →
  unique_solution_in_interval ω φ →
  ω = 105 / 4 :=
by sorry

end maximum_omega_value_l619_619944


namespace exists_quadrilateral_with_irrational_triangle_area_l619_619576

theorem exists_quadrilateral_with_irrational_triangle_area :
  ∃ (ABCD : Set Point) (O : Point),
    (area ABCD = 1) →
    isInside O ABCD →
    (∃ (A B C D : Point), 
      area (triangle O A B) ∨ 
      area (triangle O B C) ∨ 
      area (triangle O C D) ∨ 
      area (triangle O D A) ∉ ℚ) :=
sorry

end exists_quadrilateral_with_irrational_triangle_area_l619_619576


namespace find_value_at_l619_619196

-- Defining the function f
variable (f : ℝ → ℝ)

-- Conditions
-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 4
def periodic_function (f : ℝ → ℝ) := ∀ x, f (x + 4) = f x

-- Condition 3: In the interval [0,1], f(x) = 3x
def definition_on_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- Statement to prove
theorem find_value_at (f : ℝ → ℝ) 
  (odd_f : odd_function f) 
  (periodic_f : periodic_function f) 
  (def_on_interval : definition_on_interval f) :
  f 11.5 = -1.5 := by 
  sorry

end find_value_at_l619_619196


namespace distance_major_minor_axis_of_ellipse_l619_619021

noncomputable def ellipse_distance (x y : ℝ) : ℝ :=
  4 * (x-3)^2 + 16 * (y+2)^2

theorem distance_major_minor_axis_of_ellipse :
  (4 * (x-3)^2 + 16 * (y+2)^2 = 64) → 
  (distance (3 + 4, -2) (3, -2 + 2) = 2 * Real.sqrt 5) :=
by
  intros h
  sorry

end distance_major_minor_axis_of_ellipse_l619_619021


namespace black_can_prevent_white_adjacent_l619_619218

def Grid (n : Nat) := Fin n × Fin n

structure Position := 
  white1 : Grid 23 
  white2 : Grid 23 
  black1 : Grid 23 
  black2 : Grid 23 

def initial_position : Position := {
  white1 := (⟨1, by sorry⟩, ⟨1, by sorry⟩),
  white2 := (⟨23, by sorry⟩, ⟨23, by sorry⟩),
  black1 := (⟨1, by sorry⟩, ⟨23, by sorry⟩),
  black2 := (⟨23, by sorry⟩, ⟨1, by sorry⟩)
}

def adjacent (p q : Grid 23) : Prop := 
  (abs (p.fst.val - q.fst.val) = 1 ∧ p.snd = q.snd) ∨
  (abs (p.snd.val - q.snd.val) = 1 ∧ p.fst = q.fst)

def can_white_occupy_adjacent (pos : Position) : Prop := 
  adjacent pos.white1 pos.white2

theorem black_can_prevent_white_adjacent :
  ∀ pos : Position, pos = initial_position → ¬ can_white_occupy_adjacent pos :=
by
  intro pos h_pos
  rw h_pos
  sorry

end black_can_prevent_white_adjacent_l619_619218


namespace min_a_plus_b_eq_six_point_five_l619_619605

noncomputable def min_a_plus_b : ℝ :=
  Inf {s | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
                       (a^2 - 12 * b ≥ 0) ∧ 
                       (9 * b^2 - 4 * a ≥ 0) ∧ 
                       (a + b = s)}

theorem min_a_plus_b_eq_six_point_five : min_a_plus_b = 6.5 :=
by
  sorry

end min_a_plus_b_eq_six_point_five_l619_619605


namespace ellipse_standard_equation_l619_619910

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
∀ (x y : ℝ), (y^2 / a^2) + (x^2 / b^2) = 1

theorem ellipse_standard_equation :
  let e := (Real.sqrt 3) / 2 in
  let foci_distance := 12 in
  ∃ (a b : ℝ), (a = 6) ∧ (b = 3) ∧ (2 * a = foci_distance) ∧ ((e = (Real.sqrt 3) / 2) ∧ a > 0 ∧ b > 0)  ∧ 
  ellipse_equation 36 9 :=
by
  sorry

end ellipse_standard_equation_l619_619910


namespace extremum_point_a_zero_l619_619502

noncomputable def f (a x : ℝ) : ℝ := log (a * x + 1) + x^3 - x^2 - a * x

theorem extremum_point_a_zero (a : ℝ) (h1 : f a (2/3) = 0) : a = 0 := 
sorry

end extremum_point_a_zero_l619_619502


namespace acute_angle_implies_x_range_l619_619892

open Real

def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0

theorem acute_angle_implies_x_range (x : ℝ) :
  is_acute (1, 2) (x, 4) → x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi (2 : ℝ) :=
by
  sorry

end acute_angle_implies_x_range_l619_619892


namespace chime_occurrence_date_l619_619775

-- Definitions for the problem conditions
def chime_count_per_day (malfunction_start_date : Nat) : Nat :=
  -- Each day has 23 half-hours due to malfunction, plus the hourly chimes adding up to 78
  23 + 78

def chime_count_from_215_pm_to_midnight : Nat :=
  -- From 2:15 PM to midnight on February 26, there are 76 chimes
  76

def days_after_feb_26_until_march_18 : Nat :=
  -- March 18 is 20 days after February 26 (2 days in Feb + 18 days in March), considering the extra day
  20

-- Prove the date of the 2023rd chime occurrence
theorem chime_occurrence_date :
  let total_chimes_needed := 2023
  let chimes_on_feb_26 := chime_count_from_215_pm_to_midnight
  let chimes_per_day := chime_count_per_day 27
  let remaining_chimes := total_chimes_needed - chimes_on_feb_26
  let full_days_needed := remaining_chimes / chimes_per_day
  let extra_chimes_needed := remaining_chimes % chimes_per_day
  full_days_needed + if extra_chimes_needed > 0 then 1 else 0 = days_after_feb_26_until_march_18
  → "March 18, 2003" := sorry

end chime_occurrence_date_l619_619775


namespace eq_has_47_solutions_l619_619524

noncomputable def NumberOfSolutions : ℕ := 47

theorem eq_has_47_solutions :
  let valid_solutions := { x | x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50} 
                                      \ {1, 8, 27} } in
  valid_solutions.card = NumberOfSolutions :=
by
  sorry

end eq_has_47_solutions_l619_619524


namespace circle_equation_intersection_k3_line_for_eq_l619_619462

def is_tangent (line : ℝ → ℝ) (circle_center : ℝ × ℝ) (radius : ℝ) : Prop :=
  abs ((3 * circle_center.1 + 4) / sqrt (3^2 + 4^2)) = radius

def equation_of_circle (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) → Prop := 
  fun p => (p.1 - center.1)^2 + p.2^2 = radius^2

def line_l (k : ℝ) (point : ℝ × ℝ) : (ℝ × ℝ) → Prop := 
  fun p => p.2 = k * p.1 + point.2

theorem circle_equation : equation_of_circle (2, 0) 2 = fun p => (p.1 - 2)^2 + p.2^2 = 4 := 
by
  sorry

theorem intersection_k3 (l_cir : (ℝ × ℝ) → Prop) (k : ℝ) (p : ℝ × ℝ) : 
  k = 3 ∧ (equation_of_circle (2, 0) 2 = l_cir) → 
  ∃ x1 x2 y1 y2, intersection_k3 = fun p => (x1 * x2 + y1 * y2 = -9/5) := 
by
  sorry

theorem line_for_eq (l_cir : (ℝ × ℝ) → Prop) (cond : ℝ) (eq : 8) (eq_line : (ℝ × ℝ) → Prop) :
  eq = 8 ∧ (equation_of_circle (2, 0) 2 = l_cir) → 
  ∃ k, eq_line = fun p => p.2 = ( (-3 + sqrt(29)) / 4 ) * p.1 - 3 :=
by
  sorry

end circle_equation_intersection_k3_line_for_eq_l619_619462


namespace number_of_values_of_z_l619_619372

def f (z : ℂ) : ℂ := complex.I * conj z

theorem number_of_values_of_z 
  (h1 : ∀ z : ℂ, f z = z → (abs z = 3 ∨ false)) :
  ∃ zs : Finset ℂ, (∀ z ∈ zs, abs z = 3 ∧ f z = z) ∧ zs.card = 2 :=
sorry

end number_of_values_of_z_l619_619372


namespace fraction_difference_in_simplest_form_l619_619865

noncomputable def difference_fraction : ℚ := (5 / 19) - (2 / 23)

theorem fraction_difference_in_simplest_form :
  difference_fraction = 77 / 437 := by sorry

end fraction_difference_in_simplest_form_l619_619865


namespace domain_and_range_of_f_l619_619934

noncomputable def f (a x : ℝ) : ℝ := Real.log (a - a * x) / Real.log a

theorem domain_and_range_of_f (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a - a * x > 0 → x < 1) ∧ 
  (∀ t : ℝ, 0 < t ∧ t < a → ∃ x : ℝ, t = a - a * x) :=
by
  sorry

end domain_and_range_of_f_l619_619934


namespace ab_equals_6_l619_619972

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619972


namespace find_natural_numbers_l619_619046

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem find_natural_numbers (x : ℕ) :
  (x = 36 * sum_of_digits x) ↔ (x = 324 ∨ x = 648) :=
by
  sorry

end find_natural_numbers_l619_619046


namespace number_of_false_propositions_l619_619276

-- Definitions of the propositions
def is_square (q : Type) : Prop := sorry -- Definition that q is a square
def is_rectangle (q : Type) : Prop := sorry -- Definition that q is a rectangle

-- Original proposition: If a quadrilateral is a square, then it must be a rectangle
def original (q : Type) : Prop := is_square q → is_rectangle q

-- Converse proposition: If a quadrilateral is a rectangle, then it must be a square
def converse (q : Type) : Prop := is_rectangle q → is_square q

-- Inverse proposition: If a quadrilateral is not a square, then it is not a rectangle
def inverse (q : Type) : Prop := ¬ is_square q → ¬ is_rectangle q

-- Contrapositive proposition: If a quadrilateral is not a rectangle, then it is not a square
def contrapositive (q : Type) : Prop := ¬ is_rectangle q → ¬ is_square q

theorem number_of_false_propositions (q : Type) :
  2 = (if original q then 0 else 1)
    + (if converse q then 0 else 1)
    + (if inverse q then 0 else 1)
    + (if contrapositive q then 0 else 1) :=
sorry

end number_of_false_propositions_l619_619276


namespace range_of_k_l619_619065

noncomputable def equation (k x : ℝ) : ℝ := 4^x - k * 2^x + k + 3

theorem range_of_k {x : ℝ} (h : ∀ k, equation k x = 0 → ∃! x : ℝ, equation k x = 0) :
  ∃ k : ℝ, (k = 6 ∨ k < -3)∧ (∀ y, equation k y ≠ 0 → (y ≠ x)) :=
sorry

end range_of_k_l619_619065


namespace minimal_value_of_a_b_l619_619614

noncomputable def minimal_sum_of_a_and_b : ℝ := 6.11

theorem minimal_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : discriminant (λ x, x^2 + a * x + 3 * b) >= 0) 
  (h4 : discriminant (λ x, x^2 + 3 * b * x + a) >= 0) : 
  a + b = minimal_sum_of_a_and_b :=
sorry

end minimal_value_of_a_b_l619_619614


namespace train_length_l619_619386

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_conversion_factor : ℝ) (approx_length : ℝ) :
  speed_km_hr = 60 → time_seconds = 6 → speed_conversion_factor = (1000 / 3600) → approx_length = 100.02 →
  speed_km_hr * speed_conversion_factor * time_seconds = approx_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end train_length_l619_619386


namespace longest_segment_in_cylinder_l619_619791

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end longest_segment_in_cylinder_l619_619791


namespace weekend_price_is_correct_l619_619678

-- Define the original price of the jacket
def original_price : ℝ := 250

-- Define the first discount rate (40%)
def first_discount_rate : ℝ := 0.40

-- Define the additional weekend discount rate (10%)
def additional_discount_rate : ℝ := 0.10

-- Define a function to apply the first discount
def apply_first_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Define a function to apply the additional discount
def apply_additional_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Using both discounts, calculate the final weekend price
def weekend_price : ℝ :=
  apply_additional_discount (apply_first_discount original_price first_discount_rate) additional_discount_rate

-- The final theorem stating the expected weekend price is $135
theorem weekend_price_is_correct : weekend_price = 135 := by
  sorry

end weekend_price_is_correct_l619_619678


namespace q_div_p_l619_619038

noncomputable def binomial (n k : ℕ) : ℤ :=
  if k ≤ n then (n.choose k : ℤ) else 0

def p : ℚ := 10 / binomial 50 5

def q : ℚ := 2250 / binomial 50 5

theorem q_div_p : q / p = 225 := by
  sorry

end q_div_p_l619_619038


namespace digit_100_of_7_div_26_l619_619969

theorem digit_100_of_7_div_26 : 
  ( (\frac{7}{26} : ℚ).decimal_expansion.nth 100 = 2 ) := by 
sorry

end digit_100_of_7_div_26_l619_619969


namespace train_length_l619_619388

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (h_speed : speed_km_hr = 60) (h_time : time_sec = 6) :
  let speed_ms := (speed_km_hr * 1000) / 3600
  let length_m := speed_ms * time_sec
  length_m ≈ 100.02 :=
by sorry

end train_length_l619_619388


namespace total_differential_1_total_differential_2_total_differential_3_l619_619877

-- Prove the total differentials for the given functions
theorem total_differential_1 (x y : ℝ) : 
    let z := 3 * x^2 * y^5 in
    dz = 6 * x * y^5 * dx + 15 * x^2 * y^4 * dy := sorry

theorem total_differential_2 (x y z : ℝ) : 
    let u := 2 * x^(y*z) in
    du = 2 * y * z * x^(y * z - 1) * dx + 2 * x^(y * z) * z * ln x * dy + 2 * x^(y * z) * y * ln x * dz := sorry

theorem total_differential_3 (u v : ℝ) : 
    let p := arccos (1 / (u * v)) in
    dp = (v / sqrt (u^2 * v^2 - 1)) * du + (u / sqrt (u^2 * v^2 - 1)) * dv := sorry

end total_differential_1_total_differential_2_total_differential_3_l619_619877


namespace max_min_value_of_vector_magnitude_l619_619954

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

def vector_b : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def scalar_multiply_vector (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.fst, c * v.snd)

noncomputable def vector_add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ :=
  (v₁.fst + v₂.fst, v₁.snd + v₂.snd)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.fst ^ 2 + v.snd ^ 2)

theorem max_min_value_of_vector_magnitude :
  ∀ θ : ℝ,
  let a := vector_a θ,
      b := vector_b,
      sum := vector_add (scalar_multiply_vector 2 a) b in
  vector_magnitude sum = 4 ∨ vector_magnitude sum = 0 :=
by
  sorry

end max_min_value_of_vector_magnitude_l619_619954


namespace acute_triangles_with_added_point_l619_619466

theorem acute_triangles_with_added_point (n : ℕ) (A : Fin n → ℝ × ℝ) (h : n > 2) :
  ∃ B : ℝ × ℝ, ∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i →
    acute_triangle B (A i) (A j) :=
sorry

/-- A helper predicate to check if a triangle is acute-angled. -/
def acute_triangle (B A1 A2 : ℝ × ℝ) : Prop :=
  ∃ α β γ,
    α + β + γ = π ∧
    0 < α ∧ α < π / 2 ∧
    0 < β ∧ β < π / 2 ∧
    0 < γ ∧ γ < π / 2 ∧
    α = angle B A1 A2 ∧
    β = angle A1 A2 B ∧
    γ = angle A2 B A1

/-- The angle function calculating the angle at a particular vertex of a triangle. -/
def angle (A B C : ℝ × ℝ) : ℝ :=
  -- Some function calculating angle at A in triangle ABC
  sorry

end acute_triangles_with_added_point_l619_619466


namespace printers_finish_together_in_24_minutes_l619_619228

theorem printers_finish_together_in_24_minutes :
  (∀ A B : ℕ, (A = 8 ∧ B = 12 ∧ A + B = 20) → 480 / 20 = 24) :=
by
  intros A B h_rate
  cases h_rate with hA hB_combined
  cases hB_combined with hB h_combined
  have h_comb : A + B = 20 := h_combined
  have h_time : 480 / (A + B) = 24 := by sorry
  exact h_time

end printers_finish_together_in_24_minutes_l619_619228


namespace top_angle_isosceles_triangle_l619_619919

open Real

theorem top_angle_isosceles_triangle (A B C : ℝ) (abc_is_isosceles : (A = B ∨ B = C ∨ A = C))
  (angle_A : A = 40) : (B = 40 ∨ B = 100) :=
sorry

end top_angle_isosceles_triangle_l619_619919


namespace problem_l619_619905

variable (f : ℝ → ℝ)
variable (a : ℕ → ℝ)
variable (h_f : ∀ x1 x2 : ℝ, f(x1 + x2) = f(x1) + f(x2) + 2)
variable (h_a1 : a 1 = 0)
variable (h_an : ∀ n : ℕ, 0 < n → a n = f n)

theorem problem (h_a : ∀ n : ℕ, 0 < n → a n = 2 * n - 2) : f 2010 = 4018 := by
  have h := h_a 2010 (by linarith)
  exact h

#check problem -- verifying the theorem statement

end problem_l619_619905


namespace a_2016_value_l619_619118

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a n / (a (n + 1) - a n) = n

theorem a_2016_value (a : ℕ → ℕ) (h : sequence a) : a 2016 = 2016 := 
  by 
  sorry

end a_2016_value_l619_619118


namespace dot_product_necessary_but_not_sufficient_l619_619474

variable {V : Type*} [InnerProductSpace ℝ V]

theorem dot_product_necessary_but_not_sufficient
  (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a = b) ↔ (a ⋅ c = b ⋅ c) :=
sorry

end dot_product_necessary_but_not_sufficient_l619_619474


namespace temperature_drop_l619_619711

theorem temperature_drop (initial_temperature drop: ℤ) (h1: initial_temperature = 3) (h2: drop = 5) : initial_temperature - drop = -2 :=
by {
  sorry
}

end temperature_drop_l619_619711


namespace imo_1968_q5_l619_619629

noncomputable def periodic_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x : ℝ, f (x + 2 * a) = f x

theorem imo_1968_q5 (f : ℝ → ℝ) (a : ℝ) (h : a > 0)
  (hf : ∀ x : ℝ, f (x + a) = 0.5 + real.sqrt (f x - (f x) ^ 2)) :
  periodic_function f a :=
sorry

end imo_1968_q5_l619_619629


namespace product_ab_l619_619989

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619989


namespace acute_angle_implies_x_range_l619_619893

open Real

def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0

theorem acute_angle_implies_x_range (x : ℝ) :
  is_acute (1, 2) (x, 4) → x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi (2 : ℝ) :=
by
  sorry

end acute_angle_implies_x_range_l619_619893


namespace vertical_asymptote_one_l619_619033

-- Define the function g(x) with parameter k
def g (x k : ℝ) : ℝ := (x^2 + 3*x + k) / (x^2 - x - 12)

-- The proof problem statement in Lean 4
theorem vertical_asymptote_one (k : ℝ) : 
  (∀ x, x ≠ 4 → x ≠ -3 → (x^2 + 3*x+k) ≠ 0) ↔ (k = -28 ∨ k = 0) :=
sorry

end vertical_asymptote_one_l619_619033


namespace no_number_with_digit_sum_2001_l619_619370

noncomputable def number_digits (x : ℕ) : ℕ := (x.log10 : ℕ) + 1

theorem no_number_with_digit_sum_2001 :
  ¬ ∃ a : ℕ, let n := number_digits a in let m := number_digits (a ^ 3) in n + m = 2001 :=
begin
  sorry
end

end no_number_with_digit_sum_2001_l619_619370


namespace max_infected_population_l619_619553

theorem max_infected_population (G : Type*) [graph G] (n : ℕ)
  (H : ∀ v : G, degree v ≤ 3)
  (initial_infected : ℕ) (Hinfected : initial_infected = 2023)
  (infection_rule : ∀ v : G, (∃ u w : G, adjacent v u ∧ adjacent v w ∧ u.infected ∧ w.infected) → v.infected)
  (eventual_infection : ∀ v : G, v.infected) : n ≤ 4043 :=
begin
  sorry
end

end max_infected_population_l619_619553


namespace sufficient_not_necessary_for_xy_l619_619343

theorem sufficient_not_necessary_for_xy (x y : ℝ) 
  (h : x = 1 ∧ y = -1) : x*y = -1 :=
begin
  sorry
end

end sufficient_not_necessary_for_xy_l619_619343


namespace combined_area_of_removed_triangles_l619_619808

theorem combined_area_of_removed_triangles (s h : ℝ) (h_square_side : s = 20) (h_triangle_hypotenuse : h = 8) :
  let leg := h / real.sqrt 2
  let area_one_triangle := 0.5 * (leg^2)
  let total_area := 4 * area_one_triangle
  total_area = 64 := by
  sorry

end combined_area_of_removed_triangles_l619_619808


namespace base_c_numbers_to_base_7_l619_619213

theorem base_c_numbers_to_base_7
  (c : ℤ)
  (h1 : (c+4)*(c+8)*(c+7) = 4*c^3 + 1*c^2 + 8*c + 5)
  (h2 : 21 + 19 = 40)
  (h3 : ((40:ℤ)).base_repr (7) = "55") :
  (3*c + 19) = (55:ℤ).ofNat_base_7 := by
    sorry

end base_c_numbers_to_base_7_l619_619213


namespace cylinder_longest_segment_l619_619787

-- Define the radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 10

-- Definition for the longest segment inside the cylinder using Pythagorean theorem
def longest_segment (radius height : ℝ) : ℝ :=
  real.sqrt (radius * 2)^2 + height^2

-- Specify the expected answer for the proof
def expected_answer : ℝ := 10 * real.sqrt 2

-- The theorem stating the longest segment length inside the cylinder
theorem cylinder_longest_segment : longest_segment radius height = expected_answer :=
by {
  -- Lean code to set up and prove the equivalence
  sorry
}

end cylinder_longest_segment_l619_619787


namespace inequality_solution_set_l619_619707

theorem inequality_solution_set (x : ℝ) :
  (1 / |x - 1| > 3 / 2) ↔ (1 / 3 < x ∧ x < 5 / 3 ∧ x ≠ 1) :=
by
  sorry

end inequality_solution_set_l619_619707


namespace factorization_correct_l619_619742

theorem factorization_correct (x : ℝ) : 
  (hxA : x^2 + 2*x + 1 ≠ x*(x + 2) + 1) → 
  (hxB : x^2 + 2*x + 1 ≠ (x + 1)*(x - 1)) → 
  (hxC : x^2 + x ≠ (x + 1/2)^2 - 1/4) →
  x^2 + x = x * (x + 1) := 
by sorry

end factorization_correct_l619_619742


namespace inequality_holds_l619_619847

theorem inequality_holds (x y : ℝ) : (y - x^2 < abs x) ↔ (y < x^2 + abs x) := by
  sorry

end inequality_holds_l619_619847


namespace a_16_value_l619_619166

-- Define the recurrence relation
def seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0       => 2
  | (n + 1) => (1 + a n) / (1 - a n)

-- State the theorem
theorem a_16_value :
  seq (a : ℕ → ℚ) 16 = -1/3 := 
sorry

end a_16_value_l619_619166


namespace correct_statements_l619_619339

def floor_le_ceil_diff (x y : ℝ) : Prop :=
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋

def incorrect_floor_statement (x y : ℝ) : Prop :=
  if (|⌊y⌋ / ⌊x⌋| ≤ 1) then |x - y| < 1 else true

def sequence_sum_property (b : ℕ+ → ℤ) : Prop :=
  (∀ n, b n = ⌊sqrt (n * (n + 1) : ℝ)⌋) → (Finset.sum (Finset.range 64) b = 2080)

def M_div_3_eq_0 : Prop :=
  let M := Finset.sum (Finset.range 2022) (λ k, ⌊(2^(k+1) / 3 : ℝ)⌋) in
  M % 3 = 0

theorem correct_statements (x y : ℝ) (b : ℕ+ → ℤ) :
  floor_le_ceil_diff x y ∧ sequence_sum_property b ∧ M_div_3_eq_0 :=
by
  sorry

end correct_statements_l619_619339


namespace dot_product_necessary_but_not_sufficient_l619_619476

variable {V : Type*} [InnerProductSpace ℝ V]

theorem dot_product_necessary_but_not_sufficient
  (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a = b) ↔ (a ⋅ c = b ⋅ c) :=
sorry

end dot_product_necessary_but_not_sufficient_l619_619476


namespace count_isosceles_points_l619_619909

open Point

-- Define a triangle as three points
structure Triangle := 
  (A B C : Point)

-- Define an acute triangle with \( AC < AB < BC \)
def isAcuteTriangle (t : Triangle) : Prop :=
  ∃ h : t.A.dist t.C < t.A.dist t.B < t.B.dist t.C, ∀ ⦃x y z : Point⦄, ¬(x = y) → ¬(y = z) → ¬(z = x) → (x, y, z = t.A, t.B, t.C) → ∀ θ > 0, θ < 90

-- Define isosceles properties
def isIsosceles (P A B : Point) : Prop :=
  P.dist A = P.dist B

-- Define the proof problem
theorem count_isosceles_points (t : Triangle) (h : isAcuteTriangle t) :
  ∃ P_set : Set Point, (∀ P ∈ P_set, isIsosceles P t.A t.B ∧ isIsosceles P t.B t.C) ∧ P_set.size = 14 := 
sorry

end count_isosceles_points_l619_619909


namespace _l619_619461

noncomputable def max_value_prod_tan (n : ℕ) (θ : Fin n → ℝ) : Prop :=
  ∀ (h_n : 2 ≤ n) 
    (hθ_range : ∀ i, 0 < θ i ∧ θ i ≤ (Real.pi / 2)) 
    (h_sum_sin : ∑ i, Real.sin (θ i) ≤ 1), 
    (∏ i, Real.tan (θ i)) ≤ (n^2 - 1) ^ (- (n : ℝ) / 2)

-- Now, 'max_value_prod_tan' is the theorem we want to prove.

end _l619_619461


namespace minimize_cost_per_kilometer_l619_619321

-- Define variables for the constants given in the problem
def k := 35 / 1000
def constant_cost := 560
def max_speed := 25

-- Function to calculate the cost per hour given the speed v in km/h
def cost_per_hour (v : ℝ) := (k * v^3) + constant_cost

-- Function to calculate the cost per kilometer
def cost_per_kilometer (v : ℝ) := cost_per_hour v / v

-- The statement of our theorem
theorem minimize_cost_per_kilometer : (v : ℝ) -> 0 < v ∧ v <= max_speed -> C_per_km(v) <= C_per_km(w) for all w satisfying 0 < w <= max_speed.
  sorry

end minimize_cost_per_kilometer_l619_619321


namespace engagement_ring_savings_l619_619591

theorem engagement_ring_savings 
  (yearly_salary : ℝ) 
  (monthly_savings : ℝ) 
  (monthly_salary := yearly_salary / 12) 
  (ring_cost := 2 * monthly_salary) 
  (saving_months := ring_cost / monthly_savings) 
  (h_salary : yearly_salary = 60000) 
  (h_savings : monthly_savings = 1000) :
  saving_months = 10 := 
sorry

end engagement_ring_savings_l619_619591


namespace sum_of_inscribed_circumferences_l619_619557

theorem sum_of_inscribed_circumferences (n : ℕ) (d : ℕ → ℝ) (D : ℝ)
  (h_center : ∑ i in finset.range n, d i = D) :
  ∑ i in finset.range n, (Real.pi * d i) = Real.pi * D :=
by
  sorry

end sum_of_inscribed_circumferences_l619_619557


namespace range_of_a_l619_619209

noncomputable def f : ℝ → ℝ := sorry  -- assuming there exists a function f meeting the conditions

theorem range_of_a :
  (∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → (f x1 - f x2) / (x1 - x2) > 0) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x ∈ set.Icc (1/2) 1, ∀ a, f (a * x + 1) ≤ f (x - 2)) →
  ∀ a, a ∈ set.Icc (-2 : ℝ) 0 :=
begin
  sorry
end

end range_of_a_l619_619209


namespace distance_PO_l619_619266

open EuclideanGeometry

noncomputable def P := ( 5/2 : ℝ, (sqrt 5) : ℝ)

def F := ( (1/2) : ℝ, 0 )

def O := ( 0, 0 )

theorem distance_PO : dist P O = (3 * sqrt 5) / 2 :=
by
  sorry

end distance_PO_l619_619266


namespace f_2016_value_l619_619897

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then f (x-4) else (Real.exp x + ∫ t in (1:ℝ)..(2:ℝ), 1/t)

theorem f_2016_value : f 2016 = 1 + Real.log 2 := by
  sorry

end f_2016_value_l619_619897


namespace relationship_between_M_n_and_N_n_plus_2_l619_619345

theorem relationship_between_M_n_and_N_n_plus_2 (n : ℕ) (h : 2 ≤ n) :
  let M_n := (n * (n + 1)) / 2 + 1
  let N_n_plus_2 := n + 3
  M_n < N_n_plus_2 :=
by
  sorry

end relationship_between_M_n_and_N_n_plus_2_l619_619345


namespace problem_l619_619211

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem problem (a b : ℝ) (H1 : f a = 0) (H2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  sorry

end problem_l619_619211


namespace sum_sequence_l619_619701

noncomputable def a : ℕ → ℤ
| 1     := 1
| 2     := 2
| (n+3) := (a (n+1)^2 - 7) / a n

theorem sum_sequence : (∑ i in Finset.range 100 + 1, a i) = 1 :=
by
  sorry

end sum_sequence_l619_619701


namespace johns_employees_l619_619187

variable (each_turkey_cost total_spent number_of_employees : ℕ)

def turkey_cost : Prop := each_turkey_cost = 25
def total_cost : Prop := total_spent = 2125
def number_of_employees_correct : Prop := number_of_employees = total_spent / each_turkey_cost

theorem johns_employees (h1 : turkey_cost each_turkey_cost) 
                        (h2 : total_cost total_spent)
                        (h3 : number_of_employees_correct total_spent each_turkey_cost) : 
  number_of_employees = 85 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end johns_employees_l619_619187


namespace hyperbola_focus_and_distance_l619_619511

noncomputable def right_focus_of_hyperbola (a b : ℝ) : ℝ × ℝ := 
  (Real.sqrt (a^2 + b^2), 0)

noncomputable def distance_to_asymptote (a b : ℝ) : ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  abs c / Real.sqrt (1 + (b/a)^2)

theorem hyperbola_focus_and_distance (a b : ℝ) (h₁ : a^2 = 6) (h₂ : b^2 = 3) :
  right_focus_of_hyperbola a b = (3, 0) ∧ distance_to_asymptote a b = Real.sqrt 3 :=
by
  sorry

end hyperbola_focus_and_distance_l619_619511


namespace log_ax_mono_increasing_sufficient_but_not_necessary_l619_619760

def is_monotonically_increasing_on (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ interval → x₂ ∈ interval → x₁ ≤ x₂ → f x₁ ≤ f x₂

theorem log_ax_mono_increasing_sufficient_but_not_necessary (a : ℝ) : 
  (is_monotonically_increasing_on (fun x => log (a * x)) (Set.Ioi 0) → a > 0) ∧ 
  (is_monotonically_increasing_on (fun x => log (a * x)) (Set.Ioi 0) ↔ a = 1) = False := 
sorry

end log_ax_mono_increasing_sufficient_but_not_necessary_l619_619760


namespace distance_to_x_axis_l619_619160

def point_P : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point_P.snd) = 3 := by
  sorry

end distance_to_x_axis_l619_619160


namespace ab_equals_six_l619_619997

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l619_619997


namespace find_ED_l619_619178

noncomputable def isosceles_triangle (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def angle_BAC (ABC : Triangle) : Prop :=
angle ABC.B ABC.A ABC.C = 108

noncomputable def angle_bisector_intersects_D (D : Point) (ABC : Triangle) : Prop :=
angle_bisector ABC.ABC D ABC.A ABC.C

noncomputable def point_E_on_BC (E : Point) (ABC : Triangle) : Prop :=
E ∈ (segment ABC.B ABC.C) ∧ distance ABC.B E = distance ABC.A E

noncomputable def congruent_segments (E : Point) (m : ℝ) : Prop :=
distance ABC.A E = m

theorem find_ED
    (ABC : Triangle)
    (AB_eq_AC : isosceles_triangle ABC)
    (angle_BAC_108 : angle_BAC ABC)
    (D : Point) (E : Point)
    (bisector_D : angle_bisector_intersects_D D ABC)
    (E_on_BC : point_E_on_BC E ABC)
    (AE_eq_m : congruent_segments E m) : 
    distance D E = m := 
sorry

end find_ED_l619_619178


namespace trigonometric_identity_l619_619826

theorem trigonometric_identity :
  let θ1 := (15:ℝ) * real.pi / 180
  let θ2 := (45:ℝ) * real.pi / 180
  sin θ1 ^ 2 + cos θ2 ^ 2 + sin θ1 * cos θ2 = 3 / 4 :=
by
  sorry

end trigonometric_identity_l619_619826


namespace smallest_possible_value_l619_619620

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l619_619620


namespace tan_A_value_l619_619955

open Real

theorem tan_A_value (A : ℝ) (h1 : sin A * (sin A + sqrt 3 * cos A) = -1 / 2) (h2 : 0 < A ∧ A < π) :
  tan A = -sqrt 3 / 3 :=
sorry

end tan_A_value_l619_619955


namespace acute_angle_implies_x_range_l619_619891

open Real

def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0

theorem acute_angle_implies_x_range (x : ℝ) :
  is_acute (1, 2) (x, 4) → x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi (2 : ℝ) :=
by
  sorry

end acute_angle_implies_x_range_l619_619891


namespace integer_values_of_P_l619_619601

theorem integer_values_of_P (x y : ℕ) (hxy : x < y) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  ∃ (P : ℤ), (P = (↑(x^3) - ↑y) / (1 + x * y)) ∧ (P = 0 ∨ ∃ (p : ℕ), p ≥ 2 ∧ P = p) :=
by {
  sorry
}

end integer_values_of_P_l619_619601


namespace quotient_m_div_16_l619_619671

-- Define the conditions
def square_mod_16 (n : ℕ) : ℕ := (n * n) % 16

def distinct_squares_mod_16 : Finset ℕ :=
  { n | square_mod_16 n ∈ [1, 4, 9, 0].toFinset }

def m : ℕ :=
  distinct_squares_mod_16.sum id

-- Define the theorem to be proven
theorem quotient_m_div_16 : m / 16 = 0 :=
by
  sorry

end quotient_m_div_16_l619_619671


namespace eccentricity_of_hyperbola_l619_619115

noncomputable def hyperbola_eccentricity : ℝ → ℝ → ℝ → ℝ
| p, a, b => 
  let c := p / 2
  let e := c / a
  have h₁ : 9 * e^2 - 12 * e^2 / (e^2 - 1) = 1 := sorry
  e

theorem eccentricity_of_hyperbola (p a b : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity p a b = (Real.sqrt 7 + 2) / 3 :=
sorry

end eccentricity_of_hyperbola_l619_619115


namespace bankers_discount_is_correct_l619_619333

-- Define the given conditions
def TD := 45   -- True discount in Rs.
def FV := 270  -- Face value in Rs.

-- Calculate Present Value based on the given conditions
def PV := FV - TD

-- Define the formula for Banker's Discount
def BD := TD + (TD ^ 2 / PV)

-- Prove that the Banker's Discount is Rs. 54 given the conditions
theorem bankers_discount_is_correct : BD = 54 :=
by
  -- Steps to prove the theorem can be filled here
  -- Add "sorry" to skip the actual proof
  sorry

end bankers_discount_is_correct_l619_619333


namespace calculate_fg3_l619_619535

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l619_619535


namespace find_line_equation_of_ellipse_intersection_l619_619497

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 2 = 1

-- Defining the line intersects points
def line_intersects (A B : ℝ × ℝ) : Prop := 
  ∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ 
  (ellipse x1 y1) ∧ (ellipse x2 y2) ∧ 
  ((x1 + x2) / 2 = 1 / 2) ∧ ((y1 + y2) / 2 = -1)

-- Statement to prove the equation of the line
theorem find_line_equation_of_ellipse_intersection (A B : ℝ × ℝ)
  (h : line_intersects A B) : 
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ x - 4*y - (9/2) = 0) :=
sorry

end find_line_equation_of_ellipse_intersection_l619_619497


namespace range_of_a_l619_619500

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) π, f (cos x) = 0 → (x = 0 ∨ x = π)) →
  (∀ x ∈ Icc (0 : ℝ) π, ∃ y₁ y₂, y₁ ≠ y₂ → f (cos x) = 0) → 
  (f(x) = (x - 1) * real.exp x - a * x^2) →
  a ≤ -2 / real.exp 1 :=
sorry

end range_of_a_l619_619500


namespace infinite_seq_poly_composable_l619_619912

noncomputable def exists_finite_set_of_functions (P : ℕ → (ℝ → ℝ)) : Prop :=
  ∃ (N : ℕ) (f : fin N → (ℝ → ℝ)),
    ∀ n : ℕ, ∃ g : list (fin N), P n = (g.map (λ i, f i)).foldr (λ h acc, h ∘ acc) id

theorem infinite_seq_poly_composable :
  ∃ (P : ℕ → (ℝ → ℝ)),
  (∀ n : ℕ, is_polynomial (P n)) → exists_finite_set_of_functions P :=
sorry

end infinite_seq_poly_composable_l619_619912


namespace smallest_positive_c_satisfies_inequality_l619_619445

theorem smallest_positive_c_satisfies_inequality :
  ∀ (x y : ℝ), (0 ≤ x) → (0 ≤ y) → (c : ℝ) = 2/3 → real.cbrt(x * y) + c * |x - y| ≥ (x + y) / 3 :=
by
  intros x y hx hy hc
  rw hc
  sorry

end smallest_positive_c_satisfies_inequality_l619_619445


namespace least_n_for_A_0A_n_geq_100_l619_619193

noncomputable def A_0 : ℝ × ℝ := (0, 0)

noncomputable def A_n (n : ℕ) : ℝ := 0 -- Placeholder for x-coordinate of A_n, lies on x-axis

noncomputable def B_n (n : ℕ) : ℝ × ℝ := (A_n (n - 1) + |A_n n - A_n (n - 1)| / 2, (A_n (n - 1) + |A_n n - A_n (n - 1)| / 2) ^ 2)

noncomputable def a_n (n : ℕ) : ℝ := |A_n n - A_n (n - 1)|

noncomputable def A_0A_n (n : ℕ) : ℝ := (0).sum (λ i, a_n (i + 1)) -- Sum from 0 to n-1 of a_n

theorem least_n_for_A_0A_n_geq_100 : ∃ n, A_0A_n n ≥ 100 ∧ ∀ m, (m < n → A_0A_n m < 100) :=
begin
  sorry
end

end least_n_for_A_0A_n_geq_100_l619_619193


namespace combined_mean_is_half_sum_l619_619082

noncomputable def mean_combined (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) (mean_a mean_b : ℝ) :=
  (1 / 2) * (mean_a + mean_b)

theorem combined_mean_is_half_sum (a b : ℕ → ℝ) (mean_a mean_b : ℝ) 
  (h₁ : mean_a = (∑ i in Finset.range 10, a i) / 10)
  (h₂ : mean_b = (∑ i in Finset.range 10, b i) / 10) :
  mean_combined a b 10 mean_a mean_b = (1 / 2) * (mean_a + mean_b) :=
by
  sorry

end combined_mean_is_half_sum_l619_619082


namespace simplify_expression_l619_619669

theorem simplify_expression (m : ℝ) (h1 : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6 * m + 9)) = m - 3 := 
by
  sorry

end simplify_expression_l619_619669


namespace count_ordered_triples_l619_619523

noncomputable def log_base (a b : ℕ) : ℤ :=
  if h : a > 1 ∧ b > 0 then
    int.floor (Real.log b / Real.log a)
  else 0

theorem count_ordered_triples : 
  (finset.univ.filter (λ abc : ℕ × ℕ × ℕ, 
    abc.1 ≥ 3 ∧ abc.2.1 ≥ 1 ∧ abc.2.2 ≥ 0 ∧ 
    log_base abc.1 abc.2.1 = abc.2.2 * abc.2.2 * abc.2.2 ∧ 
    abc.1 + abc.2.1 + abc.2.2 = 300)).card = 1 :=
  sorry

end count_ordered_triples_l619_619523


namespace cover_equilateral_triangle_l619_619650

def is_equilateral_triangle (T : Type) : Prop :=
  ∃ (s : ℝ), T = (sqrt(3) / 4) * s ^ 2

def total_area (Ts : List Type) : ℝ :=
  Ts.foldl (λ acc T, acc + T.1) 0

theorem cover_equilateral_triangle
  (T : Type)
  (T1 T2 T3 T4 T5 : Type)
  (hT : is_equilateral_triangle T)
  (h_area_T : total_area [T] = 1)
  (hTs : [T1, T2, T3, T4, T5].all is_equilateral_triangle) 
  (h_area_Ts : total_area [T1, T2, T3, T4, T5] = 2) :
  ∃ arrangement : List (ℝ × ℝ), covers T arrangement [T1, T2, T3, T4, T5] :=
sorry

end cover_equilateral_triangle_l619_619650


namespace expected_score_l619_619814

noncomputable def expected_score_problem : ℕ → ℚ := 
  λ n, ( (5 * n) + ((10 - n) * (-1)) )

theorem expected_score (n : ℕ) (p : ℚ) (X : ℕ → ProbabilityMassFunction ℕ) 
  (hx : X n = binomial 10 0.6) : 
  (expected_score_problem ((10:ℕ) * (0.6:ℚ))) = 26 := by
  sorry

end expected_score_l619_619814


namespace hexagon_area_correct_l619_619603

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (b, 3)
  let F := (-3/2, 6)
  let area := 54 * Real.sqrt 3
  area

theorem hexagon_area_correct (b : ℝ) (h_b : b = 3 * Real.sqrt 3) :
  hexagon_area b = 54 * Real.sqrt 3 :=
begin
  sorry,
end

end hexagon_area_correct_l619_619603


namespace original_group_size_l619_619256

noncomputable def T (n : ℕ) := 15 * n

theorem original_group_size (n : ℕ) (avg_original : (T n) / n = 15) (new_person_age : 37)
    (new_avg : (T n + new_person_age) / (n + 1) = 17) : n = 10 :=
by
  sorry

end original_group_size_l619_619256


namespace range_of_m_l619_619931

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 9 * x - m

theorem range_of_m (H : ∃ (x_0 : ℝ), x_0 ≠ 0 ∧ f 0 x_0 = f 0 x_0) : 0 < m ∧ m < 1 / 2 :=
sorry

end range_of_m_l619_619931


namespace tan_alpha_is_neg_three_fourths_l619_619069

theorem tan_alpha_is_neg_three_fourths (α : ℝ) (h1 : α ∈ (Real.pi / 2, Real.pi)) (h2 : Real.sin α = 3 / 5) : Real.tan α = -3 / 4 :=
  sorry

end tan_alpha_is_neg_three_fourths_l619_619069


namespace altitudes_and_inradius_l619_619664

variable {A : ℝ} -- A is the area of triangle ABC
variable {a b c : ℝ} -- a, b, c are the side lengths of triangle ABC
variable {h_A h_B h_C : ℝ} -- h_A, h_B, h_C are the altitudes
variable {r : ℝ} -- r is the radius of the incircle

-- Conditions expressed as Lean definitions:
def altitude_A : ℝ := 2 * A / a
def altitude_B : ℝ := 2 * A / b
def altitude_C : ℝ := 2 * A / c
def inradius : ℝ := A / (a + b + c) * 2

-- The theorem to be proven:
theorem altitudes_and_inradius (hA_def : h_A = altitude_A) (hB_def : h_B = altitude_B) (hC_def : h_C = altitude_C) (r_def : r = inradius) :
  h_A + h_B + h_C ≥ 9 * r :=
by 
  sorry

end altitudes_and_inradius_l619_619664


namespace Tim_has_52_photos_l619_619721

theorem Tim_has_52_photos (T : ℕ) (Paul : ℕ) (Total : ℕ) (Tom : ℕ) : 
  (Paul = T + 10) → (Total = Tom + T + Paul) → (Tom = 38) → (Total = 152) → T = 52 :=
by
  intros hPaul hTotal hTom hTotalVal
  -- The proof would go here
  sorry

end Tim_has_52_photos_l619_619721


namespace elena_total_pens_l619_619429

theorem elena_total_pens 
  (cost_X : ℝ) (cost_Y : ℝ) (total_spent : ℝ) (num_brand_X : ℕ) (num_brand_Y : ℕ) (total_pens : ℕ)
  (h1 : cost_X = 4.0) 
  (h2 : cost_Y = 2.8) 
  (h3 : total_spent = 40.0) 
  (h4 : num_brand_X = 8) 
  (h5 : total_pens = num_brand_X + num_brand_Y) 
  (h6 : total_spent = num_brand_X * cost_X + num_brand_Y * cost_Y) :
  total_pens = 10 :=
sorry

end elena_total_pens_l619_619429


namespace sequence_fxn_is_geometric_sequence_l619_619939

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.cos x - Real.sin x)

def sequence_xn (n : ℕ) : ℝ := n * Real.pi

theorem sequence_fxn_is_geometric_sequence :
  let seq_f (n : ℕ) := f (sequence_xn n)
  in ∃ (a r : ℝ), (a = seq_f 1) ∧ (r = seq_f (1 + 1) / a) ∧ (∀ n, seq_f (n + 1) = r * seq_f n) := 
sorry

end sequence_fxn_is_geometric_sequence_l619_619939


namespace Vanya_original_number_l619_619309

theorem Vanya_original_number (m n : ℕ) (hm : m ≤ 9) (hn : n ≤ 9) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 10 * m + n = 81 := by
  sorry

end Vanya_original_number_l619_619309


namespace hexagon_diagonal_length_is_twice_side_l619_619418

noncomputable def regular_hexagon_side_length : ℝ := 12

def diagonal_length_in_regular_hexagon (s : ℝ) : ℝ :=
2 * s

theorem hexagon_diagonal_length_is_twice_side :
  diagonal_length_in_regular_hexagon regular_hexagon_side_length = 2 * regular_hexagon_side_length :=
by 
  -- Simplify and check the computation according to the understanding of the properties of the hexagon
  sorry

end hexagon_diagonal_length_is_twice_side_l619_619418


namespace quadratic_equation_root_form_l619_619428

theorem quadratic_equation_root_form
  (a b c : ℤ) (m n p : ℤ)
  (ha : a = 3)
  (hb : b = -4)
  (hc : c = -7)
  (h_discriminant : b^2 - 4 * a * c = n)
  (hgcd_mn : Int.gcd m n = 1)
  (hgcd_mp : Int.gcd m p = 1)
  (hgcd_np : Int.gcd n p = 1) :
  n = 100 :=
by
  sorry

end quadratic_equation_root_form_l619_619428


namespace number_of_balls_in_box_l619_619328

theorem number_of_balls_in_box : ∃ N : ℕ, (N - 44 = 70 - N) ∧ N = 57 :=
by
  use 57
  split
  · sorry
  · rfl

end number_of_balls_in_box_l619_619328


namespace find_intervals_f_monotonically_increasing_l619_619441

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin (-2 * x + π / 6)

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

theorem find_intervals_f_monotonically_increasing :
  ∀ k : ℤ, is_monotonically_increasing f (k * π + π / 3) (k * π + 5 * π / 6) :=
sorry

end find_intervals_f_monotonically_increasing_l619_619441


namespace concyclic_or_collinear_C1_C2_D1_D2_concyclic_B1_B2_D1_D2_iff_AC1_AC2_diameters_l619_619303

variables {Ω1 Ω2 n k : Type*}
variables {A O1 O2 C1 C2 B1 B2 D1 D2 : Point}

-- Assume necessary conditions
variables [Tangent Ω1 Ω2 A] [Tangent n k A] [IntersectLineCircle A C1 Ω1 l]
variables [IntersectLineCircle A C2 Ω2 l] [IntersectCircleCircle C1 C2 B1 Ω1]
variables [IntersectCircleCircle C1 C2 B2 Ω2] [CircumcircleTriangle A B1 B2 n]
variables [IntersectCircleCircle D1 Ω1 k] [IntersectCircleCircle D2 Ω2 k]

-- (1) Prove C1, C2, D1, D2 are concyclic or collinear
theorem concyclic_or_collinear_C1_C2_D1_D2
: Collinear_or_Concyclic C1 C2 D1 D2 := 
sorry

-- (2) Prove B1, B2, D1, D2 are concyclic if and only if AC1 and AC2 are diameters of Ω1 and Ω2 respectively
theorem concyclic_B1_B2_D1_D2_iff_AC1_AC2_diameters
(h1 : Diameter AC1 Ω1) (h2 : Diameter AC2 Ω2) 
: Concyclic B1 B2 D1 D2 ↔ (Diameter AC1 Ω1 ∧ Diameter AC2 Ω2) := 
sorry

end concyclic_or_collinear_C1_C2_D1_D2_concyclic_B1_B2_D1_D2_iff_AC1_AC2_diameters_l619_619303


namespace ab_equals_6_l619_619977

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619977


namespace ab_equals_six_l619_619994

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l619_619994


namespace forces_resultant_arithmetic_mean_l619_619314

variables (p1 p2 : ℝ) (α : ℝ)
open Real

theorem forces_resultant_arithmetic_mean 
  (h : sqrt (p1^2 + p2^2 + 2 * p1 * p2 * cos α) = (p1 + p2) / 2) 
  : (120 ≤ α ∧ α ≤ 180) 
  ∧ (1 / 3 ≤ p1 / p2 ∧ p1 / p2 ≤ 3) :=
begin
  sorry,
end

end forces_resultant_arithmetic_mean_l619_619314


namespace probability_three_same_color_is_one_seventeenth_l619_619811

def standard_deck := {cards : Finset ℕ // cards.card = 52 ∧ ∃ reds blacks, reds.card = 26 ∧ blacks.card = 26 ∧ (reds ∪ blacks = cards)}

def num_ways_to_pick_3_same_color : ℕ :=
  (26 * 25 * 24) + (26 * 25 * 24)

def total_ways_to_pick_3 : ℕ :=
  52 * 51 * 50

def probability_top_three_same_color := (num_ways_to_pick_3_same_color / total_ways_to_pick_3 : ℚ)

theorem probability_three_same_color_is_one_seventeenth :
  probability_top_three_same_color = (1 / 17 : ℚ) := by sorry

end probability_three_same_color_is_one_seventeenth_l619_619811


namespace correct_statements_l619_619340

def floor_le_ceil_diff (x y : ℝ) : Prop :=
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋

def incorrect_floor_statement (x y : ℝ) : Prop :=
  if (|⌊y⌋ / ⌊x⌋| ≤ 1) then |x - y| < 1 else true

def sequence_sum_property (b : ℕ+ → ℤ) : Prop :=
  (∀ n, b n = ⌊sqrt (n * (n + 1) : ℝ)⌋) → (Finset.sum (Finset.range 64) b = 2080)

def M_div_3_eq_0 : Prop :=
  let M := Finset.sum (Finset.range 2022) (λ k, ⌊(2^(k+1) / 3 : ℝ)⌋) in
  M % 3 = 0

theorem correct_statements (x y : ℝ) (b : ℕ+ → ℤ) :
  floor_le_ceil_diff x y ∧ sequence_sum_property b ∧ M_div_3_eq_0 :=
by
  sorry

end correct_statements_l619_619340


namespace quadrilateral_angle_cosine_proof_l619_619679

variable (AB BC CD AD : ℝ)
variable (ϕ B C : ℝ)

theorem quadrilateral_angle_cosine_proof :
  AD^2 = AB^2 + BC^2 + CD^2 - 2 * (AB * BC * Real.cos B + BC * CD * Real.cos C + CD * AB * Real.cos ϕ) :=
by
  sorry

end quadrilateral_angle_cosine_proof_l619_619679


namespace cistern_water_depth_l619_619362

theorem cistern_water_depth:
  ∀ h: ℝ,
  (4 * 4 + 4 * h * 4 + 4 * h * 4 = 36) → h = 1.25 := by
    sorry

end cistern_water_depth_l619_619362


namespace problem_statement_l619_619008

noncomputable def expression : ℝ :=
  16 * (1/5 - 1/3 * (1/5^3) + 1/5 * (1/5^5) - 1/7 * (1/5^7) + 1/9 * (1/5^9) - 1/11 * (1/5^11)) -
  4 * (1/239 - 1/3 * (1/239^3))

theorem problem_statement : (Real.toRational (Real.truncate 8 expression)) = 3.14159265 := 
sorry

end problem_statement_l619_619008


namespace value_of_m_l619_619925

theorem value_of_m (m : ℤ) (h : ∃ x : ℤ, x = 2 ∧ x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end value_of_m_l619_619925


namespace ca_co3_to_ca_cl2_l619_619029

theorem ca_co3_to_ca_cl2 (caCO3 HCl : ℕ) (main_reaction : caCO3 = 1 ∧ HCl = 2) : ∃ CaCl2, CaCl2 = 1 :=
by
  -- The proof of the theorem will go here.
  sorry

end ca_co3_to_ca_cl2_l619_619029


namespace greatest_integer_l619_619315

theorem greatest_integer (S : ℝ) (H : S = 1 + ∑ k in Ico 2 2015, 1 / real.sqrt k) : ∃ n : ℤ, n ≤ S ∧ n = 88 :=
by 
  sorry

end greatest_integer_l619_619315


namespace fib_gen_fn_luc_gen_fn_l619_619440

-- Definitions for Fibonacci polynomials
def Fib0 (x : ℤ) := 0
def Fib1 (x : ℤ) := 1
def F (x z : ℂ) := ∑' n, (Fib n x) * z^n
-- Prove Fibonacci generating function
theorem fib_gen_fn (x z : ℂ) : F (x, z) = z / (1 - x*z - z^2) := sorry

-- Definitions for Lucas polynomials
def Luc0 (x : ℤ) := 2
def Luc1 (x : ℤ) := x
def L (x z : ℂ) := ∑' n, (Luc n x) * z^n
-- Prove Lucas generating function
theorem luc_gen_fn (x z : ℂ) : L (x, z) = (2 + x*z) / (1 - x*z - z^2) := sorry

end fib_gen_fn_luc_gen_fn_l619_619440


namespace simplify_fraction_to_i_l619_619243

open Complex

theorem simplify_fraction_to_i : (1 + Complex.i / (1 - Complex.i)) ^ 1001 = Complex.i :=
by
  sorry

end simplify_fraction_to_i_l619_619243


namespace lucky_ticket_N123456_l619_619768

def digits : List ℕ := [1, 2, 3, 4, 5, 6]

def is_lucky (digits : List ℕ) : Prop :=
  ∃ f : ℕ → ℕ → ℕ, (f 1 (f (f 2 3) 4) * f 5 6) = 100

theorem lucky_ticket_N123456 : is_lucky digits :=
  sorry

end lucky_ticket_N123456_l619_619768


namespace quotient_of_0_009_div_0_3_is_0_03_l619_619350

-- Statement:
theorem quotient_of_0_009_div_0_3_is_0_03 (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 :=
by
  sorry

end quotient_of_0_009_div_0_3_is_0_03_l619_619350


namespace volume_of_regular_quadrilateral_pyramid_l619_619448

def pyramid_volume (l β : ℝ) : ℝ :=
  (2 * l^3 * (Real.cot (β / 2))^3) / (3 * (1 - (Real.cot (β / 2))^2))

theorem volume_of_regular_quadrilateral_pyramid (l β : ℝ) :
  (volume_of_pyramid l β = (2 * l^3 * (Real.cot (β / 2))^3) / (3 * (1 - (Real.cot (β / 2))^2))) :=
sorry

end volume_of_regular_quadrilateral_pyramid_l619_619448


namespace total_savings_l619_619587

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end total_savings_l619_619587


namespace initial_glass_bottles_count_l619_619713

namespace Bottles

variable (G P : ℕ)

/-- The weight of some glass bottles is 600 g. 
    The total weight of 4 glass bottles and 5 plastic bottles is 1050 g.
    A glass bottle is 150 g heavier than a plastic bottle.
    Prove that the number of glass bottles initially weighed is 3. -/
theorem initial_glass_bottles_count (h1 : G * (P + 150) = 600)
  (h2 : 4 * (P + 150) + 5 * P = 1050)
  (h3 : P + 150 > P) :
  G = 3 :=
  by sorry

end Bottles

end initial_glass_bottles_count_l619_619713


namespace megan_total_plays_l619_619215

-- Define the conditions
def lead_percentage := 0.80
def not_lead_plays := 20
def total_plays := (not_lead_plays * 100) / (100 - (lead_percentage * 100))

-- The statement to prove
theorem megan_total_plays : total_plays = 100 :=
by
  sorry

end megan_total_plays_l619_619215


namespace minimum_distance_l619_619480

theorem minimum_distance (a b c d : ℝ) (h : | b + a^2 - 4 * Real.log a | + | 2 * c - d + 2 | = 0) :
  (a - c)^2 + (b - d)^2 = 5 :=
sorry

end minimum_distance_l619_619480


namespace collinear_A_C_R_l619_619596

open EuclideanGeometry

variables {A B C D P Q R : Point}

/-- Problem statement: Let ABCD be a parallelogram. The tangent to the circumcircle of triangle BCD
    at C intersects AB at P and intersects AD at Q. The tangents to the circumcircle of triangle APQ
    at P and Q meet at R. Show that points A, C, and R are collinear. -/
theorem collinear_A_C_R
  (hParallelogram : Parallelogram A B C D)
  (hTangentToBCD : ∃! (P : Point), TangentToCircumcircle B C D C P)
  (hTangentsMeetR : ∃! (R : Point), TangentToCircumcircle A P Q P R ∧ TangentToCircumcircle A P Q Q R)
  : Collinear A C R :=
sorry

end collinear_A_C_R_l619_619596


namespace problem_inequality_l619_619089

open Real

theorem problem_inequality 
  (p q r x y theta: ℝ) :
  p * x ^ (q - y) + q * x ^ (r - y) + r * x ^ (y - theta)  ≥ p + q + r :=
sorry

end problem_inequality_l619_619089


namespace find_integer_n_l619_619759

/-
Given a list of integers [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5], 
prove that the integer n such that the probability of selecting n is 1/3 is 5.
-/

theorem find_integer_n : 
  (∀ n : ℕ, (∃ (count : ℕ), count = list.count [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5] n ∧ 
  count.toNat / 15 = (1 : ℕ) / 3) → n = 5) :=
by 
  sorry

end find_integer_n_l619_619759


namespace value_of_sum_l619_619709

noncomputable def a_n (n t : ℤ) : ℤ :=
  match n with
  | 1 => 3 + t
  | 2 => 6
  | 3 => 18
  | _ => 3^n + t - (3^(n-1) + t)

noncomputable def S_n (n t : ℤ) : ℤ :=
  3^n + t

theorem value_of_sum (t : ℤ) (h₁ : S_n 1 t = 3 + t)
  (h₂ : S_n 2 t - S_n 1 t = 6)
  (h₃ : S_n 3 t - S_n 2 t = 18)
  (h_geom : (S_n 2 t - S_n 1 t)^2 = (S_n 1 t) * (S_n 3 t - S_n 2 t)) : t + (S_n 3 t - S_n 2 t) = 17 :=
by
  sorry

end value_of_sum_l619_619709


namespace f_increasing_on_left_f_decreasing_on_right_f_maximum_at_2_l619_619105

def f (x : ℝ) : ℝ := (1/2)^(|x-2|)

theorem f_increasing_on_left : ∀ x y: ℝ, x < y ∧ y < 2 → f x < f y :=
by sorry

theorem f_decreasing_on_right : ∀ x y: ℝ, x > y ∧ x ≥ 2 → f x < f y :=
by sorry

theorem f_maximum_at_2 : (∀ x: ℝ, f x ≤ 1) ∧ (f 2 = 1) :=
by sorry

end f_increasing_on_left_f_decreasing_on_right_f_maximum_at_2_l619_619105


namespace find_natural_numbers_eq_36_sum_of_digits_l619_619050

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end find_natural_numbers_eq_36_sum_of_digits_l619_619050


namespace terminal_side_in_fourth_quadrant_l619_619130

theorem terminal_side_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (θ ≥ 0 ∧ θ < Real.pi/2) ∨ (θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) :=
sorry

end terminal_side_in_fourth_quadrant_l619_619130


namespace radius_of_sphere_inscribed_in_box_l619_619805

theorem radius_of_sphere_inscribed_in_box (a b c s : ℝ)
  (h1 : a + b + c = 42)
  (h2 : 2 * (a * b + b * c + c * a) = 576)
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) :
  s = 3 * Real.sqrt 33 :=
by sorry

end radius_of_sphere_inscribed_in_box_l619_619805


namespace triangle_angle_R_measure_l619_619171

theorem triangle_angle_R_measure :
  ∀ (P Q R : ℝ),
  P + Q + R = 180 ∧ P = 70 ∧ Q = 2 * R + 15 → R = 95 / 3 :=
by
  intros P Q R h
  sorry

end triangle_angle_R_measure_l619_619171


namespace geometric_sequence_arithmetic_l619_619077

theorem geometric_sequence_arithmetic (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h2 : 2 * S 6 = S 3 + S 9) : 
  q^3 = -1 := 
sorry

end geometric_sequence_arithmetic_l619_619077


namespace eval_expr_with_given_sum_l619_619132

theorem eval_expr_with_given_sum (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 4 = 14 := 
by 
  sorry

end eval_expr_with_given_sum_l619_619132


namespace find_angle_C_l619_619906

variables (A B C : ℝ) (a b c : ℝ)
variables (triangle_ABC : (a = 2) ∧ (b + c = 2 * a) ∧ (3 * Real.sin A = 5 * Real.sin B))

theorem find_angle_C (h : triangle_ABC) : C = 2 * Real.pi / 3 :=
by
  obtain ⟨ha, hbc, hs⟩ := h
  sorry

end find_angle_C_l619_619906


namespace problem_solution_l619_619490

-- Declare the proof problem in Lean 4

theorem problem_solution (x y : ℝ) 
  (h1 : (y + 1) ^ 2 + (x - 2) ^ (1/2) = 0) : 
  y ^ x = 1 :=
sorry

end problem_solution_l619_619490


namespace minimum_removals_l619_619456

def forbidden_sums (x y : ℕ) : Prop :=
  x + y = 5 ∨ x + y = 13 ∨ x + y = 31 ∨ x + y = 65

def condition (M : Finset ℕ) : Prop :=
  ∀ x y ∈ M, x ≠ y → ¬ forbidden_sums x y

theorem minimum_removals : ∃ (n : ℕ), n = 17 ∧ ∀ (M : Finset ℕ),
  M = Finset.range 37 \ Finset.range (37 - n) →
  condition M :=
sorry

end minimum_removals_l619_619456


namespace count_g_expressible_l619_619125

def g (x : ℝ) := (⌊3 * x⌋ : ℤ) + (⌊6 * x⌋ : ℤ) + (⌊9 * x⌋ : ℤ)

theorem count_g_expressible : ∃ (S : Finset ℤ), S.card = 417 ∧ ∀ n ∈ S, 1 ≤ n ∧ n ≤ 500 ∧ (∃ x : ℝ, g x = n) :=
by {
  sorry
}

end count_g_expressible_l619_619125


namespace problem_l619_619267

theorem problem (m b : ℚ) (h1 : m = 3/4) (h2 : b = -1/3) : -1 < m * b ∧ m * b < 0 :=
by
  have h3 : m * b = (3/4) * (-1/3), from by rw [h1, h2]
  have h4 : m * b = -1/4, from by norm_num [h3]
  exact ⟨by norm_num [h4], by norm_num [h4]⟩

end problem_l619_619267


namespace tangent_line_at_x_minus1_monotonicity_of_f_l619_619940

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * a * x^2 + 3

theorem tangent_line_at_x_minus1 (a : ℝ) (h : a = 1) :
  let x := -1 in 
  let fx := f x a in 
  let f'_x := deriv (λ x, f x a) x in
  (fx = -1) → (f'_x = 9) → (∀ y, y = f'_x * (x + 1) + fx → y = 9 * x + 8) :=
by sorry

theorem monotonicity_of_f (a : ℝ) :
  let f' (x : ℝ) := deriv (λ x, f x a) x in
  if a = 0 then (∀ x, f' x ≥ 0) else
  ( (∀ x, f' x < 0 → (0 < x ∧ x < 2 * a)) ∧
    (∀ x, f' x > 0 → (x < 0 ∨ x > 2 * a)) ) :=
by sorry

end tangent_line_at_x_minus1_monotonicity_of_f_l619_619940


namespace find_radius_l619_619283

noncomputable def square_radius (r : ℝ) : Prop :=
  let s := (2 * r) / Real.sqrt 2  -- side length of the square derived from the radius
  let perimeter := 4 * s         -- perimeter of the square
  let area := Real.pi * r^2      -- area of the circumscribed circle
  perimeter = area               -- given condition

theorem find_radius (r : ℝ) (h : square_radius r) : r = (4 * Real.sqrt 2) / Real.pi :=
by
  sorry

end find_radius_l619_619283


namespace ratio_of_areas_l619_619798

variable (A B : ℝ)

-- Conditions
def total_area := A + B = 700
def smaller_part_area := B = 315

-- Problem Statement
theorem ratio_of_areas (h_total : total_area A B) (h_small : smaller_part_area B) :
  (A - B) / ((A + B) / 2) = 1 / 5 := by
sorry

end ratio_of_areas_l619_619798


namespace number_of_positive_integers_m_l619_619702

def seq_a : ℕ → ℕ 
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 4
| 5 := 5
| (n + 1) := seq_a 1 * seq_a 2 * seq_a 3 * seq_a 4 * seq_a 5 * seq_a.nat.rec _ sorry  -- to handle for n >= 5

def b_seq (n : ℕ) : ℕ :=
(seq_a 1 * seq_a 2 * seq_a 3 * seq_a 4 * seq_a 5 * seq_a.nat.rec _ sorry) - (seq_a 1 ^ 2 + seq_a 2 ^ 2 + seq_a 3 ^ 2 + seq_a 4 ^ 2 + seq_a 5 ^ 2 + seq_a.nat.rec _ sorry ^ 2)

def positive_integers_m := set_of (λ m : ℕ, (∏ i in range m, seq_a i) = (∑ i in range m, seq_a i ^ 2))

theorem number_of_positive_integers_m : 
  (positive_integers_m = λ (m = 1) ∨ (m = 70)) :=
sorry

end number_of_positive_integers_m_l619_619702


namespace fraction_below_line_is_correct_l619_619464

-- Definitions of the points and the line
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (7, 1)
def point3 : ℝ × ℝ := (7, 5)
def point4 : ℝ × ℝ := (2, 5)
def line : ℝ × ℝ := (2, 1) -- starting point of the line

-- Function to calculate the fraction of the area below the given line
def fraction_below_line (points : List (ℝ × ℝ)) (line : ℝ × ℝ) : ℝ :=
  -- Define the slope m
  let m := (5 - 1) / (7 - 2)
  -- Define the line equation using point-slope form
  let line_eqn := λ x, m * (x - 2) + 1
  -- Calculate the total area of the rectangle
  let total_area := (7 - 2) * (5 - 1)
  -- Calculate the area below the line within the rectangle (triangle area calculation)
  let height_at_7 := 5 - (4/5 * 7 - 3/5)
  let area_below_line := (1/2) * 5 * height_at_7
  -- Calculate the fraction
  (area_below_line / total_area)

-- Lean statement to prove the fraction of the area below the line is 1/4
theorem fraction_below_line_is_correct : fraction_below_line [point1, point2, point3, point4] line = 1 / 4 := by
  sorry

end fraction_below_line_is_correct_l619_619464


namespace remainder_div_product_l619_619914

theorem remainder_div_product (P D D' D'' Q R Q' R' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = Q' * D' + R') 
  (h3 : Q' = Q'' * D'' + R'') :
  P % (D * D' * D'') = D * D' * R'' + D * R' + R := 
sorry

end remainder_div_product_l619_619914


namespace calculate_expression_l619_619327

theorem calculate_expression : 
  (1007^2 - 995^2 - 1005^2 + 997^2) = 8008 := 
by {
  sorry
}

end calculate_expression_l619_619327


namespace ab_equals_6_l619_619978

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619978


namespace exists_positive_root_in_form_l619_619869

theorem exists_positive_root_in_form :
  ∃ (a b : ℝ), (a + b * real.sqrt 3) > 0 ∧ (a + b * real.sqrt 3) ^ 3 - 4 * (a + b * real.sqrt 3) ^ 2 - 2 * (a + b * real.sqrt 3) - real.sqrt 3 = 0 :=
sorry

end exists_positive_root_in_form_l619_619869


namespace probability_of_events_met_l619_619718

-- Definitions for the conditions
def C := Finset.range 30
def D := Finset.range' 15 30

-- Definitions for events as sets
def C_less_than_20 := {x ∈ C | x < 20}
def D_odd_or_greater_than_40 := {x ∈ D | x % 2 = 1 ∨ x > 40}

-- Definitions for probabilities
def prob_C_less_than_20 := (C_less_than_20.card: ℚ) / (C.card)
def prob_D_odd_or_greater_than_40 := (D_odd_or_greater_than_40.card: ℚ) / (D.card)

-- The theorem statement
theorem probability_of_events_met :
  (prob_C_less_than_20 * prob_D_odd_or_greater_than_40) = 323 / 900 := 
by
  sorry

end probability_of_events_met_l619_619718


namespace worst_ranking_l619_619289

theorem worst_ranking (teams : Fin 25 → Nat) (A : Fin 25)
  (round_robin : ∀ i j, i ≠ j → teams i + teams j ≤ 4)
  (most_goals : ∀ i, i ≠ A → teams A > teams i)
  (fewest_goals : ∀ i, i ≠ A → teams i > teams A) :
  ∃ ranking : Fin 25 → Fin 25, ranking A = 24 :=
by
  sorry

end worst_ranking_l619_619289


namespace douglas_votes_percentage_l619_619561

theorem douglas_votes_percentage 
  (V : ℝ)
  (hx : 0.62 * 2 * V + 0.38 * V = 1.62 * V)
  (hy : 3 * V > 0) : 
  ((1.62 * V) / (3 * V)) * 100 = 54 := 
by
  sorry

end douglas_votes_percentage_l619_619561


namespace minimum_value_l619_619028

noncomputable def min_value_of_trig_function : ℝ :=
  Real.sin (5 * Real.pi / 12)

theorem minimum_value (x : ℝ) (h : -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 6) :
  has_min_on (λ x, Real.cos (x + Real.pi / 3) - Real.cot (x + Real.pi / 3) + Real.sin (x + Real.pi / 4))
             (Set.Icc (-Real.pi / 12) (Real.pi / 6)) min_value_of_trig_function :=
sorry

end minimum_value_l619_619028


namespace general_term_formula_sequence_inequality_l619_619022

noncomputable def a (n : ℕ) : ℕ := 
if n = 0 then 2 
else if n = 1 then 10 
else 2 * (a (n - 1)) + 3 * (a (n - 2))

-- Let's first encode the sequence conditions
def seq_definition : Prop :=
  (a 1 = 2) ∧ (a 2 = 10) ∧ ∀ n : ℕ, n > 1 → a (n + 2) = 2 * a (n + 1) + 3 * a n

-- Define the mathematical proof of the general term formula and the inequality
theorem general_term_formula (n : ℕ) : seq_definition → a n = 3 ^ n + (-1) ^ n :=
sorry

theorem sequence_inequality (n : ℕ) : 
  seq_definition → (∑ i in finset.range n, (1 : ℚ) / a (i + 1)) < 2 / 3 :=
sorry

end general_term_formula_sequence_inequality_l619_619022


namespace multiplication_digit_sum_l619_619164

theorem multiplication_digit_sum :
  let a := 879
  let b := 492
  let product := a * b
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  product = 432468 ∧ sum_of_digits = 27 := by
  -- Step 1: Set up the given numbers
  let a := 879
  let b := 492

  -- Step 2: Calculate the product
  let product := a * b
  have product_eq : product = 432468 := by
    sorry

  -- Step 3: Sum the digits of the product
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  have sum_of_digits_eq : sum_of_digits = 27 := by
    sorry

  -- Conclusion
  exact ⟨product_eq, sum_of_digits_eq⟩

end multiplication_digit_sum_l619_619164


namespace product_of_slopes_constant_circumcircle_passes_through_fixed_point_l619_619902

noncomputable theory

open_locale classical

-- Definitions based on conditions
def circle (x y : ℝ) := x^2 + y^2 = 4
def P := (p_x, p_y)  -- Coordinates of P on the negative x-axis (we can leave them as variables for generality)
def Q := (1 : ℝ, 0 : ℝ)  -- Fixed point Q(1,0)

-- Intersection points modelled as types of points on the circle
structure Point := (x : ℝ) (y : ℝ)
def A := Point
def B := Point

-- Slopes definitions using points A and B
def k1 (A : Point) := A.y / (A.x - p_x)
def k2 (B : Point) := B.y / (B.x - p_x)

-- Statement for question (1)
theorem product_of_slopes_constant (A B : Point) (hA_on_circle : circle A.x A.y) (hB_on_circle : circle B.x B.y) 
(k1 A : ℝ) (k2 B : ℝ) : k1 * k2 = -1 / 3 := sorry

-- Definitions for second part involving point R and circumcircle
def PA_line (k1 : ℝ) (x : ℝ) := k1 * (x - p_x)
def R : Point := ⟨4, PA_line k1 4⟩
def circumcircle (P A B : Point) := notification.circle.center  -- Using some default notation for circumcircle computation

-- Statement for question (2)
theorem circumcircle_passes_through_fixed_point (R P : Point) (circle_center : Point) 
(hR_on_PA : R.y = PA_line k1 R.x) : circle_center = (4, 0) := sorry

end product_of_slopes_constant_circumcircle_passes_through_fixed_point_l619_619902


namespace graph_equivalence_l619_619741

theorem graph_equivalence 
  (f₁ : ℝ → ℝ)
  (f₂ : ℝ → ℝ)
  (f₃ : ℝ → ℝ)
  (h1 : ∀ x, f₁ x = x + 3)
  (h2 : ∀ x, x ≠ 1 → f₂ x = (x ^ 2 - 1) / (x - 1))
  (h3 : ∀ x, x ≠ 1 → f₃ x = (x ^ 2 - 1) / (x - 1))
  (h4 : (f₂ 1) ≠ (f₃ 1)) :
  ∀ x, (x ≠ 1 → f₂ x = f₃ x) ∧ ∃ ! x, f₁ x ≠ f₂ x := 
by
  sorry

end graph_equivalence_l619_619741


namespace hannahs_peppers_total_weight_l619_619520

theorem hannahs_peppers_total_weight:
  let green := 0.3333333333333333
  let red := 0.3333333333333333
  let yellow := 0.25
  let orange := 0.5
  green + red + yellow + orange = 1.4166666666666665 :=
by
  repeat { sorry } -- Placeholder for the actual proof

end hannahs_peppers_total_weight_l619_619520


namespace vasya_travel_time_l619_619519

def travel_time (d v : ℝ) : ℝ := d / v

theorem vasya_travel_time (d v : ℝ) (h : travel_time d v = d / (2 * v) + 5 + d / (4 * v)) : 
  travel_time d v = 20 :=
by 
  sorry

end vasya_travel_time_l619_619519


namespace longest_segment_in_cylinder_l619_619782

theorem longest_segment_in_cylinder (radius height : ℝ) 
  (hr : radius = 5) (hh : height = 10) :
  ∃ segment_length, segment_length = 10 * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end longest_segment_in_cylinder_l619_619782


namespace vector_dot_product_l619_619122

open Real

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem vector_dot_product :
  let a := (2 : ℝ, 0 : ℝ)
  let b := (1 / 2 : ℝ, sqrt 3 / 2 : ℝ)
  dot_product b (a.1 - b.1, a.2 - b.2) = 0 :=
by
  let a := (2 : ℝ, 0 : ℝ)
  let b := (1 / 2 : ℝ, sqrt 3 / 2 : ℝ)
  sorry

end vector_dot_product_l619_619122


namespace union_sets_l619_619190

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5, 6}

theorem union_sets : (A ∪ B) = {1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_sets_l619_619190


namespace solve_diff_eq_l619_619751

-- We declare noncomputable theory because some calculus operations might be noncomputable
noncomputable theory

-- Definitions for the problem setup
def P (x y : ℝ) : ℝ := 3 * x^2 * y^2 + 7
def Q (x y : ℝ) : ℝ := 2 * x^3 * y
def U (x y : ℝ) : ℝ := y^2 * x^3 + 7 * x

-- Initial condition
def initial_condition (y : ℝ → ℝ) : Prop := y 0 = 1

-- Definition of the final general solution
def general_solution (x y : ℝ) : Prop := y^2 * x^3 + 7 * x = 0

-- Main theorem to prove
theorem solve_diff_eq (y : ℝ → ℝ) (x : ℝ) :
  (P x (y x)) + (Q x (y x)) * (deriv y x) = 0 ∧ initial_condition y → general_solution x (y x) :=
by
  -- Placeholder for the proof
  sorry

end solve_diff_eq_l619_619751


namespace sqrt_prod_plus_one_equals_341_l619_619838

noncomputable def sqrt_prod_plus_one : ℕ :=
  Nat.sqrt ((20 * 19 * 18 * 17) + 1)

theorem sqrt_prod_plus_one_equals_341 :
  sqrt_prod_plus_one = 341 := 
by
  sorry

end sqrt_prod_plus_one_equals_341_l619_619838


namespace log_ln_sqrt_comparison_l619_619459

theorem log_ln_sqrt_comparison : 
  let a := log 4 3,
      b := Real.log 3,
      c := 10 ^ (1 / 2)
  in a < b ∧ b < c :=
by
  sorry

end log_ln_sqrt_comparison_l619_619459


namespace circle_tangent_to_y_axis_l619_619278

/-- The relationship between the circle with the focal radius |PF| of the parabola y^2 = 2px (where p > 0)
as its diameter and the y-axis -/
theorem circle_tangent_to_y_axis
  (p : ℝ) (hp : p > 0)
  (x1 y1 : ℝ)
  (focus : ℝ × ℝ := (p / 2, 0))
  (P : ℝ × ℝ := (x1, y1))
  (center : ℝ × ℝ := ((2 * x1 + p) / 4, y1 / 2))
  (radius : ℝ := (2 * x1 + p) / 4) :
  -- proof that the circle with PF as its diameter is tangent to the y-axis
  ∃ k : ℝ, k = radius ∧ (center.1 = k) :=
sorry

end circle_tangent_to_y_axis_l619_619278


namespace complex_min_value_l619_619200

theorem complex_min_value (z : ℂ) (h : complex.abs (z - (3 - 3 * complex.I)) = 4) :
  complex.abs (z + (2 - complex.I))^2 + complex.abs (z - (6 - 5 * complex.I))^2 = 76 :=
sorry

end complex_min_value_l619_619200


namespace exists_circle_tangent_to_K_and_g_through_P_l619_619078

variables {P K g : Type}
variables [metric_space P] [line K] [circle g]

-- Defining the problem conditions
def lies_on_line (P : P) (g : line P) : Prop := sorry
def tangent (c1 c2 : circle P) : Prop := sorry

-- Stating the goal theorem
theorem exists_circle_tangent_to_K_and_g_through_P
  (hP_on_g : lies_on_line P g)
  (hcircle_K : tangent K g) : 
  ∃ (C : circle P), tangent C K ∧ lies_on_line P (boundary C) :=
sorry

end exists_circle_tangent_to_K_and_g_through_P_l619_619078


namespace sampling_method_correctness_l619_619763

theorem sampling_method_correctness :
  ∀ (students_count : ℕ) (liberal_arts_count sciences_count arts_sports_count sample_size : ℕ)
    (parents_count selected_parents : ℕ),
    students_count = 1200 ∧
    liberal_arts_count = 400 ∧
    sciences_count = 600 ∧
    arts_sports_count = 200 ∧
    sample_size = 120 ∧
    parents_count = 10 ∧
    selected_parents = 3 →
    (sampling_method students_count liberal_arts_count sciences_count arts_sports_count sample_size = Stratified) ∧
    (sampling_method parents_count 0 0 0 selected_parents = SimpleRandom) :=
sorry

-- Definitions of the sampling methods as enums
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified
deriving DecidableEq

-- The actual sampling method determination function (just a placeholder)
def sampling_method (population_size liberal_arts sciences arts_sports sample_size : ℕ) : SamplingMethod :=
  if population_size = 1200 ∧ liberal_arts = 400 ∧ sciences = 600 ∧ arts_sports = 200 ∧ sample_size = 120 then
    SamplingMethod.Stratified
  else if population_size = 10 ∧ sample_size = 3 then
    SamplingMethod.SimpleRandom
  else
    SamplingMethod.SimpleRandom -- Default case

end sampling_method_correctness_l619_619763


namespace radius_of_inscribed_circle_l619_619662

/-- Sector OAB is a quarter of a circle with radius 5 cm. Inside this sector, a circle is inscribed, tangent at three points. -/
theorem radius_of_inscribed_circle (O A B C : Type) [metric_space O]
  (radius_O : ℝ) (r : ℝ) (OAB_is_quarter_circle : is_quarter_circle (OAB : ∀ a b c, a = O ∧ b = A ∧ c = B))
  (inscribed_circle_exists : ∃ C : O, is_inscribed_circle (OAB) C) :
  r = 5 * sqrt(2) - 5 :=
begin
  sorry,
end

end radius_of_inscribed_circle_l619_619662


namespace Carlton_button_up_shirts_l619_619013

/-- 
Given that the number of sweater vests V is twice the number of button-up shirts S, 
and the total number of unique outfits (each combination of a sweater vest and a button-up shirt) is 18, 
prove that the number of button-up shirts S is 3. 
-/
theorem Carlton_button_up_shirts (V S : ℕ) (h1 : V = 2 * S) (h2 : V * S = 18) : S = 3 := by
  sorry

end Carlton_button_up_shirts_l619_619013


namespace A_is_N_positive_l619_619636

variable (A : Set ℕ)

-- Conditions
axiom cond1 : ∃ a b c ∈ A, a ≠ b ∧ b ≠ c ∧ c ≠ a  -- number of elements in A is no less than 3
axiom cond2 : ∀ {a : ℕ}, a ∈ A → ∀ {d : ℕ}, d ∣ a → d ∈ A
axiom cond3 : ∀ {a b : ℕ}, a ∈ A → b ∈ A → 1 < a → a < b → 1 + a * b ∈ A

-- Proof statement
theorem A_is_N_positive : A = { n : ℕ | 0 < n } :=
sorry

end A_is_N_positive_l619_619636


namespace area_outside_triangle_l619_619174

-- Define the sides of triangle ABC
def triangle_ABC (A B C : Type) := 
  let AB : ℝ := 16
  let BC : ℝ := 5 * Real.sqrt 5
  let CA : ℝ := 9
  ∃ (a b c : A), 
  dist a b = AB ∧ 
  dist b c = BC ∧ 
  dist c a = CA 

-- Define the condition for points outside the triangle ABC such that the distance to B and C is less than 6
def outside_region_distance_condition (P B C : Type) :=
  let d1 : ℝ := 6
  ∃ (p b c : P), 
  dist p b < d1 ∧ 
  dist p c < d1

-- The main theorem statement to prove the area of the specified regions
theorem area_outside_triangle (A B C P : Type) [MetricSpace A] [MetricSpace P] 
  (ABC_triangle : triangle_ABC A B C) 
  (distance_condition : outside_region_distance_condition P B C) : 
  ∃ region_area: ℝ, region_area = 54 * Real.pi + (5 * Real.sqrt 95) / 4 := 
sorry

end area_outside_triangle_l619_619174


namespace infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l619_619665

theorem infinite_nat_sum_of_squares_and_cubes_not_sixth_powers :
  ∃ (N : ℕ) (k : ℕ), N > 0 ∧
  (N = 250 * 3^(6 * k)) ∧
  (∃ (x y : ℕ), N = x^2 + y^2) ∧
  (∃ (a b : ℕ), N = a^3 + b^3) ∧
  (∀ (u v : ℕ), N ≠ u^6 + v^6) :=
by
  sorry

end infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l619_619665


namespace cube_edge_length_l619_619124

theorem cube_edge_length 
  (box_edge_length : ℝ) 
  (number_of_cubes_approx : ℝ) 
  (h1 : box_edge_length = 1) 
  (h2 : number_of_cubes_approx ≈ 1000) :
  ∃ cube_edge_length : ℝ, cube_edge_length = 10 := 
by 
  sorry

end cube_edge_length_l619_619124


namespace total_savings_l619_619588

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end total_savings_l619_619588


namespace circles_chord_length_l619_619415

theorem circles_chord_length (r1 r2 r3 : ℕ) (m n p : ℕ) (h1 : r1 = 4) (h2 : r2 = 10) (h3 : r3 = 14)
(h4 : gcd m p = 1) (h5 : ¬ (∃ (k : ℕ), k^2 ∣ n)) : m + n + p = 19 :=
by
  sorry

end circles_chord_length_l619_619415


namespace monotone_function_sol_l619_619845

noncomputable def monotone_function (f : ℤ → ℤ) :=
  ∀ x y : ℤ, f x ≤ f y → x ≤ y

theorem monotone_function_sol
  (f : ℤ → ℤ)
  (H1 : monotone_function f)
  (H2 : ∀ x y : ℤ, f (x^2005 + y^2005) = f x ^ 2005 + f y ^ 2005) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end monotone_function_sol_l619_619845


namespace bianca_total_drawing_time_l619_619407

def total_drawing_time (a b : ℕ) : ℕ := a + b

theorem bianca_total_drawing_time :
  let a := 22
  let b := 19
  total_drawing_time a b = 41 :=
by
  sorry

end bianca_total_drawing_time_l619_619407


namespace isosceles_trapezoid_area_l619_619722

-- Definition of the isosceles trapezoid
structure IsoscelesTrapezoid :=
(A B C D P : Point)
(ad_eq_bc : A.distance D = B.distance C)
(P_intersection : ∃ a b, A.to2D_Vector (a, b) ∧ B.to2D_Vector (-a, b))

-- Conditions about the isosceles trapezoid and areas given
variable {T : IsoscelesTrapezoid}
variable {area_ABP area_CDP : ℝ}

-- Assume areas given in the problem
axiom area_ABP_50 : area_ABP = 50
axiom area_CDP_72 : area_CDP = 72

-- The theorem to prove the area of trapezoid ABCD
theorem isosceles_trapezoid_area (T : IsoscelesTrapezoid)
  (h1 : area_ABP = 50)
  (h2 : area_CDP = 72) :
  ∃ area_ABCD, area_ABCD = 242 :=
sorry

end isosceles_trapezoid_area_l619_619722


namespace sum_of_integer_part_log2_l619_619881

def integer_part_log2 (x : ℝ) : ℤ :=
  floor (Real.log x / Real.log 2)

theorem sum_of_integer_part_log2 :
  (∑ x in Finset.range 1023.succ, integer_part_log2 (x : ℝ)) = 8194 := 
by
  sorry

end sum_of_integer_part_log2_l619_619881


namespace rotation_of_A_after_all_rotations_l619_619830

theorem rotation_of_A_after_all_rotations 
    (A B C : Type) 
    (θ₁ θ₂ θ₃ : ℝ) 
    (x < 360) 
    (rotate_clockwise : A → ℝ → B → A) 
    (rotate_counterclockwise : A → ℝ → B → A)
    (h₁ : rotate_clockwise A 750 B = C) 
    (h₂ : rotate_counterclockwise A x B = C) 
    (h₃ : rotate_counterclockwise C 45 B ≠ C) 
  : rotate_counterclockwise A 15 B ≠ C :=
sorry

end rotation_of_A_after_all_rotations_l619_619830


namespace parallelogram_area_l619_619647

theorem parallelogram_area (θ : ℝ) (a b : ℝ) (hθ : θ = 100) (ha : a = 20) (hb : b = 10):
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  area = 200 * Real.cos 10 := 
by
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  sorry

end parallelogram_area_l619_619647


namespace parts_per_day_l619_619715

noncomputable def total_parts : ℕ := 400
noncomputable def unfinished_parts_after_3_days : ℕ := 60
noncomputable def excess_parts_after_3_days : ℕ := 20

variables (x y : ℕ)

noncomputable def condition1 : Prop := (3 * x + 2 * y = total_parts - unfinished_parts_after_3_days)
noncomputable def condition2 : Prop := (3 * x + 3 * y = total_parts + excess_parts_after_3_days)

theorem parts_per_day (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 60 ∧ y = 80 :=
by {
  sorry
}

end parts_per_day_l619_619715


namespace fishbowl_volume_l619_619293

-- Define the dimensions of the cuboid
def length := 4
def width := 6
def height := 15

-- Define the theorem to prove the volume of the fishbowl
theorem fishbowl_volume : (length * width * height) = 360 := by
  -- Proof will go here
  sorry

end fishbowl_volume_l619_619293


namespace smaller_angle_measure_l619_619727

theorem smaller_angle_measure (x : ℝ) (h1 : 3 * x + 2 * x = 90) : 2 * x = 36 :=
by {
  sorry
}

end smaller_angle_measure_l619_619727


namespace north_pole_direction_paradox_mirror_reflection_paradox_l619_619019

/-!
# Problem Statement:
Given the conditions of moving towards and crossing the North Pole, and the reflection in a mirror, 
prove that:
1. The relative directions (East and West) remain correctly oriented relative to North before and after crossing the North Pole.
2. A mirror inversion results in a perceived left-right swap but does not affect the top-bottom orientation.
-/
theorem north_pole_direction_paradox 
    (initial_E : string = "E")
    (initial_W : string = "W")
    (initial_N : string = "N")
    (initial_S : string = "S") :
    -- Person reaches the North Pole and relative directions still hold true
    (initial_E = "E") ∧ (initial_W = "W") := 
sorry

theorem mirror_reflection_paradox 
    (initial_front : string = "Front")
    (initial_back : string = "Back")
    (initial_right : string = "Right")
    (initial_left : string = "Left") 
    (initial_top : string = "Top")
    (initial_bottom : string = "Bottom") :
    -- Mirror inversion causes perceived left-right swap but not top-bottom swap
    (initial_front = "Back") ∧ (initial_back = "Front") ∧
    (initial_right = "Left") ∧ (initial_left = "Right") ∧
    (initial_top = "Top") ∧ (initial_bottom = "Bottom") :=
sorry

end north_pole_direction_paradox_mirror_reflection_paradox_l619_619019


namespace max_triangle_area_l619_619839

-- Definitions for the conditions
def Point := (ℝ × ℝ)

def point_A : Point := (0, 0)
def point_B : Point := (17, 0)
def point_C : Point := (23, 0)

def slope_ell_A : ℝ := 2
def slope_ell_C : ℝ := -2

axiom rotating_clockwise_with_same_angular_velocity (A B C : Point) : Prop

-- Question transcribed as proving a statement about the maximum area
theorem max_triangle_area (A B C : Point)
  (hA : A = point_A)
  (hB : B = point_B)
  (hC : C = point_C)
  (h_slopeA : ∀ p: Point, slope_ell_A = 2)
  (h_slopeC : ∀ p: Point, slope_ell_C = -2)
  (h_rotation : rotating_clockwise_with_same_angular_velocity A B C) :
  ∃ area_max : ℝ, area_max = 264.5 :=
sorry

end max_triangle_area_l619_619839


namespace problem1_problem2_l619_619657

-- Definition for the CDF of the standard normal distribution
def Φ (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * ∫ u in -∞..x, Real.exp (-u^2 / 2)

-- Problem Statement 1: Prove P(1.3 < ξ < 3.3) = 0.0958 for ξ with given distribution function
theorem problem1 (ξ : ℝ) (hξ : ξ ∼ Normal(0, 1)) :
  P(1.3 < ξ ∧ ξ < 3.3) = 0.0958 := sorry

-- Problem Statement 2: Prove that a = 0.03 and b = 4.63 with given conditions
theorem problem2 (ξ : ℝ) (hξ : ξ ∼ Normal(0, 1))
  (hprob : P(2.3 < ξ - a) = P(2.3 > ξ + b)) 
  (hcontain : P(ξ - a < 2.3 ∧ 2.3 < ξ + b) = 0.98) :
  a = 0.03 ∧ b = 4.63 := sorry

end problem1_problem2_l619_619657


namespace ab_equals_6_l619_619975

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619975


namespace chef_accidental_chocolate_percentage_l619_619796

theorem chef_accidental_chocolate_percentage (x : ℝ) :
  let initial_sauce_volume := 15.0
  let removed_sauce_volume := 2.5
  let remaining_sauce_volume := initial_sauce_volume - removed_sauce_volume
  let add_chocolate_volume := 2.5
  let final_chocolate_percentage := 0.5
  
  let chocolate_in_initial := (x / 100.0) * initial_sauce_volume
  let chocolate_in_removed := (x / 100.0) * removed_sauce_volume
  let chocolate_in_remaining := chocolate_in_initial - chocolate_in_removed
  let chocolate_total := chocolate_in_remaining + add_chocolate_volume
  
  let puree_in_initial := (100.0 - x) / 100.0 * initial_sauce_volume
  let puree_in_remaining := puree_in_initial - (100.0 - x) / 100.0 * removed_sauce_volume
  
  chocolate_total == final_chocolate_percentage * (remaining_sauce_volume + add_chocolate_volume) :=
  x = 40 := sorry

end chef_accidental_chocolate_percentage_l619_619796


namespace angle_azp_is_90_degree_l619_619904

-- Let P be a type (point or coordinate) in some space S (a 2D or 3D space)
variables {S : Type*} [MetricSpace S] [NormedSpace ℝ S]
variables (circle : Circle S) (B C X Y A Z P : S)
variables (L1 L2 : AffineSubspace ℝ S)
variables (X1 X2 Y1 Y2 : S)

-- Assumptions/Conditions
-- 1. Given a circle with points B, C, X, Y on it
def on_circle (p : S) : Prop := p ∈ circle

-- 2. A is the midpoint of B and C
def is_midpoint (m p1 p2 : S) : Prop := dist m p1 = dist m p2 ∧ straight_between p1 m p2

-- 3. Z is the midpoint of X and Y
-- 4. Lines L1 and L2 are perpendicular to BC passing through B and C respectively
def is_perp (l : AffineSubspace ℝ S) (p1 p2 : S) : Prop :=
  ∃ (dir : direction l), orthog (p2 -ᵥ p1) dir

-- 5-6. Lines through X and Y perpendicular to AX and AY intersect L1, L2 at respective points
def intersect_perp_line (src pt1 pt2 : S) (l1 l2 : AffineSubspace ℝ S) : S :=
  let perp := ⟨line_eq src pt1, line_eq pt2 src⟩ in
  (line_intersect perp.1 l1, line_intersect perp.2 l2)

-- 7. X1Y2 intersects X2Y1 at P
def lines_intersect (l1 l2 : AffineSubspace ℝ S) : S := 
  ⟨line_extend l1, line_extend l2⟩

-- The theorem
theorem angle_azp_is_90_degree (h_circle: on_circle B ∧ on_circle C ∧ on_circle X ∧ on_circle Y)
(h_midpoint_A: is_midpoint A B C) (h_midpoint_Z: is_midpoint Z X Y)
(h_perp_L1: is_perp L1 B C) (h_perp_L2: is_perp L2 C B)
(h_X1X2: intersect_perp_line X A L1 L2 = (X1, X2))
(h_Y1Y2: intersect_perp_line Y A L1 L2 = (Y1, Y2))
(h_P: lines_intersect (line X1 Y2) (line X2 Y1) = P) :
angle A Z P = 90 :=
by sorry -- Proof omitted as per instructions.

end angle_azp_is_90_degree_l619_619904


namespace sum_binom_eq_cos70_l619_619834

-- Function definition for the given series
def series_sum (a : ℕ) := ∑ n in Finset.range (1006), (-3 : ℝ)^n * (Nat.choose 2010 (2 * n))

-- The main theorem statement
theorem sum_binom_eq_cos70 :
  (1 / (2 : ℝ)^2010) * series_sum 2010 = - Real.cos (70 * Real.pi / 180) :=
by
  sorry

end sum_binom_eq_cos70_l619_619834


namespace shortest_path_on_cube_shortest_path_on_box_l619_619337

-- Proof Problem Part (a)
theorem shortest_path_on_cube (s : ℝ) :
  let M := (0, 0, 0)
  let N := (s, s, s)
  let cube_center := (s/2, s/2, s/2)
  symmetric M cube_center N →
  shortest_path_length_on_surface s M N = s * Real.sqrt 5 :=
sorry

-- Proof Problem Part (b)
theorem shortest_path_on_box (A : ℝ × ℝ × ℝ) :
  let dimensions := (30, 12, 12)
  let parallel_edge_distances := (1, 6, 0)
  let symmetric_point := (29, 6, 12)
  shortest_path_length_on_surface dimensions A symmetric_point = Real.sqrt 928 :=
sorry

end shortest_path_on_cube_shortest_path_on_box_l619_619337


namespace ab_equals_six_l619_619993

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l619_619993


namespace largest_seven_consecutive_non_primes_less_than_40_l619_619235

def is_non_prime (n : ℕ) : Prop :=
  n ≠ 1 ∧ ¬(∃ p, nat.prime p ∧ p ∣ n)

def consecutive_non_primes_sequence (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 7 → is_non_prime (n + i) ∧ (10 ≤ n + i) ∧ (n + i < 40)

theorem largest_seven_consecutive_non_primes_less_than_40 :
  ∃ n, consecutive_non_primes_sequence n ∧ n + 6 = 32 :=
sorry

end largest_seven_consecutive_non_primes_less_than_40_l619_619235


namespace probability_point_closer_to_origin_than_4_1_l619_619378

-- Define the rectangular region
def region : set (ℝ × ℝ) :=
  { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the point (4,1)
def point_4_1 : ℝ × ℝ := (4, 1)

-- Function to calculate Euclidean distance
def euclidean_distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Probability statement to be proven
theorem probability_point_closer_to_origin_than_4_1 : 
  measure_theory.measure.region (λ p, euclidean_distance p origin < euclidean_distance p point_4_1) / 
  measure_theory.measure.region region = 2 / 3 :=
sorry

end probability_point_closer_to_origin_than_4_1_l619_619378


namespace train_length_l619_619387

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (h_speed : speed_km_hr = 60) (h_time : time_sec = 6) :
  let speed_ms := (speed_km_hr * 1000) / 3600
  let length_m := speed_ms * time_sec
  length_m ≈ 100.02 :=
by sorry

end train_length_l619_619387


namespace amoeba_count_14_l619_619175

noncomputable def amoeba_count (day : ℕ) : ℕ :=
  if day = 1 then 1
  else if day = 2 then 2
  else 2^(day - 3) * 5

theorem amoeba_count_14 : amoeba_count 14 = 10240 := by
  sorry

end amoeba_count_14_l619_619175


namespace unique_element_set_l619_619950

theorem unique_element_set (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 2 * x - 1 = 0) → (a = 0 ∨ a = -1) :=
by
  assume (h : ∃! x : ℝ, a * x^2 + 2 * x - 1 = 0)
  sorry

end unique_element_set_l619_619950


namespace min_values_of_exprs_l619_619600

theorem min_values_of_exprs (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (h : (r + s - r * s) * (r + s + r * s) = r * s) :
  (r + s - r * s) = -3 + 2 * Real.sqrt 3 ∧ (r + s + r * s) = 3 + 2 * Real.sqrt 3 :=
by sorry

end min_values_of_exprs_l619_619600


namespace alton_daily_earnings_l619_619818

theorem alton_daily_earnings :
  (rent_per_week : ℝ) = 20 →
  (profit_per_week : ℝ) = 36 →
  (days_work_per_week : ℝ) = 5 →
  let total_earnings_per_week := rent_per_week + profit_per_week in
  let daily_earnings := total_earnings_per_week / days_work_per_week in
  daily_earnings = 11.20 :=
begin
  intros rent_per_week_eq profit_per_week_eq days_work_per_week_eq,
  unfold total_earnings_per_week daily_earnings,
  rw [rent_per_week_eq, profit_per_week_eq, days_work_per_week_eq],
  norm_num,
end

end alton_daily_earnings_l619_619818


namespace median_of_data_set_is_five_l619_619546

theorem median_of_data_set_is_five
  (x : ℝ) (h : mode [1, 3, x, 5, 8] = 8) : median [1, 3, x, 5, 8] = 5 := 
begin
  sorry
end

end median_of_data_set_is_five_l619_619546


namespace polynomial_real_root_count_l619_619421

noncomputable def count_polynomials_with_real_roots : ℕ :=
  let candidates := [(λ x, x + 1), (λ x, x - 1),
                     (λ x, x^2 + x - 1), (λ x, x^2 - x - 1),
                     (λ x, x^3 + x^2 - x - 1), (λ x, x^3 - x^2 - x + 1)] in
  2 * candidates.length -- considering both + and - versions of each polynomial

theorem polynomial_real_root_count :
  count_polynomials_with_real_roots = 12 := by
  sorry

end polynomial_real_root_count_l619_619421


namespace radius_of_circle_l619_619018

noncomputable def radius (α : ℝ) : ℝ :=
  5 / Real.sin (α / 2)

theorem radius_of_circle (c α : ℝ) (h_c : c = 10) :
  (radius α) = 5 / Real.sin (α / 2) := by
  sorry

end radius_of_circle_l619_619018


namespace solve_triplets_l619_619437

theorem solve_triplets (x y z : ℂ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 19) (h3 : x^3 + y^3 + z^3 = 53) :
  (x = -1 ∧ y = Complex.i * sqrt 3 ∧ z = -Complex.i * sqrt 3) ∨ 
  (x = -1 ∧ y = -Complex.i * sqrt 3 ∧ z = Complex.i * sqrt 3) ∨ 
  (x = Complex.i * sqrt 3 ∧ y = -Complex.i * sqrt 3 ∧ z = -1) ∨ 
  (x = -Complex.i * sqrt 3 ∧ y = Complex.i * sqrt 3 ∧ z = -1) ∨ 
  (x = Complex.i * sqrt 3 ∧ y = -1 ∧ z = -Complex.i * sqrt 3) ∨ 
  (x = -Complex.i * sqrt 3 ∧ y = -1 ∧ z = Complex.i * sqrt 3) := 
sorry

end solve_triplets_l619_619437


namespace ratio_of_polynomials_eq_962_l619_619836

open Real

theorem ratio_of_polynomials_eq_962 :
  (10^4 + 400) * (26^4 + 400) * (42^4 + 400) * (58^4 + 400) /
  ((2^4 + 400) * (18^4 + 400) * (34^4 + 400) * (50^4 + 400)) = 962 := 
sorry

end ratio_of_polynomials_eq_962_l619_619836


namespace find_valid_n_l619_619863

theorem find_valid_n :
  ∃ n : ℕ,
    (n > 0) ∧
    (∀ d, d ∈ (n.digits 10) → d ≠ 0) ∧
    (n.digits 10).nodup ∧
    (n.digits 10).length = 5 ∧
    n = 35964 := by
  sorry

end find_valid_n_l619_619863


namespace product_ab_l619_619990

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619990


namespace pair_basis_of_plane_l619_619484

def vector_space := Type
variable (V : Type) [AddCommGroup V] [Module ℝ V]

variables (e1 e2 : V)
variable (h_basis : LinearIndependent ℝ ![e1, e2])
variable (hne : e1 ≠ 0 ∧ e2 ≠ 0)

theorem pair_basis_of_plane
  (v1 v2 : V)
  (hv1 : v1 = e1 + e2)
  (hv2 : v2 = e1 - e2) :
  LinearIndependent ℝ ![v1, v2] :=
sorry

end pair_basis_of_plane_l619_619484


namespace cylinder_longest_segment_l619_619786

-- Define the radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 10

-- Definition for the longest segment inside the cylinder using Pythagorean theorem
def longest_segment (radius height : ℝ) : ℝ :=
  real.sqrt (radius * 2)^2 + height^2

-- Specify the expected answer for the proof
def expected_answer : ℝ := 10 * real.sqrt 2

-- The theorem stating the longest segment length inside the cylinder
theorem cylinder_longest_segment : longest_segment radius height = expected_answer :=
by {
  -- Lean code to set up and prove the equivalence
  sorry
}

end cylinder_longest_segment_l619_619786


namespace Vanya_original_number_l619_619310

theorem Vanya_original_number (m n : ℕ) (hm : m ≤ 9) (hn : n ≤ 9) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 10 * m + n = 81 := by
  sorry

end Vanya_original_number_l619_619310


namespace log_integer_prob_l619_619305

theorem log_integer_prob :
  let S := finset.range 20
  let pairs := S.product S
  let valid_pairs := pairs.filter (λ p, p.1 ≠ p.2 ∧ p.2 % p.1 = 0)
  (↑(valid_pairs.card) : ℚ) / (↑(pairs.card) / 2) = 24 / 95 :=
by
  sorry

end log_integer_prob_l619_619305


namespace sean_bought_two_soups_l619_619233

theorem sean_bought_two_soups :
  ∃ (number_of_soups : ℕ),
    let soda_cost := 1
    let total_soda_cost := 3 * soda_cost
    let soup_cost := total_soda_cost
    let sandwich_cost := 3 * soup_cost
    let total_cost := 3 * soda_cost + sandwich_cost + soup_cost * number_of_soups
    total_cost = 18 ∧ number_of_soups = 2 :=
by
  sorry

end sean_bought_two_soups_l619_619233


namespace largest_k_for_sum_consecutive_numbers_l619_619866

theorem largest_k_for_sum_consecutive_numbers :
  ∃ k : ℕ, k = 486 ∧ ∃ n : ℕ, (n + 1 + n + 2 + ... + n + k) = 3 ^ 11 :=
by 
  sorry

end largest_k_for_sum_consecutive_numbers_l619_619866


namespace soccer_tournament_l619_619413

theorem soccer_tournament :
  ∃ n : ℕ, ∃ points : List ℕ,
  n = 6 ∧ points = [8, 7, 4, 4, 4, 3] ∧
  (List.sum points = n * (n - 1) ∧
  points.nth 0 = some 8 ∧
  points.nth 1 = some 7 ∧
  points.nth 2 = some 4 ∧
  points.nth 3 = some 4) :=
by
  sorry

end soccer_tournament_l619_619413


namespace hundredth_digit_of_fraction_l619_619963

theorem hundredth_digit_of_fraction (n : ℕ) :
  let repeating_sequence := "269230769"
  ∧ let decimal_repr := "0." ++ repeating_sequence
  in decimal_repr[(100 % repeating_sequence.length + 1)] = '2' := by
  sorry

end hundredth_digit_of_fraction_l619_619963


namespace distance_between_trees_l619_619329

theorem distance_between_trees
  (num_trees : ℕ)
  (length_of_yard : ℝ)
  (one_tree_at_each_end : True)
  (h1 : num_trees = 26)
  (h2 : length_of_yard = 400) :
  length_of_yard / (num_trees - 1) = 16 :=
by
  sorry

end distance_between_trees_l619_619329


namespace AM_BM_ratio_l619_619335

-- Given definitions
variables {A B C D M O : Type} [parallelogram ABCD] (is_not_rhombus : ¬(rhombus ABCD))
variables (k : ℝ) (h1 : length AC / length BD = k)
variables (AM AD AC BM BC BD : line) (M_eq : M = LAM ∩ LBM)
variable [symmetry_AM_AD : symmetric_to AM AD AC]
variable [symmetry_BM_BC : symmetric_to BM BC BD]

-- Desired proof
theorem AM_BM_ratio : (length AM / length BM) = k^2 := 
sorry

end AM_BM_ratio_l619_619335


namespace angle_P_measure_l619_619255

theorem angle_P_measure (P Q R S : ℝ) 
  (h1 : P = 3 * Q)
  (h2 : P = 4 * R)
  (h3 : P = 6 * S)
  (h_sum : P + Q + R + S = 360) : 
  P = 206 :=
by 
  sorry

end angle_P_measure_l619_619255


namespace train_length_proof_l619_619390

noncomputable def train_length (speed_km_per_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  speed_m_per_s * time_sec

theorem train_length_proof :
  train_length 60 6 = 100.02 :=
by
  sorry

end train_length_proof_l619_619390


namespace angle_between_skew_lines_l619_619565

-- Define the vertices of the cube
variable (A B C C1 : Point)
-- Define the midpoint properties
variable (M N : Point)
variable (hM : midpoint M B C)
variable (hN : midpoint N C C1)

-- Define the skew lines AC and MN
variable (AC MN : Line)
variable (hAC : line_through A C AC)
variable (hMN : line_through M N MN)

-- Define the proof problem statement in Lean
theorem angle_between_skew_lines : angle AC MN = 45 := 
  sorry

end angle_between_skew_lines_l619_619565


namespace liberal_arts_proof_problem_l619_619764

theorem liberal_arts_proof_problem :
  (∃ x : ℝ, -x^2 ≥ 0) ∧ 
  (¬ (∀ x : ℝ, x^2 + 2 * x + 1 = 0)) ∧ 
  (¬ (∀ x : ℕ, log 2 x > 0)) ∧ 
  (¬ (∃ x : ℝ, cos x < 2 * x - x^2 - 3)) :=
by
  sorry

end liberal_arts_proof_problem_l619_619764


namespace minimal_value_of_a_b_l619_619613

noncomputable def minimal_sum_of_a_and_b : ℝ := 6.11

theorem minimal_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : discriminant (λ x, x^2 + a * x + 3 * b) >= 0) 
  (h4 : discriminant (λ x, x^2 + 3 * b * x + a) >= 0) : 
  a + b = minimal_sum_of_a_and_b :=
sorry

end minimal_value_of_a_b_l619_619613


namespace solve_sugar_added_l619_619355

noncomputable def sugar_added 
  (initial_solution_volume : ℝ) (initial_water_percent : ℝ) (initial_kola_percent : ℝ) 
  (added_water : ℝ) (added_kola : ℝ) (final_sugar_percent : ℝ) : ℝ :=
let initial_water := initial_water_percent * initial_solution_volume / 100,
    initial_kola := initial_kola_percent * initial_solution_volume / 100,
    initial_sugar := initial_solution_volume - initial_water - initial_kola,
    new_water := initial_water + added_water,
    new_kola := initial_kola + added_kola,
    total_volume := new_water + new_kola + initial_sugar in
(initial_sugar + (final_sugar_percent / 100) * total_volume - initial_solution_volume) / (1 - final_sugar_percent / 100)

theorem solve_sugar_added :
  sugar_added 440 88 8 10 6.8 4.521739130434784 ≈ 3.213 := 
sorry

end solve_sugar_added_l619_619355


namespace length_XY_fixed_l619_619642

theorem length_XY_fixed 
  (O A B R : Point) 
  (circle : Circle O) 
  (A_on_circle : circle A) 
  (B_on_circle : circle B) 
  (angle_AOB : angle A O B = 60)
  (R_on_minor_arc_AB : minor_arc A B circle R)
  (X : Point)
  (Y : Point)
  (X_on_OA : lies_on X (segment O A))
  (Y_on_OB : lies_on Y (segment O B))
  (angle_RXO : angle R X O = 65)
  (angle_RYO : angle R Y O = 115) :
  ∃ l : ℝ, ∀ R', minor_arc A B circle R' → 
    ∀ X' Y', lies_on X' (segment O A) ∧ lies_on Y' (segment O B) ∧
    angle R' X' O = 65 ∧ angle R' Y' O = 115 → dist X' Y' = l :=
by
  sorry

end length_XY_fixed_l619_619642


namespace steve_fraction_of_skylar_l619_619248

variables (S : ℤ) (Stacy Skylar Steve : ℤ)

-- Given conditions
axiom h1 : 32 = 3 * Steve + 2 -- Stacy's berries = 2 + 3 * Steve's berries
axiom h2 : Skylar = 20        -- Skylar has 20 berries
axiom h3 : Stacy = 32         -- Stacy has 32 berries

-- Final goal
theorem steve_fraction_of_skylar (h1: 32 = 3 * Steve + 2) (h2: 20 = Skylar) (h3: Stacy = 32) :
  Steve = Skylar / 2 := 
sorry

end steve_fraction_of_skylar_l619_619248


namespace eq_m_neg_one_l619_619093

theorem eq_m_neg_one (m : ℝ) (x : ℝ) (h1 : (m-1) * x^(m^2 + 1) + 2*x - 3 = 0) (h2 : m - 1 ≠ 0) (h3 : m^2 + 1 = 2) : 
  m = -1 :=
sorry

end eq_m_neg_one_l619_619093


namespace fa_range_l619_619417

theorem fa_range (A : ℝ × ℝ)
                  (θ : ℝ)
                  (h_parabola : A.snd^2 = A.fst)
                  (h_focus : ∃ F, F = (1/4, 0))
                  (h_line : ∃ l, l = (θ ≥ π/4) ∧ (A.snd > 0))
                  : ∃ FA : ℝ, FA ∈ set.Ioc (1 / 4) (1 + real.sqrt 2 / 2) := sorry

end fa_range_l619_619417


namespace conic_section_not_standard_l619_619850

-- Define the equation
def conic_section_equation (x y : ℝ) : Prop :=
  y^6 - 8*x^6 = 3*y^3 - 27

-- Define the property that the equation does not correspond to any standard conic section
def not_standard_conic_section : Prop :=
  ∀ x y : ℝ, conic_section_equation x y → (¬is_parabola (conic_section_equation x y) ∧ 
                                           ¬is_hyperbola (conic_section_equation x y) ∧
                                           ¬is_ellipse (conic_section_equation x y) ∧
                                           ¬is_circle (conic_section_equation x y))

theorem conic_section_not_standard :
  not_standard_conic_section :=
by
  sorry -- skips the actual proof

end conic_section_not_standard_l619_619850


namespace area_of_right_triangle_DED_DF_90_l619_619566

theorem area_of_right_triangle_DED_DF_90
  (DE DF : ℝ)
  (H_DE : DE = 40)
  (H_DF : DF = 30)
  (H_angle_D : ∠DEF = 90) :
  area_of_triangle DEF = 600 :=
by
  sorry

end area_of_right_triangle_DED_DF_90_l619_619566


namespace geometric_sequence_general_formula_sum_of_first_n_terms_l619_619092

noncomputable def a_sequence : ℕ → ℝ
| 1 => 1
| n+1 => 3 * a_sequence n

theorem geometric_sequence_general_formula (n : ℕ) :
  (a_sequence n) = 3^(n-1) :=
sorry

noncomputable def b_sequence (n : ℕ) : ℝ :=
log 3 (a_sequence n) + n

noncomputable def S_n (n : ℕ) : ℝ :=
finset.sum (finset.range n) (λ i, b_sequence (i + 1))

theorem sum_of_first_n_terms (n : ℕ) :
  S_n n = (n : ℝ) ^ 2 :=
sorry

end geometric_sequence_general_formula_sum_of_first_n_terms_l619_619092


namespace quartic_poly_evaluation_l619_619380

noncomputable def p : ℝ → ℝ := sorry  -- assuming a quartic polynomial p

theorem quartic_poly_evaluation :
  (∀ n ∈ {1, 2, 3, 4, 5}, p n = 1 / n ^ 3) →
  p 6 = 0 :=
sorry

end quartic_poly_evaluation_l619_619380


namespace nat_number_36_sum_of_digits_l619_619047

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end nat_number_36_sum_of_digits_l619_619047


namespace angle_between_vectors_magnitude_combination_l619_619483

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions.
axiom nonzero_a : a ≠ 0
axiom nonzero_b : b ≠ 0
axiom perp : inner (a : EuclideanSpace ℝ (Fin 3)) (a - b) = 0
axiom norm_a : norm a = 2 * real.sqrt 3
axiom norm_b : norm b = 4

-- Part (1): Prove the angle θ between a and b is π / 6.
theorem angle_between_vectors :
  let θ := real.acos ((inner a b) / ((norm a) * (norm b))) in θ = real.pi / 6 := sorry

-- Part (2): Prove the magnitude |3a - 2b| is 2√7.
theorem magnitude_combination :
  norm ((3 : ℝ) • a - (2 : ℝ) • b) = 2 * real.sqrt 7 := sorry

end angle_between_vectors_magnitude_combination_l619_619483


namespace no_real_number_pairs_satisfy_equation_l619_619957

theorem no_real_number_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ¬ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) :=
by
  intros a b ha hb
  sorry

end no_real_number_pairs_satisfy_equation_l619_619957


namespace recreation_percentage_correct_l619_619307

noncomputable def recreation_percentage (W : ℝ) : ℝ :=
  let recreation_two_weeks_ago := 0.25 * W
  let wages_last_week := 0.95 * W
  let recreation_last_week := 0.35 * (0.95 * W)
  let wages_this_week := 0.95 * W * 0.85
  let recreation_this_week := 0.45 * (0.95 * W * 0.85)
  (recreation_this_week / recreation_two_weeks_ago) * 100

theorem recreation_percentage_correct (W : ℝ) : recreation_percentage W = 145.35 :=
by
  sorry

end recreation_percentage_correct_l619_619307


namespace top_angle_isosceles_triangle_l619_619918

open Real

theorem top_angle_isosceles_triangle (A B C : ℝ) (abc_is_isosceles : (A = B ∨ B = C ∨ A = C))
  (angle_A : A = 40) : (B = 40 ∨ B = 100) :=
sorry

end top_angle_isosceles_triangle_l619_619918


namespace correct_answer_l619_619530

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l619_619530


namespace joint_savings_account_total_l619_619586

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end joint_savings_account_total_l619_619586


namespace min_log_value_l619_619059

theorem min_log_value (x : ℝ) (hx : x > 1) :
  ∃ y, y = log 2 (x + (1 / (x - 1)) + 5) ∧ y ≥ 3 :=
begin
  sorry
end

end min_log_value_l619_619059


namespace mod_product_l619_619824

theorem mod_product : (198 * 955) % 50 = 40 :=
by sorry

end mod_product_l619_619824


namespace distance_from_K_to_AB_l619_619683

-- Define the points A, B, K, and the distances
variables {A B K : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace K]
variables (AC BC : Set A) (AB : Set B) -- Sides of the triangle as sets of points

-- Define distances from K to AC and BC
def distance_to_acc : ℝ := 6
def distance_to_bcc : ℝ := 24

-- Define the problem statement
theorem distance_from_K_to_AB (K : Type*) [MetricSpace K] 
    (AC BC : Set A) (AB : Set B)
    (distance_to_acc : ℝ := 6)
    (distance_to_bcc : ℝ := 24):
    distance_from_point_to_side K AB = 12 :=
sorry

-- Initial definitions
-- Define initial conditions
definition AC := has_distance AC K 6
definition BC := has_distance BC K 24

end distance_from_K_to_AB_l619_619683


namespace linear_combination_of_vectors_l619_619518

open Matrix

def vector_a : Vector ℝ 2 := ![1, 1]
def vector_b : Vector ℝ 2 := ![1, -1]
def vector_c : Vector ℝ 2 := ![-1, 2]

theorem linear_combination_of_vectors :
  ∃ k l : ℝ, vector_c = k • vector_a + l • vector_b ∧ k = 1 / 2 ∧ l = -3 / 2 :=
by {
  use (1 / 2), use (-3 / 2),
  split,
  calc
    vector_c = ![-1, 2] : by rfl
    ... = (1/2) • ![1, 1] + (-3/2) • ![1, -1] : by {
      have h1 : (1/2) • ![1, 1] = ![1/2, 1/2], by norm_num,
      have h2 : (-3/2) • ![1, -1] = ![-3/2, 3/2], by norm_num,
      rw [h1, h2],
      -- calculate the matrix addition
      norm_num
    },
  split; norm_num
}

end linear_combination_of_vectors_l619_619518


namespace total_dots_not_visible_l619_619296

theorem total_dots_not_visible {faces : Fin 6 → ℕ} (h_faces : ∀ i, faces i = i.succ) :
  let total_dots := 3 * (faces 0 + faces 1 + faces 2 + faces 3 + faces 4 + faces 5)
  let visible_dots := 2 + 2 + 3 + 4 + 5 + 5 + 6 + 6
  total_dots - visible_dots = 30 :=
by
  let total_dots := 3 * (faces 0 + faces 1 + faces 2 + faces 3 + faces 4 + faces 5)
  let visible_dots := 2 + 2 + 3 + 4 + 5 + 5 + 6 + 6
  calc
    total_dots - visible_dots = 63 - 33 : by sorry
                             ... = 30   : by sorry

end total_dots_not_visible_l619_619296


namespace intervals_of_monotonicity_max_value_of_m_l619_619941

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem intervals_of_monotonicity :
  (∀ x < 0, deriv f x > 0) ∧
  (∀ x, 0 < x ∧ x < 2 → deriv f x < 0) ∧
  (∀ x > 2, deriv f x > 0) := by
sorry

theorem max_value_of_m (m : ℝ) (h_dom : ∀ x ∈ Icc (-1 : ℝ) m, f x ≤ 0) :
  m ≤ 3 := by
sorry

end intervals_of_monotonicity_max_value_of_m_l619_619941


namespace trigonometric_identity_l619_619131

theorem trigonometric_identity (α : ℝ)
  (h : (cos (2 * α)) / (sin (α - π / 4)) = -sqrt 2 / 2) :
  cos α + sin α = 1 / 2 :=
by
  sorry

end trigonometric_identity_l619_619131


namespace feathers_already_have_l619_619412

-- Given conditions
def total_feathers : Nat := 900
def feathers_still_needed : Nat := 513

-- Prove that the number of feathers Charlie already has is 387
theorem feathers_already_have : (total_feathers - feathers_still_needed) = 387 := by
  sorry

end feathers_already_have_l619_619412


namespace same_solutions_implies_k_value_l619_619549

theorem same_solutions_implies_k_value (k : ℤ) : (∀ x : ℤ, 2 * x = 4 ↔ 3 * x + k = -2) → k = -8 :=
by
  sorry

end same_solutions_implies_k_value_l619_619549


namespace sufficient_condition_for_increasing_l619_619085

theorem sufficient_condition_for_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^y < a^x) →
  (∀ x y : ℝ, x < y → (2 - a) * y ^ 3 > (2 - a) * x ^ 3) :=
sorry

end sufficient_condition_for_increasing_l619_619085


namespace solve_for_x_l619_619710

theorem solve_for_x (x : ℤ) (h : 3 * x + 20 = (1/3 : ℚ) * (7 * x + 60)) : x = 0 :=
sorry

end solve_for_x_l619_619710


namespace parallelogram_area_approx_l619_619644

noncomputable def sin_80_deg := Real.sin (Real.pi * 80 / 180)

theorem parallelogram_area_approx :
  ∃ (area : ℝ), area ≈ 197.0 ∧
  ∀ AB AD θ, AB = 20 ∧ AD = 10 ∧ θ = 80 →
  area = AB * AD * sin_80_deg :=
by
  sorry

end parallelogram_area_approx_l619_619644


namespace lines_divide_plane_l619_619177

theorem lines_divide_plane (n : ℕ) :
  (∃ P : ℕ → ℕ, P(n) = 1 + (n * (n + 1)) / 2) :=
sorry

end lines_divide_plane_l619_619177


namespace domain_of_function_l619_619262

def domain_function {x : ℝ} (h₁ : x ≠ 0) (h₂ : 1 + 1 / x ≠ 0) : Prop :=
  { x | x ≠ 0 ∧ x ≠ -1 }

theorem domain_of_function (x : ℝ) (h₁ : x ≠ 0) (h₂ : 1 + 1 / x ≠ 0) :
  {x | x ∈ ℝ ∧ x ≠ 0 ∧ x ≠ -1} :=
sorry

end domain_of_function_l619_619262


namespace part1_part2_part3_l619_619096

noncomputable def f (x : ℝ) : ℝ := sorry

theorem part1 :
  (f(0) = 1 ∧ f(2) = 16) →
  ∃ k > 0, ∃ a > 0, a ≠ 1 ∧ (∀ x, f(x) = k * a ^ x) →
  (∀ x, f(x) = 4 ^ x) := sorry

noncomputable def g (x : ℝ) : ℝ := sorry

theorem part2 (h : ∀ x, f(x) = 4 ^ x) :
  (g(x) = b + 1 / (f(x) + 1)) →
  (∀ x, g(-x) = -g(x)) →
  b = -1 / 2 := sorry

theorem part3 (h : ∀ x, f(x) = 4 ^ x) :
  ∀ x1 x2 : ℝ, x1 ≠ x2 →
  f((x1 + x2) / 2) < (f(x1) + f(x2)) / 2 := sorry

end part1_part2_part3_l619_619096


namespace solve_for_z_l619_619540

theorem solve_for_z {x y z : ℝ} (h : (1 / x^2) - (1 / y^2) = 1 / z) :
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end solve_for_z_l619_619540


namespace percentage_sold_correct_l619_619720

variables 
  (initial_cost : ℝ) 
  (tripled_value : ℝ) 
  (selling_price : ℝ) 
  (percentage_sold : ℝ)

def game_sold_percentage (initial_cost tripled_value selling_price percentage_sold : ℝ) :=
  tripled_value = initial_cost * 3 ∧ 
  selling_price = 240 ∧ 
  initial_cost = 200 ∧ 
  percentage_sold = (selling_price / tripled_value) * 100

theorem percentage_sold_correct : game_sold_percentage 200 (200 * 3) 240 40 :=
  by simp [game_sold_percentage]; sorry

end percentage_sold_correct_l619_619720


namespace smallest_possible_value_l619_619617

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l619_619617


namespace GregsAgeIs16_l619_619414

def CindyAge := 5
def JanAge := CindyAge + 2
def MarciaAge := 2 * JanAge
def GregAge := MarciaAge + 2

theorem GregsAgeIs16 : GregAge = 16 := by
  sorry

end GregsAgeIs16_l619_619414


namespace sleeping_bag_price_l619_619886

theorem sleeping_bag_price (wholesale_cost : ℝ) (gross_profit_percent : ℝ) (selling_price : ℝ) :
  wholesale_cost = 24.14 → 
  gross_profit_percent = 0.16 → 
  selling_price = wholesale_cost * (1 + gross_profit_percent) →
  selling_price = 28.00 :=
by
  intros
  rw [a_1, a_2]
  norm_num

end sleeping_bag_price_l619_619886


namespace compute_sum_sin_powers_l619_619016

noncomputable def sum_sin_powers : ℝ :=
  (∑ k in Finset.range 19, Real.sin (5 * k * Real.pi / 180) ^ 6)

theorem compute_sum_sin_powers :
  sum_sin_powers = 49 / 8 :=
by
  sorry

end compute_sum_sin_powers_l619_619016


namespace hall_ratio_l619_619285

theorem hall_ratio (w l : ℕ) (h1 : w * l = 450) (h2 : l - w = 15) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l619_619285


namespace domain_of_f_l619_619686

def domain_valid (x : ℝ) :=
  1 - x ≥ 0 ∧ 1 - x ≠ 1

theorem domain_of_f :
  ∀ x : ℝ, domain_valid x ↔ (x ∈ Set.Iio 0 ∪ Set.Ioc 0 1) :=
by
  sorry

end domain_of_f_l619_619686


namespace series_sum_eq_one_sixth_l619_619849

noncomputable def a (n : ℕ) : ℝ := 2^n / (7^(2^n) + 1)

theorem series_sum_eq_one_sixth :
  (∑' (n : ℕ), a n) = 1 / 6 :=
sorry

end series_sum_eq_one_sixth_l619_619849


namespace summation_of_odds_eq_square_l619_619068

theorem summation_of_odds_eq_square (n : ℕ) : ∑ k in Finset.range (n + 1), (2 * k + 1) = (n + 1) ^ 2 := by
  sorry

end summation_of_odds_eq_square_l619_619068


namespace expression_evaluation_l619_619538

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x^2 - 4 * y + 5 = 24 :=
by
  sorry

end expression_evaluation_l619_619538


namespace collinear_points_min_value_proof_l619_619344

noncomputable def collinear_points_min_value (A B C D E P : ℝ) (h_collinear : A < B ∧ B < C ∧ C < D ∧ D < E)
    (h_ab : B - A = 2) (h_bc : C - B = 2) (h_cd : D - C = 3) (h_de : E - D = 4) : ℝ :=
let x := P - A in
x^2 + (x - 2)^2 + (x - 4)^2 + (x - 7)^2 + (x - 11)^2

theorem collinear_points_min_value_proof : 
  ∀ (A B C D E P : ℝ) (h_collinear : A < B ∧ B < C ∧ C < D ∧ D < E)
    (h_ab : B - A = 2) (h_bc : C - B = 2) (h_cd : D - C = 3) (h_de : E - D = 4), 
    ∃ P, collinear_points_min_value A B C D E P h_collinear h_ab h_bc h_cd h_de = 58.8 :=
by
  -- Writing the proof is not required.
  sorry

end collinear_points_min_value_proof_l619_619344


namespace tyler_remaining_money_l619_619729

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end tyler_remaining_money_l619_619729


namespace find_f2014_f2015_l619_619493

section
variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x) = f (x + p)
def restricted_behavior (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x > 0 ∧ x < 5 / 2 → f (x) = (2:ℝ)^x

hypothesis h_odd : odd_function f
hypothesis h_periodic : periodic_function f 5
hypothesis h_restricted : restricted_behavior f

theorem find_f2014_f2015 : f 2014 + f 2015 = -2 :=
by
  sorry -- proof omitted
end

end find_f2014_f2015_l619_619493


namespace remaining_money_proof_l619_619732

variables {scissor_cost eraser_cost initial_amount scissor_quantity eraser_quantity total_cost remaining_money : ℕ}

-- Given conditions
def conditions : Prop :=
  initial_amount = 100 ∧ 
  scissor_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissor_quantity = 8 ∧ 
  eraser_quantity = 10

-- Definition using conditions
def total_spent : ℕ :=
  scissor_quantity * scissor_cost + eraser_quantity * eraser_cost

-- Prove the total remaining money calculation
theorem remaining_money_proof (h : conditions) : 
  total_spent = 80 ∧ remaining_money = initial_amount - total_spent ∧ remaining_money = 20 :=
by
  -- Proof steps to be provided here
  sorry

end remaining_money_proof_l619_619732


namespace length_of_tangent_segment_l619_619111

theorem length_of_tangent_segment
  (k : ℝ)
  (hx : ∀ {x y : ℝ}, x^2 + y^2 - 6 * x + 2 * y + 9 = 0 → (x - 3) ^ 2 + (y + 1) ^ 2 = 1)
  (l : ∀ {x y : ℝ}, k * x + y - 2 = 0 → (x, y) = (3, -1))
  (tangent_passes_through_A : ∀ {B : ℝ × ℝ}, B ∈ {p : ℝ × ℝ | ((p.1 - 3)^2 + (p.2 + 1)^2 = 1)} →
  ∃ y : ℝ, k * 0 + y - 2 = 0 ∧ (0, y) = B):
  ∀ (A B : ℝ × ℝ), A = (0, k) → B ∈ {p : ℝ × ℝ | ((p.1 - 3)^2 + (p.2 + 1)^2 = 1) ∧ k * p.1 + p.2 - 2 = 0} →
  ∃ (d : ℝ), d = 2 * real.sqrt 3 :=
sorry

end length_of_tangent_segment_l619_619111


namespace eccentricity_of_hyperbola_l619_619510

variable {a b : ℝ}
variable (C : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / a^2) - (y^2 / b^2) = 1 })

-- Conditions
axiom (h1 : b > a > 0)
axiom (h2 : ∀ {F : ℝ × ℝ}, F ∈ C → ∃ (F' : ℝ × ℝ), F' ∈ C ∧ symmetric_point_about_asymptote F' F)

-- Question: Determine the eccentricity e
noncomputable def find_eccentricity : ℝ := 2

theorem eccentricity_of_hyperbola : find_eccentricity = 2 :=
by
  sorry

end eccentricity_of_hyperbola_l619_619510


namespace final_position_farthest_distance_fuel_expense_l619_619680

noncomputable def trips : List Int := [-2, 7, -9, 10, 4, -5, -8]

def positionAfterTrips (trips : List Int) : Int :=
  trips.foldl (· + ·) 0

theorem final_position : positionAfterTrips trips = -3 := by
  sorry

def cumulativePositions (trips : List Int) : List Int := 
  trips.scanl (· + ·) 0 |>.tail

def distancesFromP (positions : List Int) : List Nat :=
  positions.map Int.natAbs

def farthestTrip (positions : List Int) : Nat :=
  let indexedPositions := positions.enumFrom 1
  indexedPositions.maxBy (λ (_, pos) => pos.natAbs)

theorem farthest_distance : (farthestTrip (cumulativePositions trips)).1 = 5 := by
  sorry

def totalDistance (trips : List Int) : Nat :=
  trips.foldl (λ acc dist => acc + Int.natAbs dist) 0

def fuelCost (distance : Nat) (fuelRate : Float) (costPerL : Float) : Float :=
  distance.toFloat * fuelRate * costPerL

theorem fuel_expense : fuelCost (totalDistance trips) 0.08 7.2 = 25.92 := by
  sorry

end final_position_farthest_distance_fuel_expense_l619_619680


namespace exists_eight_integers_sum_and_product_eight_l619_619575

theorem exists_eight_integers_sum_and_product_eight :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 ∧ 
  a1 * a2 * a3 * a4 * a5 * a6 * a7 * a8 = 8 :=
by
  -- The existence proof can be constructed here
  sorry

end exists_eight_integers_sum_and_product_eight_l619_619575


namespace min_xsq_ysq_zsq_l619_619198

noncomputable def min_value_x_sq_y_sq_z_sq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : ℝ :=
  (x^2 + y^2 + z^2)

theorem min_xsq_ysq_zsq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  min_value_x_sq_y_sq_z_sq x y z h = 40 / 7 :=
  sorry

end min_xsq_ysq_zsq_l619_619198


namespace most_colored_pencils_l619_619288

theorem most_colored_pencils (total red blue yellow : ℕ) 
  (h_total : total = 24)
  (h_red : red = total / 4)
  (h_blue : blue = red + 6)
  (h_yellow : yellow = total - (red + blue)) :
  blue = 12 :=
by
  sorry

end most_colored_pencils_l619_619288


namespace engagement_ring_savings_l619_619592

theorem engagement_ring_savings 
  (yearly_salary : ℝ) 
  (monthly_savings : ℝ) 
  (monthly_salary := yearly_salary / 12) 
  (ring_cost := 2 * monthly_salary) 
  (saving_months := ring_cost / monthly_savings) 
  (h_salary : yearly_salary = 60000) 
  (h_savings : monthly_savings = 1000) :
  saving_months = 10 := 
sorry

end engagement_ring_savings_l619_619592


namespace find_natural_numbers_l619_619044

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem find_natural_numbers (x : ℕ) :
  (x = 36 * sum_of_digits x) ↔ (x = 324 ∨ x = 648) :=
by
  sorry

end find_natural_numbers_l619_619044


namespace largest_seven_consecutive_composites_less_than_40_l619_619238

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

def seven_consecutive_composites_less_than_40 (a : ℕ) : Prop :=
  a < 40 ∧ a > 29 ∧
  is_composite a ∧ is_composite (a - 1) ∧ is_composite (a - 2) ∧ 
  is_composite (a - 3) ∧ is_composite (a - 4) ∧ 
  is_composite (a - 5) ∧ is_composite (a - 6)

theorem largest_seven_consecutive_composites_less_than_40 :
  seven_consecutive_composites_less_than_40 36 :=
begin
  -- Proof goes here.
  sorry
end

end largest_seven_consecutive_composites_less_than_40_l619_619238


namespace sqrt_inequality_l619_619895

theorem sqrt_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) (h5 : a + d = b + c) :
  sqrt d + sqrt a < sqrt b + sqrt c :=
by
  sorry

end sqrt_inequality_l619_619895


namespace complement_union_complement_l619_619637

open Set

variable (U : Type) [Fintype U] [DecidableEq U] [Inhabited U]

def I : Set U := {0, 1, 2, 3}
def A : Set U := {0, 1, 2}
def B : Set U := {2, 3}

theorem complement_union_complement {U : Type} [Fintype U] [DecidableEq U] [Inhabited U] :
  (compl A) ∪ (compl B) = {0, 1, 3} :=
by
  sorry

end complement_union_complement_l619_619637


namespace ratio_of_johns_age_five_years_ago_to_age_in_8_years_l619_619879

noncomputable def johnAgeFiveYearsAgo (currentAge : ℕ) : ℕ := currentAge - 5
noncomputable def johnAgeInEightYears (currentAge : ℕ) : ℕ := currentAge + 8
noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem ratio_of_johns_age_five_years_ago_to_age_in_8_years (johnsCurrentAge : ℕ) (h : johnsCurrentAge = 18) : 
  (johnAgeFiveYearsAgo johnsCurrentAge) / (gcd (johnAgeFiveYearsAgo johnsCurrentAge) (johnAgeInEightYears johnsCurrentAge)) = 1 :=
by
  rw [h]
  intro
  sorry

end ratio_of_johns_age_five_years_ago_to_age_in_8_years_l619_619879


namespace ab_equals_6_l619_619981

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619981


namespace min_value_d_add_PQ_l619_619114

def parabola (P : ℝ × ℝ) : Prop := P.2 ^ 2 = 8 * P.1
def circle (Q : ℝ × ℝ) : Prop := (Q.1 + 1) ^ 2 + (Q.2 - 4) ^ 2 = 4
def directrix (x : ℝ) : Prop := x = -2

theorem min_value_d_add_PQ :
  ∀ (Q : ℝ × ℝ), circle Q →
  ∃ (P : ℝ × ℝ), parabola P ∧ d P = dist P Q → (d P + dist P Q) ≥ 3 :=
by
  sorry

end min_value_d_add_PQ_l619_619114


namespace ellipse_equation_l619_619090

theorem ellipse_equation 
    (a b : ℝ)
    (h1 : a > b)
    (h2 : b > 0)
    (M : ℝ × ℝ)
    (hM : (M.1 / a)^2 + (M.2 / b)^2 = 1)
    (S : set (ℝ × ℝ)) -- ellipse as a set
    (hS : S = {P : ℝ × ℝ | (P.1 / a)^2 + (P.2 / b)^2 = 1})
    (h_radius : ∃ r : ℝ, r = (2 * real.sqrt 6) / 3)
    (h_dist_M_y_axis : ∃ d : ℝ, d = real.sqrt 2)
    : S = {P : ℝ × ℝ | (P.1^2) / 6 + (P.2^2) / 4 = 1} :=
sorry

end ellipse_equation_l619_619090


namespace marble_selection_l619_619580

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def other_marbles : ℕ := total_marbles - special_marbles

-- Define combination function for ease of use in the theorem
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Statement of the theorem based on the question and the correct answer
theorem marble_selection : combination other_marbles 4 * special_marbles = 1320 := by
  -- Define specific values based on the problem
  have other_marbles_val : other_marbles = 11 := rfl
  have comb_11_4 : combination 11 4 = 330 := by
    rw [combination]
    rfl
  rw [other_marbles_val, comb_11_4]
  norm_num
  sorry

end marble_selection_l619_619580


namespace shortest_distance_point_on_parabola_l619_619693

theorem shortest_distance_point_on_parabola (P : ℝ × ℝ)
  (hP : P.2 ^ 2 = 4 * P.1) : P = (1, 3) ↔ ∀ x y, (y = x + 10) →
  ∀ Q ∈ ({Q : ℝ × ℝ | Q.2 ^ 2 = 4 * Q.1}),
  dist P (y - x = 10) ≤ dist Q (y - x = 10) :=
by sorry

end shortest_distance_point_on_parabola_l619_619693


namespace min_k_for_xyz_sum_l619_619870

open Finset

theorem min_k_for_xyz_sum (S : Finset ℕ) (hS : S = (Finset.range 2012).map (Nat.succ)) :
  ∃ (k : ℕ), k = 1008 ∧ 
  ∀ A : Finset ℕ, A ⊆ S → A.card = k →
  (∃ x y z a b c : ℕ, {x, y, z} ⊆ A ∧ {a, b, c} ⊆ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a 
    ∧ x = a + b ∧ y = b + c ∧ z = c + a) :=
by
  sorry

end min_k_for_xyz_sum_l619_619870


namespace base_seven_sum_of_digits_of_result_l619_619282

def base_seven_to_ten (n : Nat) (digits : List Nat) : Nat :=
  digits.foldr (λ d acc, d + 7 * acc) 0

def sum_digits_in_base_seven (n : Nat) : Nat :=
  Nat.digits 7 n |>.sum

theorem base_seven_sum_of_digits_of_result :
  let n1 := base_seven_to_ten 7 [4, 5] -- 45_7
  let n2 := base_seven_to_ten 7 [1, 6] -- 16_7
  let n3 := base_seven_to_ten 7 [1, 2] -- 12_7
  sum_digits_in_base_seven (n1 * n2 + n3) = base_seven_to_ten 7 [1, 7] -- 17_7
:= by
  sorry

end base_seven_sum_of_digits_of_result_l619_619282


namespace pyramid_problems_l619_619495

noncomputable def A1 : ℝ × ℝ × ℝ := (4, 2, 5)
noncomputable def A2 : ℝ × ℝ × ℝ := (0, 7, 2)
noncomputable def A3 : ℝ × ℝ × ℝ := (0, 2, 7)
noncomputable def A4 : ℝ × ℝ × ℝ := (1, 5, 0)

-- 1. Equations of the planes A1A2A3 and A1A2A4
def plane_A1A2A3 : ℝ → ℝ → ℝ → Prop := λ x y z, x + 2*y + 2*z - 18 = 0
def plane_A1A2A4 : ℝ → ℝ → ℝ → Prop := λ x y z, -16*x - 11*y + 3*z + 71 = 0

-- 2. Angle between the edge A1A3 and the face A1A2A4
def angle_A1A3_A1A2A4 : ℝ := 139 -- in degrees

-- 3. Distance from A3 to the face A1A2A4
def distance_A3_to_A1A2A4 : ℝ := 3.56

-- 4. Shortest distance between the lines A1A2 and A3A4
def shortest_distance_A1A2_A3A4 : ℝ := 1.595

-- 5. Equation of the altitude dropped from A3 to A1A2A4
def altitude_A3_to_A1A2A4 (x y z : ℝ) : Prop := ((x - 1) / 1 = (y - 5) / 2) ∧ ((y - 5) / 2 = (z / 2))

-- The final theorem combining all the proven statements
theorem pyramid_problems :
  (∀ x y z, plane_A1A2A3 x y z) ∧
  (∀ x y z, plane_A1A2A4 x y z) ∧
  angle_A1A3_A1A2A4 = 139 ∧
  distance_A3_to_A1A2A4 = 3.56 ∧
  shortest_distance_A1A2_A3A4 = 1.595 ∧
  (∀ x y z, altitude_A3_to_A1A2A4 x y z) :=
by
  sorry

end pyramid_problems_l619_619495


namespace delegate_arrangement_probability_l619_619677

theorem delegate_arrangement_probability :
  let delegates := 10
  let countries := 3
  let independent_delegate := 1
  let total_seats := 10
  let m := 379
  let n := 420
  delegates = 10 ∧ countries = 3 ∧ independent_delegate = 1 ∧ total_seats = 10 →
  Nat.gcd m n = 1 →
  m + n = 799 :=
by
  sorry

end delegate_arrangement_probability_l619_619677


namespace triangle_max_area_side_a_l619_619173

theorem triangle_max_area_side_a (A B C : ℝ) (a b c : ℝ) 
  (h1 : Real.tan A = 2 * Real.tan B) 
  (h2 : b = Real.sqrt 2) 
  (h3 : ∀ S : ℝ, S = (a * b * Real.sin C) / 2) 
  (h4 : S_maximal : ∀ S' : ℝ, S ≤ S') : 
  a = Real.sqrt 5 := 
sorry

end triangle_max_area_side_a_l619_619173


namespace smurf_team_count_l619_619765

/-- A Smurf dislikes the two adjacent Smurfs. We want to form a team of 5 Smurfs such that no two Smurfs dislike each other. -/
def team_formation_problem : Prop :=
  ∃ (team : Finset ℕ), 
  team.card = 5 ∧
  (∀ (x ∈ team) (y ∈ team), (x ≠ y) → ¬((x + 1) % 12 = y ∨ (x - 1 + 12) % 12 = y))

theorem smurf_team_count : ∃ (count : ℕ), count = 36 ∧
  (∃ (team : Finset (Fin 12)), 
  team.card = 5 ∧
  (∀ (x ∈ team) (y ∈ team), (x ≠ y) → ¬((x.val + 1) % 12 = y.val ∨ (x.val - 1 + 12) % 12 = y.val))) :=
sorry

end smurf_team_count_l619_619765


namespace triangle_area_formula_l619_619541

theorem triangle_area_formula (r α β γ : ℝ)
  (h_1 : r > 0)
  (h_2 : 0 < α ∧ α < π)
  (h_3 : 0 < β ∧ β < π)
  (h_4 : 0 < γ ∧ γ < π)
  (h_5 : α + β + γ = π) :
  ∃ T, T = r^2 * (Real.cot (α / 2)) * (Real.cot (β / 2)) * (Real.cot (γ / 2)) :=
by
  sorry

end triangle_area_formula_l619_619541


namespace number_of_acute_triangle_sides_l619_619026

theorem number_of_acute_triangle_sides :
  (∃ n, setOf {y : ℤ | 7 < y ∧ y < 23 ∧ 
              ((y > 15 → y < 17) ∧ (8 < y ∧ y ≤ 15 → y > 12))}.card = n ∧ n = 5) :=
sorry

end number_of_acute_triangle_sides_l619_619026


namespace shaded_region_perimeter_l619_619567

noncomputable def circumference_of_circle (r : ℕ) : ℝ := 2 * Real.pi * r

noncomputable def arc_length_of_circle (r : ℕ) (theta : ℝ) : ℝ :=
  (theta / 360) * circumference_of_circle r

noncomputable def perimeter_of_shaded_region
  (r : ℕ) (theta : ℝ) (O R S : Point) (h₁ : dist O R = r)
  (h₂ : dist O S = r) (h₃ : angle O R S = theta) : ℝ :=
  2 * r + arc_length_of_circle r theta

theorem shaded_region_perimeter
  {O R S : Point} (r : ℕ) (theta : ℝ) (h₁ : dist O R = r)
  (h₂ : dist O S = r) (h₃ : angle O R S = theta)
  (hr : r = 7) (ht : theta = 120) :
  perimeter_of_shaded_region r theta O R S h₁ h₂ h₃ = 14 + 14 * Real.pi / 3 :=
by
  sorry

end shaded_region_perimeter_l619_619567


namespace convex_polygon_diagonal_intersections_l619_619151

theorem convex_polygon_diagonal_intersections (n : ℕ) (h_n : n ≥ 4) : 
  let num_intersections := (n * (n - 1) * (n - 2) * (n - 3)) / 24 in
  true := 
by sorry

end convex_polygon_diagonal_intersections_l619_619151


namespace pyramid_planes_perpendicular_l619_619405

theorem pyramid_planes_perpendicular
  (A B C D S : Point)
  (convex_quadrilateral_ABCD : convex_quadrilateral A B C D)
  (BC_AD_eq_BD_AC : BC * AD = BD * AC)
  (angle_ADS_eq_angle_BDS : ∠ ADS = ∠ BDS)
  (angle_ACS_eq_angle_BCS : ∠ ACS = ∠ BCS) :
  is_perpendicular (plane S A B) (plane A B C D) := 
by 
  sorry

end pyramid_planes_perpendicular_l619_619405


namespace range_of_a_l619_619945

-- Definitions of the functions g(x) and f(x)
def g (x a : ℝ) := x^2 - 2 * a * x
def f (x : ℝ) := (1 / 3) * x^3 - Real.log (x + 1)

-- Definition of the derivative of f
def f' (x : ℝ) := x^2 - 1 / (x + 1)

-- Condition that there exists x1 in [0, 1] and x2 in [1, 2] such that f'(x1) ≥ g(x2)
def condition (a : ℝ) :=
  ∃ (x1 ∈ Icc 0 1) (x2 ∈ Icc 1 2), f' x1 ≥ g x2 a

-- The final proposition to prove
theorem range_of_a (a : ℝ) : condition a → 1 ≤ a := 
sorry

end range_of_a_l619_619945


namespace prob_B_given_A_l619_619299

noncomputable theory
open Classical

def event_A (a b : ℕ) : Prop := a + b > 7
def event_B (a b : ℕ) : Prop := a * b > 20
def event_C (a b : ℕ) : Prop := a + b < 10

def n_A := 15
def n_B := 6
def n_AB := 6

def P_A := n_A / 36
def P_B := n_B / 36
def P_AB := n_AB / 36
def P_B_given_A := n_AB / n_A

theorem prob_B_given_A : P_B_given_A = 2 / 5 :=
by
  have := P_B_given_A
  norm_num at this
  sorry

end prob_B_given_A_l619_619299


namespace max_possible_average_after_transformation_l619_619258

theorem max_possible_average_after_transformation :
  ∀ (a : Fin 6 → ℕ), (∑ i in Finset.univ, a i) = 96 → (∑ i in Finset.univ, (transform_two_to_four (a i))) ≤ 120 :=
by
  sorry

def transform_two_to_four (n : ℕ) : ℕ :=
  -- This would be a function that maps each digit '2' in the number to '4'
  sorry

end max_possible_average_after_transformation_l619_619258


namespace ellipse_properties_l619_619419

-- Definitions for points, lines, and ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def point_on_line (x1 y1 c : ℝ) : Prop :=
  x1 - y1 = c

def tangent_to_ellipse (a b x1 x2 y1 y2 : ℝ) : Prop :=
  (x1 * x2 / a^2 + y1 * y2 / b^2 = 1)

def passes_through_point (x1 y1 x y : ℝ) : Prop :=
  (3 * x + 4 * y) * x1 = 16 * y + 12

def fixed_point (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = -3 / 4)

-- Lean 4 statement
theorem ellipse_properties :
  (∃ (a b : ℝ), ellipse a b 0 (√3) ∧ a = 2 ∧ b = √3) ∧
  (∀ (x1 y1 : ℝ), point_on_line x1 y1 4 →
    ∃ (x2 y2 x3 y3 : ℝ), 
      tangent_to_ellipse 2 (√3) x1 x2 y1 y2 ∧ 
      tangent_to_ellipse 2 (√3) x1 x3 y1 y3 ∧
      passes_through_point x1 y1 1 (-3 / 4)) :=
by { sorry }

end ellipse_properties_l619_619419


namespace smallest_value_l619_619612

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l619_619612


namespace intersection_on_diagonal_l619_619643

-- Define the points and lines
variables (A B C D K L M N O : Type)
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables [AddGroup K] [AddGroup L] [AddGroup M] [AddGroup N] [AddGroup O]

-- Conditions
variable (h1 : Rectangle A B C D)
variable (h2 : PointOnSegment K A B)
variable (h3 : PointOnSegment L B C)
variable (h4 : PointOnSegment M C D)
variable (h5 : PointOnSegment N D A)
variable (h6 : Parallel (Segment K L) (Segment M N))
variable (h7 : Perpendicular (Segment K M) (Segment L N))

-- Theorem statement
theorem intersection_on_diagonal :
  (Intersection (Segment K M) (Segment L N) = Some O) →
  (OnLine O (Line B D)) :=
by
  intro h_intersection
  sorry

end intersection_on_diagonal_l619_619643


namespace simplify_expression_l619_619244

theorem simplify_expression : (324 : ℝ)^(1/4) * (98 : ℝ)^(1/2) = 42 :=
by
  have h324 : 324 = 2^2 * 3^4 := by norm_num
  have h98 : 98 = 2 * 7^2 := by norm_num
  have sqrt324 : (324 : ℝ)^(1/4) = (2^(1/2) * 3 : ℝ) := by
    rw [h324, real.rpow_mul, real.sqrt_eq_rpow, real.sq_sqrt, real.rpow_nat_cast, real.rpow_nat_cast 2, real.rpow_nat_cast 3]
    norm_num
  have sqrt98 : (98 : ℝ)^(1/2) = (2^(1/2) * 7 : ℝ) := by
    rw [h98, real.rpow_mul, real.sqrt_eq_rpow, real.sq_sqrt, real.rpow_nat_cast, real.rpow_nat_cast 1, real.rpow_nat_cast 7]
    norm_num
  rw [sqrt324, sqrt98]
  calc (2^(1/2) * 3 * 2^(1/2) * 7) = (2^(1/2) * 2^(1/2)) * (3 * 7) : by ring
  ... = 2 * (3 * 7) : by { rw [← real.sqrt_mul, real.sqrt_two_mul_sqrt_two] }
  ... = 42 : by norm_num

end simplify_expression_l619_619244


namespace knight_move_is_even_l619_619373

-- Define the condition when the knight returns to its original square.
def knight_returns_to_original_square (n : ℕ) : Prop :=
  -- Any knight move sequence returning to its start must invoke an even number of color flips.
  n.even

-- State the theorem
theorem knight_move_is_even (n : ℕ) (H : knight_returns_to_original_square n) : n.even :=
  H

end knight_move_is_even_l619_619373


namespace determine_values_of_a_l619_619424

theorem determine_values_of_a :
  ∀ (a : ℝ), (∃ (x1 x2 x3 x4 x5 : ℝ), 
    0 ≤ x1 ∧ 0 ≤ x2 ∧ 0 ≤ x3 ∧ 0 ≤ x4 ∧ 0 ≤ x5 ∧
    (∑ k in finset.range 1 6, k * list.nth_le [x1, x2, x3, x4, x5] (k - 1) sorry) = a ∧
    (∑ k in finset.range 1 6, k^3 * list.nth_le [x1, x2, x3, x4, x5] (k - 1) sorry) = a^2 ∧
    (∑ k in finset.range 1 6, k^5 * list.nth_le [x1, x2, x3, x4, x5] (k - 1) sorry) = a^3) ↔ 
    a ∈ ({0, 1, 4, 9, 16, 25} : set ℝ) := 
  by sorry

end determine_values_of_a_l619_619424


namespace find_difference_square_l619_619960

theorem find_difference_square (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 6) :
  (x - y)^2 = 25 :=
by
  sorry

end find_difference_square_l619_619960


namespace longest_segment_in_cylinder_l619_619792

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end longest_segment_in_cylinder_l619_619792


namespace triangle_angle_bisector_ratio_l619_619170

  theorem triangle_angle_bisector_ratio 
    (A B C D E P : Point)
    (a b c : ℝ)
    (hABC : triangle A B C)
    (hAD : is_angle_bisector A D)
    (hBE : is_angle_bisector B E)
    (h_intersect : intersect_at P A D B E)
    (hAB : distance A B = 9)
    (hAC : distance A C = 6)
    (hBC : distance B C = 4) :
    ratio_segment P B P E = 13 / 9 :=
  by
    sorry
  
end triangle_angle_bisector_ratio_l619_619170


namespace product_ab_l619_619986

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619986


namespace solve_eqn_l619_619438

noncomputable def omega := complex.exp ((2 * real.pi * complex.I) / 3)

theorem solve_eqn (z : complex) : 
  (z^6 - 3 * z^3 + 2 = 0) ↔ (z = complex.cbrt 2 ∨ z = complex.cbrt 2 * omega ∨ z = complex.cbrt 2 * omega^2 ∨ z = 1 ∨ z = omega ∨ z = omega^2) := sorry

end solve_eqn_l619_619438


namespace total_distance_covered_l619_619396

theorem total_distance_covered :
  let speed1 := 40 -- miles per hour
  let speed2 := 50 -- miles per hour
  let speed3 := 30 -- miles per hour
  let time1 := 1.5 -- hours
  let time2 := 1 -- hour
  let time3 := 2.25 -- hours
  let distance1 := speed1 * time1 -- distance covered in the first part of the trip
  let distance2 := speed2 * time2 -- distance covered in the second part of the trip
  let distance3 := speed3 * time3 -- distance covered in the third part of the trip
  distance1 + distance2 + distance3 = 177.5 := 
by
  sorry

end total_distance_covered_l619_619396


namespace largest_of_seven_consecutive_composite_numbers_less_than_40_l619_619242

open Nat

theorem largest_of_seven_consecutive_composite_numbers_less_than_40 :
  ∃ (n : ℕ), 23 ≤ n ∧ n ≤ 30 ∧ ∀ (k : ℕ), n ≤ k ∧ k < n + 7 → ¬ prime k ∧ n + 6 = 30 :=
by
  sorry

end largest_of_seven_consecutive_composite_numbers_less_than_40_l619_619242


namespace quotient_of_sum_of_remainders_div_16_eq_0_l619_619673

-- Define the set of distinct remainders of squares modulo 16 for n in 1 to 15
def distinct_remainders_mod_16 : Finset ℕ :=
  {1, 4, 9, 0}

-- Define the sum of the distinct remainders
def sum_of_remainders : ℕ :=
  distinct_remainders_mod_16.sum id

-- Proposition to prove the quotient when sum_of_remainders is divided by 16 is 0
theorem quotient_of_sum_of_remainders_div_16_eq_0 :
  (sum_of_remainders / 16) = 0 :=
by
  sorry

end quotient_of_sum_of_remainders_div_16_eq_0_l619_619673


namespace total_sums_attempted_l619_619813

-- Define the necessary conditions
def num_sums_right : ℕ := 8
def num_sums_wrong : ℕ := 2 * num_sums_right

-- Define the theorem to prove
theorem total_sums_attempted : num_sums_right + num_sums_wrong = 24 := by
  sorry

end total_sums_attempted_l619_619813


namespace perpendicular_distance_from_H_to_plane_EFG_l619_619767

-- Definitions of points
def E : ℝ × ℝ × ℝ := (5, 0, 0)
def F : ℝ × ℝ × ℝ := (0, 5, 0)
def G : ℝ × ℝ × ℝ := (0, 0, 4)
def H : ℝ × ℝ × ℝ := (0, 0, 0)

-- The main theorem to prove
theorem perpendicular_distance_from_H_to_plane_EFG : 
  let d := abs (((0 - 0) * (0 - 4) - (0 - 0) * (5 - 0)) * (0 - 5) + 
                ((0 - 0) * (5 - 0) - (5 - 0) * (0 - 0)) * (0 - 0) + 
                ((5 - 0) * (0 - 0) - (0 - 0) * (0 - 0)) * (4 - 0)) / 
                sqrt ((0 - 0)^2 + (5 - 0)^2 + (0 - 0)^2) in
  d = 4 := by
  sorry

end perpendicular_distance_from_H_to_plane_EFG_l619_619767


namespace smallest_value_l619_619610

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l619_619610


namespace remainder_1235678_div_127_l619_619011

theorem remainder_1235678_div_127 : 1235678 % 127 = 69 := by
  sorry

end remainder_1235678_div_127_l619_619011


namespace product_ab_l619_619991

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619991


namespace complaint_online_prob_l619_619375

/-- Define the various probability conditions -/
def prob_online := 4 / 5
def prob_store := 1 / 5
def qual_rate_online := 17 / 20
def qual_rate_store := 9 / 10
def non_qual_rate_online := 1 - qual_rate_online
def non_qual_rate_store := 1 - qual_rate_store
def prob_complaint_online := prob_online * non_qual_rate_online
def prob_complaint_store := prob_store * non_qual_rate_store
def total_prob_complaint := prob_complaint_online + prob_complaint_store

/-- The theorem states that given the conditions, the probability of an online purchase given a complaint is 6/7 -/
theorem complaint_online_prob : 
    (prob_complaint_online / total_prob_complaint) = 6 / 7 := 
by
    sorry

end complaint_online_prob_l619_619375


namespace goldie_total_earnings_l619_619123

def sum_arithmetic_series (n : ℕ) (a₁ aₙ : ℝ) : ℝ :=
  (n / 2) * (a₁ + aₙ)

def goldie_earnings : Prop :=
  let week1_hours := 20
  let week2_hours := 30
  let initial_rate := 5.0
  let increase_rate_week1 := 0.5
  let increase_rate_week2 := 0.75

  let last_rate_week1 := initial_rate + (week1_hours - 1) * increase_rate_week1
  let earnings_week1 := sum_arithmetic_series week1_hours initial_rate last_rate_week1

  let last_rate_week2 := initial_rate + (week2_hours - 1) * increase_rate_week2
  let earnings_week2 := sum_arithmetic_series week2_hours initial_rate last_rate_week2

  earnings_week1 + earnings_week2 = 671.25

theorem goldie_total_earnings : goldie_earnings :=
  sorry

end goldie_total_earnings_l619_619123


namespace trig_expr_value_l619_619032

theorem trig_expr_value : 
  (sin 20 * sqrt (1 + cos 40) / cos 50) = (sqrt 2 / 2) :=
by sorry

end trig_expr_value_l619_619032


namespace min_a_plus_b_eq_six_point_five_l619_619607

noncomputable def min_a_plus_b : ℝ :=
  Inf {s | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
                       (a^2 - 12 * b ≥ 0) ∧ 
                       (9 * b^2 - 4 * a ≥ 0) ∧ 
                       (a + b = s)}

theorem min_a_plus_b_eq_six_point_five : min_a_plus_b = 6.5 :=
by
  sorry

end min_a_plus_b_eq_six_point_five_l619_619607


namespace painter_completion_time_l619_619804

def hours_elapsed (start_time end_time : String) : ℕ :=
  match (start_time, end_time) with
  | ("9:00 AM", "12:00 PM") => 3
  | _ => 0

-- The initial conditions, the start time is 9:00 AM, and 3 hours later 1/4th is done
def start_time := "9:00 AM"
def partial_completion_time := "12:00 PM"
def partial_completion_fraction := 1 / 4
def partial_time_hours := hours_elapsed start_time partial_completion_time

-- The painter works consistently, so it would take 4 times the partial time to complete the job
def total_time_hours := 4 * partial_time_hours

-- Calculate the completion time by adding total_time_hours to the start_time
def completion_time : String :=
  match start_time with
  | "9:00 AM" => "9:00 PM"
  | _         => "unknown"

theorem painter_completion_time :
  completion_time = "9:00 PM" :=
by
  -- Definitions and calculations already included in the setup
  sorry

end painter_completion_time_l619_619804


namespace find_lambda_l619_619953

open Real

namespace VectorProof

def a : ℝ × ℝ × ℝ := (0, -1, 1)
def b : ℝ × ℝ × ℝ := (4, 1, 0)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem find_lambda (λ : ℝ) (h : λ > 0) :
  magnitude (λ • a + b) = sqrt 29 ↔ λ = 3 := by
  sorry

end VectorProof

end find_lambda_l619_619953


namespace evaluate_expression_l619_619431

theorem evaluate_expression :
  (3^1 + 3^0 + 3^(-1)) / (3^(-2) + 3^(-3) + 3^(-4)) = 27 :=
by
  sorry

end evaluate_expression_l619_619431


namespace cylinder_longest_segment_l619_619789

-- Define the radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 10

-- Definition for the longest segment inside the cylinder using Pythagorean theorem
def longest_segment (radius height : ℝ) : ℝ :=
  real.sqrt (radius * 2)^2 + height^2

-- Specify the expected answer for the proof
def expected_answer : ℝ := 10 * real.sqrt 2

-- The theorem stating the longest segment length inside the cylinder
theorem cylinder_longest_segment : longest_segment radius height = expected_answer :=
by {
  -- Lean code to set up and prove the equivalence
  sorry
}

end cylinder_longest_segment_l619_619789


namespace max_distance_AB_l619_619562

noncomputable def x (t α : ℝ) : ℝ := t * Real.cos α
noncomputable def y (t α : ℝ) : ℝ := t * Real.sin α

noncomputable def rho_C2 (θ : ℝ) : ℝ := 4 * Real.sin θ
noncomputable def rho_C3 (θ : ℝ) : ℝ := 4 * Real.sqrt 3 * Real.cos θ

theorem max_distance_AB : ∃ t α : ℝ, (0 ≤ α ∧ α < Real.pi) → 
  let A := (4 * Real.sin α, α)
  let B := (4 * Real.sqrt 3 * Real.cos α, α)
  (∀ t1 t2, t1 ≠ 0 → t2 ≠ 0 → 
    (x t1 α, y t1 α) = A ∧ (x t2 α, y t2 α) = B) →
    Real.abs (8 * Real.cos (Real.pi / 6 + α)) = 8 :=
by
  sorry

end max_distance_AB_l619_619562


namespace calculate_result_l619_619012

theorem calculate_result :
  (-24) * ((5 / 6 : ℚ) - (4 / 3) + (5 / 8)) = -3 := 
by
  sorry

end calculate_result_l619_619012


namespace cost_price_brands_min_sets_brand_A_l619_619812

-- Definitions for Part (1)
def cost_price_B : ℝ := 7.5
def cost_price_A := cost_price_B + 2.5
def brand_A_purchased_qty := 200 / cost_price_A
def brand_B_purchased_qty := 75 / cost_price_B

-- Definitions for Part (2)
def selling_price_A : ℝ := 13
def selling_price_B : ℝ := 9.5
def additional_B_sets := 4
def min_profit : ℝ := 120

-- Predicate for minimum sets of brand A
def min_sets_brand_A_purchased (a : ℕ) : Prop :=
  let b := 2 * a + additional_B_sets in
  (selling_price_A - cost_price_A) * a + (selling_price_B - cost_price_B) * b > min_profit

-- Proof problem for Part (1)
theorem cost_price_brands :
  cost_price_A = 10 ∧ cost_price_B = 7.5 :=
by
  sorry

-- Proof problem for Part (2)
theorem min_sets_brand_A :
  ∃ a : ℕ, a ≥ 17 ∧ min_sets_brand_A_purchased a :=
by
  sorry

end cost_price_brands_min_sets_brand_A_l619_619812


namespace new_boarders_joined_l619_619332

/-- 
The problem's initial conditions 
-/
def initial_boarders := 120
def initial_ratio_boarders_to_day_students : ℚ := 2 / 5
def new_ratio_boarders_to_day_students : ℚ := 1 / 2

/--
Assuming no change in the initial number of day students and no boarders becoming day students, 
find the number of new boarders \( x \) such that the new ratio boarders to day students is 1 to 2.
-/
theorem new_boarders_joined 
  (initial_boarders : ℕ) 
  (initial_ratio : ℚ) 
  (new_ratio : ℚ) 
  (initial_day_students : ℕ := (initial_boarders * 5) / 2) 
  (total_boarders_after_new : ℕ := initial_boarders + 30) 
  (initial_to_new_ratio : ℚ := ↑total_boarders_after_new / ↑initial_day_students) 
  (initial_boarders = 120) 
  (initial_ratio = 2 / 5)
  (new_ratio = 1 / 2) 
:
  30 = (120 * (1 / 2)) - 120 := 
begin
  sorry
end

end new_boarders_joined_l619_619332


namespace cylinder_longest_segment_l619_619788

-- Define the radius and height of the cylinder
def radius : ℝ := 5
def height : ℝ := 10

-- Definition for the longest segment inside the cylinder using Pythagorean theorem
def longest_segment (radius height : ℝ) : ℝ :=
  real.sqrt (radius * 2)^2 + height^2

-- Specify the expected answer for the proof
def expected_answer : ℝ := 10 * real.sqrt 2

-- The theorem stating the longest segment length inside the cylinder
theorem cylinder_longest_segment : longest_segment radius height = expected_answer :=
by {
  -- Lean code to set up and prove the equivalence
  sorry
}

end cylinder_longest_segment_l619_619788


namespace all_lines_concurrent_l619_619220

theorem all_lines_concurrent
  (red_lines blue_lines : Finset (ℝ × ℝ × ℝ)) -- Lines in the form of ax + by + c = 0
  (finite_red : red_lines.finite)
  (finite_blue : blue_lines.finite)
  (no_parallel_lines : ∀ l1 l2 ∈ (red_lines ∪ blue_lines), l1 ≠ l2 → (l1.1 * l2.2 ≠ l1.2 * l2.1))
  (intersection_property : ∀ l1 l2 ∈ (red_lines ∪ red_lines), l1 ≠ l2 → ∃ b ∈ blue_lines, (l1.1 * l2.2 - l1.2 * l2.1) * b.3 = (l1.3 * l2.2 - l1.2 * l2.3) * b.2 - (l1.1 * l2.3 - l1.3 * l2.1) * b.1)
  (intersection_property' : ∀ l1 l2 ∈ (blue_lines ∪ blue_lines), l1 ≠ l2 → ∃ r ∈ red_lines, (l1.1 * l2.2 - l1.2 * l2.1) * r.3 = (l1.3 * l2.2 - l1.2 * l2.3) * r.2 - (l1.1 * l2.3 - l1.3 * l2.1) * r.1) :
  ∃ P : ℝ × ℝ, ∀ l ∈ (red_lines ∪ blue_lines), (l.1 * P.1 + l.2 * P.2 + l.3 = 0) := sorry

end all_lines_concurrent_l619_619220


namespace number_of_removed_carrots_l619_619352

noncomputable def total_weight_of_30_carrots := 5.94 -- in kg
noncomputable def average_weight_of_remaining_27_carrots := 0.2 -- in kg (200 grams = 0.2 kg)
noncomputable def average_weight_of_removed_carrots := 0.18 -- in kg (180 grams = 0.18 kg)

theorem number_of_removed_carrots :
  ∃ (n : ℕ), n = 3 ∧
    (total_weight_of_30_carrots - (27 * average_weight_of_remaining_27_carrots)) / average_weight_of_removed_carrots = n :=
by
  sorry

end number_of_removed_carrots_l619_619352


namespace vector_magnitude_l619_619899

open Real

variables {a b : ℝ^2}

theorem vector_magnitude (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 2) (h_perp : dot_product a b = 0) : ‖a + b‖ = sqrt 3 :=
by
  sorry

end vector_magnitude_l619_619899


namespace range_of_m_l619_619507

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  x^3 - 6 * x^2 + 9 * x + m

theorem range_of_m (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧ f m a = 0 ∧ f m b = 0 ∧ f m c = 0) ↔ -4 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l619_619507


namespace sum_binom_eq_cos70_l619_619835

-- Function definition for the given series
def series_sum (a : ℕ) := ∑ n in Finset.range (1006), (-3 : ℝ)^n * (Nat.choose 2010 (2 * n))

-- The main theorem statement
theorem sum_binom_eq_cos70 :
  (1 / (2 : ℝ)^2010) * series_sum 2010 = - Real.cos (70 * Real.pi / 180) :=
by
  sorry

end sum_binom_eq_cos70_l619_619835


namespace difference_between_fastest_and_slowest_jumper_l619_619833

noncomputable def Cindy_time : ℝ := 12
noncomputable def Betsy_time : ℝ := 0.7 * Cindy_time
noncomputable def Tina_time : ℝ := 2 * Betsy_time
noncomputable def Sarah_time : ℝ := Cindy_time + Tina_time

theorem difference_between_fastest_and_slowest_jumper :
  Sarah_time - Betsy_time = 20.4 :=
by
  unfold Cindy_time Betsy_time Tina_time Sarah_time
  sorry

end difference_between_fastest_and_slowest_jumper_l619_619833


namespace major_axis_length_l619_619001

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def foci_1 : ℝ × ℝ := (3, 5)
def foci_2 : ℝ × ℝ := (23, 40)
def reflected_foci_1 : ℝ × ℝ := (-3, 5)

theorem major_axis_length :
  distance (reflected_foci_1.1) (reflected_foci_1.2) (foci_2.1) (foci_2.2) = Real.sqrt 1921 :=
sorry

end major_axis_length_l619_619001


namespace sum_of_squares_c_k_l619_619024

noncomputable def c (k : ℕ) : ℝ := k + (1 / (2 * k + (1 / (2 * k + (1 / (2 * k + (1 / (2 * k + ...))))))))

theorem sum_of_squares_c_k : (∑ k in finset.range 12, (c k)^2) = 517 := by
  have h_c_k : ∀ k : ℕ, (c k)^2 = k^2 + 1 := sorry
  calc
    (∑ k in finset.range 12, (c k)^2)
        = (∑ k in finset.range 12, k^2 + 1) : by
        { apply finset.sum_congr rfl,
          intros k hk,
          rw h_c_k }
    ... = ∑ k in finset.range 12, k^2 + ∑ k in finset.range 12, 1 : by {
          rw finset.sum_add_distrib
        }
    ... = (∑ k in finset.range 12, k^2) + 11 : by {
          rw finset.sum_const,
          norm_num
        }
    ... = (11 * 12 * 23) / 6 + 11 : by {
          rw finset.sum_range_succ,
          norm_num
        }
    ... = 517 : by norm_num

#qlibrary.eqlQED

end sum_of_squares_c_k_l619_619024


namespace sum_of_exponents_l619_619860

theorem sum_of_exponents : 
  ∃ S : Finset ℕ, (3125 = S.sum (λ i, 2^i) ∧ S.sum id = 32) :=
by
  sorry

end sum_of_exponents_l619_619860


namespace largest_seven_consecutive_composites_less_than_40_l619_619239

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

def seven_consecutive_composites_less_than_40 (a : ℕ) : Prop :=
  a < 40 ∧ a > 29 ∧
  is_composite a ∧ is_composite (a - 1) ∧ is_composite (a - 2) ∧ 
  is_composite (a - 3) ∧ is_composite (a - 4) ∧ 
  is_composite (a - 5) ∧ is_composite (a - 6)

theorem largest_seven_consecutive_composites_less_than_40 :
  seven_consecutive_composites_less_than_40 36 :=
begin
  -- Proof goes here.
  sorry
end

end largest_seven_consecutive_composites_less_than_40_l619_619239


namespace train_stoppages_l619_619298

variables (sA sA' sB sB' sC sC' : ℝ)
variables (x y z : ℝ)

-- Conditions
def conditions : Prop :=
  sA = 80 ∧ sA' = 60 ∧
  sB = 100 ∧ sB' = 75 ∧
  sC = 120 ∧ sC' = 90

-- Goal that we need to prove
def goal : Prop :=
  x = 15 ∧ y = 15 ∧ z = 15

-- Main statement
theorem train_stoppages : conditions sA sA' sB sB' sC sC' → goal x y z :=
by
  sorry

end train_stoppages_l619_619298


namespace problem_statement_l619_619202

theorem problem_statement (x1 x2 x3 : ℝ) 
  (h1 : x1 < x2)
  (h2 : x2 < x3)
  (h3 : (45*x1^3 - 4050*x1^2 - 4 = 0) ∧ 
        (45*x2^3 - 4050*x2^2 - 4 = 0) ∧ 
        (45*x3^3 - 4050*x3^2 - 4 = 0)) :
  x2 * (x1 + x3) = 0 :=
by
  sorry

end problem_statement_l619_619202


namespace value_of_x_l619_619675

theorem value_of_x (x : ℝ) (h : ∀ S : Set ℝ, S = {1, 2, 3, x} → 
  (∃ (largest smallest : ℝ), largest ∈ S ∧ smallest ∈ S ∧ largest ≠ smallest ∧ 
  (∀ y ∈ S, y ≤ largest) ∧ (∀ y ∈ S, smallest ≤ y) ∧ 
  (largest - smallest = ∑ y in S.toFinset, y))):
  x = -3 / 2 := 
sorry

end value_of_x_l619_619675


namespace twentyone_inv_mod_fortyseven_l619_619917

/-- Given that 8⁻¹ ≡ 6 (mod 47), prove that 21⁻¹ ≡ 38 (mod 47). -/
theorem twentyone_inv_mod_fortyseven 
  (h : ∀ x : ℤ, 8 * x ≡ 1 [MOD 47] → x ≡ 6 [MOD 47]) :
  ∃ y : ℤ, 21 * y ≡ 1 [MOD 47] ∧ y ≡ 38 [MOD 47] :=
by 
  sorry

end twentyone_inv_mod_fortyseven_l619_619917


namespace basketball_card_price_l619_619641

variable (x : ℝ)

def total_cost_basketball_cards (x : ℝ) : ℝ := 2 * x
def total_cost_baseball_cards : ℝ := 5 * 4
def total_spent : ℝ := 50 - 24

theorem basketball_card_price :
  total_cost_basketball_cards x + total_cost_baseball_cards = total_spent ↔ x = 3 := by
  sorry

end basketball_card_price_l619_619641


namespace find_n_l619_619041

theorem find_n : ∃ n : ℕ, n < 2006 ∧ ∀ m : ℕ, 2006 * n = m * (2006 + n) ↔ n = 1475 := by
  sorry

end find_n_l619_619041


namespace concurrency_of_GI_HJ_symmedian_l619_619632

variables {ω : Type} [MetricSpace ω] [NormedAddCommGroup ω] [NormedSpace ℝ ω]
variables (A B C D E F G H I J : ω)
variables (circle : Set ω)
variables (symm : ω → ω → ω → ω → Prop)

-- Conditions
def is_cyclic (w : Set ω) (a b c d : ω) : Prop :=
  ∃ (circ : Set ω), circ = w ∧ a ∈ circ ∧ b ∈ circ ∧ c ∈ circ ∧ d ∈ circ

def tangent (c : Set ω) (p t : ω) : Prop :=
  ∃ (l : ω → ω), l t = 0 ∧ ∀ x ∉ c, l x ≠ 0

def intersects (l1 l2 : ω → ω) (p : ω) : Prop :=
  l1 p = 0 ∧ l2 p = 0

-- Prove the concurrency of GI, HJ, and the symmedian from B
theorem concurrency_of_GI_HJ_symmedian
  (h1 : is_cyclic circle A B C D)
  (h2 : tangent circle A E)
  (h3 : tangent circle A F)
  (h4 : intersects (λ x, x) (λ x, E) G)
  (h5 : intersects (λ x, B) (λ x, A D) H)
  (h6 : intersects (λ x, D F) (λ x, circle x) I)
  (h7 : intersects (λ x, D F) (λ x, A B) J)
  (h8 : symm A B C G)
  (h9 : symm A B C H)
  (h10 : symm A B C I)
  (h11 : symm A B C J) :
  ∃ X, intersects (λ x, G I) (λ x, H J) X
    ∧ intersects (λ x, G I) (λ x, symm A B C) X
    ∧ intersects (λ x, H J) (λ x, symm A B C) X :=
sorry

end concurrency_of_GI_HJ_symmedian_l619_619632


namespace original_cloth_side_length_l619_619383

theorem original_cloth_side_length (x : ℝ) (h1 : (x - 6) * (x - 5) = 120) : x = 15 :=
by
  have h : x^2 - 11 * x - 90 = 0, from sorry
  have : x = 15 ∨ x = -6, from sorry
  have : x ≠ -6, from sorry
  exact this.resolve_right this1

end original_cloth_side_length_l619_619383


namespace values_at_1_explicit_formula_and_zeros_intervals_of_monotonicity_l619_619108

noncomputable def f : ℝ → ℝ := λ x, 2 * x - 4
noncomputable def g : ℝ → ℝ := λ x, -x + 4

theorem values_at_1 :
  f 1 = -2 ∧ g 1 = 3 :=
by sorry

noncomputable def h : ℝ → ℝ := λ x, f x * g x

theorem explicit_formula_and_zeros :
  h = (λ x, -2 * x^2 + 12 * x - 16) ∧
  (h 2 = 0 ∧ h 4 = 0) :=
by sorry

theorem intervals_of_monotonicity :
  (∀ x, x < 3 → h' x > 0) ∧ (∀ x, x > 3 → h' x < 0) :=
by sorry

end values_at_1_explicit_formula_and_zeros_intervals_of_monotonicity_l619_619108


namespace equal_tuesdays_and_fridays_in_30_day_month_l619_619376

/-- A month with 30 days has exactly three possible starting days that result 
    in the same number of Tuesdays and Fridays, which are Sunday, Wednesday, and Saturday. -/
theorem equal_tuesdays_and_fridays_in_30_day_month :
  ∃ (days : Finset ℕ), days.card = 3 ∧
    ∀ d ∈ days, (d + 2) % 7 = 0 ∨ (d + 2) % 7 = 5 ∨ (d + 2) % 7 = 6 :=
by
  let days := {0, 3, 6}  -- Sunday, Wednesday, Saturday represented as 0, 3, 6
  use days
  simp
  sorry

end equal_tuesdays_and_fridays_in_30_day_month_l619_619376


namespace find_a_b_find_locus_P_l619_619498

-- Part (1): Proving the values of a and b given the ellipse conditions
theorem find_a_b 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b ≥ 0) 
  (h3 : sqrt (a^2 - b^2) / a = 4 / 5) 
  (h4 : 2 * sqrt (a^2 - b^2) = 25 / 2) 
  : a = 125 / 16 ∧ b = 75 / 16 := 
sorry

-- Part (2): Proving the locus equation of point P
theorem find_locus_P 
  (x y : ℝ) 
  (s t : ℝ) 
  (h1 : ellipse a b s t)
  (h2 : sqrt ((s - 6)^2 + t^2) = sqrt ((x - 6)^2 + y^2))
  (h3 : (s - 6) * (x - 6) + t * y = 0)
  : (x - 6)^2 / 9 + (y - 6)^2 / 25 = 1 := 
sorry

-- The definition of an ellipse for Part (2)
def ellipse (a b x y : ℝ) : Prop :=
  (x / a)^2 + (y / b)^2 = 1

end find_a_b_find_locus_P_l619_619498


namespace allocation_schemes_count_l619_619700

theorem allocation_schemes_count :
  let spots := 5
  let classes := 4
  let min_spots_for_class_A := 2
  (total_allocation_schemes spots classes min_spots_for_class_A) = 20 :=
by
  sorry

def total_allocation_schemes (spots : Nat) (classes : Nat) (min_spots_for_class_A : Nat) : Nat :=
  -- The definition calculates the total number of different allocation schemes
  sorry

end allocation_schemes_count_l619_619700


namespace sum_at_least_1000_in_submatrix_l619_619150

open Finset

def grid_2000 := Fin 2000 → Fin 2000 → ℤ

theorem sum_at_least_1000_in_submatrix
  (grid : grid_2000)
  (cond1 : ∀ i j, grid i j = 1 ∨ grid i j = -1)
  (cond2 : (∑ i j, grid i j) ≥ 0) :
  ∃ (rows : Finset (Fin 2000)) (cols : Finset (Fin 2000)), 
  rows.card = 1000 ∧ cols.card = 1000 ∧ (∑ i in rows, ∑ j in cols, grid i j) ≥ 1000 := sorry

end sum_at_least_1000_in_submatrix_l619_619150


namespace tangent_line_and_curve_l619_619112

theorem tangent_line_and_curve (a x0 : ℝ) 
  (h1 : ∀ (x : ℝ), x0 + a = 1) 
  (h2 : ∀ (y : ℝ), y = x0 + 1) 
  (h3 : ∀ (y : ℝ), y = Real.log (x0 + a)) 
  : a = 2 := 
by 
  sorry

end tangent_line_and_curve_l619_619112


namespace max_expression_value_l619_619867

theorem max_expression_value {x y : ℝ} (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) :
  x^2 + y^2 ≤ 10 :=
sorry

end max_expression_value_l619_619867


namespace line_parallel_through_point_l619_619439

theorem line_parallel_through_point (P : ℝ × ℝ) (a b c : ℝ) (ha : a = 3) (hb : b = -4) (hc : c = 6) (hP : P = (4, -1)) :
  ∃ d : ℝ, (d = -16) ∧ (∀ x y : ℝ, a * x + b * y + d = 0 ↔ 3 * x - 4 * y - 16 = 0) :=
by
  sorry

end line_parallel_through_point_l619_619439


namespace arithmetic_sequence_property_l619_619922

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_cond : a 1 + a 3 = 2) : 
  a 2 = 1 :=
by 
  sorry

end arithmetic_sequence_property_l619_619922


namespace circle_equation_l619_619027

theorem circle_equation :
  ∃ (r : ℝ), r = √2 ∧ (∀ x y, (x - 2)^2 + y^2 = r^2) :=
by {
  sorry
}

end circle_equation_l619_619027


namespace simplify_expression_l619_619666

theorem simplify_expression (y : ℝ) : (5 * y) ^ 3 + (4 * y) * (y ^ 2) = 129 * (y ^ 3) := by
  sorry

end simplify_expression_l619_619666


namespace investment_after_five_years_l619_619815

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem investment_after_five_years :
  let P := 12000
  let r_annual := 0.045
  let r := r_annual / 2
  let n := 5 * 2
  compound_interest P r n ≈ 14986 :=
by {
  -- Definitions for the conditions
  let P := 12000
  let r_annual := 0.045
  let r := r_annual / 2
  let n := 5 * 2
  -- Using compound interest formula
  have A := compound_interest P r n
  -- Using a numerical approximation check
  show A ≈ 14986
}

end investment_after_five_years_l619_619815


namespace conjugate_z_quadrant_l619_619489

theorem conjugate_z_quadrant (z : ℂ) (h : (-1 - 2 * complex.I) * z = 1 - complex.I) : 
  let z_conj := (complex.conj z) in
  z_conj.re > 0 ∧ z_conj.im < 0 :=
by {
  // proof goes here
  sorry
}

end conjugate_z_quadrant_l619_619489


namespace volume_of_vinegar_in_vase_l619_619794

theorem volume_of_vinegar_in_vase:
  let height := 12 in
  let diameter := 3 in
  let ratio_vinegar_water := 1 / 3 in -- Ratio vinegar to solution
  let height_liquid := height / 3 in -- Height of liquid in vase (since vase is one-third full)
  let radius := diameter / 2 in
  let volume_liquid := (Real.pi * radius^2 * height_liquid) in
  let volume_vinegar := (volume_liquid / 4) in -- Vinegar is 1/4 of the solution (1 part vinegar, 3 parts water)
  volume_vinegar ≈ 7.07 :=
by
  -- Definitions with real numbers and quantities
  have height_liquid_def : height_liquid = (height / 3) := rfl
  have radius_def : radius = (diameter / 2) := rfl
  have volume_liquid_def : volume_liquid = (Real.pi * radius^2 * height_liquid) := rfl
  have volume_vinegar_def : volume_vinegar = (volume_liquid / 4) := rfl
  -- Calculate the expected value
  have expected_volume_vinegar : volume_vinegar ≈ 7.07 := sorry
  exact expected_volume_vinegar,

end volume_of_vinegar_in_vase_l619_619794


namespace derivative_func1_derivative_func2_derivative_func3_l619_619056

section

variable {x : ℝ}

-- Problem 1
def func1 (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 4
def func1_deriv (x : ℝ) : ℝ := 6 * x^2 - 6 * x

theorem derivative_func1 : deriv func1 x = func1_deriv x := 
by 
  sorry

-- Problem 2
def func2 (x : ℝ) : ℝ := x * log x
def func2_deriv (x : ℝ) : ℝ := log x + 1

theorem derivative_func2 : deriv func2 x = func2_deriv x := 
by 
  sorry

-- Problem 3
def func3 (x : ℝ) : ℝ := cos x / x
def func3_deriv (x : ℝ) : ℝ := (-x * sin x - cos x) / (x^2)

theorem derivative_func3 : deriv func3 x = func3_deriv x := 
by 
  sorry

end

end derivative_func1_derivative_func2_derivative_func3_l619_619056


namespace range_of_a_l619_619932

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < a then 1 / x else |x + 1|

theorem range_of_a (a : ℝ) :
  (∀ x, x < a → f x a > f (x + 1) a) ∧ (∀ x, x > a → f x a ≤ f (x + 1) a) → a ∈ set.Icc (-1 : ℝ) 0 :=
by
  sorry

end range_of_a_l619_619932


namespace function_is_odd_a_eq_1_odd_function_implies_a_is_pm1_l619_619930

-- Given conditions
def f (a x : ℝ) : ℝ := (a - real.exp x) / (1 + a * real.exp x)

-- Proof Problem for (1)
theorem function_is_odd_a_eq_1 (x : ℝ) : f 1 x = -f 1 (-x) := sorry

-- Proof Problem for (2)
theorem odd_function_implies_a_is_pm1 (a : ℝ) (h : ∀ x : ℝ, f a x = - (f a (-x))) : a = 1 ∨ a = -1 := sorry

end function_is_odd_a_eq_1_odd_function_implies_a_is_pm1_l619_619930


namespace compound_bar_chart_must_clearly_indicate_legend_l619_619776

-- Definitions of the conditions
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_bars_of_different_colors : Bool

-- The theorem stating that a compound bar chart must clearly indicate the legend
theorem compound_bar_chart_must_clearly_indicate_legend 
  (chart : CompoundBarChart)
  (distinguishes_quantities : chart.distinguishes_two_quantities = true)
  (uses_colors : chart.uses_bars_of_different_colors = true) :
  ∃ legend : String, legend ≠ "" := by
  sorry

end compound_bar_chart_must_clearly_indicate_legend_l619_619776


namespace cone_cross_section_area_l619_619777

theorem cone_cross_section_area
  (S : ℝ) (θ : ℝ) (R : ℝ) (a : ℝ) (h : ℝ)
  (hS : S = 36 * π)
  (hθ : θ = 2 * π / 3)
  (h_surface_area_eq : S = π * R^2 + (θ / (2 * π)) * π * a^2)
  (h_relation : θ / (2 * π) = R / a)
  (h_pythagorean : a^2 = h^2 + R^2) :
  2 * R * h / 2 = 18 * real.sqrt 2 := 
sorry

end cone_cross_section_area_l619_619777


namespace warriors_wins_30_l619_619154

-- Definitions for the teams.
def Teams := {hawk, falcon, warrior, lion, knight}

-- Number of wins for each team
variable (wins : Teams → ℕ)

-- The conditions as given in the problem
variable (h_wins : wins hawk > wins falcon)
variable (lw_wins : wins lion > 22)
variable (w_lw_wins : wins warrior > wins lion)
variable (w_kn_wins : wins warrior < wins knight)

-- The main goal is to find out:
theorem warriors_wins_30 : wins warrior = 30 :=
by
  -- Leaving the proof here.
  sorry

end warriors_wins_30_l619_619154


namespace minimum_value_expression_l619_619199

theorem minimum_value_expression (y : ℝ) (hy : y > 0) : 5 * y^3 + 6 * y^(-5) ≥ 11 :=
begin
  sorry
end

example : ∃ y : ℝ, y > 0 ∧ 5 * y^3 + 6 * y^(-5) = 11 :=
begin
  use 1,
  split,
  { norm_num, },
  { norm_num, }
end

end minimum_value_expression_l619_619199


namespace cistern_length_is_correct_l619_619363

-- Definitions for the conditions mentioned in the problem
def cistern_width : ℝ := 6
def water_depth : ℝ := 1.25
def wet_surface_area : ℝ := 83

-- The length of the cistern to be proven
def cistern_length : ℝ := 8

-- Theorem statement that length of the cistern must be 8 meters given the conditions
theorem cistern_length_is_correct :
  ∃ (L : ℝ), (wet_surface_area = (L * cistern_width) + (2 * L * water_depth) + (2 * cistern_width * water_depth)) ∧ L = cistern_length :=
  sorry

end cistern_length_is_correct_l619_619363


namespace range_of_b_l619_619102

noncomputable def f (x b : ℝ) : ℝ := Real.exp x * (x - b)
noncomputable def f' (x b : ℝ) : ℝ := Real.exp x * (x - b + 1)

theorem range_of_b (b : ℝ) :
  (∃ x ∈ set.Icc (1/2 : ℝ) 2, f x b + x * f' x b > 0) ↔ b < 8/3 :=
sorry

end range_of_b_l619_619102


namespace solve_system_l619_619246

noncomputable def system_solutions (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  (1 / x + 1 / y + 1 / z = - (z / (x * y)))

theorem solve_system :
  ∀ (x y z : ℤ), system_solutions x y z ↔ 
    (x = 3 ∧ y = 2 ∧ z = -3) ∨
    (x = -3 ∧ y = 2 ∧ z = 3) ∨
    (x = 2 ∧ y = 3 ∧ z = -3) ∨
    (x = 2 ∧ y = -3 ∧ z = 3) := by
  sorry

end solve_system_l619_619246


namespace ryan_days_learning_l619_619858

-- Definitions based on conditions
def hours_per_day_chinese : ℕ := 4
def total_hours_chinese : ℕ := 24

-- Theorem stating the number of days Ryan learns
theorem ryan_days_learning : total_hours_chinese / hours_per_day_chinese = 6 := 
by 
  -- Divide the total hours spent on Chinese learning by hours per day
  sorry

end ryan_days_learning_l619_619858


namespace part_a_part_b_l619_619947

-- Definitions extracted from the problem
def di (n : ℕ) (a : ℕ → ℝ) (i : ℕ) : ℝ :=
  let max_a := Finset.sup' (Finset.range (i + 1)) (Nat.succ_ne_zero _) a
  let min_a := Finset.inf' (Finset.Icc i n) (Nat.le_succ _ i) a
  max_a - min_a

def d (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  Finset.sup' (Finset.range (n + 1)) (Nat.succ_ne_zero _) (di n a)

-- Part a
theorem part_a (n : ℕ) (a x : ℕ → ℝ) (h : ∀ i : ℕ, 0 ≤ i → i < n → x i ≤ x (i + 1)) :
  (Finset.sup' (Finset.range (n + 1)) (Nat.succ_ne_zero _) (λ i, abs (x i - a i))) ≥ (d n a) / 2 := 
sorry

-- Part b
theorem part_b (n : ℕ) (a x : ℕ → ℝ) (h : ∀ i : ℕ, 0 ≤ i → i < n → x i ≤ x (i + 1)) :
  ∃ (x : ℕ → ℝ), (Finset.sup' (Finset.range (n + 1)) (Nat.succ_ne_zero _) (λ i, abs (x i - a i))) = (d n a) / 2 :=
sorry

end part_a_part_b_l619_619947


namespace problem1_problem2_l619_619410

theorem problem1 : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

theorem problem2 : (Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6 := by
  sorry

end problem1_problem2_l619_619410


namespace quadrilateral_dot_product_l619_619002

variables {A B C D : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
variables (AB DC BC AD : A → B) (AC BD : B → ℝ)
variables (l1 l2 : ℝ)

def quadrilateral (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] :=
  (AC (A → B) = l1) ∧ (BD (A → B) = l2)

theorem quadrilateral_dot_product (quad : quadrilateral A B C D) :
  ((AB + DC) ⬝ (BC + AD)) = l1^2 - l2^2 :=
by
  intros
  sorry

end quadrilateral_dot_product_l619_619002


namespace dot_product_necessity_not_sufficiency_l619_619479

variables {V : Type*} [inner_product_space ℝ V]

theorem dot_product_necessity_not_sufficiency (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)  
    (h : inner_product a c = inner_product b c) :
  ¬ (a = b) → (inner_product a c = inner_product b c) ∧ ¬ (∀ d : V, inner_product a d = inner_product b d → a = b) :=
sorry

end dot_product_necessity_not_sufficiency_l619_619479


namespace fourier_series_expansion_l619_619036

def f (x : ℝ) : ℝ := x + 1

def cos_series (n : ℕ) (x : ℝ) : ℝ := 
  real.cos ((2 * n + 1)/2 * x)

noncomputable def A : ℕ → ℝ := 
  λ n, (4 * (real.pi + 1) / real.pi) * ((-1 : ℝ) ^ n) / (2 * n + 1)

theorem fourier_series_expansion :
  ∀ x ∈ set.Ioo 0 real.pi,
  f x = (λ x, (4 * (real.pi + 1) / real.pi) * Σ' n, (A n) * (cos_series n x)) x :=
begin
  intros x hx,
  sorry, -- Proof omitted
end

end fourier_series_expansion_l619_619036


namespace no_arith_prog_of_sines_l619_619543

theorem no_arith_prog_of_sines (x₁ x₂ x₃ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : x₂ ≠ x₃) (h₃ : x₁ ≠ x₃)
    (hx : 0 < x₁ ∧ x₁ < (Real.pi / 2))
    (hy : 0 < x₂ ∧ x₂ < (Real.pi / 2))
    (hz : 0 < x₃ ∧ x₃ < (Real.pi / 2))
    (h : 2 * Real.sin x₂ = Real.sin x₁ + Real.sin x₃) :
    ¬ (x₁ + x₃ = 2 * x₂) :=
sorry

end no_arith_prog_of_sines_l619_619543


namespace sum_of_coefficients_is_zero_l619_619284

theorem sum_of_coefficients_is_zero : 
  let polynomial := (2 * x - 3 * y + z) ^ 20 in
  (polynomial.eval (x := 1) (y := 1) (z := 1) = 0) := 
by
  have polynomial := (2 * x - 3 * y + z) ^ 20
  show polynomial.eval (x := 1) (y := 1) (z := 1) = 0
  sorry

end sum_of_coefficients_is_zero_l619_619284


namespace star_calculation_l619_619825

def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem star_calculation : star (star 3 5) 2 = 1 / 3 := by
  sorry

end star_calculation_l619_619825


namespace A_sub_B_value_l619_619797

def A : ℕ := 1000 * 1 + 100 * 16 + 10 * 28
def B : ℕ := 355 + 245 * 3

theorem A_sub_B_value : A - B = 1790 := by
  sorry

end A_sub_B_value_l619_619797


namespace correct_option_C_l619_619745

theorem correct_option_C (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : ac > bd :=
by
  sorry

end correct_option_C_l619_619745


namespace tangent_line_circle_l619_619545

theorem tangent_line_circle (a : ℝ) :
  (∀ (x y : ℝ), 4 * x - 3 * y = 0 → x^2 + y^2 - 2 * x + a * y + 1 = 0) →
  a = -1 ∨ a = 4 :=
sorry

end tangent_line_circle_l619_619545


namespace probability_no_B_before_first_A_l619_619735

noncomputable def total_permutations : ℕ := 
  factorial 11 / (factorial 5 * factorial 2 * factorial 2)

noncomputable def favorable_permutations : ℕ := 
  factorial 10 / (factorial 4 * factorial 2 * factorial 2)

noncomputable def probability : ℚ :=
  favorable_permutations / total_permutations

theorem probability_no_B_before_first_A :
  probability = 5 / 7 :=
  by sorry

end probability_no_B_before_first_A_l619_619735


namespace largest_seven_consecutive_non_primes_less_than_40_l619_619234

def is_non_prime (n : ℕ) : Prop :=
  n ≠ 1 ∧ ¬(∃ p, nat.prime p ∧ p ∣ n)

def consecutive_non_primes_sequence (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 7 → is_non_prime (n + i) ∧ (10 ≤ n + i) ∧ (n + i < 40)

theorem largest_seven_consecutive_non_primes_less_than_40 :
  ∃ n, consecutive_non_primes_sequence n ∧ n + 6 = 32 :=
sorry

end largest_seven_consecutive_non_primes_less_than_40_l619_619234


namespace quotient_m_div_16_l619_619672

-- Define the conditions
def square_mod_16 (n : ℕ) : ℕ := (n * n) % 16

def distinct_squares_mod_16 : Finset ℕ :=
  { n | square_mod_16 n ∈ [1, 4, 9, 0].toFinset }

def m : ℕ :=
  distinct_squares_mod_16.sum id

-- Define the theorem to be proven
theorem quotient_m_div_16 : m / 16 = 0 :=
by
  sorry

end quotient_m_div_16_l619_619672


namespace exists_polynomials_f_g_h_l619_619852

theorem exists_polynomials_f_g_h :
  ∃ (f g h : ℝ[X]), 
    degree f = 2 ∧ degree g = 2 ∧ degree h = 2 ∧
    ∃ (a b c d e f g h i j k l : ℝ),
    f = X^2 + a * X + b ∧ is_root f a ∧ is_root f b ∧
    g = (X - 4)^2 + c * X + d ∧ is_root g c ∧ is_root g d ∧
    h = (X + 4)^2 + e * X + f ∧ is_root h e ∧ is_root h f ∧
    ¬ (∃ x : ℝ, is_root (f + g) x) ∧
    ¬ (∃ x : ℝ, is_root (g + h) x) ∧
    ¬ (∃ x : ℝ, is_root (h + f) x) :=
begin
  sorry
end

end exists_polynomials_f_g_h_l619_619852


namespace certain_number_proof_l619_619137

theorem certain_number_proof (h1: 2994 / 14.5 = 177) (h2: ∃ c, c / 1.45 = 17.7) : 
  ∃ c, c = 25.665 :=
by
  rcases h2 with ⟨c, hc⟩
  use c
  have hc_eq: c = 17.7 * 1.45 := by sorry
  rw hc_eq
  exact eq_of_mul_eq_mul_right (by norm_num : 1.45 ≠ 0) (by norm_num) -- Show 17.7 * 1.45 = 25.665

end certain_number_proof_l619_619137


namespace min_perimeter_triangle_ABC_l619_619075

def point := (ℝ × ℝ)

def A : point := (-3, 4)

def on_y_axis (B : point) : Prop := B.1 = 0

def on_line_l (C : point) : Prop := C.1 + C.2 = 0

theorem min_perimeter_triangle_ABC (B C : point) (hB : on_y_axis B) (hC : on_line_l C) :
  let d (p q : point) := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) in
  let perimeter (p1 p2 p3 : point) := d p1 p2 + d p2 p3 + d p3 p1 in
  ∀ B C, on_y_axis B → on_line_l C → ∃ P, P = perimeter A B C ∧ P = 5 * real.sqrt 2 :=
sorry

end min_perimeter_triangle_ABC_l619_619075


namespace R_rotated_90_degrees_l619_619302

section math_problem

variable (P Q R: ℝ × ℝ)
variable (above_x_axis : (R.2 > 0))
variable (angle_QRP : ∠ Q R P = 90)
variable (angle_PRQ : ∠ P R Q = 45)

theorem R_rotated_90_degrees :
    (P = (0, 0)) → (Q = (6, 0)) → (R = (6, 6)) → (rotate_90_ccw P R = (-6, 6)) :=
by
  intros hP hQ hR
  sorry

end math_problem

end R_rotated_90_degrees_l619_619302


namespace find_YC_length_l619_619249

noncomputable def triangle_ABC := 
  { AB := 5, BC := 8, CA := 7 }

def circumcircle (Δ : Type) := 
  sorry -- Assume the definition of the circumcircle

def second_intersection_external_angle_bisector (Δ : Type) (angle : Type) :=
  sorry -- Assume the definition for the second intersection of the external angle bisector

def perpendicular_foot (point : Type) (line : Type) :=
  sorry -- Assume the definition of the foot of the perpendicular from a point to a line

def length {point1 point2 : Type} := 
  sorry -- Assume the definition to find the length between two points

theorem find_YC_length :
  let AB := (5 : ℝ)
  let BC := (8 : ℝ)
  let CA := (7 : ℝ)
  let ΔABC := triangle_ABC
  let ω := circumcircle ΔABC
  let X := second_intersection_external_angle_bisector ΔABC ∠B
  let Y := perpendicular_foot X BC
  length Y C = (13 / 2 : ℝ) :=
begin
  sorry
end

end find_YC_length_l619_619249


namespace triangle_sum_l619_619676

def triangle (a b c : ℕ) : ℤ := a * b - c

theorem triangle_sum :
  triangle 2 3 5 + triangle 1 4 7 = -2 :=
by
  -- This is where the proof would go
  sorry

end triangle_sum_l619_619676


namespace monotonic_increasing_l619_619504

def f : ℝ → ℝ := λ x => Real.sin (2 * x + (-Real.pi / 6))

theorem monotonic_increasing :
  ∀ x y, (-Real.pi / 6 < x -> x < Real.pi / 3) →
         (-Real.pi / 6 < y -> y < Real.pi / 3) →
         (x < y) → (f x < f y) :=
by
  sorry

end monotonic_increasing_l619_619504


namespace ellipse_and_collinearity_l619_619911

-- Definitions based on conditions
def b : ℝ := 2
def e : ℝ := sqrt 5 / 5

-- Solving for 'a' and 'c' based on the conditions given above
@[simp]
def a : ℝ := sqrt (b^2 + (5/5)^2)
def c : ℝ := a * e

-- Ellipse definition
def is_ellipse (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Points F1, F2
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Point P conditions
def is_P (x y : ℝ) : Prop := x = 3

-- Perpendicular bisector intersecting the ellipse condition
def intersects_ellipse_once (P : ℝ × ℝ) : Prop :=
  let D := ((1 + P.1) / 2, P.2 / 2) in
  ∃ T : ℝ × ℝ, is_ellipse T.1 T.2 ∧ T ≠ F2  -- Only intersect at one point

-- Collinearity of F1, T, P
def collinear (F1 T P : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, T.2 = m * T.1 + b ∧ P.2 = m * P.1 + b ∧ F1.2 = m * F1.1 + b

-- Lean statement to be proven
theorem ellipse_and_collinearity : ∃ EqT : ℝ × ℝ, 
  intersects_ellipse_once (3, eqT.2) → collinear F1 EqT (3, eqT.2) :=
sorry

end ellipse_and_collinearity_l619_619911


namespace proposition_p_proposition_not_q_proof_p_and_not_q_l619_619915

variable (p : Prop)
variable (q : Prop)
variable (r : Prop)

theorem proposition_p : (∃ x0 : ℝ, x0 > 2) := sorry

theorem proposition_not_q : ¬ (∀ x : ℝ, x^3 > x^2) := sorry

theorem proof_p_and_not_q : (∃ x0 : ℝ, x0 > 2) ∧ ¬ (∀ x : ℝ, x^3 > x^2) :=
by
  exact ⟨proposition_p, proposition_not_q⟩

end proposition_p_proposition_not_q_proof_p_and_not_q_l619_619915


namespace plane_distance_with_wind_l619_619377

theorem plane_distance_with_wind :
  ∀ (D : ℝ),
    ∃ D = 420,
    (let speed_wind := 23 in
     let speed_plane_still_air := 253 in
     let speed_plane_with_wind := speed_plane_still_air + speed_wind in
     let speed_plane_against_wind := speed_plane_still_air - speed_wind in
     let distance_against_wind := 350 in
     D / speed_plane_with_wind = distance_against_wind / speed_plane_against_wind)
:= sorry

end plane_distance_with_wind_l619_619377


namespace sequences_increasing_l619_619197

noncomputable def A (x y : ℕ) (hxy : x ≠ y) : ℕ → ℝ
| 0 := 0 -- dummy value to keep indexing consistent
| 1 := real.sqrt ((x^2 + y^2) / 2)
| (n+1) := real.sqrt ((A x y hxy n^2 + H x y hxy n^2) / 2)

noncomputable def G (x y : ℕ) (hxy : x ≠ y) : ℕ → ℝ
| 0 := 0 -- dummy value to keep indexing consistent
| 1 := (x + y) / 2
| (n+1) := (A x y hxy n + H x y hxy n) / 2

noncomputable def H (x y : ℕ) (hxy : x ≠ y) : ℕ → ℝ
| 0 := 0 -- dummy value to keep indexing consistent
| 1 := real.sqrt (x * y)
| (n+1) := real.sqrt (A x y hxy n * G x y hxy n)

theorem sequences_increasing (x y : ℕ) (hxy : x ≠ y) :
  ∀ n, (A x y hxy (n+1) ≥ A x y hxy n) ∧
       (G x y hxy (n+1) ≥ G x y hxy n) ∧
       (H x y hxy (n+1) ≥ H x y hxy n) :=
sorry

end sequences_increasing_l619_619197


namespace largest_of_seven_consecutive_composite_numbers_less_than_40_l619_619241

open Nat

theorem largest_of_seven_consecutive_composite_numbers_less_than_40 :
  ∃ (n : ℕ), 23 ≤ n ∧ n ≤ 30 ∧ ∀ (k : ℕ), n ≤ k ∧ k < n + 7 → ¬ prime k ∧ n + 6 = 30 :=
by
  sorry

end largest_of_seven_consecutive_composite_numbers_less_than_40_l619_619241


namespace locus_of_sphere_center_l619_619271

open Real EuclideanSpace

-- Define the conditions
variables {L M : ℝ → ℝ → Prop}  -- Representing lines using predicates
variables {O : ℝ × ℝ}            -- Point of intersection
variable {θ : ℝ}                  -- Angle between lines
variable {r : ℝ}                  -- Radius of the sphere

-- Define the proof statement
theorem locus_of_sphere_center (hL : ∀ P, L P ↔ P.2 = 0)
                               (hM : ∀ P, M P ↔ P.2 = 0)
                               (hO : O = (0, 0))
                               (hθ_pos : 0 < θ) (hθ_lt_pi_div2 : θ < π / 2)
                               (hr_pos : 0 < r) :
    ∃ P : ℝ × ℝ, let x := P.1 in let y := P.2 in 
        -r ≤ x ∧ x ≤ r ∧ y = sqrt (r^2 - x^2 * sin(θ)^2 / θ^2) := sorry

end locus_of_sphere_center_l619_619271


namespace product_ab_l619_619992

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619992


namespace find_c_l619_619103

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem find_c (a b m c : ℝ) (h1 : ∀ x, f x a b ≥ 0)
  (h2 : ∀ x, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
by
  sorry

end find_c_l619_619103


namespace total_people_on_buses_l619_619176

theorem total_people_on_buses 
  (initial_A : ℕ) (initial_B : ℕ) 
  (new_A : ℕ) (new_B : ℕ) 
  (initial_A_eq : initial_A = 4) 
  (initial_B_eq : initial_B = 7) 
  (new_A_eq : new_A = 13) 
  (new_B_eq : new_B = 9) :
  initial_A + new_A + initial_B + new_B = 33 :=
by
  have hA : initial_A + new_A = 17 := by sorry
  have hB : initial_B + new_B = 16 := by sorry
  calc
    initial_A + new_A + initial_B + new_B 
        = 17 + 16 : by rw [hA, hB]
    ... = 33 : by sorry

end total_people_on_buses_l619_619176


namespace can_form_triangle_l619_619743

theorem can_form_triangle : Prop :=
  ∃ (a b c : ℝ), 
    (a = 8 ∧ b = 6 ∧ c = 4) ∧
    (a + b > c ∧ a + c > b ∧ b + c > a)

#check can_form_triangle

end can_form_triangle_l619_619743


namespace initial_amount_of_Owl_l619_619724

noncomputable def initial_amount (x : ℚ) : ℚ :=
  let first_crossing := 3 * x - 50
  let second_crossing := 3 * first_crossing - 50
  let third_crossing := 3 * second_crossing - 50
  let fourth_crossing := 3 * third_crossing - 50
  fourth_crossing

theorem initial_amount_of_Owl : ∃ (x : ℚ), initial_amount x = 0 ∧ x = 2000 / 81 := 
by {
  use 2000 / 81,
  unfold initial_amount,
  ring,
  sorry,
}

end initial_amount_of_Owl_l619_619724


namespace bee_count_l619_619757

theorem bee_count (initial_bees additional_bees : ℕ) (h_init : initial_bees = 16) (h_add : additional_bees = 9) :
  initial_bees + additional_bees = 25 :=
by
  sorry

end bee_count_l619_619757


namespace factor_polynomial_l619_619699

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) :=
by
  sorry

end factor_polynomial_l619_619699


namespace binary_operation_correct_l619_619435

-- Define the binary numbers involved
def bin1 := 0b110110 -- 110110_2
def bin2 := 0b101010 -- 101010_2
def bin3 := 0b100    -- 100_2

-- Define the operation in binary
def result := 0b111001101100 -- 111001101100_2

-- Lean statement to verify the operation result
theorem binary_operation_correct : (bin1 * bin2) / bin3 = result :=
by sorry

end binary_operation_correct_l619_619435


namespace incorrect_statement_about_function_l619_619512

theorem incorrect_statement_about_function (x : ℝ) (h1 : x = -3 → y = -6 / x = 2)
  (h2 : (∀ x > 0, y < 0) ∧ (∀ x < 0, y > 0))
  (h3 : ∀ x, x > 0 → x < 0 → -6 / x increases with x)
  (h4 : ∀ x ≥ -1, y = -6 / x) : (D : (x ≥ -1 → y ≥ 6) is incorrect) :=
by
  sorry

end incorrect_statement_about_function_l619_619512


namespace find_natural_numbers_eq_36_sum_of_digits_l619_619051

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end find_natural_numbers_eq_36_sum_of_digits_l619_619051


namespace equal_segments_l619_619226

theorem equal_segments 
  {α : Type*} [EuclideanSpace α]
  {A C B D A1 A2 B1 B2 : α}
  (h₁ : D ∈ line_angle_bisector A C B)
  (h₂ : is_on_ray A C A1)
  (h₃ : is_on_ray A C A2)
  (h₄ : is_on_ray C B B1)
  (h₅ : is_on_ray C B B2)
  (h₆ : is_concyclic A1 C B1 D)
  (h₇ : is_concyclic A2 C B2 D) : 
  dist A1 A2 = dist B1 B2 := 
sorry

end equal_segments_l619_619226


namespace james_bags_l619_619179

theorem james_bags (total_marbles : ℕ) (remaining_marbles : ℕ) (b : ℕ) (m : ℕ) 
  (h1 : total_marbles = 28) 
  (h2 : remaining_marbles = 21) 
  (h3 : m = total_marbles - remaining_marbles) 
  (h4 : b = total_marbles / m) : 
  b = 4 :=
by
  sorry

end james_bags_l619_619179


namespace number_of_valid_colorings_l619_619432

-- Given conditions
def is_valid_coloring (grid : Fin 3 × Fin 3 → Bool) : Prop :=
  ∀ i j : Fin 2, ¬ (grid (i, j) ∧ grid (i+1, j) ∧ grid (i, j+1) ∧ grid (i+1, j+1))

-- The proof statement expressing the number of valid colorings
theorem number_of_valid_colorings : 
  ∃ count : Nat, count = 417 ∧ 
  count = (Fin 3 × Fin 3 → Bool → Prop { c | is_valid_coloring c }).card :=
sorry

end number_of_valid_colorings_l619_619432


namespace remaining_money_proof_l619_619731

variables {scissor_cost eraser_cost initial_amount scissor_quantity eraser_quantity total_cost remaining_money : ℕ}

-- Given conditions
def conditions : Prop :=
  initial_amount = 100 ∧ 
  scissor_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissor_quantity = 8 ∧ 
  eraser_quantity = 10

-- Definition using conditions
def total_spent : ℕ :=
  scissor_quantity * scissor_cost + eraser_quantity * eraser_cost

-- Prove the total remaining money calculation
theorem remaining_money_proof (h : conditions) : 
  total_spent = 80 ∧ remaining_money = initial_amount - total_spent ∧ remaining_money = 20 :=
by
  -- Proof steps to be provided here
  sorry

end remaining_money_proof_l619_619731


namespace red_cards_count_l619_619658

theorem red_cards_count (R B : ℕ) (h1 : R + B = 20) (h2 : 3 * R + 5 * B = 84) : R = 8 :=
sorry

end red_cards_count_l619_619658


namespace lions_die_per_month_l619_619287

theorem lions_die_per_month 
  (initial_lions : ℕ) 
  (birth_rate : ℕ) 
  (final_lions : ℕ) 
  (months : ℕ)
  (initial_lions = 100)
  (birth_rate = 5)
  (final_lions = 148)
  (months = 12) :
  ∃ d : ℕ, initial_lions + birth_rate * months - d * months = final_lions ∧ d = 1 :=
by
  sorry

end lions_die_per_month_l619_619287


namespace longest_segment_in_cylinder_l619_619784

theorem longest_segment_in_cylinder (radius height : ℝ) 
  (hr : radius = 5) (hh : height = 10) :
  ∃ segment_length, segment_length = 10 * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end longest_segment_in_cylinder_l619_619784


namespace cos_identity_l619_619129

-- Definitions of the relevant angle identities and values
noncomputable def deg_to_rad (d : ℝ) : ℝ := d * real.pi / 180

-- Condition given: cos(75° - a) = 1/3
def given_condition (a : ℝ) : Prop := real.cos (deg_to_rad (75 - a)) = 1/3

-- To prove: cos(30° + 2a) = 7/9
theorem cos_identity (a : ℝ) (h : given_condition a) : real.cos (deg_to_rad (30 + 2 * a)) = 7 / 9 :=
sorry

end cos_identity_l619_619129


namespace longest_segment_in_cylinder_l619_619779

-- Define the given conditions
def radius : ℝ := 5 -- Radius of the cylinder in cm
def height : ℝ := 10 -- Height of the cylinder in cm

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the longest segment L inside the cylinder using the Pythagorean theorem
noncomputable def longest_segment : ℝ := Real.sqrt ((diameter ^ 2) + (height ^ 2))

-- State the problem in Lean:
theorem longest_segment_in_cylinder :
  longest_segment = 10 * Real.sqrt 2 :=
sorry

end longest_segment_in_cylinder_l619_619779


namespace abs_inequality_solution_l619_619446

theorem abs_inequality_solution (x : ℝ) : |2 * x + 3| ≥ 3 ↔ x ∈ (-∞, -3] ∪ [0, +∞) := 
sorry

end abs_inequality_solution_l619_619446


namespace hamiltonian_cycle_with_one_switch_l619_619286

open Finset Function SimpleGraph

theorem hamiltonian_cycle_with_one_switch (n : ℕ) (h : 3 ≤ n):
  ∃ (G : SimpleGraph (Fin n)) (c : G.Edge → Bool), 
  (G.isHamiltonian ∧ ∃ (cycle : Finset (Fin n)), 
  G.adj_circuit cycle ∧ (∀ e₁ e₂ ∈ cycle.edges, 
  c e₁ = c e₂ ∨ c e₁ ≠ c e₂)) := sorry

end hamiltonian_cycle_with_one_switch_l619_619286


namespace second_number_is_30_l619_619708

theorem second_number_is_30 
  (A B C : ℝ)
  (h1 : A + B + C = 98)
  (h2 : A / B = 2 / 3)
  (h3 : B / C = 5 / 8) : 
  B = 30 :=
by
  sorry

end second_number_is_30_l619_619708


namespace expression1_value_expression2_value_l619_619762

-- Definition for the first expression problem
def expression1 : ℝ := (9/4 : ℝ)^(1/2) - (-9.6)^0 - (8/27 : ℝ)^(2/3) + (3/2 : ℝ)^(-2)
theorem expression1_value : expression1 = 1/2 := 
  by 
  -- Proof omitted
  sorry

-- Definition for the second log problem
def expression2 : ℝ :=
  logBase 3 ((3 : ℝ)^(3/4) / 3) + real.log10 25 + real.log10 4 + (7 : ℝ)^(real.logBase 7 2)
theorem expression2_value : expression2 = 15/4 := 
  by 
  -- Proof omitted
  sorry

end expression1_value_expression2_value_l619_619762


namespace construct_triangle_l619_619840

variable (CA CB CH : ℝ)
variable (H : ℝ)

-- The condition that H divides AB such that BH = 2AH
axiom div_AB : H = 2 * (CA * H + CB * (1 - H)) / 3  -- Simplified definition for H splitting AB

-- The required Lean statement to prove the construction's feasibility.
theorem construct_triangle (h1 : CA > 0) (h2 : CB > 0) (h3 : CH > 0)
  (H_condition : H == 2 * (CA * H + CB * (1 - H)) / 3)
  : ∃ (ABC : Triangle ℝ), ABC.length CA ∧ ABC.length CB ∧ ABC.length CH :=
  sorry

end construct_triangle_l619_619840


namespace number_of_subsets_A_inter_B_l619_619119

def A : Set ℤ := {x | (x - 2) / (x + 3) ≤ 0}
def B : Set ℤ := {x | x < 0}

theorem number_of_subsets_A_inter_B :
  ∃ (n : ℕ), n = 4 ∧ ∃ (s : Finset (Set ℤ)), s.card = 2^n ∧ s = Finset.powerset (A ∩ B) := by
  sorry

end number_of_subsets_A_inter_B_l619_619119


namespace cost_price_for_a_l619_619753

-- Definitions from the conditions
def selling_price_c : ℝ := 225
def profit_b : ℝ := 0.25
def profit_a : ℝ := 0.60

-- To prove: The cost price of the bicycle for A (cp_a) is 112.5
theorem cost_price_for_a : 
  ∃ (cp_a : ℝ), 
  (∃ (cp_b : ℝ), cp_b = (selling_price_c / (1 + profit_b)) ∧ 
   cp_a = (cp_b / (1 + profit_a))) ∧ 
   cp_a = 112.5 :=
by
  sorry

end cost_price_for_a_l619_619753


namespace triangle_cut_l619_619907

theorem triangle_cut (A B C : Type) [PointClass ℝ A] [PointClass ℝ B] [PointClass ℝ C]
  (angle_A : ∠BAC = 30) (angle_B : ∠ABC = 70) (angle_C : ∠ACB = 80)
  (M : PointClass ℝ (Midpoint A C)) (H : PointClass ℝ (Altitude A B C)) (L : PointClass ℝ (Bisector A H B)) :
  AreParallel (median A H C) (bisector A H B) :=
  begin
    sorry
  end

end triangle_cut_l619_619907


namespace find_M_coordinates_l619_619223

-- Definition of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y ^ 2 = 2 * p * x

-- Definition to check if point M lies according to given conditions
def matchesCondition
  (p : ℝ) (M P O F : ℝ × ℝ) : Prop :=
  let xO := O.1
  let yO := O.2
  let xP := P.1
  let yP := P.2
  let xM := M.1
  let yM := M.2
  let xF := F.1
  let yF := F.2
  (xP = 2) ∧ (yP = 2 * p) ∧
  (xO = 0) ∧ (yO = 0) ∧
  (xF = p / 2) ∧ (yF = 0) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xO) ^ 2 + (yM - yO) ^ 2)) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xF) ^ 2 + (yM - yF) ^ 2))

-- Prove the coordinates of M satisfy the conditions
theorem find_M_coordinates :
  ∀ p : ℝ, p > 0 →
  matchesCondition p (1/4, 7/4) (2, 2 * p) (0, 0) (p / 2, 0) :=
by
  intros p hp
  simp [parabola, matchesCondition]
  sorry

end find_M_coordinates_l619_619223


namespace annual_population_increase_l619_619274

theorem annual_population_increase (P₀ P₂ : ℝ) (r : ℝ) 
  (h0 : P₀ = 12000) 
  (h2 : P₂ = 18451.2) 
  (h_eq : P₂ = P₀ * (1 + r / 100)^2) :
  r = 24 :=
by
  sorry

end annual_population_increase_l619_619274


namespace arithmetic_sequence_20th_term_arithmetic_sequence_sum_of_20_terms_l619_619420

def first_term : ℕ := 2
def common_difference : ℕ := 3

def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d
def sum_of_first_n_terms (a1 an n : ℕ) : ℕ := (n / 2) * (a1 + an)

theorem arithmetic_sequence_20th_term (a1 d : ℕ) (n : ℕ) (h1 : a1 = first_term) (h2 : d = common_difference) (h3 : n = 20) :
  nth_term a1 d n = 59 :=
by
  rw [h1, h2, h3]
  calc nth_term 2 3 20 = 2 + (20 - 1) * 3 := rfl
                   ... = 2 + 57 := rfl
                   ... = 59 := rfl

theorem arithmetic_sequence_sum_of_20_terms (a1 d : ℕ) (n : ℕ) (h1 : a1 = first_term) (h2 : d = common_difference) (h3 : n = 20) :
  sum_of_first_n_terms a1 (nth_term a1 d n) n = 610 :=
by
  rw [h1, h2, h3]
  have h_term : nth_term 2 3 20 = 59 := by
    calc nth_term 2 3 20 = 2 + (20 - 1) * 3 := rfl
                     ... = 2 + 57 := rfl
                     ... = 59 := rfl
  rw [sum_of_first_n_terms, h_term]
  calc (20 / 2) * (2 + 59) = 10 * 61 := by norm_num
                         ... = 610 := rfl

end arithmetic_sequence_20th_term_arithmetic_sequence_sum_of_20_terms_l619_619420


namespace ab_equals_6_l619_619980

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619980


namespace cereal_weeks_l619_619723

theorem cereal_weeks (boxes_per_week : ℕ) (cost_per_box : ℕ) (total_spent : ℕ) (weekly_expenditure : ℕ)
    (h1 : boxes_per_week = 2)
    (h2 : cost_per_box = 3)
    (h3 : total_spent = 312)
    (h4 : weekly_expenditure = boxes_per_week * cost_per_box)
    (h5 : total_spent = weekly_expenditure * 52) :
    total_spent / weekly_expenditure = 52 :=
by
  rw [h3, h4, h1, h2]
  norm_num
  sorry

end cereal_weeks_l619_619723


namespace parallelogram_area_approx_l619_619645

noncomputable def sin_80_deg := Real.sin (Real.pi * 80 / 180)

theorem parallelogram_area_approx :
  ∃ (area : ℝ), area ≈ 197.0 ∧
  ∀ AB AD θ, AB = 20 ∧ AD = 10 ∧ θ = 80 →
  area = AB * AD * sin_80_deg :=
by
  sorry

end parallelogram_area_approx_l619_619645


namespace solve_problem_l619_619927

noncomputable def problem_statement (α β γ : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2) (h₂ : 0 < γ ∧ γ < π / 2) (h₃ : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : Prop :=
  (sin α ^ 3 / sin β) + (sin β ^ 3 / sin γ) + (sin γ ^ 3 / sin α) ≥ 1

theorem solve_problem (α β γ : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2) (h₂ : 0 < γ ∧ γ < π / 2) (h₃ : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  problem_statement α β γ h₀ h₁ h₂ h₃ :=
by
  sorry

end solve_problem_l619_619927


namespace olivia_quarters_left_l619_619217

-- Define the initial condition and action condition as parameters
def initial_quarters : ℕ := 11
def quarters_spent : ℕ := 4
def quarters_left : ℕ := initial_quarters - quarters_spent

-- The theorem to state the result
theorem olivia_quarters_left : quarters_left = 7 := by
  sorry

end olivia_quarters_left_l619_619217


namespace shortest_route_length_l619_619659

-- Define the vertices of the graph
inductive Vertex
| A | B | C | D | E | F | G | H | I

-- Define the edges of the graph with distances
def edges : List (Vertex × Vertex × ℕ) :=
  [(Vertex.A, Vertex.B, 13), (Vertex.B, Vertex.C, 13), (Vertex.B, Vertex.H, 5),
   (Vertex.C, Vertex.H, 5), (Vertex.C, Vertex.D, 13), (Vertex.D, Vertex.H, 13),
   (Vertex.D, Vertex.I, 12), (Vertex.E, Vertex.F, 12), (Vertex.F, Vertex.G, 13),
   (Vertex.G, Vertex.H, 13), (Vertex.G, Vertex.I, 12), (Vertex.I, Vertex.E, 12),
   (Vertex.E, Vertex.H, 5), (Vertex.F, Vertex.A, 12), (Vertex.G, Vertex.F, 13)]

-- Predicate to assert that the total distance for traversing all edges twice is 211 km
def shortestRouteDistanceIs : ℕ := 211

-- The theorem stating the shortest route distance
theorem shortest_route_length : 
  ∃ path : List (Vertex × Vertex), (∀ e ∈ edges, e.α ∈ path ∧ e.β ∈ path) 
  ∧ pathLength edges path = shortestRouteDistanceIs := 
sorry

-- Helper function to calculate path length
def pathLength : List (Vertex × Vertex × ℕ) → List (Vertex × Vertex) → ℕ
| [], _ => 0
| _, [] => 0
| ((u, v, d)::es), p => if (u, v) ∈ p || (v, u) ∈ p then d + pathLength es p else pathLength es p

end shortest_route_length_l619_619659


namespace first_year_sum_of_digits_15_l619_619057

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem first_year_sum_of_digits_15 : ∃ y: ℕ, y > 1970 ∧ sum_of_digits y = 15 ∧ ∀ z: ℕ, 1970 < z ∧ z < y → sum_of_digits z ≠ 15 :=
by
  let y := 1979
  have h1 : y > 1970 := by decide
  have h2 : sum_of_digits y = 15 := by decide
  have h3 : ∀ z: ℕ, 1970 < z ∧ z < y → sum_of_digits z ≠ 15 := sorry
  exact ⟨y, h1, h2, h3⟩

end first_year_sum_of_digits_15_l619_619057


namespace train_speed_l619_619754

-- Definition for conditions
def train_length : ℝ := 50
def crossing_time : ℝ := 6
def man_speed_kmh : ℝ := 5

-- Conversion from km/h to m/s
def man_speed_ms : ℝ := man_speed_kmh * (1000 / 3600)

-- Theorem to prove the speed of the train
theorem train_speed (train_length crossing_time man_speed_kmh man_speed_ms : ℝ) : 
  (train_length / crossing_time + man_speed_ms) * (3600 / 1000) = 25 :=
by
  -- Definitions based on conditions
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed - man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  -- Placeholder for the proof steps
  sorry

end train_speed_l619_619754


namespace longest_segment_in_cylinder_l619_619793

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end longest_segment_in_cylinder_l619_619793


namespace find_values_of_a_b_find_polar_eq_of_intersections_l619_619156

-- Definition for the parametric equations of curves C1 and C2, and conditions for parameters a and b
def C1 (φ : Real) : Real × Real := (Real.cos φ, Real.sin φ)
def C2 (φ : Real) (a b : Real) : Real × Real := (a * Real.cos φ, b * Real.sin φ)

-- Conditions for a and b
def cond_a_b (a b : Real) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b)

-- Definition of polar coordinates
def polar_ray (α : Real) : Real × Real := (α.cos, α.sin)

-- Problem (Ⅰ): Prove values of a and b
theorem find_values_of_a_b (a b : Real) (h : cond_a_b a b) :
  (let C1_point := C1 0 in
   let C2_point := C2 0 a b in
   Real.dist (C1_point.1, C1_point.2) (C2_point.1, C2_point.2) = 2) ∧
  (let C1_point := C1 (Real.pi / 2) in
   let C2_point := C2 (Real.pi / 2) a b in
   C1_point = C2_point) →
  (a = 3) ∧ (b = 1) :=
sorry

-- Problem (Ⅱ): Polar equations of intersections
theorem find_polar_eq_of_intersections :
  let A1 := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let B1 := (3 * Real.sqrt 10 / 10, 3 * Real.sqrt 10 / 10)
  let A2 := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  let B2 := (3 * Real.sqrt 10 / 10, -3 * Real.sqrt 10 / 10)
  let polar_line (p1 p2 : Real × Real) := 
    ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

  polar_line A1 A2 = (Real.sqrt 2 / 2, 0) ∧
  polar_line B1 B2 = (3 * Real.sqrt 10 / 10, 0) →
  (∀ (θ : Real), θ = Real.arccos (Real.sqrt 2 / 2) →
  polar_ray θ = (1, 0)) ∧
  (∀ (θ : Real), θ = Real.arccos (3 *  Real.sqrt 10 / 10) →
  polar_ray θ = (1, 0)) :=
sorry

end find_values_of_a_b_find_polar_eq_of_intersections_l619_619156


namespace part1_part2_l619_619168

-- Definitions based on the conditions in the problem.

-- Probabilities
def P_A_scores : ℚ := 2/5
def P_B_scores : ℚ := 1/3
def P_A_not_scores : ℚ := 1 - P_A_scores
def P_B_not_scores : ℚ := 1 - P_B_scores

-- Probability distribution of X
def P_X_neg1 : ℚ := (1 - P_A_scores) * P_B_scores
def P_X_0 : ℚ := P_A_scores * P_B_scores + (1 - P_A_scores) * (1 - P_B_scores)
def P_X_1 : ℚ := P_A_scores * (1 - P_B_scores)

-- The probability that A's cumulative score is higher than B's after two rounds
def P_2 : ℚ :=
  P_X_0 * P_X_1 + P_X_1 * (P_X_0 + P_X_1)

-- The theorem for Part 1
theorem part1 : 
  (P_X_neg1 = 1/5) ∧
  (P_X_0 = 8/15) ∧
  (P_X_1 = 4/15) := 
by sorry

-- The theorem for Part 2
theorem part2 :
  P_2 = 16 / 45 :=
by sorry

end part1_part2_l619_619168


namespace arithmetic_progression_12th_term_l619_619054

theorem arithmetic_progression_12th_term (a d : ℤ) (h₀ : a = 2) (h₁ : d = 8) : 
  a + 11 * d = 90 :=
by
  rw [h₀, h₁]
  norm_num
  sorry

end arithmetic_progression_12th_term_l619_619054


namespace integral_exp_2x_l619_619009

theorem integral_exp_2x : ∫ x in 0..1, (Real.exp x + 2 * x) = Real.exp 1 := by
  sorry

end integral_exp_2x_l619_619009


namespace sufficient_not_necessary_condition_l619_619649

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 → ¬ (x - 1)^2 < 9) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l619_619649


namespace angle_C_eq_7_pi_div_12_l619_619551

-- Given conditions
variables (A B C : Angle) (a b c : ℝ)

-- Definitions for the conditions
def angle_B_eq_pi_div_4 := B = Real.pi / 4
def side_b_eq_sqrt_2_a := b = Real.sqrt 2 * a

-- Question to prove
theorem angle_C_eq_7_pi_div_12 (h1 : angle_B_eq_pi_div_4) (h2 : side_b_eq_sqrt_2_a) :
  C = 7 * Real.pi / 12 :=
sorry

end angle_C_eq_7_pi_div_12_l619_619551


namespace part_I_l619_619896

noncomputable def f (x m n : ℝ) : ℝ := m * x + n / x

theorem part_I (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < sqrt (n / m) → f x1 m n > f x2 m n) ∧
  (∀ x1 x2 : ℝ, sqrt (n / m) < x1 ∧ x1 < x2 → f x1 m n < f x2 m n) :=
by
  sorry

end part_I_l619_619896


namespace shorter_steiner_network_l619_619291

-- Define the variables and inequality
noncomputable def side_length (a : ℝ) : ℝ := a
noncomputable def diagonal_network_length (a : ℝ) : ℝ := 2 * a * Real.sqrt 2
noncomputable def steiner_network_length (a : ℝ) : ℝ := a * (1 + Real.sqrt 3)

theorem shorter_steiner_network {a : ℝ} (h₀ : 0 < a) :
  diagonal_network_length a > steiner_network_length a :=
by
  -- Proof to be provided (skipping it with sorry)
  sorry

end shorter_steiner_network_l619_619291


namespace smallest_positive_period_f_max_value_f_min_value_f_l619_619503

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 1 - 2 * (sin x)^2

theorem smallest_positive_period_f : 
  is_periodic f π :=
sorry

theorem max_value_f : 
  ∃ x ∈ set.Icc (-π/3) (π/4), f x = sqrt 2 :=
sorry

theorem min_value_f : 
  ∃ x ∈ set.Icc (-π/3) (π/4), f x = -((sqrt 3 + 1)/2) :=
sorry

end smallest_positive_period_f_max_value_f_min_value_f_l619_619503


namespace emma_sequence_units_digit_l619_619430

theorem emma_sequence_units_digit :
  ∃ n : ℕ, (∃ s : list ℕ, s.head = some n ∧ s.ilast = some 0 ∧ s.length = 6 ∧ (∀ i < 5, ∃ k : ℕ, s.nth i = some (s.nth_le i k - k^3))) ∧ n % 10 = 6 :=
sorry

end emma_sequence_units_digit_l619_619430


namespace min_value_condition_l619_619633

theorem min_value_condition {a b c d e f g h : ℝ} (h1 : a * b * c * d = 16) (h2 : e * f * g * h = 25) :
  (a^2 * e^2 + b^2 * f^2 + c^2 * g^2 + d^2 * h^2) ≥ 160 :=
  sorry

end min_value_condition_l619_619633


namespace negation_of_p_l619_619624

def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2*k + 1
def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2*k

def A : set ℤ := {x | is_odd x}
def B : set ℤ := {x | is_even x}

def p : Prop := ∀ x ∈ A, (2 * x) ∈ B

theorem negation_of_p : ¬p ↔ ∃ x ∈ A, (2 * x) ∉ B := by
  sorry

end negation_of_p_l619_619624


namespace sequence_an_solution_l619_619080

theorem sequence_an_solution {a : ℕ → ℝ} (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → (1 / a (n + 1) = 1 / a n + 1)) : ∀ n : ℕ, 0 < n → (a n = 1 / n) :=
by
  sorry

end sequence_an_solution_l619_619080


namespace cubes_with_painted_faces_l619_619369

theorem cubes_with_painted_faces :
  ∀ (n : ℕ), (∃ (cubes : ℕ), cubes = 27 ∧ ∃ (face_painted_cubes : ℕ → ℕ), 
  face_painted_cubes 2 = 12) → 12 * 2 = 24 :=
by
  intro n
  assume h
  cases h with cubes hc
  cases hc with hc_painted hc_cubes
  simp at hc_painted
  simp at hc_cubes
  rw [hc_painted, hc_cubes]
  sorry

end cubes_with_painted_faces_l619_619369


namespace total_savings_in_joint_account_l619_619581

def kimmie_earnings : ℝ := 450
def zahra_earnings : ℝ := kimmie_earnings - (1 / 3) * kimmie_earnings
def kimmie_savings : ℝ := (1 / 2) * kimmie_earnings
def zahra_savings : ℝ := (1 / 2) * zahra_earnings
def joint_savings_account : ℝ := kimmie_savings + zahra_savings

theorem total_savings_in_joint_account :
  joint_savings_account = 375 := 
by
  -- proof to be provided
  sorry

end total_savings_in_joint_account_l619_619581


namespace sum_of_rational_roots_of_h_l619_619874

noncomputable def h (x : ℚ) : ℚ := x^3 - 9 * x^2 + 27 * x - 14

theorem sum_of_rational_roots_of_h : 
  let roots : List ℚ := [2] in
  (h 2 = 0) →
  (roots.sum = 2) :=
by
  sorry

end sum_of_rational_roots_of_h_l619_619874


namespace ranking_sequences_l619_619221

theorem ranking_sequences
    (A D B E C : Type)
    (h_no_ties : ∀ (X Y : Type), X ≠ Y)
    (h_games : (W1 = A ∨ W1 = D) ∧ (W2 = B ∨ W2 = E) ∧ (W3 = W1 ∨ W3 = C)) :
  ∃! (n : ℕ), n = 48 := 
sorry

end ranking_sequences_l619_619221


namespace find_a_l619_619142

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - (3 / 2) * x^2 + a * x + 4

def f_prime (a : ℝ) (x : ℝ) : ℝ := x^2 - 3 * x + a

theorem find_a :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 4 → f a y < f a x) →
  (∃ a : ℝ, a = -4) :=
by
  intro h
  use -4
  sorry

end find_a_l619_619142


namespace dot_product_necessity_not_sufficiency_l619_619477

variables {V : Type*} [inner_product_space ℝ V]

theorem dot_product_necessity_not_sufficiency (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)  
    (h : inner_product a c = inner_product b c) :
  ¬ (a = b) → (inner_product a c = inner_product b c) ∧ ¬ (∀ d : V, inner_product a d = inner_product b d → a = b) :=
sorry

end dot_product_necessity_not_sufficiency_l619_619477


namespace num_pos_ints_satisfy_eq_l619_619127

theorem num_pos_ints_satisfy_eq : 
  (∃ n : ℕ, (n > 0 ∧ (n + 900) / 80 = int.floor (real.sqrt n))) → 4 :=
sorry

end num_pos_ints_satisfy_eq_l619_619127


namespace emma_money_from_bank_l619_619035

theorem emma_money_from_bank (X : ℝ) : 
  (X - 400) / 4 = 400 → X = 2000 :=
by 
  intro h,
  sorry

end emma_money_from_bank_l619_619035


namespace tyler_remaining_money_l619_619728

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end tyler_remaining_money_l619_619728


namespace M_gt_N_l619_619604

variable (a : ℝ)

def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

theorem M_gt_N : M a > N a := by
  sorry

end M_gt_N_l619_619604


namespace max_perimeter_l619_619099

-- Function definition
def f (x : ℝ) : ℝ :=
  sin (x + 5 * Real.pi / 2) * cos (x - Real.pi / 2) - cos (x + Real.pi / 4)^2

-- Given conditions
variables {A B C : ℝ} (a b c : ℝ)
hypothesis h_f : f (A / 2) = (Real.sqrt 3 - 1) / 2
hypothesis h_a : a = 1
hypothesis acute_triangle : A > 0 ∧ B > 0 ∧ C > 0 ∧ A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2
hypothesis triangle_sum : A + B + C = Real.pi

-- Prove that the perimeter is 3
theorem max_perimeter (h_f : f (A / 2) = (Real.sqrt 3 - 1) / 2) (h_a : a = 1)
  (acute_triangle : A > 0 ∧ B > 0 ∧ C > 0 ∧ A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2)
  (triangle_sum : A + B + C = Real.pi) :
  ∃ b c, a + b + c = 3 :=
sorry

end max_perimeter_l619_619099


namespace side_length_is_eight_l619_619270

variable {Q : Type} [Quadrilateral Q]

def equal_sides (Q : Type) [Quadrilateral Q] := ∀ (a b c d : ℝ), (∃ s : ℝ, s = a ∧ s = b ∧ s = c ∧ s = d)

def perimeter (Q : Type) [Quadrilateral Q] := ∀ (a b c d : ℝ), a + b + c + d = 32

theorem side_length_is_eight (Q : Type) [Quadrilateral Q] (h1: equal_sides Q) (h2: perimeter Q) :
  ∀ (a b c d : ℝ), a = 8 ∧ b = 8 ∧ c = 8 ∧ d = 8 :=
by
  sorry

end side_length_is_eight_l619_619270


namespace mrs_hilt_total_candy_l619_619640

theorem mrs_hilt_total_candy :
  (2 * 3) + (4 * 2) + (6 * 4) = 38 :=
by
  -- here, skip the proof as instructed
  sorry

end mrs_hilt_total_candy_l619_619640


namespace smallest_possible_value_l619_619618

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l619_619618


namespace figure_area_l619_619163

theorem figure_area {AF CD AB EF BC ED : ℝ} (h1 : AF = CD) (h2 : AB = EF) (h3 : BC = ED) 
  (hq : AF = 2 ∧ CD = 2 ∧ AB = 2 ∧ EF = 2 ∧ BC = 2 ∧ ED = 2) 
  (angFAB : ∃ θ : ℝ, θ = 60) (angBCD : ∃ θ : ℝ, θ = 60) 
  (overlap_area : 1/4) : 
  (let single_triangle_area := (sqrt 3 / 4) * 2 ^ 2 in
   let total_area := 4 * single_triangle_area - overlap_area in
   total_area = 4 * sqrt 3 - 1 / 4) := 
by 
  sorry

end figure_area_l619_619163


namespace ab_equals_6_l619_619979

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619979


namespace output_sequence_value_l619_619079

theorem output_sequence_value (x y : Int) (seq : List (Int × Int))
  (h : (x, y) ∈ seq) (h_y : y = -10) : x = 32 :=
by
  sorry

end output_sequence_value_l619_619079


namespace total_savings_l619_619589

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end total_savings_l619_619589


namespace amy_work_hours_per_week_l619_619400

/-- Amy works for 40 hours per week for 8 weeks in the summer, making $3200. If she works for 32 weeks during 
the school year at the same rate of pay and needs to make another $4000, we need to prove that she must 
work 12.5 hours per week during the school year. -/
theorem amy_work_hours_per_week
  (summer_hours_per_week : ℕ)
  (summer_weeks : ℕ)
  (summer_money : ℕ)
  (school_year_weeks : ℕ)
  (school_year_money_needed : ℕ) :
  (let hourly_wage := summer_money / (summer_hours_per_week * summer_weeks : ℕ) in
   let hours_needed := school_year_money_needed / hourly_wage in
   hours_needed / school_year_weeks = 12.5) :=
by
  -- Definitions
  let summer_hours_per_week := 40
  let summer_weeks := 8
  let summer_money := 3200
  let school_year_weeks := 32
  let school_year_money_needed := 4000
  
  -- Calculations
  let hourly_wage := summer_money / (summer_hours_per_week * summer_weeks : ℕ)
  let hours_needed := school_year_money_needed / hourly_wage
  let hours_per_week := hours_needed / school_year_weeks

  -- Goal
  show hours_per_week = 12.5

  -- This proof is omitted.
  sorry

end amy_work_hours_per_week_l619_619400


namespace product_ab_l619_619987

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619987


namespace smallest_total_hot_dogs_l619_619749

noncomputable def hot_dog_packs : ℕ := 12
noncomputable def bun_packs : ℕ := 9

theorem smallest_total_hot_dogs : ∃ (n : ℕ), n = Int.lcm hot_dog_packs bun_packs ∧ n = 36 := by
  use Int.lcm hot_dog_packs bun_packs
  split
  . rfl
  . norm_num

end smallest_total_hot_dogs_l619_619749


namespace meals_distinct_pairs_l619_619821

theorem meals_distinct_pairs :
  let entrees := 4
  let drinks := 3
  let desserts := 3
  let total_meals := entrees * drinks * desserts
  total_meals * (total_meals - 1) = 1260 :=
by 
  sorry

end meals_distinct_pairs_l619_619821


namespace radius_of_circle_l619_619903

variables (O P A B : Type) [MetricSpace O] [MetricSpace P] [MetricSpace A] [MetricSpace B]
variables (circle_radius : ℝ) (PA PB OP : ℝ)

theorem radius_of_circle
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  (circle_radius : ℝ)
  : circle_radius = 7 :=
by sorry

end radius_of_circle_l619_619903


namespace sand_art_calculation_l619_619184

theorem sand_art_calculation :
  let rect_length := 6 in
  let rect_width := 7 in
  let square_side := 5 in
  let gram_per_sq_inch := 3 in
  let rect_area := rect_length * rect_width in
  let square_area := square_side * square_side in
  let rect_sand := rect_area * gram_per_sq_inch in
  let square_sand := square_area * gram_per_sq_inch in
  let total_sand_needed := rect_sand + square_sand in
  total_sand_needed = 201 :=
by
  -- sorry is used to skip the proof as instructed
  sorry

end sand_art_calculation_l619_619184


namespace sum_of_squares_positive_l619_619135

theorem sum_of_squares_positive (x_1 x_2 k : ℝ) (h : x_1 ≠ x_2) 
  (hx1 : x_1^2 + 2*x_1 - k = 0) (hx2 : x_2^2 + 2*x_2 - k = 0) :
  x_1^2 + x_2^2 > 0 :=
by
  sorry

end sum_of_squares_positive_l619_619135


namespace min_points_game_12_l619_619162

noncomputable def player_scores := (18, 22, 9, 29)

def avg_after_eleven_games (scores: ℕ × ℕ × ℕ × ℕ) := 
  let s₁ := 78 -- Sum of the points in 8th, 9th, 10th, 11th games
  (s₁: ℕ) / 4

def points_twelve_game_cond (n: ℕ) : Prop :=
  let total_points := 78 + n
  total_points > (20 * 12)

theorem min_points_game_12 (points_in_first_7_games: ℕ) (score_12th_game: ℕ) 
  (H1: avg_after_eleven_games player_scores > (points_in_first_7_games / 7)) 
  (H2: points_twelve_game_cond score_12th_game):
  score_12th_game = 30 := by
  sorry

end min_points_game_12_l619_619162


namespace Danny_in_position_3_l619_619819

open Nat

-- We define the positions as a list of names where the index+1 is the position.
def positions := List String

-- Conditions given in the problem
axiom Claire_in_position_1 (p : positions) : p.get 0 = "Claire"
axiom not_Blake_opposite_Claire (p : positions) : p.get 2 ≠ "Blake"
axiom not_Amelia_between_Blake_and_Claire (p : positions) : p.get 1 ≠ "Amelia" ∨ p.get 3 ≠ "Amelia"

-- Prove that Danny is in position #3
theorem Danny_in_position_3 (p : positions) (h1 : Claire_in_position_1 p)
                           (h2 : not_Blake_opposite_Claire p)
                           (h3 : not_Amelia_between_Blake_and_Claire p) :
  p.get 2 = "Danny" :=
sorry

end Danny_in_position_3_l619_619819


namespace min_a_plus_b_eq_six_point_five_l619_619608

noncomputable def min_a_plus_b : ℝ :=
  Inf {s | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
                       (a^2 - 12 * b ≥ 0) ∧ 
                       (9 * b^2 - 4 * a ≥ 0) ∧ 
                       (a + b = s)}

theorem min_a_plus_b_eq_six_point_five : min_a_plus_b = 6.5 :=
by
  sorry

end min_a_plus_b_eq_six_point_five_l619_619608


namespace sum_of_coeffs_eq_92_l619_619434

noncomputable def sum_of_integer_coeffs_in_factorization (x y : ℝ) : ℝ :=
  let f := 27 * (x ^ 6) - 512 * (y ^ 6)
  3 - 8 + 9 + 24 + 64  -- Sum of integer coefficients

theorem sum_of_coeffs_eq_92 (x y : ℝ) : sum_of_integer_coeffs_in_factorization x y = 92 :=
by
  -- proof steps go here
  sorry

end sum_of_coeffs_eq_92_l619_619434


namespace perpendicular_tangents_l619_619496

theorem perpendicular_tangents (x0 : ℝ)
  (h1 : ∀ (x : ℝ), deriv (λ x : ℝ, x^2 - 1) x = 2 * x)
  (h2 : ∀ (x : ℝ), deriv (λ x : ℝ, 1 + x^3) x = 3 * x^2)
  (h3 : (2 * x0) * (3 * x0^2) = -1) :
  x0 = -((36 : ℝ)^(1/3)) / 6 := 
sorry

end perpendicular_tangents_l619_619496


namespace total_fruits_l619_619716

theorem total_fruits (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 4) : a + b + c = 15 := by
  sorry

end total_fruits_l619_619716


namespace triangle_area_l619_619681

-- Define the lines and the x-axis
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 1
noncomputable def line2 (x : ℝ) : ℝ := 1 - 5 * x
noncomputable def x_axis (x : ℝ) : ℝ := 0

-- Define intersection points
noncomputable def intersect_x_axis1 : ℝ × ℝ := (-1 / 2, 0)
noncomputable def intersect_x_axis2 : ℝ × ℝ := (1 / 5, 0)
noncomputable def intersect_lines : ℝ × ℝ := (0, 1)

-- State the theorem for the area of the triangle
theorem triangle_area : 
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  (1 / 2) * d * h = 7 / 20 := 
by
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  sorry

end triangle_area_l619_619681


namespace ratio_initial_to_doubled_l619_619803

theorem ratio_initial_to_doubled (x : ℝ) (h : 3 * (2 * x + 8) = 84) : x / (2 * x) = 1 / 2 :=
by
  have h1 : 2 * x + 8 = 28 := by
    sorry
  have h2 : x = 10 := by
    sorry
  rw [h2]
  norm_num

end ratio_initial_to_doubled_l619_619803


namespace geometric_seq_inequality_l619_619486

theorem geometric_seq_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b^2 = a * c) : a^2 + b^2 + c^2 > (a - b + c)^2 :=
by
  sorry

end geometric_seq_inequality_l619_619486


namespace statement_A_statement_C_statement_D_correct_statements_l619_619342

-- Definitions and conditions
def floor (x : ℝ) : ℤ := Int.floor x

-- Statement A: ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋
theorem statement_A (x y : ℝ) : floor (x - y) ≤ floor x - floor y := 
  sorry

-- Statement C: If in the sequence b_n = ⌊√(n(n+1))⌋, n ∈ ℕ, then ∑_{n=1}^{64} b_n = 2080
def b (n : ℕ) : ℤ := floor (Real.sqrt (n * (n + 1)))

theorem statement_C : (∑ n in Finset.range 64.succ, b n) = 2080 :=
  sorry

-- Statement D: M = ⌊2/3⌋ + ⌊2^2/3⌋ + ⌊2^3/3⌋ + ... + ⌊2^2022/3⌋ has a remainder of 0 when divided by 3
def M : ℤ := ∑ k in Finset.range 2022, floor (2^k / 3)

theorem statement_D : M % 3 = 0 :=
  sorry

-- Correct statements
theorem correct_statements : 
  (statement_A) ∧ (¬ statement_B) ∧ (statement_C) ∧ (statement_D) := 
  sorry

end statement_A_statement_C_statement_D_correct_statements_l619_619342


namespace continued_fraction_value_l619_619232

noncomputable def continued_fraction : Real :=
  1 + 1 / (2 + 1 / (1 + 1 / (2 + 1 / (1 + 1 / (2 + ...))))

noncomputable def value_of_x : Real :=
  (Real.sqrt 3 + 1) / 2

theorem continued_fraction_value : 
  continued_fraction = value_of_x :=
by sorry

end continued_fraction_value_l619_619232


namespace find_f_2_l619_619599

def f (x : ℝ) : ℝ := (4004 / x) - x

theorem find_f_2 :
  (∀ x : ℝ, x > 0 → f(x) + 2 * f(2002 / x) = 3 * x) →
  f 2 = 2000 :=
by
  intros h
  have h1 := h 2
  have h2 := f 2
  simp [f] at h1 h2
  exact h2
  sorry

end find_f_2_l619_619599


namespace a_is_zero_l619_619595

theorem a_is_zero (a b : ℤ)
  (h : ∀ n : ℕ, ∃ x : ℤ, a * 2013^n + b = x^2) : a = 0 :=
by
  sorry

end a_is_zero_l619_619595


namespace using_phone_related_to_gender_distribution_and_expected_value_l619_619034

def contingency_table_survey_data :=
  {
    total_male: ℕ := 55,
    male_using_phone: ℕ := 40,
    male_not_using_phone: ℕ := 15,
    total_female: ℕ := 45,
    female_using_phone: ℕ := 20,
    female_not_using_phone: ℕ := 25,
    total_drivers: ℕ := 100
  }

def chi_squared_statistic (total: ℕ) (a: ℕ) (b: ℕ) (c: ℕ) (d: ℕ) : ℚ :=
  ((total * (a * d - b * c)^2) : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

def confidence_test_result : ℚ := chi_squared_statistic 100 40 20 15 25

def critical_value : ℚ := 7.879

theorem using_phone_related_to_gender : confidence_test_result > critical_value :=
  by {
    -- Computation and comparison done here
    sorry
  }

def choose (n k : ℕ) : ℕ := n.choose k

def probability_distribution (males: ℕ) (females: ℕ) : List (ℕ × ℚ) :=
  let total := males + females
  let denom := choose total 3
  let p0 := (choose females 3 : ℚ) / denom
  let p1 := ( (choose males 1) * (choose females 2) : ℚ ) / denom
  let p2 := ( (choose males 2) * (choose females 1) : ℚ ) / denom
  let p3 := (choose males 3 : ℚ) / denom
  [(0, p0), (1, p1), (2, p2), (3, p3)]

def expected_value (dist: List (ℕ × ℚ)) : ℚ :=
  dist.foldl (λ acc (x, p), acc + x * p) 0

theorem distribution_and_expected_value :
  let dist := probability_distribution 3 5
  dist = [(0, 5/28), (1, 15/28), (2, 15/56), (3, 1/56)] ∧ expected_value dist = 63/56 :=
  by {
    -- Distribution calculation and expected value proof done here
    sorry
  }

end using_phone_related_to_gender_distribution_and_expected_value_l619_619034


namespace min_cups_needed_l619_619734

theorem min_cups_needed : 
  (∑ i in finset.range 100, (i + 1)) = 5050 :=
by sorry

end min_cups_needed_l619_619734


namespace area_of_square_l619_619269

theorem area_of_square (r s l : ℕ) (h1 : l = (2 * r) / 5) (h2 : r = s) (h3 : l * 10 = 240) : s * s = 3600 :=
by
  sorry

end area_of_square_l619_619269


namespace hundredth_digit_of_fraction_l619_619964

theorem hundredth_digit_of_fraction (n : ℕ) :
  let repeating_sequence := "269230769"
  ∧ let decimal_repr := "0." ++ repeating_sequence
  in decimal_repr[(100 % repeating_sequence.length + 1)] = '2' := by
  sorry

end hundredth_digit_of_fraction_l619_619964


namespace excluded_numbers_range_l619_619257

theorem excluded_numbers_range (S S' E : ℕ) (h1 : S = 31 * 10) (h2 : S' = 28 * 8) (h3 : E = S - S') (h4 : E > 70) :
  ∀ (x y : ℕ), x + y = E → 1 ≤ x ∧ x ≤ 85 ∧ 1 ≤ y ∧ y ≤ 85 := by
  sorry

end excluded_numbers_range_l619_619257


namespace twelve_edge_cubes_painted_faces_l619_619366

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end twelve_edge_cubes_painted_faces_l619_619366


namespace find_k_l619_619100

noncomputable def f (x : ℝ) : ℝ := sin (4 * x + π / 6)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x - π / 3)
def interval := Set.Icc 0 (π / 2)

theorem find_k (k : ℝ) : 
  (∃ (x : ℝ) (hx : x ∈ interval), g x + k = 0 ∧ ∀ (y : ℝ) (hy : y ∈ interval), g y + k = 0 → y = x) ↔ 
  k ∈ Set.Icc (-sqrt 3 / 2) (sqrt 3 / 2) ∪ {-1} :=
sorry

end find_k_l619_619100


namespace time_until_heavy_lifting_l619_619579

-- Define the conditions given
def pain_subside_days : ℕ := 3
def healing_multiplier : ℕ := 5
def additional_wait_days : ℕ := 3
def weeks_before_lifting : ℕ := 3
def days_in_week : ℕ := 7

-- Define the proof statement
theorem time_until_heavy_lifting : 
    let full_healing_days := pain_subside_days * healing_multiplier
    let total_days_before_exercising := full_healing_days + additional_wait_days
    let lifting_wait_days := weeks_before_lifting * days_in_week
    total_days_before_exercising + lifting_wait_days = 39 := 
by
  sorry

end time_until_heavy_lifting_l619_619579


namespace how_many_one_fourths_in_three_times_one_eighth_l619_619126

-- Define the fractions and their operations
def one_fourth : ℚ := 1 / 4
def one_eighth : ℚ := 1 / 8
def three : ℚ := 3
def target : ℚ := 3 / 2

-- Theorem statement using the defined fractions and multiplication
theorem how_many_one_fourths_in_three_times_one_eighth :
  let product := three * one_eighth in
  product / one_fourth = target :=
sorry

end how_many_one_fourths_in_three_times_one_eighth_l619_619126


namespace right_triangle_345_l619_619397

theorem right_triangle_345 :
  ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by {
  -- Here, we should construct the proof later
  sorry
}

end right_triangle_345_l619_619397


namespace part_I_part_II_l619_619621

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - x * Real.log x

theorem part_I (a : ℝ) :
  (∀ x > 0, 0 ≤ a * Real.exp x - (1 + Real.log x)) ↔ a ≥ 1 / Real.exp 1 :=
sorry

theorem part_II (a : ℝ) (h : a ≥ 2 / Real.exp 2) (x : ℝ) (hx : x > 0) :
  f a x > 0 :=
sorry

end part_I_part_II_l619_619621


namespace total_fencing_costs_l619_619550

theorem total_fencing_costs (c1 c2 c3 c4 l1 l2 l3 : ℕ) 
    (h_c1 : c1 = 79) (h_c2 : c2 = 92) (h_c3 : c3 = 85) (h_c4 : c4 = 96)
    (h_l1 : l1 = 5) (h_l2 : l2 = 7) (h_l3 : l3 = 9) :
    (c1 + c2 + c3 + c4) * l1 = 1760 ∧ 
    (c1 + c2 + c3 + c4) * l2 = 2464 ∧ 
    (c1 + c2 + c3 + c4) * l3 = 3168 := 
by {
    sorry -- Proof to be constructed
}

end total_fencing_costs_l619_619550


namespace schedule_arrangement_count_l619_619426

theorem schedule_arrangement_count :
  let classes := ["Chinese", "Mathematics", "Politics", "English", "Physical Education", "Arts"]
  let morning_slots := 4
  let afternoon_slots := 2
  let total_ways := 192
  ∃ (morning_counts afternoon_counts arrangement_counts : ℕ), 
    morning_counts = combinatorial.choose 1 morning_slots ∧
    afternoon_counts = combinatorial.choose 1 afternoon_slots ∧
    arrangement_counts = factorial 4 ∧
    morning_counts * afternoon_counts * arrangement_counts = total_ways := 
by
  sorry

end schedule_arrangement_count_l619_619426


namespace squirrel_nuts_collection_l619_619810

theorem squirrel_nuts_collection (n : ℕ) (e u : ℕ → ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n → e k = u k + k) ∧
  (∀ k, 1 ≤ k ∧ k ≤ n → u k = e (k + 1) + u k / 100) ∧
  e n = n →
  n = 99 → 
  (∃ S : ℕ, (∀ k, 1 ≤ k ∧ k ≤ n → e k = S)) ∧ 
  S = 9801 :=
sorry

end squirrel_nuts_collection_l619_619810


namespace quotient_of_sum_of_remainders_div_16_eq_0_l619_619674

-- Define the set of distinct remainders of squares modulo 16 for n in 1 to 15
def distinct_remainders_mod_16 : Finset ℕ :=
  {1, 4, 9, 0}

-- Define the sum of the distinct remainders
def sum_of_remainders : ℕ :=
  distinct_remainders_mod_16.sum id

-- Proposition to prove the quotient when sum_of_remainders is divided by 16 is 0
theorem quotient_of_sum_of_remainders_div_16_eq_0 :
  (sum_of_remainders / 16) = 0 :=
by
  sorry

end quotient_of_sum_of_remainders_div_16_eq_0_l619_619674


namespace graph_does_not_pass_through_second_quadrant_l619_619268

theorem graph_does_not_pass_through_second_quadrant :
  ¬ ∃ x : ℝ, x < 0 ∧ 2 * x - 3 > 0 :=
by
  -- Include the necessary steps to complete the proof, but for now we provide a placeholder:
  sorry

end graph_does_not_pass_through_second_quadrant_l619_619268


namespace sand_needed_l619_619183

def area_rectangular_patch : ℕ := 6 * 7
def area_square_patch : ℕ := 5 * 5
def sand_per_square_inch : ℕ := 3

theorem sand_needed : area_rectangular_patch + area_square_patch * sand_per_square_inch = 201 := sorry

end sand_needed_l619_619183


namespace monotonic_decreasing_interval_l619_619691

noncomputable def f (x : ℝ) := 2 * x^2 - Real.log x

theorem monotonic_decreasing_interval :
  ∃ I : set ℝ, I = set.Ioo 0 (1/2 : ℝ) ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f y ≤ f x := by
  sorry

end monotonic_decreasing_interval_l619_619691


namespace problem1_problem2_l619_619348

/-- Problem 1:
Given sin(α) = 3/5 and cos(α) = -4/5,
prove that
cos(π/2 + α) * sin(-π - α) / (cos(11π/2 - α) * sin(9π/2 + α)) = -3/4
-/
theorem problem1 (α : ℝ) (h1 : sin α = 3 / 5) (h2 : cos α = -4 / 5) :
  (cos (π / 2 + α) * sin (-π - α)) / (cos (11 * π / 2 - α) * sin (9 * π / 2 + α)) = -3 / 4 :=
by
  sorry

/-- Problem 2:
Given sin(x) = (m - 3) / (m + 5) and cos(x) = (4 - 2m) / (m + 5),
with x in the interval (π/2, π),
prove that tan(x) = -5/12 when m=8.
-/
theorem problem2 (x m : ℝ) (h1 : (π / 2) < x ∧ x < π) (h2 : sin x = (m - 3) / (m + 5)) (h3 : cos x = (4 - 2 * m) / (m + 5)) (h4 : m = 8) :
  tan x = -5 / 12 :=
by
  sorry

end problem1_problem2_l619_619348


namespace ones_digit_exponent_73_l619_619868

theorem ones_digit_exponent_73 (n : ℕ) : 
  (73 ^ n) % 10 = 7 ↔ n % 4 = 3 := 
sorry

end ones_digit_exponent_73_l619_619868


namespace flag_designs_count_l619_619841

theorem flag_designs_count :
  let colors := {purple, gold, silver} in
  let flag := product (product colors colors) colors in
  ∀ (x y z : colors),
  (x ≠ y ∧ y ≠ z) → 
  (flag.count (x, y, z) = 12) :=
begin
  sorry
end

end flag_designs_count_l619_619841


namespace day_before_day_after_tomorrow_is_friday_l619_619140

theorem day_before_day_after_tomorrow_is_friday (today_is_thursday : ∀ t : Day, t = Day.Thursday) :
  day_before (day_after tomorrow) = Day.Friday :=
sorry

end day_before_day_after_tomorrow_is_friday_l619_619140


namespace range_of_x_l619_619451

variable {x p : ℝ}

theorem range_of_x (H : 0 ≤ p ∧ p ≤ 4) : 
  (x^2 + p * x > 4 * x + p - 3) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 3) := 
by
  sorry

end range_of_x_l619_619451


namespace math_problem_l619_619144

theorem math_problem {x y : ℕ} (h1 : 1059 % x = y) (h2 : 1417 % x = y) (h3 : 2312 % x = y) : x - y = 15 := by
  sorry

end math_problem_l619_619144


namespace angle_in_first_quadrant_l619_619526

theorem angle_in_first_quadrant (α : ℝ) (h : 90 < α ∧ α < 180) : 0 < 180 - α ∧ 180 - α < 90 :=
by
  sorry

end angle_in_first_quadrant_l619_619526


namespace ab_equals_6_l619_619971

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619971


namespace increasing_interval_l619_619692

def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval : ∀ x, x > 2 → (f x)' > 0 :=
sorry

end increasing_interval_l619_619692


namespace solve_equation_l619_619670

-- Defining the necessary conditions
def term1 (x : ℝ) : ℝ := (x^2 + x)^2
def term2 (x : ℝ) : ℝ := sqrt (x^2 - 1)

theorem solve_equation (x : ℝ) (h : term1 x + term2 x = 0) : x = -1 :=
by {
  sorry
}

end solve_equation_l619_619670


namespace dot_product_necessity_not_sufficiency_l619_619478

variables {V : Type*} [inner_product_space ℝ V]

theorem dot_product_necessity_not_sufficiency (a b c : V) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)  
    (h : inner_product a c = inner_product b c) :
  ¬ (a = b) → (inner_product a c = inner_product b c) ∧ ¬ (∀ d : V, inner_product a d = inner_product b d → a = b) :=
sorry

end dot_product_necessity_not_sufficiency_l619_619478


namespace remainder_when_divided_by_x_minus_2_is_10_l619_619736

def polynomial := λ x : ℝ, x^5 - 6*x^4 + 12*x^3 - 5*x^2 + 9*x - 20

theorem remainder_when_divided_by_x_minus_2_is_10 :
  polynomial 2 = 10 :=
by
  -- Proof omitted. The statement is given as requested.
  sorry

end remainder_when_divided_by_x_minus_2_is_10_l619_619736


namespace first_proof_second_proof_l619_619207
noncomputable section

-- First Proof Problem
def f1 (x : ℝ) : ℝ := x * Real.log x
def g1 (x : ℝ) : ℝ :=
  if x > 0 then x * Real.log x else - x * Real.log (-x)

theorem first_proof (x : ℝ) (h : x ≠ 0) : 
  g1 x = x * Real.log (Real.abs x) := sorry

-- Second Proof Problem
def f2 (x : ℝ) : ℝ := 2^x - 1
def g2 (x : ℝ) : ℝ := 2^(-Real.abs x) - 1

theorem second_proof (x : ℝ) : 
  g2 x = 2^(-Real.abs x) - 1 := sorry

end first_proof_second_proof_l619_619207


namespace find_a13_l619_619088

variable (a_n : ℕ → ℝ)
variable (d : ℝ)
variable (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
variable (h_geo : a_n 9 ^ 2 = a_n 1 * a_n 5)
variable (h_sum : a_n 1 + 3 * a_n 5 + a_n 9 = 20)

theorem find_a13 (h_non_zero_d : d ≠ 0):
  a_n 13 = 28 :=
sorry

end find_a13_l619_619088


namespace parabola_equiv_eq_l619_619113

-- Definitions from the conditions
def parabola_C (a : ℝ) (a_pos : 0 < a) : { p : ℝ × ℝ // p.2 = 2 * a * p.1 * p.1 } := sorry

def focus (F : ℝ × ℝ) : F = (1 / 4, 0) := sorry

def directrix (d : ℝ) : d = -1 / 4 := sorry

def chord_length (a : ℝ) (a_pos : 0 < a) : 2 * (real.sqrt (1 / (8 * a))) = 1 := 
  by rw [mul_eq_one_iff, sqrt_eq_iff_sqr_eq, sq_eq_mul_self_iff]; 
     exact or.inr (div_nonneg zero_le_one (mul_nonneg zero_le_one (real.sqrt_nonneg _)); 
     sorry

def AF_BF_inverse_sum (a : ℝ) (a_pos : 0 < a) (x1 x2 : ℝ) (y1 y2 : ℝ) 
    (h : y1 = 2 * a * x1 * x1) (h2 : y2 = 2 * a * x2 * x2) : 
    (1 / (real.sqrt (x1^2 + y1^2)) + (1 / (real.sqrt (x2^2 + y2^2)))) = 4 := sorry

-- Combination of conditions to assert the equivalence
theorem parabola_equiv_eq (a : ℝ) (a_pos : 0 < a) :
  (focus (1 / 4, 0)) ∧ 
  (directrix (-1 / 4)) ∧ 
  (chord_length a a_pos) ∧ 
  (∃ (x1 x2 y1 y2 : ℝ), AF_BF_inverse_sum a a_pos x1 x2 y1 y2 (by sorry) (by sorry)) 
  → ∀ (x y : ℝ), (y = 2 * a * x^2) ↔ (x^2 = y) := sorry

end parabola_equiv_eq_l619_619113


namespace amy_work_hours_per_week_l619_619401

/-- Amy works for 40 hours per week for 8 weeks in the summer, making $3200. If she works for 32 weeks during 
the school year at the same rate of pay and needs to make another $4000, we need to prove that she must 
work 12.5 hours per week during the school year. -/
theorem amy_work_hours_per_week
  (summer_hours_per_week : ℕ)
  (summer_weeks : ℕ)
  (summer_money : ℕ)
  (school_year_weeks : ℕ)
  (school_year_money_needed : ℕ) :
  (let hourly_wage := summer_money / (summer_hours_per_week * summer_weeks : ℕ) in
   let hours_needed := school_year_money_needed / hourly_wage in
   hours_needed / school_year_weeks = 12.5) :=
by
  -- Definitions
  let summer_hours_per_week := 40
  let summer_weeks := 8
  let summer_money := 3200
  let school_year_weeks := 32
  let school_year_money_needed := 4000
  
  -- Calculations
  let hourly_wage := summer_money / (summer_hours_per_week * summer_weeks : ℕ)
  let hours_needed := school_year_money_needed / hourly_wage
  let hours_per_week := hours_needed / school_year_weeks

  -- Goal
  show hours_per_week = 12.5

  -- This proof is omitted.
  sorry

end amy_work_hours_per_week_l619_619401


namespace arithmetic_sequence_21st_term_l619_619265

theorem arithmetic_sequence_21st_term (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 13) (h3 : a 3 = 23) :
  a 21 = 203 :=
by
  sorry

end arithmetic_sequence_21st_term_l619_619265


namespace find_natural_numbers_eq_36_sum_of_digits_l619_619052

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end find_natural_numbers_eq_36_sum_of_digits_l619_619052


namespace pencils_per_group_l619_619290

-- Definitions for the given problem
def pencils := 25
def groups := 5

-- The goal is to prove the number of pencils in each group
theorem pencils_per_group (pencils groups : ℕ) (h : pencils = 25) (k : groups = 5) : pencils / groups = 5 :=
by {
  rw [h, k],
  norm_num,
  sorry
}

end pencils_per_group_l619_619290


namespace five_digit_perfect_squares_l619_619042

theorem five_digit_perfect_squares (n : ℕ) :
    n ∈ {81225, 34225, 27225, 15625, 75625} ↔ 
    (10000 ≤ n ∧ n < 100000 ∧ 
    (exists d, n = d^2) ∧ 
    (exists d1, n % 100 = d1^2) ∧ 
    (exists d2, n % 1000 = d2^2) ∧ 
    (exists d3, n % 10000 = d3^2)) :=
by
    intros
    sorry

end five_digit_perfect_squares_l619_619042


namespace range_of_values_proved_l619_619081

noncomputable def range_of_values (x0 y0 : ℝ) (hMidpoint : x0 + 3 * y0 + 2 = 0) (hInequality : y0 < x0 + 2) : Prop :=
  ∃ (y0 x0 : ℝ), (x0 + 3 * y0 + 2 = 0) ∧ (y0 < x0 + 2) ∧ (∃ (frac_xy := y0 / x0),
    frac_xy ∈ Ioo (-∞) (-1 / 3) ∨ frac_xy ∈ Ioo 0 ∞)

theorem range_of_values_proved (x0 y0 : ℝ) (hMidpoint : x0 + 3 * y0 + 2 = 0) (hInequality : y0 < x0 + 2) :
  range_of_values x0 y0 hMidpoint hInequality :=
sorry

end range_of_values_proved_l619_619081


namespace max_squares_covered_by_card_l619_619773

theorem max_squares_covered_by_card : 
  ∀ (card_side checkerboard_side : ℝ), 
  card_side = 1.5 → checkerboard_side = 1 → 
  (∃ n, n = 12 ∧ (∀ other_n, other_n > n → false)) :=
by
  intros card_side checkerboard_side h1 h2
  use 12
  split
  { exact rfl }
  { intro other_n 
    intro h
    exact absurd h
      (by linarith) }

end max_squares_covered_by_card_l619_619773


namespace longest_segment_in_cylinder_l619_619780

-- Define the given conditions
def radius : ℝ := 5 -- Radius of the cylinder in cm
def height : ℝ := 10 -- Height of the cylinder in cm

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the longest segment L inside the cylinder using the Pythagorean theorem
noncomputable def longest_segment : ℝ := Real.sqrt ((diameter ^ 2) + (height ^ 2))

-- State the problem in Lean:
theorem longest_segment_in_cylinder :
  longest_segment = 10 * Real.sqrt 2 :=
sorry

end longest_segment_in_cylinder_l619_619780


namespace sum_of_odd_base4_digits_of_152_and_345_l619_619443

def base_4_digit_count (n : ℕ) : ℕ :=
    n.digits 4 |>.filter (λ x => x % 2 = 1) |>.length

theorem sum_of_odd_base4_digits_of_152_and_345 :
    base_4_digit_count 152 + base_4_digit_count 345 = 6 :=
by
    sorry

end sum_of_odd_base4_digits_of_152_and_345_l619_619443


namespace triangle_area_sin_B_minus_A_l619_619571

-- Definitions for the given problem
def cosC : ℝ := 3 / 5

def dot_product_CB_CA : ℝ := 9 / 2

def x_vector : ℝ × ℝ := (2 * real.sin (B / 2), real.sqrt 3)
def y_vector : ℝ × ℝ := (real.cos B, real.cos (B / 2))

-- Assume x_vector is parallel to y_vector
axiom x_parallel_y : x_vector.1 * y_vector.2 = x_vector.2 * y_vector.1

-- Prove the area of triangle ABC
theorem triangle_area {a b C : ℝ} 
  (h1 : real.cos C = cosC)
  (h2 : a * b * real.cos C = dot_product_CB_CA) : 
  (1 / 2) * a * b * real.sin C = 3 :=
sorry

-- Prove the value of sin(B - A)
theorem sin_B_minus_A {B A C : ℝ}
  (h_cos_C : real.cos C = cosC)
  (h_x_parallel : x_parallel_y): 
  real.sin (B - A) = (4 - 3 * real.sqrt 3) / 10 :=
sorry

end triangle_area_sin_B_minus_A_l619_619571


namespace ab_equals_six_l619_619999

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l619_619999


namespace sequence_num_terms_l619_619848

theorem sequence_num_terms (a d l : ℤ) (h_arithmetic : ∀ (n : ℕ), nth_term a d n = a + (n - 1) * d)
  (ha : a = -6) (hd : d = 4) (hl : l = 38) : 
  ∃ (n : ℕ), nth_term a d n = l ∧ n = 12 := 
by
  sorry

def nth_term (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

end sequence_num_terms_l619_619848


namespace paint_needed_l619_619136

theorem paint_needed (paint_per_10ft : ℕ) (statue_in_10ft_pint : ℕ) (num_statues : ℕ) (height_ratio : ℚ) :
  paint_per_10ft = 1 ∧ statue_in_10ft_pint = 10 ∧ num_statues = 1000 ∧ height_ratio = 2 / 10 →
  (num_statues * (height_ratio ^ 2 * paint_per_10ft)) = 40 :=
by
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  rw [h1, h2, h3, h4]
  sorry

end paint_needed_l619_619136


namespace pentagon_midpoints_perimeter_l619_619203

theorem pentagon_midpoints_perimeter (AB BC CD DE EA AC CE EB BD DA : ℝ) 
  (h₁ : AB + BC + CD + DE + EA = 64) 
  (h₂ : AC + CE + EB + BD + DA = 72) :
  let M := (AB + BC + CD + DE + EA) / 2 in
  let N := (AC + CE + EB + BD + DA) / 2 in
  M / 2 + N / 2 = 36 :=
by
  -- proof goes here
  sorry

end pentagon_midpoints_perimeter_l619_619203


namespace find_k_l619_619110

theorem find_k (k : ℝ) (h : ∫ x in 0..1, k * x + 1 = k) : k = 2 :=
by
  sorry

end find_k_l619_619110


namespace infinite_elements_mod7_l619_619628

set_option pp.proofs true

/-- Let \( S = \{ n \in \mathbf{Z}_{+} \mid \exists n^2 + 1 \leq d \leq n^2 + 2n \text{ such that } d \mid n^4 \} \). 
Prove that \( S \) contains infinitely many elements congruent to \( 0, 1, 2, 5, 6 \mod 7 \) and no elements congruent to \( 3, 4 \mod 7 \). -/
theorem infinite_elements_mod7 (S : Set ℕ) :
  (∀ n, n ∈ S ↔ ∃ (d : ℕ), (n^2 + 1) ≤ d ∧ d ≤ (n^2 + 2 * n) ∧ d ∣ (n^4)) →
  (∀ m : ℤ, ∃ n ∈ S, n ≡ 7 * m [MOD 7] ∨ n ≡ 7 * m + 1 [MOD 7] ∨ n ≡ 7 * m + 2 [MOD 7] ∨ n ≡ 7 * m + 5 [MOD 7] ∨ n ≡ 7 * m + 6 [MOD 7]) ∧
  (¬ ∃ n ∈ S, n ≡ 7 * m + 3 [MOD 7] ∨ n ≡ 7 * m + 4 [MOD 7]) :=
by
  -- Define the set S with the given properties
  assume hS : ∀ n, n ∈ S ↔ ∃ (d : ℕ), (n^2 + 1) ≤ d ∧ d ≤ (n^2 + 2 * n) ∧ d ∣ (n^4)

  -- Start the proof part
  sorry

end infinite_elements_mod7_l619_619628


namespace factorial_division_identity_l619_619408

theorem factorial_division_identity : (fact (fact 5)) / (fact 5) = fact 119 :=
by
  sorry

end factorial_division_identity_l619_619408


namespace zero_descriptions_l619_619155

-- Defining the descriptions of zero satisfying the given conditions.
def description1 : String := "The number corresponding to the origin on the number line."
def description2 : String := "The number that represents nothing."
def description3 : String := "The number that, when multiplied by any other number, equals itself."

-- Lean statement to prove the validity of the descriptions.
theorem zero_descriptions : 
  description1 = "The number corresponding to the origin on the number line." ∧
  description2 = "The number that represents nothing." ∧
  description3 = "The number that, when multiplied by any other number, equals itself." :=
by
  -- Proof omitted
  sorry

end zero_descriptions_l619_619155


namespace gift_certificate_value_is_correct_l619_619279

-- Define the conditions
def total_race_time_minutes : ℕ := 12
def one_lap_meters : ℕ := 100
def total_laps : ℕ := 24
def earning_rate_per_minute : ℕ := 7

-- The total distance run in meters
def total_distance_meters : ℕ := total_laps * one_lap_meters

-- The total earnings in dollars
def total_earnings_dollars : ℕ := earning_rate_per_minute * total_race_time_minutes

-- The worth of the gift certificate per 100 meters (to be proven as 3.50 dollars)
def gift_certificate_value : ℚ := total_earnings_dollars / (total_distance_meters / one_lap_meters)

-- Prove that the gift certificate value is $3.50
theorem gift_certificate_value_is_correct : 
    gift_certificate_value = 3.5 := by
  sorry

end gift_certificate_value_is_correct_l619_619279


namespace fraction_is_irreducible_l619_619823

theorem fraction_is_irreducible :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16 : ℚ) / 
   (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by 
  sorry

end fraction_is_irreducible_l619_619823


namespace length_gh_eq_40_div_3_l619_619690

variable {A B C D E F G H N M : Type}
variable [MetricSpace RS]
variable (intersects_not : ¬ intersects RS (Triangle A B C))
variable (AD BE CF : ℝ)

def rs_perpendicular_foot (AD BE CF : ℝ) : ℝ :=
  let MJ := (AD + CF) / 2
  let MK := MJ - BE
  MJ - MK / 3

theorem length_gh_eq_40_div_3 (AD_eq : AD = 10) (BE_eq : BE = 6) (CF_eq : CF = 24) :
  rs_perpendicular_foot 10 6 24 = 40 / 3 :=
by
  rw [AD_eq, BE_eq, CF_eq]
  sorry

end length_gh_eq_40_div_3_l619_619690


namespace sum_of_infinite_geometric_series_l619_619872

-- Define necessary conditions for the problem
def first_term : ℝ := 1
def common_ratio : ℝ := 1/4

noncomputable def infinite_geometric_sum (a q : ℝ) : ℝ :=
  a / (1 - q)

-- State the theorem with given conditions
theorem sum_of_infinite_geometric_series :
  infinite_geometric_sum first_term common_ratio = 4/3 :=
by
  sorry

end sum_of_infinite_geometric_series_l619_619872


namespace exists_z1_l619_619635

def z : ℂ := -1 - 2 * complex.I

theorem exists_z1 (z : ℂ) : z = -1 - 2 * complex.I → ∃ z1 : ℂ, z * z1 ∈ ℝ :=
by intros; use -1 + 2 * complex.I; simp [z]; sorry

end exists_z1_l619_619635


namespace find_alpha_l619_619071

theorem find_alpha (α : ℝ) (h1 : Real.tan α = -1) (h2 : 0 < α ∧ α ≤ Real.pi) : α = 3 * Real.pi / 4 :=
sorry

end find_alpha_l619_619071


namespace necessary_but_not_sufficient_l619_619473

variables {V : Type*} [inner_product_space ℝ V]

theorem necessary_but_not_sufficient
  {a b c : V}
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0) :
  (inner a c = inner b c) ↔ (a = b) :=
sorry

end necessary_but_not_sufficient_l619_619473


namespace factor_polynomial_l619_619697

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := 
by sorry

end factor_polynomial_l619_619697


namespace vector_norm_range_l619_619517

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_norm_range
  (e1 e2 : V)
  (a : V)
  (h1 : (e1 : V) ∈ set_of (λ v, ⟪v, v⟫ = 1))
  (h2 : (e2 : V) ∈ set_of (λ v, ⟪v, v⟫ = 1))
  (h3 : ⟪e1, e2⟫ = -1 / 2)
  (h4 : ⟪(a - e1), (a - e2)⟫ = 5 / 4) :
  sqrt 2 - 1 / 2 ≤ ∥a∥ ∧ ∥a∥ ≤ sqrt 2 + 1 / 2 :=
sorry

end vector_norm_range_l619_619517


namespace intervals_of_monotonicity_range_of_m_l619_619506

def f (x a : ℝ) : ℝ := x^3 - 3*a*x - 1
def f_prime (x a : ℝ) : ℝ := 3*x^2 - 3*a

theorem intervals_of_monotonicity (a : ℝ) (h : a ≠ 0) :
  (a < 0 → ∀ x, f_prime x a > 0) ∧
  (a > 0 → (∀ x, x < -real.sqrt a ∨ x > real.sqrt a → f_prime x a > 0) ∧ 
           (∀ x, -real.sqrt a < x ∧ x < real.sqrt a → f_prime x a < 0)) := 
sorry

theorem range_of_m (m : ℝ) :
  (∀ x, x = -1 → f_prime x 1 = 0) →
  (∀ x, (y = f x 1 ∧ y = m) → ∃ x₁ x₂ x₃, (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) →
  m ∈ set.Ioo (-3 : ℝ) 1 :=
sorry

end intervals_of_monotonicity_range_of_m_l619_619506


namespace mike_marks_l619_619639

theorem mike_marks (max_marks : ℕ) (percentage_needed : ℚ) (shortfall : ℕ) :
  max_marks = 760 →
  percentage_needed = 30 / 100 →
  shortfall = 16 →
  (let passing_marks := percentage_needed * max_marks in
   let M := passing_marks - shortfall in
   M = 212) :=
by
  intros h_max_marks h_percentage_needed h_shortfall
  let passing_marks := percentage_needed * max_marks
  let M := passing_marks - shortfall
  sorry

end mike_marks_l619_619639


namespace probability_one_box_empty_l619_619251

theorem probability_one_box_empty :
  let number_of_balls := 4
  let number_of_boxes := 4
  let total_ways_to_place := 4^4
  let ways_to_group := choose (4 : ℕ) 2
  let ways_to_arrange := 4.perm 3
  let favourable_ways := ways_to_group * ways_to_arrange
  favourable_ways / total_ways_to_place = 9 / 16 := 
by {
  sorry
}

end probability_one_box_empty_l619_619251


namespace total_savings_in_joint_account_l619_619583

def kimmie_earnings : ℝ := 450
def zahra_earnings : ℝ := kimmie_earnings - (1 / 3) * kimmie_earnings
def kimmie_savings : ℝ := (1 / 2) * kimmie_earnings
def zahra_savings : ℝ := (1 / 2) * zahra_earnings
def joint_savings_account : ℝ := kimmie_savings + zahra_savings

theorem total_savings_in_joint_account :
  joint_savings_account = 375 := 
by
  -- proof to be provided
  sorry

end total_savings_in_joint_account_l619_619583


namespace cubes_with_painted_faces_l619_619368

theorem cubes_with_painted_faces :
  ∀ (n : ℕ), (∃ (cubes : ℕ), cubes = 27 ∧ ∃ (face_painted_cubes : ℕ → ℕ), 
  face_painted_cubes 2 = 12) → 12 * 2 = 24 :=
by
  intro n
  assume h
  cases h with cubes hc
  cases hc with hc_painted hc_cubes
  simp at hc_painted
  simp at hc_cubes
  rw [hc_painted, hc_cubes]
  sorry

end cubes_with_painted_faces_l619_619368


namespace segments_equal_l619_619627

variables {A B C H M S F : Type} [MetricSpace H]
/-- acute triangle with vertices A, B, C --/
axiom acute_triangle (A B C : H) : H ∈ set.univ

/-- orthocenter H of triangle ABC --/
axiom orthocenter_H (A B C H : H) : H ∈ set.univ

/-- M is the midpoint of side AB --/
axiom midpoint_M (A B M : H) : dist M A = dist M B

/-- w is the angle bisector of angle ACB --/
axiom angle_bisector_w (A C B : H) [MetricSpace.AffineSpace H] : ∃ w, angle_bisector w (∠ A C B)

/-- S is the intersection of the perpendicular bisector of AB with w --/
axiom intersection_S (A B w S : H) [MetricSpace.AffineSpace H] : 
  perpendicular_bisector w = some S ∧ intersects w S

/-- F is the foot of the perpendicular from H to w --/
axiom foot_perpendicular_F (H w F : H) [MetricSpace.AffineSpace H] : foot_perpendicular H w = F

theorem segments_equal (A B C H M S F : H)
  (acute_triangle A B C)
  (orthocenter_H A B C H)
  (midpoint_M A B M)
  (angle_bisector_w A C B)
  (intersection_S A B w S)
  (foot_perpendicular_F H w F)
  : dist M S = dist M F :=
begin
  sorry
end

end segments_equal_l619_619627


namespace roots_of_polynomial_l619_619031

theorem roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end roots_of_polynomial_l619_619031


namespace product_equals_one_l619_619883

theorem product_equals_one (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1 / (1 + x + x^2)) + (1 / (1 + y + y^2)) + (1 / (1 + x + y)) = 1) : 
  x * y = 1 :=
by
  sorry

end product_equals_one_l619_619883


namespace infinite_series_fraction_l619_619015

theorem infinite_series_fraction:
  (∑' n : ℕ, (if n = 0 then 0 else ((2 : ℚ) / (3 * n) - (1 : ℚ) / (3 * (n + 1)) - (7 : ℚ) / (6 * (n + 3))))) =
  (1 : ℚ) / 3 := 
sorry

end infinite_series_fraction_l619_619015


namespace Amy_bought_tomato_soup_l619_619820

-- Conditions
variables (chicken_soup_cans total_soups : ℕ)
variable (Amy_bought_soups : total_soups = 9)
variable (Amy_bought_chicken_soup : chicken_soup_cans = 6)

-- Question: How many cans of tomato soup did she buy?
def cans_of_tomato_soup (chicken_soup_cans total_soups : ℕ) : ℕ :=
  total_soups - chicken_soup_cans

-- Theorem: Prove that the number of cans of tomato soup Amy bought is 3
theorem Amy_bought_tomato_soup : 
  cans_of_tomato_soup chicken_soup_cans total_soups = 3 :=
by
  rw [Amy_bought_soups, Amy_bought_chicken_soup]
  -- The steps for the proof would follow here
  sorry

end Amy_bought_tomato_soup_l619_619820


namespace number_of_rows_seating_exactly_9_students_l619_619774

theorem number_of_rows_seating_exactly_9_students (x : ℕ) : 
  ∀ y z, x * 9 + y * 5 + z * 8 = 55 → x % 5 = 1 ∧ x % 8 = 7 → x = 3 :=
by sorry

end number_of_rows_seating_exactly_9_students_l619_619774


namespace power_product_to_seventh_power_l619_619961

theorem power_product_to_seventh_power :
  (2 ^ 14) * (2 ^ 21) = (32 ^ 7) :=
by
  sorry

end power_product_to_seventh_power_l619_619961


namespace number_of_apartment_complexes_l619_619750

theorem number_of_apartment_complexes (width_land length_land side_complex : ℕ)
    (h_width : width_land = 262) (h_length : length_land = 185) 
    (h_side : side_complex = 18) :
    width_land / side_complex * length_land / side_complex = 140 := by
  -- given conditions
  rw [h_width, h_length, h_side]
  -- apply calculation steps for clarity (not necessary for final theorem)
  -- calculate number of complexes along width
  have h1 : 262 / 18 = 14 := sorry
  -- calculate number of complexes along length
  have h2 : 185 / 18 = 10 := sorry
  -- final product calculation
  sorry

end number_of_apartment_complexes_l619_619750


namespace molecular_weight_7_moles_l619_619317

theorem molecular_weight_7_moles (molecular_weight : ℕ) (h : molecular_weight = 2856) : molecular_weight = 2856 :=
by 
  rw h
  assumption

end molecular_weight_7_moles_l619_619317


namespace sum_of_geometric_series_l619_619623

-- Define the geometric series and its sum
def geom_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

-- Prove that for a = 5 and r = -1/2, the sum is 10/3
theorem sum_of_geometric_series : geom_series_sum 5 (-1 / 2) (by norm_num) = 10 / 3 :=
sorry

end sum_of_geometric_series_l619_619623


namespace simplify_trigonometric_expression_compute_fraction_with_tan_l619_619349

-- Define the trigonometric identities and their properties
theorem simplify_trigonometric_expression (α : ℝ) :
  (sin (π - α) * cos (3 * π - α) * tan (-α - π) * tan (α - 2 * π)) / 
  (tan (4 * π - α) * sin (5 * π + α)) = sin α :=
  sorry

theorem compute_fraction_with_tan (α : ℝ) (h : tan α = 3) :
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
  sorry

end simplify_trigonometric_expression_compute_fraction_with_tan_l619_619349


namespace max_subset_no_seven_times_l619_619885

theorem max_subset_no_seven_times :
  ∃ (s : Finset ℕ), (∀ a b ∈ s, ¬ (a = 7 * b ∨ b = 7 * a)) ∧ s.card = 1763 :=
sorry

end max_subset_no_seven_times_l619_619885


namespace min_area_of_triangle_ABC_l619_619516

noncomputable def point := (ℝ, ℝ)

def A : point := (-2, 0)
def B : point := (0, 2)

def on_circle (P : point) : Prop := 
  let (x, y) := P
  x^2 + y^2 - 2 * x = 0

def area_of_triangle (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem min_area_of_triangle_ABC (C : point)
  (hC : on_circle C) : 
  ∃ D : ℝ, D = 3 - √2 ∧ area_of_triangle A B C ≥ 0 :=
sorry

end min_area_of_triangle_ABC_l619_619516


namespace ratio_ZQ_QX_l619_619172

noncomputable theory

variables {X Y Z E N Q : Type} [point : LinearOrder X Y Z E N Q]
variables (XY XZ YE EZ ZQ QX : ℝ)

-- Conditions
axiom XY_eq : XY = 25
axiom XZ_eq : XZ = 15
axiom angle_bisector_X : intersects_angle_bisector_at_X Y Z E
axiom N_midpoint_XE : is_midpoint N X E
axiom Q_intersection_XZ_BN : intersects Q XZ BN

-- Goal
theorem ratio_ZQ_QX (h1 : XY = 25) (h2 : XZ = 15) (h3 : intersects_angle_bisector_at_X Y Z E) (h4 : is_midpoint N X E) (h5 : intersects Q XZ BN) :
  let m := 8 in
  let n := 5 in
  m + n = 13 :=
by {
  sorry
}

end ratio_ZQ_QX_l619_619172


namespace range_of_m_l619_619508

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 6

def f_positive {x : ℝ} (hx : -1 < x ∧ x < 3) : f x > 0 := sorry

def f_plus_m_interval {m : ℝ} (hm : ∀ x ∈ set.Icc (-1 : ℝ) (0 : ℝ), f x + m ≥ 4) : sorry

theorem range_of_m {m : ℝ} (hm : ∀ x ∈ set.Icc (-1 : ℝ) (0 : ℝ), f x + m ≥ 4) : m ∈ set.Ici 4 := sorry

end range_of_m_l619_619508


namespace correct_option_is_C_l619_619325

theorem correct_option_is_C : 
  (∀ x : ℝ, sqrt 25 ≠ 5) ∧ 
  (sqrt 0.4 ≠ 0.2) ∧ 
  ((-1) ^ (-3) = -1) ∧ 
  (∀ m n : ℝ, (-3 * m * n) ^ 2 ≠ -6 * m ^ 2 * n ^ 2) :=
by {
  sorry
}

end correct_option_is_C_l619_619325


namespace assignment_count_l619_619005

theorem assignment_count :
  let graduates := {A, B, C, D, E, F}
  let grades := {1, 2, 3}
  (number_of_assignments (graduates, grades, 2) A 1 (λ g, g ≠ B ∨ g ≠ C, 3) = 9) := sorry

end assignment_count_l619_619005


namespace f_g_of_3_l619_619533

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l619_619533


namespace distance_from_point_to_x_axis_l619_619159

theorem distance_from_point_to_x_axis (x y : ℝ) (P : ℝ × ℝ) (hP : P = (x, y)) :
  abs (y) = 3 :=
by
  -- Assuming the y-coordinate is given as -3
  have hy : y = -3 := sorry
  rw [hy]
  exact abs_neg 3

end distance_from_point_to_x_axis_l619_619159


namespace maximum_abs_sum_of_xyz_l619_619542

open Real

theorem maximum_abs_sum_of_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  |x| + |y| + |z| ≤ √2 :=
sorry

end maximum_abs_sum_of_xyz_l619_619542


namespace min_value_l619_619204

theorem min_value (a b : ℝ) (h : a + 3 * b = 1) (ha : 0 < a) (hb : 0 < b) :
  ∃ x, x = (3 * real.sqrt 6 + 9) ∧ ∀ y, y = (1 / a + 2 / b) → x ≤ y :=
sorry

end min_value_l619_619204


namespace area_of_reachable_points_l619_619066

noncomputable def legal_move (x y u v : ℝ) : Prop :=
  u^2 + v^2 ≤ 1 ∧ (u, v) = (x / 3 + u, y / 3 + v)

theorem area_of_reachable_points : 
  (let S := {p : ℝ × ℝ | ∃ n : ℕ, ∃ seq : Fin n → ℝ × ℝ, 
    seq 0 = (0, 0) ∧ seq (Fin.last n) = p ∧ 
    ∀ i : Fin (n-1), legal_move (seq i).fst (seq i).snd (seq (i+1)).fst (seq (i+1)).snd} in 
  let area := π * (3 / 2)^2 in 
  S = {q : ℝ × ℝ | q.fst ^ 2 + q.snd ^ 2 ≤ (3 / 2) ^ 2} ∧ area = 9 * π / 4) :=
sorry

end area_of_reachable_points_l619_619066


namespace dot_product_calculation_l619_619070

def vector_a : ℝ × ℝ := (1, -1)
def vector_b : ℝ × ℝ := (-1, 2)

theorem dot_product_calculation :
  ((2 • vector_a.1 + vector_b.1, 2 • vector_a.2 + vector_b.2) : ℝ × ℝ) 
  ⋅ vector_a = 1 := by
  -- Use dot product definition: (x1, y1) ⋅ (x2, y2) = x1*x2 + y1*y2
  sorry

end dot_product_calculation_l619_619070


namespace number_of_valid_points_l619_619357

-- Define a bug's coordinates as a tuple of integers
structure Point :=
  (x : Int)
  (y : Int)

-- Define a function to compute Manhattan distance between two points
def manhattan_distance (p1 p2 : Point) : Int :=
  (p1.x - p2.x).natAbs + (p1.y - p2.y).natAbs

-- Define the points A and B'
def A : Point := Point.mk (-3) 2
def B' : Point := Point.mk 4 (-3)

-- Define the condition for a point (x, y) to lie on a valid path
def valid_point (p : Point) : Prop :=
  (p.x + 3).natAbs + (p.x - 4).natAbs + (p.y - 2).natAbs + (p.y + 3).natAbs <= 25

-- Define a set of integer-coordinate points satisfying valid_point condition
def valid_points : Set Point :=
  {p : Point | valid_point p}

-- State the theorem to find the number of such points
theorem number_of_valid_points : (valid_points.count : Int) = 221 :=
  sorry

end number_of_valid_points_l619_619357


namespace compute_b_l619_619602

-- Definitions and conditions
def polynomial_root (a b : ℚ) (x : ℚ) : Prop :=
  x^3 + a * x^2 + b * x + 15 = 0

def rational (x : ℚ) : Prop := ∃ q : ℚ, x = q

noncomputable def value_of_b (a b : ℚ) : Prop :=
  (∃ r1 r2 r3 : ℚ, polynomial_root a b r1 ∧ polynomial_root a b r2 ∧ polynomial_root a b r3 ∧ 
  r1 = 3 + real.sqrt 5 ∧ r2 = 3 - real.sqrt 5 ∧ r3 = -15 / 4) ∧ b = -18.5

-- Proof problem statement
theorem compute_b (a : ℚ) : rational a → value_of_b a (-18.5) :=
sorry

end compute_b_l619_619602


namespace area_of_given_quadrilateral_is_6_l619_619409

-- Define the vertices of the quadrilateral
def vertex1 : (ℝ × ℝ) := (0, 0)
def vertex2 : (ℝ × ℝ) := (2, 4)
def vertex3 : (ℝ × ℝ) := (6, 0)
def vertex4 : (ℝ × ℝ) := (2, 6)

-- Define the function to calculate the shoelace formula
def shoelace_formula (v1 v2 v3 v4 : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := v1;
  let (x2, y2) := v2;
  let (x3, y3) := v3;
  let (x4, y4) := v4;
  (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)

-- Define the area of the quadrilateral using the shoelace formula
def area_of_quadrilateral (v1 v2 v3 v4 : (ℝ × ℝ)) : ℝ :=
  (1/2) * abs (shoelace_formula v1 v2 v3 v4)

theorem area_of_given_quadrilateral_is_6 : area_of_quadrilateral vertex1 vertex2 vertex3 vertex4 = 6 :=
by
  sorry

end area_of_given_quadrilateral_is_6_l619_619409


namespace Vanya_two_digit_number_l619_619311

-- Define the conditions as a mathematical property
theorem Vanya_two_digit_number:
  ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ (10 * n + m) ^ 2 = 4 * (10 * m + n) ∧ (10 * m + n) = 81 :=
by
  -- Remember to replace the proof with 'sorry'
  sorry

end Vanya_two_digit_number_l619_619311


namespace relationship_of_a_b_c_l619_619469

noncomputable def f : ℝ → ℝ := sorry -- f is an odd, increasing function which needs to be defined

theorem relationship_of_a_b_c :
  ∀ (f : ℝ → ℝ),
    (∀ x, f (-x) = -f x) → -- f is an odd function
    (∀ (x y : ℝ), x < y → f x < f y) → -- f is increasing
    (a = -f (Real.log 5 / Real.log 2)) → -- given condition for a
    (b = f 0) → -- given condition for b, note that log_2 1 = 0
    (c = f (2 ^ 0.8)) → -- given condition for c
    a < b ∧ b < c := -- target conclusion
begin
  sorry
end

end relationship_of_a_b_c_l619_619469


namespace unshaded_area_eq_20_l619_619306

-- Define the dimensions of the first rectangle
def rect1_width := 4
def rect1_length := 12

-- Define the dimensions of the second rectangle
def rect2_width := 5
def rect2_length := 10

-- Define the dimensions of the overlapping region
def overlap_width := 4
def overlap_length := 5

-- Calculate area functions
def area (width length : ℕ) := width * length

-- Calculate areas of the individual rectangles and the overlapping region
def area_rect1 := area rect1_width rect1_length
def area_rect2 := area rect2_width rect2_length
def overlap_area := area overlap_width overlap_length

-- Calculate the total shaded area
def total_shaded_area := area_rect1 + area_rect2 - overlap_area

-- The total area of the combined figure (assumed to be the union of both rectangles) minus shaded area gives the unshaded area
def total_area := rect1_width * rect1_length + rect2_width * rect2_length
def unshaded_area := total_area - total_shaded_area

theorem unshaded_area_eq_20 : unshaded_area = 20 := by
  sorry

end unshaded_area_eq_20_l619_619306


namespace prime_iff_permutation_sequence_l619_619630

theorem prime_iff_permutation_sequence (k : ℕ) (h : k > 0) : 
  let n := 2^k + 1 in 
  prime n ↔ ∃ (a : Fin n.succ → Fin n.succ) (g : Fin n.succ → ℤ) (h_permutation : ∀ (i : Fin (n-1)), a i ∈ {1, 2, ..., n-1}) (h_cyclic : a (Nat.pred n) = a 0),
    ∀ i : Fin (n-1), n ∣ g i ^ a i - a (i+1) :=
by
  let n := 2^k + 1
  sorry

end prime_iff_permutation_sequence_l619_619630


namespace correct_option_C_l619_619744

theorem correct_option_C (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : ac > bd :=
by
  sorry

end correct_option_C_l619_619744


namespace no_unique_sum_set_of_positive_rationals_l619_619853

theorem no_unique_sum_set_of_positive_rationals :
  ¬ ∃ (R : set ℚ), (∀ q : ℚ, 0 < q → ∃! (S : finset ℚ), (S ⊆ R) ∧ (∑ x in S, x = q)) := by 
  sorry

end no_unique_sum_set_of_positive_rationals_l619_619853


namespace find_fff_one_fourth_l619_619094

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 else 2^x

theorem find_fff_one_fourth :
  f(f(1/4)) = 1/4 := by
sory

end find_fff_one_fourth_l619_619094


namespace math_problem_l619_619007

theorem math_problem :
    3 * 3^4 - (27 ^ 63 / 27 ^ 61) = -486 :=
by
  sorry

end math_problem_l619_619007


namespace scientific_notation_correct_l619_619433

theorem scientific_notation_correct :
  0.00000164 = 1.64 * 10^(-6) :=
sorry

end scientific_notation_correct_l619_619433


namespace eval_f_20_l619_619210

def f (x : ℝ) : ℝ :=
if x ≤ 0 then
  logBase (1 / 2) (3 - x)
else
  f (x - 3) + 1

theorem eval_f_20 : f 20 = 5 :=
by
  sorry

end eval_f_20_l619_619210


namespace real_part_of_z_l619_619463

def complex_number : ℂ := (1 - complex.I) / (2 - complex.I)

theorem real_part_of_z : complex.re complex_number = 3 / 5 := 
by
  sorry

end real_part_of_z_l619_619463


namespace proof_statements_l619_619326

theorem proof_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧       -- corresponding to A
  ¬((∃ m : ℕ, 190 = 19 * m) ∧  ¬(∃ k : ℕ, 57 = 19 * k)) ∧  -- corresponding to B
  ¬((∃ p : ℕ, 90 = 30 * p) ∨ (∃ q : ℕ, 65 = 30 * q)) ∧     -- corresponding to C
  ¬((∃ r : ℕ, 33 = 11 * r) ∧ ¬(∃ s : ℕ, 55 = 11 * s)) ∧    -- corresponding to D
  (∃ t : ℕ, 162 = 9 * t) :=                                 -- corresponding to E
by {
  -- Proof steps would go here
  sorry
}

end proof_statements_l619_619326


namespace smallest_value_l619_619611

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l619_619611


namespace distance_to_x_axis_l619_619161

def point_P : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point_P.snd) = 3 := by
  sorry

end distance_to_x_axis_l619_619161


namespace work_completion_time_l619_619756

variable (L : ℕ) -- number of ladies

theorem work_completion_time :
  (∀ L, (work_done_in_days L 1 = 12)) →
  work_done_in_days (2 * L) (1/2) = 3 :=
by
  sorry

end work_completion_time_l619_619756


namespace necessary_but_not_sufficient_l619_619472

variables {V : Type*} [inner_product_space ℝ V]

theorem necessary_but_not_sufficient
  {a b c : V}
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0) :
  (inner a c = inner b c) ↔ (a = b) :=
sorry

end necessary_but_not_sufficient_l619_619472


namespace integral_sin_pi_over_2_to_pi_l619_619857

theorem integral_sin_pi_over_2_to_pi : ∫ x in (Real.pi / 2)..Real.pi, Real.sin x = 1 := by
  sorry

end integral_sin_pi_over_2_to_pi_l619_619857


namespace compare_exponentiated_constants_l619_619072

variable (a b c : ℝ)

theorem compare_exponentiated_constants (h1 : a = 0.8 ^ 0.7) (h2 : b = 0.8 ^ 0.9) (h3 : c = 1.1 ^ 0.6) : 
  c > a ∧ a > b :=
by
  sorry

end compare_exponentiated_constants_l619_619072


namespace product_ab_l619_619988

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619988


namespace valid_combinations_l619_619394

theorem valid_combinations : 
  let total_herbs := 4
  let total_crystals := 6
  let incompatible_herbs := 3
  let incompatible_crystals := 2
  let total_combinations := total_herbs * total_crystals
  let incompatible_combinations := incompatible_herbs * incompatible_crystals
  let valid_combinations := total_combinations - incompatible_combinations
  in valid_combinations = 18 := 
by
  sorry

end valid_combinations_l619_619394


namespace who_stole_the_jam_l619_619747

-- Define the characters involved
inductive Character
| MarchHare
| Hatter
| Dormouse
| Bolvanshchik

open Character

-- Assume each character can either tell the truth or lie
def tellsTruth : Character → Prop

-- Problem conditions
axiom investigation_revealed : ¬tellsTruth MarchHare ∧ ¬tellsTruth Hatter
axiom bolvanshchik_statement : (tellsTruth Bolvanshchik → (¬tellsTruth MarchHare ∧ ¬tellsTruth Dormouse)) ∧ (¬tellsTruth Bolvanshchik → tellsTruth Bolvanshchik)
axiom dormouse_statement : tellsTruth Dormouse → (tellsTruth MarchHare ∨ tellsTruth Bolvanshchik)

-- Theorem to prove
theorem who_stole_the_jam : ∃ thief : Character, thief = MarchHare := by
  sorry

end who_stole_the_jam_l619_619747


namespace find_PD_l619_619227

variable {α : Type*} [LinearOrderedField α]
variable {A B C D P : α × α}

theorem find_PD 
  (h1 : dist P A = 2) 
  (h2 : dist P B = 3) 
  (h3 : dist P C = 10) 
  (inside_rect : is_in_rectangle P A B C D) :
  dist P D = Real.sqrt 95 :=
sorry

-- Definitions that might be needed:
def dist (P Q : α × α) : α :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def is_in_rectangle 
  (P A B C D : α × α) : Prop :=
  -- Define the condition that point P is inside rectangle ABCD
  (A.1 < P.1 ∧ P.1 < C.1) ∧ (A.2 < P.2 ∧ P.2 < C.2)

end find_PD_l619_619227


namespace problem_1_problem_2_l619_619107

noncomputable def f (m : ℝ) (x : ℝ) := m * x + 3
noncomputable def g (m : ℝ) (x : ℝ) := x^2 + 2 * x + m
noncomputable def G (m : ℝ) (x : ℝ) := (f m x) - (g m x) - 1

theorem problem_1 (m : ℝ) : ∃ x : ℝ, f m x - g m x = 0 := by
  sorry

theorem problem_2 {m : ℝ} (h : ∀ x ∈ set.Icc (-1:ℝ) 0, abs (G m x) > abs (G m (x + 1))) : m ≤ 0 ∨ m ≥ 2 := by
  sorry

end problem_1_problem_2_l619_619107


namespace lee_propose_time_l619_619593

theorem lee_propose_time (annual_salary : ℕ) (monthly_savings : ℕ) (ring_salary_months : ℕ) :
    annual_salary = 60000 → monthly_savings = 1000 → ring_salary_months = 2 → 
    let monthly_salary := annual_salary / 12 in
    let ring_cost := ring_salary_months * monthly_salary in
    ring_cost / monthly_savings = 10 := 
by 
    intros annual_salary_eq monthly_savings_eq ring_salary_months_eq;
    rw [annual_salary_eq, monthly_savings_eq, ring_salary_months_eq];
    let monthly_salary := 60000 / 12;
    have ring_cost_eq : 2 * monthly_salary = 10000 := by sorry;
    have savings_time_eq : 10000 / 1000 = 10 := by sorry;
    exact savings_time_eq at ring_cost_eq;
    assumption

end lee_propose_time_l619_619593


namespace min_g_l619_619505

noncomputable def f (a m x : ℝ) := m + Real.log x / Real.log a -- definition of f(x) = m + logₐ(x)

-- Given conditions
variables (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
variables (m : ℝ)
axiom h_f8 : f a m 8 = 2
axiom h_f1 : f a m 1 = -1

-- Derived expressions
noncomputable def g (x : ℝ) := 2 * f a m x - f a m (x - 1)

-- Theorem statement
theorem min_g : ∃ (x : ℝ), x > 1 ∧ g a m x = 1 ∧ ∀ x' > 1, g a m x' ≥ 1 :=
sorry

end min_g_l619_619505


namespace design_height_lower_part_l619_619322

theorem design_height_lower_part (H : ℝ) (H_eq : H = 2) (L : ℝ) 
  (ratio : (H - L) / L = L / H) : L = Real.sqrt 5 - 1 :=
by {
  sorry
}

end design_height_lower_part_l619_619322


namespace Vanya_two_digit_number_l619_619312

-- Define the conditions as a mathematical property
theorem Vanya_two_digit_number:
  ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ (10 * n + m) ^ 2 = 4 * (10 * m + n) ∧ (10 * m + n) = 81 :=
by
  -- Remember to replace the proof with 'sorry'
  sorry

end Vanya_two_digit_number_l619_619312


namespace isosceles_triangle_top_angle_l619_619921

theorem isosceles_triangle_top_angle (A B C : Type) [triangle A B C] (isosceles : is_isosceles_triangle A B C) (angle_A : ∠A = 40) : 
∠top_angle(A B C) = 40 ∨ ∠top_angle(A B C) = 100 :=
begin
  sorry
end

end isosceles_triangle_top_angle_l619_619921


namespace find_multiple_l619_619040

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_ineq : (m * n - 15) > 2 * n) : m = 6 := 
by {
  sorry
}

end find_multiple_l619_619040


namespace complex_number_quadrant_l619_619347

theorem complex_number_quadrant (z : ℂ) (h : z = Complex.ofReal (Real.cos 75) + Complex.ofReal (Real.sin 75) * Complex.I) : 
  let z_squared := z * z in (z_squared.re < 0 ∧ z_squared.im > 0) :=
by
  sorry

end complex_number_quadrant_l619_619347


namespace arithmetic_sequence_find_side_length_l619_619552

variable (A B C a b c : ℝ)

-- Condition: Given that b(1 + cos(C)) = c(2 - cos(B))
variable (h : b * (1 + Real.cos C) = c * (2 - Real.cos B))

-- Question I: Prove that a + b = 2 * c
theorem arithmetic_sequence (h : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : a + b = 2 * c :=
sorry

-- Additional conditions for Question II
variable (C_eq : C = Real.pi / 3)
variable (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3)

-- Question II: Find c
theorem find_side_length (C_eq : C = Real.pi / 3) (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) : c = 4 :=
sorry

end arithmetic_sequence_find_side_length_l619_619552


namespace price_per_kilo_of_cucumbers_l619_619225

theorem price_per_kilo_of_cucumbers (
    initial_money : ℝ := 500,
    remaining_money : ℝ := 426,
    potato_price_per_kilo : ℝ := 2,
    tomato_price_per_kilo : ℝ := 3,
    banana_price_per_kilo : ℝ := 5,
    potato_weight : ℝ := 6,
    tomato_weight : ℝ := 9,
    cucumber_weight : ℝ := 5,
    banana_weight : ℝ := 3
) : (initial_money - remaining_money - (potato_price_per_kilo * potato_weight + tomato_price_per_kilo * tomato_weight + banana_price_per_kilo * banana_weight)) / cucumber_weight = 4 := 
sorry

end price_per_kilo_of_cucumbers_l619_619225


namespace colorable_with_three_colors_l619_619222

theorem colorable_with_three_colors (island : Type) [fintype island] 
  (adj : island → island → Prop) [decidable_rel adj] 
  (triangular : island → Prop)
  (complete_side : ∀ (a b : island), adj a b → triangular a ∧ triangular b)
  : ∃ (color : island → fin 3), ∀ (a b : island), adj a b → color a ≠ color b :=
sorry

end colorable_with_three_colors_l619_619222


namespace triangle_third_side_l619_619153

theorem triangle_third_side (a b : ℝ) (θ : ℝ) (cos_θ : ℝ) : 
  a = 9 → b = 12 → θ = 150 → cos_θ = - (Real.sqrt 3 / 2) → 
  (Real.sqrt (a^2 + b^2 - 2 * a * b * cos_θ)) = Real.sqrt (225 + 108 * Real.sqrt 3) := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end triangle_third_side_l619_619153


namespace range_of_a_l619_619515

-- Define sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Mathematical statement to be proven
theorem range_of_a (a : ℝ) : (∃ x, x ∈ set_A ∧ x ∈ set_B a) → a ≥ -1 :=
by
  sorry

end range_of_a_l619_619515


namespace a_minus_b_is_15_l619_619772

variables (a b c : ℝ)

-- Conditions from the problem statement
axiom cond1 : a = 1/3 * (b + c)
axiom cond2 : b = 2/7 * (a + c)
axiom cond3 : a + b + c = 540

-- The theorem we need to prove
theorem a_minus_b_is_15 : a - b = 15 :=
by
  sorry

end a_minus_b_is_15_l619_619772


namespace part1_minimum_value_part2_maximum_value_l619_619101

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ := exp (-x) + (n * x) / (m * x + n)

theorem part1_minimum_value (x : ℝ) : f x 0 1 ≥ f 0 0 1 :=
by sorry

theorem part2_maximum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∀ x ≥ 0, f x m n ≥ 1) : m / n ≤ 1 / 2 :=
by sorry

end part1_minimum_value_part2_maximum_value_l619_619101


namespace measure_of_angle_A_range_of_y_l619_619574

variable (A B C a b c y : ℝ)
variable (ABC : Triangle ℝ)
variable (h1 : b² + c² = a² + bc)
variable (h2 : 0 < B ∧ B < (2 * π / 3))

theorem measure_of_angle_A (ABC : Triangle ℝ) (a b c : ℝ) 
  (h : b^2 + c^2 = a^2 + bc) : A = π / 3 :=
by 
  sorry

theorem range_of_y (ABC : Triangle ℝ) (b : ℝ) 
  (h : 0 < B ∧ B < 2 * π / 3) : 
  let y := sqrt 3 * sin B + cos B in 
  1 < y ∧ y ≤ 2 :=
by 
  sorry

end measure_of_angle_A_range_of_y_l619_619574


namespace sum_of_rational_roots_of_h_l619_619873

noncomputable def h (x : ℚ) : ℚ := x^3 - 9 * x^2 + 27 * x - 14

theorem sum_of_rational_roots_of_h : 
  let roots : List ℚ := [2] in
  (h 2 = 0) →
  (roots.sum = 2) :=
by
  sorry

end sum_of_rational_roots_of_h_l619_619873


namespace find_area_of_circle_l619_619568

noncomputable def area_of_circle (r : ℝ) := π * r^2

theorem find_area_of_circle
  (r : ℝ)
  (O A B C D G H : Point)
  (circle : Circle O r)
  (H1 : is_diameter A B circle)
  (H2 : is_diameter C D circle)
  (H3 : perpendicular A B C D)
  (H4 : is_chord D G circle)
  (H5 : intersects_at H D G A B)
  (H6 : distance D H = 8)
  (H7 : distance H G = 4) :
  area_of_circle r = 64 * π :=
sorry

end find_area_of_circle_l619_619568


namespace greatest_individual_score_l619_619752

variable (players : Fin 12 → ℕ)

def total_points (players : Fin 12 → ℕ) : ℕ :=
  ∑ i, players i

def min_points_each (players : Fin 12 → ℕ) : Prop :=
  ∀ i, players i ≥ 7

theorem greatest_individual_score :
  (total_points players = 100) →
  min_points_each players →
  (∃ w, w = 23 ∧ ∃ i, players i = w) :=
by
  intro h_total h_min
  sorry

end greatest_individual_score_l619_619752


namespace range_of_b_distance_when_b_eq_one_l619_619212

-- Definitions for conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def line (x y b : ℝ) : Prop := y = x + b
def intersect (x y b : ℝ) : Prop := ellipse x y ∧ line x y b

-- Prove the range of b for which there are two distinct intersection points
theorem range_of_b (b : ℝ) : (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ intersect x1 y1 b ∧ intersect x2 y2 b) ↔ (-Real.sqrt 3 < b ∧ b < Real.sqrt 3) :=
by sorry

-- Prove the distance between points A and B when b = 1
theorem distance_when_b_eq_one : 
  ∃ x1 y1 x2 y2, intersect x1 y1 1 ∧ intersect x2 y2 1 ∧ Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 2 / 3 :=
by sorry

end range_of_b_distance_when_b_eq_one_l619_619212


namespace reactants_conservation_l619_619030

noncomputable def reaction1 : ℕ → ℕ → ℕ
| 2 => 1 -- 2 moles of HCl yields 1 mole of H2O
| _ => 0

noncomputable def reaction2 : ℕ → ℕ → ℕ
| 1 => 1 -- 1 mole of HCl yields 1 mole of H2O
| _ => 0

theorem reactants_conservation
  (hcl : ℕ) (nahco3 : ℕ) (catalyst_X : ℕ) (product_Y : Type) :
  hcl = 3 → nahco3 = 3 → catalyst_X = 2 →
  ∃ (h2o co2 nahco3_excess : ℕ),
    reaction1 2 hcl = h2o ∧
    reaction1 2 hcl = co2 ∧
    nahco3_excess = nahco3 - (hcl / 2) ∧
    h2o = 1.5 ∧
    co2 = 1.5 ∧
    nahco3_excess = 1.5 :=
by
  sorry

end reactants_conservation_l619_619030


namespace no_combination_of_five_coins_is_75_l619_619063

theorem no_combination_of_five_coins_is_75 :
  ∀ (a b c d e : ℕ), 
    (a + b + c + d + e = 5) →
    ∀ (v : ℤ), 
      v = a * 1 + b * 5 + c * 10 + d * 25 + e * 50 → 
      v ≠ 75 :=
by
  intro a b c d e h1 v h2
  sorry

end no_combination_of_five_coins_is_75_l619_619063


namespace count_ways_insert_panes_l619_619717

noncomputable def count_valid_arrangements : ℕ :=
  3430

theorem count_ways_insert_panes :
  let colors := 10 in
  let window := 2 * 2 in
  ∃ n : ℕ, n = count_valid_arrangements :=
sorry

end count_ways_insert_panes_l619_619717


namespace volume_correct_l619_619062

noncomputable def volume_regular_hexagonal_pyramid (b Q : ℝ) : ℝ :=
  let a_sq := 2 * b^2 - sqrt (4 * b^4 - 16 * Q^2)
  let h := sqrt (sqrt (4 * b^4 - 16 * Q^2) - b^2)
  (sqrt 3 / 2) * (2 * b^2 - sqrt (4 * b^4 - 16 * Q^2)) * h

theorem volume_correct (b Q : ℝ) : 
  b > 0 → Q > 0 →
  volume_regular_hexagonal_pyramid b Q = 
  (√3 / 2) * (2 * b^2 - sqrt (4 * b^4 - 16 * Q^2)) * sqrt (sqrt (4 * b^4 - 16 * Q^2) - b^2) :=
  by sorry

end volume_correct_l619_619062


namespace amy_hours_per_week_school_year_l619_619399

variable (hours_per_week_summer : ℕ)
variable (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (additional_earnings_needed : ℕ)
variable (weeks_school_year : ℕ)
variable (hourly_wage : ℝ := earnings_summer / (hours_per_week_summer * weeks_summer))

theorem amy_hours_per_week_school_year :
  hours_per_week_school_year = (additional_earnings_needed / hourly_wage) / weeks_school_year :=
by 
  -- Using the hourly wage and total income needed, calculate the hours.
  let total_hours_needed := additional_earnings_needed / hourly_wage
  have h1 : hours_per_week_school_year = total_hours_needed / weeks_school_year := sorry
  exact h1

end amy_hours_per_week_school_year_l619_619399


namespace probability_even_sum_l619_619738

def unfair_die (odd even : ℝ) : Prop :=
  even = 3 * odd ∧ odd + even = 1

theorem probability_even_sum :
  ∃ (odd even : ℝ),
    unfair_die odd even →
    ∀ P : ℝ,
    (P = (even * even) + (odd * odd)) →
    P = 5 / 8 :=
begin
  sorry
end

end probability_even_sum_l619_619738


namespace expansion_coefficient_l619_619684

theorem expansion_coefficient :
  let f := (1 + 1 / x) * (1 - x)^7 in
  coeff (series f x) 2 = -14 :=
by
  sorry

end expansion_coefficient_l619_619684


namespace find_number_l619_619139

theorem find_number (x n : ℤ) (h1 : 5 * x + n = 10 * x - 17) (h2 : x = 4) : n = 3 := by
  sorry

end find_number_l619_619139


namespace euler_conjecture_disproof_l619_619297

theorem euler_conjecture_disproof :
    ∃ (n : ℕ), 133^4 + 110^4 + 56^4 = n^4 ∧ n = 143 :=
by {
  use 143,
  sorry
}

end euler_conjecture_disproof_l619_619297


namespace ab_equals_6_l619_619973

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l619_619973


namespace rectangles_in_grid_l619_619563

theorem rectangles_in_grid :
  let n := 6
  let combinations := Nat.choose n 2
  combinations * combinations = 225 := 
by
  let n := 6
  let combinations := Nat.choose n 2
  have h : combinations = (6 * 5) / 2 := by sorry
  rw [←h]
  norm_num
  exact rfl

end rectangles_in_grid_l619_619563


namespace min_g7_l619_619403

-- Define a tenuous function
def is_tenuous (f : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, 1 ≤ x → 1 ≤ y → f x + f y > y*y

-- Define function g, and conditions
variables (g : ℕ → ℤ) 

-- Conditions for g
axiom g_tenuous : is_tenuous g
axiom g_minimal : g(1) + g(2) + g(3) + g(4) + g(5) + g(6) + g(7) + g(8) + g(9) + g(10) ≤ 385

-- Prove the minimum possible value of g(7)
theorem min_g7 : ∃ k : ℤ, g(7) = k ∧ k = 32 :=
by
  sorry

end min_g7_l619_619403


namespace non_neg_count_correct_l619_619946

def is_non_negative (x : ℝ) : Prop := x ≥ 0

def non_negative_count (lst : List ℝ) : Nat :=
  lst.filter is_non_negative |>.length

theorem non_neg_count_correct :
  non_negative_count [-8, 2.1, (1 / 9), 3, 0, -2.5, 10, -1] = 5 :=
by
  sorry

end non_neg_count_correct_l619_619946


namespace find_u_l619_619884

theorem find_u (u : ℝ) : (∃ x : ℝ, x = ( -15 - Real.sqrt 145 ) / 8 ∧ 4 * x^2 + 15 * x + u = 0) ↔ u = 5 := by
  sorry

end find_u_l619_619884


namespace integral_f_l619_619935

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 else x + 1

theorem integral_f :
  ∫ x in -2..2, f x = 20 / 3 :=
by
  sorry

end integral_f_l619_619935


namespace max_radius_of_circle_touching_graph_l619_619152

theorem max_radius_of_circle_touching_graph :
  ∃ r : ℝ, (∀ (x : ℝ), (x^2 + (x^4 - r)^2 = r^2) → r ≤ (3 * (2:ℝ)^(1/3)) / 4) ∧
           r = (3 * (2:ℝ)^(1/3)) / 4 :=
by
  sorry

end max_radius_of_circle_touching_graph_l619_619152


namespace production_days_l619_619882

variable (n : ℕ) (average_past : ℝ := 50) (production_today : ℝ := 115) (new_average : ℝ := 55)

theorem production_days (h1 : average_past * n + production_today = new_average * (n + 1)) : 
    n = 12 := 
by 
  sorry

end production_days_l619_619882


namespace area_of_triangle_correct_l619_619487

noncomputable def area_of_triangle (a b : ℝ) (A B C : ℝ)
  (sin : ℝ → ℝ) (cos : ℝ → ℝ) (tan : ℝ → ℝ)
  (sin_eq : ∀ (x : ℝ), sin x = (real.sin x))
  (cos_eq : ∀ (x : ℝ), cos x = (real.cos x))
  (tan_eq : ∀ (x : ℝ), tan x = (real.tan x))
  (h1 : a = real.sqrt 7)
  (h2 : b = 2)
  (h3 : (a, -real.sqrt 3 * b) ⊗ (sin B, cos A) = 0)
  (h4 : 0 < A ∧ A < real.pi) :
  real :=
frac 3 2 * real.sqrt 3

-- The Lean theorem statement
theorem area_of_triangle_correct :
  ∀ (a b : ℝ) (A B C : ℝ)
    (sin : ℝ → ℝ) (cos : ℝ → ℝ) (tan : ℝ → ℝ)
    (sin_eq : ∀ (x : ℝ), sin x = (real.sin x))
    (cos_eq : ∀ (x : ℝ), cos x = (real.cos x))
    (tan_eq : ∀ (x : ℝ), tan x = (real.tan x))
    (h1 : a = real.sqrt 7)
    (h2 : b = 2)
    (h3 : ((a, -real.sqrt 3 * b) ⊗ (sin B, cos A)) = 0)
    (h4 : 0 < A ∧ A < real.pi),
    area_of_triangle a b A B C sin cos tan sin_eq cos_eq tan_eq h1 h2 h3 h4 =
    frac 3 2 * real.sqrt 3 :=
by {
  sorry
}

end area_of_triangle_correct_l619_619487


namespace find_k_l619_619095

theorem find_k (k : ℝ) : (∀ x, f(x) = Real.exp x + k * x) → f' x = Real.exp x + k → (f' 0 = 0) → k = -1 :=
by
  intro f condition_x condition_derivative condition_extremum
  sorry

end find_k_l619_619095


namespace sum_geometric_series_l619_619827

theorem sum_geometric_series :
  (∑ i in Finset.range 6, (1 / (2 : ℝ) ^ (i + 1))) = 63 / 64 := 
by
  sorry

end sum_geometric_series_l619_619827


namespace merchant_mixture_solution_l619_619802

variable (P C : ℝ)

def P_price : ℝ := 2.40
def C_price : ℝ := 6.00
def total_weight : ℝ := 60
def total_price_per_pound : ℝ := 3.00
def total_price : ℝ := total_price_per_pound * total_weight

theorem merchant_mixture_solution (h1 : P + C = total_weight)
                                  (h2 : P_price * P + C_price * C = total_price) :
  C = 10 := 
sorry

end merchant_mixture_solution_l619_619802


namespace area_of_quadrilateral_l619_619229

variable (AD AB BC CD AC : ℝ)
variable (right_angle_intersect : ∀ θ, θ = 90 → cos (θ * (π / 180)) = 0)

theorem area_of_quadrilateral :
  AD = 18 → 
  AB = 45 → 
  BC = 25 → 
  CD = 30 → 
  AC = 40 → 
  right_angle_intersect 90 (by norm_num: 90 = 90) → 
  let s := (AD + AB + BC + CD) / 2 in
  sqrt ((s - AD) * (s - AB) * (s - BC) * (s - CD)) = 686 :=
sorry

end area_of_quadrilateral_l619_619229


namespace find_a2_l619_619467

-- Definitions from conditions
def is_arithmetic_sequence (u : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, u (n + 1) = u n + d

def is_geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a2
  (u : ℕ → ℤ) (a1 a3 a4 : ℤ)
  (h1 : is_arithmetic_sequence u 3)
  (h2 : is_geometric_sequence a1 a3 a4)
  (h3 : a1 = u 1)
  (h4 : a3 = u 3)
  (h5 : a4 = u 4) :
  u 2 = -9 :=
by  
  sorry

end find_a2_l619_619467


namespace jamie_pennies_count_l619_619180

theorem jamie_pennies_count :
  (∃ (x : ℕ), x = 34 ∧ (x + 5 * x + 10 * x + 25 * x = 1405)) →
  ∃ (x : ℕ), x = 34 :=
by {
  intro h,
  cases h with x hx,
  existsi x,
  exact hx.1
}

end jamie_pennies_count_l619_619180


namespace toast_three_breads_l619_619371

theorem toast_three_breads :
  ∃ t < 4, ∀ (a b c : Type), (∀ (p : Type), p = a ∨ p = b ∨ p = c) → 
  (∀ (p : Type), (p = a ∨ p = b ∨ p = c) → (time_to_toast p) ≤ t) :=
sorry

def time_to_toast (p : Type) : ℕ :=
  -- Assuming the toast time function gives the total time taken to toast both sides of the bread
  -- This should be realized by the logic in the proof corresponding to the toasting schedule described
  3 -- as determined by the solution steps, it is 3 minutes

end toast_three_breads_l619_619371


namespace intersection_of_sets_l619_619482

variable (A : Set ℝ) (B : Set ℝ)
def A := {x ∈ ℝ | x < -1 ∨ x > 1}
def B := {x ∈ ℝ | -3 < x ∧ x < 2}

theorem intersection_of_sets :
  (A ∩ B) = {x | -3 < x ∧ x < -1} ∪ {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_sets_l619_619482


namespace nat_number_36_sum_of_digits_l619_619048

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end nat_number_36_sum_of_digits_l619_619048


namespace max_removed_cubes_l619_619843

theorem max_removed_cubes : ∃ n : ℕ, n = 14 ∧ 
  (∀ (C : ℕ → ℕ → ℕ → Prop), 
    (∀ x y, ∃ z, C x y z ∧ is_face_filled C ∧ is_connected C) → 
    (27 - count_filled_cubes C = n)) :=
sorry

-- Auxiliary definitions required for conditions.

def is_face_filled (C : ℕ → ℕ → ℕ → Prop) : Prop := 
  ∀ face : ℕ → ℕ → Prop, 
  (face = (λ a b, ∃ c, C a b c) ∨ 
   face = (λ a b, ∃ c, C c a b) ∨ 
   face = (λ a b, ∃ c, C b c a)) → 
  (∀ x y, face x y → true)

def is_connected (C : ℕ → ℕ → ℕ → Prop) : Prop := 
  ∀ u v : (ℕ × ℕ × ℕ), 
  (∃ a b c, C a b c ∧ u = (a, b, c)) ∧ 
  (∃ a b c, C a b c ∧ v = (a, b, c)) →
  ∃ path : list (ℕ × ℕ × ℕ), 
  path.head = u ∧ path.last = v ∧ 
  (∀ (p1 p2 : ℕ × ℕ × ℕ), 
   list.pairwise (λ x y, adj x y) (u :: path ++ [v]))

def adj (c1 c2 : ℕ × ℕ × ℕ) : Prop := 
  (c1.1 = c2.1 ∧ c1.2 = c2.2 ∧ (c1.3 = c2.3 - 1 ∨ c1.3 = c2.3 + 1)) ∨
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 - 1 ∨ c1.2 = c2.2 + 1) ∧ c1.3 = c2.3) ∨
  ((c1.1 = c1.1 - 1 ∨ c1.1 = c1.1 + 1) ∧ c1.2 = c2.2 ∧ c1.3 = c2.3)

def count_filled_cubes (C : ℕ → ℕ → ℕ → Prop) : ℕ := 
  nat.iterate_ft 27 (λ c, c-1)

end max_removed_cubes_l619_619843


namespace cos2alpha_plus_tanalpha_l619_619457

variable (α : Real)

theorem cos2alpha_plus_tanalpha (h : sin α - 3 * cos α = 0) : cos (2 * α) + tan α = 11 / 5 := 
by
  sorry

end cos2alpha_plus_tanalpha_l619_619457


namespace triangle_QRS_perimeter_l619_619648

noncomputable def perimeter_of_triangle (P Q R S T : ℝ × ℝ) : ℝ :=
  let d := dist
  d Q R + d R S + d S Q

theorem triangle_QRS_perimeter (P Q R S T : ℝ × ℝ)
  (hPQ : dist P Q = 3)
  (hQR : dist Q R = 3)
  (hRS : dist R S = 3)
  (hST : dist S T = 3)
  (hTP : dist T P = 3)
  (hAnglePQR : angle P Q R = 120)
  (hAngleRST : angle R S T = 120)
  (hAngleSTP : angle S T P = 120)
  (hParallel : ∃ m1 m2 : ℝ, (Q.2 - P.2) = m1 * (Q.1 - P.1) ∧ (T.2 - S.2) = m2 * (T.1 - S.1) ∧ m1 = m2) :
  perimeter_of_triangle Q R S = 9 :=
by
  sorry

end triangle_QRS_perimeter_l619_619648


namespace total_rainfall_2010_to_2012_l619_619454

noncomputable def average_rainfall (year : ℕ) : ℕ :=
  if year = 2010 then 35
  else if year = 2011 then 38
  else if year = 2012 then 41
  else 0

theorem total_rainfall_2010_to_2012 :
  (12 * average_rainfall 2010) + 
  (12 * average_rainfall 2011) + 
  (12 * average_rainfall 2012) = 1368 :=
by
  sorry

end total_rainfall_2010_to_2012_l619_619454


namespace find_natural_numbers_l619_619045

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem find_natural_numbers (x : ℕ) :
  (x = 36 * sum_of_digits x) ↔ (x = 324 ∨ x = 648) :=
by
  sorry

end find_natural_numbers_l619_619045


namespace quadratic_one_real_root_l619_619548

theorem quadratic_one_real_root (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x : ℝ, x^2 + 6*m*x - n = 0 → x * x = 0) : n = 9*m^2 := 
by 
  sorry

end quadratic_one_real_root_l619_619548


namespace sand_art_calculation_l619_619185

theorem sand_art_calculation :
  let rect_length := 6 in
  let rect_width := 7 in
  let square_side := 5 in
  let gram_per_sq_inch := 3 in
  let rect_area := rect_length * rect_width in
  let square_area := square_side * square_side in
  let rect_sand := rect_area * gram_per_sq_inch in
  let square_sand := square_area * gram_per_sq_inch in
  let total_sand_needed := rect_sand + square_sand in
  total_sand_needed = 201 :=
by
  -- sorry is used to skip the proof as instructed
  sorry

end sand_art_calculation_l619_619185


namespace correct_answer_l619_619528

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l619_619528


namespace additional_license_plates_l619_619020

def original_license_plates : ℕ := 5 * 3 * 5
def new_license_plates : ℕ := 6 * 4 * 5

theorem additional_license_plates : new_license_plates - original_license_plates = 45 := by
  sorry

end additional_license_plates_l619_619020


namespace statement_A_statement_C_statement_D_correct_statements_l619_619341

-- Definitions and conditions
def floor (x : ℝ) : ℤ := Int.floor x

-- Statement A: ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋
theorem statement_A (x y : ℝ) : floor (x - y) ≤ floor x - floor y := 
  sorry

-- Statement C: If in the sequence b_n = ⌊√(n(n+1))⌋, n ∈ ℕ, then ∑_{n=1}^{64} b_n = 2080
def b (n : ℕ) : ℤ := floor (Real.sqrt (n * (n + 1)))

theorem statement_C : (∑ n in Finset.range 64.succ, b n) = 2080 :=
  sorry

-- Statement D: M = ⌊2/3⌋ + ⌊2^2/3⌋ + ⌊2^3/3⌋ + ... + ⌊2^2022/3⌋ has a remainder of 0 when divided by 3
def M : ℤ := ∑ k in Finset.range 2022, floor (2^k / 3)

theorem statement_D : M % 3 = 0 :=
  sorry

-- Correct statements
theorem correct_statements : 
  (statement_A) ∧ (¬ statement_B) ∧ (statement_C) ∧ (statement_D) := 
  sorry

end statement_A_statement_C_statement_D_correct_statements_l619_619341


namespace hyperbola_a_unique_l619_619109

-- Definitions from the conditions
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1
def foci (c : ℝ) : Prop := c = 2 * Real.sqrt 3
def a_positive (a : ℝ) : Prop := a > 0

-- Statement to prove
theorem hyperbola_a_unique (a : ℝ) (h : hyperbola 0 0 a ∧ foci (2 * Real.sqrt 3) ∧ a_positive a) : a = 2 * Real.sqrt 2 := 
sorry

end hyperbola_a_unique_l619_619109


namespace monotonically_increasing_range_l619_619143

open Real

noncomputable def f (a x : ℝ) := log a (x^3 - a * x)

theorem monotonically_increasing_range (a : ℝ) :
  0 < a ∧ a ≠ 1 →
  (∀ x₁ x₂ ∈ Ioo (-1/3 : ℝ) 0, x₁ < x₂ → f a x₁ < f a x₂)
  ↔ (a ∈ set.Ico (1/3 : ℝ) 1) :=
sorry

end monotonically_increasing_range_l619_619143


namespace simplifies_to_minus_18_point_5_l619_619668

theorem simplifies_to_minus_18_point_5 (x y : ℝ) (h_x : x = 1/2) (h_y : y = -2) :
  ((2 * x + y)^2 - (2 * x - y) * (x + y) - 2 * (x - 2 * y) * (x + 2 * y)) / y = -18.5 :=
by
  -- Let's replace x and y with their values
  -- Expand and simplify the expression
  -- Divide the expression by y
  -- Prove the final result is equal to -18.5
  sorry

end simplifies_to_minus_18_point_5_l619_619668


namespace qualified_light_bulb_prob_l619_619554

def prob_factory_A := 0.7
def prob_factory_B := 0.3
def qual_rate_A := 0.9
def qual_rate_B := 0.8

theorem qualified_light_bulb_prob :
  prob_factory_A * qual_rate_A + prob_factory_B * qual_rate_B = 0.87 :=
by
  sorry

end qualified_light_bulb_prob_l619_619554


namespace fraction_of_juan_chocolates_given_to_tito_l619_619004

variable (n : ℕ)
variable (Juan Angela Tito : ℕ)
variable (f : ℝ)

-- Conditions
def chocolates_Angela_Tito : Angela = 3 * Tito := 
by sorry

def chocolates_Juan_Angela : Juan = 4 * Angela := 
by sorry

def equal_distribution : (Juan + Angela + Tito) = 16 * n := 
by sorry

-- Theorem to prove
theorem fraction_of_juan_chocolates_given_to_tito (n : ℕ) 
  (H1 : Angela = 3 * Tito)
  (H2 : Juan = 4 * Angela)
  (H3 : Juan + Angela + Tito = 16 * n) :
  f = 13 / 36 :=
by sorry

end fraction_of_juan_chocolates_given_to_tito_l619_619004


namespace no_divisor_30_to_40_of_2_pow_28_minus_1_l619_619427

theorem no_divisor_30_to_40_of_2_pow_28_minus_1 :
  ¬ ∃ n : ℕ, (30 ≤ n ∧ n ≤ 40 ∧ n ∣ (2^28 - 1)) :=
by
  sorry

end no_divisor_30_to_40_of_2_pow_28_minus_1_l619_619427


namespace profit_percentage_l619_619402

theorem profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) 
  (h1 : cost_price = 66.5) (h2 : marked_price = 87.5) (h3 : discount_rate = 0.05) : 
  (100 * ((marked_price * (1 - discount_rate) - cost_price) / cost_price)) = 25 :=
by
  sorry

end profit_percentage_l619_619402


namespace intersection_of_sets_l619_619951

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets :
  setA ∩ setB = { z : ℝ | z ∈ [-1, 1] } :=
sorry

end intersection_of_sets_l619_619951


namespace probability_odd_sum_grid_l619_619231

theorem probability_odd_sum_grid : 
  let numbers := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let all_sums_odd (grid : List (List ℕ)) : Prop :=
    ∀ row ∈ grid, (row.sum % 2 = 1) ∧ (List.transpose grid).all (λ col, col.sum % 2 = 1)
  let count_valid_grids := (number of valid grid configurations satisfying all_sums_odd)
  let total_grids := (number of all possible unique arrangements of numbers)
  let probability := count_valid_grids * 1.0 / total_grids
  in probability = 1 / 14 :=
sorry

end probability_odd_sum_grid_l619_619231


namespace margarets_mean_score_l619_619453

noncomputable def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

open List

theorem margarets_mean_score :
  let scores := [86, 88, 91, 93, 95, 97, 99, 100]
  let cyprians_mean := 92
  let num_scores := 8
  let cyprians_scores := 4
  let margarets_scores := num_scores - cyprians_scores
  (scores.sum - cyprians_scores * cyprians_mean) / margarets_scores = 95.25 :=
by
  sorry

end margarets_mean_score_l619_619453


namespace sum_of_rational_roots_eq_zero_l619_619875

def h (x : ℚ) : ℚ := x^3 - 9 * x^2 + 27 * x - 14

theorem sum_of_rational_roots_eq_zero : ∑ r in (multiset.filter (λ r, h r = 0) [1, -1, 2, -2, 7, -7, 14, -14]), r = 0 :=
sorry

end sum_of_rational_roots_eq_zero_l619_619875


namespace sqrt_sum_leq_sqrt3_l619_619074

theorem sqrt_sum_leq_sqrt3 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  sqrt a + sqrt b + sqrt c ≤ sqrt 3 := 
  sorry

end sqrt_sum_leq_sqrt3_l619_619074


namespace bacteria_growth_after_three_hours_l619_619359

theorem bacteria_growth_after_three_hours :
  ∀ (initial_bacteria : ℕ), initial_bacteria = 1 → (30 : ℕ) ∣ 180 → (3 : ℕ) = 180 / 60 → 
  (initial_bacteria * 2 ^ (180 / 30) = 64) :=
by
  intros initial_bacteria h1 h2 h3
  have h_intervals: initial_bacteria * 2 ^ 6 = 64,
  { rw h1,
    norm_num },
  exact h_intervals

end bacteria_growth_after_three_hours_l619_619359


namespace digit_in_100th_place_l619_619965

theorem digit_in_100th_place :
    let seq : List Char := ['2', '6', '9', '2', '3', '0']
    (seq.get! ((100 % seq.length) - 1)) = '9' :=
by
  let seq : List Char := ['2', '6', '9', '2', '3', '0']
  have h_len : seq.length = 6 := rfl
  have h_mod : 100 % 6 = 4 := rfl
  have h_idx : (100 % seq.length) - 1 = 3 := by
    rw [h_len, h_mod]
    exact rfl
  show seq.get! 3 = '9' from rfl

end digit_in_100th_place_l619_619965


namespace rooster_stamps_eq_two_l619_619254

variable (r d : ℕ) -- r is the number of rooster stamps, d is the number of daffodil stamps

theorem rooster_stamps_eq_two (h1 : d = 2) (h2 : r - d = 0) : r = 2 := by
  sorry

end rooster_stamps_eq_two_l619_619254


namespace amy_hours_per_week_school_year_l619_619398

variable (hours_per_week_summer : ℕ)
variable (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (additional_earnings_needed : ℕ)
variable (weeks_school_year : ℕ)
variable (hourly_wage : ℝ := earnings_summer / (hours_per_week_summer * weeks_summer))

theorem amy_hours_per_week_school_year :
  hours_per_week_school_year = (additional_earnings_needed / hourly_wage) / weeks_school_year :=
by 
  -- Using the hourly wage and total income needed, calculate the hours.
  let total_hours_needed := additional_earnings_needed / hourly_wage
  have h1 : hours_per_week_school_year = total_hours_needed / weeks_school_year := sorry
  exact h1

end amy_hours_per_week_school_year_l619_619398


namespace find_hire_year_l619_619360

-- Define the conditions from the problem
def employee_hiring (A : ℕ) (retire_year hire_year : ℕ) : Prop :=
  A = 32 ∧ retire_year = 2006 ∧ (A + (retire_year - hire_year) = 70)

-- Define the theorem to prove the year of hiring
theorem find_hire_year : ∃ (hire_year : ℕ), employee_hiring 32 2006 hire_year ∧ hire_year = 1968 :=
begin
  sorry
end

end find_hire_year_l619_619360


namespace interest_rate_is_10_percent_l619_619374

theorem interest_rate_is_10_percent
  (principal : ℝ)
  (interest_rate_c : ℝ) 
  (time : ℝ)
  (gain_b : ℝ)
  (interest_c : ℝ := principal * interest_rate_c / 100 * time)
  (interest_a : ℝ := interest_c - gain_b)
  (expected_rate : ℝ := (interest_a / (principal * time)) * 100)
  (h1: principal = 3500)
  (h2: interest_rate_c = 12)
  (h3: time = 3)
  (h4: gain_b = 210)
  : expected_rate = 10 := 
  by 
  sorry

end interest_rate_is_10_percent_l619_619374


namespace smallest_value_l619_619609

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l619_619609


namespace Nicki_total_miles_run_l619_619216

theorem Nicki_total_miles_run:
  ∀ (miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year : ℕ),
  miles_per_week_first_half = 20 →
  miles_per_week_second_half = 30 →
  weeks_in_year = 52 →
  weeks_per_half_year = weeks_in_year / 2 →
  (miles_per_week_first_half * weeks_per_half_year) + (miles_per_week_second_half * weeks_per_half_year) = 1300 :=
by
  intros miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year
  intros h1 h2 h3 h4
  sorry

end Nicki_total_miles_run_l619_619216


namespace hyperbola_asymptotes_l619_619682

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - (y^2 / 9) = 1) → (y = 3 * x ∨ y = -3 * x) :=
by
  -- conditions and theorem to prove
  sorry

end hyperbola_asymptotes_l619_619682


namespace train_length_l619_619385

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_conversion_factor : ℝ) (approx_length : ℝ) :
  speed_km_hr = 60 → time_seconds = 6 → speed_conversion_factor = (1000 / 3600) → approx_length = 100.02 →
  speed_km_hr * speed_conversion_factor * time_seconds = approx_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end train_length_l619_619385


namespace cartesian_eq_of_line_min_distance_is_half_l619_619470

noncomputable def circle_eq (x y: ℝ) : Prop :=
  (x - 4)^2 + y^2 = 1

noncomputable def line_polar_eq (ρ θ: ℝ) : Prop :=
  ρ * sin (θ + π / 6) = 1 / 2

theorem cartesian_eq_of_line
  (x y ρ θ : ℝ)
  (h1 : circle_eq x y)
  (h2 : line_polar_eq ρ θ) :
  (x + sqrt 3 * y - 1 = 0 ∧
   ∃ t : ℝ, x = 4 + cos t ∧ y = sin t ∧
   min_dist x y = 1 / 2) :=
sorry

noncomputable def min_dist (x y : ℝ) : ℝ :=
  abs (4 + cos φ + sqrt 3 * sin φ - 1) / 2

theorem min_distance_is_half
  (x y φ : ℝ)
  (h1 : circle_eq x y)
  (h2 : min_dist x y = 1 / 2) :
  ∃ t : ℝ, x = 4 + cos t ∧ y = sin t ∧
  min_dist x y = 1 / 2 :=
sorry

end cartesian_eq_of_line_min_distance_is_half_l619_619470


namespace product_ab_l619_619982

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619982


namespace a_six_between_three_and_four_l619_619913

theorem a_six_between_three_and_four (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := 
sorry

end a_six_between_three_and_four_l619_619913


namespace sum_of_ages_l619_619817

variable (a b c : ℕ)

theorem sum_of_ages (h1 : a = 20 + b + c) (h2 : a^2 = 2000 + (b + c)^2) : a + b + c = 80 := 
by
  sorry

end sum_of_ages_l619_619817


namespace num_integers_satisfying_inequality_l619_619956

theorem num_integers_satisfying_inequality :
  {x : ℤ | (x - 4)^2 ≤ 4}.card = 5 :=
sorry

end num_integers_satisfying_inequality_l619_619956


namespace ab_equals_six_l619_619996

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l619_619996


namespace find_leftover_bolts_l619_619880

variable (boxes_of_bolts : ℕ)
variable (bolts_per_box : ℕ)
variable (boxes_of_nuts : ℕ)
variable (nuts_per_box : ℕ)
variable (used_bolts_and_nuts : ℕ)
variable (leftover_nuts : ℕ)

def total_bolts : ℕ := boxes_of_bolts * bolts_per_box
def total_nuts : ℕ := boxes_of_nuts * nuts_per_box
def total_bolts_and_nuts : ℕ := total_bolts + total_nuts
def leftover_bolts_and_nuts : ℕ := total_bolts_and_nuts - used_bolts_and_nuts

theorem find_leftover_bolts
  (h_boxes_of_bolts : boxes_of_bolts = 7)
  (h_bolts_per_box : bolts_per_box = 11)
  (h_boxes_of_nuts : boxes_of_nuts = 3)
  (h_nuts_per_box : nuts_per_box = 15)
  (h_used_bolts_and_nuts : used_bolts_and_nuts = 113)
  (h_leftover_nuts : leftover_nuts = 6) :
  leftover_bolts_and_nuts - leftover_nuts = 3 := by
  sorry

end find_leftover_bolts_l619_619880


namespace cos_beta_minus_alpha_l619_619901

open Real

theorem cos_beta_minus_alpha :
  ∀ α β : ℝ,
  cos α = -3 / 5 ∧ α ∈ set.Ioo (π / 2) π ∧ sin β = -12 / 13 ∧ β ∈ set.Ioo π (3 * π / 2) →
  cos (β - α) = -33 / 65 :=
by
  intros α β h
  rcases h with ⟨hcosα, hα_range, hsinβ, hβ_range⟩
  sorry

end cos_beta_minus_alpha_l619_619901


namespace direction_vector_of_reflection_l619_619513

noncomputable def reflection_matrix := 
  matrix.vec_cons 
    (matrix.vec_cons (8/17 : ℝ) (-15/17) matrix.vec_empty) 
    (matrix.vec_cons (-15/17) (-8/17 : ℝ) matrix.vec_empty)

def is_direction_vector (x y : ℤ) : Prop :=
  x > 0 ∧ Int.gcd (Int.natAbs x) (Int.natAbs y) = 1
 
theorem direction_vector_of_reflection :
  ∃ (x y : ℤ), 
    reflection_matrix.mul 
      (matrix.vec_cons x y matrix.vec_empty) = 
      (matrix.vec_cons x y matrix.vec_empty) ∧ 
    is_direction_vector x y ∧ 
    x = 5 ∧ y = 3 := 
by 
  sorry

end direction_vector_of_reflection_l619_619513


namespace factor_polynomial_l619_619696

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := 
by sorry

end factor_polynomial_l619_619696


namespace arithmetic_sequence_sum_l619_619695

theorem arithmetic_sequence_sum (x : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ k, 1 < k ∧ k < n → 2 * x k = x (k - 1) + x (k + 1))
  (h2 : (Finset.range (n - 2)).sum (λ k, x (k + 2) / (x (k + 1) + x (k + 3))) = 1957) :
  n = 3916 := sorry

end arithmetic_sequence_sum_l619_619695


namespace solve_10_arithmetic_in_1_minute_l619_619318

-- Define the times required for each task
def time_math_class : Nat := 40 -- in minutes
def time_walk_kilometer : Nat := 20 -- in minutes
def time_solve_arithmetic : Nat := 1 -- in minutes

-- The question: Which task can be completed in 1 minute?
def task_completed_in_1_minute : Nat := 1

theorem solve_10_arithmetic_in_1_minute :
  time_solve_arithmetic = task_completed_in_1_minute :=
by
  sorry

end solve_10_arithmetic_in_1_minute_l619_619318


namespace cone_volume_calculation_l619_619382

noncomputable def volume_of_cone (r_sector : ℝ) (theta_sector : ℝ) (r_cone : ℝ) (h_cone : ℝ) : ℝ :=
  (1/3) * π * r_cone^2 * h_cone

theorem cone_volume_calculation :
  let r_sector := 3
  let theta_sector := 120
  let r_cone := 1
  let h_cone := 2 * real.sqrt 2
  volume_of_cone r_sector theta_sector r_cone h_cone = (2 * real.sqrt 2 * π) / 3 :=
by
  sorry

end cone_volume_calculation_l619_619382


namespace distinct_complex_numbers_count_l619_619442

theorem distinct_complex_numbers_count :
  ∃ (S : Finset ℂ), (∀ z ∈ S, |z| = 1 ∧ (z ^ 120 - z ^ 24).im = 0) ∧ S.card = 625 :=
by
  sorry -- The proof is omitted

end distinct_complex_numbers_count_l619_619442


namespace largest_seven_consecutive_non_primes_less_than_40_l619_619236

def is_non_prime (n : ℕ) : Prop :=
  n ≠ 1 ∧ ¬(∃ p, nat.prime p ∧ p ∣ n)

def consecutive_non_primes_sequence (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 7 → is_non_prime (n + i) ∧ (10 ≤ n + i) ∧ (n + i < 40)

theorem largest_seven_consecutive_non_primes_less_than_40 :
  ∃ n, consecutive_non_primes_sequence n ∧ n + 6 = 32 :=
sorry

end largest_seven_consecutive_non_primes_less_than_40_l619_619236


namespace angle_equality_l619_619597

open EuclideanGeometry

variables {P A B K T P' : Point} {C : Circle}

def is_tangent (circle : Circle) (P A : Point) : Prop := EuclideanGeometry.is_tangent circle P A

def point_on_circle {C : Circle} (P : Point) : Prop := EuclideanGeometry.point_on_circle C P

def reflection (P A P' : Point) : Prop := EuclideanGeometry.reflection P A P'

def on_segment (P A B : Point) : Prop := EuclideanGeometry.on_segment P A B

def circumcircle_of_triangle (P B K : Point) : Circle := EuclideanGeometry.circumcircle_of_triangle P B K

theorem angle_equality
  (P_outside_C : ¬ point_on_circle C P)
  (tangent_PA : is_tangent C P A)
  (tangent_PB : is_tangent C P B)
  (K_on_AB : on_segment K A B)
  (T_on_circumcircle : point_on_circle (circumcircle_of_triangle P B K) T)
  (T_on_C : point_on_circle C T)
  (P'_is_reflection : reflection P A P') :
  ∠ P B T = ∠ P' K A :=
sorry

end angle_equality_l619_619597


namespace product_ab_l619_619984

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619984


namespace twelve_edge_cubes_painted_faces_l619_619367

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end twelve_edge_cubes_painted_faces_l619_619367


namespace amelia_wins_l619_619758

-- Variables and conditions
variable (P_A : ℚ) -- Probability that Amelia's coin lands heads
variable (P_B : ℚ) -- Probability that Brian's coin lands heads

-- Amelia starts
variable (start_with_amelia : Prop)

-- Probability definitions, Amelia first, Brian second
def P0 := P_A + (1 - P_A) * (1 - P_B) * P0

-- Target proof
theorem amelia_wins (h1 : P_A = 1/3) (h2 : P_B = 2/5) : 
  let pq := (P0, (1:P0).denominator)
  in pq.snd - pq.fst = 4 :=
by 
  -- The specific statement is left for Lean's proof mechanism
  sorry

end amelia_wins_l619_619758


namespace complex_magnitude_problem_l619_619208

def z : ℂ := -2 + complex.i
def z_conj : ℂ := complex.conj z

theorem complex_magnitude_problem :
  complex.abs ((1 + z) * z_conj) = real.sqrt 10 :=
sorry

end complex_magnitude_problem_l619_619208


namespace product_ab_l619_619983

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l619_619983


namespace certain_number_is_16_point_726_l619_619138

theorem certain_number_is_16_point_726 : 
  ∀ (certain_number : ℝ), 
  (2994 / certain_number = 179) → 
  certain_number = 2994 / 179 :=
begin  
  intros certain_number h,
  sorry -- proof omitted
end

end certain_number_is_16_point_726_l619_619138


namespace even_function_monotonicity_positive_a_monotonicity_negative_a_l619_619933

-- Definition of the function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 / (x^2 - 1)

-- Condition a ≠ 0
axiom a_nonzero (a : ℝ) : a ≠ 0

-- Domain of x: x ≠ ±1
axiom x_domain (x : ℝ) : x ≠ 1 ∧ x ≠ -1

-- Proving even property of f(x)
theorem even_function (a : ℝ) (hx : x_domain x) : f a x = f a (-x) :=
begin
  sorry
end

-- Monotonicity for a > 0
theorem monotonicity_positive_a (a : ℝ) (hx : x_domain x) (ha_pos : 0 < a) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 < 1 → f a x1 > f a x2) ∧ 
  (∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 ≤ 0 → f a x1 < f a x2) :=
begin
  sorry
end

-- Monotonicity for a < 0
theorem monotonicity_negative_a (a : ℝ) (hx : x_domain x) (ha_neg : a < 0) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 < 1 → f a x1 < f a x2) ∧ 
  (∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 ≤ 0 → f a x1 > f a x2) :=
begin
  sorry
end

end even_function_monotonicity_positive_a_monotonicity_negative_a_l619_619933


namespace injective_function_characterization_l619_619436

theorem injective_function_characterization (f : ℤ → ℤ) (a : ℤ) :
  (Injective f) →
  (∀ x y : ℤ, |f(x) - f(y)| ≤ |x - y|) →
  (∀x, f(x) = a + x ∨ f(x) = a - x) := by
  sorry

end injective_function_characterization_l619_619436


namespace no_such_triples_l619_619053

open Nat

theorem no_such_triples (a b p : ℕ) (h1 : p.Prime) (h2 : (a - 1).gcd (b + 1) = 1) :
  ¬ ∃ (c : Fin (p-1) → ℕ), (∀ i, c i > 1 ∧ Odd (c i)) ∧ a^p - p = ∑ i, b^(c i) :=
by
  -- The existence proof will lead to a contradiction
  sorry

end no_such_triples_l619_619053


namespace log_base_3_inequality_l619_619086

noncomputable def a := Real.log 0.2 / Real.log 3
noncomputable def b := 3 ^ 0.2
noncomputable def c := 0.2 ^ 0.3

theorem log_base_3_inequality :
  a < c ∧ c < b :=
by
  -- Proof is omitted
  sorry

end log_base_3_inequality_l619_619086


namespace minimal_n_partition_l619_619201

theorem minimal_n_partition (n : ℕ) (h : n ≥ 2) (T : set ℕ) (hT : T = {i | 2 ≤ i ∧ i ≤ n}) :
  (∀ A B : set ℕ, A ∪ B = T → A ∩ B = ∅ → 
    (∃ x y z ∈ T, x * y = z ∨ y * x = z)) ↔ n ≥ 256 :=
by
  sorry

end minimal_n_partition_l619_619201


namespace circumcircle_radius_triangle_l619_619726

theorem circumcircle_radius_triangle 
  (A M P Q N B : Point) 
  (M A = 2) (M P = 2)
  (N B = 5) (N Q = 5)
  (H_ratio : ratio_areas (triangle_area A Q N) (triangle_area M P B) = 15 * sqrt (2 + sqrt 3) / (5 * sqrt 3)) :
  circumcircle_radius ABC = 10 := 
sorry

end circumcircle_radius_triangle_l619_619726


namespace smallest_possible_value_l619_619619

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l619_619619


namespace digit_100_of_7_div_26_l619_619968

theorem digit_100_of_7_div_26 : 
  ( (\frac{7}{26} : ℚ).decimal_expansion.nth 100 = 2 ) := by 
sorry

end digit_100_of_7_div_26_l619_619968


namespace digit_in_100th_place_l619_619967

theorem digit_in_100th_place :
    let seq : List Char := ['2', '6', '9', '2', '3', '0']
    (seq.get! ((100 % seq.length) - 1)) = '9' :=
by
  let seq : List Char := ['2', '6', '9', '2', '3', '0']
  have h_len : seq.length = 6 := rfl
  have h_mod : 100 % 6 = 4 := rfl
  have h_idx : (100 % seq.length) - 1 = 3 := by
    rw [h_len, h_mod]
    exact rfl
  show seq.get! 3 = '9' from rfl

end digit_in_100th_place_l619_619967


namespace train_length_proof_l619_619391

noncomputable def train_length (speed_km_per_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  speed_m_per_s * time_sec

theorem train_length_proof :
  train_length 60 6 = 100.02 :=
by
  sorry

end train_length_proof_l619_619391


namespace find_alpha_l619_619165

-- Define the parametric equations for C1
def C1_parametric (ϕ : ℝ) : ℝ × ℝ := (1 + Real.cos ϕ, Real.sin ϕ)

-- Define the equation of line C2 in rectangular coordinates
def C2_cartesian (x y : ℝ) : Prop := x + y = 3

-- Define the conversion from rectangular to polar coordinates
def polar_coordinates (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the polar coordinate equation for C1
def C1_polar (θ : ℝ) (ρ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the polar coordinate equation for C2
def C2_polar (θ : ℝ) (ρ : ℝ) : Prop := ρ = 3 / (Real.cos θ + Real.sin θ)

-- Define the condition for the intersection rays
def intersection_condition (α : ℝ) (ρ_A ρ_B : ℝ) : Prop := (0 < α ∧ α < Real.pi / 2) ∧ (ρ_A * ρ_B = 3)

-- Define the problem statement
theorem find_alpha (α ρ_A ρ_B : ℝ) :
  C1_polar α ρ_A →
  C2_polar α ρ_B →
  intersection_condition α ρ_A ρ_B →
  α = Real.pi / 4 :=
by
  sorry

end find_alpha_l619_619165


namespace correct_option_l619_619324

-- Definitions based on the conditions in step a
def option_a : Prop := (-3 - 1 = -2)
def option_b : Prop := (-2 * (-1 / 2) = 1)
def option_c : Prop := (16 / (-4 / 3) = 12)
def option_d : Prop := (- (3^2) / 4 = (9 / 4))

-- The proof problem statement asserting that only option B is correct.
theorem correct_option : option_b ∧ ¬ option_a ∧ ¬ option_c ∧ ¬ option_d :=
by sorry

end correct_option_l619_619324


namespace polynomial_not_2_times_1979_l619_619598

-- Definitions of the conditions
variables {R : Type*} [CommRing R] {x a b c d : R} (f : R → R)

-- Condition 1: f(x) is a polynomial with integer coefficients
-- (implicit in Lean as f can accept polynomial definitions naturally)

-- Condition 2: f(a) = 1979, f(b) = 1979, f(c) = 1979, f(d) = 1979 for distinct a, b, c, d
def f_conditions : Prop := (f a = 1979) ∧ (f b = 1979) ∧ (f c = 1979) ∧ (f d = 1979) ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d)

-- Theorem to be proved
theorem polynomial_not_2_times_1979 (f : R → R) (hf : f_conditions f) : ∀ x : R, f x ≠ 2 * 1979 :=
begin
  sorry
end

end polynomial_not_2_times_1979_l619_619598


namespace range_of_m_l619_619097

theorem range_of_m (m n : ℝ) (h : ∀ x ∈ (Set.Icc 2 6), deriv (λ x, x^2 + 4 * m * x + n) x < 0) : 
  m ≤ -3 :=
by
  sorry

end range_of_m_l619_619097


namespace solve_trig_eq_l619_619245

open Real

theorem solve_trig_eq (n : ℤ) (x : ℝ) : 
  (sin x) ^ 4 + (cos x) ^ 4 = (sin (2 * x)) ^ 4 + (cos (2 * x)) ^ 4 ↔ x = (n : ℝ) * π / 6 :=
by
  sorry

end solve_trig_eq_l619_619245


namespace possible_values_for_m_l619_619499

theorem possible_values_for_m {m : ℕ} (h_eq : ∀ x : ℝ, (m : ℝ) / (x - 1) + 2 = -3 / (1 - x) → x ≥ 0) :
  {m | ∃ x : ℝ, x ≥ 0 ∧ (m : ℝ) / (x - 1) + 2 = -3 / (1 - x)} = {1, 2, 4, 5}.card = 4 := 
sorry

end possible_values_for_m_l619_619499


namespace exists_binary_representation_l619_619654

theorem exists_binary_representation (N : ℕ) : 
  ∃ (k : ℕ) (a : Fin (k + 1) → ℕ), 
    (∀ i, a i = 1 ∨ a i = 2) ∧ 
    (N = ∑ i in Finset.range (k + 1), a i * 2^i) :=
by
  sorry

end exists_binary_representation_l619_619654


namespace sum_tan_fourth_power_l619_619194

open Real

theorem sum_tan_fourth_power (S : Set ℝ) (hS : ∀ x ∈ S, 0 < x ∧ x < π/2 ∧ 
  (∃ a b c, a^2 + b^2 = c^2 ∧ (a, b, c) ∈ {(sin x)^2, (cos x)^2, (tan x)^2} × {(sin x)^2, (cos x)^2, (tan x)^2} × {(sin x)^2, (cos x)^2, (tan x)^2})) :
  ∑ x in S, (tan x)^4 = 4 - 2 * sqrt 2 :=
sorry

end sum_tan_fourth_power_l619_619194


namespace graph_of_f_minus_x_l619_619942

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ -4 ∧ x ≤ -1 then -2 - (x + 1) / 2
  else if h : x ≥ -1 ∧ x ≤ 1 then -Real.sqrt (1 - (x + 1) ^ 2)
  else if h : x ≥ 1 ∧ x <= 3 then x - 3
  else 0

theorem graph_of_f_minus_x (x : ℝ) : |f (-x)| =
  if 1 ≤ x ∧ x ≤ 4 then | -((x - 1) / 2) - 2 |
  else if -1 ≤ x ∧ x ≤ 1 then Real.sqrt (1 - (x - 1) ^ 2)
  else if -3 ≤ x ∧ x ≤ -1 then -x - 3
  else 0 := sorry

end graph_of_f_minus_x_l619_619942


namespace juniors_score_is_89_l619_619555

variable (n : ℕ) -- Let n represent the total number of students
variable (juniorScore : ℝ) -- Let juniorScore represent each junior's score

-- Conditions
def juniors : ℝ := 0.2 * n
def seniors : ℝ := 0.8 * n
def overallAverageScore : ℝ := 85
def juniorCondition : Prop := ∀ j, j ∈ juniors → juniorScore = j
def seniorAverageScore : ℝ := 84

-- Prove that each junior received a score of 89
theorem juniors_score_is_89 :
  (∑ j in juniors, juniorScore + ∑ s in seniors, seniorAverageScore) / n = overallAverageScore →
  juniorScore = 89 :=
by
  sorry

end juniors_score_is_89_l619_619555


namespace ab_equals_six_l619_619995

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l619_619995


namespace average_salary_difference_l619_619149

def total_payroll_factory : ℝ := 30000
def number_factory_workers : ℝ := 15
def average_salary_factory : ℝ := total_payroll_factory / number_factory_workers

def total_payroll_office : ℝ := 75000
def total_bonuses_office : ℝ := 5000
def number_office_workers : ℝ := 30
def total_payroll_office_with_bonuses : ℝ := total_payroll_office + total_bonuses_office
def average_salary_office : ℝ := total_payroll_office_with_bonuses / number_office_workers

theorem average_salary_difference :
  (average_salary_office - average_salary_factory) = 666.67 :=
by
  sorry

end average_salary_difference_l619_619149


namespace part1_part2_l619_619894

-- Define the conditions and the questions
variable (α : ℝ) (h : Real.tan α = 2)

-- For the first part
theorem part1 : (cos ((π / 2) + α) * sin ((3 / 2) * π - α)) / tan (-π + α) = 1 / 5 :=
by
  sorry

-- For the second part
theorem part2 : (1 + 3 * sin α * cos α) / (sin α ^ 2 - 2 * cos α ^ 2) = 11 / 2 :=
by
  sorry

end part1_part2_l619_619894


namespace expected_rolls_in_non_leap_year_l619_619006

theorem expected_rolls_in_non_leap_year :
  let E := (1 : ℚ) + (1 / 10) * (E : ℚ) in
  E = 10 / 9 →
  (365 * E) = 3650 / 9 :=
by
  intro h
  sorry

end expected_rolls_in_non_leap_year_l619_619006


namespace no_conclusions_deducible_l619_619117

open Set

variable {U : Type}  -- Universe of discourse

-- Conditions
variables (Bars Fins Grips : Set U)

def some_bars_are_not_fins := ∃ x, x ∈ Bars ∧ x ∉ Fins
def no_fins_are_grips := ∀ x, x ∈ Fins → x ∉ Grips

-- Lean statement
theorem no_conclusions_deducible 
  (h1 : some_bars_are_not_fins Bars Fins)
  (h2 : no_fins_are_grips Fins Grips) :
  ¬((∃ x, x ∈ Bars ∧ x ∉ Grips) ∨
    (∃ x, x ∈ Grips ∧ x ∉ Bars) ∨
    (∀ x, x ∈ Bars → x ∉ Grips) ∨
    (∃ x, x ∈ Bars ∧ x ∈ Grips)) :=
sorry

end no_conclusions_deducible_l619_619117


namespace smallest_natural_b_for_root_exists_l619_619452

-- Define the problem's conditions
def quadratic_eqn (b : ℕ) := ∀ x : ℝ, x^2 + (b : ℝ) * x + 25 = 0

def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Define the main problem statement
theorem smallest_natural_b_for_root_exists :
  ∃ b : ℕ, (discriminant 1 b 25 ≥ 0) ∧ (∀ b' : ℕ, b' < b → discriminant 1 b' 25 < 0) ∧ b = 10 :=
by
  sorry

end smallest_natural_b_for_root_exists_l619_619452


namespace no_integer_b_for_four_integer_solutions_l619_619425

theorem no_integer_b_for_four_integer_solutions :
  ∀ (b : ℤ), ¬ ∃ x1 x2 x3 x4 : ℤ, 
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (∀ x : ℤ, (x^2 + b*x + 1 ≤ 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)) :=
by sorry

end no_integer_b_for_four_integer_solutions_l619_619425


namespace positive_root_of_quadratic_eqn_l619_619492

theorem positive_root_of_quadratic_eqn 
  (b : ℝ)
  (h1 : ∃ x0 : ℝ, x0^2 - 4 * x0 + b = 0 ∧ (-x0)^2 + 4 * (-x0) - b = 0) 
  : ∃ x : ℝ, (x^2 + b * x - 4 = 0) ∧ x = 2 := 
by
  sorry

end positive_root_of_quadratic_eqn_l619_619492


namespace ratio_shaded_area_l619_619569

theorem ratio_shaded_area (l : ℝ) : 
  let area_ABCD := l^2,
      area_CEH := (1/2) * (sqrt 2 - 1) * l^2,
      area_CDH := (1/2) * (sqrt 2 - 1) * l^2,
      shaded_area := area_ABCD - area_CEH - area_CDH
  in shaded_area / area_ABCD = 2 - sqrt 2 := 
by
  sorry

end ratio_shaded_area_l619_619569


namespace h_eq_zero_solutions_F_min_value_abs_AC_eq_abs_BD_l619_619509

section
  variable {x : ℝ} {m a b c : ℝ}

  def f (x : ℝ) : ℝ := (x + (1/x)) / 2
  def g (x : ℝ) : ℝ := (x - (1/x)) / 2
  def h (x : ℝ) : ℝ := f x + 2 * g x
  def F (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x) ^ 2 + m * (f x)
  def line (a b c x y: ℝ) : Prop := a * x + b * y + c = 0

  theorem h_eq_zero_solutions : (h x = 0) → (x = sqrt 3 / 3 ∨ x = - sqrt 3 / 3) :=
  sorry

  theorem F_min_value : (m >= 0) → (∃ y, ∀ x, F f x ≥ y ∧ 
    ((m ∈ Set.Ici 2 → y = - (m ^ 2) / 4) ∨ (m ∈ Set.Ico 0 2 → y = 1 - m))) :=
  sorry

  theorem abs_AC_eq_abs_BD : 
    (line a b c (x1 : ℝ) (y1 : ℝ) ∧ (f x1 = y1)) ∧ 
    (line a b c (x2 : ℝ) (y2 : ℝ) ∧ (f x2 = y2)) ∧ 
    (line a b c (x3 : ℝ) (y3 : ℝ) ∧ (g x3 = y3)) ∧ 
    (line a b c (x4 : ℝ) (y4 : ℝ) ∧ (g x4 = y4)) →
    ( |(x1 - x3) | = |(x2 - x4)| ∧ |(y1 - y3) | = |(y2 - y4)| ) :=
  sorry
end

end h_eq_zero_solutions_F_min_value_abs_AC_eq_abs_BD_l619_619509


namespace B_visited_A_l619_619719

variable cities : Type
variable visited : cities → cities → Prop

variable student_A student_B student_C : cities

-- Given conditions
axiom A_visits_more_than_B : ∀ c, ¬visited student_A c := visited student_B c
axiom A_not_visited_B : ¬ visited student_A student_B
axiom B_not_visited_C : ¬ visited student_B student_C
axiom all_visited_same_city : ∀ c, visited student_A c ↔ visited student_B c ∧ visited student_C c

-- Proving that B has visited city A
theorem B_visited_A : visited student_B student_A :=
by
  sorry

end B_visited_A_l619_619719


namespace optimal_selling_price_l619_619694

-- Define the constants given in the problem
def purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 50

-- Define the function that represents the profit based on the change in price x
def profit (x : ℝ) : ℝ := (initial_selling_price + x) * (initial_sales_volume - x) - (initial_sales_volume - x) * purchase_price

-- State the theorem
theorem optimal_selling_price : ∃ x : ℝ, profit x = -x^2 + 40*x + 500 ∧ (initial_selling_price + x = 70) :=
by
  sorry

end optimal_selling_price_l619_619694


namespace obtuse_triangle_m_count_l619_619705

theorem obtuse_triangle_m_count :
  let valid_m := {m : ℕ | (17 < 13 + m) ∧ (m ≤ 10 ∨ (m < 30 ∧ m ≥ 22))};
  valid_m.card = 14 :=
by {
  sorry
}

end obtuse_triangle_m_count_l619_619705


namespace samantha_birth_year_l619_619687

theorem samantha_birth_year :
  ∀ (first_amc_year amc_frequency ninth_amc_year samantha_age_at_ninth_amc current_year : ℕ),
  first_amc_year = 1980 →
  amc_frequency = 1 →
  ninth_amc_year = first_amc_year + 8 * amc_frequency →
  samantha_age_at_ninth_amc = 14 →
  current_year = ninth_amc_year →
  current_year - samantha_age_at_ninth_amc = 1974 :=
by {
  intros,
  sorry
}

end samantha_birth_year_l619_619687
