import Analysis.SpecialFunctions.Pow
import Analysis.SpecialFunctions.Trigonometric
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Graphs
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Conditional
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.MeasureTheory.ProbabilityTheory.Basic
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.Probability.Conditional
import Mathlib.Tactic
import Mathlib.Topology.EuclideanSpace.Basic

namespace people_per_seat_l360_360624

def ferris_wheel_seats : ℕ := 4
def total_people_riding : ℕ := 20

theorem people_per_seat : total_people_riding / ferris_wheel_seats = 5 := by
  sorry

end people_per_seat_l360_360624


namespace percentage_of_tip_l360_360376

-- Given conditions
def steak_cost : ℝ := 20
def drink_cost : ℝ := 5
def total_cost_before_tip : ℝ := 2 * (steak_cost + drink_cost)
def billy_tip_payment : ℝ := 8
def billy_tip_coverage : ℝ := 0.80

-- Required to prove
theorem percentage_of_tip : ∃ P : ℝ, (P = (billy_tip_payment / (billy_tip_coverage * total_cost_before_tip)) * 100) ∧ P = 20 := 
by {
  sorry
}

end percentage_of_tip_l360_360376


namespace find_lambda_collinear_l360_360081

def vector := ℝ × ℝ

def λ : ℝ := -1

-- Given vectors
def a : vector := (1, 2)
def b : vector := (2, 0)
def c : vector := (1, -2)

-- Scalars to be determined
variable (k λ : ℝ)

def collinear (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, (v1.1 * k, v1.2 * k) = v2

theorem find_lambda_collinear : collinear (λ • ((1, 2) + (2, 0))) (1, -2) :=
by
  split
  existsi -1
  -- Proof goes here
  sorry

end find_lambda_collinear_l360_360081


namespace number_of_ways_to_cut_pipe_l360_360972

theorem number_of_ways_to_cut_pipe : 
  (∃ (x y: ℕ), 2 * x + 3 * y = 15) ∧ 
  (∃! (x y: ℕ), 2 * x + 3 * y = 15) :=
by
  sorry

end number_of_ways_to_cut_pipe_l360_360972


namespace ones_digit_of_largest_power_of_two_dividing_factorial_l360_360862

theorem ones_digit_of_largest_power_of_two_dividing_factorial (n : ℕ) :
  (n = 5) → (nat.digits 10 (2 ^ (31))) = [8] :=
by
  intro h
  rw h
  have fact: nat.fact (2 ^ n) = 32!
  { simp [nat.fact_pow, mul_comm] }
  sorry

end ones_digit_of_largest_power_of_two_dividing_factorial_l360_360862


namespace compound_interest_l360_360978

theorem compound_interest (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (CI : ℝ) :
  SI = 40 → R = 5 → T = 2 → SI = (P * R * T) / 100 → CI = P * ((1 + R / 100) ^ T - 1) → CI = 41 :=
by sorry

end compound_interest_l360_360978


namespace percent_jasmine_after_addition_l360_360322

theorem percent_jasmine_after_addition :
  ∀ (initial_solution_volume initial_jasmine_percent added_jasmine added_water : ℕ),
    initial_solution_volume = 100 →
    initial_jasmine_percent = 10 →
    added_jasmine = 5 →
    added_water = 10 →
    let initial_jasmine_volume := initial_solution_volume * initial_jasmine_percent / 100 in
    let final_jasmine_volume := initial_jasmine_volume + added_jasmine in
    let final_solution_volume := initial_solution_volume + added_jasmine + added_water in
    (float_of_nat final_jasmine_volume / float_of_nat final_solution_volume * 100 ∈ set.Ioo 13.03 13.05)
by
  intros initial_solution_volume initial_jasmine_percent added_jasmine added_water
         h1 h2 h3 h4 initial_jasmine_volume final_jasmine_volume final_solution_volume
  sorry

end percent_jasmine_after_addition_l360_360322


namespace transformImpossible_l360_360129

-- Define the initial grid configuration
def initialGrid : List (List ℕ) := [
  [2, 6, 2],
  [4, 7, 3],
  [3, 6, 5]
]

-- Define the target grid configuration
def targetGrid : List (List ℕ) := [
  [1, 0, 0],
  [0, 2, 0],
  [0, 0, 1]
]

-- Define the condition for adjacency
def adjacent (i j k l : ℕ) : Prop :=
  (i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨
  (j = l ∧ (i = k + 1 ∨ i = k - 1))

-- Define the transformation operation
def validTransform (grid₁ grid₂ : List (List ℕ)) : Prop :=
  ∃ n : ℤ, (∀ i j k l, adjacent i j k l → 
    (grid₂[i][j] = grid₁[i][j] + n ∧ grid₂[k][l] = grid₁[k][l] + n))

-- The theorem we want to prove
theorem transformImpossible :
  ¬ ∃ grid, validTransform initialGrid grid ∧ grid = targetGrid := 
sorry

end transformImpossible_l360_360129


namespace find_x_l360_360338

theorem find_x (t : ℤ) : 
∃ x : ℤ, (x % 7 = 3) ∧ (x^2 % 49 = 44) ∧ (x^3 % 343 = 111) ∧ (x = 343 * t + 17) :=
sorry

end find_x_l360_360338


namespace integer_values_within_interval_l360_360955

theorem integer_values_within_interval : 
  (finset.card (finset.filter (λ x : ℤ, |(x : ℝ)| < 2 * real.pi) finset.Icc (-6 : ℤ) 6)) = 13 := 
begin
  sorry
end

end integer_values_within_interval_l360_360955


namespace maximal_lambda_l360_360006

theorem maximal_lambda (n : ℕ) (n_pos : 0 < n) (x : Fin n → ℝ) 
  (x_pos : ∀ i, 0 < x i) (h_sum : (∑ i, 1 / x i) = n) : 
  ∃ λ, λ = 1 / Real.exp 1 ∧ (∏ i, x i - 1 ≥ λ * (∑ i, x i - n)) :=
begin
  -- Exact proof goes here
  sorry
end

end maximal_lambda_l360_360006


namespace general_formula_range_of_m_l360_360893

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 1 then 2
else if n = 2 then 4
else sequence (n-1) + (sequence (n-1) - sequence (n-2)) * 2

theorem general_formula (n : ℕ) (h1 : sequence 1 = 2) (h2 : sequence 2 = 4)
  (h3 : ∀ n, n ≥ 2 → sequence (n+1) + 2 * sequence (n-1) = 3 * sequence n) :
  sequence n = 2^n :=
sorry

def b_n (n : ℕ) : ℕ := sequence n - 1

def S_n (n : ℕ) : ℝ :=
∑ i in finset.range n, (sequence (i + 1) : ℝ) / (b_n (i + 1) * b_n (i + 2))

theorem range_of_m (m : ℝ) (h : ∀ n, S_n n ≥ (8 * m^2 / 3) - 2 * m) :
  -1/4 ≤ m ∧ m ≤ 1 :=
sorry

end general_formula_range_of_m_l360_360893


namespace polynomial_symmetry_l360_360585

noncomputable theory

variable {α : Type*} [LinearOrderedField α]

def P (x : α) : α

def polynomial_of_degree_six (p : α → α) [is_polynomial_degree_6 : polynomial p]

def derivative_at_zero_is_zero (p : α → α) [is_derivative_zero : deriv p 0 = 0]

def P_is_even : Prop :=
  ∀ x : α, P x = P (-x)

theorem polynomial_symmetry (a b : α) (h_0_lt_a : 0 < a) (h_a_lt_b : a < b) (h_P_a_eq_P_neg_a : P a = P (-a))
  (h_P_b_eq_P_neg_b : P b = P (-b)) (h_P_prime_0_eq_0 : derivative_at_zero_is_zero P) : P_is_even := by
  sorry

end polynomial_symmetry_l360_360585


namespace sequence_inequality_l360_360899

-- Definitions for the sequences a and b
variable (a b : ℕ → ℝ) 

-- Conditions of the problem
variable (h_a : ∀ (n : ℕ), n > 0 → (a n)^2 ≤ a (n-1) * a (n+1))
variable (h_b : ∀ (n : ℕ), n > 0 → (b n)^2 ≤ b (n-1) * b (n+1))

-- Defining the sequence c
noncomputable def c : ℕ → ℝ
| 0       := a 0 * b 0
| (n + 1) := ∑ i in Finset.range (n + 2), (Nat.choose (n + 1) i : ℝ) * a i * b (n + 1 - i)

-- Problem Statement
theorem sequence_inequality : ∀ (n : ℕ), n > 0 → (c a b n)^2 ≤ (c a b (n-1)) * (c a b (n+1)) :=
by
  intros n hn
  sorry

end sequence_inequality_l360_360899


namespace dot_product_magnitude_l360_360161

open Real

variables (a b : ℝ^3) 

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross (u v : ℝ^3) : ℝ^3 := (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)
def dot (u v : ℝ^3) : ℝ := u.1*v.1 + u.2*v.2 + u.3*v.3

axiom norm_a : norm a = 3
axiom norm_b : norm b = 4
axiom cross_norm : norm (cross a b) = 6

theorem dot_product_magnitude : abs (dot a b) = 6 * sqrt 3 := by
  sorry

end dot_product_magnitude_l360_360161


namespace smallest_n_for_473_in_quotient_l360_360079

def is_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_n_for_473_in_quotient (m n : ℕ) (h_rel_prime : is_rel_prime m n) (h_m_lt_n : m < n)
  (h_contains_473 : (∀ ε > 0, ∃ k : ℕ,  | ((m:ℚ) / n : ℚ) - (k / 1000) | < ε ∧ k % 1000 = 473)) : n = 477 :=
sorry

end smallest_n_for_473_in_quotient_l360_360079


namespace radius_of_circle_with_area_64pi_l360_360634

def circle_radius (A : ℝ) : ℝ := 
  real.sqrt (A / real.pi)

theorem radius_of_circle_with_area_64pi :
  circle_radius (64 * real.pi) = 8 :=
by sorry

end radius_of_circle_with_area_64pi_l360_360634


namespace tom_has_10_violet_balloons_l360_360685

theorem tom_has_10_violet_balloons
  (initial_balloons : ℕ)
  (fraction_given : ℚ)
  (balloons_given : ℕ)
  (balloons_left : ℕ)
  (h1 : initial_balloons = 30)
  (h2 : fraction_given = 2/3)
  (h3 : balloons_given = (fraction_given * initial_balloons).toNat)
  (h4 : balloons_left = initial_balloons - balloons_given) :
  balloons_left = 10 :=
by
  rw [h1, h2] at h3
  have h5 : balloons_given = 20 := by norm_num [h2, h1]
  rw [h5] at h4
  have h6 : balloons_left = 10 := by norm_num [h1, h5]
  exact h6

end tom_has_10_violet_balloons_l360_360685


namespace determine_b_l360_360582

variable (b : ℝ)
def p (x : ℝ) := 5 * x - 4
def q (x : ℝ) := 4 * x - b

theorem determine_b (h : p (q 5) = 16) : b = 16 := by
  sorry

end determine_b_l360_360582


namespace find_positive_number_l360_360759

theorem find_positive_number (x : ℝ) (h_pos : 0 < x) (h_eq : (2 / 3) * x = (49 / 216) * (1 / x)) : x = 24.5 :=
by
  sorry

end find_positive_number_l360_360759


namespace equation_solution_l360_360938

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l360_360938


namespace min_value_of_f_at_extreme_point_l360_360097

def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * x - 3 * Real.log x

theorem min_value_of_f_at_extreme_point :
  (∀ x : ℝ, x ≠ 0 → f x 2 = (1/2) * x^2 - 2 * x - 3 * Real.log x ∧ (∃ x : ℝ, (x ≠ 0) ∧ f x 2 = 0) → x = 3) →
  f 3 2 = - (3 / 2) - 3 * Real.log 3 :=
by
  sorry

end min_value_of_f_at_extreme_point_l360_360097


namespace total_ways_to_fill_grid_l360_360800

def validMatrix (A B C D E F G H I : ℕ) : Prop := 
  (A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0 ∧ F > 0 ∧ G > 0 ∧ H > 0 ∧ I > 0) ∧
  (D % A = 0 ∧ E % B = 0 ∧ F % C = 0 ∧ G % D = 0 ∧ H % E = 0 ∧ I % F = 0) ∧
  (B % A = 0 ∧ C % B = 0 ∧ E % D = 0 ∧ F % E = 0 ∧ H % G = 0 ∧ I % H = 0)

theorem total_ways_to_fill_grid : 
  (Σ' (A B C D E F G H I : ℕ), validMatrix A B C D E F G H I).to_list.length = 136 := 
sorry

end total_ways_to_fill_grid_l360_360800


namespace mary_pick_three_red_marbles_l360_360209

-- Define the conditions
def rm_count := 8
def bm_count := 7
def total_marbles := rm_count + bm_count
def trials := 6
def red_marble_prob := (8:ℚ) / total_marbles
def blue_marble_prob := (7:ℚ) / total_marbles

-- Define the binomial coefficient function
noncomputable def binom_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

-- Probability of picking exactly three red marbles in 6 trials
noncomputable def three_red_marbles_probability :=
  binom_coefficient 6 3 * (red_marble_prob ^ 3) * (blue_marble_prob ^ 3:ℚ)

-- Statement to prove the probability is as calculated
theorem mary_pick_three_red_marbles :
  three_red_marbles_probability = 6881280 / 38107875 :=
by
  sorry

end mary_pick_three_red_marbles_l360_360209


namespace subset_implies_range_of_m_l360_360023

variable (m : ℝ)
def A := set.Iic m  -- A = (-∞, m]
def B := set.Ioc 1 2  -- B = (1, 2]

theorem subset_implies_range_of_m (h : B ⊆ A) : m ∈ set.Ici 2 := 
by sorry

end subset_implies_range_of_m_l360_360023


namespace smallest_pos_int_ends_in_6_divisible_by_11_l360_360298

theorem smallest_pos_int_ends_in_6_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 6 ∧ 11 ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 6 ∧ 11 ∣ m → n ≤ m := by
  sorry

end smallest_pos_int_ends_in_6_divisible_by_11_l360_360298


namespace triangle_B_eq_45_l360_360111

theorem triangle_B_eq_45 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : A + B + C = π)
  (h8 : a > 0) (h9 : b > 0) (h10 : c > 0)
  (h_law_of_sines : sin A / a = cos B / b)
  (h_equal_sides : a = b) :
  B = π / 4 :=
by
  sorry

end triangle_B_eq_45_l360_360111


namespace range_of_a_l360_360976

noncomputable def min_expr (x: ℝ) : ℝ := x + 2/(x - 2)

theorem range_of_a (a: ℝ) : 
  (∀ x > 2, a ≤ min_expr x) ↔ a ≤ 2 + 2 * Real.sqrt 2 := 
by
  sorry

end range_of_a_l360_360976


namespace average_percent_score_l360_360595

def num_students : ℕ := 180

def score_distrib : List (ℕ × ℕ) :=
[(95, 12), (85, 30), (75, 50), (65, 45), (55, 30), (45, 13)]

noncomputable def total_score : ℕ :=
(95 * 12) + (85 * 30) + (75 * 50) + (65 * 45) + (55 * 30) + (45 * 13)

noncomputable def average_score : ℕ :=
total_score / num_students

theorem average_percent_score : average_score = 70 :=
by 
  -- Here you would provide the proof, but for now we will leave it as:
  sorry

end average_percent_score_l360_360595


namespace largest_integer_x_l360_360715

theorem largest_integer_x : ∃ x : ℤ, (7 - 3 * x > 20) ∧ (x ≥ -10) ∧ ∀ y : ℤ, (7 - 3 * y > 20) ∧ (y ≥ -10) → y ≤ x :=
begin
  use -5,
  split,
  { -- Proof of 7 - 3 * (-5) > 20 goes here
    sorry
  },
  split,
  { -- Proof of -5 ≥ -10 goes here
    sorry
  },
  { -- Proof that for all integers y that satisfy the conditions, y ≤ -5 goes here
    sorry
  }
end

end largest_integer_x_l360_360715


namespace binomial_coeff_and_coeff_of_x8_l360_360560

theorem binomial_coeff_and_coeff_of_x8 (x : ℂ) :
  let expr := (x^2 + 4*x + 4)^5
  let expansion := (x + 2)^10
  ∃ (binom_coeff_x8 coeff_x8 : ℤ),
    binom_coeff_x8 = 45 ∧ coeff_x8 = 180 :=
by
  sorry

end binomial_coeff_and_coeff_of_x8_l360_360560


namespace repeating_decimals_sum_l360_360418

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l360_360418


namespace derivative_limit_l360_360538

variable {α β : Type*}
variables (f : ℝ → ℝ)

noncomputable def derivative_at (f : ℝ → ℝ) (a : ℝ) :=
  ∃! L : ℝ, filter.tendsto (λ h, (f (a + h) - f a) / h) (nhds_within 0 (set.Ioi 0)) (𝓝 L)

theorem derivative_limit (h : derivative_at f 2) (h_deriv : derivative_at f 2 ∈ { -2 }) :
  filter.tendsto (λ x, (f x - f 2) / (x - 2)) (nhds_within 2 (set.Ioi 2)) (𝓝 (-2)) := sorry

end derivative_limit_l360_360538


namespace solve_system_l360_360302

theorem solve_system :
  ∀ (x y : ℝ) (triangle : ℝ), 
  (2 * x - 3 * y = 5) ∧ (x + y = triangle) ∧ (x = 4) →
  (y = 1) ∧ (triangle = 5) :=
by
  -- Skipping the proof steps
  sorry

end solve_system_l360_360302


namespace density_function_Y_l360_360074

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-x^2 / 2)

theorem density_function_Y (y : ℝ) (hy : 0 < y) : 
  (∃ (g : ℝ → ℝ), (∀ y, g y = (1 / Real.sqrt (2 * Real.pi * y)) * Real.exp (- y / 2))) :=
sorry

end density_function_Y_l360_360074


namespace remainder_when_sum_div_by_8_l360_360093

theorem remainder_when_sum_div_by_8 (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end remainder_when_sum_div_by_8_l360_360093


namespace sum_of_volume_and_icing_area_l360_360777

-- Define the cube parameters
def cube_side_length : ℝ := 3

-- Define the points and lengths for triangle ABC
structure Point (P : Type*) :=
(x : P) (y : P) (z : P)

def A : Point ℝ := ⟨0, 0, 0⟩
def B : Point ℝ := ⟨cube_side_length / 2, 0, 0⟩
def C : Point ℝ := ⟨cube_side_length, 0, 0⟩

-- Define the lengths
def length_AB : ℝ := cube_side_length / 2
def length_BC : ℝ := length_AB
def length_AC : ℝ := cube_side_length

-- Calculate the area of the triangle ABC
def area_ΔABC : ℝ := (1 / 2) * cube_side_length * length_AB

-- Calculate the volume of the triangular-based pyramid
def volume_triangle_pyramid : ℝ := area_ΔABC * cube_side_length

-- Calculate the icing area
def icing_area : ℝ := area_ΔABC + 2 * (cube_side_length + length_AB + length_BC)

-- Sum of the volume and the icing area
def total_sum : ℝ := volume_triangle_pyramid + icing_area

#check volume_triangle_pyramid -- to check types
#check icing_area            -- to check types
#check total_sum

theorem sum_of_volume_and_icing_area : total_sum = 21 := 
sorry

end sum_of_volume_and_icing_area_l360_360777


namespace negation_of_proposition_l360_360656

theorem negation_of_proposition :
  (∀ x y : ℝ, (x * y = 0 → x = 0 ∨ y = 0)) →
  (∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
sorry

end negation_of_proposition_l360_360656


namespace perspective_triangle_area_l360_360765

theorem perspective_triangle_area :
  ∀ (ABC A'B'C' : Type) [equilateral_triangle ABC 1],
  let S_original := (sqrt 3) / 4,
      relation := (sqrt 2) / 4,
      S_perspective := S_original * relation
  in S_perspective = (sqrt 6) / 16 :=
by
  sorry

end perspective_triangle_area_l360_360765


namespace pauline_minimum_spend_l360_360602

theorem pauline_minimum_spend :
  let total_amount : ℝ := 250
  let selected_items_total : ℝ := 100
  let remaining_items : ℝ := total_amount - selected_items_total
  let discount_15 : ℝ := selected_items_total * 0.85
  let bogo_offer : ℝ := selected_items_total * 0.5
  let sales_tax_rate : ℝ := 0.08
  let total_with_discount_15 : ℝ := discount_15 + remaining_items
  let total_with_bogo_offer : ℝ := bogo_offer + remaining_items
  let total_with_tax_15 : ℝ := total_with_discount_15 * (1 + sales_tax_rate)
  let total_with_tax_bogo : ℝ := total_with_bogo_offer * (1 + sales_tax_rate)
  in total_with_tax_15 > total_with_tax_bogo ∧ total_with_tax_bogo = 216 := by
    let total_amount : ℝ := 250
    let selected_items_total : ℝ := 100
    let remaining_items : ℝ := total_amount - selected_items_total
    let discount_15 : ℝ := selected_items_total * 0.85
    let bogo_offer : ℝ := selected_items_total * 0.5
    let sales_tax_rate : ℝ := 0.08
    let total_with_discount_15 : ℝ := discount_15 + remaining_items
    let total_with_bogo_offer : ℝ := bogo_offer + remaining_items
    let total_with_tax_15 : ℝ := total_with_discount_15 * (1 + sales_tax_rate)
    let total_with_tax_bogo : ℝ := total_with_bogo_offer * (1 + sales_tax_rate)
    refine ⟨_, _⟩
    sorry
    sorry

end pauline_minimum_spend_l360_360602


namespace max_xyz_l360_360591

theorem max_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) 
(h5 : x^2 + y^2 + z^2 = x * z + y * z + x * y) : xyz ≤ (8 / 27) :=
sorry

end max_xyz_l360_360591


namespace tens_digit_of_3_pow_2013_l360_360300

theorem tens_digit_of_3_pow_2013 : (3^2013 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_3_pow_2013_l360_360300


namespace repeating_decimal_sum_l360_360268

theorem repeating_decimal_sum (c d : ℕ) (h : 7 / 19 = (c * 10 + d) / 99) : c + d = 9 :=
sorry

end repeating_decimal_sum_l360_360268


namespace ellipse_eccentricity_l360_360062

-- Definitions
def ellipse_eq (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def condition1 (a b : ℝ) : Prop := a > b ∧ b > 0
def dot_product_oa_af_zero (OA AF : ℝ² → ℝ) : Prop := ∀ u v, OA u = 0 → AF v = 0
def dot_product_oa_of_half (OA OF : ℝ² → ℝ) : Prop := ∀ u v, OA u * OF v = (1/2) * (OF v)^2

-- Main statement
theorem ellipse_eccentricity (a b c e : ℝ) (x y : ℝ) 
  (h1 : ellipse_eq x y a b) 
  (h2 : condition1 a b) 
  (h3 : dot_product_oa_af_zero (λ (u : ℝ), x) (λ (v : ℝ), y)) 
  (h4 : dot_product_oa_of_half (λ (u : ℝ), x) (λ (v : ℝ), y)) :
  e = (√10 - √2) / 2 := 
sorry

end ellipse_eccentricity_l360_360062


namespace sufficient_but_not_necessary_condition_l360_360030

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  ∀ (z : ℂ), (z = (3 * complex.i - a) / complex.i) → 
  (a > -1) → 
  (z.re > 0 ∧ z.im > 0) := sorry

end sufficient_but_not_necessary_condition_l360_360030


namespace average_of_middle_three_l360_360250

-- Define the conditions based on the problem statement
def isPositiveWhole (n: ℕ) := n > 0
def areDifferent (a b c d e: ℕ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def isMaximumDifference (a b c d e: ℕ) := max a (max b (max c (max d e))) - min a (min b (min c (min d e)))
def isSecondSmallest (a b c d e: ℕ) := b = 3 ∧ (a < b ∧ (c < b ∨ d < b ∨ e < b) ∧ areDifferent a b c d e)
def totalSumIs30 (a b c d e: ℕ) := a + b + c + d + e = 30

-- Average of the middle three numbers calculated
theorem average_of_middle_three {a b c d e: ℕ} (cond1: isPositiveWhole a)
  (cond2: isPositiveWhole b) (cond3: isPositiveWhole c) (cond4: isPositiveWhole d)
  (cond5: isPositiveWhole e) (cond6: areDifferent a b c d e) (cond7: b = 3)
  (cond8: max a (max c (max d e)) - min a (min c (min d e)) = 16)
  (cond9: totalSumIs30 a b c d e) : (a + c + d) / 3 = 4 :=
by sorry

end average_of_middle_three_l360_360250


namespace sum_of_repeating_decimals_l360_360413

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l360_360413


namespace dot_product_magnitude_l360_360181

variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜 → 𝕜} 

-- Define the conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_a_cross_b : ℝ := 6

-- Theorem statement
theorem dot_product_magnitude :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∥a × b∥ = norm_a_cross_b) → abs ((a.toReal).dot (b.toReal)) = 6 * sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end dot_product_magnitude_l360_360181


namespace unit_digit_of_12_pow_100_l360_360760

def unit_digit_pow (a: ℕ) (n: ℕ) : ℕ :=
  (a ^ n) % 10

theorem unit_digit_of_12_pow_100 : unit_digit_pow 12 100 = 6 := by
  sorry

end unit_digit_of_12_pow_100_l360_360760


namespace mark_sandwiches_l360_360392

/--
Each day of a 6-day workweek, Mark bought either an 80-cent donut or a $1.20 sandwich. 
His total expenditure for the week was an exact number of dollars.
Prove that Mark bought exactly 3 sandwiches.
-/
theorem mark_sandwiches (s d : ℕ) (h1 : s + d = 6) (h2 : ∃ k : ℤ, 120 * s + 80 * d = 100 * k) : s = 3 :=
by
  sorry

end mark_sandwiches_l360_360392


namespace positive_factors_of_2550_with_more_than_4_factors_l360_360839

theorem positive_factors_of_2550_with_more_than_4_factors :
  let factors := 3
  ∃ (d : ℕ), prime_factors 2550 = [2, 3, 5, 5, 17] ∧ d = factors := by
  sorry

end positive_factors_of_2550_with_more_than_4_factors_l360_360839


namespace first_crane_height_l360_360386

-- Define building heights
def building1 := 200
def building2 := 100
def building3 := 140

-- Define crane heights
def crane2 := 120
def crane3 := 147

-- Average height of buildings
def avg_building_height := (building1 + building2 + building3) / 3

-- Given condition: Cranes are on average 13% taller than buildings
def avg_crane_height := 1.13 * avg_building_height

-- Define the height of the first crane
def H := (avg_crane_height * 3) - (crane2 + crane3)

-- Prove that the height of the first crane (H) is 230.2 feet
theorem first_crane_height : H = 230.2 :=
by sorry

end first_crane_height_l360_360386


namespace repeating_decimal_sum_l360_360408

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l360_360408


namespace math_proof_problem_l360_360118

def property_o (S : Matrix (Fin 3) (Fin 9) ℕ) (u : Fin 3 → ℕ) : Prop :=
  ∀ k : Fin 9, ∃ i : Fin 3, S i k ≤ u i

def has_diff_cols_least_vals (S : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∃ (col1 col2 col3 : Fin 3),
    col1 ≠ col2 ∧ col2 ≠ col3 ∧ col1 ≠ col3 ∧
    ∃ (u : Fin 3 → ℕ),
      (∀ i : Fin 3, u i = min (S i 0) (min (S i 1) (S i 2))) ∧
      (∀ i : Fin 3, ∃ k : Fin 3, u i = S i k)

def unique_satisfying_column (P : Matrix (Fin 3) (Fin 9) ℕ) (S : Matrix (Fin 3) (Fin 3) ℕ) (u : Fin 3 → ℕ) : Prop :=
  ∃! k* : Fin 9, k* > 2 ∧ 
  ∃ S' : Matrix (Fin 3) (Fin 3) ℕ,
    S' = λ i j, if j < 2 then S i j else P i k* ∧ property_o S' u

theorem math_proof_problem :
  ∀ (P : Matrix (Fin 3) (Fin 9) ℕ),
    (∀ i : Fin 3, Finset.card (Finset.image (λ j, P i j) Finset.univ) = 9) →
    (∀ j : Fin 6, ∑ i : Fin 3, P i j = 1) →
    P 1 6 = 0 ∧ P 2 7 = 0 ∧ P 3 8 = 0 →
    (∀ j ∈ {2, 7, 8}, ∀ i : Fin 3, P i j > 1) →
    has_diff_cols_least_vals (P.map (λ x, if x ≤ 0 then 0 else x)) ∧
    ∃ (u : Fin 3 → ℕ), property_o P u →
    unique_satisfying_column P (P.map (λ x, if x ≤ 0 then 0 else x).submatrix (λ i, i : Fin 3) (λ j, j)) u :=
sorry

end math_proof_problem_l360_360118


namespace problem_solution_l360_360923

-- Define the conditions
def PropositionI (α β : Plane) (a : Line) (h1 : a ∈ α) (b : Line) (h2 : b ∈ β) (h3 : skew a b)
  (c : Line) (h4 : c = α ∩ β) : Prop := 
∃ p : Point, p ∈ c → (p ∈ a ∨ p ∈ b → ¬(p ∈ a ∧ p ∈ b))

def PropositionII : Prop := 
¬∃ (ℓ : ℕ → Line), ∀ i j, i ≠ j → skew (ℓ i) (ℓ j)

-- The main theorem
theorem problem_solution : ¬PropositionI ∧ ¬PropositionII := 
by
  sorry

end problem_solution_l360_360923


namespace sam_distance_walked_when_meet_l360_360474

theorem sam_distance_walked_when_meet:
  ∀ (d_t d_f d_s s_f s_s t: ℝ), 
  d_t = 40 → 
  d_t = d_f + d_s → 
  s_f = 4 → 
  s_s = 4 → 
  d_f = s_f * t → 
  d_s = s_s * t → 
  t = d_t / (s_f + s_s) →
  d_s = 20 := 
by
  intros d_t d_f d_s s_f s_s t h1 h2 h3 h4 h5 h6 h7
  rw [h1, h3, h4] at h7
  rw [h3] at h5
  rw [h4] at h6
  have h8: t = 5 := by 
    calc d_t / (s_f + s_s) = 40 / 8 :=
      by rw h7
    ... = 5 : by norm_num
  rw h8 at h6
  calc d_s = s_s * 5 : by rw h6
  ... = 20 : by norm_num

end sam_distance_walked_when_meet_l360_360474


namespace max_single_player_salary_l360_360996

theorem max_single_player_salary (n : ℕ) (m : ℕ) (T : ℕ) (n_pos : n = 18) (m_pos : m = 20000) (T_pos : T = 800000) :
  ∃ x : ℕ, (∀ y : ℕ, y ≤ x → y ≤ 460000) ∧ (17 * m + x ≤ T) :=
by
  sorry

end max_single_player_salary_l360_360996


namespace sum_of_roots_of_quadratic_eq_l360_360727

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l360_360727


namespace fractional_equation_l360_360751

def distanceA : ℝ := 25
def distanceB : ℝ := 32
def timeSaved : ℝ := 1 / 4
def speedIncrease : ℝ := 1.6 

-- Define the problem statement
theorem fractional_equation (x : ℝ) (hx : x > 0) :
  (distanceA / x) - (distanceB / (speedIncrease * x)) = timeSaved :=
sorry

end fractional_equation_l360_360751


namespace quotient_transformation_l360_360267

theorem quotient_transformation (A B : ℕ) (h1 : B ≠ 0) (h2 : (A : ℝ) / B = 0.514) :
  ((10 * A : ℝ) / (B / 100)) = 514 :=
by
  -- skipping the proof
  sorry

end quotient_transformation_l360_360267


namespace cross_square_side_length_l360_360776

theorem cross_square_side_length (A : ℝ) (s : ℝ) (h1 : A = 810) 
(h2 : (2 * (s / 2)^2 + 2 * (s / 4)^2) = A) : s = 36 := by
  sorry

end cross_square_side_length_l360_360776


namespace find_x_l360_360539

def max_diff_eq_sum (S : Set ℝ) : Prop :=
  let max_elem := max (max (max 1 2) 3) x
  let min_elem := min (min (min 1 2) 3) x
  let sum_elems := 1 + 2 + 3 + x
  max_elem - min_elem = sum_elems

theorem find_x :
  ∀ x : ℝ, (max_diff_eq_sum {1, 2, 3, x}) → x = -3 / 2 :=
by
  sorry

end find_x_l360_360539


namespace circle_radius_difference_l360_360615

-- Non-met conditions:
noncomputable theory

-- Definition of the problem's conditions
def UnitDisk (C : Type) := { disk : C // ∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 6) ∧ (1 ≤ j ∧ j ≤ 6) ∧ i ≠ j → disjoint C i C j }

def TangentCycle (C : Type) := ∀ i : Fin 6, Tangent C i (Fin.cycleNext i)

-- Definition of the smallest and largest possible circles containing the disks
def MinRadius : ℝ := 3
def MaxRadius : ℝ := 2 + Real.sqrt 3

-- The theorem to be proved
theorem circle_radius_difference (C: ℝ) (h1: UnitDisk C) (h2: TangentCycle C) : MaxRadius - MinRadius = Real.sqrt 3 - 1 :=
by sorry

end circle_radius_difference_l360_360615


namespace smaller_angle_at_seven_oclock_l360_360701

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l360_360701


namespace valid_paths_grid_l360_360524

/-- A formal statement of the combinatorial problem involving paths on a grid with forbidden segments. -/
theorem valid_paths_grid (A B : Point) (forbidden_segments : list Segment) :
  count_valid_paths A B forbidden_segments = 237 := 
sorry

end valid_paths_grid_l360_360524


namespace exists_two_numbers_in_set_l360_360037

theorem exists_two_numbers_in_set 
  (a : ℕ → ℝ)
  (h : ∀ i j, i < j → a i < a j)
  (ha_len : ∀ i, i < 7) :
  ∃ x y, x ≠ y ∧ ∃ i j, 0 ≤ i < j < 7 ∧ (x = a j) ∧ (y = a i)
  ∧ 0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 / Real.sqrt 3 :=
by
  sorry

end exists_two_numbers_in_set_l360_360037


namespace sum_seq_first_20_terms_l360_360920

noncomputable def a_n (n : ℕ) : ℚ := 6 * n - n^2
noncomputable def b_n (n : ℕ) : ℚ := a_n n - (if h : n ≥ 1 then a_n (n - 1) else 0)
noncomputable def seq (n : ℕ) : ℚ := 1 / (b_n n * b_n (n + 1))

theorem sum_seq_first_20_terms :
  ∑ i in Finset.range 20, seq i = -4 / 35 := sorry

end sum_seq_first_20_terms_l360_360920


namespace dot_product_magnitude_l360_360156

variable {a b : EuclideanSpace ℝ (Fin n)}
variable (norm_a : ∥a∥ = 3)
variable (norm_b : ∥b∥ = 4)
variable (norm_cross_ab : ∥a × b∥ = 6)

theorem dot_product_magnitude : ∥a∥ = 3 → ∥b∥ = 4 → ∥a × b∥ = 6 → |a ∘ b| = 6 * real.sqrt 3 := 
by 
  intro norm_a norm_b norm_cross_ab
  sorry

end dot_product_magnitude_l360_360156


namespace sine_triangle_inequality_l360_360198

theorem sine_triangle_inequality 
  {a b c : ℝ} (h_triangle : a + b + c ≤ 2 * Real.pi) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (ha_lt_pi : a < Real.pi) (hb_lt_pi : b < Real.pi) (hc_lt_pi : c < Real.pi) :
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sine_triangle_inequality_l360_360198


namespace ratios_arithmetic_seq_l360_360799

-- Define points and line segments corresponding to the problem's conditions
variables {A B C M P Q N : Type*} 

-- Define the median of the triangle
def is_median (A B C M : Type*) : Prop := ∃ (midpoint : Type*), midpoint = M ∧ (1/2) * (B + C) = M

-- Assumptions about the points' locations
def line_intersects (A B C P Q N M : Type*) : Prop :=
  ∃ (line : Type*), line ∈ span({A, B, C}) ∧ 
  (line ∩ span({A, B}) = {P}) ∧ 
  (line ∩ span({A, C}) = {Q}) ∧ 
  (line ∩ span({A, M}) = {N})

-- Definition of the ratios
def ratio_1 (A B P : Type*) : Type* := AB / AP
def ratio_2 (A M N : Type*) : Type* := AM / AN
def ratio_3 (A C Q : Type*) : Type* := AC / AQ

-- Theorem to prove the ratios form an arithmetic sequence
theorem ratios_arithmetic_seq (h_median : is_median A B C M) (h_intersects : line_intersects A B C P Q N M) :
  (ratio_1 A B P) + (ratio_3 A C Q) = 2 * (ratio_2 A M N) :=
sorry

end ratios_arithmetic_seq_l360_360799


namespace work_completion_time_l360_360774

theorem work_completion_time (A B C : Type) [has_div A] [has_div B] [has_div C]
(hA : A = 6) (hB : B = 5) (hC : C = 7.5) :
  1 / (1 / A + 1 / B + 1 / C) = 2 :=
by {
  sorry
}

end work_completion_time_l360_360774


namespace marathons_yards_l360_360781

theorem marathons_yards
  (miles_per_marathon : ℕ)
  (yards_per_marathon : ℕ)
  (miles_in_yard : ℕ)
  (marathons_run : ℕ)
  (total_miles : ℕ)
  (total_yards : ℕ)
  (y : ℕ) :
  miles_per_marathon = 30
  → yards_per_marathon = 520
  → miles_in_yard = 1760
  → marathons_run = 8
  → total_miles = (miles_per_marathon * marathons_run) + (yards_per_marathon * marathons_run) / miles_in_yard
  → total_yards = (yards_per_marathon * marathons_run) % miles_in_yard
  → y = 640 := 
by
  intros
  sorry

end marathons_yards_l360_360781


namespace problem_l360_360491

theorem problem (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := by
  sorry

end problem_l360_360491


namespace select_students_l360_360882

theorem select_students (n m : ℕ) (hn : n = 4) (hm : m = 3) :
  (nat.choose 4 3) * (nat.choose 3 1) + (nat.choose 4 2) * (nat.choose 3 2) + (nat.choose 4 1) * (nat.choose 3 3) = 34 :=
by 
  sorry

end select_students_l360_360882


namespace equation_solution_l360_360936

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l360_360936


namespace no_common_root_l360_360017

theorem no_common_root (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) := 
sorry

end no_common_root_l360_360017


namespace equilateral_A_l360_360568

theorem equilateral_A'B'C' (A B C P A' B' C' : Point) 
  (h₁ : ∠ BPC = ∠ A + 60)
  (h₂ : ∠ APC = ∠ B + 60)
  (h₃ : ∠ APB = ∠ C + 60)
  (h₄ : ∀ Q, Q ≠ P → (Q ∈ line_through A P ∨ Q ∈ line_through B P ∨ Q ∈ line_through C P) → Q ∉ circle_ABC)
  : equilateral (triangle A' B' C') :=
by
  sorry

end equilateral_A_l360_360568


namespace orange_balls_count_l360_360357

-- Defining all the balls and conditions
def total_balls : ℕ := 120
def red_balls : ℕ := 30
def blue_balls : ℕ := 20
def yellow_balls : ℕ := 10
def green_balls : ℕ := 5
def pink_balls : ℕ := 2 * green_balls
def orange_balls : ℕ := 3 * pink_balls
def purple_balls : ℕ := orange_balls - pink_balls

-- Proving the main statement about the number of orange balls
theorem orange_balls_count : orange_balls = 30 :=
by
  have h1 : red_balls + blue_balls + yellow_balls + green_balls = 65 := by
    simp [red_balls, blue_balls, yellow_balls, green_balls]
  have h2 : total_balls - (red_balls + blue_balls + yellow_balls + green_balls) = 55 := by
    rw [h1]
    simp [total_balls]
  have h3 : pink_balls = 2 * green_balls := by
    simp [pink_balls]
  have h4 : pink_balls = 10 := by
    simp [green_balls, h3]
  have h5 : orange_balls = 3 * pink_balls := by
    simp [orange_balls]
  have h6 : orange_balls = 3 * 10 := by
    rw [h4]
    simp [h5]
  have h7 : orange_balls = 30 := by
    rw [h6]
    simp
  exact h7

end orange_balls_count_l360_360357


namespace correct_statements_count_l360_360794

/-- Checking conditions about planar geometry statements -/
def statements : List Prop :=
  [ True, -- A triangle is definitely a planar figure
    True, -- The diagonals of a quadrilateral intersecting at one point makes it planar
    False, -- The center of a circle and two points on the circle can determine a plane
    True -- At most, three parallel lines can determine three planes
  ]

/-- The number of true statements in the list is 3 -/
theorem correct_statements_count : statements.count (λ s, s) = 3 :=
  sorry

end correct_statements_count_l360_360794


namespace find_max_value_l360_360029

-- Define the function f
noncomputable def f (θ : ℝ) : ℝ :=
  (2 * Real.sin θ * Real.cos θ) / ((Real.sin θ + 1) * (Real.cos θ + 1))

-- Define the interval of θ
def θ_interval (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

-- Define the maximum value
def max_value : ℝ :=
  6 - 4 * Real.sqrt 2

-- The theorem to be proved
theorem find_max_value : ∃ θ ∈ Ioo 0 (Real.pi / 2), f θ = max_value := 
by
  sorry

end find_max_value_l360_360029


namespace max_moves_l360_360141

theorem max_moves (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ moves : ℕ, moves = a * c + b * a + c * b :=
by
  use a * c + b * a + c * b
  sorry

end max_moves_l360_360141


namespace rectangular_playground_vertical_length_l360_360349

theorem rectangular_playground_vertical_length (s : ℝ) (h : ℝ) (v : ℝ) (sq_side : ℝ) (rec_hor : ℝ) (eq_times : 4 * sq_side = 2 * (rec_hor + v)) :
  v = 15 :=
by
  rw [sq_side, rec_hor, ←eq_times]
  norm_num
  sorry
  -- We use provided conditions :
  -- Let sq_side = 12
  -- Let rec_hor = 9
  -- Given that eq_times : 4 * sq_side = 2 * (rec_hor + v)
  -- We need to prove v = 15

end rectangular_playground_vertical_length_l360_360349


namespace product_of_solutions_eq_neg_three_l360_360872

theorem product_of_solutions_eq_neg_three :
  (∏ x in {x : ℝ | 2^(3*x + 1) - 17*2^(2*x) + 2^(x + 3) = 0}, x) = -3 :=
by
  sorry

end product_of_solutions_eq_neg_three_l360_360872


namespace min_matches_to_win_champion_min_total_matches_if_wins_11_l360_360684

-- Define the conditions and problem in Lean 4
def teams := ["A", "B", "C"]
def players_per_team : ℕ := 9
def initial_matches : ℕ := 0

-- The minimum number of matches the champion team must win
theorem min_matches_to_win_champion (H : ∀ t ∈ teams, t ≠ "Champion" → players_per_team = 0) :
  initial_matches + 19 = 19 :=
by
  sorry

-- The minimum total number of matches if the champion team wins 11 matches
theorem min_total_matches_if_wins_11 (wins_by_champion : ℕ := 11) (H : wins_by_champion = 11) :
  initial_matches + wins_by_champion + (players_per_team * 2 - wins_by_champion) + 4 = 24 :=
by
  sorry

end min_matches_to_win_champion_min_total_matches_if_wins_11_l360_360684


namespace angle_BAD_40_degrees_l360_360686

theorem angle_BAD_40_degrees
  (A B C D : Type)
  [Triangle ABC]
  [Triangle ADC]
  (h_AB_BC : AB = BC)
  (h_AD_DC : AD = DC)
  (h_D_inside_ABC : D ∈ int_triangle ABC)
  (h_ABC_50 : ∠ ABC = 50)
  (h_ADC_130 : ∠ ADC = 130) :
  ∠ BAD = 40 :=
by
  sorry

end angle_BAD_40_degrees_l360_360686


namespace problem_statement_l360_360951
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l360_360951


namespace solution_l360_360894

noncomputable def f : ℕ → ℕ := sorry

theorem solution :
  (strictly_increasing f ∧ ∀ m n : ℕ, f (n + f m) = f n + m + 1) →
  f 2023 = 2024 := 
sorry

end solution_l360_360894


namespace smallest_Y_l360_360150

noncomputable def S : ℕ := 111111111000

theorem smallest_Y : ∃ Y : ℕ, Y = S / 18 ∧ Y = 6172839500 := by
  use 6172839500
  split
  · calc
      S / 18 = 111111111000 / 18 := by sorry
    _ = 6172839500 := by sorry
  · exact rfl

end smallest_Y_l360_360150


namespace probability_different_questions_l360_360782
  
  theorem probability_different_questions :
    let total_events := 4 ^ 4
    let favorable_outcomes := 4!
    let probability := favorable_outcomes.to_rat / total_events
    probability = (3:ℚ) / 32 :=
  by
    sorry
  
end probability_different_questions_l360_360782


namespace problem_statement_l360_360949
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l360_360949


namespace angle_B_is_pi_over_3_l360_360984

theorem angle_B_is_pi_over_3
  (A B C a b c : ℝ)
  (h1 : b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2)
  (h2 : 0 < B)
  (h3 : B < Real.pi)
  (h4 : 0 < A)
  (h5 : A < Real.pi)
  (h6 : 0 < C)
  (h7 : C < Real.pi) :
  B = Real.pi / 3 :=
by
  sorry

end angle_B_is_pi_over_3_l360_360984


namespace number_of_movies_in_series_l360_360683

variables (watched_movies remaining_movies total_movies : ℕ)

theorem number_of_movies_in_series 
  (h_watched : watched_movies = 4) 
  (h_remaining : remaining_movies = 4) :
  total_movies = watched_movies + remaining_movies :=
by
  sorry

end number_of_movies_in_series_l360_360683


namespace smaller_circle_radius_l360_360132

-- Given conditions
def larger_circle_radius : ℝ := 10
def number_of_smaller_circles : ℕ := 7

-- The goal
theorem smaller_circle_radius :
  ∃ r : ℝ, (∃ D : ℝ, D = 2 * larger_circle_radius ∧ D = 4 * r) ∧ r = 2.5 :=
by
  sorry

end smaller_circle_radius_l360_360132


namespace conditional_probability_event_l360_360790

def tetrahedral_die := {3, 5, 7, 9}

def event_A (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s, x ≠ y ∧ y ≠ z ∧ x ≠ z → (x + y > z ∧ x + z > y ∧ y + z > x)

def event_B (s : Finset ℕ) : Prop :=
  3 ∈ s

def possible_sets : Finset (Finset ℕ) :=
  {{3, 5, 7}, {3, 5, 9}, {3, 7, 9}}

noncomputable def P_A_given_B (B : Finset (Finset ℕ)) (A : Finset (Finset ℕ)) : ℚ :=
  (B.filter event_A).card / B.card

theorem conditional_probability_event (hB : possible_sets.filter event_B = possible_sets) :
  P_A_given_B possible_sets event_A = 2 / 3 :=
by
  sorry

end conditional_probability_event_l360_360790


namespace find_a_l360_360963

variable {a : ℝ}

theorem find_a (h : ∫ x in 0..a, x = 1) : a = Real.sqrt 2 :=
sorry

end find_a_l360_360963


namespace angle_between_clock_hands_at_7_oclock_l360_360708

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l360_360708


namespace parabola_transformation_l360_360265

theorem parabola_transformation :
  ∀ (x y : ℝ), y = 3 * x^2 → (∃ z : ℝ, z = x - 1 ∧ y = 3 * z^2 - 2) :=
by
  sorry

end parabola_transformation_l360_360265


namespace root_bounds_l360_360613

noncomputable def sqrt (r : ℝ) (n : ℕ) := r^(1 / n)

theorem root_bounds (a b c d : ℝ) (n p x y : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hn : 0 < n) (hp : 0 < p) (hx : 0 < x) (hy : 0 < y) :
  sqrt d y < sqrt (a * b * c * d) (n + p + x + y) ∧
  sqrt (a * b * c * d) (n + p + x + y) < sqrt a n := 
sorry

end root_bounds_l360_360613


namespace cos_sin_circle_l360_360469

theorem cos_sin_circle (t : ℝ) : 
  let x := Real.cos(2 * t)
  let y := Real.sin(2 * t)
  in x^2 + y^2 = 1 :=
by
  let x := Real.cos(2 * t)
  let y := Real.sin(2 * t)
  have h := Real.cos_square_add_sin_square (2 * t)
  rw [←h]
  exact h
  sorry

end cos_sin_circle_l360_360469


namespace standard_equation_of_ellipse_max_OM_value_l360_360558

open Set
open Real

-- Conditions
variables {a b c : ℝ} (h1 : a > b > 0) (h2 : 2 * a = 2 * sqrt 2) (h3 : c^2 = a^2 - b^2) (ellipse : (x y : ℝ) → x^2 / a^2 + y^2 / b^2 = 1)
variables {F1 F2 : ℝ × ℝ} (h4 : dist F1 F2 = 2) (point_on_ellipse : (P : ℝ × ℝ) → P ∈ {P | ellipse P.1 P.2})
variables (line_l : (x y : ℝ) → y = k*x + m) (intersection_AB : (A B : ℝ × ℝ) → A ∈ intersection (ellipse A.1 A.2) (line_l A.1 A.2) ∧ B ∈ intersection (ellipse B.1 B.2) (line_l B.1 B.2) ∧ dist A B = 2)

-- Required proofs
theorem standard_equation_of_ellipse : ∃ a b : ℝ, a = sqrt 2 ∧ b^2 = 1 ∧ (∀ x y, ellipse x y ↔ x^2 / 2 + y^2 = 1) :=
sorry

theorem max_OM_value : ∃ (k m : ℝ) (OM : ℝ), (∀ A B : ℝ × ℝ, intersection_AB A B → ∃ M : ℝ × ℝ, M = ( ⟨-2*k*m / (2*k^2 + 1), m / (2*k^2 + 1)⟩) ∧ dist 0 M = sqrt (3 - 2*sqrt 3)) :=
sorry

end standard_equation_of_ellipse_max_OM_value_l360_360558


namespace average_rainfall_per_hour_eq_l360_360985

-- Define the conditions
def february_days_non_leap_year : ℕ := 28
def hours_per_day : ℕ := 24
def total_rainfall_in_inches : ℕ := 280
def total_hours_in_february : ℕ := february_days_non_leap_year * hours_per_day

-- Define the goal
theorem average_rainfall_per_hour_eq :
  total_rainfall_in_inches / total_hours_in_february = 5 / 12 :=
sorry

end average_rainfall_per_hour_eq_l360_360985


namespace sheepdog_catches_sheep_in_20_seconds_l360_360212

theorem sheepdog_catches_sheep_in_20_seconds :
  ∀ (sheep_speed dog_speed : ℝ) (initial_distance : ℝ),
    sheep_speed = 12 →
    dog_speed = 20 →
    initial_distance = 160 →
    (initial_distance / (dog_speed - sheep_speed)) = 20 := by
  intros sheep_speed dog_speed initial_distance h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end sheepdog_catches_sheep_in_20_seconds_l360_360212


namespace midpoint_of_line_segment_l360_360130

theorem midpoint_of_line_segment :
  let z1 := Complex.mk (-7) 5
  let z2 := Complex.mk 5 (-3)
  (z1 + z2) / 2 = Complex.mk (-1) 1 := by sorry

end midpoint_of_line_segment_l360_360130


namespace blue_eggs_candy_fraction_l360_360323

noncomputable def fraction_of_blue_eggs_with_candy (E : ℕ) (hE : E > 0)
  (blue_eggs : ℕ := (4 * E) / 5)
  (purple_eggs : ℕ := (E) / 5)
  (purple_eggs_with_candy : ℕ := (purple_eggs) / 2)
  (total_eggs_with_candy : ℕ := (3 * E) / 10)
  (fraction_of_blue_eggs_with_candy : ℚ) : Prop :=
  fraction_of_blue_eggs_with_candy = 1 / 4 

theorem blue_eggs_candy_fraction (E : ℕ) (hE : E > 0)
  (blue_eggs : ℕ := (4 * E) / 5)
  (purple_eggs : ℕ := (E) / 5)
  (purple_eggs_with_candy : ℕ := (purple_eggs) / 2)
  (total_eggs_with_candy : ℕ := (3 * E) / 10)
  (fraction_of_blue_eggs_with_candy : ℚ := 1 / 4) : 
  fraction_of_blue_eggs_with_candy (E) (hE) := 
sorry

end blue_eggs_candy_fraction_l360_360323


namespace cost_of_3600_pens_l360_360339

-- Define the conditions
def cost_per_200_pens : ℕ := 50
def pens_bought : ℕ := 3600

-- Define a theorem to encapsulate our question and provide the necessary definitions
theorem cost_of_3600_pens : cost_per_200_pens / 200 * pens_bought = 900 := by sorry

end cost_of_3600_pens_l360_360339


namespace rectangle_exists_l360_360552

theorem rectangle_exists (k : ℕ) (h : k = 457) : ∀ (grid : fin 57 × fin 57 → bool),
  (∃ p1 p2 p3 p4 : fin 57 × fin 57,
    grid p1 = tt ∧ grid p2 = tt ∧ grid p3 = tt ∧ grid p4 = tt ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧
    p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1.1 = p2.1 ∧ p3.1 = p4.1 ∧
    p1.2 = p3.2 ∧ p2.2 = p4.2) :=
by
  sorry

end rectangle_exists_l360_360552


namespace solution_correctness_l360_360048

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

def is_monotone (f : ℝ → ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ≤ b → f(a) ≤ f(b)

noncomputable def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f(x + 1) ≤ f(-1)}

theorem solution_correctness
  (f : ℝ → ℝ)
  (even_f : is_even f)
  (domain_f : ∀ x, -2 ≤ x ∧ x ≤ 2 → x ∈ (Icc (-2 : ℝ) (2 : ℝ)))
  (monotone_f : is_monotone (λ x, f (-x)) ∧ ∀ x ∈ (Icc (-2 : ℝ) (0 : ℝ)), is_monotone f) :
  solution_set f = {x | -3 ≤ x ∧ x ≤ -2 ∨ 0 ≤ x ∧ x ≤ 1} :=
by
  skip

end solution_correctness_l360_360048


namespace ratio_of_areas_of_equilateral_triangles_in_semi_and_full_circle_l360_360720

theorem ratio_of_areas_of_equilateral_triangles_in_semi_and_full_circle (r : ℝ) :
  let A1 := (sqrt 3 / 4) * ((4 * r / sqrt 3) ^ 2)
  let A2 := (sqrt 3 / 4) * ((r * sqrt 3) ^ 2)
  (A1 / A2) = (64 * sqrt 3 / 27) :=
by
  sorry

end ratio_of_areas_of_equilateral_triangles_in_semi_and_full_circle_l360_360720


namespace integer_parts_of_S_n_l360_360270

noncomputable def a : ℕ → ℝ
| 1     := 4 / 3
| (n+1) := a n * (a n - 1) + 1

noncomputable def S : ℕ → ℝ
| 1     := 1 / (4 / 3)
| (n+1) := S n + 1 / a (n + 1)

theorem integer_parts_of_S_n : ∃ S_set : Set ℤ, S_set = {0, 1, 2} ∧ 
  ∀ n : ℕ, n > 0 → ⌊S n⌋ ∈ S_set := 
by
  sorry

end integer_parts_of_S_n_l360_360270


namespace small_number_of_uphill_paths_l360_360143

-- Definitions based on the given conditions
def is_valley (n : ℕ) (board : matrix (fin n) (fin n) ℕ) (i j : fin n) : Prop :=
  ∀ di dj, abs di + abs dj = 1 → ∃ (x y : fin n), board (i + di) (j + dj) > board i j

def is_uphill_path (n : ℕ) (board : matrix (fin n) (fin n) ℕ) (path : list (fin n × fin n)) : Prop :=
  ∃ (path0 : (fin n × fin n)), (∀ (i j : fin n), (i, j) ∈ path → is_valley n board i j) ∧
  ∀ (k : ℕ) (hk : k < path.length - 1),
    let (i1, j1) := path.nth_le k hk in
    let (i2, j2) := path.nth_le (k + 1) (nat.lt_of_succ_lt hkl) in
    abs (i1 - i2) + abs (j1 - j2) = 1 ∧
    board i1 j1 < board i2 j2

-- Statement of the proof problem in Lean 4
theorem small_number_of_uphill_paths (n : ℕ) (h1 : 0 < n) (board : matrix (fin n) (fin n) ℕ) :
  ∃ paths : list (list (fin n × fin n)),
    (∀ path, path ∈ paths → is_uphill_path n board path) ∧
    paths.length = 2 * n * (n - 1) + 1 :=
sorry

end small_number_of_uphill_paths_l360_360143


namespace intersection_of_sets_l360_360204

theorem intersection_of_sets : 
  let A := {1, 2, 3}
  let B := {-2, 2}
  A ∩ B = {2} :=
by
  sorry

end intersection_of_sets_l360_360204


namespace smaller_angle_at_seven_oclock_l360_360699

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l360_360699


namespace triangle_ABC_cos2A_eq_sins_l360_360546

-- Definition of the problem in Lean 4 with corresponding conditions and statements to be proven
theorem triangle_ABC_cos2A_eq_sins (a b c A B C : ℝ) 
  (h_cos2A_sin: cos (2 * A) + 2 * (sin (π + B))^2 + 2 * (cos (π / 2 + C))^2 - 1 = 2 * sin B * sin C)
  (hb : b = 4) 
  (hc : c = 5) 
  (hb_positive : 0 < b)
  (hc_positive : 0 < c) 
  (A_pos : 0 < A) 
  (A_lt_pi : A < π)
  (cos_A : cos A = 1 / 2) :
  A = π / 3 ∧ sin B = 2 * sqrt 7 / 7 :=
by
  sorry

end triangle_ABC_cos2A_eq_sins_l360_360546


namespace spend_amount_7_l360_360773

variable (x y z w : ℕ) (k : ℕ)

theorem spend_amount_7 
  (h1 : 10 * x + 15 * y + 25 * z + 40 * w = 100 * k)
  (h2 : x + y + z + w = 30)
  (h3 : (x = 5 ∨ x = 10) ∧ (y = 5 ∨ y = 10) ∧ (z = 5 ∨ z = 10) ∧ (w = 5 ∨ w = 10)) : 
  k = 7 := 
sorry

end spend_amount_7_l360_360773


namespace art_club_artworks_l360_360674

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end art_club_artworks_l360_360674


namespace find_largest_prime_factor_l360_360002

def problem_statement : Prop :=
  ∀ (a b c : ℕ), a = 12^3 ∧ b = 15^4 ∧ c = 6^5 →
  (∃ p : ℕ, prime p ∧ p ∣ (a + b - c) ∧ ∀ q : ℕ, prime q ∧ q ∣ (a + b - c) → q ≤ p)

theorem find_largest_prime_factor :
  ∀ (a b c : ℕ), a = 12^3 → b = 15^4 → c = 6^5 →
  ∃ p : ℕ, prime p ∧ p ∣ (a + b - c) ∧ 
  (∀ q : ℕ, prime q ∧ q ∣ (a + b - c) → q ≤ p) :=
begin
  sorry
end

end find_largest_prime_factor_l360_360002


namespace solve_inequalities_l360_360618

theorem solve_inequalities (x : ℝ) : 
  (x^2 > x + 2) ∧ (4 * x^2 ≤ 4 * x + 15) ↔ 
  (x ∈ set.Ico (-3 / 2 : ℝ) (-1) ∪ set.Ioc (2 : ℝ) (5 / 2)) :=
by sorry

end solve_inequalities_l360_360618


namespace Euleria_odd_paving_l360_360275

noncomputable def pave_roads_odd_degree : Prop := 
  ∃ (G : Graph V) (H : Subgraph G), 
    G.num_vertices = 1000 ∧
    G.is_connected ∧ 
    (∀ v : V, ((G.degree v).odd))

theorem Euleria_odd_paving :
  ∃ (G : Graph), 
    (G.num_vertices = 1000 ∧ G.is_connected ∧ ∃ H : Subgraph G, (∀ v : V, H.degree v % 2 = 1)) := sorry

end Euleria_odd_paving_l360_360275


namespace selection_methods_count_l360_360883

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem selection_methods_count :
  let females := 8
  let males := 4
  (binomial females 2 * binomial males 1) + (binomial females 1 * binomial males 2) = 112 :=
by
  sorry

end selection_methods_count_l360_360883


namespace sheepdog_catches_sheep_in_20_seconds_l360_360213

theorem sheepdog_catches_sheep_in_20_seconds :
  ∀ (sheep_speed dog_speed : ℝ) (initial_distance : ℝ),
    sheep_speed = 12 →
    dog_speed = 20 →
    initial_distance = 160 →
    (initial_distance / (dog_speed - sheep_speed)) = 20 := by
  intros sheep_speed dog_speed initial_distance h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end sheepdog_catches_sheep_in_20_seconds_l360_360213


namespace count_valid_numbers_l360_360818

open BigOperators

def is_valid_number (n : ℕ) : Prop :=
  n.to_string.length = 4 ∧
  (∀ i, n.to_string.nth i ∈ ['1', '2', '3']) ∧
  '1' ∈ n.to_string ∧ '2' ∈ n.to_string ∧ '3' ∈ n.to_string ∧
  ∀ i, i < 3 → n.to_string.nth i ≠ n.to_string.nth (i + 1)

theorem count_valid_numbers : 
  {n | is_valid_number n}.card = 66 := 
sorry

end count_valid_numbers_l360_360818


namespace dot_product_magnitude_l360_360152

variable {a b : EuclideanSpace ℝ (Fin n)}
variable (norm_a : ∥a∥ = 3)
variable (norm_b : ∥b∥ = 4)
variable (norm_cross_ab : ∥a × b∥ = 6)

theorem dot_product_magnitude : ∥a∥ = 3 → ∥b∥ = 4 → ∥a × b∥ = 6 → |a ∘ b| = 6 * real.sqrt 3 := 
by 
  intro norm_a norm_b norm_cross_ab
  sorry

end dot_product_magnitude_l360_360152


namespace max_colored_cells_in_4x50_rect_l360_360367

theorem max_colored_cells_in_4x50_rect : 
  let rect := (λ (i j : ℕ), (0 <= i < 4) ∧ (0 <= j < 50))
  in 
  let step := (λ (cells : finset (ℕ × ℕ)) (cell : ℕ × ℕ),
    cell ∉ cells ∧ (∀ (i j : ℕ), (cell = (i, j)) → (finset.card (cells.filter (λ (p, q), (abs (p - i) + abs (q - j) = 1))) <= 1))
  in 
  ∃ (colored : finset (ℕ × ℕ)),
    ∀ (start : ℕ × ℕ),
      start ∈ rect →
      colored = (finset.univ : finset (ℕ × ℕ)).filter (step finset.empty start) →
      finset.card colored ≤ 75 := 
sorry

end max_colored_cells_in_4x50_rect_l360_360367


namespace problem_statement_l360_360027

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^4 + 2*m^3 - m + 2007 = 2007 := 
by 
  sorry

end problem_statement_l360_360027


namespace difference_between_longest_and_shortest_fish_l360_360347

def fish_lengths := [0.45, 0.67, 0.95, 0.53, 0.72]

theorem difference_between_longest_and_shortest_fish :
  list.maximum fish_lengths - list.minimum fish_lengths = 0.50 := by
  sorry

end difference_between_longest_and_shortest_fish_l360_360347


namespace number_of_subsets_of_A_l360_360518

theorem number_of_subsets_of_A : 
    let A := {x : ℝ | (x > 0) ∧ (x^2 - 4 = 0)} in
    ∃ n : ℕ, n = fintype.card (set A) ∧ n = 2 :=
by
  -- let A be the set as defined in the problem
  let A := {x : ℝ | (x > 0) ∧ (x^2 - 4 = 0)},
  -- provide the answer directly
  use fintype.card (set A),
  -- prove the answer
  split,
  -- fintype.card A = 2
  sorry,
  -- n = 2
  sorry

end number_of_subsets_of_A_l360_360518


namespace correct_propositions_l360_360746

theorem correct_propositions :
  (∀ (P₁ P₂ P₃ : Point),
    ¬Collinear P₁ P₂ P₃ →
    ∃ (C : Circle), OnCircle P₁ C ∧ OnCircle P₂ C ∧ OnCircle P₃ C) ∧
  (∀ (C : Circle) (d : Line) (ch : Chord),
    DiamPerpChord C d ch → Bisects d ch) ∧
  (∀ (C : Circle) (a₁ a₂ : Arc),
    EqualArcs C a₁ a₂ → EqualCentralAngles a₁ a₂) ∧
  (¬∀ (T : Triangle) (O : Point),
    Circumcenter T O →
    ∀ (s : Side), DistanceToSide O T s = DistanceToSide O T t) ∧
  (¬∀ (C : Circle) (minor major : Arc),
    MajorArc C major → MinorArc C minor → major.length = major.angle.toRadians ∧ minor.length = minor.angle.toRadians → major.length > minor.length) :=
sorry

end correct_propositions_l360_360746


namespace inequalities_hold_l360_360966

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b ≥ 2) :=
by
  sorry

end inequalities_hold_l360_360966


namespace max_row_or_column_sum_is_42_l360_360900

noncomputable def isPrime (n : Nat) : Prop := Nat.Prime n

def in_grid (nums : List Nat) (g : List (List Nat)) : Prop :=
  g.map (λ r => r.sum).all (λ s => isPrime s) ∧
  List.transpose g |>.map (λ c => c.sum).all (λ s => isPrime s)

def problem : Prop :=
  ∃ (g : List (List Nat)), g.flatten = [2, 5, 8, 11, 14, 17] ∧ in_grid [2, 5, 8, 11, 14, 17] g ∧
  List.transpose g |>.map (λ c => c.sum).maximum = some 42

theorem max_row_or_column_sum_is_42 : problem :=
sorry

end max_row_or_column_sum_is_42_l360_360900


namespace ones_digit_large_power_dividing_32_factorial_l360_360859

theorem ones_digit_large_power_dividing_32_factorial :
  let n := 32!
  let largestPower := 2^31
  ones_digit largestPower = 8 :=
by
  sorry

end ones_digit_large_power_dividing_32_factorial_l360_360859


namespace admission_methods_correct_l360_360125

-- Define the number of famous schools.
def famous_schools : ℕ := 8

-- Define the number of students.
def students : ℕ := 3

-- Define the total number of different admission methods:
def admission_methods (schools : ℕ) (students : ℕ) : ℕ :=
  Nat.choose schools 2 * 3

-- The theorem stating the desired result.
theorem admission_methods_correct :
  admission_methods famous_schools students = 84 :=
by
  sorry

end admission_methods_correct_l360_360125


namespace range_of_m_l360_360931

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x
noncomputable def g (x m : ℝ) : ℝ := (1 / 2)^x + m

theorem range_of_m (m : ℝ) : 
  m ≤ 5 / 2 → 
  ∀ x1 : ℝ, x1 ∈ Icc 1 2 → 
  ∃ x2 : ℝ, x2 ∈ Icc (-1) 1 ∧ f x1 ≥ g x2 m := 
by
  intros
  sorry

end range_of_m_l360_360931


namespace eccentricity_of_hyperbola_is_5_over_3_l360_360901

noncomputable def eccentricity_of_hyperbola
  (a b : ℝ)
  (h₁ : F₁ = -c)
  (h₂ : F₂ = c)
  (h₃ : c = sqrt (a^2 + b^2))
  (h₄ : ∀ P : ℝ × ℝ, P ∈ hyperbola a b → |P.1 - F₂.1| = |F₁ - F₂|)
  (h₅ : distance_from_F₂_to_line (line_through_P_and_other_focus P F₁) = 2 * a)
  : ℝ :=
c / a

theorem eccentricity_of_hyperbola_is_5_over_3
  (a b : ℝ)
  (h₁ : F₁ = -c)
  (h₂ : F₂ = c)
  (h₃ : c = sqrt (a^2 + b^2))
  (h₄ : ∀ P : ℝ × ℝ, P ∈ hyperbola a b → |P.1 - F₂.1| = |F₁ - F₂|)
  (h₅ : distance_from_F₂_to_line (line_through_P_and_other_focus P F₁) = 2 * a)
  : eccentricity_of_hyperbola a b h₁ h₂ h₃ h₄ h₅ = 5 / 3 :=
sorry

end eccentricity_of_hyperbola_is_5_over_3_l360_360901


namespace ones_digit_large_power_dividing_32_factorial_l360_360856

theorem ones_digit_large_power_dividing_32_factorial :
  let n := 32!
  let largestPower := 2^31
  ones_digit largestPower = 8 :=
by
  sorry

end ones_digit_large_power_dividing_32_factorial_l360_360856


namespace distinct_tower_heights_l360_360239

theorem distinct_tower_heights (n : ℕ) (h : n = 68)
  (h_bricks : ∀ k, k ∈ {3, 8, 12}) : 
  ∃ d_heights, d_heights = 581 :=
by
  sorry

end distinct_tower_heights_l360_360239


namespace vertex_sum_of_cube_l360_360226

noncomputable def cube_vertex_sum (a : Fin 8 → ℕ) : ℕ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7

def face_sums (a : Fin 8 → ℕ) : List ℕ :=
  [
    a 0 + a 1 + a 2 + a 3, -- first face
    a 0 + a 1 + a 4 + a 5, -- second face
    a 0 + a 3 + a 4 + a 7, -- third face
    a 1 + a 2 + a 5 + a 6, -- fourth face
    a 2 + a 3 + a 6 + a 7, -- fifth face
    a 4 + a 5 + a 6 + a 7  -- sixth face
  ]

def total_face_sum (a : Fin 8 → ℕ) : ℕ :=
  List.sum (face_sums a)

theorem vertex_sum_of_cube (a : Fin 8 → ℕ) (h : total_face_sum a = 2019) :
  cube_vertex_sum a = 673 :=
sorry

end vertex_sum_of_cube_l360_360226


namespace find_m_l360_360044

variable {α : Type*} [DecidableEq α]

-- Definitions and conditions
def A (m : ℤ) : Set ℤ := {-1, 3, m ^ 2}
def B : Set ℤ := {3, 4}

theorem find_m (m : ℤ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end find_m_l360_360044


namespace security_scheduling_methods_l360_360828

-- Definitions and conditions for the problem
def persons : Type := {A, B, C}
def days : Type := Fin 5 -- The 5-day scheduling period

-- The main theorem statement:
theorem security_scheduling_methods :
  ∃ (schedule : days → persons), (∀ p : persons, ∃ d : days, schedule d = p) ∧
                                (∀ d : days, ∃ p : persons, schedule d = p) ∧
                                (∀ d : Fin 4, schedule d ≠ schedule (d + 1)) ∧
                                (card {f : days → persons |
                                  (∀ p : persons, ∃ d : days, f d = p) ∧
                                  (∀ d : days, ∃ p : persons, f d = p) ∧
                                  (∀ d : Fin 4, f d ≠ f (d + 1))} = 42) :=
sorry

end security_scheduling_methods_l360_360828


namespace parallelogram_ratio_l360_360639

-- Definitions based on given conditions
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_ratio (A : ℝ) (B : ℝ) (h : ℝ) (H1 : A = 242) (H2 : B = 11) (H3 : A = parallelogram_area B h) :
  h / B = 2 :=
by
  -- Proof goes here
  sorry

end parallelogram_ratio_l360_360639


namespace rate_percent_simple_interest_l360_360297

theorem rate_percent_simple_interest
  (SI P : ℚ) (T : ℕ) (R : ℚ) : SI = 160 → P = 800 → T = 4 → (P * R * T / 100 = SI) → R = 5 :=
  by
  intros hSI hP hT hFormula
  -- Assertion that R = 5 is correct based on the given conditions and formula
  sorry

end rate_percent_simple_interest_l360_360297


namespace find_principal_amount_l360_360740

theorem find_principal_amount (P r : ℝ) (A2 A3 : ℝ) (n2 n3 : ℕ) 
  (h1 : n2 = 2) (h2 : n3 = 3) 
  (h3 : A2 = 8820) 
  (h4 : A3 = 9261) 
  (h5 : r = 0.05) 
  (h6 : A2 = P * (1 + r)^n2) 
  (h7 : A3 = P * (1 + r)^n3) : 
  P = 8000 := 
by 
  sorry

end find_principal_amount_l360_360740


namespace perpendicular_condition_l360_360080

def vector_a : ℝ × ℝ := (4, 3)
def vector_b : ℝ × ℝ := (-1, 2)

def add_vector_scaled (a b : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  (a.1 + k * b.1, a.2 + k * b.2)

def sub_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perpendicular_condition (k : ℝ) :
  dot_product (add_vector_scaled vector_a vector_b k) (sub_vector vector_a vector_b) = 0 ↔ k = 23 / 3 :=
by
  sorry

end perpendicular_condition_l360_360080


namespace inequality_transformations_l360_360967

theorem inequality_transformations (a b : ℝ) (h : a > b) :
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) :=
by
  sorry

end inequality_transformations_l360_360967


namespace general_term_formula_l360_360503

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Given condition: the sum of the first n terms of sequence a is S_n = n^2 + n
def S_n_def (n : ℕ) : Prop :=
  S n = n^2 + n

-- Definition of the general term a_n based on the sum S_n
noncomputable def a_n_def (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

-- We want to prove that a_n = 2n
theorem general_term_formula (n : ℕ) (h : S_n_def S) : a_n_def S n = 2n :=
  sorry

end general_term_formula_l360_360503


namespace artwork_collection_l360_360668

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end artwork_collection_l360_360668


namespace complement_intersection_l360_360962

def A : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

theorem complement_intersection :
  (Set.univ \ A) ∩ B = {x | x > 7} :=
by
  sorry

end complement_intersection_l360_360962


namespace ones_digit_large_power_dividing_32_factorial_l360_360858

theorem ones_digit_large_power_dividing_32_factorial :
  let n := 32!
  let largestPower := 2^31
  ones_digit largestPower = 8 :=
by
  sorry

end ones_digit_large_power_dividing_32_factorial_l360_360858


namespace sphere_diameter_from_cylinder_l360_360770

theorem sphere_diameter_from_cylinder (cyl_diameter cyl_height : ℝ) (num_spheres : ℕ) (cyl_diameter_eq : cyl_diameter = 8) (cyl_height_eq : cyl_height = 48) (num_spheres_eq : num_spheres = 9) : 
  let cyl_radius := cyl_diameter / 2,
      cylinder_volume := π * cyl_radius^2 * cyl_height,
      sphere_volume := cylinder_volume / num_spheres,
      sphere_radius := ((3 * sphere_volume) / (4 * π))^(1 / 3),
      sphere_diameter := 2 * sphere_radius
  in sphere_diameter = 8 :=
by 
  sorry

end sphere_diameter_from_cylinder_l360_360770


namespace largest_prime_factor_of_expression_l360_360005

noncomputable def expr := 12^3 + 15^4 - 6^5

theorem largest_prime_factor_of_expression :
  ∃ p, prime p ∧ p ∣ expr ∧ (∀ q, prime q ∧ q ∣ expr → q ≤ p) ∧ p = 12193 := 
sorry

end largest_prime_factor_of_expression_l360_360005


namespace workshop_c_defective_rate_l360_360548

theorem workshop_c_defective_rate :
  let P_B1 := 0.45 in
  let P_B2 := 0.35 in
  let P_B3 := 0.20 in
  let P_A_given_B1 := 0.02 in
  let P_A_given_B2 := 0.03 in
  let P_A := 0.0295 in
  ∃ m : ℝ, (m = 0.05) ∧ (P_A = P_A_given_B1 * P_B1 + P_A_given_B2 * P_B2 + m * P_B3) :=
by
  sorry

end workshop_c_defective_rate_l360_360548


namespace dot_product_magnitude_l360_360159

open Real

variables (a b : ℝ^3) 

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross (u v : ℝ^3) : ℝ^3 := (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)
def dot (u v : ℝ^3) : ℝ := u.1*v.1 + u.2*v.2 + u.3*v.3

axiom norm_a : norm a = 3
axiom norm_b : norm b = 4
axiom cross_norm : norm (cross a b) = 6

theorem dot_product_magnitude : abs (dot a b) = 6 * sqrt 3 := by
  sorry

end dot_product_magnitude_l360_360159


namespace dot_product_magnitude_l360_360188

variables {a b : ℝ^3}

-- Norm of vectors
def norm_a := ‖a‖ = 3
def norm_b := ‖b‖ = 4
def cross_product_norm := ‖a × b‖ = 6

theorem dot_product_magnitude (h1 : norm_a) (h2 : norm_b) (h3 : cross_product_norm) : (abs (a • b)) = 6 * real.sqrt 3 :=
sorry

end dot_product_magnitude_l360_360188


namespace ones_digit_of_largest_power_of_2_in_20_fact_l360_360869

open Nat

def largest_power_of_2_in_factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else
    let sum_of_powers := ∑ m in range (n+1), m / 2
    sum_of_powers

def ones_digit_of_power_of_2 (exp : ℕ) : ℕ :=
  let cycle := [2, 4, 8, 6]
  cycle[exp % 4]

theorem ones_digit_of_largest_power_of_2_in_20_fact (n : ℕ) (h : n = 20) : 
  ones_digit_of_power_of_2 (largest_power_of_2_in_factorial n) = 4 :=
by
  rw [h]
  have : largest_power_of_2_in_factorial 20 = 18 := by
    -- Insert the calculations for largest_power_of_2_in_factorial here
    sorry
  rw [this]
  have : ones_digit_of_power_of_2 18 = 4 := by
    -- Insert the cycle calculations here
    sorry
  exact this

end ones_digit_of_largest_power_of_2_in_20_fact_l360_360869


namespace an_gt_1_l360_360889

variables {a : ℝ} (a1 : ℕ+ → ℝ)
hypothesis ha : 0 < a ∧ a < 1
hypothesis hbase : a1 1 = 1 + a
hypothesis hrec : ∀ n : ℕ+, a1 (n + 1) = 1 / (a1 n) + a

theorem an_gt_1 (n : ℕ+) : a1 n > 1 :=
  sorry

end an_gt_1_l360_360889


namespace line_passes_fixed_point_maximal_distance_line_l360_360072

-- Define the line equation
def line_l (a b x y : ℝ) : Prop :=
  (2 * a + b) * x + (a + b) * y + a - b = 0

-- Define the fixed point A
def fixed_point (x y : ℝ) : Prop :=
  x = -2 ∧ y = 3

-- Prove that the line l passes through the fixed point A(-2, 3)
theorem line_passes_fixed_point (a b : ℝ) : line_l a b (-2) 3 :=
begin
  sorry
end

-- Define the distance maximized line equation
def distance_maximized_line (x y : ℝ) : Prop :=
  5 * x + y + 7 = 0

-- Prove that the equation of the line l when the distance from point P(3,4) to line l is maximized
theorem maximal_distance_line (a b : ℝ) (P_x P_y : ℝ) (hP : P_x = 3 ∧ P_y = 4) :
  distance_maximized_line (-2) 3 :=
begin
  sorry
end

end line_passes_fixed_point_maximal_distance_line_l360_360072


namespace angle_between_clock_hands_at_7_oclock_l360_360710

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l360_360710


namespace sum_of_repeating_decimals_l360_360412

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l360_360412


namespace max_sum_of_factors_of_48_l360_360151

theorem max_sum_of_factors_of_48 : ∃ (heartsuit clubsuit : ℕ), heartsuit * clubsuit = 48 ∧ heartsuit + clubsuit = 49 :=
by
  -- We insert sorry here to skip the actual proof construction.
  sorry

end max_sum_of_factors_of_48_l360_360151


namespace radius_of_circle_l360_360636

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l360_360636


namespace area_of_rectangle_l360_360306

-- Define the conditions
def width : ℕ := 6
def perimeter : ℕ := 28

-- Define the theorem statement
theorem area_of_rectangle (w : ℕ) (p : ℕ) (h_width : w = width) (h_perimeter : p = perimeter) :
  ∃ l : ℕ, (2 * (l + w) = p) → (l * w = 48) :=
by
  use 8
  intro h
  simp only [h_width, h_perimeter] at h
  sorry

end area_of_rectangle_l360_360306


namespace no_combination_sums_to_20_l360_360447

def is_sum_20 (f : Fin 9 → Int) : Prop :=
  ∑ i in Finset.range 9, f i * (i + 1) = 20

theorem no_combination_sums_to_20 :
  ∀ f : Fin 9 → Int, is_sum_20 f → False :=
by
  sorry

end no_combination_sums_to_20_l360_360447


namespace proof_triangle_l360_360564

-- Define the variables and constants
variables {a b c : ℝ}
variables {A B C : ℝ}
variables {bc_perimeter : ℝ}

-- Define the conditions of the triangle and given vectors
def conditions (triangle_ABC : Prop) :=
  (∃ a b c A B C : ℝ, 
    sides_opposite a b c A B C ∧
    let m := (a, 2 * b - c) in
    let n := (cos A, cos C) in
    parallel m n)

noncomputable def proof_problem (a b c A perimeter : ℝ) :=
  let area := 2 * sqrt 3 in
  let b_plus_c := 7 in
  let expected_A := π / 3 in
  let expected_perimeter := 12 in
  conditions (a b c A B C) →
  (A = expected_A ∧ perimeter = expected_perimeter)

-- The proof statement (without proof)
theorem proof_triangle : Proof_problem :=
by sorry

end proof_triangle_l360_360564


namespace sum_of_possible_values_of_x_l360_360744

def list := [4, 9, x, 4, 9, 4, 11, x]

def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

def median (l : List ℝ) : ℝ :=
  let sorted_l := l.qsort (· ≤ ·);
  if sorted_l.length % 2 = 0 then
    (sorted_l.get! (sorted_l.length / 2 - 1) + sorted_l.get! (sorted_l.length / 2)) / 2
  else
    sorted_l.get! (sorted_l.length / 2)

def mode (l : List ℝ) : ℝ :=
  let freq_map := l.foldr (λ x m, m.insert x (m.findD x 0 + 1)) (Std.HashMap.ofList []);
  freq_map.toList.argMax (λ x, x.2).1

def geometric_progression (a b c : ℝ) : Prop :=
  a < b ∧ b < c ∧ b / a = c / b

theorem sum_of_possible_values_of_x :
  ∃ x : ℝ, (mean list = (41 + 2 * x) / 8) ∧
          (mode list = 4) ∧
          (geometric_progression 4 (median list) ((41 + 2 * x) / 8)) ∧
          x = 15.5 :=
sorry

end sum_of_possible_values_of_x_l360_360744


namespace min_price_floppy_cd_l360_360273

theorem min_price_floppy_cd (x y : ℝ) (h1 : 4 * x + 5 * y ≥ 20) (h2 : 6 * x + 3 * y ≤ 24) : 3 * x + 9 * y ≥ 22 :=
by
  -- The proof is not provided as per the instructions.
  sorry

end min_price_floppy_cd_l360_360273


namespace percentage_reduction_price_increase_l360_360332

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end percentage_reduction_price_increase_l360_360332


namespace degree_of_f_l360_360573

theorem degree_of_f (p f g : Polynomial ℝ) :
  p.degree < 100 ∧ ¬ (x^3 - x) ∣ p →
  (d 100 (λ x, p / (x^3 - x)) = f / g) →
  Polynomial.degree f = 200 := by
  sorry

end degree_of_f_l360_360573


namespace cannot_fold_5x5_to_1x8_cannot_fold_5x5_to_1x7_l360_360646

-- Definition of distances
def diagonal_5x5 := real.sqrt (5^2 + 5^2)
def diagonal_1x8 := real.sqrt (1^2 + 8^2)
def diagonal_1x7 := real.sqrt (1^2 + 7^2)

-- Problem (a)
theorem cannot_fold_5x5_to_1x8 : diagonal_5x5 < diagonal_1x8 := by
  sorry

-- Problem (b)
theorem cannot_fold_5x5_to_1x7 :
  diagonal_5x5 = diagonal_1x7 → false := by
  sorry

end cannot_fold_5x5_to_1x8_cannot_fold_5x5_to_1x7_l360_360646


namespace dot_product_magnitude_l360_360184

variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜 → 𝕜} 

-- Define the conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_a_cross_b : ℝ := 6

-- Theorem statement
theorem dot_product_magnitude :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∥a × b∥ = norm_a_cross_b) → abs ((a.toReal).dot (b.toReal)) = 6 * sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end dot_product_magnitude_l360_360184


namespace quadratic_roots_difference_l360_360982

theorem quadratic_roots_difference (a b : ℝ) 
  (h1 : ∃ (x : ℝ), x^2 - (a + 1) * x + a = 0)
  (h2 : is_root (λ x : ℝ, x^2 - (a + 1) * x + a) 2) : 
    a - b = 1 :=
by
  sorry

end quadratic_roots_difference_l360_360982


namespace train_cross_platform_time_l360_360771

theorem train_cross_platform_time 
  (length_train : ℝ)
  (time_pole : ℝ)
  (length_platform : ℝ) :
  length_train = 400 → 
  time_pole = 30 → 
  length_platform = 200 →
  let speed := length_train / time_pole in
  let total_distance := length_train + length_platform in
  let time := total_distance / speed in
  time = 45 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have speed_def : speed = 400 / 30 := rfl
  rw [speed_def]
  have total_distance_def : total_distance = 400 + 200 := rfl
  rw [total_distance_def]
  have : 13.33 ≈ 400 / 30 := by norm_num
  rw this
  have : 600 / 13.33 ≈ 45 := by norm_num
  exact this

end train_cross_platform_time_l360_360771


namespace pentagon_area_ln_coordinates_l360_360667

theorem pentagon_area_ln_coordinates
  (f : ℝ → ℝ) (x_coords : ℕ → ℕ)
  (vertices := set (ℝ × ℝ))
  (area : ℝ)
  (h₁ : ∀ x, f x = real.log x)
  (h₂ : ∀ n, x_coords n = 10 + n)
  (h₃ : vertices = { (x_coords 0, f (x_coords 0)),  (x_coords 1, f (x_coords 1)), (x_coords 2, f (x_coords 2)), (x_coords 3, f (x_coords 3)), (x_coords 4, f (x_coords 4)) })
  (h₄ : area = real.log (546 / 540)) :
  x_coords 0 = 10 :=
by
  sorry

end pentagon_area_ln_coordinates_l360_360667


namespace exists_N_with_large_digit_sum_l360_360231

open Nat

def digit_sum (n : ℕ) (b : ℕ) : ℕ :=
  if b ≤ 1 then 0 else
  let rec aux (m : ℕ) (acc : ℕ) :=
    match m with
    | 0 => acc
    | _ => aux (m / b) (acc + (m % b))
  aux n 0

theorem exists_N_with_large_digit_sum (m : ℕ) :
  ∃ (N : ℕ), ∀ (b : ℕ), 2 ≤ b ∧ b ≤ 1389 → digit_sum N b > m :=
sorry

end exists_N_with_large_digit_sum_l360_360231


namespace find_a_l360_360047

-- Define the condition that a is a real number
variable (a : ℝ)

-- Define the conjugate function for complex numbers
def conj (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Define the expression and its simplification
def isPureImaginary : Prop :=
  let z := ((a : ℂ) - Complex.i) / (1 + Complex.i)
  z.re = 0

-- The main theorem to prove
theorem find_a : isPureImaginary a → a = 1 :=
begin
  intro h,
  -- Proof is omitted (write your proof here)
  sorry
end

end find_a_l360_360047


namespace odd_function_value_l360_360056

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the function definition

-- Prove that f(-1/2) = -1/2 given the conditions
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = x) →
  f (-1/2) = -1/2 :=
by
  sorry

end odd_function_value_l360_360056


namespace area_of_triangle_BDE_l360_360544

-- Define the context of the triangle and its properties
variables (A B C D E O H : Type)
variables [triangle A B C]
variables [circumcenter O A B C]
variables [orthocenter H A B C]
variables [point_on D B C]
variables [point_on E A B]
variables {a : ℝ}

-- Given conditions
variables (H1 : angle B A C = 60)
variables (H2 : distance B D = distance B H)
variables (H3 : distance B E = distance B O)
variables (H4 : distance B O = a)

-- Prove that the area of triangle BDE is (sqrt 3 / 4) * a^2
theorem area_of_triangle_BDE : 
  area B D E = (sqrt 3 / 4) * a^2 := 
  sorry

end area_of_triangle_BDE_l360_360544


namespace min_value_of_function_at_extreme_point_l360_360095

noncomputable def f (x : ℝ) (a : ℝ) := (1/2) * x^2 - a * x - 3 * Real.log x

theorem min_value_of_function_at_extreme_point :
  (∀ a : ℝ, ∃ (x : ℝ) (hx : x = 3), f x a = f x 2 → f 3 2 = -(3/2) - 3 * Real.log 3) :=
begin
  sorry
end

end min_value_of_function_at_extreme_point_l360_360095


namespace sum_of_squares_l360_360907

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 5) (h2 : ab + bc + ac = 5) : a^2 + b^2 + c^2 = 15 :=
by sorry

end sum_of_squares_l360_360907


namespace dot_product_magnitude_l360_360176

variables {a b : EuclideanSpace 3 ℝ}

/- Given conditions -/
def norm_a : ℝ := ‖a‖ = 3
def norm_b : ℝ := ‖b‖ = 4
def norm_cross : ℝ := ‖a × b‖ = 6

/- Desired property to prove -/
theorem dot_product_magnitude :
  norm_a →
  norm_b →
  norm_cross →
  |(a ⋅ b)| = 6 * real.sqrt 3 :=
by
  sorry

end dot_product_magnitude_l360_360176


namespace simplest_form_of_expression_l360_360622

theorem simplest_form_of_expression (c : ℝ) : ((3 * c + 5 - 3 * c) / 2) = 5 / 2 :=
by 
  sorry

end simplest_form_of_expression_l360_360622


namespace roof_collapse_days_l360_360370

-- Definitions based on the conditions
def roof_capacity_pounds : ℕ := 500
def leaves_per_pound : ℕ := 1000
def leaves_per_day : ℕ := 100

-- Statement of the problem and the result
theorem roof_collapse_days :
  let total_leaves := roof_capacity_pounds * leaves_per_pound in
  let days := total_leaves / leaves_per_day in
  days = 5000 :=
by
  -- To be proven, so we use sorry for now
  sorry

end roof_collapse_days_l360_360370


namespace repeating_decimal_sum_l360_360444

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l360_360444


namespace degree_of_multiplied_polynomials_l360_360742

noncomputable def polynomial_degree (p : Polynomial ℝ) : ℕ :=
  p.nat_degree

theorem degree_of_multiplied_polynomials :
  polynomial_degree (Polynomial.C (1 : ℝ) * Polynomial.X^7 *
                    (Polynomial.C (1 : ℝ) * Polynomial.X + Polynomial.C (1 : ℝ) * Polynomial.X⁻¹) *
                    (Polynomial.C (1 : ℝ) * Polynomial.C (1 : ℝ) +
                     Polynomial.C (2 : ℝ) * Polynomial.X⁻¹ +
                     Polynomial.C (3 : ℝ) * Polynomial.X^(-3))) = 8 :=
by
  sorry

end degree_of_multiplied_polynomials_l360_360742


namespace at_least_two_identical_squares_l360_360788

theorem at_least_two_identical_squares :
  ∀ (rectangles : finset (ℕ × ℕ)),
    rectangles.card = 100 ∧
    (∃ squares : finset ℕ,
      squares.card = 9 ∧ ∀ s ∈ squares, ∃ w h, (w, h) ∈ rectangles ∧ w = h) →
    ∃ (x y ∈ (squares : finset ℕ)), x = y ∧ x ≠ y :=
by
  sorry

end at_least_two_identical_squares_l360_360788


namespace coefficient_B_is_1_l360_360071

-- Definitions based on the conditions
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- Given conditions
def condition1 (A B C D : ℝ) := g A B C D (-2) = 0 
def condition2 (A B C D : ℝ) := g A B C D 0 = -1
def condition3 (A B C D : ℝ) := g A B C D 2 = 0

-- The main theorem to prove
theorem coefficient_B_is_1 (A B C D : ℝ) 
  (h1 : condition1 A B C D) 
  (h2 : condition2 A B C D) 
  (h3 : condition3 A B C D) : 
  B = 1 :=
sorry

end coefficient_B_is_1_l360_360071


namespace number_251_is_not_unitary_number_1000_is_unitary_sum_six_permutations_is_multiple_of_222_sum_six_permutations_eq_2220_l360_360395

def is_unitary_number (n : ℕ) : Prop :=
  let rec digit_sum (x : ℕ) : ℕ :=
    match x with
    | 0       => 0
    | x + 1   => x % 10 + digit_sum (x / 10)
  in digit_sum n % 9 = 1

theorem number_251_is_not_unitary : ¬ is_unitary_number 251 := sorry

theorem number_1000_is_unitary : is_unitary_number 1000 := sorry

variable (a b : ℕ)

axiom h_a : 2 ≤ a ∧ a ≤ 8
axiom h_b : 2 ≤ b ∧ b ≤ 8
axiom h_unitary : is_unitary_number (100 * a + 10 * b + 1)

theorem sum_six_permutations_is_multiple_of_222 : 
  let n := 100 * a + 10 * b + 1 in
  (n + (100 * a + 10 + b) + (100 * b + 10 * a + 1) + (100 * b + 10 + a) + (100 + 10 * a + b) + (100 + 10 * b + a)) % 222 = 0 := sorry

theorem sum_six_permutations_eq_2220 : 
  (100 * a + 10 * b + 1 + 100 * a + 10 + b + 100 * b + 10 * a + 1 + 100 * b + 10 + a + 100 + 10 * a + b + 100 + 10 * b + a) = 2220 := sorry

end number_251_is_not_unitary_number_1000_is_unitary_sum_six_permutations_is_multiple_of_222_sum_six_permutations_eq_2220_l360_360395


namespace max_solitar_game_result_l360_360383

-- Condition definitions:
def gpd (n : ℕ) : ℕ :=
  if n < 2 then n else
  let primes := (List.finRange n).filter (λ p, Nat.Prime p ∧ n % p = 0)
  primes.maximum' (by simp [Nat.Prime, Nat.divisors])

def solitar_game (nums : List ℕ) : ℕ :=
  match nums with
  | [] => 1
  | [n] => n
  | _ =>
    let pairs := nums.pairwise (by decide)
    let sums := pairs.map (λ pair, pair.fst + pair.snd)
    let new_nums := sums.map gpd
    solitar_game new_nums

theorem max_solitar_game_result : solitar_game (List.finRange 16) = 19 :=
  sorry

end max_solitar_game_result_l360_360383


namespace polynomials_common_zero_k_l360_360471

theorem polynomials_common_zero_k
  (k : ℝ) :
  (∃ x : ℝ, (1988 * x^2 + k * x + 8891 = 0) ∧ (8891 * x^2 + k * x + 1988 = 0)) ↔ (k = 10879 ∨ k = -10879) :=
sorry

end polynomials_common_zero_k_l360_360471


namespace cornelia_age_l360_360116

theorem cornelia_age :
  ∃ C : ℕ, 
  (∃ K : ℕ, K = 30 ∧ (C + 20 = 2 * (K + 20))) ∧
  ((K - 5)^2 = 3 * (C - 5)) := by
  sorry

end cornelia_age_l360_360116


namespace circle_radius_l360_360627

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l360_360627


namespace smallest_number_divisibility_l360_360723

-- Define the numbers
def a := 12
def b := 16
def c := 18
def d := 21
def e := 28

-- Define the function to calculate LCM
def lcm (m n : ℕ) : ℕ := m / Nat.gcd m n * n

-- Calculate the LCM of the given numbers
def lcm_list (lst: List ℕ) : ℕ :=
  lst.foldl lcm 1

def lcm_abcde : ℕ := lcm_list [a, b, c, d, e]

-- The desired number
def x : ℕ := lcm_abcde + 5

theorem smallest_number_divisibility :
  x = 1013 :=
by 
  -- Proof omitted
  sorry

end smallest_number_divisibility_l360_360723


namespace digit2_in_primes_under_50_l360_360797

theorem digit2_in_primes_under_50 : 
    (List.filter (λ x => '2' ∈ x.digits 10) (List.filter Nat.Prime (Nat.range 50))).sum (λ n => n.digit_count 2) = 3 := by
  sorry

end digit2_in_primes_under_50_l360_360797


namespace problem_BD_correct_l360_360040

noncomputable theory

open MeasureTheory
open ProbabilityTheory

variables (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)

variables (A B : Set Ω)
variables (P_A : P A = 0.5) (P_B : P B = 0.2)

theorem problem_BD_correct :
  (Disjoint A B → P (A ∪ B) = 0.7) ∧ 
  (condProb P B A = 0.2 → IndepEvents A B P) :=
by 
  intro HAB;
  apply And.intro;
  {
    -- Prove B
    intro h_disjoint;
    have h_Union : P (A ∪ B) = P A + P B := P.add_disjoint h_disjoint;
    rw [P_A, P_B] at h_Union;
    norm_num at h_Union;
    exact h_Union;
  };
  {
    -- Prove D
    intro h_condProb;
    refine ⟨?_⟩;
    {
      intro _ _;
      simp only [IndependentSets, MeasureTheoreticProbability, condProb, tprod, MeasureTheory.cond, habspace];
      simp at h_condProb
      assumption
    }
  };
  sorry -- further missing justification required

end problem_BD_correct_l360_360040


namespace avg_remaining_two_l360_360249

-- Defining the given conditions
variable (six_num_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ)

-- Defining the known values
axiom avg_val : six_num_avg = 3.95
axiom avg_group1 : group1_avg = 3.6
axiom avg_group2 : group2_avg = 3.85

-- Stating the problem to prove that the average of the remaining 2 numbers is 4.4
theorem avg_remaining_two (h : six_num_avg = 3.95) 
                           (h1: group1_avg = 3.6)
                           (h2: group2_avg = 3.85) : 
  4.4 = ((six_num_avg * 6) - (group1_avg * 2 + group2_avg * 2)) / 2 := 
sorry

end avg_remaining_two_l360_360249


namespace dot_product_magnitude_l360_360185

variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜 → 𝕜} 

-- Define the conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_a_cross_b : ℝ := 6

-- Theorem statement
theorem dot_product_magnitude :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∥a × b∥ = norm_a_cross_b) → abs ((a.toReal).dot (b.toReal)) = 6 * sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end dot_product_magnitude_l360_360185


namespace value_of_a_l360_360565

-- Define the given conditions
def b := 2
def A := 45 * Real.Angle.deg
def C := 75 * Real.Angle.deg
noncomputable def sin := Real.sin
noncomputable def angle := Real.Angle

-- Define the calculated value of B
def B := 180 * angle.deg - A - C

-- Define sine values for specific angles
noncomputable def sin_45 := sin (45 * angle.deg)
noncomputable def sin_60 := sin (60 * angle.deg)

-- Define side a using the Sine Rule and given conditions
noncomputable def a := b * sin_45 / sin_60

-- Statement to prove
theorem value_of_a : a = (2 / 3) * Real.sqrt 6 := by sorry

end value_of_a_l360_360565


namespace sum_of_factorials_perfect_square_l360_360453

def is_perfect_square (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

def sum_of_factorials (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ k, k.factorial)

theorem sum_of_factorials_perfect_square (n : ℕ) :
  is_perfect_square (sum_of_factorials n) ↔ n = 1 ∨ n = 3 :=
by sorry

end sum_of_factorials_perfect_square_l360_360453


namespace slope_of_line_angle_60_deg_l360_360979

theorem slope_of_line_angle_60_deg : 
  ∀ (θ : ℝ), θ = 60 * Real.pi / 180 → Real.tan θ = Real.sqrt 3 :=
begin
  sorry
end

end slope_of_line_angle_60_deg_l360_360979


namespace tournament_matches_l360_360554

theorem tournament_matches (n : ℕ) (h : n = 128) : ∃ m : ℕ, m = n - 1 := 
by 
  rw h
  use 127
  rw nat.sub_one
  exact rfl

end tournament_matches_l360_360554


namespace max_lambda_l360_360206

theorem max_lambda
  (x y : ℝ)
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : x^3 + y^3 = x - y) :
  ∃ λ : ℝ, (∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x^3 + y^3 = x - y) → (x^2 + λ * y^2 ≤ 1)) ∧
  (λ = 2 + 2 * Real.sqrt 2) :=
sorry

end max_lambda_l360_360206


namespace sin_double_angle_plus_pi_over_4_l360_360965

theorem sin_double_angle_plus_pi_over_4 (α : ℝ) 
  (h : Real.tan α = 3) : 
  Real.sin (2 * α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end sin_double_angle_plus_pi_over_4_l360_360965


namespace smallest_lcm_l360_360092

theorem smallest_lcm (m n : ℕ) (hm : 10000 ≤ m ∧ m < 100000) (hn : 10000 ≤ n ∧ n < 100000) (h : Nat.gcd m n = 5) : Nat.lcm m n = 20030010 :=
sorry

end smallest_lcm_l360_360092


namespace min_value_f_range_of_a_inequality_two_zeros_l360_360041

noncomputable def f (x : ℝ) : ℝ := x * Real.log (x + 1)

noncomputable def g (a x : ℝ) : ℝ := a * (x + 1 / (x + 1) - 1)

theorem min_value_f : ∃ x, f x = 0 ∧ ∀ y, f y ≥ f x := by
  sorry

theorem range_of_a (a : ℝ) : (∀ x ∈ Ioo (-1 : ℝ) 0, f x ≤ g a x) → 1 ≤ a := by
  sorry

theorem inequality_two_zeros (b x1 x2 : ℝ) (hb : f x1 = b ∧ f x2 = b) : 
  2 * (abs (x1 - x2)) > Real.sqrt (b ^ 2 + 4 * b) + 2 * Real.sqrt b - b := by
  sorry

end min_value_f_range_of_a_inequality_two_zeros_l360_360041


namespace principal_range_of_argument_l360_360891

theorem principal_range_of_argument {z : ℂ} (hz : abs (2 * z + 1 / z) = 1) :
  ∃ (k : ℕ), k ≤ 1 ∧ 
  (k * π + π / 2 - 1 / 2 * real.arccos (3 / 4) ≤ complex.arg z ∧ complex.arg z ≤ k * π + π / 2 + 1 / 2 * real.arccos (3 / 4)) :=
sorry

end principal_range_of_argument_l360_360891


namespace smallest_Y_l360_360148

theorem smallest_Y (S : ℕ) (h1 : (∀ d ∈ S.digits 10, d = 0 ∨ d = 1)) (h2 : 18 ∣ S) : 
  (∃ (Y : ℕ), Y = S / 18 ∧ ∀ (S' : ℕ), (∀ d ∈ S'.digits 10, d = 0 ∨ d = 1) → 18 ∣ S' → S' / 18 ≥ Y) → 
  Y = 6172839500 :=
sorry

end smallest_Y_l360_360148


namespace abs_dot_product_l360_360171

variables (a b : ℝ^3)

-- Conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_cross_ab : ℝ := 6

-- Theorem statement
theorem abs_dot_product (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (hcross : ‖a × b‖ = norm_cross_ab) :
  |(a ⬝ b)| = 6 * sqrt 3 :=
by
  sorry

end abs_dot_product_l360_360171


namespace identify_speaker_as_vampire_l360_360366

-- Define the Transylvanian statement about sanity
def transylvanian_statement := "I am not a sane person"

-- Define a Transylvanian
axiom Transylvanian (x : Type) : Prop

-- Define a vampire
axiom Vampire (x : Type) : Prop

-- Define the loss of sanity
axiom lost_sanity (x : Type) : Prop

-- Assume the entity making the statement is a Transylvanian
variable {x : Type}
variable (H : Transylvanian x)

-- Define the proof problem statement
theorem identify_speaker_as_vampire (H1 : transylvanian_statement) : Vampire x ∧ lost_sanity x := 
by sorry

end identify_speaker_as_vampire_l360_360366


namespace arithmetic_mean_of_integers_from_neg6_to_6_l360_360292

theorem arithmetic_mean_of_integers_from_neg6_to_6 :
  ∃ (mean : ℚ), mean = (finset.range (12 + 1)).sum (λ x, (x - 6)) / (12 + 1) :=
by 
  let n := 12 + 1
  let integers := (finset.range n).map (λ x, x - 6)
  let sum := integers.sum
  have hsum : sum = 0, from sorry
  use (sum : ℚ) / n
  simp [hsum]
  exact sorry

end arithmetic_mean_of_integers_from_neg6_to_6_l360_360292


namespace remainder_of_polynomial_division_l360_360377

theorem remainder_of_polynomial_division :
  ∀ (z : ℚ), let p := 4 * z ^ 4 - 3 * z ^ 3 + 2 * z ^ 2 - 16 * z + 9
              let d := 4 * z + 6
  remainder p d = 173 / 12 := 
by
  intros
  let p := 4 * z ^ 4 - 3 * z ^ 3 + 2 * z ^ 2 - 16 * z + 9
  let d := 4 * z + 6
  -- Mathematical steps would go here
  sorry

end remainder_of_polynomial_division_l360_360377


namespace inequality_proof_l360_360230

-- Define positive real numbers
variables {x y z : ℝ}

-- State the proof problem
theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  sqrt (x^2 + 3*y^2) + sqrt (x^2 + z^2 + x*z) > sqrt (z^2 + 3*y^2 + 3*y*z) :=
sorry

end inequality_proof_l360_360230


namespace minimum_a_value_l360_360059

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), 0 < x → 0 < y → x^2 + 2 * x * y ≤ a * (x^2 + y^2)) ↔ a ≥ (Real.sqrt 5 + 1) / 2 := 
sorry

end minimum_a_value_l360_360059


namespace cos_angle_between_vectors_l360_360515

open Real EuclideanSpace

-- Define the vectors and their conditions
variables (a b : ℝ × ℝ)
variable (θ : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 2 • a + b = (3, 3)
def condition2 : Prop := a - 2 • b = (-1, 4)

-- Define the cos θ definition using the formula given in the problem statement
def cos_theta (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / ((sqrt (a.1 ^ 2 + a.2 ^ 2)) * (sqrt (b.1 ^ 2 + b.2 ^ 2)))

-- Lean 4 statement for theorem
theorem cos_angle_between_vectors : condition1 a b → condition2 a b → cos_theta a b = - (sqrt 10) / 10 :=
by
  intro h1 h2
  sorry

end cos_angle_between_vectors_l360_360515


namespace polygon_exists_with_properties_l360_360229

theorem polygon_exists_with_properties (n : ℕ) (h : n ≥ 14) : 
  ∃ (polygon : Type) [has_sides polygon n] [has_property polygon], true :=
sorry

end polygon_exists_with_properties_l360_360229


namespace player_one_wins_l360_360601

theorem player_one_wins (n : ℕ) :
  let bills := (list.range (2 * n + 1)).tail in
  ∃ strategy : (list ℕ) → ℕ,
    ∀ (turns : list ℕ), (∀ t ∈ turns, t ∈ bills) →
    wins_with_strategy strategy turns →
    first_player_accumulates_greater_sum strategy turns :=
sorry

end player_one_wins_l360_360601


namespace fraction_shiny_igneous_rocks_l360_360117

-- Definitions and conditions
def num_sedimentary_rocks (S : ℕ) : Prop := true
def num_igneous_rocks (I : ℕ) : Prop := I = (1 / 2 : ℚ) * S
def shiny_igneous_rocks (Si : ℕ) : Prop := Si = 30
def shiny_sedimentary_rocks (Ss : ℕ) : Prop := Ss = (1 / 5 : ℚ) * S
def total_rocks (S I : ℕ) : Prop := S + I = 270

-- Theorem to prove
theorem fraction_shiny_igneous_rocks (S I Si Ss: ℕ) 
  (h1 : num_sedimentary_rocks S)
  (h2 : num_igneous_rocks I)
  (h3 : shiny_igneous_rocks Si)
  (h4 : shiny_sedimentary_rocks Ss)
  (h5 : total_rocks S I) : (Si : ℚ) / I = 1 / 3 := by
  sorry

end fraction_shiny_igneous_rocks_l360_360117


namespace original_class_strength_l360_360313

theorem original_class_strength (x : ℕ) 
    (avg_original : ℕ)
    (num_new : ℕ) 
    (avg_new : ℕ) 
    (decrease : ℕ)
    (h1 : avg_original = 40)
    (h2 : num_new = 17)
    (h3 : avg_new = 32)
    (h4 : decrease = 4)
    (h5 : (40 * x + 17 * avg_new) = (x + num_new) * (40 - decrease))
    : x = 17 := 
by {
  sorry
}

end original_class_strength_l360_360313


namespace total_gray_area_l360_360990

noncomputable def smallest_circle_radius : ℝ := sorry
def middle_circle_radius (r : ℝ) : ℝ := 2 * r
def largest_circle_radius (r : ℝ) : ℝ := 3 * r
def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem total_gray_area 
  (r_smallest : ℝ)
  (h1 : middle_circle_radius r_smallest = 2 * r_smallest)
  (h2 : largest_circle_radius r_smallest = 3 * r_smallest)
  (h3 : ∀ r, middle_circle_radius r - r = 1) 
  (h4 : ∀ r, largest_circle_radius r - middle_circle_radius r = 1) :
  let area_smallest := area_of_circle r_smallest,
      area_middle := area_of_circle (middle_circle_radius r_smallest),
      area_largest := area_of_circle (largest_circle_radius r_smallest),
      gray_1 := area_middle - area_smallest,
      gray_2 := area_largest - area_middle in
  (gray_1 + gray_2) = 8 * π * r_smallest^2 := by
  sorry

end total_gray_area_l360_360990


namespace find_domain_of_f_zeros_of_F_in_interval_l360_360514

noncomputable def f (x : ℝ) : ℝ := (1 + Real.tan x) * (Real.sin x) ^ 2

def domain_f := {x : ℝ | ∀ k : ℤ, x ≠ (π / 2 + k * π)}

theorem find_domain_of_f : ∀ x : ℝ, x ∈ domain_f :=
by
  intro x
  sorry

noncomputable def F (x : ℝ) : ℝ := f x - 2

def find_zeros_of_F : Set ℝ := {x | 0 < x ∧ x < π ∧ F x = 0}

theorem zeros_of_F_in_interval : find_zeros_of_F = {π / 4, π / 2} :=
by
  sorry

end find_domain_of_f_zeros_of_F_in_interval_l360_360514


namespace parameterized_line_l360_360345

theorem parameterized_line (s m : ℝ) : 
  (∃ t : ℝ, ∀ x y : ℝ, (x, y) = (s, -3) + t * (3, m) ∧ y = 2 * x + 5) → 
    (s = -4 ∧ m = 6) := 
by 
  assume h,
  sorry

end parameterized_line_l360_360345


namespace sqrt_power_expression_l360_360450

-- Define the conditions and expressions
def sqrt5 : Real := Real.sqrt 5
def expression : Real := (Real.sqrt (sqrt5 ^ 5)) ^ 4

-- State the theorem to be proved
theorem sqrt_power_expression : expression = 9765625 :=
by
  sorry -- Proof is not required

end sqrt_power_expression_l360_360450


namespace find_number_l360_360767

def problem (x : ℝ) : Prop :=
  0.25 * x = 130 + 190

theorem find_number (x : ℝ) (h : problem x) : x = 1280 :=
by 
  sorry

end find_number_l360_360767


namespace sum_of_roots_l360_360737

theorem sum_of_roots (a b c: ℝ) (h: a ≠ 0) (h_eq : a = 1) (h_eq2 : b = -6) (h_eq3 : c = 8):
    let Δ := b ^ 2 - 4 * a * c in
    let root1 := (-b + real.sqrt Δ) / (2 * a) in
    let root2 := (-b - real.sqrt Δ) / (2 * a) in
    root1 + root2 = 6 :=
by
  sorry

end sum_of_roots_l360_360737


namespace points_on_equation_correct_l360_360947

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l360_360947


namespace sin_double_angle_given_cos_identity_l360_360887

theorem sin_double_angle_given_cos_identity (α : ℝ) 
  (h : Real.cos (α + π / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 3 / 4 :=
by
  sorry

end sin_double_angle_given_cos_identity_l360_360887


namespace repeating_decimal_sum_l360_360396

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l360_360396


namespace power_function_value_l360_360058

/-- Given a power function passing through a certain point, find the value at a specific point -/
theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h : f x = x ^ α) 
  (h_passes : f (1/4) = 4) : f 2 = 1/2 :=
sorry

end power_function_value_l360_360058


namespace intersection_and_line_eqns_l360_360933

noncomputable def l1 : AffinePlane.Coord ℝ := { x : ℝ, y : ℝ // x - 2 * y + 4 = 0 }
noncomputable def l2 : AffinePlane.Coord ℝ := { x : ℝ, y : ℝ // x + y - 2 = 0 }

-- 1. Definition for the intersection point of l1 and l2
def P : AffinePlane.Point ℝ := ⟨0, 2⟩

-- 2. Equations of the lines passing through P
def l_parallel_to_l3 : AffinePlane.Coord ℝ := { x : ℝ, y : ℝ // 3 * x - 4 * y + 8 = 0 }
def l_perpendicular_to_l3 : AffinePlane.Coord ℝ := { x : ℝ, y : ℝ //  4 * x + 3 * y - 6 = 0 }

-- Theorem stating the required results
theorem intersection_and_line_eqns :
  (P.x - 2 * P.y + 4 = 0 ∧ P.x + P.y - 2 = 0) ∧
  (3 * P.x - 4 * P.y + 8 = 0) ∧
  (4 * P.x + 3 * P.y - 6 = 0) :=
by {
  -- Intersection solves as (0, 2)
  have P_coord : (0 - 2 * 2 + 4 = 0) ∧ (0 + 2 - 2 = 0) := by {
    -- x = 0, y = 2 satisfies both lines
    auto,
  },

  -- Parallel line: 3 * 0 - 4 * 2 + 8 = 0
  have parallel_eqn : 3 * 0 - 4 * 2 + 8 = 0 := by {
    -- correct equation
    auto,
  },

  -- Perpendicular line: 4 * 0 + 3 * 2 - 6 = 0
  have perpendicular_eqn : 4 * 0 + 3 * 2 - 6 = 0 := by {
    -- correct equation
    auto,
  },

  exact ⟨P_coord, ⟨parallel_eqn, perpendicular_eqn⟩⟩
}

end intersection_and_line_eqns_l360_360933


namespace inequality_solution_l360_360480

theorem inequality_solution (n : ℤ) (h1 : n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3) (h2 : (-1 : ℤ) ^ n > (-1 : ℤ) ^ n) : n = -1 ∨ n = 2 :=
    sorry

end inequality_solution_l360_360480


namespace tower_height_l360_360652

noncomputable def height_of_tower 
  (c : ℝ) 
  (delta epsilon alpha : ℝ) 
  (tan_delta tan_epsilon sin_alpha : ℝ) : ℝ := 
  let b := (c * sin (33.291667 * Math.pi / 180)) / sin (43.358333 * Math.pi / 180)
  b * tan (delta * Math.pi / 180)

theorem tower_height : 
  ∀ (c : ℝ) (delta epsilon alpha : ℝ),
    c = 333.4 →
    delta = 12.083333 →
    epsilon = 6.916667 →
    alpha = 103.35 →
    tan delta = 0.2137 →
    tan epsilon = 0.1218 →
    sin alpha = 0.961 →
    height_of_tower c delta epsilon alpha 0.2137 0.1218 0.961 ≈ 57.1 :=
by
  intros c delta epsilon alpha hc hdelta hepsilon halpha htan_delta htan_epsilon hsin_alpha
  sorry

end tower_height_l360_360652


namespace circle_radius_l360_360629

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l360_360629


namespace hidden_faces_dots_sum_l360_360881

theorem hidden_faces_dots_sum (visible_faces : List ℕ) (h : visible_faces = [1, 1, 2, 3, 3, 4, 5, 6]) :
  84 - (1 + 1 + 2 + 3 + 3 + 4 + 5 + 6) = 59 :=
by { rw h, norm_num, }

end hidden_faces_dots_sum_l360_360881


namespace limit_problem_l360_360815

open Real

noncomputable def problem_statement : Prop :=
  ∀ (x : ℝ), Filter.Tendsto (λ x, (x * (2 + sin (1/x)) + 8 * cos x) ^ (1/3)) (𝓝 0) (𝓝 2)

theorem limit_problem : problem_statement :=
  by
    sorry

end limit_problem_l360_360815


namespace monotonic_function_range_l360_360975

theorem monotonic_function_range (m : ℝ) :
  (∀ x y : ℝ, (x^3 + x^2 + m * x + 1) ≤ (y^3 + y^2 + m * y + 1) ∨
                  (x^3 + x^2 + m * x + 1) ≥ (y^3 + y^2 + m * y + 1)) →
  m ≥ (1 / 3) :=
begin
  sorry
end

end monotonic_function_range_l360_360975


namespace arithmetic_sequence_sixtieth_term_l360_360649

theorem arithmetic_sequence_sixtieth_term (a1 : ℕ) (a15 : ℕ) (d : ℕ) (a_60 : ℕ) (h1 : a1 = 7) (h2 : a15 = 35) (h_d : d = 2) : (a1 + (60 - 1) * d) = a_60 :=
by 
  rw [h1, h2, h_d]
  have h_d_value : 14 * d = (a15 - a1), by linarith,
  have d_value : d = 2, by linarith,
  rw [d_value] at *,
  exact (calc
    a1 + (60 - 1) * d = 7 + 59 * 2 : by rw [h1, d_value]
    ... = 7 + 118 : by norm_num
    ... = 125 : by norm_num),
  exact h_d_value,
  exact sorry

end arithmetic_sequence_sixtieth_term_l360_360649


namespace min_value_of_f_f_is_constant_l360_360066

def f (x α β : ℝ) := (Real.sin x)^2 + (Real.sin (x + α))^2 + (Real.sin (x + β))^2

theorem min_value_of_f : ∀ x : ℝ, 0 ≤ α ∧ α = Real.pi / 4 ∧ β = 3 * Real.pi / 4 → f x α β ≥ 1 :=
by
  intros
  sorry

theorem f_is_constant : ∃ α β : ℝ, 0 ≤ α ∧ α ≤ β ∧ β ≤ Real.pi ∧ 
    (∀ x₁ x₂ : ℝ, f x₁ α β = f x₂ α β) ↔ (α = Real.pi / 3 ∧ β = 2 * Real.pi / 3) :=
by
  sorry

end min_value_of_f_f_is_constant_l360_360066


namespace sum_arithmetic_sequence_eq_square_l360_360598

theorem sum_arithmetic_sequence_eq_square (n : ℕ) :
    (∑ k in Finset.range (2 * n - 1), (n + k)) = (2 * n - 1)^2 := 
by 
  sorry

end sum_arithmetic_sequence_eq_square_l360_360598


namespace clock_angle_at_7_oclock_l360_360704

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l360_360704


namespace find_hit_rate_B_l360_360689

structure ShootingCompetition :=
(hit_rate_A : ℚ)
(hit_rate_B : ℚ)
(Pr_equal_2 : ℚ)

def event_A : ShootingCompetition → Prop :=
λ c, c.hit_rate_A = 3/5

def event_B : ShootingCompetition → Prop :=
λ c, ∃ p, c.hit_rate_B = p

def condition_sum_scores (c : ShootingCompetition) : Prop :=
c.Pr_equal_2 = 9/20

theorem find_hit_rate_B (c : ShootingCompetition)
  (h_A : event_A c)
  (h_equal_2 : condition_sum_scores c)
  (h_independent : True) : c.hit_rate_B = 3/4 :=
sorry

end find_hit_rate_B_l360_360689


namespace find_n_l360_360452

theorem find_n : ∃ n : ℕ, (∃ A B : ℕ, A ≠ B ∧ 10^(n-1) ≤ A ∧ A < 10^n ∧ 10^(n-1) ≤ B ∧ B < 10^n ∧ (10^n * A + B) % (10^n * B + A) = 0) ↔ n % 6 = 3 :=
by
  sorry

end find_n_l360_360452


namespace Ann_is_16_l360_360798

variable (A S : ℕ)

theorem Ann_is_16
  (h1 : A = S + 5)
  (h2 : A + S = 27) :
  A = 16 :=
by
  sorry

end Ann_is_16_l360_360798


namespace trapezoid_intersection_l360_360792

-- Given conditions.
variables (M K P E A B C D H : Type)
variable (trapezoid_inscribed : IsTrapezoid M K P E)
variable (parallel_sides : Parallel MK PE AC)

-- Target statement to prove.
theorem trapezoid_intersection (parallel_condition : Parallel_sides_are_parallel MK PE AC (trapezoid_inscribed)) :
  ∃ H, IntersectsAtPoint ME BD KP H :=
by
  sorry

end trapezoid_intersection_l360_360792


namespace museum_discount_l360_360391

theorem museum_discount
  (Dorothy_age : ℕ)
  (total_family_members : ℕ)
  (regular_ticket_cost : ℕ)
  (discountapplies_age : ℕ)
  (before_trip : ℕ)
  (after_trip : ℕ)
  (spend : ℕ := before_trip - after_trip)
  (adults_tickets : ℕ := total_family_members - 2)
  (youth_tickets : ℕ := 2)
  (total_cost := adults_tickets * regular_ticket_cost + youth_tickets * (regular_ticket_cost - regular_ticket_cost * discount))
  (discount : ℚ)
  (expected_spend : ℕ := 44) :
  total_cost = spend :=
by
  sorry

end museum_discount_l360_360391


namespace percentage_reduction_price_increase_l360_360334

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end percentage_reduction_price_increase_l360_360334


namespace zero_inside_interval_l360_360259

-- Define the function
def f (x : ℝ) := 2 * x - 5

-- State the main theorem
theorem zero_inside_interval : 
  (∃ x ∈ Ioo 2 3, f x = 0) := 
sorry

end zero_inside_interval_l360_360259


namespace heartsuit_symmetric_solution_l360_360385

def heartsuit (a b : ℝ) : ℝ :=
  a^3 * b - a^2 * b^2 + a * b^3

theorem heartsuit_symmetric_solution :
  ∀ x y : ℝ, (heartsuit x y = heartsuit y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end heartsuit_symmetric_solution_l360_360385


namespace division_by_reciprocal_l360_360087

theorem division_by_reciprocal :
  (10 / 3) / (1 / 5) = 50 / 3 := 
sorry

end division_by_reciprocal_l360_360087


namespace a_n_formula_S_n_sum_l360_360272

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 1 then 1 else real.sqrt n

def a_n' := λ n : ℕ, real.sqrt n

noncomputable def b_n (n : ℕ) : ℝ :=
  1 / (a_n n + a_n (n + 1))

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ k in finset.range n, b_n (k + 1)

theorem a_n_formula (n : ℕ) (h : 1 ≤ n) : 
  a_n n = real.sqrt n :=
by sorry

theorem S_n_sum (n : ℕ) : 
  S_n n = real.sqrt (n + 1) - 1 :=
by sorry

end a_n_formula_S_n_sum_l360_360272


namespace ones_digit_large_power_dividing_32_factorial_l360_360855

theorem ones_digit_large_power_dividing_32_factorial :
  let n := 32!
  let largestPower := 2^31
  ones_digit largestPower = 8 :=
by
  sorry

end ones_digit_large_power_dividing_32_factorial_l360_360855


namespace probability_negative_product_l360_360758

theorem probability_negative_product : 
  let m := { -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4 }
  let t := { -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7 }
  (7 * 7 + 4 * 4) / (12 * 12) = 65 / 144 :=
by
  sorry

end probability_negative_product_l360_360758


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360844

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  let n := 32 in
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i) in 
  (2 ^ k) % 10 = 8 :=
by
  let n := 32
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i)
  
  have h1 : k = 31 := by sorry
  have h2 : (2 ^ 31) % 10 = 8 := by sorry
  
  exact h2

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360844


namespace repeating_decimal_sum_l360_360443

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l360_360443


namespace sequence_non_zero_l360_360481

noncomputable def sequence : ℕ → ℤ
| 0       := 0
| 1       := 1
| 2       := 2
| (n + 2) := if (sequence n * sequence (n + 1)) % 2 = 0 then 5 * sequence (n + 1) - 3 * sequence n 
             else sequence (n + 1) - sequence n

theorem sequence_non_zero : ∀ n : ℕ, sequence n ≠ 0 := 
sorry

end sequence_non_zero_l360_360481


namespace repeating_decimal_sum_l360_360440

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l360_360440


namespace dot_product_magnitude_l360_360165

open Real

variables (a b : ℝ^3) 

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross (u v : ℝ^3) : ℝ^3 := (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)
def dot (u v : ℝ^3) : ℝ := u.1*v.1 + u.2*v.2 + u.3*v.3

axiom norm_a : norm a = 3
axiom norm_b : norm b = 4
axiom cross_norm : norm (cross a b) = 6

theorem dot_product_magnitude : abs (dot a b) = 6 * sqrt 3 := by
  sorry

end dot_product_magnitude_l360_360165


namespace number_of_integers_satisfying_inequality_l360_360822

theorem number_of_integers_satisfying_inequality :
  (∃ (n : ℤ), 3 * (n - 1) * (n + 5) < 0) → 
  (finset.Icc (-4) 0).card = 5 := by
  sorry

end number_of_integers_satisfying_inequality_l360_360822


namespace count_two_digit_numbers_l360_360682

def bags : Finset ℕ := {2, 3, 4}

def valid_two_digit_numbers (tens ones : ℕ) : Prop :=
  tens ≠ ones ∧ tens ∈ bags ∧ ones ∈ bags

def two_digit_numbers : Finset (ℕ × ℕ) :=
  (bags.product bags).filter (λ p, valid_two_digit_numbers p.1 p.2)

theorem count_two_digit_numbers : two_digit_numbers.card = 6 := by
  sorry

end count_two_digit_numbers_l360_360682


namespace mass_percentage_K_correct_l360_360348

noncomputable def massOfKBr : ℝ := 12
noncomputable def massOfKBrO3 : ℝ := 18
noncomputable def molarMassOfKBr : ℝ := 119
noncomputable def molarMassOfKBrO3 : ℝ := 167
noncomputable def molarMassOfK : ℝ := 39

noncomputable def molesOfKInKBr : ℝ := massOfKBr / molarMassOfKBr
noncomputable def molesOfKInKBrO3 : ℝ := massOfKBrO3 / molarMassOfKBrO3
noncomputable def totalMolesOfK : ℝ := molesOfKInKBr + molesOfKInKBrO3
noncomputable def totalMassOfK : ℝ := totalMolesOfK * molarMassOfK
noncomputable def totalMassOfMixture : ℝ := massOfKBr + massOfKBrO3

noncomputable def massPercentageOfK : ℝ := (totalMassOfK / totalMassOfMixture) * 100

theorem mass_percentage_K_correct : massPercentageOfK ≈ 27.118 := by sorry

end mass_percentage_K_correct_l360_360348


namespace division_remainder_sum_l360_360826

theorem division_remainder_sum :
  let dividend := 73648
  let divisor := 874
  let quotient := 73648 / 874
  let remainder := 73648 % 874
  quotient = 84 ∧ remainder = 232 ∧ remainder + 375 = 607 :=
by
  let dividend := 73648
  let divisor := 874
  let quotient := 73648 / 874
  let remainder := 73648 % 874
  split
  case left =>
    show quotient = 84
    by
      sorry
  case right =>
    split
    case left =>
      show remainder = 232
      by
        sorry
    case right =>
      show remainder + 375 = 607
      by
        sorry

end division_remainder_sum_l360_360826


namespace initial_roses_count_l360_360282

theorem initial_roses_count
  (initial_orchids : ℕ)
  (final_roses : ℕ)
  (final_orchids : ℕ)
  (orchids_more_than_roses : ℕ) :
  initial_orchids = 12 →
  final_roses = 11 →
  final_orchids = 20 →
  orchids_more_than_roses = 9 →
  (final_orchids - orchids_more_than_roses = initial_orchids - final_roses) →
  ∃ initial_roses : ℕ, initial_roses = 3 :=
by
  intros h1 h2 h3 h4 h5
  use 3
  rw [← h1, ← h2, ← h3, ← h4] at h5
  exact h5

sorry

end initial_roses_count_l360_360282


namespace angle_between_clock_hands_at_7_oclock_l360_360712

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l360_360712


namespace measure_of_angle_R_is_86_67_l360_360285

noncomputable def measure_of_angle_R (P Q R : ℝ) : Prop :=
triangle_is_isosceles_with_congruent_angles P Q ∧ (R = P + 40)

theorem measure_of_angle_R_is_86_67 (P Q R : ℝ) :
  measure_of_angle_R P Q R → R = 260 / 3 :=
by
  intro h

  have h_sum : 2 * P + (P + 40) = 180 := sorry
  have h_equals : 3 * P + 40 = 180 := sorry
  have hy : P = 140 / 3 := sorry
  calc R = P + 40
      ... = 140 / 3 + 40
      ... = 260 / 3
  sorry

end measure_of_angle_R_is_86_67_l360_360285


namespace determine_BX_l360_360227

theorem determine_BX (R : ℝ) (φ ψ : ℝ) (hR_pos : 0 < R) :
  ∃ BX : ℝ, BX = (2 * R * Real.sin φ * Real.sin ψ) / Real.sin (Real.abs (φ + ψ)) := 
sorry

end determine_BX_l360_360227


namespace least_trees_required_l360_360342

theorem least_trees_required : ∃ n : ℕ, (∀ k ∈ [6, 7, 8], k ∣ n) ∧ (∀ m < n, (∀ k ∈ [6, 7, 8], k ∣ m) → False) :=
by
  existsi 168
  split
  · intros k hk
    cases hk
    case inl hin_fst =>
      rw hin_fst; exact dvd_lcm_left _ _
    case inr hin_snd =>
      cases hin_snd
      case inl hin_snd_inl => 
        rw hin_snd_inl; exact dvd_lcm_right_left _ _
      case inr hin_snd_snd =>
        cases hin_snd_snd
        case inl hin_snd_snd_inl =>
          rw hin_snd_snd_inl
          rw [Nat.lcm_comm]
          exact dvd_lcm_right_right _ _
        case inr hin_snd_snd_snd =>
          exfalso
          exact List.not_mem_nil _ hin_snd_snd_snd
  · intros m hlt hdvd
    have : m ∣ 168 := hdvd.elim (λ h, h _ (List.mem_cons_self _ _)) (λ h, h _ (List.mem_cons_of_mem _ (List.mem_cons _ _ _)))
    cases this
    case eq.refl _ =>
      linarith
    case :=
      exfalso
      have := Nat.le_of_dvd (Nat.pos_of_ne_zero (λ H, hlt (by rwa [H]))).lt hlt this
      linarith


end least_trees_required_l360_360342


namespace ratio_of_areas_eq_l360_360315

open Real

variables (a α : ℝ) (hα : 0 < α ∧ α < π / 3)
variables (A B C D : Type) [affine_space ℝ A B C D]

def side_length_eq (ABC : triangle A B C) : Prop :=
  dist A B = a ∧ dist B C = a ∧ dist C A = a

def angle_eq_alpha (L : line A D) (AC : line A C) : Prop :=
  angle L AC = α

def ratios_of_areas (ABC : triangle A B C) (ADC : triangle A D C) : ℝ :=
  area ADC / area ABC

theorem ratio_of_areas_eq (ABC : triangle A B C) (L : line A D) (AC : line A C)
  (h1 : side_length_eq ABC)
  (h2 : angle_eq_alpha L AC)
  : ratios_of_areas ABC (triangle.mk A D C) = sin (2 * α) / sqrt 3 := sorry

end ratio_of_areas_eq_l360_360315


namespace dot_product_magnitude_l360_360160

open Real

variables (a b : ℝ^3) 

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross (u v : ℝ^3) : ℝ^3 := (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)
def dot (u v : ℝ^3) : ℝ := u.1*v.1 + u.2*v.2 + u.3*v.3

axiom norm_a : norm a = 3
axiom norm_b : norm b = 4
axiom cross_norm : norm (cross a b) = 6

theorem dot_product_magnitude : abs (dot a b) = 6 * sqrt 3 := by
  sorry

end dot_product_magnitude_l360_360160


namespace points_on_equation_correct_l360_360945

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l360_360945


namespace largest_prime_factor_of_expression_l360_360004

noncomputable def expr := 12^3 + 15^4 - 6^5

theorem largest_prime_factor_of_expression :
  ∃ p, prime p ∧ p ∣ expr ∧ (∀ q, prime q ∧ q ∣ expr → q ≤ p) ∧ p = 12193 := 
sorry

end largest_prime_factor_of_expression_l360_360004


namespace num_valid_permutations_l360_360276

open Finset

def people := {0, 1, 2, 3, 4}

def valid_permutations : Finset (List ℕ) :=
  univ.filter (λ l : List ℕ, 
    l.head ≠ 0 ∧ 
    l.last ≠ 1 ∧ 
    l.nodup ∧
    ∀ x ∈ l, x ∈ people)

theorem num_valid_permutations : valid_permutations.card = 78 :=
by
  sorry

end num_valid_permutations_l360_360276


namespace sum_of_distances_from_point_P_to_vertices_is_160_l360_360133

variables {A B C P : Type} [MetricSpace P]
variables {a b c r : ℝ}
variables {x y : ℝ}

axiom triangle_sides (t : Triangle P) : (t.A = A) → (t.B = B) → (t.C = C) → (t.sideA = a) → (t.sideB = b) → (t.sideC = c)
axiom cos_ratio (t : Triangle P) : (cos t.A / cos t.B = 4 / 3) ∧ (b / a = 4 / 3)
axiom side_c (t : Triangle P) : c = 10

axiom incircle (t : Triangle P) : incircle t P
axiom distance_squared_sum (t : Triangle P) : d_min + d_max = 160

theorem sum_of_distances_from_point_P_to_vertices_is_160 (t : Triangle P)
  (h1 : triangle_sides t (t.A = A) (t.B = B) (t.C = C) (t.sideA = a) (t.sideB = b) (t.sideC = c))
  (h2 : cos_ratio t (cos t.A / cos t.B = 4 / 3) ∧ (b / a = 4 / 3))
  (h3 : side_c t (c = 10))
  (h4 : incircle t P) :
  distance_squared_sum t := by
  sorry

end sum_of_distances_from_point_P_to_vertices_is_160_l360_360133


namespace exists_large_cube_construction_l360_360238

theorem exists_large_cube_construction (n : ℕ) :
  ∃ N : ℕ, ∀ n > N, ∃ k : ℕ, k^3 = n :=
sorry

end exists_large_cube_construction_l360_360238


namespace average_visitors_per_day_in_month_l360_360780

theorem average_visitors_per_day_in_month (avg_visitors_sunday : ℕ) (avg_visitors_other_days : ℕ) (days_in_month : ℕ) (starts_sunday : Bool) :
  avg_visitors_sunday = 140 → avg_visitors_other_days = 80 → days_in_month = 30 → starts_sunday = true → 
  (∀ avg_visitors, avg_visitors = (4 * avg_visitors_sunday + 26 * avg_visitors_other_days) / days_in_month → avg_visitors = 88) :=
by
  intros h1 h2 h3 h4
  have total_visitors : ℕ := 4 * avg_visitors_sunday + 26 * avg_visitors_other_days
  have avg := total_visitors / days_in_month
  have visitors : ℕ := 2640
  sorry

end average_visitors_per_day_in_month_l360_360780


namespace percentage_reduction_price_increase_l360_360336

open Real

-- Part 1: Finding the percentage reduction each time
theorem percentage_reduction (P₀ P₂ : ℝ) (x : ℝ) (h₀ : P₀ = 50) (h₁ : P₂ = 32) (h₂ : P₀ * (1 - x) ^ 2 = P₂) :
  x = 0.20 :=
by
  dsimp at h₀ h₁,
  rw h₀ at h₂,
  rw h₁ at h₂,
  simp at h₂,
  sorry

-- Part 2: Determining the price increase per kilogram
theorem price_increase (P y : ℝ) (profit_per_kg : ℝ) (initial_sales : ℝ) 
  (price_increase_limit : ℝ) (sales_decrease_rate : ℝ) (target_profit : ℝ)
  (h₀ : profit_per_kg = 10) (h₁ : initial_sales = 500) (h₂ : price_increase_limit = 8)
  (h₃ : sales_decrease_rate = 20) (h₄ : target_profit = 6000) (0 < y ∧ y ≤ price_increase_limit)
  (h₅ : (profit_per_kg + y) * (initial_sales - sales_decrease_rate * y) = target_profit) :
  y = 5 :=
by
  dsimp at h₀ h₁ h₂ h₃ h₄,
  rw [h₀, h₁, h₂, h₃, h₄] at h₅,
  sorry

end percentage_reduction_price_increase_l360_360336


namespace number_of_factors_l360_360621

theorem number_of_factors (a b c : ℕ) (h_a : (∃ q1 : ℕ, Nat.Prime q1 ∧ a = q1^2))
                                  (h_b : (∃ q2 : ℕ, Nat.Prime q2 ∧ b = q2^2))
                                  (h_c : (∃ q3 : ℕ, Nat.Prime q3 ∧ c = q3^2))
                                  (h_distinct: a ≠ b ∧ b ≠ c ∧ a ≠ c):
  Nat.factors_count (a^3 * b^4 * c^5) = 693 :=
by
  sorry

end number_of_factors_l360_360621


namespace space_conditions_hold_l360_360650

-- Define the conditions in the plane
axiom plane_two_lines_parallel {L1 L2 L3 : Type} (h1 : Parallel L1 L3) (h2 : Parallel L2 L3) : Parallel L1 L2
axiom plane_perpendicular_parallel {L : Type} {P1 P2 : Type} (h1 : Perpendicular L P1) (h2 : Parallel P1 P2) : Perpendicular L P2
axiom plane_two_lines_perpendicular {P1 P2 : Type} (L : Type) (h1 : Perpendicular P1 L) (h2 : Perpendicular P2 L) : Parallel P1 P2
axiom plane_line_intersect_parallel {L : Type} {P1 P2 : Type} (h1 : Intersect L P1) (h2 : Parallel P1 P2) : Intersect L P2

-- Define the properties in space
axiom space_two_lines_parallel {L1 L2 L3 : Type} (h1 : Parallel L1 L3) (h2 : Parallel L2 L3) : Parallel L1 L2
axiom space_perpendicular_parallel {L : Type} {P1 P2 : Type} (h1 : Perpendicular L P1) (h2 : Parallel P1 P2) : Perpendicular L P2

-- Statement to prove: When conditions hold in space
theorem space_conditions_hold : 
  (∀ {L1 L2 L3 : Type}, (Parallel L1 L3) → (Parallel L2 L3) → (Parallel L1 L2)) ∧ 
  (∀ {L : Type} {P1 P2 : Type}, (Perpendicular L P1) → (Parallel P1 P2) → (Perpendicular L P2)) :=
begin
  split,
  { intros L1 L2 L3 h1 h2,
    apply space_two_lines_parallel h1 h2 },
  { intros L P1 P2 h1 h2,
    apply space_perpendicular_parallel h1 h2 },
end

end space_conditions_hold_l360_360650


namespace perpendicular_line_eq_l360_360648

-- Define the condition for line passing through the point (1, 0)
def passes_through (x y : ℝ) (pt : ℝ × ℝ) :=
  let (x1, y1) := pt in (y - y1) = -(2 * -(x - x1) + 2 * 1)

-- Define the perpendicular condition
def is_perpendicular (a b c : ℝ) :=
  a * 1 + b * 2 ≠ 0

-- Define the line equation
def line_eq (a b c x y : ℝ) := a * x + b * y + c = 0

-- The theorem statement
theorem perpendicular_line_eq {x y : ℝ} {a b c : ℝ} (h : passes_through x y (1, 0)) (h1 : is_perpendicular a b c) :
  line_eq 2 1 (-2) x y := by
  sorry

end perpendicular_line_eq_l360_360648


namespace sum_S_p_l360_360015

-- Define the arithmetic sequence sum function S_p
def S_p (p : ℕ) : ℤ :=
  2500 * p - 1225

theorem sum_S_p : (finset.range 8).sum (λ p, S_p (p + 1)) = 80200 :=
by
  sorry

end sum_S_p_l360_360015


namespace gcd_154_and_90_l360_360455

theorem gcd_154_and_90 : Nat.gcd 154 90 = 2 := by
  sorry

end gcd_154_and_90_l360_360455


namespace one_greater_others_less_l360_360660

theorem one_greater_others_less {a b c : ℝ} (h1 : a > 0 ∧ b > 0 ∧ c > 0) (h2 : a * b * c = 1) (h3 : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
by
  sorry

end one_greater_others_less_l360_360660


namespace sum_of_roots_of_quadratic_eq_l360_360732

theorem sum_of_roots_of_quadratic_eq : 
  ∀ (a b c : ℝ), (x^2 - 6 * x + 8 = 0) → (a = 1 ∧ b = -6 ∧ c = 8) → -b / a = 6 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_eq_l360_360732


namespace calculation_error_l360_360120

def percentage_error (actual expected : ℚ) : ℚ :=
  (actual - expected) / expected * 100

theorem calculation_error :
  let correct_result := (5 / 3) * 3
  let incorrect_result := (5 / 3) / 3
  percentage_error incorrect_result correct_result = 88.89 := by
  sorry

end calculation_error_l360_360120


namespace age_relation_l360_360910

variable (x y z : ℕ)

theorem age_relation (h1 : x > y) : (z > y) ↔ (∃ w, w > 0 ∧ y + z > 2 * x) :=
sorry

end age_relation_l360_360910


namespace max_shapes_in_8x14_grid_l360_360690

def unit_squares := 3
def grid_8x14 := 8 * 14
def grid_points (m n : ℕ) := (m + 1) * (n + 1)
def shapes_grid_points := 8
def max_shapes (total_points shape_points : ℕ) := total_points / shape_points

theorem max_shapes_in_8x14_grid 
  (m n : ℕ) (shape_points : ℕ) 
  (h1 : m = 8) (h2 : n = 14)
  (h3 : shape_points = 8) :
  max_shapes (grid_points m n) shape_points = 16 := by
  sorry

end max_shapes_in_8x14_grid_l360_360690


namespace common_ratio_of_geometric_seq_l360_360643

variable {α : Type} [LinearOrderedField α] 
variables (a d : α) (h₁ : d ≠ 0) (h₂ : (a + 2 * d) / (a + d) = (a + 5 * d) / (a + 2 * d))

theorem common_ratio_of_geometric_seq : (a + 2 * d) / (a + d) = 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l360_360643


namespace minimal_m_l360_360228

def fib : Nat → Nat
| 0     => 0
| 1     => 1
| n + 2 => fib (n + 1) + fib n

theorem minimal_m (m : Nat) (xs : Fin m → Nat) :
  (∀ n < 2019, ∃ (s : Finset (Fin m)), s.sum (λ i, xs i) = fib n) ↔ m = 1009 :=
sorry

end minimal_m_l360_360228


namespace repeating_decimal_sum_l360_360400

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l360_360400


namespace square_root_approximation_l360_360378

theorem square_root_approximation :
  let n : ℝ := (10^100 - 1) / 10^100 in
  abs (sqrt n - (1 - 0.5 * 10^(-100) - 0.125 * 10^(-200))) ≤ 10^(-301) :=
by
  -- The proof is left as an exercise.
  sorry

end square_root_approximation_l360_360378


namespace fourth_year_area_l360_360321

-- Define the initial conditions
def initial_area : ℕ := 10000
def annual_increase : ℤ := 20

-- Define the function for afforested area in the nth year
def afforested_area (n : ℕ) : ℕ :=
  initial_area * (1 + annual_increase / 100) ^ n

-- State the theorem
theorem fourth_year_area : afforested_area 3 = 17280 := by sorry

end fourth_year_area_l360_360321


namespace find_width_l360_360612

theorem find_width (A : ℕ) (hA : A ≥ 120) (w : ℕ) (l : ℕ) (hl : l = w + 20) (h_area : w * l = A) : w = 4 :=
by sorry

end find_width_l360_360612


namespace expression_evaluation_l360_360810

theorem expression_evaluation :
  (-2: ℤ)^3 + ((36: ℚ) / (3: ℚ)^2 * (-1 / 2: ℚ)) + abs (-5: ℤ) = -5 :=
by
  sorry

end expression_evaluation_l360_360810


namespace clock_angle_at_seven_l360_360696

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l360_360696


namespace prime_minister_stays_l360_360762

-- Definitions based on conditions
def paper_content (p : Set String) : Prop :=
  p = {"Stay", "Leave"} ∨ p = {"Leave", "Leave"}

def prime_minister_action : Prop :=
  ∃ p, p ∈ ({"Stay", "Leave"} : Set String) → (({"Stay", "Leave"}).erase p = {"Stay"} ∨ {"Stay"})

-- Theorem based on the correct answer
theorem prime_minister_stays (p : Set String) (h : paper_content p) :
  prime_minister_action → (∃ q, q ∈ p ∧ (p.erase q = {"Stay"} ∨ p.erase q = {"Leave"}))
  sorry

end prime_minister_stays_l360_360762


namespace repeating_decimal_sum_in_lowest_terms_l360_360436

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l360_360436


namespace abs_dot_product_l360_360170

variables (a b : ℝ^3)

-- Conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_cross_ab : ℝ := 6

-- Theorem statement
theorem abs_dot_product (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (hcross : ‖a × b‖ = norm_cross_ab) :
  |(a ⬝ b)| = 6 * sqrt 3 :=
by
  sorry

end abs_dot_product_l360_360170


namespace max_min_values_of_y_l360_360841

noncomputable def y (x : ℝ) : ℝ := - (Real.cos x)^2 + sqrt 3 * (Real.cos x) + 5/4

theorem max_min_values_of_y :
  (∀ x : ℝ, y x ≤ 2) ∧ (∃ x : ℝ, y x = 2) ∧ (∀ x : ℝ, y x ≥ (1/4 - sqrt 3)) ∧ (∃ x : ℝ, y x = (1/4 - sqrt 3)) :=
by sorry

end max_min_values_of_y_l360_360841


namespace g_expression_g_increasing_l360_360025

noncomputable def f (x : ℝ) := Real.sin x

noncomputable def g (x : ℝ) := 2 * Real.sin (x / 2 - Real.pi / 6)

theorem g_expression : 
  ∀ x, g(x) = 2 * Real.sin (x / 2 - Real.pi / 6) := 
sorry

theorem g_increasing : 
  ∀ k : ℤ, ∀ x : ℝ, 
  (2 * k * Real.pi - Real.pi / 3) ≤ x ∧ x ≤ (2 * k * Real.pi + Real.pi / 3) → 
  ∀ x, Monotone (g) := 
sorry

end g_expression_g_increasing_l360_360025


namespace largest_common_value_less_than_500_for_sequences_l360_360640

-- Define the first arithmetic progression
def sequence1 (n : ℕ) : ℕ := 2 + 3 * n

-- Define the second arithmetic progression
def sequence2 (m : ℕ) : ℕ := 3 + 7 * m

-- Statement to prove the largest common value less than 500
theorem largest_common_value_less_than_500_for_sequences :
  ∃ (a : ℕ), a < 500 ∧ (∃ n, a = sequence1 n) ∧ (∃ m, a = sequence2 m) ∧ a = 479 :=
by
  sorry

end largest_common_value_less_than_500_for_sequences_l360_360640


namespace count_possible_r_values_l360_360657

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

def is_four_place_decimal (r : ℚ) : Prop :=
  ∃ a b c d : ℕ, is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ 
    r = a / 10 + b / 10^2 + c / 10^3 + d / 10^4

def is_closest_to_3_over_11 (r : ℚ) : Prop :=
  let three_over_eleven := 3 / 11 
  ∧ abs (r - three_over_eleven) < abs (r - (1 / 4))
  ∧ abs (r - three_over_eleven) < abs (r - (1 / 3))
  ∧ abs (r - three_over_eleven) < abs (r - (3 / 12))
  ∧ abs (r - three_over_eleven) < abs (r - (3 / 10))

theorem count_possible_r_values :
  let three_over_eleven := 3 / 11 in
  ∃ (n : ℕ), n = 417 ∧
  (finset.filter (λ r : ℚ, is_four_place_decimal r ∧ is_closest_to_3_over_11 r)
  (finset.range 10000).val).card = n := sorry

end count_possible_r_values_l360_360657


namespace repeating_decimals_sum_l360_360430

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l360_360430


namespace series_converges_to_one_l360_360879

noncomputable def K (n : ℕ) : ℝ :=
  1 + (finset.range n).sum (λ k, 1 / (k + 1)^2)

noncomputable def series_sum : ℝ :=
  ∑' n, 1 / ((n + 1) * K n * K (n + 1))

theorem series_converges_to_one :
  series_sum = 1 :=
by
  sorry

end series_converges_to_one_l360_360879


namespace min_value_of_f_at_extreme_point_l360_360096

def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * x - 3 * Real.log x

theorem min_value_of_f_at_extreme_point :
  (∀ x : ℝ, x ≠ 0 → f x 2 = (1/2) * x^2 - 2 * x - 3 * Real.log x ∧ (∃ x : ℝ, (x ≠ 0) ∧ f x 2 = 0) → x = 3) →
  f 3 2 = - (3 / 2) - 3 * Real.log 3 :=
by
  sorry

end min_value_of_f_at_extreme_point_l360_360096


namespace clock_angle_at_7_oclock_l360_360706

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l360_360706


namespace dot_product_magnitude_l360_360177

variables {a b : EuclideanSpace 3 ℝ}

/- Given conditions -/
def norm_a : ℝ := ‖a‖ = 3
def norm_b : ℝ := ‖b‖ = 4
def norm_cross : ℝ := ‖a × b‖ = 6

/- Desired property to prove -/
theorem dot_product_magnitude :
  norm_a →
  norm_b →
  norm_cross →
  |(a ⋅ b)| = 6 * real.sqrt 3 :=
by
  sorry

end dot_product_magnitude_l360_360177


namespace age_condition_l360_360912

theorem age_condition (x y z : ℕ) (h1 : x > y) : 
  (z > y) ↔ (y + z > 2 * x) ∧ (∀ x y z, y + z > 2 * x → z > y) := sorry

end age_condition_l360_360912


namespace find_XY_in_triangle_l360_360449

noncomputable def triangle_XYZ_side_length : ℝ :=
  let Z : Type := 0
  let angle_X : ℝ := 60
  let ZY : ℝ := 24
  let XY : ℝ := ZY / 2
  in XY

theorem find_XY_in_triangle : triangle_XYZ_side_length = 12 :=
by
  sorry

end find_XY_in_triangle_l360_360449


namespace general_term_is_sum_of_b_n_l360_360896

-- Given: Arithmetic sequence with nonzero common difference
variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Definitions for conditions
def arithmetic_seq (d : ℤ) := ∃ a1 : ℤ, ∀ n, a n = a1 + (n - 1) * d
def sum_of_first_n (d : ℤ) (h : arithmetic_seq d) : ℕ → ℤ 
| 0      := 0
| (n+1)  := sum_of_first_n n + a (n+1)

def geometric_prog (x y z : ℤ) := y * y = x * z

-- Conditions given
axiom a1 (d : ℤ) (h : arithmetic_seq d) : sum_of_first_n d h 7 = 70
axiom a2 (d : ℤ) (h : arithmetic_seq d) : geometric_prog (a 1) (a 2) (a 6)

def S (d : ℤ) := sum_of_first_n d (sorry)

-- To be proved: 1. General term, 2. Sum of sequence b_n
theorem general_term_is (d : ℤ) (h : arithmetic_seq d) : d ≠ 0 → a = λ n, 3 * n - 2 :=
sorry

variables {b : ℕ → ℤ}
def b_n (S : ℕ → ℤ) (n : ℕ) : ℤ := 3 / (2 * S n + 4 * n)

theorem sum_of_b_n (d : ℤ) (h : arithmetic_seq d) :
  ∃ (T : ℕ → ℤ), (∀ n, T n = (b_n (sum_of_first_n d h) n)) → T = λ n, n / (n + 1) :=
sorry

end general_term_is_sum_of_b_n_l360_360896


namespace smallest_N_l360_360578

-- Definitions based on the problem statement
def is_multiple_of_3 (N : ℕ) : Prop := ∃ k : ℕ, N = 3 * k

def Q (N : ℕ) : ℚ :=
  let favorable_positions := (Nat.floor (3 / 7 * N) + 1) + (N - Nat.ceil (4 / 7 * N) + 1)
  favorable_positions / (N + 1)

def condition (N : ℕ) : Prop := Q(N) < (1 : ℚ) / 2

-- Statement of the proof problem
theorem smallest_N : ∃ N : ℕ, is_multiple_of_3 N ∧ condition N ∧ ∀ M : ℕ, is_multiple_of_3 M ∧ condition M → N ≤ M :=
begin
  sorry
end

end smallest_N_l360_360578


namespace ones_digit_of_largest_power_of_two_dividing_factorial_l360_360864

theorem ones_digit_of_largest_power_of_two_dividing_factorial (n : ℕ) :
  (n = 5) → (nat.digits 10 (2 ^ (31))) = [8] :=
by
  intro h
  rw h
  have fact: nat.fact (2 ^ n) = 32!
  { simp [nat.fact_pow, mul_comm] }
  sorry

end ones_digit_of_largest_power_of_two_dividing_factorial_l360_360864


namespace truncated_tetrahedron_area_relation_l360_360251

theorem truncated_tetrahedron_area_relation
  (A₁ A₂ P : ℝ)
  (h1: A₂ ≤ A₁)
  (h2: ∃ r₁ r₂ : ℝ, ∀ A: ℝ, 
        A = sqrt (A₁ * A₂) → 
        P = (sqrt A₁ + sqrt A₂) * (sqrt (sqrt A₁) + sqrt (sqrt A₂))^2): 
  P = (sqrt A₁ + sqrt A₂) * (sqrt (sqrt A₁) + sqrt (sqrt A₂))^2 :=
by
  sorry

end truncated_tetrahedron_area_relation_l360_360251


namespace two_digit_number_reverse_sum_eq_99_l360_360647

theorem two_digit_number_reverse_sum_eq_99 :
  ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ ((10 * a + b) - (10 * b + a) = 5 * (a + b))
  ∧ (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_number_reverse_sum_eq_99_l360_360647


namespace four_digit_not_multiples_of_4_or_9_l360_360086

theorem four_digit_not_multiples_of_4_or_9 (h1 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 4 ∣ n ↔ (250 ≤ n / 4 ∧ n / 4 ≤ 2499))
                                         (h2 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 9 ∣ n ↔ (112 ≤ n / 9 ∧ n / 9 ≤ 1111))
                                         (h3 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 36 ∣ n ↔ (28 ≤ n / 36 ∧ n / 36 ≤ 277)) :
                                         (9000 - ((2250 : ℕ) + 1000 - 250)) = 6000 :=
by sorry

end four_digit_not_multiples_of_4_or_9_l360_360086


namespace ones_digit_of_largest_power_of_two_dividing_factorial_l360_360865

theorem ones_digit_of_largest_power_of_two_dividing_factorial (n : ℕ) :
  (n = 5) → (nat.digits 10 (2 ^ (31))) = [8] :=
by
  intro h
  rw h
  have fact: nat.fact (2 ^ n) = 32!
  { simp [nat.fact_pow, mul_comm] }
  sorry

end ones_digit_of_largest_power_of_two_dividing_factorial_l360_360865


namespace marble_count_l360_360989

theorem marble_count (p y v : ℝ) (h1 : y + v = 10) (h2 : p + v = 12) (h3 : p + y = 5) :
  p + y + v = 13.5 :=
sorry

end marble_count_l360_360989


namespace Delta_x_harmonic_of_harmonic_Delta_y_harmonic_of_harmonic_l360_360587

def harmonic (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ),
    (∂ f (x, y) / ∂ x ^ 2) + (∂ f (x, y) / ∂ y ^ 2) = 0

def Δx (f : ℝ × ℝ → ℝ) (x y : ℝ) : ℝ :=
  f (x + 1, y) - f (x, y)

def Δy (f : ℝ × ℝ → ℝ) (x y : ℝ) : ℝ :=
  f (x, y + 1) - f (x, y)

theorem Delta_x_harmonic_of_harmonic (f : ℝ × ℝ → ℝ) (h : harmonic f) :
  harmonic (λ p, Δx f p.1 p.2) :=
sorry

theorem Delta_y_harmonic_of_harmonic (f : ℝ × ℝ → ℝ) (h : harmonic f) :
  harmonic (λ p, Δy f p.1 p.2) :=
sorry

end Delta_x_harmonic_of_harmonic_Delta_y_harmonic_of_harmonic_l360_360587


namespace sum_of_coordinates_F_l360_360131

-- Definitions for the points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 8⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨10, 2⟩

-- Definitions for the midpoints D and E
def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def D : Point := midpoint A B
def E : Point := midpoint B C

-- Line definitions
def line (p1 p2 : Point) : ℝ → ℝ := 
  λ x, p1.y + (p2.y - p1.y) / (p2.x - p1.x) * (x - p1.x)

def AE : ℝ → ℝ := line A E
def CD : ℝ → ℝ := line C D

-- Find the intersection of AE and CD
noncomputable def F : Point :=
  let x := sorry in  -- x-coordinate of the intersection to be solved for
  ⟨x, AE x⟩

-- Proof statement
theorem sum_of_coordinates_F : (F.x + F.y) = 13 :=
  sorry

end sum_of_coordinates_F_l360_360131


namespace ones_digit_of_largest_power_of_2_in_20_fact_l360_360866

open Nat

def largest_power_of_2_in_factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else
    let sum_of_powers := ∑ m in range (n+1), m / 2
    sum_of_powers

def ones_digit_of_power_of_2 (exp : ℕ) : ℕ :=
  let cycle := [2, 4, 8, 6]
  cycle[exp % 4]

theorem ones_digit_of_largest_power_of_2_in_20_fact (n : ℕ) (h : n = 20) : 
  ones_digit_of_power_of_2 (largest_power_of_2_in_factorial n) = 4 :=
by
  rw [h]
  have : largest_power_of_2_in_factorial 20 = 18 := by
    -- Insert the calculations for largest_power_of_2_in_factorial here
    sorry
  rw [this]
  have : ones_digit_of_power_of_2 18 = 4 := by
    -- Insert the cycle calculations here
    sorry
  exact this

end ones_digit_of_largest_power_of_2_in_20_fact_l360_360866


namespace angle_quadrant_l360_360529

-- Problem statement
theorem angle_quadrant (k : ℤ) : 
  let α := k * 180 + 45 in 
  (0 < α % 360 ∧ α % 360 < 90) ∨ (180 < α % 360 ∧ α % 360 < 270) :=
by 
  sorry

end angle_quadrant_l360_360529


namespace solve_equation_l360_360941

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l360_360941


namespace problem_statement_l360_360897

noncomputable def α : ℝ :=
  let P := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  in if P = (sqrt 3 / 2, -1 / 2) then 11 * Real.pi / 6 else 0

theorem problem_statement :
  0 ≤ α ∧ α < 2 * Real.pi ∧ (Real.sin α = sqrt 3 / 2) ∧ (Real.cos α = -1 / 2) → 
  α = 11 * Real.pi / 6 :=
sorry

end problem_statement_l360_360897


namespace mean_of_integers_neg6_to_6_l360_360294

-- Define the set of integers from -6 to 6
def integers_from_neg6_to_6 := set.range (λ n : ℤ, n) ∩ set.Icc (-6 : ℤ) 6

-- Define the arithmetic mean of a finite set of integers
def arithmetic_mean (s : set ℤ) : ℚ :=
  (s.to_finset.sum id) / (s.to_finset.card : ℚ)

-- The main theorem to state the proof problem
theorem mean_of_integers_neg6_to_6 : 
  arithmetic_mean integers_from_neg6_to_6 = 0.0 :=
by
  sorry

end mean_of_integers_neg6_to_6_l360_360294


namespace num_true_propositions_is_1_l360_360922

-- Define the propositions as booleans
def proposition_1 : Bool := false -- (1) Both tosses being heads and both being tails are not complementary
def proposition_2 : Bool := true  -- (2) Both tosses being heads and both being tails are mutually exclusive
def proposition_3 : Bool := false -- (3) "At most 2 defective" and "At least 2 defective" overlap

-- Counting the number of true propositions
def count_true_propositions : Nat :=
  [proposition_1, proposition_2, proposition_3].count id

theorem num_true_propositions_is_1 : count_true_propositions = 1 :=
  by
  sorry

end num_true_propositions_is_1_l360_360922


namespace number_of_ordered_pairs_l360_360523

noncomputable theory

open BigOperators

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem number_of_ordered_pairs (h : ∀ a b : ℕ, a > 0 ∧ b > 0 → a * b + 35 = 22 * lcm a b + 14 * gcd a b) :
  ∃ p, ({(a, b) : ℕ × ℕ | 1 ≤ a ∧ 1 ≤ b ∧ a * b + 35 = 22 * lcm a b + 14 * gcd a b}.size = p ∧ p = 4) := 
sorry

end number_of_ordered_pairs_l360_360523


namespace no_green_ball_in_bag_l360_360119

theorem no_green_ball_in_bag (bag : Set String) (h : bag = {"red", "yellow", "white"}): ¬ ("green" ∈ bag) :=
by
  sorry

end no_green_ball_in_bag_l360_360119


namespace dot_product_magnitude_l360_360186

variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜 → 𝕜} 

-- Define the conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_a_cross_b : ℝ := 6

-- Theorem statement
theorem dot_product_magnitude :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∥a × b∥ = norm_a_cross_b) → abs ((a.toReal).dot (b.toReal)) = 6 * sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end dot_product_magnitude_l360_360186


namespace binom_two_eq_l360_360290

-- Define the binomial coefficient for general k
def binom (n k : ℕ) : ℕ := n.choose k

-- Theorem statement
theorem binom_two_eq (n : ℕ) (h : n ≥ 2) : binom n 2 = n * (n - 1) / 2 :=
by {
  rw [binom, nat.choose_eq_factorial_div_factorial n 2],
  -- ... remaining proof steps ...
  sorry -- will replace with actual proof
}

end binom_two_eq_l360_360290


namespace susie_bob_ratio_l360_360787

theorem susie_bob_ratio (slices_per_small : ℕ)
  (slices_per_large : ℕ)
  (small_pizzas_purchased : ℕ)
  (large_pizzas_purchased : ℕ)
  (george_pieces : ℕ)
  (bob_extra_pieces : ℕ)
  (bill_pieces : ℕ)
  (fred_pieces : ℕ)
  (mark_pieces : ℕ)
  (total_pieces_leftover : ℕ) :
  slices_per_small = 4 →
  slices_per_large = 8 →
  small_pizzas_purchased = 3 →
  large_pizzas_purchased = 2 →
  george_pieces = 3 →
  bob_extra_pieces = 1 →
  bill_pieces = 3 →
  fred_pieces = 3 →
  mark_pieces = 3 →
  total_pieces_leftover = 10 →
  let total_slices := (small_pizzas_purchased * slices_per_small) + (large_pizzas_purchased * slices_per_large) in
  let george_total := george_pieces in
  let bob_total := george_pieces + bob_extra_pieces in
  let others_total := bill_pieces + fred_pieces + mark_pieces in
  let total_eaten_slices := total_slices - total_pieces_leftover in
  let susie_pieces := total_eaten_slices - (george_total + bob_total + others_total) in
  susie_pieces / bob_total = 1 / 2 := 
by
  intros;
  sorry

end susie_bob_ratio_l360_360787


namespace gcd_324_243_135_l360_360001

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 := by
  sorry

end gcd_324_243_135_l360_360001


namespace solve_expression_l360_360616

theorem solve_expression :
  2^3 + 2 * 5 - 3 + 6 = 21 :=
by
  sorry

end solve_expression_l360_360616


namespace edward_end_money_l360_360317

theorem edward_end_money : 
  (spring_earnings summer_earnings supplies_expense final_amount : ℕ) 
  (h1 : spring_earnings = 2) 
  (h2 : summer_earnings = 27) 
  (h3 : supplies_expense = 5) 
  (h4 : final_amount = spring_earnings + summer_earnings - supplies_expense) : 
  final_amount = 24 := 
by 
  sorry

end edward_end_money_l360_360317


namespace pentagon_process_termination_l360_360369

theorem pentagon_process_termination
  (x1 x2 x3 x4 x5 : ℤ)
  (H_sum_pos : x1 + x2 + x3 + x4 + x5 > 0)
  (H_op : ∀ i, x_i < 0 → ∃ n, repeat_operation n (x1, x2, x3, x4, x5) = y)
  : ∃ N, ∀ n ≥ N, all_nonneg (repeat_op n (x1, x2, x3, x4, x5)) := sorry

end pentagon_process_termination_l360_360369


namespace repeating_decimal_sum_l360_360397

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l360_360397


namespace prop1_prop2_l360_360064

-- Given the function f(x) = (x-a)[x^2 - (b-1)x - b]
def f (x a b : ℝ) : ℝ := (x - a) * (x^2 - (b - 1) * x - b)

-- Proposition 1: When a = b, f(x) = (x-1)^2(x+1)
theorem prop1 (a b : ℝ) (h : a = b) : f x a b = (x - 1) ^ 2 * (x + 1) :=
by
  sorry

-- Proposition 2: When f(x) has three zeros, the maximum value of f(x) is 2sqrt(3)/9
theorem prop2 : ∃ x : ℝ, f x 1 1 = x * (x - 1) * (x + 1) ∧ is_max 2.sqrt(3) / 9 :=
by
  sorry

end prop1_prop2_l360_360064


namespace abs_dot_product_l360_360168

variables (a b : ℝ^3)

-- Conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_cross_ab : ℝ := 6

-- Theorem statement
theorem abs_dot_product (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (hcross : ‖a × b‖ = norm_cross_ab) :
  |(a ⬝ b)| = 6 * sqrt 3 :=
by
  sorry

end abs_dot_product_l360_360168


namespace find_a_given_x_l360_360980

theorem find_a_given_x : ∀ (a : ℝ), (∀ (x : ℝ), x - 2 * a + 5 = 0 → x = -2) → a = 3 / 2 :=
by
  intros a h
  apply h
  sorry

end find_a_given_x_l360_360980


namespace clock_angle_at_7_oclock_l360_360707

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l360_360707


namespace hyperbola_eccentricity_l360_360494

open Real

variables {a b c : ℝ} (F1 F2 P Q : Point)
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def isosceles_triangle (F2 P F1 : Point) : Prop := F2.dist P = F2.dist F1
def angle_bisector_ratio (F1 Q F2 : Point) : Prop := F1.dist Q / Q.dist F2 = 3 / 2
def eccentricity (c a : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity 
  (h1 : 0 < a) 
  (h2 : 0 < b)
  (h3 : hyperbola P.x P.y a b)
  (h4 : isosceles_triangle F2 P F1)
  (h5 : angle_bisector_ratio F1 Q F2)
  : eccentricity c a = 2 := 
sorry

end hyperbola_eccentricity_l360_360494


namespace radius_of_circle_with_area_64pi_l360_360632

def circle_radius (A : ℝ) : ℝ := 
  real.sqrt (A / real.pi)

theorem radius_of_circle_with_area_64pi :
  circle_radius (64 * real.pi) = 8 :=
by sorry

end radius_of_circle_with_area_64pi_l360_360632


namespace range_of_m_l360_360146

noncomputable def G (x m : ℝ) : ℝ := (8 * x^2 + 24 * x + 5 * m) / 8

theorem range_of_m (G_is_square : ∃ c d, ∀ x, G x m = (c * x + d) ^ 2) : 3 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l360_360146


namespace apple_count_difference_l360_360678

theorem apple_count_difference
    (original_green : ℕ)
    (additional_green : ℕ)
    (red_more_than_green : ℕ)
    (green_now : ℕ := original_green + additional_green)
    (red_now : ℕ := original_green + red_more_than_green)
    (difference : ℕ := green_now - red_now)
    (h_original_green : original_green = 32)
    (h_additional_green : additional_green = 340)
    (h_red_more_than_green : red_more_than_green = 200) :
    difference = 140 :=
by
  sorry

end apple_count_difference_l360_360678


namespace paint_cost_l360_360611

theorem paint_cost
  (base1 base2 h : ℝ) 
  (tri_side : ℝ) 
  (paint_coverage : ℝ) 
  (paint_cost_per_gallon : ℝ)
  (two_trap_area : ℝ := 2 * (1 / 2 * (base1 + base2) * h))
  (triangle_area : ℝ := (real.sqrt 3 / 4) * tri_side^2)
  (total_area : ℝ := two_trap_area + triangle_area)
  (gallons_needed : ℝ := (total_area / paint_coverage).ceil) :
  base1 = 8 →
  base2 = 4 →
  h = 5 →
  tri_side = 6 →
  paint_coverage = 100 →
  paint_cost_per_gallon = 15 →
  gallons_needed = 1 →
  (gallons_needed * paint_cost_per_gallon) = 15 :=
by 
  intros,
  /- Proof would go here -/
  sorry

end paint_cost_l360_360611


namespace find_L_l360_360446

noncomputable def L_value : ℕ := 3

theorem find_L
  (a b : ℕ)
  (cows : ℕ := 5 * b)
  (chickens : ℕ := 5 * a + 7)
  (insects : ℕ := b ^ (a - 5))
  (legs_cows : ℕ := 4 * cows)
  (legs_chickens : ℕ := 2 * chickens)
  (legs_insects : ℕ :=  6 * insects)
  (total_legs : ℕ := legs_cows + legs_chickens + legs_insects) 
  (h1 : cows = insects)
  (h2 : total_legs = (L_value * 100 + L_value * 10 + L_value) + 1) :
  L_value = 3 := sorry

end find_L_l360_360446


namespace power_function_half_value_l360_360533

theorem power_function_half_value :
  ∃ α : ℝ, (f : ℝ → ℝ) = (λ x, x^α) ∧ (f 4 / f 2 = 3) → f (1/2) = 1/3 :=
by
  sorry

end power_function_half_value_l360_360533


namespace equation_solution_l360_360934

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l360_360934


namespace prob_student_A_consecutive_days_l360_360623

/--
There are 4 students, and each student is assigned to participate in a 5-day volunteer activity.
- Student A participates for exactly 2 days.
- Each of the other 3 students participates for exactly 1 day.

Prove that the probability of student A participating in two consecutive days is 2/5.
-/
theorem prob_student_A_consecutive_days :
  let total_events := Nat.choose 5 2 * nat.perm 3 3,
      favorable_events := 4 * nat.perm 3 3
  in (favorable_events / total_events : ℚ) = 2 / 5 :=
by
  sorry

end prob_student_A_consecutive_days_l360_360623


namespace ones_digit_of_largest_power_of_two_dividing_factorial_l360_360863

theorem ones_digit_of_largest_power_of_two_dividing_factorial (n : ℕ) :
  (n = 5) → (nat.digits 10 (2 ^ (31))) = [8] :=
by
  intro h
  rw h
  have fact: nat.fact (2 ^ n) = 32!
  { simp [nat.fact_pow, mul_comm] }
  sorry

end ones_digit_of_largest_power_of_two_dividing_factorial_l360_360863


namespace general_term_formula_l360_360502

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Given condition: the sum of the first n terms of sequence a is S_n = n^2 + n
def S_n_def (n : ℕ) : Prop :=
  S n = n^2 + n

-- Definition of the general term a_n based on the sum S_n
noncomputable def a_n_def (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

-- We want to prove that a_n = 2n
theorem general_term_formula (n : ℕ) (h : S_n_def S) : a_n_def S n = 2n :=
  sorry

end general_term_formula_l360_360502


namespace number_of_elements_leq_correct_l360_360010

def number_of_elements_leq (s : list ℝ) (x : ℝ) : ℤ :=
  s.countp (λ y => y ≤ x)

theorem number_of_elements_leq_correct : 
  number_of_elements_leq [0.8, 0.5, 0.3] 0.4 = 1 :=
by 
  sorry

end number_of_elements_leq_correct_l360_360010


namespace analogical_reasoning_correct_l360_360747

theorem analogical_reasoning_correct :
  (∀ (a x y : ℝ), (log a (x + y) ≠ log a x + log a y)) ∧ -- Logarithmic properties
  (∀ (x y : ℝ), (sin (x + y) ≠ sin x + sin y)) ∧        -- Sine addition formula
  (∀ (x y : ℝ) (n : ℕ), ((x + y)^n ≠ x^n + y^n)) ∧     -- Binomial theorem
  (∀ (x y z : ℝ), ((x * y) * z = x * (y * z)))         -- Associative property of multiplication
  →
  ∃ (x y z : ℝ), ((x * y) * z = x * (y * z)) :=        -- Correct reasoning is D
by
  sorry

end analogical_reasoning_correct_l360_360747


namespace repeating_decimal_sum_l360_360407

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l360_360407


namespace smallest_n_condition_l360_360461

theorem smallest_n_condition :
  ∃ (n : ℕ) (a : Fin n → ℝ),
    (∑ i, a i > 0) ∧
    (∑ i, (a i)^3 < 0) ∧
    (∑ i, (a i)^5 > 0) ∧
    (∀ (m : ℕ) (b : Fin m → ℝ),
      (m < n) →
      (∑ i, b i > 0) →
      (∑ i, (b i)^3 < 0) →
      (∑ i, (b i)^5 > 0) →
      False) := by
  sorry

end smallest_n_condition_l360_360461


namespace difference_of_squares_l360_360311

theorem difference_of_squares (x y : ℝ) (h₁ : x + y = 20) (h₂ : x - y = 10) : x^2 - y^2 = 200 :=
by {
  sorry
}

end difference_of_squares_l360_360311


namespace shaded_area_correct_l360_360645

-- Define the isosceles triangle with base and height
def isosceles_triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

-- Define the conditions given in the problem
def base := 21
def height := 28

-- Calculated areas of triangle and square
def triangle_area := isosceles_triangle_area base height
def square_side := 12
def square_area := square_side * square_side

-- The shaded region's area is the difference between the triangle's area and the square's area
def shaded_area := triangle_area - square_area

-- The theorem to prove
theorem shaded_area_correct : shaded_area = 150 := by
  -- We have given conditions and calculated expected result 
  sorry

end shaded_area_correct_l360_360645


namespace balance_3_proof_l360_360269

def weight_balances (Δ ◯ □ : ℕ) :=
  3 * Δ + ◯ = 6 * □ ∧
  2 * Δ + 4 * ◯ = 8 * □

def balance_3 (Δ ◯ □ : ℕ) :=
  4 * Δ + 3 * ◯

def correct_squares (□ : ℕ) :=
  □ = 10

theorem balance_3_proof (Δ ◯ □ : ℕ) (h : weight_balances Δ ◯ □) :
  correct_squares (balance_3 Δ ◯ □ / □) :=
by
  sorry

end balance_3_proof_l360_360269


namespace smallest_square_area_l360_360692

theorem smallest_square_area (r : ℝ) (h : r = 5) : ∃ s, s ≥ 2 * r ∧ s^2 = 100 :=
by {
  use 10, sorry,
}

end smallest_square_area_l360_360692


namespace necessary_but_not_sufficient_l360_360318

-- Conditions 
variable {x : Real}
def condition1 : Prop := x > 0
def condition2 : Prop := abs (x - 1) < 1

-- Question
theorem necessary_but_not_sufficient :
  (condition1 → condition2) ∧ (condition2 → condition1) := by
  sorry

end necessary_but_not_sufficient_l360_360318


namespace range_of_a_for_quadratic_inequality_l360_360075

theorem range_of_a_for_quadratic_inequality (a : ℝ) :
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1 > 0) →
  (a ≤ -2 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_for_quadratic_inequality_l360_360075


namespace effective_loss_percentage_l360_360353

variable (S C D T : ℝ)
variable (h1 : C = 0.99 * S)
variable (h2 : D = 0.90 * S)
variable (h3 : T = 0.045 * S)

theorem effective_loss_percentage (S : ℝ) (h1 : C = 0.99 * S) (h2 : D = 0.90 * S) (h3 : T = 0.045 * S) :
  let NS := D - T in
  let P := NS - C in
  let LossPercentage := (P / C) * 100 in
  LossPercentage = -13.64 := by
  sorry

end effective_loss_percentage_l360_360353


namespace artworks_collected_l360_360672

theorem artworks_collected (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) :
  students = 15 →
  artworks_per_student_per_quarter = 2 →
  quarters_per_year = 4 →
  num_years = 2 →
  (students * artworks_per_student_per_quarter * quarters_per_year * num_years) = 240 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end artworks_collected_l360_360672


namespace proof_geom_arith_seq_l360_360908

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- All the terms of the geometric sequence {a_n} are positive
axiom geom_seq_positive : ∀ n : ℕ, a n > 0

-- Geometric sequence with common ratio q
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q

-- \(a_3, \frac{1}{2}a_5, a_4\) form an arithmetic sequence
axiom arith_seq : 2 * (a 4 * q^2 * 1/2) = a 3 + a 3 * q

theorem proof_geom_arith_seq :
  q = (sqrt 5 + 1) / 2 →
  (a 3 + a 5) / (a 4 + a 6) = (sqrt 5 - 1) / 2 :=
by
  intro hq
  sorry

end proof_geom_arith_seq_l360_360908


namespace intersection_sets_l360_360519

def setA : set ℝ := {y | ∃ x : ℝ, y = x^2 - 2 * x + 3}
def setB : set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2 * x + 7}

theorem intersection_sets : setA ∩ setB = {y | 2 ≤ y ∧ y ≤ 8} :=
by sorry

end intersection_sets_l360_360519


namespace triangle_angle_ABC_l360_360997

theorem triangle_angle_ABC
  (ABD CBD ABC : ℝ) 
  (h1 : ABD = 70)
  (h2 : ABD + CBD + ABC = 200)
  (h3 : CBD = 60) : ABC = 70 := 
sorry

end triangle_angle_ABC_l360_360997


namespace proof_problem_l360_360813

noncomputable def problem_eq : Prop :=
  sqrt (1 / 2) * sqrt 8 - (sqrt 3) ^ 2 = -1

theorem proof_problem : problem_eq := by
  sorry

end proof_problem_l360_360813


namespace equal_area_division_l360_360035

structure Point := (x y : ℝ)
structure Triangle := (A B C : Point)

def divides_equal_area (A B D : Point) (line: Point → Point → Prop) : Prop :=
  sorry  -- Define the formalization of dividing the area function.

theorem equal_area_division (A B C D : Point) (T : Triangle) :
  divides_equal_area T.A T.B D (λ p1 p2, p1.x - p2.x = 0 ∨ p1.y - p2.y = 0) →
  ∃ E F G H J K : Point, True :=
sorry  -- Proof would encompass constructing all points and validating the equal area division.

end equal_area_division_l360_360035


namespace smallest_n_is_1770_l360_360459

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

def is_smallest_n (n : ℕ) : Prop :=
  n = sum_of_digits n + 1755 ∧ (∀ m : ℕ, (m < n → m ≠ sum_of_digits m + 1755))

theorem smallest_n_is_1770 : is_smallest_n 1770 :=
sorry

end smallest_n_is_1770_l360_360459


namespace planes_pass_through_same_line_l360_360224

theorem planes_pass_through_same_line
    (A B C D : Point)
    (K_n L_n M_n : ℕ → Point)
    (hKn : ∀ n, AB = n * AK_n(n))
    (hLn : ∀ n, AC = (n+1) * AL_n(n))
    (hMn : ∀ n, AD = (n+2) * AM_n(n)) :
    ∃ (l : Line), ∀ n, K_n(n) * L_n(n) * M_n(n) ∈ l :=
  sorry

end planes_pass_through_same_line_l360_360224


namespace repeating_decimal_sum_l360_360403

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l360_360403


namespace inequality_solution_l360_360824

theorem inequality_solution {x : ℝ} (h : -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2) : 
  x ∈ Set.Ioo (-2 : ℝ) (10 / 3) :=
by
  sorry

end inequality_solution_l360_360824


namespace max_length_interval_l360_360820

noncomputable def f (x : ℝ) : ℝ := min (Real.sin (2 * x + (Real.pi / 6))) (Real.cos (2 * x))

theorem max_length_interval (s t : ℝ) (h1 : ∀ x ∈ set.Icc s t, f x ∈ set.Icc (-1 : ℝ) (1 / 2)) :
  t - s = (5 * Real.pi) / 6 :=
by
  sorry

end max_length_interval_l360_360820


namespace window_savings_l360_360789

theorem window_savings :
  let price := 120
  let discount := λ n : Nat, (n / 10) * 2
  let cost := λ n : Nat, (n - discount n) * price
  let dave_needs := 9
  let doug_needs := 11
  let separate_cost := cost dave_needs + cost doug_needs
  let joint_needs := dave_needs + doug_needs
  let joint_cost := cost joint_needs
  let savings := separate_cost - joint_cost
  savings = 120 := 
 by 
  sorry

end window_savings_l360_360789


namespace cost_price_of_article_l360_360757

theorem cost_price_of_article (S P C : ℝ) (hS : S = 400) (hP : P = 25) :
  C = 320 ↔ (S = C * (1 + P / 100)) :=
by
  -- conditions
  assume hC : C = 320,
  -- replace the conditions in the terms of the theorem
  rw [hS, hP, hC],
  -- prove the equation
  sorry

end cost_price_of_article_l360_360757


namespace sum_of_roots_of_quadratic_eq_l360_360725

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l360_360725


namespace third_term_in_expansion_of_1_add_i_to_the_4th_l360_360195

-- Define the imaginary unit 'i'
def i : ℂ := complex.I

-- Statement of the theorem
theorem third_term_in_expansion_of_1_add_i_to_the_4th :
  (binomial 4 2) * (1 : ℂ) ^ 2 * i ^ 2 = -6 :=
by
  sorry

end third_term_in_expansion_of_1_add_i_to_the_4th_l360_360195


namespace length_vector_eq_three_l360_360600

theorem length_vector_eq_three (A B : ℝ) (hA : A = -1) (hB : B = 2) : |B - A| = 3 :=
by
  sorry

end length_vector_eq_three_l360_360600


namespace repeating_decimal_sum_l360_360439

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l360_360439


namespace find_fraction_number_l360_360960

theorem find_fraction_number :
  ∃ (x : ℚ), 
    (∃ n : ℕ, x = n / 12 ∧ 1 ≤ n ∧ n < 120) ∧ 
    let dec := x.num / x.denom in
    1 ≤ dec ∧ dec < 10 ∧ 
    x = 102 / 12 :=
by
  sorry

end find_fraction_number_l360_360960


namespace clock_angle_at_seven_l360_360693

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l360_360693


namespace minimum_number_of_colors_l360_360553

theorem minimum_number_of_colors (n : ℕ) (h_n : 2 ≤ n) :
  ∀ (f : (Fin n) → ℕ),
  (∀ i j : Fin n, i ≠ j → f i ≠ f j) →
  (∃ c : ℕ, c = n) :=
by sorry

end minimum_number_of_colors_l360_360553


namespace ones_digit_of_largest_power_of_two_dividing_factorial_l360_360860

theorem ones_digit_of_largest_power_of_two_dividing_factorial (n : ℕ) :
  (n = 5) → (nat.digits 10 (2 ^ (31))) = [8] :=
by
  intro h
  rw h
  have fact: nat.fact (2 ^ n) = 32!
  { simp [nat.fact_pow, mul_comm] }
  sorry

end ones_digit_of_largest_power_of_two_dividing_factorial_l360_360860


namespace matt_house_wall_height_l360_360214

noncomputable def height_of_walls_in_matt_house : ℕ :=
  let living_room_side := 40
  let bedroom_side_1 := 10
  let bedroom_side_2 := 12

  let perimeter_living_room := 4 * living_room_side
  let perimeter_living_room_3_walls := perimeter_living_room - living_room_side

  let perimeter_bedroom := 2 * (bedroom_side_1 + bedroom_side_2)

  let total_perimeter_to_paint := perimeter_living_room_3_walls + perimeter_bedroom
  let total_area_to_paint := 1640

  total_area_to_paint / total_perimeter_to_paint

theorem matt_house_wall_height :
  height_of_walls_in_matt_house = 10 := by
  sorry

end matt_house_wall_height_l360_360214


namespace chocolate_given_to_Shaina_is_correct_l360_360137

noncomputable def chocolateWeight (total: ℚ) (piles: ℕ) : ℚ := total / piles
noncomputable def chocolateGivenToShaina (eachPileWeight: ℚ) : ℚ := 2 * eachPileWeight

theorem chocolate_given_to_Shaina_is_correct :
  ∀ (total : ℚ) (piles : ℕ), total = (70 / 7) → piles = 5 → chocolateGivenToShaina (chocolateWeight total piles) = 4 := 
by
  intros total piles h_total h_piles
  -- this is a proof problem placeholder
  rw [h_total, h_piles]
  dsimp [chocolateWeight, chocolateGivenToShaina]
  have total_div_piles : (70 / 7) / 5 = 2 := by norm_num
  rw total_div_piles
  norm_num
  sorry

end chocolate_given_to_Shaina_is_correct_l360_360137


namespace certain_number_l360_360769

theorem certain_number (x y : ℝ) (h1 : 0.65 * x = 0.20 * y) (h2 : x = 210) : y = 682.5 :=
by
  sorry

end certain_number_l360_360769


namespace geometric_sequence_sum_l360_360917

theorem geometric_sequence_sum {a₁ q : ℝ} (h1 : a₁ + a₁ * q^2 = 5) 
                            (h2 : a₁ * a₁ * q^2 = 4)
                            (h3 : q > 1) :
  let S₆ := a₁ * (1 + q + q^2 + q^3 + q^4 + q^5) in
  S₆ = 63 :=
by
  sorry

end geometric_sequence_sum_l360_360917


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360846

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  let n := 32 in
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i) in 
  (2 ^ k) % 10 = 8 :=
by
  let n := 32
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i)
  
  have h1 : k = 31 := by sorry
  have h2 : (2 ^ 31) % 10 = 8 := by sorry
  
  exact h2

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360846


namespace contrapositive_of_p_is_true_l360_360252

-- Define the conditions x = 2 and y = 3
def cond1 := (x = 2)
def cond2 := (y = 3)

-- Define the proposition p
def p := (x = 2 ∧ y = 3) → (x + y = 5)

-- Define the contrapositive of p
def contrapositive_of_p := (x + y ≠ 5) → (x ≠ 2 ∨ y ≠ 3)

-- State the theorem
theorem contrapositive_of_p_is_true : contrapositive_of_p :=
by
  sorry

end contrapositive_of_p_is_true_l360_360252


namespace general_term_a_l360_360513

def f (x : ℕ) : ℝ := 1 / 2^(x - 1)

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 6 - 2 * a n + f (n - 1)

theorem general_term_a (a : ℕ → ℝ) :
  (∀ n, S n a = (6 - 2 * a n + f (n - 1))) →
  (a 1 = 6 - 2 * (a 1) + 2) →
  (∀ n ≥ 2, a n = (2 / 3) * a (n - 1) - (2 / 3) * (1 / 2^(n - 1))) →
  ∀ n, a n = (2 / 3)^n + 4 / 2^n := by
  intro hS h1 hrec
  sorry

end general_term_a_l360_360513


namespace sum_of_repeating_decimals_l360_360415

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l360_360415


namespace log_equation_1_log_equation_2_log_equation_3_l360_360240

-- Part (a)
theorem log_equation_1 (x : ℝ) (h : 0.2 * log x (1 / 32) = -0.5) : x = 4 :=
sorry

-- Part (b)
theorem log_equation_2 (x : ℝ) (h : log (x - 1) 3 = 2) : x = 1 + real.sqrt 3 :=
sorry

-- Part (c)
theorem log_equation_3 (x : ℝ) (h : log (log 3) x = 2) : x = 3 ^ real.sqrt 3 :=
sorry

end log_equation_1_log_equation_2_log_equation_3_l360_360240


namespace average_price_of_mixture_l360_360244

def average_price_per_kg (costs : List ℕ) (weights : List ℕ) := 
  let total_cost := costs.foldl (· + ·) 0
  let total_weight := weights.foldl (· + ·) 0
  total_cost / total_weight.toFloat

theorem average_price_of_mixture :
  let basmati_cost := 10 * 20
  let long_grain_cost := 5 * 25
  let short_grain_cost := 15 * 18
  let medium_grain_cost := 8 * 22
  let costs := [basmati_cost, long_grain_cost, short_grain_cost, medium_grain_cost]
  let weights := [10, 5, 15, 8]
  average_price_per_kg costs weights = 20.29 :=
by 
  let basmati_cost := 10 * 20
  let long_grain_cost := 5 * 25
  let short_grain_cost := 15 * 18
  let medium_grain_cost := 8 * 22
  let total_cost := basmati_cost + long_grain_cost + short_grain_cost + medium_grain_cost
  let total_weight := 10 + 5 + 15 + 8
  let avg_price := total_cost.toFloat / total_weight.toFloat
  -- Using float to compare results correctly
  have h : avg_price = 20.289473684210527 := by norm_num
  exact h.trans sorry

end average_price_of_mixture_l360_360244


namespace modulus_of_z_l360_360018

noncomputable def z : ℂ := ((2 - complex.I)^2) / complex.I

theorem modulus_of_z : complex.abs z = 5 := 
by {
  sorry
}

end modulus_of_z_l360_360018


namespace repeating_decimal_sum_in_lowest_terms_l360_360434

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l360_360434


namespace triangle_longest_side_l360_360654

theorem triangle_longest_side (y : ℝ) : 10 + (y + 6) + (3y + 5) = 49 → max 10 (max (y + 6) (3y + 5)) = 26 :=
by
  -- conditions
  intro h
  -- proof should be placed here
  sorry

end triangle_longest_side_l360_360654


namespace tetrahedron_altitude_inequality_l360_360557

-- Let A, B, C, D be points in 3D space forming a tetrahedron
variables {A B C D : Point}

-- Let h_a, h_b, h_c, h_d be the altitudes from vertices A, B, C, D to the opposite faces
variable {h_a h_b h_c h_d : ℝ}

-- Let the volume of the tetrahedron be V
variable {V : ℝ}

-- The theorem to be proved
theorem tetrahedron_altitude_inequality 
  (h_a_pos : 0 < h_a) (h_b_pos : 0 < h_b) (h_c_pos : 0 < h_c) (h_d_pos : 0 < h_d)
  (volume_expr : V = 1/3 * area A B C * h_d) 
  (volume_expr_b : V = 1/3 * area A B D * h_c)
  (volume_expr_c : V = 1/3 * area A C D * h_b)
  (volume_expr_d : V = 1/3 * area B C D * h_a) :
  (1 / h_a) < (1 / h_b) + (1 / h_c) + (1 / h_d) :=
sorry

end tetrahedron_altitude_inequality_l360_360557


namespace max_integer_terms_in_geometric_sequence_l360_360992

-- Definitions only corresponding to conditions found in the problem
noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : list ℝ :=
(list.range n).map (λ i => a * r ^ i)

theorem max_integer_terms_in_geometric_sequence :
  ∀ (a r : ℝ) (n : ℕ), r > 1 →
  (∀ k, k < n → (a * r ^ k) ∈ set.Icc 100 1000 → (a * r ^ k) ∈ set.Icc 100 1000 → Nat.floor (a * r ^ k) = a * r ^ k) →
  n ≤ 6 := 
begin
  intros,
  sorry -- proof not required
end

end max_integer_terms_in_geometric_sequence_l360_360992


namespace no_rearrangement_of_power_of_two_l360_360073

theorem no_rearrangement_of_power_of_two (k n : ℕ) (hk : k > 3) (hn : n > k) : 
  ∀ m : ℕ, 
    (m.toDigits = (2^k).toDigits → m ≠ 2^n) :=
by
  sorry

end no_rearrangement_of_power_of_two_l360_360073


namespace problem_l360_360508

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else real.sqrt (1 - x)

theorem problem :
  f (f (-3)) = 5 :=
by
  sorry

end problem_l360_360508


namespace false_proposition_b_l360_360359

-- Definitions from conditions
def vertical_angles_are_equal : Prop :=
  ∀ {l₁ l₂ : Line} (h : intersects l₁ l₂), vertical_angles l₁ l₂ h -> congruent_angles l₁ l₂ h

def corresponding_angles_are_equal (l₁ l₂ : Line) (t : Transversal) : Prop :=
  parallel l₁ l₂ → corresponding_angles l₁ l₂ t

def shortest_perpendicular_segment {p : Point} {l : Line} : Prop :=
  ∀ q (h : on_line q l), p ≠ q → distance p (foot p l) ≤ distance p q 

def unique_perpendicular {p : Point} (l : Line) : Prop :=
  ∃! m, perpendicular p l m

-- Proof Problem Statement
theorem false_proposition_b :
  ¬ ∀ (l₁ l₂ : Line) (t : Transversal), corresponding_angles l₁ l₂ t :=
sorry

end false_proposition_b_l360_360359


namespace frank_boxes_l360_360022

theorem frank_boxes (n m : ℕ) (h1 : n = 13) (h2 : m = 8) : n - m = 5 := by
  rw [h1, h2]
  norm_num
  sorry

end frank_boxes_l360_360022


namespace sum_f_1_to_50_l360_360507

def f (x : ℝ) := x^2 - 53*x + 196 + |x^2 - 53*x + 196|

theorem sum_f_1_to_50 : (∑ i in Finset.range 50 (λ x, f (x + 1))) = 660 := 
by
  sorry

end sum_f_1_to_50_l360_360507


namespace sum_of_repeating_decimals_l360_360416

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l360_360416


namespace root_in_interval_l360_360456

noncomputable def f (x : ℝ) := 2^x + 3*x - 7

theorem root_in_interval : 
  ∃ r, (1:ℝ) < r ∧ r < (2:ℝ) ∧ f(r) = 0 :=
by
  have h1 : f(1) = -2 := by rw [f]; norm_num
  have h2 : f(2) = 3 := by rw [f]; norm_num
  sorry

end root_in_interval_l360_360456


namespace fraction_planted_of_triangle_l360_360833

theorem fraction_planted_of_triangle
  (a b : ℕ) (htri : a = 5 ∧ b = 12)
  (h_eq_legs : ∀ (x : ℕ), x = a ∧ x = b)
  (h_distance : ∃ (d : ℕ), d = 3) :
  planted_area_fraction : ℚ := 
by
  have h_triangle_area : (a * b) / 2 = 30 := sorry
  have h_unplanted_area : (15 / 17) ^ 2 / 2 := sorry
  have h_planted_area : 30 - h_unplanted_area := sorry
  have h_fraction : h_planted_area / 30 = 2665 / 2890 := sorry
  exact h_fraction

end fraction_planted_of_triangle_l360_360833


namespace part1_part2_l360_360885

def A (x y : ℝ) : ℝ := 3 * x ^ 2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := - x ^ 2 + x * y - 1

theorem part1 (x y : ℝ) : A x y + 3 * B x y = 5 * x * y - 2 * x - 4 := by
  sorry

theorem part2 (y : ℝ) : (∀ x : ℝ, 5 * x * y - 2 * x - 4 = -4) → y = 2 / 5 := by
  sorry

end part1_part2_l360_360885


namespace angle_BAD_60_degrees_l360_360556

theorem angle_BAD_60_degrees 
  (ABCD : Type) [parallelogram ABCD] (A B C D E K T : Point)
  (AD_length : (dist A D) = 6) 
  (angle_bisector_ADC_intersect_AB_at_E : intersect (angleBisector (angle A D C)) (line A B) = E)
  (inscribed_circle_ADE_touches_AE_at_K_AD_at_T : circle K in_triangle ADE ∧ circle T in_triangle ADE)
  (KT_length : (dist K T) = 3) : angle A B D = 60 :=
sorry

end angle_BAD_60_degrees_l360_360556


namespace find_angle_sum_l360_360559

variables 
  (A B D F G : Type) 
  (x y : ℝ)
  (h1 : A ≠ B ∧ B ≠ D ∧ D ≠ F ∧ F ≠ G ∧ ∃ l : line, A ∈ l ∧ B ∈ l ∧ D ∈ l ∧ F ∈ l ∧ G ∈ l)
  (h2 : ∃ C : Type, right_angle (angle B C D))
  (h3 : ∃ E : Type, right_angle (angle D E F))
  (h4 : MeasurableAngle (angle A B C) x)
  (h5 : MeasurableAngle (angle C D E) 80)
  (h6 : MeasurableAngle (angle E F G) y)

theorem find_angle_sum : x + y = 280 :=
by
  sorry

end find_angle_sum_l360_360559


namespace not_divisible_by_1955_l360_360136

theorem not_divisible_by_1955 (n : ℤ) : ¬ ∃ k : ℤ, (n^2 + n + 1) = 1955 * k :=
by
  sorry

end not_divisible_by_1955_l360_360136


namespace quadratic_monotonic_interval_a_leq_2_l360_360109

theorem quadratic_monotonic_interval_a_leq_2
  (a : ℝ) :
  (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f(x) ≤ f(y)) ↔ a ≤ 2 :=
by
  let f : ℝ → ℝ := λ x, x^2 - a * x + 1
  sorry

end quadratic_monotonic_interval_a_leq_2_l360_360109


namespace sum_of_roots_l360_360736

theorem sum_of_roots (a b c: ℝ) (h: a ≠ 0) (h_eq : a = 1) (h_eq2 : b = -6) (h_eq3 : c = 8):
    let Δ := b ^ 2 - 4 * a * c in
    let root1 := (-b + real.sqrt Δ) / (2 * a) in
    let root2 := (-b - real.sqrt Δ) / (2 * a) in
    root1 + root2 = 6 :=
by
  sorry

end sum_of_roots_l360_360736


namespace find_f_zero_f_monotonic_increasing_solve_inequality_l360_360031

variable {f : ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 2
axiom cond2 : ∀ x : ℝ, x > 0 → f(x) > 2

-- 1. Prove that f(0) = 2
theorem find_f_zero : f 0 = 2 := sorry

-- 2. Prove that f(x) is monotonic increasing on ℝ
theorem f_monotonic_increasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f(x₁) < f(x₂) := sorry

-- 3. Solve the inequality f(2t^2 - t - 3) - 2 < 0
theorem solve_inequality : ∀ t : ℝ, f(2 * t^2 - t - 3) - 2 < 0 ↔ (-1 < t ∧ t < 3 / 2) := sorry

end find_f_zero_f_monotonic_increasing_solve_inequality_l360_360031


namespace solve_for_x_l360_360876

theorem solve_for_x (x : ℝ) (h1 : x ≠ -3) (h2 : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5)) : x = -9 :=
by
  sorry

end solve_for_x_l360_360876


namespace solve_equation_l360_360942

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l360_360942


namespace dot_product_magnitude_l360_360163

open Real

variables (a b : ℝ^3) 

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross (u v : ℝ^3) : ℝ^3 := (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)
def dot (u v : ℝ^3) : ℝ := u.1*v.1 + u.2*v.2 + u.3*v.3

axiom norm_a : norm a = 3
axiom norm_b : norm b = 4
axiom cross_norm : norm (cross a b) = 6

theorem dot_product_magnitude : abs (dot a b) = 6 * sqrt 3 := by
  sorry

end dot_product_magnitude_l360_360163


namespace prob_within_range_l360_360516

variables (ξ : ℕ → ℚ) (c : ℚ)

-- Condition for the distribution
def dist_condition (k : ℕ) : Prop :=
  ξ k = c / (k * (k + 1))

-- Sum of probabilities is 1
def sum_prob_condition : Prop :=
  (ξ 1) + (ξ 2) + (ξ 3) = 1

-- Define probability P(0.5 < ξ < 2.5)
def prob_condition : Prop :=
  (ξ 1) + (ξ 2) = 8 / 9

theorem prob_within_range (h1 : sum_prob_condition ξ c)
  (h2 : dist_condition ξ 1) 
  (h3 : dist_condition ξ 2)
  (h4 : dist_condition ξ 3) : 
  prob_condition ξ :=
begin
  sorry
end

end prob_within_range_l360_360516


namespace diplomats_speak_french_l360_360219

-- Define the number of diplomats who speak French
def speaksFrench (f r t : ℕ) : ℕ := f + r

theorem diplomats_speak_french :
  (T = 150) → 
  (∃ d, d = 32 ∧ ¬(d = 0) ∧ nat.div (d * 5) 100 = 32) →
  (∃ n, n = 20 ∧ nat.div (n * T) 100 = T - 120) →
  (∃ b, b = 10 ∧ nat.div (b * T) 100 = 15) →
  F = 47 :=
  by
    -- T - Total number of diplomats
    let T := 150
    -- d = Diplomats who do not speak Russian
    let d := 32
    -- n = Diplomats who speak neither French nor Russian
    let n := 20
    -- b = Diplomats who speak both languages
    let b := 15
    -- F = Total diplomats who speak French
    let F := 47
    sorry

end diplomats_speak_french_l360_360219


namespace max_value_fraction_l360_360076

theorem max_value_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ max_val, max_val = 7 / 5 ∧ ∀ (x y : ℝ), 
    (x + y - 2 ≥ 0) → (y - x - 1 ≤ 0) → (x ≤ 1) → (x + 2*y) / (2*x + y) ≤ max_val :=
sorry

end max_value_fraction_l360_360076


namespace range_of_a_l360_360043

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) := ∀ x₁ x₂ : ℝ, x₁ < x₂ → -(5 - 2 * a)^x₁ > -(5 - 2 * a)^x₂

theorem range_of_a (a : ℝ) : (p a ∨ q a) → ¬ (p a ∧ q a) → a ≤ -2 := by 
  sorry

end range_of_a_l360_360043


namespace max_sin_expression_l360_360890

open Real

theorem max_sin_expression (α β : ℝ) (hα : 0 ≤ α) (hαπ : α ≤ π) (hβ : 0 ≤ β) (hβπ : β ≤ π) :
  (∃ (M : ℝ), M = (\sin α + \sin (α + β)) * \sin β ∧ (∀ (α β : ℝ) (hα : 0 ≤ α) (hαπ : α ≤ π) (hβ : 0 ≤ β) (hβπ : β ≤ π), 
    (\sin α + \sin (α + β) * \sin β) ≤ M)) ∧ M = 8 * Real.sqrt 3 / 9 :=
sorry

end max_sin_expression_l360_360890


namespace angle_between_clock_hands_at_7_oclock_l360_360711

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l360_360711


namespace opposite_of_neg_three_sevenths_l360_360261

theorem opposite_of_neg_three_sevenths:
  ∀ x : ℚ, (x = -3 / 7) → (∃ y : ℚ, y + x = 0 ∧ y = 3 / 7) :=
by
  sorry

end opposite_of_neg_three_sevenths_l360_360261


namespace max_red_points_on_circle_l360_360279

-- Define the conditions of the problem
def circle_points : Nat := 800

-- Define the sequence generator based on the given rule
def red_points_sequence (start : Nat) : List Nat :=
  List.iterate (λ x, (2 * x) % circle_points) start circle_points

-- Define the theorem with the correct answer
theorem max_red_points_on_circle : ∀ (start : Nat), start ≠ 0 → (List.toFinset (red_points_sequence start)).card ≤ 25 :=
by
  sorry

end max_red_points_on_circle_l360_360279


namespace second_train_speed_l360_360688

open Real

def train_speed (d₁ d₂ t₁ t₂ s₁ : ℝ) : ℝ := 
  let d₁_covered := s₁ * t₁
  let d₂_covered := d₂ - d₁_covered
  d₂_covered / t₂

theorem second_train_speed :
  train_speed 0 200 5 4 20 = 25 := 
sorry

end second_train_speed_l360_360688


namespace translation_of_point_l360_360127

-- We define the original point A and the translation value
def A : (ℝ × ℝ) := (1, 2)
def translation_x : ℝ := 3

-- We state that after translating point A 3 units to the right, point B has coordinates (4, 2)
theorem translation_of_point :
  let B := (A.1 + translation_x, A.2) in
  B = (4, 2) :=
by
  -- This is a placeholder to indicate the proof is not provided
  sorry

end translation_of_point_l360_360127


namespace repeating_decimal_sum_l360_360404

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l360_360404


namespace scaling_affine_transformation_l360_360608

variable {𝕜 : Type*} [Field 𝕜]
variable {V : Type*} [AddCommGroup V] [Module 𝕜 V]
variable (l : V) (k : 𝕜)

-- Definitions for points and stretched points
variable (A B C A' B' C' : V)
variable (h : C - A = (C - B) * (B - A))

-- Definition for the stretching transformation
def stretched (p : V) : V := l + k * (p - l)

-- Theorem to be proved: if C lies on the line AB, then C' lies on the line A'B'
theorem scaling_affine_transformation
  (hA' : A' = stretched A)
  (hB' : B' = stretched B)
  (hC' : C' = stretched C)
  : ∃ t : 𝕜, C' = A' + t * (B' - A') :=
sorry

end scaling_affine_transformation_l360_360608


namespace complex_sum_l360_360808

theorem complex_sum (x y z w : ℂ) (h1 : x = 20) (h2 : y = 20) (h3 : z = exp (3 * real.pi * complex.I / 13)) (h4 : w = exp (21 * real.pi * complex.I / 26)) :
  x * z + y * w = 40 * complex.cos (3 * real.pi / 13) * exp (27 * real.pi * complex.I / 52) :=
by {
  sorry
}

end complex_sum_l360_360808


namespace dot_product_magnitude_l360_360179

variables {a b : EuclideanSpace 3 ℝ}

/- Given conditions -/
def norm_a : ℝ := ‖a‖ = 3
def norm_b : ℝ := ‖b‖ = 4
def norm_cross : ℝ := ‖a × b‖ = 6

/- Desired property to prove -/
theorem dot_product_magnitude :
  norm_a →
  norm_b →
  norm_cross →
  |(a ⋅ b)| = 6 * real.sqrt 3 :=
by
  sorry

end dot_product_magnitude_l360_360179


namespace determine_m_l360_360817

def f (x : ℝ) := 5 * x^2 + 3 * x + 7
def g (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 1

theorem determine_m (m : ℝ) : f 5 - g 5 m = 55 → m = -7 :=
by
  unfold f
  unfold g
  sorry

end determine_m_l360_360817


namespace roof_collapse_l360_360373

theorem roof_collapse (roof_limit : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) :
  roof_limit = 500 → leaves_per_day = 100 → leaves_per_pound = 1000 → 
  let d := (roof_limit * leaves_per_pound) / leaves_per_day in d = 5000 :=
by
  intros h₁ h₂ h₃
  sorry

end roof_collapse_l360_360373


namespace distance_between_points_is_10_l360_360296

-- Define coordinates of the points
def point1 : ℝ × ℝ := (1, -1)
def point2 : ℝ × ℝ := (7, 7)

-- Define the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The theorem statement
theorem distance_between_points_is_10 :
  distance point1 point2 = 10 :=
by
  sorry -- Proof of the theorem

end distance_between_points_is_10_l360_360296


namespace largest_c_value_l360_360834

open Real

theorem largest_c_value :
  ∀ (n : ℕ) (x : Fin n → ℝ), 
  ∑ i j, (n - |i - j|) * x i * x j ≥ (1 / 2) * ∑ i, x i ^ 2 :=
by
  sorry

end largest_c_value_l360_360834


namespace least_possible_b_l360_360246

open Nat

/-- 
  Given conditions:
  a and b are consecutive Fibonacci numbers with a > b,
  and their sum is 100 degrees.
  We need to prove that the least possible value of b is 21 degrees.
-/
theorem least_possible_b (a b : ℕ) (h1 : fib a = fib (b + 1))
  (h2 : a > b) (h3 : a + b = 100) : b = 21 :=
sorry

end least_possible_b_l360_360246


namespace sum_of_values_such_that_gx_is_negative_three_l360_360201

def g (x : ℝ) : ℝ :=
if x < -3 then 3 * x + 6 else -x^2 - 2 * x + 2

theorem sum_of_values_such_that_gx_is_negative_three : 
  let S := {x : ℝ | g x = -3} in S.sum = -4 :=
by
  sorry

end sum_of_values_such_that_gx_is_negative_three_l360_360201


namespace path_area_and_construction_cost_l360_360350

theorem path_area_and_construction_cost 
  (field_length : ℝ) (field_width : ℝ) (path_width : ℝ) (construction_cost_per_sqm : ℝ)
  (h1 : field_length = 85) (h2 : field_width = 55) (h3 : path_width = 2.5) (h4 : construction_cost_per_sqm = 2) :
  let total_length := field_length + 2 * path_width
      total_width := field_width + 2 * path_width
      area_with_path := total_length * total_width
      area_of_field := field_length * field_width
      area_of_path := area_with_path - area_of_field
      total_cost := area_of_path * construction_cost_per_sqm
  in area_of_path = 725 ∧ total_cost = 1450 := 
by
  sorry

end path_area_and_construction_cost_l360_360350


namespace ones_digit_of_largest_power_of_2_in_20_fact_l360_360870

open Nat

def largest_power_of_2_in_factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else
    let sum_of_powers := ∑ m in range (n+1), m / 2
    sum_of_powers

def ones_digit_of_power_of_2 (exp : ℕ) : ℕ :=
  let cycle := [2, 4, 8, 6]
  cycle[exp % 4]

theorem ones_digit_of_largest_power_of_2_in_20_fact (n : ℕ) (h : n = 20) : 
  ones_digit_of_power_of_2 (largest_power_of_2_in_factorial n) = 4 :=
by
  rw [h]
  have : largest_power_of_2_in_factorial 20 = 18 := by
    -- Insert the calculations for largest_power_of_2_in_factorial here
    sorry
  rw [this]
  have : ones_digit_of_power_of_2 18 = 4 := by
    -- Insert the cycle calculations here
    sorry
  exact this

end ones_digit_of_largest_power_of_2_in_20_fact_l360_360870


namespace sum_numerator_denominator_l360_360020

def numbers_set := {n : ℕ | 1 ≤ n ∧ n ≤ 2000}
def remaining_set (s : set ℕ) := {n : ℕ | 1 ≤ n ∧ n ≤ 2000 ∧ n ∉ s}
def choose_four (s : set ℕ) := {t : finset ℕ | t ⊆ s ∧ t.card = 4}
noncomputable def hyperbrick_enclose_prob (c_set : set ℕ) (d_set : set ℕ) : ℚ :=
  let hyperbox_dims := {y : finset ℕ | y ⊆ d_set ∧ y.card = 4} in
  let valid_hyperbox_counts := (choose_four c_set).card in
  (valid_hyperbox_counts : ℚ) / (hyperbox_dims.card : ℚ)

theorem sum_numerator_denominator : 
  ∑ (n : ℕ) in (rat.num_denom (hyperbrick_enclose_prob (choose_four numbers_set) (choose_four (remaining_set numbers_set)))).to_list, n = 17 :=
sorry

end sum_numerator_denominator_l360_360020


namespace brush_length_percentage_increase_l360_360814

-- Define the length of Carla's brush in inches
def carla_brush_length_in_inches : ℝ := 12

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the length of Carmen's brush in centimeters
def carmen_brush_length_in_cm : ℝ := 45

-- Noncomputable definition to calculate the percentage increase
noncomputable def percentage_increase : ℝ :=
  let carla_brush_length_in_cm := carla_brush_length_in_inches * inch_to_cm
  (carmen_brush_length_in_cm - carla_brush_length_in_cm) / carla_brush_length_in_cm * 100

-- Statement to prove the percentage increase is 47.6%
theorem brush_length_percentage_increase :
  percentage_increase = 47.6 :=
sorry

end brush_length_percentage_increase_l360_360814


namespace sheepdog_catches_sheep_in_20_seconds_l360_360210

noncomputable def speed_sheep : ℝ := 12 -- feet per second
noncomputable def speed_sheepdog : ℝ := 20 -- feet per second
noncomputable def initial_distance : ℝ := 160 -- feet

theorem sheepdog_catches_sheep_in_20_seconds :
  (initial_distance / (speed_sheepdog - speed_sheep)) = 20 :=
by
  sorry

end sheepdog_catches_sheep_in_20_seconds_l360_360210


namespace max_rays_with_obtuse_angles_l360_360716

theorem max_rays_with_obtuse_angles {O : Point} :
  ∃ (n : ℕ), n = 4 ∧
  (∀ (rays : Fin n → Line), pairwise (λ i j, obtuse_angle (rays i) (rays j))) ∧
  (∀ (m : ℕ), (m > 4) → 
    ¬(∃ (rays : Fin m → Line), pairwise (λ i j, obtuse_angle (rays i) (rays j)))) :=
sorry

end max_rays_with_obtuse_angles_l360_360716


namespace coin_probability_not_unique_l360_360013

variables (p : ℝ) (w : ℝ)
def binomial_prob := 10 * p^3 * (1 - p)^2

theorem coin_probability_not_unique (h : binomial_prob p = 144 / 625) : 
  ∃ p1 p2, p1 ≠ p2 ∧ binomial_prob p1 = 144 / 625 ∧ binomial_prob p2 = 144 / 625 :=
by 
  sorry

end coin_probability_not_unique_l360_360013


namespace find_integer_l360_360365

theorem find_integer (n : ℤ) (h : 5 * (n - 2) = 85) : n = 19 :=
sorry

end find_integer_l360_360365


namespace percentage_reduction_price_increase_l360_360335

open Real

-- Part 1: Finding the percentage reduction each time
theorem percentage_reduction (P₀ P₂ : ℝ) (x : ℝ) (h₀ : P₀ = 50) (h₁ : P₂ = 32) (h₂ : P₀ * (1 - x) ^ 2 = P₂) :
  x = 0.20 :=
by
  dsimp at h₀ h₁,
  rw h₀ at h₂,
  rw h₁ at h₂,
  simp at h₂,
  sorry

-- Part 2: Determining the price increase per kilogram
theorem price_increase (P y : ℝ) (profit_per_kg : ℝ) (initial_sales : ℝ) 
  (price_increase_limit : ℝ) (sales_decrease_rate : ℝ) (target_profit : ℝ)
  (h₀ : profit_per_kg = 10) (h₁ : initial_sales = 500) (h₂ : price_increase_limit = 8)
  (h₃ : sales_decrease_rate = 20) (h₄ : target_profit = 6000) (0 < y ∧ y ≤ price_increase_limit)
  (h₅ : (profit_per_kg + y) * (initial_sales - sales_decrease_rate * y) = target_profit) :
  y = 5 :=
by
  dsimp at h₀ h₁ h₂ h₃ h₄,
  rw [h₀, h₁, h₂, h₃, h₄] at h₅,
  sorry

end percentage_reduction_price_increase_l360_360335


namespace x_increase_80_percent_l360_360661

noncomputable def percentage_increase (x1 x2 : ℝ) : ℝ :=
  ((x2 / x1) - 1) * 100

theorem x_increase_80_percent
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 * y1 = x2 * y2)
  (h2 : y2 = (5 / 9) * y1) :
  percentage_increase x1 x2 = 80 :=
by
  sorry

end x_increase_80_percent_l360_360661


namespace max_sum_square_dist_l360_360658

noncomputable def pentagon_vertices : list (ℂ) := 
  [1, exp (2 * π * I / 5), exp (4 * π * I / 5), exp (6 * π * I / 5), exp (8 * π * I / 5)]

def S (points : list (ℂ)) : ℝ :=
  ∑ i in (points.toFinset), ∑ j in (points.toFinset \ {i}), abs(i - j)^2

theorem max_sum_square_dist :
  ∀ (points : list (ℂ)), 
    (∀ p ∈ points, p ∈ pentagon_vertices) ∧ 
    points.length = 2018 → 
    lcm (counts points) = [612, 235, 468, 468, 235] ∧ 
    S points = maximum (λ points, S points) :=
begin
  -- Proof omitted
  sorry
end

end max_sum_square_dist_l360_360658


namespace points_on_equation_correct_l360_360946

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l360_360946


namespace equation_solution_l360_360937

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l360_360937


namespace children_order_berries_final_weight_l360_360288

-- Problem 1: Order of the Children
theorem children_order
    (vika_ahead_sonia : ∀ i j : ℕ, vika i → sonia j → i < j)
    (vika_after_alla : ∀ i j : ℕ, alla i → vika j → i < j)
    (not_next_to_alla : ∀ i : ℕ, borya i → ¬ next_to alla i)
    (denis_not_next_to : ∀ i : ℕ, denis i → ¬ (next_to alla i ∨ next_to vika i ∨ next_to borya i)) :
    ∃ (i j k l m : ℕ), alla i ∧ vika j ∧ borya k ∧ sonia l ∧ denis m ∧ 
                      i < j ∧ j < k ∧ k < l ∧ l < m ∧
                      ¬ next_to alla k ∧ ¬ (next_to alla m ∨ next_to vika m ∨ next_to borya m) :=
sorry

-- Problem 2: Change in Weight of Berries
theorem berries_final_weight
    (initial_weight : ℕ)
    (initial_water_content : ℕ)
    (final_water_content : ℕ)
    (initial_weight_eq : initial_weight = 100)
    (initial_water_content_eq : initial_water_content = 99)
    (final_water_content_eq : final_water_content = 98) :
    final_weight = 50 :=
sorry

end children_order_berries_final_weight_l360_360288


namespace number_of_boys_is_90_l360_360666

-- Define the conditions
variables (B G : ℕ)
axiom sum_condition : B + G = 150
axiom percentage_condition : G = (B / 150) * 100

-- State the theorem
theorem number_of_boys_is_90 : B = 90 :=
by
  -- We can skip the proof for now using sorry
  sorry

end number_of_boys_is_90_l360_360666


namespace children_got_on_the_bus_l360_360768

-- Definitions
def original_children : ℕ := 26
def current_children : ℕ := 64

-- Theorem stating the problem
theorem children_got_on_the_bus : (current_children - original_children = 38) :=
by {
  sorry
}

end children_got_on_the_bus_l360_360768


namespace largest_power_of_2_dividing_32_factorial_ones_digit_l360_360852

def power_of_two_ones_digit (n: ℕ) : ℕ :=
  let digits_cycle := [2, 4, 8, 6]
  digits_cycle[(n % 4) - 1]

theorem largest_power_of_2_dividing_32_factorial_ones_digit :
  power_of_two_ones_digit 31 = 8 := by
  sorry

end largest_power_of_2_dividing_32_factorial_ones_digit_l360_360852


namespace clock_angle_at_7_oclock_l360_360705

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l360_360705


namespace dot_product_magnitude_l360_360158

variable {a b : EuclideanSpace ℝ (Fin n)}
variable (norm_a : ∥a∥ = 3)
variable (norm_b : ∥b∥ = 4)
variable (norm_cross_ab : ∥a × b∥ = 6)

theorem dot_product_magnitude : ∥a∥ = 3 → ∥b∥ = 4 → ∥a × b∥ = 6 → |a ∘ b| = 6 * real.sqrt 3 := 
by 
  intro norm_a norm_b norm_cross_ab
  sorry

end dot_product_magnitude_l360_360158


namespace max_tickets_l360_360014

-- Define the conditions
def ticket_cost (n : ℕ) : ℝ :=
  if n ≤ 6 then 15 * n
  else 13.5 * n

-- Define the main theorem
theorem max_tickets (budget : ℝ) : (∀ n : ℕ, ticket_cost n ≤ budget) → budget = 120 → n ≤ 8 :=
  by
  sorry

end max_tickets_l360_360014


namespace quadrilateral_area_proof_l360_360387

noncomputable def areaOfQuadrilateral (a b c d : ℝ) (h₁ : a ≠ c) (h₃ : b ≠ d) : ℝ :=
let diagonal := Math.sqrt (a^2 + c^2 - 2 * a * c * Math.cos (Math.pi / 3)) in
let s1 := (a + c + diagonal) / 2 in
let s2 := (b + d + diagonal) / 2 in
let area1 := Math.sqrt(s1 * (s1 - a) * (s1 - c) * (s1 - diagonal)) in
let area2 := Math.sqrt(s2 * (s2 - b) * (s2 - d) * (s2 - diagonal)) in
area1 + area2

theorem quadrilateral_area_proof :
  areaOfQuadrilateral 25 30 25 18 25 ≠ 18 25 ≠ 30 = 
  -- Expected final area, replace this placeholder with the computed value
  sorry

end quadrilateral_area_proof_l360_360387


namespace acute_triangle_prime_angles_isosceles_l360_360998

theorem acute_triangle_prime_angles_isosceles :
  ∃ (x y z : ℕ), 
    (2 ≤ x ∧ 2 ≤ y ∧ 2 ≤ z) ∧ 
    (nat.prime x ∧ nat.prime y ∧ nat.prime z) ∧ 
    (x + y + z = 180) ∧ 
    (x < 90 ∧ y < 90 ∧ z < 90) ∧ 
    ((x = 2 ∧ y = 89 ∧ z = 89) ∨ (x = 89 ∧ y = 2 ∧ z = 89) ∨ (x = 89 ∧ y = 89 ∧ z = 2)) ∧ 
    (x ≠ y → x ≠ z → y ≠ z → (x = 2 ∧ y = 89 ∧ z = 89) ∧ 
     ∃ u v w : ℕ, u = y ∧ v = y ∧ w = z ∧ 
     (u = v ∨ v = w ∨ w = u)) := 
by
  sorry

end acute_triangle_prime_angles_isosceles_l360_360998


namespace repeating_decimal_sum_in_lowest_terms_l360_360431

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l360_360431


namespace problem_statement_l360_360490

-- Define the ellipse parameters and properties
structure Ellipse where
  a b : ℝ
  h : a > b ∧ b > 0
  e : ℝ
  eccentricity_condition : e = 1/2

def ellipse_eq (E : Ellipse) : Prop :=
  E.eccentricity_condition → 
  E.a = 2 ∧ E.b = sqrt 3 → 
  ∀ x y : ℝ, (x^2 / E.a^2) + (y^2 / E.b^2) = 1 ↔ (x^2 / 4) + (y^2 / 3) = 1 

-- Define the points and the slope properties
def slope_sum_constant (E : Ellipse) (T : ℝ × ℝ) : Prop :=
  T = (4, 0) →
  ∀ A B : ℝ × ℝ, 
  ∀ k : ℝ,
  (A, B) ∈ find_intersections E (y - k * (x - 1)) →
  let mTA := (A.2 - T.2) / (A.1 - T.1) in
  let mTB := (B.2 - T.2) / (B.1 - T.1) in
  mTA + mTB = 0 

-- Define the existence of intersection points
def find_intersections (E : Ellipse) (l_eq : ℝ) : set (ℝ × ℝ) :=
  { (x, y) : ℝ × ℝ | y = k(x - 1) ∧ (x^2 / E.a^2) + (y^2 / E.b^2) = 1 }

-- Proof problem statement
theorem problem_statement (E : Ellipse) (T : ℝ × ℝ) :
  ellipse_eq E ∧ slope_sum_constant E T :=
sorry

end problem_statement_l360_360490


namespace log_sqrt2_over_log4_eq_quarter_log_sqrt2_gt1_iff_range_l360_360816

noncomputable definition log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem log_sqrt2_over_log4_eq_quarter :
  log_base 4 (Real.sqrt 2) = 1 / 4 :=
by
  sorry

theorem log_sqrt2_gt1_iff_range (x : ℝ) :
  log_base x (Real.sqrt 2) > 1 ↔ 1 < x ∧ x < Real.sqrt 2 :=
by
  sorry

end log_sqrt2_over_log4_eq_quarter_log_sqrt2_gt1_iff_range_l360_360816


namespace minimum_degree_g_l360_360382

-- Definitions of the degrees of the polynomials
variables {R : Type*} [CommRing R]

def degree_f (f : R[X]) : Nat := 10
def degree_h (h : R[X]) : Nat := 13
def equation (f g h : R[X]) : Prop := 5 * f + 7 * g = h

-- Statement of the problem
theorem minimum_degree_g (f g h : R[X]) 
  (hf : degree f = degree_f f)
  (hh : degree h = degree_h h)
  (heq : equation f g h) :
  ∃ n, degree g = n ∧ n = 13 :=
begin
  sorry
end

end minimum_degree_g_l360_360382


namespace count_repeating_decimals_between_1_and_15_l360_360467

def is_repeating_decimal (n d : ℕ) : Prop :=
  ∀ p, p.prime ∧ (p ∣ d) → p = 2 ∨ p = 5 → False

theorem count_repeating_decimals_between_1_and_15 : 
  (finset.filter (λ n, is_repeating_decimal n 18) (finset.range 16)).card = 14 :=
  sorry

end count_repeating_decimals_between_1_and_15_l360_360467


namespace points_on_equation_correct_l360_360944

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l360_360944


namespace boys_girls_difference_l360_360278

/--
If there are 550 students in a class and the ratio of boys to girls is 7:4, 
prove that the number of boys exceeds the number of girls by 150.
-/
theorem boys_girls_difference : 
  ∀ (students boys_ratio girls_ratio : ℕ),
  students = 550 →
  boys_ratio = 7 →
  girls_ratio = 4 →
  (students * boys_ratio) % (boys_ratio + girls_ratio) = 0 ∧
  (students * girls_ratio) % (boys_ratio + girls_ratio) = 0 →
  (students * boys_ratio - students * girls_ratio) / (boys_ratio + girls_ratio) = 150 :=
by
  intros students boys_ratio girls_ratio h_students h_boys_ratio h_girls_ratio h_divisibility
  -- The detailed proof would follow here, but we add 'sorry' to bypass it.
  sorry

end boys_girls_difference_l360_360278


namespace half_yearly_EAR_quarterly_EAR_monthly_EAR_daily_EAR_l360_360388

def nominal_rate : ℝ := 0.12
def EAR (i : ℝ) (n : ℕ) : ℝ := (1 + i / n) ^ n - 1

theorem half_yearly_EAR : EAR nominal_rate 2 = 0.1236 := by sorry
theorem quarterly_EAR : EAR nominal_rate 4 = 0.1255 := by sorry
theorem monthly_EAR : EAR nominal_rate 12 = 0.1268 := by sorry
theorem daily_EAR : EAR nominal_rate 365 = 0.1275 := by sorry

end half_yearly_EAR_quarterly_EAR_monthly_EAR_daily_EAR_l360_360388


namespace sum_of_roots_of_quadratic_eq_l360_360733

theorem sum_of_roots_of_quadratic_eq : 
  ∀ (a b c : ℝ), (x^2 - 6 * x + 8 = 0) → (a = 1 ∧ b = -6 ∧ c = 8) → -b / a = 6 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_eq_l360_360733


namespace AC_is_tangent_to_O_CD_equals_HF_lengths_BF_AF_l360_360113

-- Definitions and conditions
variables {A B C E F O D H : Type}
variables {β : Real} -- angle C in ΔABC
variables {cd eh : Real} -- lengths CD and EH

-- Given conditions
variables (AngleBisector : β = 90)
variables (AngleBisects : ∃ E : Type, (AngleBisector ∧ (∃ perpE : E, perpE)))
variables (Perpendiculars : ∀ E : Type, ∃ F : Type, (Perpendiculars ∧ (∃ perpF : F, perpF)))
variables (Circumcircle : ∀ B E F : Type, ∃ O : Type, (Circumcircle ∧ (∃ intersectsO : O, intersectsO)))

-- Problem (1)
theorem AC_is_tangent_to_O (h1 : AngleBisector) (h2 : AngleBisects) (h3 : Perpendiculars) (h4 : Circumcircle) :
  is_tangent_to O A C :=
sorry

-- Problem (2)
theorem CD_equals_HF (h1 : AngleBisector) (h2 : AngleBisects) (h3 : Perpendiculars) (h4 : Circumcircle)
  (perpendicularEH : H ⊥ AB) :
  eq CD HF :=
sorry

-- Problem (3)
theorem lengths_BF_AF (h1 : AngleBisector) (h2 : AngleBisects) (h3 : Perpendiculars) (h4 : Circumcircle)
  (perpendicularEH : H ⊥ AB) (h_cd : cd = 1) (h_eh : eh = 3) :
  is_length_of BF 10 ∧ is_length_of AF (5 / 4) :=
sorry

end AC_is_tangent_to_O_CD_equals_HF_lengths_BF_AF_l360_360113


namespace find_sum_of_solutions_l360_360205

noncomputable theory
open Classical

def symmetric_function (f : ℝ → ℝ) := ∀ x : ℝ, f(x) = f(4 - x)

def strictly_increasing_on (f : ℝ → ℝ) (s : Set ℝ) := ∀ x1 x2 ∈ s, x1 < x2 → f(x1) < f(x2)

theorem find_sum_of_solutions (f : ℝ → ℝ)
  (H1 : symmetric_function f)
  (H2 : strictly_increasing_on f {x | 2 < x}) :
  ∑ x in { x | f(2 - x) = f((3 * x + 11) / (x + 4)) }.to_finset = -8 :=
by
  sorry

end find_sum_of_solutions_l360_360205


namespace compute_value_l360_360801

variables {p q r : ℝ}

theorem compute_value (h1 : (p * q) / (p + r) + (q * r) / (q + p) + (r * p) / (r + q) = -7)
                      (h2 : (p * r) / (p + r) + (q * p) / (q + p) + (r * q) / (r + q) = 8) :
  (q / (p + q) + r / (q + r) + p / (r + p)) = 9 :=
sorry

end compute_value_l360_360801


namespace find_m_l360_360487

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) :
  (∀ n, S n = (n * (3 * n - 1)) / 2) →
  (a 1 = 1) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (a m = 3 * m - 2) →
  (a 4 * a 4 = a 1 * a m) →
  m = 34 :=
by
  intro hS h1 ha1 ha2 hgeom
  sorry

end find_m_l360_360487


namespace inequality_proof_l360_360589

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_condition : 1/x + 1/y + 1/z < 1/(x*y*z)) :
  2*x/Real.sqrt(1 + x^2) + 2*y/Real.sqrt(1 + y^2) + 2*z/Real.sqrt(1 + z^2) < 3 :=
sorry

end inequality_proof_l360_360589


namespace smallest_number_gt_sum_digits_1755_l360_360458

theorem smallest_number_gt_sum_digits_1755 :
  ∃ (n : ℕ) (a b c d : ℕ), a ≠ 0 ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ n = (a + b + c + d) + 1755 ∧ n = 1770 :=
by {
  sorry
}

end smallest_number_gt_sum_digits_1755_l360_360458


namespace value_sum_a_i_l360_360445

noncomputable def f (x : ℝ) : ℝ := 1 - x + x^2 - x^3 + ... + (-1)^{19} * x^{19} + x^{20}
def y (x : ℝ) : ℝ := x + 1
def g (y : ℝ) : ℝ := sorry -- this is a placeholder for the polynomial in terms of y

theorem value_sum_a_i : 
  let a := sorry -- placeholder for the constant term in the polynomial g
      a_1 := sorry -- placeholder for the coefficient of y in g
      -- ... other coefficients a_i placeholders ...
      a_{20} := sorry -- placeholder for the coefficient of y^20 in g
  a + a_1 + ... + a_{20} = 1 := by
  sorry

end value_sum_a_i_l360_360445


namespace sum_of_digits_10_pow_102_sub_197_l360_360741

def sum_of_digits (n : Nat) : Nat :=
  if n == 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_10_pow_102_sub_197 : sum_of_digits (10 ^ 102 - 197) = 911 :=
by
  sorry

end sum_of_digits_10_pow_102_sub_197_l360_360741


namespace trisect_arc_of_trisect_base_l360_360572

/-- Let ABC be an equilateral triangle and Γ the semicircle drawn exteriorly to the triangle,
having BC as diameter. Show that if a line passing through A trisects BC, it also trisects
the arc Γ. -/
theorem trisect_arc_of_trisect_base
  (A B C : Point)
  (is_eq_triangle : is_equilateral_triangle A B C)
  (Γ : semicircle B C)
  (line_through_A_trisects_BC : ∃ T, (T ∈ (line_through A ∩ closed_segment B C)) ∧ ((dist B T) = (dist T C) / 2)) :
  ∃ T', (T' ∈ (line_through A ∩ arc Γ)) ∧ trisects_arc B C Γ T' :=
  sorry

end trisect_arc_of_trisect_base_l360_360572


namespace time_for_B_alone_l360_360753

def work_rate (time : ℝ) : ℝ := 1 / time

variables (A B C : ℝ)
variables (hA : A = work_rate 4) (hBC : B + C = work_rate 2) (hAC : A + C = work_rate 2)

theorem time_for_B_alone : 1 / B = 4 := by
  rw [work_rate] at hA hBC hAC
  sorry

end time_for_B_alone_l360_360753


namespace repeating_decimal_sum_in_lowest_terms_l360_360432

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l360_360432


namespace product_of_possible_values_l360_360803

theorem product_of_possible_values (N : ℤ) (M L : ℤ) 
(h1 : M = L + N)
(h2 : M - 3 = L + N - 3)
(h3 : L + 5 = L + 5)
(h4 : |(L + N - 3) - (L + 5)| = 4) :
N = 12 ∨ N = 4 → (12 * 4 = 48) :=
by sorry

end product_of_possible_values_l360_360803


namespace dot_product_magnitude_l360_360162

open Real

variables (a b : ℝ^3) 

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross (u v : ℝ^3) : ℝ^3 := (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)
def dot (u v : ℝ^3) : ℝ := u.1*v.1 + u.2*v.2 + u.3*v.3

axiom norm_a : norm a = 3
axiom norm_b : norm b = 4
axiom cross_norm : norm (cross a b) = 6

theorem dot_product_magnitude : abs (dot a b) = 6 * sqrt 3 := by
  sorry

end dot_product_magnitude_l360_360162


namespace radius_of_circle_l360_360635

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l360_360635


namespace multiplication_scaling_l360_360051

theorem multiplication_scaling (h : 28 * 15 = 420) : 
  (28 / 10) * (15 / 10) = 2.8 * 1.5 ∧ 
  (28 / 100) * 1.5 = 0.28 * 1.5 ∧ 
  (28 / 1000) * (15 / 100) = 0.028 * 0.15 :=
by 
  sorry

end multiplication_scaling_l360_360051


namespace train_speed_from_clicks_l360_360662

theorem train_speed_from_clicks (speed_mph : ℝ) (rail_length_ft : ℝ) (clicks_heard : ℝ) :
  rail_length_ft = 40 →
  clicks_heard = 1 →
  (60 * rail_length_ft * clicks_heard * speed_mph / 5280) = 27 :=
by
  intros h1 h2
  sorry

end train_speed_from_clicks_l360_360662


namespace age_condition_l360_360913

theorem age_condition (x y z : ℕ) (h1 : x > y) : 
  (z > y) ↔ (y + z > 2 * x) ∧ (∀ x y z, y + z > 2 * x → z > y) := sorry

end age_condition_l360_360913


namespace parabola_shift_l360_360262

theorem parabola_shift (x : ℝ) : 
  let y := 3 * x^2 in
  (fun x => 3 * (x - 1)^2 - 2) = (fun x => y)[x := (x - 1)] - 2 :=
sorry

end parabola_shift_l360_360262


namespace election_total_votes_l360_360999

theorem election_total_votes
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (h1 : votes_A = 55 * total_votes / 100)
  (h2 : votes_B = 35 * total_votes / 100)
  (h3 : votes_C = total_votes - votes_A - votes_B)
  (h4 : votes_A = votes_B + 400) :
  total_votes = 2000 := by
  sorry

end election_total_votes_l360_360999


namespace problem_statement_l360_360049

-- Define the imaginary unit i
def imaginary_unit : ℂ := Complex.I

-- The main theorem statement
theorem problem_statement (m n : ℝ) (h : m * (1 + imaginary_unit) = 1 + n * imaginary_unit) :
  (Complex.div (m + n * imaginary_unit) (m - n * imaginary_unit)) ^ 2 = -1 :=
begin
  sorry
end

end problem_statement_l360_360049


namespace area_of_ellipse_l360_360550

def endpoints_of_major_axis : (ℝ × ℝ) := (-10, 2)
def endpoints_of_major_axis' : (ℝ × ℝ) := (10, 2)
def passes_through_points_1 : (ℝ × ℝ) := (8, 6)
def passes_through_points_2 : (ℝ × ℝ) := (-8, -2)
def semi_major_axis_length : ℝ := 10
def semi_minor_axis_length : ℝ := 20 / 3

theorem area_of_ellipse 
  (a_eq : (endpoints_of_major_axis.fst + endpoints_of_major_axis'.fst) / 2 = 0)
  (b_eq : (endpoints_of_major_axis.snd + endpoints_of_major_axis'.snd) / 2 = 2)
  (ellipse_eq_1 : (8 - 0)^2 / semi_major_axis_length^2 + (6 - 2)^2 / semi_minor_axis_length^2 = 1)
  (ellipse_eq_2 : (-8 - 0)^2 / semi_major_axis_length^2 + (-2 - 2)^2 / semi_minor_axis_length^2 = 1) :
  let A := Real.pi * semi_major_axis_length * semi_minor_axis_length in
  A = 200 * Real.pi / 3 :=
by
  sorry

end area_of_ellipse_l360_360550


namespace isosceles_triangles_with_18_matches_l360_360522

theorem isosceles_triangles_with_18_matches :
  let x_values := { x : ℕ | 4.5 < x ∧ x < 9 }
  in #x_values = 4 :=
by
  sorry

end isosceles_triangles_with_18_matches_l360_360522


namespace repaved_inches_before_today_l360_360775

theorem repaved_inches_before_today :
  let A := 4000
  let B := 3500
  let C := 2500
  let repaved_A := 0.70 * A
  let repaved_B := 0.60 * B
  let repaved_C := 0.80 * C
  let total_repaved_before := repaved_A + repaved_B + repaved_C
  let repaved_today := 950
  let new_total_repaved := total_repaved_before + repaved_today
  new_total_repaved - repaved_today = 6900 :=
by
  sorry

end repaved_inches_before_today_l360_360775


namespace angle_EMF_90_l360_360566

open EuclideanGeometry

-- Define the triangle and points on it.
variable (A B C M E F : Point)

-- Define the conditions mathematically
def B_angle_120 (t : Triangle) : Prop :=
  t.angles B = 120

def midpoint_M (t : Triangle) (M : Point) : Prop :=
  M = midpoint t.A t.C

def E_and_F_conditions (t : Triangle) (E F : Point) : Prop :=
  dist t.A E = dist E F ∧ dist E F = dist F t.C

-- Main goal to prove
theorem angle_EMF_90 (t : Triangle)
  (hB : B_angle_120 t)
  (hM_mid : midpoint_M t M)
  (hE_F : E_and_F_conditions t E F) :
  angle E M F = 90 :=
by
  sorry

end angle_EMF_90_l360_360566


namespace smallest_polynomial_degree_is_seven_l360_360620

noncomputable def has_rational_polynomial : Prop :=
  ∃ p : Polynomial ℚ, 
    p ≠ 0 ∧ 
    p.eval (3 - Real.sqrt 8) = 0 ∧
    p.eval (5 + Real.sqrt 12) = 0 ∧
    p.eval (16 - 2 * Real.sqrt 9) = 0 ∧
    p.eval (- Real.sqrt 3) = 0

theorem smallest_polynomial_degree_is_seven
  (h : has_rational_polynomial) : ∃ p : Polynomial ℚ, p.degree = 7 :=
sorry

end smallest_polynomial_degree_is_seven_l360_360620


namespace collinear_PI_D_l360_360575

noncomputable theory
open_locale classical

variables {A B C I D E F P : Type*}
variables [point A] [point B] [point C] [point I] [point D] [point E] [point F] [point P]

-- Definitions based on given conditions
variable (triangle_ABC : triangle_type A B C)
variable (incenter_I : incenter_type triangle_ABC I)
variable (tangent_D : tangent_point_type incenter_I B C D)
variable (tangent_E : tangent_point_type incenter_I C A E)
variable (tangent_F : tangent_point_type incenter_I A B F)
variable (same_side_P_A_EF : same_side_type P A E F)
variable (angle_PEF_ABC : angle_eq_type P E F A B C)
variable (angle_PFE_ACB : angle_eq_type P F E A C B)

-- The theorem to prove P, I, D are collinear given the conditions
theorem collinear_PI_D
  (h1 : triangle_ABC)
  (h2 : incenter_I)
  (h3 : tangent_D)
  (h4 : tangent_E)
  (h5 : tangent_F)
  (h6 : same_side_P_A_EF)
  (h7 : angle_PEF_ABC)
  (h8 : angle_PFE_ACB) :
  collinear P I D :=
sorry

end collinear_PI_D_l360_360575


namespace math_problem_l360_360493
noncomputable theory

open Classical

-- Define the necessary variables and conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (m n : ℝ × ℝ)

-- Define the vectors
def m : ℝ × ℝ := (sqrt 3, cos A + 1)
def n : ℝ × ℝ := (sin A, -1)

-- Define the problem conditions and required proof
theorem math_problem : 
  (m.1 * n.1 + m.2 * n.2 = 0) ∧ 
  (a = 2) ∧ 
  (cos B = sqrt 3 / 3) ∧ 
  (sin A = sqrt 3 / 2) ∧ 
  (sin B = sqrt 6 / 3) →
  A = π / 3 ∧ 
  b = 4 * sqrt 2 / 3 := 
sorry

end math_problem_l360_360493


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360845

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  let n := 32 in
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i) in 
  (2 ^ k) % 10 = 8 :=
by
  let n := 32
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i)
  
  have h1 : k = 31 := by sorry
  have h2 : (2 ^ 31) % 10 = 8 := by sorry
  
  exact h2

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360845


namespace factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l360_360303

theorem factorization_A (x y : ℝ) : x^2 - 2 * x * y = x * (x - 2 * y) :=
  by sorry

theorem factorization_B (x y : ℝ) : x^2 - 25 * y^2 = (x - 5 * y) * (x + 5 * y) :=
  by sorry

theorem factorization_C (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 :=
  by sorry

theorem factorization_D_incorrect (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) :=
  by sorry

theorem factorization_D_correct (x : ℝ) : x^2 + x - 2 = (x + 2) * (x - 1) :=
  by sorry

end factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l360_360303


namespace angle_between_clock_hands_at_7_oclock_l360_360709

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l360_360709


namespace lines_perpendicular_l360_360609

variables {α : Type*} 
open_locale classical

-- Define the points and their conditions
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Define the quadrilateral and inscribed circle conditions
structure Quadrilateral :=
(A B C D : Point)
(omega : set Point) -- Circle containing the points

-- Define the midpoints of the arcs
structure Midpoints :=
(M N P Q : Point) -- Midpoints of the arcs based on the points A, B, C, D

-- Define the proof that MP and NQ are perpendicular
theorem lines_perpendicular (q : Quadrilateral) (m : Midpoints) 
  (H : q.A ∈ q.omega ∧ q.B ∈ q.omega ∧ q.C ∈ q.omega ∧ q.D ∈ q.omega)
  (H1 : midpoint q.A q.B = m.M ∧ midpoint q.B q.C = m.N ∧ midpoint q.C q.D = m.P ∧ midpoint q.D q.A = m.Q) : 
  is_perpendicular (line_through m.M m.P) (line_through m.N m.Q) :=
begin
  sorry
end

end lines_perpendicular_l360_360609


namespace arithmetic_sequence_sum_nine_l360_360489

variable {α : Type*} [LinearOrderedField α]

/-- An arithmetic sequence (a_n) is defined by a starting term a_1 and a common difference d. -/
def arithmetic_seq (a d n : α) : α := a + (n - 1) * d

/-- The sum of the first n terms of an arithmetic sequence. -/
def arithmetic_sum (a d n : α) : α := n / 2 * (2 * a + (n - 1) * d)

/-- Prove that for a given arithmetic sequence where a_2 + a_4 + a_9 = 24, the sum of the first 9 terms is 72. -/
theorem arithmetic_sequence_sum_nine 
  {a d : α}
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 4 + arithmetic_seq a d 9 = 24) :
  arithmetic_sum a d 9 = 72 := 
by
  sorry

end arithmetic_sequence_sum_nine_l360_360489


namespace hyperbola_m_range_l360_360506

theorem hyperbola_m_range (m : ℝ) (h_eq : ∀ x y, (x^2 / m) - (y^2 / (2*m - 1)) = 1) : 
  0 < m ∧ m < 1/2 :=
sorry

end hyperbola_m_range_l360_360506


namespace monthly_salary_is_correct_l360_360346

noncomputable def man's_salary : ℝ :=
  let S : ℝ := 6500
  S

theorem monthly_salary_is_correct (S : ℝ) (h1 : S * 0.20 = S * 0.20) (h2 : S * 0.80 * 1.20 + 260 = S):
  S = man's_salary :=
by sorry

end monthly_salary_is_correct_l360_360346


namespace inequality_proof_l360_360090

theorem inequality_proof (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 :=
sorry

end inequality_proof_l360_360090


namespace person_opening_100th_door_l360_360358

-- Define the initial order of spelunkers
def spelunkers := ["Albert", "Ben", "Cyril", "Dan", "Erik", "Filip", "Gábo"]

-- Function to determine the person who will open the nth door
def nth_person_opening_door (n : Nat) : String :=
  let position := (n % spelunkers.length) in
  spelunkers.getOrElse position "Unknown"

-- Theorem to state the person who opens the 100th door
theorem person_opening_100th_door : nth_person_opening_door 100 = "Ben" :=
  by {
    -- Specific proof details would go here.
    sorry
  }

end person_opening_100th_door_l360_360358


namespace rows_with_exactly_10_people_l360_360355

theorem rows_with_exactly_10_people (x : ℕ) (total_people : ℕ) (row_nine_seat : ℕ) (row_ten_seat : ℕ) 
    (H1 : row_nine_seat = 9) (H2 : row_ten_seat = 10) 
    (H3 : total_people = 55) 
    (H4 : total_people = x * row_ten_seat + (6 - x) * row_nine_seat) 
    : x = 1 :=
by
  sorry

end rows_with_exactly_10_people_l360_360355


namespace archer_weekly_spend_l360_360363

noncomputable def total_shots_per_week (shots_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  shots_per_day * days_per_week

noncomputable def arrows_recovered (total_shots : ℕ) (recovery_rate : ℝ) : ℕ :=
  (total_shots : ℝ) * recovery_rate |> Int.ofNat

noncomputable def net_arrows_used (total_shots : ℕ) (recovered_arrows : ℕ) : ℕ :=
  total_shots - recovered_arrows

noncomputable def cost_of_arrows (net_arrows : ℕ) (cost_per_arrow : ℝ) : ℝ :=
  net_arrows * cost_per_arrow

noncomputable def team_contribution (total_cost : ℝ) (contribution_rate : ℝ) : ℝ :=
  total_cost * contribution_rate

noncomputable def archer_spend (total_cost : ℝ) (team_contribution : ℝ) : ℝ :=
  total_cost - team_contribution

theorem archer_weekly_spend
  (shots_per_day : ℕ) (days_per_week : ℕ) (recovery_rate : ℝ)
  (cost_per_arrow : ℝ) (contribution_rate : ℝ) :
  archer_spend
    (cost_of_arrows
      (net_arrows_used
        (total_shots_per_week shots_per_day days_per_week)
        (arrows_recovered
          (total_shots_per_week shots_per_day days_per_week)
          recovery_rate))
      cost_per_arrow)
    (team_contribution
      (cost_of_arrows
        (net_arrows_used
          (total_shots_per_week shots_per_day days_per_week)
          (arrows_recovered
            (total_shots_per_week shots_per_day days_per_week)
            recovery_rate))
        cost_per_arrow)
      contribution_rate) = 1056 :=
by
  sorry

end archer_weekly_spend_l360_360363


namespace min_groups_for_twins_activities_l360_360012

theorem min_groups_for_twins_activities : 
  ∃ (k : ℕ), k = 14 ∧ 
  (∀ (A a B b C c D d E e : Type) 
    (groups : list (list (Type))),
    ¬(∃ g ∈ groups, A ∈ g ∧ a ∈ g) ∧ 
    (∀ x y, x ≠ y ∧ (x, y) ∉ {(A, a), (B, b), (C, c), (D, d), (E, e)} → 
      ∃ g ∈ groups, x ∈ g ∧ y ∈ g ∧ 
      (∀ h ∈ groups, x ∈ h → y ∉ h)) ∧ 
    (∃ k', k' ≤ k ∧ 
      ∀ p ∈ [A, a, B, b, C, c, D, d, E, e], 
      ∃ g₁ g₂ ∈ groups, 
      p ∈ g₁ ∧ p ∈ g₂ ∧ 
      (∀ g₃ ∈ groups, p ∈ g₃ → g₃ = g₁ ∨ g₃ = g₂)) )) :=
by
  sorry

end min_groups_for_twins_activities_l360_360012


namespace find_m_l360_360540

theorem find_m (m : ℝ) : 
  (∀ x : ℝ, f x = cos (x - π / 6) + cos (x + π / 6) + sin x + m) → 
  (∀ x : ℝ, f x ≤ 1) → m = -1 := 
begin
  -- Definitions and conditions
  assume (h : ∀ x : ℝ, f x = cos (x - π / 6) + cos (x + π / 6) + sin x + m)
  (h_max : ∀ x : ℝ, f x ≤ 1),
  sorry
end

end find_m_l360_360540


namespace AM_perpendicular_BC_l360_360895

-- Geometry entities and conditions
variables (A B C D E F G M: Point)
variable (BC: Segment)
variable (AB: Line)
variable (AC: Line)
variables (DF EG: Line)
variables (point_on_semicircle_D: Point)
variables (point_on_semicircle_E: Point)

-- Conditions
axiom BC_diameter_of_semicircle : Segment.diameter BC
axiom D_intersects_AB_semicircle : Point.on_semicircle point_on_semicircle_D AB BC
axiom E_intersects_AC_semicircle : Point.on_semicircle point_on_semicircle_E AC BC
axiom D_perpendicular_BC : Perpendicular D BC F
axiom E_perpendicular_BC : Perpendicular E BC G
axiom DG_EF_intersect_at_M : Intersect DG EF M

-- The theorem to be proved
theorem AM_perpendicular_BC : Perpendicular (Line.join A M) BC :=
sorry

end AM_perpendicular_BC_l360_360895


namespace degree_of_p_l360_360254

-- Define the polynomial
def p : Polynomial ℝ := Polynomial.X ^ 2 - 2 * Polynomial.X + 3

-- State the degree of the polynomial
theorem degree_of_p : degree p = 2 := sorry

end degree_of_p_l360_360254


namespace new_tax_rate_l360_360354

theorem new_tax_rate (income : ℝ) (original_rate : ℝ) (savings : ℝ) (new_rate : ℝ) :
  income = 45000 ∧ original_rate = 0.40 ∧ savings = 3150 ∧ new_rate = (14850 / 45000 * 100) → new_rate = 33 :=
by
  intros h
  cases h with h_income h1
  cases h1 with h_original_rate h2
  cases h2 with h_savings h_new_rate
  rw [h_income, h_original_rate, h_savings] at h_new_rate
  exact h_new_rate

end new_tax_rate_l360_360354


namespace least_positive_integer_divisors_l360_360389

theorem least_positive_integer_divisors (n m k : ℕ) (h₁ : (∀ d : ℕ, d ∣ n ↔ d ≤ 2023))
(h₂ : n = m * 6^k) (h₃ : (∀ d : ℕ, d ∣ 6 → ¬(d ∣ m))) : m + k = 80 :=
sorry

end least_positive_integer_divisors_l360_360389


namespace max_distance_sum_l360_360038

theorem max_distance_sum {P : ℝ × ℝ} 
  (C : Set (ℝ × ℝ)) 
  (hC : ∀ (P : ℝ × ℝ), P ∈ C ↔ (P.1 - 3)^2 + (P.2 - 4)^2 = 1)
  (A : ℝ × ℝ := (0, -1))
  (B : ℝ × ℝ := (0, 1)) :
  ∃ P : ℝ × ℝ, 
    P ∈ C ∧ (P = (18 / 5, 24 / 5)) :=
by
  sorry

end max_distance_sum_l360_360038


namespace product_of_roots_cubic_l360_360381

open Polynomial

theorem product_of_roots_cubic :
  Polynomial.roots (X^3 - 6 * X^2 + 11 * X + 30 : Polynomial ℝ).prod = -30 :=
by
  -- Placeholder for the actual proof steps to be filled in
  sorry

end product_of_roots_cubic_l360_360381


namespace non_similar_stars_count_l360_360956

noncomputable def euler_totient (n : ℕ) : ℕ :=
  n * ((1 - (1 / 2)) * (1 - (1 / 3)) * (1 - (1 / 5))).to_rat

theorem non_similar_stars_count (n : ℕ) (h : n = 1440) :
  let phi_n := euler_totient n,
      multiples_of_6 := n / 6
  in (phi_n - multiples_of_6) / 2 = 192 :=
by
  -- Placeholder to satisfy syntactical correctness, the proof will go here.
  sorry

end non_similar_stars_count_l360_360956


namespace solve_equation_l360_360940

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l360_360940


namespace expression_evaluation_l360_360811

theorem expression_evaluation :
  (-2: ℤ)^3 + ((36: ℚ) / (3: ℚ)^2 * (-1 / 2: ℚ)) + abs (-5: ℤ) = -5 :=
by
  sorry

end expression_evaluation_l360_360811


namespace part1_part2_l360_360930

-- Part 1

theorem part1 (f : ℝ → ℝ) (x : ℝ)
  (hf : ∀ x, f(x) = |sin x|) :
  sin 1 ≤ f(x) + f(x + 1) ∧ f(x) + f(x + 1) ≤ 2 * cos (1 / 2) :=
by
  sorry

-- Part 2

theorem part2 (f : ℝ → ℝ)
  (hf : ∀ x, f(x) = |sin x|)
  (n : ℕ) (hn : n > 0) :
  ∑ k in Finset.range (2 * n) + n (λ k, f(n + k) / (n + k + 1)) > sin (1) / 2 :=
by
  sorry

end part1_part2_l360_360930


namespace number_of_distinct_products_l360_360957

   -- We define the set S
   def S : Finset ℕ := {2, 3, 5, 11, 13}

   -- We define what it means to have a distinct product of two or more elements
   def distinctProducts (s : Finset ℕ) : Finset ℕ :=
     (s.powerset.filter (λ t => 2 ≤ t.card)).image (λ t => t.prod id)

   -- We state the theorem that there are exactly 26 distinct products
   theorem number_of_distinct_products : (distinctProducts S).card = 26 :=
   sorry
   
end number_of_distinct_products_l360_360957


namespace max_integer_n_l360_360835

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 2 * 4 ^ (n - 1)

-- Define the sequence T_n
def T (n : ℕ) : ℝ := Real.log 2 (a 1) + ∑ i in Finset.range n, Real.log 2 (a (i + 1))

-- Define the product involving T_n's
def product_T (n : ℕ) : ℝ :=
  ∏ i in Finset.range (n - 1), (1 - 1 / T (i + 2))

-- Problem statement
theorem max_integer_n (n : ℕ) : product_T n > 100 / 213 → n ≤ 28 := by
  sorry

end max_integer_n_l360_360835


namespace ratio_of_areas_of_equilateral_triangles_in_semi_and_full_circle_l360_360721

theorem ratio_of_areas_of_equilateral_triangles_in_semi_and_full_circle (r : ℝ) :
  let A1 := (sqrt 3 / 4) * ((4 * r / sqrt 3) ^ 2)
  let A2 := (sqrt 3 / 4) * ((r * sqrt 3) ^ 2)
  (A1 / A2) = (64 * sqrt 3 / 27) :=
by
  sorry

end ratio_of_areas_of_equilateral_triangles_in_semi_and_full_circle_l360_360721


namespace quadrilateral_sides_eq_l360_360319

-- Define the conditions
variables {α β : ℝ} -- angles
variables (A B C D K : Type)
variables [HasAngle A B K α] [HasAngle A B A α] [HasAngle B C A α] [HasAngle D A C β] [HasAngle D C A β]
-- Assume inscribed quadrilateral and segments condition
variables (inscribed : IsInscribedQuadrilateral A B C D K)

-- The proof statement
theorem quadrilateral_sides_eq (BD CD : Length) :
  A B A = α ∧ B C A = α ∧ C D A = α ∧ D A C = β ∧ D C A = β ∧ inscribed → BD = CD :=
sorry

end quadrilateral_sides_eq_l360_360319


namespace vector_magnitude_l360_360082

open Real

noncomputable def vec_mag_equality 
  (a b : ℝ) 
  (ha : |a| = 2) 
  (θ : ℝ) 
  (hθ : θ = π/3) 
  (hab : |a - 2*b| = 2*sqrt(7)) : Prop :=
  |b| = 3

theorem vector_magnitude :
  ∀ (a b : ℝ), 
  |a| = 2 → 
  ∃ θ (hθ : θ = π/3), 
  |a - 2*b| = 2*sqrt(7) → 
  vec_mag_equality a b |a| (π/3) |a - 2*b| := 
by
  intros a b ha θ hθ hab
  rw hθ
  sorry

end vector_magnitude_l360_360082


namespace angle_A_and_area_of_triangle_l360_360903

theorem angle_A_and_area_of_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (b - c) ^ 2 = a ^ 2 - b * c)
  (h2 : a = 3)
  (h3 : sin C = 2 * sin B)
  (h4 : a / sin A = b / sin B ∧ b / sin B = c / sin C ∧ c / sin C = a / sin A) : 
  (A = π / 3) ∧ 
  (1 / 2 * b * c * sin A = 3 * sqrt (3) / 2) :=
by
  sorry

end angle_A_and_area_of_triangle_l360_360903


namespace function_five_zeros_range_a_l360_360069

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x else a * Real.log x

def g (a : ℝ) (x : ℝ) : ℝ :=
  f(a, x) - f(a, -x)

theorem function_five_zeros_range_a (a : ℝ) :
  (∃ x1 x2 x3 x4 x5 : ℝ, g(a, x1) = 0 ∧ g(a, x2) = 0 ∧ g(a, x3) = 0 ∧ g(a, x4) = 0 ∧ g(a, x5) = 0) ↔ a ∈ Set.Iio (-Real.exp 1) :=
sorry

end function_five_zeros_range_a_l360_360069


namespace rhombus_is_definitely_symmetrical_l360_360748

def is_symmetrical (shape : Type) : Prop :=
  ∃ axis : Type, ∀ (s : shape), s.symmetrical_about axis

def Triangle : Type := sorry -- fill in the precise definition
def Parallelogram : Type := sorry -- fill in the precise definition
def Rhombus : Type := sorry -- fill in the precise definition
def Trapezoid : Type := sorry -- fill in the precise definition

theorem rhombus_is_definitely_symmetrical : 
  (is_symmetrical Rhombus) ∧ ¬(is_symmetrical Triangle) ∧ ¬(is_symmetrical Parallelogram) ∧ ¬(is_symmetrical Trapezoid) := 
  by
    sorry

end rhombus_is_definitely_symmetrical_l360_360748


namespace circle_radius_l360_360628

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l360_360628


namespace abs_dot_product_l360_360166

variables (a b : ℝ^3)

-- Conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_cross_ab : ℝ := 6

-- Theorem statement
theorem abs_dot_product (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (hcross : ‖a × b‖ = norm_cross_ab) :
  |(a ⬝ b)| = 6 * sqrt 3 :=
by
  sorry

end abs_dot_product_l360_360166


namespace not_equal_d_l360_360304

def frac_14_over_6 : ℚ := 14 / 6
def mixed_2_and_1_3rd : ℚ := 2 + 1 / 3
def mixed_neg_2_and_1_3rd : ℚ := -(2 + 1 / 3)
def mixed_3_and_1_9th : ℚ := 3 + 1 / 9
def mixed_2_and_4_12ths : ℚ := 2 + 4 / 12
def target_fraction : ℚ := 7 / 3

theorem not_equal_d : mixed_3_and_1_9th ≠ target_fraction :=
by sorry

end not_equal_d_l360_360304


namespace maximal_set_size_l360_360142

theorem maximal_set_size (n : ℕ) (h1 : n ≥ 2) (S : Finset ℕ) :
  (∀ x y ∈ S, ¬ (x ∣ y ∧ x ≠ y) ∧ (Nat.gcd x y > 1)) →
  ∃ k, k = ⌊(n + 2) / 4⌋ ∧ S.card ≤ k :=
by
  sorry

end maximal_set_size_l360_360142


namespace repeating_decimal_sum_l360_360399

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l360_360399


namespace find_p_b_l360_360052

noncomputable def p (X : set ℝ) : ℝ := sorry
def p_given (X Y : set ℝ) : ℝ := p (X ∩ Y) / p Y

variable (a b : set ℝ)
variable (cond_a : p a = 2 / 15)
variable (cond_b_given_a : p_given b a = 3)
variable (cond_union : p (a ∪ b) = 6 / 15)

theorem find_p_b (a b : set ℝ) (cond_a : p a = 2 / 15) (cond_b_given_a : p_given b a = 3)
  (cond_union : p (a ∪ b) = 6 / 15) : p b = 2 / 3 :=
sorry

end find_p_b_l360_360052


namespace expr_undefined_iff_l360_360019

theorem expr_undefined_iff (x : ℝ) : (x^2 - 9 = 0) ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end expr_undefined_iff_l360_360019


namespace abs_dot_product_l360_360167

variables (a b : ℝ^3)

-- Conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_cross_ab : ℝ := 6

-- Theorem statement
theorem abs_dot_product (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (hcross : ‖a × b‖ = norm_cross_ab) :
  |(a ⬝ b)| = 6 * sqrt 3 :=
by
  sorry

end abs_dot_product_l360_360167


namespace perpendicular_vectors_m_val_angle_when_m_minus_one_l360_360520

section
variables (a b : ℝ × ℝ) (m : ℝ)

def is_perpendicular (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2 = 0)

def angle_between (a b : ℝ × ℝ) : ℝ :=
  real.acos ((a.1 * b.1 + a.2 * b.2) / (real.sqrt (a.1^2 + a.2^2) * real.sqrt (b.1^2 + b.2^2)))

theorem perpendicular_vectors_m_val (h : is_perpendicular (1, 2) (-3, m)) : m = 3 / 2 :=
sorry

theorem angle_when_m_minus_one (hm : m = -1) : angle_between (1, 2) (-3, m) = 3 * real.pi / 4 :=
sorry

end

end perpendicular_vectors_m_val_angle_when_m_minus_one_l360_360520


namespace find_point_on_curve_l360_360542

noncomputable def curve (x : ℝ) : ℝ := real.exp (-x)

def tangent_paralle_parallel_line_condition (x : ℝ) : Prop :=
  (deriv curve x) = -2

theorem find_point_on_curve :
  ∃ (x y : ℝ), y = curve x ∧ tangent_paralle_parallel_line_condition x ∧ (x, y) = (-real.log 2, 2) :=
begin
  sorry,
end

end find_point_on_curve_l360_360542


namespace determine_k_l360_360527

theorem determine_k (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by
  sorry

end determine_k_l360_360527


namespace repeating_decimal_sum_l360_360442

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l360_360442


namespace smallest_positive_period_and_center_of_symmetry_max_min_in_interval_l360_360510

def f (x : ℝ) := -2 * Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1

-- Part 1: Prove the smallest positive period and the center of symmetry
theorem smallest_positive_period_and_center_of_symmetry :
  (∀ k : ℤ, ∃ T : ℝ, T = π ∧ (∀ x : ℝ, f (x + T) = f x) ∧ (∀ k : ℤ, (f (2 * (k * π / 2 - π / 12)) = 0))) :=
sorry

-- Part 2: Prove the maximum and minimum value in the given interval
theorem max_min_in_interval :
  (∀ x ∈ set.Icc (-π/6 : ℝ) (π/3 : ℝ), -1 ≤ f x ∧ f x ≤ 2) :=
sorry

end smallest_positive_period_and_center_of_symmetry_max_min_in_interval_l360_360510


namespace problem_statement_l360_360952
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l360_360952


namespace simplify_fraction_lemma_l360_360812

noncomputable def simplify_fraction (a : ℝ) (h : a ≠ 5) : ℝ :=
  (a^2 - 5 * a) / (a - 5)

theorem simplify_fraction_lemma (a : ℝ) (h : a ≠ 5) : simplify_fraction a h = a := by
  sorry

end simplify_fraction_lemma_l360_360812


namespace dynamic_load_L_value_l360_360364

theorem dynamic_load_L_value (T H : ℝ) (hT : T = 3) (hH : H = 6) : 
  (L : ℝ) = (50 * T^3) / (H^3) -> L = 6.25 := 
by 
  sorry 

end dynamic_load_L_value_l360_360364


namespace graphs_symmetric_respect_to_x_equals_1_l360_360802

-- Define the function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x-1)
def g (x : ℝ) : ℝ := f (x - 1)

-- Define h(x) = f(1 - x)
def h (x : ℝ) : ℝ := f (1 - x)

-- The theorem that their graphs are symmetric with respect to the line x = 1
theorem graphs_symmetric_respect_to_x_equals_1 :
  ∀ x : ℝ, g f x = h f x ↔ f x = f (2 - x) :=
sorry

end graphs_symmetric_respect_to_x_equals_1_l360_360802


namespace sum_a_b_inequality_condition_l360_360032

-- Definitions and conditions
def a_seq (r : ℝ) : ℕ → ℝ
| 0 => 0
| n => 2^(n-1)

def b_seq {r : ℝ} (n : ℕ) := 2 (1 + real.log (2^(n-1)))

def T_sum {r : ℝ} (n : ℕ) := (n-1) * 2^(n+1) + 2

def k_val := (3 : ℝ) / 4 * real.sqrt 2

-- Questions

-- 1. Sum of terms a_n * b_n
theorem sum_a_b (r : ℝ) (n : ℕ) : 
  let a := a_seq r;
  let b := b_seq;
  T_sum n = ∑ i in finset.range n, a i * b i :=
sorry

-- 2. Inequality condition
theorem inequality_condition {r : ℝ} (n : ℕ) : 
  let b := b_seq;
  (∏ i in finset.range n, (1 + b i) / b i) ≥ k_val * real.sqrt((n + 1) : ℝ) :=
sorry

end sum_a_b_inequality_condition_l360_360032


namespace area_of_R_l360_360139

def R :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 + floor(p.1) + floor(p.2) ≤ 5}

def area (s : set (ℝ × ℝ)) : ℝ := sorry -- Definition of area, generally involves measure theory

theorem area_of_R :
  area R = 9 / 2 :=
sorry

end area_of_R_l360_360139


namespace derivative_f_at_1_l360_360479

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * f' 1

theorem derivative_f_at_1 : deriv f 1 = -2 :=
by
  sorry

end derivative_f_at_1_l360_360479


namespace eval_floor_neg_7_div_2_l360_360829

def floor (x : ℝ) : ℤ := 
  if h : ∃ z : ℤ, (z : ℝ) = x then h.some
  else ⌊x⌋

theorem eval_floor_neg_7_div_2 : floor (-7 / 2) = -4 := by
  -- Proof steps would go here
  sorry

end eval_floor_neg_7_div_2_l360_360829


namespace donna_soda_crates_l360_360827

def soda_crates (bridge_limit : ℕ) (truck_empty : ℕ) (crate_weight : ℕ) (dryer_weight : ℕ) (num_dryers : ℕ) (truck_loaded : ℕ) (produce_ratio : ℕ) : ℕ :=
  sorry

theorem donna_soda_crates :
  soda_crates 20000 12000 50 3000 3 24000 2 = 20 :=
sorry

end donna_soda_crates_l360_360827


namespace largest_angle_of_triangle_l360_360563

noncomputable def triangle_angle_largest (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  cos A = 3/4 ∧
  C = 2 * A ∧
  A < C ∧ B < C

theorem largest_angle_of_triangle (a b c : ℝ) (A B C : ℝ) :
  A + B + C = π →
  0 < a → 0 < b → 0 < c →
  cos A = 3/4 →
  C = 2 * A →
  triangle_angle_largest A B C a b c :=
by
  intros hABC ha hb hc hcosA hCA
  -- We should prove that C is the largest angle under given conditions
  have h1 : A < C, sorry
  have h2 : B < C, sorry
  exact ⟨hABC, ha, hb, hc, hcosA, hCA, h1, h2⟩

end largest_angle_of_triangle_l360_360563


namespace hilary_big_toenails_count_l360_360954

def fit_toenails (total_capacity : ℕ) (big_toenail_space_ratio : ℕ) (current_regular : ℕ) (additional_regular : ℕ) : ℕ :=
  (total_capacity - (current_regular + additional_regular)) / big_toenail_space_ratio

theorem hilary_big_toenails_count :
  fit_toenails 100 2 40 20 = 10 :=
  by
    sorry

end hilary_big_toenails_count_l360_360954


namespace part1_part2_l360_360904

variables {f : ℝ → ℝ} (m : ℝ)

-- Conditions
def odd_function (f : ℝ → ℝ) := ∀ x ∈ [-1,1], f(-x) = -f(x)
def increasing_function (f : ℝ → ℝ) := ∀ x₁ x₂ ∈ [-1,1], x₁ < x₂ → f(x₁) < f(x₂)
def condition1 := odd_function f
def condition2 : f 1 = 1 := sorry
def condition3 : ∀ a b ∈ [-1,1], a + b ≠ 0 → (f(a) + f(b)) / (a + b) > 0 := sorry

-- Proof problem statement for part (1)
theorem part1 : condition1 f → condition2 → condition3 → increasing_function f := sorry

-- Proof problem statement for part (2)
theorem part2 : (∃ x ∈ [-1,1], ∀ a ∈ [-1,1], f(x) ≥ m^2 - 2*a*m - 2) ↔ (m ∈ [-1,1]) := sorry

end part1_part2_l360_360904


namespace alexander_eq_alice_l360_360664

-- Definitions and conditions
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.07

-- Calculation functions for Alexander and Alice
def alexander_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let taxed_price := price * (1 + tax)
  let discounted_price := taxed_price * (1 - discount)
  discounted_price

def alice_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

-- Proof that the difference between Alexander's and Alice's total is 0
theorem alexander_eq_alice : 
  alexander_total original_price discount_rate sales_tax_rate = 
  alice_total original_price discount_rate sales_tax_rate :=
by
  sorry

end alexander_eq_alice_l360_360664


namespace smallest_Y_l360_360147

theorem smallest_Y (S : ℕ) (h1 : (∀ d ∈ S.digits 10, d = 0 ∨ d = 1)) (h2 : 18 ∣ S) : 
  (∃ (Y : ℕ), Y = S / 18 ∧ ∀ (S' : ℕ), (∀ d ∈ S'.digits 10, d = 0 ∨ d = 1) → 18 ∣ S' → S' / 18 ≥ Y) → 
  Y = 6172839500 :=
sorry

end smallest_Y_l360_360147


namespace smallest_real_in_domain_of_gg_l360_360099

def g (x : ℝ) : ℝ := real.sqrt (x - 2)

theorem smallest_real_in_domain_of_gg :
  ∃ x ∈ ℝ, (∀ t, g (g t) = real.sqrt (real.sqrt (t - 2) - 2) → t ≥ x) ∧ x = 18 :=
by
  sorry

end smallest_real_in_domain_of_gg_l360_360099


namespace PisotNumberIrreducible_l360_360196

def hasProperty (P : Polynomial ℤ) : Prop :=
  (∀ z : ℂ, P.eval z = 0 → abs z < 1 ∨ abs z > 1) ∧
  (∃ z : ℂ, P.eval z = 0 ∧ abs z > 1) ∧
  P.eval 0 ≠ 0

theorem PisotNumberIrreducible 
  (P : Polynomial ℤ) 
  (h1 : P.coeff 0 ≠ 0) 
  (h2 : ∀ z : ℂ, P.eval z = 0 → abs z < 1 ∨ abs z > 1) 
  (h3 : ∃ z : ℂ, P.eval z = 0 ∧ abs z > 1):
  irreducible P := 
by 
  sorry

end PisotNumberIrreducible_l360_360196


namespace slope_through_A_and_B_l360_360484

theorem slope_through_A_and_B :
  let k := (B.2 - A.2) / (B.1 - A.1)
  in (A = (0,0)) →
     (B = (1,-1)) →
     k = -1 :=
by
  intro k A B hA hB
  subst hA
  subst hB
  unfold k
  simp
  have : (-1 - 0) / (1 - 0) = -1 := by norm_num
  exact this

end slope_through_A_and_B_l360_360484


namespace solve_for_x_l360_360528

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 :=
by
  sorry

end solve_for_x_l360_360528


namespace clock_angle_at_7_oclock_l360_360703

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l360_360703


namespace dot_product_magnitude_l360_360173

variables {a b : EuclideanSpace 3 ℝ}

/- Given conditions -/
def norm_a : ℝ := ‖a‖ = 3
def norm_b : ℝ := ‖b‖ = 4
def norm_cross : ℝ := ‖a × b‖ = 6

/- Desired property to prove -/
theorem dot_product_magnitude :
  norm_a →
  norm_b →
  norm_cross →
  |(a ⋅ b)| = 6 * real.sqrt 3 :=
by
  sorry

end dot_product_magnitude_l360_360173


namespace sum_of_roots_of_quadratic_eq_l360_360726

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l360_360726


namespace max_possible_distance_l360_360123

noncomputable def max_distance (A P : ℝ × ℝ) : ℝ := 
  dist A P 

def point_A_position (W X Y Z P : ℝ × ℝ) : ℝ × ℝ :=
  -- Position calculation based on W, X, Y, Z and the point 
  -- P being the center of the larger square.
  sorry

def points (WX_edge : ℝ) : 
  ℝ × ℝ := 
  -- Defining the positions of points W, X, Y, Z
  sorry

theorem max_possible_distance 
  (side_small_sq side_large_sq : ℝ)
  (W X Y Z P : ℝ × ℝ) : 
  side_small_sq = 2 → 
  side_large_sq = 6 → 
  P = (0, 0) → 
  let ABCD := point_A_position W X Y Z P in
  max_distance (fst ABCD) P = 6 :=
begin
  intros h1 h2 h3,
  sorry
end

end max_possible_distance_l360_360123


namespace average_speed_l360_360100

theorem average_speed (D : ℝ) (hD: D > 0) :
  let t1 := D / (3 * 60) in
  let t2 := D / (3 * 24) in
  let t3 := D / (3 * 48) in
  let total_time := t1 + t2 + t3 in
  let average_speed := D / total_time in
  average_speed = 720 / 19 :=
by
  sorry

end average_speed_l360_360100


namespace roof_collapse_l360_360375

theorem roof_collapse (roof_limit : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) :
  roof_limit = 500 → leaves_per_day = 100 → leaves_per_pound = 1000 → 
  let d := (roof_limit * leaves_per_pound) / leaves_per_day in d = 5000 :=
by
  intros h₁ h₂ h₃
  sorry

end roof_collapse_l360_360375


namespace largest_power_of_2_dividing_32_factorial_ones_digit_l360_360851

def power_of_two_ones_digit (n: ℕ) : ℕ :=
  let digits_cycle := [2, 4, 8, 6]
  digits_cycle[(n % 4) - 1]

theorem largest_power_of_2_dividing_32_factorial_ones_digit :
  power_of_two_ones_digit 31 = 8 := by
  sorry

end largest_power_of_2_dividing_32_factorial_ones_digit_l360_360851


namespace repeating_decimals_sum_l360_360417

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l360_360417


namespace seventy_seventh_digit_is_three_l360_360103

-- Define the sequence of digits from the numbers 60 to 1 in decreasing order.
def sequence_of_digits : List Nat :=
  (List.range' 1 60).reverse.bind (fun n => n.digits 10)

-- Define a function to get the nth digit from the list.
def digit_at_position (n : Nat) : Option Nat :=
  sequence_of_digits.get? (n - 1)

-- The statement to prove
theorem seventy_seventh_digit_is_three : digit_at_position 77 = some 3 :=
sorry

end seventy_seventh_digit_is_three_l360_360103


namespace xiao_ming_stones_total_l360_360750

theorem xiao_ming_stones_total :
  ∑ i in (finset.range 8).map (λ x, x + 1), i = 36 →
  ∑ i in (finset.range 8), 2 ^ (i + 1) = 510 :=
by
  intros h1 h2
  sorry

end xiao_ming_stones_total_l360_360750


namespace complex_number_solution_l360_360534

theorem complex_number_solution (a b : ℝ) (z : ℂ) (h1 : z = a + b * complex.I) 
(h2 : complex.abs z + z = 5 + (√3) * complex.I) 
(h3 : b = √3) : z = 11 / 5 + (√3) * complex.I := by
  sorry

end complex_number_solution_l360_360534


namespace arithmetic_root_of_expression_l360_360501

theorem arithmetic_root_of_expression 
  (x y : ℤ)
  (h1 : sqrt (x + 7) = 3)
  (h2 : cbrt (2 * x - y - 13) = -2):
  sqrt (5 * x - 6 * y) = 4 := 
sorry

end arithmetic_root_of_expression_l360_360501


namespace max_ballpoint_pens_l360_360326

def ballpoint_pen_cost : ℕ := 10
def gel_pen_cost : ℕ := 30
def fountain_pen_cost : ℕ := 60
def total_pens : ℕ := 20
def total_cost : ℕ := 500

theorem max_ballpoint_pens : ∃ (x y z : ℕ), 
  x + y + z = total_pens ∧ 
  ballpoint_pen_cost * x + gel_pen_cost * y + fountain_pen_cost * z = total_cost ∧ 
  1 ≤ x ∧ 
  1 ≤ y ∧
  1 ≤ z ∧
  ∀ x', ((∃ y' z', x' + y' + z' = total_pens ∧ 
                    ballpoint_pen_cost * x' + gel_pen_cost * y' + fountain_pen_cost * z' = total_cost ∧ 
                    1 ≤ x' ∧ 
                    1 ≤ y' ∧
                    1 ≤ z') → x' ≤ x) :=
  sorry

end max_ballpoint_pens_l360_360326


namespace positive_diff_two_largest_prime_factors_l360_360717

theorem positive_diff_two_largest_prime_factors (a b c d : ℕ) (h : 178469 = a * b * c * d) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) 
  (hle1 : a ≤ b) (hle2 : b ≤ c) (hle3 : c ≤ d):
  d - c = 2 := by sorry

end positive_diff_two_largest_prime_factors_l360_360717


namespace right_triangle_point_condition_l360_360194

theorem right_triangle_point_condition 
  (a b x c s : ℝ) (BC_eq_a : BC = a) (AB_eq_b : AB = b) (AP_eq_x : AP = x) 
  (PB_eq_b_minus_x : PB = b - x) (AC_eq_c : AC = c) 
  (angle_B_eq_90 : ∠ B = 90) 
  (s_eq_sum: s = x^2 + (b - x)^2 + (a / b * x)^2) 
  (BP_sq_eq: 2 * (b - x)^2 = 2 * (b ^ 2 - 2 * b * x + x^2)) :
  s = 2 * (b - x)^2 ↔ x = b^2 / sqrt (a^2 + 2 * b^2) :=
begin
  sorry
end

end right_triangle_point_condition_l360_360194


namespace no_infinite_positive_sequence_l360_360308

theorem no_infinite_positive_sequence (x : ℕ → ℝ) (h_pos : ∀ n, x n > 0) 
  (h_relation : ∀ n, x (n + 2) = real.sqrt (x (n + 1)) - real.sqrt (x n)) :
  false := 
by
  sorry

end no_infinite_positive_sequence_l360_360308


namespace dot_product_magnitude_l360_360182

variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜 → 𝕜} 

-- Define the conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_a_cross_b : ℝ := 6

-- Theorem statement
theorem dot_product_magnitude :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∥a × b∥ = norm_a_cross_b) → abs ((a.toReal).dot (b.toReal)) = 6 * sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end dot_product_magnitude_l360_360182


namespace sequence_limit_3_l360_360140

-- Define the sequence
def sequence (x : ℕ → ℝ) := 
  x 1 = 1 ∧ x 2 = 9 ∧ x 3 = 9 ∧ x 4 = 1 ∧ 
  ∀ n ≥ 1, x (n + 4) = (x n * x (n + 1) * x (n + 2) * x (n + 3))^(1/4)

-- Prove that the sequence converges to 3
theorem sequence_limit_3 {x : ℕ → ℝ} (hx : sequence x) :
  (tendsto x at_top (nhds 3)) :=
  sorry

end sequence_limit_3_l360_360140


namespace eval_sum_of_logs_tan_l360_360393

noncomputable def sum_of_logs_tan : ℝ :=
  (list.range (29)).map (λ k, real.log 2 (real.tan (real.pi / 60 * (k + 1)))).sum

theorem eval_sum_of_logs_tan :
  sum_of_logs_tan = 0 :=
sorry

end eval_sum_of_logs_tan_l360_360393


namespace ellipse_equation_1_ellipse_equation_2_k1_ellipse_equation_2_k2_l360_360875

noncomputable def first_condition := (c : ℝ) (e : ℝ) (h1 : c = 6) (h2 : e = 2 / 3) : ℝ × ℝ :=
  let a := c / e in
  let b2 := a^2 - c^2 in
  (a, b2)

noncomputable def second_condition (a : ℝ) (c : ℝ) (k : ℝ) (h1: a = 2*k) (h2 : c = sqrt 3 * k) : ℝ × ℝ :=
  let b := sqrt (a^2 - c^2) in
  (a, b)

theorem ellipse_equation_1 : 
  ∃ a b, (first_condition 6 (2 / 3) rfl rfl) = (a, b) ∧ 
  (b = 45) ∧ 
  (a = 9) ∧
  (a^2 = b + c^2) ∧ 
  (c = 6) → 
  (x y : ℝ), (y^2 / 81 + x^2 / 45 = 1) :=
sorry

theorem ellipse_equation_2_k1 : 
  ∃ a b, (second_condition (2 * 1) (sqrt 3 * 1) 1 rfl rfl) = (a, b) ∧ 
  (b = 1) ∧ 
  (a = 2) →
  (x y : ℝ), (x^2 / 4 + y^2 = 1) :=
sorry

theorem ellipse_equation_2_k2 : 
  ∃ a b, (second_condition (2 * 2) (sqrt 3 * 2) 2 rfl rfl) = (a, b) ∧ 
  (b = 2) ∧ 
  (a = 4) →
  (x y : ℝ), (y^2 / 16 + x^2 / 4 = 1) :=
sorry

end ellipse_equation_1_ellipse_equation_2_k1_ellipse_equation_2_k2_l360_360875


namespace archer_arrows_weekly_expenditure_l360_360361

/-- An archer shoots a certain number of shots and recovers a percentage of arrows, each with a cost.
Given the team contribution, calculate the archer's weekly expenditure on arrows. -/
theorem archer_arrows_weekly_expenditure
    (shots_per_day : ℕ)
    (days_per_week : ℕ)
    (recovery_rate : ℝ)
    (cost_per_arrow : ℝ)
    (team_contribution_rate : ℝ)
    (archer_expenditure_per_week : ℝ) :
    shots_per_day = 200 ∧
    days_per_week = 4 ∧
    recovery_rate = 0.2 ∧
    cost_per_arrow = 5.5 ∧
    team_contribution_rate = 0.7 ∧
    archer_expenditure_per_week = 1056 → 
    let total_shots_per_week := shots_per_day * days_per_week in
    let recovered_arrows := total_shots_per_week * recovery_rate in
    let used_arrows := total_shots_per_week - recovered_arrows in
    let total_cost_of_arrows := used_arrows * cost_per_arrow in
    let team_contribution := total_cost_of_arrows * team_contribution_rate in
    let archers_cost := total_cost_of_arrows - team_contribution in 
    archers_cost = archer_expenditure_per_week := 
by {
    intros h,
    cases h with hsptd hdptw,
    cases hsptd with hrs hrate hctr,
    cases hrate with hpa htepw,
    have h1 : total_shots_per_week = 200 * 4, from rfl,
    have h2 : recovered_arrows = (200 * 4 : ℝ) * 0.2, from rfl,
    have h3 : used_arrows = (200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2), from rfl,
    have h4 : total_cost_of_arrows = ((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5, from rfl,
    have h5 : team_contribution = (((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5) * 0.7, from rfl,
    have h6 : archers_cost = (((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5) - ((((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5) * 0.7), from rfl,
    exact htepw,
}

end archer_arrows_weekly_expenditure_l360_360361


namespace pi_is_irrational_l360_360745

theorem pi_is_irrational : 
  let numbers := [1/7, real.pi, -1, 0]
  ∃ x ∈ numbers, irrational x ∧ ∀ y ∈ numbers, irrational y → y = real.pi :=
begin
  sorry
end

end pi_is_irrational_l360_360745


namespace p_plus_q_is_32_l360_360253

noncomputable def slope_sums_isosceles_trapezoid : ℚ := 
  let E := (30, 150)
  let H := (31, 159)
  let translated_E := (0, 0)
  let translated_H := (1, 9)
  let relative_prime (a b : ℕ) := ∀ d > 1, d ∣ a → d ∣ b → False
  ∑ m in { 4/5, -1, -5/4, 1 }, |m|

theorem p_plus_q_is_32 :
  ∃ p q : ℕ, relative_prime p q ∧ slope_sums_isosceles_trapezoid = p / q ∧ p + q = 32 :=
begin
  sorry
end

end p_plus_q_is_32_l360_360253


namespace value_of_sum_l360_360968

theorem value_of_sum (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 - 2 * a * b = 2 * a * b) : a + b = 2 ∨ a + b = -2 :=
sorry

end value_of_sum_l360_360968


namespace golden_section_MP_length_l360_360918

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem golden_section_MP_length (MN : ℝ) (hMN : MN = 2) (P : ℝ) 
  (hP : P > 0 ∧ P < MN ∧ P / (MN - P) = (MN - P) / P)
  (hMP_NP : MN - P < P) :
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_MP_length_l360_360918


namespace find_other_number_l360_360287

theorem find_other_number
  (n m lcm gcf : ℕ)
  (h_n : n = 40)
  (h_lcm : lcm = 56)
  (h_gcf : gcf = 10)
  (h_lcm_gcf : lcm * gcf = n * m) : m = 14 :=
by
  sorry

end find_other_number_l360_360287


namespace problem_l360_360969

theorem problem (x : ℝ) (h : x^2 + 5 * x - 990 = 0) : x^3 + 6 * x^2 - 985 * x + 1012 = 2002 :=
sorry

end problem_l360_360969


namespace minimum_value_l360_360655

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

theorem minimum_value : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≥ f (-1) :=
by
  sorry

end minimum_value_l360_360655


namespace eval_floor_neg_7_div_2_l360_360830

def floor (x : ℝ) : ℤ := 
  if h : ∃ z : ℤ, (z : ℝ) = x then h.some
  else ⌊x⌋

theorem eval_floor_neg_7_div_2 : floor (-7 / 2) = -4 := by
  -- Proof steps would go here
  sorry

end eval_floor_neg_7_div_2_l360_360830


namespace repeating_decimals_sum_l360_360429

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l360_360429


namespace repeating_decimal_sum_in_lowest_terms_l360_360435

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l360_360435


namespace absolute_value_inequality_solution_l360_360874

theorem absolute_value_inequality_solution (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := 
sorry

end absolute_value_inequality_solution_l360_360874


namespace repeating_decimals_sum_l360_360428

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l360_360428


namespace sin_theta_tan_theta_iff_first_third_quadrant_l360_360305

open Real

-- Definitions from conditions
def in_first_or_third_quadrant (θ : ℝ) : Prop :=
  (0 < θ ∧ θ < π / 2) ∨ (π < θ ∧ θ < 3 * π / 2)

def sin_theta_plus_tan_theta_positive (θ : ℝ) : Prop :=
  sin θ + tan θ > 0

-- Proof statement
theorem sin_theta_tan_theta_iff_first_third_quadrant (θ : ℝ) :
  sin_theta_plus_tan_theta_positive θ ↔ in_first_or_third_quadrant θ :=
sorry

end sin_theta_tan_theta_iff_first_third_quadrant_l360_360305


namespace maximum_value_of_a_l360_360107

theorem maximum_value_of_a (x : ℝ) (a : ℝ) (h : ∀ x ∈ (1,2), x^2 - |a| * x + a - 1 > 0) : a ≤ 2 :=
sorry

end maximum_value_of_a_l360_360107


namespace Janice_age_l360_360114

theorem Janice_age (x : ℝ) (h : x + 12 = 8 * (x - 2)) : x = 4 := by
  sorry

end Janice_age_l360_360114


namespace extreme_values_l360_360509

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4 * x + 6

theorem extreme_values :
  (∃ x : ℝ, f x = 34/3 ∧ (x = -2 ∨ x = 4)) ∧
  (∃ x : ℝ, f x = 2/3 ∧ x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 4, f x ≤ 34/3 ∧ 2/3 ≤ f x) :=
by
  sorry

end extreme_values_l360_360509


namespace smallest_n_is_1770_l360_360460

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

def is_smallest_n (n : ℕ) : Prop :=
  n = sum_of_digits n + 1755 ∧ (∀ m : ℕ, (m < n → m ≠ sum_of_digits m + 1755))

theorem smallest_n_is_1770 : is_smallest_n 1770 :=
sorry

end smallest_n_is_1770_l360_360460


namespace sector_area_l360_360247

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 120) : 
  (theta / 360) * π * r^2 = 3 * π :=
by 
  sorry

end sector_area_l360_360247


namespace sum_of_x_such_that_gx_eq_neg3_l360_360203

def g (x : ℝ) : ℝ :=
if x < -3 then 3 * x + 6 else -x^2 - 2 * x + 2

theorem sum_of_x_such_that_gx_eq_neg3 : 
  (Finset.filter (λ x : ℝ, g x = -3) {x | -10 ≤ x ∧ x ≤ 10}.finite_to_set).sum id = 1 :=
by
  -- Conditions given
  have h₁ : { x : ℝ | g x = -3 } ⊆ {1} := sorry,
  -- Summarize the single value of the element set fulfilling the condition
  have h₂ : Finset.filter (λ x : ℝ, g x = -3) {x | -10 ≤ x ∧ x ≤ 10}.finite_to_set = {1} := by 
    finish [h₁],
  -- Sum of the set with valid solutions
  show (Finset.filter (λ x : ℝ, g x = -3) {x | -10 ≤ x ∧ x ≤ 10}.finite_to_set).sum id = 1 by rw [h₂]; simp

end sum_of_x_such_that_gx_eq_neg3_l360_360203


namespace no_nonzero_algebraic_number_with_infinite_n_l360_360390

noncomputable def exists_nonzero_algebraic_number_with_infinite_n (α : ℂ) : Prop :=
  α ≠ 0 ∧ |α| ≠ 1 ∧ (∃ᶠ n in at_top, ∃ β_n : ℂ, β_n ∈ 𝔽 α ∧ β_n^n = α)

theorem no_nonzero_algebraic_number_with_infinite_n :
  ¬ (∃ α : ℂ, exists_nonzero_algebraic_number_with_infinite_n α) :=
sorry

end no_nonzero_algebraic_number_with_infinite_n_l360_360390


namespace billiard_table_distances_l360_360795

noncomputable def ellipse_billiard_distances (a c : ℝ) (h : a > c) : set ℝ :=
  {2 * (a - c), 2 * (a + c), 4 * a}

theorem billiard_table_distances (A B : ℝ) (a c : ℝ) (h1 : a > 0) (h2 : c > 0) (h3 : A = 0) (h4 : B = 2 * c) :
  ellipse_billiard_distances a c h1 = {2 * (a - c), 2 * (a + c), 4 * a} :=
by
  sorry

end billiard_table_distances_l360_360795


namespace calculate_total_votes_l360_360124

-- Conditions
variables (V : ℕ) -- V is the total number of votes cast
variables (P : ℝ) -- percentage of valid votes: 0.95
variables (W : ℝ) (S : ℝ) (M : ℕ) -- W is 45% as a fraction, S is 35% as a fraction, and M is the majority

-- Constants
def percent_valid : ℝ := 0.95
def winner_fraction : ℝ := 0.45
def second_fraction : ℝ := 0.35
def majority_votes : ℕ := 285

-- Main Statement
theorem calculate_total_votes (h1 : percent_valid = 0.95)
                              (h2 : winner_fraction = 0.45)
                              (h3 : second_fraction = 0.35)
                              (h4 : majority_votes = 285)
                              (H : V ∈ ℕ) : 
  let valid_votes := percent_valid * V in
  let winning_votes := winner_fraction * valid_votes in
  let second_votes := second_fraction * valid_votes in
  winning_votes - second_votes = majority_votes → 
  V = 3000 :=
sorry

end calculate_total_votes_l360_360124


namespace find_minimum_l360_360749

theorem find_minimum (a b c : ℝ) : ∃ (m : ℝ), m = min a (min b c) := 
  sorry

end find_minimum_l360_360749


namespace correct_proposition_l360_360042

-- Definitions of the problem conditions
variable (m n : Line) (α β : Plane)

-- Proposition 1: m ⊥ α, n ⊥ β, m ⊥ n ⊢ α ⊥ β
axiom perp1 : m ⊥ α → n ⊥ β → m ⊥ n → α ⊥ β
-- Proposition 2: m ‖ α, n ‖ β, m ‖ n ⊢ α ‖ β
axiom parallel1 : m ‖ α → n ‖ β → m ‖ n → α ‖ β → false
-- Proposition 3: m ⊥ α, n ‖ β, m ⊥ n ⊢ α ⊥ β
axiom perp2 : m ⊥ α → n ‖ β → m ⊥ n → α ⊥ β → false
-- Proposition 4: m ⊥ α, n ‖ β, m ‖ n ⊢ α ‖ β
axiom parallel2 : m ⊥ α → n ‖ β → m ‖ n → α ‖ β → false

-- Proof problem
theorem correct_proposition : (perp1 m α n β true) ∧ 
  ¬(parallel1 m α n β true) ∧ 
  ¬(perp2 m α n β true) ∧ 
  ¬(parallel2 m α n β true) :=
by
  exact (perp1 m α n β true)
  exact (parallel1 m α n β true → false)
  exact (perp2 m α n β true → false)
  exact (parallel2 m α n β true → false)


end correct_proposition_l360_360042


namespace probability_between_21_and_30_l360_360215

/-- 
Given Melinda rolls two standard six-sided dice, 
forming a two-digit number with the two numbers rolled,
prove that the probability of forming a number between 21 and 30, inclusive, is 11/36.
-/
theorem probability_between_21_and_30 :
  let dice := set.range (λ n : ℕ, n + 1) ∩ {n | n ≤ 6},
      form_number (a b : ℕ) := 10 * a + b,
      valid_numbers := {n | 21 ≤ n ∧ n ≤ 30},
      probability (s : set ℕ) := (s.card : ℚ) / 36
  in probability {n | ∃ a b ∈ dice, form_number a b = n} = 11 / 36 :=
by
  sorry

end probability_between_21_and_30_l360_360215


namespace evaluate_floor_of_negative_seven_halves_l360_360832

def floor_of_negative_seven_halves : ℤ :=
  Int.floor (-7 / 2)

theorem evaluate_floor_of_negative_seven_halves :
  floor_of_negative_seven_halves = -4 :=
by
  sorry

end evaluate_floor_of_negative_seven_halves_l360_360832


namespace cubic_sum_inequality_strengthened_cubic_sum_inequality_further_strengthened_cubic_sum_inequality_l360_360492

theorem cubic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) 
  ≥ a^2 + b^2 + c^2 + d^2 := 
begin
  sorry,
end

theorem strengthened_cubic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) 
  ≥ a^2 + b^2 + c^2 + d^2 + (2/9) * ((a-b)^2 + (a-c)^2 + (a-d)^2 + (b-c)^2 + (b-d)^2 + (c-d)^2) := 
begin
  sorry,
end

theorem further_strengthened_cubic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) 
  ≥ a^2 + b^2 + c^2 + d^2 + (1/3) * ((a-b)^2 + (a-c)^2 + (a-d)^2 + (b-c)^2 + (b-d)^2 + (c-d)^2) := 
begin
  sorry,
end

end cubic_sum_inequality_strengthened_cubic_sum_inequality_further_strengthened_cubic_sum_inequality_l360_360492


namespace coplanarity_condition_l360_360606

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c d : V)

-- Defining what it means for points to be coplanar in Lean.
def coplanar (a b c d : V) : Prop :=
∃ (m n : ℝ), a - d = m • (b - d) + n • (c - d)

-- The main theorem statement.
theorem coplanarity_condition :
  (∀ (coeffs : ℝ × ℝ × ℝ × ℝ), coeffs.1 • a + coeffs.2 • b =
    coeffs.3 • c + coeffs.4 • d → coeffs.1 + coeffs.2 = coeffs.3 + coeffs.4) ↔ coplanar a b c d :=
sorry

end coplanarity_condition_l360_360606


namespace dot_product_magnitude_l360_360187

variables {a b : ℝ^3}

-- Norm of vectors
def norm_a := ‖a‖ = 3
def norm_b := ‖b‖ = 4
def cross_product_norm := ‖a × b‖ = 6

theorem dot_product_magnitude (h1 : norm_a) (h2 : norm_b) (h3 : cross_product_norm) : (abs (a • b)) = 6 * real.sqrt 3 :=
sorry

end dot_product_magnitude_l360_360187


namespace smaller_angle_at_seven_oclock_l360_360702

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l360_360702


namespace ratio_of_sums_l360_360351

variable {α : Type} [LinearOrderedField α]

def is_arithmetic_progression (a_n : ℕ → α) (d : α) : Prop :=
∀ n, a_n (n + 1) = a_n n + d

def is_geometric_progression (a1 a2 a4 : α) : Prop :=
a2^2 = a1 * a4

def sum_arithmetic_progression (a1 d : α) (n : ℕ) : α :=
(n:α) / 2 * (2 * a1 + (n - 1) * d)

theorem ratio_of_sums (a1 d : α) :
  d ≠ 0 →
  sum_arithmetic_progression a1 d 4 / sum_arithmetic_progression a1 d 2 = (10 : α) / 3 :=
by
  intros h_d_ne_zero
  have h_geo : is_geometric_progression a1 (a1 + d) (a1 + 3 * d),
  from sorry
  sorry

end ratio_of_sums_l360_360351


namespace min_value_expression_l360_360008

theorem min_value_expression (θ : ℝ) : 
  (frac (1 : ℝ) (2 - Real.cos θ ^ 2) + frac (1 : ℝ) (2 - Real.sin θ ^ 2)) ≥ (4 / 3) :=
sorry

end min_value_expression_l360_360008


namespace jake_weight_l360_360098

theorem jake_weight {J S : ℝ} (h1 : J - 20 = 2 * S) (h2 : J + S = 224) : J = 156 :=
by
  sorry

end jake_weight_l360_360098


namespace range_m_l360_360921

-- Ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Midpoint of two points (symmetric about a line)
def symmetric_about_line (y_line : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  let x1 := A.1 in let y1 := A.2
  let x2 := B.1 in let y2 := B.2
  y_line ((x1 + x2) / 2) = (y1 + y2) / 2

-- Symmetry about the line y = 4x + m
def line (m x : ℝ) : ℝ := 4 * x + m

-- The main result to prove
theorem range_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ symmetric_about_line (line m) A B) →
  -((2 * Real.sqrt 13) / 13) < m ∧ m < (2 * Real.sqrt 13) / 13 :=
by
  sorry

end range_m_l360_360921


namespace range_of_a_if_f_decreasing_l360_360974

theorem range_of_a_if_f_decreasing (a : ℝ) 
  (h : ∀ x ≤ 4, deriv (λ x : ℝ, x^2 + 2 * (a + 1) * x + 2) x ≤ 0) :
  a ≤ -5 :=
by
  sorry

end range_of_a_if_f_decreasing_l360_360974


namespace largest_power_of_2_dividing_32_factorial_ones_digit_l360_360848

def power_of_two_ones_digit (n: ℕ) : ℕ :=
  let digits_cycle := [2, 4, 8, 6]
  digits_cycle[(n % 4) - 1]

theorem largest_power_of_2_dividing_32_factorial_ones_digit :
  power_of_two_ones_digit 31 = 8 := by
  sorry

end largest_power_of_2_dividing_32_factorial_ones_digit_l360_360848


namespace smaller_solution_eq_neg16_l360_360722

theorem smaller_solution_eq_neg16 : 
  (∀ x : ℝ, x^2 + 12 * x - 64 = 0 → x = -16 ∨ x = 4) →
  min (ite (x = -16) 0 (ite (x = 4) 1 2)) = -16 :=
by
  sorry

end smaller_solution_eq_neg16_l360_360722


namespace artwork_collection_l360_360669

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end artwork_collection_l360_360669


namespace proof_problem_l360_360532

theorem proof_problem (x : ℝ) (h1 : x = 3) (h2 : 2 * x ≠ 5) (h3 : x + 5 ≠ 3) 
                      (h4 : 7 - x ≠ 2) (h5 : 6 + 2 * x ≠ 14) :
    3 * x - 1 = 8 :=
by 
  sorry

end proof_problem_l360_360532


namespace smallest_m_l360_360743

theorem smallest_m (m : ℕ) (h_pos : 0 < m) :
  (∀(a b: ℕ), a = 60 ∧ gcd a m * 20 = lcm a m → (m = 3)) :=
by {
  sorry,
}

end smallest_m_l360_360743


namespace dot_product_magnitude_l360_360192

variables {a b : ℝ^3}

-- Norm of vectors
def norm_a := ‖a‖ = 3
def norm_b := ‖b‖ = 4
def cross_product_norm := ‖a × b‖ = 6

theorem dot_product_magnitude (h1 : norm_a) (h2 : norm_b) (h3 : cross_product_norm) : (abs (a • b)) = 6 * real.sqrt 3 :=
sorry

end dot_product_magnitude_l360_360192


namespace repeating_decimals_sum_l360_360424

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l360_360424


namespace remainder_when_divided_by_44_l360_360783

theorem remainder_when_divided_by_44 (N : ℕ) (Q : ℕ) (R : ℕ)
  (h1 : N = 44 * 432 + R)
  (h2 : N = 31 * Q + 5) :
  R = 2 :=
sorry

end remainder_when_divided_by_44_l360_360783


namespace smaller_angle_at_seven_oclock_l360_360700

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l360_360700


namespace calculate_expression_l360_360809

noncomputable def expr1 : ℝ := (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3)
noncomputable def expr2 : ℝ := (2 * Real.sqrt 2 - 1) ^ 2
noncomputable def combined_expr : ℝ := expr1 + expr2

-- We need to prove the main statement
theorem calculate_expression : combined_expr = 8 - 4 * Real.sqrt 2 :=
by
  sorry

end calculate_expression_l360_360809


namespace cos_alpha_plus_beta_l360_360046

variable (α β : ℝ)
variable (hα : Real.sin α = (Real.sqrt 5) / 5)
variable (hβ : Real.sin β = (Real.sqrt 10) / 10)
variable (hα_obtuse : π / 2 < α ∧ α < π)
variable (hβ_obtuse : π / 2 < β ∧ β < π)

theorem cos_alpha_plus_beta : Real.cos (α + β) = Real.sqrt 2 / 2 ∧ α + β = 7 * π / 4 := by
  sorry

end cos_alpha_plus_beta_l360_360046


namespace diana_hours_mon_wed_fri_l360_360825

theorem diana_hours_mon_wed_fri (H : ℕ) :
  let h_tue_thu := 15 * 2,
      h_week := 1800 / 30 in
  H = h_week - h_tue_thu :=
by
  let h_tue_thu := 15 * 2
  let h_week := 1800 / 30
  show H = h_week - h_tue_thu
  sorry

end diana_hours_mon_wed_fri_l360_360825


namespace find_y_from_equation_l360_360877

theorem find_y_from_equation (y : ℕ) 
  (h : (12 ^ 2) * (6 ^ 3) / y = 72) : 
  y = 432 :=
  sorry

end find_y_from_equation_l360_360877


namespace constant_term_expansion_l360_360981

noncomputable def constant_term_in_expansion (n : ℕ) (x : ℝ) : ℝ :=
  ∑ r in Finset.range (n + 1), (Nat.choose n r) * (2^((n - r) / 2) * x^((n - r) / 2)) * ((-1)^r * (1 / x)^(r / 2))

theorem constant_term_expansion (x : ℝ) (h : x ≠ 0) (h_sum : (2^6) = 64) :
  constant_term_in_expansion 6 x = -160 :=
sorry

end constant_term_expansion_l360_360981


namespace sum_of_cubes_l360_360739

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l360_360739


namespace find_x_l360_360821

-- Define the arithmetic-geometric series
def series (x : ℚ) : ℚ := 1 + 7*x + 13*x^2 + 19*x^3 + ∑' n, (6*n + 1)*x^n 

-- Define the condition
def condition (x : ℚ) : Prop := series x = 100

-- State the theorem to prove
theorem find_x (x : ℚ) (h : condition x) : x = 251 / 400 := 
by
  sorry

end find_x_l360_360821


namespace notebook_price_l360_360987

theorem notebook_price (students_buying_notebooks n c : ℕ) (total_students : ℕ := 36) (total_cost : ℕ := 990) :
  students_buying_notebooks > 18 ∧ c > n ∧ students_buying_notebooks * n * c = total_cost → c = 15 :=
by
  sorry

end notebook_price_l360_360987


namespace ones_digit_large_power_dividing_32_factorial_l360_360857

theorem ones_digit_large_power_dividing_32_factorial :
  let n := 32!
  let largestPower := 2^31
  ones_digit largestPower = 8 :=
by
  sorry

end ones_digit_large_power_dividing_32_factorial_l360_360857


namespace arcsin_range_l360_360906

theorem arcsin_range (α : ℝ ) (x : ℝ ) (h₁ : x = Real.cos α) (h₂ : -Real.pi / 4 ≤ α ∧ α ≤ 3 * Real.pi / 4) : 
-Real.pi / 4 ≤ Real.arcsin x ∧ Real.arcsin x ≤ Real.pi / 2 :=
sorry

end arcsin_range_l360_360906


namespace complex_sum_l360_360320

noncomputable def complex_equation (a b : ℝ) : Prop :=
  (1 + Complex.i) * (2 + Complex.i) = a + b * Complex.i

theorem complex_sum :
  ∀ (a b : ℝ), complex_equation a b → a + b = 4 := by
sorry

end complex_sum_l360_360320


namespace clock_angle_at_seven_l360_360694

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l360_360694


namespace smallest_Y_l360_360149

noncomputable def S : ℕ := 111111111000

theorem smallest_Y : ∃ Y : ℕ, Y = S / 18 ∧ Y = 6172839500 := by
  use 6172839500
  split
  · calc
      S / 18 = 111111111000 / 18 := by sorry
    _ = 6172839500 := by sorry
  · exact rfl

end smallest_Y_l360_360149


namespace repeating_decimals_sum_l360_360419

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l360_360419


namespace weight_of_10m_l360_360983

-- Defining the proportional weight conditions
variable (weight_of_rod : ℝ → ℝ)

-- Conditional facts about the weight function
axiom weight_proportional : ∀ (length1 length2 : ℝ), length1 ≠ 0 → length2 ≠ 0 → 
  weight_of_rod length1 / length1 = weight_of_rod length2 / length2
axiom weight_of_6m : weight_of_rod 6 = 14.04

-- Theorem stating the weight of a 10m rod
theorem weight_of_10m : weight_of_rod 10 = 23.4 := 
sorry

end weight_of_10m_l360_360983


namespace sum_of_values_such_that_gx_is_negative_three_l360_360200

def g (x : ℝ) : ℝ :=
if x < -3 then 3 * x + 6 else -x^2 - 2 * x + 2

theorem sum_of_values_such_that_gx_is_negative_three : 
  let S := {x : ℝ | g x = -3} in S.sum = -4 :=
by
  sorry

end sum_of_values_such_that_gx_is_negative_three_l360_360200


namespace problem_statement_l360_360659

-- Definitions for the problem setup
variables (p q r s : ℝ)
def g (x : ℝ) := x^4 + p * x^3 + q * x^2 + r * x + s

-- The polynomial has real coefficients and specific roots
axiom h1 : g (-3 * Complex.I) = 0
axiom h2 : g (3 * Complex.I) = 0
axiom h3 : g (1 + Complex.I) = 0
axiom h4 : g (1 - Complex.I) = 0

-- The proof goal
theorem problem_statement : p + q + r + s = 9 :=
sorry

end problem_statement_l360_360659


namespace power_function_k_plus_alpha_l360_360258

theorem power_function_k_plus_alpha
  (k α : ℝ)
  (H : f x = k * x^α)
  (H_point : f (1/2) = sqrt(2)/2) :
  k + α = 3/2 := sorry

end power_function_k_plus_alpha_l360_360258


namespace total_polled_votes_l360_360309

theorem total_polled_votes (V : ℕ) (H1 : 2 * V / 10 - 3 * V / 10 = -500) : 
    (V : ℕ) + 10 = 844 :=
by
  sorry

end total_polled_votes_l360_360309


namespace repeating_decimals_count_l360_360465

theorem repeating_decimals_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 15 ∧ RepeatingDecimal (n / 18)}.card = 10 :=
by
  sorry

end repeating_decimals_count_l360_360465


namespace new_ratio_of_horses_to_cows_l360_360221

theorem new_ratio_of_horses_to_cows
    (x : ℕ)
    (h1 : 3 > 1)
    (condition_1 : let horses := 3 * x in let cows := x in horses > cows)
    (condition_2 : let horses_after := 3 * x - 15 in let cows_after := x + 15 in horses_after = cows_after + 30) :
    5 > 3 :=
by
    sorry

end new_ratio_of_horses_to_cows_l360_360221


namespace ticket_cost_l360_360804

-- Define the conditions
def armband_cost : ℝ := 15
def num_rides_with_armband : ℝ := 20

-- Define the cost of individual ticket that needs to be proven
def individual_ticket_cost : ℝ := armband_cost / num_rides_with_armband

-- The theorem statement
theorem ticket_cost (armband_cost_eq : armband_cost = 15)
                     (num_rides_eq : num_rides_with_armband = 20) :
                     individual_ticket_cost = 0.75 :=
by
  -- Using the given conditions
  rw [armband_cost_eq, num_rides_eq]
  simp
  sorry

end ticket_cost_l360_360804


namespace repeating_decimals_count_l360_360464

theorem repeating_decimals_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 15 ∧ RepeatingDecimal (n / 18)}.card = 10 :=
by
  sorry

end repeating_decimals_count_l360_360464


namespace repeating_decimal_sum_l360_360438

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l360_360438


namespace adjust_pendulum_clock_l360_360340

-- Definitions of the various conditions and variables
variables (t h t1 T : ℝ)
variables (start_pendulum_clock show_arbitrary_time : Prop)
variables (accurate_hospital_clock : Prop)
variables (consistent_speed : Prop)
variables (record_arrival_departure : Prop)

-- Definitions based on the conditions
def total_time_shown := t
def spending_time_at_hospital := h
def one_way_travel_time := (t - h) / 2
def departure_time_at_hospital := T

-- The statement we need to prove
theorem adjust_pendulum_clock :
  (total_time_shown = t) →
  (spending_time_at_hospital = h) →
  (one_way_travel_time = (t - h) / 2) →
  (departure_time_at_hospital = T) →
  consistent_speed →
  accurate_hospital_clock →
  record_arrival_departure →
  T + one_way_travel_time = T + (t - h) / 2 :=
by
  intros
  sorry

end adjust_pendulum_clock_l360_360340


namespace range_of_m_l360_360500

-- Define the circle equation as a predicate
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

-- Define the center and radius given the form of the circle equation
def center_radius_condition (m : ℝ) : Prop :=
  ∃ c r, ((c.1 - 1/2)^2 + (c.2 + 1/2)^2 = r^2) ∧ (r^2 = 1/2 - m)

-- Define the origin being outside the circle condition
def origin_outside_circle (m : ℝ) : Prop :=
  sqrt (1/2 * 1/2 + 1/2 * 1/2) > sqrt (1/2 - m)

-- The main theorem that proves the range of m
theorem range_of_m (m : ℝ) : (0 < m ∧ m < 1/2) ↔
  (circle_eq 0 0 m → center_radius_condition m → origin_outside_circle m) :=
by
  sorry

end range_of_m_l360_360500


namespace roof_collapse_days_l360_360371

-- Definitions based on the conditions
def roof_capacity_pounds : ℕ := 500
def leaves_per_pound : ℕ := 1000
def leaves_per_day : ℕ := 100

-- Statement of the problem and the result
theorem roof_collapse_days :
  let total_leaves := roof_capacity_pounds * leaves_per_pound in
  let days := total_leaves / leaves_per_day in
  days = 5000 :=
by
  -- To be proven, so we use sorry for now
  sorry

end roof_collapse_days_l360_360371


namespace f_2017_5_l360_360916

-- Define that f is an odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define f as given in the problem
noncomputable def f : ℝ → ℝ := λ x, if 0 ≤ x ∧ x ≤ 1 then x^3 else sorry

axiom f_property : ∀ x : ℝ, f x = f (2 - x)

-- Show the required value
theorem f_2017_5 : f 2017.5 = 1 / 8 :=
by
  sorry

end f_2017_5_l360_360916


namespace arithmetic_sequence_cn_l360_360593

noncomputable def f (x : ℝ) (n : ℕ) : ℝ := (x - 1)^2 + n

def a_n (n : ℕ) : ℝ := f 1 n
def b_n (n : ℕ) : ℝ := max (f (-1) n) (f 3 n)
def c_n (n : ℕ) : ℝ := b_n n - a_n n / b_n n

theorem arithmetic_sequence_cn : ∃ d ≠ 0, ∀ n : ℕ, c_n (n + 1) - c_n n = d :=
by
  sorry

end arithmetic_sequence_cn_l360_360593


namespace intersecting_lines_product_l360_360579

theorem intersecting_lines_product 
  (a b : ℝ)
  (T : Set (ℝ × ℝ)) (S : Set (ℝ × ℝ))
  (hT : T = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ a * x + y - 3 = 0})
  (hS : S = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x - y - b = 0})
  (h_intersect : (2, 1) ∈ T) (h_intersect_S : (2, 1) ∈ S) :
  a * b = 1 := 
by
  sorry

end intersecting_lines_product_l360_360579


namespace minimal_absolute_difference_condition_2030_l360_360007

theorem minimal_absolute_difference_condition_2030 :
  ∃ (a1 a2 ... am b1 b2 ... bn : ℕ), 
    a1 ≥ a2 ∧ a2 ≥ ... ∧ am > 0 ∧ 
    b1 ≥ b2 ∧ b2 ≥ ... ∧ bn > 0 ∧ 
    (a1 + b1) = arg_minimal ∧ 
    (abs(a1 - b1) = 1) := 
sorry

end minimal_absolute_difference_condition_2030_l360_360007


namespace cost_of_fencing_per_meter_l360_360260

def length_breadth_relation (b l : ℕ) : Prop :=
  l = b + 10

def plot_perimeter (l b : ℕ) : ℕ :=
  2 * (l + b)

def cost_per_meter (total_cost perimeter : ℕ) : ℝ :=
  total_cost / perimeter

theorem cost_of_fencing_per_meter
  (b l : ℕ)
  (hc : 5300 = 5300)
  (hl : l = 55)
  (h_length_breadth : length_breadth_relation b l)
  (h_perimeter : plot_perimeter l b = 200) :
  cost_per_meter 5300 200 = 26.5 := by
  sorry

end cost_of_fencing_per_meter_l360_360260


namespace green_apples_more_than_red_apples_l360_360679

noncomputable def num_original_green_apples : ℕ := 32
noncomputable def num_more_red_apples_than_green : ℕ := 200
noncomputable def num_delivered_green_apples : ℕ := 340
noncomputable def num_original_red_apples : ℕ :=
  num_original_green_apples + num_more_red_apples_than_green
noncomputable def num_new_green_apples : ℕ :=
  num_original_green_apples + num_delivered_green_apples

theorem green_apples_more_than_red_apples :
  num_new_green_apples - num_original_red_apples = 140 :=
by {
  sorry
}

end green_apples_more_than_red_apples_l360_360679


namespace count_digit_5_in_list_1_to_54_l360_360327

theorem count_digit_5_in_list_1_to_54 : (finset.range 54).filter (λ n, n.digits 10).count 5 = 10 := sorry

end count_digit_5_in_list_1_to_54_l360_360327


namespace artwork_collection_l360_360670

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end artwork_collection_l360_360670


namespace logic_problem_l360_360898

variable (p q : Prop)

theorem logic_problem (h₁ : ¬ p) (h₂ : p ∨ q) : p = False ∧ q = True :=
by
  sorry

end logic_problem_l360_360898


namespace triangle_perimeter_l360_360561

variable {ι : Type}
variables (A B C M D O : ι)
variable {dist : ι → ι → ℕ}

-- Let the function dist represent the distance between points
-- Conditions
variable (AB_eq_1 : dist A B = 1)
variable (AM_median : AM)  -- AM is the median
variable (BD_bisector : BD) -- BD is the angle bisector
variable (AM_perp_BD : AM ⊥ BD) -- AM is perpendicular to BD

noncomputable def perimeter_triangle {A B C M D O : ι}
  (hAB : dist A B = 1)
  (BD_prop : True) -- since it's just required not to be used in assumptions
  (AM_prop : True)
  (AM_mid : ∀ {X}, dist B M = dist M C = 1) : ℕ :=
  dist A B + dist B C + dist A C

theorem triangle_perimeter
  (hAB : dist A B = 1)
  (hAM_perp_BD : AM ⊥ BD)
  (hAM_mid : ∀ {X}, dist B M = 1 ∧ dist M C = 1) :
  perimeter_triangle hAB (by trivial) (by trivial) hAM_mid = 5 := sorry

end triangle_perimeter_l360_360561


namespace expected_mixed_color_pairs_l360_360681

theorem expected_mixed_color_pairs (n : ℕ) : 
  E[ mixed_color_pairs(n) ] = n^2 / (2 * n - 1) := 
  sorry

end expected_mixed_color_pairs_l360_360681


namespace sum_of_repeating_decimals_l360_360411

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l360_360411


namespace largest_power_of_2_dividing_32_factorial_ones_digit_l360_360850

def power_of_two_ones_digit (n: ℕ) : ℕ :=
  let digits_cycle := [2, 4, 8, 6]
  digits_cycle[(n % 4) - 1]

theorem largest_power_of_2_dividing_32_factorial_ones_digit :
  power_of_two_ones_digit 31 = 8 := by
  sorry

end largest_power_of_2_dividing_32_factorial_ones_digit_l360_360850


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360847

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  let n := 32 in
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i) in 
  (2 ^ k) % 10 = 8 :=
by
  let n := 32
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i)
  
  have h1 : k = 31 := by sorry
  have h2 : (2 ^ 31) % 10 = 8 := by sorry
  
  exact h2

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360847


namespace trigonometric_identity_proof_l360_360964

theorem trigonometric_identity_proof :
  ∀ (α : ℝ), (π - π / 2 < α ∧ α < π) ∧ tan (π - α) = 3 / 4 →
    1 / (sin ((π + α) / 2) * sin ((π - α) / 2)) = 10 :=
by
  intros α h
  sorry

end trigonometric_identity_proof_l360_360964


namespace trigonometric_identity_proof_l360_360806

theorem trigonometric_identity_proof (α : ℝ) :
  (sin (π + α)) ^ 2 - cos (π + α) * cos (-α) + 1 = 2 :=
by
  -- Given conditions
  have h1 : sin (π + α) = -sin α, from sorry,
  have h2 : cos (π + α) = -cos α, from sorry,
  have h3 : cos (-α) = cos α, from sorry,
  have h4 : (sin α) ^ 2 + (cos α) ^ 2 = 1, from sorry,
  sorry

end trigonometric_identity_proof_l360_360806


namespace solve_for_z_l360_360050

open Complex

theorem solve_for_z (z : ℂ) (i : ℂ) (h1 : i = Complex.I) (h2 : z * i = 1 + i) : z = 1 - i :=
by sorry

end solve_for_z_l360_360050


namespace infinite_non_members_l360_360892

theorem infinite_non_members
    (n : ℕ)
    (d : ℕ → ℕ)
    (hn : ∀ i : ℕ, i < n → 0 < d i)
    (h_sum : (finset.range n).sum (λ i, (d i)⁻¹) < 1) :
    ∃ (A : set ℕ), infinite A ∧ (∀ (i : ℕ), i < n → disjoint A {m | ∃ (k : ℕ), m = k * d i + 1}) :=
sorry

end infinite_non_members_l360_360892


namespace boss_contribution_l360_360596

variable (boss_contrib : ℕ) (todd_contrib : ℕ) (employees_contrib : ℕ)
variable (cost : ℕ) (n_employees : ℕ) (emp_payment : ℕ)
variable (total_payment : ℕ)

-- Conditions
def birthday_gift_conditions :=
  cost = 100 ∧
  todd_contrib = 2 * boss_contrib ∧
  employees_contrib = n_employees * emp_payment ∧
  n_employees = 5 ∧
  emp_payment = 11 ∧
  total_payment = boss_contrib + todd_contrib + employees_contrib

-- The proof goal
theorem boss_contribution
  (h : birthday_gift_conditions boss_contrib todd_contrib employees_contrib cost n_employees emp_payment total_payment) :
  boss_contrib = 15 :=
by
  sorry

end boss_contribution_l360_360596


namespace solve_equation_l360_360943

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l360_360943


namespace normal_vector_of_line_cosine_of_angle_between_lines_distance_from_point_to_line_l360_360547

-- Part (1): Prove the normal vector relationship
theorem normal_vector_of_line (A B C : ℝ) (h : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) :
  ∀ (x y : ℝ), ∇(Ax + By + C = 0) = (A, B) := by
sorry

-- Part (2): Prove cosine between two intersecting lines
theorem cosine_of_angle_between_lines (A1 B1 C1 A2 B2 C2 : ℝ)
 (h1 : A1 ≠ 0 ∧ B1 ≠ 0 ∧ C1 ≠ 0)
 (h2 : A2 ≠ 0 ∧ B2 ≠ 0 ∧ C2 ≠ 0) :
  let m1 := (-B1, A1),
      m2 := (-B2, A2) in
  ∀ (θ : ℝ), 
    θ = atan2 (A1 * A2 + B1 * B2) 
               (sqrt (A1^2 + B1^2) * sqrt (A2^2 + B2^2)) := by
sorry

-- Part (3): Prove the distance formula from point to line
theorem distance_from_point_to_line (A B C x0 y0 : ℝ) (h : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) :
  let P0 := (x0, y0) in
  ∀ d : ℝ, 
    d = |A * x0 + B * y0 + C| / sqrt (A^2 + B^2) := by
sorry

end normal_vector_of_line_cosine_of_angle_between_lines_distance_from_point_to_line_l360_360547


namespace consumption_volume_unchanged_l360_360121

-- Define initial parameters
def demand_function (p : ℝ) : ℝ := 100 - p
def marginal_cost : ℝ := 10
def new_demand_function (p : ℝ) : ℝ := 200 - 2 * p

-- State the proof problem
theorem consumption_volume_unchanged :
  let Q_eq_before := demand_function marginal_cost in
  let Q_eq_after := new_demand_function marginal_cost in
  Q_eq_before = Q_eq_after :=
by
  -- Initial equilibrium quantity before embargo
  have h1 : Q_eq_before = demand_function marginal_cost, by rfl
  -- Initial equilibrium quantity after embargo
  have h2 : Q_eq_after = new_demand_function marginal_cost, by rfl
  -- They are both calculated at the same price point, marginal cost
  have h3 : demand_function marginal_cost = new_demand_function marginal_cost, sorry
  exact h3

end consumption_volume_unchanged_l360_360121


namespace art_club_artworks_l360_360675

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end art_club_artworks_l360_360675


namespace parabola_transformation_l360_360264

theorem parabola_transformation :
  ∀ (x y : ℝ), y = 3 * x^2 → (∃ z : ℝ, z = x - 1 ∧ y = 3 * z^2 - 2) :=
by
  sorry

end parabola_transformation_l360_360264


namespace students_not_playing_sports_l360_360993

theorem students_not_playing_sports :
  ∀ (total_students football_players volleyball_players exactly_one_of_two total_not_playing: ℕ),
  total_students = 40 →
  football_players = 20 →
  volleyball_players = 19 →
  exactly_one_of_two = 15 →
  total_not_playing = total_students - (football_players + volleyball_players - (20 + 19 - exactly_one_of_two)) →
  total_not_playing = 13 :=
by
  intros _ _ _ _ _ h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end students_not_playing_sports_l360_360993


namespace find_larger_number_l360_360255

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1500) (h2 : y = 6 * x + 15) : y = 1797 := by
  sorry

end find_larger_number_l360_360255


namespace ones_digit_of_largest_power_of_2_in_20_fact_l360_360871

open Nat

def largest_power_of_2_in_factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else
    let sum_of_powers := ∑ m in range (n+1), m / 2
    sum_of_powers

def ones_digit_of_power_of_2 (exp : ℕ) : ℕ :=
  let cycle := [2, 4, 8, 6]
  cycle[exp % 4]

theorem ones_digit_of_largest_power_of_2_in_20_fact (n : ℕ) (h : n = 20) : 
  ones_digit_of_power_of_2 (largest_power_of_2_in_factorial n) = 4 :=
by
  rw [h]
  have : largest_power_of_2_in_factorial 20 = 18 := by
    -- Insert the calculations for largest_power_of_2_in_factorial here
    sorry
  rw [this]
  have : ones_digit_of_power_of_2 18 = 4 := by
    -- Insert the cycle calculations here
    sorry
  exact this

end ones_digit_of_largest_power_of_2_in_20_fact_l360_360871


namespace arrangement_count_eq_2880_l360_360526

noncomputable def count_arrangements : ℕ :=
(1! * 4! * 5!)

theorem arrangement_count_eq_2880 :
  count_arrangements = 2880 :=
by
  -- Proof omitted
  sorry

end arrangement_count_eq_2880_l360_360526


namespace cotangent_identity_l360_360576

theorem cotangent_identity
  (A B C : ℝ) (a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (triangle_angles_sum : A + B + C = π)
  (angle_a_relation : a = 2 * R * sin A)
  (angle_b_relation : b = 2 * R * sin B)
  (angle_c_relation : c = 2 * R * sin C)
  (R : ℝ) :
  (a - b) * cot (C / 2) + (b - c) * cot (A / 2) + (c - a) * cot (B / 2) = 0 :=
by
  sorry

end cotangent_identity_l360_360576


namespace minimum_value_proof_l360_360914

variables {a b : ℝ}
variables [inner_product_space ℝ V] [normed_space ℝ V]

noncomputable def angle_between_vectors_frac_two_pi_by_three (a b : V) : Prop :=
∡ a b = 2 * real.pi / 3

noncomputable def equal_magnitudes (a b : V) : Prop := 
∥a∥ = ∥a + b∥

noncomputable def min_value {V : Type*} [inner_product_space ℝ V] [normed_space ℝ V]
  (a b : V) : Prop :=
∀ t : ℝ, (∥2 • a + t • b∥ / ∥2 • b∥) ≥ ∥(2 • a + (-1) • b)∥ / ∥2 • b∥

theorem minimum_value_proof (a b : V)
  (h1 : angle_between_vectors_frac_two_pi_by_three a b)
  (h2 : equal_magnitudes a b) : 
min_value a b :=
sorry

end minimum_value_proof_l360_360914


namespace vojta_correct_sum_l360_360289

theorem vojta_correct_sum (S A B C : ℕ)
  (h1 : S + (10 * B + C) = 2224)
  (h2 : S + (10 * A + B) = 2198)
  (h3 : S + (10 * A + C) = 2204)
  (A_digit : 0 ≤ A ∧ A < 10)
  (B_digit : 0 ≤ B ∧ B < 10)
  (C_digit : 0 ≤ C ∧ C < 10) :
  S + 100 * A + 10 * B + C = 2324 := 
sorry

end vojta_correct_sum_l360_360289


namespace sum_K_eq_9801_l360_360584

def K (x : ℕ) : ℕ :=
  (Finset.filter (λ (ab : ℕ × ℕ), ab.1 < x ∧ ab.2 < x ∧ Nat.gcd ab.1 ab.2 = 1) ((Finset.range x).product (Finset.range x))).card

theorem sum_K_eq_9801 :
  (Finset.range 100).sum (λ k, K (100 / (k + 1))) = 9801 := 
sorry

end sum_K_eq_9801_l360_360584


namespace proposition_A_l360_360793

-- Define a predicate for a pure imaginary number
def isPureImaginary (z : ℂ) : Prop :=
  ∃ (y : ℝ), z = complex.I * y

-- The Lean statement reflecting the mathematically equivalent proof problem
theorem proposition_A (z : ℂ) (h : z^2 < 0) : isPureImaginary z :=
  sorry

end proposition_A_l360_360793


namespace largest_power_of_2_dividing_32_factorial_ones_digit_l360_360853

def power_of_two_ones_digit (n: ℕ) : ℕ :=
  let digits_cycle := [2, 4, 8, 6]
  digits_cycle[(n % 4) - 1]

theorem largest_power_of_2_dividing_32_factorial_ones_digit :
  power_of_two_ones_digit 31 = 8 := by
  sorry

end largest_power_of_2_dividing_32_factorial_ones_digit_l360_360853


namespace num_regular_soda_l360_360779

def diet_soda := 53
def apples := 42
def more_regular_than_diet := 26

def regular_soda : ℕ := diet_soda + more_regular_than_diet

theorem num_regular_soda (h1 : diet_soda = 53) (h2 : diet_soda + more_regular_than_diet = 79) : regular_soda = 79 :=
by
  rw [h1, add_comm]
  exact h2

end num_regular_soda_l360_360779


namespace num_valid_pairs_l360_360958

theorem num_valid_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 150 ∧ ((a + 1 / b) / (1 / a + b) = 17) ↔ (a = 17 * b) ∧ b ≤ 8) :=
by
  sorry

end num_valid_pairs_l360_360958


namespace min_fB_value_l360_360986

def is_nice_set (s : set (ℕ × ℕ)) : Prop :=
  ∀ (p1 p2 : ℕ × ℕ), p1 ∈ s → p2 ∈ s → (p1.1 = p2.1 ∨ p1.2 = p2.2) → p1 = p2

def no_standard_rectangle (B E : set (ℕ × ℕ)) : Prop :=
  ∀ (p1 p2 : ℕ × ℕ), p1 ∈ B → p2 ∈ B → p1 ≠ p2 → 
  ∃ (e : ℕ × ℕ), e ∈ E ∧ (e.1 > min p1.1 p2.1 ∧ e.1 < max p1.1 p2.1 ∧ 
                           e.2 > min p1.2 p2.2 ∧ e.2 < max p1.2 p2.2)

theorem min_fB_value (B : set (ℕ × ℕ)) (hB: is_nice_set B) (hB_card: B.card = 2016) :
  ∃ (n : ℕ), n = 2015 ∧ ∀ (E : set (ℕ × ℕ)), is_nice_set (B ∪ E) → no_standard_rectangle B E → E.card = n :=
sorry

end min_fB_value_l360_360986


namespace minimum_value_expression_l360_360091

theorem minimum_value_expression (a b : ℝ) (h : a * b > 0) : 
  ∃ m : ℝ, (∀ x y : ℝ, x * y > 0 → (4 * y / x + (x - 2 * y) / y) ≥ m) ∧ m = 2 :=
by
  sorry

end minimum_value_expression_l360_360091


namespace price_reduction_percentage_price_increase_amount_l360_360331

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end price_reduction_percentage_price_increase_amount_l360_360331


namespace repeating_decimal_sum_in_lowest_terms_l360_360433

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l360_360433


namespace pages_left_to_do_l360_360237

theorem pages_left_to_do (total_problems finished_problems problems_per_page : ℕ) (h_total : total_problems = 60) (h_finished : finished_problems = 20) (h_per_page : problems_per_page = 8) :
  (total_problems - finished_problems) / problems_per_page = 5 :=
by
  rw [h_total, h_finished, h_per_page]
  sorry

end pages_left_to_do_l360_360237


namespace art_club_artworks_l360_360676

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end art_club_artworks_l360_360676


namespace range_of_a_l360_360823

def f (x a : ℝ) : ℝ := a - x + x * Real.exp x

theorem range_of_a (a : ℝ) : (∃ x₀ > -1, f x₀ a ≤ 0) ↔ a ≤ 0 :=
by
  sorry

end range_of_a_l360_360823


namespace decreasing_quadratic_function_l360_360105

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x : ℝ, x ≤ 4 → deriv (λ x, x^2 + 2 * (a - 1) * x + 2) x ≤ 0) ↔ a ≤ -3 := by
  -- We state the main condition in terms of the derivative needing to be non-positive for x ≤ 4
  sorry

end decreasing_quadratic_function_l360_360105


namespace polygon_sum_fractions_less_than_two_l360_360281

theorem polygon_sum_fractions_less_than_two (n : ℕ) (a : Fin n → ℝ)
  (h_triangle_ineq : ∀ i, a i < ∑ j in Finset.univ.erase i, a j) :
  (∑ i, a i / ∑ j in Finset.univ.erase i, a j) < 2 :=
  sorry

end polygon_sum_fractions_less_than_two_l360_360281


namespace green_apples_more_than_red_apples_l360_360680

noncomputable def num_original_green_apples : ℕ := 32
noncomputable def num_more_red_apples_than_green : ℕ := 200
noncomputable def num_delivered_green_apples : ℕ := 340
noncomputable def num_original_red_apples : ℕ :=
  num_original_green_apples + num_more_red_apples_than_green
noncomputable def num_new_green_apples : ℕ :=
  num_original_green_apples + num_delivered_green_apples

theorem green_apples_more_than_red_apples :
  num_new_green_apples - num_original_red_apples = 140 :=
by {
  sorry
}

end green_apples_more_than_red_apples_l360_360680


namespace average_of_175_results_l360_360463

theorem average_of_175_results (x y : ℕ) (hx : x = 100) (hy : y = 75) 
(a b : ℚ) (ha : a = 45) (hb : b = 65) :
  ((x * a + y * b) / (x + y) = 53.57) :=
sorry

end average_of_175_results_l360_360463


namespace radius_of_circle_l360_360638

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l360_360638


namespace max_binomial_coefficient_l360_360128

theorem max_binomial_coefficient (a b : ℕ) (n : ℕ) 
  (h₁ : (a + b)^n = (2:ℕ)^n) 
  (h₂ : 2^8 = 256)
  (h₃ : 8 = n) : 
  nat.choose n (n / 2) = 70 := 
by 
  sorry

end max_binomial_coefficient_l360_360128


namespace card_game_probability_l360_360772

theorem card_game_probability :
  let A_wins := 4;  -- number of heads needed for A to win all cards
  let B_wins := 4;  -- number of tails needed for B to win all cards
  let total_flips := 5;  -- exactly 5 flips
  (Nat.choose total_flips 1 + Nat.choose total_flips 1) / (2^total_flips) = 5 / 16 :=
by
  sorry

end card_game_probability_l360_360772


namespace part1_part2_l360_360924

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a+1)*x + a

theorem part1 (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f a x < 0) → a ≤ -2 := sorry

theorem part2 (a x : ℝ) :
  f a x > 0 ↔
  (a > 1 ∧ (x < -a ∨ x > -1)) ∨
  (a = 1 ∧ x ≠ -1) ∨
  (a < 1 ∧ (x < -1 ∨ x > -a)) := sorry

end part1_part2_l360_360924


namespace smallest_number_gt_sum_digits_1755_l360_360457

theorem smallest_number_gt_sum_digits_1755 :
  ∃ (n : ℕ) (a b c d : ℕ), a ≠ 0 ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ n = (a + b + c + d) + 1755 ∧ n = 1770 :=
by {
  sorry
}

end smallest_number_gt_sum_digits_1755_l360_360457


namespace six_digit_numbers_l360_360084

theorem six_digit_numbers :
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
  sorry

end six_digit_numbers_l360_360084


namespace accurate_value_of_K_l360_360995

-- Define the value of K and the error
def K : ℝ := 3.76982
def error : ℝ := 0.00245

-- Define the upper and lower bounds of K
def K_upper : ℝ := K + error
def K_lower : ℝ := K - error

-- State the theorem to be proven
theorem accurate_value_of_K : (Real.floor (K_upper * 100) / 100 = 3.77) ∧ (Real.floor (K_lower * 100) / 100 = 3.77) :=
  by
  sorry

end accurate_value_of_K_l360_360995


namespace repeating_decimals_sum_l360_360420

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l360_360420


namespace dot_product_magnitude_l360_360193

variables {a b : ℝ^3}

-- Norm of vectors
def norm_a := ‖a‖ = 3
def norm_b := ‖b‖ = 4
def cross_product_norm := ‖a × b‖ = 6

theorem dot_product_magnitude (h1 : norm_a) (h2 : norm_b) (h3 : cross_product_norm) : (abs (a • b)) = 6 * real.sqrt 3 :=
sorry

end dot_product_magnitude_l360_360193


namespace triangle_congruence_l360_360485

-- Define the basic setup: a parallelogram ABCD
variable (A B C D E F G : Point)
variable [LinearOrderedField ℝ] [Plane ℝ]
variable {h1 : Quadrilateral A B C D}
variable { h_parallelogram : Parallelogram A B C D}
variable {h2 : Angle A B D > π / 2}

-- Define the conditions of triangles DCE, BCF, and EFG
variable { h_DCE : Triangle D C E }
variable { h_BCF : Triangle B C F }
variable { h_EDC_eq_CBF : ∠ E D C = ∠ C B F }
variable { h_DCE_eq_BFC : ∠ D C E = ∠ B F C }

-- Define the construction of triangle EFG outside CEF
variable { h_EFG : Triangle E F G }
variable { h_EFG_outside : ConstructedOutsideTriangle E F G C E F}
variable { h_EFG_eq_CFB : ∠ E F G = ∠ C F B }
variable { h_FEG_eq_CED : ∠ F E G = ∠ C E D }

theorem triangle_congruence
  (h_parallel : Parallelogram A B C D)
  (h_ang : ∠ A B D > π / 2)
  (h_tri_DCE : Triangle D C E)
  (h_tri_BCF : Triangle B C F)
  (h_EDC_CBF : ∠ E D C = ∠ C B F)
  (h_DCE_BFC : ∠ D C E = ∠ B F C)
  (h_tri_EFG : Triangle E F G)
  (h_EFG_ext : ConstructedOutsideTriangle E F G C E F)
  (h_cfg_FG_CF : ∠ E F G = ∠ C F B)
  (h_FEG_CE : ∠ F E G = ∠ C E D)
  : CongruentTriangles (Triangle.mk A E F) (Triangle.mk G E F) :=
  sorry
  
end triangle_congruence_l360_360485


namespace dot_product_magnitude_l360_360178

variables {a b : EuclideanSpace 3 ℝ}

/- Given conditions -/
def norm_a : ℝ := ‖a‖ = 3
def norm_b : ℝ := ‖b‖ = 4
def norm_cross : ℝ := ‖a × b‖ = 6

/- Desired property to prove -/
theorem dot_product_magnitude :
  norm_a →
  norm_b →
  norm_cross →
  |(a ⋅ b)| = 6 * real.sqrt 3 :=
by
  sorry

end dot_product_magnitude_l360_360178


namespace four_digit_number_count_l360_360472

theorem four_digit_number_count : 
  (∃ digits : Finset (Fin 4), 
    ∀ d ∈ digits, d.val + 1 ∈ {1, 2, 3, 4} ∧
    digits.card = 4 ∧
    (∃ i j k l : Fin 4, 
      digits = {i,j,k,l} ∧
      set.count i.val {x | x ∈ {1,3}} + 
      set.count j.val {x | x ∈ {1,3}} + 
      set.count k.val {x | x ∈ {1,3}} + 
      set.count l.val {x | x ∈ {1,3}} = 2 ∧
      ((i.val + 1 ∈ {2,4} ∧ j.val + 1 ∈ {1,3} ∧ k.val + 1 ∈ {1,3}) ∨ 
        (j.val + 1 ∈ {2,4} ∧ k.val + 1 ∈ {1,3} ∧ l.val + 1 ∈ {1,3}))
    )
  ).card = 8 :=
begin
  sorry
end

end four_digit_number_count_l360_360472


namespace age_relation_l360_360911

variable (x y z : ℕ)

theorem age_relation (h1 : x > y) : (z > y) ↔ (∃ w, w > 0 ∧ y + z > 2 * x) :=
sorry

end age_relation_l360_360911


namespace archer_arrows_weekly_expenditure_l360_360360

/-- An archer shoots a certain number of shots and recovers a percentage of arrows, each with a cost.
Given the team contribution, calculate the archer's weekly expenditure on arrows. -/
theorem archer_arrows_weekly_expenditure
    (shots_per_day : ℕ)
    (days_per_week : ℕ)
    (recovery_rate : ℝ)
    (cost_per_arrow : ℝ)
    (team_contribution_rate : ℝ)
    (archer_expenditure_per_week : ℝ) :
    shots_per_day = 200 ∧
    days_per_week = 4 ∧
    recovery_rate = 0.2 ∧
    cost_per_arrow = 5.5 ∧
    team_contribution_rate = 0.7 ∧
    archer_expenditure_per_week = 1056 → 
    let total_shots_per_week := shots_per_day * days_per_week in
    let recovered_arrows := total_shots_per_week * recovery_rate in
    let used_arrows := total_shots_per_week - recovered_arrows in
    let total_cost_of_arrows := used_arrows * cost_per_arrow in
    let team_contribution := total_cost_of_arrows * team_contribution_rate in
    let archers_cost := total_cost_of_arrows - team_contribution in 
    archers_cost = archer_expenditure_per_week := 
by {
    intros h,
    cases h with hsptd hdptw,
    cases hsptd with hrs hrate hctr,
    cases hrate with hpa htepw,
    have h1 : total_shots_per_week = 200 * 4, from rfl,
    have h2 : recovered_arrows = (200 * 4 : ℝ) * 0.2, from rfl,
    have h3 : used_arrows = (200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2), from rfl,
    have h4 : total_cost_of_arrows = ((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5, from rfl,
    have h5 : team_contribution = (((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5) * 0.7, from rfl,
    have h6 : archers_cost = (((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5) - ((((200 * 4 : ℝ) - ((200 * 4 : ℝ) * 0.2)) * 5.5) * 0.7), from rfl,
    exact htepw,
}

end archer_arrows_weekly_expenditure_l360_360360


namespace sarahs_packages_l360_360236

def num_cupcakes_before : ℕ := 60
def num_cupcakes_ate : ℕ := 22
def cupcakes_per_package : ℕ := 10

theorem sarahs_packages : (num_cupcakes_before - num_cupcakes_ate) / cupcakes_per_package = 3 :=
by
  sorry

end sarahs_packages_l360_360236


namespace repeating_decimal_sum_l360_360402

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l360_360402


namespace fraction_value_l360_360961

theorem fraction_value (x : ℝ) (h : 1 - 6 / x + 9 / (x^2) = 0) : 2 / x = 2 / 3 :=
  sorry

end fraction_value_l360_360961


namespace charlyn_total_viewable_area_l360_360380

noncomputable def rectangle_length : ℝ := 8
noncomputable def rectangle_width : ℝ := 4
noncomputable def viewing_distance : ℝ := 1.5

def total_viewable_area : ℝ :=
  let inside_area := rectangle_length * rectangle_width
  let reduced_length := rectangle_length - 2 * viewing_distance
  let reduced_width := rectangle_width - 2 * viewing_distance
  let unseen_inner_area := max 0 reduced_length * max 0 reduced_width
  let viewable_inside_area := inside_area - unseen_inner_area

  let outside_length_area := 2 * rectangle_length * viewing_distance
  let outside_width_area := 2 * rectangle_width * viewing_distance
  let corner_circular_area := 4 * (Real.pi * viewing_distance^2 / 4)

  viewable_inside_area + outside_length_area + outside_width_area + corner_circular_area

theorem charlyn_total_viewable_area : total_viewable_area = 70 := sorry

end charlyn_total_viewable_area_l360_360380


namespace repeating_decimal_sum_l360_360401

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l360_360401


namespace tropical_poly_factorization_l360_360199

noncomputable def tropical_add (a b : ℝ) : ℝ := min a b
noncomputable def tropical_mul (a b : ℝ) : ℝ := a + b

def tropical_poly (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  Finset.min (Finset.range (n + 1)) (λ k => a k + k * x)

def factored_form (a : ℝ) (r : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  a + Finset.sum (Finset.range n) (λ i => min x (r i))

theorem tropical_poly_factorization (a : ℕ → ℝ) (n : ℕ) (x : ℝ) (h : a n ≠ ∞) :
  ∃ r : ℕ → ℝ, (tropical_poly a n x = factored_form (a n) r n x) := by
  sorry

end tropical_poly_factorization_l360_360199


namespace range_of_a_l360_360108

theorem range_of_a
  (a : ℝ)
  (h : ∀ x : ℝ, |x + 1| + |x - 3| ≥ a) : a ≤ 4 :=
sorry

end range_of_a_l360_360108


namespace coeff_x2_in_expansion_l360_360642

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

theorem coeff_x2_in_expansion : 
  let expr := (x / y - y / real.sqrt x)^8 in
  ∃ c : ℝ, (c * x^2) ∈ expr.coeff ∧ c = 70 :=
by {
  -- This proof is based on applying the binomial theorem and identifying the coefficient as shown.
  sorry
}

end coeff_x2_in_expansion_l360_360642


namespace no_int_representation_l360_360233

theorem no_int_representation (A B : ℤ) : (99999 + 111111 * Real.sqrt 3) ≠ (A + B * Real.sqrt 3)^2 :=
by
  sorry

end no_int_representation_l360_360233


namespace cybil_ronda_probability_l360_360819

theorem cybil_ronda_probability :
  let total_cards := 15
  let cybil_cards := 6
  let ronda_cards := 9
  let draw_cards := 3
  let total_combinations := Nat.choose total_cards draw_cards
  let no_cybil_combinations := Nat.choose ronda_cards draw_cards
  let no_ronda_combinations := Nat.choose cybil_cards draw_cards
  let p_no_cybil := no_cybil_combinations / total_combinations
  let p_no_ronda := no_ronda_combinations / total_combinations
  let p_at_least_one_each := 1 - p_no_cybil - p_no_ronda
  p_at_least_one_each = (351 / 455) :=
by
  unfold total_cards cybil_cards ronda_cards draw_cards total_combinations no_cybil_combinations no_ronda_combinations p_no_cybil p_no_ronda p_at_least_one_each
  norm_num
  sorry

end cybil_ronda_probability_l360_360819


namespace constant_term_of_polynomial_l360_360145

noncomputable def a : ℝ := ∫ x in 0..Real.pi, (Real.sin x - 1 + 2 * Real.cos (x / 2) ^ 2)

theorem constant_term_of_polynomial : 
  let p := (a * Real.sqrt x - 1 / Real.sqrt x) ^ 6 * (x ^ 2 + 2)
  in constant_term p = -332 := by
  sorry

end constant_term_of_polynomial_l360_360145


namespace nonzero_integer_solutions_l360_360838

-- Define the equation to be solved
def equation (a b : ℤ) : Prop :=
  (a^2 + b) * (a + b^2) = (a - b)^2

-- Define the nonzero condition
def nonzero (a b : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- Assert the main theorem stating the required solutions
theorem nonzero_integer_solutions (a b : ℤ) :
  nonzero a b ∧ equation a b ↔ (a, b) ∈ {(0, 1), (1, 0), (-1, -1), (2, -1), (-1, 2)} :=
sorry

end nonzero_integer_solutions_l360_360838


namespace prove_AB_l360_360505

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 4) = 1

-- Define the condition about the midpoint
def midpoint (A B : ℝ × ℝ) : Prop := 
  ∀ M : ℝ × ℝ, M = (2, 1) → (fst A + fst B) / 2 = 2 ∧ (snd A + snd B) / 2 = 1

-- Define the equation of line AB
def line_equation (A B : ℝ × ℝ) : Prop := ∃ k : ℝ, ∀ x y : ℝ, 
  (y - 1 = k * (x - 2)) → x + 2*y - 4 = 0

-- Define the length of line segment AB
def length_AB (A B : ℝ × ℝ) : ℝ := real.sqrt ((fst B - fst A)^2 + (snd B - snd A)^2)

-- The theorem to prove the correct equation of line AB and its length
theorem prove_AB (A B : ℝ × ℝ) (h1 : ellipse (fst A) (snd A)) (h2 : ellipse (fst B) (snd B)) 
  (hmid : midpoint A B) : 
  line_equation A B ∧ length_AB A B = 2 * real.sqrt 5 := 
by sorry

end prove_AB_l360_360505


namespace abs_dot_product_l360_360172

variables (a b : ℝ^3)

-- Conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_cross_ab : ℝ := 6

-- Theorem statement
theorem abs_dot_product (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (hcross : ‖a × b‖ = norm_cross_ab) :
  |(a ⬝ b)| = 6 * sqrt 3 :=
by
  sorry

end abs_dot_product_l360_360172


namespace sum_gcd_lcm_168_l360_360299

def gcd_54_72 : ℕ := Nat.gcd 54 72

def lcm_50_15 : ℕ := Nat.lcm 50 15

def sum_gcd_lcm : ℕ := gcd_54_72 + lcm_50_15

theorem sum_gcd_lcm_168 : sum_gcd_lcm = 168 := by
  sorry

end sum_gcd_lcm_168_l360_360299


namespace NewYearSeasonMarkup_is_25percent_l360_360784

variable (C N : ℝ)
variable (h1 : N >= 0)
variable (h2 : 0.92 * (1 + N) * 1.20 * C = 1.38 * C)

theorem NewYearSeasonMarkup_is_25percent : N = 0.25 :=
  by
  sorry

end NewYearSeasonMarkup_is_25percent_l360_360784


namespace rectangle_area_difference_196_l360_360218

noncomputable def max_min_area_difference (P : ℕ) (A_max A_min : ℕ) : Prop :=
  ( ∃ l w : ℕ, 2 * l + 2 * w = P ∧ A_max = l * w ) ∧
  ( ∃ l' w' : ℕ, 2 * l' + 2 * w' = P ∧ A_min = l' * w' ) ∧
  (A_max - A_min = 196)

theorem rectangle_area_difference_196 : max_min_area_difference 60 225 29 :=
by
  sorry

end rectangle_area_difference_196_l360_360218


namespace volume_quadrangular_pyramid_l360_360266

-- Definitions based on conditions
variables {α : Type*} [linear_ordered_field α] {P A B C D M : Euclidean_space ℝ (fin 3)}
variables {angle_ABCD : α} {dist_M_to_base : α} {dist_M_to_lateral : α}

-- Condition definitions
def is_rhombus (P A B C D : Euclidean_space ℝ (fin 3)) : Prop :=
  ∃ a b c d, rhombus a b c d ∧ convex_hull ℝ ({P} ∪ {A, B, C, D}) = 
  { x | is_rhomboid x a b c d }

def angle_60_deg (θ : α) : Prop := θ = π / 3

def point_M_equidistant (M P A B C D: Euclidean_space ℝ (fin 3)): Prop :=
  dist_M_to_base = dist_euclidean M (convex_hull ℝ {A, B, C, D})
  ∧ ∀ (X ∈ {A, B, C, D}), dist_M_to_lateral = dist_euclidean M X

-- Volume Statement
noncomputable def pyramid_volume {P A B C D : Euclidean_space ℝ (fin 3)}
  (h_rhombus: is_rhombus P A B C D)
  (h_angle_base: angle_60_deg angle_ABCD)
  (h_point_M: point_M_equidistant M P A B C D)
  : α := 
  8 * real.sqrt 3

-- Goal Statement
theorem volume_quadrangular_pyramid
  (h_rhombus: is_rhombus P A B C D)
  (h_angle_base: angle_60_deg angle_ABCD)
  (h_point_M: point_M_equidistant M P A B C D)
  : pyramid_volume h_rhombus h_angle_base h_point_M = 8 * real.sqrt 3 :=
begin
  sorry -- Proof of the volume calculation
end

end volume_quadrangular_pyramid_l360_360266


namespace sum_of_roots_l360_360734

theorem sum_of_roots (a b c: ℝ) (h: a ≠ 0) (h_eq : a = 1) (h_eq2 : b = -6) (h_eq3 : c = 8):
    let Δ := b ^ 2 - 4 * a * c in
    let root1 := (-b + real.sqrt Δ) / (2 * a) in
    let root2 := (-b - real.sqrt Δ) / (2 * a) in
    root1 + root2 = 6 :=
by
  sorry

end sum_of_roots_l360_360734


namespace equilateral_tangent_triangle_area_l360_360475

theorem equilateral_tangent_triangle_area (R : ℝ) : 
  ∃ (A B C O : Point), 
    (is_tangent A B O) ∧
    (is_tangent A C O) ∧ 
    (tri_eq A B C) ∧
    (radius O B = R) ∧ 
    (radius O C = R) → 
  area (triangle A B C) = (3 * Real.sqrt 3 / 4) * R^2 :=
sorry

end equilateral_tangent_triangle_area_l360_360475


namespace range_of_a_l360_360104

open Real

theorem range_of_a {a : ℝ} :
  (∃ x : ℝ, (√(a * x ^ 2 + a * x + 2) = a * x + 2)) →
  a ∈ {-8} ∪ Icc 1 (∞) :=
sorry

end range_of_a_l360_360104


namespace find_a4_l360_360077

noncomputable def sequence (n : ℕ) : ℤ
| 0       := 0           -- Since the sequence starts from n=1, we define a_0=0 as a dummy value.
| 1       := 3
| 2       := 6
| (n + 3) := sequence (n + 2) - sequence (n + 1)

theorem find_a4 : sequence 4 = -3 := 
by {
    show sequence 4 = -3,
    sorry
}

end find_a4_l360_360077


namespace problem_proof_l360_360063

theorem problem_proof (c d : ℝ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 6 + d = 9 + c) : 
  5 - c = 6 := 
sorry

end problem_proof_l360_360063


namespace find_projection_l360_360873

noncomputable def a : ℝ × ℝ := (-3, 2)
noncomputable def b : ℝ × ℝ := (5, -1)
noncomputable def p : ℝ × ℝ := (21/73, 56/73)
noncomputable def d : ℝ × ℝ := (8, -3)

theorem find_projection :
  ∃ t : ℝ, (t * d.1 - a.1, t * d.2 + a.2) = p ∧
          (p.1 - a.1) * d.1 + (p.2 - a.2) * d.2 = 0 :=
by
  sorry

end find_projection_l360_360873


namespace dot_product_magnitude_l360_360164

open Real

variables (a b : ℝ^3) 

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross (u v : ℝ^3) : ℝ^3 := (u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1)
def dot (u v : ℝ^3) : ℝ := u.1*v.1 + u.2*v.2 + u.3*v.3

axiom norm_a : norm a = 3
axiom norm_b : norm b = 4
axiom cross_norm : norm (cross a b) = 6

theorem dot_product_magnitude : abs (dot a b) = 6 * sqrt 3 := by
  sorry

end dot_product_magnitude_l360_360164


namespace expect_X_after_first_round_probability_game_ends_in_four_rounds_geometric_progression_Pi_l360_360603

-- Definitions of the conditions
def initial_chips_A : ℕ := 3
def initial_chips_B : ℕ := 3
def prob_A_wins_round : ℚ := 0.3
def prob_B_wins_round : ℚ := 0.2
def P (i : ℕ) : ℚ

section part1

-- Part 1: Probability distribution and expectation after first round
def X : Type := ℕ -- the number of chips A has after the first round

def P_X (x : ℕ) : ℚ :=
  match x with
  | 2 => 0.2
  | 3 => 0.5
  | 4 => 0.3
  | _ => 0

def E_X : ℚ :=
  2 * 0.2 + 3 * 0.5 + 4 * 0.3 

theorem expect_X_after_first_round (X : ℕ) : E_X = 3.1 :=
  sorry

end part1

section part2
-- Part 2: Probability of the game ending after four rounds
def prob_game_ends_in_4 : ℚ := 0.0525

theorem probability_game_ends_in_four_rounds : sorry :=
  sorry

end part2

section part3
-- Part 3: Geometric progression of the difference in probabilities
def P_ (i : ℕ) : ℚ
| 0 := 0
| 6 := 1
| _ := sorry

theorem geometric_progression_Pi (i : ℕ) (h : i ∈ {0, 1, 2, 3, 4, 5}): 
  (P_ (i + 1) - P_ i) / (P_ i - P_ (i - 1)) = (2 / 3) :=
  sorry

end part3

end expect_X_after_first_round_probability_game_ends_in_four_rounds_geometric_progression_Pi_l360_360603


namespace lowest_test_score_dropped_l360_360312

theorem lowest_test_score_dropped (A B C D : ℕ) 
  (h_avg_four : A + B + C + D = 140) 
  (h_avg_three : A + B + C = 120) : 
  D = 20 := 
by
  sorry

end lowest_test_score_dropped_l360_360312


namespace problem1_problem2_l360_360928

def f (x : ℝ) : ℝ := (3 * Real.exp x) / (1 + Real.exp x)

theorem problem1 (x : ℝ) : f x + f (-x) = 3 :=
by sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, x > 0 → f (4 - a * x) + f (x^2) ≥ 3) → a ≤ 4 :=
by sorry

end problem1_problem2_l360_360928


namespace cos_five_six_pi_sub_x_l360_360495

theorem cos_five_six_pi_sub_x (x : ℝ) (h : Real.cos(π / 6 + x) = 1 / 3) : Real.cos(5 * π / 6 - x) = -1 / 3 :=
by sorry

end cos_five_six_pi_sub_x_l360_360495


namespace raisin_addition_l360_360569

theorem raisin_addition : 
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  yellow_raisins + black_raisins = 0.7 := 
by
  sorry

end raisin_addition_l360_360569


namespace solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l360_360011

-- Definitions for the inequality ax^2 - 2ax + 2a - 3 < 0
def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Requirement (1): The solution set is ℝ
theorem solution_set_all_real (a : ℝ) (h : a ≤ 0) : 
  ∀ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (2): The solution set is ∅
theorem solution_set_empty (a : ℝ) (h : a ≥ 3) : 
  ¬∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (3): There is at least one real solution
theorem exists_at_least_one_solution (a : ℝ) (h : a < 3) : 
  ∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

end solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l360_360011


namespace no_all_real_roots_l360_360138

noncomputable def polynomial (a b c d : ℝ) : Polynomial ℝ :=
  Polynomial.C d + Polynomial.X * (Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X * (Polynomial.X * (Polynomial.X * Polynomial.X + Polynomial.C a))))

theorem no_all_real_roots (a b c d : ℝ) (h : ¬(a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0)) :
  ¬ ∀ (α : ℝ), is_root (polynomial a b c d) α :=
sorry

end no_all_real_roots_l360_360138


namespace fraction_of_grid_covered_by_triangle_l360_360356

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))|

noncomputable def area_of_grid : ℝ := 7 * 6

noncomputable def fraction_covered : ℝ :=
  area_of_triangle (-1, 2) (3, 5) (2, 2) / area_of_grid

theorem fraction_of_grid_covered_by_triangle : fraction_covered = (3 / 28) :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l360_360356


namespace dot_product_magnitude_l360_360174

variables {a b : EuclideanSpace 3 ℝ}

/- Given conditions -/
def norm_a : ℝ := ‖a‖ = 3
def norm_b : ℝ := ‖b‖ = 4
def norm_cross : ℝ := ‖a × b‖ = 6

/- Desired property to prove -/
theorem dot_product_magnitude :
  norm_a →
  norm_b →
  norm_cross →
  |(a ⋅ b)| = 6 * real.sqrt 3 :=
by
  sorry

end dot_product_magnitude_l360_360174


namespace infinite_nu_p_mod_d_l360_360462

open Nat

def is_prime (p : ℕ) : Prop := Nat.Prime p

def nu_p (p n : ℕ) : ℕ :=
  (nat.factorial n).multiplicity p

def satisfies_condition (p : ℕ) (d : ℕ) (n : ℕ) : Prop :=
  (nu_p p n) % d = 0

theorem infinite_nu_p_mod_d (d : ℕ) 
  (primes : Finset ℕ) (hprimes : ∀ p ∈ primes, is_prime p) :
  ∃ᶠ n in at_top, ∀ p ∈ primes, satisfies_condition p d n :=
sorry

end infinite_nu_p_mod_d_l360_360462


namespace check_3x5_board_cannot_be_covered_l360_360325

/-- Define the concept of a checkerboard with a given number of rows and columns. -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Define the number of squares on a checkerboard. -/
def num_squares (cb : Checkerboard) : ℕ :=
  cb.rows * cb.cols

/-- Define whether a board can be completely covered by dominoes. -/
def can_be_covered_by_dominoes (cb : Checkerboard) : Prop :=
  (num_squares cb) % 2 = 0

/-- Instantiate the specific checkerboard scenarios. -/
def board_3x4 := Checkerboard.mk 3 4
def board_3x5 := Checkerboard.mk 3 5
def board_4x4 := Checkerboard.mk 4 4
def board_4x5 := Checkerboard.mk 4 5
def board_6x3 := Checkerboard.mk 6 3

/-- Statement to prove which board cannot be covered completely by dominoes. -/
theorem check_3x5_board_cannot_be_covered : ¬ can_be_covered_by_dominoes board_3x5 :=
by
  /- We leave out the proof steps here as requested. -/
  sorry

end check_3x5_board_cannot_be_covered_l360_360325


namespace cos_angle_AFB_l360_360932

theorem cos_angle_AFB
  (p : ℝ) (h₀ : 0 < p)
  (A B F : ℝ × ℝ)
  (h₁ : ∃ m : ℝ, A = (m, real.sqrt (2 * p * m)) ∧ B = (m, -real.sqrt (2 * p * m)))
  (h₂ : F = (p / 2, 0))
  (h₃ : centroid F (0, 0) A B)
  : cos (angle A F B) = -23 / 25 := by
  sorry

end cos_angle_AFB_l360_360932


namespace artworks_collected_l360_360671

theorem artworks_collected (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) :
  students = 15 →
  artworks_per_student_per_quarter = 2 →
  quarters_per_year = 4 →
  num_years = 2 →
  (students * artworks_per_student_per_quarter * quarters_per_year * num_years) = 240 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end artworks_collected_l360_360671


namespace quartic_polynomial_has_given_roots_l360_360836

theorem quartic_polynomial_has_given_roots :
  ∃ (p : Polynomial ℚ), 
    p.monic ∧ 
    roots p = {3 + Real.sqrt 5, 3 - Real.sqrt 5, 2 + Real.sqrt 2, 2 - Real.sqrt 2} :=
by
  sorry

end quartic_polynomial_has_given_roots_l360_360836


namespace total_amount_paid_correct_l360_360786

-- Definitions of wholesale costs, retail markups, and employee discounts
def wholesale_cost_video_recorder : ℝ := 200
def retail_markup_video_recorder : ℝ := 0.20
def employee_discount_video_recorder : ℝ := 0.30

def wholesale_cost_digital_camera : ℝ := 150
def retail_markup_digital_camera : ℝ := 0.25
def employee_discount_digital_camera : ℝ := 0.20

def wholesale_cost_smart_tv : ℝ := 800
def retail_markup_smart_tv : ℝ := 0.15
def employee_discount_smart_tv : ℝ := 0.25

-- Calculation of retail prices
def retail_price (wholesale_cost : ℝ) (markup : ℝ) : ℝ :=
  wholesale_cost * (1 + markup)

-- Calculation of employee prices
def employee_price (retail_price : ℝ) (discount : ℝ) : ℝ :=
  retail_price * (1 - discount)

-- Retail prices
def retail_price_video_recorder := retail_price wholesale_cost_video_recorder retail_markup_video_recorder
def retail_price_digital_camera := retail_price wholesale_cost_digital_camera retail_markup_digital_camera
def retail_price_smart_tv := retail_price wholesale_cost_smart_tv retail_markup_smart_tv

-- Employee prices
def employee_price_video_recorder := employee_price retail_price_video_recorder employee_discount_video_recorder
def employee_price_digital_camera := employee_price retail_price_digital_camera employee_discount_digital_camera
def employee_price_smart_tv := employee_price retail_price_smart_tv employee_discount_smart_tv

-- Total amount paid by the employee
def total_amount_paid := 
  employee_price_video_recorder 
  + employee_price_digital_camera 
  + employee_price_smart_tv

theorem total_amount_paid_correct :
  total_amount_paid = 1008 := 
  by 
    sorry

end total_amount_paid_correct_l360_360786


namespace repeating_decimals_sum_l360_360423

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l360_360423


namespace find_f_2019_l360_360257

noncomputable def f (x : ℝ) : ℝ :=
if -3 < x ∧ x <= 0 then 3^(x - 1)
else f (x - 3)

theorem find_f_2019 : f 2019 = 1 / 3 := by
  sorry

end find_f_2019_l360_360257


namespace range_of_a_l360_360543

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.exp x + (1 - Real.sqrt Real.exp 1) * x - a

variables {f : ℝ → ℝ}

-- Condition: f is continuous on ℝ
axiom f_continuous : Continuous f

-- Condition: f(-x) + f(x) = x^2
axiom f_symmetry : ∀ x : ℝ, f(-x) + f(x) = x^2

-- Condition: When x ≤ 0, f'(x) < x
axiom f_derivative_condition : ∀ x : ℝ, x ≤ 0 → (deriv f) x < x

-- Condition: ∃ x₀ such that f(x₀) + 1/2 ≤ f(1 - x₀) + x₀ and x₀ is a fixed point of g
axiom fixed_point_exists : ∃ x₀ : ℝ, (f(x₀) + 1 / 2 ≤ f(1 - x₀) + x₀) ∧ (g x₀ a = x₀)

theorem range_of_a : ∀ a : ℝ, 
  (∃ x₀ : ℝ, (f(x₀) + 1 / 2 ≤ f(1 - x₀) + x₀) ∧ (g x₀ a = x₀)) ↔ a ≥ Real.sqrt (Real.exp 1) / 2 :=
by
  sorry

end range_of_a_l360_360543


namespace points_on_ellipse_l360_360468

noncomputable def x (t : ℝ) : ℝ := (3 - t^2) / (1 + t^2)
noncomputable def y (t : ℝ) : ℝ := 4 * t / (1 + t^2)

theorem points_on_ellipse : ∀ t : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (x t / a)^2 + (y t / b)^2 = 1 := 
sorry

end points_on_ellipse_l360_360468


namespace problem_statement_l360_360953
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l360_360953


namespace log_problem_l360_360089

noncomputable def solve_logarithmic_problem (x : ℝ) : ℝ :=
  if log 5 (log 4 (log 3 x)) = 1 then x^(1/3) else (0 : ℝ)

theorem log_problem (x : ℝ) (h : log 5 (log 4 (log 3 x)) = 1) : solve_logarithmic_problem x = 3^341 := by
  sorry

end log_problem_l360_360089


namespace pool_filling_time_l360_360785

noncomputable def pipeFillingTime (A B C D E : ℤ) (fillA fillB fillC fillD emptyE : ℚ) : ℚ :=
  let rateA := 1 / fillA
  let rateB := 1 / fillB
  let rateC := 1 / fillC
  let rateD := 1 / fillD
  let rateE := -1 / emptyE
  let netRate := rateA + rateB + rateC + rateD + rateE
  1 / netRate

theorem pool_filling_time : pipeFillingTime 8 12 16 20 40 (8 : ℚ) (12 : ℚ) (16 : ℚ) (20 : ℚ) (40 : ℚ) = 240 / 71 :=
  by
  unfold pipeFillingTime
  norm_num
  sorry

end pool_filling_time_l360_360785


namespace ones_digit_of_largest_power_of_2_in_20_fact_l360_360868

open Nat

def largest_power_of_2_in_factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else
    let sum_of_powers := ∑ m in range (n+1), m / 2
    sum_of_powers

def ones_digit_of_power_of_2 (exp : ℕ) : ℕ :=
  let cycle := [2, 4, 8, 6]
  cycle[exp % 4]

theorem ones_digit_of_largest_power_of_2_in_20_fact (n : ℕ) (h : n = 20) : 
  ones_digit_of_power_of_2 (largest_power_of_2_in_factorial n) = 4 :=
by
  rw [h]
  have : largest_power_of_2_in_factorial 20 = 18 := by
    -- Insert the calculations for largest_power_of_2_in_factorial here
    sorry
  rw [this]
  have : ones_digit_of_power_of_2 18 = 4 := by
    -- Insert the cycle calculations here
    sorry
  exact this

end ones_digit_of_largest_power_of_2_in_20_fact_l360_360868


namespace candy_distribution_l360_360597

theorem candy_distribution :
  let bags := 4
  let candies := 9
  (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 :=
by
  -- define variables for bags and candies
  let bags := 4
  let candies := 9
  have h : (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 := sorry
  exact h

end candy_distribution_l360_360597


namespace sum_of_x_such_that_gx_eq_neg3_l360_360202

def g (x : ℝ) : ℝ :=
if x < -3 then 3 * x + 6 else -x^2 - 2 * x + 2

theorem sum_of_x_such_that_gx_eq_neg3 : 
  (Finset.filter (λ x : ℝ, g x = -3) {x | -10 ≤ x ∧ x ≤ 10}.finite_to_set).sum id = 1 :=
by
  -- Conditions given
  have h₁ : { x : ℝ | g x = -3 } ⊆ {1} := sorry,
  -- Summarize the single value of the element set fulfilling the condition
  have h₂ : Finset.filter (λ x : ℝ, g x = -3) {x | -10 ≤ x ∧ x ≤ 10}.finite_to_set = {1} := by 
    finish [h₁],
  -- Sum of the set with valid solutions
  show (Finset.filter (λ x : ℝ, g x = -3) {x | -10 ≤ x ∧ x ≤ 10}.finite_to_set).sum id = 1 by rw [h₂]; simp

end sum_of_x_such_that_gx_eq_neg3_l360_360202


namespace question1_question2_question3_l360_360039

noncomputable def z1 : ℂ := 2 - 3 * Complex.I
noncomputable def z2 : ℂ := (15 - 5 * Complex.I) / ((2 + Complex.I) ^ 2)

theorem question1 : z1 + Complex.conj(z2) = 3 := by
  sorry

theorem question2 : z1 * z2 = -7 - 9 * Complex.I := by
  sorry

theorem question3 : z1 / z2 = (11 / 10) + (3 / 10) * Complex.I := by
  sorry

end question1_question2_question3_l360_360039


namespace problem_statement_l360_360562

noncomputable theory

variables {X Y Z P Q R : Type*}
variables {α β γ δ ε : ℝ}

def triangle_angles_add_to_180 (angle1 angle2 angle3 : ℝ) : Prop :=
  angle1 + angle2 + angle3 = 180

def external_angle (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 180 - angle1

def splitting_external_angle (angle_XZ : ℝ) (delta epsilon : ℝ) : Prop :=
  delta + epsilon = angle_XZ

axiom given_triangle_conditions 
  (α β γ δ ε: ℝ)
  (conds: triangle_angles_add_to_180 (α + β + γ) (δ + ε) 180): 
  α + β + γ = δ + ε

theorem problem_statement : α + β + γ = δ + ε :=
sorry

end problem_statement_l360_360562


namespace bamboo_middle_sections_volume_l360_360245

theorem bamboo_middle_sections_volume :
  ∃ (a1 d : ℝ),
  3 * a1 + 3 * d = 3.9 ∧
  4 * a1 + 26 * d = 3 ∧
  ((a1 + 3 * d) + (a1 + 4 * d) = 2.1) :=
begin
  sorry
end

end bamboo_middle_sections_volume_l360_360245


namespace necessary_but_not_sufficient_condition_l360_360028

variable (x : ℝ)

def p := |x| > 1
def q := x^2 + 5x + 6 < 0

theorem necessary_but_not_sufficient_condition : (∀ x, q x → p x) ∧ (¬ (∀ x, p x → q x)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l360_360028


namespace clock_angle_at_seven_l360_360697

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l360_360697


namespace angle_between_MD_and_NB_is_60_l360_360626

-- Definitions of points and geometric constructs
variables {Point : Type*} [emetric_space Point]
variables {A B C D M N : Point}
variables {angle : Point → Point → Point → ℝ}
variables {is_rhombus : Point → Point → Point → Point → Prop}
variables (intersects_AB_at : Point → Point → Point → Point → Prop)
variables (intersects_AD_at : Point → Point → Point → Point → Prop)

-- Given conditions
def conditions (A B C D M N : Point) : Prop :=
  is_rhombus A B C D ∧
  angle A B D = 60 ∧
  intersects_AB_at C M A B ∧
  intersects_AD_at C N A D

theorem angle_between_MD_and_NB_is_60 (h : conditions A B C D M N) :
  angle (line_through_points M D) (line_through_points N B) = 60 :=
sorry

end angle_between_MD_and_NB_is_60_l360_360626


namespace altitudes_bisect_internal_angle_bisector_altitudes_bisect_external_angle_bisector_l360_360232

/-- If ∠BAC = 60°, prove the line segment joining the feet of the altitudes from B and C bisects AA' --/
theorem altitudes_bisect_internal_angle_bisector (A B C : Type)
  [triangle A B C] (h₁ : ∠BAC = 60°) : 
  bisects (join (foot_altitude A B C B) (foot_altitude A B C C)) (angle_bisector A B C) := 
sorry

/-- If ∠BAC = 120° and AB ≠ AC, prove the line segment joining the feet of the altitudes from B and C bisects AA'' --/
theorem altitudes_bisect_external_angle_bisector (A B C : Type)
  [triangle A B C] (h₁ : ∠BAC = 120°) (h₂ : AB ≠ AC) : 
  bisects (join (foot_altitude A B C B) (foot_altitude A B C C)) (external_angle_bisector A B C) := 
sorry

end altitudes_bisect_internal_angle_bisector_altitudes_bisect_external_angle_bisector_l360_360232


namespace unknown_number_is_15_l360_360970

theorem unknown_number_is_15 (x : ℝ) (h : 45 - (28 - (37 - (x - 20))) = 59) : x = 15 :=
by {
    sorry,
}

end unknown_number_is_15_l360_360970


namespace apple_count_difference_l360_360677

theorem apple_count_difference
    (original_green : ℕ)
    (additional_green : ℕ)
    (red_more_than_green : ℕ)
    (green_now : ℕ := original_green + additional_green)
    (red_now : ℕ := original_green + red_more_than_green)
    (difference : ℕ := green_now - red_now)
    (h_original_green : original_green = 32)
    (h_additional_green : additional_green = 340)
    (h_red_more_than_green : red_more_than_green = 200) :
    difference = 140 :=
by
  sorry

end apple_count_difference_l360_360677


namespace methane_combined_l360_360009

def balancedEquation (CH₄ O₂ CO₂ H₂O : ℕ) : Prop :=
  CH₄ = 1 ∧ O₂ = 2 ∧ CO₂ = 1 ∧ H₂O = 2

theorem methane_combined {moles_CH₄ moles_O₂ moles_H₂O : ℕ}
  (h₁ : moles_O₂ = 2)
  (h₂ : moles_H₂O = 2)
  (h_eq : balancedEquation moles_CH₄ moles_O₂ 1 moles_H₂O) : 
  moles_CH₄ = 1 :=
by
  sorry

end methane_combined_l360_360009


namespace root_in_interval_iff_a_outside_range_l360_360067

theorem root_in_interval_iff_a_outside_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 ∧ a * x + 1 = 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end root_in_interval_iff_a_outside_range_l360_360067


namespace max_value_of_reciprocal_sums_of_zeros_l360_360065

noncomputable def quadratic_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + 2 * x - 1

noncomputable def linear_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x + 1

theorem max_value_of_reciprocal_sums_of_zeros (k : ℝ) (x1 x2 : ℝ)
  (h0 : -1 < k ∧ k < 0)
  (hx1 : x1 ∈ Set.Ioc 0 1 → quadratic_part k x1 = 0)
  (hx2 : x2 ∈ Set.Ioi 1 → linear_part k x2 = 0)
  (hx_distinct : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = 9 / 4 :=
sorry

end max_value_of_reciprocal_sums_of_zeros_l360_360065


namespace proof_problem_l360_360905

-- Definitions
variables {Point Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions provided
axiom lines_parallel : m ∥ n
axiom line_perpendicular_plane : m ⟂ α

-- Goal: Given the above conditions, prove that n ⟂ α
theorem proof_problem : n ⟂ α :=
sorry

end proof_problem_l360_360905


namespace alphabet_letters_count_l360_360549

variables (D S T : ℕ)

def number_with_both := 13
def number_with_straight_but_no_dot := 24
def number_with_dot_but_no_straight := 3

theorem alphabet_letters_count :
  (T = number_with_dot_but_no_straight + number_with_straight_but_no_dot + number_with_both) :=
begin
  let number_with_dot := number_with_dot_but_no_straight + number_with_both,
  let number_with_straight := number_with_straight_but_no_dot + number_with_both,
  have h1: D = number_with_dot,
  have h2: S = number_with_straight,
  
  rw [h1, h2],
  rw [number_with_both, number_with_straight_but_no_dot, number_with_dot_but_no_straight],
  exact D - 13 + S - 24 + 13,
  sorry
end

end alphabet_letters_count_l360_360549


namespace reach_pair_l360_360135

theorem reach_pair
  (n : ℕ) (h_odd : n % 2 = 1) (h_gt : n > 1) :
  ∃ b : ℕ, b % 2 = 0 ∧ b < n ∧ (∃ k : ℕ, (n, b).iterate_operation k = (b, n)) := sorry

-- Definitions needed for the proof (definition of iterate_operation)

def iterate_operation (p : ℕ × ℕ) (k : ℕ) : ℕ × ℕ :=
  if k = 0 then 
    p
  else 
    let (x, y) := iterate_operation p (k - 1) in
      if x % 2 = 0 then 
        (x / 2, y + x / 2)
      else 
        (x + y / 2, y / 2)

end reach_pair_l360_360135


namespace operation_performed_is_squaring_l360_360973

theorem operation_performed_is_squaring (x y : ℝ) (h1 : y = 68.70953354520753) 
(h2 : y^2 - x^2 = 4321) : ∃ x_squared : ℝ, x_squared = x^2 :=
by {
  use x^2,
  sorry,
}

end operation_performed_is_squaring_l360_360973


namespace vector_dot_product_range_correct_l360_360531

open Real

noncomputable def vector_dot_product_range (c d : EuclideanSpace ℝ (Fin 2)) (h_c_norm : ∥c∥ = 5) (h_d_norm : ∥d∥ = 13) : Set ℝ :=
{x | -65 ≤ x ∧ x ≤ 65}

theorem vector_dot_product_range_correct (c d : EuclideanSpace ℝ (Fin 2)) (h_c_norm : ∥c∥ = 5) (h_d_norm : ∥d∥ = 13) :
  ∃ (x : ℝ), x ∈ vector_dot_product_range c d h_c_norm h_d_norm → x = dot_product c d :=
sorry

end vector_dot_product_range_correct_l360_360531


namespace problem_statement_l360_360902

variable {a : ℝ} (ha : 0 ≤ a ∧ a ≤ 1)
def f1 (x : ℝ) : ℝ := x - a
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := -x^3 + x^2

theorem problem_statement :
  (∀ x0 : ℝ, ∃ i j, {i, j} ⊆ {1, 2, 3} ∧ (if i = 1 then f1 else if i = 2 then f2 else f3) x0 * (if j = 1 then f1 else if j = 2 then f2 else f3) x0 ≥ 0) ∧
  (∀ i j, {i, j} ⊆ {1, 2, 3} → ∃ x0 : ℝ, (if i = 1 then f1 else if i = 2 then f2 else f3) x0 * (if j = 1 then f1 else if j = 2 then f2 else f3) x0 < 0) :=
by sorry

end problem_statement_l360_360902


namespace tangent_line_at_x_2_increasing_on_1_to_infinity_l360_360068

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

-- Subpart I
theorem tangent_line_at_x_2 (a b : ℝ) :
  (a / 2 + 2 = 1) ∧ (2 + a * Real.log 2 = 2 + b) → (a = -2 ∧ b = -2 * Real.log 2) :=
by
  sorry

-- Subpart II
theorem increasing_on_1_to_infinity (a : ℝ) :
  (∀ x > 1, (x + a / x) ≥ 0) → (a ≥ -1) :=
by
  sorry

end tangent_line_at_x_2_increasing_on_1_to_infinity_l360_360068


namespace george_abe_together_l360_360884

/-- George can do a certain job in 70 minutes while Abe can do the same job in 30 minutes.
   Prove that the time taken for both to do the job together is 21 minutes. -/
theorem george_abe_together (george_time abe_time : ℝ) (h_george : george_time = 70) (h_abe : abe_time = 30) :
  1 / ((1 / george_time) + (1 / abe_time)) = 21 :=
by
  rw [h_george, h_abe]
  dsimp
  rw [(1 / 70) + (1 / 30)]
  sorry

end george_abe_together_l360_360884


namespace students_neither_l360_360599

def total_students : ℕ := 150
def students_math : ℕ := 85
def students_physics : ℕ := 63
def students_chemistry : ℕ := 40
def students_math_physics : ℕ := 20
def students_physics_chemistry : ℕ := 15
def students_math_chemistry : ℕ := 10
def students_all_three : ℕ := 5

theorem students_neither:
  total_students - 
  (students_math + students_physics + students_chemistry 
  - students_math_physics - students_physics_chemistry 
  - students_math_chemistry + students_all_three) = 2 := 
by sorry

end students_neither_l360_360599


namespace minimum_pencils_in_one_box_l360_360115

theorem minimum_pencils_in_one_box 
  (total_pencils : ℕ) (boxes : ℕ) (max_capacity : ℕ)
  (h_pencils : total_pencils = 74) (h_boxes : boxes = 13) (h_capacity : max_capacity = 6) : 
  ∃ min_pencils : ℕ, min_pencils = 2 :=
by 
  use 2
  sorry

end minimum_pencils_in_one_box_l360_360115


namespace tangent_line_eq_function_above_x_axis_l360_360925

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m * x^2 - x) * Real.log x + (1 / 2) * m * x^2

theorem tangent_line_eq (h : f 0 x = -x * Real.log x) : 
  let df := derivative (λ x => f 0 x),
      t_eq := (f 0 1, df 1) in
  t_eq = (0, -1) ∧ ∀ y, y = -x + 1 := 
by
  sorry

theorem function_above_x_axis (h1 : ∀ m, 0 < m → m ≤ 1 → 
  ∀ x, x > 0 → f m x > 0) : 
  1 / (2 * Real.sqrt (Real.exp 1)) < m ∧ m ≤ 1 → 
  ∀ x, x > 0 → f m x > 0 := 
by
  sorry

end tangent_line_eq_function_above_x_axis_l360_360925


namespace valid_m_values_l360_360451

theorem valid_m_values (m : ℕ) (hm : (number_of_divisors m)^4 = m) : 
  m = 625 ∨ m = 6561 ∨ m = 4100625 :=
begin
  sorry
end

end valid_m_values_l360_360451


namespace sum_equals_one_l360_360807

noncomputable def sum_proof (x y z : ℝ) (h : x * y * z = 1) : ℝ :=
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x))

theorem sum_equals_one (x y z : ℝ) (h : x * y * z = 1) : 
  sum_proof x y z h = 1 := sorry

end sum_equals_one_l360_360807


namespace prob_all_co_captains_l360_360277

theorem prob_all_co_captains :
  let p1 := 1 / (Nat.choose 6 3) * 1 / 4,
      p2 := 1 / (Nat.choose 8 3) * 1 / 4,
      p3 := 1 / (Nat.choose 9 3) * 1 / 4,
      p4 := 1 / (Nat.choose 10 3) * 1 / 4 in
  p1 + p2 + p3 + p4 = 37 / 1680 :=
by
  let p1 := 1 / (Nat.choose 6 3) * 1 / 4,
      p2 := 1 / (Nat.choose 8 3) * 1 / 4,
      p3 := 1 / (Nat.choose 9 3) * 1 / 4,
      p4 := 1 / (Nat.choose 10 3) * 1 / 4
  have h : p1 + p2 + p3 + p4 = (1 / 20) / 4 + (1 / 56) / 4 + (1 / 84) / 4 + (1 / 120) / 4 := by sorry
  have h' : (1 / 20) / 4 + (1 / 56) / 4 + (1 / 84) / 4 + (1 / 120) / 4 = 37 / 1680 := by sorry
  exact h.trans h'

end prob_all_co_captains_l360_360277


namespace count_valid_n_digit_numbers_l360_360473

open Nat

def count_valid_numbers (n : ℕ) : ℕ :=
  3^n - 3 * 2^n + 3

theorem count_valid_n_digit_numbers (n : ℕ) :
  ∃ f : ℕ → fin 3 → ℕ, (∀ x : ℕ, f x 0 + f x 1 + f x 2 = n) ∧
    (∀ x : ℕ, f x 0 > 0 ∧ f x 1 > 0 ∧ f x 2 > 0) ∧
    count_valid_numbers n = 3^n - 3 * 2^n + 3 := sorry

end count_valid_n_digit_numbers_l360_360473


namespace dot_product_magnitude_l360_360157

variable {a b : EuclideanSpace ℝ (Fin n)}
variable (norm_a : ∥a∥ = 3)
variable (norm_b : ∥b∥ = 4)
variable (norm_cross_ab : ∥a × b∥ = 6)

theorem dot_product_magnitude : ∥a∥ = 3 → ∥b∥ = 4 → ∥a × b∥ = 6 → |a ∘ b| = 6 * real.sqrt 3 := 
by 
  intro norm_a norm_b norm_cross_ab
  sorry

end dot_product_magnitude_l360_360157


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360843

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  let n := 32 in
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i) in 
  (2 ^ k) % 10 = 8 :=
by
  let n := 32
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i)
  
  have h1 : k = 31 := by sorry
  have h2 : (2 ^ 31) % 10 = 8 := by sorry
  
  exact h2

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360843


namespace avg_age_new_persons_l360_360641

-- Define initial conditions
def initial_avg_age : ℝ := 16
def initial_persons : ℕ := 20
def new_persons : ℕ := 20
def final_avg_age : ℝ := 15.5

-- Define the theorem to prove the average age (A) of the new persons who joined
theorem avg_age_new_persons : 
  (let total_initial_age := initial_avg_age * initial_persons in
   let total_final_age := final_avg_age * (initial_persons + new_persons) in
   let total_new_age := total_final_age - total_initial_age in
   let A := total_new_age / new_persons in
   A = 15) :=
sorry

end avg_age_new_persons_l360_360641


namespace shoe_pairing_probability_l360_360286

theorem shoe_pairing_probability :
  let m := 5
  let n := 36
  let total_probability := (m : ℚ) / n
  total_probability = 5 / 36 → m + n = 41 :=
begin
  intros h,
  exact (by injection h),
end

end shoe_pairing_probability_l360_360286


namespace speed_of_current_l360_360307

-- Define the constant speed of the boat relative to the water
def boat_speed : ℝ := 16

-- Define the time taken for upstream and downstream trips in hours
def upstream_time : ℝ := 20 / 60
def downstream_time : ℝ := 15 / 60

-- Define the statement to prove the speed of the current
theorem speed_of_current : ∃ c : ℝ, c = 16 / 7 :=
  by
  sorry

end speed_of_current_l360_360307


namespace limit_expression_l360_360761

theorem limit_expression :
  let seq := λ n : ℕ, (n * (71 * n).sqrt - (64 * n^6 + 9)^(1/3)) / ((n - n^(1/3)) * (11 + n^2).sqrt) in
  filter.tendsto seq filter.at_top (nhds (-4)) :=
by sorry

end limit_expression_l360_360761


namespace artworks_collected_l360_360673

theorem artworks_collected (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) :
  students = 15 →
  artworks_per_student_per_quarter = 2 →
  quarters_per_year = 4 →
  num_years = 2 →
  (students * artworks_per_student_per_quarter * quarters_per_year * num_years) = 240 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end artworks_collected_l360_360673


namespace quadratic_distinct_roots_l360_360977

theorem quadratic_distinct_roots (m : ℝ) : 
  ((m - 2) * x ^ 2 + 2 * x + 1 = 0) → (m < 3 ∧ m ≠ 2) :=
by
  sorry

end quadratic_distinct_roots_l360_360977


namespace min_value_of_function_l360_360110

noncomputable def minimization_problem 
  (AB AC : ℝ) 
  (BAC : ℝ) 
  (dot_product : ℝ) 
  (S_ABC : ℝ) 
  (x y : ℝ) 
  (z : ℝ) 
  (f_M : ℝ × ℝ × ℝ) : ℝ :=
  if h : (AB * AC * Math.cos BAC = dot_product ∧ BAC = Real.pi / 6 ∧ x + y = (1 / 2) ∧ f_M = (x, y, z) ∧ z = 1 / 2)
  then (\frac{1}{x} + \frac{4}{y}).recOn sorry
  else Inf (range (\frac{1}{x} + \frac{4}{y}))

theorem min_value_of_function 
  (dot_product : ℝ) 
  (BAC_angle : ℝ) 
  (x y z : ℝ)
  (f_M : ℝ × ℝ × ℝ): 
  (∀ (AB AC : ℝ), 
   AB * AC * Math.cos BAC_angle = dot_product 
   → BAC_angle = Real.pi / 6 
   → f_M = (x, y, z) 
   → z = 1 / 2 
   → x + y = 1 / 2 
   → (\frac{1}{x} + \frac{4}{y}) = 18) :=
sorry

end min_value_of_function_l360_360110


namespace total_path_traveled_l360_360235

theorem total_path_traveled (AB CD BC DA : ℝ) (h1 : AB = 3) (h2 : CD = 3) (h3 : BC = 10) (h4 : DA = 10) :
  let AD := Real.sqrt (AB^2 + DA^2) in
  let arc1 := (1/4) * 2 * Real.pi * AD in
  let arc2 := (1/4) * 2 * Real.pi * BC in
  let arc3 := (1/4) * 2 * Real.pi * CD in
  (arc1 + arc2 + arc3) = (6.5 + (Real.sqrt 109) / 2) * Real.pi :=
by
  sorry

end total_path_traveled_l360_360235


namespace dot_product_magnitude_l360_360153

variable {a b : EuclideanSpace ℝ (Fin n)}
variable (norm_a : ∥a∥ = 3)
variable (norm_b : ∥b∥ = 4)
variable (norm_cross_ab : ∥a × b∥ = 6)

theorem dot_product_magnitude : ∥a∥ = 3 → ∥b∥ = 4 → ∥a × b∥ = 6 → |a ∘ b| = 6 * real.sqrt 3 := 
by 
  intro norm_a norm_b norm_cross_ab
  sorry

end dot_product_magnitude_l360_360153


namespace average_production_last_5_days_l360_360551

theorem average_production_last_5_days
  (average_production_first_25_days : ℕ)
  (total_days : ℕ)
  (average_production_per_month : ℕ) :
  average_production_first_25_days = 50 →
  total_days = 30 →
  average_production_per_month = 45 →
  (let total_production_first_25_days := average_production_first_25_days * 25 in
   let total_production_entire_month := average_production_per_month * total_days in
   let total_production_last_5_days := total_production_entire_month - total_production_first_25_days in
   let average_production_last_5_days := total_production_last_5_days / 5 in
   average_production_last_5_days = 20) :=
begin
  intros h1 h2 h3,
  let total_production_first_25_days := 50 * 25,
  let total_production_entire_month := 45 * 30,
  let total_production_last_5_days := total_production_entire_month - total_production_first_25_days,
  have h_total : total_production_last_5_days = 100,
  { simp [total_production_entire_month, total_production_first_25_days] },
  have h_average : total_production_last_5_days / 5 = 20,
  { simp [h_total] },
  exact h_average,
end

end average_production_last_5_days_l360_360551


namespace parameter_range_l360_360929

open Real

noncomputable def f (a x : ℝ) : ℝ := log a (a * x^2 + (a + 2) * x + (a + 2))

def range_of_parameter (a : ℝ) : Prop :=
a ∈ (Set.Ioo (-2:ℝ) (-1:ℝ)) ∪ 
    (Set.Ioo (-1:ℝ) (0:ℝ)) ∪ 
    (Set.Ioi (2/3:ℝ))

theorem parameter_range (a : ℝ) :
  (∃ x : ℝ, has_maximum_or_minimum (f a x)) ↔ range_of_parameter a :=
sorry

end parameter_range_l360_360929


namespace prove_expression_l360_360061

-- Defining the variables and conditions
variables (a b : ℝ) (x y : ℝ)
variables (P Q : ℝ × ℝ)
def curve (x y : ℝ) := (y^2 / b) - (x^2 / a) = 1
def line (x y : ℝ) := x + y - 2 = 0
def non_zero (a b : ℝ) := a ≠ 0 ∧ b ≠ 0
def neq (a b : ℝ) := a ≠ b
def orthogonal (P Q : ℝ × ℝ) := P.1 * Q.1 + P.2 * Q.2 = 0

-- Mathematically equivalent proof problem
theorem prove_expression (h1 : non_zero a b) (h2 : neq a b)
  (h3 : line P.1 P.2) (h4 : line Q.1 Q.2)
  (h5 : curve P.1 P.2) (h6 : curve Q.1 Q.2) (h7 : orthogonal P Q) : 
  1 / b - 1 / a = 1 / 2 :=
sorry

end prove_expression_l360_360061


namespace sum_of_roots_of_quadratic_eq_l360_360731

theorem sum_of_roots_of_quadratic_eq : 
  ∀ (a b c : ℝ), (x^2 - 6 * x + 8 = 0) → (a = 1 ∧ b = -6 ∧ c = 8) → -b / a = 6 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_eq_l360_360731


namespace find_lambda_l360_360053

variables {R : Type*} [LinearOrderedField R] (a b : EuclideanSpace R (Fin 2))

-- Given conditions
def angle (a b : EuclideanSpace R (Fin 2)) : Real := 5 * Real.pi / 6
def length_a : R := 1
def length_b : R := 2
def orthogonal_condition (λ : R) : Prop := 
  (sqrt 3 • a + λ • b) ⬝ a = 0

-- Theorem stating that λ = 1 under the given conditions
theorem find_lambda (h_angle : ∀a b, angle a b = 5 * Real.pi / 6)
                    (h_len_a : ∀a, ‖a‖ = 1)
                    (h_len_b : ∀b, ‖b‖ = 2)
                    (h_orthogonal : orthogonal_condition a b λ) :
                    λ = 1 :=
sorry

end find_lambda_l360_360053


namespace ones_digit_large_power_dividing_32_factorial_l360_360854

theorem ones_digit_large_power_dividing_32_factorial :
  let n := 32!
  let largestPower := 2^31
  ones_digit largestPower = 8 :=
by
  sorry

end ones_digit_large_power_dividing_32_factorial_l360_360854


namespace min_a_plus_b_l360_360498

noncomputable def f (x : ℝ) : ℝ := log (x / 2)

noncomputable def f'' (x : ℝ) : ℝ := - (x ^ (-2))

theorem min_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : b * ∫ x in 1..b, x ^ (-3) = 2 * f'' a + (1 / 2) * b - 1) : a + b = 9 / 2 :=
by
  sorry

end min_a_plus_b_l360_360498


namespace smaller_angle_at_seven_oclock_l360_360698

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l360_360698


namespace percentage_reduction_price_increase_l360_360333

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end percentage_reduction_price_increase_l360_360333


namespace functional_equation_l360_360837

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(f(x) + y) = 2*x + f(-f(f(x)) + f(y))) → (∀ x : ℝ, f(x) = x) :=
sorry

end functional_equation_l360_360837


namespace find_S_polynomial_l360_360586

noncomputable def S (z : ℂ) : ℂ := -z + 1

theorem find_S_polynomial (P : ℂ → ℂ) (S : ℂ → ℂ) (h1 : ∀ z : ℂ, z^{11} + 1 = (z^3 - z + 2) * P(z) + S(z)) (h2 : ∀ z : ℂ, ∃ a b c : ℂ, S z = a * z^2 + b * z + c ∧ a = 0) : S = (λ z, -z + 1) :=
by
  sorry

end find_S_polynomial_l360_360586


namespace sum_of_roots_of_quadratic_eq_l360_360730

theorem sum_of_roots_of_quadratic_eq : 
  ∀ (a b c : ℝ), (x^2 - 6 * x + 8 = 0) → (a = 1 ∧ b = -6 ∧ c = 8) → -b / a = 6 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_eq_l360_360730


namespace value_of_P_l360_360971

noncomputable def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

theorem value_of_P :
  let P := Finset.sum (Finset.range 1000) (λ k, f ((k + 1 : ℝ) / 1001)) in
  P = 500 :=
by
  let f := λ x, 4^x / (4^x + 2)
  have hf_eq : ∀ x, f x + f (1 - x) = 1 :=
    sorry
  let P := Finset.sum (Finset.range 1000) (λ k, f ((k + 1 : ℝ) / 1001))
  have h_pairs : P = 500 :=
    sorry
  exact h_pairs

end value_of_P_l360_360971


namespace a_values_for_quadratic_l360_360470

theorem a_values_for_quadratic (a : ℝ) : (∃ x : ℝ, x^(a^2-7) - 3 * x - 2 = 0) → (a = 3 ∨ a = -3) :=
by
  intro h
  have : a^2 - 7 = 2
  sorry

end a_values_for_quadratic_l360_360470


namespace sum_of_repeating_decimals_l360_360410

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l360_360410


namespace repeating_decimal_sum_l360_360409

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l360_360409


namespace angle_equality_l360_360577

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def orthogonal_projection (M : Point) (l : Line) : Point := sorry
noncomputable def perpendicular_line (l : Line) (P : Point) : Line := sorry
noncomputable def intersection (l1 l2 : Line) : Point := sorry
noncomputable def angle (A B C : Point) : Real := sorry
noncomputable def triangle (A B C : Point) : Prop := sorry

variables (A B C : Point)
variables (hABC : triangle A B C)

def M := midpoint A B
def X := orthogonal_projection M (line_through A B)
def Y := orthogonal_projection M (line_through B C)
def d := perpendicular_line (line_through A B) A
def d' := perpendicular_line (line_through A B) B
def E := intersection (line_through M X) d
def F := intersection (line_through M Y) d'
def Z := intersection (line_through E F) (line_through C M)

theorem angle_equality :
  angle A Z B = angle F M E := sorry

end angle_equality_l360_360577


namespace problem_statement_l360_360950
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l360_360950


namespace tangent_line_parallel_range_a_l360_360651

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log x + 1/2 * x^2 + a * x

theorem tangent_line_parallel_range_a (a : ℝ) :
  (∃ x > 0, deriv (f a) x = 3) ↔ a ≤ 1 :=
by
  sorry

end tangent_line_parallel_range_a_l360_360651


namespace price_returns_to_original_l360_360122

def price_adjustments (initial_price : ℝ) (increases decreases : List ℝ) : List ℝ :=
  increases.zip decreases |>.scanl 
    (λ price (inc, dec), price * (1 + inc) * (1 - dec))
    initial_price

theorem price_returns_to_original 
  (initial_price : ℝ)
  (x : ℝ)
  (H1: initial_price > 0)
  (H2: price_adjustments initial_price [0.15, -0.15, 0.3, -x] [0, 0, 0, 0] = [initial_price])
  : x = 0.21 :=
by
  sorry

end price_returns_to_original_l360_360122


namespace find_function_l360_360840

noncomputable def F (x : ℝ) : ℝ := (4 / 3) * x^3 - 9 / x - 35

theorem find_function :
  (∀ x : ℝ, deriv (F) x = 4 * x^2 + 9 * x^(-2)) ∧ F 3 = -2 :=
by
  split
  sorry

end find_function_l360_360840


namespace recreation_percentage_l360_360225

noncomputable def week_1_wage : ℝ := 1 -- We can normalize W1 to 1 for simplicity
def week_1_recreation_percentage : ℝ := 0.55
def week_2_wage_decrease_percentage : ℝ := 0.10
def week_3_tax_percentage : ℝ := 0.05
def week_4_wage_decrease_percentage : ℝ := 0.15

theorem recreation_percentage :
  let W1 := week_1_wage in
  let W2 := W1 * (1 - week_2_wage_decrease_percentage) * (1 - week_3_tax_percentage) in
  let W3 := W1 * (1 - week_2_wage_decrease_percentage) * (1 - week_3_tax_percentage) in
  let W4 := W3 * (1 - week_4_wage_decrease_percentage) in
  let total_recreation :=
    (week_1_recreation_percentage * W1) +
    (0.65 * W2) +
    (0.60 * W3) +
    (0.75 * W4) in
  let total_earned := W1 + (W1 * (1 - week_2_wage_decrease_percentage)) + (W1 * (1 - week_2_wage_decrease_percentage)) + W4 in
  (total_recreation / total_earned) * 100 ≈ 61.52 :=
by
  let W1 := week_1_wage
  let W2 := W1 * (1 - week_2_wage_decrease_percentage) * (1 - week_3_tax_percentage)
  let W3 := W1 * (1 - week_2_wage_decrease_percentage) * (1 - week_3_tax_percentage)
  let W4 := W3 * (1 - week_4_wage_decrease_percentage)
  let total_recreation :=
    (week_1_recreation_percentage * W1) +
    (0.65 * W2) +
    (0.60 * W3) +
    (0.75 * W4)
  let total_earned := W1 + (W1 * (1 - week_2_wage_decrease_percentage)) + (W1 * (1 - week_2_wage_decrease_percentage)) + W4
  have h1 : (total_recreation / total_earned) * 100 ≈ math.log 61.52 := sorry
  triv

end recreation_percentage_l360_360225


namespace exists_triangle_free_not_4_colorable_l360_360476

/-- Define a graph as a structure with vertices and edges. -/
structure Graph (V : Type*) :=
  (adj : V → V → Prop)
  (symm : ∀ x y, adj x y → adj y x)
  (irreflexive : ∀ x, ¬adj x x)

/-- A definition of triangle-free graph. -/
def triangle_free {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c : V), G.adj a b → G.adj b c → G.adj c a → false

/-- A definition that a graph cannot be k-colored. -/
def not_k_colorable {V : Type*} (G : Graph V) (k : ℕ) : Prop :=
  ¬∃ (f : V → ℕ), (∀ (v : V), f v < k) ∧ (∀ (v w : V), G.adj v w → f v ≠ f w)

/-- There exists a triangle-free graph that is not 4-colorable. -/
theorem exists_triangle_free_not_4_colorable : ∃ (V : Type*) (G : Graph V), triangle_free G ∧ not_k_colorable G 4 := 
sorry

end exists_triangle_free_not_4_colorable_l360_360476


namespace probability_divisibility_by_2_l360_360796

theorem probability_divisibility_by_2:
    (set.count {n : ℤ | 1 ≤ n ∧ n ≤ 100 ∧ (n * (n + 1)) % 2 = 0} ∩ {n : ℤ | 1 ≤ n ∧ n ≤ 100}) / (set.count {n : ℤ | 1 ≤ n ∧ n ≤ 100}) = (1/2 : ℚ) :=
by sorry

end probability_divisibility_by_2_l360_360796


namespace fraction_of_project_completed_in_one_hour_l360_360536

noncomputable def fraction_of_project_completed_together (a b : ℝ) : ℝ :=
  (1 / a) + (1 / b)

theorem fraction_of_project_completed_in_one_hour (a b : ℝ) :
  fraction_of_project_completed_together a b = (1 / a) + (1 / b) := by
  sorry

end fraction_of_project_completed_in_one_hour_l360_360536


namespace dot_product_magnitude_l360_360189

variables {a b : ℝ^3}

-- Norm of vectors
def norm_a := ‖a‖ = 3
def norm_b := ‖b‖ = 4
def cross_product_norm := ‖a × b‖ = 6

theorem dot_product_magnitude (h1 : norm_a) (h2 : norm_b) (h3 : cross_product_norm) : (abs (a • b)) = 6 * real.sqrt 3 :=
sorry

end dot_product_magnitude_l360_360189


namespace l_shaped_area_l360_360448

-- Define the configuration and the final proof statement
theorem l_shaped_area (side_ABCD side_square1 side_square2 side_square3 : ℕ)
  (h_ABCD : side_ABCD = 5) (h_square1 : side_square1 = 1) (h_square2 : side_square2 = 1) (h_square3 : side_square3 = 3) :
  let area_L := 2 * (side_square3 * side_square1) + (side_square1 * side_square1)
  in area_L = 7 := by
  -- Setting side lengths for clarity
  let total_side := side_ABCD
  let l1 := side_square1
  let l2 := side_square2
  let l3 := side_square3
  -- Calculate the area of rectangles and squares
  let area_rectangles := 2 * (l3 * l1)
  let area_square := l1 ^ 2
  -- Sum of all areas
  let area_L := area_rectangles + area_square
  -- The final proof skips here
  sorry

end l_shaped_area_l360_360448


namespace size_of_angle_B_area_of_triangle_l360_360545

open Real

variables {a b c : ℝ} {A B C : ℝ}

-- Defining the conditions
def cos_A : ℝ := sqrt 10 / 10
def cos_C : ℝ := sqrt 5 / 5

-- Part 1: Proving the size of angle B
theorem size_of_angle_B (hA : cos_A = sqrt 10 / 10) (hC : cos_C = sqrt 5 / 5) :
    B = π / 4 :=
sorry

-- Part 2: Proving the area of the triangle when c = 4
theorem area_of_triangle (hA : cos_A = sqrt 10 / 10) (hC : cos_C = sqrt 5 / 5) (hc : c = 4) :
    1 / 2 * (√10) * c * (3 * sqrt 10 / 10) = 6 :=
sorry

end size_of_angle_B_area_of_triangle_l360_360545


namespace altitude_circumcenter_l360_360607

open EuclideanGeometry

variable {A B C D K M O : Point}
variable {R : ℝ}

theorem altitude_circumcenter (h : ∀ (A B C : Triangle), altitude B D = R * sqrt 2) :
  ∀ {triangle : Triangle}, altitude B D = R * sqrt 2 →
  (foot D A B = K ∧ foot D B C = M ∧ circumcenter A B C = O) →
  line_through K M O := by
  sorry

end altitude_circumcenter_l360_360607


namespace repeating_decimals_sum_l360_360421

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l360_360421


namespace sum_of_roots_of_quadratic_eq_l360_360724

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l360_360724


namespace at_least_one_woman_selected_l360_360101

/-- Lean notation for probability calculation setup -/
noncomputable def probability_at_least_one_woman_selected : ℚ :=
  let total_people := 15 in
  let men := 10 in
  let women := 5 in
  let total_selections := 4 in
  let prob_all_men : ℚ :=
    (men / total_people) *
    ((men - 1) / (total_people - 1)) *
    ((men - 2) / (total_people - 2)) *
    ((men - 3) / (total_people - 3))
  in
  1 - prob_all_men
  

theorem at_least_one_woman_selected :
  probability_at_least_one_woman_selected = 84 / 91 :=
sorry

end at_least_one_woman_selected_l360_360101


namespace quadrilaterals_equal_by_sides_median_l360_360234

-- Define the concept of a quadrilateral
structure Quadrilateral :=
(A B C D : ℝ × ℝ) -- Four vertices in ℝ²

-- Define the concept of a median line (as the median of the diagonals)
def median_line (q : Quadrilateral) : ℝ × ℝ :=
((q.A.1 + q.C.1) / 2, (q.A.2 + q.C.2) / 2) -- Example median from A to C

-- Define equality of quadrilaterals by their sides and median line
def equal_quadrilaterals (q1 q2 : Quadrilateral) : Prop :=
  -- Sides are equal
  (dist q1.A q1.B = dist q2.A q2.B) ∧
  (dist q1.B q1.C = dist q2.B q2.C) ∧
  (dist q1.C q1.D = dist q2.C q2.D) ∧
  (dist q1.D q1.A = dist q2.D q2.A) ∧
  -- Median lines are equal
  (median_line q1 = median_line q2)

-- Problem statement theorem
theorem quadrilaterals_equal_by_sides_median (q1 q2 : Quadrilateral) :
  -- The sides and median line of q1 are equal to those of q2
  equal_quadrilaterals q1 q2 →
  -- Then the quadrilaterals are equal
  q1 = q2 :=
by
  -- Proof would go here, but it's omitted according to instructions
  sorry

end quadrilaterals_equal_by_sides_median_l360_360234


namespace union_sets_l360_360886

open Set Real

def A : Set ℝ := { x | 1 / 8 < 2^(-x) ∧ 2^(-x) < 1 / 2 }
def B : Set ℝ := { x | log 2 (x - 2) < 1 }

theorem union_sets : A ∪ B = { x | 1 < x ∧ x < 4 } := by
  sorry

end union_sets_l360_360886


namespace repeating_decimal_sum_l360_360398

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l360_360398


namespace min_shots_for_probability_at_least_075_l360_360241

theorem min_shots_for_probability_at_least_075 (hit_rate : ℝ) (target_probability : ℝ) :
  hit_rate = 0.25 → target_probability = 0.75 → ∃ n : ℕ, n = 4 ∧ (1 - hit_rate)^n ≤ 1 - target_probability := by
  intros h_hit_rate h_target_probability
  sorry

end min_shots_for_probability_at_least_075_l360_360241


namespace problem1_problem2_problem3_l360_360486

-- Definitions of given conditions
noncomputable def f (x : ℝ) (a : ℝ) := 
  (√( (1 - x^2) / (1 + x^2) )) + (a * √( (1 + x^2) / (1 - x^2) ))

-- Problem 1: Prove minimum value of f(x) when a = 1
theorem problem1 : (∀ x : ℝ, f x 1 ≥ 2) ∧ (∃ x : ℝ, f x 1 = 2) := 
sorry

-- Problem 2: Prove monotonicity of f(x) when a = 1
theorem problem2 : (∀ x ∈ set.Ico 0 1, ∀ x₁ x₂, x₁ < x₂ → f x₁ 1 < f x₂ 1) ∧ (∀ x ∈ set.Ico (-1) 0, ∀ x₁ x₂, x₁ < x₂ → f x₁ 1 > f x₂ 1) :=
sorry

-- Problem 3: Range of a for triangle side lengths condition
theorem problem3 (r s t : ℝ) : 
  (-2 * √5 / 5 ≤ r ∧ r ≤ 2 * √5 / 5) ∧ 
  (-2 * √5 / 5 ≤ s ∧ s ≤ 2 * √5 / 5) ∧
  (-2 * √5 / 5 ≤ t ∧ t ≤ 2 * √5 / 5) → 
  (∃ a : ℝ, 1/15 < a ∧ a < 5/3 ∧ 
    f r a + f s a > f t a ∧ 
    f r a + f t a > f s a ∧ 
    f s a + f t a > f r a) :=
sorry

end problem1_problem2_problem3_l360_360486


namespace infinitely_many_sums_of_squares_l360_360610

noncomputable def pell_sol : ℕ → ℕ := sorry -- A placeholder for the nth solution to the Pell equation x^2 - 2*y^2 = 1

theorem infinitely_many_sums_of_squares :
  ∃ (f : ℕ → ℕ), (∀ m : ℕ, (f m)^2 - 2*((pell_sol m)^2) = 1) ∧ (∀ m ≥ 0, ∃ a b c d e f : ℕ, ((f m)^2 - 1) = a^2 + b^2 ∧ ((f m)^2) = c^2 + d^2 ∧ ((f m)^2 + 1) = e^2 + f^2) :=
by {
  sorry
}

end infinitely_many_sums_of_squares_l360_360610


namespace probability_distinct_roots_l360_360499

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, g x ≠ 0
axiom cond2 : ∀ x : ℝ, (f' x) * (g x) > (f x) * (g' x)
axiom cond3 : ∀ (a b : ℕ), (a, b) ∈ (fin 6).prod fin 6

axiom cond4 : ∀ x, f x = (7 : ℝ)^x * g x
axiom cond5 : (f 1 / g 1) + (f (-1) / g (-1)) ≥ 10 / 3

theorem probability_distinct_roots : 
  let outcomes := (fin 6).prod fin 6 in
  let valid_outcomes := { (a, b) | a > 0 ∧ a < 7 ∧ b > 0 ∧ b < 7 ∧ 64 - 4 * (a * b) > 0 } in
  (set.card valid_outcomes).to_real / (set.card outcomes).to_real = 13 / 36 :=
sorry

end probability_distinct_roots_l360_360499


namespace dictionary_cost_l360_360571

theorem dictionary_cost
  (dinosaur_book_cost : ℕ)
  (cookbook_cost : ℕ)
  (saved_amount : ℕ)
  (needed_amount : ℕ)
  (total_amount : ℕ := saved_amount + needed_amount)
  (total_books_cost : ℕ := dinosaur_book_cost + cookbook_cost)
  : dinosaur_book_cost = 19 ∧
    cookbook_cost = 7 ∧
    saved_amount = 8 ∧
    needed_amount = 29 →
    total_amount = 37 ∧
    total_books_cost = 26 →
    ∃ dictionary_cost : ℕ, dictionary_cost = 11 := by
  -- definition of saved amount and needed amount
  intro h,
  have hs : saved_amount = 8, from h.1.2,
  have hn : needed_amount = 29, from h.1.3,
  have ht : total_amount = 37, from h.2.1,
  have hd : dinosaur_book_cost = 19, from h.1.1,
  have hc : cookbook_cost = 7, from h.1.2.1,
  have htc : total_books_cost = 26, from h.2.2,
  have dictionary_cost := total_amount - total_books_cost,
  use dictionary_cost,
  rw ht at dictionary_cost,
  rw htc at dictionary_cost,
  norm_num at dictionary_cost,
  trivial

end dictionary_cost_l360_360571


namespace problem_solution_l360_360617

theorem problem_solution (x : ℝ) : 
  (x - 2) / (x - 1) > (4 * x - 1) / (3 * x + 8) ↔ 
  (x ∈ set.Ioo (-3 : ℝ) (-2) ∨ x ∈ set.Ioo (-8 / 3) 1) := 
sorry

end problem_solution_l360_360617


namespace dot_product_magnitude_l360_360155

variable {a b : EuclideanSpace ℝ (Fin n)}
variable (norm_a : ∥a∥ = 3)
variable (norm_b : ∥b∥ = 4)
variable (norm_cross_ab : ∥a × b∥ = 6)

theorem dot_product_magnitude : ∥a∥ = 3 → ∥b∥ = 4 → ∥a × b∥ = 6 → |a ∘ b| = 6 * real.sqrt 3 := 
by 
  intro norm_a norm_b norm_cross_ab
  sorry

end dot_product_magnitude_l360_360155


namespace area_DBCE_eq_47_l360_360991

variable (ABC DBCE ADE : Type)
variable {area : ∀ (t : Type), ℝ}
variable [SimilarTriangles ABC ADE]
variable [SimilarTriangles ABC DBCE]

axiom area_ABC : area ABC = 50
axiom area_ADE_sum_of_5_smallest : (∑ _ in (finset.range 3), (1 : ℝ)) = 3
axiom area_smallest_triangle : ∀ (t : Type) [IsSmallestTriangle t], area t = 1

theorem area_DBCE_eq_47 (h1 : SimilarTriangles ABC ADE) (h2: SimilarTriangles ABC DBCE) : 
  area DBCE = 47 := 
by 
  sorry

end area_DBCE_eq_47_l360_360991


namespace repeating_decimal_sum_l360_360406

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l360_360406


namespace seating_probability_l360_360878

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def catalan (n : ℕ) : ℕ := binomial (2 * n) n / (n + 1)

theorem seating_probability :
  let arrangements := binomial 11 5
  let valid_arrangements := catalan 6
  let probability := valid_arrangements / arrangements
  let m := 2
  let n := 7
  m + n = 9 :=
by
  let arrangements := binomial 11 5
  let valid_arrangements := catalan 6
  let probability := valid_arrangements / arrangements
  have probability_eq : probability = 2/7 := sorry
  let m := 2
  let n := 7
  show m + n = 9, from rfl

end seating_probability_l360_360878


namespace area_of_region_l360_360291

-- Define the given equation as a proposition
def original_equation (x y : ℝ) : Prop := 
  x^2 + y^2 - 10 = 4y - 6x - 2

-- Define the area of a circle with a given radius
def circle_area (r : ℝ) := π * r^2

-- Prove that given the original equation describing a circle,
-- the area of the region is equal to 25π
theorem area_of_region : 
  (∀ x y : ℝ, original_equation x y) → 
  circle_area 5 = 25 * π :=
by
  sorry

end area_of_region_l360_360291


namespace ones_digit_of_largest_power_of_2_in_20_fact_l360_360867

open Nat

def largest_power_of_2_in_factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else
    let sum_of_powers := ∑ m in range (n+1), m / 2
    sum_of_powers

def ones_digit_of_power_of_2 (exp : ℕ) : ℕ :=
  let cycle := [2, 4, 8, 6]
  cycle[exp % 4]

theorem ones_digit_of_largest_power_of_2_in_20_fact (n : ℕ) (h : n = 20) : 
  ones_digit_of_power_of_2 (largest_power_of_2_in_factorial n) = 4 :=
by
  rw [h]
  have : largest_power_of_2_in_factorial 20 = 18 := by
    -- Insert the calculations for largest_power_of_2_in_factorial here
    sorry
  rw [this]
  have : ones_digit_of_power_of_2 18 = 4 := by
    -- Insert the cycle calculations here
    sorry
  exact this

end ones_digit_of_largest_power_of_2_in_20_fact_l360_360867


namespace moores_law_2000_l360_360217

noncomputable def number_of_transistors (year : ℕ) : ℕ :=
  if year = 1990 then 1000000
  else 1000000 * 2 ^ ((year - 1990) / 2)

theorem moores_law_2000 :
  number_of_transistors 2000 = 32000000 :=
by
  unfold number_of_transistors
  rfl

end moores_law_2000_l360_360217


namespace largest_divisor_of_n4_minus_n_l360_360880

theorem largest_divisor_of_n4_minus_n (n : ℤ) (h : ∃ k : ℤ, n = 4 * k) : 4 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l360_360880


namespace repeating_decimal_sum_l360_360441

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l360_360441


namespace find_single_digit_number_l360_360619

theorem find_single_digit_number (n : ℕ) : 
  (5 < n ∧ n < 9 ∧ n > 7) ↔ n = 8 :=
by
  sorry

end find_single_digit_number_l360_360619


namespace price_reduction_percentage_price_increase_amount_l360_360329

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end price_reduction_percentage_price_increase_amount_l360_360329


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360842

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  let n := 32 in
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i) in 
  (2 ^ k) % 10 = 8 :=
by
  let n := 32
  let k := ∑ i in finset.range (nat.log 2 n + 1), n / (2 ^ i)
  
  have h1 : k = 31 := by sorry
  have h2 : (2 ^ 31) % 10 = 8 := by sorry
  
  exact h2

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l360_360842


namespace count_repeating_decimals_between_1_and_15_l360_360466

def is_repeating_decimal (n d : ℕ) : Prop :=
  ∀ p, p.prime ∧ (p ∣ d) → p = 2 ∨ p = 5 → False

theorem count_repeating_decimals_between_1_and_15 : 
  (finset.filter (λ n, is_repeating_decimal n 18) (finset.range 16)).card = 14 :=
  sorry

end count_repeating_decimals_between_1_and_15_l360_360466


namespace Taran_original_number_is_12_l360_360243

open Nat

theorem Taran_original_number_is_12 (x : ℕ)
  (h1 : (5 * x) + 5 - 5 = 73 ∨ (5 * x) + 5 - 6 = 73 ∨ (5 * x) + 6 - 5 = 73 ∨ (5 * x) + 6 - 6 = 73 ∨ 
       (6 * x) + 5 - 5 = 73 ∨ (6 * x) + 5 - 6 = 73 ∨ (6 * x) + 6 - 5 = 73 ∨ (6 * x) + 6 - 6 = 73) : x = 12 := by
  sorry

end Taran_original_number_is_12_l360_360243


namespace function_behavior_l360_360341

open Function

variable {f : ℝ → ℝ}

theorem function_behavior 
  (h1: ∀ x ∈ ℝ, f(-x) = f(x)) 
  (h2: ∀ x ∈ ℝ, f(x) = f(2 - x))
  (h3: ∀ x1 x2 ∈ Icc 1 2, x1 < x2 → f(x1) > f(x2)):
  (∀ x1 x2 ∈ Icc (-2) (-1), x1 < x2 → f(x1) < f(x2)) ∧ 
  (∀ x1 x2 ∈ Icc 3 4, x1 < x2 → f(x1) > f(x2)) := 
sorry

end function_behavior_l360_360341


namespace cube_opposite_face_k_l360_360216

-- Defining the basic structure and problem
structure Cube :=
  (size : ℕ)
  (cubes : ℕ)

def MishaCube : Cube := {size := 3, cubes := 16}

theorem cube_opposite_face_k (c : Cube) (word : String) (face : String) : c = MishaCube → word = "KOT" → face = "O" → opposite_face c face = "K" :=
by  
  intros hc hw hf
  sorry

end cube_opposite_face_k_l360_360216


namespace part1_part2_l360_360521

def m (x : ℝ) : (ℝ × ℝ) := (Real.sin x, -1/2)
def n (x : ℝ) : (ℝ × ℝ) := (Real.sqrt 3 * Real.cos x, Real.cos (2 * x))
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem part1 (x : ℝ) : 
  f x = Real.sin (2 * x - π / 6) ∧ (∀ x, f x ≤ 1) ∧ (∃ t > 0, ∀ x, f (x + t) = f x) :=
by
  sorry

def g (x : ℝ) : ℝ := Real.sin (2 * x + π / 6)

theorem part2 : ∀ x ∈ Set.Icc (0 : ℝ) (π / 2), g x ∈ Set.Icc ((-1)/2) 1 :=
by
  sorry

end part1_part2_l360_360521


namespace distance_P_to_line_equals_one_l360_360057

noncomputable def f (x m : ℝ) : ℝ := (m - 1) * x^2 + x

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def distanceFromPointToLine (x0 y0 A B C : ℝ) : ℝ :=
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

theorem distance_P_to_line_equals_one (m : ℝ)
  (h1 : isOddFunction (f _ m))
  (P : ℝ × ℝ)
  (hP : P = (m, 2))
  (line_eq : 3 * P.1 + 4 * P.2 - 6 = 0) : 
  distanceFromPointToLine P.1 P.2 3 4 (-6) = 1 := sorry

end distance_P_to_line_equals_one_l360_360057


namespace monotonic_interval_fx_gx_gt_one_sequence_bound_l360_360511

section Monotonicity

variable (f : ℝ → ℝ) (a : ℝ) (x : ℝ) 

def f_def : f x = Real.exp x - a * x - a := sorry

theorem monotonic_interval_fx (a : ℝ) :
  ((a ≤ 0) ∧ (∀ x y : ℝ, x < y → f x < f y))
  ∨ ((a > 0) ∧ (∀ x y : ℝ, x < y ∧ y < Real.log a → f x > f y) ∧ (∀ x y : ℝ, x > y ∧ x > Real.log a → f x > f y)) :=
by 
  sorry

end Monotonicity

section FunctionG

def g (x : ℝ) : ℝ := 2 * (Real.exp x - x - 1) / (x ^ 2)

theorem gx_gt_one (x : ℝ) (h : x > 0) : g x > 1 :=
by 
  sorry

end FunctionG

section Sequence

variable (x_n : ℕ → ℝ)

def seq_def : x_n 0 = 1 / 3 ∧ (∀ n : ℕ, Real.exp (x_n (n + 1)) = g (x_n n)) := sorry

theorem sequence_bound (h : seq_def x_n) (n : ℕ) : 2^n * (Real.exp (x_n n) - 1) < 1 :=
by 
  sorry

end Sequence

end monotonic_interval_fx_gx_gt_one_sequence_bound_l360_360511


namespace mean_of_integers_neg6_to_6_l360_360295

-- Define the set of integers from -6 to 6
def integers_from_neg6_to_6 := set.range (λ n : ℤ, n) ∩ set.Icc (-6 : ℤ) 6

-- Define the arithmetic mean of a finite set of integers
def arithmetic_mean (s : set ℤ) : ℚ :=
  (s.to_finset.sum id) / (s.to_finset.card : ℚ)

-- The main theorem to state the proof problem
theorem mean_of_integers_neg6_to_6 : 
  arithmetic_mean integers_from_neg6_to_6 = 0.0 :=
by
  sorry

end mean_of_integers_neg6_to_6_l360_360295


namespace part_2_part_3_l360_360034

section
  variable (A : Set ℕ)
  variables (a : ℕ → ℕ) (n k i j m : ℕ)
  -- Condition: Set A and its properties
  variable (property_P : ∀ k, 2 ≤ k → ∃ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ a k = a i + a j)
  variable (order_A : ∀ p q, 1 ≤ p ∧ p < q ∧ q ≤ n → a p < a q)

  -- Question (Ⅰ)
  def has_property_P (s : List ℕ) : Prop :=
    ∀ k, 2 ≤ k ∧ k ≤ s.length → ∃ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ s.length ∧ s.nth (k-1) = (s.nth (i-1)).bind (λ x, (s.nth (j-1)).map (λ y, x + y))

  noncomputable def check_property_1_3_4 : Bool := has_property_P [1, 3, 4]
  noncomputable def check_property_1_2_3_6 : Bool := has_property_P [1, 2, 3, 6]

  -- Question (Ⅱ): Prove that a_n ≤ 2a_1 + a_2 + ... + a_{n-1} for n ≥ 2 given property P
  theorem part_2 : n ≥ 2 → property_P A a n → a n ≤ 2 * (∑ i in Finset.range (n-1), a (i+1)) :=
    sorry

  -- Question (Ⅲ): Prove the minimum sum of A with a_n = 72
  theorem part_3 (A_n_72 : a n = 72) : ∀ A, A.nodup → set.A (a :: A) → ∃ m, m = 147 :=
    sorry
end

end part_2_part_3_l360_360034


namespace sheepdog_catches_sheep_in_20_seconds_l360_360211

noncomputable def speed_sheep : ℝ := 12 -- feet per second
noncomputable def speed_sheepdog : ℝ := 20 -- feet per second
noncomputable def initial_distance : ℝ := 160 -- feet

theorem sheepdog_catches_sheep_in_20_seconds :
  (initial_distance / (speed_sheepdog - speed_sheep)) = 20 :=
by
  sorry

end sheepdog_catches_sheep_in_20_seconds_l360_360211


namespace I2_1_I2_2_I2_3_I2_4_l360_360687

section I2_1
variables {n k : ℕ}

def cards_drawn_replaced : ℚ := (13 / 52) * (12 / 51)
def probability_of_two_hearts (a : ℕ) : Prop := cards_drawn_replaced = 1 / a

theorem I2_1 : ∃ a : ℕ, probability_of_two_hearts a := by
  use 17
  sorry
end I2_1

section I2_2
variables (n k : ℕ)

def comb (n k : ℕ) := (n.choose k)
def num_ways_to_choose (b a : ℕ) : Prop := comb a b = 136

theorem I2_2 : ∃ b, num_ways_to_choose (15) (17) := by
  use 136
  sorry
end I2_2

section I2_3
variables {b a : ℕ}

def num_signals (s : ℕ) : ℕ := (2^s) - 1
def signals_possible (c : ℕ) : Prop := num_signals 4 = c

theorem I2_3 : ∃ c, signals_possible c := by
  use 15
  sorry
end I2_3

section I2_4
variables (total red : ℕ)

def probability_red_ball : ℚ := (red / total)
def probability_red_given_c (c : ℕ) : Prop := probability_red_ball c 3 = 1 / 5

theorem I2_4 : ∃ c, probability_red_given_c c := by
  use 15
  sorry
end I2_4

end I2_1_I2_2_I2_3_I2_4_l360_360687


namespace tan_PAB_of_point_in_triangle_l360_360604

theorem tan_PAB_of_point_in_triangle 
  (A B C P : Point)
  (AB BC CA : ℝ)
  (h1 : AB = 8)
  (h2 : BC = 15)
  (h3 : CA = 17)
  (h4 : angle P A B = angle P B C)
  (h5 : angle P B C = angle P C A)
  (h6 : 0 < angle P A B ∧ angle P A B < π / 2)  -- Assuming P is inside imaginary triangle
  : tan (angle P A B) = 120 / 289 :=
sorry

end tan_PAB_of_point_in_triangle_l360_360604


namespace flagpole_break_height_l360_360778

theorem flagpole_break_height (x : ℝ) 
  (h_orig : 10) 
  (h_distance : 2) : 
  x = (2 * Real.sqrt 3) / 3 :=
sorry

end flagpole_break_height_l360_360778


namespace find_largest_prime_factor_l360_360003

def problem_statement : Prop :=
  ∀ (a b c : ℕ), a = 12^3 ∧ b = 15^4 ∧ c = 6^5 →
  (∃ p : ℕ, prime p ∧ p ∣ (a + b - c) ∧ ∀ q : ℕ, prime q ∧ q ∣ (a + b - c) → q ≤ p)

theorem find_largest_prime_factor :
  ∀ (a b c : ℕ), a = 12^3 → b = 15^4 → c = 6^5 →
  ∃ p : ℕ, prime p ∧ p ∣ (a + b - c) ∧ 
  (∀ q : ℕ, prime q ∧ q ∣ (a + b - c) → q ≤ p) :=
begin
  sorry
end

end find_largest_prime_factor_l360_360003


namespace least_tablets_l360_360752

theorem least_tablets (num_A num_B : ℕ) (hA : num_A = 10) (hB : num_B = 14) :
  ∃ n, n = 12 ∧
  ∀ extracted_tablets, extracted_tablets > 0 →
    (∃ (a b : ℕ), a + b = extracted_tablets ∧ a ≥ 2 ∧ b ≥ 2) :=
by
  sorry

end least_tablets_l360_360752


namespace equation_solution_l360_360935

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end equation_solution_l360_360935


namespace find_omega_l360_360070

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

-- Conditions
variable (ω : ℝ) (φ : ℝ)
#check (∀ x₁ x₂ : ℝ, (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/3) → f x₁ ω φ < f x₂ ω φ) -- f is monotonically increasing in (0, π/3)
#check (f (π/6) ω φ + f (π/3) ω φ = 0) -- f(π/6) + f(π/3) = 0
#check (f 0 ω φ = -1) -- f(0) = -1

theorem find_omega (ω : ℝ) (φ : ℝ) 
  (mono_incr : ∀ x₁ x₂ : ℝ, (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/3) → f x₁ ω φ < f x₂ ω φ) 
  (eqn1 : f (π/6) ω φ + f (π/3) ω φ = 0) 
  (eqn2 : f 0 ω φ = -1) : 
ω = 2 := 
sorry

end find_omega_l360_360070


namespace edge_lengths_of_tetrahedron_l360_360665

theorem edge_lengths_of_tetrahedron (lPQ lRS lPR lQR lPS lQS : ℕ)
  (h_edges : {lPQ, lRS, lPR, lQR, lPS, lQS} = {8, 15, 21, 29, 35, 45})
  (h_lPQ : lPQ = 45) :
  lRS = 8 :=
sorry

end edge_lengths_of_tetrahedron_l360_360665


namespace ones_digit_of_largest_power_of_two_dividing_factorial_l360_360861

theorem ones_digit_of_largest_power_of_two_dividing_factorial (n : ℕ) :
  (n = 5) → (nat.digits 10 (2 ^ (31))) = [8] :=
by
  intro h
  rw h
  have fact: nat.fact (2 ^ n) = 32!
  { simp [nat.fact_pow, mul_comm] }
  sorry

end ones_digit_of_largest_power_of_two_dividing_factorial_l360_360861


namespace num_elements_in_M_l360_360207

noncomputable def countM : ℕ :=
  (M : set (ℕ × ℕ)).to_finset.card

def M : set (ℕ × ℕ) := { (x, y) |
  (1 / real.sqrt x - 1 / real.sqrt y = 1 / real.sqrt 45) ∧ (x > 0) ∧ (y > 0)
}

theorem num_elements_in_M : countM = 1 :=
  sorry

end num_elements_in_M_l360_360207


namespace monotonically_increasing_f_x1_x2_greater_2x0_l360_360926

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * Real.log x - (1 / 2) * m * x^2

-- Condition for m = 1
def f_m_eq_1 := f 1

-- Define the function g(x)
def g (x : ℝ) (m : ℝ) : ℝ := f x m - (m - 4) * x

-- Statement (I): Prove that f(x) is monotonically increasing on (0, 2) when m = 1
theorem monotonically_increasing_f : ∀ x, 0 < x ∧ x < 2 → (∃ m, m = 1 ∧ f_m_eq_1 x > 0) := sorry

-- Statement (II): Prove that x₁ + x₂ > 2 * x₀
theorem x1_x2_greater_2x0 (x₁ x₂ x₀ : ℝ) (h1 : x₁ ≠ x₂) (m : ℝ) (h2 : m > 0) (k : ℝ) 
  (hmn : k = g'(x₀)) : x₁ + x₂ > 2 * x₂ := sorry

end monotonically_increasing_f_x1_x2_greater_2x0_l360_360926


namespace combined_salaries_correct_l360_360663

noncomputable def combined_salaries_BCDE (A B C D E : ℕ) : Prop :=
  (A = 8000) →
  ((A + B + C + D + E) / 5 = 8600) →
  (B + C + D + E = 35000)

theorem combined_salaries_correct 
  (A B C D E : ℕ) 
  (hA : A = 8000) 
  (havg : (A + B + C + D + E) / 5 = 8600) : 
  B + C + D + E = 35000 :=
sorry

end combined_salaries_correct_l360_360663


namespace zeros_of_f_in_interval_l360_360959

noncomputable def f (x : ℝ) : ℝ := x * Real.cos (x^2)

theorem zeros_of_f_in_interval : (∃ x₁ x₂ x₃ x₄ x₅ ∈ set.Icc 0 4, ∀ x ∈ set.Icc 0 4, f x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ ∨ x = x₅) :=
by
  sorry

end zeros_of_f_in_interval_l360_360959


namespace ukuleles_and_violins_l360_360021

theorem ukuleles_and_violins (U V : ℕ) : 
  (4 * U + 6 * 4 + 4 * V = 40) → (U + V = 4) :=
by
  intro h
  sorry

end ukuleles_and_violins_l360_360021


namespace distinct_four_digit_numbers_with_repeated_digit_l360_360085

open Finset

theorem distinct_four_digit_numbers_with_repeated_digit :
  let digits := {1, 2, 3, 4, 5}
  in (∃ d ∈ digits, 
        ∃ (positions : Finset (Fin 4)) (h_pos : positions.card = 2)
        (d1 d2 ∈ digits \ {d}),
        positions.pairwise (≠) ∧ (d = d1 ∨ d = d2) ∧ d1 ≠ d2)
      → 5 * 6 * 4 * 3 = 360 :=
by
  sorry

end distinct_four_digit_numbers_with_repeated_digit_l360_360085


namespace equilateral_triangle_ratio_l360_360718

theorem equilateral_triangle_ratio (r : ℝ) (h : r > 0) :
  let A1 := (sqrt 3) * r^2
  let A2 := 16 * r^2 / 3
  A1 / A2 = 3 * sqrt 3 / 16 :=
by sorry

end equilateral_triangle_ratio_l360_360718


namespace change_for_50_cents_using_one_of_each_coin_l360_360525
open Nat

def coins := List (Nat × Nat)
def valid_coins := [(1, 1), (5, 5), (10, 10), (25, 25)]

def is_valid_combination (c : coins) : Bool :=
  (c.all (λ (coin, value), (coin, value) ∈ valid_coins)) ∧ 
  (c.length = List.eraseDupBy (·.1) c).length ∧ 
  (c.sum (λ (coin, value), value) = 50) ∧ 
  (∀ coin, List.count (·.1 coin) c ≤ 1)

noncomputable def possible_combinations := 
  (List.sublists' valid_coins).filter is_valid_combination

theorem change_for_50_cents_using_one_of_each_coin : 
  List.length possible_combinations = 8 :=
by
  sorry

end change_for_50_cents_using_one_of_each_coin_l360_360525


namespace cyclist_wait_time_l360_360343

noncomputable def hiker_speed : ℝ := 5 / 60
noncomputable def cyclist_speed : ℝ := 25 / 60
noncomputable def wait_time : ℝ := 5
noncomputable def distance_ahead : ℝ := cyclist_speed * wait_time
noncomputable def catching_time : ℝ := distance_ahead / hiker_speed

theorem cyclist_wait_time : catching_time = 25 := by
  sorry

end cyclist_wait_time_l360_360343


namespace circles_intersect_at_two_points_l360_360605

noncomputable def point_intersection_count (A B : ℝ × ℝ) (rA rB d : ℝ) : ℕ :=
  let distance := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2
  if rA + rB >= d ∧ d >= |rA - rB| then 2 else if d = rA + rB ∨ d = |rA - rB| then 1 else 0

theorem circles_intersect_at_two_points :
  point_intersection_count (0, 0) (8, 0) 3 6 8 = 2 :=
by 
  -- Proof for the statement will go here
  sorry

end circles_intersect_at_two_points_l360_360605


namespace g_eq_half_l360_360927

noncomputable def f (x : ℝ) : ℝ :=
  2 / (4^x + 2)

noncomputable def g (n : ℕ) : ℝ :=
  f 0 + (Finset.range n).sum (λ k, f ((k:ℕ+1) / n)) + f 1

theorem g_eq_half (n : ℕ) (hn : n > 0) : g n = (n + 1) / 2 :=
  sorry

end g_eq_half_l360_360927


namespace gcd_lcm_lemma_l360_360581

theorem gcd_lcm_lemma (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 33) (h_lcm : Nat.lcm a b = 90) : Nat.gcd a b = 3 :=
by
  sorry

end gcd_lcm_lemma_l360_360581


namespace parabola_shift_l360_360263

theorem parabola_shift (x : ℝ) : 
  let y := 3 * x^2 in
  (fun x => 3 * (x - 1)^2 - 2) = (fun x => y)[x := (x - 1)] - 2 :=
sorry

end parabola_shift_l360_360263


namespace length_bisector_in_triangle_l360_360134

theorem length_bisector_in_triangle
  (K L M N P Q : Point)
  (k m q : ℝ)
  (h_triangle : Triangle K L M)
  (h_bisector : Bisector M N)
  (h_circle : Circle M N P Q)
  (h_touch : TouchesAt N K L)
  (h_intersect_KM : IntersectsAt P K M)
  (h_intersect_LM : IntersectsAt Q L M)
  (h_KP : Distance K P = k)
  (h_QM : Distance Q M = m)
  (h_LQ : Distance L Q = q) :
  Distance M N = sqrt (km * (m + q) / q) := sorry

end length_bisector_in_triangle_l360_360134


namespace cos_pi_minus_alpha_l360_360477

theorem cos_pi_minus_alpha (α : ℝ) (h : sin (α / 2) = 2 / 3) : cos (π - α) = -1 / 9 :=
sorry

end cos_pi_minus_alpha_l360_360477


namespace problem_1_problem_2_l360_360496

-- Definition of the triangle and knowns
variables (a b c : ℝ) (A B C : ℝ) (S : ℝ)

-- Conditions
axiom cos_ratio : cos B / cos C = b / (2 * a - c)
axiom tan_relation : 1 / tan A + 1 / tan B = sin C / (sqrt 3 * sin A * cos B)
axiom area_eq : 4 * sqrt 3 * S + 3 * (b^2 - a^2) = 3 * c^2
axiom side_b : b = 2 * sqrt 3
axiom sides_sum_eq : a + c = 4

-- Problem (1)
theorem problem_1 : S = sqrt 3 / 3 := 
sorry

-- Acute-angled triangle condition
axiom acute_triangle : 0 < A ∧ A < pi/2 ∧ 0 < B ∧ B < pi/2 ∧ 0 < C ∧ C < pi/2

-- Problem (2)
theorem problem_2 : (sqrt 3 + 1) / 2 < (b + c) / a ∧ (b + c) / a < sqrt 3 + 2 :=
sorry

end problem_1_problem_2_l360_360496


namespace lucy_finishes_20th_book_on_sunday_l360_360208

/-- Lucy read 20 books, one at a time. 
The first book took her 1 day to read, 
the second book took her 2 days, 
the third book took her 3 days, and so on, 
with each book taking her 1 more day to read than the previous book. 
Lucy started reading her first book on a Sunday.
Prove that she finished her 20th book on a Sunday. 
--/
theorem lucy_finishes_20th_book_on_sunday :
  let total_days := (20 * (20 + 1)) / 2 in
  total_days % 7 = 0 :=
by
  sorry

end lucy_finishes_20th_book_on_sunday_l360_360208


namespace sphere_cross_section_property_l360_360805

theorem sphere_cross_section_property 
    {S : Type*} [metric_space S] [sphere S]
    (r : ℝ) (d1 d2 : ℝ) (hc1 : d1 = d2) (hc2 : d1 < d2) :
    ∀ (c1 c2 : set S), 
    (c1 = {p : S | dist p (center S) = r - d1}) →
    (c2 = {p : S | dist p (center S) = r - d2}) →
    (measure c1 = measure c2) → 
    (measure c1 > measure c2) :=
by { sorry }

end sphere_cross_section_property_l360_360805


namespace problem_statement_l360_360078

-- Definitions of sets S and P
def S : Set ℝ := {x | x^2 - 3 * x - 10 < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2 * a + 15}

-- Proof statement
theorem problem_statement (a : ℝ) : 
  (S = {x | -2 < x ∧ x < 5}) ∧ (S ⊆ P a → a ∈ Set.Icc (-5 : ℝ) (-3 : ℝ)) :=
by
  sorry

end problem_statement_l360_360078


namespace part_a_part_b_l360_360324

theorem part_a (lines : set (line ℝ)) (h : ∀ (l1 l2 l3 ∈ lines), ∃ (c : circle ℝ), c.radius = 1 ∧ ∀ (l ∈ {l1, l2, l3}, c ∩ l ≠ ∅)) :
  ∃ (c : circle ℝ), c.radius = 1 ∧ ∀ (l ∈ lines, c ∩ l ≠ ∅) :=
sorry

theorem part_b (lines : set (line ℝ)) (h : diameter lines ≤ 1) :
  ∃ (c : circle ℝ), (∀ (l ∈ lines, c ∩ l ≠ ∅)) ∧ c.radius ≤ real.sqrt 3 / 6 :=
sorry

end part_a_part_b_l360_360324


namespace repeating_decimals_sum_l360_360422

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l360_360422


namespace circle_radius_l360_360630

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l360_360630


namespace Timmy_needs_additional_speed_l360_360283

noncomputable def Timmy_speeds : List ℝ := [36, 34, 38]
noncomputable def wind_resistance_range : Set ℝ := { x | 3 ≤ x ∧ x ≤ 5 }
noncomputable def required_speed_no_resistance : ℝ := 40

def average (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def average_speed : ℝ :=
  average Timmy_speeds

noncomputable def average_wind_resistance : ℝ :=
  (3 + 5) / 2

noncomputable def total_required_speed : ℝ :=
  required_speed_no_resistance + average_wind_resistance

noncomputable def speed_difference : ℝ :=
  total_required_speed - average_speed

theorem Timmy_needs_additional_speed :
  speed_difference = 8 :=
by
  sorry

end Timmy_needs_additional_speed_l360_360283


namespace sum_of_repeating_decimals_l360_360414

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l360_360414


namespace extremum_range_of_m_l360_360106

theorem extremum_range_of_m (m : ℝ) : (∃ x : ℝ, (λ x => exp x + m * x).deriv x = 0) ↔ m < 0 :=
by
  sorry

end extremum_range_of_m_l360_360106


namespace repeating_decimals_sum_l360_360426

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l360_360426


namespace solve_problem_l360_360033

noncomputable def a : ℕ → ℕ
| 0       := 0
| (k + 1) := if (k + 1) % 4 = 1 then 1 else
             if (k + 1) % 4 = 3 then 0 else
             a ((k + 1) / 2)

lemma a_4n_3 (n : ℕ) (hn : 1 ≤ n) : a (4 * n - 3) = 1 := by
  sorry

lemma a_4n_1 (n : ℕ) (hn : 1 ≤ n) : a (4 * n - 1) = 0 := by
  sorry

lemma a_2n (n : ℕ) (hn : 1 ≤ n) : a (2 * n) = a n := by
  sorry

theorem solve_problem : a 2009 + a 2014 = 1 := by
  have h1 : a 2009 = 1 := by
    sorry
  have h2 : a 2014 = 0 := by
    sorry
  rw [h1, h2]
  norm_num

end solve_problem_l360_360033


namespace roof_collapse_l360_360374

theorem roof_collapse (roof_limit : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) :
  roof_limit = 500 → leaves_per_day = 100 → leaves_per_pound = 1000 → 
  let d := (roof_limit * leaves_per_pound) / leaves_per_day in d = 5000 :=
by
  intros h₁ h₂ h₃
  sorry

end roof_collapse_l360_360374


namespace range_of_a_l360_360060

theorem range_of_a (a : ℝ) :
  let p1 := (2 : ℝ, 1 : ℝ)
      p2 := (-2 : ℝ, 3 : ℝ)
      line_val (x y : ℝ) := 3 * x - 2 * y + a in
  (line_val p1.1 p1.2) * (line_val p2.1 p2.2) < 0 ↔ -4 < a ∧ a < 12 := 
sorry

end range_of_a_l360_360060


namespace distinct_values_of_S_l360_360394

-- Definition of the problem conditions
def i : ℂ := Complex.I

-- Definition of the expression S
def S (n : ℤ) := (i^n + i^(-n))^2

-- Statement of the theorem to prove
theorem distinct_values_of_S : 
  ∃ D : Finset ℂ, D = {S 0, S 1, S 2, S 3} ∧ D.card = 2 :=
by
  -- To be filled with proof
  sorry

end distinct_values_of_S_l360_360394


namespace select_non_intersecting_chords_l360_360144

theorem select_non_intersecting_chords 
  (n : ℕ) (k : ℕ) (h1 : 2 * k + 1 < n) (chords : set (ℕ × ℕ)) 
  (h2 : chords.card = n * k + 1) 
  (h3 : ∀ chord ∈ chords, chord.1 < chord.2 ∧ chord.1 ≤ n ∧ chord.2 ≤ n) :
  ∃ subset_chords : set (ℕ × ℕ), subset_chords ⊆ chords ∧ subset_chords.card = k + 1 ∧ 
  (∀ c1 c2 ∈ subset_chords, c1 ≠ c2 → ¬(c1 ∩ c2).nonempty) :=
sorry

end select_non_intersecting_chords_l360_360144


namespace thor_hammer_weight_exceeds_2000_l360_360625

/--  The Mighty Thor uses a hammer that doubles in weight each day as he trains.
      Starting on the first day with a hammer that weighs 7 pounds, prove that
      on the 10th day the hammer's weight exceeds 2000 pounds. 
-/
theorem thor_hammer_weight_exceeds_2000 :
  ∃ n : ℕ, 7 * 2^(n - 1) > 2000 ∧ n = 10 :=
by
  sorry

end thor_hammer_weight_exceeds_2000_l360_360625


namespace probability_line_does_not_intersect_circle_l360_360482

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def line_eq (x y k : ℝ) : Prop := y = k * (x + 3)
def k_interval (k : ℝ) : Prop := -1 ≤ k ∧ k ≤ 1

-- Theorem statement
theorem probability_line_does_not_intersect_circle :
  (∀ l C : Prop, l = ∀ x y k : ℝ, line_eq x y k → C = ∀ x y : ℝ, circle_eq x y → 
  ∀ k ∈ k_interval, 
  (k < - (Real.sqrt 2) / 4 ∨ k > (Real.sqrt 2) / 4) → 
  ((2 * (1 - (Real.sqrt 2) / 4)) / 2 = (4 - Real.sqrt 2) / 4)) :=
by
  sorry

end probability_line_does_not_intersect_circle_l360_360482


namespace semicircle_parametric_equation_correct_l360_360126

-- Define the conditions of the problem in terms of Lean definitions and propositions.

def semicircle_parametric_equation : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ (Real.pi / 2) →
    ∃ α : ℝ, α = 2 * θ ∧ 0 ≤ α ∧ α ≤ Real.pi ∧
    (∃ (x y : ℝ), x = 1 + Real.cos α ∧ y = Real.sin α)

-- Statement that we will prove
theorem semicircle_parametric_equation_correct : semicircle_parametric_equation :=
  sorry

end semicircle_parametric_equation_correct_l360_360126


namespace roof_collapse_days_l360_360372

-- Definitions based on the conditions
def roof_capacity_pounds : ℕ := 500
def leaves_per_pound : ℕ := 1000
def leaves_per_day : ℕ := 100

-- Statement of the problem and the result
theorem roof_collapse_days :
  let total_leaves := roof_capacity_pounds * leaves_per_pound in
  let days := total_leaves / leaves_per_day in
  days = 5000 :=
by
  -- To be proven, so we use sorry for now
  sorry

end roof_collapse_days_l360_360372


namespace genetic_disorder_probability_l360_360242

-- Given conditions:
def p_D := 1 / 200
def p_Dc := 1 - p_D
def p_T_given_D := 1
def p_T_given_Dc := 0.05

-- Goal: Prove that the probability that a person who tests positive actually has the disorder is 20/219
theorem genetic_disorder_probability :
  let p_T := p_T_given_D * p_D + p_T_given_Dc * p_Dc in
  (p_T_given_D * p_D / p_T) = (20 / 219) :=
by
  -- This example is focused on setting up the correct Lean statement, and does not require a proof
  sorry

end genetic_disorder_probability_l360_360242


namespace dot_product_magnitude_l360_360180

variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜 → 𝕜} 

-- Define the conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_a_cross_b : ℝ := 6

-- Theorem statement
theorem dot_product_magnitude :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∥a × b∥ = norm_a_cross_b) → abs ((a.toReal).dot (b.toReal)) = 6 * sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end dot_product_magnitude_l360_360180


namespace unique_plane_through_line_parallel_to_skew_line_l360_360102

open Plane

-- Statement of the problem
theorem unique_plane_through_line_parallel_to_skew_line 
  (a b : Line) (h_skew: ¬∃ P : Plane, a ≤ P ∧ b ≤ P) : 
  ∃! P : Plane, a ≤ P ∧ parallel P b :=
sorry

end unique_plane_through_line_parallel_to_skew_line_l360_360102


namespace minimum_points_in_polygon_l360_360488

/-!
# Problem:
Given an \( n \)-sided polygon with \( k \) points distributed inside it, such that every triangle formed by any 3 vertices of the polygon contains at least 1 point, find the minimum value of \( k \).

## Conditions:
- An \( n \)-sided polygon with vertices \(A_1, A_2, \ldots, A_n\)
- \(k\) points inside the polygon
- Every triangle formed by any 3 vertices of the polygon contains at least one of these \(k\) points

## Conclusion:
The minimum number of \( k \) such points is \( n-2 \).
-/

theorem minimum_points_in_polygon (n : ℕ) : 
  ∃ k, (∀ (A : fin n → Prop) (points : fin k → Prop) (triangle_contains_point : ∀ (i j l : fin n), ∃ p, points p ∧ triangle (A i) (A j) (A l) contains p), k >= n - 2) := 
sorry

end minimum_points_in_polygon_l360_360488


namespace integer_points_on_circle_l360_360310

theorem integer_points_on_circle {r : ℕ} (h : r = 5) :
  {p : ℤ × ℤ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}.to_finset.card = 12 :=
by
  sorry

end integer_points_on_circle_l360_360310


namespace isosceles_right_triangle_hypotenuse_length_l360_360555

/-- In an isosceles right triangle ABC with AC = BC, D bisects hypotenuse AB. 
From vertex C, perpendicular cevians CE and CF are drawn to AD and DB, respectively.
Given that CE = sin x and CF = cos x for 0 < x < π/2,
the length of hypotenuse AB is equal to 1. -/
theorem isosceles_right_triangle_hypotenuse_length
  (A B C D E F : Type)
  (AC BC CE CF : ℝ)
  (x : ℝ)
  (h_AC_eq_BC : AC = BC)
  (h_D_bisects_AB : D = midpoint ℝ A B)
  (h_CE_perpendicular_AD : CE ⊥ (line A D))
  (h_CF_perpendicular_DB : CF ⊥ (line B D))
  (h_CE_eq_sin_x : CE = sin x)
  (h_CF_eq_cos_x : CF = cos x)
  (h_x_range : 0 < x ∧ x < π / 2) :
  c = 1 :=
by
  sorry

end isosceles_right_triangle_hypotenuse_length_l360_360555


namespace sum_of_roots_l360_360738

theorem sum_of_roots (a b c: ℝ) (h: a ≠ 0) (h_eq : a = 1) (h_eq2 : b = -6) (h_eq3 : c = 8):
    let Δ := b ^ 2 - 4 * a * c in
    let root1 := (-b + real.sqrt Δ) / (2 * a) in
    let root2 := (-b - real.sqrt Δ) / (2 * a) in
    root1 + root2 = 6 :=
by
  sorry

end sum_of_roots_l360_360738


namespace root_interval_l360_360766

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 2 * x - 1

theorem root_interval : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
by
  have h_decreasing : ∀ x y : ℝ, x < y → f x < f y :=
    sorry -- Proof that f is increasing on (-1, +∞)
  have h_f0 : f 0 = -1 := by
    sorry -- Calculation that f(0) = -1
  have h_f1 : f 1 = Real.log 2 + 1 := by
    sorry -- Calculation that f(1) = ln(2) + 1
  have h_exist_root : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
    by
      sorry -- Existence of a root in (0,1)
  exact h_exist_root

end root_interval_l360_360766


namespace bees_in_beehive_after_six_days_l360_360280

theorem bees_in_beehive_after_six_days : 
  ∃ n, n = 6 ∧ let a₁ := 4 in S₆ := a₁ * (1 - 3^6) / (1 - 3) in S₆ = 1456 :=
by
  use 6
  have a₁ := 4
  let S₆ := a₁ * (1 - 3^6) / (1 - 3)
  show S₆ = 1456
  sorry

end bees_in_beehive_after_six_days_l360_360280


namespace points_on_equation_correct_l360_360948

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l360_360948


namespace f_value_l360_360888

noncomputable def f (α : Real) : Real :=
  (Real.tan (π + α) * Real.sin (2 * π - α) * Real.cos (π - α)) /
  (Real.sin (α + (3/2 : Real) * π) * Real.cos (α - (3/2 : Real) * π))

theorem f_value (α : Real) (hα : Real.sin α = -2/3) (h_range : α ∈ Icc (-π) (-π / 2)) :
  f α = 2 * Real.sqrt 5 / 5 :=
sorry

end f_value_l360_360888


namespace size_of_angle_C_length_of_side_c_l360_360045

-- Given that A, B, and C are the angles of triangle ABC and vectors m and n form the given relationship.
variables (A B C : ℝ)
variables (a b c : ℝ) -- lengths of the sides opposite to A, B, and C

-- Given vectors and their dot product relation to sin 2C
def vector_m : ℝ × ℝ := (Real.sin A, Real.sin B)
def vector_n : ℝ × ℝ := (Real.cos B, Real.cos A)

-- Assuming known conditions for the dot product and arithmetic sequence
axiom dot_product_eq : vector_m A B • vector_n A B = Real.sin (2 * C)

-- The size of angle C is π/3 (Problem 1)
theorem size_of_angle_C : A + B + C = π ∧ dot_product_eq A B C → C = π / 3 :=
by
  sorry

-- Given specific condition for problem 2, find the length of side c
axiom sin_arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B
axiom CA_AB_AC_condition : 18 = a * b * (1 / 2) -- Using cos C = 1/2 from previous step's result

theorem length_of_side_c : 
  (A + B + C = π) ∧ (dot_product_eq A B C) ∧ (C = π / 3) ∧ (sin_arithmetic_sequence A B C) ∧ (CA_AB_AC_condition a b) →
  c = 6 :=
by
  sorry

end size_of_angle_C_length_of_side_c_l360_360045


namespace sum_of_extrema_of_f_l360_360512

noncomputable def f : ℝ → ℝ := λ x, x * Real.log (|x|) + 1

theorem sum_of_extrema_of_f : 
  (let min_val := ∃ x, x > 0 ∧ x = (1 / Real.exp 1) ∧ f x = 1 - (1 / Real.exp 1),
       max_val := ∃ x, x < 0 ∧ x = -(1 / Real.exp 1) ∧ f x = 1 + (1 / Real.exp 1)
  in min_val ∧ max_val) → 
  (∀ min_val max_val, f min_val + f max_val = 2) := sorry

end sum_of_extrema_of_f_l360_360512


namespace repeating_decimal_sum_l360_360405

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l360_360405


namespace dot_product_magnitude_l360_360154

variable {a b : EuclideanSpace ℝ (Fin n)}
variable (norm_a : ∥a∥ = 3)
variable (norm_b : ∥b∥ = 4)
variable (norm_cross_ab : ∥a × b∥ = 6)

theorem dot_product_magnitude : ∥a∥ = 3 → ∥b∥ = 4 → ∥a × b∥ = 6 → |a ∘ b| = 6 * real.sqrt 3 := 
by 
  intro norm_a norm_b norm_cross_ab
  sorry

end dot_product_magnitude_l360_360154


namespace equilateral_triangle_ratio_l360_360719

theorem equilateral_triangle_ratio (r : ℝ) (h : r > 0) :
  let A1 := (sqrt 3) * r^2
  let A2 := 16 * r^2 / 3
  A1 / A2 = 3 * sqrt 3 / 16 :=
by sorry

end equilateral_triangle_ratio_l360_360719


namespace base5_first_digit_927_l360_360714

theorem base5_first_digit_927 :
  let n := 927
  let b := 5
  let base_5_repr := Nat.digits b n
  base_5_repr.head = 1 := by
  sorry

end base5_first_digit_927_l360_360714


namespace seven_digit_div_by_11_l360_360541

def divisible_by_eleven (n : ℕ) : Prop :=
  (λ n : ℕ, let digits := n.digits 10 in
    (digits.enum.filter (λ p, p.1 % 2 = 0)).sum - 
    (digits.enum.filter (λ p, p.1 % 2 = 1)).sum) n % 11 = 0

theorem seven_digit_div_by_11 (m : ℕ) (h1 : (7 * 1000000 + 4 * 100000 + 2 * 10000 + m * 1000 + 8 * 100 + 3 * 10 + 4) % 11 = 0) : m = 3 :=
sorry

end seven_digit_div_by_11_l360_360541


namespace length_of_CFD_l360_360583

-- Define a triangle ABC with its vertices and the midpoints of its sides
variable {A B C D E F G : Point}
variable [Triangle ABC]
variable [Midpoint D BC]
variable [Midpoint E AC]
variable [Midpoint F AB]
variable [Centroid G ABC]

-- Given conditions
variable (h1 : AD ⊥ BE)
variable (h2 : Length AD = 18)
variable (h3 : Length BE = 13.5)

-- Theorem: Calculate the length of the third median CF
theorem length_of_CFD {CF : Real} :
  Length CF = 22.5 := by
  sorry

end length_of_CFD_l360_360583


namespace radius_of_circle_with_area_64pi_l360_360631

def circle_radius (A : ℝ) : ℝ := 
  real.sqrt (A / real.pi)

theorem radius_of_circle_with_area_64pi :
  circle_radius (64 * real.pi) = 8 :=
by sorry

end radius_of_circle_with_area_64pi_l360_360631


namespace dinosaur_book_cost_l360_360570

theorem dinosaur_book_cost (D : ℕ) : 
  (11 + D + 7 = 37) → (D = 19) := 
by 
  intro h
  sorry

end dinosaur_book_cost_l360_360570


namespace point_inside_circle_A_is_inside_circle_l360_360764

def radius : ℝ := 4
def distance_to_center : ℝ := 3

theorem point_inside_circle (r d : ℝ) (hr : r = radius) (hd : d = distance_to_center) : d < r → A_inside_circle :=
by
  assume h : d < r
  -- Here the proof would be, but it is omitted
  sorry

-- Definition of the point being inside the circle
def A_inside_circle : Prop := distance_to_center < radius

theorem A_is_inside_circle : A_inside_circle :=
by
  -- Demonstrating the given specific values
  have hr : radius = 4 := rfl
  have hd : distance_to_center = 3 := rfl
  have h : distance_to_center < radius := by
    rw [hd, hr]
    exact lt_of_lt_of_le (by norm_num) (le_refl 4)
  -- Therefore, the point A is inside the circle
  exact point_inside_circle radius distance_to_center hr hd h

end point_inside_circle_A_is_inside_circle_l360_360764


namespace probability_of_quarter_l360_360344

theorem probability_of_quarter :
  let value_quarters := 10.00
  let value_nickels := 10.00
  let value_pennies := 10.00
  let num_quarters := value_quarters / 0.25
  let num_nickels := value_nickels / 0.05
  let num_pennies := value_pennies / 0.01
  let total_coins := num_quarters + num_nickels + num_pennies
  let probability := num_quarters / total_coins
  in probability = 1 / 31 :=
by sorry

end probability_of_quarter_l360_360344


namespace min_value_of_function_at_extreme_point_l360_360094

noncomputable def f (x : ℝ) (a : ℝ) := (1/2) * x^2 - a * x - 3 * Real.log x

theorem min_value_of_function_at_extreme_point :
  (∀ a : ℝ, ∃ (x : ℝ) (hx : x = 3), f x a = f x 2 → f 3 2 = -(3/2) - 3 * Real.log 3) :=
begin
  sorry
end

end min_value_of_function_at_extreme_point_l360_360094


namespace second_quadrant_necessary_not_sufficient_l360_360763

variable (α : ℝ)

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → is_second_quadrant α) ∧ ¬ (∀ α, is_second_quadrant α → is_obtuse α) := by
  sorry

end second_quadrant_necessary_not_sufficient_l360_360763


namespace repeating_decimals_sum_l360_360427

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l360_360427


namespace radius_of_circle_l360_360637

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l360_360637


namespace no_solution_nat_x_satisfies_eq_l360_360197

def sum_digits (x : ℕ) : ℕ := x.digits 10 |>.sum

theorem no_solution_nat_x_satisfies_eq (x : ℕ) :
  ¬ (x + sum_digits x + sum_digits (sum_digits x) = 2014) :=
by
  sorry

end no_solution_nat_x_satisfies_eq_l360_360197


namespace books_selection_l360_360223

def choose_books : ℕ := binomial 6 4 * binomial 4 2 * binomial 10 2

theorem books_selection :
  ∃ n, n = 4050 ∧ choose_books = n :=
by
  use 4050
  split
  rfl
  sorry

end books_selection_l360_360223


namespace loop_while_i_leq_10_l360_360301

noncomputable def loop_value : ℕ :=
  let rec loop i :=
    if i ≤ 10 then loop (i + 1) else i
  loop 1

theorem loop_while_i_leq_10 : loop_value = 11 := by
  sorry

end loop_while_i_leq_10_l360_360301


namespace given_problem_l360_360915

variables {ℝ : Type} [linear_ordered_field ℝ]
variable (f : ℝ → ℝ)

-- Given that f is monotonically increasing on [1, +∞)
def monotone_increasing_on_Ici (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 1 ≤ x → x ≤ y → f x ≤ f y

-- Given that f(x+1) is an even function
def even_function_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x+1) = f(-x+1)

theorem given_problem (h1 : monotone_increasing_on_Ici f) (h2 : even_function_shifted f) :
  f(-2) > f(2) :=
sorry

end given_problem_l360_360915


namespace hyperbolic_linear_interaction_l360_360653

variables {x k : ℝ}

theorem hyperbolic_linear_interaction (h1 : k < 1) (h2 : |k| < 1) :
  let y_hyp := (k - 1) / x in
  let y_lin := k * (x + 1) in
  (y_hyp * y_lin = (k-1) * k * (x + 1) / x) ∧  -- Expression for intersection
  (k - 1 < 0) := sorry

end hyperbolic_linear_interaction_l360_360653


namespace archer_weekly_spend_l360_360362

noncomputable def total_shots_per_week (shots_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  shots_per_day * days_per_week

noncomputable def arrows_recovered (total_shots : ℕ) (recovery_rate : ℝ) : ℕ :=
  (total_shots : ℝ) * recovery_rate |> Int.ofNat

noncomputable def net_arrows_used (total_shots : ℕ) (recovered_arrows : ℕ) : ℕ :=
  total_shots - recovered_arrows

noncomputable def cost_of_arrows (net_arrows : ℕ) (cost_per_arrow : ℝ) : ℝ :=
  net_arrows * cost_per_arrow

noncomputable def team_contribution (total_cost : ℝ) (contribution_rate : ℝ) : ℝ :=
  total_cost * contribution_rate

noncomputable def archer_spend (total_cost : ℝ) (team_contribution : ℝ) : ℝ :=
  total_cost - team_contribution

theorem archer_weekly_spend
  (shots_per_day : ℕ) (days_per_week : ℕ) (recovery_rate : ℝ)
  (cost_per_arrow : ℝ) (contribution_rate : ℝ) :
  archer_spend
    (cost_of_arrows
      (net_arrows_used
        (total_shots_per_week shots_per_day days_per_week)
        (arrows_recovered
          (total_shots_per_week shots_per_day days_per_week)
          recovery_rate))
      cost_per_arrow)
    (team_contribution
      (cost_of_arrows
        (net_arrows_used
          (total_shots_per_week shots_per_day days_per_week)
          (arrows_recovered
            (total_shots_per_week shots_per_day days_per_week)
            recovery_rate))
        cost_per_arrow)
      contribution_rate) = 1056 :=
by
  sorry

end archer_weekly_spend_l360_360362


namespace parallelogram_perimeter_l360_360112

variables {X Y Z G H I : Type} [MetricSpace X]
variables (dist : X → X → ℝ)
variables (XY YZ XZ : ℝ)
variables (G_on_XY : Real)
variables (GH_is_parallel_XZ : Bool)
variables (HI_is_parallel_YZ : Bool)

-- Given conditions
def conditions := XY = 18 ∧ YZ = 18 ∧ XZ = 14 ∧ 
                  G_on_XY = 3 / 5 ∧ 
                  GH_is_parallel_XZ = true ∧ 
                  HI_is_parallel_YZ = true

-- Prove the perimeter of parallelogram XGHI
theorem parallelogram_perimeter (h : conditions) : 
  dist X G + dist G H + dist H I + dist I X = 30.8 :=
sorry

end parallelogram_perimeter_l360_360112


namespace canal_depth_l360_360644

theorem canal_depth (A : ℝ) (w_top w_bottom : ℝ) (h : ℝ) 
    (hA : A = 10290) 
    (htop : w_top = 6) 
    (hbottom : w_bottom = 4) 
    (harea : A = 1 / 2 * (w_top + w_bottom) * h) : 
    h = 2058 :=
by
  -- here goes the proof steps
  sorry

end canal_depth_l360_360644


namespace repeating_decimal_sum_in_lowest_terms_l360_360437

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l360_360437


namespace min_value_3_pow_a_plus_3_pow_b_l360_360537

theorem min_value_3_pow_a_plus_3_pow_b (a b : ℝ) (h : a + b = 2) : 3^a + 3^b ≥ 6 := 
sorry

end min_value_3_pow_a_plus_3_pow_b_l360_360537


namespace number_of_boxes_l360_360567

theorem number_of_boxes (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → exists k : ℕ, ((k = 6) ∧ ((n - i + 1) + k = n - i + 1))
    ∨ ((n - i + 1) = k + 16) (exists (m : ℕ), n = 11 * m)) :=
by sorry

end number_of_boxes_l360_360567


namespace solution_set_l360_360055

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≥ 0 then x^2 - 6*x else -(x^2 - 6*(-x))

theorem solution_set (a : ℝ) :
(f 0 = 0 ∧ ∀ x, f(-x) = -f(x) ∧ (∀ x ≥ 0, f(x) = x^2 - 6*x)) →
  {x | f x < |x|} = {x | x > 0 ∧ x < 7} ∪ {x | x < -5} :=
begin
  sorry
end

end solution_set_l360_360055


namespace sequence_is_geometric_bounds_on_S_n_l360_360919

def geom_seq (a_n : ℕ → ℝ) :=
  ∃ r a_0, ∀ n, a_n n = a_0 * r^n

theorem sequence_is_geometric
  (S_n : ℕ → ℝ)
  (h : ∀ n, S_n n = 3 - 2 * (λ a_n:ℕ → ℝ, a_n n))
  (a_n : ℕ → ℝ)
  (ha : ∀ n, a_n n = if n = 0 then 1 else (2/3)^n) :
  geom_seq a_n :=
sorry

theorem bounds_on_S_n
  (S_n : ℕ → ℝ)
  (a_n : ℕ → ℝ)
  (h : ∀ n, S_n n = 3 - 2 * (λ a_n:ℕ → ℝ, a_n n))
  (ha : ∀ n, a_n n = if n = 0 then 1 else (2/3)^n) :
  ∀ n : ℕ, 1 ≤ S_n n ∧ S_n n < 3 :=
sorry

end sequence_is_geometric_bounds_on_S_n_l360_360919


namespace percentage_reduction_price_increase_l360_360337

open Real

-- Part 1: Finding the percentage reduction each time
theorem percentage_reduction (P₀ P₂ : ℝ) (x : ℝ) (h₀ : P₀ = 50) (h₁ : P₂ = 32) (h₂ : P₀ * (1 - x) ^ 2 = P₂) :
  x = 0.20 :=
by
  dsimp at h₀ h₁,
  rw h₀ at h₂,
  rw h₁ at h₂,
  simp at h₂,
  sorry

-- Part 2: Determining the price increase per kilogram
theorem price_increase (P y : ℝ) (profit_per_kg : ℝ) (initial_sales : ℝ) 
  (price_increase_limit : ℝ) (sales_decrease_rate : ℝ) (target_profit : ℝ)
  (h₀ : profit_per_kg = 10) (h₁ : initial_sales = 500) (h₂ : price_increase_limit = 8)
  (h₃ : sales_decrease_rate = 20) (h₄ : target_profit = 6000) (0 < y ∧ y ≤ price_increase_limit)
  (h₅ : (profit_per_kg + y) * (initial_sales - sales_decrease_rate * y) = target_profit) :
  y = 5 :=
by
  dsimp at h₀ h₁ h₂ h₃ h₄,
  rw [h₀, h₁, h₂, h₃, h₄] at h₅,
  sorry

end percentage_reduction_price_increase_l360_360337


namespace square_area_minimized_l360_360016

theorem square_area_minimized 
  (a b : ℝ) (h_line : ∀ (x : ℝ), a = 2 * x + 3 ∨ b = 2 * x + 3)
  (h_parabola : ∀ (x : ℝ), a = x^2 ∨ b = x^2)
  : 200 = Inf (set_of (λ (A : ℝ), exists x y : ℝ, 
      (A = (x - y)^2 * (1 + (x + y)^2)) ∧ (2 * x + 3 = a ∨ a = x^2) 
      ∧ (2 * y + 3 = b ∨ b = y^2))) :=
sorry

end square_area_minimized_l360_360016


namespace radius_of_circle_with_area_64pi_l360_360633

def circle_radius (A : ℝ) : ℝ := 
  real.sqrt (A / real.pi)

theorem radius_of_circle_with_area_64pi :
  circle_radius (64 * real.pi) = 8 :=
by sorry

end radius_of_circle_with_area_64pi_l360_360633


namespace distance_between_points_l360_360713

theorem distance_between_points {x1 y1 x2 y2 : ℝ} (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 8) (h4 : y2 = 9) : 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 6 * real.sqrt 5 :=
by
  have hx : x2 - x1 = 6 := by linarith [h1, h3]
  have hy : y2 - y1 = 12 := by linarith [h2, h4]
  rw [hx, hy]
  calc
    real.sqrt (6^2 + 12^2) = real.sqrt (36 + 144) : by congr; norm_num
    ... = real.sqrt 180 : by ring_nf
    ... = 6 * real.sqrt 5 : sorry

end distance_between_points_l360_360713


namespace prove_problem_statement_l360_360592

noncomputable def problem_statement : Prop :=
  let E := (0, 0)
  let F := (2, 4)
  let G := (6, 2)
  let H := (7, 0)
  let line_through_E x y := y = -2 * x + 14
  let intersection_x := 37 / 8
  let intersection_y := 19 / 4
  let intersection_point := (intersection_x, intersection_y)
  let u := 37
  let v := 8
  let w := 19
  let z := 4
  u + v + w + z = 68

theorem prove_problem_statement : problem_statement :=
  sorry

end prove_problem_statement_l360_360592


namespace find_common_ratio_l360_360909

noncomputable def geom_series_common_ratio (q : ℝ) : Prop :=
  ∃ (a1 : ℝ), a1 > 0 ∧ (a1 * q^2 = 18) ∧ (a1 * (1 + q + q^2) = 26)

theorem find_common_ratio (q : ℝ) :
  geom_series_common_ratio q → q = 3 :=
sorry

end find_common_ratio_l360_360909


namespace ellipse_eccentricity_l360_360000

theorem ellipse_eccentricity :
  (∃ x y : ℝ, x^2 / 9 + y^2 / 4 = 1) →
  ∃ e : ℝ, e = Real.sqrt 5 / 3 :=
by
  intro h
  use Real.sqrt 5 / 3
  sorry

end ellipse_eccentricity_l360_360000


namespace arithmetic_sequence_geometric_subsequence_l360_360054

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) = a n + 1)
  (h2 : (a 3)^2 = a 1 * a 7) :
  a 5 = 6 :=
sorry

end arithmetic_sequence_geometric_subsequence_l360_360054


namespace arctan_taylor_series_l360_360691

theorem arctan_taylor_series (x : ℝ) (h : |x| < 1) :
  has_series (λ n, (-1)^n * x^(2*n + 1) / (2*n + 1)) (arctan x) :=
sorry

end arctan_taylor_series_l360_360691


namespace relationship_between_b_and_g_l360_360368

-- Definitions based on the conditions
def n_th_boy_dances (n : ℕ) : ℕ := n + 5
def last_boy_dances_with_all : Prop := ∃ b g : ℕ, (n_th_boy_dances b = g)

-- The main theorem to prove the relationship between b and g
theorem relationship_between_b_and_g (b g : ℕ) (h : last_boy_dances_with_all) : b = g - 5 :=
by
  sorry

end relationship_between_b_and_g_l360_360368


namespace tan_pi_over_12_minus_tan_pi_over_6_l360_360614

theorem tan_pi_over_12_minus_tan_pi_over_6 :
  (Real.tan (Real.pi / 12) - Real.tan (Real.pi / 6)) = 7 - 4 * Real.sqrt 3 :=
  sorry

end tan_pi_over_12_minus_tan_pi_over_6_l360_360614


namespace tan_half_sum_pi_over_four_l360_360530

-- Define the problem conditions
variable (α : ℝ)
variable (h_cos : Real.cos α = -4 / 5)
variable (h_quad : α > π ∧ α < 3 * π / 2)

-- Define the theorem to prove
theorem tan_half_sum_pi_over_four (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_quad : α > π ∧ α < 3 * π / 2) :
  Real.tan (π / 4 + α / 2) = -1 / 2 := sorry

end tan_half_sum_pi_over_four_l360_360530


namespace dot_product_magnitude_l360_360190

variables {a b : ℝ^3}

-- Norm of vectors
def norm_a := ‖a‖ = 3
def norm_b := ‖b‖ = 4
def cross_product_norm := ‖a × b‖ = 6

theorem dot_product_magnitude (h1 : norm_a) (h2 : norm_b) (h3 : cross_product_norm) : (abs (a • b)) = 6 * real.sqrt 3 :=
sorry

end dot_product_magnitude_l360_360190


namespace minimum_moves_l360_360379

/- The initial grid configuration is such that the number at the intersection 
of the ith column and jth row is i + j. The target grid configuration is such that 
the number at the intersection of the ith column and jth row is 2n + 2 - i - j. 
We can perform swaps of two non-intersecting equal rectangles with one dimension 
equal to n. Prove that the minimum number of moves required to reach the target 
configuration is n - 1. -/

theorem minimum_moves (n : ℕ) : 
  (∃ f : ℕ → ℕ → ℕ, (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → f i j = i + j) ∧
  (∃ g : ℕ → ℕ → ℕ, (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → g i j = 2 * n + 2 - i - j) ∧
  (∀ swap_count : ℕ, (swap_count ≥ n - 1) → 
  ∀ f_final, (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → f_final i j = g i j) ∧ 
  (∃ swap_list : list (list (list ℕ)), ∃ swap_rectangles : list (list (list ℕ)), 
  (∀ s ∈ swap_list, s ∈ swap_rectangles) ∧ list.length swap_list = swap_count)))) :=
sorry

end minimum_moves_l360_360379


namespace trevor_pages_l360_360284

theorem trevor_pages (p1 p2 p3 : ℕ) (h1 : p1 = 72) (h2 : p2 = 72) (h3 : p3 = p1 + 4) : 
    p1 + p2 + p3 = 220 := 
by 
    sorry

end trevor_pages_l360_360284


namespace triangle_arithmetic_sequence_sides_relation_l360_360504

theorem triangle_arithmetic_sequence_sides_relation (A B C : ℝ)
  (h1 : A + B + C = π) -- Sum of interior angles of a triangle is π radians
  (h2 : 2 * B = A + C) -- Arithmetic sequence condition
  (a b c : ℝ) -- sides opposite to angles A, B, and C respectively
  (h3 : real_law_cosines A a b c)
  (h4 : real_law_cosines B a b c)
  (h5 : real_law_cosines C a b c):
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end triangle_arithmetic_sequence_sides_relation_l360_360504


namespace emma_finishes_first_l360_360384

noncomputable def david_lawn_area : ℝ := sorry
noncomputable def emma_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 3
noncomputable def fiona_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 4

noncomputable def david_mowing_rate : ℝ := sorry
noncomputable def fiona_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 6
noncomputable def emma_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 2

theorem emma_finishes_first (z w : ℝ) (hz : z > 0) (hw : w > 0) :
  (z / w) > (2 * z / (3 * w)) ∧ (3 * z / (2 * w)) > (2 * z / (3 * w)) :=
by
  sorry

end emma_finishes_first_l360_360384


namespace final_amoeba_is_blue_l360_360314

theorem final_amoeba_is_blue
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ)
  (merge : ∀ (a b : ℕ), a ≠ b → ∃ c, a + b - c = a ∧ a + b - c = b ∧ a + b - c = c)
  (initial_counts : n1 = 26 ∧ n2 = 31 ∧ n3 = 16)
  (final_count : ∃ a, a = 1) :
  ∃ color, color = "blue" := sorry

end final_amoeba_is_blue_l360_360314


namespace sum_of_roots_of_quadratic_eq_l360_360728

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l360_360728


namespace sum_of_roots_of_quadratic_eq_l360_360729

theorem sum_of_roots_of_quadratic_eq : 
  ∀ (a b c : ℝ), (x^2 - 6 * x + 8 = 0) → (a = 1 ∧ b = -6 ∧ c = 8) → -b / a = 6 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_eq_l360_360729


namespace probability_of_multiple_of_225_l360_360535

-- Define the set of positive multiples of 3 under 100
def multiples_of_3_under_100 : finset ℕ := finset.filter (λ x, x % 3 = 0) (finset.range 100)

-- Define the set of prime numbers less than 100
def primes_under_100 : finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}

-- Define the condition for a product to be a multiple of 225
def is_multiple_of_225 (n : ℕ) : Prop := 225 ∣ n

-- Probability calculation
noncomputable def probability_product_is_multiple_of_225 : ℚ := do
    let favorable_events := finset.filter (λ p, p = 5) primes_under_100
    (favorable_events.card : ℚ) / (primes_under_100.card : ℚ)

theorem probability_of_multiple_of_225 : probability_product_is_multiple_of_225 = 1 / 25 :=
by
   sorry

end probability_of_multiple_of_225_l360_360535


namespace periodic_derivative_cos_l360_360026

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.cos x
| (n+1) := λ x, (f n)' x

theorem periodic_derivative_cos :
  f 2011 = λ x, Real.sin x :=
by 
  sorry

end periodic_derivative_cos_l360_360026


namespace new_area_after_modification_l360_360248

theorem new_area_after_modification (s : ℝ) (h : s^2 = 625) :
  let new_side1 := 0.80 * s,
      new_side2 := 1.20 * s
  in new_side1 * new_side2 = 600 := 
by 
  sorry

end new_area_after_modification_l360_360248


namespace sequence_bound_l360_360590

theorem sequence_bound (a : ℕ → ℝ) (n : ℕ) 
  (h₁ : a 0 = 0) 
  (h₂ : a (n + 1) = 0)
  (h₃ : ∀ k, 1 ≤ k → k ≤ n → a (k - 1) - 2 * (a k) + (a (k + 1)) ≤ 1) 
  : ∀ k, 0 ≤ k → k ≤ n + 1 → a k ≤ (k * (n + 1 - k)) / 2 :=
sorry

end sequence_bound_l360_360590


namespace sum_of_roots_l360_360735

theorem sum_of_roots (a b c: ℝ) (h: a ≠ 0) (h_eq : a = 1) (h_eq2 : b = -6) (h_eq3 : c = 8):
    let Δ := b ^ 2 - 4 * a * c in
    let root1 := (-b + real.sqrt Δ) / (2 * a) in
    let root2 := (-b - real.sqrt Δ) / (2 * a) in
    root1 + root2 = 6 :=
by
  sorry

end sum_of_roots_l360_360735


namespace smallest_positive_period_sin_2x_value_l360_360478

open Real

-- Definitions
def a_vector (x : ℝ) : ℝ × ℝ := (cos x - sin x, 2 * sin x)
def b_vector (x : ℝ) : ℝ × ℝ := (cos x + sin x, sqrt 3 * cos x)
def f (x : ℝ) : ℝ := (a_vector x).fst * (b_vector x).fst + (a_vector x).snd * (b_vector x).snd

-- Proof statements
theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π ∧
  (∀ k : ℤ, x ∈ Icc (k * π - π / 3) (k * π + π / 6) → (differentiable ℝ (λ x, f x)) ∧ monotone_on (λ x, f x) (Icc (k * π - π / 3) (k * π + π / 6))) :=
sorry

theorem sin_2x_value (x : ℝ) (hx : x ∈ Icc (-π / 4) (π / 6)) (hfx : f x = 10 / 13) :
  sin (2 * x) = (5 * sqrt 3 - 12) / 26 :=
sorry

end smallest_positive_period_sin_2x_value_l360_360478


namespace difference_of_roots_l360_360454

theorem difference_of_roots (a b c : ℝ) (h_eq : a = 1 ∧ b = -9 ∧ c = 14) :
  let roots := (1 / a) * (9) - 2 * ( ( (a * x ^ 2) + (b * x)  + c )  / 4 ) in
  let r_1_plus_r_2 : ℝ := -b / a in
  let r_1_times_r_2 : ℝ := c / a in
  let d : ℝ := (r_1_plus_r_2 * r_1_plus_r_2) - 4 * r_1_times_r_2 in
  sqrt d = 5 :=
sorry

end difference_of_roots_l360_360454


namespace standard_equation_of_ellipse_max_value_of_AB_l360_360036
noncomputable section

-- Define the conditions for the ellipse and the line
def eccentricity : ℝ := sqrt 3 / 2
def major_axis_length : ℝ := 4
def slope_of_line : ℝ := 1

-- Define the statements for the requested proofs
theorem standard_equation_of_ellipse :
  ∀ (e a : ℝ), e = sqrt 3 / 2 → 2 * a = 4 → (a > 0) → {
    let b := sqrt (a^2 - (e * a)^2) in  -- b = sqrt(a² - c²) setting c = e * a
    ∀ (ell_eq : (x y : ℝ) → Prop), 
      ell_eq = (λ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) → 
      ∀ x y, ell_eq x y → (x^2 / 4 + y^2 = 1)} :=
begin
  sorry
end

theorem max_value_of_AB :
  ∀ (b : ℝ), -sqrt 5 < b → b < sqrt 5 →
    let distance_AB := (λ b : ℝ, 4 * sqrt 2 / 5 * sqrt (5 - b^2)) in
    ∀ max_AB : ℝ, 
      max_AB = distance_AB 0 →
      max_AB = 4 * sqrt 10 / 5 :=
begin
  sorry
end

end standard_equation_of_ellipse_max_value_of_AB_l360_360036


namespace price_reduction_percentage_price_increase_amount_l360_360330

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end price_reduction_percentage_price_increase_amount_l360_360330


namespace z_imaginary_iff_m_ne_1_and_2_z_purely_imaginary_iff_m_eq_neg_half_z_on_angle_bisector_iff_m_eq_0_or_2_l360_360483

variable (m : ℝ) -- m is a real number

def z (m : ℝ) : ℂ := (2 + complex.i) * m^2 - 3 * m * (1 + complex.i) - 2 * (1 - complex.i)

theorem z_imaginary_iff_m_ne_1_and_2 (m : ℝ) :
  ¬(z m).re = 0 ∧ (z m).im ≠ 0 ↔ m ≠ 1 ∧ m ≠ 2 := 
sorry

theorem z_purely_imaginary_iff_m_eq_neg_half (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -1/2 := 
sorry

theorem z_on_angle_bisector_iff_m_eq_0_or_2 (m : ℝ) :
  2 * m^2 - 3 * m - 2 = -(m^2 - 3 * m + 2) ↔ m = 0 ∨ m = 2 := 
sorry

end z_imaginary_iff_m_ne_1_and_2_z_purely_imaginary_iff_m_eq_neg_half_z_on_angle_bisector_iff_m_eq_0_or_2_l360_360483


namespace find_smallest_B_l360_360274

theorem find_smallest_B (B : ℕ)  
  (h1 : ∀ m : ℕ, m = B → (∑ i in (list.replicate B [2,0,0,8]).join, i) % 3 = 0)
  (h2 : (B ≠ 0 ∧ (list.last (list.replicate B [2,0,0,8]).join (by simp)) % 5 = 0)) :
  B = 3 :=
by
  sorry

end find_smallest_B_l360_360274


namespace average_speed_of_car_l360_360328

noncomputable def averageSpeed : ℚ := 
  let speed1 := 45     -- kph
  let distance1 := 15  -- km
  let speed2 := 55     -- kph
  let distance2 := 30  -- km
  let speed3 := 65     -- kph
  let time3 := 35 / 60 -- hours
  let speed4 := 52     -- kph
  let time4 := 20 / 60 -- hours
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  let totalDistance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let totalTime := time1 + time2 + time3 + time4
  totalDistance / totalTime

theorem average_speed_of_car :
  abs (averageSpeed - 55.85) < 0.01 := 
  sorry

end average_speed_of_car_l360_360328


namespace probability_106_heavier_l360_360220

theorem probability_106_heavier (masses : list ℕ) (h : masses = [101, 102, 103, 104, 105, 106]) : 
  (probability (106 in heavier_pan masses) = 80/100) :=
by
  sorry

end probability_106_heavier_l360_360220


namespace part1_part2_l360_360517

noncomputable def a (n : ℕ) : ℤ :=
  15 * n + 2 + (15 * n - 32) * 16^(n-1)

theorem part1 (n : ℕ) : 15^3 ∣ (a n) := by
  sorry

-- Correct answer for part (2) bundled in a formal statement:
theorem part2 (n k : ℕ) : 1991 ∣ (a n) ∧ 1991 ∣ (a (n + 1)) ∧
    1991 ∣ (a (n + 2)) ↔ n = 89595 * k := by
  sorry

end part1_part2_l360_360517


namespace volume_of_parallelepiped_l360_360271

theorem volume_of_parallelepiped 
  (m n Q : ℝ) 
  (ratio_positive : 0 < m ∧ 0 < n)
  (Q_positive : 0 < Q)
  (h_square_area : ∃ a b : ℝ, a / b = m / n ∧ (a^2 + b^2) = Q) :
  ∃ (V : ℝ), V = (m * n * Q * Real.sqrt Q) / (m^2 + n^2) :=
sorry

end volume_of_parallelepiped_l360_360271


namespace limit_sum_of_perimeters_l360_360352

variable (s : ℝ) -- The side length of the initial square

-- Definitions of the infinite series which represents the sum of the perimeters
def perimeter_series : ℕ → ℝ
| 0       => 4 * s
| (n + 1) => (perimeter_series n) / 2

-- Sum of the infinite geometric series
def sum_infinite_series (a : ℝ) (r : ℝ) : ℝ :=
a / (1 - r)

-- The mathematically equivalent statement in Lean 4
theorem limit_sum_of_perimeters (s : ℝ) (h : s ≥ 0) :
  (sum_infinite_series (4 * s) (1 / 2)) = 8 * s :=
by
  sorry

end limit_sum_of_perimeters_l360_360352


namespace total_days_2004_to_2008_l360_360088

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (years : List ℕ) : ℕ :=
  years.foldr (λ y acc, days_in_year y + acc) 0

theorem total_days_2004_to_2008 :
  total_days [2004, 2005, 2006, 2007, 2008] = 1827 :=
by
  sorry

end total_days_2004_to_2008_l360_360088


namespace chameleons_all_green_l360_360222

def initial_counts : (ℕ × ℕ × ℕ) := (7, 10, 17)
def transformation_rules := ∀ (j r v : ℕ), -- transformation rules go here

theorem chameleons_all_green (j r v : ℕ) (initial_counts : j = 7 ∧ r = 10 ∧ v = 17) 
  (rules : transformation_rules) :
  ∃ k : ℕ, (j = 0 ∧ r = 0 ∧ v = k) ∧ k = 34 :=
by
  sorry

end chameleons_all_green_l360_360222


namespace exam_question_bound_l360_360316

theorem exam_question_bound (n_students : ℕ) (k_questions : ℕ) (n_answers : ℕ) 
    (H_students : n_students = 25) (H_answers : n_answers = 5) 
    (H_condition : ∀ (i j : ℕ) (H1 : i < n_students) (H2 : j < n_students) (H_neq : i ≠ j), 
      ∀ q : ℕ, q < k_questions → ∀ ai aj : ℕ, ai < n_answers → aj < n_answers → 
      ((ai = aj) → (i = j ∨ q' > 1))) : 
    k_questions ≤ 6 := 
sorry

end exam_question_bound_l360_360316


namespace negation_proposition_l360_360024

theorem negation_proposition (a b c : ℝ) : 
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) := 
by
  -- proof goes here
  sorry

end negation_proposition_l360_360024


namespace line_equation_with_slope_through_point_l360_360256

theorem line_equation_with_slope_through_point :
  ∀ (x y : ℝ), (∃ y1 x1 : ℝ, (y1 = 1) ∧ (x1 = 3) ∧ (y - y1 = 2 * (x - x1))) → (2 * x - y - 5 = 0) :=
by
  intros x y h
  cases h with y1 h
  cases h with x1 h1
  cases h1 with h_y1 h1
  cases h1 with h_x1 h2
  rw [h_y1, h_x1] at h2
  sorry

end line_equation_with_slope_through_point_l360_360256


namespace mark_repayment_l360_360594

noncomputable def totalDebt (days : ℕ) : ℝ :=
  if days < 3 then
    20 + (20 * 0.10 * days)
  else
    35 + (20 * 0.10 * 3) + (35 * 0.10 * (days - 3))

theorem mark_repayment :
  ∃ (x : ℕ), totalDebt x ≥ 70 ∧ x = 12 :=
by
  -- Use this theorem statement to prove the corresponding lean proof
  sorry

end mark_repayment_l360_360594


namespace solve_equation_l360_360939

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l360_360939


namespace volume_of_new_cube_is_2744_l360_360755

-- Define the volume function for a cube given side length
def volume_of_cube (side : ℝ) : ℝ := side ^ 3

-- Given the original cube with a specific volume
def original_volume : ℝ := 343

-- Find the side length of the original cube by taking the cube root of the volume
def original_side_length := (original_volume : ℝ)^(1/3)

-- The side length of the new cube is twice the side length of the original cube
def new_side_length := 2 * original_side_length

-- The volume of the new cube should be calculated
def new_volume := volume_of_cube new_side_length

-- Theorem stating that the new volume is 2744 cubic feet
theorem volume_of_new_cube_is_2744 : new_volume = 2744 := sorry

end volume_of_new_cube_is_2744_l360_360755


namespace find_ac_bc_val_l360_360497

variable (a b c d : ℚ)
variable (h_neq : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h1 : (a + c) * (a + d) = 1)
variable (h2 : (b + c) * (b + d) = 1)

theorem find_ac_bc_val : (a + c) * (b + c) = -1 := 
by 
  sorry

end find_ac_bc_val_l360_360497


namespace evaluate_floor_of_negative_seven_halves_l360_360831

def floor_of_negative_seven_halves : ℤ :=
  Int.floor (-7 / 2)

theorem evaluate_floor_of_negative_seven_halves :
  floor_of_negative_seven_halves = -4 :=
by
  sorry

end evaluate_floor_of_negative_seven_halves_l360_360831


namespace competition_participants_solved_all_three_l360_360994

theorem competition_participants_solved_all_three
  (p1 p2 p3 : ℕ → Prop)
  (total_participants : ℕ)
  (h1 : ∃ n, n = 85 * total_participants / 100 ∧ ∀ k, k < n → p1 k)
  (h2 : ∃ n, n = 80 * total_participants / 100 ∧ ∀ k, k < n → p2 k)
  (h3 : ∃ n, n = 75 * total_participants / 100 ∧ ∀ k, k < n → p3 k) :
  ∃ n, n ≥ 40 * total_participants / 100 ∧ ∀ k, k < n → p1 k ∧ p2 k ∧ p3 k :=
by
  sorry

end competition_participants_solved_all_three_l360_360994


namespace largest_power_of_2_dividing_32_factorial_ones_digit_l360_360849

def power_of_two_ones_digit (n: ℕ) : ℕ :=
  let digits_cycle := [2, 4, 8, 6]
  digits_cycle[(n % 4) - 1]

theorem largest_power_of_2_dividing_32_factorial_ones_digit :
  power_of_two_ones_digit 31 = 8 := by
  sorry

end largest_power_of_2_dividing_32_factorial_ones_digit_l360_360849


namespace lim_an_add_c_lim_c_an_lim_an_add_bn_lim_an_mul_bn_lim_inv_an_l360_360580

variable (a : ℝ) (b : ℝ) (an : ℕ → ℝ) (bn : ℕ → ℝ) (c : ℝ)

-- Conditions
axiom lim_an : filter.tendsto an filter.at_top (𝓝 a)
axiom lim_bn : filter.tendsto bn filter.at_top (𝓝 b)

-- Part (a)
theorem lim_an_add_c : filter.tendsto (λ n, an n + c) filter.at_top (𝓝 (a + c)) :=
sorry

theorem lim_c_an : filter.tendsto (λ n, c * an n) filter.at_top (𝓝 (c * a)) :=
sorry

-- Part (b)
theorem lim_an_add_bn : filter.tendsto (λ n, an n + bn n) filter.at_top (𝓝 (a + b)) :=
sorry

-- Part (c)
theorem lim_an_mul_bn : filter.tendsto (λ n, an n * bn n) filter.at_top (𝓝 (a * b)) :=
sorry

-- Part (d)
axiom an_ne_zero : ∀ n, an n ≠ 0
axiom a_ne_zero : a ≠ 0

theorem lim_inv_an : filter.tendsto (λ n, 1 / an n) filter.at_top (𝓝 (1 / a)) :=
sorry

end lim_an_add_c_lim_c_an_lim_an_add_bn_lim_an_mul_bn_lim_inv_an_l360_360580


namespace midpoint_equivalence_l360_360574

-- Define the geometric setup for the problem
variables {A B C D E U V X Y : Type}
variable (Γ : Type)
variable [IsConvexQuadrilateral A B C D Γ]
variable (E : IntersectionOfDiagonals A B C D)

-- Define line passing through E intersecting given segments and circle Γ
variable (Δ : LineThroughPoint E (Segment AB) (Segment CD) (CircleΓ Γ))
variable [IntersectsSegment Δ (Segment AB) U]
variable [IntersectsSegment Δ (Segment CD) V]
variable [IntersectsCircle Δ (CircleΓ Γ) X Y]

-- Define midpoints
variable (MidpointXY : Midpoint E X Y)
variable (MidpointUV : Midpoint E U V)

-- The main statement to be proven
theorem midpoint_equivalence :
  Midpoint E X Y ↔ Midpoint E U V :=
sorry

end midpoint_equivalence_l360_360574


namespace exists_k_undecided_l360_360588

def tournament (n : ℕ) : Type :=
  { T : Fin n → Fin n → Prop // ∀ i j, T i j = ¬T j i }

def k_undecided (n k : ℕ) (T : tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k → ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → T.1 p a

theorem exists_k_undecided (k : ℕ) (hk : 0 < k) : ∃ (n : ℕ), n > k ∧ ∃ (T : tournament n), k_undecided n k T :=
by
  sorry

end exists_k_undecided_l360_360588


namespace dot_product_magnitude_l360_360191

variables {a b : ℝ^3}

-- Norm of vectors
def norm_a := ‖a‖ = 3
def norm_b := ‖b‖ = 4
def cross_product_norm := ‖a × b‖ = 6

theorem dot_product_magnitude (h1 : norm_a) (h2 : norm_b) (h3 : cross_product_norm) : (abs (a • b)) = 6 * real.sqrt 3 :=
sorry

end dot_product_magnitude_l360_360191


namespace clock_angle_at_seven_l360_360695

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l360_360695


namespace arithmetic_mean_of_integers_from_neg6_to_6_l360_360293

theorem arithmetic_mean_of_integers_from_neg6_to_6 :
  ∃ (mean : ℚ), mean = (finset.range (12 + 1)).sum (λ x, (x - 6)) / (12 + 1) :=
by 
  let n := 12 + 1
  let integers := (finset.range n).map (λ x, x - 6)
  let sum := integers.sum
  have hsum : sum = 0, from sorry
  use (sum : ℚ) / n
  simp [hsum]
  exact sorry

end arithmetic_mean_of_integers_from_neg6_to_6_l360_360293


namespace tourist_group_people_count_l360_360791

def large_room_people := 3
def small_room_people := 2
def small_rooms_rented := 1
def people_in_small_room := small_rooms_rented * small_room_people

theorem tourist_group_people_count : 
  ∀ x : ℕ, x ≥ 1 ∧ (x + small_rooms_rented) = (people_in_small_room + x * large_room_people) → 
  (people_in_small_room + x * large_room_people) = 5 := 
  by
  sorry

end tourist_group_people_count_l360_360791


namespace count_valid_x_l360_360083

-- Define the conditions that x must satisfy
def valid_x (x : ℕ) : Prop :=
  (334 ≤ x ∧ x ≤ 499)

-- Prove that the number of valid x is 166
theorem count_valid_x : (finset.filter valid_x (finset.range 500)).card = 166 := 
sorry

end count_valid_x_l360_360083


namespace dot_product_magnitude_l360_360175

variables {a b : EuclideanSpace 3 ℝ}

/- Given conditions -/
def norm_a : ℝ := ‖a‖ = 3
def norm_b : ℝ := ‖b‖ = 4
def norm_cross : ℝ := ‖a × b‖ = 6

/- Desired property to prove -/
theorem dot_product_magnitude :
  norm_a →
  norm_b →
  norm_cross →
  |(a ⋅ b)| = 6 * real.sqrt 3 :=
by
  sorry

end dot_product_magnitude_l360_360175


namespace repeating_decimals_sum_l360_360425

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l360_360425


namespace leak_empties_cistern_in_20_hours_l360_360754

variable (R : ℝ) (L : ℝ) (effective_rate : ℝ)

-- Conditions
def fill_rate : Prop := R = 1 / 4
def effective_fill_rate : Prop := effective_rate = 1 / 5
def leak_rate : Prop := R - L = effective_rate

-- Goal
def time_to_empty : ℝ := 1 / L

theorem leak_empties_cistern_in_20_hours (h1 : fill_rate R) (h2 : effective_fill_rate effective_rate) (h3 : leak_rate R L effective_rate) :
  time_to_empty L = 20 :=
by
  sorry

end leak_empties_cistern_in_20_hours_l360_360754


namespace students_playing_football_l360_360988

theorem students_playing_football :
  (∃ (F L B N: ℕ), 
    L = 20 ∧ 
    B = 17 ∧ 
    N = 9 ∧ 
    38 = F + (L - B) + (B - B) + N ∧ 
    38 = 20 - 17 + F + 9) → 
  ∃ F, F = 26 :=
by
  intro h
  rcases h with ⟨F, L, B, N, hL, hB, hN, hTotal, hFinal⟩
  have F_correct : F = 26
  { -- introduce assumptions and use them in this block
  -- replace this with the correct calculation steps if needed
    sorry 
  }
  use F
  exact F_correct

end students_playing_football_l360_360988


namespace dot_product_magnitude_l360_360183

variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜 → 𝕜} 

-- Define the conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_a_cross_b : ℝ := 6

-- Theorem statement
theorem dot_product_magnitude :
  (∥a∥ = norm_a) → (∥b∥ = norm_b) → (∥a × b∥ = norm_a_cross_b) → abs ((a.toReal).dot (b.toReal)) = 6 * sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end dot_product_magnitude_l360_360183


namespace abs_dot_product_l360_360169

variables (a b : ℝ^3)

-- Conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def norm_cross_ab : ℝ := 6

-- Theorem statement
theorem abs_dot_product (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (hcross : ‖a × b‖ = norm_cross_ab) :
  |(a ⬝ b)| = 6 * sqrt 3 :=
by
  sorry

end abs_dot_product_l360_360169


namespace average_speed_of_train_l360_360756

theorem average_speed_of_train (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 125) (h2 : d2 = 270) (h3 : t1 = 2.5) (h4 : t2 = 3) :
  (d1 + d2) / (t1 + t2) = 71.82 :=
by
  sorry

end average_speed_of_train_l360_360756
