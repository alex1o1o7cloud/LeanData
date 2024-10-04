import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Stats
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.GCD
import Mathlib.Data.Nat.Prime
import Mathlib.Data.ProbTheory.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Geometry.Euclidean.Cyclic
import Mathlib.LinearAlgebra.Basic
import Mathlib.Prime
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Real

namespace abs_condition_l600_600206

theorem abs_condition (x y : ℝ) (h : |x - sqrt y| = x + sqrt y) : x + y = 0 :=
sorry

end abs_condition_l600_600206


namespace completing_the_square_l600_600795

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l600_600795


namespace smallest_positive_period_f_l600_600629

def f (x : ℝ) : ℝ :=
  sqrt 3 * Real.sin (2 * x - Real.pi / 6) + 2 * (Real.sin (x - Real.pi / 12)) ^ 2

theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
by
  use Real.pi
  sorry

end smallest_positive_period_f_l600_600629


namespace find_2023rd_letter_l600_600788

/-- Define the repeating sequence and its length -/
def sequence : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
def sequence_length : Nat := sequence.length

/-- The question asking for the 2023rd letter in the sequence -/
def positional_indx : Nat := 2023 % sequence_length

/-- The target proof statement -/
theorem find_2023rd_letter : sequence.get! positional_indx = 'G' := 
sorry

end find_2023rd_letter_l600_600788


namespace eval_six_times_f_l600_600279

def f (x : Int) : Int :=
  if x % 2 == 0 then
    x / 2
  else
    5 * x + 1

theorem eval_six_times_f : f (f (f (f (f (f 7))))) = 116 := 
by
  -- Skipping proof body (since it's not required)
  sorry

end eval_six_times_f_l600_600279


namespace crystal_final_segment_distance_l600_600064

theorem crystal_final_segment_distance :
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2 -- as nx, ny
  let southwest_component := southwest_distance / Real.sqrt 2 -- as sx, sy
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  Real.sqrt (net_north^2 + net_west^2) = 2 * Real.sqrt 3 :=
by
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2
  let southwest_component := southwest_distance / Real.sqrt 2
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  exact sorry

end crystal_final_segment_distance_l600_600064


namespace gcd_47_power_l600_600056

theorem gcd_47_power : 
  let a := 47^5 + 1,
      b := 47^5 + 47^3 + 1
  in  gcd a b = 1 := 
by {
  sorry
}

end gcd_47_power_l600_600056


namespace quadratic_completion_l600_600563

theorem quadratic_completion :
  (∀ x : ℝ, (∃ a h k : ℝ, (x ^ 2 - 2 * x - 1 = a * (x - h) ^ 2 + k) ∧ (a = 1) ∧ (h = 1) ∧ (k = -2))) :=
sorry

end quadratic_completion_l600_600563


namespace distance_from_center_to_plane_l600_600886

noncomputable theory
open_locale classical

variables (O : ℝ^3) (r : ℝ)
variables (a b c : ℝ)
variables (d : ℝ)

-- Conditions
-- Sphere with center O and radius 8
def sphere_center := O
def sphere_radius := 8

-- Triangle with sides 13, 14, 15 tangent to the sphere
def triangle_sides := [13, 14, 15]

-- Question: Determine the distance between O and the plane determined by the triangle
def distance_to_plane := 4 * real.sqrt 3 

-- Theorem Statement
theorem distance_from_center_to_plane 
  (h_center : sphere_center = O)
  (h_radius : sphere_radius = 8)
  (h_sides : triangle_sides = [13, 14, 15]) : 
  d = distance_to_plane :=
begin
  sorry
end

end distance_from_center_to_plane_l600_600886


namespace Mongolian_Mathematical_Olympiad_54th_l600_600703

theorem Mongolian_Mathematical_Olympiad_54th {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^4 + b^4 + c^4 + (a^2 / (b + c)^2) + (b^2 / (c + a)^2) + (c^2 / (a + b)^2) ≥ a * b + b * c + c * a :=
sorry

end Mongolian_Mathematical_Olympiad_54th_l600_600703


namespace find_original_price_l600_600695

theorem find_original_price (sale_price : ℕ) (discount : ℕ) (original_price : ℕ) 
  (h1 : sale_price = 60) 
  (h2 : discount = 40) 
  (h3 : original_price = sale_price / ((100 - discount) / 100)) : original_price = 100 :=
by
  sorry

end find_original_price_l600_600695


namespace bens_car_payment_l600_600053

variable (G : ℝ) (T : ℝ) (C : ℝ)

theorem bens_car_payment
  (hG : G = 3000)
  (hT : T = 1 / 3)
  (hC : C = 0.20) :
  let after_tax_income := G - T * G in
  let car_payment := C * after_tax_income in
  car_payment = 400 := sorry

end bens_car_payment_l600_600053


namespace determine_a_l600_600998

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x ^ 2 + a else 2 ^ x

theorem determine_a (a : ℝ) (h1 : a > -1) (h2 : f a (f a (-1)) = 4) : a = 1 :=
sorry

end determine_a_l600_600998


namespace solve_inequality_l600_600311

theorem solve_inequality (a x : ℝ) :
  (a = 1/2 → (x ≠ 1/2 → (x - a) * (x + a - 1) > 0)) ∧
  (a < 1/2 → ((x > (1 - a) ∨ x < a) → (x - a) * (x + a - 1) > 0)) ∧
  (a > 1/2 → ((x > a ∨ x < (1 - a)) → (x - a) * (x + a - 1) > 0)) :=
by
  sorry

end solve_inequality_l600_600311


namespace a_2005_l600_600961

noncomputable def a : ℕ → ℤ := sorry 

axiom a3 : a 3 = 5
axiom a5 : a 5 = 8
axiom exists_n : ∃ (n : ℕ), n > 0 ∧ a n + a (n + 1) + a (n + 2) = 7

theorem a_2005 : a 2005 = -6 := by {
  sorry
}

end a_2005_l600_600961


namespace max_dist_AC_l600_600182

open Real EuclideanGeometry

variables (P A B C : ℝ × ℝ)
  (hPA : dist P A = 1)
  (hPB : dist P B = 1)
  (hPA_PB : dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = - 1 / 2)
  (hBC : dist B C = 1)

theorem max_dist_AC : ∃ C : ℝ × ℝ, dist A C ≤ dist A B + dist B C ∧ dist A C = sqrt 3 + 1 :=
by
  sorry

end max_dist_AC_l600_600182


namespace overall_percent_decrease_l600_600256

def original_price : ℝ := 100
def first_discount : ℝ := 0.2
def second_discount : ℝ := 0.15
def loyalty_discount : ℝ := 0.1
def sales_tax : ℝ := 0.05

def price_after_first_discount := original_price * (1 - first_discount)
def price_after_second_discount := price_after_first_discount * (1 - second_discount)
def price_after_loyalty_discount := price_after_second_discount * (1 - loyalty_discount)
def final_price_after_tax := price_after_loyalty_discount * (1 + sales_tax)

def percent_decrease := ((original_price - final_price_after_tax) / original_price) * 100

theorem overall_percent_decrease :
  percent_decrease = 35.74 :=
by
  sorry

end overall_percent_decrease_l600_600256


namespace minerals_found_today_l600_600254

noncomputable def yesterday_gemstones := 21
noncomputable def today_minerals := 48
noncomputable def today_gemstones := 21

theorem minerals_found_today :
  (today_minerals - (2 * yesterday_gemstones) = 6) :=
by
  sorry

end minerals_found_today_l600_600254


namespace generating_function_equivalence_l600_600467

noncomputable def generating_function_repetition_allowed (x : ℂ) (m : ℕ) : ℂ :=
  ∏ i in (finset.range m).map (finset.nat_emb (λ i, i + 1)), 1 / (1 - x ^ i)

noncomputable def generating_function_m_at_least_once (x : ℂ) (m : ℕ) : ℂ :=
  ∏ i in (finset.range (m - 1)).map (finset.nat_emb (λ i, i + 1)), 1 / (1 - x ^ i) * (x ^ m / (1 - x ^ m))

theorem generating_function_equivalence (x : ℂ) (m : ℕ) (hm : m > 0) :
  generating_function_m_at_least_once x m = generating_function_repetition_allowed x m - generating_function_repetition_allowed x (m - 1) :=
begin
  sorry,
end

end generating_function_equivalence_l600_600467


namespace radius_of_circle_l600_600743

theorem radius_of_circle (r : ℝ) : 3 * 2 * Real.pi * r = Real.pi * r^2 → r = 6 :=
by {
  intro h,
  have h1 : 6 * Real.pi * r = Real.pi * r^2 := by rw [←mul_assoc, ←h],
  have h2 : 6 * r = r^2 := by rw [←mul_div_cancel_left 'Real.pi, h1],
  have h3 : r^2 - 6 * r = 0 := by ring,
  have h4 : r * (r - 6) = 0 := by rw h3,
  cases eq_zero_or_eq_zero_of_mul_eq_zero h4 with h5 h6,
  { exact h5, },
  { exact h6, }
} sorry

end radius_of_circle_l600_600743


namespace value_of_a_star_b_l600_600349

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l600_600349


namespace circle_diameter_length_l600_600428

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l600_600428


namespace pump_capacity_l600_600391

-- Define parameters and assumptions
def tank_volume : ℝ := 1000
def fill_percentage : ℝ := 0.85
def fill_time : ℝ := 1
def num_pumps : ℝ := 8
def pump_efficiency : ℝ := 0.75
def required_fill_volume : ℝ := fill_percentage * tank_volume

-- Assumed total effective capacity must meet the required fill volume
theorem pump_capacity (C : ℝ) : 
  (num_pumps * pump_efficiency * C = required_fill_volume) → 
  C = 850.0 / 6.0 :=
by
  sorry

end pump_capacity_l600_600391


namespace speed_in_still_water_l600_600009

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_up : upstream_speed = 26) (h_down : downstream_speed = 30) :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end speed_in_still_water_l600_600009


namespace triangle_parallel_lines_sum_l600_600601

theorem triangle_parallel_lines_sum {A B C P I F E H D G : Type*} 
  [InTriangle P A B C] [ParallelLines IF BC] [ParallelLines EH AC] [ParallelLines DG AB] :
  (IF_length / BC_length) + (EH_length / AC_length) + (DG_length / AB_length) = 2 :=
by sorry

end triangle_parallel_lines_sum_l600_600601


namespace xiao_ming_second_half_time_l600_600836

theorem xiao_ming_second_half_time :
  ∀ (total_distance : ℕ) (speed1 : ℕ) (speed2 : ℕ), 
    total_distance = 360 →
    speed1 = 5 →
    speed2 = 4 →
    let t_total := total_distance / (speed1 + speed2) * 2
    let half_distance := total_distance / 2
    let t2 := half_distance / speed2
    half_distance / speed2 + (half_distance / speed1) = 44 :=
sorry

end xiao_ming_second_half_time_l600_600836


namespace compute_expression_l600_600912

noncomputable def a : ℝ := 125^(1/3)
noncomputable def b : ℝ := (-2/3)^0
noncomputable def c : ℝ := Real.log 8 / Real.log 2

theorem compute_expression : a - b - c = 1 := by
  sorry

end compute_expression_l600_600912


namespace jumping_bug_ways_l600_600438

-- Define the problem with given conditions and required answer
theorem jumping_bug_ways :
  let starting_position := 0
  let ending_position := 3
  let jumps := 5
  let jump_options := [1, -1]
  (∃ (jump_seq : Fin jumps → ℤ), (∀ i, jump_seq i ∈ jump_options ∧ (List.sum (List.ofFn jump_seq) = ending_position)) ∧
  (List.count (-1) (List.ofFn jump_seq) = 1)) →
  (∃ n : ℕ, n = 5) :=
by
  sorry  -- Proof to be completed

end jumping_bug_ways_l600_600438


namespace xy_value_l600_600199

theorem xy_value (x y : ℝ) 
  (h1 : 8^x / 4^(x + y) = 16) 
  (h2 : 16^(x + y) / 4^(7 * y) = 1024) : 
  x * y = 30 := 
by 
  sorry

end xy_value_l600_600199


namespace completing_the_square_l600_600799

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l600_600799


namespace general_term_formula_l600_600148

-- Define the problem parameters
variables (a : ℤ)

-- Definitions based on the conditions
def first_term : ℤ := a - 1
def second_term : ℤ := a + 1
def third_term : ℤ := 2 * a + 3

-- Define the theorem to prove the general term formula
theorem general_term_formula :
  2 * (first_term a + 1) = first_term a + third_term a → a = 0 →
  ∀ n : ℕ, a_n = 2 * n - 3 := 
by
  intro h1 h2
  sorry

end general_term_formula_l600_600148


namespace probability_is_1_div_28_l600_600374

noncomputable def probability_valid_combinations : ℚ :=
  let total_combinations := Nat.choose 8 3
  let valid_combinations := 2
  valid_combinations / total_combinations

theorem probability_is_1_div_28 :
  probability_valid_combinations = 1 / 28 := by
  sorry

end probability_is_1_div_28_l600_600374


namespace ratio_of_ages_in_two_years_l600_600010

theorem ratio_of_ages_in_two_years (S M : ℕ) (h1: M = S + 28) (h2: M + 2 = (S + 2) * 2) (h3: S = 26) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l600_600010


namespace extreme_point_f_l600_600656

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 1)

theorem extreme_point_f :
  ∃ x : ℝ, (∀ y : ℝ, y ≠ 0 → (Real.exp y * y < 0 ↔ y < x)) ∧ x = 0 :=
by
  sorry

end extreme_point_f_l600_600656


namespace part_one_part_two_l600_600593

noncomputable def f (x a : ℝ) : ℝ :=
  |3 * x + 1 / a| + 3 * |x - a|

theorem part_one (x : ℝ) :
  (-∞, -1] ∪ [5/3, ∞) = { x | f x 1 ≥ 8 } :=
sorry

theorem part_two (a x : ℝ) (ha : a > 0) :
  f x a ≥ 2 * sqrt 3 :=
sorry

end part_one_part_two_l600_600593


namespace tiling_2xn_with_dominoes_l600_600229

theorem tiling_2xn_with_dominoes (n : ℕ) : 
  let u : ℕ → ℕ := λ n, if n = 1 then 1 else if n = 2 then 2 else u (n-1) + u (n-2)
  in u n = Nat.fib (n+1) :=
by
  sorry

end tiling_2xn_with_dominoes_l600_600229


namespace solve_complex_equation_l600_600722

theorem solve_complex_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ((a + b * complex.I) * (a + 2 * complex.I) * (a + 4 * complex.I) = 1001 * complex.I) ∧ a = real.sqrt 101 :=
by
  sorry

end solve_complex_equation_l600_600722


namespace sqrt_36_is_pm_6_cube_root_sqrt_64_is_2_abs_neg_sqrt_2_is_sqrt_2_l600_600575

def sqrt_36 (x : ℝ) : Prop := x^2 = 36
def cube_root_sqrt_64 (x : ℝ) : Prop := x^3 = real.sqrt 64
def abs_neg_sqrt_2 (x : ℝ) : Prop := x = real.abs ( - real.sqrt 2)

theorem sqrt_36_is_pm_6 : sqrt_36 6 ∧ sqrt_36 (-6) :=
sorry

theorem cube_root_sqrt_64_is_2 : cube_root_sqrt_64 2 :=
sorry

theorem abs_neg_sqrt_2_is_sqrt_2 : abs_neg_sqrt_2 (real.sqrt 2) :=
sorry

end sqrt_36_is_pm_6_cube_root_sqrt_64_is_2_abs_neg_sqrt_2_is_sqrt_2_l600_600575


namespace area_difference_correct_l600_600015

def area_difference (a b : ℕ) (a_eq : a = 48) (b_eq : b = 64) : ℕ :=
  let c := Math.sqrt(gpow(a, 2) + gpow(b, 2));
  let x := 14; -- from the derived calculation
  let leg_length := b - x; -- leg_length = 64 - 14 = 50 mm
  let v := Math.sqrt((c / 2) ^ 2 + leg_length ^ 2 - leg_length ^ 2);
  let overlap_area := (c * v) / 2;
  let initial_area := a * b;
  initial_area - overlap_area

theorem area_difference_correct : area_difference 48 64 48 64 = 1200 :=
  by
    sorry

end area_difference_correct_l600_600015


namespace correct_population_definition_l600_600670

-- Definitions to represent the given conditions
def total_students : ℕ := 112000
def sample_size : ℕ := 200

-- Statement asserting the correct statistical interpretation
theorem correct_population_definition :
  (∃ (population : set ℕ), population = {x | x ∈ range total_students}) :=
  sorry

end correct_population_definition_l600_600670


namespace rounding_to_1_75_l600_600894

theorem rounding_to_1_75 (a : ℝ) : (1.745 ≤ a ∧ a < 1.755) ↔ (Real.round (a * 100) / 100 = 1.75) :=
by
  sorry

end rounding_to_1_75_l600_600894


namespace factorials_sum_of_two_squares_l600_600936

-- Define what it means for a number to be a sum of two squares.
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem factorials_sum_of_two_squares :
  {n : ℕ | n < 14 ∧ is_sum_of_two_squares (n!)} = {2, 6} :=
by
  sorry

end factorials_sum_of_two_squares_l600_600936


namespace Walter_receives_49_bananas_l600_600247

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end Walter_receives_49_bananas_l600_600247


namespace smallest_positive_period_and_monotonic_increase_range_of_g_for_given_interval_l600_600996

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x * (Real.sin x + Real.cos x)

noncomputable def g (x : ℝ) : ℝ :=
  f (x + Real.pi / 4)

theorem smallest_positive_period_and_monotonic_increase :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-Real.pi / 8 + k * Real.pi) (3 * Real.pi / 8 + k * Real.pi), 
    (∀ y ∈ Set.Icc x (3 * Real.pi / 8 + k * Real.pi), f y ≥ f x)) :=
by
  sorry

theorem range_of_g_for_given_interval :
  Set.range (g ∘ (λ x, Set.Icc 0 (Real.pi / 2))) = Set.Icc 0 (Real.sqrt 2 + 1) :=
by
  sorry

end smallest_positive_period_and_monotonic_increase_range_of_g_for_given_interval_l600_600996


namespace proof_problem_l600_600473

-- Definitions of points and vectors
def C : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (3, 4)
def N : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (0, 1)

-- Definition of vector operations
def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2)

-- Vectors needed
def AC : ℝ × ℝ := vector_sub C A
def AM : ℝ × ℝ := vector_sub M A
def AN : ℝ × ℝ := vector_sub N A

-- The Lean proof statement
theorem proof_problem :
  (∃ (x y : ℝ), AC = (x * AM.1 + y * AN.1, x * AM.2 + y * AN.2) ∧
     (x, y) = (2 / 3, 1 / 2)) ∧
  (9 * (2 / 3:ℝ) ^ 2 + 16 * (1 / 2:ℝ) ^ 2 = 8) :=
by
  sorry

end proof_problem_l600_600473


namespace sec_neg_420_eq_2_l600_600082

theorem sec_neg_420_eq_2 :
  let cos_60 := (1 / 2 : ℝ)
  in let cos_neg_60 := cos_60
  in let cos_neg_420 := cos_neg_60
  in let sec_neg_420 := 1 / cos_neg_420
  in sec_neg_420 = 2 :=
by
  sorry

end sec_neg_420_eq_2_l600_600082


namespace maximize_annual_avg_profit_l600_600517

-- Define the problem conditions
def purchase_cost : ℕ := 90000 
def initial_operating_cost : ℕ := 20000
def annual_cost_increase : ℕ := 20000
def annual_income : ℕ := 110000

-- Define the sequence for operating cost
def operating_cost (n : ℕ) : ℕ := 2 * n * 10000 -- in yuan

-- Total operating cost after n years
def total_operating_cost (n : ℕ) : ℕ := n * n * 10000 + n * 10000 -- in yuan

-- Total profit after n years
def total_profit (n : ℕ) : ℕ := 110000 * n - total_operating_cost n - purchase_cost

-- Annual average profit
def annual_avg_profit (n : ℕ) : ℕ := (total_profit n) / n

-- The goal statement
theorem maximize_annual_avg_profit : ∃ n : ℕ, n = 3 ∧ (∀ m : ℕ, m > 0 → annual_avg_profit n ≥ annual_avg_profit m) := sorry

end maximize_annual_avg_profit_l600_600517


namespace f_odd_and_periodic_l600_600280

noncomputable def f : ℝ → ℝ := sorry  -- The actual function definition is skipped

-- Condition 1
axiom h1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
-- Condition 2
axiom h2 : ∀ x : ℝ, f (20 + x) = - f (20 - x)

-- Theorem statement
theorem f_odd_and_periodic : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 40) = f x) :=
begin
  sorry  -- Proof omitted
end

end f_odd_and_periodic_l600_600280


namespace probability_of_red_ball_l600_600677

-- Define the conditions
def num_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

-- Calculate the probability
def probability_drawing_red_ball : ℚ := red_balls / num_balls

-- The theorem statement to be proven
theorem probability_of_red_ball : probability_drawing_red_ball = 2 / 3 :=
by
  sorry

end probability_of_red_ball_l600_600677


namespace marks_in_biology_l600_600537

theorem marks_in_biology (marks_english marks_math marks_physics marks_chemistry avg_marks : ℕ)
  (h_eng : marks_english = 61)
  (h_math : marks_math = 65)
  (h_phy : marks_physics = 82)
  (h_chem : marks_chemistry = 67)
  (h_avg : avg_marks = 72) :
  let marks_bio := avg_marks * 5 - (marks_english + marks_math + marks_physics + marks_chemistry) in
  marks_bio = 85 :=
by
  sorry

end marks_in_biology_l600_600537


namespace smallest_number_last_digits_4444_l600_600098

-- Define a predicate for a number being composed only of 4's and 9's
def composed_of_4s_and_9s (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 4 ∨ d = 9

-- Define a predicate for the exact count condition
def has_one_more_4_than_9s (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.count 4 = digits.count 9 + 1

-- Define the main property to be proved
theorem smallest_number_last_digits_4444 :
  ∃ (n : ℕ),
    composed_of_4s_and_9s n ∧
    nat.mod n 4 = 0 ∧
    nat.mod n 9 = 0 ∧
    has_one_more_4_than_9s n ∧
    n.digits 10 |>.drop (n.digits 10).length - 4 = [4, 4, 4, 4] :=
sorry

end smallest_number_last_digits_4444_l600_600098


namespace work_hours_to_pay_off_debt_l600_600239

theorem work_hours_to_pay_off_debt (initial_debt paid_amount hourly_rate remaining_debt work_hours : ℕ) 
  (h₁ : initial_debt = 100) 
  (h₂ : paid_amount = 40) 
  (h₃ : hourly_rate = 15) 
  (h₄ : remaining_debt = initial_debt - paid_amount) 
  (h₅ : work_hours = remaining_debt / hourly_rate) : 
  work_hours = 4 :=
by
  sorry

end work_hours_to_pay_off_debt_l600_600239


namespace range_of_a_l600_600211

variables (a : ℝ)

theorem range_of_a (h : ∀ x : ℝ, x > 0 → 2 * x * real.log x ≥ -x^2 + a * x - 3) : a ≤ 4 := by
  sorry

end range_of_a_l600_600211


namespace completing_square_l600_600806

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l600_600806


namespace overall_percentage_change_l600_600719

variable {S : ℝ}
def decrease_by (x : ℝ) (amount : ℝ) : ℝ := amount * (1 - x / 100)
def increase_by (x : ℝ) (amount : ℝ) : ℝ := amount * (1 + x / 100)

noncomputable def salary_change (initial_salary : ℝ) : ℝ :=
  let salary1 := decrease_by 40 initial_salary
  let salary2 := increase_by 30 salary1
  let salary3 := decrease_by 20 salary2
  let final_salary := increase_by 10 salary3
  ((final_salary - initial_salary) / initial_salary) * 100

theorem overall_percentage_change :
  salary_change S = -31.36 :=
by
  sorry

end overall_percentage_change_l600_600719


namespace arcsin_one_eq_pi_div_two_l600_600518

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := by
  sorry

end arcsin_one_eq_pi_div_two_l600_600518


namespace sixteenth_number_l600_600370

/-- Prove that the 16th 4-digit number in numerical order, using the digits 1, 3, 6, and 8 exactly once, is 6381. -/
theorem sixteenth_number (numbers : list ℕ) : 
  (permutations [1, 3, 6, 8]).nth 15 = some 6381 :=
by
  sorry

end sixteenth_number_l600_600370


namespace part1_part2_l600_600623

noncomputable def f : ℝ → ℝ 
| x => if 0 ≤ x then 2^x - 1 else -2^(-x) + 1

theorem part1 (x : ℝ) (h : x < 0) : f x = -2^(-x) + 1 := sorry

theorem part2 (a : ℝ) : f a ≤ 3 ↔ a ≤ 2 := sorry

end part1_part2_l600_600623


namespace find_number_l600_600357

open Nat

theorem find_number 
  (A B : ℕ) 
  (HCF : ℕ → ℕ → ℕ) 
  (LCM : ℕ → ℕ → ℕ) 
  (h1 : B = 156) 
  (h2 : HCF A B = 12) 
  (h3 : LCM A B = 312) : 
  A = 24 :=
by
  sorry

end find_number_l600_600357


namespace sum_rational_root_cs_l600_600100

def sum_of_valid_c (c : ℤ) : Prop :=
  c ≤ 30 ∧ ∃ k : ℤ, 81 + 4 * c = k^2

theorem sum_rational_root_cs : 
  (∑ c in Finset.filter sum_of_valid_c (Finset.range 31), c) = 32 := 
by
  -- Proof omitted
  sorry

end sum_rational_root_cs_l600_600100


namespace find_ff_neg_1_l600_600737

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 else -x + 1

theorem find_ff_neg_1 : f (f (-1)) = 6 :=
sorry

end find_ff_neg_1_l600_600737


namespace remaining_payment_correct_l600_600847

-- Definitions based on conditions
def part_payment : ℝ := 300
def payment_percentage : ℝ := 0.05
def total_cost_of_car : ℝ := part_payment / payment_percentage
def remaining_amount_to_be_paid : ℝ := total_cost_of_car - part_payment

-- Proof problem statement
theorem remaining_payment_correct :
  remaining_amount_to_be_paid = 5700 :=
sorry

end remaining_payment_correct_l600_600847


namespace parameterization_solution_l600_600738

/-- Proof problem statement:
  Given the line equation y = 3x - 11 and its parameterization representation,
  the ordered pair (s, h) that satisfies both conditions is (3, 15).
-/
theorem parameterization_solution : ∃ s h : ℝ, 
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (s, -2) + t • (5, h)) ∧ y = 3 * x - 11) → 
  (s = 3 ∧ h = 15) :=
by
  -- introduce s and h 
  use 3
  use 15
  -- skip the proof
  sorry

end parameterization_solution_l600_600738


namespace cheryl_found_more_eggs_l600_600920

theorem cheryl_found_more_eggs :
  let kevin_eggs := 5
  let bonnie_eggs := 13
  let george_eggs := 9
  let cheryl_eggs := 56
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end cheryl_found_more_eggs_l600_600920


namespace height_of_box_l600_600853

-- Define the constants and spaces related to the spheres and the box.
structure Box :=
  (length : ℝ)
  (width : ℝ)
  (height : ℝ)

structure Sphere :=
  (radius : ℝ)
  (center : ℝ × ℝ × ℝ)

-- Conditions given in the problem.
def larger_sphere : Sphere :=
  { radius := 3, center := (3, 3, 3) }

def smaller_sphere (i : ℕ) : Sphere :=
  { radius := 1.5, center := (1.5 * (i % 2), 1.5 * ((i / 2) % 2), 1.5 * (i / 4)) }

def box : Box :=
  { length := 6, width := 6, height := 9 }

-- Statement of the problem to be proved.
theorem height_of_box : box.height = 9 :=
by
  sorry

end height_of_box_l600_600853


namespace average_rate_of_change_sqrt_1_to_4_l600_600322

def average_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

def f (x : ℝ) : ℝ := Real.sqrt x

theorem average_rate_of_change_sqrt_1_to_4 : 
  average_rate_of_change f 1 4 = 1 / 3 := by
  sorry

end average_rate_of_change_sqrt_1_to_4_l600_600322


namespace cos_neg_double_angle_l600_600151

theorem cos_neg_double_angle (α : ℝ) (h : sin α = 2 / 3) : cos (-2 * α) = 1 / 9 :=
by
  sorry

end cos_neg_double_angle_l600_600151


namespace pipe_ratio_l600_600715

theorem pipe_ratio (A B : ℝ) (hA : A = 1 / 12) (hAB : A + B = 1 / 3) : B / A = 3 := by
  sorry

end pipe_ratio_l600_600715


namespace woodworker_furniture_legs_l600_600891

theorem woodworker_furniture_legs (num_chairs num_tables : ℕ) (legs_per_chair legs_per_table : ℕ) :
  num_chairs = 6 →
  num_tables = 4 →
  legs_per_chair = 4 →
  legs_per_table = 4 →
  let total_legs := num_chairs * legs_per_chair + num_tables * legs_per_table in
  total_legs = 40 :=
by
  intros h_chairs h_tables h_legs_chair h_legs_table
  rw [h_chairs, h_tables, h_legs_chair, h_legs_table]
  have : 6 * 4 + 4 * 4 = 40 := by norm_num
  exact this


end woodworker_furniture_legs_l600_600891


namespace angleina_speed_from_grocery_to_gym_l600_600902

variable (v : ℝ) (h1 : 720 / v - 40 = 240 / v)

theorem angleina_speed_from_grocery_to_gym : 2 * v = 24 :=
by
  sorry

end angleina_speed_from_grocery_to_gym_l600_600902


namespace df_calculation_l600_600261

noncomputable def solve_df (ABCD_side : ℝ) (Area_square : ℝ) (BX : ℝ) :=
∃ (DF : ℝ), 
  (ABCD_side = 13) ∧ 
  (Area_square = 169) ∧ 
  (BX = 6) ∧ 
  (DF = Real.sqrt 13)

theorem df_calculation : 
  solve_df 13 169 6 :=
begin
  use Real.sqrt 13,
  split, 
  { refl },
  split,
  { refl },
  split,
  { refl },
  { refl }
end

end df_calculation_l600_600261


namespace sqrt_expression_identity_l600_600908

theorem sqrt_expression_identity : 
    ( √5 + √2 ) * ( √5 - √2 ) - ( ( √3 - √2 ) ^ 2 ) = 2 * √6 - 2 :=
by
    sorry

end sqrt_expression_identity_l600_600908


namespace sqrt_expr_simplification_l600_600513

theorem sqrt_expr_simplification :
  (real.sqrt 27) / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - (6 * real.sqrt 2) = (6 * real.sqrt 2) :=
by
  sorry

end sqrt_expr_simplification_l600_600513


namespace player1_max_score_l600_600378

/--
Two players play alternately on a 5×5 board. The first player always enters a 1 into an empty square,
and the second player always enters a 0 into an empty square. When the board is full, the sum of the
numbers in each of the nine 3×3 squares is calculated. Prove that the largest score the first player
can make, regardless of the responses of the second player, is 6.
-/
theorem player1_max_score : 
  ∀ (board : Σ (r c : Fin 5) (filled : Fin 25), (board (r, c) = 1 ∨ board (r, c) = 0))
  (Sum3x3 : ∀ (r c : Fin 3), ∑ x in Finset.filter (λ y : Fin 9, y < r + 3 ∧ y < c + 3), board x),
  (Sum3x3.range.max) = 6 :=
sorry

end player1_max_score_l600_600378


namespace aprils_plant_arrangement_l600_600048

theorem aprils_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 3
  let total_units := basil_plants + 1
  
  (fact total_units * fact tomato_plants = 4320) :=
by
  unfold basil_plants
  unfold tomato_plants
  unfold total_units
  apply eq.refl
  sorry

end aprils_plant_arrangement_l600_600048


namespace proof_problem_l600_600851

variable (α β γ : Type) (l : Prop)

-- Definitions for the conditions given in the problem
def prop1 (h1 : α ⊥ β) (h2 : β ⊥ γ) : α ⊥ γ := sorry
def prop2 (h1 : α ‖ β) (h2 : l ‖ β) : l ‖ α := sorry
def prop3 (h1 : l ⊥ α) (h2 : l ‖ β) : α ⊥ β := sorry
def prop4 (h1 : α ‖ β) (h2 : α ⊥ γ) : β ⊥ γ := sorry

-- The proof problem statement
theorem proof_problem :
  (¬ prop1 (∃ h1, α ⊥ β) (∃ h2, β ⊥ γ))
  ∧ (¬ prop2 (∃ h1, α ‖ β) (∃ h2, l ‖ β))
  ∧ prop3 (∃ h1, l ⊥ α) (∃ h2, l ‖ β)
  ∧ prop4 (∃ h1, α ‖ β) (∃ h2, α ⊥ γ) :=
  sorry

end proof_problem_l600_600851


namespace proof_BC_length_l600_600913

open Classical

variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace D] [MetricSpace E] (circle : A) (radius : ℝ) 
          (on_circle : B → Prop) (intersect_at_A : B × C → A) 
          (length_AD length_AE : ℝ) (right_angle : Prop) (length_BC : ℝ)

-- Given conditions
variable (r6 : radius = 6)    
variable (pts_on_circle : ∀ (x : B), on_circle x)    
variable (AB_inter : intersect_at_A (B, C) = A)
variable (AD5 : length_AD = 5)    
variable (AE4 : length_AE = 4)    
variable (angle_90 : right_angle = true)

noncomputable def length_BC := (12 + 9 * Real.sqrt 15) / 5

theorem proof_BC_length :
  length_BC AD5 AE4 angle_90 r6 pts_on_circle AB_inter = (12 + 9 * Real.sqrt 15) / 5 :=
by
  sorry

end proof_BC_length_l600_600913


namespace part_I_part_II_l600_600631

noncomputable def f (x : ℝ) (a : ℝ) (omega : ℝ) : ℝ :=
  2 * a * sin(omega * x) * cos(omega * x) + 2 * sqrt 3 * (cos (omega * x))^2 - sqrt 3

theorem part_I (a omega : ℝ) (h_a : 0 < a) (h_omega : 0 < omega)
  (h_max : ∀ x, f x a omega ≤ 2)
  (h_period : ∃ T > 0, ∀ x, f (x + T) a omega = f x a omega) :
  f x a omega = 2 * sin (2 * x + π / 3) :=
sorry

theorem part_II (alpha : ℝ) (h_falpha : f alpha 1 1 = 4 / 3) :
  sin (4 * alpha + π / 6) = -1 / 9 :=
sorry

end part_I_part_II_l600_600631


namespace probability_triangle_segments_l600_600975

theorem probability_triangle_segments :
  let segments := [2, 3, 4, 5]
  let total_combinations := 4.choose 3
  let valid_combinations := (
    if (2 + 3 > 4) ∧ (2 + 4 > 3) ∧ (3 + 4 > 2) then 1 else 0
  ) + (
    if (2 + 4 > 5) ∧ (2 + 5 > 4) ∧ (4 + 5 > 2) then 1 else 0
  ) + (
    if (3 + 4 > 5) ∧ (3 + 5 > 4) ∧ (4 + 5 > 3) then 1 else 0
  )
  let probability := valid_combinations / total_combinations
  probability = 3 / 4 :=
by
  sorry

end probability_triangle_segments_l600_600975


namespace dice_sum_probability_15_eq_41_l600_600392

/-- Prove that the probability of rolling a sum of 15 on 8 standard dice is equal to
    the probability of rolling a sum of 41 on 8 standard dice. -/
theorem dice_sum_probability_15_eq_41 :
  let dice_faces := {1, 2, 3, 4, 5, 6}
  ∃ q : ℚ, (probability (sum_top_faces 8 dice_faces = 15) = q) ∧ 
           (probability (sum_top_faces 8 dice_faces = 41) = q) := 
sorry

end dice_sum_probability_15_eq_41_l600_600392


namespace min_value_is_3024_l600_600110

def is_good_partition (A A1 A2 : Finset ℕ) : Prop :=
  A1 ∪ A2 = A ∧ A1 ∩ A2 = ∅ ∧ A1 ≠ ∅ ∧ A2 ≠ ∅ ∧
  Finset.lcm (A1 : Set ℕ) = Finset.gcd (A2 : Set ℕ)

noncomputable def min_n_with_good_partitions (P : ℕ) : ℕ :=
  Inf {n : ℕ | ∃ (A : Finset ℕ), A.card = n ∧ (∃ (G : Finset (Finset ℕ × Finset ℕ)),
   G.card = P ∧ ∀ (p : Finset ℕ × Finset ℕ) ∈ G, is_good_partition A p.1 p.2)}

theorem min_value_is_3024 : min_n_with_good_partitions 2015 = 3024 :=
  sorry

end min_value_is_3024_l600_600110


namespace not_identity_element_l600_600538

def S := {x : ℝ // x ≠ 0}
def star (a b : S) : S := ⟨a.1 * b.1 + 1, sorry⟩ -- Including sorry for the non-zero condition proof

theorem not_identity_element (h : ∀ a : S, star a ⟨1, sorry⟩ = a ∧ star ⟨1, sorry⟩ a = a) : false :=
by
  have h1 : ∀ a : S, star a ⟨1, sorry⟩ = ⟨a.1 + 1, sorry⟩ := sorry
  have h2 : ∀ a : S, star ⟨1, sorry⟩ a = ⟨a.1 + 1, sorry⟩ := sorry
  -- Detail the proof step here
  sorry

end not_identity_element_l600_600538


namespace circle_diameter_l600_600430

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l600_600430


namespace limit_of_a_n_l600_600184

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Define the sequence {a_n} 
def a_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + (1 / (2 * a n))

-- Define the limit condition we need to prove
def a_limit (a : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - real.sqrt n| < ε

-- State the final proof problem
theorem limit_of_a_n (a : ℕ → ℝ) (h : a_sequence a) : a_limit a :=
sorry

end limit_of_a_n_l600_600184


namespace range_of_a_l600_600177

noncomputable theory

def f (x a : ℝ) : ℝ := exp (-x) - 2 * x - a
def curve (x : ℝ) : ℝ := x^3 + x + 1

theorem range_of_a (a : ℝ) :
  (∃ (x₀ ∈ Icc (-1 : ℝ) 1), y₀ = curve x₀ ∧ f y₀ a = y₀) ↔
  (a ∈ Icc (exp (-3) - 9) (exp 1 + 3)) :=
sorry

end range_of_a_l600_600177


namespace determine_b_l600_600995

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x < 1 then 3 * x - b else 2 ^ x

theorem determine_b (b : ℝ) :
  f (f (5 / 6) b) b = 4 ↔ b = 1 / 2 :=
by sorry

end determine_b_l600_600995


namespace integral_evaluation_l600_600560

theorem integral_evaluation : ∫ x in 0..2, (2 * x - 3 * x ^ 2) = -4 :=
by sorry

end integral_evaluation_l600_600560


namespace quadratic_roots_eccentricities_l600_600766

theorem quadratic_roots_eccentricities :
  (∃ x y : ℝ, 3 * x^2 - 4 * x + 1 = 0 ∧ 3 * y^2 - 4 * y + 1 = 0 ∧ 
              (0 ≤ x ∧ x < 1) ∧ y = 1) :=
by
  -- Proof would go here
  sorry

end quadratic_roots_eccentricities_l600_600766


namespace binomial_coeff_12_3_l600_600525

/-- The binomial coefficient is defined as: 
  \binom{n}{k} = \frac{n!}{k!(n-k)!} -/
theorem binomial_coeff_12_3 : Nat.binom 12 3 = 220 := by
  sorry

end binomial_coeff_12_3_l600_600525


namespace find_point_C_l600_600684

theorem find_point_C :
  ∃ C : ℂ, 
    (let A := (2 + 1 * complex.i) in
    let BA := (1 + 2 * complex.i) in
    let BC := (3 - 1 * complex.i) in
    let B := A - BA in
    C = B + BC) := 
begin
  use (4 - 2 * complex.i),
  sorry
end

end find_point_C_l600_600684


namespace arrange_abc_l600_600118

noncomputable def a : ℝ := Real.log (4) / Real.log (0.3)
noncomputable def b : ℝ := Real.log (0.2) / Real.log (0.3)
noncomputable def c : ℝ := (1 / Real.exp 1) ^ Real.pi

theorem arrange_abc (a := a) (b := b) (c := c) : b > c ∧ c > a := by
  sorry

end arrange_abc_l600_600118


namespace complex_number_solution_l600_600594

theorem complex_number_solution (z : ℂ) (h : (1 + 2 * complex.I) * z = 4 + 3 * complex.I) : z = 2 - complex.I :=
by sorry

end complex_number_solution_l600_600594


namespace part_a_part_b_l600_600697

variable {n : Type*} [fintype n] [decidable_eq n]

def Rn := n → ℝ

variables (α : ℝ) (α_nonzero : α ≠ 0)
variables (F G : Rn →ₗ[ℝ] Rn)
variable (commute_condition : F.comp G - G.comp F = α • F)

theorem part_a (k : ℕ) : (F^k).comp G - G.comp (F^k) = α * k • (F^k) :=
sorry

theorem part_b : ∃ k ≥ 1, (F^k : Rn →ₗ[ℝ] Rn) = 0 :=
sorry

end part_a_part_b_l600_600697


namespace number_of_valid_conclusions_l600_600267

variables {a b : ℝ} (va : ℝ) (vab : ℝ → ℝ → ℝ) 
variables (h_neq_zero_a : va ≠ 0) (h_neq_zero_b : ∀ b, vab va b ≠ 0)

open_locale classical

lemma vector_magnitude_eq (a b : ℝ) : (a + b)^2 = (a - b)^2 ↔ a * b = 0 := sorry
lemma vector_magnitude_sum (a b : ℝ) : (a + b)^2 = (va^2) + (vab va b)^2 ↔ a = b := sorry
lemma vector_magnitude_lt (a b : ℝ) :  (a)^2 + (va)^2 < (a - va)^2 ↔ a < 0 := sorry
lemma vector_magnitude_cond (a b : ℝ) : (a + b)^2 = (a - b)^2 ↔ a ≥ b := sorry

theorem number_of_valid_conclusions : 2 := by
  split,
  exact vector_magnitude_eq a b,
  exact vector_magnitude_sum a b,
  exact vector_magnitude_lt a b,
  exact vector_magnitude_cond a b,
  all_goals { sorry }

end number_of_valid_conclusions_l600_600267


namespace point_A_in_QuadrantIII_l600_600679

-- Define the Cartesian Point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point being in Quadrant III
def inQuadrantIII (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Given point A
def A : Point := { x := -1, y := -2 }

-- The theorem stating that point A lies in Quadrant III
theorem point_A_in_QuadrantIII : inQuadrantIII A :=
  by
    sorry

end point_A_in_QuadrantIII_l600_600679


namespace proof_equivalent_expression_l600_600829

def equivalent_expression : Prop :=
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^{16} + 5^{16}) * (4^{32} + 5^{32}) * (4^{64} + 5^{64}) = 9^{127}

theorem proof_equivalent_expression : equivalent_expression := by
  sorry

end proof_equivalent_expression_l600_600829


namespace circle_radius_l600_600754

theorem circle_radius (r : ℝ) (h_circumference : 2 * Real.pi * r) 
                      (h_area : Real.pi * r^2) 
                      (h_equation : 3 * (2 * Real.pi * r) = Real.pi * r^2) : 
                      r = 6 :=
by
  sorry

end circle_radius_l600_600754


namespace view_from_gazebo_l600_600682

noncomputable def circular_garden_radius := 50

def tree_radius_unobstructed : ℝ := 1 / Real.sqrt 2501
def tree_radius_obstructed : ℝ := 1 / 50

def view_clear (r : ℝ) : Prop :=
  r < tree_radius_unobstructed

def view_obstructed (r : ℝ) : Prop :=
  r = tree_radius_obstructed

theorem view_from_gazebo :
  ∀ r : ℝ, (r < tree_radius_unobstructed → view_clear r) ∧ (r = tree_radius_obstructed → view_obstructed r) :=
by
  sorry

end view_from_gazebo_l600_600682


namespace parabola_line_intersect_l600_600599

noncomputable def parabola_focus : ℝ × ℝ := (2, 0)
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

theorem parabola_line_intersect (A B : ℝ × ℝ) (x1 x2 : ℝ) (y1 y2 : ℝ) 
    (hA : A = (x1, y1)) (hB : B = (x2, y2)) (h_focus : ∃ k, y1 = k * (x1 - 2) ∧ y2 = k * (x2 - 2)) 
    (h_points : parabola x1 y1 ∧ parabola x2 y2) (h_dist_AB : abs ((x2 - x1)^2 + (y2 - y1)^2)^0.5 = 10) :
    (abs (x1 - 2) + abs (x2 - 2)) * abs ((x1 - x2) * k + (y1 - y2) * (1/k)) = 20 := sorry

end parabola_line_intersect_l600_600599


namespace carl_typing_speed_l600_600515

theorem carl_typing_speed (words_per_day: ℕ) (minutes_per_day: ℕ) (total_words: ℕ) (days: ℕ) : 
  words_per_day = total_words / days ∧ 
  minutes_per_day = 4 * 60 ∧ 
  (words_per_day / minutes_per_day) = 50 :=
by 
  sorry

end carl_typing_speed_l600_600515


namespace negative_implies_neg_reciprocal_positive_l600_600205

theorem negative_implies_neg_reciprocal_positive {x : ℝ} (h : x < 0) : -x⁻¹ > 0 :=
sorry

end negative_implies_neg_reciprocal_positive_l600_600205


namespace solve_first_sales_amount_l600_600444

noncomputable def first_sales_amount
  (S : ℝ) (R : ℝ) (next_sales_royalties : ℝ) (next_sales_amount : ℝ) : Prop :=
  (3 = R * S) ∧ (next_sales_royalties = 0.85 * R * next_sales_amount)

theorem solve_first_sales_amount (S R : ℝ) :
  first_sales_amount S R 9 108 → S = 30.6 :=
by
  intro h
  sorry

end solve_first_sales_amount_l600_600444


namespace prime_dates_in_2004_l600_600287

def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the prime months in 2004
def prime_months := {2, 3, 5, 7, 11}

-- February in a leap year has 29 days
-- March, May, and July have 31 days
-- November has 30 days

-- Define the prime days in respective months
def prime_days_feb := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
def prime_days_mar := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
def prime_days_may := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
def prime_days_jul := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
def prime_days_nov := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def count_prime_dates_2004 : ℕ := 
  prime_days_feb.size + prime_days_mar.size + prime_days_may.size + prime_days_jul.size + prime_days_nov.size

theorem prime_dates_in_2004 : count_prime_dates_2004 = 53 :=
by
  sorry

end prime_dates_in_2004_l600_600287


namespace problem_f_of_f_neg1_eq_neg1_l600_600707

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

-- State the proposition to be proved
theorem problem_f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := by
  sorry

end problem_f_of_f_neg1_eq_neg1_l600_600707


namespace max_area_of_rectangular_yard_l600_600077

theorem max_area_of_rectangular_yard (P : ℕ) (hP : P = 150) (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 75) :
  ∃ A, A = x * (75 - x) ∧ A ≤ 1406 :=
by
  have x_le_37_5 : x ≤ 37 ∨ x ≥ 38 := sorry
  cases x_le_37_5 with
  | inl hle =>
    use x * (75 - x)
    split
    · sorry
    · sorry
  | inr hge =>
    use x * (75 - x)
    split
    · sorry
    · sorry

end max_area_of_rectangular_yard_l600_600077


namespace k_not_possible_l600_600990

theorem k_not_possible (S : ℕ → ℚ) (a b : ℕ → ℚ) (n k : ℕ) (k_gt_2 : k > 2) :
  (S n = (n^2 + n) / 2) →
  (a n = S n - S (n - 1)) →
  (b n = 1 / a n) →
  (2 * b (n + 2) = b n + b (n + k)) →
  k ≠ 4 ∧ k ≠ 10 :=
by
  -- Proof goes here (skipped)
  sorry

end k_not_possible_l600_600990


namespace possible_value_of_S_n_plus_1_l600_600696

def S (n : ℕ) : ℕ := sorry -- Placeholder for the sum of digits function.

theorem possible_value_of_S_n_plus_1 (n : ℕ) (h : S(n) = 1274) : S(n + 1) = 1239 :=
sorry

end possible_value_of_S_n_plus_1_l600_600696


namespace abs_sum_inequality_l600_600408

theorem abs_sum_inequality (x : ℝ) : (|x - 2| + |x + 3| < 7) ↔ (-6 < x ∧ x < 3) :=
sorry

end abs_sum_inequality_l600_600408


namespace dodecagon_product_l600_600718

open Complex

variables {x y : ℝ}
noncomputable def Q : ℕ → ℂ
| 1 := 1.5 + 0 * I
| 7 := 2.5 + 0 * I
| n := exp (2 * Real.pi * I * (n - 1) / 12) + 2

theorem dodecagon_product :
  (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6) * (Q 7) * (Q 8) * (Q 9) * (Q 10) * (Q 11) * (Q 12) = 4095 :=
sorry

end dodecagon_product_l600_600718


namespace smallest_three_digit_number_exists_l600_600029

def is_valid_permutation_sum (x y z : ℕ) : Prop :=
  let perms := [100*x + 10*y + z, 100*x + 10*z + y, 100*y + 10*x + z, 100*z + 10*x + y, 100*y + 10*z + x, 100*z + 10*y + x]
  perms.sum = 2220

theorem smallest_three_digit_number_exists : ∃ (x y z : ℕ), x < y ∧ y < z ∧ x + y + z = 10 ∧ is_valid_permutation_sum x y z ∧ 100 * x + 10 * y + z = 127 :=
by {
  -- proof goal and steps would go here if we were to complete the proof
  sorry
}

end smallest_three_digit_number_exists_l600_600029


namespace solve_for_m_l600_600655

theorem solve_for_m (x m : ℝ) (h : (∃ x, (x - 1) / (x - 4) = m / (x - 4))): 
  m = 3 :=
by {
  sorry -- placeholder to indicate where the proof would go
}

end solve_for_m_l600_600655


namespace average_high_temperature_l600_600035

theorem average_high_temperature :
  let highs := [51, 63, 59, 56, 47, 64, 52] in
  ∑ i in highs, i / highs.length = 56 := by
  sorry

end average_high_temperature_l600_600035


namespace find_line_eq_l600_600943

noncomputable def line_eq (x y : ℝ) : Prop :=
  (∃ a : ℝ, a ≠ 0 ∧ (a * x - y = 0 ∨ x + y - a = 0)) 

theorem find_line_eq : line_eq 2 3 :=
by
  sorry

end find_line_eq_l600_600943


namespace radius_of_circle_l600_600750

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * real.pi * r) = real.pi * r^2) : r = 6 :=
by {
    sorry
}

end radius_of_circle_l600_600750


namespace number_of_people_in_group_l600_600323

theorem number_of_people_in_group
    (n : ℕ)
    (h1 : 6.3 * n = 63)
    (h2 : 128 - 65 = 63) :
    n = 10 :=
begin
  sorry
end

end number_of_people_in_group_l600_600323


namespace set_C_when_m_3_range_of_m_if_A_inter_B_eq_B_l600_600187

-- Definitions based on the given problem
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}
def C (m : ℝ) : Set ℤ := {x : ℤ | x ∈ A ∨ x ∈ B m}

-- Statement for the first question
theorem set_C_when_m_3 : C 3 = {-3, -2, -1, 0, 1, 2, 3, 4, 5} :=
by
  sorry

-- Statement for the second question
theorem range_of_m_if_A_inter_B_eq_B {m : ℝ} (h : A ∩ B m = B m) : m ≤ 3 :=
by
  sorry

end set_C_when_m_3_range_of_m_if_A_inter_B_eq_B_l600_600187


namespace sum_base5_l600_600948

theorem sum_base5 : 
  let a : ℕ := 201  -- base 5
  let b : ℕ := 324  -- base 5
  let c : ℕ := 143  -- base 5
  let sum := a + b + c  -- sum in base 5
  -- convert sum in base 5 to the result sum to check
  int.toNat (nat.digits_clean 5 [2, 4, 3, 1]) = 1123 :=
by
  sorry

end sum_base5_l600_600948


namespace max_value_of_sum_l600_600155

theorem max_value_of_sum (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) 
  (hgeom : sqrt 3 * b = sqrt ((1 - a) * (1 + a))): a + sqrt 3 * b ≤ sqrt 2 := 
sorry

end max_value_of_sum_l600_600155


namespace smallest_n_l600_600097

theorem smallest_n 
  : ∃ n : ℕ+, (∀ m : ℕ, (m > 1 → m % 2 = 1 → 2^1989 ∣ (m^n - 1))) ∧ n = 2^1987 :=
  sorry

end smallest_n_l600_600097


namespace num_handshakes_7_people_l600_600106

theorem num_handshakes_7_people : 
  let n := 7 in let H := n * (n - 1) / 2 in H = 21 :=
by
  sorry

end num_handshakes_7_people_l600_600106


namespace train_passes_jogger_in_23_77_seconds_l600_600008

noncomputable def jogger_speed_kmh : ℝ := 7
noncomputable def train_speed_kmh : ℝ := 60
noncomputable def initial_gap_m : ℝ := 150
noncomputable def train_length_m : ℝ := 200

noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * (1000 / 3600)
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
noncomputable def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_m : ℝ := initial_gap_m + train_length_m
noncomputable def time_to_pass_s : ℝ := total_distance_m / relative_speed_ms

theorem train_passes_jogger_in_23_77_seconds :
  time_to_pass_s ≈ 23.77 :=
by
  -- The proof goes here
  sorry

end train_passes_jogger_in_23_77_seconds_l600_600008


namespace baker_new_cakes_l600_600475

theorem baker_new_cakes (original_cakes sold_cakes current_cakes : ℕ) :
  original_cakes = 121 → sold_cakes = 105 → current_cakes = 186 →
  current_cakes - (original_cakes - sold_cakes) = 170 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end baker_new_cakes_l600_600475


namespace box_dimension_min_sum_l600_600733

theorem box_dimension_min_sum :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 := by
  sorry

end box_dimension_min_sum_l600_600733


namespace find_a_l600_600657

theorem find_a (a b c : ℚ)
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) : a = 10 / 11 :=
by
  sorry

end find_a_l600_600657


namespace initial_turban_price_l600_600192

theorem initial_turban_price
  (annual_salary: ℝ)
  (months_worked: ℝ)
  (usd_received: ℝ)
  (final_turban_value_inr: ℝ)
  (initial_exchange_rate: ℝ)
  (final_exchange_rate: ℝ)
  : (annual_salary = 90) →
    (months_worked = 9) →
    (usd_received = 55) →
    (final_turban_value_inr = 2500) →
    (initial_exchange_rate = 50) →
    (final_exchange_rate = 45) →
    let monthly_salary := annual_salary / 12 in
    let prorated_salary := monthly_salary * months_worked in
    let turban_value_usd := prorated_salary - usd_received in
    let sold_turban_usd := final_turban_value_inr / final_exchange_rate in
    let devaluation_rate := (initial_exchange_rate - final_exchange_rate) / initial_exchange_rate in
    let initial_turban_value_inr := final_turban_value_inr / (1 - devaluation_rate) in
    let initial_turban_value_usd := initial_turban_value_inr / initial_exchange_rate in
    initial_turban_value_usd = 55.56 :=
begin
  intros,
  let monthly_salary := annual_salary / 12,
  let prorated_salary := monthly_salary * months_worked,
  let turban_value_usd := prorated_salary - usd_received,
  let sold_turban_usd := final_turban_value_inr / final_exchange_rate,
  let devaluation_rate := (initial_exchange_rate - final_exchange_rate) / initial_exchange_rate,
  let initial_turban_value_inr := final_turban_value_inr / (1 - devaluation_rate),
  let initial_turban_value_usd := initial_turban_value_inr / initial_exchange_rate,
  sorry,
end

end initial_turban_price_l600_600192


namespace dolls_per_store_l600_600411

theorem dolls_per_store (total_dolls : Nat) (defective_dolls : Nat) (stores : Nat)
  (h1 : total_dolls = 40) (h2 : defective_dolls = 4) (h3 : stores = 4) :
  (total_dolls - defective_dolls) / stores = 9 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end dolls_per_store_l600_600411


namespace y_intercept_probability_l600_600637

noncomputable def probability_y_intercept_greater_than_1 (b : ℝ) : ℝ :=
  if h : b ∈ Icc (-2 : ℝ) 3 then
    (3 - 1) / (3 - (-2))
  else 0

theorem y_intercept_probability :
  ∀ b ∈ Icc (-2 : ℝ) 3, probability_y_intercept_greater_than_1 b = 2 / 5 := by
  intros b hb
  dsimp [probability_y_intercept_greater_than_1]
  rw if_pos hb
  norm_num
  sorry

end y_intercept_probability_l600_600637


namespace problem1_circle_equation_problem2_chord_length_l600_600536

theorem problem1_circle_equation : 
  ∃ x y : ℝ, 
    (x^2 + y^2 - 3 * real.sqrt 3 * x - 3 * y + 8 = 0) :=
sorry

theorem problem2_chord_length :
  ∀ (t : ℝ), 
    let x := -1 + (real.sqrt 3 / 2) * t in
    let y := (1 / 2) * t in
    ((x + 1) ^ 2 + y ^ 2 - (real.sqrt 3 * x + 3 * y) + 8 = 0) →
      ∃ t1 t2 : ℝ, 
        (t1 + t2 = 6 + real.sqrt 3) ∧ (t1 * t2 = 9 + 3 * real.sqrt 3) ∧ 
        (real.abs (t1 - t2) = real.sqrt 3) :=
sorry

end problem1_circle_equation_problem2_chord_length_l600_600536


namespace circle_diameter_l600_600433

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l600_600433


namespace num_valid_arrangements_l600_600472

/-- The number of different arrangements of the numbers 1, 2, ..., 7 such that the sum of the first k elements is 
never divisible by 3 for k = 1 to 7 is 360. -/
theorem num_valid_arrangements : 
  (finset.univ.permutations.filter (λ σ, ∀ k, (finset.range (k+1)).sum (λ i, σ.nth i 0) % 3 ≠ 0)).card = 360 :=
sorry

end num_valid_arrangements_l600_600472


namespace find_third_month_sales_l600_600879

def monthly_sales : ℕ → ℕ
| 0 := 7435
| 1 := 7920
| 2 := 0 -- Placeholder for the third month
| 3 := 8230
| 4 := 7560
| 5 := 6000
| _ := 0 -- Assume sales are 0 for months outside 0-5

theorem find_third_month_sales (avg_sale : ℕ) (months : ℕ) (total_known_sales : ℕ) : 
  let total_sales_needed := avg_sale * months in
  let sales_in_third_month := total_sales_needed - total_known_sales in
  avg_sale = 7500 →
  months = 6 →
  total_known_sales = (monthly_sales 0) + (monthly_sales 1) + (monthly_sales 3) + (monthly_sales 4) + (monthly_sales 5) →
  sales_in_third_month = 7855 :=
by 
  intros avg_sale months total_known_sales avg_7500 months_6
  have total_known_sales_value : total_known_sales = 7435 + 7920 + 8230 + 7560 + 6000 := by sorry
  have total_sales_needed_value : total_sales_needed = 7500 * 6 := by sorry
  have sales_in_third_month_value : sales_in_third_month = 45000 - 37145 := by sorry
  sorry

end find_third_month_sales_l600_600879


namespace ellipse_eccentricity_l600_600209

theorem ellipse_eccentricity (m : ℝ) (h1 : 0 < m) (h2 : ∃ x y : ℝ, x^2 / 4 + y^2 / m = 1)
  (h3 : eccentricity(e : ℝ) : Ellipse -> Ellipse.e = sqrt 3 / 2) 
  (epi : (sqrt (m - 4)) / sqrt m = sqrt 3 / 2 ∨ (sqrt (4 - m)) / 2 = sqrt 3 / 2) :
  m = 1 ∨ m = 16 := 
by sorry

end ellipse_eccentricity_l600_600209


namespace Walter_receives_49_bananas_l600_600248

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end Walter_receives_49_bananas_l600_600248


namespace faster_train_passes_slower_in_72_seconds_l600_600383

-- Define the conditions
def length_of_each_train : ℝ := 100 -- length in meters
def speed_of_faster_train : ℝ := 46 -- speed in km/hr
def speed_of_slower_train : ℝ := 36 -- speed in km/hr

-- Conversion factor from km/hr to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Calculate the relative speed in m/s
def relative_speed : ℝ := (speed_of_faster_train - speed_of_slower_train) * kmph_to_mps

-- Calculate the total distance to be covered
def total_distance : ℝ := 2 * length_of_each_train

-- Define the proof problem
theorem faster_train_passes_slower_in_72_seconds : total_distance / relative_speed = 72 := 
by
  sorry

end faster_train_passes_slower_in_72_seconds_l600_600383


namespace log_transform_solution_l600_600153

theorem log_transform_solution (x : ℝ) (log_cond : log 16 (x - 6) = 1 / 2) : 
  1 / log x 4 ≈ 1.6607 :=
by 
  sorry

end log_transform_solution_l600_600153


namespace slope_of_line_l600_600636

-- Define the point and the line equation with a generic slope
def point : ℝ × ℝ := (-1, 2)

def line (a : ℝ) := a * (point.fst) + (point.snd) - 4 = 0

-- The main theorem statement
theorem slope_of_line (a : ℝ) (h : line a) : ∃ m : ℝ, m = 2 :=
by
  -- The slope of the line derived from the equation and condition
  sorry

end slope_of_line_l600_600636


namespace definite_integral_eval_l600_600925

noncomputable def evaluate_definite_integral : ℝ :=
  ∫ x in 0..1, (Real.exp x + x)

theorem definite_integral_eval :
  evaluate_definite_integral = Real.exp 1 - 1 / 2 :=
by
  sorry

end definite_integral_eval_l600_600925


namespace completing_the_square_l600_600798

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l600_600798


namespace problem_proof_l600_600958

noncomputable def a : ℝ := (sqrt 5 + sqrt 3) / (sqrt 5 - sqrt 3)
noncomputable def b : ℝ := (sqrt 5 - sqrt 3) / (sqrt 5 + sqrt 3)

theorem problem_proof : (b / a) + (a / b) = 62 := 
by
  sorry

end problem_proof_l600_600958


namespace equation_linear_implies_k_equals_neg2_l600_600210

theorem equation_linear_implies_k_equals_neg2 (k : ℤ) (x : ℝ) :
  (k - 2) * x^(abs k - 1) = k + 1 → abs k - 1 = 1 ∧ k - 2 ≠ 0 → k = -2 :=
by
  sorry

end equation_linear_implies_k_equals_neg2_l600_600210


namespace probability_of_drawing_two_red_shoes_l600_600368

/-- Given there are 7 red shoes and 3 green shoes, 
    and a total of 10 shoes, if two shoes are drawn randomly,
    prove that the probability of drawing both shoes as red is 7/15. -/
theorem probability_of_drawing_two_red_shoes :
  let total_shoes := 10
  let red_shoes := 7
  let green_shoes := 3
  let total_ways := Nat.choose total_shoes 2
  let red_ways := Nat.choose red_shoes 2
  (1 : ℚ) * red_ways / total_ways = 7 / 15  := by
  sorry

end probability_of_drawing_two_red_shoes_l600_600368


namespace factorize_x_cubed_minus_9x_l600_600934

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l600_600934


namespace factorize_x_cube_minus_9x_l600_600928

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l600_600928


namespace completing_the_square_l600_600811

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l600_600811


namespace minimum_zeros_l600_600771

theorem minimum_zeros (n : ℕ) (a : Fin n → ℤ) (h : n = 2011)
  (H : ∀ i j k : Fin n, a i + a j + a k ∈ Set.range a) : 
  ∃ (num_zeros : ℕ), num_zeros ≥ 2009 ∧ (∃ f : Fin (num_zeros) → Fin n, ∀ i : Fin (num_zeros), a (f i) = 0) :=
sorry

end minimum_zeros_l600_600771


namespace complex_number_second_quadrant_l600_600992

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i * (1 + i)

-- Define a predicate to determine if a complex number is in the second quadrant
def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The main statement
theorem complex_number_second_quadrant : is_second_quadrant z := by
  sorry

end complex_number_second_quadrant_l600_600992


namespace train_speed_l600_600398

theorem train_speed (distance time : ℕ) (h1 : distance = 180) (h2 : time = 9) : distance / time = 20 := by
  sorry

end train_speed_l600_600398


namespace line_AB_eq_ellipse_eq_l600_600972

-- Given definitions
def ellipse (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) := ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)
def eccentricity (e : ℝ) := e = Real.sqrt 6 / 3
def midpoint (A B N : ℝ × ℝ) := N = ((fst A + fst B) / 2, (snd A + snd B) / 2)

-- Statements to prove

-- 1. Proving the equation of line AB
theorem line_AB_eq (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (e : ℝ) (h₃ : eccentricity e) 
  (A B : ℝ × ℝ) (N : ℝ × ℝ) (h₄ : midpoint A B N) (hN : N = (3,1)) :
  ∀ (x y : ℝ), y = -x + 4 := 
sorry

-- 2. Proving the equation of the ellipse given the tangency condition
theorem ellipse_eq (b : ℝ) (h₁ : b > 0) (A B : ℝ × ℝ) :
  ∀ (x y : ℝ), b = Real.sqrt 8 / 3 → ellipse 24 (8^(1/2)) 4 (sqrt 8) x y :=
sorry

end line_AB_eq_ellipse_eq_l600_600972


namespace probability_of_conditions_l600_600381

def spinner1 := {2, 3, 4}
def spinner2 := {1, 3, 5}

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def conditions_satisfied (a b : ℕ) : Prop :=
  is_odd (a + b) ∧ is_even (a * b)

def valid_pairs : ℕ :=
  (spinner1.product spinner2).count (λ (p : ℕ × ℕ), conditions_satisfied p.1 p.2)

def total_pairs : ℕ :=
  (spinner1.product spinner2).card

theorem probability_of_conditions :
  (valid_pairs : ℚ) / (total_pairs : ℚ) = 5 / 9 :=
by
  sorry

end probability_of_conditions_l600_600381


namespace determine_Tn_unique_l600_600579

theorem determine_Tn_unique (a d : ℤ) (S T : ℕ → ℤ) (hS : ∀ n, S n = n * (2 * a + (n - 1) * d) / 2)
  (hT : ∀ n, T n = ∑ k in finset.range (n+1), S k)
  (hS2023 : S 2023 = 2023 * (a + 1011 * d)) :
  ∃ n, n = 3034 :=
by
  use 3034
  sorry

end determine_Tn_unique_l600_600579


namespace fraction_of_girls_at_outing_l600_600289

theorem fraction_of_girls_at_outing 
  (Orwell_total : ℕ)
  (Orwell_boys_to_girls_ratio : ℕ × ℕ)
  (Huxley_total : ℕ)
  (Huxley_girls_to_boys_ratio : ℕ × ℕ)
  (Orwell_total = 300)
  (Orwell_boys_to_girls_ratio = (3, 2))
  (Huxley_total = 240)
  (Huxley_girls_to_boys_ratio = (3, 2)) :
  (120 + 144) / (300 + 240) = 22 / 45 :=
by
  sorry

end fraction_of_girls_at_outing_l600_600289


namespace sin_A_plus_C_l600_600681

variables {A B C : ℝ}
variables {a b c : ℝ}

-- (Conditions)
axiom acute_triangle : ∀ (A B C : ℝ), 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π
axiom sides_arithmetic_sequence : ∀ (a b c : ℝ), a + c = 2 * b
axiom sin_A_minus_C : ∀ (A C : ℝ), sin(A - C) = √3 / 2

-- (Question and Correct Answer)
theorem sin_A_plus_C : sin(A + C) = √39 / 8 :=
by
  have h1 : acute_triangle A B C := sorry,
  have h2 : sides_arithmetic_sequence a b c := sorry,
  have h3 : sin_A_minus_C A C := sorry,
  sorry

end sin_A_plus_C_l600_600681


namespace smallest_angle_of_triangle_l600_600763

theorem smallest_angle_of_triangle (x : ℕ) 
  (h1 : ∑ angles in {x, 3 * x, 5 * x}, angles = 180)
  (h2 : (3 * x) = middle_angle)
  (h3 : (5 * x) = largest_angle) 
  : x = 20 := 
by
  sorry

end smallest_angle_of_triangle_l600_600763


namespace age_sum_is_ninety_l600_600037

theorem age_sum_is_ninety (a b c : ℕ)
  (h1 : a = 20 + b + c)
  (h2 : a^2 = 1800 + (b + c)^2) :
  a + b + c = 90 := 
sorry

end age_sum_is_ninety_l600_600037


namespace range_x_for_inequality_l600_600607

variables {f : ℝ → ℝ}

-- Definition: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition: f is monotonically increasing in [0, +∞)
def is_monotone_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- The theorem holding the proof problem
theorem range_x_for_inequality (h_odd : is_odd_function f) (h_mono : is_monotone_nonneg f) : 
  ∀ x, f (2x - 1) > f 1 ↔ x > 1 :=
by
sor

end range_x_for_inequality_l600_600607


namespace tank_capacity_75_l600_600870

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end tank_capacity_75_l600_600870


namespace b_general_formula_sum_first_20_terms_is_300_l600_600141

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a(n) + 1 else a(n) + 2

-- Define the sequence b_n as bₙ = a₂ₙ
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof goal for part (1)
theorem b_general_formula (n : ℕ) : b n = 3 * n - 1 := sorry

-- Sum of the first 20 terms of the sequence a_n
def sum_first_20_terms : ℕ :=
  (List.range (20)).sum (λ n, a n)

-- Proof goal for part (2)
theorem sum_first_20_terms_is_300 : sum_first_20_terms = 300 := sorry

end b_general_formula_sum_first_20_terms_is_300_l600_600141


namespace arithmetic_seq_value_l600_600226

variable {α : Type*} [LinearOrderedRing α] 
variables (a1 d : α) (a : ℕ → α) 

-- Arithmetic sequence definition
def an_arithmetic (a : ℕ → α) (a1 d : α) := ∀ n : ℕ, a n = a1 + n * d

-- Given condition
variable (h_condition : a 4 + a 6 + a 8 = 12)

-- Theorem statement
theorem arithmetic_seq_value (h_arith : an_arithmetic a a1 d) : 
  a 8 - (1 / 2) * a 10 = 2 := 
sorry

end arithmetic_seq_value_l600_600226


namespace probability_of_green_is_correct_l600_600769

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of green balls
def green_balls : ℕ := 7

-- Define the number of red balls
def red_balls : ℕ := 3

-- The hypothesis that there are 10 balls in total, 7 green balls, and 3 red balls
axiom total_eq_sum : total_balls = green_balls + red_balls

-- Define the probability of drawing a green ball
def probability_of_green : ℚ := green_balls / total_balls.to_rat -- Convert total_balls to a rational number

-- The theorem that states what we want to prove
theorem probability_of_green_is_correct : probability_of_green = 7 / 10 :=
by
  sorry

end probability_of_green_is_correct_l600_600769


namespace count_valid_natural_numbers_l600_600088

theorem count_valid_natural_numbers (n : ℕ) (h : n = 454500) :
  (finset.filter (λ k : ℕ, (k * (k - 1)) % 505 = 0) 
  (finset.range (n + 1))).card = 3600 :=
by
  sorry

end count_valid_natural_numbers_l600_600088


namespace sum_possible_a1_l600_600971

theorem sum_possible_a1 {k a_1 a_2 a_3 a_4 a_5 : ℤ} 
  (h : ∀ n : ℕ, n > 0 → a_{n+1} = k * a_n + 2 * k - 2)
  (conds : {a_2, a_3, a_4, a_5} ⊆ { -272, -32, -2, 8, 88, 888 })
  (hk : k ≠ 0 ∧ k ≠ 1) :
  ∃ a1_sum : ℚ, a1_sum = 2402 / 3 := 
begin
  let s := {a_1 | a_1 = -2 ∨ a_1 = - (16 / 3) ∨ a_1 = 808},
  have h_s : s.sum = 2402 / 3, from sorry,
  exact ⟨2402 / 3, h_s⟩
end

end sum_possible_a1_l600_600971


namespace bus_driver_total_hours_l600_600857

def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_compensation : ℝ := 976
def max_regular_hours : ℝ := 40

theorem bus_driver_total_hours :
  ∃ (hours_worked : ℝ), 
  (hours_worked = max_regular_hours + (total_compensation - (regular_rate * max_regular_hours)) / overtime_rate) ∧
  hours_worked = 52 :=
by
  sorry

end bus_driver_total_hours_l600_600857


namespace chessboard_piece_removal_l600_600786

theorem chessboard_piece_removal : 
  ∀ (chessboard : fin 8 → fin 8 → bool), 
    (∀ row, (finset.univ.filter (λ column, chessboard row column)).card = 4) → 
    (∀ column, (finset.univ.filter (λ row, chessboard row column)).card = 4) →
    ∃ rem_pieces : finset (fin 8 × fin 8), 
      rem_pieces.card = 24 ∧ 
      (∀ row, (finset.univ.filter (λ column, chessboard row column ∧ (row, column) ∉ rem_pieces)).card = 1) ∧ 
      (∀ column, (finset.univ.filter (λ row, chessboard row column ∧ (row, column) ∉ rem_pieces)).card = 1) :=
sorry

end chessboard_piece_removal_l600_600786


namespace min_total_weight_l600_600842

theorem min_total_weight (crates: Nat) (weight_per_crate: Nat) (h1: crates = 6) (h2: weight_per_crate ≥ 120): 
  crates * weight_per_crate ≥ 720 :=
by
  sorry

end min_total_weight_l600_600842


namespace triangle_centroid_equiv_l600_600263

open Set Function

variable (A B C D E F G : Type) [has_scalar ℝ A] [AddGroup A] [Module ℝ A]

-- Points D, E, F are trisectors of segments BC, CA, AB respectively
def is_trisection_point (B C D : A) : Prop := (3 : ℝ) • (D - B) = C - B ∧ (3 : ℝ) • (C - D) = C - D

-- The centroid of triangle ABC
def centroid (A B C : A) : A := (A + B + C) / 3

theorem triangle_centroid_equiv (A B C D E F G : A) 
  (hD : is_trisection_point B C D)
  (hE : is_trisection_point C A E)
  (hF : is_trisection_point A B F) : 
  centroid A B C = centroid D E F :=
sorry

end triangle_centroid_equiv_l600_600263


namespace curve_length_on_cube_l600_600220

noncomputable def curve_length_of_points (A : Point) (l : ℝ) (d : ℝ) : ℝ :=
let S := sphere A d
in let F := faces_of_cube l
in arc_length_of_intersection S F

theorem curve_length_on_cube :
  let A : Point := ⟨0, 0, 0⟩
  let l : ℝ := 1
  let d : ℝ := (2 * real.sqrt 3) / 3
  in curve_length_of_points A l d = (5 * real.sqrt 3 * real.pi) / 6 :=
by sorry

end curve_length_on_cube_l600_600220


namespace probability_of_pairing_margo_irma_l600_600219

noncomputable theory

section
-- Constants
def total_students : ℕ := 40
def forbidden_pairs_count : ℕ := 3  -- Each of the specific 5 students has 3 classmates they prefer not to be paired with
def possible_pairs : ℕ := total_students - 1 - forbidden_pairs_count  -- Total choices minus one less Margo itself and three forbidden students

-- Condition Definitions
def is_in_list (student : ℕ) (forbidden_list : list ℕ) : Prop := student ∈ forbidden_list

-- Theorem stating the probability calculation
theorem probability_of_pairing_margo_irma 
  (margo irma : ℕ) 
  (forbidden_list_margo : list ℕ) 
  (h1 : is_in_list irma forbidden_list_margo = false)
  (h2 : forbidden_list_margo.length = forbidden_pairs_count) :
  (1 : ℝ) / possible_pairs = (1 : ℝ) / 36 :=
by
  -- The proof is omitted, as instructed
  sorry
end

end probability_of_pairing_margo_irma_l600_600219


namespace b_n_formula_sum_first_20_terms_l600_600132

-- Definition of sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a n + 1 else a n + 2

-- Definition of sequence b_n as a 2n-th term of a_n
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof problem 1: General formula for b_n
theorem b_n_formula (n : ℕ) : b n = 3 * n - 1 :=
sorry

-- Sum of the first 20 terms of the sequence a_n
theorem sum_first_20_terms : (Finset.range 20).sum a = 300 :=
sorry

end b_n_formula_sum_first_20_terms_l600_600132


namespace range_of_x_l600_600706

def g (x : ℝ) : ℝ :=
if x < 0 then ln (1 - x) else ln (1 + x)

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 else g x

theorem range_of_x (x : ℝ) (h : f (2 - x^2) > f x) : -2 < x ∧ x < 1 :=
by
  sorry

end range_of_x_l600_600706


namespace intersection_A_B_solution_inequalities_l600_600988

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def C : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = C :=
by
  sorry

theorem solution_inequalities (x : ℝ) :
  (2 * x^2 + x - 1 > 0) ↔ (x < -1 ∨ x > 1/2) :=
by
  sorry

end intersection_A_B_solution_inequalities_l600_600988


namespace right_triangle_construction_l600_600063

theorem right_triangle_construction (a_1 a_2 : ℝ) :
  ∃ (A B C D : Point),
  (∠ACB = 90°) ∧
  (is_angle_bisector A C B A C) ∧
  (dist C D = a_1) ∧ 
  (dist D B = a_2) := 
sorry

end right_triangle_construction_l600_600063


namespace angle_bisector_circles_l600_600838

theorem angle_bisector_circles {A B C E F D : Point} :
  BD_is_the_angle_bisector_of_ABC A B C D ∧ 
  circumcircle_intersecting_BDC_at_E A B C D E ∧ 
  circumcircle_intersecting_ABD_at_F B C D F →
  (dist A E = dist C F) := 
by
  sorry

end angle_bisector_circles_l600_600838


namespace salary_raise_l600_600760

variable (S P : ℝ)

theorem salary_raise : (S : ℝ) > 0 → P = 100 * (1 - 0.85) / 0.85 →
  S = 0.85 * S + (P / 100) * (0.85 * S) ∧ P ≈ 17.65 := by
  sorry

end salary_raise_l600_600760


namespace completing_the_square_l600_600814

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l600_600814


namespace nth_monomial_sequence_l600_600454

theorem nth_monomial_sequence (a : ℝ) (n : ℕ) : (nth_monomial : ℕ → ℝ) n = real.sqrt n * a ^ n :=
by
  assume a n
  sorry

end nth_monomial_sequence_l600_600454


namespace parallel_transitivity_l600_600619

variable (Line Plane : Type)
variable (m n : Line)
variable (α : Plane)

-- Definitions for parallelism
variable (parallel : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Conditions
variable (m_n_parallel : parallel m n)
variable (m_alpha_parallel : parallelLinePlane m α)
variable (n_outside_alpha : ¬ parallelLinePlane n α)

-- Proposition to be proved
theorem parallel_transitivity (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : parallelLinePlane m α) 
  : parallelLinePlane n α :=
sorry 

end parallel_transitivity_l600_600619


namespace quadratic_inequality_solution_l600_600962

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 5 * x + 6 < 0} = {x : ℝ | 2 < x ∧ x < 3} :=
begin
  sorry
end

end quadratic_inequality_solution_l600_600962


namespace power_function_even_and_decreasing_l600_600625

theorem power_function_even_and_decreasing (m : ℤ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_decreasing : ∀ x y : ℝ, 0 < x → x < y → f y < f x) :
  m = 1 :=
by
  have h1 : m^2 - 2 * m - 3 < 0 := sorry
  have h2 : Even (m^2 - 2 * m - 3) := sorry
  sorry

end power_function_even_and_decreasing_l600_600625


namespace perpendicular_lines_necessary_but_not_sufficient_l600_600157

-- Definitions
variable {α : Type} [plane α]
variables {a b : line α} (l : line)

-- Conditions
def lines_in_same_plane (a b : line α) (α : plane α) : Prop :=
  lies_in α a ∧ lies_in α b ∧ a ≠ b

def line_outside_plane (l : line) (α : plane α) : Prop :=
  ¬ lies_in α l

-- Proof Problem Statement
theorem perpendicular_lines_necessary_but_not_sufficient (α : plane α) (a b : line α) (l : line) :
  (lines_in_same_plane a b α) → (line_outside_plane l α) → 
  (l ⊥ a ∧ l ⊥ b) ↔ ((∀ (m : line α), lies_in α m → l ⊥ m) ∧ ¬ (∀ (m : line α), lies_in α m → l ⊥ m)) :=
by
  intros h₁ h₂,
  sorry

end perpendicular_lines_necessary_but_not_sufficient_l600_600157


namespace identity_proof_l600_600717

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a + b)^4 = 2 * (a^2 + a * b + b^2)^2 := 
sorry

end identity_proof_l600_600717


namespace burn_out_odds_type_A_burn_out_odds_type_B_burn_out_odds_type_C_l600_600112

theorem burn_out_odds_type_A :
  let initial_odds_A := 1 / 3
      rate_decrease_A := 1 / 2
      odds_six_months_to_one_year_A := initial_odds_A * rate_decrease_A
  in odds_six_months_to_one_year_A = 1 / 6 := by
  sorry

theorem burn_out_odds_type_B :
  let initial_odds_B := 1 / 4
      rate_decrease_B := 1 / 3
      odds_six_months_to_one_year_B := initial_odds_B * rate_decrease_B
  in odds_six_months_to_one_year_B = 1 / 12 := by
  sorry

theorem burn_out_odds_type_C :
  let initial_odds_C := 1 / 5
      rate_decrease_C := 1 / 4
      odds_six_months_to_one_year_C := initial_odds_C * rate_decrease_C
  in odds_six_months_to_one_year_C = 1 / 20 := by
  sorry

end burn_out_odds_type_A_burn_out_odds_type_B_burn_out_odds_type_C_l600_600112


namespace sequence_condition_general_formula_sum_of_first_n_terms_l600_600970

-- Define the sequence a_n according to the given condition
def a (n : ℕ) (hn : n > 0) : ℝ := 2 / (2 * n - 1)

-- The given condition for the sequence
theorem sequence_condition (n : ℕ) (hn : n > 0) :
  (∑ i in Finset.range (n + 1), (2 * i + 1) * a (i+1) (by linarith)) = 2 * (n + 1) :=
sorry

-- Prove the general formula for a_n
theorem general_formula (n : ℕ) (hn : n > 0) :
  a n hn = 2 / (2 * n - 1) :=
sorry

-- Prove the sum of the first n terms of {a_n / (2n+1)}
theorem sum_of_first_n_terms (n : ℕ) (hn : n > 0) :
  (∑ i in Finset.range n, a (i + 1) (by linarith) / (2 * (i + 1) + 1)) = 2 * n / (2 * n + 1) :=
sorry

end sequence_condition_general_formula_sum_of_first_n_terms_l600_600970


namespace area_ratio_dodecagon_l600_600222

theorem area_ratio_dodecagon (P Q R S : Point) (dodecagon : Polygon) 
  (h_regular : dodecagon.is_regular 12) 
  (h_midpoints : P.midpoint (dodecagon.side 2) ∧ Q.midpoint (dodecagon.side 7) 
    ∧ R.midpoint (dodecagon.side 0) ∧ S.midpoint (dodecagon.side 6)) :
  area (Polygon.mk [dodecagon.vertex 0, dodecagon.vertex 1, dodecagon.vertex 2, P]) / 
  area (Polygon.mk [dodecagon.vertex 4, dodecagon.vertex 5, dodecagon.vertex 6, dodecagon.vertex 7, P, Q]) = 2 / 5 :=
sorry

end area_ratio_dodecagon_l600_600222


namespace abs_eq_cos_has_three_roots_l600_600734

theorem abs_eq_cos_has_three_roots :
  ∃ x1 : ℝ, |0| = cos 0 ∧ |x1| = cos x1 ∧ |-x1| = cos (-x1) ∧ 
  x1 ≠ 0 ∧ 
  set.countable {x : ℝ | |x| = cos x} :=
sorry

end abs_eq_cos_has_three_roots_l600_600734


namespace outermost_ring_circles_9x9_l600_600325

-- Note: We're not defining the exact pattern here, as it's assumed given from the problem context.
-- Also, we'll assume the pattern is such that the entire outermost ring in a 9x9 matrix is circles,
-- as implied by the correct answer.

def outermost_circle_count (n : ℕ) : ℕ := n * n

theorem outermost_ring_circles_9x9 :
  ∀ (grid_4x4_pattern : ℕ → ℕ),
  grid_4x4_pattern 4 = outermost_circle_count 4 →
  grid_4x4_pattern 9 = outermost_circle_count 9 :=
by
  intros grid_4x4_pattern h
  rw h
  sorry

end outermost_ring_circles_9x9_l600_600325


namespace solve_z_plus_inv_y_l600_600315

theorem solve_z_plus_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 :=
sorry

end solve_z_plus_inv_y_l600_600315


namespace max_sqrt_expr_l600_600270

noncomputable def max_value_sqrt_expr (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 8) : Prop :=
  (Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 9)

/-- Prove the maximum value of the given expression under specified conditions -/
theorem max_sqrt_expr (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 8) :
  max_value_sqrt_expr a b c ha hb hc h_sum :=
begin
  sorry
end

end max_sqrt_expr_l600_600270


namespace angle_bisector_AD_l600_600376

namespace Geometry

-- Define the basic structures such as points and circles
structure Point where
  x : ℝ 
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

open Point Circle

-- Define the problem conditions as hypotheses in Lean
variable (A B C D : Point)
variable (O O' : Circle)
variable (h1 : O.center = A)
variable (h2 : O.radius = (A - B).norm)
variable (h3 : O'.center = A)
variable (h4 : tangent B C D O O') -- BC chord is tangent to the smaller circle at D

-- Define the conclusion to prove in Lean
theorem angle_bisector_AD 
  (h1 : O.center = A)
  (h2 : O.radius = (A - B).norm)
  (h3 : O'.center = A)
  (h4 : tangent B C D O O')
  : is_angle_bisector A D (triangle ABC) :=
sorry

end Geometry

end angle_bisector_AD_l600_600376


namespace father_age_is_32_l600_600776

noncomputable def father_age (D F : ℕ) : Prop :=
  F = 4 * D ∧ (F + 5) + (D + 5) = 50

theorem father_age_is_32 (D F : ℕ) (h : father_age D F) : F = 32 :=
by
  sorry

end father_age_is_32_l600_600776


namespace correct_statements_C_D_l600_600833

-- Definitions for statement C
def transform_log2_graph (y : ℕ → ℝ) : Prop := 
  ∀ x, y x = log 2 (sqrt (x - 1))

def log2_graph_transformation (y log2 : ℕ → ℝ) : Prop := 
  ∀ x, y x = (1 / 2) * log2 x ∧ y (x + 1) = log 2 (sqrt (x - 1))

-- Definitions for statement D
def satisfy_eq1 (x : ℝ) : Prop := x + log x = 2
def satisfy_eq2 (x : ℝ) : Prop := log (1 - x) - x = 1

def sum_is_one (x₁ x₂ : ℝ) : Prop :=
  satisfy_eq1 x₁ ∧ satisfy_eq2 x₂ → x₁ + x₂ = 1

-- The final proof problem
theorem correct_statements_C_D (y log2 : ℕ → ℝ) (x₁ x₂ : ℝ) : 
  transform_log2_graph y ∧ log2_graph_transformation y log2 ∧ satisfy_eq1 x₁ ∧ satisfy_eq2 x₂ → sum_is_one x₁ x₂ :=
sorry

end correct_statements_C_D_l600_600833


namespace radius_of_circle_l600_600745

theorem radius_of_circle (r : ℝ) : 3 * 2 * Real.pi * r = Real.pi * r^2 → r = 6 :=
by {
  intro h,
  have h1 : 6 * Real.pi * r = Real.pi * r^2 := by rw [←mul_assoc, ←h],
  have h2 : 6 * r = r^2 := by rw [←mul_div_cancel_left 'Real.pi, h1],
  have h3 : r^2 - 6 * r = 0 := by ring,
  have h4 : r * (r - 6) = 0 := by rw h3,
  cases eq_zero_or_eq_zero_of_mul_eq_zero h4 with h5 h6,
  { exact h5, },
  { exact h6, }
} sorry

end radius_of_circle_l600_600745


namespace Diane_bakes_160_gingerbreads_l600_600550

-- Definitions
def trays1Count : Nat := 4
def gingerbreads1PerTray : Nat := 25
def trays2Count : Nat := 3
def gingerbreads2PerTray : Nat := 20

def totalGingerbreads : Nat :=
  (trays1Count * gingerbreads1PerTray) + (trays2Count * gingerbreads2PerTray)

-- Problem statement
theorem Diane_bakes_160_gingerbreads :
  totalGingerbreads = 160 := by
  sorry

end Diane_bakes_160_gingerbreads_l600_600550


namespace integral_equiv_pi_over_4_l600_600409

theorem integral_equiv_pi_over_4 : 
  ∫ x in 0..(π / 2), 
    (cos x ^ 4 + sin x * cos x ^ 3 + sin x ^ 2 * cos x ^ 2 + sin x ^ 3 * cos x) / 
    (sin x ^ 4 + cos x ^ 4 + 2 * sin x * cos x ^ 3 + 2 * sin x ^ 2 * cos x ^ 2 + 2 * sin x ^ 3 * cos x) 
  = π / 4 := 
by 
  sorry

end integral_equiv_pi_over_4_l600_600409


namespace intersection_points_lie_on_ellipse_l600_600113

theorem intersection_points_lie_on_ellipse (s : ℝ) : 
  ∃ (x y : ℝ), (2 * s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) ∧ (x^2 / 16 + y^2 / 9 = 1) :=
sorry

end intersection_points_lie_on_ellipse_l600_600113


namespace all_numbers_equal_l600_600369

theorem all_numbers_equal (x : Fin 101 → ℝ) 
  (h : ∀ i : Fin 100, x i.val^3 + x ⟨(i.val + 1) % 101, sorry⟩ = (x ⟨(i.val + 1) % 101, sorry⟩)^3 + x ⟨(i.val + 2) % 101, sorry⟩) :
  ∀ i j : Fin 101, x i = x j := 
by 
  sorry

end all_numbers_equal_l600_600369


namespace Walter_gets_49_bananas_l600_600245

variable (Jefferson_bananas : ℕ) (Walter_bananas : ℕ) (total_bananas : ℕ) (shared_bananas : ℕ)

def problem_conditions : Prop :=
  Jefferson_bananas = 56 ∧ Walter_bananas = Jefferson_bananas - (Jefferson_bananas / 4)

theorem Walter_gets_49_bananas (h : problem_conditions) : 
  let combined_bananas := Jefferson_bananas + Walter_bananas in
  let shared_bananas := combined_bananas / 2 in
  shared_bananas = 49 :=
by
  sorry

end Walter_gets_49_bananas_l600_600245


namespace bucket_fill_time_l600_600284

theorem bucket_fill_time :
  ∀ (a b c d : ℝ), 
  (a = 16) →  
  (b = 12) →  
  (c = 8) →  
  (d = 6) →  
  (1 / a + 1 / b + 1 / c - 1 / d = 5 / 48) →  
  (48 / 5 = 9.6) →  
  (1 / (5 / 48) = 48 / 5) := 
by 
  intros a b c d ha hb hc hd hnet htime
  rw [htime]
  sorry

end bucket_fill_time_l600_600284


namespace number_of_valid_programs_l600_600457

-- Define the set of courses
inductive Course
| English | Algebra | Geometry | History | Art | Latin | Science | Music

-- Define the conditions as the set of rules
def isValidProgram (s : Finset Course) : Prop :=
  Course.English ∈ s ∧ Course.History ∈ s ∧ 
  (Course.Algebra ∈ s ∨ Course.Geometry ∈ s) ∧ s.card = 5

-- The finite set of all courses
def allCourses : Finset Course := 
  Finset.ofList [Course.English, Course.Algebra, Course.Geometry, Course.History, Course.Art, Course.Latin, Course.Science, Course.Music]

-- Define the proof statement
theorem number_of_valid_programs : 
  (Finset.filter isValidProgram (Finset.powersetLen 5 allCourses)).card = 16 :=
  sorry

end number_of_valid_programs_l600_600457


namespace area_of_square_l600_600335

def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
def radius_of_circle (s : ℝ) : ℝ := s
def area_of_rectangle (l : ℝ) (b : ℝ) : ℝ := l * b

theorem area_of_square (s : ℝ) (b : ℝ) (a_rect : ℝ) :
  (length_of_rectangle (radius_of_circle s) = (2 / 5) * s) → 
  (radius_of_circle s = s) → 
  (b = 10) →
  (a_rect = 140) →
  s^2 = 1225 :=
by
  sorry

end area_of_square_l600_600335


namespace arnold_plates_count_l600_600471

def arnold_barbell := 45
def mistaken_weight := 600
def actual_weight := 470
def weight_difference_per_plate := 10

theorem arnold_plates_count : 
  ∃ n : ℕ, mistaken_weight - actual_weight = n * weight_difference_per_plate ∧ n = 13 := 
sorry

end arnold_plates_count_l600_600471


namespace triangle_largest_angle_l600_600663

theorem triangle_largest_angle (A B C : ℚ) (sinA sinB sinC : ℚ) 
(h_ratio : sinA / sinB = 3 / 5)
(h_ratio2 : sinB / sinC = 5 / 7)
(h_sum : A + B + C = 180) : C = 120 := 
sorry

end triangle_largest_angle_l600_600663


namespace angle_between_lines_proof_l600_600976

noncomputable def l1_angle : ℝ := 60 * (Real.pi / 180) -- Convert 60 degrees to radians
noncomputable def l2_angle : ℝ := 120 * (Real.pi / 180) -- Convert 120 degrees to radians
noncomputable def angle_between_lines_l1_l2 : ℝ := l2_angle - l1_angle

theorem angle_between_lines_proof : 
  let l1 := (√3, -1, 2) -- coefficients of x, y, constant term for l1
  let l2 := (3, √3, -5) -- coefficients of x, y, constant term for l2
  angle_between_lines_l1_l2 = (60 * (Real.pi / 180)) :=   -- converting 60 degrees to radians
by
  sorry

end angle_between_lines_proof_l600_600976


namespace time_for_train_to_pass_l600_600844

-- Defining the conditions
def length_first_train: ℝ := 280 -- in meters
def length_second_train: ℝ := 360 -- in meters
def speed_first_train: ℝ := 72 -- in km/h
def speed_second_train: ℝ := 36 -- in km/h
def kmph_to_mps (speed: ℝ): ℝ := speed * (5 / 18) -- Conversion function from km/h to m/s

-- Defining the relative speed in m/s
def relative_speed: ℝ := kmph_to_mps (speed_first_train - speed_second_train)

-- Total distance to be covered
def total_distance: ℝ := length_first_train + length_second_train

-- Time calculation function
def time_to_pass (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- The theorem to prove
theorem time_for_train_to_pass : time_to_pass total_distance relative_speed = 64 := by
  -- The proof will go here
  sorry

end time_for_train_to_pass_l600_600844


namespace solve_equation_l600_600850

theorem solve_equation :
  ∀ x : ℝ, 81 * (1 - x) ^ 2 = 64 ↔ x = 1 / 9 ∨ x = 17 / 9 :=
by
  sorry

end solve_equation_l600_600850


namespace aprils_plant_arrangement_l600_600047

theorem aprils_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 3
  let total_units := basil_plants + 1
  
  (fact total_units * fact tomato_plants = 4320) :=
by
  unfold basil_plants
  unfold tomato_plants
  unfold total_units
  apply eq.refl
  sorry

end aprils_plant_arrangement_l600_600047


namespace polygon_RSP_PQRSTU_condition_l600_600321

theorem polygon_RSP_PQRSTU_condition (area_PQRSTU PQ QR TU: ℕ) (h1: area_PQRSTU = 68) (h2: PQ = 10) (h3: QR = 7) (h4: TU = 6): 
  RS + ST = 3 := 
by 
  sorry

end polygon_RSP_PQRSTU_condition_l600_600321


namespace waiter_earned_in_tips_l600_600051

def waiter_customers := 7
def customers_didnt_tip := 5
def tip_per_customer := 3
def customers_tipped := waiter_customers - customers_didnt_tip
def total_earnings := customers_tipped * tip_per_customer

theorem waiter_earned_in_tips : total_earnings = 6 :=
by
  sorry

end waiter_earned_in_tips_l600_600051


namespace distance_A_to_l_l600_600620

noncomputable def distance_from_point_to_line (r θ : ℝ) (rho : ℝ → ℝ) (l : ℝ → Prop) : ℝ :=
  let x := r * (Real.cos θ)
  let y := r * (Real.sin θ)
  have h_l : l (x, y) := by
    sorry -- Converting the polar equation to rectangular coordinates.
  (abs (sqrt 3 * x + y - 1)) / (Real.sqrt ((sqrt 3)^2 + 1^2))

theorem distance_A_to_l:
  let A := (2, (Real.pi / 6))
  let l := λ p : ℝ × ℝ, (sqrt 3) * p.1 + p.2 = 1
  distance_from_point_to_line 2 (Real.pi / 6) (ρ θ := ρ * Real.sin (θ + Real.pi / 3)) l = 3 / 2 :=
sorry

end distance_A_to_l_l600_600620


namespace other_factor_of_936_mul_w_l600_600861

theorem other_factor_of_936_mul_w (w : ℕ) (h_w_pos : 0 < w)
  (h_factors_936w : ∃ k, 936 * w = k * (3^3)) 
  (h_factors_936w_2 : ∃ m, 936 * w = m * (10^2))
  (h_w : w = 120):
  ∃ n, n = 45 :=
by
  sorry

end other_factor_of_936_mul_w_l600_600861


namespace completing_the_square_l600_600819

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l600_600819


namespace count_4_digit_mountain_numbers_l600_600544

def is_mountain_number (n : ℕ) : Prop :=
  let d1 := (n / 1000) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  (1000 ≤ n ∧ n < 10000) ∧
  (d3 > d1 ∧ d3 > d2 ∧ d3 > d4)

theorem count_4_digit_mountain_numbers : 
  {n : ℕ | is_mountain_number n}.card = 330 :=
sorry

end count_4_digit_mountain_numbers_l600_600544


namespace minimum_score_for_fourth_term_l600_600781

variable (score1 score2 score3 score4 : ℕ)
variable (avg_required : ℕ)

theorem minimum_score_for_fourth_term :
  score1 = 80 →
  score2 = 78 →
  score3 = 76 →
  avg_required = 85 →
  4 * avg_required - (score1 + score2 + score3) ≤ score4 :=
by
  sorry

end minimum_score_for_fourth_term_l600_600781


namespace correct_equation_l600_600002

theorem correct_equation (x : ℝ) (h1 : 2000 > 0) (h2 : x > 0) (h3 : x + 40 > 0) :
  (2000 / x) - (2000 / (x + 40)) = 3 :=
by
  sorry

end correct_equation_l600_600002


namespace coeff_x_105_l600_600567

def polynomial : Polynomial ℚ :=
  (x - 1) * (x^2 - 2) * (x^3 - 3) * (x^4 - 4) * (x^5 - 5) 
  * (x^6 - 6) * (x^7 - 7) * (x^8 - 8) * (x^9 - 9) * (x^10 - 10) 
  * (x^11 - 11) * (x^12 - 12) * (x^13 - 13) * (x^14 - 14) * (x^15 - 15)

theorem coeff_x_105 : polynomial.coeff 105 = -177 := 
by 
  sorry

end coeff_x_105_l600_600567


namespace calculate_expression_l600_600497

theorem calculate_expression :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2 = 6 * real.sqrt 2) :=
by
  sorry

end calculate_expression_l600_600497


namespace work_hours_to_pay_off_debt_l600_600238

theorem work_hours_to_pay_off_debt (initial_debt paid_amount hourly_rate remaining_debt work_hours : ℕ) 
  (h₁ : initial_debt = 100) 
  (h₂ : paid_amount = 40) 
  (h₃ : hourly_rate = 15) 
  (h₄ : remaining_debt = initial_debt - paid_amount) 
  (h₅ : work_hours = remaining_debt / hourly_rate) : 
  work_hours = 4 :=
by
  sorry

end work_hours_to_pay_off_debt_l600_600238


namespace union_of_A_and_B_l600_600978

noncomputable def A : Set ℝ := {x | ∃ y, y = Real.log (-x^2 - 3 * x + 4) ∧ -4 < x ∧ x < 1}
noncomputable def B : Set ℝ := {x | ∃ y, y = 2^(2 - x^2) ∧ 0 < y ∧ y ≤ 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -4 < x ∧ x ≤ 4} :=
by
  sorry

end union_of_A_and_B_l600_600978


namespace limit_example_l600_600480

theorem limit_example : (Real.limit (λ h : ℝ, (3 + h)^2 - 3^2) (0 : ℝ)) = 6 := by
  sorry

end limit_example_l600_600480


namespace number_of_valid_n_l600_600095

theorem number_of_valid_n : 
  {n : ℕ // n ≤ 2000 ∧ ∃ k : ℕ, 21 * n = k * k}.card = 9 := 
sorry

end number_of_valid_n_l600_600095


namespace car_length_l600_600924

variables (L E C : ℕ)

theorem car_length (h1 : 150 * E = L + 150 * C) (h2 : 30 * E = L - 30 * C) : L = 113 * E :=
by
  sorry

end car_length_l600_600924


namespace limit_goes_to_infinity_l600_600482

noncomputable def limit_expression (n : ℕ) : ℝ :=
  (n * real.root 6 n + real.root 3 (32 * n^10 + 1)) / ((n + real.root 4 n) * real.root 3 (n^3 - 1))

theorem limit_goes_to_infinity : 
  tendsto (λ n, limit_expression n) at_top (nhds ∞) :=
by sorry

end limit_goes_to_infinity_l600_600482


namespace derangement_formula_l600_600069

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n
  
def derangement (n : ℕ) : ℕ :=
  let sum := ∑ k in Finset.range (n + 1), ((-1)^k / factorial k)
  factorial n * sum

theorem derangement_formula (n : ℕ) : 
  derangement n = factorial n * ∑ k in Finset.range (n + 1), (-1) ^ k / factorial k := 
sorry

end derangement_formula_l600_600069


namespace capacity_of_each_bag_is_approximately_63_l600_600774

noncomputable def capacity_of_bag (total_sand : ℤ) (num_bags : ℤ) : ℤ :=
  Int.ceil (total_sand / num_bags)

theorem capacity_of_each_bag_is_approximately_63 :
  capacity_of_bag 757 12 = 63 :=
by
  sorry

end capacity_of_each_bag_is_approximately_63_l600_600774


namespace max_k_for_tangent_line_l600_600581

theorem max_k_for_tangent_line (k : ℝ) (h₁ : ∃ x y : ℝ, y = k * x - 2) :
  (∃ P : ℝ × ℝ, (P.2 = k * P.1 - 2) ∧ (x, y) ∈ ∂(metric.ball (0, 0) 1)) →
  k ≤ real.sqrt 3 :=
by sorry

end max_k_for_tangent_line_l600_600581


namespace corrected_sample_variance_l600_600085

/-- Given: 
  n = 10
  Sample points (x_i, n_i): 
    (102, 2),
    (104, 3),
    (108, 5)
  
  Prove:
  The corrected sample variance s_X^2 is 9.49.
-/
theorem corrected_sample_variance (x : Fin 3 → ℝ) (n_i : Fin 3 → ℝ) (sample_size : ℕ) :
  sample_size = 10 ∧
  (x 0 = 102 ∧ n_i 0 = 2) ∧
  (x 1 = 104 ∧ n_i 1 = 3) ∧
  (x 2 = 108 ∧ n_i 2 = 5) → 
  let u_i := fun i => x i - 104 in
  let s_u_sq := (∑ i, n_i i * u_i i ^ 2 - (∑ i, n_i i * u_i i) ^ 2 / sample_size) / (sample_size - 1) in
  s_u_sq = 9.49 :=
begin
  sorry
end

end corrected_sample_variance_l600_600085


namespace cube_diagonal_approx_l600_600390

-- Define the volumes
def rect_volume := 45 * 40 * 1
def square_area := 45 * 40
def square_volume := square_area * 1
def tri_base_area := (30 * 20) / 2
def tri_volume := tri_base_area * 50

-- Combined volume of all three shapes
def total_volume := rect_volume + square_volume + tri_volume

-- Side length of the cube
def cube_side := real.cbrt total_volume

-- Diagonal of the cube
def cube_diagonal := cube_side * real.sqrt 3

-- Prove that the diagonal is approximately 45.89 meters
theorem cube_diagonal_approx : abs (cube_diagonal - 45.89) < 0.01 := by
  sorry

end cube_diagonal_approx_l600_600390


namespace cost_of_bike_l600_600436

theorem cost_of_bike:
  ∃ (B : ℝ), let n := 20 in
  let b := 10 in
  let t := 5 in
  let e := 5 in
  let T := 1.5 * B in
  let E := 3 * T in
  let total_cost_single_gym := b * B + t * T + e * E in
  let total_cost_all_gyms := n * total_cost_single_gym in
  total_cost_all_gyms = 455000 → B = 700 := by
  sorry

end cost_of_bike_l600_600436


namespace largest_neg_integer_solution_l600_600572

theorem largest_neg_integer_solution 
  (x : ℤ) 
  (h : 34 * x + 6 ≡ 2 [ZMOD 20]) : 
  x = -6 := 
sorry

end largest_neg_integer_solution_l600_600572


namespace necessary_but_not_sufficient_l600_600049

variable (a b : ℝ)

def proposition_A : Prop := a > 0
def proposition_B : Prop := a > b ∧ a⁻¹ > b⁻¹

theorem necessary_but_not_sufficient : (proposition_B a b → proposition_A a) ∧ ¬(proposition_A a → proposition_B a b) :=
by
  sorry

end necessary_but_not_sufficient_l600_600049


namespace math_club_competition_scores_l600_600880

open List

theorem math_club_competition_scores:
  let scores := [92, 89, 96, 94, 98, 96, 95] in
  let sorted_scores := [89, 92, 94, 95, 96, 96, 98] in
  let mode := 96 in
  let median := 95 in
  mode_of_list scores = Some mode ∧ median_of_list scores = some median := by
  sorry

end math_club_competition_scores_l600_600880


namespace locus_of_points_l600_600642

-- Define the equilateral triangles and the point M
variables {α : Type} [real_field α]
variables (A B C A₁ B₁ C₁ M : α × α)

-- Define the condition that triangles are equilateral
def equilateral_triangle (A B C : α × α) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define the area of a triangle given its vertices
def triangle_area (A B C : α × α) : α :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Define the problem of finding the geometrical locus
theorem locus_of_points (h₁ : equilateral_triangle A B C) (h₂ : equilateral_triangle A₁ B₁ C₁) :
  (triangle_area M A B = triangle_area M A₁ B₁) →
  ∃ l : set (α × α), is_line l ∧ (M ∈ l ∨ is_circle (set_of (λ P, equidistant P A B C))) :=
begin
  sorry
end

end locus_of_points_l600_600642


namespace kayla_waiting_years_l600_600645

def minimum_driving_age : ℕ := 18
def kimiko_age : ℕ := 26
def kayla_age : ℕ := kimiko_age / 2
def years_until_kayla_can_drive : ℕ := minimum_driving_age - kayla_age

theorem kayla_waiting_years : years_until_kayla_can_drive = 5 :=
by
  sorry

end kayla_waiting_years_l600_600645


namespace negation_of_universal_proposition_l600_600183
open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 :=
by
  sorry

end negation_of_universal_proposition_l600_600183


namespace count_interesting_quadruples_l600_600067

def interesting_quadruples (a b c d : ℤ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + 2 * d > b + 2 * c 

theorem count_interesting_quadruples : 
  (∃ n : ℤ, n = 582 ∧ ∀ a b c d : ℤ, interesting_quadruples a b c d → n = 582) :=
sorry

end count_interesting_quadruples_l600_600067


namespace convert_base_3_to_base_10_l600_600534

theorem convert_base_3_to_base_10 : 
  (1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) = 142 :=
by
  sorry

end convert_base_3_to_base_10_l600_600534


namespace plane_divided_into_four_regions_l600_600916

theorem plane_divided_into_four_regions :
  let L1 := {p : ℝ × ℝ | p.2 = 3 * p.1}
  let L2 := {p : ℝ × ℝ | p.2 = (1/3) * p.1}
  (∀ p : ℝ × ℝ, (p ∈ L1 ∨ p ∈ L2) ↔ (p.2 = 3 * p.1 ∨ p.2 = (1/3) * p.1)) →
  (∃ r1 r2 r3 r4 : set (ℝ × ℝ),
    (∀ p : ℝ × ℝ, p ∈ (r1 ∪ r2 ∪ r3 ∪ r4)) ∧
    disjoint r1 r2 ∧ disjoint r2 r3 ∧ disjoint r3 r4 ∧ disjoint r4 r1 ∧
    ∀ r, r = r1 ∨ r = r2 ∨ r = r3 ∨ r = r4) :=
by sorry

end plane_divided_into_four_regions_l600_600916


namespace p_plus_q_4063_l600_600275

theorem p_plus_q_4063 
  (p q : ℕ) 
  (h1 : Nat.coprime p q) 
  (h2 : p > 0) 
  (h3 : q > 0) 
  (h4 : (p : ℚ) / (q : ℚ) = (1 / 2^1) + (2 / 3^2) + (3 / 2^3) + (4 / 5^2)) : 
  p + q = 4063 := 
  sorry

end p_plus_q_4063_l600_600275


namespace price_is_approx_85_7_percent_l600_600843

-- Definitions given in the conditions
variables (P A B : ℝ)
axiom h1 : P = 1.5 * A
axiom h2 : P = 2 * B

-- Define the combined assets
def combined_assets (A B : ℝ) : ℝ :=
  A + B

-- Define the percentage function
def percentage_of_combined_assets (P : ℝ) (combined_assets : ℝ) : ℝ :=
  (P / combined_assets) * 100

-- The proof statement: The price of company KW is approximately 85.7% of the combined assets of companies A and B if they merge.
theorem price_is_approx_85_7_percent (h1 : P = 1.5 * A) (h2 : P = 2 * B) :
  percentage_of_combined_assets P (combined_assets (P / 1.5) (P / 2)) ≈ 85.7 :=
by sorry

end price_is_approx_85_7_percent_l600_600843


namespace exponentiation_properties_l600_600960

theorem exponentiation_properties
  (a : ℝ) (m n : ℕ) (hm : a^m = 9) (hn : a^n = 3) : a^(m - n) = 3 :=
by
  sorry

end exponentiation_properties_l600_600960


namespace distinct_sum_count_is_11_l600_600539

def is_special_fraction (a b : ℕ) : Prop := a + b = 20 ∧ a > 0 ∧ b > 0

theorem distinct_sum_count_is_11 :
  (nat.filter (λ n : ℕ, ∃ (a1 b1 a2 b2 : ℕ), 
               is_special_fraction a1 b1 ∧ 
               is_special_fraction a2 b2 ∧ 
               n = (a1 * b2) + (a2 * b1) / (a1 * a2)) 
               (list.range 39)).length = 11 := 
sorry

end distinct_sum_count_is_11_l600_600539


namespace B_cubed_inv_l600_600611

open Matrix

variables {α : Type*} [CommRing α]

def B_inv : Matrix (Fin 2) (Fin 2) α := ![![3, 4], ![-2, -3]]

theorem B_cubed_inv (B : Matrix (Fin 2) (Fin 2) α) (hB : B⁻¹ = B_inv) : 
  (B ^ 3)⁻¹ = B^(-3) := 
by
  sorry

end B_cubed_inv_l600_600611


namespace annual_storage_cost_300_optimal_order_quantity_min_cost_l600_600076

def T (A B C x : ℝ) : ℝ := (B * x / 2) + (A * C / x)

-- Given conditions
def A : ℝ := 6000
def B : ℝ := 120
def C : ℝ := 2500

-- Lean 4 statements for the problems:
theorem annual_storage_cost_300 :
  T A B C 300 = 68000 := 
sorry

theorem optimal_order_quantity_min_cost :
  T A B C 500 = 60000 ∧ (∀ (x : ℝ), T A B C x ≥ T A B C 500) := 
sorry

end annual_storage_cost_300_optimal_order_quantity_min_cost_l600_600076


namespace paperboy_has_863_ways_l600_600442
noncomputable def ways_to_deliver_papers : ℕ :=
  let D : ℕ → ℕ := 
    λ n, if n = 1 then 2
         else if n = 2 then 4
         else if n = 3 then 7
         else if n = 4 then D 3
         else if n = 9 then D 8
         else D (n - 1) + D (n - 2) + D (n - 3)
  in D 12

theorem paperboy_has_863_ways :
  ways_to_deliver_papers = 863 := 
sorry

end paperboy_has_863_ways_l600_600442


namespace Cheryl_more_eggs_than_others_l600_600923

-- Definitions based on conditions
def KevinEggs : Nat := 5
def BonnieEggs : Nat := 13
def GeorgeEggs : Nat := 9
def CherylEggs : Nat := 56

-- Main theorem statement
theorem Cheryl_more_eggs_than_others : (CherylEggs - (KevinEggs + BonnieEggs + GeorgeEggs) = 29) :=
by
  -- Proof would go here
  sorry

end Cheryl_more_eggs_than_others_l600_600923


namespace maximum_n_l600_600360

-- Definition of the increasing sequence of integers starting from a1
def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- The proof problem in Lean 4 statement
theorem maximum_n (a : ℕ → ℕ) (h_inc : sequence a) (h_a1 : a 1 ≥ 3) (sum_eq_100 : ∑ i in finset.range n, a (i + 1) = 100) :
  n = 11 :=
sorry

end maximum_n_l600_600360


namespace second_player_wins_optimal_play_l600_600292

def players_take_turns : Prop := sorry
def win_condition (box_count : ℕ) : Prop := box_count = 21

theorem second_player_wins_optimal_play (boxes : Fin 11 → ℕ)
    (h_turns : players_take_turns)
    (h_win : ∀ i : Fin 11, win_condition (boxes i)) : 
    ∃ P : ℕ, P = 2 :=
sorry

end second_player_wins_optimal_play_l600_600292


namespace cost_of_product_l600_600448

theorem cost_of_product (x : ℝ) (a : ℝ) (h : a > 0) :
  (1 + a / 100) * (x / (1 + a / 100)) = x :=
by
  field_simp [ne_of_gt h]
  sorry

end cost_of_product_l600_600448


namespace circle_diameter_length_l600_600429

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l600_600429


namespace length_of_EF_l600_600678

theorem length_of_EF (AB BC : ℝ) (hAB: AB = 10) (hBC: BC = 6)
  (DE DF : ℝ) (hDE : DE = DF)
  (area_DEF : ℝ) (hArea_DEF: area_DEF = (10 * 6) / 3) :
  ∃ EF : ℝ, EF = 4 * Real.sqrt 5 :=
by
  -- Calculate rectangle area
  have hArea : (10 * 6 : ℝ) = 60 := rfl
  -- Calculate area of DEF
  have hArea_DEF_val : (60 / 3 : ℝ) = 20 := rfl
  -- Using DE = DF, calculate DE:
  have hDE_val : (DE * DE = 40) :=
    by linarith [(hArea_DEF_val : 20 = (1 / 2) * DE * DE)]
  -- Calculate EF using Pythagorean theorem
  have hEF_val : (EF * EF = 80) :=
    by linarith [hDE_val, hDE]
  by
  {use 4 * Real.sqrt 5}
  {sorry}

end length_of_EF_l600_600678


namespace find_m_l600_600626

theorem find_m (α : ℝ) (m : ℝ) (h1 : Float.cos α = -4/5) (h2 : ∃ y : ℝ, (m, y) = (m, -3)) :
  m = -4 :=
by
  sorry

end find_m_l600_600626


namespace distance_M_to_B_l600_600234

-- Define the lengths
def AC : ℝ := real.sqrt 6
def BS : ℝ := 1

-- Areas of the faces
def Q : ℝ := sorry  -- area of ASB and BSC
def area_ASB : ℝ := Q
def area_BSC : ℝ := Q
def area_ASC : ℝ := 2 * Q

-- Conditions of the problem
axiom intersecting_line (AC BS : ℝ) : Prop := sorry -- Defines the line intersecting the midpoint of BS and perpendicular to AC and BS
axiom area_cond (area_ASB area_BSC area_ASC : ℝ) : area_ASB = area_BSC ∧ area_ASC = 2 * area_BSC
axiom distance_cond (M B S : ℝ) : M.dist B + M.dist S = M.dist_face SABC AS + M.dist_face SABC SB + M.dist_face SABC SC + M.dist_face SABC AC

-- Proposition
theorem distance_M_to_B (M B S A C : Point) (AC BS : ℝ) (Q : ℝ)
  (h1 : AC = sqrt 6)
  (h2 : BS = 1)
  (h3 : intersecting_line AC BS)
  (h4 : area_cond Q Q (2 * Q))
  (h5 : distance_cond M B S) :
  M.dist B = sqrt 10 / 6 :=
sorry

end distance_M_to_B_l600_600234


namespace area_of_triangle_PF1F2_l600_600973

noncomputable def ellipseEquation (x y : ℝ) : Prop := 
  (x^2) / 169 + (y^2) / 144 = 1

noncomputable def distance (P F : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)

noncomputable def areaOfTriangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt(s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_PF1F2 (x y : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) 
  (hP : ellipseEquation x y) (hF1F2 : F1 = (5, 0) ∧ F2 = (-5, 0))
  (hdistPF1 : distance P F1 = 10)
  (hdistPF2 : distance P F2 = 16):
  areaOfTriangle 10 10 16 = 48 :=
sorry

end area_of_triangle_PF1F2_l600_600973


namespace ascending_descending_sixth_ascending_descending_fifth_l600_600529

/-- Consider seven-digit natural numbers in which each of the digits 1, 2, 3, 4, 5, 6, 7 appears exactly once. 
(a) Prove that there are exactly 6 such numbers where the first six digits are in ascending order 
    and the sixth and seventh digits are in descending order. -/
theorem ascending_descending_sixth {α: Type*} [Fintype α] [DecidableEq α] : 
    ∀ (s : Finset ℕ), s.card = 7 → (s = ({1, 2, 3, 4, 5, 6, 7}: Finset ℕ)) → 
    ∃! (l : List ℕ), l.length = 7 ∧ (l.take 6).Ascending ∧ (l.drop 5).Descending :=
by sorry

/-- Consider seven-digit natural numbers in which each of the digits 1, 2, 3, 4, 5, 6, 7 appears exactly once. 
(b) Prove that there are exactly 15 such numbers where the first five digits are in ascending order 
    and the fifth to seventh digits are in descending order. -/
theorem ascending_descending_fifth {α: Type*} [Fintype α] [DecidableEq α] : 
    ∀ (s : Finset ℕ), s.card = 7 → (s = ({1, 2, 3, 4, 5, 6, 7}: Finset ℕ)) → 
    ∃! (l : List ℕ), l.length = 7 ∧ (l.take 5).Ascending ∧ (l.drop 4).Descending :=
by sorry

end ascending_descending_sixth_ascending_descending_fifth_l600_600529


namespace minimum_area_triangle_POQ_l600_600078

theorem minimum_area_triangle_POQ
  (x y : ℝ)
  (hx : (x^2 / 4 + y^2 = 1))
  (P Q : ℝ × ℝ)
  (hP : (P.1^2 / 4 + P.2^2 = 1))
  (hQ : (Q.1^2 / 4 + Q.2^2 = 1))
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (h_perp : P.1 * Q.1 + P.2 * Q.2 = 0) :
  ∃ a : ℝ, a = 4 / 5 ∧ ∀ area : ℝ, area = 1 / 2 * dist O P * dist O Q → area ≥ a :=
begin
  sorry
end

end minimum_area_triangle_POQ_l600_600078


namespace buying_beams_l600_600320

theorem buying_beams (x : ℕ) (h : 3 * (x - 1) * x = 6210) :
  3 * (x - 1) * x = 6210 :=
by {
  sorry
}

end buying_beams_l600_600320


namespace KLMN_is_rhombus_l600_600217

-- Define a cyclic quadrilateral
variable (A B C D K L M N P Q : Type) [geometry : CyclicQuadrilateral A B C D]

-- Define the points K, L, M, and N on AB, BC, CD, and DA respectively such that KLMN is a parallelogram
variable (on_line_K : PointOnLine K A B)
variable (on_line_L : PointOnLine L B C)
variable (on_line_M : PointOnLine M C D)
variable (on_line_N : PointOnLine N D A)
variable (parallelogram_KLMN : Parallelogram K L M N)

-- Define the common points
variable (common_point_P : IntersectPoint P AB DC NL)
variable (common_point_Q : IntersectPoint Q AD BC KM)

-- Define the cyclic property of ABCD
variable (cyclic_property : Angle A + Angle C = π)

-- Definition of the problem statement
theorem KLMN_is_rhombus
  (h1 : PointOnLine K A B)
  (h2 : PointOnLine L B C)
  (h3 : PointOnLine M C D)
  (h4 : PointOnLine N D A)
  (h5 : Parallelogram K L M N)
  (h6 : IntersectPoint P AB DC NL)
  (h7 : IntersectPoint Q AD BC KM)
  (h8 : Angle A + Angle C = π) : Rhombus K L M N := sorry

end KLMN_is_rhombus_l600_600217


namespace sum_of_numbers_l600_600483

theorem sum_of_numbers : 145 + 33 + 29 + 13 = 220 :=
by
  sorry

end sum_of_numbers_l600_600483


namespace contemporaries_probability_calc_l600_600379

theorem contemporaries_probability_calc :
  ∃ (Alice_life Bob_life span : ℕ) (P : ℚ), 
    Alice_life = 110 ∧ 
    Bob_life = 90 ∧ 
    span = 600 ∧ 
    P = 5 / 9 ∧ 
    (calc_probability Alice_life Bob_life span = P) :=
begin
  let Alice_life := 110,
  let Bob_life := 90,
  let span := 600,
  let P := 5 / 9,
  have HP : calc_probability Alice_life Bob_life span = P, 
  { sorry }, 
  exact ⟨Alice_life, Bob_life, span, P, rfl, rfl, rfl, HP⟩,
end

-- Define calculation function that should do the proper calculation:
noncomputable def calc_probability (Alice_life Bob_life span : ℕ) : ℚ := sorry

end contemporaries_probability_calc_l600_600379


namespace principal_trebled_after_five_years_l600_600363

-- Define the necessary variables and assumptions
variables (P R : ℝ) (Y : ℝ)

-- Given conditions as definitions in Lean
def condition1 : Prop := P * R = 6000
def condition2 : Prop := 1200 = (P * R * Y) / 100 + (3 * P * R * (10 - Y)) / 100

-- The statement to prove
theorem principal_trebled_after_five_years (h1 : condition1) (h2 : condition2) : Y = 5 :=
by
  unfold condition1 at h1
  unfold condition2 at h2
  sorry

end principal_trebled_after_five_years_l600_600363


namespace correct_option_C_l600_600466

-- Define the conditions for the phenomena
def phenomenon1 := "A traffic officer records more than 300 cars passing in one hour"
def phenomenon2 (a : Int) := a + 1 ∈ Int
def phenomenon3 := "Firing a shell and hitting the target"
def phenomenon4 := "Inspecting a product on the assembly line to determine if it is a qualified product or a defective one"

-- Define which phenomena are random
def is_random (phenomenon : String) : Prop :=
  phenomenon = "A traffic officer records more than 300 cars passing in one hour" ∨
  phenomenon = "Firing a shell and hitting the target" ∨
  phenomenon = "Inspecting a product on the assembly line to determine if it is a qualified product or a defective one"

-- The theorem to be proved
theorem correct_option_C : 
  (is_random phenomenon1) ∧ 
  (¬ (∀ a : Int, is_random (phenomenon2 a))) ∧ 
  (is_random phenomenon3) ∧ 
  (is_random phenomenon4) :=
by
  sorry

end correct_option_C_l600_600466


namespace solve_for_x_l600_600308

noncomputable def proof_problem (x : ℝ) : Prop :=
  (log 2 (3*x+9) / (5*x-3) + log 2 (5*x-3) / (x-2) = 2) 
  ∧ ((3*x+9) / (5*x-3) > 0) 
  ∧ ((5*x-3) / (x-2) > 0)

theorem solve_for_x : proof_problem 17 :=
sorry

end solve_for_x_l600_600308


namespace area_and_sum_of_coefficients_l600_600731

noncomputable def area_of_parallelogram (z1 z2 : ℂ) : ℝ :=
  abs ((z1 * conj z2).im)

def complex_equations (z1 z2 : ℂ) : Prop :=
  (z1 ^ 2 = 9 + 9 * real.sqrt 7 * complex.I) ∧
  (z2 ^ 2 = 3 + 3 * real.sqrt 2 * complex.I)

theorem area_and_sum_of_coefficients :
  ∃ (p q r s : ℕ),
    (∃ z1 z2 : ℂ, complex_equations z1 z2 ∧
                   area_of_parallelogram z1 z2 = 63 * real.sqrt 35 - 27 * real.sqrt 15) ∧
    p + q + r + s = 140 :=
begin
  sorry
end

end area_and_sum_of_coefficients_l600_600731


namespace range_of_sin_cos_expression_l600_600158

variable (a b c A B C : ℝ)

theorem range_of_sin_cos_expression
  (h1 : a = b)
  (h2 : c * Real.sin A = -a * Real.cos C) :
  1 < 2 * Real.sin (A + Real.pi / 6) :=
sorry

end range_of_sin_cos_expression_l600_600158


namespace a_mul_b_value_l600_600346

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l600_600346


namespace graph_f_does_not_pass_quadrant_II_l600_600334

noncomputable def f : ℝ → ℝ := λ x, 3 * x - 2

theorem graph_f_does_not_pass_quadrant_II :
  ∀ (x y : ℝ), (y = f x) → ¬ ((x < 0) ∧ (y > 0)) :=
by sorry

end graph_f_does_not_pass_quadrant_II_l600_600334


namespace calculate_shaded_area_l600_600787

-- Define the conditions

def rectangle_length := 30
def rectangle_width := 10
def circle_diameter := rectangle_width
def circle_radius := circle_diameter / 2
def area_rectangle := rectangle_length * rectangle_width
def area_one_circle := Real.pi * (circle_radius ^ 2)
def total_area_circles := 3 * area_one_circle
def shaded_area := area_rectangle - total_area_circles

-- We state that, given the above conditions, the shaded area is as calculated.

theorem calculate_shaded_area :
  shaded_area = 300 - 75 * Real.pi :=
  by sorry

end calculate_shaded_area_l600_600787


namespace find_natural_triples_l600_600566

theorem find_natural_triples (x y z : ℕ) : 
  (x+1) * (y+1) * (z+1) = 3 * x * y * z ↔ 
  (x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 2) ∨ (x, y, z) = (3, 2, 2) ∨
  (x, y, z) = (5, 1, 4) ∨ (x, y, z) = (5, 4, 1) ∨ (x, y, z) = (4, 1, 5) ∨ (x, y, z) = (4, 5, 1) ∨ 
  (x, y, z) = (1, 4, 5) ∨ (x, y, z) = (1, 5, 4) ∨ (x, y, z) = (8, 1, 3) ∨ (x, y, z) = (8, 3, 1) ∨
  (x, y, z) = (3, 1, 8) ∨ (x, y, z) = (3, 8, 1) ∨ (x, y, z) = (1, 3, 8) ∨ (x, y, z) = (1, 8, 3) :=
by {
  sorry
}

end find_natural_triples_l600_600566


namespace segment_length_reflection_l600_600782

theorem segment_length_reflection (Z : ℝ×ℝ) (Z' : ℝ×ℝ) (hx : Z = (5, 2)) (hx' : Z' = (5, -2)) :
  dist Z Z' = 4 := by
  sorry

end segment_length_reflection_l600_600782


namespace circle_diameter_l600_600424

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d, d = 4 :=
by
  let r := Real.sqrt 4
  let d := 2 * r
  use d
  simp only [Real.sqrt_eq_rfl, mul_eq_zero, ne.def, not_false_iff]
  linarith
  sorry

end circle_diameter_l600_600424


namespace find_a_if_parallel_l600_600212

-- Define the parallel condition for the given lines
def is_parallel (a : ℝ) : Prop :=
  let slope1 := -a / 2
  let slope2 := 3
  slope1 = slope2

-- Prove that a = -6 under the parallel condition
theorem find_a_if_parallel (a : ℝ) (h : is_parallel a) : a = -6 := by
  sorry

end find_a_if_parallel_l600_600212


namespace covering_percentage_77_l600_600388

-- Definition section for conditions
def radius_of_circle (r a : ℝ) := 2 * r * Real.pi = 4 * a
def center_coincide (a b : ℝ) := a = b

-- Theorem to be proven
theorem covering_percentage_77
  (r a : ℝ)
  (h_radius: radius_of_circle r a)
  (h_center: center_coincide 0 0) : 
  (r^2 * Real.pi - 0.7248 * r^2) / (r^2 * Real.pi) * 100 = 77 := by
  sorry

end covering_percentage_77_l600_600388


namespace length_of_AB_l600_600586

-- Definitions for the conditions
variables {P O A B : Type} [metric_space P] [metric_space O]
variables (PA PB OA : ℝ) -- Tangent segments and radius
variables (h1 : PA = PB) -- PA equals PB as they are tangent segments from P
variables (h2 : (1 / OA^2) + (1 / PA^2) = 1 / 16) -- Given condition

-- The goal is to find the length of AB
theorem length_of_AB (h1 : PA = PB) (h2 : (1 / OA^2) + (1 / PA^2) = 1 / 16) : (2 * sqrt(PA^2 - OA^2)) = 8 := 
by
  sorry

end length_of_AB_l600_600586


namespace diagonal_length_of_regular_octagon_l600_600945

theorem diagonal_length_of_regular_octagon (s : ℝ) (h_s : s = 12) : 
  ∀ a d, (a - d).norm = 12 * Real.sqrt 2 → True :=
by
  sorry

end diagonal_length_of_regular_octagon_l600_600945


namespace triangle_extension_theorem_l600_600233

noncomputable def geometric_property (A B C M A₁ B₁ C₁ : Point) : Prop :=
  ∃ ABC_triangle : Triangle,
    M ∈ triangle_plane ABC_triangle ∧
    M = intersection_perpendicular_line A A₁ ∧
    M = intersection_perpendicular_line B B₁ ∧
    M = intersection_perpendicular_line C C₁ ∧
    (points_on_extensions_or_one_extension ABC_triangle A₁ B₁ C₁)

-- This is a statement that asserts the property we intend to prove.
theorem triangle_extension_theorem (A B C M A₁ B₁ C₁ : Point)
  (h1 : Triangle ABC A B C)
  (h2 : M ∈ h1.plane)
  (h3 : is_perpendicular_intersection M A A₁)
  (h4 : is_perpendicular_intersection M B B₁)
  (h5 : is_perpendicular_intersection M C C₁) :
  geometric_property A B C M A₁ B₁ C₁ :=
begin
  sorry -- The proof is omitted as requested.
end

end triangle_extension_theorem_l600_600233


namespace part1_arithmetic_sequence_part2_sum_t_n_l600_600145

-- Definitions for the sequence and sums
def seq (a : ℕ → ℝ) := ∀ n : ℕ, 0 < a n
def sum_first (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, S n = ∑ i in finset.range n, a i
def a1_eq_1 (a : ℕ → ℝ) := a 0 = 1
def relation_k (k : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, k * S n = a n * a (n + 1)

-- Part 1: Prove sequence is arithmetic when k = 2
theorem part1_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : seq a) (h_sum : sum_first S a) (h_a1 : a1_eq_1 a) (h_relation : relation_k 2 S a) :
  ∀ n : ℕ, a (n + 1) - a n = 1 :=
sorry

-- Part 2: Find the sum T_n when k = 1/2
def reciprocal_S2n_sum (S : ℕ → ℝ) (T : ℕ → ℝ) := 
  ∀ n : ℕ, T n = ∑ i in finset.range n, 1 / S (2 * i)

theorem part2_sum_t_n (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)
  (h_seq : seq a) (h_sum : sum_first S a) (h_a1 : a1_eq_1 a) (h_relation : relation_k (1 / 2) S a)
  (h_T : reciprocal_S2n_sum S T) :
  ∀ n : ℕ, T n = 3 / 2 - (2 * n + 3) / (n * (n + 3) + 2) :=
sorry

end part1_arithmetic_sequence_part2_sum_t_n_l600_600145


namespace completing_the_square_l600_600792

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l600_600792


namespace tan_sum_identity_l600_600530

theorem tan_sum_identity 
  (α β γ : ℝ) 
  (h_cond : α + β + γ = 90) 
  (h_tan_α : ∃ x, tan α = x)
  (h_tan_β : ∃ x, tan β = x)
  (h_tan_γ : ∃ x, tan γ = x) 
: tan α * tan β + tan β * tan γ + tan γ * tan α = 1 := 
sorry

end tan_sum_identity_l600_600530


namespace right_triangle_A_l600_600394

theorem right_triangle_A :
  ∃ (a b c : ℕ), a = 5 ∧ b = 4 ∧ c = 3 ∧ a^2 = b^2 + c^2 :=
by
  use 5, 4, 3
  simp
  norm_num
  sorry

end right_triangle_A_l600_600394


namespace find_x_l600_600105

theorem find_x (x : ℝ) (h : sqrt (9 - 2 * x) = 8) : x = -55 / 2 := 
sorry

end find_x_l600_600105


namespace line_equation_through_point_with_slope_l600_600568

theorem line_equation_through_point_with_slope (
    x1 y1 : ℝ, 
    m : ℝ
) (
    h_point : x1 = -3 ∧ y1 = 2, 
    h_slope_angle : m = Real.sqrt 3
) : (∀ x y : ℝ, (y - y1 = m * (x - x1)) ↔ (y - 2 = Real.sqrt 3 * (x + 3))) :=
by
  sorry

end line_equation_through_point_with_slope_l600_600568


namespace find_f0_l600_600596

noncomputable def f : ℕ → ℕ :=
sorry

axiom A1 : ∀ n, f(f(n)) + f(n) = 2 * n + 3
axiom A2 : f 2013 = 2014

theorem find_f0 : f 0 = 1 :=
by
  sorry

end find_f0_l600_600596


namespace completing_the_square_l600_600800

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l600_600800


namespace find_d_l600_600846

theorem find_d (a d : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * d) : d = 49 :=
sorry

end find_d_l600_600846


namespace max_n_increasing_seq_l600_600358

theorem max_n_increasing_seq (a : ℕ → ℕ) (h₁ : ∀ n, a n < a (n + 1))
  (h₂ : a 0 ≥ 3) (h₃ : (List.range n).sum a = 100) : n ≤ 11 :=
by
  simp at h₃
  sorry

end max_n_increasing_seq_l600_600358


namespace ellipse_focus_sum_property_l600_600900

theorem ellipse_focus_sum_property :
  ∀ (a b : ℝ) (x y : ℝ) (F1 F2 A B : ℝ × ℝ) (AB_distance : ℝ),
  a^2 = 16 → b^2 = 9 →
  (x, y) ∈ { p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1 } →
  (A = (x, y) ∨ B = (x, y)) →
  (distance A B) = 6 →
  let F1 := (sqrt(16 - 9), 0) in
  let F2 := (-sqrt(16 - 9), 0) in
  (dist A F1 + dist B F1) = 10 :=
by intro a b x y F1 F2 A B AB_distance hA hB h_ellipse_points h_A_or_B h_AB_distance;
   sorry

end ellipse_focus_sum_property_l600_600900


namespace factorization_l600_600080

-- Given conditions:
def cond1 := (x : ℤ) → x^2 + 2x + 1
def cond2 := (x : ℤ) → x^2 + 8x + 15
def cond3 := (x : ℤ) → x^2 + 6x + 5

-- The theorem to prove the factorization
theorem factorization (x : ℤ) : 
  (cond1 x) * (cond2 x) + cond3 x = (x + 1) * (x + 5) * (x + 2)^2 :=
by
  sorry

end factorization_l600_600080


namespace sqrt_div_val_l600_600365

theorem sqrt_div_val (n : ℕ) (h : n = 3600) : (Nat.sqrt n) / 15 = 4 := by 
  sorry

end sqrt_div_val_l600_600365


namespace total_signup_options_l600_600412

theorem total_signup_options (students teams : ℕ) (h_students : students = 5) (h_teams : teams = 3) : 
  (teams ^ students) = 3^5 := 
by
  rw [h_students, h_teams]
  rfl

end total_signup_options_l600_600412


namespace a_mul_b_value_l600_600344

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l600_600344


namespace factorize_x_cubed_minus_9x_l600_600929

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l600_600929


namespace complete_the_square_l600_600826

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l600_600826


namespace required_C6H6_l600_600940

-- Definitions for chemical entities
constant C6H6 : Type
constant CH4 : Type
constant C6H5CH3 : Type
constant H2 : Type

-- Number of moles required for reactants and products
constant moles : Type
constant n_C6H6 : moles
constant n_CH4 : moles
constant n_C6H5CH3 : moles
constant n_H2 : moles

-- Initial conditions
axiom CH4_initial : n_CH4 = 3

-- Balanced chemical equation condition
axiom reaction_balance : ∀ (n_C6H6 n_CH4 n_C6H5CH3 n_H2 : moles),
  (n_C6H6 = n_CH4) ∧ (n_C6H5CH3 = n_CH4) ∧ (n_H2 = n_CH4)

-- Required proof statement
theorem required_C6H6 : n_C6H6 = 3 :=
by {
  have balance := reaction_balance n_C6H6 n_CH4 n_C6H5CH3 n_H2,
  rw CH4_initial at balance,
  exact balance.left,
}

end required_C6H6_l600_600940


namespace length_of_the_tiger_l600_600030

noncomputable def tiger_length 
  (constant_speed : Prop)
  (passes_blade_in_one_second : ∀ (distance : ℝ), distance = tiger_length * time ∧ time = 1) 
  (runs_tree_trunk_in_five_seconds : ∀ (distance : ℝ), distance = 20 ∧ time = 5) : 
  Prop := 
  tiger_length = 4

theorem length_of_the_tiger 
  (constant_speed : Prop) 
  (passes_blade_in_one_second : ∀ (distance : ℝ), distance = tiger_length * time ∧ time = 1) 
  (runs_tree_trunk_in_five_seconds : ∀ (distance : ℝ), distance = 20 ∧ time = 5) : 
  tiger_length constant_speed passes_blade_in_one_second runs_tree_trunk_in_five_seconds := 
sorry

end length_of_the_tiger_l600_600030


namespace blue_higher_than_yellow_l600_600854

noncomputable def prob_blue_higher_than_yellow : ℚ :=
  ∑' (k : ℕ) in (Set.univ : Set ℕ) \ {0}, 3^(-2 * k)

theorem blue_higher_than_yellow :
  let p_same_bin := prob_blue_higher_than_yellow in
  let p_diff_bin := 1 - p_same_bin in
  let p_blue_higher := p_diff_bin / 2 in
  p_blue_higher = 7 / 16 := by
{
  sorry
}

end blue_higher_than_yellow_l600_600854


namespace peter_age_l600_600693

variable (x y : ℕ)

theorem peter_age : 
  (x = (3 * y) / 2) ∧ ((4 * y - x) + 2 * y = 54) → x = 18 :=
by
  intro h
  cases h
  sorry

end peter_age_l600_600693


namespace arithmetic_proof_l600_600413

theorem arithmetic_proof : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end arithmetic_proof_l600_600413


namespace fraction_value_l600_600767

theorem fraction_value : (1998 - 998) / 1000 = 1 :=
by
  sorry

end fraction_value_l600_600767


namespace cost_per_person_is_125_l600_600316

-- Defining the conditions
def totalCost : ℤ := 25000000000
def peopleSharing : ℤ := 200000000

-- Define the expected cost per person based on the conditions
def costPerPerson : ℤ := totalCost / peopleSharing

-- Proving that the cost per person is 125 dollars.
theorem cost_per_person_is_125 : costPerPerson = 125 := by
  sorry

end cost_per_person_is_125_l600_600316


namespace smallest_n_sqrt_diff_l600_600389

theorem smallest_n_sqrt_diff (n : ℕ) (h : sqrt n - sqrt (n - 1) < 0.02) : n = 626 :=
sorry

end smallest_n_sqrt_diff_l600_600389


namespace divisible_term_exists_l600_600362

theorem divisible_term_exists (k : ℕ) (h : k > 0) : 
  ∃ m : ℕ, (let a : ℕ → ℕ :=
    λ n, if n = 1 then 1 else
      (a (n - 1)) + (a (nat.floor (real.sqrt (n - 1))))) in 
  a m % k = 0 :=
sorry

end divisible_term_exists_l600_600362


namespace problem_lean_statement_l600_600314

theorem problem_lean_statement :
  ∃ m : ℕ, (197 * 879) % 60 = m ∧ 0 ≤ m ∧ m < 60 ∧ m = 3 := 
by {
  use 3,
  norm_num,
  sorry
}

end problem_lean_statement_l600_600314


namespace smallest_primer_is_6_primer6_l600_600019

def distinct_prime_factors (n : ℕ) : ℕ :=
  (Finset.filter Nat.prime (Nat.factors n)).card

def is_prime (n : ℕ) : Prop :=
  Nat.prime n

def is_primer (n : ℕ) : Prop :=
  is_prime (distinct_prime_factors n)

theorem smallest_primer_is_6 : ∀ n : ℕ, is_primer n → n >= 6 :=
begin
  sorry
end

theorem primer6 : is_primer 6 :=
begin
  -- Proof not required as per the problem statement
  sorry
end

end smallest_primer_is_6_primer6_l600_600019


namespace find_overall_mean_score_l600_600713

variable (M N E : ℝ)
variable (m n e : ℝ)

theorem find_overall_mean_score :
  M = 85 → N = 75 → E = 65 →
  m / n = 4 / 5 → n / e = 3 / 2 →
  ((85 * m) + (75 * n) + (65 * e)) / (m + n + e) = 82 :=
by
  sorry

end find_overall_mean_score_l600_600713


namespace probability_all_students_same_canteen_l600_600021

theorem probability_all_students_same_canteen (num_canteens : ℕ) (num_students : ℕ) :
  num_canteens = 2 → num_students = 3 → 
  let p := (2 : ℕ) / (2 ^ num_students : ℕ) in 
  p = (1 : ℕ) / 4 :=
by
  intros h1 h2,
  have h_total_outcomes : 2 ^ 3 = 8 := by norm_num,
  have h_favorable_outcomes : 2 = 2 := rfl,
  let p := 2 / 8,
  have hp : p = 1 / 4 := by norm_num,
  rw [h1, h2, h_total_outcomes, h_favorable_outcomes, hp],
  sorry

end probability_all_students_same_canteen_l600_600021


namespace monthly_earnings_l600_600443

theorem monthly_earnings (savings_per_month : ℤ) (total_needed : ℤ) (total_earned : ℤ)
  (H1 : savings_per_month = 500)
  (H2 : total_needed = 45000)
  (H3 : total_earned = 360000) :
  total_earned / (total_needed / savings_per_month) = 4000 := by
  sorry

end monthly_earnings_l600_600443


namespace hyperbola_eccentricity_l600_600162

theorem hyperbola_eccentricity :
  (∃ (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0),
    let e := ((a^2 + b^2)^(1/2)) / a in
    (a / b = sqrt 3) →
    e = (2 * sqrt 3) / 3) :=
begin
  sorry
end

end hyperbola_eccentricity_l600_600162


namespace ratio_of_birds_to_trees_and_stones_l600_600338

theorem ratio_of_birds_to_trees_and_stones (stones birds : ℕ) (h_stones : stones = 40)
  (h_birds : birds = 400) (trees : ℕ) (h_trees : trees = 3 * stones + stones) :
  (birds : ℚ) / (trees + stones) = 2 :=
by
  -- The actual proof steps would go here.
  sorry

end ratio_of_birds_to_trees_and_stones_l600_600338


namespace elements_with_first_digit_8_l600_600699

noncomputable def number_of_elements_with_first_digit_8 : ℕ :=
  let S := {k : ℕ | k ≤ 3000}
  let log10_8 := Real.log10 8
  let first_digit_8_interval := (0.90309, 0.95424)
  (S.filter (λ k, let frac_part := (k * log10_8) % 1 in frac_part ≥ first_digit_8_interval.1 ∧ frac_part < first_digit_8_interval.2)).card

theorem elements_with_first_digit_8 :
  let S := {k : ℕ | k ≤ 3000}
  let given_number_of_digits := 2713
  let first_digit_8 := 8
  (S.filter (λ k, let frac_part := (k * Float.log10 8) % 1 in frac_part ≥ 0.90309 ∧ frac_part < 0.95424)).card = 154 :=
by
  sorry

end elements_with_first_digit_8_l600_600699


namespace find_x_l600_600403

noncomputable theory

def average (a b c d e f g h i j : ℕ) : ℕ := (a + b + c + d + e + f + g + h + i + j) / 10

theorem find_x
  (x : ℕ)
  (h : average 54 55 57 58 59 62 62 63 65 x = 60) :
  x = 65 :=
sorry

end find_x_l600_600403


namespace avg_weight_of_entire_class_l600_600773

-- Definitions based on conditions
def students_A := 50
def avg_weight_A := 50
def total_weight_A := students_A * avg_weight_A

def students_B := 70
def avg_weight_B := 60
def total_weight_B := students_B * avg_weight_B

def students_C := 40
def avg_weight_C := 55
def total_weight_C := students_C * avg_weight_C

def students_D := 80
def avg_weight_D := 70
def total_weight_D := students_D * avg_weight_D

def students_E := 60
def avg_weight_E := 65
def total_weight_E := students_E * avg_weight_E

def total_weight := total_weight_A + total_weight_B + total_weight_C + total_weight_D + total_weight_E
def total_students := students_A + students_B + students_C + students_D + students_E

def avg_weight_entire_class := total_weight / total_students

-- Theorem to prove
theorem avg_weight_of_entire_class :
  avg_weight_entire_class = 61.33 := 
by
  -- Proof goes here
  sorry

end avg_weight_of_entire_class_l600_600773


namespace num_tuples_eq_num_valid_tuples_l600_600698

def valid_tuple (n : ℕ) (x : Fin n → ℕ) := 
  (∀ k, x k ∈ {0, 1, 2}) ∧ 
  ((List.sum (List.ofFn x) - List.prod (List.ofFn x)) % 3 = 0)

noncomputable def num_valid_tuples (n : ℕ) : ℕ :=
  3^(n-1) - 2^(n-2) + 3 * ((∑ k in Finset.range (n-1+1), if (2 * (n-1) - 2) % 6 = k then (Nat.choose (n-1) k) else 0))

theorem num_tuples_eq_num_valid_tuples (n : ℕ) (h : 2 ≤ n) :
  (nat.card {x : Fin n → ℕ // valid_tuple n x}) = num_valid_tuples n :=
sorry

end num_tuples_eq_num_valid_tuples_l600_600698


namespace trig_identity_solution_exists_l600_600477

noncomputable def find_trig_solution : Prop := 
  ∃ (x y z : ℝ) (n m k : ℤ),
    x = (π / 2) + (π : ℝ) * n ∧
    y = (π / 2) + (π : ℝ) * m ∧
    z = (π / 2) + (π : ℝ) * k

theorem trig_identity_solution_exists :
  (∃ (x y z : ℝ), (cos x + cos y + cos z + cos (x + y + z) = 0)) ↔ find_trig_solution := 
sorry

end trig_identity_solution_exists_l600_600477


namespace solve_for_x_l600_600215

-- Problem definition
def problem_statement (x : ℕ) : Prop :=
  (3 * x / 7 = 15) → x = 35

-- Theorem statement in Lean 4
theorem solve_for_x (x : ℕ) : problem_statement x :=
by
  intros h
  sorry

end solve_for_x_l600_600215


namespace completing_square_l600_600805

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l600_600805


namespace correct_exponentiation_l600_600831

theorem correct_exponentiation (a : ℝ) :
  (a^2 * a^3 = a^5) ∧
  (a^2 + a^3 ≠ a^5) ∧
  (a^6 + a^2 ≠ a^4) ∧
  (3 * a^3 - a^2 ≠ 2 * a) :=
by
  sorry

end correct_exponentiation_l600_600831


namespace diane_bakes_gingerbreads_l600_600552

open Nat

theorem diane_bakes_gingerbreads :
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  total_gingerbreads1 + total_gingerbreads2 = 160 := 
by
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  exact Eq.refl (total_gingerbreads1 + total_gingerbreads2)

end diane_bakes_gingerbreads_l600_600552


namespace interior_diagonals_sum_l600_600020

-- Define the variables and conditions
variables {x y z : ℝ}

def is_rectangular_box (x y z : ℝ) : Prop :=
  2 * (x * y + y * z + z * x) = 150 ∧
  4 * (x + y + z) = 60 ∧
  x = 2 * z

-- Define the proposition: sum of all interior diagonals
def sum_interior_diagonals (x y z : ℝ) :=
  4 * real.sqrt (x^2 + y^2 + z^2)

-- Prove that this sum equals 20√2 given the conditions
theorem interior_diagonals_sum :
  ∀ {x y z : ℝ}, is_rectangular_box x y z → sum_interior_diagonals x y z = 20 * real.sqrt 2 :=
by
  intros x y z h
  cases h with ha hb hc
  sorry

end interior_diagonals_sum_l600_600020


namespace a_can_finish_remaining_work_in_5_days_l600_600840

theorem a_can_finish_remaining_work_in_5_days (a_work_rate b_work_rate : ℝ) (total_days_b_works : ℝ):
  a_work_rate = 1/15 → 
  b_work_rate = 1/15 → 
  total_days_b_works = 10 → 
  ∃ (remaining_days_for_a : ℝ), remaining_days_for_a = 5 :=
by
  intros h1 h2 h3
  -- We are skipping the proof itself
  sorry

end a_can_finish_remaining_work_in_5_days_l600_600840


namespace xyz_value_l600_600204

noncomputable def positive (x : ℝ) : Prop := 0 < x

theorem xyz_value (x y z : ℝ) (hx : positive x) (hy : positive y) (hz : positive z): 
  (x + 1/y = 5) → (y + 1/z = 2) → (z + 1/x = 8/3) → x * y * z = (17 + Real.sqrt 285) / 2 :=
by
  sorry

end xyz_value_l600_600204


namespace complete_the_square_l600_600824

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l600_600824


namespace b_n_formula_sum_first_20_terms_l600_600131

-- Definition of sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a n + 1 else a n + 2

-- Definition of sequence b_n as a 2n-th term of a_n
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof problem 1: General formula for b_n
theorem b_n_formula (n : ℕ) : b n = 3 * n - 1 :=
sorry

-- Sum of the first 20 terms of the sequence a_n
theorem sum_first_20_terms : (Finset.range 20).sum a = 300 :=
sorry

end b_n_formula_sum_first_20_terms_l600_600131


namespace b_general_formula_sum_first_20_terms_is_300_l600_600144

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a(n) + 1 else a(n) + 2

-- Define the sequence b_n as bₙ = a₂ₙ
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof goal for part (1)
theorem b_general_formula (n : ℕ) : b n = 3 * n - 1 := sorry

-- Sum of the first 20 terms of the sequence a_n
def sum_first_20_terms : ℕ :=
  (List.range (20)).sum (λ n, a n)

-- Proof goal for part (2)
theorem sum_first_20_terms_is_300 : sum_first_20_terms = 300 := sorry

end b_general_formula_sum_first_20_terms_is_300_l600_600144


namespace problem_1_problem_2_l600_600236
open Real

noncomputable def problem_1_statement (a b c C : ℝ) (h : 2 * sqrt 3 * a * b * sin C = a^2 + b^2 - c^2) : Prop :=
  C = π / 6

noncomputable def problem_2_statement (a b A B C : ℝ) (h1 : a * sin B = b * cos A)
                                      (h2 : a = 2) (h3 : C = π / 6) : Prop :=
  let c : ℝ := sqrt 2 in
  let area := (1/2) * a * c * sin (A + C) in
  area = (sqrt 3 + 1) / 2

theorem problem_1 (a b c C : ℝ) (h : 2 * sqrt 3 * a * b * sin C = a^2 + b^2 - c^2) : problem_1_statement a b c C h :=
sorry

theorem problem_2 (a b A B C : ℝ) (h1 : a * sin B = b * cos A)
                              (h2 : a = 2) (h3 : C = π / 6) : problem_2_statement a b A B C h1 h2 h3 :=
sorry

end problem_1_problem_2_l600_600236


namespace minimum_value_of_expression_l600_600612

theorem minimum_value_of_expression (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 1 / a + 1 / b = 1) : 
  ∃ a b, (a > 0) ∧ (b > 0) ∧ (1 / a + 1 / b = 1) ∧ (b = a / (a - 1)) ∧ (2 * sqrt 6 = min {y | ∃ a b, y = (3 / (a-1) + 2 * (b-1))}) :=
begin
  sorry
end

end minimum_value_of_expression_l600_600612


namespace triangle_ADE_area_l600_600235

open EuclideanGeometry

noncomputable def area_of_triangle_ADE : ℝ :=
  let A := (0, 0)
  let B := (5, 0)
  let D := (0, 5)
  let E := (0, 6)
  1 / 2 * 5 * 5

theorem triangle_ADE_area : area_of_triangle_ADE = 12.5 :=
by
  -- Definitions:
  let A := (0, 0)
  let B := (5, 0)
  let D := (0, 5)
  let E := (0, 6)
  
  -- Conditions:
  -- AD is perpendicular to AB
  have h1 : AD ⊥ AB := sorry,
  -- AD = AB = 5
  have h2 : dist A D = 5 := sorry,
  have h3 : dist A B = 5 := sorry,
  -- BC = 10
  have h4 : dist B C = 10 := sorry,
  -- Point E on BC such that BE = 4
  have h5 : dist B E = 4 := sorry,
  -- DE is parallel to AB
  have h6 : DE ∥ AB := sorry,

  -- Calculation of area of triangle ADE
  have h7 : area (triangle A D E) = 12.5 := sorry,
  exact h7

end triangle_ADE_area_l600_600235


namespace Charles_chocolate_milk_total_l600_600054

theorem Charles_chocolate_milk_total (milk_per_glass syrup_per_glass total_milk total_syrup : ℝ) 
(h_milk_glass : milk_per_glass = 6.5) (h_syrup_glass : syrup_per_glass = 1.5) (h_total_milk : total_milk = 130) (h_total_syrup : total_syrup = 60) :
  (min (total_milk / milk_per_glass) (total_syrup / syrup_per_glass) * (milk_per_glass + syrup_per_glass) = 160) :=
by
  sorry

end Charles_chocolate_milk_total_l600_600054


namespace rocking_chairs_and_stools_arrangement_l600_600373

/--
There is a committee composed of six women and four men. 
When they meet, they sit in a row---the women in indistinguishable rocking chairs and the men on indistinguishable stools.
The meeting requires that the first and last seat in the row must be a rocking chair.
This Lean statement represents the theorem that there are 70 distinct ways to arrange the six rocking chairs and four stools for a meeting.
-/
theorem rocking_chairs_and_stools_arrangement : 
  let women := 6
  let men := 4
  let slots := women + men - 2
  ∃ n : ℕ, women > 1 ∧ nat.choose slots (slots - men) = n ∧ n = 70 :=
by
  -- to assert the conditions applied within the problem.
  let women := 6
  let men := 4
  have slots := women + men - 2
  use nat.choose slots (slots - men)
  have result := nat.choose slots (slots - men)
  show women > 1 ∧ result = 70
  sorry

end rocking_chairs_and_stools_arrangement_l600_600373


namespace value_of_f_8_minus_f_4_l600_600648

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_8_minus_f_4 :
  -- Conditions
  (∀ x, f (-x) = -f x) ∧              -- odd function
  (∀ x, f (x + 5) = f x) ∧            -- period of 5
  (f 1 = 1) ∧                         -- f(1) = 1
  (f 2 = 3) →                         -- f(2) = 3
  -- Goal
  f 8 - f 4 = -2 :=
sorry

end value_of_f_8_minus_f_4_l600_600648


namespace log_expression_in_terms_of_a_l600_600591

theorem log_expression_in_terms_of_a (a : ℝ) (h : a = real.log 2 / real.log 3) :
  real.log 8 / real.log 3 - real.log (3 / 4) / real.log 3 = 5 * a - 1 := 
sorry

end log_expression_in_terms_of_a_l600_600591


namespace concyclic_points_l600_600516

-- Defining conditions for the problem in Lean
variables {ω : Type*} [metric_space ω] [normed_group ω] [normed_space ℝ ω]
variables {A B C D E F P : ω}
variables {circle : set ω} (hcircle : ∀ x ∈ circle, ∃ y ∈ circle, {x, y} ≠ ∅)

-- Geometric properties and given conditions
variables (hAB : A ≠ B) (hCD : C ≠ D) (hE : E ∈ (line_through A B) ∩ (line_through C D)) 
variables (hAD_AE_EB : dist A D = dist A E ∧ dist A E = dist E B)
variables (hF_on_CE : F ∈ segment C E) (hED_CF : dist E D = dist C F)
variables (hP_bisector_AFC : ∃ P ∈ arc D A C, (angle A F C).bisector ∈ line_through A E)

-- Prove that points are concyclic
theorem concyclic_points : concyclic {A, E, F, P} :=
sorry

end concyclic_points_l600_600516


namespace maximum_black_squares_l600_600061

theorem maximum_black_squares (n : ℕ) (grid : Fin 9 × Fin 9 → Bool) :
  (∀ f : Fin 9 × Fin 9 → Bool, (∑ i, ∑ j, if f (i, j) then 1 else 0 = n → 
    ∃ r c, (f (r, c) = false ∧ f (r, c+1) = false ∧ f (r, c+2) = false ∧ f (r, c+3) = false) ∨
           (f (r, c) = false ∧ f (r+1, c) = false ∧ f (r+2, c) = false ∧ f (r+3, c) = false)))
    ↔ n = 19 :=
sorry

end maximum_black_squares_l600_600061


namespace aprils_plant_arrangement_l600_600046

theorem aprils_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 3
  let total_units := basil_plants + 1
  
  (fact total_units * fact tomato_plants = 4320) :=
by
  unfold basil_plants
  unfold tomato_plants
  unfold total_units
  apply eq.refl
  sorry

end aprils_plant_arrangement_l600_600046


namespace exists_a_div_by_3_l600_600083

theorem exists_a_div_by_3 (a : ℝ) (h : ∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 4) = k) :
  ∃ k : ℤ, a = k / 3 :=
by
  sorry

end exists_a_div_by_3_l600_600083


namespace binomial_coeff_12_3_l600_600526

/-- The binomial coefficient is defined as: 
  \binom{n}{k} = \frac{n!}{k!(n-k)!} -/
theorem binomial_coeff_12_3 : Nat.binom 12 3 = 220 := by
  sorry

end binomial_coeff_12_3_l600_600526


namespace M_inter_N_eq_l600_600641

def M : set ℝ := { y | ∃ x : ℝ, y = real.sin x }
def N : set ℕ := {0, 1, 2}

theorem M_inter_N_eq : M ∩ N = {0, 1} :=
by {
  sorry
}

end M_inter_N_eq_l600_600641


namespace power_of_complex_l600_600216

theorem power_of_complex : ((√2 / 2) + (√2 / 2) * complex.I)^4 = -1 := by
  sorry

end power_of_complex_l600_600216


namespace number_of_incorrect_inequalities_l600_600589

theorem number_of_incorrect_inequalities (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (ite (|a| > |b|) 0 1) + (ite (a < b) 0 1) + (ite (a + b < ab) 0 1) + (ite (a^3 > b^3) 0 1) = 3 :=
sorry

end number_of_incorrect_inequalities_l600_600589


namespace areas_relation_l600_600864

noncomputable def circumscribed_circle_area (r : ℝ) : ℝ := π * (r^2)

noncomputable def area_of_triangle (a b : ℝ) : ℝ := (1 / 2) * a * b

theorem areas_relation (A B r : ℝ) (h_triple: 15^2 + 20^2 = 25^2) :
  let C := (circumscribed_circle_area r) / 2 in
  let tri_area := area_of_triangle 15 20 in
  A + B + tri_area = C := 
by
  -- Definitions and problem transformations
  sorry

end areas_relation_l600_600864


namespace subsequence_sum_one_l600_600262
open MeasureTheory Topology

variable {α : Type*} [MeasurableSpace α] [TopologicalSpace α] [BorelSpace α]

def positive_measure (X : Set α) [MeasurableSet X] (μ : MeasureTheory.Measure α) := 
  μ X > 0

theorem subsequence_sum_one {X : Set ℝ} {μ : MeasureTheory.Measure ℝ}
  (hX : positive_measure X μ)
  (a_n : ℕ → ℝ) (h_bounded : ∃ A, ∀ n, |a_n n| < A - 1) :
  ∃ (x ∈ X) (y_n : ℕ → X), 
  Filter.Tendsto (λ n, ∑ i in Finset.range (n + 1), (i * (y_n i - x) - a_n i)) 
  Filter.atTop (Filter.principal {1}) :=
sorry

end subsequence_sum_one_l600_600262


namespace exist_lines_with_three_colors_l600_600557

theorem exist_lines_with_three_colors
  (P : Type) [plane P] (colors : P → fin 4)
  (h1 : ∃ p : P, colors p = 0)
  (h2 : ∃ p : P, colors p = 1)
  (h3 : ∃ p : P, colors p = 2)
  (h4 : ∃ p : P, colors p = 3) :
  ∃ ℓ : line P, (∃ p1 p2 p3 : P, p1 ∈ ℓ ∧ p2 ∈ ℓ ∧ p3 ∈ ℓ ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (colors p1 ≠ colors p2) ∧ (colors p2 ≠ colors p3) ∧ (colors p1 ≠ colors p3)) :=
sorry

end exist_lines_with_three_colors_l600_600557


namespace part1_part2_l600_600230

-- Define the parametric equations for curve C1
def parametric_C1 (α : ℝ) : ℝ × ℝ :=
  (2 * real.sqrt 5 * real.cos α, 2 * real.sin α)

-- Define the polar equation transformed for curve C2
def polar_C2 (ρ θ : ℝ) : ℝ :=
  ρ^2 + 4 * ρ * real.cos θ - 2 * ρ * real.sin θ + 4

noncomputable def standard_eq_C1 : Prop :=
  ∀ x y α : ℝ, parametric_C1 α = (x, y) → (x / (2 * real.sqrt 5))^2 + (y / 2)^2 = 1

noncomputable def standard_eq_C2 : Prop :=
  ∀ ρ θ x y : ℝ, (x = ρ * real.cos θ) → (y = ρ * real.sin θ) → polar_C2 ρ θ = 0 → 
  (x + 2)^2 + (y - 1)^2 = 1

noncomputable def length_AB : Prop :=
  ∀ x y t1 t2 : ℝ, 
  (x = -4 + (real.sqrt 2 / 2) * t1) →
  (y = (real.sqrt 2 / 2) * t1) →
  ((t1^2 - 3 * real.sqrt 2 * t1 + 4 = 0) ∧
  (t2^2 - 3 * real.sqrt 2 * t2 + 4 = 0)) →
  (|t1 - t2| = real.sqrt 2)

theorem part1 : standard_eq_C1 ∧ standard_eq_C2 :=
by {
  sorry,
}

theorem part2 : length_AB :=
by {
  sorry,
}

end part1_part2_l600_600230


namespace circle_radius_l600_600752

theorem circle_radius (r : ℝ) (h_circumference : 2 * Real.pi * r) 
                      (h_area : Real.pi * r^2) 
                      (h_equation : 3 * (2 * Real.pi * r) = Real.pi * r^2) : 
                      r = 6 :=
by
  sorry

end circle_radius_l600_600752


namespace minimize_surface_area_base_edge_length_l600_600367

noncomputable def volume (a h : ℝ) : ℝ := (sqrt 3) / 4 * a^2 * h

noncomputable def surface_area (a h : ℝ) : ℝ := 3 * a * h + (sqrt 3 / 2) * a^2

noncomputable def height (a : ℝ) : ℝ := 32 * (sqrt 3) / (3 * a^2)

theorem minimize_surface_area_base_edge_length :
  ∃ a : ℝ, volume a (height a) = 8 ∧ 
           ∀ b : ℝ, b ≠ a → surface_area a (height a) < surface_area b (height b) :=
begin
  let a := 2 * real.cbrt 4,
  use a,
  split,
  {
    -- Showing volume is 8 when a = 2 * real.cbrt 4
    sorry
  },
  {
    -- Showing surface area is minimized for this a
    sorry
  }
end

end minimize_surface_area_base_edge_length_l600_600367


namespace completing_the_square_l600_600794

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l600_600794


namespace volume_of_polyhedron_l600_600768

theorem volume_of_polyhedron (V : ℝ) (hV : 0 ≤ V) :
  ∃ P : ℝ, P = V / 6 :=
by
  sorry

end volume_of_polyhedron_l600_600768


namespace arithmetic_sequence_inequality_l600_600127

-- Condition definitions
def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

def satisfies_conditions (a : ℕ → ℝ) : Prop :=
a 2 = 5 ∧
(a 4) ^ 2 = a 1 * a 13

-- General term formula derived from conditions
def general_term (a : ℕ → ℝ) : Prop :=
∀ n, a n = 2 * n + 1

-- Condition for sum of the first n terms
def sum_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (∑ k in finset.range n, a k)

-- Required proof problem statement within Lean 4 environment
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (S : ℕ → ℝ) :
  is_arithmetic_sequence a →
  satisfies_conditions a →
  general_term a →
  sum_n_terms a S →
  (∑ k in finset.range n, 1 / S k) < 3 / 4 :=
by
  assume h1 h2 h3 h4,
  sorry

end arithmetic_sequence_inequality_l600_600127


namespace Wilson_sledding_l600_600835

variable (T S : ℕ)

theorem Wilson_sledding (h1 : S = T / 2) (h2 : (2 * T) + (3 * S) = 14) : T = 4 := by
  sorry

end Wilson_sledding_l600_600835


namespace number_of_valid_n_l600_600094

theorem number_of_valid_n : 
  {n : ℕ // n ≤ 2000 ∧ ∃ k : ℕ, 21 * n = k * k}.card = 9 := 
sorry

end number_of_valid_n_l600_600094


namespace f_neg_one_eq_five_l600_600126

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 1 else f (x + 3)

theorem f_neg_one_eq_five : f (-1) = 5 := 
by
  -- Proof goes here
  sorry

end f_neg_one_eq_five_l600_600126


namespace kaylin_age_l600_600258

theorem kaylin_age : 
  ∀ (Freyja Eli Sarah Kaylin : ℕ), 
    Freyja = 10 ∧ 
    Eli = Freyja + 9 ∧ 
    Sarah = 2 * Eli ∧ 
    Kaylin = Sarah - 5 -> 
    Kaylin = 33 :=
by
  intro Freyja Eli Sarah Kaylin
  intro h
  cases h with hF h1
  cases h1 with hE h2
  cases h2 with hS hK
  sorry

end kaylin_age_l600_600258


namespace sum_b_eq_l600_600099

-- Define the sequences a_n and b_n
def a (n : ℕ) : ℕ := n + 2
def b (n : ℕ) : ℝ := 2 / (3^n : ℝ) + n

-- State the theorem
theorem sum_b_eq :
  ∑ i in Finset.range 10, b (i + 1) = 56 - (1 / (3^10 : ℝ)) :=
by
  sorry

end sum_b_eq_l600_600099


namespace calculate_expression_l600_600500

theorem calculate_expression :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2 = 6 * real.sqrt 2) :=
by
  sorry

end calculate_expression_l600_600500


namespace min_sum_ab_max_product_ab_l600_600956

theorem min_sum_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) : a + b ≥ 2 :=
by
  sorry

theorem max_product_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : a * b ≤ 1 / 4 :=
by
  sorry

end min_sum_ab_max_product_ab_l600_600956


namespace probability_queens_attack_l600_600784

theorem probability_queens_attack:
  let total_pairs := nat.choose 64 2,
  row_pairs := 8 * (nat.choose 8 2),
  col_pairs := 8 * (nat.choose 8 2),
  diag_pairs := 2 * (1 + (nat.choose 3 2) + (nat.choose 4 2) + (nat.choose 5 2) + (nat.choose 6 2) + (nat.choose 7 2)) + (nat.choose 8 2),
  attacking_pairs := row_pairs + col_pairs + diag_pairs
  in (total_pairs = 2016) ∧ (row_pairs = 224) ∧ (col_pairs = 224) ∧ (diag_pairs = 140) ∧ (attacking_pairs = 588) →
  (attacking_pairs / total_pairs : ℚ) = 7 / 24 :=
sorry

end probability_queens_attack_l600_600784


namespace midpoint_ma_eq_mb_l600_600848

variables {A B C K L M : Type}

/-- Given a triangle ABC and a tangent at A to the circumcircle of the triangle intersects the extension of side BC past B at K. L is the midpoint of AC. M is a point on AB such that ∠AKM = ∠CKL. Prove that MA = MB. -/
theorem midpoint_ma_eq_mb 
  (h_circumcircle_tangent : tangent_at_point_circle A (circumcircle △ABC) K (BC_extended_at B))
  (h_midpoint_L : midpoint L (segment AC))
  (h_point_M : on_segment M (segment AB))
  (h_equal_angles : ∠AKM = ∠CKL) : 
  distance M A = distance M B :=
sorry

end midpoint_ma_eq_mb_l600_600848


namespace cosine_of_angle_between_diagonals_l600_600016

variable (a b : ℝ × ℝ × ℝ)

def vector_a := (3, -1, 2) : ℝ × ℝ × ℝ
def vector_b := (2, 2, -1) : ℝ × ℝ × ℝ

def sum_vectors : ℝ × ℝ × ℝ := 
  (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2, vector_a.3 + vector_b.3)

def diff_vectors : ℝ × ℝ × ℝ := 
  (vector_b.1 - vector_a.1, vector_b.2 - vector_a.2, vector_b.3 - vector_a.3)

def dot_product (x y : ℝ × ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2 + x.3 * y.3

def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def cos_theta : ℝ :=
  dot_product sum_vectors diff_vectors / (norm sum_vectors * norm diff_vectors)

theorem cosine_of_angle_between_diagonals :
  cos_theta = -5 / Real.sqrt 513 :=
sorry

end cosine_of_angle_between_diagonals_l600_600016


namespace rational_exponent_simplification_l600_600528

theorem rational_exponent_simplification :
  (- (3 + 3/8) : ℚ) ^ (-2/3 : ℚ) = (4/9 : ℚ) :=
by
  sorry

end rational_exponent_simplification_l600_600528


namespace ellipse_m_range_l600_600326

theorem ellipse_m_range (m : ℝ) 
  (h1 : m + 9 > 25 - m) 
  (h2 : 25 - m > 0) 
  (h3 : m + 9 > 0) : 
  8 < m ∧ m < 25 := 
by
  sorry

end ellipse_m_range_l600_600326


namespace masha_wins_l600_600406

def num_matches : Nat := 111

-- Define a function for Masha's optimal play strategy
-- In this problem, we'll denote both players' move range and the condition for winning.
theorem masha_wins (n : Nat := num_matches) (conditions : n > 0 ∧ n % 11 = 0 ∧ (∀ k : Nat, 1 ≤ k ∧ k ≤ 10 → ∃ new_n : Nat, n = k + new_n)) : True :=
  sorry

end masha_wins_l600_600406


namespace calculate_expression_l600_600493

theorem calculate_expression : (sqrt 27 / (sqrt 3 / 2) * (2 * sqrt 2) - (6 * sqrt 2)) = 6 * sqrt 2 :=
by
  -- Taking these steps from the solution, we should finally arrive at the required proof
  sorry

end calculate_expression_l600_600493


namespace completing_the_square_l600_600820

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l600_600820


namespace total_sum_l600_600887

theorem total_sum (p q r s t : ℝ) (P : ℝ) 
  (h1 : q = 0.75 * P) 
  (h2 : r = 0.50 * P) 
  (h3 : s = 0.25 * P) 
  (h4 : t = 0.10 * P) 
  (h5 : s = 25) 
  :
  p + q + r + s + t = 260 :=
by 
  sorry

end total_sum_l600_600887


namespace quadratic_roots_identity_l600_600947

noncomputable def a := - (2 / 5 : ℝ)
noncomputable def b := (1 / 5 : ℝ)
noncomputable def quadraticRoots := (a, b)

theorem quadratic_roots_identity :
  a + b ^ 2 = - (9 / 25 : ℝ) := 
by 
  rw [a, b]
  sorry

end quadratic_roots_identity_l600_600947


namespace sqrt_expr_simplification_l600_600511

theorem sqrt_expr_simplification :
  (real.sqrt 27) / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - (6 * real.sqrt 2) = (6 * real.sqrt 2) :=
by
  sorry

end sqrt_expr_simplification_l600_600511


namespace symmetric_point_exists_l600_600096

def point := ℝ × ℝ × ℝ

def line (x y z : ℝ) (t : ℝ) : point := (x + t, y - 0.2 * t, z + 2 * t)

def midpoint (A B : point) : point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

theorem symmetric_point_exists :
  ∀ (M M' : point) (Lx Ly Lz : ℝ),
  M = (-1, 2, 0) →
  Lx = 0.5 →
  Ly = -0.7 →
  Lz = 2 →
  (M - M') = 2 * (line Lx Ly Lz t - M') →
  M' = (-2, -3, 0) :=
by
  sorry

end symmetric_point_exists_l600_600096


namespace fourth_lowest_years_of_service_l600_600339

theorem fourth_lowest_years_of_service (years_of_service : Finset ℕ) (h_card : years_of_service.card = 8)
  (h_range : years_of_service.max' (by linarith) - years_of_service.min' (by linarith) = 14)
  (h_third_lowest : (years_of_service.sort (≤)).nth 2 = some 9) :
  ∀ (y : ℕ), y ∈ (years_of_service.sort (≤)).nth 3 → y ≥ 9 := 
by 
  sorry

end fourth_lowest_years_of_service_l600_600339


namespace minimum_a_plus_3b_l600_600647

-- Define the conditions
variables (a b : ℝ)
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_eq : a + 3 * b = 1 / a + 3 / b

-- State the theorem
theorem minimum_a_plus_3b (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a + 3 * b = 1 / a + 3 / b) : 
  a + 3 * b ≥ 4 :=
sorry

end minimum_a_plus_3b_l600_600647


namespace longer_diagonal_length_l600_600452

-- Define the properties of the rhombus
def side_length : ℝ := 40
def shorter_diagonal : ℝ := 30

-- The goal is to prove that the length of the longer diagonal is 10 * sqrt(55)
theorem longer_diagonal_length : 
  let half_shorter_diagonal := shorter_diagonal / 2,
      half_longer_diagonal := √(side_length^2 - half_shorter_diagonal^2) in
  2 * half_longer_diagonal = 10 * √55 := 
by
  sorry

end longer_diagonal_length_l600_600452


namespace solve_percentage_of_X_in_B_l600_600709

variable (P : ℝ)

def liquid_X_in_A_percentage : ℝ := 0.008
def mass_of_A : ℝ := 200
def mass_of_B : ℝ := 700
def mixed_solution_percentage_of_X : ℝ := 0.0142
def target_percentage_of_P_in_B : ℝ := 0.01597

theorem solve_percentage_of_X_in_B (P : ℝ) 
  (h1 : mass_of_A * liquid_X_in_A_percentage + mass_of_B * P = (mass_of_A + mass_of_B) * mixed_solution_percentage_of_X) :
  P = target_percentage_of_P_in_B :=
sorry

end solve_percentage_of_X_in_B_l600_600709


namespace a_mul_b_value_l600_600345

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l600_600345


namespace compare_M_N_P_l600_600977

theorem compare_M_N_P (a b c : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < c) (h₃ : c < 1)
    (M := 2^a) (N := 5^(-b)) (P := (1/7)^c) :
    M > N ∧ N > P := by
  sorry

end compare_M_N_P_l600_600977


namespace calculate_expression_l600_600494

theorem calculate_expression : (sqrt 27 / (sqrt 3 / 2) * (2 * sqrt 2) - (6 * sqrt 2)) = 6 * sqrt 2 :=
by
  -- Taking these steps from the solution, we should finally arrive at the required proof
  sorry

end calculate_expression_l600_600494


namespace circle_diameter_l600_600423

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d, d = 4 :=
by
  let r := Real.sqrt 4
  let d := 2 * r
  use d
  simp only [Real.sqrt_eq_rfl, mul_eq_zero, ne.def, not_false_iff]
  linarith
  sorry

end circle_diameter_l600_600423


namespace cube_and_sphere_path_l600_600005

noncomputable def path_length (edge_length cube_radius : ℝ) : ℝ :=
  2 * Real.sqrt (cube_radius^2 + cube_radius^2) * π

theorem cube_and_sphere_path
  (cube_edge_length sphere_radius : ℝ) :
  cube_edge_length = 2 ∧ sphere_radius = 1 →
  path_length cube_edge_length sphere_radius = 2 * Real.sqrt 2 * π :=
by
  sorry

end cube_and_sphere_path_l600_600005


namespace BCDE_is_parallelogram_l600_600671

-- Define the geometry problem setting
variables {A B C D E : Type} [is_convex_pentagon A B C D E]
variables {angle_AEB angle_CAB angle_AED angle_ACB angle_ADE : ℝ}
variables {BC DE : ℝ}

-- Define the given conditions
axiom eq1 : BC = DE
axiom eq2 : ∠ ABE = ∠ CAB = ∠ AED - 90
axiom eq3 : ∠ ACB = ∠ ADE

-- Statement to prove the polygon is a parallelogram
theorem BCDE_is_parallelogram : parallelogram B C D E := sorry

end BCDE_is_parallelogram_l600_600671


namespace find_a_harmonic_series_inequality_l600_600705

-- Define the necessary functions and assumptions
def f (x a : ℝ) := Real.exp x - a * x - 1

-- Equivalent Lean statement for Question 1
theorem find_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 0) ↔ (a = 1) := 
sorry

-- Equivalent Lean statement for Question 2
theorem harmonic_series_inequality (n : ℕ) (hn : 0 < n) :
  ∑ i in Finset.range (n + 1), 1 / (i + 1 : ℝ) > Real.log (n + 1) :=
sorry

end find_a_harmonic_series_inequality_l600_600705


namespace circle_diameter_l600_600432

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l600_600432


namespace combined_ages_l600_600732

theorem combined_ages (h_age : ℕ) (diff : ℕ) (years_later : ℕ) (hurley_age : h_age = 14) 
                       (age_difference : diff = 20) (years_passed : years_later = 40) : 
                       h_age + diff + years_later * 2 = 128 := by
  sorry

end combined_ages_l600_600732


namespace problem_statement_l600_600590

noncomputable def lg := Real.log10

theorem problem_statement (a b : ℝ) (h1 : a + lg a = 10) (h2 : b + 10^b = 10) : 
  a + b = 10 := 
sorry

end problem_statement_l600_600590


namespace completing_the_square_l600_600796

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l600_600796


namespace range_of_m_l600_600173

-- Define the parameters and conditions
def is_ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 + m * y^2 = 1
def eccentricity (m : ℝ) : ℝ := 
  if m > 1 then 
    let a := 1
    let b := 1 / m 
    let c := real.sqrt (a^2 - b^2)
    c / a
  else if 0 < m < 1 then 
    let a := 1 / m
    let b := 1 
    let c := real.sqrt (a^2 - b^2)
    c / a
  else 
    0

-- Problem statement
theorem range_of_m (m : ℝ) (h1 : 0 < m) (h2 : eccentricity m > 1 / 2) (h3 : eccentricity m < 1) :
  (0 < m ∧ m < 3/4) ∨ (4/3 < m) :=
by
  sorry

end range_of_m_l600_600173


namespace smallest_sphere_radius_l600_600318

-- Definitions and conditions directly from the problem
def radius_of_small_spheres : ℝ := 2
def bounding_box_corners : set (ℝ × ℝ × ℝ) := 
  { (x, y, z) | abs x = 2 ∧ abs y = 2 ∧ abs z = 2 }

def smallest_enclosing_sphere_radius :=
  (2 * Real.sqrt 3) + 2

-- Lean statement to prove that radius_of_enclosing_sphere == smallest_enclosing_sphere_radius
theorem smallest_sphere_radius :
  ∀ (spheres : ℝ × ℝ × ℝ → Prop),
  (∀ p, spheres p → p ∈ bounding_box_corners) →
  ∀ (r : ℝ), 
  (∀ p, spheres p → dist p (0, 0, 0) ≤ r + radius_of_small_spheres) →
  r = smallest_enclosing_sphere_radius :=
by
  intros spheres hspheres r hr
  sorry

end smallest_sphere_radius_l600_600318


namespace valid_codes_count_l600_600286

def valid_codes (her_code target_code : ℕ) : Bool :=
  let digits_match (d1 d2 : ℕ) : ℕ :=
    (if d1 = d2 then 1 else 0)
  let num_matches := digits_match (her_code / 100) (target_code / 100) +
                     digits_match ((her_code / 10) % 10) ((target_code / 10) % 10) +
                     digits_match (her_code % 10) (target_code % 10)
  if num_matches >= 2 then false else true

def transposed_code (x y z : ℕ) : Bool :=
  (x ≠ 0 ∨ y ≠ 0) ∧ (x ≠ z ∨ y ≠ z) ∧
  ((x * 100 + y * 10 + z) ≠ (y * 100 + x * 10 + z)) ∧
  ((x * 100 + y * 10 + z) ≠ (y * 100 + z * 10 + x)) ∧
  ((x * 100 + y * 10 + z) ≠ (z * 100 + x * 10 + y))

def target_codes : Finset ℕ :=
  { code : ℕ // 
    let x := code / 100
    let y := (code / 10) % 10
    let z := code % 10
    code < 125 ∧ valid_codes code 45 ∧ transposed_code x y z }

theorem valid_codes_count : target_codes.card = 110 := by
  sorry

end valid_codes_count_l600_600286


namespace sqrt_diff_inequality_l600_600592

-- Define the conditions
variables (a b : ℝ)
variables (h1 : a > b)
variables (h2 : b > 0)

-- Translate the proof problem into Lean statement
theorem sqrt_diff_inequality (ha : a > b) (hb : b > 0) : sqrt a - sqrt b < sqrt (a - b) :=
by sorry

end sqrt_diff_inequality_l600_600592


namespace value_of_a_star_b_l600_600351

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l600_600351


namespace magnitude_of_z_l600_600124

open Complex

theorem magnitude_of_z (a : ℝ) (h_imag : z.im = 2) (h_pure_imag : (z^2 + 3).re = 0) : abs z = sqrt 5 :=
by sorry

end magnitude_of_z_l600_600124


namespace line_equation_coeff_sum_l600_600327

theorem line_equation_coeff_sum (x1 y1 x2 y2 : ℝ) (h₁ : (x1, y1) = (1, 3)) (h₂ : (x2, y2) = (4, -2)) :
  let m := (y2 - y1) / (x2 - x1),
      b := y1 - m * x1
  in m + b = 3 :=
by
  have h₃ : m = (y2 - y1) / (x2 - x1),
  use (y2 - y1) / (x2 - x1)
  have h₄ : b = y1 - m * x1,
  use y1 - m * x1 
  -- Here would be the proof steps, but use sorry
  sorry

end line_equation_coeff_sum_l600_600327


namespace arrangment_ways_basil_tomato_l600_600044

theorem arrangment_ways_basil_tomato (basil_plants tomato_plants : Finset ℕ) 
  (hb : basil_plants.card = 5) 
  (ht : tomato_plants.card = 3) 
  : (∃ total_ways : ℕ, total_ways = 4320) :=
by
  sorry

end arrangment_ways_basil_tomato_l600_600044


namespace number_of_correct_propositions_l600_600899

theorem number_of_correct_propositions : 
    (∀ a b : ℝ, a < b → ¬ (a^2 < b^2)) ∧ 
    (∀ a : ℝ, (∀ x : ℝ, |x + 1| + |x - 1| ≥ a ↔ a ≤ 2)) ∧ 
    (¬ (∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0) → 
    1 = 1 := 
by
  sorry

end number_of_correct_propositions_l600_600899


namespace find_C_and_area_of_ABC_l600_600664

noncomputable theory

-- Define the conditions given in the problem
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides opposite to A, B, C respectively

-- Given conditions
axiom condition1 : c = 2
axiom condition2 : 2 * Real.sin(A) = Real.sqrt 3 * a * Real.cos(C)
axiom condition3 : 2 * Real.sin(2 * A) + Real.sin(2 * B + C) = Real.sin(C)

-- Proof of the angle and area
theorem find_C_and_area_of_ABC (htrig : (Real.angle (Real.two_π) A B C = π)) :
  C = π / 3 ∧ (1/2) * b * 2 = (2 * Real.sqrt 3) / 3 :=
sorry
 
end find_C_and_area_of_ABC_l600_600664


namespace a_mul_b_value_l600_600347

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l600_600347


namespace special_operation_value_l600_600353

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l600_600353


namespace tank_capacity_75_l600_600871

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end tank_capacity_75_l600_600871


namespace find_smallest_value_l600_600269

noncomputable def smallest_possible_value (a b c : ℤ) (ξ : ℂ) : ℝ :=
  complex.abs (a + b * ξ + c * ξ^3)

theorem find_smallest_value (a b c : ℤ) (ξ : ℂ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_ξ4 : ξ^4 = 1) (h_ξ1 : ξ ≠ 1) : smallest_possible_value a b c ξ = 1 :=
sorry

end find_smallest_value_l600_600269


namespace factorize_x_cubed_minus_9x_l600_600930

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l600_600930


namespace circles_are_externally_tangent_l600_600149

-- Define the first circle C1 as x^2 + y^2 = 1
def circle_C1 (x y : ℝ) := x^2 + y^2 = 1

-- Define the second circle C2 as x^2 + y^2 - 8y + 7 = 0
def circle_C2 (x y : ℝ) := x^2 + y^2 - 8*y + 7 = 0

-- Coordinates of the centers of C1 and C2
def center_C1 : ℝ × ℝ := (0, 0)
def center_C2 : ℝ × ℝ := (0, 4)

-- Define the radii of C1 and C2
def radius_C1 : ℝ := 1
def radius_C2 : ℝ := 3

-- Define the distance between the centers of C1 and C2
def distance_centers : ℝ := Real.sqrt ((0 - 0)^2 + (4 - 0)^2)

-- Define the sum of the radii of C1 and C2
def sum_radii : ℝ := radius_C1 + radius_C2

-- Prove that |C1C2| = R + r implying circles are externally tangent
theorem circles_are_externally_tangent :
  distance_centers = sum_radii :=
by
  sorry

end circles_are_externally_tangent_l600_600149


namespace machine_work_rates_l600_600777

theorem machine_work_rates :
  (∃ x : ℝ, (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2)) = 1 / x ∧ x = 1 / 2) :=
by
  sorry

end machine_work_rates_l600_600777


namespace ben_examined_7_trays_l600_600905

open Int

def trays_of_eggs (total_eggs : ℕ) (eggs_per_tray : ℕ) : ℕ := total_eggs / eggs_per_tray

theorem ben_examined_7_trays : trays_of_eggs 70 10 = 7 :=
by
  sorry

end ben_examined_7_trays_l600_600905


namespace integral_correct_l600_600559

noncomputable def integral_problem := ∫ x in 0..1, (2 + Real.sqrt (1 - x^2))

theorem integral_correct : integral_problem = (Real.pi / 4) + 2 :=
  sorry

end integral_correct_l600_600559


namespace base3_to_base10_conversion_l600_600531

theorem base3_to_base10_conversion : 
  1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0 = 142 :=
by {
  -- calculations
  sorry
}

end base3_to_base10_conversion_l600_600531


namespace imaginary_part_of_z_l600_600170

def z : ℂ := 15 * Complex.I / (3 + 4 * Complex.I)

theorem imaginary_part_of_z : z.im = 9 / 5 := by
  sorry

end imaginary_part_of_z_l600_600170


namespace limit_at_neg3_l600_600481

noncomputable def limit_function : Prop := 
  limit (λ x : ℝ, (x^2 + 2 * x - 3) / (x^3 + 4 * x^2 + 3 * x)) (-3) (-2/3)

theorem limit_at_neg3 : limit_function := 
  sorry

end limit_at_neg3_l600_600481


namespace deleted_files_l600_600290

variable {initial_files : ℕ}
variable {files_per_folder : ℕ}
variable {folders : ℕ}

noncomputable def files_deleted (initial_files files_in_folders : ℕ) : ℕ :=
  initial_files - files_in_folders

theorem deleted_files (h1 : initial_files = 27) (h2 : files_per_folder = 6) (h3 : folders = 3) :
  files_deleted initial_files (files_per_folder * folders) = 9 :=
by
  sorry

end deleted_files_l600_600290


namespace point_Q_and_d_l600_600761

theorem point_Q_and_d :
  ∃ (a b c d : ℝ),
    (∀ x y z : ℝ, (x - 2)^2 + (y - 3)^2 + (z + 4)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) ∧
    (8 * a - 6 * b + 32 * c = d) ∧ a = 6 ∧ b = 0 ∧ c = 12 ∧ d = 151 :=
by
  existsi 6, 0, 12, 151
  sorry

end point_Q_and_d_l600_600761


namespace part1_b1_b2_part1_general_formula_part2_sum_20_l600_600139

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then a n + 1 else a n + 2

def b (n : ℕ) : ℕ := a (2 * n)

-- Proving b_1 = 2 and b_2 = 5
theorem part1_b1_b2 : b 1 = 2 ∧ b 2 = 5 :=
by {
  unfold b a,
  simp,
  split,
  {
    rfl -- proof for b_1 = 2
  },
  {
    rfl -- proof for b_2 = 5
  }
}

-- Proving the general formula for b_n
theorem part1_general_formula (n : ℕ) : b n = 3 * n - 1 :=
by {
  induction n with k ih,
  {
    unfold b a,
    simp,
    rfl
  },
  {
    rename ih ih_k,
    unfold b a,
    simp,
    rw [ih_k],
    calc 3 * (k + 1) - 1 = 3 * k + 3 - 1 : by ring
                       ... = 3 * k + 2     : by ring
                       ... = a (2 * k + 2) : by sorry -- Detailed proof needed
  }
}

-- Proving the sum of the first 20 terms of the sequence a_n
theorem part2_sum_20 : (Finset.range 20).sum a = 300 :=
by {
  unfold a,
  have h1 : finset.sum (finset.range 10) (λ n, 3 * n + 1) = 145,
  {
    sorry -- Compute sum of odd terms
  },
  have h2 : finset.sum (finset.range 10) (λ n, 3 * n + 2) = 155,
  {
    sorry -- Compute sum of even terms
  },
  have h3 : finset.sum (finset.range 20) a = 145 + 155,
  {
    sorry -- Combine sums
  },
  exact h3,
}

end part1_b1_b2_part1_general_formula_part2_sum_20_l600_600139


namespace proof_problem_l600_600156

noncomputable def question (α : ℝ) : ℝ :=
  (1 + 2 * Mathlib.sin α * Mathlib.cos α) / (Mathlib.sin α ^ 2 - Mathlib.cos α ^ 2)

theorem proof_problem (α : ℝ) (h : Mathlib.tan α = -1/2) : 
  question α = -1/3 := 
  sorry

end proof_problem_l600_600156


namespace combined_new_weight_oranges_apples_l600_600867

def initial_orange_weight : ℝ := 5
def initial_apple_weight : ℝ := 3
def orange_water_percentage_initial : ℝ := 0.95
def apple_water_percentage_initial : ℝ := 0.90
def orange_water_loss_percentage : ℝ := 0.07
def apple_water_loss_percentage : ℝ := 0.04

/-- Proof that the combined new weight of oranges and apples is 7.5595 kilograms --/
theorem combined_new_weight_oranges_apples :
  let orange_water_initial := initial_orange_weight * orange_water_percentage_initial,
      apple_water_initial := initial_apple_weight * apple_water_percentage_initial,
      orange_pulp_initial := initial_orange_weight - orange_water_initial,
      apple_pulp_initial := initial_apple_weight - apple_water_initial,
      orange_water_lost := orange_water_loss_percentage * orange_water_initial,
      apple_water_lost := apple_water_loss_percentage * apple_water_initial,
      orange_water_new := orange_water_initial - orange_water_lost,
      apple_water_new := apple_water_initial - apple_water_lost,
      orange_weight_new := orange_pulp_initial + orange_water_new,
      apple_weight_new := apple_pulp_initial + apple_water_new,
      combined_weight_new := orange_weight_new + apple_weight_new
  in combined_weight_new = 7.5595 := sorry

end combined_new_weight_oranges_apples_l600_600867


namespace slope_of_line_tangent_to_circle_l600_600653

variable (k : ℝ)

theorem slope_of_line_tangent_to_circle :
  ∀ (k : ℝ),
  ∃ (x y : ℝ),
  (x - 1) * k = y ∧             -- Line equation y = k(x - 1)
  ((x - 4)^2 + y^2 = 4) ∧       -- Circle equation (x - 4)^2 + y^2 = 4
  ((4k - k) / (real.sqrt (k^2 + 1)) = 2) →  -- Distance from center (4, 0) to the line is 2
  k = (2 * real.sqrt 5) / 5 ∨ k = -(2 * real.sqrt 5) / 5 := sorry

end slope_of_line_tangent_to_circle_l600_600653


namespace sum_of_integral_c_with_rational_roots_l600_600102

theorem sum_of_integral_c_with_rational_roots
  (h : ∀ c : ℤ, c ≤ 30 → ∃ k : ℤ, k^2 = 81 + 4*c ∧ ∃ p q : ℚ, p*q = c ∧ p + q = 9) :
  ∑ c in (Finset.filter (λ c : ℤ, ∃ k : ℤ, k^2 = 81 + 4*c) (Finset.Icc (-20) 30)), c = -28 :=
by
  sorry

end sum_of_integral_c_with_rational_roots_l600_600102


namespace number_of_true_props_l600_600547

-- Define the conditions and propositions
def prop1 : Prop := ∃ x : ℝ, x^2 + x - 1 < 0
def prop2 : Prop := ∀ x : ℝ, x^2 + x - 1 > 0 → False
def line_perpendicular (m : ℝ) : Prop := 
  (m = -1 ∨ m = 0) → 
  (∀ x y : ℝ, (m * x + (2 * m - 1) * y + 1 = 0) ∧ (3 * x + m * y + 3 = 0))

def prop3 (x y : ℝ) (p : x ≠ y) (q : sin x ≠ sin y) : Prop := 
  q → p ∧ (p ∧ q → False)

def increasing_property (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f (x + 1) > f x) ↔ (∀ x y : ℝ, x < y → f x < f y)

-- The statement that needs to be proved
theorem number_of_true_props : 
  (¬prop1 ∧ prop2 (0 : ℝ) ∧ prop3 0 1 (0 ≠ 1) (sin 0 ≠ sin 1) ∧ ¬increasing_property int.floor) ->
  (∃ n : ℕ, n = 2) :=
by
  sorry

end number_of_true_props_l600_600547


namespace validate_conclusions_l600_600762

def conclusion_correctness 
  (OI_Fresh OI_Deli OI_Dairy OI_DailyNecessities OI_Other : ℝ)
  (NP_Fresh NP_Deli NP_Dairy NP_DailyNecessities NP_Other : ℝ)
  (Total_OPR : ℝ) : Prop :=
  let conclusion_1 := OI_Deli == min OI_Deli OI_Fresh,
                       OI_Dairy, 
                       OI_DailyNecessities, 
                       OI_Other
  let conclusion_2 := NP_Fresh > (1/2) * (NP_Fresh 
                       + NP_Deli 
                       + NP_Dairy
                       + NP_DailyNecessities 
                       + NP_Other)
  let conclusion_3 := (NP_DailyNecessities / OI_DailyNecessities) * 
                      Total_OPR > max ((NP_Fresh / OI_Fresh) 
                      * Total_OPR ,
                      (NP_Deli / OI_Deli) 
                      * Total_OPR ,
                      (NP_Dairy / OI_Dairy) 
                      * Total_OPR,
                      (NP_Other / OI_Other) 
                      * Total_OPR )
  let conclusion_4 := (NP_Fresh / OI_Fresh) 
                      * Total_OPR > 0.4
  conclusion_2 ∧ conclusion_3 ∧ conclusion_4

theorem validate_conclusions:
  conclusion_correctness 0.486 0.158 0.201 0.108 0.047 0.658 (-0.043) 0.165 0.202 0.018 0.325 :=
by {
  sorry
}

end validate_conclusions_l600_600762


namespace coefficient_xy2_l600_600957

theorem coefficient_xy2 (a : ℝ) (h : a = ∫ x in 0..(real.pi / 2), sin x + cos x) :
  (coeff (expand((1 + a * x)^6 * (1 + y)^4)) x y^2  = 72) := 
by
  have ha : a = 2 := sorry
  have h_poly : polynomial_expansion a := sorry
  show (coeff (expand((1 + a * x)^6 * (1 + y)^4)) x y^2 = 72), from sorry
end

end coefficient_xy2_l600_600957


namespace exists_not_odd_l600_600208

variable (f : ℝ → ℝ)

-- Define the condition that f is not an odd function
def not_odd_function := ¬ (∀ x : ℝ, f (-x) = -f x)

-- Lean statement to prove the correct answer
theorem exists_not_odd (h : not_odd_function f) : ∃ x : ℝ, f (-x) ≠ -f x :=
sorry

end exists_not_odd_l600_600208


namespace distance_problem_l600_600446

theorem distance_problem (x y n : ℝ) (h1 : y = 15) (h2 : Real.sqrt ((x - 2) ^ 2 + (15 - 7) ^ 2) = 13) (h3 : x > 2) :
  n = Real.sqrt ((2 + Real.sqrt 105) ^ 2 + 15 ^ 2) := by
  sorry

end distance_problem_l600_600446


namespace evaluate_expression_l600_600079

theorem evaluate_expression :
  (4 * 6) / (12 * 14) * ((8 * 12 * 14) / (4 * 6 * 8)) = 1 := 
by 
  sorry

end evaluate_expression_l600_600079


namespace find_a_and_k_sigma_b_n_l600_600147

noncomputable def arithmetic_seq (a n : ℕ) : ℕ :=
  if n = 1 then a - 1 else if n = 2 then 4 else if n = 3 then 2 * a else sorry

noncomputable def S (a d k : ℕ) : ℕ :=
  k * (a - 1) + ((k * (k - 1)) / 2) * d

noncomputable def b (S n : ℕ) : ℕ :=
  S / n

theorem find_a_and_k (a k : ℕ) (H1 : S a 2 k = 30) : 
  a = 3 ∧ k = 5 := 
begin
  sorry
end

theorem sigma_b_n (a n : ℕ) (H2 : ∀ k : ℕ, k < n + 1 → b (S a 2 k) k = k + 1) :
  (Σ k in range(n + 1), b (S a 2 k) k) = (n * (n + 3)) / 2 := 
begin
  sorry
end

end find_a_and_k_sigma_b_n_l600_600147


namespace avg_price_per_book_l600_600300

theorem avg_price_per_book (n1 n2 p1 p2 : ℕ) (h1 : n1 = 65) (h2 : n2 = 55) (h3 : p1 = 1380) (h4 : p2 = 900) :
    (p1 + p2) / (n1 + n2) = 19 := by
  sorry

end avg_price_per_book_l600_600300


namespace product_eq_one_of_abs_log_eq_l600_600178

theorem product_eq_one_of_abs_log_eq (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := 
sorry

end product_eq_one_of_abs_log_eq_l600_600178


namespace train_speed_is_correct_l600_600888

-- Define the length of the train and the time to cross the post as constants
def train_length : ℝ := 350
def crossing_time : ℝ := 15

-- Define the speed of the train according to the conditions given
def train_speed : ℝ := train_length / crossing_time

-- The hypothesis stating the expected speed of the train
def expected_speed : ℝ := 23.33

-- The theorem to be proven
theorem train_speed_is_correct : train_speed = expected_speed := 
by
  sorry

end train_speed_is_correct_l600_600888


namespace count_valid_natural_numbers_l600_600089

theorem count_valid_natural_numbers (n : ℕ) (h : n = 454500) :
  (finset.filter (λ k : ℕ, (k * (k - 1)) % 505 = 0) 
  (finset.range (n + 1))).card = 3600 :=
by
  sorry

end count_valid_natural_numbers_l600_600089


namespace simplify_expression_l600_600486

-- Define the expression to be simplified
def expression : ℝ := (sqrt 27 / (sqrt 3 / 2)) * (2 * sqrt 2) - 6 * sqrt 2

-- State the theorem to be proven
theorem simplify_expression : expression = 6 * sqrt 2 :=
by
  sorry

end simplify_expression_l600_600486


namespace range_of_m_l600_600659

theorem range_of_m (m : ℚ) :
  (∀ x : ℝ, mx^2 - mx - 1 < 2x^2 - 2x) ↔ (-2 < m ∧ m <= 2) :=
by
  sorry

end range_of_m_l600_600659


namespace seniors_prefer_physical_books_l600_600224

/--
In a survey done by the community library, the table below shows a partially filled record of people who prefer physical books over e-books. 

Given the conditions:
- Total people preferring physical books: 180
- Adults preferring physical books: 80
- Seniors preferring e-books: 130
- Total people preferring e-books: 200

Prove that the number of seniors preferring physical books is 100.
-/
theorem seniors_prefer_physical_books (total_physical : ℕ) (adults_physical : ℕ) (seniors_ebooks : ℕ) (total_ebooks : ℕ)
  (h1 : total_physical = 180) (h2 : adults_physical = 80) (h3 : seniors_ebooks = 130) (h4 : total_ebooks = 200) :
  ∃ seniors_physical : ℕ, seniors_physical = total_physical - adults_physical ∧ seniors_physical = 100 :=
by
  use (total_physical - adults_physical)
  split
  · exact rfl
  · rw [h1, h2]
    norm_num
    sorry -- Complete the proof which is straightforward arithmetic

end seniors_prefer_physical_books_l600_600224


namespace cost_price_is_1000_l600_600399

variable (C : ℝ) (S : ℝ) (C_new : ℝ) (S_new : ℝ)

-- Define the conditions:
def condition1 : Prop := S = 1.05 * C
def condition2 : Prop := C_new = 0.95 * C
def condition3 : Prop := S_new = 1.10 * C_new ∧ S_new = S - 5

-- Define the main theorem to prove the cost price is 1000:
theorem cost_price_is_1000 (h1 : condition1) (h2 : condition2) (h3 : condition3) : C = 1000 :=
by
  sorry

end cost_price_is_1000_l600_600399


namespace circle_diameter_l600_600425

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d, d = 4 :=
by
  let r := Real.sqrt 4
  let d := 2 * r
  use d
  simp only [Real.sqrt_eq_rfl, mul_eq_zero, ne.def, not_false_iff]
  linarith
  sorry

end circle_diameter_l600_600425


namespace chord_lengths_l600_600903

noncomputable def d : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def s : ℝ := (-1 + Real.sqrt 5) / 2

theorem chord_lengths (d s : ℝ) (cond1 : d - s = 1) (cond2 : d * s = 1) : 
  (d - s = 1) ∧ (d * s = 1) ∧ ((d^2 - s^2) = Real.sqrt 5) :=
by
  have d_def : d = (1 + Real.sqrt 5) / 2 := rfl
  have s_def : s = (-1 + Real.sqrt 5) / 2 := rfl
  exact ⟨cond1, cond2, sorry⟩

end chord_lengths_l600_600903


namespace variance_of_X_is_correct_l600_600372

/-!
  There is a batch of products, among which there are 12 genuine items and 4 defective items.
  If 3 items are drawn with replacement, and X represents the number of defective items drawn,
  prove that the variance of X is 9 / 16 given that X follows a binomial distribution B(3, 1 / 4).
-/

noncomputable def variance_of_binomial : Prop :=
  let n := 3
  let p := 1 / 4
  let variance := n * p * (1 - p)
  variance = 9 / 16

theorem variance_of_X_is_correct : variance_of_binomial := by
  sorry

end variance_of_X_is_correct_l600_600372


namespace stratified_sample_A_eq_30_l600_600772

-- Define the conditions
variables {students_A students_B students_C sample_size total_population : ℕ}

-- The conditions given in the problem
def condition_1 := students_A = 3600
def condition_2 := students_B = 5400
def condition_3 := students_C = 1800
def condition_4 := sample_size = 90
def condition_5 := total_population = students_A + students_B + students_C

-- The proportion of the sample size to the total population
def sample_proportion := (sample_size: ℚ) / total_population

-- The expected number of students to be drawn from School A using stratified sampling
def expected_sample_A := (students_A: ℚ) * sample_proportion

-- A theorem stating that the expected number of students from School A is 30
theorem stratified_sample_A_eq_30 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) :
  expected_sample_A = 30 := sorry

end stratified_sample_A_eq_30_l600_600772


namespace pairwise_products_same_digit_l600_600778

theorem pairwise_products_same_digit
  (a b c : ℕ)
  (h_ab : a % 10 ≠ b % 10)
  (h_ac : a % 10 ≠ c % 10)
  (h_bc : b % 10 ≠ c % 10)
  : (a * b % 10 = a * c % 10) ∧ (a * b % 10 = b * c % 10) :=
  sorry

end pairwise_products_same_digit_l600_600778


namespace sqrt_expression_eq_l600_600507

theorem sqrt_expression_eq :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2) = 6 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_eq_l600_600507


namespace investor_profits_l600_600892

/-- Problem: Given the total contributions and profit sharing conditions, calculate the amount 
    each investor receives. -/

theorem investor_profits :
  ∀ (A_contribution B_contribution C_contribution D_contribution : ℝ) 
    (A_profit B_profit C_profit D_profit : ℝ) 
    (total_capital total_profit : ℝ),
    total_capital = 100000 → 
    A_contribution = B_contribution + 5000 →
    B_contribution = C_contribution + 10000 →
    C_contribution = D_contribution + 5000 →
    total_profit = 60000 →
    A_profit = (35 / 100) * total_profit * (1 + 10 / 100) →
    B_profit = (30 / 100) * total_profit * (1 + 8 / 100) →
    C_profit = (20 / 100) * total_profit * (1 + 5 / 100) → 
    D_profit = (15 / 100) * total_profit →
    (A_profit = 23100 ∧ B_profit = 19440 ∧ C_profit = 12600 ∧ D_profit = 9000) :=
by
  intros
  sorry

end investor_profits_l600_600892


namespace diameter_of_circle_l600_600421

theorem diameter_of_circle (A : ℝ) (h : A = 4 * real.pi) : ∃ d : ℝ, d = 4 :=
  sorry

end diameter_of_circle_l600_600421


namespace Diane_bakes_160_gingerbreads_l600_600549

-- Definitions
def trays1Count : Nat := 4
def gingerbreads1PerTray : Nat := 25
def trays2Count : Nat := 3
def gingerbreads2PerTray : Nat := 20

def totalGingerbreads : Nat :=
  (trays1Count * gingerbreads1PerTray) + (trays2Count * gingerbreads2PerTray)

-- Problem statement
theorem Diane_bakes_160_gingerbreads :
  totalGingerbreads = 160 := by
  sorry

end Diane_bakes_160_gingerbreads_l600_600549


namespace walter_equal_share_l600_600251

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end walter_equal_share_l600_600251


namespace completing_square_l600_600808

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l600_600808


namespace linear_function_properties_l600_600701

noncomputable def f : ℤ → ℤ := λ x, 4 * x - 17

def is_geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem linear_function_properties
  (f : ℤ → ℤ)
  (h1 : ∀ x, f x = 4 * x - 17)
  (h2 : is_geometric_sequence (f 2) (f 5) (f 4))
  (h3 : f 8 = 15) :
  (∀ x, f x = 4 * x - 17) ∧ 
  (∀ n, ∑ i in finset.range n.succ, f (2 * i) = 4 * n * n - 13 * n) :=
sorry

end linear_function_properties_l600_600701


namespace find_b_l600_600639

theorem find_b (a b c y1 y2 : ℝ) (h1 : y1 = a * 2^2 + b * 2 + c) 
              (h2 : y2 = a * (-2)^2 + b * (-2) + c) 
              (h3 : y1 - y2 = -12) : b = -3 :=
by 
  sorry

end find_b_l600_600639


namespace find_a1_l600_600989

theorem find_a1 (S : ℕ → ℝ) (T : ℕ → ℝ)
  (a : ℕ → ℝ)
  (h1 : S 3 = a 2 + 4 * a 1)
  (h2 : T 5 = 243)
  (h3 : ∀ n, a n = a 1 * (3 : ℝ)^(n-1))
  (h4 : T n = ∏ i in finset.range n, a (i+1)) :
  a 1 = 1 := by
  sorry

end find_a1_l600_600989


namespace factorize_x_cube_minus_9x_l600_600926

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l600_600926


namespace part1_part2_l600_600633

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x^2 - 3*x + 2)

theorem part1 (h₁ : a = 1) : ∀ x : ℝ, (0 < x ∧ x < 1/2 ∨ x > 1 → f x 1 > f (1 / 2) 1) ∧ (1 / 2 < x ∧ x < 1 → f x 1 < f 1 1) :=
sorry

theorem part2 (h₂ : ∀ x > 1, f x a ≥ 0) : a ≤ 1 :=
sorry

end part1_part2_l600_600633


namespace max_principals_in_10_years_l600_600075

theorem max_principals_in_10_years : ∀ term_length num_years,
  (term_length = 4) ∧ (num_years = 10) →
  ∃ max_principals, max_principals = 3
:=
  by intros term_length num_years h
     sorry

end max_principals_in_10_years_l600_600075


namespace fixed_point_exists_l600_600987

variable {A : ℝ × ℝ}
variable {B : ℝ × ℝ}
variable {F : ℝ × ℝ}
variable {M : ℝ × ℝ}
variable {N : ℝ × ℝ}
variable {Q : ℝ × ℝ}
variable {p : ℝ}
variable {C : ℝ → ℝ}

noncomputable def parabola (p : ℝ) : ℝ → ℝ := fun x => real.sqrt (2 * p * x)

noncomputable def area_triangle (A B F : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (F.2 - A.2) - (F.1 - A.1) * (B.2 - A.2))

theorem fixed_point_exists
  (hA : A = (-1, 0))
  (hB : B = (1, -1))
  (hF : F = (p / 2, 0))
  (hArea : area_triangle A B F = 1) :
  ∃!(P : ℝ × ℝ), ∀ M N Q, (C := parabola 2) ∧ (C M.1 = M.2) ∧ (C N.1 = N.2) ∧ (C Q.1 = Q.2)
  ∧ ∃ slope : ℝ, ∀ line_slope : ℝ, line_slope ≠ slope ∧ line_slope M.1 ≠ M.2 ∧ Q.1 ≠ N.1 
  ∧ nq := ensure_fixed_point ((Q.2 - N.2) / (Q.1 - N.1))
  hypothesis :
  P = (1, -4) :=
sorry

end fixed_point_exists_l600_600987


namespace b_n_formula_sum_first_20_terms_l600_600130

-- Definition of sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a n + 1 else a n + 2

-- Definition of sequence b_n as a 2n-th term of a_n
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof problem 1: General formula for b_n
theorem b_n_formula (n : ℕ) : b n = 3 * n - 1 :=
sorry

-- Sum of the first 20 terms of the sequence a_n
theorem sum_first_20_terms : (Finset.range 20).sum a = 300 :=
sorry

end b_n_formula_sum_first_20_terms_l600_600130


namespace ab_operation_l600_600342

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l600_600342


namespace circumcircle_pqr_midpoint_l600_600602

theorem circumcircle_pqr_midpoint (ABC : Triangle) (H : Point)
  (D E F Q R P : Point) :
  isAcuteTriangle ABC →
  isOrthocenter H ABC →
  isFoot H.b BC D →
  isFoot H.c CA E →
  isFoot H.a AB F →
  lineThrough D (parallelTo EF) →
  lineIntersects D (parallelTo EF) (side CA) Q →
  lineIntersects D (parallelTo EF) (side AB) R →
  lineIntersects EF (side BC) P →
  circumcircle PQR (midpoint BC) :=
by
  sorry

end circumcircle_pqr_midpoint_l600_600602


namespace neg_proposition_equiv_l600_600281

theorem neg_proposition_equiv :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) :=
by
  sorry

end neg_proposition_equiv_l600_600281


namespace height_of_bars_in_3D_bar_chart_represents_frequency_l600_600667

theorem height_of_bars_in_3D_bar_chart_represents_frequency
  {A B C D : Prop}
  (hA : A = "The frequency of each categorical variable")
  (hB : B = "The percentage of the categorical variable")
  (hC : C = "The sample size of the categorical variable")
  (hD : D = "The specific value of the categorical variable")
  (hQuestion : ∀ (x : Prop), x = A ∨ x = B ∨ x = C ∨ x = D) :
  ∃ answer : Prop, answer = A :=
by sorry

end height_of_bars_in_3D_bar_chart_represents_frequency_l600_600667


namespace binom_12_3_equal_220_l600_600522

theorem binom_12_3_equal_220 : Nat.choose 12 3 = 220 := by sorry

end binom_12_3_equal_220_l600_600522


namespace sin_alpha_value_l600_600167

variable (α : ℝ)

-- Define the condition that the terminal side of angle α intersects the unit circle at the specified point.
def point_of_intersection_condition : Prop :=
  ∃ α, sin α = sin (11 * π / 6) ∧ cos α = cos (11 * π / 6)

-- State the theorem to be proven.
theorem sin_alpha_value (h : point_of_intersection_condition α) : sin α = sqrt 3 / 2 :=
sorry

end sin_alpha_value_l600_600167


namespace arithmetic_sequence_sixtieth_term_l600_600329

theorem arithmetic_sequence_sixtieth_term (a₁ a₂₁ a₆₀ d : ℕ) 
  (h1 : a₁ = 7)
  (h2 : a₂₁ = 47)
  (h3 : a₂₁ = a₁ + 20 * d) : 
  a₆₀ = a₁ + 59 * d := 
  by
  have HD : d = 2 := by 
    rw [h1] at h3
    rw [h2] at h3
    linarith
  rw [HD]
  rw [h1]
  sorry

end arithmetic_sequence_sixtieth_term_l600_600329


namespace limit_of_differentiable_at_l600_600202

variable {α : Type*} [NormedField α] [NormedSpace ℝ α] [CompleteSpace α]
variables {f : ℝ → α} {x₀ : ℝ}

theorem limit_of_differentiable_at (h : DifferentiableAt ℝ f x₀) :
  tendsto (λ Δx : ℝ, (f (x₀ - Δx) - f x₀) / (2 * Δx)) (𝓝 0) (𝓝 (-1/2 * (fderiv ℝ f x₀ 1))) :=
sorry

end limit_of_differentiable_at_l600_600202


namespace four_digit_integers_count_l600_600194

-- Define conditions of the problem
def valid_first_digit (d : ℕ) : Prop := d = 2 ∨ d = 6 ∨ d = 9
def valid_second_digit (d : ℕ) : Prop := d = 2 ∨ d = 6 ∨ d = 9
def valid_third_digit (d : ℕ) : Prop := d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8
def valid_fourth_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 7

-- Prove the number of valid 4-digit integers is 135
theorem four_digit_integers_count : 
  let count := 9 * 5 * 3 in 
  count = 135 :=
by 
  sorry

end four_digit_integers_count_l600_600194


namespace cyclists_meet_l600_600783

noncomputable def cyclist_speeds := 7
noncomputable def cyclist_speeds := 8
noncomputable def circumference := 180
noncomputable def time := 12

theorem cyclists_meet : 
  (7 * 12 + 8 * 12 = 180) :=
by {
  calc
    7 * 12 + 8 * 12 = 84 + 96 : by ring
    ... = 180 : by norm_num
}

end cyclists_meet_l600_600783


namespace probability_prime_sum_gt_20_is_zero_l600_600555

open Nat

def primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum_gt_20 (a b : ℕ) : Prop :=
  (a + b > 20) ∧ Prime (a + b)

theorem probability_prime_sum_gt_20_is_zero :
  (∃ (p1 p2 : ℕ) (h1 : p1 ∈ primes) (h2 : p2 ∈ primes) (h3 : p1 ≠ p2), is_prime_sum_gt_20 p1 p2) → false :=
by sorry

end probability_prime_sum_gt_20_is_zero_l600_600555


namespace abs_fraction_square_lt_four_iff_l600_600946

theorem abs_fraction_square_lt_four_iff (x : ℝ) : (| (8 - x) / 4 | ^ 2 < 4) ↔ (0 < x ∧ x < 16) :=
sorry

end abs_fraction_square_lt_four_iff_l600_600946


namespace sqrt_expr_simplification_l600_600508

theorem sqrt_expr_simplification :
  (real.sqrt 27) / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - (6 * real.sqrt 2) = (6 * real.sqrt 2) :=
by
  sorry

end sqrt_expr_simplification_l600_600508


namespace sum_inequality_l600_600123

noncomputable def a (k : ℕ) : ℝ := 1 / ((k + 1) * real.sqrt k)
noncomputable def b (n : ℕ) : ℝ := 2 * (1 - 1 / real.sqrt (n + 1))

theorem sum_inequality (n : ℕ) : 
  (∑ k in finset.range (n + 1).filter (λ k, k ≠ 0), a k) < b n := 
sorry

end sum_inequality_l600_600123


namespace derek_savings_l600_600543

-- Defining the initial amount of money Derek has in January
def initial_amount_in_january (P : ℕ) : Prop :=
  let final_amount := P * 2^11 in
  final_amount = 4096

-- The statement we need to prove
theorem derek_savings : ∃ P : ℕ, initial_amount_in_january P ∧ P = 2 :=
by
  sorry

end derek_savings_l600_600543


namespace f_even_l600_600159

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even (a : ℝ) (h1 : is_even f) (h2 : ∀ x, -1 ≤ x ∧ x ≤ a) : f a = 2 :=
  sorry

end f_even_l600_600159


namespace expand_expression_l600_600562

theorem expand_expression (x : ℝ) : 
  (11 * x^2 + 5 * x - 3) * (3 * x^3) = 33 * x^5 + 15 * x^4 - 9 * x^3 :=
by 
  sorry

end expand_expression_l600_600562


namespace smallest_domain_of_g_l600_600387

noncomputable def g : ℕ → ℕ 
| x := match x % 2 with
  | 0 => x / 2
  | _ => 3 * x + 1

def g_domain : set ℕ := 
  {x | ∃ y, y ≠ 0 ∧ (λ y, ∃ n, (nat.iterate g n 12) = x) y}

theorem smallest_domain_of_g : g_domain.finite ∧ g_domain.to_finset.card = 19 :=
by sorry

end smallest_domain_of_g_l600_600387


namespace proof_N_square_numbers_l600_600690

theorem proof_N_square_numbers (N : ℕ) (h : ∃ E : set ℕ, E.card = N * N ∧
  (∀ a ∈ E, a > 1 → ∃ F : set ℕ, F ⊆ E ∧ F.card = a - 1 ∧ (∀ x ∈ F, x < a ∧ a % x = 0))
  ∧ ∀ x ∈ E, x = 1 ∨ x > 1) : N = 1 := 
by sorry

end proof_N_square_numbers_l600_600690


namespace jihyung_pickup_school_supply_l600_600252

theorem jihyung_pickup_school_supply : 
  ∀ (pencils erasers : ℕ), 
  pencils = 2 → erasers = 4 → pencils + erasers = 6 :=
begin
  intros pencils erasers hp he,
  rw [hp, he],
  norm_num,
end

end jihyung_pickup_school_supply_l600_600252


namespace special_operation_value_l600_600355

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l600_600355


namespace permutation_count_is_1385_l600_600196

def permutations := {x : List ℕ // x ~ List.range 8}

noncomputable def count_permutations_satisfying_conditions : ℕ :=
  -- Function to count the required permutations
  sorry

theorem permutation_count_is_1385 :
  count_permutations_satisfying_conditions = 1385 := by
  sorry

end permutation_count_is_1385_l600_600196


namespace product_of_differences_l600_600407

-- Define the context where x and y are real numbers
variables (x y : ℝ)

-- State the theorem to be proved
theorem product_of_differences (x y : ℝ) : 
  (-x + y) * (-x - y) = x^2 - y^2 :=
sorry

end product_of_differences_l600_600407


namespace largest_distance_l600_600789

-- Define the points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def dist (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

-- Conditions
def S1_center : Point3D := {x := -3, y := -15, z := 10}
def S2_center : Point3D := {x := 7, y := 3, z := -19}
def S1_radius : ℝ := 23
def S2_radius : ℝ := 95

-- Question with correct answer (without steps)
theorem largest_distance :
  let d := dist S1_center S2_center in
  d = Real.sqrt 1265 → 
  ∃ A B : Point3D, dist S1_center A = S1_radius ∧ dist S2_center B = S2_radius ∧ dist A B = 118 + Real.sqrt 1265 :=
by
  intro d hd
  use (S1_center, S2_center)  -- example points (though not the actual points A and B)
  sorry -- proof omitted

end largest_distance_l600_600789


namespace reflection_composition_rotation_l600_600298

-- Definitions of geometric objects
variables {R : Type*} {V : Type*} [inner_product_space R V] [finite_dimensional R V]

-- Assuming l is a line, P1 and P2 are planes intersecting along line l
variable (l : submodule R V)
variable (P1 P2 : submodule R V)

-- Theorem to be proven
theorem reflection_composition_rotation (hP1l: l ≤ P1) (hP2l: l ≤ P2) 
  (h_intersect_angle: ∃ θ : ℝ, ∀ x ∈ l, ∃ y ∈ P1, ∃ z ∈ P2, y ≠ z ∧ angle y z = θ) :
  ∃ θ : ℝ, ∀ x ∈ P1, ∀ y ∈ P2, reflection_about_plane P1 (reflection_about_plane P2 x) = rotate_about_line l (2 * θ) :=
sorry

end reflection_composition_rotation_l600_600298


namespace find_x_l600_600014

theorem find_x :
  ∃ x : ℝ, x = (1/x) * (-x) - 3*x + 4 ∧ x = 3/4 :=
by
  sorry

end find_x_l600_600014


namespace largest_integral_value_of_y_l600_600571

theorem largest_integral_value_of_y : 
  (1 / 4 : ℝ) < (y / 7 : ℝ) ∧ (y / 7 : ℝ) < (3 / 5 : ℝ) → y ≤ 4 :=
by
  sorry

end largest_integral_value_of_y_l600_600571


namespace completing_the_square_l600_600817

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l600_600817


namespace cos_angle_F1PF2_l600_600152

theorem cos_angle_F1PF2 
  (a b c : ℝ)
  (h0 : a > b)
  (h1 : b > 0)
  (h2 : c = Real.sqrt (a^2 - b^2))
  (P : ℝ × ℝ)
  (ellipse_cond : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (area_cond : ∃ (A : ℝ), A = (P.1 - -c)*(P.2 - 0) / 2 ∨ A = (P.1 - c)*(P.2 - 0) / 2 ∧ A = (Real.sqrt 2 / 2 * b^2)) :
  cos (Real.arctan2 P.2 (P.1 + c) - Real.arctan2 P.2 (P.1 - c)) = 1 / 3 :=
sorry

end cos_angle_F1PF2_l600_600152


namespace diane_bakes_gingerbreads_l600_600551

open Nat

theorem diane_bakes_gingerbreads :
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  total_gingerbreads1 + total_gingerbreads2 = 160 := 
by
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  exact Eq.refl (total_gingerbreads1 + total_gingerbreads2)

end diane_bakes_gingerbreads_l600_600551


namespace binom_12_3_equal_220_l600_600524

theorem binom_12_3_equal_220 : Nat.choose 12 3 = 220 := by sorry

end binom_12_3_equal_220_l600_600524


namespace walter_equal_share_l600_600250

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end walter_equal_share_l600_600250


namespace valid_triples_complete_l600_600710

-- Define the base multiple in degrees
def epsilon := 18

-- Define a structure for angle triples
structure AngleTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  angle_sum : a + b + c = 10
  is_multiple_of_epsilon : ∀ x ∈ [a, b, c], ∃ k, x = k

-- Define the list of valid angle triples
def valid_angle_triples : List (ℕ × ℕ × ℕ) :=
  [(1, 1, 8), (1, 2, 7), (1, 3, 6), (1, 4, 5),
   (2, 2, 6), (2, 3, 5), (2, 4, 4), (3, 3, 4)]

-- The main theorem
theorem valid_triples_complete :
  ∀ t : AngleTriple, 
    (t.a, t.b, t.c) ∈ valid_angle_triples :=
  by contradiction
     sorry

end valid_triples_complete_l600_600710


namespace xiao_ming_expected_profit_l600_600395

noncomputable def expectedProfit : real :=
  let ticketCost := 10
  let winProb := 0.02
  let winPrize := 300
  let lossProb := 0.98
  
  (winPrize - ticketCost) * winProb + (-ticketCost) * lossProb

theorem xiao_ming_expected_profit : expectedProfit = -4 := by sorry

end xiao_ming_expected_profit_l600_600395


namespace combined_girls_avg_Lincoln_boys_girls_avg_l600_600904

-- Define parameters representing the number of students
variable (J j L l : ℕ)

-- Given conditions in terms of equations
def Jefferson_combined_avg := (68 * J + 73 * j) / (J + j) = 70
def Combined_boys_avg := (68 * J + 78 * L) / (J + L) = 76

-- Define proofs or conditions using given equations and simplification assumptions
theorem combined_girls_avg (h1 : Jefferson_combined_avg J j) (h2 : Combined_boys_avg J L) (hj2 : J = L) (hj3 : j = j) : 
  (73 * j + 85 * j) / (j + j) = 79 :=
sorry

theorem Lincoln_boys_girls_avg (h2 : Combined_boys_avg J L) (hj2 : J = L) (hj3 : j = j) : 
  (78 * J + 85 * (2 * J / 3)) / (J + 2 * J / 3) = 80.8 :=
sorry

end combined_girls_avg_Lincoln_boys_girls_avg_l600_600904


namespace cost_of_each_box_of_cereal_l600_600312

theorem cost_of_each_box_of_cereal
  (total_groceries_cost : ℝ)
  (gallon_of_milk_cost : ℝ)
  (number_of_cereal_boxes : ℕ)
  (banana_cost_each : ℝ)
  (number_of_bananas : ℕ)
  (apple_cost_each : ℝ)
  (number_of_apples : ℕ)
  (cookie_cost_multiplier : ℝ)
  (number_of_cookie_boxes : ℕ) :
  total_groceries_cost = 25 →
  gallon_of_milk_cost = 3 →
  number_of_cereal_boxes = 2 →
  banana_cost_each = 0.25 →
  number_of_bananas = 4 →
  apple_cost_each = 0.5 →
  number_of_apples = 4 →
  cookie_cost_multiplier = 2 →
  number_of_cookie_boxes = 2 →
  (total_groceries_cost - (gallon_of_milk_cost + (banana_cost_each * number_of_bananas) + 
                           (apple_cost_each * number_of_apples) + 
                           (number_of_cookie_boxes * (cookie_cost_multiplier * gallon_of_milk_cost)))) / 
  number_of_cereal_boxes = 3.5 := 
sorry

end cost_of_each_box_of_cereal_l600_600312


namespace product_gcd_lcm_l600_600907

open Nat

theorem product_gcd_lcm: gcd 15 (gcd 10 6) * lcm 15 (lcm 10 6) = 30 := by
  sorry

end product_gcd_lcm_l600_600907


namespace factorization_l600_600081

variable (m n : ℤ)

theorem factorization : 2 * m * n - 6 * m = 2 * m * (n - 3) :=
by
  sorry

end factorization_l600_600081


namespace total_insects_l600_600775

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) (caterpillars_every_third_leaf : ℕ) :
  leaves = 84 →
  ladybugs_per_leaf = 139 →
  ants_per_leaf = 97 →
  caterpillars_every_third_leaf = 53 →
  (84 * 139) + (84 * 97) + (53 * (84 / 3)) = 21308 := 
by
  sorry

end total_insects_l600_600775


namespace age_difference_l600_600366

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 :=
by
  sorry

end age_difference_l600_600366


namespace find_m_l600_600188

variable (m : ℝ)
def A := {2, m}
def B := {1, m^2}

theorem find_m (h : A ∪ B = {1, 2, 3, 9}) : m = 3 := 
by
  /- start of proof -/
  sorry

end find_m_l600_600188


namespace iron_balls_count_l600_600901

-- Conditions
def length_bar := 12  -- in cm
def width_bar := 8    -- in cm
def height_bar := 6   -- in cm
def num_bars := 10
def volume_iron_ball := 8  -- in cubic cm

-- Calculate the volume of one iron bar
def volume_one_bar := length_bar * width_bar * height_bar

-- Calculate the total volume of the ten iron bars
def total_volume := volume_one_bar * num_bars

-- Calculate the number of iron balls
def num_iron_balls := total_volume / volume_iron_ball

-- The proof statement
theorem iron_balls_count : num_iron_balls = 720 := by
  sorry

end iron_balls_count_l600_600901


namespace cyclic_quadrilateral_theorem_l600_600278

noncomputable def cyclic_quadrilateral (A B C D : Type) := sorry

open cyclic_quadrilateral

theorem cyclic_quadrilateral_theorem (A B C D : Type) (AB CD BC AD AC BD : ℝ)
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : ∃ K : Type, K ∈ [A, C] ∧ ∠(A B K) = ∠(D B C)) :
  AB * CD + BC * AD = AC * BD := sorry

end cyclic_quadrilateral_theorem_l600_600278


namespace find_n_l600_600073

theorem find_n :
  ∃ (n : ℕ), (0 < n) ∧ ((n+3)! + (n+1)! = n! * 728) ∧ (n = 8) :=
by
  sorry

end find_n_l600_600073


namespace possible_degrees_of_remainder_l600_600791

open Polynomial

theorem possible_degrees_of_remainder (i : ℂ) :
  let d := degree (3 * X ^ 3 + -2 * X ^ 2 + (i : mv_polynomial ℂ ℂ) * X + 6) in
  d = 3 →
  ∀ (f : Polynomial ℂ), 
  ∃ (r : Polynomial ℂ), degree r ≤ d - 1 ∧ 
    (∃ q : Polynomial ℂ, f = q * (3 * X ^ 3 + -2 * X ^ 2 + (i : mv_polynomial ℂ ℂ) * X + 6) + r) :=
begin
  sorry
end

end possible_degrees_of_remainder_l600_600791


namespace overtime_hours_proof_l600_600440

-- Define the conditions
variable (regular_pay_rate : ℕ := 3)
variable (regular_hours : ℕ := 40)
variable (overtime_multiplier : ℕ := 2)
variable (total_pay : ℕ := 180)

-- Calculate the regular pay for 40 hours
def regular_pay : ℕ := regular_pay_rate * regular_hours

-- Calculate the extra pay received beyond regular pay
def extra_pay : ℕ := total_pay - regular_pay

-- Calculate overtime pay rate
def overtime_pay_rate : ℕ := overtime_multiplier * regular_pay_rate

-- Calculate the number of overtime hours
def overtime_hours (extra_pay : ℕ) (overtime_pay_rate : ℕ) : ℕ :=
  extra_pay / overtime_pay_rate

-- The theorem to prove
theorem overtime_hours_proof :
  overtime_hours extra_pay overtime_pay_rate = 10 := by
  sorry

end overtime_hours_proof_l600_600440


namespace ab_operation_l600_600343

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l600_600343


namespace solve_for_x_l600_600191

set_option pp.explicit true

def vector (a b : ℝ) := (a, b)

def norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem solve_for_x (x : ℝ) :
  let a := vector x 1
  let b := vector (x + 2) 1
  norm (a.1 + b.1, a.2 + b.2) = norm (a.1 - b.1, a.2 - b.2) → x = -1 :=
by
  intros
  sorry

end solve_for_x_l600_600191


namespace can_choose_P_l600_600004

-- Define the objects in the problem,
-- types, constants, and assumptions as per the problem statement.

theorem can_choose_P (cube : ℝ) (P Q R S T A B C D : ℝ)
  (edge_length : cube = 10)
  (AR_RB_eq_CS_SB : ∀ AR RB CS SB, (AR / RB = 7 / 3) ∧ (CS / SB = 7 / 3))
  : ∃ P, 2 * (Q - R) = (P - Q) + (R - S) := by
  sorry

end can_choose_P_l600_600004


namespace positive_diff_40_x_l600_600478

theorem positive_diff_40_x
  (x : ℝ)
  (h : (40 + x + 15) / 3 = 35) :
  abs (x - 40) = 10 :=
sorry

end positive_diff_40_x_l600_600478


namespace tangent_line_through_point_l600_600627

noncomputable def tangent_line_count : ℕ :=
  let P := (2, 2)
  let f := λ x : ℝ, 3 * x - x^3
  let f' := λ x : ℝ, 3 - 3 * x^2
  let is_tangent (a : ℝ) := 3 * a - a^3 - (3 - 3 * a^2) * a = 2 - (3 * a - a^3)
  in
    set.count {a : ℝ | is_tangent a} sorry

theorem tangent_line_through_point (f tangent_line_count : ℕ) : tangent_line_count = 3 :=
  sorry

end tangent_line_through_point_l600_600627


namespace fractions_of_grades_are_A_l600_600218

theorem fractions_of_grades_are_A :
  ∃ (T : ℕ) (F : ℚ), T ≈ 500 ∧ (F * T + 1/4 * T + 1/2 * T + 25 = T) ∧ F ≈ 1/5 :=
by
  sorry

end fractions_of_grades_are_A_l600_600218


namespace compare_abc_l600_600959

def a : ℝ := Real.log 2 / Real.log 3
def b : ℝ := Real.log 2
def c : ℝ := 5 ^ (-1 / 2 : ℝ)

theorem compare_abc : c < a ∧ a < b :=
  by
    sorry

end compare_abc_l600_600959


namespace convert_base_3_to_base_10_l600_600533

theorem convert_base_3_to_base_10 : 
  (1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) = 142 :=
by
  sorry

end convert_base_3_to_base_10_l600_600533


namespace calculate_expression_l600_600499

theorem calculate_expression :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2 = 6 * real.sqrt 2) :=
by
  sorry

end calculate_expression_l600_600499


namespace part_I_part_II_part_III_l600_600185

open Nat

noncomputable def a : ℕ+ → ℝ 
| 1 := 1
| (n + 1) := (Real.sqrt (a n + 1) - 1) / 2

theorem part_I (n : ℕ+) : a n > 0 :=
sorry

theorem part_II (n : ℕ+) : a n / (a n + 4) < a (n + 1) ∧ a (n + 1) < a n / 4 :=
sorry

theorem part_III (n : ℕ+) : 3 / (4 ^ n) < a n ∧ a n ≤ 1 / (4 ^ (n - 1)) :=
sorry

end part_I_part_II_part_III_l600_600185


namespace find_integer_l600_600662

noncomputable def least_possible_sum (x y z k : ℕ) : Prop :=
  2 * x = 5 * y ∧ 5 * y = 6 * z ∧ x + k + z = 26

theorem find_integer (x y z : ℕ) (h : least_possible_sum x y z 6) :
  6 = (26 - x - z) :=
  by {
    sorry
  }

end find_integer_l600_600662


namespace simplify_expression_l600_600487

-- Define the expression to be simplified
def expression : ℝ := (sqrt 27 / (sqrt 3 / 2)) * (2 * sqrt 2) - 6 * sqrt 2

-- State the theorem to be proven
theorem simplify_expression : expression = 6 * sqrt 2 :=
by
  sorry

end simplify_expression_l600_600487


namespace count_ways_to_sum_420_l600_600228

def is_valid_k (k : ℕ) : Prop :=
  (k > 1) ∧ (420 % k = 0) ∧ (k % 2 = 1 ∨ k = 2)

noncomputable def is_valid_n (k : ℕ) : ℕ → Prop :=
  ∃ n : ℕ, (n ≥ 1) ∧ (420 = k * n + (k * (k - 1)) / 2)

theorem count_ways_to_sum_420 :
  (finset.filter is_valid_k (finset.range (420 + 1))).card = 5 := sorry

end count_ways_to_sum_420_l600_600228


namespace sqrt_expression_eq_l600_600503

theorem sqrt_expression_eq :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2) = 6 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_eq_l600_600503


namespace sum_quotient_bounds_l600_600742

theorem sum_quotient_bounds (n : ℕ) (a : ℕ → ℕ)
  (h₁ : 5 ≤ n)
  (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ q_i : ℕ, a i = q_i * a ((i + 1) % n) + a ((i - 1 + n) % n))
  : 2 * n ≤ (Finset.range n).sum (λ i, (a ((i - 1 + n) % n) + a ((i + 1) % n)) / a i) ∧
    (Finset.range n).sum (λ i, (a ((i - 1 + n) % n) + a ((i + 1) % n)) / a i) < 3 * n :=
by
  sorry

end sum_quotient_bounds_l600_600742


namespace value_of_a_star_b_l600_600348

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l600_600348


namespace smaller_circle_radius_l600_600000

theorem smaller_circle_radius (r : ℝ) (h : r < 1 ∧ r > 0) :
  (∀ (c : ℕ), c = 7 → r < 1/2 → false) :=
by
  -- assume we have a large circle of radius 1 covered by 7 smaller circles
  assume r radius_of_smaller_circles
  /- begin proof -/
  sorry

end smaller_circle_radius_l600_600000


namespace real_solutions_l600_600084

noncomputable def given_eq := 
  λ x : ℝ, ((x - 1) * (x - 3) * (x - 5) * (x - 6) * (x - 3) * (x - 1))
   / ((x - 3) * (x - 6) * (x - 3))

theorem real_solutions :
  {x : ℝ | given_eq x = 2 ∧ x ≠ 3 ∧ x ≠ 6} = {2 + Real.sqrt 2, 2 - Real.sqrt 2} :=
by
  sorry

end real_solutions_l600_600084


namespace max_n_increasing_seq_l600_600359

theorem max_n_increasing_seq (a : ℕ → ℕ) (h₁ : ∀ n, a n < a (n + 1))
  (h₂ : a 0 ≥ 3) (h₃ : (List.range n).sum a = 100) : n ≤ 11 :=
by
  simp at h₃
  sorry

end max_n_increasing_seq_l600_600359


namespace maximum_k_l600_600708

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

theorem maximum_k (x : ℝ) (h₀ : x > 0) (k : ℤ) (a := 1) (h₁ : (x - k) * f_prime x a + x + 1 > 0) : k = 2 :=
sorry

end maximum_k_l600_600708


namespace sum_of_g_9_l600_600273

def f (x : ℝ) := x^2 - 8 * x + 23
def g (y : ℝ) := 3 * y + 4

theorem sum_of_g_9 : 
  let values_of_x := {x | f x = 9} in
  let possible_values_g := {g 9 | x ∈ values_of_x} in
  ∑ y in possible_values_g, y = 35 := 
by
  sorry

end sum_of_g_9_l600_600273


namespace sqrt_expression_eq_l600_600504

theorem sqrt_expression_eq :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2) = 6 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_eq_l600_600504


namespace special_operation_value_l600_600352

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l600_600352


namespace common_ratio_geom_series_l600_600918

theorem common_ratio_geom_series 
  (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 4 / 7) 
  (h₂ : a₂ = 20 / 21) :
  ∃ r : ℚ, r = 5 / 3 ∧ a₂ / a₁ = r := 
sorry

end common_ratio_geom_series_l600_600918


namespace program_output_l600_600036

theorem program_output {a b c : ℕ} (initial_a : a = 2) (initial_b : b = 3) (initial_c : c = 4) :
    let a := b,
    let b := c,
    let c := a
  in a = 3 ∧ b = 4 ∧ c = 3 := by
  sorry

end program_output_l600_600036


namespace circle_radius_l600_600753

theorem circle_radius (r : ℝ) (h_circumference : 2 * Real.pi * r) 
                      (h_area : Real.pi * r^2) 
                      (h_equation : 3 * (2 * Real.pi * r) = Real.pi * r^2) : 
                      r = 6 :=
by
  sorry

end circle_radius_l600_600753


namespace parabola_and_midpoint_problem_l600_600163

open Real

-- Define the given conditions and problem

def focus_of_parabola_coincides_with_hyperbola_focus (p : ℝ) (hp : p > 0) : Prop :=
  let focus_hyperbola := (3 : ℝ)
  let focus_parabola := (p / 2 : ℝ)
  focus_parabola = focus_hyperbola

def midpoint_distance_to_directrix (x1 x2 : ℝ) (p : ℝ) (hp : p > 0) (hAB : abs (x1 - x2) = 8) : ℝ :=
  let midpoint := (x1 + x2) / 2
  let directrix := -p / 2
  abs (midpoint - directrix)

-- The Lean proof statement of the equivalent problem

theorem parabola_and_midpoint_problem :
  ∀ (p : ℝ) (hp : p > 0) (x1 x2 : ℝ),
  focus_of_parabola_coincides_with_hyperbola_focus p hp →
  abs (x1 - x2) = 8 →
  (y^2 = 12 * x ∧ midpoint_distance_to_directrix x1 x2 p hp |AB|  = 4) :=
by
  sorry

end parabola_and_midpoint_problem_l600_600163


namespace sequence_product_mod_five_l600_600573

theorem sequence_product_mod_five : 
  let seq := List.range 20 |>.map (λ k => 10 * k + 3)
  seq.prod % 5 = 1 := 
by
  sorry

end sequence_product_mod_five_l600_600573


namespace passengers_on_plane_l600_600910

variables (P : ℕ) (fuel_per_mile : ℕ := 20) (fuel_per_person : ℕ := 3) (fuel_per_bag : ℕ := 2)
variables (num_crew : ℕ := 5) (bags_per_person : ℕ := 2) (trip_distance : ℕ := 400)
variables (total_fuel : ℕ := 106000)

def total_people := P + num_crew
def total_bags := bags_per_person * total_people
def total_fuel_per_mile := fuel_per_mile + fuel_per_person * P + fuel_per_bag * total_bags
def total_trip_fuel := trip_distance * total_fuel_per_mile

theorem passengers_on_plane : total_trip_fuel = total_fuel → P = 33 := 
by
  sorry

end passengers_on_plane_l600_600910


namespace cell_division_after_three_hours_l600_600417

theorem cell_division_after_three_hours : (2 ^ 6) = 64 := by
  sorry

end cell_division_after_three_hours_l600_600417


namespace pigeonhole_divisible_l600_600214

theorem pigeonhole_divisible (n : ℕ) (a : Fin (n + 1) → ℕ) (h : ∀ i, 1 ≤ a i ∧ a i ≤ 2 * n) :
  ∃ i j, i ≠ j ∧ a i ∣ a j :=
by
  sorry

end pigeonhole_divisible_l600_600214


namespace cone_height_l600_600013

theorem cone_height (R r h : ℝ) (π : ℝ := Real.pi) :
  let Vsphere := (4 / 3) * π * R^3,
      Vcone := (1 / 3) * π * r^2 * h,
      l := 3 * r,
      S := π * r * l,
      s := π * r^2,
      rsq := h^2 / 8
  in (R = Real.cbrt 16) →
     (S = 3 * s) →
     (Vsphere = Vcone) →
     h = 8 :=
by
  intros
  sorry

end cone_height_l600_600013


namespace fraction_is_one_fifth_l600_600860

theorem fraction_is_one_fifth (f : ℚ) (h1 : f * 50 - 4 = 6) : f = 1 / 5 :=
by
  sorry

end fraction_is_one_fifth_l600_600860


namespace work_done_together_l600_600855

theorem work_done_together
    (fraction_work_left : ℚ)
    (A_days : ℕ)
    (B_days : ℚ) :
    A_days = 20 →
    fraction_work_left = 2 / 3 →
    4 * (1 / 20 + 1 / B_days) = 1 / 3 →
    B_days = 30 := 
by
  intros hA hfrac heq
  sorry

end work_done_together_l600_600855


namespace dragons_total_games_l600_600476

theorem dragons_total_games (y x : ℕ) (h1 : x = 60 * y / 100) (h2 : (x + 8) = 55 * (y + 11) / 100) : y + 11 = 50 :=
by
  sorry

end dragons_total_games_l600_600476


namespace maximum_sum_of_triangle_l600_600306

theorem maximum_sum_of_triangle (a b c d e f : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) (h4 : d = 13) (h5 : e = 14) (h6 : f = 15) :
  3 * (max a c e) = 39 :=
by
  sorry

end maximum_sum_of_triangle_l600_600306


namespace find_interval_solution_l600_600938

def interval_solution : Set ℝ := {x | 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) <= 7}

theorem find_interval_solution (x : ℝ) :
  x ∈ interval_solution ↔
  x ∈ Set.Ioc (49 / 20 : ℝ) (14 / 5 : ℝ) := 
sorry

end find_interval_solution_l600_600938


namespace completing_the_square_l600_600793

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l600_600793


namespace analytic_expression_and_symmetry_l600_600164

noncomputable def f (A : ℝ) (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem analytic_expression_and_symmetry {A ω φ : ℝ}
  (hA : A > 0) 
  (hω : ω > 0)
  (h_period : ∀ x, f A ω φ (x + 2) = f A ω φ x)
  (h_max : f A ω φ (1 / 3) = 2) :
  (f 2 π (π / 6) = fun x => 2 * Real.sin (π * x + π / 6)) ∧
  (∃ k : ℤ, k = 5 ∧ (1 / 3 + k = 16 / 3) ∧ (21 / 4 ≤ 1 / 3 + ↑k) ∧ (1 / 3 + ↑k ≤ 23 / 4)) :=
  sorry

end analytic_expression_and_symmetry_l600_600164


namespace earnings_correct_l600_600720

def price_8inch : ℝ := 5
def price_12inch : ℝ := 2.5 * price_8inch
def price_16inch : ℝ := 3 * price_8inch
def price_20inch : ℝ := 4 * price_8inch
def price_24inch : ℝ := 5.5 * price_8inch

noncomputable def earnings_monday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 1 * price_16inch + 2 * price_20inch + 1 * price_24inch

noncomputable def earnings_tuesday : ℝ :=
  5 * price_8inch + 1 * price_12inch + 4 * price_16inch + 2 * price_24inch

noncomputable def earnings_wednesday : ℝ :=
  4 * price_8inch + 3 * price_12inch + 3 * price_16inch + 1 * price_20inch

noncomputable def earnings_thursday : ℝ :=
  2 * price_8inch + 2 * price_12inch + 2 * price_16inch + 1 * price_20inch + 3 * price_24inch

noncomputable def earnings_friday : ℝ :=
  6 * price_8inch + 4 * price_12inch + 2 * price_16inch + 2 * price_20inch

noncomputable def earnings_saturday : ℝ :=
  1 * price_8inch + 3 * price_12inch + 3 * price_16inch + 4 * price_20inch + 2 * price_24inch

noncomputable def earnings_sunday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 4 * price_16inch + 3 * price_20inch + 1 * price_24inch

noncomputable def total_earnings : ℝ :=
  earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday + earnings_saturday + earnings_sunday

theorem earnings_correct : total_earnings = 1025 := by
  -- proof goes here
  sorry

end earnings_correct_l600_600720


namespace sum_of_digits_1_to_50_l600_600396

theorem sum_of_digits_1_to_50 : 
  let digits := (list.join (list.map (λ n => (n.to_char_list.filter (λ c => c.is_digit)).map (λ c => c.to_nat - '0'.to_nat)) (list.range' 1 50)))
  in digits.sum = 330 := 
by 
  sorry

end sum_of_digits_1_to_50_l600_600396


namespace lcm_division_l600_600264

open Nat

-- Define the LCM function for a list of integers
def list_lcm (l : List Nat) : Nat := l.foldr (fun a b => Nat.lcm a b) 1

-- Define the sequence ranges
def range1 := List.range' 20 21 -- From 20 to 40 inclusive
def range2 := List.range' 41 10 -- From 41 to 50 inclusive

-- Define P and Q
def P : Nat := list_lcm range1
def Q : Nat := Nat.lcm P (list_lcm range2)

-- The theorem statement
theorem lcm_division : (Q / P) = 55541 := by
  sorry

end lcm_division_l600_600264


namespace circle_diameter_l600_600422

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d, d = 4 :=
by
  let r := Real.sqrt 4
  let d := 2 * r
  use d
  simp only [Real.sqrt_eq_rfl, mul_eq_zero, ne.def, not_false_iff]
  linarith
  sorry

end circle_diameter_l600_600422


namespace dihedral_minus_solid_equals_expression_l600_600299

-- Definitions based on the conditions provided.
noncomputable def sumDihedralAngles (P : Polyhedron) : ℝ := sorry
noncomputable def sumSolidAngles (P : Polyhedron) : ℝ := sorry
def numFaces (P : Polyhedron) : ℕ := sorry

-- Theorem statement we want to prove.
theorem dihedral_minus_solid_equals_expression (P : Polyhedron) :
  sumDihedralAngles P - sumSolidAngles P = 2 * Real.pi * (numFaces P - 2) :=
sorry

end dihedral_minus_solid_equals_expression_l600_600299


namespace sarah_friends_apples_l600_600301

-- Definitions of initial conditions
def initial_apples : ℕ := 25
def left_apples : ℕ := 3
def apples_given_teachers : ℕ := 16
def apples_eaten : ℕ := 1

-- Theorem that states the number of friends who received apples
theorem sarah_friends_apples :
  (initial_apples - left_apples - apples_given_teachers - apples_eaten = 5) :=
by
  sorry

end sarah_friends_apples_l600_600301


namespace binom_12_3_equal_220_l600_600523

theorem binom_12_3_equal_220 : Nat.choose 12 3 = 220 := by sorry

end binom_12_3_equal_220_l600_600523


namespace tan_alpha_plus_pi_over_four_eq_neg_three_minus_two_sqrt_two_l600_600984

theorem tan_alpha_plus_pi_over_four_eq_neg_three_minus_two_sqrt_two
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : cos (2 * α) + (cos α) ^ 2 = 0) :
  tan (α + π / 4) = -3 - 2 * sqrt 2 := 
  sorry

end tan_alpha_plus_pi_over_four_eq_neg_three_minus_two_sqrt_two_l600_600984


namespace ab_operation_l600_600340

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l600_600340


namespace combined_area_of_removed_triangles_l600_600024

theorem combined_area_of_removed_triangles :
  let x : ℕ := 16
  let side_length : ℕ := 20
  let area_of_one_triangle (x : ℕ) := (1 / 2) * x^2
  let combined_area (x : ℕ) := 4 * area_of_one_triangle x
  combined_area x = 512 :=
by
  let x : ℕ := 16
  let side_length : ℕ := 20
  let area_of_one_triangle x := (1 / 2) * (x^2 : ℕ)
  let combined_area x := 4 * area_of_one_triangle x
  have h : area_of_one_triangle 16 = (1 / 2) * (16^2 : ℕ) := sorry
  have hg : combined_area 16 = 4 * ((1 / 2) * (16^2 : ℕ)) := sorry
  exact (by linarith : 4 * ((1 / 2) * (16^2 : ℕ)) = 512) sorry

end combined_area_of_removed_triangles_l600_600024


namespace find_a_l600_600704

def lambda : Set ℝ := { x | ∃ (a b : ℤ), x = a + b * Real.sqrt 3 }

theorem find_a (a : ℤ) (x : ℝ)
  (h1 : x = 7 + a * Real.sqrt 3)
  (h2 : x ∈ lambda)
  (h3 : (1 / x) ∈ lambda) :
  a = 4 ∨ a = -4 :=
sorry

end find_a_l600_600704


namespace find_x_l600_600540

def operation (a b : ℝ) : ℝ :=
  if a > b then 3 / (a - b)
  else b / (b - a)

theorem find_x (x : ℝ) : operation 2 x = 3 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end find_x_l600_600540


namespace arithmetic_sequence_tenth_term_l600_600736

theorem arithmetic_sequence_tenth_term :
  ∀ (a : ℚ) (a_20 : ℚ) (a_10 : ℚ),
    a = 5 / 11 →
    a_20 = 9 / 11 →
    a_10 = a + (9 * ((a_20 - a) / 19)) →
    a_10 = 1233 / 2309 :=
by
  intros a a_20 a_10 h_a h_a_20 h_a_10
  sorry

end arithmetic_sequence_tenth_term_l600_600736


namespace amy_minimum_disks_l600_600040

theorem amy_minimum_disks :
  ∃ (d : ℕ), (d = 19) ∧ ( ∀ (f : ℕ), 
  (f = 40) ∧ ( ∀ (n m k : ℕ), 
  (n + m + k = f) ∧ ( ∀ (a b c : ℕ),
  (a = 8) ∧ (b = 15) ∧ (c = (f - a - b))
  ∧ ( ∀ (size_a size_b size_c : ℚ),
  (size_a = 0.6) ∧ (size_b = 0.55) ∧ (size_c = 0.45)
  ∧ ( ∀ (disk_space : ℚ),
  (disk_space = 1.44)
  ∧ ( ∀ (x y z : ℕ),
  (x = n * ⌈size_a / disk_space⌉) 
  ∧ (y = m * ⌈size_b / disk_space⌉) 
  ∧ (z = k * ⌈size_c / disk_space⌉)
  ∧ (x + y + z = d)) ∧ (size_a * a + size_b * b + size_c * c ≤ disk_space * d)))))) := sorry

end amy_minimum_disks_l600_600040


namespace hyperbola_focal_length_l600_600166

theorem hyperbola_focal_length (m : ℝ) (h_eq : m * x^2 + 2 * y^2 = 2) (h_imag_axis : -2 / m = 4) : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 := 
sorry

end hyperbola_focal_length_l600_600166


namespace all_values_satisfying_f_five_times_eq_x_l600_600066

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x < 0.5 then 2 * x else 2 * x - 1

-- Define the composition function applied five times
def f_five_times (x : ℝ) : ℝ :=
  f (f (f (f (f x))))

-- The statement to be proven
theorem all_values_satisfying_f_five_times_eq_x :
  ∀ x : ℝ, f_five_times x = x ↔ x ∈ {0, 1 / 31, 2 / 31, ..., 30 / 31, 1} :=
sorry

end all_values_satisfying_f_five_times_eq_x_l600_600066


namespace no_local_minimum_f_g_zeros_product_gt_e_squared_l600_600282

-- Problem 1:
theorem no_local_minimum_f (f : ℝ → ℝ) (a : ℝ) (b : ℝ) (x1 x2 : ℝ) (hx1 : x1 = 1 / real.e) (hx2 : x2 = real.e) :
  f = λ x, 2 * real.log x - (1 / 2) * x^2 →
  ¬∃ x ∈ set.Icc x1 x2, 
    (∀ ε > 0, ∃ δ > 0, ∀ y ∈ set.Icc (x - δ) (x + δ), f y ≥ f x) := sorry

-- Problem 2:
theorem g_zeros_product_gt_e_squared (k x1 x2 : ℝ) (hk : ∀ x, real.log x - k * x = 0 ∧ x1 ≠ x2 ) :
  x1 * x2 > real.exp 2 := sorry

end no_local_minimum_f_g_zeros_product_gt_e_squared_l600_600282


namespace complete_the_square_l600_600825

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l600_600825


namespace tank_capacity_l600_600874

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end tank_capacity_l600_600874


namespace bisecting_chord_line_eqn_l600_600942

theorem bisecting_chord_line_eqn :
  ∀ (x1 y1 x2 y2 : ℝ),
  y1 ^ 2 = 16 * x1 →
  y2 ^ 2 = 16 * x2 →
  (x1 + x2) / 2 = 2 →
  (y1 + y2) / 2 = 1 →
  ∃ (a b c : ℝ), a = 8 ∧ b = -1 ∧ c = -15 ∧
  ∀ (x y : ℝ), y = 8 * x - 15 → a * x + b * y + c = 0 :=
by 
  sorry

end bisecting_chord_line_eqn_l600_600942


namespace sum_of_squares_inequality_l600_600277

variable {n : ℕ}
variable {x y z : Fin n → ℝ}
variable (perm_z : ∀ i j, i < j → z i < z j)

theorem sum_of_squares_inequality
  (hx : ∀ i j, i < j → x i ≥ x j)
  (hy : ∀ i j, i < j → y i ≥ y j)
  (hz_permutation : ∃ σ : Fin n → Fin n, ∀ i, z i = y (σ i)) :
  ∑ i, (x i - y i) ^ 2 ≤ ∑ i, (x i - z i) ^ 2 :=
by {
  sorry
}

end sum_of_squares_inequality_l600_600277


namespace fraction_of_raisins_in_mixture_l600_600401

def cost_of_raisins (R : ℝ) := 3 * R
def cost_of_nuts (R : ℝ) := 3 * (3 * R)
def total_cost (R : ℝ) := cost_of_raisins R + cost_of_nuts R

theorem fraction_of_raisins_in_mixture (R : ℝ) (hR_pos : R > 0) : 
  cost_of_raisins R / total_cost R = 1 / 4 :=
by
  sorry

end fraction_of_raisins_in_mixture_l600_600401


namespace m_n_sum_l600_600445

-- Define the given points
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (5, 1)
def point3 : ℝ × ℝ := (8, 4)

-- Define the conditions of the problem
def line_of_fold (p1 p2 : ℝ × ℝ) := 
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let slope := (p2.2 - p1.2) / (p2.1 - p1.1)
  let perpendicular_slope := -1 / slope
  (midpoint, perpendicular_slope)

def new_points_fold (p1 p2 : ℝ × ℝ) (l : ℝ → ℝ) := 
  let slope := (p2.2 - p1.2) / (p2.1 - p1.1)
  (p1, p2, slope)

-- Hypothesis conditions
def conditions := 
  let fold1 := line_of_fold point1 point2
  let fold2 := line_of_fold point3 (8, 4)
  fold1.2 = 2 ∧ fold2.2 = -1 / 2

-- Prove that m + n = 10.67
theorem m_n_sum : 
  ∀ (m n : ℝ), 
  conditions → 
  m = 16 - 2 * n → 
  n = m → 
  m + n = 32 / 3 := 
sorry

end m_n_sum_l600_600445


namespace find_red_chairs_l600_600666

noncomputable def red_chairs := Nat
noncomputable def yellow_chairs := Nat
noncomputable def blue_chairs := Nat

theorem find_red_chairs
    (R Y B : Nat)
    (h1 : Y = 2 * R)
    (h2 : B = Y - 2)
    (h3 : R + Y + B = 18) :
    R = 4 := by
  sorry

end find_red_chairs_l600_600666


namespace tank_capacity_l600_600872

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end tank_capacity_l600_600872


namespace count_positive_integers_satisfying_inequality_l600_600545

theorem count_positive_integers_satisfying_inequality :
  ∃ n : ℕ, n = 4 ∧ ∀ x : ℕ, (10 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 50) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := 
by
  sorry

end count_positive_integers_satisfying_inequality_l600_600545


namespace find_p_for_exactly_three_solutions_l600_600949

theorem find_p_for_exactly_three_solutions :
  (∃ (p : ℝ), p = 1 / 4 ∧ ∀ x : ℝ, (p * x^2 = |x-1|) →
  {(x : ℝ) | p * x^2 = |x-1|}.finite.to_finset.card = 3) 
  :=
sorry

end find_p_for_exactly_three_solutions_l600_600949


namespace average_marks_l600_600458

variable (P C M : ℕ)

theorem average_marks :
  P = 140 →
  (P + M) / 2 = 90 →
  (P + C) / 2 = 70 →
  (P + C + M) / 3 = 60 :=
by
  intros hP hM hC
  sorry

end average_marks_l600_600458


namespace number_of_people_in_room_l600_600779

-- Definitions from conditions:
def number_of_empty_chairs : ℕ := 10
def fraction_of_empty_chairs : ℝ := 1 / 6
def fraction_of_seated_people : ℝ := 3 / 5
def fraction_of_occupied_chairs : ℝ := 5 / 6

-- The theorem statement
theorem number_of_people_in_room :
  let total_chairs := number_of_empty_chairs / fraction_of_empty_chairs in
  let occupied_chairs := total_chairs * fraction_of_occupied_chairs in
  let total_people := occupied_chairs / fraction_of_seated_people in
  total_people = 83 :=
by
  sorry

end number_of_people_in_room_l600_600779


namespace quadratic_even_coefficient_l600_600384

theorem quadratic_even_coefficient (a b c : ℤ) (h : ∃ x : ℚ, x^2 * a + x * b + c = 0) (ha : a ≠ 0) : (¬ (even a ∨ even b ∨ even c)) → false := 
by
  sorry

end quadratic_even_coefficient_l600_600384


namespace face_value_of_bond_l600_600259

variable (F : ℝ) (S : ℝ)

def interest_from_face_value (F : ℝ) : ℝ :=
  0.08 * F

def interest_from_selling_price (S : ℝ) : ℝ :=
  0.065 * S

axiom approx_selling_price : S ≈ 6153.846153846153

theorem face_value_of_bond :
  F = 5000 :=
by
  have interest_face_value : interest_from_face_value F = 0.08 * F := rfl
  have interest_selling_price : interest_from_selling_price S = 0.065 * S := rfl
  have selling_price_eq : interest_from_selling_price S ≈ 0.065 * 6153.846153846153 := sorry
  have interest_eq : interest_face_value = interest_selling_price := sorry
  have face_value_calc : F = 5000 := sorry
  exact face_value_calc

end face_value_of_bond_l600_600259


namespace sum_of_interior_edges_l600_600449

-- Define the problem parameters
def width_of_frame : ℝ := 2 -- width of the frame pieces in inches
def exposed_area : ℝ := 30 -- exposed area of the frame in square inches
def outer_edge_length : ℝ := 6 -- one of the outer edge length in inches

-- Define the statement to prove
theorem sum_of_interior_edges :
  ∃ (y : ℝ), (6 * y - 2 * (y - width_of_frame * 2) = exposed_area) ∧
  (2 * (6 - width_of_frame * 2) + 2 * (y - width_of_frame * 2) = 7) :=
sorry

end sum_of_interior_edges_l600_600449


namespace true_proposition_l600_600609

def p1 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def p2 : Prop := ∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), x^2 - 1 ≥ 0

theorem true_proposition : (¬ p1) ∧ p2 :=
  by
    sorry

end true_proposition_l600_600609


namespace statement3_statement4_problem_solution_l600_600174

theorem statement3 (a b : ℝ) (h1 : a + b > 0) (h2 : a * b > 0) : a > 0 ∧ b > 0 :=
sorry

theorem statement4 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b / a < 0) (h3 : c / a > 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x > 0 ∧ ∃ y : ℝ, y ≠ x ∧ a * y^2 + b * y + c = 0 ∧ y > 0 :=
sorry

theorem problem_solution :
  (statement3, statement4) = (true, true) :=
sorry

end statement3_statement4_problem_solution_l600_600174


namespace gcd_subtract_twelve_l600_600574

theorem gcd_subtract_twelve (a b : ℕ) (n : ℕ) (h₁ : a = 4760) (h₂ : b = 280) (h₃ : n = 12) :
  let g := Nat.gcd a b in g - n = 268 :=
by
  sorry

end gcd_subtract_twelve_l600_600574


namespace maximum_volume_crystalline_polyhedron_l600_600447

theorem maximum_volume_crystalline_polyhedron (n : ℕ) (R : ℝ) (h : n > 3) :
  ∃ V, V = \frac{32 * (n - 1) * R^3}{81} * sin (\frac{2 * π}{n - 1}) := 
sorry

end maximum_volume_crystalline_polyhedron_l600_600447


namespace no_solution_l600_600582

-- Define the greatest integer function
def floor (t : ℝ) : ℤ := Int.floor t

-- Define the function f(x)
def f (x : ℝ) : ℤ :=
  floor x + floor (2 * x) + floor (4 * x) + floor (8 * x) + floor (16 * x) + floor (32 * x)

theorem no_solution : ¬ ∃ x : ℝ, f x = 12345 := by
  sorry

end no_solution_l600_600582


namespace original_price_given_discounts_l600_600468

theorem original_price_given_discounts (p q d : ℝ) (h : d > 0) :
  ∃ x : ℝ, x * (1 + (p - q) / 100 - p * q / 10000) = d :=
by
  sorry

end original_price_given_discounts_l600_600468


namespace Walter_receives_49_bananas_l600_600246

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end Walter_receives_49_bananas_l600_600246


namespace percentage_hydrocarbons_second_source_l600_600017

theorem percentage_hydrocarbons_second_source 
  (x : ℕ)
  (h1 : 50 * 0.55 = 27.5)
  (h2 : 20 * 0.25 = 5)
  (h3 : 5 + 30 * (x / 100) = 27.5) : 
  x = 75 := 
by 
  sorry

end percentage_hydrocarbons_second_source_l600_600017


namespace base3_to_base10_conversion_l600_600532

theorem base3_to_base10_conversion : 
  1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0 = 142 :=
by {
  -- calculations
  sorry
}

end base3_to_base10_conversion_l600_600532


namespace pq_square_diff_l600_600203

variable {p q : ℝ}

theorem pq_square_diff : p + q = 10 → p - q = 4 → p^2 - q^2 = 40 :=
by
  intros h1 h2
  rw [eq_iff_eq_cancel_right.mpr h1, eq_iff_eq_cancel_right.mpr h2]
  exact sorry

end pq_square_diff_l600_600203


namespace arithmetic_sequence_sum_is_18_l600_600615

variable (a : ℕ → ℕ)

theorem arithmetic_sequence_sum_is_18
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 18 := 
sorry

end arithmetic_sequence_sum_is_18_l600_600615


namespace ratio_Dante_Paolo_l600_600291

-- Definitions
def Paolo_coconuts : ℕ := 14
def Dante_after_selling : ℕ := 32
def Coconuts_difference : ℕ := 10

-- Theorem statement
theorem ratio_Dante_Paolo : ∃ x : ℕ, 14 * x - 10 = 32 ∧ (32 / 14).nat_gcd = 16 / 7 := sorry

end ratio_Dante_Paolo_l600_600291


namespace min_value_orthogonal_vectors_l600_600643

theorem min_value_orthogonal_vectors (x y : ℝ) 
  (h1 : let a := (x - 1, 2) in let b := (4, y) in (a.1 * b.1 + a.2 * b.2) = 0) 
  (h2 : 2 * x + y = 2) : 9^x + 3^y = 6 :=
sorry

end min_value_orthogonal_vectors_l600_600643


namespace frequency_and_relative_frequency_of_group3_l600_600885

-- Define the conditions
def sample_capacity : ℕ := 100
def num_groups : ℕ := 8
def freq_g1 : ℕ := 10
def freq_g2 : ℕ := 13
def freq_g4 : ℕ := 14
def freq_g5 : ℕ := 15
def freq_g6 : ℕ := 13
def freq_g7 : ℕ := 12
def freq_g8 : ℕ := 9

-- The frequency of Group 3 and its relative frequency
theorem frequency_and_relative_frequency_of_group3 :
  let freq_g3 := sample_capacity - (freq_g1 + freq_g2 + freq_g4 + freq_g5 + freq_g6 + freq_g7 + freq_g8) in
  freq_g3 = 14 ∧ (freq_g3 : ℝ) / (sample_capacity : ℝ) = 0.14 :=
by
  let freq_g3 := sample_capacity - (freq_g1 + freq_g2 + freq_g4 + freq_g5 + freq_g6 + freq_g7 + freq_g8)
  show freq_g3 = 14 ∧ (freq_g3 : ℝ) / (sample_capacity : ℝ) = 0.14
  sorry

end frequency_and_relative_frequency_of_group3_l600_600885


namespace electronic_items_stock_l600_600660

-- Define the base statements
def all_in_stock (S : Type) (p : S → Prop) : Prop := ∀ x, p x
def some_not_in_stock (S : Type) (p : S → Prop) : Prop := ∃ x, ¬ p x

-- Define the main theorem statement
theorem electronic_items_stock (S : Type) (p : S → Prop) :
  ¬ all_in_stock S p → some_not_in_stock S p :=
by
  intros
  sorry

end electronic_items_stock_l600_600660


namespace nyusha_wins_probability_l600_600714

-- Define the number of coins Nyusha and Barash have
def N : ℕ := 2022
def B : ℕ := 2023

-- Define the probability of heads for each coin
def p : ℝ := 0.5

-- Define the probability that Nyusha wins given the conditions
theorem nyusha_wins_probability : 
  (probability (λ (X Y : ℕ), X > Y ∨ (X = Y)) = 0.5) :=
sorry

end nyusha_wins_probability_l600_600714


namespace tooth_fairy_amount_per_tooth_after_first_l600_600464

theorem tooth_fairy_amount_per_tooth_after_first
  (total_teeth : ℕ := 20)
  (total_money : ℕ := 54)
  (unpaid_teeth : ℕ := 2)
  (first_tooth_money : ℕ := 20)
  (remaining_teeth := total_teeth - 1 - unpaid_teeth)
  (remaining_money := total_money - first_tooth_money) :
  remaining_money / remaining_teeth = 2 :=
by
  -- The condition that halfway through the calculation must hold:
  -- remaining_teeth = 17 ⟹ total_teeth - 1 - unpaid_teeth = 20 - 1 - 2
  have remaining_teeth_eq : remaining_teeth = 17, by simp only [remaining_teeth, total_teeth, unpaid_teeth]; norm_num,
  -- remaining_money = 34 ⟹ total_money - first_tooth_money = 54 - 20
  have remaining_money_eq : remaining_money = 34, by simp only [remaining_money, total_money, first_tooth_money]; norm_num,
  
  -- Finally, substituting remaining_teeth and remaining_money to calculate per tooth amount
  show remaining_money / remaining_teeth = 2, by
    rw [remaining_teeth_eq, remaining_money_eq]
    norm_num

end tooth_fairy_amount_per_tooth_after_first_l600_600464


namespace b1_b2_values_general_formula_b_sum_first_20_l600_600135

def seq_a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then seq_a n + 1 else seq_a n + 2

def seq_b (n : ℕ) : ℕ := seq_a (2 * n)

theorem b1_b2_values : seq_b 1 = 2 ∧ seq_b 2 = 5 := by
  sorry

theorem general_formula_b (n : ℕ) : seq_b n = 3 * n - 1 := by
  sorry

theorem sum_first_20 : (Finset.range 20).sum seq_a = 300 := by
  sorry

end b1_b2_values_general_formula_b_sum_first_20_l600_600135


namespace shift_fraction_pi_over_3_l600_600333

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x) - sqrt 3 * cos (2 * x)

theorem shift_fraction_pi_over_3 :
  ∀ (x : ℝ), g (x) = f (x - π / 3) :=
by
  sorry

end shift_fraction_pi_over_3_l600_600333


namespace ratio_stamps_l600_600193

noncomputable def stamps_total := 240
noncomputable def stamps_sister := 60
noncomputable def stamps_harry := stamps_total - stamps_sister

theorem ratio_stamps (h_total : stamps_total = 240) (h_sister : stamps_sister = 60) :
  stamps_harry / stamps_sister = 3 :=
by
  simp [stamps_total, stamps_sister, stamps_harry]
  sorry

end ratio_stamps_l600_600193


namespace nested_sqrt_simplification_l600_600652

theorem nested_sqrt_simplification (y : ℝ) (hy : y ≥ 0) : 
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := 
sorry

end nested_sqrt_simplification_l600_600652


namespace part_I_part_II_l600_600638

-- Part (I) definitions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def point (x y : ℝ) : Prop := true  -- Ensure the point exists
def line_eq (x y : ℝ) : Prop := y = x - 1
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 16

-- Part (I) statement
theorem part_I :
  ∀ (x y x1 y1 x2 y2 : ℝ),
  parabola x1 y1 → parabola x2 y2 →
  point 1 0 → line_eq x1 y1 → line_eq x2 y2 →
  circle_eq x y ↔ true :=
sorry

-- Part (II) definitions
def line_general_eq (x y k m : ℝ) : Prop := x = k * y + m
def fixed_point_const (AM BM : ℝ) : Prop := (1 / AM^2) + (1 / BM^2) = 1 / 4

-- Part (II) statement
theorem part_II :
  ∃ (m : ℝ), (∀ (x y k : ℝ), m = 2 →
  line_general_eq x y k m →
  fixed_point_const (sqrt((k^4 + m / 2)^2)) (sqrt((k^4 + m / 2)^2))) :=
sorry

end part_I_part_II_l600_600638


namespace tessa_apples_left_l600_600725

theorem tessa_apples_left : 
  ∀ (initial_apples : ℝ) (additional_apples : ℝ) (apples_for_pie : ℝ), 
  initial_apples = 10.0 →
  additional_apples = 5.0 →
  apples_for_pie = 4.0 →
  initial_apples + additional_apples - apples_for_pie = 11.0 :=
by
  intros initial_apples additional_apples apples_for_pie h_initial h_additional h_pie
  rw [h_initial, h_additional, h_pie]
  norm_num

end tessa_apples_left_l600_600725


namespace smallest_m_q_largest_prime_l600_600276

theorem smallest_m_q_largest_prime :
  let q := largest_prime_with_digits 2009 in
  let m := 1 in
  (q^2 - m) % 15 = 0 :=
by
  let q := 7 -- Replace with actual largest prime with 2009 digits
  let m := 1
  sorry

end smallest_m_q_largest_prime_l600_600276


namespace suki_bag_weight_is_22_l600_600313

noncomputable def weight_of_suki_bag : ℝ :=
  let bags_suki := 6.5
  let bags_jimmy := 4.5
  let weight_jimmy_per_bag := 18.0
  let total_containers := 28
  let weight_per_container := 8.0
  let total_weight_jimmy := bags_jimmy * weight_jimmy_per_bag
  let total_weight_combined := total_containers * weight_per_container
  let total_weight_suki := total_weight_combined - total_weight_jimmy
  total_weight_suki / bags_suki

theorem suki_bag_weight_is_22 : weight_of_suki_bag = 22 :=
by
  sorry

end suki_bag_weight_is_22_l600_600313


namespace partition_exists_l600_600070

theorem partition_exists (n : ℕ) (h : n ≥ 2) :
  ∃ P : finset (finset ℕ),
  (P.card = n) ∧
  (∀ S ∈ P, ∃ a b c, S = {a, b, c} ∧ a < b ∧ b < c ∧ ∃ d1 d2, d1 ≠ d2 ∧ d1 ∈ {n-1, n, n+1} ∧ d2 ∈ {n-1, n, n+1} ∧ (b - a = d1) ∧ (c - b = d2)) :=
sorry

end partition_exists_l600_600070


namespace b1_b2_values_general_formula_b_sum_first_20_l600_600134

def seq_a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then seq_a n + 1 else seq_a n + 2

def seq_b (n : ℕ) : ℕ := seq_a (2 * n)

theorem b1_b2_values : seq_b 1 = 2 ∧ seq_b 2 = 5 := by
  sorry

theorem general_formula_b (n : ℕ) : seq_b n = 3 * n - 1 := by
  sorry

theorem sum_first_20 : (Finset.range 20).sum seq_a = 300 := by
  sorry

end b1_b2_values_general_formula_b_sum_first_20_l600_600134


namespace express_y_in_terms_of_x_l600_600953

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) : y = (4 - 2 * x) / 3 := 
by
  sorry

end express_y_in_terms_of_x_l600_600953


namespace cot_theta_implies_zero_l600_600116

variable (θ : ℝ) -- Define variable θ in ℝ

-- Define the condition cot θ = 3
axiom cot_theta_eq_three : Real.cot θ = 3

-- Define the problem statement
theorem cot_theta_implies_zero :
  Real.cot θ = 3 →
  (\frac{1 - Real.sin θ}{Real.cos θ}) - (\frac{Real.cos θ}{1 + Real.sin θ}) = 0 :=
by
  intro h
  sorry

end cot_theta_implies_zero_l600_600116


namespace parabola_equation_value_of_m_minimum_area_of_triangle_l600_600598

-- Problem 1: Equation of the parabola
theorem parabola_equation :
  ∃ p > 0, ∀ (x y : ℝ), (x^2 = 2 * p * y) ↔ (x^2 = 4 * y) :=
sorry

-- Problem 2: Value of m
theorem value_of_m :
  ∃ (m : ℝ), m > 0 ∧ (∀ (x₁ x₂ : ℝ) (k : ℝ) (p : ℝ),
    (x₁ + x₂ = 4 * k ∧ x₁ * x₂ = -4 * m ∧ 
     (1 + k^2) * (-4 * m) + k * m * (4 * k) + m^2 = 0) → m = 4) :=
sorry

-- Problem 3: Minimum area of triangle ABD
theorem minimum_area_of_triangle :
  ∀ (x₁ x₂ x₃ k : ℝ) (A B D : ℝ × ℝ), x₁ < x₂ →
  (x₁^2 = 4 * (A.2)) ∧ (x₂^2 = 4 * (B.2)) ∧ (x₃^2 = 4 * (D.2)) →
  ((x₁^2 - 4 * k * x₁ + 4 * k^2) = (1 + k^2)) →
  let area := 2 * (k + 1/k)^3 in
  ∃ min_area : ℝ, min_area = 16 :=
sorry

end parabola_equation_value_of_m_minimum_area_of_triangle_l600_600598


namespace proof_l600_600608

-- Define propositions
def p := ∀ x : ℝ, x > 0 → 3^x > 1
def q := ∀ a b : ℝ, a < b → a^2 < b^2

-- Theorem statement
theorem proof : p ∧ ¬q :=
by
  sorry

end proof_l600_600608


namespace central_angle_double_score_l600_600001

theorem central_angle_double_score 
  (prob: ℚ)
  (total_angle: ℚ)
  (num_regions: ℚ)
  (eq_regions: ℚ → Prop)
  (double_score_prob: prob = 1/8)
  (total_angle_eq: total_angle = 360)
  (num_regions_eq: num_regions = 6) 
  : ∃ x: ℚ, (prob = x / total_angle) → x = 45 :=
by
  sorry

end central_angle_double_score_l600_600001


namespace part1_part2_l600_600154

open Real

variables (a b c : Vector ℝ) (theta : ℝ)
variables [comparableSpace : ∀ {r n : ℝ}, n > 0 → r ∈ comparableSpace]

-- Given conditions
def conditions_1 : Prop := 
  (a = (1, -sqrt 3)) ∧ (parallel c a) ∧ (norm c = 4)

def conditions_2 : Prop := 
  (b = (1, -sqrt 3)) ∧ (norm b = 1) ∧ ((a + 2 * b) ⊥ (2 * a - b))

-- Expected answers
def answer_1 : Prop := 
  (c = (2, -2 * sqrt 3) ∨ c = (-2, 2 * sqrt 3))

def answer_2 : Prop := 
  (theta = π)

theorem part1 : conditions_1 → answer_1 := by
  sorry

theorem part2 : conditions_2 → answer_2 := by
  sorry

end part1_part2_l600_600154


namespace can_lids_total_l600_600893

theorem can_lids_total (boxes : ℕ) (lids_per_box : ℕ) (already_has : ℕ)
  (h_boxes : boxes = 3) (h_lids_per_box : lids_per_box = 13) (h_already_has : already_has = 14) :
  (boxes * lids_per_box + already_has = 53) :=
by
  rw [h_boxes, h_lids_per_box, h_already_has]
  norm_num
  sorry

end can_lids_total_l600_600893


namespace min_value_expr_l600_600980

theorem min_value_expr (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1)
  (h4 : a + b = 1) :
  ( ( (2 * a + b) / (a * b) - 3 ) * c + ( (real.sqrt 2) / (c - 1) ) ) ≥ 4 + 2 * real.sqrt 2 :=
by
  sorry

end min_value_expr_l600_600980


namespace dollar_expansion_l600_600068

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2 + a * b

theorem dollar_expansion : dollar ((x - y) ^ 3) ((y - x) ^ 3) = -((x - y) ^ 6) := by
  sorry

end dollar_expansion_l600_600068


namespace rectangle_area_is_30_l600_600675

def Point := (ℤ × ℤ)

def vertices : List Point := [(-5, 1), (1, 1), (1, -4), (-5, -4)]

theorem rectangle_area_is_30 :
  let length := (vertices[1].1 - vertices[0].1).natAbs
  let width := (vertices[0].2 - vertices[2].2).natAbs
  length * width = 30 := by
  sorry

end rectangle_area_is_30_l600_600675


namespace minimum_path_proof_l600_600884

noncomputable def minimum_path (r : ℝ) (h : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let R := Real.sqrt (r^2 + h^2)
  let theta := 2 * Real.pi * (R / (2 * Real.pi * r))
  let A := (d1, 0)
  let B := (-d2 * Real.cos (theta / 2), -d2 * Real.sin (theta / 2))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem minimum_path_proof :
  minimum_path 800 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 562.158 := 
by 
  sorry

end minimum_path_proof_l600_600884


namespace calculate_expression_l600_600491

theorem calculate_expression : (sqrt 27 / (sqrt 3 / 2) * (2 * sqrt 2) - (6 * sqrt 2)) = 6 * sqrt 2 :=
by
  -- Taking these steps from the solution, we should finally arrive at the required proof
  sorry

end calculate_expression_l600_600491


namespace multiplication_problem_solution_l600_600687

theorem multiplication_problem_solution (a b c : ℕ) 
  (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1) 
  (h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h3 : (a * 100 + b * 10 + b) * c = b * 1000 + c * 100 + b * 10 + 1) : 
  a = 5 ∧ b = 3 ∧ c = 7 := 
sorry

end multiplication_problem_solution_l600_600687


namespace arcsin_one_eq_pi_div_two_l600_600521

noncomputable def arcsin : ℝ → ℝ := sorry -- Define arcsin function

theorem arcsin_one_eq_pi_div_two : arcsin 1 = Real.pi / 2 := sorry

end arcsin_one_eq_pi_div_two_l600_600521


namespace value_of_a_star_b_l600_600350

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l600_600350


namespace probability_scrapped_l600_600859

variable (P_A P_B_given_not_A : ℝ)
variable (prob_scrapped : ℝ)

def fail_first_inspection (P_A : ℝ) := 1 - P_A
def fail_second_inspection_given_fails_first (P_B_given_not_A : ℝ) := 1 - P_B_given_not_A

theorem probability_scrapped (h1 : P_A = 0.8) (h2 : P_B_given_not_A = 0.9) (h3 : prob_scrapped = fail_first_inspection P_A * fail_second_inspection_given_fails_first P_B_given_not_A) :
  prob_scrapped = 0.02 := by
  sorry

end probability_scrapped_l600_600859


namespace tangent_line_at_one_l600_600328

def f (x : ℝ) : ℝ := x^3 + x

theorem tangent_line_at_one :
  let df := deriv f
  let slope := df 1
  let point := (1 : ℝ, f 1)
  let line_eq : ℝ → ℝ := λ x, slope * (x - point.1) + point.2
  ∀ x y, y = line_eq x ↔ 4 * x - y - 2 = 0 :=
by
  sorry

end tangent_line_at_one_l600_600328


namespace radius_of_circle_l600_600758

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = Real.pi * r ^ 2) : r = 6 := 
by 
  sorry

end radius_of_circle_l600_600758


namespace correct_statements_count_l600_600060

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m*x^2 - 6*m*x - 5

def statement_1 (m : ℝ) (a : ℝ) : Prop :=
  let x1 := 3 + a
  let x2 := 3 - a
  quadratic_function m x1 = quadratic_function m x2

def statement_2 (m : ℝ) : Prop :=
  quadratic_function m 2 = -13 → 
  ∀ (x1 x2 : ℝ), (x1 > x2 ∧ x2 > 9 / 2) → 
  (quadratic_function m x1 - quadratic_function m x2) / (x1 - x2) < 0

def statement_3 (m : ℝ) : Prop :=
  (3 ≤ m ∧ m < 4 / 9) ∨ (-4 / 9 < m ∧ m ≤ -1 / 3)

def statement_4 (m n : ℝ) : Prop :=
  m > 0 → n ≤ 3 → 
  ∀ x, n ≤ x ∧ x ≤ 3 → -14 ≤ quadratic_function m x ∧ quadratic_function m x ≤ n^2 + 1 → 
  n = 1

theorem correct_statements_count : 
  ∀ m : ℝ, (statement_1 m) ∧ ¬(statement_2 m) ∧ (statement_3 m) ∧ ¬(statement_4 m 1) → 2 :=
sorry

end correct_statements_count_l600_600060


namespace binomial_coeff_12_3_l600_600527

/-- The binomial coefficient is defined as: 
  \binom{n}{k} = \frac{n!}{k!(n-k)!} -/
theorem binomial_coeff_12_3 : Nat.binom 12 3 = 220 := by
  sorry

end binomial_coeff_12_3_l600_600527


namespace symmetric_axis_l600_600181

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 4)

theorem symmetric_axis (ω : ℝ) (x : ℝ) (hx1: ω > 0) 
  (hx2: ∀ x, 0 < x ∧ x < 3 * π / 4 → f ω x > 1 / 2) 
  (hωmax: ω = 7 / 9) : x = 9 * π / 28 := 
sorry

end symmetric_axis_l600_600181


namespace triangle_area_zero_after_doubling_sides_l600_600688

theorem triangle_area_zero_after_doubling_sides
  (A B C : Type) [metricSpace A] [metricSpace B] [metricSpace C]
  (AB AC BC : ℝ) (hAB : AB = 12) (hAC : AC = 7) (hBC : BC = 10)
  (AB' AC' BC' : ℝ) (hAB' : AB' = 24) (hAC' : AC' = 14) (hBC' : BC' = 10) :
  (AB' + AC' = BC' ∨ AC' + BC' = AB' ∨ AB' + BC' = AC') → 
  area' = 0 :=
by
  sorry

end triangle_area_zero_after_doubling_sides_l600_600688


namespace factorize_x_cubed_minus_9x_l600_600933

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l600_600933


namespace probability_increase_l600_600672

variables (k l : ℕ) (p : ℝ)

def q := 1 - p

noncomputable def binom (n k : ℕ) := nat.choose n k

theorem probability_increase (hkl : k < 15 ∧ l < 15) :
  let A_hits := 14 - k,
      B_hits := 14 - l in
  (binom (k + l) k) * (p^k) * (q^(l + 1)) = 
  (binom (k + l) k) * (p^k) * ((1 - p)^(l + 1)) :=
by sorry

end probability_increase_l600_600672


namespace sum_rational_root_cs_l600_600101

def sum_of_valid_c (c : ℤ) : Prop :=
  c ≤ 30 ∧ ∃ k : ℤ, 81 + 4 * c = k^2

theorem sum_rational_root_cs : 
  (∑ c in Finset.filter sum_of_valid_c (Finset.range 31), c) = 32 := 
by
  -- Proof omitted
  sorry

end sum_rational_root_cs_l600_600101


namespace right_triangle_with_arithmetic_progression_sides_l600_600764

theorem right_triangle_with_arithmetic_progression_sides (a : ℤ) (d : ℤ) :
  (d ≠ 0) ∧ (a = 8 * d) → 
  (∃ x ∈ {24, 60, 85, 100, 375}, ∃ n : ℤ, (x = 6 * n ∨ x = 8 * n ∨ x = 10 * n)) :=
by 
  sorry

end right_triangle_with_arithmetic_progression_sides_l600_600764


namespace asymptotics_of_x_l600_600108

theorem asymptotics_of_x {n : ℕ} (hn : 0 < n) :
  ∃ (C : ℝ) (hC : C > 0), x = Θ(C * (log (n : ℝ) / log (log (n : ℝ)))) :=
sorry

end asymptotics_of_x_l600_600108


namespace geometric_sequence_property_l600_600221

variable {a : ℕ → ℝ} (k : ℝ) (q : ℝ)

/-- 
  Given a positive geometric sequence {a_n} with common ratio q > 0,
  and the terms a_2, 3a_1, and a_3 form an arithmetic sequence,
  and the sum of the first n terms S_n = k * a_n - 1,
  prove that a_{2022} = 2^{2021}.
-/
theorem geometric_sequence_property
  (hp : ∀ n, a n > 0)
  (hq : q > 0)
  (ha : a 2 = a 1 * q)
  (haa : a 3 = a 1 * q^2)
  (ar_seq : 2 * 3 * a 1 = a 2 + a 3)
  (Sn_def : ∀ n, (∑ i in Finset.range (n+1), a i) = k * a n - 1)
  : a 2022 = 2^2021 := 
sorry

end geometric_sequence_property_l600_600221


namespace kaylin_age_l600_600257

theorem kaylin_age : 
  ∀ (Freyja Eli Sarah Kaylin : ℕ), 
    Freyja = 10 ∧ 
    Eli = Freyja + 9 ∧ 
    Sarah = 2 * Eli ∧ 
    Kaylin = Sarah - 5 -> 
    Kaylin = 33 :=
by
  intro Freyja Eli Sarah Kaylin
  intro h
  cases h with hF h1
  cases h1 with hE h2
  cases h2 with hS hK
  sorry

end kaylin_age_l600_600257


namespace ellipse_standard_eq_fixed_points_existence_l600_600605

-- Conditions for part (Ⅰ)
def e := (Real.sqrt 2) / 2
def focus_1 : ℝ × ℝ := (-Real.sqrt 2, 0)

-- Statement for part (Ⅰ)
theorem ellipse_standard_eq (h_eccentricity : e = (Real.sqrt 2) / 2) (h_focus : focus_1 = (-Real.sqrt 2, 0)) :
  ∃ a b : ℝ, (a = 2) ∧ (b = Real.sqrt 2) ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∨ y ≠ 0)) :=
sorry

-- Conditions for part (Ⅱ)
def C2_eq (x y : ℝ) := x^2 + 2 * y^2 = 12
def point_T (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := (x1 + 2 * x2, y1 + 2 * y2)
def slopes_product_cond (x1 y1 x2 y2 : ℝ) := (y1 / x1) * (y2 / x2) = -1/2

-- Statement for part (Ⅱ)
theorem fixed_points_existence (h_trajectory : ∀ u v : ℝ, (u^2 / 4 + v^2 / 2 = 1) → ((2*v - u)^2 + 2*(u + v)^2 = 12))
  (h_T_condition : ∀ x1 y1 x2 y2, C2_eq x1 y1 → C2_eq x2 y2 → point_T x1 y1 x2 y2)
  (h_slopes_cond : ∀ x1 y1 x2 y2, slopes_product_cond x1 y1 x2 y2) :
  ∃ F1 F2 : ℝ × ℝ, (F1 = (-Real.sqrt 30, 0)) ∧ (F2 = (Real.sqrt 30, 0)) ∧
  (∀ x y, (∃ m n, point_T m n x y = (F1.1 + F2.1, F1.2 + F2.2)) → (x^2 / 60 + y^2 / 30 = 1)) :=
sorry

end ellipse_standard_eq_fixed_points_existence_l600_600605


namespace find_light_green_paint_amount_l600_600856

theorem find_light_green_paint_amount :
  ∃ (x : ℝ), 
    let a := 0.20
    let b := 0.40
    let c := 1.66666666667
    let d := 0.25
    in (a * x + b * c = d * (x + c)) ∧ x = 5 :=
by
  sorry

end find_light_green_paint_amount_l600_600856


namespace sqrt_expression_eq_l600_600505

theorem sqrt_expression_eq :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2) = 6 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_eq_l600_600505


namespace alpha_plus_2beta_l600_600979

noncomputable def sin_square (θ : ℝ) := (Real.sin θ)^2
noncomputable def sin_double (θ : ℝ) := Real.sin (2 * θ)

theorem alpha_plus_2beta (α β : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) 
(hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : 3 * sin_square α + 2 * sin_square β = 1)
(h2 : 3 * sin_double α - 2 * sin_double β = 0) : 
α + 2 * β = 5 * Real.pi / 6 :=
by
  sorry

end alpha_plus_2beta_l600_600979


namespace max_value_of_f_l600_600266

def f (x : ℝ) : ℝ := min (2^x) (min (x + 2) (10 - x))

theorem max_value_of_f : ∃ x ≥ 0, f x = 6 :=
by
  use some x, sorry

end max_value_of_f_l600_600266


namespace cookies_per_sheet_is_16_l600_600863

-- Define the number of members
def members : ℕ := 100

-- Define the number of sheets each member bakes
def sheets_per_member : ℕ := 10

-- Define the total number of cookies baked
def total_cookies : ℕ := 16000

-- Calculate the total number of sheets baked
def total_sheets : ℕ := members * sheets_per_member

-- Define the number of cookies per sheet as a result of given conditions
def cookies_per_sheet : ℕ := total_cookies / total_sheets

-- Prove that the number of cookies on each sheet is 16 given the conditions
theorem cookies_per_sheet_is_16 : cookies_per_sheet = 16 :=
by
  -- Assuming all the given definitions and conditions
  sorry

end cookies_per_sheet_is_16_l600_600863


namespace stock_decrease_2007_l600_600025

theorem stock_decrease_2007 (x : ℝ) (h : x > 0) :
  let y := 1.25 * x in
  (y - 0.8 * y = x) → ∃ p : ℝ, p = 20 :=
by
  intro y
  intro h₀
  use 20
  sorry

end stock_decrease_2007_l600_600025


namespace sqrt_expression_eq_l600_600502

theorem sqrt_expression_eq :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2) = 6 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_eq_l600_600502


namespace find_x_solutions_l600_600564

section ProofProblem

def integer_part (x : ℝ) : ℝ := ⌊x⌋
def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem find_x_solutions (x : ℝ) :
  2 * integer_part x + fractional_part (3 * x) = (7 : ℝ) / 3 ↔
  x = (1 : ℝ) + (1 : ℝ) / 9 ∨
  x = (1 : ℝ) + (4 : ℝ) / 9 ∨
  x = (1 : ℝ) + (7 : ℝ) / 9
:=
by {
  sorry
}

end ProofProblem

end find_x_solutions_l600_600564


namespace diameter_of_circle_l600_600419

theorem diameter_of_circle (A : ℝ) (h : A = 4 * real.pi) : ∃ d : ℝ, d = 4 :=
  sorry

end diameter_of_circle_l600_600419


namespace smallest_positive_period_and_axis_of_symmetry_l600_600120

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem smallest_positive_period_and_axis_of_symmetry :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ k : ℤ, ∀ x, 2 * x - Real.pi / 4 = k * Real.pi + Real.pi / 2 → x = k * Real.pi / 2 - Real.pi / 8) :=
  sorry

end smallest_positive_period_and_axis_of_symmetry_l600_600120


namespace count_distinct_colorings_l600_600914

def regular_2n_gon (n : ℕ) := { P : Type | ∃ (A : Π i : fin (2 * n), P), true }

def valid_coloring {n : ℕ} (P : regular_2n_gon n) :=
  ∃ (color : Π i : fin (2 * n), fin 3),
    (∀ j : fin 3, ∃ i : fin (2 * n), color i = j) ∧
    (∀ (E : P → Prop) (hE : ∀ x, E x → ∃ i j, E x → color i ≠ color j → ∃ (k : fin n), E x → (i + k) % (2 * n) = j % (2 * n)), true)

theorem count_distinct_colorings (n : ℕ) (P : regular_2n_gon n) : 
  ∃ (f : Π (P₁ P₂ : P → fin 3), P₁ ≠ P₂), 6 * n := sorry

end count_distinct_colorings_l600_600914


namespace radius_of_circle_l600_600755

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = Real.pi * r ^ 2) : r = 6 := 
by 
  sorry

end radius_of_circle_l600_600755


namespace track_length_l600_600906

-- Define the conditions and question
theorem track_length 
  (x : ℝ) -- length of the track
  (b_speed s_speed : ℝ) -- Brenda's and Sally's speeds
  (start_diametrically_opposite : true) -- initial condition
  (meeting_after_120_by_brenda : b_speed * 120 / b_speed = 120) -- Brenda runs 120 meters
  (first_meeting_point : s_speed = 2 * b_speed) -- Sally runs twice the speed of Brenda
  (sally_runs_180_past_first_meeting : true)
  (second_meeting_point : s_speed * (240 / s_speed + 180 / s_speed) = 420) -- Sally runs total 420 meters
  (brenda_90_past_first_meeting : b_speed * (120 / b_speed + 90 / b_speed) = 210) -- Brenda runs total 210 meters
  :

  x = 360 := -- the length of the track

begin
  sorry -- proof to be filled
end

end track_length_l600_600906


namespace calculate_expression_l600_600492

theorem calculate_expression : (sqrt 27 / (sqrt 3 / 2) * (2 * sqrt 2) - (6 * sqrt 2)) = 6 * sqrt 2 :=
by
  -- Taking these steps from the solution, we should finally arrive at the required proof
  sorry

end calculate_expression_l600_600492


namespace arithmetic_geometric_seq_l600_600604

theorem arithmetic_geometric_seq (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_diff : d = 2)
  (h_geom : (a 1)^2 = a 0 * (a 0 + 6)) :
  a 1 = -6 :=
by 
  sorry

end arithmetic_geometric_seq_l600_600604


namespace angle_BAC_eq_60_l600_600115

-- Conditions
structure TriangleWithCircumcenterIncenter (A B C O I : Type) :=
  (is_circumcenter : TurTriangleCircumcenter A B C O)
  (is_incenter : TurTriangleIncenter A B C I)
  (angle_OIB_eq_30 : ∀ {O B I : Type}, angle O I B = 30)

-- Theorem statement
theorem angle_BAC_eq_60 {A B C O I : Type} [TriangleWithCircumcenterIncenter A B C O I] :
  angle A B C = 60 :=
begin
  sorry
end

end angle_BAC_eq_60_l600_600115


namespace largest_possible_radius_tangent_circle_l600_600072

theorem largest_possible_radius_tangent_circle :
  ∃ (r : ℝ), 0 < r ∧
    (∀ x y, (x - r)^2 + (y - r)^2 = r^2 → 
    ((x = 9 ∧ y = 2) → (r = 17))) :=
by
  sorry

end largest_possible_radius_tangent_circle_l600_600072


namespace time_to_cross_lamp_post_l600_600889

theorem time_to_cross_lamp_post (bridge_distance train_distance : ℝ) (total_time : ℝ)
  (h_bridge_distance : bridge_distance = 150)
  (h_train_distance : train_distance = 75)
  (h_total_time : total_time = 7.5) :
  (train_distance / ((bridge_distance + train_distance) / total_time)) = 2.5 :=
by {
  rw [h_bridge_distance, h_train_distance, h_total_time],
  norm_num,
  sorry,
}

end time_to_cross_lamp_post_l600_600889


namespace skew_lines_not_two_points_l600_600380

open Classical

theorem skew_lines_not_two_points (L1 L2 : Line) (H_skew : ¬∃ P, P ∈ L1 ∧ P ∈ L2) (P : Plane) :
  ∃ P1 P2, P1 ∈ (L1.projected_onto P) ∧ P2 ∈ (L1.projected_onto P) → False :=
by sorry

end skew_lines_not_two_points_l600_600380


namespace sum_of_solutions_l600_600650

theorem sum_of_solutions (x : ℝ) (h : x^2 - 3 * x = 12) : x = 3 := by
  sorry

end sum_of_solutions_l600_600650


namespace probability_blue_ball_l600_600669

noncomputable theory

variables (P_red P_yellow P_blue : ℝ)

-- Conditions
axiom h1 : P_red = 0.48
axiom h2 : P_yellow = 0.35
axiom h3 : P_red + P_yellow + P_blue = 1

-- Goal to prove
theorem probability_blue_ball : P_blue = 0.17 :=
by
  -- We'll assume the proof content is given correctly
  sorry

end probability_blue_ball_l600_600669


namespace completing_the_square_l600_600813

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l600_600813


namespace line_through_midpoint_of_PQ_l600_600225

theorem line_through_midpoint_of_PQ :
  ∀ {A B C A' B' D P Q : Point} 
    (h_acute : acute_angled_triangle A B C)
    (h_alt1 : altitude A A')
    (h_alt2 : altitude B B')
    (h_circumcircle : D ∈ circumcircle of triangle ABC)
    (h_arc : arc_ACB_not_containing_A D)
    (h_intersect1 : intersection (line_through_points A A') (line_through_points B D) = P)
    (h_intersect2 : intersection (line_through_points B B') (line_through_points A D) = Q),
  passes_through_midpoint (line_through_points A' B') P Q :=
by
  sorry

end line_through_midpoint_of_PQ_l600_600225


namespace sum_of_first_n_terms_b_n_l600_600146

noncomputable def a_n (n : ℕ) : ℕ := 3 * n - 2

noncomputable def b_n (n : ℕ) : ℕ := 2 ^ (a_n n) + 2 * n

def S_n (n : ℕ) : ℕ := (∑ i in Finset.range n, b_n (i + 1))

theorem sum_of_first_n_terms_b_n (n : ℕ) : 
  S_n n = (2 / 7) * (8 ^ n - 1) + n * (n + 1) :=
sorry

end sum_of_first_n_terms_b_n_l600_600146


namespace probability_A_mean_X_l600_600726

-- Define the conditions and events
noncomputable def prob_pass_level_1 : ℝ := 3 / 4
noncomputable def prob_pass_level_2 : ℝ := 2 / 3
noncomputable def prob_pass_level_3 : ℝ := 1 / 2
noncomputable def prob_continue : ℝ := 1 / 2

-- Define the events A1 and A2 based on the problem description
noncomputable def event_A1 : ℝ := prob_pass_level_1 * prob_continue * (1 - prob_pass_level_2)
noncomputable def event_A2 : ℝ := prob_pass_level_1 * prob_continue * prob_pass_level_2 * prob_continue * (1 - prob_pass_level_3)

-- Prove the probability that the guest successfully passes the first level but receives zero charity fund
theorem probability_A : (event_A1 + event_A2) = 3 / 16 := sorry

-- Define the probability distribution of X
noncomputable def prob_X_0 : ℝ := 1 - prob_pass_level_1 + (event_A1 + event_A2)
noncomputable def prob_X_1000 : ℝ := prob_pass_level_1 * prob_continue
noncomputable def prob_X_3000 : ℝ := prob_pass_level_1 * prob_continue * prob_pass_level_2 * prob_continue
noncomputable def prob_X_6000 : ℝ := prob_pass_level_1 * prob_continue * prob_pass_level_2 * prob_continue * prob_pass_level_3

-- Prove the mean of X
theorem mean_X : (0 * prob_X_0 + 1000 * prob_X_1000 + 3000 * prob_X_3000 + 6000 * prob_X_6000) = 1125 := sorry

end probability_A_mean_X_l600_600726


namespace Bob_drained_3500_liters_l600_600459

theorem Bob_drained_3500_liters
  (initial_water : ℕ)
  (evaporated_water : ℕ)
  (rain_10min : ℕ)
  (rain_duration : ℕ)
  (final_water : ℕ)
  (H1 : initial_water = 6000)
  (H2 : evaporated_water = 2000)
  (H3 : rain_10min = 350)
  (H4 : rain_duration = 30)
  (H5 : final_water = 1550) :
  let water_after_evaporation := initial_water - evaporated_water in
  let rain_intervals := rain_duration / 10 in
  let rainwater_added := rain_intervals * rain_10min in
  let water_after_rain := water_after_evaporation + rainwater_added in
  water_after_rain - final_water = 3500 :=
by
  sorry

end Bob_drained_3500_liters_l600_600459


namespace students_choice_count_l600_600055

def zodiac_signs := {"rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "sheep", "monkey", "rooster", "dog", "pig"}

def student_A_likes := {"ox", "horse"}
def student_B_likes := {"ox", "dog", "sheep"}
def student_C_likes := zodiac_signs

def satisfied_choice (A_choice B_choice C_choice : String) : Prop :=
  (A_choice ∈ student_A_likes) ∧
  (B_choice ∈ student_B_likes) ∧
  (C_choice ∈ student_C_likes) ∧
  (A_choice ≠ B_choice) ∧
  (A_choice ≠ C_choice) ∧
  (B_choice ≠ C_choice)

def count_choices : ℕ :=
  (zodiac_signs.filter (fun A_choice =>
    (student_B_likes.filter (fun B_choice =>
      (student_C_likes.filter (fun C_choice =>
        satisfied_choice A_choice B_choice C_choice)).card > 0)).card > 0)).card

theorem students_choice_count : count_choices = 50 :=
by
  sorry

end students_choice_count_l600_600055


namespace jack_sixth_quiz_score_l600_600694

theorem jack_sixth_quiz_score (scores : Fin 5 → ℝ) (h_scores : scores = ![76, 82, 79, 84, 91]) 
  (target_mean : ℝ) (h_target_mean : target_mean = 85) :
  ∃ sixth_score : ℝ, (scores.sum + sixth_score) / 6 = target_mean ∧ sixth_score = 98 :=
by
  -- Assume target mean and current scores are given
  sorry

end jack_sixth_quiz_score_l600_600694


namespace g_negative_one_value_l600_600160

noncomputable def f (x : ℝ) : ℝ := sorry

noncomputable def g (x : ℝ) : ℝ := f(x) + 2

axiom f_is_odd : ∀ x, f(-x) = -f(x)

axiom f_one_is_one : f(1) = 1

theorem g_negative_one_value : g(-1) = -1 :=
by
  -- Proof goes here
  sorry

end g_negative_one_value_l600_600160


namespace probability_both_authentic_probability_one_authentic_one_defective_probability_at_least_one_defective_l600_600039

-- Definitions of conditions for the problem
def total_products := 5
def authentic_products := 3
def defective_products := 2
def total_outcomes := 10

-- Theorem for the probability that both selected products are authentic
theorem probability_both_authentic (total_products = 5)
                                   (authentic_products = 3)
                                   (defective_products = 2)
                                   (total_outcomes = 10) :
  3 / 10 = 3 / 10 :=
by
  sorry

-- Theorem for the probability that one selected product is authentic and the other is defective
theorem probability_one_authentic_one_defective (total_products = 5)
                                                 (authentic_products = 3)
                                                 (defective_products = 2)
                                                 (total_outcomes = 10) :
  6 / 10 = 3 / 5 :=
by
  sorry

-- Theorem for the probability that at least one selected product is defective
theorem probability_at_least_one_defective (total_products = 5)
                                           (authentic_products = 3)
                                           (defective_products = 2)
                                           (total_outcomes = 10) :
  7 / 10 = 7 / 10 :=
by
  sorry

end probability_both_authentic_probability_one_authentic_one_defective_probability_at_least_one_defective_l600_600039


namespace Cheryl_more_eggs_than_others_l600_600922

-- Definitions based on conditions
def KevinEggs : Nat := 5
def BonnieEggs : Nat := 13
def GeorgeEggs : Nat := 9
def CherylEggs : Nat := 56

-- Main theorem statement
theorem Cheryl_more_eggs_than_others : (CherylEggs - (KevinEggs + BonnieEggs + GeorgeEggs) = 29) :=
by
  -- Proof would go here
  sorry

end Cheryl_more_eggs_than_others_l600_600922


namespace brick_tower_heights_l600_600558

theorem brick_tower_heights (num_bricks : ℕ) (heights : list ℕ) 
  (h_num_bricks : num_bricks = 80)
  (h_heights : heights = [3, 9, 18]) :
  ∃ (n : ℕ), (n = 401) := sorry

end brick_tower_heights_l600_600558


namespace quadractic_equation_root_l600_600600

theorem quadractic_equation_root (b c : ℝ) (h : c ≠ 0) :
  (∃ x : ℝ, x^2 + 2 * b * x - 5 * c = 0 ∧ x = c) → 2 * b + c = 5 :=
by
  intro h1
  cases h1 with x hx
  cases hx with h_eq h_x_val
  have hc_eq := h_eq.subst (eq.symm h_x_val)
  sorry

end quadractic_equation_root_l600_600600


namespace completing_the_square_l600_600803

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l600_600803


namespace completing_the_square_l600_600818

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l600_600818


namespace total_time_to_complete_l600_600780

noncomputable def time_to_clean_keys (n : Nat) (t : Nat) : Nat := n * t

def assignment_time : Nat := 10
def time_per_key : Nat := 3
def remaining_keys : Nat := 14

theorem total_time_to_complete :
  time_to_clean_keys remaining_keys time_per_key + assignment_time = 52 := by
  sorry

end total_time_to_complete_l600_600780


namespace cheryl_found_more_eggs_l600_600921

theorem cheryl_found_more_eggs :
  let kevin_eggs := 5
  let bonnie_eggs := 13
  let george_eggs := 9
  let cheryl_eggs := 56
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end cheryl_found_more_eggs_l600_600921


namespace diamond_comm_l600_600542

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

theorem diamond_comm (x y : ℝ) : diamond x y = diamond y x := by
  sorry

end diamond_comm_l600_600542


namespace sum_of_areas_of_great_rectangles_l600_600883

/-- A rectangle is termed "great" if the number of square units in its area equals
three times the number of units in its perimeter, and it has integer side lengths.
Calculate the sum of all different possible areas of such great rectangles. -/
theorem sum_of_areas_of_great_rectangles :
  let is_great (a b : ℕ) := a * b = 3 * (2 * a + 2 * b) in
  (finset.univ.product finset.univ).filter (λ (ab : ℕ × ℕ), is_great ab.fst ab.snd)
  .image (λ (ab : ℕ × ℕ), ab.fst * ab.snd)
  .sum = 942 :=
sorry

end sum_of_areas_of_great_rectangles_l600_600883


namespace constant_term_in_binomial_expansion_l600_600613

theorem constant_term_in_binomial_expansion :
  let a := ∫ x in Set.Icc e (e^2), (1 / x)
  a = 1 → constant_term_6 := ∑ r in Finset.range 7, if (12 - 3 * r = 0) then (binom 6 r) * (-1)^r else 0
  constant_term_6 = 15 := by
  sorry

end constant_term_in_binomial_expansion_l600_600613


namespace largest_solution_l600_600944

theorem largest_solution (x : ℝ) (h1 : ⌊x⌋ = 6 + 50 * (x - ⌊x⌋)) 
  (h2 : 0 ≤ (x - ⌊x⌋) ∧ (x - ⌊x⌋) < 1) (h3 : ⌊x⌋ ∈ ℤ) : 
  x ≤ 55.98 :=
sorry

end largest_solution_l600_600944


namespace exponential_inequality_example_l600_600759

theorem exponential_inequality_example (a b : ℝ) (h : 1.5 > 0 ∧ 1.5 ≠ 1) (h2 : 2.3 < 3.2) : 1.5 ^ 2.3 < 1.5 ^ 3.2 :=
by 
  sorry

end exponential_inequality_example_l600_600759


namespace exists_set_avg_subset_power_l600_600721

-- We state the main theorem to prove
theorem exists_set_avg_subset_power (n : ℕ) : 
  ∃ (X : Finset ℕ), 
  (X.card = n) ∧ 
  (∀ (S : Finset ℕ), S ⊆ X ∧ S.nonempty → 
    ∃ (m : ℕ), (↑(S.sum id) / S.card : ℚ) = ↑(m^2) ∨ 
                 ↑(S.sum id) / S.card = ↑(m^3) ∨ 
                 (∃ k : ℕ, k > 3 ∧ ↑(S.sum id) / S.card = ↑(m^k))) :=
sorry

end exists_set_avg_subset_power_l600_600721


namespace salt_solution_mixture_l600_600862

theorem salt_solution_mixture (x : ℝ) 
    (h1 : 0.15 * (1 + x) = 0.45 * x) : x = 0.5 :=
  sorry

end salt_solution_mixture_l600_600862


namespace min_value_quadratic_l600_600119

theorem min_value_quadratic (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 1) :
  (∀ x, (a * x^2 + 2 * x + b = 0) → x = -1 / a) →
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ (∀ a b, a > b → b > 0 → a * b = 1 →
     c ≤ (a^2 + b^2) / (a - b)) :=
by
  sorry

end min_value_quadratic_l600_600119


namespace inequality_proof_l600_600981

theorem inequality_proof
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (ha1 : 0 < a1) (ha2 : 0 < a2) (ha3 : 0 < a3)
  (hb1 : 0 < b1) (hb2 : 0 < b2) (hb3 : 0 < b3) :
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 ≥
    4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) :=
sorry

end inequality_proof_l600_600981


namespace timothy_movies_count_l600_600375

variable (T : ℕ)

def timothy_movies_previous_year (T : ℕ) :=
  let timothy_2010 := T + 7
  let theresa_2010 := 2 * (T + 7)
  let theresa_previous := T / 2
  T + timothy_2010 + theresa_2010 + theresa_previous = 129

theorem timothy_movies_count (T : ℕ) (h : timothy_movies_previous_year T) : T = 24 := 
by 
  sorry

end timothy_movies_count_l600_600375


namespace integral_log_sin_l600_600952

theorem integral_log_sin (h : ∫ (x : ℝ) in 0..(Real.pi / 2), log (sin (2 * x)) =
  ∫ (x : ℝ) in 0..(Real.pi / 2), log (sin x) + ∫ (x : ℝ) in 0..(Real.pi / 2), log (cos x) + ∫ (x : ℝ) in 0..(Real.pi / 2), log 2) :
  ∫ (x : ℝ) in 0..(Real.pi / 2), log (sin x) = - (Real.pi / 2) * log 2 :=
by {
  sorry
}

end integral_log_sin_l600_600952


namespace count_valid_k_l600_600090

theorem count_valid_k (n : ℕ) (h : n = 454500) :
  {k : ℕ | k ≤ n ∧ 505 ∣ (k * (k - 1))}.to_finset.card = 3600 :=
by
  sorry

end count_valid_k_l600_600090


namespace area_of_square_field_l600_600727

theorem area_of_square_field (t : ℝ) (v : ℝ) (h1 : t = 6.0008333333333335) (h2 : v = 1.2) :
  let D := v * t * 1000 / 3600 in
  let s := D / (Real.sqrt 2) in
  let A := s ^ 2 in
  A = 25939744.8 :=
by
  -- Variables
  let D := v * t * 1000 / 3600
  let s := D / (Real.sqrt 2)
  let A := s ^ 2
  -- Proof step (skipped)
  sorry

end area_of_square_field_l600_600727


namespace circle_center_positive_x_axis_circle_equation_max_value_sqrt3m_plus_n_l600_600597

noncomputable def line_eqn (x : ℝ) : ℝ := (sqrt 3 / 3) * x

noncomputable def circle_eqn (x y a : ℝ) : Prop := (x - a) ^ 2 + y ^ 2 = 4

theorem circle_center_positive_x_axis (a : ℝ) (ha : 0 < a) : 
  (sqrt 3 / 3 * a) / sqrt (1 + 1 / 3) = 1 → a = 2 :=
by
  sorry

theorem circle_equation : ∃ a, 0 < a ∧ circle_eqn 2 0 a :=
by
  have h : (sqrt 3 / 3 * 2) / sqrt (1 + 1 / 3) = 1 := by sorry
  use 2
  split
  · exact zero_lt_two
  · exact h

noncomputable def circle_P_eqn (m n : ℝ) : Prop := (m - 2) ^ 2 + n ^ 2 = 4

theorem max_value_sqrt3m_plus_n (m n : ℝ) : 
  circle_P_eqn m n → ∃ α : ℝ, (m = 2 + 2 * Real.cos α) ∧ (n = 2 * Real.sin α)
  →  ∃ M : ℝ, ∀ α : ℝ, 2 * sqrt 3 + 4 * Real.sin (α + π / 3) ≤ M
  := by
  sorry

end circle_center_positive_x_axis_circle_equation_max_value_sqrt3m_plus_n_l600_600597


namespace radius_of_circle_l600_600757

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = Real.pi * r ^ 2) : r = 6 := 
by 
  sorry

end radius_of_circle_l600_600757


namespace moving_circle_passes_through_fixed_point_l600_600441
-- We will start by importing the necessary libraries and setting up the problem conditions.

-- Define the parabola y^2 = 8x.
def parabola (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = 8 * p.1

-- Define the line x + 2 = 0.
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.1 = -2

-- Define the fixed point.
def fixed_point : ℝ × ℝ :=
  (2, 0)

-- Define the moving circle passing through the fixed point.
def moving_circle (p : ℝ × ℝ) (c : ℝ × ℝ) :=
  p = fixed_point

-- Bring it all together in the theorem.
theorem moving_circle_passes_through_fixed_point (c : ℝ × ℝ) (p : ℝ × ℝ)
  (h_parabola : parabola c)
  (h_tangent : tangent_line p) :
  moving_circle p c :=
sorry

end moving_circle_passes_through_fixed_point_l600_600441


namespace smallest_x_for_f_2002_l600_600006

noncomputable def f (x : ℝ) : ℝ :=
if 2 ≤ x ∧ x ≤ 4 then 2 - |x - 3| else 0 

theorem smallest_x_for_f_2002 :
  ∃ x : ℝ, x > 0 ∧ f (2002) = f (x) ∧ (∀ y : ℝ, y > 0 ∧ f (2002) = f (y) → x ≤ y) :=
begin
  let fx := 2002 / 1024.0,
  have h1 : 2 ≤ fx ∧ fx ≤ 4,
  {
    sorry, -- This is a placeholder for proving fx is within [2, 4]
  },
  let f2002 := 1024 * (2 - |fx - 3|),
  have h2 : f 2002 = f2002,
  {
    sorry, -- This is a placeholder for proving f(2002) = 1094
  },
  let y := 3 - 954 / 1024,
  have hy : 2 ≤ y ∧ y ≤ 4,
  {
    sorry, -- This is a placeholder for proving y is within [2, 4]
  },
  let smallest_x := 4^5 * y,
  use smallest_x,
  split,
  {
    sorry, -- Prove that smallest_x > 0
  },
  split,
  {
    rw <- h2,
    have hy_eq : f y = 2 - |y - 3|,
    {
      sorry, -- This is a placeholder for the functional equation f(x) = 2 - |x - 3|
    },
    have hx : f smallest_x = 1094,
    {
      sorry, -- Prove f(smallest_x) = 1094 using the definition and functional properties of f
    },
    exact hx,
  },
  {
    intros y hy_conditions,
    exact sorry, -- This is a placeholder for proving minimality of smallest_x
  }
end

end smallest_x_for_f_2002_l600_600006


namespace solution_set_inequality_l600_600213

theorem solution_set_inequality (a m : ℝ) (h : ∀ x : ℝ, (x > m ∧ x < 1) ↔ 2 * x^2 - 3 * x + a < 0) : m = 1 / 2 :=
by
  -- Insert the proof here
  sorry

end solution_set_inequality_l600_600213


namespace sailboat_rental_cost_l600_600237

-- Define the conditions
def rental_per_hour_ski := 80
def hours_per_day := 3
def days := 2
def cost_ski := (hours_per_day * days * rental_per_hour_ski)
def additional_cost := 120

-- Statement to prove
theorem sailboat_rental_cost :
  ∃ (S : ℕ), cost_ski = S + additional_cost → S = 360 := by
  sorry

end sailboat_rental_cost_l600_600237


namespace transformed_interval_l600_600062

noncomputable def transformation (x : ℝ) : ℝ := 8 * x - 2

theorem transformed_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2 ≤ transformation x ∧ transformation x ≤ 6 := by
  intro x h
  unfold transformation
  sorry

end transformed_interval_l600_600062


namespace exists_positive_x_le_sqrt_x_add_one_l600_600336

theorem exists_positive_x_le_sqrt_x_add_one (h : ∀ x > 0, √x > x + 1) :
  ∃ x > 0, √x ≤ x + 1 :=
sorry

end exists_positive_x_le_sqrt_x_add_one_l600_600336


namespace diameter_of_circle_l600_600418

theorem diameter_of_circle (A : ℝ) (h : A = 4 * real.pi) : ∃ d : ℝ, d = 4 :=
  sorry

end diameter_of_circle_l600_600418


namespace find_m_plus_n_l600_600023

noncomputable def k (x : ℝ) : ℝ :=
  let A_c := (√13 / 2 + 1) * π * x^2
  let A_f := (16 * π + 8 * π * √13) - (√13 / 2 + 1) * π * x^2
  let V_c := 1 / 2 * π * x^3
  let V_f := 32 * π - 1 / 2 * π * x^3
  A_c / A_f

theorem find_m_plus_n : 
  ∃ m n : ℕ, k (8 * √13 / 9) = (169 : ℝ) / (775 : ℝ)
  ∧ nat.coprime m n 
  ∧ m + n = 944 :=
sorry

end find_m_plus_n_l600_600023


namespace radius_of_circle_l600_600748

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * real.pi * r) = real.pi * r^2) : r = 6 :=
by {
    sorry
}

end radius_of_circle_l600_600748


namespace factorize_x_cubed_minus_9x_l600_600932

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l600_600932


namespace same_terminal_side_angle_l600_600828

theorem same_terminal_side_angle (θ1 θ2 : ℝ) (k : ℤ) :
   θ1 = 1303 ∧ θ2 = -137 ∧ θ1 - θ2 = k * 360 ↔ θ1 ≡ θ2 [ZMOD 360] :=
by
  sorry

end same_terminal_side_angle_l600_600828


namespace merchant_loss_l600_600012

theorem merchant_loss (n m : ℝ) (h₁ : n ≠ m) : 
  let x := n / m
  let y := m / n
  x + y > 2 := by
sorry

end merchant_loss_l600_600012


namespace math_problem_l600_600371

theorem math_problem
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : 2 * x * y = 5) : 
  ∃ (a b c d : ℕ), (x = (a + b * real.sqrt c) / d ∨ x = (a - b * real.sqrt c) / d) ∧ (a + b + c + d = 23) := 
by
  sorry

end math_problem_l600_600371


namespace numberOfChromiumAtoms_l600_600866

noncomputable def molecularWeightOfCompound : ℕ := 296
noncomputable def atomicWeightOfPotassium : ℝ := 39.1
noncomputable def atomicWeightOfOxygen : ℝ := 16.0
noncomputable def atomicWeightOfChromium : ℝ := 52.0

def numberOfPotassiumAtoms : ℕ := 2
def numberOfOxygenAtoms : ℕ := 7

theorem numberOfChromiumAtoms
    (mw : ℕ := molecularWeightOfCompound)
    (awK : ℝ := atomicWeightOfPotassium)
    (awO : ℝ := atomicWeightOfOxygen)
    (awCr : ℝ := atomicWeightOfChromium)
    (numK : ℕ := numberOfPotassiumAtoms)
    (numO : ℕ := numberOfOxygenAtoms) :
  numK * awK + numO * awO + (mw - (numK * awK + numO * awO)) / awCr = 2 := 
by
  sorry

end numberOfChromiumAtoms_l600_600866


namespace f_max_min_l600_600839

def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom cauchy_f : ∀ x y : ℝ, f (x + y) = f x + f y
axiom less_than_zero : ∀ x : ℝ, x > 0 → f x < 0
axiom f_one : f 1 = -2

theorem f_max_min : (∀ x ∈ [-3, 3], f (-3) = 6 ∧ f 3 = -6) :=
by sorry

end f_max_min_l600_600839


namespace simplify_expression_l600_600488

-- Define the expression to be simplified
def expression : ℝ := (sqrt 27 / (sqrt 3 / 2)) * (2 * sqrt 2) - 6 * sqrt 2

-- State the theorem to be proven
theorem simplify_expression : expression = 6 * sqrt 2 :=
by
  sorry

end simplify_expression_l600_600488


namespace two_color_plane_division_l600_600303

theorem two_color_plane_division (n : ℕ) 
  (lines : Fin n → Set (ℝ × ℝ)) 
  (divides_plane : ∀ (i : Fin n), ∃ (f : ℝ → ℝ), ∀ (x y: ℝ), (x, y) ∈ lines i ↔ y = f x) :
  ∃ (coloring : Set (ℝ × ℝ) → fin 2),
  ∀ (region1 region2 : Set (ℝ × ℝ)), 
  (region1 ∩ region2).Nonempty → 
  coloring region1 ≠ coloring region2 :=
by
  sorry

end two_color_plane_division_l600_600303


namespace trig_identity_solution_l600_600955

theorem trig_identity_solution
  (α : ℝ) (β : ℝ)
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan β = -1 / 3) :
  (3 * Real.sin α * Real.cos β - Real.sin β * Real.cos α) / (Real.cos α * Real.cos β + 2 * Real.sin α * Real.sin β) = 11 / 4 :=
by
  sorry

end trig_identity_solution_l600_600955


namespace ann_fare_90_miles_l600_600896

-- Define the conditions as given in the problem
def fare (distance : ℕ) : ℕ := 30 + distance * 2

-- Theorem statement
theorem ann_fare_90_miles : fare 90 = 210 := by
  sorry

end ann_fare_90_miles_l600_600896


namespace b_general_formula_sum_first_20_terms_is_300_l600_600143

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a(n) + 1 else a(n) + 2

-- Define the sequence b_n as bₙ = a₂ₙ
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof goal for part (1)
theorem b_general_formula (n : ℕ) : b n = 3 * n - 1 := sorry

-- Sum of the first 20 terms of the sequence a_n
def sum_first_20_terms : ℕ :=
  (List.range (20)).sum (λ n, a n)

-- Proof goal for part (2)
theorem sum_first_20_terms_is_300 : sum_first_20_terms = 300 := sorry

end b_general_formula_sum_first_20_terms_is_300_l600_600143


namespace probability_A_adjacent_to_B_not_C_l600_600122

theorem probability_A_adjacent_to_B_not_C :
  let people := ['A', 'B', 'C', 'D', 'E', 'F'] in
  let total_arrangements := Finset.permutations ('A'::'B'::'C'::'D'::'E'::'F'::[]) in
  let favorable_arrangements := { p ∈ total_arrangements | 
    let adj := (zip p.tail! p) ++ (zip p p.tail!) in
    ('A', 'B') ∈ adj ∧ ('B', 'A') ∈ adj ∧  
    ¬(('A', 'C') ∈ adj ∨ ('C', 'A') ∈ adj) } in
  (favorable_arrangements.card : ℚ) / total_arrangements.card = 4 / 15 :=
by
  -- The proof is omitted
  sorry

end probability_A_adjacent_to_B_not_C_l600_600122


namespace problem_statement_min_value_g_l600_600165

noncomputable def f (ω x : ℝ) : ℝ := sin (π - ω * x) * cos (ω * x) + cos (ω * ω * x)

def period_condition (ω : ℝ) : Prop := ∃ T > 0, ∀ x, f ω (x + T) = f ω x ∧ T = π

noncomputable def g (x : ℝ) : ℝ := f 1 (2 * x)

theorem problem_statement :
  ∀ (ω : ℝ), ω > 0 ∧ period_condition ω → ω = 1 :=
sorry

theorem min_value_g :
  ∀ x ∈ set.Icc 0 (π / 16), 1 ≤ g x :=
sorry

end problem_statement_min_value_g_l600_600165


namespace sum_greater_than_2_l600_600180

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : -1 < x ∧ x < 1 then log a (x + 1)
else log a (3 - x) + a - 1

theorem sum_greater_than_2 (a : ℝ) (x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) 
  (h₂ : f a x₁ = f a x₂) (h₃ : x₁ ≠ x₂) (h₄ : -1 < x₁ ∧ x₁ < 1) (h₅ : 1 < x₂ ∧ x₂ < 3) : 
  x₁ + x₂ > 2 := 
by 
  sorry

end sum_greater_than_2_l600_600180


namespace exists_palindromic_product_l600_600295

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let digits := toDigits 10 n
  digits = digits.reverse

theorem exists_palindromic_product (x : ℕ) (hx : ¬ (10 ∣ x)) : ∃ y : ℕ, is_palindrome (x * y) :=
by
  -- Prove that there exists a natural number y such that x * y is a palindromic number
  sorry

end exists_palindromic_product_l600_600295


namespace speed_conversion_l600_600917

theorem speed_conversion (x : ℚ) (conversion_factor : ℚ) : 
  (x : ℚ) = (15 / 36) → conversion_factor = 3.6 → (x * conversion_factor = 1.5) := by
  intros hx hcf
  rw hx
  rw hcf
  norm_num
  sorry

end speed_conversion_l600_600917


namespace speed_in_still_water_l600_600400

/-- A man can row upstream at 37 km/h and downstream at 53 km/h, 
    prove that the speed of the man in still water is 45 km/h. --/
theorem speed_in_still_water 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ)
  (h1 : upstream_speed = 37)
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := 
by 
  sorry

end speed_in_still_water_l600_600400


namespace f_neg_2017_l600_600176

noncomputable def f : ℝ → ℝ :=
λ x, if x > 2 then f (x - 4)
      else if -2 ≤ x ∧ x ≤ 2 then Real.exp x
      else f (-x)

theorem f_neg_2017 : f (-2017) = Real.exp 1 := by
  sorry

end f_neg_2017_l600_600176


namespace clayton_shells_l600_600253

theorem clayton_shells :
  ∀ (shells_jillian shells_savannah shells_per_friend total_friends shells_clayton : ℕ),
    shells_jillian = 29 →
    shells_savannah = 17 →
    shells_per_friend = 27 →
    total_friends = 2 →
    shells_clayton = (shells_per_friend * total_friends) - (shells_jillian + shells_savannah) →
    shells_clayton = 8 :=
by
  intros shells_jillian shells_savannah shells_per_friend total_friends shells_clayton
  intro h_jillian
  intro h_savannah
  intro h_per_friend
  intro h_total_friends
  intro h_clayton
  rw [← h_jillian, ← h_savannah, ← h_per_friend, ← h_total_friends, ← h_clayton]
  sorry

end clayton_shells_l600_600253


namespace mary_less_euros_l600_600711

def initial_amounts : ℝ × ℝ × ℝ × ℝ := (30, 18, 24, 1500)  -- amounts in USD, GBP, EUR, JPY respectively
def conversion_rates : ℝ × ℝ × ℝ := (0.85, 1.15, 0.77)      -- USD to EUR, GBP to EUR, JPY to EUR/100 respectively

def final_amounts (u : ℝ) (g : ℝ) (e : ℝ) (j : ℝ) 
  (r_usd_eur : ℝ) (r_gbp_eur : ℝ) (r_jpy_eur : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let m_usd := u - 0.60 * u in                             -- Michelle's remaining USD
  let a_usd_in_eur := 0.60 * u * r_usd_eur in              -- Alice's received EUR from Michelle's USD
  let a_gbp_in_eur := g * r_gbp_eur in                     -- Alice's GBP converted to EUR
  let a_eur := a_usd_in_eur + a_gbp_in_eur in              -- Alice's total EUR
  let m_gives := 0.50 * e in                               -- Marco gives 50% of his EUR
  let marco_remaining_eur := e - m_gives in                -- Marco's remaining EUR
  let mary_remaining_jpy := j - 600 in                     -- Mary spends 600 JPY
  let mary_remaining_eur := mary_remaining_jpy / 100 * r_jpy_eur in -- Mary's remaining EUR
  (m_usd * r_usd_eur, a_eur, marco_remaining_eur, mary_remaining_eur) -- Final combined amounts

theorem mary_less_euros :
  let (usd, gbp, eur, jpy) := initial_amounts in
  let (r_usd_eur, r_gbp_eur, r_jpy_eur) := conversion_rates in
  let (michelle_eur, alice_eur, marco_eur, mary_eur) := final_amounts usd gbp eur jpy r_usd_eur r_gbp_eur r_jpy_eur in
  mary_eur - (michelle_eur + alice_eur + marco_eur) = -51.27 := 
by
  sorry

end mary_less_euros_l600_600711


namespace find_f2_l600_600330

theorem find_f2 :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 2 * f x - 3 * f (1 / x) = x ^ 2) ∧ f 2 = 93 / 32) :=
sorry

end find_f2_l600_600330


namespace part1_b1_b2_part1_general_formula_part2_sum_20_l600_600137

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then a n + 1 else a n + 2

def b (n : ℕ) : ℕ := a (2 * n)

-- Proving b_1 = 2 and b_2 = 5
theorem part1_b1_b2 : b 1 = 2 ∧ b 2 = 5 :=
by {
  unfold b a,
  simp,
  split,
  {
    rfl -- proof for b_1 = 2
  },
  {
    rfl -- proof for b_2 = 5
  }
}

-- Proving the general formula for b_n
theorem part1_general_formula (n : ℕ) : b n = 3 * n - 1 :=
by {
  induction n with k ih,
  {
    unfold b a,
    simp,
    rfl
  },
  {
    rename ih ih_k,
    unfold b a,
    simp,
    rw [ih_k],
    calc 3 * (k + 1) - 1 = 3 * k + 3 - 1 : by ring
                       ... = 3 * k + 2     : by ring
                       ... = a (2 * k + 2) : by sorry -- Detailed proof needed
  }
}

-- Proving the sum of the first 20 terms of the sequence a_n
theorem part2_sum_20 : (Finset.range 20).sum a = 300 :=
by {
  unfold a,
  have h1 : finset.sum (finset.range 10) (λ n, 3 * n + 1) = 145,
  {
    sorry -- Compute sum of odd terms
  },
  have h2 : finset.sum (finset.range 10) (λ n, 3 * n + 2) = 155,
  {
    sorry -- Compute sum of even terms
  },
  have h3 : finset.sum (finset.range 20) a = 145 + 155,
  {
    sorry -- Combine sums
  },
  exact h3,
}

end part1_b1_b2_part1_general_formula_part2_sum_20_l600_600137


namespace election_votes_distribution_l600_600676

theorem election_votes_distribution:
  ∃ (T c1 c2 c3 c4 c5 : ℕ),
    (c1 = 0.30 * T) ∧
    (c2 = 0.20 * T) ∧
    (0.15 * T = 3000) ∧
    (c4 = 0.25 * T) ∧
    (c5 = 2 * 3000) ∧
    (T = 20000) ∧
    (c1 = 6000) ∧
    (c2 = 4000) ∧
    (c3 = 3000) ∧
    (c4 = 5000) ∧
    (c5 = 6000) :=
by
  sorry

end election_votes_distribution_l600_600676


namespace average_price_of_fruit_l600_600050

theorem average_price_of_fruit :
  ∃ (A O : ℕ), A + O = 10 ∧ (40 * A + 60 * (O - 4)) / (A + O - 4) = 50 → 
  (40 * A + 60 * O) / 10 = 54 :=
by
  sorry

end average_price_of_fruit_l600_600050


namespace distance_between_parallel_lines_l600_600058

open Real EuclideanGeometry

def point (x y : ℝ) : ℝ × ℝ := (x, y)

def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

def dot_prod (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

def proj (a b : ℝ × ℝ) : ℝ × ℝ := 
  let coeff := (dot_prod a b) / (dot_prod b b)
  (coeff * b.1, coeff * b.2)

def norm (a : ℝ × ℝ) : ℝ := sqrt (a.1^2 + a.2^2)

theorem distance_between_parallel_lines :
  let a : ℝ × ℝ := point 3 (-4)
  let b : ℝ × ℝ := point 0 0
  let d : ℝ × ℝ := point 2 (-3)
  let v := vector_sub a b
  let p := proj v d
  let orthogonal_part := vector_sub v p
  in norm orthogonal_part = sqrt 13 / 13 :=
by
  sorry

end distance_between_parallel_lines_l600_600058


namespace probability_both_universities_visited_l600_600463

open ProbabilityTheory

theorem probability_both_universities_visited :
  (∃ (students : Fin 4 → Bool),
    (∃ i, students i = true) ∧ (∃ j, students j = false)) → 
  (1 - ((1 / 2) ^ 4 + (1 / 2) ^ 4)) = 7 / 8 :=
by
  sorry

end probability_both_universities_visited_l600_600463


namespace cylindrical_container_dimensions_l600_600479

def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem cylindrical_container_dimensions :
  ∃ r h : ℝ, volume_of_cylinder r h = 100 ∧ h = 2 * r ∧ r ≈ 2.515 ∧ h ≈ 5.03 :=
by
  sorry

end cylindrical_container_dimensions_l600_600479


namespace distribution_schemes_l600_600553

-- Definitions for the problem conditions
def students : Set String := {"Student1", "Student2", "Student3", "Student4"}
def labs : Set String := {"LabA", "LabB", "LabC"}

-- The total number of different distribution schemes, ensuring each lab receives at least one student
theorem distribution_schemes : ∃ distribs : finset (students → labs), distribs.card = 36 :=
by
  sorry

end distribution_schemes_l600_600553


namespace magnitude_of_z_l600_600125

open Complex

theorem magnitude_of_z (a : ℝ) (h_imag : z.im = 2) (h_pure_imag : (z^2 + 3).re = 0) : abs z = sqrt 5 :=
by sorry

end magnitude_of_z_l600_600125


namespace percentage_error_calc_l600_600026

theorem percentage_error_calc (x : ℝ) (h : x ≠ 0) : 
  let correct_result := x * (5 / 3)
  let incorrect_result := x * (3 / 5)
  let percentage_error := (correct_result - incorrect_result) / correct_result * 100
  percentage_error = 64 := by
  sorry

end percentage_error_calc_l600_600026


namespace find_a_of_max_five_l600_600967

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  abs (a * sin (2 * x) + cos (2 * x)) + 
  abs (sin x + cos x) * abs ((1 + a) * sin x + (1 - a) * cos x)

theorem find_a_of_max_five (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 5) ↔ a = sqrt 3 ∨ a = -sqrt 3 := 
sorry

end find_a_of_max_five_l600_600967


namespace factorize_x_cube_minus_9x_l600_600927

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l600_600927


namespace Walter_gets_49_bananas_l600_600244

variable (Jefferson_bananas : ℕ) (Walter_bananas : ℕ) (total_bananas : ℕ) (shared_bananas : ℕ)

def problem_conditions : Prop :=
  Jefferson_bananas = 56 ∧ Walter_bananas = Jefferson_bananas - (Jefferson_bananas / 4)

theorem Walter_gets_49_bananas (h : problem_conditions) : 
  let combined_bananas := Jefferson_bananas + Walter_bananas in
  let shared_bananas := combined_bananas / 2 in
  shared_bananas = 49 :=
by
  sorry

end Walter_gets_49_bananas_l600_600244


namespace find_f_f_2_l600_600179

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 0.5 else 3^x

theorem find_f_f_2 : f (f 2) = 1 / 3 :=
by
  sorry

end find_f_f_2_l600_600179


namespace b1_b2_values_general_formula_b_sum_first_20_l600_600133

def seq_a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then seq_a n + 1 else seq_a n + 2

def seq_b (n : ℕ) : ℕ := seq_a (2 * n)

theorem b1_b2_values : seq_b 1 = 2 ∧ seq_b 2 = 5 := by
  sorry

theorem general_formula_b (n : ℕ) : seq_b n = 3 * n - 1 := by
  sorry

theorem sum_first_20 : (Finset.range 20).sum seq_a = 300 := by
  sorry

end b1_b2_values_general_formula_b_sum_first_20_l600_600133


namespace circle_diameter_length_l600_600427

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l600_600427


namespace same_solution_m_iff_m_eq_2_l600_600584

theorem same_solution_m_iff_m_eq_2 (m y : ℝ) (h1 : my - 2 = 4) (h2 : y - 2 = 1) : m = 2 :=
by {
  sorry
}

end same_solution_m_iff_m_eq_2_l600_600584


namespace correct_operation_l600_600830

theorem correct_operation (a : ℝ) : 2 * (a^2) * a = 2 * (a^3) := by sorry

end correct_operation_l600_600830


namespace even_product_probability_l600_600588

open Probability

noncomputable def probability_even_product : ℚ :=
  let total_outcomes := 6 * 6 * 6,
      odd_outcomes := 3 * 3 * 3,
      even_outcomes := total_outcomes - odd_outcomes in
  even_outcomes / total_outcomes

theorem even_product_probability :
  probability_even_product = 7 / 8 :=
sorry

end even_product_probability_l600_600588


namespace median_from_D_l600_600405

section MediansInTetrahedron

variables (A B C D S : Point)

-- Assume that S is the centroid of the face ABC
axiom centroid_S (A B C S : Point) : centroid A B C S

-- The theorem stating the median length formula
theorem median_from_D (DS_squared DA_squared DB_squared DC_squared AB_squared BC_squared CA_squared : ℝ)
  (hS : centroid_S A B C S) :
  DS_squared = (1 / 3) * (DA_squared + DB_squared + DC_squared) - (1 / 9) * (AB_squared + BC_squared + CA_squared) :=
sorry

end MediansInTetrahedron

end median_from_D_l600_600405


namespace variance_of_scores_l600_600027

def scores : List ℕ := [110, 114, 121, 119, 126]

noncomputable def mean (l : List ℕ) : ℝ := (l.sum : ℝ) / (l.length : ℝ)

noncomputable def variance (l : List ℕ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x : ℝ - μ) ^ 2)).sum / (l.length)

theorem variance_of_scores : variance scores = 30.8 := sorry

end variance_of_scores_l600_600027


namespace calculate_expression_l600_600501

theorem calculate_expression :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2 = 6 * real.sqrt 2) :=
by
  sorry

end calculate_expression_l600_600501


namespace sum_of_integral_c_with_rational_roots_l600_600103

theorem sum_of_integral_c_with_rational_roots
  (h : ∀ c : ℤ, c ≤ 30 → ∃ k : ℤ, k^2 = 81 + 4*c ∧ ∃ p q : ℚ, p*q = c ∧ p + q = 9) :
  ∑ c in (Finset.filter (λ c : ℤ, ∃ k : ℤ, k^2 = 81 + 4*c) (Finset.Icc (-20) 30)), c = -28 :=
by
  sorry

end sum_of_integral_c_with_rational_roots_l600_600103


namespace more_minutes_per_question_l600_600255

theorem more_minutes_per_question 
  (english_questions: ℕ) (math_questions: ℕ) 
  (english_time: ℝ) (math_time: ℝ) 
  (english_questions = 50) 
  (math_questions = 20) 
  (english_time = 80) 
  (math_time = 110) : 
  (math_time / math_questions) - (english_time / english_questions) = 3.9 := 
by
  sorry

end more_minutes_per_question_l600_600255


namespace found_ratio_l600_600042

namespace Rhinestones

def total_rhinestones_needed : ℕ := 45

def rhinestones_bought : ℕ := total_rhinestones_needed / 3

def remaining_rhinestones_needed : ℕ := 21

def rhinestones_acquired : ℕ := total_rhinestones_needed - remaining_rhinestones_needed

def rhinestones_found_in_supplies : ℕ := rhinestones_acquired - rhinestones_bought

def ratio_found_to_needed : ratio := (rhinestones_found_in_supplies, total_rhinestones_needed)

theorem found_ratio : ratio_found_to_needed = (1, 5) :=
by
-- placeholder for the proof
sorry

end Rhinestones

end found_ratio_l600_600042


namespace A_and_B_finish_work_together_in_12_days_l600_600028

theorem A_and_B_finish_work_together_in_12_days 
  (T_B : ℕ) 
  (T_A : ℕ)
  (h1 : T_B = 18) 
  (h2 : T_A = 2 * T_B) : 
  1 / (1 / T_A + 1 / T_B) = 12 := 
by 
  sorry

end A_and_B_finish_work_together_in_12_days_l600_600028


namespace sqrt_expr_simplification_l600_600510

theorem sqrt_expr_simplification :
  (real.sqrt 27) / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - (6 * real.sqrt 2) = (6 * real.sqrt 2) :=
by
  sorry

end sqrt_expr_simplification_l600_600510


namespace cot_theta_implies_zero_l600_600117

variable (θ : ℝ) -- Define variable θ in ℝ

-- Define the condition cot θ = 3
axiom cot_theta_eq_three : Real.cot θ = 3

-- Define the problem statement
theorem cot_theta_implies_zero :
  Real.cot θ = 3 →
  (\frac{1 - Real.sin θ}{Real.cos θ}) - (\frac{Real.cos θ}{1 + Real.sin θ}) = 0 :=
by
  intro h
  sorry

end cot_theta_implies_zero_l600_600117


namespace four_rectangles_arrangement_exists_l600_600692

theorem four_rectangles_arrangement_exists :
  ∃ (R₁ R₂ R₃ R₄ : Type) (A B C D : R₁ ∩ R₂ ∩ R₃ ∩ R₄ → Prop),
    (¬ (∀ x, A x ∧ B x ∧ C x ∧ D x)) ∧
    (∀ (x : R₁ ∩ R₂), A x ∧ B x) ∧
    (∀ (x : R₂ ∩ R₃), B x ∧ C x) ∧
    (∀ (x : R₃ ∩ R₄), C x ∧ D x) ∧
    (∀ (x : R₄ ∩ R₁), D x ∧ A x) :=
begin
  sorry
end

end four_rectangles_arrangement_exists_l600_600692


namespace smallest_n_l600_600556

-- Define the costs of candies
def cost_purple := 24
def cost_yellow := 30

-- Define the number of candies Lara can buy
def pieces_red := 10
def pieces_green := 16
def pieces_blue := 18
def pieces_yellow := 22

-- Define the total money Lara has equivalently expressed by buying candies
def lara_total_money (n : ℕ) := n * cost_purple

-- Prove the smallest value of n that satisfies the conditions stated
theorem smallest_n : ∀ n : ℕ, 
  (lara_total_money n = 10 * pieces_red * cost_purple) ∧
  (lara_total_money n = 16 * pieces_green * cost_purple) ∧
  (lara_total_money n = 18 * pieces_blue * cost_purple) ∧
  (lara_total_money n = pieces_yellow * cost_yellow) → 
  n = 30 :=
by
  intro
  sorry

end smallest_n_l600_600556


namespace ratio_of_volumes_l600_600897

-- Definitions of the initial conditions
def alex_diameter : ℝ := 8
def alex_height : ℝ := 16
def felicia_diameter : ℝ := 10
def felicia_height : ℝ := 8

-- Definition of radius calculated from diameter
def alex_radius : ℝ := alex_diameter / 2
def felicia_radius : ℝ := felicia_diameter / 2

-- Definition of volume of a cylinder given its radius and height
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Volumes of Alex's and Felicia's cans
def alex_volume : ℝ := volume alex_radius alex_height
def felicia_volume : ℝ := volume felicia_radius felicia_height

-- The goal is to prove that the ratio of Alex's can volume to Felicia's can volume is 32/25
theorem ratio_of_volumes : alex_volume / felicia_volume = 32 / 25 := by
  sorry

end ratio_of_volumes_l600_600897


namespace prob_at_least_one_three_l600_600876

variable (X1 X2 X3 : ℕ)
variable (hx1 : X1 ∈ {1, 2, 3, 4, 5, 6})
variable (hx2 : X2 ∈ {1, 2, 3, 4, 5, 6})
variable (hx3 : X3 ∈ {1, 2, 3, 4, 5, 6})
variable (h_condition : X1 + X2 = 2 * X3)

theorem prob_at_least_one_three : 
  (∃ X1 X2 X3, X1 ∈ {1, 2, 3, 4, 5, 6} ∧ X2 ∈ {1, 2, 3, 4, 5, 6} ∧ X3 ∈ {1, 2, 3, 4, 5, 6} ∧ X1 + X2 = 2 * X3) →
  ((∃ X1, X1 ∈ {3}) ∨ (∃ X2, X2 ∈ {3}) ∨ (∃ X3, X3 ∈ {3})) :=
sorry

end prob_at_least_one_three_l600_600876


namespace problem_equivalence_l600_600161

noncomputable def ellipse : Prop :=
  ∃ a b : ℝ, a > b ∧ 2 * a * b = 2 * sqrt 2 ∧ (a^2 = 2 ∧ b^2 = 1) ∧
  (∀ x y, (x,y) ∈ {(x,y) | x^2 / a^2 + y^2 / b^2 = 1} ↔ (x,y) ∈ {(x,y) | x^2 / 2 + y^2 = 1})

noncomputable def max_area_line : Prop :=
  ∃ k : ℝ, (k = sqrt 14 / 2 ∨ k = -sqrt 14 / 2) ∧ 
    (∀ x y, y = k * x + 2 ∧ (x,y) ∈ {(x,y) | x^2 / 2 + y^2 = 1} 
    → ∃ A B : ℝ × ℝ, A ≠ B ∧ A.1^2 / 2 + A.2^2 = 1 ∧ B.1^2 / 2 + B.2^2 = 1 ∧
    let d := 2 / sqrt (1 + k^2) in 
    let AB := sqrt (1 + k^2) * sqrt (((A.1 + B.1)^2 - 4 * A.1 * B.1) / (1 + 2 * k^2)) in
    let S := 1 / 2 * AB * d in 
    S = sqrt 2 / 2)

#check ellipse
#check max_area_line

theorem problem_equivalence : ellipse ∧ max_area_line :=
  sorry

end problem_equivalence_l600_600161


namespace find_time_to_fill_tank_l600_600439

noncomputable def time_to_fill_tanker (TA : ℝ) : Prop :=
  let RB := 1 / 40
  let fill_time := 29.999999999999993
  let half_fill_time := fill_time / 2
  let RAB := (1 / TA) + RB
  (RAB * half_fill_time = 1 / 2) → (TA = 120)

theorem find_time_to_fill_tank : ∃ TA, time_to_fill_tanker TA :=
by
  use 120
  sorry

end find_time_to_fill_tank_l600_600439


namespace gym_hours_tuesday_equals_friday_l600_600242

-- Definitions
def weekly_gym_hours : ℝ := 5
def monday_hours : ℝ := 1.5
def wednesday_hours : ℝ := 1.5
def friday_hours : ℝ := 1
def total_weekly_hours : ℝ := weekly_gym_hours - (monday_hours + wednesday_hours + friday_hours)

-- Theorem statement
theorem gym_hours_tuesday_equals_friday : 
  total_weekly_hours = friday_hours :=
by
  sorry

end gym_hours_tuesday_equals_friday_l600_600242


namespace polygon_inequality_l600_600003

theorem polygon_inequality
  (n : ℕ) (h : 3 ≤ n)
  (A : fin n → ℝ × ℝ)
  (O : ℝ × ℝ)
  (R : ℝ)
  (r : fin (n - 2) → ℝ)
  (inscribed : ∀ i : fin n, dist O (A i) = R)
  (inside : ∀ i : fin n, is_inside_polygon O (A i))
  (convex : is_convex_polygon A)
  (inradii : ∀ (i : fin (n - 2)), r i = inradius (triangle (A 0) (A i.cast_succ) (A (i.succ.cast_succ)) (A (i.succ.succ.cast_succ)))) :
  (finset.univ.sum r) ≤ R * ((n : ℝ) * (Real.cos (Real.pi / (n : ℝ))) - (n : ℝ) + 2) :=
by sorry

end polygon_inequality_l600_600003


namespace special_operation_value_l600_600354

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l600_600354


namespace walter_equal_share_l600_600249

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end walter_equal_share_l600_600249


namespace range_of_a_l600_600331

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x ≤ y → f a x ≤ f a y) ↔ (0 ≤ a ∧ a ≤ 1/3) 
where
  f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a - 1) * x - 3
:= sorry

end range_of_a_l600_600331


namespace radius_of_circle_l600_600747

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * real.pi * r) = real.pi * r^2) : r = 6 :=
by {
    sorry
}

end radius_of_circle_l600_600747


namespace range_of_a_l600_600109

-- Define a statement asserting the appropriate inequality
noncomputable def inequality_holds_for_all_x (a : ℝ) : Prop :=
  ∀ (x : ℝ), (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0

-- The main statement to be proved
theorem range_of_a (a : ℝ) : inequality_holds_for_all_x a ↔ a ∈ Icc (-2 : ℝ) 2 := sorry

end range_of_a_l600_600109


namespace sandwiches_left_l600_600285

theorem sandwiches_left 
    (initial_sandwiches : ℕ)
    (first_coworker : ℕ)
    (second_coworker : ℕ)
    (third_coworker : ℕ)
    (kept_sandwiches : ℕ) :
    initial_sandwiches = 50 →
    first_coworker = 4 →
    second_coworker = 3 →
    third_coworker = 2 * first_coworker →
    kept_sandwiches = 3 * second_coworker →
    initial_sandwiches - (first_coworker + second_coworker + third_coworker + kept_sandwiches) = 26 :=
by
  intros h_initial h_first h_second h_third h_kept
  rw [h_initial, h_first, h_second, h_third, h_kept]
  simp
  norm_num
  sorry

end sandwiches_left_l600_600285


namespace not_perfect_square_for_n_greater_than_11_l600_600296

theorem not_perfect_square_for_n_greater_than_11 (n : ℤ) (h1 : n > 11) :
  ∀ m : ℤ, n^2 - 19 * n + 89 ≠ m^2 :=
sorry

end not_perfect_square_for_n_greater_than_11_l600_600296


namespace sum_first_5n_l600_600661

theorem sum_first_5n (n : ℕ) (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210) : 
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_l600_600661


namespace sum_of_2x2_table_is_zero_l600_600227

theorem sum_of_2x2_table_is_zero {a b c d : ℤ} 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_eq : a + b = c + d)
  (prod_eq : a * c = b * d) :
  a + b + c + d = 0 :=
by sorry

end sum_of_2x2_table_is_zero_l600_600227


namespace equivalent_statements_l600_600915
  
variables {A B : Prop}

theorem equivalent_statements :
  ((A ∧ B) → ¬ (A ∨ B)) ↔ ((A ∨ B) → ¬ (A ∧ B)) :=
sorry

end equivalent_statements_l600_600915


namespace find_distance_between_centers_l600_600674

variable {AB BD AD radius distance : ℝ}
namespace geometry_theorem

def parallelogram (A B C D : Type) : Prop :=
  -- The definition of a parallelogram can optionally be expanded 
  -- to include that opposite sides are equal and parallel if needed
  
def inscribed_circle_radius (S : ℝ) (p : ℝ) : ℝ :=
  S / p

def distance_between_centers (radius : ℝ) (diagonal : ℝ) (semi_perimeter_diff : ℝ) : ℝ :=
  sqrt (diagonal^2 + (2 * radius)^2)

theorem find_distance_between_centers 
 (h_parallelogram : parallelogram A B C D)
 (h_sides : AB = 2 ∧ BD = 3 ∧ AD = 4)
 (h_area : ∀ p : ℝ, let S := sqrt (p * (p - AB) * (p - BD) * (p - AD)) 
                    in S / p = (sqrt (135 / 16)) / (9 / 2))
 (h_radius : radius = sqrt(15) / 6)
 (h_semi_perimeter_diff : ∀ p : ℝ, (p - AD) = 1 / 2)
 (h_diagonal : BD = 3) :
  distance_between_centers radius BD (1 / 2) = sqrt(51) / 3 := 
sorry

end geometry_theorem

end find_distance_between_centers_l600_600674


namespace count_valid_k_l600_600091

theorem count_valid_k (n : ℕ) (h : n = 454500) :
  {k : ℕ | k ≤ n ∧ 505 ∣ (k * (k - 1))}.to_finset.card = 3600 :=
by
  sorry

end count_valid_k_l600_600091


namespace probability_calculation_l600_600875

/-
Define the events and conditions
-/
def event (x y z : ℕ) : Prop := (z > x ∧ z > y)

noncomputable def probability_event : ℚ :=
  let total_possible_outcomes := 8 * 8 * 8
  let favorable_outcomes :=
    (∑ x in Finset.range 8, ∑ y in Finset.range 8,
      (8 - (max x y + 1))) -- summing over all (x, y) pairs
  favorable_outcomes % total_possible_outcomes

theorem probability_calculation : probability_event = 7 / 64 := sorry

end probability_calculation_l600_600875


namespace usual_time_to_office_l600_600404

theorem usual_time_to_office
  (S T : ℝ) 
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = (4 / 5) * S * (T + 10)):
  T = 40 := 
by
  sorry

end usual_time_to_office_l600_600404


namespace sine_angle_between_vectors_l600_600729

theorem sine_angle_between_vectors (D A C D' B M N : ℝ × ℝ × ℝ)
  (hD : D = (0, 0, 0))
  (hA : A = (1, 0, 0))
  (hC : C = (0, 1, 0))
  (hD' : D' = (0, 0, 1))
  (hB : B = (1, 1, 0))
  (hM : M = (1/2, 1/2, 0))
  (hN : N=(1/2, 0, 1/2)) :
  let BD' := (D'.1 - B.1, D'.2 - B.2, D'.3 - B.3)
  let MN := (N.1 - M.1, N.2 - M.2, N.3 - M.3) in
  (Real.sin (Real.acos ((BD'.1 * MN.1 + BD'.2 * MN.2 + BD'.3 * MN.3) 
    / (Real.sqrt (BD'.1 ^ 2 + BD'.2 ^ 2 + BD'.3 ^ 2) 
    * Real.sqrt (MN.1 ^ 2 + MN.2 ^ 2 + MN.3 ^ 2))))) = (Real.sqrt 3 / 3) :=
by
  sorry

end sine_angle_between_vectors_l600_600729


namespace total_shaded_area_l600_600456

theorem total_shaded_area (S T : ℝ) (hS : 12 / S = 4) (hT : S / T = 4) : 
  let small_square_area := (T ^ 2)
  let total_small_squares_area := 8 * small_square_area
  let large_square_area := (S ^ 2)
  let total_shaded_area := total_small_squares_area + large_square_area
  total_shaded_area = 13.5 := 
by
  have h1 : S = 3 :=
    by linarith [hS]
  have h2 : T = 3 / 4 :=
    by linarith [hT, h1]
  have small_square_area_calc : small_square_area = (3 / 4) ^ 2 := 
    by rw [h2]
  have total_small_squares_area_calc : total_small_squares_area = 8 * (9 / 16) := 
    by rw [small_square_area_calc]; linarith
  have large_square_area_calc : large_square_area = 9 := 
    by rw [h1]; linarith
  have total_shaded_area_calc : total_shaded_area = (8 * (9 / 16)) + 9 := 
    by rw [total_small_squares_area_calc, large_square_area_calc]; linarith
  have total_shaded_area_result : total_shaded_area = 13.5 := 
    by rw [total_shaded_area_calc]; linarith
  exact total_shaded_area_result

end total_shaded_area_l600_600456


namespace six_digit_number_divisible_by_37_l600_600382

theorem six_digit_number_divisible_by_37 (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) (h3 : 37 ∣ (a + b)) : 37 ∣ (1000 * a + b) :=
sorry

end six_digit_number_divisible_by_37_l600_600382


namespace find_f_of_minus_one_half_l600_600985

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = - f x
axiom f_def_on_interval : ∀ x, 0 ≤ x ∧ x < 1 → f x = x

theorem find_f_of_minus_one_half : f (-1 / 2) = - 1 / 2 :=
by {
  have h0 : 0 ≤ 1 / 2 ∧ 1 / 2 < 1 := by norm_num,
  have h1 : f (1 / 2) = 1 / 2 := f_def_on_interval 1 / 2 h0,
  have h2 : f (-1 / 2) = - f (1 / 2) := odd_function f (1 / 2),
  rw h1 at h2,
  exact h2,
}

end find_f_of_minus_one_half_l600_600985


namespace simplify_expression_l600_600489

-- Define the expression to be simplified
def expression : ℝ := (sqrt 27 / (sqrt 3 / 2)) * (2 * sqrt 2) - 6 * sqrt 2

-- State the theorem to be proven
theorem simplify_expression : expression = 6 * sqrt 2 :=
by
  sorry

end simplify_expression_l600_600489


namespace find_j_l600_600033

def original_number (a b k : ℕ) : ℕ := 10 * a + b
def sum_of_digits (a b : ℕ) : ℕ := a + b
def modified_number (b a : ℕ) : ℕ := 20 * b + a

theorem find_j
  (a b k j : ℕ)
  (h1 : original_number a b k = k * sum_of_digits a b)
  (h2 : modified_number b a = j * sum_of_digits a b) :
  j = (199 + k) / 10 :=
sorry

end find_j_l600_600033


namespace completing_square_l600_600809

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l600_600809


namespace ab_operation_l600_600341

theorem ab_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 15) (h_mul : a * b = 36) : 
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := 
by 
  sorry

end ab_operation_l600_600341


namespace measure_of_angle_l600_600324

theorem measure_of_angle (x : ℝ) (h : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l600_600324


namespace number_of_n_l600_600093

-- Definition of a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- The main statement to prove
theorem number_of_n (n : ℕ) :
  (∃ k : ℕ, k ∈ finset.Icc 1 2000 ∧ (21 * k) ∈ (finset.range 2001).filter is_perfect_square) = 9 :=
sorry

end number_of_n_l600_600093


namespace complex_quadrant_l600_600683

-- Define the complex number and its square
def z : ℂ := (2 - complex.I)^2

-- Prove the point corresponding to the complex number z is located in the fourth quadrant
theorem complex_quadrant : ∃ x y : ℝ, z = complex.mk x y ∧ x > 0 ∧ y < 0 := 
  sorry

end complex_quadrant_l600_600683


namespace C_translation_l600_600993

noncomputable theory

def A_initial := (-1, 2)
def B_initial := (1, -1)
def C_initial := (2, 1)

def A_translated := (-3, a : ℤ)
def B_translated := (b : ℤ, 3)

def translate_point (P : ℤ × ℤ) (dx dy : ℤ) : ℤ × ℤ :=
  (P.1 + dx, P.2 + dy)

theorem C_translation (a b : ℤ) :
  let dx := (A_translated.1 - A_initial.1)
  let dy := (B_translated.2 - B_initial.2)
  translate_point C_initial dx dy = (0, 5) :=
begin
  sorry
end

end C_translation_l600_600993


namespace tan_of_angle_through_point_l600_600168

theorem tan_of_angle_through_point (α : ℝ) (hα : ∃ x y : ℝ, (x = 1) ∧ (y = 2) ∧ (y/x = (Real.sin α) / (Real.cos α))) :
  Real.tan α = 2 :=
sorry

end tan_of_angle_through_point_l600_600168


namespace quadrilateral_inequality_l600_600691

open EuclideanGeometry

noncomputable theory
open_locale big_operators

variables {A B C D P K : Point}
variables (h1 : ConvexQuadrilateral A B C D)
variables (h2 : AB ≈ CD)
variables (h3 : ∠ P B A + ∠ P C D = 180)

theorem quadrilateral_inequality (h1 : ConvexQuadrilateral A B C D) (h2 : AB ≈ CD)
  (h3 : ∠ P B A + ∠ P C D = 180) : PB + PC < AD :=
sorry

end quadrilateral_inequality_l600_600691


namespace angle_YZX_eq_120_l600_600911

-- Definitions based on the given conditions
variable (A B C X Y Z : Point)
variable (ΔABC : Triangle)
variable (Γ : Circle)
variable (angle : Point → Point → Point → ℝ)

-- Conditions
axiom (incircle_ABC : isInCircle Γ ΔABC)
axiom (circumcircle_XYZ : isCircumCircle Γ (Triangle.mk X Y Z))
axiom (on_BC : onLine X (Line.mk B C))
axiom (on_AB : onLine Y (Line.mk A B))
axiom (on_AC : onLine Z (Line.mk A C))
axiom (angle_A : angle B A C = 60)
axiom (angle_B : angle A B C = 80)
axiom (angle_C : angle A C B = 40)

-- The question to prove
theorem angle_YZX_eq_120 :
  angle Y Z X = 120 := sorry

end angle_YZX_eq_120_l600_600911


namespace retailer_profit_calculation_l600_600841

theorem retailer_profit_calculation : 
  ∀ (purchase_price overhead_expenses selling_price : ℕ), 
    purchase_price = 225 → 
    overhead_expenses = 30 → 
    selling_price = 300 → 
    ((selling_price - (purchase_price + overhead_expenses) : ℚ) / (purchase_price + overhead_expenses) * 100).round 2 = 17.65 :=
by
  intros purchase_price overhead_expenses selling_price hp ho hs
  -- The actual proof is omitted; show the statement we aim to prove:
  sorry

end retailer_profit_calculation_l600_600841


namespace tank_capacity_l600_600873

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end tank_capacity_l600_600873


namespace kyuhyung_has_19_candies_l600_600074

-- Definitions for conditions
def kyuhyungCandies : ℕ
def dongminCandies : ℕ := kyuhyungCandies + 5
def totalCandies : ℕ := kyuhyungCandies + dongminCandies

-- The theorem to prove
theorem kyuhyung_has_19_candies (h1 : dongminCandies = kyuhyungCandies + 5)
                               (h2 : totalCandies = 43):
  kyuhyungCandies = 19 :=
by
  sorry

end kyuhyung_has_19_candies_l600_600074


namespace staffing_ways_l600_600909

-- Define the problem conditions
def total_resumes : ℕ := 30
def rejected_resumes : ℕ := 12
def remaining_candidates : ℕ := total_resumes - rejected_resumes -- 18
def skilled_in_wm : ℕ := 2
def skilled_in_rs : ℕ := 2
def skilled_in_ft : ℕ := 1
def skilled_in_sa : ℕ := 1
def skilled_candidates : ℕ := skilled_in_wm + skilled_in_rs + skilled_in_ft + skilled_in_sa -- 6
def versatile_candidates : ℕ := remaining_candidates - skilled_candidates -- 12

-- Define the number of ways to assign positions based on the conditions
def ways_to_assign_wm : ℕ := skilled_in_wm
def ways_to_assign_rs : ℕ := skilled_in_rs
def ways_to_assign_ft : ℕ := skilled_in_ft
def ways_to_assign_sa : ℕ := skilled_in_sa

-- Define the total number of ways to staff the battle station
def total_ways_to_staff : ℕ := 
  ways_to_assign_wm * 
  ways_to_assign_rs * 
  ways_to_assign_ft * 
  ways_to_assign_sa * 
  versatile_candidates * 
  (versatile_candidates - 1) -- 528

-- Define the theorem to be proved
theorem staffing_ways : total_ways_to_staff = 528 :=
by
  unfold total_resumes rejected_resumes remaining_candidates
         skilled_in_wm skilled_in_rs skilled_in_ft skilled_in_sa
         skilled_candidates versatile_candidates
         ways_to_assign_wm ways_to_assign_rs ways_to_assign_ft ways_to_assign_sa
         total_ways_to_staff
  dsimp
  ring

end staffing_ways_l600_600909


namespace three_numbers_lcm_ratio_l600_600377

theorem three_numbers_lcm_ratio
  (x : ℕ)
  (h1 : 3 * x.gcd 4 = 1)
  (h2 : (3 * x * 4 * x) / x.gcd (3 * x) = 180)
  (h3 : ∃ y : ℕ, y = 5 * (3 * x))
  : (3 * x = 45 ∧ 4 * x = 60 ∧ 5 * (3 * x) = 225) ∧
      lcm (lcm (3 * x) (4 * x)) (5 * (3 * x)) = 900 :=
by
  sorry

end three_numbers_lcm_ratio_l600_600377


namespace circle_radius_tangent_to_equilateral_triangle_and_axes_l600_600041

theorem circle_radius_tangent_to_equilateral_triangle_and_axes :
  ∀ (A B O : EuclideanGeometry.Point ℝ ℝ)
  (r : ℝ), 
  EuclideanGeometry.is_equilateral_triangle A B (EuclideanGeometry.Point.mk 1 (sqrt 3)) 2 →
  EuclideanGeometry.is_tangent A B r O →
  EuclideanGeometry.is_tangent_to_axes O r →
  r = 1 :=
by
  intros A B O r
  assume is_eq_triangle is_tangent_to_AB is_tangent_to_axes 
  sorry

end circle_radius_tangent_to_equilateral_triangle_and_axes_l600_600041


namespace cos_pi_over_2_plus_2theta_l600_600954

theorem cos_pi_over_2_plus_2theta (θ : ℝ) (hcos : Real.cos θ = 1 / 3) (hθ : 0 < θ ∧ θ < Real.pi) :
    Real.cos (Real.pi / 2 + 2 * θ) = - (4 * Real.sqrt 2) / 9 := 
sorry

end cos_pi_over_2_plus_2theta_l600_600954


namespace sqrt_expr_simplification_l600_600509

theorem sqrt_expr_simplification :
  (real.sqrt 27) / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - (6 * real.sqrt 2) = (6 * real.sqrt 2) :=
by
  sorry

end sqrt_expr_simplification_l600_600509


namespace minimum_value_of_expression_l600_600651

theorem minimum_value_of_expression {x y : ℝ} (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ m : ℝ, m = 0.75 ∧ ∀ z : ℝ, (∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y = 1 ∧ z = 2 * x + 3 * y ^ 2) → z ≥ m :=
sorry

end minimum_value_of_expression_l600_600651


namespace complete_the_square_l600_600827

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l600_600827


namespace cows_difference_l600_600673

/--
In a field, there are 600 cows consisting of males, females, and transgender cows. The ratio of males to females to transgender cows is 5:3:2.
Among the males, 50% have horns, 40% are spotted, and 20% are brown.
Among the females, 35% are spotted, 25% have horns, and 60% are white.
Among the transgender cows, 45% have a unique pattern, 30% have both spots and horns, and 50% are black.
-/
def total_cows : ℕ := 600
def ratio_males : ℕ := 5
def ratio_females : ℕ := 3
def ratio_transgender : ℕ := 2
def percent_horned_males : ℕ := 50
def percent_spotted_males : ℕ := 40
def percent_brown_males : ℕ := 20
def percent_spotted_females : ℕ := 35
def percent_horned_females : ℕ := 25
def percent_white_females : ℕ := 60
def percent_unique_pattern_transgender : ℕ := 45
def percent_spots_horns_transgender : ℕ := 30
def percent_black_transgender : ℕ := 50

theorem cows_difference : 
  let total_ratio := ratio_males + ratio_females + ratio_transgender in
  let males := total_cows * ratio_males / total_ratio in
  let females := total_cows * ratio_females / total_ratio in
  let transgender := total_cows * ratio_transgender / total_ratio in
  let spotted_females := percent_spotted_females * females / 100 in
  let horned_males := percent_horned_males * males / 100 in
  let brown_males := percent_brown_males * males / 100 in
  let unique_pattern_transgender := percent_unique_pattern_transgender * transgender / 100 in
  let white_horned_females := percent_horned_females * percent_white_females * females / 10000 in
  spotted_females - (horned_males + brown_males + unique_pattern_transgender + white_horned_females) = -291 :=
by sorry

end cows_difference_l600_600673


namespace weighted_average_is_92_l600_600837

variable (quiz_score : ℕ) (quiz_weight : ℕ)
variable (midterm_score : ℕ) (midterm_weight : ℕ)
variable (final_score : ℕ) (final_weight : ℕ)
variable (weighted_avg : ℕ)

-- Define the scores and weights
def quiz_score : ℕ := 89
def quiz_weight : ℕ := 3
def midterm_score : ℕ := 91
def midterm_weight : ℕ := 3
def final_score : ℕ := 95
def final_weight : ℕ := 4

-- Sum of the scores times their respective weights
def sum_weighted_scores : ℕ :=
  (quiz_score * quiz_weight) + (midterm_score * midterm_weight) + (final_score * final_weight)

-- Sum of the weights
def sum_weights : ℕ :=
  quiz_weight + midterm_weight + final_weight

-- Calculate the weighted average
def weighted_avg : ℕ :=
  sum_weighted_scores / sum_weights

-- The goal is to prove the weighted average is 92
theorem weighted_average_is_92 : weighted_avg = 92 :=
  by
  -- The proof is not required. Placeholder for now.
  sorry

end weighted_average_is_92_l600_600837


namespace completing_the_square_l600_600815

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l600_600815


namespace largest_integer_2016_power_divide_factorial_l600_600059

theorem largest_integer_2016_power_divide_factorial : 
  (∃ m : ℕ, 2016^m ∣ (fact 2016) ∧ ∀ n : ℕ, (2016^n ∣ (fact 2016)) → n ≤ 334) :=
begin
  sorry
end

end largest_integer_2016_power_divide_factorial_l600_600059


namespace find_some_number_eq_0_3_l600_600104

theorem find_some_number_eq_0_3 (X : ℝ) (h : 2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002) :
  X = 0.3 :=
by sorry

end find_some_number_eq_0_3_l600_600104


namespace average_price_of_racket_l600_600455

theorem average_price_of_racket
  (total_amount_made : ℝ)
  (number_of_pairs_sold : ℕ)
  (h1 : total_amount_made = 490) 
  (h2 : number_of_pairs_sold = 50) : 
  (total_amount_made / number_of_pairs_sold : ℝ) = 9.80 := 
  by
  sorry

end average_price_of_racket_l600_600455


namespace derivative_at_pi_div2_l600_600997

open Real

def f (x : ℝ) := 3 * sin x - 4 * cos x

theorem derivative_at_pi_div2 : (f' (π / 2)) = 4 :=
by
  sorry

end derivative_at_pi_div2_l600_600997


namespace train_passes_man_in_4_4_seconds_l600_600460

noncomputable def train_speed_kmph : ℝ := 84
noncomputable def man_speed_kmph : ℝ := 6
noncomputable def train_length_m : ℝ := 110

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def man_speed_mps : ℝ :=
  kmph_to_mps man_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps + man_speed_mps

noncomputable def passing_time : ℝ :=
  train_length_m / relative_speed_mps

theorem train_passes_man_in_4_4_seconds :
  passing_time = 4.4 :=
by
  sorry -- Proof not required, skipping the proof logic

end train_passes_man_in_4_4_seconds_l600_600460


namespace total_time_for_journey_l600_600034

theorem total_time_for_journey (d1 d2 : ℝ) (s1 s2 : ℝ) (T : ℝ) 
  (h_total_distance : d1 + d2 = 336) 
  (h_first_half_speed : s1 = 21) 
  (h_second_half_speed : s2 = 24) 
  (h_first_half_distance : d1 = 168)
  (h_second_half_distance : d2 = 168) :
  T = (d1 / s1) + (d2 / s2) →
  T = 15 :=
by
  intros h_T
  have t1: d1 / s1 = 8 := by sorry
  have t2: d2 / s2 = 7 := by sorry
  rw [t1, t2, add_comm (d2 / s2) (d1 / s1)] at h_T
  exact h_T

end total_time_for_journey_l600_600034


namespace math_problem_l600_600618

theorem math_problem
  (a b c : ℚ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c))
    / (2 / (a * b) - 2 * a * b / c)
    / (101 / c)
  ) = -1 / 202 := 
sorry

end math_problem_l600_600618


namespace num_valid_sequences_l600_600272

theorem num_valid_sequences : 
  let a := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (∀ (l : List ℕ), l.perm a → 
   (∀ i, 1 < i ∧ i ≤ 12 → (l ! (i - 1) + 1) ∈ l.take (i - 1) ∧ 
                        (l ! (i - 1) - 1) ∈ l.take (i - 1))) →
  (l : List ℕ) → l.length = 12 →
  l.perm a →
  (∀ i, 2 ≤ i ∧ i ≤ 12 → (l ! (i - 1) + 1) ∈ l.take (i - 1) ∧ 
                        (l ! (i - 1) - 1) ∈ l.take (i - 1)) →
  l.length = 12 →
  l.perm a →
  512 = 512 :=
by
  sorry

end num_valid_sequences_l600_600272


namespace rectangle_covered_in_layers_l600_600437

-- Given definitions and a statement about the grid figure properties
def grid_figure_property (m n : ℕ) (F : matrix (fin m) (fin n) ℕ) :=
  ∀ (A : matrix (fin m) (fin n) ℝ), (0 < matrix.sum A) →
    ∃ (x y : fin m) (r : fin 4), 0 < matrix.sum (matrix.submatrix A (λ i, (i + x) % m) (λ j, (j + y) % n))

-- Main statement: Proving that the rectangle can be covered by figure F in several layers
theorem rectangle_covered_in_layers {m n : ℕ} (F : matrix (fin m) (fin n) ℕ) :
  grid_figure_property m n F →
  ∃ (d : matrix (fin m) (fin n) ℚ), ∀ i j, 1 = matrix.sum (matrix.entrywise (λ t, t * d i j) F) :=
sorry

end rectangle_covered_in_layers_l600_600437


namespace arcsin_one_eq_pi_div_two_l600_600519

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := by
  sorry

end arcsin_one_eq_pi_div_two_l600_600519


namespace calculate_expression_l600_600498

theorem calculate_expression :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2 = 6 * real.sqrt 2) :=
by
  sorry

end calculate_expression_l600_600498


namespace cos_of_theta_l600_600461

theorem cos_of_theta
  (A : ℝ) (a : ℝ) (m : ℝ) (θ : ℝ) 
  (hA : A = 40) 
  (ha : a = 12) 
  (hm : m = 10) 
  (h_area: A = (1/2) * a * m * Real.sin θ) 
  : Real.cos θ = (Real.sqrt 5) / 3 :=
by
  sorry

end cos_of_theta_l600_600461


namespace g_value_l600_600630

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 else g x

axiom f_odd : ∀ x : ℝ, f (-x) = - (f x)

theorem g_value : g 2 = 4 :=
by {
  sorry
}

end g_value_l600_600630


namespace problem1_problem2_problem3_problem4_l600_600723

-- Problem 1
theorem problem1 (x : ℝ) (h : x * (5 * x + 4) = 5 * x + 4) : x = -4 / 5 ∨ x = 1 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : -3 * x^2 + 22 * x - 24 = 0) : x = 6 ∨ x = 4 / 3 := 
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : (x + 8) * (x + 1) = -12) : x = -4 ∨ x = -5 := 
sorry

-- Problem 4
theorem problem4 (x : ℝ) (h : (3 * x + 2) * (x + 3) = x + 14) : x = -4 ∨ x = 2 / 3 := 
sorry

end problem1_problem2_problem3_problem4_l600_600723


namespace sum_of_circle_center_coordinates_l600_600741

open Real

theorem sum_of_circle_center_coordinates :
  let x1 := 5
  let y1 := 3
  let x2 := -7
  let y2 := 9
  let x_m := (x1 + x2) / 2
  let y_m := (y1 + y2) / 2
  x_m + y_m = 5 := by
  sorry

end sum_of_circle_center_coordinates_l600_600741


namespace smallest_positive_period_monotonic_intervals_l600_600632

noncomputable def f (x : ℝ) : ℝ :=
  5 * sin x * cos x - 5 * sqrt 3 * (cos x)^2 + (5 / 2) * sqrt 3

theorem smallest_positive_period :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π :=
sorry

theorem monotonic_intervals :
  (∀ k : ℤ, ∀ x, (k * π - π / 12 <= x ∧ x <= k * π + 5 * π / 12) → monotone (f ∘ λ x, x)) ∧
  (∀ k : ℤ, ∀ x, (k * π + 5 * π / 12 <= x ∧ x <= k * π + 11 * π / 12) → antitone (f ∘ λ x, x)) :=
sorry

end smallest_positive_period_monotonic_intervals_l600_600632


namespace kayla_waiting_years_l600_600644

def minimum_driving_age : ℕ := 18
def kimiko_age : ℕ := 26
def kayla_age : ℕ := kimiko_age / 2
def years_until_kayla_can_drive : ℕ := minimum_driving_age - kayla_age

theorem kayla_waiting_years : years_until_kayla_can_drive = 5 :=
by
  sorry

end kayla_waiting_years_l600_600644


namespace completing_the_square_l600_600816

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l600_600816


namespace completing_the_square_l600_600801

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l600_600801


namespace number_of_possible_values_of_r_eq_894_l600_600337

noncomputable def r_possible_values : ℕ :=
  let lower_bound := 0.3125
  let upper_bound := 0.4018
  let min_r := 3125  -- equivalent to the lowest four-digit decimal ≥ 0.3125
  let max_r := 4018  -- equivalent to the highest four-digit decimal ≤ 0.4018
  1 + max_r - min_r  -- total number of possible values

theorem number_of_possible_values_of_r_eq_894 :
  r_possible_values = 894 :=
by
  sorry

end number_of_possible_values_of_r_eq_894_l600_600337


namespace arcsin_one_eq_pi_div_two_l600_600520

noncomputable def arcsin : ℝ → ℝ := sorry -- Define arcsin function

theorem arcsin_one_eq_pi_div_two : arcsin 1 = Real.pi / 2 := sorry

end arcsin_one_eq_pi_div_two_l600_600520


namespace find_prime_base14_alternate_l600_600937

def base14_alternate_form (A : ℕ) (n : ℕ) : Prop :=
  A = (14^0 + 14^2 + 14^4 + ... + 14^(2*n))

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_prime_base14_alternate :
  ∃ p, (∃ n, base14_alternate_form p n) ∧ is_prime p ∧ p = 197 := by
sorry

end find_prime_base14_alternate_l600_600937


namespace fraction_pairs_l600_600951

theorem fraction_pairs (n : ℕ) (h : n > 2009) : 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 1 ≤ a ∧ a ≤ n ∧
  1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧
  1/a + 1/b = 1/c + 1/d := 
sorry

end fraction_pairs_l600_600951


namespace cannot_form_1x1x2_blocks_l600_600868

theorem cannot_form_1x1x2_blocks :
  let edge_length := 7
  let total_cubes := edge_length * edge_length * edge_length
  let central_cube := (3, 3, 3)
  let remaining_cubes := total_cubes - 1
  let checkerboard_color (x y z : Nat) : Bool := (x + y + z) % 2 = 0
  let num_white (k : Nat) := if k % 2 = 0 then 25 else 24
  let num_black (k : Nat) := if k % 2 = 0 then 24 else 25
  let total_white := 170
  let total_black := 171
  total_black > total_white →
  ¬(remaining_cubes % 2 = 0 ∧ total_white % 2 = 0 ∧ total_black % 2 = 0) → 
  ∀ (block: Nat × Nat × Nat → Bool) (x y z : Nat), block (x, y, z) = ((x*y*z) % 2 = 0) := sorry

end cannot_form_1x1x2_blocks_l600_600868


namespace completing_the_square_l600_600812

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l600_600812


namespace distance_from_site_l600_600011

-- Definitions based on the conditions
def speed_of_sound : ℝ := 330 / 1  -- Speed of sound in m/s

def time_difference : ℝ := 12  -- Time difference in seconds

def expected_distance : ℝ := 3960  -- Expected distance when heard the second blast in meters

-- The statement we want to prove
theorem distance_from_site : speed_of_sound * time_difference = expected_distance := by
  sorry

end distance_from_site_l600_600011


namespace jake_work_hours_l600_600240

-- Definitions for the conditions
def initial_debt : ℝ := 100
def amount_paid : ℝ := 40
def work_rate : ℝ := 15

-- The main theorem stating the number of hours Jake needs to work
theorem jake_work_hours : ∃ h : ℝ, initial_debt - amount_paid = h * work_rate ∧ h = 4 :=
by 
  -- sorry placeholder indicating the proof is not required
  sorry

end jake_work_hours_l600_600240


namespace radius_of_circle_l600_600749

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * real.pi * r) = real.pi * r^2) : r = 6 :=
by {
    sorry
}

end radius_of_circle_l600_600749


namespace nth_inequality_l600_600963

theorem nth_inequality (n : ℕ) (x : ℝ) (h : x > 0) : x + n^n / x^n ≥ n + 1 := 
sorry

end nth_inequality_l600_600963


namespace radius_of_circle_l600_600746

theorem radius_of_circle (r : ℝ) : 3 * 2 * Real.pi * r = Real.pi * r^2 → r = 6 :=
by {
  intro h,
  have h1 : 6 * Real.pi * r = Real.pi * r^2 := by rw [←mul_assoc, ←h],
  have h2 : 6 * r = r^2 := by rw [←mul_div_cancel_left 'Real.pi, h1],
  have h3 : r^2 - 6 * r = 0 := by ring,
  have h4 : r * (r - 6) = 0 := by rw h3,
  cases eq_zero_or_eq_zero_of_mul_eq_zero h4 with h5 h6,
  { exact h5, },
  { exact h6, }
} sorry

end radius_of_circle_l600_600746


namespace total_bill_before_tip_l600_600765

-- Declare the initial conditions
variables (share_amount : ℝ) (num_people : ℕ) (tip_percent : ℝ)
-- Define the values based on the problem statement
def share_amount := 50.97
def num_people := 3
def tip_percent := 0.10

-- Define the equation representing the total paid after the tip
def total_paid_after_tip := share_amount * num_people

-- State the theorem and the proof goal
theorem total_bill_before_tip : 
  let total_paid := share_amount * num_people in
  ∀ B : ℝ, 
  (1.10 * B = total_paid) -> 
  B = 139.01 := 
by
  sorry

end total_bill_before_tip_l600_600765


namespace simplify_expression_l600_600485

-- Define the expression to be simplified
def expression : ℝ := (sqrt 27 / (sqrt 3 / 2)) * (2 * sqrt 2) - 6 * sqrt 2

-- State the theorem to be proven
theorem simplify_expression : expression = 6 * sqrt 2 :=
by
  sorry

end simplify_expression_l600_600485


namespace completing_the_square_l600_600802

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l600_600802


namespace minimal_n_l600_600111

theorem minimal_n (k : ℕ) : ∃ (n : ℕ), (∀ (a : ℕ → ℝ) (H₀ : ∀ (i : ℕ), 0 < a (i + 1) - a i ∧ a (i + 1) - a i < a i - a (i - 1)) (H₁ : ∃ (ij : Finset (ℕ × ℕ)), ij.card = k ∧ ∀ (p ∈ ij), a p.1 - a p.2 = 1), n = 2 * k) :=
begin
  sorry
end

end minimal_n_l600_600111


namespace additional_days_l600_600450

def total_refrigerators : ℕ := 1590
def days_produced : ℕ := 12
def prod_rate_first_12_days : ℕ := 80
def increased_prod_rate : ℕ := 90

theorem additional_days
  (total_refrigerators = 1590)
  (days_produced = 12)
  (prod_rate_first_12_days = 80)
  (increased_prod_rate = 90) :
  let produced_first_12_days := prod_rate_first_12_days * days_produced
  let remaining_refrigerators := total_refrigerators - produced_first_12_days
  remaining_refrigerators / increased_prod_rate = 7 :=
by
  sorry

end additional_days_l600_600450


namespace calculate_expression_l600_600495

theorem calculate_expression : (sqrt 27 / (sqrt 3 / 2) * (2 * sqrt 2) - (6 * sqrt 2)) = 6 * sqrt 2 :=
by
  -- Taking these steps from the solution, we should finally arrive at the required proof
  sorry

end calculate_expression_l600_600495


namespace linear_function_quadrants_l600_600986

theorem linear_function_quadrants (m : ℝ) : 
  (∀ x : ℝ, x < 0 → y = m*x - 1  ∈ {y : ℝ | y > 0}) ∧
  (∀ x : ℝ, x < 0 → y = m*x - 1  ∈ {y : ℝ | y < 0}) ∧
  (∀ x : ℝ, x > 0 → y = m*x - 1  ∈ {y : ℝ | y < 0})
  → m < 0 := 
sorry

end linear_function_quadrants_l600_600986


namespace part1_l600_600610

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3 * a - 10 ≤ x ∧ x < 2 * a + 1}
def Q : Set ℝ := {x | |2 * x - 3| ≤ 7}

-- Define the complement of Q in ℝ
def Q_complement : Set ℝ := {x | x < -2 ∨ x > 5}

-- Define the specific value of a
def a : ℝ := 2

-- Define the specific set P when a = 2
def P_a2 : Set ℝ := {x | -4 ≤ x ∧ x < 5}

-- Define the intersection
def intersection : Set ℝ := {x | -4 ≤ x ∧ x < -2}

theorem part1 : P a ∩ Q_complement = intersection := sorry

end part1_l600_600610


namespace range_lambda_l600_600640

theorem range_lambda (a : ℕ → ℝ) (λ : ℝ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, (1 ≤ n) → (a n, 2 * a (n + 1)) ∈ {p : ℝ × ℝ | p.1 - 1/2 * p.2 + 1 = 0})
  (h₃ : ∀ n : ℕ, (1 ≤ n) → (∑ i in finset.range n, (1 / (i + 1 + a i))) ≥ λ) : 
  λ ≤ 1/2 :=
sorry

end range_lambda_l600_600640


namespace completing_the_square_l600_600797

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l600_600797


namespace find_p_l600_600621

variable {p : ℝ} (hp : 0 < p)
variable (x y : ℝ)
variable (h_intersect : ∃ x y, y^2 = 2 * p * x ∧ y = sqrt(3) * x)
variable (F : (ℝ × ℝ)) (A : (ℝ × ℝ)) 
variable (hA : A = (2 * p / 3, 2 * sqrt(3) * p / 3))
variable (hF : F = (p / 2, 0))
variable (h_dist : dist A F = 7)

theorem find_p : p = 6 := by
  sorry

end find_p_l600_600621


namespace completing_square_l600_600807

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l600_600807


namespace equal_area_dividing_line_l600_600107

-- Definitions for centers of disks and line dividing the disks into equal areas
def center_of_square (C1 C2 C3 C4 : Point) : Point :=
  -- function definition to calculate the center of the square
  sorry

def divides_equal_area (C1 C2 C3 C4 C5 : Point) (r : ℝ) (line : Line) : Prop :=
  -- function definition that checks if the given line divides the area equally
  sorry

-- Main statement to prove
theorem equal_area_dividing_line (C1 C2 C3 C4 C5 : Point) (r : ℝ) :
  let O := center_of_square C1 C2 C3 C4 in
  let F := C5 in
  let line := Line_through_points O F in
  divides_equal_area C1 C2 C3 C4 C5 r line :=
sorry

end equal_area_dividing_line_l600_600107


namespace jake_work_hours_l600_600241

-- Definitions for the conditions
def initial_debt : ℝ := 100
def amount_paid : ℝ := 40
def work_rate : ℝ := 15

-- The main theorem stating the number of hours Jake needs to work
theorem jake_work_hours : ∃ h : ℝ, initial_debt - amount_paid = h * work_rate ∧ h = 4 :=
by 
  -- sorry placeholder indicating the proof is not required
  sorry

end jake_work_hours_l600_600241


namespace probability_scrapped_l600_600858

variable (P_A P_B_given_not_A : ℝ)
variable (prob_scrapped : ℝ)

def fail_first_inspection (P_A : ℝ) := 1 - P_A
def fail_second_inspection_given_fails_first (P_B_given_not_A : ℝ) := 1 - P_B_given_not_A

theorem probability_scrapped (h1 : P_A = 0.8) (h2 : P_B_given_not_A = 0.9) (h3 : prob_scrapped = fail_first_inspection P_A * fail_second_inspection_given_fails_first P_B_given_not_A) :
  prob_scrapped = 0.02 := by
  sorry

end probability_scrapped_l600_600858


namespace not_sum_of_squares_of_form_4m_plus_3_l600_600716

theorem not_sum_of_squares_of_form_4m_plus_3 (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ a b : ℤ, n = a^2 + b^2 :=
by
  sorry

end not_sum_of_squares_of_form_4m_plus_3_l600_600716


namespace three_digit_number_congruences_l600_600546

theorem three_digit_number_congruences : 
  (∃y: ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 1945 * y + 243 ≡ 605 [MOD 17] ∧ 
  (∃ n: ℕ, ∀ k ∈ set.Icc 6 58, y = 17 * k + 6) ∧ (58 - 6 + 1 = 53)) := sorry

end three_digit_number_congruences_l600_600546


namespace motorcyclist_speed_equals_30_l600_600007

-- Definitions and conditions from the problem
def hiker_speed := 6 -- hiker's speed in miles per hour
def time_motorcyclist_travel := 12 / 60.0 -- 12 minutes converted to hours
def time_hiker_travel := 48 / 60.0 -- 48 minutes converted to hours
def distance_hiker(time : ℝ) := hiker_speed * time -- distance traveled by the hiker

-- The proof statement to calculate and verify the motorcyclist's speed
theorem motorcyclist_speed_equals_30 
  (M : ℝ)
  (hiker_speed : ℝ := hiker_speed)
  (time_motorcyclist_travel : ℝ := time_motorcyclist_travel)
  (time_hiker_travel : ℝ := time_hiker_travel)
  (distance_hiker : ℝ := distance_hiker time_hiker_travel) :
  M * time_motorcyclist_travel = 6 ↔ M = 30 := 
by
  sorry

end motorcyclist_speed_equals_30_l600_600007


namespace susie_vacuum_time_l600_600317

variable (time_per_room : ℕ) (num_rooms : ℕ)

theorem susie_vacuum_time 
  (h1 : time_per_room = 20)
  (h2 : num_rooms = 6) :
  (time_per_room * num_rooms) / 60 = 2 := 
by
  rw [h1, h2]
  norm_num
  sorry

end susie_vacuum_time_l600_600317


namespace highest_power_of_2_divides_l600_600569

theorem highest_power_of_2_divides (n : ℕ) : ∃ k : ℕ, (2^k ∣ (∏ i in finset.range (2 * n + 1), (if n < i then i else 1))) ∧ (2^k ≤ 2^n) :=
by
  sorry

end highest_power_of_2_divides_l600_600569


namespace coordinates_of_point_Q_l600_600294

-- Defining the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Defining initial point P
def point_P : ℝ × ℝ := (1, 0)

-- Defining the arc length traveled counterclockwise
def arc_length : ℝ := π / 3

-- Main theorem statement
theorem coordinates_of_point_Q
  (x y : ℝ)
  (h : unit_circle x y)
  (hx0 : unit_circle (1 : ℝ) 0)
  (arc : real.angle (x, y) = real.angle (cos (π / 3), sin (π / 3))) :
  (x, y) = (1/2, sqrt 3 / 2) :=
sorry

end coordinates_of_point_Q_l600_600294


namespace sqrt_expression_eq_l600_600506

theorem sqrt_expression_eq :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2) = 6 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_eq_l600_600506


namespace test_question_count_l600_600397

theorem test_question_count :
  ∃ (x y : ℕ), x + y = 30 ∧ 5 * x + 10 * y = 200 ∧ x = 20 :=
by
  sorry

end test_question_count_l600_600397


namespace inequality_solution_l600_600364

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (3 * x - 1) > 1 ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end inequality_solution_l600_600364


namespace how_many_tickets_left_l600_600052

-- Define the conditions
def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define the total tickets won by Tom
def total_tickets : ℕ := tickets_from_whack_a_mole + tickets_from_skee_ball

-- State the theorem to be proved: how many tickets Tom has left
theorem how_many_tickets_left : total_tickets - tickets_spent_on_hat = 50 := by
  sorry

end how_many_tickets_left_l600_600052


namespace solution_to_equation1_solution_to_equation2_l600_600577

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := 3 * x^3 + 4 = -20

-- State the theorems with the correct answers
theorem solution_to_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = -3) :=
by
  sorry

theorem solution_to_equation2 (x : ℝ) : equation2 x ↔ (x = -2) :=
by
  sorry

end solution_to_equation1_solution_to_equation2_l600_600577


namespace booth_visibility_booth_minimum_distance_booth_maximum_distance_l600_600680

def is_visible_under_angle (P F : Point) (angle : ℝ) : Prop := sorry

noncomputable def booth_locus (L : ℝ) : set Point := sorry

theorem booth_visibility (L : ℝ) :
  ∀ P ∈ booth_locus L, is_visible_under_angle P booth 90 :=
sorry

theorem booth_minimum_distance (L : ℝ) (P : Point) (hP : P ∈ booth_locus L) :
  distance P booth = 0 :=
sorry

theorem booth_maximum_distance (L : ℝ) (P : Point) (hP : P ∈ booth_locus L) :
  distance P booth = L / 2 :=
sorry

end booth_visibility_booth_minimum_distance_booth_maximum_distance_l600_600680


namespace prob_exactly_two_same_project_l600_600319

open Nat

theorem prob_exactly_two_same_project : 
  let total_ways := 7^3 in
  let choose_two := choose 3 2 in
  let ways_to_assign_two := 7 * 6 in
  let favorable_ways := choose_two * ways_to_assign_two in
  let probability := favorable_ways / total_ways in
  probability = (18 : ℚ) / 49 :=
by
  let total_ways := 7^3
  let choose_two := choose 3 2
  let ways_to_assign_two := 7 * 6
  let favorable_ways := choose_two * ways_to_assign_two
  let probability := favorable_ways / total_ways

  -- Expected total value checks
  have h_fw : favorable_ways = 18 * 7 := rfl
  have h_tw : total_ways = 7 * 7 * 7 := rfl
  
  -- Calculate probability
  have h_exp : favorable_ways / total_ways = (18 : ℚ) / 49 := by 
    rw [h_fw, h_tw]
    norm_num

  exact h_exp

end prob_exactly_two_same_project_l600_600319


namespace total_area_of_figure_l600_600686

theorem total_area_of_figure :
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  total_area = 89 := by
  -- Definitions
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  -- Proof
  sorry

end total_area_of_figure_l600_600686


namespace right_triangle_not_determined_by_condition_A_l600_600465

-- Definitions of the conditions
def a : ℝ := 1 / 3
def b : ℝ := 1 / 4
def c : ℝ := 1 / 5

-- Condition for right triangle using Pythagorean theorem
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Proof problem statement
theorem right_triangle_not_determined_by_condition_A :
  ¬ is_right_triangle (1 / 3) (1 / 4) (1 / 5) :=
by {
  sorry
}

end right_triangle_not_determined_by_condition_A_l600_600465


namespace part1_part2_l600_600128

noncomputable def circle_center : ℝ × ℝ := (-(√3), 0)

def pointA : ℝ × ℝ := (√3, 0)

def point_on_circle (x y : ℝ) : Prop :=
  (x + √3) ^ 2 + y ^ 2 = 16

def is_on_perpendicular_bisector (A P E : ℝ × ℝ) : Prop :=
  ∥E.1 - A.1∥ + ∥E.2 - A.2∥ = ∥E.1 - P.1∥ + ∥E.2 - P.2∥

def trajectory_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

theorem part1 (P : ℝ × ℝ) (hP : point_on_circle P.1 P.2) (E : ℝ × ℝ)
  (hE : is_on_perpendicular_bisector pointA P E) : trajectory_eq E.1 E.2 :=
sorry

def pointM : ℝ × ℝ := (-2, 0)
def pointN : ℝ × ℝ := (2, 0)

def is_on_line (T : ℝ × ℝ) : Prop :=
  T.1 = 4

def intersect_trajectory (T M N : ℝ × ℝ) (hM : M = pointM) (hN : N = pointN) (hT : is_on_line T) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let line_TM := sorry in -- intersection logic, e.g., line equations
  let line_TN := sorry in -- intersection logic, e.g., line equations
  (line_TM, line_TN)  -- C and D

def line_eq (C D : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - C.2) * (D.1 - C.1) = (D.2 - C.2) * (x - C.1)

theorem part2 (T C D : ℝ × ℝ) (C D : (ℝ × ℝ)) (hT : is_on_line T) :
  ∀ (CD : (ℝ × ℝ) × (ℝ × ℝ)),
    ∃ (fixed_point : ℝ × ℝ), fixed_point = (1, 0) :=
sorry

end part1_part2_l600_600128


namespace angle_equality_of_tangency_l600_600865

-- Define the properties of the triangle ABC with incenter I
variables {A B C I T : ℝ}

-- Assume ∆ABC with the incenter I
-- The circle tangent to AB and BC and also tangent to the circumcircle of ∆ ABC at point T
-- Show that ∠ATI = ∠CTI

theorem angle_equality_of_tangency
  (h_triangle : is_triangle A B C)
  (h_incenter : is_incenter I A B C)
  (h_tangent : is_tangent_circle AB BC A B C T) :
  ∠ATI = ∠CTI :=
sorry

end angle_equality_of_tangency_l600_600865


namespace range_of_quadratic_l600_600616

def function_range {x : ℝ} (h : x ∈ Ioo (-1 : ℝ) 3) : set ℝ :=
{y | ∃ x : ℝ, x ∈ Ioo (-1 : ℝ) 3 ∧ y = (x - 2)^2}

theorem range_of_quadratic (h : x ∈ Ioo (-1 : ℝ) 3) : function_range h = Ico 0 9 := by
  sorry

end range_of_quadratic_l600_600616


namespace total_fruits_l600_600470

theorem total_fruits (Mike_fruits Matt_fruits Mark_fruits : ℕ)
  (Mike_receives : Mike_fruits = 3)
  (Matt_receives : Matt_fruits = 2 * Mike_fruits)
  (Mark_receives : Mark_fruits = Mike_fruits + Matt_fruits) :
  Mike_fruits + Matt_fruits + Mark_fruits = 18 := by
  sorry

end total_fruits_l600_600470


namespace sum_kth_powers_l600_600881

open Nat

/-- Define the set of natural numbers with at most n digits. -/
def D_n (n : ℕ) : set ℕ := { x | x < 10^n }

/-- Define the subset of D_n where the sum of the digits is even. -/
def A_n (n : ℕ) : set ℕ := { x ∈ D_n n | digit_sum x % 2 = 0 }

/-- Define the subset of D_n where the sum of the digits is odd. -/
def B_n (n : ℕ) : set ℕ := { x ∈ D_n n | digit_sum x % 2 = 1 }

/-- Define the digit sum function. -/
def digit_sum (x : ℕ) : ℕ :=
  (List.map digitSum (Nat.digits 10 x)).sum

/-- Prove that if 1 ≤ k < n, then the sum of the k-th powers of all the
numbers in the first group (with an odd sum of digits) equals the sum
of the k-th powers of all the numbers in the second group
(with an even sum of digits). -/
theorem sum_kth_powers (n k : ℕ) (hk1 : 1 ≤ k) (hkn : k < n) :
  (∑ a in A_n n, a ^ k) = (∑ b in B_n n, b ^ k) := sorry

end sum_kth_powers_l600_600881


namespace sum_b_p_equals_59955_l600_600583

noncomputable def b (p : ℕ) : ℕ := Nat.ceil (Real.sqrt ↑p - 1/2)

theorem sum_b_p_equals_59955 : (∑ p in Finset.range 2007, b (p + 1)) = 59955 :=
by
  sorry

end sum_b_p_equals_59955_l600_600583


namespace ellipse_problem_l600_600622

-- Definitions to set up the conditions
def ellipse_center_origin (C : Set (ℝ × ℝ)) : Prop :=
  (0, 0) ∈ C

def line_passes_through_vertices (L : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
      (√3 * x1 - 2 * y1 - 4 * √3 = 0) ∧ (x1, y1) ∈ C ∧ 
      (√3 * x2 - 2 * y2 - 4 * √3 = 0) ∧ (x2, y2) ∈ C

def ellipse_standard_equation (C : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ (x^2 / 16 + y^2 / 12 = 1)

def product_of_slopes_constant (A R M N : ℝ × ℝ) (P Q : Set (ℝ × ℝ)) : Prop :=
  ∃ m1 m2 : ℝ, 
    line M R m1 ∧ line N R m2 ∧ 
    m1 * m2 = -12 / 7

-- The actual proposition combining all parts
theorem ellipse_problem (C L : Set (ℝ × ℝ)) (A R M N : ℝ × ℝ) (P Q : Set (ℝ × ℝ)) 
    (hC_center : ellipse_center_origin C) 
    (hL_vertices : line_passes_through_vertices L C) 
    (hA : A = (-4, 0))
    (hR : R = (3, 0)) :
  ellipse_standard_equation C ∧ product_of_slopes_constant A R M N P Q := 
by
  sorry

end ellipse_problem_l600_600622


namespace find_a_l600_600617

-- Definitions based on conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ set.univ, f (-x) = -f (x)

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a * real.log x - a * x + 1 else -f a (-x)

-- The condition about the minimum value of f(x) when x ∈ (-2, 0) implies that it has a minimum value of 1
def has_min_value (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Ioo (-2 : ℝ) (0 : ℝ), ∃ c ∈ Ioo (-2 : ℝ) (0 : ℝ), f x ≥ 1

-- The problem to prove
theorem find_a (a : ℝ) :
  (is_odd_function (f a)) →
  (∀ x ∈ Ioo 0 2, f a x = a * real.log x - a * x + 1) →
  (has_min_value (f a) a) →
  a = 2 :=
by
  intros h_odd h_function h_min
  -- Proof is omitted
  sorry

end find_a_l600_600617


namespace binom_divisibility_count_l600_600646

theorem binom_divisibility_count : 
  (finset.filter (λ k : ℕ, (nat.choose 2012 k) % 2012 = 0) (finset.range 2013)).card = 1498 :=
by
  sorry

end binom_divisibility_count_l600_600646


namespace completing_the_square_l600_600821

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l600_600821


namespace stool_height_l600_600038

theorem stool_height 
  (light_bulb_below_ceiling : ℕ) 
  (ceiling_above_floor : ℕ) 
  (alice_height : ℕ) 
  (alice_reach_above_head : ℕ) :
  ceiling_above_floor = 300 ∧ light_bulb_below_ceiling = 15 ∧ alice_height = 150 ∧ alice_reach_above_head = 40 →
  ∃ h : ℕ, h = 95 ∧ (alice_height + alice_reach_above_head + h = ceiling_above_floor - light_bulb_below_ceiling) :=
begin
  intros h,
  let total_reach := alice_height + alice_reach_above_head,
  let light_bulb_floor := ceiling_above_floor - light_bulb_below_ceiling,
  use (light_bulb_floor - total_reach),
  split,
  sorry,
  sorry
end

end stool_height_l600_600038


namespace line_intersects_circle_l600_600966

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

noncomputable def outside (x₀ y₀ : ℝ) : Prop := x₀^2 + y₀^2 > 4

noncomputable def line (x₀ y₀ x y : ℝ) : Prop := x₀ * x + y₀ * y = 4

theorem line_intersects_circle (x₀ y₀ : ℝ) (hx₀y₀ : outside x₀ y₀) :
  ∃ x y : ℝ, circle x y ∧ line x₀ y₀ x y :=
begin
  sorry
end

end line_intersects_circle_l600_600966


namespace maximum_n_l600_600361

-- Definition of the increasing sequence of integers starting from a1
def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- The proof problem in Lean 4 statement
theorem maximum_n (a : ℕ → ℕ) (h_inc : sequence a) (h_a1 : a 1 ≥ 3) (sum_eq_100 : ∑ i in finset.range n, a (i + 1) = 100) :
  n = 11 :=
sorry

end maximum_n_l600_600361


namespace time_to_complete_l600_600415

noncomputable def time_to_complete_mile : Real :=
  let highway_length : Real := 5280  -- length in feet
  let highway_width : Real := 80     -- width in feet
  let quarter_circle_radius : Real := highway_width / 2
  let speed : Real := 10            -- speed in miles per hour
  let quarter_circle_length : Real := (2 * Real.pi * quarter_circle_radius) / 4
  let num_quarter_circles : Real := highway_length / (highway_width * 4) * 4
  let total_distance : Real := num_quarter_circles * quarter_circle_length
  let total_distance_miles : Real := total_distance / 5280
  total_distance_miles / speed

theorem time_to_complete :
  time_to_complete_mile = Real.pi / 10 :=
by
  -- Proof to be provided
  sorry

end time_to_complete_l600_600415


namespace range_of_x_l600_600541

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 * x else 2 * -x

theorem range_of_x {x : ℝ} :
  f (1 - 2 * x) < f 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end range_of_x_l600_600541


namespace area_of_triangle_ABD_l600_600685

-- Define the trapezoid ABCD
structure Trapezoid (A B C D : Type) :=
(area : ℝ)
(length_AB : ℝ)
(length_CD : ℝ)
(is_trapezoid : Prop)

-- Given initial conditions
def input_conditions (A B C D : Type) : Trapezoid A B C D :=
  { area := 18,
    length_AB := l_AB,
    length_CD := 3 * l_AB,
    is_trapezoid := true }

-- Prove the area of triangle ABD is 4.5
noncomputable def triangle_area_proof {A B C D: Type} (input : Trapezoid A B C D) : Prop :=
  ∃ (area_ABD : ℝ), area_ABD = 4.5 ∧ 
  input.area = 18 ∧
  input.length_CD = 3 * input.length_AB

theorem area_of_triangle_ABD (A B C D : Type) :
  ∃ (ABD : Trapezoid A B C D), triangle_area_proof ABD :=
by
  sorry

end area_of_triangle_ABD_l600_600685


namespace complement_of_A_in_U_is_correct_l600_600283

-- Define universal set U and set A
def U := {1, 2, 3, 4, 5}
def A := {1, 2}

-- Define the complement of A with respect to U
def compl_U_A := U \ A

-- Prove that the complement of A with respect to U is {3, 4, 5}
theorem complement_of_A_in_U_is_correct : compl_U_A = {3, 4, 5} :=
by
  sorry

end complement_of_A_in_U_is_correct_l600_600283


namespace integer_solutions_2n_plus_7_eq_x2_l600_600565

theorem integer_solutions_2n_plus_7_eq_x2 :
  ∀ (n x : ℤ), 2^n + 7 = x^2 → (n = 1 ∧ x = 3) ∨ (n = 1 ∧ x = -3) → True :=
sorry

end integer_solutions_2n_plus_7_eq_x2_l600_600565


namespace polynomial_degree_n_f_l600_600402

noncomputable def f : ℕ → ℚ := sorry

theorem polynomial_degree_n_f (n : ℕ) (h_poly : ∃ p : ℚ[X], degree p = n ∧ ∀ x, p.eval x = f x)
(h_f0 : f 0 = 0)
(h_f1 : f 1 = 1 / 2)
(h_f2 : f 2 = 2 / 3)
(h_fn : ∀ k, 0 ≤ k ∧ k ≤ n → f k = k / (k + 1)) :
  (if n % 2 = 1 then f (n + 1) = n / (n + 2) else f (n + 1) = 1) := 
sorry

end polynomial_degree_n_f_l600_600402


namespace selling_price_per_bowl_l600_600882

-- Definitions based on the conditions
def num_bowls : Nat := 118
def cost_per_bowl : ℕ := 12
def num_bowls_sold : Nat := 102
def gain_percentage : ℝ := 8.050847457627118 / 100
noncomputable def total_cost : ℝ := num_bowls * cost_per_bowl

-- The statement to prove
theorem selling_price_per_bowl :
  let profit := gain_percentage * total_cost
  let total_sp := total_cost + profit
  let selling_price_per_bowl := total_sp / num_bowls_sold
  selling_price_per_bowl = 15 := by
  sorry

end selling_price_per_bowl_l600_600882


namespace problem_triang_XYZ_l600_600665

theorem problem_triang_XYZ (X Y Z F G Q : Type)
  [InTriangle XYZ : Triangle X Y Z]
  (H_YF_FG : (segment Y F) / (segment F G) = 4 / 1)
  (H_XG_GZ : (segment X G) / (segment G Z) = 2 / 1)
  (H_Q_intersect : ∃ Q, Q = intersection (line Y F) (line X G)) :
  (segment Y Q) / (segment Q F) = 4 :=
  sorry

end problem_triang_XYZ_l600_600665


namespace pairs_sold_l600_600022

-- Define the given conditions
def initial_large_pairs : ℕ := 22
def initial_medium_pairs : ℕ := 50
def initial_small_pairs : ℕ := 24
def pairs_left : ℕ := 13

-- Translate to the equivalent proof problem
theorem pairs_sold : (initial_large_pairs + initial_medium_pairs + initial_small_pairs) - pairs_left = 83 := by
  sorry

end pairs_sold_l600_600022


namespace trig_relationship_l600_600964

-- Defining the conditions
variables {α β : ℝ}

-- Assuming α and β are within the given intervals
theorem trig_relationship (hα : α ∈ {0, (Real.pi / 2)})
    (hβ : β ∈ {0, (Real.pi / 2)})
    (h_tan : Real.tan α = (1 + Real.sin β) / Real.cos β) : 
    2 * α - β = (Real.pi / 2) := by
  sorry

end trig_relationship_l600_600964


namespace opposite_face_of_turquoise_is_orange_l600_600435

def colors := ℕ → String
constants (O S Y V I T : String) (cube_coloring : colors)

-- Conditions
axiom unique_colors : (cube_coloring 0 = O) ∧ (cube_coloring 1 = S) ∧ (cube_coloring 2 = Y) ∧ (cube_coloring 3 = V) ∧ (cube_coloring 4 = I) ∧ (cube_coloring 5 = T)
axiom view1 : (cube_coloring 0 = O) ∧ (cube_coloring 2 = Y) ∧ (cube_coloring 1 = S)
axiom view2 : (cube_coloring 0 = O) ∧ (cube_coloring 4 = I) ∧ (cube_coloring 1 = S)
axiom view3 : (cube_coloring 0 = O) ∧ (cube_coloring 3 = V) ∧ (cube_coloring 1 = S)

-- Prove that color opposite to T is O
theorem opposite_face_of_turquoise_is_orange :
  cube_coloring 5 = T →
  cube_coloring 0 = O :=
by sorry

end opposite_face_of_turquoise_is_orange_l600_600435


namespace foolish_spy_statement_l600_600849

-- Definitions
inductive Character
| Knight
| Liar
| Spy

open Character

def always_tells_truth (c : Character) : Prop :=
  c = Knight

def always_lies (c : Character) : Prop :=
  c = Liar

noncomputable def can_either_lie_or_tell_truth (c : Character) : Prop :=
  c = Spy

-- The statement under consideration
def statement_about_self (statement : String) : Prop :=
  statement = "I am not a knight"

-- The theorem to be proven
theorem foolish_spy_statement : ∀ (c : Character),
  statement_about_self "I am not a knight" →
  (always_tells_truth c → false) →
  (always_lies c → false) →
  can_either_lie_or_tell_truth c :=
by
  intro c hc hkn hlr
  cases c
  case Knight =>
    apply hkn sorry
  case Liar =>
    apply hlr sorry
  case Spy =>
    exact can_either_lie_or_tell_truth c

end foolish_spy_statement_l600_600849


namespace color_grid_cells_l600_600585

theorem color_grid_cells (k : ℕ) : 
  ∃ (cells : set (ℤ × ℤ)), 
    (∀ x, x ∈ cells → ∃ k', 0 < k' ∧ k' ≤ k) ∧
    (∀ i, ∃ hlines, (∀ (hx : x ∈ hlines), (hx ∈ cells) ∨ (hx ∉ cells ∧ hx ∉ cells))) ∥
    (∀ i, ∃ vlines, (∀ (vx : x ∈ vlines), (vx ∈ cells) ∨ (vx ∉ cells ∧ vx ∉ cells))) ∥
    (∀ j, ∃ dlines, (∀ (dx : x ∈ dlines), (dx ∈ cells) ∨ (dx ∉ cells ∧ dx ∉ cells))) := 
sorry

end color_grid_cells_l600_600585


namespace range_of_a_l600_600634

noncomputable def f (a x : ℝ) : ℝ :=
  a^x + x^2 - x * Real.log a

theorem range_of_a (a : ℝ) (h : ∀ x1 x2 ∈ set.Icc (0 : ℝ) 1, |f a x1 - f a x2| ≤ a - 1) :
  a ≥ Real.exp 1 :=
begin
  sorry
end

end range_of_a_l600_600634


namespace p_necessary_for_q_l600_600649

variable (x : ℝ)

def p := (x - 3) * (|x| + 1) < 0
def q := |1 - x| < 2

theorem p_necessary_for_q : (∀ x, q x → p x) ∧ (∃ x, q x) ∧ (∃ x, ¬(p x ∧ q x)) := by
  sorry

end p_necessary_for_q_l600_600649


namespace max_cables_used_eq_375_l600_600469

-- Conditions for the problem
def total_employees : Nat := 40
def brand_A_computers : Nat := 25
def brand_B_computers : Nat := 15

-- The main theorem we want to prove
theorem max_cables_used_eq_375 
  (h_employees : total_employees = 40)
  (h_brand_A_computers : brand_A_computers = 25)
  (h_brand_B_computers : brand_B_computers = 15)
  (cables_connectivity : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), Prop)
  (no_initial_connections : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), ¬ cables_connectivity a b)
  (each_brand_B_connected : ∀ (b : Fin brand_B_computers), ∃ (a : Fin brand_A_computers), cables_connectivity a b)
  : ∃ (n : Nat), n = 375 := 
sorry

end max_cables_used_eq_375_l600_600469


namespace part1_part2_l600_600175

-- Definition of the function f
def f (x a : ℝ) : ℝ := (x + a) / Real.exp x

-- Part (I)
theorem part1 (a : ℝ) : (∀ x: ℝ, x < 2 → deriv (λ x, f x a) x ≥ 0) ↔ a ≤ -1 := by
  sorry

-- Part (II)
theorem part2 {x0 : ℝ} (hx₀ : x0 < 1) :
  let f (x : ℝ) := x / Real.exp x
  let g (x : ℝ) := (deriv f x0) * (x - x0) + f x0
  ∀ x : ℝ, f x ≤ g x := by
  sorry

end part1_part2_l600_600175


namespace sector_max_area_l600_600969

noncomputable def max_sector_area (R c : ℝ) : ℝ := 
  if h : R = c / 4 then c^2 / 16 else 0 -- This is just a skeleton, actual proof requires conditions
-- State the theorem that relates conditions to the maximum area.
theorem sector_max_area (R c α : ℝ) 
  (hc : c = 2 * R + R * α) : 
  (∃ R, R = c / 4) → max_sector_area R c = c^2 / 16 :=
by 
  sorry

end sector_max_area_l600_600969


namespace cos_alpha_correct_l600_600991

def point := (ℝ × ℝ)

noncomputable def cos_alpha (P : point) : ℝ := 
  if h : P ≠ (0, 0) then
    let x := P.1 
    let y := P.2 
    let r := Real.sqrt (x^2 + y^2) 
    x / r
  else 0

theorem cos_alpha_correct (P : point) (h : P = (4, -3)) : cos_alpha P = 4 / 5 := 
by
  sorry

end cos_alpha_correct_l600_600991


namespace proof_abc_l600_600268

noncomputable def abc_non_suff_necessary (a b c : ℝ) : Prop :=
  ¬ ((abc = 1 → (1/ℝ.sqrt a + 1/ℝ.sqrt b + 1/ℝ.sqrt c ≤ a + b + c)) ∧
     ((1/ℝ.sqrt a + 1/ℝ.sqrt b + 1/ℝ.sqrt c ≤ a + b + c) → abc = 1))

theorem proof_abc (a b c : ℝ) (h : abc = 1) :
  abc_non_suff_necessary a b c :=
sorry

end proof_abc_l600_600268


namespace variance_implies_standard_deviation_l600_600169

noncomputable def variance_to_standard_deviation (σ² : ℝ) (h : σ² = 2) : ℝ :=
  let σ := real.sqrt σ²
  in σ

theorem variance_implies_standard_deviation :
  ∀ (σ² : ℝ), σ² = 2 → variance_to_standard_deviation σ² (by assumption) = real.sqrt 2 :=
by
  assume σ² h
  unfold variance_to_standard_deviation
  rw h
  apply rfl

end variance_implies_standard_deviation_l600_600169


namespace ratio_a3_b3_l600_600189

theorem ratio_a3_b3 (a : ℝ) (ha : a ≠ 0)
  (h1 : a = b₁)
  (h2 : a * q * b = 2)
  (h3 : b₄ = 8 * a * q^3) :
  (∃ r : ℝ, r = -5 ∨ r = -3.2) :=
by
  sorry

end ratio_a3_b3_l600_600189


namespace total_miles_hilt_l600_600712

def totalMiles (miles : List (String × Nat)) : Nat :=
  miles.foldl (fun acc (_, n) => acc + n) 0

constant miles_run : List (String × Nat)
constant miles_swim : List (String × Nat)
constant miles_bike : List (String × Nat)

axiom miles_run_def : miles_run = [("Monday", 3), ("Wednesday", 2), ("Friday", 7)]
axiom miles_swim_def : miles_swim = [("Monday", 1), ("Friday", 2)]
axiom miles_bike_def : miles_bike = [("Wednesday", 6), ("Friday", 3), ("Sunday", 10)]

theorem total_miles_hilt : 
  totalMiles (miles_run ++ miles_swim ++ miles_bike) = 34 :=
by 
  rw [miles_run_def, miles_swim_def, miles_bike_def]
  sorry

end total_miles_hilt_l600_600712


namespace find_s_l600_600628

section
variables {a b c p q s : ℕ}

-- Conditions given in the problem
variables (h1 : a + b = p)
variables (h2 : p + c = s)
variables (h3 : s + a = q)
variables (h4 : b + c + q = 18)
variables (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
variables (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0)

-- Statement of the problem
theorem find_s (h1 : a + b = p) (h2 : p + c = s) (h3 : s + a = q) (h4 : b + c + q = 18)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
  (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0) :
  s = 9 :=
sorry
end

end find_s_l600_600628


namespace arrangement_count_l600_600890

def department_count : ℕ := 3
def people_per_department : ℕ := 2
def total_people : ℕ := department_count * people_per_department
def returning_people : ℕ := 2
def max_people_per_department : ℕ := 1

theorem arrangement_count :
  let category1_choices := department_count * 2
  let category2_choices := (choose department_count 2) * (2 ^ 2) * 3
  let total_arrangements := category1_choices + category2_choices
  total_arrangements = 42 := by
sorry

end arrangement_count_l600_600890


namespace solve_logarithmic_eq_l600_600307

theorem solve_logarithmic_eq (x : ℝ) :
  log 3 ((4 * x + 8) / (6 * x - 2)) + log 3 ((6 * x - 2) / (x - 3)) = 3 →
  x = 89 / 23 :=
by
  sorry

end solve_logarithmic_eq_l600_600307


namespace b_general_formula_sum_first_20_terms_is_300_l600_600142

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a(n) + 1 else a(n) + 2

-- Define the sequence b_n as bₙ = a₂ₙ
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof goal for part (1)
theorem b_general_formula (n : ℕ) : b n = 3 * n - 1 := sorry

-- Sum of the first 20 terms of the sequence a_n
def sum_first_20_terms : ℕ :=
  (List.range (20)).sum (λ n, a n)

-- Proof goal for part (2)
theorem sum_first_20_terms_is_300 : sum_first_20_terms = 300 := sorry

end b_general_formula_sum_first_20_terms_is_300_l600_600142


namespace main_theorem_l600_600965

noncomputable def proof_problem (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2)

noncomputable def equality_case (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  α = π / 3 → 2 * Real.sin (2 * α) = Real.cos (α / 2)

theorem main_theorem (α : ℝ) (h1 : 0 < α) (h2 : α < π) :
  proof_problem α h1 h2 ∧ equality_case α h1 h2 :=
by
  sorry

end main_theorem_l600_600965


namespace tournament_log2_n_l600_600935

theorem tournament_log2_n (teams : ℕ) (games : ℕ) 
    (no_draws : ∀ t₁ t₂ : ℕ, t₁ ≠ t₂ → t₁ < teams → t₂ < teams → 0.5)
    (outcomes : ℕ)
    (unique_wins: ℕ)
    (power_2_factor : ℕ) :
  let m := unique_wins
  let n := 2 ^ (games - power_2_factor)
  coprime m n ∧ log 2 n = 1178 := 
  m = 50! ∧ outcomes = 2 ^ games ∧ unique_wins = m / 2 ^ power_2_factor ∧ games = 1225 ∧ power_2_factor = 47 := 
sorry

end tournament_log2_n_l600_600935


namespace sum_of_minimal_area_m_l600_600356

noncomputable def point1 : ℝ × ℝ := (2, 3)
noncomputable def point2 : ℝ × ℝ := (10, 8)
noncomputable def point3 (m : ℤ) : ℝ × ℝ := (6, m)

noncomputable def area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)).abs

theorem sum_of_minimal_area_m : 
  ∃ (m1 m2 : ℤ), (area point1 point2 (point3 m1) = 0) → (area point1 point2 (point3 m2) = 0) → 
  m1 + m2 = 13 :=
by sorry

end sum_of_minimal_area_m_l600_600356


namespace digit_2567_l600_600271

def nth_digit_in_concatenation (n : ℕ) : ℕ :=
  sorry

theorem digit_2567 : nth_digit_in_concatenation 2567 = 8 :=
by
  sorry

end digit_2567_l600_600271


namespace cylinder_lateral_surface_area_l600_600728

-- Define the base area and lateral surface attributes
def base_area_of_cylinder (S : ℝ) : Prop :=
  S > 0

def lateral_surface_unfolds_into_square (S : ℝ) : Prop :=
  true -- Assume the unfolding condition is always given implicitly.

-- Define the statement to be proved
theorem cylinder_lateral_surface_area (S : ℝ) (h_base_area : base_area_of_cylinder S) (h_unfolds : lateral_surface_unfolds_into_square S) : 
  let r := real.sqrt (S / real.pi) in
  let circumference := 2 * real.pi * r in
  let height := circumference in
  let lateral_surface_area := height * height in
  lateral_surface_area = 4 * real.pi * S := 
by sorry

end cylinder_lateral_surface_area_l600_600728


namespace tangent_line_through_P_l600_600171

-- Define the curve
def curve (x : ℝ) : ℝ := (1 / 3) * x^3 + 4 / 3

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the tangent line equations
def tangent1 (x y : ℝ) : Prop := 4 * x - y - 4 = 0
def tangent2 (x y : ℝ) : Prop := x - y + 2 = 0

-- The proof statement
theorem tangent_line_through_P
  (x_ P : ℝ × ℝ)
  (hP : P = (2, 4))
  (h_curve : ∀ x, curve x = (1 / 3) * x^3 + 4 / 3)
  : (tangent1 P.1 P.2) ∨ (tangent2 P.1 P.2) :=
sorry

end tangent_line_through_P_l600_600171


namespace solve_equation_l600_600310

theorem solve_equation (x : ℝ) (h : x ≠ 1) : -x^2 = (2 * x + 4) / (x - 1) → (x = -2 ∨ x = 1) :=
by
  sorry

end solve_equation_l600_600310


namespace sequence_factorial_l600_600186

theorem sequence_factorial (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n > 0 → a n = n * a (n - 1)) :
  ∀ n : ℕ, a n = Nat.factorial n :=
by
  sorry

end sequence_factorial_l600_600186


namespace minimum_positive_period_of_f_l600_600740

noncomputable def f (x : ℝ) : ℝ :=
  (sin (2 * x) + sin (2 * x + π / 3)) / (cos (2 * x) + cos (2 * x + π / 3))

theorem minimum_positive_period_of_f : 
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T)  :=
by
  use π / 2
  split
  . linarith
  . sorry

end minimum_positive_period_of_f_l600_600740


namespace simplest_quadratic_radical_expr_l600_600393

-- Define the given radical expressions
def expr_A (a : ℝ) := Real.sqrt (2 * a)
def expr_B (a : ℝ) := Real.sqrt (a^2)
def expr_C (x y : ℝ) := Real.sqrt (5 * x^2 * y)
def expr_D := Real.sqrt (1 / 3)

-- State the theorem
theorem simplest_quadratic_radical_expr (a x y : ℝ) :
  expr_A a = Real.sqrt (2 * a) ∧
  expr_B a = Real.sqrt (a^2) ∧
  expr_C x y = Real.sqrt (5 * x^2 * y) ∧
  expr_D = Real.sqrt (1 / 3) →
  expr_A a = Real.sqrt (2 * a) := 
sorry

end simplest_quadratic_radical_expr_l600_600393


namespace range_of_m_l600_600999

noncomputable def f (a x : ℝ) := a * (x^2 + 1) + Real.log x

theorem range_of_m (a m : ℝ) (h₁ : a ∈ Set.Ioo (-4 : ℝ) (-2))
  (h₂ : ∀ x ∈ Set.Icc (1 : ℝ) (3), ma - f a x > a^2) : m ≤ -2 := 
sorry

end range_of_m_l600_600999


namespace number_of_valid_eight_digit_numbers_l600_600087

-- Define the problem statement and conditions
def is_valid_eight_digit_number (digits : List ℕ) : Prop :=
  digits.length = 8 ∧ (∀ d ∈ digits, d ≤ 9) ∧ digits.product = 7000

-- Define the theorem to state the correctness of the answer
theorem number_of_valid_eight_digit_numbers : 
  ∃! (n : ℕ), n = 5600 ∧ ∃ (numbers : List (List ℕ)), 
    (∀ digits ∈ numbers, is_valid_eight_digit_number digits) ∧ 
    numbers.length = n :=
sorry

end number_of_valid_eight_digit_numbers_l600_600087


namespace find_length_QT_l600_600223

variable (a : ℝ)
variable (h : a^2 ≥ 5)

theorem find_length_QT (QR : ℝ) (PR : ℝ) (PT : ℝ) (PQ2 : ℝ) :
  QR = 2 → PR = a → PT = 3 →
  PQ2 = PR ^ 2 + QR ^ 2 → PQ2 = PT ^ 2 + QT ^ 2 →
  QT = real.sqrt (a^2 - 5) := 
by 
  intros hQR hPR hPT hpq1 hpq2
  have h0 : PQ2 = a^2 + 4 := by rw [hPR, hQR]; linarith
  have h1 : PQ2 = 9 + QT ^ 2 := by rw [hPT]; linarith
  rw h0 at h1
  linarith
  use sqrt_le hpq1

end find_length_QT_l600_600223


namespace lines_parallel_distance_correct_l600_600624

-- Definitions for the problem context
def line1 (x y : ℝ) (a : ℝ) : Prop := x + a * y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def are_parallel (a : ℝ) : Prop := a = -1 / 2

-- Function to compute the distance between two parallel lines given specific coefficients
def distance_between_lines (C1 C2 A B : ℝ) : ℝ := |C2 - C1| / (Real.sqrt (A^2 + B^2))

-- Mathematical proof problem statement based on the given problem conditions
theorem lines_parallel_distance_correct (a : ℝ) (h_parallel : are_parallel a) :
  distance_between_lines (-1) 1 2 (-1) = 3 * Real.sqrt 5 / 5 :=
by
  sorry

end lines_parallel_distance_correct_l600_600624


namespace possible_r_values_l600_600032

noncomputable def triangle_area (r : ℝ) : ℝ := (r - 3) ^ (3 / 2)

theorem possible_r_values :
  {r : ℝ | 16 ≤ triangle_area r ∧ triangle_area r ≤ 128} = {r : ℝ | 7 ≤ r ∧ r ≤ 19} :=
by
  sorry

end possible_r_values_l600_600032


namespace complete_the_square_l600_600823

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l600_600823


namespace calculate_expression_l600_600490

theorem calculate_expression : (sqrt 27 / (sqrt 3 / 2) * (2 * sqrt 2) - (6 * sqrt 2)) = 6 * sqrt 2 :=
by
  -- Taking these steps from the solution, we should finally arrive at the required proof
  sorry

end calculate_expression_l600_600490


namespace cards_red_side_up_l600_600514

/-- 
  Given 100 cards numbered 1 to 100. Initially, all are red side up.
  Flipping operation: first, flip all even-numbered cards; second, flip all multiples of 3.
  Prove: 33 cards have the red side up.
-/
theorem cards_red_side_up : 
  let N := 100,
      even_cards := {n | 1 ≤ n ∧ n ≤ N ∧ n % 2 = 0},
      multiple_of_3 := {n | 1 ≤ n ∧ n ≤ N ∧ n % 3 = 0},
      flipped_once := even_cards,
      flipped_twice := even_cards ∩ multiple_of_3
  in
  (100 - ((even_cards \ flipped_twice).union (multiple_of_3 \ flipped_twice)).card = 33) :=
sorry

end cards_red_side_up_l600_600514


namespace period_of_cosine_shifted_determine_period_of_cosine_3x_plus_shift_l600_600548

theorem period_of_cosine_shifted (k : ℝ) (a : ℝ) :
  ∀ x : ℝ, cos (k * x + a) = cos (k * (x + (2 * π / k)) + a) :=
by
  sorry

theorem determine_period_of_cosine_3x_plus_shift :
  ∀ x : ℝ, cos (3 * x + (π / 6)) = cos (3 * (x + (2 * π / 3)) + (π / 6)) :=
by
  exact period_of_cosine_shifted 3 (π / 6)

end period_of_cosine_shifted_determine_period_of_cosine_3x_plus_shift_l600_600548


namespace unique_red_coloring_l600_600974

theorem unique_red_coloring (n : ℕ) (h : n > 2) :
  ∃! A : finset ℕ,
    A ⊆ finset.range (2 * n + 1) ∧
    A.card = n + 1 ∧
    ∀ x y z ∈ A, x ≠ y → y ≠ z → z ≠ x → x + y ≠ z ∧
    A = finset.range (n + 1) \u (finset.range (2 * n + 1) \ finset.range n) :=
sorry

end unique_red_coloring_l600_600974


namespace coefficient_of_x7_in_expansion_l600_600730

theorem coefficient_of_x7_in_expansion :
  (let expr := (x^2 + x)^5 in
  find_coeff expr x 7) = 10 :=
sorry

end coefficient_of_x7_in_expansion_l600_600730


namespace find_abc_value_l600_600200

noncomputable def abc_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * (b + c) = 168) (h5 : b * (c + a) = 153) (h6 : c * (a + b) = 147) : ℝ :=
abc

theorem find_abc_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * (b + c) = 168) (h5 : b * (c + a) = 153) (h6 : c * (a + b) = 147) :
  abc_value a b c h1 h2 h3 h4 h5 h6 = 720 :=
sorry

end find_abc_value_l600_600200


namespace radius_of_circle_l600_600744

theorem radius_of_circle (r : ℝ) : 3 * 2 * Real.pi * r = Real.pi * r^2 → r = 6 :=
by {
  intro h,
  have h1 : 6 * Real.pi * r = Real.pi * r^2 := by rw [←mul_assoc, ←h],
  have h2 : 6 * r = r^2 := by rw [←mul_div_cancel_left 'Real.pi, h1],
  have h3 : r^2 - 6 * r = 0 := by ring,
  have h4 : r * (r - 6) = 0 := by rw h3,
  cases eq_zero_or_eq_zero_of_mul_eq_zero h4 with h5 h6,
  { exact h5, },
  { exact h6, }
} sorry

end radius_of_circle_l600_600744


namespace tank_capacity_75_l600_600869

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end tank_capacity_75_l600_600869


namespace range_of_root_difference_l600_600635

variable (a b c d : ℝ)
variable (x1 x2 : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_of_root_difference
  (h1 : a ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hroot1 : f a b c x1 = 0)
  (hroot2 : f a b c x2 = 0)
  : |x1 - x2| ∈ Set.Ico (Real.sqrt 3 / 3) (2 / 3) := sorry

end range_of_root_difference_l600_600635


namespace circle_eq_problem1_circle_eq_problem2_l600_600941

-- Problem 1
theorem circle_eq_problem1 :
  (∃ a b r : ℝ, (x - a)^2 + (y - b)^2 = r^2 ∧
  a - 2 * b - 3 = 0 ∧
  (2 - a)^2 + (-3 - b)^2 = r^2 ∧
  (-2 - a)^2 + (-5 - b)^2 = r^2) ↔
  (x + 1)^2 + (y + 2)^2 = 10 :=
sorry

-- Problem 2
theorem circle_eq_problem2 :
  (∃ D E F : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ∧
  (1:ℝ)^2 + (0:ℝ)^2 + D * 1 + E * 0 + F = 0 ∧
  (-1:ℝ)^2 + (-2:ℝ)^2 - D * 1 - 2 * E + F = 0 ∧
  (3:ℝ)^2 + (-2:ℝ)^2 + 3 * D - 2 * E + F = 0) ↔
  x^2 + y^2 - 2 * x + 4 * y + 1 = 0 :=
sorry

end circle_eq_problem1_circle_eq_problem2_l600_600941


namespace count_valid_two_digit_integers_l600_600994

def digits : List ℕ := [1, 3, 5, 8, 9]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_valid_two_digit (n : ℕ) : Prop := 
  10 ≤ n ∧ n < 100 ∧ is_odd n ∧ 
  ∃ (t u : ℕ), t ≠ u ∧ t ∈ digits ∧ u ∈ digits ∧ n = 10 * t + u

theorem count_valid_two_digit_integers : 
  (Finset.filter is_valid_two_digit (Finset.range 100)).card = 16 := by
  sorry

end count_valid_two_digit_integers_l600_600994


namespace solve_trig_equations_l600_600410

theorem solve_trig_equations (x : ℝ) (k : ℤ) :
    (1 - Real.cos(6 * x) = Real.tan(3 * x) ∧
    1 + Real.sin(2 * x) = (Real.cos(3 * x) + Real.sin(3 * x))^2) →
    (∃ k : ℤ, x = k * Real.pi / 3) ∨ (∃ k : ℤ, x = Real.pi / 12 * (4 * k + 1)) :=
by
  sorry

end solve_trig_equations_l600_600410


namespace probability_of_draw_l600_600668

/-- Given a 6-deck game of cards (each deck contains 52 cards), the probability of drawing a spade 
and then specifically the King of Diamonds consecutively without replacement is 3/622. -/
theorem probability_of_draw :
  let total_cards := 52 * 6 in
  let total_spades := 13 * 6 in
  let total_kod := 6 in
  let p_spade := total_spades / total_cards in
  let p_kod_given_spade := total_kod / (total_cards - 1) in
  (p_spade * p_kod_given_spade = 3 / 622) :=
by
  sorry

end probability_of_draw_l600_600668


namespace bags_bought_l600_600462

theorem bags_bought (initial_bags : ℕ) (bags_given : ℕ) (final_bags : ℕ) (bags_bought : ℕ) :
  initial_bags = 20 → 
  bags_given = 4 → 
  final_bags = 22 → 
  bags_bought = final_bags - (initial_bags - bags_given) → 
  bags_bought = 6 := 
by
  intros h_initial h_given h_final h_buy
  rw [h_initial, h_given, h_final] at h_buy
  exact h_buy

#check bags_bought

end bags_bought_l600_600462


namespace isosceles_right_triangle_example_l600_600832

theorem isosceles_right_triangle_example :
  (5 = 5) ∧ (5^2 + 5^2 = (5 * Real.sqrt 2)^2) :=
by {
  sorry
}

end isosceles_right_triangle_example_l600_600832


namespace shannon_increase_in_C_l600_600288

theorem shannon_increase_in_C : 
  ∀ (W : ℝ) (C1 C2 : ℝ), 
  (C1 = W * real.logb 2 1000) →
  (C2 = W * real.logb 2 12000) → 
  (real.logb 2 2 = 0.3010) → 
  (real.logb 2 3 = 0.4771) → 
  (real.logb 2 5 = 0.6990) → 
  ((C2 - C1) / C1 ≈ 0.36 : Prop) :=
begin
  intros W C1 C2 hC1 hC2 hlog2_2 hlog2_3 hlog2_5,
  sorry
end

end shannon_increase_in_C_l600_600288


namespace sum_binomials_odd_l600_600982

theorem sum_binomials_odd (n : ℕ) (hn : 0 < n) : 
  (∑ l in Finset.range (n + 1), (Nat.choose (2 * n + l) n) * (Nat.choose (n - 1 + l) (n - 1))) % 2 = 1 := 
by
  sorry

end sum_binomials_odd_l600_600982


namespace alice_distance_proof_l600_600898

noncomputable def alice_distance (m1 m2 f nrm: ℝ) : ℝ :=
  let m_to_f := 3.28084
  let west1 := m1 * m_to_f
  let combined_west := (m2 * m_to_f) + f
  let west_dist := west1 + combined_west
  let total_north := nrm
  real.sqrt ((west_dist * west_dist) + (total_north * total_north))

theorem alice_distance_proof :
  alice_distance 12 12 50 40 ≈ 134.8 := 
by {
  sorry
}

end alice_distance_proof_l600_600898


namespace completing_the_square_l600_600810

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l600_600810


namespace diameter_of_circle_l600_600420

theorem diameter_of_circle (A : ℝ) (h : A = 4 * real.pi) : ∃ d : ℝ, d = 4 :=
  sorry

end diameter_of_circle_l600_600420


namespace maximum_primes_in_expressions_l600_600950

/-- Given 20 expressions of the form a + b = c involving unique natural numbers,
  with the condition that there are no three odd numbers among a, b, and c in any expression,
  and the goal to use as many prime numbers as possible, prove that the maximum
  number of prime numbers that can be used is 41. -/
theorem maximum_primes_in_expressions :
  ∃ (expressions : list (ℕ × ℕ × ℕ)), 
    expressions.length = 20 ∧
    (∀ (e : ℕ × ℕ × ℕ), e ∈ expressions → ∃ (a b c : ℕ), e = (a, b, c) ∧ a + b = c) ∧
    (∀ (e : ℕ × ℕ × ℕ), e ∈ expressions → ¬(is_odd (e.1) ∧ is_odd (e.2) ∧ is_odd (e.3))) ∧
    (∀ (e : ℕ × ℕ × ℕ), ∀ (a b c : ℕ), e = (a, b, c) → 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (a ∈ finset.primes ∨ b ∈ finset.primes ∨ c ∈ finset.primes)) ∧
    finset.count finset.primes (expressions.to_finset.image (λ e, e.1)).card +
      finset.count finset.primes (expressions.to_finset.image (λ e, e.2)).card +
      finset.count finset.primes (expressions.to_finset.image (λ e, e.3)).card = 41 :=
sorry

end maximum_primes_in_expressions_l600_600950


namespace man_speed_l600_600031

theorem man_speed {m l: ℝ} (TrainLength : ℝ := 385) (TrainSpeedKmH : ℝ := 60)
  (PassTimeSeconds : ℝ := 21) (RelativeSpeed : ℝ) (ManSpeedKmH : ℝ) 
  (ConversionFactor : ℝ := 3.6) (expected_speed : ℝ := 5.99) : 
  RelativeSpeed = TrainSpeedKmH/ConversionFactor + m/ConversionFactor ∧ 
  TrainLength = RelativeSpeed * PassTimeSeconds →
  abs (m*ConversionFactor - expected_speed) < 0.01 :=
by
  sorry

end man_speed_l600_600031


namespace simplify_expression_l600_600895

theorem simplify_expression : (7 - (-3) + (-5) - (+2) = 7 + 3 - 5 - 2) :=
  sorry

end simplify_expression_l600_600895


namespace min_value_of_squared_sum_l600_600702

open Real

theorem min_value_of_squared_sum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
  ∃ m, m = (x^2 + y^2 + z^2) ∧ m = 16 / 3 :=
by
  sorry

end min_value_of_squared_sum_l600_600702


namespace arrangment_ways_basil_tomato_l600_600043

theorem arrangment_ways_basil_tomato (basil_plants tomato_plants : Finset ℕ) 
  (hb : basil_plants.card = 5) 
  (ht : tomato_plants.card = 3) 
  : (∃ total_ways : ℕ, total_ways = 4320) :=
by
  sorry

end arrangment_ways_basil_tomato_l600_600043


namespace triangle_ABC_degenerate_l600_600150

noncomputable def point (x y : ℝ) := (x, y)

def A : ℝ × ℝ := point 1 2
def B : ℝ × ℝ := point 1 2
def C : ℝ × ℝ := point 1 2

theorem triangle_ABC_degenerate : A = B ∧ B = C → (∀ a b c : ℝ × ℝ, a = b → b = c → (¬∃ (T : ℝ × ℝ × ℝ), T = (a, b, c))) :=
by
  intro h
  sorry

end triangle_ABC_degenerate_l600_600150


namespace matrix_sum_correct_l600_600576

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![2, 5]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![−6, 8], ![-3, -10]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![−2, 5], ![-1, -5]]

theorem matrix_sum_correct : A + B = C := 
by sorry

end matrix_sum_correct_l600_600576


namespace scott_ratatouille_yield_l600_600302

theorem scott_ratatouille_yield :
  let eggplants := 5 * 2
  let zucchini := 4 * 2
  let tomatoes := 4 * 3.5
  let onions := 3 * 1
  let basil := 1 * 5
  let total_cost := eggplants + zucchini + tomatoes + onions + basil
  let cost_per_quart := 10
  total_cost / cost_per_quart = 4 :=
by
  simp [eggplants, zucchini, tomatoes, onions, basil, total_cost, cost_per_quart]
  sorry

end scott_ratatouille_yield_l600_600302


namespace gcd_poly_l600_600614

-- Defining the conditions as stated in part a:
def is_even_multiple_of_1171 (b : ℤ) : Prop :=
  ∃ k : ℤ, b = 1171 * k * 2

-- Stating the main theorem based on the conditions and required proof in part c:
theorem gcd_poly (b : ℤ) (h : is_even_multiple_of_1171 b) : Int.gcd (3 * b ^ 2 + 47 * b + 79) (b + 17) = 1 := by
  sorry

end gcd_poly_l600_600614


namespace sin_phi_l600_600265

noncomputable def vectors (x y z : V) :=
  (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) ∧
  ¬(x = k • y) ∧ ¬(x = k • z) ∧ ¬(y = k • z) ∧
  ((x × y) × z = (1/2) * ∥y∥ * ∥z∥ * x)

theorem sin_phi (x y z : V) (φ : ℝ) [normed_space ℝ V] 
  (h1 : vectors x y z) :
  sin φ = (√3 / 2) :=
sorry

end sin_phi_l600_600265


namespace moles_of_water_produced_l600_600195

-- Definitions for the chemical reaction
def moles_NaOH := 4
def moles_H₂SO₄ := 2

-- The balanced chemical equation tells us the ratio of NaOH to H₂O
def chemical_equation (moles_NaOH moles_H₂SO₄ moles_H₂O moles_Na₂SO₄: ℕ) : Prop :=
  2 * moles_NaOH = 2 * moles_H₂O ∧ moles_H₂SO₄ = 1 ∧ moles_Na₂SO₄ = 1

-- The actual proof statement
theorem moles_of_water_produced : 
  ∀ (m_NaOH m_H₂SO₄ m_Na₂SO₄ : ℕ), 
  chemical_equation m_NaOH m_H₂SO₄ 4 m_Na₂SO₄ → moles_H₂O = 4 :=
by
  intros m_NaOH m_H₂SO₄ m_Na₂SO₄ chem_eq
  -- Placeholder for the actual proof.
  sorry

end moles_of_water_produced_l600_600195


namespace S_63_value_l600_600190

noncomputable def b (n : ℕ) : ℚ := (3 + (-1)^(n-1))/2

noncomputable def a : ℕ → ℚ
| 0       => 0
| 1       => 2
| (n+2)   => if (n % 2 = 0) then - (a (n+1))/2 else 2 - 2*(a (n+1))

noncomputable def S : ℕ → ℚ
| 0       => 0
| (n+1)   => S n + a (n+1)

theorem S_63_value : S 63 = 464 := by
  sorry

end S_63_value_l600_600190


namespace total_chips_l600_600386

def chips_count (V_c V_v S_c S_v : ℕ) : ℕ :=
  V_c + V_v + S_c + S_v

theorem total_chips :
  ∀ (V_c V_v S_c S_v : ℕ),
    (V_c = S_c + 5) →
    (S_v = (3 / 4) * V_v) →
    (V_v = 20) →
    (S_c = 25) →
    chips_count V_c V_v S_c S_v = 90 :=
by
  intros V_c V_v S_c S_v h1 h2 h3 h4
  rw [h4, h3, h1, h2]
  have V_c_val : V_c = 30 := by linarith,
  have S_v_val : S_v = 15 := by norm_num,
  rw [V_c_val, S_v_val, h3, h4]
  norm_num
  sorry

end total_chips_l600_600386


namespace sequence_from_520_to_523_is_0_to_3_l600_600654

theorem sequence_from_520_to_523_is_0_to_3 
  (repeating_pattern : ℕ → ℕ)
  (h_periodic : ∀ n, repeating_pattern (n + 5) = repeating_pattern n) :
  ((repeating_pattern 520, repeating_pattern 521, repeating_pattern 522, repeating_pattern 523) = (repeating_pattern 0, repeating_pattern 1, repeating_pattern 2, repeating_pattern 3)) :=
by {
  sorry
}

end sequence_from_520_to_523_is_0_to_3_l600_600654


namespace complete_the_square_l600_600822

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l600_600822


namespace circumcircles_intersect_at_single_point_l600_600603

variables {A B C P Q : Type} [EuclideanGeometry A B C P Q] 

/-- Given an acute-angled triangle ABC and 
a point P that does not coincide with the orthocenter of triangle ABC, 
the circumcircles passing through the midpoints of the sides of triangles 
PAB, PAC, PBC, and ABC, as well as the circumcircle passing through the 
orthogonal projections of point P onto the sides of triangle ABC, 
intersect at a single point. -/
theorem circumcircles_intersect_at_single_point (h_acute : is_acute_triangle A B C)
  (h_not_orthocenter : P ≠ orthocenter A B C) :
  ∃ Q, (circumcircle (midpoint A P) (midpoint B C)) Q ∧ 
       (circumcircle (midpoint B P) (midpoint A C)) Q ∧ 
       (circumcircle (midpoint C P) (midpoint A B)) Q ∧ 
       (circumcircle (midpoint A B) (midpoint B C)) Q ∧
       (circumcircle (foot P A B) (foot P B C) (foot P C A)) Q :=
sorry

end circumcircles_intersect_at_single_point_l600_600603


namespace find_a_value_l600_600983

theorem find_a_value
  (a : ℕ)
  (x y : ℝ)
  (h1 : a * x + y = -4)
  (h2 : 2 * x + y = -2)
  (hx_neg : x < 0)
  (hy_pos : y > 0) :
  a = 3 :=
by
  sorry

end find_a_value_l600_600983


namespace simplify_fraction_l600_600304

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end simplify_fraction_l600_600304


namespace last_digit_of_expression_l600_600580

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the sequence in the expression
def sequence (n : ℕ) : list ℕ := 
  (list.range n).map (λ k, if k.even then factorial (k+1) else - factorial (k+1))

-- Define a function to calculate the last digit of the sum
def last_digit (n : ℕ) : ℕ := (sequence n).sum % 10

-- The theorem stating that the last digit of the given expression is 1
theorem last_digit_of_expression : last_digit 2014 = 1 := 
by
  sorry

end last_digit_of_expression_l600_600580


namespace correct_smallest_constant_l600_600606

noncomputable def smallest_constant_inequality (n : ℕ) (hn : n ≥ 2) : ℝ :=
  Inf {c : ℝ | ∀ (x : Fin n → ℝ), (∀ i, 0 ≤ x i) →
    ∑ i j in Finset.filter (λ p : Fin n × Fin n, p.1 < p.2) Finset.univ.product Finset.univ,
    x i * x j * (x i ^ 2 + x j ^ 2) ≤
    c * ((∑ i, x i) ^ 4)}

theorem correct_smallest_constant (n : ℕ) (hn : n ≥ 2) :
  smallest_constant_inequality n hn = 1 / 8 :=
sorry

end correct_smallest_constant_l600_600606


namespace part1_b1_b2_part1_general_formula_part2_sum_20_l600_600138

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then a n + 1 else a n + 2

def b (n : ℕ) : ℕ := a (2 * n)

-- Proving b_1 = 2 and b_2 = 5
theorem part1_b1_b2 : b 1 = 2 ∧ b 2 = 5 :=
by {
  unfold b a,
  simp,
  split,
  {
    rfl -- proof for b_1 = 2
  },
  {
    rfl -- proof for b_2 = 5
  }
}

-- Proving the general formula for b_n
theorem part1_general_formula (n : ℕ) : b n = 3 * n - 1 :=
by {
  induction n with k ih,
  {
    unfold b a,
    simp,
    rfl
  },
  {
    rename ih ih_k,
    unfold b a,
    simp,
    rw [ih_k],
    calc 3 * (k + 1) - 1 = 3 * k + 3 - 1 : by ring
                       ... = 3 * k + 2     : by ring
                       ... = a (2 * k + 2) : by sorry -- Detailed proof needed
  }
}

-- Proving the sum of the first 20 terms of the sequence a_n
theorem part2_sum_20 : (Finset.range 20).sum a = 300 :=
by {
  unfold a,
  have h1 : finset.sum (finset.range 10) (λ n, 3 * n + 1) = 145,
  {
    sorry -- Compute sum of odd terms
  },
  have h2 : finset.sum (finset.range 10) (λ n, 3 * n + 2) = 155,
  {
    sorry -- Compute sum of even terms
  },
  have h3 : finset.sum (finset.range 20) a = 145 + 155,
  {
    sorry -- Combine sums
  },
  exact h3,
}

end part1_b1_b2_part1_general_formula_part2_sum_20_l600_600138


namespace locus_of_P_l600_600172

-- Definitions based on the given conditions
def ellipse_g (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def left_focal_point (F : ℝ × ℝ) : Prop := F = (-1, 0)
def tangent_through_P (P M N : ℝ × ℝ) (p_tangent : ℝ → ℝ) : Prop := 
  ellipse_g M.1 M.2 ∧ 
  ellipse_g N.1 N.2 ∧ 
  p_tangent = λ x, (P.2 - P.2 * x / P.1 + P.2)
def angle_condition (F M N : ℝ × ℝ) : Prop := 
  let cos_angle := (F.1 + M.1) * (F.1 + N.1) + F.2 * M.2 * N.2 in
  cos_angle / (real.sqrt ((F.1 + M.1)^2 + M.2^2) * real.sqrt ((F.1 + N.1)^2 + N.2^2)) = 1 / 2
def locus_equation (x y : ℝ) : Prop := (x - 1)^2 / 6 + y^2 / 2 = 1

-- main theorem statement to prove the locus given the conditions
theorem locus_of_P (P M N F : ℝ × ℝ) (p_tangent : ℝ → ℝ) :
  ellipse_g M.1 M.2 →
  ellipse_g N.1 N.2 →
  left_focal_point F →
  tangent_through_P P M N p_tangent →
  angle_condition F M N →
  locus_equation P.1 P.2 :=
begin
  sorry -- proof is not provided, only statement is required.
end

end locus_of_P_l600_600172


namespace value_of_c_l600_600700

variable (A B C : ℝ)
variable (a b c: ℝ)
variable (cos_C sin_A sin_B: ℝ)

-- Conditions given
def a_eq_2 : a = 2 := sorry
def cos_C_eq_neg_quarter : cos_C = - 1 / 4 := sorry
def sin_relation : 3 * sin_A = 2 * sin_B := sorry
-- Sine rule relation derived: 3a = 2b
def sine_rule_relation (a b : ℝ) : 3 * a = 2 * b := sorry

-- Implementation begins
noncomputable def find_c : Prop :=
  a = 2 ∧ cos_C = -1/4 ∧ 3 * sin_A = 2 * sin_B → c = 4

-- Top-level goal
theorem value_of_c : find_c A B C a b c cos_C sin_A sin_B :=
by {
  sorry
}

end value_of_c_l600_600700


namespace number_of_n_l600_600092

-- Definition of a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- The main statement to prove
theorem number_of_n (n : ℕ) :
  (∃ k : ℕ, k ∈ finset.Icc 1 2000 ∧ (21 * k) ∈ (finset.range 2001).filter is_perfect_square) = 9 :=
sorry

end number_of_n_l600_600092


namespace four_non_coplanar_points_determine_four_planes_l600_600834

theorem four_non_coplanar_points_determine_four_planes :
    ∃ (p1 p2 p3 p4 : ℝ × ℝ × ℝ), 
        ¬(∃ a b c d : ℝ, a * p1.1 + b * p1.2 + c * p1.3 + d = 0) ∧
        ¬(∃ a b c d : ℝ, a * p2.1 + b * p2.2 + c * p2.3 + d = 0) ∧
        ¬(∃ a b c d : ℝ, a * p3.1 + b * p3.2 + c * p3.3 + d = 0) ∧
        ¬(∃ a b c d : ℝ, a * p4.1 + b * p4.2 + c * p4.3 + d = 0) ∧
        (∃ a1 b1 c1 d1 : ℝ, a1 * p1.1 + b1 * p1.2 + c1 * p1.3 + d1 * p1.4 = 0) ∧
        (∃ a2 b2 c2 d2 : ℝ, a2 * p2.1 + b2 * p2.2 + c2 * p2.3 + d2 * p2.4 = 0) ∧
        (∃ a3 b3 c3 d3 : ℝ, a3 * p3.1 + b3 * p3.2 + c3 * p3.3 + d3 * p3.4 = 0) ∧
        (∃ a4 b4 c4 d4 : ℝ, a4 * p4.1 + b4 * p4.2 + c4 * p4.3 + d4 * p4.4 = 0) :=
sorry

end four_non_coplanar_points_determine_four_planes_l600_600834


namespace b1_b2_values_general_formula_b_sum_first_20_l600_600136

def seq_a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then seq_a n + 1 else seq_a n + 2

def seq_b (n : ℕ) : ℕ := seq_a (2 * n)

theorem b1_b2_values : seq_b 1 = 2 ∧ seq_b 2 = 5 := by
  sorry

theorem general_formula_b (n : ℕ) : seq_b n = 3 * n - 1 := by
  sorry

theorem sum_first_20 : (Finset.range 20).sum seq_a = 300 := by
  sorry

end b1_b2_values_general_formula_b_sum_first_20_l600_600136


namespace scout_troop_profit_l600_600453

theorem scout_troop_profit 
  (bars_purchased : ℕ) 
  (cost_per_3_bars : ℕ) 
  (bars_sold : ℕ) 
  (selling_price_per_5_bars : ℕ) 
  (total_bars : bars_purchased = 1500)
  (cost_rate : cost_per_3_bars = 1)
  (sell_rate : selling_price_per_5_bars = 3) :
  ((bars_sold * (0.6)) - (bars_purchased * (1/3))) = 400 := 
by 
  sorry

end scout_troop_profit_l600_600453


namespace part_a_part_d_l600_600198

-- Total number of balls
def balls : Nat := 8

-- Total number of boxes
def boxes : Nat := 4

-- Condition for part (a): Each box should have at least 1 ball
theorem part_a : (multichoose (balls - 1) (boxes - 1)) = 35 := by
  -- Lean proof goes here
  sorry

-- Condition for part (d): Box 2 must contain at least 2 balls, and each other box at least 1
theorem part_d : 
  ((multichoose (balls - 2 - (boxes - 1)) (boxes - 2)) + 
  (multichoose (balls - 3 - (boxes - 1)) (boxes - 2)) + 
  (multichoose (balls - 4 - (boxes - 1)) (boxes - 2)) + 
  (multichoose (balls - 5 - (boxes - 1)) (boxes - 2))) = 20 := by 
  -- Lean proof goes here
  sorry

end part_a_part_d_l600_600198


namespace completing_square_l600_600804

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l600_600804


namespace tan_alpha_value_l600_600121

theorem tan_alpha_value (α : Real) (h : tan (α - π / 4) = 1 / 5) : tan α = 3 / 2 :=
by sorry

end tan_alpha_value_l600_600121


namespace cistern_fill_time_l600_600434

theorem cistern_fill_time (hA : ∀ t : ℕ, 2 * (t / 2) = t)
                          (hB : ∀ t : ℕ, 6 * (t / 6) = t)
                          (hC : ∀ t : ℕ, 4 * (t / 4) = t) : 
  let rate_A := (1 / 2 : ℝ), 
      rate_B := (1 / 6 : ℝ), 
      rate_C := (-1 / 4 : ℝ) in 
  (1 : ℝ) / (rate_A + rate_B + rate_C) = 2.4 := 
by 
  let rate_A := (1 / 2 : ℝ),
      rate_B := (1 / 6 : ℝ),
      rate_C := (-1 / 4 : ℝ) in
  calc 
  (1 : ℝ) / (rate_A + rate_B + rate_C) = (1 : ℝ) / (5 / 12 : ℝ) : by simp only [rate_A, rate_B, rate_C]; field_simp
  ... = 2.4 : by norm_num

#exit

end cistern_fill_time_l600_600434


namespace find_solutions_l600_600309

theorem find_solutions (x y : ℝ) :
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2 ∧ x^2 * y = 20 * x^2 + 3 * y^2) ↔ 
    (x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2) :=
by sorry

end find_solutions_l600_600309


namespace tangent_circle_exists_l600_600385

noncomputable theory

structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

def tangent_to_line (c : Circle) (l : Line) : Prop :=
  -- Distance from the center of the circle to the line equals the radius
  let (cx, cy) := c.center in
  abs (l.slope * cx - cy + l.y_intercept) / (real.sqrt (l.slope^2 + 1)) = c.radius

def tangent_to_circle (c1 c2 : Circle) : Prop :=
  -- Distance between circle centers equals the sum of their radii
  let (c1x, c1y) := c1.center in
  let (c2x, c2y) := c2.center in
  dist (c1x, c1y) (c2x, c2y) = c1.radius + c2.radius

def non_intersecting (l : Line) (c : Circle) : Prop :=
  -- Minimum distance from the line to the center of the circle is greater than the radius
  let (cx, cy) := c.center in
  abs (l.slope * cx - cy + l.y_intercept) / (real.sqrt (l.slope^2 + 1)) > c.radius

theorem tangent_circle_exists
  (r R : ℝ) (L : Line) (C : Circle)
  (h1 : non_intersecting L C)
  (h2 : C.radius = R) :
  ∃ (O' : ℝ × ℝ), 
  ∃ (circle' : Circle), 
    circle'.center = O' ∧ 
    circle'.radius = r ∧ 
    tangent_to_line circle' L ∧ 
    tangent_to_circle circle' C :=
sorry

end tangent_circle_exists_l600_600385


namespace fraction_area_above_line_l600_600739

def point := (ℝ × ℝ)
def Line (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  (m, b)

def squareArea (p1 p2 p3 p4 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let b := abs (x2 - x1)
  let h := abs (y2 - y1)
  b * h

theorem fraction_area_above_line : 
  let p1 := (1, 3)
  let p2 := (4, 1)
  let v1 := (1, 1)
  let v2 := (1, 4)
  let v3 := (4, 4)
  let v4 := (4, 1)
  let line_eq := Line p1 p2
  let area_square := squareArea v1 v2 v3 v4
  (2 / 3) * area_square = area_square - 3 := 
  sorry

end fraction_area_above_line_l600_600739


namespace fox_catches_rabbit_rabbit_escapes_fox_l600_600877

section fox_and_rabbit

variables {u v : ℝ} -- Define the variables u and v

-- Part (a): If 2u > v, the fox can catch the rabbit
theorem fox_catches_rabbit (huv : 2 * u > v) : 
  ∀ (R F : ℝ × ℝ), (R = (0, 1) ∨ R = (1, 0) ∨ R = (-1, 0)) → (F = (0, 0)) → 
  ∃ t : ℝ, (t ≥ 0) ∧ (let Y := R.1 + t * u in 
            let X := F.1 + t * v in X = Y ∨ Y = 0) ∧ 
            (let Y := R.2 + t * u in 
            let X := F.2 + t * v in X = Y ∨ Y = 0) := sorry

-- Part (b): If 2u ≤ v, the rabbit can always run away from the fox
theorem rabbit_escapes_fox (huv : 2 * u ≤ v) : 
  ∀ (R F : ℝ × ℝ), (R = (0, 1) ∨ R = (1, 0) ∨ R = (-1, 0)) → (F = (0, 0)) → 
  ∀ t : ℝ, (t ≥ 0) → (let Y := R.1 + t * u in 
            let X := F.1 + t * v in ¬ (X = Y)) ∧ 
            (let Y := R.2 + t * u in 
            let X := F.2 + t * v in ¬ (X = Y)) := sorry

end fox_and_rabbit

end fox_catches_rabbit_rabbit_escapes_fox_l600_600877


namespace postcards_ways_odd_l600_600561

-- Define the graph structure and its properties
structure CityResidents (V : Type) :=
(adj : V → V → Bool) 
(even_common_friends : ∀ v1 v2 : V, ∃ n, n % 2 = 0 ∧ (adj v1 v2 = tt → v1 ≠ v2 → ∀ u : V, (adj v1 u = tt ∧ adj v2 u = tt) → adj v1 u = tt))

-- Define the conditions for sending and receiving postcards
def sends_postcard (V : Type) [Fintype V] (adj : V → V → Bool) (v : V) : Bool :=
if card (filter (adj v) (@Finset.univ V _)) % 2 = 1 then tt else ff

def receives_postcard (V : Type) [Fintype V] (send : V → Bool) (v : V) : Bool :=
∃ u : V, send u = tt ∧ adj u v = tt

-- Mathematical Proof Problem: Prove that the number of postcard sending ways is odd.
theorem postcards_ways_odd (V : Type) [Fintype V] (G : CityResidents V) :
  ∃ !f : (V → Bool), ((∃ v : V, sends_postcard V G.adj v = tt) ∧ 
                      (∀ v : V, receives_postcard V (sends_postcard V G.adj) v → 
                                card (filter (receives_postcard V (sends_postcard V G.adj)) (@Finset.univ V _)) <= 1)) :=
sorry

end postcards_ways_odd_l600_600561


namespace smallest_value_is_nine_l600_600201

noncomputable def smallest_possible_value (a b c d : ℝ) : ℝ :=
  (⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ : ℝ)

theorem smallest_value_is_nine {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_possible_value a b c d = 9 :=
sorry

end smallest_value_is_nine_l600_600201


namespace solve_quadratic_equation_l600_600939

noncomputable def f (x : ℝ) := 
  5 / (Real.sqrt (x - 9) - 8) - 
  2 / (Real.sqrt (x - 9) - 5) + 
  6 / (Real.sqrt (x - 9) + 5) - 
  9 / (Real.sqrt (x - 9) + 8)

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x ≥ 9 → f x = 0 → 
  x = 19.2917 ∨ x = 8.9167 :=
by sorry

end solve_quadratic_equation_l600_600939


namespace eval_ff_ff_3_l600_600260

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem eval_ff_ff_3 : f (f (f (f 3))) = 8 :=
  sorry

end eval_ff_ff_3_l600_600260


namespace find_number_l600_600845

theorem find_number (x : ℕ) (h : (x / 5) - 154 = 6) : x = 800 := by
  sorry

end find_number_l600_600845


namespace sqrt_expr_simplification_l600_600512

theorem sqrt_expr_simplification :
  (real.sqrt 27) / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - (6 * real.sqrt 2) = (6 * real.sqrt 2) :=
by
  sorry

end sqrt_expr_simplification_l600_600512


namespace hexagon_concyclic_l600_600231

-- Define the points and their geometric relationships
variables {A B C D E F P Q R : EuclideanGeometry.Point}

-- Define the conditions given in the problem
axiom convex_hexagon : EuclideanGeometry.ConvexHexagon A B C D E F
axiom diagonals_equal : EuclideanGeometry.distance A D = EuclideanGeometry.distance B E 
                        ∧ EuclideanGeometry.distance B E = EuclideanGeometry.distance C F
axiom intersect_P : EuclideanGeometry.PointOnLineIntersection P A D ∧ EuclideanGeometry.PointOnLineIntersection P C F
axiom intersect_R : EuclideanGeometry.PointOnLineIntersection R B E ∧ EuclideanGeometry.PointOnLineIntersection R C F
axiom intersect_Q : EuclideanGeometry.PointOnLineIntersection Q A D ∧ EuclideanGeometry.PointOnLineIntersection Q B E
axiom AP_eq_PF : EuclideanGeometry.distance A P = EuclideanGeometry.distance P F
axiom BR_eq_CR : EuclideanGeometry.distance B R = EuclideanGeometry.distance R C
axiom DQ_eq_EQ : EuclideanGeometry.distance D Q = EuclideanGeometry.distance Q E

-- The theorem to prove that A, B, C, D, E, and F are concyclic
theorem hexagon_concyclic : EuclideanGeometry.Concyclic A B C D E F :=
by
  -- Proof omitted
  sorry

end hexagon_concyclic_l600_600231


namespace length_of_DB_l600_600232

theorem length_of_DB 
  (α β : ℝ)
  (AC AD : ℝ)
  (right_ABC : α = π / 2)
  (right_ADB : β = π / 2)
  (AC_val : AC = 21)
  (AD_val : AD = 6) : 
  ∃ (DB : ℝ), DB = 3 * sqrt 10 :=
by
  sorry

end length_of_DB_l600_600232


namespace binomial_minus_floor_divisible_by_seven_l600_600197

theorem binomial_minus_floor_divisible_by_seven (n : ℕ) (h : n > 7) :
  ((Nat.choose n 7 : ℤ) - ⌊(n : ℤ) / 7⌋) % 7 = 0 :=
  sorry

end binomial_minus_floor_divisible_by_seven_l600_600197


namespace correct_answer_l600_600474

theorem correct_answer : 
  let sum := 53 + 28 in
  let result := sum - 5 in
  roundToNearestTen result = 80 :=
by
  admit

def roundToNearestTen (n : Int) : Int :=
  if n % 10 >= 5 then 
    n + (10 - n % 10)
  else 
    n - (n % 10)

end correct_answer_l600_600474


namespace factorize_x_cubed_minus_9x_l600_600931

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l600_600931


namespace tetrahedron_intersection_area_l600_600451

theorem tetrahedron_intersection_area 
  (side_length : ℝ)
  (height : ℝ)
  (side_length_eq : side_length = 2)
  (height_eq : height = 1)
  :
  ∃ (a b c : ℕ) (area : ℝ), area = (2 * (sqrt 3) - 3) / 4 ∧ a + b + c = 9 := 
  by 
    use [2, 3, 4, (2 * (sqrt 3) - 3) / 4]
    simp
    sorry

end tetrahedron_intersection_area_l600_600451


namespace gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l600_600057

theorem gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1 :
  Int.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
  -- proof goes here
  sorry

end gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l600_600057


namespace simplify_expression_l600_600305

theorem simplify_expression (α : ℝ) :
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) - (2 * Real.cos (2 * α)^2 - 1)) /
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) + (2 * Real.cos (2 * α)^2 - 1)) = Real.tan (2 * α) :=
by
  sorry

end simplify_expression_l600_600305


namespace arrangment_ways_basil_tomato_l600_600045

theorem arrangment_ways_basil_tomato (basil_plants tomato_plants : Finset ℕ) 
  (hb : basil_plants.card = 5) 
  (ht : tomato_plants.card = 3) 
  : (∃ total_ways : ℕ, total_ways = 4320) :=
by
  sorry

end arrangment_ways_basil_tomato_l600_600045


namespace rectangular_to_spherical_l600_600535

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem rectangular_to_spherical :
  let (x, y, z) := (3 : ℝ, 4 : ℝ, -5 : ℝ)
  let rho := sqrt (x^2 + y^2 + z^2)
  let phi := Real.arccos (z / rho)
  let theta := Real.arccos (x / (rho * Real.sin phi))
  (rho, theta, phi) = (sqrt 50, 2 * Real.pi / 3, 3 * Real.pi / 4) := 
by
  let x := (3 : ℝ)
  let y := (4 : ℝ)
  let z := (-5 : ℝ)
  let rho := sqrt (x^2 + y^2 + z^2)
  let phi := Real.arccos (z / rho)
  let theta := Real.arccos (x / (rho * Real.sin phi))
  have hrho : rho = sqrt 50 := sorry
  have hphi : phi = 3 * Real.pi / 4 := sorry
  have htheta : theta = 2 * Real.pi / 3 := sorry
  show (rho, theta, phi) = (sqrt 50, 2 * Real.pi / 3, 3 * Real.pi / 4) from sorry

end rectangular_to_spherical_l600_600535


namespace find_k_value_l600_600578

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 :=
by
  sorry

end find_k_value_l600_600578


namespace max_distance_from_M_to_AB_l600_600968

noncomputable def parabola (y : ℝ) : ℝ := y^2

def point (x y : ℝ) := (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem max_distance_from_M_to_AB
  (M : ℝ × ℝ := (1, -1))
  (A B : ℝ × ℝ)
  (hA : A.1 = parabola A.2)
  (hB : B.1 = parabola B.2)
  (dot_product_zero : (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0)
  (line_AB : ℝ → ℝ × ℝ)
  (hline : ∃ (m n : ℝ), line_AB = λ y, (m * y + n, y) ∧ (line_AB A.2 = A) ∧ (line_AB B.2 = B) )
  : ∃ P : ℝ × ℝ, distance M P = real.sqrt 5 :=
begin
  sorry
end

end max_distance_from_M_to_AB_l600_600968


namespace range_of_a_l600_600878

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_of_a (a : ℝ) (h1 : ∀ x, -1 ≤ x ∧ x ≤ 1 → f x ≤ f (x + 1))
(h2 : ∀ x, f (-x) = - f x) (cond : f (-a + 1) + f (4 * a - 5) > 0) :
  4 / 3 < a ∧ a ≤ 3 / 2 :=
begin
  sorry
end

end range_of_a_l600_600878


namespace dave_apps_added_l600_600065

theorem dave_apps_added (initial_apps : ℕ) (total_apps_after_adding : ℕ) (apps_added : ℕ) 
  (h1 : initial_apps = 17) (h2 : total_apps_after_adding = 18) 
  (h3 : total_apps_after_adding = initial_apps + apps_added) : 
  apps_added = 1 := 
by
  -- proof omitted
  sorry

end dave_apps_added_l600_600065


namespace find_multiplier_n_l600_600207

variable (x y n : ℝ)

theorem find_multiplier_n (h1 : 5 * x = n * y) 
  (h2 : x * y ≠ 0) 
  (h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998) : 
  n = 6 := 
by
  sorry

end find_multiplier_n_l600_600207


namespace probability_third_smallest_is_4_l600_600587

theorem probability_third_smallest_is_4 :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} in
  let n := 12 in
  let k := 8 in
  let t := 4 in
  ∃ p : ℚ, 
    (S.card = n ∧ k < n ∧ t ∈ S ∧ t > 1 ∧ t < n) 
    → p = 56 / 165 := sorry

end probability_third_smallest_is_4_l600_600587


namespace coin_finding_moves_l600_600852

theorem coin_finding_moves (n : ℕ) (h1 : n = 100)
  (h2 : ∀ m < n, (flip_thimbles m = true ↔ (coin_position = m)))
  (h3 : ∀ k (coin_position : ℕ), coin_position ≤ k ∧ coin_position ≥ 0 → coin_position + 1 = (coin_position + 1) % n ∨ coin_position - 1 = (coin_position + n - 1) % n) :
  ∃ moves, moves = 33 :=
by {
  sorry
}

end coin_finding_moves_l600_600852


namespace optimal_testing_strategy_l600_600554

noncomputable section

open Classical

variables (p : ℝ) (n : ℕ) (X : ℕ → ℝ) (Y : ℕ → ℝ)

-- Given conditions
def probability_of_disease := 0.1
def number_of_tests_individual := 4

-- Expected values for mixed samples plan
def probability_mixed_negative := 0.656
def probability_x_6 := 0.1 * 0.9
def probability_x_2 := 0.8 * 0.81
def expected_tests_individual := 4
def expected_tests_mixed := 2.376

-- Theorem to be proved
theorem optimal_testing_strategy :
  expected_tests_mixed < expected_tests_individual :=
sorry

end optimal_testing_strategy_l600_600554


namespace find_least_positive_theta_l600_600086

theorem find_least_positive_theta :
    ∃ θ, θ > 0 ∧ θ < 180 ∧ (sin (15 * Real.pi / 180) = cos (40 * Real.pi / 180) + cos (θ * Real.pi / 180)) ∧ θ = 115 :=
by sorry

end find_least_positive_theta_l600_600086


namespace men_women_dance_at_least_n_dancing_pairs_l600_600770

theorem men_women_dance_at_least_n_dancing_pairs 
  (n : ℕ) (h_n_pos : 0 < n) :
  ∃ (dance_schedule : ℕ → fin 2n → fin 2n), 
    (∀ t, ∑ i in finset.range n, if (is_man ((dance_schedule t i).val % (2 * n))) ∧ (is_woman i.val) ∨ (is_woman ((dance_schedule t i).val % (2 * n))) ∧ (is_man i.val) then 1 else 0) ≥ n := 
sorry

end men_women_dance_at_least_n_dancing_pairs_l600_600770


namespace quadratic_expression_l600_600332

theorem quadratic_expression (a b c : ℤ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 1) : 2 * a - b + c = 2 :=
by
  rw [h1, h2, h3]
  calc
    2 * 2 - 3 + 1 = 4 - 3 + 1 : by norm_num
    ... = 2 : by norm_num

end quadratic_expression_l600_600332


namespace product_remainder_equivalence_l600_600274

noncomputable def k : ℕ := sorry -- Assume k is a positive integer
noncomputable def p : ℕ := 8 * k + 5 -- Prime number p

-- Assume p is a prime number
axiom p_prime : Nat.Prime p

-- Assume the integers r_1, r_2, ..., r_{2k+1}
noncomputable def r : fin (2 * k + 2) → ℕ := sorry -- Define r as a function

-- The main theorem to be proved
theorem product_remainder_equivalence :
  (∏ (i j : fin (2 * k + 2)) (h : i < j), (r i)^4 + (r j)^4) % p = (-1 : ℤ)^((k * (k + 1)) / 2) % p :=
sorry

end product_remainder_equivalence_l600_600274


namespace largest_divisor_of_expression_l600_600071

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 + n^3 - n - 1) :=
sorry

end largest_divisor_of_expression_l600_600071


namespace correct_relationship_in_triangle_l600_600689

theorem correct_relationship_in_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A + B) = Real.sin C :=
sorry

end correct_relationship_in_triangle_l600_600689


namespace circle_equation_l600_600735

theorem circle_equation :
  let center := (-2, 3)
  let r : ℝ := 2
  let equation := (x : ℝ) (y : ℝ) => (x + 2)^2 + (y - 3)^2 = r^2
  in equation x y ↔ (x + 2)^2 + (y - 3)^2 = 4
:= sorry

end circle_equation_l600_600735


namespace find_angle_CEM_l600_600293

-- Define the points and circles
variables {A B C M D E : Type*}
variables [midpoint M A B]
variables [circle ω1 : Type*] [passes_through ω1 A] [passes_through ω1 M] [tangent_to_line ω1 AC]
variables [circle ω2 : Type*] [passes_through ω2 B] [passes_through ω2 M] [tangent_to_line ω2 BC]
variables [intersects ω1 ω2 D]
variables [symmetric E D AB]

-- Main theorem statement
theorem find_angle_CEM : ∀ (A B C M D E : Type*)
  [midpoint M A B]
  [circle ω1 : Type*] [passes_through ω1 A] [passes_through ω1 M] [tangent_to_line ω1 AC]
  [circle ω2 : Type*] [passes_through ω2 B] [passes_through ω2 M] [tangent_to_line ω2 BC]
  [intersects ω1 ω2 D]
  [symmetric E D AB],
  angle C E M = 180 :=
sorry

end find_angle_CEM_l600_600293


namespace eventually_repeating_seq_count_correct_calculator_result_positive_difference_perfect_squares_l600_600414

-- Definition of eventually repeating sequence
def eventually_repeating_seq (x : ℝ) : Prop :=
  ∃ n m : ℕ, ∀ k : ℕ, k ≥ m → x ^ (n + k) = x ^ k

-- Number of real numbers x such that the sequence eventually repeats
theorem eventually_repeating_seq_count : 
  {x : ℝ | eventually_repeating_seq x}.to_finset.card = 3 :=
sorry

-- Given T = 3 and k such that k^3 + k = 30, prove k^2 = 9
theorem correct_calculator_result : 
  ∃ k : ℝ, (k^3 + k = 30 ∧ k ^ 2 = 9) :=
sorry

-- Given T = 3, prove the positive difference between the largest and smallest perfect squares
theorem positive_difference_perfect_squares :
  let T := 3
  in ∃ max_sq min_sq : ℕ, 
      max_sq - min_sq = 9 ∧
      max_sq = (max { x^2 + y^2 | (x y : ℕ), ⌈√T⌉ ≤ x ∧ x ≤ T ∧ ⌈√T⌉ ≤ y ∧ y ≤ T }) ∧
      min_sq = (min { x^2 + y^2 | (x y : ℕ), ⌈√T⌉ ≤ x ∧ x ≤ T ∧ ⌈√T⌉ ≤ y ∧ y ≤ T }) :=
sorry

end eventually_repeating_seq_count_correct_calculator_result_positive_difference_perfect_squares_l600_600414


namespace scientific_notation_correct_l600_600724

-- Definitions and conditions
def million := 1000000
def number := 748 * million

-- Theorem statement
theorem scientific_notation_correct :
  (748 * million = 7.5 * 10^8) :=
sorry

end scientific_notation_correct_l600_600724


namespace simplify_expression_l600_600484

-- Define the expression to be simplified
def expression : ℝ := (sqrt 27 / (sqrt 3 / 2)) * (2 * sqrt 2) - 6 * sqrt 2

-- State the theorem to be proven
theorem simplify_expression : expression = 6 * sqrt 2 :=
by
  sorry

end simplify_expression_l600_600484


namespace locus_of_center_is_ellipse_l600_600919

theorem locus_of_center_is_ellipse (x y : ℝ) :
  let O1 : ℝ × ℝ := (-3, 0)
  let O2 : ℝ × ℝ := (3, 0)
  let R1 : ℝ := 2
  let R2 : ℝ := 10
  let cart_eq := (O1.1 + R1) + (O2.1 - R2) = 12
  in (∀ (M : ℝ × ℝ), dist (M, O1) + dist (M, O2) = 12 
       → ∃ (a b : ℝ), ∃ (e : ℝ), 0 < e < 1 ∧ ellipse_prop a b e ) :=
sorry

end locus_of_center_is_ellipse_l600_600919


namespace largest_four_digit_number_l600_600570

def is_four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N ≤ 9999

def sum_of_digits (N : ℕ) : ℕ :=
  let a := N / 1000
  let b := (N % 1000) / 100
  let c := (N % 100) / 10
  let d := N % 10
  a + b + c + d

def is_divisible (N S : ℕ) : Prop := N % S = 0

theorem largest_four_digit_number :
  ∃ N : ℕ, is_four_digit_number N ∧ is_divisible N (sum_of_digits N) ∧
  (∀ M : ℕ, is_four_digit_number M ∧ is_divisible M (sum_of_digits M) → N ≥ M) ∧ N = 9990 :=
by
  sorry

end largest_four_digit_number_l600_600570


namespace radius_of_circle_l600_600756

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = Real.pi * r ^ 2) : r = 6 := 
by 
  sorry

end radius_of_circle_l600_600756


namespace part1_b1_b2_part1_general_formula_part2_sum_20_l600_600140

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then a n + 1 else a n + 2

def b (n : ℕ) : ℕ := a (2 * n)

-- Proving b_1 = 2 and b_2 = 5
theorem part1_b1_b2 : b 1 = 2 ∧ b 2 = 5 :=
by {
  unfold b a,
  simp,
  split,
  {
    rfl -- proof for b_1 = 2
  },
  {
    rfl -- proof for b_2 = 5
  }
}

-- Proving the general formula for b_n
theorem part1_general_formula (n : ℕ) : b n = 3 * n - 1 :=
by {
  induction n with k ih,
  {
    unfold b a,
    simp,
    rfl
  },
  {
    rename ih ih_k,
    unfold b a,
    simp,
    rw [ih_k],
    calc 3 * (k + 1) - 1 = 3 * k + 3 - 1 : by ring
                       ... = 3 * k + 2     : by ring
                       ... = a (2 * k + 2) : by sorry -- Detailed proof needed
  }
}

-- Proving the sum of the first 20 terms of the sequence a_n
theorem part2_sum_20 : (Finset.range 20).sum a = 300 :=
by {
  unfold a,
  have h1 : finset.sum (finset.range 10) (λ n, 3 * n + 1) = 145,
  {
    sorry -- Compute sum of odd terms
  },
  have h2 : finset.sum (finset.range 10) (λ n, 3 * n + 2) = 155,
  {
    sorry -- Compute sum of even terms
  },
  have h3 : finset.sum (finset.range 20) a = 145 + 155,
  {
    sorry -- Combine sums
  },
  exact h3,
}

end part1_b1_b2_part1_general_formula_part2_sum_20_l600_600140


namespace Walter_gets_49_bananas_l600_600243

variable (Jefferson_bananas : ℕ) (Walter_bananas : ℕ) (total_bananas : ℕ) (shared_bananas : ℕ)

def problem_conditions : Prop :=
  Jefferson_bananas = 56 ∧ Walter_bananas = Jefferson_bananas - (Jefferson_bananas / 4)

theorem Walter_gets_49_bananas (h : problem_conditions) : 
  let combined_bananas := Jefferson_bananas + Walter_bananas in
  let shared_bananas := combined_bananas / 2 in
  shared_bananas = 49 :=
by
  sorry

end Walter_gets_49_bananas_l600_600243


namespace circle_radius_l600_600751

theorem circle_radius (r : ℝ) (h_circumference : 2 * Real.pi * r) 
                      (h_area : Real.pi * r^2) 
                      (h_equation : 3 * (2 * Real.pi * r) = Real.pi * r^2) : 
                      r = 6 :=
by
  sorry

end circle_radius_l600_600751


namespace b_n_formula_sum_first_20_terms_l600_600129

-- Definition of sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a n + 1 else a n + 2

-- Definition of sequence b_n as a 2n-th term of a_n
def b (n : ℕ) : ℕ := a (2 * n)

-- Proof problem 1: General formula for b_n
theorem b_n_formula (n : ℕ) : b n = 3 * n - 1 :=
sorry

-- Sum of the first 20 terms of the sequence a_n
theorem sum_first_20_terms : (Finset.range 20).sum a = 300 :=
sorry

end b_n_formula_sum_first_20_terms_l600_600129


namespace ball_bounces_to_less_than_two_feet_l600_600416

noncomputable def bounce_height (n : ℕ) : ℝ := 20 * (3 / 4) ^ n

theorem ball_bounces_to_less_than_two_feet : ∃ k : ℕ, bounce_height k < 2 ∧ k = 7 :=
by
  -- We need to show that bounce_height k < 2 when k = 7
  sorry

end ball_bounces_to_less_than_two_feet_l600_600416


namespace meet_without_contact_probability_l600_600785

noncomputable def prob_meet_without_contact : ℝ :=
  let total_area := 1
  let outside_area := (1 / 8) * 2
  total_area - outside_area

theorem meet_without_contact_probability :
  prob_meet_without_contact = 3 / 4 :=
by
  sorry

end meet_without_contact_probability_l600_600785


namespace proof_problem_l600_600595

noncomputable def f : ℝ → ℝ := sorry

-- Conditions as definitions
def is_odd_function (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def is_even_shifted_function (f : ℝ → ℝ) := ∀ x, f(-x + 1) = f(x + 1)

def condition_1 := is_odd_function f
def condition_2 := is_even_shifted_function f
def condition_3 := f 1 = 1

-- The proof we need to formulate
theorem proof_problem : condition_1 ∧ condition_2 ∧ condition_3 → f 3 + f 4 = -1 := by
  intros h
  sorry

end proof_problem_l600_600595


namespace diagonals_of_convex_quadrilateral_are_perpendicular_l600_600297

noncomputable def diagonals_are_perpendicular 
  (A B C D : Type) [convex_quadrilateral A B C D]
  (incircle_tangent_to_sides : tangent_circle_to_sides A B C D)
  (excircle_tangent_to_extensions : tangent_circle_to_extensions A B C D) : Prop :=
  perpendicular (diagonal A C) (diagonal B D)

theorem diagonals_of_convex_quadrilateral_are_perpendicular
  (A B C D : Type) [convex_quadrilateral A B C D]
  (incircle_tangent_to_sides : tangent_circle_to_sides A B C D)
  (excircle_tangent_to_extensions : tangent_circle_to_extensions A B C D) :
  diagonals_are_perpendicular A B C D incircle_tangent_to_sides excircle_tangent_to_extensions :=
  sorry

end diagonals_of_convex_quadrilateral_are_perpendicular_l600_600297


namespace calculate_expression_l600_600496

theorem calculate_expression :
  (real.sqrt 27 / (real.sqrt 3 / 2) * (2 * real.sqrt 2) - 6 * real.sqrt 2 = 6 * real.sqrt 2) :=
by
  sorry

end calculate_expression_l600_600496


namespace circle_diameter_length_l600_600426

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l600_600426


namespace set_B_listing_method_l600_600114

variable (A : Set ℕ) (B : Set ℕ)

theorem set_B_listing_method (hA : A = {1, 2, 3}) (hB : B = {x | x ∈ A}) :
  B = {1, 2, 3} :=
  by
    sorry

end set_B_listing_method_l600_600114


namespace circle_diameter_l600_600431

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l600_600431


namespace range_of_a_in_log_inequality_l600_600658

theorem range_of_a_in_log_inequality (a : ℝ) :
  (∀ x : ℝ, real.logb 2 (|x + 1| + |x - 7|) ≥ a) → a ≤ 3 :=
begin
  sorry
end

end range_of_a_in_log_inequality_l600_600658


namespace decimal_zeroes_between_decimal_point_and_first_nonzero_digit_l600_600790

theorem decimal_zeroes_between_decimal_point_and_first_nonzero_digit :
  (number_of_zeroes_between_decimal_point_and_first_nonzero (5 / 480) = 1) :=
sorry

end decimal_zeroes_between_decimal_point_and_first_nonzero_digit_l600_600790


namespace probability_of_x_less_than_2y_in_rectangle_l600_600018

noncomputable def area_trapezoid : ℝ := 0.5 * (6 + 4) * 2
noncomputable def area_rectangle : ℝ := 6 * 2

theorem probability_of_x_less_than_2y_in_rectangle :
  let probability : ℝ := area_trapezoid / area_rectangle in
  probability = 5 / 6 :=
by
  sorry

end probability_of_x_less_than_2y_in_rectangle_l600_600018
