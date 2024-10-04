import Mathlib
import Mathlib.Algebra.ArithmeticSeq
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Seq
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Pi
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle
import Mathlib.Geometry.Ellipse
import Mathlib.Geometry.Euclidean.Angle
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Geometry.Hyperbola
import Mathlib.Geometry.Triangle.Basic
import Mathlib.Geometry.Triangle.Congruence
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.ProbDistrib.Normal
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real
import Mathlib.Topology.MetricSpace.Basic
import data.real.basic

namespace min_value_of_a_l474_474189

-- Defining the properties of the function f
variable {f : ℝ → ℝ}
variable (even_f : ∀ x, f x = f (-x))
variable (mono_f : ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Necessary condition involving f and a
variable {a : ℝ}
variable (a_condition : f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1)

-- Main statement proving that the minimum value of a is 1/2
theorem min_value_of_a : a = 1/2 :=
sorry

end min_value_of_a_l474_474189


namespace area_bound_by_curves_l474_474808

open Real

noncomputable def area_of_region : ℝ :=
  ∫ (y in (1 : ℝ)..exp 3), 1 / (y * sqrt (1 + log y))

theorem area_bound_by_curves :
  area_of_region = 2 :=
by
  sorry

end area_bound_by_curves_l474_474808


namespace rectangles_cover_circle_l474_474504

theorem rectangles_cover_circle (rectangles : list ℝ) (h1 : ∀ x ∈ rectangles, x > 0) (h2 : rectangles.sum = 16) : 
  ∃ translations : list ℝ × list ℝ, ∀ p : ℝ × ℝ, dist p (0, 0) <= 1 → ∃ r ∈ rectangles, (p.fst + translations.1.head * r = 0 ∨ p.snd + translations.2.head * r = 0) :=
sorry

end rectangles_cover_circle_l474_474504


namespace largest_value_of_x_l474_474783

theorem largest_value_of_x : 
  ∃ x, (fractional_eq_condition x) ∧ (x = (63 + Real.sqrt 2457) / 54) := 
by
  let fractional_eq_condition (x : ℝ) : Prop := (3 * x / 7 + 2 / (9 * x) = 1)
  sorry

end largest_value_of_x_l474_474783


namespace solution_when_a_is_1_solution_for_arbitrary_a_l474_474526

-- Let's define the inequality and the solution sets
def inequality (a x : ℝ) : Prop :=
  ((a + 1) * x - 3) / (x - 1) < 1

def solutionSet_a_eq_1 (x : ℝ) : Prop :=
  1 < x ∧ x < 2

def solutionSet_a_eq_0 (x : ℝ) : Prop :=
  1 < x
  
def solutionSet_a_lt_0 (a x : ℝ) : Prop :=
  x < (2 / a) ∨ 1 < x

def solutionSet_0_lt_a_lt_2 (a x : ℝ) : Prop :=
  1 < x ∧ x < (2 / a)

def solutionSet_a_eq_2 : Prop :=
  false

def solutionSet_a_gt_2 (a x : ℝ) : Prop :=
  (2 / a) < x ∧ x < 1

-- Prove the solution for a = 1
theorem solution_when_a_is_1 : ∀ (x : ℝ), inequality 1 x ↔ solutionSet_a_eq_1 x :=
by sorry

-- Prove the solution for arbitrary real number a
theorem solution_for_arbitrary_a : ∀ (a x : ℝ),
  (a < 0 → inequality a x ↔ solutionSet_a_lt_0 a x) ∧
  (a = 0 → inequality a x ↔ solutionSet_a_eq_0 x) ∧
  (0 < a ∧ a < 2 → inequality a x ↔ solutionSet_0_lt_a_lt_2 a x) ∧
  (a = 2 → inequality a x → solutionSet_a_eq_2) ∧
  (a > 2 → inequality a x ↔ solutionSet_a_gt_2 a x) :=
by sorry

end solution_when_a_is_1_solution_for_arbitrary_a_l474_474526


namespace angle_MKF_equals_45_degrees_l474_474178

def parabola_focus : (ℝ × ℝ) := (1 / 2, 0)  -- Focus of y^2 = 4x
def parabola_directrix : ℝ → Prop := λ x, x = -1  -- Directrix of y^2 = 4x

noncomputable def point_on_parabola : (ℝ × ℝ) := (1, 2)  -- Given a point on the parabola y^2 = 4x
noncomputable def intersection_directrix_x_axis : (ℝ × ℝ) := (-1, 0)  -- The intersection point of the directrix and x-axis

theorem angle_MKF_equals_45_degrees :
  let M := point_on_parabola,
      F := parabola_focus,
      K := intersection_directrix_x_axis in
  |F.1 - M.1| + |F.2 - M.2| = 2 →  -- Condition: |MF| = 2
  angle (M, K, F) = 45 :=  -- Conclusion: angle MKF = 45°
sorry

end angle_MKF_equals_45_degrees_l474_474178


namespace express_polynomial_in_form_l474_474899

theorem express_polynomial_in_form (x : ℝ) :
  ∃ a h k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 :=
by
  use [3, -3/2, 53/4]
  sorry

end express_polynomial_in_form_l474_474899


namespace prove_a_squared_minus_b_eq_zero_l474_474300

variable {a b : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : {1, a + b, a} = {0, b / a, b})

theorem prove_a_squared_minus_b_eq_zero (a b : ℝ) (h₁ : a ≠ 0) (h₂ : {1, a + b, a} = {0, b / a, b}) : a^2 - b = 0 :=
sorry

end prove_a_squared_minus_b_eq_zero_l474_474300


namespace mike_winning_strategy_min_k_l474_474695

-- Definition of the game conditions and the goal statement.
theorem mike_winning_strategy_min_k : 
  ∃ k : ℕ, 
    (k = 16) ∧ (
      ∀ (board : matrix (fin 8) (fin 8) char), 
      (∀ s ∈ finset.fin_range (8 * 8), s < k → (board s / 2 = 'M')) →
      (∀ t ∈ finset.fin_range (8 * 8), k ≤ t < k+1 → (board.t t = 'H')) →
      (∃ i j h, (board (i, j) = 'H') ∨ (board (i, h) = 'M') ∨ (board (i, j+1) = 'M') ∨ (board (i+1, j) = 'M') ∨
      (board (i+2, j) = 'h') ∨ (board (i+1, j+1) = 'm') ∨ (board (i+1, j+2) = 'm') ∨
      (board (i+2, j+2) = 'M'))
     sorry

end mike_winning_strategy_min_k_l474_474695


namespace greatest_integer_gcd_l474_474774

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l474_474774


namespace log_prob_greater_than_one_is_half_l474_474486

-- Define the set of numbers
def S : Set ℕ := {2, 3, 4}

-- Define that we are selecting two numbers randomly from S
def selected_pairs : Finset (ℕ × ℕ) := Finset.product ({2, 3, 4} : Finset ℕ) ({2, 3, 4} : Finset ℕ)

-- Define that m and n are not equal to each other in selected pairs
def valid_pairs : Finset (ℕ × ℕ) := selected_pairs.filter (λ p, p.1 ≠ p.2)

-- Define the function to check if the logarithm value is greater than 1
def log_greater_than_one (m n : ℕ) : Prop := (log m.toReal n.toReal > 1)

noncomputable def probability_log_greater_than_one : ℚ :=
  (valid_pairs.filter (λ p, log_greater_than_one p.1 p.2)).card / valid_pairs.card

-- Theorem statement
theorem log_prob_greater_than_one_is_half : probability_log_greater_than_one = 1 / 2 :=
by sorry

end log_prob_greater_than_one_is_half_l474_474486


namespace work_done_in_one_day_l474_474802

theorem work_done_in_one_day (A_days B_days : ℝ) (hA : A_days = 6) (hB : B_days = A_days / 2) : 
  (1 / A_days + 1 / B_days) = 1 / 2 := by
  sorry

end work_done_in_one_day_l474_474802


namespace gcd_of_powers_of_two_minus_one_l474_474761

theorem gcd_of_powers_of_two_minus_one : 
  gcd (2^1015 - 1) (2^1020 - 1) = 1 :=
sorry

end gcd_of_powers_of_two_minus_one_l474_474761


namespace potato_bag_weight_l474_474801

-- Defining the weight of the bag of potatoes as a variable W
variable (W : ℝ)

-- Given condition: The weight of the bag is described by the equation
def weight_condition (W : ℝ) := W = 12 / (W / 2)

-- Proving the weight of the bag of potatoes is 12 lbs:
theorem potato_bag_weight : weight_condition W → W = 12 :=
by
  sorry

end potato_bag_weight_l474_474801


namespace find_slope_l474_474497

theorem find_slope (k b x y y2 : ℝ) (h1 : y = k * x + b) (h2 : y2 = k * (x + 3) + b) (h3 : y2 - y = -2) : k = -2 / 3 := by
  sorry

end find_slope_l474_474497


namespace cross_product_and_magnitude_l474_474903

-- Definitions of Given Vectors
def vector1 : ℝ × ℝ × ℝ := (3, 1, 4)
def vector2 : ℝ × ℝ × ℝ := (2, -3, 6)

-- Definition of the cross product of two vectors
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- Definition of the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Problem Statement
theorem cross_product_and_magnitude :
  cross_product vector1 vector2 = (18, -10, -11) ∧ magnitude (cross_product vector1 vector2) = Real.sqrt 545 :=
by
  sorry

end cross_product_and_magnitude_l474_474903


namespace smallest_integer_in_set_l474_474589

theorem smallest_integer_in_set (n : ℤ) (h : ∃ k: ℤ, k = n + 8 ∧ k < 3 * (n + 4)) : n = 0 := 
by
  have h1 : n + 8 < 3 * (n + 4) := sorry
  have h2 : 8 < 2 * n + 12 := sorry
  have h3 : -4 < 2 * n := sorry
  have h4 : -2 < n := sorry
  have h5 : n ∈ {x : ℤ | x.even ∧ x > -2} := sorry
  -- n must be the smallest even integer greater than -2, which is 0
  sorry

end smallest_integer_in_set_l474_474589


namespace smallest_positive_solution_l474_474139

theorem smallest_positive_solution :
  ∃ (x : ℝ), (0 < x) ∧ (tan (4 * x) + tan (6 * x) = (1 / cos (6 * x)) + 1) ∧ x = π / 28 :=
by
  sorry

end smallest_positive_solution_l474_474139


namespace Ilya_defeats_dragon_l474_474244

-- Conditions
def prob_two_heads : ℚ := 1 / 4
def prob_one_head : ℚ := 1 / 3
def prob_no_heads : ℚ := 5 / 12

-- Main statement in Lean
theorem Ilya_defeats_dragon : 
  (prob_no_heads + prob_one_head + prob_two_heads = 1) → 
  (∀ n : ℕ, ∃ m : ℕ, m ≤ n) → 
  (∑ n, (prob_no_heads + prob_one_head + prob_two_heads) ^ n) = 1 := 
sorry

end Ilya_defeats_dragon_l474_474244


namespace seven_day_of_month_is_Thursday_l474_474258

-- Define prime number check
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem conditions and goal
theorem seven_day_of_month_is_Thursday 
  (months_days : ℕ)
  (h1: months_days = 30 ∨ months_days = 31)
  (sundays : list ℕ)
  (h2: sundays.length = 4 ∨ sundays.length = 5)
  (h3 : ∃ (a b c : ℕ), [a, b, c].all is_prime ∧ [a, b, c].all (λ x, x ∈ sundays)) :
  (nth_weekday_of_month 7 = weekday.Thursday) :=
sorry

end seven_day_of_month_is_Thursday_l474_474258


namespace faye_earned_total_l474_474469

-- Definitions of the necklace sales
def bead_necklaces := 3
def bead_price := 7
def gemstone_necklaces := 7
def gemstone_price := 10
def pearl_necklaces := 2
def pearl_price := 12
def crystal_necklaces := 5
def crystal_price := 15

-- Total amount calculation
def total_amount := 
  bead_necklaces * bead_price + 
  gemstone_necklaces * gemstone_price + 
  pearl_necklaces * pearl_price + 
  crystal_necklaces * crystal_price

-- Proving the total amount equals $190
theorem faye_earned_total : total_amount = 190 := by
  sorry

end faye_earned_total_l474_474469


namespace purely_imaginary_z_point_on_line_z_l474_474312

variable (m : ℝ)

def z_real_part (m : ℝ) : ℝ := m^2 - 8 * m + 15
def z_imag_part (m : ℝ) : ℝ := m^2 - 5 * m - 14
def z (m : ℝ) := z_real_part m + z_imag_part m * complex.I

theorem purely_imaginary_z (m : ℝ) :
  z_real_part m = 0 ∧ z_imag_part m ≠ 0 → m = 3 ∨ m = 5 :=
by
  sorry

theorem point_on_line_z (m : ℝ) :
  (z_real_part m - z_imag_part m - 2 = 0) → m = 9 :=
by
  sorry

end purely_imaginary_z_point_on_line_z_l474_474312


namespace math_city_police_officers_needed_l474_474323

theorem math_city_police_officers_needed :
  let streets := 10 
  ∑ i in range streets, i = 45 := sorry

end math_city_police_officers_needed_l474_474323


namespace no_solution_exists_l474_474759

theorem no_solution_exists (n : ℤ) (h1 : n + 15 > 20) (h2 : -3 * n > -9) : false :=
by {
  have h3 : n > 5 := by linarith,
  have h4 : n < 3 := by linarith,
  linarith,
}

end no_solution_exists_l474_474759


namespace range_fx_when_a_is_minus_4_range_a_for_fx_eq_0_two_distinct_real_roots_l474_474198

noncomputable def fx (x : ℝ) (a : ℝ) : ℝ := 4^x + a * 2^x + 3

theorem range_fx_when_a_is_minus_4 :
  ∀ x ∈ set.Icc (0:ℝ) 2, ∃ y ∈ set.Icc (-1:ℝ) 3, fx x (-4) = y :=
sorry

theorem range_a_for_fx_eq_0_two_distinct_real_roots :
  (∀ x > 0, ∃ t > 1, 4^x + a * 2^x + 3 = 0) → a ∈ set.Ioo (-4:ℝ) (-2 * real.sqrt 3) :=
sorry

end range_fx_when_a_is_minus_4_range_a_for_fx_eq_0_two_distinct_real_roots_l474_474198


namespace salad_quantity_percentage_difference_l474_474788

noncomputable def Tom_rate := 2/3 -- Tom's rate (lb/min)
noncomputable def Tammy_rate := 3/2 -- Tammy's rate (lb/min)
noncomputable def Total_salad := 65 -- Total salad chopped (lb)
noncomputable def Time_to_chop := Total_salad / (Tom_rate + Tammy_rate) -- Time to chop 65 lb (min)
noncomputable def Tom_chop := Time_to_chop * Tom_rate -- Total chopped by Tom (lb)
noncomputable def Tammy_chop := Time_to_chop * Tammy_rate -- Total chopped by Tammy (lb)
noncomputable def Percent_difference := (Tammy_chop - Tom_chop) / Tom_chop * 100 -- Percent difference

theorem salad_quantity_percentage_difference : Percent_difference = 125 :=
by
  sorry

end salad_quantity_percentage_difference_l474_474788


namespace f_a_lt_f_b_l474_474985

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

variables (a b : ℝ)

-- Condition: e < b < a
axiom e_lt_b_lt_a : Real.exp 1 < b ∧ b < a

-- Theorem to prove: f(a) < f(b)
theorem f_a_lt_f_b (h : e_lt_b_lt_a) : f a < f b := sorry

end f_a_lt_f_b_l474_474985


namespace catalan_generating_function_l474_474676

noncomputable def C (x : ℝ) : ℝ := ∑ n : ℕ, (C_n x^n)

theorem catalan_generating_function (C_n : ℕ → ℝ) (x : ℝ) :
  (C x = x * C x * C x + 1) →
  (C x = (1 - real.sqrt(1 - 4 * x)) / (2 * x)) := sorry

end catalan_generating_function_l474_474676


namespace base5_representation_three_consecutive_digits_l474_474635

theorem base5_representation_three_consecutive_digits :
  ∃ (digits : ℕ), 
    (digits = 3) ∧ 
    (∃ (a1 a2 a3 : ℕ), 
      94 = a1 * 5^2 + a2 * 5^1 + a3 * 5^0 ∧
      a1 = 3 ∧ a2 = 3 ∧ a3 = 4 ∧
      (a1 = a3 + 1) ∧ (a2 = a3 + 2)) := 
    sorry

end base5_representation_three_consecutive_digits_l474_474635


namespace sequence_problem_l474_474373
open Real

noncomputable def sequence_a : ℕ → ℚ
| n := if n = 1 then 1/2 else (∃ k : ℕ, k ≥ 2 ∧ (k-1) * (k-1 + 1) / 2 < n ∧ n ≤ k * (k + 1) / 2) ≠ ⊥

-- Sum of the first n terms of sequence_a
noncomputable def sum_SN (n : ℕ) : ℚ :=
∑ i in Finset.range n, sequence_a (i + 1)

theorem sequence_problem :
  ∃ (k : ℕ), sum_SN k < 100 ∧ sum_SN (k + 1) ≥ 100 ∧ sequence_a k = 13 / 21 ∧ k = 203 :=
by
  sorry

end sequence_problem_l474_474373


namespace problem1_problem2_l474_474968

-- Define the sets A and B and the universal set U
def U := Set ℝ
def A : Set ℝ := { x | 2 * x - 8 < 0 }
def B : Set ℝ := { x | 0 < x ∧ x < 6 }

-- Problem (1) 
theorem problem1 : (A ∩ B) = { x | 0 < x ∧ x < 4 } := by
  sorry

-- Problem (2)
theorem problem2 : ((U \ A) ∪ B) = { x | 0 < x } := by
  sorry

end problem1_problem2_l474_474968


namespace man_speed_in_still_water_l474_474405

theorem man_speed_in_still_water (V_m V_s : ℝ) 
  (h1 : V_m + V_s = 8)
  (h2 : V_m - V_s = 6) : 
  V_m = 7 := 
by
  sorry

end man_speed_in_still_water_l474_474405


namespace sum_sin_squared_l474_474638

theorem sum_sin_squared : ∑ i in Finset.range 89, Real.sin (i+1 : ℝ * Real.pi / 180)^2 = 44.5 := 
sorry

end sum_sin_squared_l474_474638


namespace sum_of_segments_lengths_l474_474890

theorem sum_of_segments_lengths (AB : ℕ) (n : ℕ) (h : AB = 12) (h1 : n = 6) :
  (∑ i in Finset.range n, (n-i) * AB / n) = 30 := 
sorry

end sum_of_segments_lengths_l474_474890


namespace trigonometric_identity_l474_474926

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : sin α = 1 / 3)
  (h₂ : α ∈ set.Ioo (π / 2) π) :
  (cos α = -2 * real.sqrt 2 / 3) ∧ (tan α = -real.sqrt 2 / 4) :=
by
  sorry

end trigonometric_identity_l474_474926


namespace number_of_children_l474_474382

theorem number_of_children (total_people : ℕ) (num_adults num_children : ℕ)
  (h1 : total_people = 42)
  (h2 : num_children = 2 * num_adults)
  (h3 : num_adults + num_children = total_people) :
  num_children = 28 :=
by
  sorry

end number_of_children_l474_474382


namespace parabola_equation_l474_474957

theorem parabola_equation (V K : Type) [vertex V] [focus K] 
  (h1 : parabola_vertex_at_origin V) 
  (h2 : parabola_focus_is_right_hyperbola_focus K) : 
  parabola_equation = "y^2 = 20x" :=
sorry

end parabola_equation_l474_474957


namespace greatest_integer_less_than_200_with_gcd_18_l474_474770

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l474_474770


namespace budget_for_bulbs_l474_474401

theorem budget_for_bulbs (num_crocus_bulbs : ℕ) (cost_per_crocus : ℝ) (budget : ℝ)
  (h1 : num_crocus_bulbs = 22)
  (h2 : cost_per_crocus = 0.35)
  (h3 : budget = num_crocus_bulbs * cost_per_crocus) :
  budget = 7.70 :=
sorry

end budget_for_bulbs_l474_474401


namespace four_n_div_four_remainder_zero_l474_474986

theorem four_n_div_four_remainder_zero (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := 
by
  sorry

end four_n_div_four_remainder_zero_l474_474986


namespace calculate_fraction_l474_474102

variable (a b c d e f g h i j k l : ℚ)

def mixed_num_to_improper (a b : ℚ) := a * b + c

theorem calculate_fraction :
  (a = 2 ∧ b = 1/4 ∧ c = 0.25 ∧ d = 2 ∧ e = 3/4 ∧ f = 1/2 ∧ g = 2 ∧ h = 1/5 ∧ i = 2/5) →
  ((a * (4/1) + b + c) / (d * (4/1) + e - f) + ((2 * 0.5) / (g * (5/1) + h - i)) = 5/3) :=
by
  sorry

end calculate_fraction_l474_474102


namespace tan_square_value_l474_474161

theorem tan_square_value (α : ℝ) (h : cos (2 * α) = -1/9) : tan α ^ 2 = 5 / 4 :=
sorry

end tan_square_value_l474_474161


namespace simplify_expression_l474_474103

theorem simplify_expression : 
  abs (Real.sqrt 3 - 1) + Real.cbrt (-8) + Real.sqrt 9 = Real.sqrt 3 := 
by 
  sorry

end simplify_expression_l474_474103


namespace num_permutations_with_exactly_4_deranged_l474_474909

def derangement (n : ℕ) (σ : Fin n → Fin n) : Prop :=
  ∀ i, σ i ≠ i

def num_derangements (n : ℕ) : ℕ :=
  (Nat.factorial n) * (∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k / (Nat.factorial k))

def binom (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem num_permutations_with_exactly_4_deranged :
  let n := 8
  let k := 4
  (binom n k) * (num_derangements k) = 630 :=
by
  -- Specify the values
  let n := 8
  let k := 4
  -- Compute binomial coefficient and number of derangements
  have binom_8_4 : binom n k = 70 := sorry
  have derangements_4 : num_derangements k = 9 := sorry
  -- Prove the final multiplication
  show (binom n k) * (num_derangements k) = 630
  rw [binom_8_4, derangements_4]
  norm_num
  exact 70 * 9 = 630

end num_permutations_with_exactly_4_deranged_l474_474909


namespace greatest_integer_gcd_6_l474_474767

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l474_474767


namespace probability_sum_25_of_two_20_faced_dice_l474_474064

theorem probability_sum_25_of_two_20_faced_dice:
  let die1 := {i : ℕ | 1 ≤ i ∧ i ≤ 19} ∪ {0} -- First die faces
  let die2 := ({i : ℕ | (1 ≤ i ∧ i ≤ 7) ∨ (9 ≤ i ∧ i ≤ 19)} ∪ {0}) -- Second die faces
  let valid_combinations := {p : ℕ × ℕ | p.1 + p.2 = 25 ∧ p.1 ∈ die1 ∧ p.2 ∈ die2}
  let total_outcomes := 400
  let prob := valid_combinations.finite.to_finset.card.to_rat / total_outcomes
  
  prob = (13 : ℚ) / 400 :=
sorry

end probability_sum_25_of_two_20_faced_dice_l474_474064


namespace pagesReadOnThursday_l474_474891

variable (pagesWednesday pagesFriday pagesTotal : ℕ)

theorem pagesReadOnThursday 
  (h_wed : pagesWednesday = 18) 
  (h_fri : pagesFriday = 23) 
  (h_total : pagesTotal = 60) : 
  ∃ pagesThursday, pagesThursday = pagesTotal - (pagesWednesday + pagesFriday) ∧ pagesThursday = 19 := 
by
  use pagesTotal - (pagesWednesday + pagesFriday)
  rw [h_wed, h_fri, h_total]
  norm_num
  sorry

end pagesReadOnThursday_l474_474891


namespace solve_expression_l474_474883

theorem solve_expression : ∃ y : ℝ, y = 3 + 3 * Real.sqrt 3 / 2 :=
by
  let expr := 3 + 3 / (2 + 3 / (1 + 3 / (Real.factorial 2 + ...) ))
  have : ∀ y : ℝ, y = 3 + 3 / (2 + 3 / y) → y = 3 + 3 * Real.sqrt 3 / 2 := sorry
  exact ⟨_, this _ sorry⟩

end solve_expression_l474_474883


namespace find_p_l474_474352

noncomputable def f (p : ℝ) : ℝ := 2 * p - 20

theorem find_p : (f ∘ f ∘ f) p = 6 → p = 18.25 := by
  sorry

end find_p_l474_474352


namespace suraj_average_increase_l474_474341

theorem suraj_average_increase
  (A : ℝ)
  (h1 : 9 * A + 200 = 10 * 128) :
  128 - A = 8 :=
by
  sorry

end suraj_average_increase_l474_474341


namespace australian_math_competition_1989_l474_474948

noncomputable def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y * y = x
noncomputable def within_range (d : List ℕ) (n : ℕ) : Prop := ∀ (i : ℕ), i < d.length → (d[i] = 0 ∨ d[i] = 1 ∨ d[i] = 2)
noncomputable def sum_expression (d : List ℕ) : ℕ := d.enum.map (λ (nk : ℕ × ℕ), nk.snd * 3 ^ nk.fst).sum

theorem australian_math_competition_1989 (n : ℕ) (d : List ℕ)
  (h0 : 8 * 53 * n ≥ 0)
  (h1 : d.length = n + 1)
  (h2 : within_range d n)
  (h3 : is_perfect_square (sum_expression d)) :
  ∃ (i : ℕ), i ≤ n ∧ d[i] = 1 :=
by
  sorry

end australian_math_competition_1989_l474_474948


namespace find_m_n_sum_l474_474206

theorem find_m_n_sum (n m : ℝ) (d : ℝ) 
(h1 : ∀ x y, 2*x + y + n = 0) 
(h2 : ∀ x y, 4*x + m*y - 4 = 0) 
(hd : d = (3/5) * Real.sqrt 5) 
: m + n = -3 ∨ m + n = 3 :=
sorry

end find_m_n_sum_l474_474206


namespace balls_in_rightmost_box_l474_474377

theorem balls_in_rightmost_box (r b : Fin 10 → ℕ) 
  (h1 : ∑ i, r i = 13) 
  (h2 : ∑ i, b i = 11) 
  (h3 : ∀ i : Fin 9, r i.succ + b i.succ ≥ r i + b i) 
  (h4 : ∀ i j : Fin 10, i ≠ j → (r i, b i) ≠ (r j, b j)) : 
  r 9 = 3 ∧ b 9 = 1 :=
sorry

end balls_in_rightmost_box_l474_474377


namespace minimum_f_value_and_count_l474_474316

def f (a : Fin 2020 → ℕ) : ℕ :=
  (∑ i in Finset.range 2019, (a i)^2) - (∑ i in Finset.range 1008, a (2*i) * a (2*i + 2))

theorem minimum_f_value_and_count (a : Fin 2020 → ℕ) (h₁ : a 0 = 1) (h₂ : a 2019 = 99) (h₃ : ∀ i j, i ≤ j → i < 2019 → j < 2019 → a i ≤ a j):
  ∃ n : ℕ, n = f a ∧ n = 3700 ∧ ∃ k, k = Nat.choose 1968 48 := 
sorry

end minimum_f_value_and_count_l474_474316


namespace problem1_problem2_problem3_problem4_l474_474869

def problem1_statement : Prop :=
  sqrt 50 + sqrt 32 - sqrt 2 = 8 * sqrt 2

def problem2_statement : Prop :=
  sqrt (3 ^ 2) + real.cbrt (-8) - abs (1 - sqrt 2) = 2 - sqrt 2

def problem3_statement : Prop :=
  (sqrt 5 - 1) ^ 2 - (sqrt 6 - sqrt 5) * (sqrt 6 + sqrt 5) = 5 - 2 * sqrt 5

def problem4_statement : Prop :=
  (sqrt 8 + sqrt 18) / sqrt 2 + (sqrt 24 - sqrt (1 / 6)) / sqrt 3 = 5 + 11 * sqrt 2 / 6

theorem problem1 : problem1_statement :=
  by sorry

theorem problem2 : problem2_statement :=
  by sorry

theorem problem3 : problem3_statement :=
  by sorry

theorem problem4 : problem4_statement :=
  by sorry

end problem1_problem2_problem3_problem4_l474_474869


namespace range_of_f_l474_474120

def f (x : ℝ) : ℝ := √3 * sin (2 * x) + 2 * cos (x) ^ 2 - 1

theorem range_of_f : set.range f = {y | -2 ≤ y ∧ y ≤ 2} :=
sorry

end range_of_f_l474_474120


namespace volume_tetrahedron_O_l474_474755

variable {d e f : ℝ}

theorem volume_tetrahedron_O DEF : 
  d^2 + e^2 = 64 →
  e^2 + f^2 = 100 →
  f^2 + d^2 = 144 →
  (1 / 6 * real.sqrt d * real.sqrt e * real.sqrt f : ℝ) = 110 / 3 :=
by
  intros h₁ h₂ h₃
  sorry

end volume_tetrahedron_O_l474_474755


namespace minimum_sum_areas_squares_l474_474675

theorem minimum_sum_areas_squares
  (AB BC AC : ℝ) (AB_eq : AB = 5) (BC_eq : BC = 4) (AC_eq : AC = 3)
  (side_P side_Q : ℝ)
  (area_P : ℝ := side_P ^ 2) (area_Q : ℝ := side_Q ^ 2) :
  ∃ (side_P side_Q : ℝ), (side_P + side_Q + (3/4) * side_P + (4/3) * side_Q = 5) ∧
  (area_P + area_Q = 144/49) :=
begin
  sorry
end

end minimum_sum_areas_squares_l474_474675


namespace evaluate_expression_l474_474787

theorem evaluate_expression : 
    (1 / ( (-5 : ℤ) ^ 4) ^ 2 ) * (-5 : ℤ) ^ 9 = -5 :=
by sorry

end evaluate_expression_l474_474787


namespace integral_value_l474_474451

noncomputable def definite_integral_simplified : ℝ :=
  ∫ x in -π / 2 .. 0, 2^8 * (Real.sin x)^2 * (Real.cos x)^6

theorem integral_value : definite_integral_simplified = 5 * π := by
  sorry

end integral_value_l474_474451


namespace ceil_sum_sqrt_eval_l474_474897

theorem ceil_sum_sqrt_eval : 
  (⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉) = 22 := 
by
  sorry

end ceil_sum_sqrt_eval_l474_474897


namespace birds_finish_half_nuts_in_3_65_hours_l474_474053

variables (N : ℕ) -- Total number of nuts

-- Crow's rate: 1/20 nuts per hour
def crow_rate := N / 20

-- Sparrow's rate: 1/18 nuts per hour
def sparrow_rate := N / 18

-- Parrot's rate: 1/32 nuts per hour
def parrot_rate := N / 32

-- Combined rate of all birds
def combined_rate := crow_rate N + sparrow_rate N + parrot_rate N

-- Time needed to eat half the nuts together
def time_to_eat_half_nuts := (N / 2) / combined_rate N

theorem birds_finish_half_nuts_in_3_65_hours : time_to_eat_half_nuts N = 3.65 :=
  by sorry

end birds_finish_half_nuts_in_3_65_hours_l474_474053


namespace combination_8_5_l474_474620

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474620


namespace choose_five_from_eight_l474_474601

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474601


namespace sum_of_four_powers_l474_474896

theorem sum_of_four_powers (a : ℕ) : 4 * a^3 = 500 :=
by
  rw [Nat.pow_succ, Nat.pow_succ]
  sorry

end sum_of_four_powers_l474_474896


namespace triangle_XYZ_cosine_l474_474273

theorem triangle_XYZ_cosine {X Y Z : Type} [metric_space X] [metric_space Y] [metric_space Z] 
  (right_angle : ∀ (Y : ℝ), is_right_angle Y) 
  (cos_X : ∀ (X Y : ℝ), cos X = 3 / 5)
  (YZ_equal_10 : YZ = 10) : 
  ∀ (XY : ℝ), XY = 6 :=
by
  sorry

end triangle_XYZ_cosine_l474_474273


namespace remainder_x150_l474_474137

theorem remainder_x150 (x : ℝ) : 
  ∃ r : ℝ, ∃ q : ℝ, x^150 = q * (x - 1)^3 + 11175*x^2 - 22200*x + 11026 := 
by
  sorry

end remainder_x150_l474_474137


namespace Tessa_has_apples_l474_474723

variable (T : ℕ) [has_add ℕ]

def apples_original := 4
def apples_given := 5

def total_apples (apples_original apples_given : ℕ) := apples_original + apples_given

theorem Tessa_has_apples :
  total_apples apples_original apples_given = 9 :=
by
  sorry

end Tessa_has_apples_l474_474723


namespace complex_transformation_result_l474_474107

noncomputable def complex_transform (z : ℂ) (θ : ℝ) (k : ℝ) : ℂ :=
  let cis_θ := complex.of_real (θ.cos) + complex.I * complex.of_real (θ.sin)
  k * (z * cis_θ)

theorem complex_transformation_result :
  complex_transform (-4 - 6 * complex.I) (real.pi / 3) 2 =
  (-4 + 6 * real.sqrt 3) + (-4 * real.sqrt 3 - 6) * complex.I :=
by
  sorry

end complex_transformation_result_l474_474107


namespace geometric_series_problem_l474_474301

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_problem
  (c d : ℝ)
  (h : geometric_series_sum (c/d) (1/d) = 6) :
  geometric_series_sum (c/(c + 2 * d)) (1/(c + 2 * d)) = 3 / 4 := by
  sorry

end geometric_series_problem_l474_474301


namespace comprehensive_evaluation_score_l474_474793

theorem comprehensive_evaluation_score
  (learning_score : ℕ) (physical_education_score : ℕ) (arts_score : ℕ)
  (learning_weight : ℕ) (pe_weight : ℕ) (arts_weight : ℕ)
  (total_weight : ℕ)
  (h_learning : learning_score = 90)
  (h_pe : physical_education_score = 80)
  (h_arts : arts_score = 85)
  (h_learning_weight : learning_weight = 5)
  (h_pe_weight : pe_weight = 3)
  (h_arts_weight : arts_weight = 2)
  (h_total_weight : total_weight = 10) :
  (learning_score * learning_weight + physical_education_score * pe_weight + arts_score * arts_weight) / total_weight = 86 :=
by
  rw [h_learning, h_pe, h_arts, h_learning_weight, h_pe_weight, h_arts_weight, h_total_weight]
  sorry

end comprehensive_evaluation_score_l474_474793


namespace combination_choosing_four_socks_l474_474713

theorem combination_choosing_four_socks (n k : ℕ) (h_n : n = 7) (h_k : k = 4) :
  (nat.choose n k) = 35 :=
by
  rw [h_n, h_k, nat.choose_succ_succ, nat.choose_succ_succ, nat.choose_succ_succ, nat.choose_zero_succ]
  simp only [nat.choose_succ_succ, nat.factorial_succ, nat.factorial, nat.succ_sub_succ_eq_sub, nat.sub_zero,
    nat.pred_succ, nat.factorial_succ, nat.choose_self, show nat.factorial 0 = 1 by rfl, tsub_zero,
    mul_one, mul_over_nat_eq, nat.succ.sub_prime, nat.succ_sub_succ_eq_sub, nat.factorial_zero]
  norm_num
  sorry

end combination_choosing_four_socks_l474_474713


namespace otimes_h_h_h_eq_h_l474_474114

variable (h : ℝ)

def otimes (x y : ℝ) : ℝ := x^3 - y

theorem otimes_h_h_h_eq_h : otimes h (otimes h h) = h := by
  -- Proof goes here, but is omitted
  sorry

end otimes_h_h_h_eq_h_l474_474114


namespace least_k_l474_474151

noncomputable def w (n : ℕ) : ℕ := 
  if h : n = 0 then 0 else 
  (Nat.factors n).toFinset.card

theorem least_k (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, k = 5 ∧ 2 ^ w n ≤ k * (n ^ (1 / 4 : ℝ)) := 
by
  use 5
  sorry

end least_k_l474_474151


namespace max_divisible_n_l474_474467

def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

noncomputable def canBeDivided (n : ℕ) (A B : Set ℕ) : Prop :=
  (∀ x ∈ A, ∀ y ∈ A, x ≠ y → ¬isPerfectSquare (x + y)) ∧
  (∀ x ∈ B, ∀ y ∈ B, x ≠ y → ¬isPerfectSquare (x + y)) ∧
  (A ∪ B = { x | 1 ≤ x ∧ x ≤ n }) ∧
  (A ∩ B = ∅)

theorem max_divisible_n : ∃ n, n = 14 ∧
  ∃ A B : Set ℕ, canBeDivided n A B := 
begin
  sorry
end

end max_divisible_n_l474_474467


namespace dragon_defeated_l474_474252

universe u

inductive Dragon
| heads (count : ℕ) : Dragon

open Dragon

-- Conditions based on probabilities
def chop_probability : ℚ := 1 / 4
def one_grow_probability : ℚ := 1 / 3
def no_grow_probability : ℚ := 5 / 12

noncomputable def probability_defeated_dragon : ℚ :=
  if h : chop_probability + one_grow_probability + no_grow_probability = 1 then
    -- Define the recursive probability of having zero heads eventually (∞ case)
    let rec prob_defeat (d : Dragon) : ℚ :=
      match d with
      | heads 0     => 1  -- Base case: no heads mean dragon is defeated
      | heads (n+1) => 
        no_grow_probability * prob_defeat (heads n) + -- Successful strike
        one_grow_probability * prob_defeat (heads (n + 1)) + -- Neutral strike
        chop_probability * prob_defeat (heads (n + 2)) -- Unsuccessful strike
    prob_defeat (heads 3)  -- Initial condition with 3 heads
  else 0

-- Final theorem statement asserting the probability of defeating the dragon => 1
theorem dragon_defeated : probability_defeated_dragon = 1 :=
sorry

end dragon_defeated_l474_474252


namespace maximum_omega_l474_474961

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) := 2 * Real.cos (ω * x + φ)

theorem maximum_omega (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ ∧ φ < π / 2) 
  (h3 : f ω φ 0 = Real.sqrt 2)
  (h4 : ∀ x ∈ Icc 0 (π / 2), f ω φ x ≥ f ω φ (π / 2)) : 
  ω ≤ 3 / 2 :=
sorry

end maximum_omega_l474_474961


namespace range_f_monotonic_f_max_area_triangle_l474_474492

noncomputable def f (x : ℝ) : ℝ := (√3) * sin x * cos x - (cos x) ^ 2 + (1 / 2)

theorem range_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) : -1 / 2 ≤ f(x) ∧ f(x) ≤ 1 :=
sorry

theorem monotonic_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) : monotonic (λ x, f(x)) (set.Icc 0 (π / 3)) :=
sorry

theorem max_area_triangle {a b c : ℝ} (h : a + b + c = 1) (B : ℝ) 
  (hB : f(B) = -1 / 2) : 
  let area := (1 / 2) * a * c * (√3 / 2)
  in area ≤ (7 * √3 - 12) / 4 :=
sorry

end range_f_monotonic_f_max_area_triangle_l474_474492


namespace initial_time_to_cover_distance_l474_474054

-- Definitions based on the conditions
def distance : ℕ := 324
def speed : ℕ := 36
def factor : ℝ := 3 / 2

-- The given proof problem rewritten in Lean 4
theorem initial_time_to_cover_distance :
  ∃ T : ℝ, distance = speed * (factor * T) ∧ T = 6 := 
sorry

end initial_time_to_cover_distance_l474_474054


namespace determine_a_for_f_nonnegative_l474_474156

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x + 1

theorem determine_a_for_f_nonnegative :
  ∀ a, (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f a x ≥ 0) ↔ a = 4 :=
by {
  sorry
}

end determine_a_for_f_nonnegative_l474_474156


namespace area_PQRSTU_correct_l474_474331

-- Define the lengths of the sides based on the conditions given
def PR : ℝ := 6
def RS : ℝ := 8
def UT : ℝ := 5
def TS : ℝ := 3

-- Define the areas of triangles PQRS and QTUS
def area_PQRS : ℝ := 1 / 2 * PR * RS
def area_QTUS : ℝ := 1 / 2 * UT * TS

-- Define the total area of polygon PQRSTU
def area_PQRSTU : ℝ := area_PQRS + area_QTUS

-- Prove that the area of polygon PQRSTU is 31.5 square units
theorem area_PQRSTU_correct : area_PQRSTU = 31.5 := by
  unfold area_PQRSTU area_PQRS area_QTUS PR RS UT TS
  simp
  norm_num
  sorry

end area_PQRSTU_correct_l474_474331


namespace acute_triangle_tangent_difference_range_l474_474270

theorem acute_triangle_tangent_difference_range {A B C a b c : ℝ} 
    (h1 : a^2 + b^2 > c^2) (h2 : b^2 + c^2 > a^2) (h3 : c^2 + a^2 > b^2)
    (hb2_minus_ha2_eq_ac : b^2 - a^2 = a * c) :
    1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < (2 * Real.sqrt 3 / 3) :=
by
  sorry

end acute_triangle_tangent_difference_range_l474_474270


namespace domain_of_f_l474_474881

def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 7*x^2 - 4*x + 2) / (x^3 - 3*x^2 + 2*x)

theorem domain_of_f : 
  ∀ x : ℝ, (x ∈ (-∞, 0) ∨ x ∈ (0, 1) ∨ x ∈ (1, 2) ∨ x ∈ (2, ∞)) ↔ 
  (∃ y : ℝ, f y = f x) :=
begin
  sorry
end

end domain_of_f_l474_474881


namespace trigonometric_identity_l474_474489

theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (-1 + Real.sqrt 3) / 2 :=
sorry

end trigonometric_identity_l474_474489


namespace find_q_minus_p_values_l474_474740

theorem find_q_minus_p_values (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) 
    (h : (p * (q + 1) + q * (p + 1)) * (n + 2) = 2 * n * p * q) : 
    q - p = 2 ∨ q - p = 3 ∨ q - p = 5 :=
sorry

end find_q_minus_p_values_l474_474740


namespace greatest_4_digit_number_l474_474762

theorem greatest_4_digit_number
  (n : ℕ)
  (h1 : n % 5 = 3)
  (h2 : n % 9 = 2)
  (h3 : 1000 ≤ n)
  (h4 : n < 10000) :
  n = 9962 := 
sorry

end greatest_4_digit_number_l474_474762


namespace find_sequence_index_l474_474268

theorem find_sequence_index (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) - 3 = a n)
  (h₃ : ∃ n, a n = 2023) : ∃ n, a n = 2023 ∧ n = 675 := 
by 
  sorry

end find_sequence_index_l474_474268


namespace solve_for_x_l474_474470

theorem solve_for_x (x : ℝ) (h : 2 * log 3 x = log 3 (5 * x)) : x = 5 :=
by
  -- skip the proof by adding sorry
  sorry

end solve_for_x_l474_474470


namespace triangle_area_inequality_l474_474745

-- Define the variables involved
variables {a b c T : ℝ} 

-- Define the semi-perimeter s in terms of a, b, c
def s : ℝ := (a + b + c) / 2

-- Define the area T using Heron's formula
def area (a b c : ℝ) : ℝ := real.sqrt (s * (s - a) * (s - b) * (s - c))

-- State the theorem in the form of a Lean 4 statement
theorem triangle_area_inequality (a b c T : ℝ) (h : T = area a b c) : 
  T ^ 2 ≤ (a * b * c * (a + b + c)) / 16 :=
by
  sorry

end triangle_area_inequality_l474_474745


namespace choose_five_from_eight_l474_474606

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474606


namespace combination_eight_choose_five_l474_474630

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474630


namespace log_8_x_eq_3_75_l474_474988

theorem log_8_x_eq_3_75 (x : ℝ) (h : log 8 x = 3.75) : x = 1024 * sqrt 2 :=
sorry

end log_8_x_eq_3_75_l474_474988


namespace train_length_correct_l474_474855

noncomputable def train_length (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * time

theorem train_length_correct :
  train_length 17.998560115190784 36 = 179.98560115190784 :=
by
  sorry

end train_length_correct_l474_474855


namespace boat_speed_in_still_water_l474_474417

def speed_of_boat (V_b : ℝ) : Prop :=
  let stream_speed := 4  -- speed of the stream in km/hr
  let downstream_distance := 168  -- distance traveled downstream in km
  let time := 6  -- time taken to travel downstream in hours
  (downstream_distance = (V_b + stream_speed) * time)

theorem boat_speed_in_still_water : ∃ V_b, speed_of_boat V_b ∧ V_b = 24 := 
by
  exists 24
  unfold speed_of_boat
  simp
  sorry

end boat_speed_in_still_water_l474_474417


namespace extreme_value_of_function_l474_474351

theorem extreme_value_of_function :
  ∃ x : ℝ, (∀ y : ℝ, 2*y*Real.exp(y) ≥ 2*x*Real.exp(x)) ∧ 2*x*Real.exp(x) = -2/Real.exp(1) :=
sorry

end extreme_value_of_function_l474_474351


namespace infinite_series_sum_l474_474304

noncomputable def r : ℝ :=
  Classical.choose (exists_unique_pos_real_sol (λ x : ℝ, x^3 + (3/4 : ℝ) * x - 1 = 0))

theorem infinite_series_sum :
  let S := ∑' (n : ℕ), (↑n + 2 : ℝ) * r^((n + 1) * 3)
  S = (64 * r * (2 - (3/4) * r)) / 9 :=
by
  sorry -- Proof is omitted

end infinite_series_sum_l474_474304


namespace find_k_l474_474183

variable {k x1 x2 : ℝ}

-- Given conditions
def quadratic_eq_roots : Prop := 
  x1^2 + (2 * k - 1) * x1 - k - 1 = 0 ∧ x2^2 + (2 * k - 1) * x2 - k - 1 = 0

def sum_and_product_eq : Prop := 
  x1 + x2 - 4 * (x1 * x2) = 2

-- The theorem statement
theorem find_k (h_roots : quadratic_eq_roots) (h_sum_product : sum_and_product_eq) : k = -3 / 2 :=
sorry

end find_k_l474_474183


namespace part_a_part_b_l474_474355

-- Definitions for the problem
def bags : Sort := {bag : Fin 5 // ∀ (i : Fin 5), bag i >= 100}
def weight (b : bags) : Nat :=
  match b with
  | 0 => 10
  | 1 => 11
  | 2 => 12
  | 3 => 13
  | 4 => 14
  | _ => 0

-- Part (a)
def can_determine_10g (pointed_bag : bags) : Prop :=
  ∃ (left_coins right_coins : Fin 99 → bags), by
    have : ∀ (i : Fin 99), weight (right_coins i) = 10
    sorry

-- Part (b)
def can_determine_weight (pointed_bag : bags) : Prop :=
  ∃ (weighings : List (Fin 100 → bags)) (results : List (Bool × (Fin 100 → bags))), by
    have : ∀ (w : Fin 100), weight (weighings.head! w) = weight pointed_bag
    sorry

theorem part_a (pointed_bag : bags) : can_determine_10g pointed_bag := 
  sorry

theorem part_b (pointed_bag : bags) : can_determine_weight pointed_bag := 
  sorry

end part_a_part_b_l474_474355


namespace Nell_has_123_more_baseball_cards_than_Ace_cards_l474_474324

def Nell_cards_diff (baseball_cards_new : ℕ) (ace_cards_new : ℕ) : ℕ :=
  baseball_cards_new - ace_cards_new

theorem Nell_has_123_more_baseball_cards_than_Ace_cards:
  (Nell_cards_diff 178 55) = 123 :=
by
  -- proof here
  sorry

end Nell_has_123_more_baseball_cards_than_Ace_cards_l474_474324


namespace total_prime_factors_33_l474_474406

def expression := (4^13) * (7^5) * (11^2)

theorem total_prime_factors_33 : 
  (PrimeCount (4^13) + PrimeCount (7^5) + PrimeCount (11^2)) = 33 := 
by 
  sorry

end total_prime_factors_33_l474_474406


namespace flower_shop_ratio_l474_474583

theorem flower_shop_ratio (V C T R : ℕ) 
(total_flowers : V + C + T + R > 0)
(tulips_ratio : T = V / 4)
(roses_tulips_equal : R = T)
(carnations_fraction : C = 2 / 3 * (V + T + R + C)) 
: V / C = 1 / 3 := 
by
  -- Proof omitted
  sorry

end flower_shop_ratio_l474_474583


namespace max_min_not_always_greater_false_derivative_not_always_max_min_extreme_in_interval_extreme_value_point_l474_474886

section

variable {α : Type*} [TopologicalSpace α]

-- Problem 1
theorem max_min_not_always_greater (f : α → ℝ) :
    ¬(∀ x y, f x ≥ f y) → ∃ x y, f x < f y :=
sorry

-- Problem 2
theorem false_derivative (f : ℝ → ℝ) :
    ¬(∀ a, deriv f a = 0 → ∃ x, is_extreme_point f x) :=
sorry

-- Problem 3
theorem not_always_max_min (f : ℝ → ℝ) :
    ¬(∃ a b, ∀ x, f x ≤ f a ∧ f x ≥ f b) :=
sorry

-- Problem 4
theorem extreme_in_interval {f : ℝ → ℝ} {a b : ℝ} (h : ∈ set.Icc a b) :
    ∃ x ∈ set.Icc a b, is_extreme_point f x :=
sorry

-- Problem 5
theorem extreme_value_point (f : ℝ → ℝ) (x : ℝ) :
    is_extreme_point f x → ∃ y, f y = x :=
sorry

end

end max_min_not_always_greater_false_derivative_not_always_max_min_extreme_in_interval_extreme_value_point_l474_474886


namespace logarithmic_identity_l474_474563

theorem logarithmic_identity (x : ℝ) (h : log 5 (log 4 (log 3 x)) = 0) : x⁻¹^(1 / 3) = 1 / (3 * real.cbrt 3) :=
by
  have : log 4 (log 3 x) = 1 := sorry,
  have : log 3 x = 4 := sorry,
  have : x = 81 := sorry,
  have : x⁻¹^(1 / 3) = 81⁻¹^(1 / 3) := sorry,
  exact this

end logarithmic_identity_l474_474563


namespace volume_ratio_l474_474094

-- Definitions and Theorem Statement
variables {a m x : ℝ}

-- Condition: Relationship between cone and cylinder using similar triangles
axiom similar_triangles : a/m = x/(m - x)

-- Condition: Height of the cone
axiom cone_height : m = a / 2 * sqrt 3

-- Theorem: The ratio of the volume of the cone to the volume of the cylinder
theorem volume_ratio (a m x : ℝ)
  (similar_triangles : a/m = x/(m - x))
  (cone_height : m = a/2 * sqrt 3) :
  ∃ k, k = (26 + 15 * sqrt 3) / 18 :=
sorry

end volume_ratio_l474_474094


namespace smallest_n_is_60_l474_474395

def smallest_n (n : ℕ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ (24 ∣ n^2) ∧ (450 ∣ n^3) ∧ ∀ m : ℕ, 24 ∣ m^2 → 450 ∣ m^3 → m ≥ n

theorem smallest_n_is_60 : smallest_n 60 :=
  sorry

end smallest_n_is_60_l474_474395


namespace probability_card_1_and_2_same_envelope_l474_474004

-- Definitions to set up conditions
def cards := {1, 2, 3, 4, 5, 6}
def envelopes := {1, 2, 3}

-- Function to calculate combinations
noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem probability_card_1_and_2_same_envelope :
  let total_ways := comb 6 2 * comb 4 2 * comb 2 2,
      favorable_ways := envelopes.card * comb 4 2 in
  (favorable_ways : ℚ) / total_ways = 1 / 5 :=
by
  sorry

end probability_card_1_and_2_same_envelope_l474_474004


namespace combination_8_5_l474_474619

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474619


namespace linda_savings_l474_474690

theorem linda_savings :
  let original_price_per_notebook := 3.75
  let discount_rate := 0.15
  let quantity := 12
  let total_price_without_discount := quantity * original_price_per_notebook
  let discount_amount_per_notebook := original_price_per_notebook * discount_rate
  let discounted_price_per_notebook := original_price_per_notebook - discount_amount_per_notebook
  let total_price_with_discount := quantity * discounted_price_per_notebook
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 6.75 :=
by {
  sorry
}

end linda_savings_l474_474690


namespace distinct_four_digit_integers_l474_474363

open Nat

theorem distinct_four_digit_integers (count_digs_18 : ℕ) :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (∃ d1 d2 d3 d4 : ℕ,
      d1 * d2 * d3 * d4 = 18 ∧
      d1 > 0 ∧ d1 < 10 ∧
      d2 > 0 ∧ d2 < 10 ∧
      d3 > 0 ∧ d3 < 10 ∧
      d4 > 0 ∧ d4 < 10 ∧
      n = d1 * 1000 + d2 * 100 + d3 * 10 + d4)) →
  count_digs_18 = 24 :=
sorry

end distinct_four_digit_integers_l474_474363


namespace basis_plane_vectors_l474_474501

variables {K : Type*} [Field K]
variables {V : Type*} [AddCommGroup V] [Module K V] (a b : V)

-- Definitions of non-zero and non-collinear vectors
def non_zero (v : V) := v ≠ 0
def non_collinear (u v : V) := ∃ k : K, k • u ≠ v ⬝ v ≠ 0

-- Non-zero and non-collinear conditions for vectors a and b
axiom h1 : non_zero a
axiom h2 : non_zero b
axiom h3 : ¬ collinear a b

-- Prove that a + b and a - b can serve as a basis for plane vectors
theorem basis_plane_vectors (a b : V) (h1 : non_zero a) (h2 : non_zero b) (h3 : ¬ non_collinear a b) :
  (a + b) ≠ (0:V) ∧ (a - b) ≠ (0:V) ∧ ¬ collinear (a + b) (a - b) :=
sorry

end basis_plane_vectors_l474_474501


namespace solution_set_inequality_l474_474512

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f(x) - x^2

-- Conditions
axiom even_f : ∀ x : ℝ, f(-x) = f(x)
axiom g_increasing : ∀ {a b : ℝ}, 0 ≤ a → a ≤ b → g(a) ≤ g(b)

-- The proof statement
theorem solution_set_inequality :
  {x : ℝ | f(x + 2) - f(2) > x^2 + 4 * x} = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end solution_set_inequality_l474_474512


namespace percentage_spent_on_meat_l474_474651

def total_cost_broccoli := 3 * 4
def total_cost_oranges := 3 * 0.75
def total_cost_vegetables := total_cost_broccoli + total_cost_oranges + 3.75
def total_cost_chicken := 3 * 2
def total_cost_meat := total_cost_chicken + 3
def total_cost_groceries := total_cost_meat + total_cost_vegetables

theorem percentage_spent_on_meat : 
  (total_cost_meat / total_cost_groceries) * 100 = 33 := 
by
  sorry

end percentage_spent_on_meat_l474_474651


namespace combination_seven_choose_four_l474_474717

theorem combination_seven_choose_four : nat.choose 7 4 = 35 := by
  -- Proof
  sorry

end combination_seven_choose_four_l474_474717


namespace complex_root_condition_l474_474686

open Complex

theorem complex_root_condition (u v : ℂ) 
    (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
    (h2 : abs (u + v) = abs (u * v + 1)) :
    u = 1 ∨ v = 1 :=
sorry

end complex_root_condition_l474_474686


namespace percentage_of_five_digit_numbers_with_repeated_digits_l474_474987

theorem percentage_of_five_digit_numbers_with_repeated_digits :
  let total_five_digit_numbers := 90000
  let without_repeated_digits := 27216
  let with_repeated_digits := total_five_digit_numbers - without_repeated_digits
  let percentage := (with_repeated_digits : ℝ) / (total_five_digit_numbers : ℝ) * 100
  let x := Float.round (percentage * 10) / 10
  x = 69.8 := by
  sorry

end percentage_of_five_digit_numbers_with_repeated_digits_l474_474987


namespace part_a_part_b_l474_474354

-- Definitions for the problem
def bags : Sort := {bag : Fin 5 // ∀ (i : Fin 5), bag i >= 100}
def weight (b : bags) : Nat :=
  match b with
  | 0 => 10
  | 1 => 11
  | 2 => 12
  | 3 => 13
  | 4 => 14
  | _ => 0

-- Part (a)
def can_determine_10g (pointed_bag : bags) : Prop :=
  ∃ (left_coins right_coins : Fin 99 → bags), by
    have : ∀ (i : Fin 99), weight (right_coins i) = 10
    sorry

-- Part (b)
def can_determine_weight (pointed_bag : bags) : Prop :=
  ∃ (weighings : List (Fin 100 → bags)) (results : List (Bool × (Fin 100 → bags))), by
    have : ∀ (w : Fin 100), weight (weighings.head! w) = weight pointed_bag
    sorry

theorem part_a (pointed_bag : bags) : can_determine_10g pointed_bag := 
  sorry

theorem part_b (pointed_bag : bags) : can_determine_weight pointed_bag := 
  sorry

end part_a_part_b_l474_474354


namespace g_simplified_form_g_range_l474_474960

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - x)

noncomputable def g (x : ℝ) : ℝ :=
  sin x * f (sin (2 * x)) + (real.sqrt 6 + real.sqrt 2) / 4 * f (cos (4 * x))

theorem g_simplified_form (x : ℝ) (h : x ∈ set.Icc (-π / 4) 0) :
  g x = -sin (2 * x - π / 6) - 1 / 2 :=
sorry

theorem g_range : set.image g (set.Icc (-π / 4) 0) = set.Ioc 0 (1 / 2) :=
sorry

end g_simplified_form_g_range_l474_474960


namespace shortest_distance_ln_curve_to_line_l474_474370

noncomputable def shortestDistance (x : ℝ) : ℝ :=
  let y := Real.log (2 * x - 1) in
  Real.abs (2 * x - y + 3) / Real.sqrt (2 ^ 2 + (-1) ^ 2)

theorem shortest_distance_ln_curve_to_line :
  ∀ (x : ℝ), shortestDistance x ≥ 0 ∧
  (∃ p : ℝ, p = 1 ∧ shortestDistance p = Real.sqrt 5) :=
by
  sorry

end shortest_distance_ln_curve_to_line_l474_474370


namespace triangle_is_equilateral_l474_474827

-- Define the conditions
variables {P Q R A B C D E F : Type*}
-- Points where the circle intersects the triangle sides
variables [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (circ : Circle P Q R A B C D E F)

-- The main theorem to prove the triangle is equilateral
theorem triangle_is_equilateral 
  (h1 : divides_into_three_equal_parts circ A B P Q)
  (h2 : divides_into_three_equal_parts circ C D Q R)
  (h3 : divides_into_three_equal_parts circ E F R P) :
  segment_length P Q = segment_length Q R ∧ 
  segment_length Q R = segment_length R P ∧ 
  segment_length R P = segment_length P Q :=
sorry

end triangle_is_equilateral_l474_474827


namespace rectangle_area_l474_474036

theorem rectangle_area (length_of_rectangle radius_of_circle side_of_square : ℝ)
  (h1 : length_of_rectangle = (2 / 5) * radius_of_circle)
  (h2 : radius_of_circle = side_of_square)
  (h3 : side_of_square * side_of_square = 1225)
  (breadth_of_rectangle : ℝ)
  (h4 : breadth_of_rectangle = 10) : 
  length_of_rectangle * breadth_of_rectangle = 140 := 
by 
  sorry

end rectangle_area_l474_474036


namespace min_value_of_fraction_l474_474236

open Real

theorem min_value_of_fraction (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ m, m = Inf (set_of (λ k, ∃ x y, x^2 + y^2 = 4 ∧ k = xy / (x + y - 2))) ∧ m = 1 - √2 :=
begin
  sorry
end

end min_value_of_fraction_l474_474236


namespace problem_statement_l474_474670

noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

def T : Set ℝ := {y | ∃ (x : ℝ), x ≥ 0 ∧ y = g x}

def P : ℝ := 3
def Q : ℝ := 4 / 3

theorem problem_statement :
  Q ∈ T ∧ P ∉ T := by
  sorry

end problem_statement_l474_474670


namespace solve_congruence_l474_474476

-- Define the given congruence
theorem solve_congruence (x : ℤ) : 10 * x + 3 ≡ 6 [MOD 15] → x ≡ 0 [MOD 3] := by
  sorry

end solve_congruence_l474_474476


namespace kekai_ratio_l474_474284

/-
Kekai sells 5 shirts at $1 each,
5 pairs of pants at $3 each,
and he has $10 left after giving some money to his parents.
Our goal is to prove the ratio of the money Kekai gives to his parents
to the total money he earns from selling his clothes is 1:2.
-/

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := (shirts_sold * shirt_price) + (pants_sold * pants_price)
def money_given_to_parents : ℕ := total_earnings - money_left
def ratio (a b : ℕ) := (a / Nat.gcd a b, b / Nat.gcd a b)

theorem kekai_ratio : ratio money_given_to_parents total_earnings = (1, 2) :=
  by
    sorry

end kekai_ratio_l474_474284


namespace combination_8_5_is_56_l474_474616

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474616


namespace regular_price_of_polo_shirt_l474_474799

/--
Zane purchases 2 polo shirts from the 40% off rack at the men's store. 
The polo shirts are priced at a certain amount at the regular price. 
He paid $60 for the shirts. 
Prove that the regular price of each polo shirt is $50.
-/
theorem regular_price_of_polo_shirt (P : ℝ) 
  (h1 : ∀ (x : ℝ), x = 0.6 * P → 2 * x = 60) : 
  P = 50 :=
sorry

end regular_price_of_polo_shirt_l474_474799


namespace find_a_intervals_monotonicity_extremum_l474_474522

variable (a : ℝ)
noncomputable def f (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

theorem find_a (h : ∀ x : ℝ, f x = x / 4 + a / x - Real.log x - 3 / 2) (ha : f' 1 = -2) :
  a = 5 / 4 := by
  sorry

noncomputable def f' (x : ℝ) : ℝ := 1 / 4 - a / x^2 - 1 / x

theorem intervals_monotonicity_extremum (h₁ : ∀ x : ℝ, f' x = 1 / 4 - (5 / 4) / x^2 - 1 / x) :
  (∀ x ∈ (0, 5), f' x < 0) ∧ (∀ x ∈ (5, ∞), f' x > 0) ∧ (∃ y : ℝ, y = f 5 ∧ y = -Real.log 5) := by
  sorry

end find_a_intervals_monotonicity_extremum_l474_474522


namespace bus_speed_excluding_stoppages_l474_474898

theorem bus_speed_excluding_stoppages (v : ℕ): (45 : ℝ) = (5 / 6 * v) → v = 54 :=
by
  sorry

end bus_speed_excluding_stoppages_l474_474898


namespace time_to_hit_plane_l474_474633

theorem time_to_hit_plane
  (d : ℝ) (q : ℝ) (m : ℝ)
  (ε₀ : ℝ) (π : ℝ)
  (h_d : d = 1)
  (h_q : q = 1.6 * 10 ^ (-19))
  (h_m : m = 1.67 * 10 ^ (-27))
  (h_ε₀ : ε₀ = 8.85 * 10 ^ (-12))
  (h_π : π = Math.pi) :
  let T₂ := sqrt((16 * π^3 * ε₀ * m * d^3) / q^2)
  let T₁ := 2 * π * sqrt((2 * π * ε₀ * m * d^3) / q)
  let t := T₁ / 2
  t = 5.98 :=
by {
  sorry
}

end time_to_hit_plane_l474_474633


namespace sum_of_first_fifteen_terms_l474_474142

noncomputable def a₃ : ℝ := -5
noncomputable def a₅ : ℝ := 2.4
noncomputable def a₁ : ℝ := -12.4
noncomputable def d : ℝ := 3.7

noncomputable def S₁₅ : ℝ := 15 / 2 * (2 * a₁ + 14 * d)

theorem sum_of_first_fifteen_terms :
  S₁₅ = 202.5 := 
by
  sorry

end sum_of_first_fifteen_terms_l474_474142


namespace comb_8_5_eq_56_l474_474595

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474595


namespace am_gm_inequality_l474_474945

theorem am_gm_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ a + b + c :=
by
  sorry

end am_gm_inequality_l474_474945


namespace solve_trig_eqn_solution_set_l474_474475

theorem solve_trig_eqn_solution_set :
  {x : ℝ | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} =
  {x : ℝ | 2 * Real.sin ((2 / 3) * x) = 1} :=
by
  sorry

end solve_trig_eqn_solution_set_l474_474475


namespace germination_percentage_is_correct_l474_474921

noncomputable def percentage_germinated (seeds_planted1 seeds_planted2 germinated1_percentage germinated2_percentage : ℝ) : ℝ :=
  let germinated1 := seeds_planted1 * (germinated1_percentage / 100)
  let germinated2 := seeds_planted2 * (germinated2_percentage / 100)
  let total_seeds_planted := seeds_planted1 + seeds_planted2
  let total_seeds_germinated := germinated1 + germinated2
  (total_seeds_germinated / total_seeds_planted) * 100

theorem germination_percentage_is_correct :
  percentage_germinated 500 200 30 50 = 35.71 :=
by
  calc
    percentage_germinated 500 200 30 50
        = ((500 * 0.30 + 200 * 0.50) / (500 + 200)) * 100 : by sorry
    ... = (250 / 700) * 100 : by sorry
    ... = 35.71 : by sorry

end germination_percentage_is_correct_l474_474921


namespace match_processes_count_l474_474456

-- Define the sets and the number of interleavings
def team_size : ℕ := 4 -- Each team has 4 players

-- Define the problem statement
theorem match_processes_count :
  (Nat.choose (2 * team_size) team_size) = 70 := by
  -- This is where the proof would go, but we'll use sorry as specified
  sorry

end match_processes_count_l474_474456


namespace student_count_l474_474418

open Nat

theorem student_count :
  ∃ n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by {
  -- placeholder for the proof
  sorry
}

end student_count_l474_474418


namespace jellybean_probability_l474_474816

theorem jellybean_probability :
  let total_jellybeans := 15
  let green_jellybeans := 6
  let purple_jellybeans := 2
  let yellow_jellybeans := 7
  let total_picked := 4
  let total_ways := Nat.choose total_jellybeans total_picked
  let ways_to_pick_two_yellow := Nat.choose yellow_jellybeans 2
  let ways_to_pick_two_non_yellow := Nat.choose (total_jellybeans - yellow_jellybeans) 2
  let successful_outcomes := ways_to_pick_two_yellow * ways_to_pick_two_non_yellow
  let probability := successful_outcomes / total_ways
  probability = 4 / 9 := by
sorry

end jellybean_probability_l474_474816


namespace expr1_correct_expr2_correct_l474_474046

section Statements

-- Defining the first expression
def expr1 : ℚ := (2 + (4/5)) ^ 0 + 2 ^ -2 * (2 + (1/4)) ^ - (1/2) - (8 / 27) ^ (1/3)

-- Defining the correct answer for the first expression
def ans1 : ℚ := 1 / 2

-- Theorem stating that the first expression evaluates to the correct answer
theorem expr1_correct : expr1 = ans1 :=
  by
  sorry

-- Defining the second expression
def expr2 : ℚ := (25 / 16) ^ 0.5 + (27 / 8) ^ (-1/3) - 2 * (π ^ 0) + 
                  4 ^ (log 5 / log 4) - log (exp 5) + log10 200 - log10 2

-- Defining the correct answer for the second expression
def ans2 : ℚ := 23 / 12

-- Theorem stating that the second expression evaluates to the correct answer
theorem expr2_correct : expr2 = ans2 :=
  by
  sorry

end Statements

end expr1_correct_expr2_correct_l474_474046


namespace ellipse_properties_and_segment_length_l474_474958

open Real

noncomputable def ellipse_1_equation (a : ℝ) (h : a > 1) : Prop := 
  ∀ x y : ℝ, (x^2 / a^2 + y^2 = 1)

noncomputable def curve_2_equation (t : ℝ) (ht : 0 < t ∧ t ≤ sqrt 2 / 2) : Prop :=
  ∀ x y : ℝ, (x - t)^2 + y^2 = (t^2 + sqrt 3 * t)^2

theorem ellipse_properties_and_segment_length (a b c t : ℝ) (h_a : a > 1) 
  (h_t : 0 < t ∧ t ≤ sqrt 2 / 2) 
  (h_major : 2a) (h_minor : 2b) (h_focus : 2c) 
  (h_focal_dist : 2 * (2 * c)^2 = (2 * a)^2 + (2 * b)^2) 
  (h_minor_value : b = 1) :
  (ellipse_1_equation a h_a) ∧ 
  (curve_2_equation t h_t) ∧ 
  (a^2 = 3 ∧ c^2 = 2 ∧ ∀ k : ℝ, 0 < k^2 ∧ k^2 ≤ 1 → ∃ m : ℝ, m = sqrt(k^2 + 1) ) ∧ 
  (∃ l_pt_A : ℝ × ℝ, l_pt_A = (-sqrt 3, 0)) ∧
  (∀ k : ℝ, y = k * (x + sqrt 3) → ∃ AB_len : ℝ, AB_len = sqrt 6 / 2) := 
    sorry

end ellipse_properties_and_segment_length_l474_474958


namespace ellipse_x_intersect_l474_474446

theorem ellipse_x_intersect (F1 F2 : ℝ × ℝ)
  (hf1 : F1 = (0, 2))
  (hf2 : F2 = (4, 0))
  (hintersect_orig : (0 : ℝ, 0) = (0, 0)) :
  ∃ x : ℝ, (x = 24 / 5) ∧ (𝕋 : ℝ × ℝ) = (x, 0) :=
by
  sorry

end ellipse_x_intersect_l474_474446


namespace find_a_that_solves_integral_l474_474965

open Real

theorem find_a_that_solves_integral :
  (∫ x in 0..π/4, sin x - (-sqrt 2) * cos x) = -sqrt 2 / 2 :=
sorry

end find_a_that_solves_integral_l474_474965


namespace calculate_interest_rate_l474_474860

noncomputable def compoundInterestFormula (P A : ℝ) (r : ℝ) (n t : ℕ) : Prop := 
  A = P * (1 + r / n) ^ (n * t)

theorem calculate_interest_rate :
  ∃ r : ℝ, compoundInterestFormula 6000 7260 r 1 2 ∧ r = 0.1 :=
by 
  use 0.1
  unfold compoundInterestFormula
  split
  sorry -- Proof omitted

end calculate_interest_rate_l474_474860


namespace angular_speed_proportion_l474_474590

variable (x y z w k : ℕ)
variable (ω_A ω_B ω_C ω_D : ℝ)
variable (k : ℝ)

-- Conditions given in the problem
axiom gear_ratios : 10 * x * ω_A = 15 * y * ω_B ∧ 15 * y * ω_B = 12 * z * ω_C ∧ 12 * z * ω_C = 20 * w * ω_D

-- Question to prove
theorem angular_speed_proportion :
  10 * x * ω_A = k →
  15 * y * ω_B = k →
  12 * z * ω_C = k →
  20 * w * ω_D = k →
  ω_A : ω_B : ω_C : ω_D = 12 * y * z * w : 8 * x * z * w : 10 * x * y * w : 6 * x * y * z :=
by
  sorry

end angular_speed_proportion_l474_474590


namespace evaluate_expression_l474_474786

theorem evaluate_expression : 
    (1 / ( (-5 : ℤ) ^ 4) ^ 2 ) * (-5 : ℤ) ^ 9 = -5 :=
by sorry

end evaluate_expression_l474_474786


namespace time_to_gather_leaves_in_minutes_l474_474320

-- Definitions of the conditions
def leaves_collected_per_cycle : ℕ := 5  -- Dad collects 5 leaves every 45 seconds
def leaves_scattered_per_cycle : ℕ := 3  -- Liam scatters 3 leaves every 45 seconds
def total_leaves_needed : ℕ := 50         -- Total of 50 leaves needed
def cycle_time_seconds : ℕ := 45          -- Each cycle takes 45 seconds
def seconds_per_minute : ℕ := 60

-- The goal to prove
theorem time_to_gather_leaves_in_minutes :
  (cycle_time_seconds * ((total_leaves_needed - 2) / (leaves_collected_per_cycle - leaves_scattered_per_cycle)) + cycle_time_seconds) / seconds_per_minute = 18.75 :=
by
  -- Placeholder for the proof
  sorry

end time_to_gather_leaves_in_minutes_l474_474320


namespace combination_8_5_is_56_l474_474612

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474612


namespace good_numbers_l474_474237

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : List ℕ), 
    a.Perm (List.range n) ∧
    ∀ k : ℕ, k < n → is_perfect_square (k + a.nthLe k sorry + 1) 

theorem good_numbers :
  {n ∈ {11, 13, 15, 17, 19} | is_good_number n} = {13, 15, 17, 19} := 
sorry

end good_numbers_l474_474237


namespace triangular_weight_l474_474743

noncomputable def rectangular_weight := 90
variables {C T : ℕ}

-- Conditions
axiom cond1 : C + T = 3 * C
axiom cond2 : 4 * C + T = T + C + rectangular_weight

-- Question: How much does the triangular weight weigh?
theorem triangular_weight : T = 60 :=
sorry

end triangular_weight_l474_474743


namespace father_son_age_ratio_proof_l474_474832

def father's_age := 64
def son's_age := 16

lemma age_ratio : father's_age / son's_age = 4 := by
  have h1: father's_age = 64 := rfl
  have h2: son's_age = 16 := rfl
  calc father's_age / son's_age
      = 64 / 16 : by rw [h1, h2]
  ... = 4 : by norm_num

-- The ratio of the father's age to the son's age
def father_to_son_age_ratio : Prop :=
  father's_age / son's_age = 4

theorem father_son_age_ratio_proof : father_to_son_age_ratio := age_ratio

end father_son_age_ratio_proof_l474_474832


namespace num_factors_of_multiples_of_180_l474_474225

theorem num_factors_of_multiples_of_180 :
  let m := 2^12 * 3^10 * 5^15 in
  ∃ n : ℕ, n = 1485 ∧ ∀ f ∈ finset.filter (λ x, 180 ∣ x) (set.to_finset (set_of (λ x, x ∣ m))), 1 ≤ f :=
sorry

end num_factors_of_multiples_of_180_l474_474225


namespace order_of_f_l474_474173

variable {f : ℝ → ℝ}

-- f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)

-- Condition 1: f is an odd function
axiom odd_f : is_odd_function f
-- Condition 2: satisfies f(x-4) = -f(x)
axiom f_shift : ∀ x, f(x-4) = -f(x)
-- Condition 3: f is increasing on interval [0, 2]
axiom f_increasing : ∀ x y, 0 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 2 → x < y → f(x) < f(y)

theorem order_of_f : f(-25) < f(80) ∧ f(80) < f(11) :=
by
  sorry

end order_of_f_l474_474173


namespace number_of_valid_sequences_l474_474555

-- Conditions
def permutations : List (List Nat) :=
  [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

def shares_at_least_one_common_entry (a b : List Nat) : Bool :=
  ∃ x : Nat, x ∈ a ∧ x ∈ b

-- Main statement of the problem
theorem number_of_valid_sequences : 
  (number_of_ways_to_arrange_permutations permutations shares_at_least_one_common_entry) = 144 :=
sorry

end number_of_valid_sequences_l474_474555


namespace find_AC_length_l474_474274

noncomputable def problem (A B C D E : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] : Prop :=
  let AC := 4
  ∧ (AC ⊥ AB)
  ∧ (DE ⊥ AB)
  ∧ (BD = 2)
  ∧ (DE = 2)
  ∧ (EC = 2)
  ∧ (D ∈ AC)
  ∧ (E ∈ AB)

theorem find_AC_length : 
  problem A B C D E → AC = 4 := by
  sorry

end find_AC_length_l474_474274


namespace ilya_defeats_dragon_l474_474240

section DragonAndIlya

def Probability (n : ℕ) : Type := ℝ

noncomputable def probability_of_defeat : Probability 3 :=
  let p_no_regrow := 5 / 12
  let p_one_regrow := 1 / 3
  let p_two_regrow := 1 / 4
  -- Assuming recursive relationship develops to eventually reduce heads to zero
  1

-- Prove that the probability_of_defeat is equal to 1
theorem ilya_defeats_dragon : probability_of_defeat = 1 :=
by
  -- Formal proof would be provided here
  sorry

end DragonAndIlya

end ilya_defeats_dragon_l474_474240


namespace num_solutions_sin_3x_eq_cos_2x_l474_474739

theorem num_solutions_sin_3x_eq_cos_2x :
  (set_of (λ x : ℝ, sin (3 * x) = cos (2 * x) ∧ 0 ≤ x ∧ x ≤ 2 * π)).finite.card = 4 :=
sorry

end num_solutions_sin_3x_eq_cos_2x_l474_474739


namespace largest_number_Ahn_can_get_l474_474443

theorem largest_number_Ahn_can_get : 
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 2 * (2/3 * (300 - n)) = 1160 / 3 :=
begin
  sorry
end

end largest_number_Ahn_can_get_l474_474443


namespace total_cost_l474_474281

theorem total_cost (off_the_rack_suit_cost tailored_suit_cost shirt_cost shoe_cost tie_cost 
                   sale_discount tax_rate shipping_fee : ℝ)
                   (h1 : off_the_rack_suit_cost = 300)
                   (h2 : tailored_suit_cost = (3 * off_the_rack_suit_cost) + 200)
                   (h3 : shirt_cost = 80)
                   (h4 : shoe_cost = 120)
                   (h5 : tie_cost = 40)
                   (h6 : sale_discount = 0.1)
                   (h7 : tax_rate = 0.08)
                   (h8 : shipping_fee = 25) :
  let off_the_rack_suit_with_discount := off_the_rack_suit_cost * (1 - sale_discount)
      total_suits_cost := off_the_rack_suit_cost + off_the_rack_suit_with_discount
      total_cost_pre_tax := total_suits_cost + tailored_suit_cost + shirt_cost + shoe_cost + tie_cost
      tax := total_cost_pre_tax * tax_rate
      total_cost_with_tax := total_cost_pre_tax + tax
      total_amount_paid := total_cost_with_tax + shipping_fee
  in total_amount_paid = 2087.80 :=
by sorry

end total_cost_l474_474281


namespace largest_square_side_l474_474797

variable (length width : ℕ)
variable (h_length : length = 54)
variable (h_width : width = 20)
variable (num_squares : ℕ)
variable (h_num_squares : num_squares = 3)

theorem largest_square_side : (length : ℝ) / num_squares = 18 := by
  sorry

end largest_square_side_l474_474797


namespace geometric_relationships_l474_474233

variable (Point : Type) (Line Plane : Type)
variable (a : Line) (α : Plane) (A B : Point)

-- Conditions
variable [has_mem Point Line] [has_mem Point Plane] [has_subset Line Plane]

def point_on_line (A : Point) (a : Line) : Prop := A ∈ a
def line_in_plane (a : Line) (α : Plane) : Prop := a ⊂ α
def point_in_plane (B : Point) (α : Plane) : Prop := B ∈ α

theorem geometric_relationships (h1 : point_on_line A a) (h2 : line_in_plane a α) (h3 : point_in_plane B α) :
  A ∈ a ∧ a ⊂ α ∧ B ∈ α :=
by
  exact ⟨h1, h2, h3⟩

end geometric_relationships_l474_474233


namespace train_speed_kmh_l474_474853

def train_length := 100 -- in meters
def crossing_time := 20 -- in seconds
def conversion_factor := 3.6 -- from m/s to km/h

theorem train_speed_kmh :
  (train_length / crossing_time) * conversion_factor = 18 := by
  sorry

end train_speed_kmh_l474_474853


namespace keith_attended_games_l474_474008

theorem keith_attended_games (total_games : ℕ) (missed_games : ℕ) (total_games_eq : total_games = 8) (missed_games_eq : missed_games = 4) : total_games - missed_games = 4 :=
by
  rw [total_games_eq, missed_games_eq]
  rfl

end keith_attended_games_l474_474008


namespace sufficient_but_not_necessary_l474_474305

theorem sufficient_but_not_necessary (x y : ℝ) (h1 : x ≥ 2) (h2 : y ≥ 2) : (x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧ ¬(x + y ≥ 4 → x ≥ 2 ∧ y ≥ 2) :=
by
  -- Proof for sufficient condition
  have suff : x + y ≥ 4 := by
    linarith,
  -- Proof that this is not a necessary condition
  have not_nec : ¬(x + y ≥ 4 → x ≥ 2 ∧ y ≥ 2) := by
    intro h,
    have example : x = 1 ∧ y = 5 := by
      split; linarith,
    cases example with hx hy,
    exact not_intro (λ _y, hx) (h (by linarith)),
  exact ⟨suff, not_nec⟩,
sorry

end sufficient_but_not_necessary_l474_474305


namespace exponent_of_2_in_1991_m_minus_1_l474_474308

-- Define the exponent of 2 in the prime factorization of a number
def exp_two (n : ℤ) : ℕ := (multiplicity 2 n).get_or_else 0

noncomputable
def prime_exponent_1991_m_minus_1 (m : ℕ) : ℕ :=
  if odd m then 1
  else let k : ℕ := (multiplicity 2 m).get_or_else 0
       in k + 3

theorem exponent_of_2_in_1991_m_minus_1 (m : ℕ) :
  exp_two (1991^m - 1) = prime_exponent_1991_m_minus_1 m :=
sorry

end exponent_of_2_in_1991_m_minus_1_l474_474308


namespace point_not_on_graph_f_at_4_find_x_l474_474196

def f (x : ℝ) : ℝ := (x + 2) / (x - 6)

theorem point_not_on_graph : f 3 ≠ 14 := 
by sorry

theorem f_at_4 : f 4 = -3 := 
by sorry

theorem find_x (x : ℝ) : f x = 2 → x = 14 :=
by sorry

end point_not_on_graph_f_at_4_find_x_l474_474196


namespace determine_M_l474_474877

-- Conditions
def hyperbola1 : Prop := ∀ x y : ℝ, (y^2 / 16 - x^2 / 25 = 1)
def hyperbola2 (M : ℝ) : Prop := ∀ x y : ℝ, (x^2 / 36 - y^2 / M = 1)
def same_asymptotes (M : ℝ) : Prop :=
  let asymptotes1 := (λ x : ℝ, (4 / 5) * x) $\pm (- (4 / 5) * x)
  let asymptotes2 := (λ x : ℝ, (sqrt M / 6) * x) $\pm (- (sqrt M / 6) * x)
  asymptotes1 = asymptotes2

-- The proof problem
theorem determine_M : ∃ M : ℝ, hyperbola1 ∧ hyperbola2 M ∧ same_asymptotes M ∧ M = 576 / 25 :=
by
  sorry

end determine_M_l474_474877


namespace greatest_integer_gcd_6_l474_474764

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l474_474764


namespace Ilya_defeats_dragon_l474_474245

-- Conditions
def prob_two_heads : ℚ := 1 / 4
def prob_one_head : ℚ := 1 / 3
def prob_no_heads : ℚ := 5 / 12

-- Main statement in Lean
theorem Ilya_defeats_dragon : 
  (prob_no_heads + prob_one_head + prob_two_heads = 1) → 
  (∀ n : ℕ, ∃ m : ℕ, m ≤ n) → 
  (∑ n, (prob_no_heads + prob_one_head + prob_two_heads) ^ n) = 1 := 
sorry

end Ilya_defeats_dragon_l474_474245


namespace curve_intersects_every_plane_l474_474709

theorem curve_intersects_every_plane (A B C D : ℝ) (h : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0) :
  ∃ t : ℝ, A * t + B * t^3 + C * t^5 + D = 0 :=
by
  sorry

end curve_intersects_every_plane_l474_474709


namespace coefficient_of_x3_in_expansion_l474_474725

theorem coefficient_of_x3_in_expansion :
  let poly := (fun x : ℤ => (x + (1 / x ^ 2017) + 1) ^ 8) in
  (coefficient poly 3) = 56 :=
by
  sorry

end coefficient_of_x3_in_expansion_l474_474725


namespace sufficient_but_not_necessary_l474_474983

theorem sufficient_but_not_necessary (X Y : ℝ) :
  (X > 2 ∧ Y > 3) → (X + Y > 5 ∧ X * Y > 6) ∧ (¬(X > 2 ∧ Y > 3) → (X + Y > 5 ∧ X * Y > 6)) :=
by
  intros h
  cases h with hx hy
  split
  {
    -- Prove (X + Y > 5 ∧ X * Y > 6)
    sorry
  }
  {
    -- Provide a counterexample
    use 1, 9
    split
    {
      -- 1 + 9 > 5
      sorry
    }
    {
      -- 1 * 9 > 6
      sorry
    }
    {
      -- ¬(1 > 2 ∧ 9 > 3)
      sorry
    }
  }

end sufficient_but_not_necessary_l474_474983


namespace monotone_fn_range_l474_474933

open Function

def f (a : ℝ) (x : ℝ) : ℝ := log a (6 * a * x ^ 2 - 2 * x + 3)

theorem monotone_fn_range (a : ℝ) :
  0 < a ∧ a ≠ 1 ↔ (∀ x y ∈ Icc (3 / 2) 2, x ≤ y → f a x ≤ f a y)
  ↔ a ∈ Ioo (1 / 24) (1 / 12) ∪ Ioi 1 :=
begin
  sorry
end

end monotone_fn_range_l474_474933


namespace pictures_per_album_l474_474335

-- Definitions based on the conditions
def phone_pics := 35
def camera_pics := 5
def total_pics := phone_pics + camera_pics
def albums := 5 

-- Statement that needs to be proven
theorem pictures_per_album : total_pics / albums = 8 := by
  sorry

end pictures_per_album_l474_474335


namespace hyperbola_equation_correct_l474_474514

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :=
  (x y : ℝ) -> (x^2 / 5) - (y^2 / 20) = 1

theorem hyperbola_equation_correct {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :
  hyperbola_equation a b a_pos b_pos focal_len asymptote_slope :=
by {
  sorry
}

end hyperbola_equation_correct_l474_474514


namespace line_circle_intersect_or_tangent_l474_474361

theorem line_circle_intersect_or_tangent (k : ℝ) :
  (∃ (x y : ℝ), x - k * y + 1 = 0 ∧ x^2 + y^2 = 1) ∨ 
  (∃ (x : ℝ), x - k * ((x^2-1)/ (2*k)) + 1 = 0 ∧ x^2 + ((x^2-1)/ (2*k))^2 = 1) :=
begin
  sorry
end

end line_circle_intersect_or_tangent_l474_474361


namespace minimize_travel_distance_l474_474413

theorem minimize_travel_distance (h1_orders h2_orders h3_orders h4_orders h5_orders h6_orders : Nat)
(houses_convex_hexagon : Prop) :
h1_orders = 1 ∧
h2_orders = 2 ∧
h3_orders = 3 ∧
h4_orders = 4 ∧
h5_orders = 5 ∧
h6_orders = 15 ∧
houses_convex_hexagon →
deliver_to_house = 6 :=
by
  intros h1_orders_eq h2_orders_eq h3_orders_eq h4_orders_eq h5_orders_eq h6_orders_eq convex_hexagon
  sorry

end minimize_travel_distance_l474_474413


namespace min_value_fraction_l474_474154

theorem min_value_fraction (x : ℝ) (h : x > 4) : ∃ z : ℝ, z = 2 * Real.sqrt 19 ∧ ∀ y : ℝ, (y > 4) → (frac_expr y) >= z :=
by
  let frac_expr := λ y : ℝ, (y + 15) / Real.sqrt (y - 4)
  sorry

end min_value_fraction_l474_474154


namespace symmetric_point_yaxis_correct_l474_474643

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetric_yaxis (P : Point3D) : Point3D :=
  { x := -P.x, y := P.y, z := P.z }

theorem symmetric_point_yaxis_correct (P : Point3D) (P' : Point3D) :
  P = {x := 1, y := 2, z := -1} → 
  P' = symmetric_yaxis P → 
  P' = {x := -1, y := 2, z := -1} :=
by
  intros hP hP'
  rw [hP] at hP'
  simp [symmetric_yaxis] at hP'
  exact hP'

end symmetric_point_yaxis_correct_l474_474643


namespace symmetric_angle_set_l474_474956

theorem symmetric_angle_set (α β : ℝ) (k : ℤ) 
  (h1 : β = 2 * (k : ℝ) * Real.pi + Real.pi / 12)
  (h2 : α = -Real.pi / 3)
  (symmetric : α + β = -Real.pi / 4) :
  ∃ k : ℤ, β = 2 * (k : ℝ) * Real.pi + Real.pi / 12 :=
sorry

end symmetric_angle_set_l474_474956


namespace rectangle_area_l474_474081

theorem rectangle_area (side_length : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  side_length^2 = 64 → 
  rect_width = side_length →
  rect_length = 3 * rect_width →
  rect_width * rect_length = 192 := 
by
  intros h1 h2 h3
  sorry

end rectangle_area_l474_474081


namespace find_number_l474_474742

noncomputable def isPrime (n : ℕ) : Prop := sorry

theorem find_number (p q r N : ℕ) :
  isPrime p → isPrime q → isPrime r →
  N = p * q * r →
  p^2 + q^2 + r^2 = 2331 →
  (∃ m, m < N → coprime N (m + 1) → m = 7559) →
  (∃ σN : ℕ, σN = (1 + p) * (1 + q) * (1 + r) ∧ σN = 10560) →
  N = 8987 :=
sorry

end find_number_l474_474742


namespace modulo_residue_l474_474884

theorem modulo_residue : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 31 ∧ (-1237 % 31) = x := 
  sorry

end modulo_residue_l474_474884


namespace apple_distribution_correct_l474_474557

def distribute_apples (apples packages : ℕ) (condition : apples = 7 ∧ packages = 4) : ℕ :=
  if condition then
    350  -- we are directly assigning the result as we are not proving here.
  else
    0

theorem apple_distribution_correct : distribute_apples 7 4 (by simp) = 350 := 
by {
  simp [distribute_apples],
  sorry
}

end apple_distribution_correct_l474_474557


namespace find_point_C_l474_474330

noncomputable def point_C := (4, 8) -- Defines the coordinates of point C

theorem find_point_C :
  ∃ (C : ℝ × ℝ), 
    let A := (11, 9),
        B := (-2, 2),
        D := (1, 5),
        C := point_C in       
    -- Midpoint condition
    D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
    -- Isosceles condition
    dist A B = dist A C :=
begin
  use point_C,
  sorry
end

end find_point_C_l474_474330


namespace count_valid_sums_l474_474461

-- Definition of the set
def S : Set ℤ := {2, 5, 8, 11, 14, 17, 20}

-- Predicate to check if a number is a multiple of 4
def is_multiple_of_4 (n : ℤ) : Prop := n % 4 = 0

-- Predicate to check if the sum of two distinct members from the set plus 6 is a multiple of 4
def valid_sum (a b : ℤ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ is_multiple_of_4 (a + b + 6)

-- The proof goal is to show that there are exactly 3 distinct integers that satisfy the valid_sum condition
theorem count_valid_sums : 
  (Finset.univ.filter (λ (pair : ℤ × ℤ), valid_sum pair.1 pair.2)).card = 3 :=
by sorry

end count_valid_sums_l474_474461


namespace number_of_valid_m_values_l474_474319

-- Definitions
def isDivisor (a b : ℕ) := b % a = 0

def countDivisorsGreaterThan (n k : ℕ) : ℕ :=
  (finset.Icc k n).filter (λ d, isDivisor d n).card

-- The main statement
theorem number_of_valid_m_values (m n : ℕ) (h : m * n = 720) (hm_gt : m > 1) (hn_gt : n > 1) :
  countDivisorsGreaterThan 720 1 - 2 = 28 :=
sorry

end number_of_valid_m_values_l474_474319


namespace greatest_integer_gcd_l474_474775

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l474_474775


namespace substitution_not_sufficient_for_identity_proof_l474_474871

theorem substitution_not_sufficient_for_identity_proof {α : Type} (f g : α → α) :
  (∀ x : α, f x = g x) ↔ ¬ (∀ x, f x = g x ↔ (∃ (c : α), f c ≠ g c)) := by
  sorry

end substitution_not_sufficient_for_identity_proof_l474_474871


namespace tangent_slope_of_cubic_l474_474746

theorem tangent_slope_of_cubic (P : ℝ × ℝ) (tangent_at_P : ℝ) (h1 : P.snd = P.fst ^ 3)
  (h2 : tangent_at_P = 3) : P = (1,1) ∨ P = (-1,-1) :=
by
  sorry

end tangent_slope_of_cubic_l474_474746


namespace solve_equation_l474_474127

theorem solve_equation :
  ∃ x : ℚ, 
  let y := (29 / 12 : ℚ) in 
  x = y * y ∧ 
  (Real.sqrt (x : ℝ) + 3 * Real.sqrt ((x : ℝ)^3 + 7 * (x : ℝ)) + Real.sqrt ((x : ℝ) + 7) = 50 - (x : ℝ)^2) := 
by
  sorry

end solve_equation_l474_474127


namespace max_points_of_intersection_l474_474819

-- Definitions based on the conditions in a)
def intersects_circle (l : ℕ) : ℕ := 2 * l  -- Each line intersects the circle at most twice
def intersects_lines (n : ℕ) : ℕ := n * (n - 1) / 2  -- Number of intersection points between lines (combinatorial)

-- The main statement that needs to be proved
theorem max_points_of_intersection (lines circle : ℕ) (h_lines_distinct : lines = 3) (h_no_parallel : ∀ (i j : ℕ), i ≠ j → i < lines → j < lines → true) (h_no_common_point : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(true)) : (intersects_circle lines + intersects_lines lines = 9) := 
  by
    sorry

end max_points_of_intersection_l474_474819


namespace no_non_similar_triangles_with_geometric_angles_l474_474213

theorem no_non_similar_triangles_with_geometric_angles :
  ¬ ∃ (a r : ℤ), 0 < a ∧ 0 < r ∧ a ≠ ar ∧ a ≠ ar^2 ∧ ar ≠ ar^2 ∧
  a + ar + ar^2 = 180 :=
sorry

end no_non_similar_triangles_with_geometric_angles_l474_474213


namespace percent_defective_units_shipped_l474_474267

variable (P : Real)
variable (h1 : 0.07 * P = d)
variable (h2 : 0.0035 * P = s)

theorem percent_defective_units_shipped (h1 : 0.07 * P = d) (h2 : 0.0035 * P = s) : 
  (s / d) * 100 = 5 := sorry

end percent_defective_units_shipped_l474_474267


namespace grandfather_time_difference_l474_474342

-- Definitions based on the conditions
def treadmill_days : ℕ := 4
def miles_per_day : ℕ := 2
def monday_speed : ℕ := 6
def tuesday_speed : ℕ := 3
def wednesday_speed : ℕ := 4
def thursday_speed : ℕ := 3
def walk_speed : ℕ := 3

-- The theorem statement
theorem grandfather_time_difference :
  let monday_time := (miles_per_day : ℚ) / monday_speed
  let tuesday_time := (miles_per_day : ℚ) / tuesday_speed
  let wednesday_time := (miles_per_day : ℚ) / wednesday_speed
  let thursday_time := (miles_per_day : ℚ) / thursday_speed
  let actual_total_time := monday_time + tuesday_time + wednesday_time + thursday_time
  let walk_total_time := (treadmill_days * miles_per_day : ℚ) / walk_speed
  (walk_total_time - actual_total_time) * 60 = 80 := sorry

end grandfather_time_difference_l474_474342


namespace students_exceed_hamsters_l474_474097

-- Definitions corresponding to the problem conditions
def students_per_classroom : ℕ := 20
def hamsters_per_classroom : ℕ := 1
def number_of_classrooms : ℕ := 5

-- Lean 4 statement to express the problem
theorem students_exceed_hamsters :
  (students_per_classroom * number_of_classrooms) - (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end students_exceed_hamsters_l474_474097


namespace connie_marbles_l474_474459

theorem connie_marbles (j c : ℕ) (h1 : j = 498) (h2 : j = c + 175) : c = 323 :=
by
  -- Placeholder for the proof
  sorry

end connie_marbles_l474_474459


namespace area_of_triangles_equal_l474_474122

open Real

variables (P Q R S A C B K L : Type) [metric_space C]
variables [T1_space C] [locally_compact_space C] [compact_space C] [complete_space C]
variables [connected_space C] [nonempty C]
variables (α : ℝ)
variables (PQ_diameter : Prop)
variables (RS_chord_perpendicular_to_PQ_intersecting_A : Prop)
variables (C_on_circle : Prop)
variables (B_inside_with_BC_parallel_PQ_and_BC_eq_RA : Prop)
variables (AK_perpendicular_to_CQ : Prop)
variables (BL_perpendicular_to_CQ : Prop)

-- Diams and cords intersection
noncomputable def PQ_is_diameter : Prop := PQ_diameter
noncomputable def RS_perpendicular_to_PQ_intersect_A : Prop := RS_chord_perpendicular_to_PQ_intersecting_A
noncomputable def C_lies_on_circle : Prop := C_on_circle
noncomputable def B_conditions : Prop := B_inside_with_BC_parallel_PQ_and_BC_eq_RA
noncomputable def AK_is_perpendicular_to_CQ : Prop := AK_perpendicular_to_CQ
noncomputable def BL_is_perpendicular_to_CQ : Prop := BL_perpendicular_to_CQ

theorem area_of_triangles_equal 
  (hPQ : PQ_is_diameter) (hRS : RS_perpendicular_to_PQ_intersect_A)
  (hC : C_lies_on_circle) (hB : B_conditions)
  (hAK : AK_is_perpendicular_to_CQ) (hBL : BL_is_perpendicular_to_CQ) :
  area (triangle A C K) = area (triangle B C L) :=
sorry

end area_of_triangles_equal_l474_474122


namespace min_visible_sum_of_4x4x4_cube_l474_474049

theorem min_visible_sum_of_4x4x4_cube (dice_capacity : ℕ) (opposite_sum : ℕ) (corner_dice edge_dice center_face_dice innermost_dice : ℕ) : 
  dice_capacity = 64 ∧ 
  opposite_sum = 7 ∧ 
  corner_dice = 8 ∧ 
  edge_dice = 24 ∧ 
  center_face_dice = 24 ∧ 
  innermost_dice = 8 → 
  ∃ min_sum, min_sum = 144 := by
  sorry

end min_visible_sum_of_4x4x4_cube_l474_474049


namespace balls_equal_after_operations_l474_474811

-- Define the problem conditions
def children_sit_in_circle (n : ℕ) (balls : ℕ → ℕ) : Prop :=
  ∀ i, even (balls i)

-- Define the operation that each child performs
def operation (balls : ℕ → ℕ) (i : ℕ) : ℕ :=
  let left := balls ((i - 1) % n)
  let right := balls ((i + 1) % n)
  if even (balls i) then
    (balls i) / 2 + left / 2 + right / 2
  else
    (balls i) + 1

-- Define the final state condition
def equal_balls (balls : ℕ → ℕ) : Prop :=
  ∀ i j, balls i = balls j

-- Main theorem statement
theorem balls_equal_after_operations (n : ℕ) (balls : ℕ → ℕ)
  (h : children_sit_in_circle n balls) :
  ∃ k, equal_balls (operation ∘ operation^[k] balls) :=
sorry

end balls_equal_after_operations_l474_474811


namespace count_ordered_quadruples_l474_474680

theorem count_ordered_quadruples :
  let S := {n | 1 ≤ n ∧ n ≤ 100} in
  ∃ (t : Finset (ℕ × ℕ × ℕ × ℕ)), 
  (∀ (quad : ℕ × ℕ × ℕ × ℕ), quad ∈ t ↔ quad.1 ∈ S ∧ 
    quad.2 ∈ S ∧ quad.3 ∈ S ∧ quad.4 ∈ S ∧ 
    quad.1 ≠ quad.2 ∧ quad.1 ≠ quad.3 ∧ quad.1 ≠ quad.4 ∧
    quad.2 ≠ quad.3 ∧ quad.2 ≠ quad.4 ∧ quad.3 ≠ quad.4 ∧
    ((quad.1 ^ 2 + quad.2 ^ 2 + quad.3 ^ 2) * 
    (quad.2 ^ 2 + quad.3 ^ 2 + quad.4 ^ 2) = 
    (quad.1 * quad.2 + quad.2 * quad.3 + quad.3 * quad.4)^2)) ∧
  t.card = 40 :=
by sorry

end count_ordered_quadruples_l474_474680


namespace problem_1_problem_2_l474_474967

variables {a b c x x₁ x₂ : ℝ}

-- Given conditions in the problem
def quadratic_function (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

def quadratic_roots (a b c x₁ x₂ : ℝ) : Prop :=
  a > 0 ∧ 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a ∧
  quadratic_function a b c x₁ - x₁ = 0 ∧
  quadratic_function a b c x₂ - x₂ = 0

-- Proof Problem 1
theorem problem_1 (h : quadratic_roots a b c x₁ x₂) (hx : x ∈ set.Ioo 0 x₁) :
  x < quadratic_function a b c x ∧ quadratic_function a b c x < x₁ :=
sorry

-- Proof Problem 2
theorem problem_2 (h : quadratic_roots a b c x₁ x₂) (h_sym : ∃ x_in : ℝ, quadratic_function a b c x_in = quadratic_function a b c x₁ + quadratic_function a b c x₂) :
  x₁ < x₂ / 2 :=
sorry

end problem_1_problem_2_l474_474967


namespace find_sum_of_coefficients_l474_474845

noncomputable def sum_of_lengths_of_15_gon_in_circle : ℕ :=
  let radius := 15
  let sum_all_lengths := 15 * (2 * radius * (Real.sin (π / 15) + Real.sin (2 * π / 15) + Real.sin (3 * π / 15) + Real.sin (4 * π / 15))) in
  let a := 875
  let b := 0
  let c := 0
  let d := 1
  let e := 1
  a + b + c + d + e

theorem find_sum_of_coefficients :
  sum_of_lengths_of_15_gon_in_circle = 877 :=
by
  sorry

end find_sum_of_coefficients_l474_474845


namespace payment_for_C_l474_474403

/-- Definition of the work rate for A and B -/
def work_rate_A : ℝ := 1 / 6
def work_rate_B : ℝ := 1 / 8

/-- Total work done by A and B together in 3 days -/
def total_work_AB_3_days : ℝ := 3 * (work_rate_A + work_rate_B)

/-- Remaining work done by C -/
def work_done_C : ℝ := 1 - total_work_AB_3_days

/-- Total payment -/
def total_payment : ℝ := 3840

/-- Share of the payment for C -/
def payment_C : ℝ := work_done_C * total_payment

-- Theorem stating the payment for C
theorem payment_for_C : payment_C = 480 := by
  sorry

end payment_for_C_l474_474403


namespace smallest_value_l474_474568

theorem smallest_value (x : ℝ) (h1 : 1 < x) (h2 : x < 2) :
  ∀ y ∈ {x, x^2, 2 * x, sqrt x, 1 / x}, (1 / x) ≤ y := 
sorry

end smallest_value_l474_474568


namespace radius_of_circle_l474_474058

variable (r M N : ℝ)

theorem radius_of_circle (h1 : M = Real.pi * r^2) 
  (h2 : N = 2 * Real.pi * r) 
  (h3 : M / N = 15) : 
  r = 30 :=
sorry

end radius_of_circle_l474_474058


namespace compare_powers_l474_474760

theorem compare_powers : 2^24 < 10^8 ∧ 10^8 < 5^12 :=
by 
  -- proofs omitted
  sorry

end compare_powers_l474_474760


namespace count_primes_with_reversed_prime_l474_474752

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

noncomputable def reverse_two_digit (n : ℕ) : ℕ :=
  let a := n / 10 in
  let b := n % 10 in
  b * 10 + a

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit_prime n ∧ is_two_digit_prime (reverse_two_digit n)

theorem count_primes_with_reversed_prime :
  ∃ l : List ℕ, (∀ n ∈ l, satisfies_condition n) ∧ l.length = 9 :=
begin
  sorry
end

end count_primes_with_reversed_prime_l474_474752


namespace greatest_int_with_conditions_l474_474780

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l474_474780


namespace combination_eight_choose_five_l474_474626

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474626


namespace find_point_P_proof_l474_474507

variable {Point : Type} [AffineSpace Point ℝ]

def O : Point := Origin Point
def A : Point := ⟨1, 3⟩
def B : Point := ⟨3, -1⟩
def P : Point := ⟨7 / 3, 1 / 3⟩

theorem find_point_P_proof :
  exists P : Point,
    (AP (A -ᵥ - O) = 2 • (B -ᵥ P)))
    ∧
    AP = P :=
begin
  sorry
end

end find_point_P_proof_l474_474507


namespace sum_of_first_fifteen_terms_l474_474147

open Nat

-- Define the conditions
def a3 : ℝ := -5
def a5 : ℝ := 2.4

-- The arithmetic progression terms formula
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- The sum of the first n terms formula
def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- The main theorem to prove
theorem sum_of_first_fifteen_terms :
  ∃ (a1 d : ℝ), (a a1 d 3 = a3) ∧ (a a1 d 5 = a5) ∧ (Sn a1 d 15 = 202.5) :=
sorry

end sum_of_first_fifteen_terms_l474_474147


namespace hyperbola_asymptote_solution_l474_474729

theorem hyperbola_asymptote_solution (a b h k : ℝ) (a_pos : a > 0) (b_pos : b > 0) 
  (asymptote_1 : ∀ x : ℝ, y = 3 * x + 2)
  (asymptote_2 : ∀ x : ℝ, y = -3 * x - 4)
  (passes_through : (1, 5))
  (standard_form : ∀ x y : ℝ, (y-k)^2 / a^2 - (x-h)^2 / b^2 = 1) :
  h = -1 → k = -1 → a = 4 * real.sqrt 2 → b = (4 * real.sqrt 2) / 3 →
  a + h = 4 * real.sqrt 2 - 1 :=
sorry

end hyperbola_asymptote_solution_l474_474729


namespace find_range_of_a_l474_474199

noncomputable def f (a x : ℝ) : ℝ := -Real.exp x * (2 * x + 1) - a * x + a

theorem find_range_of_a : ∀ (a : ℝ), a > -1 → 
  (∃! (x0 : ℤ), f a x0 > 0) ↔ a ∈ Icc (-(1/(2 * Real.exp 1))) (-(1 / (Real.exp 1 ^ 2))) :=
by
  sorry

end find_range_of_a_l474_474199


namespace slope_bisects_parallelogram_l474_474112

noncomputable def bisecting_line_slope_coprime_sum
  (vertices : List (ℝ × ℝ))
  (h_vert : vertices = [ (12, 50), (12, 120), (30, 160), (30, 90) ]) : ℕ :=
let m := 5 in
let n := 1 in
m + n

-- We assert the condition for the slope of the line (m/n) bisecting the parallelogram correctly
theorem slope_bisects_parallelogram (vertices : List (ℝ × ℝ))
  (h_vert : vertices = [ (12, 50), (12, 120), (30, 160), (30, 90) ])
  (bisecting_line_slope_coprime_sum vertices h_vert = 6) : True :=
sorry

end slope_bisects_parallelogram_l474_474112


namespace num_valid_polynomials_correct_l474_474215

noncomputable def num_valid_polynomials : ℕ :=
  let P : ℕ → ℕ := λ x, sorry in
  let valid_p : ℕ → Prop := λ n, ∀ x ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ), 0 ≤ P x n ∧ P x n < 120 in
  (Finset.range 6).count valid_p

theorem num_valid_polynomials_correct :
  num_valid_polynomials = 86400000 :=
sorry

end num_valid_polynomials_correct_l474_474215


namespace equation_of_line_l474_474350

theorem equation_of_line (x_intercept slope : ℝ)
  (hx : x_intercept = 2) (hm : slope = 1) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -1 ∧ c = -2 ∧ (∀ x y : ℝ, y = slope * (x - x_intercept) ↔ a * x + b * y + c = 0) := sorry

end equation_of_line_l474_474350


namespace infinitely_many_happy_composite_numbers_l474_474419

theorem infinitely_many_happy_composite_numbers :
  ∃ᶠ (n : ℕ) in at_top, composite n ∧ (¬ prime (2 ^ (2 ^ n) + 1) ∨ ¬ prime (6 ^ (2 ^ n) + 1)) := 
sorry

end infinitely_many_happy_composite_numbers_l474_474419


namespace sin_pi_2alpha_l474_474162

variable (α : ℝ)

axiom sin_diff_pi : sin (α - π / 4) = 3 / 5

theorem sin_pi_2alpha : sin (π + 2 * α) = -7 / 25 :=
by
  have h1 : sin (α - π / 4) = 3 / 5 := sin_diff_pi
  -- additional proof steps would go here
  sorry

end sin_pi_2alpha_l474_474162


namespace expectation_bound_l474_474679

noncomputable def finite_expectation (X : ℝ → ℝ) (f : ℝ → ℝ) :=
  ∫ x in Ioi 0, f x * (X > x) ∂x < ∞

theorem expectation_bound 
  (X Y : ℝ → ℝ) 
  (h_nonneg_X : ∀ x, 0 ≤ X x) 
  (h_nonneg_Y : ∀ y, 0 ≤ Y y)
  (α : ℝ) (β : ℝ) 
  (h_alpha : α > 1) 
  (h_beta : β > 0) 
  (h_prob : ∀ x > 0, ∫ (X > α * x) * (Y ≤ x) ∂x ≤ β * ∫ (X > x) ∂x)
  (f : ℝ → ℝ) 
  (h_f_zero : f 0 = 0)
  (h_f_nonneg : ∀ x, 0 ≤ f x) 
  (h_f_increasing : ∀ x y, x ≤ y → f x ≤ f y) 
  (L_f : ℝ) 
  (h_L_f : L_f = ⨆ x > 0, f (α * x) / f x)
  (h_L_f_bound : L_f < 1 / β)
  (fin_exp_X : finite_expectation X f)
  (fin_exp_Y : finite_expectation Y f) :
  ∫ x in Ioi 0, f x * (X > x) ∂x ≤ (L_f / (1 - β * L_f)) * ∫ y in Ioi 0, f y * (Y > y) ∂y :=
sorry

end expectation_bound_l474_474679


namespace find_max_n_l474_474266

noncomputable def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

theorem find_max_n (a q : ℝ) (n : ℕ) (h1 : 0 < a)
  (h2 : a < (geometric_sequence a q 4)) 
  (h3 : geometric_sequence a q 4 = 1)
  (h4 : 1 < q) :
  n ≤ 7 → (∑ i in finset.range n, (geometric_sequence a q (i + 1) - 1 / geometric_sequence a q (i + 1))) ≤ 0 :=
by
  sorry

end find_max_n_l474_474266


namespace shifted_parabola_eq_l474_474262

-- Definitions
def original_parabola (x y : ℝ) : Prop := y = 3 * x^2

def shifted_origin (x' y' x y : ℝ) : Prop :=
  (x' = x + 1) ∧ (y' = y + 1)

-- Target statement
theorem shifted_parabola_eq : ∀ (x y x' y' : ℝ),
  original_parabola x y →
  shifted_origin x' y' x y →
  y' = 3*(x' - 1)*(x' - 1) + 1 → 
  y = 3*(x + 1)*(x + 1) - 1 :=
by
  intros x y x' y' h_orig h_shifted h_new_eq
  sorry

end shifted_parabola_eq_l474_474262


namespace curve_line_and_circle_l474_474727

theorem curve_line_and_circle : 
  ∀ x y : ℝ, (x^3 + x * y^2 = 2 * x) ↔ (x = 0 ∨ x^2 + y^2 = 2) :=
by
  sorry

end curve_line_and_circle_l474_474727


namespace price_decrease_approx_l474_474402

def original_price : ℝ := 74.95
def sale_price : ℝ := 59.95
def amount_of_decrease : ℝ := original_price - sale_price
def percentage_decrease : ℝ := (amount_of_decrease / original_price) * 100

theorem price_decrease_approx : percentage_decrease ≈ 20 :=
by 
  sorry

end price_decrease_approx_l474_474402


namespace comb_8_5_eq_56_l474_474600

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474600


namespace find_number_smaller_than_neg_3_l474_474091

theorem find_number_smaller_than_neg_3 : (-\pi < -3 ∧ -\sqrt{2} > -3 ∧ 1 > -3 ∧ 0 > -3) :=
by
  -- We will skip the detailed proofs with 'sorry' to match the requirements.
  sorry

end find_number_smaller_than_neg_3_l474_474091


namespace sum_of_digits_l474_474374

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 4 + 258 = 7 * 100 + b * 10 + 2) (h2 : (7 * 100 + b * 10 + 2) % 3 = 0) :
  a + b = 4 :=
sorry

end sum_of_digits_l474_474374


namespace chord_midpoint_l474_474995

theorem chord_midpoint (P : (ℝ × ℝ)) (hP : P = (1, 1))
  (hC : ∃ C : ℝ × ℝ, C = (3, 0))
  (hCircle : ∃ k : ℝ, ∃ t : ℝ, (k, t) = P ∧ (k - 3)^2 + t^2 = 9)
  (hMidpoint : ∃ M N : ℝ × ℝ, P = ((fst M + fst N) / 2, (snd M + snd N) / 2)) :
  ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -1 ∧ (∀ x y : ℝ, y = 2 * x - 1 ↔ 2 * x - y - 1 = 0) :=
sorry

end chord_midpoint_l474_474995


namespace find_a5_l474_474642

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1/3) ∧ (∀ n, n ≥ 2 → a n = (-1)^n * 2 * a (n - 1))

theorem find_a5 : ∃ a : ℕ → ℚ, sequence a ∧ a 5 = 16/3 :=
by
  sorry

end find_a5_l474_474642


namespace polygon_sides_l474_474750

theorem polygon_sides (sum_missing_angles sum_known_angles total_sum : ℝ) (h1: sum_missing_angles = 120)
  (h2: sum_known_angles = 3240) (h3: total_sum = sum_missing_angles + sum_known_angles):
  ∃ n : ℕ, 180 * (n - 2) = total_sum ∧ n = 20 :=
by 
  have h4: total_sum = 3360 := by
    rw [h1, h2, h3]
  have h5: ∀ n, 180 * (n - 2) = 3360 → n = 20 := by
    intro n hn
    have hn_eq: n - 2 = 3360 / 180 := (eq_div_iff (by norm_num : (180 : ℝ) ≠ 0)).mp hn
    have hn_val: (n : ℝ) = 20 := by
      linarith
    exact nat.coe_nat_inj.mp hn_val

  use 20
  constructor
  { exact h5 20 (by rw [h4]; norm_num) }
  { exact rfl }

end polygon_sides_l474_474750


namespace angle_at_3_30_l474_474448

def h : ℕ := 3
def m : ℕ := 30
def angle (h m : ℕ) : ℝ := |(60 * h - 11 * m) / 2|

theorem angle_at_3_30 : angle 3 30 = 75 :=
by
  sorry

end angle_at_3_30_l474_474448


namespace required_run_rate_l474_474420

/-- 
Prove that the required run rate in the remaining 40 overs to reach the target 
of 252 runs is 5.5 runs/over, given that the run rate in the first 10 overs was 3.2 
and the target is 252 runs. 
-/
theorem required_run_rate (run_rate_10_overs : ℝ) (target : ℝ) (overs_played : ℝ) (overs_remaining : ℝ) 
  (runs_scored : ℝ) (required_run_rate : ℝ):
  run_rate_10_overs = 3.2 →
  target = 252 →
  overs_played = 10 →
  overs_remaining = 40 →
  runs_scored = run_rate_10_overs * overs_played →
  required_run_rate = (target - runs_scored) / overs_remaining →
  required_run_rate = 5.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at *
  rw [h5] at h6
  exact h6

end required_run_rate_l474_474420


namespace fraction_multiplication_l474_474784

theorem fraction_multiplication (a b c d e f : ℚ) : 
  a = 3 / 4 → b = 1 / 2 → c = 2 / 5 → d = 5020 → e = 753 → 
  ((a * b * c) * d = e) :=
by
  intros h1 h2 h3 h4 h5
  calc (a * b * c) * d = e : sorry

end fraction_multiplication_l474_474784


namespace sum_gcd_twice_lcm_eq_93_l474_474785

-- Definitions required by the conditions
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem sum_gcd_twice_lcm_eq_93 :
  let a := 15
  let b := 9
  gcd a b + 2 * lcm a b = 93 := by
{
  -- Definitions of a and b
  let a := 15
  let b := 9

  -- Statement about the greatest common divisor
  have gcd_ab : gcd a b = 3 := by sorry

  -- Statement about the least common multiple
  have lcm_ab : lcm a b = 45 := by sorry
  
  -- Concluding the proof
  calc
    gcd a b + 2 * lcm a b = 3 + 2 * 45 : by rw [gcd_ab, lcm_ab]
    ... = 3 + 90         : by rfl
    ... = 93             : by rfl
}

end sum_gcd_twice_lcm_eq_93_l474_474785


namespace irrational_sqrt_sine_cosine_l474_474152

theorem irrational_sqrt_sine_cosine (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) :
  ¬ (∃ (a b c d : ℤ), b ≠ 0 ∧ d ≠ 0 ∧ 
        ((√(Real.sin θ) = a / b) ∧ (√(Real.cos θ) = c / d))) :=
by
  sorry

end irrational_sqrt_sine_cosine_l474_474152


namespace log_b_243_values_l474_474216

theorem log_b_243_values : 
  ∃! (s : Finset ℕ), (∀ b ∈ s, ∃ n : ℕ, b^n = 243) ∧ s.card = 2 :=
by 
  sorry

end log_b_243_values_l474_474216


namespace num_5_digit_even_div_by_5_l474_474551

theorem num_5_digit_even_div_by_5 : ∃! (n : ℕ), n = 500 ∧ ∀ (d : ℕ), 
  10000 ≤ d ∧ d ≤ 99999 → 
  (∀ i, i ∈ [0, 1, 2, 3, 4] → ((d / 10^i) % 10) % 2 = 0) ∧
  (d % 10 = 0) → 
  n = 500 := sorry

end num_5_digit_even_div_by_5_l474_474551


namespace travis_apples_l474_474012

theorem travis_apples
  (price_per_box : ℕ)
  (num_apples_per_box : ℕ)
  (total_money : ℕ)
  (total_boxes : ℕ)
  (total_apples : ℕ)
  (h1 : price_per_box = 35)
  (h2 : num_apples_per_box = 50)
  (h3 : total_money = 7000)
  (h4 : total_boxes = total_money / price_per_box)
  (h5 : total_apples = total_boxes * num_apples_per_box) :
  total_apples = 10000 :=
sorry

end travis_apples_l474_474012


namespace distribute_people_across_boats_l474_474587
-- Import necessary libraries

-- Define capacities of boats
def capacities := [3, 5, 7, 4, 6]

-- Define the total number of people
def total_people := 15

-- Define the distribution of people across the boats
def distribution := [3, 5, 7, 0, 0]

-- Statement of the proof problem in Lean
theorem distribute_people_across_boats :
  (∀ i < capacities.length, distribution[i] ≤ capacities[i]) ∧ (distribution.sum = total_people) :=
  by
    -- This is just a placeholder proof, implement the actual proof here
    sorry

end distribute_people_across_boats_l474_474587


namespace length_DE_is_6_radius_of_circumcircle_ADEC_is_3sqrt65_l474_474275

-- Definitions of the geometry involved
structure Triangle (α : Type) [LinearOrder α] :=
(A B C : α)

structure Segment (α : Type) [LinearOrder α] :=
(P Q : α)

noncomputable def median_of_triangle [LinearOrder α] (T : Triangle α) : Segment α := sorry
noncomputable def angle_bisector [LinearOrder α] (S : Segment α) (P : α) : Segment α := sorry
noncomputable def intersection_point [LinearOrder α] (S1 S2 : Segment α) : α := sorry
noncomputable def circumradius_quadrilateral [LinearOrder α] (A B C D : α) : α := sorry

-- Given conditions
variable {α : Type} [LinearOrder α]

variable (A B C M D E P : α)
variable (BP MP : ℕ)

def triangle_ABC := Triangle.mk A B C
def median_BM := median_of_triangle triangle_ABC

def segment_BM := Segment.mk B M
def segment_MD := angle_bisector segment_BM A
def segment_ME := angle_bisector segment_BM C
def segment_DE := Segment.mk D E
def intersection_P := intersection_point segment_BM segment_DE

-- Assumptions
axiom BP_val : BP = 1
axiom MP_val : MP = 3

-- Theorem statements
theorem length_DE_is_6 : length segment_DE = 6 := sorry

theorem radius_of_circumcircle_ADEC_is_3sqrt65 : circumradius_quadrilateral A D E C = 3 * sqrt 65 := sorry

end length_DE_is_6_radius_of_circumcircle_ADEC_is_3sqrt65_l474_474275


namespace total_interest_10_years_l474_474037

-- Principal and Rate of interest
variables (P R : ℝ)
-- Simple Interest formula
def simple_interest (P R T : ℝ) := (P * R * T) / 100

-- Conditions given in the problem
axiom condition1 : simple_interest P R 10 = 1400
axiom condition2 : ∀ P, simple_interest (3 * P) R 5 = 210

-- Prove the total interest at the end of the tenth year
theorem total_interest_10_years : simple_interest P R 5 + simple_interest (3 * P) R 5 = 280 := sorry

end total_interest_10_years_l474_474037


namespace angle_between_hands_at_3_27_l474_474392

noncomputable def minute_hand_angle (m : ℕ) : ℝ :=
  (m / 60.0) * 360.0

noncomputable def hour_hand_angle (h : ℕ) (m : ℕ) : ℝ :=
  ((h + m / 60.0) / 12.0) * 360.0

theorem angle_between_hands_at_3_27 : 
  minute_hand_angle 27 - hour_hand_angle 3 27 = 58.5 :=
by
  rw [minute_hand_angle, hour_hand_angle]
  simp
  sorry

end angle_between_hands_at_3_27_l474_474392


namespace balls_in_boxes_l474_474556

theorem balls_in_boxes :
  let balls := 6
  let boxes := 4
  let num_ways := 84
  0 ≤ balls → 0 ≤ boxes →
  (exists (p : List ℕ), p.sum = 6 ∧ p.length = 4 ∧ (multiset.card (multiset.map multiset.card (multiset.powerset (finset.range (balls + boxes))) = num_ways)) := 
  sorry

end balls_in_boxes_l474_474556


namespace complex_product_correct_l474_474362

theorem complex_product_correct :
  (let i : ℂ := complex.I in
  let z : ℂ := -1 + 3 * i in
  z * i = -3 - i) :=
by
let i : ℂ := complex.I
have h₁ : i^2 = -1 := complex.I_mul_I
have z : ℂ := -1 + 3 * i
have h₂ : z * i = (-1 + 3 * i) * i := rfl
calc
  z * i = (-1 + 3 * i) * i : by rw h₂
    ... = (-1) * i + (3 * i) * i : by rw complex.add_mul
    ... = -i + 3 * (i * i) : by rw [complex.mul_assoc, complex.mul_add, complex.one_mul, complex.mul_comm]
    ... = -i + 3 * (-1) : by rw h₁
    ... = -3 - i : by ring

end complex_product_correct_l474_474362


namespace hyperbola_properties_l474_474935

open Real

noncomputable def hyperbola_eq (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

noncomputable def asymptotes_eq (a b : ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → (y = (b / a) * x) ∨ (y = -(b / a) * x)

noncomputable def line_eq (t : ℝ) : Prop :=
  ∀ (x y : ℝ), y = x + t

theorem hyperbola_properties (a b : ℝ) :
  hyperbola_eq (7/2) (sqrt (33/4)) →
  asymptotes_eq (7/2) (sqrt (33/4)) →
  ∃ (t : ℝ), overline_OA_perp_overline_OB (line_eq t) :=
by
  sorry

end hyperbola_properties_l474_474935


namespace choose_five_from_eight_l474_474607

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474607


namespace intersection_M_N_l474_474561

def M : Set ℝ := { x | sqrt x < 2 }
def N : Set ℝ := { x | 3 * x ≥ 1 }

theorem intersection_M_N : M ∩ N = { x | (1 / 3 : ℝ) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l474_474561


namespace projection_is_correct_l474_474022

theorem projection_is_correct :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (4, -1)
  let p : ℝ × ℝ := (15/58, 35/58)
  let d : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)
  ∃ v : ℝ × ℝ, 
    (a.1 * v.1 + a.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧
    (b.1 * v.1 + b.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧ 
    (p.1 * d.1 + p.2 * d.2 = 0) :=
sorry

end projection_is_correct_l474_474022


namespace sin_eq_sin_sin_unique_solution_l474_474978

noncomputable def numberOfSolutions : ℝ := 1

theorem sin_eq_sin_sin_unique_solution :
  set.count ({ x : ℝ | x ∈ set.Icc 0 (real.arcsin 0.9) ∧ real.sin x = real.sin (real.sin x) }) = numberOfSolutions :=
by
  sorry

end sin_eq_sin_sin_unique_solution_l474_474978


namespace range_of_a_l474_474932

variable (x a : ℝ)

def P := (2 * x - 1) / (x - 1) ≤ 0
def Q := x^2 - (2 * a + 1) * x + a * (a + 1) < 0

theorem range_of_a :
  (∀ x, P x → Q x) ∧ (∃ x, ¬(P x) ∧ Q x) →
  a ∈ Set.Ico 0 (1 / 2) :=
by
  sorry

end range_of_a_l474_474932


namespace prime_pairs_solution_l474_474034

theorem prime_pairs_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  7 * p * q^2 + p = q^3 + 43 * p^3 + 1 ↔ (p = 2 ∧ q = 7) :=
by
  sorry

end prime_pairs_solution_l474_474034


namespace problem_part1_problem_part2_l474_474271

variables (a b c A B C : ℝ)
variables (m n : ℝ × ℝ) 

-- Conditions
def m_condition : m = (b, -real.sqrt 3 * a) := by sorry
def n_condition : n = (real.cos A, real.sin B) := by sorry
def perpendicular_condition : (m.fst * n.fst + m.snd * n.snd) = 0 := by sorry
def angle_relation : B + real.pi / 12 = A := by sorry
def side_a : a = 2 := by sorry

-- Questions
def find_A :=
  A = real.pi / 6

def find_area :=
  (1 / 2) * a * c * real.sin B = (real.sqrt 3 - 1)

-- Proof statements
theorem problem_part1 : 
  m_condition ∧ n_condition ∧ perpendicular_condition → find_A :=
by sorry

theorem problem_part2 :
  m_condition ∧ n_condition ∧ perpendicular_condition ∧ (find_A) ∧ side_a ∧ angle_relation → find_area :=
by sorry

end problem_part1_problem_part2_l474_474271


namespace greatest_int_with_conditions_l474_474778

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l474_474778


namespace number_of_books_to_break_even_is_4074_l474_474849

-- Definitions from problem conditions
def fixed_costs : ℝ := 35630
def variable_cost_per_book : ℝ := 11.50
def selling_price_per_book : ℝ := 20.25

-- The target number of books to sell for break-even
def target_books_to_break_even : ℕ := 4074

-- Lean statement to prove that number of books to break even is 4074
theorem number_of_books_to_break_even_is_4074 :
  let total_costs (x : ℝ) := fixed_costs + variable_cost_per_book * x
  let total_revenue (x : ℝ) := selling_price_per_book * x
  ∃ x : ℝ, total_costs x = total_revenue x → x = target_books_to_break_even := by
  sorry

end number_of_books_to_break_even_is_4074_l474_474849


namespace solve_equation1_solve_equation2_l474_474720

noncomputable def solutions_equation1 : Set ℝ := { x | x^2 - 2 * x - 8 = 0 }
noncomputable def solutions_equation2 : Set ℝ := { x | x^2 - 2 * x - 5 = 0 }

theorem solve_equation1 :
  solutions_equation1 = {4, -2} := 
by
  sorry

theorem solve_equation2 :
  solutions_equation2 = {1 + Real.sqrt 6, 1 - Real.sqrt 6} :=
by
  sorry

end solve_equation1_solve_equation2_l474_474720


namespace wire_cut_ratio_l474_474439

theorem wire_cut_ratio (p q : ℝ) (h1: p > 0) (h2: q > 0) 
  (h3: (p:ℝ) = 4 * √((π * q^2) / 4) ) (h4: (q:ℝ) = 2 * π * √((p^2) / (16 * π))) :
  p / q = 4 / Real.sqrt π := by
  sorry

end wire_cut_ratio_l474_474439


namespace mass_percentage_Al_in_Al2S3_l474_474907

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_S : ℝ := 32.06
noncomputable def moles_Al_in_Al2S3 : ℝ := 2
noncomputable def moles_S_in_Al2S3 : ℝ := 3

theorem mass_percentage_Al_in_Al2S3 : 
  let total_mass_Al := moles_Al_in_Al2S3 * molar_mass_Al in
  let total_mass_S := moles_S_in_Al2S3 * molar_mass_S in
  let molar_mass_Al2S3 := total_mass_Al + total_mass_S in
  (total_mass_Al / molar_mass_Al2S3) * 100 ≈ 35.95 :=
by
  sorry

end mass_percentage_Al_in_Al2S3_l474_474907


namespace defeat_dragon_probability_l474_474248

noncomputable theory

def p_two_heads_grow : ℝ := 1 / 4
def p_one_head_grows : ℝ := 1 / 3
def p_no_heads_grow : ℝ := 5 / 12

-- We state the probability that Ilya will eventually defeat the dragon
theorem defeat_dragon_probability : 
  ∀ (expected_value : ℝ), 
  (expected_value = p_two_heads_grow * 2 + p_one_head_grows * 1 + p_no_heads_grow * 0) →
  expected_value < 1 →
  prob_defeat (count_heads n : ℕ) > 0 :=
by
  sorry

end defeat_dragon_probability_l474_474248


namespace four_digit_odd_even_div_by_5_l474_474207

theorem four_digit_odd_even_div_by_5 : 
  let is_even := λ n : ℕ, n % 2 = 0
  let digits_even := [0, 2, 4, 6, 8]
  let digits_odd := [1, 3, 5, 7, 9]
  in ∃ (a b c d : ℕ), 
    (1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧ 1000 * a + 100 * b + 10 * c + d ≤ 9999) ∧
    ( (∀ x ∈ [a, b, c, d], is_even x) ∨ (∀ x ∈ [a, b, c, d], ¬ is_even x) ) ∧
    (1000 * a + 100 * b + 10 * c + d % 5 = 0) ∧
    (4 * 5 * 5 * 1 = 100) :=
by
  sorry

end four_digit_odd_even_div_by_5_l474_474207


namespace proof_problem_l474_474928

variable (a b : ℝ)

theorem proof_problem (h1: a > 0) (h2: exp(a) + log(b) = 1) :
  a + log(b) < 0 ∧
  exp(a) + b > 2 ∧
  log(a) + exp(b) ≥ 0 ∧
  a + b > 1 :=
by {
  sorry
}

end proof_problem_l474_474928


namespace log_product_arithmetic_sequence_l474_474634

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := sorry

def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x^2 + 6 * x - 1

def is_extreme_point (x : ℝ) : Prop := f.derivative x = 0

theorem log_product_arithmetic_sequence :
  ∀ (a_n : ℕ → ℝ),
    (∀ n, a_n n = arithmetic_sequence n) →
    is_extreme_point (a_n 2) →
    is_extreme_point (a_n 4032) →
    a_n 2 + a_n 4032 = 2 * a_n 2017 →
    a_n 2 * a_n 4032 = 6 →
    log 2 (a_n 2 * a_n 2017 * a_n 4032) = 3 + log 2 3 := sorry

end log_product_arithmetic_sequence_l474_474634


namespace cost_one_dozen_pens_l474_474806

variable (cost_of_pen cost_of_pencil : ℝ)
variable (ratio : ℝ)
variable (dozen_pens_cost : ℝ)

axiom cost_equation : 3 * cost_of_pen + 5 * cost_of_pencil = 200
axiom ratio_pen_pencil : cost_of_pen = 5 * cost_of_pencil

theorem cost_one_dozen_pens : dozen_pens_cost = 12 * cost_of_pen := 
  by
    sorry

end cost_one_dozen_pens_l474_474806


namespace problem_l474_474203

noncomputable def sequence : ℕ → ℚ
| 0 := a₀
| 1 := a₁
| 2 := a₂
| 3 := a₃
| 4 := 1 / 8
| (n + 5) := sequence(n + 4) + 10 * 3^n

theorem problem (a : ℕ → ℚ)
  (h₁ : a 4 = 1 / 8)
  (h₂ : ∀ n : ℕ, a (n + 2) - a n ≤ 3^n)
  (h₃ : ∀ n : ℕ, a (n + 4) - a n ≥ 10 * 3^n) :
  a 2016 = (81^504 - 80) / 8 := 
  sorry

end problem_l474_474203


namespace hyperbola_eccentricity_l474_474951

/-- Given the parabola x^2 = 4y with focus F₁ and directrix intersecting point F₂,
    and a tangent line to the parabola passing through F₂ with point of tangency A,
    where A lies on the hyperbola with foci F₁ and F₂, show that the eccentricity 
    of the hyperbola is √2 + 1. -/
theorem hyperbola_eccentricity (F₁ F₂ A : Point) :
  parabola F₁ ("x^2 = 4y") → 
  directrix_symm_point F₁ F₂ → 
  tangent_through_directrix F₁ F₂ A → 
  lies_on_hyperbola A F₁ F₂ →
  hyperbola_eccentricity A F₁ F₂ = √2 + 1 :=
by
  sorry

end hyperbola_eccentricity_l474_474951


namespace people_eat_only_vegetarian_l474_474586

def number_of_people_eat_only_veg (total_veg : ℕ) (both_veg_nonveg : ℕ) : ℕ :=
  total_veg - both_veg_nonveg

theorem people_eat_only_vegetarian
  (total_veg : ℕ) (both_veg_nonveg : ℕ)
  (h1 : total_veg = 28)
  (h2 : both_veg_nonveg = 12)
  : number_of_people_eat_only_veg total_veg both_veg_nonveg = 16 := by
  sorry

end people_eat_only_vegetarian_l474_474586


namespace regular_tetrahedron_volume_approx_l474_474918

noncomputable def regular_tetrahedron_volume (a : ℝ) : ℝ :=
  (a^3) / (6 * Real.sqrt 2)

theorem regular_tetrahedron_volume_approx (a : ℝ)
    (h : ℝ) (H_mid_face : 2 = h)
    (H_mid_edge : Real.sqrt 10 = a / Real.sqrt 2) :
    abs (regular_tetrahedron_volume a - 309.84) < 0.01 :=
begin
  sorry
end

end regular_tetrahedron_volume_approx_l474_474918


namespace production_line_B_units_l474_474831

theorem production_line_B_units (total_units : ℕ) 
  (lines : ℕ) (h_total: total_units = 16800)
  (h_arithmetic_sequence: ∃ (a d : ℕ), lines = [a, a + d, a + 2 * d]) :
  ∃ (units_B : ℕ), units_B = 5600 :=
by
  -- Introduce the assumptions
  cases h_arithmetic_sequence with a ha
  cases ha with d hd
  -- Assume the total number of units equation
  have h_units : total_units = a + (a + d) + (a + 2 * d) := sorry
  -- Solve for the parameter values
  find a, d such that the sum matches
  sorry
  -- Derive units_B value
  let units_B := a + d
  use units_B
  -- Conclude that units_B = 5600
  have h_units_B_correct : units_B = 5600 := sorry
  exact h_units_B_correct

end production_line_B_units_l474_474831


namespace negation_of_proposition_l474_474711

theorem negation_of_proposition (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := 
by {
  assume h : ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0,
  cases h with x hx,
  let lhs := (x + 1)^2,
  have h1 : lhs ≥ 0 := by apply pow_two_nonneg (x + 1),
  calc
    x^2 + 2*x + 2 = (x + 1)^2 + 1 : by ring
    ... ≥ 0 + 1   : by exact add_le_add h1 (by linarith)
    ... = 1       : by ring
,
  linarith
}

end negation_of_proposition_l474_474711


namespace geometric_sequence_max_value_ad_l474_474516

theorem geometric_sequence_max_value_ad (a b c d : ℝ)
  (h1 : ∃ r : ℝ, a = r * b ∧ b = r * c ∧ c = r * d)
  (h2 : ∀ x, (ln x - x) ≤ c)
  (h3 : (ln b - b) = c) :
  a * d = -1 :=
sorry

end geometric_sequence_max_value_ad_l474_474516


namespace find_stickers_before_birthday_l474_474647

variable (stickers_received : ℕ) (total_stickers : ℕ)

def stickers_before_birthday (stickers_received total_stickers : ℕ) : ℕ :=
  total_stickers - stickers_received

theorem find_stickers_before_birthday (h1 : stickers_received = 22) (h2 : total_stickers = 61) : 
  stickers_before_birthday stickers_received total_stickers = 39 :=
by 
  have h1 : stickers_received = 22 := h1
  have h2 : total_stickers = 61 := h2
  rw [h1, h2]
  rfl

end find_stickers_before_birthday_l474_474647


namespace digits_of_fraction_l474_474552

noncomputable def number_of_decimal_digits (x : ℕ) : ℕ :=
  if h : (x ≠ 0) then
    let d := real.log x / real.log 10 in
    d.to_nat
  else 0

theorem digits_of_fraction :
  number_of_decimal_digits ((5 : ℕ)^7) / (8^3 * 125^2) = 6 :=
by
  sorry

end digits_of_fraction_l474_474552


namespace collinear_points_of_tangency_and_parallel_diameters_l474_474238

noncomputable theory

open EuclideanGeometry

theorem collinear_points_of_tangency_and_parallel_diameters 
  (O O1 A B C D E : Point)
  (circle1 : Circle O (dist O A))
  (circle2 : Circle O1 (dist O1 C))
  (h1 : Circle Touched circle1 circle2 E)
  (h2 : Line (segment A B) || Line (segment C D)) :
  Collinear {A, E, D} :=
sorry

end collinear_points_of_tangency_and_parallel_diameters_l474_474238


namespace number_of_tulips_l474_474646

-- Define the conditions
variables (T : ℕ) -- Number of tulips
constants (carnations roses : ℕ)
#check carnations
axiom carnations_eq : carnations = 375
axiom roses_eq : roses = 320
axiom flower_cost : ℕ → ℕ
axiom flower_cost_eq : ∀ n, flower_cost n = 2 * n
axiom total_expenses : ℕ
axiom total_expenses_eq : total_expenses = 1890

-- The main statement
theorem number_of_tulips :
  flower_cost T + flower_cost carnations + flower_cost roses = total_expenses → T = 250 :=
begin
  intros h,
  simp [flower_cost_eq, carnations_eq, roses_eq] at h,
  sorry
end

end number_of_tulips_l474_474646


namespace sin_pi_over_4_l474_474640

theorem sin_pi_over_4 (α : ℝ) (hα1 : sin α = 4/5) (hα2 : cos α = -3/5) :
  sin (α + real.pi / 4) = real.sqrt 2 / 10 :=
by
  sorry

end sin_pi_over_4_l474_474640


namespace bridge_length_l474_474434

-- Definitions of conditions
def train_length : ℝ := 360 -- in meters
def train_speed : ℝ := 75 * (1000 / 3600) -- convert 75 km/h to m/s
def passing_time : ℝ := 24 -- in seconds

-- The theorem to prove
theorem bridge_length : (train_speed * passing_time) - train_length = 140 := by
  sorry

end bridge_length_l474_474434


namespace comb_8_5_eq_56_l474_474597

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474597


namespace original_cost_of_milk_is_3_l474_474066

variable (M C : ℝ)

-- Conditions
axiom current_price_of_milk : M - 2
axiom cereal_discount : C - 1
axiom savings_condition : 3 * (M - 2) + 5 = 8

-- Proof statement
theorem original_cost_of_milk_is_3
  (h1 : current_price_of_milk = 0)
  (h2 : cereal_discount = C - 1)
  (h3 : savings_condition = 8) :
  M = 3 :=
  sorry

end original_cost_of_milk_is_3_l474_474066


namespace eighth_finger_is_six_l474_474327

noncomputable def g : ℕ → ℕ :=
  λ x, match x with
  | 0 => 8
  | 1 => 7
  | 2 => 6
  | 3 => 5
  | 4 => 4
  | 5 => 3
  | 6 => 2
  | 7 => 1
  | 8 => 0
  | _ => 0  -- Assuming g is not defined for other values for simplicity.

def eighth_finger_value : ℕ :=
  g (g (g (g (g (g (g 2))))))

theorem eighth_finger_is_six : eighth_finger_value = 6 :=
  sorry

end eighth_finger_is_six_l474_474327


namespace chandra_pairings_l474_474105

variable (bowls : ℕ) (glasses : ℕ)

theorem chandra_pairings : 
  bowls = 5 → 
  glasses = 4 → 
  bowls * glasses = 20 :=
by intros; 
    sorry

end chandra_pairings_l474_474105


namespace cosine_cubed_decomposition_l474_474006

theorem cosine_cubed_decomposition (b1 b2 b3 : ℝ)
  (h : ∀ θ : ℝ, cos θ ^ 3 = b1 * cos θ + b2 * cos (2 * θ) + b3 * cos (3 * θ)) :
  b1 ^ 2 + b2 ^ 2 + b3 ^ 2 = 5 / 8 :=
sorry

end cosine_cubed_decomposition_l474_474006


namespace exercise_books_l474_474028

/-- Given conditions:
    1. 250 yuan total
    2. 100 exercise books total
    3. Each Chinese exercise book costs 2 yuan
    4. Each English exercise book costs 4 yuan
-/
theorem exercise_books : 
  ∃ (chinese_books english_books : ℕ), 
    chinese_books + english_books = 100 ∧ 
    2 * chinese_books + 4 * english_books = 250 ∧ 
    chinese_books = 75 ∧ 
    english_books = 25 :=
begin
  use [75, 25],
  simp [*],
  sorry
end

end exercise_books_l474_474028


namespace cost_price_of_article_l474_474804

theorem cost_price_of_article :
  ∃ (C : ℝ), 
  (∃ (G : ℝ), C + G = 500 ∧ C + 1.15 * G = 570) ∧ 
  C = (100 / 3) :=
by sorry

end cost_price_of_article_l474_474804


namespace fill_time_eight_faucets_l474_474159

theorem fill_time_eight_faucets (r : ℝ) (h1 : 4 * r * 8 = 150) :
  8 * r * (50 / (8 * r)) * 60 = 80 := by
  sorry

end fill_time_eight_faucets_l474_474159


namespace max_sum_of_digits_between_12_23_00_59_l474_474063

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

theorem max_sum_of_digits_between_12_23_00_59 : 
  ∀ (h m : Nat), (12 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60) → 
  sum_of_digits h + sum_of_digits m ≤ 24 :=
begin
  intros h m,
  sorry
end

end max_sum_of_digits_between_12_23_00_59_l474_474063


namespace existence_of_f_g_condition1_existence_of_f_g_condition2_l474_474041

-- Definitions
def f_condition1 (f g : ℤ → ℤ) := ∀ x : ℤ, f (f x) = x ∧ g (g x) = x ∧ f (g x) > x ∧ g (f x) > x
def f_condition2 (f g : ℤ) := ∀ x : ℤ, f (f x) < x ∧ g (g x) < x ∧ f (g x) > x ∧ g (f x) > x

-- Problem statement
theorem existence_of_f_g_condition1 : ¬ ∃ (f g : ℤ → ℤ), f_condition1 f g :=
by
  sorry

theorem existence_of_f_g_condition2 : ∃ (f g : ℤ → ℤ), f_condition2 f g :=
by
  let f : ℤ → ℤ := λ x, if x % 2 = 0 then -2 * |x| - 2 else 2 * |x| + 2
  let g : ℤ → ℤ := λ x, if x % 2 = 0 then 2 * |x| + 1 else -2 * |x| - 1
  use [f, g]
  sorry

end existence_of_f_g_condition1_existence_of_f_g_condition2_l474_474041


namespace min_distinct_differences_l474_474307

-- Definitions based on conditions
def distinct_positive_integers (s : Finset ℕ) := s.card = 20 ∧ ∀ a ∈ s, ∀ b ∈ s, a ≠ b → 0 < a ∧ 0 < b

def sumset_contains_201_elements (s : Finset ℕ) :=
  (s.product s).image (λ (p : ℕ × ℕ), p.1 + p.2) .card = 201

def differences (s : Finset ℕ) :=
  (s.product s).filter (λ (p : ℕ × ℕ), p.1 ≠ p.2) .image (λ (p : ℕ × ℕ), (p.1 - p.2).natAbs)

-- Theorem that needs to be proven
theorem min_distinct_differences (s : Finset ℕ) :
  distinct_positive_integers s →
  sumset_contains_201_elements s →
  100 ≤ (differences s).card :=
begin
  assume h_distinct h_sumset,
  sorry
end

end min_distinct_differences_l474_474307


namespace locus_of_P_is_ellipse_locus_of_P_is_hyperbola_l474_474533

variable {A O B P : Point}
variable {R : ℝ}

-- Scenario 1: Prove that the locus of point P is an ellipse with O and A as foci and OB as the major axis length
theorem locus_of_P_is_ellipse (hA_inside_O : A ∈ circle O R) (hB_on_O : B ∈ circle O R)
  (hP_eq_bisector : is_perpendicular_bisector(O, B, P)) :
  is_ellipse_with_foci_and_major_axis_length P O A OB := by
  sorry

-- Scenario 2: Prove that the locus of point P is a hyperbola with O and A as foci and OB as the length of the real axis
theorem locus_of_P_is_hyperbola (hA_outside_O : A ∉ interior (circle O R)) (hB_on_O : B ∈ circle O R)
  (hP_eq_bisector : is_perpendicular_bisector(O, B, P)) :
  is_hyperbola_with_foci_and_real_axis_length P O A OB := by
  sorry

end locus_of_P_is_ellipse_locus_of_P_is_hyperbola_l474_474533


namespace distinct_ordered_pairs_count_l474_474209

theorem distinct_ordered_pairs_count :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 ^ 4 * p.2 ^ 2 - 10 * p.1 ^ 2 * p.2 + 9 = 0}.to_finset.card = 3 :=
sorry

end distinct_ordered_pairs_count_l474_474209


namespace man_speed_in_still_water_l474_474404

noncomputable def speed_of_man_in_still_water (vm vs : ℝ) : Prop :=
  -- Condition 1: v_m + v_s = 8
  vm + vs = 8 ∧
  -- Condition 2: v_m - v_s = 5
  vm - vs = 5

-- Proving the speed of the man in still water is 6.5 km/h
theorem man_speed_in_still_water : ∃ (v_m : ℝ), (∃ v_s : ℝ, speed_of_man_in_still_water v_m v_s) ∧ v_m = 6.5 :=
by
  sorry

end man_speed_in_still_water_l474_474404


namespace conical_jar_price_equivalence_l474_474069

theorem conical_jar_price_equivalence :
  (∀ (r h : ℝ), let V := (1/3) * π * r^2 * h in 
  (r = 1.5 ∧ h = 4 → V = 6 * π) ∧ 
  (r = 3 ∧ h = 8 → V = 24 * π)) ∧ 
  (let price_per_volume := (0.60 / (6 * π)) in 
  let price_6in_8in := price_per_volume * (24 * π) in price_6in_8in = 2.40)
  :=
sorry

end conical_jar_price_equivalence_l474_474069


namespace hockey_league_games_l474_474003

theorem hockey_league_games (teams : ℕ) (games_per_matchup : ℕ) (h1 : teams = 17) (h2 : games_per_matchup = 10) :
  let total_games := (teams * (teams - 1) / 2) * games_per_matchup in
  total_games = 1360 :=
by
  subst h1
  subst h2
  let total_games := (17 * (17 - 1) / 2) * 10
  calc
    total_games = ((17 * 16) / 2) * 10 : rfl
             ... = (272 / 2) * 10 : by norm_num
             ... = 136 * 10 : rfl
             ... = 1360 : rfl

end hockey_league_games_l474_474003


namespace school_fundraised_amount_l474_474013

def school_fundraising : ℝ :=
  let mrs_johnson_class := 2300
  let mrs_sutton_class := mrs_johnson_class / 2
  let miss_rollin_class := mrs_sutton_class * 8
  let total_school_raised := miss_rollin_class * 3
  let admin_fee := total_school_raised * 0.02
  total_school_raised - admin_fee

theorem school_fundraised_amount : school_fundraising = 27048 :=
by
  -- begin
  -- proof steps would go here
  -- end

  sorry

end school_fundraised_amount_l474_474013


namespace find_unknown_rate_of_blankets_l474_474426

open Real

theorem find_unknown_rate_of_blankets :
  ∃ x : ℝ, x = 225 ∧ (3 * 100 + 3 * 150 + 2 * x) / 8 = 150 :=
by
  have total_blankets := 8
  have average_price := 150
  have total_cost := 1200
  have total_known_cost := 3 * 100 + 3 * 150
  use 225
  simp [total_known_cost, total_cost, average_price, total_blankets]
  norm_num
  sorry

end find_unknown_rate_of_blankets_l474_474426


namespace question1_answer_question2_answer_l474_474256

-- Definitions for the problem
variables {A B C : Real} -- Angles of the triangle
variables {a b c : Real} -- Sides opposite to angles A, B, C respectively

-- Condition provided in the problem
axiom eq1 : a * sin B - sqrt 3 * b * cos A = 0

-- Values for question 2
axiom a_val : a = sqrt 7
axiom b_val : b = 2

noncomputable def find_angle_A : Real :=
  if 0 < A ∧ A < Real.pi then
    atan (sqrt 3)
  else
    0 -- Default value, not expected to be reached

noncomputable def area : Real :=
  let A := find_angle_A in
  if A = Real.pi / 3 then
    let c := sqrt 7 in -- Using the value derived from cosine law
    1 / 2 * b_val * c * sin (Real.pi / 3)
  else 
    0 -- Default value, not expected to be reached

theorem question1_answer :
  (A = Real.pi / 3) := sorry

theorem question2_answer :
  (a = sqrt 7) →
  (b = 2) →
  ((1 / 2 * b * 3 * (sqrt 3 / 2)) = (3 * sqrt 3 / 2)) := sorry

end question1_answer_question2_answer_l474_474256


namespace sum_of_first_fifteen_terms_l474_474144

noncomputable def a₃ : ℝ := -5
noncomputable def a₅ : ℝ := 2.4
noncomputable def a₁ : ℝ := -12.4
noncomputable def d : ℝ := 3.7

noncomputable def S₁₅ : ℝ := 15 / 2 * (2 * a₁ + 14 * d)

theorem sum_of_first_fifteen_terms :
  S₁₅ = 202.5 := 
by
  sorry

end sum_of_first_fifteen_terms_l474_474144


namespace triangle_ab_length_l474_474276

/-- In triangle ABC, point N lies on side AB such that AN = 3NB; the median AM intersects CN at point O.
Given AM = 7 cm, CN = 7 cm, and ∠NOM = 60°, prove that AB = 4√7 cm. -/
theorem triangle_ab_length (A B C N M O : Point) (x : Real)
(hN : x > 0)  -- NB is x, thus x > 0
(h_AN_3NB : dist A N = 3 * dist N B)
(h_AM : dist A M = 7)
(h_CN : dist C N = 7)
(h_nom_60 : ∠ N O M = 60) :
  dist A B = 4 * Real.sqrt 7 :=
sorry

end triangle_ab_length_l474_474276


namespace regular_tetrahedron_volume_approx_l474_474917

noncomputable def regular_tetrahedron_volume (a : ℝ) : ℝ :=
  (a^3) / (6 * Real.sqrt 2)

theorem regular_tetrahedron_volume_approx (a : ℝ)
    (h : ℝ) (H_mid_face : 2 = h)
    (H_mid_edge : Real.sqrt 10 = a / Real.sqrt 2) :
    abs (regular_tetrahedron_volume a - 309.84) < 0.01 :=
begin
  sorry
end

end regular_tetrahedron_volume_approx_l474_474917


namespace complement_U_A_complement_A_D_l474_474205

universe u

open Set

def U := { n : ℕ | n ∈ Finset.range 10 ∧ n > 0 }
def A := { 1, 2, 3, 4, 5, 6 : ℕ }
def D := { 1, 2, 3 : ℕ }

theorem complement_U_A : compl A = { 7, 8, 9, 10 } :=
by
  sorry

theorem complement_A_D : (A \ D) = { 4, 5, 6 } :=
by
  sorry

end complement_U_A_complement_A_D_l474_474205


namespace range_of_m_l474_474234

variable (m : ℝ)
def complex_number (m : ℝ) : ℂ := 
  (m^2 + m - 1 : ℝ) + (4 * m^2 - 8 * m + 3 : ℝ) * complex.i

def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem range_of_m (m : ℝ) (h : is_in_first_quadrant (complex.conj (complex_number m))) : 
  (m > (-1 + real.sqrt 5) / 2 ∧ m < 3 / 2) :=
sorry

end range_of_m_l474_474234


namespace log_sum_l474_474900

theorem log_sum : log 5 50 + log 5 20 = 6 :=
by
  sorry

end log_sum_l474_474900


namespace payment_methods_20_yuan_l474_474383

theorem payment_methods_20_yuan :
  let ten_yuan_note := 10
  let five_yuan_note := 5
  let one_yuan_note := 1
  ∃ (methods : Nat), 
    methods = 9 ∧ 
    ∃ (num_10 num_5 num_1 : Nat),
      (num_10 * ten_yuan_note + num_5 * five_yuan_note + num_1 * one_yuan_note = 20) →
      methods = 9 :=
sorry

end payment_methods_20_yuan_l474_474383


namespace remaining_capital_min_m_for_30_million_exceeds_l474_474852

def initial_capital := 50000
def growth_rate := 1.5
def remittance := 10000
def a (n : ℕ) : ℝ :=
  if n = 1 then (growth_rate * initial_capital - remittance) / 10
  else growth_rate * a (n - 1) - remittance / 10

theorem remaining_capital (n : ℕ) : a n = 4500 * (pow (3 / 2) (n - 1)) + 2000 := sorry

theorem min_m_for_30_million_exceeds (m : ℕ) (h : 4500 * (pow (3 / 2) (m - 1)) + 2000 > 30000) : m = 6 := sorry

end remaining_capital_min_m_for_30_million_exceeds_l474_474852


namespace remainder_4_exp_3023_mod_9_l474_474020

theorem remainder_4_exp_3023_mod_9 :
  (4 : ℤ)^(3023) % 9 = 7 :=
by
  -- Given conditions
  have h1: (4 : ℤ)^1 % 9 = 4 := by norm_num,
  have h2: (4 : ℤ)^2 % 9 = 7 := by norm_num,
  have h3: (4 : ℤ)^3 % 9 = 1 := by norm_num,
  -- Objective
  sorry

end remainder_4_exp_3023_mod_9_l474_474020


namespace susan_can_drive_with_50_l474_474722

theorem susan_can_drive_with_50 (car_efficiency : ℕ) (gas_price : ℕ) (money_available : ℕ) 
  (h1 : car_efficiency = 40) (h2 : gas_price = 5) (h3 : money_available = 50) : 
  car_efficiency * (money_available / gas_price) = 400 :=
by
  sorry

end susan_can_drive_with_50_l474_474722


namespace complex_quadrant_l474_474493

open Complex

theorem complex_quadrant : (1 / (1 + Complex.i)).re > 0 ∧ (1 / (1 + Complex.i)).im < 0 := by
  sorry

end complex_quadrant_l474_474493


namespace largest_angle_150_degrees_l474_474061

-- Definitions and conditions as derived:
variable (ABCD : Type)
variable [ConvexQuadrilateral ABCD]
variable (r : ℝ)
variable (A B C D : Point ABCD)
variable (A' : Point ABCD)
variable [NonRude ABCD] -- Assuming NonRude is a predicate indicating ABCD is not rude
variable [RudeA'BCD : ∀ A' ≠ A, A'distance ≤ r → Rude A'BCD]

theorem largest_angle_150_degrees (h_non_rude : NonRude ABCD)
    (h_rude_A' : ∀ A' p, p ≠ A ∧ A.distance p ≤ r → Rude (Quadrilateral.mk A' B C D)) :
    largest_angle ABCD = 150 :=
sorry

end largest_angle_150_degrees_l474_474061


namespace Ilya_defeats_dragon_l474_474246

-- Conditions
def prob_two_heads : ℚ := 1 / 4
def prob_one_head : ℚ := 1 / 3
def prob_no_heads : ℚ := 5 / 12

-- Main statement in Lean
theorem Ilya_defeats_dragon : 
  (prob_no_heads + prob_one_head + prob_two_heads = 1) → 
  (∀ n : ℕ, ∃ m : ℕ, m ≤ n) → 
  (∑ n, (prob_no_heads + prob_one_head + prob_two_heads) ^ n) = 1 := 
sorry

end Ilya_defeats_dragon_l474_474246


namespace regular_tetrahedron_volume_correct_l474_474915

noncomputable def volume_of_regular_tetrahedron 
  (height_midpoint_to_face_distance : ℝ) 
  (height_midpoint_to_edge_distance : ℝ) : ℝ :=
if height_midpoint_to_face_distance = 2 ∧ height_midpoint_to_edge_distance = real.sqrt 10 then 80 * real.sqrt 15 else 0

theorem regular_tetrahedron_volume_correct :
  volume_of_regular_tetrahedron 2 (real.sqrt 10) ≈ 309.84 :=
by
  field_simp
  norm_num
  sorry

end regular_tetrahedron_volume_correct_l474_474915


namespace doubling_period_l474_474093

theorem doubling_period (initial_capacity: ℝ) (final_capacity: ℝ) (years: ℝ) (initial_year: ℝ) (final_year: ℝ) (doubling_period: ℝ) :
  initial_capacity = 0.4 → final_capacity = 4100 → years = (final_year - initial_year) →
  initial_year = 2000 → final_year = 2050 →
  2 ^ (years / doubling_period) * initial_capacity = final_capacity :=
by
  intros h_initial h_final h_years h_i_year h_f_year
  sorry

end doubling_period_l474_474093


namespace orchids_initial_count_l474_474007

def initial_orchids (initial_roses : ℕ) (final_orchids : ℕ) (final_roses : ℕ) (added_roses : ℕ) : ℕ :=
  final_orchids + final_roses - (initial_roses + added_roses)

theorem orchids_initial_count : initial_orchids 13 91 14 1 = 91 :=
by
  rw [initial_orchids]
  sorry

end orchids_initial_count_l474_474007


namespace power_mod_eq_five_l474_474721

theorem power_mod_eq_five
  (m : ℕ)
  (h₀ : 0 ≤ m)
  (h₁ : m < 8)
  (h₂ : 13^5 % 8 = m) : m = 5 :=
by 
  sorry

end power_mod_eq_five_l474_474721


namespace integer_in_sqrt2_sqrt12_l474_474481

theorem integer_in_sqrt2_sqrt12 (a : ℤ) (h1 : real.sqrt 2 < a) (h2 : a < real.sqrt 12) : a = 2 ∨ a = 3 := 
by
  sorry

end integer_in_sqrt2_sqrt12_l474_474481


namespace candies_distribution_l474_474549

noncomputable def total_candies := 300
noncomputable def sour_candies := 0.375 * total_candies
noncomputable def good_candies := total_candies - sour_candies
noncomputable def henley_sour := 1/3 * sour_candies
noncomputable def remaining_sour := sour_candies - henley_sour
noncomputable def each_good := good_candies / 4
noncomputable def henley_candies := henley_sour + each_good
noncomputable def brother_sour_and_good := remaining_sour + each_good
noncomputable def other_candies := each_good

theorem candies_distribution :
  henley_candies = 84 ∧
  brother_sour_and_good = 122 ∧
  other_candies = 47 :=
by
  -- Proof skipped
  sorry

end candies_distribution_l474_474549


namespace road_travel_cost_l474_474078

-- Define the conditions of the problem
def lawn_length : ℝ := 90
def lawn_breadth : ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_m : ℝ := 3

-- Define the areas of the roads and their intersection
def road1_area : ℝ := road_width * lawn_breadth
def road2_area : ℝ := road_width * lawn_length
def intersection_area : ℝ := road_width * road_width

-- Define the total area without double-counting
def total_road_area : ℝ := road1_area + road2_area - intersection_area

-- Define the total cost
def total_cost : ℝ := total_road_area * cost_per_sq_m

-- Prove that the total cost is Rs. 4200
theorem road_travel_cost : total_cost = 4200 := 
by
  -- Include intermediate expressions for clarity
  have road1_area_correct : road1_area = 600 := by sorry
  have road2_area_correct : road2_area = 900 := by sorry
  have intersection_area_correct : intersection_area = 100 := by sorry
  have total_road_area_correct : total_road_area = 1400 := by sorry
  have total_cost_correct : total_cost = 4200 := by sorry
  show total_cost = 4200 from total_cost_correct

end road_travel_cost_l474_474078


namespace one_in_M_l474_474530

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := sorry

end one_in_M_l474_474530


namespace intersection_complement_eq_l474_474204

def U : set ℕ := { x | 0 ≤ x ∧ x ≤ 6 }
def A : set ℕ := {1, 3, 6}
def B : set ℕ := {1, 4, 5}
def C_U_B : set ℕ := U \ B

theorem intersection_complement_eq : A ∩ C_U_B = {3, 6} := by
  sorry

end intersection_complement_eq_l474_474204


namespace find_mn_l474_474580

variables {A B C M N O : Type*} [add_group A] [add_group B] [add_group C]
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C]
variables [vector_space ℝ M] [vector_space ℝ N] [vector_space ℝ O]
variables (a b c m n : ℝ)
variables (AB AM AC AN AO : A)

-- Definitions
def midpoint (A B : A) (O : O) : Prop :=
  O = (A + B) / 2

def collinear (O A B : O) : Prop := sorry -- dependency to prove that points O, M, N are collinear

def AB_eq_m_AM (AB AM : A) (m : ℝ) : Prop :=
  AB = m • AM

def AC_eq_n_AN (AC AN : A) (n : ℝ) : Prop :=
  AC = n • AN

-- The proof goal
theorem find_mn
  (h_mid : midpoint B C O)
  (h_AB : AB_eq_m_AM AB AM m)
  (h_AC : AC_eq_n_AN AC AN n)
  (h_collinear : collinear O M N) :
  m + n = 2 :=
sorry

end find_mn_l474_474580


namespace proof_inequality_l474_474278

variable {a b c : ℝ}

theorem proof_inequality (h : a * b < 0) : a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := by
  sorry

end proof_inequality_l474_474278


namespace n_divides_difference_l474_474176

open Set

theorem n_divides_difference (n : ℕ) (A B C : Set ℝ) (h1 : 2 ≤ n) 
    (hA : A.card = n) (hB : B.card = n) (hC : C.card = n) 
    (hDisjoint1 : Disjoint A B) (hDisjoint2 : Disjoint A C) (hDisjoint3 : Disjoint B C) 
    (a : ℕ := {t | t ∈ A ×ˢ B ×ˢ C ∧ t.1.1 < t.1.2 ∧ t.1.2 < t.2}.card)
    (b : ℕ := {t | t ∈ A ×ˢ B ×ˢ C ∧ t.1.1 > t.1.2 ∧ t.1.2 > t.2}.card)
    : n ∣ (a - b) := sorry

end n_divides_difference_l474_474176


namespace smallest_value_l474_474671

noncomputable def smallest_possible_value (a b : ℝ) : ℝ := 2 * a + b

theorem smallest_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 ≥ 3 * b) (h4 : b^2 ≥ (8 / 9) * a) :
  smallest_possible_value a b = 5.602 :=
sorry

end smallest_value_l474_474671


namespace flour_per_loaf_proof_l474_474697

-- Define the conditions
def total_flour_needed (loaves : ℕ) (flour : ℝ) : Prop :=
  loaves = 2 ∧ flour = 5

-- Define the goal
def flour_per_loaf (flour per_loaf : ℝ) : Prop :=
  flour = 2.5 * per_loaf

-- The theorem statement
theorem flour_per_loaf_proof : 
  ∀ (loaves : ℕ) (total_flour per_loaf_flour : ℝ), 
    total_flour_needed loaves total_flour → 
    flour_per_loaf total_flour per_loaf_flour :=
by
  intros loaves total_flour per_loaf_flour
  intro h
  cases h with h_loaves h_flour
  rw [h_loaves, h_flour]
  -- Here should be the proof steps
  sorry

end flour_per_loaf_proof_l474_474697


namespace equivalent_statements_l474_474399

-- Definitions
variables (P Q : Prop)

-- Original statement
def original_statement := P → Q

-- Statements
def statement_I := P → Q
def statement_II := Q → P
def statement_III := ¬ Q → ¬ P
def statement_IV := ¬ P ∨ Q

-- Proof problem
theorem equivalent_statements : 
  (statement_III P Q ∧ statement_IV P Q) ↔ original_statement P Q :=
sorry

end equivalent_statements_l474_474399


namespace part_a_part_b_l474_474117

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def complexity (n : ℕ) : ℕ :=
  if h : n > 1 then nat.factors n.length else 0

theorem part_a (n : ℕ) (h : n > 1) : is_power_of_two n → ∀ m : ℕ, (n ≤ m ∧ m ≤ 2*n) → complexity m ≤ complexity n :=
by {
  intro h_power,
  intro m,
  intro h_range,
  sorry
}

theorem part_b (n : ℕ) (h : n > 1) : ¬∃ n : ℕ, ∀ m : ℕ, (n ≤ m ∧ m ≤ 2*n) → complexity m < complexity n :=
by {
  sorry
}

end part_a_part_b_l474_474117


namespace max_distance_PA_l474_474518

theorem max_distance_PA (a b n : ℝ) (theta : ℝ) (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop)
  (h_curve : ∀ x y, C x y ↔ (x = 2 * Real.cos theta ∧ y = 3 * Real.sin theta)) 
  (h_line: ∀ x y, l x y ↔ (2 * x + y = -n))
  (h_point_on_curve : C P.1 P.2) :
  let m := λ x y, 2*x + y + n = 0 in
  ∀ A : ℝ × ℝ, (P ≠ A) → 
  ((∃ θ, A = (P.1 + θ * Real.sin (30 * ℝ.pi / 180), P.2 + θ * Real.cos (30 * ℝ.pi / 180))) → 
  let distance := (5 * (Real.sqrt (5 : ℝ)))/5 in
  ∃ PA, PA = (22 * Real.sqrt (5) / 5 : ℝ) :=
sorry

end max_distance_PA_l474_474518


namespace possible_whispers_l474_474282

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem possible_whispers (d : ℕ) 
  (h1 : d.digits 10.length = 2022) 
  (h2 : ∃ sum_z : ℕ, digit_sum (digit_sum d) = sum_z ∧ sum_z < 100) 
  : digit_sum (digit_sum (digit_sum d)) = 1 → 
    (sum_z = 19 ∨ sum_z = 28) := 
sorry

end possible_whispers_l474_474282


namespace mens_wages_l474_474048

theorem mens_wages (W B : ℕ) (earnings : ℕ) (h1 : 5*men = W*women) (h2 : W*women = 7*boys) 
    (h3 : earnings = 90 ) : mens_wages = 37.5 :=
  sorry

end mens_wages_l474_474048


namespace horse_grazing_area_l474_474068

noncomputable def grazing_area (radius : ℝ) : ℝ :=
  (1 / 4) * Real.pi * radius^2

theorem horse_grazing_area :
  let length := 46
  let width := 20
  let rope_length := 17
  rope_length <= length ∧ rope_length <= width →
  grazing_area rope_length = 72.25 * Real.pi :=
by
  sorry

end horse_grazing_area_l474_474068


namespace find_lambda_and_A_squared_l474_474508

open Matrix

theorem find_lambda_and_A_squared :
  ∃ λ : ℝ, λ = 1 ∧
  ∃ a : ℝ, a = 0 ∧
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = ![![1, a], ![-1, 2]] ∧
  (A ⬝ A) = ![![1, 0], ![-3, 4]] :=
by
  sorry

end find_lambda_and_A_squared_l474_474508


namespace new_table_capacity_is_six_l474_474072

-- Definitions based on the conditions
def total_tables : ℕ := 40
def extra_new_tables : ℕ := 12
def total_customers : ℕ := 212
def original_table_capacity : ℕ := 4

-- Main statement to prove
theorem new_table_capacity_is_six (O N C : ℕ) 
  (h1 : O + N = total_tables)
  (h2 : N = O + extra_new_tables)
  (h3 : O * original_table_capacity + N * C = total_customers) :
  C = 6 :=
sorry

end new_table_capacity_is_six_l474_474072


namespace equal_books_for_students_l474_474379

-- Define the conditions
def num_girls : ℕ := 15
def num_boys : ℕ := 10
def total_books : ℕ := 375
def books_for_girls : ℕ := 225
def books_for_boys : ℕ := total_books - books_for_girls -- Calculate books for boys

-- Define the theorem
theorem equal_books_for_students :
  books_for_girls / num_girls = 15 ∧ books_for_boys / num_boys = 15 :=
by
  sorry

end equal_books_for_students_l474_474379


namespace problem_1_part_1_problem_1_part_2_problem_2_condition_l474_474962

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2

theorem problem_1_part_1 
  (a : ℝ)
  (h_derivative : deriv (λx, f x a) 1 = 3) :
  let y := f in
  y 1 a = 1 ∧ 3 = y.deriv_eval 1 in 
  ∃ m b,
    y x a = m * x + b := 
  h₁ :
    y 1 a = 1
  h_tangent :
    3 = 3 * 1 - 2 a :=
  sorry

theorem problem_1_part_2 
  (a = 0)
  (f_x : ∀ x , f x a = x^3) : 
  -- Prove that the maximum value of f(x) on [0, 2] is 8
  is_max_on (f x 0) (∈ [[0,2]]) 8 :=
  sorry

theorem problem_2_condition 
  (a : ℝ)
  (h : ∀ x ∈ Icc (0 : ℝ) 2, f x a + x ≥ 0) :
  -- Prove that the range of a is (-∞, 2]
  a ≤ 2 :=
  sorry

end problem_1_part_1_problem_1_part_2_problem_2_condition_l474_474962


namespace father_age_l474_474427

variable (F S x : ℕ)

-- Conditions
axiom h1 : F + S = 75
axiom h2 : F = 8 * (S - x)
axiom h3 : F - x = S

-- Theorem to prove
theorem father_age : F = 48 :=
sorry

end father_age_l474_474427


namespace largest_n_divisibility_condition_l474_474391

theorem largest_n_divisibility_condition : ∃ n : ℕ, (n + 11) ∣ (n^3 + 99) ∧ n = 1221 := 
by
  have h := by norm_num
  exact ⟨1221, h, by norm_num⟩

end largest_n_divisibility_condition_l474_474391


namespace find_composite_value_l474_474520

def f (x : ℝ) : ℝ :=
if x < 0 then exp x else log x

theorem find_composite_value : f (f (1 / exp 1)) = 1 / exp 1 := by
  sorry

end find_composite_value_l474_474520


namespace while_loop_output_correct_do_while_loop_output_correct_l474_474400

def while_loop (a : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (7 - i)).map (λ n => (i + n, a + n + 1))

def do_while_loop (x : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (10 - i + 1)).map (λ n => (i + n, x + (n + 1) * 10))

theorem while_loop_output_correct : while_loop 2 1 = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)] := 
sorry

theorem do_while_loop_output_correct : do_while_loop 100 1 = [(1, 110), (2, 120), (3, 130), (4, 140), (5, 150), (6, 160), (7, 170), (8, 180), (9, 190), (10, 200)] :=
sorry

end while_loop_output_correct_do_while_loop_output_correct_l474_474400


namespace min_distance_origin_to_line_l474_474188

theorem min_distance_origin_to_line 
  (x y : ℝ) 
  (h : x + y = 4) : 
  ∃ P : ℝ, P = 2 * Real.sqrt 2 ∧ 
    (∀ Q : ℝ, Q = Real.sqrt (x^2 + y^2) → P ≤ Q) :=
by
  sorry

end min_distance_origin_to_line_l474_474188


namespace increasing_interval_l474_474574

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (2 * x^2 + x)

theorem increasing_interval (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) 
  (h₃ : ∀ x, (1/2 < x ∧ x < 1) → f a x > 0) : 
  (∀ x, f a x ∈ (0, +∞) → x ∈ (0, +∞)) :=
begin
  sorry
end

end increasing_interval_l474_474574


namespace find_k_l474_474525

-- Definitions based on the given conditions
def f (x k : ℝ) : ℝ := x * (x + k) * (x + 2 * k) * (x - 3 * k)

-- The problem statement in Lean
theorem find_k (k : ℝ) (h : (deriv (λ x : ℝ, f x k) 0) = 6) : k = -1 :=
  sorry

end find_k_l474_474525


namespace domain_of_tan_x_plus_pi_four_find_beta_in_tan_cos_relation_l474_474523

-- Proof Problem for Question 1
theorem domain_of_tan_x_plus_pi_four :
  ∀ x : ℝ, (f x = tan (x + π / 4) → ∀ k : ℤ, x ≠ k * π + π / 4) :=
sorry

-- Proof Problem for Question 2
theorem find_beta_in_tan_cos_relation
  (β : ℝ) (h₁ : β ∈ Ioo 0 π)
  (h₂ : tan (β + π / 4) = 2 * cos (β - π / 4)) :
  β = π / 12 ∨ β = 3 * π / 4 :=
sorry

end domain_of_tan_x_plus_pi_four_find_beta_in_tan_cos_relation_l474_474523


namespace slope_of_asymptotes_is_one_l474_474688

-- Given definitions and axioms
variables (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (A1 : ℝ × ℝ := (-a, 0))
  (A2 : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (c, b^2 / a))
  (C : ℝ × ℝ := (c, -b^2 / a))
  (Perp : (b^2 / a) / (c + a) * -(b^2 / a) / (c - a) = -1)

-- Proof goal
theorem slope_of_asymptotes_is_one : a = b → (∀ m : ℝ, m = (b / a) ∨ m = -(b / a)) ↔ ∀ m : ℝ, m = 1 ∨ m = -1 :=
by
  sorry

end slope_of_asymptotes_is_one_l474_474688


namespace petya_wins_l474_474706

theorem petya_wins (
  board : fin 101 × fin 101, 
  petya_first_move : (51, 51) ∈ board, 
  move : (bool × nat) → fin 101 × fin 101 → list (fin 101 × fin 101) → Prop, 
  legal_move : ∀ (p : bool × nat) (pos : fin 101 × fin 101) (board : list (fin 101 × fin 101)), 
    (p.1 = tt → (pos.1, pos.2) = (51, 51) ∨ (move (!p.1, p.2 + 1) pos board)) ∧
    (p.1 = ff → ∃ (distance : ℕ), distance = p.2 ∨ distance = p.2 + 1 ∧ move (p.1, distance) pos board)
  ) : ∃ (winner : bool), winner = ff := 
sorry

end petya_wins_l474_474706


namespace c_le_one_l474_474482

theorem c_le_one (c : ℝ) 
  (h : ∀ (n : ℕ), n > 0 → fract (n * (real.sqrt 3)) > c / (n * (real.sqrt 3))) : 
  c ≤ 1 :=
sorry

end c_le_one_l474_474482


namespace wheel_rpm_l474_474807

noncomputable def rpm_of_wheel (radius_cm : ℝ) (speed_kph : ℝ) : ℝ :=
  let speed_cm_per_min := (speed_kph * 100000) / 60 in
  let circumference := 2 * Real.pi * radius_cm in
  speed_cm_per_min / circumference

theorem wheel_rpm :
  rpm_of_wheel 70 66 ≈ 2500.57 :=
by sorry

end wheel_rpm_l474_474807


namespace storage_box_painting_ways_l474_474851

-- Define the problem conditions
def storage_box_faces : ℕ := 6
def available_colors : ℕ := 6
def adjacent_faces (f1 f2 : ℕ) : Prop :=
  -- Adjacent faces based on the problem (T, B, F, Ba, L, R)
  (f1, f2) ∈ {(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
              (1, 2), (2, 1), (1, 3), (3, 1), (2, 4), (4, 2),
              (2, 5), (5, 2), (3, 4), (4, 3), (3, 5), (5, 3),
              (4, 5), (5, 4)}

-- Define the theorem stating the number of ways to paint the storage box
theorem storage_box_painting_ways : 
  ∃ ways : ℕ, ways = 6 * 5 * 4 * 3 * 3 * 2 :=
by
  sorry

end storage_box_painting_ways_l474_474851


namespace inequality_l474_474174

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def derivative_condition (f f' : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f' x + f x / x > 0

def f (x : ℝ) : ℝ := sorry -- Placeholder for the actual function
def f' (x : ℝ) : ℝ := sorry -- Placeholder for the actual derivative of f

def a : ℝ := (1 / 2) * f (1 / 2)
def b : ℝ := -2 * f (-2)
def c : ℝ := (Real.log (1 / 2)) * f (Real.log (1 / 2))

theorem inequality (h_odd : odd_function f) (h_deriv : derivative_condition f f') :
  a < c ∧ c < b := sorry

end inequality_l474_474174


namespace normal_distribution_point_probability_zero_l474_474221

variables {μ σ : ℝ} (a : ℝ) 

def normal_distribution (x : ℝ) : Prop := 
  ∃ (μ σ : ℝ), true

theorem normal_distribution_point_probability_zero {X : ℝ → Prop} (hx : normal_distribution X) :
  P(X = a) = 0 :=
sorry

end normal_distribution_point_probability_zero_l474_474221


namespace ratio_of_fifteenth_term_l474_474296

theorem ratio_of_fifteenth_term (a b d e : ℕ) (h₁ : ∀ n, 
  let P_n := n * (2 * a + (n - 1) * d) / 2 in
  let Q_n := n * (2 * b + (n - 1) * e) / 2 in
  P_n = (5 * n + 3) * Q_n / (3 * n + 11))
  (h₂ : 2 * a * 17 + (13 - 2) * d = 2 * b * 17 + (13 - 2) * e)
  (h₃ : ∃ n, a = 4 * b / 7) :
  let a₁₅ := a + 14 * d in
  let b₁₅ := b + 14 * e in
  a₁₅ / b₁₅ = 71 / 52 := sorry

end ratio_of_fifteenth_term_l474_474296


namespace joan_final_oranges_l474_474658

def joan_oranges_initial := 75
def tom_oranges := 42
def sara_sold := 40
def christine_added := 15

theorem joan_final_oranges : joan_oranges_initial + tom_oranges - sara_sold + christine_added = 92 :=
by 
  sorry

end joan_final_oranges_l474_474658


namespace heath_plants_per_hour_l474_474546

theorem heath_plants_per_hour (rows : ℕ) (plants_per_row : ℕ) (hours : ℕ) (total_plants : ℕ) :
  rows = 400 ∧ plants_per_row = 300 ∧ hours = 20 ∧ total_plants = rows * plants_per_row →
  total_plants / hours = 6000 :=
by
  sorry

end heath_plants_per_hour_l474_474546


namespace area_of_equilateral_triangle_l474_474385

theorem area_of_equilateral_triangle
  (A B C D E : Type) 
  (side_length : ℝ) 
  (medians_perpendicular : Prop) 
  (BD CE : ℝ)
  (inscribed_circle : Prop)
  (equilateral_triangle : A = B ∧ B = C) 
  (s : side_length = 18) 
  (BD_len : BD = 15) 
  (CE_len : CE = 9) 
  : ∃ area, area = 81 * Real.sqrt 3
  :=
by {
  sorry
}

end area_of_equilateral_triangle_l474_474385


namespace correct_answer_is_b_l474_474791

-- Definitions of each statement
def statement1 := ∀ (samples : Type) (populations : Type), regression_equation_applicable_to_samples (samples) ∧ regression_equation_applicable_to_populations (populations)
def statement2 := ∀ (reg_eq : regression_equation), has_temporality (reg_eq)
def statement3 := ∀ (x : sample_range), affects_applicability (x)
def statement4 := ∀ (reg_eq : regression_equation), forecast_value_is_precise (reg_eq)

-- The condition stating which statements are correct
def correct_statements : Prop :=
  statement2 ∧ statement3

-- Prove that the correct statements are ② and ③
theorem correct_answer_is_b : correct_statements :=
by 
  sorry

end correct_answer_is_b_l474_474791


namespace arithmetic_progression_sum_15_terms_l474_474148

theorem arithmetic_progression_sum_15_terms (a₃ a₅ : ℝ) (h₁ : a₃ = -5) (h₂ : a₅ = 2.4) : 
  let d := (a₅ - a₃) / 2 in
  let a₁ := a₃ - 2 * d in
  (15 / 2) * (2 * a₁ + 14 * d) = 202.5 :=
by
  sorry

end arithmetic_progression_sum_15_terms_l474_474148


namespace defeat_dragon_probability_l474_474250

noncomputable theory

def p_two_heads_grow : ℝ := 1 / 4
def p_one_head_grows : ℝ := 1 / 3
def p_no_heads_grow : ℝ := 5 / 12

-- We state the probability that Ilya will eventually defeat the dragon
theorem defeat_dragon_probability : 
  ∀ (expected_value : ℝ), 
  (expected_value = p_two_heads_grow * 2 + p_one_head_grows * 1 + p_no_heads_grow * 0) →
  expected_value < 1 →
  prob_defeat (count_heads n : ℕ) > 0 :=
by
  sorry

end defeat_dragon_probability_l474_474250


namespace find_parabola_and_hyperbola_equations_l474_474191

-- Define the equations of the parabola and the hyperbola.
def parabola_eq : (x y : ℝ) → Prop := λ x y, y^2 = -4 * x
def hyperbola_eq : (x y : ℝ) → Prop := λ x y, (x^2) / (1/4) - (y^2) / (3/4) = 1

-- Given that the vertex of the parabola is at the origin,
-- and other provided conditions, the goal is to show these equations.
theorem find_parabola_and_hyperbola_equations (x y : ℝ) :
  (parabola_eq (-3/2) (sqrt 6)) ∧
  (hyperbola_eq (-3/2) (sqrt 6)) →
  ∀ x y, parabola_eq x y ∧ hyperbola_eq x y :=
by {
  sorry
}

end find_parabola_and_hyperbola_equations_l474_474191


namespace combination_8_5_l474_474621

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474621


namespace crates_hold_kilos_l474_474062

-- Conditions
variables {x : ℕ} -- kilograms of tomatoes a crate can hold
constant three_crates : ℕ := 3
constant cost_of_crates : ℕ := 330
constant price_per_kg : ℕ := 6
constant rotten_tomatoes : ℕ := 3
constant tommys_profit : ℕ := 12

-- Theorem to prove
theorem crates_hold_kilos (h1 : (3 * x - rotten_tomatoes) * price_per_kg - cost_of_crates = tommys_profit) : x = 20 :=
sorry

end crates_hold_kilos_l474_474062


namespace sqrt_equation_solution_l474_474565

theorem sqrt_equation_solution (s : ℝ) :
  (sqrt (3 * sqrt (s - 1)) = real.rpow (9 - s) (1/4)) → s = 1.8 :=
sorry

end sqrt_equation_solution_l474_474565


namespace combination_8_5_is_56_l474_474611

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474611


namespace reflection_about_x_axis_l474_474996

theorem reflection_about_x_axis (a : ℝ) : 
  (A : ℝ × ℝ) = (3, a) → (B : ℝ × ℝ) = (3, 4) → A = (3, -4) → a = -4 :=
by
  intros A_eq B_eq reflection_eq
  sorry

end reflection_about_x_axis_l474_474996


namespace intersection_complement_l474_474941

def A := {x : ℝ | -1 < x ∧ x < 6}
def B := {x : ℝ | x^2 < 4}
def complement_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem intersection_complement :
  A ∩ (complement_R B) = {x : ℝ | 2 ≤ x ∧ x < 6} := by
sorry

end intersection_complement_l474_474941


namespace sin_double_angle_second_quadrant_l474_474947

-- Define the conditions: α in the second quadrant and cos α = -3/5
variables {α : ℝ}
hypothesis (h1 : (π / 2) < α ∧ α < π)        -- α is in the second quadrant
hypothesis (h2 : cos α = -3 / 5)             -- cos α = -3/5

-- Prove that sin 2α = -24/25
theorem sin_double_angle_second_quadrant : sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_second_quadrant_l474_474947


namespace function_range_x2_minus_2x_l474_474882

theorem function_range_x2_minus_2x : 
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 3 → -1 ≤ x^2 - 2 * x ∧ x^2 - 2 * x ≤ 3 :=
by
  intro x hx
  sorry

end function_range_x2_minus_2x_l474_474882


namespace prime_existence_of_xy_l474_474718

theorem prime_existence_of_xy (p : ℕ) (hp : Nat.Prime p) (hpg : p ≥ 7) : ∃ n : ℕ, ∃ (x y : ℕ → ℤ), (∀ i : ℕ, i < n → x i % p ≠ 0 ∧ y i % p ≠ 0) ∧ (∀ i : ℕ, i < n → (x i)^2 + (y i)^2 ≡ (x ((i + 1) % n))^2 [MOD p]) :=
sorry

end prime_existence_of_xy_l474_474718


namespace percentage_increase_in_capacity_l474_474011

theorem percentage_increase_in_capacity (initial_capacity_per_hand : ℕ) (total_capacity_after_specialization : ℕ) :
  initial_capacity_per_hand = 80 →
  total_capacity_after_specialization = 352 →
  ((total_capacity_after_specialization / 2 - (initial_capacity_per_hand * 2)) / (initial_capacity_per_hand * 2)) * 100 = 10 :=
by
  intros h_initial h_specialization
  -- initial_capacity_per_hand = 80
  sorry
  -- total_capacity_after_specialization = 352
  sorry
  -- Calculate the final percentage increase.
  sorry

end percentage_increase_in_capacity_l474_474011


namespace interpolation_formula_l474_474550

-- Definitions
variables {f : ℝ → ℝ} {n : ℝ} {x : ℝ}
-- Conditions
variable (h1 : n < x) (h2 : x < n + 1)
-- Goal
theorem interpolation_formula (f_n : f(n)) (f_n1 : f(n+1)) :
  f(x) = f(n) + (x - n) * (f(n+1) - f(n)) :=
sorry

end interpolation_formula_l474_474550


namespace smallest_prime_12_less_than_square_l474_474396

-- Defining prime number and asserting the given conditions
def is_smallest_prime_diff_square (p : ℕ) : Prop :=
  prime p ∧
  ∃ n : ℕ, prime n ∧ p = n^2 - 12 ∧ p > 0

-- Proving the existence of the smallest prime number according to the given conditions
theorem smallest_prime_12_less_than_square : ∃ p, is_smallest_prime_diff_square p ∧ p = 13 :=
by 
  sorry

end smallest_prime_12_less_than_square_l474_474396


namespace num_valid_subsets_l474_474457

def set_with_property (S : set ℤ) :=
  (S ⊆ {-3, -2, -1, 0, 1, 2, 3}) ∧ 
  (S ≠ ∅) ∧ 
  (∀ (k : ℕ) (a : fin k → ℤ), (∀ i, a i ∈ S) → (function.injective a) → ∑ i, a i ≠ 0)

theorem num_valid_subsets : 
  (finset.univ.powerset.filter (λ S, set_with_property S.1)).card = 24 :=
by
  sorry

end num_valid_subsets_l474_474457


namespace exists_divisor_c_of_f_l474_474314

theorem exists_divisor_c_of_f (f : ℕ → ℕ) 
  (h₁ : ∀ n, f n ≥ 2)
  (h₂ : ∀ m n, f (m + n) ∣ (f m + f n)) :
  ∃ c > 1, ∀ n, c ∣ f n :=
sorry

end exists_divisor_c_of_f_l474_474314


namespace true_propositions_count_l474_474566

theorem true_propositions_count {a b c : ℝ} (h : a ≤ b) : 
  (if (c^2 ≥ 0 ∧ a * c^2 ≤ b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ a * c^2 > b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a * c^2 ≤ b * c^2) → ¬(a ≤ b)) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a ≤ b) → ¬(a * c^2 ≤ b * c^2)) then 1 else 0) = 2 :=
sorry

end true_propositions_count_l474_474566


namespace diagonal_difference_l474_474876

-- Definitions
def original_matrix : matrix (fin 5) (fin 5) ℤ :=
![![5, 6, 7, 8, 9],
  ![10, 11, 12, 13, 14],
  ![15, 16, 17, 18, 19],
  ![20, 21, 22, 23, 24],
  ![25, 26, 27, 28, 29]]

def reversed_matrix : matrix (fin 5) (fin 5) ℤ :=
![![5, 6, 7, 8, 9],
  ![10, 11, 12, 13, 14],
  ![19, 18, 17, 16, 15],
  ![20, 21, 22, 23, 24],
  ![29, 28, 27, 26, 25]]

-- Question
theorem diagonal_difference (m : matrix (fin 5) (fin 5) ℤ) :
  m = reversed_matrix →
  abs ((m 0 0 + m 1 1 + m 2 2 + m 3 3 + m 4 4) - 
       (m 0 4 + m 1 3 + m 2 2 + m 3 1 + m 4 0)) = 8 :=
sorry

end diagonal_difference_l474_474876


namespace quadratic_distinct_roots_l474_474576

theorem quadratic_distinct_roots (k : ℝ) : 
  k < 5 ∧ k ≠ 1 ↔ ∃ x : ℝ, (k-1)*x^2 + 4*x + 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ (k-1)*y^2 + 4*y + 1 = 0 :=
begin
  sorry
end

end quadratic_distinct_roots_l474_474576


namespace inradius_of_triangle_l474_474874

theorem inradius_of_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 16) (h3 : c = 17) : 
    let s := (a + b + c) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
    let r := area / s
    r = Real.sqrt 21 := by
  sorry

end inradius_of_triangle_l474_474874


namespace regular_tetrahedron_volume_correct_l474_474916

noncomputable def volume_of_regular_tetrahedron 
  (height_midpoint_to_face_distance : ℝ) 
  (height_midpoint_to_edge_distance : ℝ) : ℝ :=
if height_midpoint_to_face_distance = 2 ∧ height_midpoint_to_edge_distance = real.sqrt 10 then 80 * real.sqrt 15 else 0

theorem regular_tetrahedron_volume_correct :
  volume_of_regular_tetrahedron 2 (real.sqrt 10) ≈ 309.84 :=
by
  field_simp
  norm_num
  sorry

end regular_tetrahedron_volume_correct_l474_474916


namespace element_of_A_l474_474318

noncomputable def A : set ℝ := 
  {x : ℝ | x ∉ ℚ ∧ (√3 * x^2 - (2 * √3 - √2) * x - 2 * √2 = 0)}

theorem element_of_A :
  (2 - √6 / 3) ∈ A :=
sorry

end element_of_A_l474_474318


namespace probability_below_8_l474_474080

theorem probability_below_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) : 
  1 - (p10 + p9 + p8) = 0.40 :=
by 
  rw [h1, h2, h3]
  sorry

end probability_below_8_l474_474080


namespace intersection_P_Q_l474_474925

def P : Set ℝ := {x : ℝ | x < 1}
def Q : Set ℝ := {x : ℝ | x^2 < 4}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := 
  sorry

end intersection_P_Q_l474_474925


namespace height_of_marys_brother_l474_474692

namespace ProofProblem

variable {height_min : ℕ}
variable (Mary_height : ℕ) (Brother_height : ℕ)

-- Conditions
def min_height_to_ride_kingda_ka : height_min = 140 := sorry
def mary_is_2_3_height_of_brother : Mary_height = (2 * Brother_height) / 3 := sorry
def mary_needs_20_more_cm : Mary_height + 20 = 140 := sorry

-- Proof statement
theorem height_of_marys_brother : Brother_height = 180 :=
by
  have h1 : Mary_height + 20 = 140 := mary_needs_20_more_cm
  have h2 : Mary_height = (2 * Brother_height) / 3 := mary_is_2_3_height_of_brother
  have h3 : Mary_height = 120 := by linarith
  have h4 : Brother_height = 180 := by linarith
  exact sorry
end ProofProblem

end height_of_marys_brother_l474_474692


namespace segments_perpendicular_l474_474830

-- Define the geometrical objects involved
variables (T : Type) [planar_figure T]
variables (trapezoid : T) (circle : T) (O : point T) (A B : point T)

-- Assume the necessary conditions
axiom trapezoid_has_inscribed_circle : inscribed_circle trapezoid circle
axiom center_of_circle : center circle O
axiom endpoints_of_leg : leg_endpoints trapezoid A B
axiom segments_OA_OB : connects O A ∧ connects O B

-- State the theorem to be proved
theorem segments_perpendicular :
  (perpendicular (line_segment O A) (line_segment O B)) :=
sorry

end segments_perpendicular_l474_474830


namespace proof_propositions_true_l474_474193

-- Definitions for the conditions
def proposition_1 (A B : Point) (α : Plane) : Prop :=
  ((dist A α = dist B α) → (line_through A B).parallel_to α)

def proposition_2 (l₁ l₂ : Line) (α : Plane) : Prop :=
  ((angle_between l₁ α = angle_between l₂ α) → l₁.parallel_to l₂)

def proposition_3 (a : Line) (α β : Plane) : Prop :=
  (a.parallel_to α) ∧ (a.perpendicular_to β) ∧ (line_through α β.exists) → α.perpendicular_to β

def proposition_4 (α β γ : Plane) : Prop :=
  (α.intersects β) ∧ (α.perpendicular_to γ) ∧ (β.perpendicular_to γ) → (line_of_intersection α β).perpendicular_to γ

-- The theorem statement for the correct propositions
theorem proof_propositions_true (A B : Point) (α β γ : Plane) (a : Line) (l₁ l₂ : Line) :
  proposition_3 a α β ∧ proposition_4 α β γ :=
by
  sorry

end proof_propositions_true_l474_474193


namespace triangle_is_equilateral_l474_474828

-- Define the conditions
variables {P Q R A B C D E F : Type*}
-- Points where the circle intersects the triangle sides
variables [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (circ : Circle P Q R A B C D E F)

-- The main theorem to prove the triangle is equilateral
theorem triangle_is_equilateral 
  (h1 : divides_into_three_equal_parts circ A B P Q)
  (h2 : divides_into_three_equal_parts circ C D Q R)
  (h3 : divides_into_three_equal_parts circ E F R P) :
  segment_length P Q = segment_length Q R ∧ 
  segment_length Q R = segment_length R P ∧ 
  segment_length R P = segment_length P Q :=
sorry

end triangle_is_equilateral_l474_474828


namespace fred_change_received_l474_474326

theorem fred_change_received :
  let ticket_price := 5.92
  let ticket_count := 2
  let borrowed_movie_price := 6.79
  let amount_paid := 20.00
  let total_cost := (ticket_price * ticket_count) + borrowed_movie_price
  let change := amount_paid - total_cost
  change = 1.37 :=
by
  sorry

end fred_change_received_l474_474326


namespace quadratic_distinct_roots_l474_474575

theorem quadratic_distinct_roots (k : ℝ) : 
  k < 5 ∧ k ≠ 1 ↔ ∃ x : ℝ, (k-1)*x^2 + 4*x + 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ (k-1)*y^2 + 4*y + 1 = 0 :=
begin
  sorry
end

end quadratic_distinct_roots_l474_474575


namespace total_obstacle_course_time_l474_474286

-- Definitions for the given conditions
def first_part_time : Nat := 7 * 60 + 23
def second_part_time : Nat := 73
def third_part_time : Nat := 5 * 60 + 58

-- State the main theorem
theorem total_obstacle_course_time :
  first_part_time + second_part_time + third_part_time = 874 :=
by
  sorry

end total_obstacle_course_time_l474_474286


namespace Alyssa_initial_puppies_l474_474087

theorem Alyssa_initial_puppies : 
  ∀ (a b c : ℕ), b = 7 → c = 5 → a = b + c → a = 12 := 
by
  intros a b c hb hc hab
  rw [hb, hc] at hab
  exact hab

end Alyssa_initial_puppies_l474_474087


namespace years_since_mothers_death_l474_474656

noncomputable def jessica_age_at_death (x : ℕ) : ℕ := 40 - x
noncomputable def mother_age_at_death (x : ℕ) : ℕ := 2 * jessica_age_at_death x

theorem years_since_mothers_death (x : ℕ) : mother_age_at_death x + x = 70 ↔ x = 10 :=
by
  sorry

end years_since_mothers_death_l474_474656


namespace evaluate_256_pow_5_div_4_l474_474893

theorem evaluate_256_pow_5_div_4 : 256 ^ (5 / 4) = 1024 :=
by
  -- Condition 1: Express 256 as a power of 2
  have h1 : 256 = 2 ^ 8 := by rfl

  -- Condition 2: Apply power of powers property
  have h2 : 256 ^ (5 / 4) = (2 ^ 8) ^ (5 / 4) := by rw [h1]
  have h3 : (2 ^ 8) ^ (5 / 4) = 2 ^ (8 * (5 / 4)) := by norm_num

  -- Calculating the exponential result
  have h4 : 2 ^ (8 * (5 / 4)) = 2 ^ 10 := by norm_num

  -- The final answer
  show 2 ^ 10 = 1024
  exact by norm_num

end evaluate_256_pow_5_div_4_l474_474893


namespace radian_measure_of_negative_150_degree_l474_474365

theorem radian_measure_of_negative_150_degree  : (-150 : ℝ) * (Real.pi / 180) = - (5 * Real.pi / 6) := by
  sorry

end radian_measure_of_negative_150_degree_l474_474365


namespace legs_paws_in_pool_l474_474462

def total_legs_paws (num_humans : Nat) (human_legs : Nat) (num_dogs : Nat) (dog_paws : Nat) : Nat :=
  (num_humans * human_legs) + (num_dogs * dog_paws)

theorem legs_paws_in_pool :
  total_legs_paws 2 2 5 4 = 24 := by
  sorry

end legs_paws_in_pool_l474_474462


namespace rational_terms_binomial_expansion_coefficient_of_x2_l474_474517

theorem rational_terms_binomial_expansion
  (x : ℝ)
  (n : ℕ)
  (h_sum_odd_coeff : 2^(n-1) = 512) :
  (C 10 0 * x^5 = 1 * x^5) ∧ (C 10 6 * x^4 = 210 * x^4) :=
by
  sorry

theorem coefficient_of_x2
  (x : ℝ)
  (n : ℕ)
  (h_n : n = 10) :
  (∑ k in finset.range (n + 1), if 3 ≤ k then C k 2 else 0) = 164 :=
by
  sorry

end rational_terms_binomial_expansion_coefficient_of_x2_l474_474517


namespace decimal_digit_count_does_not_determine_size_l474_474737

theorem decimal_digit_count_does_not_determine_size:
  ¬ ∀ (a b : ℚ), (nat_digit_count a > nat_digit_count b) → (a > b) :=
by
  intro H
  have h : ¬ (3.456 > 4.6) := dec_trivial
  have h_digits : nat_digit_count 3.456 > nat_digit_count 4.6 := dec_trivial
  exact H 3.456 4.6 h_digits h

end decimal_digit_count_does_not_determine_size_l474_474737


namespace radius_of_larger_circle_l474_474009

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) 
  (h1 : ∀ a b c : ℝ, a = 2 ∧ b = 2 ∧ c = 2) 
  (h2 : ∀ x y z : ℝ, (x = 4) ∧ (y = 4) ∧ (z = 4) ) 
  (h3 : ∀ A B : ℝ, A * 2 = 2) : 
  R = 2 + 2 * Real.sqrt 3 :=
by
  sorry

end radius_of_larger_circle_l474_474009


namespace choose_five_from_eight_l474_474608

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474608


namespace crease_length_l474_474839

theorem crease_length (A B C D E : ℝ) (h1 : dist A B = 5) (h2 : dist A C = 12) (h3 : dist B C = 13)
  (h4 : dist D A = 2.5) (h5 : dist D B = 2.5) (h6 : dist E A = 6) (h7 : dist E C = 6) 
  (h8 : dist D E = 6.5) : dist D E = real.sqrt 54.5 :=
sorry

end crease_length_l474_474839


namespace smallest_positive_angle_same_terminal_side_l474_474371

theorem smallest_positive_angle_same_terminal_side : 
  ∃ k : ℤ, (∃ α : ℝ, α > 0 ∧ α = -660 + k * 360) ∧ (∀ β : ℝ, β > 0 ∧ β = -660 + k * 360 → β ≥ α) :=
sorry

end smallest_positive_angle_same_terminal_side_l474_474371


namespace surface_area_is_50π_l474_474844

-- Define the lengths of the edges
def edge1 : ℝ := 3
def edge2 : ℝ := 4
def edge3 : ℝ := 5

-- The space diagonal of the rectangular solid
def space_diagonal := Real.sqrt (edge1 ^ 2 + edge2 ^ 2 + edge3 ^ 2)

-- The radius of the sphere
def R : ℝ := space_diagonal / 2

-- The surface area of the sphere
def surface_area_sphere := 4 * Real.pi * R ^ 2

-- Final statement
theorem surface_area_is_50π : surface_area_sphere = 50 * Real.pi := 
  by sorry

end surface_area_is_50π_l474_474844


namespace amount_of_pizza_needed_l474_474384

theorem amount_of_pizza_needed :
  (1 / 2 + 1 / 3 + 1 / 6) = 1 := by
  sorry

end amount_of_pizza_needed_l474_474384


namespace asymptotes_of_hyperbola_l474_474904

theorem asymptotes_of_hyperbola (x y : ℝ) :
  (∃ x y : ℝ, (x ^ 2) / 4 - (y ^ 2) / 2 = 1) →
  ∃ k : ℝ, (k = (sqrt 2) / 2 ∨ k = -(sqrt 2) / 2) ∧ (y = k * x) :=
begin
  intros H,
  -- Proof is omitted
  sorry
end

end asymptotes_of_hyperbola_l474_474904


namespace circle_tangent_to_circumcircle_l474_474665

-- Definitions related to the problem
variable (A B C D K E : Type)
variable [has_circumcircle A B C]
variable [triangle_is_acute ∠ABC]
variable [lt (distance A B) (distance A C)]
variable [lt (distance A C) (distance B C)]
variable [diametrically_opposite D A c]
variable [point_on_line K B D]
variable [eq (distance K B) (distance K C)]
variable [intersects (circle K (distance K C)) AC E]

-- Theorem statement without proof
theorem circle_tangent_to_circumcircle 
  (A B C D E K: Point)
  (h1: triangle_is_acute A B C)
  (h2: AB < AC)
  (h3: AC < BC)
  (h4: circumcircle A B C = c)
  (h5: diametrically_opposite D A c)
  (h6: K ∈ BD)
  (h7: KB = KC)
  (h8: (K, KC).circle_intersects_AC E) :
  tangent (circle B K E) c := 
sorry

end circle_tangent_to_circumcircle_l474_474665


namespace rounded_product_approx_l474_474100

theorem rounded_product_approx :
  let a := 2.5
  let b := 56.2 + 0.15
  let rounded_b := round b
  a * rounded_b = 140 :=
by
  sorry

end rounded_product_approx_l474_474100


namespace gunny_bag_capacity_in_tons_l474_474328

def ton_to_pounds := 2200
def pound_to_ounces := 16
def packets := 1760
def packet_weight_pounds := 16
def packet_weight_ounces := 4

theorem gunny_bag_capacity_in_tons :
  ((packets * (packet_weight_pounds + (packet_weight_ounces / pound_to_ounces))) / ton_to_pounds) = 13 :=
sorry

end gunny_bag_capacity_in_tons_l474_474328


namespace concentration_of_salt_l474_474056

theorem concentration_of_salt (V₁ V₂ : ℝ) (C₂ : ℝ) :
  V₁ = 1 ∧ V₂ = 0.25 ∧ C₂ = 0.5 →
  let V_total := V₁ + V₂ in
  let salt_amount := V₂ * C₂ in
    (salt_amount / V_total) = 0.1 :=
by
  intros h
  let V₁ := 1
  let V₂ := 0.25
  let C₂ := 0.5
  let V_total := V₁ + V₂
  let salt_amount := V₂ * C₂
  sorry

end concentration_of_salt_l474_474056


namespace part1_part2_l474_474540
-- Import the entire Mathlib library for broader usage

-- Definition of the given vectors
def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

-- Part 1: Prove the dot product when x = -1 is 31
theorem part1 : (a.1 * (-1) + a.2 * (5)) = 31 := by
  sorry

-- Part 2: Prove the value of x when the vectors are parallel
theorem part2 : (4 : ℝ) / x = (7 : ℝ) / (x + 6) → x = 8 := by
  sorry

end part1_part2_l474_474540


namespace vector_combination_lambda_mu_sum_l474_474974

theorem vector_combination_lambda_mu_sum :
  ∀ (λ μ : ℝ),
    let a := (1, 1)
    let b := (-1, 1)
    let c := (4, 2)
    c = (λ * a.1 + μ * b.1, λ * a.2 + μ * b.2) →
    λ + μ = 2 :=
by
  intros λ μ a b c h
  dsimp at h
  sorry

end vector_combination_lambda_mu_sum_l474_474974


namespace functional_equation_solution_l474_474472

theorem functional_equation_solution
  (f : ℝ → ℝ)
  (h k : ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → f(x) * f(y) = y^h * f(x/2) + x^k * f(y/2))
  → ((h = k ∧ (∀ x, f x = 0 ∨ f x = 2 * (x/2)^h)) ∨ (h ≠ k ∧ ∀ x, f x = 0)) :=
sorry

end functional_equation_solution_l474_474472


namespace GP_reciprocal_sum_l474_474311

theorem GP_reciprocal_sum (r : ℝ) (h : r > 1) : 
  (∑ i in range 5, (3 * r^i)⁻¹) = (r^5 - 1) / (3 * r^5 * (r - 1)) :=
by
  sorry

end GP_reciprocal_sum_l474_474311


namespace sixth_grader_count_l474_474864

theorem sixth_grader_count : 
  ∃ x y : ℕ, (3 / 7) * x = (1 / 3) * y ∧ x + y = 140 ∧ x = 61 :=
by {
  sorry  -- Proof not required
}

end sixth_grader_count_l474_474864


namespace max_sum_two_numbers_l474_474889

theorem max_sum_two_numbers (digits_used_once : ∀ d, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → ∃ !n₁ n₂ : ℕ, 
  (n₁ + n₂ = 973841) ∧ (∀ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ∃ i, i ∈ [n₁, n₂] ∧ nat.digit i = d)) :
  ∃ n₁ n₂ : ℕ, n₁ = 8741 ∨ n₂ = 8741 :=
by
  sorry

end max_sum_two_numbers_l474_474889


namespace three_wheels_possible_two_wheels_not_possible_l474_474338

-- Define the conditions as hypotheses
def wheels_spokes (total_spokes_visible : ℕ) (max_spokes_per_wheel : ℕ) (wheels : ℕ) : Prop :=
  total_spokes_visible >= wheels * max_spokes_per_wheel ∧ wheels ≥ 1

-- Prove if a) three wheels is a possible solution
theorem three_wheels_possible : ∃ wheels, wheels = 3 ∧ wheels_spokes 7 3 wheels := by
  sorry

-- Prove if b) two wheels is not a possible solution
theorem two_wheels_not_possible : ¬ ∃ wheels, wheels = 2 ∧ wheels_spokes 7 3 wheels := by
  sorry

end three_wheels_possible_two_wheels_not_possible_l474_474338


namespace intersection_of_sets_l474_474942

noncomputable def setA : Set ℕ := { x : ℕ | x^2 ≤ 4 * x ∧ x > 0 }

noncomputable def setB : Set ℕ := { x : ℕ | 2^x - 4 > 0 ∧ 2^x - 4 ≤ 4 }

theorem intersection_of_sets : { x ∈ setA | x ∈ setB } = {3} :=
by
  sorry

end intersection_of_sets_l474_474942


namespace math_problem_l474_474480

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 :=
by
  sorry

end math_problem_l474_474480


namespace point_on_line_l474_474235

theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (A * x₀ + B * y₀ + C = 0) ↔ (A * (x₀ - x₀) + B * (y₀ - y₀) = 0) :=
by 
  sorry

end point_on_line_l474_474235


namespace isoperimetric_inequality_l474_474710

theorem isoperimetric_inequality (S : ℝ) (P : ℝ) : S ≤ P^2 / (4 * Real.pi) :=
sorry

end isoperimetric_inequality_l474_474710


namespace distinct_ways_to_place_digits_l474_474227

-- Definitions to capture the conditions in Lean code
def boxes := finset (fin 4)
def digits := finset (fin 4)
def choices : finset (fin 4) := {0, 1, 2, 3}

-- Statement of the problem
theorem distinct_ways_to_place_digits : 
  boxes.card = 4 → digits.card = 4 → choices.card = 4 → 
  ∃! a : list (fin 4), a.perm [0, 1, 2, 3] ∧ a.length = 4 ∧ list.nodup a ∧ 
  list.permutations a = 24 :=
by
  intro h1 h2 h3,
  sorry

end distinct_ways_to_place_digits_l474_474227


namespace largest_constant_inequality_l474_474133

theorem largest_constant_inequality :
  ∃ C : ℝ, (∀ x y : ℝ, x^2 + y^2 + 1 ≥ C * (x + y)) ∧
  (∀ x y : ℝ, x^2 + y^2 + 1 ≥ (real.sqrt 2) * (x + y)) ∧
  ∀ C' : ℝ, (∀ x y : ℝ, x^2 + y^2 + 1 ≥ C' * (x + y)) → C' ≤ real.sqrt 2 :=
sorry

end largest_constant_inequality_l474_474133


namespace ellipse_equation_l474_474953

open Real

theorem ellipse_equation
  (h1 : Center ellipse = (0, 0))
  (h2 : eccentricity ellipse = 1/2)
  (h3 : ∃ f : (ℝ × ℝ), Focus parabola = f ∧ Focus ellipse = f) :
  equation ellipse = "x^2 / 16 + y^2 / 12 = 1" :=
sorry

end ellipse_equation_l474_474953


namespace focal_length_of_hyperbola_l474_474730

theorem focal_length_of_hyperbola : 
  let a := 1
  let b := 1 
  let c := (a^2 + b^2).sqrt in
  2 * c = 2 * Real.sqrt 2 := 
by
  let a := 1
  let b := 1 
  let c := (a^2 + b^2).sqrt 
  exact rfl -- The proof is omitted.


end focal_length_of_hyperbola_l474_474730


namespace train_crossing_time_l474_474407

-- Condition definitions
def length_train : ℝ := 100
def length_bridge : ℝ := 150
def speed_kmph : ℝ := 54
def speed_mps : ℝ := 15

-- Given the conditions, prove the time to cross the bridge is 16.67 seconds
theorem train_crossing_time :
  (100 + 150) / (54 * 1000 / 3600) = 16.67 := by sorry

end train_crossing_time_l474_474407


namespace intersection_M_N_l474_474969

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ -3}

-- Prove the intersection of M and N is [1, 2)
theorem intersection_M_N : (M ∩ N) = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l474_474969


namespace tangent_circle_problem_l474_474684

theorem tangent_circle_problem :
  let u1 := (λ x y : ℝ, x^2 + y^2 - 6 * x + 8 * y - 23)
  let u2 := (λ x y : ℝ, x^2 + y^2 + 6 * x + 8 * y - 77)
  let center_u1 := (3, -4)
  let center_u2 := (-3, -4)
  ∃ (b : ℝ), (∀ (x y : ℝ), y = b * x → 
    let dist_u1 := sqrt ((x - 3)^2 + (y - 4)^2)
    let dist_u2 := sqrt ((x + 3)^2 + (y + 4)^2)
    6 - dist_u1 = dist_u2
  ) → (∃ (n : ℝ) (p q : ℕ), n^2 = 9 / 16 ∧ p + q = 25) := by sorry

end tangent_circle_problem_l474_474684


namespace contractor_total_received_l474_474060

-- Define the conditions
def days_engaged : ℕ := 30
def daily_earnings : ℝ := 25
def fine_per_absence_day : ℝ := 7.50
def days_absent : ℕ := 4

-- Define the days worked based on conditions
def days_worked : ℕ := days_engaged - days_absent

-- Define the total earnings and total fines
def total_earnings : ℝ := days_worked * daily_earnings
def total_fines : ℝ := days_absent * fine_per_absence_day

-- Define the total amount received
def total_amount_received : ℝ := total_earnings - total_fines

-- State the theorem
theorem contractor_total_received :
  total_amount_received = 620 := 
by
  sorry

end contractor_total_received_l474_474060


namespace greatest_int_with_conditions_l474_474781

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l474_474781


namespace num_terms_divisible_by_303_l474_474214

theorem num_terms_divisible_by_303:
  let seq := λ n : ℕ => 10^n + 1
  in (∑ n in Finset.range 3030, if seq n % 303 = 0 then 1 else 0) = 46 :=
sorry

end num_terms_divisible_by_303_l474_474214


namespace value_of_k_l474_474071

theorem value_of_k (k: ℚ) :
  (∃ l : ℚ, l ∈ set.range (λ (x : ℚ), (7 - x : ℚ) * (5 - 10 : ℚ) = (x + 11) * (k - 5)) ∧
  l ∈ set.range (λ (x : ℚ), (3 + x : ℚ) * (k - 10 : ℚ) = (x + 7) * (10 - k))) → 
  k = 65 / 9 :=
begin
  sorry
end

end value_of_k_l474_474071


namespace find_x_value_l474_474477

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) :
  (Real.tan (150 - x * Real.pi / 180) = 
   (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
   (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180))) → 
  x = 110 := 
by 
  sorry

end find_x_value_l474_474477


namespace dot_product_self_l474_474537

def vec_a : ℝ × ℝ × ℝ := (Real.sin (12 * Real.pi / 180), Real.cos (12 * Real.pi / 180), -1)

theorem dot_product_self : let a := vec_a in
  a.1 * a.1 + a.2 * a.2 + a.3 * a.3 = 2 := 
by 
  let a := vec_a
  sorry

end dot_product_self_l474_474537


namespace triangle_equilateral_l474_474826

theorem triangle_equilateral (P Q R : Point) (circle : Circle) 
(h1 : divides_into_three_equal_parts circle P Q R) : 
equilateral_triangle P Q R :=
sorry

end triangle_equilateral_l474_474826


namespace combination_eight_choose_five_l474_474631

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474631


namespace ab_value_l474_474541

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := 
by
  sorry

end ab_value_l474_474541


namespace number_of_Sunzi_books_l474_474834

theorem number_of_Sunzi_books
    (num_books : ℕ) (total_cost : ℕ)
    (price_Zhuangzi price_Kongzi price_Mengzi price_Laozi price_Sunzi : ℕ)
    (num_Zhuangzi num_Kongzi num_Mengzi num_Laozi num_Sunzi : ℕ) :
  num_books = 300 →
  total_cost = 4500 →
  price_Zhuangzi = 10 →
  price_Kongzi = 20 →
  price_Mengzi = 15 →
  price_Laozi = 30 →
  price_Sunzi = 12 →
  num_Zhuangzi = num_Kongzi →
  num_Sunzi = 4 * num_Laozi + 15 →
  num_Zhuangzi + num_Kongzi + num_Mengzi + num_Laozi + num_Sunzi = num_books →
  price_Zhuangzi * num_Zhuangzi +
  price_Kongzi * num_Kongzi +
  price_Mengzi * num_Mengzi +
  price_Laozi * num_Laozi +
  price_Sunzi * num_Sunzi = total_cost →
  num_Sunzi = 75 :=
by
  intros h_nb h_tc h_pZ h_pK h_pM h_pL h_pS h_nZ h_nS h_books h_cost
  sorry

end number_of_Sunzi_books_l474_474834


namespace three_digit_numbers_with_middle_digit_as_average_l474_474217

theorem three_digit_numbers_with_middle_digit_as_average : 
  (finset.filter 
     (λ n : ℕ, 
       let A := n / 100 in 
       let B := (n / 10) % 10 in 
       let C := n % 10 in 
       B = (A + C) / 2
     )
     (finset.range' 100 900)).card = 45 := 
sorry

end three_digit_numbers_with_middle_digit_as_average_l474_474217


namespace triangle_is_equilateral_l474_474829

-- Define the conditions
variables {P Q R A B C D E F : Type*}
-- Points where the circle intersects the triangle sides
variables [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (circ : Circle P Q R A B C D E F)

-- The main theorem to prove the triangle is equilateral
theorem triangle_is_equilateral 
  (h1 : divides_into_three_equal_parts circ A B P Q)
  (h2 : divides_into_three_equal_parts circ C D Q R)
  (h3 : divides_into_three_equal_parts circ E F R P) :
  segment_length P Q = segment_length Q R ∧ 
  segment_length Q R = segment_length R P ∧ 
  segment_length R P = segment_length P Q :=
sorry

end triangle_is_equilateral_l474_474829


namespace cost_of_staying_23_days_is_350_l474_474345

-- Define constants
def first_week_cost_per_day : ℝ := 18.00
def additional_week_cost_per_day : ℝ := 14.00
def days_in_week : ℕ := 7

-- Main proof statement
theorem cost_of_staying_23_days_is_350 :
  let first_week_days := days_in_week,
      remaining_days := 23 - first_week_days,
      full_weeks := remaining_days / days_in_week,
      full_weeks_days := full_weeks * days_in_week,
      extra_days := remaining_days % days_in_week,
      cost_first_week := (first_week_days * first_week_cost_per_day),
      cost_full_weeks := (full_weeks_days * additional_week_cost_per_day),
      cost_extra_days := (extra_days * additional_week_cost_per_day),
      total_cost := cost_first_week + cost_full_weeks + cost_extra_days
  in total_cost = 350.00 := by
  sorry

end cost_of_staying_23_days_is_350_l474_474345


namespace find_length_AD_l474_474412

theorem find_length_AD 
  (ABC_right : ∀ (A B C : Type) (AC BC : ℝ) (is_right_angled : angle A B C = π/2), (A = C ∧ B = B)) 
  (D_on_AC : ∀ (D A C : Type) (AC : ℝ), (D ∈ segment A C))
  (angle_relation : ∀ {A B C D : Type} (θ : ℝ), (angle A B C = 2 * angle D B C))
  (DC_eq_1 : ∀ D C : Type, (distance D C = 1))
  (BD_eq_3 : ∀ B D : Type, (distance B D = 3)) : 
  ∃ AD : ℝ, AD = 9/7 := 
by
  sorry

end find_length_AD_l474_474412


namespace area_relation_of_ABCD_DCE_l474_474096

theorem area_relation_of_ABCD_DCE :
  ∀ (O A B C D E : Point) (h1 : Diameter O A B) (h2 : AB = 4) (h3 : BC = 3)
    (h4 : AngleBisector O A B C D) (h5 : Intersection AD (ExtensionOf BC) E),
  (AreaQuadrilateral A B C D = 7 * AreaTriangle D C E) :=
begin
  sorry
end

end area_relation_of_ABCD_DCE_l474_474096


namespace area_of_rectangle_PQRS_l474_474260

-- Define the given conditions
variables (PQ PS RS RU RT U T : ℝ)
variables (PU : ℝ) (ST : ℝ)

-- Given conditions
axiom trisect_angle_R : angle R trisected by RT RU
axiom U_on_PQ : U on segment PQ
axiom T_on_PS : T on segment PS
axiom PU_eq_3 : PU = 3
axiom ST_eq_1 : ST = 1

-- Statement to prove
theorem area_of_rectangle_PQRS : area PQRS = 12 :=
sorry

end area_of_rectangle_PQRS_l474_474260


namespace doubletons_exist_l474_474707

theorem doubletons_exist 
  (S : Finset ℤ) 
  (hS : S.card = 46) : 
  ∃ (u v x y : ℤ), 
    {u, v} ∈ S.powerset 
    ∧ {x, y} ∈ S.powerset 
    ∧ {u, v} ≠ {x, y}
    ∧ (u + v) % 2016 = (x + y) % 2016 := 
by
  sorry

end doubletons_exist_l474_474707


namespace tan_pi_minus_alpha_eq_three_l474_474505

variable {α : Real}

theorem tan_pi_minus_alpha_eq_three (h : sin (α - Real.pi) = 3 * cos α) : tan (Real.pi - α) = 3 :=
sorry

end tan_pi_minus_alpha_eq_three_l474_474505


namespace rational_solutions_l474_474924

noncomputable def rationalPerfectSquare (x : ℚ) : Prop :=
  ∃ (n : ℚ), 1 + 105 * 2^x = n^2

theorem rational_solutions : {x : ℚ | rationalPerfectSquare x} = {3, 4, -4, -6, -8} :=
by
  -- Proof to be provided
  sorry

end rational_solutions_l474_474924


namespace student_B_final_score_l474_474082

theorem student_B_final_score :
  let correct_responses := 91
  let incorrect_responses := 100 - correct_responses
  let final_score := correct_responses - 2 * incorrect_responses
  in final_score = 73 :=
by
  let correct_responses := 91
  let incorrect_responses := 100 - correct_responses
  let final_score := correct_responses - 2 * incorrect_responses
  show final_score = 73
  sorry

end student_B_final_score_l474_474082


namespace count_non_empty_subsets_of_odd_numbers_greater_than_one_l474_474976

-- Condition definitions
def given_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def odd_numbers_greater_than_one (s : Finset ℕ) : Finset ℕ := 
  s.filter (λ x => x % 2 = 1 ∧ x > 1)

-- The problem statement
theorem count_non_empty_subsets_of_odd_numbers_greater_than_one : 
  (odd_numbers_greater_than_one given_set).powerset.card - 1 = 15 := 
by 
  sorry

end count_non_empty_subsets_of_odd_numbers_greater_than_one_l474_474976


namespace find_a7_a8_l474_474265

variable {R : Type*} [LinearOrderedField R]
variable {a : ℕ → R}

-- Conditions
def cond1 : a 1 + a 2 = 40 := sorry
def cond2 : a 3 + a 4 = 60 := sorry

-- Goal 
theorem find_a7_a8 : a 7 + a 8 = 135 := 
by 
  -- provide the actual proof here
  sorry

end find_a7_a8_l474_474265


namespace range_of_m_l474_474515

-- Define points A and B
variables (A B : ℝ × ℝ)
def A : ℝ × ℝ := (2, 1)
def B (m : ℝ) : ℝ × ℝ := (1, m)

-- Define the condition that the line passing through A and B has an acute angle of inclination
def acute_inclination (m : ℝ) : Prop :=
  let k_AB := (m - 1) / (1 - 2)
  k_AB > 0

-- The theorem to be proved
theorem range_of_m (m : ℝ) (h : acute_inclination m) : m < 1 :=
by
  sorry

end range_of_m_l474_474515


namespace arithmetic_seq_sum_ratio_l474_474669

theorem arithmetic_seq_sum_ratio (a1 d : ℝ) (S : ℕ → ℝ) 
  (hSn : ∀ n, S n = n * a1 + d * (n * (n - 1) / 2))
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 9 / S 6 = 2 :=
by
  sorry

end arithmetic_seq_sum_ratio_l474_474669


namespace iceberg_submersion_l474_474859

theorem iceberg_submersion (V_total V_immersed S_total S_submerged : ℝ) :
  convex_polyhedron ∧ floating_on_sea ∧
  V_total > 0 ∧ V_immersed > 0 ∧ S_total > 0 ∧ S_submerged > 0 ∧
  (V_immersed / V_total >= 0.90) ∧ ((S_total - S_submerged) / S_total >= 0.50) :=
sorry

end iceberg_submersion_l474_474859


namespace triangle_properties_l474_474398

-- Define a structure for the triangle with given sides.
structure Triangle where
  a b c : ℝ
  is_isosceles : a = b ∨ b = c ∨ a = c

-- Define the specific triangle with sides 13, 13, and 10.
def specific_triangle : Triangle := {
  a := 13,
  b := 13,
  c := 10,
  is_isosceles := Or.inl rfl
}

-- Define a function to compute the sum of the squares of the medians.
noncomputable def sum_of_squares_of_medians (T : Triangle) : ℝ :=
  let m_a := if T.a = T.b ∧ T.b = T.c then
               sqrt ((2 * T.a^2 + 2 * T.b^2 - T.c^2) / 4)
             else if T.a = T.b ∨ T.a = T.c then
               T.c / 2
             else if T.b = T.c then
               T.a / 2
             else
               sqrt ((2 * T.b^2 + 2 * T.c^2 - T.a^2) / 4)
  let m_b := sqrt ((2 * T.b^2 + 2 * T.c^2 - T.a^2) / 4)
  let m_c := sqrt ((2 * T.a^2 + 2 * T.c^2 - T.c^2) / 4)
  (m_a^2 + m_b^2 + m_c^2)

-- Define a function to compute the area of the triangle using Heron's formula.
noncomputable def area (T : Triangle) : ℝ :=
  let s := (T.a + T.b + T.c) / 2
  sqrt (s * (s - T.a) * (s - T.b) * (s - T.c))

-- The proof problem statement in Lean.
theorem triangle_properties : sum_of_squares_of_medians specific_triangle = 278.5 ∧ area specific_triangle = 60 := by
  sorry

end triangle_properties_l474_474398


namespace solve_C3_l474_474754

theorem solve_C3 (x y : ℝ) : 
  (∃ (g : ℝ → ℝ), (∀ x, g x = 2^(x+1) + 1) ∧ (y = g x) ↔ (x = g y)) → 
  y = log (x-1) / log 2 - 1 :=
by 
  sorry

end solve_C3_l474_474754


namespace num_positive_integers_log_inequality_l474_474910

theorem num_positive_integers_log_inequality :
  ∃ n : ℕ, n = 50 ∧ ∀ x : ℕ, 30 < x ∧ x < 90 → 
  log 10 (x - 30) + log 10 (90 - x) < 3 :=
sorry

end num_positive_integers_log_inequality_l474_474910


namespace geometric_sequence_divisible_l474_474313

theorem geometric_sequence_divisible :
  (∃ (n : ℕ), (∃ (a₁ a₂ : ℚ), a₁ = 5 / 3 ∧ a₂ = 25 ∧ ∀ k, 
  let a_k := a₁ * (15 : ℚ)^(k - 1) in a_k * (3 : ℚ)⁻¹ ^ (k - 1) * 5^k ≥ 125_000) ∧ n = 6) :=
sorry

end geometric_sequence_divisible_l474_474313


namespace probability_second_odd_given_first_odd_l474_474528

theorem probability_second_odd_given_first_odd : 
  let numbers := {1, 2, 3, 4, 5}
  let is_odd (n : ℕ) : Prop := n % 2 = 1
  let draw_without_replacement (draws : Finset ℕ) (n : ℕ) := draws.erase n
  let events := {n ∈ numbers | is_odd n}
  let first_draw := {n : ℕ | n ∈ events}
  let second_draw := {m : ℕ | m ∈ draw_without_replacement(numbers, first_draw.some)}
  let P (s : Finset ℕ) : ℚ := s.card / numbers.card
  let A := {m ∈ numbers | is_odd m}
  let B := {m ∈ draw_without_replacement(numbers, first_draw.some) | is_odd m}
  let P_A := P(A)
  let P_AB := P({n ∈ A ∪ B})
  let P_B_given_A := P_AB / P_A
  in P_B_given_A = 1/2 := sorry

end probability_second_odd_given_first_odd_l474_474528


namespace smallest_number_l474_474089

theorem smallest_number (a b c d : ℝ) (h1 : a = -5) (h2 : b = 0) (h3 : c = 1/2) (h4 : d = Real.sqrt 2) : a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by
  sorry

end smallest_number_l474_474089


namespace count_multiples_2_or_5_not_10_l474_474977

theorem count_multiples_2_or_5_not_10 (n : ℕ) : n = 200 → 
  let count_2 := (200 / 2) in
  let count_5 := (200 / 5) in
  let count_10 := (200 / 10) in
  count_2 + count_5 - count_10 - count_10 = 100 :=
by
  intros h
  sorry

end count_multiples_2_or_5_not_10_l474_474977


namespace johnny_buys_10000_balls_l474_474329

variable (x : ℕ)

theorem johnny_buys_10000_balls :
  (0.1 * (1 - 0.3) * x = 700) → x = 10000 :=
by
  intro h
  -- h: 0.1 * (1 - 0.3) * x = 700
  let price_per_ball := 0.1 * (1 - 0.3)
  have h1 : price_per_ball = 0.07 := by
    -- calculation to reduce term
    sorry
  rw [h1] at h
  -- simplified h becomes
  -- 0.07 * x = 700
  have h2 : x = 700 / 0.07 := by
    -- solve for x
    sorry
  rw h2
  -- verifying final answer
  have hx : 700 / 0.07 = 10000 := by
    -- division calculation
    sorry
  exact hx

end johnny_buys_10000_balls_l474_474329


namespace nth_term_equation_l474_474699

theorem nth_term_equation (n : ℕ) (hn : 1 ≤ n) :
    sqrt ((2 * n^2) / (2 * n + 1) - (n - 1)) = sqrt ((n + 1) * (2 * n + 1)) / (2 * n + 1) :=
by
  sorry  -- Proof goes here

end nth_term_equation_l474_474699


namespace a_three_equals_35_l474_474190

-- Define the mathematical sequences and functions
def S (n : ℕ) : ℕ := 5 * n^2 + 10 * n

def a (n : ℕ) : ℕ := S (n + 1) - S n

-- The proposition we want to prove
theorem a_three_equals_35 : a 2 = 35 := by 
  sorry

end a_three_equals_35_l474_474190


namespace choose_five_from_eight_l474_474603

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474603


namespace domain_of_f_l474_474018

noncomputable def f (x : ℝ) : ℝ := (1 / (x + 9)) + (1 / (x^2 - 9)) + (1 / (x^3 - 9))

theorem domain_of_f : 
  ∀ x : ℝ, (x ≠ -9) ∧ (x ≠ 3) ∧ (x ≠ -3) ∧ (x ≠ real.cbrt 9) ↔ 
    (x ∈ (-∞, -9) ∪ (-9, -3) ∪ (-3, 3) ∪ (3, real.cbrt 9) ∪ (real.cbrt 9, ∞)) :=
by sorry

end domain_of_f_l474_474018


namespace greatest_integer_less_than_200_with_gcd_18_l474_474768

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l474_474768


namespace circumference_of_largest_circle_l474_474410

theorem circumference_of_largest_circle (side_length : ℝ) (h : side_length = 12) : 
  let radius := side_length / 2 in
  let circumference := 2 * Real.pi * radius in
  circumference = 12 * Real.pi := by
  unfold radius circumference
  rw [h]
  norm_num
  sorry

end circumference_of_largest_circle_l474_474410


namespace power_of_imaginary_unit_l474_474494

theorem power_of_imaginary_unit (i : ℂ) (h : i^4 = 1) : i^{2016} = 1 := 
by
  sorry

end power_of_imaginary_unit_l474_474494


namespace expected_value_of_game_eq_neg_5_div_4_l474_474076

variable (pH pT : ℚ := 1/4 and 3/4 respectively properties)
variable (wH lT : ℚ := 4 and -3 respectively properties)
variable (E : ℚ := expected value computation result)

theorem expected_value_of_game_eq_neg_5_div_4 (pH pT wH lT E : ℚ)
  (h1 : pH = 1/4) (h2 : pT = 3/4) (h3 : wH = 4) (h4 : lT = -3) :
  E = -5/4 :=
begin
  sorry
end

end expected_value_of_game_eq_neg_5_div_4_l474_474076


namespace triangle_equilateral_l474_474822

variable {P Q R A B C D E F : Type*}
variable (triangle : P → Q → R → Prop)
variable (circle : Type*)
variable (intersect_eq_parts : ∀ {X Y Z}, circle → triangle X Y Z → Prop)
-- Define the condition that the circle intersects each side of the triangle into three equal parts
variable (three_eq_parts : ∀ (X Y Z : P) (c : circle), 
  triangle X Y Z → 
  intersect_eq_parts c (triangle X Y Z) →
  ∃ (A B : P), 
    (X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) ∧
    (dist X A = dist A B ∧ dist A B = dist B Y) ∧ -- PA = AB = BQ
    ∃ (C D : P), (dist Y C = dist C D ∧ dist C D = dist D Z) ∧ -- QC = CD = DR
    ∃ (E F : P), (dist Z E = dist E F ∧ dist E F = dist F X)) -- RE = EF = FP

-- The goal is to prove that the triangle is equilateral given the circle dividing each side into three equal parts.
theorem triangle_equilateral : 
  (∀ (P Q R : P), triangle P Q R → 
   (∃ (c : circle), 
     intersect_eq_parts c (triangle P Q R) ∧
     three_eq_parts P Q R c (triangle P Q R) (intersect_eq_parts c (triangle P Q R)) →
   (dist P Q = dist Q R ∧ dist Q R = dist R P) :=
begin
  sorry
end

end triangle_equilateral_l474_474822


namespace initial_house_cats_l474_474840

theorem initial_house_cats (H : ℕ) 
  (siamese_cats : ℕ := 38) 
  (cats_sold : ℕ := 45) 
  (cats_left : ℕ := 18) 
  (initial_total_cats : ℕ := siamese_cats + H) 
  (after_sale_cats : ℕ := initial_total_cats - cats_sold) : 
  after_sale_cats = cats_left → H = 25 := 
by
  intro h
  sorry

end initial_house_cats_l474_474840


namespace unique_integer_solution_l474_474339

theorem unique_integer_solution (x y z : ℤ) (h : 2 * x^2 + 3 * y^2 = z^2) : x = 0 ∧ y = 0 ∧ z = 0 :=
by {
  sorry
}

end unique_integer_solution_l474_474339


namespace complex_addition_l474_474809

def complex_number (z : ℂ) : ℝ × ℝ :=
  (z.re, z.im)

theorem complex_addition:
  let z := (1 + complex.I) * (1 - complex.I) in
  z.re + z.im = 2 := 
by 
  sorry

end complex_addition_l474_474809


namespace regions_bounded_by_blue_lines_l474_474636

theorem regions_bounded_by_blue_lines (n : ℕ) : 
  (2 * n^2 + 3 * n + 2) -(n - 1) * (2 * n + 1) ≥ 4 * n + 2 :=
by
  sorry

end regions_bounded_by_blue_lines_l474_474636


namespace characteristic_function_correct_l474_474116

variables {U : Type*} (A B : set U)

def f_char (S : set U) (x : U) : ℕ := if x ∈ S then 1 else 0

theorem characteristic_function_correct :
  (A ⊆ B → ∀ x, f_char A x ≤ f_char B x) ∧
  (∀ x, f_char Aᶜ x = 1 - f_char A x) ∧
  (∀ x, f_char (A ∩ B) x = f_char A x * f_char B x) ∧
  ¬ (∀ x, f_char (A ∪ B) x = f_char A x + f_char B x) :=
begin
  sorry
end

end characteristic_function_correct_l474_474116


namespace deriv_value_l474_474487

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem deriv_value (h : tendsto (λΔx, (f (x₀ + Δx) - f (x₀ - Δx)) / (3 * Δx)) (𝓝 0) (𝓝 1)) :
  deriv f x₀ = 3 / 2 :=
sorry

end deriv_value_l474_474487


namespace triangle_equilateral_l474_474824

theorem triangle_equilateral (P Q R : Point) (circle : Circle) 
(h1 : divides_into_three_equal_parts circle P Q R) : 
equilateral_triangle P Q R :=
sorry

end triangle_equilateral_l474_474824


namespace car_R_average_speed_l474_474375

theorem car_R_average_speed :
  ∀ (v : ℝ), (v > 0 ∧
  (800 / v) - 2 = 800 / (v + 10)) ↔ v = 50 :=
by
  intro v
  split
  · intro h
    sorry
  · intro h
    sorry

end car_R_average_speed_l474_474375


namespace heath_plants_per_hour_l474_474545

theorem heath_plants_per_hour (rows : ℕ) (plants_per_row : ℕ) (hours : ℕ) (total_plants : ℕ) :
  rows = 400 ∧ plants_per_row = 300 ∧ hours = 20 ∧ total_plants = rows * plants_per_row →
  total_plants / hours = 6000 :=
by
  sorry

end heath_plants_per_hour_l474_474545


namespace intersection_on_median_l474_474703

open EuclideanGeometry

namespace TriangleIntersectionMedian

variables {A B C E F P : Point}

/-- Initially given conditions --/
variable (h_triangle : IsTriangle A B C)
variable (h_points_on_AB : OnLineSegment A B E ∧ OnLineSegment A B F)
variable (h_equal_segments : dist A E = dist B F)
variable (h_parallel_E : Parallel (LineThrough E P) (LineThrough B C))
variable (h_parallel_F : Parallel (LineThrough F P) (LineThrough A C))

/-- Prove that the constructed lines intersect on the median of the triangle --/
theorem intersection_on_median :
  OnMedianIntersect A B C P :=
sorry

end TriangleIntersectionMedian

end intersection_on_median_l474_474703


namespace roots_polynomial_sum_reciprocals_l474_474672

-- Definition of the roots of the given polynomial
def roots_of_polynomial : fin 2023 → ℂ :=
  λ n, classical.some (exists_root_of_coeff_eq_polynomial 
    (λ k, ite (k = 2023) (-1387) 1))

-- Define the sum of the reciprocals of (1 - root)
def sum_of_reciprocals : ℂ :=
  ∑ n in finset.range 2023, 1 / (1 - roots_of_polynomial n)

-- Statement to prove
theorem roots_polynomial_sum_reciprocals (a : fin 2023 → ℂ)
  (h : ∀n, (roots_of_polynomial n) = (a n)) :
  sum_of_reciprocals = 3212.5 := by sorry

end roots_polynomial_sum_reciprocals_l474_474672


namespace bridge_angle_sum_l474_474052

theorem bridge_angle_sum 
  (A B C D E F : Type) 
  [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Geometry E] [Geometry F] 
  (AB AC DE DF AD CE: ℝ)
  (h1: AB = AC) 
  (h2: DE = DF) 
  (h3: angle A B C = 25) 
  (h4: angle D E F = 35) 
  (h5: is_parallel AD CE) : 
  angle D A C + angle A D E = 150 :=
by
  sorry

end bridge_angle_sum_l474_474052


namespace triangle_equilateral_l474_474823

variable {P Q R A B C D E F : Type*}
variable (triangle : P → Q → R → Prop)
variable (circle : Type*)
variable (intersect_eq_parts : ∀ {X Y Z}, circle → triangle X Y Z → Prop)
-- Define the condition that the circle intersects each side of the triangle into three equal parts
variable (three_eq_parts : ∀ (X Y Z : P) (c : circle), 
  triangle X Y Z → 
  intersect_eq_parts c (triangle X Y Z) →
  ∃ (A B : P), 
    (X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) ∧
    (dist X A = dist A B ∧ dist A B = dist B Y) ∧ -- PA = AB = BQ
    ∃ (C D : P), (dist Y C = dist C D ∧ dist C D = dist D Z) ∧ -- QC = CD = DR
    ∃ (E F : P), (dist Z E = dist E F ∧ dist E F = dist F X)) -- RE = EF = FP

-- The goal is to prove that the triangle is equilateral given the circle dividing each side into three equal parts.
theorem triangle_equilateral : 
  (∀ (P Q R : P), triangle P Q R → 
   (∃ (c : circle), 
     intersect_eq_parts c (triangle P Q R) ∧
     three_eq_parts P Q R c (triangle P Q R) (intersect_eq_parts c (triangle P Q R)) →
   (dist P Q = dist Q R ∧ dist Q R = dist R P) :=
begin
  sorry
end

end triangle_equilateral_l474_474823


namespace greatest_integer_gcd_l474_474777

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l474_474777


namespace class_funding_reached_l474_474582

-- Definition of the conditions
def students : ℕ := 45
def goal : ℝ := 3000
def full_payment_students : ℕ := 25
def full_payment_amount : ℝ := 60
def merit_students : ℕ := 10
def merit_payment_per_student_euro : ℝ := 40
def euro_to_usd : ℝ := 1.20
def financial_needs_students : ℕ := 7
def financial_needs_payment_per_student_pound : ℝ := 30
def pound_to_usd : ℝ := 1.35
def discount_students : ℕ := 3
def discount_payment_per_student_cad : ℝ := 68
def cad_to_usd : ℝ := 0.80
def administrative_fee_yen : ℝ := 10000
def yen_to_usd : ℝ := 0.009

-- Definitions of amounts
def full_payment_amount_total : ℝ := full_payment_students * full_payment_amount
def merit_payment_amount_total : ℝ := merit_students * merit_payment_per_student_euro * euro_to_usd
def financial_needs_payment_amount_total : ℝ := financial_needs_students * financial_needs_payment_per_student_pound * pound_to_usd
def discount_payment_amount_total : ℝ := discount_students * discount_payment_per_student_cad * cad_to_usd
def administrative_fee_usd : ℝ := administrative_fee_yen * yen_to_usd

-- Definition of total collected
def total_collected : ℝ := 
  full_payment_amount_total + 
  merit_payment_amount_total + 
  financial_needs_payment_amount_total + 
  discount_payment_amount_total - 
  administrative_fee_usd

-- The final theorem statement
theorem class_funding_reached : total_collected = 2427.70 ∧ goal - total_collected = 572.30 := by
  sorry

end class_funding_reached_l474_474582


namespace ten_digit_multiples_of_11111_l474_474208

/-
Problem Statement:
Prove that there are 3456 10-digit positive integers such that all digits are pairwise distinct 
and the integer is a multiple of 11111.

Conditions:
1. The number must be a 10-digit integer.
2. All digits (0-9) must be pairwise distinct.
3. The number must be a multiple of 11111.
-/

theorem ten_digit_multiples_of_11111 : 
  ∃ (n : ℕ), 10 ≤ n ∧ n < 10^10 ∧ (∀ (d1 d2 : ℕ), d1 ≠ d2 → decidable (d1 ∈ n.digits 10) → decidable (d2 ∈ n.digits 10)) ∧ n % 11111 = 0 :=
sorry

end ten_digit_multiples_of_11111_l474_474208


namespace list_price_is_35_l474_474445

theorem list_price_is_35 (x : ℝ) : 
  let alice_selling_price := x - 15;
      alice_commission := 0.15 * alice_selling_price;
      bob_selling_price := x - 25;
      bob_commission := 0.30 * bob_selling_price in
  alice_commission = bob_commission → x = 35 :=
by
  sorry

end list_price_is_35_l474_474445


namespace smallest_nonprime_gt1_with_prime_factors_ge_20_in_range_500_550_l474_474303

def smallest_nonprime_gt1_with_prime_factors_ge_20_and_in_range : ℕ :=
  let n := 529 in
  if (¬ (Prime n)) ∧ (∀ p : ℕ, Prime p → p ∣ n → p ≥ 20) ∧ 1 < n ∧ 500 < n ∧ n ≤ 550 then
    n
  else
    sorry

theorem smallest_nonprime_gt1_with_prime_factors_ge_20_in_range_500_550 :
  smallest_nonprime_gt1_with_prime_factors_ge_20_and_in_range = 529 :=
by
  unfold smallest_nonprime_gt1_with_prime_factors_ge_20_and_in_range
  sorry

end smallest_nonprime_gt1_with_prime_factors_ge_20_in_range_500_550_l474_474303


namespace downstream_speed_l474_474425

-- Define the speeds
def V_man_still : ℝ := 20 -- Speed of the man in still water
def V_upstream : ℝ := 15 -- Speed of the man rowing upstream

-- Define the stream speed calculation based on given conditions
def V_s : ℝ := V_man_still - V_upstream

-- Define the downstream speed calculation based on the stream speed
def V_downstream : ℝ := V_man_still + V_s

-- Prove that the speed of the man rowing downstream is 25 km/h
theorem downstream_speed : V_downstream = 25 := by
  -- Assert that V_s is calculated correctly
  have h_Vs : V_s = 5 := by
    -- Calculate V_s
    simp [V_s, V_man_still, V_upstream]
    -- Verify the calculation
    rw [show V_man_still - V_upstream = 5, by norm_num]
  -- Calculate V_downstream using the value of V_s
  simp [V_downstream, V_s, h_Vs]
  -- Verify the calculation
  rw [show V_man_still + 5 = 25, by norm_num]
  -- Finish the proof
  exact rfl

end downstream_speed_l474_474425


namespace final_answer_correct_l474_474170

-- Define the initial volume V0
def V0 := 1

-- Define the volume increment ratio for new tetrahedra
def volume_ratio := (1 : ℚ) / 27

-- Define the recursive volume increments
def ΔP1 := 4 * volume_ratio
def ΔP2 := 16 * volume_ratio
def ΔP3 := 64 * volume_ratio
def ΔP4 := 256 * volume_ratio

-- Define the total volume V4
def V4 := V0 + ΔP1 + ΔP2 + ΔP3 + ΔP4

-- The target volume as a rational number
def target_volume := 367 / 27

-- Define the fraction components
def m := 367
def n := 27

-- Define the final answer
def final_answer := m + n

-- Proof statement to verify the final answer
theorem final_answer_correct :
  V4 = target_volume ∧ (Nat.gcd m n = 1) ∧ final_answer = 394 :=
by
  -- The specifics of the proof are omitted
  sorry

end final_answer_correct_l474_474170


namespace angle_C_in_triangle_l474_474272

theorem angle_C_in_triangle (A B C : ℝ)
  (hA : A = 60)
  (hAC : C = 2 * B)
  (hSum : A + B + C = 180) : C = 80 :=
sorry

end angle_C_in_triangle_l474_474272


namespace log_23_cannot_be_computed_l474_474943

axiom log_27 : ℝ := 1.4314
axiom log_32 : ℝ := 1.5052

def can_compute_log (n : ℝ) : Prop :=
  match n with
  | 23 => False
  | _  => True -- Simplified placeholder for the sake of this statement

theorem log_23_cannot_be_computed :
  ¬ can_compute_log 23 :=
sorry

end log_23_cannot_be_computed_l474_474943


namespace range_f_l474_474959

noncomputable def f : ℝ → ℝ
| x := if x ≤ 1 then 3^(1 - x) else 1 - log x / log 3

theorem range_f (x : ℝ) : (0 ≤ x) → (f x ≤ 3) :=
begin
  sorry
end

end range_f_l474_474959


namespace sin_double_angle_l474_474180

theorem sin_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 6) = 1 / 4) : 
  Real.sin (2 * α + 5 * Real.pi / 6) = -√15 / 8 := 
sorry

end sin_double_angle_l474_474180


namespace vertical_asymptote_once_l474_474466

theorem vertical_asymptote_once (c : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + c) / (x^2 - x - 12) = (x^2 + 2*x + c) / ((x - 4) * (x + 3))) → 
  (c = -24 ∨ c = -3) :=
by 
  sorry

end vertical_asymptote_once_l474_474466


namespace centroid_distance_sum_geq_three_times_inradius_l474_474044

def triangle := {A B C : ℝ × ℝ}

def distance_point_to_line (P l1 l2: ℝ × ℝ) : ℝ :=
  abs ((l2.1 - l1.1) * (l1.2 - P.2) - (l1.1 - P.1) * (l2.2 - l1.2)) /
      sqrt ((l2.1 - l1.1)^2 + (l2.2 - l1.2)^2)

def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def inradius (A B C : ℝ × ℝ) (a b c : ℝ) (Δ : ℝ) := 
  Δ / ((a + b + c) / 2)

theorem centroid_distance_sum_geq_three_times_inradius
  (A B C : ℝ × ℝ) (a b c : ℝ) (Δ : ℝ) : 
  let G := centroid A B C
      inr := inradius A B C a b c Δ
      ha := distance_point_to_line G B C
      hb := distance_point_to_line G A C
      hc := distance_point_to_line G A B
  in
  ha + hb + hc ≥ 3 * inr := sorry

end centroid_distance_sum_geq_three_times_inradius_l474_474044


namespace greatest_integer_gcd_6_l474_474765

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l474_474765


namespace solution_set_inequality_l474_474471

open Set

theorem solution_set_inequality :
  {x : ℝ | (x+1)/(x-4) ≥ 3} = Iio 4 ∪ Ioo 4 (13/2) ∪ {13/2} :=
by
  sorry

end solution_set_inequality_l474_474471


namespace circle_radius_l474_474912

def circle_eq (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 - 12 * y + 24 = 0

theorem circle_radius (x y : ℝ) (h : circle_eq x y) : 
  ∃ r : ℝ, r = sqrt (11) / 2 := 
sorry

end circle_radius_l474_474912


namespace total_distance_traveled_l474_474416

/-- The total distance traveled by a beam of light emitted from point P 
    (1, 1, 1), reflected on the xy-plane, and absorbed at point Q (3, 3, 6)
    is equal to sqrt(57). -/
theorem total_distance_traveled :
  let P := (1 : ℝ, 1 : ℝ, 1 : ℝ)
  let Q := (3 : ℝ, 3 : ℝ, 6 : ℝ)
  let M := (1 : ℝ, 1 : ℝ, -1 : ℝ)
  dist M Q = Real.sqrt 57 :=
by
  sorry

end total_distance_traveled_l474_474416


namespace feathers_given_to_sister_l474_474653

theorem feathers_given_to_sister (hawk_feathers: ℕ) (eagle_ratio: ℕ) (remaining: ℕ)
  (total_feathers : hawk_feathers = 6)
  (eagle_feathers : eagle_ratio = 17)
  (remaining_half : remaining = 49) : ∃ feathers_given : ℕ, feathers_given = 10 :=
by
  have total_eagle_feathers : ℕ := 6 * 17
  have total_initial_feathers : ℕ := 6 + total_eagle_feathers
  have remaining_feathers_before_sell : ℕ := 2 * 49
  have feathers_given := total_initial_feathers - remaining_feathers_before_sell
  exists 10
  sorry

end feathers_given_to_sister_l474_474653


namespace combination_8_5_l474_474624

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474624


namespace find_desired_line_l474_474906

-- Definitions of the conditions in Lean
variables {x y : ℝ}

def line1 := 4 * x + 3 * y - 1 = 0
def line2 := x + 2 * y + 1 = 0
def line3 := x - 2 * y - 1 = 0

-- Lean statement that encapsulates the problem
theorem find_desired_line (h1 : line1) (h2 : line2) (h3_perp : line3) : 2 * x + y - 1 = 0 :=
sorry

end find_desired_line_l474_474906


namespace kopatych_expression_value_l474_474285

theorem kopatych_expression_value (n : ℕ) (O K S Ё Я Ж Ш И Л Ц : ℕ) :
  let f := λ x, list.take_right 2 (nat.digits 10 x)
  ∃ (n : ℕ), 
    list.perm (f (n^1)) [O, K] ∧
    list.perm (f (n^2)) [S, Ё] ∧
    list.perm (f (n^3)) [Я, Ж] ∧
    list.perm (f (n^4)) [Ш, И] ∧
    list.perm (f (n^5)) [Л, И] ∧
    Ё * Ж * И * K - Л * O * Ц * Я * Ш = 189 :=
begin
  sorry
end

end kopatych_expression_value_l474_474285


namespace janet_spent_33_percent_on_meat_l474_474649

theorem janet_spent_33_percent_on_meat :
  let broccoli_cost := 3 * 4
  let oranges_cost := 3 * 0.75
  let cabbage_cost := 3.75
  let bacon_cost := 1 * 3
  let chicken_cost := 2 * 3
  let total_meat_cost := bacon_cost + chicken_cost
  let total_grocery_cost := broccoli_cost + oranges_cost + cabbage_cost + bacon_cost + chicken_cost
  total_meat_cost / total_grocery_cost * 100 ≈ 33 := -- use ≈ for floating point comparison
by
  sorry

end janet_spent_33_percent_on_meat_l474_474649


namespace symmetric_point_correct_l474_474502

def symmetric_point (P A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₀, y₀, z₀) := A
  (2 * x₀ - x₁, 2 * y₀ - y₁, 2 * z₀ - z₁)

def P : ℝ × ℝ × ℝ := (3, -2, 4)
def A : ℝ × ℝ × ℝ := (0, 1, -2)
def expected_result : ℝ × ℝ × ℝ := (-3, 4, -8)

theorem symmetric_point_correct : symmetric_point P A = expected_result :=
  by
    sorry

end symmetric_point_correct_l474_474502


namespace similarity_of_triangles_l474_474506

variables {A B C G M E F Q P : Type*}

-- Definitions of the points and lines according to given conditions
def is_centroid (G : Type*) (A B C : Type*) := sorry -- G is the centroid of triangle ABC
def is_midpoint (M : Type*) (B C : Type*) := sorry -- M is the midpoint of side BC
def passes_through (line G E : Type*) := sorry -- line passes through G and intersects AB at E and AC at F
def parallel (line1 line2 : Type*) := sorry -- line through G parallel to BC
def intersects (line1 line2 point : Type*) := sorry -- defining intersection of lines at a point

-- Stating the problem 
theorem similarity_of_triangles
  (h1 : is_centroid G A B C)
  (h2 : is_midpoint M B C)
  (h3 : passes_through G E) (h3' : passes_through G F)
  (h4 : parallel (G, E) (B, C))
  (h5 : intersects (E, C) (B, G) Q)
  (h6 : intersects (F, B) (C, G) P) :
  similar (MPQ) (ABC) :=
sorry

end similarity_of_triangles_l474_474506


namespace largest_subset_no_multiples_l474_474019

theorem largest_subset_no_multiples : ∀ (S : Finset ℕ), (S = Finset.range 101) → 
  ∃ (A : Finset ℕ), A ⊆ S ∧ (∀ x ∈ A, ∀ y ∈ A, x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x)) ∧ A.card = 50 :=
by
  sorry

end largest_subset_no_multiples_l474_474019


namespace average_speed_of_train_l474_474435

theorem average_speed_of_train
  (d1 d2 : ℝ) (t1 t2 : ℝ)
  (h1 : d1 = 290) (h2 : d2 = 400) (h3 : t1 = 4.5) (h4 : t2 = 5.5) :
  ((d1 + d2) / (t1 + t2)) = 69 :=
by
  -- proof steps can be filled in later
  sorry

end average_speed_of_train_l474_474435


namespace comb_8_5_eq_56_l474_474599

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474599


namespace projection_vector_l474_474789

variables {R : Type*} [Field R] (v p : R^2)

def a : R^2 := ![6, 2]
def b : R^2 := ![-2, 4]
def expected_p : R^2 := ![14 / 17, 56 / 17]

theorem projection_vector :
  (∃ v : R^2, (∃ p : R^2, 
    ∀ u : R^2, (u = a ∨ u = b) → (u - p) ⬝ v = 0)) → 
  p = expected_p := sorry

end projection_vector_l474_474789


namespace coefficient_of_x3_is_10_l474_474347

noncomputable def find_a (a : ℝ) : ℝ :=
  if (∀ x : ℝ, x ≠ 0 → (∑ r in finset.range 6, (nat.choose 5 r) * (x^(5 - r) * (a/x)^r) * if (5 - 2*r) = 3 then 1 else 0) = 10)
  then a
  else 0

theorem coefficient_of_x3_is_10 : find_a 2 = 2 :=
by
  sorry

end coefficient_of_x3_is_10_l474_474347


namespace ball_initial_height_l474_474442

theorem ball_initial_height (c : ℝ) (d : ℝ) (h : ℝ) 
  (H1 : c = 4 / 5) 
  (H2 : d = 1080) 
  (H3 : d = h + 2 * h * c / (1 - c)) : 
  h = 216 :=
sorry

end ball_initial_height_l474_474442


namespace concurrency_of_reflected_lines_l474_474441

open Classical

variables {A B C O U V W : Type*} [Inhabited A] [Inhabited B] [Inhabited C]
variables [InCircle A B C O] [Reflection O U A] [Reflection O V B] [Reflection O W C]
variables (H : Orthocenter A B C)
variables {l_A l_B l_C : Type*} [Parallel l_A A V W] [Parallel l_B B W U] [Parallel l_C C U V]

theorem concurrency_of_reflected_lines :
  concurrency l_A l_B l_C :=
begin
  sorry
end

end concurrency_of_reflected_lines_l474_474441


namespace all_lamps_on_iff_even_l474_474888

-- Define the grid and the switching mechanism

def lamp_state (n : ℕ) := (Σ (i : fin n) (j : fin n), bool)

def toggle_lamps (n : ℕ) (initial_state : lamp_state n) : lamp_state n :=
  let toggle_state (s : lamp_state n) (pos : Σ (i : fin n) (j : fin n)) : lamp_state n :=
    ⟨pos.1, pos.2, bnot (s.2)⟩ -- toggle the state at position pos

  let row_col_toggle (s : lamp_state n) (i j : fin n) : lamp_state n :=
    s.update (λ pos, if pos.1 = i ∨ pos.2 = j then ⟨pos.1, pos.2, bnot (pos.2)⟩ else pos)

  -- applying the toggle on all required positions
  initial_state.map (λ pos, row_col_toggle (toggle_state initial_state pos) pos.1 pos.2)

-- Define the main theorem
theorem all_lamps_on_iff_even (n : ℕ) :
  (∀ initial_state : lamp_state n, ∃ final_state : lamp_state n, final_state = toggle_lamps n initial_state)
  ↔ n % 2 = 0 :=
sorry

end all_lamps_on_iff_even_l474_474888


namespace find_n_l474_474923

theorem find_n (n : ℕ) (h : Real.root n (17 * Real.sqrt 5 + 38) + Real.root n (17 * Real.sqrt 5 - 38) = Real.sqrt 20) : n = 3 :=
sorry

end find_n_l474_474923


namespace proportion_is_equation_l474_474077

/-- A proportion containing unknowns is an equation -/
theorem proportion_is_equation (P : Prop) (contains_equality_sign: Prop)
  (indicates_equality : Prop)
  (contains_unknowns : Prop) : (contains_equality_sign ∧ indicates_equality ∧ contains_unknowns ↔ True) := by
  sorry

end proportion_is_equation_l474_474077


namespace problem_ab_plus_a_plus_b_l474_474306

noncomputable def polynomial := fun x : ℝ => x^4 - 6 * x - 2

theorem problem_ab_plus_a_plus_b :
  ∀ (a b : ℝ), polynomial a = 0 → polynomial b = 0 → (a * b + a + b) = 4 :=
by
  intros a b ha hb
  sorry

end problem_ab_plus_a_plus_b_l474_474306


namespace mike_daily_work_hours_l474_474696

def total_hours_worked : ℕ := 15
def number_of_days_worked : ℕ := 5

theorem mike_daily_work_hours : total_hours_worked / number_of_days_worked = 3 :=
by
  sorry

end mike_daily_work_hours_l474_474696


namespace compute_f_2024_l474_474833

noncomputable def f : ℝ → ℝ := sorry

axiom even_f_x_plus_1 : ∀ x, f(x + 1) = f(1 - x)
axiom recurrence_relation : ∀ x, f(x + 2) = f(x + 1) - f(x)
axiom initial_condition : f(1) = 1 / 2

theorem compute_f_2024 : f(2024) = 1 / 4 :=
by
  sorry

end compute_f_2024_l474_474833


namespace tangent_circles_l474_474757

-- Definitions of points and circles
variables (P Q A1 B1 C1 D1 A2 B2 C2 D2 R : Point)
variables (circle1 circle2 : Circle)

-- Assumptions from the problem
-- Two circles intersect at points P and Q
axiom h1 : P ∈ circle1
axiom h2 : Q ∈ circle1
axiom h3 : P ∈ circle2
axiom h4 : Q ∈ circle2

-- A line through P intersects the circles again at points A1 and B1
axiom h5 : line P intersects circle1 = [A1, B1]
axiom h6 : line P intersects circle2 = [A2, B2]

-- The tangents at A1 and B1 to the circumcircle of triangle A1RB1 intersect at point C1
axiom h7 : tangent A1 P intersects tangent B1 P = C1

-- Line C1R intersects A1B1 at point D1
axiom h8 : line C1 R intersects line A1 B1 = D1

-- Similarly defined points for the second set of points
axiom h9 : tangent A2 P intersects tangent B2 P = C2
axiom h10 : line C2 R intersects line A2 B2 = D2

-- The statement to prove tangency of circles through D1, D2, P and C1, C2, R
theorem tangent_circles : tangent (circle D1 D2 P) (circle C1 C2 R) = true := sorry

end tangent_circles_l474_474757


namespace b_is_geometric_a_general_formula_sum_na_formula_l474_474748

-- Definition of the sequence sum
def S (n : ℕ) : ℤ :=
  2 * (a n) - 3 * n

-- Definition of the sequence a_n using conditions
def a : ℕ → ℤ
| 1       := 3
| (n + 1) := 2 * a n + 3

-- Definition of the sequence b_n
def b (n : ℕ) : ℤ :=
  a n + 3

-- Proof statement for Problem 1
theorem b_is_geometric (n : ℕ) : b (n + 1) = 2 * b n :=
by sorry

-- General formula for a_n
theorem a_general_formula (n : ℕ) : a n = 3 * 2 ^ n - 3 :=
by sorry

-- Definition of the sequence n * a_n
def na (n : ℕ) : ℤ :=
  n * a n

-- Proof statement for Problem 2
theorem sum_na_formula (n : ℕ) :
  ∑ k in Finset.range n, na (k + 1) = (6 * n - 6) * 2 ^ n + 6 - (3 * n * (n + 1)) / 2 :=
by sorry

end b_is_geometric_a_general_formula_sum_na_formula_l474_474748


namespace intersection_vectors_magnitude_l474_474200

theorem intersection_vectors_magnitude (k : ℝ) (hk : k > 0) :
  let f1 := λ x : ℝ, k * x + 1,
      f2 := λ x : ℝ, (x + 1) / x in
  -- Define intersection points A and B
  let A := (1 / real.sqrt k, 1 + real.sqrt k),
      B := (-1 / real.sqrt k, 1 - real.sqrt k),
      -- Sum of vectors OA and OB
      sum_vector := (0, 2 : ℝ) in
  -- Magnitude of the sum of vectors
  real.sqrt ((sum_vector.1 ^ 2) + (sum_vector.2 ^ 2)) = 2 := sorry

end intersection_vectors_magnitude_l474_474200


namespace combination_seven_choose_four_l474_474716

theorem combination_seven_choose_four : nat.choose 7 4 = 35 := by
  -- Proof
  sorry

end combination_seven_choose_four_l474_474716


namespace carrots_planted_per_hour_l474_474548

theorem carrots_planted_per_hour (rows plants_per_row hours : ℕ) (h1 : rows = 400) (h2 : plants_per_row = 300) (h3 : hours = 20) :
  (rows * plants_per_row) / hours = 6000 := by
  sorry

end carrots_planted_per_hour_l474_474548


namespace distance_between_given_parallel_lines_l474_474873

noncomputable def distance_between_parallel_lines 
  (a1 a2 : ℝ) 
  (b1 b2 : ℝ) 
  (d1 d2 : ℝ) : ℝ :=
let a : ℝ × ℝ := (a1, a2),
    b : ℝ × ℝ := (b1, b2),
    d : ℝ × ℝ := (d1, d2) in
let v : ℝ × ℝ := (b.1 - a.1, b.2 - a.2) in
let dot (x y : ℝ × ℝ) := x.1 * y.1 + x.2 * y.2 in
let p : ℝ × ℝ := (dot v d) / (dot d d) • d in
let c : ℝ × ℝ := (v.1 - p.1, v.2 - p.2) in
real.sqrt ((c.1 * c.1 + c.2 * c.2) : ℝ)

theorem distance_between_given_parallel_lines : 
  distance_between_parallel_lines 
    4 (-1) 
    3 (-2) 
    2 (-6) = 
    (2 * real.sqrt 10) / 5 :=
sorry

end distance_between_given_parallel_lines_l474_474873


namespace complement_M_l474_474534

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M (U M : Set ℝ) : (U \ M) = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end complement_M_l474_474534


namespace marbles_left_l474_474074

/-- Proving the total number of marbles left in the magician's hat, given initial conditions and rules for removal. -/
theorem marbles_left (initial_red : ℕ) (initial_blue : ℕ) (initial_green : ℕ) (red_taken : ℕ) :
  initial_red = 40 → 
  initial_blue = 60 →
  initial_green = 35 →
  red_taken = 5 →
  let blue_taken := 5 * red_taken in
  let green_taken := blue_taken / 2 in
  initial_red - red_taken + (initial_blue - blue_taken) + (initial_green - green_taken) = 93 :=
by {
  intros h_red h_blue h_green h_taken,
  simp only,
  sorry
}

end marbles_left_l474_474074


namespace objects_meeting_time_l474_474067

theorem objects_meeting_time 
  (initial_velocity : ℝ) (g : ℝ) (t_delay : ℕ) (t_meet : ℝ) 
  (hv : initial_velocity = 120) 
  (hg : g = 9.8) 
  (ht : t_delay = 5)
  : t_meet = 14.74 :=
sorry

end objects_meeting_time_l474_474067


namespace problem1_problem2_l474_474687

noncomputable def f (a x : ℝ) : ℝ := a * sin (2 * x) + 2 * cos (x) ^ 2

theorem problem1 (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

noncomputable def f_sqrt3 (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * cos (x) ^ 2

theorem problem2 :
  f_sqrt3 (π / 4) = sqrt 3 + 1 →
  (sqrt 3 * sin (2 * x) + 2 * cos (x) ^ 2 = 1 - sqrt 2) →
  -π ≤ x ∧ x ≤ π →
  x = -5 * π / 12 ∨ x = 13 * π / 12 := by
  sorry

end problem1_problem2_l474_474687


namespace opposite_of_neg5_l474_474360

-- Define the concept of the opposite of a number
def opposite (x : Int) : Int :=
  -x

-- The proof problem: Prove that the opposite of -5 is 5
theorem opposite_of_neg5 : opposite (-5) = 5 :=
by
  sorry

end opposite_of_neg5_l474_474360


namespace sequence_count_pos_iff_l474_474922

-- Definitions based on the given conditions
def valid_sequence (n d : ℕ) (seq : Fin d → Fin (n+1) ) : Prop :=
  (∀ i : Fin d, seq i ∈ Fin (n+1)) ∧
  (∀ i : Fin (d - 1), seq i ≠ seq ⟨i.val.succ, i.is_lt.succ $ Nat.pred_ltₓ $ zero_lt_iff.mpr (Nat.succ_posₓ d)⟩.1) ∧
  (∀ (i j k l : Fin d), i.val < j.val → j.val < k.val → k.val < l.val → 
    seq i = seq k → seq j = seq l → False)

def S_n_d (n d : ℕ) := { seq : Fin d → Fin (n+1) // valid_sequence n d seq }

-- The theorem statement
theorem sequence_count_pos_iff (n d : ℕ) : 0 < (S_n_d n d).toFinset.card ↔ d ≤ 2 * n - 1 :=
by
  sorry

end sequence_count_pos_iff_l474_474922


namespace mean_height_is_60_l474_474372

def heights : List ℕ := [49, 52, 53, 55, 58, 58, 59, 60, 61, 61, 62, 66, 68, 69, 69]

def mean_height (xs : List ℕ) : ℕ :=
  xs.sum / xs.length

theorem mean_height_is_60 (heights : List ℕ) (h_len : heights.length = 15) (h_sum : heights.sum = 900) :
  mean_height heights = 60 :=
by
  rw [mean_height, h_sum, h_len]
  exact Nat.div_eq_of_eq_mul (by norm_num) (by norm_num)

#check mean_height_is_60

end mean_height_is_60_l474_474372


namespace find_added_purple_socks_l474_474280

variable (initial_green_socks : ℕ) (initial_purple_socks : ℕ) (initial_orange_socks : ℕ) (x : ℕ) 

-- Conditions
def initial_total_socks : ℕ := initial_green_socks + initial_purple_socks + initial_orange_socks
def new_total_purple_socks : ℕ := initial_purple_socks + x
def new_total_socks : ℕ := initial_total_socks + x

-- Probability condition
def probability_of_purple (x : ℕ) : Prop :=
  (new_total_purple_socks initial_purple_socks x) / (new_total_socks initial_green_socks initial_purple_socks initial_orange_socks x) = 3 / 5

-- The proof problem
theorem find_added_purple_socks
  (h1 : initial_green_socks = 6)
  (h2 : initial_purple_socks = 18)
  (h3 : initial_orange_socks = 12)
  (h4 : probability_of_purple initial_green_socks initial_purple_socks initial_orange_socks x) :
  x = 9 :=
sorry

end find_added_purple_socks_l474_474280


namespace tetrahedron_edge_length_l474_474919

-- Definitions corresponding to the conditions of the problem.
def radius : ℝ := 2

def diameter : ℝ := 2 * radius

/-- Centers of four mutually tangent balls -/
def center_distance : ℝ := diameter

/-- The side length of the square formed by the centers of four balls on the floor. -/
def side_length_of_square : ℝ := center_distance

/-- The edge length of the tetrahedron circumscribed around the four balls. -/
def edge_length_tetrahedron : ℝ := side_length_of_square

-- The statement to be proved.
theorem tetrahedron_edge_length :
  edge_length_tetrahedron = 4 :=
by
  sorry  -- Proof to be constructed

end tetrahedron_edge_length_l474_474919


namespace regions_bounded_by_blue_lines_l474_474637

theorem regions_bounded_by_blue_lines (n : ℕ) : 
  (2 * n^2 + 3 * n + 2) -(n - 1) * (2 * n + 1) ≥ 4 * n + 2 :=
by
  sorry

end regions_bounded_by_blue_lines_l474_474637


namespace greatest_int_with_conditions_l474_474779

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l474_474779


namespace hyperbola_equation_l474_474905

-- Definitions for the given problem.
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 5) = 1
def perpendicular_asymptotes (x y : ℝ) : Prop := x^2 - y^2 = λ

-- Main statement of the theorem
theorem hyperbola_equation : ∃ λ : ℝ, (λ = 8) ∧ ∀ x y : ℝ, ellipse x y → perpendicular_asymptotes x y := by
  sorry

end hyperbola_equation_l474_474905


namespace game_show_valid_guesses_l474_474422

theorem game_show_valid_guesses :
  let digits := [1, 1, 1, 2, 2, 3, 3, 3, 3]
  let num_prizes := 3
  let total_digits := 9
  let permutations := Nat.fact total_digits / (Nat.fact 3 * Nat.fact 2 * Nat.fact 4)
  let valid_partitions := Nat.choose (total_digits - 1) (num_prizes - 1) - 27
  permutations * valid_partitions = 1050 :=
by
  let digits := [1, 1, 1, 2, 2, 3, 3, 3, 3]
  let num_prizes := 3
  let total_digits := 9
  let permutations := Nat.fact total_digits / (Nat.fact 3 * Nat.fact 2 * Nat.fact 4)
  let valid_partitions := Nat.choose (total_digits - 1) (num_prizes - 1) - 27
  show permutations * valid_partitions = 1050
  sorry

end game_show_valid_guesses_l474_474422


namespace angle_bisector_eq_30_l474_474257

-- Define the structure of an equilateral triangle
def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define angle bisector
def angle_bisector (A B C D : Point) : Prop :=
  ∠ A B D = ∠ D B C

-- Problem statement: Prove m∠CBD = 30° given the conditions
theorem angle_bisector_eq_30
  {A B C D : Point}
  (h1 : equilateral_triangle A B C)
  (h2 : angle_bisector A B C D)
  (h3 : m∠ABC = 60) : m∠CBD = 30 :=
by sorry

end angle_bisector_eq_30_l474_474257


namespace intersection_M_N_eq_interval_l474_474970

noncomputable def M : Set ℝ := {x | log10 (1 - x) < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def expected : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N_eq_interval :
  (M ∩ N) = expected :=
by 
  sorry

end intersection_M_N_eq_interval_l474_474970


namespace janet_spent_33_percent_on_meat_l474_474650

theorem janet_spent_33_percent_on_meat :
  let broccoli_cost := 3 * 4
  let oranges_cost := 3 * 0.75
  let cabbage_cost := 3.75
  let bacon_cost := 1 * 3
  let chicken_cost := 2 * 3
  let total_meat_cost := bacon_cost + chicken_cost
  let total_grocery_cost := broccoli_cost + oranges_cost + cabbage_cost + bacon_cost + chicken_cost
  total_meat_cost / total_grocery_cost * 100 ≈ 33 := -- use ≈ for floating point comparison
by
  sorry

end janet_spent_33_percent_on_meat_l474_474650


namespace count_functions_satisfying_condition_l474_474733

-- Define the function and its condition
def satisfies_condition (f : {1, 2, 3} → {1, 2, 3}) : Prop :=
  ∀ x, f (f x) ≥ f x

-- Define the count of all such functions
def count_satisfying_functions : ℕ :=
  10

-- Proof statement
theorem count_functions_satisfying_condition :
  ∑ (f : {1, 2, 3} → {1, 2, 3}), if satisfies_condition f then 1 else 0 = count_satisfying_functions :=
by
  sorry

end count_functions_satisfying_condition_l474_474733


namespace electrician_hourly_wage_l474_474817

theorem electrician_hourly_wage
    (total_hours : ℕ)
    (bricklayer_hours : ℕ)
    (electrician_hours : ℕ)
    (bricklayer_rate : ℕ)
    (total_payment : ℕ)
    (bricklayer_payment : ℕ)
    (electrician_payment : ℕ)
    (electrician_rate : ℕ):
    total_hours = bricklayer_hours + electrician_hours →
    bricklayer_rate = 12 →
    total_payment = 1350 →
    bricklayer_hours = electrician_hours →
    electrician_hours = 22.5 →
    bricklayer_payment = bricklayer_hours * bricklayer_rate →
    electrician_payment = total_payment - bricklayer_payment →
    electrician_rate = electrician_payment / electrician_hours →
    electrician_rate = 48 :=
by
  intros
  sorry

end electrician_hourly_wage_l474_474817


namespace find_a_l474_474491

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 2 then x^2 - 4 else |x - 3| + a

theorem find_a (a : ℝ) (h : f (f (Real.sqrt 6) a) a = 3) : a = 2 := by
  sorry

end find_a_l474_474491


namespace number_of_M_subsets_l474_474560

def P : Set ℤ := {0, 1, 2}
def Q : Set ℤ := {0, 2, 4}

theorem number_of_M_subsets (M : Set ℤ) (hP : M ⊆ P) (hQ : M ⊆ Q) : 
  ∃ n : ℕ, n = 4 :=
by
  sorry

end number_of_M_subsets_l474_474560


namespace eu_countries_2012_forms_set_l474_474027

def higher_level_skills_students := false -- Condition A can't form a set.
def tall_trees := false -- Condition B can't form a set.
def developed_cities := false -- Condition D can't form a set.
def eu_countries_2012 := true -- Condition C forms a set.

theorem eu_countries_2012_forms_set : 
  higher_level_skills_students = false ∧ tall_trees = false ∧ developed_cities = false ∧ eu_countries_2012 = true :=
by {
  sorry
}

end eu_countries_2012_forms_set_l474_474027


namespace magnitude_parallel_vectors_l474_474539

theorem magnitude_parallel_vectors :
  ∀ (m : ℝ), (∃ k : ℝ, ∀ a b : ℝ × ℝ, a = (1, 2) ∧ b = (-2, m) ∧ b = k • a) →
    |2 • (1, 2 : ℝ × ℝ) + 3 • (-2, m : ℝ × ℝ)| = 4 * real.sqrt 5 :=
begin
  intros m h,
  sorry
end

end magnitude_parallel_vectors_l474_474539


namespace finitely_many_operations_l474_474454

structure Point :=
  (x : ℤ)
  (y : ℤ)

def distance_sq (A B : Point) : ℤ :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2

def is_parallelogram (A B C D : Point) : Prop :=
  (A.x + B.x = C.x + D.x) ∧ (A.y + B.y = C.y + D.y)

theorem finitely_many_operations (S : finset Point) :
  (∀ (A B : Point), A ∈ S → B ∈ S → A ≠ B → 
  ∃ (C D : Point), C ∉ S ∧ D ∉ S ∧ is_parallelogram A B C D ∧ distance_sq A B > distance_sq C D) →
  ∃ N, ∀ k > N, ¬ (∃ (S' : finset Point), S'.card = k ∧ 
  finset.disjoint S S' ∧ 
  (∀ (A B : Point), A ∈ S → B ∈ S → A ≠ B → 
  ∃ (C D : Point), C ∈ S' ∧ D ∈ S' ∧ is_parallelogram A B C D ∧ distance_sq A B > distance_sq C D)) :=
sorry

end finitely_many_operations_l474_474454


namespace pilak_divisors_unique_l474_474079

-- Definition of consecutive Fibonacci numbers
def is_fibonacci (n : ℕ) : Prop :=
  ∃ a b c d: ℕ, (a = 1 ∧ b = 1)
  ∧ (∀ n. nat.bodd n = tt → fib n + fib (n + 1) = fib (n + 2))

-- Definition of consecutive triangular numbers
def is_triangular (n : ℕ) : Prop :=
  ∃ m₁ m₂ : ℕ, (m₁ * (m₁ + 1)) / 2 = n ∧ (m₂ * (m₂ + 1)) / 2 = n

-- Definition of pilak set
def is_pilak (s : Set ℕ) : Prop :=
  ∃ (F T : Set ℕ), 
    (Disjoint F T) ∧
    (2 ≤ F.card) ∧ (2 ≤ T.card) ∧
    (∀ f ∈ F, is_fibonacci f) ∧
    (∀ t ∈ T, is_triangular t)

-- Set of divisors of n except n itself
def proper_divisors (n : ℕ) : Set ℕ :=
  { d | d ∣ n ∧ d < n }

theorem pilak_divisors_unique :
  ∀ n : ℕ, is_pilak (proper_divisors n) ↔ n = 30 := 
by
  sorry

end pilak_divisors_unique_l474_474079


namespace dragon_defeated_l474_474251

universe u

inductive Dragon
| heads (count : ℕ) : Dragon

open Dragon

-- Conditions based on probabilities
def chop_probability : ℚ := 1 / 4
def one_grow_probability : ℚ := 1 / 3
def no_grow_probability : ℚ := 5 / 12

noncomputable def probability_defeated_dragon : ℚ :=
  if h : chop_probability + one_grow_probability + no_grow_probability = 1 then
    -- Define the recursive probability of having zero heads eventually (∞ case)
    let rec prob_defeat (d : Dragon) : ℚ :=
      match d with
      | heads 0     => 1  -- Base case: no heads mean dragon is defeated
      | heads (n+1) => 
        no_grow_probability * prob_defeat (heads n) + -- Successful strike
        one_grow_probability * prob_defeat (heads (n + 1)) + -- Neutral strike
        chop_probability * prob_defeat (heads (n + 2)) -- Unsuccessful strike
    prob_defeat (heads 3)  -- Initial condition with 3 heads
  else 0

-- Final theorem statement asserting the probability of defeating the dragon => 1
theorem dragon_defeated : probability_defeated_dragon = 1 :=
sorry

end dragon_defeated_l474_474251


namespace emily_investment_change_l474_474581

def initial_investment : ℝ := 200
def first_year_loss_rate : ℝ := 0.10
def second_year_gain_rate : ℝ := 0.15
def third_year_loss_rate : ℝ := 0.05

theorem emily_investment_change :
    let final_investment : ℝ := 
        let after_first_year := initial_investment * (1 - first_year_loss_rate) in
        let after_second_year := after_first_year * (1 + second_year_gain_rate) in
        let after_third_year := after_second_year * (1 - third_year_loss_rate) in
        after_third_year,
    let percentage_change : ℝ := 
        ((final_investment - initial_investment) / initial_investment) * 100 in
    percentage_change = -1.67 := by
    sorry

end emily_investment_change_l474_474581


namespace complement_union_AB_is_correct_l474_474689

-- Define the universal set U
def U := { x : ℕ | x < 6 ∧ x > 0 }

-- Define sets A and B
def A := {1, 3} : Set ℕ
def B := {3, 5} : Set ℕ

-- Define the union of sets A and B
def union_AB := A ∪ B

-- Define the complement of the union_AB relative to U
def complement_union_AB := { x ∈ U | x ∉ union_AB }

-- State the theorem to prove
theorem complement_union_AB_is_correct : 
  complement_union_AB = {2, 4} := 
sorry

end complement_union_AB_is_correct_l474_474689


namespace solve_n_minus_m_l474_474309

theorem solve_n_minus_m :
  ∃ m n, 
    (m ≡ 4 [MOD 7]) ∧ 100 ≤ m ∧ m < 1000 ∧ 
    (n ≡ 4 [MOD 7]) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
    n - m = 903 :=
by
  sorry

end solve_n_minus_m_l474_474309


namespace points_on_ellipse_l474_474485

theorem points_on_ellipse (t : ℝ) : 
  let x := 2 * Real.cos t
  let y := 3 * Real.sin t
  in (x, y) = (2 * Real.cos t, 3 * Real.sin t) → (x^2 / 4 + y^2 / 9 = 1) :=
by
  intro t
  let x := 2 * Real.cos t
  let y := 3 * Real.sin t
  have h_cos_sin : Real.cos t ^ 2 + Real.sin t ^ 2 = 1 := Real.cos_sq_add_sin_sq t
  calc 
    (x^2 / 4 + y^2 / 9) = ((2 * Real.cos t)^2 / 4 + (3 * Real.sin t)^2 / 9) : by rfl
                     ... = (4 * (Real.cos t)^2 / 4 + 9 * (Real.sin t)^2 / 9) : by sorry
                     ... = (Real.cos t)^2 + (Real.sin t)^2 : by sorry
                     ... = 1 : h_cos_sin

end points_on_ellipse_l474_474485


namespace son_can_do_work_alone_in_10_days_l474_474803

theorem son_can_do_work_alone_in_10_days : 
  ∀ (M S: ℝ), 
    M = 1 / 10 → 
    M + S = 1 / 5 → 
    1 / S = 10 :=
by 
  -- assume conditions
  intros M S hM hMS,
  -- define intermediate results and provide final goal statement
  sorry

end son_can_do_work_alone_in_10_days_l474_474803


namespace total_notebooks_l474_474796

-- Definitions from the conditions
def Yoongi_notebooks : Nat := 3
def Jungkook_notebooks : Nat := 3
def Hoseok_notebooks : Nat := 3

-- The proof problem
theorem total_notebooks : Yoongi_notebooks + Jungkook_notebooks + Hoseok_notebooks = 9 := 
by 
  sorry

end total_notebooks_l474_474796


namespace finite_order_elements_subgroup_l474_474045

variable {G : Type*} [Group G]

def commutator_subgroup (G : Type*) [Group G] : Subgroup G := Subgroup.normalClosure (set_of (λ g1, ∀ (g2 : G), g1 * g2 * g1⁻¹ * g2⁻¹ = 1))

def has_finite_order (g : G) : Prop := ∃ n : ℕ, 0 < n ∧ g ^ n = 1

def T (G : Type*) [Group G] : set G := { g : G | has_finite_order g }

theorem finite_order_elements_subgroup (G : Type*) [Group G] (h : ∀ g ∈ commutator_subgroup G, has_finite_order g) : 
  is_subgroup {g : G | has_finite_order g} :=
sorry

end finite_order_elements_subgroup_l474_474045


namespace c_plus_d_eq_130_l474_474155

theorem c_plus_d_eq_130 (c d : ℕ) (h1 : c > 0) (h2 : d > 0) 
  (h3 : log c (c+2) * log (c+2) (c+4) * ... * log (d-4) (d-2) * log (d-2) d = 3) 
  (h4 : (d-2) - c + 2 = 435) : 
  c + d = 130 := sorry

end c_plus_d_eq_130_l474_474155


namespace only_swimming_students_l474_474468

theorem only_swimming_students (total_students : ℕ) (students_swimming : ℕ) 
  (students_track_and_field : ℕ) (students_ball_games : ℕ) 
  (students_swimming_track_and_field : ℕ) 
  (students_swimming_ball_games : ℕ) 
  (no_one_in_all_three : Prop) 
  (h_total : total_students = 28)
  (h_swimming : students_swimming = 15)
  (h_track_and_field : students_track_and_field = 8)
  (h_ball_games : students_ball_games = 14)
  (h_swimming_track_and_field : students_swimming_track_and_field = 3)
  (h_swimming_ball_games : students_swimming_ball_games = 3)
  (h_no_one_in_all_three : no_one_in_all_three = (¬(∃ n, n ≤ total_students ∧ n = students_swimming ∧ n = students_track_and_field ∧ n = students_ball_games))) :
  (students_swimming - (students_swimming_track_and_field + students_swimming_ball_games) = 9) :=
by
  rw [h_swimming, h_swimming_track_and_field, h_swimming_ball_games]
  norm_num

end only_swimming_students_l474_474468


namespace cos_sq_minus_sin_sq_l474_474955

noncomputable def alpha : ℝ := sorry

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem cos_sq_minus_sin_sq : Real.cos alpha ^ 2 - Real.sin alpha ^ 2 = -3/5 := by
  sorry

end cos_sq_minus_sin_sq_l474_474955


namespace smaller_number_l474_474000

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 :=
by
  sorry

end smaller_number_l474_474000


namespace find_four_real_numbers_l474_474902

theorem find_four_real_numbers
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 * x3 * x4 = 2)
  (h2 : x2 + x1 * x3 * x4 = 2)
  (h3 : x3 + x1 * x2 * x4 = 2)
  (h4 : x4 + x1 * x2 * x3 = 2) :
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) :=
sorry

end find_four_real_numbers_l474_474902


namespace function_satisfies_properties_l474_474794

def periodic (f : ℝ → ℝ) (T : ℝ) := ∀ x : ℝ, f (x + T) = f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x₁ x₂ : ℝ, x₁ ∈ set.Icc a b → x₂ ∈ set.Icc a b → x₁ < x₂ → f x₁ > f x₂

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem function_satisfies_properties : 
  periodic (λ x, -sin (π * x / 2)) 4 ∧ 
  decreasing_on_interval (λ x, -sin (π * x / 2)) 0 1 ∧ 
  odd_function (λ x, -sin (π * x / 2)) :=
by
  sorry

end function_satisfies_properties_l474_474794


namespace north_american_birth_month_percentage_l474_474353

theorem north_american_birth_month_percentage :
  ∀ (total_north_americans born_in_october : ℕ),
    total_north_americans = 120 →
    born_in_october = 14 →
    (born_in_october / total_north_americans:ℚ * 100) = 1167 / 100 :=
begin
  intros total_north_americans born_in_october h1 h2,
  rw [h1, h2],
  norm_num,
end

end north_american_birth_month_percentage_l474_474353


namespace quadratic_solution_l474_474364

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let Δ := b^2 - 4 * a * c
  ( (-b + real.sqrt Δ) / (2 * a), (-b - real.sqrt Δ) / (2 * a) )

theorem quadratic_solution (d : ℝ) : 
  (quadratic_roots 1 7 d = ( (-7 + real.sqrt (d+1)) / 2, (-7 - real.sqrt (d+1)) / 2 )) → 
  d = 48 / 5 :=
by
  sorry

end quadratic_solution_l474_474364


namespace log_m_n_suff_not_nec_l474_474931

theorem log_m_n_suff_not_nec (m n : ℝ) (h1 : m > 0) (h2 : m ≠ 1) :
  (log m n > 0) → (1 - m) * (1 - n) > 0 ∧
  ¬ ((1 - m) * (1 - n) > 0 → log m n > 0) :=
by
  sorry

end log_m_n_suff_not_nec_l474_474931


namespace minimum_value_f4_2_l474_474858

-- Definitions of the functions given in the conditions
def f1 (x : ℝ) : ℝ := x + 1/x
def f2 (x : ℝ) : ℝ := sin x + 1/sin x
def f3 (x : ℝ) : ℝ := (x^2 + 3) / sqrt(x^2 + 2)
def f4 (x : ℝ) : ℝ := exp x + 4/exp x - 2

-- Proving the function with minimum value 2
theorem minimum_value_f4_2 : ∀ x : ℝ, (f4 x ≤ 2 → f4 x = 2) :=
begin
  -- Proof goes here
  sorry
end

end minimum_value_f4_2_l474_474858


namespace greatest_integer_less_than_200_with_gcd_18_l474_474772

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l474_474772


namespace clock_minute_hand_sweep_radians_l474_474393

theorem clock_minute_hand_sweep_radians :
  let degrees_to_radians := λ (d : ℝ), d * (Real.pi / 180)
  let total_degrees := 45 * 6
  degrees_to_radians total_degrees = (3 / 2) * Real.pi :=
by
  -- Definitions of conditions based on the problem statement
  let degrees_per_minute := 6 : ℝ
  let minutes := 45
  let total_degrees := minutes * degrees_per_minute
  let degrees_to_radians := λ (d : ℝ), d * Real.pi / 180

  -- Assert the equivalence we want to prove
  show degrees_to_radians total_degrees = (3 / 2) * Real.pi
  sorry

end clock_minute_hand_sweep_radians_l474_474393


namespace max_k_value_l474_474229

noncomputable def max_k (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : ℝ :=
  let k := (-1 + Real.sqrt 7) / 2
  k

theorem max_k_value (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_k_value_l474_474229


namespace solveCubicEquation_l474_474880

-- Define the condition as a hypothesis
def equationCondition (x : ℝ) : Prop := (7 - x)^(1/3) = -5/3

-- State the theorem to be proved
theorem solveCubicEquation : ∃ x : ℝ, equationCondition x ∧ x = 314 / 27 :=
by 
  sorry

end solveCubicEquation_l474_474880


namespace combination_8_5_is_56_l474_474614

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474614


namespace evaluate_256_pow_5_div_4_l474_474892

theorem evaluate_256_pow_5_div_4 : 256 ^ (5 / 4) = 1024 :=
by
  -- Condition 1: Express 256 as a power of 2
  have h1 : 256 = 2 ^ 8 := by rfl

  -- Condition 2: Apply power of powers property
  have h2 : 256 ^ (5 / 4) = (2 ^ 8) ^ (5 / 4) := by rw [h1]
  have h3 : (2 ^ 8) ^ (5 / 4) = 2 ^ (8 * (5 / 4)) := by norm_num

  -- Calculating the exponential result
  have h4 : 2 ^ (8 * (5 / 4)) = 2 ^ 10 := by norm_num

  -- The final answer
  show 2 ^ 10 = 1024
  exact by norm_num

end evaluate_256_pow_5_div_4_l474_474892


namespace repeated_digit_percentage_l474_474870

noncomputable def percentage_of_repeated_digits : ℚ :=
  let total := 90000
  let no_repeated_digits := 9 * 9 * 8 * 7 * 6
  let repeated_digits := total - no_repeated_digits
  (repeated_digits * 100) / total

theorem repeated_digit_percentage : percentage_of_repeated_digits = 69.8 :=
by
  unfold percentage_of_repeated_digits
  have h_total : total = 90000 := rfl
  have h_no_repeated : no_repeated_digits = 27216 := by norm_num
  have h_repeated : repeated_digits = 90000 - 27216 := rfl
  have h_percentage : (62784 * 100) / 90000 = 69.8 := by norm_num
  exact h_percentage

end repeated_digit_percentage_l474_474870


namespace number_of_true_propositions_l474_474090

-- Definitions of the propositions
def prop_1 := ∀ (L₁ L₂ : ℝ → ℝ → Prop) (L₃ : ℝ → ℝ → Prop),
  (∀ x y, L₁ x y → L₃ x y) ∧ (∀ x y, L₂ x y → L₃ x y) → ∀ x y, L₁ x y = L₂ x y

def prop_2 := ∀ (P : ℝ × ℝ) (r : ℝ), ∀ (Q : ℝ × ℝ),
  (dist P Q = r) ↔ (circle P r).contains Q

def prop_3 := ∀ (Q : quadrilateral), (three_right_angles Q) → (rectangle Q)

def prop_4 := ∀ (P : ℝ × ℝ) (L : ℝ → ℝ → Prop), ∃! (M : ℝ → ℝ → Prop), perpendicular P L M

-- The main theorem statement
theorem number_of_true_propositions : (count_true_propositions prop_1 prop_2 prop_3 prop_4) = 0 :=
by
  sorry

end number_of_true_propositions_l474_474090


namespace find_printer_price_l474_474751

variable (C P M : ℝ)

theorem find_printer_price
  (h1 : C + P + M = 3000)
  (h2 : P = (1/4) * (C + P + M + 800)) :
  P = 950 :=
sorry

end find_printer_price_l474_474751


namespace log_base_condition_l474_474564

open Real

theorem log_base_condition (x : ℝ) (hx : log (2 * x) 216 = x) : 
  x = 3 ∧ (x ≠ 2 ∧ (x ≠ ⌊x⌋ ^ 2) ∧ (x ≠ ⌊x⌋ ^ 3)) :=
by
  sorry

end log_base_condition_l474_474564


namespace range_of_x_l474_474172

section
  variable (f : ℝ → ℝ)

  -- Conditions:
  -- 1. f is an even function
  def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

  -- 2. f is monotonically increasing on [0, +∞)
  def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

  -- Range of x
  def in_range (x : ℝ) : Prop := (1 : ℝ) / 3 < x ∧ x < (2 : ℝ) / 3

  -- Main statement
  theorem range_of_x (f_is_even : is_even f) (f_is_mono : mono_increasing_on_nonneg f) :
    ∀ x, f (2 * x - 1) < f ((1 : ℝ) / 3) ↔ in_range x := 
  by
    sorry
end

end range_of_x_l474_474172


namespace ananthu_can_complete_work_in_45_days_l474_474088

def amit_work_rate : ℚ := 1 / 15

def time_amit_worked : ℚ := 3

def total_work : ℚ := 1

def total_days : ℚ := 39

noncomputable def ananthu_days (x : ℚ) : Prop :=
  let amit_work_done := time_amit_worked * amit_work_rate
  let remaining_work := total_work - amit_work_done
  let ananthu_work_rate := remaining_work / (total_days - time_amit_worked)
  1 /x = ananthu_work_rate

theorem ananthu_can_complete_work_in_45_days :
  ananthu_days 45 :=
by
  sorry

end ananthu_can_complete_work_in_45_days_l474_474088


namespace rabbits_ate_three_potatoes_l474_474691

variable (initial_potatoes remaining_potatoes eaten_potatoes : ℕ)

-- Definitions from the conditions
def mary_initial_potatoes : initial_potatoes = 8 := sorry
def mary_remaining_potatoes : remaining_potatoes = 5 := sorry

-- The goal to prove
theorem rabbits_ate_three_potatoes :
  initial_potatoes - remaining_potatoes = 3 := sorry

end rabbits_ate_three_potatoes_l474_474691


namespace train_length_l474_474854

/--
  A train runs with a speed of 45 km/hr and takes 44 seconds to pass a platform 190 m long.
  Prove that the length of the train is 360 meters.
-/
theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (platform_length : ℕ) 
  (h1 : speed_kmh = 45) 
  (h2 : time_s = 44) 
  (h3 : platform_length = 190): 
  let speed_ms := speed_kmh * 5 / 18 in
  let total_distance := speed_ms * time_s in
  let train_length := total_distance - platform_length in
  train_length = 360 :=
by
  sorry

end train_length_l474_474854


namespace initial_weight_l474_474848

theorem initial_weight (W : ℝ) 
  (h1 : W > 0)
  (h2 : ((0.42 * W - 15) * 0.88 = 570)) : 
  W = 1578 :=
begin
  sorry
end

end initial_weight_l474_474848


namespace tan_theta_correct_l474_474220

noncomputable def tan_theta : Real :=
  let θ : Real := sorry
  if h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4) then
    if h : Real.sin θ + Real.cos θ = 17 / 13 then
      Real.tan θ
    else
      0
  else
    0

theorem tan_theta_correct {θ : Real} (h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 := sorry

end tan_theta_correct_l474_474220


namespace sphere_surface_area_l474_474850

-- Define the parameters and conditions
def surface_area_of_cube (a : ℝ) := 6 * a^2
def cube_edge_length (S : ℝ) := sqrt (S / 6)
def sphere_radius (a : ℝ) := (a * sqrt 3) / 2
def surface_area_of_sphere (R : ℝ) := 4 * Real.pi * R^2

-- The main theorem to prove
theorem sphere_surface_area (S : ℝ) (h : surface_area_of_cube (cube_edge_length S) = S) :
  surface_area_of_sphere (sphere_radius (cube_edge_length S)) = 27 * Real.pi :=
by
  -- conditions and variables setup
  -- sorry to indicate proof steps are omitted
  sorry

end sphere_surface_area_l474_474850


namespace parabola_circle_tangent_l474_474573

theorem parabola_circle_tangent {p : ℝ} (h₁: (∀ x y : ℝ, (x - 3)^2 + y^2 = 16 → False) -> ((x + p / 2) * (x + p / 2) + y * y = (4) * (4)) 
(h₂ : y^2 = 2 * p * x -> x = - p / 2) (p > 0):
   p = 2 :=
sorry

end parabola_circle_tangent_l474_474573


namespace smallest_positive_angle_l474_474913

theorem smallest_positive_angle (x : ℝ) (h : sin (3 * x) * sin (4 * x) = cos (3 * x) * cos (4 * x)) :
  x = (real.pi / 2) / 14 :=
sorry

end smallest_positive_angle_l474_474913


namespace parabola_latus_rectum_l474_474513

theorem parabola_latus_rectum (p : ℝ) (H : ∀ y : ℝ, y^2 = 2 * p * -2) : p = 4 :=
sorry

end parabola_latus_rectum_l474_474513


namespace joe_remaining_distance_l474_474659

-- Define the constants and conditions
def gas_used_per_distance (total_distance : ℕ) (used_gas_fraction : ℚ) : ℚ :=
  used_gas_fraction / total_distance

def total_distance_with_full_tank (per_distance_usage : ℚ) : ℚ :=
  1 / per_distance_usage

def remaining_distance (total_distance : ℚ) (traveled_distance : ℚ) : ℚ :=
  total_distance - traveled_distance

-- The proof statement
theorem joe_remaining_distance : 
  let traveled_distance : ℕ := 165
  let used_gas_fraction : ℚ := 3 / 8
  let per_distance_usage := gas_used_per_distance traveled_distance used_gas_fraction
  let full_tank_distance := total_distance_with_full_tank per_distance_usage
  in remaining_distance full_tank_distance traveled_distance = 275 := 
by {
  sorry -- Proof omitted
}

end joe_remaining_distance_l474_474659


namespace combination_choosing_four_socks_l474_474714

theorem combination_choosing_four_socks (n k : ℕ) (h_n : n = 7) (h_k : k = 4) :
  (nat.choose n k) = 35 :=
by
  rw [h_n, h_k, nat.choose_succ_succ, nat.choose_succ_succ, nat.choose_succ_succ, nat.choose_zero_succ]
  simp only [nat.choose_succ_succ, nat.factorial_succ, nat.factorial, nat.succ_sub_succ_eq_sub, nat.sub_zero,
    nat.pred_succ, nat.factorial_succ, nat.choose_self, show nat.factorial 0 = 1 by rfl, tsub_zero,
    mul_one, mul_over_nat_eq, nat.succ.sub_prime, nat.succ_sub_succ_eq_sub, nat.factorial_zero]
  norm_num
  sorry

end combination_choosing_four_socks_l474_474714


namespace problem1_problem2_l474_474535

noncomputable def U := set.univ  -- U = ℝ
def A := { x : ℝ | x > 2 }
def B := { x : ℝ | -1 < x ∧ x < 3 }

theorem problem1 : A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } :=
by sorry

theorem problem2 : (U \ B) ∪ A = { x : ℝ | x ≤ -1 ∨ x > 2 } :=
by sorry

end problem1_problem2_l474_474535


namespace riding_time_fraction_l474_474792

-- Definitions for conditions
def M : ℕ := 6
def total_days : ℕ := 6
def max_time_days : ℕ := 2
def part_time_days : ℕ := 2
def fixed_time : ℝ := 1.5
def total_riding_time : ℝ := 21

-- Prove the statement
theorem riding_time_fraction :
  ∃ F : ℝ, 2 * M + 2 * fixed_time + 2 * F * M = total_riding_time ∧ F = 0.5 :=
by
  exists 0.5
  sorry

end riding_time_fraction_l474_474792


namespace circle_radius_is_sqrt153_l474_474057

-- Definitions and conditions directly from the problem
def center_O : ℝ := 0
def point_P : ℝ := 9
def point_Q : ℝ := 15
def distance_OP : ℝ := abs (center_O - point_P)
def distance_OQ : ℝ := abs (center_O - point_Q)

-- Given the conditions
def conditions := 
  (on_line : ∃ l, center_O ∈ l ∧ point_P ∈ l ∧ point_Q ∈ l)
  ∧ (between_O_P_Q : center_O < point_Q ∧ center_O > point_P)
  ∧ (tangents : is_tangent_to_circle center_O point_P ∧ is_tangent_to_circle center_O point_Q)
  ∧ (at_right_angle : ∃ R, is_on_circle center_O R ∧ ∠PRQ = 90)

-- Prove that the radius of the circle is √153
theorem circle_radius_is_sqrt153 : 
  distance_OP = 9 → 
  distance_OQ = 15 → 
  conditions →
  ∃ r, r = √153 := 
by 
  intros hOP hOQ hcond
  sorry

end circle_radius_is_sqrt153_l474_474057


namespace model_height_rounded_nearest_whole_l474_474092

noncomputable def actual_height : ℝ := 289
noncomputable def scale_ratio : ℝ := 20

noncomputable def model_height (h : ℝ) (r : ℝ) : ℝ := h / r

theorem model_height_rounded_nearest_whole
  (h : ℝ := actual_height)
  (r : ℝ := scale_ratio) :
  Int.round (model_height h r) = 14 := sorry

end model_height_rounded_nearest_whole_l474_474092


namespace matchstick_winning_strategy_condition_l474_474841
-- Import necessary libraries

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci(n + 1) + fibonacci(n)

-- Define the condition of the matchstick game
def matchstick_game_winning_strategy (n : ℕ) : Prop :=
  n > 1 ∧ 
  (∀ k, n ≠ fibonacci k → ∃ r ≤ n, (r ≥ 1 ∧ r < n ∧ (∀ m > 0, m ≤ 2 * r → matchstick_game_winning_strategy (n - m))))

-- Statement of the theorem
theorem matchstick_winning_strategy_condition (n : ℕ) (h : n > 1) :
  (∃ k, n = fibonacci k) ↔ ¬ matchstick_game_winning_strategy n :=
sorry

end matchstick_winning_strategy_condition_l474_474841


namespace find_angle_between_vec_a_and_vec_b_l474_474490

open Real 

def vector (α : Type*) [InnerProductSpace ℝ α] := α

def vec_a : vector (ℝ × ℝ) := (3, 3)

def norm_vec_b : ℝ := 6

def is_perpendicular (vec_a vec_b : vector (ℝ × ℝ)) := (innerProduct vec_a (⟨vec_a.1 - vec_b.1, vec_a.2 - vec_b.2⟩ : ℝ × ℝ)) = 0

theorem find_angle_between_vec_a_and_vec_b (vec_a : vector (ℝ × ℝ))
  (norm_vec_b : ℝ)
  (h1: vec_a = (3, 3))
  (h2: norm_vec_b = 6)
  (h3: is_perpendicular vec_a (norm_vec_b * ⟨cos (π / 4), sin (π / 4)⟩)) : 
    ∃ α : ℝ, α = π / 4 := 
begin
  sorry
end

end find_angle_between_vec_a_and_vec_b_l474_474490


namespace proof_problem_l474_474930

theorem proof_problem (a b : ℝ) (h1 : 0 < a) (h2 : exp a + log b = 1) : 
  (a + log b < 0) ∧ (exp a + b > 2) ∧ (a + b > 1) := 
  sorry

end proof_problem_l474_474930


namespace triangle_equilateral_l474_474821

variable {P Q R A B C D E F : Type*}
variable (triangle : P → Q → R → Prop)
variable (circle : Type*)
variable (intersect_eq_parts : ∀ {X Y Z}, circle → triangle X Y Z → Prop)
-- Define the condition that the circle intersects each side of the triangle into three equal parts
variable (three_eq_parts : ∀ (X Y Z : P) (c : circle), 
  triangle X Y Z → 
  intersect_eq_parts c (triangle X Y Z) →
  ∃ (A B : P), 
    (X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) ∧
    (dist X A = dist A B ∧ dist A B = dist B Y) ∧ -- PA = AB = BQ
    ∃ (C D : P), (dist Y C = dist C D ∧ dist C D = dist D Z) ∧ -- QC = CD = DR
    ∃ (E F : P), (dist Z E = dist E F ∧ dist E F = dist F X)) -- RE = EF = FP

-- The goal is to prove that the triangle is equilateral given the circle dividing each side into three equal parts.
theorem triangle_equilateral : 
  (∀ (P Q R : P), triangle P Q R → 
   (∃ (c : circle), 
     intersect_eq_parts c (triangle P Q R) ∧
     three_eq_parts P Q R c (triangle P Q R) (intersect_eq_parts c (triangle P Q R)) →
   (dist P Q = dist Q R ∧ dist Q R = dist R P) :=
begin
  sorry
end

end triangle_equilateral_l474_474821


namespace cosine_angle_between_planes_eq_l474_474758

variables (A1 B1 C1 D1 A2 B2 C2 D2 : ℝ)

def cosine_angle_between_planes (A1 B1 C1 A2 B2 C2 : ℝ) :=
  (|A1 * A2 + B1 * B2 + C1 * C2|) / (Real.sqrt (A1 ^ 2 + B1 ^ 2 + C1 ^ 2) * Real.sqrt (A2 ^ 2 + B2 ^ 2 + C2 ^ 2))

theorem cosine_angle_between_planes_eq :
  ∀ α : ℝ, (A1 * x + B1 * y + C1 * z + D1 = 0) → (A2 * x + B2 * y + C2 * z + D2 = 0) →
  α = Real.acos (cosine_angle_between_planes A1 B1 C1 A2 B2 C2) →
  cos α = cosine_angle_between_planes A1 B1 C1 A2 B2 C2 :=
begin
  intros α plane1 plane2 angle_def,
  sorry
end

end cosine_angle_between_planes_eq_l474_474758


namespace sum_partial_fraction_l474_474914

theorem sum_partial_fraction :
  ∑ k in Finset.range 10, (2 : ℚ) / (k + 1) / (k + 2) = 20 / 11 :=
by
  sorry

end sum_partial_fraction_l474_474914


namespace manager_monthly_salary_l474_474035

theorem manager_monthly_salary (average_salary_20 : ℝ) (new_average_salary_21 : ℝ) (m : ℝ) 
  (h1 : average_salary_20 = 1300) 
  (h2 : new_average_salary_21 = 1400) 
  (h3 : 20 * average_salary_20 + m = 21 * new_average_salary_21) : 
  m = 3400 := 
by 
  -- Proof is omitted
  sorry

end manager_monthly_salary_l474_474035


namespace probability_both_selected_is_correct_l474_474039

def prob_selection_x : ℚ := 1 / 7
def prob_selection_y : ℚ := 2 / 9
def prob_both_selected : ℚ := prob_selection_x * prob_selection_y

theorem probability_both_selected_is_correct : prob_both_selected = 2 / 63 := 
by 
  sorry

end probability_both_selected_is_correct_l474_474039


namespace greek_cross_cut_and_assemble_l474_474113

theorem greek_cross_cut_and_assemble :
  ∀ (cross : Type) [symmetry cross] [greek_cross cross],
    ∃ (parts : list cross) (smaller_cross : cross) (remaining_parts : list cross),
      (smaller_cross ∈ parts) ∧ (smaller_cross ≡ cross) ∧ (remaining_parts ⊆ parts) ∧ 
      (assembled_square remaining_parts) := 
by
  sorry

end greek_cross_cut_and_assemble_l474_474113


namespace integer_pair_solution_l474_474868

theorem integer_pair_solution (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end integer_pair_solution_l474_474868


namespace least_multiple_7_fact_l474_474571

-- Define the conditions as Lean definitions
def is_positive_integer (n : ℕ) : Prop :=
  n > 0

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the statement of the proof
theorem least_multiple_7_fact : ∃ m : ℕ, is_positive_integer 7 ∧ (factorial 7 = m) ∧ m = 5040 :=
by
  use 5040
  split
  -- Proof that 7 is a positive integer
  exact dec_trivial
  split
  -- Proof that the factorial of 7 equals to m
  exact dec_trivial
  -- Proof that m equals 5040
  exact dec_trivial

end least_multiple_7_fact_l474_474571


namespace sequence_a1_zero_l474_474291

theorem sequence_a1_zero (c : ℝ) (a : ℕ → ℝ) (h_c : c > 2)
  (h_seq : ∀ n : ℕ, a n = a (n - 1)^2 - a (n - 1) ∧ a n < 1 / (sqrt (c * n))) : a 1 = 0 := 
sorry

end sequence_a1_zero_l474_474291


namespace smallest_gcd_for_system_l474_474484

theorem smallest_gcd_for_system :
  ∃ n : ℕ, n > 0 ∧ 
    (∀ a b c : ℤ,
     gcd (gcd a b) c = n →
     ∃ x y z : ℤ, 
       (x + 2*y + 3*z = a) ∧ 
       (2*x + y - 2*z = b) ∧ 
       (3*x + y + 5*z = c)) ∧ 
  n = 28 :=
sorry

end smallest_gcd_for_system_l474_474484


namespace length_AD_max_area_l474_474288

theorem length_AD_max_area {R1 R2 : ℝ} 
  (hR1_pos : 0 < R1) (hR2_pos : 0 < R2) 
  (A : Point) (D : Point)
  (O1 O2 : Point)
  (hO1_radius : dist O1 A = R1) (hO2_radius : dist O2 A = R2)
  (L : Line) (B C : Point)
  (hB_on_O1 : B ∈ L ∧ B ∈ Circle O1 R1)
  (hC_on_O2 : C ∈ L ∧ C ∈ Circle O2 R2)
  (hA_on_intersection : A ∈ Circle O1 R1 ∧ A ∈ Circle O2 R2)
  (hD_on_intersection : D ∈ Circle O1 R1 ∧ D ∈ Circle O2 R2) :
  let dist_AD := dist A D in
  dist_AD = (2 * R1 * R2) / (Real.sqrt (R1^2 + R2^2)) := by
  sorry

end length_AD_max_area_l474_474288


namespace comb_8_5_eq_56_l474_474593

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474593


namespace proof_problem_l474_474929

theorem proof_problem (a b : ℝ) (h1 : 0 < a) (h2 : exp a + log b = 1) : 
  (a + log b < 0) ∧ (exp a + b > 2) ∧ (a + b > 1) := 
  sorry

end proof_problem_l474_474929


namespace greatest_integer_gcd_l474_474776

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l474_474776


namespace maximum_swaps_for_victory_l474_474857

-- Define the size of the grid
def grid_size : ℕ := 2011
def total_numbers : ℕ := grid_size * grid_size

-- Define the condition for a grid to be strictly increasing across rows and down columns
def strictly_increasing (grid : ℕ × ℕ → ℕ) : Prop :=
  (∀ i j, i < grid_size → j < grid_size → (i < j → grid (i, j) < grid (i, j + 1)) 
  ) ∧ (∀ i j, i < grid_size → j < grid_size → (i < j → grid (i + 1, j) < grid (i, j)))

-- Define the swap operation ensuring the grid remains strictly increasing
def grid_swap (grid : ℕ × ℕ → ℕ) (i j k l : ℕ) (h : strictly_increasing grid) : ℕ × ℕ → ℕ := sorry

-- Define Alice's grids (set of uniquely filled grids)
def alice_grids : set (ℕ × ℕ → ℕ) := 
  {g | strictly_increasing g ∧ (∃ n : ℕ, n < 2010 ∧ g ≠ grid_size)}

-- Define Bob's initial grid
def bob_initial_grid : ℕ × ℕ → ℕ := sorry

-- Define the condition where Bob wins
def bob_wins (bob_grid : ℕ × ℕ → ℕ) (alice_grid : ℕ × ℕ → ℕ) : Prop :=
  ∃ i j, i < grid_size ∧ j < grid_size ∧ (bob_grid (i, j) = alice_grid (i, j))

-- Define the theorem for the maximum number of swaps
theorem maximum_swaps_for_victory : ∀ bob_grid : ℕ × ℕ → ℕ, bob_wins bob_initial_grid bob_grid → 
  (∃ swaps : ℕ, swaps ≤ 1 ∧ ∀ g ∈ alice_grids, ∃ i j, g (i, j) = bob_grid (i, j)) := sorry

end maximum_swaps_for_victory_l474_474857


namespace rectangle_area_find_rectangle_area_l474_474820

noncomputable def radius_of_circle : ℝ :=
let eqn := 2*x^2 + 2*y^2 - 20*x - 8*y + 36 = 0 in
let par_eqn := x^2 + y^2 - 10*x - 4*y + 18 = 0 in
sqrt 11

def width_of_rectangle (r : ℝ) : ℝ := 
2 * r

def height_of_rectangle (r : ℝ) : ℝ := 
2 * r

theorem rectangle_area : ℝ :=
let r := radius_of_circle in
let width := width_of_rectangle r in
let height := height_of_rectangle r in
width * height

theorem find_rectangle_area :
rectangle_area = 44 :=
sorry

end rectangle_area_find_rectangle_area_l474_474820


namespace distance_between_A_and_B_l474_474667

noncomputable def A := (1, 2) -- Vertex of the first parabola
noncomputable def B := (-2, 6) -- Vertex of the second parabola

theorem distance_between_A_and_B :
  (A = (1, 2)) ∧ (B = (-2, 6)) →
  dist A B = 5 := by
  sorry

end distance_between_A_and_B_l474_474667


namespace percentage_gain_correct_l474_474837

variable (cost_price_per_liter : ℝ) (milk_volume : ℝ) (water_percentage : ℝ) (selling_price : ℝ)

-- Condition definitions
def total_volume := milk_volume * (1 + water_percentage / 100)
def selling_price_per_liter := selling_price / total_volume
def percentage_gain := ((selling_price_per_liter - cost_price_per_liter) / cost_price_per_liter) * 100

-- Given specific values from conditions
def milk_cost := 12
def milk_qty := 1
def water_content := 20
def selling_amt := 15

-- The equivalent Lean 4 statement to prove
theorem percentage_gain_correct :
  percentage_gain milk_cost milk_qty water_content selling_amt = 4.17 := sorry

end percentage_gain_correct_l474_474837


namespace sum_coords_A_eq_neg9_l474_474668

variable (A B C : ℝ × ℝ)
variable (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
variable (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
variable (hB : B = (2, 5))
variable (hC : C = (4, 11))

theorem sum_coords_A_eq_neg9 
  (A B C : ℝ × ℝ)
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
  (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
  (hB : B = (2, 5))
  (hC : C = (4, 11)) : 
  A.1 + A.2 = -9 :=
  sorry

end sum_coords_A_eq_neg9_l474_474668


namespace family_vacation_total_days_l474_474421

theorem family_vacation_total_days
  (R_m R_a : ℕ)
  (Total_days : ℕ)
  (h1 : R_m + R_a = 13)
  (h2 : R_m = Total_days - 11)
  (h3 : R_a = Total_days - 12) :
  Total_days = 18 :=
begin
  sorry
end

end family_vacation_total_days_l474_474421


namespace total_length_of_scale_l474_474847

theorem total_length_of_scale (num_parts : ℕ) (length_per_part : ℕ) 
  (h1: num_parts = 4) (h2: length_per_part = 20) : 
  num_parts * length_per_part = 80 := by
  sorry

end total_length_of_scale_l474_474847


namespace find_distinct_numbers_l474_474681

def f (x : ℝ) : ℝ := 4 * x - x^2

theorem find_distinct_numbers :
∃ a b : ℝ, a ≠ b ∧ f(a) = b ∧ f(b) = a ∧ 
(a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2 ∨ 
 a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) :=
by
  sorry

end find_distinct_numbers_l474_474681


namespace perpendicular_equiv_l474_474936

variable {α : Type} [LinearOrderedField α]

variable (P : Type) [Plane α]

def line (l : P) := sorry -- Define line in the plane

def perpendicular (l : P) (α : Plane α) : Prop := sorry -- Define perpendicularity relation between a line and a plane

theorem perpendicular_equiv (l : P) (α : Plane α) :
  (∀ lines ∈ α, perpendicular l lines) ↔ (perpendicular l α) :=
sorry

end perpendicular_equiv_l474_474936


namespace find_a_for_minimum_value_of_f_l474_474964

noncomputable def f (x a : ℝ) := (Real.sqrt 3) * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + a

theorem find_a_for_minimum_value_of_f :
  (∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), f x a ≥ -1) ∧ (∃ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), f x a = -1)
  → a = -1 :=
by
  sorry

end find_a_for_minimum_value_of_f_l474_474964


namespace avg_of_first_5_subjects_l474_474863

variable (avg6 : ℕ) (marks6 : ℕ) (totalMarks : ℕ) (avg5 : ℕ)

-- Conditions
axiom (h1 : avg6 = 79)
axiom (h2 : marks6 = 104)
axiom (h3 : totalMarks = avg6 * 6)
axiom (h4 : totalMarks - marks6 = 370)
axiom (h5 : avg5 = 370 / 5)

-- Theorem
theorem avg_of_first_5_subjects : avg5 = 74 := by
  sorry

end avg_of_first_5_subjects_l474_474863


namespace range_F_l474_474119

-- Define the function and its critical points
def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem range_F : ∀ y : ℝ, y ∈ Set.range F ↔ -4 ≤ y := by
  sorry

end range_F_l474_474119


namespace length_chord_AB_l474_474753

def cos_30 := Real.cos (Real.pi / 6)
def sin_30 := Real.sin (Real.pi / 6)
def cos_60 := Real.cos (Real.pi / 3)
def sin_60 := Real.sin (Real.pi / 3)

def A : (ℝ × ℝ) := (cos_30, sin_30)
def B : (ℝ × ℝ) := (cos_60, sin_60)

noncomputable def distance_AB := Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem length_chord_AB : distance_AB = (Real.sqrt 6 - Real.sqrt 2) / 2 :=
by
  sorry

end length_chord_AB_l474_474753


namespace max_finish_points_l474_474424

theorem max_finish_points (k : ℕ) : k = 2020 → 
  ∃ N : ℕ, (N = Nat.choose 2020 1010) :=
by
  intro hk
  use Nat.choose 2020 1010
  sorry

end max_finish_points_l474_474424


namespace new_op_4_3_l474_474984

def new_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem new_op_4_3 : new_op 4 3 = 13 :=
by
  -- Placeholder for the proof
  sorry

end new_op_4_3_l474_474984


namespace function_properties_f_l474_474153

theorem function_properties_f (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : 1 < x1) (h3 : x1 < x2) :
  let f := λ x, (3 / 2) ^ x in
  (f (x1 + x2) = f x1 * f x2) ∧
  (f (x1 * x2) ≠ f x1 * f x2) ∧
  (f ((x1 + x2) / 2) ≤ (f x1 + f x2) / 2) ∧
  ((f x1 - f x2) / (x1 - x2) > 0) ∧
  (f x1 / (x1 - 1) > f x2 / (x2 - 1)) :=
by
  sorry

end function_properties_f_l474_474153


namespace simplify_and_rationalize_l474_474719

theorem simplify_and_rationalize :
  ( (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 9 / Real.sqrt 13) = 
    (3 * Real.sqrt 15015) / 1001 ) :=
by
  sorry

end simplify_and_rationalize_l474_474719


namespace harry_to_sue_nuts_ratio_l474_474449

-- Definitions based on conditions
def sue_nuts : ℕ := 48
def bill_nuts (harry_nuts : ℕ) : ℕ := 6 * harry_nuts
def total_nuts (harry_nuts : ℕ) : ℕ := bill_nuts harry_nuts + harry_nuts

-- Proving the ratio
theorem harry_to_sue_nuts_ratio (H : ℕ) (h1 : sue_nuts = 48) (h2 : bill_nuts H + H = 672) : H / sue_nuts = 2 :=
by
  sorry

end harry_to_sue_nuts_ratio_l474_474449


namespace dragon_defeated_l474_474253

universe u

inductive Dragon
| heads (count : ℕ) : Dragon

open Dragon

-- Conditions based on probabilities
def chop_probability : ℚ := 1 / 4
def one_grow_probability : ℚ := 1 / 3
def no_grow_probability : ℚ := 5 / 12

noncomputable def probability_defeated_dragon : ℚ :=
  if h : chop_probability + one_grow_probability + no_grow_probability = 1 then
    -- Define the recursive probability of having zero heads eventually (∞ case)
    let rec prob_defeat (d : Dragon) : ℚ :=
      match d with
      | heads 0     => 1  -- Base case: no heads mean dragon is defeated
      | heads (n+1) => 
        no_grow_probability * prob_defeat (heads n) + -- Successful strike
        one_grow_probability * prob_defeat (heads (n + 1)) + -- Neutral strike
        chop_probability * prob_defeat (heads (n + 2)) -- Unsuccessful strike
    prob_defeat (heads 3)  -- Initial condition with 3 heads
  else 0

-- Final theorem statement asserting the probability of defeating the dragon => 1
theorem dragon_defeated : probability_defeated_dragon = 1 :=
sorry

end dragon_defeated_l474_474253


namespace no_non_similar_triangles_with_geometric_angles_l474_474211

theorem no_non_similar_triangles_with_geometric_angles :
  ¬∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ a ≠ a * r ∧ a ≠ a * r * r ∧ a * r ≠ a * r * r ∧
  a + a * r + a * r * r = 180 :=
by
  sorry

end no_non_similar_triangles_with_geometric_angles_l474_474211


namespace area_of_T_l474_474297

-- Define the complex number ζ
def ζ (x : ℝ) (y : ℝ) : ℂ := complex.mk (x) (y)

-- Define the complex values for ζ and ζ^2
noncomputable def zeta : ℂ := ζ (1 / 2) (real.sqrt 3 / 2)
noncomputable def zeta_squared : ℂ := ζ (-1 / 2) (real.sqrt 3 / 2)

-- Define the set T
def T : set ℂ := {z | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ z = a + b * zeta + c * zeta_squared }

-- Define the function to calculate the area
noncomputable def area (S : set ℂ) : ℝ := sorry

-- Theorem to prove the area of T
theorem area_of_T : area T = 4 * real.sqrt 3 := sorry

end area_of_T_l474_474297


namespace comb_8_5_eq_56_l474_474598

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474598


namespace number_of_integer_solutions_l474_474358

theorem number_of_integer_solutions :
  { (x : ℤ) × (y : ℤ) × (z : ℤ) | |x * y * z| = 4 }.toFinset.card = 48 := 
sorry

end number_of_integer_solutions_l474_474358


namespace range_of_m_l474_474175

noncomputable def circle_condition (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 1

noncomputable def point_A (m : ℝ) : ℝ × ℝ :=
  (-m, 0)

noncomputable def point_B (m : ℝ) : ℝ × ℝ :=
  (m, 0)

noncomputable def point_on_circle (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  P = (x, y) ∧ circle_condition x y

theorem range_of_m (m : ℝ) :
  ∀ (x y : ℝ) (P : ℝ × ℝ),
  point_on_circle P x y →
  P = ((λ t, -t) x, (λ t, 0) y) →
  | ((3 - 0)^2 + (4 - 0)^2 - 1) | ≤ m ∧ m ≤ | ((3 - 0)^2 + (4 - 0)^2 + 1) | :=
sorry

end range_of_m_l474_474175


namespace arithmetic_sequence_sum_first_nine_terms_l474_474181

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := sorry

axiom b_seq : (n : ℕ) → Prop
  | 5 := geometric_sequence 5 = 2
  | _ := ∀ m, m < 5 ∨ m > 5 → geometric_sequence m ≠ 2

axiom a_seq : (n : ℕ) → Prop
  | 5 := arithmetic_sequence 5 = 2
  | _ := ∀ m, m < 5 ∨ m > 5 → arithmetic_sequence m ≠ 2

theorem arithmetic_sequence_sum_first_nine_terms :
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) (a_seq : ℕ → ℝ) 
  (h5 : a_seq 5 = 2) :
  a_seq 1 + a_seq 2 + a_seq 3 + a_seq 4 + a_seq 5 + a_seq 6 + a_seq 7 + a_seq 8 + a_seq 9 = 36 :=
by
  sorry

end arithmetic_sequence_sum_first_nine_terms_l474_474181


namespace question1_question2_l474_474194

noncomputable def f (x : ℝ) : ℝ := sin (x + π / 2) * cos (x - π / 3)

def smallest_positive_period : ℝ := π

theorem question1 : ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ x, T ≤ π → f (x + T) = f x) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

def max_of_g_in_interval : ℝ := 2

theorem question2 : ∃ x ∈ set.Icc (0 : ℝ) (π / 3), g x = max_of_g_in_interval :=
sorry

end question1_question2_l474_474194


namespace parallel_mn_ab_l474_474346

theorem parallel_mn_ab
  (O A B C D E M N : Type)
  [metric_space O]
  (circle_center : O)
  (chord_CD : linear_map O C D)
  (diameter_AB : linear_map O A B)
  (radius_OC : linear_map O circle_center C)
  (chord_AE : linear_map O A E)
  (chord_DE : linear_map O D E)
  (chord_BC : linear_map O B C)
  (OC_meet_AE_at_M : intersection_point M OC chord_AE)
  (DE_intersect_BC_at_N : intersection_point N chord_DE chord_BC)
  (perp_CD_AB : ∀ x : O, chord_CD x ∧ diameter_AB x → ⊥) :
  line_parallel (line_through M N) (line_through A B) := 
sorry

end parallel_mn_ab_l474_474346


namespace x_equals_l474_474562

variable (x y: ℝ)

theorem x_equals:
  (x / (x - 2) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 1)) → x = 2 * y^2 + 6 * y + 2 := by
  sorry

end x_equals_l474_474562


namespace local_minimum_value_l474_474735

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x

theorem local_minimum_value :
  ∃ x : ℝ, (f(x) = 0) ∧ (∀ y : ℝ, f'(x) < 0 ↔ y ≠ x) ∧ 
                (∀ z : ℝ, z < x ∨ z > x → f(z) > 0) :=
by
sorry

end local_minimum_value_l474_474735


namespace combination_8_5_is_56_l474_474610

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474610


namespace length_of_segment_AB_l474_474835

-- Define the parabola and its properties
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ C.1 = 3

-- Main statement of the problem
theorem length_of_segment_AB
  (A B : ℝ × ℝ)
  (hA : parabola_equation A.1 A.2)
  (hB : parabola_equation B.1 B.2)
  (C : ℝ × ℝ)
  (hfoc : focus (1, 0))
  (hm : midpoint_condition A B C) :
  dist A B = 8 :=
by sorry

end length_of_segment_AB_l474_474835


namespace calculate_expression_l474_474452

/-- Calculate the expression 2197 + 180 ÷ 60 × 3 - 197. -/
theorem calculate_expression : 2197 + (180 / 60) * 3 - 197 = 2009 := by
  sorry

end calculate_expression_l474_474452


namespace sum_two_digit_numbers_formed_l474_474141

theorem sum_two_digit_numbers_formed (a b : ℕ) (digits : set ℕ) :
  digits = {2, 5} → 
  a = 22 → b = 25 → 
  (∑ x in {22, 25, 52, 55}, x) = 154 := by
  sorry

end sum_two_digit_numbers_formed_l474_474141


namespace subset_relation_l474_474685

def P := {x : ℝ | x < 2}
def Q := {y : ℝ | y < 1}

theorem subset_relation : Q ⊆ P := 
by {
  sorry
}

end subset_relation_l474_474685


namespace sum_geometric_sequence_l474_474166

noncomputable def geometric_sequence_sum (n : ℕ) : ℕ → ℝ :=
  λ n, (3 - (1 / 3)^(n - 1)) / 2

theorem sum_geometric_sequence (n : ℕ) (a_1 : ℝ) (r : ℝ) (S_n : ℕ → ℝ) :
  a_1 = 1 → r = 1 / 3 → S_n = geometric_sequence_sum n → S_n n = (3 - (1 / 3)^(n - 1)) / 2 :=
begin
  intros h1 h2 h3,
  simp [geometric_sequence_sum, h1, h2, h3],
  sorry
end

end sum_geometric_sequence_l474_474166


namespace equation_of_line_through_point_l474_474795

theorem equation_of_line_through_point (
  p : ℝ × ℝ,
  circle_center : ℝ × ℝ,
  r : ℝ,
  chord_length : ℝ,
  h_p : p = (1, 0),
  h_circle : (circle_center = (1, 1)) ∧ (r = 1),
  h_chord_length : chord_length = real.sqrt 2
) : (∃ (k : ℝ), p.2 = k * (p.1 - 1) ∧ (chord_length = 2 * real.sqrt (r^2 - (1 / (k^2 + 1)))) ⟶ (k = 1 ∨ k = -1)) :=
sorry

end equation_of_line_through_point_l474_474795


namespace count_parallelograms_392_l474_474075

-- Define the conditions in Lean
def is_lattice_point (x y : ℕ) : Prop :=
  ∃ q : ℕ, x = q ∧ y = q

def on_line_y_eq_x (x y : ℕ) : Prop :=
  y = x ∧ is_lattice_point x y

def on_line_y_eq_mx (x y : ℕ) (m : ℕ) : Prop :=
  y = m * x ∧ is_lattice_point x y ∧ m > 1

def area_parallelogram (q s m : ℕ) : ℕ :=
  (m - 1) * q * s

-- Define the target theorem
theorem count_parallelograms_392 :
  (∀ (q s m : ℕ),
    on_line_y_eq_x q q →
    on_line_y_eq_mx s (m * s) m →
    area_parallelogram q s m = 250000) →
  (∃! n : ℕ, n = 392) :=
sorry

end count_parallelograms_392_l474_474075


namespace infinite_geometric_series_sum_l474_474101

theorem infinite_geometric_series_sum :
  let a := 1
  let r := (1 : ℝ) / 4 in
  abs r < 1 →
  (∑' n : ℕ, a * r^n) = 4 / 3 :=
by
  sorry

end infinite_geometric_series_sum_l474_474101


namespace parallel_vectors_implies_scalar_l474_474966

-- Defining the vectors a and b
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Stating the condition and required proof
theorem parallel_vectors_implies_scalar (m : ℝ) (h : (vector_a.snd / vector_a.fst) = (vector_b m).snd / (vector_b m).fst) : m = -4 :=
by sorry

end parallel_vectors_implies_scalar_l474_474966


namespace train_speed_first_part_l474_474085

theorem train_speed_first_part (x v : ℝ) (h1 : 0 < x) (h2 : 0 < v) 
  (h_avg_speed : (3 * x) / (x / v + 2 * x / 20) = 22.5) : v = 30 :=
sorry

end train_speed_first_part_l474_474085


namespace almond_croissant_price_l474_474544

theorem almond_croissant_price (R : ℝ) (T : ℝ) (W : ℕ) (total_spent : ℝ) (regular_price : ℝ) (weeks_in_year : ℕ) :
  R = 3.50 →
  T = 468 →
  W = 52 →
  (total_spent = 468) →
  (weekly_regular : ℝ) = 52 * 3.50 →
  (almond_total_cost : ℝ) = (total_spent - weekly_regular) →
  (A : ℝ) = (almond_total_cost / 52) →
  A = 5.50 := by
  intros hR hT hW htotal_spent hweekly_regular halmond_total_cost hA
  sorry

end almond_croissant_price_l474_474544


namespace num_squares_between_sqrt_18_and_sqrt_200_l474_474979

theorem num_squares_between_sqrt_18_and_sqrt_200 : 
  let x := Real.sqrt 18
  let y := Real.sqrt 200
  count_squares_between (⌈x⌉ : ℕ) (⌊y⌋ : ℕ) = 10 :=
by
  sorry

end num_squares_between_sqrt_18_and_sqrt_200_l474_474979


namespace combination_eight_choose_five_l474_474632

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474632


namespace concurrency_of_ceva_l474_474741

open EuclideanGeometry

theorem concurrency_of_ceva
  (A B C P Q R A1 B1 C1 : Point)
  (ω : Circle)
  (hA1 : A1 ∈ ω)
  (hB1 : B1 ∈ ω)
  (hC1 : C1 ∈ ω)
  (h1 : ∠(B, A1, P) = ∠(C, A1, Q))
  (h2 : ∠(C, B1, P) = ∠(A, B1, R))
  (h3 : ∠(A, C1, R) = ∠(B, C1, Q)) :
  Concurrency (Line.mk A A1) (Line.mk B B1) (Line.mk C C1) :=
by
  sorry

end concurrency_of_ceva_l474_474741


namespace equivalent_sum_of_exponents_l474_474810

theorem equivalent_sum_of_exponents : 3^3 + 3^3 + 3^3 = 3^4 :=
by
  sorry

end equivalent_sum_of_exponents_l474_474810


namespace moles_of_NH3_formed_l474_474908

-- Conditions
def moles_NH4Cl : ℕ := 3 -- 3 moles of Ammonium chloride
def total_moles_NH3_formed : ℕ := 3 -- The total moles of Ammonia formed

-- The balanced chemical reaction implies a 1:1 molar ratio
lemma reaction_ratio (n : ℕ) : total_moles_NH3_formed = n := by
  sorry

-- Prove that the number of moles of NH3 formed is equal to 3
theorem moles_of_NH3_formed : total_moles_NH3_formed = moles_NH4Cl := 
reaction_ratio moles_NH4Cl

end moles_of_NH3_formed_l474_474908


namespace probability_same_color_is_correct_l474_474982
open Nat

noncomputable def total_ways_to_choose_three (n : Nat) : Nat :=
  nat.choose n 3

noncomputable def favorable_cases (red blue green : Nat) : Nat :=
  nat.choose red 3 + nat.choose blue 3 + nat.choose green 3

noncomputable def probability_same_color (red blue green : Nat) : Rat :=
  let total := red + blue + green
  let favorable := favorable_cases red blue green
  favorable / total_ways_to_choose_three total

theorem probability_same_color_is_correct :
  probability_same_color 6 5 3 = 31 / 364 := 
by
  sorry

end probability_same_color_is_correct_l474_474982


namespace rectangular_prism_has_8_vertices_l474_474474

def rectangular_prism_vertices := 8

theorem rectangular_prism_has_8_vertices : rectangular_prism_vertices = 8 := by
  sorry

end rectangular_prism_has_8_vertices_l474_474474


namespace playground_fund_after_fees_l474_474015

-- Definitions based on the given conditions
def mrs_johnsons_class := 2300
def mrs_suttons_class := mrs_johnsons_class / 2
def miss_rollins_class := mrs_suttons_class * 8
def total_raised := miss_rollins_class * 3
def admin_fees := total_raised * 0.02
def playground_amount := total_raised - admin_fees

-- The theorem to be proved
theorem playground_fund_after_fees : playground_amount = 27048 := by
  sorry

end playground_fund_after_fees_l474_474015


namespace fraction_A_BC_l474_474055

-- Definitions for amounts A, B, C and the total T
variable (T : ℝ) (A : ℝ) (B : ℝ) (C : ℝ)

-- Given conditions
def conditions : Prop :=
  T = 300 ∧
  A = 120.00000000000001 ∧
  B = (6 / 9) * (A + C) ∧
  A + B + C = T

-- The fraction of the amount A gets compared to B and C together
def fraction (x : ℝ) : Prop :=
  A = x * (B + C)

-- The proof goal
theorem fraction_A_BC : conditions T A B C → fraction A B C (2 / 3) :=
by
  sorry

end fraction_A_BC_l474_474055


namespace cos_mul_tan_lt_zero_quadrant_l474_474509

theorem cos_mul_tan_lt_zero_quadrant (α : ℝ) (h1 : cos α * tan α < 0) (h2 : cos α ≠ 0) :
  ∃ n : ℤ, (n = 3 ∨ n = 4) ∧ (α ∈ set.Icc (n * π / 2) ((n + 1) * π / 2)) :=
  sorry

end cos_mul_tan_lt_zero_quadrant_l474_474509


namespace cent_piece_value_l474_474010

theorem cent_piece_value (Q P : ℕ) 
  (h1 : Q + P = 29)
  (h2 : 25 * Q + P = 545)
  (h3 : Q = 17) : 
  P = 120 := by
  sorry

end cent_piece_value_l474_474010


namespace ratio_of_green_to_blue_l474_474872

def balls (total blue red green yellow : ℕ) : Prop :=
  total = 36 ∧ blue = 6 ∧ red = 4 ∧ yellow = 2 * red ∧ green = total - (blue + red + yellow)

theorem ratio_of_green_to_blue (total blue red green yellow : ℕ) (h : balls total blue red green yellow) :
  (green / blue = 3) :=
by
  -- Unpack the conditions
  obtain ⟨total_eq, blue_eq, red_eq, yellow_eq, green_eq⟩ := h
  -- Simplify values based on the given conditions
  have blue_val := blue_eq
  have green_val := green_eq
  rw [blue_val, green_val]
  sorry

end ratio_of_green_to_blue_l474_474872


namespace length_of_KL_l474_474095

-- Definitions for the problem
variables {K L O A B C D : Point}
variables (r : ℝ) (h : ℝ) (area : ℝ) (BC AD : ℝ) (AB CD : ℝ)

-- Conditions
def is_isosceles_trapezoid (A B C D : Point) : Prop := 
  parallel BC AD ∧ (dist A B = dist C D)

def circle_inscribed_trapezoid (O : Point) (r : ℝ) : Prop :=
  ∀ P ∈ [A, B, C, D], dist O P = r

def circle_tangency_points (K : Point) (L : Point) (AB CD : Line) : Prop :=
  touching_circle_point K AB ∧ touching_circle_point L CD

axiom circle_radius : r = 3
axiom trapezoid_area : area = 48
axiom trapezoid_isosceles : is_isosceles_trapezoid A B C D
axiom circle_inscribed : circle_inscribed_trapezoid O r
axiom tangency_points : circle_tangency_points K L AB CD

-- Theorem statement (conclusion)
theorem length_of_KL : dist K L = 4.5 :=
sorry

end length_of_KL_l474_474095


namespace maurice_earnings_l474_474693

theorem maurice_earnings (bonus_per_10_tasks : ℕ → ℕ) (num_tasks : ℕ) (total_earnings : ℕ) :
  (∀ n, n * (bonus_per_10_tasks n) = 6 * n) →
  num_tasks = 30 →
  total_earnings = 78 →
  bonus_per_10_tasks num_tasks / 10 = 3 →
  (total_earnings - (bonus_per_10_tasks num_tasks / 10) * 6) / num_tasks = 2 :=
by
  intros h_bonus h_num_tasks h_total_earnings h_bonus_count
  sorry

end maurice_earnings_l474_474693


namespace exists_int_solutions_for_equations_l474_474887

theorem exists_int_solutions_for_equations : 
  ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 :=
by
  sorry

end exists_int_solutions_for_equations_l474_474887


namespace least_n_for_reducible_fraction_l474_474134

theorem least_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, n - 13 = 71 * k) ∧ n = 84 := by
  sorry

end least_n_for_reducible_fraction_l474_474134


namespace circle_equation_and_k_value_l474_474165

-- Condition Definitions
def A := (-2 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 2 : ℝ)
def center (a : ℝ) : ℝ × ℝ := (a, a)
def line_l (k : ℝ) : ℝ → ℝ := λ x, k * x + 1

-- Proof Problem
theorem circle_equation_and_k_value (a r k : ℝ) :
  (A = (x_A, y_A) ∧ B = (x_B, y_B) ∧ center(a) = (a, a) ∧ 
    (x_A - a)^2 + y_A^2 = r^2 ∧ (x_B - a)^2 + y_B^2 = r^2 ∧
    r = 2 ∧ a = 0 ∧
    x^2 + y^2 = 4 ∧
    ∃ P Q : ℝ × ℝ, (line_l k (P.fst) = P.snd) ∧ (line_l k (Q.fst) = Q.snd) ∧ 
    (P.fst^2 + P.snd^2 = 4) ∧ (Q.fst^2 + Q.snd^2 = 4) ∧ 
    ⟦vector (0, 0) 2 cos (P.fst, P.snd)⟧ = -2 ) → 
  (x^2 + y^2 = 4) ∧ (k = 0) :=
begin
  sorry
end

end circle_equation_and_k_value_l474_474165


namespace pete_numbers_count_l474_474705

theorem pete_numbers_count :
  ∃ x_values : Finset Nat, x_values.card = 4 ∧
  ∀ x ∈ x_values, ∃ y z : Nat, 
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x + y) * z = 14 ∧ (x * y) + z = 14 :=
by
  sorry

end pete_numbers_count_l474_474705


namespace intersect_on_circle_l474_474287

-- Define the geometric conditions and prove the statement
theorem intersect_on_circle (A B C C1 B1 A1 D E : Point) (ω ω1 : Circle) (ℓ : Line)
  (h1 : convex_hexagon A B C C1 B1 A1)
  (h2: A ≠ B) (h3: B ≠ C) (h4: A ≠ C)
  (h5 : A_1 ≠ B1) (h6: B1 ≠ C1) (h7 : A_1 ≠ C1)
  (h8: ω.is_in_circumcircle A B C)
  (h9 : ω1.is_in_circumcircle A1 B C1)
  (h10 : equal_lengths (segment A B) (segment B C))
  (h11 : ∀ (X : Point), is_on_line X ℓ ↔ midpoint X ℓ)
  (h12 : is_on_line D (line_through A C1 ∩ line_through A1 C))
  (h13 : second_intersection ω ω1 E)
  : is_intersect_on ω (line_through B B1) (line_through D E) := 
sorry -- proof goes here

end intersect_on_circle_l474_474287


namespace sum_of_first_fifteen_terms_l474_474145

open Nat

-- Define the conditions
def a3 : ℝ := -5
def a5 : ℝ := 2.4

-- The arithmetic progression terms formula
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- The sum of the first n terms formula
def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- The main theorem to prove
theorem sum_of_first_fifteen_terms :
  ∃ (a1 d : ℝ), (a a1 d 3 = a3) ∧ (a a1 d 5 = a5) ∧ (Sn a1 d 15 = 202.5) :=
sorry

end sum_of_first_fifteen_terms_l474_474145


namespace num_integers_in_interval_l474_474975

theorem num_integers_in_interval : 
  let lower_bound := Int.floor (- (9 : ℕ) * Real.pi / 2)
  let upper_bound := Int.floor ((12 : ℕ) * Real.pi)
  lower_bound = -14 ∧ upper_bound = 38 → 
  Set.card { x : ℤ | lower_bound ≤ x ∧ x ≤ upper_bound } = 53 :=
by 
  -- defining the approximation of pi
  let Real.pi_approx := 3.14
  
  -- defining the calculated bounds 
  let lower_bound := Int.floor (- (9 : ℕ) * Real.pi_approx / 2)
  let upper_bound := Int.floor ((12 : ℕ) * Real.pi_approx)
  
  sorry

end num_integers_in_interval_l474_474975


namespace triangle_equilateral_l474_474825

theorem triangle_equilateral (P Q R : Point) (circle : Circle) 
(h1 : divides_into_three_equal_parts circle P Q R) : 
equilateral_triangle P Q R :=
sorry

end triangle_equilateral_l474_474825


namespace probability_P_X_geq_5_l474_474169

-- Assumptions
variables {X : Type*} [MeasureTheory.ProbabilityMeasure X]

-- X follows normal distribution N(3, δ^2)
def is_normal_distribution (X : Type*) [MeasureTheory.ProbabilityMeasure X] : Prop :=
MeasureTheory.ProbabilityMeasure (fun x => PDFNormal x 3 δ^2)

-- Probability condition
axiom P_condition : MeasureTheory.ProbabilityMeasure.P (1 < X ∧ X ≤ 3) = 0.3

-- Proof statement
theorem probability_P_X_geq_5 (hX : is_normal_distribution X) : 
  MeasureTheory.ProbabilityMeasure.P (X ≥ 5) = 0.2 :=
sorry

end probability_P_X_geq_5_l474_474169


namespace collinearity_circumcenters_orthocenter_l474_474294

open EuclideanGeometry

variables {A B C D E F X Y Z : Point}
variables (circumcenter : Triangle → Point) (orthocenter : Triangle → Point)

-- Hexagon ABCDEF with parallel sides and equal products
axiom hexagon_conditions (h1 : segment A B ∥ segment D E)
                         (h2 : segment B C ∥ segment E F)
                         (h3 : segment C D ∥ segment F A)
                         (h4 : segment_length A B * segment_length D E = segment_length B C * segment_length E F)
                         (h5 : segment_length A B * segment_length D E = segment_length C D * segment_length F A)

-- Midpoints X, Y, Z of segments AD, BE, CF respectively
axiom midpoints (X : midpoint A D) (Y : midpoint B E) (Z : midpoint C F)

theorem collinearity_circumcenters_orthocenter :
  collinear [(circumcenter (triangle A C E)), (circumcenter (triangle B D F)), (orthocenter (triangle X Y Z))] :=
sorry

end collinearity_circumcenters_orthocenter_l474_474294


namespace Billys_score_on_AMC8_l474_474702

theorem Billys_score_on_AMC8 (correctly_answered : ℕ) 
  (incorrectly_answered : ℕ) (unanswered : ℕ) 
  (points_correct : ℕ) (points_incorrect : ℕ) 
  (points_unanswered : ℕ) :
  correctly_answered = 13 →
  incorrectly_answered = 7 →
  unanswered = 5 →
  points_correct = 1 →
  points_incorrect = 0 →
  points_unanswered = 0 →
  (correctly_answered * points_correct + 
   incorrectly_answered * points_incorrect + 
   unanswered * points_unanswered) = 13 :=
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end Billys_score_on_AMC8_l474_474702


namespace determine_judgments_l474_474197

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) ^ x - Real.log x / Real.log 2

theorem determine_judgments
  (a b c d : ℝ)
  (h₀ : 0 < a)
  (h₁ : a < b)
  (h₂ : b < c)
  (h₃ : f a * f b * f c < 0)
  (h₄ : f d = 0) :
  ({d < a, d > b, d < c} : set (Prop)) :=
begin
  sorry
end

end determine_judgments_l474_474197


namespace dragon_defeated_l474_474254

universe u

inductive Dragon
| heads (count : ℕ) : Dragon

open Dragon

-- Conditions based on probabilities
def chop_probability : ℚ := 1 / 4
def one_grow_probability : ℚ := 1 / 3
def no_grow_probability : ℚ := 5 / 12

noncomputable def probability_defeated_dragon : ℚ :=
  if h : chop_probability + one_grow_probability + no_grow_probability = 1 then
    -- Define the recursive probability of having zero heads eventually (∞ case)
    let rec prob_defeat (d : Dragon) : ℚ :=
      match d with
      | heads 0     => 1  -- Base case: no heads mean dragon is defeated
      | heads (n+1) => 
        no_grow_probability * prob_defeat (heads n) + -- Successful strike
        one_grow_probability * prob_defeat (heads (n + 1)) + -- Neutral strike
        chop_probability * prob_defeat (heads (n + 2)) -- Unsuccessful strike
    prob_defeat (heads 3)  -- Initial condition with 3 heads
  else 0

-- Final theorem statement asserting the probability of defeating the dragon => 1
theorem dragon_defeated : probability_defeated_dragon = 1 :=
sorry

end dragon_defeated_l474_474254


namespace ilya_defeats_dragon_l474_474239

section DragonAndIlya

def Probability (n : ℕ) : Type := ℝ

noncomputable def probability_of_defeat : Probability 3 :=
  let p_no_regrow := 5 / 12
  let p_one_regrow := 1 / 3
  let p_two_regrow := 1 / 4
  -- Assuming recursive relationship develops to eventually reduce heads to zero
  1

-- Prove that the probability_of_defeat is equal to 1
theorem ilya_defeats_dragon : probability_of_defeat = 1 :=
by
  -- Formal proof would be provided here
  sorry

end DragonAndIlya

end ilya_defeats_dragon_l474_474239


namespace defeat_dragon_probability_l474_474247

noncomputable theory

def p_two_heads_grow : ℝ := 1 / 4
def p_one_head_grows : ℝ := 1 / 3
def p_no_heads_grow : ℝ := 5 / 12

-- We state the probability that Ilya will eventually defeat the dragon
theorem defeat_dragon_probability : 
  ∀ (expected_value : ℝ), 
  (expected_value = p_two_heads_grow * 2 + p_one_head_grows * 1 + p_no_heads_grow * 0) →
  expected_value < 1 →
  prob_defeat (count_heads n : ℕ) > 0 :=
by
  sorry

end defeat_dragon_probability_l474_474247


namespace number_of_unique_products_l474_474030

-- Define the sets a and b
def setA : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}
def setB : Set ℕ := {2, 4, 6, 19, 21, 24, 27, 31, 35}

-- Define the number of unique products
def numUniqueProducts : ℕ := 405

-- Statement that needs to be proved
theorem number_of_unique_products :
  (∀ A1 ∈ setA, ∀ B ∈ setB, ∀ A2 ∈ setA, ∃ p, p = A1 * B * A2) ∧ 
  (∃ count, count = 45 * 9) ∧ 
  (∃ result, result = numUniqueProducts) :=
  by {
    sorry
  }

end number_of_unique_products_l474_474030


namespace find_kn_l474_474129

theorem find_kn (k n : ℕ) (h : k * n^2 - k * n - n^2 + n = 94) : k = 48 ∧ n = 2 := 
by 
  sorry

end find_kn_l474_474129


namespace orthogonal_diagonals_l474_474259

variables {a : ℝ} (A B C : ℝ × ℝ × ℝ) (A₁ B₁ C₁ : ℝ × ℝ × ℝ)

-- Assume we have a regular triangular prism with given vertices
def regular_triangular_prism (A₁ B₁ C₁ A B C : ℝ × ℝ × ℝ) : Prop :=
  -- Conditions for regular triangular prism can be detailed
  sorry

-- Diagonals on the side faces
def diagonal_AB₁ := B₁ - A
def diagonal_BC₁ := C₁ - B
def diagonal_CA₁ := A₁ - C

-- Orthogonality of specific diagonals
def orthogonal (v w : ℝ × ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0

theorem orthogonal_diagonals {A B C A₁ B₁ C₁ : ℝ × ℝ × ℝ} 
  (h_prism : regular_triangular_prism A₁ B₁ C₁ A B C)
  (h1 : orthogonal (diagonal_AB₁ A B₁) (diagonal_BC₁ B C₁)) :
  orthogonal (diagonal_A₁C A₁ C) (diagonal_AB₁ A B₁) :=
begin
  sorry
end

end orthogonal_diagonals_l474_474259


namespace domain_of_f_l474_474349

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1)^0 / real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | 2 - x > 0 ∧ x ≠ 1 / 2} = ((-∞:ℝ), (1 / 2)) ∪ ((1 / 2), 2) :=
by
  sorry

end domain_of_f_l474_474349


namespace johns_total_expenditure_l474_474660

-- Conditions
def treats_first_15_days : ℕ := 3 * 15
def treats_next_15_days : ℕ := 4 * 15
def total_treats : ℕ := treats_first_15_days + treats_next_15_days
def cost_per_treat : ℝ := 0.10
def discount_threshold : ℕ := 50
def discount_rate : ℝ := 0.10

-- Intermediate calculations
def total_cost_without_discount : ℝ := total_treats * cost_per_treat
def discounted_cost_per_treat : ℝ := cost_per_treat * (1 - discount_rate)
def total_cost_with_discount : ℝ := total_treats * discounted_cost_per_treat

-- Main theorem statement
theorem johns_total_expenditure : total_cost_with_discount = 9.45 :=
by
  -- Place proof here
  sorry

end johns_total_expenditure_l474_474660


namespace ratio_of_sums_l474_474973

noncomputable def first_sum : Nat := 
  let sequence := (List.range' 1 15)
  let differences := (List.range' 2 30).map (fun x => 2 * x)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum))

noncomputable def second_sum : Nat :=
  let sequence := (List.range' 1 15)
  let differences := (List.range' 1 29).filterMap (fun x => if x % 2 = 1 then some x else none)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum) - 135)

theorem ratio_of_sums : (first_sum / second_sum : Rat) = (160 / 151 : Rat) :=
  sorry

end ratio_of_sums_l474_474973


namespace cost_of_12_cheaper_fruits_l474_474648

-- Defining the price per 10 apples in cents.
def price_per_10_apples : ℕ := 200

-- Defining the price per 5 oranges in cents.
def price_per_5_oranges : ℕ := 150

-- No bulk discount means per item price is just total cost divided by the number of items
def price_per_apple := price_per_10_apples / 10
def price_per_orange := price_per_5_oranges / 5

-- Given the calculation steps, we have to prove that the cost for 12 cheaper fruits (apples) is 240
theorem cost_of_12_cheaper_fruits : 12 * price_per_apple = 240 := by
  -- This step performs the proof, which we skip with sorry
  sorry

end cost_of_12_cheaper_fruits_l474_474648


namespace exists_good_point_l474_474495

theorem exists_good_point (N : ℕ) (S : Fin N → ℤ) (k : ℕ) (hN : N = 2017) 
(hk : k ≤ 672) (hlabels : ∀ i, S i = 1 ∨ S i = -1) 
(hsum : ∀ i j (dir : Bool), 
  let start := i,
      end_ := if dir then ((start + j) % N) else ((start - j + N) % N) in
  ∑ m in list.range (N - 1), S (if m < list.range.index_of end_ then i + m else i + m - N) > 0) :
  ∃ i, ∀ j (dir : Bool), 
    let start := i,
        end_ := if dir then ((start + j) % N) else ((start - j + N) % N) in
    ∑ m in list.range (N - 1), S (if m < list.range.index_of end_ then i + m else i + m - N) > 0 := sorry

end exists_good_point_l474_474495


namespace general_term_smallest_term_index_l474_474954

theorem general_term (n : ℕ) : 
  (S n : ℕ → ℤ) := n ^ 2 - 10 * n →
  (S (n + 1) : ℕ → ℤ) → 
  (a n : ℕ → ℤ) := S (n + 1) - S n := 
  a n = 2 * n - 11 :=
sorry

theorem smallest_term_index : 
  (S : ℕ → ℤ) := (λ n, n ^ 2 - 10 * n) →
  (a n : ℕ → ℤ) := S (n + 1) - S n := 
  (a 1 = 2 * 1 - 11) ∧ (∀ n, a n ≥ 2 * 1 - 11) :=
sorry

end general_term_smallest_term_index_l474_474954


namespace mailman_distribution_l474_474836

theorem mailman_distribution 
    (total_mail_per_block : ℕ)
    (blocks : ℕ)
    (houses_per_block : ℕ)
    (h1 : total_mail_per_block = 32)
    (h2 : blocks = 55)
    (h3 : houses_per_block = 4) :
  total_mail_per_block / houses_per_block = 8 :=
by
  sorry

end mailman_distribution_l474_474836


namespace decagon_side_length_difference_eq_radius_l474_474230

noncomputable def ω := Complex.exp (2 * Real.pi * Complex.I / 10)

theorem decagon_side_length_difference_eq_radius (R : ℝ) :
  let A := R,
      B := R * ω,
      D := R * (ω ^ 3)
  in abs (abs (A - B) - abs (A - D)) = R :=
sorry

end decagon_side_length_difference_eq_radius_l474_474230


namespace min_value_sqrt_expression_l474_474989

theorem min_value_sqrt_expression (x : ℝ) :
  ∃ l, l = 2 * Real.sqrt 13 ∧ 
  l = infi (λ x: ℝ, Real.sqrt (x^2 + 4 * x + 5) + Real.sqrt (x^2 - 8 * x + 25)) :=
by
  sorry

end min_value_sqrt_expression_l474_474989


namespace quadratic_min_values_sum_l474_474111

noncomputable def P (x : ℝ) : ℝ := x^2 + 35 * x + (c : ℝ)
noncomputable def Q (x : ℝ) : ℝ := x^2 + 8 * x + (f : ℝ)

theorem quadratic_min_values_sum (P Q : ℝ → ℝ) (hP : ∀ x, P x = x^2 + 35 * x + c)
    (hQ : ∀ x, Q x = x^2 + 8 * x + f)
    (hPZeros : ∀ x, Q(P(x)) = x^4 + 70x + 84585)
    (hQZeros : ∀ x, P(Q(x)) = x^4 + 280x + 23125) :
    ((min x, P(x)) + (min x, Q(x))) = 76.75 := 
  sorry

end quadratic_min_values_sum_l474_474111


namespace permutations_satisfy_inequality_l474_474473

theorem permutations_satisfy_inequality :
  ∀ (σ : Equiv.Perm (Fin 4)),
  let b := fun n : Fin 4 => σ (n + 1) in
  ( (b 0)^2 + 1 ) / 2 * ( (b 1)^2 + 2 ) / 2 * ( (b 2)^2 + 3 ) / 2 * ( (b 3)^2 + 4 ) / 2 ≥ 24 :=
sorry

end permutations_satisfy_inequality_l474_474473


namespace good_tipper_bill_amount_l474_474160

theorem good_tipper_bill_amount {B : ℝ} 
    (h₁ : 0.05 * B + 1/20 ≥ 0.20 * B) 
    (h₂ : 0.15 * B = 3.90) : 
    B = 26.00 := 
by 
  sorry

end good_tipper_bill_amount_l474_474160


namespace value_of_b_l474_474657

theorem value_of_b 
  (b : ℝ)
  (h1 : ∀ x : ℝ, x^2 + b*x + 50 = (x + √42)^2 + 8)
  (h2 : b > 0) : 
  b = 2 * √42 := 
by 
  -- the proof would go here
  sorry

end value_of_b_l474_474657


namespace distinct_integers_sum_441_l474_474510

-- Define the variables and conditions
variables (a b c d : ℕ)

-- State the conditions: a, b, c, d are distinct positive integers and their product is 441
def distinct_positive_integers (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
def positive_integers (a b c d : ℕ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define the main statement to be proved
theorem distinct_integers_sum_441 (a b c d : ℕ) (h_distinct : distinct_positive_integers a b c d) 
(h_positive : positive_integers a b c d) 
(h_product : a * b * c * d = 441) : a + b + c + d = 32 :=
by
  sorry

end distinct_integers_sum_441_l474_474510


namespace pen_and_pencil_total_cost_l474_474429

theorem pen_and_pencil_total_cost :
  ∀ (pen pencil : ℕ), pen = 4 → pen = 2 * pencil → pen + pencil = 6 :=
by
  intros pen pencil
  intro h1
  intro h2
  sorry

end pen_and_pencil_total_cost_l474_474429


namespace divide_subtract_multiply_l474_474333

theorem divide_subtract_multiply :
  (-5) / ((1/4) - (1/3)) * 12 = 720 := by
  sorry

end divide_subtract_multiply_l474_474333


namespace children_on_bus_l474_474813

theorem children_on_bus (initial_children additional_children total_children : ℕ) (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = initial_children + additional_children → total_children = 64 :=
by
  -- Proof goes here
  sorry

end children_on_bus_l474_474813


namespace least_positive_integer_n_l474_474529

theorem least_positive_integer_n (n : ℕ) (hn : n = 10) :
  (2:ℝ)^(1 / 5 * (n * (n + 1) / 2)) > 1000 :=
by
  sorry

end least_positive_integer_n_l474_474529


namespace largest_n_for_perfect_square_l474_474569

theorem largest_n_for_perfect_square :
  ∃ n : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ n = k ^ 2 ∧ ∀ m : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ m = l ^ 2 → m ≤ n  → n = 972 :=
sorry

end largest_n_for_perfect_square_l474_474569


namespace area_of_triangle_ABC_eq_1_l474_474756

noncomputable def area_tri_ABC (A B C O : Type) (triangleABC : IsoscelesRightTriangle A B C) 
(inscribedCircle : InscribedCircle O π) : ℝ := 
1

theorem area_of_triangle_ABC_eq_1 :
∀ (A B C O : Type) 
(triangleABC : IsoscelesRightTriangle A B C) 
(inscribedCircle : InscribedCircle O (π : ℝ)),
  area_tri_ABC A B C O triangleABC inscribedCircle = 1 :=
by
sorr

end area_of_triangle_ABC_eq_1_l474_474756


namespace problem1_problem2_problem3_l474_474499

-- Prove that $\frac{a_1}{d} = \frac{4}{3}$ given the problem's conditions
theorem problem1 (d : ℝ) (a₁ a₈ : ℝ) (h₁ : d ≠ 0) (h₂ : (a₁ + 2 * d) ^ 2 = a₁ * (a₁ + 7 * d)) :
  a₁ / d = 4 / 3 := 
sorry

-- Prove that $\frac{a_1}{d} = 1$ given the problem's conditions
theorem problem2 (d : ℝ) (a₁ : ℝ) (q : ℝ) (h₁ : d ≠ 0) (h₂ : ∀ n : ℕ, aₙ = a₁ + d * (n - 1) ∧ a_{k n} = a_{k₁} * q^n) :
  a₁ / d = 1 :=
sorry

-- Prove that $a_1 \in [2, +\infty)$ given the problem's conditions
theorem problem3 (d : ℝ) (a₁ : ℝ) (q : ℝ) (h₁ : d ≠ 0) (h₂ : ∀ n : ℕ, kₙ = k₁ * q^(n-1)) (h₃ : ∀ n : ℕ, aₙ + a_{kₙ} > 2 * kₙ) :
  2 ≤ a₁ :=
sorry

end problem1_problem2_problem3_l474_474499


namespace no_set_of_eleven_no_six_divisible_by_six_l474_474277

theorem no_set_of_eleven_no_six_divisible_by_six (A : Fin 11 → ℕ) : 
  ∃ (B : Finset (Fin 11)), B.card = 6 ∧ (∑ i in B, A i) % 6 = 0 := by
  sorry

end no_set_of_eleven_no_six_divisible_by_six_l474_474277


namespace miki_sandcastle_height_l474_474866

theorem miki_sandcastle_height (h_s : ℝ) (d : ℝ) (h_s_value : h_s = 0.5) (d_value : d = 0.33) : 
  h_s + d = 0.83 :=
by
  rw [h_s_value, d_value]
  exact add_0_5_0_33

lemma add_0_5_0_33 : 0.5 + 0.33 = 0.83 := 
by
  norm_num

end miki_sandcastle_height_l474_474866


namespace solution_to_system_is_correct_l474_474747

noncomputable def system_solution : ℤ × ℤ :=
let (x, y) := (3, -2) in
(x, y)

theorem solution_to_system_is_correct :
  ∃ (x y : ℤ), (x + y = 1) ∧ (4 * x + y = 10) ∧ (x, y) = (3, -2) :=
by {
  use (3, -2),
  simp,
  split,
  {
    exact rfl,
  },
  split,
  {
    exact rfl,
  },
  {
    exact rfl,
  },
}

end solution_to_system_is_correct_l474_474747


namespace min_value_frac_sum_l474_474299

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 2) : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 3 * b = 2 ∧ (2 / a + 4 / b) = 14) :=
by
  sorry

end min_value_frac_sum_l474_474299


namespace gcd_sum_pairs_l474_474386

theorem gcd_sum_pairs (a b : ℕ) (h₁ : a + b = 915) (h₂ : gcd a b = 61) :
  {p : ℕ × ℕ | ∃ x y : ℕ, p = (61 * x, 61 * y) ∧ x + y = 15 ∧ gcd x y = 1}.card = 8 :=
sorry

end gcd_sum_pairs_l474_474386


namespace find_min_value_l474_474136

noncomputable def expression (x : ℝ) : ℝ :=
  (Real.sin x ^ 8 + Real.cos x ^ 8 + 2) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 2)

theorem find_min_value : ∃ x : ℝ, expression x = 5 / 4 :=
sorry

end find_min_value_l474_474136


namespace horner_no_85169_l474_474388

def poly (x : ℕ) : ℕ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_evaluation (x : ℕ) : ℕ :=
  let step1 := 7 * x + 3
  let step2 := step1 * x - 5
  step2 * x + 11

theorem horner_no_85169 (x : ℕ) : x = 23 → horner_evaluation x ≠ 85169 := by
  intro hx
  simp [hx, horner_evaluation, poly]
  sorry

end horner_no_85169_l474_474388


namespace choose_five_from_eight_l474_474605

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474605


namespace jason_grew_20_canteloupes_l474_474661

theorem jason_grew_20_canteloupes (k f t j : ℕ) 
    (h1 : k = 29) 
    (h2 : f = 16) 
    (h3 : t = 65) 
    (h4 : k + f + j = t): 
    j = 20 := 
by 
  rw [h1, h2] at h4 
  simp at h4 
  exact (nat.add_sub_cancel' h4).symm

end jason_grew_20_canteloupes_l474_474661


namespace f_90_eq_999_l474_474115

noncomputable def f : ℕ → ℕ
| n := if n >= 1000 then n - 3 else f(f(n + 7))

theorem f_90_eq_999 : f(90) = 999 := 
by
  sorry

end f_90_eq_999_l474_474115


namespace final_temperature_is_58_32_l474_474654

-- Initial temperature
def T₀ : ℝ := 40

-- Sequence of temperature adjustments
def T₁ : ℝ := 2 * T₀
def T₂ : ℝ := T₁ - 30
def T₃ : ℝ := T₂ * (1 - 0.30)
def T₄ : ℝ := T₃ + 24
def T₅ : ℝ := T₄ * (1 - 0.10)
def T₆ : ℝ := T₅ + 8
def T₇ : ℝ := T₆ * (1 + 0.20)
def T₈ : ℝ := T₇ - 15

-- Proof statement
theorem final_temperature_is_58_32 : T₈ = 58.32 :=
by sorry

end final_temperature_is_58_32_l474_474654


namespace intersection_area_le_two_thirds_l474_474332

variable {A B C : Point}
variable {point : Point}

-- Definition of area of a triangle
def area_triangle (A B C : Point) : ℝ := sorry

-- Definition of the symmetric triangle with respect to a point
def symmetric_triangle (A B C : Point) (point : Point) : Triangle := sorry

-- Definition of the polygon obtained by intersecting original triangle and its symmetric triangle
def intersection_polygon (A B C : Point) (symmetric : Triangle) : Polygon := sorry

-- Definition of the area of that intersected polygon
def area_polygon (polygon : Polygon) : ℝ := sorry

-- Main theorem
theorem intersection_area_le_two_thirds (A B C point : Point) :
  let intersected_polygon := intersection_polygon A B C (symmetric_triangle A B C point) in
  area_polygon intersected_polygon ≤ (2 / 3) * area_triangle A B C :=
  sorry

end intersection_area_le_two_thirds_l474_474332


namespace slope_probability_l474_474163

def is_slope_negative (A B : ℤ) : Prop := A ≠ B ∧ -A * B < 0

def total_cases : ℕ := 4 * 3 -- C(4, 2) with A ≠ B

def negative_slope_cases : ℕ :=
  -- Count pairs (A, B) such that A ≠ B and -A/B < 0
  -- This can be checked manually or through code by enumerating the pairs
  let pairs := [(−3, −1), (−3, 1), (−3, 2), 
                (−1, −3), (−1, 1), (−1, 2), 
                (1, −3), (1, −2), (2, −3), 
                (2, −1), (2, 1)] in
  pairs.filter (λ p, is_slope_negative p.1 p.2).length

theorem slope_probability :
  let probability := (negative_slope_cases : ℚ) / (total_cases : ℚ)
  probability = 4 / 11 :=
by
  sorry

end slope_probability_l474_474163


namespace repeating_number_divisible_l474_474437

theorem repeating_number_divisible (abcdef : ℕ) (h_abcdef : abcdef < 1000000 ∧ abcdef >= 100000) : 
  let n := 1000000 * abcdef + abcdef in
  (1000001 ∣ n) := 
by 
  sorry

end repeating_number_divisible_l474_474437


namespace hyperbola_triangle_area_l474_474179

noncomputable def hyperbola_area (P : Point) (F1 F2 : Point) (hP : on_hyperbola P) (hF : is_focus F1 F2)
  (hangle : angle F1 P F2 = 60) : ℝ :=
  area_of_triangle P F1 F2

theorem hyperbola_triangle_area (P : Point) (F1 F2 : Point) (hP : on_hyperbola P) (hF : is_focus F1 F2)
  (hangle : angle F1 P F2 = 60) : hyperbola_area P F1 F2 hP hF hangle = 3 * sqrt 3 :=
sorry

end hyperbola_triangle_area_l474_474179


namespace probability_two_different_color_chips_l474_474381

theorem probability_two_different_color_chips : 
  let blue_chips := 4
  let yellow_chips := 5
  let green_chips := 3
  let total_chips := blue_chips + yellow_chips + green_chips
  let prob_diff_color := 
    ((blue_chips / total_chips) * ((yellow_chips + green_chips) / (total_chips - 1))) + 
    ((yellow_chips / total_chips) * ((blue_chips + green_chips) / (total_chips - 1))) + 
    ((green_chips / total_chips) * ((blue_chips + yellow_chips) / (total_chips - 1)))
  in
  prob_diff_color = 47 / 66 :=
by
  sorry

end probability_two_different_color_chips_l474_474381


namespace intersection_with_horizontal_asymptote_l474_474157

open Real

noncomputable def g : ℝ → ℝ := λ x, (3 * x^2 - 8 * x - 9) / (x^2 - 5 * x + 6)

theorem intersection_with_horizontal_asymptote :
  ∃ x : ℝ, g(x) = 3 ∧ x = 27 / 7 :=
by
  sorry

end intersection_with_horizontal_asymptote_l474_474157


namespace complex_z24_condition_l474_474184

open Complex

theorem complex_z24_condition (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (5 * π / 180)) : 
  z^24 + z⁻¹^24 = -1 := sorry

end complex_z24_condition_l474_474184


namespace determine_x_l474_474121

variables {m n x : ℝ}
variable (k : ℝ)
variable (Hmn : m ≠ 0 ∧ n ≠ 0)
variable (Hk : k = 5 * (m^2 - n^2))

theorem determine_x (H : (x + 2 * m)^2 - (x - 3 * n)^2 = k) : 
  x = (5 * m^2 - 9 * n^2) / (4 * m + 6 * n) := by
  sorry

end determine_x_l474_474121


namespace grade_assignment_count_l474_474842

noncomputable def grading_ways : ℕ :=
  nat.choose 12 2 * 3^10

theorem grade_assignment_count : grading_ways = 3906234 :=
by
  unfold grading_ways
  have h1 : nat.choose 12 2 = 66 := by sorry
  have h2 : 3^10 = 59049 := by sorry
  rw [h1, h2]
  norm_num

end grade_assignment_count_l474_474842


namespace max_non_attacking_rooks_l474_474701

theorem max_non_attacking_rooks : ∃ n : ℕ, n ≤ 16 ∧ (∀ k : ℕ, k > 16 → (¬ (∃ p : fin 8 → fin 8, 
  (∀ i, (∃ j, p i = ⟨j, sorry⟩) → i < 8 ∧ ∀ x y, x ≠ y → p x ≠ p y) ∧ 
  (∀ i, (∃ j, p i = ⟨j, sorry⟩) → i < 8 ∧ ∀ x y, x ≠ y → p x ≠ p y)))) ∧ 
  (∃ p : fin 8 → fin 8, 
  (∀ i, (∃ j, p i = ⟨j, sorry⟩) → i < 8 ∧ ∀ x y, x ≠ y → p x ≠ p y) ∧ 
  (∀ i, (∃ j, p i = ⟨j, sorry⟩) → i < 8 ∧ ∀ x y, x ≠ y → p x ≠ p y))) :=
begin
  use 16,
  split,
  { exact le_refl 16 },
  split,
  { intros k hk,
    exfalso,
    sorry },
  { let w_pos := λ i : fin 8, i,
    let b_pos := λ i : fin 8, ⟨8 - i.1, sorry⟩,
    use w_pos,
    use b_pos,
    split,
    { intros i hi hdiff,
      sorry },
    { intros i hi hdiff,
      sorry } }
end

end max_non_attacking_rooks_l474_474701


namespace rate_of_interest_l474_474394

theorem rate_of_interest (P SI : ℝ) (T : ℝ) (P_val : P = 400) (SI_val : SI = 180) (T_val : T = 2) :
  ∃ R : ℝ, R = 22.5 :=
by {
  have formula := SI = (P * R * T) / 100,
  have R_val := (SI * 100) / (P * T),
  rw [P_val, SI_val, T_val] at *,
  use R_val,
  sorry
}

end rate_of_interest_l474_474394


namespace marble_problem_l474_474862

theorem marble_problem (a : ℚ) :
  (a + 2 * a + 3 * 2 * a + 5 * (3 * 2 * a) + 2 * (5 * (3 * 2 * a)) = 212) ↔
  (a = 212 / 99) :=
by
  sorry

end marble_problem_l474_474862


namespace ellipse_cosine_ratio_const_l474_474232

theorem ellipse_cosine_ratio_const (F F' M T : ℝ) (h1 : is_focus_of_ellipse F) (h2 : is_arbitrary_point_on_ellipse M) (h3 : is_point_of_tangent_intersection_with_major_axis T M) :
  ∃ k, ∀ F M T,
    cos (angle F M T) / cos (angle M T F) = k := sorry

end ellipse_cosine_ratio_const_l474_474232


namespace P_zero_value_l474_474109

def P : ℝ → ℝ

axiom P_condition : ∀ x y : ℝ, (|y^2 - P(x)| ≤ 2 * |x| ↔ |x^2 - P(y)| ≤ 2 * |y|)

theorem P_zero_value : P 0 ∈ set.Iio 0 ∪ {1} := 
sorry

end P_zero_value_l474_474109


namespace find_angle_B_l474_474971

variable {A B C : ℝ}
variable (m n : ℝ × ℝ)
variable (θ : ℝ)

def vectors_angle (m n : ℝ × ℝ) : ℝ :=
  Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2)))

theorem find_angle_B
  (m_eq : m = (Real.sin B, 1 - Real.cos B))
  (n_eq : n = (2, 0))
  (angle_condition : θ = π / 3)
  (angle_eq : vectors_angle m n = θ) :
  B = 2 * π / 3 :=
by
  -- Proof goes here
  sorry

end find_angle_B_l474_474971


namespace count_sequences_from_neg7_to_7_l474_474558

def count_nondecreasing_abs_value_sequences (A : Finset ℤ) : ℕ :=
  if A = {x : ℤ | -7 ≤ x ∧ x ≤ 7} 
  then 2 ^ 7
  else 0

theorem count_sequences_from_neg7_to_7 :
  count_nondecreasing_abs_value_sequences {x : ℤ | -7 ≤ x ∧ x ≤ 7} = 128 := by
    sorry

end count_sequences_from_neg7_to_7_l474_474558


namespace greatest_int_with_conditions_l474_474782

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l474_474782


namespace gcd_plus_lcm_eight_twelve_l474_474397

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_plus_lcm_eight_twelve :
  gcd 8 12 + lcm 8 12 = 28 :=
by {
  /-
  1. Calculate the GCD of 8 and 12.
     Nat.gcd 8 12 = 4.
  2. Calculate the LCM of 8 and 12.
     Nat.lcm 8 12 = 24.
  3. Sum the GCD and LCM.
     4 + 24 = 28.
  -/
  have h1 : gcd 8 12 = 4 := Nat.gcd_eq_right 12,  -- Predefined computation
  have h2 : lcm 8 12 = 24 := Nat.lcm_eq_mul_div_gcd  -- Predefined computation
  rw [h1, h2], -- Replace GCD and LCM with their values
  simp,       -- Simplify the sum to get 28
  sorry      -- Placeholder for complete proof
}

end gcd_plus_lcm_eight_twelve_l474_474397


namespace carmen_rope_gcd_l474_474104

/-- Carmen has three ropes with lengths 48, 64, and 80 inches respectively.
    She needs to cut these ropes into pieces of equal length for a craft project,
    ensuring no rope is left unused.
    Prove that the greatest length in inches that each piece can have is 16. -/
theorem carmen_rope_gcd :
  Nat.gcd (Nat.gcd 48 64) 80 = 16 := by
  sorry

end carmen_rope_gcd_l474_474104


namespace log_base_4_of_64_sqrt_2_l474_474123

theorem log_base_4_of_64_sqrt_2 : Real.logBase 4 (64 * Real.sqrt 2) = 13 / 4 := by
  sorry

end log_base_4_of_64_sqrt_2_l474_474123


namespace percentage_relationship_l474_474164

theorem percentage_relationship (x : ℝ) (h1 : 5.76 = x * 0.4) (h2 : x = 120) : 5.76 = 0.12 * (0.4 * x) :=
by
  sorry

end percentage_relationship_l474_474164


namespace no_non_similar_triangles_with_geometric_angles_l474_474212

theorem no_non_similar_triangles_with_geometric_angles :
  ¬ ∃ (a r : ℤ), 0 < a ∧ 0 < r ∧ a ≠ ar ∧ a ≠ ar^2 ∧ ar ≠ ar^2 ∧
  a + ar + ar^2 = 180 :=
sorry

end no_non_similar_triangles_with_geometric_angles_l474_474212


namespace percentage_spent_on_meat_l474_474652

def total_cost_broccoli := 3 * 4
def total_cost_oranges := 3 * 0.75
def total_cost_vegetables := total_cost_broccoli + total_cost_oranges + 3.75
def total_cost_chicken := 3 * 2
def total_cost_meat := total_cost_chicken + 3
def total_cost_groceries := total_cost_meat + total_cost_vegetables

theorem percentage_spent_on_meat : 
  (total_cost_meat / total_cost_groceries) * 100 = 33 := 
by
  sorry

end percentage_spent_on_meat_l474_474652


namespace num_terms_expansion_l474_474998

theorem num_terms_expansion (x y z : ℝ) : 
  (∃ m : ℝ, ∃ a b c : ℕ, m * x^a * y^b * z^c ∧ (a + b + c = 10)) → 
  ((x + y + z) ^ 10).terms.count = 66 :=
sorry

end num_terms_expansion_l474_474998


namespace sum_of_first_fifteen_terms_l474_474143

noncomputable def a₃ : ℝ := -5
noncomputable def a₅ : ℝ := 2.4
noncomputable def a₁ : ℝ := -12.4
noncomputable def d : ℝ := 3.7

noncomputable def S₁₅ : ℝ := 15 / 2 * (2 * a₁ + 14 * d)

theorem sum_of_first_fifteen_terms :
  S₁₅ = 202.5 := 
by
  sorry

end sum_of_first_fifteen_terms_l474_474143


namespace mary_spend_fraction_l474_474805

-- Define the conditions
variables {S : ℝ} -- S is Mary's monthly salary, which is a real number.
variable {x : ℝ} -- x is the fraction of her salary that she saves each month.

-- Define the main theorem
theorem mary_spend_fraction (h1 : 12 * x * S = 7 * (1 - x) * S) : (1 - x) = 12 / 19 := by 
  -- Simplify the equation derived from the conditions
  have h2 : 12 * x = 7 * (1 - x) := Eq.trans (by ring) h1
  have h3 : 12 * x = 7 - 7 * x := by linarith
  have h4 : 12 * x + 7 * x = 7 := by linarith
  have h5 : 19 * x = 7 := by linarith
  have hx : x = 7 / 19 := by linarith
  -- Show that the required fraction of her salary that she spends is correct
  show 1 - x = 12 / 19, by rw hx; norm_num

end mary_spend_fraction_l474_474805


namespace combination_8_5_is_56_l474_474609

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474609


namespace num_zeros_at_end_of_product_l474_474218

theorem num_zeros_at_end_of_product (a b : ℕ) (ha : a = 3^2 * 5) (hb : b = 2^3 * 5^3) : 
  (num_zeros (a * b) = 3) :=
by sorry

end num_zeros_at_end_of_product_l474_474218


namespace angle_between_clock_hands_at_20_minutes_past_twelve_l474_474389

noncomputable def angle_formed_by_clock_hands (minute_angle_per_minute hour_angle_per_hour minutes : ℕ) : ℕ :=
  let minute_hand_angle := minute_angle_per_minute * minutes
  let hour_hand_angle := hour_angle_per_hour * (minutes / 60.0)
  int.natAbs (minute_hand_angle - hour_hand_angle)

theorem angle_between_clock_hands_at_20_minutes_past_twelve :
  angle_formed_by_clock_hands 6 30 20 = 110 := 
  sorry

end angle_between_clock_hands_at_20_minutes_past_twelve_l474_474389


namespace integral_inequality_l474_474292

noncomputable def continuous_periodic_function (α T : ℝ) (f : ℝ → ℝ) [hf : ∀ x, f x > 0]
  [hf_continuous : continuous f]
  [hf_periodic : ∀ x, f (x + T) = f x]
  : Prop :=
  ∫ x in 0..T, f x / f (x + α) ≥ T

theorem integral_inequality (f : ℝ → ℝ) (T α : ℝ)
  (hf_pos : ∀ x, f x > 0)
  (hf_continuous : continuous f)
  (hf_periodic : ∀ x, f (x + T) = f x)
  : ∫ x in 0..T, (f x) / (f (x + α)) ≥ T := 
by
  sorry

end integral_inequality_l474_474292


namespace ordering_inequality_l474_474666

noncomputable def a : ℝ := Real.exp (0.3 * Real.log π)
noncomputable def b : ℝ := Real.log 3 / Real.log π
noncomputable def c : ℝ := Real.log (Real.sin (2 * π / 3)) / Real.log 3

-- Statement to be proved
theorem ordering_inequality : a > b ∧ b > c :=
by {
  sorry
}

end ordering_inequality_l474_474666


namespace right_triangle_x_value_l474_474641

theorem right_triangle_x_value (BM MA BC CA: ℝ) (M_is_altitude: BM + MA = BC + CA)
  (x: ℝ) (h: ℝ) (d: ℝ) (M: BM = x) (CB: BC = h) (CA: CA = d) :
  x = (2 * h * d - d ^ 2 / 4) / (2 * d + 2 * h) := by
  sorry

end right_triangle_x_value_l474_474641


namespace combination_eight_choose_five_l474_474627

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474627


namespace probability_5_3_l474_474428

def P : ℕ → ℕ → ℝ
| 0, 0 => 1
| x, 0 => 0
| 0, y => 0
| a, b => (P (a-1) b + P a (b-1) + P (a-1) (b-1)) / 3

theorem probability_5_3 : ∃ m n : ℕ, P 5 3 = (m : ℝ) / 3^n ∧ m % 3 ≠ 0 ∧ m + n = 1269 :=
by
  obtain ⟨m, n, hprob, hm, hmn⟩ : ∃ m n : ℕ, P 5 3 = (m : ℝ) / 3^n ∧ m % 3 ≠ 0 ∧ m + n = 1269,
  -- here we will provide proof details which are to be constructed
  sorry

end probability_5_3_l474_474428


namespace parabola_fixed_point_l474_474498

open Classical

structure Point (α : Type) :=
(x y : α)

def parabola_has_vertex_at_origin (C : Type → Type) [has_vertex_origin : ∀ {α}, C α → Prop] := ∀ {α}, has_vertex_origin α

def parabola_focus_on_axis (C : Type → Type) [focus_axis : ∀ {α}, C α → Prop] := ∀ {α}, focus_axis α

def point_on_parabola {C : Type → Type} [point_on : ∀ {α}, Point α → C α → Prop] := ∀ {α} (P : Point α) (c : C α), point_on P c

def chord_condition {C : Type → Type} [chord : ∀ {α}, Point α → Point α → α → C α → Prop] := ∀ {α} (B P Q : Point α) (c : C α) (k_BP k_BQ : α), chord B P k_BP c ∧ chord B Q k_BQ c ∧ k_BP * k_BQ = -2

def fixed_point_condition {C : Type → Type} [passes_fixed_point : ∀ {α}, Point α → Point α → Point α → C α → Prop] := ∀ {α} 
(B P Q fixed : Point α) (c : C α), chord_condition B P Q c ∧ passes_fixed_point P Q fixed c → fixed = Point.mk 3 2

variable (C : Type → Type)
variable (parabola : ∀ {α}, parabola_has_vertex_at_origin C ∧ parabola_focus_on_axis C)
variable [P_on : ∀ {α}, Point α → C α → Prop]
variable [ch : ∀ {α}, Point α → Point α → α → C α → Prop]
variable [passes : ∀ {α}, Point α → Point α → Point α → C α → Prop]

theorem parabola_fixed_point (A B : Point ℝ) (k_BP k_BQ : ℝ) (c : C ℝ) :
  point_on_parabola A c ∧ point_on_parabola B c ∧ A = Point.mk 1 2 ∧ B = Point.mk 1 (-2) ∧ chord_condition B (Point.mk 0 0) (Point.mk 0 0) c k_BP k_BQ →
  fixed_point_condition B (Point.mk 0 0) (Point.mk 0 0) (Point.mk 3 2) c :=
sorry

end parabola_fixed_point_l474_474498


namespace sum_of_first_fifteen_terms_l474_474146

open Nat

-- Define the conditions
def a3 : ℝ := -5
def a5 : ℝ := 2.4

-- The arithmetic progression terms formula
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- The sum of the first n terms formula
def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- The main theorem to prove
theorem sum_of_first_fifteen_terms :
  ∃ (a1 d : ℝ), (a a1 d 3 = a3) ∧ (a a1 d 5 = a5) ∧ (Sn a1 d 15 = 202.5) :=
sorry

end sum_of_first_fifteen_terms_l474_474146


namespace fixed_point_g_l474_474182

theorem fixed_point_g (b : ℝ) (hb_pos : 0 < b) (hb_neq_one : b ≠ 1) (a : ℝ) (ha : a = 1) :
  ∃ x, g x = 1 :=
by
  let g := λ x : ℝ, b ^ (x - a)
  have a_eq_1 : a = 1 := ha
  use 1
  simp [g, a_eq_1]
  exact one_pow _


end fixed_point_g_l474_474182


namespace solution_l474_474017

def Z_star := {z : ℤ // z ≠ 0}
def N_0 := ℕ

def f (a : Z_star) : N_0

axiom axiom1 : ∀ a b : Z_star, (a.val + b.val ≠ 0) → f ⟨a.val + b.val, sorry⟩ ≥ min (f a) (f b)
axiom axiom2 : ∀ a b : Z_star, f ⟨a.val * b.val, sorry⟩ = f a + f b

theorem solution :
  ∀ f : Z_star → N_0, 
    (∀ a b : Z_star, (a.val + b.val ≠ 0) → f (⟨a.val + b.val, sorry⟩) ≥ min (f a) (f b)) →
    (∀ a b : Z_star, f (⟨a.val * b.val, sorry⟩) = f a + f b) →
    (f = λ x, 0) ∨ (∃ p : ℕ, Prime p ∧ f = λ x, padic_valration p x.val) :=
by
  sorry

end solution_l474_474017


namespace fourth_hexagon_dots_l474_474108

   -- Define the number of dots in the first, second, and third hexagons
   def hexagon_dots (n : ℕ) : ℕ :=
     match n with
     | 1 => 1
     | 2 => 8
     | 3 => 22
     | 4 => 46
     | _ => 0

   -- State the theorem to be proved
   theorem fourth_hexagon_dots : hexagon_dots 4 = 46 :=
   by
     sorry
   
end fourth_hexagon_dots_l474_474108


namespace combination_8_5_l474_474617

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474617


namespace lines_through_point_with_equal_intercepts_l474_474570

def point (x y : ℝ) : Prop := ∃ (a b : ℝ), (a, b) = (x, y)

theorem lines_through_point_with_equal_intercepts :
  let P := (11, 1) in
  let lines := { L : ℝ → ℝ | ∃ a b, L = (λ x, a + b * x) ∧ point 11 1 } in
  ∃ (n : ℕ), n = 2 ∧ #(lines) n := 
sorry

end lines_through_point_with_equal_intercepts_l474_474570


namespace minimize_product_roots_p0_l474_474431

-- Definitions for the Lean code
def quadratic_bold_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, p(x) = x^2 + b * x + c) ∧
  ∃ a b c : ℝ, ∃ roots : list ℝ, (roots.length = 4) ∧ (∀ r ∈ roots, p(r) = 0) ∧
  (∀ x, p(x) = (x - a)^2 + b * x + c)

theorem minimize_product_roots_p0 :
  ∃ p : ℝ → ℝ, quadratic_bold_polynomial p ∧ p(0) = 0 :=
by {
  sorry
}

end minimize_product_roots_p0_l474_474431


namespace sum_of_transformed_roots_l474_474460

noncomputable def P (x : ℝ) : ℝ := x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x - 42

theorem sum_of_transformed_roots :
  let roots := {b : ℝ | IsRoot P b} in 
  ∑ (b in roots.to_finset) (if b = 0 then 0 else 1 / (1 + b)) = 10 / 41 :=
by
  sorry

end sum_of_transformed_roots_l474_474460


namespace evaluate_expression_l474_474458

open Real

noncomputable def E : ℝ :=
  0.064 ^ (- 1 / 3) - (- 1 / 8) ^ 0 + 16 ^ (3 / 4) + 0.25 ^ (1 / 2)

theorem evaluate_expression : E = 10 := by
  sorry

end evaluate_expression_l474_474458


namespace inequality_solutions_l474_474043

theorem inequality_solutions (f : ℝ → ℝ) (a x : ℝ)
  (h1 : ∀ (t : ℝ), t ∈ Set.Ici 1 → (∃ y : ℝ, f ((t^y - 4^(-3)) / 2) = y) )
  (h2 : a ≠ 0)
  (h3 : ∀ (y : ℝ), y ≥ 0 → f ((y / 2) + 1) = y):
   (a > 0 → f (a / (x - 2 * a)) ≤ 1 ↔ x ∈ Icc (-26*a / 17) -a) ∧ 
   (a < 0 → f (a / (x - 2 * a)) ≤ 1 ↔ x ∈ Icc -a (42*a / 17)) :=
by
  sorry

end inequality_solutions_l474_474043


namespace conversation_date_l474_474704

def is_good_month (days: Nat) : Prop :=
  ∃ V S P: ℕ, (V = S) ∧ (S = P) ∧ (V + S + P = days)

theorem conversation_date : 
  ∀ m d y : ℕ,
  (is_good_month 28 ∨ is_good_month 29) →
  (∀ k, is_good_month (days_in_month k)) →
  day_of_week m d y = "Monday" →
  (m = 2 ∧ (d = 28 ∨ d = 29)) :=
by
  sorry

end conversation_date_l474_474704


namespace greatest_integer_less_than_200_with_gcd_18_l474_474771

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l474_474771


namespace lining_cost_is_correct_l474_474321

noncomputable def cost_per_yard_of_lining (total_cost : ℝ) (velvet_cost_per_yard : ℝ) (velvet_yards : ℕ) 
    (pattern_cost : ℝ) (thread_cost_per_spool : ℝ) (thread_spools : ℕ) (buttons_cost : ℝ) 
    (trim_cost_per_yard : ℝ) (trim_yards : ℕ) (lining_yards : ℕ) (discount_rate : ℝ) : ℝ :=
  let velvet_total_cost := velvet_cost_per_yard * velvet_yards
  let thread_total_cost := thread_cost_per_spool * thread_spools
  let trim_total_cost := trim_cost_per_yard * trim_yards
  let lining_cost := 4 * total_cost
  let discount_velvet := discount_rate * velvet_total_cost
  let discount_lining := discount_rate * lining_cost
  let total_cost_before_discount := velvet_total_cost + pattern_cost + thread_total_cost + buttons_cost + trim_total_cost + lining_cost
  let total_discount := discount_velvet + discount_lining
  let total_cost_after_discount := total_cost_before_discount - total_discount
  (total_cost_after_discount - 200) / 3.6

theorem lining_cost_is_correct : cost_per_yard_of_lining 310.50 24 5 15 3 2 14 19 3 4 0.10 = 30.694444... :=
sorry

end lining_cost_is_correct_l474_474321


namespace infinite_bounded_sequence_unique_l474_474126

open Nat

theorem infinite_bounded_sequence_unique {a : ℕ → ℕ} :
  (∀ n ≥ 2, a (n+1) = (a n + a (n-1)) / gcd (a n) (a (n-1))) →
  (∃ d, ∀ n, a n = d) ↔ (∀ n, a n = 2) :=
by
  intros h
  existsi 2
  sorry

end infinite_bounded_sequence_unique_l474_474126


namespace light_path_length_correct_l474_474070

def point : Type := ℝ × ℝ

def A : point := (-3, 5)
def B : point := (2, 15)

def line := ℝ → point → bool

def l : line := λ x y, 3 * x - 4 * y + 4 = 0

noncomputable def distance (p1 p2 : point) : ℝ :=
  (Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

noncomputable def light_path_length (A B : point) (l : line) : ℝ :=
  let C := (0, 1) in  -- Intersection point
  let A' := (3, -3) -- Reflection point
  distance A A' + distance A' B

theorem light_path_length_correct : light_path_length A B l = 10 + 5 * Real.sqrt 13 := by
  sorry

end light_path_length_correct_l474_474070


namespace find_Mara_bags_l474_474322

/-- Define the number of bags Mara has -/
variable (x : ℕ)

/-- Define the number of marbles per bag for Mara and calculate the total number of marbles Mara has -/
def Mara_marbles (x : ℕ) : ℕ := 2 * x

/-- Define the number of marbles Markus has -/
def Markus_marbles : ℕ := 2 * 13

/-- Define the condition that Markus has 2 more marbles than Mara -/
def Markus_has_more_marbles (x : ℕ) : Prop := Markus_marbles = Mara_marbles x + 2

/-- Given the conditions, prove that Mara has 12 bags -/
theorem find_Mara_bags (h : Markus_has_more_marbles x) : x = 12 :=
by
  sorry

end find_Mara_bags_l474_474322


namespace problem_i31_problem_i32_problem_i33_problem_i34_l474_474228

-- Problem I3.1
theorem problem_i31 (a : ℝ) :
  a = 1.8 * 5.0865 + 1 - 0.0865 * 1.8 → a = 10 :=
by sorry

-- Problem I3.2
theorem problem_i32 (a b : ℕ) (oh ok : ℕ) (OABC : Prop) :
  oh = ok ∧ oh = a ∧ ok = a ∧ OABC ∧ (b = AC) → b = 10 :=
by sorry

-- Problem I3.3
theorem problem_i33 (b c : ℕ) :
  b = 10 → c = (10 - 2) :=
by sorry

-- Problem I3.4
theorem problem_i34 (c d : ℕ) :
  c = 30 → d = 3 * c → d = 90 :=
by sorry

end problem_i31_problem_i32_problem_i33_problem_i34_l474_474228


namespace smaller_number_l474_474001

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 :=
by
  sorry

end smaller_number_l474_474001


namespace smaller_number_is_24_l474_474038

theorem smaller_number_is_24 (x y : ℕ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : x = 24 :=
by
  sorry

end smaller_number_is_24_l474_474038


namespace total_days_from_1999_to_2005_eq_2557_l474_474980

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def days_in_year (y : ℕ) : ℕ :=
  if is_leap_year y then 366 else 365

def range_of_years := [1999, 2000, 2001, 2002, 2003, 2004, 2005]

def total_days_in_year_range : ℕ :=
  range_of_years.foldl (λ acc y, acc + days_in_year y) 0

theorem total_days_from_1999_to_2005_eq_2557 :
  total_days_in_year_range = 2557 :=
by
  sorry

end total_days_from_1999_to_2005_eq_2557_l474_474980


namespace lee_annual_salary_l474_474664

variable (monthly_savings : ℕ) (months_saving : ℕ) (months_salary_ratio : ℕ)

theorem lee_annual_salary :
  (monthly_savings = 1000) ∧ (months_saving = 10) ∧ (months_salary_ratio = 2) →
  let total_ring_cost := monthly_savings * months_saving in
  let monthly_salary := total_ring_cost / months_salary_ratio in
  let annual_salary := monthly_salary * 12 in
  annual_salary = 60000 :=
by
  sorry

end lee_annual_salary_l474_474664


namespace vertex_of_parabola_l474_474726

theorem vertex_of_parabola (x : ℝ) : 
    (∃ (h k : ℝ), ∀ x, (y = x^2 - 1) → (h, k) = (0, -1)) :=
begin
  sorry
end

end vertex_of_parabola_l474_474726


namespace book_arrangement_count_l474_474219

theorem book_arrangement_count :
  let n := 6
  let identical_pairs := 2
  let total_arrangements_if_unique := n.factorial
  let ident_pair_correction := (identical_pairs.factorial * identical_pairs.factorial)
  (total_arrangements_if_unique / ident_pair_correction) = 180 := by
  sorry

end book_arrangement_count_l474_474219


namespace distribution_ways_l474_474444

theorem distribution_ways :
  let friends := 12
  let problems := 6
  (friends ^ problems = 2985984) :=
by
  sorry

end distribution_ways_l474_474444


namespace checkerboard_pattern_l474_474455

-- Define the infinite grid as a set of coordinates
def infinite_grid_plane := ℤ × ℤ

-- Define the color type
inductive color
| black
| white

-- Define a 2018-cell block
def is_2018_cell_block (b: set infinite_grid_plane) : Prop :=
  ∃ x0 y0, b = { (x, y) | x ≥ x0 ∧ x < x0 + 2018 ∧ y ≥ y0 ∧ y < y0 + 2018 }

-- Define the condition of equal black and white cells in any 2018-cell block
def equal_colored_2018_block (grid : infinite_grid_plane → color) (b: set infinite_grid_plane) : Prop :=
  is_2018_cell_block b ∧
  ((b.filter (λ c, grid c = color.black)).size = (b.filter (λ c, grid c = color.white)).size)

-- Define the checkerboard pattern condition
def is_checkerboard (grid : infinite_grid_plane → color) : Prop :=
  ∀ x y, grid (x, y) = color.black ↔ (x + y) % 2 = 0

-- The main theorem that needs to be proved
theorem checkerboard_pattern (grid : infinite_grid_plane → color) 
  (h: ∀ b, is_2018_cell_block b → equal_colored_2018_block grid b) : is_checkerboard grid := 
sorry

end checkerboard_pattern_l474_474455


namespace combination_choosing_four_socks_l474_474712

theorem combination_choosing_four_socks (n k : ℕ) (h_n : n = 7) (h_k : k = 4) :
  (nat.choose n k) = 35 :=
by
  rw [h_n, h_k, nat.choose_succ_succ, nat.choose_succ_succ, nat.choose_succ_succ, nat.choose_zero_succ]
  simp only [nat.choose_succ_succ, nat.factorial_succ, nat.factorial, nat.succ_sub_succ_eq_sub, nat.sub_zero,
    nat.pred_succ, nat.factorial_succ, nat.choose_self, show nat.factorial 0 = 1 by rfl, tsub_zero,
    mul_one, mul_over_nat_eq, nat.succ.sub_prime, nat.succ_sub_succ_eq_sub, nat.factorial_zero]
  norm_num
  sorry

end combination_choosing_four_socks_l474_474712


namespace expenditure_ratio_l474_474031

theorem expenditure_ratio (I_A I_B E_A E_B : ℝ) (h1 : I_A / I_B = 5 / 6)
  (h2 : I_B = 7200) (h3 : 1800 = I_A - E_A) (h4 : 1600 = I_B - E_B) :
  E_A / E_B = 3 / 4 :=
sorry

end expenditure_ratio_l474_474031


namespace p_and_q_and_not_not_p_or_q_l474_474559

theorem p_and_q_and_not_not_p_or_q (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end p_and_q_and_not_not_p_or_q_l474_474559


namespace roots_of_quadratic_l474_474231

theorem roots_of_quadratic (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = 0) (h3 : a - b + c = 0) : 
  (a * 1 ^2 + b * 1 + c = 0) ∧ (a * (-1) ^2 + b * (-1) + c = 0) :=
sorry

end roots_of_quadratic_l474_474231


namespace series_bound_l474_474949

theorem series_bound (n : ℕ) (h : 2 ≤ n) :
  (4:ℝ) / 7 < (finset.range (2 * n) \ finset.range (n)).sum (λ k, (-1) ^ k * (1 / (k + 1 : ℝ))) ∧
  (finset.range (2 * n) \ finset.range (n)).sum (λ k, (-1) ^ k * (1 / (k + 1 : ℝ))) < real.sqrt 2 / 2 :=
sorry

end series_bound_l474_474949


namespace functional_equation_solution_l474_474125

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (f x + f y)) = f x + y) : ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equation_solution_l474_474125


namespace find_projection_l474_474024

noncomputable def projection_vector (v : ℝ × ℝ) (a b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
a + (λ t, t • (b - a)) = p ∧ 
(p.1 * v.1 + p.2 * v.2 = 0)

theorem find_projection :
  ∃ p : ℝ × ℝ, 
  projection_vector (7, -3) (-3, 2) (4, -1) p ∧ p = (15/58, 35/58) :=
begin
  use (15/58, 35/58),
  sorry
end

end find_projection_l474_474024


namespace area_inequality_a_l474_474032

theorem area_inequality_a
  (circle : Type)
  (A B C D E : circle) 
  (arc_division : is_arc_division B C D E 4)
  (S : circle → circle → circle → ℝ) :
  S A C E < 8 * S B C D := 
  sorry

end area_inequality_a_l474_474032


namespace find_g_of_2_l474_474993

theorem find_g_of_2 {g : ℝ → ℝ} (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
sorry

end find_g_of_2_l474_474993


namespace part1_part2_l474_474790

-- Define the main condition of the farthest distance formula
def distance_formula (S h : ℝ) : Prop := S^2 = 1.7 * h

-- Define part 1: Given h = 1.7, prove S = 1.7
theorem part1
  (h : ℝ)
  (hyp : h = 1.7)
  : ∃ S : ℝ, distance_formula S h ∧ S = 1.7 :=
by
  sorry
  
-- Define part 2: Given S = 6.8 and height of eyes to ground 1.5, prove the height of tower = 25.7
theorem part2
  (S : ℝ)
  (h1 : ℝ)
  (height_eyes_to_ground : ℝ)
  (hypS : S = 6.8)
  (height_eyes_to_ground_eq : height_eyes_to_ground = 1.5)
  : ∃ h : ℝ, distance_formula S h ∧ (h - height_eyes_to_ground) = 25.7 :=
by
  sorry

end part1_part2_l474_474790


namespace urn_probability_l474_474861

-- Necessary definitions corresponding to the conditions.
axiom initial_urn : ℕ → ℕ → ℕ -- Represent the initial state of the urn (red balls, blue balls, total balls).
axiom operation : ℕ → ℕ → ℕ -- Represent the operation (red balls added, blue balls added, new total).

-- Main theorem statement.
theorem urn_probability :
  initial_urn 2 1 3 → 
  (∀ n, n = 5 → operation (λ red blue, if .. then (red + 1, blue) else (red, blue + 1))) →
  initial_urn _ _ 12 → 
  (6 + 6 = 12) →
  probability_urn 6 6 = 8 / 21 :=
sorry

end urn_probability_l474_474861


namespace find_g_of_2_l474_474992

theorem find_g_of_2 {g : ℝ → ℝ} (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
sorry

end find_g_of_2_l474_474992


namespace gangster_survival_l474_474343

/-- Ten gangsters stand in a field. Each pair of gangsters has a different distance
between them and each gangster shoots the nearest gangster dead upon the clock striking.
We need to prove that the largest number of gangsters that can survive is 7. -/
theorem gangster_survival {G : Type} [fintype G] [decidable_eq G] (gangsters : fin 10 → G)
  (dist : G → G → ℝ) (unique_distances : ∀ (i j k : G), i ≠ j → j ≠ k → k ≠ i → 
  dist i j ≠ dist i k) :
  ∃ (survivors : fin 10 → G), (fin 10 → G) → (fin 7 → G) where 
  (∀ (i j : G), i ≠ j → dist i j ≠ dist i j) ∧ (∀ (i : G), ∃ j : G, dist i j = 0) :=
begin
  sorry
end

end gangster_survival_l474_474343


namespace sequence_contains_composite_l474_474838

theorem sequence_contains_composite (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 1 → (a (n + 1) = 2 * a n + 1 ∨ a (n + 1) = 2 * a n - 1))
  (h2 : ∀ n, a n > 0) 
  (h3 : ∃ n, ∃ m, n ≠ m) : 
  ∃ n, ∃ d, d ≠ 1 ∧ d ≠ a n ∧ d ∣ a n :=
by sorry

end sequence_contains_composite_l474_474838


namespace find_f_ln_1_half_l474_474946

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def f : ℝ → ℝ :=
λ x, if x > 0 then exp (-x) - 2 else -(exp (x) - 2)

theorem find_f_ln_1_half 
  (h_odd : odd_function f)
  (h_pos : ∀ x > 0, f x = exp (-x) - 2) :
  f (log (1 / 2)) = 3 / 2 :=
sorry

end find_f_ln_1_half_l474_474946


namespace derivative_y_l474_474042
open Real

noncomputable def y (x : ℝ) := (2 / (3 * x - 2)) * sqrt(-3 + 12 * x - 9 * x^2) +
                                log (((1 + sqrt(-3 + 12 * x - 9 * x^2)) / (3 * x - 2)))

noncomputable def z (x : ℝ) := sqrt(-3 + 12 * x - 9 * x^2)

theorem derivative_y (x : ℝ) :
  (deriv y x) = (3 - 9 * x) / (sqrt (-3 + 12 * x - 9 * x^2) * (3 * x - 2)) :=
sorry

end derivative_y_l474_474042


namespace inscribed_circles_tangent_same_point_l474_474168

theorem inscribed_circles_tangent_same_point
  (A B C D : Point)
  (h : ∃O, inscribed_circle ABCD O)
  (P1 Q1 R : Point)
  (P2 Q2 R : Point) :
  (circle_in_triangle_tangent ABC P1 Q1 R) →
  (circle_in_triangle_tangent ADC P2 Q2 R) →
  (tangent_points_coincide R :=
  sorry

end inscribed_circles_tangent_same_point_l474_474168


namespace area_triangle_AGB_l474_474644

noncomputable def point := ℝ × ℝ
def length (a b : point) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

structure Triangle :=
  (A B C D E G : point)
  (AB AD BE : ℝ)
  (D_is_midpoint : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (E_is_midpoint : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (AD_length : length A D = 15)
  (BE_length : length B E = 24)
  (AB_length : length A B = 30)
  (AG_extends_AD : ∃ x, G = (A.1 + x * (D.1 - A.1), A.2 + x * (D.2 - A.2))) -- Extend AD to G on circumcircle

axiom area_of_triangle : Triangle → ℝ -- Axiomatize area calculation

theorem area_triangle_AGB (T : Triangle) : 
  let area_ABC := area_of_triangle T in
  let area_AGB := (5 / 6) * area_ABC in
  area_of_triangle {T with C := T.B} = area_AGB :=
sorry

end area_triangle_AGB_l474_474644


namespace number_of_sets_P_l474_474536

-- Define the universal set U such that U = {x ∈ ℤ | -5 < x < 5}
def U : set ℤ := {x | -5 < x ∧ x < 5}

-- Define the set S = {-1, 1, 3}
def S : set ℤ := {-1, 1, 3}

-- Statement to prove
theorem number_of_sets_P : 
  ∃ (n : ℕ), n = 8 ∧ ∀ P : set ℤ, (U \ P ⊆ S) → (set.card {P | U \ P ⊆ S} = n) :=
sorry

end number_of_sets_P_l474_474536


namespace greatest_possible_grapes_thrown_out_l474_474050

theorem greatest_possible_grapes_thrown_out (n : ℕ) : 
  n % 7 ≤ 6 := by 
  sorry

end greatest_possible_grapes_thrown_out_l474_474050


namespace maximum_xy_l474_474572

theorem maximum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : xy ≤ 2 :=
sorry

end maximum_xy_l474_474572


namespace intersection_of_sets_l474_474531

noncomputable def A : Set ℝ := {x | 2^(2 * x + 1) ≥ 4}
noncomputable def B : Set ℝ := {x | 2 - x > 0}

theorem intersection_of_sets :
  A ∩ B = {x | 1 / 2 ≤ x ∧ x < 2} := sorry

end intersection_of_sets_l474_474531


namespace choose_five_from_eight_l474_474602

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474602


namespace g_of_2_eq_14_l474_474990

theorem g_of_2_eq_14 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 := 
sorry

end g_of_2_eq_14_l474_474990


namespace sin_sum_max_l474_474950

theorem sin_sum_max (x y : ℝ) (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2) (h_sum : x + y = π/3) : 
  ∃ M, M = 1 ∧ ∀ a b, (0 < a ∧ a < π/2) ∧ (0 < b ∧ b < π/2) ∧ (a + b = π/3) → sin a + sin b ≤ M := 
sorry

end sin_sum_max_l474_474950


namespace comb_8_5_eq_56_l474_474596

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474596


namespace shaded_region_area_l474_474336

-- Define the given problem as per the conditions
structure Circle (P : Type) := (center : P) (radius : ℝ)
structure Point (P : Type) (x : ℝ) (y : ℝ)
structure Line (P : Type) := (start : P) (end : P) 
structure Orthogonal (L1 L2 : Line ℝ)

noncomputable def Area (P : Type) (c : Circle P) (s : Set P) : ℝ := sorry

-- Define all conditions
def semicircle_center := Point ℝ 0 0 
def semicircle := Circle Point semicircle_center 2
def point_D := Point ℝ 2 0 
def line_CD := Line semicircle_center point_D 
def line_AB := Line semicircle_center (Point ℝ 0 2) -- Placeholder for line AB

-- Orthogonality condition
def ortho_CD_AB := Orthogonal line_CD line_AB

def point_E := Point ℝ (3 * 2 * 2) 0 -- Placeholder as per the problem definitions
def point_F := Point ℝ 0 (3 * 2 * 2) -- Placeholder as per the problem definitions
def line_BE := Line Point point_D point_E 
def line_AF := Line Point point_D point_F

-- Prove that the area of AEFBDA is 7π
theorem shaded_region_area : Area Point semicircle (Set.Point AEFBDA) = 7 * π := 
by
  sorry

end shaded_region_area_l474_474336


namespace combination_8_5_l474_474622

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474622


namespace g_of_2_eq_14_l474_474991

theorem g_of_2_eq_14 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 := 
sorry

end g_of_2_eq_14_l474_474991


namespace find_projection_l474_474025

noncomputable def projection_vector (v : ℝ × ℝ) (a b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
a + (λ t, t • (b - a)) = p ∧ 
(p.1 * v.1 + p.2 * v.2 = 0)

theorem find_projection :
  ∃ p : ℝ × ℝ, 
  projection_vector (7, -3) (-3, 2) (4, -1) p ∧ p = (15/58, 35/58) :=
begin
  use (15/58, 35/58),
  sorry
end

end find_projection_l474_474025


namespace locus_of_intersections_is_circle_l474_474110

noncomputable def locus_of_intersections (A : Point) (π : Plane) (α : ℝ) (h₀ : A ∉ π) (hα : α ≠ 0) : Set Point :=
  let O := orthogonal_projection A π
  { M | M ∈ π ∧ (∃ L : Line, A ∈ L ∧ M ∈ L ∧ ∀ N ∈ L, angle_between_line_and_plane N π = α) }

theorem locus_of_intersections_is_circle (A : Point) (π : Plane) (α : ℝ) (h₀ : A ∉ π) (hα : α ≠ 0) :
  ∃ O : Point, ∃ r : ℝ, 
    O = orthogonal_projection A π ∧ 
    r = (distance A (orthogonal_projection A π)) * Real.cot α ∧ 
    ∀ M : Point, M ∈ locus_of_intersections A π α h₀ hα ↔ dist M O = r :=
by 
  sorry

end locus_of_intersections_is_circle_l474_474110


namespace quadratic_distinct_real_roots_l474_474577

theorem quadratic_distinct_real_roots (k : ℝ) : 
  (∀ (x : ℝ), (k - 1) * x^2 + 4 * x + 1 = 0 → False) ↔ (k < 5 ∧ k ≠ 1) :=
by
  sorry

end quadratic_distinct_real_roots_l474_474577


namespace letter_1992_in_sequence_l474_474390

def sequence : List Char := ['A', 'B', 'C', 'D', 'E', 'D', 'C', 'B', 'A']

def nth_letter_in_repeating_sequence (n : Nat) (seq : List Char) : Char :=
  seq.get! (n % seq.length)

theorem letter_1992_in_sequence : nth_letter_in_repeating_sequence 1992 sequence = 'C' :=
by
  sorry

end letter_1992_in_sequence_l474_474390


namespace max_P_value_l474_474483

noncomputable def P (a : ℝ) : ℝ :=
   ∫ x in 0..a, ∫ y in 0..1, if (Real.sin (π * x))^2 + (Real.sin (π * y))^2 > 1 then 1 else 0

theorem max_P_value : 
   ∃ (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1), P 1 = 2 - Real.sqrt 2 :=
by {
  use 1,
  split,
  { exact le_refl 1 },
  { exact le_of_eq (eq.refl 1) },
  sorry
}

end max_P_value_l474_474483


namespace find_angles_l474_474744

-- Definitions for the cyclic quadrilateral and conditions
variable (A B C D X : Type)
variable (angle : A → B → C → D → ℝ)

variable (AB AD DC BX : ℝ)
variable (angleB angleXDC : ℝ)
variable [h01 : AB = BX] [h02 : AD = DC]
variable [h03 : angle B B C D = 34] [h04 : angle X D C C = 52]

-- Theorem to prove the angles
theorem find_angles (AB : ℝ) (AD DC : ℝ) (BX : ℝ) (angleB angleXDC : ℝ) :
  AB = BX ∧ AD = DC ∧ angle B B C D = 34 ∧ angle X D C C = 52 →
  angle A X C D = 107 ∧ angle A C B B = 47 :=
by
  intros
  sorry

end find_angles_l474_474744


namespace school_fundraised_amount_l474_474014

def school_fundraising : ℝ :=
  let mrs_johnson_class := 2300
  let mrs_sutton_class := mrs_johnson_class / 2
  let miss_rollin_class := mrs_sutton_class * 8
  let total_school_raised := miss_rollin_class * 3
  let admin_fee := total_school_raised * 0.02
  total_school_raised - admin_fee

theorem school_fundraised_amount : school_fundraising = 27048 :=
by
  -- begin
  -- proof steps would go here
  -- end

  sorry

end school_fundraised_amount_l474_474014


namespace find_cars_oil_change_l474_474279

variable (minutes_per_wash : ℕ) (minutes_per_oil_change : ℕ) (minutes_per_tire_set : ℕ) : Prop
variable (cars_washed : ℕ) (tire_sets_changed : ℕ) (total_work_time : ℕ) : Prop

def total_minutes_washing (minutes_per_wash : ℕ) (cars_washed : ℕ) : ℕ := 
  cars_washed * minutes_per_wash

def total_minutes_tire_change (minutes_per_tire_set : ℕ) (tire_sets_changed : ℕ) : ℕ := 
  tire_sets_changed * minutes_per_tire_set

def total_minutes_oil_change (total_work_time minutes_washing minutes_tire_change : ℕ) : ℕ := 
  total_work_time - minutes_washing - minutes_tire_change

def cars_oil_changed (minutes_per_oil_change total_minutes_oil_change : ℕ) : ℕ := 
  total_minutes_oil_change / minutes_per_oil_change

theorem find_cars_oil_change (
  h1 : minutes_per_wash = 10) 
  (h2 : minutes_per_oil_change = 15) 
  (h3 : minutes_per_tire_set = 30)
  (h4 : cars_washed = 9)
  (h5 : tire_sets_changed = 2)
  (h6 : total_work_time = 240) : 
  cars_oil_changed minutes_per_oil_change (
   total_minutes_oil_change 
     total_work_time 
     (total_minutes_washing minutes_per_wash cars_washed)
     (total_minutes_tire_change minutes_per_tire_set tire_sets_changed)) = 6 := by 
  sorry

end find_cars_oil_change_l474_474279


namespace num_first_graders_in_class_l474_474378

def numKindergartners := 14
def numSecondGraders := 4
def totalStudents := 42

def numFirstGraders : Nat := totalStudents - (numKindergartners + numSecondGraders)

theorem num_first_graders_in_class :
  numFirstGraders = 24 :=
by
  sorry

end num_first_graders_in_class_l474_474378


namespace book_arrangement_l474_474083

theorem book_arrangement (total_books : ℕ) (geometry_books : ℕ) (number_theory_books : ℕ) (first_book_geometry : Prop)
  (h_total : total_books = 9)
  (h_geometry : geometry_books = 4)
  (h_number_theory : number_theory_books = 5)
  (h_first_geometry : first_book_geometry)
  : nat.choose 8 3 = 56 := 
by {
  -- Since we know total_books = geometry_books + number_theory_books and first_book_geometry holds,
  -- we just calculate the combination directly as in the problem statement:
  calc
  nat.choose 8 3 = 56 : by sorry -- skipping the proof step, as instructed.
}

end book_arrangement_l474_474083


namespace perimeter_of_polygon_l474_474662

theorem perimeter_of_polygon {n : ℕ} (P : ℝ × ℝ) : 
  (∀ (i : ℕ), i < n → ∃ (C : ℝ × ℝ), dist P C = 1 ∧ dist P (C + 1) = 1 ∧ (C - P).norm = 1 ∧ (C + (1 : ℝ)) - P).norm = 1) 
  ∧ (P ∈ region_of_polygon (set_of (λ (C : ℝ × ℝ), dist P C = 1))) → 
  perimeter_of_polygon P n = 4 * Real.pi :=
by
  sorry

end perimeter_of_polygon_l474_474662


namespace solution_in_quadrant_II_l474_474673

theorem solution_in_quadrant_II (k x y : ℝ) (h1 : 2 * x + y = 6) (h2 : k * x - y = 4) : x < 0 ∧ y > 0 ↔ k < -2 :=
by
  sorry

end solution_in_quadrant_II_l474_474673


namespace star_7_2_l474_474463

def star (a b : ℕ) := 4 * a - 4 * b

theorem star_7_2 : star 7 2 = 20 := 
by
  sorry

end star_7_2_l474_474463


namespace wheel_diameter_correct_l474_474438

noncomputable def wheel_diameter (distance : ℝ) (revolutions : ℝ) : ℝ :=
  let circumference := distance / revolutions
  circumference / Real.pi

theorem wheel_diameter_correct : wheel_diameter 968 11.010009099181074 ≈ 27.979 := by
  sorry

end wheel_diameter_correct_l474_474438


namespace possible_value_of_b_l474_474177

theorem possible_value_of_b (a b : ℕ) (H1 : b ∣ (5 * a - 1)) (H2 : b ∣ (a - 10)) (H3 : ¬ b ∣ (3 * a + 5)) : 
  b = 49 :=
sorry

end possible_value_of_b_l474_474177


namespace sum_of_reciprocals_l474_474340

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + y = 3 * x * y) (h2 : x - y = 2) : (1/x + 1/y) = 4/3 :=
by
  -- Proof omitted
  sorry

end sum_of_reciprocals_l474_474340


namespace distance_from_minus_one_is_four_or_minus_six_l474_474359

theorem distance_from_minus_one_is_four_or_minus_six :
  {x : ℝ | abs (x + 1) = 5} = {-6, 4} :=
sorry

end distance_from_minus_one_is_four_or_minus_six_l474_474359


namespace exists_zero_in_interval_l474_474132

   noncomputable def f : ℝ → ℝ := λ x, (Real.log x - (1 / x))

   theorem exists_zero_in_interval :
     ∃ c ∈ (Set.Ioo 2 3), f c = 0 :=
   begin
     sorry
   end
   
end exists_zero_in_interval_l474_474132


namespace total_cost_oranges_and_apples_l474_474865

def cost_of_oranges (price_per_pound: ℝ) (weight: ℝ) : ℝ :=
  (3 / 4) * weight

def cost_of_apples (price_per_pound: ℝ) (weight: ℝ) : ℝ :=
  (5 / 6) * weight

theorem total_cost_oranges_and_apples :
  (cost_of_oranges 3 12) + (cost_of_apples 5 18) = 24 :=
by
  -- Here we calculate each cost individually and then add them
  sorry

end total_cost_oranges_and_apples_l474_474865


namespace range_of_a_l474_474224

theorem range_of_a (a : ℝ) (h : sqrt a^2 = -a) : a ≤ 0 := by
  sorry

end range_of_a_l474_474224


namespace left_handed_rock_music_lovers_l474_474585

theorem left_handed_rock_music_lovers (total_club_members left_handed_members rock_music_lovers right_handed_dislike_rock: ℕ)
  (h1 : total_club_members = 25)
  (h2 : left_handed_members = 10)
  (h3 : rock_music_lovers = 18)
  (h4 : right_handed_dislike_rock = 3)
  (h5 : total_club_members = left_handed_members + (total_club_members - left_handed_members))
  : (∃ x : ℕ, x = 6 ∧ x + (left_handed_members - x) + (rock_music_lovers - x) + right_handed_dislike_rock = total_club_members) :=
sorry

end left_handed_rock_music_lovers_l474_474585


namespace average_speed_l474_474084

theorem average_speed (x : ℝ) (h₀ : x > 0) : 
  let time1 := x / 90
  let time2 := 2 * x / 20
  let total_distance := 3 * x
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 27 := 
by
  sorry

end average_speed_l474_474084


namespace Eithan_savings_account_l474_474387

variable (initial_amount wife_firstson_share firstson_remaining firstson_secondson_share 
          secondson_remaining secondson_thirdson_share thirdson_remaining 
          charity_donation remaining_after_charity tax_rate final_remaining : ℝ)

theorem Eithan_savings_account:
  initial_amount = 5000 →
  wife_firstson_share = initial_amount * (2/5) →
  firstson_remaining = initial_amount - wife_firstson_share →
  firstson_secondson_share = firstson_remaining * (3/10) →
  secondson_remaining = firstson_remaining - firstson_secondson_share →
  thirdson_remaining = secondson_remaining * (1-0.30) →
  charity_donation = 200 →
  remaining_after_charity = thirdson_remaining - charity_donation →
  tax_rate = 0.05 →
  final_remaining = remaining_after_charity * (1 - tax_rate) →
  final_remaining = 927.2 := 
  by
    intros
    sorry

end Eithan_savings_account_l474_474387


namespace weight_of_new_person_l474_474344

theorem weight_of_new_person
  (average_weight_increase : 8 * 5)
  (replaced_weight : 65) :
  ∃ (new_weight : ℕ), new_weight = replaced_weight + average_weight_increase := 
by
  use 105
  have h : 105 = 65 + 40 := rfl
  exact h

end weight_of_new_person_l474_474344


namespace no_non_similar_triangles_with_geometric_angles_l474_474210

theorem no_non_similar_triangles_with_geometric_angles :
  ¬∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ a ≠ a * r ∧ a ≠ a * r * r ∧ a * r ≠ a * r * r ∧
  a + a * r + a * r * r = 180 :=
by
  sorry

end no_non_similar_triangles_with_geometric_angles_l474_474210


namespace range_of_d_l474_474317

variable {S : ℕ → ℝ} -- S is the sum of the series
variable {a : ℕ → ℝ} -- a is the arithmetic sequence

theorem range_of_d (d : ℝ) (h1 : a 3 = 12) (h2 : S 12 > 0) (h3 : S 13 < 0) :
  -24 / 7 < d ∧ d < -3 := sorry

end range_of_d_l474_474317


namespace harkamal_shopping_l474_474543

theorem harkamal_shopping 
  (grapes_cost : ℕ := 8 * 70)
  (mangoes_cost : ℕ := 9 * 55)
  (apples_cost : ℕ := 4 * 40)
  (oranges_cost : ℕ := 6 * 30)
  (pineapples_cost : ℕ := 2 * 90)
  (cherries_cost : ℕ := 5 * 100)
  (total_cost : ℕ := grapes_cost + mangoes_cost + apples_cost + oranges_cost + pineapples_cost + cherries_cost)
  (discount : ℝ := 0.05 * total_cost)
  (discounted_total : ℝ := total_cost.toReal - discount)
  (sales_tax : ℝ := 0.10 * discounted_total)
  (final_amount : ℝ := discounted_total + sales_tax) :
  final_amount = 2168.375 :=
sorry

end harkamal_shopping_l474_474543


namespace problem_solution_l474_474171

open Real

noncomputable
def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∃ (x y : ℝ), x = 1 ∧ y = sqrt 2 / 2 ∧
  x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ (c1 c2 m : ℝ), c1 = c2 ∧ a = sqrt 2 * b ∧ m = b ∧ 
  (1 + sqrt 2 / 2) ^ 2 / (2 * c1^2) + (sqrt 2 / 2) ^ 2 / (c2^2) = 1) ∧
  a = sqrt 2 ∧ b = 1 ∧
  ∀ (x y : ℝ), x^2 / 2 + y^2 = 1

noncomputable
def fixed_point_T : Prop :=
  ∃ (T : ℝ × ℝ), T = (0, 1) ∧ 
  ∀ (m n : ℝ), ∃ (A B : ℝ × ℝ), 
  A ≠ B ∧
  ((m * A.1 + n * (A.2 + 1/3 * n) = 0 ∧ 
  m * B.1 + n * (B.2 + 1/3 * n) = 0)) ∧
  (circle_with_diameter A B passes_through (0, 1))

theorem problem_solution : 
  ellipse_equation ∧ fixed_point_T :=
  by sorry

end problem_solution_l474_474171


namespace sum_of_infinite_series_l474_474875

noncomputable def T : ℝ := ∑ n in (finset.Ico 1 1000), (5 * n - 2) / (3^n)

theorem sum_of_infinite_series : T = 1 / 4 := 
by sorry

end sum_of_infinite_series_l474_474875


namespace finite_difference_zero_for_k4_l474_474920

def sequence (n : ℕ) : ℤ := (n : ℤ)^3 - (n : ℤ)

noncomputable def finite_difference (k : ℕ) (u : ℕ → ℤ) : (ℕ → ℤ) :=
  nat.rec_on k u (λ k' diff_fn n, diff_fn (n + 1) - diff_fn n)

theorem finite_difference_zero_for_k4 : 
  ∃ k, k = 4 ∧ ∀ n, finite_difference k sequence n = 0 :=
by 
  sorry

end finite_difference_zero_for_k4_l474_474920


namespace count_non_return_sampling_l474_474261

-- Definitions of the sampling methods
def random_sampling : Type := unit
def stratified_sampling : Type := unit
def systematic_sampling : Type := unit

-- Definition to state that a sampling method is non-return sampling
def is_non_return_sampling (method : Type) : Prop :=
  method = unit  -- since all methods are non-return sampling, we define it simply

-- Set the list of sampling methods
def sampling_methods : List Type := [random_sampling, stratified_sampling, systematic_sampling]

-- The theorem to prove
theorem count_non_return_sampling : (sampling_methods.filter is_non_return_sampling).length = 3 := by
  sorry

end count_non_return_sampling_l474_474261


namespace restore_c_l474_474086

open Nat

theorem restore_c (a b c x y z : ℕ)
  (h1 : a = x * y)
  (h2 : b = x * z)
  (h3 : c = y * z)
  (coprime_x_y : gcd x y = 1)
  (coprime_x_z : gcd x z = 1)
  (coprime_y_z : gcd y z = 1) :
  c = lcm a b / gcd a b :=
by
  sorry

end restore_c_l474_474086


namespace twelve_vases_visibility_thirteen_vases_no_visibility_l474_474337

/-
Part (a): Prove that if there are 12 vases, there are two people who can see each other.
-/
theorem twelve_vases_visibility :
  ∀ (people : Finset Point) (vases : Finset Point),
    people.card = 7 →
    vases.card = 12 →
    (∃ p1 p2 ∈ people, p1 ≠ p2 ∧ ∀ vase ∈ vases, Segment p1 p2 ∉ vase) :=
by
  sorry

/-
Part (b): Prove that with 13 vases, one can arrange people and vases such that no two people can see each other.
-/
theorem thirteen_vases_no_visibility :
  ∃ (people : Finset Point) (vases : Finset Point),
    people.card = 7 ∧
    vases.card = 13 ∧
    ∀ p1 p2 ∈ people, p1 ≠ p2 → ∃ vase ∈ vases, Segment p1 p2 ∈ vase :=
by
  sorry

end twelve_vases_visibility_thirteen_vases_no_visibility_l474_474337


namespace right_triangle_on_complex_plane_l474_474554

theorem right_triangle_on_complex_plane : 
∀ (z : ℂ), z ≠ 0 ∧ (∀ w : ℂ, w ≠ 0 → (w^2 ≠ w ∧ ∃ θ : ℝ, z = cos θ + sin θ * complex.I)) → 
z = complex.I ∨ z = -complex.I  

end right_triangle_on_complex_plane_l474_474554


namespace greatest_integer_gcd_l474_474773

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l474_474773


namespace new_students_count_l474_474592

variables (initial_students students_left final_students new_students : ℕ)
hypothesis (h_initial : initial_students = 8)
hypothesis (h_left : students_left = 5)
hypothesis (h_final : final_students = 11)

theorem new_students_count :
  new_students = final_students - (initial_students - students_left) :=
sorry

end new_students_count_l474_474592


namespace combination_8_5_is_56_l474_474615

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474615


namespace basketball_team_wins_l474_474815

theorem basketball_team_wins (played_games won_percentage : ℝ) (additional_games : ℤ) (target_percentage : ℝ) :
  played_games = 40 → won_percentage = 0.7 → additional_games = 10 → target_percentage = 0.6 →
  ∃ (lost_games : ℤ), lost_games = 8 ∧ (let total_games := played_games + additional_games in
                                         let won_games := won_percentage * played_games + (additional_games - lost_games : ℝ) in
                                         won_games / total_games = target_percentage) :=
by
  intros played_games_eq won_percentage_eq additional_games_eq target_percentage_eq
  use 8
  rw [played_games_eq, won_percentage_eq, additional_games_eq]
  dsimp
  norm_num
  rw [mul_div, ← add_div, mul_comm 0.7 40, ← mul_assoc, mul_div]
  norm_num
  ring

# Note: Additional normalization and specific rewrites are not required for the problem statement itself.

end basketball_team_wins_l474_474815


namespace kamal_marks_in_chemistry_l474_474283

theorem kamal_marks_in_chemistry
  (eng : ℕ := 76)
  (math : ℕ := 65)
  (phys : ℕ := 82)
  (bio : ℕ := 85)
  (avg : ℕ := 75) :
  let chem := 67
  in (eng + math + phys + bio + chem) / 5 = avg :=
by
  -- the proof will be here
  sorry

end kamal_marks_in_chemistry_l474_474283


namespace find_a_l474_474994

theorem find_a (a b c : ℝ) (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
                 (h2 : a * 15 * 7 = 1.5) : a = 6 :=
sorry

end find_a_l474_474994


namespace comb_8_5_eq_56_l474_474594

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l474_474594


namespace sum_of_squares_geq_one_over_n_l474_474478

theorem sum_of_squares_geq_one_over_n {n : ℕ} {a : Fin n → ℝ}
  (h_sum : ∑ i, a i = 1) : 
  ∑ i, (a i)^2 ≥ 1 / n := 
sorry

end sum_of_squares_geq_one_over_n_l474_474478


namespace a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l474_474414

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

def complex_from_a (a : ℝ) : ℂ :=
  (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

theorem a_minus_two_sufficient_but_not_necessary_for_pure_imaginary :
  (is_pure_imaginary (complex_from_a (-2))) ∧ ¬ (∀ (a : ℝ), is_pure_imaginary (complex_from_a a) → a = -2) :=
by
  sorry

end a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l474_474414


namespace combination_8_5_is_56_l474_474613

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l474_474613


namespace Turan_2_l474_474315

noncomputable theory

open Classical

variable {V : Type*} (G : SimpleGraph V) [Fintype V] [DecidableEq V]

def no_K3 (G : SimpleGraph V) := ∀ (a b c : V), G.Adj a b → G.Adj b c → G.Adj c a → False

theorem Turan_2 (G : SimpleGraph V) [NoK3 : no_K3 G] : G.edgeFinset.card ≤ Fintype.card V * Fintype.card V / 4 := sorry

end Turan_2_l474_474315


namespace cannot_determine_remaining_pictures_l474_474800

theorem cannot_determine_remaining_pictures (taken_pics : ℕ) (dolphin_show_pics : ℕ) (total_pics : ℕ) :
  taken_pics = 28 → dolphin_show_pics = 16 → total_pics = 44 → 
  (∀ capacity : ℕ, ¬ (total_pics + x = capacity)) → 
  ¬ ∃ remaining_pics : ℕ, remaining_pics = capacity - total_pics :=
by {
  sorry
}

end cannot_determine_remaining_pictures_l474_474800


namespace midpoints_collinear_l474_474708

-- Define the type for points 
structure Point where
  x : ℝ
  y : ℝ

-- Define the notion of a line 
def Line (p1 p2 : Point) : Set Point :=
  { p : Point | ∃ l : ℝ, p.x = p1.x + l * (p2.x - p1.x) ∧ p.y = p1.y + l * (p2.y - p1.y) }

-- Collinearity predicate
def collinear (p1 p2 p3 : Point) : Prop :=
  ∃ (c1 c2 : ℝ), p3.x = c1 * p1.x + c2 * p2.x ∧ p3.y = c1 * p1.y + c2 * p2.y

-- Midpoint of two points
def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

-- Definition of Complete Quadrilateral
noncomputable def complete_quadrilateral := {ABCD : List Point // 
  ABCD.length = 4 ∧ 
  (∃ E, E.x ∈ Line ABCD[0] ABCD[1] ∧ E.x ∈ Line ABCD[2] ABCD[3]) ∧ 
  (∃ F, F.x ∈ Line ABCD[0] ABCD[3] ∧ F.x ∈ Line ABCD[1] ABCD[2])}

-- Main theorem statement
theorem midpoints_collinear {ABCD : complete_quadrilateral} 
        {E F X Y Z : Point}
        (hE : E ∈ Line (ABCD.val[0]) (ABCD.val[1]) ∧ E ∈ Line (ABCD.val[2]) (ABCD.val[3]))
        (hF : F ∈ Line (ABCD.val[0]) (ABCD.val[3]) ∧ F ∈ Line (ABCD.val[1]) (ABCD.val[2]))
        (hX : X = midpoint (ABCD.val[0]) (ABCD.val[2]))
        (hY : Y = midpoint (ABCD.val[1]) (ABCD.val[3]))
        (hZ : Z = midpoint E F) : collinear X Y Z := sorry

end midpoints_collinear_l474_474708


namespace combination_8_5_l474_474623

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474623


namespace triangle_side_solution_l474_474255

theorem triangle_side_solution
  (b : ℝ) (angle_A : ℝ) (angle_C : ℝ)
  (h_b : b = real.sqrt 3)
  (h_angle_A : angle_A = real.pi / 4)
  (h_angle_C : angle_C = 5 * real.pi / 12) :
  ∃ a : ℝ, a = real.sqrt 2 :=
by
  sorry

end triangle_side_solution_l474_474255


namespace determine_coin_weights_l474_474867

-- Let’s define the types and the conditions formally in Lean 4

-- Define the coins for Alexander and Boris
inductive Coin : Type
| A1 | A2 | A3 | B1 | B2 | B3

-- Define the weights of the coins
def weight : Coin → ℕ
| Coin.A1 := 9
| Coin.A2 := 10
| Coin.A3 := 11
| Coin.B1 := 9
| Coin.B2 := 10
| Coin.B3 := 11

-- The main proof statement
theorem determine_coin_weights :
  (∃ f : (Coin → Coin → Prop),
    (f Coin.A1 Coin.B1 → 
     (f Coin.A2 Coin.B2 → 
      (f Coin.A3 Coin.B3 → true)) 
     ∨ (¬ f Coin.A2 Coin.B2 → true)) 
    ∨ (¬ f Coin.A1 Coin.B1 → 
       (f Coin.A3 Coin.B3 → true) 
       ∨ (¬ f Coin.A3 Coin.B3 → true))
  ) → ∀ c : Coin, (weight c = 9 ∨ weight c = 10 ∨ weight c = 11) :=
sorry

end determine_coin_weights_l474_474867


namespace percentage_error_in_measuring_side_l474_474447

-- Definitions based on the problem's conditions
def actual_side (x : ℝ) := x
def measured_side (x e : ℝ) := x + e
def actual_area (x : ℝ) := x^2
def measured_area (x e : ℝ) := (x + e)^2
def area_error (x : ℝ) := 0.0404 * x^2
def percentage_error_side (x e : ℝ) := (e / x) * 100

-- The theorem to be proven
theorem percentage_error_in_measuring_side (x e : ℝ)
  (h1 : measured_area x e = actual_area x + area_error x)
  (approx_small_error : e^2 ≈ 0) :
  percentage_error_side x e = 2.02 :=
by
  sorry

end percentage_error_in_measuring_side_l474_474447


namespace hovse_number_count_l474_474798

def valid_hovse_number (n : ℕ) : Prop :=
  ∀ x, x ≠ 0 → x < n → (∃ k, k < n ∧ (n = 3 * k + 1 ∨ n = 9 * k + 1 ∨ n = 27 * k + 3 ∨ k = ⌊n / 3⌋))

theorem hovse_number_count : (finset.range 2018).filter (λ n, valid_hovse_number n).card = 127 :=
sorry

end hovse_number_count_l474_474798


namespace smallest_positive_period_func1_smallest_positive_period_func2_l474_474138

-- Definition 1 for y = cos x + sin x
def func1 (x : Real) : Real := cos x + sin x

-- Theorem: The smallest positive period of func1 is None since the function is constant
theorem smallest_positive_period_func1 : 
    ∀ (T > 0), ¬ ∃ (T > 0), ∀ (x : Real), func1 (x + T) = func1 x := 
by
    sorry

-- Definition 2 for y = 2 cos x sin (x + π/3) - √3 sin^2 x + sin x cos x
def func2 (x : Real) : Real := 2 * cos x * sin (x + π/3) - sqrt 3 * sin x * sin x + sin x * cos x

-- Theorem: The smallest positive period of func2 is π
theorem smallest_positive_period_func2 : 
    ∃ (T > 0), (T = π) ∧ ∀ (x : Real), func2 (x + T) = func2 x := 
by
    sorry

end smallest_positive_period_func1_smallest_positive_period_func2_l474_474138


namespace total_area_of_sheet_l474_474843

theorem total_area_of_sheet (A B : ℝ) (h1 : A = 4 * B) (h2 : A = B + 2208) : A + B = 3680 :=
by
  sorry

end total_area_of_sheet_l474_474843


namespace intersection_correct_l474_474488

variable {x y : ℝ}

def A := {y | ∃ x, y = -x^2 + 2 * x + 2}
def B := {y | ∃ x, y = 2^x - 1}
def A_inter_B := {y | -1 < y ∧ y ≤ 3}

theorem intersection_correct :
  A ∩ B = A_inter_B := 
by 
  sorry

end intersection_correct_l474_474488


namespace max_sum_seq_l474_474269

theorem max_sum_seq (a : ℕ → ℝ) (h1 : a 1 = 0)
  (h2 : abs (a 2) = abs (a 1 - 1)) 
  (h3 : abs (a 3) = abs (a 2 - 1)) 
  (h4 : abs (a 4) = abs (a 3 - 1)) 
  : ∃ M, (∀ (b : ℕ → ℝ), b 1 = 0 → abs (b 2) = abs (b 1 - 1) → abs (b 3) = abs (b 2 - 1) → abs (b 4) = abs (b 3 - 1) → (b 1 + b 2 + b 3 + b 4) ≤ M) 
    ∧ (a 1 + a 2 + a 3 + a 4 = M) :=
  sorry

end max_sum_seq_l474_474269


namespace inequality_solution_set_no_positive_a_b_exists_l474_474519

def f (x : ℝ) := abs (2 * x - 1) - abs (2 * x - 2)
def k := 1

theorem inequality_solution_set :
  { x : ℝ | f x ≥ x } = { x : ℝ | x ≤ -1 ∨ x = 1 } :=
sorry

theorem no_positive_a_b_exists (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ¬ (a + 2 * b = k ∧ 2 / a + 1 / b = 4 - 1 / (a * b)) :=
sorry

end inequality_solution_set_no_positive_a_b_exists_l474_474519


namespace eval_256_pow_5_over_4_l474_474894

theorem eval_256_pow_5_over_4 : 256 = 2^8 → 256^(5/4 : ℝ) = 1024 :=
by
  intro h
  sorry

end eval_256_pow_5_over_4_l474_474894


namespace percentage_of_rejected_products_is_0_75_percent_l474_474663

-- Definitions based on the given conditions
def total_products (P : ℝ) := P
def percentage_john_rejected := 0.005
def percentage_jane_rejected := 0.009
def proportion_jane_inspected := 0.625

-- The theorem to be proved
theorem percentage_of_rejected_products_is_0_75_percent (P : ℝ) (h1 : P > 0) :
  ((proportion_jane_inspected * percentage_jane_rejected * P + (1 - proportion_jane_inspected) * percentage_john_rejected * P) / P) * 100 = 0.75 :=
by
  sorry

end percentage_of_rejected_products_is_0_75_percent_l474_474663


namespace solution_set_of_fx_lt0_l474_474879

theorem solution_set_of_fx_lt0 (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) → -- odd function
  (∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2) → -- increasing on (0, ∞)
  f 1 = 0 → -- f(1) = 0
  {x | f x < 0} = {x | x < -1 ∨ (0 < x ∧ x < 1)} :=
begin
  intros h_odd h_incr h_f1_zero,
  sorry
end

end solution_set_of_fx_lt0_l474_474879


namespace find_equidistant_point_l474_474911

theorem find_equidistant_point :
  ∃ (x z : ℝ),
    ((x - 1)^2 + 4^2 + z^2 = (x - 2)^2 + 2^2 + (z - 3)^2) ∧
    ((x - 1)^2 + 4^2 + z^2 = (x - 3)^2 + 9 + (z + 2)^2) ∧
    (x + 2 * z = 5) ∧
    (x = 15 / 8) ∧
    (z = 5 / 8) :=
by
  sorry

end find_equidistant_point_l474_474911


namespace a_general_T_n_l474_474749

def S (n : ℕ) : ℕ := ∑ k in finset.range (n + 1), a k
def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * S n + 1

theorem a_general (n : ℕ) : a n = 3 ^ (n-1)
by sorry

def b (n : ℕ) := 5 + (n-1) * 2
def T (n : ℕ) : ℕ := ∑ k in finset.range (n + 1), b k

theorem T_n (n : ℕ) : T n = n^2 + 2*n
by sorry

end a_general_T_n_l474_474749


namespace inequality_proof_l474_474185
open Nat

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 1) (h4 : n > 0) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end inequality_proof_l474_474185


namespace circle_equation_l474_474131

theorem circle_equation (a : ℝ) (h0 : a = (√7)/7 ∨ a = -(√7)/7) :
  (∀ x y : ℝ, (x - a)^2 + (y - 3*a)^2 = 9/7 ∨ (x + a)^2 + (y + 3*a)^2 = 9/7) :=
by sorry

end circle_equation_l474_474131


namespace range_of_x_l474_474186

theorem range_of_x (m : ℝ) (hm : m ≠ 0) (x : ℝ) 
  (h : ∀ m ≠ 0, |2 * m - 1| + |1 - m| ≥ |m| * (|x - 1| - |2 * x + 3|)) : 
  x ∈ Iic (-3) ∪ Ici (-1) := 
sorry

end range_of_x_l474_474186


namespace no_real_solution_f_eq_f_neg_l474_474731

def f : ℝ → ℝ
noncomputable def g (x : ℝ) := - (1/x^3 + 3/x)
axiom functional_equation (x : ℝ) : x ≠ 0 → f x + 2 * f (1/x) = x^3 + 3*x

theorem no_real_solution_f_eq_f_neg :
  ¬ ∃ x : ℝ, f x = f (-x) ∧ x ≠ 0 ∧ f x + 2 * f (1/x) = x^3 + 3*x :=
by
  sorry

end no_real_solution_f_eq_f_neg_l474_474731


namespace sum_of_valid_N_l474_474073

noncomputable def machine_transform : ℕ → ℕ
| n := if n % 2 = 0 then n / 2 else 4 * n + 1

def process (n : ℕ) : ℕ := 
Nat.iterate 6 machine_transform n

theorem sum_of_valid_N :
  (Finset.sum (Finset.filter (λ n, process n = 1) (Finset.range 100)) id) = 85 :=
sorry

end sum_of_valid_N_l474_474073


namespace intersection_M_N_l474_474532

def M := {p : ℝ × ℝ | p.snd = 2 - p.fst}
def N := {p : ℝ × ℝ | p.fst - p.snd = 4}
def intersection := {p : ℝ × ℝ | p = (3, -1)}

theorem intersection_M_N : M ∩ N = intersection := 
by sorry

end intersection_M_N_l474_474532


namespace relationship_among_abc_l474_474944

noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := Real.log 4
noncomputable def c : ℝ := (1/3 : ℝ) ^ 0.2

theorem relationship_among_abc (a : ℝ) (b : ℝ) (c : ℝ) :
  a = Real.log 3 / Real.log (1/2) →
  b = Real.log 4 →
  c = (1/3 : ℝ) ^ 0.2 →
  a < c ∧ c < b :=
by {
  intro h1 h2 h3,
  rw [h1, h2, h3],
  sorry
}

end relationship_among_abc_l474_474944


namespace B_complete_time_l474_474099

-- Define the conditions
def less_work_by_A (work_B: ℝ) : ℝ := 0.8 * work_B
def time_taken_by_A := 15 / 2

-- Define the total work done by A
def total_work (work_B: ℝ) : ℝ := (less_work_by_A work_B) * time_taken_by_A

-- Define the time taken by B to complete the same total work
def time_taken_by_B (work_B: ℝ) : ℝ := total_work work_B / work_B

-- State the theorem
theorem B_complete_time (work_B: ℝ) : time_taken_by_B work_B = 6 :=
by
  sorry

end B_complete_time_l474_474099


namespace number_of_integer_b_count_number_of_integer_b_l474_474464

theorem number_of_integer_b (b : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ x1^2 + b * x1 + 10 ≤ 0 ∧ x2^2 + b * x2 + 10 ≤ 0 ∧
  ∀ x : ℤ, (x1 ≠ x ∧ x2 ≠ x) → ¬ (x^2 + b * x + 10 ≤ 0)) ↔ b = -7 ∨ b = -6 ∨ b = 6 ∨ b = 7 :=
sorry

theorem count_number_of_integer_b :
  (finset.card {b : ℤ | (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ x1^2 + b * x1 + 10 ≤ 0 ∧ x2^2 + b * x2 + 10 ≤ 0 ∧
    ∀ x : ℤ, (x1 ≠ x ∧ x2 ≠ x) → ¬ (x^2 + b * x + 10 ≤ 0))}) = 4 :=
sorry

end number_of_integer_b_count_number_of_integer_b_l474_474464


namespace number_of_girls_l474_474325

theorem number_of_girls (total_children boys : ℕ) (h1 : total_children = 60) (h2 : boys = 19) :
  total_children - boys = 41 :=
by
  rw [h1, h2]
  norm_num

end number_of_girls_l474_474325


namespace total_profit_correct_l474_474698

def natasha_money : ℤ := 60
def carla_money : ℤ := natasha_money / 3
def cosima_money : ℤ := carla_money / 2
def sergio_money : ℤ := (3 * cosima_money) / 2

def natasha_items : ℤ := 4
def carla_items : ℤ := 6
def cosima_items : ℤ := 5
def sergio_items : ℤ := 3

def natasha_profit_margin : ℚ := 0.10
def carla_profit_margin : ℚ := 0.15
def cosima_sergio_profit_margin : ℚ := 0.12

def natasha_item_cost : ℚ := (natasha_money : ℚ) / natasha_items
def carla_item_cost : ℚ := (carla_money : ℚ) / carla_items
def cosima_item_cost : ℚ := (cosima_money : ℚ) / cosima_items
def sergio_item_cost : ℚ := (sergio_money : ℚ) / sergio_items

def natasha_profit : ℚ := natasha_items * natasha_item_cost * natasha_profit_margin
def carla_profit : ℚ := carla_items * carla_item_cost * carla_profit_margin
def cosima_profit : ℚ := cosima_items * cosima_item_cost * cosima_sergio_profit_margin
def sergio_profit : ℚ := sergio_items * sergio_item_cost * cosima_sergio_profit_margin

def total_profit : ℚ := natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_correct : total_profit = 11.99 := 
by sorry

end total_profit_correct_l474_474698


namespace math_problem_statement_l474_474677

def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2021 }
def F := { f : ℕ → ℕ | ∀ s, s ∈ S → f s ∈ S }

def T_f (f : ℕ → ℕ) : Set ℕ := { f_2021 : ℕ | ∃ s ∈ S, (Nat.iterate f 2021 s) = f_2021 }

noncomputable def T_f_size (f : ℕ → ℕ) : ℕ := (T_f f).card

noncomputable def sum_T_f_size_mod_2017 : ℕ :=
  ∑ f in F, (T_f_size f) % 2017

theorem math_problem_statement : sum_T_f_size_mod_2017 = 255 :=
  sorry

end math_problem_statement_l474_474677


namespace max_different_sums_is_7_l474_474423

def coin : Type :=
| penny
| nickel
| dime
| quarter

def value : coin → ℕ
| coin.penny := 1
| coin.nickel := 5
| coin.dime := 10
| coin.quarter := 25

def all_pairs (l : list coin) : list (coin × coin) :=
  list.bind l (λ a, list.map (λ b, (a, b)) l)

def possible_sums (pairs : list (coin × coin)) : list ℕ :=
  list.map (λ p, value p.1 + value p.2) pairs

def maximum_number_of_different_sums (coins : list coin) : ℕ :=
  list.length (list.erase_dup (possible_sums (all_pairs coins)))

theorem max_different_sums_is_7 :
  maximum_number_of_different_sums [coin.penny, coin.penny, coin.penny, coin.nickel, coin.dime, coin.quarter] = 7 :=
by
  sorry

end max_different_sums_is_7_l474_474423


namespace combination_eight_choose_five_l474_474629

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474629


namespace f_value_2005_l474_474065

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_2005 (f_even : ∀ x : ℝ, f(-x) = f(x))
    (f_periodicity : ∀ x : ℝ, f(x + 8) = f(x) + f(4))
    (f_definition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f(x) = 4 - x) :
  f(2005) = 1 := sorry

end f_value_2005_l474_474065


namespace min_distance_sum_l474_474167

open Real EuclideanGeometry

-- Define the parabola y^2 = 4x
noncomputable def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1 

-- Define the fixed point M
def M : ℝ × ℝ := (2, 3)

-- Define the line l: x = -1
def line_l (P : ℝ × ℝ) : ℝ := abs (P.1 + 1)

-- Define the distance from point P to point M
def distance_to_M (P : ℝ × ℝ) : ℝ := dist P M

-- Define the distance from point P to line l
def distance_to_line (P : ℝ × ℝ) := line_l P 

-- Define the sum of distances
def sum_of_distances (P : ℝ × ℝ) : ℝ := distance_to_M P + distance_to_line P

-- Prove the minimum value of the sum of distances
theorem min_distance_sum : ∃ P, parabola P ∧ sum_of_distances P = sqrt 10 := sorry

end min_distance_sum_l474_474167


namespace functional_equation_solution_l474_474411

noncomputable def func_form (f : ℝ → ℝ) : Prop :=
  ∃ α β : ℝ, (α = 1 ∨ α = -1 ∨ α = 0) ∧ (∀ x, f x = α * x + β ∨ f x = α * x ^ 3 + β)

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a * b ^ 2 + b * c ^ 2 + c * a ^ 2) - f (a ^ 2 * b + b ^ 2 * c + c ^ 2 * a)) →
  func_form f :=
sorry

end functional_equation_solution_l474_474411


namespace max_g_equals_sqrt3_l474_474963

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.pi / 9) + Real.sin (5 * Real.pi / 9 - x)

noncomputable def g (x : ℝ) : ℝ :=
  f (f x)

theorem max_g_equals_sqrt3 : ∀ x, g x ≤ Real.sqrt 3 :=
by
  sorry

end max_g_equals_sqrt3_l474_474963


namespace greatest_integer_gcd_6_l474_474766

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l474_474766


namespace rain_in_both_areas_l474_474584

variable (P1 P2 : ℝ)
variable (hP1 : 0 < P1 ∧ P1 < 1)
variable (hP2 : 0 < P2 ∧ P2 < 1)

theorem rain_in_both_areas :
  ∀ P1 P2, (0 < P1 ∧ P1 < 1) → (0 < P2 ∧ P2 < 1) → (1 - P1) * (1 - P2) = (1 - P1) * (1 - P2) :=
by
  intros P1 P2 hP1 hP2
  sorry

end rain_in_both_areas_l474_474584


namespace kangaroo_fraction_sum_l474_474380

theorem kangaroo_fraction_sum (G P : ℕ) (hG : 1 ≤ G) (hP : 1 ≤ P) (hTotal : G + P = 2016) : 
  (G * (P / G) + P * (G / P) = 2016) :=
by
  sorry

end kangaroo_fraction_sum_l474_474380


namespace angle_A_value_side_a_value_l474_474187

-- Define norm-num extension to handle real and trigonometric functions.
noncomputable def sin := Real.sin
noncomputable def cos := Real.cos
noncomputable def sqrt := Real.sqrt
noncomputable def pi := Real.pi

-- Define triangle conditions as hypotheses
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ) -- S represents the area of the triangle

-- Define the conditions
axiom h1 : (sin B - sin C)^2 = sin^2 A - sin B * sin C
axiom h2 : a / sin A = b / sin B
axiom h3 : b / sin B = c / sin C
axiom h4 : b + c = 4
axiom h5 : S = (sqrt 3) / 2

-- Define the theorems to be proved

-- Part 1: Prove the value of angle A
theorem angle_A_value : A = pi / 3 := by sorry

-- Part 2: Prove the value of side a
theorem side_a_value : a = sqrt 10 := by sorry

end angle_A_value_side_a_value_l474_474187


namespace Ilya_defeats_dragon_l474_474243

-- Conditions
def prob_two_heads : ℚ := 1 / 4
def prob_one_head : ℚ := 1 / 3
def prob_no_heads : ℚ := 5 / 12

-- Main statement in Lean
theorem Ilya_defeats_dragon : 
  (prob_no_heads + prob_one_head + prob_two_heads = 1) → 
  (∀ n : ℕ, ∃ m : ℕ, m ≤ n) → 
  (∑ n, (prob_no_heads + prob_one_head + prob_two_heads) ^ n) = 1 := 
sorry

end Ilya_defeats_dragon_l474_474243


namespace exists_perpendicular_line_in_plane_l474_474937

noncomputable theory

-- Defining the Plane and Line in space
structure Plane :=
  (point_set : set (ℝ × ℝ × ℝ))
  (in_plane : ∃ (p1 p2 : ℝ × ℝ × ℝ), p1 ≠ p2 ∧ (∀ (p3 : ℝ × ℝ × ℝ), p3 ∈ point_set → coplanar p1 p2 p3))

structure Line :=
  (point_set : set (ℝ × ℝ × ℝ))
  (in_line : ∃ (p1 p2 : ℝ × ℝ × ℝ), p1 ≠ p2 ∧ (∀ (p3 : ℝ × ℝ × ℝ), p3 ∈ point_set → collinear p1 p2 p3))

def coplanar (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), a * (p1.1) + b * (p1.2) + c * (p1.3) = d ∧ a * (p2.1) + b * (p2.2) + c * (p2.3) = d ∧ a * (p3.1) + b * (p3.2) + c * (p3.3) = d

def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), (p1.1 - p2.1) * a + (p1.2 - p2.2) * b + (p1.3 - p2.3) * c = 0 ∧ (p2.1 - p3.1) * a + (p2.2 - p3.2) * b + (p2.3 - p3.3) * c = 0

def perpendicular (ℓ₁ ℓ₂ : Line) : Prop :=
  ∃ (p1 p2 p3 : ℝ × ℝ × ℝ), p1 ≠ p2 ∧ p1 ∈ ℓ₁.point_set ∧ p2 ∈ ℓ₁.point_set ∧ 
                              p3 ≠ p2 ∧ p3 ∈ ℓ₂.point_set ∧
                              (p1.1 - p2.1) * (p2.1 - p3.1) + (p1.2 - p2.2) * (p2.2 - p3.2) + (p1.3 - p2.3) * (p2.3 - p3.3) = 0

theorem exists_perpendicular_line_in_plane (α : Plane) (ℓ : Line) : 
  ∃ b : Line, (b.point_set ⊆ α.point_set) ∧ perpendicular ℓ b :=
by
  sorry

end exists_perpendicular_line_in_plane_l474_474937


namespace ilya_defeats_dragon_l474_474242

section DragonAndIlya

def Probability (n : ℕ) : Type := ℝ

noncomputable def probability_of_defeat : Probability 3 :=
  let p_no_regrow := 5 / 12
  let p_one_regrow := 1 / 3
  let p_two_regrow := 1 / 4
  -- Assuming recursive relationship develops to eventually reduce heads to zero
  1

-- Prove that the probability_of_defeat is equal to 1
theorem ilya_defeats_dragon : probability_of_defeat = 1 :=
by
  -- Formal proof would be provided here
  sorry

end DragonAndIlya

end ilya_defeats_dragon_l474_474242


namespace range_of_m_l474_474223

theorem range_of_m (m : ℝ) (h : sqrt (2 * m + 1) > sqrt (m^2 + m - 1)) : m ∈ Ico ((sqrt 5 - 1) / 2) 2 :=
by
  sorry

end range_of_m_l474_474223


namespace range_of_a_l474_474195

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 + a * x
noncomputable def g (x : ℝ) : ℝ := 1 / (Real.exp x)

def f_deriv (a x : ℝ) : ℝ := x^2 + 2 * x + a

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (1/2 : ℝ) 2, ∃ x2 ∈ set.Icc (1/2) 2, f_deriv a x1 ≤ g x2) →
  a ≤ (Real.sqrt Real.exp 1) / Real.exp 1 - 8 :=
sorry

end range_of_a_l474_474195


namespace maxwell_walking_speed_l474_474694

theorem maxwell_walking_speed (distance_between_homes : ℕ)
  (brad_speed : ℕ)
  (maxwell_distance_traveled : ℕ)
  (meet_in_middle : 2 * maxwell_distance_traveled = distance_between_homes)
  : maxwell_distance_traveled * brad_speed / (distance_between_homes - maxwell_distance_traveled) = 24 := 
by
  have h1 : distance_between_homes = 72 := rfl
  have h2 : brad_speed = 12 := rfl
  have h3 : maxwell_distance_traveled = 24 := rfl
  sorry

end maxwell_walking_speed_l474_474694


namespace train_speed_l474_474433

-- Define conditions
def length_train : ℝ := 310 -- length of the train in meters
def length_bridge : ℝ := 140 -- length of the bridge in meters
def time_to_pass : ℝ := 36 -- time to pass the bridge in seconds

-- Define the total distance using the given lengths
def total_distance : ℝ := length_train + length_bridge

-- Define speed in meters per second
def speed_mps : ℝ := total_distance / time_to_pass

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph (speed : ℝ) : ℝ := speed * 3.6

-- Calculate the speed in kilometers per hour
def speed_kph : ℝ := mps_to_kph speed_mps

-- Theorem: The speed of the train in km/h is 45
theorem train_speed : speed_kph = 45 := 
sorry

end train_speed_l474_474433


namespace sum_of_m_n_l474_474029

open Classical

noncomputable theory

variable {A B C D E: ℝ}

def right_angle (a b c : ℝ) : Prop := a*a + b*b = c*c
def relatively_prime (m n : ℕ) : Prop := nat.gcd m n = 1

variables {AC BC AD DE DB : ℝ}
variable (h₁ : right_angle AC BC (AC + BC))
variable (h₂ : AC = 3)
variable (h₃ : BC = 4)
variable (h₄ : right_angle AD (AC + BC) (AD + AC + BC))
variable (h₅ : AD = 12)
variable (h₆ : DE / DB = 63 / 65)

theorem sum_of_m_n :
  ∃ m n : ℕ, relatively_prime m n ∧ DE / DB = (m : ℝ) / (n : ℝ) ∧ m + n = 128 := 
by {
  sorry
}

end sum_of_m_n_l474_474029


namespace quadratic_function_coefficients_l474_474938

theorem quadratic_function_coefficients :
  ∃ a b c : ℝ, (∀ x : ℝ, y = a * x^2 + b * x + c) ∧
  (y = a * (2:ℝ)^2 + b * (2:ℝ) + c = -1) ∧
  (y = c = 11) ∧
  (a = 3) ∧ (b = -12) ∧ (c = 11) := by
{
  sorry
}

end quadratic_function_coefficients_l474_474938


namespace speed_of_stream_l474_474033

theorem speed_of_stream
  (v_a v_s : ℝ)
  (h1 : v_a - v_s = 4)
  (h2 : v_a + v_s = 6) :
  v_s = 1 :=
by {
  sorry
}

end speed_of_stream_l474_474033


namespace y_coordinate_of_P_l474_474348

theorem y_coordinate_of_P (x y : ℝ) (h1 : |y| = 1/2 * |x|) (h2 : |x| = 12) :
  y = 6 ∨ y = -6 :=
sorry

end y_coordinate_of_P_l474_474348


namespace amount_to_add_l474_474415

theorem amount_to_add (students : ℕ) (total_cost : ℕ) (h1 : students = 9) (h2 : total_cost = 143) : 
  ∃ k : ℕ, total_cost + k = students * (total_cost / students + 1) :=
by
  sorry

end amount_to_add_l474_474415


namespace combination_eight_choose_five_l474_474628

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474628


namespace trains_at_starting_positions_after_2016_minutes_l474_474736

-- Definitions corresponding to conditions
def round_trip_minutes (line: String) : Nat :=
  if line = "red" then 14
  else if line = "blue" then 16
  else if line = "green" then 18
  else 0

def is_multiple_of (n m : Nat) : Prop :=
  n % m = 0

-- Formalize the statement to be proven
theorem trains_at_starting_positions_after_2016_minutes :
  ∀ (line: String), 
  line = "red" ∨ line = "blue" ∨ line = "green" →
  is_multiple_of 2016 (round_trip_minutes line) :=
by
  intro line h
  cases h with
  | inl red =>
    sorry
  | inr hb =>
    cases hb with
    | inl blue =>
      sorry
    | inr green =>
      sorry

end trains_at_starting_positions_after_2016_minutes_l474_474736


namespace train_crossing_time_l474_474981

variable (length1 length2 : ℕ)
variable (speed1_kmph speed2_kmph speed1_mps speed2_mps relative_speed total_length time_cross : ℕ)

def kmph_to_mps (kmph : ℕ) : ℕ := (kmph * 1000) / 3600

def relative_speed (speed1_mps speed2_mps : ℕ) : ℕ := speed1_mps - speed2_mps

def total_length (length1 length2 : ℕ) : ℕ := length1 + length2

def time_to_cross (total_length relative_speed : ℕ) : ℕ := total_length / relative_speed

theorem train_crossing_time :
  length1 = 420 →
  speed1_kmph = 72 →
  length2 = 640 →
  speed2_kmph = 36 →
  speed1_mps = kmph_to_mps speed1_kmph →
  speed2_mps = kmph_to_mps speed2_kmph →
  relative_speed speed1_mps speed2_mps = 10 →
  total_length length1 length2 = 1060 →
  time_to_cross (total_length length1 length2) (relative_speed speed1_mps speed2_mps) = 106 :=
by
  sorry

end train_crossing_time_l474_474981


namespace circle_equation_origin_radius_l474_474878

theorem circle_equation_origin_radius (r : ℝ) : ∃ x y : ℝ, (x - 0)^2 + (y - 0)^2 = r^2 :=
by
  simp
  use (λ x y : ℝ, x^2 + y^2 = r^2)
  sorry

end circle_equation_origin_radius_l474_474878


namespace number_of_max_area_triangles_is_16_l474_474678

def is_lattice_point_in_circle (x y : ℤ) : Prop :=
  x^2 + y^2 ≤ 11

def set_of_lattice_points_in_circle : set (ℤ × ℤ) :=
  {p | is_lattice_point_in_circle p.1 p.2}

noncomputable def area_of_triangle (p1 p2 p3 : ℤ × ℤ) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

noncomputable def max_triangle_area : ℝ := 12

def number_of_max_area_triangles : ℕ :=
  16

theorem number_of_max_area_triangles_is_16 :
  (∃ M : ℝ, M = max_triangle_area) → 
  (∃ count : ℕ, count = number_of_max_area_triangles ∧
    ∀ (p1 p2 p3 : ℤ × ℤ), 
      p1 ∈ set_of_lattice_points_in_circle ∧
      p2 ∈ set_of_lattice_points_in_circle ∧
      p3 ∈ set_of_lattice_points_in_circle ∧
      area_of_triangle p1 p2 p3 = max_triangle_area →
      count = 16) :=
sorry

end number_of_max_area_triangles_is_16_l474_474678


namespace combination_eight_choose_five_l474_474625

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l474_474625


namespace polar_eq_of_circle_range_OP_OQ_l474_474639

-- Define the parametric equations of the circle C
def circle_parametric (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the polar equation of the line l
def line_polar (θ : ℝ) : ℝ := sqrt(3) / (Real.sin θ + sqrt(3) * Real.cos θ)

-- Define the polar equation of the circle C to be proved
def polar_eq_circle (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Proof problem 1: Prove the polar equation of circle C
theorem polar_eq_of_circle (α θ : ℝ) (h1 : circle_parametric α = (2 + 2 * Real.cos α, 2 * Real.sin α))
  : polar_eq_circle θ = 4 * Real.cos θ := sorry

-- Define |OP| and |OQ| based on their definitions from the provided solution
def OP (θ : ℝ) : ℝ := 4 * Real.cos θ
def OQ (θ : ℝ) : ℝ := sqrt(3) / (Real.sin θ + sqrt(3) * Real.cos θ)

-- Proof problem 2: Prove the range of |OP| * |OQ|
theorem range_OP_OQ (θ : ℝ) (h1 : OP θ = 4 * Real.cos θ) (h2 : OQ θ = sqrt(3) / (Real.sin θ + sqrt(3) * Real.cos θ)) 
  (h3 : (Real.pi / 6) ≤ θ ∧ θ ≤ (Real.pi / 3)) 
  : 2 ≤ OP θ * OQ θ ∧ OP θ * OQ θ ≤ 3 := sorry

end polar_eq_of_circle_range_OP_OQ_l474_474639


namespace geometric_sequence_a4_l474_474368

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * q)
    (h_a2 : a 2 = 1)
    (h_q : q = 2) : 
    a 4 = 4 :=
by
  -- Skip the proof as instructed
  sorry

end geometric_sequence_a4_l474_474368


namespace find_c_value_l474_474728

theorem find_c_value (x c : ℝ) (h₁ : 3 * x + 8 = 5) (h₂ : c * x + 15 = 3) : c = 12 :=
by
  -- This is where the proof steps would go, but we will use sorry for now.
  sorry

end find_c_value_l474_474728


namespace circle_not_touch_diag_eq_prob_circle_not_touch_diag_sum_l474_474059

-- Step by step translation and formulation of the problem in Lean 4

def rectangle : Type := ℝ × ℝ
def circle : Type := {r : ℝ // r > 0}
def diagonal : rectangle → rectangle → ℝ := λ a b, (a.1 - b.1)^2 + (a.2 - b.2)^2 -- Euclidean distance
noncomputable def area (a : ℝ) (b : ℝ) : ℝ := a * b 
def probability_not_touch_diagonal (rect : rectangle) (r : ℝ) : ℚ := ⟨375, 884⟩

theorem circle_not_touch_diag_eq (rect : rectangle) (r : ℝ) (h1 : rect = (15, 36)) 
    (h2 : r = 1) : 
    probability_not_touch_diagonal rect r = ⟨375, 884⟩ :=
sorry

theorem prob_circle_not_touch_diag_sum (rect : rectangle) (r : ℝ) (h1 : rect = (15, 36)) 
    (h2 : r = 1) (h : probability_not_touch_diagonal rect r = ⟨375, 884⟩) :
    let m : ℚ := 375 
    let n : ℚ := 884 in 
    m + n = 1259 :=
by {
    intros,
    exact h,
    sorry
}

end circle_not_touch_diag_eq_prob_circle_not_touch_diag_sum_l474_474059


namespace arithmetic_progression_sum_15_terms_l474_474149

theorem arithmetic_progression_sum_15_terms (a₃ a₅ : ℝ) (h₁ : a₃ = -5) (h₂ : a₅ = 2.4) : 
  let d := (a₅ - a₃) / 2 in
  let a₁ := a₃ - 2 * d in
  (15 / 2) * (2 * a₁ + 14 * d) = 202.5 :=
by
  sorry

end arithmetic_progression_sum_15_terms_l474_474149


namespace part1_part2_l474_474500

-- Defining the functions f and g
def f (x : ℝ) : ℝ := Real.exp x - 1
def g (x : ℝ) : ℝ := Real.log (x + 1)

-- Part (1) proof statement
theorem part1 (k : ℝ) (h : ∀ x > 0, f x ≥ k * g x) : k ≤ 1 :=
sorry

-- Definitions for part (2)
structure Point (α : Type) := 
  (x : α) 
  (y : α)

def A (x1 : ℝ) (hx1 : x1 > 0) : Point ℝ := ⟨x1, f x1⟩
def B (x2 : ℝ) (hx2 : x2 > 0) : Point ℝ := ⟨x2, -g x2⟩

-- Part (2) proof statement
theorem part2 (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) 
  (acute_angle : (A x1 hx1).x * (B x2 hx2).x + (A x1 hx1).y * (B x2 hx2).y > 0) :
  x2 > x1^2 :=
sorry

end part1_part2_l474_474500


namespace calculate_run_rate_l474_474409

theorem calculate_run_rate (first_10_overs_rate : ℝ) (target_runs : ℝ) (total_overs : ℝ) (initial_overs : ℝ) :
  first_10_overs_rate = 4.8 ∧ target_runs = 282 ∧ total_overs = 50 ∧ initial_overs = 10 →
  (target_runs - first_10_overs_rate * initial_overs) / (total_overs - initial_overs) = 5.85 :=
begin
  sorry
end

end calculate_run_rate_l474_474409


namespace sum_a_b_l474_474567

-- Define the function g and the conditions on a and b
def g (x : ℝ) (a b : ℝ) : ℝ := (x + 5) / (x^2 + a * x + b)

-- We need to prove the sum of a and b given the vertical asymptotes
theorem sum_a_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, x ≠ 2 → (x^2 + a * x + b) ≠ 0) 
  (h2 : ∀ x : ℝ, x ≠ -3 → (x^2 + a * x + b) ≠ 0)
  (ha : x^2 + a * x + b = (x - 2) * (x + 3)) :
  a + b = -5 := 
sorry

end sum_a_b_l474_474567


namespace neg_p_l474_474202

theorem neg_p (p : ∀ x : ℝ, x^2 ≥ 0) : ∃ x : ℝ, x^2 < 0 := 
sorry

end neg_p_l474_474202


namespace power_of_product_l474_474450

variable (x y: ℝ)

theorem power_of_product :
  (-2 * x * y^3)^2 = 4 * x^2 * y^6 := 
by
  sorry

end power_of_product_l474_474450


namespace playground_fund_after_fees_l474_474016

-- Definitions based on the given conditions
def mrs_johnsons_class := 2300
def mrs_suttons_class := mrs_johnsons_class / 2
def miss_rollins_class := mrs_suttons_class * 8
def total_raised := miss_rollins_class * 3
def admin_fees := total_raised * 0.02
def playground_amount := total_raised - admin_fees

-- The theorem to be proved
theorem playground_fund_after_fees : playground_amount = 27048 := by
  sorry

end playground_fund_after_fees_l474_474016


namespace reflection_on_circumcircle_l474_474293

variable (A B C I M_A M_B M_C X : Type)
  [geometry (triangle A B C)]
  [incenter I (triangle A B C)]
  [midpoint M_A (minor_arc B C)]
  [midpoint M_B (minor_arc C A)]
  [midpoint M_C (minor_arc A B)]
  [reflection_point X (M_A reflection_over (line I M_B))]

theorem reflection_on_circumcircle :
  lies_on_circle X (circumcircle (triangle I M_B M_C)) :=
sorry

end reflection_on_circumcircle_l474_474293


namespace h_f_neg5_l474_474302

def f (x : ℝ) : ℝ := 4 * x^2 + 6
def g (y : ℝ) : ℝ := if y = 106 then 8 else 0  -- Using a conditional to match the specific given condition
def h (x : ℝ) : ℝ := g x + 2

theorem h_f_neg5 : h (f (-5)) = 10 :=
by
  have h₁ : f (-5) = 106 := by
    simp [f]
  have h₂ : g (f (-5)) = 8 := by
    simp [h₁, g]
  simp [h, h₂]
  exact rfl

end h_f_neg5_l474_474302


namespace find_x_values_l474_474128

-- Defining the given condition as a function
def equation (x : ℝ) : Prop :=
  (4 / (Real.sqrt (x + 5) - 7)) +
  (3 / (Real.sqrt (x + 5) - 2)) +
  (6 / (Real.sqrt (x + 5) + 2)) +
  (9 / (Real.sqrt (x + 5) + 7)) = 0

-- Statement of the theorem in Lean
theorem find_x_values :
  equation ( -796 / 169) ∨ equation (383 / 22) :=
sorry

end find_x_values_l474_474128


namespace number_of_values_a_l474_474118

-- Definitions of conditions
def condition₁ (a : ℝ) : Prop := -(1 - a)^2 ≥ 0

-- Problem statement to prove
theorem number_of_values_a (a : ℝ) :
  (∃! a : ℝ, condition₁ a) :=
begin
  -- Placeholder for proof
  sorry
end

end number_of_values_a_l474_474118


namespace defeat_dragon_probability_l474_474249

noncomputable theory

def p_two_heads_grow : ℝ := 1 / 4
def p_one_head_grows : ℝ := 1 / 3
def p_no_heads_grow : ℝ := 5 / 12

-- We state the probability that Ilya will eventually defeat the dragon
theorem defeat_dragon_probability : 
  ∀ (expected_value : ℝ), 
  (expected_value = p_two_heads_grow * 2 + p_one_head_grows * 1 + p_no_heads_grow * 0) →
  expected_value < 1 →
  prob_defeat (count_heads n : ℕ) > 0 :=
by
  sorry

end defeat_dragon_probability_l474_474249


namespace arcsin_cos_eq_cases_l474_474683

theorem arcsin_cos_eq_cases (x : ℝ) (h : -π ≤ x ∧ x ≤ π) : 
  arcsin (cos x) = 
    if -π ≤ x ∧ x ≤ 0 then 
      x + π / 2 
    else 
      π / 2 - x := 
by
  sorry

end arcsin_cos_eq_cases_l474_474683


namespace composition_of_groups_l474_474098

noncomputable def problem_conditions 
  (n : ℕ) 
  (boys girls : ℕ) 
  (total_handshakes : ℕ) 
  (groupA groupB : ℕ) 
  (boy_pairs girl_pairs : ℕ) 
  (total_boy_girl_pairs : ℕ)
  (diff_village_pairs : ℕ) 
  (same_village_pairs : ℕ)
  (boy_dancers_A girl_dancers_A boy_escorts_A girl_escorts_A : ℕ)
  (boy_dancers_B girl_dancers_B boy_escorts_B girl_escorts_B : ℕ) :=
  n = 44 ∧
  boys = girls + 1 ∧ 
  total_handshakes = 946 ∧
  groupA > groupB ∧
  (groupA, groupB) = (23, 21) ∧
  total_boy_girl_pairs = 484 ∧
  diff_village_pairs = 246∧
  same_village_pairs = 150 ∧
  boy_dancers_A = 8 ∧
  girl_dancers_A = 10 ∧
  boy_escorts_A = 2 ∧
  girl_escorts_A = 3 ∧
  boy_dancers_B = 10 ∧
  girl_dancers_B = 7  ∧
  boy_escorts_B = 2  ∧
  girl_escorts_B = 2

theorem composition_of_groups :
  problem_conditions 44 22 22 946 23 21 484 246 150 8 10 2 3 10 7 2 2 := by 
  sorry

end composition_of_groups_l474_474098


namespace jessica_needs_8_bottles_l474_474655

theorem jessica_needs_8_bottles (fl_oz_needed : ℝ) (mL_per_bottle : ℝ) (fl_oz_per_L : ℝ)
  (h1 : fl_oz_needed = 60) (h2 : mL_per_bottle = 250) (h3 : fl_oz_per_L = 33.8) :
  let L_needed := fl_oz_needed / fl_oz_per_L
  let mL_needed := L_needed * 1000
  let bottles_needed := mL_needed / mL_per_bottle
  let rounded_bottles_needed := Real.ceil bottles_needed in
  rounded_bottles_needed = 8 :=
by
  sorry

end jessica_needs_8_bottles_l474_474655


namespace classroom_arrangement_exponent_sum_l474_474814

-- Each definition used in Lean 4 statement is directly related to the conditions in step a).

-- Definitions for the problem.
def numTables : Nat := 5
def numChairsPerTable : Nat := 4

-- Derangements count for each table
def derangements (n : Nat) : Nat := n! * (Finset.range n).sum fun i => (-1)^i / Nat.factorial i

-- Total number of different classroom arrangements
noncomputable def totalArrangements : Nat := 
  let arrangementPerTable := derangements numChairsPerTable
  let totalArrangements := numTables! * (numChairsPerTable!)^numTables * arrangementPerTable ^ numTables
  totalArrangements

-- This is to express totalArrangements in the form 2^a * 3^b * 5^c, and check the equality
theorem classroom_arrangement_exponent_sum (a b c : Nat) (h : totalArrangements = 2^a * 3^b * 5^c) :
  a + b + c = 35 :=
sorry

end classroom_arrangement_exponent_sum_l474_474814


namespace train_speed_is_correct_l474_474856

-- Definitions of the problem
def length_of_train : ℕ := 360
def time_to_pass_bridge : ℕ := 25
def length_of_bridge : ℕ := 140
def conversion_factor : ℝ := 3.6

-- Distance covered by the train plus the length of the bridge
def total_distance : ℕ := length_of_train + length_of_bridge

-- Speed calculation in m/s
def speed_in_m_per_s := total_distance / time_to_pass_bridge

-- Conversion to km/h
def speed_in_km_per_h := speed_in_m_per_s * conversion_factor

-- The proof goal: the speed of the train is 72 km/h
theorem train_speed_is_correct : speed_in_km_per_h = 72 := by
  sorry

end train_speed_is_correct_l474_474856


namespace constant_term_in_binomial_expansion_l474_474465

theorem constant_term_in_binomial_expansion : 
  let a := (λ (x : ℝ), sqrt x + (1 / x))
  let n := 9
  let term := λ r, (Nat.choose n r) * (sqrt x) ^ (n - r) * (1 / x) ^ r
  let binomial_expansion := Sum this over all r in {0..9}
  ∃ (c : ℕ), (c = Nat.choose 9 3) ∧ (c = 84) → ((∑ r in Finset.range(n + 1), term r) = 84) := 
begin
  sorry
end

end constant_term_in_binomial_expansion_l474_474465


namespace choose_five_from_eight_l474_474604

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l474_474604


namespace mean_median_difference_l474_474588

-- Definition of student score fractions and scores
def percent_65 := 0.20
def percent_75 := 0.25
def percent_85 := 0.15
def percent_95 := 0.30
def percent_105 := 1 - (percent_65 + percent_75 + percent_85 + percent_95) -- 0.10

def score_65 := 65
def score_75 := 75
def score_85 := 85
def score_95 := 95
def score_105 := 105

-- Median score value based on percentages
def median := score_95

-- Mean score calculation
noncomputable def mean := (percent_65 * score_65) + (percent_75 * score_75) + (percent_85 * score_85) + 
                          (percent_95 * score_95) + (percent_105 * score_105)

-- Difference between the median and mean score values
noncomputable def difference := median - mean

-- Proof statement
theorem mean_median_difference : difference = 11.5 := by
  sorry

end mean_median_difference_l474_474588


namespace arithmetic_expression_value_l474_474453

theorem arithmetic_expression_value :
  15 * 36 + 15 * 3^3 = 945 :=
by
  sorry

end arithmetic_expression_value_l474_474453


namespace hyperbola_eccentricity_l474_474952

-- Formalizing the conditions
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)

-- Main statement to prove
theorem hyperbola_eccentricity (h_eq_slope : b / a = sqrt 3) : 
  let e := sqrt (1 + (b / a)^2) in e = 2 := by
  sorry

end hyperbola_eccentricity_l474_474952


namespace carrots_planted_per_hour_l474_474547

theorem carrots_planted_per_hour (rows plants_per_row hours : ℕ) (h1 : rows = 400) (h2 : plants_per_row = 300) (h3 : hours = 20) :
  (rows * plants_per_row) / hours = 6000 := by
  sorry

end carrots_planted_per_hour_l474_474547


namespace sweatshirt_cost_l474_474542

/--
Hannah bought 3 sweatshirts and 2 T-shirts.
Each T-shirt cost $10.
Hannah spent $65 in total.
Prove that the cost of each sweatshirt is $15.
-/
theorem sweatshirt_cost (S : ℝ) (h1 : 3 * S + 2 * 10 = 65) : S = 15 :=
by
  sorry

end sweatshirt_cost_l474_474542


namespace no_solutions_theta_l474_474901

theorem no_solutions_theta :
  ∀ (θ : ℝ), 
  (3 * real.pi / 2 ≤ θ ∧ θ ≤ 2 * real.pi) →
  ¬(∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) →
    x^2 * real.cos (θ + real.pi / 4) - x * (1 - x) + (1 - x)^2 * real.sin (θ + real.pi / 4) > 0) :=
by
  intros θ hθ.
  unfold real.pi at hθ.
  sorry

end no_solutions_theta_l474_474901


namespace probability_red_condition_l474_474051

noncomputable def probability_other_side_red (total_cards red_sides total_red_sides) : ℝ :=
  if h : total_red_sides ≠ 0 then red_sides / total_red_sides else 0

theorem probability_red_condition
  (black_black_cards black_red_cards red_red_cards blue_blue_card : ℕ)
  (total_red_sides red_sides : ℕ)
  (hred_sides : red_sides = 4)
  (htotal_red_sides : total_red_sides = 7) :
  probability_other_side_red (black_black_cards + black_red_cards + red_red_cards + blue_blue_card) red_sides total_red_sides
  = 4 / 7 :=
by {
  dsimp [probability_other_side_red],
  rw [hred_sides, htotal_red_sides],
  exact if_pos (by norm_num)
}

end probability_red_condition_l474_474051


namespace cloth_sold_l474_474432

variable (totalSellingPrice : ℝ) (lossPerMetre : ℝ) (costPricePerMetre : ℝ)

theorem cloth_sold (totalSellingPrice = 18000) (lossPerMetre = 5) (costPricePerMetre = 41) : 
  (totalSellingPrice / (costPricePerMetre - lossPerMetre) = 500) :=
by
  sorry

end cloth_sold_l474_474432


namespace curve_is_circle_l474_474885

noncomputable def curve_eqn_polar (r θ : ℝ) : Prop :=
  r = 1 / (Real.sin θ + Real.cos θ)

theorem curve_is_circle : ∀ r θ, curve_eqn_polar r θ →
  ∃ x y : ℝ, r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ 
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by
  sorry

end curve_is_circle_l474_474885


namespace rectangle_equiv_l474_474940

variables {Point : Type} [metric_space Point]
variables (A B C D : Point)

def is_rectangle (A B C D : Point) : Prop :=
  dist A B = dist C D ∧ dist B C = dist D A ∧ 
  dist A C ^ 2 = dist A B ^ 2 + dist B C ^ 2

theorem rectangle_equiv (A B C D : Point) :
  (is_rectangle A B C D) ↔ 
  (∀ (X : Point), dist A X ^ 2 + dist C X ^ 2 = dist B X ^ 2 + dist D X ^ 2) :=
sorry

end rectangle_equiv_l474_474940


namespace B_completes_work_in_18_days_l474_474440

variable {A B : ℝ}
variable (x : ℝ)

-- Conditions provided
def A_works_twice_as_fast_as_B (h1 : A = 2 * B) : Prop := true
def together_finish_work_in_6_days (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : Prop := true

-- Theorem to prove: It takes B 18 days to complete the work independently
theorem B_completes_work_in_18_days (h1 : A = 2 * B) (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : x = 18 := by
  sorry

end B_completes_work_in_18_days_l474_474440


namespace det_matrix_A_eq_zero_l474_474124

variable (α β : ℝ)

def matrix_A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![Real.sin α * Real.cos β, -Real.sin α * Real.sin β, Real.cos α],
    ![Real.cos β, Real.sin β, 0],
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, Real.sin α]
  ]

theorem det_matrix_A_eq_zero : Matrix.det (matrix_A α β) = 0 :=
  sorry

end det_matrix_A_eq_zero_l474_474124


namespace quadratic_distinct_real_roots_l474_474578

theorem quadratic_distinct_real_roots (k : ℝ) : 
  (∀ (x : ℝ), (k - 1) * x^2 + 4 * x + 1 = 0 → False) ↔ (k < 5 ∧ k ≠ 1) :=
by
  sorry

end quadratic_distinct_real_roots_l474_474578


namespace f_even_and_monotonically_decreasing_l474_474732

noncomputable def f (x : ℝ) := Real.log (Real.abs x) / Real.log 2

theorem f_even_and_monotonically_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, x < y → y < 0 → f y < f x) :=
by
  sorry

end f_even_and_monotonically_decreasing_l474_474732


namespace problem_l474_474356

def r : ℝ
def s : ℝ
def x : ℝ

axiom condition1 : r ≤ 3*x + 4 ∧ 3*x + 4 ≤ s
axiom condition2 : (s - r) / 3 = 12

theorem problem : s - r = 36 := by
  sorry

end problem_l474_474356


namespace sum_A_p_l474_474682

section
variable (p : ℕ) (hp : p % 2 = 0) 

def A_p := {x : ℕ | 2^p < x ∧ x < 2^(p+1) ∧ ∃ m : ℕ, x = 3 * m}

theorem sum_A_p (p : ℕ) (hp : p % 2 = 0) : 
  ∑ x in (A_p p), x = 2^(2*p-1) - 2^(p-1) := sorry
end

end sum_A_p_l474_474682


namespace length_other_diagonal_l474_474846

variables (d1 d2 : ℝ) (Area : ℝ)

theorem length_other_diagonal 
  (h1 : Area = 432)
  (h2 : d1 = 36) :
  d2 = 24 :=
by
  -- Insert proof here
  sorry

end length_other_diagonal_l474_474846


namespace convex_combination_l474_474290

variables
  {X : Set (ℝ × ℝ)}
  {a1 a2 b1 b2 : ℝ}
  (h1 : ∀ ⦃x1 y1 x2 y2⦄, (x1, y1) ∈ X → (x2, y2) ∈ X → (x1, y1) ≠ (x2, y2) → (x1 > x2 ∧ y1 > y2) ∨ (x1 < x2 ∧ y1 < y2))
  (h2 : ∃ p1 p2, p1 ∈ X ∧ p2 ∈ X ∧ (a1, _root_.le) p1.1 a2 ∧ (a1, _root_.le) p2.1 a2 ∧ (b1, _root_.le) p1.2 b2 ∧ (b1, _root_.le) p2.2 b2)
  (h3 : ∀ ⦃x1 y1 x2 y2 λ⦄, (x1, y1) ∈ X → (x2, y2) ∈ X → λ ∈ Icc 0 1 → (λ * x1 + (1 - λ) * x2, λ * y1 + (1 - λ) * y2) ∈ X)

theorem convex_combination (x y : ℝ) (hx : (x, y) ∈ X) :
  ∃ λ ∈ Icc (0 : ℝ) 1, x = λ * a1 + (1 - λ) * a2 ∧ y = λ * b1 + (1 - λ) * b2 :=
sorry

end convex_combination_l474_474290


namespace gcd_a_b_l474_474298

-- Define a and b
def a : ℕ := 333333
def b : ℕ := 9999999

-- Prove that gcd(a, b) = 3
theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l474_474298


namespace factor_expression_1_factor_expression_2_l474_474334

theorem factor_expression_1 (a b c : ℝ) : a^2 + 2 * a * b + b^2 + a * c + b * c = (a + b) * (a + b + c) :=
  sorry

theorem factor_expression_2 (a x y : ℝ) : 4 * a^2 - x^2 + 4 * x * y - 4 * y^2 = (2 * a + x - 2 * y) * (2 * a - x + 2 * y) :=
  sorry

end factor_expression_1_factor_expression_2_l474_474334


namespace distance_sum_leq_eight_l474_474295

def is_on_curve (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  abs x / 4 + abs y / 3 = 1

def F1 : ℝ × ℝ := (-Real.sqrt 7, 0)
def F2 : ℝ × ℝ := (Real.sqrt 7, 0)

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem distance_sum_leq_eight (P : ℝ × ℝ) (h : is_on_curve P) :
  distance P F1 + distance P F2 ≤ 8 :=
sorry

end distance_sum_leq_eight_l474_474295


namespace problem_solution_l474_474289

noncomputable def f (x : ℝ) : ℝ :=
  (6 * x + 1) / (32 * x + 8) - (2 * x - 1) / (32 * x - 8)

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x ^ 2 + y ^ 2)

noncomputable def min_distance (f : ℝ → ℝ) : ℝ :=
  let distances := {d | ∃ x > 0, y > 0, y = f x ∧ d = distance x y}
  classical.some (Set.bdd_below.distances)

theorem problem_solution :
  let d := min_distance f in
  (⌊1000 * d⌋ = 433) :=
by
  sorry

end problem_solution_l474_474289


namespace polynomial_expansion_coefficient_l474_474130

theorem polynomial_expansion_coefficient :
  (polynomial.X + 3 * polynomial.X - 2 * polynomial.X ^ 2) ^ 5.coeff 8 = -720 :=
sorry

end polynomial_expansion_coefficient_l474_474130


namespace max_residents_in_block_l474_474264

theorem max_residents_in_block :
  let lower_section_floors := 4
  let middle_section_floors := 5
  let upper_section_floors := 6
  let lower_section_per_floor := (2 * 4) + (2 * 5) + (1 * 6)
  let middle_section_per_floor := (3 * 3) + (2 * 4) + (1 * 6)
  let upper_section_per_floor := (3 * 8) + (2 * 10)
  let lower_section_capacity := lower_section_floors * lower_section_per_floor
  let middle_section_capacity := middle_section_floors * middle_section_per_floor
  let upper_section_capacity := upper_section_floors * upper_section_per_floor
  in (lower_section_capacity + middle_section_capacity + upper_section_capacity) = 475 :=
by
  sorry

end max_residents_in_block_l474_474264


namespace avg_percent_of_10_students_l474_474226

theorem avg_percent_of_10_students:
  (∀ (n1 n2 n3 : ℕ) (a1 a3 : ℕ) (a2 : ℝ), 
    (n1 = 15) ∧ (n2 = 10) ∧ (n3 = 25) ∧ (a1 = 75) ∧ (a3 = 83) ∧
    (a1 * n1 + a2 * n2) / (n1 + n2) = a3 → a2 = 95) :=
begin
  intros,
  sorry
end

end avg_percent_of_10_students_l474_474226


namespace base_b_square_l474_474738

theorem base_b_square (b : ℕ) (h : b > 2) : ∃ k : ℕ, 121 = k ^ 2 :=
by
  sorry

end base_b_square_l474_474738


namespace find_slope_of_line_l474_474734

theorem find_slope_of_line
  (k : ℝ)
  (circle_eq : ∀ (x y : ℝ), (x - 2)^2 + y^2 = 4)
  (line_eq : ∀ (x y : ℝ) , y = k * x)
  (chord_length : ∀ (O A : ℝ × ℝ), |O.1 - A.1|^2 + |O.2 - A.2|^2 = (2 * sqrt 3)^2) :
  k = sqrt 3 / 3 ∨ k = - (sqrt 3 / 3) :=
by
  sorry

end find_slope_of_line_l474_474734


namespace proof_problem_l474_474927

variable (a b : ℝ)

theorem proof_problem (h1: a > 0) (h2: exp(a) + log(b) = 1) :
  a + log(b) < 0 ∧
  exp(a) + b > 2 ∧
  log(a) + exp(b) ≥ 0 ∧
  a + b > 1 :=
by {
  sorry
}

end proof_problem_l474_474927


namespace arccos_one_eq_zero_l474_474047

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l474_474047


namespace equation_of_circumcircle_equations_of_line_l_fixed_points_of_circle_N_l474_474538

-- Step 1: Define the given vertices and the circumcircle properties.
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (1, 3)
def circumcircle (a b c : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry  -- Define it with specific details.

-- Step 2: Prove the equation for the circumcircle M.
theorem equation_of_circumcircle : 
  ∃ D E F : ℝ, (circumcircle A B C).1 = (0, 0) ∧ (circumcircle A B C).2 = (D, E) ∧ (circumcircle A B C).3 = F ∧
  (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 6 * y + 8 = 0) := 
sorry

-- Step 3: Define point D.
def D : ℝ × ℝ := (1/2, 2)

-- Step 4: Prove the equations of line l that meet the given conditions.
theorem equations_of_line_l : 
  ∀ p : ℝ × ℝ, 
  (∀ (l : ℝ → ℝℝ), chord_length (circumcircle A B C) l = √3 → (l = (λ x, if x = 1/2 then 0 else 6x - 8y + 13)) → 
  p = (1/2, 2) ) :=
sorry

-- Step 5: Define points E, F, circle N, and prove it passes through fixed points.
theorem fixed_points_of_circle_N : 
  ∀ P : ℝ × ℝ, 
  on_circumcircle P (circumcircle A B C) → P ≠ A → P ≠ B → 
  ∃ E F : ℝ × ℝ, 
  (proj_x (line PA) = E.x) ∧ (proj_x (line PB) = F.x) ∧ 
  (∃ N : ℝ × ℝ × ℝ, (diam N).1 = E.x ∧ (diam N).2 = F.x ∧ 
  (N_passes_through (0, 2√2) N) ∧ (N_passes_through (0, -2√2) N)) :=
sorry

end equation_of_circumcircle_equations_of_line_l_fixed_points_of_circle_N_l474_474538


namespace distinct_lines_on_plane_l474_474496

theorem distinct_lines_on_plane (n : ℕ) (points : Fin n → ℝ × ℝ) (h_non_collinear : ¬(∀ (i j k : Fin n), 
  ∃ (a b c : ℤ), a * (points i).fst + b * (points i).snd + c = 0 ∧ 
  a * (points j).fst + b * (points j).snd + c = 0 ∧ 
  a * (points k).fst + b * (points k).snd + c = 0)) : 
  ∃ (lines : Fin (n * (n - 1) / 2) → Set (ℝ × ℝ)), set.finite lines ∧ lines.card ≥ n :=
sorry

end distinct_lines_on_plane_l474_474496


namespace points_concyclic_l474_474939

-- Define the setup for points and circles
variables {A B C D I J K X Y : Type*}
variables [incircle : Circle (Triangle A B C)] [Gamma : Circle]

-- Define assumptions and conditions
variables (tangent_Gamma_I : Tangent Gamma (incircle))
variables (passes_through_BC : PassesThrough Gamma B ∧ PassesThrough Gamma C)
variables (intersects_AB_X : Intersects Gamma (Line A B) X)
variables (intersects_AC_Y : Intersects Gamma (Line A C) Y)
variables (excenter_J : Excenter (Triangle A X Y) J)
variables (tangent_D : Tangent (incircle) (Line B C) D)

-- Main theorem statement
theorem points_concyclic : Concyclic D I J K :=
sorry

end points_concyclic_l474_474939


namespace compute_N_l474_474106

theorem compute_N : 
  (100^2 + 99^2 - 98^2 - 97^2 + 96^2 + ... + 4^2 + 3^2 - 2^2 - 1^2) = 10100 := 
by sorry

end compute_N_l474_474106


namespace no_integer_a_geq_1_exists_l474_474645

theorem no_integer_a_geq_1_exists
  : ∀ (a : ℤ), a ≥ 1 → ∀ (x : ℕ), ¬(nat.gcd (x^2 + 3) ((x + a)^2 + 3) = 1) := by
  sorry

end no_integer_a_geq_1_exists_l474_474645


namespace option_a_option_b_l474_474026

variable (a : ℝ)

/-- This statement proves that 8^(-2/3) = 1/4. -/
theorem option_a : 8^(-2/3 : ℝ) = 1 / 4 :=
by sorry

/-- This statement proves that (sqrt(-a^3)) / (-a) = sqrt(-a) when a < 0. -/
theorem option_b (h : a < 0) : (Real.sqrt (-a^3)) / (-a) = Real.sqrt (-a) :=
by sorry

end option_a_option_b_l474_474026


namespace tournament_prob_unique_wins_l474_474158

theorem tournament_prob_unique_wins :
  ∃ (m n : ℕ), (m.gcd n = 1) ∧ (2:ℕ)^990 * m = 2^41 * 45.factorial ∧ (nat.logs_of 2 n = 949) ∧ (2^41 * 45.factorial) / (2:ℕ)^990 = m := 
by 
  sorry

end tournament_prob_unique_wins_l474_474158


namespace vectors_inner_product_sum_l474_474972

open Real

theorem vectors_inner_product_sum
  (A B C : Vect3) 
  (h₁ : (A - B).norm = sqrt 3)
  (h₂ : (B - C).norm = sqrt 5)
  (h₃ : (C - A).norm = 2 * sqrt 2)
  (h₄ : A - B + B - C + C - A = 0) :
  inner (A - B) (B - C) + inner (B - C) (C - A) + inner (C - A) (A - B) = -8 :=
by
  sorry

end vectors_inner_product_sum_l474_474972


namespace sum_g_squared_l474_474479

noncomputable def g (n : ℕ) : ℝ :=
  ∑' m, if m ≥ 3 then 1 / (m : ℝ)^n else 0

theorem sum_g_squared :
  (∑' n, if n ≥ 3 then (g n)^2 else 0) = 1 / 288 :=
by
  sorry

end sum_g_squared_l474_474479


namespace eval_256_pow_5_over_4_l474_474895

theorem eval_256_pow_5_over_4 : 256 = 2^8 → 256^(5/4 : ℝ) = 1024 :=
by
  intro h
  sorry

end eval_256_pow_5_over_4_l474_474895


namespace greatest_integer_less_than_200_with_gcd_18_l474_474769

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l474_474769


namespace ratio_of_hypotenuses_l474_474999

noncomputable def hypotenuse_ratio_of_isosceles_right_triangles
  (A B : ℝ) (hA : A > 0) (hB : B > 0) (hA_eq : B = 2 * A) : ℝ :=
√2

theorem ratio_of_hypotenuses 
  (A B : ℝ) 
  (hA : A > 0) 
  (hB : B > 0) 
  (hA_eq : B = 2 * A) : 
  hypotenuse_ratio_of_isosceles_right_triangles A B hA hB hA_eq = √2 := 
by
  sorry

end ratio_of_hypotenuses_l474_474999


namespace arithmetic_progression_sum_15_terms_l474_474150

theorem arithmetic_progression_sum_15_terms (a₃ a₅ : ℝ) (h₁ : a₃ = -5) (h₂ : a₅ = 2.4) : 
  let d := (a₅ - a₃) / 2 in
  let a₁ := a₃ - 2 * d in
  (15 / 2) * (2 * a₁ + 14 * d) = 202.5 :=
by
  sorry

end arithmetic_progression_sum_15_terms_l474_474150


namespace find_2016th_number_l474_474700

-- This sequence pattern
noncomputable def sequence_pattern (n : ℕ) : ℕ := 
  let group := (n - 1) / 6 + 1
  in if (n % 6 = 1 ∨ n % 6 = 6) then group
     else if (n % 6 = 2 ∨ n % 6 = 5) then group + 1
     else group + 2

theorem find_2016th_number : sequence_pattern 2016 = 336 := sorry

end find_2016th_number_l474_474700


namespace glass_bottles_in_second_scenario_l474_474002

theorem glass_bottles_in_second_scenario
  (G P x : ℕ)
  (h1 : 3 * G = 600)
  (h2 : G = P + 150)
  (h3 : x * G + 5 * P = 1050) :
  x = 4 :=
by 
  -- Proof is omitted
  sorry

end glass_bottles_in_second_scenario_l474_474002


namespace triathlete_average_speed_5_l474_474436

-- Definitions for speeds and distances
def swim_speed := 2
def bike_speed := 25
def run_speed := 8
def segment_length := 5

-- Harmonic mean of the provided speeds
def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / (1 / a + 1 / b + 1 / c)

-- Proof statement
theorem triathlete_average_speed_5 :
  harmonic_mean swim_speed bike_speed run_speed ≈ 5 :=
by sorry

end triathlete_average_speed_5_l474_474436


namespace initial_pigs_l474_474005

theorem initial_pigs (x : ℕ) (h1 : x + 22 = 86) : x = 64 :=
by
  sorry

end initial_pigs_l474_474005


namespace papers_above_140_l474_474818

noncomputable def num_papers_above_140_in_sample : ℕ :=
100 * 0.05

theorem papers_above_140 (mean σ : ℝ) (n_students n_sample : ℕ) : 
  (mean = 120) → 
  (P 100 < ξ < 120 = 0.45) → 
  (n_students = 2000) → 
  (n_sample = 100) → 
  samples_above_140_in_sample = 5 := 
begin
  intros mean_eq dist_eq n_students_eq n_sample_eq,
  sorry
end

end papers_above_140_l474_474818


namespace quadratic_function_correct_value_l474_474503

noncomputable def quadratic_function_value (a b x x1 x2 : ℝ) :=
  a * x^2 + b * x + 5

theorem quadratic_function_correct_value
  (a b x1 x2 : ℝ)
  (h_a : a ≠ 0)
  (h_A : quadratic_function_value a b x1 x1 x2 = 2002)
  (h_B : quadratic_function_value a b x2 x1 x2 = 2002) :
  quadratic_function_value a b (x1 + x2) x1 x2 = 5 :=
by
  sorry

end quadratic_function_correct_value_l474_474503


namespace max_min_values_l474_474357

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem max_min_values : 
  ∃ (max_val min_val : ℝ), 
    max_val = 7 ∧ min_val = -20 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max_val) ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min_val ≤ f x) := 
by
  sorry

end max_min_values_l474_474357


namespace probability_x_lt_2y_is_2_over_5_l474_474430

noncomputable def rectangle_area : ℝ :=
  5 * 2

noncomputable def triangle_area : ℝ :=
  1 / 2 * 4 * 2

noncomputable def probability_x_lt_2y : ℝ :=
  triangle_area / rectangle_area

theorem probability_x_lt_2y_is_2_over_5 :
  probability_x_lt_2y = 2 / 5 := by
  sorry

end probability_x_lt_2y_is_2_over_5_l474_474430


namespace conjugate_of_square_l474_474511

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number (1 + i)
def z : ℂ := 1 + i

-- State the theorem
theorem conjugate_of_square : complex.conj (z ^ 2) = -2 * i :=
by
  -- skipped proof steps
  sorry

end conjugate_of_square_l474_474511


namespace find_a1_in_arithmetic_sequence_l474_474263

theorem find_a1_in_arithmetic_sequence (d n a_n : ℤ) (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10) :
  ∃ a1 : ℤ, a1 = -38 :=
by
  sorry

end find_a1_in_arithmetic_sequence_l474_474263


namespace largest_angle_in_isosceles_l474_474591

theorem largest_angle_in_isosceles (α β γ : ℝ) (h1 : α = β)  (h2 : α = 50) (ht : α + β + γ = 180) : 
  ∃ θ, θ = γ ∧ θ = 80 :=
begin
  sorry
end

end largest_angle_in_isosceles_l474_474591


namespace find_f_1002_l474_474366

noncomputable def f : ℕ → ℝ :=
  sorry

theorem find_f_1002 (f : ℕ → ℝ) 
  (h : ∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) :
  f 1002 = 21 :=
sorry

end find_f_1002_l474_474366


namespace hyperbola_eccentricity_l474_474201

variable (a b c : ℝ)
hypothesis (a_pos : a > 0)
hypothesis (b_pos : b > 0)
hypothesis (focus_def : c^2 = a^2 + b^2)
hypothesis (intersect_x : ∀ x, (∃ y, (x, y) ∈ ({p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1})) → x = c)

theorem hyperbola_eccentricity : ∃ e : ℝ, e = sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_l474_474201


namespace greatest_integer_gcd_6_l474_474763

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l474_474763


namespace part1_part2_part3_l474_474192

noncomputable def g : ℝ → ℝ := λ x, 2^x

def f (x : ℝ) (m n : ℕ) : ℝ := (-g x + n) / (2 * g x + m)

theorem part1 : g 2 = 4 → g = (λ x, 2^x) :=
by 
  intro h
  sorry

theorem part2 : 
  (∀ x, f x 2 1 = -f (-x) 2 1) → (n = 1 ∧ m = 2) :=
by 
  intro h
  sorry

theorem part3 : 
  (∀ t, f (t^2 - 2 * t) 2 1 + f (2 * t^2 - k) 2 1 < 0) → k < -1/3 :=
by 
  intro h
  sorry

end part1_part2_part3_l474_474192


namespace projection_is_correct_l474_474023

theorem projection_is_correct :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (4, -1)
  let p : ℝ × ℝ := (15/58, 35/58)
  let d : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)
  ∃ v : ℝ × ℝ, 
    (a.1 * v.1 + a.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧
    (b.1 * v.1 + b.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧ 
    (p.1 * d.1 + p.2 * d.2 = 0) :=
sorry

end projection_is_correct_l474_474023


namespace min_value_of_expression_l474_474135

theorem min_value_of_expression : ∀ x : ℝ, 4^x - 2^x + 1 ≥ 3 / 4 := 
by
  sorry

end min_value_of_expression_l474_474135


namespace combination_8_5_l474_474618

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l474_474618


namespace greatest_difference_four_digit_numbers_l474_474997

theorem greatest_difference_four_digit_numbers : 
  ∃ (d1 d2 d3 d4 : ℕ), (d1 = 0 ∨ d1 = 3 ∨ d1 = 4 ∨ d1 = 8) ∧ 
                      (d2 = 0 ∨ d2 = 3 ∨ d2 = 4 ∨ d2 = 8) ∧ 
                      (d3 = 0 ∨ d3 = 3 ∨ d3 = 4 ∨ d3 = 8) ∧ 
                      (d4 = 0 ∨ d4 = 3 ∨ d4 = 4 ∨ d4 = 8) ∧ 
                      d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
                      d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
                      (∃ n1 n2, n1 = 1000 * 8 + 100 * 4 + 10 * 3 + 0 ∧ 
                                n2 = 1000 * 3 + 100 * 0 + 10 * 4 + 8 ∧ 
                                n1 - n2 = 5382) :=
by {
  sorry
}

end greatest_difference_four_digit_numbers_l474_474997


namespace no_integers_exist_l474_474040

theorem no_integers_exist :
  ¬ (∃ x y : ℤ, (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2) :=
by
  sorry

end no_integers_exist_l474_474040


namespace rectangle_area_invariant_l474_474724

theorem rectangle_area_invariant
    (x y : ℕ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 3) * (y + 2)) :
    x * y = 15 :=
by sorry

end rectangle_area_invariant_l474_474724


namespace tangent_product_value_l474_474222

theorem tangent_product_value (A B : ℝ) (hA : A = 20) (hB : B = 25) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
sorry

end tangent_product_value_l474_474222


namespace younger_by_17_l474_474376

variables (A B C : ℕ)

-- Given condition
axiom age_condition : A + B = B + C + 17

-- To show
theorem younger_by_17 : A - C = 17 :=
by
  sorry

end younger_by_17_l474_474376


namespace combination_seven_choose_four_l474_474715

theorem combination_seven_choose_four : nat.choose 7 4 = 35 := by
  -- Proof
  sorry

end combination_seven_choose_four_l474_474715


namespace smallest_number_of_weights_l474_474021

/-- The smallest number of weights in a set that can be divided into 4, 5, and 6 equal piles is 11. -/
theorem smallest_number_of_weights (n : ℕ) (M : ℕ) : (∀ k : ℕ, (k = 4 ∨ k = 5 ∨ k = 6) → M % k = 0) → n = 11 :=
sorry

end smallest_number_of_weights_l474_474021


namespace roots_polynomial_expression_l474_474310

theorem roots_polynomial_expression {p q r : ℝ}
  (h1 : p ≠ q) (h2 : q ≠ r) (h3 : r ≠ p)
  (h4 : Polynomial.root (Polynomial.X^3 - 15*Polynomial.X^2 + 22*Polynomial.X - 8) p)
  (h5 : Polynomial.root (Polynomial.X^3 - 15*Polynomial.X^2 + 22*Polynomial.X - 8) q)
  (h6 : Polynomial.root (Polynomial.X^3 - 15*Polynomial.X^2 + 22*Polynomial.X - 8) r) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 406 :=
by
  sorry

end roots_polynomial_expression_l474_474310


namespace solution_of_inequality_l474_474140

noncomputable def solutionSet (a x : ℝ) : Set ℝ :=
  if a > 0 then {x | -a < x ∧ x < 3 * a}
  else if a < 0 then {x | 3 * a < x ∧ x < -a}
  else ∅

theorem solution_of_inequality (a x : ℝ) :
  (x^2 - 2 * a * x - 3 * a^2 < 0 ↔ x ∈ solutionSet a x) :=
sorry

end solution_of_inequality_l474_474140


namespace third_number_l474_474812

theorem third_number (x : ℝ) 
    (h : 217 + 2.017 + 2.0017 + x = 221.2357) : 
    x = 0.217 :=
sorry

end third_number_l474_474812


namespace required_paving_stones_l474_474553

def courtyard_length : Real := 158.5
def courtyard_width : Real := 35.4
def stone_length : Real := 3.2
def stone_width : Real := 2.7

def area_courtyard : Real := courtyard_length * courtyard_width
def area_stone : Real := stone_length * stone_width

def number_of_stones : Real := area_courtyard / area_stone

theorem required_paving_stones :
  Int.ceil number_of_stones = 650 :=
by
  sorry

end required_paving_stones_l474_474553


namespace ilya_defeats_dragon_l474_474241

section DragonAndIlya

def Probability (n : ℕ) : Type := ℝ

noncomputable def probability_of_defeat : Probability 3 :=
  let p_no_regrow := 5 / 12
  let p_one_regrow := 1 / 3
  let p_two_regrow := 1 / 4
  -- Assuming recursive relationship develops to eventually reduce heads to zero
  1

-- Prove that the probability_of_defeat is equal to 1
theorem ilya_defeats_dragon : probability_of_defeat = 1 :=
by
  -- Formal proof would be provided here
  sorry

end DragonAndIlya

end ilya_defeats_dragon_l474_474241


namespace radius_of_circle_is_sqrt2_l474_474369

noncomputable def radius_of_circle : ℝ :=
  let ρ := 2
  let θ : ℝ := sorry -- In reality, θ will vary, so this is just a placeholder
  let φ := π / 4
  let r := ρ * Real.sin φ
  r

theorem radius_of_circle_is_sqrt2 :
  radius_of_circle = Real.sqrt 2 := by
  -- Since θ cancels out in the computation, r simplifies to sqrt(2)
  sorry

end radius_of_circle_is_sqrt2_l474_474369


namespace correct_eccentricity_l474_474527

noncomputable def hyperbola_eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  let asymptote := (b / a) in
  let x_line := (-a^2 / real.sqrt (a^2 + b^2)) in
  let d := (b * a * e) / real.sqrt (a^2 + b^2) in
  let L := 2 * b^3 / (a * real.sqrt (a^2 + b^2)) in
  (asymptote * a * e / real.sqrt (a^2 + b^2) = 2 * b^3 / (a * real.sqrt (a^2 + b^2))) →
  (e = 2)

theorem correct_eccentricity (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  hyperbola_eccentricity a b 2 :=
by
  obtain ⟨a_nonzero, b_nonzero⟩ := h,
  sorry

end correct_eccentricity_l474_474527


namespace log_product_arithmetic_seq_common_diff_l474_474521

variable {a : ℕ → ℤ}
variable {f : ℤ → ℤ}

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

theorem log_product_arithmetic_seq_common_diff :
  (∀ x : ℤ, f x = 2^x) →
  is_arithmetic_sequence a 2 →
  f (a 2 + a 4 + a 6 + a 8 + a 10) = 4 →
  (log 2 (f (a 1) * f (a 2) * f (a 3) * f (a 4) * f (a 5) * f (a 6) * f (a 7) * f (a 8) * f (a 9) * f (a 10)) = -6) :=
by
  sorry

end log_product_arithmetic_seq_common_diff_l474_474521


namespace object_speed_mph_l474_474408

theorem object_speed_mph (distance_feet : ℝ) (time_seconds : ℝ) (ft_per_mile : ℝ) (sec_per_hour : ℝ) :
  distance_feet = 300 → time_seconds = 5 → ft_per_mile = 5280 → sec_per_hour = 3600 →
  distance_feet / ft_per_mile / (time_seconds / sec_per_hour) ≈ 40.9091 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end object_speed_mph_l474_474408


namespace max_re_sum_eq_96_l474_474674

noncomputable def max_real_part_sum_w (z : ℕ → ℂ) (w : ℕ → ℂ) : ℝ :=
  ∑ j in finset.range 24, complex.re (w j)

theorem max_re_sum_eq_96 (z : ℕ → ℂ) (w : ℕ → ℂ)
  (h1 : ∀ j, z j = complex.of_real 16 * complex.exp (complex.I * (2 * real.pi * j / 24))) 
  (h2 : ∀ j, w j = z j ∨ w j = complex.I * z j ∨ w j = -complex.I * z j)
  : max_real_part_sum_w z w = 96 :=
sorry

end max_re_sum_eq_96_l474_474674


namespace sum_of_cube_edges_l474_474934

theorem sum_of_cube_edges (edge_len : ℝ) (num_edges : ℕ) (lengths : ℝ) (h1 : edge_len = 15) (h2 : num_edges = 12) : lengths = num_edges * edge_len :=
by
  sorry

end sum_of_cube_edges_l474_474934


namespace hyperbola_eccentricity_l474_474579

variable (a b c e : ℝ)
variables (ha : a > 0) (hb : b > 0) (hb_a : b = 2 * a) 

theorem hyperbola_eccentricity : 
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = sqrt 5 :=
by 
  sorry

end hyperbola_eccentricity_l474_474579


namespace find_s_l474_474367

theorem find_s
  (p q r s : ℝ)
  (h : p^2 + q^2 + r^2 + 4 = s + sqrt (p + q + r - s)) :
  s = 5 / 4 :=
sorry

end find_s_l474_474367


namespace max_value_of_b_ln_2_approximation_l474_474524

noncomputable def f (x : ℝ) : ℝ := exp x - exp (-x) - 2 * x
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (2 * x) - 4 * b * f x

theorem max_value_of_b :
  ∀ x > 0, g x 2 > 0 := 
  sorry

theorem ln_2_approximation :
  0.6928 < Real.log 2 ∧ Real.log 2 < 0.6934 := 
  sorry

end max_value_of_b_ln_2_approximation_l474_474524
