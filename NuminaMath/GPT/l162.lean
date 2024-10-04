import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.Finset
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Parity
import Mathlib.Analysis.Calculus.Basic
import Mathlib.Analysis.Calculus.Trigonometry
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecificLimits
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Modulo
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Combinatorics.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.NatGcd
import Mathlib.Order.Floor
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace probability_all_pairs_lucky_expected_lucky_pairs_gt_half_l162_162554

-- Define the context and parameters
variables (n : ℕ)

-- Indicator variable for the k-th pair being lucky
def xi (k : ℕ) (n : ℕ) : Prop :=
  true

-- Define the sum of lucky pairs
def X (n : ℕ) : ℕ :=
  finset.sum (finset.range n) (λ k, if xi k n then 1 else 0)

-- Part (a): Probability that all pairs are lucky
theorem probability_all_pairs_lucky (n : ℕ) : 
  let p := (2 ^ n * factorial n : ℝ) / (factorial (2 * n)) in
  p = 2^n * n! / (2*n)! := sorry

-- Part (b): Expected number of lucky pairs is greater than 0.5
theorem expected_lucky_pairs_gt_half (n : ℕ) : 
  let E := (n : ℝ) / (2 * n - 1) in
  E > 0.5 := sorry

end probability_all_pairs_lucky_expected_lucky_pairs_gt_half_l162_162554


namespace simplify_expression_l162_162220

theorem simplify_expression :
  let expr := 4 * (x^2 - 2 * x^3) + 3 * (x - x^2 + 2 * x^4) - (5 * x^3 - 2 * x^2 + x)
  coeff_of_p2 (simplify expr) = 3 := by
  sorry

end simplify_expression_l162_162220


namespace exp_eval_l162_162481

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162481


namespace find_x_squared_plus_4_l162_162712

theorem find_x_squared_plus_4 (x : ℝ) (h : 3^(2*x) + 27 = 15 * 3^x) : x^2 + 4 = 5 :=
sorry

end find_x_squared_plus_4_l162_162712


namespace range_of_a_l162_162694

def f (a x : ℝ) : ℝ := Real.log x + x^2 - a * x
def h (x : ℝ) : ℝ := 1 / (2 : ℝ) * (3 * x ^ 2 + 1 / x ^ 2 - 6 * x)

theorem range_of_a 
  (a : ℝ)
  (h_incr : ∀ (x : ℝ), x > 0 → (1/x + 2*x - a) ≥ 0)
  (h_le : ∀ (x : ℝ), 0 < x ∧ x ≤ 1 → f a x ≤ h x) 
  : 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := 
sorry

end range_of_a_l162_162694


namespace A_union_B_eq_l162_162035

noncomputable def A (a : ℤ) : Set ℤ := {5, Int.log2 (a + 3)}
def B (a b : ℤ) : Set ℤ := {a, b}
def condition (a b : ℤ) : Prop := ∀ x, x ∈ ({5, Int.log2 (a + 3)} : Set ℤ) ∩ ({a, b} : Set ℤ) ↔ (x = 2)

theorem A_union_B_eq {a b : ℤ} (h : condition a b) (ha : a = 1) (hb : b = 2) :
  ({5, Int.log2 (a + 3)} : Set ℤ) ∪ ({a, b} : Set ℤ) = {1, 2, 5} :=
by
  sorry

end A_union_B_eq_l162_162035


namespace exists_P_satisfying_sqrt3_PA_eq_PB_min_value_PA_plus_PB_is_5sqrt2_max_value_PA_minus_PB_is_2sqrt5_l162_162263

-- Define the points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (1, -3)

-- Define the line l
def line_l (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

-- Define a function to compute distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Statement B: There exists a point P on the line such that sqrt(3) * |PA| = |PB|
theorem exists_P_satisfying_sqrt3_PA_eq_PB : 
  ∃ P : ℝ × ℝ, line_l P ∧ real.sqrt 3 * distance P A = distance P B := sorry

-- Statement C: The minimum value of |PA| + |PB| is 5 * sqrt(2)
theorem min_value_PA_plus_PB_is_5sqrt2 :
  ∀ P : ℝ × ℝ, line_l P → distance P A + distance P B ≥ 5 * real.sqrt 2 := sorry

-- Statement D: The maximum value of ||PA| - |PB|| is 2 * sqrt(5)
theorem max_value_PA_minus_PB_is_2sqrt5 :
  ∀ P : ℝ × ℝ, line_l P → abs (distance P A - distance P B) ≤ 2 * real.sqrt 5 := sorry

end exists_P_satisfying_sqrt3_PA_eq_PB_min_value_PA_plus_PB_is_5sqrt2_max_value_PA_minus_PB_is_2sqrt5_l162_162263


namespace conical_pile_new_height_l162_162941

noncomputable def newHeight (d : ℝ) (initialHeight : ℝ) (additionalVolume : ℝ) : ℝ :=
  let r := d / 2
  let initialVolume := (1 / 3) * π * r^2 * initialHeight
  let totalVolume := initialVolume + additionalVolume
  (3 * totalVolume) / (π * r^2)

theorem conical_pile_new_height :
  newHeight 10 6 2 = 6 + (6 / (25 * π)) :=
by
  sorry

end conical_pile_new_height_l162_162941


namespace angle_KJG_eq_135_l162_162752

structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_angles : angle1 + angle2 + angle3 = 180

structure Square where
  side_length : ℝ
  angle1 : ℝ := 90
  angle2 : ℝ := 90
  angle3 : ℝ := 90
  angle4 : ℝ := 90

noncomputable def triangle_JKL : Triangle := 
{ angle1 := 45, angle2 := 45, angle3 := 90,
  sum_angles := by norm_num }

noncomputable def square_GHIJ : Square := 
{ side_length := 1 }  -- Assign an arbitrary side_length just to instantiate

noncomputable def square_JKLK : Square :=
{ side_length := 1 }  -- Assign an arbitrary side_length just to instantiate

theorem angle_KJG_eq_135 : triangle_JKL.angle1 = 45 ∧ triangle_JKL.angle2 = 45 ∧ 
  square_GHIJ.angle1 = 90 ∧ square_JKLK.angle1 = 90 → 
  ∠ KJG = 135 := 
by {
  intros h,
  sorry 
}

end angle_KJG_eq_135_l162_162752


namespace undefined_value_of_expression_l162_162646

theorem undefined_value_of_expression (a : ℝ) : (a^3 - 8 = 0) → (a = 2) := by
  sorry

end undefined_value_of_expression_l162_162646


namespace factorize_expression_l162_162982

theorem factorize_expression (x y : ℝ) : 25 * x - x * y ^ 2 = x * (5 + y) * (5 - y) := by
  sorry

end factorize_expression_l162_162982


namespace jill_arrives_before_jack_l162_162011

theorem jill_arrives_before_jack {distance speed_jill speed_jack : ℝ} (h1 : distance = 1) 
  (h2 : speed_jill = 10) (h3 : speed_jack = 4) :
  (60 * (distance / speed_jack) - 60 * (distance / speed_jill)) = 9 :=
by
  sorry

end jill_arrives_before_jack_l162_162011


namespace max_s_value_l162_162092

noncomputable def max_s (m n : ℝ) : ℝ := (m-1)^2 + (n-1)^2 + (m-n)^2

theorem max_s_value (m n : ℝ) (h : m^2 - 4 * n ≥ 0) : 
    ∃ s : ℝ, s = (max_s m n) ∧ s ≤ 9/8 := sorry

end max_s_value_l162_162092


namespace problem_solution_l162_162384

theorem problem_solution (x y z : ℕ)
  (h1 : x ≥ y) (h2 : y ≥ z)
  (h3 : x^2 - y^2 - z^2 + x * y = 2251)
  (h4 : x^2 + 2 * y^2 + 2 * z^2 - 2 * x * y - 3 * x * z - 3 * y * z = -2090) :
  x = 253 :=
by sorry

end problem_solution_l162_162384


namespace R_depends_on_d_and_k_l162_162788

section ArithmeticProgression

variables (a d k : ℕ)
variables (s1 s2 s4 : ℕ)

-- Definitions for sums in arithmetic progression
def s1 := (k * (2 * a + (k - 1) * d)) / 2
def s2 := k * (2 * a + 2 * k * d - d)
def s4 := 2 * k * (2 * a + 4 * k * d - d)

-- Definition of R
def R := s4 - s2 - s1

theorem R_depends_on_d_and_k : 
  ∀ (a d k : ℕ), depends_on R d ∧ depends_on R k := 
sorry

end ArithmeticProgression

end R_depends_on_d_and_k_l162_162788


namespace cricket_current_average_l162_162574

theorem cricket_current_average (A : ℕ) (h1: 10 * A + 77 = 11 * (A + 4)) : 
  A = 33 := 
by 
  sorry

end cricket_current_average_l162_162574


namespace monotonicity_of_g_range_of_a_for_no_zeros_l162_162696

noncomputable def f (x : ℝ) : ℝ := Real.exp(x / 2) - x / 4

noncomputable def g (x : ℝ) : ℝ := (x + 1) * (1 / 2 * Real.exp(x / 2) - 1 / 4)

noncomputable def F (a x : ℝ) : ℝ := Real.log(x + 1) - a * (Real.exp(x / 2) - x / 4) + 4

theorem monotonicity_of_g : ∀ x > -1, 0 < (x + 3) * Real.exp(x / 2) - 1 :=
sorry

theorem range_of_a_for_no_zeros :
  ∀ a > 4, ∀ x > -1, F a x ≠ 0 :=
sorry

end monotonicity_of_g_range_of_a_for_no_zeros_l162_162696


namespace train_speed_l162_162179

theorem train_speed 
    (train_length : ℕ) 
    (bridge_length : ℕ) 
    (crossing_time : ℕ) 
    (train_length_val : train_length = 110) 
    (bridge_length_val : bridge_length = 265) 
    (crossing_time_val : crossing_time = 30) : 
    (train_length + bridge_length) / crossing_time * 3.6 = 45 := 
by
    sorry

end train_speed_l162_162179


namespace value_of_2_68_times_0_74_l162_162134

theorem value_of_2_68_times_0_74 : 
  (268 * 74 = 19732) → (2.68 * 0.74 = 1.9732) :=
by intro h1; sorry

end value_of_2_68_times_0_74_l162_162134


namespace distance_from_A_to_B_l162_162876

theorem distance_from_A_to_B
    (perimeter_smaller_square : ℝ)
    (area_larger_square : ℝ)
    (h1 : perimeter_smaller_square = 8)
    (h2 : area_larger_square = 64) :
    let side_smaller_square := perimeter_smaller_square / 4,
        side_larger_square := real.sqrt area_larger_square,
        horizontal_side := side_smaller_square + side_larger_square,
        vertical_side := side_larger_square - side_smaller_square in
    (real.sqrt (horizontal_side ^ 2 + vertical_side ^ 2)).round = 12 :=
by
    sorry

end distance_from_A_to_B_l162_162876


namespace exists_real_number_l162_162016

-- Conditions
variables (n : ℕ) (h_pos : n > 0)
          (a : Fin n → ℝ) (h_distinct : Function.Injective a)
          (f : Fin n → ℝ → ℝ) (h_bounded : ∀ i, Bounded (range (f i)))

-- Statement of the theorem to be proved
theorem exists_real_number :
  ∃ x : ℝ, (∑ i in Finset.univ, f i x) - (∑ i in Finset.univ, f i (x - a i)) < 1 :=
sorry

end exists_real_number_l162_162016


namespace math_problem_proof_l162_162148

-- Definitions based on identified conditions
def x : ℝ := Real.sqrt 20.8
def y : ℝ := x^2
def z : ℝ := 104 / y
def w : ℝ := z^3
def result : ℝ := 550 - w

-- The proof problem
theorem math_problem_proof : result = 425 :=
by
  sorry

end math_problem_proof_l162_162148


namespace intervals_of_monotonic_increase_find_max_value_l162_162396

noncomputable def f (x a : ℝ) := cos (2 * x + π / 3) + sqrt 3 * sin (2 * x) + 2 * a

-- Prove that the intervals of monotonic increase for f(x)
theorem intervals_of_monotonic_increase (k : ℤ) : 
  ∀ a, 
  ∃ I : set ℝ, (I = set.Icc (-π / 3 + k * π) (π / 6 + k * π)) ∧ 
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x a ≤ f y a) := 
sorry

-- Given that the minimum value of f(x) in [0, π/4] is 0, prove that the maximum value is 1/2
theorem find_max_value (a : ℝ) (h : (∀ x ∈ set.Icc (0 : ℝ) (π / 4), f x a ≥ 0) 
                                 ∧ ∃ x ∈ set.Icc (0 : ℝ) (π / 4), f x a = 0) :
  a = -1/4 → ∃ x ∈ set.Icc (0 : ℝ) (π / 4), f x (-1 / 4) = 1 / 2 :=
sorry

end intervals_of_monotonic_increase_find_max_value_l162_162396


namespace bags_filled_on_sunday_l162_162830

theorem bags_filled_on_sunday (total_bags : Nat) (bags_saturday : Nat) (cans_per_bag : Nat) (total_cans : Nat) : 
  (bags_saturday = 3) → (cans_per_bag = 9) → (total_cans = 63) → total_bags = total_cans / cans_per_bag → 
  (total_bags - bags_saturday = 4) :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end bags_filled_on_sunday_l162_162830


namespace ratio_of_areas_l162_162003

noncomputable def triangle_XYZ := {XY : ℝ, YZ : ℝ, XZ : ℝ}

noncomputable def points_PQ := {P : ℝ, Q : ℝ}

axiom assume_XYZ_PQ (XYZ : triangle_XYZ) (PQ : points_PQ) :
  XYZ.XY = 30 → XYZ.YZ = 45 → XYZ.XZ = 54 → PQ.P = 23 → PQ.Q = 18 →
  let Area_Trig_XPQ := (23/30) * (18/54) in
  let Area_Quadrilateral_PQZY := 1 - (23/30)^2 in
  (Area_Trig_XPQ / Area_Quadrilateral_PQZY) = 529 / 371

-- The theorem statement:
theorem ratio_of_areas (XYZ : triangle_XYZ) (PQ : points_PQ) :
  XYZ.XY = 30 → XYZ.YZ = 45 → XYZ.XZ = 54 → PQ.P = 23 → PQ.Q = 18 →
  (529 / 900 / (1 - 529 / 900)) = (529 / 371) :=
sorry -- proof goes here

end ratio_of_areas_l162_162003


namespace demand_for_profit_l162_162575

-- Definitions
def profit (x : ℝ) : ℝ :=
  if x ∈ set.Icc 100 130 then 800 * x - 39000 else
  if x ∈ set.Ioc 130 150 then 65000 else 0

theorem demand_for_profit (x : ℝ) (h₁ : 100 ≤ x) (h₂ : x ≤ 150) :
  (57_000 ≤ profit x) ↔ (120 ≤ x ∧ x ≤ 150) :=
sorry

end demand_for_profit_l162_162575


namespace equidistant_point_l162_162637

def point := ℝ × ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem equidistant_point :
  ∃ y, distance (0, y, 0) (-2, 4, -6) = distance (0, y, 0) (8, 5, 1) ∧ y = 17 :=
by
  sorry

end equidistant_point_l162_162637


namespace car_original_cost_price_l162_162545

theorem car_original_cost_price (selling_price_friend : ℝ) (loss_percentage : ℝ) (gain_percentage : ℝ) :
  selling_price_friend = 54000 → loss_percentage = 0.11 → gain_percentage = 0.20 →
  let y := selling_price_friend / (1 + gain_percentage) in
  let x := y / (1 - loss_percentage) in
  x ≈ 50561.80 :=
by sorry

end car_original_cost_price_l162_162545


namespace exp_eval_l162_162487

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162487


namespace simplify_expression_l162_162023

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : a ≠ 1)
variable (h3 : 0 < b)

theorem simplify_expression : a ^ Real.log (1 / b ^ Real.log a) = 1 / b ^ (Real.log a) ^ 2 :=
by
  sorry

end simplify_expression_l162_162023


namespace money_initial_amounts_l162_162868

theorem money_initial_amounts (x : ℕ) (A B : ℕ) 
  (h1 : A = 8 * x) 
  (h2 : B = 5 * x) 
  (h3 : (A - 50) = 4 * (B + 100) / 5) : 
  A = 800 ∧ B = 500 := 
sorry

end money_initial_amounts_l162_162868


namespace sum_of_roots_to_decimal_l162_162124

theorem sum_of_roots_to_decimal :
  let f : ℝ → ℝ := λ x, 3 * x^3 - 7 * x^2 + 2 * x 
  ∃ (α β : ℝ), 
    (f(α) = 0 ∧ f(β) = 0 ∧ α ≠ 0 ∧ β ≠ 0) →
    (Float.ofInt ((α + β)*100).round/100 = 2.33) :=
  
by
  -- Placeholder for the proof steps
  sorry

end sum_of_roots_to_decimal_l162_162124


namespace Sammy_has_8_bottle_caps_l162_162413

def Billie_caps : Nat := 2
def Janine_caps (B : Nat) : Nat := 3 * B
def Sammy_caps (J : Nat) : Nat := J + 2

theorem Sammy_has_8_bottle_caps : 
  Sammy_caps (Janine_caps Billie_caps) = 8 := 
by
  sorry

end Sammy_has_8_bottle_caps_l162_162413


namespace possible_values_l162_162270

-- Definitions for the absolute value expressions
def abs_val (x : ℝ) : ℝ := if x >= 0 then x else -x

def sgn (x : ℝ) : ℝ := if x > 0 then 1 else -1

-- Main theorem
theorem possible_values (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : 
  let expr := (a * b) / (abs_val (a * b)) + c / (abs_val c) + (a * b * c) / (abs_val (a * b * c))
  in expr = 3 ∨ expr = -1 :=
by
  sorry

end possible_values_l162_162270


namespace biased_coin_probability_l162_162531

noncomputable def h := 2 - Real.sqrt 2

def prob_heads_twice_eq_heads_four_times : Prop :=
  (choose 5 2) * (h^2) * ((1 - h)^3) = (choose 5 4) * (h^4) * (1 - h)

def prob_heads_three_times : ℚ :=
  (choose 5 3) * (h^3) * (Real.sqrt 2 - 1)^2

def fraction_in_lowest_terms (p : ℚ) : Prop :=
  ∃ i j : ℤ, i.gcd j = 1 ∧ p = i / j

def i_plus_j (p : ℚ) : ℤ :=
  p.num.natAbs + p.den

theorem biased_coin_probability : i_plus_j prob_heads_three_times = 283 :=
by
  have h_val : h = 2 - Real.sqrt 2, from sorry,
  have cond : prob_heads_twice_eq_heads_four_times, from sorry,
  show i_plus_j prob_heads_three_times = 283, from sorry

end biased_coin_probability_l162_162531


namespace even_product_probability_l162_162839

theorem even_product_probability:
  let s := set.Icc 6 18 in
  ((∃ a b : ℤ, a ≠ b ∧ a ∈ s ∧ b ∈ s ∧ ¬even (a * b)) → (2 / 13) - (19 / 39) = (21 / 26)) := sorry

end even_product_probability_l162_162839


namespace enlarged_grid_perimeter_l162_162836

theorem enlarged_grid_perimeter (original_rows : ℕ) (original_columns : ℕ)
  (side_length : ℝ) (extra_rows : ℕ) : 
  original_rows = 3 → original_columns = 5 → side_length = 1 → extra_rows = 1 →
  let new_rows := original_rows + extra_rows in
  let horizontal_segments := original_columns + new_rows + new_rows * original_columns in
  let vertical_segments := (new_rows + 1) * original_columns + original_columns + 1 in
  2 * horizontal_segments + 2 * vertical_segments = 37 := 
by 
  intros original_rows_eq original_columns_eq side_length_eq extra_rows_eq
  let new_rows := 3 + 1
  let horizontal_segments := 5 + 4 + 4 * 5
  let vertical_segments := (4 + 1) * 5 + 5 + 1
  have h1 : 2 * horizontal_segments + 2 * vertical_segments = 37 := sorry
  exact h1

end enlarged_grid_perimeter_l162_162836


namespace area_of_triangle_BFG_l162_162015

theorem area_of_triangle_BFG (A B C D E F G : Point) (s : ℝ) 
  (h_right_triangle : triangle ABC ∧ is_right_triangle ABC ∧ hypoteneuse AC)
  (h_square_inscribed : square_inscribed A B C D E F G)
  (h_AG : AG = 2)
  (h_CF : CF = 5) :
  area_of_triangle BFG = 500 / 841 :=
by
  sorry

end area_of_triangle_BFG_l162_162015


namespace multiples_of_10_have_highest_average_between_1_and_201_l162_162703

theorem multiples_of_10_have_highest_average_between_1_and_201 :
  let avg (a b : ℕ) := (a + b) / 2 in
  avg 7 (7 * (201 / 7).to_nat) < avg 10 (10 * (201 / 10).to_nat) ∧ 
  avg 8 (8 * (201 / 8).to_nat) < avg 10 (10 * (201 / 10).to_nat) ∧
  avg 9 (9 * (201 / 9).to_nat) < avg 10 (10 * (201 / 10).to_nat) ∧
  avg 11 (11 * (201 / 11).to_nat) < avg 10 (10 * (201 / 10).to_nat) :=
by
  let avg (a b : ℕ) := (a + b) / 2
  have a1 := avg 7 (7 * (201 / 7).to_nat)
  have a2 := avg 8 (8 * (201 / 8).to_nat)
  have a3 := avg 9 (9 * (201 / 9).to_nat)
  have a4 := avg 10 (10 * (201 / 10).to_nat)
  have a5 := avg 11 (11 * (201 / 11).to_nat)
  show a1 < a4 ∧ a2 < a4 ∧ a3 < a4 ∧ a5 < a4
  sorry

end multiples_of_10_have_highest_average_between_1_and_201_l162_162703


namespace exponentiation_example_l162_162474

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162474


namespace shaded_area_correct_l162_162219

noncomputable def semicircle_area (R : ℝ) : ℝ := (π * R^2) / 2

def shaded_area (R : ℝ) (alpha : ℝ) : ℝ := 
  if alpha = π / 3 then (2 * π * R^2) / 3 else 0

theorem shaded_area_correct (R : ℝ) : shaded_area R (π / 3) = (2 * π * R^2) / 3 := by
  sorry

end shaded_area_correct_l162_162219


namespace smallest_n_conditions_l162_162905

theorem smallest_n_conditions :
  ∃ n : ℕ, 0 < n ∧ (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^4) ∧ n = 54 :=
by
  sorry

end smallest_n_conditions_l162_162905


namespace correct_operations_l162_162908

theorem correct_operations :
  ∀ (a t x y b : ℝ),
    (a^6 / a^(-2) ≠ a^4) →
    (-2t * (3t + t^2 -1) = -6t^2 - 2t^3 + 2t) →
    ((-2 * x * y^3)^2 = 4 * x^2 * y^6) →
    ((a - b) * (a + b) ≠ a^2 - b * c) →
    True :=
by
  intros a t x y b h1 h2 h3 h4
  trivial

end correct_operations_l162_162908


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l162_162611

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l162_162611


namespace sum_of_solutions_l162_162997

theorem sum_of_solutions:
  ∑ (x : ℝ) in {x | 2 * cos (2 * x) * (cos (2 * x) - cos (1007 * π^2 / x)) = cos (4 * x) - 1 ∧ 0 < x}, x = 1080 * π :=
by
  sorry

end sum_of_solutions_l162_162997


namespace integral_evaluation_l162_162213

noncomputable def semicircle_integral : ℝ :=
∫ x in -2..2, sqrt (4 - x^2)

lemma integral_semicircle_equals_2_pi :
  semicircle_integral = 2 * Real.pi :=
sorry

theorem integral_evaluation :
  (∫ x in -2..2, 1 + sqrt (4 - x^2)) = 4 + 2 * Real.pi :=
by
  have h1 : (∫ x in -2..2, sqrt (4 - x^2)) = 2 * Real.pi, from integral_semicircle_equals_2_pi,
  have h2 : (∫ x in -2..2, 1) = 4, from intervalIntegral.integral_of_le (-2 : ℝ) 2
    (by norm_num),
  calc
    (∫ x in -2..2, 1 + sqrt (4 - x^2))
        = (∫ x in -2..2, 1) + (∫ x in -2..2, sqrt (4 - x^2)) : by apply intervalIntegral.integral_add
        ... = 4 + 2 * Real.pi : by rw [h2, h1]

end integral_evaluation_l162_162213


namespace cosine_func_monotone_and_min_value_l162_162859

-- Defining the function
def cosine_func (x : ℝ) : ℝ := cos (2 * x - π / 6)

-- The problem statement proves that the function is monotonically decreasing in the given interval
-- and has a minimum value of -√3/2 in the specific interval.
theorem cosine_func_monotone_and_min_value : 
  ∀ x ∈ Icc (π / 12) (π / 2), 
    monotone_on (λ x, cos (2 * x - π / 6)) (Icc (π / 12) (π / 2)) ∧
    ∃ x ∈ Icc 0 (π / 2), cos (2 * x - π / 6) = - √ 3 / 2 :=
by
  -- This is where the proof would go
  sorry

end cosine_func_monotone_and_min_value_l162_162859


namespace range_of_uv_sq_l162_162667

theorem range_of_uv_sq (u v w : ℝ) (h₀ : 0 ≤ u) (h₁ : 0 ≤ v) (h₂ : 0 ≤ w) (h₃ : u + v + w = 2) :
  0 ≤ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ∧ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ≤ 1 :=
sorry

end range_of_uv_sq_l162_162667


namespace triangle_ineq_min_value_l162_162759

theorem triangle_ineq_min_value
  (ABC : Triangle)
  (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : ABC.∠_ABC = 120 * Real.pi / 180)
  (D : Point) (h₄ : PointOnLine D ABC.AC)
  (h₅ : dist ABC.B D = 1) : 
  ∃ a b c : ℝ, (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 4 * a + c = 9 := 
sorry

end triangle_ineq_min_value_l162_162759


namespace ellipse_standard_equation_l162_162678

-- Definitions based on the provided conditions
def foci_on_y_axis : Prop := true
def sum_distances (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 8
def focal_length : ℝ := 2 * Real.sqrt 15

-- Coordinates of foci F₁ and F₂ given that they lie on the y-axis
def F₁ : ℝ × ℝ := (0, focal_length / 2)
def F₂ : ℝ × ℝ := (0, -focal_length / 2)

-- Condition that for any point on the ellipse, the sum of distances to the foci is 8
axiom distances_property (P : ℝ × ℝ) : sum_distances P F₁ F₂

-- Target standard equation of the ellipse to be proved
theorem ellipse_standard_equation : 
  (∃ (a b : ℝ), a = 4 ∧ b = 1 ∧ 
  ∀ (x y : ℝ), sum_distances (x, y) F₁ F₂ → y^2 / 16 + x^2 = 1) :=
sorry

end ellipse_standard_equation_l162_162678


namespace lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l162_162921

-- Define a cube with edge length a
structure Cube :=
  (a : ℝ) -- Edge length of the cube

-- Define a pyramid with a given height
structure Pyramid :=
  (h : ℝ) -- Height of the pyramid

-- The main theorem statement for part 4A
theorem lateral_edges_in_same_plane (c : Cube) (p : Pyramid) : p.h = c.a ↔ (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
  O1 = (c.a / 2, c.a / 2, -p.h) ∧
  O2 = (c.a / 2, -p.h, c.a / 2) ∧
  O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

-- The main theorem statement for part 4B
theorem edges_in_planes_for_all_vertices (c : Cube) (p : Pyramid) : p.h = c.a ↔ ∀ (v : ℝ × ℝ × ℝ), -- Iterate over cube vertices
  (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
    O1 = (c.a / 2, c.a / 2, -p.h) ∧
    O2 = (c.a / 2, -p.h, c.a / 2) ∧
    O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

end lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l162_162921


namespace smallest_m_value_l162_162365

theorem smallest_m_value : ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0 ∧ m = 111 := by
  sorry

end smallest_m_value_l162_162365


namespace at_least_one_non_negative_l162_162454

variable (x : ℝ)
def a : ℝ := x^2 - 1
def b : ℝ := 2*x + 2

theorem at_least_one_non_negative (x : ℝ) : ¬ (a x < 0 ∧ b x < 0) :=
by
  sorry

end at_least_one_non_negative_l162_162454


namespace cyclic_sum_identity_l162_162777

noncomputable theory
open_locale classical

-- Define the polynomial and roots
def polynomial := λ x : ℝ, x^3 - 2 * x^2 + 3 * x - 4

-- Assume a, b, and c are distinct roots of the polynomial
variables {a b c : ℝ}
axiom root_a : polynomial a = 0
axiom root_b : polynomial b = 0
axiom root_c : polynomial c = 0
axiom distinct_ab : a ≠ b
axiom distinct_ac : a ≠ c
axiom distinct_bc : b ≠ c

-- Vieta's formulas conditions
axiom vieta_sum : a + b + c = 2
axiom vieta_product_sum : a * b + b * c + c * a = 3
axiom vieta_product : a * b * c = 4

-- The proof goal
theorem cyclic_sum_identity : 
    (1 / (a * (b^2 + c^2 - a^2))) + 
    (1 / (b * (c^2 + a^2 - b^2))) + 
    (1 / (c * (a^2 + b^2 - c^2))) = -1 / 8 := 
sorry

end cyclic_sum_identity_l162_162777


namespace det_matrix_divisible_by_k_pow_n_minus_1_l162_162053

noncomputable def determinant_of_matrix
  (n : ℕ) (a : Fin n → ℝ) (k : ℝ) : 
  matrix (Fin n) (Fin n) ℝ :=
  λ i j, if i = j then a i * a i + k else a i * a j

theorem det_matrix_divisible_by_k_pow_n_minus_1 (n : ℕ) (a : Fin n → ℝ) (k : ℝ) :
  matrix.det (determinant_of_matrix n a k) = k^n-1 * (k + (∑ i, a i * a i)) :=
by
  sorry

end det_matrix_divisible_by_k_pow_n_minus_1_l162_162053


namespace smallest_integer_log_sum_l162_162904

theorem smallest_integer_log_sum (a b : ℝ) (k : ℕ) (f : ℕ → ℝ) (P : ∏ k in finset.range 101, (f k) = 51) :
  a = 3 ∧ b = 4 → 3 < log a 51 ∧ log a 51 < b :=
by
  sorry

end smallest_integer_log_sum_l162_162904


namespace fraction_zero_x_value_l162_162724

theorem fraction_zero_x_value (x : ℝ) (h : (x^2 - 4) / (x - 2) = 0) (h2 : x ≠ 2) : x = -2 :=
sorry

end fraction_zero_x_value_l162_162724


namespace students_selected_strawberry_milk_l162_162234

theorem students_selected_strawberry_milk 
  (chocolate : ℕ) 
  (strawberry : ℕ) 
  (regular : ℕ) 
  (total : ℕ) 
  (h_chocolate : chocolate = 2) 
  (h_regular : regular = 3) 
  (h_total : total = 20) 
  (h_eq : chocolate + strawberry + regular = total) 
  : strawberry = 15 :=
by 
  have h1 : 2 + strawberry + 3 = 20 := by rw [h_chocolate, h_regular, h_total]
  have h2 : strawberry + 5 = 20 := by linarith
  have h3 : strawberry = 15 := by linarith
  exact h3

end students_selected_strawberry_milk_l162_162234


namespace calculate_a_over_b_l162_162622

noncomputable def system_solution (x y a b : ℝ) : Prop :=
  (8 * x - 5 * y = a) ∧ (10 * y - 15 * x = b) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧ (b ≠ 0)

theorem calculate_a_over_b (x y a b : ℝ) (h : system_solution x y a b) : a / b = 8 / 15 :=
by
  sorry

end calculate_a_over_b_l162_162622


namespace smallest_5_digit_palindrome_base3_l162_162165

def is_palindrome {α : Type} [DecidableEq α] (l : List α) : Prop :=
  l = l.reverse

def to_base (n : ℕ) (b : ℕ) : List ℕ :=
  let rec go (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else go (n / b) ((n % b) :: acc)
  go n []

-- The 5-digit palindrome in base 3 we are considering
def candidate_palindrome_3 := to_base 81 3

def candidate_5_digit_palindrome := 10001
def candidate_3_digit_palindrome_9 := 99

theorem smallest_5_digit_palindrome_base3 :
  ∃ (n : ℕ), is_palindrome (to_base n 3) ∧ 
  (n : ℕ) = candidate_palindrome_3 /\
  ∃ (b : ℕ), b ≠ 3 ∧ is_palindrome (to_base n b) ∧ (List.length (to_base n b) = 3) :=
  sorry

end smallest_5_digit_palindrome_base3_l162_162165


namespace count_valid_arrays_l162_162388

noncomputable def num_valid_arrays 
  (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1^2 + a2^2 + a3^2 + a4^2) * (a2^2 + a3^2 + a4^2 + a5^2) = 
  (a1 * a2 + a2 * a3 + a3 * a4 + a4 * a5)^2 ∧ 
  a1 < 2021 ∧ a2 < 2021 ∧ a3 < 2021 ∧ a4 < 2021 ∧ a5 < 2021 ∧ 
  a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧
  a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧
  a3 ≠ a4 ∧ a3 ≠ a5 ∧
  a4 ≠ a5

theorem count_valid_arrays : 
  ∃ n : ℕ, n = 424 ∧ 
  {p : (ℕ × ℕ × ℕ × ℕ × ℕ) // num_valid_arrays p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2}.finite.card = n :=
sorry

end count_valid_arrays_l162_162388


namespace area_PQR_l162_162896

def P : ℝ × ℝ := (-2, 2)
def Q : ℝ × ℝ := (8, 2)
def R : ℝ × ℝ := (6, -4)

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

theorem area_PQR : area_of_triangle P Q R = 30 :=
sorry

end area_PQR_l162_162896


namespace participants_last_year_l162_162087

open Real

theorem participants_last_year (m : ℝ) (h : m = m * (1 + 0.1)) : m / (1 + 0.1) = m / 1.1 :=
by
  rw [←Real.add_div, Real.div_one, mul_div_eq_div_mul, @mul_one ℝ _ m, add_comm] at h
  exact h

end participants_last_year_l162_162087


namespace limit_S_n_l162_162659

def S_n (n : ℕ) : ℝ :=
  if 0 < n then π * (4^(n + 1) - 1) / (4^n + 1) else 0

theorem limit_S_n : (filter.at_top.filter_limit (λ n, S_n n) (4 * π)) :=
by
  sorry

end limit_S_n_l162_162659


namespace Madeline_flower_problem_l162_162805

theorem Madeline_flower_problem
  (half_die : ∀ (n : ℕ), n / 2)
  (seeds_per_pack : ℕ := 25)
  (cost_per_pack : ℕ := 5)
  (money_spent : ℕ := 10) :
  ∃ (seeds : ℕ), seeds / 2 = 25 :=
by
  let packs := money_spent / cost_per_pack
  let total_seeds := packs * seeds_per_pack
  let expected_blooms := total_seeds / 2
  use total_seeds
  sorry

end Madeline_flower_problem_l162_162805


namespace jill_total_earnings_over_three_months_l162_162767

theorem jill_total_earnings_over_three_months :
  (let first_month_earnings := 30 * 10 in
   let second_month_earnings := 2 * first_month_earnings in
   let third_month_earnings := second_month_earnings / 2 in
   first_month_earnings + second_month_earnings + third_month_earnings = 1200) :=
by
  sorry

end jill_total_earnings_over_three_months_l162_162767


namespace sum_perpendiculars_equals_hypotenuse_l162_162111

theorem sum_perpendiculars_equals_hypotenuse
  {A B C D F H}
  (hABC_rt : ∠A B C = 90)
  (hAC_square : is_square (A, C))
  (hBC_square : is_square (B, C))
  (hD_perp : is_perpendicular D (segment(A, B)))
  (hF_perp : is_perpendicular F (segment(A, B)))
  (hAC : length(A, C) = length(C, D)) 
  (hBC : length(B, C) = length(C, E)): 
  length(D, H) + length(F, G) = length(B, C) :=
sorry

end sum_perpendiculars_equals_hypotenuse_l162_162111


namespace exponentiation_rule_example_l162_162503

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162503


namespace option_A_correct_l162_162128

theorem option_A_correct (x y : ℝ) (hy : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 :=
by
  sorry

end option_A_correct_l162_162128


namespace terminal_side_point_l162_162293

variable {ℝ : Type*} [LinOrderedField ℝ]

def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

theorem terminal_side_point (θ : ℝ) (m : ℝ) :
  (∀ x : ℝ, f x ≤ f θ) →
  f θ = 5 →
  ∃ m, (θ = Real.arctan (3 / 4) + n * 2 * π) ∧ (∀ k : ℤ, (f x = 5) → (P(4, m) = P(4, 3))))
sorry

end terminal_side_point_l162_162293


namespace sum_of_cousins_ages_l162_162421

theorem sum_of_cousins_ages :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧
    a * b = 36 ∧ c * d = 40 ∧ a + b + c + d + e = 33 :=
by
  sorry

end sum_of_cousins_ages_l162_162421


namespace smallest_perimeter_triangle_ABC_l162_162883

noncomputable def perimeter_of_triangle_ABC (x y : ℕ) := 2 * (x + y)

-- Let's define the conditions and the problem as a Lean theorem statement
theorem smallest_perimeter_triangle_ABC (x y : ℕ) :
  (x > 0 ∧ y > 5 ∧ y < 8 ∧ 2 * y = y + y) ∧ (BI = 8) ∧ (AB = AC) ∧ (BC = 2 * y) ∧
  (let midP := (BC / 2) in (BD^2 + midP^2 = BI^2) ∧ (AB = ax ∧ AC = ax)) →
  (∃ x y : ℕ, perimeter_of_triangle_ABC x y = 108) :=
sorry

end smallest_perimeter_triangle_ABC_l162_162883


namespace probability_square_product_l162_162985

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def count_favorable_outcomes : ℕ :=
  (1, 1) + (1, 4) + (2, 2) + (4, 1) + (3, 3) + (9, 1) + (2, 8) + (5, 5) + (6, 6) + (7, 7) + (8, 8)

theorem probability_square_product (t : ℕ) (d : ℕ) :
  let total_outcomes := 15 * 8 in 
  let favorable_outcomes := count_favorable_outcomes in
  let probability := favorable_outcomes / total_outcomes in
  (1 ≤ t ∧ t ≤ 15) ∧ (1 ≤ d ∧ d ≤ 8) →
  (∃ k, k * k = t * d) →
  (is_prime t ∨ is_square t ∨ is_prime d ∨ is_square d) →
  probability = 11 / 120 := 
by
  sorry

end probability_square_product_l162_162985


namespace cylinder_volume_eq_l162_162718

-- Define the radius and height
def radius (a : ℝ) : ℝ := a
def height (a : ℝ) : ℝ := a

-- Define the volume of the cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Prove that the volume of a cylinder with radius a and height a is π * a^3
theorem cylinder_volume_eq (a : ℝ) : volume (radius a) (height a) = π * a^3 := by
  sorry

end cylinder_volume_eq_l162_162718


namespace exponentiation_rule_example_l162_162504

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162504


namespace problem_l162_162674

-- The function f(x)
def f (a b A B : ℝ) (x : ℝ) : ℝ := 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x)

-- Statement requiring proof: given that f(x) >= 0 for all x, prove a^2 + b^2 <= 2 and A^2 + B^2 <= 1
theorem problem 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, f a b A B x ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := 
sorry

end problem_l162_162674


namespace length_of_one_side_of_hexagon_l162_162866

variable (P : ℝ) (n : ℕ)
-- Condition: perimeter P is 60 inches
def hexagon_perimeter_condition : Prop := P = 60
-- Hexagon has six sides
def hexagon_sides_condition : Prop := n = 6
-- The question asks for the side length
noncomputable def side_length_of_hexagon : ℝ := P / n

-- Prove that if a hexagon has a perimeter of 60 inches, then its side length is 10 inches
theorem length_of_one_side_of_hexagon (hP : hexagon_perimeter_condition P) (hn : hexagon_sides_condition n) :
  side_length_of_hexagon P n = 10 := by
  sorry

end length_of_one_side_of_hexagon_l162_162866


namespace maria_min_score_fourth_quarter_l162_162449

theorem maria_min_score_fourth_quarter (x : ℝ) :
  (82 + 77 + 78 + x) / 4 ≥ 85 ↔ x ≥ 103 :=
by
  sorry

end maria_min_score_fourth_quarter_l162_162449


namespace exp_eval_l162_162485

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162485


namespace octahedron_volume_l162_162074

theorem octahedron_volume (a b c : ℝ) (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = 5) :
  let volume := (1/3) * (a * c) * (b / 2)
  in 2 * volume = 10 :=
by
  have ha : a = 3 := h₀
  have hb : b = 4 := h₁
  have hc : c = 5 := h₂
  have volume_def : volume = (1/3) * (a * c) * (b / 2) := rfl
  sorry

end octahedron_volume_l162_162074


namespace relay_selection_ways_l162_162580

-- Define the problem conditions
def sprinters : Finset ℕ := {0, 1, 2, 3, 4, 5}  -- 6 sprinters labeled from 0 to 5

def first_leg_restriction (p : ℕ) : Prop := p ≠ 0  -- Sprinter A (0) cannot run first
def fourth_leg_restriction (p : ℕ) : Prop := p ≠ 1  -- Sprinter B (1) cannot run fourth

-- The main theorem statement
theorem relay_selection_ways :
  ∑ (a ∈ sprinters) (ha : first_leg_restriction a),
  ∑ (b ∈ (sprinters \ {a})) (hb : fourth_leg_restriction b),
  ∑ (c ∈ (sprinters \ {a, b})),
  ∑ (d ∈ (sprinters \ {a, b, c})),
  1 = 252 :=
by sorry

end relay_selection_ways_l162_162580


namespace min_value_of_expression_l162_162669

noncomputable def problem_statement (a b : ℝ) : Prop :=
  ln ((2 - b) / a) = 2 * a + 2 * b - 4

noncomputable def expression (a b : ℝ) : ℝ :=
  1 / a + 2 / b + 2 / (a * b)

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : problem_statement a b) :
  expression a b = (5 + 2 * Real.sqrt 6) / 2 :=
sorry

end min_value_of_expression_l162_162669


namespace rectangle_width_l162_162253

theorem rectangle_width (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 + y^2 = 25) : y = 3 := 
by 
  sorry

end rectangle_width_l162_162253


namespace factor_sum_l162_162983

theorem factor_sum : 
  (∃ d e, x^2 + 9 * x + 20 = (x + d) * (x + e)) ∧ 
  (∃ e f, x^2 - x - 56 = (x + e) * (x - f)) → 
  ∃ d e f, d + e + f = 19 :=
by
  sorry

end factor_sum_l162_162983


namespace set_intersection_complement_l162_162701

-- Define the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ ¬ x ∈ A}

-- Define the intersection of B and complement_U_A
def B_inter_complement_U_A : Set ℕ := B ∩ complement_U_A

-- The statement to prove: B ∩ complement_U_A = {6, 7}
theorem set_intersection_complement :
  B_inter_complement_U_A = {6, 7} := by sorry

end set_intersection_complement_l162_162701


namespace max_m_value_l162_162285

theorem max_m_value (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^4 + 16 * m + 8 = k * (k + 1)) : m ≤ 2 :=
sorry

end max_m_value_l162_162285


namespace select_n_based_on_expected_profit_prob_of_one_failure_is_correct_l162_162153

-- Define the given conditions
def company_conditions := {
  num_production_lines : ℕ := 3,
  prob_failure : ℝ := 1 / 3,
  salary_worker : ℝ := 10000,
  profit_no_failure : ℝ := 120000,
  profit_repaired_failure : ℝ := 80000,
  profit_unrepaired_failure : ℝ := 0,
  max_failures : ℕ := 1
}

-- Define the probabilities and expected profits under the given conditions
def prob_exactly_one_failure (C : company_conditions) : ℝ :=
  nat.choose C.num_production_lines 1 * (C.prob_failure) ^ 1 * (1 - C.prob_failure) ^ (C.num_production_lines - 1)

def expected_profit (C : company_conditions) (n : ℕ) : ℝ :=
  let p0 := (1 - C.prob_failure) ^ C.num_production_lines in -- Probability of no failure
  let p1 := nat.choose C.num_production_lines 1 * (C.prob_failure) ^ 1 * (1 - C.prob_failure) ^ (C.num_production_lines - 1) in -- Probability of exactly 1 failure
  let p2 := nat.choose C.num_production_lines 2 * (C.prob_failure) ^ 2 * (1 - C.prob_failure) ^ (C.num_production_lines - 2) in -- Probability of exactly 2 failures
  let p3 := (C.prob_failure) ^ 3 in -- Probability of exactly 3 failures
  (p0 * (3 * C.profit_no_failure - n * C.salary_worker) +
   p1 * (2 * C.profit_no_failure + C.profit_repaired_failure - n * C.salary_worker) +
   p2 * (C.profit_no_failure + 2 * C.profit_repaired_failure - n * C.salary_worker) +
   p3 * (2 * C.profit_repaired_failure - n * C.salary_worker)) / 1000 -- in thousand dollars

theorem select_n_based_on_expected_profit (C : company_conditions) : 
  expected_profit C 2 > expected_profit C 1 := 
sorry

-- State the theorem for the probability of exactly 1 failure
theorem prob_of_one_failure_is_correct (C : company_conditions) :
  prob_exactly_one_failure C = 4 / 9 := 
sorry

end select_n_based_on_expected_profit_prob_of_one_failure_is_correct_l162_162153


namespace ellipse_circle_no_intersection_probability_l162_162279

theorem ellipse_circle_no_intersection_probability (a b e : ℝ) 
    (h1 : a > b) (h2 : b > 0) (h3 : 0 < e) (h4 : e < 2)
    (h5 : c = sqrt (a^2 - b^2)) (h6 : c < b) :
    probability (0 < e ∧ e < sqrt 2 / 2) (0 < e ∧ e < 2) = sqrt 2 / 4 :=
sorry

end ellipse_circle_no_intersection_probability_l162_162279


namespace ellipse_equation_slope_property_l162_162259

-- Definitions
def ellipse_L (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_b_rel : a > b) : Prop :=
  ∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1)

def parabola_focus : (ℝ × ℝ) := (2, 0)

def point_on_ellipse (a : ℝ) (b : ℝ) : Prop :=
  (2 ^ 2 / a ^ 2 + (sqrt 2) ^ 2 / b ^ 2) = 1

-- The main statement
theorem ellipse_equation_slope_property :
  ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1))
     ∧ (2 ^ 2 / a ^ 2 + (sqrt 2) ^ 2 / b ^ 2 = 1)
     ∧ (a = 2 * sqrt 2) ∧ (b = 2) ∧ (∀ k b l_pos l_nparallel, ∀ M: ℝ × ℝ,
     (M.1 = - (2 * k * b) / (1 + 2 * k ^ 2)) ∧ (M.2 = (b / (1 + 2 * k ^ 2))) →
     - (1 / 2) * (1 / k) = -1 / 2) :=
  sorry

end ellipse_equation_slope_property_l162_162259


namespace sum_of_midpoint_coordinates_l162_162120

-- Defining the endpoints of the segment
def pointA : (ℝ × ℝ) := (10, 7)
def pointB : (ℝ × ℝ) := (4, -3)

-- Defining the midpoint coordinates
def midpoint : (ℝ × ℝ) := ((pointA.1 + pointB.1) / 2, (pointA.2 + pointB.2) / 2)

-- Proving that the sum of the coordinates of the midpoint is 9
theorem sum_of_midpoint_coordinates : midpoint.1 + midpoint.2 = 9 :=
by
  -- Proof will be written here; using sorry to skip it for now
  sorry

end sum_of_midpoint_coordinates_l162_162120


namespace cubic_polynomial_condition_l162_162158

theorem cubic_polynomial_condition (p : ℚ[X]) (h_cubic : p.degree = 3) 
  (h_conds : ∀ n ∈ ({1, 2, 3, 4, 5} : Finset ℚ), p.eval n = 1 / n^2) :
  p.eval 6 = 0 := 
sorry

end cubic_polynomial_condition_l162_162158


namespace positive_solutions_count_l162_162314

theorem positive_solutions_count :
  ∃ n : ℕ, n = 9 ∧
  (∀ (x y : ℕ), 5 * x + 10 * y = 100 → 0 < x ∧ 0 < y → (∃ k : ℕ, k < 10 ∧ n = 9)) :=
sorry

end positive_solutions_count_l162_162314


namespace smallest_seven_digit_number_divisible_by_127_l162_162118

theorem smallest_seven_digit_number_divisible_by_127 :
  ∃ n : ℕ, 1000000 ≤ n ∧ n % 127 = 0 ∧ (∀ m : ℕ, 1000000 ≤ m ∧ m % 127 = 0 → n ≤ m) :=
begin
  use 1000125,
  split,
  { exact nat.le_refl 1000125 },
  split,
  { norm_num },
  { intros m h1 h2,
    sorry -- Proof to show 1000125 is indeed the smallest number fulfilling the conditions.
  }
end

end smallest_seven_digit_number_divisible_by_127_l162_162118


namespace sequence_no_consecutive_ones_l162_162949

theorem sequence_no_consecutive_ones (m n : ℕ) (h : Nat.coprime m n) :
  (∃ b : ℕ → ℕ, b 1 = 2 ∧ b 2 = 3 ∧ (∀ n, b (n + 2) = b (n + 1) + b n) ∧
  let b12 := b 12 in
  m = b12 ∧ n = 2^12) →
  m + n = 4473 :=
by
  intros hb
  obtain ⟨b, hb1, hb2, hr, hb12, h2⟩ := hb
  sorry

end sequence_no_consecutive_ones_l162_162949


namespace sum_positive_real_solutions_l162_162999

theorem sum_positive_real_solutions :
  ∃ (s : ℝ), (∀ x : ℝ, (0 < x) → 2 * cos(2 * x) * (cos(2 * x) - cos(1007 * π^2 / x)) = cos(4 * x) - 1) ∧ s = 1080 * π :=
by sorry

end sum_positive_real_solutions_l162_162999


namespace exponentiation_example_l162_162476

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162476


namespace treasure_probability_l162_162167

def prob_treasure := (1 : ℚ) / 4
def prob_traps := (1 : ℚ) / 12
def prob_neither := (2 : ℚ) / 3
def num_islands := 8
def num_treasure_islands := 5

theorem treasure_probability :
  (∃ (n k : ℕ), 
    n = num_islands ∧ 
    k = num_treasure_islands ∧ 
    ( (Nat.choose n k : ℚ) * (prob_treasure ^ k) * (prob_neither ^ (n - k)) = 7 / 432)
  ) :=
sorry

end treasure_probability_l162_162167


namespace production_analysis_l162_162934

def daily_change (day: ℕ) : ℤ :=
  match day with
  | 0 => 40    -- Monday
  | 1 => -30   -- Tuesday
  | 2 => 90    -- Wednesday
  | 3 => -50   -- Thursday
  | 4 => -20   -- Friday
  | 5 => -10   -- Saturday
  | 6 => 20    -- Sunday
  | _ => 0     -- Invalid day, just in case

def planned_daily_production : ℤ := 500

def actual_production (day: ℕ) : ℤ :=
  planned_daily_production + (List.sum (List.map daily_change (List.range (day + 1))))

def total_production : ℤ :=
  List.sum (List.map actual_production (List.range 7))

theorem production_analysis :
  ∃ largest_increase_day smallest_increase_day : ℕ,
    largest_increase_day = 2 ∧  -- Wednesday
    smallest_increase_day = 1 ∧  -- Tuesday
    total_production = 3790 ∧
    total_production > 7 * planned_daily_production := by
  sorry

end production_analysis_l162_162934


namespace range_of_a_l162_162652

theorem range_of_a (a : ℝ) : (∃ (x : ℝ), x^2 - x + a = 0) → a ≥ -1/4 :=
by {
  intro h,
  have discriminant_nonneg : 1 + 4 * a ≥ 0,
  {
    cases h with x hx,
    let Δ := 1 + 4 * a,
    exact Δ_nonneg_of_quadratic_real_roots hx,
  },
  linarith,
}

end range_of_a_l162_162652


namespace average_speed_l162_162150

theorem average_speed {
  distance_flat : ℝ := 4.8,
  time_flat : ℝ := 33,
  distance_uphill : ℝ := 0.5,
  time_uphill : ℝ := 11,
  distance_downhill : ℝ := 2.2,
  time_downhill : ℝ := 18,
  miles_to_km : ℝ := 1.60934,
  total_distance_miles : ℝ := distance_flat + distance_uphill + distance_downhill,
  total_time_minutes : ℝ := time_flat + time_uphill + time_downhill,
  total_distance_km : ℝ := total_distance_miles * miles_to_km,
  total_time_hours : ℝ := total_time_minutes / 60
} : 
  total_distance_km / total_time_hours ≈ 11.68 := 
sorry

end average_speed_l162_162150


namespace correct_calculation_l162_162535

theorem correct_calculation :
  (∃ x : ℝ, x ^ 3 = -125 ∧ x = -5) ∧
  (¬ (∃ y : ℝ, y ^ 2 = (-5) ^ 2 ∧ y = -5)) ∧
  (¬ (√2 + √3 = √5)) ∧
  (¬ ((√5 + 1) ^ 2 = 6)) :=
by {
  sorry
}

end correct_calculation_l162_162535


namespace ratio_of_segments_l162_162342

theorem ratio_of_segments (a b c r s : ℝ) (h₁ : a : b = 2 : 5) (h₂ : c^2 = a^2 + b^2)
  (h₃ : r = a^2 / c) (h₄ : s = b^2 / c) : r / s = 4 / 25 :=
by
  sorry

end ratio_of_segments_l162_162342


namespace ball_hits_ground_at_time_l162_162855

theorem ball_hits_ground_at_time :
  ∃ t : ℝ, (-5 * t^2 + 4.6 * t + 4 = 0) ∧ (0 ≤ t) ∧ (t ≈ 1.4658) :=
by
  use 1.4658
  sorry

end ball_hits_ground_at_time_l162_162855


namespace satisfies_conditions_l162_162668

open Real

def point_P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

def condition1 (a : ℝ) : Prop := (point_P a).fst = 0

def condition2 (a : ℝ) : Prop := (point_P a).snd = 5

def condition3 (a : ℝ) : Prop := abs ((point_P a).fst) = abs ((point_P a).snd)

theorem satisfies_conditions :
  ∃ P : ℝ × ℝ, P = (12, 12) ∨ P = (-12, -12) ∨ P = (4, -4) ∨ P = (-4, 4) :=
by
  sorry

end satisfies_conditions_l162_162668


namespace exponentiation_identity_l162_162515

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162515


namespace SavingsInequality_l162_162911

theorem SavingsInequality (n : ℕ) : 52 + 15 * n > 70 + 12 * n := 
by sorry

end SavingsInequality_l162_162911


namespace polygon_sides_l162_162086

theorem polygon_sides (n : ℕ) (c : ℕ) 
  (h₁ : c = n * (n - 3) / 2)
  (h₂ : c = 2 * n) : n = 7 :=
sorry

end polygon_sides_l162_162086


namespace max_marks_l162_162915

variable (M : ℝ)

-- Conditions
def needed_to_pass (M : ℝ) := 0.20 * M
def pradeep_marks := 390
def marks_short := 25
def total_marks_needed := pradeep_marks + marks_short

-- Theorem statement
theorem max_marks : needed_to_pass M = total_marks_needed → M = 2075 := by
  sorry

end max_marks_l162_162915


namespace square_non_neg_eq_zero_iff_square_zero_l162_162329

theorem square_non_neg (a : ℝ) : a^2 ≥ 0 := 
by sorry

theorem eq_zero_iff_square_zero (a : ℝ) : a^2 = 0 ↔ a = 0 := 
by sorry

example (a : ℝ) : ¬ (∀ a : ℝ, a^2 > 0) :=
by {
  have h := @square_non_neg 0,
  linarith,
}

end square_non_neg_eq_zero_iff_square_zero_l162_162329


namespace factor_expression_l162_162632

theorem factor_expression (x y z : ℝ) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 ≠ 0 →
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
    (x + y) * (y + z) * (z + x) :=
by
  intro h
  sorry

end factor_expression_l162_162632


namespace distance_from_car_to_stream_l162_162760

-- Define the given distances as noncomputable
def total_distance : ℝ := 0.7
def distance_stream_to_meadow : ℝ := 0.4
def distance_meadow_to_campsite : ℝ := 0.1

-- Define the proof problem
theorem distance_from_car_to_stream : 
  let distance_from_car_to_stream := total_distance - distance_stream_to_meadow - distance_meadow_to_campsite in
  distance_from_car_to_stream = 0.2 :=
by
  -- This is where the proof would go
  sorry

end distance_from_car_to_stream_l162_162760


namespace pressure_force_on_cylindrical_glass_l162_162940

noncomputable def total_pressure_force_on_lateral_surface (h r ρ g : ℝ) : ℝ :=
  ρ * g * (0.08 * π) * (h^2 / 2)

theorem pressure_force_on_cylindrical_glass :
  total_pressure_force_on_lateral_surface 0.1 0.04 13600 9.81 = 167.6 := 
by
  -- (mathematical steps omitted)
  sorry

end pressure_force_on_cylindrical_glass_l162_162940


namespace average_salary_rest_of_workers_l162_162073

theorem average_salary_rest_of_workers
  (avg_salary_all : ℝ)
  (num_all_workers : ℕ)
  (avg_salary_techs : ℝ)
  (num_techs : ℕ)
  (avg_salary_rest : ℝ)
  (num_rest : ℕ) :
  avg_salary_all = 8000 →
  num_all_workers = 21 →
  avg_salary_techs = 12000 →
  num_techs = 7 →
  num_rest = num_all_workers - num_techs →
  avg_salary_rest = (avg_salary_all * num_all_workers - avg_salary_techs * num_techs) / num_rest →
  avg_salary_rest = 6000 :=
by
  intros h_avg_all h_num_all h_avg_techs h_num_techs h_num_rest h_avg_rest
  sorry

end average_salary_rest_of_workers_l162_162073


namespace trigonometric_identity_l162_162061

theorem trigonometric_identity (x y : ℝ) :
  (sin (x + y) * cos (2 * y) + cos (x + y) * sin (2 * y)) = sin (x + 3 * y) :=
by 
  sorry

end trigonometric_identity_l162_162061


namespace grazing_months_l162_162133

theorem grazing_months
    (total_rent : ℝ)
    (c_rent : ℝ)
    (a_oxen : ℕ)
    (a_months : ℕ)
    (b_oxen : ℕ)
    (c_oxen : ℕ)
    (c_months : ℕ)
    (b_months : ℝ)
    (total_oxen_months : ℝ) :
    total_rent = 140 ∧
    c_rent = 36 ∧
    a_oxen = 10 ∧
    a_months = 7 ∧
    b_oxen = 12 ∧
    c_oxen = 15 ∧
    c_months = 3 ∧
    c_rent / total_rent = (c_oxen * c_months) / total_oxen_months ∧
    total_oxen_months = (a_oxen * a_months) + (b_oxen * b_months) + (c_oxen * c_months)
    → b_months = 5 := by
    sorry

end grazing_months_l162_162133


namespace find_AB_l162_162004

-- Let our definitions based on the given conditions
def cos_C_over_2 : ℝ := (Real.sqrt 5) / 5
def BC : ℝ := 1
def AC : ℝ := 5

-- Main theorem statement to prove
theorem find_AB {AB : ℝ} :
  cos_C_over_2 = (Real.sqrt 5) / 5 ∧ BC = 1 ∧ AC = 5 → AB = 4 * Real.sqrt 2 :=
sorry

end find_AB_l162_162004


namespace andrew_spent_total_amount_l162_162595

/-- Conditions:
1. Andrew played a total of 7 games.
2. Cost distribution for games:
   - 3 games cost $9.00 each
   - 2 games cost $12.50 each
   - 2 games cost $15.00 each
3. Additional expenses:
   - $25.00 on snacks
   - $20.00 on drinks
-/
def total_cost_games : ℝ :=
  (3 * 9) + (2 * 12.5) + (2 * 15)

def cost_snacks : ℝ := 25
def cost_drinks : ℝ := 20

def total_spent (cost_games cost_snacks cost_drinks : ℝ) : ℝ :=
  cost_games + cost_snacks + cost_drinks

theorem andrew_spent_total_amount :
  total_spent total_cost_games 25 20 = 127 := by
  -- The proof is omitted
  sorry

end andrew_spent_total_amount_l162_162595


namespace area_of_MNFK_l162_162058

theorem area_of_MNFK (ABNF CMKD MNFK : ℝ) (BN : ℝ) (KD : ℝ) (ABMK : ℝ) (CDFN : ℝ)
  (h1 : BN = 8) (h2 : KD = 9) (h3 : ABMK = 25) (h4 : CDFN = 32) :
  MNFK = 31 :=
by
  have hx : 8 * (MNFK + 25) - 25 = 9 * (MNFK + 32) - 32 := sorry
  exact sorry

end area_of_MNFK_l162_162058


namespace average_last_12_results_l162_162847

theorem average_last_12_results (S25 S12 S_last12 : ℕ) (A : ℕ) 
  (h1 : S25 = 25 * 24) 
  (h2: S12 = 12 * 14) 
  (h3: 12 * A = S_last12)
  (h4: S25 = S12 + 228 + S_last12) : A = 17 := 
by
  sorry

end average_last_12_results_l162_162847


namespace ramu_profit_percent_l162_162825

theorem ramu_profit_percent :
  let cost_of_car := 42000
  let cost_of_repairs := 8000
  let selling_price := 64900
  let total_cost := cost_of_car + cost_of_repairs
  let profit := selling_price - total_cost
  (profit / total_cost * 100) = 29.8 :=
by
  -- Definitions
  let cost_of_car := (42000 : ℝ)
  let cost_of_repairs := (8000 : ℝ)
  let selling_price := (64900 : ℝ)
  let total_cost := cost_of_car + cost_of_repairs
  let profit := selling_price - total_cost
  
  -- Calculation
  have h_total_cost : total_cost = 50000 := by norm_num
  have h_profit : profit = 14900 := by norm_num
  have h_profit_percent : (profit / total_cost * 100) = 29.8 := by norm_num
  
  -- Assertion
  exact h_profit_percent

end ramu_profit_percent_l162_162825


namespace shortest_side_length_l162_162044

-- Define the problem conditions
def tangent_segments (a b : ℝ) (r : ℝ) :=
  a = 9 ∧ b = 15 ∧ r = 5

-- Define the lengths
def side_length_A (a b : ℝ) : ℝ := a + b
def side_length_B (x : ℝ) : ℝ := 2 * x
def side_length_C (x a : ℝ) : ℝ := x + a

-- Prove the shortest side length is 17 given the conditions
theorem shortest_side_length {a b r x : ℝ} 
  (h : tangent_segments a b r) 
  (radius_eq : r = 5) 
  (x_val : x = 8) : 
  side_length_C x a = 17 :=
by
  -- Extract the conditions
  rcases h with ⟨h_a, h_b, h_r⟩
  -- Substitute a = 9, b = 15, and x = 8, then simplify
  have a_eq : a = 9 := h_a
  have b_eq : b = 15 := h_b
  have x_eq : x = 8 := x_val
  have r_eq : r = 5 := h_r
  
  -- Compute side_length_C
  calc 
    side_length_C x a 
        = (x + a) : rfl
    ... = (8 + 9) : by rw [x_eq, a_eq]
    ... = 17 : rfl

end shortest_side_length_l162_162044


namespace connie_tickets_l162_162193

theorem connie_tickets (total_tickets earbuds_tickets bracelets_tickets koala_tickets : ℕ)
  (h_total : total_tickets = 50)
  (h_earbuds : earbuds_tickets = 10)
  (h_bracelets : bracelets_tickets = 15)
  (h_remaining : koala_tickets = total_tickets - (earbuds_tickets + bracelets_tickets)) :
  (koala_tickets:total_tickets) = 1:2 :=
by sorry

end connie_tickets_l162_162193


namespace find_c_for_same_solution_l162_162228

theorem find_c_for_same_solution (c : ℝ) (x : ℝ) :
  (3 * x + 5 = 1) ∧ (c * x + 15 = -5) → c = 15 :=
by
  sorry

end find_c_for_same_solution_l162_162228


namespace abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l162_162559

/-- Part 1: Prove that the number \overline{abba} is divisible by 11 -/
theorem abba_divisible_by_11 (a b : ℕ) : 11 ∣ (1000 * a + 100 * b + 10 * b + a) :=
sorry

/-- Part 2: Prove that the number \overline{aaabbb} is divisible by 37 -/
theorem aaabbb_divisible_by_37 (a b : ℕ) : 37 ∣ (1000 * 111 * a + 111 * b) :=
sorry

/-- Part 3: Prove that the number \overline{ababab} is divisible by 7 -/
theorem ababab_divisible_by_7 (a b : ℕ) : 7 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) :=
sorry

/-- Part 4: Prove that the number \overline{abab} - \overline{baba} is divisible by 9 and 101 -/
theorem abab_baba_divisible_by_9_and_101 (a b : ℕ) :
  9 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) ∧
  101 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) :=
sorry

end abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l162_162559


namespace angle_preserved_under_inversion_l162_162822

theorem angle_preserved_under_inversion 
  (l1 l2 : Line) (M : Point) (O : Point) :
  (intersect l1 l2 M) → 
  ∃ (l1' l2' : GeometricEntity), 
    (inversion l1 O = l1') ∧ (inversion l2 O = l2') ∧ (angle l1 l2 = angle l1' l2') :=
by
  sorry

end angle_preserved_under_inversion_l162_162822


namespace common_element_in_sets_l162_162247

theorem common_element_in_sets:
  ∃ a : α, (∀ i : fin 1978, a ∈ S i) :=
begin
  sorry
end

end common_element_in_sets_l162_162247


namespace david_august_tips_multiple_l162_162624

-- Definitions of conditions
def average_tips (total_tips : ℝ) (months : ℕ) : ℝ := total_tips / months

def total_tips (average_monthly_tips : ℝ) (months : ℕ) : ℝ := average_monthly_tips * months

-- The problem stated as a theorem
theorem david_august_tips_multiple (total_tips : ℝ) (A : ℝ) (x : ℝ) :
  (∀ A : ℝ, total_tips = 7 * A) →
  (∀ x : ℝ, total_tips = 6 * A + x * A) →
  (6 * A + x * A = total_tips) →
  (x * A = 0.4 * total_tips) →
  x = 2.8 :=
by
  intro h1 h2 h3 h4
  sorry

end david_august_tips_multiple_l162_162624


namespace problem_conditions_l162_162738

def arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 2) = a n + 2 * d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * ((a 0 + a (n - 1)) / 2)

variable {a : ℕ → ℝ}
variable (d : ℝ)
variable [has_arithmetic_seq : arithmetic_seq a]

theorem problem_conditions
  (h1 : a 1 + a 4 + a 7 = 15)
  (h2 : a 8 = 6)
  (h3 : a 0 > 0)
  (h4 : sum_of_first_n_terms a 3 = sum_of_first_n_terms a 8) :
  (a 2 + a 6 = 10) ∧ 
  ∃ d : ℝ, ∀ n : ℕ, (sum_of_first_n_terms a (n + 1)) / (n + 1) = (d / 2) * n + (a 0 - d / 2) :=
  sorry

end problem_conditions_l162_162738


namespace least_prime_factor_of_expression_l162_162899

theorem least_prime_factor_of_expression :
  Nat.minFactor (9^4 - 9^3) = 2 :=
by
  sorry

end least_prime_factor_of_expression_l162_162899


namespace additional_length_correct_l162_162173

def additionalTrackLength (elevation : ℝ) (initial_grade final_grade : ℝ) : ℝ :=
  let lengthAtInitialGrade := elevation / initial_grade
  let lengthAtFinalGrade := elevation / final_grade
  lengthAtFinalGrade - lengthAtInitialGrade

theorem additional_length_correct :
  additionalTrackLength 1200 0.04 0.03 = 10000 :=
by {
  simp [additionalTrackLength],
  norm_num,
  sorry
}

end additional_length_correct_l162_162173


namespace infinite_solutions_exists_l162_162140

theorem infinite_solutions_exists :
  ∃ (x : ℕ → ℤ), (∑ i in (range 10), (x i)^3 = 600) ∧ (∀ n : ℤ, x 6 = 0 ∧ x 7 = 6 ∧ x 8 = n ∧ x 9 = -n) :=
by
  sorry

end infinite_solutions_exists_l162_162140


namespace cryptarithm_solution_l162_162064

theorem cryptarithm_solution :
  ∃ (Γ О Р А Н В К : ℕ),
    (Γ ≠ О ∧ Γ ≠ Р ∧ Γ ≠ А ∧ Γ ≠ Н ∧ Γ ≠ В ∧ Γ ≠ К ∧
     О ≠ Р ∧ О ≠ А ∧ О ≠ Н ∧ О ≠ В ∧ О ≠ К ∧
     Р ≠ А ∧ Р ≠ Н ∧ Р ≠ В ∧ Р ≠ К ∧
     А ≠ Н ∧ А ≠ В ∧ А ≠ К ∧
     Н ≠ В ∧ Н ≠ К ∧
     В ≠ К) ∧
    (0 ≤ Γ ∧ Γ ≤ 9 ∧ 0 ≤ О ∧ О ≤ 9 ∧ 0 ≤ Р ∧ Р ≤ 9 ∧
     0 ≤ А ∧ А ≤ 9 ∧ 0 ≤ Н ∧ Н ≤ 9 ∧ 0 ≤ В ∧ В ≤ 9 ∧ 0 ≤ К ∧ К ≤ 9) ∧
    (Γ = 6 ∧ О = 9 ∧ Р = 4 ∧ А = 7 ∧ Н = 0 ∧ В = 1 ∧ К = 8) ∧
    (1000 * Γ + 100 * О + 10 * Р + А + 10000 * О + 1000 * Г + 100 * О + 10 * Н + В = 100000 * В + 10000 * У + 1000 * Л + 100 * К + 10 * А + Н) :=
begin
  use [6, 9, 4, 7, 0, 1, 8],
  repeat {split},
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  { intro h, linarith },
  repeat {linarith},
  sorry
end

end cryptarithm_solution_l162_162064


namespace find_e_l162_162392

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_e (r s d e : ℝ) 
  (h1 : quadratic 2 (-4) (-6) r = 0)
  (h2 : quadratic 2 (-4) (-6) s = 0)
  (h3 : r + s = 2) 
  (h4 : r * s = -3)
  (h5 : d = -(r + s - 6))
  (h6 : e = (r - 3) * (s - 3)) : 
  e = 0 :=
sorry

end find_e_l162_162392


namespace sufficient_condition_for_reciprocal_inequality_l162_162272

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) (h : b < a ∧ a < 0) : (1 / a) < (1 / b) :=
sorry

end sufficient_condition_for_reciprocal_inequality_l162_162272


namespace value_of_a_l162_162721

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 2 → x^2 - x + a < 0) → a = -2 :=
by
  intro h
  sorry

end value_of_a_l162_162721


namespace sugar_solution_sweeter_l162_162731

variables (a b m : ℝ)

theorem sugar_solution_sweeter (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  (a / b < (a + m) / (b + m)) :=
sorry

end sugar_solution_sweeter_l162_162731


namespace cosB_is_half_sinA_sinC_is_three_fourths_l162_162560

-- Definitions of the given conditions in a)
variables {A B C a b c : ℝ}
axiom triangleAngles : A + B + C = 180
axiom arithmeticSequence : 2 * B = A + C
axiom geometricSequence : b^2 = a * c

-- Proof goal for part (1)
theorem cosB_is_half : cos B = 1 / 2 := 
by 
  sorry

-- Proof goal for part (2)
theorem sinA_sinC_is_three_fourths (h: cos B = 1 / 2): sin A * sin C = 3 / 4 := 
by 
  sorry

end cosB_is_half_sinA_sinC_is_three_fourths_l162_162560


namespace max_value_expr_max_l162_162380

noncomputable def max_value_expr (x : ℝ) : ℝ :=
  (x^2 + 3 - (x^4 + 9).sqrt) / x

theorem max_value_expr_max (x : ℝ) (hx : 0 < x) :
  max_value_expr x ≤ (6 * (6:ℝ).sqrt) / (6 + 3 * (2:ℝ).sqrt) :=
sorry

end max_value_expr_max_l162_162380


namespace mrs_browns_sail_number_l162_162037

theorem mrs_browns_sail_number (a b : ℕ) :
  ∃ (n : ℕ) (child_ages : Fin 7 → ℕ),
    n = a * 1000 + a * 100 + b * 10 + b ∨ n = a * 1000 + b * 100 + a * 10 + b ∧
    b = 5 ∧
    ∑ i, child_ages i = 35 ∧ -- since the ages are distinct and sum up considering 5+4+3+2+1 and two extras summing 20.
    (∀ (i j : Fin 7), i ≠ j → child_ages i ≠ child_ages j) ∧ 
    ((child_ages 0 = 5) ∧ ∀ i : Fin 7, child_ages i ∤ n) → 
    ∃ k : Fin 7, child_ages k = 4 :=
begin
  sorry
end

end mrs_browns_sail_number_l162_162037


namespace integral_inequalities_l162_162446

theorem integral_inequalities : 
  (∫ x in 0..1, Real.sqrt x < ∫ x in 0..1, 3 * x) ∧
  (∫ x in 0..(Real.pi / 4), Real.sin x < ∫ x in 0..(Real.pi / 4), Real.cos x) ∧
  (∫ x in 0..1, Real.exp (-x) < ∫ x in 0..1, Real.exp ( - x^2 )) ∧
  (∫ x in 0..2, Real.sin x < ∫ x in 0..2, x) :=
by
  -- Proofs omitted
  sorry

end integral_inequalities_l162_162446


namespace age_difference_is_56_l162_162550

def age_problem (A B C D : ℕ) : Prop :=
  (A + B = B + C + 11) ∧
  (A + C = C + D + 15) ∧
  (B + D = 36) ∧
  (A = D * D) ∧
  (A = 64) ∧ (B = 28) ∧ (C = 53) ∧ (D = 8) →
  (A - D = 56)

theorem age_difference_is_56 : ∃ A B C D : ℕ, age_problem A B C D :=
by
  existsi 64, 28, 53, 8
  unfold age_problem
  split
  all_goals { try { sorry } }

end age_difference_is_56_l162_162550


namespace unit_vector_orthogonal_l162_162989

def vector1 : ℝ^3 := ⟨2, 1, 1⟩
def vector2 : ℝ^3 := ⟨3, 0, 1⟩
def cross_product := ⟨1, 1, -3⟩
def magnitude := real.sqrt (1^2 + 1^2 + (-3)^2)
def unit_vector := ⟨1 / magnitude, 1 / magnitude, -3 / magnitude⟩

theorem unit_vector_orthogonal :
  (vector1 ⬝ unit_vector = 0) ∧
  (vector2 ⬝ unit_vector = 0) ∧
  (real.sqrt (unit_vector.1^2 + unit_vector.2^2 + unit_vector.3^2) = 1) :=
sorry

end unit_vector_orthogonal_l162_162989


namespace fraction_c_d_l162_162623

theorem fraction_c_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) :
  c / d = -8 / 15 :=
sorry

end fraction_c_d_l162_162623


namespace group_initial_men_count_l162_162071

theorem group_initial_men_count
  (A : ℕ) (M : ℕ) :
  (∀ (A : ℕ) (M : ℕ),
    let old_men_total_age := M * A,
    let removed_men_total_age := 26 + 30,
    let women_average_age := 42,
    let new_average_age := A + 4,
    let new_total_age := (M - 2) * A + 2 * 42 in
    old_men_total_age - removed_men_total_age + 2 * women_average_age = M * new_average_age) →
  M = 7 :=
by 
  sorry

end group_initial_men_count_l162_162071


namespace difference_between_floor_solutions_l162_162993

theorem difference_between_floor_solutions :
  let largest_solution := max {x // \lfloor x / 3 \rfloor = 102}.to_finset :=
  let smallest_solution := min {x // \lfloor x / 3 \rfloor = -102}.to_finset :=
  largest_solution - smallest_solution = 614 :=
by
  sorry

end difference_between_floor_solutions_l162_162993


namespace percentage_increase_in_expenses_l162_162163

-- Declare constants for man's salary, usual savings, and new savings
def S : ℝ := 6000
def usual_savings : ℝ := 0.2 * S
def new_savings : ℝ := 240
def reduction_in_savings : ℝ := usual_savings - new_savings
def P : ℝ

-- Define the statement to prove the percentage increase in expenses
theorem percentage_increase_in_expenses :
  (P / 100) * S = reduction_in_savings → P = 16 :=
by
  sorry  -- Proof to be completed

end percentage_increase_in_expenses_l162_162163


namespace ratio_of_areas_l162_162620

-- Define the properties of the large square and the smaller squares.
def side_length_of_small_square : ℝ := 1
def number_of_small_squares_per_side : ℕ := 5

-- Define the properties of the shaded square which is formed by connecting the centers of four adjacent smaller squares.
def side_length_of_shaded_square : ℝ := real.sqrt (side_length_of_small_square ^ 2 + side_length_of_small_square ^ 2)

-- Calculate the areas of the shaded square and the large square.
def area_of_shaded_square : ℝ := side_length_of_shaded_square ^ 2
def area_of_large_square : ℝ := (number_of_small_squares_per_side * side_length_of_small_square) ^ 2

-- Prove the ratio of the area of the shaded square to the area of the large square.
theorem ratio_of_areas : (area_of_shaded_square / area_of_large_square) = 2 / 25 := 
by
  -- (Proof steps would go here)
  sorry

end ratio_of_areas_l162_162620


namespace eggs_left_for_breakfast_l162_162884

theorem eggs_left_for_breakfast :
  let total_eggs := 36
      crepes_eggs := (2 * total_eggs) / 5
      remaining_eggs_after_crepes := total_eggs - crepes_eggs
      cupcakes_eggs := (3 * remaining_eggs_after_crepes) / 7
      remaining_eggs_after_cupcakes := remaining_eggs_after_crepes - cupcakes_eggs
      quiche_eggs := remaining_eggs_after_cupcakes / 2
      remaining_eggs := remaining_eggs_after_cupcakes - quiche_eggs
  in crepes_eggs = Int.floor 14 ∧ cupcakes_eggs = Int.floor 9 ∧ quiche_eggs = Int.floor 6 → remaining_eggs = 7 :=
by
  sorry

end eggs_left_for_breakfast_l162_162884


namespace relative_starts_l162_162336

-- Define the participants
inductive Participant
| A : Participant
| B : Participant
| C : Participant
| D : Participant

-- Define the relative start function
def relative_start (p1 p2 : Participant) : ℝ

-- Given conditions
axiom a_gives_b_start : relative_start Participant.A Participant.B = 120
axiom a_gives_c_start : relative_start Participant.A Participant.C = 250
axiom a_gives_d_start : relative_start Participant.A Participant.D = 320

-- Prove the resulting starts
theorem relative_starts :
  relative_start Participant.B Participant.C = 130 ∧
  relative_start Participant.B Participant.D = 200 ∧
  relative_start Participant.C Participant.D = 70 ∧
  -- reverse relative starts
  relative_start Participant.B Participant.A = -120 ∧
  relative_start Participant.C Participant.A = -250 ∧
  relative_start Participant.D Participant.A = -320 ∧
  relative_start Participant.C Participant.B = -130 ∧
  relative_start Participant.D Participant.B = -200 ∧
  relative_start Participant.D Participant.C = -70 :=
by
  sorry

end relative_starts_l162_162336


namespace monotonic_increasing_interval_cosine_l162_162869

theorem monotonic_increasing_interval_cosine :
  ∀ (x : ℝ), x ∈ set.Icc (- (Real.pi / 2)) (Real.pi / 2) →
  monotonic_increasing_on (λ x, 2 * Real.cos (2 * x - Real.pi / 4))
  (set.Icc (- 3 * Real.pi / 8) (Real.pi / 8)) :=
by
  sorry

end monotonic_increasing_interval_cosine_l162_162869


namespace problem_C_plus_D_l162_162101

theorem problem_C_plus_D (C D : ℚ)
  (h : ∀ x, (D * x - 17) / (x^2 - 8 * x + 15) = C / (x - 3) + 5 / (x - 5)) :
  C + D = 5.8 :=
sorry

end problem_C_plus_D_l162_162101


namespace rhombus_area_l162_162425

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 30) (h2 : d2 = 16) : (d1 * d2) / 2 = 240 := by
  sorry

end rhombus_area_l162_162425


namespace real_part_of_complex_div_l162_162556

noncomputable def complexDiv (c1 c2 : ℂ) := c1 / c2

theorem real_part_of_complex_div (i_unit : ℂ) (h_i : i_unit = Complex.I) :
  (Complex.re (complexDiv (2 * i_unit) (1 + i_unit)) = 1) :=
by
  sorry

end real_part_of_complex_div_l162_162556


namespace sin_cos_max_min_correct_l162_162245

noncomputable def sin_cos_max_min (x y : ℝ) (h : Real.sin x + Real.sin y = 1 / 3) : ℝ × ℝ :=
  let μ := Real.sin y + Real.cos x ^ 2
  have h1: Real.sin y = 1 / 3 - Real.sin x, by sorry
  let t := Real.sin x
  let μ_expr := -t^2 - t + 4 / 3
  let max_val := (19 / 12: ℝ)
  let min_val := (-2 / 3: ℝ)
  (max_val, min_val)

theorem sin_cos_max_min_correct (x y : ℝ) (h : Real.sin x + Real.sin y = 1 / 3) :
  sin_cos_max_min x y h = (19 / 12, -2 / 3) :=
  sorry

end sin_cos_max_min_correct_l162_162245


namespace mean_elements_geq_half_n_plus_one_l162_162672

theorem mean_elements_geq_half_n_plus_one
  (m n : ℕ) 
  (m_pos : 0 < m)
  (n_pos : 0 < n)
  (a : Fin m → ℕ) 
  (dist : Function.Injective a)
  (elem_range : ∀ i, a i ∈ Finset.range n.succ)
  (condition : ∀ i j : Fin m, i ≤ j → a i + a j ≤ n → ∃ k : Fin m, a i + a j = a k) :
  (∑ i, a i : ℝ) / m ≥ (n + 1 : ℝ) / 2 := by
sorry

end mean_elements_geq_half_n_plus_one_l162_162672


namespace quadratic_expression_value_l162_162020

theorem quadratic_expression_value:
  (x1 x2 : ℝ) 
  (h1: x1 + x2 = 3) 
  (h2: x1 * x2 = 1) :
  x1^2 + 3 * x1 * x2 + x2^2 = 10 :=
by
  sorry

end quadratic_expression_value_l162_162020


namespace imaginary_part_of_z_is_1_l162_162863

-- Define the imaginary unit i
def i : ℂ := ⟨0, 1⟩

-- Define the complex number z in the problem
def z : ℂ := 2 / (1 - i)

-- The statement of the problem in Lean 4
theorem imaginary_part_of_z_is_1 : z.im = 1 :=
sorry

end imaginary_part_of_z_is_1_l162_162863


namespace gideon_fraction_of_marbles_l162_162241

variable (f : ℝ)

theorem gideon_fraction_of_marbles (marbles : ℝ) (age_now : ℝ) (age_future : ℝ) (remaining_marbles : ℝ) (future_age_with_remaining_marbles : Bool)
  (h1 : marbles = 100)
  (h2 : age_now = 45)
  (h3 : age_future = age_now + 5)
  (h4 : remaining_marbles = 2 * (1 - f) * marbles)
  (h5 : remaining_marbles = age_future)
  (h6 : future_age_with_remaining_marbles = (age_future = 50)) :
  f = 3 / 4 :=
by
  sorry

end gideon_fraction_of_marbles_l162_162241


namespace distance_between_stripes_l162_162951

theorem distance_between_stripes
  (h1 : ∀ (curbs_are_parallel : Prop), curbs_are_parallel → true)
  (h2 : ∀ (distance_between_curbs : ℝ), distance_between_curbs = 60 → true)
  (h3 : ∀ (length_of_curb : ℝ), length_of_curb = 20 → true)
  (h4 : ∀ (stripe_length : ℝ), stripe_length = 75 → true) :
  ∃ (d : ℝ), d = 16 :=
by
  sorry

end distance_between_stripes_l162_162951


namespace product_of_differences_of_squares_is_diff_of_square_l162_162372

-- Define when an integer is a difference of squares of positive integers
def diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ n = x^2 - y^2

-- State the main theorem
theorem product_of_differences_of_squares_is_diff_of_square 
  (a b c d : ℕ) (h₁ : diff_of_squares a) (h₂ : diff_of_squares b) (h₃ : diff_of_squares c) (h₄ : diff_of_squares d) : 
  diff_of_squares (a * b * c * d) := by
  sorry

end product_of_differences_of_squares_is_diff_of_square_l162_162372


namespace cos_diff_alpha_l162_162319

theorem cos_diff_alpha (α : ℝ) (h1 : cos (π / 3 + α) = 1 / 3) (h2 : π / 2 < α ∧ α < 3 * π / 2) :
  cos (π / 6 - α) = -2 * real.sqrt 2 / 3 :=
sorry

end cos_diff_alpha_l162_162319


namespace chess_team_combination_l162_162045

theorem chess_team_combination 
  (players : Finset ℕ) (quadruplets : Finset ℕ) 
  (h_players : players.card = 18) 
  (h_quadruplets : quadruplets.card = 4) 
  (h_team : quadruplets ⊆ players) :
  ∃ (num_ways : ℕ), num_ways = (Nat.choose 14 4) ∧ num_ways = 1001 :=
by
  sorry

end chess_team_combination_l162_162045


namespace power_of_powers_eval_powers_l162_162527

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162527


namespace jerry_avg_speed_l162_162360

variable {min_per_hour : ℝ}
variable {beth_avg_speed : ℝ}
variable {beth_route_longer : ℝ}
variable {jerry_trip_time : ℝ}
variable {beth_extra_time : ℝ}
variable {beth_trip_distance : ℝ}
variable {jerry_trip_distance : ℝ}

theorem jerry_avg_speed : 
  (min_per_hour = 60) →
  (beth_avg_speed = 30) →
  (beth_route_longer = 5) →
  (jerry_trip_time = 30 / min_per_hour) →
  (beth_extra_time = 20 / min_per_hour) →
  (beth_trip_distance = beth_avg_speed * (jerry_trip_time + beth_extra_time)) →
  (jerry_trip_distance = beth_trip_distance - beth_route_longer) →
  (jerry_trip_distance = 20) →
  (jerry_trip_time > 0) →
  jerry_trip_distance / jerry_trip_time = 40 := 
by {
  intros,
  sorry
}

end jerry_avg_speed_l162_162360


namespace increase_C_l162_162196

variable (m e P p : ℝ)
variable (hme, hP, hp : 0 < m ∧ 0 < e ∧ 0 < P ∧ 0 < p)

def C (m e P p : ℝ) : ℝ := m * e / (P + m * p)

theorem increase_C (m1 m2 e P p : ℝ) 
  (hme : 0 < m1) (hme2 : 0 < m2) (he : 0 < e) (hP : 0 < P) (hp : 0 < p)
  (h_m : m1 < m2) :
  C m1 e P p < C m2 e P p := by
  sorry

end increase_C_l162_162196


namespace necessary_but_not_sufficient_l162_162300

-- Define \(\frac{1}{x} < 2\) and \(x > \frac{1}{2}\)
def condition1 (x : ℝ) : Prop := 1 / x < 2
def condition2 (x : ℝ) : Prop := x > 1 / 2

-- Theorem stating that condition1 is necessary but not sufficient for condition2
theorem necessary_but_not_sufficient (x : ℝ) : condition1 x → condition2 x ↔ true :=
sorry

end necessary_but_not_sufficient_l162_162300


namespace find_x_in_gp_l162_162987

theorem find_x_in_gp :
  ∃ x : ℤ, (30 + x)^2 = (10 + x) * (90 + x) ∧ x = 0 :=
by
  sorry

end find_x_in_gp_l162_162987


namespace irrational_c_has_exactly_one_solution_l162_162236

noncomputable theory

open Real

def has_exactly_one_solution (c : ℝ) : Prop :=
  ∀ x₀ : ℝ, (1 + sin (c * x₀) ^ 2 = cos x₀) → x₀ = 0

theorem irrational_c_has_exactly_one_solution (c : ℝ) (h_irr : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ c = p / q) :
  has_exactly_one_solution c :=
by
  sorry

end irrational_c_has_exactly_one_solution_l162_162236


namespace false_option_C_l162_162695

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem false_option_C : 
  ¬ (∀ g (g = λ (x : ℝ), Real.sin (2 * x)), 
    ∃ h, (h = λ (x : ℝ), Real.sin (2 * (x + Real.pi / 6))) ∧
         ∀ x, f x = h (x - Real.pi / 3))
:= by
  -- Add proof here
  sorry

end false_option_C_l162_162695


namespace eccentricity_of_ellipse_l162_162684

-- Definitions:
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def focus_distance (a b c : ℝ) : Prop :=
  c^2 = a^2 - b^2

def circle_passing_points (a b c : ℝ) : Prop :=
  let r := (a + c) / 2 in
  true

def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Theorem to prove:
theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : ellipse a b x y) (h2 : focus_distance a b c) (h3 : circle_passing_points a b c) :
  eccentricity a c = (Real.sqrt 5 - 1) / 2 :=
by sorry

end eccentricity_of_ellipse_l162_162684


namespace maria_novel_stamp_collection_l162_162145

structure Stamps :=
  (China : ℕ)
  (Japan : ℕ)
  (Canada : ℕ)
  (Mexico : ℕ)

def cost_per_stamp : String → ℕ 
| "China"   => 7
| "Japan"   => 7
| "Canada"  => 3
| "Mexico"  => 4
| _         => 0

def total_cost_in_dollars (stamps : Stamps) : ℕ → ℕ → Float 
| cost_per_stamp "Canada", cost_per_stamp "Mexico" =>
    (stamps.Canada * 3 + stamps.Mexico * 4) / 100.0
| _ => 0.0

def stamps_1960s := Stamps.mk 5 6 7 8
def stamps_1970s := Stamps.mk 9 7 6 5

def NorthAmerican_Stamps_from_both_decades : Stamps :=
  Stamps.mk 0 0 (stamps_1960s.Canada + stamps_1970s.Canada) (stamps_1960s.Mexico + stamps_1970s.Mexico)

theorem maria_novel_stamp_collection : 
    total_cost_in_dollars NorthAmerican_Stamps_from_both_decades (cost_per_stamp "Canada") (cost_per_stamp "Mexico") = 0.91 :=
  sorry

end maria_novel_stamp_collection_l162_162145


namespace snow_white_seven_piles_l162_162427

def split_pile_action (piles : List ℕ) : Prop :=
  ∃ pile1 pile2, pile1 > 0 ∧ pile2 > 0 ∧ pile1 + pile2 + 1 ∈ piles

theorem snow_white_seven_piles :
  ∃ piles : List ℕ, piles.length = 7 ∧ ∀ pile ∈ piles, pile = 3 :=
sorry

end snow_white_seven_piles_l162_162427


namespace diameter_of_circumscribed_circle_l162_162325

theorem diameter_of_circumscribed_circle 
  (b : ℝ) (angle_B : ℝ) (b_value : b = 15) (angle_B_value : angle_B = 45) (sin_45 : Real.sin (angle_B * Real.pi / 180) = Real.sqrt 2 / 2) : 
  ∃ D : ℝ, D = b / (Real.sin (angle_B * Real.pi / 180)) ∧ D = 15 * Real.sqrt 2 :=
by
  use b / (Real.sin (angle_B * Real.pi / 180))
  split
  sorry
  sorry

end diameter_of_circumscribed_circle_l162_162325


namespace parallel_lines_solution_l162_162328

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y + 1 = 0 → 2 * x + a * y + 2 = 0 → (a = 1 ∨ a = -2)) :=
by
  sorry

end parallel_lines_solution_l162_162328


namespace john_safe_weight_l162_162770

-- Assuming the conditions provided that form the basis of our problem.
def max_capacity : ℝ := 1000
def safety_margin : ℝ := 0.20
def john_weight : ℝ := 250
def safe_weight (max_capacity safety_margin john_weight : ℝ) : ℝ := 
  (max_capacity * (1 - safety_margin)) - john_weight

-- The main theorem to prove based on the provided problem statement.
theorem john_safe_weight : safe_weight max_capacity safety_margin john_weight = 550 := by
  -- skipping the proof details as instructed
  sorry

end john_safe_weight_l162_162770


namespace exponentiation_example_l162_162473

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162473


namespace hyperbola_focal_length_fractional_inequality_solution_set_ellipse_trajectory_eq_hyperbola_distance_to_foci_sequence_a_n_hyperbola_chord_eq_l162_162558

-- Problem 1
theorem hyperbola_focal_length : 
  let a := sqrt 7
  let b := sqrt 3
  2 * sqrt (a^2 + b^2) = 2 * sqrt 10 := 
by sorry

-- Problem 2
theorem fractional_inequality_solution_set : 
  { x : ℝ | 0 ≤ x ∧ x < 1 } = [0, 1) := 
by sorry

-- Problem 3
theorem ellipse_trajectory_eq :
  let F₁ := (-4, 0)
  let F₂ := (4, 0)
  let d := 10
  (∀ M, |M - F₁| + |M - F₂| = d) ↔ 
  (∀ M, M ∈ { P : ℝ × ℝ | P.1^2 / 25 + P.2^2 / 9 = 1 }) := 
by sorry

-- Problem 4
theorem hyperbola_distance_to_foci (P : ℝ × ℝ) : 
  let H := { P : ℝ × ℝ  | P.1^2 / 16 - P.2^2 / 9 = 1 }
  let F₁ := (-4, 0)
  let F₂ := (4, 0)
  let d₁ := 10
  (P ∈ H ∧ |P - F₁| = d₁) → |P - F₂| = 18 := 
by sorry

-- Problem 5
theorem sequence_a_n : 
  let a_1 := 1
  ({a : ℕ → ℝ | ∀ n ≥ 2, n * a n = (n - 1) * a (n - 1)}).nth 2017 = 1 / 2017 := 
by sorry

-- Problem 6
theorem hyperbola_chord_eq (m n : ℝ) (H : m > 0 ∧ n > 0) (Hmn : 1 / m + 1 / n = 2) : 
  (m + n = 2) ∧ let M := (1,1), H := let A := { P : ℝ × ℝ  | P.1^2 / 4 - P.2^2 / 2 = 1 } in
  let mid := ∀ M ∈ A M / 2 = 1 in
  let slope := (1 / 2) → y - 1 = (1 / 2) * (x - 1) :=
by sorry

end hyperbola_focal_length_fractional_inequality_solution_set_ellipse_trajectory_eq_hyperbola_distance_to_foci_sequence_a_n_hyperbola_chord_eq_l162_162558


namespace probability_inequality_l162_162040

open Probability

theorem probability_inequality :
  let outcomes : List (ℕ × ℕ) := List.product (List.range 9) (List.range 9)
  let event := outcomes.filter (λ (ab : ℕ × ℕ), ab.1 - 2 * ab.2 + 10 > 0)
  (event.length : ℚ) / (outcomes.length : ℚ) = 61 / 81 :=
by
  sorry

end probability_inequality_l162_162040


namespace sqrt_a_plus_b_is_3_l162_162682

theorem sqrt_a_plus_b_is_3
  (a b : ℝ)
  (h1 : sqrt (2 * a - 1) = 3 ∨ sqrt (2 * a - 1) = -3)
  (h2 : (3 * a + 2 * b + 4)^(1/3) = 3) :
  sqrt (a + b) = 3 := by
  -- Proof goes here
  sorry

end sqrt_a_plus_b_is_3_l162_162682


namespace andy_start_problem_l162_162957

theorem andy_start_problem (n : ℕ) (total_solved : ℕ) (last_problem : ℕ) (start : ℕ) :
  last_problem = 125 → total_solved = 56 → start = (last_problem - total_solved + 1) → start = 70 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3.symm
  sorry

end andy_start_problem_l162_162957


namespace exponentiation_example_l162_162472

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162472


namespace maximum_possible_value_of_e_l162_162378

noncomputable def b (n : ℕ) : ℤ := (10^n - 2) / 8

def e (n : ℕ) : ℤ := Int.gcd (b n) (b (n + 2))

theorem maximum_possible_value_of_e : ∀ n : ℕ, ∃ k : ℤ, e n = k ∧ k = 1 := by
  intros n
  use 1
  sorry

end maximum_possible_value_of_e_l162_162378


namespace slopes_identity_l162_162687

noncomputable def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4

structure Point where
  x : ℝ
  y : ℝ

def M := Point.mk (-2) 0
def N := Point.mk 2 0
def P := Point.mk (-2) 2

def slope (p1 p2 : Point) : ℝ := 
  if p2.x - p1.x = 0 then 0 else (p2.y - p1.y) / (p2.x - p1.x)

axiom A_in_first_quadrant (A B : Point) : ellipse A.x A.y ∧ ellipse B.x B.y ∧ 0 < A.x ∧ 0 < A.y ∧ 0 < B.x ∧ 0 < B.y 
axiom A_on_BP (A B : Point) : A.x >= B.x ∧ A.y >= B.y 
axiom OP_intersects_NA (A C : Point) : C.x = (2 * A.y) / (A.x + A.y - 2) ∧ C.y = (2 * A.y) / (2 - A.x - A.y)

def k_AM (A : Point) : ℝ := slope A M
def k_AC (A C : Point) : ℝ := slope A C
def k_MB (B : Point) : ℝ := slope M B
def k_MC (C : Point) : ℝ := slope M C

theorem slopes_identity (A B C : Point)
  (h1 : ellipse A.x A.y) (h2 : ellipse B.x B.y)
  (h3 : A_on_BP A B)
  (h4 : OP_intersects_NA A C) :
  (k_MB B) / (k_AM A) = (k_AC A C) / (k_MC C) := by
  sorry

end slopes_identity_l162_162687


namespace domain_of_sqrt_sin_and_sqrt_cos_l162_162660

open real

noncomputable def domain_of_function (x : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * π ≤ x ∧ x ≤ π / 3 + 2 * k * π

theorem domain_of_sqrt_sin_and_sqrt_cos (x : ℝ) :
  (sqrt (sin x) + sqrt (cos x - 1 / 2)).domain = { x | domain_of_function x } :=
sorry

end domain_of_sqrt_sin_and_sqrt_cos_l162_162660


namespace polar_coordinates_of_point_l162_162198

noncomputable def point_rectangular_to_polar (x y : ℝ) : ℝ × ℝ := 
  let r := Real.sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  (r, θ)

theorem polar_coordinates_of_point :
  point_rectangular_to_polar 1 (-1) = (Real.sqrt 2, 7 * Real.pi / 4) :=
by
  unfold point_rectangular_to_polar
  sorry

end polar_coordinates_of_point_l162_162198


namespace power_calc_l162_162465

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162465


namespace coloring_impossible_l162_162357

theorem coloring_impossible :
  let grid_size := 8 in
  let initial_color := (fun _ _ => false) in
  let neighboring_gray (g : ℕ -> ℕ -> bool) (x y : ℕ) :=
    (if x > 0 then g (x - 1) y else false) +
    (if x < grid_size - 1 then g (x + 1) y else false) +
    (if y > 0 then g x (y - 1) else false) +
    (if y < grid_size - 1 then g x (y + 1) else false) in
  
  ∀ g : (ℕ → ℕ → bool),
  (∃ i j, g i j = true ∧ ((i ≠ 0 ∨ j ≠ 0) → g 0 0 = true)) →
  (∀ x y, g x y = true → (neighboring_gray g x y = 1 ∨ neighboring_gray g x y = 3)) →
  (∃ g_prime : ℕ → ℕ → bool, (∀ x y, g x y = true → g_prime x y = true) ∧ (∃ x y, g_prime x y = false) )
  sorry

end coloring_impossible_l162_162357


namespace least_number_to_make_divisible_l162_162126

theorem least_number_to_make_divisible :
  let n := 4499 * 17
  let remainder := n % 23
  23 - remainder = 15 := 
by
  let n := 4499 * 17
  have remainder : n % 23 = 8 := sorry
  have needed_addition : 23 - remainder = 15 := by
    calc
      23 - 8 = 15 : by sorry
  exact needed_addition

end least_number_to_make_divisible_l162_162126


namespace Sammy_has_8_bottle_caps_l162_162414

-- Definitions representing the conditions
def BilliesBottleCaps := 2
def JaninesBottleCaps := 3 * BilliesBottleCaps
def SammysBottleCaps := JaninesBottleCaps + 2

-- Goal: Prove that Sammy has 8 bottle caps
theorem Sammy_has_8_bottle_caps : 
  SammysBottleCaps = 8 := 
sorry

end Sammy_has_8_bottle_caps_l162_162414


namespace adjacent_cells_large_n_diff_l162_162894

theorem adjacent_cells_large_n_diff (n : ℕ) (h : n ≥ 2) :
  ∃ (A B : ℕ), A ≠ B ∧ abs (A - B) ≥ (2 - Real.sqrt 2) * n ∧
  ∃ (i j : ℕ), 1 ≤ i ∧ i < n ∧ j < n ∧ 1 ≤ j ∧ -- indices of the cells
  abs ((i * n + j + 1) - (A * n + B + 1)) = 1 := 
sorry

end adjacent_cells_large_n_diff_l162_162894


namespace susan_ate_6_candies_l162_162975

-- Definitions based on the problem conditions
def candies_tuesday := 3
def candies_thursday := 5
def candies_friday := 2
def candies_left := 4

-- The total number of candies bought
def total_candies_bought := candies_tuesday + candies_thursday + candies_friday

-- The number of candies eaten
def candies_eaten := total_candies_bought - candies_left

-- Theorem statement to prove that Susan ate 6 candies
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  -- Proof will be provided here
  sorry

end susan_ate_6_candies_l162_162975


namespace range_of_x_plus_y_minimum_AB_l162_162740

noncomputable def line_parametric (t : ℝ) (α : ℝ) (hα : 0 ≤ α ∧ α < Real.pi) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

def curve_cartesian (x y : ℝ) : Prop :=
  x^2 = 4 * y

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def curve_polar (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ * Real.cos θ = 4 * Real.sin θ

theorem range_of_x_plus_y (x y : ℝ) (h : curve_cartesian x y) : -1 ≤ x + y := by
  sorry

theorem minimum_AB (t₁ t₂ : ℝ) (α : ℝ) (hα : 0 ≤ α ∧ α < Real.pi)
  (h₁ : curve_cartesian ((line_parametric t₁ α hα).1) ((line_parametric t₁ α hα).2))
  (h₂ : curve_cartesian ((line_parametric t₂ α hα).1) ((line_parametric t₂ α hα).2)) : 
  abs ((line_parametric t₁ α hα).1 - (line_parametric t₂ α hα).1) * (Real.sqrt (1 + (Real.tan α)^2)) ≥ 4 := by
  sorry

end range_of_x_plus_y_minimum_AB_l162_162740


namespace intersection_and_area_max_l162_162302

noncomputable def parametric_eq (θ : ℝ) := 
  (⟨-2 + 2 * Real.cos θ, 2 * Real.sin θ⟩ : ℝ × ℝ)

def polar_eq (θ : ℝ) := 4 * Real.sin θ

theorem intersection_and_area_max 
  (p1 : ∃ θ₁, parametric_eq θ₁ = ⟨0, 0⟩) 
  (p2 : ∃ θ₂, parametric_eq θ₂ = ⟨-2, 2⟩)
  (a : ℝ) (b : ℝ) :
  (a, b) ∈ {⟨0, 0⟩, ⟨-2, 2⟩} ∧ 
  max_area_triangle a b := 
  ⟨{⟨0, 0⟩, ⟨-2, 2⟩}, 2 + 2 * Real.sqrt 2⟩ :=
sorry

end intersection_and_area_max_l162_162302


namespace greatest_possible_points_top_three_l162_162344

noncomputable def tournament_total_points : ℕ := 168

theorem greatest_possible_points_top_three :
    ∃ p : ℕ, 
    (∀ (A B C : ℕ), p = A ∧ p = B ∧ p = C ∧ 
    (A + B + C ≤ tournament_total_points) ∧ 
    A ≤ tournament_total_points − 56 ∧ 
    B ≤ tournament_total_points − 56 ∧ 
    C ≤ tournament_total_points − 56) ∧
    p = 36 :=
by 
    let A := 36
    let B := 36
    let C := 36
    have total_points := 168
    have points_distribution := 3 * 56 = 168
    use 36
    split
    sorry

end greatest_possible_points_top_three_l162_162344


namespace moment_of_inertia_unit_masses_moment_of_inertia_general_masses_l162_162141

-- Define part (a)
theorem moment_of_inertia_unit_masses (n : ℕ) (a : (ℕ × ℕ) → ℝ) : 
  (I_O = (1 / n) * ∑ i in finset.range n, ∑ j in finset.range i, a (i, j) ^ 2) :=
sorry

-- Define part (b)
theorem moment_of_inertia_general_masses (n : ℕ) (m : ℝ) (m_i : ℕ → ℝ) (a : (ℕ × ℕ) → ℝ) :
  (I_O = (1 / m) * ∑ i in finset.range n, ∑ j in finset.range i, m_i i * m_i j * a (i, j) ^ 2) :=
sorry

end moment_of_inertia_unit_masses_moment_of_inertia_general_masses_l162_162141


namespace largest_polygon_area_l162_162585

def area_polygon_A : ℝ := 3 * 1 + 2 * 0.5
def area_polygon_B : ℝ := 4 * 1 + 2 * (1 * Real.sqrt 2 / 2)
def area_polygon_C : ℝ := 2 * 1 + 4 * 0.5
def area_polygon_D : ℝ := 6 * 1
def area_polygon_E : ℝ := 3 * 1 + 0.5 * 2 * 2

theorem largest_polygon_area :
  max (max (max (max area_polygon_A area_polygon_B) area_polygon_C) area_polygon_D) area_polygon_E = area_polygon_D :=
by
  sorry

end largest_polygon_area_l162_162585


namespace greatest_drop_is_third_quarter_l162_162103

def priceStart (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 10
  | 2 => 7
  | 3 => 9
  | 4 => 5
  | _ => 0 -- default case for invalid quarters

def priceEnd (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 7
  | 2 => 9
  | 3 => 5
  | 4 => 6
  | _ => 0 -- default case for invalid quarters

def priceChange (quarter : ℕ) : ℤ :=
  priceStart quarter - priceEnd quarter

def greatestDropInQuarter : ℕ :=
  if priceChange 1 > priceChange 3 then 1
  else if priceChange 2 > priceChange 1 then 2
  else if priceChange 3 > priceChange 4 then 3
  else 4

theorem greatest_drop_is_third_quarter :
  greatestDropInQuarter = 3 :=
by
  -- proof goes here
  sorry

end greatest_drop_is_third_quarter_l162_162103


namespace cup_percentage_l162_162453

theorem cup_percentage (C : ℝ) (h1 : C > 0) (h2 : 0 < 2 / 3 * C ≤ C) (h3 : 6 ≠ 0): 
  ((2 / 3 * C / 6) / C) * 100 = 100 / 9 :=
by
  sorry

end cup_percentage_l162_162453


namespace ABC_is_isosceles_or_rectangle_l162_162356

theorem ABC_is_isosceles_or_rectangle
  (A B C D E : Type)
  [Geometry A]
  (hD : midpoint B C = D)
  (hE : orthogonal_projection (line_through A D) C = E)
  (h_angle : angle A C E = angle A B C) : 
  is_isosceles_or_rectangle A B C :=
sorry

end ABC_is_isosceles_or_rectangle_l162_162356


namespace exponentiation_example_l162_162470

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162470


namespace find_k_l162_162076

-- Definitions based on given conditions
def ellipse_equation (x y : ℝ) (k : ℝ) : Prop :=
  5 * x^2 + k * y^2 = 5

def is_focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = 2

-- Statement of the problem
theorem find_k (k : ℝ) :
  (∀ x y, ellipse_equation x y k) →
  is_focus 0 2 →
  k = 1 :=
sorry

end find_k_l162_162076


namespace purely_imaginary_complex_l162_162235

theorem purely_imaginary_complex :
  ∀ (x y : ℤ), (x - 4) ≠ 0 → (y^2 - 3*y - 4) ≠ 0 → (∃ (z : ℂ), z = ⟨0, x^2 + 3*x - 4⟩) → 
    (x = 4 ∧ y ≠ 4 ∧ y ≠ -1) :=
by
  intro x y hx hy hz
  sorry

end purely_imaginary_complex_l162_162235


namespace cone_height_l162_162848

theorem cone_height (a α : ℝ) (hα : 0 < α ∧ α < real.pi / 2) :
  let BC := a / real.sin α,
      r := BC / 2,
      MN := a / 2,
      LN := (a * real.cot α) / (2 * real.sin α),
      ML := real.sqrt ((LN ^ 2) + (MN ^ 2)) in
  ML = (a * real.sqrt (real.cot α ^ 2 + real.sin α ^ 2)) / (2 * real.sin α) :=
by
  -- insert proof here
  sorry

end cone_height_l162_162848


namespace trigonometric_identity_l162_162654

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (2 * α) + Real.sin (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α) = -1 :=
by
  sorry

end trigonometric_identity_l162_162654


namespace arccos_of_sqrt3_div_2_l162_162614

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l162_162614


namespace jill_three_months_income_l162_162764

theorem jill_three_months_income :
  let FirstMonthIncome := 10 * 30 in
  let SecondMonthIncome := (10 * 2) * 30 in
  let ThirdMonthIncome := (10 * 2) * (30 / 2) in
  FirstMonthIncome + SecondMonthIncome + ThirdMonthIncome = 1200 :=
by
  sorry

end jill_three_months_income_l162_162764


namespace guitar_sequence_count_l162_162708

theorem guitar_sequence_count : 
  let S := "GUITAR".toList in
  let valid_sequences := {seq : List Char | 
    seq.head? = some 'R' ∧ seq.getLast? = some 'T' ∧ 
    seq.length = 5 ∧ (seq ⊆ S) ∧ (seq.nodup)} 
  in valid_sequences.card = 24 :=
sorry

end guitar_sequence_count_l162_162708


namespace range_of_a_l162_162689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x >= 1 then (1/2)^x - 1 else (a - 2) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f a x1 > f a x2) →
  (1/2) ≤ a ∧ a < 2 :=
by
  intro h
  have h1 : a - 2 < 0 := by sorry
  have h2 : 1/2 - 1 ≤ a - 2 + 1 := by sorry
  split
  · linarith
  · exact h1

end range_of_a_l162_162689


namespace volume_of_cylindrical_tin_l162_162135

-- Define the given conditions
def diameter : ℝ := 6
def height : ℝ := 5

-- Define auxiliary definitions based on the conditions
def radius : ℝ := diameter / 2
def volume : ℝ := π * radius^2 * height

-- Statement to prove
theorem volume_of_cylindrical_tin : |volume - 141.37| < 0.01 :=
by
  sorry

end volume_of_cylindrical_tin_l162_162135


namespace selection_methods_l162_162953

theorem selection_methods (fitters turners master_workers : Nat)
  (h_fitters : fitters = 5)
  (h_turners : turners = 4)
  (h_master_workers : master_workers = 2) :
  ∃ (methods : Nat), methods = 185 :=
by
  let C := Nat.choose
  have scenario1 : Nat := C 7 4
  have scenario2 : Nat := C 4 3 * C 2 1 * C 6 4
  have scenario3 : Nat := C 4 2 * C 5 4 * C 2 2
  let total_methods := scenario1 + scenario2 + scenario3
  use total_methods
  sorry

end selection_methods_l162_162953


namespace round_124_496_to_nearest_integer_l162_162829

theorem round_124_496_to_nearest_integer : Int.floor (124.496 + 0.5) = 124 := 
by
  sorry

end round_124_496_to_nearest_integer_l162_162829


namespace cyclic_quadrilateral_eq_l162_162597

theorem cyclic_quadrilateral_eq (A B C D : ℝ) (AB AD BC DC : ℝ)
  (h1 : AB = AD) (h2 : based_on_laws_of_cosines) : AC ^ 2 = BC * DC + AB ^ 2 :=
sorry

end cyclic_quadrilateral_eq_l162_162597


namespace cos_2x_eq_cos_2y_l162_162826

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end cos_2x_eq_cos_2y_l162_162826


namespace domain_of_f_lg_x_l162_162327

theorem domain_of_f_lg_x : 
  ({x : ℝ | -1 ≤ x ∧ x ≤ 1} = {x | 10 ≤ x ∧ x ≤ 100}) ↔ (∃ f : ℝ → ℝ, ∀ x ∈ {x : ℝ | -1 ≤ x ∧ x ≤ 1}, f (x * x + 1) = f (Real.log x)) :=
sorry

end domain_of_f_lg_x_l162_162327


namespace bill_has_six_times_more_nuts_l162_162601

-- Definitions for the conditions
def sue_has_nuts : ℕ := 48
def harry_has_nuts (sueNuts : ℕ) : ℕ := 2 * sueNuts
def combined_nuts (harryNuts : ℕ) (billNuts : ℕ) : ℕ := harryNuts + billNuts
def bill_has_nuts (totalNuts : ℕ) (harryNuts : ℕ) : ℕ := totalNuts - harryNuts

-- Statement to prove
theorem bill_has_six_times_more_nuts :
  ∀ sueNuts billNuts harryNuts totalNuts,
    sueNuts = sue_has_nuts →
    harryNuts = harry_has_nuts sueNuts →
    totalNuts = 672 →
    combined_nuts harryNuts billNuts = totalNuts →
    billNuts = bill_has_nuts totalNuts harryNuts →
    billNuts = 6 * harryNuts :=
by
  intros sueNuts billNuts harryNuts totalNuts hsueNuts hharryNuts htotalNuts hcombinedNuts hbillNuts
  sorry

end bill_has_six_times_more_nuts_l162_162601


namespace angle_between_clock_hands_at_half_past_eight_l162_162849

theorem angle_between_clock_hands_at_half_past_eight :
  angle_between_hour_and_minute_hands 8 30 = 75 :=
sorry

end angle_between_clock_hands_at_half_past_eight_l162_162849


namespace part1_inequality_solution_part2_inequality_solution_l162_162658

-- Part 1: Range of m

theorem part1_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ (-4 < m ∧ m ≤ 0) ∨ m = 0 :=
by sorry

-- Part 2: Solution to the inequality

theorem part2_inequality_solution (m : ℝ) :
  (m < -4 → ∀ x : ℝ, (x < (m + real.sqrt (m^2 + 4 * m)) / (2 * m) 
                 ∨ (m - real.sqrt (m^2 + 4 * m)) / (2 * m) < x ↔ 
       m * x^2 - m * x - 1 < 0)) ∧
  (m = -4 → ∀ x : ℝ, (x ≠ 1 / 2 → m * x^2 - m * x - 1 < 0)) ∧
  (-4 < m ∧ m ≤ 0 → ∀ x : ℝ, (m * x^2 - m * x - 1 < 0)) ∧
  (m > 0 → ∀ x : ℝ, ((m - real.sqrt (m^2 + 4 * m)) / (2 * m) < x ∧ 
                      x < (m + real.sqrt (m^2 + 4 * m)) / (2 * m)) ↔ 
       m * x^2 - m * x - 1 < 0) :=
by sorry

end part1_inequality_solution_part2_inequality_solution_l162_162658


namespace least_element_of_S_is_4_l162_162376

theorem least_element_of_S_is_4 :
  ∃ S : Finset ℕ, S.card = 7 ∧ (S ⊆ Finset.range 16) ∧
  (∀ {a b : ℕ}, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)) ∧
  (∀ T : Finset ℕ, T.card = 7 → (T ⊆ Finset.range 16) →
  (∀ {a b : ℕ}, a ∈ T → b ∈ T → a < b → ¬ (b % a = 0)) →
  ∃ x : ℕ, x ∈ T ∧ x = 4) :=
by
  sorry

end least_element_of_S_is_4_l162_162376


namespace arctan_combination_correct_l162_162617

noncomputable def arctan_combination_identity : ℝ :=
  let x : ℝ := 75
  let y : ℝ := 15
  let z : ℝ := 45
  in real.arctan (real.tan (x * real.pi / 180) - 3 * real.tan (y * real.pi / 180) + real.tan (z * real.pi / 180))

theorem arctan_combination_correct : 
  arctan_combination_identity = 30 * real.pi / 180 := by
  -- Proof to be inserted
  sorry

end arctan_combination_correct_l162_162617


namespace television_final_price_l162_162178

theorem television_final_price :
  let original_price := 1200
  let discount_percent := 0.30
  let tax_percent := 0.08
  let rebate := 50
  let discount := discount_percent * original_price
  let sale_price := original_price - discount
  let tax := tax_percent * sale_price
  let price_including_tax := sale_price + tax
  let final_amount := price_including_tax - rebate
  final_amount = 857.2 :=
by
{
  -- The proof would go here, but it's omitted as per instructions.
  sorry
}

end television_final_price_l162_162178


namespace sin_theta_value_l162_162377

variables {a b c : ℝ × ℝ × ℝ} -- Declare variables a, b, and c as 3D vectors

-- Define the norm ||a|| of a 3D vector a
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the cross product of two 3D vectors
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- Define the dot product of two 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the sine of the angle between two vectors
def sin_theta (u v : ℝ × ℝ × ℝ) : ℝ :=
  norm (cross_product u v) / (norm u * norm v)

-- Prove that sin θ = 4/7 given the conditions
theorem sin_theta_value
  (h_a : norm a = 1)
  (h_b : norm b = 7)
  (h_c : norm c = 4)
  (h_eq : cross_product a (cross_product a b) = c) :
  sin_theta a b = 4 / 7 :=
by
  sorry -- Proof placeholder

end sin_theta_value_l162_162377


namespace energy_calculation_l162_162180

noncomputable def stormy_day_energy_production 
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (proportional_increase : ℝ) : ℝ :=
  proportional_increase * (energy_per_day * days * number_of_windmills)

theorem energy_calculation
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (wind_speed_proportion : ℝ)
  (stormy_day_energy_per_windmill : ℝ) (s : ℝ)
  (H1 : energy_per_day = 400) 
  (H2 : days = 2) 
  (H3 : number_of_windmills = 3) 
  (H4 : stormy_day_energy_per_windmill = s * energy_per_day)
  : stormy_day_energy_production energy_per_day days number_of_windmills s = s * (400 * 3 * 2) :=
by
  sorry

end energy_calculation_l162_162180


namespace new_tax_rate_is_30_percent_l162_162720

theorem new_tax_rate_is_30_percent
  (original_rate : ℝ)
  (annual_income : ℝ)
  (tax_saving : ℝ)
  (h1 : original_rate = 0.45)
  (h2 : annual_income = 48000)
  (h3 : tax_saving = 7200) :
  (100 * (original_rate * annual_income - tax_saving) / annual_income) = 30 := 
sorry

end new_tax_rate_is_30_percent_l162_162720


namespace card_arrangement_problem_l162_162047

-- Given the conditions
def cards : Finset ℕ := {1, 2, 3, 4, 5, 6}
def boxes : Finset ℕ := {1, 2, 3, 4}

-- Define a function to count valid arrangements
def count_valid_arrangements : ℕ := 
  let total_ways := 
    (Fintype.card (Finset.powersetLen 2 cards)) * 
    (Fintype.card (Finset.powersetLen 2 (cards \ {3, 6}))) 
    * (boxes.card).choose(4) in
  let exclude_ways := 
    (Fintype.card (Finset.powersetLen 5 (cards \ {3, 6}))) 
    * (boxes.card).choose(4) in
  total_ways - exclude_ways

-- The statement of the problem
theorem card_arrangement_problem : count_valid_arrangements = 1320 := by
  sorry

end card_arrangement_problem_l162_162047


namespace exists_nat_N_l162_162627

theorem exists_nat_N :
  ∃ N : ℕ, (2^2023 ∣ N ∧ ¬ (2^2024 ∣ N)) ∧ (card (finset.of_list (N.digits 10)) = 3 ∧ (¬ 0 ∈ (finset.of_list (N.digits 10)))) ∧ (∃ q : ℕ, q > (N.digits 10).length * 999 / 1000 ∧ ∀ d ∈ N.digits 10, d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9) :=
begin
  sorry
end

end exists_nat_N_l162_162627


namespace line_plane_relation_l162_162332

open Geometry

variables {Point Line Plane : Type}
variables (l : Line) (a : Line) (α : Plane)
variable dist : Point → Line → ℝ
variable within_plane : Line → Plane → Prop
variable equal_distances : ∃ (P1 P2 : Point), (dist P1 a = dist P2 a) ∧ (l contains P1) ∧ (l contains P2)

theorem line_plane_relation (hl : within_plane l α) (ha : within_plane a α) (heq_dist : equal_distances) :
  (within_plane l α) ∨ (∃ (P : Point), (l intersects α at P)) ∨ (exists_in_plane l α) := 
sorry

end line_plane_relation_l162_162332


namespace power_calc_l162_162462

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162462


namespace number_of_super_balanced_permutations_number_of_super_balanced_permutations_with_p1_l162_162775

def is_super_balanced (N : ℕ) (p : Fin N → ℕ) : Prop :=
  ∀ (l r : Fin N), r - l ≥ 3 →
    (¬ (p l = finset.min' (finset.image p (finset.Icc l r)) (by sorry)) ∨
     ¬ (p r = finset.max' (finset.image p (finset.Icc l r)) (by sorry)))

noncomputable def super_balanced_count (N : ℕ) : ℕ :=
  2^(N-1)

noncomputable def super_balanced_count_with_p1 (N : ℕ) (M : ℕ) : ℕ :=
  Nat.choose (N - 1) (M - 1)

theorem number_of_super_balanced_permutations (N : ℕ) (hN : 0 < N) :
  ∃ (count : ℕ), count = super_balanced_count N :=
by
  use super_balanced_count N
  sorry

theorem number_of_super_balanced_permutations_with_p1 (N M : ℕ) (hN : 0 < N) (hM : M ≤ N) :
  ∃ (count : ℕ), count = super_balanced_count_with_p1 N M :=
by
  use super_balanced_count_with_p1 N M
  sorry

end number_of_super_balanced_permutations_number_of_super_balanced_permutations_with_p1_l162_162775


namespace smallest_perimeter_is_23_l162_162584

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def are_consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧ b = a + 2 ∧ c = b + 2

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_perimeter_is_23 : 
  ∃ (a b c : ℕ), are_consecutive_odd_primes a b c ∧ satisfies_triangle_inequality a b c ∧ is_prime (a + b + c) ∧ (a + b + c) = 23 :=
by
  sorry

end smallest_perimeter_is_23_l162_162584


namespace positive_difference_of_quadratic_solutions_l162_162091

theorem positive_difference_of_quadratic_solutions :
  ∀ (x : ℝ), (x^2 - 5 * x + 11 = x + 53) → 
    let a := 1
    let b := -6
    let c := -42
    let d := b^2 - 4*a*c
    let root1 := (3 + Real.sqrt 51)
    let root2 := (3 - Real.sqrt 51)
    (|root1 - root2| = 2 * Real.sqrt 51) :=
begin
  intros,
  sorry
end

end positive_difference_of_quadratic_solutions_l162_162091


namespace asymptotes_of_hyperbola_l162_162274

theorem asymptotes_of_hyperbola 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (F : ℝ × ℝ := (c, 0)) 
  (symmetric_point_condition : ∃ (x₀ y₀ : ℝ), (y₀ / 2 = (1 / 3) * (x₀ + c) / 2) ∧ (y₀ / (x₀ - c) = -3) ∧ ((x₀, y₀) ∈ {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2 - p.2 ^ 2 / b ^ 2 = 1})):
  asymptote_equation : ∃ (m : ℝ), (m = sqrt (6) / 2 ∧ (∀ (x y : ℝ), (y = m * x ∨ y = -m * x))) :=
sorry

end asymptotes_of_hyperbola_l162_162274


namespace exists_2002_consecutive_integers_with_150_primes_l162_162833

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_count_in_range (n k : ℕ) : ℕ :=
List.length (List.filter is_prime (List.range k).map (fun i => i + n))

theorem exists_2002_consecutive_integers_with_150_primes :
  (∃ n, prime_count_in_range n 2002 = 150) :=
begin
  -- Proof will go here
  sorry
end

end exists_2002_consecutive_integers_with_150_primes_l162_162833


namespace smallest_x_undefined_l162_162202

theorem smallest_x_undefined :
  (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1 ∨ x = 8) → (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1) :=
by
  sorry

end smallest_x_undefined_l162_162202


namespace find_lambda_perpendicular_l162_162310

noncomputable def vector_m (λ : ℝ) : ℝ × ℝ := (λ, 1)
noncomputable def vector_n (λ : ℝ) : ℝ × ℝ := (λ + 1, 2)

theorem find_lambda_perpendicular :
  ∀ (λ : ℝ), let m := vector_m λ in
             let n := vector_n λ in
             (m.1 + n.1, m.2 + n.2) ⬝ (m.1 - n.1, m.2 - n.2) = 0 → λ = -2 :=
by
  intros λ m n h
  sorry

end find_lambda_perpendicular_l162_162310


namespace sum_sequence_conjecture_l162_162700

theorem sum_sequence_conjecture (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ+, a n = (8 * n) / ((2 * n - 1) ^ 2 * (2 * n + 1) ^ 2)) →
  (∀ n : ℕ+, S n = (S n + a (n + 1))) →
  (∀ n : ℕ+, S 1 = 8 / 9) →
  (∀ n : ℕ+, S n = ((2 * n + 1) ^ 2 - 1) / (2 * n + 1) ^ 2) :=
by {
  sorry
}

end sum_sequence_conjecture_l162_162700


namespace length_of_string_l162_162571

theorem length_of_string (circumference height : ℝ) (loops : ℕ) 
  (h_circumference : circumference = 5) (h_height : height = 20) (h_loops : loops = 5) : 
  ∃ (length : ℝ), length = 5 * Real.sqrt 41 :=
by 
  use 5 * Real.sqrt 41
  sorry

end length_of_string_l162_162571


namespace shell_arrangements_count_l162_162364

def num_shell_arrangements_equiv : ℕ :=
  39916800

theorem shell_arrangements_count (n : ℕ) (k : ℕ) (star_points : Finset (Fin 12)) 
  (unique_shells : Finset (Fin 12)) (h1: star_points.card = 12)
  (h2: unique_shells.card = 12) :
  n = 12! / 12 → 
  n = num_shell_arrangements_equiv :=
sorry

end shell_arrangements_count_l162_162364


namespace angle_solution_exists_l162_162618

theorem angle_solution_exists :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ 9 * (Real.sin x) * (Real.cos x)^4 - 9 * (Real.sin x)^4 * (Real.cos x) = 1 / 2 ∧ x = 30 :=
by
  sorry

end angle_solution_exists_l162_162618


namespace ellipseC_l162_162686

noncomputable def ellipseEquation (a b : ℝ) (h0 : a > b) (h1 : b > 0) : Prop :=
  ( ∃ e : ℝ, e = (sqrt 3)/2 ∧ a^2 - b^2 = (e * a)^2 ∧ a = 2 * b)

noncomputable def minimumTriangleArea (a b : ℝ) (l l1 l2 : ℝ → ℝ → Prop) (O : ℝ → ℝ → Prop) : Prop :=
  ( ∀ P Q : ℝ × ℝ, P ∈ {P : ℝ × ℝ | l P.1 P.2} ∧ Q ∈ {Q : ℝ × ℝ | l Q.1 Q.2} ∧
    l1 P.1 P.2 ∧ l2 Q.1 Q.2 → 
    (∃ S : ℝ, S = 8 ∧ S = minimum {s : ℝ | ∃ P Q : ℝ × ℝ, 
      O P.1 P.2 ∧ O Q.1 Q.2 ∧ intersect_line_ellipse P Q C}))

theorem ellipseC (a b : ℝ) (h0 : a = 4) (h1 : b = 2) (h2 : a > b) (h3 : b > 0) :
  ellipseEquation a b h2 h3 ∧ minimumTriangleArea a b (λ x y, ∀ k : ℝ, y = k * x + 4) 
      (λ x y, x - 2 * y = 0) (λ x y, x + 2 * y = 0) (λ x y, x = 0 ∧ y = 0) := by
  sorry

end ellipseC_l162_162686


namespace exp_eval_l162_162484

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162484


namespace probability_of_all_heads_or_tails_l162_162229

def num_favorable_outcomes : ℕ := 2

def total_outcomes : ℕ := 2 ^ 5

def probability_all_heads_or_tails : ℚ := num_favorable_outcomes / total_outcomes

theorem probability_of_all_heads_or_tails :
  probability_all_heads_or_tails = 1 / 16 := by
  -- Proof goes here
  sorry

end probability_of_all_heads_or_tails_l162_162229


namespace car_average_speed_l162_162568

-- Define the conditions
def time_taken : ℝ := 4.5
def distance_covered : ℝ := 360

-- Define the average speed computation
def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

-- State the theorem
theorem car_average_speed : average_speed distance_covered time_taken = 80 := 
sorry

end car_average_speed_l162_162568


namespace cos_sum_zero_l162_162283

theorem cos_sum_zero (n : ℤ) (h : n % 7 = 1 ∨ n % 7 = 3 ∨ n % 7 = 4) :
  cos (↑n / 7 * Real.pi - 13 / 14 * Real.pi) +
  cos (3 * ↑n / 7 * Real.pi - 3 / 14 * Real.pi) +
  cos (5 * ↑n / 7 * Real.pi - 3 / 14 * Real.pi) = 0 :=
sorry

end cos_sum_zero_l162_162283


namespace rectangle_area_invariant_l162_162844

theorem rectangle_area_invariant (l w : ℝ) (A : ℝ) 
  (h0 : A = l * w)
  (h1 : A = (l + 3) * (w - 1))
  (h2 : A = (l - 1.5) * (w + 2)) :
  A = 13.5 :=
by
  sorry

end rectangle_area_invariant_l162_162844


namespace mushrooms_collected_l162_162237

theorem mushrooms_collected (x1 x2 x3 x4 : ℕ) 
  (h1 : x1 + x2 = 7) 
  (h2 : x1 + x3 = 9)
  (h3 : x2 + x3 = 10) : x1 = 3 ∧ x2 = 4 ∧ x3 = 6 ∧ x4 = 7 :=
by
  sorry

end mushrooms_collected_l162_162237


namespace exponentiation_example_l162_162478

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162478


namespace cross_section_area_l162_162862

noncomputable def area_cross_section (H α β : ℝ) : ℝ :=
  (H ^ 2 * sin (α + β) * sin (α - β)) / (sin α ^ 2 * sin β * sin (2 * β))

theorem cross_section_area (H α β : ℝ) :
  area_cross_section H α β = (H ^ 2 * sin (α + β) * sin (α - β)) / (sin α ^ 2 * sin β * sin (2 * β)) :=
by
  sorry

end cross_section_area_l162_162862


namespace power_mod_congruence_smallest_exponent_l162_162795

theorem power_mod_congruence (m k : ℕ) (h_prime : Nat.prime (2^(2^m) + 1)) :
  let p := 2^(2^m) + 1
  (2^(2^(m+1) * p^k)) % (p^(k+1)) = 1 := by
  sorry

theorem smallest_exponent (m k : ℕ) (h_prime : Nat.prime (2^(2^m) + 1)) :
  let p := 2^(2^m) + 1
  ∀ n : ℕ, (2^n % (p^(k+1)) = 1) → n ≥ 2^(m+1) * p^k → n = 2^(m+1) * p^k :=
  by sorry

end power_mod_congruence_smallest_exponent_l162_162795


namespace problem_condition_slope_at_A_Sn_value_l162_162655

noncomputable def f (x : ℝ) := x^2 + x

def S (n : ℕ) := ∑ i in Finset.range n, (1 / f (i + 1))

theorem problem_condition (n : ℕ):
  f'(x) = 2x + 1 :=
sorry

theorem slope_at_A :
  deriv f 1 = 3 :=
sorry

theorem Sn_value :
  S 2017 = 2017 / 2018 :=
sorry

end problem_condition_slope_at_A_Sn_value_l162_162655


namespace largest_quantity_l162_162214

noncomputable def D := (2007 / 2006) + (2007 / 2008)
noncomputable def E := (2007 / 2008) + (2009 / 2008)
noncomputable def F := (2008 / 2007) + (2008 / 2009)

theorem largest_quantity : D > E ∧ D > F :=
by { sorry }

end largest_quantity_l162_162214


namespace base_of_second_exponent_l162_162322

theorem base_of_second_exponent (a b : ℕ) (x : ℕ) 
  (h1 : (18^a) * (x^(3 * a - 1)) = (2^6) * (3^b)) 
  (h2 : a = 6) 
  (h3 : 0 < a)
  (h4 : 0 < b) : x = 3 := 
by
  sorry

end base_of_second_exponent_l162_162322


namespace ratio_of_radii_l162_162552

variable {M N A B O : Point}
variable {OM ON : ℝ}
variable (h1 : OM ≠ 0)
variable (h2 : ON / OM = 12 / 13)

theorem ratio_of_radii 
  (hM_touch : touches_circle M A O)
  (hM_touch' : touches_circle M B O)
  (hN_touch1 : touches_circle N A O)
  (hN_touch2 : touches_ray N B A)
  (hN_touch3 : touches_extended_circle N O B):
  let r := radius M A
  let R := radius N A
  r / R = 65 / 144 :=
sorry

end ratio_of_radii_l162_162552


namespace volume_of_cube_l162_162041

theorem volume_of_cube (a : ℕ) (h : (a^3 - a = a^3 - 5)) : a^3 = 125 :=
by {
  -- The necessary algebraic manipulation follows
  sorry
}

end volume_of_cube_l162_162041


namespace geometric_sequence_a6_l162_162967

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_a6 
  (a_1 q : ℝ) 
  (a2_eq : a_1 + a_1 * q = -1)
  (a3_eq : a_1 - a_1 * q ^ 2 = -3) : 
  a_n a_1 q 6 = -32 :=
sorry

end geometric_sequence_a6_l162_162967


namespace sum_of_digits_y_coordinate_of_A_l162_162782

theorem sum_of_digits_y_coordinate_of_A :
  ∀ (A B C : ℝ × ℝ), 
  (∃ (m n : ℝ), (B = (m, m^2)) ∧ (C = (n, n^2)) ∧ (B.snd = C.snd)) ∧
  ((B.fst = (A.fst + C.fst) / 2) ∧
  (∃ (area : ℝ), (A.snd - B.snd) * abs (C.fst - A.fst) / 2 = area) ∧ area = 2000) →
  let k := A.snd in (k.to_int.digits.sum = 18) :=
sorry

end sum_of_digits_y_coordinate_of_A_l162_162782


namespace locus_of_centers_l162_162083

theorem locus_of_centers (a : ℝ) (A B : ℝ × ℝ) 
  (h_dist : dist A B = 2 * a) :
  ∃ (C : ℝ × ℝ), ∀ (O : ℝ × ℝ), dist O A = a ∧ dist O B = a 
    → O ∈ line_perpendicular_bisector A B :=
by
  sorry

end locus_of_centers_l162_162083


namespace sufficient_but_not_necessary_l162_162923

theorem sufficient_but_not_necessary (x : ℝ) : (|x - 1| < 2) → (x < 3) :=
by {
  intros h,
  sorry -- The proof will show that |x - 1| < 2 implies x < 3.
}

end sufficient_but_not_necessary_l162_162923


namespace revenue_difference_l162_162741

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end revenue_difference_l162_162741


namespace problem_statement_l162_162796

open Real

noncomputable def is_quadratic_residue_mod (a p : ℕ) : Prop :=
  ∃ x : ℕ, x^2 ≡ a [MOD p]

noncomputable def A (p : ℕ) : Finset ℕ :=
  (Finset.Ico 1 p).filter (λ a, is_quadratic_residue_mod a p)

noncomputable def B (p : ℕ) : Finset ℕ :=
  (Finset.Ico 1 p).filter (λ a, ¬is_quadratic_residue_mod a p)

noncomputable def sum_cos (s : Finset ℕ) (p : ℕ) : ℝ :=
  s.sum (λ a, cos (a * π / p))

theorem problem_statement (n : ℕ) (hn : 0 < n) (p : ℕ) (hp : p = 4 * n + 1) (prime_p : Nat.Prime p):
  let A' := A p
  let B' := B p
  (sum_cos A' p)^2 + (sum_cos B' p)^2 = (p + 1) / 8 :=
  sorry

end problem_statement_l162_162796


namespace find_a5_plus_a7_l162_162743

variable {a : ℕ → ℝ}

theorem find_a5_plus_a7 (h : a 3 + a 9 = 16) : a 5 + a 7 = 16 := 
sorry

end find_a5_plus_a7_l162_162743


namespace sum_squares_chords_sum_squares_segments_l162_162009

noncomputable section

section GeometricChords

variables (R a L1 L2 L3 S1 S2 S3 : ℝ)
variables (a1 a2 a3 : ℝ) (h1 : a = real.sqrt (a1^2 + a2^2 + a3^2))

-- Part (a): Prove sum of squares of the lengths of the chords
theorem sum_squares_chords (h_R_pos : R > 0) (h_a_pos : a > 0) :
  L1^2 + L2^2 + L3^2 = 12 * R^2 - 4 * a^2 :=
sorry

-- Part (b): Prove sum of squares of the lengths of the segments of the chords
theorem sum_squares_segments (h_segments : 6 * R^2 - 2 * a^2) :
  S1^2 + S2^2 + S3^2 = 6 * R^2 - 2 * a^2 :=
sorry

end GeometricChords

end sum_squares_chords_sum_squares_segments_l162_162009


namespace decreasing_function_range_l162_162320

def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 2*(a-1)*x + 2

theorem decreasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → y ≤ 5 → quadratic_function(a, x) ≥ quadratic_function(a, y)) ↔ (a ≥ 6) := 
sorry

end decreasing_function_range_l162_162320


namespace expected_participants_2013_l162_162749

-- Conditions
def initial_participants : Nat := 500
def growth_rate : ℝ := 1.6
def max_capacity : Nat := 2000

-- Function to calculate participants each year
noncomputable def participants (year : Nat) : ℕ :=
  let calc (n : ℕ) := (initial_participants : ℝ) * growth_rate^n
  if calc year > max_capacity then max_capacity else Nat.floor (calc year)

-- Statement to prove
theorem expected_participants_2013 : participants 3 = 2000 := by
  sorry

end expected_participants_2013_l162_162749


namespace correct_statements_l162_162955

-- Conditions translated to definitions
def statement_1 : Prop := ∃ steps : Set String, steps = "unclear" ∨ steps = "ambiguous"
def statement_2 : Prop := ∀ (algo : Set String), (algo.contains "correct") → algo.contains "definite_result"
def statement_3 : Prop := ∃ (algo1 algo2 : Set String), algo1 = algo2 → False
def statement_4 : Prop := ∀ (algo : Set String), (algo.contains "correct") → algo.contains "finite_steps"

-- Proof that among the four statements, (2), (3), and (4) are correct while (1) is incorrect.
theorem correct_statements : ¬statement_1 ∧ statement_2 ∧ statement_3 ∧ statement_4 := by
  sorry

end correct_statements_l162_162955


namespace tank_capacity_percentage_l162_162916

noncomputable def radius (C : ℝ) := C / (2 * Real.pi)
noncomputable def volume (r h : ℝ) := Real.pi * r^2 * h

theorem tank_capacity_percentage :
  let r_M := radius 8
  let r_B := radius 10
  let V_M := volume r_M 10
  let V_B := volume r_B 8
  (V_M / V_B * 100) = 80 :=
by
  sorry

end tank_capacity_percentage_l162_162916


namespace log_sum_l162_162189

theorem log_sum : log 2 1 + log 2 4 = 2 :=
by
  sorry

end log_sum_l162_162189


namespace sum_of_x_satisfying_eq_l162_162389

noncomputable def g (x : ℝ) := 10 * x + 5

theorem sum_of_x_satisfying_eq : 
  let inv_g := fun y => (y - 5) / 10 in
  let h : ℝ → ℝ := fun x => g ((5 * x)⁻¹) in
  ∑ x in { x | inv_g x = g (h x) }.to_finset, x = 55 :=
by 
  sorry -- Proof to be provided

end sum_of_x_satisfying_eq_l162_162389


namespace range_of_m_l162_162223

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m - 1) * x + 1 = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l162_162223


namespace min_a_3_l162_162437

def sequence (a : ℕ → ℝ) : Prop :=
a 10 = 1 ∧ (∀ n, n ≥ 3 → a n = 
  let s := { |a i - a j| | 1 ≤ i, j < n -1 } in Inf s)

theorem min_a_3 (a_3 : ℝ) :
  ∀ a : ℕ → ℝ, sequence a → a 3 = 21 :=
sorry

end min_a_3_l162_162437


namespace total_wheels_in_parking_lot_l162_162341

-- Definitions (conditions)
def cars := 14
def wheels_per_car := 4
def missing_wheels_per_missing_car := 1
def missing_cars := 2

def bikes := 5
def wheels_per_bike := 2

def unicycles := 3
def wheels_per_unicycle := 1

def twelve_wheeler_trucks := 2
def wheels_per_twelve_wheeler_truck := 12
def damaged_wheels_per_twelve_wheeler_truck := 3
def damaged_twelve_wheeler_trucks := 1

def eighteen_wheeler_trucks := 1
def wheels_per_eighteen_wheeler_truck := 18

-- The total wheels calculation proof
theorem total_wheels_in_parking_lot :
  ((cars * wheels_per_car - missing_cars * missing_wheels_per_missing_car) +
   (bikes * wheels_per_bike) +
   (unicycles * wheels_per_unicycle) +
   (twelve_wheeler_trucks * wheels_per_twelve_wheeler_truck - damaged_twelve_wheeler_trucks * damaged_wheels_per_twelve_wheeler_truck) +
   (eighteen_wheeler_trucks * wheels_per_eighteen_wheeler_truck)) = 106 := by
  sorry

end total_wheels_in_parking_lot_l162_162341


namespace square_side_length_equals_4_l162_162088

theorem square_side_length_equals_4 (s : ℝ) (h : s^2 = 4 * s) : s = 4 :=
sorry

end square_side_length_equals_4_l162_162088


namespace least_prime_factor_of_9_pow_4_minus_9_pow_3_l162_162901

theorem least_prime_factor_of_9_pow_4_minus_9_pow_3 : 
  ∃ p, p = 2 ∧ prime p ∧ p ∣ (9^4 - 9^3) :=
by
  -- We can formalize each step of the solution but for the statement itself we can
  -- just confirm the existence of such a prime (specifically 2 in this case) that divides the given expression.
  sorry

end least_prime_factor_of_9_pow_4_minus_9_pow_3_l162_162901


namespace fraction_girls_at_dance_l162_162964

theorem fraction_girls_at_dance :
  let total_students_colfax := 300
  let boys_to_girls_ratio_colfax := (3, 2)
  let total_students_winthrop := 200
  let boys_to_girls_ratio_winthrop := (3, 4)
  
  let total_students_dance := total_students_colfax + total_students_winthrop
  let girls_colfax := (boys_to_girls_ratio_colfax.snd * total_students_colfax) /
                       (boys_to_girls_ratio_colfax.fst + boys_to_girls_ratio_colfax.snd)
  let girls_winthrop := (boys_to_girls_ratio_winthrop.snd * total_students_winthrop) /
                         (boys_to_girls_ratio_winthrop.fst + boys_to_girls_ratio_winthrop.snd)
  let total_girls := girls_colfax + girls_winthrop
  (total_girls / total_students_dance) = (328 : ℚ) / 700 := by
  sorry

end fraction_girls_at_dance_l162_162964


namespace probability_even_product_is_correct_l162_162840

noncomputable def probability_even_product : ℚ :=
  let n := 13 in            -- total number of integers from 6 to 18 inclusive
  let total_combinations := Nat.choose n 2 in
  let even_count := 7 in    -- number of even integers in the range
  let odd_count := n - even_count in
  let odd_combinations := Nat.choose odd_count 2 in
  let even_product_combinations := total_combinations - odd_combinations in
  even_product_combinations / total_combinations

theorem probability_even_product_is_correct : probability_even_product = 9 / 13 := by
  sorry

end probability_even_product_is_correct_l162_162840


namespace total_volume_l162_162397

noncomputable def volume_pyramid : ℝ :=
  1 / 6

noncomputable def volume_prism : ℝ :=
  real.sqrt 2 - 1

theorem total_volume :
  volume_pyramid * 2 + volume_prism = real.sqrt 2 - 2 / 3 :=
by {
  sorry
}

end total_volume_l162_162397


namespace sum_of_two_special_numbers_l162_162886

noncomputable def ends_with_9_zeros (n : ℕ) : Prop :=
  exists k : ℕ, n = 10^9 * k

def has_110_divisors (n : ℕ) : Prop :=
  nat.divisors_count n = 110

theorem sum_of_two_special_numbers :
  ∃ (n1 n2 : ℕ), ends_with_9_zeros n1 ∧ ends_with_9_zeros n2 ∧
    has_110_divisors n1 ∧ has_110_divisors n2 ∧
    n1 ≠ n2 ∧ n1 + n2 = 7000000000 :=  
  sorry

end sum_of_two_special_numbers_l162_162886


namespace find_boys_and_girls_l162_162349

-- defining the conditions according to the problem description
def class := ℕ -- representing number of students

-- A function to interpret the pair (number of classmates, number of female classmates)
def responses (m d : ℕ) (resp : ℕ × ℕ) : Prop :=
  (resp.1 = m - 1 ∧ abs (resp.2 - d) = 2) ∨
  (resp.2 = d ∧ abs (resp.1 - (m - 1)) = 2)

-- list of given answers
def given_answers : List (ℕ × ℕ) := [(12, 18), (15, 15), (11, 15)]

-- the main proof problem
theorem find_boys_and_girls : ∃ m d : ℕ, 
  m = 13 ∧
  d = 16 ∧
  (∀ resp ∈ given_answers, responses m d resp) :=
begin
  sorry
end

end find_boys_and_girls_l162_162349


namespace exponentiation_example_l162_162475

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162475


namespace average_male_grade_l162_162070

theorem average_male_grade (avg_all avg_fem : ℝ) (N_male N_fem : ℕ) 
    (h1 : avg_all = 90) 
    (h2 : avg_fem = 92) 
    (h3 : N_male = 8) 
    (h4 : N_fem = 12) :
    let total_students := N_male + N_fem
    let total_sum_all := avg_all * total_students
    let total_sum_fem := avg_fem * N_fem
    let total_sum_male := total_sum_all - total_sum_fem
    let avg_male := total_sum_male / N_male
    avg_male = 87 :=
by 
  let total_students := N_male + N_fem
  let total_sum_all := avg_all * total_students
  let total_sum_fem := avg_fem * N_fem
  let total_sum_male := total_sum_all - total_sum_fem
  let avg_male := total_sum_male / N_male
  sorry

end average_male_grade_l162_162070


namespace median_unchanged_l162_162152

-- Given conditions as definitions in Lean
def scores := [a, b, c, d, e]  -- List of five different scores
variable (a b c d e : ℝ)
-- Assume five scores are different and ordered in ascending
axiom distinct_scores : ∀ i j, i ≠ j → scores.nth i ≠ scores.nth j
axiom ordered_scores : a < b ∧ b < c ∧ c < d ∧ d < e

-- Error in recording the lowest score
def recorded_scores := [a', b, c, d, e]
variable (a' : ℝ)
axiom error_in_lowest : a' < a

-- Statement proving that the median is unaffected by the lower recording of the lowest score.
theorem median_unchanged (h : a' < a) : List.median recorded_scores = c := sorry

end median_unchanged_l162_162152


namespace no_unique_x_coord_l162_162814

-- Definition of some distances as per conditions given:
def distance_x_axis (x y : ℝ) : ℝ := abs y
def distance_y_axis (x y : ℝ) : ℝ := abs x
def distance_line (x y : ℝ) : ℝ := abs (x + y - 2) / Real.sqrt 2

-- Mathematical problem stated as:
theorem no_unique_x_coord (x y : ℝ) :
  distance_x_axis x y = distance_y_axis x y →
  distance_x_axis x y = distance_line x y →
  (∃ unique x : ℝ, ∀ y : ℝ, distance_x_axis x y = distance_y_axis x y ∧ distance_x_axis x y = distance_line x y) → False :=
by
  sorry

end no_unique_x_coord_l162_162814


namespace exp_eval_l162_162488

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162488


namespace average_daily_timing_error_l162_162937

theorem average_daily_timing_error (daily_errors : List ℕ) (frequencies : List ℕ) (total_watches : ℕ):
  daily_errors = [0, 1, 2, 3] →
  frequencies = [3, 4, 2, 1] →
  total_watches = 10 →
  let numerator := (0 * 3 + 1 * 4 + 2 * 2 + 3 * 1) in
  let correction := 1.1 in 
  (numerator / total_watches : ℝ) = correction := by
  intros
  simp [numerator, total_watches, correction]
  sorry

end average_daily_timing_error_l162_162937


namespace factor_expression_l162_162633

theorem factor_expression (x y z : ℝ) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3 ) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3 ) = 
  (x + y) * (y + z) * (z + x) := 
by
  sorry

end factor_expression_l162_162633


namespace complex_solving_l162_162227

theorem complex_solving :
  ∃ (x y : ℝ), 
    let a := x^2 in
    let b := y^2 in
    (a^3 - 15 * a^2 * b + 15 * a * b^2 - b^3 = -8) ∧
    (a^2 - 10 * a * b + b^2 = -4 / 3) ∧
    ((x + y * complex.I)^6 = -8 - 8 * complex.I) :=
sorry

end complex_solving_l162_162227


namespace sports_day_popularity_order_l162_162598

theorem sports_day_popularity_order:
  let dodgeball := (3:ℚ) / 8
  let chess_tournament := (9:ℚ) / 24
  let track := (5:ℚ) / 16
  let swimming := (1:ℚ) / 3
  dodgeball = (18:ℚ) / 48 ∧
  chess_tournament = (18:ℚ) / 48 ∧
  track = (15:ℚ) / 48 ∧
  swimming = (16:ℚ) / 48 ∧
  list.sort (>=) [dodgeball, chess_tournament, track, swimming] = [swimming, dodgeball, chess_tournament, track] :=
by {
  sorry
}

end sports_day_popularity_order_l162_162598


namespace average_age_choir_l162_162845

theorem average_age_choir : 
  let avg_age_females := 28
    let num_females := 12
    let avg_age_males := 36
    let num_males := 18
    let avg_age_children := 10
    let num_children := 10
    let total_age := (avg_age_females * num_females) + (avg_age_males * num_males) + (avg_age_children * num_children)
    let total_people := num_females + num_males + num_children
    let average_age := total_age / total_people
in average_age = 27.1 := by 
    sorry

end average_age_choir_l162_162845


namespace shape_division_l162_162976

-- Define the conditions
def is_L_shaped (s : Set (ℕ × ℕ)) : Prop := 
  ∃ a b, s = {(a, b), (a, b+1), (a+1, b)}

def is_cross_shaped (s : Set (ℕ × ℕ)) : Prop := 
  ∃ a b, s = {(a, b), (a, b+1), (a, b-1), (a+1, b), (a-1, b)}

-- Define the grid shape
def is_grid_shape (shape : Set (ℕ × ℕ)) : Prop :=
  shape = { (x, y) | x < 6 ∧ y < 5 }

-- The Lean statement to prove
theorem shape_division (shape : Set (ℕ × ℕ)) :
  is_grid_shape shape →
  (∃ shapes : list (Set (ℕ × ℕ)), shapes.length = 4 ∧ ∀ s ∈ shapes, is_L_shaped s ∧ ∀ s1 s2 ∈ shapes, s1 ≠ s2 → s1 ∩ s2 = ∅) →
  (∃ shapes : list (Set (ℕ × ℕ)), shapes.length = 5 ∧ ∀ s ∈ shapes, is_cross_shaped s ∧ ∀ s1 s2 ∈ shapes, s1 ≠ s2 → s1 ∩ s2 = ∅) →
  shape = { (x, y) | x < 6 ∧ y < 5 } :=
by
  sorry

end shape_division_l162_162976


namespace sum_of_roots_l162_162854

-- Initial condition of the problem as a definition
def equation (x : ℝ) : Prop :=
  1 / x + 1 / (x + 3) - 1 / (x + 6) - 1 / (x + 9) - 1 / (x + 12) - 1 / (x + 15) +
  1 / (x + 18) + 1 / (x + 21) = 0

-- The theorem statement translating the problem
theorem sum_of_roots (x a b c d : ℝ) (h : equation x) (root_form : x = -a + sqrt (b + c * sqrt d) ∨ 
                                                          x = -a - sqrt (b + c * sqrt d) ∨
                                                          x = -a + sqrt (b - c * sqrt d) ∨ 
                                                          x = -a - sqrt (b - c * sqrt d)) :
  a + b + c + d = 57.5 :=
sorry

end sum_of_roots_l162_162854


namespace prob_neither_l162_162872

variable P : Set ℝ → ℝ
variable A B : Set ℝ

axiom probA : P A = 0.15
axiom probB : P B = 0.40
axiom probAandB : P (A ∩ B) = 0.15

theorem prob_neither : P (Aᶜ ∩ Bᶜ) = 0.60 := sorry

end prob_neither_l162_162872


namespace term_in_sequence_l162_162147

   theorem term_in_sequence (n : ℕ) (h1 : 1 ≤ n) (h2 : 6 * n + 1 = 2005) : n = 334 :=
   by
     sorry
   
end term_in_sequence_l162_162147


namespace eccentricity_of_ellipse_range_of_k_l162_162258

noncomputable theory

-- Ellipse E: x²/a² + y²/b² = 1 with foci on the x-axis
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- The condition when B is the top vertex, area(ΔABD) = 2ab
def part1_conditions (a b : ℝ) : Prop :=
  let ε := ((a^2 + b^2) = 4 * b^2) ∧ (a^2 = 3 * b^2) in
  ∃ ε, ε

-- Proof statement for part 1
theorem eccentricity_of_ellipse (a b : ℝ) (h : part1_conditions a b) : a / sqrt (a^2 + b^2) = sqrt (6) / 3 :=
by sorry

-- The condition when b = √3 and 2|AB| = |AC|
def part2_conditions (a : ℝ) (k : ℝ) : Prop :=
  let b := sqrt 3
  let left_vertex := (-a, 0)
  let k_cond := k > 0 in
  let eq_rel := (a^2 = (6 * k^2 - 3 * k) / (k^3 - 2)) ∧ (a^2 > 3) ∧ (2 * madness|AB| = |AC|) in
  ∃ eq_rel, eq_rel

-- Proof statement for part 2
theorem range_of_k (a k : ℝ) (h : part2_conditions a k) : (real.cbrt 2 < k) ∧ (k < 2) :=
by sorry

end eccentricity_of_ellipse_range_of_k_l162_162258


namespace trajectory_equation_l162_162579

-- Definition of the distance functions based on the problem conditions.
def distance_to_fixed_point (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 1)^2 + y^2)

def distance_to_y_axis (x : ℝ) : ℝ :=
  Real.abs x

-- The condition that describes the problem
def satisfies_condition (x y : ℝ) : Prop :=
  distance_to_fixed_point x y = distance_to_y_axis x + 1

-- The Lean statement that needs to be proved
theorem trajectory_equation (x y : ℝ) : satisfies_condition x y ↔ 
  (y = 0 ∧ x ≤ 0) ∨ (y^2 = 4 * x ∧ x ≥ 0) :=
by
  sorry

end trajectory_equation_l162_162579


namespace ball_probability_l162_162543

theorem ball_probability (total_balls white green yellow red purple : ℕ) 
  (h_total : total_balls = 60)
  (h_white : white = 22)
  (h_green : green = 18)
  (h_yellow : yellow = 5)
  (h_red : red = 6)
  (h_purple : purple = 9) :
  (total_balls - red - purple) / total_balls = 3 / 4 :=
by
  sorry

end ball_probability_l162_162543


namespace line_distance_equality_l162_162262

theorem line_distance_equality (A B : ℝ × ℝ) (l_1 l_2 l_3 l_4 : ℝ → ℝ → Prop) :
  A = (1, 3) →
  B = (-5, 1) →
  l_1 x y ↔ x - 3 * y - 8 = 0 →
  l_2 x y ↔ 3 * x + y + 4 = 0 →
  l_3 x y ↔ 3 * x - y + 6 = 0 →
  l_4 x y ↔ 2 * x + y + 2 = 0 →
  (∀ l, (l = l_1 ∨ l = l_4) ∧ (∀ p, p ≠ A ∧ p ≠ B → dist p l_1 = dist p l_4)) :=
by sorry

end line_distance_equality_l162_162262


namespace third_graders_wore_green_shirts_l162_162068

theorem third_graders_wore_green_shirts :
    let cost_kinder := 101 * 5.80 in
    let cost_first := 113 * 5 in
    let cost_second := 107 * 5.60 in
    let total_spent := 2317 in
    let cost_per_green_shirt := 5.25 in
    let third_graders_cost := total_spent - (cost_kinder + cost_first + cost_second) in
    third_graders_cost / cost_per_green_shirt = 108 :=
by
  sorry


end third_graders_wore_green_shirts_l162_162068


namespace factor_expression_l162_162634

theorem factor_expression (x y z : ℝ) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3 ) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3 ) = 
  (x + y) * (y + z) * (z + x) := 
by
  sorry

end factor_expression_l162_162634


namespace relationship_of_a_b_c_l162_162243

theorem relationship_of_a_b_c :
  let a := 1.27 ^ 0.2
  let b := Real.logb 0.3 (Real.tan (Real.pi * 46 / 180))
  let c := 2 * Real.sin (Real.pi * 29 / 180)
  a > c ∧ c > b := 
by
  sorry

end relationship_of_a_b_c_l162_162243


namespace div_by_19_l162_162407

theorem div_by_19 (n : ℕ) : 19 ∣ (26^n - 7^n) :=
sorry

end div_by_19_l162_162407


namespace A_B_C_D_l162_162907

theorem A : -1 ∉ Nat := by
  sorry

theorem B : {1} ⊆ Int := by
  sorry

theorem C : ¬ (0 ∈ ∅) := by
  sorry

theorem D : ∅ ⊂ {0} := by
  sorry

end A_B_C_D_l162_162907


namespace chess_tournament_rounds_needed_l162_162561

theorem chess_tournament_rounds_needed
  (num_players : ℕ)
  (num_games_per_round : ℕ)
  (H1 : num_players = 20)
  (H2 : num_games_per_round = 10) :
  (num_players * (num_players - 1)) / num_games_per_round = 38 :=
by
  sorry

end chess_tournament_rounds_needed_l162_162561


namespace bat_selling_price_l162_162311

-- Definitions of given conditions
def cards_sold_price := 25
def glove_original_price := 30
def glove_discount := 0.20
def glove_sold_price := glove_original_price * (1 - glove_discount)
def cleats_pair_price := 10
def cleats_total_price := 2 * cleats_pair_price
def total_amount := 79

-- Definition of the selling price of the bat
def bat_sold_price := total_amount - (cards_sold_price + glove_sold_price + cleats_total_price)

-- Lean statement for the proof
theorem bat_selling_price :
  bat_sold_price = 10 :=
by
  sorry

end bat_selling_price_l162_162311


namespace Sammy_has_8_bottle_caps_l162_162412

def Billie_caps : Nat := 2
def Janine_caps (B : Nat) : Nat := 3 * B
def Sammy_caps (J : Nat) : Nat := J + 2

theorem Sammy_has_8_bottle_caps : 
  Sammy_caps (Janine_caps Billie_caps) = 8 := 
by
  sorry

end Sammy_has_8_bottle_caps_l162_162412


namespace range_of_m_l162_162301

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (x * (Real.exp x) - x + 1)
noncomputable def m_range := Set.Icc (Real.exp 2 / (2 * (Real.exp 2) - 1)) 1

theorem range_of_m :
  ∃ m : ℝ, (∀ x : ℝ, m < f x ↔ f x = 0 ∨ f x = 1) ↔ m ∈ m_range := sorry

end range_of_m_l162_162301


namespace queens_in_each_corner_l162_162813

def chessboard := fin 100 → fin 100

noncomputable def is_valid_placement (queens : set (fin 100 × fin 100)) :=
  (∀ q1 q2 ∈ queens, q1 ≠ q2 → ¬ is_attacking q1 q2)

noncomputable def is_attacking (q1 q2 : fin 100 × fin 100) :=
  q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ (q1.1 - q2.1).nat_abs = (q1.2 - q2.2).nat_abs

def corner_square (corner : fin 2 × fin 2) (p : fin 50 × fin 50) :=
  ((corner.1 : ℕ) * 50 + p.1, (corner.2 : ℕ) * 50 + p.2)

theorem queens_in_each_corner (queens : set (fin 100 × fin 100)) (h1 : is_valid_placement queens) (h2 : queens.card = 100) :
  ∀ corner : fin 2 × fin 2, ∃ p : fin 50 × fin 50, corner_square corner p ∈ queens :=
by sorry

end queens_in_each_corner_l162_162813


namespace yellow_lights_count_l162_162099

theorem yellow_lights_count (total_lights : ℕ) (red_lights : ℕ) (blue_lights : ℕ) (yellow_lights : ℕ) :
  total_lights = 95 → red_lights = 26 → blue_lights = 32 → yellow_lights = total_lights - (red_lights + blue_lights) → yellow_lights = 37 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end yellow_lights_count_l162_162099


namespace polynomial_bound_l162_162801

open Real

theorem polynomial_bound (f : Polynomial ℝ) (hf_deg : 1 ≤ f.natDegree) :
  ∀ c > 0, ∃ n0 : ℕ, ∀ (P : Polynomial ℝ), n0 ≤ P.natDegree →
  (P.leadingCoeff = 1) →
  ∃ A : Finset ℤ, A.card = P.natDegree + 1 ∧
  ∃ x ∈ A, |f.eval (P.eval x)| > c :=
by
  sorry

end polynomial_bound_l162_162801


namespace difference_between_max_and_min_l162_162918

def maxNumber (a b c d : Nat) : Nat := List.foldl (λ acc x => acc * 10 + x) 0 (List.reverse (List.sort (λ x y => x > y) [a, b, c, d]))

def minNumber (a b c d : Nat) : Nat := List.foldl (λ acc x => acc * 10 + x) 0 (List.sort (λ x y => x < y) [a, b, c, d])

theorem difference_between_max_and_min : maxNumber 7 3 1 4 - minNumber 7 3 1 4 = 6084 := by
  sorry

end difference_between_max_and_min_l162_162918


namespace max_value_expression_l162_162115

theorem max_value_expression (p : ℝ) (q : ℝ) (h : q = p - 2) :
  ∃ M : ℝ, M = -70 + 96.66666666666667 ∧ (∀ p : ℝ, -3 * p^2 + 24 * p - 50 + 10 * q ≤ M) :=
sorry

end max_value_expression_l162_162115


namespace second_pipe_fill_time_l162_162166

theorem second_pipe_fill_time (x : ℝ) :
  let rate1 := 1 / 8
  let rate2 := 1 / x
  let combined_rate := 1 / 4.8
  rate1 + rate2 = combined_rate → x = 12 :=
by
  intros
  sorry

end second_pipe_fill_time_l162_162166


namespace ellipse_has_correct_equation_l162_162256

noncomputable def ellipse_Equation (a b : ℝ) (eccentricity : ℝ) (triangle_perimeter : ℝ) : Prop :=
  let c := a * eccentricity
  (a > b) ∧ (b > 0) ∧ (eccentricity = (Real.sqrt 3) / 3) ∧ (triangle_perimeter = 4 * (Real.sqrt 3)) ∧
  (a = Real.sqrt 3) ∧ (b^2 = a^2 - c^2) ∧
  (c = 1) ∧
  (b = Real.sqrt 2) ∧
  (∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1) ↔ ((x^2 / 3) + (y^2 / 2) = 1))

theorem ellipse_has_correct_equation : ellipse_Equation (Real.sqrt 3) (Real.sqrt 2) ((Real.sqrt 3) / 3) (4 * (Real.sqrt 3)) := 
sorry

end ellipse_has_correct_equation_l162_162256


namespace touching_squares_same_color_probability_l162_162312

theorem touching_squares_same_color_probability :
  let m := 0
  let n := 1
  100 * m + n = 1 :=
by
  let m := 0
  let n := 1
  sorry -- Proof is omitted as per instructions

end touching_squares_same_color_probability_l162_162312


namespace find_b_l162_162893

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 :=
by
  sorry

end find_b_l162_162893


namespace power_of_powers_l162_162489

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162489


namespace power_of_powers_l162_162494

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162494


namespace range_of_function_l162_162434

theorem range_of_function : 
  ∀ (x : ℝ), (3 ≤ x ∧ x ≤ 4) → (1 ≤ (sqrt (x - 3) + sqrt (12 - 3 * x)) ∧ (sqrt (x - 3) + sqrt (12 - 3 * x)) ≤ 2) :=
by
  sorry

end range_of_function_l162_162434


namespace problem_l162_162395

noncomputable def coefficient (n : ℕ) [Fact (2 ≤ n)] : ℝ :=
  if h : 1 ≤ n then (n * (n - 1)) / 2 else 0

theorem problem : ∑ n in (finset.range 2014).image (λ x, x + 2), (1 : ℝ) / coefficient n = 4028 / 2015 := 
sorry

end problem_l162_162395


namespace rhombus_diagonals_not_equal_l162_162537

-- Define the concept of a rhombus
structure Rhombus (α : Type*) [MetricSpace α] :=
  (vertices : Fin 4 → α)
  (sides_equal : ∀ i j, i < 4 → j < 4 → dist (vertices i) (vertices ((i + 1) % 4)) = dist (vertices j) (vertices ((j + 1) % 4)))
  (diagonals_perpendicular : ⟦0⟧ ≠ ⟦2⟧ → ⟦1⟧ ≠ ⟦3⟧ → angle (vertices 0) (vertices 2) (vertices 1) = π/2)
  (diagonals_bisect : ∀ i j, i < 4 → j < 4 → dist (vertices i) (vertices j) = dist (vertices ((i + 2) % 4)) (vertices ((j + 2) % 4)))

-- Define the assertion that a statement about rhombus is incorrect
def incorrect_statement (α : Type*) [MetricSpace α] (R : Rhombus α) : Prop :=
  ¬(dist (R.vertices 0) (R.vertices 2) = dist (R.vertices 1) (R.vertices 3))

-- Prove the assertion
theorem rhombus_diagonals_not_equal (α : Type*) [MetricSpace α] (R : Rhombus α) :
  incorrect_statement α R := sorry

end rhombus_diagonals_not_equal_l162_162537


namespace period_sine_transformed_l162_162117

theorem period_sine_transformed (x : ℝ) : 
  let y := 3 * Real.sin ((x / 3) + (Real.pi / 4))
  ∃ p : ℝ, (∀ x : ℝ, y = 3 * Real.sin ((x + p) / 3 + (Real.pi / 4)) ↔ y = 3 * Real.sin ((x / 3) + (Real.pi / 4))) ∧ p = 6 * Real.pi :=
sorry

end period_sine_transformed_l162_162117


namespace sum_of_reciprocals_le_30_l162_162031

theorem sum_of_reciprocals_le_30 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (h_distinct : Function.Injective a)
  (h_no_9 : ∀ i : Fin n, ∀ (d ∈ Nat.digits 10 (a i)), d ≠ 9) :
  (Finset.univ : Finset (Fin n)).sum (λ i, (a i)⁻¹: ℚ) ≤ 30 := sorry

end sum_of_reciprocals_le_30_l162_162031


namespace janice_typing_proof_l162_162763

noncomputable def janice_typing : Prop :=
  let initial_speed := 6
  let error_speed := 8
  let corrected_speed := 5
  let typing_duration_initial := 20
  let typing_duration_corrected := 15
  let erased_sentences := 40
  let typing_duration_after_lunch := 18
  let total_sentences_end_of_day := 536

  let sentences_initial_typing := typing_duration_initial * error_speed
  let sentences_post_error_typing := typing_duration_corrected * initial_speed
  let sentences_final_typing := typing_duration_after_lunch * corrected_speed

  let sentences_total_typed := sentences_initial_typing + sentences_post_error_typing - erased_sentences + sentences_final_typing

  let sentences_started_with := total_sentences_end_of_day - sentences_total_typed

  sentences_started_with = 236

theorem janice_typing_proof : janice_typing := by
  sorry

end janice_typing_proof_l162_162763


namespace rhombus_longer_diagonal_l162_162176

theorem rhombus_longer_diagonal 
  (a b : ℝ) 
  (h₁ : a = 61) 
  (h₂ : b = 44) :
  ∃ d₂ : ℝ, d₂ = 2 * Real.sqrt (a * a - (b / 2) * (b / 2)) :=
sorry

end rhombus_longer_diagonal_l162_162176


namespace arithmetic_sequence_general_term_max_sum_l162_162737

noncomputable def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  (a_n 3 + a_n 4 = 15) ∧ (a_n 2 * a_n 5 = 54) ∧ (∀ m n, m < n → a_n m > a_n n)

theorem arithmetic_sequence_general_term_max_sum :
  ∀ (a_n : ℕ → ℝ), arithmetic_sequence a_n → 
    (∀ n, a_n n = 11 - ↑n) ∧ (∑ i in finset.range 11, a_n i = 55) :=
by {
  sorry
}

end arithmetic_sequence_general_term_max_sum_l162_162737


namespace mayoral_election_possible_l162_162628

variable (N : Type) [Fintype N] [Nonempty N]
          (acquainted : N → N → Prop)
          [decidable_rel acquainted]

def at_least_30_percent (n : N) : Prop :=
  Fintype.card { x // acquainted n x } ≥ (Fintype.card N * 3) / 10

theorem mayoral_election_possible :
  (∀ n : N, at_least_30_percent N acquainted n) →
  (∀ n : N, ∃ c1 c2 : N, c1 ≠ c2 ∧
    (∀ x : N, (acquainted x c1 ∨ acquainted x c2) → 
     ∃ y : N, (acquainted y c1 ∨ acquainted y c2) ∧
              (Fintype.card { y // acquainted x y } ≥ Fintype.card N / 2))) :=
sorry

end mayoral_election_possible_l162_162628


namespace sum_of_segments_le_longest_side_l162_162817

open Triangle Geometry

variable {A B C M A1 B1 C1 : Point}

theorem sum_of_segments_le_longest_side (A B C M A1 B1 C1 : Point) 
  (hA1 : A1 ∈ Segment B C) (hB1 : B1 ∈ Segment A C) (hC1 : C1 ∈ Segment A B)
  (hAA1 : A1 ∈ Segment A M) (hBB1 : B1 ∈ Segment B M) (hCC1 : C1 ∈ Segment C M)
  (hIntersect : M ∈ Segment A1 ∩ Segment B1 ∩ Segment C1)
  (hLongest : ∀ (s ∈ { Segment A B, Segment B C, Segment A C }), length s ≤ length (longest_side of_triangle A B C)) :
  length (Segment M A1) + length (Segment M B1) + length (Segment M C1) ≤ length (longest_side of_triangle A B C) :=
by
  sorry

end sum_of_segments_le_longest_side_l162_162817


namespace sum_of_coefficients_l162_162249

theorem sum_of_coefficients :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℤ),
  (x : ℤ) →
  (x^2 - 3 * x + 1)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^{10} →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10}) = -2 :=
by
  sorry

end sum_of_coefficients_l162_162249


namespace correct_set_relationship_l162_162305

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) (C : Set ℕ)

-- Universal set and given subsets
def U := {1, 2, 3, 4, 5, 6, 7, 8}
def A := {3, 4, 5}
def B := {1, 3, 6}
def C := {2, 7, 8}

theorem correct_set_relationship :
  (C ∪ A) ∩ (C ∪ B) = C := by
  sorry

end correct_set_relationship_l162_162305


namespace exists_number_with_eight_divisors_d_definition_l162_162100

theorem exists_number_with_eight_divisors :
  ∃ n : ℕ, (d(n) = 8) where d(n) := (∏ e in (prime_factors n), (nat.factor_powers n e) + 1) by
-- Definitions and other necessary constructs 
def prime_factors (n : ℕ) : finset ℕ :=
  {p ∈ finset.filter nat.prime (finset.range (n + 1)) | p ∣ n}

def nat.factor_powers (n : ℕ) (p : ℕ) : ℕ :=
  if hp : p ∈ prime_factors n then
    nat.find (λ k, p^k ∣ n ∧ ¬ (p^(k + 1)) ∣ n)
  else
    0

theorem d_definition (n : ℕ) : 
  d(n) = (∏ e in (prime_factors n), (nat.factor_powers n e) + 1) :=
  sorry

-- Here we should prove the actual theorem, using the definitions
sorry

end exists_number_with_eight_divisors_d_definition_l162_162100


namespace central_symmetry_ray_opposite_l162_162056

-- Definitions for points, lines, and central symmetry
structure Point := (x : ℝ) (y : ℝ)
structure Line := (p1 p2 : Point)

def central_symmetry (O : Point) (P : Point) : Point :=
  ⟨2 * O.x - P.x, 2 * O.y - P.y⟩

def are_parallel (l1 l2 : Line) : Prop :=
  let Line.mk ⟨x1, y1⟩ ⟨x2, y2⟩ := l1 in
  let Line.mk ⟨x3, y3⟩ ⟨x4, y4⟩ := l2 in
  (y2 - y1) * (x4 - x3) = (x2 - x1) * (y4 - y3)

noncomputable def ray_opposite (O A X : Point) : Prop :=
  let A1 := central_symmetry O A in
  let X1 := central_symmetry O X in
  are_parallel ⟨A, X⟩ ⟨A1, X1⟩

theorem central_symmetry_ray_opposite (A X O : Point) :
  ray_opposite O A X :=
sorry

end central_symmetry_ray_opposite_l162_162056


namespace exponentiation_identity_l162_162511

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162511


namespace volume_ratio_of_sphere_surface_area_l162_162303

theorem volume_ratio_of_sphere_surface_area 
  {V1 V2 V3 : ℝ} 
  (h : V1/V3 = 1/27 ∧ V2/V3 = 8/27) 
  : V1 + V2 = (1/3) * V3 := 
sorry

end volume_ratio_of_sphere_surface_area_l162_162303


namespace polynomial_remainder_x5_plus_3_l162_162642

theorem polynomial_remainder_x5_plus_3 {R : Type*} [CommRing R] :
  ∃ q : R[X], (X^5 + 3 : R[X]) = (X + 1)^2 * q + 2 := 
by
  sorry

end polynomial_remainder_x5_plus_3_l162_162642


namespace circle_passes_through_fixed_point_l162_162278

theorem circle_passes_through_fixed_point :
  ∀ (C : ℝ × ℝ), (C.2 ^ 2 = 4 * C.1) ∧ (C.1 = -1 + (C.1 + 1)) → ∃ P : ℝ × ℝ, P = (1, 0) ∧
    (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = (C.1 + 1) ^ 2 + (0 - C.2) ^ 2 :=
by
  sorry

end circle_passes_through_fixed_point_l162_162278


namespace bruna_can_determine_numbers_l162_162144

theorem bruna_can_determine_numbers (a b c d e : ℕ) 
    (h1 : a + b ≤ a + c)
    (h2 : List.pairwise (<=) [a, b, c, d, e])
    (h3 : Multiset.map (λ (x : ℕ × ℕ), x.1 + x.2) ((Multiset.powerset (Multiset.ofList [a, b, c, d, e])).filter (λ s, s.card = 2)) = {24, 28, 30, 30, 32, 34, 36, 36, 40, 42} : Multiset ℕ) :
    (set.univ.sum [a, b, c, d, e]) / 4 = 83 :=
by sorry

end bruna_can_determine_numbers_l162_162144


namespace scheduling_arrangements_336_l162_162338

-- Define the types and conditions
variable {T : Type} [Fintype T] (teachers : Finset T) (sessions : Finset (Fin 6))
variable (math_teacher english_teacher : T)
variable (monday tuesday wednesday thursday friday sunday : Fin 6)
variable (schedule : Fin 6 → T)

-- State the conditions
axiom math_teacher_ne_monday_wednesday : schedule monday ≠ math_teacher ∧ schedule wednesday ≠ math_teacher
axiom english_teacher_ne_tuesday_thursday : schedule tuesday ≠ english_teacher ∧ schedule thursday ≠ english_teacher

-- Verify there are 336 different possible schedules
theorem scheduling_arrangements_336 
  (h1 : ∀ t ∈ teachers, ∃ d ∈ sessions, schedule d = t)
  (h2 : ∀ d ∈ sessions, ∃ t ∈ teachers, schedule d = t)
  (h3 : ∀ d1 d2 ∈ sessions, d1 ≠ d2 → schedule d1 ≠ schedule d2) :
  card { σ : Fin 6 → T | (∀ t ∈ teachers, ∃ d ∈ sessions, σ d = t) ∧
         (∀ d ∈ sessions, ∃ t ∈ teachers, σ d = t) ∧
         (∀ d1 d2 ∈ sessions, d1 ≠ d2 → σ d1 ≠ σ d2) ∧
         σ monday ≠ math_teacher ∧ σ wednesday ≠ math_teacher ∧
         σ tuesday ≠ english_teacher ∧ σ thursday ≠ english_teacher } = 336 := 
sorry

end scheduling_arrangements_336_l162_162338


namespace square_number_form_l162_162636

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def repeat_digit (d : ℕ) (len : ℕ) : ℕ :=
  d * (10 ^ len - 1) / 9

theorem square_number_form (n a b : ℕ) (h_pos_n : n > 0)
  (h_a : a ≥ 0 ∧ a < 10) (h_b : b ≥ 0 ∧ b < 10) :
  let num := repeat_digit a n * 10^n + repeat_digit b n in
  is_square num → num = 16 ∨ num = 25 ∨ num = 36 ∨ num = 49 ∨ num = 64 ∨ num = 81 ∨ num = 7744 :=
begin
  sorry
end

end square_number_form_l162_162636


namespace estimate_flight_time_around_earth_l162_162837

theorem estimate_flight_time_around_earth 
  (radius : ℝ) 
  (speed : ℝ)
  (h_radius : radius = 6000) 
  (h_speed : speed = 600) 
  : abs (20 * Real.pi - 63) < 1 :=
by
  sorry

end estimate_flight_time_around_earth_l162_162837


namespace total_games_played_l162_162445

def number_of_games (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem total_games_played :
  number_of_games 9 2 = 36 :=
by
  -- Proof to be filled in later
  sorry

end total_games_played_l162_162445


namespace exponentiation_example_l162_162469

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162469


namespace sin_half_angle_l162_162711

theorem sin_half_angle
  (theta : ℝ)
  (h1 : Real.sin theta = 3 / 5)
  (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  Real.sin (theta / 2) = - (3 * Real.sqrt 10 / 10) :=
by
  sorry

end sin_half_angle_l162_162711


namespace abs_expression_eq_l162_162965

theorem abs_expression_eq : ∀ (π : ℝ), π < 10 → |π - |π - 10|| = 10 - 2 * π := by
  intros π h
  sorry

end abs_expression_eq_l162_162965


namespace clever_value_points_l162_162662

def clever_value_point (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
∃ x₀ : ℝ, f x₀ = f' x₀

def f1 (x : ℝ) : ℝ := x^2
def f1' (x : ℝ) : ℝ := 2 * x

def f2 (x : ℝ) : ℝ := Real.exp (-x)
def f2' (x : ℝ) : ℝ := -Real.exp (-x)

def f3 (x : ℝ) : ℝ := Real.log x
noncomputable def f3' (x : ℝ) : ℝ := 1 / x

def f4 (x : ℝ) : ℝ := Real.tan x
noncomputable def f4' (x : ℝ) : ℝ := 1 / (Real.cos x)^2

def f5 (x : ℝ) : ℝ := x + 1 / x
noncomputable def f5' (x : ℝ) : ℝ := 1 - 1 / x^2

theorem clever_value_points :
  (clever_value_point f1 f1') ∧
  ¬ (clever_value_point f2 f2') ∧
  (clever_value_point f3 f3') ∧
  ¬ (clever_value_point f4 f4') ∧
  (clever_value_point f5 f5') :=
by sorry

end clever_value_points_l162_162662


namespace exp_eval_l162_162480

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162480


namespace max_management_fee_l162_162160

-- Conditions
def price_first_year : ℕ := 70
def sales_volume_first_year : ℕ := 11800   -- in pieces
def management_fee_rate (x : ℝ) := x / 100
def price_increment (x : ℝ) := (70 * x / 100) / (1 - x / 100)
def sales_volume_second_year (x : ℝ) := 11800 - 1000 * x
def price_second_year (x : ℝ) := price_first_year + price_increment x
def total_management_fee (x : ℝ) := (price_second_year x * sales_volume_second_year x) * management_fee_rate x

-- Proof problem
theorem max_management_fee (h : total_management_fee x ≥ 140000) : x ≤ 10 := by
  sorry

end max_management_fee_l162_162160


namespace rhombus_area_is_128sqrt3_l162_162583

noncomputable def rhombus_area (radius : ℝ) : ℝ :=
let d1 := radius in
let d2 := 2 * (radius / 2 * real.sqrt 3) in
1/2 * d1 * d2

theorem rhombus_area_is_128sqrt3 :
  rhombus_area 16 = 128 * real.sqrt 3 :=
by
  sorry

end rhombus_area_is_128sqrt3_l162_162583


namespace sum_f_div_2500_eq_567_l162_162025

def f (n : ℕ) : ℕ := 
  let m := (Real.sqrt (Real.sqrt n)).toNat
  if Real.sqrt (Real.sqrt n).toReal - m < 1 / 2 then m else m + 1

def sum_f_div (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), 1 / (f k).toReal

theorem sum_f_div_2500_eq_567 :
  sum_f_div 2500 = 567 := 
by
  sorry

end sum_f_div_2500_eq_567_l162_162025


namespace white_balls_count_l162_162933

theorem white_balls_count (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) (W : ℕ)
    (h_total : total_balls = 100)
    (h_green : green_balls = 30)
    (h_yellow : yellow_balls = 10)
    (h_red : red_balls = 37)
    (h_purple : purple_balls = 3)
    (h_prob : prob_neither_red_nor_purple = 0.6)
    (h_computation : W = total_balls * prob_neither_red_nor_purple - (green_balls + yellow_balls)) :
    W = 20 := 
sorry

end white_balls_count_l162_162933


namespace part1_decreasing_part1_range_part2_minimum_value_l162_162296

theorem part1_decreasing (x : ℝ) : 
    is_decreasing_on (f : ℝ → ℝ) (set.Ici (1 : ℝ)) := 
sorry

theorem part1_range (x : ℝ) : 
    range_of (f : ℝ → ℝ) (set.Ici (1 : ℝ)) = set.Ici (1 : ℝ) :=
sorry

theorem part2_minimum_value (a : ℝ) (h : a ≥ -1) :
    minimum_value (f : ℝ → ℝ) (set.Icc (-1 : ℝ) (1 : ℝ)) = 
    if (-1 : ℝ ≤ a ∧ a < 1) then -a^2 + 2 
    else 3 - 2 * a :=
sorry

end part1_decreasing_part1_range_part2_minimum_value_l162_162296


namespace geometric_sequence_a3_eq_2_l162_162750

theorem geometric_sequence_a3_eq_2 
  (a_1 a_3 a_5 : ℝ) 
  (h1 : a_1 * a_3 * a_5 = 8) 
  (h2 : a_3^2 = a_1 * a_5) : 
  a_3 = 2 :=
by 
  sorry

end geometric_sequence_a3_eq_2_l162_162750


namespace irie_decomposition_l162_162458

/-- Definition of an irie number -/
def irie (k : ℕ) : ℝ := 1 + 1 / k

/-- The main theorem: Every integer n ≥ 2 can be written as the product of r distinct irie numbers if r ≥ n - 1. -/
theorem irie_decomposition (n r : ℕ) (hn : n ≥ 2) (hr : r ≥ n - 1) : 
  ∃ (irie_numbers : list ℝ), (∀ k, k ∈ irie_numbers → ∃ m : ℕ, irie m = k) ∧ (irie_numbers.length = r) ∧ (irie_numbers.prod = n) :=
sorry

end irie_decomposition_l162_162458


namespace sum_negative_integers_abs_less_than_4_l162_162439

theorem sum_negative_integers_abs_less_than_4 : 
  ∑ i in {x : ℤ | -4 < x ∧ x < 0}, i = -6 :=
by 
  sorry

end sum_negative_integers_abs_less_than_4_l162_162439


namespace triangle_inequality_ineq_l162_162553

theorem triangle_inequality_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hac : a + c > b) (hbc : b + c > a) :
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c)) ≥ 3 :=
begin
  sorry,
end

end triangle_inequality_ineq_l162_162553


namespace haley_initial_video_files_l162_162705

theorem haley_initial_video_files :
  ∀ (V : ℕ), ∃ (m d f : ℕ), m = 27 ∧ d = 11 ∧ f = 58 ∧ (m + V - d = f) → V = 42 :=
by
  intro V
  use 27, 11, 58
  intro h
  simp at h
  exact h.right.right.right

end haley_initial_video_files_l162_162705


namespace part1_part2_l162_162762

noncomputable def T := {t : ℝ | t ≤ 1}

theorem part1 (t : ℝ) :
  (∃ x : ℝ, |x - 1| - |x - 2| ≥ t) ↔ t ∈ T :=
by sorry

theorem part2 (m n : ℝ) (hm : 1 < m) (hn : 1 < n) (ht : ∀ t ∈ T, log 3 m * log 3 n ≥ t) :
  m + n = 6 :=
by sorry

end part1_part2_l162_162762


namespace maria_needs_minimum_nickels_l162_162806

theorem maria_needs_minimum_nickels :
  let book_cost := 35.50
  let twenty_dollar_count := 2
  let twenty_dollar_value := 20
  let quarter_count := 12
  let quarter_value := 0.25
  let nickel_value := 0.05
  let total_money := (twenty_dollar_count * twenty_dollar_value) 
                   + (quarter_count * quarter_value)
  in ∃ n : ℕ, (total_money + n * nickel_value) ≥ book_cost ∧ n = 150 :=
by sorry

end maria_needs_minimum_nickels_l162_162806


namespace negation_of_prop_p_l162_162871

open Classical

theorem negation_of_prop_p:
  (¬ ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≤ 1 / 2) ↔ ∃ x : ℕ, x > 0 ∧ (1 / 2) ^ x > 1 / 2 := 
by
  sorry

end negation_of_prop_p_l162_162871


namespace proof_of_cube_section_angle_l162_162661

noncomputable def cube_section_angle {K C₁ : ℝ × ℝ × ℝ} (K_coords : K = (5, 5, 4))
    (C₁_coords : C₁ = (0, 5, 5)) (BD₁_parallel : ∀ k, K + k • (C₁ - K) = C₁ + k • (0 - 5, 0 - 5, 5 - 0))
    : Prop :=
    let n₁ := (5, 20, 25)
    let n₂ := (1, 0, 0)
    let cos_angle := (n₁.1 * n₂.1 + n₁.2 * n₂.2 + n₁.3 * n₂.3) / (Real.sqrt (n₁.1^2 + n₁.2^2 + n₁.3^2) * Real.sqrt (n₂.1^2 + n₂.2^2 + n₂.3^2))
    5 * K.1 + 20 * K.2 + 25 * K.3 = 225 ∧ cos_angle = (1 / Real.sqrt 42)

theorem proof_of_cube_section_angle : cube_section_angle (rfl) (rfl) sorry :=
sorry

end proof_of_cube_section_angle_l162_162661


namespace cesaro_sum_501_l162_162250

/-- Let {a_n} be a sequence where n = 1, 2, ..., 500, and let S_n be its partial sum. 
Given that the Cesàro sum of {a_n} for n = 1, 2, ..., 500 is 2004, 
prove that the Cesàro sum of the 501-term sequence (2), (a_1), (a_2), ..., (a_{500}) is 2002. -/
theorem cesaro_sum_501 (a : ℕ → ℝ)
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = ∑ i in finset.range n, a (i + 1))
  (h2 : (∑ i in finset.range 500, S (i + 1)) / 500 = 2004) :
  (2 + (∑ i in finset.range 500, (2 + S (i + 1)))) / 501 = 2002 :=
  sorry

end cesaro_sum_501_l162_162250


namespace complete_square_eq_l162_162591

theorem complete_square_eq (x : ℝ) : (x^2 - 6 * x - 5 = 0) -> (x - 3)^2 = 14 :=
by
  intro h
  sorry

end complete_square_eq_l162_162591


namespace kim_min_pours_l162_162629

-- Define the initial conditions
def initial_volume (V : ℝ) : ℝ := V
def pour (V : ℝ) : ℝ := 0.9 * V

-- Define the remaining volume after n pours
def remaining_volume (V : ℝ) (n : ℕ) : ℝ := V * (0.9)^n

-- State the problem: After 7 pours, the remaining volume is less than half the initial volume
theorem kim_min_pours (V : ℝ) (hV : V > 0) : remaining_volume V 7 < V / 2 :=
by
  -- Because the proof is not required, we use sorry
  sorry

end kim_min_pours_l162_162629


namespace continuous_function_exists_iff_odd_l162_162647

theorem continuous_function_exists_iff_odd (n : ℕ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ (∀ y : ℝ, card {x | f x = y}.to_finset = n)) ↔ Odd n :=
sorry

end continuous_function_exists_iff_odd_l162_162647


namespace sphere_radius_of_same_volume_l162_162594

theorem sphere_radius_of_same_volume (h r : ℝ) (v_cone : ℝ) :
  h = 4 → r = 1 → v_cone = (1 / 3) * Real.pi * r^2 * h → 
  ∃ (radius : ℝ), (4 / 3) * Real.pi * radius^3 = v_cone ∧ radius = 1 :=
by
  intros h_eq r_eq v_eq
  use 1
  split
  · rw [r_eq, h_eq] at v_eq
    have v_sphere : (4 / 3) * Real.pi * (1:ℝ)^3 = (1/3) * Real.pi * (1:ℝ)^2 * 4 := by
      simp
    exact v_eq.symm.trans v_sphere
  · rfl

end sphere_radius_of_same_volume_l162_162594


namespace stratified_sampling_first_year_l162_162450

theorem stratified_sampling_first_year :
  ∀ (students_first_year students_second_year students_third_year sampled_third_year : ℕ)
  (sampled_ratio : ℚ),
  students_first_year = 300 →
  students_second_year = 260 →
  students_third_year = 280 →
  sampled_third_year = 14 →
  sampled_ratio = (14 : ℚ) / 280 →
  (sampled_first_year = (300 : ℚ) * sampled_ratio) →
  sampled_first_year = 15 :=
by
  intro students_first_year students_second_year students_third_year sampled_third_year sampled_ratio
  intro h_first h_second h_third h_sampled h_ratio h_calc
  have : sampled_first_year = 15 := by
    rw [h_first, h_third, h_sampled] at h_ratio
    norm_num at h_ratio
    rw [h_ratio] at h_calc
    norm_num at h_calc
    exact h_calc
  exact this

end stratified_sampling_first_year_l162_162450


namespace cost_per_foot_l162_162127

theorem cost_per_foot (area : ℝ) (total_cost : ℝ) (side_length : ℝ) (perimeter : ℝ) (cost_per_foot : ℝ)
  (h1 : area = 144) (h2 : total_cost = 2784) (h3 : side_length = real.sqrt area) (h4 : perimeter = 4 * side_length)
  (h5 : cost_per_foot = total_cost / perimeter) : cost_per_foot = 58 :=
by
  sorry

end cost_per_foot_l162_162127


namespace problem_part1_problem_part2_l162_162277

variable (A : Real)
variable (BC AM MN : Real)
variable (AB AC : Real → Real)
variable (bm2_sqrt3 : Real := sqrt 3)

-- Conditions
axiom area_triangle_abc : BC * sin A / 2 = bm2_sqrt3 / 2
axiom dot_product_ab_ac : AB AC = -1

-- Proof to Check: 
theorem problem_part1 :
  (cos A = -1/2) ∧ (BC ≥ sqrt 6) := by
  -- Given conditions and calculations
sorry

-- Further conditions on midpoint and length AM
axiom am_length : AM = bm2_sqrt3 / 2
axiom an_bisector : N (AC) (BC) = BM (AB) / 2

-- Proof to Check:
theorem problem_part2 :
  MN = sqrt 7 / 6 := by
  -- Given midpoint and bisector conditions
sorry

end problem_part1_problem_part2_l162_162277


namespace John_can_put_weight_on_bar_l162_162772

-- Definitions for conditions
def max_capacity : ℕ := 1000
def safety_margin : ℕ := 200  -- 20% of 1000
def johns_weight : ℕ := 250

-- Statement to prove
theorem John_can_put_weight_on_bar : ∀ (weight_on_bar : ℕ),
  weight_on_bar + johns_weight ≤ max_capacity - safety_margin → weight_on_bar = 550 :=
by
  intro weight_on_bar
  intros h_condition
  have h_max_weight : max_capacity - safety_margin = 800 := by simp [max_capacity, safety_margin]
  have h_safe_weight : 800 - johns_weight = 550 := by simp [johns_weight]
  rw [←h_safe_weight] at h_condition
  exact Eq.trans (Eq.symm h_condition) (Eq.refl 550)

end John_can_put_weight_on_bar_l162_162772


namespace exponentiation_rule_example_l162_162507

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162507


namespace ratio_work_done_5_days_l162_162066

-- Definitions and necessary conditions
def persons : Type := ℕ
def work : Type := ℕ  

def total_work (P : persons) (W : work) : Prop :=
  ∃ work_rate_P work_rate_2P, 
    W = 20 * work_rate_P ∧
    work_rate_2P = 2 * work_rate_P ∧
    W / work_rate_P = 20 ∧
    work_rate_2P = 2 * work_rate_P

-- Total work done by P persons in 20 days
def W_p_20_days (P : persons) (W : work) : work :=
  W

-- Portion of work done by 2P persons in 5 days
def W_2P_5_days (P : persons) (W : work) : work :=
  W / 20

-- Ratio of work done in 5 days to total work is 1:20
theorem ratio_work_done_5_days (P : persons) (W : work) 
  (h : total_work P W) : W_2P_5_days P W / W_p_20_days P W = 1 / 20 :=
by
  sorry

end ratio_work_done_5_days_l162_162066


namespace geographic_info_tech_helps_western_development_l162_162753

namespace GeographicInfoTech

def monitors_three_gorges_project : Prop :=
  -- Point ①
  true

def monitors_ecological_environment_meteorological_changes_and_provides_accurate_info : Prop :=
  -- Point ②
  true

def tracks_migration_tibetan_antelopes : Prop :=
  -- Point ③
  true

def addresses_ecological_environment_issues_in_southwest : Prop :=
  -- Point ④
  true

noncomputable def provides_services_for_development_western_regions : Prop :=
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes -- A (①②③)

-- Theorem stating that geographic information technology helps in ①, ②, ③ given its role
theorem geographic_info_tech_helps_western_development (h : provides_services_for_development_western_regions) :
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes := 
by
  exact h

end GeographicInfoTech

end geographic_info_tech_helps_western_development_l162_162753


namespace exists_b_c_l162_162780

theorem exists_b_c (p a : ℤ) (hp_prime : p.prime) (hdiv : p ∣ 2 * a^2 - 1) :
  ∃ b c : ℤ, p = 2 * b^2 - c^2 := 
sorry

end exists_b_c_l162_162780


namespace time_to_store_vaccine_l162_162569

def final_temp : ℤ := -24
def current_temp : ℤ := -4
def rate_of_change : ℤ := -5

theorem time_to_store_vaccine : 
  ∃ t : ℤ, current_temp + rate_of_change * t = final_temp ∧ t = 4 :=
by
  use 4
  sorry

end time_to_store_vaccine_l162_162569


namespace min_value_l162_162221

open Real

theorem min_value (x : ℝ) (hx : x > 0) : 6 * x + 1 / x^2 ≥ 7 * (6 ^ (1 / 3)) :=
sorry

end min_value_l162_162221


namespace first_player_wins_l162_162442

-- Definitions of the problem
def game_points := 20
def nonintersecting_chords_strategy := ∀ (second_player_move : Π (group : Fin 20 → Prop), Prop), 
  ∃ (first_player_move : Π (group : Fin 20 → Prop), Prop), 
    first_player_move = second_player_move

-- Statement of the theorem
theorem first_player_wins (strategy : nonintersecting_chords_strategy) : 
  ∃ (winning_strategy : nonintersecting_chords_strategy), 
    (∀ player_move, player_move = second_player_move → winning_strategy player_move = player_move) :=
begin
  -- Proof to be added here
  sorry,
end

end first_player_wins_l162_162442


namespace number_of_points_in_S2017_l162_162457

-- Define initial conditions
def S_0 : Set (ℕ × ℕ) := {(0, 0)}

def S (k : ℕ) : Set (ℕ × ℕ) :=
  if k = 0 then S_0
  else {p | ∃ q ∈ S (k - 1), (abs (p.1 - q.1) + abs (p.2 - q.2)) = 1}

-- Define the theorem statement
theorem number_of_points_in_S2017 : (S 2017).card = 4096 := 
sorry

end number_of_points_in_S2017_l162_162457


namespace smallest_polygons_l162_162857

theorem smallest_polygons (contains_84_unit_squares : Type) 
  (no_2x2_square : contains_84_unit_squares → Prop)
  (f : contains_84_unit_squares) 
  (h1 : no_2x2_square f) :
  ∃ (n : ℕ), n = 12 ∧ min_polygons contains_84_unit_squares no_2x2_square f n :=
sorry

end smallest_polygons_l162_162857


namespace a_n_formula_b_n_formula_sum_elements_C_l162_162683

theorem a_n_formula (a_n : ℕ → ℝ) (q : ℝ) (h1 : q > 1) (S_n : ℕ → ℝ) (h2 : S_n 3 = 7)
  (h3 : ∀ n, a_n n + 3, 3 * a_n (n + 1), a_n (n + 2) + 4 form_arithmetic_sequence) :
  ∀ n, a_n n = 2^(n-1) :=
by
  sorry

theorem b_n_formula (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h1 : ∀ n ∈ ℕ+, 6 * T_n n = (3 * n + 1) * b_n n + 2) :
  ∀ n, b_n n = 3 * n - 2 :=
by
  sorry

theorem sum_elements_C (a_n b_n : ℕ → ℝ)
  (h1 : ∀ n, a_n n = 2^(n-1)) 
  (h2 : ∀ n, b_n n = 3 * n - 2) :
  let A := {x | ∃ k, k ∈ (1..10) ∧ x = a_n k}
  let B := {x | ∃ k, k ∈ (1..40) ∧ x = b_n k}
  sum_elements (A ∪ B) = 3318 :=
by
  sorry

end a_n_formula_b_n_formula_sum_elements_C_l162_162683


namespace percentage_shaded_is_25_l162_162885

-- Definitions based on the conditions
def square_side_len : ℝ := 10
def rectangle_length : ℝ := 20
def rectangle_width : ℝ := 10

-- Area computation
def area_square : ℝ := square_side_len * square_side_len
def area_rectangle : ℝ := rectangle_length * rectangle_width
def overlap_len : ℝ := 5
def shaded_area : ℝ := overlap_len * rectangle_width
def percentage_shaded : ℝ := (shaded_area / area_rectangle) * 100

-- The proof statement
theorem percentage_shaded_is_25 :
  percentage_shaded = 25 := by
    sorry

end percentage_shaded_is_25_l162_162885


namespace power_of_powers_eval_powers_l162_162524

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162524


namespace larger_number_is_30_l162_162098

-- Formalizing the conditions
variables (x y : ℝ)

-- Define the conditions given in the problem
def sum_condition : Prop := x + y = 40
def ratio_condition : Prop := x / y = 3

-- Formalize the problem statement
theorem larger_number_is_30 (h1 : sum_condition x y) (h2 : ratio_condition x y) : x = 30 :=
sorry

end larger_number_is_30_l162_162098


namespace reptiles_in_swamps_l162_162095

theorem reptiles_in_swamps
  (swamps : ℕ)
  (reptiles_per_swamp : ℕ)
  (h_swamps : swamps = 4)
  (h_reptiles_per_swamp : reptiles_per_swamp = 356) :
  swamps * reptiles_per_swamp = 1424 :=
by
  rw [h_swamps, h_reptiles_per_swamp]
  exact (nat.mul_eq_one_right 4 356).symm

end reptiles_in_swamps_l162_162095


namespace team_selection_count_l162_162875

-- Definitions based on conditions
def number_of_girls : ℕ := 4
def number_of_boys : ℕ := 6
def girls_to_choose : ℕ := 3
def boys_to_choose : ℕ := 2

-- Using the combination formula for selections
def choose (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Lean statement for the proof
theorem team_selection_count : 
  choose number_of_girls girls_to_choose * choose number_of_boys boys_to_choose = 60 :=
by
  sorry

end team_selection_count_l162_162875


namespace number_of_males_in_village_l162_162433

theorem number_of_males_in_village
  (P : ℕ) (hP : P = 800)
  (k : ℕ) (hk : k = 4)
  (f : ℕ) (hf : f = 3) :
  (P / k) = 200 :=
by
  rw [hP, hk]
  norm_num

end number_of_males_in_village_l162_162433


namespace system_solution_exists_l162_162206

theorem system_solution_exists (n : ℕ) (x : Fin n → ℕ) (hn : n > 0) (hx_pos : ∀ i, 0 < x i)
  (h1 : (Finset.univ.sum (λ i, x i)) = 16)
  (h2 : (Finset.univ.sum (λ i, 1 / (x i : ℚ))) = 1) : 
  n = 4 ∧ ∀ i, x i = 4 :=
by
  sorry

end system_solution_exists_l162_162206


namespace min_value_sin_func_shift_l162_162861

theorem min_value_sin_func_shift (x : ℝ) (ϕ : ℝ) (hϕ : |ϕ| < π / 2) :
  (∀ x ∈ Set.Icc (0 : ℝ) (π / 2), f (x - π / 6) = sin (2 * x + ϕ)) →
  (∀ x ∈ Set.Icc (0 : ℝ) (π / 2), -√3 / 2 ≤ sin (2 * x - π / 3) ∧ sin (2 * x - π / 3) ≤ 1) →
  ∃ x ∈ Set.Icc (0 : ℝ) (π / 2), sin (2 * x - π / 3) = -√3 / 2 := sorry

end min_value_sin_func_shift_l162_162861


namespace patrick_age_l162_162046

theorem patrick_age (r_age_future : ℕ) (years_future : ℕ) (half_age : ℕ → ℕ) 
  (h1 : r_age_future = 30) (h2 : years_future = 2) 
  (h3 : ∀ n, half_age n = n / 2) :
  half_age (r_age_future - years_future) = 14 :=
by
  sorry

end patrick_age_l162_162046


namespace convert_and_subtract_base_8_and_base_9_to_10_l162_162602

theorem convert_and_subtract_base_8_and_base_9_to_10 :
  let a := 7 * 8^4 + 6 * 8^3 + 4 * 8^2 + 3 * 8^1 + 2 * 8^0,
      b := 2 * 9^3 + 5 * 9^2 + 4 * 9^1 + 1 * 9^0
  in a - b = 30126 :=
by
  sorry

end convert_and_subtract_base_8_and_base_9_to_10_l162_162602


namespace mod_residue_17_l162_162977

theorem mod_residue_17 : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  -- We first compute the modulo 17 residue of each term given in the problem:
  -- 513 == 0 % 17
  -- 68 == 0 % 17
  -- 289 == 0 % 17
  -- 34 == 0 % 17
  -- -10 == 7 % 17
  sorry

end mod_residue_17_l162_162977


namespace shaded_area_correct_l162_162155

noncomputable def shaded_area (r1 r2 : ℝ) (C1 C2 : ℝ × ℝ) (C : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ :=
  let AC := real.dist C1 C
  let BC := real.dist C2 C
  let AB := real.dist A B
  let α := real.arccos (AC / r2)
  let triangle_area := 1/2 * AC * real.sqrt (r2^2 - AC^2)
  let sector_area := (α / (2 * real.pi)) * (real.pi * r2^2)
  (2 * sector_area - 2 * triangle_area - real.pi * r1^2)

theorem shaded_area_correct :
  shaded_area 2 3 (3,0) (0,0) (1,2) (−1,−2) = 
  (9 * (real.arccos (2 / 3)) / real.pi) - (4 * real.sqrt 5 / 3) - 4 * real.pi := by
  sorry

end shaded_area_correct_l162_162155


namespace messenger_catch_up_time_l162_162578

-- Define the conditions
variable (x : ℝ) -- speed of Ilya Muromets (meters per second)
variable (t : ℝ) -- time in seconds after dismounting

-- Define the speeds based on given conditions
def speed_ilya := x
def speed_messenger := 2 * x
def speed_horse := 10 * x

-- Define the distance traveled in 10 seconds
def distance_ilya := 10 * speed_ilya
def distance_horse := 10 * speed_horse

-- Total distance the messenger needs to cover to catch up to Ilya Muromets
def total_distance := distance_ilya + distance_horse

-- Relative speed of the messenger towards Ilya Muromets
def relative_speed := speed_messenger - speed_ilya

-- Time required to catch up
def time_to_catch_up := total_distance / relative_speed

-- Prove that given the conditions, the time to catch up is 110 seconds
theorem messenger_catch_up_time : time_to_catch_up = 110 := by
  simp [time_to_catch_up, total_distance, relative_speed, distance_ilya, distance_horse, speed_messenger, speed_ilya, speed_horse]
  simp [x]
  sorry

end messenger_catch_up_time_l162_162578


namespace red_ball_second_draw_probability_l162_162238

theorem red_ball_second_draw_probability :
  let total_balls := 20
  let red_balls := 10
  let black_balls := 10
  let p_first_red := red_balls / total_balls
  let p_first_black := black_balls / total_balls
  let p_second_red_given_first_red := (red_balls - 1) / (total_balls - 1)
  let p_second_red_given_first_black := red_balls / (total_balls - 1)
  p_first_red * p_second_red_given_first_red + p_first_black * p_second_red_given_first_black = 1 / 2 :=
by
  -- proof steps go here
  sorry

end red_ball_second_draw_probability_l162_162238


namespace max_arith_prog_count_l162_162201

theorem max_arith_prog_count :
  ∀ (a : Fin 101 → ℝ), (∀ i j, (i < j) → a i < a j) →
  ∃ (s t u : Fin 101), s ≠ t ∧ t ≠ u ∧ s ≠ u ∧ a t = (a s + a u) / 2 →
  2500 :=
by
  sorry -- Proof is omitted

end max_arith_prog_count_l162_162201


namespace smallest_M_bound_l162_162225

theorem smallest_M_bound :
  ∃ (M : ℝ), (∀ (a b c : ℝ),
    | a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2) |
    ≤ M * (a^2 + b^2 + c^2)^2)
  ∧ M = 9 * Real.sqrt 2 / 32 := 
sorry

end smallest_M_bound_l162_162225


namespace tom_miles_mon_wed_fri_l162_162108

noncomputable def total_weeks := 52
noncomputable def weekly_fee := 100
noncomputable def total_cost := 7800
noncomputable def cost_per_mile := 0.1
noncomputable def miles_per_day := 100

-- Prove that Tom drives 10400 miles on Monday, Wednesday, and Friday in a year.
theorem tom_miles_mon_wed_fri:
  let total_fees := total_weeks * weekly_fee in
  let miles_cost := total_cost - total_fees in
  let total_miles_year := miles_cost / cost_per_mile in
  let days_driven := 3 * total_weeks * miles_per_day in
  let miles_mon_wed_fri := total_miles_year - days_driven in
  miles_mon_wed_fri = 10400 :=
by
  sorry

end tom_miles_mon_wed_fri_l162_162108


namespace find_x_l162_162172

theorem find_x (x : ℝ) (hx_pos : 0 < x) (h: (x / 100) * x = 4) : x = 20 := by
  sorry

end find_x_l162_162172


namespace deer_families_stayed_l162_162444

-- Define the initial number of deer families
def initial_deer_families : ℕ := 79

-- Define the number of deer families that moved out
def moved_out_deer_families : ℕ := 34

-- The theorem stating how many deer families stayed
theorem deer_families_stayed : initial_deer_families - moved_out_deer_families = 45 :=
by
  -- Proof will be provided here
  sorry

end deer_families_stayed_l162_162444


namespace sum_of_polynomials_l162_162379

def f (x : ℝ) : ℝ := -3 * x^2 + x - 4
def g (x : ℝ) : ℝ := -5 * x^2 + 3 * x - 8
def h (x : ℝ) : ℝ := 5 * x^2 + 5 * x + 1

theorem sum_of_polynomials (x : ℝ) : 
  f(x) + g(x) + h(x) = -3 * x^2 + 9 * x - 11 := 
by 
  sorry

end sum_of_polynomials_l162_162379


namespace log1999_not_polynomial_ratio_l162_162821

variable (x : Real)

theorem log1999_not_polynomial_ratio (f g : Real[X]) (h_coprime : IsCoprime f g) :
  ((λ x, Real.log x / Real.log 1999) ≠ (λ x, f.eval x / g.eval x)) := 
sorry

end log1999_not_polynomial_ratio_l162_162821


namespace positive_number_property_l162_162169

theorem positive_number_property (x : ℝ) (h_pos : x > 0) (h_property : 0.01 * x * x = 4) : x = 20 :=
sorry

end positive_number_property_l162_162169


namespace probability_die_sum_odd_l162_162887

namespace CoinDieProblem

-- Define the conditions and question as a statement
theorem probability_die_sum_odd :
  let coin_tosses := { outcome | outcome ∈ {'H', 'T'}^2 }
  let prob_head (coin_toss: fin 2 → char) : ℚ := 0.5
  let die_roll := fin 6
  let prob_die_odd (roll: die_roll) : ℚ := if roll ∈ {0, 2, 4} then 0.5 else 0
  let prob_2_dice_odd := 2 * 0.25 in
  (∑ outcome in coin_tosses, 
   prob_head outcome[0] * prob_head outcome[1] * 
   (if hd_count outcome[0] + hd_count outcome[1] = 0 then 0 else 
    if hd_count outcome[0] + hd_count outcome[1] = 1 then prob_die_odd else 
    prob_2_dice_odd)) = 3/8 := sorry

end CoinDieProblem

end probability_die_sum_odd_l162_162887


namespace at_least_one_heart_or_king_l162_162928

noncomputable def probability_heart_or_king_in_three_draws : ℚ := 
  1 - (36 / 52)^3

theorem at_least_one_heart_or_king :
  probability_heart_or_king_in_three_draws = 1468 / 2197 := 
by
  sorry

end at_least_one_heart_or_king_l162_162928


namespace prod_geq_n_pow_n_plus_one_l162_162027

noncomputable theory

open Real

theorem prod_geq_n_pow_n_plus_one {n : ℕ} (x : Fin (n + 1) → ℝ) (hx : ∀ i, 0 < x i) 
  (h : ∑ i : Fin (n + 1), 1 / (1 + x i) = 1) : (∏ i : Fin (n + 1), x i) ≥ n^(n+1) :=
by
  sorry

end prod_geq_n_pow_n_plus_one_l162_162027


namespace exponentiation_example_l162_162477

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162477


namespace g_value_at_50_l162_162858

noncomputable def g : ℝ → ℝ :=
sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - y ^ 2 * g x = g (x / y)) :
  g 50 = 0 :=
by
  sorry

end g_value_at_50_l162_162858


namespace find_lengths_l162_162334

theorem find_lengths (a : ℝ) (B : ℝ) (S : ℝ) (sin_B : ℝ) (cos_B : ℝ) : 
  a = 1 → B = real.pi / 4 → S = 2 → sin_B = real.sqrt 2 / 2 → cos_B = real.sqrt 2 / 2 → 
  ∃ c b : ℝ, c = 4 * real.sqrt 2 ∧ b = 5 :=
by
  intros ha hB hS hsin_B hcos_B
  use 4 * real.sqrt 2
  use 5
  -- skipping the proof since its not required.
  sorry

end find_lengths_l162_162334


namespace determine_a_plus_b_l162_162393

def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 3 then a * x + 2
  else if x >= -3 ∧ x <= 3 then x - 6
  else 2 * x - b

theorem determine_a_plus_b (a b : ℝ) (h_cont : ∀ (x : ℝ), 
   (x = 3 → (a * 3 + 2 = 3 - 6)) ∧
   (x = -3 → (-3 - 6 = 2 * (-3) - b))) : (a + b = 4/3) :=
begin
  sorry
end

end determine_a_plus_b_l162_162393


namespace sum_of_squares_of_coeffs_l162_162123

def poly_coeffs_squared_sum (p : Polynomial ℤ) : ℤ :=
  p.coeff 5 ^ 2 + p.coeff 3 ^ 2 + p.coeff 0 ^ 2

theorem sum_of_squares_of_coeffs (p : Polynomial ℤ) (h : p = 5 * (Polynomial.C 1 * Polynomial.X ^ 5 + Polynomial.C 2 * Polynomial.X ^ 3 + Polynomial.C 3)) :
  poly_coeffs_squared_sum p = 350 :=
by
  sorry

end sum_of_squares_of_coeffs_l162_162123


namespace number_of_new_students_l162_162072

/-- The average age of the original class is 40 years. --/
def avg_age_original_class := 40

/-- The average age of new students is 32 years. --/
def avg_age_new_students := 32

/-- New average age of the class after new students join is 36 years. --/
def new_avg_age := 36

/-- The original strength of the class. --/
def original_strength := 15

/-- The number of new students who joined. --/
def x := 15

theorem number_of_new_students :
  (original_strength * avg_age_original_class + x * avg_age_new_students = new_avg_age * (original_strength + x)) → (x = 15) :=
by {
  intros,
  sorry
}

end number_of_new_students_l162_162072


namespace exponentiation_identity_l162_162510

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162510


namespace sufficient_b_c_zero_necessary_c_zero_l162_162877

noncomputable def quadratic_passes_origin (a b c : ℝ) : Prop :=
  ∃ x : ℝ, (x = 0) ∧ (a * x^2 + b * x + c = 0)

theorem sufficient_b_c_zero (a b c : ℝ) (h : a ≠ 0) : quadratic_passes_origin a b c ↔ (b = 0 ∧ c = 0) :=
by {
  split,
  {
    intro h1,
    obtain ⟨x, hx, hy⟩ := h1,
    rw [hx, zero_mul, zero_mul, add_zero, add_zero] at hy,
    exact ⟨by simp, hy⟩,
  },
  {
    rintro ⟨hb, hc⟩,
    use 0,
    rw [hb, hc],
    exact ⟨rfl, by simp⟩,
  }
}

theorem necessary_c_zero (a b c : ℝ) (h : a ≠ 0) : quadratic_passes_origin a b c ↔ c = 0 :=
sorry

end sufficient_b_c_zero_necessary_c_zero_l162_162877


namespace hyperbola_eccentricity_l162_162251

noncomputable def hyperbola_eq (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
noncomputable def circle_eq (c x y : ℝ) : Prop := x^2 + y^2 = c^2
noncomputable def focal_distance (a b : ℝ) : ℝ := 2 * Real.sqrt(a^2 + b^2)

theorem hyperbola_eccentricity (a b c : ℝ) (P : ℝ × ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : hyperbola_eq a b P.1 P.2)
  (h4 : circle_eq c P.1 P.2)
  (h5 : focal_distance a b = 2 * c)
  (h6 : ∠ P (c, 0) ⟦ -c, 0 ⟧ = π / 3) : Real.sqrt 3 + 1 := sorry

end hyperbola_eccentricity_l162_162251


namespace larger_number_is_22_l162_162043

theorem larger_number_is_22 (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 :=
by
  sorry

end larger_number_is_22_l162_162043


namespace part_a_part_b_l162_162776

-- Definitions and conditions
def is_good_subset (T : finset ℕ) (S : finset ℕ) : Prop :=
  ∀ t ∈ T, ∃ s ∈ S, nat.gcd t s > 1

def A (T : finset ℕ) : finset (finset ℕ × finset ℕ) :=
  {p | (p.1 ⊆ T) ∧ (p.2 ⊆ T) ∧ (∀ x ∈ p.1, ∀ y ∈ p.2, nat.gcd x y = 1)}

-- Problem Statements
theorem part_a (T : finset ℕ) (X₀ : finset ℕ) (h₁ : ∀ x ∈ T, x ≥ 1) 
  (h₂ : ¬ is_good_subset T X₀) :
  even ((A T).filter (λ p, p.1 = X₀)).card :=
sorry

theorem part_b (T : finset ℕ) (h₁ : ∀ x ∈ T, x ≥ 1) :
  odd ((T.powerset.filter (is_good_subset T)).card) :=
sorry

end part_a_part_b_l162_162776


namespace angle_respects_triangle_circles_l162_162570

variable {α β γ : ℝ} -- Angles of the triangle.
variable {r_a r_b r_c : ℝ} -- Radii of the circles inscribed in the angles.

-- The main statement:
theorem angle_respects_triangle_circles
  (h1 : cot (γ / 2) + cot (α / 2) = 2 / r_b)
  (h2 : cot (β / 2) + cot (α / 2) = 2 / r_c)
  (h3 : cot (γ / 2) + cot (β / 2) = 2 / r_a) :
  cot (α / 2) + cot (β / 2) = cot (α / 2) + cot (γ / 2) :=
sorry

end angle_respects_triangle_circles_l162_162570


namespace lines_perpendicular_l162_162261

variable (a b : Line) (α : Plane)

theorem lines_perpendicular (h1 : Parallel a α) (h2 : Perpendicular b α) : Perpendicular a b :=
  sorry

end lines_perpendicular_l162_162261


namespace exactly_two_skills_l162_162733

theorem exactly_two_skills {total_students students_no_poetry students_no_paint students_no_instrument : Nat}
  (h1 : total_students = 150)
  (h2 : students_no_poetry = 80)
  (h3 : students_no_paint = 90)
  (h4 : students_no_instrument = 60)
  (h5 : ∀ s, (¬(s ∈ students_no_poetry) ∧ ¬(s ∈ students_no_paint) ∧ ¬(s ∈ students_no_instrument)) → False)
: total_students - (students_no_poetry + students_no_paint + students_no_instrument) = 70 := 
sorry

end exactly_two_skills_l162_162733


namespace parallel_OI_ell_l162_162184

noncomputable def is_parallel (L1 L2 : Line) : Prop := sorry

variables {A B C M N P Q S O I : Point}
variables (ell : Line)

-- Conditions
axiom h1 : (M ∈ segment AB) 
axiom h2 : (Q ∈ segment AC) 
axiom h3 : (N ∈ segment BC) 
axiom h4 : (P ∈ segment BC) 
axiom h5 : (S = intersection (line MN) (line PQ))
axiom h6 : (ell = angle_bisector ∠MSQ)
axiom h7 : (O = circumcenter triangle ABC)
axiom h8 : (I = incenter triangle ABC)

-- Target Proof
theorem parallel_OI_ell : is_parallel (line O I) ell :=
sorry

end parallel_OI_ell_l162_162184


namespace carlos_marbles_l162_162606

theorem carlos_marbles :
  ∃ N : ℕ, N > 2 ∧
  (N % 6 = 2) ∧
  (N % 7 = 2) ∧
  (N % 8 = 2) ∧
  (N % 11 = 2) ∧
  N = 3698 :=
by
  sorry

end carlos_marbles_l162_162606


namespace four_digit_combinations_l162_162240

/--
From the numbers 1, 2, 3, 4, 5, 6, and 7, if two odd numbers and two even numbers are selected to form a four-digit number without repeating digits, then the number of such four-digit numbers is 216.
-/
theorem four_digit_combinations : 
  let digits := {1, 2, 3, 4, 5, 6, 7} in
  let odds := {1, 3, 5, 7} in
  let evens := {2, 4, 6} in
  ∑ (a in (odds.toFinset.product odds.toFinset), 
     ∑ (b in (evens.toFinset.product evens.toFinset), 
       if a.1 ≠ a.2 ∧ b.1 ≠ b.2 then 
         ∑ (c in (( {a.1, a.2, b.1, b.2} : Finset ℕ).toList.permutations), 
           1)
       else 
         0)) = 216 :=
begin
  sorry
end

end four_digit_combinations_l162_162240


namespace magnitude_of_z_l162_162665

noncomputable theory

variables (z z1 z2 : ℂ)
variables (h1 : z1 ≠ z2)
variables (h2 : z1^2 = -2 - 2 * (complex.I * (real.sqrt 3)))
variables (h3 : z2^2 = -2 - 2 * (complex.I * (real.sqrt 3)))
variables (h4 : complex.abs (z - z1) = 4)
variables (h5 : complex.abs (z - z2) = 4)

theorem magnitude_of_z :
  complex.abs z = 2 * real.sqrt 3 :=
sorry

end magnitude_of_z_l162_162665


namespace f_four_l162_162680

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f_add_one : ∀ x : ℝ, f(-x + 1) = -f(x + 1)
axiom even_f_sub_one : ∀ x : ℝ, f(x - 1) = f(-x - 1)
axiom f_at_zero : f(0) = 2

theorem f_four : f(4) = -2 := by
    sorry

end f_four_l162_162680


namespace cartesian_equation_of_curve_intersection_polar_coordinate_l162_162419

theorem cartesian_equation_of_curve (ρ θ x y : ℝ) (hC : sin θ = √3 * ρ * cos θ * cos θ) :
    y = √3 * x * x → 
    (∃ (ρ θ : ℝ), ρ = 2 ∧ θ = π / 3) :=
by
  sorry

theorem intersection_polar_coordinate (t x y : ℝ) 
    (hl : x = 1 + 1/2 * t ∧ y = √3 + √3 * t) 
    (hC_cartesian : y = √3 * x * x) :
    (∃ ρ θ, 
        sqrt (x*x + y*y) = 2 ∧ atan2 y x = π / 3) :=
by
  sorry

end cartesian_equation_of_curve_intersection_polar_coordinate_l162_162419


namespace intersection_of_domains_l162_162853

open Set

theorem intersection_of_domains :
  (range (λ x : ℝ, exp x)) ∩ (range (λ x : ℝ, log x)) = {y | 0 < y} :=
by
  sorry

end intersection_of_domains_l162_162853


namespace distance_from_P_to_line_l162_162287

theorem distance_from_P_to_line :
  ∀ (x y : ℝ), (x ^ 2) / 25 + (y ^ 2) / 9 = 1 → dist ⟨x, y⟩ ⟨4, 0⟩ = 4 →
  dist ⟨x, y⟩ ⟨-25 / 4, y⟩ = 15 / 2 :=
by
  intros x y heq hdist
  sorry

end distance_from_P_to_line_l162_162287


namespace power_of_powers_l162_162492

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162492


namespace firstGradeMuffins_l162_162810

-- Define the conditions as the number of muffins baked by each class
def mrsBrierMuffins : ℕ := 18
def mrsMacAdamsMuffins : ℕ := 20
def mrsFlanneryMuffins : ℕ := 17

-- Define the total number of muffins baked
def totalMuffins : ℕ := mrsBrierMuffins + mrsMacAdamsMuffins + mrsFlanneryMuffins

-- Prove that the total number of muffins baked is 55
theorem firstGradeMuffins : totalMuffins = 55 := by
  sorry

end firstGradeMuffins_l162_162810


namespace benny_birthday_money_l162_162187

theorem benny_birthday_money (leftover : ℕ) (spent : ℕ) (total : ℕ) : 
  leftover = 33 → 
  spent = 34 → 
  total = leftover + spent → 
  total = 67 := 
by 
  intros h_leftover h_spent h_total
  rw [h_leftover, h_spent, Nat.add_comm] at h_total
  exact h_total
  sorry

end benny_birthday_money_l162_162187


namespace total_tickets_sold_l162_162162

def SeniorPrice : Nat := 10
def RegularPrice : Nat := 15
def TotalSales : Nat := 855
def RegularTicketsSold : Nat := 41

theorem total_tickets_sold : ∃ (S R : Nat), R = RegularTicketsSold ∧ 10 * S + 15 * R = TotalSales ∧ S + R = 65 :=
by
  sorry

end total_tickets_sold_l162_162162


namespace determine_base_l162_162456

theorem determine_base (b : ℕ) (h : (∑ i in Finset.range b, i) = Nat.ofDigits b [4, 5]) : b = 10 :=
by
  sorry

end determine_base_l162_162456


namespace largest_of_three_numbers_l162_162136

theorem largest_of_three_numbers 
  (a b c : ℕ)
  (hcf_eq_59 : nat.gcd (nat.gcd a b) c = 59)
  (pf1 : 13 ∣ nat.lcm (nat.lcm a b) c)
  (pf2 : 16 ∣ nat.lcm (nat.lcm a b) c)
  (pf3 : 23 ∣ nat.lcm (nat.lcm a b) c) :
  a = 282256 ∨ b = 282256 ∨ c = 282256 :=
by
  sorry

end largest_of_three_numbers_l162_162136


namespace common_points_of_line_and_circle_l162_162085

def line (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (1 + 3 * m) * x + (3 - 2 * m) * y + 8 * m - 12 = 0

def circle : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - 2 * x - 6 * y + 1 = 0

theorem common_points_of_line_and_circle :
  ∀ m : ℝ, ∃ p1 p2 : ℝ × ℝ, line m p1.1 p1.2 ∧ circle p1.1 p1.2 ∧ 
                       line m p2.1 p2.2 ∧ circle p2.1 p2.2 ∧ 
                       p1 ≠ p2 :=
by
  sorry

end common_points_of_line_and_circle_l162_162085


namespace remainder_of_polynomial_l162_162996

theorem remainder_of_polynomial (q : ℚ[X]) :
  (3 * X ^ 5 + 2 * X ^ 4 - 5 * X ^ 3 + 6 * X - 8) % (X ^ 2 + 3 * X + 2) = 34 * X + 24 := 
by sorry

end remainder_of_polynomial_l162_162996


namespace ants_on_track_possible_counts_l162_162596

theorem ants_on_track_possible_counts (n : ℕ) (x y : ℕ) 
  (h_track_length : true) -- 60 cm length is just context, not used in the proof.
  (h_speed : true) -- Speed of 1 cm/s is just context, not used directly.
  (h_collisions : 2 * x * y = 48) : 
  n = 10 ∨ n = 11 ∨ n = 14 ∨ n = 25 :=
by
  have xy_eq_24 : x * y = 24 := sorry
  -- Define the potential answers based on xy = 24.
  have valid_counts := [⟨1, 24⟩, ⟨2, 12⟩, ⟨3, 8⟩, ⟨4, 6⟩]
  have valid_answers := valid_counts.map (λ ⟨x, y⟩ => x + y)
  sorry

end ants_on_track_possible_counts_l162_162596


namespace smaller_hexagon_area_ratio_l162_162375

theorem smaller_hexagon_area_ratio
  (P Q R S T U V W X Y Z A : Point)
  (hexagon_reg : RegularHexagon P Q R S T U)
  (V_mid : Midpoint V P Q)
  (W_mid : Midpoint W Q R)
  (X_mid : Midpoint X R S)
  (Y_mid : Midpoint Y S T)
  (Z_mid : Midpoint Z T U)
  (A_mid : Midpoint A U P)
  (bounded_hex : BoundedHexagon (P, V) (Q, W) (R, X) (S, Y) (T, Z) (U, A)) :
  let m := 4
  let n := 7
  m + n = 11 := by sorry

end smaller_hexagon_area_ratio_l162_162375


namespace disjoint_subsets_less_elements_l162_162452

open Nat

theorem disjoint_subsets_less_elements (m : ℕ) (A B : Finset ℕ) (hA : A ⊆ Finset.range (m + 1))
  (hB : B ⊆ Finset.range (m + 1)) (h_disjoint : Disjoint A B)
  (h_sum : A.sum id = B.sum id) : ↑(A.card) < m / Real.sqrt 2 ∧ ↑(B.card) < m / Real.sqrt 2 := 
sorry

end disjoint_subsets_less_elements_l162_162452


namespace focus_of_parabola_l162_162972

theorem focus_of_parabola (x y : ℝ) : (y^2 = 4 * x) → (x = 2 ∧ y = 0) :=
by
  sorry

end focus_of_parabola_l162_162972


namespace max_value_of_sum_l162_162794

theorem max_value_of_sum (
    a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
    (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1) ≤ 1) :=
begin
    -- Proof would go here
    sorry
end

end max_value_of_sum_l162_162794


namespace exponentiation_identity_l162_162514

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162514


namespace power_of_powers_eval_powers_l162_162525

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162525


namespace find_m_squared_plus_n_squared_l162_162142

theorem find_m_squared_plus_n_squared (m n : ℝ) (h1 : (m - n) ^ 2 = 8) (h2 : (m + n) ^ 2 = 2) : m ^ 2 + n ^ 2 = 5 :=
by
  sorry

end find_m_squared_plus_n_squared_l162_162142


namespace trains_meet_time_l162_162890

noncomputable def kmph_to_mps (v_kmph : ℝ) : ℝ :=
  (v_kmph * 1000) / 3600

theorem trains_meet_time :
  let l1 := 90 -- length of the first train in meters
  let l2 := 95 -- length of the second train in meters
  let d := 250 -- initial distance between the trains in meters
  let v1_kmph := 64 -- speed of the first train in kmph
  let v2_kmph := 92 -- speed of the second train in kmph
  let v1 := kmph_to_mps v1_kmph -- speed of the first train in m/s
  let v2 := kmph_to_mps v2_kmph -- speed of the second train in m/s
  let relative_speed := v1 + v2 -- relative speed in m/s
  let total_distance := l1 + l2 + d -- total distance to be covered in meters
  let time := total_distance / relative_speed -- time for the trains to meet in seconds
  time ≈ 10.04 := sorry

end trains_meet_time_l162_162890


namespace tetrahedron_volume_and_height_l162_162139

noncomputable def vector := (ℝ × ℝ × ℝ)
def A1 : vector := (4, -1, 3)
def A2 : vector := (-2, 1, 0)
def A3 : vector := (0, -5, 1)
def A4 : vector := (3, 2, -6)

def scalarTripleProduct (u v w : vector) : ℝ :=
  u.1 * (v.2 * w.3 - v.3 * w.2) - u.2 * (v.1 * w.3 - v.3 * w.1) + u.3 * (v.1 * w.2 - v.2 * w.1)

def vectorMagnitude (u : vector) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

def crossProduct (u v : vector) : vector :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def volume (A1 A2 A3 A4 : vector) : ℝ :=
  (1 / 6) * real.abs (scalarTripleProduct (A2 - A1) (A3 - A1) (A4 - A1))

def areaOfTriangle (A1 A2 A3 : vector) : ℝ :=
  (1 / 2) * vectorMagnitude (crossProduct (A2 - A1) (A3 - A1))

def height (volume : ℝ) (triangleArea : ℝ) : ℝ :=
  (3 * volume) / triangleArea

theorem tetrahedron_volume_and_height :
  volume A1 A2 A3 A4 = 136 / 3 ∧ height (volume A1 A2 A3 A4) (areaOfTriangle A1 A2 A3) = 17 / Real.sqrt 5 :=
by
  sorry

end tetrahedron_volume_and_height_l162_162139


namespace john_total_spending_l162_162362

def t_shirt_price : ℝ := 20
def num_t_shirts : ℝ := 3
def t_shirt_offer_discount : ℝ := 0.50
def t_shirt_total_cost : ℝ := (2 * t_shirt_price) + (t_shirt_price * t_shirt_offer_discount)

def pants_price : ℝ := 50
def num_pants : ℝ := 2
def pants_total_cost : ℝ := pants_price * num_pants

def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def jacket_total_cost : ℝ := jacket_original_price * (1 - jacket_discount)

def hat_price : ℝ := 15

def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10
def shoes_total_cost : ℝ := shoes_original_price * (1 - shoes_discount)

def clothes_tax_rate : ℝ := 0.05
def shoes_tax_rate : ℝ := 0.08

def clothes_total_cost : ℝ := t_shirt_total_cost + pants_total_cost + jacket_total_cost + hat_price
def total_cost_before_tax : ℝ := clothes_total_cost + shoes_total_cost

def clothes_tax : ℝ := clothes_total_cost * clothes_tax_rate
def shoes_tax : ℝ := shoes_total_cost * shoes_tax_rate

def total_cost_including_tax : ℝ := total_cost_before_tax + clothes_tax + shoes_tax

theorem john_total_spending :
  total_cost_including_tax = 294.57 := by
  sorry

end john_total_spending_l162_162362


namespace power_calc_l162_162466

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162466


namespace max_value_expr_max_l162_162381

noncomputable def max_value_expr (x : ℝ) : ℝ :=
  (x^2 + 3 - (x^4 + 9).sqrt) / x

theorem max_value_expr_max (x : ℝ) (hx : 0 < x) :
  max_value_expr x ≤ (6 * (6:ℝ).sqrt) / (6 + 3 * (2:ℝ).sqrt) :=
sorry

end max_value_expr_max_l162_162381


namespace solution_exists_l162_162714

noncomputable def problem :=
  let a b c d : ℕ in
  0 ≤ a ∧ a < 11 ∧
  0 ≤ b ∧ b < 11 ∧
  0 ≤ c ∧ c < 11 ∧
  0 ≤ d ∧ d < 11 ∧
  (a + 3*b + 4*c + 2*d) % 11 = 3 ∧
  (3*a + b + 2*c + d) % 11 = 5 ∧
  (2*a + 4*b + c + 3*d) % 11 = 7 ∧
  (a + b + c + d) % 11 = 2 →
  (a * b * c * d) % 11 = 9

theorem solution_exists : problem :=
sorry

end solution_exists_l162_162714


namespace Q_y_coordinates_product_is_minus_82_l162_162050

noncomputable def Q_y_coordinates_product : ℝ :=
  let Q_x := 4
  let distance := 10
  let P_x := 1
  let P_y := -3
  let distance_formula := λ y : ℝ, Real.sqrt ((Q_x - P_x) ^ 2 + (P_y - y) ^ 2)
  let ys := { y : ℝ | distance_formula y = distance }
  (-3 + Real.sqrt 91) * (-3 - Real.sqrt 91)

theorem Q_y_coordinates_product_is_minus_82 :
  Q_y_coordinates_product = -82 :=
sorry

end Q_y_coordinates_product_is_minus_82_l162_162050


namespace area_of_triangle_l162_162408

theorem area_of_triangle {a b c : ℕ} (h1 : a = 15) (h2 : b = 14) (h3 : c = 13) :
  let S := real.sqrt ((1 / 4.0 : ℝ) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2.0)^2))
  in S = 84.0 :=
by
  have S_def : S = real.sqrt ((1 / 4.0 : ℝ) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2.0)^2)),
  from rfl,

  sorry

end area_of_triangle_l162_162408


namespace count_odd_digit_4_digit_div5_l162_162706

def odd_digits := [1, 3, 5, 7, 9]

theorem count_odd_digit_4_digit_div5 :
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ d ∈ (n.digits 10), d ∈ odd_digits) ∧ n % 5 = 0}.to_finset.card = 125 :=
by
  sorry

end count_odd_digit_4_digit_div5_l162_162706


namespace arithmetic_sequence_sum_l162_162751

theorem arithmetic_sequence_sum (b : ℕ → ℝ) (h_arith : ∀ n, b (n+1) - b n = b 2 - b 1) (h_b5 : b 5 = 2) :
  b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 18 := 
sorry

end arithmetic_sequence_sum_l162_162751


namespace find_a_range_for_two_distinct_roots_l162_162298

def f (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem find_a_range_for_two_distinct_roots :
  ∀ (a : ℝ), 3 ≤ a ∧ a ≤ 7 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = a ∧ f x2 = a :=
by
  -- The proof will be here
  sorry

end find_a_range_for_two_distinct_roots_l162_162298


namespace song_liking_possibilities_l162_162592

-- Definitions for the problem
def liking_sets (songs : Finset String) (Amy Beth Jo : Finset String) : Prop :=
  -- no song is liked by all three
  (∀ s, s ∈ songs → ¬ (s ∈ Amy ∧ s ∈ Beth ∧ s ∈ Jo))
  -- Amy and Beth like at least 2 songs Jo does not like
  ∧ (Amy ∩ Beth).card ≥ 2
  -- Beth and Jo like at least 2 songs Amy does not like
  ∧ (Beth ∩ Jo).card ≥ 2
  -- Jo and Amy like at least 2 songs Beth does not like
  ∧ (Jo ∩ Amy).card ≥ 2

def possible_ways (songs : Finset String) : ℕ :=
  -- We assume there are 5 songs being discussed
  ∑ (Amy Beth Jo : Finset String), if liking_sets songs Amy Beth Jo then 1 else 0

-- Main theorem statement
theorem song_liking_possibilities : 
  possible_ways (Finset.range 5) = 150 :=
sorry

end song_liking_possibilities_l162_162592


namespace arrangement_exists_l162_162021

noncomputable def seq_exists (S : Finset ℕ) (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ Nat.gcd a b = 1 ∧ (
    ∃ (s : Fin 1995 → ℕ), ∀ i : Fin 1994, (s (i + 1) - s i) % 1995 = a ∨
      (s (i + 1) - s i) % 1995 = -a ∨
      (s (i + 1) - s i) % 1995 = b ∨
      (s (i + 1) - s i) % 1995 = -b
  )

theorem arrangement_exists :
  seq_exists (Finset.range 1995) a b :=
sorry

end arrangement_exists_l162_162021


namespace problem1_problem2_l162_162557

-- Problem (1)
theorem problem1 (f : ℝ → ℝ) (h : ∀ x ≠ 0, f (2 / x + 2) = x + 1) : 
  ∀ x ≠ 2, f x = x / (x - 2) :=
sorry

-- Problem (2)
theorem problem2 (f : ℝ → ℝ) (h : ∃ k b, ∀ x, f x = k * x + b ∧ k ≠ 0)
  (h' : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) :
  ∀ x, f x = 2 * x + 7 :=
sorry

end problem1_problem2_l162_162557


namespace max_value_is_377_l162_162022

noncomputable def max_value (a b c : ℝ^3) : ℝ :=
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2

theorem max_value_is_377 (a b c : ℝ^3) (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hc : ‖c‖ = 4) :
  max_value a b c = 377 :=
sorry

end max_value_is_377_l162_162022


namespace total_pieces_of_clothing_l162_162190

-- Define the conditions:
def boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

-- Define the target statement:
theorem total_pieces_of_clothing : (boxes * (scarves_per_box + mittens_per_box)) = 32 :=
by
  sorry

end total_pieces_of_clothing_l162_162190


namespace power_of_powers_eval_powers_l162_162523

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162523


namespace problem1_problem2_problem3_problem4_l162_162143

open Set

namespace MathProof

variable (U A B : Set ℕ)

def universal_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13}
def A_set := {2, 4, 6, 8}
def B_set := {3, 4, 5, 6, 8, 9, 11}

-- (1) Prove A ∪ B = {2, 3, 4, 5, 6, 8, 9, 11}
theorem problem1 : A_set ∪ B_set = {2, 3, 4, 5, 6, 8, 9, 11} := by
  unfold A_set B_set
  sorry

-- (2) Prove C_U A = {1, 3, 5, 7, 9, 10, 11, 13}
theorem problem2 : universal_set \ A_set = {1, 3, 5, 7, 9, 10, 11, 13} := by
  unfold universal_set A_set
  sorry

-- (3) Prove C_U (A ∩ B) = {1, 2, 3, 5, 7, 9, 10, 11, 13}
theorem problem3 : universal_set \ (A_set ∩ B_set) = {1, 2, 3, 5, 7, 9, 10, 11, 13} := by
  unfold universal_set A_set B_set
  sorry

-- (4) Prove A ∪ (C_U B) = {1, 2, 4, 6, 7, 8, 10, 13}
theorem problem4 : A_set ∪ (universal_set \ B_set) = {1, 2, 4, 6, 7, 8, 10, 13} := by
  unfold universal_set A_set B_set
  sorry

end MathProof

end problem1_problem2_problem3_problem4_l162_162143


namespace income_expenditure_ratio_l162_162081

variable (I S E : ℕ)
variable (hI : I = 16000)
variable (hS : S = 3200)
variable (hExp : S = I - E)

theorem income_expenditure_ratio (I S E : ℕ) (hI : I = 16000) (hS : S = 3200) (hExp : S = I - E) : I / Nat.gcd I E = 5 ∧ E / Nat.gcd I E = 4 := by
  sorry

end income_expenditure_ratio_l162_162081


namespace factorize_poly_l162_162979

theorem factorize_poly (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) := by
  sorry

end factorize_poly_l162_162979


namespace not_perpendicular_l162_162577

theorem not_perpendicular
  (a : ℝ^3)
  (α : set (ℝ^3))
  (h1 : ∃ p, p ∈ α ∧ p ∈ a)
  (h2 : ∃ L : fin 2011 → set (ℝ^3), ∀ i, (L i) ∈ α ∧ is_equidistant L i a ∧ ¬(a ∩ L i).nonempty)
: ¬ (is_perpendicular a α) :=
sorry

end not_perpendicular_l162_162577


namespace Sammy_has_8_bottle_caps_l162_162415

-- Definitions representing the conditions
def BilliesBottleCaps := 2
def JaninesBottleCaps := 3 * BilliesBottleCaps
def SammysBottleCaps := JaninesBottleCaps + 2

-- Goal: Prove that Sammy has 8 bottle caps
theorem Sammy_has_8_bottle_caps : 
  SammysBottleCaps = 8 := 
sorry

end Sammy_has_8_bottle_caps_l162_162415


namespace profit_percent_is_correct_l162_162326

-- Definitions based on problem conditions
def selling_price : ℝ := 100
def cost_price : ℝ := 0.82 * selling_price
def profit : ℝ := selling_price - cost_price
def profit_percent : ℝ := (profit / cost_price) * 100

-- The theorem statement
theorem profit_percent_is_correct : profit_percent ≈ 21.95 := 
by
    sorry

end profit_percent_is_correct_l162_162326


namespace probability_square_area_l162_162400

theorem probability_square_area (AB : ℝ) (M : ℝ) (h1 : AB = 12) (h2 : 0 ≤ M) (h3 : M ≤ AB) :
  (∃ (AM : ℝ), (AM = M) ∧ (36 ≤ AM^2 ∧ AM^2 ≤ 81)) → 
  (∃ (p : ℝ), p = 1/4) :=
by
  sorry

end probability_square_area_l162_162400


namespace log_expression_value_l162_162205

noncomputable def log_expression : ℝ :=
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2

theorem log_expression_value : log_expression = 3 / 2 :=
  sorry

end log_expression_value_l162_162205


namespace circumcenter_of_perpendicular_and_equal_distances_l162_162663

open EuclideanGeometry

variables {P A B C O : Point}

theorem circumcenter_of_perpendicular_and_equal_distances (
  h₁ : distance P A = distance P B,
  h₂ : distance P B = distance P C,
  h₃ : perp P O (plane A B C),
  h₄ : foot P (plane A B C) = O
) : circumcenter A B C = O := 
sorry

end circumcenter_of_perpendicular_and_equal_distances_l162_162663


namespace vector_expression_non_simplifiable_l162_162181

theorem vector_expression_non_simplifiable 
  (V : Type) [Add V] [Neg V] [Zero V] 
  (A B C D M O : V)
  (AB CD BC AD MB CM BM OC OA : V)
  (h1 : BC = C - B)
  (h2 : CD = D - C)
  (h3 : AD = D - A)
  (h4 : MB = B - M)
  (h5 : CM = M - C)
  (h6 : BM = M - B)
  (h7 : OC = C - O)
  (h8 : OA = A - O) :
  ¬ ((OC - OA - CD) = AD) :=
sorry

end vector_expression_non_simplifiable_l162_162181


namespace arccos_of_sqrt3_div_2_l162_162616

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l162_162616


namespace arccos_of_sqrt3_div_2_l162_162615

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l162_162615


namespace exponentiation_rule_example_l162_162499

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162499


namespace locus_of_F_l162_162017

-- Define the circle and its diameters
def circle (k : Type) (center : k) (radius : ℝ) (AB CD : k × k) : Prop :=
  -- AB and CD are diameters of the circle
  is_diameter AB center radius ∧ is_diameter CD center radius ∧ 
  -- AB ⊥ CD
  is_perpendicular AB CD center

-- Define the point P on the circle
def on_circle (P : k) (k : Type) (center : k) (radius : ℝ) : Prop :=
  is_on_circle P center radius

-- Define point E
def intersection_PA_CD (P A E CD : k) : Prop :=
  line_intersection P A CD E

-- Define line through E parallel to AB
def parallel_E_AB (E AB: k) : Prop :=
  is_parallel E AB

-- Define intersection of a parallel line with CP
def intersection_parallel_CP (E AB P C F : k) : Prop :=
  line_intersection E AB P C F

-- Define the locus of F
def locus_F (F : k) (BD : k) (P A: k) : Prop :=
  on_line F BD ∧ F ≠ A

-- The proof statement with all conditions
theorem locus_of_F (k : Type) (center : k) (radius : ℝ) (AB CD P A E F : k):
  circle k center radius AB CD →
  on_circle P k center radius →
  intersection_PA_CD P A E CD →
  parallel_E_AB E AB →
  intersection_parallel_CP E AB P center F →
  locus_F F BD P A := 
begin
  sorry
end

end locus_of_F_l162_162017


namespace linda_fraction_savings_l162_162803

theorem linda_fraction_savings (savings tv_cost : ℝ) (f : ℝ) 
  (h1 : savings = 800) 
  (h2 : tv_cost = 200) 
  (h3 : f * savings + tv_cost = savings) : 
  f = 3 / 4 := 
sorry

end linda_fraction_savings_l162_162803


namespace angle_B_lt_90_l162_162725

theorem angle_B_lt_90 {a b c : ℝ} (h_arith : b = (a + c) / 2) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (A B C : ℝ), B < 90 :=
sorry

end angle_B_lt_90_l162_162725


namespace direction_vector_of_line_l162_162867

theorem direction_vector_of_line (P : Matrix (Fin 3) (Fin 3) ℚ) 
    (P_eq: P = ![
        [1/5, -2/5, 0],
        [-2/5, 4/5, 0],
        [0, 0, 0]
      ]) :
    ∃ v : Fin 3 → ℤ, v = ![1, -2, 0] ∧ gcd v[0] (gcd v[1] v[2]) = 1 :=
by sorry

end direction_vector_of_line_l162_162867


namespace tangents_parallel_l162_162968

variable {R : Type*} [Field R]

-- Let f be a function from ratios to slopes
variable (φ : R -> R)

-- Given points (x, y) and (x₁, y₁) with corresponding conditions
variable (x x₁ y y₁ : R)

-- Conditions
def corresponding_points := y / x = y₁ / x₁
def homogeneous_diff_eqn := ∀ x y, (y / x) = φ (y / x)

-- Prove that the tangents are parallel
theorem tangents_parallel (h_corr : corresponding_points x x₁ y y₁)
  (h_diff_eqn : ∀ (x x₁ y y₁ : R), y' = φ (y / x) ∧ y₁' = φ (y₁ / x₁)) :
  y' = y₁' :=
by
  sorry

end tangents_parallel_l162_162968


namespace exponentiation_identity_l162_162509

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162509


namespace seashells_remaining_l162_162106

theorem seashells_remaining (initial_seashells : ℕ) (seashells_given : ℕ) : initial_seashells = 679 → seashells_given = 172 → (initial_seashells - seashells_given) = 507 :=
by
  intros h1 h2
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add'
    (by
      norm_num
      exact Nat.add_sub_cancel_left 507 172)

end seashells_remaining_l162_162106


namespace count_valid_n_l162_162232

theorem count_valid_n 
  (n : ℤ) 
  (h1 : ∃ k : ℤ, 8000 * (2/5)^n = k)
  (h2 : 3 ∣ 8000 * (2/5)^n) : 
  ∃ (s : Finset ℤ), s.card = 10 ∧ 
  ∀ x ∈ s, 8000 * (2/5)^x ∈ ℤ ∧ (3 ∣ 8000 * (2/5)^x) :=
by
  sorry

end count_valid_n_l162_162232


namespace min_elements_in_set_l162_162874

-- Problem statement: Define the necessary conditions for a set A
def satisfies_condition (A : Set ℝ) : Prop :=
  ∀ a ∈ A, (1 / (1 - a)) ∈ A

-- We aim to prove that if a set A satisfies the condition, then it must have at least 3 elements.
theorem min_elements_in_set (A : Set ℝ) (h : satisfies_condition A) : ∃ s ⊆ A, s.card >= 3 := sorry

end min_elements_in_set_l162_162874


namespace zhen_zhen_test_score_l162_162540

theorem zhen_zhen_test_score
  (avg1 avg2 : ℝ) (n m : ℝ)
  (h1 : avg1 = 88)
  (h2 : avg2 = 90)
  (h3 : n = 4)
  (h4 : m = 5) :
  avg2 * m - avg1 * n = 98 :=
by
  -- Given the hypotheses h1, h2, h3, and h4,
  -- we need to show that avg2 * m - avg1 * n = 98.
  sorry

end zhen_zhen_test_score_l162_162540


namespace work_rate_problem_l162_162151

theorem work_rate_problem :
  ∃ (x : ℝ), 
    (0 < x) ∧ 
    (10 * (1 / x + 1 / 40) = 0.5833333333333334) ∧ 
    (x = 30) :=
by
  sorry

end work_rate_problem_l162_162151


namespace area_of_rectangle_l162_162604

def length : ℝ := 0.5
def width : ℝ := 0.24

theorem area_of_rectangle :
  length * width = 0.12 :=
by
  sorry

end area_of_rectangle_l162_162604


namespace find_p_q_d_l162_162034

def f (p q d : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then p * x + 4
  else if x = 0 then p * q
  else q * x + d

theorem find_p_q_d :
  ∃ p q d : ℕ, f p q d 3 = 7 ∧ f p q d 0 = 6 ∧ f p q d (-3) = -12 ∧ (p + q + d = 13) :=
by
  sorry

end find_p_q_d_l162_162034


namespace sum_first_2017_terms_seq_l162_162297

theorem sum_first_2017_terms_seq :
  let f (n : ℕ) := n * (n + 1)
  let S (n : ℕ) := ∑ i in finset.range n, (1 / f i.succ)
  S 2017 = 2017 / 2018 := by
  sorry

end sum_first_2017_terms_seq_l162_162297


namespace complex_enclosed_area_is_correct_l162_162895

noncomputable def complex_enclosed_area (z : ℂ) : ℂ := 1 / z

theorem complex_enclosed_area_is_correct :
  (∀ z : ℂ, (1 - 2 * complex.I) * z + (-2 * complex.I - 1) * complex.conj z = 6 * complex.I → 
    set.area (set_of (complex_enclosed_area z)) = (5 * Real.pi) / 36) :=
sorry

end complex_enclosed_area_is_correct_l162_162895


namespace inequality_f_l162_162797

noncomputable def f (x y z : ℝ) : ℝ :=
  x * y + y * z + z * x - 2 * x * y * z

theorem inequality_f (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ f x y z ∧ f x y z ≤ 7 / 27 :=
  sorry

end inequality_f_l162_162797


namespace hexagon_area_ratio_l162_162347

theorem hexagon_area_ratio (ABCDEF : Hexagon) (s : ℝ) (h_reg_hex : regular ABCDEF s)
  (mid_BC : (∃ M : Point, midpoint M B C)) (mid_EF : (∃ N : Point, midpoint N E F))
  (g : Line) (h : Line) (h_g_parallel : ∀ p, (p ∈ sides ABCDEF AB ↔ p ∈ Line containing BC M))
  (h_h_parallel : ∀ q, (q ∈ sides ABCDEF DE ↔ q ∈ Line containing EF N))
  (quad := quadrilateral formed by g h CD FA)
: (area quad / area ABCDEF) = 1/4 :=
sorry

end hexagon_area_ratio_l162_162347


namespace solve_quadratic_rewriting_l162_162827

noncomputable def quadratic_rewriting {g h j : ℤ} : Prop :=
  ∀ (x : ℝ), (4 * x^2 - 16 * x - 21) = (g * x + h)^2 + j

theorem solve_quadratic_rewriting :
  ∃ g h j : ℤ, quadratic_rewriting ∧ g * h = -8 :=
by
  sorry

end solve_quadratic_rewriting_l162_162827


namespace megatech_basic_astrophysics_degrees_l162_162131

def budget_allocation (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :=
  100 - (microphotonics + home_electronics + food_additives + gm_microorganisms + industrial_lubricants)

noncomputable def degrees_for_astrophysics (percentage: ℕ) :=
  (percentage * 360) / 100

theorem megatech_basic_astrophysics_degrees (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  gm_microorganisms = 29 →
  industrial_lubricants = 8 →
  degrees_for_astrophysics (budget_allocation microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants) = 54 :=
by
  sorry

end megatech_basic_astrophysics_degrees_l162_162131


namespace baseball_card_decrease_l162_162564

theorem baseball_card_decrease (x : ℝ) (h : (1 - x / 100) * (1 - x / 100) = 0.64) : x = 20 :=
by
  sorry

end baseball_card_decrease_l162_162564


namespace simplify_expr1_simplify_expr2_simplify_expr3_simplify_expr4_l162_162063

-- Problem 1
theorem simplify_expr1 :
  ( (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - 2 = 1 ) :=
sorry

-- Problem 2
theorem simplify_expr2 :
  ( (Real.sqrt 12 - Real.sqrt 48 + 3 * Real.sqrt (1/3)) / (2 * Real.sqrt 3) = -1/2 ) :=
sorry

-- Problem 3
theorem simplify_expr3 :
  ( Real.cbrt (-8) - (1/3:ℝ)⁻¹ + 202 * 3^0 = -4 ) :=
sorry

-- Problem 4
theorem simplify_expr4 :
  ( (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 ) :=
sorry

end simplify_expr1_simplify_expr2_simplify_expr3_simplify_expr4_l162_162063


namespace log_3897_between_consecutive_integers_l162_162878

theorem log_3897_between_consecutive_integers :
  ∃ a b : ℕ, (∀ x : ℝ, 3 < x ∧ x < 4 → x = Real.log10 3897) ∧ a + b = 7 := 
sorry

end log_3897_between_consecutive_integers_l162_162878


namespace least_prime_factor_of_9_pow_4_minus_9_pow_3_l162_162900

theorem least_prime_factor_of_9_pow_4_minus_9_pow_3 : 
  ∃ p, p = 2 ∧ prime p ∧ p ∣ (9^4 - 9^3) :=
by
  -- We can formalize each step of the solution but for the statement itself we can
  -- just confirm the existence of such a prime (specifically 2 in this case) that divides the given expression.
  sorry

end least_prime_factor_of_9_pow_4_minus_9_pow_3_l162_162900


namespace john_remaining_amount_l162_162768

-- Definitions based on the conditions
def saved_amount_base8 := 5555
def airline_ticket_cost := 1200
def visa_cost := 200
def saved_amount_base10 := 5 * 8^3 + 5 * 8^2 + 5 * 8^1 + 5 * 8^0 -- Convert base-8 to base-10
def total_cost := airline_ticket_cost + visa_cost

-- Prove John has the correct remaining amount
theorem john_remaining_amount : 
  let remaining_amount : ℕ := saved_amount_base10 - total_cost in 
  remaining_amount = 1525 := 
by 
  sorry

end john_remaining_amount_l162_162768


namespace faye_scored_47_pieces_l162_162644

variable (X : ℕ) -- X is the number of pieces of candy Faye scored on Halloween.

-- Definitions based on the conditions
def initial_candy_count (X : ℕ) : ℕ := X - 25
def after_sister_gave_40 (X : ℕ) : ℕ := initial_candy_count X + 40
def current_candy_count (X : ℕ) : ℕ := after_sister_gave_40 X

-- Theorem to prove the number of pieces of candy Faye scored on Halloween
theorem faye_scored_47_pieces (h : current_candy_count X = 62) : X = 47 :=
by
  sorry

end faye_scored_47_pieces_l162_162644


namespace maximal_points_coloring_l162_162621

/-- Given finitely many points in the plane where no three points are collinear,
which are colored either red or green, such that any monochromatic triangle
contains at least one point of the other color in its interior, the maximal number
of such points is 8. -/
theorem maximal_points_coloring (points : Finset (ℝ × ℝ))
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ¬ ∃ k b, ∀ p ∈ [p1, p2, p3], p.2 = k * p.1 + b)
  (colored : (ℝ × ℝ) → Prop)
  (h_coloring : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    colored p1 = colored p2 → colored p2 = colored p3 →
    ∃ p, p ∈ points ∧ colored p ≠ colored p1) :
  points.card ≤ 8 :=
sorry

end maximal_points_coloring_l162_162621


namespace diagonal_of_square_l162_162138

theorem diagonal_of_square (length_rect width_rect : ℝ) (h1 : length_rect = 45) (h2 : width_rect = 40)
  (area_rect : ℝ) (h3 : area_rect = length_rect * width_rect) (area_square : ℝ) (h4 : area_square = area_rect)
  (side_square : ℝ) (h5 : side_square^2 = area_square) (diagonal_square : ℝ) (h6 : diagonal_square = side_square * Real.sqrt 2) :
  diagonal_square = 60 := by
  sorry

end diagonal_of_square_l162_162138


namespace b_remaining_work_days_l162_162542

-- Definitions of the conditions
def together_work (a b: ℕ) := a + b = 12
def alone_work (a: ℕ) := a = 20
def c_work (c: ℕ) := c = 30
def initial_work_days := 5

-- Question to prove:
theorem b_remaining_work_days (a b c : ℕ) (h1 : together_work a b) (h2 : alone_work a) (h3 : c_work c) : 
  let b_rate := 1 / 30 
  let remaining_work := 25 / 60
  let work_to_days := remaining_work / b_rate
  work_to_days = 12.5 := 
sorry

end b_remaining_work_days_l162_162542


namespace problem_1_holds_problem_2_holds_l162_162254

-- Definitions for the sequences and initial problem
def a_seq : ℕ → ℕ
| 0     := 2
| (n+1) := 3 * a_seq n + 2 * n - 1

def b_seq : ℕ → ℕ
| 0     := 2
| 1     := 6
-- we will skip implementing further terms since the goal is to state the result

-- Sum of the first n terms of b_seq
def T_seq (n : ℕ) : ℕ := (finset.range n).sum b_seq

noncomputable def problem_1 : Prop := 
  ∀ n : ℕ, a_seq n + n = 3^n

noncomputable def problem_2 : Prop :=
  ∀ n : ℕ, b_seq n = 4 * n - 2

-- Assertion statements
theorem problem_1_holds : problem_1 := by sorry
theorem problem_2_holds : problem_2 := by sorry

end problem_1_holds_problem_2_holds_l162_162254


namespace find_integer_pairs_l162_162222

theorem find_integer_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (m n : ℤ), (m, n) ∈ S ↔ mn ≤ 0 ∧ m^3 + n^3 - 37 * m * n = 343) ∧ S.card = 9 :=
sorry

end find_integer_pairs_l162_162222


namespace exp_eval_l162_162479

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162479


namespace A_eq_B_l162_162192

variables (n : ℕ) (A_girls B_girls : Fin n → Type) (B_boys : Fin (2 * n - 1) → Type)

-- City A: Each girl knows every boy
-- City B: Girl g_i knows boys b_1, b_2, ..., b_(2i-1)
axiom CityA_Knows : ∀ (g : Fin n), (Fin n → Prop)
axiom CityB_Knows : ∀ (i : Fin n) (j : ℕ), (∀ (j : ℕ), j < 2 * (i + 1) - 1 → B_boys j)

-- A(r) and B(r) definitions
noncomputable def A_r (r : ℕ) : ℕ :=
  if h : r ≤ n then
    let p := factorial n div (factorial (n - r) * factorial r) in
    p * p * factorial r
  else
    0

noncomputable def B_r (r : ℕ) : ℕ :=
  if h : r ≤ n then
    sorry -- Recursive definition as per solution, using axioms
  else
    0

theorem A_eq_B (r : ℕ) (hr : r ≤ n) : A_r n r = B_r n r :=
begin
  sorry -- The proof would be based on the provided key steps from the solution
end

end A_eq_B_l162_162192


namespace angle_between_MP_NQ_l162_162735

def isMidpoint (p1 p2 mid : Point3D) : Prop :=
  dist p1 mid = dist p2 mid

def isCentroid (v1 v2 v3 centroid : Point3D) : Prop :=
  dist v1 centroid = dist v2 centroid ∧ dist v2 centroid = dist v3 centroid

noncomputable def angleBetweenLines (p1 p2 p3 p4 : Point3D) : ℝ :=
  sorry  -- Placeholder for actual calculation

theorem angle_between_MP_NQ (A B C D M P N Q : Point3D)
  (h1 : isRegularTetrahedron A B C D) 
  (h2 : isMidpoint A B M) 
  (h3 : isCentroid B C D P) 
  (h4 : isMidpoint B C N) 
  (h5 : isCentroid A B D Q) :
  angleBetweenLines M P N Q ≈ 112.885 :
  sorry

end angle_between_MP_NQ_l162_162735


namespace construct_triangle_l162_162197

theorem construct_triangle (h b m : ℝ) (H : (m^2 ≥ h^2)) :
  ∃ (A B C : (ℝ × ℝ)),
    A = (0, 0) ∧
    B = (b, 0) ∧
    (C = (b/2 + Real.sqrt (m^2 - h^2), h) ∨ C = (b/2 - Real.sqrt (m^2 - h^2), h) ∨ C = (b/2 + Real.sqrt (m^2 - h^2), -h) ∨ C = (b/2 - Real.sqrt (m^2 - h^2), -h)) :=
begin
  -- No proof required
  sorry
end

end construct_triangle_l162_162197


namespace shepherds_sheep_l162_162586

theorem shepherds_sheep (x y : ℕ) 
  (h1 : x - 4 = y + 4) 
  (h2 : x + 4 = 3 * (y - 4)) : 
  x = 20 ∧ y = 12 := 
by 
  sorry

end shepherds_sheep_l162_162586


namespace negation_of_p_l162_162051

open Classical

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x > 2

-- Define the negation of proposition p
def not_p : Prop := ∃ x : ℝ, x^2 + x ≤ 2

theorem negation_of_p : ¬p ↔ not_p :=
by sorry

end negation_of_p_l162_162051


namespace ones_digit_37_37_pow_28_28_l162_162641

theorem ones_digit_37_37_pow_28_28 :
  let ones_digit_cycle_of_7 := [7, 9, 3, 1];
  ∀ n : ℕ, ∀ m : ℕ,
    (37 ≡ 7 [MOD 10]) →
    (28 ≡ 0 [MOD 4]) →
    (0 ^ m ≡ 0 [MOD 4]) →
    (37 * (28 ^ 28) ≡ 0 [MOD 4]) →
    ones_digit_cycle_of_7.nth_le ((37 * (28 ^ 28)) % 4 % 4) (sorry) = 1 :=
by
  intros _
         n m h37mod10 h28mod4 h0mod4 h37mul28mod4
  sorry

end ones_digit_37_37_pow_28_28_l162_162641


namespace sphere_to_cube_surface_area_ratio_l162_162094

namespace SurfaceAreaRatio

-- Given conditions
variables (a : ℝ)
def cube_edge_length := a
def sphere_radius := (Real.sqrt 3 / 2) * a
def sphere_surface_area := 4 * Real.pi * (sphere_radius a)^2
def cube_surface_area := 6 * a^2

-- Proof goal
theorem sphere_to_cube_surface_area_ratio (a : ℝ) : 
  (sphere_surface_area a) / (cube_surface_area a) = Real.pi / 2 :=
by
  sorry

end SurfaceAreaRatio

end sphere_to_cube_surface_area_ratio_l162_162094


namespace problem1_problem2_problem3_problem4_l162_162962

-- Problem 1
theorem problem1 : (-49) - (+91) - (-5) + (-9) = -144 := by
  sorry

-- Problem 2
theorem problem2 : -4 / 36 * (-1 / 9) = (1 : ℚ) / 81 := by
  sorry

-- Problem 3
theorem problem3 : 24 * (1 / 6 - 0.75 - 2/3) = -30 := by
  sorry

-- Problem 4
theorem problem4 : -(2^4) - 6 / (-2) * abs (-1 / 3) = -15 := by
  sorry

end problem1_problem2_problem3_problem4_l162_162962


namespace transform_C2_to_C1_l162_162286

noncomputable def C_1 : ℝ → ℝ := λ x : ℝ, Real.sin (2 * x + 2 * Real.pi / 3)
noncomputable def C_2 : ℝ → ℝ := λ x : ℝ, Real.sin x

theorem transform_C2_to_C1 :
  ∀ x : ℝ, (C_2 (2 * x - Real.pi / 3)) = C_1 x :=
by
  sorry

end transform_C2_to_C1_l162_162286


namespace age_ratio_2_to_1_l162_162368

variable (x : ℕ)
variable (julio_age james_age : ℕ)
variable (future_age_ratio : ℕ → ℕ → Prop)

-- Define the conditions as given in the problem
def julio_current_age : ℕ := 36
def james_current_age : ℕ := 11

-- Condition stating Julio's future age will be twice James's future age
def future_age_condition (x : ℕ) : Prop :=
  julio_current_age + x = 2 * (james_current_age + x)

axiom future_age_ratio_def (j_age jms_age : ℕ) : Prop :=
  (j_age = julio_current_age + x) ∧ (jms_age = james_current_age + x) → future_age_ratio j_age jms_age

-- Problem: Prove the ratio of Julio's age to James's age after a certain number of years is 2:1
theorem age_ratio_2_to_1 (x : ℕ) (j : jul_age james : james_age) : future_age_condition x → future_age_ratio (julio_current_age + x) (james_current_age + x) :=
by
  sorry

end age_ratio_2_to_1_l162_162368


namespace measure_angle_RSP_l162_162746

def Line (Point : Type) [InnerProductSpace ℝ Point] (p1 p2 : Point) := ∃ (a : ℝ), ∀ t : ℝ, p1 = a • p2

variable (Point : Type) [InnerProductSpace ℝ Point]

variables (A B P Q R S : Point)

def Parallel (l1 l2 : Set Point) : Prop := ∃ (v : Point), v ≠ 0 ∧ (v ∈ Span ℝ l1) ∧ (v ∈ Span ℝ l2)

axiom parallel_AB_PQ : Parallel (Line Point A B) (Line Point P Q)

axiom R_on_PQ : ∃ t : ℝ, R = t • (Q - P) + P

variables (a b c x : ℝ)

axiom angle_PRS : angle P R S = 2 * x
axiom angle_ARB : angle A R B = 3 * x
axiom angle_BRA : angle B R A = 4 * x

theorem measure_angle_RSP : angle R S P = 60 :=
by sorry

end measure_angle_RSP_l162_162746


namespace simplify_complex_expr_l162_162834

theorem simplify_complex_expr :
  ((-5 - 3 * complex.i) - (2 - 7 * complex.i)) * 2 = -14 + 8 * complex.i := 
by
  sorry

end simplify_complex_expr_l162_162834


namespace power_of_powers_l162_162493

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162493


namespace revenue_difference_l162_162742

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end revenue_difference_l162_162742


namespace exponentiation_identity_l162_162512

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162512


namespace total_students_end_of_year_l162_162739

theorem total_students_end_of_year 
  (initial_students : ℝ) 
  (added_students : ℝ) 
  (new_students : ℝ)
  (initial_count : initial_students = 10.0)
  (added_count : added_students = 4.0)
  (new_count : new_students = 42.0) :
  initial_students + added_students + new_students = 56.0 := 
by 
  rw [initial_count, added_count, new_count]
  exact rfl

end total_students_end_of_year_l162_162739


namespace polynomial_remainder_l162_162224

theorem polynomial_remainder (x : ℝ) :
  let p := 5 * x^3 - 9 * x^2 + 3 * x + 17 in
  p.eval 2 = 27 :=
by
  let p := Polynomial.C 5 * Polynomial.X ^ 3 - Polynomial.C 9 * Polynomial.X ^ 2 + Polynomial.C 3 * Polynomial.X + Polynomial.C 17
  have step1 : p.eval 2 = 5 * (2 : ℝ)^3 - 9 * (2 : ℝ)^2 + 3 * (2 : ℝ) + 17 := by sorry
  have step2 : 5 * (2 : ℝ)^3 - 9 * (2 : ℝ)^2 + 3 * (2 : ℝ) + 17 = 27 := by norm_num
  exact eq.trans step1 step2

end polynomial_remainder_l162_162224


namespace sum_of_series_l162_162409

theorem sum_of_series (n : ℕ) : 
  ∑ k in finset.range (n + 1), 8 * (10^k - 10^(k - 1)) = 
  8 * (10^(n + 1) - 10 - 9 * n) / 81 :=
sorry

end sum_of_series_l162_162409


namespace ratio_of_heights_l162_162946

-- Given conditions
def cone_original_circumference : ℝ := 18 * π
def cone_original_height : ℝ := 24
def cone_new_volume : ℝ := 162 * π

-- Definition of radius based on original circumference
def cone_radius : ℝ := cone_original_circumference / (2 * π)
-- Definition of new height variable
variable (h : ℝ)

-- Volume formula for the shortened cone
def cone_new_volume_eq : Prop := (1 / 3) * π * (cone_radius ^ 2) * h = cone_new_volume

-- Proof goal
theorem ratio_of_heights (h : ℝ) (h_given : cone_new_volume_eq) : h / cone_original_height = 1 / 4 := by
  sorry

end ratio_of_heights_l162_162946


namespace difference_in_weight_between_green_and_red_l162_162726

-- Define the number of small and large peaches for red, yellow, and green
def n_sr := 7
def n_lr := 5
def n_sy := 71
def n_ly := 14
def n_sg := 8
def n_lg := 11

-- Assume each small peach weighs "x" units and each large peach weighs "2x" units
variable (x : ℝ)

-- Calculate the total weight of red, yellow, and green peaches
def weight_red := n_sr * x + n_lr * 2 * x
def weight_yellow := n_sy * x + n_ly * 2 * x
def weight_green := n_sg * x + n_lg * 2 * x

-- Prove that the difference in weight between green and red peaches is "13x" units
theorem difference_in_weight_between_green_and_red :
  (weight_green - weight_red) = 13 * x :=
by
  sorry

end difference_in_weight_between_green_and_red_l162_162726


namespace probability_of_exceeding_rounding_error_l162_162873

-- Define the parameters for the problem
noncomputable def scale_division_value : ℝ := 0.1
noncomputable def rounding_threshold : ℝ := 0.02
noncomputable def pdf_uniform_on_interval (x a b : ℝ) : ℝ :=
  if x ∈ set.Icc a b then 1 / (b - a) else 0

-- Define the integral of the PDF from a to b
noncomputable def integral_of_pdf (a b c d : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ (x : ℝ) in set.Ioc a b, f x

-- The statement to prove
theorem probability_of_exceeding_rounding_error :
  integral_of_pdf 0.02 0.08 (-scale_division_value/2) (scale_division_value/2) 
    (λ x, pdf_uniform_on_interval x (-scale_division_value/2) (scale_division_value/2)) = 0.6 :=
by
  sorry

end probability_of_exceeding_rounding_error_l162_162873


namespace simplify_sin_cos_diff_l162_162062

theorem simplify_sin_cos_diff :
  sin (15 * π / 180)^4 - sin (75 * π / 180)^4 = - (real.sqrt 3 / 2) := by
sorry

end simplify_sin_cos_diff_l162_162062


namespace matt_homework_time_l162_162807

variable (T : ℝ)
variable (h_math : 0.30 * T = math_time)
variable (h_science : 0.40 * T = science_time)
variable (h_others : math_time + science_time + 45 = T)

theorem matt_homework_time (h_math : 0.30 * T = math_time)
                             (h_science : 0.40 * T = science_time)
                             (h_others : math_time + science_time + 45 = T) :
  T = 150 := by
  sorry

end matt_homework_time_l162_162807


namespace arccos_sqrt3_div_2_eq_pi_div_6_l162_162608

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l162_162608


namespace mode_and_median_correct_l162_162950

open Finset

noncomputable def data_set := {2, 7, 6, 3, 4, 7}

def mode (s : Finset ℕ) : ℕ := 7 -- as 7 appears most frequently
def median (s : Finset ℕ) : ℚ := 5 -- the median is calculated as (4 + 6) / 2

theorem mode_and_median_correct : 
  mode data_set = 7 ∧ median data_set = 5 := by
  -- proof will go here
  sorry

end mode_and_median_correct_l162_162950


namespace Sa_divisible_by_10_l162_162406

theorem Sa_divisible_by_10 (a : ℕ) (h₀ : a ∈ (set.range 10)) :
    let ka (k : ℕ) := 10 * k + a
    let Sa := ∑ k in finset.range 10, (ka k)^2005
    Sa % 10 = 0 :=
by
  sorry

end Sa_divisible_by_10_l162_162406


namespace negation_equivalence_l162_162084

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by
  sorry

end negation_equivalence_l162_162084


namespace factorize_poly_l162_162978

theorem factorize_poly (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) := by
  sorry

end factorize_poly_l162_162978


namespace question_1_question_2_period_question_2_monotonic_interval_l162_162308

variables {x : ℝ} (k : ℤ)

def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x)
def b (x : ℝ) : ℝ × ℝ := (cos x, sin x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def is_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem question_1 (hx : x ∈ Icc 0 (π / 2)) (hab : ‖a x‖ = ‖b x‖) : x = π / 6 :=
sorry

theorem question_2_period : is_period f π :=
sorry

theorem question_2_monotonic_interval :
  ∀ k : ℤ, ∀ x, ∀ y, (-(π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π ∧ -(π / 6) + k * π ≤ y ∧ y ≤ π / 3 + k * π) →
  (x ≤ y → f x ≤ f y) :=
sorry

end question_1_question_2_period_question_2_monotonic_interval_l162_162308


namespace sum_of_coefficients_l162_162203

noncomputable def polynomial : Polynomial ℤ := 
  2 * (4 * X^6 + 9 * X^3 - 5) + 8 * (X^4 - 8 * X + 6)

theorem sum_of_coefficients : polynomial.eval 1 = 8 := by
  sorry

end sum_of_coefficients_l162_162203


namespace find_x_l162_162048

-- Define the initial point A with coordinates A(x, -2)
def A (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the transformation of moving 5 units up and 3 units to the right to obtain point B
def transform (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 + 5)

-- Define the final point B with coordinates B(1, y)
def B (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the proof problem
theorem find_x (x y : ℝ) (h : transform (A x) = B y) : x = -2 :=
by sorry

end find_x_l162_162048


namespace triangle_AB_value_l162_162757

-- Define the necessary geometric entities and relationships
noncomputable def area_of_triangle (a b c : ℝ) (theta : ℝ) : ℝ :=
  (1/2) * a * b * real.sin theta

/-- 
  Given a triangle ABC with ∠A = 60°, AC = 2, point D lies on side BC, 
  AD is the angle bisector of ∠CAB, and the area of △ADB is 2√3, then the value of AB is 2 + 2√3 
--/
theorem triangle_AB_value {A B C D : Type} [geometry Point Line triangle Proper_Archimedean_Categorial] :
  ∀ (AC AB AD : ℝ) (theta : ℝ), 
  AC = 2 → 
  theta = real.pi / 3 → -- 60 degrees in radians
  (area_of_triangle AC AD (theta / 2)) + 2 * real.sqrt 3 =
  (area_of_triangle AC AB theta) →
  area_of_triangle AB AD (theta / 2) = 2 * real.sqrt 3 →
  AB = 2 + 2 * real.sqrt 3 :=
  sorry

end triangle_AB_value_l162_162757


namespace sum_of_midpoint_coordinates_l162_162119

-- Defining the endpoints of the segment
def pointA : (ℝ × ℝ) := (10, 7)
def pointB : (ℝ × ℝ) := (4, -3)

-- Defining the midpoint coordinates
def midpoint : (ℝ × ℝ) := ((pointA.1 + pointB.1) / 2, (pointA.2 + pointB.2) / 2)

-- Proving that the sum of the coordinates of the midpoint is 9
theorem sum_of_midpoint_coordinates : midpoint.1 + midpoint.2 = 9 :=
by
  -- Proof will be written here; using sorry to skip it for now
  sorry

end sum_of_midpoint_coordinates_l162_162119


namespace alyssa_plums_correct_l162_162954

def total_plums : ℕ := 27
def jason_plums : ℕ := 10
def alyssa_plums : ℕ := 17

theorem alyssa_plums_correct : alyssa_plums = total_plums - jason_plums := by
  sorry

end alyssa_plums_correct_l162_162954


namespace find_x_l162_162171

theorem find_x (x : ℝ) (hx_pos : 0 < x) (h: (x / 100) * x = 4) : x = 20 := by
  sorry

end find_x_l162_162171


namespace coefficient_of_x_neg3_in_expansion_l162_162200

theorem coefficient_of_x_neg3_in_expansion :
  let T_r (r : ℕ) := (nat.choose 6 r) * (2^(6 - r)) * ((-1)^r) * (12 - 3 * r)
  coef := T_r 5
  coef = -12 :=
by
  sorry

end coefficient_of_x_neg3_in_expansion_l162_162200


namespace part_a_part_b_part_c_l162_162028

def pn (n : ℕ) : ℝ := (1 + Real.sqrt 2 + Real.sqrt 3) ^ n +
                      (1 - Real.sqrt 2 + Real.sqrt 3) ^ n +
                      (1 + Real.sqrt 2 - Real.sqrt 3) ^ n +
                      (1 - Real.sqrt 2 - Real.sqrt 3) ^ n

def qn (n : ℕ) : ℝ := (1 + Real.sqrt 2 + Real.sqrt 3) ^ n -
                      (1 - Real.sqrt 2 + Real.sqrt 3) ^ n +
                      (1 + Real.sqrt 2 - Real.sqrt 3) ^ n -
                      (1 - Real.sqrt 2 - Real.sqrt 3) ^ n

def rn (n : ℕ) : ℝ := (1 + Real.sqrt 2 + Real.sqrt 3) ^ n +
                      (1 - Real.sqrt 2 + Real.sqrt 3) ^ n -
                      (1 + Real.sqrt 2 - Real.sqrt 3) ^ n -
                      (1 - Real.sqrt 2 - Real.sqrt 3) ^ n

def sn (n : ℕ) : ℝ := (1 + Real.sqrt 2 + Real.sqrt 3) ^ n -
                      (1 - Real.sqrt 2 + Real.sqrt 3) ^ n -
                      (1 + Real.sqrt 2 - Real.sqrt 3) ^ n +
                      (1 - Real.sqrt 2 - Real.sqrt 3) ^ n

theorem part_a :
  tendsto (λ n, (pn n) / (qn n)) at_top (𝓝 (Real.sqrt 2)) :=
by
  sorry

theorem part_b :
  tendsto (λ n, (pn n) / (rn n)) at_top (𝓝 (Real.sqrt 3)) :=
by
  sorry

theorem part_c :
  tendsto (λ n, (pn n) / (sn n)) at_top (𝓝 (Real.sqrt 6)) :=
by
  sorry

end part_a_part_b_part_c_l162_162028


namespace range_of_a_l162_162316

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 ↔ x > Real.log a / Real.log 2) → 0 < a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l162_162316


namespace min_value_of_quadratic_l162_162265

open Real

theorem min_value_of_quadratic 
  (x y z : ℝ) 
  (h : 3 * x + 2 * y + z = 1) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ 3 / 34 := 
sorry

end min_value_of_quadratic_l162_162265


namespace ticket_price_for_children_l162_162367

open Nat

theorem ticket_price_for_children
  (C : ℕ)
  (adult_ticket_price : ℕ := 12)
  (num_adults : ℕ := 3)
  (num_children : ℕ := 3)
  (total_cost : ℕ := 66)
  (H : num_adults * adult_ticket_price + num_children * C = total_cost) :
  C = 10 :=
sorry

end ticket_price_for_children_l162_162367


namespace q_fibonacci_spiral_distance_l162_162230

theorem q_fibonacci_spiral_distance (q : ℝ) (hq : 0 < q ∧ q < 1) :
  let x := (1 + q) / (1 + q^2),
      y := (1 - q) / (1 + q^2)
  in (x - 1/2)^2 + (y - 1/2)^2 = 1/2 := 
by
  sorry

end q_fibonacci_spiral_distance_l162_162230


namespace moles_CH3Cl_formed_l162_162640

noncomputable def moles (x : String) := x

theorem moles_CH3Cl_formed (CH4 Cl2 CH3Cl HCl : String)
  (reaction : moles CH4 + moles Cl2 = moles CH3Cl + moles HCl)
  (initial_moles_CH4 : real)
  (initial_moles_Cl2 : real) :
  initial_moles_CH4 = 2 ∧ initial_moles_Cl2 = 2 →
  moles CH3Cl = 2 :=
by
  sorry

end moles_CH3Cl_formed_l162_162640


namespace hyperbola_eccentricity_l162_162675

noncomputable def eccentricity_e (a1 c: ℝ) : ℝ := sqrt ((4 - (c / a1)^2) / 3)
def sqrt_two_over_two : ℝ := real.sqrt (2) / 2

theorem hyperbola_eccentricity
  (a1 a2 c: ℝ)
  (e: ℝ)
  (h_ecc_ellipse: sqrt_two_over_two = 1 / (real.sqrt 2 / 2))
  (h_shared_foci: 4 * c^2 = a1^2 + 3 * a2^2)
  (h_angle_P: cos (π / 3) = 1/2) :
  e = real.sqrt(6) / 2 :=
by
  sorry

end hyperbola_eccentricity_l162_162675


namespace symmetric_about_origin_implies_odd_l162_162282

variable {F : Type} [Field F] (f : F → F)
variable (x : F)

theorem symmetric_about_origin_implies_odd (H : ∀ x, f (-x) = -f x) : f x + f (-x) = 0 := 
by 
  sorry

end symmetric_about_origin_implies_odd_l162_162282


namespace find_a_b_l162_162702

variables (A B : Set ℝ) (a b : ℝ)

-- Definitions based on conditions
def set_A : Set ℝ := { x | (-2 < x ∧ x < -1) ∨ x > 1 }
def set_B : Set ℝ := { x | x^2 + a * x + b ≤ 0 }

-- The theorem statement
theorem find_a_b (h1 : A = set_A)
  (h2 : B = set_B)
  (h3 : A ∪ B = {x | -2 < x})
  (h4 : A ∩ B = {x | 1 < x ∧ x ≤ 3}) :
  a = -4 ∧ b = 3 := 
sorry

end find_a_b_l162_162702


namespace sum_of_midpoint_coordinates_l162_162122

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 10) (h2 : y1 = 7) (h3 : x2 = 4) (h4 : y2 = -3) :
  let M := (x1 + x2) / 2, (y1 + y2) / 2 in
  (M.1 + M.2) = 9 :=
by 
  have hM : M = ((10 + 4) / 2, (7 + -3) / 2), from sorry,
  have hM' : M = (7, 2), from sorry,
  have sum_midpoint := 7 + 2,
  show sum_midpoint = 9, 
  from sorry

end sum_of_midpoint_coordinates_l162_162122


namespace find_a_plus_b_l162_162645

variable (a : ℝ) (b : ℝ)
def op (x y : ℝ) : ℝ := x + 2 * y + 3

theorem find_a_plus_b (a b : ℝ) (h1 : op (op (a^3) (a^2)) a = b)
    (h2 : op (a^3) (op (a^2) a) = b) : a + b = 21/8 :=
  sorry

end find_a_plus_b_l162_162645


namespace percentage_increase_l162_162089

theorem percentage_increase (P : ℝ) (h : 200 * (1 + P/100) * 0.70 = 182) : 
  P = 30 := 
sorry

end percentage_increase_l162_162089


namespace solve_f_1991_2_1990_l162_162231

-- Define the sum of digits function for an integer k
def sum_of_digits (k : ℕ) : ℕ := k.digits 10 |>.sum

-- Define f1(k) as the square of the sum of digits of k
def f1 (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

-- Define the recursive sequence fn as given in the problem
def fn : ℕ → ℕ → ℕ
| 0, k => k
| n + 1, k => f1 (fn n k)

-- Define the specific problem statement
theorem solve_f_1991_2_1990 : fn 1991 (2 ^ 1990) = 4 := sorry

end solve_f_1991_2_1990_l162_162231


namespace no_positive_integers_m_n_l162_162416

theorem no_positive_integers_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  m^3 + 11^3 ≠ n^3 :=
sorry

end no_positive_integers_m_n_l162_162416


namespace competition_points_l162_162572

noncomputable def valid_pairs : Finset (ℕ × ℕ) :=
  {(25, 2), (12, 4), (3, 13)}

theorem competition_points (n k : ℕ) (h : n ≥ 2) (received_points_per_day : ∀ day : ℕ, day ≤ k → Finset (Fin n) → ℕ)
  (total_points : Fin n → ℕ) (h_distinct_points : ∀ day ≤ k, receive_distinct_points day)
  (h_equal_points : ∀ player : Fin n, total_points player = 26) :
  (n, k) ∈ valid_pairs :=
sorry

def receive_distinct_points (day : ℕ) (day_leq_k : day ≤ k) : Prop :=
  ∃ points : Fin n → ℕ, (∀ player : Fin n, points player ∈ Finset.range (n - 1) + 1) ∧
                         (∀ player1 player2 : Fin n, player1 ≠ player2 → points player1 ≠ points player2)

end competition_points_l162_162572


namespace probability_two_blue_marbles_l162_162161

def jar_contents : nat × nat × nat × nat := (3, 4, 9, 4) -- Red, Blue, White, Green marbles

def total_marbles : nat := jar_contents.1 + jar_contents.2 + jar_contents.3 + jar_contents.4

/--
Given a jar with 3 red marbles, 4 blue marbles, 9 white marbles, and 4 green marbles,
prove that the probability of drawing two blue marbles without replacement is 3/95.
-/
theorem probability_two_blue_marbles : 
  total_marbles = 20 → 
  (jar_contents.2 > 1 → (4 / 20) * (3 / 19) = 3 / 95) := by
  intros,
  sorry

end probability_two_blue_marbles_l162_162161


namespace major_premise_of_irrational_l162_162924

theorem major_premise_of_irrational (π : ℝ) (h1 : ¬∃ (n m : ℕ), (π = (n : ℝ) / (m : ℝ))) (h2 : ∀ (r : ℝ), ¬∃ (n m : ℕ), (r = (n : ℝ) / (m : ℝ)) → (is_infinite_non_repeating_decimal r)):
  ∀ (x : ℝ), ¬∃ (n m : ℕ), (x = (n : ℝ) / (m : ℝ)) → (is_infinite_non_repeating_decimal x) :=
by
  sorry

end major_premise_of_irrational_l162_162924


namespace grandpa_age_times_jungmin_age_l162_162533

-- Definitions based on the conditions
def grandpa_age_last_year : ℕ := 71
def jungmin_age_last_year : ℕ := 8
def grandpa_age_this_year : ℕ := grandpa_age_last_year + 1
def jungmin_age_this_year : ℕ := jungmin_age_last_year + 1

-- The statement to prove
theorem grandpa_age_times_jungmin_age :
  grandpa_age_this_year / jungmin_age_this_year = 8 :=
by
  sorry

end grandpa_age_times_jungmin_age_l162_162533


namespace expression_A_is_fraction_l162_162536

-- Definition of the expressions
def A := 1 / x

-- Lean 4 statement to prove A is a fraction
theorem expression_A_is_fraction (x : ℝ) (hx : x ≠ 0) : ∃ (n d : ℝ), A = n / d :=
by
  unfold A
  use 1, x
  sorry

end expression_A_is_fraction_l162_162536


namespace handshake_count_l162_162732

theorem handshake_count
  (total_people : ℕ := 40)
  (groupA_size : ℕ := 30)
  (groupB_size : ℕ := 10)
  (groupB_knowsA_5 : ℕ := 3)
  (groupB_knowsA_0 : ℕ := 7)
  (handshakes_between_A_and_B5 : ℕ := groupB_knowsA_5 * (groupA_size - 5))
  (handshakes_between_A_and_B0 : ℕ := groupB_knowsA_0 * groupA_size)
  (handshakes_within_B : ℕ := groupB_size * (groupB_size - 1) / 2) :
  handshakes_between_A_and_B5 + handshakes_between_A_and_B0 + handshakes_within_B = 330 :=
sorry

end handshake_count_l162_162732


namespace net_increase_correct_l162_162386

-- Definitions for the given conditions
def S1 : ℕ := 10
def B1 : ℕ := 15
def S2 : ℕ := 12
def B2 : ℕ := 8
def S3 : ℕ := 9
def B3 : ℕ := 11

def P1 : ℕ := 250
def P2 : ℕ := 275
def P3 : ℕ := 260
def C1 : ℕ := 100
def C2 : ℕ := 110
def C3 : ℕ := 120

def Sale_profit1 : ℕ := S1 * P1
def Sale_profit2 : ℕ := S2 * P2
def Sale_profit3 : ℕ := S3 * P3

def Repair_cost1 : ℕ := B1 * C1
def Repair_cost2 : ℕ := B2 * C2
def Repair_cost3 : ℕ := B3 * C3

def Net_profit1 : ℕ := Sale_profit1 - Repair_cost1
def Net_profit2 : ℕ := Sale_profit2 - Repair_cost2
def Net_profit3 : ℕ := Sale_profit3 - Repair_cost3

def Total_net_profit : ℕ := Net_profit1 + Net_profit2 + Net_profit3

def Net_Increase : ℕ := (B1 - S1) + (B2 - S2) + (B3 - S3)

-- The theorem to be proven
theorem net_increase_correct : Net_Increase = 3 := by
  sorry

end net_increase_correct_l162_162386


namespace average_reciprocal_value_l162_162971

variable {n : ℕ}

def average_reciprocal (a : ℕ → ℝ) (n : ℕ) : ℝ := n / (∑ i in finset.range n, a i)

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := (a n + 1) / 4

theorem average_reciprocal_value {a : ℕ → ℝ} : 
  average_reciprocal a n = 1 / (2 * n + 1) → 
  (∑ i in finset.range 2017, 1 / (b a i * b a (i+1))) = 2017 / 2018 :=
sorry

end average_reciprocal_value_l162_162971


namespace ce_de_sum_l162_162728

open Real

-- Define the given conditions
def radius : ℝ := 10
def AB_length : ℝ := 2 * radius
def AE : ℝ := AB_length - 4
def BE : ℝ := 4
def angle_AEC : ℝ := π / 6  -- 30 degrees in radians

-- The definition of CE^2 + DE^2 given the above conditions
theorem ce_de_sum : ∀ (radius AE BE angle_AEC : ℝ),
  radius = 10 →
  BE = 4 →
  angle_AEC = π / 6 →
  CE^2 + DE^2 = 200 :=
begin
  intros,
  sorry -- To be proven
end

end ce_de_sum_l162_162728


namespace alfred_gain_percent_l162_162547

theorem alfred_gain_percent :
  let purchase_price := 4700
  let repair_costs := 800
  let selling_price := 5800
  let total_cost := purchase_price + repair_costs
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 5.45 := 
by
  sorry

end alfred_gain_percent_l162_162547


namespace find_m_l162_162681
noncomputable theory

-- Definitions and conditions
def line (x : ℝ) : ℝ := 3 * x + 1
def curve (x m n : ℝ) : ℝ := x^3 + m * x + n
def point := (1 : ℝ, 4 : ℝ)
def tangent_condition (m n : ℝ) : Prop := curve 1 m n = 4 ∧ (3 + m = 3)

-- Statement to prove
theorem find_m (m n : ℝ) (h : tangent_condition m n) : m = 0 :=
sorry

end find_m_l162_162681


namespace student_missed_20_l162_162952

theorem student_missed_20 {n : ℕ} (S_correct : ℕ) (S_incorrect : ℕ) 
    (h1 : S_correct = n * (n + 1) / 2)
    (h2 : S_incorrect = S_correct - 20) : 
    S_incorrect = n * (n + 1) / 2 - 20 := 
sorry

end student_missed_20_l162_162952


namespace remainder_1493_1998_mod_500_l162_162902

theorem remainder_1493_1998_mod_500 :
  (1493 * 1998) % 500 = 14 :=
by {
  have h1493 : 1493 % 500 = 493 := rfl,
  have h1998 : 1998 % 500 = 498 := rfl,
  have h1 : 493 % 500 = 493 := nat.mod_eq_of_lt (by norm_num),
  have h2 : 498 % 500 = 498 := nat.mod_eq_of_lt (by norm_num),
  calc
    (1493 * 1998) % 500
        = (493 * 498) % 500 : by rw [←h1493, ←h1998]
    ... = ((500 - 7) * (500 - 2)) % 500 : by norm_num [sub_eq_add_neg]
    ... = (-7 * -2) % 500 : by norm_num
    ... = 14 % 500 : by norm_num

  done
}

end remainder_1493_1998_mod_500_l162_162902


namespace power_of_powers_eval_powers_l162_162521

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162521


namespace exp_eval_l162_162486

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162486


namespace even_friends_group_exists_l162_162729

theorem even_friends_group_exists {n : ℕ} (boys girls : Finset ℕ)
  (friendship : ∀ b ∈ boys, ∃ evens : Finset ℕ, (evens.card % 2 = 0) ∧ evens ⊆ girls) :
  ∃ G ⊆ boys, ∀ g ∈ girls, (Finset.filter (λ b, b ∈ G ∧ friendship b) boys).card % 2 = 0 :=
by {
  sorry
}

end even_friends_group_exists_l162_162729


namespace cookie_and_cheese_stick_recipes_l162_162599

def students_usually_attend : ℕ := 150
def attendance_rate : ℚ := 0.60
def cookies_per_student : ℕ := 3
def cookies_per_recipe : ℕ := 15
def cheese_sticks_per_student : ℕ := 1
def cheese_sticks_per_recipe : ℕ := 20

theorem cookie_and_cheese_stick_recipes :
  let students_attending := students_usually_attend * attendance_rate in
  let total_cookies_needed := students_attending * cookies_per_student in
  let total_cheese_sticks_needed := students_attending * cheese_sticks_per_student in
  let cookie_recipes_needed := ceil (total_cookies_needed / cookies_per_recipe) in
  let cheese_stick_recipes_needed := ceil (total_cheese_sticks_needed / cheese_sticks_per_recipe) in
  cookie_recipes_needed = 18 ∧ cheese_stick_recipes_needed = 5 :=
by 
  sorry

end cookie_and_cheese_stick_recipes_l162_162599


namespace first_day_exceeds_200_l162_162727

-- Bacteria population doubling function
def bacteria_population (n : ℕ) : ℕ := 4 * 3 ^ n

-- Prove the smallest day where bacteria count exceeds 200 is 4
theorem first_day_exceeds_200 : ∃ n : ℕ, bacteria_population n > 200 ∧ ∀ m < n, bacteria_population m ≤ 200 :=
by 
    -- Proof will be filled here
    sorry

end first_day_exceeds_200_l162_162727


namespace conference_no_complete_K6_l162_162920

theorem conference_no_complete_K6 (G : SimpleGraph (Fin 500)) (h : ∀ v, G.degree v = 400) :
  ¬ ∃ (S : Finset (Fin 500)), S.card = 6 ∧ ∀ (v w : Fin 500), v ∈ S → w ∈ S → v ≠ w → G.adj v w :=
sorry

end conference_no_complete_K6_l162_162920


namespace OS_perpendicular_BC_l162_162799

noncomputable def centroid {α : Type*} [euclidean_space α] (A B C : Point α) : Point α :=
  (1/3) • (A + B + C)

theorem OS_perpendicular_BC (A B C O S : Point ℝ)
  (hAOB : collinear A O B)
  (hBOC : collinear B O C)
  (hACB : collinear A C B)
  (hSCircle : ∃ r, circle_center_radius O r S)
  (hAngleBisector : is_angle_bisector A S O) :
  are_perpendicular (line_through_points O S) (line_through_points B C) :=
begin
  sorry
end

end OS_perpendicular_BC_l162_162799


namespace total_exercise_hours_l162_162038

-- Defining the exercise durations in minutes
def natasha_minutes := 30 * 7
def esteban_minutes := 10 * 9
def charlotte_minutes := 20 + 45 + 70 + 40 + 60

-- Converting total minutes into hours
def natasha_hours := natasha_minutes / 60
def esteban_hours := esteban_minutes / 60
def charlotte_hours := charlotte_minutes / 60

-- Define the total hours and the statement to prove
def total_hours := natasha_hours + esteban_hours + charlotte_hours

theorem total_exercise_hours : total_hours = 8.92 := sorry

end total_exercise_hours_l162_162038


namespace exponentiation_identity_l162_162517

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162517


namespace nat_pow_eq_iff_divides_l162_162431

theorem nat_pow_eq_iff_divides (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : a = b^n :=
sorry

end nat_pow_eq_iff_divides_l162_162431


namespace fiona_finishes_first_l162_162207

open Real

-- Definitions based on the problem conditions
variables (d f : ℝ) (h_d_pos : 0 < d) (h_f_pos : 0 < f)

-- Condition: areas of the gardens
def diana_area := d
def fiona_area := d / 2
def elena_area := d / 3

-- Condition: mowing rates
def fiona_rate := f
def diana_rate := f / 2
def elena_rate := f / 3

-- Time calculations
def diana_time := diana_area / diana_rate
def fiona_time := fiona_area / fiona_rate
def elena_time := elena_area / elena_rate

-- Account for Diana's delayed start
def diana_total_time := diana_time + 1

-- The theorem to prove who finishes mowing first
theorem fiona_finishes_first
  (h_fiona_time : fiona_time = d / (2 * f))
  (h_elena_time : elena_time = d / f)
  (h_diana_time : diana_time = 2 * d / f)
  (h_diana_total_time : diana_total_time = 2 * d / f + 1) : 
  fiona_time < elena_time ∧ fiona_time < diana_total_time :=
by 
  sorry

end fiona_finishes_first_l162_162207


namespace Paula_needs_52_tickets_l162_162402

theorem Paula_needs_52_tickets :
  let g := 2
  let b := 4
  let r := 3
  let f := 1
  let t_g := 4
  let t_b := 5
  let t_r := 7
  let t_f := 3
  g * t_g + b * t_b + r * t_r + f * t_f = 52 := by
  intros
  sorry

end Paula_needs_52_tickets_l162_162402


namespace provider_choices_count_l162_162369

theorem provider_choices_count :
  let num_providers := 25
  let num_s_providers := 6
  let remaining_providers_after_laura := num_providers - 1
  let remaining_providers_after_brother := remaining_providers_after_laura - 1

  (num_providers * num_s_providers * remaining_providers_after_laura * remaining_providers_after_brother) = 75900 :=
by
  sorry

end provider_choices_count_l162_162369


namespace cryptarithm_solution_l162_162353

theorem cryptarithm_solution :
  ∃ (F R Y H A M : ℕ),
    (∀ d, d ∈ {F, R, Y, H, A, M} → d < 10) ∧ 
    (∀ p q: ℕ, (p ∈ {F, R, Y, H, A, M} ∧ q ∈ {F, R, Y, H, A, M} → p ≠ q)) ∧
    (let FRY := 100 * F + 10 * R + Y in 
     let HAM := 100 * H + 10 * A + M in 
     7 * (1000 * FRY + HAM) = 6 * (1000 * HAM + FRY)) ∧ 
    (let FRY := 100 * F + 10 * R + Y in FRY = 461) ∧
    (let HAM := 100 * H + 10 * A + M in HAM = 538) := 
sorry

end cryptarithm_solution_l162_162353


namespace solve_system_of_equations_l162_162541

theorem solve_system_of_equations : 
  ∃ (x y z u : ℝ), 
    x + y = 12 ∧ 
    x / z = 3 / 2 ∧ 
    z + u = 10 ∧ 
    y * u = 36 ∧ 
    x = 6 ∧ 
    y = 6 ∧ 
    z = 4 ∧ 
    u = 6 := 
by {
  use 6, 6, 4, 6,
  sorry
}

end solve_system_of_equations_l162_162541


namespace arccos_sqrt3_div_2_eq_pi_div_6_l162_162609

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l162_162609


namespace find_ab_sum_l162_162671

variables (a b : ℝ)
def quadratic_eq (x : ℂ) : ℂ := a * x^2 + b * x + 2
def is_root (x : ℂ) (f : ℂ → ℂ) : Prop := f x = 0

theorem find_ab_sum (h1 : is_root (1 + complex.i) (quadratic_eq a b)) : a + b = -1 :=
sorry

end find_ab_sum_l162_162671


namespace readers_both_scifi_and_literature_l162_162548

theorem readers_both_scifi_and_literature (total readers_scifi readers_literature : ℕ) 
  (h1 : total = 400) 
  (h2 : readers_scifi = 250) 
  (h3 : readers_literature = 230) : 
  ∃ readers_both, readers_both = readers_scifi + readers_literature - total ∧ readers_both = 80 :=
by {
  use (readers_scifi + readers_literature - total),
  split,
  { exact rfl },
  { rw [h1, h2, h3], norm_num }
}

end readers_both_scifi_and_literature_l162_162548


namespace inequality_proof_l162_162026

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4 :=
by {
  sorry
}

end inequality_proof_l162_162026


namespace election_winner_votes_l162_162551

theorem election_winner_votes (V : ℝ) (h1 : 0.56 * V - 0.44 * V = 288) : 0.56 * V = 1344 :=
by
  have solve_for_V : V = 2400 := by 
    field_simp at h1
    linarith
  rw solve_for_V
  norm_num

end election_winner_votes_l162_162551


namespace no_positive_int_squares_l162_162055

theorem no_positive_int_squares (n : ℕ) (h_pos : 0 < n) :
  ¬ (∃ a b c : ℕ, a ^ 2 = 2 * n ^ 2 + 1 ∧ b ^ 2 = 3 * n ^ 2 + 1 ∧ c ^ 2 = 6 * n ^ 2 + 1) := by
  sorry

end no_positive_int_squares_l162_162055


namespace new_ratio_milk_to_water_l162_162335

def total_volume : ℕ := 100
def initial_milk_ratio : ℚ := 3
def initial_water_ratio : ℚ := 2
def additional_water : ℕ := 48

def new_milk_volume := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume
def new_water_volume := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume + additional_water

theorem new_ratio_milk_to_water :
  new_milk_volume / (new_water_volume : ℚ) = 15 / 22 :=
by
  sorry

end new_ratio_milk_to_water_l162_162335


namespace roots_of_equation_l162_162643

theorem roots_of_equation :
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by
  sorry

end roots_of_equation_l162_162643


namespace measure_of_MNP_l162_162354

-- Define the conditions of the pentagon
variables {M N P Q S : Type} -- Define the vertices of the pentagon
variables {MN NP PQ QS SM : ℝ} -- Define the lengths of the sides
variables (MNP QNS : ℝ) -- Define the measures of the involved angles

-- State the conditions
-- Pentagon sides are equal
axiom equal_sides : MN = NP ∧ NP = PQ ∧ PQ = QS ∧ QS = SM ∧ SM = MN 
-- Angle relation
axiom angle_relation : MNP = 2 * QNS

-- The goal is to prove that measure of angle MNP is 60 degrees
theorem measure_of_MNP : MNP = 60 :=
by {
  sorry -- The proof goes here
}

end measure_of_MNP_l162_162354


namespace tulip_percentage_l162_162576

-- Define necessary fractions
def fraction_pink (total_flowers : ℝ) : ℝ := 3 / 4 * total_flowers
def fraction_red_non_pink (total_flowers : ℝ) : ℝ := 7 / 20 * total_flowers
def fraction_lilies (total_flowers : ℝ) : ℝ := 1 / 10 * total_flowers

-- Determine the fractions of tulips
def fraction_pink_tulips (total_flowers : ℝ) : ℝ := 1 / 2 * fraction_pink total_flowers
def fraction_red_tulips (total_flowers : ℝ) : ℝ := 2 / 3 * fraction_red_non_pink total_flowers

-- Calculate the total fraction of tulips
def fraction_tulips (total_flowers : ℝ) : ℝ :=
  fraction_pink_tulips total_flowers + fraction_red_tulips total_flowers

-- Convert the fraction of tulips to percentage
def percentage_tulips (total_flowers : ℝ) : ℝ :=
  fraction_tulips total_flowers / total_flowers * 100

-- The final theorem statement in Lean 4
theorem tulip_percentage (total_flowers : ℝ) : percentage_tulips total_flowers = 61 := by
  sorry

end tulip_percentage_l162_162576


namespace count_unique_elements_in_set_l162_162438

def f (x : ℝ) : ℝ := Real.floor x + Real.floor (2 * x) + Real.floor (3 * x)

theorem count_unique_elements_in_set : 
  (Finset.image f (Finset.Icc 1 100)).card = 67 := 
sorry

end count_unique_elements_in_set_l162_162438


namespace power_mean_inequality_l162_162798

theorem power_mean_inequality 
  (n : ℕ) 
  (x : Fin n → ℝ) 
  (m : ℝ) 
  (a : ℝ) 
  (s : ℝ)
  (hx_pos : ∀ i, 0 < x i)
  (hm_pos : 0 < m) 
  (ha_pos : 0 < a) 
  (hx_sum : (Finset.univ.sum (λ i, x i) = s)) 
  (hs_le_n : s ≤ n) :
  Finset.univ.sum (λ i, (x i)^m + 1 / (x i)^m + a)^n ≥ n * ((s / n)^m + (n / s)^m + a)^n := 
by 
  sorry

end power_mean_inequality_l162_162798


namespace yoongi_has_5_carrots_l162_162600

def yoongis_carrots (initial_carrots sister_gave: ℕ) : ℕ :=
  initial_carrots + sister_gave

theorem yoongi_has_5_carrots : yoongis_carrots 3 2 = 5 := by 
  sorry

end yoongi_has_5_carrots_l162_162600


namespace solve_system_l162_162065

noncomputable def system_of_eqs (a b c : ℝ) : Prop :=
  a^2 * b^2 - a^2 - a * b + 1 = 0 ∧
  a^2 * c - a * b - a - c = 0 ∧
  a * b * c = -1

theorem solve_system : ∃ a b c : ℝ, system_of_eqs a b c ∧ a = -1 ∧ b = -1 ∧ c = -1 :=
by {
  use [-1, -1, -1],
  simp [system_of_eqs],
  sorry
}

end solve_system_l162_162065


namespace number_of_cars_l162_162734

theorem number_of_cars 
  (num_bikes : ℕ) (num_wheels_total : ℕ) (wheels_per_bike : ℕ) (wheels_per_car : ℕ)
  (h1 : num_bikes = 10) (h2 : num_wheels_total = 76) (h3 : wheels_per_bike = 2) (h4 : wheels_per_car = 4) :
  ∃ (C : ℕ), C = 14 := 
by
  sorry

end number_of_cars_l162_162734


namespace pq_bad_number_divisibility_l162_162390

def is_coprime (p q : ℕ) : Prop :=
  Nat.gcd p q = 1

def is_pq_bad_number (p q n : ℕ) : Prop :=
  ¬ ∃ x y : ℕ, n = p * x + q * y

noncomputable def S (p q : ℕ) : ℕ :=
  let bad_numbers := { n | is_pq_bad_number p q n }
  bad_numbers.sum (λ n, n ^ 2019)

theorem pq_bad_number_divisibility (p q : ℕ) (hpq: is_coprime p q) (h1: 1 < p) (h2: 1 < q) :
  ∃ λ : ℕ, (p - 1) * (q - 1) ∣ λ * S p q :=
sorry

end pq_bad_number_divisibility_l162_162390


namespace power_of_powers_eval_powers_l162_162528

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162528


namespace line_and_circle_relationship_l162_162271

noncomputable def relationship_between_line_and_circle (a b θ : ℝ) (ha : a^2 * sin θ + a * cos θ - π / 4 = 0) (hb : b^2 * sin θ + b * cos θ - π / 4 = 0) : Prop :=
  ∀ d, (d = |((a * b * (a - b)) / (b^2 - a^2))| / sqrt (((b - a) / (b^2 - a^2))^2 + 1)) → (d < 1 ∨ d = 1 ∨ d > 1)

theorem line_and_circle_relationship (a b θ : ℝ) (ha : a^2 * sin θ + a * cos θ - π / 4 = 0) (hb : b^2 * sin θ + b * cos θ - π / 4 = 0) :
  ¬ ∀ d, (d = |((a * b * (a - b)) / (b^2 - a^2))| / sqrt (((b - a) / (b^2 - a^2))^2 + 1)) → (d < 1 ∨ d = 1 ∨ d > 1) := 
sorry

end line_and_circle_relationship_l162_162271


namespace chebyshev_inequality_example_l162_162422

noncomputable def X : Type := sorry -- The type X represents the random variable of the length of a part

theorem chebyshev_inequality_example
  (E_X : ℝ := 50) -- Mean of the length 
  (Var_X : ℝ := 0.1) -- Variance of the length
  : (probability (set.Ioo 49.5 50.5) X) ≥ 0.6 :=
by
  -- Definitions needed to represent the problem in a proof system
  let mu := E_X
  let sigma_sq := Var_X
  let ε := 0.5
  have h1 : (probability (set.Icc (-ε) ε) (λ x, x - mu) * indicator X) ≥ 0.6 := sorry
  exact h1

end chebyshev_inequality_example_l162_162422


namespace area_ratio_of_concentric_circles_l162_162109

-- Define the conditions
def concentric_circles {O : Type*} (C1 C2 : circle O) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = 60 / 360 * 2 * π ∧ θ₂ = 72 / 360 * 2 * π → arc_length C1 θ₁ = arc_length C2 θ₂

-- Define the circles and their properties
def smaller_circle {O : Type*} := circle O
def larger_circle {O : Type*} := circle O

-- Main theorem to prove
theorem area_ratio_of_concentric_circles 
  {O : Type*} (C1 : smaller_circle) (C2 : larger_circle)
  (h : concentric_circles C1 C2) :
  let r1 := radius C1 in
  let r2 := radius C2 in
  let A1 := area C1 in
  let A2 := area C2 in
  A1 / A2 = 36 / 25 :=
by
  sorry

end area_ratio_of_concentric_circles_l162_162109


namespace real_roots_iff_l162_162626

theorem real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 + 2 * k = 0) ↔ (-1 ≤ k ∧ k ≤ 0) :=
by sorry

end real_roots_iff_l162_162626


namespace range_of_t_l162_162350

-- Define the constants and conditions
variables (k : ℝ) (t : ℝ)
variables (h1 : k > 0) (h2 : t ≠ 0) (h3 : t ≠ -2)

-- Define the functions and points
def direct_proportion := λ x : ℝ, k * x
def inverse_proportion := λ x : ℝ, k / x

def A := (t, direct_proportion k t)
def B := (t + 2, direct_proportion k (t + 2))
def C := (t, inverse_proportion k t)
def D := (t + 2, inverse_proportion k (t + 2))

-- Define the differences
def p_minus_m := k * t - k / t
def q_minus_n := k * (t + 2) - k / (t + 2)

-- The proof statement
theorem range_of_t (h : ((p_minus_m k t) * (q_minus_n k t) < 0)) : 
    (-3 < t ∧ t < -2) ∨ (0 < t ∧ t < 1) :=
sorry

end range_of_t_l162_162350


namespace volume_of_cube_l162_162042

theorem volume_of_cube (a : ℕ) (h : (a^3 - a = a^3 - 5)) : a^3 = 125 :=
by {
  -- The necessary algebraic manipulation follows
  sorry
}

end volume_of_cube_l162_162042


namespace exponentiation_rule_example_l162_162501

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162501


namespace no_real_solution_l162_162217

theorem no_real_solution (x : ℝ) : 
  x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 → 
  ¬ (
    (1 / ((x - 1) * (x - 3))) + (1 / ((x - 3) * (x - 5))) + (1 / ((x - 5) * (x - 7))) = 1 / 4
  ) :=
by sorry

end no_real_solution_l162_162217


namespace solve_digit_problem_l162_162747

-- Definitions of characters as distinct digits and their sum
variables (中 环 杯 是 最 棒 的 : ℕ) (h1: 中 ≠ 环) (h2: 中 ≠ 杯) (h3: 中 ≠ 是) (h4: 中 ≠ 最) (h5: 中 ≠ 棒) (h6: 中 ≠ 的)
          (h7: 环 ≠ 杯) (h8: 环 ≠ 是) (h9: 环 ≠ 最) (h10: 环 ≠ 棒) (h11: 环 ≠ 的)
          (h12: 杯 ≠ 是) (h13: 杯 ≠ 最) (h14: 杯 ≠ 棒) (h15: 杯 ≠ 的)
          (h16: 是 ≠ 最) (h17: 是 ≠ 棒) (h18: 是 ≠ 的)
          (h19: 最 ≠ 棒) (h20: 最 ≠ 的)
          (h21: 棒 ≠ 的)

-- Statement of the problem
theorem solve_digit_problem (h_sum: 中 * 1000 + 环 * 100 + 杯 * 10 + 是 + 最 * 100 + 棒 * 10 + 的 = 2013) : 
  ∃ 中 环 杯 是 最 棒 的, 中 + 环 + 杯 + 是 + 最 + 棒 + 的 = (a_solution : nat) :=
begin
  sorry
end

end solve_digit_problem_l162_162747


namespace time_for_trains_to_cross_l162_162889

def length_train1 := 500 -- 500 meters
def length_train2 := 750 -- 750 meters
def speed_train1 := 60 * 1000 / 3600 -- 60 km/hr to m/s
def speed_train2 := 40 * 1000 / 3600 -- 40 km/hr to m/s
def relative_speed := speed_train1 + speed_train2 -- relative speed in m/s
def combined_length := length_train1 + length_train2 -- sum of lengths of both trains

theorem time_for_trains_to_cross :
  (combined_length / relative_speed) = 45 := 
by
  sorry

end time_for_trains_to_cross_l162_162889


namespace possible_values_of_N_l162_162852

theorem possible_values_of_N (N : ℤ) (h : N^2 - N = 12) : N = 4 ∨ N = -3 :=
sorry

end possible_values_of_N_l162_162852


namespace train_crossing_time_l162_162007

variable (L : ℕ) (v_kmhr : ℕ)

def time_to_cross_pole (L : ℕ) (v_kmhr : ℕ) : ℚ :=
  let v_ms := (v_kmhr * 1000) / 3600 in
  L / v_ms

theorem train_crossing_time : 
  (time_to_cross_pole 100 54 = 20 / 3) := 
by
  sorry

end train_crossing_time_l162_162007


namespace complex_quadrant_l162_162906

theorem complex_quadrant (m : ℝ) (h : m < 1) : 
  let z := complex.mk 1 (m - 1)
  in z.im < 0 :=
by
  let z := complex.mk 1 (m - 1)
  suffices z.im < 0 by sorry
  sorry

end complex_quadrant_l162_162906


namespace inverse_of_p_is_false_l162_162670

def is_angle_in_second_quadrant (θ : ℝ) : Prop :=
  π/2 < θ ∧ θ < π

def P (θ : ℝ) : Prop :=
  sin θ * (1 - 2 * cos (θ / 2) ^ 2) > 0

theorem inverse_of_p_is_false (θ : ℝ) (h : is_angle_in_second_quadrant θ) :
  ¬((¬ is_angle_in_second_quadrant θ) → P θ) :=
sorry

end inverse_of_p_is_false_l162_162670


namespace bus_speed_incl_stoppages_l162_162215

theorem bus_speed_incl_stoppages (v_excl : ℝ) (minutes_stopped : ℝ) :
  v_excl = 64 → minutes_stopped = 13.125 →
  v_excl - (v_excl * (minutes_stopped / 60)) = 50 :=
by
  intro v_excl_eq minutes_stopped_eq
  rw [v_excl_eq, minutes_stopped_eq]
  have hours_stopped : ℝ := 13.125 / 60
  have distance_lost : ℝ := 64 * hours_stopped
  have v_incl := 64 - distance_lost
  sorry

end bus_speed_incl_stoppages_l162_162215


namespace solution_set_of_inequality_l162_162097

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x + 5) / (x - 1) > x ↔ x < -1 ∨ (1 < x ∧ x < 5) :=
sorry

end solution_set_of_inequality_l162_162097


namespace piecewise_function_solution_l162_162266

def f (x : ℝ) : ℝ := x * Real.log x

def f_T (T : ℝ) (x : ℝ) : ℝ :=
  if f x ≥ T then f x else T

theorem piecewise_function_solution :
  f (f_T 2 Real.exp) = 3 :=
by
  sorry

end piecewise_function_solution_l162_162266


namespace solve_for_w_l162_162715

theorem solve_for_w (w : ℕ) (h : w^2 - 5 * w = 0) (hp : w > 0) : w = 5 :=
sorry

end solve_for_w_l162_162715


namespace range_foci_ratio_l162_162673

noncomputable def ellipse := {P : ℝ × ℝ // (P.1 ^ 2) / 16 + (P.2 ^ 2) / 12 = 1}
def foci_distance (P : ellipse) (F1 F2 : ℝ × ℝ) := |dist P.val F1 - dist P.val F2| / dist P.val F1
def foci_F1 := (-4, 0)
def foci_F2 := (4, 0)

theorem range_foci_ratio : 
  ∀ (P : ellipse),
  0 ≤ foci_distance P foci_F1 foci_F2 ∧ foci_distance P foci_F1 foci_F2 ≤ 2 := 
sorry

end range_foci_ratio_l162_162673


namespace find_ellipse_equation_find_line_equation_l162_162257

open Real

-- Definitions based on conditions
def semi_minor_axis := 1
def distance_to_line (c : ℝ) : ℝ := abs (c + 2 * sqrt 2) / sqrt 2
def line_distance := 3
def semi_major_axis (b : ℝ) (c : ℝ) : ℝ := sqrt (1 + 2)
def focus_function (c : ℝ) : Prop := c = sqrt 2
def equation_of_ellipse (a b : ℝ) : Prop := ∀ x y, (x ^ 2) / (a ^ 2) + y ^ 2 = 1

-- Proof problem: Part Ⅰ
theorem find_ellipse_equation : ∃ a b : ℝ, (b = semi_minor_axis) → 
  (∀ c, (distance_to_line c = line_distance) → (focus_function c) → (a = semi_major_axis b c)) ∧
  (equation_of_ellipse (semi_major_axis semi_minor_axis (sqrt 2)) semi_minor_axis) :=
sorry

-- Proof problem: Part Ⅱ
theorem find_line_equation : ∀ A B : (ℝ × ℝ), (A = (0,1)) →
  (∃ k : ℝ, ∀ x y : ℝ, (x, y) = B → y = k * x + 1) → (k = 1 ∨ k = -1) :=
sorry

end find_ellipse_equation_find_line_equation_l162_162257


namespace prove_healthy_diet_multiple_l162_162411

variable (rum_on_pancakes rum_earlier rum_after_pancakes : ℝ)
variable (healthy_multiple : ℝ)

-- Definitions from conditions
def Sally_gave_rum_on_pancakes : Prop := rum_on_pancakes = 10
def Don_had_rum_earlier : Prop := rum_earlier = 12
def Don_can_have_rum_after_pancakes : Prop := rum_after_pancakes = 8

-- Concluding multiple for healthy diet
def healthy_diet_multiple : Prop := healthy_multiple = (rum_on_pancakes + rum_after_pancakes - rum_earlier) / rum_on_pancakes

theorem prove_healthy_diet_multiple :
  Sally_gave_rum_on_pancakes rum_on_pancakes →
  Don_had_rum_earlier rum_earlier →
  Don_can_have_rum_after_pancakes rum_after_pancakes →
  healthy_diet_multiple rum_on_pancakes rum_earlier rum_after_pancakes healthy_multiple →
  healthy_multiple = 0.8 := 
by
  intros h1 h2 h3 h4
  sorry

end prove_healthy_diet_multiple_l162_162411


namespace count_good_numbers_l162_162582

def sum_of_set := Finset.sum (Finset.range 21)

def is_good (k : ℕ) : Prop :=
  k > 20 ∧ 210 % k = 0

def good_numbers := {k : ℕ | is_good k}

theorem count_good_numbers : Finset.card (good_numbers) = 7 :=
by
  sorry

end count_good_numbers_l162_162582


namespace unit_vector_orthogonal_l162_162988

def vector1 : ℝ^3 := ⟨2, 1, 1⟩
def vector2 : ℝ^3 := ⟨3, 0, 1⟩
def cross_product := ⟨1, 1, -3⟩
def magnitude := real.sqrt (1^2 + 1^2 + (-3)^2)
def unit_vector := ⟨1 / magnitude, 1 / magnitude, -3 / magnitude⟩

theorem unit_vector_orthogonal :
  (vector1 ⬝ unit_vector = 0) ∧
  (vector2 ⬝ unit_vector = 0) ∧
  (real.sqrt (unit_vector.1^2 + unit_vector.2^2 + unit_vector.3^2) = 1) :=
sorry

end unit_vector_orthogonal_l162_162988


namespace midsegment_equals_height_l162_162850

variables {A B C D M N K L : Type*} [IsoscelesTrapezoid A B C D] [PerpendicularDiagonal A C D B]

-- Assume A, B, C, D, are points in Euclidean plane forming an isosceles trapezoid
-- with perpendicular diagonals AC and BD.
axiom isosceles_trapezoid (A B C D : Type*): is_isosceles_trapezoid A B C D
axiom diagonals_perpendicular (A B C D : Type*): are_perpendicular (diagonal A C) (diagonal B D)

-- Assume M and N are midpoints of bases BC and AD respectively.
axiom midpoint_BC (B C M : Type*): is_midpoint B C M
axiom midpoint_AD (A D N : Type*): is_midpoint A D N

-- Prove that the midsegment MN is equal to the height of the trapezoid.
theorem midsegment_equals_height (A B C D M N : Type*) [isosceles_trapezoid A B C D] [diagonals_perpendicular A B C D]
  [midpoint_BC B C M] [midpoint_AD A D N] : 
  length(midsegment M N) = height A B C D :=
by sorry

end midsegment_equals_height_l162_162850


namespace exponentiation_rule_example_l162_162506

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162506


namespace function_even_l162_162079

noncomputable def f (x : ℝ) : ℝ := x ^ -2

theorem function_even : (∀ x : ℝ, x ≠ 0 → f (-x) = f x) := 
by 
  intros x hx
  unfold f
  simp

end function_even_l162_162079


namespace probability_of_winning_reward_l162_162593

-- Definitions representing the problem conditions
def red_envelopes : ℕ := 4
def card_types : ℕ := 3

-- Theorem statement: Prove the probability of winning the reward is 4/9
theorem probability_of_winning_reward : 
  (∃ (n m : ℕ), n = card_types^red_envelopes ∧ m = (Nat.choose red_envelopes 2) * (Nat.factorial 3)) → 
  (m / n = 4/9) :=
by
  sorry  -- Proof to be filled in

end probability_of_winning_reward_l162_162593


namespace sum_of_midpoint_coordinates_l162_162121

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 10) (h2 : y1 = 7) (h3 : x2 = 4) (h4 : y2 = -3) :
  let M := (x1 + x2) / 2, (y1 + y2) / 2 in
  (M.1 + M.2) = 9 :=
by 
  have hM : M = ((10 + 4) / 2, (7 + -3) / 2), from sorry,
  have hM' : M = (7, 2), from sorry,
  have sum_midpoint := 7 + 2,
  show sum_midpoint = 9, 
  from sorry

end sum_of_midpoint_coordinates_l162_162121


namespace hyperbola_eccentricity_l162_162299

theorem hyperbola_eccentricity 
    (a b : ℝ) (ha : a > 0) (hb : b > 0) :
    (∃ (k : ℝ), k = b / a ∧ (forall (p : ℝ × ℝ), p = (2, -3) → p.2 = k * p.1)) →
    let e := Real.sqrt (1 + (b / a) ^ 2) in
    e = Real.sqrt (13) / 2 :=
by
  intros h
  let k := b / a
  let e := Real.sqrt (1 + k^2)
  have h1 : k = 3 / 2 := sorry -- Asymptote calculation must match the point (2, -3)
  rw h1
  have h2 : e = Real.sqrt (1 + (3 / 2)^2) := rfl
  rw h2
  norm_num
  exact div_eq_div_left_of_eq Real.zero_lt_two (by norm_num)

end hyperbola_eccentricity_l162_162299


namespace minimum_value_l162_162269

noncomputable def min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :=
  a + 2 * b

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :
  min_value a b h₁ h₂ h₃ ≥ 2 * Real.sqrt 2 :=
sorry

end minimum_value_l162_162269


namespace cost_price_computer_table_l162_162090

theorem cost_price_computer_table (C : ℝ) (S : ℝ) (H1 : S = C + 0.60 * C) (H2 : S = 2000) : C = 1250 :=
by
  -- Proof goes here
  sorry

end cost_price_computer_table_l162_162090


namespace power_calc_l162_162459

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162459


namespace part1_part2_l162_162255

-- Define the sequence a_n
def a : ℕ → ℝ
| 0       := 1
| (n + 1) := 3 * a n / (2 * a n + 3)

-- Prove that the sequence {1 / a_n} is arithmetic with first term 1 and common difference 2/3
theorem part1: ∀ n : ℕ, 1 / a (n + 1) = 1 / a n + 2 / 3 :=
by
  sorry

-- Define T_{2n}
def T_2n : ℕ → ℝ
| 0       := 0
| n       := ∑ k in finset.range (n + 1), (1 / (a (2 * k) * a (2 * k + 1)) - 1 / (a (2 * k + 1) * a (2 * k + 2)))

-- Prove that T_{2n} = -4/9 (2n^2 + 3n)
theorem part2: ∀ n : ℕ, T_2n n = -(4 / 9) * (2 * n^2 + 3 * n) :=
by
  sorry

end part1_part2_l162_162255


namespace angle_BAO_l162_162745

noncomputable def problem_statement
  (CD : ℝ) (O A E F B : ℝ) (OD : ℝ) (angle_EOF : ℝ) : Prop :=
  (CD ≠ 0) ∧
  (angle_EOF = 30) ∧
  (2 * OD = AB) →
  (∃ (BAO : ℝ), BAO = 37.5)

theorem angle_BAO {CD : ℝ} {O A E F B : ℝ} {OD angle_EOF : ℝ} :
  problem_statement CD O A E F B OD angle_EOF → ∃ (BAO : ℝ), BAO = 37.5 :=
by 
  sorry

end angle_BAO_l162_162745


namespace exponentiation_rule_example_l162_162500

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162500


namespace power_calc_l162_162461

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162461


namespace parallelogram_area_perpendicular_vector_l162_162307

-- Define the points in space
def A : ℝ × ℝ × ℝ := (0, 2, 3)
def B : ℝ × ℝ × ℝ := (-2, 1, 6)
def C : ℝ × ℝ × ℝ := (1, -1, 5)

-- Define vectors AB and AC
def AB : ℝ × ℝ × ℝ := (-2, -1, 3)
def AC : ℝ × ℝ × ℝ := (1, -3, 2)

-- Define the first proof problem: The area of the parallelogram with sides AB and AC
theorem parallelogram_area :
  let area := 7 * Real.sqrt 3 in
  True := (by sorry)

-- Define a vector a that is perpendicular to both AB and AC and has a norm of 2 * sqrt 3
theorem perpendicular_vector :
  let a1 : ℝ × ℝ × ℝ := (2, 2, 2)
  let a2 : ℝ × ℝ × ℝ := (-2, -2, -2)
  ∥a1∥ = 2 * Real.sqrt 3 ∧
  ∥a2∥ = 2 * Real.sqrt 3 ∧
  (a1.1 * AB.1 + a1.2 * AB.2 + a1.3 * AB.3 = 0) ∧
  (a1.1 * AC.1 + a1.2 * AC.2 + a1.3 * AC.3 = 0) ∧
  (a2.1 * AB.1 + a2.2 * AB.2 + a2.3 * AB.3 = 0) ∧
  (a2.1 * AC.1 + a2.2 * AC.2 + a2.3 * AC.3 = 0) := (by sorry)

end parallelogram_area_perpendicular_vector_l162_162307


namespace exponentiation_identity_l162_162513

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162513


namespace find_D_coordinates_l162_162819

-- Definitions of the points P and Q
def P := (2, -2) : ℝ × ℝ
def Q := (6, 4) : ℝ × ℝ

-- Definition stating D is twice as far from P as it is from Q
def is_twice_as_far (P Q D : ℝ × ℝ) : Prop :=
  dist P D = 2 * dist D Q

-- The theorem stating the coordinates of D
theorem find_D_coordinates (D : ℝ × ℝ) (h : is_twice_as_far P Q D) : D = (3, -0.5) :=
by sorry

end find_D_coordinates_l162_162819


namespace limit_np_n_l162_162666

noncomputable def p_n (n : ℕ) := 1 - ( (n * (n-1) * (n-2) * (n-3) * (n-4) * (n-5))^2 / (fact 6 * (n^2 * (n^2-1) * (n^2-2) * (n^2-3) * (n^2-4) * (n^2-5)))) * fact 6

theorem limit_np_n : tendsto (λ n : ℕ, n * p_n n) at_top (𝓝 30) :=
by sorry

end limit_np_n_l162_162666


namespace jill_total_earnings_over_three_months_l162_162766

theorem jill_total_earnings_over_three_months :
  (let first_month_earnings := 30 * 10 in
   let second_month_earnings := 2 * first_month_earnings in
   let third_month_earnings := second_month_earnings / 2 in
   first_month_earnings + second_month_earnings + third_month_earnings = 1200) :=
by
  sorry

end jill_total_earnings_over_three_months_l162_162766


namespace factor_expression_l162_162631

theorem factor_expression (x y z : ℝ) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 ≠ 0 →
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
    (x + y) * (y + z) * (z + x) :=
by
  intro h
  sorry

end factor_expression_l162_162631


namespace intersect_exists_l162_162096

variable {α : Type*} [DecidableEq α]

theorem intersect_exists (n : ℕ) (A B : Finset ℕ) :
  A.card = n → B.card = n → 
  (∑ x in A, x) = n^2 → (∑ x in B, x) = n^2 → 
  ∃ x, x ∈ A ∧ x ∈ B := 
by
  intros h1 h2 h3 h4
  sorry

end intersect_exists_l162_162096


namespace sum_products_nonzero_l162_162112

theorem sum_products_nonzero:
  (Table : Fin 25 → Fin 25 → ℤ)
  (h1 : ∀ i j, Table i j = 1 ∨ Table i j = -1)
  (S : ℤ := ∑ i, ∏ j, Table i j + ∑ j, ∏ i, Table i j) :
  S ≠ 0 :=
by
  sorry

end sum_products_nonzero_l162_162112


namespace arithmetic_sequence_30th_term_l162_162420

-- Definitions
def a₁ : ℤ := 8
def d : ℤ := -3
def n : ℕ := 30

-- The statement to be proved
theorem arithmetic_sequence_30th_term :
  a₁ + (n - 1) * d = -79 :=
by
  sorry

end arithmetic_sequence_30th_term_l162_162420


namespace exp_eval_l162_162483

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162483


namespace simplify_and_evaluate_l162_162835

theorem simplify_and_evaluate :
  ∀ m : ℚ, m = 2 →
  ( (2 * m - 6) / (m ^ 2 - 9) / ( (2 * m + 2) / (m + 3) ) - m / (m + 1) ) = -1 / 3 :=
by
  intro m hm
  rw [hm]
  norm_num
  sorry

end simplify_and_evaluate_l162_162835


namespace student_club_joinings_l162_162177

def Student : Type := ℕ
def Club : Type := ℕ

noncomputable def ways_to_join_clubs (students : List Student) (clubs : List Club) : ℕ := sorry

theorem student_club_joinings : 
  ∀ (students : List Student) (clubs : List Club),
  students.length = 5 → 
  clubs.length = 4 →
  ∀ A ∈ students, 
  ¬ (∀ c ∈ clubs, c ≠ 42) → -- assuming "Anime Club" is indexed at 42
  (∀ student ∈ students, ∃! club ∈ clubs, student ∈ allocation[club]) → 
  (∀ club ∈ clubs, ∃ student ∈ students, student ∈ allocation[club]) →
  ways_to_join_clubs students clubs = 180 :=
by intros
   sorry

end student_club_joinings_l162_162177


namespace find_x_l162_162784

def oslash (a b : ℝ) : ℝ :=
  (real.sqrt (2 * a + b)) ^ 3

theorem find_x (x : ℝ) (h : oslash 4 x = 27) : x = 1 :=
by sorry

end find_x_l162_162784


namespace incoming_freshman_class_count_l162_162843

theorem incoming_freshman_class_count :
  ∃ n : ℕ, 
    n < 500 ∧ 
    n % 19 = 17 ∧ 
    n % 18 = 9 ∧ 
    n = 207 :=
begin
  sorry,
end

end incoming_freshman_class_count_l162_162843


namespace CM_perpendicular_AO2_l162_162157

section GeometryProblem

-- Assumptions for the problem
variables (O1 O2 A B M C : Type)
variables (r1 r2 : ℝ)
variables (circle1 : Circle O1 r1)
variables (circle2 : Circle O2 r2)

-- Tangent at A and B, M is the intersection point
variables (is_tangent1 : is_tangent circle1 A)
variables (is_tangent2 : is_tangent circle2 B)
variables (intersects_tangent : intersects_tangent_at M A B)

-- AC is the diameter of the first circle
variables (diameter_AC : is_diameter circle1 A C)

-- The theorem to be proven
theorem CM_perpendicular_AO2 : perpendicular (line_through C M) (line_through A O2) := 
sorry

end GeometryProblem

end CM_perpendicular_AO2_l162_162157


namespace power_of_powers_l162_162491

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162491


namespace segments_contain_one_of_numbers_l162_162812

theorem segments_contain_one_of_numbers (p q : ℕ) (hpq : Nat.coprime p q) :
  ∀ i : ℕ, 1 ≤ i ∧ i < p + q - 1 →
  ∃ k : ℕ, (k/p ≤ i/(p+q) ∧ i/(p+q) < (k+1)/p) ∨ 
           (k/q ≤ i/(p+q) ∧ i/(p+q) < (k+1)/q) :=
sorry

end segments_contain_one_of_numbers_l162_162812


namespace perp_lines_l162_162704

noncomputable def line_1 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (5 - k) * y + 1
noncomputable def line_2 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => 2 * (k - 3) * x - 2 * y + 3

theorem perp_lines (k : ℝ) : 
  let l1 := line_1 k
  let l2 := line_2 k
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k = 1 ∨ k = 4)) :=
by
    sorry

end perp_lines_l162_162704


namespace parallelogram_angles_l162_162435

theorem parallelogram_angles (k : ℝ) (h_k : 0 < k) :
  ∃ (A B : ℝ), 
    A = 3 * real.arccos ((2 + k) / (2 * k)) ∧
    B = real.pi - 3 * real.arccos ((2 + k) / (2 * k)) :=
begin
  -- sorry means that the proof is omitted.
  sorry
end

end parallelogram_angles_l162_162435


namespace second_layer_ratio_l162_162960

theorem second_layer_ratio
  (first_layer_sugar third_layer_sugar : ℕ)
  (third_layer_factor : ℕ)
  (h1 : first_layer_sugar = 2)
  (h2 : third_layer_sugar = 12)
  (h3 : third_layer_factor = 3) :
  third_layer_sugar = third_layer_factor * (2 * first_layer_sugar) →
  second_layer_factor = 2 :=
by
  sorry

end second_layer_ratio_l162_162960


namespace proofA_proofB_proofC_proofD_l162_162758

namespace TriangleProof

variables {A B C : ℝ} {a b c: ℝ}

-- Define angle and side relationships and given conditions
def angleSideRelations := ∀ (A B C : ℝ) (a b c : ℝ), 
                         ∃ (π : ℝ), A + B + C = π ∧ 
                         a / (Real.sin A) = b / (Real.sin B) ∧ 
                         a / (Real.sin A) = c / (Real.sin C)

def sinRelation := ∀ (A B C : ℝ), Real.sin B + Real.sin C = 2 * Real.sin A

-- Conditions for Question A
def questionAConditions := A = Real.pi / 3 ∧ c = 1

-- Proof statement for Question A
theorem proofA (h1 : angleSideRelations) (h2 : sinRelation) (h3 : questionAConditions) : a = 1 :=
by sorry

-- Conditions for Question B
def questionBConditions := A = Real.pi / 3 ∧ c = 1

-- Expected area calculation for Question B
def expectedArea := (Real.sqrt 3) / 4

-- Proof statement for Question B
theorem proofB (h1 : angleSideRelations) (h2 : sinRelation) (h3 : questionBConditions) : 
  ∀ (area : ℝ), area ≠ π  → area = expectedArea :=
by sorry

-- Conditions for Question C
def questionCConditions := b = 2

-- Proof statement for Question C
theorem proofC (h1 : angleSideRelations) (h2 : sinRelation) (h3 : questionCConditions) : A ≤ Real.pi / 3 :=
by sorry

-- Conditions for Question D
def questionDConditions := b = 2

-- Range for perimeter calculation for Question D
def rangePerimeter := ∀ (perimeter : ℝ), 4 < perimeter ∧ perimeter < 12

-- Proof statement for Question D
theorem proofD (h1 : angleSideRelations) (h2 : sinRelation) (h3 : questionDConditions) : rangePerimeter (a + b + c) :=
by sorry

end TriangleProof

end proofA_proofB_proofC_proofD_l162_162758


namespace john_safe_weight_l162_162771

-- Assuming the conditions provided that form the basis of our problem.
def max_capacity : ℝ := 1000
def safety_margin : ℝ := 0.20
def john_weight : ℝ := 250
def safe_weight (max_capacity safety_margin john_weight : ℝ) : ℝ := 
  (max_capacity * (1 - safety_margin)) - john_weight

-- The main theorem to prove based on the provided problem statement.
theorem john_safe_weight : safe_weight max_capacity safety_margin john_weight = 550 := by
  -- skipping the proof details as instructed
  sorry

end john_safe_weight_l162_162771


namespace sum_of_reciprocals_of_squares_l162_162549

theorem sum_of_reciprocals_of_squares (x y : ℕ) (hxy : x * y = 17) : 
  1 / (x:ℚ)^2 + 1 / (y:ℚ)^2 = 290 / 289 := 
by
  sorry

end sum_of_reciprocals_of_squares_l162_162549


namespace find_y_orthogonal_l162_162635

theorem find_y_orthogonal :
  (∃ y : ℝ, (2 : ℝ) * (-3 : ℝ) + (-4 : ℝ) * y + (5 : ℝ) * (2 : ℝ) = 0) →
  (y = (1 : ℝ)) :=
begin
  sorry -- proof will be here
end

end find_y_orthogonal_l162_162635


namespace decreasing_sequence_range_of_lambda_l162_162429

theorem decreasing_sequence_range_of_lambda (a_n : ℕ → ℝ) (λ : ℝ) :
  (∀ n : ℕ, 0 < n → a_n n = -2 * (n:ℝ)^2 + λ * n) →
  (∀ n : ℕ, 0 < n → a_n (n + 1) - a_n n < 0) ↔ λ < 6 :=
by
  sorry

end decreasing_sequence_range_of_lambda_l162_162429


namespace length_of_longer_leg_of_smallest_triangle_l162_162209

noncomputable def hypotenuse_of_largest_triangle := 16
noncomputable def largest_triangle_shorter_leg := hypotenuse_of_largest_triangle / 2
noncomputable def largest_triangle_longer_leg := largest_triangle_shorter_leg * Real.sqrt 3

noncomputable def second_triangle_hypotenuse := largest_triangle_longer_leg
noncomputable def second_triangle_shorter_leg := second_triangle_hypotenuse / 2
noncomputable def second_triangle_longer_leg := second_triangle_shorter_leg * Real.sqrt 3

noncomputable def smallest_triangle_hypotenuse := second_triangle_longer_leg
noncomputable def smallest_triangle_shorter_leg := smallest_triangle_hypotenuse / 2
noncomputable def smallest_triangle_longer_leg := smallest_triangle_shorter_leg * Real.sqrt 3

theorem length_of_longer_leg_of_smallest_triangle :
  smallest_triangle_longer_leg = 6 * Real.sqrt 3 := 
sorry

end length_of_longer_leg_of_smallest_triangle_l162_162209


namespace minimum_value_of_alpha_l162_162275

def y (x : ℝ) : ℝ := (3 - exp x) / (exp x + 1)

def slope_of_tangent_line (x : ℝ) : ℝ := 
  (-4) / (exp x + exp (-x) + 2)

def tan_alpha_at_point_P (x : ℝ) : ℝ :=
  arctan (slope_of_tangent_line x)

theorem minimum_value_of_alpha :
  ∃ x : ℝ, tan_alpha_at_point_P x = 3 * real.pi / 4 := sorry

end minimum_value_of_alpha_l162_162275


namespace ellipse_flatter_m_range_l162_162969

theorem ellipse_flatter_m_range :
  (∀ {x y : ℝ}, (x^2) / 4 + (y^2) / 3 = 1) ∧
  (∀ {x y : ℝ}, (x^2) / 9 + (y^2) / m = 1) ∧
  (∀ {x y : ℝ}, ((x^2 / 4 + y^2 / 3 = 1) -> (x^2 / 9 + y^2 / m = 1 -> y^2 / 3 < y^2 / m)))
  → m ∈ Ioo (27 / 4) 9 :=
by sorry

end ellipse_flatter_m_range_l162_162969


namespace rational_if_x7_and_x12_are_rational_not_necessarily_rational_if_x9_and_x12_are_rational_l162_162391

-- First case: If \( x \in \mathbb{R} \) and both \( x^7 \) and \( x^{12} \) are rational, then \( x \) is rational.
theorem rational_if_x7_and_x12_are_rational (x : ℝ) (hx7 : x^7 ∈ ℚ) (hx12 : x^12 ∈ ℚ) : x ∈ ℚ := sorry

-- Second case: If \( x \in \mathbb{R} \) and both \( x^9 \) and \( x^{12} \) are rational, then \( x \) is not necessarily rational.
theorem not_necessarily_rational_if_x9_and_x12_are_rational (x : ℝ) (hx9 : x^9 ∈ ℚ) (hx12 : x^12 ∈ ℚ) : ¬ (x ∈ ℚ) ∨ (x ∈ ℚ) := sorry

end rational_if_x7_and_x12_are_rational_not_necessarily_rational_if_x9_and_x12_are_rational_l162_162391


namespace solve_for_z_l162_162653

theorem solve_for_z (i z : ℂ) (h0 : i^2 = -1) (h1 : i / z = 1 + i) : z = (1 + i) / 2 :=
by
  sorry

end solve_for_z_l162_162653


namespace power_calc_l162_162463

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162463


namespace max_quotient_is_100_l162_162956

noncomputable def max_quotient : ℕ :=
  let max_value := 100
  ∃ (A B C : ℕ), (1 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ (0 ≤ C ∧ C ≤ 9) ∧ 
  max_quotient = (100 * A + 10 * B + C) / (A + B + C)

theorem max_quotient_is_100 : max_quotient = 100 :=
  sorry

end max_quotient_is_100_l162_162956


namespace mike_picked_l162_162359

-- Define the number of pears picked by Jason, Keith, and the total number of pears picked.
def jason_picked : ℕ := 46
def keith_picked : ℕ := 47
def total_picked : ℕ := 105

-- Define the goal that we need to prove: the number of pears Mike picked.
theorem mike_picked (jason_picked keith_picked total_picked : ℕ) 
  (h1 : jason_picked = 46) 
  (h2 : keith_picked = 47) 
  (h3 : total_picked = 105) 
  : (total_picked - (jason_picked + keith_picked)) = 12 :=
by sorry

end mike_picked_l162_162359


namespace min_value_of_expr_l162_162033

noncomputable def min_value_expr (x y : ℝ) := 
  (x + 1/y + 1) * (x + 1/y - 2022) + (y + 1/x + 1) * (y + 1/x - 2022)

theorem min_value_of_expr : ∀ (x y : ℝ), (0 < x) → (0 < y) → min_value_expr x y = -2048042 := 
by {
  intro x y hx hy,
  sorry
}

end min_value_of_expr_l162_162033


namespace weekly_milk_production_l162_162932

-- Conditions
def number_of_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 1000
def days_in_week : ℕ := 7

-- Statement to prove
theorem weekly_milk_production : (number_of_cows * milk_per_cow_per_day * days_in_week) = 364000 := by
  sorry

end weekly_milk_production_l162_162932


namespace good_carrots_l162_162984

theorem good_carrots (Faye_picked : ℕ) (Mom_picked : ℕ) (bad_carrots : ℕ)
    (total_carrots : Faye_picked + Mom_picked = 28)
    (bad_carrots_count : bad_carrots = 16) : 
    28 - bad_carrots = 12 := by
  -- Proof goes here
  sorry

end good_carrots_l162_162984


namespace classroom_lamps_total_ways_l162_162730

theorem classroom_lamps_total_ways (n : ℕ) (h : n = 4) : (2^n - 1) = 15 :=
by
  sorry

end classroom_lamps_total_ways_l162_162730


namespace compute_S_l162_162619

-- Define the sum series S
def S : ℝ := ∑ i in finset.range 100, (3 + (i + 1) * 9) / 3^(100 - i)

-- State the equivalent proof problem
theorem compute_S : S = 450 + 1 / (2 * 3^99) :=
by
  sorry

end compute_S_l162_162619


namespace acute_angle_probability_l162_162019

noncomputable def isAcute : ℝ := 5 / 12

theorem acute_angle_probability :
  ( ∀ (m n : ℕ) (hm : 1 ≤ m ∧ m ≤ 6) (hn : 1 ≤ n ∧ n ≤ 6),
    if m - n > 0 then true else false ) →
  ( 
    (∃ favorable_outcomes : ℕ, favorable_outcomes = 15) → 
    (∃ total_outcomes : ℕ, total_outcomes = 36) → 
    (isAcute = (15 / 36 : ℝ))
  ) := 
begin 
  intros h,
  use 15,
  use 36,
  exact (show (5/12 : ℝ) = (15 / 36 : ℝ), by norm_num)
end

end acute_angle_probability_l162_162019


namespace I_0_value_I_n_minus_I_n_minus_1_I_5_value_l162_162233

noncomputable def I (n : ℕ) : ℝ := ∫ x in (Real.pi / 4)..(Real.pi / 2), (Real.cos ((2 * n + 1) * x)) / (Real.sin x)

theorem I_0_value : I 0 = (1 / 2) * Real.log 2 :=
sorry

theorem I_n_minus_I_n_minus_1 (n : ℕ) (hn : n > 0) : 
  I n - I (n - 1) = 
  if n % 4 = 0 then 0 
  else if n % 4 = 1 then -1 / n 
  else if n % 4 = 2 then 2 / n 
  else -1 / n :=
sorry

theorem I_5_value : I 5 = (1 / 2) * Real.log 2 - 8 / 15 :=
sorry

end I_0_value_I_n_minus_I_n_minus_1_I_5_value_l162_162233


namespace mobius_as_composition_l162_162052

variable (a b c d R z w : ℂ)
variable (δ : ℂ)

def delta : ℂ := a * d - b * c

theorem mobius_as_composition 
  (hδ : δ ≠ 0) 
  (hz : z ≠ 0) 
  (ha : z ≠ -d/c) :
  w = (a * z + b) / (c * z + d) → 
  ∃ t : ℂ, ∃ α β : ℂ, \w = α + β / z :=
sorry

end mobius_as_composition_l162_162052


namespace unit_vector_orthogonal_to_a_and_b_l162_162991

noncomputable def a : ℝ × ℝ × ℝ := (2, 1, 1)
noncomputable def b : ℝ × ℝ × ℝ := (3, 0, 1)
noncomputable def u : ℝ × ℝ × ℝ := (1 / Real.sqrt 11, 1 / Real.sqrt 11, -3 / Real.sqrt 11)

theorem unit_vector_orthogonal_to_a_and_b : 
  ∃ (u : ℝ × ℝ × ℝ), 
    (u.1 = 1 / Real.sqrt 11 ∧ u.2 = 1 / Real.sqrt 11 ∧ u.3 = -3 / Real.sqrt 11) ∧
    a.1 * u.1 + a.2 * u.2 + a.3 * u.3 = 0 ∧
    b.1 * u.1 + b.2 * u.2 + b.3 * u.3 = 0 :=
sorry

end unit_vector_orthogonal_to_a_and_b_l162_162991


namespace type_C_count_l162_162447

theorem type_C_count (A B C C1 C2 : ℕ) (h1 : A + B + C = 25) (h2 : A + B + C2 = 17) (h3 : B + C2 = 12) (h4 : C2 = 8) (h5: B = 4) (h6: A = 5) : C = 16 :=
by {
  -- Directly use the given hypotheses.
  sorry
}

end type_C_count_l162_162447


namespace area_of_triangle_l162_162333

theorem area_of_triangle (A : ℝ) (b : ℝ) (a : ℝ) (hA : A = 60) (hb : b = 4) (ha : a = 2 * Real.sqrt 3) : 
  1 / 2 * a * b * Real.sin (60 * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l162_162333


namespace find_normal_vector_to_plane_l162_162664

structure Point3D := (x y z : ℝ)
structure Vector3D := (x y z : ℝ)

noncomputable def magnitude (v : Vector3D) : ℝ :=
  real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def dot_product (v₁ v₂ : Vector3D) : ℝ :=
  v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z

def vector_from_points (P Q : Point3D) : Vector3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def perpendicular_to_plane (P Q R : Point3D) (n : Vector3D) : Prop :=
  let AB := vector_from_points P Q
  let AC := vector_from_points P R
  dot_product n AB = 0 ∧ dot_product n AC = 0

def normal_vector (P Q R : Point3D) : Vector3D :=
  let AB := vector_from_points P Q
  let AC := vector_from_points P R
  ⟨AB.y * AC.z - AB.z * AC.y,
   AB.z * AC.x - AB.x * AC.z,
   AB.x * AC.y - AB.y * AC.x⟩

theorem find_normal_vector_to_plane (A B C : Point3D) (n : Vector3D)
  (h₁ : A = ⟨1, -2, -1⟩) (h₂ : B = ⟨0, -3, 1⟩) (h₃ : C = ⟨2, -2, 1⟩)
  (h₄ : magnitude n = real.sqrt 21) :
  perpendicular_to_plane A B C n ∧ (n = ⟨-2, 4, 1⟩ ∨ n = ⟨2, -4, -1⟩) :=
sorry

end find_normal_vector_to_plane_l162_162664


namespace determine_alloy_cubes_l162_162248

-- Definitions based on the conditions formed from part a)
def metal_cubes := {x : ℕ // x = 20}
def aluminum_cubes (n : ℕ) := {x : ℕ // x > 0 ∧ x < n}
def alloy_cubes (n : ℕ) := {x : ℕ // x = n - (aluminum_cubes n).val}

-- Statement in Lean 4
theorem determine_alloy_cubes :
  ∀ n : ℕ, ∃ k ≤ 11, ∀ c : metal_cubes, ∃ a : alloy_cubes n, 
  count_alloy_cubes c = some a := sorry

end determine_alloy_cubes_l162_162248


namespace Kolya_always_wins_l162_162403

-- Condition definitions
def polynomial (a b : ℤ) : ℤ → ℤ := λ x : ℤ, x^2 + a * x + b

def has_integer_roots (a b : ℤ) : Prop :=
  ∃ x y : ℤ, x * y = b ∧ x + y = -a

-- Main theorem
theorem Kolya_always_wins (a b : ℤ) :
  ∃ k : ℕ → (ℤ × ℤ), 
    (k 0 = (a, b)) ∧ 
    (∀ n, let (an, bn) := k n in 
      let (an1, bn1) := k (n + 1) in 
        (an1 = an + 1 ∨ an1 = an - 1 ∨ an1 = an + 3 ∨ an1 = an - 3) ∨ 
        (bn1 = bn + 1 ∨ bn1 = bn - 1 ∨ bn1 = bn + 3 ∨ bn1 = bn - 3)) ∧
    ∃ n, let (an, bn) := k n in has_integer_roots an bn :=
sorry

end Kolya_always_wins_l162_162403


namespace G_51_value_l162_162713

noncomputable def G : ℕ → ℚ
| 1 := 3
| (n+1) := (3 * G n + 2) / 3

theorem G_51_value : G 51 = 36 + 1 / 3 :=
by {
  sorry
}

end G_51_value_l162_162713


namespace right_triangle_A_l162_162783

noncomputable def isRightTriangleAtA' (a b c k : ℝ) : Prop :=
  let i := complex.I;
  let a' := complex.of_real a + k * i * (complex.of_real c - complex.of_real b);
  let b' := complex.of_real b + k * i * (complex.of_real a - complex.of_real c);
  let c' := complex.of_real c + k * i * (complex.of_real b - complex.of_real a);
  (b' - a') * (complex.conj (c' - a')) = 0

theorem right_triangle_A'_B'_C' (A B C k : ℝ) (hABC : k > 0) (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A) :
  isRightTriangleAtA' A B C k := 
sorry

end right_triangle_A_l162_162783


namespace asymptotes_of_hyperbola_l162_162077

theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, (y^2 / 4 - x^2 / 9 = 1) → (y = (2 / 3) * x ∨ y = -(2 / 3) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l162_162077


namespace coefficient_x8_l162_162748

-- Let’s define the polynomial p1 and p2
def p1 := (x : ℕ) - 1
def p2 := (x : ℕ) + 2
def polynomial := (x : ℕ) - 1 * (x+2)^(8 : ℕ)

-- Create a proof obligation that the coefficient of x^8 in the expansion is 15
theorem coefficient_x8 (x : ℕ) : 
  coeff (expand (p1 * p2^8) x) 8 = 15 := sorry

end coefficient_x8_l162_162748


namespace reinforcement_1600_l162_162942

/-- A garrison of 2000 men has provisions for 54 days. After 18 days, a reinforcement arrives, and it is now found that the provisions will last only for 20 days more. We define the initial total provisions, remaining provisions after 18 days, and form equations to solve for the unknown reinforcement R.
We need to prove that R = 1600 given these conditions.
-/
theorem reinforcement_1600 (P : ℕ) (M1 M2 : ℕ) (D1 D2 : ℕ) (R : ℕ) :
  M1 = 2000 →
  D1 = 54 →
  D2 = 20 →
  M2 = 2000 + R →
  P = M1 * D1 →
  (M1 * (D1 - 18) = M2 * D2) →
  R = 1600 :=
by
  intros hM1 hD1 hD2 hM2 hP hEquiv
  sorry

end reinforcement_1600_l162_162942


namespace vector_sum_is_zero_l162_162306

variables {V : Type*} [AddCommGroup V]

variables (AB CF BC FA : V)

-- Condition: Vectors form a closed polygon
def vectors_form_closed_polygon (AB CF BC FA : V) : Prop :=
  AB + BC + CF + FA = 0

theorem vector_sum_is_zero
  (h : vectors_form_closed_polygon AB CF BC FA) :
  AB + BC + CF + FA = 0 :=
  h

end vector_sum_is_zero_l162_162306


namespace maximum_sum_of_first_n_terms_is_S5_l162_162351

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
def condition1 : Prop := a 5 > 0
def condition2 : Prop := a 4 + a 7 < 0
def condition3 : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range (n + 1), a i

-- Main statement
theorem maximum_sum_of_first_n_terms_is_S5
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) :
  ∃ n, S n ≤ S 4 ∧ S n ≤ S 5 ∧ S n ≤ S 6 ∧ S n = S 5 := by
  sorry

end maximum_sum_of_first_n_terms_is_S5_l162_162351


namespace solution_set_fractional_inequality_l162_162226

theorem solution_set_fractional_inequality : {x : ℝ | (x + 5) / (3 - x) ≥ 0} = Icc (-5 : ℝ) 3 \ {3} := sorry

end solution_set_fractional_inequality_l162_162226


namespace power_calc_l162_162460

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162460


namespace domain_of_composed_function_l162_162280

theorem domain_of_composed_function (f : ℝ → ℝ) (h : ∀ x, f x ∈ set.Ioo 1 2) :
  ∀ x, f (2^x) ∈ set.Ioo 0 1 :=
sorry

end domain_of_composed_function_l162_162280


namespace trailing_zeros_factorial_base15_l162_162603

theorem trailing_zeros_factorial_base15 (n : ℕ) (h1 : n = 15) : 
  nat.num_trailing_zeros_in_base (factorial n) 15 = 3 := 
sorry

end trailing_zeros_factorial_base15_l162_162603


namespace power_of_powers_l162_162498

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162498


namespace sum_of_first_100_terms_l162_162000

def sequence (n : ℕ) : ℤ :=
if n = 1 then 1
else if n = 2 then 3
else sequence (n - 1) - sequence (n - 2)

theorem sum_of_first_100_terms : (Finset.sum (Finset.range 100) sequence) = 5 :=
by
  sorry

end sum_of_first_100_terms_l162_162000


namespace max_value_proof_l162_162264

noncomputable def maximum_value (x y z : ℝ) : ℝ := 
  (2/x) + (1/y) - (2/z) + 2

theorem max_value_proof {x y z : ℝ} 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0):
  maximum_value x y z ≤ 3 :=
sorry

end max_value_proof_l162_162264


namespace acuteAngleAt725_l162_162116

noncomputable def hourHandPosition (h : ℝ) (m : ℝ) : ℝ :=
  h * 30 + m / 60 * 30

noncomputable def minuteHandPosition (m : ℝ) : ℝ :=
  m / 60 * 360

noncomputable def angleBetweenHands (h m : ℝ) : ℝ :=
  abs (hourHandPosition h m - minuteHandPosition m)

theorem acuteAngleAt725 : angleBetweenHands 7 25 = 72.5 :=
  sorry

end acuteAngleAt725_l162_162116


namespace average_age_combined_rooms_l162_162339

theorem average_age_combined_rooms :
  (8 * 30 + 5 * 22) / (8 + 5) = 26.9 := by
  sorry

end average_age_combined_rooms_l162_162339


namespace inequality_2m_leq_n_plus_k_l162_162860

noncomputable def f : ℤ → ℤ := sorry
noncomputable def g : ℤ → ℤ := sorry

def m : ℤ := (∑ (x : ℤ) in (Finset.filter (λ (x : ℤ), |x| ≤ 1000) Finset.univ), (∑ (y : ℤ) in (Finset.filter (λ (y : ℤ), |y| ≤ 1000) Finset.univ), if f x = g y then 1 else 0))
def n : ℤ := (∑ (x : ℤ) in (Finset.filter (λ (x : ℤ), |x| ≤ 1000) Finset.univ), (∑ (y : ℤ) in (Finset.filter (λ (y : ℤ), |y| ≤ 1000) Finset.univ), if f x = f y then 1 else 0))
def k : ℤ := (∑ (x : ℤ) in (Finset.filter (λ (x : ℤ), |x| ≤ 1000) Finset.univ), (∑ (y : ℤ) in (Finset.filter (λ (y : ℤ), |y| ≤ 1000) Finset.univ), if g x = g y then 1 else 0))

theorem inequality_2m_leq_n_plus_k : 2 * m ≤ n + k := sorry

end inequality_2m_leq_n_plus_k_l162_162860


namespace problem_1_problem_2_problem_3_l162_162289

open Real

/-- Given the function f defined as: f(x) = (x^2 - x - 1/a) * exp(ax), and a ≠ 0.

Question (Ⅰ) when a = 1/2 --/
theorem problem_1 (a : ℝ) (ha : a = 1/2) 
: ∃ x₁ x₂, (λ x, (x^2 - x - 1 / a) * exp (a * x)) x₁ = 0 ∧
             (λ x, (x^2 - x - 1 / a) * exp (a * x)) x₂ = 0 :=
sorry

/-- Question (Ⅱ) Determine the intervals of monotonicity for f(x). --/
theorem problem_2 (a : ℝ) (x : ℝ) (ha : a ≠ 0) : 
∃ I1 I2 I3, (
  if a < -2 then (I1 = Ioc (-∞) (-2 / a) ∧ I2 = Ioo (-2 / a) 1 ∧ I3 = Ioc 1 ∞) ∧ 
  (∀ x ∈ I1 ∪ I3, f' x < 0) ∧ (∀ x ∈ I2, f' x > 0)
else if a = -2 then (I1 = Ioo (-∞) 1 ∧ I2 = ∅ ∧ I3 = Ioo 1 ∞) ∧ 
  (∀ x ∈ I1 ∪ I3, f' x < 0)
else if -2 < a < 0 then (I1 = Ioo (-∞) 1 ∧ I2 = Ioo (-2 / a) ∞ ∧ I3 = Ioo 1 (-2 / a)) ∧ 
  (∀ x ∈ I1 ∪ I2, f' x < 0) ∧ (∀ x ∈ I3, f' x > 0)
else if 0 < a then (I1 = Ioo (-∞) (-2 / a) ∧ I2 = Ioo (-2 / a) 1 ∧ I3 = Ioo 1 ∞) ∧ 
  (∀ x ∈ I1 ∪ I3, f' x > 0) ∧ (∀ x ∈ I2, f' x < 0)
else false) := 
sorry
  
/-- Question (Ⅲ) when a > 0, if f(x) + 2/a >= 0 holds for all x ∈ ℝ, find the range of a --/
theorem problem_3 (a : ℝ) (ha : 0 < a) (h : ∀ x, (λ x, (x^2 - x - 1 / a) * exp (a * x) + 2 / a) x ≥ 0) : a ∈ Icc (0 : ℝ) (log 2) :=
sorry

end problem_1_problem_2_problem_3_l162_162289


namespace sum_of_quarter_circles_l162_162851

-- Define the diameter D and the number of parts n
variables (D : ℝ) (n : ℕ)

-- Statement to prove
theorem sum_of_quarter_circles (D : ℝ) (n : ℕ) (hn : 0 < n):
  (∑ i in Finset.range n, π * (D / n) / 4) = π * D / 4 :=
begin
  sorry
end

end sum_of_quarter_circles_l162_162851


namespace bookshelf_prices_purchasing_plans_l162_162154

/-
We are given the following conditions:
1. 3 * x + 2 * y = 1020
2. 4 * x + 3 * y = 1440

From these conditions, we need to prove that:
1. Price of type A bookshelf (x) is 180 yuan.
2. Price of type B bookshelf (y) is 240 yuan.

Given further conditions:
1. The school plans to purchase a total of 20 bookshelves.
2. Type B bookshelves not less than type A bookshelves.
3. Maximum budget of 4320 yuan.

We need to prove that the following plans are valid:
1. 8 type A bookshelves, 12 type B bookshelves.
2. 9 type A bookshelves, 11 type B bookshelves.
3. 10 type A bookshelves, 10 type B bookshelves.
-/

theorem bookshelf_prices (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 1020) 
  (h2 : 4 * x + 3 * y = 1440) : 
  x = 180 ∧ y = 240 :=
by sorry

theorem purchasing_plans (m : ℕ) 
  (h3 : 8 ≤ m ∧ m ≤ 10) 
  (h4 : 180 * m + 240 * (20 - m) ≤ 4320) 
  (h5 : 20 - m ≥ m) : 
  m = 8 ∨ m = 9 ∨ m = 10 :=
by sorry

end bookshelf_prices_purchasing_plans_l162_162154


namespace c_work_rate_l162_162913

/--
A can do a piece of work in 4 days.
B can do it in 8 days.
With the assistance of C, A and B completed the work in 2 days.
Prove that C alone can do the work in 8 days.
-/
theorem c_work_rate :
  (1 / 4 + 1 / 8 + 1 / c = 1 / 2) → c = 8 :=
by
  intro h
  sorry

end c_work_rate_l162_162913


namespace f_ln_inv_4_eq_l162_162290

-- Define the function f as per the given problem
def f : ℝ → ℝ :=
λ x, if x > 0 then Real.exp x else sorry -- We use sorry here to skip the proof

-- Define the specific value ln(1/4)
def ln_inv_4 : ℝ := Real.log (1 / 4)

-- State the theorem to prove that f(ln(1/4)) == e^2 / 4
theorem f_ln_inv_4_eq : f ln_inv_4 = Real.exp 2 / 4 := sorry

end f_ln_inv_4_eq_l162_162290


namespace tangent_line_at_a_eq_1_minimum_value_on_interval_secant_condition_l162_162295

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 - (a + 2) * x + Real.log x

-- Statement for Problem I
theorem tangent_line_at_a_eq_1 (a : ℝ) (h : a = 1) : 
  let x := 1 in
  let y := f a x in
  let f' (x : ℝ) := 2 * x - 3 + 1 / x in
  f' x = 0 → y = -2 := 
sorry

-- Statement for Problem II
theorem minimum_value_on_interval (a : ℝ) (h : 0 < a) :
  (∀ x ∈ Set.Icc 1 Real.exp, f a x ≥ -2) → a ≥ 1 := 
sorry

-- Statement for Problem III
theorem secant_condition (a : ℝ) :
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > -2) → 
  0 ≤ a ∧ a ≤ 8 := 
sorry

end tangent_line_at_a_eq_1_minimum_value_on_interval_secant_condition_l162_162295


namespace expected_chocolate_bars_l162_162816

theorem expected_chocolate_bars (n : ℕ) (h : n > 0) : 
  (expected_number n) = n * (∑ i in (Finset.range n).succ, 1 / (i : ℝ)) := 
sorry

end expected_chocolate_bars_l162_162816


namespace power_of_powers_eval_powers_l162_162522

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162522


namespace object_speed_approximate_l162_162716

theorem object_speed_approximate :
  let distance_ft := 400
  let time_sec := 4
  let ft_per_mile := 5280
  let sec_per_hour := 3600
  let distance_miles := distance_ft / ft_per_mile
  let time_hours := time_sec / sec_per_hour
  let speed_mph := distance_miles / time_hours
  speed_mph ≈ 68.18 :=
by
  -- Definitions
  let distance_ft := 400
  let time_sec := 4
  let ft_per_mile := 5280
  let sec_per_hour := 3600
  let distance_miles := distance_ft / ft_per_mile
  let time_hours := time_sec / sec_per_hour
  let speed_mph := distance_miles / time_hours

  -- Proof starts
  have distance_miles_calc : distance_miles = 400 / 5280 := rfl
  have time_hours_calc : time_hours = 4 / 3600 := rfl
  have speed_mph_calc : speed_mph = distance_miles / time_hours := rfl
  
  -- Therefore, approximate speed is correct
  sorry

end object_speed_approximate_l162_162716


namespace max_value_of_expr_l162_162383

noncomputable theory

open Real

/-- Let x be a positive real number. Prove that the maximum possible value of 
    (x² + 3 - sqrt(x⁴ + 9)) / x is 3 - sqrt(6). -/
theorem max_value_of_expr (x : ℝ) (hx : 0 < x) :
  Sup {y : ℝ | ∃ x : ℝ, 0 < x ∧ y = (x^2 + 3 - sqrt (x^4 + 9)) / x} = 3 - sqrt 6 :=
sorry

end max_value_of_expr_l162_162383


namespace part1_part2_l162_162697

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

theorem part1 (x : ℝ) : ∀ (a : ℝ), a = 3 → f x a ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3 := by
  intro a hyp
  rw hyp
  sorry -- proof steps are omitted

theorem part2 (x a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry -- proof steps are omitted

end part1_part2_l162_162697


namespace nancy_pots_created_on_Wednesday_l162_162811

def nancy_pots_conditions (pots_Monday pots_Tuesday total_pots : ℕ) : Prop :=
  pots_Monday = 12 ∧ pots_Tuesday = 2 * pots_Monday ∧ total_pots = 50

theorem nancy_pots_created_on_Wednesday :
  ∀ pots_Monday pots_Tuesday total_pots,
  nancy_pots_conditions pots_Monday pots_Tuesday total_pots →
  (total_pots - (pots_Monday + pots_Tuesday) = 14) := by
  intros pots_Monday pots_Tuesday total_pots h
  -- proof would go here
  sorry

end nancy_pots_created_on_Wednesday_l162_162811


namespace negation_of_universal_l162_162870

theorem negation_of_universal (P : Prop) :
  (¬ (∀ x : ℝ, 0 < x → √x > x + 1)) ↔ (∃ x : ℝ, 0 < x ∧ √x ≤ x + 1) :=
by
  sorry

end negation_of_universal_l162_162870


namespace cube_div_identity_l162_162529

theorem cube_div_identity (a b : ℕ) (h1 : a = 6) (h2 : b = 3) : 
  (a^3 - b^3) / (a^2 + a * b + b^2) = 3 :=
by {
  sorry
}

end cube_div_identity_l162_162529


namespace evaluate_polynomial_at_2_l162_162891

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 3 * x^3 - 2 * x^2 + x

theorem evaluate_polynomial_at_2 : f 2 = 34 :=
by
  -- Use Horner's method in nested form
  suffices h : (((3 * 2 - 5) * 2 + 3) * 2 - 2) * 2 + 1 = 34 from h
  sorry

end evaluate_polynomial_at_2_l162_162891


namespace area_of_GHFD_l162_162755

noncomputable def trapezoid_area (a b h : ℝ) : ℝ := h * ((a + b) / 2)

variables (AB CD h : ℝ)
variable A : AB = 10
variable B : CD = 26
variable H : h = 15

def midpoint_length (x : ℝ) : ℝ := x / 2

def average_length (x y : ℝ) : ℝ := (x + y) / 2

theorem area_of_GHFD : trapezoid_area (average_length AB (midpoint_length CD)) (midpoint_length CD) (h / 2) = 91.875 :=
by
    rw [average_length, midpoint_length, trapezoid_area]
    dsimp
    sorry

end area_of_GHFD_l162_162755


namespace simplify_fraction_l162_162125

theorem simplify_fraction : 
    (3 ^ 1011 + 3 ^ 1009) / (3 ^ 1011 - 3 ^ 1009) = 5 / 4 := 
by
  sorry

end simplify_fraction_l162_162125


namespace power_of_powers_eval_powers_l162_162520

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162520


namespace select_six_squares_mod_12_l162_162832

theorem select_six_squares_mod_12 (s : Finset ℤ) (h₁ : s.card = 11) :
  ∃ a b c d e f ∈ s, (a^2 + b^2 + c^2) % 12 = (d^2 + e^2 + f^2) % 12 :=
by
  sorry

end select_six_squares_mod_12_l162_162832


namespace focus_of_parabola_y_eq_x_sq_l162_162423

theorem focus_of_parabola_y_eq_x_sq : ∃ (f : ℝ × ℝ), f = (0, 1/4) ∧ (∃ (p : ℝ), p = 1/2 ∧ ∀ x, y = x^2 → y = 2 * p * (0, y).snd) :=
by
  sorry

end focus_of_parabola_y_eq_x_sq_l162_162423


namespace count_3digit_base6_divisible_by_3_l162_162313

theorem count_3digit_base6_divisible_by_3 :
  let digits := {0, 2, 4}
  (count : ℕ := 
    {n | let (a, b, c) := (n / 6^2, (n / 6) % 6, n % 6)
         a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ 0 ∧ (a + b + c) % 3 = 0 ∧ n < 6^3}).count = 5 :=
by
  sorry

end count_3digit_base6_divisible_by_3_l162_162313


namespace AA_BP_BQ_concyclic_l162_162005

noncomputable def triangle := Type

noncomputable def inscribed_circle (Δ : triangle) := Type

variables (A B C P Q A' B' : triangle)

-- Conditions
variable (Δ : triangle)
variable (P Q A' B' : triangle)
variable [is_tangent Δ P Q (side BC)]
variable [is_reflection A' A (side BC)]
variable [is_reflection B' B (side CA)]

-- Goal
theorem AA_BP_BQ_concyclic :
  concyclic {A', B', P, Q} :=
sorry

end AA_BP_BQ_concyclic_l162_162005


namespace range_of_a_l162_162657

noncomputable def g (x : ℝ) : ℝ := abs (x-1) - abs (x-2)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (g x ≥ a^2 + a + 1)) ↔ (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l162_162657


namespace problem_9794_l162_162014

theorem problem_9794
  (ABC : Type) [triangle ABC]
  (H : orthocenter ABC)
  (O : circumcenter ABC)
  (P Q : point)
  (X : point)
  (A B C : point)
  (Bx Ox Bc : ℝ)
  (h1 : Bx = 2)
  (h2 : Ox = 1)
  (h3 : Bc = 5)
  (h4 : scalene ABC)
  (h5 : acute ABC)
  (h6 : tangent_line_through_A A H O P)
  (h7 : circumcircle_intersect A O P B H P Q)
  (h8 : line_PQ_intersects_BO P Q B O X) :
  AB * AC = √299 + 3 * √11 := sorry

end problem_9794_l162_162014


namespace given_problem_l162_162130

noncomputable def fraction_of (a b : ℚ) : ℚ := a * b
noncomputable def subtract_fractions (a b : ℚ) : ℚ := a - b
noncomputable def multiply_fractions (a b : ℚ) : ℚ := a * b

theorem given_problem :
  let part1 := subtract_fractions (fraction_of (5/8) (3/7)) (fraction_of (2/3) (1/4)) in
  let part2 := fraction_of (7/9) (fraction_of (2/5) (fraction_of (1/2) 5040)) in
  multiply_fractions part1 part2 = 79 :=
by
  sorry

end given_problem_l162_162130


namespace stephanie_gas_bill_l162_162418

noncomputable def electricity_bill : ℤ := 60
noncomputable def gas_bill_total : ℤ
noncomputable def gas_bill_paid : ℤ := (3 / 4) * gas_bill_total
noncomputable def gas_bill_extra_payment : ℤ := 5
noncomputable def gas_bill_remaining_amount : ℤ := 30
noncomputable def water_bill : ℤ := 40
noncomputable def water_bill_paid : ℤ := water_bill / 2
noncomputable def internet_bill : ℤ := 25
noncomputable def internet_bill_paid : ℤ := 4 * 5

noncomputable def total_gas_bill : ℤ := 4 * 30

theorem stephanie_gas_bill : gas_bill_total = 120 := by
  have electricity : electricity_bill = 60 := by sorry
  have gas : (3 / 4) * gas_bill_total + 5 + 30 = 30 := by sorry
  have water : water_bill = 40 := by sorry
  have internet : 4 * 5 = 20 := by sorry
  have total_payment_needed := 30 := by sorry
  have total_gas : gas_bill_total = 4 * 30 := by sorry
  exact total_gas

end stephanie_gas_bill_l162_162418


namespace angle_A_is_70_degrees_l162_162340

-- Define the angles B, C, and D
def B : ℝ := 120
def C : ℝ := 30
def D : ℝ := 110

-- Prove that angle A is equal to 70 degrees given the conditions.
theorem angle_A_is_70_degrees (B C D : ℝ) (hB : B = 120) (hC : C = 30) (hD : D = 110) : 
  ∃ A : ℝ, A = 180 - C - (D - C) ∧ A = 70 :=
by {
  -- Calculation steps
  let C' := D - C,
  have hC' : C' = 80 := by norm_num [hD, hC],
  let A' := 180 - B,
  have hA' : A' = 60 := by norm_num [hB],
  let A := 180 - C - C',
  have hA : A = 70 := by norm_num [hC', hC],
  exact ⟨A, hA, rfl⟩
}

end angle_A_is_70_degrees_l162_162340


namespace find_de_over_ef_l162_162002

-- Definitions based on problem conditions
variables {A B C D E F : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F] 
variables (a b c d e f : A) 
variables (α β γ δ : ℝ)

-- Conditions
-- AD:DB = 2:3
def d_def : A := (3 / 5) • a + (2 / 5) • b
-- BE:EC = 1:4
def e_def : A := (4 / 5) • b + (1 / 5) • c
-- Intersection F of DE and AC
def f_def : A := (5 • d) - (10 • e)

-- Target Proof
theorem find_de_over_ef (h_d: d = d_def a b) (h_e: e = e_def b c) (h_f: f = f_def d e):
  DE / EF = 1 / 5 := 
sorry

end find_de_over_ef_l162_162002


namespace count_valid_permutations_l162_162029

def is_valid_list (b: List ℕ) : Prop :=
∀ i, 1 < i ∧ i ≤ b.length → (b[i-1] + 2 ∈ b.take (i-1) ∨ b[i-1] - 2 ∈ b.take (i-1))

def odd_integers := [1, 3, 5, 7, 9, 11, 13, 15]

def valid_permutations_count : ℕ :=
(List.permutations odd_integers).countp is_valid_list

theorem count_valid_permutations : valid_permutations_count = 128 :=
sorry

end count_valid_permutations_l162_162029


namespace company_pays_360_per_month_for_storage_l162_162935

theorem company_pays_360_per_month_for_storage :
  (∀ (l w h : ℕ) (total_volume : ℕ) (cost_per_box : ℝ),
  l = 15 ∧ w = 12 ∧ h = 10 ∧ total_volume = 1080000 ∧ cost_per_box = 0.6 →
  let single_box_volume := l * w * h in
  let number_of_boxes := total_volume / single_box_volume in
  let total_payment := number_of_boxes * cost_per_box in
  total_payment = 360) :=
sorry

end company_pays_360_per_month_for_storage_l162_162935


namespace weekly_milk_production_l162_162931

-- Conditions
def number_of_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 1000
def days_in_week : ℕ := 7

-- Statement to prove
theorem weekly_milk_production : (number_of_cows * milk_per_cow_per_day * days_in_week) = 364000 := by
  sorry

end weekly_milk_production_l162_162931


namespace infinite_non_isosceles_triangles_l162_162555

/-- There exist infinitely many non-isosceles triangles with rational side lengths,
rational lengths of altitudes, and perimeter equal to 3. -/
theorem infinite_non_isosceles_triangles :
  ∃ (t : ℕ) (a b c : ℚ), 
    (a + b + c = 3) ∧ 
    (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ 
    (t^2 - 3 * (1 : ℤ)^2 = 1) :=
begin
  sorry
end

end infinite_non_isosceles_triangles_l162_162555


namespace height_difference_l162_162110

-- Define the diameter of a ball
def diameter : ℝ := 8

-- Define the number of layers in Crate X
def layers_X : ℕ := 16

-- Height of Crate X calculation
def height_X : ℝ := layers_X * diameter

-- Distance between consecutive layers in Crate Y (distance between centers of touching balls)
def d : ℝ := 4 * Real.sqrt 3

-- Height of Crate Y calculation
def height_Y : ℝ := diameter + 15 * d

-- Theorem to prove the difference in heights
theorem height_difference :
  height_X - height_Y = 120 - 60 * Real.sqrt 3 :=
by
  -- skipping proof with sorry
  sorry

end height_difference_l162_162110


namespace power_calc_l162_162464

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162464


namespace simple_interest_rate_l162_162917

theorem simple_interest_rate (P R : ℝ) (T : ℕ) (hT : T = 10) (h_double : P * 2 = P + P * R * T / 100) : R = 10 :=
by
  sorry

end simple_interest_rate_l162_162917


namespace max_large_glasses_l162_162428

-- Definitions of the capacities relation
def jug_capacity (S L : ℝ) := 9 * S + 4 * L = 6 * S + 6 * L

-- Proving the maximum number of large glasses
theorem max_large_glasses (S L : ℝ) (h : jug_capacity S L) : 
  jug_volume = 10 * L := 
by 
  -- Simplifying the given condition
  have simplify : 9 * S + 4 * L = 6 * S + 6 * L := h
  -- Algebraic manipulation to find S
  have relation : 3 * S = 2 * L := by linarith
  -- Substitute S = 2/3 * L into the jug capacity equation
  have jug_capacity_final : 9 * (2 * L / 3) + 4 * L = jug_volume := by linarith
  -- Calculating the maximum number of large glasses
  have result : jug_volume = 10 * L := by linarith
  exact result

noncomputable def jug_volume (S L : ℝ) := 10 * L  -- Capacity of the jug in terms of large glasses

-- Sorry for the proof
sorry 

end max_large_glasses_l162_162428


namespace price_of_second_tea_l162_162006

theorem price_of_second_tea (P : ℝ) (h1 : 1 * 64 + 1 * P = 2 * 69) : P = 74 := 
by
  sorry

end price_of_second_tea_l162_162006


namespace initial_conditions_determined_l162_162210

-- Definitions for initial values and conditions
variables (x y : ℤ)
-- Initial conditions equation
def initial_work : Prop := (6 * x * (y + 2) = W)
-- First change condition equation
def first_change : Prop := (9 * (x - 5) * (y + 1) = W)
-- Second change condition equation
def second_change : Prop := (12 * (x - 7) * y = W)

-- Proof statement for the initial number of workers and daily working hours
theorem initial_conditions_determined (x y: ℤ) (W : ℤ) :
  6 * x * (y + 2) = 9 * (x - 5) * (y + 1) ∧
  6 * x * (y + 2) = 12 * (x - 7) * y → 
  x = 21 ∧ (y + 2) = 6 :=
by
  sorry

end initial_conditions_determined_l162_162210


namespace valid_probability_l162_162607

def is_valid_pair (a b : ℚ) : Prop :=
  let a_cos := Math.cos (a.to_real * Real.pi)
  let b_sin := Math.sin (b.to_real * Real.pi)
  in a_cos * (a_cos ^ 2 - b_sin ^ 2) = 0

def rational_candidates : List ℚ := 
  [0, 1, 1/2, 1/3, 2/3, 1/4, 3/4, 1/5, 2/5, 3/5, 4/5, 1/6, 5/6]

def valid_count : ℕ :=
  List.length (List.filter (λ p, is_valid_pair p.1 p.2) 
                           (List.product rational_candidates rational_candidates))

def total_count : ℕ :=
  List.length (List.product rational_candidates rational_candidates)

theorem valid_probability : (valid_count : ℚ) / total_count = 55 / 169 :=
  by sorry

end valid_probability_l162_162607


namespace exp_eval_l162_162482

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l162_162482


namespace correct_choice_l162_162539
open Classical

variable {α : Type}
variable x : ℝ
variables A B : ℝ
variables a b : ℝ
variables (x1 x2 : ℝ)

def condition1 : Prop := (¬(x^2 = 1) → ¬ (x = 1)) = false
def condition2 : Prop := ¬ ((x = -1) → (x^2 + 5 * x - 6 = 0))
def condition3 : Prop := ¬ (∀ x : ℝ, x^2 + x + 1 > 0)
def condition4 : Prop := A > B → (¬(A > B) ∨ a ≠ b → sin A > sin B)

theorem correct_choice : condition1 → condition2 → condition3 → condition4 → (D = true) :=
by {
  sorry
}

end correct_choice_l162_162539


namespace triangle_side_ratios_l162_162324

theorem triangle_side_ratios
    (A B C : ℝ) (a b c : ℝ)
    (h1 : 2 * b * Real.sin (2 * A) = a * Real.sin B)
    (h2 : c = 2 * b) :
    a / b = 2 :=
by
  sorry

end triangle_side_ratios_l162_162324


namespace robot_path_area_l162_162370

-- Representation of vertices as points in 2D space.
structure Point where
  x : ℝ
  y : ℝ

-- Vertices of the enclosed figure (polygon), assuming generalized form more complex paths.
def A : Point := ⟨ 0, 0 ⟩
def B : Point := ⟨ 1, 0 ⟩
def C : Point := ⟨ 1/2, (3:ℝ).sqrt / 2 ⟩
-- Additional vertices can be added for more complex paths.
def D : Point := ⟨ 3/2, (3:ℝ).sqrt / 2 ⟩  -- Example additional vertex

-- Dummy definition for area calculation
noncomputable def area_of_polygon (vertices : List Point) : ℝ := 
  -- TODO: Implement the area calculation using Shoelace formula or other methods
  sorry

-- Conditions - Provided vertices forming the enclosed figure
def vertices : List Point := [A, B, C, D]

-- Proof goal
theorem robot_path_area :
  area_of_polygon vertices = (13 * (3:ℝ).sqrt) / 4 := 
sorry

end robot_path_area_l162_162370


namespace KF_less_or_equal_LE_l162_162352

-- Definitions based on given conditions
variable (MNKLP : Type) [polygon MNKLP] 
variable (N L K M P : MNKLP) 
variable (KF LE : ℝ)

-- Given conditions
variable (bisects_segment_NL_KNP : Bisection (angle K N P) N L)
variable (bisects_segment_NL_KLM : Bisection (angle K L M) N L)
variable (bisects_segment_KP_MKL : Bisection (angle M K L) K P)
variable (bisects_segment_KP_NPL : Bisection (angle N P L) K P)
variable (KF_intersects_NP_F : Intersects (diagonal N P) (diagonal M K) F)
variable (LE_intersects_NP_E : Intersects (diagonal N P) (diagonal M L) E)

-- Statement to prove
theorem KF_less_or_equal_LE : KF ≤ LE := 
sorry

end KF_less_or_equal_LE_l162_162352


namespace parallel_line_perpendicular_line_l162_162638

-- Problem Condition 1
def equation_of_line_parallel (x y : ℝ) : Prop :=
  x + y - 3 = 0

theorem parallel_line (x y : ℝ) 
  (h : equation_of_line_parallel x y) 
  (pt : (1, 2) ∈ {p : ℝ × ℝ | p.1 + p.2 - 3 = 0}) : true :=
sorry

-- Problem Condition 2
def equation_of_line_perpendicular (x y : ℝ) : Prop :=
  x - 3y + 2 = 0

theorem perpendicular_line (x y : ℝ) 
  (h : equation_of_line_perpendicular x y) 
  (pt : (1, 1) ∈ {p : ℝ × ℝ | p.1 - 3 * p.2 + 2 = 0}) : true :=
sorry

end parallel_line_perpendicular_line_l162_162638


namespace determine_c_l162_162001

noncomputable def A : ℝ × ℝ × ℝ := (0, 4, 0)
noncomputable def B : ℝ × ℝ × ℝ := (-2, 2, 1)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def is_opposite (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, λ < 0 ∧ v = (λ * u.1, λ * u.2, λ * u.3)

theorem determine_c :
  let AB := vector_sub B A in
  is_opposite AB (6, 6, -3) ∧ magnitude (6, 6, -3) = 9 := sorry

end determine_c_l162_162001


namespace balls_into_boxes_l162_162709

/-- There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes. -/
theorem balls_into_boxes : (2 : ℕ) ^ 7 = 128 := by
  sorry

end balls_into_boxes_l162_162709


namespace depth_notation_l162_162346

theorem depth_notation (x y : ℤ) (hx : x = 9050) (hy : y = -10907) : -y = x :=
by
  sorry

end depth_notation_l162_162346


namespace perpendicular_line_through_point_l162_162994

def point : ℝ × ℝ := (1, 0)

def given_line (x y : ℝ) : Prop := x - y + 2 = 0

def is_perpendicular_to (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y → l2 (y - x) (-x - y + 2)

def target_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem perpendicular_line_through_point (l1 : ℝ → ℝ → Prop) (p : ℝ × ℝ) :
  given_line = l1 ∧ p = point →
  (∃ l2 : ℝ → ℝ → Prop, is_perpendicular_to l1 l2 ∧ l2 p.1 p.2) →
  target_line p.1 p.2 :=
by
  intro hp hl2
  sorry

end perpendicular_line_through_point_l162_162994


namespace power_of_powers_l162_162496

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162496


namespace find_real_pairs_l162_162992

theorem find_real_pairs (x y : ℝ) :
  y^2 + y + sqrt (y - x^2 - x * y) ≤ 3 * x * y ↔ (x = 0 ∧ y = 0) ∨ (x = 1/2 ∧ y = 1/2) :=
sorry

end find_real_pairs_l162_162992


namespace neg_or_false_of_or_true_l162_162330

variable {p q : Prop}

theorem neg_or_false_of_or_true (h : ¬ (p ∨ q) = false) : p ∨ q :=
by {
  sorry
}

end neg_or_false_of_or_true_l162_162330


namespace fraction_traveled_by_foot_l162_162756

theorem fraction_traveled_by_foot (D : ℝ) (hD : D = 30.000000000000007) :
  let F := D - (3 / 5) * D - 2,
  (F / D) = 1 / 3 :=
by {
  let F := D - (3 / 5) * D - 2,
  have hF : F = 10.000000000000003, {
    calc
    F = D - (3 / 5) * D - 2 : by simp [F]
    ... = 30.000000000000007 - 18.000000000000004 - 2 : by rw hD
    ... = 10.000000000000003 : by norm_num
  },
  have hFraction : F / D = 10.000000000000003 / 30.000000000000007, {
    rw hF,
    rw hD
  },
  calc
  F / D = 10.000000000000003 / 30.000000000000007 : by rw hFraction
  ... = 1 / 3 : by norm_num
}

end fraction_traveled_by_foot_l162_162756


namespace converse_xy_implies_x_is_true_l162_162538

/-- Prove that the converse of the proposition "If \(xy = 0\), then \(x = 0\)" is true. -/
theorem converse_xy_implies_x_is_true {x y : ℝ} (h : x = 0) : x * y = 0 :=
by sorry

end converse_xy_implies_x_is_true_l162_162538


namespace num_valid_arrangements_l162_162156

-- Define the programs
inductive Program
| A | B | C | D | E
deriving DecidableEq, Fintype

open Program

def isValidArrangement (arrangement : Fin 5 → Program) : Prop :=
  (arrangement 0 = A ∨ arrangement 1 = A) ∧       -- Condition 1: Program A must be in the first two positions
  arrangement 0 ≠ B ∧                            -- Condition 2: Program B cannot be in the first position
  arrangement 4 = C                               -- Condition 3: Program C must be in the last position

def validArrangements : Finset (Fin 5 → Program) :=
  Finset.univ.filter (λ arrangement, isValidArrangement arrangement)

theorem num_valid_arrangements : (validArrangements.card = 10) :=
  by
  sorry

end num_valid_arrangements_l162_162156


namespace train_cross_time_l162_162710

theorem train_cross_time (len1 len2 speed1_kmph speed2_kmph : ℕ) (h_len1 : len1 = 250)
    (h_len2 : len2 = 300) (h_speed1 : speed1_kmph = 72) (h_speed2 : speed2_kmph = 36) 
    (speed1 := speed1_kmph * 1000 / 3600) (speed2 := speed2_kmph * 1000 / 3600) 
    (relative_speed := speed1 - speed2) 
    (distance := len1 + len2) : distance / relative_speed = 55 :=
by
  rw [← h_len1, ← h_len2, ← h_speed1, ← h_speed2]
  have hs1 : speed1 = 20 := by sorry
  have hs2 : speed2 = 10 := by sorry
  rw [hs1, hs2]
  have hr : relative_speed = 10 := by sorry
  rw hr
  have hd : distance = 550 := by sorry
  rw hd
  norm_num

end train_cross_time_l162_162710


namespace symmetric_circles_intersect_at_orthocenter_l162_162793

variables (A B C : Point) 
def S : Circle := circumcircle_triangle A B C -- circumcircle of triangle ABC
def is_reflection (p q r : Point) : Prop := dist p r = dist q r
noncomputable def symmetric_circle (S : Circle) (p1 p2 : Point) (H : Point) : Circle := sorry -- definition for the symmetric circle

-- orthocenter: intersection of the altitudes of triangle ABC
noncomputable def orthocenter (A B C : Point) : Point := sorry

theorem symmetric_circles_intersect_at_orthocenter (A B C : Point) :
  let H := orthocenter A B C in
  ∃ P : Point, (is_reflection P A B) ∧ (is_reflection P B C) ∧ (is_reflection P C A) ∧ (P = H) ∧
  symmetric_circle S A B H ∩ symmetric_circle S B C H ∩ symmetric_circle S C A H = {H} :=
by sorry

end symmetric_circles_intersect_at_orthocenter_l162_162793


namespace part1_f_one_eq_zero_part2_f_even_part3_range_of_x_l162_162075

variable {D : Set ℝ} {f : ℝ → ℝ}

axiom h_domain : ∀ x, x ∈ D ↔ x ≠ 0
axiom h_func_eq : ∀ x1 x2 ∈ D, f (x1 * x2) = f x1 + f x2

-- (1) Prove that f(1) = 0
theorem part1_f_one_eq_zero : f 1 = 0 := 
sorry

-- (2) Prove that f(x) is an even function
theorem part2_f_even : ∀ x ∈ D, f (-x) = f x := 
sorry

-- (3) Prove the range of x given:
-- f(4) = 1
-- f(3x + 1) + f(2x - 6) ≤ 3
-- f is increasing on (0, +∞)
axiom h_f_4_eq_1 : f 4 = 1 
axiom h_f_increasing : ∀ {x1 x2 : ℝ}, 0 < x1 → x1 < x2 → f x1 < f x2
axiom h_ineq : ∀ x, f (3 * x + 1) + f (2 * x - 6) ≤ 3

theorem part3_range_of_x : 
{ x : ℝ | -7/3 ≤ x ∧ x < -1/3 } ∪ 
{ x : ℝ | -1/3 < x ∧ x < 3 } ∪ 
{ x : ℝ | 3 < x ∧ x ≤ 5 } = 
{ x : ℝ | f (3 * x + 1) + f (2 * x - 6) ≤ 3 } :=
sorry

end part1_f_one_eq_zero_part2_f_even_part3_range_of_x_l162_162075


namespace second_polygon_sides_l162_162888

theorem second_polygon_sides (s : ℝ) (P : ℝ) (n : ℕ) : 
  (50 * 3 * s = P) ∧ (n * s = P) → n = 150 := 
by {
  sorry
}

end second_polygon_sides_l162_162888


namespace total_number_of_dots_l162_162443

-- Define the basic properties of the dice
structure Dice where
  face1 : ℕ
  face2 : ℕ
  face3 : ℕ
  face4 : ℕ
  face5 : ℕ
  face6 : ℕ
  h1 : face1 + face6 = 7
  h2 : face2 + face5 = 7
  h3 : face3 + face4 = 7

-- Define what it means to be a glued structure of 7 dice
structure GluedDice where
  dice : List Dice
  h_dice_len : dice.length = 7
  glued_faces : ℕ -> ℕ -> ℕ
  h_glued_faces : ∀ i j, glued_faces i j = glued_faces j i

-- Prove that the total number of dots initially marked on the surface of the figure is 75
theorem total_number_of_dots (g : GluedDice) : 
  ∑ n in [1, 1, 6, 2, 2, 5, 3, 3, 4], 2 * n + 21 = 75 := 
by
  sorry

end total_number_of_dots_l162_162443


namespace dvd_packs_l162_162208

theorem dvd_packs (cost_per_pack : ℕ) (discount_per_pack : ℕ) (money_available : ℕ) 
  (h_cost : cost_per_pack = 107) 
  (h_discount : discount_per_pack = 106) 
  (h_money : money_available = 93) : 
  (money_available / (cost_per_pack - discount_per_pack)) = 93 := 
by 
  -- Implementation of the proof goes here
  sorry

end dvd_packs_l162_162208


namespace harmonic_difference_l162_162242

open Nat

def f : ℕ+ → ℚ
| ⟨n, pn⟩ := (Finset.range n).sum (λ i, (1 : ℚ) / (i + 1))

theorem harmonic_difference (k : ℕ) :
    f ⟨2^(k+1), pow_pos (lt_of_succ_lt (succ_lt_pow 2 k)).2⟩ - f ⟨2^k, pow_pos (lt_of_succ_lt (succ_pos k)).2⟩ = 
    (Finset.range (2^(k+1) - 2^k)).sum (λ i, (1 : ℚ) / (2^k + 1 + i)) := 
sorry

end harmonic_difference_l162_162242


namespace feet_of_perpendiculars_on_circle_l162_162648

theorem feet_of_perpendiculars_on_circle
  {A B C A1 A2 B1 B2 C1 C2 : EuclideanGeometry.Point}
  (hABC : EuclideanGeometry.Triangle A B C)
  (hA1 : EuclideanGeometry.IsPerpendicular A1 A B)
  (hA2 : EuclideanGeometry.IsPerpendicular A2 A C)
  (hB1 : EuclideanGeometry.IsPerpendicular B1 B A)
  (hB2 : EuclideanGeometry.IsPerpendicular B2 B C)
  (hC1 : EuclideanGeometry.IsPerpendicular C1 C A)
  (hC2 : EuclideanGeometry.IsPerpendicular C2 C B)
  (h_definitions : EuclideanGeometry.IsAltitudeFoot hA1 hA2 hB1 hB2 hC1 hC2):
  EuclideanGeometry.ConcyclicPoints [A1, A2, B1, B2, C1, C2] :=
begin
  sorry
end

end feet_of_perpendiculars_on_circle_l162_162648


namespace largest_number_with_condition_l162_162639

def arithmetic_mean (a b : ℕ) : ℚ :=
  (a + b : ℚ) / 2

def satisfies_condition (n : ℕ) : Prop :=
  ∀ (d : List ℕ) (h : nat.digits 10 n = d), 
    ∀ i, 1 < i ∧ i < d.length - 1 → d.nth_le i sorry < arithmetic_mean (d.nth_le (i-1) sorry) (d.nth_le (i+1) sorry)

theorem largest_number_with_condition : ∃ n, satisfies_condition n ∧ 
  (∀ m, satisfies_condition m → m ≤ 96433469) ∧ n = 96433469 :=
begin
  use 96433469,
  split,
  sorry, -- To show that 96433469 satisfies the given condition.
  split,
  { intro m,
    sorry -- To show that any number m satisfying the condition is less than or equal to 96433469.
  },
  refl -- To show that n is 96433469.
end

end largest_number_with_condition_l162_162639


namespace num_digits_of_area_code_l162_162589

-- Define the conditions
def digits := {6, 4, 3}
def valid_codes (n : ℕ) : ℕ := 3^n - 1

-- Statement
theorem num_digits_of_area_code : ∃ (n : ℕ), valid_codes n = 26 → n = 3 :=
by
  sorry

end num_digits_of_area_code_l162_162589


namespace exponentiation_rule_example_l162_162502

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162502


namespace least_distance_between_ticks_l162_162401

theorem least_distance_between_ticks :
  (∃ z : ℚ, z = 2 / 143 ∧
    (∀ (k l : ℤ), 
      (1 <= k ∧ k <= 10 ∧ 1 <= l ∧ l <= 12) ∧
      (∃ k' l' : ℤ, 
        (k = k' ∧ l = l') ∧ 
        (0 ≤ k'/11 - l'/13 ∧ k'/11 - l'/13 = z or l'/13 - k'/11 = z)))) :=
sorry

end least_distance_between_ticks_l162_162401


namespace function_is_odd_l162_162761

def f (x : ℝ) : ℝ := (5^x - 1) / (5^x + 1)

theorem function_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by
  intros x
  sorry

end function_is_odd_l162_162761


namespace hyperbola_equation_l162_162281

theorem hyperbola_equation :
  let c := sqrt 5
  let a := 2
  let b := 1
  ∀ x y : ℝ, (x^2 / 4 - y^2 = 1) ↔ 
    (2 * sqrt 5 = 2 * c ∧ 
     sqrt 5 = c ∧ 
     (b / a = 1 / 2) ∧ 
     (c^2 = a^2 + b^2)) :=
by sorry

end hyperbola_equation_l162_162281


namespace range_a_l162_162199

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define that f is increasing on the given interval
def f_increasing (a b : ℝ) (ha : a ∈ Icc (-2 : ℝ) 2) (hb : b ∈ Icc (-2 : ℝ) 2) (h : a ≤ b) : f a ≤ f b := sorry

theorem range_a (x1 x2 a : ℝ) (hx1 : x1 ∈ Icc (0 : ℝ) 3) (hx2 : x2 ∈ Icc (1 : ℝ) 2)
    (ha_cond : 2 * x2 + a / x2 - 1 > f x1)
    (H_f_incr : ∀ a b, f_increasing a b (show a ∈ Icc (-2 : ℝ) 2, by sorry)
    (show b ∈ Icc (-2 : ℝ) 2, by sorry) (show a ≤ b, by sorry)) : 1 < a ∧ a < 2 := 
sorry

end range_a_l162_162199


namespace range_of_m_l162_162802

open Set

noncomputable def M (m : ℝ) : Set ℝ := {x | x ≤ m}
noncomputable def N : Set ℝ := {y | y ≥ 1}

theorem range_of_m (m : ℝ) : M m ∩ N = ∅ → m < 1 := by
  intros h
  sorry

end range_of_m_l162_162802


namespace factorization_l162_162980

theorem factorization (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) :=
by sorry

end factorization_l162_162980


namespace at_least_one_head_probability_l162_162963

open ProbabilityTheory

noncomputable def probability_at_least_one_head := 
  let p_tails := (1 / 2) ^ 4  -- Probability of getting four tails
  let p_at_least_one_head := 1 - p_tails -- Probability of getting at least one head
  p_at_least_one_head

theorem at_least_one_head_probability : probability_at_least_one_head = (15 / 16) := by
  sorry

end at_least_one_head_probability_l162_162963


namespace B_share_is_102_l162_162410

variables (A B C : ℝ)
variables (total : ℝ)
variables (rA_B : ℝ) (rB_C : ℝ)

-- Conditions
def conditions : Prop :=
  (total = 578) ∧
  (rA_B = 2 / 3) ∧
  (rB_C = 1 / 4) ∧
  (A = rA_B * B) ∧
  (B = rB_C * C) ∧
  (A + B + C = total)

-- Theorem to prove B's share
theorem B_share_is_102 (h : conditions A B C total rA_B rB_C) : B = 102 :=
by sorry

end B_share_is_102_l162_162410


namespace price_increase_by_72_8_percent_l162_162008

theorem price_increase_by_72_8_percent (P : ℝ) :
  let P1 := P * 1.20,
      P2 := P1 * 1.20,
      P3 := P2 * 1.20
  in (P3 - P) / P * 100 = 72.8 :=
by 
  sorry

end price_increase_by_72_8_percent_l162_162008


namespace particle_probability_mn_sum_l162_162943

def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1 / 3) * P (x - 1) y + (1 / 3) * P x (y - 1) + (1 / 3) * P (x - 1) (y - 1)

theorem particle_probability : P 3 3 = 7 / 27 :=
sorry

theorem mn_sum : (let m := 7 in let n := 3 in m + n = 10) :=
by norm_num

end particle_probability_mn_sum_l162_162943


namespace plot_area_in_acres_l162_162426

-- Define the scale: 1 cm == 1 mile
def scale_cm_to_mile := 1

-- Define the conversion: 1 square mile == 500 acres
def square_mile_to_acre := 500

-- Define dimensions of the trapezoid (in cm)
def bottom_base_cm := 20
def top_base_cm := 10
def height_cm := 15

-- Define the formula to calculate the area of a trapezoid in cm²
def trapezoid_area_cm2 (bottom base top height : ℝ) := 
  ((bottom + top) * height) / 2

-- Define the expected result in acres after conversion
def expected_area_acres := 112500

-- The actual problem statement.
theorem plot_area_in_acres :
  (trapezoid_area_cm2 bottom_base_cm top_base_cm height_cm) * (scale_cm_to_mile ^ 2) * square_mile_to_acre = expected_area_acres :=
by
  sorry

end plot_area_in_acres_l162_162426


namespace find_n_satisfying_conditions_l162_162114

theorem find_n_satisfying_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -150 ≡ n [MOD 23] ∧ n = 11 :=
sorry

end find_n_satisfying_conditions_l162_162114


namespace arithmetic_sequence_terms_l162_162723

theorem arithmetic_sequence_terms (a_1 a_2 a_3 a_{n-2} a_{n-1} a_n : ℕ) (n : ℕ)
    (h1 : a_1 + a_2 + a_3 = 34)
    (h2 : a_{n-2} + a_{n-1} + a_n = 146)
    (h3 : (n / 2) * (a_1 + a_n) = 390) :
  n = 13 := by
  sorry

end arithmetic_sequence_terms_l162_162723


namespace find_g_one_l162_162080

noncomputable def g : ℝ → ℝ := sorry -- The definition of g will be derived from its functional equation

theorem find_g_one (g : ℝ → ℝ) 
  (H : ∀ x : ℝ, x ≠ 1 / 2 → g(x) + g((x + 2) / (2 - 4 * x)) = 2 * x) : 
  g 1 = 1 := 
sorry

end find_g_one_l162_162080


namespace max_value_of_expression_l162_162625

noncomputable def maxExpressionValue (a : ℝ) : ℝ :=
  sqrt (1 - a^2) * (1 + 2 * a * sqrt (1 - a^2))

theorem max_value_of_expression : ∃ θ : ℝ, 0 < θ < π ∧
  (∀ x : ℝ, 0 < x < π → cos (x / 2) * (1 + sin x) ≤ maxExpressionValue (sin (x / 2))) ∧
  cos (θ / 2) * (1 + sin θ) = maxExpressionValue (sin (θ / 2)) :=
begin
  sorry
end

end max_value_of_expression_l162_162625


namespace T_10_is_31_l162_162104

def T : ℕ → ℕ
| 0     := 0  -- Unused, there is no stage 0
| 1     := 4
| (n+2) := T (n+1) + 3

theorem T_10_is_31 : T 10 = 31 :=
by
  sorry

end T_10_is_31_l162_162104


namespace probability_P_plus_S_condition_l162_162105

open Nat

theorem probability_P_plus_S_condition :
  let S (a b c : ℕ) := a + b + c
  let P (a b c : ℕ) := a * b * c
  let favorable_count := 14620
  let total_count := 34220
  let chosen_from := (finset.range 61).erase 0
  ∀ (a b c : ℕ), a ∈ chosen_from → b ∈ chosen_from → c ∈ chosen_from →
    a ≠ b → b ≠ c → c ≠ a →
    (P a b c + S a b c + 1) % 6 = 0 →
    favorable_count / total_count = 2437 / 5707 := 
by
  sorry

end probability_P_plus_S_condition_l162_162105


namespace rationalize_sqrt5_minus_2_l162_162057

def rationalize_denominator (a b : ℝ) :=
  (1 : ℝ) / (Real.sqrt a - b)

theorem rationalize_sqrt5_minus_2 :
  rationalize_denominator 5 2 = Real.sqrt 5 + 2 :=
by
  sorry

end rationalize_sqrt5_minus_2_l162_162057


namespace general_formula_of_arithmetic_sequence_sum_of_first_n_terms_l162_162252

-- Declare the arithmetic sequence and conditions
variable (a_n : ℕ → ℕ)
variable (b_n : ℕ → ℕ)

-- Given conditions
axiom common_difference_non_zero (d : ℕ) : d ≠ 0
axiom sum_first_four_terms (a1 a2 a3 a4 : ℕ) : 
  a1 + a2 + a3 + a4 = 20
axiom geometric_sequence (a1 a2 a4 : ℕ) :
  a2^2 = a1 * a4

-- First part: General formula for \(a_n\)
theorem general_formula_of_arithmetic_sequence :
  (∀ n, a_n = 2 * n) :=
by
  assume n,
  sorry

-- Definition of \(b_n = n \cdot 2^{a_n}\)
noncomputable def b_n (n : ℕ) : ℕ := n * (2 ^ a_n n)

-- Second part: Sum of first n terms of sequence \(b_n\)
theorem sum_of_first_n_terms (S_n : ℕ → ℕ) :
  (S_n n = ∑ i in range(n), b_n i = ∑ i in range(n), i * (4 ^ i) = ( (3*n - 1) * 4^(n+1) + 4 ) / 9) :=
by
  assume n,
  sorry

end general_formula_of_arithmetic_sequence_sum_of_first_n_terms_l162_162252


namespace no_such_numbers_exist_l162_162010
open Nat

theorem no_such_numbers_exist (x y : ℕ) (h₁: x + y = 2021) : 
  ∀ d, d = gcd x y → 
  1 +  (x / d) *  (y / d) - (x / d) - (y / d) ≠ 2021 / d :=
begin
  sorry,
end

end no_such_numbers_exist_l162_162010


namespace cubic_roots_l162_162024

noncomputable def P (x : ℝ) : ℝ :=
  (-20 * x^3 + 80 * x^2 - 23 * x + 32) / 3

theorem cubic_roots (a b c : ℝ) (h : ∀ x : ℝ, x^3 - 4 * x^2 + x - 1 = 0 → (x = a ∨ x = b ∨ x = c)) :
  a + b + c = 4 ∧ P a = b + c ∧ P b = a + c ∧ P c = a + b ∧ P 4 = -20 :=
by
  have Vieta : a + b + c = 4 := sorry
  have Ha : P a = b + c := sorry
  have Hb : P b = a + c := sorry
  have Hc : P c = a + b := sorry
  have H4 : P (a + b + c) = -20 := sorry
  exact ⟨Vieta, Ha, Hb, Hc, H4⟩

end cubic_roots_l162_162024


namespace find_usual_time_to_journey_l162_162137

noncomputable def usual_time_to_journey (S : ℝ) (T : ℝ) : Prop :=
  let new_speed := (5 / 6) * S
  let time_late := 15 / 60  -- converting 15 minutes to hours
  (S / new_speed) = (T + time_late) / T

theorem find_usual_time_to_journey (S : ℝ) : ∃ T : ℝ, usual_time_to_journey S T ∧ T = 1.25 :=
by
  existsi (1.25 : ℝ)
  unfold usual_time_to_journey
  simp [mul_div_cancel_left (5 / 6 : ℝ) (by norm_num : (6 : ℝ) ≠ 0)]
  linarith


end find_usual_time_to_journey_l162_162137


namespace domain_of_f_l162_162146

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 3)) / Real.sqrt (1 - 2^x)

def domain_condition (x : ℝ) : Prop := (x + 3 > 0) ∧ (1 - 2^x > 0)

theorem domain_of_f :
  {x : ℝ | domain_condition x} = {x : ℝ | -3 < x ∧ x < 0} :=
sorry

end domain_of_f_l162_162146


namespace angle_AED_eq_120_l162_162191

-- Definition of the problem conditions
variables {A B C D E F : Type}
variables [has_point A B C D E F] [has_tangent A B C D E F]
variables (Γ : circle) (triangle_ABC : triangle A B C) (triangle_DEF : triangle D E F)
variables (tangent_point_D : D ∈ B ↔ C) (tangent_point_E : E ∈ A ↔ B) (tangent_point_F : F ∈ A ↔ C)
variables (angle_A : ∠ A = 50°) (angle_B : ∠ B = 70°) (angle_C : ∠ C = 60°)

-- Condition specifying that Γ is both the incircle and circumcircle
variables (incircle_condition : incircle Γ triangle_ABC)
variables (circumcircle_condition : circumcircle Γ triangle_DEF)

-- The statement to be proved
theorem angle_AED_eq_120 (h1 : tangent_point_D) (h2 : tangent_point_E) (h3 : tangent_point_F) :
  ∠ EAD = 120° :=
  sorry

end angle_AED_eq_120_l162_162191


namespace average_speed_of_trip_l162_162567

def total_time (distances speeds : List ℚ) : ℚ :=
  (List.zipWith (λ d s => d / s) distances speeds).sum

def total_distance (distances : List ℚ) : ℚ :=
  distances.sum

def average_speed (distances speeds : List ℚ) : ℚ :=
  total_distance distances / total_time distances speeds

theorem average_speed_of_trip :
  let distances := [60, 65, 45, 30]
  let speeds := [30, 65, 40, 20]
  average_speed distances speeds ≈ 35.56 :=
by
  sorry

end average_speed_of_trip_l162_162567


namespace determine_m_l162_162204

variable {x y z : ℝ}

theorem determine_m (h : (5 / (x + y)) = (m / (x + z)) ∧ (m / (x + z)) = (13 / (z - y))) : m = 18 :=
by
  sorry

end determine_m_l162_162204


namespace angle_ratio_eq_two_l162_162791

open Classical
open_locale Classical

variables {A B C I : Point}
variables [Incircle I A B C] -- I is the center of the incircle of triangle ABC
variables (c1 : Length CA + Length AI = Length BC)

theorem angle_ratio_eq_two (h : Incircle I A B C) (h1 : Length CA + Length AI = Length BC) :
  angle BAC / angle CBA = 2 :=
  sorry

end angle_ratio_eq_two_l162_162791


namespace check_incorrect_derivatives_l162_162129

noncomputable def incorrect_derivative_calculations := 
  [("cos(1/x)", "-1/x * sin(1/x)"), 
   ("x^2 + e^2", "2x + e^2")]

def correct_derivative_cos_inv_x : Prop :=
  ∀ x : ℝ, differentiable_at ℝ (λ x, cos(1/x)) x → 
           deriv (λ x, cos(1/x)) x = -sin(1/x) * (-1/x^2)

def correct_derivative_x_squared_plus_e_squared : Prop :=
  ∀ x : ℝ, differentiable_at ℝ (λ x, x^2 + real.exp(2)) x → 
           deriv (λ x, x^2 + real.exp(2)) x = 2 * x

theorem check_incorrect_derivatives : 
  ¬ correct_derivative_cos_inv_x ∧ ¬ correct_derivative_x_squared_plus_e_squared :=
by 
  -- sorry to bypass the actual proof
  sorry  

end check_incorrect_derivatives_l162_162129


namespace P1_P2_sum_Pk_l162_162787

variable {n : ℕ}
variable (P : Finset ℕ)

-- Assume n is a positive integer no less than 3
axiom h_n_pos : n ≥ 3

-- Assume P is the set {x | x ≤ n, x ∈ ℕ*}
axiom h_P : P = { x | x ≤ n ∧ x > 0 }

-- Definitions of P_k
def P_k (k : ℕ) : ℕ :=
  (Finset.powersetLen k P).sum (λ s, s.sum id)

-- Theorem 1: Finding P_1 and P_2
theorem P1_P2 :
  P_k 1 = n * (n + 1) / 2 ∧
  P_k 2 = n * (n + 1) * (n - 1) / 2 :=
sorry

-- Theorem 2: Sum of P_1 to P_n
theorem sum_Pk :
  ∑ k in Finset.range (n + 1), P_k k = n * (n + 1) * 2^(n - 2) :=
sorry

end P1_P2_sum_Pk_l162_162787


namespace mike_land_sale_price_l162_162808

theorem mike_land_sale_price
  (total_acres : ℕ)
  (cost_per_acre : ℝ)
  (fraction_sold : ℝ)
  (profit : ℝ)
  (total_cost := total_acres * cost_per_acre)
  (acres_sold := total_acres * fraction_sold)
  (total_selling_price := total_cost + profit)
  (P := total_selling_price / acres_sold) :
  total_acres = 200 →
  cost_per_acre = 70 →
  fraction_sold = 0.5 →
  profit = 6000 →
  P = 200 :=
by
  intros h_acres h_cost h_fraction h_profit
  unfold total_cost acres_sold total_selling_price P
  rw [h_acres, h_cost, h_fraction, h_profit]
  norm_num
  sorry

end mike_land_sale_price_l162_162808


namespace value_of_f_ln2_l162_162789

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_ln2 (f_mono : ∀ x y : ℝ, x ≤ y → f(x) ≤ f(y))
  (h : ∀ x : ℝ, f(f(x) - Real.exp x) = Real.exp 1 + 1) : f(Real.log 2) = 3 :=
sorry

end value_of_f_ln2_l162_162789


namespace reflection_xy_plane_reflection_across_point_l162_162343

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def reflect_across_xy_plane (p : Point3D) : Point3D :=
  {x := p.x, y := p.y, z := -p.z}

def reflect_across_point (a p : Point3D) : Point3D :=
  {x := 2 * a.x - p.x, y := 2 * a.y - p.y, z := 2 * a.z - p.z}

theorem reflection_xy_plane :
  reflect_across_xy_plane {x := -2, y := 1, z := 4} = {x := -2, y := 1, z := -4} :=
by sorry

theorem reflection_across_point :
  reflect_across_point {x := 1, y := 0, z := 2} {x := -2, y := 1, z := 4} = {x := -5, y := -1, z := 0} :=
by sorry

end reflection_xy_plane_reflection_across_point_l162_162343


namespace power_calc_l162_162468

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162468


namespace jill_three_months_income_l162_162765

theorem jill_three_months_income :
  let FirstMonthIncome := 10 * 30 in
  let SecondMonthIncome := (10 * 2) * 30 in
  let ThirdMonthIncome := (10 * 2) * (30 / 2) in
  FirstMonthIncome + SecondMonthIncome + ThirdMonthIncome = 1200 :=
by
  sorry

end jill_three_months_income_l162_162765


namespace expression_a_n1_geometric_sequence_general_term_sum_of_sequence_l162_162699

-- Define the quadratic equation and initial condition
variable {α β : ℝ} (a : ℕ → ℝ)
variable (n : ℕ)

-- Initial condition
axiom a1 : a 1 = 1

-- Quadratic equation root condition
axiom root_condition : ∀ {n : ℕ}, 6 * α - 2 * α * β + 6 * β = 3

-- Define a_n+1 in terms of a_n
theorem expression_a_n1 (n : ℕ) : a (n + 1) = (1 / 2) * a n + (1 / 3) :=
sorry

-- Define the sequence {a_n - 2/3} is geometric with common ratio 1/2
theorem geometric_sequence (n : ℕ) : ∃ r : ℝ, r = 1 / 2 ∧ ∀ n, a (n + 1) - 2 / 3 = r * (a n - 2 / 3) :=
sorry

-- General term formula for a_n
theorem general_term (n : ℕ) : a n = (1 / 3) * (1 / 2)^(n - 1) + (2 / 3) :=
sorry

-- Sum of the first n terms of the sequence {a_n} noted as S_n
theorem sum_of_sequence (n : ℕ) : ∑ i in Finset.range n.succ, a i = (2 * n + 2) / 3 - (1 / 3) * (1 / 2)^(n - 1) :=
sorry

end expression_a_n1_geometric_sequence_general_term_sum_of_sequence_l162_162699


namespace remainder_when_divided_by_x_minus_3_l162_162903

open Polynomial

noncomputable def remainder_example : (Polynomial ℤ) :=
  3 * X^3 - 4 * X^2 - 23 * X + 60

noncomputable def divisor_example : (Polynomial ℤ) :=
  X - 3

theorem remainder_when_divided_by_x_minus_3 :
  (remainder_example % divisor_example) = 36 := sorry

end remainder_when_divided_by_x_minus_3_l162_162903


namespace sum_of_solutions_l162_162998

theorem sum_of_solutions:
  ∑ (x : ℝ) in {x | 2 * cos (2 * x) * (cos (2 * x) - cos (1007 * π^2 / x)) = cos (4 * x) - 1 ∧ 0 < x}, x = 1080 * π :=
by
  sorry

end sum_of_solutions_l162_162998


namespace equal_probability_selection_l162_162239

theorem equal_probability_selection 
    (n_total : ℕ) (n_exclude : ℕ) (n_select : ℕ)
    (h_total : n_total = 1008) (h_exclude : n_exclude = 8) (h_select : n_select = 20) :
    ∀ (i : ℕ), i < n_total → (nat.choose (n_total - n_exclude) n_select / n_total) = (n_select / n_total) := 
by
  intro i
  intros hi
  -- Formal proof would go here.
  sorry

end equal_probability_selection_l162_162239


namespace pastries_and_juices_count_l162_162012

theorem pastries_and_juices_count 
  (budget : ℕ) 
  (cost_per_pastry : ℕ) 
  (cost_per_juice : ℕ) 
  (total_money : budget = 50)
  (pastry_cost : cost_per_pastry = 7) 
  (juice_cost : cost_per_juice = 2) : 
  ∃ (p j : ℕ), 7 * p + 2 * j ≤ 50 ∧ p + j = 7 :=
by
  sorry

end pastries_and_juices_count_l162_162012


namespace power_of_powers_l162_162495

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162495


namespace collinear_intersections_l162_162186

variables {ABC : Type*} [triangle ABC] 
variables (A B C : ABC) 
variables {T_A T_B T_C : circle ABC} 
variables {A_1 A_2 D : point ABC} 

-- Conditions
axiom ex1 : tangent T_B (extension B A) A_1
axiom ex2 : tangent T_C (extension C A) A_2
axiom ex3 : inter_line (line_of_pts B A_2) (line_of_pts C A_1) D

-- Main Statement to be proven
theorem collinear_intersections (ABC : Type*) [triangle ABC] :
  concurrent (intersection (line_of_pts A B) (line_tangent T_B T_A)) 
             (intersection (line_of_pts B C) (line_tangent T_C T_B)) 
             (intersection (line_of_pts A C) (line_tangent T_C T_A)) :=
sorry

end collinear_intersections_l162_162186


namespace find_number_of_green_balls_l162_162925

theorem find_number_of_green_balls (G : ℕ) (h : (G - 1) / (2 * G - 1) = 0.46153846153846156) : G = 7 :=
by {
  -- placeholder for the proof
  sorry
}

end find_number_of_green_balls_l162_162925


namespace positive_number_property_l162_162170

theorem positive_number_property (x : ℝ) (h_pos : x > 0) (h_property : 0.01 * x * x = 4) : x = 20 :=
sorry

end positive_number_property_l162_162170


namespace ben_subtracts_99_l162_162448

theorem ben_subtracts_99 :
  ∀ (a : ℕ), a = 50 → (a - 1)^2 = a^2 - 99 :=
by
  intro a h
  rw [h]
  calc
    (50 - 1)^2 = 49^2 : by rfl
            ... = 50^2 - 100 + 1 : by norm_num
            ... = 50^2 - 99 : by norm_num

end ben_subtracts_99_l162_162448


namespace exponentiation_rule_example_l162_162508

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162508


namespace a_2013_is_4_l162_162754

theorem a_2013_is_4
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n : ℕ, a (n+2) = (a n * a (n+1)) % 10) :
  a 2013 = 4 :=
sorry

end a_2013_is_4_l162_162754


namespace inverse_proposition_l162_162864

variable {α : Type*} [AffineSpace α]

def is_parallelogram (quad : α → Prop) : Prop :=
  ∀ (A B C D : α), quad A → quad B → quad C → quad D → (∃ (P₁ P₂ : α), (OppositeSidesParallel P₁ P₂))

def OppositeSidesParallel (A B : α) : Prop :=
  ∀ (C D : α), (Parallel (A, B) (C, D)) → (Parallel (B, C) (D, A))

theorem inverse_proposition (quad : α → Prop) :
  (∀ (A B C D : α), is_parallelogram quad → OppositeSidesParallel (A, C) (B, D)) → is_parallelogram quad :=
by
  sorry

end inverse_proposition_l162_162864


namespace triangular_array_valid_config_count_l162_162590

theorem triangular_array_valid_config_count :
    let n := 12
    let binomial (n k : ℕ) := Nat.choose n k
    let valid_top_value (x : Fin n → ℕ) := ∑ i in Finset.range 12, (binomial 11 i) * x i
    let mod4_top_value (x : Fin n → ℕ) := (x 0 - x 1 - x 10 + x 11) % 4
    (∀ (x : Fin n → ℕ), (x 0 = 0 ∨ x 0 = 1) ∧ 
                         (x 1 = 0 ∨ x 1 = 1) ∧ 
                         (x 2 = 0 ∨ x 2 = 1) ∧ 
                         (x 3 = 0 ∨ x 3 = 1) ∧ 
                         (x 4 = 0 ∨ x 4 = 1) ∧ 
                         (x 5 = 0 ∨ x 5 = 1) ∧ 
                         (x 6 = 0 ∨ x 6 = 1) ∧ 
                         (x 7 = 0 ∨ x 7 = 1) ∧ 
                         (x 8 = 0 ∨ x 8 = 1) ∧ 
                         (x 9 = 0 ∨ x 9 = 1) ∧ 
                         (x 10 = 0 ∨ x 10 = 1) ∧ 
                         (x 11 = 0 ∨ x 11 = 1)) →
    (∀ (x : Fin n → ℕ), mod4_top_value x = 0) →
    (Finset.univ.filter (λ x, mod4_top_value x = 0)).card = 1792 :=
by
  sorry

end triangular_array_valid_config_count_l162_162590


namespace AD_mutually_exclusive_not_complementary_l162_162532

-- Define the sets representing the outcomes of the events
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {2, 4, 6}
def D : Set ℕ := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set ℕ) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set ℕ) : Prop := X ∪ Y = {1, 2, 3, 4, 5, 6}

-- The statement to prove that events A and D are mutually exclusive but not complementary
theorem AD_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬ complementary A D :=
by
  sorry

end AD_mutually_exclusive_not_complementary_l162_162532


namespace capital_formula_minimum_m_l162_162959

-- Define initial conditions
def initial_capital : ℕ := 50000  -- in thousand yuan
def annual_growth_rate : ℝ := 0.5
def submission_amount : ℕ := 10000  -- in thousand yuan

-- Define remaining capital after nth year
noncomputable def remaining_capital (n : ℕ) : ℝ :=
  4500 * (3 / 2)^(n - 1) + 2000  -- in thousand yuan

-- Prove the formula for a_n
theorem capital_formula (n : ℕ) : 
  remaining_capital n = 4500 * (3 / 2)^(n - 1) + 2000 := 
by
  sorry

-- Prove the minimum value of m for which a_m > 30000
theorem minimum_m (m : ℕ) : 
  remaining_capital m > 30000 ↔ m ≥ 6 := 
by
  sorry

end capital_formula_minimum_m_l162_162959


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l162_162613

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l162_162613


namespace area_of_circle_above_below_lines_l162_162961

noncomputable def circle_area : ℝ :=
  40 * Real.pi

theorem area_of_circle_above_below_lines :
  ∃ (x y : ℝ), (x^2 + y^2 - 16*x - 8*y = 0) ∧ (y > x - 4) ∧ (y < -x + 4) ∧
  (circle_area = 40 * Real.pi) :=
  sorry

end area_of_circle_above_below_lines_l162_162961


namespace find_x_values_l162_162218

theorem find_x_values (x : ℝ) :
  (frac_expr : (2 / (x + 2) + 4 / (x + 8))) ≥ 1 / 2 ↔ x ∈ (Set.Ioo (-8 : ℝ) (-2) ∪ Set.Icc 6 8)
:= sorry

end find_x_values_l162_162218


namespace combined_area_correct_l162_162944

noncomputable def breadth : ℝ := 20
noncomputable def length : ℝ := 1.15 * breadth
noncomputable def area_rectangle : ℝ := 460
noncomputable def radius_semicircle : ℝ := breadth / 2
noncomputable def area_semicircle : ℝ := (1/2) * Real.pi * radius_semicircle^2
noncomputable def combined_area : ℝ := area_rectangle + area_semicircle

theorem combined_area_correct : combined_area = 460 + 50 * Real.pi :=
by
  sorry

end combined_area_correct_l162_162944


namespace alex_seashells_l162_162809

theorem alex_seashells (mimi_seashells kyle_seashells leigh_seashells alex_seashells : ℕ) 
    (h1 : mimi_seashells = 2 * 12) 
    (h2 : kyle_seashells = 2 * mimi_seashells) 
    (h3 : leigh_seashells = kyle_seashells / 3) 
    (h4 : alex_seashells = 3 * leigh_seashells) : 
  alex_seashells = 48 := by
  sorry

end alex_seashells_l162_162809


namespace lateral_surface_area_of_pyramid_l162_162717

def regular_square_pyramid (base_edge_length volume lateral_surface_area : ℝ) :=
  ∃ h : ℝ,
    let base_area := base_edge_length ^ 2 in
    let height := volume * 3 / base_area in
    let slant_height := Real.sqrt (height ^ 2 + (base_edge_length / 2) ^ 2) in
    lateral_surface_area = 4 * base_edge_length * slant_height / 2

theorem lateral_surface_area_of_pyramid : 
  regular_square_pyramid (2 * Real.sqrt 2) 8 (4 * Real.sqrt 22) :=
sorry

end lateral_surface_area_of_pyramid_l162_162717


namespace prove_value_of_expression_l162_162781

theorem prove_value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 :=
by 
  sorry

end prove_value_of_expression_l162_162781


namespace rounding_example_l162_162059

theorem rounding_example (x : ℝ) (h : x = 8899.50241201) : round x = 8900 :=
by
  sorry

end rounding_example_l162_162059


namespace angle_relationship_l162_162069

variable {α β : ℝ}
variable (ABC : Triangle) (AK BKC : Ray)
variable (A B : Point)

-- Define the angle bisector relationship
def angle_bisectors_equally_inclined (AK : Ray) (BKC : Ray) : Prop :=
  angle_between AK BKC = angle_between BKC AK

-- Define the given problem
theorem angle_relationship (h1 : angle_bisectors_equally_inclined AK BKC)
  (h2 : α = angle A) (h3 : β = angle B) : α = β ∨ α + β = 120 :=
by sorry

end angle_relationship_l162_162069


namespace polynomial_condition_l162_162779

noncomputable def rad (n : ℕ) : ℕ :=
  n.factors.erase_dup.prod

theorem polynomial_condition (P : ℚ[X]) :
  (∃∞ n : ℕ, P.eval n = rad n) ↔
  (∃ k : ℕ, P = (1 : ℚ[X]) * X / k) ∨ (∃ c : ℕ, nat.is_squarefree c ∧ polynomial.eval_nat c P) :=
sorry

end polynomial_condition_l162_162779


namespace parabola_coeff_sum_eq_neg_four_l162_162216

theorem parabola_coeff_sum_eq_neg_four :
  ∃ a b c : ℝ, 
  (∀ x : ℝ, (x = 4) → (a * (x - 4) ^ 2 + 2 = 2)) ∧ 
  (a * (1 - 4) ^ 2 + 2 = -4) ∧ 
  (a * (7 - 4) ^ 2 + 2 = 0) ∧ 
  (∃ b c : ℝ, x : ℝ → (y = a * x^2 + b * x + c)) ∧ 
  a + b + c = -4 :=
sorry

end parabola_coeff_sum_eq_neg_four_l162_162216


namespace transform_cos_function_l162_162107

theorem transform_cos_function :
  ∀ x : ℝ, 2 * Real.cos (x + π / 3) =
           2 * Real.cos (2 * (x - π / 12) + π / 6) := 
sorry

end transform_cos_function_l162_162107


namespace no_such_subsets_exist_l162_162974

open Classical

theorem no_such_subsets_exist :
  ¬ (∃ (A B : Set ℕ),
    (finite A ∧ 2 ≤ A.card) ∧
    (infinite B) ∧
    (∀ a1 a2 ∈ S, a1 ≠ a2 → Nat.coprime a1 a2) ∧
    (∀ m n : ℕ, Nat.coprime m n → ∃∞ x ∈ {a + b | a ∈ A ∧ b ∈ B}, x % m = n)) :=
  sorry

end no_such_subsets_exist_l162_162974


namespace sum_of_money_is_6000_l162_162588

noncomputable def original_interest (P R : ℝ) := (P * R * 3) / 100
noncomputable def new_interest (P R : ℝ) := (P * (R + 2) * 3) / 100

theorem sum_of_money_is_6000 (P R : ℝ) (h : new_interest P R - original_interest P R = 360) : P = 6000 :=
by
  sorry

end sum_of_money_is_6000_l162_162588


namespace product_of_distances_l162_162677

noncomputable def parametric_equation (t : ℝ) : ℝ × ℝ :=
  ( -1 - (1/2) * t, 2 + (Real.sqrt 3 / 2) * t )

def circle_equation : ℝ × ℝ → Prop
| (x, y) := (x - (1/2))^2 + (y - (Real.sqrt 3 / 2))^2 = 1

def line_intersects_circle (t : ℝ) : Prop :=
  circle_equation (parametric_equation t)
  
theorem product_of_distances :
  ∀ t₁ t₂ : ℝ, line_intersects_circle t₁ → line_intersects_circle t₂ →
  (|t₁| * |t₂| = 6 + 2 * Real.sqrt 3) :=
sorry

end product_of_distances_l162_162677


namespace ethanol_total_amount_l162_162183

-- Definitions based on Conditions
def total_tank_capacity : ℕ := 214
def fuel_A_volume : ℕ := 106
def fuel_B_volume : ℕ := total_tank_capacity - fuel_A_volume
def ethanol_in_fuel_A : ℚ := 0.12
def ethanol_in_fuel_B : ℚ := 0.16

-- Theorem Statement
theorem ethanol_total_amount :
  (fuel_A_volume * ethanol_in_fuel_A + fuel_B_volume * ethanol_in_fuel_B) = 30 := 
sorry

end ethanol_total_amount_l162_162183


namespace c_share_of_profit_l162_162546

theorem c_share_of_profit
  (a_investment : ℝ)
  (b_investment : ℝ)
  (c_investment : ℝ)
  (total_profit : ℝ)
  (ha : a_investment = 30000)
  (hb : b_investment = 45000)
  (hc : c_investment = 50000)
  (hp : total_profit = 90000) :
  (c_investment / (a_investment + b_investment + c_investment)) * total_profit = 36000 := 
by
  sorry

end c_share_of_profit_l162_162546


namespace factorization_l162_162981

theorem factorization (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) :=
by sorry

end factorization_l162_162981


namespace coupon1_greater_pricereduction_l162_162159

def applicable_coupon1 (x : ℝ) : ℝ :=
  if x >= 100 then 0.15 * x else 0

def applicable_coupon2 (x : ℝ) : ℝ :=
  if x >= 150 then 30 else 0

def applicable_coupon3 (x : ℝ) : ℝ :=
  if x >= 150 then 0.25 * (x - 150) else 0

theorem coupon1_greater_pricereduction (x : ℝ) (h1 : 200 < x) (h2 : x < 375) :
  applicable_coupon1 x > applicable_coupon2 x ∧ applicable_coupon1 x > applicable_coupon3 x :=
sorry

end coupon1_greater_pricereduction_l162_162159


namespace symmetric_point_lies_on_median_l162_162387

theorem symmetric_point_lies_on_median
  (A B C D K M : Point)
  (circumcircle_ABM circumcircle_BCM : Circle)
  (BM_median : is_median B M A)
  (tangent_at_A : Tangent A circumcircle_ABM)
  (tangent_at_C : Tangent C circumcircle_BCM)
  (tangents_intersect_at_D : IntersectsAt tangent_at_A tangent_at_C D)
  (K_symmetric_to_D : SymmetricToLine D K (Line.mk A C)) :
  LiesOn K (Line.mk B M) :=
sorry

end symmetric_point_lies_on_median_l162_162387


namespace part_one_equation_of_line_part_two_equation_of_line_l162_162651

-- Definition of line passing through a given point
def line_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop := P.1 / a + P.2 / b = 1

-- Condition: the sum of intercepts is 12
def sum_of_intercepts (a b : ℝ) : Prop := a + b = 12

-- Condition: area of triangle is 12
def area_of_triangle (a b : ℝ) : Prop := (1/2) * (abs (a * b)) = 12

-- First part: equation of the line when the sum of intercepts is 12
theorem part_one_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (sum_of_intercepts a b) →
  (∃ x, (x = 2 ∧ (2*x)+x - 8 = 0) ∨ (x = 3 ∧ x + 3*x - 9 = 0)) :=
by
  sorry

-- Second part: equation of the line when the area of the triangle is 12
theorem part_two_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (area_of_triangle a b) →
  ∃ x, x = 2 ∧ (2*x + 3*x - 12 = 0) :=
by
  sorry

end part_one_equation_of_line_part_two_equation_of_line_l162_162651


namespace prank_combinations_l162_162881

theorem prank_combinations :
  let monday_choices := 2
  let tuesday_choices := 3
  let wednesday_choices := 3
  let thursday_choices := 2
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 36 :=
by
  let monday_choices := 2
  let tuesday_choices := 3
  let wednesday_choices := 3
  let thursday_choices := 2
  let friday_choices := 1
  calc
    2 * 3 * 3 * 2 * 1 = 36 := by
      -- Step 1: Multiply (2 * 3)
      have h1 : 2 * 3 = 6 := rfl
      -- Step 2: Multiply h1 by 3
      have h2 : 6 * 3 = 18 := rfl
      -- Step 3: Multiply h2 by 2
      have h3 : 18 * 2 = 36 := rfl
      -- Step 4: Multiply h3 by 1
      have h4 : 36 * 1 = 36 := rfl
      exact h4

end prank_combinations_l162_162881


namespace problem_part1_problem_part2_l162_162693

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

theorem problem_part1 :
  f (Real.pi / 12) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

theorem problem_part2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  Real.sin θ = 4 / 5 →
  f (5 * Real.pi / 12 - θ) = 72 / 25 :=
by
  sorry

end problem_part1_problem_part2_l162_162693


namespace unit_vector_orthogonal_to_a_and_b_l162_162990

noncomputable def a : ℝ × ℝ × ℝ := (2, 1, 1)
noncomputable def b : ℝ × ℝ × ℝ := (3, 0, 1)
noncomputable def u : ℝ × ℝ × ℝ := (1 / Real.sqrt 11, 1 / Real.sqrt 11, -3 / Real.sqrt 11)

theorem unit_vector_orthogonal_to_a_and_b : 
  ∃ (u : ℝ × ℝ × ℝ), 
    (u.1 = 1 / Real.sqrt 11 ∧ u.2 = 1 / Real.sqrt 11 ∧ u.3 = -3 / Real.sqrt 11) ∧
    a.1 * u.1 + a.2 * u.2 + a.3 * u.3 = 0 ∧
    b.1 * u.1 + b.2 * u.2 + b.3 * u.3 = 0 :=
sorry

end unit_vector_orthogonal_to_a_and_b_l162_162990


namespace percentage_profit_l162_162013

theorem percentage_profit (selling_price original_cost : ℝ) (h1 : selling_price = 550) (h2 : original_cost = 407.41) : (selling_price - original_cost) / original_cost * 100 ≈ 34.99 :=
by {
  have profit : ℝ := selling_price - original_cost,
  have percentage_profit : ℝ := (profit / original_cost) * 100,
  rw [h1, h2] at *,
  have hprofit : profit = 550 - 407.41 := by simp,
  have hpercentage_profit : percentage_profit ≈ 34.99 := by linarith,
  exact hpercentage_profit
}

end percentage_profit_l162_162013


namespace power_of_powers_l162_162490

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162490


namespace exactly_two_lines_passing_through_P_l162_162276

noncomputable def skew_lines (a b : ℝ → ℝ³) := ∀ (t₁ t₂ : ℝ), 
  (a t₁ - b t₂) ≠ 0 /\ (a t₁ - b t₂).dot (a t₁ - b t₂) ≠ 0

variable {P : ℝ³}
variable {a b : ℝ → ℝ³}

axiom angle_between_skew_lines : skew_lines a b → a.angle b = 50 * (π / 180)

theorem exactly_two_lines_passing_through_P (h : skew_lines a b) (h_angle : a.angle b = 50 * (π / 180)) :
  ∃(m n : ℝ → ℝ³), 
  ((m 0 = P) ∧ (n 0 = P)) ∧
  ((m.angle a = 52 * (π / 180)) ∧ (n.angle a = 52 * (π / 180))) ∧
  ((m.angle b = 52 * (π / 180)) ∧ (n.angle b = 52 * (π / 180))) ∧ 
  (m ≠ n) ∧ 
  ∀ (l : ℝ → ℝ³), (l 0 = P) ∧ (l.angle a = 52 * (π / 180)) ∧ (l.angle b = 52 * (π / 180)) → (l = m ∨ l = n) :=
sorry

end exactly_two_lines_passing_through_P_l162_162276


namespace find_base_b_l162_162778

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n + 1) + fibonacci n

-- Define the infinite sum
noncomputable def fibSum (b : ℝ) : ℝ :=
  ∑' n, fibonacci (n + 1) * b ^ (n + 1)

-- Axioms/conditions for the problem
def fibSum_one (b : ℝ) : Prop :=
  fibSum b = 1

-- The theorem to be proven
theorem find_base_b : fibSum_one (-1 + Real.sqrt 2) := sorry

end find_base_b_l162_162778


namespace max_value_of_expr_l162_162382

noncomputable theory

open Real

/-- Let x be a positive real number. Prove that the maximum possible value of 
    (x² + 3 - sqrt(x⁴ + 9)) / x is 3 - sqrt(6). -/
theorem max_value_of_expr (x : ℝ) (hx : 0 < x) :
  Sup {y : ℝ | ∃ x : ℝ, 0 < x ∧ y = (x^2 + 3 - sqrt (x^4 + 9)) / x} = 3 - sqrt 6 :=
sorry

end max_value_of_expr_l162_162382


namespace TA_eq_TM_l162_162736

variables {A B C M E F K L T : Type*} [EuclideanGeometry A B C]

/--
Given an acute-angled triangle \(ABC\) where \(AC > AB\), \(M\) is the midpoint of \([BC]\), \(E\) the foot of the perpendicular from \(B\), and \(F\) the foot of the perpendicular from \(C\). \(K\) is the midpoint of \([ME]\), \(L\) is the midpoint of \([MF]\), and \(T\) is a point on the line \((KL)\) such that \((TA) \parallel (BC)\). Show that \(TA = TM\).
-/
theorem TA_eq_TM (hAcute : ∀ ∠A ∠B ∠C, A B C isAcute)
  (hACgtAB : AC > AB)
  (hMidM : midpoint M B C)
  (hPerpE : foot E B isPerpendicular)
  (hPerpF : foot F C isPerpendicular)
  (hMidK : midpoint K M E)
  (hMidL : midpoint L M F)
  (hTOnKL : pointOnLine T K L)
  (hTAParallelBC : parallel (TA) (BC)) :
TA = TM := sorry

end TA_eq_TM_l162_162736


namespace root_exists_interval_l162_162082

noncomputable def F (x : ℝ) : ℝ := log x + x

theorem root_exists_interval :
  ∃ x ∈ Ioo 0 1, F x = 0 :=
sorry

end root_exists_interval_l162_162082


namespace min_value_omega_l162_162692

def function_f (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (ω * x)

-- Given the function f(x) = 2 * sin (ω * x) where ω > 0, for x in [-π/3, π/4],
-- the function attains a minimum value of -2.
-- We want to prove the minimum value of ω that satisfies this condition.

theorem min_value_omega (ω : ℝ) (hω : 0 < ω) :
  (∀ x ∈ set.Icc (- (π / 3)) (π / 4), function_f ω x ≥ -2) →
  ω = 3 / 2 :=
begin
  sorry,
end

end min_value_omega_l162_162692


namespace probability_even_product_is_correct_l162_162841

noncomputable def probability_even_product : ℚ :=
  let n := 13 in            -- total number of integers from 6 to 18 inclusive
  let total_combinations := Nat.choose n 2 in
  let even_count := 7 in    -- number of even integers in the range
  let odd_count := n - even_count in
  let odd_combinations := Nat.choose odd_count 2 in
  let even_product_combinations := total_combinations - odd_combinations in
  even_product_combinations / total_combinations

theorem probability_even_product_is_correct : probability_even_product = 9 / 13 := by
  sorry

end probability_even_product_is_correct_l162_162841


namespace probability_of_equal_numbers_when_throwing_two_fair_dice_l162_162534

theorem probability_of_equal_numbers_when_throwing_two_fair_dice :
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes = 1 / 6 :=
by
  sorry

end probability_of_equal_numbers_when_throwing_two_fair_dice_l162_162534


namespace only_prop1_is_correct_l162_162288

-- Define conditions
noncomputable def prop1 : Prop :=
  ∀ (cube : Type) [fintype cube] (v1 v2 v3 v4 : cube), ¬ coplanar {v1, v2, v3, v4} →
  is_regular_tetrahedron {v1, v2, v3, v4}

noncomputable def prop2 : Prop :=
  ∀ (pyramid : Type) (base : triangle pyramid) (lateral_faces : set (triangle pyramid)),
  is_equilateral_triangle base →
  (∀ face ∈ lateral_faces, is_isosceles_triangle face) →
  is_regular_tetrahedron (vertices_of_pyramid pyramid base lateral_faces)

noncomputable def prop3 : Prop :=
  ∀ (prism : Type) (base : polygon prism) (lateral_faces : set (polygon prism)),
  (∃ f1 f2 ∈ lateral_faces, is_perpendicular f1 base ∧ is_perpendicular f2 base) →
  is_right_prism prism

-- Define the correctness of the proposition statements
def correct_propositions (p1 p2 p3 : Prop) : set ℕ :=
  {n | if n = 1 then p1 else if n = 2 then p2 else if n = 3 then p3 else false}

-- The statement that only proposition 1 is correct
theorem only_prop1_is_correct : correct_propositions prop1 prop2 prop3 = {1} :=
sorry

end only_prop1_is_correct_l162_162288


namespace exponentiation_identity_l162_162516

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162516


namespace sine_half_angle_mul_lt_quarter_l162_162405

theorem sine_half_angle_mul_lt_quarter {A B C : ℝ} (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (h_sum_ABC : A + B + C = π) :
  sin (A / 2) * sin (B / 2) * sin (C / 2) < 1 / 4 :=
sorry


end sine_half_angle_mul_lt_quarter_l162_162405


namespace hamburgers_left_over_l162_162945

theorem hamburgers_left_over (h_made : ℕ) (h_served : ℕ) (h_total : h_made = 9) (h_served_count : h_served = 3) : h_made - h_served = 6 :=
by
  sorry

end hamburgers_left_over_l162_162945


namespace sum_first_12_terms_l162_162358

theorem sum_first_12_terms (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n * a n)
  (h2 : a 6 + a 7 = 18) : 
  S 12 = 108 :=
sorry

end sum_first_12_terms_l162_162358


namespace min_value_when_a_is_half_range_of_a_for_positivity_l162_162656

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 2*x + a) / x

theorem min_value_when_a_is_half : 
  ∀ x ∈ Set.Ici (1 : ℝ), f x (1/2) ≥ (7 / 2) := 
by 
  sorry

theorem range_of_a_for_positivity :
  ∀ x ∈ Set.Ici (1 : ℝ), f x a > 0 ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by 
  sorry

end min_value_when_a_is_half_range_of_a_for_positivity_l162_162656


namespace log_a_b_is_one_l162_162292

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
if (-1 ≤ x ∧ x < 0) then x + a^(x + 2) else
if (0 ≤ x ∧ x ≤ 1) then b * x - 1 else 0

theorem log_a_b_is_one (a b : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) 
(h_eq : f a b (-1) = f a b 1) : Real.log a b = 1 :=
by
  sorry

end log_a_b_is_one_l162_162292


namespace power_of_powers_l162_162497

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l162_162497


namespace number_of_beautiful_equations_l162_162790

def is_beautiful_equation (A : set ℕ) (b c : ℕ) (root : ℤ) : Prop :=
  b ∈ A ∧ c ∈ A ∧ root ∈ A ∧ (root^2 - b * root - c = 0)

def count_beautiful_equations (A : set ℕ) : ℕ :=
  A.to_finset.sum (λ b, A.to_finset.sum (λ c, (A.to_finset.count (λ root, is_beautiful_equation A b c root))))

theorem number_of_beautiful_equations : count_beautiful_equations {x | x ∈ set.range (λ n, n + 1) 10} = 12 := 
by
  sorry

end number_of_beautiful_equations_l162_162790


namespace feet_of_altitudes_l162_162371

variables (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (HA HB HC MA MB MC : A)
variables (H : A) (AD BE CF : A)

noncomputable def isMidpoint (M X Y : A) : Prop := dist M X = dist M Y

theorem feet_of_altitudes
  (non_isosceles_triangle : ¬ isosceles_triangle A B C)
  (midpoints : (isMidpoint MA B C) ∧ (isMidpoint MB C A) ∧ (isMidpoint MC A B))
  (points_on_sides : (isMidpoint MA HB HC) ∧ (isMidpoint MB HA HC) ∧ (isMidpoint MC HA HB))
  : are_feet_of_altitudes HA HB HC :=
sorry

end feet_of_altitudes_l162_162371


namespace power_of_powers_eval_powers_l162_162526

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162526


namespace equation_has_infinite_solutions_l162_162054

theorem equation_has_infinite_solutions :
  ∀ t : ℤ, ∃ x y z : ℕ,
    x = 2^(15 * t + 12) ∧
    y = 2^(10 * t + 8) ∧
    z = 2^(6 * t + 5) ∧
    (x^2 + y^3 = z^5) := 
by
  intro t
  let x := 2^(15 * t + 12)
  let y := 2^(10 * t + 8)
  let z := 2^(6 * t + 5)
  use [x, y, z]
  split; assumption
  sorry

end equation_has_infinite_solutions_l162_162054


namespace part1_solution_part2_solution_l162_162688

-- Part (1) Statement
theorem part1_solution (x : ℝ) (m : ℝ) (h_m : m = -1) :
  (3 * x - m) / 2 - (x + m) / 3 = 5 / 6 → x = 0 :=
by
  intros h_eq
  rw [h_m] at h_eq
  sorry  -- Proof to be filled in

-- Part (2) Statement
theorem part2_solution (x m : ℝ) (h_x : x = 5)
  (h_eq : (3 * x - m) / 2 - (x + m) / 3 = 5 / 6) :
  (1 / 2) * m^2 + 2 * m = 30 :=
by
  rw [h_x] at h_eq
  sorry  -- Proof to be filled in

end part1_solution_part2_solution_l162_162688


namespace average_speed_for_return_trip_l162_162914

-- Definitions according to conditions in the problem
def distance_first_leg : ℝ := 18
def speed_first_leg : ℝ := 9
def distance_remaining_leg : ℝ := 12
def speed_remaining_leg : ℝ := 10
def total_round_trip_time : ℝ := 7.2

-- Derived from conditions
def time_first_leg : ℝ := distance_first_leg / speed_first_leg
def time_remaining_leg : ℝ := distance_remaining_leg / speed_remaining_leg
def total_time_to_destination : ℝ := time_first_leg + time_remaining_leg
def return_distance : ℝ := distance_first_leg + distance_remaining_leg
def time_return_trip : ℝ := total_round_trip_time - total_time_to_destination

-- The conclusion to prove
def average_speed_return_trip : ℝ := return_distance / time_return_trip

theorem average_speed_for_return_trip :
  average_speed_return_trip = 7.5 := by
  sorry

end average_speed_for_return_trip_l162_162914


namespace max_snack_bars_with_10_l162_162436

-- Define the prices of the snack packs and the budget
def price_single : ℝ := 1
def price_twin_pack : ℝ := 2.5
def price_4_pack : ℝ := 4
def budget : ℝ := 10

-- Define the number of snack bars that can be bought with a given amount of money
noncomputable def max_snack_bars (budget : ℝ) : ℕ :=
  max (nat.floor (budget / price_single))
      (max (2 * nat.floor (budget / price_twin_pack / 2))
           (4 * nat.floor (budget / price_4_pack / 4)))

-- Theorem stating the maximum number of snack bars that can be bought with $10
theorem max_snack_bars_with_10 : max_snack_bars 10 = 10 :=
by {
  sorry -- Proof goes here
}

end max_snack_bars_with_10_l162_162436


namespace probability_of_A_winning_is_correct_l162_162345

def even (n : ℕ) : Prop :=
  n % 2 = 0

def probability_A_wins : ℚ :=
  13 / 25

noncomputable def game_probability_of_A_winning : ℚ :=
  let outcomes : List (ℕ × ℕ) := [(1,1), (1,2), (1,3), (1,4), (1,5),
                                     (2,1), (2,2), (2,3), (2,4), (2,5),
                                     (3,1), (3,2), (3,3), (3,4), (3,5),
                                     (4,1), (4,2), (4,3), (4,4), (4,5),
                                     (5,1), (5,2), (5,3), (5,4), (5,5)]
    let winning_outcomes := outcomes.filter (λ pair, even (pair.1 + pair.2))
    (winning_outcomes.length : ℚ) / (outcomes.length : ℚ)

theorem probability_of_A_winning_is_correct : game_probability_of_A_winning = probability_A_wins := 
  sorry

end probability_of_A_winning_is_correct_l162_162345


namespace mike_initial_marbles_l162_162036

theorem mike_initial_marbles (n : ℕ) 
  (gave_to_sam : ℕ) (left_with_mike : ℕ)
  (h1 : gave_to_sam = 4)
  (h2 : left_with_mike = 4)
  (h3 : n = gave_to_sam + left_with_mike) : n = 8 := 
by
  sorry

end mike_initial_marbles_l162_162036


namespace exponentiation_identity_l162_162518

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l162_162518


namespace odd_function_probability_l162_162927

theorem odd_function_probability :
  let f1 : ℝ → ℝ := λ x, x^3
  let f2 : ℝ → ℝ := λ x, abs x
  let f3 : ℝ → ℝ := λ x, sin x
  let f4 : ℝ → ℝ := λ x, cos x

  let is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
  let is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

  (is_odd f1) ∧ (is_odd f3) ∧ (is_even f2) ∧ (is_even f4) →
  ∃ (p : ℚ), p = 2/3 :=
by
  sorry

end odd_function_probability_l162_162927


namespace right_triangle_legs_from_medians_l162_162970

theorem right_triangle_legs_from_medians
  (a b : ℝ) (x y : ℝ)
  (h1 : x^2 + 4 * y^2 = 4 * a^2)
  (h2 : 4 * x^2 + y^2 = 4 * b^2) :
  y^2 = (16 * a^2 - 4 * b^2) / 15 ∧ x^2 = (16 * b^2 - 4 * a^2) / 15 :=
by
  sorry

end right_triangle_legs_from_medians_l162_162970


namespace integer_solutions_determinant_l162_162958

theorem integer_solutions_determinant (a b c d : ℤ)
    (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
sorry

end integer_solutions_determinant_l162_162958


namespace minimum_chips_required_to_capture_all_cells_l162_162919

/-- Defining the rhombus board and capturing conditions -/
def rhombus_board (angle : ℝ) (divisions : ℕ) : Type :=
  { rhombus | rhombus.angle = angle ∧ rhombus.divisions = divisions }

/-- Proving the minimum number of chips required to capture all cells in the defined rhombus board -/
theorem minimum_chips_required_to_capture_all_cells (angle : ℝ) (divisions : ℕ) (h_angle : angle = 60) (h_divisions : divisions = 9) :
  let board := rhombus_board angle divisions in
  ∃ (chips : ℕ), (∀ cell ∈ board.cells, ∃ chip_pos ∈ chips.positions, cell ∈ chip_pos.captured_cells) ∧ chips = 6 :=
begin
  sorry
end

end minimum_chips_required_to_capture_all_cells_l162_162919


namespace probability_three_draws_exceed_nine_l162_162926

theorem probability_three_draws_exceed_nine :
  let chips := {1, 2, 3, 4, 5, 6}
  in let valid_draws (draws : List ℕ) := draws.sum > 9 ∧ draws.length = 3
  in let possible_draws := {draws | draws ⊆ chips ∧ draws.disjoint draws.tail ∧ valid_draws draws}
  in let total_valid_draws := {draws | draws ⊆ chips ∧ draws.disjoint draws.tail ∧ draws.sum > 9 ∧ draws.length = 3}
  in possible_draws.card = 6.choose(3) / total_valid_draws.card → 
  possible_draws.card / 30 = 2/3 :=
  sorry

end probability_three_draws_exceed_nine_l162_162926


namespace div_by_5_implication_l162_162892

theorem div_by_5_implication (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∃ k : ℕ, ab = 5 * k) : (∃ k : ℕ, a = 5 * k) ∨ (∃ k : ℕ, b = 5 * k) := 
by
  sorry

end div_by_5_implication_l162_162892


namespace number_of_valid_19_tuples_l162_162995

open Nat

-- Define the primary condition for the 19-tuple.
def is_valid_19_tuple (b : Fin 19 → ℤ) : Prop :=
  ∀ i, b i ^ 2 = (Finset.univ.erase i).sum b

-- Prove the number of valid ordered 19-tuples.
theorem number_of_valid_19_tuples : 
  {b : Fin 19 → ℤ | is_valid_19_tuple b}.toFinset.card = 54264 := sorry

end number_of_valid_19_tuples_l162_162995


namespace work_completion_time_l162_162544

theorem work_completion_time (A B : ℕ) (hA : A = 2 * B) (h_work_together : A + B = 3) (h_days : 6 * (A + B) = 18) : B = 1 :=
by
  have h_total_work := 6 * (A + B) -- total work in units
  rw h_work_together at h_total_work -- substitute A + B = 3
  rw [hA, h_days] at h_total_work
  rw mul_assoc at h_total_work
  exact eq_of_mul_eq_mul_right (by norm_num) h_total_work

end work_completion_time_l162_162544


namespace find_number_l162_162563

theorem find_number (x : ℝ) : (30 / 100) * x = (60 / 100) * 150 + 120 ↔ x = 700 :=
by
  sorry

end find_number_l162_162563


namespace merchant_marked_price_l162_162164

-- Given conditions: 30% discount on list price, 10% discount on marked price, 25% profit on selling price
variable (L : ℝ) -- List price
variable (C : ℝ) -- Cost price after discount: C = 0.7 * L
variable (M : ℝ) -- Marked price
variable (S : ℝ) -- Selling price after discount on marked price: S = 0.9 * M

noncomputable def proof_problem : Prop :=
  C = 0.7 * L ∧
  C = 0.75 * S ∧
  S = 0.9 * M ∧
  M = 103.7 / 100 * L

theorem merchant_marked_price (L : ℝ) (C : ℝ) (S : ℝ) (M : ℝ) :
  (C = 0.7 * L) → 
  (C = 0.75 * S) → 
  (S = 0.9 * M) → 
  M = 103.7 / 100 * L :=
by
  sorry

end merchant_marked_price_l162_162164


namespace triangle_ABC_perimeter_ratio_l162_162348

open Real

noncomputable def triangle_perimeter_ratio (AC BC : ℝ) (ABC_right : AC ≠ 0 ∧ BC ≠ 0) : ℝ :=
  let AB := sqrt (AC^2 + BC^2) in
  let CD := (AC * BC) / AB in
  let r := CD / 2 in
  let peripheral_ratio := (69 / 39) in -- based on the given problem solution, simplifying to get the expected ratio
  peripheral_ratio

theorem triangle_ABC_perimeter_ratio (AC BC : ℝ) (hAC_pos : 0 < AC) (hBC_pos : 0 < BC)
  (htri : right_triangle AC BC) :
  triangle_perimeter_ratio AC BC ⟨ne_of_gt hAC_pos, ne_of_gt hBC_pos⟩ = (23 / 13) :=
  sorry


end triangle_ABC_perimeter_ratio_l162_162348


namespace team_CB_days_worked_together_l162_162842

def projectA := 1 -- Project A is 1 unit of work
def projectB := 5 / 4 -- Project B is 1.25 units of work
def work_rate_A := 1 / 20 -- Team A's work rate
def work_rate_B := 1 / 24 -- Team B's work rate
def work_rate_C := 1 / 30 -- Team C's work rate

noncomputable def combined_rate_without_C := work_rate_B + work_rate_C

noncomputable def combined_total_work := projectA + projectB

noncomputable def days_for_combined_work := combined_total_work / combined_rate_without_C

-- Statement to prove the number of days team C and team B worked together
theorem team_CB_days_worked_together : 
  days_for_combined_work = 15 := 
  sorry

end team_CB_days_worked_together_l162_162842


namespace value_of_m_l162_162331

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * Real.log x - m / x

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := 2 / x + m / (x ^ 2)

theorem value_of_m : (f' 1 m = 3) → (m = 1) :=
by {
  intro h,
  sorry
}

end value_of_m_l162_162331


namespace smallest_degree_of_polynomial_with_given_roots_l162_162067

theorem smallest_degree_of_polynomial_with_given_roots :
  ∀ (p : Polynomial ℚ),
    (4 - 3 * Real.sqrt 3 ∈ p.roots ∧
     -4 - 3 * Real.sqrt 3 ∈ p.roots ∧
     2 + Real.sqrt 5 ∈ p.roots ∧
     2 - Real.sqrt 5 ∈ p.roots) →
    p ≠ 0 →
    (finite (p.roots) ∧ p.degree = 6) :=
by sorry

end smallest_degree_of_polynomial_with_given_roots_l162_162067


namespace project_team_work_equation_l162_162936

variable (x : ℝ)

def team_A_days := x + 15
def team_A_rate := 1 / team_A_days

def team_B_days := x + 36
def team_B_rate := 1 / team_B_days

def together_days := x - 17
def together_rate := 1 / together_days

theorem project_team_work_equation :
  team_A_rate x + team_B_rate x = together_rate x := 
  sorry

end project_team_work_equation_l162_162936


namespace arrange_numbers_diagonals_condition_l162_162404

theorem arrange_numbers_diagonals_condition :
  ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 30 ∧
  {a, b, c, d, e} = {2, 4, 6, 8, 10} ∧
  ∃ (m1 m2 m3 m4 : Nat), 
  (a = m1 + e + m2 ∧ c = m3 + e + m4 ∧ m1 + m3 + m4 + m2 = a + d + c + b) ∧
  (b + e + d = 20 ∧ m1 + m3 + m4 + m2 = 20) :=
sorry

end arrange_numbers_diagonals_condition_l162_162404


namespace twoN_plus_1_prime_if_fN_zero_compute_f_1997_l162_162385

def marking_procedure (N : ℤ) (i : ℤ) : Prop := 
  let n := i in
  -- Vertex numeration and marking rules
  if i = 1 then true else
  if i > 1 ∧ i <= N 
  then n + N - 1
  else 
  true

theorem twoN_plus_1_prime_if_fN_zero (N : ℕ) (hN : N > 2) (hfN : marking_procedure (N : ℤ) = 0) : 
  Nat.prime (2 * N + 1) :=
by 
  sorry

-- Auxiliary function to compute f(N) for a given N
noncomputable def f_N (N : ℤ) : ℤ := 
  sorry

theorem compute_f_1997 : f_N 1997 = 3810 :=
by 
  sorry

end twoN_plus_1_prime_if_fN_zero_compute_f_1997_l162_162385


namespace tan_value_l162_162268

open Real

theorem tan_value (α : ℝ) 
  (h1 : sin (α + π / 6) = -3 / 5)
  (h2 : -2 * π / 3 < α ∧ α < -π / 6) : 
  tan (4 * π / 3 - α) = -4 / 3 :=
sorry

end tan_value_l162_162268


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l162_162612

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l162_162612


namespace regular_polygon_exterior_angle_18_deg_has_20_sides_and_3240_sum_of_interior_angles_l162_162175

theorem regular_polygon_exterior_angle_18_deg_has_20_sides_and_3240_sum_of_interior_angles
  (n : ℕ)
  (h : 18 * n = 360) :
  n = 20 ∧ 180 * (20 - 2) = 3240 := 
by
  split
  · have h1 : n = 20 := by
      sorry
    exact h1
  · have h2 : 180 * (20 - 2) = 3240 := by
      sorry
    exact h2

end regular_polygon_exterior_angle_18_deg_has_20_sides_and_3240_sum_of_interior_angles_l162_162175


namespace angle_in_fourth_quadrant_l162_162273

theorem angle_in_fourth_quadrant (α : ℝ) (h1 : Real.sin (2 * α) < 0) (h2 : Real.sin α - Real.cos α < 0) :
  (π < α ∧ α < 2 * π) ∨ (-0 * π < α ∧ α < 0 * π) := sorry

end angle_in_fourth_quadrant_l162_162273


namespace additional_savings_correct_l162_162399

def initial_order_amount : ℝ := 10000

def option1_discount1 : ℝ := 0.20
def option1_discount2 : ℝ := 0.20
def option1_discount3 : ℝ := 0.10
def option2_discount1 : ℝ := 0.40
def option2_discount2 : ℝ := 0.05
def option2_discount3 : ℝ := 0.05

def final_price_option1 : ℝ :=
  initial_order_amount * (1 - option1_discount1) *
  (1 - option1_discount2) *
  (1 - option1_discount3)

def final_price_option2 : ℝ :=
  initial_order_amount * (1 - option2_discount1) *
  (1 - option2_discount2) *
  (1 - option2_discount3)

def additional_savings : ℝ :=
  final_price_option1 - final_price_option2

theorem additional_savings_correct : additional_savings = 345 :=
by
  sorry

end additional_savings_correct_l162_162399


namespace problem_solution_l162_162304

-- Definitions for the sequence
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2
  else 2 * seq (n - 1) + 2^n

-- Definition that {a_n / 2^n} is an arithmetic sequence
def an_div_2n_is_arithmetic : Prop :=
  ∃ a d : ℝ, (∀ n : ℕ, seq n / 2^n = a + (n - 1) * d) ∧ a = 1 ∧ d = 1

-- Definition for the sum of the first n terms
def Sn (n : ℕ) : ℕ :=
  ∑ k in range n, seq (k + 1)

-- Final theorem
theorem problem_solution (n : ℕ) (h : 1 ≤ n) :
  an_div_2n_is_arithmetic ∧ Sn n = (n - 1) * 2^(n+1) + 2 := by
  sorry

end problem_solution_l162_162304


namespace cube_unfolds_to_figure5_cube_unfolds_to_figure6_l162_162132

-- Define the cube properties
structure Cube where
  faces : Fin 6 → Square    -- A cube has six faces which are all squares

-- Part a: Prove that the surface of a cube can be cut and unfolded into a specific net (Figure 5)
theorem cube_unfolds_to_figure5 (c : Cube) :
  ∃ cuts : List (Fin 6 × Fin 6), 
    net_of_cuts c.cuts = figure5_net := sorry

-- Part b: Prove that the surface of a cube can be cut and unfolded into a specific net (Figure 6)
theorem cube_unfolds_to_figure6 (c : Cube) :
  ∃ cuts : List (Fin 6 × Fin 6), 
    net_of_cuts c.cuts = figure6_net := sorry

-- Definitions of net_of_cuts, figure5_net, and figure6_net should be added 
-- but this code provides the structure only

end cube_unfolds_to_figure5_cube_unfolds_to_figure6_l162_162132


namespace evaluate_imaginary_expression_l162_162212

theorem evaluate_imaginary_expression (i : ℂ) (h_i2 : i^2 = -1) (h_i4 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + 3 * i^34 + 2 * i^39 = -3 - 2 * i :=
by sorry

end evaluate_imaginary_expression_l162_162212


namespace rick_iron_hours_l162_162828

def can_iron_dress_shirts (h : ℕ) : ℕ := 4 * h

def can_iron_dress_pants (hours : ℕ) : ℕ := 3 * hours

def total_clothes_ironed (h : ℕ) : ℕ := can_iron_dress_shirts h + can_iron_dress_pants 5

theorem rick_iron_hours (h : ℕ) (H : total_clothes_ironed h = 27) : h = 3 :=
by sorry

end rick_iron_hours_l162_162828


namespace probability_of_selecting_WINDOW_letters_l162_162361

theorem probability_of_selecting_WINDOW_letters :
  let P_bird := 1 / (Nat.choose 4 2),
      P_winds := 3 / (Nat.choose 5 3),
      P_flow := 1 / (Nat.choose 4 2)
  in P_bird * P_winds * P_flow = 1 / 120 :=
by
  sorry

end probability_of_selecting_WINDOW_letters_l162_162361


namespace John_can_put_weight_on_bar_l162_162773

-- Definitions for conditions
def max_capacity : ℕ := 1000
def safety_margin : ℕ := 200  -- 20% of 1000
def johns_weight : ℕ := 250

-- Statement to prove
theorem John_can_put_weight_on_bar : ∀ (weight_on_bar : ℕ),
  weight_on_bar + johns_weight ≤ max_capacity - safety_margin → weight_on_bar = 550 :=
by
  intro weight_on_bar
  intros h_condition
  have h_max_weight : max_capacity - safety_margin = 800 := by simp [max_capacity, safety_margin]
  have h_safe_weight : 800 - johns_weight = 550 := by simp [johns_weight]
  rw [←h_safe_weight] at h_condition
  exact Eq.trans (Eq.symm h_condition) (Eq.refl 550)

end John_can_put_weight_on_bar_l162_162773


namespace power_calc_l162_162467

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l162_162467


namespace exists_monic_degree8_real_roots_l162_162650

variable (α : Fin 16 → ℝ)

def V (P : ℝ[X]) : ℝ :=
  (Finset.univ.sum (Finset.univ.map (P.eval) (set.univ.map (Finset Ico.coeFn α))))

theorem exists_monic_degree8_real_roots (α : Fin 16 → ℝ) :
  ∃ Q : ℝ[X], Q.monic ∧ Q.degree = 8 ∧ (∀ P : ℝ[X], P.degree < 8 → V α (Q * P) = 0) ∧
    (∀ r : ℝ, Q.eval r = 0) :=
sorry

end exists_monic_degree8_real_roots_l162_162650


namespace range_of_slopes_angle_PFQ_constant_l162_162685

-- Define the problem parameters
variables {a : ℝ} (ha : a > real.sqrt 2) (e : ℝ) (he : e = (real.sqrt 2) / 2)
noncomputable def c := real.sqrt (a^2 - 2)

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / 2) = 1

-- Define the point M distinct from vertices A and B
variables {x_0 y_0 : ℝ} (hM : ellipse_C x_0 y_0) (hx0 : x_0 ≠ 2) (hx0' : x_0 ≠ -2)

-- Part I: Find the range of possible slopes for line AM
theorem range_of_slopes (m : ℝ) (hm : -real.sqrt 2 < m ∧ m < real.sqrt 2) :
  (m / 2 > -real.sqrt 2 / 2 ∧ m / 2 < 0) ∨ (m / 2 > 0 ∧ m / 2 < real.sqrt 2 / 2) :=
sorry

-- Part II: Prove that ∠PFQ is a constant value
theorem angle_PFQ_constant :
  ∀ (x_0 y_0 : ℝ), (ellipse_C x_0 y_0) → 
      let F : (ℝ × ℝ) := (real.sqrt 2, 0) in
      let P : (ℝ × ℝ) := (0, (2 * y_0) / (x_0 + 2)) in
      let Q : (ℝ × ℝ) := (0, (2 * y_0) / (x_0 - 2)) in
      ∃ θ : ℝ, θ = 90 :=
sorry

end range_of_slopes_angle_PFQ_constant_l162_162685


namespace sufficiency_Group1_Rhombus_sufficiency_Group2_Rhombus_l162_162804

-- Definition of a rhombus
structure Rhombus (R : Type) [EuclideanGeometry R] extends Parallelogram R where
  sides_eq : ∀ a b : R, dist a b = dist b c

-- Group 1 conditions
def Group1 (R : Type) [EuclideanGeometry R] (quad : Quadrilateral R) : Prop :=
  ∀ a b c d : R, dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a

-- Group 2 conditions
def Group2 (R : Type) [EuclideanGeometry R] (quad : Quadrilateral R) : Prop :=
  Group1 R quad ∧ ∀ a b c d : R, ∃ p : R, ∠ a p c = ∠ b p d = π / 2

-- Proof statements
theorem sufficiency_Group1_Rhombus (R : Type) [EuclideanGeometry R] (quad : Quadrilateral R) :
  Group1 R quad → (isRhombus R quad) := sorry

theorem sufficiency_Group2_Rhombus (R : Type) [EuclideanGeometry R] (quad : Quadrilateral R) :
  Group2 R quad → (isRhombus R quad) := sorry

end sufficiency_Group1_Rhombus_sufficiency_Group2_Rhombus_l162_162804


namespace ratio_height_to_edge_is_sqrt2_volume_of_cone_is_correct_l162_162815

noncomputable def ratio_height_to_edge (T C C1 : Point) (A B C A1 B1 C1 : Triangle) : ℝ :=
  if (C T : C1 T = 1 : 3) then
    some_sqrt(2)
  else
    sorry -- theoretically unreachable if conditions are met

noncomputable def volume_of_cone (T C C1 : Point) (A B C A1 B1 C1 : Triangle) (BB1_length : ℝ) : ℝ :=
  if BB1_length = 8 then
    576 * π * sqrt(3) / (11 * sqrt(11))
  else
    sorry -- theoretically unreachable if conditions are met

-- To prove the equivalence of these mathematical concepts:
theorem ratio_height_to_edge_is_sqrt2 (T C1 : Point) (A B C A1 B1 C1 : Triangle)
  (h_ratio : C T : C1 T = 1 : 3) :
  ratio_height_to_edge T C C1 A B C A1 B1 C1 = sqrt(2) :=
sorry

theorem volume_of_cone_is_correct (T C1 : Point) (A B C A1 B1 C1 : Triangle) (BB1_length : ℝ)
  (h_ratio : C T : C1 T = 1 : 3) (h_BB1 : BB1_length = 8) :
  volume_of_cone T C C1 A B C A1 B1 C1 BB1_length = (576 * π * sqrt(3)) / (11 * sqrt(11)) :=
sorry

end ratio_height_to_edge_is_sqrt2_volume_of_cone_is_correct_l162_162815


namespace total_number_of_animals_l162_162366

theorem total_number_of_animals :
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := (antelopes + rabbits) - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  let giraffes := antelopes + 15
  let lions := leopards + giraffes
  let elephants := 3 * lions
  let zebras := (3 * antelopes) / 4
  let hippos := zebras + (zebras / 10)
  let total := antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants + zebras + hippos
  in total = 1334 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := (antelopes + rabbits) - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  let giraffes := antelopes + 15
  let lions := leopards + giraffes
  let elephants := 3 * lions
  let zebras := (3 * antelopes) / 4
  let hippos := zebras + (zebras / 10)
  let total := antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants + zebras + hippos
  have h : total = 1334 := by sorry
  exact h

end total_number_of_animals_l162_162366


namespace distinct_numbers_not_all_F_in_set_l162_162818

theorem distinct_numbers_not_all_F_in_set (S : Finset ℕ) (h : S.card = 21) (h1 : ∀ x ∈ S, x ≤ 1000000) (h2 : ∀ a b ∈ S, a ≠ b) :
  ∃ x ∉ S, ∃ a b ∈ S, x = a + b - Nat.gcd a b :=
by
  sorry

end distinct_numbers_not_all_F_in_set_l162_162818


namespace problem_l162_162573

noncomputable def exists_three_numbers_cube (numbers : Fin 175 → ℕ) : Prop :=
  ∀ i, ∀ j, ∀ k, 
    i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    (numbers i * numbers j * numbers k) ∣ (2 * 3 * 5 * 7) ^ 3

theorem problem :
  ∃ (numbers : Fin 175 → ℕ),
  (∀ i : Fin 175, ∀ p : ℕ, p ∣ numbers i → p ∈ {2, 3, 5, 7}) →
  ∃ i, ∃ j, ∃ k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ ∃ m : ℕ, (numbers i * numbers j * numbers k) = m ^ 3 :=
begin
  sorry
end

end problem_l162_162573


namespace tangent_line_curve_l162_162078

theorem tangent_line_curve (a b : ℚ) 
  (h1 : 3 * a + b = 1) 
  (h2 : a + b = 2) : 
  b - a = 3 := 
by 
  sorry

end tangent_line_curve_l162_162078


namespace composite_number_divisible_by_11_l162_162823

theorem composite_number_divisible_by_11 
  (a : ℕ → ℕ) (n : ℕ) (alt_sum : ℕ → ℤ) :
    alt_sum n = ∑ i in range (n + 1), (-1) ^ i * a i → 
    alt_sum n % 11 = 0 → ∃ m, m > 1 ∧ ∃ k, k > 1 ∧ m * k = alt_sum n := 
by 
  sorry

end composite_number_divisible_by_11_l162_162823


namespace rotational_homothety_center_mapping_l162_162581

-- Definitions
variable (A B C D Q O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace Q] [MetricSpace O]

def isParallelogram (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop := sorry
def isNotRhombus (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop := sorry
def isCenterOfParallelogram (O : Type) [MetricSpace O] (A B C D : Type) : Prop := sorry
def linesSymmetricToRespectDiagonals (A B C D Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace Q] : Prop := sorry

-- Theorem Statement
theorem rotational_homothety_center_mapping (h1 : isParallelogram A B C D)
    (h2 : isNotRhombus A B C D)
    (h3 : linesSymmetricToRespectDiagonals A B C D Q)
    (h4 : isCenterOfParallelogram O A B C D) :
    ∃ (Q : Type), Q = centerOfRotationalHomothety O A B C D :=
sorry

end rotational_homothety_center_mapping_l162_162581


namespace cash_still_missing_l162_162769

theorem cash_still_missing (c : ℝ) (h : c > 0) :
  (1 : ℝ) - (8 / 9) = (1 / 9 : ℝ) :=
by
  sorry

end cash_still_missing_l162_162769


namespace largest_angle_right_triangle_l162_162800

theorem largest_angle_right_triangle
  (a b c : ℝ)
  (h₁ : ∃ x : ℝ, x^2 + 4 * (c + 2) = (c + 4) * x)
  (h₂ : a + b = c + 4)
  (h₃ : a * b = 4 * (c + 2))
  : ∃ x : ℝ, x = 90 :=
by {
  sorry
}

end largest_angle_right_triangle_l162_162800


namespace f_zero_eq_zero_f_one_eq_one_f_n_is_n_l162_162966

variable (f : ℤ → ℤ)

axiom functional_eq : ∀ m n : ℤ, f (m^2 + f n) = f (f m) + n

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_one_eq_one : f 1 = 1 :=
sorry

theorem f_n_is_n : ∀ n : ℤ, f n = n :=
sorry

end f_zero_eq_zero_f_one_eq_one_f_n_is_n_l162_162966


namespace bob_initial_nickels_l162_162323

-- Conditions
variables (a b : ℕ)
axiom cond1 : b + 1 = 4 * (a - 1)
axiom cond2 : b - 1 = 3 * (a + 1)

-- Theorem statement
theorem bob_initial_nickels : ∃ a, b = 31 :=
by
  sorry

end bob_initial_nickels_l162_162323


namespace even_product_probability_l162_162838

theorem even_product_probability:
  let s := set.Icc 6 18 in
  ((∃ a b : ℤ, a ≠ b ∧ a ∈ s ∧ b ∈ s ∧ ¬even (a * b)) → (2 / 13) - (19 / 39) = (21 / 26)) := sorry

end even_product_probability_l162_162838


namespace expansion_of_expression_l162_162630

theorem expansion_of_expression (x : ℝ) :
  let a := 15 * x^2 + 5 - 3 * x
  let b := 3 * x^3
  a * b = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end expansion_of_expression_l162_162630


namespace triangle_area_l162_162102

theorem triangle_area (c b : ℝ) (c_eq : c = 15) (b_eq : b = 9) :
  ∃ a : ℝ, a^2 = c^2 - b^2 ∧ (b * a) / 2 = 54 := by
  sorry

end triangle_area_l162_162102


namespace probability_of_sum_15_l162_162149

-- Defining the set of cards and the selection process
def cards : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- A function to check if the sum of chosen cards is 15
def is_sum_15 (s : Finset ℕ) : Prop :=
  s.sum id = 15

-- The set of all possible selections of 4 cards out of 6
def all_selections : Finset (Finset ℕ) := Finset.powersetLen 4 cards

-- The set of all selections where the sum is 15
def favorable_selections : Finset (Finset ℕ) :=
  Finset.filter is_sum_15 all_selections

-- The probability of selecting 4 cards which sum up to 15
def probability_sum_15 : ℚ :=
  (favorable_selections.card : ℚ) / (all_selections.card : ℚ)

theorem probability_of_sum_15 : probability_sum_15 = 2 / 15 := by
  sorry

end probability_of_sum_15_l162_162149


namespace perpendicular_bisector_fixed_point_l162_162820

noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x

noncomputable def A (r β : ℝ) : ℝ × ℝ := (r * Real.cos β, r * Real.sin β)

noncomputable def B (r α β : ℝ) : ℝ × ℝ := (-2 * r + r * Real.cos (α + β), r * Real.sin (α + β))

theorem perpendicular_bisector_fixed_point (r α β : ℝ) :
  ∃ (D : ℝ × ℝ), D = (-r, -r * cot (α / 2)) ∧
    ∀ β, (∃ M, M = ((Prod.fst (A r β) + Prod.fst (B r α β)) / 2, (Prod.snd (A r β) + Prod.snd (B r α β)) / 2) ∧ 
    is_on_perpendicular_bisector D M (A r β) (B r α β)) :=
sorry

noncomputable def is_on_perpendicular_bisector (D M A B : ℝ × ℝ) : Prop :=
  ∃ (slope_AB : ℝ), slope_AB = 
    (Prod.snd B - Prod.snd A) / 
    (Prod.fst B - Prod.fst A) ∧ 
  ∃ (slope_perp_bis : ℝ),
    slope_perp_bis = -1 / slope_AB ∧
    (Prod.snd D - Prod.snd M) = slope_perp_bis * (Prod.fst D - Prod.fst M)

end perpendicular_bisector_fixed_point_l162_162820


namespace tangent_line_circle_l162_162719

theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*y = 0 → y = a) → (a = 0 ∨ a = 2) :=
by
  sorry

end tangent_line_circle_l162_162719


namespace symmetry_condition_l162_162291

theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - a| = |(2 - x) + 1| + |(2 - x) - a|) ↔ a = 3 :=
by
  sorry

end symmetry_condition_l162_162291


namespace total_opponent_runs_l162_162565

structure Game :=
  (runs_scored : ℕ)
  (runs_opponent : ℕ)

def games : List Game := [
  {runs_scored := 1, runs_opponent := 3},
  {runs_scored := 2, runs_opponent := 1},
  {runs_scored := 3, runs_opponent := 5},
  {runs_scored := 4, runs_opponent := 1},
  {runs_scored := 5, runs_opponent := 7},
  {runs_scored := 6, runs_opponent := 2},
  {runs_scored := 7, runs_opponent := 9},
  {runs_scored := 8, runs_opponent := 3},
  {runs_scored := 9, runs_opponent := 11},
  {runs_scored := 10, runs_opponent := 3},
  {runs_scored := 11, runs_opponent := 13},
  {runs_scored := 12, runs_opponent := 4}
]

theorem total_opponent_runs : ∑ g in games, g.runs_opponent = 62 :=
by
  sorry

end total_opponent_runs_l162_162565


namespace angle_measure_l162_162440

theorem angle_measure (x : ℝ) (h : x + (3 * x - 10) = 180) : x = 47.5 := 
by
  sorry

end angle_measure_l162_162440


namespace permutations_count_l162_162374

open Nat

theorem permutations_count :
  ∃ (a : Fin 15 → Fin 15),
    (∀ i j : Fin 7, i < j → a i > a j) ∧
    (∀ i j : Fin 8, i < j → a (Fin.add ⟨7, by decide⟩ i) < a (Fin.add ⟨7, by decide⟩ j)) ∧
    (∀ i : Fin 15, a i ∈ Finset.finRange 15) ∧
    (Finset.card (Finset.image a (Finset.univ : Finset (Fin 15))) = 15)
    ↔ (∃ s : Finset (Fin 14), s.card = 6) ∧ (choose 14 6 = 3003) :=
by
  sorry

end permutations_count_l162_162374


namespace bottle_caps_given_l162_162039

theorem bottle_caps_given (initial_caps final_caps caps_given : ℕ) :
  initial_caps = 8 →
  final_caps = 93 →
  caps_given = final_caps - initial_caps →
  caps_given = 85 :=
by
  intros h_initial h_final h_calculation
  rw [h_initial, h_final, h_calculation]
  rfl

end bottle_caps_given_l162_162039


namespace weekly_milk_production_l162_162930

theorem weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) (days_in_week : ℕ) 
  (h_num_cows : num_cows = 52) (h_milk_per_cow_per_day : milk_per_cow_per_day = 1000) 
  (h_days_in_week : days_in_week = 7) :
  num_cows * milk_per_cow_per_day * days_in_week = 364000 :=
by
  rw [h_num_cows, h_milk_per_cow_per_day, h_days_in_week]
  norm_num
  sorry

end weekly_milk_production_l162_162930


namespace num_zeros_g_l162_162679

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 2 then m * (x - 2) / x
  else if 0 < x ∧ x ≤ 2 then 3 * x - x^2
  else 0

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x m - 2

-- Statement to prove
theorem num_zeros_g (m : ℝ) : ∃ n : ℕ, (n = 4 ∨ n = 6) :=
sorry

end num_zeros_g_l162_162679


namespace johns_average_speed_l162_162363

theorem johns_average_speed 
  (cycling_time_min : ℕ := 45) 
  (cycling_speed_mph : ℕ := 12) 
  (jogging_time_hr : ℕ := 2) 
  (jogging_speed_mph : ℕ := 6) :
  (cycling_time_min / 60 + jogging_time_hr) > 0 → 
  let total_time_hr : ℝ := cycling_time_min / 60 + jogging_time_hr,
      total_distance_mi : ℝ := (cycling_speed_mph * (cycling_time_min / 60)) + (jogging_speed_mph * jogging_time_hr),
      average_speed : ℝ := total_distance_mi / total_time_hr
  in average_speed = 8 :=
sorry

end johns_average_speed_l162_162363


namespace exists_skew_lines_parallel_to_same_plane_l162_162182

theorem exists_skew_lines_parallel_to_same_plane :
  ∃ (l1 l2 : ℝ^3 → ℝ^3) (P : AffinePlane ℝ^3), 
  (l1 ∥ P) ∧ (l2 ∥ P) ∧ skew l1 l2 :=
by
  sorry

end exists_skew_lines_parallel_to_same_plane_l162_162182


namespace mod_inverse_3_40_l162_162986

theorem mod_inverse_3_40 : 3 * 27 % 40 = 1 := by
  sorry

end mod_inverse_3_40_l162_162986


namespace median_mode_correct_l162_162922

noncomputable def median (scores : List ℕ) : ℕ :=
  let sorted_scores := scores.sorted
  if sorted_scores.length % 2 = 0 then 
    (sorted_scores.get! (sorted_scores.length / 2 - 1) + sorted_scores.get! (sorted_scores.length / 2)) / 2 
  else 
    sorted_scores.get! (sorted_scores.length / 2)

noncomputable def mode (scores : List ℕ) : ℕ :=
  scores.foldr (
    λ s acc, 
      if scores.count s > scores.count acc then s else acc
  ) 0

theorem median_mode_correct (students : List ℕ) :
  students = [85, 88, 88, 88, 88, 88, 88, 88,
             90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
             93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
             94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94,
             97, 97, 97, 97, 97, 97, 97,
             99] →
  (median students = 93 ∧ mode students = 94) :=
by
  intro h
  have : students.sorted = [85, 88, 88, 88, 88, 88, 88, 88,
                            90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                            93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
                            94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94,
                            97, 97, 97, 97, 97, 97, 97,
                            99] := sorry
  have med : median students = 93 := sorry
  have mo : mode students = 94 := sorry
  exact ⟨med, mo⟩

end median_mode_correct_l162_162922


namespace work_done_per_day_l162_162566

/--
A can finish the work in 10 days.
B can do the same work in half the time taken by A.
C can finish the same work in 15 days.
B takes a break every third day and does not work, whereas A and C work continuously without any breaks.
What part of the same work can A, B, and C finish in a day on average if they all start working together?
-/
theorem work_done_per_day :
  let A_working_days := 10,
      B_working_days := 5,
      C_working_days := 15,
      total_days := 3 in
  let A_daily_work := (1 : ℚ) / A_working_days,
      B_daily_work := (1 : ℚ) / B_working_days,
      C_daily_work := (1 : ℚ) / C_working_days in
  (((2 * (A_daily_work + B_daily_work + C_daily_work)) + (A_daily_work + C_daily_work)) / total_days) = 3 / 10 := by
  -- proof goes here
  sorry

end work_done_per_day_l162_162566


namespace largest_circle_radius_l162_162938

-- Define the chessboard configuration
def chessboard := fin 8 × fin 8
def is_white (pos : chessboard) : Prop :=
  (pos.1.val + pos.2.val) % 2 = 0

-- Define the side length of each square on the chessboard
def side_length : ℝ := 1

-- Define the radius we need to prove
def largest_radius : ℝ := (Real.sqrt 10) / 2

-- Define the statement to be proved
theorem largest_circle_radius :
  ∃ (r : ℝ), r = largest_radius ∧
    (∀ (c : chessboard), is_white c → 
      ∀ (θ : ℝ), θ ∈ Icc 0 (2 * Real.pi) → 
        let x := c.1.val * side_length + side_length / 2 + r * Real.cos θ
        let y := c.2.val * side_length + side_length / 2 + r * Real.sin θ
        ∃ (c' : chessboard), is_white c' ∧
          (c'.1.val * side_length ≤ x ∧ x ≤ (c'.1.val + 1) * side_length) ∧
          (c'.2.val * side_length ≤ y ∧ y ≤ (c'.2.val + 1) * side_length)) :=
sorry

end largest_circle_radius_l162_162938


namespace hypotenuse_is_18_point_8_l162_162947

def hypotenuse_of_right_triangle (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2) * a * b = 24 ∧ a^2 + b^2 = c^2

theorem hypotenuse_is_18_point_8 (a b c : ℝ) (h : hypotenuse_of_right_triangle a b c) : c = 18.8 :=
  sorry

end hypotenuse_is_18_point_8_l162_162947


namespace average_marks_of_a_b_c_d_l162_162846

theorem average_marks_of_a_b_c_d (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : A = 43)
  (h3 : (B + C + D + E) / 4 = 48)
  (h4 : E = D + 3) :
  (A + B + C + D) / 4 = 47 :=
by
  -- This theorem will be justified
  admit

end average_marks_of_a_b_c_d_l162_162846


namespace min_value_S_l162_162698

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) (h2 : (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) > 200) :
  ∃ a b c : ℤ, a + b + c = 2 ∧ (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) = 256 :=
sorry

end min_value_S_l162_162698


namespace squares_in_50th_ring_l162_162195

noncomputable def number_of_squares_in_nth_ring (n : ℕ) : ℕ :=
  8 * n + 6

theorem squares_in_50th_ring : number_of_squares_in_nth_ring 50 = 406 := 
  by
  sorry

end squares_in_50th_ring_l162_162195


namespace least_prime_factor_of_expression_l162_162898

theorem least_prime_factor_of_expression :
  Nat.minFactor (9^4 - 9^3) = 2 :=
by
  sorry

end least_prime_factor_of_expression_l162_162898


namespace time_to_cover_length_l162_162185

-- Define the conditions
def speed_escalator : ℝ := 12
def length_escalator : ℝ := 150
def speed_person : ℝ := 3

-- State the theorem to be proved
theorem time_to_cover_length : (length_escalator / (speed_escalator + speed_person)) = 10 := by
  sorry

end time_to_cover_length_l162_162185


namespace angle_B_value_area_triangle_l162_162676

variable {α : Type*}

theorem angle_B_value 
  (a b c : ℝ) 
  (h1 : b^2 = c^2 + a^2 - real.sqrt 2 * a * c) 
  : real.arccos ((c^2 + a^2 - b^2) / (2 * a * c)) = real.pi / 4 := 
sorry

theorem area_triangle 
  (a : ℝ) (A : ℝ) (b c : ℝ) 
  (h2 : a = real.sqrt 2) 
  (h3 : real.cos A = 4 / 5) 
  (h4 : real.sin A = 3 / 5) 
  (h5 : b = (5 / 3))
  : 1 / 2 * a * b * (3 / 5 * real.sqrt 2 / 2 + 4 / 5 * real.sqrt 2 / 2) = 7 / 6 :=
sorry

end angle_B_value_area_triangle_l162_162676


namespace smallest_positive_period_monotonic_intervals_l162_162294

-- Define the function f(x)
def f (x : ℝ) : ℝ := (sin x + sqrt 3 * cos x) * cos (π / 2 - x) / tan x

-- Prove that the smallest positive period of f(x) is π
theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ ∀ T' > 0, T' < T → ∃ x, f(x + T') ≠ f(x) := sorry

-- Prove the monotonic intervals of f(x) in (0, π/2)
theorem monotonic_intervals :
  ∀ x : ℝ, 0 < x ∧ x < π / 2 →
    ((0 < x ∧ x < π / 12) → strictly_increasing_on f (0, π / 12)) ∧
    ((π / 12 < x ∧ x < π / 2) → strictly_decreasing_on f (π / 12, π / 2)) := sorry

end smallest_positive_period_monotonic_intervals_l162_162294


namespace ratio_students_sent_home_to_remaining_l162_162948

theorem ratio_students_sent_home_to_remaining (total_students : ℕ) (students_taken_to_beach : ℕ)
    (students_still_in_school : ℕ) (students_sent_home : ℕ) 
    (h1 : total_students = 1000) (h2 : students_taken_to_beach = total_students / 2)
    (h3 : students_still_in_school = 250) 
    (h4 : students_sent_home = total_students / 2 - students_still_in_school) :
    (students_sent_home / students_still_in_school) = 1 := 
by
    sorry

end ratio_students_sent_home_to_remaining_l162_162948


namespace original_meal_cost_l162_162188

theorem original_meal_cost:
  -- Let y be the original cost of the meal before tax and tip.
  let y := (33.60 / 1.26) in
  -- Tax rate is 8%
  let t := 0.08 in
  -- Tip rate is 18%
  let p := 0.18 in
  -- Total cost after tax and tip is $33.60
  let C := 33.60 in
  -- Prove that the original cost of the meal y is approximately 27 dollars
  y ≈ 27 := sorry

end original_meal_cost_l162_162188


namespace grid_converges_to_ones_l162_162744

theorem grid_converges_to_ones (grid : matrix (fin 3) (fin 3) ℤ) (h : ∀ i j, grid i j = 1 ∨ grid i j = -1) :
  ∃ n, (∀ i j, (iterate_grid_update n grid) i j = 1) := sorry

end grid_converges_to_ones_l162_162744


namespace fixed_point_of_f_l162_162691

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 1

theorem fixed_point_of_f (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : f a 1 = 2 := by
  have h : f a 1 = a^(1 - 1) + 1 := rfl
  rw [h]
  rw [pow_zero]
  simp
  dsimp
  norm_num

end fixed_point_of_f_l162_162691


namespace children_play_time_l162_162211

theorem children_play_time (total_children : ℕ) (pairs_playing : ℕ) (hours : ℕ) (total_play_time : ℕ) (equal_play_time : ℕ) : 
  total_children = 8 → pairs_playing = 2 → hours = 2 → total_play_time = pairs_playing * (hours * 60) → 
  equal_play_time = total_play_time / total_children → 
  equal_play_time = 30 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  simp at h4
  rw h4 at h5
  simp at h5
  exact h5

end children_play_time_l162_162211


namespace problem_1_problem_2_problem_3_l162_162690

noncomputable def f (x : ℝ) : ℝ := (√3 * sin (2 * x) + cos (2 * x) + 1) / (2 * cos x)

-- Prove that f(0) = 1
theorem problem_1 : f 0 = 1 := 
by 
  sorry

-- Prove that the domain of f is {x | x ≠ (π/2) + kπ, k ∈ ℤ}
theorem problem_2 : ∀ x : ℝ, (∃ k : ℤ, x = (π / 2) + k * π) ↔ (2 * cos x = 0) := 
by 
  sorry

-- Prove that the range of f on (0, π/2) is (1, 2]
theorem problem_3 : ∀ y : ℝ, (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ y = f x) ↔ 1 < y ∧ y ≤ 2 := 
by 
  sorry

end problem_1_problem_2_problem_3_l162_162690


namespace exponentiation_example_l162_162471

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_162471


namespace min_perimeter_of_polygon_with_vertices_roots_of_Q_l162_162792

open Complex

-- Define the polynomial Q(z)
def Q (z : ℂ) : ℂ := z ^ 8 - (8 * real.sqrt 2 - 10) * z ^ 4 + (8 * real.sqrt 2 - 11)

-- Define the roots explicitly
def roots : list ℂ := [1, -1, Complex.I, -Complex.I, 
                        (1 + real.sqrt 3) / 2 * (1 + Complex.I), 
                        -(1 + real.sqrt 3) / 2 * (1 + Complex.I), 
                        (1 + real.sqrt 3) / 2 * (1 - Complex.I), 
                        -(1 + real.sqrt 3) / 2 * (1 - Complex.I) ]

-- Theorem about the minimum perimeter
theorem min_perimeter_of_polygon_with_vertices_roots_of_Q :
  polygon_perimeter roots = 8 * real.sqrt 2 :=
sorry

end min_perimeter_of_polygon_with_vertices_roots_of_Q_l162_162792


namespace cost_of_carrots_and_cauliflower_l162_162649

variable {p c f o : ℝ}

theorem cost_of_carrots_and_cauliflower
  (h1 : p + c + f + o = 30)
  (h2 : o = 3 * p)
  (h3 : f = p + c) : 
  c + f = 14 := 
by
  sorry

end cost_of_carrots_and_cauliflower_l162_162649


namespace min_value_of_quadratic_l162_162530

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 12 * x + 35

theorem min_value_of_quadratic :
  ∀ x : ℝ, quadratic_function x ≥ quadratic_function 6 :=
by sorry

end min_value_of_quadratic_l162_162530


namespace sufficient_not_necessary_condition_l162_162246

noncomputable def sufficient_but_not_necessary (x y : ℝ) : Prop :=
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ (x + y > 2 → ¬(x > 1 ∧ y > 1))

theorem sufficient_not_necessary_condition (x y : ℝ) :
  sufficient_but_not_necessary x y :=
sorry

end sufficient_not_necessary_condition_l162_162246


namespace count_whole_numbers_between_sqrt18_and_sqrt200_l162_162315

theorem count_whole_numbers_between_sqrt18_and_sqrt200 :
  ∃ n : ℕ, n = 10 ∧
    ∀ x : ℤ, (5 ≤ x ∧ x ≤ 14) ↔ (⌊real.sqrt 18⌋ < x ∧ x < ⌈real.sqrt 200⌉) :=
by
  sorry

end count_whole_numbers_between_sqrt18_and_sqrt200_l162_162315


namespace tangent_line_at_point_l162_162856

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 3
def point : ℝ × ℝ := (1, 3)

theorem tangent_line_at_point :
  let slope := deriv curve 1
  slope = 2 →
  tangent_line (1, 3) slope = "y = 2 * x + 1" :=
by 
  sorry

end tangent_line_at_point_l162_162856


namespace length_of_PG_l162_162417

theorem length_of_PG (EF P G : ℝ) (h₂ : EF = 10) :
  PG = 5 * real.sqrt 2 :=
sorry

end length_of_PG_l162_162417


namespace photos_on_drive_l162_162168

def total_storage_space (num_photos : ℕ) (size_per_photo : ℕ) : ℕ :=
  num_photos * size_per_photo

def total_video_space (num_videos : ℕ) (size_per_video : ℕ) : ℕ :=
  num_videos * size_per_video

def space_left_for_photos (total_space : ℕ) (video_space : ℕ) : ℕ :=
  total_space - video_space

def number_of_photos (available_space : ℕ) (size_per_photo : ℕ) : ℕ :=
  available_space / size_per_photo

theorem photos_on_drive :
  let storage_space := total_storage_space 2000 15 / 10 in
  let video_space := total_video_space 12 200 in
  let remaining_space := space_left_for_photos storage_space video_space in
  number_of_photos remaining_space 15 / 10 = 400 :=
by 
  /*proof steps skipped*/
  sorry

end photos_on_drive_l162_162168


namespace cos_double_angle_l162_162267

theorem cos_double_angle (α β : Real) 
    (h1 : Real.sin α = Real.cos β) 
    (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1 / 2) :
    Real.cos (2 * β) = 2 / 3 :=
by
  sorry

end cos_double_angle_l162_162267


namespace syllogism_model_correct_order_l162_162909

-- Define the conditions
def is_trig_function (f: ℝ → ℝ) : Prop := ∃ a, ∀ x, f x = sin (x + a)
def is_periodic (f: ℝ → ℝ) : Prop := ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x

-- Define the statements
def statement1 : Prop := is_trig_function (sin)
def statement2 : Prop := ∀ f, is_trig_function f → is_periodic f
def statement3 : Prop := is_periodic (sin)

-- Define the syllogism model verification
theorem syllogism_model_correct_order :
  (statement2 ∧ statement1) → statement3 :=
by
  sorry

end syllogism_model_correct_order_l162_162909


namespace projection_matrix_unique_pair_l162_162430

theorem projection_matrix_unique_pair (a b : ℚ) :
  let P := Matrix.mk 2 2 (λ i j, if i = 0 then (if j = 0 then a else 1/3) else (if j = 0 then b else 2/3)) in
  P.mul P = P →
  (a, b) = (1/3, 2/3) :=
by
  sorry

end projection_matrix_unique_pair_l162_162430


namespace integral_f_eq_5_over_6_plus_3_l162_162244

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + 3 else -x

theorem integral_f_eq_5_over_6_plus_3 :
  ∫ x in -1..1, f x = (5 / 6 : ℝ) + 3 :=
sorry

end integral_f_eq_5_over_6_plus_3_l162_162244


namespace truck_initial_gas_ratio_l162_162113

-- Definitions and conditions
def truck_total_capacity : ℕ := 20

def car_total_capacity : ℕ := 12

def car_initial_gas : ℕ := car_total_capacity / 3

def added_gas : ℕ := 18

-- Goal: The ratio of the gas in the truck's tank to its total capacity before she fills it up is 1:2
theorem truck_initial_gas_ratio :
  ∃ T : ℕ, (T + car_initial_gas + added_gas = truck_total_capacity + car_total_capacity) ∧ (T : ℚ) / truck_total_capacity = 1 / 2 :=
by
  sorry

end truck_initial_gas_ratio_l162_162113


namespace sin_cos_sum_l162_162317

variable (θ : Real) (a : Real)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.sin (2 * θ) = a)

theorem sin_cos_sum (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.sin (2 * θ) = a) : 
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by
  sorry

end sin_cos_sum_l162_162317


namespace compute_PQ_length_l162_162030

open EuclideanGeometry

/-- Circumcircle of triangle ABC. -/
noncomputable def circumcircle (A B C D P Q : Point) : Prop := 
  ∃ (Γ : Circle), 
  is_circumcircle_of_triangle Γ A B C ∧ 
  on_circle D Γ ∧ 
  on_circle P Γ ∧ 
  on_circle Q Γ

/-- Given point D on segment AB such that CD bisects ∠ACB. -/
noncomputable def bisect_angle (A B C D : Point) : Prop :=
  lies_on_segment D A B ∧ 
  bisects_angle D A C B        -- CD bisects ∠ACB

/-- Given points P and Q on Γ such that PQ passes through D and is perp. to CD -/
noncomputable def PQ_condition (A B C D P Q : Point) (Γ : Circle) : Prop :=
  on_circle D Γ ∧ 
  on_circle P Γ ∧ 
  on_circle Q Γ ∧
  line_passes_through P Q D ∧      -- PQ passes through D
  line_perpendicular P Q (C, D)    -- PQ ⊥ CD

theorem compute_PQ_length
  (A B C D P Q : Point)
  (a b c : ℝ) 
  [length A B = c] 
  [length B C = a] 
  [length C A = b]
  (Γ : Circle)
  (H1 : circumcircle A B C D P Q)
  (H2 : bisect_angle A B C D)
  (H3 : PQ_condition A B C D P Q Γ) :
  length P Q = 4 * Real.sqrt 745 :=
begin
  sorry
end

end compute_PQ_length_l162_162030


namespace exponentiation_rule_example_l162_162505

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l162_162505


namespace trig_identity_l162_162912

variable {R A B C : ℝ}
variable {a b c : ℝ}

theorem trig_identity 
  (h1 : a = 2 * R * sin A)
  (h2 : b = 2 * R * sin B)
  (h3 : c = 2 * R * sin C) :
  a * cos A + b * cos B + c * cos C = 4 * R * sin A * sin B * sin C :=
by
  sorry

end trig_identity_l162_162912


namespace probability_two_red_one_green_l162_162879

theorem probability_two_red_one_green :
  let total_shoes := 6 + 8 + 5 + 3
  let red_shoes := 6 
  let green_shoes := 8
  let blue_shoes := 5 
  let yellow_shoes := 3
  ∃ (prob : ℚ), 
    prob = ( 15 * 8 : ℚ ) / 1540 ∧ 
    prob = 6 / 77 := by
  let total_shoes := 22
  let red_shoes := 6 
  let green_shoes := 8
  let blue_shoes := 5 
  let yellow_shoes := 3

  let comb := λ n k : ℕ, Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

  let total_ways_draw_3 := comb total_shoes 3
  let ways_draw_2_red := comb red_shoes 2
  let ways_draw_1_green := comb green_shoes 1

  let favorable_ways := ways_draw_2_red * ways_draw_1_green
  let prob := (favorable_ways : ℚ) / total_ways_draw_3

  have : prob = (15 * 8 : ℚ) / 1540 := by sorry
  have : prob = 6 / 77 := by sorry
  exact ⟨prob, this, this⟩

end probability_two_red_one_green_l162_162879


namespace part_a_part_b_l162_162373

-- Definition of k-typical
def k_typical (k m : ℕ) : Prop :=
  ∀ d, d ∣ m → d % k = 1

-- Part (a): "If the number of all divisors of n is k-typical, then n is the k-th power of an integer"
theorem part_a (k n : ℕ) (h : k > 0) :
  k_typical k (nat.divisors n).card → ∃ m : ℕ, n = m ^ k :=
by sorry

-- Part (b): "If k > 2, the converse of the assertion (a) is not true"
theorem part_b (k : ℕ) (h : k > 2) :
  ¬ (∀ n m : ℕ, n = m ^ k → k_typical k (nat.divisors n).card) :=
by sorry

end part_a_part_b_l162_162373


namespace option_C_correct_l162_162318

variable (a b : ℝ)

theorem option_C_correct (h : a > b) : -15 * a < -15 * b := 
  sorry

end option_C_correct_l162_162318


namespace units_digit_of_33_pow_33_mul_22_pow_22_l162_162605

theorem units_digit_of_33_pow_33_mul_22_pow_22 :
  (33 ^ (33 * (22 ^ 22))) % 10 = 1 :=
sorry

end units_digit_of_33_pow_33_mul_22_pow_22_l162_162605


namespace water_usage_l162_162939

def fee (x : ℕ) : ℕ :=
  if x ≤ 8 then 2 * x else 4 * x - 16

theorem water_usage (h : fee 9 = 20) : fee 9 = 20 := by
  sorry

end water_usage_l162_162939


namespace weekly_milk_production_l162_162929

theorem weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) (days_in_week : ℕ) 
  (h_num_cows : num_cows = 52) (h_milk_per_cow_per_day : milk_per_cow_per_day = 1000) 
  (h_days_in_week : days_in_week = 7) :
  num_cows * milk_per_cow_per_day * days_in_week = 364000 :=
by
  rw [h_num_cows, h_milk_per_cow_per_day, h_days_in_week]
  norm_num
  sorry

end weekly_milk_production_l162_162929


namespace largest_possible_percent_error_l162_162398

open Real

theorem largest_possible_percent_error :
  let length := 15
  let width := 10
  let length_error := 0.1
  let width_error := 0.1
  let min_length := length * (1 - length_error)
  let max_length := length * (1 + length_error)
  let min_width := width * (1 - width_error)
  let max_width := width * (1 + width_error)
  let actual_area := length * width
  let min_area := min_length * min_width
  let max_area := max_length * max_width
  let percent_error (computed_area : ℝ) : ℝ := ((computed_area - actual_area) / actual_area) * 100
  max (percent_error min_area) (percent_error max_area) = 21 :=
by
  sorry

end largest_possible_percent_error_l162_162398


namespace area_of_shaded_region_l162_162174

-- Definitions of the conditions
def length_cm : ℝ := 15
def width_cm : ℝ := 10
def radius_cm : ℝ := 5
def area_rectangle : ℝ := length_cm * width_cm
def area_circle : ℝ := Real.pi * radius_cm^2

-- Proof statement
theorem area_of_shaded_region :
  (area_rectangle - area_circle) = 150 - 25 * Real.pi :=
by 
  -- Proof logic would go here
  sorry

end area_of_shaded_region_l162_162174


namespace count_8_digit_odd_last_l162_162707

-- Define the constraints for the digits of the 8-digit number
def first_digit_choices := 9
def next_six_digits_choices := 10 ^ 6
def last_digit_choices := 5

-- State the theorem based on the given conditions and the solution
theorem count_8_digit_odd_last : first_digit_choices * next_six_digits_choices * last_digit_choices = 45000000 :=
by
  sorry

end count_8_digit_odd_last_l162_162707


namespace smallest_d_for_inequality_l162_162973

open Real

theorem smallest_d_for_inequality :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + 1 * |x^2 - y^2| ≥ exp ((x + y) / 2)) ∧
  (∀ d > 0, (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + d * |x^2 - y^2| ≥ exp ((x + y) / 2)) → d ≥ 1) :=
by
  sorry

end smallest_d_for_inequality_l162_162973


namespace lana_spent_l162_162774

def ticket_cost : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem lana_spent :
  ticket_cost * (tickets_for_friends + extra_tickets) = 60 := 
by
  sorry

end lana_spent_l162_162774


namespace _l162_162394

variables (m n : ℝ) [non_zero_vector m] [non_zero_vector n]

noncomputable theorem sufficient_but_not_necessary_condition
  (λ : ℝ) (hλ : λ < 0)
  (h : m = λ * n) : m • n < 0 :=
by {
  sorry
}

end _l162_162394


namespace solve_consecutive_even_sum_l162_162722

theorem solve_consecutive_even_sum :
  ∃ (x : ℤ), (∑ i in finset.range 5, (x + 2 * i - 4)) = 100 ∧ (x + 4) = 24 := by
{
  sorry
}

end solve_consecutive_even_sum_l162_162722


namespace inverse_prop_l162_162865

theorem inverse_prop (a c : ℝ) : (∀ (a : ℝ), a > 0 → a * c^2 ≥ 0) → (∀ (x : ℝ), x * c^2 ≥ 0 → x > 0) :=
by
  sorry

end inverse_prop_l162_162865


namespace find_a_minus_b_l162_162355

-- Define the periodic properties of the table
def table_period := 9
def element_position (n : ℕ) := n % table_period
def position_in_table (element : ℕ) (row : ℕ) (col : ℕ) := element = table_period * row + col

-- The third row's second column contains 8
def element_3_2 := position_in_table 8 3 2

-- Locate the positions of 2017 in the table
def cycles_2017 := (2017 / table_period).to_nat
def row_2017 := cycles_2017 * 3 + 1
def col_2017 := 1

-- The proof problem statement: the difference between a and b is 672
theorem find_a_minus_b : row_2017 - col_2017 = 672 := by
  sorry

end find_a_minus_b_l162_162355


namespace sam_initial_nickels_l162_162060

variable (n_now n_given n_initial : Nat)

theorem sam_initial_nickels (h_now : n_now = 63) (h_given : n_given = 39) (h_relation : n_now = n_initial + n_given) : n_initial = 24 :=
by
  sorry

end sam_initial_nickels_l162_162060


namespace min_sum_nonpos_l162_162785

theorem min_sum_nonpos (a b : ℤ) (h_nonpos_a : a ≤ 0) (h_nonpos_b : b ≤ 0) (h_prod : a * b = 144) : 
  a + b = -30 :=
sorry

end min_sum_nonpos_l162_162785


namespace power_of_powers_eval_powers_l162_162519

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l162_162519


namespace degree_measure_supplement_complement_l162_162897

theorem degree_measure_supplement_complement : 
  let alpha := 63 -- angle value
  let theta := 90 - alpha -- complement of the angle
  let phi := 180 - theta -- supplement of the complement
  phi = 153 := -- prove the final step
by
  sorry

end degree_measure_supplement_complement_l162_162897


namespace arccos_sqrt3_div_2_eq_pi_div_6_l162_162610

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l162_162610


namespace max_a_value_l162_162786

-- Variables representing the real numbers a, b, c, and d
variables (a b c d : ℝ)

-- Real number hypothesis conditions
-- 1. a + b + c + d = 10
-- 2. ab + ac + ad + bc + bd + cd = 20

theorem max_a_value
  (h1 : a + b + c + d = 10)
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  a ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_a_value_l162_162786


namespace deepak_present_age_l162_162093

variable (R D : ℕ)

theorem deepak_present_age 
  (h1 : R + 22 = 26) 
  (h2 : R / D = 4 / 3) : 
  D = 3 := 
sorry

end deepak_present_age_l162_162093


namespace solve_for_other_diagonal_l162_162424

noncomputable def length_of_other_diagonal
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem solve_for_other_diagonal 
  (h_area : ℝ) (h_d2 : ℝ) (h_condition : h_area = 75 ∧ h_d2 = 15) :
  length_of_other_diagonal h_area h_d2 = 10 :=
by
  -- using h_condition, prove the required theorem
  sorry

end solve_for_other_diagonal_l162_162424


namespace find_pq_product_l162_162032

theorem find_pq_product : 
  let p q : ℝ in 
  (∀ x : ℝ, 3 * x^2 - 2 * x - 8 = 0 ↔ (x = p ∨ x = q)) →
  (p + q = 2 / 3) →
  (p * q = -8 / 3) →
  (p-1)*(q-1) = -7 / 3 :=
by 
  intros p q h_roots h_sum h_product
  sorry

end find_pq_product_l162_162032


namespace pages_count_l162_162910

variables (P : ℝ)

-- conditions
def rate_printer_A := P / 60
def rate_printer_B := (P / 60) + 3
def combined_rate := P / 24

-- goal
theorem pages_count (h : rate_printer_A + rate_printer_B = combined_rate) : P = 360 :=
by sorry

end pages_count_l162_162910


namespace elizabeth_position_after_4_steps_l162_162880

theorem elizabeth_position_after_4_steps :
  ∀ (n : ℕ) (L : ℕ), L = 24 → n = 6 →
  let step_length := L / n in
  let y := 4 * step_length in
  y = 16 :=
by
  intros n L hL hn
  let step_length := L / n
  let y := 4 * step_length
  have h_step_length : step_length = 4, from sorry
  rw h_step_length
  exact sorry

end elizabeth_position_after_4_steps_l162_162880


namespace probability_obtuse_angle_is_correct_l162_162049

open Real

def pentagon_vertices : list (ℝ × ℝ) := [(0, 3), (5, 0), (2 * π + 2, 0), (2 * π + 2, 5), (0, 5)]

noncomputable def midpoint := (5 / 2, 3 / 2)

noncomputable def radius := sqrt ((5 / 2) ^ 2 + (3 / 2) ^ 2) / 2

noncomputable def semicircle_area := (1 / 2) * π * (radius ^ 2)

noncomputable def pentagon_area := (2 * π + 2) * 5 - (1 / 2) * 3 * 5

noncomputable def probability_obtuse_angle := semicircle_area / pentagon_area

theorem probability_obtuse_angle_is_correct :
  probability_obtuse_angle = 17 * π / 205 :=
by sorry

end probability_obtuse_angle_is_correct_l162_162049


namespace distance_symmetric_line_eq_l162_162284

noncomputable def distance_from_point_to_line : ℝ :=
  let x0 := 2
  let y0 := -1
  let A := 2
  let B := 3
  let C := 0
  (|A * x0 + B * y0 + C|) / (Real.sqrt (A^2 + B^2))

theorem distance_symmetric_line_eq : distance_from_point_to_line = 1 / (Real.sqrt 13) := by
  sorry

end distance_symmetric_line_eq_l162_162284


namespace maximum_area_triangle_ABC_l162_162441

noncomputable def triangle_max_area (AB BC CA : ℝ) : ℝ :=
  if 0 ≤ AB ∧ AB ≤ 1 ∧ 1 ≤ BC ∧ BC ≤ 2 ∧ 2 ≤ CA ∧ CA ≤ 3 then
    if (CA^2 = AB^2 + BC^2 ∨ AB^2 = BC^2 + CA^2 ∨ BC^2 = AB^2 + CA^2) then
      1 / 2 * AB * BC
    else
      -- Heron's formula or other method for non-right triangles (omitted here)
      sorry
  else
    0

theorem maximum_area_triangle_ABC :
  ∃ AB BC CA, 0 ≤ AB ∧ AB ≤ 1 ∧ 1 ≤ BC ∧ BC ≤ 2 ∧ 2 ≤ CA ∧ CA ≤ 3 ∧ triangle_max_area AB BC CA = 1 :=
begin
  use [1, 2, real.sqrt 5],
  split,
  -- condition 1: 0 ≤ AB ≤ 1
  split, linarith, linarith,
  -- condition 2: 1 ≤ BC ≤ 2
  split, linarith, linarith,
  -- condition 3: 2 ≤ CA ≤ 3
  split, linarith, linarith,
  -- show triangle_max_area 1 2 (sqrt 5) = 1
  have h : CA = real.sqrt 5 := by linarith,
  show triangle_max_area 1 2 (real.sqrt 5) = 1, by {
    unfold triangle_max_area,
    simp only [h, if_pos],
    rw [if_pos],
    -- Proving it using concrete values above
    norm_num,
    split, exact rfl, right, exact rfl,
  }
end

end maximum_area_triangle_ABC_l162_162441


namespace find_ellipse_eq_find_dot_product_find_acute_angle_l162_162260

-- Definitions based on the conditions
noncomputable def ellipseEq (a b : ℝ) (x y : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity_eq (a b : ℝ) := 
  Real.sqrt(1 - b^2 / a^2) = Real.sqrt(2) / 2

noncomputable def M_point := (0, 1)

noncomputable def left_vertex (a : ℝ) := (-Real.sqrt(a^2), 0)

axiom A_point_eq_a : ∀a b : ℝ, a > b > 0 → b = 1 → a^2 = 2 → b^2 = 1 → ellipseEq a b 1

axiom ob_dot_op_eq (B P : (ℝ × ℝ)) : ∀ (a b : ℝ) (x₀ y₀ : ℝ), b = 1 → a = Real.sqrt 2 → 
  x₀^2 + 2 * y₀^2 = 2 → B = (x₀, y₀) → P = (Real.sqrt 2, 2 * Real.sqrt 2 * y₀ / (x₀ + Real.sqrt 2)) →
  (x₀, y₀) ⬝ (Real.sqrt 2, 2 * Real.sqrt 2 * y₀ / (x₀ + Real.sqrt 2)) = 2

-- Main theorem statements
theorem find_ellipse_eq (a b : ℝ) (h1 : a > b > 0) (h2 : eccentricity_eq a b) (h3 : b = 1) : 
  ellipseEq a b := 
  sorry

theorem find_dot_product (a : ℝ) (B P : (ℝ × ℝ)) (h1 : a = Real.sqrt 2) : 
  ob_dot_op_eq B P := 
  sorry

theorem find_acute_angle (A B : (ℝ × ℝ)) (l : ℝ → ℝ) 
  (h1 : ∀x₁ x₂ y₁ y₂, (x₁ + x₂, y₁ + y₂) = (A, B) → |A - B| = 4/3) : 
  ∃ θ : ℝ, θ = Real.pi / 4 :=
  sorry

end find_ellipse_eq_find_dot_product_find_acute_angle_l162_162260


namespace correct_answer_l162_162309

def vector := (Int × Int)

-- Definitions of vectors given in conditions
def m : vector := (2, 1)
def n : vector := (0, -2)

def vec_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_scalar_mult (c : Int) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vec_dot (v1 v2 : vector) : Int :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition vector combined
def combined_vector := vec_add m (vec_scalar_mult 2 n)

-- The problem is to prove this:
theorem correct_answer : vec_dot (3, 2) combined_vector = 0 :=
  sorry

end correct_answer_l162_162309


namespace count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l162_162194

open Nat

def num180Unchanged : Nat := 
  let valid_pairs := [(0, 0), (1, 1), (8, 8), (6, 9), (9, 6)];
  let middle_digits := [0, 1, 8];
  (valid_pairs.length) * ((valid_pairs.length + 1) * (valid_pairs.length + 1) * middle_digits.length)

def num180UnchangedDivBy4 : Nat :=
  let valid_div4_pairs := [(0, 0), (1, 6), (6, 0), (6, 8), (8, 0), (8, 8), (9, 6)];
  let middle_digits := [0, 1, 8];
  valid_div4_pairs.length * (valid_div4_pairs.length / 5) * middle_digits.length

def sum180UnchangedNumbers : Nat :=
   1959460200 -- The sum by the given problem

theorem count_7_digit_nums_180_reversible : num180Unchanged = 300 :=
sorry

theorem count_7_digit_nums_180_reversible_divis_by_4 : num180UnchangedDivBy4 = 75 :=
sorry

theorem sum_of_7_digit_nums_180_reversible : sum180UnchangedNumbers = 1959460200 :=
sorry

end count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l162_162194


namespace probability_of_exactly_9_correct_is_zero_l162_162455
open ProbabilityTheory

/-- There are 10 English expressions and their translations into Russian. For each correctly matched pair, 1 point is given. Vasiliy chooses randomly. -/
def expressions_and_translations := list (ℕ × ℕ)

/-- Probability that Vasiliy Petrov will get exactly 9 correct pairs out of 10 by choosing randomly -/
theorem probability_of_exactly_9_correct_is_zero
  (et : expressions_and_translations)
  (h : length et = 10) :
  P (λ (s : list (ℕ × ℕ)), length (filter (λ p, p.1 = p.2) s) = 9) = 0 :=
sorry

end probability_of_exactly_9_correct_is_zero_l162_162455


namespace sin4_cos4_15_eq_neg_sqrt3_div_2_l162_162321

theorem sin4_cos4_15_eq_neg_sqrt3_div_2 :
  ∃ x : ℝ, x = 15 * (Real.pi / 180) ∧ (Real.sin x)^4 - (Real.cos x)^4 = - (Real.sqrt 3) / 2 :=
begin
  sorry
end

end sin4_cos4_15_eq_neg_sqrt3_div_2_l162_162321


namespace sandy_shirt_cost_l162_162831

def total_spent : ℝ := 33.56
def shorts_cost : ℝ := 13.99
def jacket_cost : ℝ := 7.43

theorem sandy_shirt_cost : total_spent - (shorts_cost + jacket_cost) = 12.14 := 
by 
  calc
    total_spent - (shorts_cost + jacket_cost) = 33.56 - (13.99 + 7.43) : by rfl
    ... = 33.56 - 21.42 : by rfl
    ... = 12.14 : by rfl

end sandy_shirt_cost_l162_162831


namespace tom_overall_profit_l162_162451

def initial_purchase_cost : ℝ := 20 * 3 + 30 * 5 + 15 * 10
def purchase_commission : ℝ := 0.02 * initial_purchase_cost
def total_initial_cost : ℝ := initial_purchase_cost + purchase_commission

def sale_revenue_before_commission : ℝ := 10 * 4 + 20 * 7 + 5 * 12
def sales_commission : ℝ := 0.02 * sale_revenue_before_commission
def total_sales_revenue : ℝ := sale_revenue_before_commission - sales_commission

def remaining_stock_a_value : ℝ := 10 * (3 * 2)
def remaining_stock_b_value : ℝ := 10 * (5 * 1.20)
def remaining_stock_c_value : ℝ := 10 * (10 * 0.90)
def total_remaining_value : ℝ := remaining_stock_a_value + remaining_stock_b_value + remaining_stock_c_value

def overall_profit_or_loss : ℝ := total_sales_revenue + total_remaining_value - total_initial_cost

theorem tom_overall_profit : overall_profit_or_loss = 78 := by
  sorry

end tom_overall_profit_l162_162451


namespace sugar_percentage_l162_162587

theorem sugar_percentage (x : ℝ) (h2 : 50 ≤ 100) (h1 : 1 / 4 * x + 12.5 = 20) : x = 10 :=
by
  sorry

end sugar_percentage_l162_162587


namespace maximum_take_home_income_l162_162337

noncomputable def tax_collected (x : ℝ) : ℝ := 10 * x * (x + 10)

noncomputable def income (x : ℝ) : ℝ := (x + 10) * 1000

noncomputable def take_home_pay (x : ℝ) : ℝ := income x - tax_collected x

theorem maximum_take_home_income : (∃ x : ℝ, take_home_pay x = 30250 - 10 * (x - 45)^2 ∧ x + 10 = 55) :=
begin
  sorry
end

end maximum_take_home_income_l162_162337


namespace vector_length_determined_by_angle_l162_162018

open Real

noncomputable def relaxed_version (a b : ℝ^3) (t : ℝ) 
  (angle θ : ℝ) : Prop := 
  (norm (a + t • b) = 1) → 
  ∃! θ, (∥a∥^2 * sin θ^2 = 1 ∧ θ = θ)

theorem vector_length_determined_by_angle
  (a b : ℝ^3) (h1 : a ≠ 0 ∧ b ≠ 0)
  (min_value_condition: ∀ t : ℝ, norm (a + t • b) ≥ 1)
  (θ : ℝ) 
  (angle_determined : ∃ θ, ∥a∥^2 * sin(θ)^2 = 1) :
  ∃! ∥a∥, angle θ = θ :=
begin
  sorry
end

end vector_length_determined_by_angle_l162_162018


namespace negation_proposition_equivalence_l162_162432

theorem negation_proposition_equivalence :
  (¬ ∃ x₀ : ℝ, (2 / x₀ + Real.log x₀ ≤ 0)) ↔ (∀ x : ℝ, 2 / x + Real.log x > 0) := 
sorry

end negation_proposition_equivalence_l162_162432


namespace tom_split_number_of_apples_l162_162882

theorem tom_split_number_of_apples
    (S : ℕ)
    (h1 : S = 8 * A)
    (h2 : A * 5 / 8 / 2 = 5) :
    A = 2 :=
by
  sorry

end tom_split_number_of_apples_l162_162882


namespace angle_E_l162_162824

def Quadrilateral (A B C D : Type) := Prop

variable {E F G H : Type}
variable [Quadrilateral E F G H]

noncomputable def angle_G : ℝ := 80

theorem angle_E (h_parallelogram : Quadrilateral E F G H)
  (h_angle_G : angle_G) : angle_G = 80 :=
sorry

end angle_E_l162_162824


namespace find_f1_l162_162562

noncomputable def f : ℤ → ℤ := sorry

axiom f_eq : ∀ x y : ℤ, f(x) + f(y) = f(x + 1) + f(y - 1)
axiom f_value1 : f 2016 = 6102
axiom f_value2 : f 6102 = 2016

theorem find_f1 : f 1 = 8117 := sorry

end find_f1_l162_162562
