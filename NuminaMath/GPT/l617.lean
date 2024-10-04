import Mathlib
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.ComplexBasic
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Floor
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Geometry.Triangle
import Mathlib.Algebra.Inequalities
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Differentiation.Basic
import Mathlib.Analysis.Fourier
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Real.Basic
import Mathlib.LinearAlgebra.Finrank
import Mathlib.MeasureTheory.Integral
import Mathlib.MeasureTheory.Probability.MassFunction
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.GCD.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Algebra.ContinuousFunctions
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.EuclideanSpace

namespace only_positive_integer_n_satisfying_condition_l617_617272

theorem only_positive_integer_n_satisfying_condition :
  ∀ (n : ℕ),
    (∀ (P : Polynomial ℤ),
      monic P ∧ P.degree ≤ n →
      ∃ (k : ℕ) (x_1 x_2 ... x_(k+1) : ℤ),
        0 < k ∧ k ≤ n ∧ 
        (∀ i j, i ≠ j → x_i ≠ x_j) ∧
        (P.eval x_1 + P.eval x_2 + ... + P.eval x_k = P.eval x_(k+1))) ↔
    n = 2 := sorry

end only_positive_integer_n_satisfying_condition_l617_617272


namespace circle_C_exists_tangent_lines_l617_617324

noncomputable def circle_equation (D E : ℝ) : Prop :=
  x^2 + y^2 + Dx + Ey + 3 = 0

noncomputable def symmetric_line (D E : ℝ) : Prop :=
  D^2 + E^2 = 20 ∧ D + E = -2

noncomputable def center_in_fourth_quadrant (D E : ℝ) : Prop :=
  (-D / 2) > 0 ∧ (-E / 2) < 0

noncomputable def radius (D E : ℝ) : Prop :=
  (D^2 + E^2 - 12) / 4 = 2

theorem circle_C :
  ∃ (D E : ℝ), circle_equation D E ∧ symmetric_line D E ∧ center_in_fourth_quadrant D E ∧ radius D E :=
sorry

noncomputable def tangent_lines : Prop :=
  ∃ a : ℝ, (a = -sqrt 10 / 2 ∨ a = sqrt 10 / 2) ∨
     ∃ k : ℝ, (k = (-2 + sqrt 6) / 2 ∨ k = (-2 - sqrt 6) / 2)

theorem exists_tangent_lines :
  ∃ l : ℝ → ℝ, tangent_lines l :=
sorry

end circle_C_exists_tangent_lines_l617_617324


namespace original_rent_eq_l617_617114

theorem original_rent_eq (R : ℝ)
  (h1 : 4 * 800 = 3200)
  (h2 : 4 * 850 = 3400)
  (h3 : 3400 - 3200 = 200)
  (h4 : 200 = 0.25 * R) : R = 800 := by
  sorry

end original_rent_eq_l617_617114


namespace probability_ant_at_C_l617_617625

noncomputable def ant_prob_at_C_after_7_moves : ℝ := 1 / 8

theorem probability_ant_at_C :
  let A := (0, 0)
  let C := (0, 2)
  let moves := 7
  let lattice := {p : ℤ × ℤ | true}
  let initial_position := A
  let reachable_probability (start : ℤ × ℤ) (end : ℤ × ℤ) (time : ℕ) : ℝ :=
    sorry -- Function that computes the probability based on random movement.
  reachable_probability A C moves = ant_prob_at_C_after_7_moves := sorry

end probability_ant_at_C_l617_617625


namespace kennedy_gas_mileage_l617_617872

theorem kennedy_gas_mileage :
  let miles_to_school := 20 in
  let miles_to_softball := 9 in
  let miles_to_library := 4 in
  let miles_to_coffee := 3 in
  let miles_to_burger := 5 in
  let miles_to_grocery := 7 in
  let miles_to_friend := 6 in
  let miles_to_home := 15 in
  let miles_per_gallon := 23 in
  let total_miles := miles_to_school + miles_to_softball + miles_to_library + miles_to_coffee + miles_to_burger + miles_to_grocery + miles_to_friend + miles_to_home in
  let gallons_of_gas := total_miles / miles_per_gallon in
  gallons_of_gas = 3 := 
by
  unfold total_miles gallons_of_gas
  sorry

end kennedy_gas_mileage_l617_617872


namespace question_1_question_2_minimum_value_l617_617352

open Real

def f (x : ℝ) : ℝ := abs (x - 10) - abs (x - 25)

theorem question_1 (a : ℝ) : (∀ x : ℝ, f x < 10 * a + 10) → a > 1 / 2 :=
by
  intro h
  sorry

theorem question_2 (a : ℝ) : a > 1 / 2 → 2 * a + 27 / (a^2) ≥ 9 :=
by
  intro ha
  sorry

theorem minimum_value (a : ℝ) : a = 3 → 2 * a + 27 / (a^2) = 9 :=
by
  intro ha_eq_3
  rw [ha_eq_3]
  calc
    2 * 3 + 27 / (3 ^ 2) = 6 + 3 := by norm_num
    ... = 9 := by norm_num
  sorry

end question_1_question_2_minimum_value_l617_617352


namespace determine_stone_weights_l617_617158

-- Statements for the problem conditions
axiom stones : ℕ → ℕ → ℕ → ℕ → Prop
axiom balance_scale : (ℕ × ℕ × ℕ × ℕ) → ℕ → Prop
axiom potential_error : ℕ → ℕ → Prop -- one of the weighings could have an error of ± 1 gram

-- Main statement of the problem
theorem determine_stone_weights :
  ∀ (a b c d : ℕ),
  stones a b c d →
  (∃ (x y z t : ℕ),
    balance_scale (x, y, z, t) 0 ∧
    (potential_error (a + b + c - d) x ∨
     potential_error (a + b - c + d) y ∨
     potential_error (a - b + c + d) z ∨
     potential_error (-a + b + c + d) t)) →
  (∃ (a' b' c' d' : ℕ),
    stones a' b' c' d' ∧ a = a' ∧ b = b' ∧ c = c' ∧ d = d').

end determine_stone_weights_l617_617158


namespace least_possible_diagonals_l617_617620

noncomputable def leastDiagonals : ℝ :=
  let n := 2021 in 2018

theorem least_possible_diagonals (labels : Fin 2021 → ℝ)
  (h1 : ∀ i, abs (labels i - labels ((i + 1) % 2021)) ≤ 1)
  (h2 : ∀ i j, i ≠ j → abs (labels i - labels j) ≤ 1 → is_diagonal i j) :
  leastDiagonals = 2018 :=
sorry

end least_possible_diagonals_l617_617620


namespace problem_statement_l617_617445

-- Given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x < f y

noncomputable def f (a b : ℝ) : ℝ → ℝ :=
λ x, log a (abs (a * x + b))

-- Main theorem to prove
theorem problem_statement (a b : ℝ)
  (h1 : is_even_function (f a b))
  (h2 : is_monotonically_increasing (f a b) (set.Ioi 0))
  (h3 : a > 1)
  (h4 : b = 0) :
  f a b (b - 2) < f a b (a + 1) :=
sorry

end problem_statement_l617_617445


namespace non_sophomore_instrument_count_l617_617833

def num_non_sophomores_playing_instrument (p q : ℕ) : ℕ :=
  (6 / 10) * q

theorem non_sophomore_instrument_count (p q : ℕ) (h1 : p + q = 400) (h2 : 0.5 * p + 0.4 * q = 176) : 
  num_non_sophomores_playing_instrument p q = 144 :=
sorry

end non_sophomore_instrument_count_l617_617833


namespace option_C_correct_l617_617568

theorem option_C_correct : (Real.sqrt 2) * (Real.sqrt 6) = 2 * (Real.sqrt 3) :=
by sorry

end option_C_correct_l617_617568


namespace fifteenth_odd_multiple_of_5_l617_617203

theorem fifteenth_odd_multiple_of_5 : ∃ (n : Nat), n = 15 → 10 * n - 5 = 145 :=
by
  intro n hn
  have h : 10 * 15 - 5 = 145 := by
    calc
      10 * 15 - 5 = 150 - 5 : by rw (Nat.mul_eq_mul_left n 10)
                ... = 145    : by rfl
  exact ⟨15, h⟩
  sorry

end fifteenth_odd_multiple_of_5_l617_617203


namespace Mehki_is_10_years_older_than_Jordyn_l617_617451

def Zrinka_age : Nat := 6
def Mehki_age : Nat := 22
def Jordyn_age : Nat := 2 * Zrinka_age

theorem Mehki_is_10_years_older_than_Jordyn : Mehki_age - Jordyn_age = 10 :=
by
  sorry

end Mehki_is_10_years_older_than_Jordyn_l617_617451


namespace sin_ordering_l617_617131

theorem sin_ordering :
  let rad1 := 57
  let rad2 := 114
  let rad3 := 171
  sin rad2 > sin rad1 ∧ sin rad1 > sin rad3 :=
by
  -- We approximate the radian values:
  let sin1 := sin 57
  let sin2 := sin 66
  let sin3 := sin 9
  
  -- We know the monotonicity of the sine function in (0, 90):
  have h1: sin 9 < sin 57, by sorry
  have h2: sin 57 < sin 66, by sorry
  
  -- Combining these, we get the desired order:
  exact ⟨h2, h1⟩

end sin_ordering_l617_617131


namespace product_of_three_numbers_l617_617144

theorem product_of_three_numbers :
  ∃ (x y z : ℚ), 
    (x + y + z = 30) ∧ 
    (x = 3 * (y + z)) ∧ 
    (y = 8 * z) ∧ 
    (x * y * z = 125) := 
by
  sorry

end product_of_three_numbers_l617_617144


namespace rate_per_sqm_l617_617124

theorem rate_per_sqm (length width : ℝ) (cost : ℝ) (Area : ℝ := length * width) (rate : ℝ := cost / Area) 
  (h_length : length = 5.5) (h_width : width = 3.75) (h_cost : cost = 8250) : 
  rate = 400 :=
sorry

end rate_per_sqm_l617_617124


namespace convert_base_9A3_16_to_4_l617_617705

theorem convert_base_9A3_16_to_4 :
  let h₁ := 9
  let h₂ := 10 -- A in hexadecimal
  let h₃ := 3
  let b₁ := 21 -- h₁ converted to base 4
  let b₂ := 22 -- h₂ converted to base 4
  let b₃ := 3  -- h₃ converted to base 4
  9 * 16^2 + 10 * 16^1 + 3 * 16^0 = 2 * 4^5 + 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 0 * 4^1 + 3 * 4^0 :=
by
  sorry

end convert_base_9A3_16_to_4_l617_617705


namespace Jill_arrives_9_minutes_later_l617_617036

theorem Jill_arrives_9_minutes_later
  (distance : ℝ)
  (Jack_speed : ℝ)
  (Jill_speed : ℝ)
  (h1 : distance = 1)
  (h2 : Jack_speed = 10)
  (h3 : Jill_speed = 4) :
  ((distance / Jill_speed) - (distance / Jack_speed)) * 60 = 9 := by
  -- Placeholder for the proof
  sorry

end Jill_arrives_9_minutes_later_l617_617036


namespace count_x0_values_for_x0_eq_x5_l617_617321

noncomputable def recurrence_relation (x : ℝ) : ℕ → ℝ
| 0       := x
| (n + 1) := if 2 * recurrence_relation x n < 1 then 2 * recurrence_relation x n else 2 * recurrence_relation x n - 1

theorem count_x0_values_for_x0_eq_x5 :
  ∃ n : ℕ, n = 5 → (∀ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ < 1 → recurrence_relation x₀ 5 = x₀) ∧ 
  (∀ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ < 1 → recurrence_relation x₀ 5 <> 1) → 31 :=
sorry

end count_x0_values_for_x0_eq_x5_l617_617321


namespace retirement_total_age_and_years_l617_617232

theorem retirement_total_age_and_years (birth_year hire_year retire_year birth_age : ℤ) 
(hire_year_eq : hire_year = birth_year + birth_age)
(retire_expr : retire_year = hire_year + 19) : 
(hire_year = 1987) ∧ (birth_age = 32) ∧ (retire_year = 2006) → 
(birth_age + (retire_year - hire_year) + (retire_year - hire_year) = 70) :=
by
  intros h
  obtain ⟨hire_year_eq', birth_age_eq, retire_year_eq'⟩ := h
  simp [hire_year_eq', birth_age_eq, retire_year_eq']
  exact (hire_year_eq' + (birth_year + birth_age_eq - hire_year_eq') + (birth_year + birth_age_eq - hire_year_eq') = 70)

sorry -- proof is omitted

end retirement_total_age_and_years_l617_617232


namespace entropy_selection_entropy_competition_l617_617943

-- Definitions for Question 1
def f (x : ℝ) (a : ℝ) : ℝ := -x * log x / log a

-- Definitions for Question 2
def H_competition (n : ℕ) : ℝ :=
  2 - 4 / 2^n

-- Theorems to be proved
theorem entropy_selection (a : ℝ) (h : f (1/2) a = 1/2) : 
  Σ p_k => 32 * (-p_k * log p_k / log a) = 5 :=
sorry

theorem entropy_competition (n : ℕ) (h : n > 1) : 
  Σ k : ℕ, (k ≤ n ∧ k ≠ 0) → 2 - 4 / 2^n :=
sorry

end entropy_selection_entropy_competition_l617_617943


namespace cos_angle_PST_is_1_4_l617_617842

open Real EuclideanGeometry

noncomputable def P := (0 : ℝ, 0 : ℝ)
noncomputable def Q := (5 : ℝ, 0 : ℝ)
noncomputable def R := (-5 : ℝ, 0 : ℝ)

def S := midpoint ℝ Q R
def T := midpoint ℝ R Q

theorem cos_angle_PST_is_1_4 : cos (angle P S T) = 1 / 4 := 
  sorry

end cos_angle_PST_is_1_4_l617_617842


namespace circles_intersect_l617_617766

-- Define the circles C1 and C2
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the centers and radii of the circles derived from their equations
def center1 := (0 : ℝ, 0 : ℝ)
def radius1 := 1

def center2 := (-2 : ℝ, 3 : ℝ)
def radius2 := 3

-- Define the positional relationship proof
theorem circles_intersect :
  let d := Real.sqrt ((-2 - 0)^2 + (3 - 0)^2) in
  radius1 + radius2 > d ∧ d > radius2 - radius1 :=
by
  have d := Real.sqrt (4 + 9)
  sorry

end circles_intersect_l617_617766


namespace Area_ABHC_PO_OH_l617_617767

-- Define the basic setup for the problem
variables {A B C H O P : Point}

-- Assuming an acute triangle ABC with given conditions
axiom Circumcircle_radius : real
axiom Angle_A : ℝ
axiom Orthocenter : Point
axiom Circumcenter : Point

-- Defining the conditions from the problem
def acute_triangle (A B C : Point) : Prop := 
  ∃ H O : Point, right_triangle H O ∧ circumradius A B C = 1 ∧ angle A B = 60

def intersection_point (A B C H O P : Point) : Prop :=
  acute_triangle A B C ∧ (H = Orthocenter) ∧ (O = Circumcenter) ∧
  line_segment O H ∩ extension A B = P

-- Stating the prove goals

theorem Area_ABHC (h : intersection_point A B C H O P) : 
  area (concave_quadrilateral A B H C) = sqrt 3 / 2 := 
sorry 

theorem PO_OH (h : intersection_point A B C H O P) : 
  (segment_length P O) * (segment_length O H) = 1 := 
sorry

end Area_ABHC_PO_OH_l617_617767


namespace fifteenth_odd_multiple_of_5_is_145_l617_617189

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l617_617189


namespace sum_of_number_and_its_radical_conjugate_l617_617664

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617664


namespace inequality_hold_l617_617466

theorem inequality_hold (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧ 
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ 1/8 :=
sorry

end inequality_hold_l617_617466


namespace largest_x_satisfying_inequality_l617_617276

theorem largest_x_satisfying_inequality :
  (∃ x : ℝ, 
    (∀ y : ℝ, |(y^2 - 4 * y - 39601)| ≥ |(y^2 + 4 * y - 39601)| → y ≤ x) ∧ 
    |(x^2 - 4 * x - 39601)| ≥ |(x^2 + 4 * x - 39601)|
  ) → x = 199 := 
sorry

end largest_x_satisfying_inequality_l617_617276


namespace range_of_a_l617_617074

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (2^x + 2^(-x)) / 2

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ set.Icc 1 2, a * f x + g (2 * x) ≥ 0) → 
  a ≥ -17 / 6 :=
sorry

end range_of_a_l617_617074


namespace find_number_l617_617309

-- Definitions and conditions
def unknown_number (x : ℝ) : Prop :=
  (14 / 100) * x = 98

-- Theorem to prove
theorem find_number (x : ℝ) : unknown_number x → x = 700 := by
  sorry

end find_number_l617_617309


namespace range_of_slope_angle_l617_617769

noncomputable def func (x : ℝ) (n : ℝ) : ℝ := (1 / 3) * x ^ 3 + n ^ 2 * x

theorem range_of_slope_angle (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m * n = (Real.sqrt 3) / 2) :
    let alpha := Real.atan (m^2 + n^2) in
    (Real.pi / 3) ≤ alpha ∧ alpha < (Real.pi / 2) :=
by
  sorry

end range_of_slope_angle_l617_617769


namespace find_angle_A_l617_617394

theorem find_angle_A (a b : ℝ) (B A : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = Real.pi / 4) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l617_617394


namespace problem1_problem2_l617_617791

variable {x a : ℝ}

def f (x : ℝ) : ℝ := |x - 2|

theorem problem1 : { x : ℝ | f(x) + f(x + 1) ≤ 2 } = { x | 0.5 ≤ x ∧ x ≤ 2.5 } :=
sorry

theorem problem2 (h : a < 0) : f(a * x) - a * f(x) ≥ f(2 * a) :=
sorry

end problem1_problem2_l617_617791


namespace fifteenth_odd_multiple_of_five_l617_617197

theorem fifteenth_odd_multiple_of_five :
  ∃ a : ℕ, (∀ n : ℕ, a n = 5 + (n - 1) * 10) ∧ a 15 = 145 :=
by
  let a := λ n, 5 + (n - 1) * 10
  use a
  split
  { intros n,
    refl }
  { refl }
  sorry

end fifteenth_odd_multiple_of_five_l617_617197


namespace flight_paths_equal_distance_l617_617093

theorem flight_paths_equal_distance
  (h₁ h₂ : ℕ) (w : ℕ) (x : ℝ)
  (H₁ : h₁ = 20) (H₂ : h₂ = 30) (W : w = 50) 
  (hx : x * (w - x) = 600) :
  (sqrt (x^2 + h₂^2) = 10 * sqrt 13) ∧ (sqrt ((w - x)^2 + h₁^2) = 10 * sqrt 13) := by
  sorry

end flight_paths_equal_distance_l617_617093


namespace smallest_possible_value_of_f25_l617_617596

noncomputable def adjective_function (f : ℤ → ℤ) :=
  ∀ m n : ℤ, f m + f n > max (m^2) (n^2)

noncomputable def minimizes_sum (f : ℤ → ℤ) :=
  (∀ g : ℤ → ℤ, adjective_function g → (∑ i in finset.range 30, f (i + 1)) ≤ (∑ i in finset.range 30, g (i + 1)))

theorem smallest_possible_value_of_f25 : 
  ∃ f : ℤ → ℤ, adjective_function f ∧ minimizes_sum f ∧ f 25 = 498 :=
sorry

end smallest_possible_value_of_f25_l617_617596


namespace hypotenuse_length_l617_617537

noncomputable def triangle_hypotenuse_length (AB AC : ℝ) : ℝ :=
let AX := (2 * AB) / 3 in
let XB := AB / 3 in
let AY := (2 * AC) / 3 in
let YC := AC / 3 in
let CX := 24 in
let BY := 18 in
Real.sqrt ((CX) ^ 2 + (AY) ^ 2)

theorem hypotenuse_length (AB AC : ℝ) (h₁ : AX = (2 * AB) / 3)
  (h₂ : XB = AB / 3) (h₃ : AY = (2 * AC) / 3) (h₄ : YC = AC / 3)
  (h₅ : BY = 18) (h₆ : CX = 24) :
  triangle_hypotenuse_length AB AC = 30 :=
sorry

end hypotenuse_length_l617_617537


namespace quadratic_has_unique_solution_l617_617944

theorem quadratic_has_unique_solution (a : ℚ) (h : a ≠ 0) (ha : a = 75 / 4) :
  ∃ x : ℚ, (a * x^2 + 30 * x + 12 = 0) ∧ x = -4 / 5 :=
by
  have discriminant_zero : 30^2 - 4*a*12 = 0 := by
    rw [ha]
    simp [sq]
  sorry

end quadratic_has_unique_solution_l617_617944


namespace intersection_M_N_l617_617082

def M : Set ℝ := { x | x^2 + x - 6 < 0 }
def N : Set ℝ := { x | |x - 1| ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } :=
sorry

end intersection_M_N_l617_617082


namespace midline_theorem_l617_617469

-- Define vectors and their properties
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C M N : V)

-- Define the midpoint property for M and N
def is_midpoint (M A B : V) : Prop := M = (A + B) / 2

theorem midline_theorem 
  (hM : is_midpoint M A B) 
  (hN : is_midpoint N A C) : 
  ∃ K : V, K = (B - C) / 2 ∧ K = M - N :=
begin
  -- Use the given conditions and setup the proof, which we skip here
  sorry
end

end midline_theorem_l617_617469


namespace determine_f_l617_617899

noncomputable def f (x : ℝ) : ℝ := sorry

theorem determine_f :
  (∀ x y, f (x + y) ≥ f x * f y - f (x * y) + 1) ∧
  (∀ x y, x ≤ y → f x ≤ f y) ∧
  (f 0 = 1) ∧
  (∀ x, continuous (λ x, f x)) →
  (∀ x, f x = x + 1) :=
begin
  intros h,
  sorry
end

end determine_f_l617_617899


namespace base4_to_base10_10201_l617_617267

def convert_base4_to_base10 (n : ℕ) : ℕ :=
  -- Assuming n represents the base-4 number interpreted as a base-10 integer
  match n with
  | 10201 := 1 * 4^4 + 0 * 4^3 + 2 * 4^2 + 0 * 4^1 + 1 * 4^0
  | _     := 0 -- Default case (should not happen in our context)

theorem base4_to_base10_10201 : convert_base4_to_base10 10201 = 289 :=
by
  unfold convert_base4_to_base10
  simp
  sorry

end base4_to_base10_10201_l617_617267


namespace part1_part2_l617_617121

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem part1 (a b : ℝ) (h1 : f a b 1 = 8) : a + b = 2 := by
  rw [f] at h1
  sorry

theorem part2 (a b : ℝ) (h1 : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  rw [f] at h1
  sorry

end part1_part2_l617_617121


namespace fibonacci_mod_10_periodic_l617_617171

def fibonacci : ℕ → ℕ 
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

theorem fibonacci_mod_10_periodic :
  ∃ p : ℕ, p = 60 ∧ ∀ n : ℕ, (fibonacci (n + p)) % 10 = (fibonacci n) % 10 :=
by
  sorry

end fibonacci_mod_10_periodic_l617_617171


namespace circumcenter_on_angle_bisector_l617_617626

open EuclideanGeometry

-- Define the problem as Lean 4 statement
theorem circumcenter_on_angle_bisector 
  (A O S B C : Point) (r : ℝ)
  (h_circle_incorner : inscribed_circle O r S)
  (h_symmetry : symmetric_point O A S)
  (h_tangents : tangent_to_circle A O B ∧ tangent_to_circle A O C)
  (h_intersections : intersects_side_far A B S ∧ intersects_side_far A C S) :
  lies_on_angle_bisector (circumcenter_triangle A B C) S :=
sorry

end circumcenter_on_angle_bisector_l617_617626


namespace movie_ticket_vs_popcorn_difference_l617_617604

variable (P : ℝ) -- cost of a bucket of popcorn
variable (d : ℝ) -- cost of a drink
variable (c : ℝ) -- cost of a candy
variable (t : ℝ) -- cost of a movie ticket

-- Given conditions
axiom h1 : t = 8
axiom h2 : d = P + 1
axiom h3 : c = (P + 1) / 2
axiom h4 : t + P + d + c = 22

-- Question rewritten: Prove that the difference between the normal cost of a movie ticket and the cost of a bucket of popcorn is 3.
theorem movie_ticket_vs_popcorn_difference : t - P = 3 :=
by
  sorry

end movie_ticket_vs_popcorn_difference_l617_617604


namespace intersection_M_N_eq_02_l617_617804

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_M_N_eq_02 : M ∩ N = {0, 2} := 
by sorry

end intersection_M_N_eq_02_l617_617804


namespace books_read_indeterminate_l617_617157

theorem books_read_indeterminate (movies : ℕ) (books : ℕ) (watched_movies : ℕ) (remaining_movies : ℕ) :
  movies = 8 ∧ books = 21 ∧ watched_movies = 4 ∧ remaining_movies = 4 → ∃ books_read : ℕ, true :=
begin
  intro h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  use 0, -- Number of books read is indeterminate, just to satisfy the syntax
  trivial,
end

end books_read_indeterminate_l617_617157


namespace find_value_of_B_l617_617748

theorem find_value_of_B (B : ℚ) (h : 4 * B + 4 = 33) : B = 29 / 4 :=
by
  sorry

end find_value_of_B_l617_617748


namespace longer_side_length_l617_617235

-- Define the problem conditions
def radius (r : ℝ) : Prop :=
  r = 5

def circle_area (r : ℝ) (A : ℝ) : Prop :=
  A = π * r^2

def rectangle_area (A_circle A_rect : ℝ) : Prop :=
  A_rect = 3 * A_circle

def shorter_side (r l : ℕ) : Prop :=
  l = 2 * r

-- Define the statement to be proved
theorem longer_side_length (r A_circle A_rect l longer_side : ℝ) (hr : radius r)
  (hA_circle : circle_area r A_circle) (hA_rect : rectangle_area A_circle A_rect)
  (h_shorter_side : shorter_side r l) :
  longer_side = A_rect / l :=
by sorry

end longer_side_length_l617_617235


namespace find_value_l617_617318

variable (a b c : Int)

-- Conditions from the problem
axiom abs_a_eq_two : |a| = 2
axiom b_eq_neg_seven : b = -7
axiom neg_c_eq_neg_five : -c = -5

-- Proof problem
theorem find_value : a^2 + (-b) + (-c) = 6 := by
  sorry

end find_value_l617_617318


namespace option_A_option_B_option_C_l617_617858

variable (A B C a b c : ℝ)

-- Assume that we have a triangle with sides a, b, and c opposite to angles A, B, and C respectively.
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : A > 0
axiom h5 : B > 0
axiom h6 : C > 0
axiom h7 : A + B + C = π

-- Prove the conditions from the solution
theorem option_A (hS : sin C = sin A * cos B + sin B * cos A) : c = a * cos B + b * cos A := 
  sorry

theorem option_B : 2 * sin^2 ((A + B) / 2) = 1 + cos C := 
  sorry

theorem option_C : a^2 - b^2 = c * (a * cos B - b * cos A) := 
  sorry

end option_A_option_B_option_C_l617_617858


namespace coefficients_not_divisible_by_65_l617_617377

theorem coefficients_not_divisible_by_65 :
  let n := 65
  let p := 5
  let q := 13
  ∑ k in Finset.range (n + 1), if p ∣ Nat.choose n k ∨ q ∣ Nat.choose n k then 0 else 1 = 16 :=
by
  sorry

end coefficients_not_divisible_by_65_l617_617377


namespace sum_of_conjugates_l617_617656

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617656


namespace trigonometric_solution_l617_617478

theorem trigonometric_solution (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, (n % 2 = 0 → (x = k * real.pi)) ∨ 
          (n % 2 = 1 → (x = 2 * k * real.pi ∨ x = 2 * k * real.pi - real.pi / 2)) 
:=
sorry

end trigonometric_solution_l617_617478


namespace lean_proof_l617_617864

noncomputable theory

variables {f : ℝ → ℝ}

-- Condition 1: ∀ x, f(π/2 - x) + f(x) = 0
axiom cond1 : ∀ x : ℝ, f(Real.pi / 2 - x) + f(x) = 0

-- Condition 2: ∀ x, f(π + x) = f(-x)
axiom cond2 : ∀ x : ℝ, f(Real.pi + x) = f(-x)

-- Condition 3: ∀ x ∈ [0, π/4], f(x) = cos(2 * x)
axiom cond3 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 4 → f(x) = Real.cos (2 * x)

-- The theorem to be proven
theorem lean_proof : f (79 * Real.pi / 24) = (Real.sqrt 2 - Real.sqrt 6) / 4 := 
sorry

end lean_proof_l617_617864


namespace problem_solution_l617_617891

noncomputable def externally_tangent_circles {C1 C2 : Circle} (P : Point) (A : Point) (M M' : Point) (N N' : Point) : Prop :=
  let P_tangent : P ∈ C1 ∧ P ∈ C2 := sorry
  let A_on_C2 : A ∈ C2 := sorry
  let tangents : Tangent A C1 = M × Tangent A C1 = M' := sorry
  let N_N'_intersections : (Line M N ∩ C2) \ {A} = {N} ∧ (Line M' N' ∩ C2) \ {A} = {N'} := sorry
  |P - N'| * |M - N| = |P - N| * |M' - N'|
  
theorem problem_solution
  (C1 C2 : Circle)
  (P A M M' N N' : Point)
  (P_tangent : P ∈ C1 ∧ P ∈ C2)
  (A_on_C2 : A ∈ C2)
  (tangents : (tangent A C1 = M) × (tangent A C1 = M'))
  (N_N'_intersections : (Line M N ∩ C2) \ {A} = {N} ∧ (Line M' N' ∩ C2) \ {A} = {N'})
  : |P - N'| * |M - N| = |P - N| * |M' - N'| :=
by
  sorry

end problem_solution_l617_617891


namespace sum_of_radical_conjugates_l617_617684

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617684


namespace find_ω_l617_617788

def f (ω x : ℝ) : ℝ := sin (ω * x) - sqrt 3 * cos (ω * x)

theorem find_ω (ω x1 x2 : ℝ) (hω : ω > 0) (h1 : f ω x1 = 2) (h2 : f ω x2 = 0) (h3 : abs (x1 - x2) = 3 * π) : 
  ω = 1 / 6 :=
sorry

end find_ω_l617_617788


namespace circle_radius_l617_617461

section
variables (A B O M : Type) [MetricSpace O] 
variables (OA OB : ℝ) (OM : ℝ) (r : ℝ) 
variable {circle : O → ℝ → Prop}

-- Points division condition
variable (O_divides : OA = 6 ∧ OB = 4)

-- Tangents condition
variable (Tangents_meet : ∃ M, circle O r ∧ Tangents_from_A_and_B O A B M)

-- Intersection at M with given OM
variable (OM_length : OM = 12)

theorem circle_radius :
  r = 6 * Real.sqrt 21 / 7 :=
sorry
end

end circle_radius_l617_617461


namespace least_possible_diagonals_l617_617615

theorem least_possible_diagonals :
  let n := 2021 in
  let labels := Fin n → ℝ in
  ∃ (d : ℕ), 
    (∀ (x : labels), 
      (∀ (i j : Fin n), 
        (i ≠ j ∧ i.succ ≠ j ∧ j.succ ≠ i ∧ abs (x i - x j) ≤ 1) → 
        (∃ (k l : Fin n), (k ≠ l ∧ abs (x k - x l) ≤ 1))) 
    → d ≥ 4039) 
  ∧ 
    d = 4039 :=
sorry

end least_possible_diagonals_l617_617615


namespace S6_div_a6_l617_617329

noncomputable def a_n : ℕ → ℕ
| 1       => 1
| (n + 1) => 2 * a_n n

noncomputable def S_n (n : ℕ) : ℕ :=
2 * a_n n - 1

theorem S6_div_a6 : S_n 6 / a_n 6 = 63 / 32 := by
  sorry

end S6_div_a6_l617_617329


namespace polynomial_has_real_solution_not_a_solution_l617_617700

theorem polynomial_has_real_solution : ∀ a : ℝ, ∃ x : ℝ, (1 + a) * x^4 + x^3 - (3 * a + 2) * x^2 - 4 * a = 0 :=
by {
  intro a,
  use -2,
  sorry
}

theorem not_a_solution : ∃ x0 : ℝ, ∀ a : ℝ, (1 + a) * x0^4 + x0^3 - (3 * a + 2) * x0^2 - 4 * a ≠ 0 :=
by {
  use 2,
  intro a,
  sorry
}

end polynomial_has_real_solution_not_a_solution_l617_617700


namespace polynomial_factorization_l617_617517

-- Definitions from conditions
def p (x : ℝ) : ℝ := x^6 - 2 * x^4 + 6 * x^3 + x^2 - 6 * x + 9
def q (x : ℝ) : ℝ := (x^3 - x + 3)^2

-- The theorem statement proving question == answer given conditions
theorem polynomial_factorization : ∀ x : ℝ, p x = q x :=
by
  sorry

end polynomial_factorization_l617_617517


namespace cubing_identity_l617_617816

theorem cubing_identity (x : ℂ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
  sorry

end cubing_identity_l617_617816


namespace area_of_rectangle_l617_617495

theorem area_of_rectangle (M N P Q R S X Y : Type) 
  (PQ : ℝ) (PX XY YQ : ℝ) (R_perpendicular_to_PQ S_perpendicular_to_PQ : Prop) 
  (R_through_M S_through_Q : Prop) 
  (segment_lengths : PQ = PX + XY + YQ) : PQ = 5 ∧ PX = 1 ∧ XY = 2 ∧ YQ = 2 
  → 2 * (1/2 * PQ * 2) = 10 :=
  sorry

end area_of_rectangle_l617_617495


namespace num_people_second_hour_l617_617237

theorem num_people_second_hour 
  (n1_in n2_in n1_left n2_left : ℕ) 
  (rem_hour1 rem_hour2 : ℕ)
  (h1 : n1_in = 94)
  (h2 : n1_left = 27)
  (h3 : n2_left = 9)
  (h4 : rem_hour2 = 76)
  (h5 : rem_hour1 = n1_in - n1_left)
  (h6 : rem_hour2 = rem_hour1 + n2_in - n2_left) :
  n2_in = 18 := 
  by 
  sorry

end num_people_second_hour_l617_617237


namespace polynomial_mult_of_6_l617_617893

theorem polynomial_mult_of_6 (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) : 6 ∣ P.eval 5 := 
sorry

end polynomial_mult_of_6_l617_617893


namespace find_a_and_extreme_values_l617_617360

noncomputable theory

-- Given conditions
def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x) * (x^2 - a * x + 1)

def tangent_line (x y : ℝ) : Prop := 3 * x + y - 1 = 0

theorem find_a_and_extreme_values :
  (∃ a : ℝ, 
    ∃ (x0 : ℝ) (y0 : ℝ), 
    f a 0 = y0 ∧
    tangent_line 0 y0 ∧
    (deriv (f a) 0 = -3) ∧
    (a = 4)) ∧
  (∀ x : ℝ, x = -1 → x = 3 → 
    ((deriv (f 4) x = 0) ∧
    ((∃ k : ℝ, k < 0 →  -1 < k ∧ k < 3 → deriv (f 4) k < 0) ∧
    (deriv (f 4) -1 = 0 ∧ deriv (f 4) 3 = 0) ∧ 
    ((deriv (f 4) 0 = -3) ∧
    (f 4 (-1) = 6 / Real.exp 1) ∧ (f 4 3 = -2 * (Real.exp 3))))) :=
by
  -- Proof omitted
  sorry


end find_a_and_extreme_values_l617_617360


namespace circumference_irrational_l617_617827

theorem circumference_irrational (d : ℚ) : ¬ ∃ (r : ℚ), r = π * d :=
sorry

end circumference_irrational_l617_617827


namespace Kolya_made_the_mistake_l617_617998

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l617_617998


namespace infinitely_many_a_l617_617306

theorem infinitely_many_a (n : ℕ) : ∃ a : ℕ, ∀ n : ℕ, ∃ k : ℕ, 
  let a := 3 * n^4 * k + 9 * n^2 * k^2 + 9 * k^3 in n^6 + 3 * a = (n^2 + 3 * k)^3 := 
begin
  sorry
end

end infinitely_many_a_l617_617306


namespace brocard_point_circumradii_eq_l617_617059

-- Definitions and setup for the problem
variables {ABC : Type} [triangle ABC]
variable (P : point ABC) -- P is the Brocard point of triangle ABC
variable (R : ℝ) (R1 R2 R3 : ℝ) -- circumradii

-- Hypotheses according to the conditions in step a)
hypothesis (h1 : isBrocardPoint P ABC)
hypothesis (hR : isCircumradius R ABC)
hypothesis (hR1 : isCircumradius R1 (triangle (corner A) (corner B) P))
hypothesis (hR2 : isCircumradius R2 (triangle (corner B) (corner C) P))
hypothesis (hR3 : isCircumradius R3 (triangle (corner C) (corner A) P))

-- The statement to be proven
theorem brocard_point_circumradii_eq (h1 : isBrocardPoint P ABC) (hR : isCircumradius R ABC) 
    (hR1: isCircumradius R1 (triangle (corner A) (corner B) P)) 
    (hR2: isCircumradius R2 (triangle (corner B) (corner C) P)) 
    (hR3: isCircumradius R3 (triangle (corner C) (corner A) P)) : 
         R1 * R2 * R3 = R^3 :=
by
  sorry

end brocard_point_circumradii_eq_l617_617059


namespace count_sevens_in_range_1_to_80_l617_617609

theorem count_sevens_in_range_1_to_80 : 
  ∃ (n : ℕ), n = 9 ∧ 
    (∀ k, 1 ≤ k ∧ k ≤ 80 → (list.count 7 (list.of_digit (nat.digits 10 k)) > 0) ↔ k ∈ list.sevens_list) :=
begin
  sorry
end

end count_sevens_in_range_1_to_80_l617_617609


namespace isosceles_triangle_angle_split_l617_617699

theorem isosceles_triangle_angle_split (A B C1 C2 : ℝ)
  (h_isosceles : A = B)
  (h_greater_than_third : A > C1)
  (h_split : C1 + C2 = C) :
  C1 = C2 :=
sorry

end isosceles_triangle_angle_split_l617_617699


namespace trick_or_treat_proof_l617_617534

-- Define the conditions
def children : ℕ := 3
def hours : ℕ := 4
def treats_per_house_per_kid : ℕ := 3
def total_treats : ℕ := 180

-- Define the quantity we want to prove
def houses_per_hour : ℕ := 5

theorem trick_or_treat_proof :
  (total_treats / children / hours / treats_per_house_per_kid = houses_per_hour) :=
by
  -- Use natural numbers for the definitions and the equation
  -- State the equality to be proven
  have : total_treats / children / hours / treats_per_house_per_kid = houses_per_hour,
  from rfl,
  sorry -- proof is omitted

end trick_or_treat_proof_l617_617534


namespace discriminant_zero_no_harmonic_progression_l617_617387

theorem discriminant_zero_no_harmonic_progression (a b c : ℝ) 
    (h_disc : b^2 = 24 * a * c) : 
    ¬ (2 * (1 / b) = (1 / a) + (1 / c)) := 
sorry

end discriminant_zero_no_harmonic_progression_l617_617387


namespace diagonal_divides_isosceles_not_rhombus_l617_617284

theorem diagonal_divides_isosceles_not_rhombus 
  (Q : Type) [quadrilateral Q] 
  (diags_isosceles : ∀ (d1 d2: diagonal Q), 
    divides_isosceles d1 Q ∧ divides_isosceles d2 Q) :
  ¬ rhombus Q :=
sorry

end diagonal_divides_isosceles_not_rhombus_l617_617284


namespace halfway_fraction_l617_617176

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l617_617176


namespace evaluate_expression_l617_617289

theorem evaluate_expression : (3 ^ (-3) * 7 ^ 0) / 3 ^ (-4) = 3 :=
by
  -- The proof goes here
  sorry

end evaluate_expression_l617_617289


namespace interval_satisfaction_l617_617723

theorem interval_satisfaction (a : ℝ) :
  (4 ≤ a / (3 * a - 6)) ∧ (a / (3 * a - 6) > 12) → a < 72 / 35 := 
by
  sorry

end interval_satisfaction_l617_617723


namespace abs_diff_of_two_numbers_l617_617147

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by 
  calc |x - y| = _ 
  sorry

end abs_diff_of_two_numbers_l617_617147


namespace volume_O_l617_617331

-- Definitions
def R : ℝ := 3 / 2
def r : ℝ := 2 / 3 * R
def surface_area_O (R : ℝ) : ℝ := 4 * π * R^2
def volume_sphere (r : ℝ) : ℝ := 4 / 3 * π * r^3

-- Conditions
axiom h1 : surface_area_O R = 9 * π
axiom h2 : r = 2 / 3 * R

-- Theorem statement
theorem volume_O' : volume_sphere r = 4 * π / 3 := 
  sorry

end volume_O_l617_617331


namespace sum_radical_conjugate_l617_617646

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617646


namespace strength_training_sessions_l617_617871

-- Define the problem conditions
def strength_training_hours (x : ℕ) : ℝ := x * 1
def boxing_training_hours : ℝ := 4 * 1.5
def total_training_hours : ℝ := 9

-- Prove how many times a week does Kat do strength training
theorem strength_training_sessions : ∃ x : ℕ, strength_training_hours x + boxing_training_hours = total_training_hours ∧ x = 3 := 
by {
  sorry
}

end strength_training_sessions_l617_617871


namespace radical_conjugate_sum_l617_617692

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617692


namespace correct_operation_l617_617571

variable (a b : ℝ)

theorem correct_operation : (a^2 * a^3 = a^5) :=
by sorry

end correct_operation_l617_617571


namespace ao_perpendicular_be_l617_617429

open EuclideanGeometry

-- Definition of the isosceles triangle with midpoint H on base BC.
def isosceles_triangle_and_midpoint (A B C H : Point) :=
  isosceles A B C ∧ midpoint H B C

-- Definition of the perpendicularity condition on HE and AC.
def he_perpendicular_ac (H E A C : Point) :=
  H ∈ line A C ∧ E ∈ line A C ∧ E ≠ H ∧ (perpendicular (line H E) (line A C))

-- Definition of the midpoint of HE.
def midpoint_HE (H E O : Point) :=
  midpoint O H E

-- The theorem statement.
theorem ao_perpendicular_be
  (A B C H E O : Point)
  (h1 : isosceles_triangle_and_midpoint A B C H)
  (h2 : he_perpendicular_ac H E A C)
  (h3 : midpoint_HE H E O) :
  perpendicular (line A O) (line B E) :=
sorry

end ao_perpendicular_be_l617_617429


namespace construct_parallelogram_l617_617162

/-- To construct a parallelogram, at least 4 small rods are needed. -/
def rods_needed_for_parallelogram : ℕ :=
4

theorem construct_parallelogram (sides : ℕ) (parallel : bool) (equal : bool) :
  sides = 4 → 
  parallel = true → 
  equal = true → 
  rods_needed_for_parallelogram = 4 :=
by
  sorry

end construct_parallelogram_l617_617162


namespace find_mistake_l617_617993

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l617_617993


namespace number_of_x0_eq_x5_l617_617323

noncomputable def x_seq (x0 : ℝ) (n : ℕ) : ℝ :=
  Nat.recOn n x0 (fun n xn =>
    if 2 * xn < 1 then 2 * xn else 2 * xn - 1)

theorem number_of_x0_eq_x5 : 
  ∀ (x0 : ℝ), (0 ≤ x0 ∧ x0 < 1) → 
              (x_seq x0 5 = x0) →
              (∃ count, count = 31 ∧ 
              (∀ x0 such that (0 ≤ x0 ∧ x0 < 1) ∧ x_seq x0 5 = x0, count = 31)) :=
by
  sorry

end number_of_x0_eq_x5_l617_617323


namespace geometric_sequence_first_term_l617_617984

theorem geometric_sequence_first_term (a r : ℝ)
    (h1 : a * r^2 = 3)
    (h2 : a * r^4 = 27) :
    a = 1 / 3 := by
    sorry

end geometric_sequence_first_term_l617_617984


namespace median_lap_duration_l617_617981

theorem median_lap_duration 
  (times : List ℕ) 
  (h_times : times = [45, 48, 50, 70, 75, 80, 90, 105, 125, 130, 135, 140, 145, 190, 195]) : 
  List.median times = 105 := by 
sory

end median_lap_duration_l617_617981


namespace odell_kershaw_meetings_l617_617089

-- Define the variables for the problem
def odell_radius : ℝ := 40
def kershaw_radius : ℝ := 50
def odell_speed : ℝ := 200
def kershaw_speed : ℝ := 270
def running_time : ℝ := 35

-- Define the circumferences
def odell_circumference : ℝ := 2 * Real.pi * odell_radius
def kershaw_circumference : ℝ := 2 * Real.pi * kershaw_radius

-- Define the angular speeds in radians per minute
def odell_angular_speed : ℝ := (odell_speed / odell_circumference) * 2 * Real.pi
def kershaw_angular_speed : ℝ := (kershaw_speed / kershaw_circumference) * 2 * Real.pi

-- Define the relative angular speed (since they run in opposite directions)
def relative_angular_speed : ℝ := odell_angular_speed + kershaw_angular_speed

-- Define the time taken to complete one relative cycle (2 * pi radians)
def time_per_meeting : ℝ := (2 * Real.pi) / relative_angular_speed

-- Calculate total number of meetings within the given running time
def total_meetings : ℕ := Int.floor (running_time / time_per_meeting)

-- The main theorem to prove
theorem odell_kershaw_meetings : total_meetings = 57 := 
by {
  sorry
}

end odell_kershaw_meetings_l617_617089


namespace sin_x1_sub_x2_l617_617792

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem sin_x1_sub_x2 (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < Real.pi)
  (h₄ : f x₁ = 1 / 3) (h₅ : f x₂ = 1 / 3) : 
  Real.sin (x₁ - x₂) = - (2 * Real.sqrt 2) / 3 := 
sorry

end sin_x1_sub_x2_l617_617792


namespace lisa_flight_distance_l617_617450

-- Define the given speed and time
def speed : ℝ := 32
def time : ℝ := 8

-- Define the distance formula
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

-- State the theorem to be proved
theorem lisa_flight_distance : distance speed time = 256 := by
sorry

end lisa_flight_distance_l617_617450


namespace seq_an_arithmetic_sum_b_n_l617_617358

variables {X : Type*} [Ring X] (a : ℕ → X)

def f (n : ℕ) : X := ∑ i in range (n + 1), a i

noncomputable def seq_a (n : ℕ) : X :=
  if n = 1 then 2 else 2 * n

theorem seq_an_arithmetic (a : ℕ → X) (n : ℕ) (hn : ∀ n ∈ ℕ, f n = n^2 + n) :
  sequ_a n = 2n := sorry

noncomputable def b_n (n: ℕ) (f : X) : X :=
  2^f

noncomputable def S_n (n : ℕ) (bn: ℕ → X) : X :=
 ∑ k in range n, bn k

theorem sum_b_n (n : ℕ) (hn_even : even n) (f : X) :
  S_n n (λ k, b_n k f) = 2^(n + 1) - 2 := sorry

end seq_an_arithmetic_sum_b_n_l617_617358


namespace integral_solution_l617_617875

noncomputable def integral_problem (α β : ℝ) (hα : 2 * α = Real.tan α) (hβ : 2 * β = Real.tan β) (h_distinct : α ≠ β) : ℝ :=
∫ x in 0..1, Real.sin (α * x) * Real.sin (β * x)

theorem integral_solution (α β : ℝ) (hα : 2 * α = Real.tan α) (hβ : 2 * β = Real.tan β) (h_distinct : α ≠ β) :
  integral_problem α β hα hβ h_distinct = 0 :=
by
  sorry

end integral_solution_l617_617875


namespace collatz_conjecture_probability_l617_617112

noncomputable def collatz_sequence := [10, 5, 16, 8, 4, 2, 1]

def odd_numbers := collatz_sequence.filter (λ n, n % 2 = 1)

def choose (n k : ℕ) : ℕ := n.choose k

def probability_of_two_odd_numbers_3n_plus_1_conjecture
    (seq : List ℕ) (evens odds : List ℕ) : Prop :=
  let evens := seq.filter (λ n, n % 2 = 0)
  let odds := seq.filter (λ n, n % 2 = 1)
  (evens.length = 5) ∧
  (odds.length = 2) ∧
  (choose odds.length 2) / (choose seq.length 2) = 1 / 21

theorem collatz_conjecture_probability :
  probability_of_two_odd_numbers_3n_plus_1_conjecture collatz_sequence
  (collatz_sequence.filter (λ n, n % 2 = 0))
  odd_numbers :=
by
  sorry

end collatz_conjecture_probability_l617_617112


namespace radius_of_circle_l617_617332

theorem radius_of_circle (A B C D O E F : ℝ) (side_length : ℝ)
  (h_square : A - B = side_length ∧ B - C = side_length ∧ C - D = side_length ∧ D - A = side_length)
  (h_circle_pass_through_A_B : circle O A radius ∧ circle O B radius)
  (h_circle_tangent_CD_AD : tangent circle O CD ∧ tangent circle O AD)
  : radius = 5 * real.sqrt 2 :=
sorry

end radius_of_circle_l617_617332


namespace janet_total_action_figures_l617_617040

/-- Janet owns 10 action figures, sells 6, gets 4 more in better condition,
and then receives twice her current collection from her brother.
We need to prove she ends up with 24 action figures. -/
theorem janet_total_action_figures :
  let initial := 10 in
  let after_selling := initial - 6 in
  let after_acquiring_better := after_selling + 4 in
  let from_brother := 2 * after_acquiring_better in
  after_acquiring_better + from_brother = 24 :=
by
  -- Proof would go here
  sorry

end janet_total_action_figures_l617_617040


namespace part_a_part_b_l617_617077

section
variables {A B C O H P Q : Type}  -- Points on the Plane

-- Part (a)
-- Non-equilateral, acute triangle ABC, given condition angle A = 60 degrees.
-- O is the circumcenter and H is the orthocenter.
variables (h_non_equilateral : ¬ (A = B ∧ B = C ∧ C = A))
          (h_acute : ∀ X Y Z, X ≠ Y → Y ≠ Z → Z ≠ X → (X ≠ Z ∨ (∠X Y Z ≤ 90)))
          (h_angle_A : ∠A = 60)
          (h_O_circumcenter : ∀ X, (∠X O B = 60))
          (h_H_orthocenter : ∀ X, (∠X H C = 60))
          (h_intersect_P : OH ∩ AB = P)
          (h_intersect_Q : OH ∩ AC = Q)

theorem part_a : line OH ∩ seg AB = P ∧ line OH ∩ seg AC = Q := 
  sorry

-- Part (b)
-- s and t are the areas of triangles APQ and quadrilateral BPQC respectively
variables (s t : Type) 
variables (h_APQ_area : area APQ = s)
variables (h_BPQC_area : area BPQC = t)
variables (h_range : 4/5 < s / t ∧ s / t < 1)

theorem part_b :  4/5 < s / t ∧ s / t < 1 :=
  sorry
end

end part_a_part_b_l617_617077


namespace triangle_median_length_l617_617016

theorem triangle_median_length :
  ∀ (A B C N : Point)
    (hAB : dist A B = 26)
    (hAC : dist A C = 26)
    (hBC : dist B C = 20)
    (hN_midpoint : midpoint B C N),
  dist A N = 24 :=
by
  sorry

end triangle_median_length_l617_617016


namespace remainder_of_2023rd_term_l617_617264

noncomputable def position_end (n : ℕ) := n * (n + 1) / 2

def nth_term_of_sequence (k : ℕ) : ℕ :=
  let n := (sqrt (8 * k + 1) - 1) / 2
  in if k <= position_end n then n
     else n + 1

theorem remainder_of_2023rd_term (k : ℕ) (h : k = 2023) : 
  nth_term_of_sequence k % 7 = 1 :=
by
  have pos := position_end 63
  have next_pos := position_end 64
  have hn := 63
  have hk := k - pos
  have hnk := hk + hn * 7 - 1
  sorry

end remainder_of_2023rd_term_l617_617264


namespace coefficient_of_b_l617_617817

theorem coefficient_of_b (a : ℝ) (b : ℝ) (some_number : ℝ) (h1 : 7 * a = some_number) (h2 : b = 15) 
(h3 : 42 * a * b = 674.9999999999999) : 42 * a ≈ 45 :=
by
  sorry

end coefficient_of_b_l617_617817


namespace triangles_same_area_l617_617030

open EuclideanGeometry

-- Definitions of points and properties in the context of triangle ABC.
variables {A B C R P Q S T O : Point}
variables [is_triangle A B C]
variables [is_angle_bisector C A B C A C]
variables [is_intersect_circumcircle C A B C R]
variables [is_perpendicular_bisector_intersection A B C S (segment_midpoint B C)]
variables [is_perpendicular_bisector_intersection A C T (segment_midpoint C A)]
variables [is_perpendicular_bisector_intersection_bisector C S P]
variables [is_perpendicular_bisector_intersection_bisector C T Q]

-- Theorem statement
theorem triangles_same_area :
  area (triangle R Q T) = area (triangle R P S) :=
begin
  sorry
end

end triangles_same_area_l617_617030


namespace cylinder_volume_l617_617758

theorem cylinder_volume (r h V: ℝ) (r_pos: r = 4) (lateral_area: 2 * 3.14 * r * h = 62.8) : 
    V = 125600 :=
by
  sorry

end cylinder_volume_l617_617758


namespace least_possible_diagonals_l617_617618

noncomputable def leastDiagonals : ℝ :=
  let n := 2021 in 2018

theorem least_possible_diagonals (labels : Fin 2021 → ℝ)
  (h1 : ∀ i, abs (labels i - labels ((i + 1) % 2021)) ≤ 1)
  (h2 : ∀ i j, i ≠ j → abs (labels i - labels j) ≤ 1 → is_diagonal i j) :
  leastDiagonals = 2018 :=
sorry

end least_possible_diagonals_l617_617618


namespace sum_of_true_powers_theorem_l617_617073

-- Define what constitutes a true power
def is_true_power (a b n : ℕ) : Prop := a > 1 ∧ b > 1 ∧ n = a^b

-- Define the sum of true powers
def sum_of_true_powers (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a b : Fin k → ℕ), (∀ i, is_true_power (a i) (b i) (a i ^ (b i))) ∧ (Σ i, (a i ^ (b i)) = n)

-- Formalize the problem: Every positive integer not listed as exceptional can be expressed as such a sum
theorem sum_of_true_powers_theorem : 
  ∀ n, n ≠ 1 ∧ n ≠ 3 ∧ n ≠ 5 ∧ n ≠ 6 ∧ n ≠ 7 ∧ n ≠ 10 ∧ n ≠ 11 ∧ n ≠ 14 ∧ n ≠ 15 ∧ n ≠ 19 ∧ n ≠ 23 → sum_of_true_powers n := by
  sorry

end sum_of_true_powers_theorem_l617_617073


namespace find_3m_plus_n_l617_617426

theorem find_3m_plus_n (m n : ℕ) (h1 : m > n) (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 3 * m + n = 46 :=
sorry

end find_3m_plus_n_l617_617426


namespace quadratic_no_solution_l617_617515

theorem quadratic_no_solution 
  (p q r s : ℝ) (h1 : p^2 < 4 * q) (h2 : r^2 < 4 * s) :
  (1009 * p + 1008 * r)^2 < 4 * 2017 * (1009 * q + 1008 * s) :=
by
  sorry

end quadratic_no_solution_l617_617515


namespace circle_contained_probability_l617_617815

noncomputable def probability_circle_contained_in_other 
  (a : ℕ) (S : set ℕ) (C O : ℝ → ℝ → Prop) 
  (total_basic_events : ℕ) (favorable_basic_events : ℕ) : ℚ :=
  favorable_basic_events / total_basic_events

theorem circle_contained_probability :
  let S := { n | n ∈ {1, 2, 3, 4, 5, 6, 7} } in
  let C := λ x y, x^2 + (y-2)^2 = 1 in
  let O := λ x y, x^2 + y^2 = (a : ℝ)^2 in
  (∃ (total_basic_events : ℕ) (favorable_basic_events : ℕ),
    total_basic_events = 7 ∧
    favorable_basic_events = 4 ∧
    (a > 3 → a ∈ S)) →
  probability_circle_contained_in_other a S C O 7 4 = 4 / 7 := 
by {
  intro h,
  sorry
}

end circle_contained_probability_l617_617815


namespace arithmetic_series_sum_correct_l617_617257

-- Define the parameters of the arithmetic series
def a : ℤ := -53
def l : ℤ := 3
def d : ℤ := 2

-- Define the number of terms in the series
def n : ℕ := 29

-- The expected sum of the series
def expected_sum : ℤ := -725

-- Define the nth term formula
noncomputable def nth_term (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the arithmetic series
noncomputable def arithmetic_series_sum (a l : ℤ) (n : ℕ) : ℤ :=
  (n * (a + l)) / 2

-- Statement of the proof problem
theorem arithmetic_series_sum_correct :
  arithmetic_series_sum a l n = expected_sum := by
  sorry

end arithmetic_series_sum_correct_l617_617257


namespace value_of_x_plus_y_l617_617819

theorem value_of_x_plus_y (x y : ℚ) (h1 : 1 / x + 1 / y = 5) (h2 : 1 / x - 1 / y = -9) : x + y = -5 / 14 := sorry

end value_of_x_plus_y_l617_617819


namespace radical_conjugate_sum_l617_617694

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617694


namespace right_triangle_set_exists_l617_617248

noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_triangle_set_exists :
  ∃ (a b c : ℝ), (a = 1 ∧ b = real.sqrt 2 ∧ c = real.sqrt 3) ∧ is_right_triangle a b c :=
by
  use [1, real.sqrt 2, real.sqrt 3]
  simp [is_right_triangle]
  sorry

end right_triangle_set_exists_l617_617248


namespace fifteenth_odd_multiple_of_5_is_145_l617_617190

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l617_617190


namespace graph_degree_and_smallest_n_l617_617597

theorem graph_degree_and_smallest_n
  (G : Type) [graph : SimpleGraph G]
  (h1 : ∀ (v : G), ∃ (u : G), ¬has_edge (graph.adj) v u)
  (h2 : ∀ (v₁ v₂ v₃ : G), graph.adj v₁ v₂ → graph.adj v₂ v₃ → graph.adj v₁ v₃ → false)
  (h3 : ∀ (A B : G), ¬graph.adj A B → ∃! C : G, graph.adj A C ∧ graph.adj B C) :
  (∀ (v : G), ∃ m : ℕ, (∀ w : G, w ≠ v → graph.adj v w → (∃ deg : ℕ, deg = graph.degree v)))
  ∧ (∃ m : ℕ, m ≥ 2 ∧ fintype.card G = m^2 + 1) :=
by
  sorry

end graph_degree_and_smallest_n_l617_617597


namespace polynomials_preserve_remainders_l617_617739

-- Define polynomial recursively
def P (p : ℤ → ℤ) (n : ℕ) : ℤ → ℤ :=
  match n with
  | 0 => id
  | n + 1 => p ∘ (P p n)

-- Define the core theorem to be proved
theorem polynomials_preserve_remainders (n : ℕ) :
  (∃ p : ℤ → ℤ, ∀ m : ℕ, m > 0 →
    (set.range (P p m)).subtype ({x | 0 ≤ x ∧ x < n}) = fintype.card (finset.image (λ x, x % n) (finset.range n)) ) ↔
  (∃ k : ℕ, n = 2^k) ∨ nat.prime n :=
 sorry

end polynomials_preserve_remainders_l617_617739


namespace construct_bisector_of_inaccessible_angle_l617_617327

theorem construct_bisector_of_inaccessible_angle 
  (r : Type) (p : r → r → Prop) [metric_space r] [plane_geometry r] 
  (ruler : ∀ {A B C D E F : r}, p (line_through A B) (line_through C D) 
  → p (line_through E F) (line_through C D)
  → rs_eq_length (distance A B) (thickness ruler) ε ∧ 
    rs_eq_length (distance E F) (thickness ruler) δ) 
  (inaccessible_vertex : r)
  (angle_vertex_outside_drawing : out_of_drawing inaccessible_vertex)
  (construct_intersections : ∀ {P Q R : r},
    is_parallel (line_through P Q) (line_through R Q) 
    → intersection_point := intersection P Q R Q) 
  : ∃ (bisector : r → r → Prop), bisects_angle bisector (line_through inaccessible_vertex edge1) (line_through inaccessible_vertex edge2) := 
sorry

end construct_bisector_of_inaccessible_angle_l617_617327


namespace sum_of_number_and_conjugate_l617_617668

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617668


namespace number_of_cows_l617_617401

-- Definitions of variables and conditions
variables (C H : ℕ)

-- Conditions provided in the problem
def cows_legs : ℕ := 4 * C
def chickens_legs : ℕ := 2 * H
def total_legs : ℕ := cows_legs C H + chickens_legs C H
def total_heads : ℕ := C + H

-- Given condition for the problem
def legs_condition : Prop := total_legs C H = 2 * total_heads C H + 10

-- Goal: Prove that the number of cows (C) is 5
theorem number_of_cows (h : legs_condition C H) : C = 5 :=
sorry

end number_of_cows_l617_617401


namespace total_problems_practiced_l617_617514

-- Definitions from conditions
variables (marvin_yesterday : Nat) (marvin_today : Nat) (arvin_yesterday : Nat) (arvin_today : Nat)

-- Conditions stated as definitions
def marvin_yesterday_solved : marvin_yesterday = 40 := rfl
def marvin_today_solved : marvin_today = 3 * marvin_yesterday := by rw [marvin_yesterday_solved]; exact rfl
def arvin_yesterday_solved : arvin_yesterday = 2 * marvin_yesterday := by rw [marvin_yesterday_solved]; exact rfl
def arvin_today_solved : arvin_today = 2 * marvin_today := by rw [marvin_today_solved]; exact rfl

-- The proof problem statement
theorem total_problems_practiced :
  marvin_yesterday + marvin_today + arvin_yesterday + arvin_today = 480 :=
by
  rw [marvin_yesterday_solved, marvin_today_solved, arvin_yesterday_solved, arvin_today_solved]
  sorry

end total_problems_practiced_l617_617514


namespace surface_area_of_circumscribed_sphere_l617_617760

-- Given the lengths of the edges PA, PB, and PC, and they are mutually perpendicular.
variables (PA PB PC : ℝ)
axiom mutually_perpendicular (PA PB PC : ℝ) : true  -- Dummy axiom for mutual perpendicularity
axioms (hPA : PA = 2) (hPB : PB = 1) (hPC : PC = 1)

-- Definition for the radius of the circumscribed sphere using the Pythagorean theorem in three dimensions
noncomputable def radius_of_circumscribed_sphere (PA PB PC : ℝ) : ℝ :=
  (Real.sqrt (PA^2 + PB^2 + PC^2)) / 2

-- Definition for the surface area of the sphere given radius
noncomputable def surface_area_of_sphere (R : ℝ) : ℝ :=
  4 * Real.pi * R^2

-- Statement of the theorem
theorem surface_area_of_circumscribed_sphere :
  surface_area_of_sphere (radius_of_circumscribed_sphere PA PB PC) = 6 * Real.pi :=
by
  sorry  -- Proof goes here

-- Specific case for the given problem
#check surface_area_of_circumscribed_sphere

end surface_area_of_circumscribed_sphere_l617_617760


namespace find_y_from_eqns_l617_617005

theorem find_y_from_eqns (x y : ℝ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 :=
by {
  sorry
}

end find_y_from_eqns_l617_617005


namespace simplify_factorial_expression_l617_617935

theorem simplify_factorial_expression :
  (13.factorial / (10.factorial + 3 * 9.factorial)) = 1320 :=
by
  sorry

end simplify_factorial_expression_l617_617935


namespace line_intersects_circle_l617_617411

-- Define the conditions for the polar equations of line and circle
def polar_line (ρ θ : ℝ) : Prop := ρ * cos (θ - π / 4) = 2
def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * sin θ - 2 * cos θ

-- Define the Cartesian equivalents, though we don't expand these here based on the instructions
def cartesian_line (x y : ℝ) : Prop := x + y = 2 * sqrt 2
def circle_center := (-1:ℝ, 2:ℝ)
def circle_radius := sqrt 5

-- Given the polar equations, show that the line intersects the circle
theorem line_intersects_circle : 
  (∃ (x y : ℝ), polar_line (sqrt (x^2 + y^2)) (atan2 y x) ∧ polar_circle (sqrt (x^2 + y^2)) (atan2 y x)) ∧ cartesian_line circle_center.1 circle_center.2 := 
by
  sorry

end line_intersects_circle_l617_617411


namespace jenny_jellybeans_original_l617_617715

theorem jenny_jellybeans_original (x : ℝ) 
  (h : 0.75^3 * x = 45) : x = 107 := 
sorry

end jenny_jellybeans_original_l617_617715


namespace no_infinite_replacements_l617_617023

-- We define a structure for finite sets of points in a plane
structure finite_set_of_points (α : Type*) :=
(pts : set α)
(finite_pts : pts.finite)
(no_three_collinear : ∀ a b c ∈ pts, a ≠ b → b ≠ c → c ≠ a → ¬collinear ℝ {a, b, c})

-- The problem translates to: for such a finite set with the specified conditions, the replacements can't be infinite
theorem no_infinite_replacements 
  (α : Type*) [metric_space α] [normed_group α] [normed_space ℝ α] 
  (s : finite_set_of_points α) (segments : set (α × α)) 
  (h1 : ∀ p, ∃! q, (p, q) ∈ segments ∨ (q, p) ∈ segments) 
  (h2 : ∀ a b c d, (a, b) ∈ segments ∧ (c, d) ∈ segments 
    → (∃ e f, (a, c) ∈ segments ∧ (b, d) ∈ segments) 
    → dist a c + dist b d < dist a b + dist c d) : 
  ¬ ∃ f : ℕ → set (α × α), ∀ n, (∀ m < n, f m ≠ f n) ∧ 
    ∀ i, (∃ a b c d, (a, b) ∈ f i ∧ (c, d) ∈ f i ∧ 
      ∃ e, (a, c) ∈ f (i+1) ∧ (b, d) ∈ f (i+1)) := 
sorry

end no_infinite_replacements_l617_617023


namespace prove_3a_3b_3c_l617_617820

variable (a b c : ℝ)

def condition1 := b + c = 15 - 2 * a
def condition2 := a + c = -18 - 3 * b
def condition3 := a + b = 8 - 4 * c
def condition4 := a - b + c = 3

theorem prove_3a_3b_3c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a b c) :
  3 * a + 3 * b + 3 * c = 24 / 5 :=
sorry

end prove_3a_3b_3c_l617_617820


namespace value_of_a_l617_617886

theorem value_of_a (a b c : ℂ) (h_real : a.im = 0)
  (h1 : a + b + c = 5) 
  (h2 : a * b + b * c + c * a = 7) 
  (h3 : a * b * c = 2) : a = 2 := by
  sorry

end value_of_a_l617_617886


namespace midpoints_form_dodecagon_l617_617854

noncomputable def midpoint (A B : Point) : Point := sorry

structure Square := 
  (A B C D : Point)
  (is_square : sorry)

structure EquilateralTriangle (A B K : Point) := 
  (is_equilateral : sorry)

def Point := sorry

theorem midpoints_form_dodecagon :
  ∀ (A B C D K L M N E F G H P Q R S T U V W : Point),
    Square A B C D →
    EquilateralTriangle A B K →
    EquilateralTriangle B C L →
    EquilateralTriangle C D M →
    EquilateralTriangle D A N →
    E = midpoint K L →
    F = midpoint L M →
    G = midpoint M N →
    H = midpoint N K →
    P = midpoint A K →
    Q = midpoint B K →
    R = midpoint B L →
    S = midpoint C L →
    T = midpoint C M →
    U = midpoint D M →
    V = midpoint D N →
    W = midpoint A N →
    regular_dodecagon [E, F, G, H, P, Q, R, S, T, U, V, W] := sorry

end midpoints_form_dodecagon_l617_617854


namespace hiking_speeds_class_b_time_to_endpoint_l617_617396

variable (x : Real)

-- Part 1: Proving the speeds of Class A and Class B
theorem hiking_speeds {
  (head_start : Real) (meeting_time : Real) (speed_ratio : Real) (distance_ahead : Real) : 
  (distance_ahead = 0.75) → (speed_ratio = 1.5) → (meeting_time = 0.5) → (head_start = x) →
  (0.5 * (1.5 * x - x) = distance_ahead) → 
  (1.5 * x = 4.5) ∧ (x = 3) :=
by sorry

-- Part 2: Calculating the time Class B took to reach the endpoint
theorem class_b_time_to_endpoint {
  (planned_speed : Real) (distance : Real) (extra_speed : Real) (time_saved : Real) : 
  (planned_speed = x) → (distance = 7.5) → (extra_speed = 4.5) → (time_saved = 1/6) →
  ((distance / x) - time_saved = 1 + ((distance - x) / extra_speed)) → 
  ((distance / 5) - (1 / 6) = 4/3) :=
by sorry

end hiking_speeds_class_b_time_to_endpoint_l617_617396


namespace fixed_chord_length_l617_617348

-- Define the circle equation
def circle_eq (a x y : ℝ) := x^2 + y^2 + (4 - 2*a)*x - 2*sqrt(3)*a*y + 4*a^2 - 4*a - 12 = 0

-- Define the line equation passing through the point (1, 0)
def line_eq (k x y : ℝ) := y = k * (x - 1)

-- Define the center and radius of the circle
def center (a : ℝ) := (a - 2, sqrt(3) * a)
def radius : ℝ := 4

-- Hypothesize that for any real number a, the chord length is fixed to sqrt(37)
def chord_length_fixed (k a : ℝ) : Prop :=
  let (cx, cy) := center a in
  ∃ h : ℝ, h = |k * (a - 2) - sqrt(3) * a - k| / sqrt(k^2 + 1) ∧
             2 * sqrt(radius^2 - h^2) = sqrt(37)

-- The statement of the proof problem
theorem fixed_chord_length :
  ∀ a : ℝ, ∃ k : ℝ, chord_length_fixed k a :=
by
  sorry

end fixed_chord_length_l617_617348


namespace inequality_solution_l617_617481

theorem inequality_solution (x : ℝ) : (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) ↔ (-1/2 ≤ x ∧ x < 1) :=
by
  sorry

end inequality_solution_l617_617481


namespace giraffes_count_l617_617831

variable {TotalAnimals G P : ℕ}

axiom condition1 : P = 2 * G
axiom condition2 : P = 0.20 * TotalAnimals
axiom condition3 : 2 = 0.04 * TotalAnimals

theorem giraffes_count : G = 5 := by
  sorry

end giraffes_count_l617_617831


namespace percentage_of_25_of_fifty_percent_of_500_l617_617566

-- Define the constants involved
def fifty_percent_of_500 := 0.50 * 500  -- 50% of 500

-- Prove the equivalence
theorem percentage_of_25_of_fifty_percent_of_500 : (25 / fifty_percent_of_500) * 100 = 10 := by
  -- Place proof steps here
  sorry

end percentage_of_25_of_fifty_percent_of_500_l617_617566


namespace ratio_area_circle_triangle_l617_617244

theorem ratio_area_circle_triangle (h R : ℝ) (hR : R = h / 2) :
  let A := (1 / 2) * (h / sqrt 2) * (h / sqrt 2) in
  let circle_area := π * R^2 in
  let triangle_area := A in
  let ratio := circle_area / triangle_area in
  ratio = π :=
by
  sorry

end ratio_area_circle_triangle_l617_617244


namespace sum_of_number_and_its_radical_conjugate_l617_617662

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617662


namespace exists_lambda_fractional_parts_in_interval_l617_617051

theorem exists_lambda_fractional_parts_in_interval
  (a b c : ℕ) (h1 : b > 2 * a) (h2 : c > 2 * b) :
  ∃ λ : ℝ, (frac (λ * a) ∈ Ioo (1/3 : ℝ) (2/3 : ℝ)) ∧ 
           (frac (λ * b) ∈ Ioo (1/3 : ℝ) (2/3 : ℝ)) ∧ 
           (frac (λ * c) ∈ Ioo (1/3 : ℝ) (2/3 : ℝ)) :=
by sorry

end exists_lambda_fractional_parts_in_interval_l617_617051


namespace sum_of_number_and_its_radical_conjugate_l617_617657

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617657


namespace max_integer_solution_of_inequality_l617_617140

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 8

theorem max_integer_solution_of_inequality :
  (∃ x0 : ℝ, f x0 = 0) ∧
  (∀ x : ℝ, f' x > 0) ∧
  f 3 < 0 ∧
  f 4 > 0 →
  (∀ x : ℤ, x ≤ 3) :=
by
  sorry

end max_integer_solution_of_inequality_l617_617140


namespace polynomial_multiple_of_six_l617_617896

theorem polynomial_multiple_of_six 
  (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) :
  6 ∣ P.eval 5 :=
sorry

end polynomial_multiple_of_six_l617_617896


namespace geometric_sequence_product_l617_617409

theorem geometric_sequence_product
    (a : ℕ → ℝ)
    (r : ℝ)
    (h₀ : a 1 = 1 / 9)
    (h₃ : a 4 = 3)
    (h_geom : ∀ n, a (n + 1) = a n * r) :
    (a 1) * (a 2) * (a 3) * (a 4) * (a 5) = 1 :=
sorry

end geometric_sequence_product_l617_617409


namespace fraction_halfway_between_fraction_halfway_between_l617_617179

theorem fraction_halfway_between : (3/4 : ℚ) < (5/7 : ℚ) :=
by linarith

theorem fraction_halfway_between : (41 / 56 : ℚ) = (1 / 2) * ((3 / 4) + (5 / 7)) :=
by sorry

end fraction_halfway_between_fraction_halfway_between_l617_617179


namespace solve_cos_sin_eq_one_l617_617476

open Real

theorem solve_cos_sin_eq_one (n : ℕ) (hn : n > 0) :
  {x : ℝ | cos x ^ n - sin x ^ n = 1} = {x : ℝ | ∃ k : ℤ, x = k * π} :=
by
  sorry

end solve_cos_sin_eq_one_l617_617476


namespace ones_digit_of_4567_times_3_is_1_l617_617970

theorem ones_digit_of_4567_times_3_is_1 :
  let n := 4567
  let m := 3
  (n * m) % 10 = 1 :=
by
  let n := 4567
  let m := 3
  have h : (n * m) % 10 = ((4567 * 3) % 10) := by rfl -- simplifying the product
  sorry -- this is where the proof would go, if required

end ones_digit_of_4567_times_3_is_1_l617_617970


namespace least_possible_diagonals_l617_617616

theorem least_possible_diagonals :
  let n := 2021 in
  let labels := Fin n → ℝ in
  ∃ (d : ℕ), 
    (∀ (x : labels), 
      (∀ (i j : Fin n), 
        (i ≠ j ∧ i.succ ≠ j ∧ j.succ ≠ i ∧ abs (x i - x j) ≤ 1) → 
        (∃ (k l : Fin n), (k ≠ l ∧ abs (x k - x l) ≤ 1))) 
    → d ≥ 4039) 
  ∧ 
    d = 4039 :=
sorry

end least_possible_diagonals_l617_617616


namespace fifteenth_odd_multiple_of_5_l617_617195

theorem fifteenth_odd_multiple_of_5 :
  (∃ n: ℕ, n = 15 ∧ (10 * n - 5 = 145)) :=
begin
  use 15,
  split,
  { refl },
  { norm_num }
end

end fifteenth_odd_multiple_of_5_l617_617195


namespace simplify_polynomial_l617_617102

theorem simplify_polynomial (y : ℝ) :
    (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) =
    2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
  sorry

end simplify_polynomial_l617_617102


namespace total_spent_l617_617545

theorem total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ)
  (h1 : deck_price = 8)
  (h2 : victor_decks = 6)
  (h3 : friend_decks = 2) :
  deck_price * victor_decks + deck_price * friend_decks = 64 :=
by
  sorry

end total_spent_l617_617545


namespace goods_train_length_l617_617576

theorem goods_train_length
  (speed_kmph : ℕ)
  (length_platform : ℕ)
  (time_secs : ℕ)
  (speed_kmph = 72)
  (length_platform = 240)
  (time_secs = 26) :
  (length_train : ℕ) 
  (20 * time_secs - length_platform = length_train) :=
  sorry

end goods_train_length_l617_617576


namespace even_perfect_square_factors_count_l617_617810

theorem even_perfect_square_factors_count : 
  let factors := {n : ℕ | ∃ a b : ℕ, n = 2^a * 7^b ∧ 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 10 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ 1 ≤ a}
  in factors.card = 18 :=
by
  sorry

end even_perfect_square_factors_count_l617_617810


namespace signup_ways_championship_ways_l617_617736
-- Import the necessary library

/-!
### Proof Problem 1
Given five students and four sports competitions, prove that the number of different ways they can sign up is \(4^5\).
-/

theorem signup_ways : ∀ (students : ℕ) (events : ℕ),
  students = 5 ∧ events = 4 → (events ^ students) = 1024 :=
begin
  -- Our conditions
  intros students events h,
  rcases h with ⟨hs, he⟩,
  -- Substitute given values
  rw [← hs, ← he],
  -- The expected result
  norm_num,
end

/-!
### Proof Problem 2
Given five students and four sports competitions, prove that the number of possibilities for winning the championship is \(5^4\).
-/

theorem championship_ways : ∀ (students : ℕ) (events : ℕ),
  students = 5 ∧ events = 4 → (students ^ events) = 625 :=
begin
  -- Our conditions
  intros students events h,
  rcases h with ⟨hs, he⟩,
  -- Substitute given values
  rw [← hs, ← he],
  -- The expected result
  norm_num,
end

end signup_ways_championship_ways_l617_617736


namespace find_a_l617_617345

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then 1 + a ^ x else -f a (-x)

theorem find_a (a : ℝ) (h1 : f a (-1) = -3/2) : a = 1/2 :=
by
  have h2 : f a 1 = 3/2 :=
    by rw [←f a (-1), h1]; sorry
  sorry

end find_a_l617_617345


namespace no_prime_roots_l617_617635

theorem no_prime_roots (k : ℕ) :
  (∃ p q : ℕ, prime p ∧ prime q ∧ (x^2 - 90 * x + k = (x - p) * (x - q)) ∧ (p + q = 90) ∧ (k = p * q)) → false :=
by
  sorry

end no_prime_roots_l617_617635


namespace units_digit_k_squared_plus_two_exp_k_eq_7_l617_617440

/-- Define k as given in the problem -/
def k : ℕ := 2010^2 + 2^2010

/-- Final statement that needs to be proved -/
theorem units_digit_k_squared_plus_two_exp_k_eq_7 : (k^2 + 2^k) % 10 = 7 := 
by
  sorry

end units_digit_k_squared_plus_two_exp_k_eq_7_l617_617440


namespace faye_complete_bouquets_l617_617720

theorem faye_complete_bouquets :
  let roses_initial := 48
  let lilies_initial := 40
  let tulips_initial := 76
  let sunflowers_initial := 34
  let roses_wilted := 24
  let lilies_wilted := 10
  let tulips_wilted := 14
  let sunflowers_wilted := 7
  let roses_remaining := roses_initial - roses_wilted
  let lilies_remaining := lilies_initial - lilies_wilted
  let tulips_remaining := tulips_initial - tulips_wilted
  let sunflowers_remaining := sunflowers_initial - sunflowers_wilted
  let bouquets_roses := roses_remaining / 2
  let bouquets_lilies := lilies_remaining
  let bouquets_tulips := tulips_remaining / 3
  let bouquets_sunflowers := sunflowers_remaining
  let bouquets := min (min bouquets_roses bouquets_lilies) (min bouquets_tulips bouquets_sunflowers)
  bouquets = 12 :=
by
  sorry

end faye_complete_bouquets_l617_617720


namespace number_of_solutions_eq_one_l617_617712

theorem number_of_solutions_eq_one :
  (∃! y : ℝ, (y ≠ 0) ∧ (y ≠ 3) ∧ ((3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1)) :=
  sorry

end number_of_solutions_eq_one_l617_617712


namespace george_painting_problem_l617_617311

theorem george_painting_problem (n : ℕ) (hn : n = 9) :
  let choices := nat.choose (n - 1) 1 in
  choices = 8 :=
by
  -- Import necessary combinatorial definitions and theorems
  have h : n - 1 = 8 := by linarith [hn],
  rw h,
  show nat.choose 8 1 = 8,
  apply nat.choose_one_right,
  sorry

end george_painting_problem_l617_617311


namespace prob_A_union_B_compl_l617_617403

open ProbabilityTheory

def outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 2, 3, 4}
def B_compl : Set ℕ := {5, 6}

def P (s : Set ℕ) : ℚ := s.card / outcomes.card

theorem prob_A_union_B_compl : P (A ∪ B_compl) = 2 / 3 :=
by
  have hA : P A = 1 / 3 := by simp [P, A, outcomes, Finset.card]
  have hB_compl : P B_compl = 1 / 3 := by simp [P, B_compl, outcomes, Finset.card]
  have h_disjoint : Disjoint A B_compl := by simp [Set.Disjoint, Set.Inter_eq_empty]
  calc
    P (A ∪ B_compl) = P A + P B_compl : eq.symm (probability_union_disjoint h_disjoint)
    ... = 1 / 3 + 1 / 3 : by rw [hA, hB_compl]
    ... = 2 / 3 : by norm_num

end prob_A_union_B_compl_l617_617403


namespace function_odd_on_domain_l617_617500

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem function_odd_on_domain :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x :=
by
  intros x h
  sorry

end function_odd_on_domain_l617_617500


namespace find_all_functions_l617_617756

noncomputable def satisfies_condition (f : ℚ → ℚ) (a b : ℚ) := 
  ∀ x y : ℚ, f(x + a + f(y)) = f(x + b) + y

theorem find_all_functions (a b : ℚ) (f : ℚ → ℚ) (A : ℚ) 
  (h : satisfies_condition f a b) : 
  (∀ x : ℚ, f(x) = A * x + (a - b) / 2) :=
begin
  sorry
end

end find_all_functions_l617_617756


namespace probability_at_least_one_exceeds_one_dollar_l617_617610

noncomputable def prob_A : ℚ := 2 / 3
noncomputable def prob_B : ℚ := 1 / 2
noncomputable def prob_C : ℚ := 1 / 4

theorem probability_at_least_one_exceeds_one_dollar :
  (1 - ((1 - prob_A) * (1 - prob_B) * (1 - prob_C))) = 7 / 8 :=
by
  -- The proof can be conducted here
  sorry

end probability_at_least_one_exceeds_one_dollar_l617_617610


namespace find_m_values_l617_617367

-- Defining the sets and conditions
def A : Set ℝ := { x | x ^ 2 - 9 * x - 10 = 0 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- Stating the proof problem
theorem find_m_values : {m | A ∪ B m = A} = {0, 1, -1 / 10} :=
by
  sorry

end find_m_values_l617_617367


namespace kids_outside_l617_617714

theorem kids_outside (s t n c : ℕ)
  (h1 : s = 644997)
  (h2 : t = 893835)
  (h3 : n = 1538832)
  (h4 : (n - s) = t) : c = 0 :=
by {
  sorry
}

end kids_outside_l617_617714


namespace range_of_a_l617_617008

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 2, x^2 - 2 * x + a ≤ 0) → (a ∈ Set.Iic (-3)) :=
begin
  sorry,
end

end range_of_a_l617_617008


namespace area_of_triangle_l617_617708

theorem area_of_triangle (a b c h1 h2 h3 S : ℝ) 
  (h1_pos : h1 > 0) (h2_pos : h2 > 0) (h3_pos : h3 > 0) 
  (area_side_height_rel : S = 1/2 * a * h1 ∧ S = 1/2 * b * h2 ∧ S = 1/2 * c * h3) 
  (semi_perimeter : ∀ s, s = (a + b + c) / 2) 
  (heron_formula : ∀ s, S = sqrt (s * (s - a) * (s - b) * (s - c))) :
  S = sqrt ((1 / h1 + 1 / h2 + 1 / h3) * (-1 / h1 + 1 / h2 + 1 / h3) * (1 / h1 - 1 / h2 + 1 / h3) * (1 / h1 + 1 / h2 - 1 / h3)) :=
sorry

end area_of_triangle_l617_617708


namespace DE_eq_CF_l617_617053

variables (Γ : Type*) [circle Γ]
variables (A B C D E F : Pt Γ)

-- Conditions
variables (acute_tri : acute_triangle A B C)
variables (AD_perp_BC : ⟪A, D, B⟫ ⊥ ⟪B, C, A⟫)
variables (AE_diameter : diameter A E Γ)
variables (F_intersection : intersection F ⟪A, E⟫ ⟪B, C⟫)
variables (angle_cond : ∠DAC = 2 * ∠DAB)

-- Goal
theorem DE_eq_CF : DE = CF :=
sorry

end DE_eq_CF_l617_617053


namespace sum_radical_conjugate_l617_617641

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617641


namespace angle_complement_supplement_l617_617726

theorem angle_complement_supplement (x : ℝ) (h1 : 90 - x = (1 / 2) * (180 - x)) : x = 90 := by
  sorry

end angle_complement_supplement_l617_617726


namespace probability_of_a_l617_617772

open ProbabilityTheory

theorem probability_of_a
  (a b : Event)
  (p : Measure Event)
  (ha : 0 ≤ p a ∧ p a ≤ 1)
  (hb : p b = 2 / 5)
  (indep : indep p a b)
  (hab : p (a ∩ b) = 0.08) :
  p a = 0.4 :=
by
  sorry

end probability_of_a_l617_617772


namespace sum_of_conjugates_l617_617654

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617654


namespace oblique_axonometric_projection_l617_617543

theorem oblique_axonometric_projection:
  ∀ (L1 L2: LineSegment), 
  parallel L1 L2 ∧ equal L1 L2 →
  parallel (intuitive_diagram L1) (intuitive_diagram L2) ∧ equal (intuitive_diagram L1) (intuitive_diagram L2) :=
sorry

end oblique_axonometric_projection_l617_617543


namespace complex_number_in_first_quadrant_l617_617755

def is_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem complex_number_in_first_quadrant (x y : ℝ) (h : x + y + (x - y) * complex.i = 3 - complex.i) : 
  is_first_quadrant (x + y * complex.i) :=
by
  sorry

end complex_number_in_first_quadrant_l617_617755


namespace find_mistake_l617_617994

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l617_617994


namespace area_of_triangle_with_medians_l617_617125

theorem area_of_triangle_with_medians 
  (ma mb mc : ℝ) 
  (h_ma : ma = 5) 
  (h_mb : mb = 6) 
  (h_mc : mc = 5) : 
  let sm := (ma + mb + mc) / 2 in
  let area := (4 / 3) * real.sqrt (sm * (sm - ma) * (sm - mb) * (sm - mc)) in
  area = 16 := 
by
  sorry

end area_of_triangle_with_medians_l617_617125


namespace cos_y_eq_neg_one_ninth_l617_617417

-- Definitions and conditions
def α : ℝ := Real.arccos (1 / 9)
def x (y : ℝ) : ℝ := y - α
def z (y : ℝ) : ℝ := y + α

-- Theorem to prove the required property
theorem cos_y_eq_neg_one_ninth (y : ℝ) :
  let a := 5 + Real.cos (x y),
      b := 5 + Real.cos y,
      c := 5 + Real.cos (z y) in
  a != b ∧ b != c ∧ a * c = b * b → 
  Real.cos y = - (1 / 9) :=
begin
  sorry
end

end cos_y_eq_neg_one_ninth_l617_617417


namespace part1_part2_l617_617231

-- Define the system of equations as assumptions
variables {x y : ℝ}
variable h₁ : x + 2 * y = 4
variable h₂ : 3 * x + y = 7

-- First part: Prove the solution to the equations is x=2 and y=1
theorem part1 : x = 2 ∧ y = 1 :=
by sorry

-- Second part: Define the wage condition and prove the company's violation
variables {a : ℝ}
variable h3 : a >= 50

-- Define the wage function
noncomputable def wage (a : ℝ) : ℝ := -8 * a + 3200

-- Prove the maximum possible wage under the constraints is less than 3000
theorem part2 : wage a < 3000 :=
by sorry

end part1_part2_l617_617231


namespace problem_l617_617370

noncomputable def A (x y : ℝ) : Set ℝ := {1, (x+y)/2 - 1}
noncomputable def B (x y : ℝ) : Set ℝ := {-Real.log (x*y), x}
noncomputable def S (x y : ℝ) : ℝ := 
  ∑ k in Finset.range 1011, (x^(2*(k+1)) - 1/y^(2*(k+1)))

theorem problem {x y : ℝ} (h1 : 0 < y) (h2 : y < 2) (h3 : A x y = B x y) : 
  S x y = 0 :=
sorry

end problem_l617_617370


namespace scholarship_amount_l617_617087

-- Definitions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def work_hours : ℕ := 200
def hourly_wage : ℕ := 10
def work_earnings : ℕ := work_hours * hourly_wage
def remaining_tuition : ℕ := tuition_per_semester - parents_contribution - work_earnings

-- Theorem to prove the scholarship amount
theorem scholarship_amount (S : ℕ) (h : 3 * S = remaining_tuition) : S = 3000 :=
by
  sorry

end scholarship_amount_l617_617087


namespace sum_of_radical_conjugates_l617_617688

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617688


namespace factorable_polynomial_l617_617521

-- Definitions for polynomials and their factorization conditions
def polynomial (x y m : ℤ) : ℤ := x^2 + (3 * x * y) + x + (m * y) - m

def is_linear_integer_factors (p : polynomial) := 
  ∃ A B C D : ℤ, 
    (A = 1 ∨ A = -1 ∧ D = 1 ∨ D = -1) ∧
    ∀ x y m : ℤ, p = (A * x + B * y + C) * (D * x + y + F)

-- The problem statement
theorem factorable_polynomial (m : ℤ) : 
  (∃ p : polynomial, is_linear_integer_factors p) ↔ (m = 0 ∨ m = 12) := 
sorry

end factorable_polynomial_l617_617521


namespace simplify_trig_expression_l617_617936

variable (α : ℝ)

-- Main goal is to prove the given expression simplifies to -√3 * cot(2 * α)
theorem simplify_trig_expression :
  (sin (2 * α - 3 * Real.pi) + 2 * cos (7 * Real.pi / 6 + 2 * α)) /
  (2 * cos (Real.pi / 6 - 2 * α) + sqrt 3 * cos (2 * α - 3 * Real.pi))
  = -sqrt 3 * cotan (2 * α) := 
sorry

end simplify_trig_expression_l617_617936


namespace find_k_l617_617781

-- Definitions for the conditions
def quad_eq (a b c : ℝ) (x : ℝ) := a*x^2 + b*x + c = 0

-- Given equations
axiom eqn1 (x k : ℝ) : quad_eq 1 k 6 x
axiom eqn2 (x k : ℝ) : quad_eq 1 (-k) 6 x

-- Condition on the roots
axiom root_condition (r s k : ℝ) : 
  ((∀ x, quad_eq 1 k 6 x → x = r ∨ x = s) ∧ 
  (∀ x, quad_eq 1 (-k) 6 x → x = r + 5 ∨ x = s + 5))

-- The theorem to prove
theorem find_k : ∃ k : ℝ, (∀ r s : ℝ, root_condition r s k) → k = 5 := sorry

end find_k_l617_617781


namespace value_of_x_l617_617737

theorem value_of_x (x : ℝ) (h : 2 ≤ |x - 3| ∧ |x - 3| ≤ 6) : x ∈ Set.Icc (-3 : ℝ) 1 ∪ Set.Icc 5 9 :=
by
  sorry

end value_of_x_l617_617737


namespace janet_total_l617_617044

-- Definitions based on the conditions
variable (initial_collect : ℕ) (sold : ℕ) (better_cond : ℕ)
variable (twice_size : ℕ)

-- The conditions from part a)
def janet_initial_collection := initial_collect = 10
def janet_sells := sold = 6
def janet_gets_better := better_cond = 4
def brother_gives := twice_size = 2 * (initial_collect - sold + better_cond)

-- The proof statement based on part c)
theorem janet_total (initial_collect sold better_cond twice_size : ℕ) : 
    janet_initial_collection initial_collect →
    janet_sells sold →
    janet_gets_better better_cond →
    brother_gives initial_collect sold better_cond twice_size →
    (initial_collect - sold + better_cond + twice_size = 24) :=
by
  intros h1 h2 h3 h4
  sorry

end janet_total_l617_617044


namespace integer_solution_range_l617_617392

theorem integer_solution_range {m : ℝ} : 
  (∀ x : ℤ, -1 ≤ x → x < m → (x = -1 ∨ x = 0)) ↔ (0 < m ∧ m ≤ 1) :=
by 
  sorry

end integer_solution_range_l617_617392


namespace total_young_fish_l617_617908

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end total_young_fish_l617_617908


namespace perimeter_of_ABCD_l617_617851

def Point := ℝ × ℝ

def square_dist (p1 p2 : Point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt (square_dist p1 p2)

theorem perimeter_of_ABCD :
  ∀ (A B C D : Point), 
  distance (Point.mk 0 0) (Point.mk 4 0) < distance (Point.mk 0 0) (Point.mk 7 3) →
  distance D C = 5 →
  (Point.mk 0 4).2 = 4 →
  (Point.mk 7 0).1 = 7 →
  distance A B + distance B C + distance C D + distance D A = 26 := by
  intros A B C D AD_lt_BC DC_eq_5 DN_eq_4 BN_eq_7
  sorry

end perimeter_of_ABCD_l617_617851


namespace max_grain_mass_l617_617601

def platform_length : ℝ := 10
def platform_width : ℝ := 5
def grain_density : ℝ := 1200
def angle_of_repose : ℝ := 45
def max_mass : ℝ := 175000

theorem max_grain_mass :
  let height_of_pile := platform_width / 2
  let volume_of_prism := platform_length * platform_width * height_of_pile
  let volume_of_pyramid := (1 / 3) * (platform_width * height_of_pile) * height_of_pile
  let total_volume := volume_of_prism + 2 * volume_of_pyramid
  let calculated_mass := total_volume * grain_density
  calculated_mass = max_mass :=
by {
  sorry
}

end max_grain_mass_l617_617601


namespace rock_paper_scissors_l617_617166

open Nat

-- Definitions based on problem conditions
def personA_movement (x y z : ℕ) : ℤ :=
  3 * (x : ℤ) - 2 * (y : ℤ) + (z : ℤ)

def personB_movement (x y z : ℕ) : ℤ :=
  3 * (y : ℤ) - 2 * (x : ℤ) + (z : ℤ)

def total_rounds (x y z : ℕ) : ℕ :=
  x + y + z

-- Problem statement
theorem rock_paper_scissors (x y z : ℕ) 
  (h1 : total_rounds x y z = 15)
  (h2 : personA_movement x y z = 17)
  (h3 : personB_movement x y z = 2) : x = 7 :=
by
  sorry

end rock_paper_scissors_l617_617166


namespace sum_of_primes_product_166_l617_617983

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m < n → m > 0 → n % m ≠ 0

theorem sum_of_primes_product_166
    (p1 p2 : ℕ)
    (prime_p1 : is_prime p1)
    (prime_p2 : is_prime p2)
    (product_condition : p1 * p2 = 166) :
    p1 + p2 = 85 :=
    sorry

end sum_of_primes_product_166_l617_617983


namespace fifteenth_odd_multiple_of_five_l617_617200

theorem fifteenth_odd_multiple_of_five :
  ∃ a : ℕ, (∀ n : ℕ, a n = 5 + (n - 1) * 10) ∧ a 15 = 145 :=
by
  let a := λ n, 5 + (n - 1) * 10
  use a
  split
  { intros n,
    refl }
  { refl }
  sorry

end fifteenth_odd_multiple_of_five_l617_617200


namespace prism_pyramid_fusion_l617_617094

theorem prism_pyramid_fusion :
  ∃ (result_faces result_edges result_vertices : ℕ),
    result_faces + result_edges + result_vertices = 28 ∧
    ((result_faces = 8 ∧ result_edges = 13 ∧ result_vertices = 7) ∨
    (result_faces = 7 ∧ result_edges = 12 ∧ result_vertices = 7)) :=
by
  sorry

end prism_pyramid_fusion_l617_617094


namespace part1_part2_l617_617877

structure Point :=
(x y : ℝ)

structure Line :=
(point1 point2 : Point)

def slope (L : Line) : ℝ :=
(L.point2.y - L.point1.y) / (L.point2.x - L.point1.x)

def perpendicular (L1 L2 : Line) : Prop :=
slope(L1) * slope(L2) = -1

variables (A B M F D N C O E H : Point)

-- Conditions are given as hypotheses
axiom M_is_intersection : ∃ t : ℝ, A.x + t * (B.x - A.x) = M.x ∧ A.y + t * (B.y - A.y) = M.y
axiom N_is_intersection : ∃ t : ℝ, F.x + t * (D.x - F.x) = N.x ∧ F.y + t * (D.y - F.y) = N.y

-- Lines are defined by points
def OB : Line := Line.mk O B
def DF : Line := Line.mk D F
def OC : Line := Line.mk O C
def DE : Line := Line.mk D E
def OH : Line := Line.mk O H
def MN : Line := Line.mk M N

-- Prove perpendicularities
theorem part1: perpendicular OB DF ∧ perpendicular OC DE :=
sorry

theorem part2: perpendicular OH MN :=
sorry

end part1_part2_l617_617877


namespace not_necessarily_isosceles_l617_617462

open Function

theorem not_necessarily_isosceles 
  (ABC : Type) [triangle ABC] 
  (P : point) (inside_ABC : inside_triangle P ABC) 
  (connected_to_vertices : connected_to_vertices P ABC) 
  (perpendiculars_dropped : perpendiculars_to_sides P ABC) 
  (four_equal_triangles : ∃ (t1 t2 t3 t4 : subtriangle), (∀ t1 t2 t3 t4, congruent t1 t2 t3 t4)) 
  : ¬ isosceles_triangle ABC := 
sorry

end not_necessarily_isosceles_l617_617462


namespace remainder_of_polynomial_l617_617300

def f (x : ℝ) := 9 * x^3 - 8 * x^2 + 3 * x - 4

theorem remainder_of_polynomial : 
  f(3) = 176 := by
  sorry

end remainder_of_polynomial_l617_617300


namespace exercise1_exercise2_exercise3_l617_617765

-- Definitions corresponding to the conditions in step a)
def major_axis_length : ℝ := 10
def minor_axis_length : ℝ := 8
def foci_on_y_axis : Prop := true

-- Definitions related to the ellipse
def ellipse := {e // (e.a = major_axis_length / 2) ∧ (e.b = minor_axis_length / 2) ∧ (foci_on_y_axis)}
def ellipse_equation (e : ellipse) : Prop :=
  ∃ (a b : ℝ), a = 5 ∧ b = 4 ∧ (e.equation = x^2 / b^2 + y^2 / a^2 = 1)

-- Definitions related to the foci and eccentricity of the ellipse
def foci (e : ellipse) : Prop :=
  ∃ (c : ℝ), c = sqrt (e.a^2 - e.b^2) ∧ 
  ((e.foci_left = (0, -c)) ∧ (e.foci_right = (0, c))) ∧ 
  (e.eccentricity = c / e.a)

-- Definitions related to the hyperbola
def hyperbola (e : ellipse) : Prop :=
  ∃ (a b c : ℝ), a = 3 ∧ c = 5 ∧ b = sqrt (c^2 - a^2) ∧ (a = e.c ∧ b = e.a) ∧ 
  (h.equation = y^2 / a^2 - x^2 / b^2 = 1)

-- Statement to prove the standard equation of the ellipse
theorem exercise1 :
  ∃ e : ellipse, ellipse_equation e := sorry

-- Statement to prove the coordinates of the foci of the ellipse and its eccentricity
theorem exercise2 :
  ∃ e : ellipse, foci e := sorry

-- Statement to prove the standard equation of the hyperbola
theorem exercise3 :
  ∃ h : hyperbola e, hyperbola h := sorry

end exercise1_exercise2_exercise3_l617_617765


namespace perimeter_of_ABCD_l617_617849

theorem perimeter_of_ABCD
  (AD BC AB CD : ℕ)
  (hAD : AD = 4)
  (hAB : AB = 5)
  (hBC : BC = 10)
  (hCD : CD = 7)
  (hAD_lt_BC : AD < BC) :
  AD + AB + BC + CD = 26 :=
by
  -- Proof will be provided here.
  sorry

end perimeter_of_ABCD_l617_617849


namespace thread_length_l617_617637

def side_length : ℕ := 13

def perimeter (s : ℕ) : ℕ := 4 * s

theorem thread_length : perimeter side_length = 52 := by
  sorry

end thread_length_l617_617637


namespace A_equivalence_l617_617304

noncomputable def A (x : ℝ) : ℝ :=
  (x + 2 * real.sqrt (2 * x - 4))^(-1/2) + (x - 2 * real.sqrt (2 * x - 4))^(-1/2)

theorem A_equivalence (x : ℝ) : 
  (A x = if x > 4 then 2 * real.sqrt (x - 2) / (x - 4)
         else if 2 ≤ x ∧ x < 4 then 2 * real.sqrt 2 / (4 - x) 
         else 0) := 
by
  sorry

end A_equivalence_l617_617304


namespace polynomial_multiple_of_six_l617_617895

theorem polynomial_multiple_of_six 
  (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) :
  6 ∣ P.eval 5 :=
sorry

end polynomial_multiple_of_six_l617_617895


namespace distribute_paper_clips_l617_617448

theorem distribute_paper_clips (total_paper_clips boxes : ℕ) (h_total : total_paper_clips = 81) (h_boxes : boxes = 9) : total_paper_clips / boxes = 9 := by
  sorry

end distribute_paper_clips_l617_617448


namespace fifteenth_odd_multiple_of_5_l617_617192

theorem fifteenth_odd_multiple_of_5 :
  (∃ n: ℕ, n = 15 ∧ (10 * n - 5 = 145)) :=
begin
  use 15,
  split,
  { refl },
  { norm_num }
end

end fifteenth_odd_multiple_of_5_l617_617192


namespace solution_set_of_inequality_l617_617522

theorem solution_set_of_inequality:
  {x : ℝ | 1 < abs (2 * x - 1) ∧ abs (2 * x - 1) < 3} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ 
  {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l617_617522


namespace combined_tax_rate_correct_l617_617020

def income_deduction (income : ℝ) (deduction_rate : ℝ) : ℝ :=
  income * (1 - deduction_rate)

def tax_after_deduction (income : ℝ) (deduction_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  tax_rate * (income_deduction income deduction_rate)

def effective_tax (income : ℝ) (tax_after : ℝ) (tax_break_rate : ℝ) : ℝ :=
  tax_after * (1 - tax_break_rate)

def combined_tax_rate (mork_income mindy_income bickley_income exidor_income: ℝ) : ℝ :=
  let mork_tax := tax_after_deduction mork_income 0.10 0.45
  let mindy_tax := effective_tax mindy_income (0.20 * mindy_income) 0.05
  let bickley_tax := tax_after_deduction bickley_income 0.07 0.25
  let exidor_tax := effective_tax (exidor_income / 2) (0.30 * (exidor_income / 2)) 0.08
  let total_income := mork_income + mindy_income + bickley_income + exidor_income
  let total_tax := mork_tax + mindy_tax + bickley_tax + exidor_tax
  total_tax / total_income

theorem combined_tax_rate_correct (X : ℝ) : 
  combined_tax_rate X (4 * X) (2 * X) (X / 2) = 0.2357 := by
  -- sorry used to indicate we skip the detailed proof
  sorry

end combined_tax_rate_correct_l617_617020


namespace bird_families_flew_away_to_africa_l617_617212

variable (A : ℕ)

/-- Number of bird families flew away to Asia -/
def AsiaFamilies : ℕ := 80

/-- Total number of bird families flew away for the winter -/
def TotalFamilies : ℕ := 118

/-- Number of bird families flew away to Africa -/
def AfricaFamilies (A : ℕ) : Prop := A + AsiaFamilies = TotalFamilies

theorem bird_families_flew_away_to_africa : AfricaFamilies 38 :=
by
  unfold AfricaFamilies
  rw [AsiaFamilies, TotalFamilies]
  norm_num

-- Placeholder for the actual proof
sorry

end bird_families_flew_away_to_africa_l617_617212


namespace midpoints_and_perpendiculars_concyclic_l617_617118

-- Define a cyclic quadrilateral and its properties
variables {A B C D O X Y Z W E F G H : Type}

-- Hypothesize the cyclic property and perpendicular diagonals
axioms
  (is_cyclic_quadrilateral : CyclicQuadrilateral A B C D)
  (O_intersection : ∃ O, IsIntersectionOfDiagonals O A B C D)
  (perpendicular_diagonals : ∀ O, PerpendicularDiagonals O A C B D)

-- Define the midpoints and perpendiculars
def midpoint (X Y : Type) : Type := E
def foot_of_perpendicular (O X : Type) : Type := X

-- Define the conditions for the midpoints and feet of perpendiculars
axioms
  (E_midpoint : midpoint A B E)
  (F_midpoint : midpoint B C F)
  (G_midpoint : midpoint C D G)
  (H_midpoint : midpoint D A H)
  (OX_perpendicular : foot_of_perpendicular O A B X)
  (OY_perpendicular : foot_of_perpendicular O B C Y)
  (OZ_perpendicular : foot_of_perpendicular O C D Z)
  (OW_perpendicular : foot_of_perpendicular O D A W)

-- The goal to prove they are concyclic
theorem midpoints_and_perpendiculars_concyclic :
  Concyclic E F G H X Y Z W :=
by sorry

end midpoints_and_perpendiculars_concyclic_l617_617118


namespace octagon_interior_angle_l617_617627

theorem octagon_interior_angle (n : ℕ) (h : n = 8) : (n - 2) * 180 / n = 135 :=
by
  have h1 : (8 - 2) * 180 / 8 = 135 := sorry,
  exact h1

end octagon_interior_angle_l617_617627


namespace unique_real_c_for_magnitude_eq_one_l617_617307

theorem unique_real_c_for_magnitude_eq_one 
  (c : ℝ) : ∃! c, abs(1 - 2 * complex.I - (c - 3 * complex.I)) = 1 :=
sorry

end unique_real_c_for_magnitude_eq_one_l617_617307


namespace find_number_of_men_l617_617033

noncomputable def initial_conditions (x : ℕ) : ℕ × ℕ :=
  let men := 4 * x
  let women := 5 * x
  (men, women)

theorem find_number_of_men (x : ℕ) : 
  let (initial_men, initial_women) := initial_conditions x in
  let men_after_entry := initial_men + 2 in
  let women_after_leaving := initial_women - 3 in
  2 * women_after_leaving = 24 →
  men_after_entry = 14 :=
by
  intros
  sorry

end find_number_of_men_l617_617033


namespace dice_probability_l617_617208

theorem dice_probability :
  let fair_6_sided_dice := (finset.range 6).image (+1)
  let dice_rolls := list.replicate 9 fair_6_sided_dice.cartesian_product
  let sum_is_at_least_18 (l : list ℕ) := l.sum ≥ 18
  let at_most_two_six (l : list ℕ) := l.count 6 ≤ 2
  let valid_rolls := dice_rolls.filter (λ l, sum_is_at_least_18 l ∧ at_most_two_six l)
  let m := valid_rolls.card in
  m = 13662 :=
sorry

end dice_probability_l617_617208


namespace sum_of_number_and_radical_conjugate_l617_617676

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617676


namespace roots_of_unity_count_l617_617278

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 3)
noncomputable def omega2 := Complex.exp (-2 * Real.pi * Complex.I / 3)

theorem roots_of_unity_count (c d : ℤ) (h₁ : c = 0) (h₂ : d = -1) :
  (z : Complex) → (z^3 + c * z + d = 0) → (∃ n : ℕ, z = 1 ∨ z = omega ∨ z = omega2) :=
begin
  sorry
end

end roots_of_unity_count_l617_617278


namespace equation_of_parallel_line_equation_of_perpendicular_bisector_l617_617709

-- Conditions and definitions for the first part
def point_P : ℝ × ℝ := (-1, 3)
def original_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- First part: Prove the equation of the line through P parallel to the original line
theorem equation_of_parallel_line (x y : ℝ) : 
  (original_line x y → ∃ c, x - 2 * y + c = 0 ∧ (-1, 3) ∈ c ∧ c = 7) :=
sorry

-- Conditions and definitions for the second part
def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (3, 1)

-- Second part: Prove the equation of the perpendicular bisector of line segment AB
theorem equation_of_perpendicular_bisector (x y : ℝ) : 
  ∃ c, (∃ Mx, Mx = (1 + 3) / 2) ∧ 
        (∃ My, My = (2 + 1) / 2) ∧ 
        x - y/2 - c = 0 ∧ 
        (2, 1.5) ∈ c ∧ 
        c = 4 :=
sorry

end equation_of_parallel_line_equation_of_perpendicular_bisector_l617_617709


namespace minimize_dwarf_risk_l617_617959

noncomputable def dwarf_hat_risk (p : ℕ) : ℕ := 
if p = 0 then 0 else 1

theorem minimize_dwarf_risk (p : ℕ) (hp: p < ∞) : 
  dwarf_hat_risk p = 1 := 
by
  sorry

end minimize_dwarf_risk_l617_617959


namespace geometric_sequence_a3_a5_l617_617853

variable {a : ℕ → ℝ}

theorem geometric_sequence_a3_a5 (h₀ : a 1 > 0) 
                                (h₁ : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16) : 
                                a 3 + a 5 = 4 := 
sorry

end geometric_sequence_a3_a5_l617_617853


namespace red_markers_count_l617_617698

-- Define the given conditions
def blue_markers : ℕ := 1028
def total_markers : ℕ := 3343

-- Define the red_makers calculation based on the conditions
def red_markers (total_markers blue_markers : ℕ) : ℕ := total_markers - blue_markers

-- Prove that the number of red markers is 2315 given the conditions
theorem red_markers_count : red_markers total_markers blue_markers = 2315 := by
  -- We can skip the proof for this demonstration
  sorry

end red_markers_count_l617_617698


namespace sum_of_intersection_coordinates_l617_617265

theorem sum_of_intersection_coordinates :
  let y := fun x : ℝ => (x - 2)^2
  let x := fun y : ℝ => (y + 2)^2 - 6
  (∑ i in roots (fun x => x^4 - 8 * x^3 + 28 * x^2 - 49 * x + 30), id i) +
  (∑ i in roots (fun y => y^4 + 8 * y^3 + 12 * y^2 - 17 * y + 4), id i) = 0 :=
by
  -- Statement requires a proof, the lean term will end with sorry to skip the proof.
  sorry

end sum_of_intersection_coordinates_l617_617265


namespace n_pow_one_over_n_bounded_smallest_k_for_n_pow_one_over_n_l617_617931

theorem n_pow_one_over_n_bounded (n : ℕ) (h_pos : 0 < n) : 
  1 ≤ n^(1/n : ℝ) ∧ n^(1/n : ℝ) ≤ 2 :=
sorry

theorem smallest_k_for_n_pow_one_over_n (n : ℕ) (h_pos : 0 < n) : 
  ∃ k : ℝ, (∀ n : ℕ, 0 < n → 1 ≤ n^(1/n : ℝ) ∧ n^(1/n : ℝ) ≤ k) ∧ k = real.cbrt 3 :=
sorry

end n_pow_one_over_n_bounded_smallest_k_for_n_pow_one_over_n_l617_617931


namespace find_x_l617_617343

theorem find_x (x : ℝ) (h : x ∈ set.Icc 0 real.pi) (h_eq : real.sin (x + real.sin x) = real.cos (x - real.cos x)) : x = real.pi / 4 :=
sorry

end find_x_l617_617343


namespace solve_cos_sin_eq_l617_617938

noncomputable def cosine_of_sine (x : ℝ) : ℝ := Real.cos (Real.sin x)

theorem solve_cos_sin_eq (x : ℝ) (h1 : -3 * Real.pi / 2 ≤ x) (h2 : x ≤ 3 * Real.pi / 2) :
  cosine_of_sine x = 3 * x / 2 → x ≈ -0.6527 ∨ x ≈ 0 ∨ x ≈ 0.6527 :=
by
  sorry

end solve_cos_sin_eq_l617_617938


namespace faye_coloring_books_l617_617291

theorem faye_coloring_books (x : ℕ) : 34 - x + 48 = 79 → x = 3 :=
by
  sorry

end faye_coloring_books_l617_617291


namespace gcd_of_elements_in_T_is_one_l617_617425

open Set

variables {n : ℕ} (U : Finset ℕ) (S T : Finset ℕ) (s d : ℕ)

noncomputable def greatest_common_divisor := Finset.gcd

theorem gcd_of_elements_in_T_is_one (hU_sub : U = Finset.range (n + 1)) (hS_sub : S ⊆ U)
  (hT_sub : T ⊆ U) (hS_nonempty : S.nonempty) (hs_def : greatest_common_divisor S = s)
  (hs_not_one : s ≠ 1) (hd_def : d = Finset.min' (Finset.filter (fun x => x > 1 ∧ s % x = 0) (Finset.range (s + 1))) (by sorry))
  (hT_size : ∥T∥ ≥ 1 + n / d) :
  greatest_common_divisor T = 1 := by
  sorry

end gcd_of_elements_in_T_is_one_l617_617425


namespace positive_rationals_in_set_l617_617270

def a_seq : ℕ → ℕ
| 0          := 1
| (2*n + 1)  := a_seq n
| (2*n + 2)  := a_seq n + a_seq (n + 1)

theorem positive_rationals_in_set : 
  ∀ (p q : ℕ), p > 0 → q > 0 → (nat.coprime p q) → ∃ n ≥ 1, p * a_seq n = q * a_seq (n - 1) :=
sorry

end positive_rationals_in_set_l617_617270


namespace janet_total_l617_617042

-- Definitions based on the conditions
variable (initial_collect : ℕ) (sold : ℕ) (better_cond : ℕ)
variable (twice_size : ℕ)

-- The conditions from part a)
def janet_initial_collection := initial_collect = 10
def janet_sells := sold = 6
def janet_gets_better := better_cond = 4
def brother_gives := twice_size = 2 * (initial_collect - sold + better_cond)

-- The proof statement based on part c)
theorem janet_total (initial_collect sold better_cond twice_size : ℕ) : 
    janet_initial_collection initial_collect →
    janet_sells sold →
    janet_gets_better better_cond →
    brother_gives initial_collect sold better_cond twice_size →
    (initial_collect - sold + better_cond + twice_size = 24) :=
by
  intros h1 h2 h3 h4
  sorry

end janet_total_l617_617042


namespace cyclic_quadrilateral_maximizes_area_l617_617465

noncomputable def max_area_of_cyclic_quadrilateral (a b c d e : ℝ) (β δ : ℝ) : ℝ :=
  let t := (1 / 2) * a * b * Real.sin β + (1 / 2) * c * d * Real.sin δ
  t

theorem cyclic_quadrilateral_maximizes_area (a b c d e β δ : ℝ) :
  (a^2 + b^2) - (c^2 + d^2) = 2 * a * b * Real.cos β - 2 * c * d * Real.cos δ →
  (a * b * Real.sin β) + (c * d * Real.sin δ) ≠ 0 →
  (∀ t, t ≤ max_area_of_cyclic_quadrilateral a b c d e β δ) :=
begin
  sorry
end

end cyclic_quadrilateral_maximizes_area_l617_617465


namespace circumcenter_tangent_implies_equidistance_l617_617879

-- Lean does not have direct support for geometric configurations like this
-- without more setup, so this statement provides the high-level structure.

theorem circumcenter_tangent_implies_equidistance
  (A B C O P Q M K L : Type)
  [triangle_ABC : triangle A B C]
  [circumcenter O A B C]
  [P_on_AC : point_on_side P A C]
  [Q_on_AB : point_on_side Q A B]
  [K_midpoint_PB : midpoint K P B]
  [L_midpoint_QC : midpoint L Q C]
  [M_midpoint_PQ : midpoint M P Q]
  (circle_KLM_tangent_PQ : tangent_circle_line M K L P Q) :
  distance O P = distance O Q :=
by
  -- The high-level structure of the proof goes here.
  -- The exact geometric constructions and proof in a Lean formal proof assistant
  -- would require deeper encoding of geometric axioms and definitions.
  sorry

end circumcenter_tangent_implies_equidistance_l617_617879


namespace average_marks_correct_l617_617216

-- Define the marks obtained in each subject
def english_marks := 86
def mathematics_marks := 85
def physics_marks := 92
def chemistry_marks := 87
def biology_marks := 95

-- Calculate total marks and average marks
def total_marks := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects := 5
def average_marks := total_marks / num_subjects

-- Prove that Dacid's average marks are 89
theorem average_marks_correct : average_marks = 89 := by
  sorry

end average_marks_correct_l617_617216


namespace percentage_shaded_is_18_75_l617_617408

-- conditions
def total_squares: ℕ := 16
def shaded_squares: ℕ := 3

-- claim to prove
theorem percentage_shaded_is_18_75 :
  ((shaded_squares : ℝ) / total_squares) * 100 = 18.75 := 
by
  sorry

end percentage_shaded_is_18_75_l617_617408


namespace count_statements_implying_l617_617263

-- Define the propositions p, q, and r
variables p q r : Prop

-- Define each given statement as a logical proposition
def statement1 : Prop := p ∧ ¬q ∧ ¬r
def statement2 : Prop := ¬p ∧ ¬q ∧ ¬r
def statement3 : Prop := p ∧ ¬q ∧ r
def statement4 : Prop := ¬p ∧ q ∧ ¬r

-- Define the implication that we need to check for each statement
def implies_implication (s : Prop) : Prop := s → ((p → q) → r)

-- Define a condition to check if the implication is true for a given statement
def is_true_implication (s : Prop) : Prop := implies_implication s

-- Prove that exactly 2 of the 4 statements imply the truth of ((p → q) → r)
theorem count_statements_implying :
  (is_true_implication statement1 → true) +
  (is_true_implication statement2 → true) +
  (is_true_implication statement3 → true) +
  (is_true_implication statement4 → true) = 2 := 
sorry

end count_statements_implying_l617_617263


namespace max_2b_div_a_l617_617385

theorem max_2b_div_a (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : 
  ∃ max_val, max_val = (2 * b) / a ∧ max_val = (32 / 3) :=
by
  sorry

end max_2b_div_a_l617_617385


namespace minimal_diagonals_l617_617623

def consecutive_labels (labels : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i < n - 1 → |labels (i + 1) - labels i| ≤ 1

def draw_diagonal (labels : ℕ → ℝ) (n : ℕ) (i j : ℕ) : Prop :=
  i ≠ j ∧ |labels i - labels j| ≤ 1

def count_diagonals (labels : ℕ → ℝ) (n : ℕ) : ℕ :=
  ((n.choose 2) - n) 

theorem minimal_diagonals (labels : ℕ → ℝ) (n : ℕ) (h1 : consecutive_labels labels n) (h2 : n = 2021) :
  count_diagonals labels n = 2018 :=
sorry

end minimal_diagonals_l617_617623


namespace halfway_fraction_l617_617175

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l617_617175


namespace greatest_n_divisibility_l617_617384

noncomputable section

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def five_factorial := factorial 5
def ten_factorial := factorial 10

theorem greatest_n_divisibility :
  ∃ n : Nat, (10! - 2 * (5!)^2) % 10^n = 0 ∧ ∀ m : Nat, (10! - 2 * (5!)^2) % 10^m ≠ 0 → n ≥ m :=
  by sorry

end greatest_n_divisibility_l617_617384


namespace convex_polyhedron_theorems_l617_617593

-- Definitions for convex polyhedron and symmetric properties
structure ConvexSymmetricPolyhedron (α : Type*) :=
  (isConvex : Bool)
  (isCentrallySymmetric : Bool)
  (crossSection : α → α → α)
  (center : α)

-- Definitions for proofs required
def largest_cross_section_area
  (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ∀ (p : ℝ), P.crossSection p P.center ≤ P.crossSection P.center P.center

def largest_radius_circle (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ¬∀ (p : ℝ), P.crossSection p P.center = P.crossSection P.center P.center

-- The theorem combining both statements
theorem convex_polyhedron_theorems
  (P : ConvexSymmetricPolyhedron ℝ) :
  P.isConvex = true ∧ 
  P.isCentrallySymmetric = true →
  (largest_cross_section_area P) ∧ (largest_radius_circle P) :=
by 
  sorry

end convex_polyhedron_theorems_l617_617593


namespace possible_c_value_l617_617361

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem possible_c_value (a b c : ℝ) 
  (h1 : f (-1) a b c = f (-2) a b c) 
  (h2 : f (-2) a b c = f (-3) a b c) 
  (h3 : 0 ≤ f (-1) a b c) 
  (h4 : f (-1) a b c ≤ 3) : 
  6 ≤ c ∧ c ≤ 9 := sorry

end possible_c_value_l617_617361


namespace find_y_plus_one_over_y_l617_617339

variable (y : ℝ)

theorem find_y_plus_one_over_y (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 :=
by
  sorry

end find_y_plus_one_over_y_l617_617339


namespace latus_rectum_l617_617364

noncomputable def parabola : ℝ → ℝ := λ x, 4 * x^2

theorem latus_rectum (x : ℝ) : 
  let y := parabola x in y = 4 * x^2 → y = -1/16 :=
sorry

end latus_rectum_l617_617364


namespace determine_fx_solution_l617_617319

noncomputable def f : ℝ → ℝ := sorry

theorem determine_fx_solution (x : ℝ) (h : x > 0)
  (hf : ∀ x, f(x) ∈ set.Ioi(0))
  (h_cond : ∀ x, f(1/x) ≥ x + real.sqrt(x^2 + 1)) :
  f(x) ≥ (1 + real.sqrt(1 + x^2)) / x :=
sorry

end determine_fx_solution_l617_617319


namespace simplify_radical_l617_617473

theorem simplify_radical (x : ℝ) : 
  (real.sqrt (12 * x) * real.sqrt (18 * x) * real.sqrt (27 * x) = 54 * x * real.sqrt x) :=
by 
  sorry

end simplify_radical_l617_617473


namespace profit_equation_example_l617_617226

noncomputable def profit_equation (a b : ℝ) (x : ℝ) : Prop :=
  a * (1 + x) ^ 2 = b

theorem profit_equation_example :
  profit_equation 250 360 x :=
by
  have : 25 * (1 + x) ^ 2 = 36 := sorry
  sorry

end profit_equation_example_l617_617226


namespace doctors_to_lawyers_ratio_l617_617836

theorem doctors_to_lawyers_ratio
  (d l : ℕ)
  (h1 : (40 * d + 55 * l) / (d + l) = 45)
  (h2 : d + l = 20) :
  d / l = 2 :=
by sorry

end doctors_to_lawyers_ratio_l617_617836


namespace domain_of_log_composition_l617_617559

open Real -- open the real numbers namespace

theorem domain_of_log_composition (x : ℝ) : (\forall x > 7, ∃ y, y = log 2 (log 5 (log 7 x))) :=
by
  sorry

end domain_of_log_composition_l617_617559


namespace smaller_angle_at_230_l617_617552

noncomputable def hourAngle (h m : ℕ) : ℝ := 30 * h + (m / 60) * 30
noncomputable def minuteAngle (m : ℕ) : ℝ := 6 * m

theorem smaller_angle_at_230 :
  let θ_hour := hourAngle 2 30 in
  let θ_minute := minuteAngle 30 in
  |θ_minute - θ_hour| = 105 :=
by 
  sorry

end smaller_angle_at_230_l617_617552


namespace igor_reach_top_time_l617_617116

-- Define the conditions
def cabins_numbered_consecutively := (1, 99)
def igor_initial_cabin := 42
def first_aligned_cabin := 13
def second_aligned_cabin := 12
def alignment_time := 15
def total_cabins := 99
def expected_time := 17 * 60 + 15

-- State the problem as a theorem
theorem igor_reach_top_time :
  ∃ t, t = expected_time ∧
  -- Assume the cabins are numbered consecutively
  cabins_numbered_consecutively = (1, total_cabins) ∧
  -- Igor starts in cabin #42
  igor_initial_cabin = 42 ∧
  -- Cabin #42 first aligns with cabin #13, then aligns with cabin #12, 15 seconds later
  first_aligned_cabin = 13 ∧
  second_aligned_cabin = 12 ∧
  alignment_time = 15 :=
sorry

end igor_reach_top_time_l617_617116


namespace distribute_items_l617_617439

open Nat

def g (n k : ℕ) : ℕ :=
  -- This is a placeholder for the actual function definition
  sorry

theorem distribute_items (n k : ℕ) (h : n ≥ k ∧ k ≥ 2) :
  g (n + 1) k = k * g n (k - 1) + k * g n k :=
by
  sorry

end distribute_items_l617_617439


namespace minimal_diagonals_l617_617621

def consecutive_labels (labels : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i < n - 1 → |labels (i + 1) - labels i| ≤ 1

def draw_diagonal (labels : ℕ → ℝ) (n : ℕ) (i j : ℕ) : Prop :=
  i ≠ j ∧ |labels i - labels j| ≤ 1

def count_diagonals (labels : ℕ → ℝ) (n : ℕ) : ℕ :=
  ((n.choose 2) - n) 

theorem minimal_diagonals (labels : ℕ → ℝ) (n : ℕ) (h1 : consecutive_labels labels n) (h2 : n = 2021) :
  count_diagonals labels n = 2018 :=
sorry

end minimal_diagonals_l617_617621


namespace perimeter_of_sector_l617_617328

def central_angle_deg := 54
def radius_cm := 20
def central_angle_rad := central_angle_deg * Real.pi / 180

theorem perimeter_of_sector : 
  let l := central_angle_rad * radius_cm in
  let perimeter := l + 2 * radius_cm in
  perimeter = 6 * Real.pi + 40 :=
by
  sorry

end perimeter_of_sector_l617_617328


namespace sum_radical_conjugate_l617_617644

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617644


namespace tiling_polygons_l617_617841

theorem tiling_polygons (n : ℕ) (h1 : 2 < n) (h2 : ∃ x : ℕ, x * (((n - 2) * 180 : ℝ) / n) = 360) :
  n = 3 ∨ n = 4 ∨ n = 6 := 
by
  sorry

end tiling_polygons_l617_617841


namespace sum_of_squares_of_multiplicities_le_l617_617443

-- Define the setup for the problem
variable {α : Type*}
variable (S : Finset (EuclideanSpace ℝ fin2))
variable (n : ℕ) (hS : S.card = n)
variable (hNoFourCollinear : ∀ (t : Finset (EuclideanSpace ℝ fin2)), t.card = 4 → (∀ p₁ p₂ p₃ p₄ ∈ t, ¬ affineIndependent ℝ ![p₁, p₂, p₃, p₄]))
variable (distances : Finset ℝ) (mds : Finset ℝ → ℤ)

-- Definition of multiplicity
def multiplicity (d : ℝ) : ℕ := card { p q | p ∈ S ∧ q ∈ S ∧ dist p q = d } / 2

-- Main theorem statement
theorem sum_of_squares_of_multiplicities_le : 
  (∑ d in distances, (multiplicity S d) ^ 2) ≤ n ^ 3 - n ^ 2 := 
begin
  sorry
end

end sum_of_squares_of_multiplicities_le_l617_617443


namespace Dima_can_determine_count_with_17_indices_l617_617929

theorem Dima_can_determine_count_with_17_indices : ∀ (num : ℕ), 1 ≤ num ∧ num ≤ 100 →
  ∃ indices : fin 17 → fin num, ∀ (f : fin num → ℕ), 
    (∃ g : fin num → ℕ, (∀ i : fin 17, f (indices i) = g (indices i)) → ∀ i, f i = g i) :=
by sorry

end Dima_can_determine_count_with_17_indices_l617_617929


namespace sum_of_number_and_conjugate_l617_617666

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617666


namespace solve_y_l617_617279

theorem solve_y (y : ℝ) (h : (y ^ (7 / 8)) = 4) : y = 2 ^ (16 / 7) :=
sorry

end solve_y_l617_617279


namespace k_value_if_perfect_square_l617_617004

theorem k_value_if_perfect_square (k : ℤ) (x : ℝ) (h : ∃ (a : ℝ), x^2 + k * x + 25 = a^2) : k = 10 ∨ k = -10 := by
  sorry

end k_value_if_perfect_square_l617_617004


namespace increasing_interval_l617_617968

def f (x : ℝ) : ℝ := 2^(x^2 + x - 3)

theorem increasing_interval : ∀ x, x ∈ Ioi (- (1 / 2)) → ∀ y, y ∈ Ioi x → f x < f y :=
by
  -- Mathematically equivalent statement to show f is strictly increasing in the interval (-1/2, +∞)
  sorry

end increasing_interval_l617_617968


namespace total_jail_time_proof_l617_617069

noncomputable def total_jail_time 
  (A : ℕ) (B : ℕ) (P : ℕ) (G : ℕ) (V : ℕ)
  (a : ℚ) (b : ℚ) (p : ℚ) (g : ℚ) (v : ℚ) : ℚ :=
  A * a + B * b + P * p + G * g + V * v

theorem total_jail_time_proof :
  let A := 3
  let B := 2
  let P := 6 * B
  let G := 1
  let V := 4
  let a := 42
  let b := 24
  let p := (5 / 8) * b
  let g := (3.5) * p
  let v := 10.5
  total_jail_time A B P G V a b p g v = 448.5 := 
by {
  unfold total_jail_time,
  sorry
}

end total_jail_time_proof_l617_617069


namespace ceil_square_of_neg_five_thirds_l617_617287

theorem ceil_square_of_neg_five_thirds : Int.ceil ((-5 / 3:ℚ)^2) = 3 := by
  sorry

end ceil_square_of_neg_five_thirds_l617_617287


namespace find_parameters_single_nonnegative_root_l617_617747

noncomputable def has_single_nonnegative_root (a : ℝ) : Prop :=
  let discriminant := -a^2 + 2*a + 3 in
  if a = 1 then true
  else if discriminant = 0 then true
  else if discriminant > 0 ∧ (a > 1 ∨ a < -1) then true
  else false

theorem find_parameters_single_nonnegative_root :
  {a : ℝ | has_single_nonnegative_root a} = {a | -1 ≤ a ∧ a ≤ 1 ∨ a = 3} :=
by
  sorry

end find_parameters_single_nonnegative_root_l617_617747


namespace data_set_range_l617_617137

theorem data_set_range : 
  let data_set := {0, -1, 3, 2, 4} in
  range(data_set) = 5 := sorry

end data_set_range_l617_617137


namespace mojave_population_prediction_l617_617976

def initial_population : ℕ := 4000

def growth_first_decade : ℚ := 2
def growth_second_decade : ℚ := 1.75
def growth_third_decade : ℚ := 2.5
def growth_after_third_decade : ℚ := 1.2
def growth_next_five_years : ℚ := 1.4

def population_first_decade := initial_population * growth_first_decade
def population_second_decade := population_first_decade * growth_second_decade
def population_third_decade := population_second_decade * growth_third_decade
def population_after_third_decade := population_third_decade * growth_after_third_decade
def predicted_population := population_after_third_decade * growth_next_five_years

theorem mojave_population_prediction : predicted_population = 58800 := 
by
  -- Unfold definitions
  unfold predicted_population 
  unfold population_after_third_decade 
  unfold population_third_decade 
  unfold population_second_decade 
  unfold population_first_decade 
  unfold growth_next_five_years 
  unfold growth_after_third_decade 
  unfold growth_third_decade 
  unfold growth_second_decade 
  unfold growth_first_decade 
  unfold initial_population
  have calc1 : 4000 * 2 = 8000 := by norm_num
  have calc2 : 8000 * 1.75 = 14000 := by norm_num
  have calc3 : 14000 * 2.5 = 35000 := by norm_num
  have calc4 : 35000 * 1.2 = 42000 := by norm_num
  have calc5 : 42000 * 1.4 = 58800 := by norm_num
  exact calc5

end mojave_population_prediction_l617_617976


namespace luncheon_cost_l617_617117

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + p = 3.00)
  (h2 : 5 * s + 8 * c + p = 5.40)
  (h3 : 3 * s + 4 * c + p = 3.60) :
  2 * s + 2 * c + p = 2.60 :=
sorry

end luncheon_cost_l617_617117


namespace part_a_part_b_l617_617918

-- Geometry definitions and setup
variables {ABC : Type} [triangle ABC]
variables {M : point} {B1 C1 O : point} (varphi : real)

-- Conditions
axiom isosceles_triangle_ext_AC1B (ABC : triangle) (varphi : real) : is_isosceles_triangle_ext ABC.A ABC.C ABC.B varphi
axiom isosceles_triangle_ext_AB1C (ABC : triangle) (varphi : real) : is_isosceles_triangle_ext ABC.A ABC.B ABC.C varphi
axiom M_on_median_AA1 (M B1 C1 : point) : M_on_median AA1 M
axiom M_equidistant_B1_C1 (M B1 C1 : point) : equidistant M B1 C1
axiom O_on_perp_bisector_BC (O B1 C1 : point) : O_on_perp_bisector BC O
axiom O_equidistant_B1_C1 (O B1 C1 : point) : equidistant O B1 C1

-- Part (a) statement
theorem part_a (ABC : triangle) (M B1 C1 : point) (varphi : real) 
  [isosceles_triangle_ext_AC1B ABC varphi] [isosceles_triangle_ext_AB1C ABC varphi]
  [M_on_median_AA1 M B1 C1] [M_equidistant_B1_C1 M B1 C1] : angle B1 M C1 = varphi := 
sorry

-- Part (b) statement
theorem part_b (ABC : triangle) (O B1 C1 : point) (varphi : real) 
  [isosceles_triangle_ext_AC1B ABC varphi] [isosceles_triangle_ext_AB1C ABC varphi]
  [O_on_perp_bisector_BC O B1 C1] [O_equidistant_B1_C1 O B1 C1] : angle B1 O C1 = 180 - varphi := 
sorry

end part_a_part_b_l617_617918


namespace find_k_value_l617_617746

theorem find_k_value (k : ℕ) :
  3 * 6 * 4 * k = Nat.factorial 8 → k = 560 :=
by
  sorry

end find_k_value_l617_617746


namespace productivity_increase_l617_617985

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : (7/8) * b * (1 + x / 100) = 1.05 * b)

theorem productivity_increase (x : ℝ) : x = 20 := sorry

end productivity_increase_l617_617985


namespace shooting_challenge_sequences_l617_617834

theorem shooting_challenge_sequences : ∀ (A B C : ℕ), 
  A = 4 → B = 4 → C = 2 →
  (A + B + C = 10) →
  (Nat.factorial (A + B + C) / (Nat.factorial A * Nat.factorial B * Nat.factorial C) = 3150) :=
by
  intros A B C hA hB hC hsum
  sorry

end shooting_challenge_sequences_l617_617834


namespace tangent_line_MN_l617_617220

variables {A O E F R P N M : Type*} [point A] [point O] [point E] [point F] [point R] [point P] [point N] [point M]
variables (AB AC : point A → Type*) -- distinct rays originating from A
variables (ω : set (point O)) -- circle centered at O
variables [tangent AC E ω] [tangent AB F ω] -- tangency conditions

-- Definitions for intersections
variables [segment EF] -- segment between E and F
variables [line_through O E'] (h_parallel : E' = EF) -- line through O parallel to EF
variables [intersect AB E' P] -- line E' intersects AB at P
variables [intersect PR AC N] -- line PR intersects AC at N
variables [intersect R_ AB (parallel_line_through R AC) M] -- line through R parallel to AC intersects AB at M

-- The theorem stating the question to be proved
theorem tangent_line_MN :
  tangent_line MN ω :=
sorry

end tangent_line_MN_l617_617220


namespace solve_quadratic_sqrt2_l617_617523

theorem solve_quadratic_sqrt2 :
  (∀ x : ℝ, x^2 - real.sqrt 2 * x = 0 → x = 0 ∨ x = real.sqrt 2) :=
begin
  sorry
end

end solve_quadratic_sqrt2_l617_617523


namespace evaluate_polynomial_minimize_expression_l617_617365

variable {f : ℕ → ℕ}
variable (x : ℕ) (a b : ℕ)

def polynomial (x : ℕ) : ℕ := 2 * x^6 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x

theorem evaluate_polynomial : polynomial 5 = 45 :=
by
  have v_0 := 2
  have v_1 := 2 * 5
  have v_2 := 10 * 5 - 5
  exact v_2

theorem minimize_expression (h : a + b = 45) (ha : a > 0) (hb : b > 0) : (1 : ℚ) / a + 4 / b = 1 / 5 :=
by 
  sorry

end evaluate_polynomial_minimize_expression_l617_617365


namespace construction_completion_day_l617_617948

theorem construction_completion_day :
  let start_day := "Monday"
  let starting_year := 2020
  let construction_duration := 1000
  (get_completion_day start_day starting_year construction_duration) = "Tuesday" :=
by sorry

end construction_completion_day_l617_617948


namespace fraction_of_area_of_shaded_triangle_l617_617454

theorem fraction_of_area_of_shaded_triangle : 
  let square_area := 6 * 6 in
  let triangle_area := 1 in
  (triangle_area : ℚ) / square_area = 1 / 36 :=
by
  let square_area := 6 * 6
  let triangle_area := 1
  have h : (triangle_area : ℚ) / square_area = 1 / 36 := sorry
  exact h

end fraction_of_area_of_shaded_triangle_l617_617454


namespace solve_system_l617_617221

theorem solve_system (x y : ℝ) (h1 : x * real.sqrt (x * y) + y * real.sqrt (x * y) = 10) (h2 : x^2 + y^2 = 17) :
  (x = sqrt 11 + sqrt(22) ∧ y = sqrt 11 - sqrt(22)) ∨ (x = sqrt 11 - sqrt(22) ∧ y = sqrt 11 + sqrt(22)) ∨
  (x = -sqrt 11 + sqrt(22) ∧ y = -sqrt 11 - sqrt(22)) ∨ (x = -sqrt 11 - sqrt(22) ∧ y = -sqrt 11 + sqrt(22)) :=
sorry

end solve_system_l617_617221


namespace average_speed_car_trip_l617_617386

theorem average_speed_car_trip :
  let d := 60
  let s1 := 60
  let s2 := 24
  let s3 := 72
  let s4 := 48
  let distance1 := d / 4
  let distance2 := d / 3
  let distance3 := d / 5
  let distance4 := d - (distance1 + distance2 + distance3)
  let time1 := distance1 / s1
  let time2 := distance2 / s2
  let time3 := distance3 / s3
  let time4 := distance4 / s4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := d / total_time
  average_speed ≈ 39.47 := by sorry

end average_speed_car_trip_l617_617386


namespace sum_of_number_and_radical_conjugate_l617_617679

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617679


namespace sin_half_angle_identity_proof_tan_half_angle_identity_proof_cos_half_angle_identity_proof_l617_617215

noncomputable def sin_half_angle_identity (α β γ r R : ℝ) : Prop :=
  sin (α / 2) * sin (β / 2) * sin (γ / 2) = r / (4 * R)

noncomputable def tan_half_angle_identity (α β γ r p : ℝ) : Prop :=
  tan (α / 2) * tan (β / 2) * tan (γ / 2) = r / p

noncomputable def cos_half_angle_identity (α β γ p R : ℝ) : Prop :=
  cos (α / 2) * cos (β / 2) * cos (γ / 2) = p / (4 * R)

theorem sin_half_angle_identity_proof (α β γ r R : ℝ) : sin_half_angle_identity α β γ r R :=
sorry

theorem tan_half_angle_identity_proof (α β γ r p : ℝ) : tan_half_angle_identity α β γ r p :=
sorry

theorem cos_half_angle_identity_proof (α β γ p R : ℝ) : cos_half_angle_identity α β γ p R :=
sorry

end sin_half_angle_identity_proof_tan_half_angle_identity_proof_cos_half_angle_identity_proof_l617_617215


namespace perimeter_of_figure_is_correct_l617_617953

-- Define the conditions as Lean variables and constants
def area_of_figure : ℝ := 144
def number_of_squares : ℕ := 4

-- Define the question as a theorem to be proven in Lean
theorem perimeter_of_figure_is_correct :
  let area_of_square := area_of_figure / number_of_squares
  let side_length := Real.sqrt area_of_square
  let perimeter := 9 * side_length
  perimeter = 54 :=
by
  intro area_of_square
  intro side_length
  intro perimeter
  sorry

end perimeter_of_figure_is_correct_l617_617953


namespace sum_of_radical_conjugates_l617_617682

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617682


namespace clock_angle_2_30_l617_617556

theorem clock_angle_2_30 : 
    let minute_hand_position := 180 
    let hour_hand_position := 75
    abs (minute_hand_position - hour_hand_position) = 105 := by
    sorry

end clock_angle_2_30_l617_617556


namespace rectangle_perimeter_l617_617990

theorem rectangle_perimeter (a b : ℝ) (h1 : (a + 3) * (b + 3) = a * b + 48) : 
  2 * (a + 3 + b + 3) = 38 :=
by
  sorry

end rectangle_perimeter_l617_617990


namespace problem_inequality_l617_617897

theorem problem_inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) 
  (h4 : b < 1) (h5 : 0 < c) (h6 : c < 1) (h7 : 0 < d) (h8 : d < 1) 
  (h_sum : a + b + c + d = 2) :
  sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ≤ (ac + bd) / 2 ∧
  ∃ (a b c d : ℝ), 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ 0 < d ∧ d < 1 ∧
  a + b + c + d = 2 ∧ (1 - a) * (1 - c) = (1 - b) * (1 - d) :=
  sorry

end problem_inequality_l617_617897


namespace john_initial_running_time_l617_617867

theorem john_initial_running_time (H : ℝ) (hH1 : 1.75 * H = 168 / 12)
: H = 8 :=
sorry

end john_initial_running_time_l617_617867


namespace percentage_error_x_percentage_error_y_l617_617607

theorem percentage_error_x (x : ℝ) : 
  let correct_result := x * 10
  let erroneous_result := x / 10
  (correct_result - erroneous_result) / correct_result * 100 = 99 :=
by
  sorry

theorem percentage_error_y (y : ℝ) : 
  let correct_result := y + 15
  let erroneous_result := y - 15
  (correct_result - erroneous_result) / correct_result * 100 = (30 / (y + 15)) * 100 :=
by
  sorry

end percentage_error_x_percentage_error_y_l617_617607


namespace total_cost_correct_l617_617865

-- Definitions for the conditions
def num_ladders_1 : ℕ := 10
def rungs_1 : ℕ := 50
def cost_per_rung_1 : ℕ := 2

def num_ladders_2 : ℕ := 20
def rungs_2 : ℕ := 60
def cost_per_rung_2 : ℕ := 3

def num_ladders_3 : ℕ := 30
def rungs_3 : ℕ := 80
def cost_per_rung_3 : ℕ := 4

-- Total cost calculation for the client
def total_cost : ℕ :=
  (num_ladders_1 * rungs_1 * cost_per_rung_1) +
  (num_ladders_2 * rungs_2 * cost_per_rung_2) +
  (num_ladders_3 * rungs_3 * cost_per_rung_3)

-- Statement to be proved
theorem total_cost_correct : total_cost = 14200 :=
by {
  sorry
}

end total_cost_correct_l617_617865


namespace parallelogram_angle_l617_617580

theorem parallelogram_angle (a b : ℕ) (h : a + b = 180) (exceed_by_10 : b = a + 10) : a = 85 := by
  -- proof skipped
  sorry

end parallelogram_angle_l617_617580


namespace max_grain_mass_l617_617600

def platform_length : ℝ := 10
def platform_width : ℝ := 5
def grain_density : ℝ := 1200
def angle_of_repose : ℝ := 45
def max_mass : ℝ := 175000

theorem max_grain_mass :
  let height_of_pile := platform_width / 2
  let volume_of_prism := platform_length * platform_width * height_of_pile
  let volume_of_pyramid := (1 / 3) * (platform_width * height_of_pile) * height_of_pile
  let total_volume := volume_of_prism + 2 * volume_of_pyramid
  let calculated_mass := total_volume * grain_density
  calculated_mass = max_mass :=
by {
  sorry
}

end max_grain_mass_l617_617600


namespace distinct_right_angles_l617_617172

-- Definition of conditions
def distinct_rectangles (n : ℕ) : Set (rectangle) :=
  {r | r ∈ distinct_rectangles ∧ ∃ K : Fintype, K.card = n}

-- The theorem to be proved
theorem distinct_right_angles (n : ℕ) (h : n ∈ ℕ) : 
  ∃ (R : Set (rectangle)), (R ∈ distinct_rectangles n) → 
  (∃ (angles : Set (right_angle)), angles.card ≥ 4 * ⌊real.sqrt n⌋) :=
sorry

end distinct_right_angles_l617_617172


namespace marble_prism_weight_l617_617420

theorem marble_prism_weight :
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  volume * density = 86400 :=
by
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  sorry

end marble_prism_weight_l617_617420


namespace median_of_list_is_2975_5_l617_617560

theorem median_of_list_is_2975_5 :
  let lst := (List.range 3030).map (λ x => x + 1) ++ (List.range 3030).map (λ x => (x + 1) * (x + 1))
  (List.median lst) = 2975.5 :=
by
  let lst := (List.range 3030).map (λ x => x + 1) ++ (List.range 3030).map (λ x => (x + 1) * (x + 1))
  have len_lst : lst.length = 6060 := 
    by sorry
  have med_idx1 : 3030 := 
    by sorry
  have med_idx2 : 3031 := 
    by sorry
  have med_term1 : lst.nth! (3030 - 1) = 2975 := 
    by sorry
  have med_term2 : lst.nth! 3030 = 2976 := 
    by sorry
  show lst.median = 2975.5,
  by sorry

end median_of_list_is_2975_5_l617_617560


namespace a_2018_eq_3_5_l617_617802

-- Define the sequence
def a : ℕ → ℚ
| 0     := 4 / 5
| (n+1) := if 0 ≤ a n ∧ a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1

theorem a_2018_eq_3_5 : a 2018 = 3 / 5 := by
  sorry

end a_2018_eq_3_5_l617_617802


namespace tan_cot_min_value_l617_617294

theorem tan_cot_min_value (x : ℝ) (h : 0 < x ∧ x < π) : 
  ∃ (c : ℝ), (∀ (y : ℝ), (0 < y ∧ y < π) → (tan y + cot y)^2 ≥ c) ∧ (tan x + cot x)^2 = c :=
by
  sorry

end tan_cot_min_value_l617_617294


namespace min_people_like_both_l617_617542

theorem min_people_like_both (total_people mozart_lovers beethoven_lovers : ℕ)
  (h1 : total_people = 200)
  (h2 : mozart_lovers = 150)
  (h3 : beethoven_lovers = 130) :
  ∃ (both_lovers : ℕ), both_lovers = 80 :=
by
  have h_not_beethoven : total_people - beethoven_lovers = 70 := by rw [h1, h3]; norm_num
  let max_mozart_not_beethoven := 70
  have both_lovers := mozart_lovers - max_mozart_not_beethoven
  have h4 : both_lovers = 80 := by rw [h2]; norm_num
  use both_lovers
  exact h4
  sorry

end min_people_like_both_l617_617542


namespace no_solution_inequality_l617_617219

theorem no_solution_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| < 4 * x - 1 ∧ x < a) ↔ a ≤ (2/3) := by sorry

end no_solution_inequality_l617_617219


namespace find_P_l617_617898

-- Definitions of the conditions:
def isArithmeticSequence (a_n : Nat → Int) (d : Int) : Prop := 
  ∀ n, a_n (n + 1) = a_n n + d

def sum_of_sequence (a_n : Nat → Int) (n : Nat) : Int :=
  List.sum (List.map a_n (List.range n))

def sum_of_arithmetic_sequence (a : Int) (d : Int) (n : Nat) : Int :=
  (n * (2 * a + (n - 1) * d)) / 2

-- The condition as per the problem statement:
def condition (a_n : Nat → Int) : Prop :=
  isArithmeticSequence a_n 1 ∧ sum_of_sequence a_n 100 = 2012

-- The Lean 4 statement to prove the equivalent problem:
theorem find_P (a_n : Nat → Int) (h : condition a_n) : 
  sum_of_sequence (λ k, a_n (2 * (k + 1))) 50 = 1031 :=
sorry

end find_P_l617_617898


namespace janet_total_action_figures_l617_617041

/-- Janet owns 10 action figures, sells 6, gets 4 more in better condition,
and then receives twice her current collection from her brother.
We need to prove she ends up with 24 action figures. -/
theorem janet_total_action_figures :
  let initial := 10 in
  let after_selling := initial - 6 in
  let after_acquiring_better := after_selling + 4 in
  let from_brother := 2 * after_acquiring_better in
  after_acquiring_better + from_brother = 24 :=
by
  -- Proof would go here
  sorry

end janet_total_action_figures_l617_617041


namespace area_ratio_l617_617584

universe u

variables {α : Type u}
variables (a b c : ℝ) (S_ABC S_PQRSTF : ℝ)
variables (A B C : α) (A' B' C' P Q R S T F : α)
variables [h1 : IsTriangle A B C]
variables [h2 : IsMidpointArc A' B C]
variables [h3 : IsMidpointArc B' A C]
variables [h4 : IsMidpointArc C' A B]
variables [h5 : Intersection A B P Q R S T F]

theorem area_ratio :
  S_PQRSTF / S_ABC = 1 - (a * b + a * c + b * c) / (a + b + c) ^ 2 := sorry

end area_ratio_l617_617584


namespace baker_A_pastries_l617_617631

theorem baker_A_pastries :
  ∀ (cakesA pastriesA cakesB pastriesB : ℕ),
  cakesA = 7 → pastriesA = 148 → cakesB = 10 → pastriesB = 200 →
  let finalPastriesA := (pastriesA + pastriesB) / 2 in
  let pastriesSoldA := 103 in
  finalPastriesA - pastriesSoldA = 71 :=
begin
  -- Define the inputs and initial facts
  intros cakesA pastriesA cakesB pastriesB,
  intros ha pbA hb pbB,

  -- Setup known amounts of pastries initially
  have h_finalPastriesA : finalPastriesA = 174,
  { rw [ha, pbA, hb, pbB],
    norm_num },

  -- Subtract pastries sold
  have h_remainingPastriesA : finalPastriesA - pastriesSoldA = 71,
  { rw [h_finalPastriesA],
    norm_num },

  -- Conclude the theorem
  exact h_remainingPastriesA
end

end baker_A_pastries_l617_617631


namespace trigonometric_solution_l617_617477

theorem trigonometric_solution (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, (n % 2 = 0 → (x = k * real.pi)) ∨ 
          (n % 2 = 1 → (x = 2 * k * real.pi ∨ x = 2 * k * real.pi - real.pi / 2)) 
:=
sorry

end trigonometric_solution_l617_617477


namespace product_modulo_l617_617485

theorem product_modulo : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m := 
  sorry

end product_modulo_l617_617485


namespace negation_exists_equation_l617_617923

theorem negation_exists_equation (P : ℝ → Prop) :
  (∃ x > 0, x^2 + 3 * x - 5 = 0) → ¬ (∃ x > 0, x^2 + 3 * x - 5 = 0) = ∀ x > 0, x^2 + 3 * x - 5 ≠ 0 :=
by sorry

end negation_exists_equation_l617_617923


namespace isosceles_triangles_count_l617_617091

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def is_isosceles (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  distance x1 y1 x2 y2 = distance x1 y1 x3 y3 ∨ 
  distance x1 y1 x2 y2 = distance x2 y2 x3 y3 ∨
  distance x1 y1 x3 y3 = distance x2 y2 x3 y3

def triangleA_isosceles : Prop := is_isosceles 1 4 3 4 2 2
def triangleB_isosceles : Prop := is_isosceles 4 2 4 4 6 2
def triangleC_isosceles : Prop := is_isosceles 7 1 8 4 9 1
def triangleD_isosceles : Prop := is_isosceles 0 0 2 1 1 3

theorem isosceles_triangles_count :
  triangleA_isosceles ∧ triangleB_isosceles ∧ triangleC_isosceles ∧ triangleD_isosceles → 
  4 = 4 := by
  intros
  sorry

end isosceles_triangles_count_l617_617091


namespace middle_digit_base7_l617_617238

theorem middle_digit_base7 (a b c : ℕ) 
  (h1 : N = 49 * a + 7 * b + c) 
  (h2 : N = 81 * c + 9 * b + a)
  (h3 : a < 7 ∧ b < 7 ∧ c < 7) : 
  b = 0 :=
by sorry

end middle_digit_base7_l617_617238


namespace log_ab_eq_l617_617095

theorem log_ab_eq : ∀ (a b c : ℝ), a > 0 → a ≠ 1 → b > 0 → b ≠ 1 → c > 0 → 
  log (a * b) c = (log a c * log b c) / (log a c + log b c) := by
  intros a b c ha ha1 hb hb1 hc
  sorry

end log_ab_eq_l617_617095


namespace intersection_M_N_l617_617368

noncomputable theory

def M : set ℝ := {y | ∃ x ∈ Icc (-5 : ℝ) 5, y = 2 * real.sin x}
def N : set ℝ := {x | ∃ y, y = real.log2 (x - 1) ∧ x > 1}

theorem intersection_M_N :
  (M ∩ N = {x | 1 < x ∧ x ≤ 2}) :=
sorry

end intersection_M_N_l617_617368


namespace eccentricity_of_conic_section_l617_617779

-- Define the conic section equation and focus location
def conic_section (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + a * y^2 = 1

-- Define the location of the focus
def focus (a : ℝ) : ℝ × ℝ :=
  ⟨(2 / real.sqrt (real.abs a)), 0⟩

-- Define the eccentricity problem
theorem eccentricity_of_conic_section (a : ℝ) :
  (focus a = (2 / real.sqrt (real.abs a), 0)) →
  ((a = 5) → (∃ e : ℝ, e = 2 * real.sqrt 5 / 5)) ∧
  ((a = -3) → (∃ e : ℝ, e = 2 * real.sqrt 3 / 3)) :=
by
  -- Proof will be provided here
  sorry

end eccentricity_of_conic_section_l617_617779


namespace abs_diff_of_numbers_l617_617146

theorem abs_diff_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by
  sorry

end abs_diff_of_numbers_l617_617146


namespace num_3_identity_transformations_num_n_identity_transformations_l617_617942

/-- Definitions of transformations: L (counterclockwise rotation by 90 degrees),
    R (clockwise rotation by 90 degrees), and S (reflection about the origin). -/
def L := (λ (p : ℤ × ℤ), (-p.2, p.1)) -- Rotate by 90 degrees counterclockwise
def R := (λ (p : ℤ × ℤ), (p.2, -p.1)) -- Rotate by 90 degrees clockwise
def S := (λ (p : ℤ × ℤ), (-p.1, -p.2)) -- Reflect about the origin

/-- The vertices of the square in the coordinate plane. -/
def A := (1, 1)
def B := (-1, 1)
def C := (-1, -1)
def D := (1, -1)

/-- Definition of k-identity transformation. -/
def is_k_identity (k : ℕ) (seq : list (ℤ × ℤ → ℤ × ℤ)) : Prop :=
  (seq.foldl (flip (· ∘ ·)) id) (1, 1) = (1, 1) ∧
  (seq.foldl (flip (· ∘ ·)) id) (-1, 1) = (-1, 1) ∧
  (seq.foldl (flip (· ∘ ·)) id) (-1, -1) = (-1, -1) ∧
  (seq.foldl (flip (· ∘ ·)) id) (1, -1) = (1, -1)

/-- main statements for lean 4 problem. -/
theorem num_3_identity_transformations : ∃ Q_3 : ℕ, Q_3 = 6 :=
sorry

theorem num_n_identity_transformations (n : ℕ) (h : n > 0) : ∃ Q_n : ℕ, Q_n = (3 * (-1)^n + 3^n) / 4 :=
sorry

end num_3_identity_transformations_num_n_identity_transformations_l617_617942


namespace range_of_f_solution_of_fx_eq_3_l617_617787

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ -1 then x + 2 else if -1 < x ∧ x < 2 then x^2 else 0

theorem range_of_f :
  (∀ y, ∃ x, f x = y ↔ y ∈ Set.Iio 4) := sorry

theorem solution_of_fx_eq_3 (x : ℝ) :
  f x = 3 → x = Real.sqrt 3 := sorry

end range_of_f_solution_of_fx_eq_3_l617_617787


namespace train_pass_time_l617_617577

noncomputable def trainPassingTime (train_length : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh + man_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)
  train_length / relative_speed_ms

theorem train_pass_time :
  trainPassingTime 180 55 7 ≈ 10.45 :=
by
  sorry

end train_pass_time_l617_617577


namespace S_30_zero_l617_617839

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {n : ℕ} 

-- Definitions corresponding to the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a_n n = a1 + d * n

def sum_arithmetic_sequence (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
  
-- The given conditions
axiom S_eq (S_10 S_20 : ℝ) : S 10 = S 20

-- The theorem we need to prove
theorem S_30_zero (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_arithmetic_sequence S a_n)
  (h_eq : S 10 = S 20) :
  S 30 = 0 :=
sorry

end S_30_zero_l617_617839


namespace imaginary_part_of_z_is_one_l617_617965

noncomputable def z : ℂ := (1 + 2 * complex.I) / (2 - complex.I)

theorem imaginary_part_of_z_is_one : z.im = 1 := by
  sorry

end imaginary_part_of_z_is_one_l617_617965


namespace number_of_paths_from_C_to_D_l617_617811

-- Define the grid and positions
def C := (0,0)  -- Bottom-left corner
def D := (7,3)  -- Top-right corner
def gridWidth : ℕ := 7
def gridHeight : ℕ := 3

-- Define the binomial coefficient function
-- Note: Lean already has binomial coefficient defined in Mathlib, use Nat.choose for that

-- The statement to prove
theorem number_of_paths_from_C_to_D : Nat.choose (gridWidth + gridHeight) gridHeight = 120 :=
by
  sorry

end number_of_paths_from_C_to_D_l617_617811


namespace sum_of_divisors_of_30_l617_617564

theorem sum_of_divisors_of_30 : (∑ d in (Finset.filter (λ d, 30 % d = 0) (Finset.range 31)), d) = 72 := by
  sorry

end sum_of_divisors_of_30_l617_617564


namespace equidistant_point_x_axis_l617_617185

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
    real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem equidistant_point_x_axis (x : ℝ) :
  let A := (-2, 0)
  let B := (0, 4)
  distance A (x, 0) = distance B (x, 0) → x = 3 :=
by
  intros
  have h1 : distance A (x, 0) = real.sqrt ((-2 - x)^2 + (0 - 0)^2) := rfl
  have h2 : distance B (x, 0) = real.sqrt ((0 - x)^2 + (4 - 0)^2) := rfl
  have h_eq : real.sqrt ((-2 - x)^2) = real.sqrt (x^2 + 16) := by sorry
  have h_sqr_eq : ((-2 - x)^2) = (x^2 + 16) := by sorry
  have h_simplified : -4x = 12 := by sorry
  have h_sol : x = 3 := by sorry
  exact h_sol

end equidistant_point_x_axis_l617_617185


namespace b_months_correct_l617_617579

noncomputable def horse_months_a : ℕ := 12 * 8
noncomputable def horse_months_b (x : ℕ) : ℕ := 16 * x
noncomputable def horse_months_c : ℕ := 18 * 6

noncomputable def total_horse_months (x : ℕ) : ℕ :=
  horse_months_a + horse_months_b(x) + horse_months_c

noncomputable def b_proportion (x : ℕ) : Rat :=
  (16 * x : ℚ) / (total_horse_months(x) : ℚ)

noncomputable def b_payment_proportion : Rat :=
  (348 : ℚ) / (841 : ℚ)

theorem b_months_correct : ∃ x : ℕ, b_proportion(x) = b_payment_proportion ∧ x = 9 := by
  sorry

end b_months_correct_l617_617579


namespace dodecagon_area_ratio_l617_617428

noncomputable def inner_to_outer_dodecagon_area_ratio : ℚ :=
  let s := 1 -- considering side length of larger dodecagon as unit length
  let θ := 150
  let cos75 : ℚ := real.to_rat $ real.cos (75 * real.pi / 180)
  let cos37_5 : ℚ := real.to_rat $ real.cos (37.5 * real.pi / 180)
  let MN_s_ratio := (sqrt ((5* (s^2) / 4) - (s^2 * cos75))) * cos37_5
  let area_inner := 3 * (MN_s_ratio ^ 2) * real.to_rat (real.tan (real.pi / 12))
  let area_outer := 3 * (s^2) * real.to_rat (real.tan (real.pi / 12))
  (area_inner / area_outer)

theorem dodecagon_area_ratio : ∃ m n : ℕ, gcd m n = 1 ∧ inner_to_outer_dodecagon_area_ratio = m / n

end dodecagon_area_ratio_l617_617428


namespace range_function_l617_617754

open Real

noncomputable def function_to_prove (x : ℝ) (a : ℕ) : ℝ := x + 2 * a / x

theorem range_function (a : ℕ) (h1 : a^2 - a < 2) (h2 : a ≠ 0) : 
  Set.range (function_to_prove · a) = {y : ℝ | y ≤ -2 * sqrt 2} ∪ {y : ℝ | y ≥ 2 * sqrt 2} :=
by
  sorry

end range_function_l617_617754


namespace temperature_at_tian_du_peak_height_of_mountain_peak_l617_617916

-- Problem 1: Temperature at the top of Tian Du Peak
theorem temperature_at_tian_du_peak
  (height : ℝ) (drop_rate : ℝ) (initial_temp : ℝ)
  (H : height = 1800) (D : drop_rate = 0.6) (I : initial_temp = 18) :
  (initial_temp - (height / 100 * drop_rate)) = 7.2 :=
by
  sorry

-- Problem 2: Height of the mountain peak
theorem height_of_mountain_peak
  (drop_rate : ℝ) (foot_temp top_temp : ℝ)
  (D : drop_rate = 0.6) (F : foot_temp = 10) (T : top_temp = -8) :
  (foot_temp - top_temp) / drop_rate * 100 = 3000 :=
by
  sorry

end temperature_at_tian_du_peak_height_of_mountain_peak_l617_617916


namespace proportion_of_line_segments_l617_617825

theorem proportion_of_line_segments (a b c d : ℕ)
  (h_proportion : a * d = b * c)
  (h_a : a = 2)
  (h_b : b = 4)
  (h_c : c = 3) :
  d = 6 :=
by
  sorry

end proportion_of_line_segments_l617_617825


namespace problem_1_problem_2_l617_617366

open Set Real

-- Definition of the sets A, B, and the complement of B in the real numbers
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Proof problem (1): Prove that A ∩ (complement of B) = [1, 2]
theorem problem_1 : (A ∩ (compl B)) = {x | 1 ≤ x ∧ x ≤ 2} := sorry

-- Proof problem (2): Prove that the set of values for the real number a such that C(a) ∩ A = C(a)
-- is (-∞, 3]
theorem problem_2 : { a : ℝ | C a ⊆ A } = { a : ℝ | a ≤ 3 } := sorry

end problem_1_problem_2_l617_617366


namespace inequality_solution_l617_617482

theorem inequality_solution (x : ℝ) : 
  (x - 3) / (x^2 + 5 * x + 11) ≥ 0 ↔ x ≥ 3 :=
by
  have denom_pos: ∀ (x : ℝ), x^2 + 5 * x + 11 > 0 := sorry
  rw ← @div_nonneg_iff ℝ _ (x - 3) (x^2 + 5 * x + 11) (denom_pos x)
  sorry

end inequality_solution_l617_617482


namespace no_valid_sum_seventeen_l617_617419

def std_die (n : ℕ) : Prop := n ∈ [1, 2, 3, 4, 5, 6]

def valid_dice (a b c d : ℕ) : Prop := std_die a ∧ std_die b ∧ std_die c ∧ std_die d

def sum_dice (a b c d : ℕ) : ℕ := a + b + c + d

def prod_dice (a b c d : ℕ) : ℕ := a * b * c * d

theorem no_valid_sum_seventeen (a b c d : ℕ) (h_valid : valid_dice a b c d) (h_prod : prod_dice a b c d = 360) : sum_dice a b c d ≠ 17 :=
sorry

end no_valid_sum_seventeen_l617_617419


namespace sum_of_radical_conjugates_l617_617686

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617686


namespace simplify_factorial_expression_l617_617934

theorem simplify_factorial_expression :
  (13.factorial / (10.factorial + 3 * 9.factorial)) = 1320 :=
by
  sorry

end simplify_factorial_expression_l617_617934


namespace gcd_840_1764_l617_617168

theorem gcd_840_1764 : Int.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l617_617168


namespace find_significance_level_l617_617535

noncomputable def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![[10, 40], [20, 30]]

def n := 100

def chi_squared_value (n : ℕ) (a b c d : ℕ) : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def K_squared := chi_squared_value n 10 40 20 30

def k0 := 3.841
def k0_prob := 0.05

theorem find_significance_level :
  K_squared <= 4.762 :=
by
  sorry

end find_significance_level_l617_617535


namespace find_mistake_l617_617991

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l617_617991


namespace monotonic_increasing_interval_l617_617127

noncomputable def f : ℝ → ℝ := fun x => x * Real.exp (-x)

theorem monotonic_increasing_interval :
  ∀ x, (f x = x * Real.exp (-x)) → ((1 - x) / Real.exp x ≥ 0 ↔ x ∈ Icc (-1 : ℝ) (0 : ℝ)) := by
  sorry

end monotonic_increasing_interval_l617_617127


namespace sqrt_equation_solution_l617_617299

theorem sqrt_equation_solution :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a < b ∧ (real.sqrt (3 + real.sqrt (45 + 20 * real.sqrt 5)) = real.sqrt a + real.sqrt b) ∧ (a, b) = (3, 5) :=
by
  sorry

end sqrt_equation_solution_l617_617299


namespace product_of_three_numbers_l617_617526

theorem product_of_three_numbers : 
  ∃ x y z : ℚ, x + y + z = 30 ∧ x = 3 * (y + z) ∧ y = 6 * z ∧ x * y * z = 23625 / 686 :=
by
  sorry

end product_of_three_numbers_l617_617526


namespace monotonicity_of_f_range_of_a_l617_617790

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - 1 - log x

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → (a ≤ 0 → deriv (f a) x < 0) ∧
            (a > 0 → (∃ c : ℝ, c = sqrt (1 / (2 * a)) ∧ 
                (0 < x ∧ x < c → deriv (f a) x < 0) ∧ 
                (c < x → deriv (f a) x > 0)))) :=
begin
  sorry
end

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f a x ≥ x) ↔ (2 ≤ a) :=
begin
  sorry
end

end monotonicity_of_f_range_of_a_l617_617790


namespace even_perfect_square_factors_count_l617_617808

theorem even_perfect_square_factors_count :
  let n := 2^6 * 7^10 in
  (∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ a ≥ 1 ∧ 0 ≤ b ∧ b ≤ 10 ∧ b % 2 = 0 → (2^a * 7^b | n)) ∧
  let possible_a := {2, 4, 6} in
  let possible_b := {0, 2, 4, 6, 8, 10} in
  ∃ (count : ℕ), count = possible_a.card * possible_b.card ∧ count = 18 :=
sorry

end even_perfect_square_factors_count_l617_617808


namespace total_differential_l617_617303

variable {R : Type*} [Field R] {x y z c : R}

theorem total_differential (h : z^2 - 2 * x * y = c) :
  differential z dx dy = (y / z) * dx + (x / z) * dy := by
  sorry

end total_differential_l617_617303


namespace train_speed_correct_l617_617240

noncomputable def train_length : ℝ := 50
noncomputable def passing_time : ℝ := 4.99960003199744
noncomputable def conversion_factor : ℝ := 3.6
noncomputable def calculated_speed_kmph : ℝ := 36.00287976962016

theorem train_speed_correct :
  (train_length / passing_time) * conversion_factor ≈ calculated_speed_kmph :=
by
  sorry

end train_speed_correct_l617_617240


namespace ball_event_check_l617_617310

noncomputable def ballEvent (balls : list ℕ) : Prop :=
  ∃ A B : ℕ,
    A = 1 ∧ B = 3 ∧
    (A = "exactly 1 white ball" ∧ B = "exactly 2 white balls")

theorem ball_event_check : ballEvent [1, 2, 1, 2] :=
by
  sorry

end ball_event_check_l617_617310


namespace angle_bisector_length_l617_617863

open Real
open Complex

-- Definitions for the problem
def side_lengths (AC BC : ℝ) : Prop :=
  AC = 6 ∧ BC = 9

def angle_C (angle : ℝ) : Prop :=
  angle = 120

-- Main statement to prove
theorem angle_bisector_length (AC BC angle x : ℝ)
  (h1 : side_lengths AC BC)
  (h2 : angle_C angle) :
  x = 18 / 5 :=
  sorry

end angle_bisector_length_l617_617863


namespace negation_of_even_function_l617_617509

variable {M : Type*} (f : M → M) (P : M → Prop)

-- Define even function
def is_even_function : Prop := ∀ x : M, f (-x) = f x

-- The proof statement
theorem negation_of_even_function :
  ¬ is_even_function f ↔ ∃ x : M, f (-x) ≠ f x := sorry

end negation_of_even_function_l617_617509


namespace range_of_c_l617_617333

variable {a b c : ℝ} -- Declare the variables

-- Define the conditions
def triangle_condition (a b : ℝ) : Prop :=
|a + b - 4| + (a - b + 2)^2 = 0

-- Define the proof problem
theorem range_of_c {a b c : ℝ} (h : triangle_condition a b) : 2 < c ∧ c < 4 :=
sorry -- Proof to be completed

end range_of_c_l617_617333


namespace evaluate_expression_l617_617718

-- Define the base and the exponents
def base : ℝ := 64
def exponent1 : ℝ := 0.125
def exponent2 : ℝ := 0.375
def combined_result : ℝ := 8

-- Statement of the problem
theorem evaluate_expression : (base^exponent1) * (base^exponent2) = combined_result := 
by 
  sorry

end evaluate_expression_l617_617718


namespace highest_possible_rubidium_concentration_l617_617286

noncomputable def max_rubidium_concentration (R C F : ℝ) : Prop :=
  (R + C + F > 0) →
  (0.10 * R + 0.08 * C + 0.05 * F) / (R + C + F) = 0.07 ∧
  (0.05 * F) / (R + C + F) ≤ 0.02 →
  (0.10 * R) / (R + C + F) = 0.01

theorem highest_possible_rubidium_concentration :
  ∃ R C F : ℝ, max_rubidium_concentration R C F :=
sorry

end highest_possible_rubidium_concentration_l617_617286


namespace sin_double_angle_l617_617001

theorem sin_double_angle (α : ℝ) (h : cos (π / 4 - α) = 3 / 5) : sin (2 * α) = -7 / 25 := sorry

end sin_double_angle_l617_617001


namespace opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l617_617516

theorem opposite_neg_abs_five_minus_six : -|5 - 6| = -1 := by
  sorry

theorem opposite_of_neg_abs (h : -|5 - 6| = -1) : -(-1) = 1 := by
  sorry

theorem math_problem_proof : -(-|5 - 6|) = 1 := by
  apply opposite_of_neg_abs
  apply opposite_neg_abs_five_minus_six

end opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l617_617516


namespace cyclic_YOMI_l617_617890

theorem cyclic_YOMI 
  (A X Y Z B : Point)
  (O : Point := midpoint A B)
  (K : Point := foot Y (line A B))
  (L : Point := intersection_of (line X Z) (line Y O))
  (M : Point := some (on_line (line K L)) ∧ dist M A = dist M B)
  (I : Point := reflection O (line X Z))
  (cyclic: cyclic_quadrilateral X K O Z) :
  cyclic_quadrilateral Y O M I := by
  sorry

end cyclic_YOMI_l617_617890


namespace intersection_of_sets_l617_617770

theorem intersection_of_sets :
  let A := {x : ℝ | x^2 - x - 2 ≤ 0}
  let B := {x : ℤ | 1 ≤ 2^x ∧ 2^x ≤ 8}
  A ∩ B = ({0, 1, 2} : set ℤ) :=
by
  let A := {x : ℝ | x^2 - x - 2 ≤ 0}
  let B := {x : ℤ | 1 ≤ 2^x ∧ 2^x ≤ 8}
  sorry

end intersection_of_sets_l617_617770


namespace sum_of_number_and_conjugate_l617_617669

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617669


namespace max_area_of_triangle_l617_617015

theorem max_area_of_triangle
  (a b c : ℝ)
  (cos_half_C : ℝ)
  (h_cos_half_C : cos_half_C = sqrt 5 / 3)
  (h_eq : a * real.cos (real.acos ((a^2 + c^2 - b^2) / (2 * a * c))) + 
                b * real.cos (real.acos ((c^2 + b^2 - a^2) / (2 * b * c))) = 2):
  ∃ S_max, S_max = sqrt 5 / 2 :=
by
  sorry

end max_area_of_triangle_l617_617015


namespace find_abs_α_l617_617061

noncomputable def α : ℂ := sorry
noncomputable def β : ℂ := sorry

def conditions (α β : ℂ) :=
  conj α = β ∧
  is_real (α / β^2) ∧
  abs (α - β) = 6 ∧
  re (α + β) = 4

theorem find_abs_α (α β : ℂ) (h : conditions α β) : abs α = 2 * real.sqrt 3 :=
sorry

end find_abs_α_l617_617061


namespace value_of_X_l617_617381

def M : ℕ := 2024 / 4
def N : ℕ := M / 2
def X : ℕ := M + N

theorem value_of_X : X = 759 := by
  sorry

end value_of_X_l617_617381


namespace Kolya_made_the_mistake_l617_617999

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l617_617999


namespace number_of_young_fish_l617_617912

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end number_of_young_fish_l617_617912


namespace force_game_end_no_definitive_winning_strategy_l617_617614

-- Define the conditions
variable (A_n B_n : ℕ → EuclideanSpace ℝ (Fin 2))
variable (A_1 : EuclideanSpace ℝ (Fin 2))
variable (game_length : ℝ := 1)

-- Definitions for circles and perpendicular bisectors as required in the problem
def circle (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p | (dist center p) = radius}

def perpendicular_bisector (p1 p2 : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {q | ∃ l, q = (p1 + p2) / 2 + l • ⟦p2 - p1⟧}

-- Assume the rules of the game as conditions in the Lean 4 environment
axiom no_coincide (n : ℕ) (h : B_n (n+1) ≠ A_n (n+1)) : B_n (n+1) ≠ B_n n
axiom no_overlap (n m : ℕ) (h1 : n ≠ m)
  (h2 : ∀ t, t ∈ Icc (0:ℝ) 1 → A_n t ≠ A_m t ∨ B_n t ≠ B_m t) : true

-- Main proof statement
theorem force_game_end : (∃ (strategy : ℕ → EuclideanSpace ℝ (Fin 2) → Set (EuclideanSpace ℝ (Fin 2))), 
  (∀ n, strategy n B_n = circle A_n game_length ∩ perpendicular_bisector (A_n n) (B_n n)) ∧ 
  (∀ n, strategy n A_n = circle B_n game_length ∩ perpendicular_bisector (B_n n) (A_n n)))
→ (∀ n, ∃ (x : EuclideanSpace ℝ (Fin 2)), x ∈ circle (A_n n) game_length ∩ circle (B_n n) game_length)
→ (∃ N, ∃ x ∈ (circle (A_1) game_length), x = A_n N ∨ x = B_n N) :=
by sorry

noncomputable def no_winning_strategy (A B : Set (EuclideanSpace ℝ (Fin 2))) :=
  ¬(∀ x ∈ A, ∃ y ∈ B, dist x y < 1) ∧ ¬(∀ y ∈ B, ∃ x ∈ A, dist y x < 1)

-- The second part stating that there is no winning strategy
theorem no_definitive_winning_strategy : no_winning_strategy (circle A_1 game_length) (circle A_1 game_length) :=
by sorry

end force_game_end_no_definitive_winning_strategy_l617_617614


namespace lines_concurrent_l617_617104

variables {A B C A1 A2 B1 B2 C1 C2 : Type*}

-- Define a structure for equilateral triangles
structure EquilateralTriangle (P Q R : Type*) : Prop :=
(eq_length : distance P Q = distance Q R ∧ distance Q R = distance R P ∧ distance R P = distance P Q)

-- Define a structure for convex hexagons with equal side lengths
structure EqualSideHexagon (P1 P2 P3 P4 P5 P6 : Type*) : Prop :=
(eq_side_lengths : distance P1 P2 = distance P2 P3 ∧
                   distance P2 P3 = distance P3 P4 ∧
                   distance P3 P4 = distance P4 P5 ∧
                   distance P4 P5 = distance P5 P6 ∧
                   distance P5 P6 = distance P6 P1)

-- Given conditions
def condition_equilateral_triangle : EquilateralTriangle A B C := sorry

def condition_points_on_sides : 
  A1 ∈ segment B C ∧ A2 ∈ segment B C ∧
  B1 ∈ segment C A ∧ B2 ∈ segment C A ∧
  C1 ∈ segment A B ∧ C2 ∈ segment A B := sorry

def condition_convex_hexagon : EqualSideHexagon A1 A2 B1 B2 C1 C2 := sorry

-- Assert the concurrency of the lines
theorem lines_concurrent 
 (triangle_eq : EquilateralTriangle A B C)
 (points_on_sides : A1 ∈ segment B C ∧ A2 ∈ segment B C ∧
                    B1 ∈ segment C A ∧ B2 ∈ segment C A ∧
                    C1 ∈ segment A B ∧ C2 ∈ segment A B)
 (hexagon_eq : EqualSideHexagon A1 A2 B1 B2 C1 C2) :
  concurrent (line A1 B2) (line B1 C2) (line C1 A2) := 
sorry

end lines_concurrent_l617_617104


namespace orig_polygon_sides_l617_617613

theorem orig_polygon_sides (n : ℕ) (S : ℕ) :
  (n - 1 > 2) ∧ S = 1620 → (n = 10 ∨ n = 11 ∨ n = 12) :=
by
  sorry

end orig_polygon_sides_l617_617613


namespace ones_digit_of_tripling_4567_l617_617972

theorem ones_digit_of_tripling_4567 : 
  let n := 4567 in 
  let tripled := 3 * n in
  (tripled % 10) = 1 :=
by
  sorry

end ones_digit_of_tripling_4567_l617_617972


namespace range_of_m_l617_617774

-- Definitions of functions f and g
def f (x : ℝ) := sqrt (log (1 / 2) (x - 1))
def g (x : ℝ) := (3 : ℝ) ^ (m - 2 * x - x ^ 2) - 1

-- Define the domain of f
def A := set.Ioc 1 2

-- Define the range of g
def B (m : ℝ) := set.Ioc (-1) (3 ^ (1 + m) - 1)

-- The main equivalence we need to prove
theorem range_of_m (m : ℝ) : A ⊆ B m ↔ 0 ≤ m :=
by {
  sorry
}

end range_of_m_l617_617774


namespace parker_net_income_after_taxes_l617_617459

noncomputable def parker_income : Real := sorry

theorem parker_net_income_after_taxes :
  let daily_pay := 63
  let hours_per_day := 8
  let hourly_rate := daily_pay / hours_per_day
  let overtime_rate := 1.5 * hourly_rate
  let overtime_hours_per_weekend_day := 3
  let weekends_in_6_weeks := 6
  let days_per_week := 7
  let total_days_in_6_weeks := days_per_week * weekends_in_6_weeks
  let regular_earnings := daily_pay * total_days_in_6_weeks
  let total_overtime_earnings := overtime_rate * overtime_hours_per_weekend_day * 2 * weekends_in_6_weeks
  let gross_income := regular_earnings + total_overtime_earnings
  let tax_rate := 0.1
  let net_income_after_taxes := gross_income * (1 - tax_rate)
  net_income_after_taxes = 2764.125 := by sorry

end parker_net_income_after_taxes_l617_617459


namespace value_v1_l617_617544

noncomputable def polynomial : ℚ[X] := X^6 - 5 * X^5 + 6 * X^4 + X^2 + 0.3 * X + 2

theorem value_v1 (x : ℚ) (h : x = -2) : 
  let v0 := polynomial.coeff 6,
      v1 := v0 * x + polynomial.coeff 5 
  in v1 = -7 := 
by 
  have v0 := (polynomial.coeff 6),
  have v1 := v0 * x + (polynomial.coeff 5),
  show v1 = -7, from sorry

end value_v1_l617_617544


namespace angle_between_vectors_l617_617371

-- Definitions of the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (x, sqrt 3)
def b (x : ℝ) : ℝ × ℝ := (x, -sqrt 3)

-- Definition of the angle θ in question
def θ := (2/3) * Real.pi

-- Condition that 2a + b is perpendicular to b
def perpendicular_condition (x : ℝ) : Prop :=
  let vec := (3 * x, sqrt 3) in 
  (vec.1 * (x) + vec.2 * (-sqrt 3)) = 0
  
-- Theorem statement
theorem angle_between_vectors (x : ℝ) (hx : perpendicular_condition x) :
  Real.cos θ = (a x).1 * (b x).1 + (a x).2 * (b x).2 / (Real.sqrt ((a x).1 ^ 2 + (a x).2 ^ 2) * Real.sqrt ((b x).1 ^ 2 + (b x).2 ^ 2)) := 
  sorry

end angle_between_vectors_l617_617371


namespace sum_of_conjugates_l617_617650

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617650


namespace product_of_slopes_of_MQ_and_NQ_equals_three_l617_617079

theorem product_of_slopes_of_MQ_and_NQ_equals_three :
  (∀ (P : Pointed ℝ ℝ) (hP : (P.x + 2)^2 + P.y^2 = 4 ∧ P ≠ ⟨0, 0⟩),
    let A := ⟨2, 0⟩,
        M := ⟨-1, 0⟩,
        N := ⟨1, 0⟩,
        Q := find_intersection_of_bisector_and_perpendicular P A in
    let slope (a b : Pointed ℝ ℝ) : ℝ := (b.y - a.y) / (b.x - a.x) in
    slope M Q * slope N Q = 3) :=
begin
  sorry
end

end product_of_slopes_of_MQ_and_NQ_equals_three_l617_617079


namespace sequence_x_sequence_y_sequence_z_sequence_t_l617_617919

theorem sequence_x (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^2 + n = 2) else 
   if n = 2 then (n^2 + n = 6) else 
   if n = 3 then (n^2 + n = 12) else 
   if n = 4 then (n^2 + n = 20) else true) := 
by sorry

theorem sequence_y (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2 * n^2 = 2) else 
   if n = 2 then (2 * n^2 = 8) else 
   if n = 3 then (2 * n^2 = 18) else 
   if n = 4 then (2 * n^2 = 32) else true) := 
by sorry

theorem sequence_z (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^3 = 1) else 
   if n = 2 then (n^3 = 8) else 
   if n = 3 then (n^3 = 27) else 
   if n = 4 then (n^3 = 64) else true) := 
by sorry

theorem sequence_t (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2^n = 2) else 
   if n = 2 then (2^n = 4) else 
   if n = 3 then (2^n = 8) else 
   if n = 4 then (2^n = 16) else true) := 
by sorry

end sequence_x_sequence_y_sequence_z_sequence_t_l617_617919


namespace probability_at_least_two_green_l617_617866

def total_apples := 10
def red_apples := 6
def green_apples := 4
def choose_apples := 3

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_at_least_two_green :
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) = 40 ∧ 
  binomial total_apples choose_apples = 120 ∧
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) / binomial total_apples choose_apples = 1 / 3 := by
sorry

end probability_at_least_two_green_l617_617866


namespace g_composition_evaluation_l617_617901

def g (x : ℤ) : ℤ :=
  if x < 5 then x^3 + x^2 - 6 else 2 * x - 18

theorem g_composition_evaluation : g (g (g 16)) = 2 := by
  sorry

end g_composition_evaluation_l617_617901


namespace max_mass_of_grain_l617_617603

theorem max_mass_of_grain (length width : ℝ) (angle : ℝ) (density : ℝ) 
  (h_length : length = 10) (h_width : width = 5) (h_angle : angle = 45) (h_density : density = 1200) : 
  volume * density = 175000 :=
by
  let height := width / 2
  let base_area := length * width
  let prism_volume := base_area * height
  let pyramid_volume := (1 / 3) * (width / 2 * length) * height
  let total_volume := prism_volume + 2 * pyramid_volume
  let volume := total_volume
  sorry

end max_mass_of_grain_l617_617603


namespace curve_three_lines_intersect_at_origin_l617_617950

theorem curve_three_lines_intersect_at_origin (a : ℝ) :
  ((∀ x y : ℝ, (x + 2 * y + a) * (x^2 - y^2) = 0 → 
    ((y = x ∨ y = -x ∨ y = - (1/2) * x - a/2) ∧ 
     (x = 0 ∧ y = 0)))) ↔ a = 0 :=
sorry

end curve_three_lines_intersect_at_origin_l617_617950


namespace terminal_side_of__l617_617149

theorem terminal_side_of_-1060_degrees :
  let angle : ℝ := -1060
  let revolutions : ℝ := 360
  let quadrant1_start : ℝ := 0
  let quadrant1_end : ℝ := 90
  (∃ k : ℤ, angle = k * revolutions + 20) ∧ (quadrant1_start < 20 ∧ 20 < quadrant1_end) → 
  (0º < 20 ∧ 20 < 90º) := 
by
  sorry

end terminal_side_of__l617_617149


namespace unique_triple_satisfies_conditions_l617_617378

theorem unique_triple_satisfies_conditions : 
  ∃! (a b c : ℤ), a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ 
                  (log a b = c^3) ∧ 
                  (a + b + c = 300) :=
sorry

end unique_triple_satisfies_conditions_l617_617378


namespace area_triangle_ABC_l617_617848

theorem area_triangle_ABC (AB CD height : ℝ) 
  (h_parallel : AB + CD = 20)
  (h_ratio : CD = 3 * AB)
  (h_height : height = (2 * 20) / (AB + CD)) :
  (1 / 2) * AB * height = 5 := sorry

end area_triangle_ABC_l617_617848


namespace trapezium_area_is_correct_l617_617724

/-- The lengths of the parallel sides of the trapezium. -/
def length1 : ℝ := 20
def length2 : ℝ := 18

/-- The distance between the parallel sides of the trapezium. -/
def height : ℝ := 12

/-- The formula for the area of the trapezium. -/
def trapezium_area (a b h : ℝ) : ℝ :=
  (1/2) * (a + b) * h

/-- The area of the given trapezium is 228 square centimeters. -/
theorem trapezium_area_is_correct :
  trapezium_area length1 length2 height = 228 := 
sorry

end trapezium_area_is_correct_l617_617724


namespace diagonal_divides_isosceles_not_rhombus_l617_617283

theorem diagonal_divides_isosceles_not_rhombus 
  (Q : Type) [quadrilateral Q] 
  (diags_isosceles : ∀ (d1 d2: diagonal Q), 
    divides_isosceles d1 Q ∧ divides_isosceles d2 Q) :
  ¬ rhombus Q :=
sorry

end diagonal_divides_isosceles_not_rhombus_l617_617283


namespace cyclic_trapezoid_is_isosceles_l617_617467

-- Definitions of points and circle
structure Point (α : Type) :=
(x : α) (y : α) 

structure Circle (α : Type) :=
(center : Point α) (radius : α)

-- Definition of a trapezoid
structure Trapezoid (α : Type) :=
(A B C D : Point α)
(base1 base2 : α)
(non_parallel_side1 non_parallel_side2 : α)

-- Condition: Trapezoid inscribed in a circle
def CyclicTrapezoid (α : Type) [LinearOrderedField α] 
  (T : Trapezoid α) (circle : Circle α) : Prop :=
∀ p ∈ {T.A, T.B, T.C, T.D}, ∃ r, (circle.center.x - p.x) ^ 2 + (circle.center.y - p.y) ^ 2 = r^2

-- Theorem: given a cyclic trapezoid, prove it's isosceles
theorem cyclic_trapezoid_is_isosceles (α : Type) [LinearOrderedField α] 
 (T : Trapezoid α) (circle : Circle α) 
 (h : CyclicTrapezoid α T circle) : T.non_parallel_side1 = T.non_parallel_side2 :=
by
  sorry

end cyclic_trapezoid_is_isosceles_l617_617467


namespace fifteenth_odd_multiple_of_5_is_145_l617_617187

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l617_617187


namespace sum_of_two_squares_if_and_only_if_same_parity_l617_617075

theorem sum_of_two_squares_if_and_only_if_same_parity 
  (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (Nat.even m ↔ Nat.even n) ↔ ∃ a b : ℤ, 5^m + 5^n = a^2 + b^2 :=
by
  sorry

end sum_of_two_squares_if_and_only_if_same_parity_l617_617075


namespace radical_conjugate_sum_l617_617690

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617690


namespace boys_more_than_girls_l617_617152

-- Definitions of the conditions
def total_students : ℕ := 100
def boy_ratio : ℕ := 3
def girl_ratio : ℕ := 2

-- Statement of the problem
theorem boys_more_than_girls :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) - (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 :=
by
  sorry

end boys_more_than_girls_l617_617152


namespace part1_part2_l617_617980

variable (a : ℕ → ℕ)
variable (T : ℕ → ℤ)

axiom sequence_a : a 1 = 1 ∧ ∀ n : ℕ, n > 0 → n * a (n + 1) = (n + 1) * a n + n * (n + 1)

def is_arithmetic (s : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d

theorem part1 : is_arithmetic (λ n, a n / n) := 
by
  sorry

theorem part2 : ∀ n : ℕ, n > 0 → T n = (-1)^(n+1) * nat.choose (n + 1) 2 :=
by
  sorry

end part1_part2_l617_617980


namespace quadratic_equation_root_is_conjugate_l617_617926

def quadratic_other_root (a b : ℤ) (α β : ℝ) : Prop :=
  (β = -1 / [\overline{b; a}])

theorem quadratic_equation_root_is_conjugate (a b : ℤ) (α β : ℝ)
  (h_int_coeff : ∃ p q : ℚ, p + q * sqrt d = [\overline{a; b}] ∧ α = p + q * sqrt d) :
  quadratic_other_root a b α β :=
sorry

end quadratic_equation_root_is_conjugate_l617_617926


namespace matrix_determinant_zero_l617_617259

theorem matrix_determinant_zero (a b : ℝ) : 
  Matrix.det ![
    ![1, Real.sin (2 * a), Real.sin a],
    ![Real.sin (2 * a), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 := 
by 
  sorry

end matrix_determinant_zero_l617_617259


namespace handshakes_and_highfives_count_l617_617227

theorem handshakes_and_highfives_count (n : ℕ) (h : n = 12) : 
  (∃ handshakes highfives : ℕ, handshakes = nat.choose n 2 ∧ highfives = nat.choose n 2 ∧ handshakes = 66 ∧ highfives = 66) :=
by 
  sorry

end handshakes_and_highfives_count_l617_617227


namespace norm_b_range_l617_617797

variables (a b : ℝ)
variables (A B : EuclideanSpace ℝ (Fin 2))

-- Norm condition on vector a
def norm_a : Prop := ∥A∥ = 2

-- Dot product condition
def dot_product_condition : Prop := (2 • A + B) ⬝ B = 12

-- Prove the range of norm of vector b is [2, 6]
theorem norm_b_range (h1 : norm_a) (h2 : dot_product_condition) : 2 ≤ ∥B∥ ∧ ∥B∥ ≤ 6 :=
sorry

end norm_b_range_l617_617797


namespace xy_sum_l617_617006

-- Define the problem conditions
variable (x y : ℚ)
variable (h1 : 1 / x + 1 / y = 4)
variable (h2 : 1 / x - 1 / y = -8)

-- Define the theorem to prove
theorem xy_sum : x + y = -1 / 3 := by
  sorry

end xy_sum_l617_617006


namespace integer_divisibility_l617_617722

open Nat

theorem integer_divisibility {a b : ℕ} :
  (2 * b^2 + 1) ∣ (a^3 + 1) ↔ a = 2 * b^2 + 1 := sorry

end integer_divisibility_l617_617722


namespace sum_of_number_and_its_radical_conjugate_l617_617661

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617661


namespace arith_to_geom_l617_617764

noncomputable def a (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

theorem arith_to_geom (m n : ℕ) (d : ℝ) 
  (h_pos : d > 0)
  (h_arith_seq : ∀ k : ℕ, a k d > 0)
  (h_geo_seq : (a 4 d + 5 / 2)^2 = (a 3 d) * (a 11 d))
  (h_mn : m - n = 8) : 
  a m d - a n d = 12 := 
sorry

end arith_to_geom_l617_617764


namespace mutually_exclusive_not_complementary_l617_617009

-- Definitions of events where each person receives a unique card color
inductive Person
| A | B | C | D

inductive Color
| Red | Yellow | Blue | White

open Person Color

def receives (p : Person) (c : Color) : Prop :=
  ∃ f : Person → Color, function.bijective f ∧ f p = c

theorem mutually_exclusive_not_complementary :
  (∀ f : Person → Color, function.bijective f →
    (receives A Red → ¬(receives D Red)) ∧
    ¬(¬(receives A Red) ↔ (receives D Red))) :=
by
  intros f f_bij
  split
  -- Proof of mutual exclusivity
  { intro hA
    intro hD
    -- A proof detail is required here, but we skip it with sorry.
    sorry }
  -- Proof of non-complementarity
  { intro h
    -- A proof detail is required here, but we skip it with sorry.
    sorry }

end mutually_exclusive_not_complementary_l617_617009


namespace number_of_young_fish_l617_617913

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end number_of_young_fish_l617_617913


namespace general_formula_for_sequence_l617_617801

def sequence (a : ℕ → ℕ) : Prop := 
  a 1 = 1 ∧ 
  ∀ n, n ≥ 2 → a n = 2 * (∑ i in finset.range (n), a (i+1))

theorem general_formula_for_sequence (a : ℕ → ℕ) (h : sequence a) :
  ∀ n, a n = if n = 1 then 1 else 2 * 3 ^ (n - 2) :=
sorry

end general_formula_for_sequence_l617_617801


namespace X_Y_Z_sum_eq_17_l617_617491

variable {X Y Z : ℤ}

def base_ten_representation_15_fac (X Y Z : ℤ) : Prop :=
  Z = 0 ∧ (28 + X + Y) % 9 = 8 ∧ (X - Y) % 11 = 11

theorem X_Y_Z_sum_eq_17 (X Y Z : ℤ) (h : base_ten_representation_15_fac X Y Z) : X + Y + Z = 17 :=
by
  sorry

end X_Y_Z_sum_eq_17_l617_617491


namespace triangle_y_difference_l617_617025

theorem triangle_y_difference :
  (∃ y : ℤ, y > 2 ∧ y < 16) →
  (y_max y_min : ℤ,
    (∀ y, (3 ≤ y ∧ y ≤ 15) → y_min ≤ y)
    ∧ (∀ y, (3 ≤ y ∧ y ≤ 15) → y ≤ y_max)
  )
  → y_max - y_min = 12 :=
begin
  sorry
end

end triangle_y_difference_l617_617025


namespace simplify_expression_l617_617120

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : (a^9 * a^15) / a^3 = a^21 :=
by sorry

end simplify_expression_l617_617120


namespace list_length_eq_12_l617_617376

-- Define a list of numbers in the sequence
def seq : List ℝ := [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5]

-- Define the theorem that states the number of elements in the sequence
theorem list_length_eq_12 : seq.length = 12 := 
by 
  -- Proof here
  sorry

end list_length_eq_12_l617_617376


namespace value_of_m_l617_617505

theorem value_of_m (m : ℝ) : (m + 1, 3) ∈ {p : ℝ × ℝ | p.1 + p.2 + 1 = 0} → m = -5 :=
by
  intro h
  sorry

end value_of_m_l617_617505


namespace distance_between_adjacent_symmetry_axes_l617_617274

def f (x : ℝ) : ℝ := cos (2 / 3 * x + π / 2) + cos (2 / 3 * x)

theorem distance_between_adjacent_symmetry_axes :
  let T := 3 * π in
  (T / 2) = (3 * π / 2) :=
by
  sorry

end distance_between_adjacent_symmetry_axes_l617_617274


namespace min_value_of_function_l617_617967

noncomputable def y (θ : ℝ) : ℝ := (2 - Real.sin θ) / (1 - Real.cos θ)

theorem min_value_of_function : ∃ θ : ℝ, y θ = 3 / 4 :=
sorry

end min_value_of_function_l617_617967


namespace spinner_even_product_probability_l617_617483

/-- Given two spinners A and B, where A has numbers {1, 2, 3, 4} and B has numbers {1, 2, 3}, and
both spinners are equally likely to land on any number, prove that the probability that the product of the two spinners' numbers is even is 2/3. -/
theorem spinner_even_product_probability :
  let A := {1, 2, 3, 4};
      B := {1, 2, 3};
      outcomes := [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (4, 1), (4, 2), (4, 3)];
      total_outcomes := 4 * 3;
      even_outcomes := outcomes.length in
  (even_outcomes / total_outcomes : ℚ) = 2 / 3 := 
sorry

end spinner_even_product_probability_l617_617483


namespace ratio_cost_to_marked_price_l617_617599

theorem ratio_cost_to_marked_price (x : ℝ) 
  (h_discount: ∀ y, y = marked_price → selling_price = (3/4) * y)
  (h_cost: ∀ z, z = selling_price → cost_price = (2/3) * z) :
  cost_price / marked_price = 1 / 2 :=
by
  sorry

end ratio_cost_to_marked_price_l617_617599


namespace product_of_digits_l617_617823

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : 8 ∣ (10 * A + B)) : A * B = 32 :=
sorry

end product_of_digits_l617_617823


namespace smallest_sum_of_two_3_digit_numbers_l617_617563

theorem smallest_sum_of_two_3_digit_numbers :
  ∃ (a b c d e f : ℕ), 
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
    (d ≠ e) ∧ (d ≠ f) ∧
    (e ≠ f) ∧
    {a, b, c, d, e, f} = {4, 5, 6, 7, 8, 9} ∧
    (100 * a + 10 * b + c) + (100 * d + 10 * e + f) = 1047 :=
by
  sorry

end smallest_sum_of_two_3_digit_numbers_l617_617563


namespace min_digits_fraction_l617_617561

theorem min_digits_fraction : 
  let num := 987654321
  let denom := 2^27 * 5^3
  ∃ (digits : ℕ), (10^digits * num = 987654321 * 2^27 * 5^3) ∧ digits = 27 := 
by
  sorry

end min_digits_fraction_l617_617561


namespace ones_digit_of_4567_times_3_is_1_l617_617971

theorem ones_digit_of_4567_times_3_is_1 :
  let n := 4567
  let m := 3
  (n * m) % 10 = 1 :=
by
  let n := 4567
  let m := 3
  have h : (n * m) % 10 = ((4567 * 3) % 10) := by rfl -- simplifying the product
  sorry -- this is where the proof would go, if required

end ones_digit_of_4567_times_3_is_1_l617_617971


namespace unique_real_solution_l617_617277

theorem unique_real_solution :
  ∀ x : ℝ, (x > 0 → (x ^ 16 + 1) * (x ^ 12 + x ^ 8 + x ^ 4 + 1) = 18 * x ^ 8 → x = 1) :=
by
  introv
  sorry

end unique_real_solution_l617_617277


namespace remaining_amount_spent_on_watermelons_l617_617239

def pineapple_cost : ℕ := 7
def total_spent : ℕ := 38
def pineapples_purchased : ℕ := 2

theorem remaining_amount_spent_on_watermelons:
  total_spent - (pineapple_cost * pineapples_purchased) = 24 :=
by
  sorry

end remaining_amount_spent_on_watermelons_l617_617239


namespace sum_local_values_l617_617218

theorem sum_local_values :
  let local_value_2 := 2000
  let local_value_3 := 300
  let local_value_4 := 40
  let local_value_5 := 5
  local_value_2 + local_value_3 + local_value_4 + local_value_5 = 2345 :=
by
  sorry

end sum_local_values_l617_617218


namespace total_young_fish_l617_617909

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end total_young_fish_l617_617909


namespace side_length_of_square_l617_617489

theorem side_length_of_square (s : ℝ) (h : s^2 = 6 * (4 * s)) : s = 24 :=
by sorry

end side_length_of_square_l617_617489


namespace angle_between_midpoints_of_arcs_eq_90_l617_617165

noncomputable def circle : Type := sorry
noncomputable def midpoint (x y : circle) : circle := sorry

def two_circles_meet_at_two_points (O O' : circle) (A B : circle) : Prop := sorry

def line_from_A_intersects_at_C_D (A : circle) (O O' : circle) (C D : circle) (between : ∀ {X Y Z : circle}, Prop): Prop := 
between A C D ∧ ∃ x y, between A x y ∧ between y z d ∧ ∃ (C = x) ∧ (D = y)
 
def midpoint_of_arcs (B C D M N : circle) : Prop :=
∃ arcBC arcBD, midpoint B C = M ∧ midpoint B D = N

def midpoint_of_segment (C D K : circle) : Prop := midpoint C D = K

def angle_KMN_eq_90deg (K M N : circle) : Prop :=
sorry -- To be proven

theorem angle_between_midpoints_of_arcs_eq_90 
  (O O' A B C D M N K: circle)
  (h1 : two_circles_meet_at_two_points O O' A B)
  (h2 : line_from_A_intersects_at_C_D A O O' C D --between condition must be stated here)
  (h3 : midpoint_of_arcs B C D M N)
  (h4 : midpoint_of_segment C D K)
  : angle_KMN_eq_90deg K M N := 
sorry

end angle_between_midpoints_of_arcs_eq_90_l617_617165


namespace smallest_frequency_l617_617713

theorem smallest_frequency (x : ℝ) :
  let total_sum := 1
  let common_diff := 0.05
  let sum_last_seven := 0.79
  ∃ (x : ℝ), 3*x + 2*common_diff = total_sum - sum_last_seven → x = 0.02 := 
by 
  let total_sum := 1
  let common_diff := 0.05
  let sum_last_seven := 0.79
  existsi (0.02)
  intro h
  have h_eq: 3 * 0.02 + 2 * common_diff = total_sum - sum_last_seven := sorry
  exact h_eq

end smallest_frequency_l617_617713


namespace length_of_AB_l617_617856

noncomputable def length_AB : ℝ :=
√((1 + 1) * ((-4)^2 - 4 * (-4)))

theorem length_of_AB 
  (ρ θ : ℝ) 
  (t : ℝ) 
  (x y : ℝ)
  (hM : √2 * ρ * cos (θ + π / 4) = 1)
  (hN : x = 4 * t^2 ∧ y = 4 * t) 
  (intersect : (∃ x y, x - y = 1 ∧ y^2 = 4 * x)) :
  length_AB = 8 := 
by 
-- Definitions from conditions
let x := y + 1,
have eqn : y^2 - 2*y - 4 = 0, from sorry,
sorry

end length_of_AB_l617_617856


namespace exists_weighted_sum_divisible_by_9_l617_617292

theorem exists_weighted_sum_divisible_by_9 :
  ∃ (n : ℕ), n = 12 ∧ ∀ (a : Fin n → ℤ), ∃ (idxs : Fin 9 → Fin n) (b : Fin 9 → {b : ℤ // b = 4 ∨ b = 7}),
  (∑ i, b i * a (idxs i)) % 9 = 0 :=
by
  sorry

end exists_weighted_sum_divisible_by_9_l617_617292


namespace law_of_change_x_y_l617_617108

-- Define the initial conditions and the proportional rates
def conditions (t : ℝ) (x y a k1 k2 : ℝ) : Prop :=
  (∃ t=0, x = 0 ∧ y = 0) ∧
  (∃ t=1, x = a / 8 ∧ y = 3 * a / 8) ∧
  (k1 = (Real.log 2) / 4 ∧ k2 = 3 * (Real.log 2) / 4) ∧
  (∀ t, x = a / 4 * (1 - 2^(-t)) ∧ y = 3 * a / 4 * (1 - 2^(-t)))

-- Define lean theorem to prove the conditions yield the given answers for x and y
theorem law_of_change_x_y (a k1 k2 : ℝ) : 
  ∀ t : ℝ, 
  ∃ x y : ℝ, 
  conditions t x y a k1 k2 → 
  x = a / 4 * (1 - 2^(-t)) ∧ y = 3 * a / 4 * (1 - 2^(-t)) :=
begin
  intros t,
  sorry -- proof goes here
end

end law_of_change_x_y_l617_617108


namespace is_not_prime_390629_l617_617416

theorem is_not_prime_390629 : ¬ Prime 390629 :=
sorry

end is_not_prime_390629_l617_617416


namespace value_at_points_zero_l617_617632

def odd_function (v : ℝ → ℝ) := ∀ x : ℝ, v (-x) = -v x

theorem value_at_points_zero (v : ℝ → ℝ)
  (hv : odd_function v) :
  v (-2.1) + v (-1.2) + v (1.2) + v (2.1) = 0 :=
by {
  sorry
}

end value_at_points_zero_l617_617632


namespace calc_Delta_3_4_l617_617821

def Delta (x y : ℝ) : ℝ := (x + y) / (1 + x * y + x^2 * y^2)

theorem calc_Delta_3_4 : Delta 3 4 = 7 / 157 :=
by
  unfold Delta
  rw [← div_eq_div_iff]
  norm_num
  sorry

end calc_Delta_3_4_l617_617821


namespace find_mistake_l617_617992

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l617_617992


namespace priya_speed_l617_617470

theorem priya_speed (S_R S_P distance : ℝ) 
  (h1 : S_R = 21) 
  (h2 : distance = 43) 
  : S_P = 22 :=
begin
  -- Definitions
  let time := 1, -- in hours
  
  -- Distance covered by Riya
  let distance_riya := S_R * time,
  
  -- Definition of total distance
  have h3 : distance_riya = 21, from calc
    distance_riya = S_R * 1 : by rw mul_one
    ... = 21 : by rw h1,

  -- Definition of Priya's distance
  let distance_priya := distance - distance_riya,
  
  -- Using the given distances
  have h4 : distance_priya = 22, from calc
    distance_priya = distance - distance_riya : by refl
    ... = 43 - 21 : by rw [h2, h3]
    ... = 22 : by norm_num,
    
  -- Calculation of Priya's speed
  let speed_priya := distance_priya / time,
  
  show S_P = 22, from calc
    S_P = distance_priya / 1 : by refl
    ... = 22 : by rw [h4, div_one]
end

end priya_speed_l617_617470


namespace D_seventy_two_l617_617057

def D (n : ℕ) : ℕ := 
-- Definition of D will be filled in

theorem D_seventy_two : D 72 = 121 := 
sorry

end D_seventy_two_l617_617057


namespace Nellie_lost_legos_l617_617088

def initial_legos : ℕ := 380
def legos_given_away : ℕ := 24
def legos_remaining : ℕ := 299

def legos_expected : ℕ := initial_legos - legos_given_away

theorem Nellie_lost_legos : legos_expected - legos_remaining = 57 := by
  simp [initial_legos, legos_given_away, legos_remaining, legos_expected]
  sorry  -- This is where the proof would go.

end Nellie_lost_legos_l617_617088


namespace stratified_sampling_difference_l617_617520

theorem stratified_sampling_difference
  (male_athletes : ℕ := 56)
  (female_athletes : ℕ := 42)
  (sample_size : ℕ := 28)
  (H_total : male_athletes + female_athletes = 98)
  (H_sample_frac : sample_size = 28)
  : (56 * (sample_size / 98) - 42 * (sample_size / 98) = 4) :=
sorry

end stratified_sampling_difference_l617_617520


namespace halfway_fraction_l617_617174

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l617_617174


namespace clock_angle_at_230_l617_617555

/-- Angle calculation problem: Determine the degree measure of the smaller angle formed by the 
    hands of a clock at 2:30. -/
theorem clock_angle_at_230 : 
  let angle_per_hour := 360 / 12,
      hour_position := 2 * angle_per_hour + (angle_per_hour / 2),
      minute_position := 30 * 6,
      angle_difference := abs (minute_position - hour_position)
  in angle_difference = 105 :=
by
  sorry

end clock_angle_at_230_l617_617555


namespace work_completion_days_l617_617590

theorem work_completion_days (A_days B_days : ℕ) (hA : A_days = 3) (hB : B_days = 6) : 
  (1 / ((1 / (A_days : ℚ)) + (1 / (B_days : ℚ)))) = 2 := 
by
  sorry

end work_completion_days_l617_617590


namespace f_at_1_f_at_m_f_inequality_l617_617446

variable (f : ℝ → ℝ)

-- Conditions
-- 1. Increasing function on (0, +∞)
axiom f_increasing : ∀ ⦃a b : ℝ⦄, (0 < a ∧ 0 < b ∧ a < b) → f(a) < f(b)

-- 2. Functional equation
axiom f_additive : ∀ ⦃x y : ℝ⦄, (0 < x ∧ 0 < y) → f(x * y) = f(x) + f(y)

-- 3. Specific value
axiom f_at_4 : f 4 = 1

-- 1. Find the value of f(1)
theorem f_at_1 : f 1 = 0 :=
sorry

-- 2. Find the value of m such that f(m) = 2
theorem f_at_m (m : ℝ) (h : f m = 2) : m = 16 :=
sorry

-- 3. Find the range of x such that f(x^2 - 4x - 5) < 2
theorem f_inequality (x : ℝ) (h : f (x^2 - 4 * x - 5) < 2) : -3 < x ∧ x < -1 ∨ 5 < x ∧ x < 7 :=
sorry

end f_at_1_f_at_m_f_inequality_l617_617446


namespace parabola_constants_sum_l617_617493

-- Definition based on the given conditions
structure Parabola where
  a: ℝ
  b: ℝ
  c: ℝ
  vertex_x: ℝ
  vertex_y: ℝ
  point_x: ℝ
  point_y: ℝ

-- Definitions of the specific parabola based on the problem's conditions
noncomputable def givenParabola : Parabola := {
  a := -1/4,
  b := -5/2,
  c := -1/4,
  vertex_x := 6,
  vertex_y := -5,
  point_x := 2,
  point_y := -1
}

-- Theorem proving the required value of a + b + c
theorem parabola_constants_sum : givenParabola.a + givenParabola.b + givenParabola.c = -3.25 :=
  by
  sorry

end parabola_constants_sum_l617_617493


namespace danny_thrice_jane_19_years_ago_l617_617269

-- Defining the ages of Danny and Jane
def danny_age : ℕ := 40
def jane_age : ℕ := 26

-- The hypothesis is that there exists a certain number of years ago x such that
-- Danny's age was thrice Jane's age
theorem danny_thrice_jane_19_years_ago :
  ∃ x : ℕ, danny_age - x = 3 * (jane_age - x) ∧ x = 19 :=
by
  have h1 : danny_age - 19 = 3 * (jane_age - 19), from sorry
  use 19
  exact ⟨h1, rfl⟩

end danny_thrice_jane_19_years_ago_l617_617269


namespace problem_statement_l617_617612

noncomputable theory

-- Definitions of the circle centers and their respective properties
variables (E F G H : Type) -- Centers of circles A, B, C, D respectively
variables (P Q R S : Type) -- Tangency points
variables (r₁ r₂ r₃ r₄ : ℝ)  -- Radii of circles A, B, C, D
variables (d_AC d_EF d_FG d_GH d_HE : ℝ) -- Distances between the centers

-- Given conditions
def circles_touch_externally (A B : Type) (P : Type) :=
  -- P is the point where A and B touch externally and their centers are collinear with P
  sorry

def quadrilateral_is_cyclic (P Q R S : Type) : Prop :=
  -- Definition of a cyclic quadrilateral where opposite angles sum to 180°
  sorry

def area_of_quadrilateral (P Q R S : Type) : ℝ :=
  -- Computation of the quadrilateral's area
  sorry

-- Problem statement
theorem problem_statement
  (circles_touch_externally E F P)
  (circles_touch_externally F G Q)
  (circles_touch_externally G H R)
  (circles_touch_externally H E S)
  (r₁ = 2) (r₂ = 3) (r₃ = 2) (r₄ = 3)
  (d_AC = 6)
  (d_EF = 5) (d_FG = 5) (d_GH = 5) (d_HE = 5) :
  quadrilateral_is_cyclic P Q R S ∧ area_of_quadrilateral P Q R S = 15 := by
  sorry

end problem_statement_l617_617612


namespace decreasing_functions_range_l617_617383

theorem decreasing_functions_range {a : ℝ} (h1 : ∀ x ∈ Icc 1 2, deriv (-x^2 + 2 * a * x) ≤ 0)
  (h2 : ∀ x ∈ Icc 1 2, deriv ((a + 1) ^ (1 - x)) ≤ 0) :
  0 < a ∧ a ≤ 1 :=
sorry

end decreasing_functions_range_l617_617383


namespace botanist_needs_more_flowers_l617_617588

theorem botanist_needs_more_flowers :
  ∃ n : ℕ, (601 + n) % 8 = 0 :=
by {
  let n := 7,
  use n,
  -- Goal is now (601 + 7) % 8 = 0
  show (601 + n) % 8 = 0,
  calc
    (601 + n) % 8 = 608 % 8 : by rw [add_comm, nat.add_mod] 
            ... = 0          : rfl
}

end botanist_needs_more_flowers_l617_617588


namespace remainder_of_125_div_j_l617_617744

theorem remainder_of_125_div_j (j : ℕ) (h1 : j > 0) (h2 : 75 % (j^2) = 3) : 125 % j = 5 :=
sorry

end remainder_of_125_div_j_l617_617744


namespace sum_of_conjugates_l617_617655

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617655


namespace right_triangle_area_l617_617975

theorem right_triangle_area (a b c: ℝ) (h1: c = 2) (h2: a + b + c = 2 + Real.sqrt 6) (h3: (a * b) / 2 = 1 / 2) :
  (1 / 2) * (a * b) = 1 / 2 :=
by
  -- Sorry is used to skip the proof
  sorry

end right_triangle_area_l617_617975


namespace find_mistake_l617_617997

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l617_617997


namespace solve_cos_sin_eq_one_l617_617475

open Real

theorem solve_cos_sin_eq_one (n : ℕ) (hn : n > 0) :
  {x : ℝ | cos x ^ n - sin x ^ n = 1} = {x : ℝ | ∃ k : ℤ, x = k * π} :=
by
  sorry

end solve_cos_sin_eq_one_l617_617475


namespace factorial_sum_divisor_power_five_l617_617969

theorem factorial_sum_divisor_power_five :
  let N := 102 in
  let sum_factorials := N! + (N + 1)! + (N + 2)! + (N + 3)! in
  let power_of_five (n : Nat) : Nat :=
    if n < 5 then 0 else (n / 5) + power_of_five (n / 5) in
  let power_of_five_sum := power_of_five N in
  let factorial_power := power_of_five_sum + power_of_five (N / 25) in
  Nat.find (λ n, 5^n ∣ sum_factorials) = 24 :=
by
  sorry

end factorial_sum_divisor_power_five_l617_617969


namespace solve_system_of_equations_l617_617940

theorem solve_system_of_equations:
  ∃ (x y z : ℝ), 
  x + y - z = 4 ∧
  x^2 + y^2 - z^2 = 12 ∧
  x^3 + y^3 - z^3 = 34 ∧
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l617_617940


namespace distance_between_l1_l2_l617_617363

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  (abs (C2 - C1)) / (real.sqrt (A^2 + B^2))

theorem distance_between_l1_l2 :
  distance_between_parallel_lines 1 1 (-1) 1 = real.sqrt 2 :=
by
  -- Definition of distance_between_parallel_lines is used here
  sorry

end distance_between_l1_l2_l617_617363


namespace boys_girls_students_l617_617154

theorem boys_girls_students (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : ratio_boys = 3)
  (h3 : ratio_girls = 2) :
  3 * (total_students / (ratio_boys + ratio_girls)) - 2 * (total_students / (ratio_boys + ratio_girls)) = 20 :=
by
  sorry

end boys_girls_students_l617_617154


namespace find_n_l617_617628

-- Define the conditions as hypothesis
variables (A B n : ℕ)

-- Hypothesis 1: This year, Ana's age is the square of Bonita's age.
-- A = B^2
#check (A = B^2) 

-- Hypothesis 2: Last year Ana was 5 times as old as Bonita.
-- A - 1 = 5 * (B - 1)
#check (A - 1 = 5 * (B - 1))

-- Hypothesis 3: Ana and Bonita were born n years apart.
-- A = B + n
#check (A = B + n)

-- Goal: The difference in their ages, n, should be 12.
theorem find_n (A B n : ℕ) (h1 : A = B^2) (h2 : A - 1 = 5 * (B - 1)) (h3 : A = B + n) : n = 12 :=
sorry

end find_n_l617_617628


namespace max_value_of_x_neg_two_in_interval_l617_617507

noncomputable def x_neg_two_max_value_in_interval (a b : ℝ) (f : ℝ → ℝ) :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ 4

theorem max_value_of_x_neg_two_in_interval:
  x_neg_two_max_value_in_interval (1/2) 2 (λ x, x^(-2)) :=
by {
  intros x hx,
  have h1 : (1 / 2 : ℝ) > 0 := by linarith,
  have h2 : (2 : ℝ) > 0 := by linarith,
  have hx1 : x^(-2) ≤ 4, {
    rw [← one_div_eq_inv, ← one_div_eq_inv, div_le_div_iff],
    calc
      x^(-2) = 1 / x^2 : by rw [← rpow_neg (-2) x, rpow_nat_cast, rpow_neg (-2)], sorry, apply or.intro_right, linarith, sorry},
  exact hx1
}

end max_value_of_x_neg_two_in_interval_l617_617507


namespace quadratic_roots_x_no_real_solution_y_l617_617939

theorem quadratic_roots_x (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1) := sorry

theorem no_real_solution_y (y : ℝ) : 
  ¬∃ y : ℝ, 4*y^2 - 3*y + 2 = 0 := sorry

end quadratic_roots_x_no_real_solution_y_l617_617939


namespace cupboard_cost_price_l617_617546

noncomputable def cost_price_of_cupboard (C : ℝ) : Prop :=
  let SP := 0.88 * C
  let NSP := 1.12 * C
  NSP - SP = 1650

theorem cupboard_cost_price : ∃ (C : ℝ), cost_price_of_cupboard C ∧ C = 6875 := by
  sorry

end cupboard_cost_price_l617_617546


namespace MP_plus_NQ_eq_BR_l617_617457

-- Declare points and segments on the triangle
variables {A B C P Q R M N: Type*}

-- Conditions from the problem
variables [between P A Q]
variables [on_segment R B C]
variables [intersection_points M A R C P]
variables [intersection_points N A R C Q]
variables (BC_eq_BQ : BC = BQ)
variables (CP_eq_AP : CP = AP)
variables (CR_eq_CN : CR = CN)
variables (angle_BPC_eq_angle_CRA : ∠BPC = ∠CRA)

-- Statement to prove
theorem MP_plus_NQ_eq_BR
  (BC_eq_BQ : BC = BQ)
  (CP_eq_AP : CP = AP)
  (CR_eq_CN : CR = CN)
  (angle_BPC_eq_angle_CRA : ∠BPC = ∠CRA)
  : MP + NQ = BR :=
sorry

end MP_plus_NQ_eq_BR_l617_617457


namespace initial_pens_count_l617_617213

theorem initial_pens_count (P : ℕ) (h : 2 * (P + 22) - 19 = 75) : P = 25 :=
by
  sorry

end initial_pens_count_l617_617213


namespace unique_cell_distance_50_l617_617952

noncomputable def king_dist (A B: ℤ × ℤ) : ℤ :=
  max (abs (A.1 - B.1)) (abs (A.2 - B.2))

theorem unique_cell_distance_50
  (A B C: ℤ × ℤ)
  (hAB: king_dist A B = 100)
  (hBC: king_dist B C = 100)
  (hCA: king_dist C A = 100) :
  ∃! (X: ℤ × ℤ), king_dist X A = 50 ∧ king_dist X B = 50 ∧ king_dist X C = 50 :=
sorry

end unique_cell_distance_50_l617_617952


namespace melody_initial_food_l617_617084

-- Conditions
variable (dogs : ℕ) (food_per_meal : ℚ) (meals_per_day : ℕ) (days_in_week : ℕ) (food_left : ℚ)
variable (initial_food : ℚ)

-- Values given in the problem statement
axiom h_dogs : dogs = 3
axiom h_food_per_meal : food_per_meal = 1/2
axiom h_meals_per_day : meals_per_day = 2
axiom h_days_in_week : days_in_week = 7
axiom h_food_left : food_left = 9

-- Theorem to prove
theorem melody_initial_food : initial_food = 30 :=
  sorry

end melody_initial_food_l617_617084


namespace men_in_room_l617_617034
noncomputable def numMenInRoom (x : ℕ) : ℕ := 4 * x + 2

theorem men_in_room (x : ℕ) (h_initial_ratio : true) (h_after_events : true) (h_double_women : 2 * (5 * x - 3) = 24) :
  numMenInRoom x = 14 :=
sorry

end men_in_room_l617_617034


namespace tangent_line_sin_pi_l617_617728

theorem tangent_line_sin_pi :
  (∀ x y : ℝ, y = sin x → x = π → x + y - π = 0) :=
sorry

end tangent_line_sin_pi_l617_617728


namespace diff_roots_eq_sqrt_2p2_add_2p_sub_2_l617_617273

theorem diff_roots_eq_sqrt_2p2_add_2p_sub_2 (p : ℝ) :
  let a := 1
  let b := -2 * p
  let c := p^2 - p + 1
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let r1 := (-b + sqrt_discriminant) / (2 * a)
  let r2 := (-b - sqrt_discriminant) / (2 * a)
  r1 - r2 = Real.sqrt (2*p^2 + 2*p - 2) :=
by
  sorry

end diff_roots_eq_sqrt_2p2_add_2p_sub_2_l617_617273


namespace part1_part2_l617_617359

noncomputable def f (a : ℝ) (a_pos : a > 1) (x : ℝ) : ℝ :=
  a^x + (x - 2) / (x + 1)

-- Statement for part 1
theorem part1 (a : ℝ) (a_pos : a > 1) : ∀ x : ℝ, -1 < x → f a a_pos x ≤ f a a_pos (x + ε) → 0 < ε := sorry

-- Statement for part 2
theorem part2 (a : ℝ) (a_pos : a > 1) : ¬ ∃ x : ℝ, x < 0 ∧ f a a_pos x = 0 := sorry

end part1_part2_l617_617359


namespace boys_girls_students_l617_617153

theorem boys_girls_students (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : ratio_boys = 3)
  (h3 : ratio_girls = 2) :
  3 * (total_students / (ratio_boys + ratio_girls)) - 2 * (total_students / (ratio_boys + ratio_girls)) = 20 :=
by
  sorry

end boys_girls_students_l617_617153


namespace range_of_a_l617_617464

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0

def neg_p : Prop := ∃ x : ℝ, a * x^2 + a * x + 1 < 0

theorem range_of_a (h : neg_p a) : a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
  sorry

end range_of_a_l617_617464


namespace coffee_table_price_correct_l617_617048

-- Conditions
def sofa_cost : ℕ := 1250
def armchair_cost_each : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Question: What is the price of the coffee table?
def coffee_table_price : ℕ := total_invoice - (sofa_cost + num_armchairs * armchair_cost_each)

-- Proof statement (to be completed)
theorem coffee_table_price_correct : coffee_table_price = 330 := by
  sorry

end coffee_table_price_correct_l617_617048


namespace value_range_f_l617_617528

def f (x : ℝ) : ℝ := -x^2 + 2 * x - 3

theorem value_range_f : {y : ℝ | ∃ x ∈ set.Icc 0 2, f x = y} = set.Icc (-3) (-2) :=
by
  sorry

end value_range_f_l617_617528


namespace trigonometric_identity_proof_l617_617223

theorem trigonometric_identity_proof :
  sin (4 * π / 3) * cos (5 * π / 6) * tan (-4 * π / 3) = -3 * sqrt 3 / 4 :=
by
  -- Proof goes here
  sorry

end trigonometric_identity_proof_l617_617223


namespace no_integer_roots_l617_617065

theorem no_integer_roots (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 :=
by {
  sorry
}

end no_integer_roots_l617_617065


namespace max_value_of_f_l617_617730

def f (x : ℝ) : ℝ := (Real.cos x) ^ 3 + (Real.sin x) ^ 2 - (Real.cos x)

theorem max_value_of_f :
  ∃ t ∈ Icc (-1:ℝ) 1, f (Real.arccos t) = 32/27 := by
  sorry

end max_value_of_f_l617_617730


namespace sufficient_condition_for_zero_l617_617903
open Real

theorem sufficient_condition_for_zero (a : ℝ) : 
  (1 < a ∧ a < 3) → (∃ x ∈ Ioo 2 8, log 0.5 x + x - a = 0) :=
sorry

end sufficient_condition_for_zero_l617_617903


namespace radical_conjugate_sum_l617_617696

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617696


namespace question_a_gt_b_neither_sufficient_nor_necessary_l617_617438

theorem question_a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by
  sorry

end question_a_gt_b_neither_sufficient_nor_necessary_l617_617438


namespace first_type_material_needed_l617_617706

def total_material_used : ℝ := 0.23529411764705882
def second_type_material : ℝ := 0.3
def leftover_material : ℝ := 0.3
def total_material_bought : ℝ := total_material_used + leftover_material

theorem first_type_material_needed :
  (total_material_bought - second_type_material = 0.2352941176470588) :=
by
  have total_material : ℝ := total_material_used + leftover_material,
  have first_type_material : ℝ := total_material - second_type_material,
  rw [show total_material = 0.23529411764705882 + 0.3, by refl],
  rw [show first_type_material = total_material - second_type_material, by refl],
  sorry

end first_type_material_needed_l617_617706


namespace floor_e_squared_l617_617288

theorem floor_e_squared : Real.floor (Real.exp 2) = 7 :=
by
  sorry

end floor_e_squared_l617_617288


namespace junior_score_is_90_l617_617399

-- Representations for the conditions
variable (n : ℕ)  -- Total number of students
variable (juniors seniors : ℕ) -- Number of juniors and seniors
variable (avg_junior avg_senior avg_total : ℚ) -- Average scores

-- Conditions as hypotheses
-- 20% of the students are juniors
axiom H1 : juniors = 0.2 * n
-- 80% of the students are seniors
axiom H2 : seniors = 0.8 * n
-- The average score of all students is 86
axiom H3 : avg_total = 86
-- The average score of the seniors is 85
axiom H4 : avg_senior = 85
-- All juniors have the same score, which is avg_junior
axiom H5 : avg_junior = avg_junior

-- Prove that the score each of the juniors received on the test is 90
theorem junior_score_is_90 : avg_junior = 90 := sorry

end junior_score_is_90_l617_617399


namespace possible_remainders_when_divided_by_3_l617_617567

theorem possible_remainders_when_divided_by_3 (n : ℤ) : 
  let remainder := n % 3 in 
  remainder ≠ 0 → (remainder = 1 ∨ remainder = 2) :=
by
  intros remainder_nonzero
  sorry

end possible_remainders_when_divided_by_3_l617_617567


namespace problem_part_1_problem_part_2_problem_part_3_l617_617028

def is_equidistant (P Q : ℝ × ℝ) : Prop :=
  max (abs P.1) (abs P.2) = max (abs Q.1) (abs Q.2)

-- Problem part 1
def equidistant_points_1 (A P Q R: ℝ × ℝ) : Prop :=
  is_equidistant A P ∧ is_equidistant A Q ∧ ¬ is_equidistant A R

-- Problem part 2
def equidistant_points_2 (B C : ℝ × ℝ) : Prop :=
  ∃ m, (C = ((m-1), m) ∧ is_equidistant B C)

-- Problem part 3
def equidistant_points_3 (D E : ℝ × ℝ) : Prop :=
  ∃ k, (D = (3, 4+k) ∧ E = (2k-5, 6) ∧ is_equidistant D E)

-- Statements -- No proof needed
theorem problem_part_1 :
  equidistant_points_1 (-3, 7) (3, -7) (7, 4) (2, 9) :=
sorry

theorem problem_part_2 :
  equidistant_points_2 (-4, 2) (-4, -3) ∨ equidistant_points_2 (-4, 2) (3, 4) :=
sorry

theorem problem_part_3 :
  equidistant_points_3 (3, 4+2) (2*2-5, 6) ∨ equidistant_points_3 (3, 4+9) (2*9-5, 6) :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l617_617028


namespace division_problem_l617_617229

theorem division_problem : 8900 / 6 / 4 = 1483.3333 :=
by sorry

end division_problem_l617_617229


namespace conditional_probability_B0_conditional_probability_B1_conditional_probability_B2_probability_distribution_X_l617_617527

open locale classical
open ProbabilityTheory

-- Definitions of the conditions
def event_A (boxes : Finset ℕ) (checked_boxes : Finset ℕ) : Prop :=
  ∀ i ∈ checked_boxes, boxes i = 0

def event_B0 (boxes : Finset ℕ) : Prop :=
  ∀ i ∈ boxes, boxes i = 0

def event_B1 (boxes : Finset ℕ) : Prop :=
  ∃ i ∈ boxes, boxes i = 1 ∧ ∀ j ∈ boxes, j ≠ i → boxes j = 0

def event_B2 (boxes : Finset ℕ) : Prop :=
  ∃ i j ∈ boxes, i ≠ j ∧ boxes i = 1 ∧ boxes j = 1 ∧ ∀ k ∈ boxes, k ≠ i ∧ k ≠ j → boxes k = 0

noncomputable def P (boxes : Finset ℕ) (p : Prop) : ℝ := sorry

-- Proof statements
theorem conditional_probability_B0 (boxes : Finset ℕ) (checked_boxes : Finset ℕ) :
  (event_B0 boxes → event_A boxes checked_boxes) = 1 := sorry

theorem conditional_probability_B1 (boxes : Finset ℕ) (checked_boxes : Finset ℕ) :
  (event_B1 boxes → event_A boxes checked_boxes) = 4 / 5 := sorry

theorem conditional_probability_B2 (boxes : Finset ℕ) (checked_boxes : Finset ℕ) :
  (event_B2 boxes → event_A boxes checked_boxes) = 12 / 19 := sorry

theorem probability_distribution_X :
  ∀ (X : ℕ), 
    (X = 0 → P X (X = 0) = 877 / 950) ∧ 
    (X = 1 → P X (X = 1) = 70 / 950) ∧ 
    (X = 2 → P X (X = 2) = 3 / 950) := sorry

end conditional_probability_B0_conditional_probability_B1_conditional_probability_B2_probability_distribution_X_l617_617527


namespace dwarf_hats_minimum_risk_l617_617961

theorem dwarf_hats_minimum_risk (p : ℕ) (h : p < ∞) :
  ∃ strategy : (list (fin 2)) → (list (fin 2) × list (fin 2)),
  (∀ hats : list (fin 2), hats.length = p →
   (let (first_dwarf_risk, remaining_dwarves) := strategy hats in
    ∃ at_risk : fin p → Prop,
      (at_risk 0 = (first_dwarf_risk ≠ hats.head)) ∧
      (∀ i : fin (p - 1), ¬ at_risk (i + 1)))) ∧
  ∃ (min_risk_dwarves : nat), min_risk_dwarves = 1
:= by
sorry

end dwarf_hats_minimum_risk_l617_617961


namespace sum_of_number_and_its_radical_conjugate_l617_617659

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617659


namespace range_of_a_l617_617749

def P (a : ℝ) : Set ℝ := { x : ℝ | a - 4 < x ∧ x < a + 4 }
def Q : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 < 0 }

theorem range_of_a (a : ℝ) : (∀ x, Q x → P a x) → -1 < a ∧ a < 5 :=
by
  intro h
  sorry

end range_of_a_l617_617749


namespace probability_of_valid_pairs_l617_617541

open MeasureTheory Probability

noncomputable def fair_dice_space : MassFunction (ℕ × ℕ) :=
  MassFunction.uniform (finset.univ.product finset.univ)

def valid_pair (p : ℕ × ℕ) : Prop :=
  let (a, b) := p in
  a + b ≤ 10 ∧ (a > 3 ∨ b > 3)

theorem probability_of_valid_pairs : 
  fair_dice_space.probOf valid_pair = 2 / 3 :=
sorry

end probability_of_valid_pairs_l617_617541


namespace count_valid_numbers_l617_617169

def digits := {0, 1, 2, 3, 4}

noncomputable def is_valid_number (n : List ℕ) : Prop :=
  n.length = 5 ∧
  (∀ d, d ∈ n → d ∈ digits) ∧
  n.nodup ∧
  (∃ (i : ℕ), i < 4 ∧
    (n[i] % 2 = 1 ∧ n[i+1] % 2 = 0 ∧ n[i+2] % 2 = 1))

theorem count_valid_numbers : ∃ s : Finset (List ℕ), s.card = 28 ∧ (∀ x ∈ s, is_valid_number x) :=
sorry

end count_valid_numbers_l617_617169


namespace complete_the_square_l617_617211

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x^2 + 10 * x - 3 = 0) → ((x + a)^2 = b) ∧ b = 28) :=
sorry

end complete_the_square_l617_617211


namespace farmer_shipped_30_boxes_this_week_l617_617873

-- Defining the given conditions
def last_week_boxes : ℕ := 10
def last_week_pomelos : ℕ := 240
def this_week_dozen : ℕ := 60
def pomelos_per_dozen : ℕ := 12

-- Translating conditions into mathematical statements
def pomelos_per_box_last_week : ℕ := last_week_pomelos / last_week_boxes
def this_week_pomelos_total : ℕ := this_week_dozen * pomelos_per_dozen
def boxes_shipped_this_week : ℕ := this_week_pomelos_total / pomelos_per_box_last_week

-- The theorem we prove, that given the conditions, the number of boxes shipped this week is 30.
theorem farmer_shipped_30_boxes_this_week :
  boxes_shipped_this_week = 30 :=
sorry

end farmer_shipped_30_boxes_this_week_l617_617873


namespace triangle_ratio_sum_l617_617859

noncomputable theory
open_locale classical

variables {A B C D E F : Type} [linear_ordered_field Type]

-- Conditions specified in the problem
def divides_BC (BD DC : ℝ) (r : ℝ) := BD / DC = r
def divides_AB_equally (AB AE EB : ℝ) := AE = EB
def point_on_segment (A B C : Type) := true  -- For simplicity, we assume they lie on a segment

-- The problem statement, proving the summed ratio
theorem triangle_ratio_sum (BD DC AB AE EB EF FC AF FD : ℝ)
  (h1 : divides_BC BD DC (1/2))
  (h2 : divides_AB_equally AB AE EB)
  (h3 : point_on_segment A B C)
  (h4 : point_on_segment B C D)
  (h5 : point_on_segment A B E) :
  EF / FC + AF / FD = 7 / 4 :=
sorry

end triangle_ratio_sum_l617_617859


namespace correct_propositions_l617_617638
theorem correct_propositions (a b x y : ℝ): 
  (|a - b| < 1 → |a| < |b| + 1) ∧
  (|a + b| - 2 * |a| ≤ |a - b|) ∧
  (|x| < 2 ∧ |y| > 3 → |x / y| < 2 / 3) :=
begin
  sorry
end

end correct_propositions_l617_617638


namespace find_p_probability_of_match_ending_after_4_games_l617_617233

variables (p : ℚ)

-- Conditions translated to Lean definitions
def probability_first_game_win : ℚ := 1 / 2

def probability_consecutive_games_win : ℚ := 5 / 16

-- Definitions based on conditions
def prob_second_game_win_if_won_first : ℚ := (1 + p) / 2

def prob_winning_consecutive_games (prob_first_game : ℚ) (prob_second_game_if_won_first : ℚ) : ℚ :=
prob_first_game * prob_second_game_if_won_first

-- Main Theorem Statements to be proved
theorem find_p 
    (h_eq : prob_winning_consecutive_games probability_first_game_win (prob_second_game_win_if_won_first p) = probability_consecutive_games_win) :
    p = 1 / 4 :=
sorry

-- Given p = 1/4, probabilities for each scenario the match ends after 4 games
def prob_scenario1 : ℚ := (1 / 2) * ((1 + 1/4) / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2)
def prob_scenario2 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2)
def prob_scenario3 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2) * ((1 + 1/4) / 2)

def total_probability_ending_in_4_games : ℚ :=
2 * (prob_scenario1 + prob_scenario2 + prob_scenario3)

theorem probability_of_match_ending_after_4_games (hp : p = 1 / 4) :
    total_probability_ending_in_4_games = 165 / 512 :=
sorry

end find_p_probability_of_match_ending_after_4_games_l617_617233


namespace find_xyz_l617_617887

-- Given conditions
variables {α : Type*} [Field α]
variables (a b c x y z : α)

-- Definitions based on conditions
def condition1 := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0
def condition2 := a = (b + c) / (x - 3)
def condition3 := b = (a + c) / (y - 3)
def condition4 := c = (a + b) / (z - 3)
def condition5 := x * y + x * z + y * z = 9
def condition6 := x + y + z = 6

-- Proof statement
theorem find_xyz :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ condition6 → x * y * z = 14 :=
by {
  sorry
}

end find_xyz_l617_617887


namespace norma_initial_cards_l617_617452

def initial_card_count (lost: ℕ) (left: ℕ) : ℕ :=
  lost + left

theorem norma_initial_cards : initial_card_count 70 18 = 88 :=
  by
    -- skipping proof
    sorry

end norma_initial_cards_l617_617452


namespace proof_max_magnitude_proof_min_magnitude_proof_value_f_x0_l617_617768

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def vec_b : ℝ × ℝ := (Real.sqrt 3, -1)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def max_magnitude (x : ℝ) : ℝ :=
  magnitude (2 * vec_a x - vec_b)

noncomputable def min_magnitude (x : ℝ) : ℝ :=
  magnitude (2 * vec_a x - vec_b)

noncomputable def f (x : ℝ) : ℝ :=
  (vec_a x).1 * ((vec_a x).1 + vec_b.1) + (vec_a x).2 * ((vec_a x).2 + vec_b.2)

theorem proof_max_magnitude : ∀ x : ℝ, x ∈ ℝ → max_magnitude x ≤ 4 :=
sorry

theorem proof_min_magnitude : ∀ x : ℝ, x ∈ ℝ → min_magnitude x ≥ 0 :=
sorry

theorem proof_value_f_x0 : ∀ x₀ : ℝ, x₀ ∈ ℝ → (f x₀ = 3 ∨ f x₀ = -1) :=
sorry

end proof_max_magnitude_proof_min_magnitude_proof_value_f_x0_l617_617768


namespace probability_of_two_digit_number_divisible_by_3_l617_617351

def digits : List ℕ := [1, 2, 3, 4]

def two_digit_numbers (digits : List ℕ) : List ℕ :=
  digits.bind (λ d1 => digits.filter (≠ d1).map (λ d2 => 10 * d1 + d2))

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def count_divisibles (l : List ℕ) : ℕ :=
  l.countp is_divisible_by_3

noncomputable def probability_divisible_by_3 (digits : List ℕ) : ℚ :=
  let nums := two_digit_numbers digits
  (count_divisibles nums) / (nums.length)

theorem probability_of_two_digit_number_divisible_by_3 :
  probability_divisible_by_3 digits = 1 / 3 :=
by
  sorry

end probability_of_two_digit_number_divisible_by_3_l617_617351


namespace external_angle_bisector_circumcircle_l617_617492

theorem external_angle_bisector_circumcircle (A B C D : Point)
  (h_triangle : ¬ Collinear {A, B, C})
  (h_circumcircle : OnCircumcircle C A B D)
  (h_angle_bisector : ExternalAngleBisector C D A B) :
  length A D = length B D :=
sorry

end external_angle_bisector_circumcircle_l617_617492


namespace total_oreos_l617_617037

theorem total_oreos (jordans_oreos : ℕ) (h : jordans_oreos = 11) : 
  let james_oreos := 2 * jordans_oreos + 3 in
  jordans_oreos + james_oreos = 36 :=
by
  sorry

end total_oreos_l617_617037


namespace boys_more_than_girls_l617_617151

-- Definitions of the conditions
def total_students : ℕ := 100
def boy_ratio : ℕ := 3
def girl_ratio : ℕ := 2

-- Statement of the problem
theorem boys_more_than_girls :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) - (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 :=
by
  sorry

end boys_more_than_girls_l617_617151


namespace complement_union_eq_zero_or_negative_l617_617882

def U : Set ℝ := Set.univ

def P : Set ℝ := { x | x > 1 }

def Q : Set ℝ := { x | x * (x - 2) < 0 }

theorem complement_union_eq_zero_or_negative :
  (U \ (P ∪ Q)) = { x | x ≤ 0 } := by
  sorry

end complement_union_eq_zero_or_negative_l617_617882


namespace coplanar_points_iff_l617_617293

def are_points_coplanar (b : ℝ) : Prop := 
  let vec1 := (1, b, 0)
  let vec2 := (0, 1, b)
  let vec3 := (b, 0, 1)
  matrix.det ![
    ![1, 0, b],
    ![b, 1, 0],
    ![0, b, 1]
  ] = 0

theorem coplanar_points_iff (b : ℝ) : are_points_coplanar b ↔ b = -1 := by
  sorry

end coplanar_points_iff_l617_617293


namespace uncool_students_in_two_classes_l617_617835

theorem uncool_students_in_two_classes
  (students_class1 : ℕ)
  (cool_dads_class1 : ℕ)
  (cool_moms_class1 : ℕ)
  (both_cool_class1 : ℕ)
  (students_class2 : ℕ)
  (cool_dads_class2 : ℕ)
  (cool_moms_class2 : ℕ)
  (both_cool_class2 : ℕ)
  (h1 : students_class1 = 45)
  (h2 : cool_dads_class1 = 22)
  (h3 : cool_moms_class1 = 25)
  (h4 : both_cool_class1 = 11)
  (h5 : students_class2 = 35)
  (h6 : cool_dads_class2 = 15)
  (h7 : cool_moms_class2 = 18)
  (h8 : both_cool_class2 = 7) :
  (students_class1 - ((cool_dads_class1 - both_cool_class1) + (cool_moms_class1 - both_cool_class1) + both_cool_class1) +
   students_class2 - ((cool_dads_class2 - both_cool_class2) + (cool_moms_class2 - both_cool_class2) + both_cool_class2)
  ) = 18 :=
sorry

end uncool_students_in_two_classes_l617_617835


namespace sum_of_conjugates_l617_617649

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617649


namespace gamma_start_time_correct_l617_617253

noncomputable def trisection_points (AB : ℕ) : Prop := AB ≥ 3

structure Walkers :=
  (d : ℕ) -- Total distance AB
  (Vα : ℕ) -- Speed of person α
  (Vβ : ℕ) -- Speed of person β
  (Vγ : ℕ) -- Speed of person γ

def meeting_times (w : Walkers) := 
  w.Vα = w.d / 72 ∧ 
  w.Vβ = w.d / 36 ∧ 
  w.Vγ = w.Vβ

def start_times_correct (startA timeA_meetC : ℕ) (startB timeB_reachesA: ℕ) (startC_latest: ℕ): Prop :=
  startA = 0 ∧ 
  startB = 12 ∧
  timeA_meetC = 24 ∧ 
  timeB_reachesA = 30 ∧
  startC_latest = 16

theorem gamma_start_time_correct (AB : ℕ) (w : Walkers) (t : Walkers → Prop) : 
  trisection_points AB → 
  meeting_times w →
  start_times_correct 0 24 12 30 16 → 
  ∃ tγ_start, tγ_start = 16 :=
sorry

end gamma_start_time_correct_l617_617253


namespace average_molecular_weight_benzoic_acid_l617_617636

def atomic_mass_C : ℝ := (12 * 0.9893) + (13 * 0.0107)
def atomic_mass_H : ℝ := (1 * 0.99985) + (2 * 0.00015)
def atomic_mass_O : ℝ := (16 * 0.99762) + (17 * 0.00038) + (18 * 0.00200)

theorem average_molecular_weight_benzoic_acid :
  (7 * atomic_mass_C) + (6 * atomic_mass_H) + (2 * atomic_mass_O) = 123.05826 :=
by {
  sorry
}

end average_molecular_weight_benzoic_acid_l617_617636


namespace partial_deriv_z_partial_deriv_u_l617_617732

noncomputable def z (x y : ℝ) : ℝ := x^3 - 2 * x^2 * y + 3 * y^2
noncomputable def u (x y t : ℝ) : ℝ := Real.exp (x * y * t)

theorem partial_deriv_z (x y : ℝ) :
  (∂^2 / ∂x^2, z) x y = 6 * x - 4 * y ∧
  (∂^2 / ∂y^2, z) x y = 6 ∧
  (∂^2 / ∂x ∂y, z) x y = (∂^2 / ∂y ∂x, z) x y ∧
  (∂^2 / ∂x ∂y, z) x y = -4 * x :=
sorry

theorem partial_deriv_u (x y t : ℝ) :
  (∂^2 / ∂x^2, u) x y t = y^2 * t^2 * u x y t ∧
  (∂^2 / ∂y^2, u) x y t = x^2 * t^2 * u x y t ∧
  (∂^2 / ∂t^2, u) x y t = x^2 * y^2 * u x y t ∧
  (∂^2 / ∂x ∂y, u) x y t = (∂^2 / ∂y ∂x, u) x y t ∧
  (∂^2 / ∂x ∂y, u) x y t = t * (y + x * y * t) * u x y t ∧
  (∂^2 / ∂x ∂t, u) x y t = (∂^2 / ∂t ∂x, u) x y t ∧
  (∂^2 / ∂x ∂t, u) x y t = y * (1 + x * y * t) * u x y t ∧
  (∂^2 / ∂y ∂t, u) x y t = (∂^2 / ∂t ∂y, u) x y t ∧
  (∂^2 / ∂y ∂t, u) x y t = x * (1 + x * y * t) * u x y t :=
sorry

end partial_deriv_z_partial_deriv_u_l617_617732


namespace janet_total_l617_617043

-- Definitions based on the conditions
variable (initial_collect : ℕ) (sold : ℕ) (better_cond : ℕ)
variable (twice_size : ℕ)

-- The conditions from part a)
def janet_initial_collection := initial_collect = 10
def janet_sells := sold = 6
def janet_gets_better := better_cond = 4
def brother_gives := twice_size = 2 * (initial_collect - sold + better_cond)

-- The proof statement based on part c)
theorem janet_total (initial_collect sold better_cond twice_size : ℕ) : 
    janet_initial_collection initial_collect →
    janet_sells sold →
    janet_gets_better better_cond →
    brother_gives initial_collect sold better_cond twice_size →
    (initial_collect - sold + better_cond + twice_size = 24) :=
by
  intros h1 h2 h3 h4
  sorry

end janet_total_l617_617043


namespace minimize_dwarf_risk_l617_617958

noncomputable def dwarf_hat_risk (p : ℕ) : ℕ := 
if p = 0 then 0 else 1

theorem minimize_dwarf_risk (p : ℕ) (hp: p < ∞) : 
  dwarf_hat_risk p = 1 := 
by
  sorry

end minimize_dwarf_risk_l617_617958


namespace unique_fraction_10_percent_increase_l617_617266

theorem unique_fraction_10_percent_increase :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ Int.gcd x y = 1 ∧ (y + 1) * (10 * x) = 11 * x * (y + 1) :=
by
  sorry

end unique_fraction_10_percent_increase_l617_617266


namespace sum_primes_under_50_l617_617302

def satisfies_condition (p : ℕ) (f : Fin p → Fin p) : Prop :=
  ∀ x : Fin p, p ∣ f (f x).val - x.val * x.val

def is_valid_prime (p : ℕ) : Prop :=
  (Nat.Prime p) ∧ (∃ f : Fin p → Fin p, satisfies_condition p f)

theorem sum_primes_under_50 : 
  ∑ p in Finset.filter is_valid_prime (Finset.range 50), p = 50 := 
sorry

end sum_primes_under_50_l617_617302


namespace range_of_a_l617_617782

-- Definitions of given conditions
def f (a : ℝ) (x : ℝ) :=
  if x ≤ 8 then (4 - a) * x - 5
  else a ^ (x - 8)

def a_seq (a : ℝ) (n : ℕ) : ℝ :=
  f a n

axiom increasing_sequence (a : ℝ) : ∀ (n m : ℕ), n < m → a_seq a n < a_seq a m

-- Problem statement
theorem range_of_a : {a : ℝ | ∀ (n m : ℕ), n < m → a_seq a n < a_seq a m} = {a : ℝ | 3 < a ∧ a < 4} :=
by
  sorry

end range_of_a_l617_617782


namespace count_real_roots_quadratic_eqns_l617_617261

def S : Set ℤ := {2, 4, 6, 8, 10, 12}

def has_real_roots (b c : ℤ) : Prop := b^2 - 4 * c ≥ 0

theorem count_real_roots_quadratic_eqns :
  (finset.univ.filter (λ b, b ∈ S)).sum (λ b, (finset.univ.filter (λ c, c ∈ S ∧ has_real_roots b c)).card) = 25 :=
sorry

end count_real_roots_quadratic_eqns_l617_617261


namespace coplanar_points_scalar_eq_l617_617880

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D O : V) (k : ℝ)

theorem coplanar_points_scalar_eq:
  (3 • (A - O) - 2 • (B - O) + 5 • (C - O) + k • (D - O) = (0 : V)) →
  k = -6 :=
by sorry

end coplanar_points_scalar_eq_l617_617880


namespace model_car_cost_l617_617907

theorem model_car_cost (x : ℕ) :
  (5 * x) + (5 * 10) + (5 * 2) = 160 → x = 20 :=
by
  intro h
  sorry

end model_car_cost_l617_617907


namespace polynomial_mult_of_6_l617_617894

theorem polynomial_mult_of_6 (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) : 6 ∣ P.eval 5 := 
sorry

end polynomial_mult_of_6_l617_617894


namespace fraction_halfway_between_fraction_halfway_between_l617_617177

theorem fraction_halfway_between : (3/4 : ℚ) < (5/7 : ℚ) :=
by linarith

theorem fraction_halfway_between : (41 / 56 : ℚ) = (1 / 2) * ((3 / 4) + (5 / 7)) :=
by sorry

end fraction_halfway_between_fraction_halfway_between_l617_617177


namespace slope_of_line_l617_617236

-- Define the points (x1, y1) and (x2, y2)
def point1 := (-12, -39 : ℚ × ℚ)
def point2 := (400, 0 : ℚ × ℚ)

-- Define the slope calculation
def slope (p1 p2 : ℚ × ℚ) : ℚ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- State the proposition
theorem slope_of_line : slope point1 point2 = 39 / 412 := by
  sorry

end slope_of_line_l617_617236


namespace fifteenth_odd_multiple_of_5_is_145_l617_617188

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l617_617188


namespace even_perfect_square_factors_count_l617_617807

theorem even_perfect_square_factors_count :
  let n := 2^6 * 7^10 in
  (∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ a ≥ 1 ∧ 0 ≤ b ∧ b ≤ 10 ∧ b % 2 = 0 → (2^a * 7^b | n)) ∧
  let possible_a := {2, 4, 6} in
  let possible_b := {0, 2, 4, 6, 8, 10} in
  ∃ (count : ℕ), count = possible_a.card * possible_b.card ∧ count = 18 :=
sorry

end even_perfect_square_factors_count_l617_617807


namespace fixed_point_coordinates_l617_617753

theorem fixed_point_coordinates (a b x y : ℝ) 
  (h1 : a + 2 * b = 1) 
  (h2 : (a * x + 3 * y + b) = 0) :
  x = 1 / 2 ∧ y = -1 / 6 := by
  sorry

end fixed_point_coordinates_l617_617753


namespace fifteenth_odd_multiple_of_5_l617_617196

theorem fifteenth_odd_multiple_of_5 :
  (∃ n: ℕ, n = 15 ∧ (10 * n - 5 = 145)) :=
begin
  use 15,
  split,
  { refl },
  { norm_num }
end

end fifteenth_odd_multiple_of_5_l617_617196


namespace sum_of_number_and_conjugate_l617_617672

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617672


namespace problem_lean_l617_617922

noncomputable def a_b_difference : ℝ :=
  let x := Real.sqrt Real.exp 1 in
  let f (y : ℝ) (x : ℝ) := y^2 + x^4 - 2*x^2*y - 1 in
  let a := if h : -1 ≤ (ε : ℝ) := Real.sqrt (Real.exp 2) - 1 else Real.sqrt (Real.exp 2) + 1 in
  let b := if h : -1 ≤ (ε : ℝ) := Real.sqrt (Real.exp 2) + 1 else Real.sqrt (Real.exp 2) - 1 in
  Real.abs (a - b)

theorem problem_lean : 
  let (x, a, b) := (Real.sqrt Real.exp 1, (Real.sqrt (Real.exp 2) + 1), (Real.sqrt (Real.exp 2) - 1)) in
  x = Real.sqrt Real.exp 1 ∧
  (f (a : ℝ) (x : ℝ) = 0 ∧ f (b : ℝ) (x : ℝ) = 0) →
  Real.abs (a - b) = 2 :=
by
  sorry

end problem_lean_l617_617922


namespace child_ticket_cost_l617_617161

theorem child_ticket_cost (num_tickets : ℕ) (total_revenue : ℕ) (num_adults : ℕ) (adult_ticket_cost : ℕ) : 
  num_tickets = 225 → total_revenue = 1875 → num_adults = 175 → adult_ticket_cost = 9 → 
  let num_children := num_tickets - num_adults in
  let child_tickets_revenue := total_revenue - num_adults * adult_ticket_cost in
  num_children > 0 → 
  child_tickets_revenue / num_children = 6 :=
by
  intros _ _ _ _ _ _ _
  sorry

end child_ticket_cost_l617_617161


namespace nth_term_50th_is_3755_l617_617126

-- Define the sequence in Lean
def power_of_5 (n : ℕ) : ℕ := 5 ^ n

def is_modified_sequence (a : ℕ) : Prop :=
  ∃ (indices : List ℕ), (∀ i ∈ indices, i < 6) ∧ a = indices.sum.map (λ n, power_of_5 n)

def nth_term (n : ℕ) : ℕ :=
  let binary_rep := Nat.toDigits 2 n in
  let powers_of_5 := binary_rep.reverse.zipWith (λ b e, if b = 1 then power_of_5 e else 0) (List.range binary_rep.length) in
  powers_of_5.sum

theorem nth_term_50th_is_3755 : nth_term 50 = 3755 :=
by
  sorry

end nth_term_50th_is_3755_l617_617126


namespace negation_of_existence_l617_617510

theorem negation_of_existence :
  ¬ (∃ (x_0 : ℝ), x_0^2 - x_0 + 1 ≤ 0) ↔ ∀ (x : ℝ), x^2 - x + 1 > 0 :=
by
  sorry

end negation_of_existence_l617_617510


namespace quadratic_roots_r_l617_617064

theorem quadratic_roots_r (a b m p r : ℚ) :
  (∀ x : ℚ, x^2 - m * x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x : ℚ, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a + 1)) →
  r = 19 / 3 :=
by
  sorry

end quadratic_roots_r_l617_617064


namespace bullet_train_pass_time_l617_617230

noncomputable def time_to_pass (length_train : ℕ) (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) : ℝ := 
  let relative_speed_kmph := speed_train_kmph + speed_man_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * 1000 / 3600
  length_train / relative_speed_mps

def length_train := 350
def speed_train_kmph := 75
def speed_man_kmph := 12

theorem bullet_train_pass_time : 
  abs (time_to_pass length_train speed_train_kmph speed_man_kmph - 14.47) < 0.01 :=
by
  sorry

end bullet_train_pass_time_l617_617230


namespace train_crosses_pole_time_l617_617611

theorem train_crosses_pole_time
  (l : ℕ) (v_kmh : ℕ) (v_ms : ℚ) (t : ℕ)
  (h_l : l = 100)
  (h_v_kmh : v_kmh = 180)
  (h_v_ms_conversion : v_ms = v_kmh * 1000 / 3600)
  (h_v_ms : v_ms = 50) :
  t = l / v_ms := by
  sorry

end train_crosses_pole_time_l617_617611


namespace halfway_fraction_l617_617181

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l617_617181


namespace problem_statement_l617_617572

theorem problem_statement : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end problem_statement_l617_617572


namespace angle_BAD_is_correct_l617_617468

noncomputable def angle_BAD (A B C D : Point) : Real :=
  if h : valid_quadrilateral A B C D ∧ AB = BC ∧ BC = CD ∧ ∠ABC = 100 ∧ ∠BCD = 160 then 65 else 0

theorem angle_BAD_is_correct (A B C D : Point) : 
  valid_quadrilateral A B C D → AB = BC → BC = CD → ∠ABC = 100 → ∠BCD = 160 → angle_BAD A B C D = 65 :=
by {
   intros,
   rw angle_BAD,
   split_ifs,
   sorry
}

end angle_BAD_is_correct_l617_617468


namespace triangle_problem_one_triangle_problem_two_l617_617395

noncomputable def value_of_a_b_sinA_sinB (a b A B : ℝ) (h₁ : a / sin A = b / sin B) (h₂ : c = 2) (h₃ : C = 60) : ℝ := sorry

theorem triangle_problem_one (a b : ℝ) (A B : ℝ) 
(h₁ : a / sin A = b / sin B)
(h₂ : c = 2) 
(h₃ : C = 60)
: (a + b) / (sin A + sin B) = 4 * sqrt 3 / 3 := sorry

theorem triangle_problem_two (a b : ℝ)
(A B : ℝ) 
(h₁ : a / sin A = b / sin B) 
(h₂ : c = 2) 
(h₃ : C = 60) 
(h₄ : a + b = a * b) 
: S := 
have h₅ : c^2 = a^2 + b^2 - 2*a*b*cos(C), by sorry,
have h₆ : 4 = (a + b)^2 - 3*a*b, by sorry,
have area_triangle_eq : S = 1/2*a*b*sin C := sorry,
sqrt 3

end triangle_problem_one_triangle_problem_two_l617_617395


namespace sum_of_radical_conjugates_l617_617687

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617687


namespace exists_new_arc_within_old_arc_theorem_l617_617224

noncomputable def exists_new_arc_within_old_arc (n : ℕ) (k : ℕ) : Prop :=
∃ (points : Fin n → ℝ) (rotated_points : Fin n → ℝ),
(let angles := λ i, i * 2 * π / n in
 let new_angles := λ i, (i * 2 * π / n + 2 * π * k / n) % (2 * π) in
    ∃ (i j : Fin n), (angles i < angles (i + 1)) ∧ (new_angles i < new_angles (i + 1)) ∧
    new_angles i ∈ set.Icc (angles i) (angles (i + 1)) ∧
    new_angles (i + 1) ∈ set.Icc (angles i) (angles (i + 1)))

theorem exists_new_arc_within_old_arc_theorem
  (n k : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ k): exists_new_arc_within_old_arc n k :=
begin
  sorry -- Proof to be provided
end

end exists_new_arc_within_old_arc_theorem_l617_617224


namespace proof_problem_l617_617412

variable (k : ℝ)
def point_A : ℝ × ℝ := (10, 3)
def point_B : ℝ × ℝ := (-8, -6)
def point_C : ℝ × ℝ := (4, k)

-- Slope calculation for two given points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Hypothesis: slopes are equal
def slopes_equal : Prop :=
  slope point_A point_B = slope point_B point_C

-- Equation of the line in standard form
def line_equation (x y : ℝ) : Prop :=
  x - 2 * y = -4

-- x-intercept calculation
def x_intercept (y : ℝ) : ℝ :=
  (2 * y) + 4

theorem proof_problem
  (hyp : slopes_equal)
  (k_eq_zero : k = 0) :
  line_equation (10 : ℝ) 3 ∧ x_intercept 0 = 4 :=
by
  sorry

end proof_problem_l617_617412


namespace measure_angle_ACB_l617_617406

def angle_ABC_supp {α β : ℝ} : α + β = 180 → α = 35 := sorry

def triangle_angles_sum {α β γ : ℝ} (h1 : α = 75) (h2 : β = 35) : α + β + γ = 180 → γ = 70 :=
  by
    intro h3
    have h4 : 180 - (α + β) = γ := by 
      sorry -- Step showing γ = 180 - (α + β)
    sorry -- Combine all steps showing α + β + γ = 180

theorem measure_angle_ACB : ∃ γ : ℝ, γ = 70 :=
  let γ : ℝ := 70
  have h1 : 180 - 145 = 35 := by sorry -- angle ABD supplementary condition
  have h2 : triangle_angles_sum 75 35 = 70 := by sorry -- use triangle angle conditions
  use γ
  show γ = 70 from by
    rw [←h2]
    sorry

end measure_angle_ACB_l617_617406


namespace sequence_arithmetic_general_formula_for_a_n_value_of_b_n_l617_617330

noncomputable def S (n : ℕ) : ℕ := sorry

theorem sequence_arithmetic (n : ℕ) (h : n > 0) :
  ((S (n + 1) / (n + 1)) - (S n / n) = 2) :=
sorry

theorem general_formula_for_a_n (n : ℕ) : 
  let a_n := 4 * n - 2 
  ((*[defining a_n]*)) :=
sorry

theorem value_of_b_n (n : ℕ) (b : ℕ → ℕ) : 
  b n = ∑ k in (Finset.range (n + 1)), a (2 ^ k) - 2 * n - 8 :=
sorry

end sequence_arithmetic_general_formula_for_a_n_value_of_b_n_l617_617330


namespace find_a_l617_617336

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def are_perpendicular (k1 k2 : ℝ) : Prop :=
  k1 * k2 = -1

theorem find_a (a : ℝ) :
  let A := (0 : ℝ, -1 : ℝ)
  let B := (-2 * a, 0 : ℝ)
  let C := (1 : ℝ, 1 : ℝ)
  let D := (2 : ℝ, 4 : ℝ)
  let k_CD := slope C D
  let k_AB := slope A B
  are_perpendicular k_CD k_AB → a = 3 / 2 :=
by
  introv h
  sorry

end find_a_l617_617336


namespace num_cows_correct_l617_617830

-- Definitions from the problem's conditions
def total_animals : ℕ := 500
def percentage_chickens : ℤ := 10
def remaining_animals := total_animals - (percentage_chickens * total_animals / 100)
def goats (cows: ℕ) : ℕ := 2 * cows

-- Statement to prove
theorem num_cows_correct : ∃ cows, remaining_animals = cows + goats cows ∧ 3 * cows = 450 :=
by
  sorry

end num_cows_correct_l617_617830


namespace max_possible_M_l617_617070

open Nat

noncomputable def f : ℕ → ℕ := sorry

def iter_f (k : ℕ) (x : ℕ) : ℕ :=
  Nat.recOn k f (λ n h, f (h))

theorem max_possible_M :
  ∃ M : ℕ, M = 8 ∧
    (∀ m : ℕ, m < M → 
      (∀ i : ℕ, 1 ≤ i ∧ i ≤ 16 → 
        (iter_f m (i + 1) - iter_f m i) % 17 ≠ 1 ∧ 
        (iter_f m (i + 1) - iter_f m i) % 17 ≠ -1) ∧
      (iter_f m 1 - iter_f m 17) % 17 ≠ 1 ∧ 
      (iter_f m 1 - iter_f m 17) % 17 ≠ -1) ∧
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 16 → 
      (iter_f M (i + 1) - iter_f M i) % 17 = 1 ∨ 
      (iter_f M (i + 1) - iter_f M i) % 17 = -1) ∧
    (iter_f M 1 - iter_f M 17) % 17 = 1 ∨ 
    (iter_f M 1 - iter_f M 17) % 17 = -1 :=
begin
  -- The proof would go here
  sorry
end

end max_possible_M_l617_617070


namespace blue_paint_amount_l617_617313

variable (red_paint blue_paint yellow_paint : ℕ)

-- Conditions
axiom ratio_condition : red_paint : blue_paint : yellow_paint = 5 : 3 : 7
axiom yellow_paint_condition : yellow_paint = 21

-- Theorem to be proven
theorem blue_paint_amount (red_paint blue_paint yellow_paint : ℕ) 
  (h1 : red_paint : blue_paint : yellow_paint = 5 : 3 : 7)
  (h2 : yellow_paint = 21) : blue_paint = 9 :=
sorry

end blue_paint_amount_l617_617313


namespace cartesian_eq_line_l617_617027

-- Definitions corresponding to conditions
def parametric_eq_curve (θ : ℝ) : ℝ × ℝ := 
  (sqrt 2 * cos θ + 2, sqrt 2 * sin θ)

def line_l_polar (α : ℝ) : ℝ → ℝ × ℝ := 
  λρ, (ρ * cos α, ρ * sin α)

-- Lean 4 statement
theorem cartesian_eq_line (α : ℝ) (hα : α ∈ set.Icc 0 (2*π)) :
  (ExistIntersect α) ∧
  (| OA α | + | OB α | = 3) →
  ∃ m : ℝ, m = (sqrt 7 / 3) ∨ m = -(sqrt 7 / 3) ∧ 
  ∀ x y, y = m * x :=
sorry

end cartesian_eq_line_l617_617027


namespace sum_x_not_121_sum_x_111_l617_617547

def is_valid_x (x : ℝ) : Prop :=
  x = Real.sqrt 2 + 1 ∨ x = Real.sqrt 2 - 1

def S (xs : Fin 150 → ℝ) : ℝ :=
  let pairs := List.ofFn (fun i => xs ⟨2*i, by linarith⟩ * xs ⟨2*i+1, by linarith⟩)
  pairs.sum

theorem sum_x_not_121 (xs : Fin 150 → ℝ) (h : ∀ i, is_valid_x (xs i)) : S xs ≠ 121 :=
sorry

theorem sum_x_111 (xs : Fin 150 → ℝ) (h : ∀ i, is_valid_x (xs i)) : ∃ xs, S xs = 111 :=
sorry

end sum_x_not_121_sum_x_111_l617_617547


namespace side_length_of_triangle_l617_617062

noncomputable theory
open_locale classical

variable (A B C M N G : Type)
variables (AM BN AC : ℝ)

def centroid_ratios (hG : G ∈ {g : Type | g ∈ [A, B, C]}) : Prop :=
  ∀ (AG : ℝ) (GM : ℝ) (BG : ℝ) (GN : ℝ),
  AG / GM = 2 ∧ BG / GN = 2

def perpendicular_medians (AM BN : ℝ) (M N : Type) : Prop := 
  ∀ (m n : ℝ), 
  AM = m ∧ BN = n → ⟦m * m + n * n = 0⟧

theorem side_length_of_triangle
  (h1 : centroid_ratios A B C M N G)
  (h2 : perpendicular_medians AM BN M N)
  (AM_eq : AM = 15)
  (BN_eq : BN = 20) :
  AC = (20 * real.sqrt 13) / 3 := 
sorry

end side_length_of_triangle_l617_617062


namespace shape_not_in_square_l617_617496

/-- Let A, B, C, D, and E be the shapes given. Let S be the square divided into eight pieces. Prove that shape E is not one of the pieces in S. -/
theorem shape_not_in_square (S : set (set ?)) (A B C D E : set ?) : 
  (A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ E ∉ S) := by
  sorry

end shape_not_in_square_l617_617496


namespace points_with_condition_in_rectangle_l617_617920

-- Define the problem formally within Lean
theorem points_with_condition_in_rectangle :
  ∀ (points : Fin 6 → (ℝ × ℝ)), 
    (∀ i, (0 ≤ (points i).fst ∧ (points i).fst ≤ 4) ∧ (0 ≤ (points i).snd ∧ (points i).snd ≤ 3)) → 
    ∃ i j, i ≠ j ∧ (dist (points i) (points j) ≤ √5) :=
by
  sorry

end points_with_condition_in_rectangle_l617_617920


namespace twelve_people_pairing_l617_617530

noncomputable def num_ways_to_pair : ℕ := sorry

theorem twelve_people_pairing :
  (∀ (n : ℕ), n = 12 → (∃ f : ℕ → ℕ, ∀ i, f i = 2 ∨ f i = 12 ∨ f i = 7) → num_ways_to_pair = 3) := 
sorry

end twelve_people_pairing_l617_617530


namespace max_T_size_l617_617432

-- Definitions
def S := {a : Fin 7 → ℕ // ∀ i, a i = 0 ∨ a i = 1}
def dist (a b : S) : ℕ := Finset.sum (Finset.univ : Finset (Fin 7)) (λ i, |a.val i - b.val i|)
def T (t : Finset S) : Prop := ∀ a b ∈ t, a ≠ b → dist a b ≥ 3

-- Theorem statement
theorem max_T_size : ∃ (t : Finset S), T t ∧ t.card = 16 :=
sorry

end max_T_size_l617_617432


namespace total_cost_of_projectors_and_computers_l617_617519

theorem total_cost_of_projectors_and_computers :
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  (n_p * c_p + n_c * c_c) = 175200 := by
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  sorry 

end total_cost_of_projectors_and_computers_l617_617519


namespace ratio_x_y_z_w_l617_617701

theorem ratio_x_y_z_w (x y z w : ℝ) 
(h1 : 0.10 * x = 0.20 * y)
(h2 : 0.30 * y = 0.40 * z)
(h3 : 0.50 * z = 0.60 * w) : 
  (x / w) = 8 
  ∧ (y / w) = 4 
  ∧ (z / w) = 3
  ∧ (w / w) = 2.5 := 
sorry

end ratio_x_y_z_w_l617_617701


namespace range_of_c_over_a_l617_617337

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) : -2 < c / a ∧ c / a < -1 :=
by {
  sorry
}

end range_of_c_over_a_l617_617337


namespace graph_paper_squares_below_line_l617_617499

theorem graph_paper_squares_below_line
  (h : ∀ (x y : ℕ), 12 * x + 247 * y = 2976)
  (square_size : ℕ) 
  (xs : ℕ) (ys : ℕ)
  (line_eq : ∀ (x y : ℕ), y = 247 * x / 12)
  (n_squares : ℕ) :
  n_squares = 1358
  := by
    sorry

end graph_paper_squares_below_line_l617_617499


namespace gravitational_force_at_300000_l617_617964

-- Define gravitational force inversely proportional to the square of the distance
def gravitational_force (f : ℝ) (d : ℝ) : ℝ := f * d^2

-- Define the constant k
def k := gravitational_force 400 5000

-- Prove that the gravitational force is 1/9 Newtons at 300000 miles distance
theorem gravitational_force_at_300000 : 
  (gravitational_force x 300000 = k) → (x = 1 / 9) :=
by 
  -- Assuming definition and given conditions in Lean
  assume h : gravitational_force x 300000 = k,
  sorry

end gravitational_force_at_300000_l617_617964


namespace quadrilateral_with_isosceles_diagonals_not_necessarily_rhombus_l617_617282

theorem quadrilateral_with_isosceles_diagonals_not_necessarily_rhombus (Q : Type) [quadrilateral Q] :
  (∀ d1 d2 : diagonal Q, is_isosceles_triangle Q (d1) ∧ is_isosceles_triangle Q (d2)) → ¬is_rhombus Q := by
  sorry

end quadrilateral_with_isosceles_diagonals_not_necessarily_rhombus_l617_617282


namespace water_depth_when_upright_l617_617243

theorem water_depth_when_upright (height : ℝ) (diameter : ℝ) (depth_in_horizontal : ℝ) (radius : ℝ)  (volume_of_cylinder : ℝ):
  height = 20 ∧ diameter = 6 ∧ depth_in_horizontal = 4 ∧ radius = diameter / 2 ∧ volume_of_cylinder = π * radius^2 * height → height = 20 :=
by
  intro conditions
  cases conditions with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 v
  sorry

end water_depth_when_upright_l617_617243


namespace debate_team_boys_l617_617139

theorem debate_team_boys (total_groups : ℕ) (members_per_group : ℕ) (num_girls : ℕ) (total_members : ℕ) :
  total_groups = 8 →
  members_per_group = 4 →
  num_girls = 4 →
  total_members = total_groups * members_per_group →
  total_members - num_girls = 28 :=
by
  sorry

end debate_team_boys_l617_617139


namespace moles_of_CaCl2_l617_617298

/-- 
We are given the reaction: CaCO3 + 2 HCl → CaCl2 + CO2 + H2O 
with 2 moles of HCl and 1 mole of CaCO3. We need to prove that the number 
of moles of CaCl2 formed is 1.
-/
theorem moles_of_CaCl2 (HCl: ℝ) (CaCO3: ℝ) (reaction: CaCO3 + 2 * HCl = 1): CaCO3 = 1 → HCl = 2 → CaCl2 = 1 :=
by
  -- importing the required context for chemical equations and stoichiometry
  sorry

end moles_of_CaCl2_l617_617298


namespace find_f_zero_f_is_odd_l617_617956

variable (f : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f(x + y) = f x + f y)

theorem find_f_zero : f 0 = 0 :=
by 
  sorry

theorem f_is_odd : ∀ x : ℝ, f(-x) = -f x :=
by
  sorry

end find_f_zero_f_is_odd_l617_617956


namespace scientific_notation_of_population_l617_617133

theorem scientific_notation_of_population (population : Real) (h_pop : population = 6.8e6) :
    ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ (population = a * 10^n) ∧ (a = 6.8) ∧ (n = 6) :=
by
  sorry

end scientific_notation_of_population_l617_617133


namespace problem_bound_l617_617924

theorem problem_bound (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * (x * y * z) ∧ 
  y * z + z * x + x * y - 2 * (x * y * z) ≤ 7 / 27 :=
sorry

end problem_bound_l617_617924


namespace sam_must_exchange_bill_l617_617101

-- Define the conditions of the problem
def toy_prices := [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75]
def favorite_toy_price := 2.25
def sam_quarters := 10
def sam_quarter_value := 0.25
def sam_quarters_value := sam_quarters * sam_quarter_value
def total_toys := 10

-- Permutations of 10 toys
noncomputable def total_permutations := Nat.factorial total_toys

-- Favorable permutations where the favorite toy appears in the first five toys
noncomputable def favorable_permutations :=
  Nat.factorial 9 + Nat.factorial 8 + Nat.factorial 7 + Nat.factorial 6 + Nat.factorial 5

-- Probability of favorable event
noncomputable def favorable_probability := favorable_permutations.to_double / total_permutations.to_double

-- Probability that Sam must exchange his bill
noncomputable def problem_probability := 1 - favorable_probability

-- The statement we are proving
theorem sam_must_exchange_bill : problem_probability = 8 / 9 :=
by sorry

end sam_must_exchange_bill_l617_617101


namespace mod_intercepts_sum_eq_fifteen_l617_617090

theorem mod_intercepts_sum_eq_fifteen :
  ∃ x0 y0 : ℕ, 0 ≤ x0 ∧ x0 < 21 ∧ 0 ≤ y0 ∧ y0 < 21 ∧ (5 * x0 ≡ -2 [MOD 21]) ∧ (3 * y0 ≡ 2 [MOD 21]) ∧ x0 + y0 = 15 := 
by
  -- Proof here, but we omit it with sorry
  sorry

end mod_intercepts_sum_eq_fifteen_l617_617090


namespace sum_of_radical_conjugates_l617_617683

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617683


namespace g_at_6_l617_617122

open Real

-- Definition of the function g with the given conditions
def g : ℝ → ℝ
assume x y, (hx : x ≠ y) => g (x + y) = g x + g y

theorem g_at_6 :
  (∀ x y : ℝ, g (x + y) = g x + g y) →
  g 5 = 6 →
  g 6 = 36 / 5 :=
by
  sorry

end g_at_6_l617_617122


namespace roots_condition_l617_617129

theorem roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 3 ∧ x2 < 3 ∧ x1^2 - m * x1 + 2 * m = 0 ∧ x2^2 - m * x2 + 2 * m = 0) ↔ m > 9 :=
by sorry

end roots_condition_l617_617129


namespace n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l617_617003

theorem n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd :
  ∀ (n m : ℤ), (n^2 + m^3) % 2 ≠ 0 → (n + m) % 2 = 1 :=
by sorry

end n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l617_617003


namespace halfway_fraction_l617_617184

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l617_617184


namespace func3_is_F_function_func4_is_F_function_l617_617759

def F_function (f : ℝ → ℝ) : Prop :=
  ∃ m > 0, ∀ x : ℝ, |f x| ≤ m * |x|

def func1 : ℝ → ℝ := λ x, x^2
def func2 : ℝ → ℝ := λ x, sin x + cos x
def func3 : ℝ → ℝ := λ x, x / (x^2 + x + 1)
def func4 : ℝ → ℝ := λ x, sorry -- Definition needs specific odd function satisfying given condition

theorem func3_is_F_function : F_function func3 :=
sorry

theorem func4_is_F_function (f : ℝ → ℝ) (h1 : ∀ x1 x2 : ℝ, |f x1 - f x2| ≤ 2 * |x1 - x2|) (h2 : ∀ x : ℝ, f (-x) = -f x) : F_function f :=
sorry

end func3_is_F_function_func4_is_F_function_l617_617759


namespace minimal_connected_components_GXY_l617_617054

noncomputable theory
open_locale classical

-- Definitions and Assumptions
variables {G : Type} [graph G] (X Y : set (vertex G))
variables (m n : ℕ)

-- Assuming G is a connected graph
variable (connected_G : connected G)

-- Assuming X and Y are disjoint
variable (disjoint_X_Y : disjoint X Y)

-- Assuming there are no edges between X and Y
variable (no_edges_X_Y : ∀ x ∈ X, ∀ y ∈ Y, ¬ (x.adj y))

-- Assuming G/X has m connected components
variable (connected_components_GX : G / X = m)

-- Assuming G/Y has n connected components
variable (connected_components_GY : G / Y = n)

-- Statement of the minimal number of connected components
theorem minimal_connected_components_GXY : 
  num_connected_components (G / (X ∪ Y)) = max m n := sorry

end minimal_connected_components_GXY_l617_617054


namespace solve_equation_l617_617937

theorem solve_equation {x : ℂ} : (x - 2)^4 + (x - 6)^4 = 272 →
  x = 6 ∨ x = 2 ∨ x = 4 + 2 * Complex.I ∨ x = 4 - 2 * Complex.I :=
by
  intro h
  sorry

end solve_equation_l617_617937


namespace length_AP_l617_617405

theorem length_AP 
  (side_ABCD : ℝ) (side_WXYZ_ZY : ℝ) (side_WXYZ_XY : ℝ) 
  (perpendicular_AD_WX : true)
  (shaded_area_eq_half_WXYZ : true) 
  (AP: ℝ) (PD: ℝ) :
  side_ABCD = 6 → side_WXYZ_ZY = 10 → side_WXYZ_XY = 6 → 
  ∃ AP, AP = 1 :=
by 
  intros h1 h2 h3
  use 1
  sorry

end length_AP_l617_617405


namespace dwarf_hats_minimum_risk_l617_617960

theorem dwarf_hats_minimum_risk (p : ℕ) (h : p < ∞) :
  ∃ strategy : (list (fin 2)) → (list (fin 2) × list (fin 2)),
  (∀ hats : list (fin 2), hats.length = p →
   (let (first_dwarf_risk, remaining_dwarves) := strategy hats in
    ∃ at_risk : fin p → Prop,
      (at_risk 0 = (first_dwarf_risk ≠ hats.head)) ∧
      (∀ i : fin (p - 1), ¬ at_risk (i + 1)))) ∧
  ∃ (min_risk_dwarves : nat), min_risk_dwarves = 1
:= by
sorry

end dwarf_hats_minimum_risk_l617_617960


namespace max_elements_in_A_l617_617892

noncomputable def M := finset.range 1996
def A (A_subset: finset ℕ := ∅) : Prop :=
  A_subset ⊆ M ∧ ∀ x ∈ A_subset, 15 * x ∉ A_subset

theorem max_elements_in_A : ∃ (A_subset: finset ℕ), A A_subset ∧ A_subset.card = 1870 :=
sorry

end max_elements_in_A_l617_617892


namespace perimeter_of_ABCD_l617_617852

def Point := ℝ × ℝ

def square_dist (p1 p2 : Point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt (square_dist p1 p2)

theorem perimeter_of_ABCD :
  ∀ (A B C D : Point), 
  distance (Point.mk 0 0) (Point.mk 4 0) < distance (Point.mk 0 0) (Point.mk 7 3) →
  distance D C = 5 →
  (Point.mk 0 4).2 = 4 →
  (Point.mk 7 0).1 = 7 →
  distance A B + distance B C + distance C D + distance D A = 26 := by
  intros A B C D AD_lt_BC DC_eq_5 DN_eq_4 BN_eq_7
  sorry

end perimeter_of_ABCD_l617_617852


namespace customer_paid_percentage_l617_617605

-- Definition of base retail price
def base_retail_price : ℝ := 1  -- Assume base retail price is normalized to 1

-- Condition: base retail price was lowered by 5%
def lowered_retail_price (base_price : ℝ) : ℝ := base_price * 0.95

-- Condition: customer saved 14.5% off the retail price
def customer_saving_percentage : ℝ := 0.145

-- Proof: the customer paid 85.5% of the retail price
theorem customer_paid_percentage (base_price : ℝ) (a : lowered_retail_price base_price = base_price * 0.95) : 
  1 - customer_saving_percentage = 0.855 :=
by
  -- statement to match the problem condition directly
  exact rfl

end customer_paid_percentage_l617_617605


namespace fixed_point_line_l617_617741

-- Define the conditions as provided in the problem
variable (m x y : ℝ)

-- The line's equation for any real number m
def line_eq := (m - 1) * x + (2 * m - 1) * y = m - 5

-- The fixed point to be proven is (9, -4)
theorem fixed_point_line : ∀ (m : ℝ), ∃ (x y : ℝ), line_eq m x y ∧ x = 9 ∧ y = -4 :=
by
  intro m
  use [9, -4]
  split
  -- Conditions to show
  sorry
  simp
  sorry

end fixed_point_line_l617_617741


namespace arithmetic_and_geometric_sequences_l617_617762

/-
Given an arithmetic sequence with common difference d > 0,
the sum of its first n terms is Sn.
-/
variables {a : ℕ → ℝ} (d : ℝ) (S_n : ℕ → ℝ)
(h_d_pos : d > 0)
(a_arith_seq : ∀ n, a (n + 1) = a n + d)

/-
Condition: a_2 + a_4 = 8
-/
def a_2 := a 2
def a_4 := a 4
def a_3 := a 3
def a_5 := a 5
def a_8 := a 8
def a_1 := a 1

def cond1 := a_2 + a_4 = 8
def cond2 := (a_3, a_5, a_8).geometric_seq
def result1 := ∀ n, a n = n + 1

/-
Condition: For bn = 1 / (a_n * a_{n+1}), find T_n, the sum of the first n terms of {b_n}, which is n / (2n + 4)
-/
def b (n : ℕ) := 1 / (a n * a (n + 1))

def T (n : ℕ) := ∑ i in finset.range n, b i

def result2 := ∀ n, T n = n / (2 * n + 4)

theorem arithmetic_and_geometric_sequences
  (h1 : cond1) (h2 : cond2) : result1 ∧ result2 := 
by { sorry }

end arithmetic_and_geometric_sequences_l617_617762


namespace real_part_of_complex_number_l617_617979

def complex_number : ℂ := (1 + Complex.i) * (1 + Complex.i)

theorem real_part_of_complex_number : Complex.re complex_number = 0 := sorry

end real_part_of_complex_number_l617_617979


namespace number_of_elements_A_inter_Z_l617_617803

open Set Int

-- Define the set A as per the given condition.
def A : Set ℝ := {x | x^2 < 3 * x + 4}

-- Define the target intersection set.
def A_int : Set ℤ := {n : ℤ | (n : ℝ) ∈ A}

-- Define the target number of elements in A ∩ ℤ.
theorem number_of_elements_A_inter_Z : (A_int.toFinset.card = 4) :=
by
  -- Proof placeholder
  sorry

end number_of_elements_A_inter_Z_l617_617803


namespace C_sum_bound_l617_617410

-- Definitions for the sequences and constants
def a (n : ℕ) : ℝ := (3 / 2) * (1 / 2)^n  -- Since a_1 = 3/2 and S_3 = 9/2, we deduce the common ratio is 1/2
def S (n : ℕ) : ℝ := (3 / 2) * ((1 - (1 / 2)^n) / (1 - 1 / 2))
def b (n : ℕ) : ℝ := real.log 2 (6 / a (2 * n + 1))

def is_arithmetic (seq : ℕ → ℝ) : Prop := ∀ n, seq (n + 1) - seq n = seq 1 - seq 0
def C (n : ℕ) : ℝ := 1 / (b n * b (n + 1))

-- The final theorem we want to prove
theorem C_sum_bound (n : ℕ) (h_arith : is_arithmetic b) : ∑ k in finset.range (n+1), C k < 1 / 4 :=
sorry

end C_sum_bound_l617_617410


namespace sum_of_number_and_conjugate_l617_617665

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617665


namespace english_students_23_l617_617837

def survey_students_total : Nat := 35
def students_in_all_three : Nat := 2
def solely_english_three_times_than_french (x y : Nat) : Prop := y = 3 * x
def english_but_not_french_or_spanish (x y : Nat) : Prop := y + students_in_all_three = 35 ∧ y - students_in_all_three = 23

theorem english_students_23 :
  ∃ (x y : Nat), solely_english_three_times_than_french x y ∧ english_but_not_french_or_spanish x y :=
by
  sorry

end english_students_23_l617_617837


namespace vet_donation_portion_is_correct_l617_617249

def fee_per_dog : ℕ := 15
def fee_per_cat : ℕ := 13
def num_dogs : ℕ := 8
def num_cats : ℕ := 3
def donation : ℕ := 53

noncomputable def portion_donated : ℚ :=
  donation / (num_dogs * fee_per_dog + num_cats * fee_per_cat) * 100

theorem vet_donation_portion_is_correct :
  portion_donated ≈ (53 / 159 * 100 : ℚ) := sorry

end vet_donation_portion_is_correct_l617_617249


namespace fifteenth_odd_multiple_of_five_l617_617198

theorem fifteenth_odd_multiple_of_five :
  ∃ a : ℕ, (∀ n : ℕ, a n = 5 + (n - 1) * 10) ∧ a 15 = 145 :=
by
  let a := λ n, 5 + (n - 1) * 10
  use a
  split
  { intros n,
    refl }
  { refl }
  sorry

end fifteenth_odd_multiple_of_five_l617_617198


namespace sum_of_conjugates_l617_617653

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617653


namespace problem1_problem2_l617_617355

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 6) - 2 * cos x

-- Problem 1
theorem problem1 {x : ℝ} (hx : x ∈ set.Icc (π / 2) π) (hsinx : sin x = 4 / 5) :
  f x = (4 * real.sqrt 3 + 3) / 5 :=
sorry

-- Problem 2
theorem problem2 :
  set.range (λ x, f x) = set.Icc 1 2 :=
sorry


end problem1_problem2_l617_617355


namespace jan_discount_percentage_l617_617038

theorem jan_discount_percentage :
  ∃ percent_discount : ℝ,
    ∀ (roses_bought dozen : ℕ) (rose_cost amount_paid : ℝ),
      roses_bought = 5 * dozen → dozen = 12 →
      rose_cost = 6 →
      amount_paid = 288 →
      (roses_bought * rose_cost - amount_paid) / (roses_bought * rose_cost) * 100 = percent_discount →
      percent_discount = 20 :=
by
  sorry

end jan_discount_percentage_l617_617038


namespace recycling_plan_l617_617254

def annual_growth_rate (y_2020 y_2022 number_of_years : Float) : Float :=
  (y_2022 / y_2020)^(1 / number_of_years) - 1

def predict_collection (y_prev_growth_rate : Float) (years : Nat) : Float :=
  y_prev * (1 + growth_rate)^years

theorem recycling_plan :
  let y_2020 := 40000
  let y_2022 := 90000
  let number_of_years := 2
  let predicted_growth_rate := annual_growth_rate y_2020 y_2022 number_of_years
  let y_2023 := predict_collection y_2022 predicted_growth_rate 1
  in 
  predicted_growth_rate = 0.5 ∧ y_2023 > 120000
:= by
  -- Proof
  sorry

end recycling_plan_l617_617254


namespace sum_of_digits_below_1000_l617_617733

theorem sum_of_digits_below_1000 : 
  (∑ n in Finset.range 1000, (n % 10) + (n / 10 % 10) + (n / 100 % 10)) = 13500 :=
by 
  sorry

end sum_of_digits_below_1000_l617_617733


namespace smallest_sum_of_digits_is_one_l617_617046

-- Conditions
def three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def distinct_digits (a b : ℕ) : Prop :=
  let digits_a := List.dedup (Nat.digits 10 a) in
  let digits_b := List.dedup (Nat.digits 10 b) in
  (a ≠ b) ∧
  (digits_a.length = 3) ∧ (digits_b.length = 3) ∧
  ((digits_a:List ℕ).disjoint digits_b)

def satisfies_conditions (a b : ℕ) : Prop :=
  three_digit_number a ∧ three_digit_number b ∧
  distinct_digits a b ∧
  (1000 ≤ a + b ∧ a + b < 10000)

-- Sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- Lean 4 statement for the problem
theorem smallest_sum_of_digits_is_one :
  ∀ (a b : ℕ), satisfies_conditions a b → sum_of_digits (a + b) = 1 :=
by
  sorry

end smallest_sum_of_digits_is_one_l617_617046


namespace correct_equation_among_given_choices_l617_617570

theorem correct_equation_among_given_choices :
  ∀ (A B C D : Prop),
    (A ↔ (∃ (x : ℝ), x = -((7:ℝ)^(1/3))))
    → (B ↔ (∃ (x : ℝ), (x = 7) ∨ (x = -7)))
    → (C ↔ (∃ (x : ℝ), x = 5 ∨ x = -5))
    → (D ↔ (∃ (x : ℝ), x = -3))
    → A ∧ ¬B ∧ ¬C ∧ ¬D :=
begin
  intros A B C D hA hB hC hD,
  split,
  { exact hA.2 (by rfl), },
  { split,
    { intro h,
      apply hB.2,
      -- expected to show inconsistency in assumptions
      exfalso,
      contradiction,
    },
    { split,
      { intro h,
        apply hC.2,
        exfalso,
        contradiction,
      },
      { intro h,
        apply hD.2,
        exfalso,
        contradiction,
      }
    }
  }
end

end correct_equation_among_given_choices_l617_617570


namespace largest_side_of_rectangle_l617_617453

theorem largest_side_of_rectangle :
  ∃ (l w : ℝ), (2 * l + 2 * w = 240) ∧ (l * w = 12 * 240) ∧ (l = 86.835 ∨ w = 86.835) :=
by
  -- Actual proof would be here
  sorry

end largest_side_of_rectangle_l617_617453


namespace fifteenth_odd_multiple_of_5_is_145_l617_617191

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l617_617191


namespace sum_radical_conjugate_l617_617645

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617645


namespace sum_of_radical_conjugates_l617_617685

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617685


namespace cone_volume_increase_l617_617010

open Real

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def new_height (h : ℝ) : ℝ := 2 * h
noncomputable def new_volume (r h : ℝ) : ℝ := cone_volume r (new_height h)

theorem cone_volume_increase (r h : ℝ) : new_volume r h = 2 * (cone_volume r h) :=
by
  sorry

end cone_volume_increase_l617_617010


namespace dealer_pricing_l617_617575

theorem dealer_pricing
  (cost_price : ℝ)
  (discount : ℝ := 0.10)
  (profit : ℝ := 0.20)
  (num_articles_sold : ℕ := 45)
  (num_articles_cost : ℕ := 40)
  (selling_price_per_article : ℝ := (num_articles_cost : ℝ) / num_articles_sold)
  (actual_cost_price_per_article : ℝ := selling_price_per_article / (1 + profit))
  (listed_price_per_article : ℝ := selling_price_per_article / (1 - discount)) :
  100 * ((listed_price_per_article - actual_cost_price_per_article) / actual_cost_price_per_article) = 33.33 := by
  sorry

end dealer_pricing_l617_617575


namespace inscribed_quadrilateral_opposite_angles_equal_l617_617927

theorem inscribed_quadrilateral_opposite_angles_equal 
  (A B C D O : Type) 
  (c : Circle O) 
  (hA : A ∈ c)
  (hB : B ∈ c)
  (hC : C ∈ c)
  (hD : D ∈ c) :
  ∠A + ∠C = 180 ∧ ∠B + ∠D = 180 := 
sorry

end inscribed_quadrilateral_opposite_angles_equal_l617_617927


namespace condition_sufficient_necessary_l617_617222

theorem condition_sufficient_necessary (x : ℝ) :
  (1 / 3) ^ x < 1 → x > 0 ∧ (0 < x ∧ x < 1) :=
by {
  intros h,
  have h1 : (1 / 3) ^ 0 = 1, by norm_num,
  have h2 : x > 0, by {
    apply (real.rpow_lt_rpow_iff (by norm_num) zero_lt_one).mpr,
    rwa h1,
  },
  have h3 : x < 1, by {
    have h_aux : 1 / x > 1, from by { assumption },
    linarith [(by linear_combination h_aux : 1 / x < 1)],
  },
  exact ⟨h2, ⟨h2, h3⟩⟩
}

end condition_sufficient_necessary_l617_617222


namespace range_of_a_l617_617002

theorem range_of_a (a : ℝ) :
  (∀ x, x > 4 → ∀ y, y > x → f y > f x)
  → (a ≥ -3) :=
begin
  intros H,
  sorry
end

def f (x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

end range_of_a_l617_617002


namespace total_tickets_sold_l617_617245

-- Define the conditions
variables (V G : ℕ)

-- Condition 1: Total revenue from VIP and general admission
def total_revenue_eq : Prop := 40 * V + 15 * G = 7500

-- Condition 2: There are 212 fewer VIP tickets than general admission
def vip_tickets_eq : Prop := V = G - 212

-- Main statement to prove: the total number of tickets sold
theorem total_tickets_sold (h1 : total_revenue_eq V G) (h2 : vip_tickets_eq V G) : V + G = 370 :=
sorry

end total_tickets_sold_l617_617245


namespace fifteenth_odd_multiple_of_five_l617_617201

theorem fifteenth_odd_multiple_of_five :
  ∃ a : ℕ, (∀ n : ℕ, a n = 5 + (n - 1) * 10) ∧ a 15 = 145 :=
by
  let a := λ n, 5 + (n - 1) * 10
  use a
  split
  { intros n,
    refl }
  { refl }
  sorry

end fifteenth_odd_multiple_of_five_l617_617201


namespace midpoints_form_regular_hexagon_l617_617334

noncomputable section

-- Variables and assumptions for the hexagon and equilateral triangles
variables {V : Type*} [inner_product_space ℝ V]

-- Assumption of centrally symmetric hexagon
def centrally_symmetric_hexagon (p : ℕ → V) : Prop :=
  ∀ i, p i + p (i + 3) = 0

-- Assumption of equilateral triangles on each side
def equilateral_triangle (a b c : V) : Prop :=
  dist a b = dist b c ∧ dist b c = dist c a

-- Definition for midpoints of segments between neighboring triangle vertices
def midpoint (p1 p2 : V) : V :=
  (p1 + p2) / 2

-- Theorem statement
theorem midpoints_form_regular_hexagon
  {hexagon : ℕ → V}
  (H_symm : centrally_symmetric_hexagon hexagon)
  (H_triangles : ∀ i, equilateral_triangle (hexagon i) (hexagon (i + 1)) (hexagon (i + 2))) :
  ∃ midpoints : ℕ → V, (∀ i, ∃ j, is_regular_hexagon midpoints) :=
sorry

end midpoints_form_regular_hexagon_l617_617334


namespace tangent_circle_circumference_correct_l617_617824
noncomputable def tangent_circle_circumference (AC BC : Circle) (A B : Point) (AB : Segment) (len_angle : ℝ) : ℝ :=
  /- The proof would involve geometric properties of the setup described and calculation -/
  if AC.center = B ∧ BC.center = A ∧ len_angle = 15 then 93.94 else 0

-- Statement theorem
theorem tangent_circle_circumference_correct :
  ∀ (AC BC : Circle) (A B : Point) (AB : Segment),
  (AC.center = B) →
  (BC.center = A) →
  (AC.measure BC = 15) →
  tangent_circle_circumference AC BC A B AB 15 = 93.94 :=
begin
  intros AC BC A B AB hAC hBC hlen,
  sorry,
end

end tangent_circle_circumference_correct_l617_617824


namespace percent_unionized_men_is_70_l617_617019

open Real

def total_employees : ℝ := 100
def percent_men : ℝ := 0.5
def percent_unionized : ℝ := 0.6
def percent_women_nonunion : ℝ := 0.8
def percent_men_nonunion : ℝ := 0.2

def num_men := total_employees * percent_men
def num_unionized := total_employees * percent_unionized
def num_nonunion := total_employees - num_unionized
def num_men_nonunion := num_nonunion * percent_men_nonunion
def num_men_unionized := num_men - num_men_nonunion

theorem percent_unionized_men_is_70 :
  (num_men_unionized / num_unionized) * 100 = 70 := by
  sorry

end percent_unionized_men_is_70_l617_617019


namespace percentage_of_music_students_l617_617987

theorem percentage_of_music_students 
  (total_students : ℕ) 
  (dance_students : ℕ) 
  (art_students : ℕ) 
  (drama_students : ℕ)
  (h_total : total_students = 2000) 
  (h_dance : dance_students = 450) 
  (h_art : art_students = 680) 
  (h_drama : drama_students = 370) 
  : (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 
:= by 
  sorry

end percentage_of_music_students_l617_617987


namespace fifteenth_odd_multiple_of_5_l617_617204

theorem fifteenth_odd_multiple_of_5 : ∃ (n : Nat), n = 15 → 10 * n - 5 = 145 :=
by
  intro n hn
  have h : 10 * 15 - 5 = 145 := by
    calc
      10 * 15 - 5 = 150 - 5 : by rw (Nat.mul_eq_mul_left n 10)
                ... = 145    : by rfl
  exact ⟨15, h⟩
  sorry

end fifteenth_odd_multiple_of_5_l617_617204


namespace sum_of_monomials_l617_617013

theorem sum_of_monomials (m n : ℕ) (x y : ℝ) (H : 3 * x^(m + 1) * y^2 + x^3 * y^n) :
  m + n = 4 := by
sorry

end sum_of_monomials_l617_617013


namespace count_x0_values_for_x0_eq_x5_l617_617320

noncomputable def recurrence_relation (x : ℝ) : ℕ → ℝ
| 0       := x
| (n + 1) := if 2 * recurrence_relation x n < 1 then 2 * recurrence_relation x n else 2 * recurrence_relation x n - 1

theorem count_x0_values_for_x0_eq_x5 :
  ∃ n : ℕ, n = 5 → (∀ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ < 1 → recurrence_relation x₀ 5 = x₀) ∧ 
  (∀ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ < 1 → recurrence_relation x₀ 5 <> 1) → 31 :=
sorry

end count_x0_values_for_x0_eq_x5_l617_617320


namespace length_of_AX_l617_617463

theorem length_of_AX
  (A B C D X : Point)
  (circle : Circle)
  (diameter_AD : circle.diameter = 1)
  (on_circle : ∀ (P : Point), P ∈ {A, B, C, D} → P ∈ circle)
  (on_diameter : X ∈ line_segment A D)
  (BX_eq_CX : distance B X = distance C X)
  (angle_conditions : 4 * angle A B C = angle B X C ∧ angle B X C = 48°):
  length A X = sin (24°) * sec (42°) := sorry

end length_of_AX_l617_617463


namespace curveC1_standard_eq_curveC2_cartesian_eq_l617_617350

-- Define the parametric equations for curve C_1
def curveC1 (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, 1 + Real.sin θ)

-- Define the polar equation for curve C_2
def curveC2 (ρ : ℝ) : Prop := ρ = 1

-- Translate the parametric equation of curve C_1 into a standard equation
theorem curveC1_standard_eq (θ : ℝ) : 
  let (x, y) := curveC1 θ in (x - 1)^2 + (y - 1)^2 = 1 := 
  sorry

-- Translate the polar equation of curve C_2 into a Cartesian coordinate equation
theorem curveC2_cartesian_eq : 
  ∀ x y : ℝ, curveC2 (Real.sqrt (x^2 + y^2)) → x^2 + y^2 = 1 := 
  sorry

end curveC1_standard_eq_curveC2_cartesian_eq_l617_617350


namespace sum_of_radii_of_tangent_circles_l617_617234

theorem sum_of_radii_of_tangent_circles (a b : ℝ) (h_c : (2 - a)^2 + (5 - a)^2 = a^2) :
  a ∈ {r | r^2 - 14 * r + 29 = 0} →
  r ∈ {r | r^2 - 14 * r + 29 = 0} →
  ∑ r in {(r : ℝ) | r^2 - 14 * r + 29}.to_finset, id = 14 :=
sorry

end sum_of_radii_of_tangent_circles_l617_617234


namespace club_officers_choice_l617_617458

theorem club_officers_choice:
  let members := 30
  let boys := 15
  let girls := 15 in
  let president_choices := members in
  let vp_choices := ((boys - 1) * girls) + ((girls - 1) * boys) in
  let secretary_choices := boys - 1 in
  president_choices * vp_choices * secretary_choices = 6300 :=
by
  /- proof would go here -/
  sorry

end club_officers_choice_l617_617458


namespace f_f_of_quarter_eq_neg_two_l617_617353

def f (x : ℝ) : ℝ := if x > 1 then log x / log (1/2) else 2 + 16^x

theorem f_f_of_quarter_eq_neg_two : f (f (1 / 4)) = -2 :=
by
  sorry

end f_f_of_quarter_eq_neg_two_l617_617353


namespace swimming_pool_dimensions_l617_617704

theorem swimming_pool_dimensions (x : ℝ) (cost : ℝ) (depth : ℝ) (short_side_cost : ℝ) (wall_cost : ℝ) (short_side_length : ℝ) (long_side_length : ℝ) (total_cost : ℝ) :
  depth = 2 ∧ 
  short_side_cost = 200 ∧ 
  wall_cost = 100 ∧ 
  total_cost = 7200 ∧ 
  short_side_length = x ∧ 
  long_side_length = 2 * x →
  (short_side_cost * x * 2 * x + wall_cost * (x + 2*x) * depth * 2 = total_cost) →
  (x = 3 ∧ 2 * x = 6) :=
begin
  sorry
end

end swimming_pool_dimensions_l617_617704


namespace slope_of_line_is_2_l617_617562

def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem slope_of_line_is_2 :
  slope 0 3 4 11 = 2 :=
by
  sorry

end slope_of_line_is_2_l617_617562


namespace sin_cos_sum_eq_five_over_sqrt_thirteen_k_range_l617_617786

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) ^ 2 - (Real.log x ^ 2 / Real.log 2) + 3
def P : ℝ × ℝ := (3, 2)
def α : ℝ := Real.arctan(2 / 3) -- since tan(α) = n / m => tan(α) = 2 / 3

-- Question 1: \( \sin α + \cos α = \frac{5}{\sqrt{13}} \)
theorem sin_cos_sum_eq_five_over_sqrt_thirteen :
  Real.sin α + Real.cos α = 5 / Real.sqrt 13 :=
by
  sorry

-- Question 2: Range of \( k \)
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi / 3) - 2
noncomputable def h (x k : ℝ) : ℝ := g x - k

theorem k_range :
  ∀ k, (∃ x1 x2, (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) ∧
                  (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧
                  h x1 k = 0 ∧ h x2 k = 0 ∧ x1 ≠ x2) ↔ k ∈ Set.Ioo (-5 : ℝ) (-7 / 2 : ℝ) :=
by
  sorry

end sin_cos_sum_eq_five_over_sqrt_thirteen_k_range_l617_617786


namespace fraction_B_or_C_l617_617988

theorem fraction_B_or_C (total_students : ℕ)
                        (percent_A : ℚ)
                        (failed_students : ℕ)
                        (h1 : total_students = 32)
                        (h2 : percent_A = 0.25)
                        (h3 : failed_students = 18) :
  let students_A := percent_A * total_students,
      students_remaining := total_students - students_A - failed_students in
  students_remaining / (total_students - students_A) = (1 : ℚ) / 4 := by
  sorry

end fraction_B_or_C_l617_617988


namespace polynomial_rational_root_property_l617_617098

theorem polynomial_rational_root_property (P : ℤ[X]) (p q : ℤ) (a : ℚ) (n : ℕ) 
    (h₁ : ∀ k : ℤ, k = p ∨ k = q → (P.eval k).abs = 1)
    (h₂ : p > q)
    (h₃ : P.eval_rat a = 0) : 
    p - q = 1 ∨ p - q = 2 ∧ a = (↑(p + q) / 2 : ℚ) :=
by
  sorry

end polynomial_rational_root_property_l617_617098


namespace range_of_m_for_two_distinct_zeros_l617_617390

noncomputable def quadratic_discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem range_of_m_for_two_distinct_zeros :
  ∀ (m : ℝ), quadratic_discriminant 1 (2*m) (m+2) > 0 ↔ (m < -1 ∨ m > 2) :=
begin
  intro m,
  rw [quadratic_discriminant, pow_two, mul_assoc, mul_comm],
  apply (lt_or_gt_of_ne (ne_of_gt (sub_pos_of_lt (by sorry)))).symm,
end

end range_of_m_for_two_distinct_zeros_l617_617390


namespace largest_integer_not_greater_than_expr_l617_617800

theorem largest_integer_not_greater_than_expr (x : ℝ) (hx : 20 * Real.sin x = 22 * Real.cos x) :
    ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := 
sorry

end largest_integer_not_greater_than_expr_l617_617800


namespace tens_digit_of_2013_squared_minus_2013_l617_617186

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 / 10 = 5 := by
  sorry

end tens_digit_of_2013_squared_minus_2013_l617_617186


namespace solve_inequality_l617_617105

theorem solve_inequality (x : ℝ) (h₀ : x ≠ -1) : 
  abs ((3*x - 2) / (x + 1)) > 3 ↔ x ∈ set.Iio (-1) ∪ set.Ioo (-1) (-1/6) :=
sorry

end solve_inequality_l617_617105


namespace fifteenth_odd_multiple_of_5_l617_617206

theorem fifteenth_odd_multiple_of_5 : ∃ (n : Nat), n = 15 → 10 * n - 5 = 145 :=
by
  intro n hn
  have h : 10 * 15 - 5 = 145 := by
    calc
      10 * 15 - 5 = 150 - 5 : by rw (Nat.mul_eq_mul_left n 10)
                ... = 145    : by rfl
  exact ⟨15, h⟩
  sorry

end fifteenth_odd_multiple_of_5_l617_617206


namespace part1_part2_l617_617585

theorem part1 : (π - 3)^0 + (-1)^(2023) - Real.sqrt 8 = -2 * Real.sqrt 2 := sorry

theorem part2 (x : ℝ) : (4 * x - 3 > 9) ∧ (2 + x ≥ 0) ↔ x > 3 := sorry

end part1_part2_l617_617585


namespace imaginary_part_z_l617_617349

-- Declare the given complex number
def z : ℂ := (1 + complex.i)^2 + complex.i^2010

-- Theorem stating that the imaginary part of z is 2
theorem imaginary_part_z : z.im = 2 := by
  -- proof
  sorry

end imaginary_part_z_l617_617349


namespace range_of_m_l617_617066

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_of_m {m : ℝ} 
  (h1 : ∀ x : ℝ, f (-x) = -f (x))      -- f(x) is odd
  (h2 : ∀ x : ℝ, f (x + 3) = f (x))    -- Smallest positive period is 3
  (h3 : f 1 > -2)                      -- f(1) > -2
  (h4 : f 2 = m - 3 / m) :             -- f(2) = m - 3/m
  m ∈ set.Ioo (0 : ℝ) 3 ∪ set.Ioo (-∞ : ℝ) (-1) :=
sorry

end range_of_m_l617_617066


namespace min_value_of_N_l617_617119

theorem min_value_of_N (N: ℕ) (h1: (∀ d ∈ digits 10 N, d = 0 ∨ d = 1)) (h2: 225 ∣ N) : N = 111111100 :=
sorry

end min_value_of_N_l617_617119


namespace find_angle_ACB_l617_617014

-- Define points A, B, C, D and the angles given in the problem constraints
variables {A B C D : Type} [euclidean_geometry A B C D]

-- Assume the known conditions
variables (angle_ABC : ℝ) (h1 : angle_ABC = 30)
variables (BD CD : ℝ) (h2 : 3 * BD = 2 * CD)
variables (angle_DAB : ℝ) (h3 : angle_DAB = 45)
variables (AD : segment A D)
variables (is_angle_bisector : is_bisector AD (angle CAB))

-- Define the goal: proving that angle ACB is 45 degrees
theorem find_angle_ACB (angle_ACB : ℝ)
  (h4 : angle_ACB = 45) : 
  angle_ABC = 30 → 3 * BD = 2 * CD → angle_DAB = 45 → is_bisector AD (angle BAC) → angle_ACB = 45 :=
by
{ sorry }

end find_angle_ACB_l617_617014


namespace sum_of_number_and_radical_conjugate_l617_617680

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617680


namespace translation_to_origin_l617_617029

theorem translation_to_origin {a b : ℤ} (h1 : a + 3 = 0) (h2 : b + 2 = 0) : (a, b) = (-3, -2) :=
by
  have ha : a = -3 := by linarith
  have hb : b = -2 := by linarith
  exact ⟨ha, hb⟩

end translation_to_origin_l617_617029


namespace geometric_sum_first_eight_terms_l617_617734

theorem geometric_sum_first_eight_terms :
  let a := (1 : ℚ) / 2,
      r := (1 : ℚ) / 3,
      n := 8 in
  (a * (1 - r^n) / (1 - r)) = 4920 / 6561 := by
sorry

end geometric_sum_first_eight_terms_l617_617734


namespace finite_set_product_sum_square_l617_617096

open Set

-- Define the problem statement in Lean 4
theorem finite_set_product_sum_square (A : Finset ℕ) (hA : ∀ x ∈ A, 0 < x) :
  ∃ (B : Finset ℕ), A ⊆ B ∧ (B.prod id = ∑ x in B, x^2) :=
sorry

end finite_set_product_sum_square_l617_617096


namespace not_n_eq_1992_l617_617413

theorem not_n_eq_1992 (n : ℕ) (a : ℕ → ℤ)
  (h1 : ∀ k, |a k| = 1)
  (h2 : ∀ j, a (n + j) = a j)
  (h3 : ∑ k in finset.range n, a k * a (k + 1) * a (k + 2) * a (k + 3) = 2) :
  n ≠ 1992 :=
sorry

end not_n_eq_1992_l617_617413


namespace find_G10_l617_617058

-- Define the conditions given in the problem
variable (G : ℝ → ℝ) 

-- The condition G(5) = 12
axiom G_at_5 : G(5) = 12

-- The condition given in the problem
axiom frac_relation : ∀ x : ℝ, 
  G(x + 4) * (16 - (64 * x + 80) / (x^2 + 8 * x + 16)) = G(4 * x)

-- The statement to prove
theorem find_G10 : G(10) = 54 :=
sorry

end find_G10_l617_617058


namespace sum_of_number_and_radical_conjugate_l617_617673

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617673


namespace sum_radical_conjugate_l617_617647

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617647


namespace remainder_when_divided_by_20_l617_617829

theorem remainder_when_divided_by_20 
  (n r : ℤ) 
  (k : ℤ)
  (h1 : n % 20 = r) 
  (h2 : 2 * n % 10 = 2)
  (h3 : 0 ≤ r ∧ r < 20)
  : r = 1 := 
sorry

end remainder_when_divided_by_20_l617_617829


namespace xy_sum_l617_617007

-- Define the problem conditions
variable (x y : ℚ)
variable (h1 : 1 / x + 1 / y = 4)
variable (h2 : 1 / x - 1 / y = -8)

-- Define the theorem to prove
theorem xy_sum : x + y = -1 / 3 := by
  sorry

end xy_sum_l617_617007


namespace solve_for_x_in_equation_l617_617142

theorem solve_for_x_in_equation : 
  ∃ x : ℝ, 0.75^x + 2 = 8 ∧ x = Real.log 6 / Real.log 0.75 := by
  sorry

end solve_for_x_in_equation_l617_617142


namespace eq_condition_implies_inequality_l617_617391

theorem eq_condition_implies_inequality (a : ℝ) (h_neg_root : 2 * a - 4 < 0) : (a - 3) * (a - 4) > 0 :=
by {
  sorry
}

end eq_condition_implies_inequality_l617_617391


namespace triathlon_bike_speed_l617_617717

theorem triathlon_bike_speed
  (total_time : ℝ)
  (swim_distance : ℝ) (swim_speed : ℝ)
  (run_distance : ℝ) (run_speed : ℝ)
  (bike_distance : ℝ)
  (required_bike_speed : ℝ) :
  total_time = 2 ∧
  swim_distance = 0.5 ∧ swim_speed = 3 ∧
  run_distance = 4 ∧ run_speed = 8 ∧
  bike_distance = 20 ∧
  let swim_time := swim_distance / swim_speed in
  let run_time := run_distance / run_speed in
  let total_non_bike_time := swim_time + run_time in
  let bike_time := total_time - total_non_bike_time in
  required_bike_speed = bike_distance / bike_time
  ↔ required_bike_speed = 15 :=
by
  intro h
  sorry

end triathlon_bike_speed_l617_617717


namespace somu_age_ratio_l617_617941

theorem somu_age_ratio (S F : ℕ) (h1 : S = 20) (h2 : S - 10 = (F - 10) / 5) : S / F = 1 / 3 :=
by
  sorry

end somu_age_ratio_l617_617941


namespace problem_minimal_positive_period_l617_617966

noncomputable def minimal_positive_period (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (x + π) ∧ π > 0 ∧ ∀ T > 0, T < π → ∃ x : ℝ, f x ≠ f (x + T)

theorem problem_minimal_positive_period : minimal_positive_period (λ x => Real.sin (2*x + Real.pi/3)) = π := sorry

end problem_minimal_positive_period_l617_617966


namespace smaller_angle_at_230_l617_617551

noncomputable def hourAngle (h m : ℕ) : ℝ := 30 * h + (m / 60) * 30
noncomputable def minuteAngle (m : ℕ) : ℝ := 6 * m

theorem smaller_angle_at_230 :
  let θ_hour := hourAngle 2 30 in
  let θ_minute := minuteAngle 30 in
  |θ_minute - θ_hour| = 105 :=
by 
  sorry

end smaller_angle_at_230_l617_617551


namespace point_inside_circle_l617_617777

theorem point_inside_circle (a : ℝ) :
  ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end point_inside_circle_l617_617777


namespace josephine_saving_l617_617870

/-- Josephine started saving a certain amount every day. After 20 days, he had saved 20 cents in total. -/
variable (x : ℕ)

/-- Josephine saved 20 cents in total after 20 days. -/
axiom h : 20 * x = 20

/-- Josephine was saving 1 cent every day. -/
theorem josephine_saving : x = 1 :=
by
  sorry

end josephine_saving_l617_617870


namespace Grace_minus_Lee_l617_617373

-- Definitions for the conditions
def Grace_calculation : ℤ := 12 - (3 * 4 - 2)
def Lee_calculation : ℤ := (12 - 3) * 4 - 2

-- Statement of the problem to prove
theorem Grace_minus_Lee : Grace_calculation - Lee_calculation = -32 := by
  sorry

end Grace_minus_Lee_l617_617373


namespace complement_of_set_M_l617_617805

open Set

def universal_set : Set ℝ := univ

def set_M : Set ℝ := {x | x^2 < 2 * x}

def complement_M : Set ℝ := compl set_M

theorem complement_of_set_M :
  complement_M = {x | x ≤ 0 ∨ x ≥ 2} :=
sorry

end complement_of_set_M_l617_617805


namespace smallest_n_integer_l617_617068

theorem smallest_n_integer (m n : ℕ) (s : ℝ) (h_m : m = (n + s)^4) (h_n_pos : 0 < n) (h_s_range : 0 < s ∧ s < 1 / 2000) : n = 8 := 
by
  sorry

end smallest_n_integer_l617_617068


namespace sum_of_number_and_its_radical_conjugate_l617_617660

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617660


namespace min_queries_to_determine_numbers_l617_617092

theorem min_queries_to_determine_numbers (cards : Fin 2005 → ℕ)
  (h_unique : Function.Injective cards)
  (query : Finset (Fin 2005) → Finset ℕ) :
  (∀ q : Finset (Fin 2005), q.card = 3 → query q = q.image cards) →
  ∃ N : ℕ, N = 1003 ∧ ∀ (strategy : ℕ → Finset (Fin 2005)), 
  (∀ i : ℕ, i < N → (strategy i).card = 3) →
  (∀ card_index : Fin 2005, ∃ i < N, card_index ∈ strategy i) → 
  ∀ card_index : Fin 2005, ∃ q : Finset (Fin 2005), q.card = 3 ∧ card_index ∈ q ∧ query q = q.image cards :=
begin
  sorry,
end

end min_queries_to_determine_numbers_l617_617092


namespace volume_of_cylinder_l617_617111

-- Define the given conditions
def base_circumference : ℝ := 4.8
def height : ℝ := 1.1
def pi_approx : ℝ := 3

-- Define the proof problem to show the volume of the cylinder
theorem volume_of_cylinder : 
  let R := base_circumference / (2 * pi_approx) in
  let V := pi_approx * R^2 * height in 
  V = 2112 := 
by
  sorry

end volume_of_cylinder_l617_617111


namespace intersection_is_empty_l617_617905

-- Define the universal set S
def S := {a, b, c, d, e : Type}

-- Define subsets M and N of S
def M := {a, c, d}
def N := {b, d, e}

-- Define the set complements with respect to the universal set S
def complement_S (A : Set S) : Set S := {x ∈ S | x ∉ A }

-- Define the problem's main question: the intersection of the complements
def intersection_of_complements : Set S := (complement_S M) ∩ (complement_S N)

-- The proposition we need to prove: this intersection is empty
theorem intersection_is_empty : intersection_of_complements = ∅ :=
by
  sorry

end intersection_is_empty_l617_617905


namespace spectral_density_stationary_random_function_l617_617301

open Complex

-- Defining the correlation function
def correlation_function (τ : ℝ) : ℝ :=
  Real.exp (-Real.abs τ) * Real.cos τ

-- Defining the spectral density
noncomputable def spectral_density (ω : ℝ) : ℝ :=
  1 / (2 * Real.pi * (1 + ω^2))

-- The theorem we want to prove
theorem spectral_density_stationary_random_function : 
  ∀ ω : ℝ, s_x ω = spectral_density ω := 
sorry

end spectral_density_stationary_random_function_l617_617301


namespace p_completes_work_in_10_days_l617_617217

noncomputable def work_completion_days : ℝ :=
  let x := 10 in
  have h1 : (2 / x + 3 * (1 / x + 1 / 6) = 1) := by sorry,
  x

theorem p_completes_work_in_10_days :
  work_completion_days = 10 := by
  sorry

end p_completes_work_in_10_days_l617_617217


namespace average_growth_rate_equation_l617_617974

-- Definitions and problem statement
def output_value_2019 : ℝ := 80
def output_value_2021 : ℝ := 96.8
def average_annual_growth_rate := x: ℝ

-- Prove that the equation 80(1 + x)^2 = 96.8 holds for the average annual growth rate
theorem average_growth_rate_equation (x : ℝ) :
  output_value_2019 * (1 + x) ^ 2 = output_value_2021 := 
by
  sorry

end average_growth_rate_equation_l617_617974


namespace brocard_fermat_equivalence_l617_617881

noncomputable def isBrocardPoint (P : Point) (A B C : Triangle) : Prop :=
  let S_PBC := area (triangle P B C)
  let S_PCA := area (triangle P C A)
  let S_PAB := area (triangle P A B)
  let S_ABC := area (triangle A B C)
  let a := side_length A B
  let b := side_length B C
  let c := side_length C A
  S_PBC / (c^2 * a^2) = S_PCA / (a^2 * b^2) ∧
  S_PCA / (a^2 * b^2) = S_PAB / (b^2 * c^2) ∧
  S_PAB / (b^2 * c^2) = S_ABC / (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)

axiom FermatPoint (P A B C : Triangle) : Prop

theorem brocard_fermat_equivalence (P A B C: Triangle):
  isBrocardPoint P A B C ↔ FermatPoint P A B C := sorry

end brocard_fermat_equivalence_l617_617881


namespace pencils_left_l617_617045

def initial_pencils := 4527
def given_to_dorothy := 1896
def given_to_samuel := 754
def given_to_alina := 307
def total_given := given_to_dorothy + given_to_samuel + given_to_alina
def remaining_pencils := initial_pencils - total_given

theorem pencils_left : remaining_pencils = 1570 := by
  sorry

end pencils_left_l617_617045


namespace arithmetic_sequence_ratio_l617_617110

theorem arithmetic_sequence_ratio
  (x y a1 a2 a3 b1 b2 b3 b4 : ℝ)
  (h1 : x ≠ y)
  (h2 : a1 = x + (1 * (a2 - a1)))
  (h3 : a2 = x + (2 * (a2 - a1)))
  (h4 : a3 = x + (3 * (a2 - a1)))
  (h5 : y = x + (4 * (a2 - a1)))
  (h6 : x = x)
  (h7 : b2 = x + (1 * (b3 - x)))
  (h8 : b3 = x + (2 * (b3 - x)))
  (h9 : y = x + (3 * (b3 - x)))
  (h10 : b4 = x + (4 * (b3 - x))) :
  (b4 - b3) / (a2 - a1) = 8 / 3 := by
  sorry

end arithmetic_sequence_ratio_l617_617110


namespace neg_proposition_P_l617_617826

theorem neg_proposition_P : 
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
by
  sorry

end neg_proposition_P_l617_617826


namespace largest_integer_le_zero_l617_617393

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero (x k : ℝ) (h1 : f x = 0) (h2 : 2 < x) (h3 : x < 3) : k ≤ x ∧ k = 2 :=
by
  sorry

end largest_integer_le_zero_l617_617393


namespace inequality_solution_l617_617106

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) ↔
  (1 < x ∧ x < 2 ∨ 3 < x ∧ x < 6) :=
by
  sorry

end inequality_solution_l617_617106


namespace 1_part1_2_part2_l617_617743

/-
Define M and N sets
-/
def M : Set ℝ := {x | x ≥ 1 / 2}
def N : Set ℝ := {y | y ≤ 1}

/-
Theorem 1: Difference set M - N
-/
theorem part1 : (M \ N) = {x | x > 1} := by
  sorry

/-
Define A and B sets and the condition A - B = ∅
-/
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {y | -1 / 2 < y ∧ y ≤ 2}

/-
Theorem 2: Range of values for a
-/
theorem part2 (a : ℝ) (h : A a \ B = ∅) : a ∈ Set.Iio (-12) ∪ Set.Ici 3 := by
  sorry

end 1_part1_2_part2_l617_617743


namespace possible_ones_digits_l617_617086

theorem possible_ones_digits (n : ℕ) (h : n % 4 = 0) : 
  ∃ d, d ∈ {0, 2, 4, 6, 8} ∧ (d = n % 10) :=
sorry

end possible_ones_digits_l617_617086


namespace total_problems_solved_is_480_l617_617512

def numberOfProblemsSolvedByMarvinYesterday := 40

def numberOfProblemsSolvedByMarvinToday (yesterday : ℕ) := 3 * yesterday

def totalProblemsSolvedByMarvin (yesterday today : ℕ) := yesterday + today

def totalProblemsSolvedByArvin (marvinTotal : ℕ) := 2 * marvinTotal

def totalProblemsSolved (marvinTotal arvinTotal : ℕ) := marvinTotal + arvinTotal

theorem total_problems_solved_is_480 :
  let y := numberOfProblemsSolvedByMarvinYesterday,
      t := numberOfProblemsSolvedByMarvinToday y,
      m_total := totalProblemsSolvedByMarvin y t,
      a_total := totalProblemsSolvedByArvin m_total,
      total := totalProblemsSolved m_total a_total in
  total = 480 :=
by
  let y := numberOfProblemsSolvedByMarvinYesterday
  let t := numberOfProblemsSolvedByMarvinToday y
  let m_total := totalProblemsSolvedByMarvin y t
  let a_total := totalProblemsSolvedByArvin m_total
  let total := totalProblemsSolved m_total a_total
  have : y = 40 := rfl
  have : t = 120 := by simp [numberOfProblemsSolvedByMarvinToday, this]
  have : m_total = 160 := by simp [totalProblemsSolvedByMarvin, this, this]
  have : a_total = 320 := by simp [totalProblemsSolvedByArvin, this]
  have : total = 480 := by simp [totalProblemsSolved, this, this]
  exact this

end total_problems_solved_is_480_l617_617512


namespace num_children_l617_617945

-- Defining the conditions
def num_adults : Nat := 10
def price_adult_ticket : Nat := 8
def total_bill : Nat := 124
def price_child_ticket : Nat := 4

-- Statement to prove: Number of children
theorem num_children (num_adults : Nat) (price_adult_ticket : Nat) (total_bill : Nat) (price_child_ticket : Nat) : Nat :=
  let cost_adults := num_adults * price_adult_ticket
  let cost_child := total_bill - cost_adults
  cost_child / price_child_ticket

example : num_children 10 8 124 4 = 11 := sorry

end num_children_l617_617945


namespace factory_A_higher_output_l617_617634

theorem factory_A_higher_output (a x : ℝ) (a_pos : a > 0) (x_pos : x > 0) 
  (h_eq_march : 1 + 2 * a = (1 + x) ^ 2) : 
  1 + a > 1 + x :=
by
  sorry

end factory_A_higher_output_l617_617634


namespace samia_total_journey_time_l617_617471

-- Define the conditions as constants or definitions
def biking_speed := 20 -- km/hr
def biking_time := 3 -- hours
def bus_speed := 60 -- km/hr
def bus_time := 0.5 -- hours
def walking_distance := 4 -- km
def walking_speed := 4 -- km/hr

-- State the total journey time problem
theorem samia_total_journey_time :
  let biking_time_total := biking_time -- time spent biking
  let bus_time_total := bus_time -- time spent on bus
  let walking_time := walking_distance / walking_speed -- time spent walking at 4 km/hr
  biking_time_total + bus_time_total + walking_time = 4.5 := sorry

end samia_total_journey_time_l617_617471


namespace abs_diff_of_numbers_l617_617145

theorem abs_diff_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by
  sorry

end abs_diff_of_numbers_l617_617145


namespace range_of_x_l617_617775

noncomputable def monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

def satisfies_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

def check_inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (1 - 2 * x^2) < f (1 + 2 * x - x^2)

theorem range_of_x 
  (f : ℝ → ℝ)
  (H_inc : monotonically_increasing_on f 2 (real.top))
  (H_sym : satisfies_symmetry f)
  (H_ineq : ∀ x, check_inequality f x) :
  ∀ x, H_ineq x → x ∈ Ioo (-2:ℝ) 0 := sorry

end range_of_x_l617_617775


namespace slope_l3_l617_617449

noncomputable def point : Type := ℝ × ℝ
def D : point := (-2, -3)
def E : point := (2, 2)  -- Solved from intersecting equations
def F : point := (22 / 5, 2)  -- Calculated from conditions

def line_1 (p : point) : Prop := 4 * p.1 - 3 * p.2 = 2
def line_2 (p : point) : Prop := p.2 = 2
def line_3 (p : point) : Prop := ∃ m, m > 0 ∧ p.2 = m * (p.1 - D.1) + D.2

def area (A B C : point) : ℝ :=
1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem slope_l3 :
  line_1 D ∧ line_1 E ∧ line_2 E ∧ line_3 F ∧ line_3 D ∧ area D E F = 6 →
  (∃ m : ℝ, m = (5 : ℝ) / (32 / 5) ∧ m = 25 / 32) := by
  intros h
  sorry

end slope_l3_l617_617449


namespace max_students_pool_visits_l617_617592

-- Define the set of days in September
def X₃₀ : Set ℕ := {x | 1 ≤ x ∧ x ≤ 30}

-- Define conditions on student's pool visits
def is_valid_student_pool_visits (student_visits : Set ℕ) : Prop :=
  student_visits ≠ ∅ ∧ student_visits ⊆ X₃₀

-- Prove the maximum number of students
theorem max_students_pool_visits : ∀ (students : Finset (Set ℕ)),
  (∀ s ∈ students, is_valid_student_pool_visits s) →
  (∀ s1 s2 ∈ students, s1 ≠ s2 → s1.card ≠ s2.card) →
  (∀ s1 s2 ∈ students, s1 ≠ s2 → ∃ d ∈ s1, d ∉ s2 ∧ ∃ d ∈ s2, d ∉ s1) →
  students.card ≤ 28 :=
sorry

end max_students_pool_visits_l617_617592


namespace star_property_l617_617796

-- Define the operation a ⋆ b = (a - b) ^ 3
def star (a b : ℝ) : ℝ := (a - b) ^ 3

-- State the theorem
theorem star_property (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := 
by 
  sorry

end star_property_l617_617796


namespace digits_of_product_l617_617374

theorem digits_of_product (n : ℕ) : 
  (4^5 * 5^10).log10.to_nat + 1 = 11 :=
sorry

end digits_of_product_l617_617374


namespace total_annual_interest_l617_617246

theorem total_annual_interest 
    (principal1 principal2 : ℝ)
    (rate1 rate2 : ℝ)
    (time : ℝ)
    (h1 : principal1 = 26000)
    (h2 : rate1 = 0.08)
    (h3 : principal2 = 24000)
    (h4 : rate2 = 0.085)
    (h5 : time = 1) :
    principal1 * rate1 * time + principal2 * rate2 * time = 4120 := 
sorry

end total_annual_interest_l617_617246


namespace find_value_at_point_l617_617346

def f : ℝ → ℝ := λ x, x^(1/2)

theorem find_value_at_point : f (1/9) = 1/3 :=
by
  sorry

end find_value_at_point_l617_617346


namespace train_initial_speed_l617_617595

variable (x : ℝ)
variable (h1 : (12 : ℝ) / 60 = 0.2)
variable (h2 : ∀ initial_speed : ℝ, ∀ delay_time : ℝ, ∀ distance : ℝ,
  delay_time = 0.2 → distance = 60 →
  (60 / initial_speed - 60 / (initial_speed + 15)) = 0.2 → initial_speed = 60)

theorem train_initial_speed :
  ∀ initial_speed : ℝ, ∀ delay_time : ℝ, ∀ distance : ℝ,
  delay_time = 0.2 → distance = 60 →
  (60 / initial_speed - 60 / (initial_speed + 15)) = 0.2 → initial_speed = 60 :=
by
  intros initial_speed delay_time distance h_delay h_distance h_equation
  rw [←h1] at h_delay
  exact h2 initial_speed delay_time distance h_delay h_distance h_equation

end train_initial_speed_l617_617595


namespace remaining_amount_division_l617_617573

-- Definitions
def total_amount : ℕ := 2100
def number_of_participants : ℕ := 8
def amount_already_raised : ℕ := 150

-- Proof problem statement
theorem remaining_amount_division :
  (total_amount - amount_already_raised) / (number_of_participants - 1) = 279 :=
by
  sorry

end remaining_amount_division_l617_617573


namespace fraction_halfway_between_fraction_halfway_between_l617_617180

theorem fraction_halfway_between : (3/4 : ℚ) < (5/7 : ℚ) :=
by linarith

theorem fraction_halfway_between : (41 / 56 : ℚ) = (1 / 2) * ((3 / 4) + (5 / 7)) :=
by sorry

end fraction_halfway_between_fraction_halfway_between_l617_617180


namespace sum_of_cubes_is_zero_l617_617143

theorem sum_of_cubes_is_zero 
  (a : Fin 10 → ℝ) 
  (h_sum_zero: ∑ i, a i = 0) 
  (h_sum_pairwise_zero: ∑ i j, ite (i < j) (a i * a j) 0 = 0) : 
  ∑ i, (a i)^3 = 0 :=
by
  sorry

end sum_of_cubes_is_zero_l617_617143


namespace range_of_a_for_f_increasing_l617_617317

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a_for_f_increasing :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_for_f_increasing_l617_617317


namespace jordan_fourth_period_shots_l617_617869

theorem jordan_fourth_period_shots (a b c d : ℕ) : 
  (a = 4) ∧ (b = 2 * a) ∧ (c = b - 3) ∧ (a + b + c + d = 21) → 
  d = 4 :=
by
  intros h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end jordan_fourth_period_shots_l617_617869


namespace sum_of_four_interior_edges_l617_617242

-- Define the given conditions
def is_two_inch_frame (w : ℕ) := w = 2
def frame_area (A : ℕ) := A = 68
def outer_edge_length (L : ℕ) := L = 15

-- Define the inner dimensions calculation function
def inner_dimensions (outerL outerH frameW : ℕ) := 
  (outerL - 2 * frameW, outerH - 2 * frameW)

-- Define the final question in Lean 4 reflective of the equivalent proof problem
theorem sum_of_four_interior_edges (w A L y : ℕ) 
  (h1 : is_two_inch_frame w) 
  (h2 : frame_area A)
  (h3 : outer_edge_length L)
  (h4 : 15 * y - (15 - 2 * w) * (y - 2 * w) = A)
  : 2 * (15 - 2 * w) + 2 * (y - 2 * w) = 26 := 
sorry

end sum_of_four_interior_edges_l617_617242


namespace polynomial_integer_values_l617_617424

open Polynomial

-- Define the main problem statement
theorem polynomial_integer_values (P : Polynomial ℝ) (n : ℕ) (k : ℤ) (hP_degree : P.degree ≤ (n : ℕ))
  (hP_int_values : ∀ i : ℕ, i ≤ n → P.eval (k + i) ∈ ℤ) :
  ∀ m : ℤ, P.eval m ∈ ℤ :=
begin
  sorry,
end

end polynomial_integer_values_l617_617424


namespace intersection_point_in_circle_l617_617123

theorem intersection_point_in_circle (a : ℝ) :
  (let P := (a, 3 * a) in (P.1 - 1)^2 + (P.2 - 1)^2 < 4) → (−(1 / 5) < a ∧ a < 1) :=
by
  intros h
  sorry

end intersection_point_in_circle_l617_617123


namespace sugar_flour_ratio_10_l617_617241

noncomputable def sugar_to_flour_ratio (sugar flour : ℕ) : ℕ :=
  sugar / flour

theorem sugar_flour_ratio_10 (sugar flour : ℕ) (hs : sugar = 50) (hf : flour = 5) : sugar_to_flour_ratio sugar flour = 10 :=
by
  rw [hs, hf]
  unfold sugar_to_flour_ratio
  norm_num
  -- sorry

end sugar_flour_ratio_10_l617_617241


namespace apple_distribution_l617_617624

theorem apple_distribution : 
  (∀ (a b c d : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → (a + b + c + d = 30) → 
  ∃ k : ℕ, k = (Nat.choose 29 3) ∧ k = 3276) :=
by
  intros a b c d h_pos h_sum
  use Nat.choose 29 3
  have h_eq : Nat.choose 29 3 = 3276 := by sorry
  exact ⟨rfl, h_eq⟩

end apple_distribution_l617_617624


namespace number_of_possible_values_for_r_l617_617130

def is_digit (x : ℕ) : Prop := x >= 0 ∧ x <= 9

theorem number_of_possible_values_for_r :
  ∃ r : ℝ, (∃ a b c d : ℕ, is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ r = a / 10 + b / 100 + c / 1000 + d / 10000) 
    ∧ (r >= 0.2614 ∧ r <= 0.2787)
    ∧ countable {r | r >= 0.2614 ∧ r <= 0.2787} = 174 :=
sorry

end number_of_possible_values_for_r_l617_617130


namespace max_value_of_m_l617_617052

variable (A : Set ℕ) (k n m : ℕ)
variable (B : Finset ℕ)

noncomputable def max_m (A : Set ℕ) (n k : ℕ) : ℕ :=
2^k

theorem max_value_of_m 
  (A : Set ℕ) (n k : ℕ) 
  (h1 : ∀ i, i > 0 ∧ i ≤ m → B i ⊆ A ∧ B i.card = k)
  (h2 : ∀ i, i > 0 ∧ i ≤ m → ∃ C : Finset ℕ, C ⊆ B i)
  (h3 : ∀ i j, i ≠ j → ((B i) ∩ (C j)) ≠ ((B j) ∩ (C i))) :
  m ≤ max_m A n k :=
by
  sorry

end max_value_of_m_l617_617052


namespace fifteenth_odd_multiple_of_5_l617_617202

theorem fifteenth_odd_multiple_of_5 : ∃ (n : Nat), n = 15 → 10 * n - 5 = 145 :=
by
  intro n hn
  have h : 10 * 15 - 5 = 145 := by
    calc
      10 * 15 - 5 = 150 - 5 : by rw (Nat.mul_eq_mul_left n 10)
                ... = 145    : by rfl
  exact ⟨15, h⟩
  sorry

end fifteenth_odd_multiple_of_5_l617_617202


namespace power_function_through_point_l617_617501

noncomputable def f : ℝ → ℝ := sorry

theorem power_function_through_point (h : ∀ x, ∃ a : ℝ, f x = x^a) (h1 : f 3 = 27) :
  f x = x^3 :=
sorry

end power_function_through_point_l617_617501


namespace incorrect_conclusion_C_l617_617789

noncomputable def f : ℝ → ℝ := λ x, Real.sin (2 * x - (Real.pi / 3))

lemma period_of_abs_f : ∃ p, p = Real.pi ∧ ∀ x, f(x) = f(x + p) :=
sorry

lemma center_of_symmetry : f(2 * Real.pi / 3) = 0 :=
sorry

lemma shifted_even_function : 
  ∀ x, f(x - Real.pi / 12) = -Real.cos(2 * x) :=
sorry

theorem incorrect_conclusion_C :
  ¬ (∀ a : ℝ, 
     (∀ x ∈ (-a..a), 
     Real.sin(2 * x - Real.pi / 3) > 0) 
     → a ≤ 5 * Real.pi / 12) :=
sorry

end incorrect_conclusion_C_l617_617789


namespace extremum_at_zero_monotonicity_f_product_inequality_l617_617357

-- Definitions and conditions for the problem
def f (x : ℝ) (a : ℝ) := log (1 + x^2) + a * x

-- Problem 1: Prove that if f(x) = ln(1 + x^2) + ax where a ≤ 0 has an extremum at x = 0, then a = 0
theorem extremum_at_zero (a : ℝ) (h : a ≤ 0) (h_extremum : ∀ x : ℝ, derivative (derivative (f x a)) x = 0 → x = 0) : a = 0 :=
  sorry

-- Problem 2: Discuss and prove the monotonicity of f(x) = ln(1 + x^2) + ax given a ≤ 0
theorem monotonicity_f (a : ℝ) (h : a ≤ 0) :
  (a = 0 → ∀ x : ℝ, if x > 0 then derivative (f x a) > 0 else derivative (f x a) < 0) ∧
  (a ≤ -1 → ∀ x : ℝ, derivative (f x a) ≤ 0) ∧
  (-1 < a → ∀ x : ℝ, (∃ b1 b2 : ℝ, b1 < b2 ∧ (derivative (f x a) > 0 ↔ b1 < x ∧ x < b2) ∧ derivative (f x a) < 0 → (x < b1 ∨ x > b2))) :=
  sorry

-- Problem 3: Prove that (1 + 1/4) * (1 + 1/16) * ... * (1 + 1/4^n) < e^(1 - 1/2^n)
theorem product_inequality (n : ℕ) (h : 0 < n) :
  (∏ i in range n, 1 + (1 / (4^i))) < real.exp (1 - (1 / (2^n))) :=
  sorry

end extremum_at_zero_monotonicity_f_product_inequality_l617_617357


namespace curve_is_circle_l617_617725

noncomputable def polar_curve (r : ℝ) : Prop :=
  r = 3 * real.sqrt 2

def is_circle (curve : Prop) : Prop :=
  curve = (polar_curve 3 (3 * real.sqrt 2))

theorem curve_is_circle : is_circle (polar_curve (3 * real.sqrt 2)) :=
by
  sorry

end curve_is_circle_l617_617725


namespace someMammalsAreForestDwellers_l617_617822

variable (Wolf ForestDweller Mammal : Type)
variable (isWolf : Wolf → Prop)
variable (isMammal : Mammal → Prop)
variable (isForestDweller : ForestDweller → Prop)

axiom wolvesAreMammals : ∀ (x : Wolf), isMammal x
axiom someForestDwellersAreWolves : ∃ (x : ForestDweller), ∃ (y : Wolf), isForestDweller x ∧ x = y

theorem someMammalsAreForestDwellers :
  ∃ (x : Mammal), ∃ (y : ForestDweller), isMammal x ∧ isForestDweller y ∧ x = y :=
by
  sorry

end someMammalsAreForestDwellers_l617_617822


namespace sum_ratio_f_l617_617316

noncomputable def f : ℕ+ → ℝ := sorry

theorem sum_ratio_f :
  (∀ a b : ℕ+, f (a + b) = f a * f b) →
  f 1 = 2 →
  ((finset.range 1014).sum (λ n, f (2 * (n + 1)) / f (2 * n + 1))) = 2018 :=
by {
  assume h_fun_eq : ∀ a b : ℕ+, f (a + b) = f a * f b,
  assume h_f1 : f 1 = 2,
  sorry -- Detailed proof will go here
}

end sum_ratio_f_l617_617316


namespace chord_length_of_intersection_l617_617793

noncomputable def chord_length (l : ℝ → ℝ → Prop) (c : ℝ × ℝ) (r : ℝ) : ℝ :=
  let d := abs (c.1 - c.2 + 3) / Real.sqrt (1^2 + (-1)^2) in
  2 * Real.sqrt (r^2 - (d / 2)^2)

theorem chord_length_of_intersection :
  chord_length (λ x y => x - y + 3 = 0) (-2, 2) (Real.sqrt 2) = Real.sqrt 6 :=
by
  sorry

end chord_length_of_intersection_l617_617793


namespace sum_of_number_and_radical_conjugate_l617_617677

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617677


namespace binomial_expansion_properties_l617_617407

-- condition
def binomial_expansion (n : Nat) := (3 * x - 1 / (2 * x)) ^ n

-- proof statement
/--
  Given the expansion (3x - 1/(2x))^8,
  1. The sum of coefficients in the expansion when x = 1 is 1 / 256.
  2. The rational terms for r = 2, r = 5, r = 8 are 7, -7/4 * x^{-4}, -1/256 * x^{-8} respectively.
-/
theorem binomial_expansion_properties :
  let n := 8 in
  let expansion := binomial_expansion n in
  (eval (expansion) 1) = (1 - 1 / 2) ^ 8 ∧
  (term_at_r expansion 2) = 7 ∧ 
  (term_at_r expansion 5) = -7 / 4 * x^{-4} ∧ 
  (term_at_r expansion 8) = -1 / 256 * x^{-8} := by
  sorry

/-- Definition to evaluate a polynomial at a given value -/
def eval (poly : Polynomial ℝ) (x : ℝ) : ℝ := poly.eval x

/-- Definition to get the term at index r in the expansion -/
def term_at_r (poly : Polynomial ℝ) (r : ℕ) : ℝ := (finiteBinomial r poly).coeff r

end binomial_expansion_properties_l617_617407


namespace amplitude_of_sine_wave_l617_617633

theorem amplitude_of_sine_wave (a b c : ℝ) (h : 0 < a) (h₁ : 0 < b) (h₂ : ∃ x, a * sin (b * x + c) = 3) : a = 3 :=
sorry

end amplitude_of_sine_wave_l617_617633


namespace price_of_coffee_table_l617_617049

-- Define the given values
def price_sofa : ℕ := 1250
def price_armchair : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Define the target value (price of the coffee table)
def price_coffee_table : ℕ := 330

-- The theorem to prove
theorem price_of_coffee_table :
  total_invoice = price_sofa + num_armchairs * price_armchair + price_coffee_table :=
by sorry

end price_of_coffee_table_l617_617049


namespace part1_part2_l617_617784

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - 2*x + 2 + a * Real.log x

theorem part1 (a : ℝ) (h : a = 1) : 
  let l := fun (x : ℝ) => x in 
  let t := (1 : ℝ) in 
  l t = 1 := 
begin
  sorry
end

theorem part2 (a x1 x2 : ℝ) (hx1x2 : x1 < x2) (hx1x2_roots : ∀ x, f x a = 0):
  f x2 a > (5 - 2 * Real.log 2) / 4 :=
begin
  sorry
end

end part1_part2_l617_617784


namespace sum_prime_factors_1242_l617_617565

-- Function to find the smallest prime factor of a number
def smallest_prime_factor (n : ℕ) : ℕ :=
  if n < 2 then n else
    (List.range (n - 1)).filter (λ x, x > 1 ∧ n % x = 0 ∧ Nat.Prime x).head'.getOrElse n

-- Function to find the largest prime factor of a number
def largest_prime_factor (n : ℕ) : ℕ :=
  if n < 2 then n else
    (List.range (n - 1)).reverse.filter (λ x, x > 1 ∧ n % x = 0 ∧ Nat.Prime x).head'.getOrElse n

-- Function to find the sum of smallest and largest prime factors
def sum_of_prime_factors (n : ℕ) : ℕ :=
  if n < 2 then 0 else
    smallest_prime_factor n + largest_prime_factor n

-- Lean 4 statement to prove that sum of largest and smallest prime factors of 1242 is 25
theorem sum_prime_factors_1242 : sum_of_prime_factors 1242 = 25 := by
  sorry

end sum_prime_factors_1242_l617_617565


namespace not_reachable_standard_state_for_odd_n_l617_617456

def chessboard (n : ℕ) : Type :=
  {m : ℕ // m < n}

def position (n : ℕ) : Type := chessboard n × chessboard n

def is_standard_state {n : ℕ} (board : position n → option (position n)) : Prop :=
  ∃ (empty_cell : position n),
    (∀ (p : position n), p ≠ empty_cell → board p = some p) ∧ board empty_cell = none

theorem not_reachable_standard_state_for_odd_n (n : ℕ) (h : n ≥ 3) (h_odd : n % 2 = 1) :
  ∀ (initial_board : position n → option (position n)),
    ∃ (moves : list (position n × position n)),
      (∀ (move : position n × position n), (move.1, move.2) ∈ moves → is_adjacent move.1 move.2) →
      ¬ is_standard_state (execute_moves initial_board moves) :=
sorry

noncomputable def is_adjacent {n : ℕ} (p1 p2 : position n) : Prop :=
  (abs (p1.1.1 - p2.1.1) = 1 ∧ p1.2.1 = p2.2.1) ∨
  (p1.1.1 = p2.1.1 ∧ abs (p1.2.1 - p2.2.1) = 1)

noncomputable def execute_moves {n : ℕ} (board : position n → option (position n)) (moves : list (position n × position n)) : position n → option (position n) :=
-- Implementation of how the moves would be executed on the board
sorry

end not_reachable_standard_state_for_odd_n_l617_617456


namespace number_of_friends_l617_617906

-- Define the initial amount of money John had
def initial_money : ℝ := 20.10 

-- Define the amount spent on sweets
def sweets_cost : ℝ := 1.05 

-- Define the amount given to each friend
def money_per_friend : ℝ := 1.00 

-- Define the amount of money left after giving to friends
def final_money : ℝ := 17.05 

-- Define a theorem to find the number of friends John gave money to
theorem number_of_friends (init_money sweets_cost money_per_friend final_money : ℝ) : 
  (init_money - sweets_cost - final_money) / money_per_friend = 2 :=
by
  sorry

end number_of_friends_l617_617906


namespace number_of_young_fish_l617_617911

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end number_of_young_fish_l617_617911


namespace tan_alpha_neg_seven_l617_617750

noncomputable def tan_alpha (α : ℝ) := Real.tan α

theorem tan_alpha_neg_seven {α : ℝ} 
  (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : Real.cos α ^ 2 + Real.sin (Real.pi + 2 * α) = 3 / 10) : 
  tan_alpha α = -7 := 
sorry

end tan_alpha_neg_seven_l617_617750


namespace ones_digit_of_tripling_4567_l617_617973

theorem ones_digit_of_tripling_4567 : 
  let n := 4567 in 
  let tripled := 3 * n in
  (tripled % 10) = 1 :=
by
  sorry

end ones_digit_of_tripling_4567_l617_617973


namespace smaller_angle_at_230_l617_617550

noncomputable def hourAngle (h m : ℕ) : ℝ := 30 * h + (m / 60) * 30
noncomputable def minuteAngle (m : ℕ) : ℝ := 6 * m

theorem smaller_angle_at_230 :
  let θ_hour := hourAngle 2 30 in
  let θ_minute := minuteAngle 30 in
  |θ_minute - θ_hour| = 105 :=
by 
  sorry

end smaller_angle_at_230_l617_617550


namespace abs_diff_a_b_l617_617308

def tau (n : ℕ) : ℕ := n.divisors.count
def S (n : ℕ) : ℕ := ∑ k in finset.range n.succ, tau (k)

noncomputable def a : ℕ := (finset.filter (λ n, (S n) % 2 = 1) (finset.range 2501)).card
noncomputable def b : ℕ := (finset.filter (λ n, (S n) % 2 = 0) (finset.range 2501)).card

theorem abs_diff_a_b : |a - b| = 1 := by
  sorry

end abs_diff_a_b_l617_617308


namespace find_angle_bisector_length_l617_617860

-- Define the problem context
variable (A B C D : Type) [Triangle ABC] 
variable (AC BC : ℝ) (angle_C : ℝ) (angle_bisector_CD : ℝ)

-- Specify given conditions
axiom AC_eq_6 : AC = 6
axiom BC_eq_9 : BC = 9
axiom angle_C_eq_120 : angle_C = 120

-- The theorem to prove the problem statement
theorem find_angle_bisector_length : angle_bisector_CD = 18 / 5 :=
by
    have h1 : AC = 6 := AC_eq_6
    have h2 : BC = 9 := BC_eq_9
    have h3 : angle_C = 120 := angle_C_eq_120
    sorry

end find_angle_bisector_length_l617_617860


namespace intersection_length_l_C1_is_1_min_distance_from_C2_to_line_l_l617_617795

-- Define the parametric forms of line l and curves C1, C2
def line_l (t : ℝ) : ℝ × ℝ := (1 + 0.5 * t, (√3/2) * t)

def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

def curve_C2 (θ : ℝ) : ℝ × ℝ := (0.5 * Real.cos θ, (√3/2) * Real.sin θ)

-- Prove the length |AB| where l intersects C1 is 1
theorem intersection_length_l_C1_is_1 :
  ∃ A B : ℝ × ℝ, A ≠ B ∧ (∃ t1 t2 θ1 θ2, line_l t1 = A ∧ line_l t2 = B ∧ curve_C1 θ1 = A ∧ curve_C1 θ2 = B) ∧ 
  Real.dist A B = 1 := by
  sorry

-- Prove the minimum distance from P on C2 to the line l is (sqrt(6)/4)*(sqrt(2)-1)
theorem min_distance_from_C2_to_line_l :
  ∃ (θ : ℝ), 
    ∀ (P : ℝ × ℝ), P = curve_C2 θ → 
    ∃ d : ℝ, (∀ t, ∃ l_point, line_l t = l_point ∧ d = Real.dist P l_point) ∧ 
    d = (√6/4) * (√2 - 1) := by
  sorry

end intersection_length_l_C1_is_1_min_distance_from_C2_to_line_l_l617_617795


namespace classroom_gpa_l617_617962

theorem classroom_gpa (x : ℝ) (h1 : (1 / 3) * x + (2 / 3) * 18 = 17) : x = 15 := 
by 
    sorry

end classroom_gpa_l617_617962


namespace trig_equalities_l617_617097

theorem trig_equalities (α β γ : ℝ) (h1 : Real.cos α = Real.tan β) (h2 : Real.cos β = Real.tan γ) (h3 : Real.cos γ = Real.tan α) :
  (Real.sin α)^2 = (Real.sin β)^2 ∧ (Real.sin β)^2 = (Real.sin γ)^2 ∧ (Real.sin γ)^2 = 4 * (Real.sin (Real.pi / 10))^2 ∧ 
  (Real.cos α)^4 = (Real.cos β)^4 ∧ (Real.cos β)^4 = (Real.cos γ)^4 :=
begin
  sorry
end

end trig_equalities_l617_617097


namespace box_colors_l617_617156

theorem box_colors (red white black : ℕ) (boxes : Fin 6 → ℕ) 
  (h_boxes : boxes = ![15, 16, 18, 19, 20, 31])
  (h_total : red + white + black = 119)
  (h_black_red : black = 2 * red)
  (h_white_one : ∃! i, boxes i % 3 = 2) :
  (∃! i, boxes i = 15 ∧ boxes.color i = red) ∧ 
  (∃! i, boxes.color i = black ∧ ∑ i in Ico 0 (boxes.count black), boxes i = 99) :=
by
  sorry

end box_colors_l617_617156


namespace amount_spent_on_shirt_l617_617915

-- Definitions and conditions
def total_spent_clothing : ℝ := 25.31
def spent_on_jacket : ℝ := 12.27

-- Goal: Prove the amount spent on the shirt is 13.04
theorem amount_spent_on_shirt : (total_spent_clothing - spent_on_jacket = 13.04) := by
  sorry

end amount_spent_on_shirt_l617_617915


namespace distance_from_right_angle_vertex_to_square_center_l617_617917

theorem distance_from_right_angle_vertex_to_square_center 
  {a b : ℝ} (h1 : a > 0) (h2 : b > 0) : 
  let AB := real.sqrt (a ^ 2 + b ^ 2)
  let diagonal := real.sqrt 2 * AB
  let M := diagonal / 2 
  sqrt (p := (a ^ 2 + b ^ 2) / 2) = M :=
sorry

end distance_from_right_angle_vertex_to_square_center_l617_617917


namespace probability_one_first_class_probability_two_first_class_probability_no_third_class_l617_617589

open Finset Nat

-- Define the conditions of the problem
def total_pens := 6
def first_class_pens := 3
def second_class_pens := 2
def third_class_pens := 1
def drawn_pens := 3

-- Define combinations
def comb (n k : ℕ) := (n.choose k).toNat

-- Calculate probabilities
def prob_of_event (f : ℝ) (s : ℝ) (t : ℝ) (total : ℝ) : ℝ := 
  (f * s * t) / total

-- Specific events' probabilities to be proven
theorem probability_one_first_class :
  prob_of_event (comb first_class_pens 1) 
                (comb (second_class_pens + third_class_pens) 2) 
                1 
                (comb total_pens drawn_pens : ℝ)
  = 9 / 20 := sorry

theorem probability_two_first_class :
  prob_of_event (comb first_class_pens 2) 
                (comb (second_class_pens + third_class_pens) 1) 
                1 
                (comb total_pens drawn_pens : ℝ)
  = 9 / 20 := sorry

theorem probability_no_third_class :
  prob_of_event (comb (first_class_pens + second_class_pens) 
                       drawn_pens) 
                1 
                1 
                (comb total_pens drawn_pens : ℝ)
  = 1 / 2 := sorry

end probability_one_first_class_probability_two_first_class_probability_no_third_class_l617_617589


namespace sum_of_number_and_radical_conjugate_l617_617675

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617675


namespace center_of_rotation_l617_617957

noncomputable def f (z : ℂ) : ℂ := 
  ((-1 - complex.I * real.sqrt 3) * z + (-2 * real.sqrt 3 + 18 * complex.I)) / 2

theorem center_of_rotation :
  ∃ c : ℂ, f c = c :=
begin
  use -2 * real.sqrt 3 - 4 * complex.I,
  sorry,
end

end center_of_rotation_l617_617957


namespace only_nonneg_solution_l617_617721

theorem only_nonneg_solution :
  ∀ (x y : ℕ), 2^x = y^2 + y + 1 → (x, y) = (0, 0) := by
  intros x y h
  sorry

end only_nonneg_solution_l617_617721


namespace fibonacci_seven_sum_every_second_term_upto_2023_sum_first_2023_terms_l617_617947

/-- Definition of Fibonacci sequence -/
def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

/-- Sum of the first n terms of the Fibonacci sequence -/
def fibonacci_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum fibonacci

/-- Proof that a_7 = 13 -/
theorem fibonacci_seven : fibonacci 7 = 13 :=
  by sorry

/-- Proof that a_1 + a_3 + ... + a_2023 = a_2024 -/
theorem sum_every_second_term_upto_2023 : 
  (Finset.range 1012).sum (λ k, fibonacci (2 * k + 1)) = fibonacci 2024 :=
  by sorry

/-- Proof that S_2023 = a_2025 - 1 -/
theorem sum_first_2023_terms : fibonacci_sum 2023 = fibonacci 2025 - 1 :=
  by sorry

end fibonacci_seven_sum_every_second_term_upto_2023_sum_first_2023_terms_l617_617947


namespace zagi_jumps_l617_617876
noncomputable def jumps_after_half (n : ℕ) : Prop :=
  if h : n >= 3 then
    let m := (n - 1) / 2 + 1
    (n - 1) % m = 0
  else false

theorem zagi_jumps (n : ℕ)
  (h : n >= 3)
  (H : ∀ m, (m <= (n - 1) / 2 + 1) → (m < n) → (n - 1) % m = 0 → m > 0) :
  ∀ m, (m <= (n - 1) / 2) → (m < n) → (n - 1) % m = 0 → m > 1 :=
begin
  sorry
end

end zagi_jumps_l617_617876


namespace negation_p_l617_617806

theorem negation_p (p : Prop) : 
  (∃ x : ℝ, x^2 ≥ x) ↔ ¬ (∀ x : ℝ, x^2 < x) :=
by 
  -- The proof is omitted
  sorry

end negation_p_l617_617806


namespace episodes_first_season_l617_617587

theorem episodes_first_season :
  ∃ (E : ℕ), (100000 * E + 200000 * (3 / 2) * E + 200000 * (3 / 2)^2 * E + 200000 * (3 / 2)^3 * E + 200000 * 24 = 16800000) ∧ E = 8 := 
by {
  sorry
}

end episodes_first_season_l617_617587


namespace mary_total_nickels_l617_617914

theorem mary_total_nickels : (7 + 12 + 9 = 28) :=
by
  sorry

end mary_total_nickels_l617_617914


namespace sum_radical_conjugate_l617_617642

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617642


namespace distance_behind_C_l617_617018

-- Conditions based on the problem
def distance_race : ℕ := 1000
def distance_B_when_A_finishes : ℕ := 50
def distance_C_when_B_finishes : ℕ := 100

-- Derived condition based on given problem details
def distance_run_by_B_when_A_finishes : ℕ := distance_race - distance_B_when_A_finishes
def distance_run_by_C_when_B_finishes : ℕ := distance_race - distance_C_when_B_finishes

-- Ratios
def ratio_B_to_A : ℚ := distance_run_by_B_when_A_finishes / distance_race
def ratio_C_to_B : ℚ := distance_run_by_C_when_B_finishes / distance_race

-- Combined ratio
def ratio_C_to_A : ℚ := ratio_C_to_B * ratio_B_to_A

-- Distance run by C when A finishes
def distance_run_by_C_when_A_finishes : ℚ := distance_race * ratio_C_to_A

-- Distance C is behind the finish line when A finishes
def distance_C_behind_when_A_finishes : ℚ := distance_race - distance_run_by_C_when_A_finishes

theorem distance_behind_C (d_race : ℕ) (d_BA : ℕ) (d_CB : ℕ)
  (hA : d_race = 1000) (hB : d_BA = 50) (hC : d_CB = 100) :
  distance_C_behind_when_A_finishes = 145 :=
  by sorry

end distance_behind_C_l617_617018


namespace abs_diff_of_two_numbers_l617_617148

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by 
  calc |x - y| = _ 
  sorry

end abs_diff_of_two_numbers_l617_617148


namespace log_inequality_l617_617925
noncomputable def prove_log_inequality (n : ℕ) (k : ℕ) (h : n ≥ 1) (hk : k = (multiset.card (unique_factorization_monoid.factorization n)).to_nat) : Prop :=
  log n ≥ k * log 2

theorem log_inequality (n : ℕ) (h₁ : n ≥ 1) : ∃ k, k = (multiset.card (unique_factorization_monoid.factorization n)).to_nat ∧ prove_log_inequality n k h₁ hk := 
by
  sorry -- Proof is left out as per the instructions

end log_inequality_l617_617925


namespace number_of_x0_eq_x5_l617_617322

noncomputable def x_seq (x0 : ℝ) (n : ℕ) : ℝ :=
  Nat.recOn n x0 (fun n xn =>
    if 2 * xn < 1 then 2 * xn else 2 * xn - 1)

theorem number_of_x0_eq_x5 : 
  ∀ (x0 : ℝ), (0 ≤ x0 ∧ x0 < 1) → 
              (x_seq x0 5 = x0) →
              (∃ count, count = 31 ∧ 
              (∀ x0 such that (0 ≤ x0 ∧ x0 < 1) ∧ x_seq x0 5 = x0, count = 31)) :=
by
  sorry

end number_of_x0_eq_x5_l617_617322


namespace ratio_mets_to_redsox_l617_617832

theorem ratio_mets_to_redsox (Y M R : ℕ)
  (h1 : Y / M = 3 / 2)
  (h2 : M = 96)
  (h3 : Y + M + R = 360) :
  M / R = 4 / 5 :=
by sorry

end ratio_mets_to_redsox_l617_617832


namespace sum_of_extreme_values_of_x_l617_617441

-- Define the variables and their conditions
variables (x y z : ℝ)

theorem sum_of_extreme_values_of_x :
  (x + y + z = 7) →
  (x^2 + y^2 + z^2 = 15) →
  (let m := min x (min y z) in
   let M := max x (max y z) in
   m + M = -4/3) :=
begin
  intros h1 h2,
  sorry, -- Proof goes here
end

end sum_of_extreme_values_of_x_l617_617441


namespace problem_l617_617063

def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 2)^(n - 1)

def sequence_S (n : ℕ) := ∑ i in finset.range (n + 1), sequence_a i

def sequence_b (n : ℕ) : ℝ :=
  n * (sequence_a n)

def sequence_T (n : ℕ) := ∑ i in finset.range (n + 1), sequence_b i

theorem problem (n : ℕ) (h1 : ∀ n, 2 * sequence_a (n + 1) + sequence_S n - 2 = 0) :
  (∀ n : ℕ, sequence_a (n + 1) = (1 / 2) ^ n) ∧ 
  (∀ n : ℕ, sequence_T n = 4 - (n + 2) * (1 / 2)^(n-1)) :=
by sorry

end problem_l617_617063


namespace symmetric_point_coords_l617_617847

def pointA : ℝ × ℝ := (1, 2)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def pointB : ℝ × ℝ := translate_left pointA 2

def pointC : ℝ × ℝ := reflect_origin pointB

theorem symmetric_point_coords :
  pointC = (1, -2) :=
by
  -- Proof omitted as instructed
  sorry

end symmetric_point_coords_l617_617847


namespace tower_no_knights_l617_617494

-- Define the problem conditions in Lean

variable {T : Type} -- Type for towers
variable {K : Type} -- Type for knights

variable (towers : Fin 9 → T)
variable (knights : Fin 18 → K)

-- Movement of knights: each knight moves to a neighboring tower every hour (either clockwise or counterclockwise)
variable (moves : K → (T → T))

-- Each knight stands watch at each tower exactly once over the course of the night
variable (stands_watch : ∀ k : K, ∀ t : T, ∃ hour : Fin 9, moves k t = towers hour)

-- Condition: at one time (say hour 1), each tower had at least two knights on watch
variable (time1 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 1
variable (cond1 : ∀ i : Fin 9, 2 ≤ time1 1 i)

-- Condition: at another time (say hour 2), exactly five towers each had exactly one knight on watch
variable (time2 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 2
variable (cond2 : ∃ seq : Fin 5 → Fin 9, (∀ i : Fin 5, time2 2 (seq i) = 1) ∧ ∀ j : Fin 4, i ≠ j → 1 ≠ seq j)

-- Prove: there exists a time when one of the towers had no knights at all
theorem tower_no_knights : ∃ hour : Fin 9, ∃ i : Fin 9, moves (knights i) (towers hour) = towers hour ∧ (∀ knight : K, moves knight (towers hour) ≠ towers hour) :=
sorry

end tower_no_knights_l617_617494


namespace problem_l617_617369

noncomputable def A (x y : ℝ) : Set ℝ := {1, (x+y)/2 - 1}
noncomputable def B (x y : ℝ) : Set ℝ := {-Real.log (x*y), x}
noncomputable def S (x y : ℝ) : ℝ := 
  ∑ k in Finset.range 1011, (x^(2*(k+1)) - 1/y^(2*(k+1)))

theorem problem {x y : ℝ} (h1 : 0 < y) (h2 : y < 2) (h3 : A x y = B x y) : 
  S x y = 0 :=
sorry

end problem_l617_617369


namespace wind_velocity_l617_617134

theorem wind_velocity (k : ℝ) (A1 A2 P1 P2 : ℝ) (V1 V2 : ℝ)
  (h1 : P1 = k * A1 * V1^2) 
  (h2 : P2 = k * A2 * V2^2) 
  (h3 : P1 = 4) 
  (h4 : A1 = 2) 
  (h5 : V1 = 20) 
  (h6 : P2 = 64) 
  (h7 : A2 = 4) : 
  V2 = 40 * real.sqrt 2 := 
  sorry

end wind_velocity_l617_617134


namespace f_decreasing_and_negative_in_interval_l617_617271

noncomputable def f : ℝ → ℝ :=
  sorry -- definition based on the given conditions to be completed

theorem f_decreasing_and_negative_in_interval :
  ∀ x, 1 < x ∧ x < 3/2 → (f x < 0 ∧ ∀ y, y > x → f y < f x) :=
begin
  sorry -- proof to be completed
end

end f_decreasing_and_negative_in_interval_l617_617271


namespace monotonicity_range_difference_inequality_l617_617362

-- Problem (I)
theorem monotonicity_range (a : ℝ) (f g : ℝ → ℝ) (h₁ : f = λ x, 1/2 * x^2 + a * log x)
  (h₂ : g = λ x, (a + 1) * x) (h₃ : a ≠ -1)
  (h₄ : ∀ x ∈ set.Icc (1 : ℝ) 3, (differentiable ℝ f) ∧ (differentiable ℝ g))
  (h₅ : ∀ x ∈ set.Icc (1 : ℝ) 3, deriv f x * deriv g x ≥ 0) :
  a > -1 ∨ a ≤ -9 :=
sorry

-- Problem (II)
theorem difference_inequality (a : ℝ) (F : ℝ → ℝ) (h₁ : 1 < a ∧ a ≤ real.exp 1)
  (h₂ : F = λ x, 1/2 * x^2 + a * log x - (a + 1) * x) (x₁ x₂ : ℝ)
  (h₃ : x₁ ∈ set.Icc 1 a) (h₄ : x₂ ∈ set.Icc 1 a) :
  |F x₁ - F x₂| < 1 :=
sorry

end monotonicity_range_difference_inequality_l617_617362


namespace intersection_probability_l617_617437

-- Define the set of possible values for a and b
def set_values : Finset ℕ := {1,2,3,4,5}

-- Define the condition for the line y = ax + b to intersect the circle x^2 + y^2 = 2
def intersects (a b : ℕ) : Prop := b^2 ≤ 2 * a^2 + 2

-- Count the total number of possible (a, b) pairs
def total_pairs : ℕ := set_values.card * set_values.card

-- Count the number of (a, b) pairs satisfying the condition
def satisfying_pairs : ℕ :=
  ((set_values.product set_values).filter (λ (p : ℕ × ℕ), intersects p.1 p.2)).card

-- Define the probability P
def probability : ℚ := satisfying_pairs / total_pairs

-- The theorem to prove the probability
theorem intersection_probability : probability = 19 / 25 :=
by
-- The proof is omitted
sorry

end intersection_probability_l617_617437


namespace snow_at_least_once_in_five_days_l617_617135

theorem snow_at_least_once_in_five_days :
  let p := (3/4 : ℚ)
  let q := (1/4 : ℚ)
  let probability_no_snow_in_five_days := q^5
  let probability_snow_at_least_once := 1 - probability_no_snow_in_five_days
    probability_snow_at_least_once = 1023 / 1024 :=
by {
  sorry
}

end snow_at_least_once_in_five_days_l617_617135


namespace range_of_m_l617_617460

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : y1 = x1^2 - 4*x1 + 3)
  (h2 : y2 = x2^2 - 4*x2 + 3) (h3 : -1 < x1) (h4 : x1 < 1)
  (h5 : m > 0) (h6 : m-1 < x2) (h7 : x2 < m) (h8 : y1 ≠ y2) :
  (2 ≤ m ∧ m ≤ 3) ∨ (m ≥ 6) :=
sorry

end range_of_m_l617_617460


namespace probability_black_pen_l617_617397

-- Define the total number of pens and the specific counts
def total_pens : ℕ := 5 + 6 + 7
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the probability calculation
def probability (total : ℕ) (count : ℕ) : ℚ := count / total

-- State the theorem
theorem probability_black_pen :
  probability total_pens black_pens = 1 / 3 :=
by sorry

end probability_black_pen_l617_617397


namespace count_valid_subsets_l617_617878

def is_valid_subset (A : Set ℕ) : Prop :=
  A ⊆ {1, 2, 3, 4, 5, 6, 7} ∧ ∀ a ∈ A, 8 - a ∈ A

theorem count_valid_subsets : 
  ∃ count, count = 15 ∧ count = (Set.to_finset {A | is_valid_subset A ∧ A ≠ ∅}).card := 
begin
  sorry
end

end count_valid_subsets_l617_617878


namespace solutions_to_equation_l617_617225

theorem solutions_to_equation (x : ℝ) : (x-1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
by 
  sorry

end solutions_to_equation_l617_617225


namespace triangle_construction_cases_l617_617326

-- Defining the given points and line
variables {E F : Type} [linear_ordered_field E] [linear_ordered_field F]
variables (A B C : E)
variables (l : set (E → E))
variables (altitude_A : E → E)

-- Conditions (E, F are feet of altitudes, and third altitude lies on l)
def feet_of_altitudes (B C AC AB : E) (E F : E) : Prop :=
  is_altitude B AC E ∧ is_altitude C AB F

-- Theorem statement
theorem triangle_construction_cases :
  (∃ A, ∃ B, ∃ C, feet_of_altitudes B C (line_through A C) (line_through A B) E F) →
  ((¬ passes_through_midpoint l (segment E F) ∧ ∃! ABC : Type, ∃ (ABC_triangle : triangle ABC), 
  feet_of_altitudes ABC_triangle.B ABC_triangle.C 
  (line_through ABC_triangle.A ABC_triangle.C) 
  (line_through ABC_triangle.A ABC_triangle.B) 
  E F) ∨ 
   (perpendicular_to l (segment E F) ∧ passes_through_midpoint l (segment E F) ∧ ∃ A B C : E, 
  ∃ ABC_triangle : triangle (A B C),
  feet_of_altitudes ABC_triangle.B ABC_triangle.C 
  (line_through ABC_triangle.A ABC_triangle.C) 
  (line_through ABC_triangle.A ABC_triangle.B) 
  E F ) ∨ 
   (passes_through_midpoint l (segment E F) ∧ ¬ perpendicular_to l (segment E F) ∧ ¬ ∃ A B C : E, 
  ∃ ABC_triangle : triangle (A B C), 
  feet_of_altitudes ABC_triangle.B ABC_triangle.C 
  (line_through ABC_triangle.A ABC_triangle.C) 
  (line_through ABC_triangle.A ABC_triangle.B) 
  E F))
:= by sorry

end triangle_construction_cases_l617_617326


namespace problem_amplitude_problem_period_problem_initial_phase_l617_617113

open Real

-- Definitions based on given problem conditions
def amplitude (f : ℝ → ℝ) : ℝ := f 0 - f (π / 2)
def period (f : ℝ → ℝ) : ℝ := 
  if ∃ T, ∀ x, f (x + T) = f x then Classical.some (exists_T) else 0
def initial_phase (f : ℝ → ℝ) : ℝ := 
  let t₀ := Classical.some (exists_t0)
  in t₀ - f t₀

-- Specific case according to the problem
def f (x : ℝ) : ℝ := (1 / 2) * sin (2 * x - π / 3)

theorem problem_amplitude : amplitude f = 1 / 2 := by
  sorry

theorem problem_period : period f = π := by
  sorry

theorem problem_initial_phase : initial_phase f = -π / 3 := by
  sorry

end problem_amplitude_problem_period_problem_initial_phase_l617_617113


namespace surface_area_cone_l617_617012

theorem surface_area_cone (r l : ℝ) (h1 : r = 1) (h2 : l = 2) : 
  (π * r^2 + π * r * l) = 3 * π :=
by
  rw [h1, h2]
  calc
    π * 1^2 + π * 1 * 2 = π + 2 * π : by rw [pow_two, mul_one, mul_one, mul_add]
    ... = 3 * π : by ring

end surface_area_cone_l617_617012


namespace number_of_solutions_l617_617742

theorem number_of_solutions (n : ℕ) : 
  ∃ solutions : ℕ, 
    (solutions = (n^2 - n + 1)) ∧ 
    (∀ x ∈ Set.Icc (1 : ℝ) n, (x^2 - ⌊x^2⌋ = frac x ^ 2) → (solutions ≠ 0)) :=
sorry

end number_of_solutions_l617_617742


namespace derivative_log_base2_l617_617727

noncomputable def log_base2 (x : ℝ) := Real.log x / Real.log 2

theorem derivative_log_base2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => log_base2 x) x = 1 / (x * Real.log 2) :=
by
  sorry

end derivative_log_base2_l617_617727


namespace purely_imaginary_value_of_m_third_quadrant_value_of_m_l617_617305

theorem purely_imaginary_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 2 * m ≠ 0) → m = -1/2 :=
by
  sorry

theorem third_quadrant_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 < 0) ∧ (m^2 - 2 * m < 0) → 0 < m ∧ m < 2 :=
by
  sorry

end purely_imaginary_value_of_m_third_quadrant_value_of_m_l617_617305


namespace repeating_fraction_period_sum_l617_617583

theorem repeating_fraction_period_sum 
  (p : ℕ) [prime : nat.prime p]
  (period_length : (1 < p) ∧ (nat.gcd 10 p = 1) ∧ ((10^(p-1) - 1) / p).natAbs * p = (10^(p-1) - 1).natAbs)
  (a b k : ℕ)
  (h1 : p - 1 = 2 * k)
  (h2 : 10^k + 1 = p * c)
  (h3 : (nat.digits 10 (1 / p)) = (a :: b :: nil)) :
  a + b = 10^k - 1 :=
sorry

end repeating_fraction_period_sum_l617_617583


namespace ribbon_cutting_l617_617532

theorem ribbon_cutting :
  ∃ (gcd_segments number_segments : ℕ), gcd_segments = Nat.gcd 28 16 ∧ gcd_segments = 4 ∧ number_segments = (28 + 16) / gcd_segments ∧ number_segments = 11 :=
by
  use Nat.gcd 28 16, (28 + 16) / Nat.gcd 28 16
  have h_gcd : Nat.gcd 28 16 = 4 := by simp [Nat.gcd]
  have h_num : (28 + 16) / Nat.gcd 28 16 = 11 := by norm_num
  simp [h_gcd, h_num]
  exact ⟨rfl, rfl⟩

end ribbon_cutting_l617_617532


namespace even_perfect_square_factors_count_l617_617809

theorem even_perfect_square_factors_count : 
  let factors := {n : ℕ | ∃ a b : ℕ, n = 2^a * 7^b ∧ 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 10 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ 1 ≤ a}
  in factors.card = 18 :=
by
  sorry

end even_perfect_square_factors_count_l617_617809


namespace intersection_subset_proper_l617_617430

-- Definitions of P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- The problem statement to prove
theorem intersection_subset_proper : P ∩ Q ⊂ P := by
  sorry

end intersection_subset_proper_l617_617430


namespace mean_score_l617_617738

theorem mean_score (M SD : ℝ) (h1 : 58 = M - 2 * SD) (h2 : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l617_617738


namespace solution_set_of_inequality_l617_617141

theorem solution_set_of_inequality :
  {x : ℝ | 4 ^ x - 6 * 2 ^ x + 8 < 0} = set.Ioo 1 2 :=
by
  sorry

end solution_set_of_inequality_l617_617141


namespace fifteenth_odd_multiple_of_5_l617_617205

theorem fifteenth_odd_multiple_of_5 : ∃ (n : Nat), n = 15 → 10 * n - 5 = 145 :=
by
  intro n hn
  have h : 10 * 15 - 5 = 145 := by
    calc
      10 * 15 - 5 = 150 - 5 : by rw (Nat.mul_eq_mul_left n 10)
                ... = 145    : by rfl
  exact ⟨15, h⟩
  sorry

end fifteenth_odd_multiple_of_5_l617_617205


namespace common_elements_count_l617_617060

theorem common_elements_count (S T : Set ℕ) (hS : S = {n | ∃ k : ℕ, k < 3000 ∧ n = 5 * (k + 1)})
    (hT : T = {n | ∃ k : ℕ, k < 3000 ∧ n = 8 * (k + 1)}) :
    S ∩ T = {n | ∃ m : ℕ, m < 375 ∧ n = 40 * (m + 1)} :=
by {
  sorry
}

end common_elements_count_l617_617060


namespace order_of_probability_l617_617629

theorem order_of_probability (E1 E2 E3 E4 E5 : Prop) : 
  (probability_of_event E1 = 2/3) ∧ 
  (probability_of_event E2 = 1) ∧ 
  (probability_of_event E3 = 1/3) ∧ 
  (probability_of_event E4 = 1/2) ∧ 
  (probability_of_event E5 = 0) → 
  order [E5, E3, E4, E1, E2] := 
sorry

-- Define probabilities
axiom probability_of_event : Prop → Real

-- Definition for order in terms of event probabilities
def order (events : list Prop) : Prop := 
  match events with
  | [a, b, c, d, e] => 
    probability_of_event a < probability_of_event b ∧
    probability_of_event b < probability_of_event c ∧
    probability_of_event c < probability_of_event d ∧
    probability_of_event d < probability_of_event e
  | _ => False

end order_of_probability_l617_617629


namespace value_of_polynomial_l617_617207

theorem value_of_polynomial : 
  99^5 - 5 * 99^4 + 10 * 99^3 - 10 * 99^2 + 5 * 99 - 1 = 98^5 := by
  sorry

end value_of_polynomial_l617_617207


namespace digit_for_divisibility_by_5_l617_617549

theorem digit_for_divisibility_by_5 (B : ℕ) (B_digit_condition : B < 10) :
  (∃ k : ℕ, 6470 + B = 5 * k) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_for_divisibility_by_5_l617_617549


namespace uma_income_l617_617582

theorem uma_income
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 5000)
  (h2 : 3 * x - 2 * y = 5000) :
  4 * x = 20000 :=
by
  sorry

end uma_income_l617_617582


namespace sum_D_E_correct_sum_of_all_possible_values_of_D_E_l617_617000

theorem sum_D_E_correct :
  ∀ (D E : ℕ), (D < 10) → (E < 10) →
  (∃ k : ℕ, (10^8 * D + 4650000 + 1000 * E + 32) = 7 * k) →
  D + E = 1 ∨ D + E = 8 ∨ D + E = 15 :=
by sorry

theorem sum_of_all_possible_values_of_D_E :
  (1 + 8 + 15) = 24 :=
by norm_num

end sum_D_E_correct_sum_of_all_possible_values_of_D_E_l617_617000


namespace sum_of_ab_for_sqrt_factorized_15_l617_617252

theorem sum_of_ab_for_sqrt_factorized_15 :
  let n := 15!
  let q := 4
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ sqrt (n) = a * sqrt (b) ∧ sum_of_all_ab n = q * n :=
sorry

-- Helper definitions would be provided here, like prime_factorization, perfect_square_factors, sum_of_all_ab, etc.

end sum_of_ab_for_sqrt_factorized_15_l617_617252


namespace fraction_halfway_between_fraction_halfway_between_l617_617178

theorem fraction_halfway_between : (3/4 : ℚ) < (5/7 : ℚ) :=
by linarith

theorem fraction_halfway_between : (41 / 56 : ℚ) = (1 / 2) * ((3 / 4) + (5 / 7)) :=
by sorry

end fraction_halfway_between_fraction_halfway_between_l617_617178


namespace sum_reciprocal_l617_617341

open Real

theorem sum_reciprocal (y : ℝ) (h₁ : y^3 + (1 / y)^3 = 110) : y + (1 / y) = 5 :=
sorry

end sum_reciprocal_l617_617341


namespace total_page_difference_is_97_l617_617024

theorem total_page_difference_is_97 : 
  let first_book := (48, 11, 24)
  let second_book := (35, 18, 28)
  let third_book := (62, 19, 12)
  let pages_first_chapter := [first_book.1, second_book.1, third_book.1]
  let pages_second_chapter := [first_book.2, second_book.2, third_book.2]
  (pages_first_chapter.zip pages_second_chapter).map (λ (p1, p2), p1 - p2).sum = 97 :=
by
  sorry

end total_page_difference_is_97_l617_617024


namespace parallelepiped_surface_area_and_volume_l617_617490

-- Define the conditions from the problem statement.
variable (a : Real)
variable (acute_angle base_angle : Real)
variable (is_rhombus : is_rhombus_with_side_length_and_angle a acute_angle)
variable (is_acute_angle_30 : acute_angle = 30 * Real.pi / 180)
variable (lateral_edge_angle : base_angle = 60 * Real.pi / 180)
variable (diagonal_perpendicular : is_diagonal_perpendicular_to_base)

-- Theorem statement for the total surface area and volume of the parallelepiped
theorem parallelepiped_surface_area_and_volume :
  total_surface_area a acute_angle base_angle diagonal_perpendicular = a^2 * (1 + Real.sqrt 3 + Real.sqrt 13)
  ∧ volume a acute_angle base_angle diagonal_perpendicular = a^3 * Real.sqrt 3 / 2 := 
sorry

end parallelepiped_surface_area_and_volume_l617_617490


namespace triangle_inequality_for_powers_l617_617548

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, (a ^ n + b ^ n > c ^ n)) ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b = c) :=
sorry

end triangle_inequality_for_powers_l617_617548


namespace sum_of_squares_of_real_roots_l617_617780

theorem sum_of_squares_of_real_roots (k : ℝ) (h1 : ∃ x : ℝ, x^4 + 2*x^3 + (3+k)*x^2 + (2+k)*x + 2*k = 0) 
    (h2 : ∏ x in {x : ℝ | x^4 + 2*x^3 + (3+k)*x^2 + (2+k)*x + 2*k = 0}, x = -2) : 
    let roots := {x : ℝ | x^4 + 2*x^3 + (3+k)*x^2 + (2+k)*x + 2*k = 0} in
    ∑ x in roots, x^2 = 5 :=
by
  sorry

end sum_of_squares_of_real_roots_l617_617780


namespace green_peaches_total_l617_617989

theorem green_peaches_total 
  (n : ℕ) (g : ℕ) 
  (h_n : n = 7)
  (h_g : g = 2) :
  n * g = 14 :=
by {
  rw [h_n, h_g],
  norm_num,
  sorry
}

end green_peaches_total_l617_617989


namespace max_mass_of_grain_l617_617602

theorem max_mass_of_grain (length width : ℝ) (angle : ℝ) (density : ℝ) 
  (h_length : length = 10) (h_width : width = 5) (h_angle : angle = 45) (h_density : density = 1200) : 
  volume * density = 175000 :=
by
  let height := width / 2
  let base_area := length * width
  let prism_volume := base_area * height
  let pyramid_volume := (1 / 3) * (width / 2 * length) * height
  let total_volume := prism_volume + 2 * pyramid_volume
  let volume := total_volume
  sorry

end max_mass_of_grain_l617_617602


namespace sum_of_cubes_from_neg50_to_50_l617_617256

theorem sum_of_cubes_from_neg50_to_50 :
  ∑ k in Finset.range 101, (k - 50 : ℤ)^3 = 0 :=
by
  sorry

end sum_of_cubes_from_neg50_to_50_l617_617256


namespace distances_form_triangle_l617_617846

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def line_equation (x y a : ℝ) : Prop :=
  x + y - 3 * a = 0

noncomputable def distance_to_line (x y A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / sqrt (A^2 + B^2) 

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sqrt (a^2 - b^2) / a

theorem distances_form_triangle
  (a b : ℝ) (ha : a = 2) (hb : b = 1)
  (ecc : eccentricity a b = sqrt 3 / 2)
  (dist_O_to_l : distance_to_line 0 0 1 1 (-6) = 3 * sqrt 2) :
  ∀ θ : ℝ,
  let d1 := distance_to_line (2 * cos θ) (sin θ) 1 1 (-6),
      d2 := distance_to_line 0 1 1 1 (-6),
      d3 := distance_to_line 2 0 1 1 (-6) in
  d1 + d2 > d3 ∧ d2 + d3 > d1 ∧ d3 + d1 > d2 :=
by {
  intros,
  let d1 := distance_to_line (2 * cos θ) (sin θ) 1 1 (-6),
  let d2 := distance_to_line 0 1 1 1 (-6),
  let d3 := distance_to_line 2 0 1 1 (-6),
  sorry
}

end distances_form_triangle_l617_617846


namespace min_tan_ABC_l617_617838

theorem min_tan_ABC (A B C : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) 
  (h4 : A + B + C = π) (h5 : Real.sin A = 3 * Real.sin B * Real.sin C) :
  ∃ (x : ℝ), x = 12 ∧ ∀ (tan_product : ℝ), tan_product = Real.tan A * Real.tan B * Real.tan C → tan_product ≥ x :=
begin
  sorry
end

end min_tan_ABC_l617_617838


namespace ab_zero_l617_617210

theorem ab_zero (a b : ℝ) (x : ℝ) (h : ∀ x : ℝ, a * x + b * x ^ 2 = -(a * (-x) + b * (-x) ^ 2)) : a * b = 0 :=
by
  sorry

end ab_zero_l617_617210


namespace sum_of_conjugates_l617_617651

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617651


namespace clock_angle_at_230_l617_617553

/-- Angle calculation problem: Determine the degree measure of the smaller angle formed by the 
    hands of a clock at 2:30. -/
theorem clock_angle_at_230 : 
  let angle_per_hour := 360 / 12,
      hour_position := 2 * angle_per_hour + (angle_per_hour / 2),
      minute_position := 30 * 6,
      angle_difference := abs (minute_position - hour_position)
  in angle_difference = 105 :=
by
  sorry

end clock_angle_at_230_l617_617553


namespace problem_estimation_l617_617608

noncomputable def lattice_points : Finset (ℕ × ℕ) := Finset.product (Finset.range 6) (Finset.range 6)

def expected_area : ℝ :=
  -- This placeholder function should compute the expected area of the convex hull
  -- of 6 distinct uniformly chosen lattice points from {1,2,3,4,5,6}^2.
  sorry

def compute_N (A : ℝ) : ℕ :=
  Nat.floor (10^4 * A)

theorem problem_estimation :
  compute_N expected_area = 104552 :=
sorry

end problem_estimation_l617_617608


namespace max_sum_of_factors_of_48_l617_617314

theorem max_sum_of_factors_of_48 (d Δ : ℕ) (h : d * Δ = 48) : d + Δ ≤ 49 :=
sorry

end max_sum_of_factors_of_48_l617_617314


namespace geometric_log_sequence_none_of_these_l617_617338

theorem geometric_log_sequence_none_of_these
  (a : ℕ) (r : ℝ)
  (h_pos : 1 < a)
  (b := a * r)
  (c := a * r^3)
  (has_geometric_progression : 1 < a ∧ a < b ∧ b < c)
  (n := a^2) :
  (¬ is_arithmetic_progression [Real.log n / Real.log a, Real.log n / Real.log b, Real.log n / Real.log c] ∧
   ¬ is_geometric_progression [Real.log n / Real.log a, Real.log n / Real.log b, Real.log n / Real.log c] ∧
   ¬ is_arithmetic_progression [1 / (Real.log n / Real.log a), 1 / (Real.log n / Real.log b), 1 / (Real.log n / Real.log c)] ∧
   ¬ ∀ k : ℝ, ∀ i j : ℕ, (Real.log n / Real.log a) * k^i = (Real.log n / Real.log b) ∧
                    (Real.log n / Real.log b) * k^j = (Real.log n / Real.log c)) :=
sorry

end geometric_log_sequence_none_of_these_l617_617338


namespace total_onions_l617_617472

theorem total_onions (sara sally fred amy matthew : Nat) 
  (hs : sara = 40) (hl : sally = 55) 
  (hf : fred = 90) (ha : amy = 25) 
  (hm : matthew = 75) :
  sara + sally + fred + amy + matthew = 285 := 
by
  sorry

end total_onions_l617_617472


namespace min_ab_value_l617_617315

noncomputable theory

-- Given parameters and conditions
variables (b a : ℝ)
variables (h_b : b > 0)
variables (perp : (b^2 + 1) * (1 / b^2) = -a)

-- Minimum value of ab
def min_ab : ℝ :=
  2
-- The statement that needs to be proved
theorem min_ab_value (hperp : (b^2 + 1) * (1 / b^2) = -a) : ab = min_ab := by
  sorry

end min_ab_value_l617_617315


namespace top_width_of_channel_l617_617949

theorem top_width_of_channel (b : ℝ) (A : ℝ) (h : ℝ) (w : ℝ) : 
  b = 8 ∧ A = 700 ∧ h = 70 ∧ (A = (1/2) * (w + b) * h) → w = 12 := 
by 
  intro h1
  sorry

end top_width_of_channel_l617_617949


namespace sum_of_radical_conjugates_l617_617681

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l617_617681


namespace find_ratio_l617_617017

-- Define the given conditions in Lean
structure Triangle :=
  (A B C : Point)
  (AB BC AC : ℝ)
  (hAB : AB = 10)
  (hBC : BC = 14)
  (hAC : AC = 18)

structure PointOnSegment (A C : Point) := 
  (N : Point)
  (AN NC : ℝ)
  (hSumRadii : (r_incircle (triangle A B N) + r_incircle (triangle C B N)) = 10)

-- Define the main theorem to be proven
theorem find_ratio (T : Triangle) (N : PointOnSegment T.A T.C)
  (hRat : (N.AN : ℝ) / N.NC = (1 : ℝ) / 2) : (1 : ℝ) + 2 = 3 :=
sorry

end find_ratio_l617_617017


namespace golf_problem_l617_617164

variable (rounds : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ)
variable (total_strokes : ℕ := rounds * avg_strokes_per_hole)
variable (total_par : ℕ := rounds * par_per_hole)
variable (strokes_over_par : ℕ := total_strokes - total_par)

theorem golf_problem (h1 : rounds = 9) (h2 : avg_strokes_per_hole = 4) (h3 : par_per_hole = 3) :
    strokes_over_par = 9 :=
by
  unfold strokes_over_par total_strokes total_par
  rw [h1, h2, h3]
  unfold total_strokes total_par
  simp
  sorry

end golf_problem_l617_617164


namespace ratio_sea_horses_penguins_l617_617630

def sea_horses := 70
def penguins := sea_horses + 85

theorem ratio_sea_horses_penguins : (70 : ℚ) / (sea_horses + 85) = 14 / 31 :=
by
  -- Proof omitted
  sorry

end ratio_sea_horses_penguins_l617_617630


namespace find_number_of_men_l617_617032

noncomputable def initial_conditions (x : ℕ) : ℕ × ℕ :=
  let men := 4 * x
  let women := 5 * x
  (men, women)

theorem find_number_of_men (x : ℕ) : 
  let (initial_men, initial_women) := initial_conditions x in
  let men_after_entry := initial_men + 2 in
  let women_after_leaving := initial_women - 3 in
  2 * women_after_leaving = 24 →
  men_after_entry = 14 :=
by
  intros
  sorry

end find_number_of_men_l617_617032


namespace alpha_beta_relationship_l617_617813

theorem alpha_beta_relationship (α β : ℝ) (h1 : α ∈ Ioo (π / 2) π) (h2 : β ∈ Ioo (π / 2) π)
  (h3 : (1 - sin (2 * α)) * sin β = cos β * cos (2 * α)) :
  β - α = π / 4 :=
by sorry

end alpha_beta_relationship_l617_617813


namespace differentiable_function_zero_l617_617056

theorem differentiable_function_zero
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_zero : f 0 = 0)
    (h_ineq : ∀ x : ℝ, 0 < |f x| ∧ |f x| < 1/2 → |deriv f x| ≤ |f x * Real.log (|f x|)|) :
    ∀ x : ℝ, f x = 0 :=
by
  sorry

end differentiable_function_zero_l617_617056


namespace right_triangle_area_l617_617402

theorem right_triangle_area (a b : ℕ) (h₁ : a = 45) (h₂ : b = 48) :
  (1 / 2 : ℚ) * a * b = 1080 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end right_triangle_area_l617_617402


namespace poly_coeff_sum_l617_617380

theorem poly_coeff_sum :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ,
  (∀ x : ℤ, ((x^2 + 1) * (x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 + a_11 * x^11))
  ∧ a_0 = -512) →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510) :=
by
  sorry

end poly_coeff_sum_l617_617380


namespace sum_radical_conjugate_l617_617648

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617648


namespace least_possible_diagonals_l617_617617

theorem least_possible_diagonals :
  let n := 2021 in
  let labels := Fin n → ℝ in
  ∃ (d : ℕ), 
    (∀ (x : labels), 
      (∀ (i j : Fin n), 
        (i ≠ j ∧ i.succ ≠ j ∧ j.succ ≠ i ∧ abs (x i - x j) ≤ 1) → 
        (∃ (k l : Fin n), (k ≠ l ∧ abs (x k - x l) ≤ 1))) 
    → d ≥ 4039) 
  ∧ 
    d = 4039 :=
sorry

end least_possible_diagonals_l617_617617


namespace proof_problem_l617_617335

noncomputable def a_n (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d
noncomputable def A_n (n : ℕ) (d : ℝ) : ℝ := n + (n * (n - 1) / 2) * d
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Assume some form of b_n
noncomputable def B_n (n : ℕ) : ℝ := ∑ k in Finset.range n, b_n (k + 1)

theorem proof_problem 
  (a_n : ℕ → ℝ → ℝ)
  (A_n : ℕ → ℝ → ℝ)
  (b_n : ℕ → ℝ)
  (B_n : ℕ → ℝ)
  (d : ℝ)
  (h₁ : ∀ n, a_n n d = 1 + (n - 1) * d) 
  (h₂ : ∀ n, A_n n d = n + (n * (n - 1) / 2) * d)
  (limit1 : Tendsto (λ n, (a_n n d / n + b_n n)) atTop (𝓝 3))
  (limit2 : Tendsto (λ n, (A_n n d / n^2 + B_n n / n)) atTop (𝓝 2)) :
  (∃ l₁ l₂ : ℝ, Tendsto (λ n, b_n n) atTop (𝓝 l₁) ∧ Tendsto (λ n, B_n n / n) atTop (𝓝 l₂)) ∧
  (l₁ = 1 ∧ l₂ = 1) :=
by
  sorry

end proof_problem_l617_617335


namespace john_hourly_rate_with_bonus_l617_617422

theorem john_hourly_rate_with_bonus:
  ∀ (daily_wage : ℝ) (work_hours : ℕ) (bonus : ℝ) (extra_hours : ℕ),
    daily_wage = 80 →
    work_hours = 8 →
    bonus = 20 →
    extra_hours = 2 →
    (daily_wage + bonus) / (work_hours + extra_hours) = 10 :=
by
  intros daily_wage work_hours bonus extra_hours
  intros h1 h2 h3 h4
  -- sorry: the proof is omitted
  sorry

end john_hourly_rate_with_bonus_l617_617422


namespace minimum_value_f_l617_617296

def f (x : ℝ) : ℝ :=
  real.sqrt (x^2 + (x - 2)^2) + real.sqrt ((x - 2)^2 + (x + 2)^2)

theorem minimum_value_f : ∃ x_min : ℝ, f x_min = 2 * real.sqrt 5 :=
by
  sorry

end minimum_value_f_l617_617296


namespace instantaneous_velocity_at_3_l617_617128

def position_function (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_3 :
  let velocity_function := (λ t, deriv position_function t)
  velocity_function 3 = 5 :=
by
  let velocity_function := (λ t, deriv position_function t)
  have deriv_s : ∀ t, deriv position_function t = -1 + 2 * t := sorry
  rw deriv_s 3
  norm_num

end instantaneous_velocity_at_3_l617_617128


namespace unique_integer_property_l617_617606

theorem unique_integer_property (x : ℝ) (h : ∃! y ∈ {x - sqrt 2, x - (1/x), x + (1/x), x^2 + 2 * sqrt 2}, ¬ ∃ (k : ℤ), y = k) : x = sqrt 2 - 1 :=
sorry

end unique_integer_property_l617_617606


namespace exists_sequence_with_intersection_cardinality_l617_617423

def sequence (n : Nat) := Vector (ℤ) n

def componentwise_multiplication {n : Nat} (a b : sequence n) : sequence n :=
  Vector.map₂ (*) a b

theorem exists_sequence_with_intersection_cardinality
  {k n : Nat}
  (B : Finset (sequence n))
  (h_card_B : B.card = k)
  (h_seq : ∀ b ∈ B, ∀ i, b[i] = 1 ∨ b[i] = -1) :
  ∃ (c : sequence n), (B ∩ (B.image (componentwise_multiplication c))).card ≤ (k * k) / (2^n) := 
sorry

end exists_sequence_with_intersection_cardinality_l617_617423


namespace Shaina_chocolate_l617_617868

-- Definitions based on the conditions
def total_chocolate : ℚ := 72 / 7
def number_of_piles : ℚ := 6
def weight_per_pile : ℚ := total_chocolate / number_of_piles
def piles_given_to_Shaina : ℚ := 2

-- Theorem stating the problem's correct answer
theorem Shaina_chocolate :
  piles_given_to_Shaina * weight_per_pile = 24 / 7 :=
by
  sorry

end Shaina_chocolate_l617_617868


namespace coffee_table_price_correct_l617_617047

-- Conditions
def sofa_cost : ℕ := 1250
def armchair_cost_each : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Question: What is the price of the coffee table?
def coffee_table_price : ℕ := total_invoice - (sofa_cost + num_armchairs * armchair_cost_each)

-- Proof statement (to be completed)
theorem coffee_table_price_correct : coffee_table_price = 330 := by
  sorry

end coffee_table_price_correct_l617_617047


namespace sum_of_number_and_conjugate_l617_617667

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617667


namespace perimeter_of_triangle_DEF_l617_617502

-- Defining the conditions as Lean definitions and propositions
def tangent_at_P (D E F P : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace P] :=
  dist D P + dist P E = dist D E

def radius (r : ℝ) := r = 15

def DP_eq_19 (D P : Type) [MetricSpace D] [MetricSpace P] := dist D P = 19

def PE_eq_31 (P E : Type) [MetricSpace P] [MetricSpace E] := dist P E = 31

-- Stating the theorem to prove the perimeter of triangle DEF
theorem perimeter_of_triangle_DEF
  (D E F P : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace P]
  (h1 : tangent_at_P D E F P)
  (h2 : radius 15)
  (h3 : DP_eq_19 D P)
  (h4 : PE_eq_31 P E) :
  let s := 50 + (11125 / 364),
  let perimeter := 2 * s
  in perimeter = (29325 / 182) :=
sorry

end perimeter_of_triangle_DEF_l617_617502


namespace spinner_product_probability_l617_617484

theorem spinner_product_probability :
  let C_numbers := {1, 2, 3, 4, 5}
  let D_numbers := {1, 2, 3, 4}
  let total_outcomes := 5 * 4
  let odd_numbers_C := {1, 3, 5}
  let odd_numbers_D := {1, 3}
  let odd_outcomes := Set.card odd_numbers_C * Set.card odd_numbers_D
  odd_outcomes / total_outcomes = 3 / 10 :=
sorry

end spinner_product_probability_l617_617484


namespace sum_of_radii_l617_617735

noncomputable def total_perimeter (r1 r2 r3 : ℝ) : ℝ :=
  π * (r1 + r2 + r3)

theorem sum_of_radii (r1 r2 r3 : ℝ) (h : total_perimeter r1 r2 r3 = 108) : r1 + r2 + r3 = 108 / π :=
by
  sorry

end sum_of_radii_l617_617735


namespace f_increasing_and_odd_l617_617498

def f (x : ℝ) : ℝ :=
  if x >= 0 then 1 - 5^(-x) else 5^x - 1

theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end f_increasing_and_odd_l617_617498


namespace percentage_grape_juice_is_27_78_l617_617594

def mixture_A_gallons : ℝ := 15
def mixture_B_gallons : ℝ := 40
def mixture_C_gallons : ℝ := 25
def pure_grape_juice_gallons : ℝ := 10

def grape_juice_percentage_A : ℝ := 0.30
def grape_juice_percentage_B : ℝ := 0.20
def grape_juice_percentage_C : ℝ := 0.10

def total_grape_juice_gallons : ℝ :=
  mixture_A_gallons * grape_juice_percentage_A + 
  mixture_B_gallons * grape_juice_percentage_B + 
  mixture_C_gallons * grape_juice_percentage_C + 
  pure_grape_juice_gallons

def total_mixture_gallons : ℝ :=
  mixture_A_gallons + mixture_B_gallons + mixture_C_gallons + pure_grape_juice_gallons

def percentage_grape_juice : ℝ :=
  (total_grape_juice_gallons / total_mixture_gallons) * 100

theorem percentage_grape_juice_is_27_78 :
  abs (percentage_grape_juice - 27.78) < 1e-2 :=
by
  sorry

end percentage_grape_juice_is_27_78_l617_617594


namespace proof_problem_l617_617855

open Classical

-- Define conditions and corresponding propositions
def condition_1 (x y : ℝ) : Prop := x > y → x^2 > y^2
def condition_2 (x : ℝ) : Prop := x > 10 → x > 5
def condition_3 (a b c : ℝ) : Prop := a = b → ac = bc
def condition_4 (a b c d : ℝ) : Prop := a - c > b - d → (a > b ∧ c < d)

-- Define sufficient but not necessary condition property
def sufficient_but_not_necessary (p q : Prop) : Prop := (p → q) ∧ ¬(q → p)

-- Statement to prove the correct options
theorem proof_problem (x y a b c d : ℝ) :
  (sufficient_but_not_necessary (x > 10) (x > 5)) ∧ (sufficient_but_not_necessary (a = b) (ac = bc)) :=
by sorry

end proof_problem_l617_617855


namespace find_matrix_N_l617_617711

noncomputable def matrix_inverse (M : Matrix (Fin 4) (Fin 4) ℚ) : Matrix (Fin 4) (Fin 4) ℚ := 
  Matrix.inv M

theorem find_matrix_N : 
    ∃ N : Matrix (Fin 4) (Fin 4) ℚ, 
      N ⬝ (λ i j, if (i, j) = (0, 0) then -2 else if (i, j) = (0, 1) then 5 else if (i, j) = (1, 0) then 3 else if (i, j) = (1, 1) then -8 else if (i, j) = (2, 2) then 4 else if (i, j) = (2, 3) then -5 else if (i, j) = (3, 2) then 6 else if (i, j) = (3, 3) then -7 else 0) 
      = 1 := 
  sorry

end find_matrix_N_l617_617711


namespace Krishan_has_4046_l617_617138

variable (Ram Gopal Krishan : ℕ) -- Define the variables

-- Conditions given in the problem
axiom ratio_Ram_Gopal : Ram * 17 = Gopal * 7
axiom ratio_Gopal_Krishan : Gopal * 17 = Krishan * 7
axiom Ram_value : Ram = 686

-- This is the goal to prove
theorem Krishan_has_4046 : Krishan = 4046 :=
by
  -- Here is where the proof would go
  sorry

end Krishan_has_4046_l617_617138


namespace base7_and_base13_addition_l617_617290

def base7_to_nat (a b c : ℕ) : ℕ := a * 49 + b * 7 + c

def base13_to_nat (a b c : ℕ) : ℕ := a * 169 + b * 13 + c

theorem base7_and_base13_addition (a b c d e f : ℕ) :
  a = 5 → b = 3 → c = 6 → d = 4 → e = 12 → f = 5 →
  base7_to_nat a b c + base13_to_nat d e f = 1109 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  unfold base7_to_nat base13_to_nat
  sorry

end base7_and_base13_addition_l617_617290


namespace clock_angle_2_30_l617_617558

theorem clock_angle_2_30 : 
    let minute_hand_position := 180 
    let hour_hand_position := 75
    abs (minute_hand_position - hour_hand_position) = 105 := by
    sorry

end clock_angle_2_30_l617_617558


namespace cos_square_sum_eq_91_l617_617639

theorem cos_square_sum_eq_91 (S : ℝ) 
  (h1 : ∀ θ : ℝ, θ ∈ set.Icc 0 180 → Real.cos (θ * Real.pi / 180) = Real.cos ((180 - θ) * Real.pi / 180)) 
  (h2 : S = ∑ θ in range 91, (Real.cos (θ * Real.pi / 180))^2 = 91 / 2) : 
  (∑ θ in range 181, (Real.cos (θ * Real.pi / 180))^2) = 91 :=
by {
  sorry
}

end cos_square_sum_eq_91_l617_617639


namespace investment_scientific_notation_l617_617488

def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ (1650000000 = a * 10^n)

theorem investment_scientific_notation :
  ∃ a n, is_scientific_notation a n ∧ a = 1.65 ∧ n = 9 :=
sorry

end investment_scientific_notation_l617_617488


namespace angle_half_second_quadrant_l617_617814

theorem angle_half_second_quadrant (α : ℝ) (k : ℤ) :
  (π / 2 + 2 * k * π < α ∧ α < π + 2 * k * π) → 
  (∃ m : ℤ, (π / 4 + m * π < α / 2 ∧ α / 2 < π / 2 + m * π)) ∨ 
  (∃ n : ℤ, (5 * π / 4 + n * π < α / 2 ∧ α / 2 < 3 * π / 2 + n * π)) :=
by
  sorry

end angle_half_second_quadrant_l617_617814


namespace tangent_proof_l617_617921

open Point Geometry Circle

noncomputable def parallelogram_proof (A B C D E F : Point) : Prop :=
  let parallelogram := isParallelogram A B C D in
  let E_on_BC := isOnLineSegment E B C in
  let F_on_AD := isOnLineSegment F A D in
  let circumcircle_abe := circumcircle A B E in
  let tangent_cf := isTangent circumcircle_abe (lineSegment C F) in
  let circumcircle_cdf := circumcircle C D F in
  let tangent_ae := isTangent circumcircle_cdf (lineSegment A E) in
  parallelogram ∧ E_on_BC ∧ F_on_AD ∧ tangent_cf → tangent_ae

theorem tangent_proof (A B C D E F : Point) (h : parallelogram_proof A B C D E F) :
  let circumcircle_cdf := circumcircle C D F in
  let tangent_ae := isTangent circumcircle_cdf (lineSegment A E) in
  tangent_ae :=
by
  sorry

end tangent_proof_l617_617921


namespace perimeter_of_ABCD_l617_617850

theorem perimeter_of_ABCD
  (AD BC AB CD : ℕ)
  (hAD : AD = 4)
  (hAB : AB = 5)
  (hBC : BC = 10)
  (hCD : CD = 7)
  (hAD_lt_BC : AD < BC) :
  AD + AB + BC + CD = 26 :=
by
  -- Proof will be provided here.
  sorry

end perimeter_of_ABCD_l617_617850


namespace solve_a_value_l617_617487

theorem solve_a_value (a b k : ℝ) 
  (h1 : a^3 * b^2 = k)
  (h2 : a = 5)
  (h3 : b = 2) :
  ∃ a', b = 8 → a' = 2.5 :=
by
  sorry

end solve_a_value_l617_617487


namespace factorial_simplification_l617_617932

theorem factorial_simplification : (13.factorial / (10.factorial + 3 * 9.factorial)) = 464 := by
  sorry

end factorial_simplification_l617_617932


namespace fifteenth_odd_multiple_of_five_l617_617199

theorem fifteenth_odd_multiple_of_five :
  ∃ a : ℕ, (∀ n : ℕ, a n = 5 + (n - 1) * 10) ∧ a 15 = 145 :=
by
  let a := λ n, 5 + (n - 1) * 10
  use a
  split
  { intros n,
    refl }
  { refl }
  sorry

end fifteenth_odd_multiple_of_five_l617_617199


namespace range_of_a_l617_617785

def g (x : ℝ) : ℝ := (2 - x) * exp x
def h (x : ℝ) (a : ℝ) : ℝ := a * x + a
def f (x : ℝ) (a : ℝ) : ℝ := g x - h x a

theorem range_of_a (a : ℝ) : 
  (∀ x : ℕ, (0 < x ∧ f x a > 0) → x ∈ {1, 2}) → a ∈ Ioo (- (1 / 4) * exp 3) 0 :=
by 
  sorry

end range_of_a_l617_617785


namespace planes_parallel_normal_eq_l617_617389

theorem planes_parallel_normal_eq 
  (x : ℝ) 
  (a : ℝ × ℝ × ℝ := (-1, 2, 4)) 
  (b : ℝ × ℝ × ℝ := (x, -1, -2)) 
  (h₁ : ∃ λ : ℝ, a = (λ * x, -λ, -2 * λ)) 
  : x = 1 / 2 :=
sorry

end planes_parallel_normal_eq_l617_617389


namespace monotonically_increasing_interval_l617_617275

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / Real.exp 1 → (Real.log x + 1) > 0 :=
by
  intros x hx
  sorry

end monotonically_increasing_interval_l617_617275


namespace total_votes_l617_617840

-- Define the conditions
variables (V : ℝ) (votes_second_candidate : ℝ) (percent_second_candidate : ℝ)
variables (h1 : votes_second_candidate = 240)
variables (h2 : percent_second_candidate = 0.30)

-- Statement: The total number of votes is 800 given the conditions.
theorem total_votes (h : percent_second_candidate * V = votes_second_candidate) : V = 800 :=
sorry

end total_votes_l617_617840


namespace factorize_a_square_minus_1_simplify_expression_l617_617586

variable (a : ℕ)

theorem factorize_a_square_minus_1 :
  a^2 - 1 = (a - 1) * (a + 1) := sorry

theorem simplify_expression (h : a^2 - 1 = (a - 1) * (a + 1)) :
  (a ≠ 1 ∧ a ≠ -1) → (a - 1) / (a^2 - 1) + 1 / (a + 1) = 2 / (a + 1) :=
by
  intro hne
  have h1 : (a - 1) / ((a - 1) * (a + 1)) = 1 / (a + 1) := sorry
  have h2 : 1 / (a + 1) + 1 / (a + 1) = 2 / (a + 1) := sorry
  exact h2

end factorize_a_square_minus_1_simplify_expression_l617_617586


namespace valid_number_count_l617_617170

def count_valid_numbers : ℕ :=
  let digits := {1, 2, 3, 4, 5}
  let even_digits := {2, 4}
  let valid_first_digits := {2, 3, 4, 5}
  -- function to count valid numbers given first and last digit are fixed
  let count_for_first_and_last (first last : ℕ) : ℕ :=
    if first ∉ digits ∨ last ∉ even_digits then 0
    else let remaining_digits := digits \ {first, last}
         -- permutations of 3 remaining digits to fill 3 middle positions
         Nat.factorial (Set.card remaining_digits)
  -- summing over all valid combinations of first and last digits
  let count := valid_first_digits.fold (λ acc first =>
    acc + even_digits.fold (λ acc' last =>
      acc' + count_for_first_and_last first last) 0
  ) 0
  count

theorem valid_number_count :
    count_valid_numbers = 36 :=
by sorry

end valid_number_count_l617_617170


namespace correct_statements_l617_617262

/-- The binomial expression (x - 1) ^ 2009 --/
def binom_expr (x : ℕ) : ℕ := (x - 1) ^ 2009

/-- Sum of coefficients of non-constant terms in the binomial expression (x - 1) ^ 2009 is 1 --/
axiom condition_1 : binom_expr 1 - 1 = 1

/-- The fifth term in the expansion of (x - 1) ^ 2009 is -C(2009, 5) * x^2004 --/
axiom condition_2 : false -- This will be explicitly regarded as false as per the solution

/-- The term with the highest coefficient in this binomial is the 1005th term --/
axiom condition_3 : true

/-- When x = 2009, the remainder of (x - 1) ^ 2009 divided by 2009 is 2008 --/
axiom condition_4 : (binom_expr 2009) % 2009 = 2008

/-- Among conditions 1 to 4, the correct ones are 3 and 4 --/
theorem correct_statements : (condition_3 ∧ condition_4) ∧ ¬condition_1 ∧ ¬condition_2 :=
by sorry

end correct_statements_l617_617262


namespace radical_conjugate_sum_l617_617689

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617689


namespace max_distinct_triangles_formed_l617_617155

def lengths : List ℝ := [3, 5, 6, 9, 10]

def triangle_condition (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_distinct_triangles_formed :
  (Finset.powerset lengths.toFinset).filter (λ s, s.card = 3 ∧ ∃ a b c, s = {a, b, c} ∧ triangle_condition a b c).card = 6 := 
sorry

end max_distinct_triangles_formed_l617_617155


namespace find_x_value_l617_617372

def perp_vectors (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x_value :
  (a : ℝ × ℝ) (b : ℝ × ℝ) (x : ℝ), a = (3, -1) → b = (1, x) → perp_vectors a b → x = 3 :=
by
  intros a b x ha hb h_perp
  have ha_eq : a = (3, -1) := ha
  have hb_eq : b = (1, x) := hb
  rw [perp_vectors, ha_eq, hb_eq] at h_perp
  sorry

end find_x_value_l617_617372


namespace area_of_rectangle_is_35_l617_617503

noncomputable def side_of_square := Real.sqrt 784
def radius_of_circle := side_of_square
def length_of_rectangle := radius_of_circle / 4
def breadth_of_rectangle := 5
def area_of_rectangle := length_of_rectangle * breadth_of_rectangle

theorem area_of_rectangle_is_35 :
  area_of_rectangle = 35 := 
by 
  -- Proving the area of the rectangle
  sorry

end area_of_rectangle_is_35_l617_617503


namespace num_ways_to_sum_121_with_fib_l617_617954

def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n+2) := fibonacci (n+1) + fibonacci n

def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fibonacci k = n

def is_sum_of_distinct_fibonacci_numbers (target : ℕ) (nums : List ℕ) : Prop :=
  nums.nodup ∧ nums.all isFibonacci ∧ nums.sum = target

-- The statement of the proof problem
theorem num_ways_to_sum_121_with_fib : 
  ∃ sols : Finset (List ℕ), 
  (∀ l ∈ sols, is_sum_of_distinct_fibonacci_numbers 121 l) ∧ sols.card = 8 := 
sorry

end num_ways_to_sum_121_with_fib_l617_617954


namespace price_of_coffee_table_l617_617050

-- Define the given values
def price_sofa : ℕ := 1250
def price_armchair : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Define the target value (price of the coffee table)
def price_coffee_table : ℕ := 330

-- The theorem to prove
theorem price_of_coffee_table :
  total_invoice = price_sofa + num_armchairs * price_armchair + price_coffee_table :=
by sorry

end price_of_coffee_table_l617_617050


namespace molecular_weight_of_aluminum_part_in_Al2_CO3_3_l617_617297

def total_molecular_weight_Al2_CO3_3 : ℝ := 234
def atomic_weight_Al : ℝ := 26.98
def num_atoms_Al_in_Al2_CO3_3 : ℕ := 2

theorem molecular_weight_of_aluminum_part_in_Al2_CO3_3 :
  num_atoms_Al_in_Al2_CO3_3 * atomic_weight_Al = 53.96 :=
by
  sorry

end molecular_weight_of_aluminum_part_in_Al2_CO3_3_l617_617297


namespace cards_ratio_l617_617421

variable (x : ℕ)

def partially_full_decks_cards := 3 * x
def full_decks_cards := 3 * 52
def total_cards_before := 200 + 34

theorem cards_ratio (h : 3 * x + full_decks_cards = total_cards_before) : x / 52 = 1 / 2 :=
by sorry

end cards_ratio_l617_617421


namespace line_through_intersections_l617_617497

-- Conditions
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem statement
theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → x - y - 3 = 0 :=
by
  sorry

end line_through_intersections_l617_617497


namespace dropouts_correct_l617_617160

/-- Definition for initial racers, racers joining after 20 minutes, and racers at finish line. -/
def initial_racers : ℕ := 50
def joining_racers : ℕ := 30
def finishers : ℕ := 130

/-- Total racers after initial join and doubling. -/
def total_racers : ℕ := (initial_racers + joining_racers) * 2

/-- The number of people who dropped out before finishing the race. -/
def dropped_out : ℕ := total_racers - finishers

/-- Proof statement to show the number of people who dropped out before finishing is 30. -/
theorem dropouts_correct : dropped_out = 30 := by
  sorry

end dropouts_correct_l617_617160


namespace halfway_fraction_l617_617182

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l617_617182


namespace minimal_diagonals_l617_617622

def consecutive_labels (labels : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i < n - 1 → |labels (i + 1) - labels i| ≤ 1

def draw_diagonal (labels : ℕ → ℝ) (n : ℕ) (i j : ℕ) : Prop :=
  i ≠ j ∧ |labels i - labels j| ≤ 1

def count_diagonals (labels : ℕ → ℝ) (n : ℕ) : ℕ :=
  ((n.choose 2) - n) 

theorem minimal_diagonals (labels : ℕ → ℝ) (n : ℕ) (h1 : consecutive_labels labels n) (h2 : n = 2021) :
  count_diagonals labels n = 2018 :=
sorry

end minimal_diagonals_l617_617622


namespace precipitation_forecast_correct_l617_617031
-- Importing the whole Mathlib to bring in necessary libraries

-- Define the problem and conditions
def precipitation_probability {α : Type} (forecast : α → ℝ) : Prop :=
  ∃ (p : ℝ), p = 0.78 ∧ forecast p = "The possibility of precipitation in the area tomorrow is 78%"

-- Theorem statement based on the conditions and correct answer
theorem precipitation_forecast_correct (p : ℝ) (forecast : ℝ → String) (h : p = 0.78) 
  (h_forecast : forecast p = "The possibility of precipitation in the area tomorrow is 78%") :
  precipitation_probability forecast :=
sorry

end precipitation_forecast_correct_l617_617031


namespace problem_solution_l617_617783

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2^x + 2^(a*x + b)

theorem problem_solution :
  (∃ a b : ℝ, 
      f 1 a b = 5/2 ∧ 
      f 2 a b = 17/4 ∧ 
      a = -1 ∧ 
      b = 0) ∧
  (∀ x : ℝ, f x (-1) 0 = f (-x) (-1) 0) ∧
  (∀ x1 x2 : ℝ, 
      x1 < x2 ∧ x1 ≤ 0 ∧ x2 ≤ 0 →
      f x1 (-1) 0 > f x2 (-1) 0) ∧
  (∀ x : ℝ, 
      f x (-1) 0 ≥ 2 ∧ 
      (∀ y : ℝ, f y (-1) 0 = 2 → y = 0)) :=
begin
  sorry
end

end problem_solution_l617_617783


namespace part1_part2_l617_617702

-- Definition of the branches of the hyperbola
def C1 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1
def C2 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1

-- Problem Part 1: Proving that P, Q, and R cannot lie on the same branch
theorem part1 (P Q R : ℝ × ℝ) (hP : C1 P) (hQ : C1 Q) (hR : C1 R) : False := by
  sorry

-- Problem Part 2: Finding the coordinates of Q and R
theorem part2 : 
  ∃ Q R : ℝ × ℝ, C1 Q ∧ C1 R ∧ 
                (Q = (2 - Real.sqrt 3, 1 / (2 - Real.sqrt 3))) ∧ 
                (R = (2 + Real.sqrt 3, 1 / (2 + Real.sqrt 3))) := 
by
  sorry

end part1_part2_l617_617702


namespace solve_inequalities_l617_617107

theorem solve_inequalities :
  {x : ℤ | (x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - 5 < -3 * x} = {-1, 0} :=
by
  sorry

end solve_inequalities_l617_617107


namespace all_people_end_at_1011_l617_617159

theorem all_people_end_at_1011 :
  ∀ (initial_positions : Fin 2023 → ℕ), 
  (∀ i, initial_positions i = i) →
  ∃ (final_positions : Fin 2023 → ℕ), 
  (∀ i, final_positions i = 1011) ∧
  all_moves_valid initial_positions final_positions :=
begin
  -- The proof is to be filled in
  sorry
end

noncomputable def all_moves_valid (initial_positions final_positions : Fin 2023 → ℕ) : Prop := sorry

end all_people_end_at_1011_l617_617159


namespace find_k_l617_617776

theorem find_k (k : ℝ) :
  (∃ x y : ℝ, y = x + 2 * k ∧ y = 2 * x + k + 1 ∧ x^2 + y^2 = 4) ↔
  (k = 1 ∨ k = -1/5) := 
sorry

end find_k_l617_617776


namespace maximum_n_l617_617889

variable (x y z : ℝ)

theorem maximum_n (h1 : x + y + z = 12) (h2 : x * y + y * z + z * x = 30) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
by
  sorry

end maximum_n_l617_617889


namespace sum_of_number_and_its_radical_conjugate_l617_617663

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617663


namespace proj_3_2_9_on_Q_l617_617431

def projection (u p : ℝ^3) : ℝ^3 :=
  let n : ℝ^3 := ⟨1, -1, 2⟩
  in u - ((u • n) / (n • n)) • n

theorem proj_3_2_9_on_Q :
  let Q := plane (orig : ℝ^3) (norm : ℝ^3) := 
    { point := λ x, x • norm = 0 } in
  let u : ℝ^3 := ⟨3, 2, 9⟩ in
  let p := projection u ⟨1, -1, 2⟩ in
  p = ⟨1 / 6, 31 / 6, 8 / 3⟩ :=
sorry

end proj_3_2_9_on_Q_l617_617431


namespace sin_cos_of_tan_is_two_l617_617752

theorem sin_cos_of_tan_is_two (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 :=
sorry

end sin_cos_of_tan_is_two_l617_617752


namespace arithmetic_sequence_nth_term_639_l617_617955

theorem arithmetic_sequence_nth_term_639 :
  ∀ (x n : ℕ) (a₁ a₂ a₃ aₙ : ℤ),
  a₁ = 3 * x - 5 →
  a₂ = 7 * x - 17 →
  a₃ = 4 * x + 3 →
  aₙ = a₁ + (n - 1) * (a₂ - a₁) →
  aₙ = 4018 →
  n = 639 :=
by
  intros x n a₁ a₂ a₃ aₙ h₁ h₂ h₃ hₙ hₙ_eq
  sorry

end arithmetic_sequence_nth_term_639_l617_617955


namespace sum_of_conjugates_l617_617652

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l617_617652


namespace find_angle_bisector_length_l617_617861

-- Define the problem context
variable (A B C D : Type) [Triangle ABC] 
variable (AC BC : ℝ) (angle_C : ℝ) (angle_bisector_CD : ℝ)

-- Specify given conditions
axiom AC_eq_6 : AC = 6
axiom BC_eq_9 : BC = 9
axiom angle_C_eq_120 : angle_C = 120

-- The theorem to prove the problem statement
theorem find_angle_bisector_length : angle_bisector_CD = 18 / 5 :=
by
    have h1 : AC = 6 := AC_eq_6
    have h2 : BC = 9 := BC_eq_9
    have h3 : angle_C = 120 := angle_C_eq_120
    sorry

end find_angle_bisector_length_l617_617861


namespace quadrilateral_with_isosceles_diagonals_not_necessarily_rhombus_l617_617281

theorem quadrilateral_with_isosceles_diagonals_not_necessarily_rhombus (Q : Type) [quadrilateral Q] :
  (∀ d1 d2 : diagonal Q, is_isosceles_triangle Q (d1) ∧ is_isosceles_triangle Q (d2)) → ¬is_rhombus Q := by
  sorry

end quadrilateral_with_isosceles_diagonals_not_necessarily_rhombus_l617_617281


namespace probability_returns_to_origin_l617_617026

def probability_back_at_origin (k : ℕ) (h : k > 0): ℝ :=
  1/4 + 3/4 * (1/9)^k

theorem probability_returns_to_origin (k : ℕ) (h : k > 0) :
  probability_back_at_origin k h = 1/4 + 3/4 * (1/9)^k :=
sorry

end probability_returns_to_origin_l617_617026


namespace inequality_solution_l617_617480

/--
Let f(x) = 2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) - 1 / 15.
Prove that f(x) < 0 if and only if x ∈ (1, 2) ∪ (3, 4) ∪ (6, 8).
-/
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) - 1 / 15

theorem inequality_solution (x : ℝ) :
  f x < 0 ↔ (1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (6 < x ∧ x < 8) :=
begin
  sorry
end

end inequality_solution_l617_617480


namespace find_mistake_l617_617996

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l617_617996


namespace gcd_456_357_l617_617167

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  sorry

end gcd_456_357_l617_617167


namespace train_pass_bridge_time_l617_617578

theorem train_pass_bridge_time (l_train l_bridge : ℕ) (speed_kmh : ℕ) (speed_conversion_factor : ℤ) :
  l_train = 360 → l_bridge = 140 → speed_kmh = 52 → speed_conversion_factor = 1000 / 3600 →
  let total_distance := l_train + l_bridge,
      speed_ms := speed_kmh * speed_conversion_factor,
      time := total_distance / speed_ms
  in time ≈ 34.64 :=
by
  sorry

end train_pass_bridge_time_l617_617578


namespace complex_point_location_l617_617325

open Complex

noncomputable def quadrant (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "on the axis"

theorem complex_point_location :
  ∀ (z : ℂ), (z - 2 * I) * (1 + I) = abs (1 - sqrt 3 * I) → quadrant z = "first quadrant" :=
by
  intros z h
  have h₁ : abs (1 - sqrt 3 * I) = 2 := by sorry
  have h₂ : (z - 2 * I) * (1 + I) = 2 := by rw [h, h₁]
  have h₃ : z = 1 + I := by sorry
  rw [h₃]
  unfold quadrant
  simp

end complex_point_location_l617_617325


namespace clock_angle_2_30_l617_617557

theorem clock_angle_2_30 : 
    let minute_hand_position := 180 
    let hour_hand_position := 75
    abs (minute_hand_position - hour_hand_position) = 105 := by
    sorry

end clock_angle_2_30_l617_617557


namespace range_of_a_l617_617799

def p (x : ℝ) : Prop := (2 * x) / (x - 1) < 1
def q (x a : ℝ) : Prop := (x + a) * (x - 3) > 0

theorem range_of_a :
  (∀ x : ℝ, p x → q x a) ∧ ¬ (∀ x : ℝ, q x a → p x) →
  a ∈ set.Iic (-1) :=
by
  sorry

end range_of_a_l617_617799


namespace find_TU2_l617_617844

-- Define the structure of the square, distances, and points
structure square (P Q R S T U : Type) :=
(PQ : ℝ)
(PT QU QT RU TU2 : ℝ)
(h1 : PQ = 15)
(h2 : PT = 7)
(h3 : QU = 7)
(h4 : QT = 17)
(h5 : RU = 17)
(h6 : TU2 = TU^2)
(h7 : TU2 = 1073)

-- The main proof statement
theorem find_TU2 {P Q R S T U : Type} (sq : square P Q R S T U) : sq.TU2 = 1073 := by
  sorry

end find_TU2_l617_617844


namespace find_B_l617_617435

theorem find_B (B: ℕ) (h1: 5457062 % 2 = 0 ∧ 200 * B % 4 = 0) (h2: 5457062 % 5 = 0 ∧ B % 5 = 0) (h3: 5450062 % 8 = 0 ∧ 100 * B % 8 = 0) : B = 0 :=
sorry

end find_B_l617_617435


namespace point_D_on_line_AC_l617_617773

variables {A B C O D : Point}
variables {α β γ : ℝ}

-- Conditions:
-- 1. Points A, B, and C lie on the circle O
def on_circle (P : Point) (O : Circle) := O.contains P

-- 2. ∠ ABC > 90°
def angle_ABC_gt_90: Prop := angle A B C > π / 2

-- 3. The angle bisector of ∠ AOB intersects the circumcircle of Δ BOC at point D
def is_angle_bisector (A O B D : Point) := angle A O D = angle D O B

-- 4. Prove that point D lies on line AC
def lies_on_line (P : Point) (A C : Point) := collinear P A C

-- Lean statement to be proven
theorem point_D_on_line_AC
  (h_on_circle : on_circle A O ∧ on_circle B O ∧ on_circle C O)
  (h_angle_gt_90 : angle_ABC_gt_90)
  (h_bisector : is_angle_bisector A O B D)
  (h_on_circle_boc : on_circle D (circumcircle B C O)) :
  lies_on_line D A C := 
begin
  sorry,
end

end point_D_on_line_AC_l617_617773


namespace perpendicular_bisectors_parallel_or_concurrent_l617_617455

variables {Z1 Z2 Z3 Z1' Z2' Z3' : Type*}

-- Definition of congruent triangles
def congruent_triangles (a b c a' b' c' : ℝ) : Prop :=
  |a - b| = |a' - b'| ∧ |a - c| = |a' - c'| ∧ (a - b) / (a - c) = (a' - b') / (a' - c')

-- Theorem statement
theorem perpendicular_bisectors_parallel_or_concurrent
  {a b c a' b' c' : ℝ}
  (h : congruent_triangles a b c a' b' c') :
  parallel_or_concurrent (perpendicular_bisector a a') 
                         (perpendicular_bisector b b') 
                         (perpendicular_bisector c c') := 
sorry

end perpendicular_bisectors_parallel_or_concurrent_l617_617455


namespace sum_of_number_and_its_radical_conjugate_l617_617658

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l617_617658


namespace polynomial_evaluation_l617_617798

theorem polynomial_evaluation :
  (5 * 3^3 - 3 * 3^2 + 7 * 3 - 2 = 127) :=
by
  sorry

end polynomial_evaluation_l617_617798


namespace find_k_range_l617_617344

theorem find_k_range (k : ℝ) (a b c : ℝ) (h₁ : a = k) (h₂ : b = k + 2) (h₃ : c = k + 4) (h_triangle : ∠C > π / 2) : 2 < k ∧ k < 6 :=
by
  sorry

end find_k_range_l617_617344


namespace find_p_zero_l617_617076

-- Condition: p is a polynomial of degree 6
def p (x : ℝ) : ℝ

noncomputable def p_deg : p.degree = 6 := sorry

-- Condition: p(3^n) = 1 / 3^n for n = 0, 1, ..., 6
def condition (n : ℕ) (hn : n ≤ 6) : p (3^n) = 1 / 3^n := sorry

-- Statement of the problem
theorem find_p_zero : p(0) = 91 / 32 := sorry

end find_p_zero_l617_617076


namespace sum_of_number_and_radical_conjugate_l617_617678

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617678


namespace markup_correct_l617_617247

-- Define the conditions as variables
variable (S : ℝ) (profit_percentage : ℝ) (expenses_percentage : ℝ) (selling_price : ℝ)

-- The conditions given in the problem
def profit_amount := profit_percentage * selling_price
def expenses_amount := expenses_percentage * selling_price
def cost := selling_price - profit_amount - expenses_amount

-- The rate of markup to be proven
def markup_rate := (selling_price - cost) / cost * 100

-- Values instantiated
axiom selling_price_value : selling_price = 10
axiom profit_percentage_value : profit_percentage = 0.20
axiom expenses_percentage_value : expenses_percentage = 0.10

theorem markup_correct :
  markup_rate selling_price profit_percentage expenses_percentage selling_price_value profit_percentage_value expenses_percentage_value = 42.857 :=
by
  sorry

end markup_correct_l617_617247


namespace monomial_exponents_l617_617388

theorem monomial_exponents (m n : ℤ) (h₁ : m + 2 * n = 3) (h₂ : n - 2 * m = 4) : m ^ n = 1 := by
  sorry

end monomial_exponents_l617_617388


namespace students_in_all_three_courses_zero_l617_617150

noncomputable theory
open_locale classical

def total_students := 24
def chess_students := 14
def debate_students := 15
def science_students := 10
def at_least_two_courses_students := 12

variable (a b d : ℕ)
def students_taking_exactly_two_courses := a + b + d
def students_taking_all_three_courses (c : ℕ) := c

theorem students_in_all_three_courses_zero :
  ∃ (c : ℕ), students_taking_exactly_two_courses a b d + students_taking_all_three_courses c = at_least_two_courses_students ∧ c = 0 :=
sorry

end students_in_all_three_courses_zero_l617_617150


namespace treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l617_617132

namespace PirateTreasure

-- Given conditions
def num_pirates_excl_captain := 100
def max_coins := 1000
def remaining_coins_99_pirates := 51
def remaining_coins_77_pirates := 29

-- Problem Part (a): Prove the number of coins in treasure
theorem treasure_contains_645_coins : 
  ∃ (N : ℕ), N < max_coins ∧ (N % 99 = remaining_coins_99_pirates ∧ N % 77 = remaining_coins_77_pirates) ∧ N = 645 :=
  sorry

-- Problem Part (b): Prove the number of pirates Barbaroxa should choose
theorem max_leftover_coins_when_choosing_93_pirates :
  ∃ (n : ℕ), n ≤ num_pirates_excl_captain ∧ (∀ k, k ≤ num_pirates_excl_captain → (645 % k) ≤ (645 % k) ∧ n = 93) :=
  sorry

end PirateTreasure

end treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l617_617132


namespace abs_diff_probability_l617_617099

noncomputable def probability_abs_diff_gt_half : ℝ :=
1/4 * (0 + 1/2) + 1/4 * 1 + 1/16

theorem abs_diff_probability : probability_abs_diff_gt_half = 9/16 := by
  sorry

end abs_diff_probability_l617_617099


namespace least_possible_diagonals_l617_617619

noncomputable def leastDiagonals : ℝ :=
  let n := 2021 in 2018

theorem least_possible_diagonals (labels : Fin 2021 → ℝ)
  (h1 : ∀ i, abs (labels i - labels ((i + 1) % 2021)) ≤ 1)
  (h2 : ∀ i j, i ≠ j → abs (labels i - labels j) ≤ 1 → is_diagonal i j) :
  leastDiagonals = 2018 :=
sorry

end least_possible_diagonals_l617_617619


namespace angle_bisector_length_l617_617862

open Real
open Complex

-- Definitions for the problem
def side_lengths (AC BC : ℝ) : Prop :=
  AC = 6 ∧ BC = 9

def angle_C (angle : ℝ) : Prop :=
  angle = 120

-- Main statement to prove
theorem angle_bisector_length (AC BC angle x : ℝ)
  (h1 : side_lengths AC BC)
  (h2 : angle_C angle) :
  x = 18 / 5 :=
  sorry

end angle_bisector_length_l617_617862


namespace martin_ball_puzzle_l617_617083

theorem martin_ball_puzzle :
  ∀ (a b : ℕ), a ≤ 100 → b ≤ 100 → (∀ (r1 r2 : ℕ), r1 ∈ set.range (λ n, n + 1) ∧ r2 ∈ set.range (λ n, n + 1) → r1 + r2 ∈ set.range (λ n, n + 101) → ∀ (b : ℕ), b ∈ (finset.range 100).map (λ n, n + 101) → r1 + r2 = b) → 
  (a + b) = 115 := 
sorry

end martin_ball_puzzle_l617_617083


namespace angle_ASB_90_l617_617228

theorem angle_ASB_90 (A B C D P Q R S : Type)
[is_square A B C D]
(P_on_CD : P ∈ segment C D)
(depth_points : P ≠ C ∧ P ≠ D)
[altitude_AQ : is_altitude A B P Q]
[altitude_BR : is_altitude B A P R]
[intersection_S : S = intersection (line_through C Q) (line_through D R)] :
angle A S B = 90 :=
sorry

end angle_ASB_90_l617_617228


namespace triangle_area_is_14_l617_617884

def vector : Type := (ℝ × ℝ)
def a : vector := (4, -1)
def b : vector := (2 * 2, 2 * 3)

noncomputable def parallelogram_area (u v : vector) : ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  abs (ux * vy - uy * vx)

noncomputable def triangle_area (u v : vector) : ℝ :=
  (parallelogram_area u v) / 2

theorem triangle_area_is_14 : triangle_area a b = 14 :=
by
  unfold a b triangle_area parallelogram_area
  sorry

end triangle_area_is_14_l617_617884


namespace prob_neither_alive_l617_617136

/-- Define the probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1 / 4

/-- Define the probability that a wife will be alive for 10 more years -/
def prob_wife_alive : ℚ := 1 / 3

/-- Prove that the probability that neither the man nor his wife will be alive for 10 more years is 1/2 -/
theorem prob_neither_alive (p_man_alive p_wife_alive : ℚ)
    (h1 : p_man_alive = prob_man_alive) (h2 : p_wife_alive = prob_wife_alive) :
    (1 - p_man_alive) * (1 - p_wife_alive) = 1 / 2 :=
by
  sorry

end prob_neither_alive_l617_617136


namespace range_of_a_l617_617888

noncomputable def f (x : ℝ) : ℝ := sorry -- f is some real differentiable function

lemma f_properties (x : ℝ) : f (-x) + f (x) = x^2 := sorry

lemma f'_properties (x : ℝ) (hx : 0 ≤ x) :  f' x > x := sorry

theorem range_of_a (a : ℝ) : (f (2 - a) - f (a) ≥ 2 - 2 * a) ↔ a ∈ set.Iic 1 :=
begin
  sorry
end

end range_of_a_l617_617888


namespace triangle_proof_l617_617078

-- Assuming angles are in radians or degrees doesn't affect arithmetic sequence property
noncomputable def angle_seq (A B C : ℝ) := B = (A + C) / 2

theorem triangle_proof (A B C a b c : ℝ) 
  (H1 : ¬ (a = b ∨ b = c ∨ c = a)) 
  (H2 : angle_seq A B C) 
  (H3 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos B) :
  (1 / (a - b) + 1 / (c - b) = 3 / (a - b + c)) :=
sorry

end triangle_proof_l617_617078


namespace sin_double_angle_of_tan_pi_sub_alpha_eq_two_l617_617382

theorem sin_double_angle_of_tan_pi_sub_alpha_eq_two 
  (α : Real) 
  (h : Real.tan (Real.pi - α) = 2) : 
  Real.sin (2 * α) = -4 / 5 := 
  by sorry

end sin_double_angle_of_tan_pi_sub_alpha_eq_two_l617_617382


namespace largest_x_is_3_l617_617710

noncomputable def largest_value_x (x : ℚ) :=
  (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 8 * x - 2

theorem largest_x_is_3 : ∃ x : ℚ, largest_value_x x ∧ ∀ y : ℚ, largest_value_x y → y ≤ 3 :=
begin
  use 3,
  split,
  { -- Proof part skipped
    sorry
  },
  { -- Proof part skipped
    sorry
  }
end

end largest_x_is_3_l617_617710


namespace range_of_f_l617_617812

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 5) / (x + 3)

-- Define the domain as [1, 4)
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x < 4

theorem range_of_f : set.range (λ x, f x) (set.Ico 1 4) = set.Ico (-3/4 : ℝ) (3/7 : ℝ) := by
  sorry

end range_of_f_l617_617812


namespace interval_bounds_l617_617902

noncomputable def sequence (x0 : ℝ) : ℕ → ℝ
| 0       := x0
| (n + 1) := (5/6) - (4/3) * (| sequence x0 n - (1/2) |)

theorem interval_bounds (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1) :
  (sequence x0 2009) ∈ Icc (7/18 : ℝ) (5/6 : ℝ) :=
sorry

end interval_bounds_l617_617902


namespace no_suitable_f_l617_617055

noncomputable def f (x : ℚ) : ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_pos (a : ℚ) (ha : a ≠ 0) : f a > 0
axiom f_add (x y : ℚ) : f (x + y) = f x * f y
axiom f_max (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) : f (x + y) ≤ max (f x) (f y)
axiom f_not_one {x : ℤ} (hx : f x ≠ 1) : True

theorem no_suitable_f (x : ℤ) (hx : f x ≠ 1) (n : ℕ) : f (1 + x + x^2 + ⋯ + x^n) ≠ 1 := sorry

end no_suitable_f_l617_617055


namespace cosine_of_angle_between_ST_and_QR_l617_617538

noncomputable def cos_between_vectors (PQ PS PR PT : ℝ) (dot_PQ_PS dot_PR_PT : ℝ)
  (Q_midpoint : ℝ -> ℝ -> Prop) 
  (PQ_eq : ℝ) (ST_eq : ℝ) (QR_eq : ℝ) (PR_eq : ℝ) : ℝ :=
let ST := ST_eq in
let QR := QR_eq in
let θ := Math.cos 1 in
angle_between_vectors ST QR θ

theorem cosine_of_angle_between_ST_and_QR :
  (∀ PQ PS PR PT: ℝ,
    ∀ dot_PQ_PS dot_PR_PT: ℝ,
    ∀ PQ_eq: PQ = 2,
    ∀ ST_eq: ST = 2,
    ∀ QR_eq: QR = 8,
    ∀ PR_eq: PR = √72,
    ∀ Q_midpoint: Q_midpoint QS QT ∧ Q_midpoint ST~QR,
    (cos_between_vectors PQ PS PR PT dot_PQ_PS dot_PR_PT Q_midpoint 2 2 8 √72) = 1 ) :=
by
  sorry

end cosine_of_angle_between_ST_and_QR_l617_617538


namespace total_players_is_59_l617_617021

-- Define the number of players from each sport.
def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def softball_players : ℕ := 13

-- Define the total number of players as the sum of the above.
def total_players : ℕ :=
  cricket_players + hockey_players + football_players + softball_players

-- Prove that the total number of players is 59.
theorem total_players_is_59 :
  total_players = 59 :=
by
  unfold total_players
  unfold cricket_players
  unfold hockey_players
  unfold football_players
  unfold softball_players
  sorry

end total_players_is_59_l617_617021


namespace integer_part_sqrt_divides_n_l617_617818

theorem integer_part_sqrt_divides_n (n : ℕ) (hn : 0 < n) (h : ⌊real.sqrt n⌋₊ ∣ n) :
  ∃ (k : ℕ), k ≠ 0 ∧ (n = k^2 ∨ n = k^2 + k ∨ n = k^2 + 2 * k) := 
sorry

end integer_part_sqrt_divides_n_l617_617818


namespace evaluate_expression_l617_617719

theorem evaluate_expression : ∀ (a b c d : ℤ), 
  a = 3 →
  b = a + 3 →
  c = b - 8 →
  d = a + 5 →
  (a + 2 ≠ 0) →
  (b - 4 ≠ 0) →
  (c + 5 ≠ 0) →
  (d - 3 ≠ 0) →
  ((a + 3) * (b - 2) * (c + 9) * (d + 1) = 1512 * (a + 2) * (b - 4) * (c + 5) * (d - 3)) :=
by
  intros a b c d ha hb hc hd ha2 hb4 hc5 hd3
  sorry

end evaluate_expression_l617_617719


namespace find_a_l617_617963

variable (a b c : ℤ)
def vertex_form_eqn := ∀ x : ℤ, y : ℤ, y = a * x^2 + b * x + c → 
  (∃ k : ℤ, y = a * (x + 2)^2 + 3)

theorem find_a (h_vertex : vertex_form_eqn a b c) (h_point : (∃ x y : ℤ, (x, y) = (1, 6) ∧ (∃ k : ℤ, 6 = a * (1 + 2)^2 + 3))) :
a = 1 / 3 :=
sorry

end find_a_l617_617963


namespace ellipse_b_plus_k_l617_617250

theorem ellipse_b_plus_k :
  let f1 := (2, 3)
  let f2 := (2, 7)
  let p := (6, 5)
  ∃ (h k a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    (Set.mem p (SetOf
      (λ (x y : ℝ), (x - h) ^ 2 / a ^ 2 + (y - k) ^ 2 / b ^ 2 = 1)) ∧ 
    b + k = 4 * Real.sqrt 5 + 5 :=
by
  sorry

end ellipse_b_plus_k_l617_617250


namespace midpoints_identical_l617_617539

variables {P : Type*} [MetricSpace P]
variables {A B C A1 B1 C1 A2 B2 C2 : P}

-- Define similarity condition for triangles
def similar (X Y Z X1 Y1 Z1 : P) : Prop :=
∃ (r : ℝ) (R : ℝ), ∀ (p q r : P), dist p q = r * dist (P p) (P q) ∧ angle p q r = angle (P p) (P q) (P r)

-- Assume triangles are similar with the given correspondences
axiom similar_tris : (similar A B C A1 B1 C1) ∧ (similar A B C A2 B2 C2)

-- Define the corresponding vertices condition
axiom correspondence : 
  (B1 = A) ∧ 
  (C1 = B) ∧ 
  (B2 = C) ∧ 
  (C2 = A)

-- Define midpoint of a segment
def midpoint (X Y : P) : P :=
  by sorry -- this will require the actual definition of midpoint in the context provided

-- The final theorem to prove
theorem midpoints_identical :
  midpoint A1 A2 = midpoint B C :=
by
  -- Proof is omitted since it's not part of the requirement
  sorry

end midpoints_identical_l617_617539


namespace janet_total_action_figures_l617_617039

/-- Janet owns 10 action figures, sells 6, gets 4 more in better condition,
and then receives twice her current collection from her brother.
We need to prove she ends up with 24 action figures. -/
theorem janet_total_action_figures :
  let initial := 10 in
  let after_selling := initial - 6 in
  let after_acquiring_better := after_selling + 4 in
  let from_brother := 2 * after_acquiring_better in
  after_acquiring_better + from_brother = 24 :=
by
  -- Proof would go here
  sorry

end janet_total_action_figures_l617_617039


namespace common_elements_UV_l617_617433
noncomputable theory

open Int

def U : Set ℤ := { k | ∃ n, 1 ≤ n ∧ n ≤ 3000 ∧ k = 5 * n }
def V : Set ℤ := { k | ∃ n, 1 ≤ n ∧ n ≤ 3000 ∧ k = 7 * n }

theorem common_elements_UV : { x | x ∈ U ∧ x ∈ V }.card = 428 := 
sorry

end common_elements_UV_l617_617433


namespace gino_initial_sticks_l617_617312

-- Definitions based on the conditions
def given_sticks : ℕ := 50
def remaining_sticks : ℕ := 13
def initial_sticks (x y : ℕ) : ℕ := x + y

-- Theorem statement based on the mathematically equivalent proof problem
theorem gino_initial_sticks :
  initial_sticks given_sticks remaining_sticks = 63 :=
by
  sorry

end gino_initial_sticks_l617_617312


namespace iphone_case_prices_l617_617574

noncomputable def original_price_iphone_11 := 82.65
noncomputable def original_price_iphone_12 := 85

theorem iphone_case_prices
    (units_iphone_11 : ℤ)
    (units_iphone_12 : ℤ)
    (price_iphone_11_discounted : ℝ)
    (price_iphone_12_discounted : ℝ)
    (discount_20 : ℝ := 0.20)
    (discount_30 : ℝ := 0.30)
    (total_price_11 : ℝ := 1620)
    (total_price_12 : ℝ := 816)
    (unit_threshold_20 : ℤ := 15)
    (unit_threshold_30 : ℤ := 25) :
    units_iphone_11 = 28 ∧
    units_iphone_12 = 12 ∧
    price_iphone_11_discounted = total_price_11 ∧
    price_iphone_12_discounted = total_price_12 →
    (units_iphone_11 > unit_threshold_30 →
        0.70 * ((units_iphone_11 : ℝ) * original_price_iphone_11) = total_price_11) ∧
    (units_iphone_12 > unit_threshold_20 ∧ units_iphone_12 ≤ unit_threshold_30 →
        0.80 * ((units_iphone_12 : ℝ) * original_price_iphone_12) = total_price_12) :=
by
  sorry 

end iphone_case_prices_l617_617574


namespace RamWeightIncrease_l617_617529

def WeightIncreasePercent (R S R' S' total_weight new_total_weight : ℝ) (k : ℝ) :=
  (R = 6 * k) ∧ (S = 5 * k) ∧ (total_weight = R + S) ∧ (new_total_weight = R' + S') ∧ (new_total_weight = 82.8) ∧ 
  (S' = 1.21 * S) ∧ (total_weight = 11 * k) →
  (R' = R + 0.148 * R) → 
  (R' = 44.08) →
  100 * (R' - R) / R = 9

theorem RamWeightIncrease : ∀ (R S R' S' total_weight new_total_weight : ℝ) (k : ℝ),
  WeightIncreasePercent R S R' S' total_weight new_total_weight k :=
begin
  sorry
end

end RamWeightIncrease_l617_617529


namespace radical_conjugate_sum_l617_617695

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617695


namespace halfway_fraction_l617_617173

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l617_617173


namespace penalty_kicks_total_l617_617946

theorem penalty_kicks_total :
  let total_players := 25
  let total_goalies := 4
  let injured_players := 2
  let injured_goalies := 1
  let remaining_players := total_players - injured_players
  let remaining_goalies := total_goalies - injured_goalies
  let shots_per_goalie := remaining_players - 1
  in remaining_goalies * shots_per_goalie = 66 :=
by
  sorry

end penalty_kicks_total_l617_617946


namespace simplify_fraction_l617_617474

theorem simplify_fraction (x : ℝ) (h1 : sin x = 2 * sin (x / 2) * cos (x / 2))
    (h2 : cos x = 2 * cos (x / 2) * cos (x / 2) - 1) :
    (1 + sin x + cos x) / (1 - sin x + cos x) = tan (π / 4 + x / 2) :=
by
  sorry

end simplify_fraction_l617_617474


namespace find_mistake_l617_617995

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l617_617995


namespace required_bisection_steps_l617_617536

-- Define the function f
def f (x : ℝ) : ℝ := 2^x + 3 * x - 7

-- Given conditions
variable (precision : ℝ) (f_a f_b : ℝ)
variable (a b : ℝ)
variable (N_needed : ℕ)

axiom condition_1 : precision = 0.05
axiom condition_2 : a = 1.5
axiom condition_3 : b = 1.25
axiom condition_4 : f_a = 0.32843
axiom condition_5 : f_b = -0.8716

-- Problem statement: prove the required number of bisections
theorem required_bisection_steps :
    (∃ n : ℕ, 1 / (2^n) < precision) ∧
    (a, b are incremented intervals) ∧
    (additional required steps = N_needed) :=
sorry

end required_bisection_steps_l617_617536


namespace find_base_r_l617_617404

-- Define the base-r to base-10 conversion function
def base_r_to_base_10 (r : ℕ) (a b c : ℕ) : ℕ :=
  a * r^2 + b * r + c

-- Define the conditions
def conditions_met (r : ℕ) : Prop :=
  base_r_to_base_10 r 4 4 0 + base_r_to_base_10 r 3 4 0 = base_r_to_base_10 r 1 0 0 0

-- Define the question as a proof statement
theorem find_base_r : ∃ r : ℕ, conditions_met r ∧ r = 8 :=
by
  sorry

end find_base_r_l617_617404


namespace distinct_digits_in_order_count_l617_617375

theorem distinct_digits_in_order_count :
  let nums := (Finset.range 3500).filter (fun n => (3120 ≤ n) 
    ∧ (let d := n.digits 10 in d.length = 4 ∧ d.nodup ∧ d = d.sorted)) in
  nums.card = 40 := by
  sorry

end distinct_digits_in_order_count_l617_617375


namespace men_in_room_l617_617035
noncomputable def numMenInRoom (x : ℕ) : ℕ := 4 * x + 2

theorem men_in_room (x : ℕ) (h_initial_ratio : true) (h_after_events : true) (h_double_women : 2 * (5 * x - 3) = 24) :
  numMenInRoom x = 14 :=
sorry

end men_in_room_l617_617035


namespace recycling_drive_target_l617_617285

-- Define the collection totals for each section
def section_collections_first_week : List ℝ := [260, 290, 250, 270, 300, 310, 280, 265]

-- Compute total collection for the first week
def total_first_week (collections: List ℝ) : ℝ := collections.sum

-- Compute collection for the second week with a 10% increase
def second_week_increase (collection: ℝ) : ℝ := collection * 1.10
def total_second_week (collections: List ℝ) : ℝ := (collections.map second_week_increase).sum

-- Compute collection for the third week with a 30% increase from the second week
def third_week_increase (collection: ℝ) : ℝ := collection * 1.30
def total_third_week (collections: List ℝ) : ℝ := (collections.map (second_week_increase)).sum * 1.30

-- Total target collection is the sum of collections for three weeks
def target (collections: List ℝ) : ℝ := total_first_week collections + total_second_week collections + total_third_week collections

-- Main theorem to prove
theorem recycling_drive_target : target section_collections_first_week = 7854.25 :=
by
  sorry -- skipping the proof

end recycling_drive_target_l617_617285


namespace maximum_area_of_triangle_l617_617442

noncomputable def parabola (p : ℝ) : ℝ := -p^2 + 8 * p - 15

def area_of_triangle (p : ℝ) : ℝ :=
  let A := (2, 5)
  let B := (5, 10)
  let C := (p, parabola p)
  abs (2 * 10 + 5 * parabola p + p * 5 - 5 * 2 - 10 * p - parabola p * 2) / 2

theorem maximum_area_of_triangle : 
  ∃ p (h : 0 ≤ p ∧ p ≤ 5), area_of_triangle p = 112.5 / 24 :=
sorry

end maximum_area_of_triangle_l617_617442


namespace geometric_sequence_equality_l617_617436

variable {α : Type*} [LinearOrderedField α]

-- For any geometric sequence {a_n} and for any positive integer n,
-- prove that the required equality holds. We assume {a_n} is a sequence
-- defined such that it forms a geometric progression.

def is_geometric (a : ℕ → α) := ∃ r : α, ∀ n : ℕ, a(n+1) = r * a(n)

theorem geometric_sequence_equality {a : ℕ → α} (h : is_geometric a) (n : ℕ) (hn : 0 < n) :
  (∏ i in Finset.range n, (1 / (a i * a (i+1)))) = (1 / (a 0 * a n)^n) :=
by
  sorry

end geometric_sequence_equality_l617_617436


namespace roe_december_savings_l617_617100

theorem roe_december_savings (monthly_savings_jan_to_jul : ℕ) (monthly_savings_aug_to_nov : ℕ) (total_goal : ℕ) (months_jan_to_jul : ℕ) (months_aug_to_nov : ℕ) :
  (monthly_savings_jan_to_jul = 10) →
  (monthly_savings_aug_to_nov = 15) →
  (total_goal = 150) →
  (months_jan_to_jul = 7) →
  (months_aug_to_nov = 4) →
  let savings_jan_to_jul := monthly_savings_jan_to_jul * months_jan_to_jul in
  let savings_aug_to_nov := monthly_savings_aug_to_nov * months_aug_to_nov in
  let savings_upto_nov := savings_jan_to_jul + savings_aug_to_nov in
  total_goal - savings_upto_nov = 20 :=
by
  intros h1 h2 h3 h4 h5
  let savings_jan_to_jul := 10 * 7
  let savings_aug_to_nov := 15 * 4
  let savings_upto_nov := savings_jan_to_jul + savings_aug_to_nov
  have h_savings_jan_to_jul : savings_jan_to_jul = 70 := by sorry
  have h_savings_aug_to_nov : savings_aug_to_nov = 60 := by sorry
  have h_savings_upto_nov : savings_upto_nov = 130 := by sorry
  show 150 - savings_upto_nov = 20 from by
    rewrite [←h_savings_upto_nov, ←h_savings_jan_to_jul, ←h_savings_aug_to_nov]
    sorry

end roe_december_savings_l617_617100


namespace convex_polygon_has_satisfying_set_l617_617434

noncomputable theory

variables {Point : Type} [Ord Point]

structure ConvexPolygon (Point : Type) :=
(vertices : list Point)
(convex : ∀ (a b c : Point), a ≠ b → b ≠ c → c ≠ a → (a, b, c) ∉ vertices)

def exists_satisfying_set (P : ConvexPolygon Point) : Prop :=
  ∃ S : set Point, S.card = P.vertices.length - 2 ∧
  ∀ I J K ∈ P.vertices, I ≠ J → J ≠ K → K ≠ I → ∃ s ∈ S, s ∈ triangle I J K

theorem convex_polygon_has_satisfying_set (P : ConvexPolygon Point) : exists_satisfying_set P :=
sorry

end convex_polygon_has_satisfying_set_l617_617434


namespace derivative_value_l617_617444

variable (f : ℝ → ℝ) (x₀ : ℝ)

-- Define the differentiability of f at x₀.
def differentiable_at (f : ℝ → ℝ) (x₀ : ℝ) := DifferentiableAt ℝ f x₀

-- Define the given limit condition.
noncomputable def limit_condition :=
  lim (λ Δx: ℝ, (f (x₀ - 3 * Δx) - f x₀) / Δx) (0 : ℝ) = 1

-- The final statement to prove.
theorem derivative_value (h₀ : differentiable_at f x₀) (h₁ : limit_condition f x₀) :
  deriv f x₀ = -1 / 3 :=
by
  sorry

end derivative_value_l617_617444


namespace fifth_day_matches_l617_617103

variable (Players : Finset ℕ := {1, 2, 3, 4, 5, 6})

def Match (x y : ℕ) := (x, y)

axiom match_first_day : Match 2 4  -- {B, D}
axiom match_second_day : Match 3 5 -- {C, E}
axiom match_third_day : Match 4 6  -- {D, F}
axiom match_fourth_day : Match 2 3 -- {B, C}

theorem fifth_day_matches :
  Match 1 2 ∧ Match 4 3 ∧ Match 5 6 :=
  sorry

#check fifth_day_matches

end fifth_day_matches_l617_617103


namespace solve_for_k_l617_617081

theorem solve_for_k (k : ℝ) (h1 : ∀ x, (f x = k * x + 1)) (h2 : f 2 = 3) : k = 1 :=
by 
  let f := fun x => k * x + 1
  have h2 : f 2 = k * 2 + 1 := by rw h1
  have eq1 : k * 2 + 1 = 3 := by rw h2
  have eq2 : k * 2 = 2 := by ring at eq1; exact eq1
  have solution : k = 1 := by ring at eq2; exact eq2
  exact solution

-- The proof can be accomplished if needed using the provided Lean steps and tactics, but it’s not required for this task.

end solve_for_k_l617_617081


namespace onions_sold_l617_617533

theorem onions_sold (initial remaining : ℕ) (h1 : initial = 98) (h2 : remaining = 33) : initial - remaining = 65 := 
by 
  rw [h1, h2]
  exact rfl

end onions_sold_l617_617533


namespace sum_of_number_and_conjugate_l617_617671

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617671


namespace find_k_l617_617883

-- Define the dilation matrix with scale factor k
def D (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![k, 0], ![0, k]]

-- Define the rotation matrix for -45 degrees
def R : Matrix (Fin 2) (Fin 2) ℝ :=
  let θ := -Real.pi / 4
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

-- Given matrix for RD
def RD : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 5], ![-5, 5]]

-- The proof problem
theorem find_k (k : ℝ) (h : k > 0) (h_eq : R ⬝ D k = RD) : k = 5 * Real.sqrt 2 :=
by
  sorry

end find_k_l617_617883


namespace distinct_values_of_S_l617_617486

theorem distinct_values_of_S :
  let i := Complex.I
  in {S : Complex | ∃ (n : ℤ), let k := (2 * n) % 4 in S = i^n + i^(k * -n)}.card = 4 :=
by
  sorry

end distinct_values_of_S_l617_617486


namespace ratio_of_probabilities_l617_617986

-- Step: Define the main theorem
theorem ratio_of_probabilities :
  let A := Nat.choose 25 5 * Nat.choose 20 5 * Nat.choose 15 3 * Nat.choose 12 3 * Nat.choose 9 2 * Nat.choose 7 2,
      B := Nat.choose 25 4 * Nat.choose 21 4 * Nat.choose 17 4 * Nat.choose 13 4 * Nat.choose 9 4 * Nat.choose 5 5 in
  (A : ℚ) / (B : ℚ) = 20 := -- Here 20 is the specific value assumed correct from the problem statement
by
  sorry

end ratio_of_probabilities_l617_617986


namespace gcd_of_differences_l617_617729

theorem gcd_of_differences (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 1351) : 
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a) = 4 :=
by
  sorry

end gcd_of_differences_l617_617729


namespace logarithmic_expression_l617_617255

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression :
  let log2 := lg 2
  let log5 := lg 5
  log2 + log5 = 1 →
  (log2^3 + 3 * log2 * log5 + log5^3 = 1) :=
by
  intros log2 log5 h
  sorry

end logarithmic_expression_l617_617255


namespace minimum_area_triangle_l617_617794

theorem minimum_area_triangle {Q : ℝ × ℝ} (hQ : Q = (2, 8)) :
  let l₁ : ℝ → ℝ := fun x => 4 * x,
      P := (6 : ℝ, 4 : ℝ),
      A (t : ℝ) := (5 * t / (t - 1), 0 : ℝ),
      area (t : ℝ) := real_geometry.signed_area ⟦(0, 0), A t, Q⟧ in
  area 2 = 40 := sorry

end minimum_area_triangle_l617_617794


namespace fifteenth_odd_multiple_of_5_l617_617194

theorem fifteenth_odd_multiple_of_5 :
  (∃ n: ℕ, n = 15 ∧ (10 * n - 5 = 145)) :=
begin
  use 15,
  split,
  { refl },
  { norm_num }
end

end fifteenth_odd_multiple_of_5_l617_617194


namespace erased_length_l617_617930

def original_length := 100 -- in cm
def final_length := 76 -- in cm

theorem erased_length : original_length - final_length = 24 :=
by
    sorry

end erased_length_l617_617930


namespace present_value_of_machine_l617_617598

theorem present_value_of_machine (F : ℝ) (r : ℝ) (P : ℝ) (t : ℕ) 
  (hF : F = 567) 
  (hr : r = 0.1) 
  (ht : t = 2) 
  (hP : P = 700) :
  F = P * (1 - r)^t := 
begin
  rw [hF, hr, ht, hP],
  norm_num,
  sorry,
end

end present_value_of_machine_l617_617598


namespace radical_conjugate_sum_l617_617691

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617691


namespace Andrea_has_winning_strategy_l617_617531

def sticks : List ℕ := List.range 1 99  -- list of sticks from 1 to 98

def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a 

def can_form_triangle (lst : List ℕ) : Prop :=
  match lst with
  | [a, b, c] := triangle_inequality a b c
  | _ => False

def Andrea_wins (sticks : List ℕ) : Prop :=
  ∃ S T : List ℕ, S.length = 48 ∧ T.length = 47 ∧ (S ⊆ sticks ∧ T ⊆ sticks ∧
  (¬ (S ∩ T).empty) ∧ (sticks.diff (S ∪ T)).length = 3 ∧ 
  can_form_triangle (sticks.diff (S ∪ T))) 

theorem Andrea_has_winning_strategy : Andrea_wins sticks := sorry

end Andrea_has_winning_strategy_l617_617531


namespace radical_conjugate_sum_l617_617693

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l617_617693


namespace algebraic_expression_eval_l617_617209

theorem algebraic_expression_eval (a b c : ℝ) (h : a * (-5:ℝ)^4 + b * (-5)^2 + c = 3): 
  a * (5:ℝ)^4 + b * (5)^2 + c = 3 :=
by
  sorry

end algebraic_expression_eval_l617_617209


namespace find_length_of_other_diagonal_l617_617951

theorem find_length_of_other_diagonal
  (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1: area = 75)
  (h2: d1 = 10) :
  d2 = 15 :=
by 
  sorry

end find_length_of_other_diagonal_l617_617951


namespace determine_S6_l617_617900

noncomputable def x : ℝ := sorry

def S (m : ℕ) : ℝ := x^m + x⁻¹^m

theorem determine_S6 (h : x + x⁻¹ = 5) : S 6 = 12098 := 
sorry

end determine_S6_l617_617900


namespace find_y_plus_one_over_y_l617_617340

variable (y : ℝ)

theorem find_y_plus_one_over_y (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 :=
by
  sorry

end find_y_plus_one_over_y_l617_617340


namespace reduced_price_16_rs_per_kg_l617_617214

noncomputable theory

variables (P R Q : ℝ)

theorem reduced_price_16_rs_per_kg 
  (h1 : R = 0.90 * P)
  (h2 : 800 = Q * P)
  (h3 : 800 = (Q + 5) * 0.90 * P) :
  R = 16.00 :=
by
  sorry

end reduced_price_16_rs_per_kg_l617_617214


namespace problem_2008_l617_617707

-- Definitions and conditions
def f : ℝ → ℝ := sorry
axiom cond1 : ∀ x : ℝ, f(x + 3) ≤ f(x) + 3
axiom cond2 : ∀ x : ℝ, f(x + 2) ≥ f(x) + 2
axiom f1 : f 1 = 2
def a_n (n : ℕ) : ℝ := f n

-- Proof statement
theorem problem_2008 : a_n 2008 = 2009 :=
by sorry

end problem_2008_l617_617707


namespace minimum_value_of_function_l617_617731

noncomputable def f (x : ℝ) : ℝ := 16^x - 4^(x+1) + 6

theorem minimum_value_of_function : ∃ x : ℝ, ∀ y : ℝ, f(y) ≥ 2 := sorry

end minimum_value_of_function_l617_617731


namespace molly_does_not_place_l617_617085

theorem molly_does_not_place (cards : Finset ℕ)
(Molly_placed : Finset (Fin 8 → ℕ)) :
(cards = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) →
(∀ i, cards.card = 12) →
(∀ (a b : Fin 8), ((Molly_placed a + Molly_placed b) % 3 = 0)) →
(¬ (3 ∈ Molly_placed) ∧
 ¬ (6 ∈ Molly_placed) ∧
 ¬ (9 ∈ Molly_placed) ∧
 ¬ (12 ∈ Molly_placed)) :=
begin
  sorry
end

end molly_does_not_place_l617_617085


namespace factorial_simplification_l617_617933

theorem factorial_simplification : (13.factorial / (10.factorial + 3 * 9.factorial)) = 464 := by
  sorry

end factorial_simplification_l617_617933


namespace total_problems_solved_is_480_l617_617511

def numberOfProblemsSolvedByMarvinYesterday := 40

def numberOfProblemsSolvedByMarvinToday (yesterday : ℕ) := 3 * yesterday

def totalProblemsSolvedByMarvin (yesterday today : ℕ) := yesterday + today

def totalProblemsSolvedByArvin (marvinTotal : ℕ) := 2 * marvinTotal

def totalProblemsSolved (marvinTotal arvinTotal : ℕ) := marvinTotal + arvinTotal

theorem total_problems_solved_is_480 :
  let y := numberOfProblemsSolvedByMarvinYesterday,
      t := numberOfProblemsSolvedByMarvinToday y,
      m_total := totalProblemsSolvedByMarvin y t,
      a_total := totalProblemsSolvedByArvin m_total,
      total := totalProblemsSolved m_total a_total in
  total = 480 :=
by
  let y := numberOfProblemsSolvedByMarvinYesterday
  let t := numberOfProblemsSolvedByMarvinToday y
  let m_total := totalProblemsSolvedByMarvin y t
  let a_total := totalProblemsSolvedByArvin m_total
  let total := totalProblemsSolved m_total a_total
  have : y = 40 := rfl
  have : t = 120 := by simp [numberOfProblemsSolvedByMarvinToday, this]
  have : m_total = 160 := by simp [totalProblemsSolvedByMarvin, this, this]
  have : a_total = 320 := by simp [totalProblemsSolvedByArvin, this]
  have : total = 480 := by simp [totalProblemsSolved, this, this]
  exact this

end total_problems_solved_is_480_l617_617511


namespace find_tricycles_l617_617022

theorem find_tricycles (b t w : ℕ) 
  (sum_children : b + t + w = 10)
  (sum_wheels : 2 * b + 3 * t = 26) :
  t = 6 :=
by sorry

end find_tricycles_l617_617022


namespace estimate_larger_than_difference_l617_617163

variable {x y : ℝ}

theorem estimate_larger_than_difference (h1 : x > y) (h2 : y > 0) :
    ⌈x⌉ - ⌊y⌋ > x - y := by
  sorry

end estimate_larger_than_difference_l617_617163


namespace pencil_probability_l617_617398

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem pencil_probability : 
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 14 :=
by
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  have h : probability = 5 / 14 := sorry
  exact h

end pencil_probability_l617_617398


namespace sum_reciprocal_l617_617342

open Real

theorem sum_reciprocal (y : ℝ) (h₁ : y^3 + (1 / y)^3 = 110) : y + (1 / y) = 5 :=
sorry

end sum_reciprocal_l617_617342


namespace find_2a6_minus_a4_l617_617763

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) = 2 * a (n + 1) - a n

theorem find_2a6_minus_a4 {a : ℕ → ℤ} 
  (h_seq : is_arithmetic_sequence a)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 6 - a 4 = 24 :=
by
  sorry

end find_2a6_minus_a4_l617_617763


namespace sun_salutations_per_year_l617_617109

-- Definitions 
def sun_salutations_per_weekday : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_per_year : ℕ := 52

-- Problem statement to prove
theorem sun_salutations_per_year :
  sun_salutations_per_weekday * weekdays_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l617_617109


namespace elimination_eq_l617_617745

variable (x y : ℝ)

def eq1 : Prop := 6 * x - 5 * y = 3
def eq2 : Prop := 3 * x + y = -15

theorem elimination_eq : eq1 ∧ eq2 → 21 * x = -72 :=
by
  intro h
  cases h with eq1_h eq2_h
  -- We'll skip the detailed steps and the actual proof
  sorry

end elimination_eq_l617_617745


namespace sum_of_number_and_radical_conjugate_l617_617674

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l617_617674


namespace sum_of_number_and_conjugate_l617_617670

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l617_617670


namespace sum_radical_conjugate_l617_617643

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l617_617643


namespace length_common_chord_l617_617540

/-- Let two circles have a radius of 15 cm and each one passes through the center of the other.
    Then the length of the common chord between them is 26√3 cm. -/
theorem length_common_chord (r : ℝ) (h_r : r = 15) :
  let d := r in
  let h := real.sqrt (r^2 - (r/2)^2) in
  let chord_length := 2 * h in
  chord_length = 26 * real.sqrt 3 :=
by
  sorry

end length_common_chord_l617_617540


namespace painting_price_decrease_l617_617978

theorem painting_price_decrease:
  ∀ (P_org P_final P_after_one_year: ℝ), 
  P_after_one_year = P_org * 1.25 → 
  P_final = P_org * 1.0625 → 
  ∃ (D: ℝ), 
  D = P_after_one_year - P_final ∧ 
  (D / P_after_one_year) * 100 = 15 :=
by
  intros
  have h1 : P_after_one_year = P_org * 1.25 := by assumption
  have h2 : P_final = P_org * 1.0625 := by assumption
  let D := P_after_one_year - P_final
  use D
  have h3 : D / P_after_one_year * 100 = 15 := sorry
  exact ⟨rfl, h3⟩

end painting_price_decrease_l617_617978


namespace convert_to_rectangular_l617_617268

theorem convert_to_rectangular :
  (2 : ℂ) * complex.exp (19 * real.pi * complex.I / 4) = -real.sqrt 2 + real.sqrt 2 * complex.I :=
sorry

end convert_to_rectangular_l617_617268


namespace total_problems_practiced_l617_617513

-- Definitions from conditions
variables (marvin_yesterday : Nat) (marvin_today : Nat) (arvin_yesterday : Nat) (arvin_today : Nat)

-- Conditions stated as definitions
def marvin_yesterday_solved : marvin_yesterday = 40 := rfl
def marvin_today_solved : marvin_today = 3 * marvin_yesterday := by rw [marvin_yesterday_solved]; exact rfl
def arvin_yesterday_solved : arvin_yesterday = 2 * marvin_yesterday := by rw [marvin_yesterday_solved]; exact rfl
def arvin_today_solved : arvin_today = 2 * marvin_today := by rw [marvin_today_solved]; exact rfl

-- The proof problem statement
theorem total_problems_practiced :
  marvin_yesterday + marvin_today + arvin_yesterday + arvin_today = 480 :=
by
  rw [marvin_yesterday_solved, marvin_today_solved, arvin_yesterday_solved, arvin_today_solved]
  sorry

end total_problems_practiced_l617_617513


namespace find_m_value_l617_617011

theorem find_m_value (m : ℝ) :
  (∃ m : ℝ, ∀ l : (ℝ × ℝ) × (ℝ × ℝ), let (A, B) := l in A = (1, m) ∧ B = (-2, sqrt 3) ∧
        (-sqrt 3) = (sqrt 3 - m) / (-2 - 1)) → m = -2 * sqrt 3 :=
by
  sorry

end find_m_value_l617_617011


namespace fifteenth_odd_multiple_of_5_l617_617193

theorem fifteenth_odd_multiple_of_5 :
  (∃ n: ℕ, n = 15 ∧ (10 * n - 5 = 145)) :=
begin
  use 15,
  split,
  { refl },
  { norm_num }
end

end fifteenth_odd_multiple_of_5_l617_617193


namespace parallel_sides_in_ngon_l617_617697

def reg_ngon_parallel_sides (n : ℕ) : Prop := 
  if even n then 
    ∃ i j : ℕ, i < n ∧ j < n ∧ i ≠ j ∧ (i + 1) % n ≠ j ∧ (j + 1) % n ≠ i ∧ 
    ((i + 1) + i) % n = ((j + 1) + j) % n
  else 
    ¬∃ i j : ℕ, i < n ∧ j < n ∧ i ≠ j ∧ (i + 1) % n ≠ j ∧ (j + 1) % n ≠ i ∧ 
    ((i + 1) + i) % n = ((j + 1) + j) % n

/-- 
Verify that for a regular n-gon:
- If n is even, there exists at least two pairs of parallel sides.
- If n is odd, it is impossible to have exactly one pair of parallel sides.
-/
theorem parallel_sides_in_ngon (n : ℕ) : 
  reg_ngon_parallel_sides n := 
sorry

end parallel_sides_in_ngon_l617_617697


namespace root_in_interval_and_second_bisection_l617_617414

noncomputable def f (x : ℝ) := x^3 + 3*x - 1

theorem root_in_interval_and_second_bisection {
    (h₀ : f 0 < 0)
    (h₁ : f 0.5 > 0) :
  (∃ x : ℝ, 0 < x ∧ x < 0.5 ∧ f x = 0) ∧ f 0.25 = 0.25^3 + 3*0.25 - 1 :=
  sorry

end root_in_interval_and_second_bisection_l617_617414


namespace distance_from_point_to_line_is_7_l617_617295

open Real

-- Definition of the point and the line.
def point : Vector ℝ 3 := ⟨[1, 2, 3], by simp⟩
def line (t : ℝ) : Vector ℝ 3 := ⟨[6 + 3 * t, 7 + 2 * t, 7 - 2 * t], by simp [add_assoc, add_mul, zero_add]⟩

-- The distance computation
def distance (p1 p2 : Vector ℝ 3) :=
  (p1 - p2).norm

-- The goal is to prove that the distance from the point to the line at the closest point is 7.
theorem distance_from_point_to_line_is_7 : ∃ t : ℝ, distance point (line t) = 7 :=
sorry

end distance_from_point_to_line_is_7_l617_617295


namespace second_divisor_l617_617591

theorem second_divisor (N : ℤ) (k : ℤ) (D : ℤ) (m : ℤ) 
  (h1 : N = 39 * k + 20) 
  (h2 : N = D * m + 7) : 
  D = 13 := sorry

end second_divisor_l617_617591


namespace vector_addition_l617_617751

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

-- State the theorem to prove 2a + b = (7, -3, 2)
theorem vector_addition : (2 • a + b) = (7, -3, 2) := by
  sorry

end vector_addition_l617_617751


namespace hexagon_perimeter_l617_617504

-- Define the length of a side of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def perimeter (num_sides side_length : ℕ) : ℕ :=
  num_sides * side_length

-- Theorem stating the perimeter of the hexagon with given side length is 42 inches
theorem hexagon_perimeter : perimeter num_sides side_length = 42 := by
  sorry

end hexagon_perimeter_l617_617504


namespace lines_intersect_on_circumcircle_l617_617071

noncomputable def incenter (A B C: Point): Point := sorry -- definition of incenter
noncomputable def circumcircle (A B C: Point): Circle := sorry -- definition of circumcircle
noncomputable def midpoint (F I: Point): Point := sorry -- definition of midpoint
noncomputable def arc (B D C: Point): Arc := sorry -- definition of arc
noncomputable def intersection (line1 line2: Line): Point := sorry -- definition of intersection of two lines

variables {A B C I D G E F: Point} {Γ : Circle}

-- Given:
axiom h1 : I = incenter A B C
axiom h2 : Γ = circumcircle A B C
axiom h3 : ∃ D, D ≠ (Γ.intersection (line_through A I))
axiom h4 : E ∈ arc B D C
axiom h5 : F ∈ segment B C ∧ ∠ BAF = ∠ CAE < (1 / 2) * ∠ BAC
axiom h6 : G = midpoint I F

-- To prove:
theorem lines_intersect_on_circumcircle : (Γ.contains (Γ.intersection (line_through D G)) ∧ Γ.contains (Γ.intersection (line_through E I))) :=
sorry

end lines_intersect_on_circumcircle_l617_617071


namespace problem1_problem2_l617_617771

-- Definitions of the sets A, B, and C based on conditions given
def setA : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def setB : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def setC (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Problem (1): Prove values of b and c
theorem problem1 (b c : ℝ) :
  (∀ x, x ∈ (setA ∩ setB) ↔ b*x^2 + 10*x + c ≥ 0) → b = -2 ∧ c = -12 := sorry

-- Universal set definition and its complement
def universalSet : Set ℝ := {x | True}
def complementA : Set ℝ := {x | (x ∉ setA)}

-- Problem (2): Range of a
theorem problem2 (a : ℝ) :
  (setC a ⊆ setB ∪ complementA) → a ∈ Set.Icc (-11/6) (9/4) := sorry

end problem1_problem2_l617_617771


namespace jerry_clock_reading_l617_617418

noncomputable def clock_reading_after_pills (pills : ℕ) (start_time : ℕ) (interval : ℕ) : ℕ :=
(start_time + (pills - 1) * interval) % 12

theorem jerry_clock_reading :
  clock_reading_after_pills 150 12 5 = 1 :=
by
  sorry

end jerry_clock_reading_l617_617418


namespace balls_in_boxes_l617_617427

theorem balls_in_boxes (p k x : ℕ) (hp : p ≥ k) (hx : x < ⌊(p * (p - k + 1)) / (2 * (k - 1))⌋) :
  ∃ boxes : Fin p → ℕ, (∀ i j : Fin p, boxes i = boxes j → i = j) ∨ ∃ l m : Fin p, l ≠ m ∧ boxes l = boxes m :=
sorry

end balls_in_boxes_l617_617427


namespace triangle_area_proof_l617_617857

noncomputable def triangle_area (A B C : Point) : Real := 
  0.5 * (B.x - A.x) * (C.y - A.y) - 0.5 * (B.y - A.y) * (C.x - A.x)

structure Triangle :=
  (A B C : Point)
  (AB_lt_AC : AB < AC := sorry)
  (H_is_orthocenter : true := sorry)
  (O_is_circumcenter : true := sorry)
  (midpoint_OH_on_BC : true := sorry)
  (BC_eq_1 : BC = 1 := sorry)
  (perimeter_eq_6 : AB + AC + BC = 6 := sorry)

theorem triangle_area_proof (T : Triangle) : 
  triangle_area T.A T.B T.C = 6/7 := 
sorry 

end triangle_area_proof_l617_617857


namespace number_of_elements_in_set_l617_617828

theorem number_of_elements_in_set (A1 B1 A2 A3 A4 B2 B3 B4 : ℝ × ℝ × ℝ) :
  (dist A1 A2 = 1 ∧ dist A1 A3 = 1 ∧ dist A1 A4 = 1 ∧ 
   dist B1 B2 = 1 ∧ dist B1 B3 = 1 ∧ dist B1 B4 = 1) →
  (∃! x, ∃ i j, i ∈ {1, 2, 3, 4} ∧ j ∈ {1, 2, 3, 4} ∧
    (x = (vector.from_tuple (B1.1 - A1.1, B1.2 - A1.2, B1.3 - A1.3)) • 
         (vector.from_tuple (B1.1 - A1.1 + (j match {1 := A2, 2 := A3, 3 := A4, 4 := B1}).1 - A1.1,
                                         B1.2 - A1.2 + (j match {1 := A2, 2 := A3, 3 := A4, 4 := B1}).2 - A1.2,
                                         B1.3 - A1.3 + (j match {1 := A2, 2 := A3, 3 := A4, 4 := B1}).3 - A1.3)) =
    1)) :=
sorry

end number_of_elements_in_set_l617_617828


namespace polynomial_root_count_l617_617260

theorem polynomial_root_count :
  ∃ (polynomial_count : ℕ), 
  (∀ (b : Fin 7 → Fin 2), 
    (X^7 + (b 6 : R) * X^6 + (b 5 : R) * X^5 + (b 4 : R) * X^4 + (b 3 : R) * X^3 + (b 2 : R) * X^2 + (b 1 : R) * X) 
    =  (X ^ 2) * (X + 1) * (X ^ 5 + a * X^4 + b * X^3 + c * X^2 + d * X + e) where 
    (b_i ∈ {0,1})) → 
  polynomial_count = 15 := 
by sorry

end polynomial_root_count_l617_617260


namespace part1_part2_part3_l617_617356

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x^2 + 1) / (a * x + b)

-- Given conditions:
-- 1. f(x) is an odd function
-- 2. f(1) = 2
theorem part1 (a b : ℝ) (f : ℝ → ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf1 : f 1 = 2) :
  f = λ x, x + (1/x) := sorry

-- Monotonicity on [1, 2]
theorem part2 (a : ℝ) (f : ℝ → ℝ) 
  (hf : f = λ x, x + (1/x)) :
  monotone (λ x : ℝ, f x) :=
begin
  sorry
end

-- Given g(x) and range of t
def g (x : ℝ) (t : ℝ) (f : ℝ → ℝ) : ℝ := f (x^2) - 2 * t * f x

-- The range of t
theorem part3 (a : ℝ) (t : ℝ) (f : ℝ → ℝ) 
  (hf : f = λ x, x + (1/x))
  (h_bound : ∀ (x1 x2 : ℝ), x1 ∈ Icc (1 : ℝ) 2 ∧ x2 ∈ Icc (1 : ℝ) 2 → abs (g x1 t f - g x2 t f) ≤ 9 / 4) :
  0 ≤ t ∧ t ≤ 9 / 2 :=
begin
  sorry
end

end part1_part2_part3_l617_617356


namespace sarah_trips_to_fill_tank_l617_617928

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * Math.pi * r^3

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  Math.pi * r^2 * h

def trips_needed (tank_volume bucket_volume : ℝ) : ℕ :=
  Nat.ceil (tank_volume / bucket_volume)

theorem sarah_trips_to_fill_tank : trips_needed (volume_cylinder 8 10) (volume_hemisphere 5) = 8 :=
  sorry

end sarah_trips_to_fill_tank_l617_617928


namespace determine_m_l617_617757

-- Define the function \( f(x) \).
def f (m : ℝ) (x : ℝ) : ℝ :=
  m + 2 / (3^x - 1)

-- State that \( f \) is an odd function.
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- The main theorem to prove that if \( f \) is an odd function, then \( m = 1 \).
theorem determine_m (m : ℝ) : is_odd_function (f m) → m = 1 := 
begin
  sorry
end

end determine_m_l617_617757


namespace compute_KX_KQ_l617_617874

-- Definitions for the conditions
variables (A B C X Y Z P Q S K : Type)
variable [equilateral_triangle ABC 8]
variable (X_on_AB : point_on_segment X A B)
variable (Y_on_AC : point_on_segment Y A C)
variable (Z_on_BC : point_on_segment Z B C)
variable (AX_len : length A X = 5)
variable (AY_len : length A Y = 3)
variable (concurrent : concurrent_lines (line_through A Z) (line_through B Y) (line_through C X) at S)
variable (P_on_circumcircle : on_circumcircle P (triangle A X Y))
variable (Q_on_circumcircle : on_circumcircle Q (triangle A X Y))
variable (intersect_ZX_P : second_intersection (line_through Z X) (circumcircle (triangle A X Y)) P)
variable (intersect_ZY_Q : second_intersection (line_through Z Y) (circumcircle (triangle A X Y)) Q)
variable (intersect_XQ_YK : intersection_point (line_through X Q) (line_through Y P) K)

-- The theorem to be proved
theorem compute_KX_KQ : KX_times_KQ = 144 :=
sorry

end compute_KX_KQ_l617_617874


namespace slope_analysis_l617_617703

theorem slope_analysis (x y : ℝ) :
  (x = 0 ∧ y = 0 → (∂ y / ∂ x) = 0) ∧
  (x = 1 ∧ y = 0 → (∂ y / ∂ x) = 1) ∧
  (x^2 + y^2 = 1 → (∂ y / ∂ x) = 1) ∧
  (x^2 + y^2 = sqrt 3 → (∂ y / ∂ x) = sqrt 3) ∧
  (x^2 + y^2 = 1 / sqrt 3 → (∂ y / ∂ x) = 1 / sqrt 3) :=
begin
  sorry
end

end slope_analysis_l617_617703


namespace solve_quadratic_eqn_l617_617479

theorem solve_quadratic_eqn (x : ℝ) : x^2 - 4 * x - 6 = 0 ↔ x = 2 + real.sqrt 10 ∨ x = 2 - real.sqrt 10 := 
by
  sorry

end solve_quadratic_eqn_l617_617479


namespace area_MNK_geq_quarter_area_ABC_l617_617072

theorem area_MNK_geq_quarter_area_ABC
  (O : Point)
  (A B C M : Point) 
  (h_triangle : IsAcuteTriangle O A B C)
  (h_M_on_AB : OnLine M A B) 
  (K : Point)
  (h_K_on_circ_AMO : OnCircumcircle K A M O)
  (h_K_on_AC : OnLine K A C)
  (N : Point)
  (h_N_on_circ_BMO : OnCircumcircle N B M O)
  (h_N_on_BC : OnLine N B C)
  : ∃ M, isMidpoint M A B ∧ Area (Triangle.mk M N K) = 1/4 * Area (Triangle.mk A B C) := sorry

end area_MNK_geq_quarter_area_ABC_l617_617072


namespace carol_rect_width_l617_617258

-- Definitions based on conditions from part a)
def carols_length : ℝ := 15
def jordans_length : ℝ := 8
def jordans_width : ℝ := 45
def equal_area_condition (carols_width jordans_area : ℝ) : Prop := carols_length * carols_width = jordans_area

-- The main theorem representing the proof problem
theorem carol_rect_width :
  ∃ (carols_width : ℝ),
    let jordans_area := jordans_length * jordans_width in
    equal_area_condition carols_width jordans_area ∧ carols_width = 24 :=
by
  sorry

end carol_rect_width_l617_617258


namespace abe_age_sum_l617_617525

theorem abe_age_sum (h : abe_age = 29) : abe_age + (abe_age - 7) = 51 :=
by
  sorry

end abe_age_sum_l617_617525


namespace line_intersects_circle_l617_617977

open Real

theorem line_intersects_circle : ∃ (p : ℝ × ℝ), (p.1 + p.2 = 1) ∧ ((p.1 - 1)^2 + (p.2 - 1)^2 = 2) :=
by
  sorry

end line_intersects_circle_l617_617977


namespace no_max_volume_α_l617_617115

noncomputable def isosceles_triangle (A B C D : Point) (α : ℝ) : Prop :=
  -- Definitions based on the conditions
  ∃ α : ℝ, α > π/4 ∧ α < π/2 ∧ 
  -- Base triangle ABC is isosceles with AB = AC = 1 and ∠ABC = α
  (AB = AC ∧ ∠ABC = α) ∧ 
  -- Plane DBC forms angle α with plane ABC
  (∡ (plane D B C) (plane A B C) = α) ∧ 
  -- ∠DAB = ∠DAC = α
  (∠DAB = α ∧ ∠DAC = α) ∧ 
  -- Points K and P on edges AD and BD respectively
  (on_edge K AD ∧ on_edge P BD) ∧ 
  -- Area of triangle KDP to the area of triangle ABD as 4 : 25
  (Area(KDP) / Area(ABD) = 4 / 25)

-- Volume calculation
noncomputable def volume_KDPC (α : ℝ) : ℝ :=
  (4 / 75) * (sin^3 α * sqrt (-cos (2 * α))) / (sin α - sqrt (-cos (2 * α)))

-- Statement for proving no specific α makes volume maximum
theorem no_max_volume_α (α : ℝ) (A B C D K P : Point) :
  isosceles_triangle A B C D α →
  ∀ α ∈ (π/4, π/2), ¬ (∃ α : ℝ, (volume_KDPC α) = max {volume_KDPC α | α ∈ (π/4, π/2)}) :=
by sorry

end no_max_volume_α_l617_617115


namespace find_a_l617_617982

-- Define the polynomial P(x)
def P (a : ℝ) (x : ℝ) := (a + x) * (1 + x)^4

-- Define a condition for the sum of the coefficients of the odd-power terms
def odd_sum_condition (a : ℝ) := 
  let c := ([ (1 : _), (4 : _), (6 : _), (4 : _), (1 : _) ] : list ℝ) in
  let s := list.sum (list.zipWith (*) c [ (a : _), 1, 0, 1, 0, 1 ]) in
  s = 32

-- Formalize the problem statement
theorem find_a : ∃ (a : ℝ), odd_sum_condition a ∧ a = 3 :=
by
  sorry

end find_a_l617_617982


namespace ratio_of_focus_to_vertex_l617_617885

theorem ratio_of_focus_to_vertex {a b : ℝ}
  (P_def : ∀ x, y = x^2 + 1)
  (V1 : (0, 1))
  (F1 : (0, 5/4))
  (A : (a, a^2 + 1))
  (B : (b, b^2 + 1))
  (right_angle : ∠ A V1 B = 90°)
  (Q : (x, y) with (x, y) = (a + b) / 2, ((a + b) / 2)^2 + 2)
  (V2 : (0, 2))
  (F2 : (0, 2 + 1 / 8)) :
  ∣F1.2 - F2.2∣ / ∣V1.2 - V2.2∣ = 9 / 8 := sorry

end ratio_of_focus_to_vertex_l617_617885


namespace CDs_per_rack_l617_617280

theorem CDs_per_rack (racks_on_shelf : ℕ) (CDs_on_shelf : ℕ) (h1 : racks_on_shelf = 4) (h2 : CDs_on_shelf = 32) : 
  CDs_on_shelf / racks_on_shelf = 8 :=
by
  sorry

end CDs_per_rack_l617_617280


namespace total_young_fish_l617_617910

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end total_young_fish_l617_617910


namespace sum_f_n_l617_617740

def f (n : ℕ) : ℝ :=
if ∃ k : ℤ, (real.logb 8 n = k) then real.logb 8 n else 0

theorem sum_f_n (S : ℝ) : 
  S = ∑ n in finset.range 1998, f n ↔ S = 55 / 3 := sorry

end sum_f_n_l617_617740


namespace binomial_15_4_l617_617640

theorem binomial_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binomial_15_4_l617_617640


namespace tangent_line_eq_l617_617354

def f (x : ℝ) : ℝ := sin x / x

def f' (x : ℝ) : ℝ := (x * cos x - sin x) / (x^2)

def M : ℝ × ℝ := (2 * Real.pi, 0)

def tangent_line_at_point (k x m : ℝ) : ℝ := k * (x - m)

theorem tangent_line_eq (x : ℝ) : 
  tangent_line_at_point (f' (2 * Real.pi)) x (2 * Real.pi) = (x / (2 * Real.pi)) - 1 :=
by
  sorry

end tangent_line_eq_l617_617354


namespace count_valid_numbers_l617_617845

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 3

def is_four_digit_number (n : ℕ) : Prop := 
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  is_valid_digit d1 ∧ is_valid_digit d2 ∧ is_valid_digit d3 ∧ is_valid_digit d4

def contains_both_digits (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  (d1 = 2 ∨ d2 = 2 ∨ d3 = 2 ∨ d4 = 2) ∧ (d1 = 3 ∨ d2 = 3 ∨ d3 = 3 ∨ d4 = 3)

def valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ is_four_digit_number n ∧ contains_both_digits n

theorem count_valid_numbers : (finset.filter valid_number (finset.range 10000)).card = 14 :=
by
  sorry

end count_valid_numbers_l617_617845


namespace parallel_EX_AP_proof_l617_617447

open EuclideanGeometry

variables {A B C H P Q R E X : Point}
variables (abc_triangle : Triangle A B C)
variables (orthocenter_H : Orthocenter H abc_triangle)
variables (circumcircle_P : PointOnCircumcircle P abc_triangle)
variables (distinct_P : P ≠ A ∧ P ≠ B ∧ P ≠ C)
variables (foot_E : FootOfAltitude E B abc_triangle)
variables (parall_PAQB : Parallelogram P A Q B)
variables (parall_PAR : Parallelogram P A R C)
variables (meeting_point_X : MeetsAt AQ HR X)
variables (parallel_EX_AP : Parallel EX AP)

theorem parallel_EX_AP_proof :
  (Orthocenter H abc_triangle) →
  (PointOnCircumcircle P abc_triangle) →
  (P ≠ A ∧ P ≠ B ∧ P ≠ C) →
  (FootOfAltitude E B abc_triangle) →
  (Parallelogram P A Q B) →
  (Parallelogram P A R C) →
  (MeetsAt AQ HR X) →
  Parallel EX AP :=
  by
    intros orthocenter_H circumcircle_P distinct_P foot_E parall_PAQB parall_PAR meeting_point_X
    exact sorry

end parallel_EX_AP_proof_l617_617447


namespace percentage_change_area_l617_617581

theorem percentage_change_area
    (L B : ℝ)
    (Area_original : ℝ) (Area_new : ℝ)
    (Length_new : ℝ) (Breadth_new : ℝ) :
    Area_original = L * B →
    Length_new = L / 2 →
    Breadth_new = 3 * B →
    Area_new = Length_new * Breadth_new →
    (Area_new - Area_original) / Area_original * 100 = 50 :=
  by
  intro h_orig_area hl_new hb_new ha_new
  sorry

end percentage_change_area_l617_617581


namespace correct_range_a_l617_617080

noncomputable def proposition_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
noncomputable def proposition_q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem correct_range_a (a : ℝ) :
  (¬ ∃ x, proposition_p a x → ¬ ∃ x, proposition_q x) →
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end correct_range_a_l617_617080


namespace length_DE_of_parallelogram_l617_617843

noncomputable def parallelogram_area (AB BC : ℝ) (angle_ABC : ℝ) : ℝ :=
  AB * BC * Real.sin angle_ABC

noncomputable def triangle_area (base height : ℝ) : ℝ :=
  0.5 * base * height

theorem length_DE_of_parallelogram (AB BC : ℝ) (angle_ABC AC : ℝ) (E : Point) 
    (AD : Point) (DE : ℝ) (parallelogram_area_ad : parallelogram_area AB BC angle_ABC = parallelogram_area AB = 1 BC = 4 angle_ABC = 60) : 
  DE = 2 * Real.sqrt 3 :=
by sorry

end length_DE_of_parallelogram_l617_617843


namespace probability_of_opposite_middle_vertex_l617_617251

noncomputable def ant_moves_to_opposite_middle_vertex_prob : ℚ := 1 / 2

-- Specification of the problem conditions
structure Octahedron :=
  (middle_vertices : Finset ℕ) -- Assume some identification of middle vertices
  (adjacent_vertices : ℕ → Finset ℕ) -- Function mapping a vertex to its adjacent vertices
  (is_middle_vertex : ℕ → Prop) -- Predicate to check if a vertex is a middle vertex
  (is_top_or_bottom_vertex : ℕ → Prop) -- Predicate to check if a vertex is a top or bottom vertex
  (start_vertex : ℕ)

variables (O : Octahedron)

-- Main theorem statement
theorem probability_of_opposite_middle_vertex :
  ∃ A B : ℕ, A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B) →
  (∀ (A B : ℕ), (A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B)) →
    ant_moves_to_opposite_middle_vertex_prob = 1 / 2) := sorry

end probability_of_opposite_middle_vertex_l617_617251


namespace real_root_implies_m_equals_1_l617_617067

theorem real_root_implies_m_equals_1 (a b c m n : ℝ) :
  (∀ x : ℂ, x = n ∧ n^2 - 2*n + 1 + (m-n)*complex.I = 0) → m = 1 :=
by
  sorry

end real_root_implies_m_equals_1_l617_617067


namespace general_formula_an_sum_Tn_l617_617347

-- Given conditions
variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom cond1 : ∀ n : ℕ, n > 0 → S n = a 1 + (n - 1) * (3 * a 1 - 3) / 2
axiom cond2 : a 1 = 3
axiom cond3 : ∀ n : ℕ, n > 0 → 2 * S n = 3 * a n - 3

-- Proving the general formula for an
theorem general_formula_an : ∀ n : ℕ, n > 0 → a n = 3^n :=
begin
  sorry
end

variables (b : ℕ → ℤ) (T : ℕ → ℤ)

-- Additional conditions for bn and Tn
axiom cond4 : ∀ n : ℕ, n > 0 → b n = a n + int.log 3 (a n)
axiom cond5 : ∀ n : ℕ, n > 0 → T n = (∑ k in finset.range n, b k.succ)

-- Proving the sum of the first n terms of bn
theorem sum_Tn : ∀ n : ℕ, n > 0 → T n = (3^(n+1) + n^2 + n - 3) / 2 :=
begin
  sorry
end

end general_formula_an_sum_Tn_l617_617347


namespace max_min_values_of_f_l617_617506

noncomputable def f (x : ℝ) : ℝ := x^2

theorem max_min_values_of_f : 
  (∀ x, -3 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 0) :=
by
  sorry

end max_min_values_of_f_l617_617506


namespace can_decompose_50x50_square_into_1x4_strips_l617_617415

theorem can_decompose_50x50_square_into_1x4_strips :
  (∀ (n : ℕ) (hk : n = 50 * 50), ∃ (k : ℕ) (hm : k = 2500 / 2),
    (hk = 50 * 50) ∧ (hm = 1250) ∧ (1250 % 4 = 0)) :=
by
  sorry

end can_decompose_50x50_square_into_1x4_strips_l617_617415


namespace clock_angle_at_230_l617_617554

/-- Angle calculation problem: Determine the degree measure of the smaller angle formed by the 
    hands of a clock at 2:30. -/
theorem clock_angle_at_230 : 
  let angle_per_hour := 360 / 12,
      hour_position := 2 * angle_per_hour + (angle_per_hour / 2),
      minute_position := 30 * 6,
      angle_difference := abs (minute_position - hour_position)
  in angle_difference = 105 :=
by
  sorry

end clock_angle_at_230_l617_617554


namespace sum_and_times_l617_617524

theorem sum_and_times 
  (a : ℕ) (ha : a = 99) 
  (b : ℕ) (hb : b = 301) 
  (c : ℕ) (hc : c = 200) : 
  a + b = 2 * c :=
by 
  -- skipping proof 
  sorry

end sum_and_times_l617_617524


namespace polynomial_divisibility_l617_617379

noncomputable def iterate_poly (p : ℕ → ℕ) (x: ℕ) : ℕ → ℕ :=
nat.rec_on n x (λ n pn, p pn)

theorem polynomial_divisibility (p : ℕ → ℕ) :
  (∀ x, p x - x ∣ p(p(x)) - p(x)) →
  ∀ x, p x - x ∣ p(p(p(...(p(x)))...2003 times) - 2 * p(p(p(...p(x)...)...2002 times) + p(p(p(...p(x)...)...2001 times) :=
begin
  sorry
end

end polynomial_divisibility_l617_617379


namespace ratio_of_radii_l617_617400

theorem ratio_of_radii (r R : ℝ) (hR : R > 0) (hr : r > 0)
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l617_617400


namespace N_is_85714_l617_617518

theorem N_is_85714 (N : ℕ) (hN : 10000 ≤ N ∧ N < 100000) 
  (P : ℕ := 200000 + N) 
  (Q : ℕ := 10 * N + 2) 
  (hQ_eq_3P : Q = 3 * P) 
  : N = 85714 := 
by 
  sorry

end N_is_85714_l617_617518


namespace range_of_a_l617_617904

section
variable (a : ℝ)
def M := set.Ico (-1 / 2 : ℝ) 2
def N := { x : ℝ | (x - a) * (x + a - 2) < 0 }
def necessary_condition (a : ℝ) : Prop := M ⊆ N

theorem range_of_a :
  necessary_condition a ↔ (a ≤ -1 / 2 ∨ a ≥ 5 / 2) :=
by
  sorry
end

end range_of_a_l617_617904


namespace not_units_digit_square_l617_617569

theorem not_units_digit_square (d : ℕ) : 
  ∃ (x : ℕ), x^2 % 10 = d ↔ d ∈ {0, 1, 4, 9, 6, 5} :=
by {
  sorry
}

end not_units_digit_square_l617_617569


namespace smallest_odd_integer_of_set_l617_617508

theorem smallest_odd_integer_of_set (S : Set Int) 
  (h1 : ∃ m : Int, m ∈ S ∧ m = 149)
  (h2 : ∃ n : Int, n ∈ S ∧ n = 159)
  (h3 : ∀ a b : Int, a ∈ S → b ∈ S → a ≠ b → (a - b) % 2 = 0) : 
  ∃ s : Int, s ∈ S ∧ s = 137 :=
by sorry

end smallest_odd_integer_of_set_l617_617508


namespace price_of_brand_X_l617_617716

theorem price_of_brand_X :
  (∀ (P : ℝ), (8 * P + 4 * 2.80 = 40) → (P = 3.60)) :=
by
  intro P h
  have h_eq : 8 * P + 11.2 = 40 := by rw [mul_add, mul_comm, ←add_assoc]; exact h
  have h_sub : 8 * P = 28.8 := by linarith
  have h_div : P = 3.60 := by linarith
  assumption

end price_of_brand_X_l617_617716


namespace max_value_bn_over_an_l617_617778

-- Definition of arithmetic sequence b_n
def b (n : ℕ) : ℤ := n - 35

-- Definition of sequence a_n satisfying the given recurrence relation
def a : ℕ → ℤ
| 0     := b 36
| (n+1) := a n + 2^n

-- Maximum value of the sequence {b_n / a_n}
theorem max_value_bn_over_an :
  ∃ n_max, (∀ n, (b n).toRat / (a n).toRat ≤ (b n_max).toRat / (a n_max).toRat) ∧
           (b n_max).toRat / (a n_max).toRat = 1 / 2^36.toRat := 
sorry

end max_value_bn_over_an_l617_617778


namespace triangle_perpendicular_bisector_properties_l617_617761

variables {A B C A1 A2 B1 B2 C1 C2 : Type} (triangle : triangle A B C)
  (A1_perpendicular : dropping_perpendicular_to_bisector A )
  (A2_perpendicular : dropping_perpendicular_to_bisector A )
  (B1_perpendicular : dropping_perpendicular_to_bisector B )
  (B2_perpendicular : dropping_perpendicular_to_bisector B )
  (C1_perpendicular : dropping_perpendicular_to_bisector C )
  (C2_perpendicular : dropping_perpendicular_to_bisector C )
  
-- Defining required structures
structure triangle (A B C : Type) :=
  (AB BC CA : ℝ)

structure dropping_perpendicular_to_bisector (v : Type) :=
  (perpendicular_to_bisector : ℝ)

namespace triangle_properties

theorem triangle_perpendicular_bisector_properties :
  2 * (A1_perpendicular.perpendicular_to_bisector + A2_perpendicular.perpendicular_to_bisector + 
       B1_perpendicular.perpendicular_to_bisector + B2_perpendicular.perpendicular_to_bisector + 
       C1_perpendicular.perpendicular_to_bisector + C2_perpendicular.perpendicular_to_bisector) = 
  (triangle.AB + triangle.BC + triangle.CA) :=
sorry

end triangle_properties

end triangle_perpendicular_bisector_properties_l617_617761


namespace halfway_fraction_l617_617183

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l617_617183
