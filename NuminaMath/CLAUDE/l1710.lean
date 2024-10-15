import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_540_l1710_171094

theorem sum_of_two_smallest_prime_factors_of_540 :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧
  p ∣ 540 ∧ q ∣ 540 ∧
  (∀ (r : ℕ), Nat.Prime r → r ∣ 540 → r ≥ p) ∧
  (∀ (r : ℕ), Nat.Prime r → r ∣ 540 → r ≠ p → r ≥ q) ∧
  p + q = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_540_l1710_171094


namespace NUMINAMATH_CALUDE_cubic_equation_roots_progression_l1710_171096

/-- Represents a cubic equation x³ + ax² + bx + c = 0 -/
structure CubicEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- The roots of a cubic equation -/
structure CubicRoots (α : Type*) [Field α] where
  x₁ : α
  x₂ : α
  x₃ : α

/-- Checks if the roots form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  roots.x₁ - roots.x₂ = roots.x₂ - roots.x₃

/-- Checks if the roots form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  roots.x₂ / roots.x₁ = roots.x₃ / roots.x₂

/-- Checks if the roots form a harmonic sequence -/
def is_harmonic_sequence {α : Type*} [Field α] (roots : CubicRoots α) : Prop :=
  (roots.x₁ - roots.x₂) / (roots.x₂ - roots.x₃) = roots.x₁ / roots.x₃

theorem cubic_equation_roots_progression {α : Type*} [Field α] (eq : CubicEquation α) (roots : CubicRoots α) :
  (is_arithmetic_progression roots ↔ (2 * eq.a^3 + 27 * eq.c) / (9 * eq.a) = eq.b) ∧
  (is_geometric_progression roots ↔ eq.b = eq.a * (eq.c^(1/3))) ∧
  (is_harmonic_sequence roots ↔ eq.a = (2 * eq.b^3 + 27 * eq.c) / (9 * eq.b^2)) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_progression_l1710_171096


namespace NUMINAMATH_CALUDE_jack_total_travel_time_l1710_171013

/-- Represents the time spent in a country during travel -/
structure CountryTime where
  customsHours : ℕ
  quarantineDays : ℕ

/-- Calculates the total hours spent in a country -/
def totalHoursInCountry (ct : CountryTime) : ℕ :=
  ct.customsHours + 24 * ct.quarantineDays

/-- The time Jack spent in each country -/
def jackTravelTime : List CountryTime := [
  { customsHours := 20, quarantineDays := 14 },  -- Canada
  { customsHours := 15, quarantineDays := 10 },  -- Australia
  { customsHours := 10, quarantineDays := 7 }    -- Japan
]

/-- Theorem stating the total time Jack spent in customs and quarantine -/
theorem jack_total_travel_time :
  List.foldl (λ acc ct => acc + totalHoursInCountry ct) 0 jackTravelTime = 789 :=
by sorry

end NUMINAMATH_CALUDE_jack_total_travel_time_l1710_171013


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l1710_171018

theorem cube_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) :
  a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l1710_171018


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1710_171049

theorem price_reduction_percentage (original_price reduction_amount : ℝ) :
  original_price = 500 →
  reduction_amount = 400 →
  (reduction_amount / original_price) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1710_171049


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1710_171006

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x < y → y ≤ 3 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1710_171006


namespace NUMINAMATH_CALUDE_charlie_spent_56250_l1710_171047

/-- The amount Charlie spent on acorns -/
def charlie_spent (alice_acorns bob_acorns charlie_acorns : ℕ) 
  (bob_total : ℚ) (alice_multiplier : ℕ) : ℚ :=
  let bob_price := bob_total / bob_acorns
  let alice_price := alice_multiplier * bob_price
  let average_price := (bob_price + alice_price) / 2
  charlie_acorns * average_price

/-- Theorem stating that Charlie spent $56,250 on acorns -/
theorem charlie_spent_56250 :
  charlie_spent 3600 2400 4500 6000 9 = 56250 := by
  sorry

end NUMINAMATH_CALUDE_charlie_spent_56250_l1710_171047


namespace NUMINAMATH_CALUDE_season_games_count_l1710_171065

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of baseball games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games_count : total_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l1710_171065


namespace NUMINAMATH_CALUDE_fraction_simplification_l1710_171046

theorem fraction_simplification :
  (270 : ℚ) / 24 * 7 / 210 * 6 / 4 = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1710_171046


namespace NUMINAMATH_CALUDE_common_roots_solution_l1710_171038

/-- Two cubic polynomials with common roots -/
def poly1 (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 14*x + 8

def poly2 (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + 17*x + 10

/-- The polynomials have two distinct common roots -/
def has_two_common_roots (a b : ℝ) : Prop :=
  ∃ r s : ℝ, r ≠ s ∧ poly1 a r = 0 ∧ poly1 a s = 0 ∧ poly2 b r = 0 ∧ poly2 b s = 0

/-- The main theorem -/
theorem common_roots_solution :
  has_two_common_roots 7 8 :=
sorry

end NUMINAMATH_CALUDE_common_roots_solution_l1710_171038


namespace NUMINAMATH_CALUDE_divisible_by_fifteen_l1710_171036

theorem divisible_by_fifteen (x : ℤ) : 
  (∃ k : ℤ, x^2 + 2*x + 6 = 15 * k) ↔ 
  (∃ t : ℤ, x = 15*t - 6 ∨ x = 15*t + 4) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_fifteen_l1710_171036


namespace NUMINAMATH_CALUDE_g_inequality_solution_set_range_of_a_l1710_171011

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - 5*a| + |2*x + 1|
def g (x : ℝ) : ℝ := |x - 1| + 3

-- Theorem for the solution set of |g(x)| < 8
theorem g_inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 := by sorry

end NUMINAMATH_CALUDE_g_inequality_solution_set_range_of_a_l1710_171011


namespace NUMINAMATH_CALUDE_quadratic_curve_point_exclusion_l1710_171044

theorem quadratic_curve_point_exclusion (a c : ℝ) (h : a * c > 0) :
  ¬∃ d : ℝ, 0 = a * 2018^2 + c * 2018 + d := by
  sorry

end NUMINAMATH_CALUDE_quadratic_curve_point_exclusion_l1710_171044


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1710_171090

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ * (5 * x₁ - 9) = -4) ∧
    (x₂ * (5 * x₂ - 9) = -4) ∧
    (x₁ = (9 + Real.sqrt 1) / 10) ∧
    (x₂ = (9 - Real.sqrt 1) / 10) ∧
    (9 + 1 + 10 = 20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1710_171090


namespace NUMINAMATH_CALUDE_function_inequality_l1710_171023

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → (x + 1) * f x + x * f' x ≥ 0) :
  f 1 < 2 * ℯ * f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1710_171023


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_l1710_171000

/-- If the line y = x + t is tangent to the curve y = e^x, then t = 1 -/
theorem tangent_line_to_exponential (t : ℝ) : 
  (∃ x₀ : ℝ, (x₀ + t = Real.exp x₀) ∧ 
             (1 = Real.exp x₀)) → 
  t = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_l1710_171000


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1710_171093

theorem necessary_not_sufficient_condition (x : ℝ) : 
  (∀ y : ℝ, y > 2 → y > 1) ∧ (∃ z : ℝ, z > 1 ∧ z ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1710_171093


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l1710_171035

/-- Represents a three-digit positive integer with no repeated digits -/
structure ThreeDigitNoRepeat where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : 
    1 ≤ hundreds ∧ hundreds ≤ 9 ∧
    0 ≤ tens ∧ tens ≤ 9 ∧
    0 ≤ ones ∧ ones ≤ 9 ∧
    hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

/-- Converts a ThreeDigitNoRepeat to its integer value -/
def ThreeDigitNoRepeat.toNat (n : ThreeDigitNoRepeat) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The largest three-digit positive integer with no repeated digits -/
def largest : ThreeDigitNoRepeat := {
  hundreds := 9
  tens := 8
  ones := 7
  is_valid := by sorry
}

/-- The smallest three-digit positive integer with no repeated digits -/
def smallest : ThreeDigitNoRepeat := {
  hundreds := 1
  tens := 0
  ones := 2
  is_valid := by sorry
}

/-- The main theorem -/
theorem difference_largest_smallest : 
  largest.toNat - smallest.toNat = 885 := by sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l1710_171035


namespace NUMINAMATH_CALUDE_solution_equivalence_l1710_171066

def solution_set : Set ℝ := {x | 1 < x ∧ x ≤ 3}

def inequality_system (x : ℝ) : Prop := 1 - x < 0 ∧ x - 3 ≤ 0

theorem solution_equivalence : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality_system x := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l1710_171066


namespace NUMINAMATH_CALUDE_equation_solution_l1710_171072

theorem equation_solution :
  ∀ N : ℚ, (5 + 6 + 7) / 3 = (2020 + 2021 + 2022) / N → N = 1010.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1710_171072


namespace NUMINAMATH_CALUDE_triangle_side_length_l1710_171079

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 → a = Real.sqrt 3 → b = 1 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c ^ 2 = a ^ 2 + b ^ 2 →
  c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1710_171079


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_two_l1710_171040

def A (m : ℝ) : Set ℝ := {-2, 3, 4*m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_two (m : ℝ) :
  B m ⊆ A m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_two_l1710_171040


namespace NUMINAMATH_CALUDE_expression_evaluation_l1710_171052

theorem expression_evaluation :
  let x : ℚ := -1/2
  3 * x^2 - (5*x - 3*(2*x - 1) + 7*x^2) = -9/2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1710_171052


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l1710_171084

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_sum_symmetric : f 5 + f (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l1710_171084


namespace NUMINAMATH_CALUDE_at_least_one_irrational_l1710_171002

theorem at_least_one_irrational (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :
  ¬(∃ (q r : ℚ), (a = ↑q ∧ b = ↑r)) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_irrational_l1710_171002


namespace NUMINAMATH_CALUDE_minimum_framing_feet_l1710_171098

-- Define the original picture dimensions
def original_width : ℕ := 5
def original_height : ℕ := 7

-- Define the enlargement factor
def enlargement_factor : ℕ := 2

-- Define the border width
def border_width : ℕ := 3

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem minimum_framing_feet :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + inches_per_foot - 1) / inches_per_foot = 6 := by
  sorry

end NUMINAMATH_CALUDE_minimum_framing_feet_l1710_171098


namespace NUMINAMATH_CALUDE_max_divisors_1_to_20_l1710_171021

def divisorCount (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def maxDivisorCount : ℕ → ℕ
  | 0 => 0
  | n + 1 => max (maxDivisorCount n) (divisorCount (n + 1))

theorem max_divisors_1_to_20 :
  maxDivisorCount 20 = 6 ∧
  divisorCount 12 = 6 ∧
  divisorCount 18 = 6 ∧
  divisorCount 20 = 6 ∧
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → divisorCount n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_divisors_1_to_20_l1710_171021


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l1710_171033

/-- The number of cakes Baker made initially -/
def total_cakes : ℕ := 48

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 44

/-- The number of cakes Baker has left -/
def remaining_cakes : ℕ := 4

/-- Theorem stating that the total number of cakes is equal to the sum of sold and remaining cakes -/
theorem baker_cakes_theorem : total_cakes = sold_cakes + remaining_cakes := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l1710_171033


namespace NUMINAMATH_CALUDE_max_value_expressions_l1710_171087

theorem max_value_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a / (2 * a + b)) + Real.sqrt (b / (2 * b + a)) ≤ 2 * Real.sqrt 3 / 3) ∧
  (Real.sqrt (a / (a + 2 * b)) + Real.sqrt (b / (b + 2 * a)) ≤ 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expressions_l1710_171087


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l1710_171068

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (hr₁ : r₁ = 10)
  (hr₂ : r₂ = 6)
  (hcd : contact_distance = 30) :
  let center_distance := Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2))
  center_distance = 2 * Real.sqrt 229 := by sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l1710_171068


namespace NUMINAMATH_CALUDE_union_of_sets_l1710_171071

def M : Set Int := {-1, 3, -5}
def N (a : Int) : Set Int := {a + 2, a^2 - 6}

theorem union_of_sets :
  ∃ a : Int, (M ∩ N a = {3}) → (M ∪ N a = {-5, -1, 3, 5}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1710_171071


namespace NUMINAMATH_CALUDE_cannot_obtain_1998_pow_7_initial_condition_final_not_divisible_main_result_l1710_171099

def board_operation (n : ℕ) : ℕ :=
  let last_digit := n % 10
  (n / 10) + 5 * last_digit

theorem cannot_obtain_1998_pow_7 (n : ℕ) (h : 7 ∣ n) :
  ∀ k : ℕ, 7 ∣ (board_operation^[k] n) ∧ (board_operation^[k] n) ≠ 1998^7 :=
by sorry

theorem initial_condition : 7 ∣ 7^1998 :=
by sorry

theorem final_not_divisible : ¬(7 ∣ 1998^7) :=
by sorry

theorem main_result : ∀ k : ℕ, (board_operation^[k] 7^1998) ≠ 1998^7 :=
by sorry

end NUMINAMATH_CALUDE_cannot_obtain_1998_pow_7_initial_condition_final_not_divisible_main_result_l1710_171099


namespace NUMINAMATH_CALUDE_triangle_properties_l1710_171067

-- Define the triangle ABC
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Condition 1: 2c = a + 2b*cos(A)
  2 * c = a + 2 * b * Real.cos A ∧
  -- Condition 2: Area of triangle ABC is √3
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3 ∧
  -- Condition 3: b = √13
  b = Real.sqrt 13

-- Theorem statement
theorem triangle_properties (a b c A B C : ℝ) 
  (h : triangle a b c A B C) : 
  B = Real.pi / 3 ∧ 
  a + b + c = 5 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1710_171067


namespace NUMINAMATH_CALUDE_train_station_distance_l1710_171063

/-- The distance to the train station -/
def distance : ℝ := 4

/-- The speed of the man in the first scenario (km/h) -/
def speed1 : ℝ := 4

/-- The speed of the man in the second scenario (km/h) -/
def speed2 : ℝ := 5

/-- The time difference between the man's arrival and the train's arrival in the first scenario (minutes) -/
def time_diff1 : ℝ := 6

/-- The time difference between the man's arrival and the train's arrival in the second scenario (minutes) -/
def time_diff2 : ℝ := -6

theorem train_station_distance :
  (distance / speed1 - distance / speed2) * 60 = time_diff1 - time_diff2 := by sorry

end NUMINAMATH_CALUDE_train_station_distance_l1710_171063


namespace NUMINAMATH_CALUDE_pens_probability_l1710_171029

theorem pens_probability (total_pens : Nat) (defective_pens : Nat) (bought_pens : Nat) :
  total_pens = 12 →
  defective_pens = 4 →
  bought_pens = 2 →
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1)) = 14 / 33 := by
  sorry

#eval (14 : ℚ) / 33 -- To verify the approximate decimal value

end NUMINAMATH_CALUDE_pens_probability_l1710_171029


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l1710_171061

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (3 / 2 * usual_rate * (usual_time - 4) = usual_rate * usual_time) →
  usual_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l1710_171061


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_range_of_c_l1710_171025

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the inequality
def inequality (a b : ℝ) := {x : ℝ | f a b x < 0}

-- Define the second quadratic function
def g (b c x : ℝ) := -x^2 + b*x + c

-- Theorem 1
theorem sum_of_a_and_b (a b : ℝ) : 
  inequality a b = {x | 2 < x ∧ x < 3} → a + b = 11 := by sorry

-- Theorem 2
theorem range_of_c (c : ℝ) : 
  (∀ x, g 6 c x ≤ 0) → c ≤ -9 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_range_of_c_l1710_171025


namespace NUMINAMATH_CALUDE_greatest_a_for_equation_l1710_171092

theorem greatest_a_for_equation :
  ∃ (a : ℝ), 
    (∀ (x : ℝ), (5 * Real.sqrt ((2 * x)^2 + 1) - 4 * x^2 - 1) / (Real.sqrt (1 + 4 * x^2) + 3) = 3 → x ≤ a) ∧
    (5 * Real.sqrt ((2 * a)^2 + 1) - 4 * a^2 - 1) / (Real.sqrt (1 + 4 * a^2) + 3) = 3 ∧
    a = Real.sqrt ((5 + Real.sqrt 10) / 2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_a_for_equation_l1710_171092


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1710_171060

theorem fraction_to_decimal (h : 625 = 5^4) : 17 / 625 = 0.0272 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1710_171060


namespace NUMINAMATH_CALUDE_octal_calculation_l1710_171007

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Performs subtraction in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Theorem stating the result of the given octal calculation --/
theorem octal_calculation : 
  octal_sub (octal_sub (to_octal 123) (to_octal 51)) (to_octal 15) = to_octal 25 :=
sorry

end NUMINAMATH_CALUDE_octal_calculation_l1710_171007


namespace NUMINAMATH_CALUDE_range_of_f_l1710_171053

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f, -8 ≤ y ∧ y ≤ 8 ∧
  ∀ z, -8 ≤ z ∧ z ≤ 8 → ∃ x, f x = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1710_171053


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1710_171085

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 15) → (∃ m : ℤ, N = 13 * m + 2) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1710_171085


namespace NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l1710_171089

theorem bill_sunday_saturday_difference (bill_sat bill_sun julia_sun : ℕ) : 
  bill_sun > bill_sat →
  julia_sun = 2 * bill_sun →
  bill_sat + bill_sun + julia_sun = 32 →
  bill_sun = 9 →
  bill_sun - bill_sat = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l1710_171089


namespace NUMINAMATH_CALUDE_complex_magnitude_l1710_171008

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1710_171008


namespace NUMINAMATH_CALUDE_intersection_reciprocal_sum_l1710_171057

/-- Given a line intersecting y = x^2 at (x₁, x₁²) and (x₂, x₂²), and the x-axis at (x₃, 0), 
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem intersection_reciprocal_sum (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) (h₃ : x₃ ≠ 0)
  (line_eq : ∃ (k m : ℝ), ∀ x y, y = k * x + m ↔ (x = x₁ ∧ y = x₁^2) ∨ (x = x₂ ∧ y = x₂^2) ∨ (x = x₃ ∧ y = 0)) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
  sorry

end NUMINAMATH_CALUDE_intersection_reciprocal_sum_l1710_171057


namespace NUMINAMATH_CALUDE_packages_per_truck_l1710_171075

theorem packages_per_truck (total_packages : ℕ) (num_trucks : ℕ) 
  (h1 : total_packages = 490) (h2 : num_trucks = 7) :
  total_packages / num_trucks = 70 := by
  sorry

end NUMINAMATH_CALUDE_packages_per_truck_l1710_171075


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l1710_171054

theorem quadratic_minimum_value : 
  ∀ x : ℝ, 3 * x^2 - 18 * x + 12 ≥ -15 ∧ 
  ∃ x : ℝ, 3 * x^2 - 18 * x + 12 = -15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l1710_171054


namespace NUMINAMATH_CALUDE_same_type_monomials_result_l1710_171069

/-- 
Given two monomials of the same type: -x^3 * y^a and 6x^b * y,
prove that (a - b)^3 = -8
-/
theorem same_type_monomials_result (a b : ℤ) : 
  (∀ x y : ℝ, -x^3 * y^a = 6 * x^b * y) → (a - b)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_same_type_monomials_result_l1710_171069


namespace NUMINAMATH_CALUDE_all_flowers_bloom_monday_l1710_171019

-- Define the days of the week
inductive Day : Type
| monday : Day
| tuesday : Day
| wednesday : Day
| thursday : Day
| friday : Day
| saturday : Day
| sunday : Day

-- Define the flower types
inductive Flower : Type
| sunflower : Flower
| lily : Flower
| peony : Flower

-- Define a function to check if a flower blooms on a given day
def blooms (f : Flower) (d : Day) : Prop := sorry

-- Define the conditions
axiom one_day_all_bloom : ∃! d : Day, ∀ f : Flower, blooms f d

axiom no_three_consecutive_days : 
  ∀ f : Flower, ∀ d1 d2 d3 : Day, 
    (blooms f d1 ∧ blooms f d2 ∧ blooms f d3) → 
    (d1 ≠ Day.monday ∨ d2 ≠ Day.tuesday ∨ d3 ≠ Day.wednesday) ∧
    (d1 ≠ Day.tuesday ∨ d2 ≠ Day.wednesday ∨ d3 ≠ Day.thursday) ∧
    (d1 ≠ Day.wednesday ∨ d2 ≠ Day.thursday ∨ d3 ≠ Day.friday) ∧
    (d1 ≠ Day.thursday ∨ d2 ≠ Day.friday ∨ d3 ≠ Day.saturday) ∧
    (d1 ≠ Day.friday ∨ d2 ≠ Day.saturday ∨ d3 ≠ Day.sunday) ∧
    (d1 ≠ Day.saturday ∨ d2 ≠ Day.sunday ∨ d3 ≠ Day.monday) ∧
    (d1 ≠ Day.sunday ∨ d2 ≠ Day.monday ∨ d3 ≠ Day.tuesday)

axiom two_flowers_not_bloom : 
  ∀ f1 f2 : Flower, f1 ≠ f2 → 
    (∃! d : Day, ¬(blooms f1 d ∧ blooms f2 d))

axiom sunflowers_not_bloom : 
  ¬blooms Flower.sunflower Day.tuesday ∧ 
  ¬blooms Flower.sunflower Day.thursday ∧ 
  ¬blooms Flower.sunflower Day.sunday

axiom lilies_not_bloom : 
  ¬blooms Flower.lily Day.thursday ∧ 
  ¬blooms Flower.lily Day.saturday

axiom peonies_not_bloom : 
  ¬blooms Flower.peony Day.sunday

-- The theorem to prove
theorem all_flowers_bloom_monday : 
  ∀ f : Flower, blooms f Day.monday ∧ 
  (∀ d : Day, d ≠ Day.monday → ¬(∀ f : Flower, blooms f d)) :=
by sorry

end NUMINAMATH_CALUDE_all_flowers_bloom_monday_l1710_171019


namespace NUMINAMATH_CALUDE_solve_for_b_l1710_171004

theorem solve_for_b (a b : ℝ) (h1 : a * b = 2 * (a + b) + 10) (h2 : b - a = 5) : b = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l1710_171004


namespace NUMINAMATH_CALUDE_rebeccas_haircut_price_l1710_171010

/-- Rebecca's hair salon pricing and earnings --/
theorem rebeccas_haircut_price 
  (perm_price : ℕ) 
  (dye_job_price : ℕ) 
  (dye_cost : ℕ) 
  (haircuts : ℕ) 
  (perms : ℕ) 
  (dye_jobs : ℕ) 
  (tips : ℕ) 
  (total_earnings : ℕ) 
  (h : perm_price = 40) 
  (i : dye_job_price = 60) 
  (j : dye_cost = 10) 
  (k : haircuts = 4) 
  (l : perms = 1) 
  (m : dye_jobs = 2) 
  (n : tips = 50) 
  (o : total_earnings = 310) : 
  ∃ (haircut_price : ℕ), 
    haircut_price * haircuts + 
    perm_price * perms + 
    dye_job_price * dye_jobs + 
    tips - 
    dye_cost * dye_jobs = total_earnings ∧ 
    haircut_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_rebeccas_haircut_price_l1710_171010


namespace NUMINAMATH_CALUDE_action_figures_total_l1710_171095

theorem action_figures_total (initial : ℕ) (added : ℕ) : 
  initial = 8 → added = 2 → initial + added = 10 := by
sorry

end NUMINAMATH_CALUDE_action_figures_total_l1710_171095


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l1710_171031

theorem absolute_value_not_positive (x : ℚ) : ¬(|2*x - 7| > 0) ↔ x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l1710_171031


namespace NUMINAMATH_CALUDE_problem_statement_l1710_171062

theorem problem_statement (a b : ℝ) (hab : a * b > 0) (hab2 : a^2 * b = 4) :
  (∃ (m : ℝ), ∀ (a b : ℝ), a * b > 0 → a^2 * b = 4 → a + b ≥ m ∧ 
    ∀ (m' : ℝ), (∀ (a b : ℝ), a * b > 0 → a^2 * b = 4 → a + b ≥ m') → m' ≤ m) ∧
  (∀ (x : ℝ), 2 * |x - 1| + |x| ≤ a + b ↔ -1/3 ≤ x ∧ x ≤ 5/3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1710_171062


namespace NUMINAMATH_CALUDE_odd_sum_not_divisible_by_three_l1710_171086

theorem odd_sum_not_divisible_by_three (x y z : ℕ) 
  (h_odd_x : Odd x) (h_odd_y : Odd y) (h_odd_z : Odd z)
  (h_positive_x : x > 0) (h_positive_y : y > 0) (h_positive_z : z > 0)
  (h_gcd : Nat.gcd x (Nat.gcd y z) = 1)
  (h_divisible : (x^2 + y^2 + z^2) % (x + y + z) = 0) :
  ¬(((x + y + z) - 2) % 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_not_divisible_by_three_l1710_171086


namespace NUMINAMATH_CALUDE_exists_common_divisor_l1710_171022

/-- A function from positive integers to positive integers greater than 1 -/
def PositiveIntegerFunction : Type := ℕ+ → ℕ+

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n)) ∣ (f m + f n)

/-- The theorem stating that there exists a common divisor greater than 1 for all values of f -/
theorem exists_common_divisor (f : PositiveIntegerFunction) 
  (h : HasDivisibilityProperty f) : 
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, c ∣ f n := by
  sorry

end NUMINAMATH_CALUDE_exists_common_divisor_l1710_171022


namespace NUMINAMATH_CALUDE_lines_perpendicular_l1710_171016

-- Define the slopes of the two lines
def slope1 : ℚ := 3 / 4
def slope2 : ℚ := -4 / 3

-- Define the equations of the two lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 16
def line2 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Theorem: The two lines are perpendicular
theorem lines_perpendicular : slope1 * slope2 = -1 := by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l1710_171016


namespace NUMINAMATH_CALUDE_cubic_equation_root_b_value_l1710_171020

theorem cubic_equation_root_b_value :
  ∀ (a b : ℚ),
  (∃ (x : ℂ), x = 1 + Real.sqrt 2 ∧ x^3 + a*x^2 + b*x + 6 = 0) →
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_b_value_l1710_171020


namespace NUMINAMATH_CALUDE_triplet_satisfies_equations_l1710_171009

theorem triplet_satisfies_equations : ∃ (x y z : ℂ),
  x + y + z = 5 ∧
  x^2 + y^2 + z^2 = 19 ∧
  x^3 + y^3 + z^3 = 53 ∧
  x = -1 ∧ y = Complex.I * Real.sqrt 3 ∧ z = -Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triplet_satisfies_equations_l1710_171009


namespace NUMINAMATH_CALUDE_B_equals_zero_one_two_l1710_171059

def B : Set ℤ := {x | -3 < 2*x - 1 ∧ 2*x - 1 < 5}

theorem B_equals_zero_one_two : B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_B_equals_zero_one_two_l1710_171059


namespace NUMINAMATH_CALUDE_truncated_cone_volume_l1710_171045

/-- The volume of a truncated cone with specific diagonal properties -/
theorem truncated_cone_volume 
  (l : ℝ) 
  (α : ℝ) 
  (h_positive : l > 0)
  (h_angle : 0 < α ∧ α < π)
  (h_diagonal_ratio : ∃ (k : ℝ), k > 0 ∧ 2 * k = l ∧ k = l / 3)
  : ∃ (V : ℝ), V = (7 / 54) * π * l^3 * Real.sin α * Real.sin (α / 2) :=
sorry

end NUMINAMATH_CALUDE_truncated_cone_volume_l1710_171045


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_local_max_l1710_171012

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x + a * (x - 1)^2

-- Theorem for part 1
theorem max_value_when_a_zero :
  ∃ (x : ℝ), ∀ (y : ℝ), f 0 y ≤ f 0 x ∧ f 0 x = 1 / Real.exp 1 :=
sorry

-- Theorem for part 2
theorem range_of_a_for_local_max :
  ∀ (a : ℝ), (∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ f a x ∧ f a x ≤ 1/2) ↔
  (a < 1 / (2 * Real.exp 1) ∨ (a > 1 / (2 * Real.exp 1) ∧ a ≤ 1/2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_when_a_zero_range_of_a_for_local_max_l1710_171012


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_slope_sum_l1710_171080

-- Define the trapezoid ABCD
structure IsoscelesTrapezoid where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ
  D : ℤ × ℤ

-- Define the conditions
def validTrapezoid (t : IsoscelesTrapezoid) : Prop :=
  t.A = (15, 15) ∧
  t.D = (16, 20) ∧
  t.B.1 ≠ t.A.1 ∧ t.B.2 ≠ t.A.2 ∧  -- No horizontal or vertical sides
  t.C.1 ≠ t.D.1 ∧ t.C.2 ≠ t.D.2 ∧
  (t.B.2 - t.A.2) * (t.D.1 - t.C.1) = (t.B.1 - t.A.1) * (t.D.2 - t.C.2) ∧  -- AB || CD
  (t.C.2 - t.B.2) * (t.D.1 - t.A.1) ≠ (t.C.1 - t.B.1) * (t.D.2 - t.A.2) ∧  -- BC not || AD
  (t.D.2 - t.A.2) * (t.C.1 - t.B.1) ≠ (t.D.1 - t.A.1) * (t.C.2 - t.B.2)    -- CD not || AB

-- Define the slope of AB
def slopeAB (t : IsoscelesTrapezoid) : ℚ :=
  (t.B.2 - t.A.2) / (t.B.1 - t.A.1)

-- Define the theorem
theorem isosceles_trapezoid_slope_sum (t : IsoscelesTrapezoid) 
  (h : validTrapezoid t) : 
  ∃ (slopes : List ℚ), (∀ s ∈ slopes, ∃ t' : IsoscelesTrapezoid, validTrapezoid t' ∧ slopeAB t' = s) ∧
                       slopes.sum = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_slope_sum_l1710_171080


namespace NUMINAMATH_CALUDE_lecture_slides_theorem_our_lecture_slides_l1710_171058

/-- Represents a lecture with slides -/
structure Lecture where
  duration : ℕ  -- Duration of the lecture in minutes
  initial_slides : ℕ  -- Number of slides changed in the initial period
  initial_period : ℕ  -- Initial period in minutes
  total_slides : ℕ  -- Total number of slides used

/-- Calculates the total number of slides used in a lecture -/
def calculate_total_slides (l : Lecture) : ℕ :=
  (l.duration * l.initial_slides) / l.initial_period

/-- Theorem stating that for the given lecture conditions, the total slides used is 100 -/
theorem lecture_slides_theorem (l : Lecture) 
  (h1 : l.duration = 50)
  (h2 : l.initial_slides = 4)
  (h3 : l.initial_period = 2) :
  calculate_total_slides l = 100 := by
  sorry

/-- The specific lecture instance -/
def our_lecture : Lecture := {
  duration := 50,
  initial_slides := 4,
  initial_period := 2,
  total_slides := 100
}

/-- Proof that our specific lecture uses 100 slides -/
theorem our_lecture_slides : 
  calculate_total_slides our_lecture = 100 := by
  sorry

end NUMINAMATH_CALUDE_lecture_slides_theorem_our_lecture_slides_l1710_171058


namespace NUMINAMATH_CALUDE_problem_solution_l1710_171017

theorem problem_solution (x y : ℚ) 
  (eq1 : 102 * x - 5 * y = 25) 
  (eq2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1710_171017


namespace NUMINAMATH_CALUDE_palindrome_count_is_420_l1710_171070

/-- Represents the count of each digit available -/
def digit_counts : List (Nat × Nat) := [(2, 2), (3, 3), (5, 4)]

/-- The total number of digits available -/
def total_digits : Nat := (digit_counts.map Prod.snd).sum

/-- A function to calculate the number of 9-digit palindromes -/
def count_palindromes (counts : List (Nat × Nat)) : Nat :=
  sorry

theorem palindrome_count_is_420 :
  total_digits = 9 ∧ count_palindromes digit_counts = 420 :=
sorry

end NUMINAMATH_CALUDE_palindrome_count_is_420_l1710_171070


namespace NUMINAMATH_CALUDE_distinct_primes_count_l1710_171078

theorem distinct_primes_count (n : ℕ) : n = 95 * 97 * 99 * 101 * 103 → 
  (Finset.card (Nat.factors n).toFinset) = 7 := by
sorry

end NUMINAMATH_CALUDE_distinct_primes_count_l1710_171078


namespace NUMINAMATH_CALUDE_circle_center_l1710_171076

/-- The center of a circle defined by the equation 4x^2 - 8x + 4y^2 - 16y + 20 = 0 is (1, 2) -/
theorem circle_center (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) → 
  (∃ (h : ℝ), h = 0 ∧ (x - 1)^2 + (y - 2)^2 = h) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l1710_171076


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1710_171050

theorem polynomial_division_remainder : ∃ (q : Polynomial ℝ), 
  x^6 - x^5 - x^4 + x^3 + x^2 - x = (x^2 - 4) * (x + 1) * q + (21*x^2 - 13*x - 32) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1710_171050


namespace NUMINAMATH_CALUDE_next_two_juicy_numbers_l1710_171024

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a * b * c * d = n ∧ 1 = 1/a + 1/b + 1/c + 1/d

theorem next_two_juicy_numbers :
  (∀ n < 6, ¬ is_juicy n) ∧
  is_juicy 6 ∧
  is_juicy 12 ∧
  is_juicy 20 ∧
  (∀ n, 6 < n ∧ n < 12 → ¬ is_juicy n) ∧
  (∀ n, 12 < n ∧ n < 20 → ¬ is_juicy n) :=
sorry

end NUMINAMATH_CALUDE_next_two_juicy_numbers_l1710_171024


namespace NUMINAMATH_CALUDE_office_network_connections_l1710_171026

/-- Represents a computer network with switches and connections -/
structure ComputerNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  num_crucial_switches : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : ComputerNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2 + network.num_crucial_switches

/-- Theorem: The total number of connections in the given network is 65 -/
theorem office_network_connections :
  let network : ComputerNetwork := {
    num_switches := 30,
    connections_per_switch := 4,
    num_crucial_switches := 5
  }
  total_connections network = 65 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l1710_171026


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1710_171037

/-- Definition of the set of points M(x, y) satisfying the given equation -/
def TrajectorySet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt (x^2 + (y-3)^2) + Real.sqrt (x^2 + (y+3)^2) = 10}

/-- Definition of an ellipse with foci (0, -3) and (0, 3), and major axis length 10 -/
def EllipseSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; 
               Real.sqrt (x^2 + (y+3)^2) + Real.sqrt (x^2 + (y-3)^2) = 10}

/-- Theorem stating that the trajectory set is equivalent to the ellipse set -/
theorem trajectory_is_ellipse : TrajectorySet = EllipseSet := by
  sorry


end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1710_171037


namespace NUMINAMATH_CALUDE_disjoint_subsets_equal_sum_l1710_171055

theorem disjoint_subsets_equal_sum (n : ℕ) (A : Finset ℕ) : 
  A.card = n → 
  (∀ a ∈ A, a > 0) → 
  A.sum id < 2^n - 1 → 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ B.sum id = C.sum id :=
sorry

end NUMINAMATH_CALUDE_disjoint_subsets_equal_sum_l1710_171055


namespace NUMINAMATH_CALUDE_goldfish_fed_by_four_scoops_l1710_171027

/-- The number of goldfish that can be fed by one scoop of fish food -/
def goldfish_per_scoop : ℕ := 8

/-- The number of scoops of fish food -/
def number_of_scoops : ℕ := 4

/-- Theorem: 4 scoops of fish food can feed 32 goldfish -/
theorem goldfish_fed_by_four_scoops : 
  number_of_scoops * goldfish_per_scoop = 32 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_fed_by_four_scoops_l1710_171027


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1710_171081

/-- Given a cubic polynomial P(X) = X^3 - 3X^2 - 1 with roots r₁, r₂, r₃,
    prove that the sum of the cubes of the roots is 24. -/
theorem sum_of_cubes_of_roots (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 + r₁^2 * 3 + 1 = 0) → 
  (r₂^3 + r₂^2 * 3 + 1 = 0) → 
  (r₃^3 + r₃^2 * 3 + 1 = 0) → 
  r₁^3 + r₂^3 + r₃^3 = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1710_171081


namespace NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_triangle_perimeter_l1710_171082

/-- Represents an equilateral triangle with inscribed circles -/
structure TriangleWithCircles where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The radius of each inscribed circle -/
  circle_radius : ℝ
  /-- Assumption that the circles touch two sides of the triangle and each other -/
  circles_touch_sides_and_each_other : True

/-- Theorem stating the perimeter of the triangle given the inscribed circles -/
theorem triangle_perimeter_with_inscribed_circles
  (t : TriangleWithCircles)
  (h : t.circle_radius = 2) :
  t.side_length = 2 * Real.sqrt 3 + 4 :=
sorry

/-- Corollary calculating the perimeter of the triangle -/
theorem triangle_perimeter
  (t : TriangleWithCircles)
  (h : t.circle_radius = 2) :
  3 * t.side_length = 6 * Real.sqrt 3 + 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_triangle_perimeter_l1710_171082


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1710_171030

theorem absolute_value_inequality (x : ℝ) :
  |x^2 - 5*x + 3| < 9 ↔ (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1710_171030


namespace NUMINAMATH_CALUDE_bruno_coconut_capacity_l1710_171083

theorem bruno_coconut_capacity (total_coconuts : ℕ) (barbie_capacity : ℕ) (total_trips : ℕ) 
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : total_trips = 12) :
  (total_coconuts - barbie_capacity * total_trips) / total_trips = 8 := by
  sorry

end NUMINAMATH_CALUDE_bruno_coconut_capacity_l1710_171083


namespace NUMINAMATH_CALUDE_photo_album_and_film_prices_l1710_171056

theorem photo_album_and_film_prices :
  ∀ (x y : ℚ),
    5 * x + 4 * y = 139 →
    4 * x + 5 * y = 140 →
    x = 15 ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_photo_album_and_film_prices_l1710_171056


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1710_171005

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, if 0 < a_1 < a_2, then a_2 > √(a_1 * a_3) -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  0 < a 1 → a 1 < a 2 → a 2 > Real.sqrt (a 1 * a 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1710_171005


namespace NUMINAMATH_CALUDE_no_positive_integer_divisible_by_its_square_plus_one_l1710_171042

theorem no_positive_integer_divisible_by_its_square_plus_one :
  ∀ n : ℕ, n > 0 → ¬(n^2 + 1 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_divisible_by_its_square_plus_one_l1710_171042


namespace NUMINAMATH_CALUDE_c_share_is_56_l1710_171051

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the money distribution problem -/
def moneyDistribution (s : Share) : Prop :=
  s.b = 0.65 * s.a ∧ 
  s.c = 0.40 * s.a ∧ 
  s.a + s.b + s.c = 287

/-- Theorem stating that under the given conditions, C's share is 56 -/
theorem c_share_is_56 :
  ∃ s : Share, moneyDistribution s ∧ s.c = 56 := by
  sorry


end NUMINAMATH_CALUDE_c_share_is_56_l1710_171051


namespace NUMINAMATH_CALUDE_five_solutions_l1710_171032

/-- The system of equations has exactly 5 real solutions -/
theorem five_solutions (x y z w : ℝ) : 
  (x = z + w + z*w*x ∧
   y = w + x + w*x*y ∧
   z = x + y + x*y*z ∧
   w = y + z + y*z*w) →
  ∃! (sol : Finset (ℝ × ℝ × ℝ × ℝ)), 
    sol.card = 5 ∧ 
    ∀ (a b c d : ℝ), (a, b, c, d) ∈ sol ↔ 
      (a = c + d + c*d*a ∧
       b = d + a + d*a*b ∧
       c = a + b + a*b*c ∧
       d = b + c + b*c*d) :=
by sorry

end NUMINAMATH_CALUDE_five_solutions_l1710_171032


namespace NUMINAMATH_CALUDE_flour_amount_l1710_171003

def recipe_flour (flour_added : ℕ) (flour_to_add : ℕ) : ℕ :=
  flour_added + flour_to_add

theorem flour_amount : recipe_flour 6 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_flour_amount_l1710_171003


namespace NUMINAMATH_CALUDE_problem_statement_l1710_171074

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ^ b = 343) (h5 : b ^ c = 10) (h6 : a ^ c = 7) : b ^ b = 1000 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1710_171074


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l1710_171088

def Q (n : ℕ) : ℚ :=
  (3^(n-1) : ℚ) / ((3*n - 2 : ℕ).factorial * n.factorial)

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → Q k < 1/1500 ↔ k ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l1710_171088


namespace NUMINAMATH_CALUDE_xiaoming_red_pens_l1710_171048

/-- The number of red pens bought by Xiaoming -/
def red_pens : ℕ := 36

/-- The total number of pens bought -/
def total_pens : ℕ := 66

/-- The original price of a red pen in yuan -/
def red_pen_price : ℚ := 5

/-- The original price of a black pen in yuan -/
def black_pen_price : ℚ := 9

/-- The discount rate for red pens -/
def red_discount : ℚ := 85 / 100

/-- The discount rate for black pens -/
def black_discount : ℚ := 80 / 100

/-- The discount rate on the total price -/
def total_discount : ℚ := 18 / 100

theorem xiaoming_red_pens :
  red_pens = 36 ∧
  red_pens ≤ total_pens ∧
  (red_pen_price * red_pens + black_pen_price * (total_pens - red_pens)) * (1 - total_discount) =
  red_pen_price * red_discount * red_pens + black_pen_price * black_discount * (total_pens - red_pens) :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_red_pens_l1710_171048


namespace NUMINAMATH_CALUDE_percentage_problem_l1710_171041

theorem percentage_problem (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = 0.3 * (x + y) →
  y = 0.4 * x →
  P = 70 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1710_171041


namespace NUMINAMATH_CALUDE_greatest_three_digit_non_divisor_l1710_171064

theorem greatest_three_digit_non_divisor : ∃ n : ℕ, 
  n = 998 ∧ 
  n ≥ 100 ∧ n < 1000 ∧ 
  ∀ m : ℕ, m > n → m < 1000 → 
    (m * (m + 1) / 2 ∣ Nat.factorial (m - 1)) ∧
  ¬(n * (n + 1) / 2 ∣ Nat.factorial (n - 1)) := by
  sorry

#check greatest_three_digit_non_divisor

end NUMINAMATH_CALUDE_greatest_three_digit_non_divisor_l1710_171064


namespace NUMINAMATH_CALUDE_cubic_root_implies_p_value_l1710_171014

theorem cubic_root_implies_p_value : ∀ p : ℝ, (3 : ℝ)^3 + p * 3 - 18 = 0 → p = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_p_value_l1710_171014


namespace NUMINAMATH_CALUDE_max_a_value_l1710_171077

theorem max_a_value (a b : ℕ) (h : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) :
  a ≤ 20 ∧ ∃ b₀ : ℕ, 5 * Nat.lcm 20 b₀ + 2 * Nat.gcd 20 b₀ = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l1710_171077


namespace NUMINAMATH_CALUDE_f_3_equals_3_l1710_171073

def f (x : ℝ) : ℝ := 2 * (x - 1) - 1

theorem f_3_equals_3 : f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_3_l1710_171073


namespace NUMINAMATH_CALUDE_min_groups_for_30_students_max_12_l1710_171097

/-- Given a total number of students and a maximum group size, 
    calculate the minimum number of equal-sized groups. -/
def min_groups (total_students : ℕ) (max_group_size : ℕ) : ℕ :=
  let divisors := (Finset.range total_students).filter (λ d => total_students % d = 0)
  let valid_divisors := divisors.filter (λ d => d ≤ max_group_size)
  total_students / valid_divisors.max' (by sorry)

/-- The theorem stating that for 30 students and a maximum group size of 12, 
    the minimum number of equal-sized groups is 3. -/
theorem min_groups_for_30_students_max_12 :
  min_groups 30 12 = 3 := by sorry

end NUMINAMATH_CALUDE_min_groups_for_30_students_max_12_l1710_171097


namespace NUMINAMATH_CALUDE_quadratic_equation_negative_root_l1710_171028

theorem quadratic_equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (0 < a ∧ a ≤ 1) ∨ a < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_negative_root_l1710_171028


namespace NUMINAMATH_CALUDE_older_friend_age_l1710_171039

theorem older_friend_age (younger_age older_age : ℕ) : 
  older_age - younger_age = 2 →
  younger_age + older_age = 74 →
  older_age = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_older_friend_age_l1710_171039


namespace NUMINAMATH_CALUDE_grocery_store_costs_l1710_171015

/-- Calculates the money paid for orders given total costs and fractions for salary and delivery --/
def money_paid_for_orders (total_costs : ℝ) (salary_fraction : ℝ) (delivery_fraction : ℝ) : ℝ :=
  let salary := salary_fraction * total_costs
  let remaining := total_costs - salary
  let delivery := delivery_fraction * remaining
  total_costs - salary - delivery

/-- Proves that given the specified conditions, the money paid for orders is $1800 --/
theorem grocery_store_costs : 
  money_paid_for_orders 4000 (2/5) (1/4) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_costs_l1710_171015


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1710_171043

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x + y = 7 * x * y → 1 / x + 1 / y = 7 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1710_171043


namespace NUMINAMATH_CALUDE_carly_job_applications_l1710_171034

theorem carly_job_applications : ∃ (x : ℕ), x + 2*x = 600 ∧ x = 200 := by sorry

end NUMINAMATH_CALUDE_carly_job_applications_l1710_171034


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l1710_171091

theorem greatest_integer_problem : 
  ∃ (m : ℕ), 
    (m < 150) ∧ 
    (∃ (a : ℕ), m = 9 * a - 2) ∧ 
    (∃ (b : ℕ), m = 5 * b + 4) ∧ 
    (∀ (n : ℕ), 
      (n < 150) → 
      (∃ (c : ℕ), n = 9 * c - 2) → 
      (∃ (d : ℕ), n = 5 * d + 4) → 
      n ≤ m) ∧
    m = 124 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l1710_171091


namespace NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l1710_171001

-- Define the base 10 number
def base_10_num : ℕ := 3500

-- Define the base 7 representation
def base_7_repr : List ℕ := [1, 3, 1, 3, 0]

-- Function to convert a list of digits in base 7 to a natural number
def to_nat (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem stating the equivalence
theorem base_10_to_base_7_conversion :
  base_10_num = to_nat base_7_repr :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l1710_171001
