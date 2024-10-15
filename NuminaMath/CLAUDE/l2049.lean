import Mathlib

namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2049_204927

/-- An angle is in the third quadrant if it's between 180° and 270° (modulo 360°) -/
def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

/-- An angle is in the second quadrant if it's between 90° and 180° (modulo 360°) -/
def is_in_second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 180

/-- An angle is in the fourth quadrant if it's between 270° and 360° (modulo 360°) -/
def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 360

theorem half_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_quadrant (α/2) ∨ is_in_fourth_quadrant (α/2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2049_204927


namespace NUMINAMATH_CALUDE_fifi_green_hangers_l2049_204928

/-- The number of green hangers in Fifi's closet -/
def green_hangers : ℕ := 4

/-- The number of pink hangers in Fifi's closet -/
def pink_hangers : ℕ := 7

/-- The number of blue hangers in Fifi's closet -/
def blue_hangers : ℕ := green_hangers - 1

/-- The number of yellow hangers in Fifi's closet -/
def yellow_hangers : ℕ := blue_hangers - 1

/-- The total number of hangers in Fifi's closet -/
def total_hangers : ℕ := 16

theorem fifi_green_hangers :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = total_hangers :=
by sorry

end NUMINAMATH_CALUDE_fifi_green_hangers_l2049_204928


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2049_204933

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference is 4 if 2S_3 - 3S_2 = 12 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- Definition of S_n
  (h2 : 2 * S 3 - 3 * S 2 = 12)  -- Given condition
  : a 2 - a 1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2049_204933


namespace NUMINAMATH_CALUDE_apple_juice_production_l2049_204984

/-- Given the annual U.S. apple production and its distribution, calculate the amount used for juice -/
theorem apple_juice_production (total_production : ℝ) (cider_percentage : ℝ) (juice_percentage : ℝ) :
  total_production = 7 →
  cider_percentage = 0.25 →
  juice_percentage = 0.60 →
  (total_production * (1 - cider_percentage) * juice_percentage) = 3.15 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_production_l2049_204984


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l2049_204909

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_sum : i + i^3 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l2049_204909


namespace NUMINAMATH_CALUDE_burgerCaloriesTheorem_l2049_204975

/-- Calculates the total calories consumed over a number of days, given the number of burgers eaten per day and calories per burger. -/
def totalCalories (burgersPerDay : ℕ) (caloriesPerBurger : ℕ) (days : ℕ) : ℕ :=
  burgersPerDay * caloriesPerBurger * days

/-- Theorem stating that eating 3 burgers per day, with 20 calories per burger, results in 120 calories consumed after two days. -/
theorem burgerCaloriesTheorem : totalCalories 3 20 2 = 120 := by
  sorry


end NUMINAMATH_CALUDE_burgerCaloriesTheorem_l2049_204975


namespace NUMINAMATH_CALUDE_xyz_bounds_l2049_204946

-- Define the problem
theorem xyz_bounds (x y z a : ℝ) (ha : a > 0) 
  (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = a^2 / 2) :
  (0 ≤ x ∧ x ≤ 2*a/3) ∧ (0 ≤ y ∧ y ≤ 2*a/3) ∧ (0 ≤ z ∧ z ≤ 2*a/3) := by
  sorry

end NUMINAMATH_CALUDE_xyz_bounds_l2049_204946


namespace NUMINAMATH_CALUDE_perfect_square_primes_l2049_204942

theorem perfect_square_primes (p : ℕ) : 
  Nat.Prime p ∧ ∃ (n : ℕ), (2^(p+1) - 4) / p = n^2 ↔ p = 3 ∨ p = 7 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_primes_l2049_204942


namespace NUMINAMATH_CALUDE_smallest_a_divisibility_l2049_204957

theorem smallest_a_divisibility : 
  ∃ (n : ℕ), 
    n % 2 = 1 ∧ 
    (55^n + 2000 * 32^n) % 2001 = 0 ∧ 
    ∀ (a : ℕ), a > 0 → a < 2000 → 
      ∀ (m : ℕ), m % 2 = 1 → (55^m + a * 32^m) % 2001 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_divisibility_l2049_204957


namespace NUMINAMATH_CALUDE_prob_three_odd_six_dice_value_l2049_204990

/-- The probability of rolling an odd number on a fair 8-sided die -/
def prob_odd_8sided : ℚ := 1/2

/-- The number of ways to choose 3 dice out of 6 -/
def choose_3_from_6 : ℕ := 20

/-- The probability of rolling exactly three odd numbers when rolling six fair 8-sided dice -/
def prob_three_odd_six_dice : ℚ :=
  (choose_3_from_6 : ℚ) * (prob_odd_8sided ^ 3 * (1 - prob_odd_8sided) ^ 3)

theorem prob_three_odd_six_dice_value : prob_three_odd_six_dice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_six_dice_value_l2049_204990


namespace NUMINAMATH_CALUDE_fraction_simplification_l2049_204982

theorem fraction_simplification (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) :
  x / (x - y) - y / (x + y) = (x^2 + y^2) / (x^2 - y^2) := by
  sorry


end NUMINAMATH_CALUDE_fraction_simplification_l2049_204982


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2049_204978

theorem rationalize_denominator :
  ∃ (A B C : ℤ),
    (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C ∧
    A = -9 ∧
    B = -4 ∧
    C = 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2049_204978


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2049_204925

/-- Calculates the price of a stock given the income, dividend rate, and investment amount. -/
theorem stock_price_calculation (income : ℝ) (dividend_rate : ℝ) (investment : ℝ) :
  income = 450 →
  dividend_rate = 0.1 →
  investment = 4860 →
  (investment / (income / dividend_rate)) * 100 = 108 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2049_204925


namespace NUMINAMATH_CALUDE_total_miles_walked_l2049_204986

/-- The number of ladies in the walking group -/
def num_ladies : ℕ := 5

/-- The number of miles walked together by the group each day -/
def group_miles_per_day : ℕ := 3

/-- The number of additional miles Jamie walks per day -/
def jamie_additional_miles_per_day : ℕ := 2

/-- The number of days they walk per week -/
def days_per_week : ℕ := 6

/-- The total miles walked by the ladies in 6 days -/
def total_miles : ℕ := num_ladies * group_miles_per_day * days_per_week + jamie_additional_miles_per_day * days_per_week

theorem total_miles_walked :
  total_miles = 120 := by sorry

end NUMINAMATH_CALUDE_total_miles_walked_l2049_204986


namespace NUMINAMATH_CALUDE_observed_price_in_local_currency_l2049_204994

-- Define constants for the given conditions
def producer_cost : ℝ := 19
def shipping_cost : ℝ := 5
def tax_rate : ℝ := 0.10
def commission_rate : ℝ := 0.20
def exchange_rate : ℝ := 0.90
def profit_rate : ℝ := 0.20

-- Define the theorem
theorem observed_price_in_local_currency :
  let base_cost := producer_cost + shipping_cost
  let total_cost := base_cost + tax_rate * base_cost
  let profit := profit_rate * total_cost
  let price_before_commission := total_cost + profit
  let distributor_price := price_before_commission / (1 - commission_rate)
  let local_price := distributor_price * exchange_rate
  local_price = 35.64 := by sorry

end NUMINAMATH_CALUDE_observed_price_in_local_currency_l2049_204994


namespace NUMINAMATH_CALUDE_shooter_probability_l2049_204956

theorem shooter_probability (p10 p9 p8 : ℝ) 
  (h1 : p10 = 0.2) 
  (h2 : p9 = 0.3) 
  (h3 : p8 = 0.1) : 
  1 - (p10 + p9) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l2049_204956


namespace NUMINAMATH_CALUDE_square_b_minus_d_l2049_204948

theorem square_b_minus_d (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 3) : 
  (b - d)^2 = 25 := by sorry

end NUMINAMATH_CALUDE_square_b_minus_d_l2049_204948


namespace NUMINAMATH_CALUDE_four_digit_sum_reverse_equals_4983_l2049_204969

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_sum_reverse_equals_4983 :
  ∃ (n : ℕ), is_four_digit n ∧ n + reverse_number n = 4983 :=
sorry

end NUMINAMATH_CALUDE_four_digit_sum_reverse_equals_4983_l2049_204969


namespace NUMINAMATH_CALUDE_carreys_fixed_amount_is_20_l2049_204950

/-- The fixed amount Carrey paid for the car rental -/
def carreys_fixed_amount : ℝ := 20

/-- The rate per kilometer for Carrey's rental -/
def carreys_rate_per_km : ℝ := 0.25

/-- The fixed amount Samuel paid for the car rental -/
def samuels_fixed_amount : ℝ := 24

/-- The rate per kilometer for Samuel's rental -/
def samuels_rate_per_km : ℝ := 0.16

/-- The number of kilometers driven by both Carrey and Samuel -/
def kilometers_driven : ℝ := 44.44444444444444

theorem carreys_fixed_amount_is_20 :
  carreys_fixed_amount + carreys_rate_per_km * kilometers_driven =
  samuels_fixed_amount + samuels_rate_per_km * kilometers_driven :=
sorry

end NUMINAMATH_CALUDE_carreys_fixed_amount_is_20_l2049_204950


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2049_204934

theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_imaginary_axis : 2 * b = 2) (h_focal_length : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 3) :
  ∃ k : ℝ, k = b / a ∧ k = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2049_204934


namespace NUMINAMATH_CALUDE_least_prime_factor_11_5_minus_11_4_l2049_204947

theorem least_prime_factor_11_5_minus_11_4 :
  Nat.minFac (11^5 - 11^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_11_5_minus_11_4_l2049_204947


namespace NUMINAMATH_CALUDE_area_of_circumscribed_circle_l2049_204953

/-- An isosceles triangle with two sides of length 4 and base of length 3 -/
structure IsoscelesTriangle where
  sideLength : ℝ
  baseLength : ℝ
  isIsosceles : sideLength = 4 ∧ baseLength = 3

/-- A circle passing through the vertices of the triangle -/
structure CircumscribedCircle (t : IsoscelesTriangle) where
  radius : ℝ
  passesThrough : True  -- This is a placeholder for the property that the circle passes through all vertices

/-- The theorem stating that the area of the circumscribed circle is 16π -/
theorem area_of_circumscribed_circle (t : IsoscelesTriangle) (c : CircumscribedCircle t) :
  π * c.radius^2 = 16 * π := by sorry

end NUMINAMATH_CALUDE_area_of_circumscribed_circle_l2049_204953


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l2049_204985

def two_successive_discounts (d : ℝ) : ℝ := (1 - d) * (1 - d)
def three_successive_discounts (d : ℝ) : ℝ := (1 - d) * (1 - d) * (1 - d)
def two_different_discounts (d1 d2 : ℝ) : ℝ := (1 - d1) * (1 - d2)

theorem smallest_advantageous_discount : ∀ n : ℕ, n ≥ 34 →
  (1 - n / 100 < two_successive_discounts 0.18) ∧
  (1 - n / 100 < three_successive_discounts 0.12) ∧
  (1 - n / 100 < two_different_discounts 0.28 0.07) ∧
  (∀ m : ℕ, m < 34 →
    (1 - m / 100 ≥ two_successive_discounts 0.18) ∨
    (1 - m / 100 ≥ three_successive_discounts 0.12) ∨
    (1 - m / 100 ≥ two_different_discounts 0.28 0.07)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l2049_204985


namespace NUMINAMATH_CALUDE_determine_relationship_l2049_204980

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - x - 2 > 0}
def Q : Set ℝ := {x : ℝ | |x - 1| > 1}

-- Define the possible relationships
inductive Relationship
  | sufficient_not_necessary
  | necessary_not_sufficient
  | necessary_and_sufficient
  | neither_sufficient_nor_necessary

-- Theorem to prove
theorem determine_relationship : Relationship :=
  sorry

end NUMINAMATH_CALUDE_determine_relationship_l2049_204980


namespace NUMINAMATH_CALUDE_constant_t_equation_l2049_204976

theorem constant_t_equation (t : ℝ) : 
  (∀ x : ℝ, (3*x^2 - 4*x + 5)*(2*x^2 + t*x + 8) = 6*x^4 - 26*x^3 + 58*x^2 - 76*x + 40) ↔ 
  t = -6 := by
sorry

end NUMINAMATH_CALUDE_constant_t_equation_l2049_204976


namespace NUMINAMATH_CALUDE_no_equivalent_expressions_l2049_204998

theorem no_equivalent_expressions (x : ℝ) (h : x > 0) : 
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ 2*(y+1)^y) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ (y+1)^(2*y+2)) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ 2*(y+0.5*y)^y) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ (2*y+2)^(2*y+2)) :=
by sorry

end NUMINAMATH_CALUDE_no_equivalent_expressions_l2049_204998


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l2049_204991

-- Define the hyperbola
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem statement
theorem hyperbola_b_value (b : ℝ) :
  (b > 0) →
  (∀ x y : ℝ, hyperbola x y b ↔ asymptotes x y) →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l2049_204991


namespace NUMINAMATH_CALUDE_sin_double_alpha_l2049_204930

theorem sin_double_alpha (α : Real) (h : Real.cos (π / 4 - α) = 4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l2049_204930


namespace NUMINAMATH_CALUDE_a_plus_b_value_m_range_l2049_204912

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 < 0}

-- Theorem 1
theorem a_plus_b_value :
  ∀ a b : ℝ, A a b = {x | -1 ≤ x ∧ x ≤ 4} → a + b = -7 :=
sorry

-- Theorem 2
theorem m_range (a b : ℝ) :
  A a b = {x | -1 ≤ x ∧ x ≤ 4} →
  (∀ x : ℝ, x ∈ A a b → x ∉ B m) →
  m ≤ -3 ∨ m ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_m_range_l2049_204912


namespace NUMINAMATH_CALUDE_german_students_l2049_204961

theorem german_students (total : ℕ) (both : ℕ) (german : ℕ) (spanish : ℕ) :
  total = 30 ∧ 
  both = 2 ∧ 
  german + spanish + both = total ∧ 
  german = 3 * spanish →
  german - both = 20 := by
  sorry

end NUMINAMATH_CALUDE_german_students_l2049_204961


namespace NUMINAMATH_CALUDE_logarithm_comparison_l2049_204996

theorem logarithm_comparison : 
  (Real.log 3.4 / Real.log 3 < Real.log 8.5 / Real.log 3) ∧ 
  ¬(π^(-0.7) < π^(-0.9)) ∧ 
  ¬(Real.log 1.8 / Real.log 0.3 < Real.log 2.7 / Real.log 0.3) ∧ 
  ¬(0.99^2.7 < 0.99^3.5) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_comparison_l2049_204996


namespace NUMINAMATH_CALUDE_expression_simplification_l2049_204904

theorem expression_simplification (x y : ℝ) 
  (h : y = Real.sqrt (x - 3) + Real.sqrt (6 - 2*x) + 2) : 
  Real.sqrt (2*x) * Real.sqrt (x/y) * (Real.sqrt (y/x) + Real.sqrt (1/y)) = 
    Real.sqrt 6 + (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2049_204904


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2049_204932

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Theorem for the first part of the problem
theorem subset_condition (a : ℝ) : 
  A ⊆ (A ∩ B a) ↔ 4/3 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem for the second part of the problem
theorem disjoint_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ a ≤ 2/3 ∨ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2049_204932


namespace NUMINAMATH_CALUDE_lorry_speed_l2049_204914

/-- Calculates the speed of a lorry crossing a bridge -/
theorem lorry_speed (lorry_length bridge_length : ℝ) (crossing_time : ℝ) :
  lorry_length = 200 →
  bridge_length = 200 →
  crossing_time = 17.998560115190784 →
  ∃ (speed : ℝ), abs (speed - 80) < 0.01 ∧ 
  speed = (lorry_length + bridge_length) / crossing_time * 3.6 := by
  sorry

#check lorry_speed

end NUMINAMATH_CALUDE_lorry_speed_l2049_204914


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2049_204943

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2049_204943


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2049_204931

/-- Represents a cube with alternating colored corners -/
structure ColoredCube where
  sideLength : ℝ
  alternatingColors : Bool

/-- Calculates the volume of the tetrahedron formed by similarly colored vertices -/
def tetrahedronVolume (cube : ColoredCube) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem blue_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.sideLength = 8) 
  (h2 : cube.alternatingColors = true) : 
  tetrahedronVolume cube = 512 / 3 := by
    sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2049_204931


namespace NUMINAMATH_CALUDE_symmetric_point_simplification_l2049_204967

theorem symmetric_point_simplification (x : ℝ) :
  (∃ P : ℝ × ℝ, P = (x + 1, 2 * x - 1) ∧ 
   (∃ P' : ℝ × ℝ, P' = (-x - 1, -2 * x + 1) ∧ 
    P'.1 > 0 ∧ P'.2 > 0)) →
  |x - 3| - |1 - x| = 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_simplification_l2049_204967


namespace NUMINAMATH_CALUDE_solution_to_equation_l2049_204901

theorem solution_to_equation : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1.5 + Real.sqrt 1.5 ∧ x₂ = 1.5 - Real.sqrt 1.5) ∧ 
    (∀ x : ℝ, x^4 + (3 - x)^4 = 130 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2049_204901


namespace NUMINAMATH_CALUDE_point_adding_procedure_l2049_204959

theorem point_adding_procedure (x : ℕ+) : ∃ x, 9 * x - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_point_adding_procedure_l2049_204959


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2049_204900

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (time : ℝ)
  (final_amount : ℝ)
  (h1 : principal = 1000)
  (h2 : time = 3)
  (h3 : final_amount = 1300) :
  (final_amount - principal) / (principal * time) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2049_204900


namespace NUMINAMATH_CALUDE_egg_leftover_proof_l2049_204951

/-- The number of eggs left over when selling a given number of eggs in cartons of 10 -/
def leftover_eggs (total_eggs : ℕ) : ℕ :=
  total_eggs % 10

theorem egg_leftover_proof (john_eggs maria_eggs nikhil_eggs : ℕ) 
  (h1 : john_eggs = 45)
  (h2 : maria_eggs = 38)
  (h3 : nikhil_eggs = 29) :
  leftover_eggs (john_eggs + maria_eggs + nikhil_eggs) = 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_leftover_proof_l2049_204951


namespace NUMINAMATH_CALUDE_point_movement_to_y_axis_l2049_204970

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The y-axis -/
def yAxis : Set Point := {p : Point | p.x = 0}

theorem point_movement_to_y_axis (m : ℝ) :
  let P : Point := ⟨m + 2, 3⟩
  let P' : Point := ⟨P.x + 3, P.y⟩
  P' ∈ yAxis → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_to_y_axis_l2049_204970


namespace NUMINAMATH_CALUDE_traffic_survey_l2049_204929

theorem traffic_survey (N : ℕ) 
  (drivers_A : ℕ) (sample_A : ℕ) (sample_B : ℕ) (sample_C : ℕ) (sample_D : ℕ) : 
  drivers_A = 96 →
  sample_A = 12 →
  sample_B = 21 →
  sample_C = 25 →
  sample_D = 43 →
  N = (sample_A + sample_B + sample_C + sample_D) * drivers_A / sample_A →
  N = 808 := by
sorry

end NUMINAMATH_CALUDE_traffic_survey_l2049_204929


namespace NUMINAMATH_CALUDE_bell_interval_problem_l2049_204962

/-- Represents the intervals of the four bells in seconds -/
structure BellIntervals where
  bell1 : ℕ
  bell2 : ℕ
  bell3 : ℕ
  bell4 : ℕ

/-- Checks if the given intervals result in the bells tolling together after the specified time -/
def tollTogether (intervals : BellIntervals) (time : ℕ) : Prop :=
  time % intervals.bell1 = 0 ∧
  time % intervals.bell2 = 0 ∧
  time % intervals.bell3 = 0 ∧
  time % intervals.bell4 = 0

/-- The main theorem to prove -/
theorem bell_interval_problem (intervals : BellIntervals) :
  intervals.bell1 = 9 →
  intervals.bell3 = 14 →
  intervals.bell4 = 18 →
  tollTogether intervals 630 →
  intervals.bell2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bell_interval_problem_l2049_204962


namespace NUMINAMATH_CALUDE_problem_solution_l2049_204958

theorem problem_solution (x y : ℚ) (hx : x = 3) (hy : y = 5) : 
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5/7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2049_204958


namespace NUMINAMATH_CALUDE_parallel_line_slope_intercept_l2049_204941

/-- The slope-intercept form of a line parallel to 4x + y - 2 = 0 and passing through (3, 2) -/
theorem parallel_line_slope_intercept :
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), 4 * x + y - 2 = 0 → y = -4 * x + b) ∧ 
    (2 = m * 3 + b) ∧
    (∀ (x y : ℝ), y = m * x + b ↔ y = -4 * x + 14) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_intercept_l2049_204941


namespace NUMINAMATH_CALUDE_product_sum_equals_30_l2049_204917

theorem product_sum_equals_30 (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 3)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 17) : 
  a * b + c * d = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equals_30_l2049_204917


namespace NUMINAMATH_CALUDE_subtract_negative_six_a_l2049_204935

theorem subtract_negative_six_a (a : ℝ) : (4 * a^2 - 3 * a + 7) - (-6 * a) = 4 * a^2 - 9 * a + 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_six_a_l2049_204935


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l2049_204966

/-- The sum of interior angles of a pentagon in degrees -/
def pentagon_angle_sum : ℕ := 540

/-- Represents the five consecutive integer angles of a pentagon -/
structure PentagonAngles where
  middle : ℕ
  valid : middle - 2 > 0 -- Ensures all angles are positive

/-- The sum of the five consecutive integer angles -/
def angle_sum (p : PentagonAngles) : ℕ :=
  (p.middle - 2) + (p.middle - 1) + p.middle + (p.middle + 1) + (p.middle + 2)

/-- The largest angle in the pentagon -/
def largest_angle (p : PentagonAngles) : ℕ := p.middle + 2

theorem pentagon_largest_angle :
  ∃ p : PentagonAngles, angle_sum p = pentagon_angle_sum ∧ largest_angle p = 110 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l2049_204966


namespace NUMINAMATH_CALUDE_polynomial_divisibility_by_five_l2049_204968

theorem polynomial_divisibility_by_five (a b c d : ℤ) :
  (∀ x : ℤ, (5 : ℤ) ∣ (a * x^3 + b * x^2 + c * x + d)) →
  (5 : ℤ) ∣ a ∧ (5 : ℤ) ∣ b ∧ (5 : ℤ) ∣ c ∧ (5 : ℤ) ∣ d := by
sorry


end NUMINAMATH_CALUDE_polynomial_divisibility_by_five_l2049_204968


namespace NUMINAMATH_CALUDE_bank_queue_properties_l2049_204916

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum possible total wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum possible total wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected value of wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧ 
  expected_wasted_time q = 84 :=
by sorry

end NUMINAMATH_CALUDE_bank_queue_properties_l2049_204916


namespace NUMINAMATH_CALUDE_prime_4n_2n_1_implies_n_power_of_3_l2049_204949

-- Define a function to check if a number is prime
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

-- Define a function to check if a number is a power of 3
def isPowerOf3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3^k

-- Theorem statement
theorem prime_4n_2n_1_implies_n_power_of_3 (n : ℕ) :
  n > 0 → isPrime (4^n + 2^n + 1) → isPowerOf3 n :=
by sorry

end NUMINAMATH_CALUDE_prime_4n_2n_1_implies_n_power_of_3_l2049_204949


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l2049_204924

/-- Given a square with vertices (-a, 0), (a, 0), (a, 2a), (-a, 2a) intersected by the line y = 2x,
    the ratio of the perimeter of one of the resulting congruent quadrilaterals to a is 5 + √5. -/
theorem square_intersection_perimeter_ratio (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, 0), (a, 0), (a, 2*a), (-a, 2*a)]
  let intersecting_line := (fun x : ℝ => 2*x)
  let quadrilateral_perimeter := 
    (a + 2*a + 2*a + Real.sqrt (a^2 + (2*a)^2))
  quadrilateral_perimeter / a = 5 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l2049_204924


namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_le_l2049_204983

theorem abs_diff_eq_diff_implies_le (x y : ℝ) :
  |x - y| = y - x → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_le_l2049_204983


namespace NUMINAMATH_CALUDE_marble_202_is_white_l2049_204906

/-- Represents the colors of marbles -/
inductive Color
  | Gray
  | White
  | Black
  | Red

/-- Returns the color of the nth marble in the repeating pattern -/
def marbleColor (n : ℕ) : Color :=
  match n % 15 with
  | 0 | 1 | 2 | 3 | 4 | 5 => Color.Gray
  | 6 | 7 | 8 => Color.White
  | 9 | 10 | 11 | 12 => Color.Black
  | _ => Color.Red

theorem marble_202_is_white :
  marbleColor 202 = Color.White := by
  sorry

end NUMINAMATH_CALUDE_marble_202_is_white_l2049_204906


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2049_204977

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![4, 3; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2049_204977


namespace NUMINAMATH_CALUDE_subcommittee_count_l2049_204910

/-- The number of people in the main committee -/
def committee_size : ℕ := 7

/-- The size of each sub-committee -/
def subcommittee_size : ℕ := 2

/-- The number of people that can be chosen for the second position in the sub-committee -/
def remaining_choices : ℕ := committee_size - 1

theorem subcommittee_count :
  (committee_size.choose subcommittee_size) / committee_size = remaining_choices :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l2049_204910


namespace NUMINAMATH_CALUDE_work_completion_proof_l2049_204937

/-- The number of days it takes for person B to complete the work alone -/
def person_b_days : ℝ := 45

/-- The fraction of work completed by both persons in 5 days -/
def work_completed_together : ℝ := 0.2777777777777778

/-- The number of days they work together -/
def days_worked_together : ℝ := 5

/-- The number of days it takes for person A to complete the work alone -/
def person_a_days : ℝ := 30

theorem work_completion_proof :
  (days_worked_together * (1 / person_a_days + 1 / person_b_days) = work_completed_together) →
  person_a_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_proof_l2049_204937


namespace NUMINAMATH_CALUDE_sin_15_105_product_l2049_204979

theorem sin_15_105_product : 4 * Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_105_product_l2049_204979


namespace NUMINAMATH_CALUDE_total_card_cost_l2049_204971

def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def card_cost : ℕ := 2

theorem total_card_cost : christmas_cards * card_cost + birthday_cards * card_cost = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_card_cost_l2049_204971


namespace NUMINAMATH_CALUDE_cannot_cut_squares_l2049_204954

theorem cannot_cut_squares (paper_length paper_width : ℝ) 
  (square1_side square2_side : ℝ) (total_area : ℝ) : 
  paper_length = 10 →
  paper_width = 8 →
  square1_side / square2_side = 4 / 3 →
  square1_side^2 + square2_side^2 = total_area →
  total_area = 75 →
  square1_side + square2_side > paper_length :=
by sorry

end NUMINAMATH_CALUDE_cannot_cut_squares_l2049_204954


namespace NUMINAMATH_CALUDE_charitable_distribution_boy_amount_l2049_204919

def charitable_distribution (initial_pennies : ℕ) 
  (farmer_pennies : ℕ) (beggar_pennies : ℕ) (boy_pennies : ℕ) : Prop :=
  initial_pennies = 42 ∧
  farmer_pennies = initial_pennies / 2 + 1 ∧
  beggar_pennies = (initial_pennies - farmer_pennies) / 2 + 2 ∧
  boy_pennies = initial_pennies - farmer_pennies - beggar_pennies - 1

theorem charitable_distribution_boy_amount :
  ∀ (initial_pennies farmer_pennies beggar_pennies boy_pennies : ℕ),
  charitable_distribution initial_pennies farmer_pennies beggar_pennies boy_pennies →
  boy_pennies = 7 :=
by sorry

end NUMINAMATH_CALUDE_charitable_distribution_boy_amount_l2049_204919


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2049_204992

theorem complex_equation_solution :
  ∃ (x y : ℂ), (3 + 5*I)*x + (2 - I)*y = 17 - 2*I ∧ x = 1 ∧ y = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2049_204992


namespace NUMINAMATH_CALUDE_lcm_of_4_6_9_l2049_204999

theorem lcm_of_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_6_9_l2049_204999


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l2049_204952

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1) + a

-- Define the given line
def given_line (x : ℝ) : ℝ := 2 * x - 6

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, tangent_line a x = given_line x) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l2049_204952


namespace NUMINAMATH_CALUDE_lcm_of_5_6_9_21_l2049_204974

theorem lcm_of_5_6_9_21 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 9 21)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_9_21_l2049_204974


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2049_204964

/-- The surface area of a cube inscribed in a sphere of radius 2 -/
theorem inscribed_cube_surface_area (r : ℝ) (a : ℝ) : 
  r = 2 →  -- The radius of the sphere is 2
  3 * a^2 = (2*r)^2 →  -- The cube's diagonal equals the sphere's diameter
  6 * a^2 = 32 :=  -- The surface area of the cube is 32
by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2049_204964


namespace NUMINAMATH_CALUDE_autobiography_to_fiction_ratio_l2049_204973

theorem autobiography_to_fiction_ratio
  (total_books : ℕ)
  (fiction_books : ℕ)
  (nonfiction_books : ℕ)
  (picture_books : ℕ)
  (h_total : total_books = 35)
  (h_fiction : fiction_books = 5)
  (h_nonfiction : nonfiction_books = fiction_books + 4)
  (h_picture : picture_books = 11)
  : (total_books - fiction_books - nonfiction_books - picture_books) / fiction_books = 2 := by
  sorry

end NUMINAMATH_CALUDE_autobiography_to_fiction_ratio_l2049_204973


namespace NUMINAMATH_CALUDE_soccer_balls_in_bag_l2049_204960

theorem soccer_balls_in_bag (initial_balls : ℕ) (additional_balls : ℕ) : 
  initial_balls = 6 → additional_balls = 18 → initial_balls + additional_balls = 24 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_in_bag_l2049_204960


namespace NUMINAMATH_CALUDE_pascal_all_even_rows_l2049_204938

/-- Returns true if a row in Pascal's triangle consists of all even numbers except for the 1s at each end -/
def isAllEvenExceptEnds (row : ℕ) : Bool := sorry

/-- Counts the number of rows in Pascal's triangle from row 2 to row 30 (inclusive) that consist of all even numbers except for the 1s at each end -/
def countAllEvenRows : ℕ := sorry

theorem pascal_all_even_rows : countAllEvenRows = 4 := by sorry

end NUMINAMATH_CALUDE_pascal_all_even_rows_l2049_204938


namespace NUMINAMATH_CALUDE_perfect_square_unique_l2049_204981

/-- Checks if a quadratic expression ax^2 + bx + c is a perfect square trinomial -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c ∧ a > 0

theorem perfect_square_unique :
  ¬ is_perfect_square_trinomial 1 0 1 ∧     -- x^2 + 1
  ¬ is_perfect_square_trinomial 1 2 (-1) ∧  -- x^2 + 2x - 1
  ¬ is_perfect_square_trinomial 1 1 1 ∧     -- x^2 + x + 1
  is_perfect_square_trinomial 1 4 4         -- x^2 + 4x + 4
  :=
sorry

end NUMINAMATH_CALUDE_perfect_square_unique_l2049_204981


namespace NUMINAMATH_CALUDE_range_of_cosine_squared_minus_two_sine_l2049_204993

theorem range_of_cosine_squared_minus_two_sine :
  ∀ (x : ℝ), -2 ≤ Real.cos x ^ 2 - 2 * Real.sin x ∧ 
             Real.cos x ^ 2 - 2 * Real.sin x ≤ 2 ∧
             ∃ (y z : ℝ), Real.cos y ^ 2 - 2 * Real.sin y = -2 ∧
                          Real.cos z ^ 2 - 2 * Real.sin z = 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_cosine_squared_minus_two_sine_l2049_204993


namespace NUMINAMATH_CALUDE_no_rectangular_prism_with_diagonals_7_8_11_l2049_204907

theorem no_rectangular_prism_with_diagonals_7_8_11 :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    ({7^2, 8^2, 11^2} : Finset ℝ) = {a^2 + b^2, b^2 + c^2, a^2 + c^2} :=
by sorry

end NUMINAMATH_CALUDE_no_rectangular_prism_with_diagonals_7_8_11_l2049_204907


namespace NUMINAMATH_CALUDE_rectangle_area_l2049_204955

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the rectangle
def isRectangle (rect : Rectangle) : Prop :=
  -- Add properties that define a rectangle
  sorry

-- Define the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

-- Define the area of a rectangle
def area (rect : Rectangle) : ℝ :=
  sorry

-- Theorem statement
theorem rectangle_area (rect : Rectangle) :
  isRectangle rect →
  sideLength rect.A rect.B = 15 →
  sideLength rect.A rect.C = 17 →
  area rect = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2049_204955


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2049_204921

/-- Given a triangle with side lengths 7, 24, and 25 units, its area is 84 square units. -/
theorem triangle_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c area =>
    a = 7 ∧ b = 24 ∧ c = 25 → area = 84

/-- The statement of the theorem -/
theorem triangle_area_proof : ∃ (area : ℝ), triangle_area 7 24 25 area :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2049_204921


namespace NUMINAMATH_CALUDE_distance_time_relationship_l2049_204995

/-- The relationship between distance and time for a car traveling at 60 km/h -/
theorem distance_time_relationship (s t : ℝ) (h : s = 60 * t) :
  s = 60 * t :=
by sorry

end NUMINAMATH_CALUDE_distance_time_relationship_l2049_204995


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_l2049_204944

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of YXX + YX + ZY given X, Y, and Z -/
def sum (X Y Z : Digit) : ℕ :=
  111 * Y.val + 12 * X.val + 10 * Z.val

/-- Predicate to check if three digits are distinct -/
def distinct (X Y Z : Digit) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z

theorem largest_three_digit_sum :
  ∃ (X Y Z : Digit), distinct X Y Z ∧ 
    sum X Y Z ≤ 999 ∧
    ∀ (A B C : Digit), distinct A B C → sum A B C ≤ sum X Y Z :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_l2049_204944


namespace NUMINAMATH_CALUDE_polynomial_factor_l2049_204922

theorem polynomial_factor (x y z : ℝ) :
  ∃ (q : ℝ → ℝ → ℝ → ℝ), 
    x^2 - y^2 - z^2 - 2*y*z + x - y - z + 2 = (x - y - z + 1) * q x y z := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l2049_204922


namespace NUMINAMATH_CALUDE_line_circle_separation_l2049_204905

/-- Given a point P(x₀, y₀) inside a circle C: x² + y² = r², 
    the line xx₀ + yy₀ = r² is separated from the circle C. -/
theorem line_circle_separation 
  (x₀ y₀ r : ℝ) 
  (h_inside : x₀^2 + y₀^2 < r^2) : 
  let d := r^2 / Real.sqrt (x₀^2 + y₀^2)
  d > r := by
sorry

end NUMINAMATH_CALUDE_line_circle_separation_l2049_204905


namespace NUMINAMATH_CALUDE_min_cost_butter_l2049_204936

/-- The cost of a 16 oz package of butter -/
def cost_16oz : ℝ := 7

/-- The cost of an 8 oz package of butter -/
def cost_8oz : ℝ := 4

/-- The cost of a 4 oz package of butter before discount -/
def cost_4oz : ℝ := 2

/-- The discount rate applied to 4 oz packages -/
def discount_rate : ℝ := 0.5

/-- The total amount of butter needed in ounces -/
def butter_needed : ℝ := 16

/-- Theorem stating that the minimum cost of purchasing 16 oz of butter is $6.0 -/
theorem min_cost_butter : 
  min cost_16oz (cost_8oz + 2 * (cost_4oz * (1 - discount_rate))) = 6 := by sorry

end NUMINAMATH_CALUDE_min_cost_butter_l2049_204936


namespace NUMINAMATH_CALUDE_anton_card_difference_l2049_204940

/-- Given that Anton has three times as many cards as Heike, Ann has the same number of cards as Heike, 
    and Ann has 60 cards, prove that Anton has 120 more cards than Ann. -/
theorem anton_card_difference (heike_cards : ℕ) (ann_cards : ℕ) (anton_cards : ℕ) 
    (h1 : anton_cards = 3 * heike_cards)
    (h2 : ann_cards = heike_cards)
    (h3 : ann_cards = 60) : 
  anton_cards - ann_cards = 120 := by
  sorry

end NUMINAMATH_CALUDE_anton_card_difference_l2049_204940


namespace NUMINAMATH_CALUDE_fraction_equality_l2049_204918

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (4*a - b) / (a + 4*b) = 3) : 
  (a - 4*b) / (4*a + b) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2049_204918


namespace NUMINAMATH_CALUDE_car_travel_and_budget_l2049_204963

/-- Represents a car with its fuel-to-distance ratio and fuel usage -/
structure Car where
  fuel_ratio : Rat
  distance_ratio : Rat
  fuel_used : ℚ
  fuel_cost : ℚ

/-- Calculates the distance traveled by a car -/
def distance_traveled (c : Car) : ℚ :=
  c.distance_ratio * c.fuel_used / c.fuel_ratio

/-- Calculates the fuel cost for a car -/
def fuel_cost (c : Car) : ℚ :=
  c.fuel_cost * c.fuel_used

theorem car_travel_and_budget (car_a car_b : Car) (budget : ℚ) :
  car_a.fuel_ratio = 4/7 ∧
  car_a.distance_ratio = 7/4 ∧
  car_a.fuel_used = 44 ∧
  car_a.fuel_cost = 7/2 ∧
  car_b.fuel_ratio = 3/5 ∧
  car_b.distance_ratio = 5/3 ∧
  car_b.fuel_used = 27 ∧
  car_b.fuel_cost = 13/4 ∧
  budget = 200 →
  distance_traveled car_a + distance_traveled car_b = 122 ∧
  fuel_cost car_a + fuel_cost car_b = 967/4 ∧
  fuel_cost car_a + fuel_cost car_b - budget = 167/4 :=
by sorry

end NUMINAMATH_CALUDE_car_travel_and_budget_l2049_204963


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l2049_204913

/-- The number of Popsicles Megan consumes in a given time period -/
def popsicles_consumed (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / minutes_per_popsicle

theorem megan_popsicle_consumption :
  popsicles_consumed 18 (5 * 60 + 36) = 18 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l2049_204913


namespace NUMINAMATH_CALUDE_sum_g_h_equals_negative_eight_l2049_204997

theorem sum_g_h_equals_negative_eight (g h : ℝ) :
  (∀ d : ℝ, (8*d^2 - 4*d + g) * (4*d^2 + h*d + 7) = 32*d^4 + (4*h-16)*d^3 - (14*d^2 - 28*d - 56)) →
  g + h = -8 := by sorry

end NUMINAMATH_CALUDE_sum_g_h_equals_negative_eight_l2049_204997


namespace NUMINAMATH_CALUDE_farm_animals_l2049_204923

theorem farm_animals (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 20 + 3 * (cows + chickens)) → 
  (cows = 20 + chickens) :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l2049_204923


namespace NUMINAMATH_CALUDE_new_average_after_increase_l2049_204911

theorem new_average_after_increase (numbers : List ℝ) (h1 : numbers.length = 8) 
  (h2 : numbers.sum / numbers.length = 8) : 
  let new_numbers := numbers.map (λ x => if numbers.indexOf x < 5 then x + 4 else x)
  new_numbers.sum / new_numbers.length = 10.5 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_increase_l2049_204911


namespace NUMINAMATH_CALUDE_nabla_calculation_l2049_204972

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_calculation : nabla (nabla 4 3) 2 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l2049_204972


namespace NUMINAMATH_CALUDE_triangle_properties_l2049_204902

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a / Real.cos t.A = t.c / (2 - Real.cos t.C) ∧
  t.b = 4 ∧
  t.c = 3 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 3

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.a = 2 ∧ 3 * Real.sin t.C + 4 * Real.cos t.C = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2049_204902


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l2049_204908

theorem power_two_plus_one_div_by_three (n : ℕ) : 
  3 ∣ (2^n + 1) ↔ Odd n := by sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l2049_204908


namespace NUMINAMATH_CALUDE_wendy_shoes_left_l2049_204939

theorem wendy_shoes_left (total : ℕ) (given_away : ℕ) (h1 : total = 33) (h2 : given_away = 14) :
  total - given_away = 19 := by
  sorry

end NUMINAMATH_CALUDE_wendy_shoes_left_l2049_204939


namespace NUMINAMATH_CALUDE_butterfly_distribution_theorem_l2049_204926

/-- Represents the movement rules for butterflies on a cube --/
structure ButterflyMovement where
  adjacent : ℕ  -- Number of butterflies moving to each adjacent vertex
  opposite : ℕ  -- Number of butterflies moving to the opposite vertex
  flyaway : ℕ   -- Number of butterflies flying away

/-- Represents the state of butterflies on a cube --/
structure CubeState where
  vertices : Fin 8 → ℕ  -- Number of butterflies at each vertex

/-- Defines the condition for equal distribution of butterflies --/
def is_equally_distributed (state : CubeState) : Prop :=
  ∀ i j : Fin 8, state.vertices i = state.vertices j

/-- Defines the evolution of the cube state according to movement rules --/
def evolve (initial : CubeState) (rules : ButterflyMovement) : ℕ → CubeState
  | 0 => initial
  | n+1 => sorry  -- Implementation of evolution step

/-- Main theorem: N must be a multiple of 45 for equal distribution --/
theorem butterfly_distribution_theorem 
  (N : ℕ) 
  (initial : CubeState) 
  (rules : ButterflyMovement) 
  (h_initial : ∃ v : Fin 8, initial.vertices v = N ∧ ∀ w : Fin 8, w ≠ v → initial.vertices w = 0)
  (h_rules : rules.adjacent = 3 ∧ rules.opposite = 1 ∧ rules.flyaway = 1) :
  (∃ t : ℕ, is_equally_distributed (evolve initial rules t)) ↔ ∃ k : ℕ, N = 45 * k :=
sorry

end NUMINAMATH_CALUDE_butterfly_distribution_theorem_l2049_204926


namespace NUMINAMATH_CALUDE_tangent_line_to_circleC_l2049_204989

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle C: x^2 + y^2 - 6y + 8 = 0 -/
def circleC : Circle := { center := (0, 3), radius := 1 }

/-- A line in the form y = kx -/
structure Line where
  k : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ p : ℝ × ℝ, p.2 = l.k * p.1 ∧ 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    ∀ q : ℝ × ℝ, q.2 = l.k * q.1 → 
      (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 ≥ c.radius^2

theorem tangent_line_to_circleC (l : Line) :
  isTangent l circleC ∧ 
  (∃ p : ℝ × ℝ, isTangent l circleC ∧ isInSecondQuadrant p) →
  l.k = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circleC_l2049_204989


namespace NUMINAMATH_CALUDE_rafael_net_pay_l2049_204988

/-- Calculates the total net pay for Rafael's work week --/
def calculate_net_pay (monday_hours : ℕ) (tuesday_hours : ℕ) (total_week_hours : ℕ) 
  (max_daily_hours : ℕ) (regular_rate : ℚ) (overtime_rate : ℚ) (bonus : ℚ) 
  (tax_rate : ℚ) (tax_credit : ℚ) : ℚ :=
  let remaining_days := 3
  let remaining_hours := total_week_hours - monday_hours - tuesday_hours
  let wednesday_hours := min max_daily_hours remaining_hours
  let thursday_hours := min max_daily_hours (remaining_hours - wednesday_hours)
  let friday_hours := remaining_hours - wednesday_hours - thursday_hours
  
  let monday_pay := regular_rate * min monday_hours max_daily_hours + 
    overtime_rate * max (monday_hours - max_daily_hours) 0
  let tuesday_pay := regular_rate * tuesday_hours
  let wednesday_pay := regular_rate * wednesday_hours
  let thursday_pay := regular_rate * thursday_hours
  let friday_pay := regular_rate * friday_hours
  
  let total_pay := monday_pay + tuesday_pay + wednesday_pay + thursday_pay + friday_pay + bonus
  let taxes := max (total_pay * tax_rate - tax_credit) 0
  
  total_pay - taxes

/-- Theorem stating that Rafael's net pay for the week is $878 --/
theorem rafael_net_pay : 
  calculate_net_pay 10 8 40 8 20 30 100 (1/10) 50 = 878 := by
  sorry

end NUMINAMATH_CALUDE_rafael_net_pay_l2049_204988


namespace NUMINAMATH_CALUDE_original_milk_cost_is_three_l2049_204903

/-- The original cost of a gallon of whole milk -/
def original_milk_cost : ℝ := 3

/-- The current price of a gallon of whole milk -/
def current_milk_price : ℝ := 2

/-- The discount on a box of cereal -/
def cereal_discount : ℝ := 1

/-- The total savings from buying 3 gallons of milk and 5 boxes of cereal -/
def total_savings : ℝ := 8

/-- Theorem stating that the original cost of a gallon of whole milk is $3 -/
theorem original_milk_cost_is_three :
  original_milk_cost = 3 ∧
  current_milk_price = 2 ∧
  cereal_discount = 1 ∧
  total_savings = 8 ∧
  3 * (original_milk_cost - current_milk_price) + 5 * cereal_discount = total_savings :=
by sorry

end NUMINAMATH_CALUDE_original_milk_cost_is_three_l2049_204903


namespace NUMINAMATH_CALUDE_distribution_of_slots_l2049_204920

theorem distribution_of_slots (n : ℕ) (k : ℕ) :
  n = 6 →
  k = 3 →
  (Nat.choose (n - 1) (k - 1) : ℕ) = 10 :=
by sorry

end NUMINAMATH_CALUDE_distribution_of_slots_l2049_204920


namespace NUMINAMATH_CALUDE_bipin_chandan_age_ratio_l2049_204915

/-- Proves that the ratio of Bipin's age to Chandan's age after 10 years is 2:1 -/
theorem bipin_chandan_age_ratio :
  let alok_age : ℕ := 5
  let bipin_age : ℕ := 6 * alok_age
  let chandan_age : ℕ := 7 + 3
  let bipin_future_age : ℕ := bipin_age + 10
  let chandan_future_age : ℕ := chandan_age + 10
  (bipin_future_age : ℚ) / chandan_future_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_bipin_chandan_age_ratio_l2049_204915


namespace NUMINAMATH_CALUDE_average_temperature_of_three_cities_l2049_204965

/-- Proves that the average temperature of three cities is 95 degrees given specific temperature relationships --/
theorem average_temperature_of_three_cities
  (temp_new_york : ℝ)
  (h1 : temp_new_york = 80)
  (temp_miami : ℝ)
  (h2 : temp_miami = temp_new_york + 10)
  (temp_san_diego : ℝ)
  (h3 : temp_san_diego = temp_miami + 25) :
  (temp_new_york + temp_miami + temp_san_diego) / 3 = 95 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_of_three_cities_l2049_204965


namespace NUMINAMATH_CALUDE_hotel_rate_problem_l2049_204945

-- Define the flat rate for the first night and the nightly rate for additional nights
variable (f : ℝ) -- Flat rate for the first night
variable (n : ℝ) -- Nightly rate for additional nights

-- Define Alice's stay
def alice_stay : ℝ := f + 4 * n

-- Define Bob's stay
def bob_stay : ℝ := f + 9 * n

-- State the theorem
theorem hotel_rate_problem (h1 : alice_stay = 245) (h2 : bob_stay = 470) : f = 65 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rate_problem_l2049_204945


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l2049_204987

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- Calculates the amount of water given the amount of corn syrup and the drink ratio -/
def water_amount (corn_syrup_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (corn_syrup_amount * ratio.water) / ratio.corn_syrup

theorem sport_formulation_water_amount :
  water_amount 3 sport_ratio = 45 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l2049_204987
