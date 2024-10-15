import Mathlib

namespace NUMINAMATH_CALUDE_ivanna_dorothy_ratio_l3762_376260

/-- Represents the scores of the three students -/
structure Scores where
  tatuya : ℚ
  ivanna : ℚ
  dorothy : ℚ

/-- The conditions of the quiz scores -/
def quiz_conditions (s : Scores) : Prop :=
  s.dorothy = 90 ∧
  (s.tatuya + s.ivanna + s.dorothy) / 3 = 84 ∧
  s.tatuya = 2 * s.ivanna ∧
  ∃ x : ℚ, 0 < x ∧ x < 1 ∧ s.ivanna = x * s.dorothy

/-- The theorem stating the ratio of Ivanna's score to Dorothy's score -/
theorem ivanna_dorothy_ratio (s : Scores) (h : quiz_conditions s) :
  s.ivanna / s.dorothy = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ivanna_dorothy_ratio_l3762_376260


namespace NUMINAMATH_CALUDE_survey_respondents_l3762_376218

theorem survey_respondents (brand_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  brand_x = 200 →
  ratio_x = 4 →
  ratio_y = 1 →
  ∃ total : ℕ, total = brand_x + (brand_x * ratio_y / ratio_x) ∧ total = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l3762_376218


namespace NUMINAMATH_CALUDE_proportion_third_number_l3762_376271

theorem proportion_third_number (y : ℝ) : 
  (0.75 : ℝ) / 1.35 = y / 9 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l3762_376271


namespace NUMINAMATH_CALUDE_uncle_height_difference_l3762_376246

/-- Given James was initially 2/3 as tall as his uncle who is 72 inches tall,
    and James grew 10 inches, prove that his uncle is now 14 inches taller than James. -/
theorem uncle_height_difference (james_initial_ratio : ℚ) (uncle_height : ℕ) (james_growth : ℕ) :
  james_initial_ratio = 2 / 3 →
  uncle_height = 72 →
  james_growth = 10 →
  uncle_height - (james_initial_ratio * uncle_height + james_growth) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_height_difference_l3762_376246


namespace NUMINAMATH_CALUDE_existence_of_large_n_with_same_digit_occurrences_l3762_376297

open Nat

-- Define a function to check if two numbers have the same digit occurrences
def sameDigitOccurrences (a b : ℕ) : Prop := sorry

-- Define the theorem
theorem existence_of_large_n_with_same_digit_occurrences :
  ∃ n : ℕ, n > 10^100 ∧
    sameDigitOccurrences (n^2) ((n+1)^2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_large_n_with_same_digit_occurrences_l3762_376297


namespace NUMINAMATH_CALUDE_infinitely_many_M_exist_l3762_376236

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number has no zero digits -/
def hasNoZeroDigits (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem infinitely_many_M_exist (N : ℕ) (hN : N > 0) :
  ∀ k : ℕ, ∃ M : ℕ, M > k ∧ hasNoZeroDigits M ∧ sumOfDigits (N * M) = sumOfDigits M :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_M_exist_l3762_376236


namespace NUMINAMATH_CALUDE_average_weight_increase_l3762_376221

/-- Proves that replacing a person weighing 70 kg with a person weighing 90 kg
    in a group of 8 people increases the average weight by 2.5 kg. -/
theorem average_weight_increase
  (n : ℕ)
  (old_weight new_weight : ℝ)
  (h_n : n = 8)
  (h_old : old_weight = 70)
  (h_new : new_weight = 90) :
  (new_weight - old_weight) / n = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3762_376221


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3762_376279

/-- Given an arithmetic sequence a, prove that if a₂ + a₈ = 12, then a₅ = 6 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_sum : a 2 + a 8 = 12) : 
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3762_376279


namespace NUMINAMATH_CALUDE_sum_expression_equals_1215_l3762_376200

theorem sum_expression_equals_1215 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3*a + 8*b + 24*c + 37*d = 2018) : 
  3*b + 8*c + 24*d + 37*a = 1215 := by
  sorry

end NUMINAMATH_CALUDE_sum_expression_equals_1215_l3762_376200


namespace NUMINAMATH_CALUDE_project_completion_time_l3762_376220

/-- The number of days A and B take working together -/
def AB_days : ℝ := 2

/-- The number of days B and C take working together -/
def BC_days : ℝ := 4

/-- The number of days C and A take working together -/
def CA_days : ℝ := 2.4

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 3

theorem project_completion_time :
  (1 / A_days) * CA_days + (1 / BC_days - (1 / AB_days - 1 / A_days)) * CA_days = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l3762_376220


namespace NUMINAMATH_CALUDE_f_extreme_value_and_negative_range_l3762_376230

/-- The function f(x) defined on (0, +∞) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x + (Real.log x - 2) / x + 1

theorem f_extreme_value_and_negative_range :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x) ∧
  f 0 (Real.exp 3) = 1 / Real.exp 3 + 1 ∧
  ∀ (m : ℝ), (∀ (x : ℝ), x > 0 → f m x < 0) ↔ m < -1 / Real.exp 3 :=
by sorry

end NUMINAMATH_CALUDE_f_extreme_value_and_negative_range_l3762_376230


namespace NUMINAMATH_CALUDE_value_of_a_l3762_376244

theorem value_of_a (a b c : ℤ) 
  (sum_ab : a + b = 2)
  (opposite_bc : b + c = 0)
  (abs_c : |c| = 1) :
  a = 3 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_value_of_a_l3762_376244


namespace NUMINAMATH_CALUDE_supermarket_spending_difference_l3762_376228

/-- 
Given:
- initial_amount: The initial amount in Olivia's wallet
- atm_amount: The amount collected from the ATM
- final_amount: The amount left after visiting the supermarket

Prove that the difference between the amount spent at the supermarket
and the amount collected from the ATM is 39 dollars.
-/
theorem supermarket_spending_difference 
  (initial_amount atm_amount final_amount : ℕ) 
  (h1 : initial_amount = 53)
  (h2 : atm_amount = 91)
  (h3 : final_amount = 14) :
  (initial_amount + atm_amount - final_amount) - atm_amount = 39 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_difference_l3762_376228


namespace NUMINAMATH_CALUDE_find_b_l3762_376240

theorem find_b (b c : ℝ) : 
  (∀ x, (3 * x^2 - 4 * x + 5/2) * (2 * x^2 + b * x + c) = 
        6 * x^4 - 11 * x^3 + 13 * x^2 - 15/2 * x + 10/2) → 
  b = -1 := by
sorry

end NUMINAMATH_CALUDE_find_b_l3762_376240


namespace NUMINAMATH_CALUDE_laundry_cost_theorem_l3762_376296

/-- Represents the cost per load of laundry in EUR cents -/
def cost_per_load (loads_per_bottle : ℕ) (regular_price : ℚ) (sale_price : ℚ) 
  (tax_rate : ℚ) (coupon : ℚ) (conversion_rate : ℚ) : ℚ :=
  let total_loads := 2 * loads_per_bottle
  let pre_tax_cost := 2 * sale_price - coupon
  let total_cost := pre_tax_cost * (1 + tax_rate)
  let cost_in_eur := total_cost * conversion_rate
  (cost_in_eur * 100) / total_loads

theorem laundry_cost_theorem (loads_per_bottle : ℕ) (regular_price : ℚ) 
  (sale_price : ℚ) (tax_rate : ℚ) (coupon : ℚ) (conversion_rate : ℚ) :
  loads_per_bottle = 80 →
  regular_price = 25 →
  sale_price = 20 →
  tax_rate = 0.05 →
  coupon = 5 →
  conversion_rate = 0.85 →
  ∃ (n : ℕ), n ≤ 20 ∧ 20 < n + 1 ∧ 
    cost_per_load loads_per_bottle regular_price sale_price tax_rate coupon conversion_rate < n + 1 ∧
    n < cost_per_load loads_per_bottle regular_price sale_price tax_rate coupon conversion_rate := by
  sorry

end NUMINAMATH_CALUDE_laundry_cost_theorem_l3762_376296


namespace NUMINAMATH_CALUDE_range_of_a_l3762_376237

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 2/y = 1) (h_ineq : ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 1 → x + 2*y > a^2 + 8*a) : 
  -4 - 2*Real.sqrt 6 < a ∧ a < -4 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3762_376237


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3762_376208

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 2, 3}

-- Define set B
def B : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3762_376208


namespace NUMINAMATH_CALUDE_complex_power_eight_l3762_376275

theorem complex_power_eight (a b : ℝ) (h : (a : ℂ) + Complex.I = 1 - b * Complex.I) :
  (a + b * Complex.I) ^ 8 = 16 := by sorry

end NUMINAMATH_CALUDE_complex_power_eight_l3762_376275


namespace NUMINAMATH_CALUDE_madison_distance_l3762_376286

/-- Represents the distance between two locations on a map --/
structure MapDistance where
  inches : ℝ

/-- Represents a travel duration --/
structure TravelTime where
  hours : ℝ

/-- Represents a speed --/
structure Speed where
  mph : ℝ

/-- Represents a map scale --/
structure MapScale where
  inches_per_mile : ℝ

/-- Calculates the actual distance traveled given speed and time --/
def calculate_distance (speed : Speed) (time : TravelTime) : ℝ :=
  speed.mph * time.hours

/-- Calculates the map distance given actual distance and map scale --/
def calculate_map_distance (actual_distance : ℝ) (scale : MapScale) : MapDistance :=
  { inches := actual_distance * scale.inches_per_mile }

/-- The main theorem --/
theorem madison_distance (travel_time : TravelTime) (speed : Speed) (scale : MapScale) :
  travel_time.hours = 3.5 →
  speed.mph = 60 →
  scale.inches_per_mile = 0.023809523809523808 →
  (calculate_map_distance (calculate_distance speed travel_time) scale).inches = 5 := by
  sorry

end NUMINAMATH_CALUDE_madison_distance_l3762_376286


namespace NUMINAMATH_CALUDE_combination_permutation_equality_l3762_376214

theorem combination_permutation_equality (n : ℕ) (hn : n > 0) : 
  3 * (Nat.choose (2 * n) 3) = 5 * (Nat.factorial n / Nat.factorial (n - 3)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_equality_l3762_376214


namespace NUMINAMATH_CALUDE_expression_evaluation_l3762_376253

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3762_376253


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3762_376281

theorem algebraic_expression_value (a b : ℝ) 
  (sum_eq : a + b = 5) 
  (product_eq : a * b = 2) : 
  a^2 - a*b + b^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3762_376281


namespace NUMINAMATH_CALUDE_composite_10201_composite_10101_l3762_376223

-- Definition for composite numbers
def IsComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

-- Theorem 1: 10201 is composite in any base > 2
theorem composite_10201 (x : ℕ) (h : x > 2) : IsComposite (x^4 + 2*x^2 + 1) := by
  sorry

-- Theorem 2: 10101 is composite in any base ≥ 2
theorem composite_10101 (x : ℕ) (h : x ≥ 2) : IsComposite (x^4 + x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_composite_10201_composite_10101_l3762_376223


namespace NUMINAMATH_CALUDE_brick_surface_area_l3762_376206

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm² -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
sorry

end NUMINAMATH_CALUDE_brick_surface_area_l3762_376206


namespace NUMINAMATH_CALUDE_white_l_shapes_count_l3762_376259

/-- Represents a 5x5 grid with white and non-white squares -/
def Grid := Matrix (Fin 5) (Fin 5) Bool

/-- Represents an "L" shape composed of three squares -/
def LShape := List (Fin 5 × Fin 5)

/-- Returns true if all squares in the L-shape are white -/
def isWhite (g : Grid) (l : LShape) : Bool :=
  l.all (fun (i, j) => g i j)

/-- Returns the number of distinct all-white L-shapes in the grid -/
def countWhiteLShapes (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are 24 distinct ways to choose an all-white L-shape -/
theorem white_l_shapes_count (g : Grid) : countWhiteLShapes g = 24 := by
  sorry

end NUMINAMATH_CALUDE_white_l_shapes_count_l3762_376259


namespace NUMINAMATH_CALUDE_equation_proof_l3762_376229

theorem equation_proof (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3762_376229


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3762_376282

/-- Given a hyperbola and a circle, if the length of the chord intercepted on the hyperbola's
    asymptotes by the circle is 2, then the eccentricity of the hyperbola is √6/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x + 5 = 0}
  let asymptotes := {(x, y) : ℝ × ℝ | y = b/a * x ∨ y = -b/a * x}
  let chord_length := 2
  chord_length = Real.sqrt (4 - 9 * b^2 / (a^2 + b^2)) * 2 →
  (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3762_376282


namespace NUMINAMATH_CALUDE_equation_solutions_l3762_376202

-- Define the equation
def equation (x : ℝ) : Prop :=
  (59 - 3*x)^(1/4) + (17 + 3*x)^(1/4) = 4

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 20 ∨ x = -10) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3762_376202


namespace NUMINAMATH_CALUDE_valid_quadruples_l3762_376235

def is_valid_quadruple (p q r n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ 
  ¬(3 ∣ (p + q)) ∧
  p + q = r * (p - q)^n

theorem valid_quadruples :
  ∀ p q r n : ℕ,
    is_valid_quadruple p q r n →
    ((p = 2 ∧ q = 3 ∧ r = 5 ∧ Even n) ∨
     (p = 3 ∧ q = 2 ∧ r = 5) ∨
     (p = 5 ∧ q = 3 ∧ r = 1 ∧ n = 3) ∨
     (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 2) ∨
     (p = 5 ∧ q = 3 ∧ r = 8 ∧ n = 1) ∨
     (p = 3 ∧ q = 5 ∧ r = 1 ∧ n = 3) ∨
     (p = 3 ∧ q = 5 ∧ r = 2 ∧ n = 2) ∨
     (p = 3 ∧ q = 5 ∧ r = 8 ∧ n = 1)) :=
by sorry


end NUMINAMATH_CALUDE_valid_quadruples_l3762_376235


namespace NUMINAMATH_CALUDE_contradictory_statements_l3762_376261

theorem contradictory_statements (a b c : ℝ) : 
  (a * b * c ≠ 0) ∧ (a * b * c = 0) ∧ (a * b ≤ 0) → False :=
sorry

end NUMINAMATH_CALUDE_contradictory_statements_l3762_376261


namespace NUMINAMATH_CALUDE_circle_line_no_intersection_l3762_376232

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 2 → y ≠ x + b) ↔ (b > 2 ∨ b < -2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_no_intersection_l3762_376232


namespace NUMINAMATH_CALUDE_negation_equivalence_l3762_376242

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 > Real.exp x) ↔ (∀ x : ℝ, x^2 ≤ Real.exp x) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3762_376242


namespace NUMINAMATH_CALUDE_factor_polynomial_l3762_376231

theorem factor_polynomial (x : ℝ) : 60 * x^4 - 150 * x^8 = -30 * x^4 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3762_376231


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_seven_l3762_376222

def digits : List Nat := [3, 5, 6, 7]

def isTwoDigitNumber (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def formedFromList (n : Nat) : Prop :=
  ∃ (d1 d2 : Nat), d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2

theorem smallest_two_digit_multiple_of_seven :
  ∃ (n : Nat), isTwoDigitNumber n ∧ formedFromList n ∧ n % 7 = 0 ∧
  (∀ (m : Nat), isTwoDigitNumber m → formedFromList m → m % 7 = 0 → n ≤ m) ∧
  n = 35 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_seven_l3762_376222


namespace NUMINAMATH_CALUDE_unique_n_less_than_180_l3762_376283

theorem unique_n_less_than_180 : ∃! n : ℕ, n < 180 ∧ n % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_less_than_180_l3762_376283


namespace NUMINAMATH_CALUDE_banana_pear_ratio_l3762_376274

theorem banana_pear_ratio : 
  ∀ (dishes bananas pears : ℕ),
  dishes = 160 →
  pears = 50 →
  dishes = bananas + 10 →
  ∃ k : ℕ, bananas = k * pears →
  bananas / pears = 3 := by
sorry

end NUMINAMATH_CALUDE_banana_pear_ratio_l3762_376274


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3762_376239

def is_valid_pairing (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), 
    (∀ i ∈ Finset.range 1008, f i ≠ f (2017 - i) ∧ f i ∈ Finset.range 2016 ∧ f (2017 - i) ∈ Finset.range 2016) ∧
    (∀ i ∈ Finset.range 1008, (i + 1) * (2017 - i) ≤ n)

theorem smallest_valid_n : (∀ m < 1017072, ¬ is_valid_pairing m) ∧ is_valid_pairing 1017072 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3762_376239


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3762_376216

theorem absolute_value_equality (x y : ℝ) : 
  |x - Real.sqrt y| = x + Real.sqrt y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3762_376216


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3762_376255

/-- The trajectory of the midpoint of a line segment from a point on a hyperbola to its perpendicular projection on a line -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₁ y₁ : ℝ, 
    -- Q(x₁, y₁) is on the hyperbola x^2 - y^2 = 1
    x₁^2 - y₁^2 = 1 ∧ 
    -- N(2x - x₁, 2y - y₁) is on the line x + y = 2
    (2*x - x₁) + (2*y - y₁) = 2 ∧ 
    -- PQ is perpendicular to the line x + y = 2
    (y - y₁) = (x - x₁) ∧ 
    -- P(x, y) is the midpoint of QN
    x = (x₁ + (2*x - x₁)) / 2 ∧ 
    y = (y₁ + (2*y - y₁)) / 2) →
  -- The trajectory equation of P(x, y)
  2*x^2 - 2*y^2 - 2*x + 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3762_376255


namespace NUMINAMATH_CALUDE_power_function_value_l3762_376225

-- Define a power function that passes through (1/2, √2/2)
def f (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem power_function_value : f (1/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l3762_376225


namespace NUMINAMATH_CALUDE_sector_angle_l3762_376265

/-- Given a sector of a circle with perimeter 8 and area 4, 
    prove that the absolute value of its central angle in radians is 2 -/
theorem sector_angle (r l θ : ℝ) 
  (h_perimeter : 2 * r + l = 8)
  (h_area : (1 / 2) * l * r = 4)
  (h_angle : θ = l / r)
  (h_positive : r > 0) : 
  |θ| = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_angle_l3762_376265


namespace NUMINAMATH_CALUDE_morning_speed_calculation_l3762_376245

theorem morning_speed_calculation 
  (total_time : ℝ) 
  (distance : ℝ) 
  (evening_speed : ℝ) 
  (h1 : total_time = 1) 
  (h2 : distance = 18) 
  (h3 : evening_speed = 30) : 
  ∃ morning_speed : ℝ, 
    distance / morning_speed + distance / evening_speed = total_time ∧ 
    morning_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_morning_speed_calculation_l3762_376245


namespace NUMINAMATH_CALUDE_candy_ratio_l3762_376219

theorem candy_ratio (m_and_m : ℕ) (starburst : ℕ) : 
  (7 : ℕ) * starburst = (4 : ℕ) * m_and_m → m_and_m = 56 → starburst = 32 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l3762_376219


namespace NUMINAMATH_CALUDE_percentage_born_in_july_l3762_376215

theorem percentage_born_in_july (total : ℕ) (born_in_july : ℕ) 
  (h1 : total = 120) (h2 : born_in_july = 18) : 
  (born_in_july : ℚ) / (total : ℚ) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_born_in_july_l3762_376215


namespace NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l3762_376287

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem existence_of_critical_point_and_positive_function :
  (∃ t : ℝ, t ∈ Set.Ioo (1/2) 1 ∧ (deriv (f 1)) t = 0 ∧
    ∀ t' : ℝ, t' ∈ Set.Ioo (1/2) 1 ∧ (deriv (f 1)) t' = 0 → t' = t) ∧
  (∃ m : ℝ, m ∈ Set.Ioo 0 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l3762_376287


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3762_376264

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem tenth_term_of_sequence : 
  let a : ℚ := 5
  let r : ℚ := 3/4
  geometric_sequence a r 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3762_376264


namespace NUMINAMATH_CALUDE_tire_circumference_constant_l3762_376227

/-- The circumference of a tire remains constant given car speed and tire rotation rate -/
theorem tire_circumference_constant
  (v : ℝ) -- Car speed in km/h
  (n : ℝ) -- Tire rotation rate in rpm
  (h1 : v = 120) -- Car speed is 120 km/h
  (h2 : n = 400) -- Tire rotation rate is 400 rpm
  : ∃ (C : ℝ), C = 5 ∧ ∀ (grade : ℝ), C = 5 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_constant_l3762_376227


namespace NUMINAMATH_CALUDE_romeo_chocolate_profit_l3762_376262

theorem romeo_chocolate_profit :
  let num_bars : ℕ := 20
  let cost_per_bar : ℕ := 8
  let total_sales : ℕ := 240
  let packaging_cost_per_bar : ℕ := 3
  let advertising_cost : ℕ := 15
  
  let total_cost : ℕ := num_bars * cost_per_bar + num_bars * packaging_cost_per_bar + advertising_cost
  let profit : ℤ := total_sales - total_cost
  
  profit = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_romeo_chocolate_profit_l3762_376262


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3762_376247

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 5*x - 14 < 0 ↔ -2 < x ∧ x < 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3762_376247


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l3762_376252

/-- Revenue function -/
def R (x : ℕ) : ℚ := 3000 * x - 20 * x^2

/-- Cost function -/
def C (x : ℕ) : ℚ := 500 * x + 4000

/-- Profit function -/
def p (x : ℕ) : ℚ := R x - C x

/-- Marginal function -/
def M (f : ℕ → ℚ) (x : ℕ) : ℚ := f (x + 1) - f x

/-- Marginal profit function -/
def Mp (x : ℕ) : ℚ := M p x

theorem profit_and_marginal_profit_maxima :
  (∃ x : ℕ, x ≤ 100 ∧ ∀ y : ℕ, y ≤ 100 → p y ≤ p x) ∧
  (∃ x : ℕ, x ≤ 100 ∧ ∀ y : ℕ, y ≤ 100 → Mp y ≤ Mp x) ∧
  (∀ x : ℕ, x ≤ 100 → p x ≤ 74120) ∧
  (∀ x : ℕ, x ≤ 100 → Mp x ≤ 2440) ∧
  (∃ x : ℕ, x ≤ 100 ∧ p x = 74120) ∧
  (∃ x : ℕ, x ≤ 100 ∧ Mp x = 2440) :=
by sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l3762_376252


namespace NUMINAMATH_CALUDE_trip_duration_l3762_376207

/-- Proves that the trip duration is 24 hours given the specified conditions -/
theorem trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 35 →
  initial_time = 4 →
  additional_speed = 53 →
  average_speed = 50 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_l3762_376207


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3762_376213

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the slope of the tangent at a point
def tangent_slope (x : ℝ) : ℝ := 8 * x

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop :=
  tangent_slope a * tangent_slope b = -1

-- Define the y-coordinate of the intersection point
def intersection_y (a b : ℝ) : ℝ := 4 * a * b

-- Theorem statement
theorem intersection_point_y_coordinate 
  (a b : ℝ) 
  (ha : parabola a = 4 * a^2) 
  (hb : parabola b = 4 * b^2) 
  (hperp : perpendicular_tangents a b) :
  intersection_y a b = -1/4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3762_376213


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3762_376257

theorem power_of_two_equality (x : ℕ) : (1 / 16 : ℝ) * (2 ^ 50) = 2 ^ x → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3762_376257


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l3762_376288

theorem purely_imaginary_complex (a : ℝ) : 
  (((2 : ℂ) + a * Complex.I) / ((1 : ℂ) - Complex.I) + (1 : ℂ) / ((1 : ℂ) + Complex.I)).re = 0 ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l3762_376288


namespace NUMINAMATH_CALUDE_number_division_problem_l3762_376211

theorem number_division_problem : ∃ x : ℝ, x / 5 = 75 + x / 6 ∧ x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l3762_376211


namespace NUMINAMATH_CALUDE_hexagon_wire_problem_l3762_376209

/-- Calculates the remaining wire length after creating a regular hexagon. -/
def remaining_wire_length (total_wire : ℝ) (hexagon_side : ℝ) : ℝ :=
  total_wire - 6 * hexagon_side

/-- Proves that given a wire of 50 cm and a regular hexagon with side length 8 cm, 
    the remaining wire length is 2 cm. -/
theorem hexagon_wire_problem :
  remaining_wire_length 50 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_wire_problem_l3762_376209


namespace NUMINAMATH_CALUDE_sequence_theorem_l3762_376234

def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ∈ ({0, 1} : Set ℕ)) ∧
  (∀ n, a n + a (n + 1) ≠ a (n + 2) + a (n + 3)) ∧
  (∀ n, a n + a (n + 1) + a (n + 2) ≠ a (n + 3) + a (n + 4) + a (n + 5))

theorem sequence_theorem (a : ℕ → ℕ) (h : sequence_property a) (h1 : a 1 = 0) :
  a 2020 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l3762_376234


namespace NUMINAMATH_CALUDE_fractional_parts_sum_l3762_376233

theorem fractional_parts_sum (x : ℝ) (h : x^3 + 1/x^3 = 18) :
  (x - ⌊x⌋) + (1/x - ⌊1/x⌋) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_parts_sum_l3762_376233


namespace NUMINAMATH_CALUDE_fraction_inequality_l3762_376273

theorem fraction_inequality (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3762_376273


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3762_376212

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, a * x^2 + 2 * a * x + 1 > 0) → 
  (0 < a ∧ a < 1) → 
  ¬ ((0 < a ∧ a < 1) ↔ (∀ x, a * x^2 + 2 * a * x + 1 > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3762_376212


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3762_376248

theorem rectangle_perimeter (L B : ℝ) 
  (h1 : L - B = 23) 
  (h2 : L * B = 3650) : 
  2 * L + 2 * B = 338 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3762_376248


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l3762_376266

/-- The set of positive real numbers -/
def PositiveReals := {x : ℝ | x > 0}

/-- The function f: S³ → S -/
noncomputable def f (x y z : ℝ) : ℝ := (y + Real.sqrt (y^2 + 4*x*z)) / (2*x)

/-- The main theorem -/
theorem function_satisfies_conditions :
  ∀ (x y z k : ℝ), x ∈ PositiveReals → y ∈ PositiveReals → z ∈ PositiveReals → k ∈ PositiveReals →
  (x * f x y z = z * f z y x) ∧
  (f x (k*y) (k^2*z) = k * f x y z) ∧
  (f 1 k (k+1) = k+1) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l3762_376266


namespace NUMINAMATH_CALUDE_integer_solution_problem_l3762_376280

theorem integer_solution_problem :
  ∀ (a b c d : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
    a > b ∧ b > c ∧ c > d →
    a * b + c * d = 34 →
    a * c - b * d = 19 →
    ((a = 1 ∧ b = 4 ∧ c = -5 ∧ d = -6) ∨
     (a = -1 ∧ b = -4 ∧ c = 5 ∧ d = 6)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_problem_l3762_376280


namespace NUMINAMATH_CALUDE_polynomial_equality_l3762_376243

theorem polynomial_equality : 
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 10406040101 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3762_376243


namespace NUMINAMATH_CALUDE_fruit_salad_problem_l3762_376241

/-- Fruit salad problem -/
theorem fruit_salad_problem (green_grapes red_grapes raspberries blueberries pineapple : ℕ) :
  red_grapes = 3 * green_grapes + 7 →
  raspberries = green_grapes - 5 →
  blueberries = 4 * raspberries →
  pineapple = blueberries / 2 + 5 →
  green_grapes + red_grapes + raspberries + blueberries + pineapple = 350 →
  red_grapes = 100 := by
  sorry

#check fruit_salad_problem

end NUMINAMATH_CALUDE_fruit_salad_problem_l3762_376241


namespace NUMINAMATH_CALUDE_triangle_altitude_proof_l3762_376284

def triangle_altitude (a b c : ℝ) : Prop :=
  let tan_BCA := 1
  let tan_BAC := 1 / 7
  let perimeter := 24 + 18 * Real.sqrt 2
  let h := 3
  -- The altitude from B to AC has length h
  (tan_BCA = 1 ∧ tan_BAC = 1 / 7 ∧ 
   a + b + c = perimeter) → 
  h = 3

theorem triangle_altitude_proof : 
  ∃ (a b c : ℝ), triangle_altitude a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_altitude_proof_l3762_376284


namespace NUMINAMATH_CALUDE_f_properties_l3762_376204

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (3^x + 1)

theorem f_properties :
  ∀ (a : ℝ),
  -- 1. Range of f when a = 1
  (∀ y : ℝ, y ∈ Set.range (f 1) ↔ 1 < y ∧ y < 3) ∧
  -- 2. f is strictly decreasing
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ∧
  -- 3. If f is odd and f(f(x)) + f(m) < 0 has solutions, then m > -1
  (((∀ x : ℝ, f a (-x) = -f a x) ∧
    (∃ x : ℝ, f a (f a x) + f a m < 0)) → m > -1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3762_376204


namespace NUMINAMATH_CALUDE_solution_set_of_system_l3762_376269

theorem solution_set_of_system : 
  let S : Set (ℝ × ℝ) := {(x, y) | x - y = 0 ∧ x^2 + y = 2}
  S = {(1, 1), (-2, -2)} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_system_l3762_376269


namespace NUMINAMATH_CALUDE_roses_picked_l3762_376251

theorem roses_picked (tulips flowers_used extra_flowers : ℕ) : 
  tulips = 4 →
  flowers_used = 11 →
  extra_flowers = 4 →
  flowers_used + extra_flowers - tulips = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_picked_l3762_376251


namespace NUMINAMATH_CALUDE_diamonds_in_tenth_figure_l3762_376267

/-- The number of diamonds in the outer circle of the nth figure -/
def outer_diamonds (n : ℕ) : ℕ := 4 + 6 * (n - 1)

/-- The total number of diamonds in the nth figure -/
def total_diamonds (n : ℕ) : ℕ := 3 * n^2 + n

theorem diamonds_in_tenth_figure : total_diamonds 10 = 310 := by sorry

end NUMINAMATH_CALUDE_diamonds_in_tenth_figure_l3762_376267


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3762_376295

theorem coefficient_x_squared_in_expansion :
  let expansion := (fun x => (x - 2/x)^8)
  let coefficient_x_squared (f : ℝ → ℝ) := 
    (1/2) * (deriv (deriv f) 0)
  coefficient_x_squared expansion = -Nat.choose 8 3 * 2^3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3762_376295


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l3762_376224

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equals_set : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l3762_376224


namespace NUMINAMATH_CALUDE_chairs_remaining_l3762_376254

theorem chairs_remaining (initial_chairs : ℕ) (difference : ℕ) (remaining_chairs : ℕ) : 
  initial_chairs = 15 → 
  difference = 12 → 
  initial_chairs - remaining_chairs = difference →
  remaining_chairs = 3 := by
  sorry

end NUMINAMATH_CALUDE_chairs_remaining_l3762_376254


namespace NUMINAMATH_CALUDE_inequality_proof_l3762_376290

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 / b) + Real.sqrt (b^2 / a) ≥ Real.sqrt a + Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3762_376290


namespace NUMINAMATH_CALUDE_shopkeeper_bananas_l3762_376276

theorem shopkeeper_bananas (oranges : ℕ) (bananas : ℕ) : 
  oranges = 600 →
  (510 : ℝ) + 0.95 * bananas = 0.89 * (oranges + bananas) →
  bananas = 400 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_bananas_l3762_376276


namespace NUMINAMATH_CALUDE_maximize_x5y2_l3762_376217

theorem maximize_x5y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 35) :
  x^5 * y^2 ≤ 25^5 * 10^2 ∧ 
  (x^5 * y^2 = 25^5 * 10^2 ↔ x = 25 ∧ y = 10) :=
by sorry

end NUMINAMATH_CALUDE_maximize_x5y2_l3762_376217


namespace NUMINAMATH_CALUDE_log_sum_40_25_l3762_376238

theorem log_sum_40_25 : Real.log 40 + Real.log 25 = 3 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_40_25_l3762_376238


namespace NUMINAMATH_CALUDE_part_one_part_two_l3762_376293

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | x^2 - 3*x ≤ 10}

-- Part 1
theorem part_one : (Set.compl (P 3) ∩ Q) = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, (P a ∪ Q = Q) ↔ a ∈ Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3762_376293


namespace NUMINAMATH_CALUDE_initial_men_correct_l3762_376272

/-- The initial number of men working -/
def initial_men : ℕ := 450

/-- The number of hours worked per day initially -/
def initial_hours : ℕ := 8

/-- The depth dug initially in meters -/
def initial_depth : ℕ := 40

/-- The new depth to be dug in meters -/
def new_depth : ℕ := 50

/-- The new number of hours worked per day -/
def new_hours : ℕ := 6

/-- The number of extra men needed for the new task -/
def extra_men : ℕ := 30

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct :
  initial_men * initial_hours * initial_depth = (initial_men + extra_men) * new_hours * new_depth :=
by sorry

end NUMINAMATH_CALUDE_initial_men_correct_l3762_376272


namespace NUMINAMATH_CALUDE_sum_a1_a5_l3762_376250

/-- For a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℕ := n^2 + 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_a1_a5 : a 1 + a 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_a1_a5_l3762_376250


namespace NUMINAMATH_CALUDE_range_of_t_range_of_a_l3762_376205

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3|

-- Part 1
theorem range_of_t (t : ℝ) : f t + f (2 * t) < 9 ↔ -1 < t ∧ t < 5 := by sorry

-- Part 2
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ f (2 * x) + |x + a| ≤ 3) ↔ a ∈ Set.Icc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_range_of_t_range_of_a_l3762_376205


namespace NUMINAMATH_CALUDE_max_matches_C_proof_l3762_376258

/-- Represents a player in the tournament -/
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

/-- The number of matches won by a player -/
def matches_won : Player → Nat
| Player.A => 2
| Player.B => 1
| _ => 0  -- We don't know for C and D, so we set it to 0

/-- The total number of matches in a round-robin tournament with 4 players -/
def total_matches : Nat := 6

/-- The maximum number of matches C can win -/
def max_matches_C : Nat := 3

/-- Theorem stating the maximum number of matches C can win -/
theorem max_matches_C_proof :
  ∀ (c_wins : Nat),
  c_wins ≤ max_matches_C ∧
  c_wins + matches_won Player.A + matches_won Player.B ≤ total_matches :=
sorry

end NUMINAMATH_CALUDE_max_matches_C_proof_l3762_376258


namespace NUMINAMATH_CALUDE_fraction_equivalence_and_decimal_l3762_376291

theorem fraction_equivalence_and_decimal : 
  let original : ℚ := 2 / 4
  let equiv1 : ℚ := 6 / 12
  let equiv2 : ℚ := 20 / 40
  let decimal : ℝ := 0.5
  (original = equiv1) ∧ (original = equiv2) ∧ (original = decimal) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_and_decimal_l3762_376291


namespace NUMINAMATH_CALUDE_marble_selection_ways_l3762_376299

def blue_marbles : ℕ := 3
def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def total_marbles : ℕ := blue_marbles + red_marbles + green_marbles
def marbles_to_choose : ℕ := 5

theorem marble_selection_ways : 
  (Nat.choose blue_marbles 1) * (Nat.choose red_marbles 1) * (Nat.choose green_marbles 1) *
  (Nat.choose (total_marbles - 3) 2) = 756 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l3762_376299


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3762_376292

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℕ), x > 0 ∧
    (a * x).gcd (b * x) = 1 ∧
    (a * x).gcd (c * x) = 1 ∧
    (b * x).gcd (c * x) = 1 ∧
    a * x + b * x + c * x = total ∧
    min (a * x) (min (b * x) (c * x)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3762_376292


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l3762_376285

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- Theorem: A space shuttle orbiting at 4 km/s is traveling at 14400 km/h -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 4 = 14400 := by
  sorry

#eval km_per_second_to_km_per_hour 4

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l3762_376285


namespace NUMINAMATH_CALUDE_f_2011_equals_sin_l3762_376210

noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => Real.cos
  | n + 1 => deriv (f n)

theorem f_2011_equals_sin : f 2011 = Real.sin := by sorry

end NUMINAMATH_CALUDE_f_2011_equals_sin_l3762_376210


namespace NUMINAMATH_CALUDE_wheat_cost_is_30_l3762_376268

/-- Represents the farm's cultivation scenario -/
structure FarmScenario where
  totalLand : ℕ
  cornCost : ℕ
  totalBudget : ℕ
  wheatAcres : ℕ

/-- Calculates the cost of wheat cultivation per acre -/
def wheatCostPerAcre (scenario : FarmScenario) : ℕ :=
  (scenario.totalBudget - (scenario.cornCost * (scenario.totalLand - scenario.wheatAcres))) / scenario.wheatAcres

/-- Theorem stating the cost of wheat cultivation per acre is 30 -/
theorem wheat_cost_is_30 (scenario : FarmScenario) 
    (h1 : scenario.totalLand = 500)
    (h2 : scenario.cornCost = 42)
    (h3 : scenario.totalBudget = 18600)
    (h4 : scenario.wheatAcres = 200) :
  wheatCostPerAcre scenario = 30 := by
  sorry

#eval wheatCostPerAcre { totalLand := 500, cornCost := 42, totalBudget := 18600, wheatAcres := 200 }

end NUMINAMATH_CALUDE_wheat_cost_is_30_l3762_376268


namespace NUMINAMATH_CALUDE_roots_transformation_l3762_376289

theorem roots_transformation (p q r : ℂ) : 
  (p^3 - 4*p^2 + 5*p + 2 = 0) ∧ 
  (q^3 - 4*q^2 + 5*q + 2 = 0) ∧ 
  (r^3 - 4*r^2 + 5*r + 2 = 0) →
  ((3*p)^3 - 12*(3*p)^2 + 45*(3*p) + 54 = 0) ∧
  ((3*q)^3 - 12*(3*q)^2 + 45*(3*q) + 54 = 0) ∧
  ((3*r)^3 - 12*(3*r)^2 + 45*(3*r) + 54 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l3762_376289


namespace NUMINAMATH_CALUDE_base10_89_equals_base5_324_l3762_376256

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 5 * acc + d) 0

theorem base10_89_equals_base5_324 : fromBase5 [4, 2, 3] = 89 := by
  sorry

end NUMINAMATH_CALUDE_base10_89_equals_base5_324_l3762_376256


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3762_376249

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 245 / Real.sqrt 175) = (15 + 2 * Real.sqrt 7) / 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3762_376249


namespace NUMINAMATH_CALUDE_max_value_theorem_l3762_376263

theorem max_value_theorem (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) :
  ∃ (max : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 5 → a + 2*b + 3*c ≤ max) ∧
  (∃ (x₀ y₀ z₀ : ℝ), x₀^2 + y₀^2 + z₀^2 = 5 ∧ x₀ + 2*y₀ + 3*z₀ = max) ∧
  max = Real.sqrt 70 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3762_376263


namespace NUMINAMATH_CALUDE_custom_op_5_3_l3762_376277

-- Define the custom operation
def custom_op (m n : ℕ) : ℕ := n ^ 2 - m

-- Theorem statement
theorem custom_op_5_3 : custom_op 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_5_3_l3762_376277


namespace NUMINAMATH_CALUDE_exists_k_composite_for_all_n_l3762_376278

theorem exists_k_composite_for_all_n : ∃ k : ℕ, ∀ n : ℕ, ∃ m : ℕ, m > 1 ∧ m ∣ (k * 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_k_composite_for_all_n_l3762_376278


namespace NUMINAMATH_CALUDE_two_triangles_from_tetrahedron_l3762_376298

-- Define a tetrahedron
structure Tetrahedron :=
  (A B C D : Point)
  (AB AC AD BC BD CD : ℝ)
  (longest_edge : AB ≥ max AC (max AD (max BC (max BD CD))))
  (AC_geq_BD : AC ≥ BD)

-- Define a triangle
structure Triangle :=
  (side1 side2 side3 : ℝ)

-- Theorem statement
theorem two_triangles_from_tetrahedron (t : Tetrahedron) : 
  ∃ (triangle1 triangle2 : Triangle), 
    (triangle1.side1 = t.BC ∧ triangle1.side2 = t.CD ∧ triangle1.side3 = t.BD) ∧
    (triangle2.side1 = t.AC ∧ triangle2.side2 = t.CD ∧ triangle2.side3 = t.AD) ∧
    (triangle1 ≠ triangle2) :=
sorry

end NUMINAMATH_CALUDE_two_triangles_from_tetrahedron_l3762_376298


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3762_376294

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3762_376294


namespace NUMINAMATH_CALUDE_johns_initial_money_l3762_376226

theorem johns_initial_money (initial_amount : ℚ) : 
  (initial_amount * (1 - (3/8 + 3/10)) = 65) → initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_johns_initial_money_l3762_376226


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3762_376203

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- Check if two lines are coincident -/
def are_coincident (l1 l2 : Line) : Prop :=
  are_parallel l1 l2 ∧ l1.a * l2.c = l2.a * l1.c

/-- The problem statement -/
theorem parallel_lines_a_value :
  ∃ (a : ℝ), 
    let l1 : Line := ⟨a, 3, 1⟩
    let l2 : Line := ⟨2, a+1, 1⟩
    are_parallel l1 l2 ∧ ¬are_coincident l1 l2 ∧ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3762_376203


namespace NUMINAMATH_CALUDE_quadratic_solution_l3762_376201

theorem quadratic_solution (a : ℝ) : (1 : ℝ)^2 + 1 + 2*a = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3762_376201


namespace NUMINAMATH_CALUDE_inequality_theorem_l3762_376270

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ - z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ - z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔
   x₁ * y₁ * x₂ * y₂ - z₁^2 * x₂^2 - z₂^2 * x₁^2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_inequality_theorem_l3762_376270
