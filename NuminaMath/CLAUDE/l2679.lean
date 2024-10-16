import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l2679_267919

theorem unique_solution (a b c : ℕ+) 
  (h1 : (Nat.gcd a.val b.val) = 1 ∧ (Nat.gcd b.val c.val) = 1 ∧ (Nat.gcd c.val a.val) = 1)
  (h2 : (a.val^2 + b.val) ∣ (b.val^2 + c.val))
  (h3 : (b.val^2 + c.val) ∣ (c.val^2 + a.val))
  (h4 : ∀ p : ℕ, Nat.Prime p → p ∣ (a.val^2 + b.val) → p % 7 ≠ 1) :
  a = 1 ∧ b = 1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2679_267919


namespace NUMINAMATH_CALUDE_least_zogs_for_dropping_beats_eating_l2679_267904

theorem least_zogs_for_dropping_beats_eating :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, k > 0 → k < n → k * (k + 1) ≤ 15 * k) → 15 * 15 < 15 * (15 + 1) :=
sorry

end NUMINAMATH_CALUDE_least_zogs_for_dropping_beats_eating_l2679_267904


namespace NUMINAMATH_CALUDE_geometry_books_shelf_filling_l2679_267993

/-- Represents the number of books that fill a shelf. -/
structure ShelfFilling where
  algebra : ℕ
  geometry : ℕ

/-- Represents the properties of the book arrangement problem. -/
structure BookArrangement where
  P : ℕ  -- Total number of algebra books
  Q : ℕ  -- Total number of geometry books
  X : ℕ  -- Number of algebra books that fill the shelf
  Y : ℕ  -- Number of geometry books that fill the shelf

/-- The main theorem about the number of geometry books (Z) that fill the shelf. -/
theorem geometry_books_shelf_filling 
  (arr : BookArrangement) 
  (fill1 : ShelfFilling)
  (fill2 : ShelfFilling)
  (h1 : fill1.algebra = arr.X ∧ fill1.geometry = arr.Y)
  (h2 : fill2.algebra = 2 * fill2.geometry)
  (h3 : arr.P + 2 * arr.Q = arr.X + 2 * arr.Y) :
  ∃ Z : ℕ, Z = (arr.P + 2 * arr.Q) / 2 ∧ 
             Z * 2 = arr.P + 2 * arr.Q ∧
             fill2.geometry = Z :=
by sorry

end NUMINAMATH_CALUDE_geometry_books_shelf_filling_l2679_267993


namespace NUMINAMATH_CALUDE_fruit_basket_strawberries_l2679_267937

def fruit_basket (num_strawberries : ℕ) : Prop :=
  let banana_cost : ℕ := 1
  let apple_cost : ℕ := 2
  let avocado_cost : ℕ := 3
  let strawberry_dozen_cost : ℕ := 4
  let half_grape_bunch_cost : ℕ := 2
  let total_cost : ℕ := 28
  let num_bananas : ℕ := 4
  let num_apples : ℕ := 3
  let num_avocados : ℕ := 2
  banana_cost * num_bananas +
  apple_cost * num_apples +
  avocado_cost * num_avocados +
  strawberry_dozen_cost * (num_strawberries / 12) +
  half_grape_bunch_cost * 2 = total_cost

theorem fruit_basket_strawberries : 
  ∃ (n : ℕ), fruit_basket n ∧ n = 24 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_strawberries_l2679_267937


namespace NUMINAMATH_CALUDE_det_special_matrix_l2679_267925

theorem det_special_matrix (x y : ℝ) : 
  Matrix.det !![0, Real.cos x, Real.sin x; 
                -Real.cos x, 0, Real.cos y; 
                -Real.sin x, -Real.cos y, 0] = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2679_267925


namespace NUMINAMATH_CALUDE_moores_law_decade_l2679_267987

/-- Moore's Law transistor growth over a decade -/
theorem moores_law_decade (initial_transistors : ℕ) (years : ℕ) : 
  initial_transistors = 250000 →
  years = 10 →
  initial_transistors * (2 ^ (years / 2)) = 8000000 :=
by sorry

end NUMINAMATH_CALUDE_moores_law_decade_l2679_267987


namespace NUMINAMATH_CALUDE_lunch_cost_calculation_l2679_267902

/-- Calculates the total cost of lunch for all students in an elementary school --/
theorem lunch_cost_calculation (third_grade_classes : ℕ) (third_grade_students_per_class : ℕ)
  (fourth_grade_classes : ℕ) (fourth_grade_students_per_class : ℕ)
  (fifth_grade_classes : ℕ) (fifth_grade_students_per_class : ℕ)
  (hamburger_cost : ℚ) (carrots_cost : ℚ) (cookie_cost : ℚ) :
  third_grade_classes = 5 →
  third_grade_students_per_class = 30 →
  fourth_grade_classes = 4 →
  fourth_grade_students_per_class = 28 →
  fifth_grade_classes = 4 →
  fifth_grade_students_per_class = 27 →
  hamburger_cost = 2.1 →
  carrots_cost = 0.5 →
  cookie_cost = 0.2 →
  (third_grade_classes * third_grade_students_per_class +
   fourth_grade_classes * fourth_grade_students_per_class +
   fifth_grade_classes * fifth_grade_students_per_class) *
  (hamburger_cost + carrots_cost + cookie_cost) = 1036 :=
by sorry

end NUMINAMATH_CALUDE_lunch_cost_calculation_l2679_267902


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2679_267945

theorem largest_divisor_of_difference_of_squares (m n : ℤ) : 
  Odd m → Odd n → n < m → 
  (∀ k : ℤ, k ∣ (m^2 - n^2) → k ≤ 8) ∧ 8 ∣ (m^2 - n^2) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2679_267945


namespace NUMINAMATH_CALUDE_outfit_combinations_l2679_267948

/-- Represents the number of shirts -/
def num_shirts : ℕ := 8

/-- Represents the number of pants -/
def num_pants : ℕ := 6

/-- Represents the number of hats -/
def num_hats : ℕ := 6

/-- Represents the number of distinct colors -/
def num_colors : ℕ := 6

/-- Calculates the number of outfit combinations where no two items are the same color -/
def valid_outfits : ℕ := 174

/-- Theorem stating that the number of valid outfit combinations is 174 -/
theorem outfit_combinations :
  (num_shirts * num_pants * num_hats) -
  (num_colors * num_hats + num_colors * num_shirts + num_colors * num_pants - num_colors) =
  valid_outfits :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2679_267948


namespace NUMINAMATH_CALUDE_crabs_per_basket_is_four_l2679_267912

/-- Represents the crab collecting scenario -/
structure CrabCollection where
  baskets_per_week : ℕ
  collections_per_week : ℕ
  price_per_crab : ℕ
  total_revenue : ℕ

/-- Calculates the number of crabs per basket -/
def crabs_per_basket (c : CrabCollection) : ℕ :=
  c.total_revenue / (c.baskets_per_week * c.collections_per_week * c.price_per_crab)

/-- Theorem stating that the number of crabs per basket is 4 -/
theorem crabs_per_basket_is_four (c : CrabCollection) 
  (h1 : c.baskets_per_week = 3)
  (h2 : c.collections_per_week = 2)
  (h3 : c.price_per_crab = 3)
  (h4 : c.total_revenue = 72) : 
  crabs_per_basket c = 4 := by
  sorry

end NUMINAMATH_CALUDE_crabs_per_basket_is_four_l2679_267912


namespace NUMINAMATH_CALUDE_class_1_wins_l2679_267950

/-- Represents the movements of the marker in a tug-of-war contest -/
def marker_movements : List ℝ := [-0.2, 0.5, -0.8, 1.4, 1.3]

/-- The winning distance in meters -/
def winning_distance : ℝ := 2

/-- Theorem stating that the sum of marker movements is at least the winning distance -/
theorem class_1_wins (movements : List ℝ := marker_movements) 
  (win_dist : ℝ := winning_distance) : 
  movements.sum ≥ win_dist := by sorry

end NUMINAMATH_CALUDE_class_1_wins_l2679_267950


namespace NUMINAMATH_CALUDE_range_of_a_l2679_267983

theorem range_of_a (P Q : Prop) (h_or : P ∨ Q) (h_not_and : ¬(P ∧ Q))
  (h_P : P ↔ ∀ x : ℝ, x^2 - 2*x > a)
  (h_Q : Q ↔ ∃ x : ℝ, x^2 + 2*a*x + 2 = 0) :
  (-2 < a ∧ a < -1) ∨ (a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2679_267983


namespace NUMINAMATH_CALUDE_apple_difference_is_twenty_l2679_267994

/-- The number of apples Cecile bought -/
def cecile_apples : ℕ := 15

/-- The total number of apples bought by Diane and Cecile -/
def total_apples : ℕ := 50

/-- The number of apples Diane bought -/
def diane_apples : ℕ := total_apples - cecile_apples

/-- Diane bought more apples than Cecile -/
axiom diane_bought_more : diane_apples > cecile_apples

/-- The difference between the number of apples Diane and Cecile bought -/
def apple_difference : ℕ := diane_apples - cecile_apples

theorem apple_difference_is_twenty : apple_difference = 20 :=
sorry

end NUMINAMATH_CALUDE_apple_difference_is_twenty_l2679_267994


namespace NUMINAMATH_CALUDE_cirrus_to_cumulus_ratio_l2679_267969

theorem cirrus_to_cumulus_ratio :
  ∀ (cirrus cumulus cumulonimbus : ℕ),
    cirrus = 144 →
    cumulonimbus = 3 →
    cumulus = 12 * cumulonimbus →
    ∃ k : ℕ, cirrus = k * cumulus →
    cirrus / cumulus = 4 :=
by sorry

end NUMINAMATH_CALUDE_cirrus_to_cumulus_ratio_l2679_267969


namespace NUMINAMATH_CALUDE_employee_age_when_hired_l2679_267916

/-- Rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1986

/-- The year the employee became eligible for retirement -/
def retirement_year : ℕ := 2006

/-- The age of the employee when hired -/
def age_when_hired : ℕ := 50

theorem employee_age_when_hired :
  rule_of_70 age_when_hired (retirement_year - hire_year) ∧
  age_when_hired = 70 - (retirement_year - hire_year) := by
  sorry

end NUMINAMATH_CALUDE_employee_age_when_hired_l2679_267916


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2679_267928

theorem perfect_square_condition (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + 2*(a+4)*x + 25 = (x + k)^2) → (a = 1 ∨ a = -9) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2679_267928


namespace NUMINAMATH_CALUDE_minimal_sequence_property_l2679_267914

def F_p (p : ℕ) : Set (ℕ → ℕ) :=
  {a | ∀ n > 0, a (n + 1) = (p + 1) * a n - p * a (n - 1) ∧ ∀ n, a n ≥ 0}

def minimal_sequence (p : ℕ) (n : ℕ) : ℕ :=
  (p^n - 1) / (p - 1)

theorem minimal_sequence_property (p : ℕ) (hp : p > 1) :
  minimal_sequence p ∈ F_p p ∧
  ∀ b ∈ F_p p, ∀ n, minimal_sequence p n ≤ b n :=
by sorry

end NUMINAMATH_CALUDE_minimal_sequence_property_l2679_267914


namespace NUMINAMATH_CALUDE_square_difference_emily_calculation_l2679_267931

theorem square_difference (a b : ℕ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by sorry

theorem emily_calculation : 50^2 - 46^2 = 384 := by sorry

end NUMINAMATH_CALUDE_square_difference_emily_calculation_l2679_267931


namespace NUMINAMATH_CALUDE_unique_c_value_l2679_267926

theorem unique_c_value : ∃! c : ℝ, ∀ x : ℝ, x * (3 * x + 1) - c > 0 ↔ x > -5/3 ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l2679_267926


namespace NUMINAMATH_CALUDE_percentage_subtraction_l2679_267905

theorem percentage_subtraction (original : ℝ) (incorrect_subtraction : ℝ) (difference : ℝ) : 
  original = 200 →
  incorrect_subtraction = 25 →
  difference = 25 →
  let incorrect_result := original - incorrect_subtraction
  let correct_result := incorrect_result - difference
  let percentage := (original - correct_result) / original * 100
  percentage = 25 := by sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l2679_267905


namespace NUMINAMATH_CALUDE_revenue_comparison_l2679_267906

theorem revenue_comparison (base_revenue : ℝ) (projected_increase : ℝ) (actual_decrease : ℝ) :
  projected_increase = 0.2 →
  actual_decrease = 0.25 →
  (base_revenue * (1 - actual_decrease)) / (base_revenue * (1 + projected_increase)) = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_revenue_comparison_l2679_267906


namespace NUMINAMATH_CALUDE_students_walking_home_l2679_267922

theorem students_walking_home (total : ℚ) 
  (bus : ℚ) (auto : ℚ) (bike : ℚ) (metro : ℚ) :
  bus = 1/3 →
  auto = 1/5 →
  bike = 1/8 →
  metro = 1/15 →
  total = 1 →
  total - (bus + auto + bike + metro) = 11/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l2679_267922


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l2679_267958

/-- Represents a chemical compound --/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements --/
def atomic_weight : ℕ → ℕ
  | 0 => 12  -- Carbon
  | 1 => 1   -- Hydrogen
  | 2 => 16  -- Oxygen
  | _ => 0   -- Other elements (not used in this problem)

/-- Calculate the molecular weight of a compound --/
def calculate_molecular_weight (c : Compound) : ℕ :=
  c.carbon * atomic_weight 0 + c.hydrogen * atomic_weight 1 + c.oxygen * atomic_weight 2

/-- Theorem: A compound with 4 Carbon, 8 Hydrogen, and molecular weight 88 has 2 Oxygen atoms --/
theorem compound_oxygen_count :
  ∀ c : Compound,
    c.carbon = 4 →
    c.hydrogen = 8 →
    c.molecular_weight = 88 →
    c.oxygen = 2 :=
by
  sorry

#check compound_oxygen_count

end NUMINAMATH_CALUDE_compound_oxygen_count_l2679_267958


namespace NUMINAMATH_CALUDE_equation_solution_l2679_267959

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ∧ x = -7/6 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2679_267959


namespace NUMINAMATH_CALUDE_matrix_equation_l2679_267979

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, 10; 8, -4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![10/7, -40/7; -4/7, 16/7]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l2679_267979


namespace NUMINAMATH_CALUDE_two_digit_sum_square_property_l2679_267981

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The set of numbers satisfying the condition -/
def valid_numbers : Set ℕ := {10, 20, 11, 30, 21, 12, 31, 22, 13}

/-- Main theorem -/
theorem two_digit_sum_square_property (A : ℕ) :
  is_two_digit A →
  (((sum_of_digits A) ^ 2 = sum_of_digits (A ^ 2)) ↔ A ∈ valid_numbers) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_square_property_l2679_267981


namespace NUMINAMATH_CALUDE_inverse_f_of_3_l2679_267991

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_of_3 :
  ∃ (y : ℝ), y < 0 ∧ f y = 3 ∧ ∀ (z : ℝ), z < 0 ∧ f z = 3 → z = y :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_of_3_l2679_267991


namespace NUMINAMATH_CALUDE_pagoda_lights_l2679_267911

/-- The number of levels in the pagoda -/
def levels : ℕ := 7

/-- The total number of lights in the pagoda -/
def total_lights : ℕ := 381

/-- The number of lights at the tip of the pagoda -/
def lights_at_tip : ℕ := 3

/-- Theorem: If a pagoda has 7 levels, the number of lights doubles at each level from bottom to top,
    and the total number of lights is 381, then the number of lights at the top level (tip) is 3. -/
theorem pagoda_lights : 
  lights_at_tip * (2^levels - 1) = total_lights := by sorry

end NUMINAMATH_CALUDE_pagoda_lights_l2679_267911


namespace NUMINAMATH_CALUDE_anthony_tax_deduction_l2679_267982

/-- Calculates the total tax deduction in cents given an hourly wage and tax rates -/
def totalTaxDeduction (hourlyWage : ℚ) (federalTaxRate : ℚ) (stateTaxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (federalTaxRate + stateTaxRate)

/-- Theorem: Given Anthony's wage and tax rates, the total tax deduction is 62.5 cents -/
theorem anthony_tax_deduction :
  totalTaxDeduction 25 (2/100) (1/200) = 125/2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_tax_deduction_l2679_267982


namespace NUMINAMATH_CALUDE_windows_preference_l2679_267996

theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) : 
  total = 210 → 
  mac = 60 → 
  no_pref = 90 → 
  ∃ (windows : ℕ), windows = total - (mac + no_pref + mac / 3) ∧ windows = 40 := by
  sorry

#check windows_preference

end NUMINAMATH_CALUDE_windows_preference_l2679_267996


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2679_267990

-- System 1
theorem system_one_solution (x z : ℚ) : 
  (3 * x - 5 * z = 6 ∧ x + 4 * z = -15) ↔ (x = -3 ∧ z = -3) := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  ((2 * x - 1) / 5 + (3 * y - 2) / 4 = 2 ∧ 
   (3 * x + 1) / 5 - (3 * y + 2) / 4 = 0) ↔ (x = 3 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2679_267990


namespace NUMINAMATH_CALUDE_min_value_problem_l2679_267971

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  (4*x/(x-1)) + (9*y/(y-1)) ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), (4*x₀/(x₀-1)) + (9*y₀/(y₀-1)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2679_267971


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l2679_267910

theorem prime_pairs_divisibility : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (6 * p * q ∣ p^3 + q^2 + 38) → 
    ((p = 3 ∧ q = 5) ∨ (p = 3 ∧ q = 13)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l2679_267910


namespace NUMINAMATH_CALUDE_expression_not_equal_one_l2679_267946

theorem expression_not_equal_one (a y : ℝ) (ha : a ≠ 0) (hay : a ≠ y) :
  (a / (a - y) + y / (a + y)) / (y / (a - y) - a / (a + y)) ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_not_equal_one_l2679_267946


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l2679_267986

/-- Calculate the total cost of John's purchases including tax -/
theorem johns_purchase_cost :
  let nike_cost : ℚ := 150
  let boots_cost : ℚ := 120
  let jacket_original_price : ℚ := 60
  let jacket_discount_rate : ℚ := 0.3
  let tax_rate : ℚ := 0.1

  let jacket_discounted_price := jacket_original_price * (1 - jacket_discount_rate)
  let total_before_tax := nike_cost + boots_cost + jacket_discounted_price
  let tax_amount := total_before_tax * tax_rate
  let total_with_tax := total_before_tax + tax_amount

  total_with_tax = 343.2 := by sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l2679_267986


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2679_267960

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2679_267960


namespace NUMINAMATH_CALUDE_only_common_term_is_one_l2679_267984

def x : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem only_common_term_is_one : ∀ n : ℕ, x n = y n ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_only_common_term_is_one_l2679_267984


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l2679_267917

theorem simplify_nested_expression (x : ℝ) :
  2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l2679_267917


namespace NUMINAMATH_CALUDE_product_closest_to_640_l2679_267985

def product : ℝ := 0.0000421 * 15864300

def options : List ℝ := [620, 640, 660, 680, 700]

theorem product_closest_to_640 : 
  (options.argmin (fun x => |x - product|)) = some 640 := by sorry

end NUMINAMATH_CALUDE_product_closest_to_640_l2679_267985


namespace NUMINAMATH_CALUDE_cost_increase_operation_l2679_267992

/-- Represents the cost function -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem: If the new cost after an operation on b is 1600% of the original cost,
    then the operation performed on b is multiplication by 2 -/
theorem cost_increase_operation (t : ℝ) (b₀ b₁ : ℝ) (h : t > 0) :
  cost t b₁ = 16 * cost t b₀ → b₁ = 2 * b₀ :=
by sorry

end NUMINAMATH_CALUDE_cost_increase_operation_l2679_267992


namespace NUMINAMATH_CALUDE_investment_average_rate_l2679_267938

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) : 
  total = 5500 ∧ 
  rate1 = 0.03 ∧ 
  rate2 = 0.07 ∧ 
  (∃ x : ℝ, x > 0 ∧ x < total ∧ rate1 * (total - x) = rate2 * x) →
  (rate1 * (total - (rate2 * total) / (rate1 + rate2)) + rate2 * ((rate2 * total) / (rate1 + rate2))) / total = 0.042 := by
  sorry

#check investment_average_rate

end NUMINAMATH_CALUDE_investment_average_rate_l2679_267938


namespace NUMINAMATH_CALUDE_fraction_of_120_l2679_267935

theorem fraction_of_120 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 120 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_120_l2679_267935


namespace NUMINAMATH_CALUDE_greatest_possible_median_l2679_267901

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  t = 20 →
  r ≤ 13 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 20) / 5 = 10 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 20 ∧
    r' = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l2679_267901


namespace NUMINAMATH_CALUDE_tourists_distribution_theorem_l2679_267995

/-- The number of ways to distribute tourists among guides -/
def distribute_tourists (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  num_guides ^ num_tourists

/-- The number of ways to distribute tourists among guides, excluding cases where some guides have no tourists -/
def distribute_tourists_with_restriction (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  distribute_tourists num_tourists num_guides - 
  (num_guides.choose 1 * (num_guides - 1) ^ num_tourists) +
  (num_guides.choose 2 * (num_guides - 2) ^ num_tourists)

/-- The theorem stating that distributing 8 tourists among 3 guides, with each guide having at least one tourist, results in 5796 possible groupings -/
theorem tourists_distribution_theorem : 
  distribute_tourists_with_restriction 8 3 = 5796 := by
  sorry

end NUMINAMATH_CALUDE_tourists_distribution_theorem_l2679_267995


namespace NUMINAMATH_CALUDE_line_slope_l2679_267907

theorem line_slope (x y : ℝ) (h : x + 2 * y - 3 = 0) : 
  ∃ m b : ℝ, m = -1/2 ∧ y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_l2679_267907


namespace NUMINAMATH_CALUDE_lincoln_county_population_l2679_267998

/-- The number of cities in the County of Lincoln -/
def num_cities : ℕ := 25

/-- The lower bound of the average population -/
def lower_bound : ℕ := 5200

/-- The upper bound of the average population -/
def upper_bound : ℕ := 5700

/-- The average population of the cities -/
def avg_population : ℚ := (lower_bound + upper_bound) / 2

/-- The total population of all cities -/
def total_population : ℕ := 136250

theorem lincoln_county_population :
  (num_cities : ℚ) * avg_population = total_population := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_population_l2679_267998


namespace NUMINAMATH_CALUDE_range_of_a_l2679_267934

open Real

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) →
  3 < a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2679_267934


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l2679_267939

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ 3 * x + 4 * y - 4 = 0
  l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ a * x + 8 * y + 2 = 0
  parallel : ∃ (k : ℝ), a = 3 * k ∧ 8 = 4 * k

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ :=
  1

/-- Theorem: The distance between the given parallel lines is 1 -/
theorem parallel_lines_distance (lines : ParallelLines) :
  distance lines = 1 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l2679_267939


namespace NUMINAMATH_CALUDE_f_2015_value_l2679_267923

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_f_1 : f 1 = 2) : 
  f 2015 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_value_l2679_267923


namespace NUMINAMATH_CALUDE_harold_grocery_expense_l2679_267970

/-- Harold's monthly finances --/
def harold_finances (grocery_expense : ℚ) : Prop :=
  let monthly_income : ℚ := 2500
  let rent : ℚ := 700
  let car_payment : ℚ := 300
  let utilities : ℚ := car_payment / 2
  let fixed_expenses : ℚ := rent + car_payment + utilities
  let remaining_after_fixed : ℚ := monthly_income - fixed_expenses
  let retirement_savings : ℚ := (remaining_after_fixed - grocery_expense) / 2
  retirement_savings = 650 ∧ remaining_after_fixed - retirement_savings - grocery_expense = 650

theorem harold_grocery_expense : 
  ∃ (expense : ℚ), harold_finances expense ∧ expense = 50 :=
sorry

end NUMINAMATH_CALUDE_harold_grocery_expense_l2679_267970


namespace NUMINAMATH_CALUDE_election_probability_l2679_267947

/-- Represents an election between two candidates -/
structure Election where
  p : ℕ  -- votes for candidate A
  q : ℕ  -- votes for candidate B
  h : p > q  -- condition that p > q

/-- 
The probability that candidate A's vote count is always greater than 
candidate B's throughout the counting process 
-/
noncomputable def winning_probability (e : Election) : ℚ :=
  (e.p - e.q : ℚ) / (e.p + e.q : ℚ)

/-- 
Theorem: The probability that candidate A's vote count is always greater than 
candidate B's throughout the counting process is (p - q) / (p + q) 
-/
theorem election_probability (e : Election) : 
  winning_probability e = (e.p - e.q : ℚ) / (e.p + e.q : ℚ) := by
  sorry

/-- Example for p = 3 and q = 2 -/
example : ∃ (e : Election), e.p = 3 ∧ e.q = 2 ∧ winning_probability e = 1/5 := by
  sorry

/-- Example for p = 1010 and q = 1009 -/
example : ∃ (e : Election), e.p = 1010 ∧ e.q = 1009 ∧ winning_probability e = 1/2019 := by
  sorry

end NUMINAMATH_CALUDE_election_probability_l2679_267947


namespace NUMINAMATH_CALUDE_valid_midpoint_on_hyperbola_l2679_267955

/-- The hyperbola equation --/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- Definition of midpoint --/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

/-- Theorem stating that (-1,-4) is the only valid midpoint --/
theorem valid_midpoint_on_hyperbola :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    (∀ (x y : ℝ), (x, y) ∈ [(1, 1), (-1, 2), (1, 3)] →
      ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
        is_on_hyperbola x₁' y₁' ∧
        is_on_hyperbola x₂' y₂' ∧
        is_midpoint x y x₁' y₁' x₂' y₂') :=
by sorry

end NUMINAMATH_CALUDE_valid_midpoint_on_hyperbola_l2679_267955


namespace NUMINAMATH_CALUDE_complex_power_difference_l2679_267957

theorem complex_power_difference (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : x - 1/x = 2*Complex.I*Real.sin θ) 
  (h3 : n > 0) : 
  x^n - 1/x^n = 2*Complex.I*Real.sin (n*θ) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2679_267957


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l2679_267949

theorem trig_expression_equals_one : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 1 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 1 / Real.cos (60 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l2679_267949


namespace NUMINAMATH_CALUDE_perpendicular_bisector_m_value_l2679_267975

/-- Given points A and B, if the equation of the perpendicular bisector of segment AB is x + 2y - 2 = 0, then m = 3 -/
theorem perpendicular_bisector_m_value (m : ℝ) : 
  let A : ℝ × ℝ := (1, -2)
  let B : ℝ × ℝ := (m, 2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + 2 * midpoint.2 - 2 = 0) → m = 3 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_m_value_l2679_267975


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2679_267978

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^30 + x^24 + x^18 + x^12 + x^6 + 1 = (x^4 + x^3 + x^2 + x + 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2679_267978


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l2679_267966

-- Define the polynomial Q(x)
def Q (a b c d e : ℂ) (x : ℂ) : ℂ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem coefficient_d_nonzero 
  (a b c d e : ℂ) 
  (h1 : ∃ u v : ℂ, ∀ x : ℂ, Q a b c d e x = x * (x - (2 + 3*I)) * (x - (2 - 3*I)) * (x - u) * (x - v))
  (h2 : Q a b c d e 0 = 0)
  (h3 : Q a b c d e (2 + 3*I) = 0)
  (h4 : ∀ x : ℂ, Q a b c d e x = 0 → x = 0 ∨ x = 2 + 3*I ∨ x = 2 - 3*I ∨ (∃ y : ℂ, y ≠ 0 ∧ y ≠ 2 + 3*I ∧ y ≠ 2 - 3*I ∧ x = y)) :
  d ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_coefficient_d_nonzero_l2679_267966


namespace NUMINAMATH_CALUDE_percentage_equality_l2679_267924

theorem percentage_equality (x : ℝ) (h : 0.3 * 0.4 * x = 24) : 0.4 * 0.3 * x = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2679_267924


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l2679_267989

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ a * x + 1 = 0) →
  (a < -1 ∨ a > 1) :=
by sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l2679_267989


namespace NUMINAMATH_CALUDE_packets_needed_l2679_267952

/-- Calculates the total number of packets needed for seedlings --/
def total_packets (oak_seedlings maple_seedlings pine_seedlings : ℕ) 
                  (oak_per_packet maple_per_packet pine_per_packet : ℕ) : ℕ :=
  (oak_seedlings / oak_per_packet) + 
  (maple_seedlings / maple_per_packet) + 
  (pine_seedlings / pine_per_packet)

/-- Theorem stating that the total number of packets needed is 395 --/
theorem packets_needed : 
  total_packets 420 825 2040 7 5 12 = 395 := by
  sorry

#eval total_packets 420 825 2040 7 5 12

end NUMINAMATH_CALUDE_packets_needed_l2679_267952


namespace NUMINAMATH_CALUDE_three_Z_five_equals_eight_l2679_267900

-- Define the operation Z
def Z (a b : ℝ) : ℝ := b + 10 * a - 3 * a^2

-- Theorem to prove
theorem three_Z_five_equals_eight : Z 3 5 = 8 := by sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_eight_l2679_267900


namespace NUMINAMATH_CALUDE_sum_of_2001_and_1015_l2679_267976

theorem sum_of_2001_and_1015 : 2001 + 1015 = 3016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_2001_and_1015_l2679_267976


namespace NUMINAMATH_CALUDE_exponential_function_point_l2679_267968

theorem exponential_function_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_point_l2679_267968


namespace NUMINAMATH_CALUDE_ab_value_proof_l2679_267929

theorem ab_value_proof (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 - i) * (a - b * i) = (-8 - i) * i) : a * b = 42 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_proof_l2679_267929


namespace NUMINAMATH_CALUDE_apple_tree_difference_l2679_267961

theorem apple_tree_difference (ava_trees lily_trees total_trees : ℕ) : 
  ava_trees = 9 → 
  total_trees = 15 → 
  lily_trees = total_trees - ava_trees → 
  ava_trees - lily_trees = 3 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_difference_l2679_267961


namespace NUMINAMATH_CALUDE_system_solution_l2679_267933

theorem system_solution :
  let eq1 := (fun (x y : ℝ) ↦ x^2 + y^2 + 6*x*y = 68)
  let eq2 := (fun (x y : ℝ) ↦ 2*x^2 + 2*y^2 - 3*x*y = 16)
  (∀ x y, eq1 x y ∧ eq2 x y ↔ 
    ((x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ 
     (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2679_267933


namespace NUMINAMATH_CALUDE_training_hours_calculation_l2679_267964

/-- Calculates the total training hours given daily training hours, initial training days, and additional training days. -/
def total_training_hours (daily_hours : ℕ) (initial_days : ℕ) (additional_days : ℕ) : ℕ :=
  daily_hours * (initial_days + additional_days)

/-- Theorem stating that training 5 hours daily for 30 days and continuing for 12 more days results in 210 total training hours. -/
theorem training_hours_calculation : total_training_hours 5 30 12 = 210 := by
  sorry

end NUMINAMATH_CALUDE_training_hours_calculation_l2679_267964


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2679_267967

/-- The side length of a rhombus given its diagonal lengths -/
theorem rhombus_side_length (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  ∃ (side : ℝ), side = 13 ∧ side^2 = (d1/2)^2 + (d2/2)^2 := by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2679_267967


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l2679_267956

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 7

-- State the theorem
theorem minimum_point_of_translated_absolute_value :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (f x₀ = 7) ∧ (x₀ = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l2679_267956


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l2679_267997

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l2679_267997


namespace NUMINAMATH_CALUDE_elvins_internet_charge_l2679_267977

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalBill : ℝ
  totalBill_eq : totalBill = callCharge + internetCharge

/-- Theorem stating Elvin's fixed monthly internet charge -/
theorem elvins_internet_charge 
  (jan : MonthlyBill) 
  (feb : MonthlyBill) 
  (h1 : jan.totalBill = 40)
  (h2 : feb.totalBill = 76)
  (h3 : feb.callCharge = 2 * jan.callCharge)
  (h4 : jan.internetCharge = feb.internetCharge) : 
  jan.internetCharge = 4 := by
sorry

end NUMINAMATH_CALUDE_elvins_internet_charge_l2679_267977


namespace NUMINAMATH_CALUDE_dot_product_of_parallel_vectors_l2679_267988

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem dot_product_of_parallel_vectors :
  let p : ℝ × ℝ := (1, -2)
  let q : ℝ × ℝ := (x, 4)
  ∀ x : ℝ, parallel p q → p.1 * q.1 + p.2 * q.2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_parallel_vectors_l2679_267988


namespace NUMINAMATH_CALUDE_task_completion_time_l2679_267913

theorem task_completion_time 
  (alice_rate : ℝ) 
  (bob_rate : ℝ) 
  (break_time : ℝ) 
  (t : ℝ) :
  alice_rate = 1/4 →
  bob_rate = 1/6 →
  break_time = 1/2 →
  (alice_rate + bob_rate) * (t - break_time) = 1 →
  (5/12) * (t - 1/2) = 1 := by
sorry

end NUMINAMATH_CALUDE_task_completion_time_l2679_267913


namespace NUMINAMATH_CALUDE_tom_has_two_yellow_tickets_l2679_267944

/-- Represents the number of tickets Tom has -/
structure TicketHoldings where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Conversion rates between ticket types -/
def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10

/-- The number of additional blue tickets Tom needs -/
def additional_blue_needed : ℕ := 163

/-- Tom's current ticket holdings -/
def toms_tickets : TicketHoldings := {
  yellow := 0,  -- We don't know this value yet, so we set it to 0
  red := 3,
  blue := 7
}

/-- Theorem stating that Tom has 2 yellow tickets -/
theorem tom_has_two_yellow_tickets :
  ∃ (y : ℕ), 
    y * (yellow_to_red * red_to_blue) + 
    toms_tickets.red * red_to_blue + 
    toms_tickets.blue + 
    additional_blue_needed = 
    2 * (yellow_to_red * red_to_blue) ∧
    y = 2 := by
  sorry


end NUMINAMATH_CALUDE_tom_has_two_yellow_tickets_l2679_267944


namespace NUMINAMATH_CALUDE_james_profit_l2679_267942

def total_toys : ℕ := 200
def buy_price : ℕ := 20
def sell_price : ℕ := 30
def sell_percentage : ℚ := 80 / 100

theorem james_profit :
  (↑total_toys * sell_percentage * sell_price : ℚ) -
  (↑total_toys * sell_percentage * buy_price : ℚ) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_james_profit_l2679_267942


namespace NUMINAMATH_CALUDE_books_left_l2679_267932

theorem books_left (initial_books given_away : ℝ) 
  (h1 : initial_books = 54.0)
  (h2 : given_away = 23.0) : 
  initial_books - given_away = 31.0 := by
sorry

end NUMINAMATH_CALUDE_books_left_l2679_267932


namespace NUMINAMATH_CALUDE_xyz_equality_l2679_267930

theorem xyz_equality (x y z : ℝ) (h : x * y * z = x + y + z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by sorry

end NUMINAMATH_CALUDE_xyz_equality_l2679_267930


namespace NUMINAMATH_CALUDE_sallys_class_size_l2679_267918

theorem sallys_class_size (total_pens : ℕ) (pens_per_student : ℕ) (pens_home : ℕ) :
  total_pens = 342 →
  pens_per_student = 7 →
  pens_home = 17 →
  ∃ (num_students : ℕ),
    num_students * pens_per_student + 2 * pens_home + (total_pens - num_students * pens_per_student) / 2 = total_pens ∧
    num_students = 44 :=
by sorry

end NUMINAMATH_CALUDE_sallys_class_size_l2679_267918


namespace NUMINAMATH_CALUDE_justin_reading_requirement_l2679_267951

/-- The number of pages Justin needs to read in one week to pass his class -/
def pages_to_pass : ℕ := 130

/-- The number of pages Justin reads on the first day -/
def first_day_pages : ℕ := 10

/-- The number of remaining days in the week -/
def remaining_days : ℕ := 6

/-- The number of pages Justin reads on each of the remaining days -/
def remaining_day_pages : ℕ := 2 * first_day_pages

theorem justin_reading_requirement :
  first_day_pages + remaining_days * remaining_day_pages = pages_to_pass := by
  sorry

end NUMINAMATH_CALUDE_justin_reading_requirement_l2679_267951


namespace NUMINAMATH_CALUDE_inequality_proof_l2679_267943

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2679_267943


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2679_267908

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2679_267908


namespace NUMINAMATH_CALUDE_no_real_roots_l2679_267915

theorem no_real_roots : ¬ ∃ x : ℝ, x^2 - x + 2 = 0 := by sorry

end NUMINAMATH_CALUDE_no_real_roots_l2679_267915


namespace NUMINAMATH_CALUDE_happy_number_512_l2679_267973

def is_happy_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 1)^2 - (2*k - 1)^2

theorem happy_number_512 :
  is_happy_number 512 ∧
  ¬is_happy_number 285 ∧
  ¬is_happy_number 330 ∧
  ¬is_happy_number 582 :=
sorry

end NUMINAMATH_CALUDE_happy_number_512_l2679_267973


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2679_267936

theorem perfect_square_condition (a : ℕ) : a ≥ 1 → (∃ k : ℕ, 1 - 8 * 3^a + 2^(a+2) * (2^a - 1) = k^2) ↔ a = 3 ∨ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2679_267936


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2679_267903

theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧
  (a^2 + 1 = b^2 + 1) ∧
  (a^4 - 9*a^2 + 6 = b^4 - 9*b^2 + 6) ∧
  (a^2 + 1 = 10) ∧
  (a^4 - 9*a^2 + 6 = 6) :=
by sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2679_267903


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2679_267965

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  lastInningsScore : ℕ
  averageIncrease : ℚ
  notOutCount : ℕ

/-- Calculates the average score after a given number of innings -/
def averageAfterInnings (performance : BatsmanPerformance) : ℚ :=
  sorry

theorem batsman_average_after_12th_innings 
  (performance : BatsmanPerformance)
  (h_innings : performance.innings = 12)
  (h_lastScore : performance.lastInningsScore = 60)
  (h_avgIncrease : performance.averageIncrease = 2)
  (h_notOut : performance.notOutCount = 0) :
  averageAfterInnings performance = 38 :=
sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2679_267965


namespace NUMINAMATH_CALUDE_cube_root_cubed_equals_identity_l2679_267980

theorem cube_root_cubed_equals_identity (x : ℝ) : (x^(1/3))^3 = x := by sorry

end NUMINAMATH_CALUDE_cube_root_cubed_equals_identity_l2679_267980


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l2679_267999

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l2679_267999


namespace NUMINAMATH_CALUDE_income_comparison_l2679_267962

theorem income_comparison (J : ℝ) (J_pos : J > 0) : 
  let T := 0.6 * J
  let M := 0.78 * J
  (M - T) / T = 0.3
  := by sorry

end NUMINAMATH_CALUDE_income_comparison_l2679_267962


namespace NUMINAMATH_CALUDE_equation_simplification_l2679_267953

theorem equation_simplification (y : ℝ) (S : ℝ) :
  5 * (2 * y + 3 * Real.sqrt 3) = S →
  10 * (4 * y + 6 * Real.sqrt 3) = 4 * S := by
sorry

end NUMINAMATH_CALUDE_equation_simplification_l2679_267953


namespace NUMINAMATH_CALUDE_inequality_proof_l2679_267940

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a * b^2 * c^3 * d^4 ≤ ((a + 2*b + 3*c + 4*d) / 10)^10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2679_267940


namespace NUMINAMATH_CALUDE_problem_statement_l2679_267954

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 + 2*a*b = -2) 
  (h2 : a*b - b^2 = -4) : 
  2*a^2 + (7/2)*a*b + (1/2)*b^2 = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2679_267954


namespace NUMINAMATH_CALUDE_unique_dot_product_solution_l2679_267921

theorem unique_dot_product_solution (a : ℝ) : 
  (∃! x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ 
    (-Real.sin x * Real.sin (3 * x) + Real.sin (2 * x) * Real.sin (4 * x) = a)) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_dot_product_solution_l2679_267921


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2679_267909

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 7*I
  z₁ / z₂ = (29:ℝ)/53 - (31:ℝ)/53 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2679_267909


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2679_267927

/-- The equations of the asymptotes of the hyperbola x²/16 - y²/9 = 1 -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y => x^2/16 - y^2/9 = 1
  ∃ (f g : ℝ → ℝ), (∀ x, f x = (3/4) * x) ∧ (∀ x, g x = -(3/4) * x) ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, h x y → (|x| > M → |y - f x| < ε ∨ |y - g x| < ε)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2679_267927


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2679_267941

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 + i) / (1 + i)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2679_267941


namespace NUMINAMATH_CALUDE_digital_earth_functions_l2679_267963

/-- Represents the digital Earth system -/
structure DigitalEarth where
  /-- Integrates information from different spaces and times -/
  integrate_info : Bool
  /-- Displays information in 3D and dynamically -/
  display_3d_dynamic : Bool
  /-- Provides experimental conditions for research -/
  provide_exp_conditions : Bool

/-- Theorem stating the correct functions of digital Earth -/
theorem digital_earth_functions (de : DigitalEarth) : 
  de.integrate_info ∧ de.display_3d_dynamic ∧ de.provide_exp_conditions := by
  sorry


end NUMINAMATH_CALUDE_digital_earth_functions_l2679_267963


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2679_267920

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ (1 / 3) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2679_267920


namespace NUMINAMATH_CALUDE_triangle_properties_l2679_267972

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = Real.sqrt 3 * t.a * t.b

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_condition t) 
  (h_angle : 0 < t.A ∧ t.A ≤ 2 * Real.pi / 3) : 
  t.C = Real.pi / 6 ∧ 
  ∀ m : ℝ, m = 2 * (Real.cos (t.A / 2))^2 - Real.sin t.B - 1 → 
  -1 ≤ m ∧ m < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2679_267972


namespace NUMINAMATH_CALUDE_cubic_difference_l2679_267974

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 47) : 
  a^3 - b^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2679_267974
