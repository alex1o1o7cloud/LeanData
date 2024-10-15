import Mathlib

namespace NUMINAMATH_CALUDE_girls_joined_l709_70912

theorem girls_joined (initial_girls final_girls : ℕ) : 
  initial_girls = 732 → final_girls = 1414 → final_girls - initial_girls = 682 :=
by
  sorry

#check girls_joined

end NUMINAMATH_CALUDE_girls_joined_l709_70912


namespace NUMINAMATH_CALUDE_pizza_toppings_l709_70984

/-- Given a pizza with the following properties:
  * It has 16 slices in total
  * Every slice has at least one topping
  * There are three toppings: cheese, chicken, and olives
  * 8 slices have cheese
  * 12 slices have chicken
  * 6 slices have olives
  This theorem proves that exactly 5 slices have all three toppings. -/
theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (chicken_slices : ℕ) (olive_slices : ℕ)
    (h_total : total_slices = 16)
    (h_cheese : cheese_slices = 8)
    (h_chicken : chicken_slices = 12)
    (h_olives : olive_slices = 6)
    (h_at_least_one : ∀ slice, slice ∈ Finset.range total_slices →
      (slice ∈ Finset.range cheese_slices ∨
       slice ∈ Finset.range chicken_slices ∨
       slice ∈ Finset.range olive_slices)) :
    ∃ all_toppings : ℕ, all_toppings = 5 ∧
      (∀ slice, slice ∈ Finset.range total_slices →
        (slice ∈ Finset.range cheese_slices ∧
         slice ∈ Finset.range chicken_slices ∧
         slice ∈ Finset.range olive_slices) ↔
        slice ∈ Finset.range all_toppings) := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l709_70984


namespace NUMINAMATH_CALUDE_michael_earnings_l709_70914

/-- Michael's earnings from selling paintings --/
theorem michael_earnings (large_price small_price : ℕ) (large_quantity small_quantity : ℕ) :
  large_price = 100 →
  small_price = 80 →
  large_quantity = 5 →
  small_quantity = 8 →
  large_price * large_quantity + small_price * small_quantity = 1140 :=
by sorry

end NUMINAMATH_CALUDE_michael_earnings_l709_70914


namespace NUMINAMATH_CALUDE_calculation_proof_l709_70983

theorem calculation_proof : 8500 + 45 * 2 - 500 / 25 + 100 = 8670 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l709_70983


namespace NUMINAMATH_CALUDE_food_court_combinations_l709_70990

/-- Represents the number of options for each meal component -/
structure MealOptions where
  entrees : Nat
  drinks : Nat
  desserts : Nat

/-- Calculates the total number of meal combinations -/
def mealCombinations (options : MealOptions) : Nat :=
  options.entrees * options.drinks * options.desserts

/-- The given meal options in the food court -/
def foodCourtOptions : MealOptions :=
  { entrees := 4, drinks := 4, desserts := 2 }

/-- Theorem: The number of distinct meal combinations in the food court is 32 -/
theorem food_court_combinations :
  mealCombinations foodCourtOptions = 32 := by
  sorry

end NUMINAMATH_CALUDE_food_court_combinations_l709_70990


namespace NUMINAMATH_CALUDE_square_area_l709_70911

/-- Given a square with one vertex at (-6, -4) and diagonals intersecting at (3, 2),
    prove that its area is 234 square units. -/
theorem square_area (v : ℝ × ℝ) (c : ℝ × ℝ) (h1 : v = (-6, -4)) (h2 : c = (3, 2)) : 
  let d := 2 * Real.sqrt ((c.1 - v.1)^2 + (c.2 - v.2)^2)
  let s := d / Real.sqrt 2
  s^2 = 234 := by sorry

end NUMINAMATH_CALUDE_square_area_l709_70911


namespace NUMINAMATH_CALUDE_jills_salary_l709_70999

theorem jills_salary (discretionary_income : ℝ) (net_salary : ℝ) : 
  discretionary_income = net_salary / 5 →
  0.30 * discretionary_income + 
  0.20 * discretionary_income + 
  0.35 * discretionary_income + 
  102 = discretionary_income →
  net_salary = 3400 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l709_70999


namespace NUMINAMATH_CALUDE_odd_function_extension_l709_70925

-- Define an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x < 0
def f_neg (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = x^2 + x

-- Theorem statement
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_neg : f_neg f) :
  ∀ x, x > 0 → f x = -x^2 + x :=
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l709_70925


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l709_70994

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, (2 * x^2 + 5 = 7 * x - 2) ↔ (x = p + q * I ∨ x = p - q * I)) →
  p + q^2 = 35/16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l709_70994


namespace NUMINAMATH_CALUDE_four_b_b_two_divisible_by_seven_l709_70982

theorem four_b_b_two_divisible_by_seven (B : ℕ) : 
  B ≤ 9 → (4000 + 110 * B + 2) % 7 = 0 ↔ B = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_b_b_two_divisible_by_seven_l709_70982


namespace NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l709_70956

/-- A cubic polynomial function -/
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- Monotonicity of a function on ℝ -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ (∀ x y, x < y → f x > f y)

/-- Intersection with x-axis at exactly one point -/
def IntersectsOnce (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

/-- Theorem stating that monotonicity is sufficient but not necessary for intersecting x-axis once -/
theorem monotonic_sufficient_not_necessary (b c d : ℝ) :
  (Monotonic (f b c d) → IntersectsOnce (f b c d)) ∧
  ¬(IntersectsOnce (f b c d) → Monotonic (f b c d)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l709_70956


namespace NUMINAMATH_CALUDE_chloe_carrots_total_l709_70968

/-- The total number of carrots Chloe has after picking, throwing out, and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

/-- Theorem stating that given the specific numbers in the problem, 
    the total number of carrots is 101. -/
theorem chloe_carrots_total : 
  total_carrots 128 94 67 = 101 := by
  sorry

end NUMINAMATH_CALUDE_chloe_carrots_total_l709_70968


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l709_70933

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 + i) / (1 + i)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l709_70933


namespace NUMINAMATH_CALUDE_min_value_expression_l709_70916

theorem min_value_expression (k n : ℝ) (h1 : k ≥ 0) (h2 : n ≥ 0) (h3 : 2 * k + n = 2) :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 2 * x + y = 2 → 2 * k^2 - 4 * n ≤ 2 * x^2 - 4 * y ∧
  ∃ k₀ n₀ : ℝ, k₀ ≥ 0 ∧ n₀ ≥ 0 ∧ 2 * k₀ + n₀ = 2 ∧ 2 * k₀^2 - 4 * n₀ = -8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l709_70916


namespace NUMINAMATH_CALUDE_maria_cookies_left_l709_70989

/-- Calculates the number of cookies Maria has left after distributing them -/
def cookiesLeft (initialCookies : ℕ) : ℕ :=
  let afterFriend := initialCookies - (initialCookies * 20 / 100)
  let afterFamily := afterFriend - (afterFriend / 3)
  let afterEating := afterFamily - 4
  let toNeighbor := afterEating / 6
  afterEating - toNeighbor

/-- Theorem stating that Maria will have 24 cookies left -/
theorem maria_cookies_left : cookiesLeft 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_left_l709_70989


namespace NUMINAMATH_CALUDE_bromine_mass_percentage_not_37_21_l709_70981

/-- The mass percentage of bromine in HBrO3 is not 37.21% -/
theorem bromine_mass_percentage_not_37_21 (H_mass Br_mass O_mass : ℝ) 
  (h1 : H_mass = 1.01)
  (h2 : Br_mass = 79.90)
  (h3 : O_mass = 16.00) :
  let HBrO3_mass := H_mass + Br_mass + 3 * O_mass
  (Br_mass / HBrO3_mass) * 100 ≠ 37.21 := by sorry

end NUMINAMATH_CALUDE_bromine_mass_percentage_not_37_21_l709_70981


namespace NUMINAMATH_CALUDE_three_equal_products_exist_l709_70927

/-- Represents a 3x3 table filled with numbers from 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if all numbers in the table are unique --/
def all_unique (t : Table) : Prop :=
  ∀ i j i' j', t i j = t i' j' → (i = i' ∧ j = j')

/-- Calculates the product of a row --/
def row_product (t : Table) (i : Fin 3) : ℕ :=
  ((t i 0).val + 1) * ((t i 1).val + 1) * ((t i 2).val + 1)

/-- Calculates the product of a column --/
def col_product (t : Table) (j : Fin 3) : ℕ :=
  ((t 0 j).val + 1) * ((t 1 j).val + 1) * ((t 2 j).val + 1)

/-- Checks if at least three products are equal --/
def three_equal_products (t : Table) : Prop :=
  ∃ p : ℕ, (
    (row_product t 0 = p ∧ row_product t 1 = p ∧ row_product t 2 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 0 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 0 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 0 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 1 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 1 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 2 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 2 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 2 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (col_product t 0 = p ∧ col_product t 1 = p ∧ col_product t 2 = p)
  )

theorem three_equal_products_exist :
  ∃ t : Table, all_unique t ∧ three_equal_products t :=
by sorry

end NUMINAMATH_CALUDE_three_equal_products_exist_l709_70927


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l709_70919

/-- Represents the number of people a can of soup can feed -/
def people_per_can : ℕ := 4

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 10

/-- Represents the number of children fed -/
def children_fed : ℕ := 20

/-- Theorem: Given the conditions, prove that 20 adults can be fed with the remaining soup -/
theorem remaining_soup_feeds_twenty_adults :
  let cans_for_children := children_fed / people_per_can
  let remaining_cans := total_cans - cans_for_children
  remaining_cans * people_per_can = 20 := by
  sorry

#check remaining_soup_feeds_twenty_adults

end NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l709_70919


namespace NUMINAMATH_CALUDE_smartphone_price_difference_l709_70995

/-- Calculate the final price after applying a discount --/
def final_price (initial_price : ℚ) (discount_percent : ℚ) : ℚ :=
  initial_price * (1 - discount_percent / 100)

/-- The problem statement --/
theorem smartphone_price_difference :
  let store_a_initial_price : ℚ := 125
  let store_a_discount : ℚ := 8
  let store_b_initial_price : ℚ := 130
  let store_b_discount : ℚ := 10
  
  final_price store_b_initial_price store_b_discount -
  final_price store_a_initial_price store_a_discount = 2 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_difference_l709_70995


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l709_70976

theorem polynomial_division_remainder (x : ℝ) : 
  x^1000 % ((x^2 + 1) * (x + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l709_70976


namespace NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l709_70930

-- Define a function to check if a number is a three-digit integer
def isThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function to check if digits are distinct
def hasDistinctDigits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.length = digits.toFinset.card

-- Define a function to check if digits form a geometric sequence
def formsGeometricSequence (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  ∃ (r : ℚ), r ≠ 0 ∧ b = a * r ∧ c = b * r

-- State the theorem
theorem smallest_three_digit_geometric_sequence :
  ∀ n : ℕ, isThreeDigitInteger n ∧ hasDistinctDigits n ∧ formsGeometricSequence n →
  124 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l709_70930


namespace NUMINAMATH_CALUDE_arun_lower_limit_l709_70913

-- Define Arun's weight as a real number
variable (W : ℝ)

-- Define the conditions
def arun_upper_limit : Prop := W < 72
def brother_opinion : Prop := 60 < W ∧ W < 70
def mother_opinion : Prop := W ≤ 67
def average_weight : Prop := (W + 67) / 2 = 66

-- Define the theorem
theorem arun_lower_limit :
  arun_upper_limit W →
  brother_opinion W →
  mother_opinion W →
  average_weight W →
  W = 65 := by sorry

end NUMINAMATH_CALUDE_arun_lower_limit_l709_70913


namespace NUMINAMATH_CALUDE_trajectory_of_M_l709_70985

/-- Given points A(-1,0) and B(1,0), and a point M(x,y), if the ratio of the slope of AM
    to the slope of BM is 3, then x = -2 -/
theorem trajectory_of_M (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) (hy : y ≠ 0) :
  (y / (x + 1)) / (y / (x - 1)) = 3 → x = -2 := by
  sorry

#check trajectory_of_M

end NUMINAMATH_CALUDE_trajectory_of_M_l709_70985


namespace NUMINAMATH_CALUDE_combined_standard_deviation_l709_70997

/-- Given two groups of numbers with known means and variances, 
    calculate the standard deviation of the combined set. -/
theorem combined_standard_deviation 
  (n₁ n₂ : ℕ) 
  (mean₁ mean₂ : ℝ) 
  (var₁ var₂ : ℝ) :
  n₁ = 10 →
  n₂ = 10 →
  mean₁ = 50 →
  mean₂ = 40 →
  var₁ = 33 →
  var₂ = 45 →
  let n_total := n₁ + n₂
  let var_total := (n₁ * var₁ + n₂ * var₂) / n_total + 
                   (n₁ * n₂ : ℝ) / (n_total ^ 2 : ℝ) * (mean₁ - mean₂) ^ 2
  Real.sqrt var_total = 8 := by
  sorry

#check combined_standard_deviation

end NUMINAMATH_CALUDE_combined_standard_deviation_l709_70997


namespace NUMINAMATH_CALUDE_complex_power_difference_l709_70935

theorem complex_power_difference (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : x - 1/x = 2*Complex.I*Real.sin θ) 
  (h3 : n > 0) : 
  x^n - 1/x^n = 2*Complex.I*Real.sin (n*θ) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l709_70935


namespace NUMINAMATH_CALUDE_closest_point_l709_70923

def u (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1 + 5*s
  | 1 => -2 + 3*s
  | 2 => -4 - 2*s

def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3
  | 1 => 3
  | 2 => 4

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 3
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (∀ t : ℝ, ‖u s - b‖^2 ≤ ‖u t - b‖^2) ↔ s = 9/38 := by sorry

end NUMINAMATH_CALUDE_closest_point_l709_70923


namespace NUMINAMATH_CALUDE_absolute_difference_inequality_l709_70918

theorem absolute_difference_inequality (x : ℝ) :
  |2*x - 4| - |3*x + 9| < 1 ↔ x < -3 ∨ x > -6/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_inequality_l709_70918


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l709_70955

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = Set.Ioo (-1) 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l709_70955


namespace NUMINAMATH_CALUDE_siding_total_cost_l709_70993

def wall_width : ℝ := 10
def wall_height : ℝ := 7
def roof_width : ℝ := 10
def roof_height : ℝ := 6
def roof_sections : ℕ := 2
def siding_width : ℝ := 10
def siding_height : ℝ := 15
def siding_cost : ℝ := 35

theorem siding_total_cost :
  let total_area := wall_width * wall_height + roof_width * roof_height * roof_sections
  let siding_area := siding_width * siding_height
  let sections_needed := Int.ceil (total_area / siding_area)
  sections_needed * siding_cost = 70 := by sorry

end NUMINAMATH_CALUDE_siding_total_cost_l709_70993


namespace NUMINAMATH_CALUDE_season_length_l709_70971

def games_per_month : ℕ := 7
def games_in_season : ℕ := 14

theorem season_length :
  games_in_season / games_per_month = 2 :=
sorry

end NUMINAMATH_CALUDE_season_length_l709_70971


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l709_70941

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ (1 / 3) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l709_70941


namespace NUMINAMATH_CALUDE_tiling_impossible_l709_70931

/-- Represents a 1 × 3 strip used for tiling -/
structure Strip :=
  (length : Nat)
  (width : Nat)
  (h_length : length = 3)
  (h_width : width = 1)

/-- Represents the figure to be tiled -/
structure Figure :=
  (total_squares : Nat)
  (color1_squares : Nat)
  (color2_squares : Nat)
  (h_total : total_squares = color1_squares + color2_squares)
  (h_color1 : color1_squares = 7)
  (h_color2 : color2_squares = 8)

/-- Represents a tiling of the figure with strips -/
structure Tiling :=
  (figure : Figure)
  (strips : List Strip)
  (h_cover : ∀ s ∈ strips, s.length = 3 ∧ s.width = 1)
  (h_no_overlap : List.Nodup strips)
  (h_complete : strips.length * 3 = figure.total_squares)

/-- The main theorem stating that tiling is impossible -/
theorem tiling_impossible (f : Figure) : ¬ ∃ t : Tiling, t.figure = f := by
  sorry

end NUMINAMATH_CALUDE_tiling_impossible_l709_70931


namespace NUMINAMATH_CALUDE_i_minus_one_in_second_quadrant_l709_70952

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem i_minus_one_in_second_quadrant :
  is_in_second_quadrant (Complex.I - 1) := by
  sorry

end NUMINAMATH_CALUDE_i_minus_one_in_second_quadrant_l709_70952


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l709_70944

/-- Two lines in the xy-plane --/
structure TwoLines :=
  (a : ℝ)

/-- The condition for two lines to be parallel --/
def are_parallel (lines : TwoLines) : Prop :=
  lines.a^2 - lines.a = 2

/-- The statement that a=2 is sufficient but not necessary for the lines to be parallel --/
theorem sufficient_not_necessary :
  (∃ (lines : TwoLines), lines.a = 2 → are_parallel lines) ∧
  (∃ (lines : TwoLines), are_parallel lines ∧ lines.a ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l709_70944


namespace NUMINAMATH_CALUDE_time_to_distance_l709_70986

/-- Theorem: Time to reach a certain distance for two people walking in opposite directions -/
theorem time_to_distance (mary_speed sharon_speed : ℝ) (initial_time initial_distance : ℝ) :
  mary_speed = 4 →
  sharon_speed = 6 →
  initial_time = 0.3 →
  initial_distance = 3 →
  ∀ d : ℝ, d > 0 → ∃ t : ℝ, t = d / (mary_speed + sharon_speed) ∧ d = (mary_speed + sharon_speed) * t :=
by sorry

end NUMINAMATH_CALUDE_time_to_distance_l709_70986


namespace NUMINAMATH_CALUDE_quadrilateral_is_trapezium_l709_70965

/-- A quadrilateral with angles x°, 5x°, 2x°, and 4x° is a trapezium -/
theorem quadrilateral_is_trapezium (x : ℝ) 
  (angle_sum : x + 5*x + 2*x + 4*x = 360) : 
  ∃ (a b c d : ℝ), 
    a + b + c + d = 360 ∧ 
    a + c = 180 ∧
    (a = x ∨ a = 5*x ∨ a = 2*x ∨ a = 4*x) ∧
    (b = x ∨ b = 5*x ∨ b = 2*x ∨ b = 4*x) ∧
    (c = x ∨ c = 5*x ∨ c = 2*x ∨ c = 4*x) ∧
    (d = x ∨ d = 5*x ∨ d = 2*x ∨ d = 4*x) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_is_trapezium_l709_70965


namespace NUMINAMATH_CALUDE_sum_double_factorial_divisible_l709_70905

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

theorem sum_double_factorial_divisible :
  (double_factorial 1985 + double_factorial 1986) % 1987 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_double_factorial_divisible_l709_70905


namespace NUMINAMATH_CALUDE_quadratic_range_l709_70951

theorem quadratic_range (a c : ℝ) (h1 : -4 ≤ a + c) (h2 : a + c ≤ -1)
  (h3 : -1 ≤ 4*a + c) (h4 : 4*a + c ≤ 5) : -1 ≤ 9*a + c ∧ 9*a + c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l709_70951


namespace NUMINAMATH_CALUDE_class_1_wins_l709_70974

/-- Represents the movements of the marker in a tug-of-war contest -/
def marker_movements : List ℝ := [-0.2, 0.5, -0.8, 1.4, 1.3]

/-- The winning distance in meters -/
def winning_distance : ℝ := 2

/-- Theorem stating that the sum of marker movements is at least the winning distance -/
theorem class_1_wins (movements : List ℝ := marker_movements) 
  (win_dist : ℝ := winning_distance) : 
  movements.sum ≥ win_dist := by sorry

end NUMINAMATH_CALUDE_class_1_wins_l709_70974


namespace NUMINAMATH_CALUDE_divisor_count_power_of_two_l709_70948

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ+) : ℕ := sorry

/-- Number of divisors function -/
def num_of_divisors (n : ℕ+) : ℕ := sorry

/-- A natural number is a power of two -/
def is_power_of_two (n : ℕ) : Prop := sorry

theorem divisor_count_power_of_two (n : ℕ+) :
  is_power_of_two (sum_of_divisors n) → is_power_of_two (num_of_divisors n) := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_power_of_two_l709_70948


namespace NUMINAMATH_CALUDE_student_marks_l709_70967

theorem student_marks (M P C : ℕ) : 
  C = P + 20 → 
  (M + C) / 2 = 40 → 
  M + P = 60 := by
sorry

end NUMINAMATH_CALUDE_student_marks_l709_70967


namespace NUMINAMATH_CALUDE_building_height_is_270_l709_70902

/-- Calculates the height of a building with specified story heights -/
def buildingHeight (totalStories : ℕ) (firstHalfHeight : ℕ) (heightIncrease : ℕ) : ℕ :=
  let firstHalfStories := totalStories / 2
  let secondHalfStories := totalStories - firstHalfStories
  let firstHalfTotalHeight := firstHalfStories * firstHalfHeight
  let secondHalfHeight := firstHalfHeight + heightIncrease
  let secondHalfTotalHeight := secondHalfStories * secondHalfHeight
  firstHalfTotalHeight + secondHalfTotalHeight

/-- Theorem: The height of a 20-story building with specified story heights is 270 feet -/
theorem building_height_is_270 :
  buildingHeight 20 12 3 = 270 :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_building_height_is_270_l709_70902


namespace NUMINAMATH_CALUDE_equation_solution_l709_70921

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ∧ x = -7/6 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l709_70921


namespace NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l709_70977

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  2^n - 2

/-- Theorem stating that f(n) correctly represents the sum of numbers in the nth row -/
theorem sum_of_nth_row (n : ℕ) : 
  f n = 2^n - 2 :=
sorry

/-- Corollary for the 100th row -/
theorem sum_of_100th_row : 
  f 100 = 2^100 - 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l709_70977


namespace NUMINAMATH_CALUDE_interest_equality_l709_70932

theorem interest_equality (P : ℝ) : 
  let I₁ := P * 0.04 * 5
  let I₂ := P * 0.05 * 4
  I₁ = I₂ ∧ I₁ = 20 := by sorry

end NUMINAMATH_CALUDE_interest_equality_l709_70932


namespace NUMINAMATH_CALUDE_discounted_three_books_cost_l709_70987

/-- The cost of two identical books without discount -/
def two_books_cost : ℝ := 36

/-- The discount rate applied to each book -/
def discount_rate : ℝ := 0.1

/-- The number of books to purchase after discount -/
def num_books_after_discount : ℕ := 3

/-- Theorem stating the total cost of three books after applying a 10% discount -/
theorem discounted_three_books_cost :
  let original_price := two_books_cost / 2
  let discounted_price := original_price * (1 - discount_rate)
  discounted_price * num_books_after_discount = 48.60 := by
  sorry

end NUMINAMATH_CALUDE_discounted_three_books_cost_l709_70987


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l709_70934

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 7

-- State the theorem
theorem minimum_point_of_translated_absolute_value :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (f x₀ = 7) ∧ (x₀ = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l709_70934


namespace NUMINAMATH_CALUDE_simple_interest_proof_l709_70962

/-- Given a principal amount for which the compound interest at 5% per annum for 2 years is 56.375,
    prove that the simple interest at 5% per annum for 2 years is 55. -/
theorem simple_interest_proof (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 56.375 → P * 0.05 * 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_proof_l709_70962


namespace NUMINAMATH_CALUDE_at_least_two_solved_five_l709_70943

/-- The number of problems in the competition -/
def num_problems : ℕ := 6

/-- The structure representing a participant in the competition -/
structure Participant where
  solved : Finset (Fin num_problems)

/-- The type of the competition -/
structure Competition where
  participants : Finset Participant
  pair_solved : ∀ (i j : Fin num_problems), i ≠ j →
    (participants.filter (λ p => i ∈ p.solved ∧ j ∈ p.solved)).card >
    (2 * participants.card) / 5
  no_all_solved : ∀ p : Participant, p ∈ participants → p.solved.card < num_problems

/-- The main theorem -/
theorem at_least_two_solved_five (comp : Competition) :
  (comp.participants.filter (λ p => p.solved.card = num_problems - 1)).card ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_solved_five_l709_70943


namespace NUMINAMATH_CALUDE_weekly_distance_increase_l709_70969

/-- Calculates the weekly distance increase for marathon training --/
theorem weekly_distance_increase 
  (initial_distance : ℚ) 
  (target_distance : ℚ) 
  (training_weeks : ℕ) 
  (h1 : initial_distance = 2) 
  (h2 : target_distance = 20) 
  (h3 : training_weeks = 27) :
  (target_distance - initial_distance) / training_weeks = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_weekly_distance_increase_l709_70969


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l709_70915

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  2 * x₁^2 - 3 * x₁ + 1 = 0 ∧ 2 * x₂^2 - 3 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l709_70915


namespace NUMINAMATH_CALUDE_abs_value_sum_l709_70903

theorem abs_value_sum (a b c : ℚ) : 
  (abs a = 2) → 
  (abs b = 2) → 
  (abs c = 3) → 
  (b < 0) → 
  (0 < a) → 
  ((a + b + c = 3) ∨ (a + b + c = -3)) := by
sorry

end NUMINAMATH_CALUDE_abs_value_sum_l709_70903


namespace NUMINAMATH_CALUDE_complement_of_intersection_l709_70960

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {1, 3, 4}

theorem complement_of_intersection (h : U = {1, 2, 3, 4} ∧ M = {1, 2, 3} ∧ N = {1, 3, 4}) :
  (M ∩ N)ᶜ = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l709_70960


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l709_70980

/-- Given an arithmetic sequence with a non-zero common difference,
    if the 2nd, 3rd, and 6th terms form a geometric sequence,
    then the common ratio of these three terms is 3. -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  (a 2) * (a 6) = (a 3)^2 →  -- 2nd, 3rd, and 6th terms form a geometric sequence
  (a 3) / (a 2) = 3 :=  -- common ratio is 3
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l709_70980


namespace NUMINAMATH_CALUDE_hexagonal_quadratic_coefficient_l709_70937

-- Define hexagonal numbers
def hexagonal (n : ℕ) : ℕ := 2 * n^2 - n

-- Define the general quadratic form for hexagonal numbers
def quadratic_form (a b c n : ℕ) : ℕ := a * n^2 + b * n + c

-- Theorem statement
theorem hexagonal_quadratic_coefficient :
  ∃ (b c : ℕ), ∀ (n : ℕ), n > 0 → hexagonal n = quadratic_form 3 b c n :=
sorry

end NUMINAMATH_CALUDE_hexagonal_quadratic_coefficient_l709_70937


namespace NUMINAMATH_CALUDE_range_of_m_for_fractional_equation_l709_70907

/-- The range of m for which the equation m/(x-2) + 1 = x/(2-x) has a non-negative solution x, where x ≠ 2 -/
theorem range_of_m_for_fractional_equation (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ m / (x - 2) + 1 = x / (2 - x)) ↔ 
  (m ≤ 2 ∧ m ≠ -2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_fractional_equation_l709_70907


namespace NUMINAMATH_CALUDE_correct_proposition_l709_70946

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 ≥ 0

-- Define proposition q
def q : Prop := 1 < 0

-- Theorem to prove
theorem correct_proposition : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l709_70946


namespace NUMINAMATH_CALUDE_total_planting_area_is_2600_l709_70992

/-- Represents the number of trees to be planted for each tree chopped -/
structure PlantingRatio :=
  (oak : ℕ)
  (pine : ℕ)

/-- Represents the number of trees chopped in each half of the year -/
structure TreesChopped :=
  (oak : ℕ)
  (pine : ℕ)

/-- Represents the space required for planting each type of tree -/
structure PlantingSpace :=
  (oak : ℕ)
  (pine : ℕ)

/-- Calculates the total area needed for tree planting during the entire year -/
def totalPlantingArea (ratio : PlantingRatio) (firstHalf : TreesChopped) (secondHalf : TreesChopped) (space : PlantingSpace) : ℕ :=
  let oakArea := (firstHalf.oak * ratio.oak * space.oak)
  let pineArea := ((firstHalf.pine + secondHalf.pine) * ratio.pine * space.pine)
  oakArea + pineArea

/-- Theorem stating that the total area needed for tree planting is 2600 m² -/
theorem total_planting_area_is_2600 (ratio : PlantingRatio) (firstHalf : TreesChopped) (secondHalf : TreesChopped) (space : PlantingSpace) :
  ratio.oak = 4 →
  ratio.pine = 2 →
  firstHalf.oak = 100 →
  firstHalf.pine = 100 →
  secondHalf.oak = 150 →
  secondHalf.pine = 150 →
  space.oak = 4 →
  space.pine = 2 →
  totalPlantingArea ratio firstHalf secondHalf space = 2600 :=
by
  sorry

end NUMINAMATH_CALUDE_total_planting_area_is_2600_l709_70992


namespace NUMINAMATH_CALUDE_nail_polish_count_l709_70910

theorem nail_polish_count (kim heidi karen : ℕ) : 
  kim = 12 →
  heidi = kim + 5 →
  karen = kim - 4 →
  heidi + karen = 25 := by sorry

end NUMINAMATH_CALUDE_nail_polish_count_l709_70910


namespace NUMINAMATH_CALUDE_hamburgers_served_l709_70936

/-- Given a restaurant that made a certain number of hamburgers and had some left over,
    calculate the number of hamburgers served. -/
theorem hamburgers_served (total : ℕ) (leftover : ℕ) (h1 : total = 9) (h2 : leftover = 6) :
  total - leftover = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_l709_70936


namespace NUMINAMATH_CALUDE_sallys_class_size_l709_70929

theorem sallys_class_size (total_pens : ℕ) (pens_per_student : ℕ) (pens_home : ℕ) :
  total_pens = 342 →
  pens_per_student = 7 →
  pens_home = 17 →
  ∃ (num_students : ℕ),
    num_students * pens_per_student + 2 * pens_home + (total_pens - num_students * pens_per_student) / 2 = total_pens ∧
    num_students = 44 :=
by sorry

end NUMINAMATH_CALUDE_sallys_class_size_l709_70929


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l709_70973

theorem trig_expression_equals_one : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 1 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 1 / Real.cos (60 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l709_70973


namespace NUMINAMATH_CALUDE_two_tangent_lines_l709_70949

-- Define the function f(x) = x³ - x²
def f (x : ℝ) : ℝ := x^3 - x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Define a point of tangency
structure TangentPoint where
  x : ℝ
  y : ℝ
  slope : ℝ

-- Define a tangent line that passes through (1,0)
def isTangentLineThroughPoint (tp : TangentPoint) : Prop :=
  tp.y = f tp.x ∧ 
  tp.slope = f' tp.x ∧
  0 = tp.y + tp.slope * (1 - tp.x)

-- Theorem: There are exactly 2 tangent lines to f(x) that pass through (1,0)
theorem two_tangent_lines : 
  ∃! (s : Finset TangentPoint), 
    (∀ tp ∈ s, isTangentLineThroughPoint tp) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l709_70949


namespace NUMINAMATH_CALUDE_inequality_proof_l709_70963

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a - b)^2 / (2 * (a + b)) ≤ Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ∧
  Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ≤ (a - b)^2 / (Real.sqrt 2 * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l709_70963


namespace NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l709_70972

def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}

def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem intersection_empty_iff_b_in_range (b : ℝ) :
  set_A ∩ set_B b = ∅ ↔ b ≥ 1/3 ∨ b ≤ -2 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l709_70972


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l709_70961

theorem smallest_integer_in_consecutive_set : 
  ∀ (n : ℤ), 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) → 
  n ≥ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l709_70961


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_2_eval_0_problem_2_eval_3_l709_70950

theorem problem_1 : -(-2)^2 + |(-Real.sqrt 3)| - 2 * Real.sin (π / 3) + (1 / 2)⁻¹ = -2 := by sorry

theorem problem_2 (m : ℝ) (h : m ≠ 2 ∧ m ≠ -2) : 
  (m / (m - 2) - 2 * m / (m^2 - 4)) + m / (m + 2) = (2 * m^2 - 2 * m) / (m^2 - 4) := by sorry

theorem problem_2_eval_0 : 
  (0 : ℝ) / (0 - 2) - 2 * 0 / (0^2 - 4) + 0 / (0 + 2) = 0 := by sorry

theorem problem_2_eval_3 : 
  (3 : ℝ) / (3 - 2) - 2 * 3 / (3^2 - 4) + 3 / (3 + 2) = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_2_eval_0_problem_2_eval_3_l709_70950


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_equals_three_l709_70938

theorem quadratic_roots_imply_m_equals_three (a m : ℤ) :
  a ≠ 1 →
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
    (a - 1) * x^2 - m * x + a = 0 ∧
    (a - 1) * y^2 - m * y + a = 0) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_equals_three_l709_70938


namespace NUMINAMATH_CALUDE_function_always_positive_range_l709_70998

theorem function_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) ↔ (-4 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_function_always_positive_range_l709_70998


namespace NUMINAMATH_CALUDE_power_equality_n_equals_one_l709_70922

theorem power_equality_n_equals_one :
  ∀ n : ℝ, (256 : ℝ) ^ (1/4 : ℝ) = 4 ^ n → n = 1 := by
sorry

end NUMINAMATH_CALUDE_power_equality_n_equals_one_l709_70922


namespace NUMINAMATH_CALUDE_other_number_is_twenty_l709_70978

theorem other_number_is_twenty (a b : ℤ) : 
  3 * a + 4 * b = 140 → (a = 20 ∨ b = 20) → (a = 20 ∧ b = 20) :=
by sorry

end NUMINAMATH_CALUDE_other_number_is_twenty_l709_70978


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l709_70945

/-- Proves that in a college with 190 girls and 494 total students, the ratio of boys to girls is 152:95 -/
theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : total_students = 494) 
  (h2 : girls = 190) : 
  (total_students - girls) / girls = 152 / 95 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l709_70945


namespace NUMINAMATH_CALUDE_sector_arc_length_l709_70957

-- Define the sector
def Sector (area : ℝ) (angle : ℝ) : Type :=
  {r : ℝ // area = (1/2) * r^2 * angle}

-- Define the theorem
theorem sector_arc_length 
  (s : Sector 4 2) : 
  s.val * 2 = 4 := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l709_70957


namespace NUMINAMATH_CALUDE_square_difference_theorem_l709_70904

theorem square_difference_theorem (x y : ℚ) 
  (h1 : x + 2 * y = 5 / 9) 
  (h2 : x - 2 * y = 1 / 9) : 
  x^2 - 4 * y^2 = 5 / 81 := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l709_70904


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l709_70970

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

end NUMINAMATH_CALUDE_parallel_lines_distance_l709_70970


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l709_70917

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  m_valid : minutes < 60

def add_time_and_duration (t : Time) (d : Duration) : Time :=
  sorry

theorem sunset_time_calculation (sunrise : Time) (daylight : Duration) :
  sunrise.hours = 6 ∧ sunrise.minutes = 45 ∧
  daylight.hours = 11 ∧ daylight.minutes = 36 →
  let sunset := add_time_and_duration sunrise daylight
  sunset.hours = 18 ∧ sunset.minutes = 21 :=
sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l709_70917


namespace NUMINAMATH_CALUDE_outfit_combinations_l709_70926

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

end NUMINAMATH_CALUDE_outfit_combinations_l709_70926


namespace NUMINAMATH_CALUDE_lemonade_intermission_l709_70900

theorem lemonade_intermission (total : ℝ) (first : ℝ) (third : ℝ) (second : ℝ)
  (h_total : total = 0.92)
  (h_first : first = 0.25)
  (h_third : third = 0.25)
  (h_sum : total = first + second + third) :
  second = 0.42 := by
sorry

end NUMINAMATH_CALUDE_lemonade_intermission_l709_70900


namespace NUMINAMATH_CALUDE_final_sum_theorem_l709_70979

def num_participants : ℕ := 43

def calculator_operation (n : ℕ) (initial_value : ℤ) : ℤ :=
  match initial_value with
  | 2 => 2^(2^n)
  | 1 => 1
  | -1 => (-1)^n
  | _ => initial_value

theorem final_sum_theorem :
  calculator_operation num_participants 2 +
  calculator_operation num_participants 1 +
  calculator_operation num_participants (-1) = 2^(2^num_participants) := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l709_70979


namespace NUMINAMATH_CALUDE_new_continental_math_institute_enrollment_l709_70953

theorem new_continental_math_institute_enrollment :
  ∃! n : ℕ, n < 550 ∧ n % 19 = 15 ∧ n % 17 = 10 ∧ n = 509 := by
  sorry

end NUMINAMATH_CALUDE_new_continental_math_institute_enrollment_l709_70953


namespace NUMINAMATH_CALUDE_min_both_composers_l709_70958

theorem min_both_composers (total : ℕ) (mozart : ℕ) (beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : mozart = 95)
  (h_beethoven : beethoven = 80)
  : ∃ (both : ℕ), both ≥ mozart + beethoven - total ∧ both = 40 :=
sorry

end NUMINAMATH_CALUDE_min_both_composers_l709_70958


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_equals_three_l709_70954

theorem quadratic_root_implies_u_equals_three (u : ℝ) : 
  (6 * ((-19 + Real.sqrt 289) / 12)^2 + 19 * ((-19 + Real.sqrt 289) / 12) + u = 0) → u = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_equals_three_l709_70954


namespace NUMINAMATH_CALUDE_grid_paths_eq_binom_binom_eq_factorial_div_grid_paths_eq_factorial_div_l709_70942

/-- The number of paths from (0,0) to (n,m) on a grid where only north and east movements are allowed -/
def grid_paths (m n : ℕ) : ℕ := sorry

/-- The binomial coefficient -/
def binom (n k : ℕ) : ℕ := sorry

theorem grid_paths_eq_binom (m n : ℕ) :
  grid_paths m n = binom (m + n) m :=
sorry

theorem binom_eq_factorial_div (n k : ℕ) :
  binom n k = (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) :=
sorry

theorem grid_paths_eq_factorial_div (m n : ℕ) :
  grid_paths m n = (Nat.factorial (m + n)) / ((Nat.factorial m) * (Nat.factorial n)) :=
sorry

end NUMINAMATH_CALUDE_grid_paths_eq_binom_binom_eq_factorial_div_grid_paths_eq_factorial_div_l709_70942


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l709_70939

theorem two_digit_number_interchange (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 → (10 * x + y) - (10 * y + x) = 54 → x - y = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l709_70939


namespace NUMINAMATH_CALUDE_triangle_side_length_l709_70991

/-- Given a triangle DEF with sides d, e, and f, where d = 7, e = 3, and cos(D - E) = 39/40,
    prove that the length of side f is equal to √(9937)/10. -/
theorem triangle_side_length (D E F : ℝ) (d e f : ℝ) : 
  d = 7 → 
  e = 3 → 
  Real.cos (D - E) = 39 / 40 → 
  f = Real.sqrt 9937 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l709_70991


namespace NUMINAMATH_CALUDE_unique_solution_l709_70940

theorem unique_solution (a b c : ℕ+) 
  (h1 : (Nat.gcd a.val b.val) = 1 ∧ (Nat.gcd b.val c.val) = 1 ∧ (Nat.gcd c.val a.val) = 1)
  (h2 : (a.val^2 + b.val) ∣ (b.val^2 + c.val))
  (h3 : (b.val^2 + c.val) ∣ (c.val^2 + a.val))
  (h4 : ∀ p : ℕ, Nat.Prime p → p ∣ (a.val^2 + b.val) → p % 7 ≠ 1) :
  a = 1 ∧ b = 1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l709_70940


namespace NUMINAMATH_CALUDE_fraction_equalities_l709_70928

theorem fraction_equalities : 
  (126 : ℚ) / 84 = 21 / 18 ∧ (268 : ℚ) / 335 = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equalities_l709_70928


namespace NUMINAMATH_CALUDE_second_polygon_sides_l709_70964

theorem second_polygon_sides (n : ℕ) (s : ℝ) : 
  s > 0 → 
  50 * (3 * s) = n * s → 
  n = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l709_70964


namespace NUMINAMATH_CALUDE_divisibility_condition_pairs_l709_70901

theorem divisibility_condition_pairs :
  ∀ m n : ℕ+,
  (∃ k : ℤ, (m : ℤ) + (n : ℤ)^2 = k * ((m : ℤ)^2 - (n : ℤ))) →
  (∃ l : ℤ, (n : ℤ) + (m : ℤ)^2 = l * ((n : ℤ)^2 - (m : ℤ))) →
  ((m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨
   (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_pairs_l709_70901


namespace NUMINAMATH_CALUDE_justin_reading_requirement_l709_70975

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

end NUMINAMATH_CALUDE_justin_reading_requirement_l709_70975


namespace NUMINAMATH_CALUDE_cindy_calculation_l709_70947

theorem cindy_calculation (x : ℝ) (h : (x + 7) * 5 = 260) : 5 * x + 7 = 232 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l709_70947


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l709_70988

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, 1 + 1996 * m + 1998 * n = m * n ↔
    (m = 1999 ∧ n = 1997^2 + 1996) ∨
    (m = 3995 ∧ n = 3993) ∨
    (m = 1997^2 + 1998 ∧ n = 1997) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l709_70988


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l709_70908

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l709_70908


namespace NUMINAMATH_CALUDE_meeting_percentage_is_25_percent_l709_70966

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 30

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the percentage of the work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_25_percent :
  meeting_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_25_percent_l709_70966


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l709_70920

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

end NUMINAMATH_CALUDE_compound_oxygen_count_l709_70920


namespace NUMINAMATH_CALUDE_fiftieth_term_l709_70906

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem fiftieth_term (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 50 = 99 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_l709_70906


namespace NUMINAMATH_CALUDE_systematic_sampling_prob_example_l709_70909

/-- Represents the probability of selection in systematic sampling -/
def systematic_sampling_probability (sample_size : ℕ) (population_size : ℕ) : ℚ :=
  sample_size / population_size

/-- Theorem: In systematic sampling with a sample size of 15 and a population size of 152,
    the probability of each person being selected is 15/152 -/
theorem systematic_sampling_prob_example :
  systematic_sampling_probability 15 152 = 15 / 152 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_prob_example_l709_70909


namespace NUMINAMATH_CALUDE_difference_of_squares_l709_70996

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l709_70996


namespace NUMINAMATH_CALUDE_greatest_prime_producing_integer_l709_70959

def f (x : ℤ) : ℤ := |5 * x^2 - 52 * x + 21|

def is_greatest_prime_producing_integer (n : ℤ) : Prop :=
  Nat.Prime (f n).natAbs ∧
  ∀ m : ℤ, m > n → ¬(Nat.Prime (f m).natAbs)

theorem greatest_prime_producing_integer :
  is_greatest_prime_producing_integer 10 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_producing_integer_l709_70959


namespace NUMINAMATH_CALUDE_students_walking_home_l709_70924

theorem students_walking_home (total : ℚ) 
  (bus : ℚ) (auto : ℚ) (bike : ℚ) (metro : ℚ) :
  bus = 1/3 →
  auto = 1/5 →
  bike = 1/8 →
  metro = 1/15 →
  total = 1 →
  total - (bus + auto + bike + metro) = 11/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l709_70924
