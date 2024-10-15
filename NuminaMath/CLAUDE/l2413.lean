import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_from_means_l2413_241365

theorem sum_of_squares_from_means (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 2.5) :
  x^2 + y^2 + z^2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_from_means_l2413_241365


namespace NUMINAMATH_CALUDE_acrobats_count_correct_l2413_241346

/-- Represents the number of acrobats in the zoo. -/
def acrobats : ℕ := 5

/-- Represents the number of elephants in the zoo. -/
def elephants : ℕ := sorry

/-- Represents the number of camels in the zoo. -/
def camels : ℕ := sorry

/-- The total number of legs in the zoo. -/
def total_legs : ℕ := 58

/-- The total number of heads in the zoo. -/
def total_heads : ℕ := 17

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  acrobats * 2 + elephants * 4 + camels * 4 = total_legs ∧
  acrobats + elephants + camels = total_heads :=
by sorry

end NUMINAMATH_CALUDE_acrobats_count_correct_l2413_241346


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2413_241367

theorem quadratic_equation_solution : ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2413_241367


namespace NUMINAMATH_CALUDE_oranges_left_l2413_241310

theorem oranges_left (total : ℕ) (percentage : ℚ) (remaining : ℕ) : 
  total = 96 → 
  percentage = 48/100 →
  remaining = total - Int.floor (percentage * total) →
  remaining = 50 := by
sorry

end NUMINAMATH_CALUDE_oranges_left_l2413_241310


namespace NUMINAMATH_CALUDE_double_age_in_years_until_double_l2413_241353

/-- The number of years until I'm twice my brother's age -/
def years_until_double : ℕ := 10

/-- My current age -/
def my_current_age : ℕ := 20

/-- My brother's current age -/
def brothers_current_age : ℕ := my_current_age - years_until_double

theorem double_age_in_years_until_double :
  (my_current_age + years_until_double) = 2 * (brothers_current_age + years_until_double) ∧
  (my_current_age + years_until_double) + (brothers_current_age + years_until_double) = 45 :=
by sorry

end NUMINAMATH_CALUDE_double_age_in_years_until_double_l2413_241353


namespace NUMINAMATH_CALUDE_lighthouse_signals_lighthouse_signals_minimum_l2413_241302

theorem lighthouse_signals (x : ℕ) : 
  (x % 15 = 2 ∧ x % 28 = 8) → x ≥ 92 :=
by sorry

theorem lighthouse_signals_minimum : 
  ∃ (x : ℕ), x % 15 = 2 ∧ x % 28 = 8 ∧ x = 92 :=
by sorry

end NUMINAMATH_CALUDE_lighthouse_signals_lighthouse_signals_minimum_l2413_241302


namespace NUMINAMATH_CALUDE_latticePoindsInsideTriangleABO_l2413_241389

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0, 30)
def B : ℤ × ℤ := (20, 10)
def O : ℤ × ℤ := (0, 0)

-- Define a function to calculate the area of a triangle
def triangleArea (p1 p2 p3 : ℤ × ℤ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Define Pick's theorem
def picksTheorem (S : ℚ) (N L : ℤ) : Prop :=
  S = N + L / 2 - 1

-- State the theorem
theorem latticePoindsInsideTriangleABO :
  ∃ (N L : ℤ),
    picksTheorem (triangleArea A B O) N L ∧
    L = 60 ∧
    N = 271 :=
  sorry

end NUMINAMATH_CALUDE_latticePoindsInsideTriangleABO_l2413_241389


namespace NUMINAMATH_CALUDE_sum_of_ages_in_future_l2413_241393

-- Define Will's age 3 years ago
def will_age_3_years_ago : ℕ := 4

-- Define the current year (relative to the problem's frame)
def current_year : ℕ := 3

-- Define the future year we're interested in
def future_year : ℕ := 5

-- Define Will's current age
def will_current_age : ℕ := will_age_3_years_ago + current_year

-- Define Diane's current age
def diane_current_age : ℕ := 2 * will_current_age

-- Define Will's future age
def will_future_age : ℕ := will_current_age + future_year

-- Define Diane's future age
def diane_future_age : ℕ := diane_current_age + future_year

-- Theorem to prove
theorem sum_of_ages_in_future : will_future_age + diane_future_age = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_future_l2413_241393


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2413_241381

/-- An arithmetic sequence with common difference d ≠ 0 -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def FormGeometricSequence (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = b * q

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ArithmeticSequence a d)
  (h_geom : FormGeometricSequence (a 2) (a 3) (a 6)) :
  ∃ q : ℝ, q = 3 ∧ FormGeometricSequence (a 2) (a 3) (a 6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2413_241381


namespace NUMINAMATH_CALUDE_probability_of_specific_match_l2413_241301

/-- The number of teams in the tournament -/
def num_teams : ℕ := 128

/-- The probability of two specific teams playing each other in a single elimination tournament -/
def probability_of_match (n : ℕ) : ℚ :=
  (n - 1) / (n * (n - 1) / 2)

/-- Theorem: In a single elimination tournament with 128 equally strong teams,
    the probability of two specific teams playing each other is 1/64 -/
theorem probability_of_specific_match :
  probability_of_match num_teams = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_match_l2413_241301


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2413_241322

theorem solution_set_of_inequality (x : ℝ) :
  (2 / (x - 1) ≥ 1) ↔ (1 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2413_241322


namespace NUMINAMATH_CALUDE_apple_difference_l2413_241312

theorem apple_difference (adam_apples jackie_apples : ℕ) 
  (adam_count : adam_apples = 9) 
  (jackie_count : jackie_apples = 10) : 
  jackie_apples - adam_apples = 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l2413_241312


namespace NUMINAMATH_CALUDE_field_length_is_96_l2413_241324

/-- Proves that the length of a rectangular field is 96 meters given specific conditions -/
theorem field_length_is_96 (w : ℝ) (l : ℝ) : 
  l = 2 * w →                   -- length is double the width
  64 = (1 / 72) * (l * w) →     -- area of pond (8^2) is 1/72 of field area
  l = 96 := by
sorry

end NUMINAMATH_CALUDE_field_length_is_96_l2413_241324


namespace NUMINAMATH_CALUDE_no_rational_solution_l2413_241392

theorem no_rational_solution : ¬∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (1 : ℚ) / (x - y)^2 + (1 : ℚ) / (y - z)^2 + (1 : ℚ) / (z - x)^2 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l2413_241392


namespace NUMINAMATH_CALUDE_flower_bee_butterfly_difference_l2413_241376

theorem flower_bee_butterfly_difference (flowers bees butterflies : ℕ) 
  (h1 : flowers = 12) 
  (h2 : bees = 7) 
  (h3 : butterflies = 4) : 
  (flowers - bees) - butterflies = 1 := by
  sorry

end NUMINAMATH_CALUDE_flower_bee_butterfly_difference_l2413_241376


namespace NUMINAMATH_CALUDE_percentage_saved_approx_l2413_241363

/-- Represents the discount information for each day of the sale -/
structure DayDiscount where
  minQuantity : Nat
  discountedQuantity : Nat
  discountedPrice : Nat

/-- Calculates the savings for a given day's discount -/
def calculateSavings (discount : DayDiscount) : Nat :=
  discount.minQuantity - discount.discountedPrice

/-- Calculates the total savings and original price for all days -/
def calculateTotals (discounts : List DayDiscount) : (Nat × Nat) :=
  let savings := discounts.map calculateSavings |>.sum
  let originalPrice := discounts.map (fun d => d.minQuantity) |>.sum
  (savings, originalPrice)

/-- The discounts for each day of the five-day sale -/
def saleDays : List DayDiscount := [
  { minQuantity := 11, discountedQuantity := 12, discountedPrice := 4 },
  { minQuantity := 15, discountedQuantity := 15, discountedPrice := 5 },
  { minQuantity := 18, discountedQuantity := 18, discountedPrice := 6 },
  { minQuantity := 21, discountedQuantity := 25, discountedPrice := 8 },
  { minQuantity := 26, discountedQuantity := 30, discountedPrice := 10 }
]

/-- Theorem stating that the percentage saved is approximately 63.74% -/
theorem percentage_saved_approx (ε : ℝ) (h : ε > 0) :
  let (savings, originalPrice) := calculateTotals saleDays
  let percentageSaved := (savings : ℝ) / (originalPrice : ℝ) * 100
  |percentageSaved - 63.74| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_saved_approx_l2413_241363


namespace NUMINAMATH_CALUDE_smartphone_sales_l2413_241334

theorem smartphone_sales (units_at_400 price_400 price_800 : ℝ) 
  (h1 : units_at_400 = 20)
  (h2 : price_400 = 400)
  (h3 : price_800 = 800)
  (h4 : ∀ (p c : ℝ), p * c = units_at_400 * price_400) :
  (units_at_400 * price_400) / price_800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_sales_l2413_241334


namespace NUMINAMATH_CALUDE_three_quadrilaterals_with_circumcenter_l2413_241354

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A point is equidistant from all vertices of a quadrilateral. -/
def has_circumcenter (q : Quadrilateral) : Prop :=
  ∃ (c : ℝ × ℝ), ∀ (i : Fin 4), dist c (q.vertices i) = dist c (q.vertices 0)

/-- A kite is a quadrilateral with two pairs of adjacent congruent sides. -/
def is_kite (q : Quadrilateral) : Prop := sorry

/-- A quadrilateral has exactly two right angles. -/
def has_two_right_angles (q : Quadrilateral) : Prop := sorry

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- A rhombus is a quadrilateral with all sides equal. -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- An equilateral trapezoid is a trapezoid with the non-parallel sides equal. -/
def is_equilateral_trapezoid (q : Quadrilateral) : Prop := sorry

/-- A quadrilateral can be inscribed in a circle. -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- The main theorem stating that exactly 3 types of the given quadrilaterals have a circumcenter. -/
theorem three_quadrilaterals_with_circumcenter : 
  ∃ (a b c : Quadrilateral),
    (is_kite a ∧ has_two_right_angles a ∧ has_circumcenter a) ∧
    (is_square b ∧ has_circumcenter b) ∧
    (is_equilateral_trapezoid c ∧ is_cyclic c ∧ has_circumcenter c) ∧
    (∀ (d : Quadrilateral), 
      (is_rhombus d ∧ ¬is_square d) → ¬has_circumcenter d) ∧
    (∀ (e : Quadrilateral),
      has_circumcenter e → 
      (e = a ∨ e = b ∨ e = c)) :=
sorry

end NUMINAMATH_CALUDE_three_quadrilaterals_with_circumcenter_l2413_241354


namespace NUMINAMATH_CALUDE_expression_simplification_l2413_241348

def simplify_expression (a b : ℤ) : ℤ :=
  -2 * (10 * a^2 + 2 * a * b + 3 * b^2) + 3 * (5 * a^2 - 4 * a * b)

theorem expression_simplification :
  simplify_expression 1 (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2413_241348


namespace NUMINAMATH_CALUDE_days_to_empty_tube_l2413_241391

-- Define the volume of the gel tube in mL
def tube_volume : ℝ := 128

-- Define the daily usage of gel in mL
def daily_usage : ℝ := 4

-- Theorem statement
theorem days_to_empty_tube : 
  (tube_volume / daily_usage : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_days_to_empty_tube_l2413_241391


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2413_241357

theorem polynomial_divisibility (a b : ℚ) : 
  (∀ x : ℚ, (x^2 - x - 2) ∣ (a * x^4 + b * x^2 + 1)) ↔ 
  (a = 1/4 ∧ b = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2413_241357


namespace NUMINAMATH_CALUDE_abs_k_less_than_abs_b_l2413_241326

/-- Given a linear function y = kx + b, prove that |k| < |b| under certain conditions --/
theorem abs_k_less_than_abs_b (k b : ℝ) : 
  (∀ x y, y = k * x + b) →  -- The function is of the form y = kx + b
  (b > 0) →  -- The y-intercept is positive
  (0 < k + b) →  -- The point (1, k+b) is above the x-axis
  (k + b < b) →  -- The point (1, k+b) is below b
  |k| < |b| := by
sorry


end NUMINAMATH_CALUDE_abs_k_less_than_abs_b_l2413_241326


namespace NUMINAMATH_CALUDE_point_difference_l2413_241314

def wildcats_rate : ℝ := 2.5
def panthers_rate : ℝ := 1.3
def half_duration : ℝ := 24

theorem point_difference : 
  wildcats_rate * half_duration - panthers_rate * half_duration = 28.8 := by
sorry

end NUMINAMATH_CALUDE_point_difference_l2413_241314


namespace NUMINAMATH_CALUDE_sum_342_78_base5_l2413_241364

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list representing a number in base 5 to a natural number in base 10 -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_342_78_base5 :
  addBase5 (toBase5 342) (toBase5 78) = [3, 1, 4, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_342_78_base5_l2413_241364


namespace NUMINAMATH_CALUDE_dark_lord_squads_l2413_241397

/-- The number of squads needed to transport swords --/
def num_squads (total_weight : ℕ) (orcs_per_squad : ℕ) (weight_per_orc : ℕ) : ℕ :=
  total_weight / (orcs_per_squad * weight_per_orc)

/-- Proof that 10 squads are needed for the given conditions --/
theorem dark_lord_squads :
  num_squads 1200 8 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dark_lord_squads_l2413_241397


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2413_241350

/-- Given that x₁ and x₂ are real roots of a quadratic equation, this theorem proves
    properties about y = x₁² + x₂² as a function of m. -/
theorem quadratic_roots_sum_squares (m : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - 2*(m-1)*x₁ + m + 1 = 0 →
  x₂^2 - 2*(m-1)*x₂ + m + 1 = 0 →
  let y := x₁^2 + x₂^2
  -- 1. y as a function of m
  y = 4*m^2 - 10*m + 2 ∧
  -- 2. Minimum value of y
  (∃ (m₀ : ℝ), y = 6 ∧ ∀ (m' : ℝ), y ≥ 6) ∧
  -- 3. y ≥ 6 for all valid m
  y ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2413_241350


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2413_241330

theorem polar_to_rectangular_conversion 
  (r φ x y : ℝ) 
  (h1 : r = 7 / (2 * Real.cos φ - 5 * Real.sin φ))
  (h2 : x = r * Real.cos φ)
  (h3 : y = r * Real.sin φ) :
  2 * x - 5 * y = 7 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2413_241330


namespace NUMINAMATH_CALUDE_C_equiv_C_param_l2413_241335

/-- A semicircular curve C in the polar coordinate system -/
def C : Set (ℝ × ℝ) := {(p, θ) | p = 2 * Real.cos θ ∧ 0 ≤ θ ∧ θ ≤ Real.pi / 2}

/-- The parametric representation of curve C -/
def C_param : Set (ℝ × ℝ) := {(x, y) | ∃ α, 0 ≤ α ∧ α ≤ Real.pi ∧ x = 1 + Real.cos α ∧ y = Real.sin α}

/-- Theorem stating that the parametric representation is equivalent to the polar representation -/
theorem C_equiv_C_param : C = C_param := by sorry

end NUMINAMATH_CALUDE_C_equiv_C_param_l2413_241335


namespace NUMINAMATH_CALUDE_columbia_arrangements_l2413_241356

def columbia_letters : Nat := 9
def repeated_i : Nat := 2
def repeated_u : Nat := 2

theorem columbia_arrangements :
  (columbia_letters.factorial) / (repeated_i.factorial * repeated_u.factorial) = 90720 := by
  sorry

end NUMINAMATH_CALUDE_columbia_arrangements_l2413_241356


namespace NUMINAMATH_CALUDE_cows_gifted_is_eight_l2413_241380

/-- Calculates the number of cows given as a gift -/
def cows_gifted (initial : ℕ) (died : ℕ) (sold : ℕ) (increased : ℕ) (bought : ℕ) (total : ℕ) : ℕ :=
  total - (initial - died - sold + increased + bought)

/-- Theorem stating that the number of cows gifted is 8 -/
theorem cows_gifted_is_eight :
  cows_gifted 39 25 6 24 43 83 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cows_gifted_is_eight_l2413_241380


namespace NUMINAMATH_CALUDE_courses_difference_count_l2413_241382

/-- The number of available courses -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways to choose courses with at least one difference -/
def ways_with_difference : ℕ := 30

/-- Theorem stating that the number of ways with at least one course different is 30 -/
theorem courses_difference_count :
  (total_courses.choose courses_per_person * courses_per_person.choose courses_per_person) +
  (total_courses.choose 1 * (total_courses - 1).choose 1 * (total_courses - 2).choose 1) =
  ways_with_difference :=
sorry

end NUMINAMATH_CALUDE_courses_difference_count_l2413_241382


namespace NUMINAMATH_CALUDE_computer_price_increase_l2413_241333

theorem computer_price_increase (y : ℝ) (h1 : 2 * y = 540) : 
  y * (1 + 0.3) = 351 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2413_241333


namespace NUMINAMATH_CALUDE_x_values_l2413_241342

def A (x : ℝ) : Set ℝ := {x, x^2}

theorem x_values (x : ℝ) (h : 1 ∈ A x) : x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l2413_241342


namespace NUMINAMATH_CALUDE_perfect_square_product_divisible_by_12_l2413_241320

theorem perfect_square_product_divisible_by_12 (n : ℤ) : 
  12 ∣ (n^2 * (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_product_divisible_by_12_l2413_241320


namespace NUMINAMATH_CALUDE_five_percent_problem_l2413_241304

theorem five_percent_problem (x : ℝ) : (5 / 100) * x = 12.75 → x = 255 := by
  sorry

end NUMINAMATH_CALUDE_five_percent_problem_l2413_241304


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l2413_241307

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_pos : a * b * c > 0) : 
  1 / a + 1 / b + 1 / c < 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l2413_241307


namespace NUMINAMATH_CALUDE_square_between_prime_sums_l2413_241355

/-- Sum of the first n prime numbers -/
def S (n : ℕ) : ℕ := sorry

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem square_between_prime_sums :
  ∀ n : ℕ, n > 0 → ∃ k : ℕ, S n < k^2 ∧ k^2 < S (n + 1) :=
sorry

end NUMINAMATH_CALUDE_square_between_prime_sums_l2413_241355


namespace NUMINAMATH_CALUDE_unique_z_value_l2413_241378

theorem unique_z_value : ∃! z : ℝ,
  (∃ x : ℤ, x = ⌊z⌋ ∧ 3 * x^2 + 19 * x - 84 = 0) ∧
  (∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ y = z - ⌊z⌋ ∧ 4 * y^2 - 14 * y + 6 = 0) ∧
  z = -11 := by
sorry

end NUMINAMATH_CALUDE_unique_z_value_l2413_241378


namespace NUMINAMATH_CALUDE_gcd_100_450_l2413_241352

theorem gcd_100_450 : Nat.gcd 100 450 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gcd_100_450_l2413_241352


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2413_241340

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2413_241340


namespace NUMINAMATH_CALUDE_car_speed_calculation_car_speed_is_21_l2413_241327

/-- Calculates the speed of a car given the walking speed of a person and the number of steps taken --/
theorem car_speed_calculation (walking_speed : ℝ) (steps_while_car_visible : ℕ) (steps_after_car_disappeared : ℕ) : ℝ :=
  let total_steps := steps_while_car_visible + steps_after_car_disappeared
  let speed_ratio := total_steps / steps_while_car_visible
  speed_ratio * walking_speed

/-- Proves that the car's speed is 21 km/h given the specific conditions --/
theorem car_speed_is_21 : 
  car_speed_calculation 3.5 27 135 = 21 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_car_speed_is_21_l2413_241327


namespace NUMINAMATH_CALUDE_tile_difference_8_9_and_9_10_l2413_241313

/-- Represents the number of tiles in the nth square of the sequence -/
def tiles (n : ℕ) : ℕ := n^2

/-- The difference in tiles between two consecutive squares -/
def tile_difference (n : ℕ) : ℕ := tiles (n + 1) - tiles n

theorem tile_difference_8_9_and_9_10 :
  (tile_difference 8 = 17) ∧ (tile_difference 9 = 19) := by
  sorry

end NUMINAMATH_CALUDE_tile_difference_8_9_and_9_10_l2413_241313


namespace NUMINAMATH_CALUDE_no_integer_solution_l2413_241358

theorem no_integer_solution :
  ¬ ∃ (a b x y : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧
    a * x - b * y = 16 ∧
    a * y + b * x = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2413_241358


namespace NUMINAMATH_CALUDE_number_of_children_selected_l2413_241366

def total_boys : ℕ := 5
def total_girls : ℕ := 5
def prob_three_boys_three_girls : ℚ := 100 / 210

theorem number_of_children_selected (n : ℕ) : 
  (total_boys = 5 ∧ total_girls = 5 ∧ 
   prob_three_boys_three_girls = 100 / (Nat.choose (total_boys + total_girls) n)) → 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_selected_l2413_241366


namespace NUMINAMATH_CALUDE_koch_snowflake_area_l2413_241308

/-- Given a sequence of curves P₀, P₁, P₂, ..., where:
    1. P₀ is an equilateral triangle with area 1
    2. Pₖ₊₁ is obtained from Pₖ by trisecting each side, constructing an equilateral 
       triangle on the middle segment, and removing the middle segment
    3. Sₙ is the area enclosed by curve Pₙ
    
    This theorem states the formula for Sₙ and its limit as n approaches infinity. -/
theorem koch_snowflake_area (n : ℕ) : 
  ∃ (S : ℕ → ℝ), 
    (∀ k, S k = (47/20) * (1 - (4/9)^k)) ∧ 
    (∀ ε > 0, ∃ N, ∀ n ≥ N, |S n - 47/20| < ε) := by
  sorry

end NUMINAMATH_CALUDE_koch_snowflake_area_l2413_241308


namespace NUMINAMATH_CALUDE_no_solution_for_floor_sum_l2413_241339

theorem no_solution_for_floor_sum (x : ℝ) : 
  ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_floor_sum_l2413_241339


namespace NUMINAMATH_CALUDE_complex_equation_l2413_241369

theorem complex_equation (z : ℂ) (h : z = 1 + I) : z^2 + 2 / z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l2413_241369


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2413_241385

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2413_241385


namespace NUMINAMATH_CALUDE_purchasing_plan_comparison_pricing_strategy_comparison_l2413_241383

-- Purchasing plans comparison
theorem purchasing_plan_comparison 
  (a b : ℝ) (m n : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : m > 0) (h5 : n > 0) :
  (2 * a * b) / (a + b) < (a + b) / 2 := by
sorry

-- Pricing strategies comparison
theorem pricing_strategy_comparison 
  (p q : ℝ) (h : p ≠ q) :
  100 * (1 + p) * (1 + q) < 100 * (1 + (p + q) / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_purchasing_plan_comparison_pricing_strategy_comparison_l2413_241383


namespace NUMINAMATH_CALUDE_special_polygon_properties_l2413_241325

/-- A polygon where the sum of interior angles is more than three times the sum of exterior angles by 180° --/
structure SpecialPolygon where
  n : ℕ
  interior_sum : ℝ
  exterior_sum : ℝ
  h : interior_sum = 3 * exterior_sum + 180

theorem special_polygon_properties (p : SpecialPolygon) :
  p.n = 9 ∧ p.n - 3 = 6 := by sorry


end NUMINAMATH_CALUDE_special_polygon_properties_l2413_241325


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2413_241303

theorem trigonometric_inequality : 
  let a := (1/2) * Real.cos (6 * (π/180)) - (Real.sqrt 3 / 2) * Real.sin (6 * (π/180))
  let b := (2 * Real.tan (13 * (π/180))) / (1 + Real.tan (13 * (π/180)) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * (π/180))) / 2)
  a < c ∧ c < b := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2413_241303


namespace NUMINAMATH_CALUDE_car_profit_percentage_l2413_241344

theorem car_profit_percentage (P : ℝ) (P_positive : P > 0) : 
  let discount_rate : ℝ := 0.20
  let increase_rate : ℝ := 0.70
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - P
  let profit_percentage : ℝ := (profit / P) * 100
  profit_percentage = 36 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l2413_241344


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2413_241345

/-- The complex number z = 2i / (1 + i) has both positive real and imaginary parts -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 * Complex.I) / (1 + Complex.I)
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2413_241345


namespace NUMINAMATH_CALUDE_lakers_win_probability_l2413_241316

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 3/4

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 1 - p_celtics

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Lakers winning the NBA finals in exactly 7 games -/
def lakers_win_in_seven : ℚ := 540/16384

theorem lakers_win_probability :
  lakers_win_in_seven = (Nat.choose 6 3 : ℚ) * p_lakers^3 * p_celtics^3 * p_lakers :=
sorry

end NUMINAMATH_CALUDE_lakers_win_probability_l2413_241316


namespace NUMINAMATH_CALUDE_green_valley_olympiad_l2413_241311

theorem green_valley_olympiad (j s : ℕ) (hj : j > 0) (hs : s > 0) 
  (h_participation : (1 : ℚ) / 3 * j = (2 : ℚ) / 3 * s) : j = 2 * s :=
sorry

end NUMINAMATH_CALUDE_green_valley_olympiad_l2413_241311


namespace NUMINAMATH_CALUDE_coin_representation_l2413_241337

def is_representable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 3 * a + 5 * b

theorem coin_representation :
  ∀ n : ℕ, n > 0 → (is_representable n ↔ n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) :=
by sorry

end NUMINAMATH_CALUDE_coin_representation_l2413_241337


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_l2413_241399

theorem fifth_root_of_unity (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * u^4 + q * u^3 + r * u^2 + s * u + t = 0)
  (h2 : q * u^4 + r * u^3 + s * u^2 + t * u + p = 0) :
  u^5 = 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_l2413_241399


namespace NUMINAMATH_CALUDE_point_in_region_l2413_241359

def region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

theorem point_in_region :
  region 0 1 ∧ ¬region 5 0 ∧ ¬region 0 7 ∧ ¬region 2 3 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_l2413_241359


namespace NUMINAMATH_CALUDE_b_72_mod_50_l2413_241373

/-- The sequence b_n defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The theorem stating that b_72 is congruent to 2 modulo 50 -/
theorem b_72_mod_50 : b 72 ≡ 2 [ZMOD 50] := by
  sorry

end NUMINAMATH_CALUDE_b_72_mod_50_l2413_241373


namespace NUMINAMATH_CALUDE_library_growth_rate_l2413_241375

theorem library_growth_rate (initial_collection : ℝ) (final_collection : ℝ) (years : ℝ) :
  initial_collection = 100000 →
  final_collection = 144000 →
  years = 2 →
  let growth_rate := ((final_collection / initial_collection) ^ (1 / years)) - 1
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_library_growth_rate_l2413_241375


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l2413_241306

def n : ℕ := 1209600000

/-- The fifth-largest divisor of n -/
def fifth_largest_divisor : ℕ := 75600000

/-- A function that returns the kth largest divisor of a number -/
def kth_largest_divisor (m k : ℕ) : ℕ := sorry

theorem fifth_largest_divisor_of_n :
  kth_largest_divisor n 5 = fifth_largest_divisor := by sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l2413_241306


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l2413_241338

/-- A cone with three spheres inside it -/
structure ConeWithSpheres where
  R : ℝ  -- radius of the base of the cone
  r : ℝ  -- radius of each sphere
  slant_height : ℝ  -- slant height of the cone
  spheres_touch : Bool  -- spheres touch each other externally
  two_touch_base : Bool  -- two spheres touch the lateral surface and base
  third_in_plane : Bool  -- third sphere touches at a point in the same plane as centers

/-- The properties of the cone and spheres arrangement -/
def cone_sphere_properties (c : ConeWithSpheres) : Prop :=
  c.R > 0 ∧ c.r > 0 ∧
  c.slant_height = 2 * c.R ∧  -- base diameter equals slant height
  c.spheres_touch ∧
  c.two_touch_base ∧
  c.third_in_plane

/-- The theorem stating the ratio of cone base radius to sphere radius -/
theorem cone_sphere_ratio (c : ConeWithSpheres) 
  (h : cone_sphere_properties c) : c.R / c.r = 5 / 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l2413_241338


namespace NUMINAMATH_CALUDE_negative_product_implies_odd_negatives_l2413_241368

theorem negative_product_implies_odd_negatives (a b c : ℝ) : 
  a * b * c < 0 → (a < 0 ∧ b < 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0) ∨ 
                   (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a > 0 ∧ b > 0 ∧ c < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_product_implies_odd_negatives_l2413_241368


namespace NUMINAMATH_CALUDE_concurrent_circles_and_collinearity_l2413_241371

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def D (triangle : Triangle) : Point := sorry
def E (triangle : Triangle) : Point := sorry
def F (triangle : Triangle) : Point := sorry

-- Define the circles
def circleAEF (triangle : Triangle) : Circle := sorry
def circleBFD (triangle : Triangle) : Circle := sorry
def circleCDE (triangle : Triangle) : Circle := sorry

-- Define concurrency
def areConcurrent (c1 c2 c3 : Circle) : Prop := sorry

-- Define collinearity
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

-- Define if a point lies on a circle
def liesOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def circumcircle (triangle : Triangle) : Circle := sorry

-- The theorem to prove
theorem concurrent_circles_and_collinearity 
  (triangle : Triangle) : 
  areConcurrent (circleAEF triangle) (circleBFD triangle) (circleCDE triangle) ∧ 
  (∃ M : Point, 
    liesOnCircle M (circleAEF triangle) ∧ 
    liesOnCircle M (circleBFD triangle) ∧ 
    liesOnCircle M (circleCDE triangle) ∧
    (liesOnCircle M (circumcircle triangle) ↔ 
      areCollinear (D triangle) (E triangle) (F triangle))) := by
  sorry

end NUMINAMATH_CALUDE_concurrent_circles_and_collinearity_l2413_241371


namespace NUMINAMATH_CALUDE_contrapositive_square_inequality_l2413_241388

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x > y) → ¬(x^2 > y^2)) ↔ (x ≤ y → x^2 ≤ y^2) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_square_inequality_l2413_241388


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l2413_241379

theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let r := (d1 * d2) / (4 * a)
  r = 60 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l2413_241379


namespace NUMINAMATH_CALUDE_walk_distance_before_rest_l2413_241370

theorem walk_distance_before_rest 
  (total_distance : ℝ) 
  (distance_after_rest : ℝ) 
  (h1 : total_distance = 1) 
  (h2 : distance_after_rest = 0.25) : 
  total_distance - distance_after_rest = 0.75 := by
sorry

end NUMINAMATH_CALUDE_walk_distance_before_rest_l2413_241370


namespace NUMINAMATH_CALUDE_gcd_456_357_l2413_241377

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_gcd_456_357_l2413_241377


namespace NUMINAMATH_CALUDE_coefficient_of_a_half_power_l2413_241328

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (a - 1/√a)^5
def expansion (a : ℝ) : ℝ → ℝ := sorry

-- Theorem statement
theorem coefficient_of_a_half_power (a : ℝ) :
  ∃ (c : ℝ), c = -10 ∧ 
  (∀ (k : ℕ), k ≠ 3 → (binomial 5 k) * (-1)^k * a^(5 - k - k/2) ≠ c * a^(1/2)) ∧
  (binomial 5 3) * (-1)^3 * a^(5 - 3 - 3/2) = c * a^(1/2) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_a_half_power_l2413_241328


namespace NUMINAMATH_CALUDE_min_sum_squares_l2413_241341

theorem min_sum_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∃ m : ℝ, m = a^2 + b^2 ∧ ∀ c d : ℝ, (∃ y : ℝ, y^4 + c*y^3 + d*y^2 + c*y + 1 = 0) → m ≤ c^2 + d^2) ∧ 
  (∃ n : ℝ, n = 4/5 ∧ n = a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2413_241341


namespace NUMINAMATH_CALUDE_custom_operation_equality_l2413_241396

-- Define the custom operation
def delta (a b : ℝ) : ℝ := a^3 - 2*b

-- State the theorem
theorem custom_operation_equality :
  let x := delta 6 8
  let y := delta 2 7
  delta (5^x) (2^y) = (5^200)^3 - 1/32 := by sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l2413_241396


namespace NUMINAMATH_CALUDE_water_usage_calculation_l2413_241300

/-- Calculates the weekly water usage for baths given the specified parameters. -/
def weekly_water_usage (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (buckets_removed : ℕ) (baths_per_week : ℕ) : ℕ :=
  let total_capacity := bucket_capacity * buckets_to_fill
  let water_removed := bucket_capacity * buckets_removed
  let water_per_bath := total_capacity - water_removed
  water_per_bath * baths_per_week

/-- Theorem stating that the weekly water usage is 9240 ounces given the specified parameters. -/
theorem water_usage_calculation :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_water_usage_calculation_l2413_241300


namespace NUMINAMATH_CALUDE_paint_project_cost_l2413_241361

/-- Calculates the total cost of paint and primer for a house painting project. -/
def total_cost (rooms : ℕ) (primer_cost : ℚ) (primer_discount : ℚ) (paint_cost : ℚ) : ℚ :=
  let discounted_primer_cost := primer_cost * (1 - primer_discount)
  let total_primer_cost := rooms * discounted_primer_cost
  let total_paint_cost := rooms * paint_cost
  total_primer_cost + total_paint_cost

/-- Proves that the total cost for paint and primer is $245.00 under given conditions. -/
theorem paint_project_cost :
  total_cost 5 30 (1/5) 25 = 245 :=
by sorry

end NUMINAMATH_CALUDE_paint_project_cost_l2413_241361


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l2413_241386

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the left focus and left vertex
def F₁ : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (-2, 0)

-- Define the dot product of PF₁ and PA
def dot_product (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + 1) * (x + 2) + y^2

-- Theorem statement
theorem ellipse_dot_product_range :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → 0 ≤ dot_product P ∧ dot_product P ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l2413_241386


namespace NUMINAMATH_CALUDE_derivative_y_l2413_241398

noncomputable def y (x : ℝ) : ℝ := Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

theorem derivative_y (x : ℝ) (h : 2 * x + 3 > 0) :
  deriv y x = (4 * Real.sqrt (x^2 + 3 * x + 2)) / (2 * x + 3) := by sorry

end NUMINAMATH_CALUDE_derivative_y_l2413_241398


namespace NUMINAMATH_CALUDE_berry_ratio_l2413_241336

theorem berry_ratio (total berries stacy steve skylar : ℕ) : 
  total = 1100 →
  stacy = 800 →
  stacy = 4 * steve →
  total = stacy + steve + skylar →
  steve = 2 * skylar := by
sorry

end NUMINAMATH_CALUDE_berry_ratio_l2413_241336


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l2413_241347

/-- Given the total cloth length, total selling price, and loss per metre, 
    calculate the cost price for one metre of cloth. -/
theorem cost_price_per_metre 
  (total_length : ℕ) 
  (total_selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (h1 : total_length = 200)
  (h2 : total_selling_price = 12000)
  (h3 : loss_per_metre = 12) : 
  (total_selling_price + total_length * loss_per_metre) / total_length = 72 := by
  sorry

#check cost_price_per_metre

end NUMINAMATH_CALUDE_cost_price_per_metre_l2413_241347


namespace NUMINAMATH_CALUDE_paulines_garden_l2413_241321

/-- Represents the number of kinds of cucumbers in Pauline's garden -/
def cucumber_kinds : ℕ := sorry

/-- The total number of spaces in the garden -/
def total_spaces : ℕ := 10 * 15

/-- The number of tomatoes planted -/
def tomatoes : ℕ := 3 * 5

/-- The number of cucumbers planted -/
def cucumbers : ℕ := cucumber_kinds * 4

/-- The number of potatoes planted -/
def potatoes : ℕ := 30

/-- The number of additional vegetables that can be planted -/
def additional_vegetables : ℕ := 85

theorem paulines_garden :
  cucumber_kinds = 5 :=
by sorry

end NUMINAMATH_CALUDE_paulines_garden_l2413_241321


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l2413_241318

theorem quadratic_function_sum (a b : ℝ) (h1 : a ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x - 1) → 
  (1 = a * 1^2 + b * 1 - 1) →
  a + b + 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_l2413_241318


namespace NUMINAMATH_CALUDE_coffee_price_increase_l2413_241384

/-- Calculates the percentage increase in coffee price given the original conditions and savings. -/
theorem coffee_price_increase 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (new_quantity : ℕ) 
  (daily_savings : ℝ) 
  (h1 : original_price = 2)
  (h2 : original_quantity = 4)
  (h3 : new_quantity = 2)
  (h4 : daily_savings = 2) : 
  (((original_price * original_quantity - daily_savings) / new_quantity - original_price) / original_price) * 100 = 50 := by
  sorry

#check coffee_price_increase

end NUMINAMATH_CALUDE_coffee_price_increase_l2413_241384


namespace NUMINAMATH_CALUDE_range_of_a_l2413_241360

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def proposition_p (a : ℝ) : Prop :=
  is_monotonically_increasing (fun x => a^x)

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ¬(¬proposition_p a ∧ ¬proposition_q a))
  (h3 : proposition_p a ∨ proposition_q a) :
  a ∈ Set.Ioo 0 1 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2413_241360


namespace NUMINAMATH_CALUDE_triangle_y_value_l2413_241362

-- Define the triangle
structure AcuteTriangle where
  a : ℝ
  y : ℝ
  area_small : ℝ

-- Define the properties of the triangle
def triangle_properties (t : AcuteTriangle) : Prop :=
  t.a > 0 ∧ t.y > 0 ∧
  6 > 0 ∧ 4 > 0 ∧
  t.area_small = 12 ∧
  (6 * (6 + t.y) = t.y * (10 + t.a)) ∧
  (1/2 * 10 * (24 / t.y) = 12)

-- Theorem statement
theorem triangle_y_value (t : AcuteTriangle) 
  (h : triangle_properties t) : t.y = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_y_value_l2413_241362


namespace NUMINAMATH_CALUDE_nonadjacent_arrangements_correct_nonadjacent_arrangements_simplified_l2413_241395

/-- The number of circular arrangements of n people where two specific people are not adjacent -/
def nonadjacent_arrangements (n : ℕ) : ℕ :=
  (n - 3) * (n - 2).factorial

/-- Theorem stating the number of circular arrangements of n people (n ≥ 3) 
    where two specific people are not adjacent -/
theorem nonadjacent_arrangements_correct (n : ℕ) (h : n ≥ 3) :
  nonadjacent_arrangements n = (n - 1).factorial - 2 * (n - 2).factorial :=
by
  sorry

/-- Corollary: The number of arrangements where two specific people are not adjacent
    is equal to (n-3)(n-2)! -/
theorem nonadjacent_arrangements_simplified (n : ℕ) (h : n ≥ 3) :
  nonadjacent_arrangements n = (n - 3) * (n - 2).factorial :=
by
  sorry

end NUMINAMATH_CALUDE_nonadjacent_arrangements_correct_nonadjacent_arrangements_simplified_l2413_241395


namespace NUMINAMATH_CALUDE_point_3_0_on_line_point_0_4_on_line_is_line_equation_main_theorem_l2413_241372

/-- The line passing through points (3, 0) and (0, 4) -/
def line_equation (x y : ℝ) : Prop := 4*x + 3*y - 12 = 0

/-- Point (3, 0) lies on the line -/
theorem point_3_0_on_line : line_equation 3 0 := by sorry

/-- Point (0, 4) lies on the line -/
theorem point_0_4_on_line : line_equation 0 4 := by sorry

/-- The equation represents a line -/
theorem is_line_equation : ∃ (m b : ℝ), ∀ (x y : ℝ), line_equation x y ↔ y = m*x + b := by sorry

/-- Main theorem: The given equation represents the unique line passing through (3, 0) and (0, 4) -/
theorem main_theorem : 
  ∀ (f : ℝ → ℝ → Prop), 
  (f 3 0 ∧ f 0 4 ∧ (∃ (m b : ℝ), ∀ (x y : ℝ), f x y ↔ y = m*x + b)) → 
  (∀ (x y : ℝ), f x y ↔ line_equation x y) := by sorry

end NUMINAMATH_CALUDE_point_3_0_on_line_point_0_4_on_line_is_line_equation_main_theorem_l2413_241372


namespace NUMINAMATH_CALUDE_equation_arrangements_l2413_241309

def word : String := "equation"

def letter_count : Nat := word.length

theorem equation_arrangements :
  let distinct_letters : Nat := 8
  let qu_as_unit : Nat := 1
  let remaining_letters : Nat := distinct_letters - 2
  let units_to_arrange : Nat := qu_as_unit + remaining_letters
  let letters_to_select : Nat := 5 - 2
  let ways_to_select : Nat := Nat.choose remaining_letters letters_to_select
  let ways_to_arrange : Nat := Nat.factorial (letters_to_select + 1)
  ways_to_select * ways_to_arrange = 480 := by
  sorry

end NUMINAMATH_CALUDE_equation_arrangements_l2413_241309


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l2413_241329

theorem subtraction_of_fractions : (1 : ℚ) / 2 - (1 : ℚ) / 8 = (3 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l2413_241329


namespace NUMINAMATH_CALUDE_log_four_one_sixtyfourth_l2413_241317

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_four_one_sixtyfourth : log 4 (1/64) = -3 := by
  sorry

end NUMINAMATH_CALUDE_log_four_one_sixtyfourth_l2413_241317


namespace NUMINAMATH_CALUDE_no_lines_satisfy_conditions_l2413_241387

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def line_through_point (a b : ℝ) : Prop :=
  6 / a + 5 / b = 1

def satisfies_conditions (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ is_prime a ∧ a + b < 20 ∧ line_through_point a b

theorem no_lines_satisfy_conditions :
  ¬ ∃ a b : ℕ, satisfies_conditions a b :=
sorry

end NUMINAMATH_CALUDE_no_lines_satisfy_conditions_l2413_241387


namespace NUMINAMATH_CALUDE_a_zero_sufficient_for_P_range_of_a_when_only_one_true_l2413_241349

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Theorem 1: a = 0 is a sufficient condition for P
theorem a_zero_sufficient_for_P : ∀ a : ℝ, a = 0 → P a := by sorry

-- Theorem 2: The range of a when only one of P and Q is true
theorem range_of_a_when_only_one_true : 
  {a : ℝ | (P a ∧ ¬Q a) ∨ (¬P a ∧ Q a)} = 
  {a : ℝ | a < 0 ∨ (1/4 < a ∧ a < 4)} := by sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_for_P_range_of_a_when_only_one_true_l2413_241349


namespace NUMINAMATH_CALUDE_trig_identity_l2413_241374

theorem trig_identity (α : ℝ) : 
  (Real.sin α + Real.sin (3 * α) - Real.sin (5 * α)) / 
  (Real.cos α - Real.cos (3 * α) - Real.cos (5 * α)) = Real.tan α :=
sorry

end NUMINAMATH_CALUDE_trig_identity_l2413_241374


namespace NUMINAMATH_CALUDE_floor_sum_inequality_floor_fractional_part_max_n_value_l2413_241315

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Proposition B
theorem floor_sum_inequality (x y : ℝ) : floor x + floor y ≤ floor (x + y) := by sorry

-- Proposition C
theorem floor_fractional_part (x : ℝ) : 0 ≤ x - floor x ∧ x - floor x < 1 := by sorry

-- Proposition D
def satisfies_conditions (t : ℝ) (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range (n - 2), floor (t ^ (k + 3)) = k + 1

theorem max_n_value :
  (∃ n : ℕ, n > 2 ∧ ∃ t : ℝ, satisfies_conditions t n) →
  (∀ n : ℕ, n > 5 → ¬∃ t : ℝ, satisfies_conditions t n) := by sorry

end NUMINAMATH_CALUDE_floor_sum_inequality_floor_fractional_part_max_n_value_l2413_241315


namespace NUMINAMATH_CALUDE_ellipse_sum_property_l2413_241390

/-- Represents an ellipse with its properties -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length
  θ : ℝ  -- rotation angle in radians

/-- Theorem: For a specific ellipse, the sum of its center coordinates and axis lengths is 11 -/
theorem ellipse_sum_property : 
  ∀ (e : Ellipse), 
  e.h = -2 ∧ e.k = 3 ∧ e.a = 6 ∧ e.b = 4 ∧ e.θ = π/4 → 
  e.h + e.k + e.a + e.b = 11 := by
sorry

end NUMINAMATH_CALUDE_ellipse_sum_property_l2413_241390


namespace NUMINAMATH_CALUDE_min_half_tiles_for_29_l2413_241394

/-- Represents a tiling of a square area -/
structure Tiling where
  size : ℕ  -- The size of the square area in unit squares
  whole_tiles : ℕ  -- Number of whole tiles used
  half_tiles : ℕ  -- Number of tiles cut in half

/-- Checks if a tiling is valid for the given area -/
def is_valid_tiling (t : Tiling) : Prop :=
  t.whole_tiles + t.half_tiles / 2 = t.size

/-- Theorem: The minimum number of tiles to be cut in half for a 29-unit square area is 12 -/
theorem min_half_tiles_for_29 :
  ∀ t : Tiling, t.size = 29 → is_valid_tiling t →
  t.half_tiles ≥ 12 ∧ ∃ t' : Tiling, t'.size = 29 ∧ is_valid_tiling t' ∧ t'.half_tiles = 12 :=
by sorry

#check min_half_tiles_for_29

end NUMINAMATH_CALUDE_min_half_tiles_for_29_l2413_241394


namespace NUMINAMATH_CALUDE_parallelogram_base_l2413_241305

theorem parallelogram_base (area height : ℝ) (h1 : area = 231) (h2 : height = 11) :
  area / height = 21 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2413_241305


namespace NUMINAMATH_CALUDE_driving_equation_correct_l2413_241323

/-- Represents a driving trip with a stop -/
structure DrivingTrip where
  speed_before_stop : ℝ
  speed_after_stop : ℝ
  stop_duration : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The equation for calculating the total distance is correct -/
theorem driving_equation_correct (trip : DrivingTrip) 
  (h1 : trip.speed_before_stop = 60)
  (h2 : trip.speed_after_stop = 90)
  (h3 : trip.stop_duration = 1/2)
  (h4 : trip.total_distance = 270)
  (h5 : trip.total_time = 4) :
  ∃ t : ℝ, 60 * t + 90 * (7/2 - t) = 270 ∧ 
           0 ≤ t ∧ t ≤ trip.total_time - trip.stop_duration :=
by sorry

end NUMINAMATH_CALUDE_driving_equation_correct_l2413_241323


namespace NUMINAMATH_CALUDE_statue_weight_l2413_241351

-- Define the initial weight and cutting percentages
def initial_weight : ℝ := 250
def first_cut : ℝ := 0.30
def second_cut : ℝ := 0.20
def third_cut : ℝ := 0.25

-- Define the final weight calculation
def final_weight : ℝ :=
  initial_weight * (1 - first_cut) * (1 - second_cut) * (1 - third_cut)

-- Theorem statement
theorem statue_weight :
  final_weight = 105 := by sorry

end NUMINAMATH_CALUDE_statue_weight_l2413_241351


namespace NUMINAMATH_CALUDE_regular_rate_is_three_dollars_l2413_241331

/-- Represents a worker's pay structure and hours worked -/
structure PayStructure where
  regularRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay based on the pay structure -/
def calculateTotalPay (p : PayStructure) : ℝ :=
  p.regularRate * p.regularHours + 2 * p.regularRate * p.overtimeHours

/-- Theorem: Given the specified pay structure, the regular rate is $3 per hour -/
theorem regular_rate_is_three_dollars (p : PayStructure) 
    (h1 : p.regularHours = 40)
    (h2 : p.overtimeHours = 10)
    (h3 : p.totalPay = 180)
    (h4 : calculateTotalPay p = p.totalPay) : 
    p.regularRate = 3 := by
  sorry

#check regular_rate_is_three_dollars

end NUMINAMATH_CALUDE_regular_rate_is_three_dollars_l2413_241331


namespace NUMINAMATH_CALUDE_remainder_problem_l2413_241319

theorem remainder_problem : (7 * 10^24 + 2^24) % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2413_241319


namespace NUMINAMATH_CALUDE_unique_sequence_l2413_241343

def sequence_condition (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) ∧
  (∀ n : ℕ, n > 0 → a (2 * n) = a n + n) ∧
  (∀ n : ℕ, n > 0 → Prime (a n) → Prime n)

theorem unique_sequence :
  ∀ a : ℕ → ℕ, sequence_condition a → ∀ n : ℕ, n > 0 → a n = n :=
sorry

end NUMINAMATH_CALUDE_unique_sequence_l2413_241343


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2413_241332

theorem sqrt_equation_solution :
  ∃ x : ℝ, x > 0 ∧ Real.sqrt 289 - Real.sqrt 625 / Real.sqrt x = 12 ∧ x = 25 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2413_241332
