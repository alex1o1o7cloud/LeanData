import Mathlib

namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2122_212277

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- The result of drawing two balls from the bag -/
inductive Draw
  | TwoRed
  | OneRedOneBlack
  | TwoBlack

/-- Define the events -/
def exactlyOneBlack (d : Draw) : Prop :=
  d = Draw.OneRedOneBlack

def exactlyTwoBlack (d : Draw) : Prop :=
  d = Draw.TwoBlack

/-- The probability of a draw given a bag -/
def prob (b : Bag) (d : Draw) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem mutually_exclusive_not_contradictory (b : Bag) 
  (h1 : b.red = 2) (h2 : b.black = 2) : 
  (∀ d, ¬(exactlyOneBlack d ∧ exactlyTwoBlack d)) ∧ 
  (∃ d, exactlyOneBlack d ∨ exactlyTwoBlack d) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2122_212277


namespace NUMINAMATH_CALUDE_probability_three_out_of_seven_greater_than_six_l2122_212252

/-- The probability of a single 12-sided die showing a number greater than 6 -/
def p_greater_than_6 : ℚ := 1 / 2

/-- The number of dice rolled -/
def n : ℕ := 7

/-- The number of dice we want to show a number greater than 6 -/
def k : ℕ := 3

/-- The probability of exactly k out of n dice showing a number greater than 6 -/
def probability_k_out_of_n (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_three_out_of_seven_greater_than_six :
  probability_k_out_of_n n k p_greater_than_6 = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_out_of_seven_greater_than_six_l2122_212252


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l2122_212220

theorem fourth_number_in_sequence (seq : Fin 6 → ℕ) 
  (h1 : seq 0 = 29)
  (h2 : seq 1 = 35)
  (h3 : seq 2 = 41)
  (h5 : seq 4 = 53)
  (h6 : seq 5 = 59)
  (h_arithmetic : ∀ i : Fin 4, seq (i + 1) - seq i = seq 1 - seq 0) :
  seq 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l2122_212220


namespace NUMINAMATH_CALUDE_average_spring_headcount_equals_10700_l2122_212238

def spring_headcount_02_03 : ℕ := 10900
def spring_headcount_03_04 : ℕ := 10500
def spring_headcount_04_05 : ℕ := 10700

def average_spring_headcount : ℚ :=
  (spring_headcount_02_03 + spring_headcount_03_04 + spring_headcount_04_05) / 3

theorem average_spring_headcount_equals_10700 :
  round average_spring_headcount = 10700 := by
  sorry

end NUMINAMATH_CALUDE_average_spring_headcount_equals_10700_l2122_212238


namespace NUMINAMATH_CALUDE_plants_eaten_first_day_l2122_212291

theorem plants_eaten_first_day (total : ℕ) (remaining : ℕ) :
  total = 30 ∧ 
  remaining = 4 ∧
  (∃ x y : ℕ, x + y + remaining + 1 = total ∧ y = (x + y + 1) / 2) →
  x = 20 :=
by sorry

end NUMINAMATH_CALUDE_plants_eaten_first_day_l2122_212291


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2122_212299

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2122_212299


namespace NUMINAMATH_CALUDE_water_level_accurate_l2122_212268

/-- Represents the water level function for a reservoir -/
def waterLevel (x : ℝ) : ℝ := 6 + 0.3 * x

/-- Theorem stating that the water level function accurately describes the reservoir's water level -/
theorem water_level_accurate (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  waterLevel x = 6 + 0.3 * x ∧ 
  waterLevel 0 = 6 ∧
  ∀ t₁ t₂, 0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ 5 → (waterLevel t₂ - waterLevel t₁) / (t₂ - t₁) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_water_level_accurate_l2122_212268


namespace NUMINAMATH_CALUDE_function_minimum_l2122_212261

/-- The function f(x) = x³ - 3x² + 4 attains its minimum value at x = 2 -/
theorem function_minimum (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 3*x^2 + 4) :
  ∃ x₀ : ℝ, x₀ = 2 ∧ ∀ x, f x₀ ≤ f x := by sorry

end NUMINAMATH_CALUDE_function_minimum_l2122_212261


namespace NUMINAMATH_CALUDE_max_profit_at_seventh_grade_l2122_212213

/-- Represents the profit function for a product with different quality grades. -/
def profit_function (x : ℕ) : ℝ :=
  let profit_per_unit := 6 + 2 * (x - 1)
  let units_produced := 60 - 4 * (x - 1)
  profit_per_unit * units_produced

/-- Represents the maximum grade available. -/
def max_grade : ℕ := 10

/-- Theorem stating that the 7th grade maximizes profit and the maximum profit is 648 yuan. -/
theorem max_profit_at_seventh_grade :
  (∃ (x : ℕ), x ≤ max_grade ∧ ∀ (y : ℕ), y ≤ max_grade → profit_function x ≥ profit_function y) ∧
  (∃ (x : ℕ), x ≤ max_grade ∧ profit_function x = 648) ∧
  profit_function 7 = 648 := by
  sorry

#eval profit_function 7  -- Should output 648

end NUMINAMATH_CALUDE_max_profit_at_seventh_grade_l2122_212213


namespace NUMINAMATH_CALUDE_max_comic_books_l2122_212235

/-- The cost function for buying comic books -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 1.2 * n else 12 + 1.1 * (n - 10)

/-- Jason's budget -/
def budget : ℚ := 15

/-- Predicate to check if a number of books is affordable -/
def is_affordable (n : ℕ) : Prop :=
  cost n ≤ budget

/-- The maximum number of comic books Jason can buy -/
def max_books : ℕ := 12

theorem max_comic_books : 
  (∀ n : ℕ, is_affordable n → n ≤ max_books) ∧ 
  is_affordable max_books :=
sorry

end NUMINAMATH_CALUDE_max_comic_books_l2122_212235


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2122_212279

/-- A polynomial of degree 4 with coefficients a, b, and c. -/
def polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  x^4 + a*x^2 + b*x + c

/-- The condition for divisibility by (x-1)^3. -/
def isDivisibleByXMinusOneCubed (a b c : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, polynomial a b c x = (x - 1)^3 * q x

/-- Theorem stating the necessary and sufficient conditions for divisibility. -/
theorem polynomial_divisibility (a b c : ℝ) :
  isDivisibleByXMinusOneCubed a b c ↔ a = 0 ∧ b = 2 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2122_212279


namespace NUMINAMATH_CALUDE_estimate_black_balls_l2122_212210

theorem estimate_black_balls (total_balls : Nat) (total_draws : Nat) (black_draws : Nat) :
  total_balls = 15 →
  total_draws = 100 →
  black_draws = 60 →
  (black_draws : Real) / total_draws * total_balls = 9 := by
  sorry

end NUMINAMATH_CALUDE_estimate_black_balls_l2122_212210


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2122_212222

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (∃ y : ℤ, y = x + 2 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ y = 3 * x) → 
  x + (x + 2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2122_212222


namespace NUMINAMATH_CALUDE_coat_drive_l2122_212284

theorem coat_drive (total_coats : ℕ) (elementary_coats : ℕ) (high_school_coats : ℕ) :
  total_coats = 9437 →
  elementary_coats = 2515 →
  high_school_coats = total_coats - elementary_coats →
  high_school_coats = 6922 :=
by
  sorry

end NUMINAMATH_CALUDE_coat_drive_l2122_212284


namespace NUMINAMATH_CALUDE_min_value_3x_minus_2y_l2122_212234

theorem min_value_3x_minus_2y (x y : ℝ) (h : 4 * (x^2 + y^2 + x*y) = 2 * (x + y)) :
  ∃ (m : ℝ), m = -1 ∧ ∀ (a b : ℝ), 4 * (a^2 + b^2 + a*b) = 2 * (a + b) → 3*a - 2*b ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_3x_minus_2y_l2122_212234


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2122_212215

/-- Given an integer n, returns the tens digit -/
def tensDigit (n : ℤ) : ℤ := (n / 10) % 10

/-- Given an integer n, returns the units digit -/
def unitsDigit (n : ℤ) : ℤ := n % 10

/-- Theorem: For any integer divisible by 5 with the sum of its last two digits being 12,
    the product of its last two digits is 35 -/
theorem last_two_digits_product (n : ℤ) : 
  n % 5 = 0 → 
  tensDigit n + unitsDigit n = 12 → 
  tensDigit n * unitsDigit n = 35 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2122_212215


namespace NUMINAMATH_CALUDE_sandwich_cost_l2122_212263

-- Define the given values
def num_sandwiches : ℕ := 3
def num_energy_bars : ℕ := 3
def num_drinks : ℕ := 2
def drink_cost : ℚ := 4
def energy_bar_cost : ℚ := 3
def energy_bar_discount : ℚ := 0.2
def total_spent : ℚ := 40.80

-- Define the theorem
theorem sandwich_cost :
  let drink_total : ℚ := num_drinks * drink_cost
  let energy_bar_total : ℚ := num_energy_bars * energy_bar_cost * (1 - energy_bar_discount)
  let sandwich_total : ℚ := total_spent - drink_total - energy_bar_total
  sandwich_total / num_sandwiches = 8.53 := by sorry

end NUMINAMATH_CALUDE_sandwich_cost_l2122_212263


namespace NUMINAMATH_CALUDE_composite_odd_number_characterization_l2122_212226

theorem composite_odd_number_characterization (c : ℕ) (h_odd : Odd c) :
  (∃ (a : ℕ), a ≤ c / 3 - 1 ∧ ∃ (k : ℕ), (2 * a - 1)^2 + 8 * c = (2 * k + 1)^2) ↔
  (∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ c = p * q) :=
sorry

end NUMINAMATH_CALUDE_composite_odd_number_characterization_l2122_212226


namespace NUMINAMATH_CALUDE_square_numbers_between_24_and_150_divisible_by_6_l2122_212201

def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem square_numbers_between_24_and_150_divisible_by_6 :
  {x : ℕ | 24 < x ∧ x < 150 ∧ is_square x ∧ x % 6 = 0} = {36, 144} := by
  sorry

end NUMINAMATH_CALUDE_square_numbers_between_24_and_150_divisible_by_6_l2122_212201


namespace NUMINAMATH_CALUDE_product_and_squared_sum_l2122_212206

theorem product_and_squared_sum (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 10) : 
  x * y = 875 ∧ (x + y)^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_product_and_squared_sum_l2122_212206


namespace NUMINAMATH_CALUDE_smallest_n_l2122_212246

/-- The smallest three-digit positive integer n such that n + 7 is divisible by 9 and n - 10 is divisible by 6 -/
theorem smallest_n : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 7)) ∧ 
  (6 ∣ (n - 10)) ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 7)) ∧ (6 ∣ (m - 10))) → False) ∧
  n = 118 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_l2122_212246


namespace NUMINAMATH_CALUDE_howard_last_week_money_l2122_212219

/-- Howard's money situation --/
def howard_money (current_money washing_money last_week_money : ℕ) : Prop :=
  current_money = washing_money + last_week_money

/-- Theorem: Howard had 26 dollars last week --/
theorem howard_last_week_money :
  ∃ (last_week_money : ℕ),
    howard_money 52 26 last_week_money ∧ last_week_money = 26 :=
sorry

end NUMINAMATH_CALUDE_howard_last_week_money_l2122_212219


namespace NUMINAMATH_CALUDE_max_D_value_l2122_212212

/-- Represents a building block with three binary attributes -/
structure Block :=
  (shape : Bool)
  (color : Bool)
  (city : Bool)

/-- The set of all possible blocks -/
def allBlocks : Finset Block := sorry

/-- The number of blocks -/
def numBlocks : Nat := Finset.card allBlocks

/-- Checks if two blocks share exactly two attributes -/
def sharesTwoAttributes (b1 b2 : Block) : Bool := sorry

/-- The number of ways to select n blocks such that each subsequent block
    shares exactly two attributes with the previously selected block -/
def D (n : Nat) : Nat := sorry

/-- The maximum value of D(n) for 2 ≤ n ≤ 8 -/
def maxD : Nat := sorry

theorem max_D_value :
  numBlocks = 8 →
  (∀ (b1 b2 : Block), b1 ∈ allBlocks ∧ b2 ∈ allBlocks → b1 ≠ b2) →
  maxD = 240 := by sorry

end NUMINAMATH_CALUDE_max_D_value_l2122_212212


namespace NUMINAMATH_CALUDE_cafeteria_pie_count_l2122_212251

/-- Given a cafeteria with apples and pie-making scenario, calculate the number of pies that can be made -/
theorem cafeteria_pie_count (total_apples handed_out apples_per_pie : ℕ) 
  (h1 : total_apples = 96)
  (h2 : handed_out = 42)
  (h3 : apples_per_pie = 6) :
  (total_apples - handed_out) / apples_per_pie = 9 :=
by
  sorry

#check cafeteria_pie_count

end NUMINAMATH_CALUDE_cafeteria_pie_count_l2122_212251


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l2122_212232

theorem complex_imaginary_part (z : ℂ) (h : z * (3 - 4*I) = 1) : z.im = 4/25 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l2122_212232


namespace NUMINAMATH_CALUDE_math_representative_selection_l2122_212211

theorem math_representative_selection (male_students female_students : ℕ) 
  (h1 : male_students = 26) 
  (h2 : female_students = 24) : 
  (male_students + female_students : ℕ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_math_representative_selection_l2122_212211


namespace NUMINAMATH_CALUDE_larger_number_proof_l2122_212207

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2122_212207


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2122_212280

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * y^2 - x^2 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 5 + x^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola m x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
   -- The foci of the hyperbola and ellipse are the same
   (x₁ = x₂ ∧ y₁ = y₂)) →
  (∀ x y : ℝ, hyperbola m x y → asymptotes x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2122_212280


namespace NUMINAMATH_CALUDE_negative_slope_probability_l2122_212254

def LineSet : Set ℤ := {-3, -1, 0, 2, 7}

def ValidPair (a b : ℤ) : Prop :=
  a ∈ LineSet ∧ b ∈ LineSet ∧ a ≠ b

def NegativeSlope (a b : ℤ) : Prop :=
  ValidPair a b ∧ (a / b < 0)

def TotalPairs : ℕ := 20

def NegativeSlopePairs : ℕ := 4

theorem negative_slope_probability :
  (NegativeSlopePairs : ℚ) / TotalPairs = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_negative_slope_probability_l2122_212254


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l2122_212228

/-- Represents a cubic polynomial of the form x³ + px² + qx + r -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The sum of zeros of a cubic polynomial -/
def sumOfZeros (poly : CubicPolynomial) : ℝ := -poly.p

/-- The product of zeros of a cubic polynomial -/
def productOfZeros (poly : CubicPolynomial) : ℝ := -poly.r

/-- The sum of coefficients of a cubic polynomial -/
def sumOfCoefficients (poly : CubicPolynomial) : ℝ := 1 + poly.p + poly.q + poly.r

/-- The y-intercept of a cubic polynomial -/
def yIntercept (poly : CubicPolynomial) : ℝ := poly.r

theorem cubic_polynomial_property (poly : CubicPolynomial) :
  sumOfZeros poly = 2 * productOfZeros poly ∧
  sumOfZeros poly = sumOfCoefficients poly ∧
  yIntercept poly = 5 →
  poly.q = -24 := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l2122_212228


namespace NUMINAMATH_CALUDE_rice_dumpling_suitable_for_sampling_only_rice_dumpling_suitable_for_sampling_l2122_212274

/-- Represents an event that could potentially be surveyed. -/
inductive Event
  | AirplaneSecurity
  | SpacecraftInspection
  | TeacherRecruitment
  | RiceDumplingQuality

/-- Characteristics that make an event suitable for sampling survey. -/
structure SamplingSurveyCharacteristics where
  large_population : Bool
  impractical_full_inspection : Bool
  representative_sample_possible : Bool

/-- Defines the characteristics of a sampling survey for each event. -/
def event_characteristics : Event → SamplingSurveyCharacteristics
  | Event.AirplaneSecurity => ⟨false, false, false⟩
  | Event.SpacecraftInspection => ⟨false, false, false⟩
  | Event.TeacherRecruitment => ⟨false, false, false⟩
  | Event.RiceDumplingQuality => ⟨true, true, true⟩

/-- Determines if an event is suitable for a sampling survey based on its characteristics. -/
def is_suitable_for_sampling (e : Event) : Prop :=
  let c := event_characteristics e
  c.large_population ∧ c.impractical_full_inspection ∧ c.representative_sample_possible

/-- Theorem stating that the rice dumpling quality investigation is suitable for a sampling survey. -/
theorem rice_dumpling_suitable_for_sampling :
  is_suitable_for_sampling Event.RiceDumplingQuality :=
by
  sorry

/-- Theorem stating that the rice dumpling quality investigation is the only event suitable for a sampling survey. -/
theorem only_rice_dumpling_suitable_for_sampling :
  ∀ e : Event, is_suitable_for_sampling e ↔ e = Event.RiceDumplingQuality :=
by
  sorry

end NUMINAMATH_CALUDE_rice_dumpling_suitable_for_sampling_only_rice_dumpling_suitable_for_sampling_l2122_212274


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l2122_212240

theorem loan_principal_calculation (interest_rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  interest_rate = 12 →
  time = 3 →
  interest = 6480 →
  interest = principal * interest_rate * time / 100 →
  principal = 18000 := by
sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l2122_212240


namespace NUMINAMATH_CALUDE_sum_of_squares_extremes_l2122_212265

theorem sum_of_squares_extremes (a b c : ℝ) : 
  a / b = 2 / 3 ∧ b / c = 3 / 4 ∧ b = 9 → a^2 + c^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_extremes_l2122_212265


namespace NUMINAMATH_CALUDE_sqrt_five_power_calculation_l2122_212244

theorem sqrt_five_power_calculation : (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 78125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_power_calculation_l2122_212244


namespace NUMINAMATH_CALUDE_a_4_equals_7_l2122_212283

-- Define the sequence sum function
def S (n : ℕ) : ℤ := n^2 - 1

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Theorem statement
theorem a_4_equals_7 : a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_7_l2122_212283


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2122_212276

theorem fraction_multiplication : (3 : ℚ) / 4 * 5 / 7 * 11 / 13 = 165 / 364 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2122_212276


namespace NUMINAMATH_CALUDE_notebooks_distributed_sang_woo_distribution_l2122_212294

theorem notebooks_distributed (initial_notebooks : ℕ) (initial_pencils : ℕ) 
  (remaining_total : ℕ) : ℕ :=
  let distributed_notebooks := 
    (initial_notebooks + initial_pencils - remaining_total) / 4
  distributed_notebooks

theorem sang_woo_distribution : notebooks_distributed 12 34 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_distributed_sang_woo_distribution_l2122_212294


namespace NUMINAMATH_CALUDE_triangle_equilateral_l2122_212258

/-- A triangle with sides a, b, c corresponding to angles A, B, C is equilateral if
    a * cos(C) = c * cos(A) and a, b, c are in geometric progression. -/
theorem triangle_equilateral (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  a * Real.cos C = c * Real.cos A →
  ∃ r : ℝ, r > 0 ∧ a = b / r ∧ b = c / r →
  a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l2122_212258


namespace NUMINAMATH_CALUDE_cube_function_property_l2122_212253

theorem cube_function_property (a : ℝ) : 
  (fun x : ℝ ↦ x^3 + 1) a = 11 → (fun x : ℝ ↦ x^3 + 1) (-a) = -9 := by
sorry

end NUMINAMATH_CALUDE_cube_function_property_l2122_212253


namespace NUMINAMATH_CALUDE_triangle_interior_point_inequality_l2122_212292

open Real

variable (A B C M : ℝ × ℝ)

def isInside (M A B C : ℝ × ℝ) : Prop := sorry

def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem triangle_interior_point_inequality 
  (h : isInside M A B C) :
  min (distance M A) (min (distance M B) (distance M C)) + 
  distance M A + distance M B + distance M C < 
  distance A B + distance B C + distance C A := by
  sorry

end NUMINAMATH_CALUDE_triangle_interior_point_inequality_l2122_212292


namespace NUMINAMATH_CALUDE_inequality_solution_l2122_212242

theorem inequality_solution (x : ℝ) : 
  (x^2 - 6*x + 8) / (x^2 - 9) > 0 ↔ x < -3 ∨ (2 < x ∧ x < 3) ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2122_212242


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l2122_212262

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

theorem sqrt_three_times_sqrt_two_equals_sqrt_six : 
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l2122_212262


namespace NUMINAMATH_CALUDE_expression_value_l2122_212259

theorem expression_value : 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2122_212259


namespace NUMINAMATH_CALUDE_rationalize_denominator_35_sqrt_35_l2122_212272

theorem rationalize_denominator_35_sqrt_35 :
  (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_35_sqrt_35_l2122_212272


namespace NUMINAMATH_CALUDE_triangle_perimeter_upper_bound_l2122_212237

theorem triangle_perimeter_upper_bound (a b c : ℝ) : 
  a = 8 → b = 15 → a + b > c → a + c > b → b + c > a → 
  ∃ n : ℕ, n = 46 ∧ ∀ m : ℕ, (m : ℝ) > a + b + c → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_upper_bound_l2122_212237


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2122_212255

/-- Given a line and a parabola in a Cartesian coordinate system, 
    this theorem states the conditions for the parabola to intersect 
    the line segment between two points on the line at two distinct points. -/
theorem parabola_line_intersection 
  (a : ℝ) 
  (h_a_neq_zero : a ≠ 0) 
  (h_line : ∀ x y : ℝ, y = (1/2) * x + 1/2 ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) 
  (h_parabola : ∀ x y : ℝ, y = a * x^2 - x + 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
   -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧
   ((1/2) * x1 + 1/2 = a * x1^2 - x1 + 1) ∧
   ((1/2) * x2 + 1/2 = a * x2^2 - x2 + 1)) ↔
  (1 ≤ a ∧ a < 9/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2122_212255


namespace NUMINAMATH_CALUDE_moe_eating_time_l2122_212200

/-- Given that a lizard named Moe eats 40 pieces of cuttlebone in 10 seconds, 
    this theorem proves that it takes 200 seconds for Moe to eat 800 pieces. -/
theorem moe_eating_time : ∀ (rate : ℝ) (pieces : ℕ) (time : ℝ),
  rate = 40 / 10 →
  pieces = 800 →
  time = pieces / rate →
  time = 200 := by sorry

end NUMINAMATH_CALUDE_moe_eating_time_l2122_212200


namespace NUMINAMATH_CALUDE_one_positive_number_l2122_212285

theorem one_positive_number (numbers : List ℝ := [3, -2.1, -1/2, 0, -9]) :
  (numbers.filter (λ x => x > 0)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_positive_number_l2122_212285


namespace NUMINAMATH_CALUDE_quadratic_equation_in_y_l2122_212217

theorem quadratic_equation_in_y : 
  ∀ x y : ℝ, 
  (3 * x^2 - 4 * x + 7 * y + 3 = 0) → 
  (3 * x - 5 * y + 6 = 0) → 
  (25 * y^2 - 39 * y + 69 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_in_y_l2122_212217


namespace NUMINAMATH_CALUDE_inequality_proof_l2122_212229

theorem inequality_proof (a b : ℝ) (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2122_212229


namespace NUMINAMATH_CALUDE_partnership_profit_share_l2122_212248

/-- 
Given three partners A, B, and C in a partnership where:
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 4400

This theorem proves that B's share of the profit is 1760.
-/
theorem partnership_profit_share 
  (investment_A investment_B investment_C : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_A = 3 * investment_B)
  (h2 : investment_B = 2/3 * investment_C)
  (h3 : total_profit = 4400) :
  (investment_B / (investment_A + investment_B + investment_C)) * total_profit = 1760 := by
  sorry


end NUMINAMATH_CALUDE_partnership_profit_share_l2122_212248


namespace NUMINAMATH_CALUDE_sam_poured_buckets_l2122_212203

/-- The number of buckets Sam initially poured into the pool -/
def initial_buckets : ℝ := 1

/-- The number of buckets Sam added later -/
def additional_buckets : ℝ := 8.8

/-- The total number of buckets Sam poured into the pool -/
def total_buckets : ℝ := initial_buckets + additional_buckets

theorem sam_poured_buckets : total_buckets = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_sam_poured_buckets_l2122_212203


namespace NUMINAMATH_CALUDE_train_car_count_l2122_212243

theorem train_car_count (total_cars : ℕ) (passenger_cars : ℕ) (cargo_cars : ℕ) : 
  total_cars = 71 →
  cargo_cars = passenger_cars / 2 + 3 →
  total_cars = passenger_cars + cargo_cars + 2 →
  passenger_cars = 44 := by
sorry

end NUMINAMATH_CALUDE_train_car_count_l2122_212243


namespace NUMINAMATH_CALUDE_sequence_perfect_square_property_l2122_212264

/-- Given two sequences of natural numbers satisfying a specific equation,
    prove that yₙ - 1 is a perfect square for all n. -/
theorem sequence_perfect_square_property
  (x y : ℕ → ℕ)
  (h : ∀ n : ℕ, (x n : ℝ) + Real.sqrt 2 * (y n : ℝ) = Real.sqrt 2 * (3 + 2 * Real.sqrt 2) ^ (2 ^ n)) :
  ∀ n : ℕ, ∃ k : ℕ, y n - 1 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_perfect_square_property_l2122_212264


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_zero_product_property_l2122_212270

-- Proposition 1
theorem quadratic_equation_real_roots (k : ℝ) (h : k > 0) :
  ∃ x : ℝ, x^2 + 2*x - k = 0 :=
sorry

-- Proposition 4
theorem zero_product_property (x y : ℝ) :
  x * y = 0 → x = 0 ∨ y = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_zero_product_property_l2122_212270


namespace NUMINAMATH_CALUDE_inequality_proof_l2122_212231

theorem inequality_proof (a b c : ℝ) : 
  ((a^2 + b^2 + a*c)^2 + (a^2 + b^2 + b*c)^2) / (a^2 + b^2) ≥ (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2122_212231


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2122_212214

/-- Given a sphere and a cone on a horizontal field with parallel sun rays,
    if the sphere's shadow extends 20 meters from its base,
    and a 3-meter-high cone casts a 5-meter-long shadow,
    then the radius of the sphere is 12 meters. -/
theorem sphere_radius_from_shadows (sphere_shadow : ℝ) (cone_height cone_shadow : ℝ)
  (h_sphere_shadow : sphere_shadow = 20)
  (h_cone_height : cone_height = 3)
  (h_cone_shadow : cone_shadow = 5) :
  sphere_shadow * (cone_height / cone_shadow) = 12 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2122_212214


namespace NUMINAMATH_CALUDE_office_age_problem_l2122_212290

theorem office_age_problem (total_persons : Nat) (total_avg : ℚ) (group1_persons : Nat) 
  (group1_avg : ℚ) (person15_age : Nat) (group2_persons : Nat) :
  total_persons = 20 →
  total_avg = 15 →
  group1_persons = 5 →
  group1_avg = 14 →
  person15_age = 86 →
  group2_persons = 9 →
  let total_age : ℚ := total_persons * total_avg
  let group1_age : ℚ := group1_persons * group1_avg
  let remaining_age : ℚ := total_age - group1_age - person15_age
  let group2_age : ℚ := remaining_age - (total_persons - group1_persons - group2_persons - 1) * total_avg
  let group2_avg : ℚ := group2_age / group2_persons
  group2_avg = 23/3 := by sorry

end NUMINAMATH_CALUDE_office_age_problem_l2122_212290


namespace NUMINAMATH_CALUDE_second_number_form_l2122_212286

theorem second_number_form (G S : ℕ) (h1 : G = 4) 
  (h2 : ∃ k : ℕ, 1642 = k * G + 6) 
  (h3 : ∃ l : ℕ, S = l * G + 4) : 
  ∃ m : ℕ, S = 4 * m + 4 := by
sorry

end NUMINAMATH_CALUDE_second_number_form_l2122_212286


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l2122_212257

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l2122_212257


namespace NUMINAMATH_CALUDE_exists_abc_factorial_sum_l2122_212296

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem exists_abc_factorial_sum :
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    100 * a + 10 * b + c = factorial a + factorial b + factorial c :=
by sorry

end NUMINAMATH_CALUDE_exists_abc_factorial_sum_l2122_212296


namespace NUMINAMATH_CALUDE_omega_sum_equals_one_l2122_212202

theorem omega_sum_equals_one (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := by
  sorry

end NUMINAMATH_CALUDE_omega_sum_equals_one_l2122_212202


namespace NUMINAMATH_CALUDE_no_three_distinct_reals_l2122_212241

theorem no_three_distinct_reals : ¬∃ (a b c p : ℝ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b * c = p ∧
  b + c * a = p ∧
  c + a * b = p := by
  sorry

end NUMINAMATH_CALUDE_no_three_distinct_reals_l2122_212241


namespace NUMINAMATH_CALUDE_complex_magnitude_l2122_212221

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = -4 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2122_212221


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2122_212247

theorem algebraic_expression_value (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_condition : a + b + c + d = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / ((a-b)*(a-c)*(a-d))) + (b^5 / ((b-a)*(b-c)*(b-d))) + 
  (c^5 / ((c-a)*(c-b)*(c-d))) + (d^5 / ((d-a)*(d-b)*(d-c))) = -9 :=
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2122_212247


namespace NUMINAMATH_CALUDE_muscle_gain_percentage_l2122_212233

/-- Proves that the percentage of body weight gained in muscle is 20% -/
theorem muscle_gain_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 120)
  (h2 : final_weight = 150)
  (h3 : ∀ (x : ℝ), x * initial_weight + (x / 4) * initial_weight = final_weight - initial_weight) :
  ∃ (muscle_gain_percent : ℝ), muscle_gain_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_muscle_gain_percentage_l2122_212233


namespace NUMINAMATH_CALUDE_georges_new_socks_l2122_212256

theorem georges_new_socks (initial_socks : ℝ) (dad_socks : ℝ) (total_socks : ℕ) 
  (h1 : initial_socks = 28)
  (h2 : dad_socks = 4)
  (h3 : total_socks = 68) :
  ↑total_socks - initial_socks - dad_socks = 36 :=
by sorry

end NUMINAMATH_CALUDE_georges_new_socks_l2122_212256


namespace NUMINAMATH_CALUDE_intersection_count_l2122_212260

/-- The set A as defined in the problem -/
def A : Set (ℤ × ℤ) := {p | ∃ m : ℤ, m > 0 ∧ p.1 = m ∧ p.2 = -3*m + 2}

/-- The set B as defined in the problem -/
def B (a : ℤ) : Set (ℤ × ℤ) := {p | ∃ n : ℤ, n > 0 ∧ p.1 = n ∧ p.2 = a*(a^2 - n + 1)}

/-- The theorem stating that there are exactly 10 integer values of a for which A ∩ B ≠ ∅ -/
theorem intersection_count :
  ∃! (s : Finset ℤ), s.card = 10 ∧ ∀ a : ℤ, a ∈ s ↔ (A ∩ B a).Nonempty :=
by sorry

end NUMINAMATH_CALUDE_intersection_count_l2122_212260


namespace NUMINAMATH_CALUDE_jill_watching_time_l2122_212205

/-- The length of the first show Jill watched, in minutes. -/
def first_show_length : ℝ := 30

/-- The length of the second show Jill watched, in minutes. -/
def second_show_length : ℝ := 4 * first_show_length

/-- The total time Jill spent watching shows, in minutes. -/
def total_watching_time : ℝ := 150

theorem jill_watching_time :
  first_show_length + second_show_length = total_watching_time :=
by sorry

end NUMINAMATH_CALUDE_jill_watching_time_l2122_212205


namespace NUMINAMATH_CALUDE_smallest_n_with_g_having_8_or_higher_l2122_212278

/-- Sum of digits in base b representation of n -/
def sumDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-five representation of n -/
def f (n : ℕ) : ℕ := sumDigits n 5

/-- g(n) is the sum of digits in base-nine representation of f(n) -/
def g (n : ℕ) : ℕ := sumDigits (f n) 9

/-- A number has a digit '8' or higher in base-nine if it's greater than or equal to 8 -/
def hasDigit8OrHigher (n : ℕ) : Prop := n ≥ 8

theorem smallest_n_with_g_having_8_or_higher :
  (∀ m : ℕ, m < 248 → ¬hasDigit8OrHigher (g m)) ∧ hasDigit8OrHigher (g 248) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_g_having_8_or_higher_l2122_212278


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l2122_212223

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (1 + b) = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1 / (x + 1) + 2 / (1 + y) = 1 → a + b ≤ x + y ∧ a + b = 2 * Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l2122_212223


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2122_212293

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.targetRuns - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.2)
  (h4 : game.targetRuns = 252) :
  requiredRunRate game = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2122_212293


namespace NUMINAMATH_CALUDE_probability_one_white_one_black_l2122_212267

/-- The probability of drawing one white ball and one black ball from a bag -/
theorem probability_one_white_one_black (white_balls black_balls : ℕ) :
  white_balls = 6 →
  black_balls = 5 →
  (white_balls.choose 1 * black_balls.choose 1 : ℚ) / (white_balls + black_balls).choose 2 = 6/11 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_white_one_black_l2122_212267


namespace NUMINAMATH_CALUDE_puzzle_solution_l2122_212275

-- Define the types of characters
inductive Character
| Human
| Ape

-- Define the types of statements
inductive StatementType
| Truthful
| Lie

-- Define a structure for a person
structure Person where
  species : Character
  statementType : StatementType

-- Define the statements made by A and B
def statement_A (b : Person) (a : Person) : Prop :=
  b.statementType = StatementType.Lie ∧ 
  b.species = Character.Ape ∧ 
  a.species = Character.Human

def statement_B (a : Person) : Prop :=
  a.statementType = StatementType.Truthful

-- Theorem stating the conclusion
theorem puzzle_solution :
  ∃ (a b : Person),
    (statement_A b a = (a.statementType = StatementType.Lie)) ∧
    (statement_B a = (b.statementType = StatementType.Lie)) ∧
    a.species = Character.Ape ∧
    a.statementType = StatementType.Lie ∧
    b.species = Character.Human ∧
    b.statementType = StatementType.Lie :=
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2122_212275


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_P_l2122_212236

def P : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (3^(i+1) + 1)) 1

theorem highest_power_of_two_dividing_P :
  ∃ (k : ℕ), (2^3030 ∣ P 2020) ∧ ¬(2^(3030 + 1) ∣ P 2020) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_P_l2122_212236


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equals_l2122_212282

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 1/w = 2 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∃ z w : ℝ, z > 0 ∧ w > 0 ∧ 1/z + 1/w = 2 ∧ z + 2*w = (3 + 2*Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equals_l2122_212282


namespace NUMINAMATH_CALUDE_pizzas_served_today_l2122_212230

theorem pizzas_served_today (lunch_pizzas dinner_pizzas : ℝ) 
  (h1 : lunch_pizzas = 12.5)
  (h2 : dinner_pizzas = 8.25) : 
  lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l2122_212230


namespace NUMINAMATH_CALUDE_pump_fill_time_pump_fill_time_proof_l2122_212266

theorem pump_fill_time (fill_time_with_leak : ℚ) (leak_drain_time : ℕ) : ℚ :=
  let fill_rate_with_leak : ℚ := 1 / fill_time_with_leak
  let leak_rate : ℚ := 1 / leak_drain_time
  let pump_rate : ℚ := fill_rate_with_leak + leak_rate
  1 / pump_rate

theorem pump_fill_time_proof :
  pump_fill_time (13/6) 26 = 2 := by sorry

end NUMINAMATH_CALUDE_pump_fill_time_pump_fill_time_proof_l2122_212266


namespace NUMINAMATH_CALUDE_good_number_characterization_twenty_nine_is_good_good_numbers_up_to_nine_correct_product_of_good_numbers_is_good_l2122_212227

def is_good_number (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + 2*x*y + 2*y^2

theorem good_number_characterization (n : ℤ) :
  is_good_number n ↔ ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

theorem twenty_nine_is_good : is_good_number 29 :=
sorry

def good_numbers_up_to_nine : List ℤ := [1, 2, 4, 5, 8, 9]

theorem good_numbers_up_to_nine_correct :
  ∀ n : ℤ, n ∈ good_numbers_up_to_nine ↔ (1 ≤ n ∧ n ≤ 9 ∧ is_good_number n) :=
sorry

theorem product_of_good_numbers_is_good (m n : ℤ) :
  is_good_number m → is_good_number n → is_good_number (m * n) :=
sorry

end NUMINAMATH_CALUDE_good_number_characterization_twenty_nine_is_good_good_numbers_up_to_nine_correct_product_of_good_numbers_is_good_l2122_212227


namespace NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l2122_212218

theorem a_greater_than_b_greater_than_one
  (n : ℕ) (hn : n ≥ 2)
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (eq1 : a^n = a + 1)
  (eq2 : b^(2*n) = b + 3*a) :
  a > b ∧ b > 1 := by
sorry

end NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l2122_212218


namespace NUMINAMATH_CALUDE_twentieth_sample_number_l2122_212271

/-- Calculates the nth number in a systematic sample. -/
def systematicSample (totalItems : Nat) (sampleSize : Nat) (firstNumber : Nat) (n : Nat) : Nat :=
  let k := totalItems / sampleSize
  firstNumber + (n - 1) * k

/-- Proves that the 20th number in the systematic sample is 395. -/
theorem twentieth_sample_number 
  (totalItems : Nat) 
  (sampleSize : Nat) 
  (firstNumber : Nat) 
  (h1 : totalItems = 1000) 
  (h2 : sampleSize = 50) 
  (h3 : firstNumber = 15) :
  systematicSample totalItems sampleSize firstNumber 20 = 395 := by
  sorry

#eval systematicSample 1000 50 15 20

end NUMINAMATH_CALUDE_twentieth_sample_number_l2122_212271


namespace NUMINAMATH_CALUDE_pond_filling_time_l2122_212250

/-- Proves the time required to fill a pond under drought conditions -/
theorem pond_filling_time (pond_capacity : ℝ) (normal_rate : ℝ) (drought_factor : ℝ) : 
  pond_capacity = 200 →
  normal_rate = 6 →
  drought_factor = 2/3 →
  (pond_capacity / (normal_rate * drought_factor) = 50) :=
by
  sorry

end NUMINAMATH_CALUDE_pond_filling_time_l2122_212250


namespace NUMINAMATH_CALUDE_dolls_count_l2122_212289

theorem dolls_count (total_toys : ℕ) (action_figure_fraction : ℚ) : 
  total_toys = 24 → action_figure_fraction = 1/4 → 
  total_toys - (total_toys * action_figure_fraction).floor = 18 := by
sorry

end NUMINAMATH_CALUDE_dolls_count_l2122_212289


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_15_l2122_212295

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem line_y_coordinate_at_15 (l : Line) 
    (h1 : l.point1 = (4, 5))
    (h2 : l.point2 = (8, 17))
    (h3 : l.point3 = (12, 29))
    (h4 : collinear l.point1 l.point2 l.point3) :
    ∃ t : ℝ, collinear l.point1 l.point2 (15, t) ∧ t = 38 := by
  sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_15_l2122_212295


namespace NUMINAMATH_CALUDE_stone_skipping_ratio_l2122_212209

theorem stone_skipping_ratio (x y : ℕ) : 
  x > 0 → -- First throw has at least one skip
  x + 2 > 0 → -- Second throw has at least one skip
  y > 0 → -- Third throw has at least one skip
  y - 3 > 0 → -- Fourth throw has at least one skip
  y - 2 = 8 → -- Fifth throw skips 8 times
  x + (x + 2) + y + (y - 3) + (y - 2) = 33 → -- Total skips is 33
  y = x + 2 -- Ratio of third to second throw is 1:1
  := by sorry

end NUMINAMATH_CALUDE_stone_skipping_ratio_l2122_212209


namespace NUMINAMATH_CALUDE_inverse_f_at_5_l2122_212204

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem inverse_f_at_5 :
  ∃ (f_inv : ℝ → ℝ), (∀ x ≥ 0, f_inv (f x) = x) ∧ f_inv 5 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_5_l2122_212204


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2122_212298

/-- Given a quadratic equation 5x^2 + kx = 8 with one root equal to 2, 
    prove that the other root is -4/5 -/
theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 5 * x^2 + k * x = 8) ∧ (2 : ℝ) ∈ {x : ℝ | 5 * x^2 + k * x = 8} →
  (-4/5 : ℝ) ∈ {x : ℝ | 5 * x^2 + k * x = 8} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2122_212298


namespace NUMINAMATH_CALUDE_suit_price_calculation_suit_price_theorem_l2122_212287

theorem suit_price_calculation (original_price : ℝ) 
  (increase_rate : ℝ) (reduction_rate : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - reduction_rate)
  final_price

theorem suit_price_theorem : 
  suit_price_calculation 300 0.2 0.1 = 324 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_suit_price_theorem_l2122_212287


namespace NUMINAMATH_CALUDE_evaluate_expression_l2122_212224

theorem evaluate_expression (a : ℝ) (h : a = 2) : (7 * a^2 - 15 * a + 5) * (3 * a - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2122_212224


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2122_212245

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) (h4 : w = 7) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 38 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2122_212245


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l2122_212288

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/-- The remainder when 10! is divided by 13 is 7 -/
theorem factorial_10_mod_13 : factorial 10 % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l2122_212288


namespace NUMINAMATH_CALUDE_other_number_from_hcf_lcm_and_one_number_l2122_212269

/-- Given two positive integers with known HCF, LCM, and one of the numbers,
    prove that the other number is as calculated. -/
theorem other_number_from_hcf_lcm_and_one_number
  (a b : ℕ+) 
  (hcf : Nat.gcd a b = 16)
  (lcm : Nat.lcm a b = 396)
  (ha : a = 36) :
  b = 176 := by
  sorry

end NUMINAMATH_CALUDE_other_number_from_hcf_lcm_and_one_number_l2122_212269


namespace NUMINAMATH_CALUDE_rectangular_plot_difference_l2122_212239

/-- Proves that for a rectangular plot with breadth 8 meters and area 18 times its breadth,
    the difference between the length and the breadth is 10 meters. -/
theorem rectangular_plot_difference (length breadth : ℝ) : 
  breadth = 8 →
  length * breadth = 18 * breadth →
  length - breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_difference_l2122_212239


namespace NUMINAMATH_CALUDE_liam_money_left_l2122_212281

/-- Calculates the amount of money Liam has left after paying his bills -/
def money_left_after_bills (
  savings_rate : ℕ
) (
  savings_duration_months : ℕ
) (
  bills_cost : ℕ
) : ℕ :=
  savings_rate * savings_duration_months - bills_cost

/-- Proves that Liam will have $8,500 left after paying his bills -/
theorem liam_money_left :
  money_left_after_bills 500 24 3500 = 8500 := by
  sorry

#eval money_left_after_bills 500 24 3500

end NUMINAMATH_CALUDE_liam_money_left_l2122_212281


namespace NUMINAMATH_CALUDE_average_age_combined_l2122_212297

theorem average_age_combined (num_students : ℕ) (num_teachers : ℕ) 
  (avg_age_students : ℚ) (avg_age_teachers : ℚ) :
  num_students = 40 →
  num_teachers = 60 →
  avg_age_students = 13 →
  avg_age_teachers = 42 →
  ((num_students : ℚ) * avg_age_students + (num_teachers : ℚ) * avg_age_teachers) / 
   ((num_students : ℚ) + (num_teachers : ℚ)) = 30.4 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2122_212297


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2122_212208

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ,
  X^4 = (X^2 + 3*X + 2) * q + (-18*X - 16) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2122_212208


namespace NUMINAMATH_CALUDE_coal_burning_duration_l2122_212249

theorem coal_burning_duration (total : ℝ) (burned_fraction : ℝ) (burned_days : ℝ) 
  (h1 : total > 0)
  (h2 : burned_fraction = 2 / 9)
  (h3 : burned_days = 6)
  (h4 : burned_fraction < 1) :
  (total - burned_fraction * total) / (burned_fraction * total / burned_days) = 21 := by
  sorry

end NUMINAMATH_CALUDE_coal_burning_duration_l2122_212249


namespace NUMINAMATH_CALUDE_town_population_distribution_l2122_212273

/-- Represents a category in the pie chart --/
structure Category where
  name : String
  percentage : ℝ

/-- Represents a pie chart with three categories --/
structure PieChart where
  categories : Fin 3 → Category
  sum_to_100 : (categories 0).percentage + (categories 1).percentage + (categories 2).percentage = 100

/-- The main theorem --/
theorem town_population_distribution (chart : PieChart) 
  (h1 : (chart.categories 0).name = "less than 5,000 residents")
  (h2 : (chart.categories 1).name = "5,000 to 20,000 residents")
  (h3 : (chart.categories 2).name = "20,000 or more residents")
  (h4 : (chart.categories 1).percentage = 40) :
  (chart.categories 1).percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_town_population_distribution_l2122_212273


namespace NUMINAMATH_CALUDE_negation_of_implication_l2122_212225

theorem negation_of_implication (x y : ℝ) : 
  ¬(x + y = 1 → x * y ≤ 1) ↔ (x + y = 1 ∧ x * y > 1) :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2122_212225


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2122_212216

theorem system_solution_ratio (x y z : ℝ) (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  let k : ℝ := 95 / 12
  (x + k * y + 4 * z = 0) →
  (4 * x + k * y - 3 * z = 0) →
  (3 * x + 5 * y - 4 * z = 0) →
  x^2 * z / y^3 = -60 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2122_212216
