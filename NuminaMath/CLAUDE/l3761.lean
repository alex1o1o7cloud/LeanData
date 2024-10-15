import Mathlib

namespace NUMINAMATH_CALUDE_functional_equation_solution_l3761_376186

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3761_376186


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l3761_376116

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- Define the open interval (-1, 1)
def openInterval : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = openInterval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l3761_376116


namespace NUMINAMATH_CALUDE_booklet_sheets_l3761_376140

/-- Given a booklet created from folded A4 sheets, prove the number of original sheets. -/
theorem booklet_sheets (n : ℕ) (h : 2 * n + 2 = 74) : n / 4 = 9 := by
  sorry

#check booklet_sheets

end NUMINAMATH_CALUDE_booklet_sheets_l3761_376140


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3761_376123

theorem arithmetic_expression_equality : 1 + 6 * 2 - 3 + 5 * 4 / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3761_376123


namespace NUMINAMATH_CALUDE_max_value_product_sum_l3761_376127

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l3761_376127


namespace NUMINAMATH_CALUDE_cost_calculation_l3761_376121

/-- Given the cost relationships between mangos, rice, and flour, 
    prove the total cost of a specific quantity of each. -/
theorem cost_calculation 
  (mango_rice_relation : ∀ (mango_cost rice_cost : ℝ), 10 * mango_cost = 24 * rice_cost)
  (flour_rice_relation : ∀ (flour_cost rice_cost : ℝ), 6 * flour_cost = 2 * rice_cost)
  (flour_cost : ℝ) (h_flour_cost : flour_cost = 23)
  : ∃ (mango_cost rice_cost : ℝ),
    4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 984.4 :=
by sorry

end NUMINAMATH_CALUDE_cost_calculation_l3761_376121


namespace NUMINAMATH_CALUDE_unique_zero_in_interval_l3761_376120

def f (x : ℝ) := -x^2 + 4*x - 4

theorem unique_zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc 1 3 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_in_interval_l3761_376120


namespace NUMINAMATH_CALUDE_joshua_friends_count_l3761_376193

/-- Given that Joshua gave 40 Skittles to each friend and the total number of Skittles given is 200,
    prove that the number of friends Joshua gave Skittles to is 5. -/
theorem joshua_friends_count (skittles_per_friend : ℕ) (total_skittles : ℕ) 
    (h1 : skittles_per_friend = 40) 
    (h2 : total_skittles = 200) : 
  total_skittles / skittles_per_friend = 5 := by
sorry

end NUMINAMATH_CALUDE_joshua_friends_count_l3761_376193


namespace NUMINAMATH_CALUDE_square_root_of_four_l3761_376170

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3761_376170


namespace NUMINAMATH_CALUDE_min_value_theorem_l3761_376171

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) (h2 : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3761_376171


namespace NUMINAMATH_CALUDE_light_distance_scientific_notation_l3761_376194

/-- The speed of light in kilometers per second -/
def speed_of_light : ℝ := 300000

/-- The time in seconds -/
def time : ℝ := 10

/-- The distance traveled by light in the given time -/
def distance : ℝ := speed_of_light * time

/-- The exponent in the scientific notation of the distance -/
def n : ℕ := 6

theorem light_distance_scientific_notation :
  ∃ (a : ℝ), a > 0 ∧ a < 10 ∧ distance = a * (10 : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_light_distance_scientific_notation_l3761_376194


namespace NUMINAMATH_CALUDE_break_time_calculation_l3761_376122

-- Define the speeds of A and B
def speed_A : ℝ := 60
def speed_B : ℝ := 40

-- Define the distances from midpoint for the two meeting points
def distance_first_meeting : ℝ := 300
def distance_second_meeting : ℝ := 150

-- Define the total distance between A and B
def total_distance : ℝ := 2 * distance_first_meeting

-- Define the theorem
theorem break_time_calculation :
  ∃ (t : ℝ), (t = 6.25 ∨ t = 18.75) ∧
  ((speed_A * (total_distance / (speed_A + speed_B) - t) = distance_first_meeting + distance_second_meeting) ∨
   (speed_A * (total_distance / (speed_A + speed_B) - t) = total_distance - (distance_first_meeting + distance_second_meeting))) :=
by
  sorry


end NUMINAMATH_CALUDE_break_time_calculation_l3761_376122


namespace NUMINAMATH_CALUDE_value_of_a_l3761_376101

/-- A sequence where each term is the sum of the two terms to its left -/
def Sequence : Type := ℤ → ℤ

/-- Property that each term is the sum of the two terms to its left -/
def is_sum_of_previous_two (s : Sequence) : Prop :=
  ∀ n : ℤ, s (n + 2) = s (n + 1) + s n

/-- The specific sequence we're interested in -/
def our_sequence : Sequence := sorry

/-- The properties of our specific sequence -/
axiom our_sequence_property : is_sum_of_previous_two our_sequence
axiom our_sequence_known_values :
  ∃ k : ℤ,
    our_sequence (k + 3) = 0 ∧
    our_sequence (k + 4) = 1 ∧
    our_sequence (k + 5) = 1 ∧
    our_sequence (k + 6) = 2 ∧
    our_sequence (k + 7) = 3 ∧
    our_sequence (k + 8) = 5 ∧
    our_sequence (k + 9) = 8

/-- The theorem to prove -/
theorem value_of_a :
  ∃ k : ℤ, our_sequence k = -3 ∧
    our_sequence (k + 3) = 0 ∧
    our_sequence (k + 4) = 1 :=
sorry

end NUMINAMATH_CALUDE_value_of_a_l3761_376101


namespace NUMINAMATH_CALUDE_paper_tray_height_l3761_376104

theorem paper_tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 120 →
  cut_distance = 6 →
  cut_angle = 45 →
  let tray_height := cut_distance
  tray_height = 6 := by sorry

end NUMINAMATH_CALUDE_paper_tray_height_l3761_376104


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3761_376161

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x > 0}

theorem union_of_M_and_N : M ∪ N = {x | x = 0 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3761_376161


namespace NUMINAMATH_CALUDE_line_slope_l3761_376197

theorem line_slope (x y : ℝ) : 
  (2 * x + Real.sqrt 3 * y - 1 = 0) → 
  (∃ m : ℝ, m = -(2 * Real.sqrt 3) / 3 ∧ 
   ∀ x₁ x₂ y₁ y₂ : ℝ, 
   x₁ ≠ x₂ → 
   (2 * x₁ + Real.sqrt 3 * y₁ - 1 = 0) → 
   (2 * x₂ + Real.sqrt 3 * y₂ - 1 = 0) → 
   m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l3761_376197


namespace NUMINAMATH_CALUDE_median_mode_difference_l3761_376130

/-- Represents the monthly income data of employees --/
structure IncomeData where
  income : List Nat
  frequency : List Nat
  total_employees : Nat

/-- Calculates the mode of the income data --/
def mode (data : IncomeData) : Nat :=
  sorry

/-- Calculates the median of the income data --/
def median (data : IncomeData) : Nat :=
  sorry

/-- The income data for the company --/
def company_data : IncomeData := {
  income := [45000, 18000, 10000, 5500, 5000, 3400, 3000, 2500],
  frequency := [1, 1, 1, 3, 6, 1, 11, 1],
  total_employees := 25
}

/-- Theorem stating that the median is 400 yuan greater than the mode --/
theorem median_mode_difference (data : IncomeData) : 
  median data = mode data + 400 :=
sorry

end NUMINAMATH_CALUDE_median_mode_difference_l3761_376130


namespace NUMINAMATH_CALUDE_closest_fraction_l3761_376174

def medals_won : ℚ := 25 / 160

def fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧ 
  ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
  f = 1/8 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l3761_376174


namespace NUMINAMATH_CALUDE_mushroom_remainder_l3761_376139

theorem mushroom_remainder (initial : ℕ) (consumed : ℕ) (remaining : ℕ) : 
  initial = 15 → consumed = 8 → remaining = initial - consumed → remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_remainder_l3761_376139


namespace NUMINAMATH_CALUDE_contrapositive_truth_l3761_376143

theorem contrapositive_truth : 
  (∀ x : ℝ, (x^2 < 1 → -1 < x ∧ x < 1)) ↔ 
  (∀ x : ℝ, (x ≤ -1 ∨ 1 ≤ x) → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_l3761_376143


namespace NUMINAMATH_CALUDE_system_one_solution_l3761_376158

theorem system_one_solution (x : ℝ) : 
  (2 * x > 1 - x ∧ x + 2 < 4 * x - 1) ↔ x > 1 := by
sorry

end NUMINAMATH_CALUDE_system_one_solution_l3761_376158


namespace NUMINAMATH_CALUDE_square_equality_solution_l3761_376159

theorem square_equality_solution (x : ℝ) : (2012 + x)^2 = x^2 ↔ x = -1006 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_solution_l3761_376159


namespace NUMINAMATH_CALUDE_problem_solution_l3761_376149

theorem problem_solution (x y : ℝ) : 
  x / y = 6 / 3 → y = 27 → x = 54 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3761_376149


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3761_376148

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y = 3) : 
  4*y + 1 - 2*x = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3761_376148


namespace NUMINAMATH_CALUDE_train_length_problem_l3761_376113

/-- The length of two trains passing each other on parallel tracks --/
theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 50 * (5/18) →
  slower_speed = 36 * (5/18) →
  passing_time = 36 →
  ∃ (train_length : ℝ), train_length = 70 ∧ 
    2 * train_length = (faster_speed - slower_speed) * passing_time :=
by sorry


end NUMINAMATH_CALUDE_train_length_problem_l3761_376113


namespace NUMINAMATH_CALUDE_apartment_cost_increase_is_40_percent_l3761_376129

/-- The percentage increase in cost of a new apartment compared to an old apartment. -/
def apartment_cost_increase (old_cost monthly_savings : ℚ) : ℚ := by
  -- Define John's share of the new apartment cost
  let johns_share := old_cost - monthly_savings
  -- Calculate the total new apartment cost (3 times John's share)
  let new_cost := 3 * johns_share
  -- Calculate the percentage increase
  exact ((new_cost - old_cost) / old_cost) * 100

/-- Theorem stating the percentage increase in apartment cost -/
theorem apartment_cost_increase_is_40_percent : 
  apartment_cost_increase 1200 (7680 / 12) = 40 := by
  sorry


end NUMINAMATH_CALUDE_apartment_cost_increase_is_40_percent_l3761_376129


namespace NUMINAMATH_CALUDE_expression_value_l3761_376112

/-- Given that when x = 30, the value of ax³ + bx - 7 is 9,
    prove that the value of ax³ + bx + 2 when x = -30 is -14 -/
theorem expression_value (a b : ℝ) : 
  (30^3 * a + 30 * b - 7 = 9) → 
  ((-30)^3 * a + (-30) * b + 2 = -14) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3761_376112


namespace NUMINAMATH_CALUDE_euler_product_theorem_l3761_376119

theorem euler_product_theorem (z₁ z₂ : ℂ) :
  z₁ = Complex.exp (Complex.I * (Real.pi / 3)) →
  z₂ = Complex.exp (Complex.I * (Real.pi / 6)) →
  z₁ * z₂ = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_euler_product_theorem_l3761_376119


namespace NUMINAMATH_CALUDE_property_width_l3761_376100

/-- Proves that the width of a rectangular property is 1000 feet given specific conditions -/
theorem property_width (property_length : ℝ) (garden_area : ℝ) 
  (h1 : property_length = 2250)
  (h2 : garden_area = 28125)
  (h3 : ∃ (property_width : ℝ), 
    garden_area = (property_width / 8) * (property_length / 10)) :
  ∃ (property_width : ℝ), property_width = 1000 := by
  sorry

end NUMINAMATH_CALUDE_property_width_l3761_376100


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3761_376114

theorem consecutive_integers_sum (x y : ℤ) : 
  (y = x + 1) → (x < Real.sqrt 5 + 1) → (Real.sqrt 5 + 1 < y) → x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3761_376114


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3761_376183

theorem inequality_solution_set : 
  {x : ℤ | (x + 3)^3 ≤ 8} = {x : ℤ | x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3761_376183


namespace NUMINAMATH_CALUDE_jeans_original_cost_l3761_376168

/-- The original cost of jeans before discounts -/
def original_cost : ℝ := 49

/-- The summer discount as a percentage -/
def summer_discount : ℝ := 0.5

/-- The additional Wednesday discount in dollars -/
def wednesday_discount : ℝ := 10

/-- The final price after all discounts -/
def final_price : ℝ := 14.5

/-- Theorem stating that the original cost is correct given the discounts and final price -/
theorem jeans_original_cost :
  final_price = original_cost * (1 - summer_discount) - wednesday_discount := by
  sorry


end NUMINAMATH_CALUDE_jeans_original_cost_l3761_376168


namespace NUMINAMATH_CALUDE_minAreaLineEquation_l3761_376195

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  slope : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The area of the triangle formed by a line and the coordinate axes -/
def triangleArea (l : Line) : ℝ :=
  sorry

/-- The line passing through (1, 2) that minimizes the triangle area -/
noncomputable def minAreaLine : Line :=
  sorry

theorem minAreaLineEquation :
  let l := minAreaLine
  l.x₀ = 1 ∧ l.y₀ = 2 ∧
  ∀ (m : Line), m.x₀ = 1 ∧ m.y₀ = 2 → triangleArea l ≤ triangleArea m ∧
  2 * l.x₀ + l.y₀ - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_minAreaLineEquation_l3761_376195


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3761_376124

/-- Two vectors are parallel if the ratio of their corresponding components is equal -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3761_376124


namespace NUMINAMATH_CALUDE_red_marbles_count_l3761_376175

theorem red_marbles_count (red green yellow different total : ℕ) : 
  green = 3 * red →
  yellow = green / 5 →
  total = 3 * green →
  different = 88 →
  total = red + green + yellow + different →
  red = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l3761_376175


namespace NUMINAMATH_CALUDE_expression_value_l3761_376151

theorem expression_value (x y : ℝ) (h : 2 * y - x = 5) :
  5 * (x - 2 * y)^2 + 3 * (x - 2 * y) + 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3761_376151


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l3761_376142

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The statement that there exist infinitely many positive integers which cannot be written as a^(d(a)) + b^(d(b)) -/
theorem infinitely_many_non_representable : 
  ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ (n : ℕ+), n ∈ S → 
      ∀ (a b : ℕ+), n ≠ a ^ (num_divisors a) + b ^ (num_divisors b) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l3761_376142


namespace NUMINAMATH_CALUDE_mary_cards_left_l3761_376153

/-- The number of baseball cards Mary has left after giving away promised cards -/
def cards_left (initial : ℕ) (promised_fred : ℕ) (promised_jane : ℕ) (promised_tom : ℕ) 
               (bought : ℕ) (received : ℕ) : ℕ :=
  initial + bought + received - (promised_fred + promised_jane + promised_tom)

/-- Theorem stating that Mary will have 6 cards left -/
theorem mary_cards_left : 
  cards_left 18 26 15 36 40 25 = 6 := by sorry

end NUMINAMATH_CALUDE_mary_cards_left_l3761_376153


namespace NUMINAMATH_CALUDE_supermarket_profit_and_discount_l3761_376177

-- Define the goods
structure Good where
  cost : ℝ
  price : ℝ

-- Define the problem parameters
def good_A : Good := { cost := 22, price := 29 }
def good_B : Good := { cost := 30, price := 40 }

-- Define the theorem
theorem supermarket_profit_and_discount 
  (total_cost : ℝ) 
  (num_A : ℕ) 
  (num_B : ℕ) 
  (second_profit_increase : ℝ) :
  total_cost = 6000 ∧ 
  num_B = (num_A / 2 + 15 : ℕ) ∧
  num_A * good_A.cost + num_B * good_B.cost = total_cost →
  (num_A * (good_A.price - good_A.cost) + num_B * (good_B.price - good_B.cost) = 1950) ∧
  ∃ discount_rate : ℝ,
    discount_rate ≥ 0 ∧ 
    discount_rate ≤ 1 ∧
    num_A * (good_A.price - good_A.cost) + 3 * num_B * ((1 - discount_rate) * good_B.price - good_B.cost) = 
    1950 + second_profit_increase ∧
    discount_rate = 0.085 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_supermarket_profit_and_discount_l3761_376177


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l3761_376138

theorem sine_cosine_relation (α : ℝ) (h : Real.cos (α + π / 12) = 1 / 5) :
  Real.sin (α + 7 * π / 12) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l3761_376138


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l3761_376115

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℝ, Real.exp (abs (2 * x + 1)) + m ≥ 0) ↔ m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l3761_376115


namespace NUMINAMATH_CALUDE_fraction_difference_product_l3761_376176

theorem fraction_difference_product : 
  let a : ℚ := 1/2
  let b : ℚ := 1/5
  a - b = 3 * (a * b) := by sorry

end NUMINAMATH_CALUDE_fraction_difference_product_l3761_376176


namespace NUMINAMATH_CALUDE_angle_Z_measure_l3761_376192

-- Define the triangle and its angles
def Triangle (X Y W Z : ℝ) : Prop :=
  -- Conditions
  X = 34 ∧ Y = 53 ∧ W = 43 ∧
  -- Additional properties of a triangle
  X > 0 ∧ Y > 0 ∧ W > 0 ∧ Z > 0 ∧
  -- Sum of angles in the larger triangle is 180°
  X + Y + W + Z = 180

-- Theorem statement
theorem angle_Z_measure (X Y W Z : ℝ) (h : Triangle X Y W Z) : Z = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_Z_measure_l3761_376192


namespace NUMINAMATH_CALUDE_birthday_probability_l3761_376103

/-- The probability that 3 boys born in June 1990 have different birthdays -/
def probability_different_birthdays : ℚ :=
  1 * (29 / 30) * (28 / 30)

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The number of boys -/
def number_of_boys : ℕ := 3

theorem birthday_probability :
  probability_different_birthdays = 203 / 225 :=
sorry

end NUMINAMATH_CALUDE_birthday_probability_l3761_376103


namespace NUMINAMATH_CALUDE_sin_max_value_l3761_376157

theorem sin_max_value (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/3 ∧ 
    (∀ y : ℝ, 0 ≤ y ∧ y ≤ π/3 → 2 * Real.sin (ω * y) ≤ 2 * Real.sin (ω * x)) ∧
    2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_max_value_l3761_376157


namespace NUMINAMATH_CALUDE_population_growth_problem_l3761_376173

theorem population_growth_problem (initial_population : ℝ) (final_population : ℝ) (second_year_decrease : ℝ) :
  initial_population = 10000 →
  final_population = 9600 →
  second_year_decrease = 20 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 20 ∧
    final_population = initial_population * (1 + first_year_increase / 100) * (1 - second_year_decrease / 100) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_problem_l3761_376173


namespace NUMINAMATH_CALUDE_sum_of_circle_circumferences_l3761_376167

/-- The sum of circumferences of an infinite series of circles inscribed in an equilateral triangle -/
theorem sum_of_circle_circumferences (r : ℝ) (h : r = 1) : 
  (2 * π * r) + (3 * (2 * π * r * (∑' n, (1/3)^n))) = 5 * π :=
sorry

end NUMINAMATH_CALUDE_sum_of_circle_circumferences_l3761_376167


namespace NUMINAMATH_CALUDE_inscribed_rectangle_semicircle_radius_l3761_376134

/-- Given a rectangle inscribed in a semi-circle with specific properties,
    prove that the radius of the semi-circle is 23.625 cm. -/
theorem inscribed_rectangle_semicircle_radius 
  (perimeter : ℝ) 
  (width : ℝ) 
  (length : ℝ) 
  (h1 : perimeter = 126)
  (h2 : length = 3 * width)
  (h3 : perimeter = 2 * length + 2 * width) : 
  (length / 2 : ℝ) = 23.625 := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_semicircle_radius_l3761_376134


namespace NUMINAMATH_CALUDE_algorithm_description_not_unique_l3761_376131

/-- Definition of an algorithm -/
structure Algorithm where
  steps : List String
  solves_problem : Bool

/-- There can be different ways to describe an algorithm -/
theorem algorithm_description_not_unique : ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ a1.solves_problem = a2.solves_problem := by
  sorry

end NUMINAMATH_CALUDE_algorithm_description_not_unique_l3761_376131


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3761_376181

/-- A rectangular prism with three distinct dimensions -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- The number of face diagonals in a rectangular prism -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

/-- Theorem: A rectangular prism with three distinct dimensions has 16 total diagonals -/
theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3761_376181


namespace NUMINAMATH_CALUDE_triangle_side_possible_value_l3761_376135

theorem triangle_side_possible_value (a : ℤ) : 
  (a > 0) → 
  (7 + 3 > a) → 
  (7 + a > 3) → 
  (3 + a > 7) → 
  (a = 8) → 
  ∃ (x y z : ℝ), x = 7 ∧ y = a ∧ z = 3 ∧ x + y > z ∧ x + z > y ∧ y + z > x :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_possible_value_l3761_376135


namespace NUMINAMATH_CALUDE_triangle_problem_l3761_376163

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_problem (a b c : ℝ) (h1 : Triangle a b c)
    (h2 : b * Real.cos C + c / 2 = a)
    (h3 : b = Real.sqrt 13)
    (h4 : a + c = 4) :
    Real.cos B = 1 / 2 ∧ 
    (1 / 2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3761_376163


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l3761_376189

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 7
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 1

-- State the theorem
theorem polynomial_sum_equality :
  ∀ x : ℝ, p x + q x + r x + s x = -2 * x^2 + 9 * x - 11 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l3761_376189


namespace NUMINAMATH_CALUDE_largest_when_first_digit_changed_l3761_376106

def original_number : ℚ := 0.123456

def change_digit (n : ℕ) (d : ℕ) : ℚ :=
  if n = 1 then 0.8 + (original_number - 0.1)
  else if n = 2 then 0.1 + 0.08 + (original_number - 0.12)
  else if n = 3 then 0.12 + 0.008 + (original_number - 0.123)
  else if n = 4 then 0.123 + 0.0008 + (original_number - 0.1234)
  else if n = 5 then 0.1234 + 0.00008 + (original_number - 0.12345)
  else 0.12345 + 0.000008 + (original_number - 0.123456)

theorem largest_when_first_digit_changed :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → change_digit 1 8 ≥ change_digit n 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_when_first_digit_changed_l3761_376106


namespace NUMINAMATH_CALUDE_initial_capacity_correct_l3761_376137

/-- The capacity of each bucket in the set of 20 buckets -/
def initial_bucket_capacity : ℝ := 13.5

/-- The number of buckets in the initial set -/
def initial_bucket_count : ℕ := 20

/-- The capacity of each bucket in the set of 30 buckets -/
def new_bucket_capacity : ℝ := 9

/-- The number of buckets in the new set -/
def new_bucket_count : ℕ := 30

/-- The theorem states that the initial bucket capacity is correct -/
theorem initial_capacity_correct : 
  initial_bucket_capacity * initial_bucket_count = new_bucket_capacity * new_bucket_count := by
  sorry

end NUMINAMATH_CALUDE_initial_capacity_correct_l3761_376137


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3761_376125

/-- The eccentricity of the hyperbola x²/3 - y²/6 = 1 is √3 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 3 ∧
  ∀ x y : ℝ, x^2 / 3 - y^2 / 6 = 1 → 
  e = Real.sqrt ((x^2 / 3 + y^2 / 6) / (x^2 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3761_376125


namespace NUMINAMATH_CALUDE_triangle_placement_theorem_l3761_376196

-- Define the types for points and angles
def Point : Type := ℝ × ℝ
def Angle : Type := ℝ

-- Define a triangle as a triple of points
structure Triangle :=
  (E F G : Point)

-- Define the property that a point lies on an arm of an angle
def lies_on_arm (P : Point) (A : Point) (angle : Angle) : Prop := sorry

-- Define the property that an angle between three points equals a given angle
def angle_equals (A B C : Point) (angle : Angle) : Prop := sorry

theorem triangle_placement_theorem 
  (T : Triangle) (angle_ABC angle_CBD : Angle) : 
  ∃ (B : Point), 
    (lies_on_arm T.E B angle_ABC) ∧ 
    (lies_on_arm T.F B angle_ABC) ∧ 
    (lies_on_arm T.G B angle_CBD) ∧
    (angle_equals T.E B T.F angle_ABC) ∧
    (angle_equals T.F B T.G angle_CBD) := by
  sorry

end NUMINAMATH_CALUDE_triangle_placement_theorem_l3761_376196


namespace NUMINAMATH_CALUDE_no_integer_root_trinomials_l3761_376178

theorem no_integer_root_trinomials : ¬∃ (a b c : ℤ),
  (∃ (x₁ x₂ : ℤ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) ∧
  (∃ (y₁ y₂ : ℤ), (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0 ∧ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_root_trinomials_l3761_376178


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l3761_376164

theorem sum_and_reciprocal_geq_two (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l3761_376164


namespace NUMINAMATH_CALUDE_propositions_proof_l3761_376111

theorem propositions_proof :
  (∀ a b : ℝ, a > b ∧ (1 / a) > (1 / b) → a * b < 0) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬(a^2 < a * b ∧ a * b < b^2)) ∧
  (∀ a b c : ℝ, c > a ∧ a > b ∧ b > 0 → ¬(a / (c - a) < b / (c - b))) ∧
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → a / b > (a + c) / (b + c)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_proof_l3761_376111


namespace NUMINAMATH_CALUDE_marathon_calories_burned_l3761_376172

/-- Represents a cycling ride with its distance relative to the base distance -/
structure Ride :=
  (distance : ℝ)

/-- Calculates the adjusted distance for a ride given the actual distance and base distance -/
def adjustedDistance (actualDistance : ℝ) (baseDistance : ℝ) : ℝ :=
  actualDistance - baseDistance

/-- Calculates the total calories burned given a list of rides, base distance, and calorie burn rate -/
def totalCaloriesBurned (rides : List Ride) (baseDistance : ℝ) (caloriesPerKm : ℝ) : ℝ :=
  (rides.map (λ ride => ride.distance + baseDistance)).sum * caloriesPerKm

theorem marathon_calories_burned 
  (rides : List Ride)
  (baseDistance : ℝ)
  (caloriesPerKm : ℝ)
  (h1 : rides.length = 10)
  (h2 : baseDistance = 15)
  (h3 : caloriesPerKm = 20)
  (h4 : rides[3].distance = adjustedDistance 16.5 baseDistance)
  (h5 : rides[6].distance = adjustedDistance 14.1 baseDistance)
  : totalCaloriesBurned rides baseDistance caloriesPerKm = 3040 := by
  sorry

end NUMINAMATH_CALUDE_marathon_calories_burned_l3761_376172


namespace NUMINAMATH_CALUDE_second_candidate_votes_l3761_376199

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) : 
  total_votes = 800 → 
  first_candidate_percentage = 70 / 100 →
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 240 := by
  sorry

#check second_candidate_votes

end NUMINAMATH_CALUDE_second_candidate_votes_l3761_376199


namespace NUMINAMATH_CALUDE_constant_dot_product_l3761_376144

open Real

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 27 + y^2 / 18 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (3, 0)

/-- The fixed point P -/
def P : ℝ × ℝ := (4, 0)

/-- A line passing through F -/
def line_through_F (k : ℝ) (x : ℝ) : ℝ := k * (x - F.1)

/-- Intersection points of the line with the ellipse -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ p.2 = line_through_F k p.1}

/-- Dot product of vectors PA and PB -/
def dot_product (A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

/-- Theorem: The dot product PA · PB is constant for any line through F -/
theorem constant_dot_product :
  ∃ (c : ℝ), ∀ (k : ℝ) (A B : ℝ × ℝ),
    A ∈ intersection_points k → B ∈ intersection_points k →
    A ≠ B → dot_product A B = c :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l3761_376144


namespace NUMINAMATH_CALUDE_midpoint_of_number_line_l3761_376117

theorem midpoint_of_number_line (a b : ℝ) (ha : a = -1) (hb : b = 3) :
  (a + b) / 2 = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_number_line_l3761_376117


namespace NUMINAMATH_CALUDE_max_value_of_f_l3761_376133

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≤ f c) ∧
  f c = (1 : ℝ) / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3761_376133


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3761_376150

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the inequality
def inequality (x : ℝ) : Prop := log_half (2*x + 1) ≥ log_half 3

-- Define the solution set
def solution_set : Set ℝ := Set.Ioc (-1/2) 1

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3761_376150


namespace NUMINAMATH_CALUDE_probability_one_science_one_humanities_l3761_376108

def total_courses : ℕ := 5
def science_courses : ℕ := 3
def humanities_courses : ℕ := 2
def courses_chosen : ℕ := 2

theorem probability_one_science_one_humanities :
  (Nat.choose science_courses 1 * Nat.choose humanities_courses 1) / Nat.choose total_courses courses_chosen = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_science_one_humanities_l3761_376108


namespace NUMINAMATH_CALUDE_x_minus_2y_bounds_l3761_376110

theorem x_minus_2y_bounds (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  0 ≤ x - 2*y ∧ x - 2*y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_2y_bounds_l3761_376110


namespace NUMINAMATH_CALUDE_william_marbles_left_l3761_376162

/-- Given that William initially has 10 marbles and shares 3 marbles with Theresa,
    prove that William will have 7 marbles left. -/
theorem william_marbles_left (initial_marbles : ℕ) (shared_marbles : ℕ) 
  (h1 : initial_marbles = 10) (h2 : shared_marbles = 3) :
  initial_marbles - shared_marbles = 7 := by
  sorry

end NUMINAMATH_CALUDE_william_marbles_left_l3761_376162


namespace NUMINAMATH_CALUDE_sum_of_digits_N_l3761_376147

/-- The smallest positive integer whose digits have a product of 1728 -/
def N : ℕ := sorry

/-- The product of the digits of N is 1728 -/
axiom N_digit_product : (N.digits 10).prod = 1728

/-- N is the smallest such positive integer -/
axiom N_smallest (m : ℕ) : m > 0 → (m.digits 10).prod = 1728 → m ≥ N

/-- The sum of the digits of N is 28 -/
theorem sum_of_digits_N : (N.digits 10).sum = 28 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_N_l3761_376147


namespace NUMINAMATH_CALUDE_square_tile_count_l3761_376118

theorem square_tile_count (n : ℕ) (h : n^2 = 81) : 
  n^2 * n^2 - (2*n - 1) = 6544 := by
  sorry

end NUMINAMATH_CALUDE_square_tile_count_l3761_376118


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3761_376109

theorem quadratic_equation_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h1 : p^2 + p*p + q = 0) (h2 : q^2 + p*q + q = 0) : p = 1 ∧ q = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3761_376109


namespace NUMINAMATH_CALUDE_cost_price_of_article_l3761_376187

/-- 
Proves that the cost price of an article is 44, given that the profit obtained 
by selling it for 66 is the same as the loss obtained by selling it for 22.
-/
theorem cost_price_of_article : ∃ (x : ℝ), 
  (66 - x = x - 22) → x = 44 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l3761_376187


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3761_376169

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 5x^2 - 1 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 1

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3761_376169


namespace NUMINAMATH_CALUDE_min_selections_for_multiple_of_five_l3761_376154

theorem min_selections_for_multiple_of_five (n : ℕ) (h : n = 30) : 
  (∀ S : Finset ℕ, S ⊆ Finset.range n → S.card ≥ 25 → ∃ x ∈ S, x % 5 = 0) ∧
  (∃ S : Finset ℕ, S ⊆ Finset.range n ∧ S.card = 24 ∧ ∀ x ∈ S, x % 5 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_min_selections_for_multiple_of_five_l3761_376154


namespace NUMINAMATH_CALUDE_square_sum_from_conditions_l3761_376102

theorem square_sum_from_conditions (x y : ℝ) 
  (h1 : x + 2 * y = 6) 
  (h2 : x * y = -6) : 
  x^2 + 4 * y^2 = 60 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_conditions_l3761_376102


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3761_376160

/-- Given a hyperbola and a parabola, prove that if the left focus of the hyperbola
    lies on the directrix of the parabola, then p = 4 -/
theorem hyperbola_parabola_intersection (p : ℝ) (hp : p > 0) : 
  (∃ x y : ℝ, x^2 / 3 - 16 * y^2 / p^2 = 1) →  -- hyperbola equation
  (∃ x y : ℝ, y^2 = 2 * p * x) →              -- parabola equation
  (- Real.sqrt (3 + p^2 / 16) = - p / 2) →    -- left focus on directrix condition
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3761_376160


namespace NUMINAMATH_CALUDE_irrational_condition_l3761_376128

-- Define the set A(x)
def A (x : ℝ) : Set ℤ := {n : ℤ | ∃ m : ℕ, n = ⌊m * x⌋}

-- State the theorem
theorem irrational_condition (α : ℝ) (h_irr : Irrational α) (h_gt_two : α > 2) :
  ∀ β : ℝ, β > 0 → (A α ⊃ A β) → ∃ n : ℤ, β = n * α :=
by sorry

end NUMINAMATH_CALUDE_irrational_condition_l3761_376128


namespace NUMINAMATH_CALUDE_max_correct_answers_15_l3761_376132

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  correct_answers : ℕ
  incorrect_answers : ℕ
  blank_answers : ℕ
  total_score : ℤ

/-- Calculates the total score for an exam result. -/
def calculate_score (result : ExamResult) : ℤ :=
  result.correct_answers * result.exam.correct_score +
  result.incorrect_answers * result.exam.incorrect_score

/-- Verifies if an exam result is valid. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct_answers + result.incorrect_answers + result.blank_answers = result.exam.total_questions ∧
  calculate_score result = result.total_score

/-- Theorem: The maximum number of correct answers for John's exam is 15. -/
theorem max_correct_answers_15 (john_exam : Exam) (john_result : ExamResult) :
  john_exam.total_questions = 25 ∧
  john_exam.correct_score = 6 ∧
  john_exam.incorrect_score = -3 ∧
  john_result.exam = john_exam ∧
  john_result.total_score = 60 ∧
  is_valid_result john_result →
  john_result.correct_answers ≤ 15 ∧
  ∃ (valid_result : ExamResult),
    valid_result.exam = john_exam ∧
    valid_result.total_score = 60 ∧
    is_valid_result valid_result ∧
    valid_result.correct_answers = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_15_l3761_376132


namespace NUMINAMATH_CALUDE_average_marks_proof_l3761_376152

/-- Given the marks in three subjects, prove that the average is 75 -/
theorem average_marks_proof (physics chemistry mathematics : ℝ) 
  (h1 : (physics + mathematics) / 2 = 90)
  (h2 : (physics + chemistry) / 2 = 70)
  (h3 : physics = 95) :
  (physics + chemistry + mathematics) / 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l3761_376152


namespace NUMINAMATH_CALUDE_max_rooks_per_color_exists_sixteen_rooks_config_max_rooks_is_sixteen_l3761_376184

/-- Represents a chessboard configuration with white and black rooks -/
structure ChessboardConfig where
  board_size : Nat
  white_rooks : Nat
  black_rooks : Nat
  non_threatening : Bool

/-- Defines a valid chessboard configuration -/
def is_valid_config (c : ChessboardConfig) : Prop :=
  c.board_size = 8 ∧ 
  c.white_rooks = c.black_rooks ∧ 
  c.non_threatening = true

/-- Theorem stating the maximum number of rooks for each color -/
theorem max_rooks_per_color (c : ChessboardConfig) : 
  is_valid_config c → c.white_rooks ≤ 16 := by
  sorry

/-- Theorem proving the existence of a configuration with 16 rooks per color -/
theorem exists_sixteen_rooks_config : 
  ∃ c : ChessboardConfig, is_valid_config c ∧ c.white_rooks = 16 := by
  sorry

/-- Main theorem proving 16 is the maximum number of rooks per color -/
theorem max_rooks_is_sixteen : 
  ∀ c : ChessboardConfig, is_valid_config c → 
    c.white_rooks ≤ 16 ∧ (∃ c' : ChessboardConfig, is_valid_config c' ∧ c'.white_rooks = 16) := by
  sorry

end NUMINAMATH_CALUDE_max_rooks_per_color_exists_sixteen_rooks_config_max_rooks_is_sixteen_l3761_376184


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l3761_376180

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Adds two binary numbers (represented as lists of bits) and returns the result as a list of bits. -/
def add_binary (a b : List Bool) : List Bool :=
  sorry -- Implementation details omitted

/-- Theorem: The sum of 1101₂, 100₂, 111₂, and 11010₂ is equal to 111001₂ -/
theorem binary_sum_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, false, true]       -- 100₂
  let c := [true, true, true]         -- 111₂
  let d := [false, true, false, true, true]  -- 11010₂
  let result := [true, false, false, true, true, true]  -- 111001₂
  add_binary (add_binary (add_binary a b) c) d = result := by
  sorry

#eval binary_to_nat [true, false, false, true, true, true]  -- Should output 57

end NUMINAMATH_CALUDE_binary_sum_theorem_l3761_376180


namespace NUMINAMATH_CALUDE_loan_duration_proof_l3761_376156

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem loan_duration_proof (principal rate total_returned : ℝ) 
  (h1 : principal = 5396.103896103896)
  (h2 : rate = 0.06)
  (h3 : total_returned = 8310) :
  ∃ t : ℝ, t = 9 ∧ total_returned = principal + simple_interest principal rate t := by
  sorry

#eval simple_interest 5396.103896103896 0.06 9

end NUMINAMATH_CALUDE_loan_duration_proof_l3761_376156


namespace NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l3761_376185

open Real

theorem intersection_perpendicular_tangents (a : ℝ) : 
  ∃ (x : ℝ), 0 < x ∧ x < π / 2 ∧ 
  2 * sin x = a * cos x ∧
  (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l3761_376185


namespace NUMINAMATH_CALUDE_doll_difference_l3761_376190

/-- The number of dolls Lindsay has with blonde hair -/
def blonde_dolls : ℕ := 4

/-- The number of dolls Lindsay has with brown hair -/
def brown_dolls : ℕ := 4 * blonde_dolls

/-- The number of dolls Lindsay has with black hair -/
def black_dolls : ℕ := brown_dolls - 2

/-- The theorem stating the difference between the combined number of black and brown-haired dolls
    and the number of blonde-haired dolls -/
theorem doll_difference : brown_dolls + black_dolls - blonde_dolls = 26 := by
  sorry

end NUMINAMATH_CALUDE_doll_difference_l3761_376190


namespace NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l3761_376198

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem odd_times_abs_even_is_odd (f g : ℝ → ℝ) 
  (h_f_odd : IsOdd f) (h_g_even : IsEven g) : 
  IsOdd (fun x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l3761_376198


namespace NUMINAMATH_CALUDE_f_continuous_at_5_l3761_376126

def f (x : ℝ) : ℝ := 2 * x^2 + 8

theorem f_continuous_at_5 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_5_l3761_376126


namespace NUMINAMATH_CALUDE_min_distance_to_i_l3761_376166

theorem min_distance_to_i (z : ℂ) (h : Complex.abs (z + Complex.I * Real.sqrt 3) + Complex.abs (z - Complex.I * Real.sqrt 3) = 4) :
  ∃ (w : ℂ), Complex.abs (w + Complex.I * Real.sqrt 3) + Complex.abs (w - Complex.I * Real.sqrt 3) = 4 ∧
    Complex.abs (w - Complex.I) ≤ Complex.abs (z - Complex.I) ∧
    Complex.abs (w - Complex.I) = Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_i_l3761_376166


namespace NUMINAMATH_CALUDE_remainder_2021_div_102_l3761_376155

theorem remainder_2021_div_102 : 2021 % 102 = 83 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2021_div_102_l3761_376155


namespace NUMINAMATH_CALUDE_vasyas_number_l3761_376105

theorem vasyas_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  1008 + 10 * n = 28 * n :=
by
  sorry

end NUMINAMATH_CALUDE_vasyas_number_l3761_376105


namespace NUMINAMATH_CALUDE_bread_recipe_scaling_l3761_376179

/-- Given a recipe that requires 60 mL of water and 80 mL of milk for every 400 mL of flour,
    this theorem proves the amount of water and milk needed for 1200 mL of flour. -/
theorem bread_recipe_scaling (flour : ℝ) (water : ℝ) (milk : ℝ) 
  (h1 : flour = 1200)
  (h2 : water = 60 * (flour / 400))
  (h3 : milk = 80 * (flour / 400)) :
  water = 180 ∧ milk = 240 := by
  sorry

end NUMINAMATH_CALUDE_bread_recipe_scaling_l3761_376179


namespace NUMINAMATH_CALUDE_proportional_function_decreasing_l3761_376145

/-- A proportional function passing through (2, -4) has a decreasing y as x increases -/
theorem proportional_function_decreasing (k : ℝ) (h1 : k ≠ 0) (h2 : k * 2 = -4) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_decreasing_l3761_376145


namespace NUMINAMATH_CALUDE_min_value_of_f_l3761_376191

/-- The quadratic function f(x) = x^2 + 12x + 5 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 5

/-- The minimum value of f(x) is -31 -/
theorem min_value_of_f : ∀ x : ℝ, f x ≥ -31 ∧ ∃ y : ℝ, f y = -31 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3761_376191


namespace NUMINAMATH_CALUDE_line_y_coordinate_l3761_376136

/-- A line passing through points (-12, y1) and (x2, 3) with x-intercept at (4, 0) has y1 = 0 -/
theorem line_y_coordinate (y1 x2 : ℝ) : 
  (∃ (m : ℝ), (3 - y1) = m * (x2 - (-12)) ∧ 
               0 - y1 = m * (4 - (-12)) ∧
               3 = m * (x2 - 4)) →
  y1 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l3761_376136


namespace NUMINAMATH_CALUDE_locus_is_circle_l3761_376182

/-- The locus of points satisfying the given equation is a circle -/
theorem locus_is_circle (x y : ℝ) : 
  (10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3*x - 4*y|) → 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_circle_l3761_376182


namespace NUMINAMATH_CALUDE_total_shirts_made_l3761_376146

-- Define the rate of shirt production
def shirts_per_minute : ℕ := 3

-- Define the working time
def working_time : ℕ := 2

-- Theorem to prove
theorem total_shirts_made : shirts_per_minute * working_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_made_l3761_376146


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l3761_376165

def total_plants : ℕ := 5
def selected_plants : ℕ := 3

theorem probability_of_selecting_A_and_B : 
  (Nat.choose total_plants selected_plants) > 0 → 
  (Nat.choose (total_plants - 2) (selected_plants - 2)) > 0 →
  (Nat.choose (total_plants - 2) (selected_plants - 2) : ℚ) / 
  (Nat.choose total_plants selected_plants : ℚ) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l3761_376165


namespace NUMINAMATH_CALUDE_intersection_condition_l3761_376107

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1

/-- The statement that f(x) intersects y = 3 at only one point -/
def intersects_once (a : ℝ) : Prop :=
  ∃! x : ℝ, f a x = 3

/-- The main theorem to be proved -/
theorem intersection_condition :
  ∀ a : ℝ, intersects_once a ↔ -1 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l3761_376107


namespace NUMINAMATH_CALUDE_euler_totient_multiple_l3761_376188

theorem euler_totient_multiple (m n : ℕ+) : ∃ a : ℕ+, ∀ i : ℕ, i ≤ n → (m : ℕ) ∣ Nat.totient (a + i) := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_multiple_l3761_376188


namespace NUMINAMATH_CALUDE_red_peaches_count_l3761_376141

theorem red_peaches_count (total : ℕ) (green : ℕ) (red : ℕ) : 
  total = 16 → green = 3 → total = red + green → red = 13 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l3761_376141
