import Mathlib

namespace NUMINAMATH_CALUDE_gaochun_population_scientific_notation_l831_83163

theorem gaochun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    (1 ≤ |a| ∧ |a| < 10) ∧ 
    425000 = a * (10 : ℝ) ^ n ∧
    a = 4.25 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_gaochun_population_scientific_notation_l831_83163


namespace NUMINAMATH_CALUDE_sandwich_combinations_l831_83142

theorem sandwich_combinations (meat_types cheese_types : ℕ) 
  (h1 : meat_types = 12) 
  (h2 : cheese_types = 8) : 
  meat_types * (cheese_types.choose 3) = 672 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l831_83142


namespace NUMINAMATH_CALUDE_hiking_team_participants_l831_83136

theorem hiking_team_participants (min_gloves : ℕ) (gloves_per_participant : ℕ) : 
  min_gloves = 86 → gloves_per_participant = 2 → min_gloves / gloves_per_participant = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_hiking_team_participants_l831_83136


namespace NUMINAMATH_CALUDE_subtraction_amount_l831_83100

theorem subtraction_amount (N : ℕ) (A : ℕ) : N = 32 → (N - A) / 13 = 2 → A = 6 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_amount_l831_83100


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l831_83108

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = -11) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l831_83108


namespace NUMINAMATH_CALUDE_cube_volume_from_body_diagonal_l831_83140

theorem cube_volume_from_body_diagonal (diagonal : ℝ) (h : diagonal = 15) :
  ∃ (side : ℝ), side * Real.sqrt 3 = diagonal ∧ side^3 = 375 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_body_diagonal_l831_83140


namespace NUMINAMATH_CALUDE_total_choices_is_81_l831_83145

/-- The number of bases available for students to choose from. -/
def num_bases : ℕ := 3

/-- The number of students choosing bases. -/
def num_students : ℕ := 4

/-- The total number of ways students can choose bases. -/
def total_choices : ℕ := num_bases ^ num_students

/-- Theorem stating that the total number of choices is 81. -/
theorem total_choices_is_81 : total_choices = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_choices_is_81_l831_83145


namespace NUMINAMATH_CALUDE_circle_range_m_value_l831_83101

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the condition for a point (x, y) to be on the circle
def on_circle (x y m : ℝ) : Prop :=
  circle_equation x y m

-- Define the condition for a point (x, y) to be on the line
def on_line (x y : ℝ) : Prop :=
  line_equation x y

-- Define the condition for the origin to be on the circle with diameter MN
def origin_on_diameter (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Range of m for which the equation represents a circle
theorem circle_range (m : ℝ) :
  (∃ x y, circle_equation x y m) → m < 5 :=
sorry

-- Theorem 2: Value of m when the circle intersects the line and origin is on the diameter
theorem m_value (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, 
    on_circle x₁ y₁ m ∧ 
    on_circle x₂ y₂ m ∧ 
    on_line x₁ y₁ ∧ 
    on_line x₂ y₂ ∧ 
    origin_on_diameter x₁ y₁ x₂ y₂) 
  → m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_circle_range_m_value_l831_83101


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l831_83170

theorem arithmetic_calculation : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l831_83170


namespace NUMINAMATH_CALUDE_cost_price_is_seven_l831_83158

/-- The cost price of an article satisfying the given condition -/
def cost_price : ℕ := sorry

/-- The selling price that results in a profit -/
def profit_price : ℕ := 54

/-- The selling price that results in a loss -/
def loss_price : ℕ := 40

/-- The profit is equal to the loss -/
axiom profit_equals_loss : profit_price - cost_price = cost_price - loss_price

theorem cost_price_is_seven : cost_price = 7 := by sorry

end NUMINAMATH_CALUDE_cost_price_is_seven_l831_83158


namespace NUMINAMATH_CALUDE_gcd_lcm_360_possibilities_l831_83134

theorem gcd_lcm_360_possibilities (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), S.card = 23 ∧ (∀ x, x ∈ S ↔ ∃ (a b : ℕ+), Nat.gcd a b = x ∧ Nat.gcd a b * Nat.lcm a b = 360)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_360_possibilities_l831_83134


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l831_83122

theorem quadratic_equation_solution (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l831_83122


namespace NUMINAMATH_CALUDE_expression_evaluation_l831_83198

theorem expression_evaluation : (3^3 + 2)^2 - (3^3 - 2)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l831_83198


namespace NUMINAMATH_CALUDE_doug_had_22_marbles_l831_83168

/-- Calculates the initial number of marbles Doug had -/
def dougs_initial_marbles (eds_marbles : ℕ) (difference : ℕ) : ℕ :=
  eds_marbles - difference

theorem doug_had_22_marbles (eds_marbles : ℕ) (difference : ℕ) 
  (h1 : eds_marbles = 27) 
  (h2 : difference = 5) : 
  dougs_initial_marbles eds_marbles difference = 22 := by
sorry

end NUMINAMATH_CALUDE_doug_had_22_marbles_l831_83168


namespace NUMINAMATH_CALUDE_dividend_divisible_by_divisor_l831_83123

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^55 + x^44 + x^33 + x^22 + x^11 + 1

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- Theorem stating that the dividend is divisible by the divisor -/
theorem dividend_divisible_by_divisor :
  ∃ q : ℂ → ℂ, ∀ x, dividend x = (divisor x) * (q x) := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisible_by_divisor_l831_83123


namespace NUMINAMATH_CALUDE_triangle_property_l831_83130

open Real

theorem triangle_property (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) 
  (hABC : A + B + C = π) (hSin : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * π / 3 ∧ 
  (∃ (a b c : ℝ), a = 3 ∧ 
    sin A / a = sin B / b ∧ 
    sin A / a = sin C / c ∧ 
    a + b + c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

#check triangle_property

end NUMINAMATH_CALUDE_triangle_property_l831_83130


namespace NUMINAMATH_CALUDE_pencils_per_row_l831_83138

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 12 → num_rows = 3 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l831_83138


namespace NUMINAMATH_CALUDE_rectangle_ratio_problem_l831_83195

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents a regular pentagon with side length a -/
structure Pentagon where
  a : ℝ

/-- Theorem statement for the rectangle ratio problem -/
theorem rectangle_ratio_problem (p : Pentagon) (r : Rectangle) : 
  -- The pentagon is regular and has side length a
  p.a > 0 →
  -- Five congruent rectangles are placed around the pentagon
  -- The shorter side of each rectangle lies against a side of the inner pentagon
  r.y = p.a →
  -- The area of the outer pentagon is 5 times that of the inner pentagon
  -- (We use this as an assumption without deriving it geometrically)
  r.x + r.y = Real.sqrt 5 * p.a →
  -- The ratio of the longer side to the shorter side of each rectangle is √5 - 1
  r.x / r.y = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_problem_l831_83195


namespace NUMINAMATH_CALUDE_max_luggage_length_l831_83179

theorem max_luggage_length : 
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 ∧
  length / width = 3 / 2 ∧
  length + width + 30 ≤ 160 →
  length ≤ 78 :=
by
  sorry

end NUMINAMATH_CALUDE_max_luggage_length_l831_83179


namespace NUMINAMATH_CALUDE_tom_score_l831_83110

/-- Calculates the score for regular enemies --/
def regularScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 10
  if kills ≥ 200 then baseScore * 2
  else if kills ≥ 150 then baseScore + (baseScore * 3 / 4)
  else if kills ≥ 100 then baseScore + (baseScore / 2)
  else baseScore

/-- Calculates the score for elite enemies --/
def eliteScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 25
  if kills ≥ 35 then baseScore + (baseScore * 7 / 10)
  else if kills ≥ 25 then baseScore + (baseScore / 2)
  else if kills ≥ 15 then baseScore + (baseScore * 3 / 10)
  else baseScore

/-- Calculates the score for boss enemies --/
def bossScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 50
  if kills ≥ 10 then baseScore + (baseScore * 2 / 5)
  else if kills ≥ 5 then baseScore + (baseScore / 5)
  else baseScore

/-- Calculates the total score --/
def totalScore (regularKills eliteKills bossKills : ℕ) : ℕ :=
  regularScore regularKills + eliteScore eliteKills + bossScore bossKills

theorem tom_score : totalScore 160 20 8 = 3930 := by
  sorry

end NUMINAMATH_CALUDE_tom_score_l831_83110


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l831_83183

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, x ≥ 4 ∧ Real.sqrt (x + 2 - 2 * Real.sqrt (x - 4)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 4)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l831_83183


namespace NUMINAMATH_CALUDE_max_value_of_sequence_l831_83167

def a (n : ℕ) : ℚ := n / (n^2 + 90)

theorem max_value_of_sequence :
  ∃ (M : ℚ), M = 1/19 ∧ ∀ (n : ℕ), a n ≤ M ∧ ∃ (k : ℕ), a k = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sequence_l831_83167


namespace NUMINAMATH_CALUDE_jacket_final_price_l831_83129

/-- The final price of a jacket after two successive discounts -/
theorem jacket_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 20 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  initial_price * (1 - discount1) * (1 - discount2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jacket_final_price_l831_83129


namespace NUMINAMATH_CALUDE_parallelogram_acute_angle_cosine_l831_83197

/-- Given a parallelogram with sides a and b where a ≠ b, if perpendicular lines drawn from
    vertices of obtuse angles form a similar parallelogram, then the cosine of the acute angle α
    is (2ab) / (a² + b²) -/
theorem parallelogram_acute_angle_cosine (a b : ℝ) (h : a ≠ b) :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧
  (∃ (similar : Bool), similar = true →
    Real.cos α = (2 * a * b) / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_acute_angle_cosine_l831_83197


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l831_83149

theorem last_three_digits_of_7_to_103 : 7^103 % 1000 = 327 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l831_83149


namespace NUMINAMATH_CALUDE_three_digit_number_property_l831_83162

theorem three_digit_number_property : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 11 : ℚ) = (n / 100 : ℕ)^2 + ((n / 10) % 10 : ℕ)^2 + (n % 10 : ℕ)^2 ∧
  n = 550 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l831_83162


namespace NUMINAMATH_CALUDE_larger_number_of_product_35_sum_12_l831_83199

theorem larger_number_of_product_35_sum_12 :
  ∀ x y : ℕ,
  x * y = 35 →
  x + y = 12 →
  max x y = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_product_35_sum_12_l831_83199


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l831_83176

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > a (n + 1)) ∧
  (∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n, a (n + 1) = r * a n) ∧
  (a 7 * a 14 = 6) ∧
  (a 4 + a 17 = 5)

/-- The main theorem stating the ratio of a_5 to a_18 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 5 / a 18 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l831_83176


namespace NUMINAMATH_CALUDE_line_moved_up_two_units_l831_83133

/-- Represents a line in the 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Moves a line vertically by a given amount --/
def moveLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- The theorem stating that moving y = 4x - 1 up by 2 units results in y = 4x + 1 --/
theorem line_moved_up_two_units :
  let original_line : Line := { slope := 4, intercept := -1 }
  let moved_line := moveLine original_line 2
  moved_line = { slope := 4, intercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_line_moved_up_two_units_l831_83133


namespace NUMINAMATH_CALUDE_circle_properties_l831_83144

-- Define the circle's circumference
def circumference : ℝ := 36

-- Theorem statement
theorem circle_properties :
  let radius := circumference / (2 * Real.pi)
  let diameter := 2 * radius
  let area := Real.pi * radius^2
  (radius = 18 / Real.pi) ∧
  (diameter = 36 / Real.pi) ∧
  (area = 324 / Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l831_83144


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l831_83186

theorem min_sum_of_squares (x y : ℝ) (h : (x + 4) * (y - 4) = 0) :
  ∃ (min : ℝ), min = 32 ∧ ∀ (a b : ℝ), (a + 4) * (b - 4) = 0 → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l831_83186


namespace NUMINAMATH_CALUDE_paint_house_time_l831_83132

/-- Given that five people can paint a house in seven hours and everyone works at the same rate,
    proves that two people would take 17.5 hours to paint the same house. -/
theorem paint_house_time (people_rate : ℝ → ℝ → ℝ) :
  (people_rate 5 7 = 1) →  -- Five people can paint the house in seven hours
  (∀ n t, people_rate n t = people_rate 1 1 * n * t) →  -- Everyone works at the same rate
  (people_rate 2 17.5 = 1) :=  -- Two people take 17.5 hours
by sorry

end NUMINAMATH_CALUDE_paint_house_time_l831_83132


namespace NUMINAMATH_CALUDE_greatest_common_measure_l831_83184

theorem greatest_common_measure (a b c : ℕ) (ha : a = 700) (hb : b = 385) (hc : c = 1295) :
  Nat.gcd a (Nat.gcd b c) = 35 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_l831_83184


namespace NUMINAMATH_CALUDE_min_stones_to_remove_is_ten_l831_83119

/-- Represents a chessboard configuration -/
def Chessboard := Fin 7 → Fin 8 → Bool

/-- Checks if there are five adjacent stones in any direction -/
def hasFiveAdjacent (board : Chessboard) : Bool :=
  sorry

/-- Counts the number of stones on the board -/
def stoneCount (board : Chessboard) : Nat :=
  sorry

/-- The minimal number of stones that must be removed -/
def minStonesToRemove : Nat := 10

/-- Theorem stating the minimal number of stones to remove -/
theorem min_stones_to_remove_is_ten :
  ∀ (initial : Chessboard),
    stoneCount initial = 56 →
    ∀ (final : Chessboard),
      (¬ hasFiveAdjacent final) →
      (stoneCount initial - stoneCount final ≥ minStonesToRemove) ∧
      (∃ (optimal : Chessboard),
        (¬ hasFiveAdjacent optimal) ∧
        (stoneCount initial - stoneCount optimal = minStonesToRemove)) :=
  sorry

end NUMINAMATH_CALUDE_min_stones_to_remove_is_ten_l831_83119


namespace NUMINAMATH_CALUDE_problem_statement_l831_83192

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l831_83192


namespace NUMINAMATH_CALUDE_odd_square_not_representable_l831_83105

def divisor_count (k : ℕ+) : ℕ := (Nat.divisors k.val).card

theorem odd_square_not_representable (M : ℕ+) (h_odd : Odd M.val) (h_square : ∃ k : ℕ+, M = k * k) :
  ¬∃ n : ℕ+, (M : ℚ) = (2 * Real.sqrt n.val / divisor_count n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_not_representable_l831_83105


namespace NUMINAMATH_CALUDE_function_difference_inequality_l831_83175

theorem function_difference_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h1 : ∀ x > 1, deriv f x > deriv g x)
  (h2 : ∀ x < 1, deriv f x < deriv g x) :
  f 2 - f 1 > g 2 - g 1 :=
by sorry

end NUMINAMATH_CALUDE_function_difference_inequality_l831_83175


namespace NUMINAMATH_CALUDE_point_on_y_axis_l831_83189

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on the y-axis
def onYAxis (p : Point2D) : Prop := p.x = 0

-- Theorem statement
theorem point_on_y_axis (p : Point2D) : onYAxis p ↔ p.x = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l831_83189


namespace NUMINAMATH_CALUDE_festival_attendance_l831_83153

theorem festival_attendance (total_students : ℕ) (total_attendees : ℕ) 
  (h_total : total_students = 1500)
  (h_attendees : total_attendees = 975)
  (girls : ℕ) (boys : ℕ)
  (h_students : girls + boys = total_students)
  (h_attendance : (3 * girls / 4 : ℚ) + (2 * boys / 5 : ℚ) = total_attendees) :
  (3 * girls / 4 : ℕ) = 803 :=
sorry

end NUMINAMATH_CALUDE_festival_attendance_l831_83153


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l831_83112

/-- The number of ways to divide n distinct objects into k groups of size m each. -/
def divide_into_groups (n k m : ℕ) : ℕ :=
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m) / (Nat.factorial k)

/-- The number of ways to distribute n distinct objects among k people, with each person receiving m objects. -/
def distribute_among_people (n k m : ℕ) : ℕ :=
  divide_into_groups n k m * (Nat.factorial k)

theorem book_distribution_theorem :
  let n : ℕ := 6  -- number of books
  let k : ℕ := 3  -- number of groups/people
  let m : ℕ := 2  -- number of books per group/person
  divide_into_groups n k m = 15 ∧
  distribute_among_people n k m = 90 := by
  sorry


end NUMINAMATH_CALUDE_book_distribution_theorem_l831_83112


namespace NUMINAMATH_CALUDE_real_equal_roots_condition_l831_83174

theorem real_equal_roots_condition (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 12 = 0 → y = x) ↔ 
  (k = -10 ∨ k = 14) := by sorry

end NUMINAMATH_CALUDE_real_equal_roots_condition_l831_83174


namespace NUMINAMATH_CALUDE_yogurt_combinations_l831_83159

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) : 
  flavors = 5 → toppings = 7 → flavors * (toppings.choose 3) = 175 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l831_83159


namespace NUMINAMATH_CALUDE_jellybean_probability_l831_83185

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 6
def blue_jellybeans : ℕ := 3
def green_jellybeans : ℕ := 6
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  let total_combinations := Nat.choose total_jellybeans picked_jellybeans
  let successful_combinations := Nat.choose red_jellybeans 2 * Nat.choose green_jellybeans 2
  (successful_combinations : ℚ) / total_combinations = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l831_83185


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l831_83151

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 4 ∧ b = 5) ∨ (a = 4 ∧ c = 5) ∨ (b = 4 ∧ c = 5)) →
  c = Real.sqrt 41 ∨ (a = 3 ∨ b = 3) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l831_83151


namespace NUMINAMATH_CALUDE_chicken_count_l831_83156

/-- The number of chickens in different locations and their relationships --/
theorem chicken_count :
  ∀ (coop run free_range barn : ℕ),
  coop = 14 →
  run = 2 * coop →
  5 * (coop + run) = 2 * free_range →
  2 * barn = coop →
  free_range = 105 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l831_83156


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_divisible_by_four_l831_83137

theorem consecutive_odd_sum_divisible_by_four (n : ℤ) : 
  4 ∣ ((2*n + 1) + (2*n + 3)) := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_divisible_by_four_l831_83137


namespace NUMINAMATH_CALUDE_optimal_viewpoint_for_scenery_l831_83165

/-- The problem setup -/
structure ScenerySetup where
  A : ℝ × ℝ
  B : ℝ × ℝ
  distance_AB : ℝ

/-- The viewing angle between two points from a given viewpoint -/
def viewing_angle (viewpoint : ℝ × ℝ) (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The optimal viewpoint maximizes the viewing angle -/
def is_optimal_viewpoint (setup : ScenerySetup) (viewpoint : ℝ × ℝ) : Prop :=
  ∀ other : ℝ × ℝ, viewing_angle viewpoint setup.A setup.B ≥ viewing_angle other setup.A setup.B

/-- The main theorem -/
theorem optimal_viewpoint_for_scenery (setup : ScenerySetup) 
    (h1 : setup.A = (Real.sqrt 2, Real.sqrt 2))
    (h2 : setup.B = (0, 2 * Real.sqrt 2))
    (h3 : setup.distance_AB = 2)
    (h4 : setup.A.2 > 0 ∧ setup.B.2 > 0) : -- Ensuring A and B are on the same side of x-axis
  is_optimal_viewpoint setup (0, 0) := by sorry

end NUMINAMATH_CALUDE_optimal_viewpoint_for_scenery_l831_83165


namespace NUMINAMATH_CALUDE_remove_seven_improves_mean_l831_83128

def scores : List ℕ := [6, 7, 7, 8, 8, 8, 9, 10]

def mode (l : List ℕ) : ℕ := sorry

def range (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def remove_score (s : List ℕ) (n : ℕ) : List ℕ := sorry

theorem remove_seven_improves_mean :
  let original_scores := scores
  let new_scores := remove_score original_scores 7
  mode new_scores = mode original_scores ∧
  range new_scores = range original_scores ∧
  mean new_scores > mean original_scores :=
sorry

end NUMINAMATH_CALUDE_remove_seven_improves_mean_l831_83128


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l831_83147

theorem triangle_abc_properties (a b c A B C : Real) :
  -- Given conditions
  (2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C) →
  (b = 2 * Real.sqrt 3) →
  (A = π / 4) →
  -- Conclusions
  (B = 2 * π / 3) ∧
  (1/2 * b * c * Real.sin A = (3 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l831_83147


namespace NUMINAMATH_CALUDE_sean_bought_two_soups_l831_83139

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 1

/-- The number of sodas Sean bought -/
def num_sodas : ℕ := 3

/-- The cost of a single soup in dollars -/
def soup_cost : ℚ := soda_cost * num_sodas

/-- The cost of the sandwich in dollars -/
def sandwich_cost : ℚ := 3 * soup_cost

/-- The total cost of all items in dollars -/
def total_cost : ℚ := 18

/-- The number of soups Sean bought -/
def num_soups : ℕ := 2

theorem sean_bought_two_soups :
  soda_cost * num_sodas + sandwich_cost + soup_cost * num_soups = total_cost :=
sorry

end NUMINAMATH_CALUDE_sean_bought_two_soups_l831_83139


namespace NUMINAMATH_CALUDE_arrangement_count_l831_83124

def arrange_people (n : ℕ) (k : ℕ) (m : ℕ) : Prop :=
  (n = 6) ∧ (k = 2) ∧ (m = 4)

theorem arrangement_count (n k m : ℕ) (h : arrange_people n k m) : 
  (Nat.choose n 2 * 2) + (Nat.choose n 3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l831_83124


namespace NUMINAMATH_CALUDE_diagonal_passes_through_800_cubes_l831_83117

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 240 × 360 × 400 rectangular solid passes through 800 unit cubes -/
theorem diagonal_passes_through_800_cubes :
  cubes_passed_by_diagonal 240 360 400 = 800 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_800_cubes_l831_83117


namespace NUMINAMATH_CALUDE_imaginary_part_reciprocal_l831_83121

theorem imaginary_part_reciprocal (a : ℝ) (h1 : a > 0) :
  let z : ℂ := a + Complex.I
  (Complex.abs z = Real.sqrt 5) →
  Complex.im (z⁻¹) = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_reciprocal_l831_83121


namespace NUMINAMATH_CALUDE_candy_ratio_in_bowl_l831_83109

-- Define the properties of each bag
def bag1_total : ℕ := 27
def bag1_red_ratio : ℚ := 1/3

def bag2_total : ℕ := 36
def bag2_red_ratio : ℚ := 1/4

def bag3_total : ℕ := 45
def bag3_red_ratio : ℚ := 1/5

-- Define the theorem
theorem candy_ratio_in_bowl :
  let total_candies := bag1_total + bag2_total + bag3_total
  let total_red := bag1_total * bag1_red_ratio + bag2_total * bag2_red_ratio + bag3_total * bag3_red_ratio
  total_red / total_candies = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_in_bowl_l831_83109


namespace NUMINAMATH_CALUDE_julio_orange_bottles_l831_83141

-- Define the number of bottles for each person and soda type
def julio_grape_bottles : ℕ := 7
def mateo_orange_bottles : ℕ := 1
def mateo_grape_bottles : ℕ := 3

-- Define the volume of soda per bottle
def liters_per_bottle : ℕ := 2

-- Define the additional amount of soda Julio has compared to Mateo
def julio_extra_liters : ℕ := 14

-- Define a function to calculate the total liters of soda
def total_liters (orange_bottles grape_bottles : ℕ) : ℕ :=
  (orange_bottles + grape_bottles) * liters_per_bottle

-- State the theorem
theorem julio_orange_bottles : 
  ∃ (julio_orange_bottles : ℕ),
    total_liters julio_orange_bottles julio_grape_bottles = 
    total_liters mateo_orange_bottles mateo_grape_bottles + julio_extra_liters ∧
    julio_orange_bottles = 4 := by
  sorry

end NUMINAMATH_CALUDE_julio_orange_bottles_l831_83141


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_inscribed_sphere_radius_regular_tetrahedron_is_correct_l831_83125

/-- Given a regular tetrahedron with face area S and volume V, 
    the radius of its inscribed sphere is 3V/(4S) -/
theorem inscribed_sphere_radius_regular_tetrahedron 
  (S V : ℝ) (S_pos : S > 0) (V_pos : V > 0) : ℝ :=
  3 * V / (4 * S)

/-- The calculated radius is indeed the radius of the inscribed sphere -/
theorem inscribed_sphere_radius_regular_tetrahedron_is_correct 
  (S V : ℝ) (S_pos : S > 0) (V_pos : V > 0) :
  inscribed_sphere_radius_regular_tetrahedron S V S_pos V_pos = 
    3 * V / (4 * S) := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_inscribed_sphere_radius_regular_tetrahedron_is_correct_l831_83125


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l831_83178

/-- The polynomial expression -/
def p (x : ℝ) : ℝ := 5*(x^5 - 2*x^3 + x) - 8*(x^5 + x^3 + 3*x) + 6*(3*x^5 - x^2 + 4)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (f : ℝ → ℝ) : ℝ :=
  sorry

theorem leading_coefficient_of_p :
  leading_coefficient p = 15 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l831_83178


namespace NUMINAMATH_CALUDE_base_of_term_l831_83126

theorem base_of_term (x : ℝ) (k : ℝ) : 
  (1/2)^23 * (1/x)^k = 1/18^23 ∧ k = 11.5 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_of_term_l831_83126


namespace NUMINAMATH_CALUDE_adults_eaten_correct_l831_83196

/-- Represents the number of adults who had their meal -/
def adults_eaten : ℕ := 42

/-- Represents the total number of adults in the group -/
def total_adults : ℕ := 55

/-- Represents the total number of children in the group -/
def total_children : ℕ := 70

/-- Represents the meal capacity for adults -/
def meal_capacity_adults : ℕ := 70

/-- Represents the meal capacity for children -/
def meal_capacity_children : ℕ := 90

/-- Represents the number of children that can be catered with the remaining food -/
def remaining_children : ℕ := 36

theorem adults_eaten_correct : 
  adults_eaten = 42 ∧
  total_adults = 55 ∧
  total_children = 70 ∧
  meal_capacity_adults = 70 ∧
  meal_capacity_children = 90 ∧
  remaining_children = 36 ∧
  meal_capacity_children - (adults_eaten * meal_capacity_children / meal_capacity_adults) = remaining_children :=
by sorry

end NUMINAMATH_CALUDE_adults_eaten_correct_l831_83196


namespace NUMINAMATH_CALUDE_oarsmen_count_l831_83104

theorem oarsmen_count (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 1.8 →
  old_weight = 53 →
  new_weight = 71 →
  (new_weight - old_weight) / average_increase = 10 := by
sorry

end NUMINAMATH_CALUDE_oarsmen_count_l831_83104


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solution_l831_83127

theorem quadratic_equation_one_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + x + 1 = 0) ↔ a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solution_l831_83127


namespace NUMINAMATH_CALUDE_angle_product_theorem_l831_83150

theorem angle_product_theorem (α β : Real) (m : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ y = Real.sqrt 3 * x ∧ x < 0 ∧ y < 0) →  -- condition 1
  ((1/2)^2 + m^2 = 1) →  -- condition 2
  (Real.sin α * Real.cos β < 0) →  -- condition 3
  (Real.cos α * Real.sin β = Real.sqrt 3 / 4 ∨ Real.cos α * Real.sin β = -Real.sqrt 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_angle_product_theorem_l831_83150


namespace NUMINAMATH_CALUDE_wage_payment_theorem_l831_83115

/-- Represents the daily wage of a worker -/
structure DailyWage where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a sum of money -/
def SumOfMoney : Type := ℝ

/-- Given two workers a and b, and a sum of money S, 
    prove that if S can pay a's wages for 21 days and 
    both a and b's wages for 12 days, then S can pay 
    b's wages for 28 days -/
theorem wage_payment_theorem 
  (a b : DailyWage) 
  (S : SumOfMoney) 
  (h1 : S = 21 * a.amount)
  (h2 : S = 12 * (a.amount + b.amount)) :
  S = 28 * b.amount := by
  sorry


end NUMINAMATH_CALUDE_wage_payment_theorem_l831_83115


namespace NUMINAMATH_CALUDE_max_blocks_is_twelve_l831_83169

/-- A block covers exactly two cells -/
structure Block where
  cells : Fin 16 → Fin 16
  covers_two : ∃ (c1 c2 : Fin 16), c1 ≠ c2 ∧ (∀ c, cells c = c1 ∨ cells c = c2)

/-- Configuration of blocks on a 4x4 grid -/
structure Configuration where
  blocks : List Block
  all_cells_covered : ∀ c : Fin 16, ∃ b ∈ blocks, ∃ c', b.cells c' = c
  removal_uncovers : ∀ b ∈ blocks, ∃ c : Fin 16, (∀ b' ∈ blocks, b' ≠ b → ∀ c', b'.cells c' ≠ c)

/-- The maximum number of blocks in a valid configuration -/
def max_blocks : ℕ := 12

/-- The theorem stating that 12 is the maximum number of blocks -/
theorem max_blocks_is_twelve :
  ∀ cfg : Configuration, cfg.blocks.length ≤ max_blocks :=
sorry

end NUMINAMATH_CALUDE_max_blocks_is_twelve_l831_83169


namespace NUMINAMATH_CALUDE_geometric_progression_naturals_l831_83190

theorem geometric_progression_naturals (a₁ : ℕ) (q : ℚ) :
  (∃ (a₁₀ a₃₀ : ℕ), a₁₀ = a₁ * q^9 ∧ a₃₀ = a₁ * q^29) →
  ∃ (a₂₀ : ℕ), a₂₀ = a₁ * q^19 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_naturals_l831_83190


namespace NUMINAMATH_CALUDE_congruence_solution_solution_properties_sum_of_solution_l831_83131

theorem congruence_solution : ∃ (y : ℤ), (10 * y + 3) % 18 = 7 % 18 ∧ y % 9 = 4 % 9 := by
  sorry

theorem solution_properties : 4 < 9 ∧ 9 ≥ 2 := by
  sorry

theorem sum_of_solution : ∃ (a m : ℤ), (10 * a + 3) % 18 = 7 % 18 ∧ a % m = a ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_solution_properties_sum_of_solution_l831_83131


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l831_83194

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_ratio :
  let seq1_sum := arithmetic_sum 4 4 68
  let seq2_sum := arithmetic_sum 5 5 85
  seq1_sum / seq2_sum = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l831_83194


namespace NUMINAMATH_CALUDE_poll_size_l831_83114

theorem poll_size (total : ℕ) (women_in_favor_percent : ℚ) (women_opposed : ℕ) : 
  (2 * women_opposed : ℚ) / (1 - women_in_favor_percent) = total →
  women_in_favor_percent = 35 / 100 →
  women_opposed = 39 →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_poll_size_l831_83114


namespace NUMINAMATH_CALUDE_number_divided_by_002_l831_83135

theorem number_divided_by_002 :
  ∃ x : ℝ, x / 0.02 = 201.79999999999998 ∧ x = 4.0359999999999996 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_002_l831_83135


namespace NUMINAMATH_CALUDE_fixed_points_existence_l831_83146

-- Define the fixed point F and line l
def F : ℝ × ℝ := (1, 0)
def l : ℝ → Prop := λ x => x = 4

-- Define the trajectory E
def E : ℝ × ℝ → Prop := λ p => (p.1^2 / 4) + (p.2^2 / 3) = 1

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) / |P.1 - 4| = 1/2

-- Define point A
def A : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem fixed_points_existence :
  ∃ Q₁ Q₂ : ℝ × ℝ,
    Q₁.2 = 0 ∧ Q₂.2 = 0 ∧
    Q₁ ≠ Q₂ ∧
    (∀ B C M N : ℝ × ℝ,
      E B ∧ E C ∧
      (∃ m : ℝ, B.1 = m * B.2 + 1 ∧ C.1 = m * C.2 + 1) ∧
      (M.1 = 4 ∧ N.1 = 4) ∧
      (∃ t : ℝ, M.2 = t * (B.1 + 2) ∧ N.2 = t * (C.1 + 2)) →
      ((Q₁.1 - M.1) * (Q₁.1 - N.1) + (Q₁.2 - M.2) * (Q₁.2 - N.2) = 0 ∧
       (Q₂.1 - M.1) * (Q₂.1 - N.1) + (Q₂.2 - M.2) * (Q₂.2 - N.2) = 0)) ∧
    Q₁ = (1, 0) ∧ Q₂ = (7, 0) :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_existence_l831_83146


namespace NUMINAMATH_CALUDE_remainder_base12_2543_div_9_l831_83177

-- Define a function to convert base-12 to decimal
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

-- Define the base-12 number 2543
def base12_2543 : List Nat := [2, 5, 4, 3]

-- Theorem statement
theorem remainder_base12_2543_div_9 :
  (base12ToDecimal base12_2543) % 9 = 8 := by
  sorry


end NUMINAMATH_CALUDE_remainder_base12_2543_div_9_l831_83177


namespace NUMINAMATH_CALUDE_olivias_dad_spending_l831_83166

theorem olivias_dad_spending (cost_per_meal : ℕ) (number_of_meals : ℕ) (total_cost : ℕ) : 
  cost_per_meal = 7 → number_of_meals = 3 → total_cost = cost_per_meal * number_of_meals → total_cost = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_olivias_dad_spending_l831_83166


namespace NUMINAMATH_CALUDE_sean_bedroom_bulbs_l831_83116

/-- The number of light bulbs Sean needs to replace in his bedroom. -/
def bedroom_bulbs : ℕ := 2

/-- The number of light bulbs Sean needs to replace in the bathroom. -/
def bathroom_bulbs : ℕ := 1

/-- The number of light bulbs Sean needs to replace in the kitchen. -/
def kitchen_bulbs : ℕ := 1

/-- The number of light bulbs Sean needs to replace in the basement. -/
def basement_bulbs : ℕ := 4

/-- The number of light bulbs per pack. -/
def bulbs_per_pack : ℕ := 2

/-- The number of packs Sean needs. -/
def packs_needed : ℕ := 6

/-- The total number of light bulbs Sean needs. -/
def total_bulbs : ℕ := packs_needed * bulbs_per_pack

theorem sean_bedroom_bulbs :
  bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs +
  (bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs) / 2 = total_bulbs :=
by sorry

end NUMINAMATH_CALUDE_sean_bedroom_bulbs_l831_83116


namespace NUMINAMATH_CALUDE_isosceles_triangle_23_perimeter_l831_83193

-- Define an isosceles triangle with side lengths 2 and 3
structure IsoscelesTriangle23 where
  base : ℝ
  side : ℝ
  is_isosceles : (base = 2 ∧ side = 3) ∨ (base = 3 ∧ side = 2)

-- Define the perimeter of the triangle
def perimeter (t : IsoscelesTriangle23) : ℝ := t.base + 2 * t.side

-- Theorem statement
theorem isosceles_triangle_23_perimeter :
  ∀ t : IsoscelesTriangle23, perimeter t = 7 ∨ perimeter t = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_23_perimeter_l831_83193


namespace NUMINAMATH_CALUDE_determinant_special_matrix_l831_83181

theorem determinant_special_matrix (a y : ℝ) : 
  Matrix.det !![a, y, y; y, a, y; y, y, a] = a^3 - 2*a*y^2 + 2*y^3 := by
  sorry

end NUMINAMATH_CALUDE_determinant_special_matrix_l831_83181


namespace NUMINAMATH_CALUDE_flat_tax_calculation_l831_83154

/-- Calculate the flat tax on a property with given characteristics -/
def calculate_flat_tax (condo_price condo_size barn_price barn_size detached_price detached_size
                        townhouse_price townhouse_size garage_price garage_size pool_price pool_size
                        tax_rate : ℝ) : ℝ :=
  let condo_value := condo_price * condo_size
  let barn_value := barn_price * barn_size
  let detached_value := detached_price * detached_size
  let townhouse_value := townhouse_price * townhouse_size
  let garage_value := garage_price * garage_size
  let pool_value := pool_price * pool_size
  let total_value := condo_value + barn_value + detached_value + townhouse_value + garage_value + pool_value
  total_value * tax_rate

theorem flat_tax_calculation :
  calculate_flat_tax 98 2400 84 1200 102 3500 96 2750 60 480 50 600 0.0125 = 12697.50 := by
  sorry

end NUMINAMATH_CALUDE_flat_tax_calculation_l831_83154


namespace NUMINAMATH_CALUDE_circle_equation_radius_l831_83103

/-- The radius of a circle given its equation in standard form -/
def circle_radius (h : ℝ) (k : ℝ) (r : ℝ) : ℝ := r

theorem circle_equation_radius :
  circle_radius 1 0 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l831_83103


namespace NUMINAMATH_CALUDE_faye_age_l831_83161

/-- Represents the ages of the people in the problem -/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  ages.diana + 5 = ages.eduardo ∧
  ages.eduardo = ages.chad + 6 ∧
  ages.faye = ages.chad + 4 ∧
  ages.diana = 17

/-- The theorem statement -/
theorem faye_age (ages : Ages) : age_conditions ages → ages.faye = 20 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l831_83161


namespace NUMINAMATH_CALUDE_expected_threes_eight_sided_dice_l831_83113

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The probability of rolling a 3 on a single die -/
def p : ℚ := 1 / n

/-- The probability of not rolling a 3 on a single die -/
def q : ℚ := 1 - p

/-- The expected number of 3's when rolling two n-sided dice -/
def expected_threes (n : ℕ) : ℚ := 
  2 * (p * p) + 1 * (2 * p * q) + 0 * (q * q)

/-- Theorem: The expected number of 3's when rolling two 8-sided dice is 1/4 -/
theorem expected_threes_eight_sided_dice : expected_threes n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expected_threes_eight_sided_dice_l831_83113


namespace NUMINAMATH_CALUDE_unique_right_triangle_existence_l831_83157

/-- A right triangle with leg lengths a and b, and hypotenuse c. -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c

/-- The difference between the sum of legs and hypotenuse. -/
def leg_hyp_diff (t : RightTriangle) : ℝ := t.a + t.b - t.c

/-- Theorem: A unique right triangle exists given one leg a and the difference d
    between the sum of the legs and the hypotenuse, if and only if d < a. -/
theorem unique_right_triangle_existence (a d : ℝ) (ha : 0 < a) :
  (∃! t : RightTriangle, t.a = a ∧ leg_hyp_diff t = d) ↔ d < a := by
  sorry

end NUMINAMATH_CALUDE_unique_right_triangle_existence_l831_83157


namespace NUMINAMATH_CALUDE_plant_arrangements_eq_144_l831_83164

/-- The number of ways to arrange 3 distinct vegetable plants and 3 distinct flower plants in a row,
    with all flower plants next to each other -/
def plant_arrangements : ℕ :=
  (Nat.factorial 4) * (Nat.factorial 3)

/-- Theorem stating that the number of plant arrangements is 144 -/
theorem plant_arrangements_eq_144 : plant_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangements_eq_144_l831_83164


namespace NUMINAMATH_CALUDE_marcus_second_goal_value_l831_83172

def team_total_points : ℕ := 70
def marcus_3point_goals : ℕ := 5
def marcus_unknown_goals : ℕ := 10
def marcus_percentage : ℚ := 1/2

theorem marcus_second_goal_value :
  ∃ (second_goal_value : ℕ),
    (marcus_3point_goals * 3 + marcus_unknown_goals * second_goal_value : ℚ) = 
      (marcus_percentage * team_total_points) ∧
    second_goal_value = 2 := by
  sorry

end NUMINAMATH_CALUDE_marcus_second_goal_value_l831_83172


namespace NUMINAMATH_CALUDE_fraction_equality_l831_83191

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (1 / a - 1 / b = 1 / 3) → (a * b / (a - b) = -3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l831_83191


namespace NUMINAMATH_CALUDE_seed_purchase_calculation_l831_83180

/-- Given the cost of seeds and the amount spent by a farmer, 
    calculate the number of pounds of seeds purchased. -/
theorem seed_purchase_calculation 
  (seed_cost : ℝ) 
  (seed_amount : ℝ) 
  (farmer_spent : ℝ) 
  (h1 : seed_cost = 44.68)
  (h2 : seed_amount = 2)
  (h3 : farmer_spent = 134.04) :
  farmer_spent / (seed_cost / seed_amount) = 6 :=
by sorry

end NUMINAMATH_CALUDE_seed_purchase_calculation_l831_83180


namespace NUMINAMATH_CALUDE_jims_gross_pay_l831_83118

theorem jims_gross_pay (G : ℝ) : 
  G - 0.25 * G - 100 = 740 → G = 1120 := by
  sorry

end NUMINAMATH_CALUDE_jims_gross_pay_l831_83118


namespace NUMINAMATH_CALUDE_ball_probability_l831_83171

theorem ball_probability (x : ℕ) : 
  (6 : ℝ) / ((6 : ℝ) + x) = (3 : ℝ) / 10 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l831_83171


namespace NUMINAMATH_CALUDE_square_sum_from_means_l831_83111

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = 10) : 
  x^2 + y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l831_83111


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_count_l831_83148

/-- Proves that the number of gem stone necklaces sold is 3, given the conditions of the problem -/
theorem gem_stone_necklaces_count :
  let bead_necklaces : ℕ := 4
  let price_per_necklace : ℕ := 3
  let total_earnings : ℕ := 21
  let gem_stone_necklaces : ℕ := (total_earnings - bead_necklaces * price_per_necklace) / price_per_necklace
  gem_stone_necklaces = 3 := by sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_count_l831_83148


namespace NUMINAMATH_CALUDE_distance_to_x_axis_distance_M_to_x_axis_l831_83182

/-- The distance from a point to the x-axis in a Cartesian coordinate system
    is equal to the absolute value of its y-coordinate. -/
theorem distance_to_x_axis (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  abs y = dist M (x, 0) :=
by sorry

/-- The distance from the point M(-9,12) to the x-axis is 12. -/
theorem distance_M_to_x_axis :
  let M : ℝ × ℝ := (-9, 12)
  dist M (-9, 0) = 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_distance_M_to_x_axis_l831_83182


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l831_83106

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B :=
by sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l831_83106


namespace NUMINAMATH_CALUDE_fertilizing_to_mowing_ratio_l831_83143

def mowing_time : ℕ := 40
def total_time : ℕ := 120

def fertilizing_time : ℕ := total_time - mowing_time

theorem fertilizing_to_mowing_ratio :
  (fertilizing_time : ℚ) / mowing_time = 2 := by sorry

end NUMINAMATH_CALUDE_fertilizing_to_mowing_ratio_l831_83143


namespace NUMINAMATH_CALUDE_largest_divisible_n_ten_is_divisible_largest_n_is_ten_l831_83173

theorem largest_divisible_n : ∀ n : ℕ, n > 10 → ¬(n + 15 ∣ n^3 + 250) := by
  sorry

theorem ten_is_divisible : (10 + 15 ∣ 10^3 + 250) := by
  sorry

theorem largest_n_is_ten : 
  ∀ n : ℕ, n > 0 → (n + 15 ∣ n^3 + 250) → n ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_ten_is_divisible_largest_n_is_ten_l831_83173


namespace NUMINAMATH_CALUDE_unique_prime_pair_solution_l831_83188

theorem unique_prime_pair_solution : 
  ∃! (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (7 * p * q^2 + p = q^3 + 43 * p^3 + 1) ∧ 
    p = 2 ∧ q = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_solution_l831_83188


namespace NUMINAMATH_CALUDE_unique_integer_pairs_l831_83120

theorem unique_integer_pairs :
  ∀ x y : ℕ+,
  x < y →
  x + y = 667 →
  (Nat.lcm x.val y.val : ℕ) / Nat.gcd x.val y.val = 120 →
  ((x = 145 ∧ y = 522) ∨ (x = 184 ∧ y = 483)) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_pairs_l831_83120


namespace NUMINAMATH_CALUDE_central_angle_regular_octagon_l831_83160

/-- The central angle of a regular octagon is 45 degrees. -/
theorem central_angle_regular_octagon :
  let total_angle : ℝ := 360
  let num_sides : ℕ := 8
  let central_angle := total_angle / num_sides
  central_angle = 45 := by sorry

end NUMINAMATH_CALUDE_central_angle_regular_octagon_l831_83160


namespace NUMINAMATH_CALUDE_min_value_a_plus_9b_l831_83152

theorem min_value_a_plus_9b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 10 * a * b) :
  8/5 ≤ a + 9*b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 10 * a₀ * b₀ ∧ a₀ + 9*b₀ = 8/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_9b_l831_83152


namespace NUMINAMATH_CALUDE_blue_pens_count_l831_83187

theorem blue_pens_count (total : ℕ) (difference : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 82 → 
  difference = 6 → 
  total = blue + red → 
  blue = red + difference → 
  blue = 44 := by
sorry

end NUMINAMATH_CALUDE_blue_pens_count_l831_83187


namespace NUMINAMATH_CALUDE_rice_grains_difference_l831_83107

def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ :=
  (List.range n).map grains_on_square |> List.sum

theorem rice_grains_difference : 
  grains_on_square 12 - sum_first_n_squares 9 = 501693 := by
  sorry

end NUMINAMATH_CALUDE_rice_grains_difference_l831_83107


namespace NUMINAMATH_CALUDE_sum_of_divisors_36_l831_83102

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_36 : sum_of_divisors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_36_l831_83102


namespace NUMINAMATH_CALUDE_japanese_students_count_l831_83155

theorem japanese_students_count (chinese : ℕ) (korean : ℕ) (japanese : ℕ) 
  (h1 : korean = (6 * chinese) / 11)
  (h2 : japanese = chinese / 8)
  (h3 : korean = 48) : 
  japanese = 11 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_count_l831_83155
